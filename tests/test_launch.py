"""실행기 검사 — run 명세, 상한의 산술, 원격 러너, 백엔드, 여섯 명령 (docs/06 Task 6).

여기서 보는 것은 다섯 가지다. (1) **명세**가 스키마 파일(`infra/run-manifest.schema.json`)에 맞고, 그 스키마가
이 검사기가 아는 키워드만 쓰는가. (2) **상한**이 하나의 벽시계 마감으로 환산되고 가장 이른 것이 구속하는가,
그리고 원격 러너가 그 마감에서 **checkpoint를 쓴 뒤** `failed(reason=budget)`으로 끝나는가 — 실행기가 죽어도
지켜지는 자리다. (3) **왕복**: prepare → launch → status → fetch가 `local` 백엔드에서 끝까지 돌고 가져온 파일의
sha256이 원격 값과 같은가. (4) **취소**: 멈춤을 확인하지 못하면 `cancelled`가 아니라 `unknown`인가. (5) **재개**:
등록되고 대조된 checkpoint에서만 잇는가.

SSH 경로는 가짜 전송으로 **명령 문장**을 검사한다(sshd가 없는 상자에서도 돈다). 진짜 ssh 왕복은 인수에서
돌리고 로그로 남긴다(`.superpowers/sdd/task-r3b-report.md` Stage B).
"""

from __future__ import annotations

import json
import shlex
import subprocess
import sys
from pathlib import Path

import pytest
import yaml
from helpers import REPO

from robo_jev.launch import launcher as launcher_module
from robo_jev.launch.launcher import (
    CancelNotConfirmed,
    SpendNotObserved,
    UncappedRun,
    artifact_plan,
    bundle_dir,
    portable_path,
    cancel,
    fetch,
    launch,
    observed_spend_seconds,
    prepare,
    registered_checkpoint,
    remaining_budget,
    resume,
    status,
)
from robo_jev.launch.manifest import (
    SCHEMA_PATH,
    SCHEMA_VERSION,
    STATES,
    SUPPORTED_KEYWORDS,
    Budget,
    budget_deadline_hours,
    budget_spend,
    load_manifest,
    new_manifest,
    note_state,
    plan_budget,
    save_manifest,
    schema,
    validate,
)
from robo_jev.launch.providers.base import Backend, Exec, Handle
from robo_jev.launch.providers.local import LocalBackend
from robo_jev.launch.providers.ssh import DEFAULT_SSH_OPTIONS, SshBackend
from robo_jev.launch.runner import check_inputs, inventory, run_bundle, should_stop

TINY_CONFIG = REPO / "configs" / "train" / "tiny_cpu.yaml"
#: 검사용으로 더 짧게 — 조리법은 그대로 두고 step 수와 틱 수만 줄인다 (step ≈ 1.7 s).
FAST = {"max_steps": 2, "stream_max_ticks": 2, "checkpoint_every": 1, "torch_threads": 1}


def tiny_config(**overrides) -> dict:
    config = yaml.safe_load(TINY_CONFIG.read_text(encoding="utf-8")) or {}
    config["dataset_manifests"] = [str(REPO / entry) if isinstance(entry, str) else {**entry, "path": str(REPO / entry["path"])} for entry in config["dataset_manifests"]]
    config.update(FAST)
    config.update(overrides)
    return config


def local_backend(tmp_path: Path, run_id: str) -> LocalBackend:
    return LocalBackend(remote_dir=str(tmp_path / "remote" / run_id), remote_repo=str(REPO), remote_python=sys.executable, gpus=0)


def prepared(tmp_path: Path, run_id: str = "r3b-unit", *, budget: Budget | None = None, backend: Backend | None = None,
             allow_no_cap: bool = False, **overrides) -> tuple[dict, Backend, Path]:  # fmt: skip
    backend = backend or local_backend(tmp_path, run_id)
    manifest_path = tmp_path / f"{run_id}.json"
    manifest = prepare(
        config=tiny_config(**overrides),
        config_path=TINY_CONFIG,
        run_id=run_id,
        backend=backend,
        budget=budget or Budget(hourly_usd=0.0, gpus=0, max_wall_hours=1.0),
        manifest_path=manifest_path,
        allow_no_cap=allow_no_cap,
    )
    return manifest, backend, manifest_path


# --------------------------------------------------------------------------
# A1 — 명세와 스키마
# --------------------------------------------------------------------------


def test_the_schema_file_is_the_contract_and_the_checker_knows_every_keyword_in_it():
    """스키마는 파일이 정본이다. 그 파일이 검사기가 모르는 키워드를 쓰면 **조용히 통과되지 않고** 거절된다."""
    assert SCHEMA_PATH.is_file(), f"{SCHEMA_PATH}가 없다 — docs/06 Task 6이 이름 지은 파일이다"
    document = schema()
    assert document["properties"]["schema_version"]["const"] == SCHEMA_VERSION
    assert set(document["properties"]["state"]["enum"]) == set(STATES)

    seen: set[str] = set()

    def walk(node) -> None:
        if isinstance(node, dict):
            if any(key in SUPPORTED_KEYWORDS for key in node):
                seen.update(node)
            for key, value in node.items():
                if key in ("properties", "items"):
                    walk(value)
                elif isinstance(value, (dict, list)) and key not in ("enum", "required"):
                    walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)

    walk(document)
    assert seen <= SUPPORTED_KEYWORDS, f"스키마가 검사기가 모르는 키워드를 쓴다: {sorted(seen - SUPPORTED_KEYWORDS)}"


def test_a_prepared_manifest_carries_everything_task_6_asked_for(tmp_path):
    """코드 리비전·설정·데이터·tokenizer·계약·seed·상한·산출물·상태 — 하나라도 비면 여기서 걸린다."""
    manifest, _, manifest_path = prepared(tmp_path)
    assert validate(manifest) == []
    assert manifest["schema_version"] == SCHEMA_VERSION
    assert manifest["state"] == "prepared"
    assert isinstance(manifest["code"]["dirty"], bool)
    assert len(manifest["train_config"]["sha256"]) == 64 and len(manifest["train_config"]["resolved_sha256"]) == 64
    assert manifest["dataset_manifests"] and all(len(entry["sha256"]) == 64 for entry in manifest["dataset_manifests"])
    assert manifest["tokenizer"]["id"] and manifest["contract"]["sha256"]
    assert manifest["seed"] == 17 and manifest["max_steps"] == FAST["max_steps"]
    assert {entry["kind"] for entry in manifest["artifacts"]} == {"checkpoint", "metrics", "config", "report", "state", "log"}
    assert manifest["history"][0]["state"] == "prepared"
    # 묶음: 설정·명세·데이터 manifest **파일**만. 데이터 레코드는 없다.
    bundle = bundle_dir(manifest_path)
    assert (bundle / "config.yaml").is_file() and (bundle / "manifest.json").is_file()
    assert list((bundle / "datasets").glob("*.json"))
    assert load_manifest(manifest_path)["run_id"] == manifest["run_id"]


@pytest.mark.parametrize(
    "mutate, needle",
    [
        (lambda m: m.pop("budget"), "필수 키 'budget'"),
        (lambda m: m.update(state="finished"), "manifest.state"),
        (lambda m: m.update(surprise=1), "모르는 키"),
        (lambda m: m["train_config"].update(sha256="nope"), "manifest.train_config.sha256"),
        (lambda m: m["budget"].update(gpus=-1), "manifest.budget.gpus"),
        (lambda m: m["artifacts"].append({"name": "x", "kind": "weights", "remote": "y"}), "manifest.artifacts"),
        (lambda m: m.update(max_steps=0), "manifest.max_steps"),
    ],
)
def test_the_schema_names_what_is_wrong(mutate, needle):
    manifest = new_manifest(run_id="x", artifacts=list(artifact_plan("x")))
    mutate(manifest)
    errors = validate(manifest)
    assert errors, "스키마가 이것을 잡았어야 한다"
    assert any(needle in error for error in errors), errors


def test_saving_an_invalid_manifest_is_refused(tmp_path):
    manifest = new_manifest(run_id="x")
    manifest["state"] = "melted"
    with pytest.raises(ValueError, match="스키마에 맞지 않는다"):
        save_manifest(tmp_path / "run.json", manifest)
    assert not (tmp_path / "run.json").exists()


def test_note_state_keeps_the_ledger_and_stamps_the_end():
    manifest = new_manifest(run_id="x")
    note_state(manifest, "running", note="떴다")
    assert manifest["finished_at"] is None
    note_state(manifest, "failed", note="예산")
    assert manifest["finished_at"] and [row["state"] for row in manifest["history"]] == ["running", "failed"]
    with pytest.raises(ValueError, match="state:"):
        note_state(manifest, "melted")


# --------------------------------------------------------------------------
# A3 — 상한의 산술
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "budget, hours, limit",
    [
        (Budget(hourly_usd=2.5, gpus=1, max_wall_hours=4.0), 4.0, "max_wall_hours"),
        (Budget(hourly_usd=2.5, gpus=1, max_wall_hours=4.0, max_usd=5.0), 2.0, "max_usd"),
        (Budget(hourly_usd=2.5, gpus=8, max_wall_hours=4.0, max_gpu_hours=8.0), 1.0, "max_gpu_hours"),
        (Budget(hourly_usd=31.92, gpus=8, max_usd=63.84, max_gpu_hours=24.0), 2.0, "max_usd"),
        # GPU가 없는 run·단가 0은 그 상한으로 구속되지 않는다 (0으로 나누지 않는다).
        (Budget(hourly_usd=0.0, gpus=0, max_usd=5.0, max_gpu_hours=5.0, max_wall_hours=3.0), 3.0, "max_wall_hours"),
        (Budget(hourly_usd=0.0, gpus=0), None, None),
    ],
)
def test_three_caps_become_one_deadline_and_the_earliest_binds(budget, hours, limit):
    deadline, name = budget_deadline_hours(budget)
    assert name == limit
    assert deadline is None if hours is None else deadline == pytest.approx(hours)


def test_spend_is_wall_times_gpus_and_wall_times_the_node_price():
    budget = Budget(hourly_usd=2.5, gpus=4)
    spend = budget_spend(budget, 3600.0)
    assert spend == {"wall_hours": 1.0, "gpu_hours": 4.0, "estimated_usd": 2.5}


def test_the_plan_says_out_loud_that_it_is_an_assumption():
    budget = Budget(hourly_usd=2.5, gpus=1, max_usd=20.0)
    plan = plan_budget(budget, seconds_per_step=63.79, max_steps=233, assumption="GB10 실측 · H100 배수 미측정")
    assert plan["wall_hours"] == pytest.approx(4.128, abs=1e-3)
    assert plan["usd"] == pytest.approx(10.32, abs=1e-2)
    assert "미측정" in plan["assumption"]
    assert plan_budget(budget, seconds_per_step=None, max_steps=10, assumption="모름")["usd"] is None


def test_remaining_budget_subtracts_what_the_parent_already_spent():
    manifest = new_manifest(run_id="x")
    manifest["budget"] = {"hourly_usd": 2.0, "gpus": 2, "max_wall_hours": 4.0, "max_gpu_hours": 8.0, "max_usd": 8.0}
    manifest["progress"] = {"elapsed_seconds": 3600.0}
    left = remaining_budget(manifest)
    assert (left.max_wall_hours, left.max_gpu_hours, left.max_usd) == (3.0, 6.0, 6.0)
    manifest["progress"] = {"elapsed_seconds": 40000.0}  # 다 썼어도 음수가 되지 않는다
    assert remaining_budget(manifest).max_wall_hours > 0


# --------------------------------------------------------------------------
# 돈 울타리 — 아무것도 구속하지 않는 명세는 만들어지지 않는다 (리뷰 1의 I3)
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "budget, needle",
    [
        # 상한을 아예 주지 않았다 — 원격의 마감이 None이라 `should_stop`이 영영 budget을 돌려주지 않는다.
        (Budget(hourly_usd=2.5, gpus=1), "구속하는 상한이 하나도 없다"),
        # 상한은 줬는데 나눌 divisor가 0이라 **버려진다** (GPU 0에 GPU-시간 상한).
        (Budget(hourly_usd=0.0, gpus=0, max_gpu_hours=5.0), "구속하는 상한이 하나도 없다"),
        # 돈 상한을 적었는데 단가가 0이다 — 사람이 적은 상한이 조용히 사라지는 짝이다.
        (Budget(hourly_usd=0.0, gpus=1, max_usd=5.0), "0 단가로는 비용 상한이"),
    ],
)
def test_prepare_refuses_a_run_that_no_cap_can_stop(tmp_path, budget, needle):
    """`prepare`가 만드는 것은 "돈을 쓰기 전에 못 박아 두는 것"이다 — 못 박히지 않은 명세는 만들지 않는다."""
    with pytest.raises(UncappedRun, match=needle):
        prepared(tmp_path, "r3b-uncapped", budget=budget)
    assert list(tmp_path.iterdir()) == [], "거절했으면 명세도 묶음도 남기지 않는다"


def test_the_cli_refuses_an_uncapped_prepare_and_says_why(tmp_path):
    """두 실수 모두 CLI에서 0이 아닌 종료 코드로 끝난다 — 상한을 잊는 것과 `--max-usd`에 단가 0을 짝짓는 것."""
    base = [sys.executable, str(REPO / "scripts" / "launch_run.py"), "prepare", "--config", str(TINY_CONFIG),
            "--run-id", "r3b-cli-uncapped", "--manifest", str(tmp_path / "run.json"), "--backend", "local"]  # fmt: skip
    for extra, needle in (
        (["--hourly-usd", "0"], "구속하는 상한이 하나도 없다"),
        (["--hourly-usd", "0", "--max-usd", "5"], "0 단가로는"),
    ):
        result = subprocess.run([*base, *extra], capture_output=True, text=True, check=False)
        assert result.returncode != 0, result.stdout
        assert needle in result.stderr + result.stdout
        assert not (tmp_path / "run.json").exists()


def test_an_uncapped_run_is_possible_but_only_by_name_and_it_is_written_down(tmp_path):
    """`--no-cap`은 상한 없이 돌 **권한**이 아니라 그렇게 하겠다는 **기록**이다 — 명세와 장부 양쪽에 남는다."""
    manifest, _, manifest_path = prepared(tmp_path, "r3b-nocap", budget=Budget(hourly_usd=0.0, gpus=0), allow_no_cap=True)
    assert validate(manifest) == []
    assert budget_deadline_hours(Budget.from_manifest(manifest)) == (None, None)
    assert any("--no-cap" in note for note in manifest["notes"]), manifest["notes"]
    assert "상한 없음" in manifest["history"][-1]["note"]
    assert "--no-cap" in load_manifest(manifest_path)["history"][-1]["note"]


# --------------------------------------------------------------------------
# 원격 러너 — 상한을 스스로 잰다
# --------------------------------------------------------------------------


def _run_the_runner(tmp_path: Path, budget: Budget, run_id: str, **overrides) -> tuple[int, dict, Path]:
    manifest, backend, manifest_path = prepared(tmp_path, run_id, budget=budget, **overrides)
    bundle = bundle_dir(manifest_path)
    Path(backend.remote_dir).mkdir(parents=True, exist_ok=True)
    target = Path(backend.remote_dir) / "bundle"
    backend.push(bundle, "bundle")
    code = run_bundle(target, repo=REPO)
    state = json.loads((Path(backend.remote_dir) / "state.json").read_text(encoding="utf-8"))
    return code, state, Path(backend.remote_dir)


def test_the_runner_stops_itself_at_the_wall_cap_saves_a_checkpoint_and_says_budget(tmp_path):
    """상한은 **원격이** 잰다 — 실행기가 죽어도 여기서 멈춘다. 그리고 멈추기 전에 checkpoint를 쓴다."""
    code, state, remote = _run_the_runner(tmp_path, Budget(hourly_usd=0.0, gpus=0, max_wall_hours=0.0008), "r3b-wall", max_steps=8)
    assert code == 2
    assert (state["state"], state["exit_reason"], state["budget_limit"]) == ("failed", "budget", "max_wall_hours")
    assert state["step"] < 8, "상한에 걸렸으니 끝까지 가지 않았다"
    assert Path(state["checkpoint"]).is_file(), "예산으로 멈출 때 checkpoint가 남아야 한다"
    assert (remote / "runs" / "r3b-wall" / "metrics.json").is_file()


def test_the_runner_names_the_usd_cap_when_money_binds_first(tmp_path):
    """벽시계는 넉넉한데 돈이 먼저 떨어지면 구속한 상한의 **이름**이 `max_usd`여야 한다."""
    budget = Budget(hourly_usd=2.5, gpus=1, max_wall_hours=1.0, max_usd=0.002)  # 0.002/2.5 h = 2.88 s
    code, state, _ = _run_the_runner(tmp_path, budget, "r3b-usd", max_steps=8)
    assert code == 2
    assert (state["exit_reason"], state["budget_limit"]) == ("budget", "max_usd")
    assert state["estimated_usd"] >= 0.002 * 0.9
    assert Path(state["checkpoint"]).is_file()


def test_the_runner_refuses_to_start_on_data_whose_hash_moved(tmp_path):
    """데이터는 묶음에 담지 않는다(경로와 sha만) — 그러므로 시작 **전에** 원격 파일을 다시 해시해 대조한다."""
    manifest, backend, manifest_path = prepared(tmp_path, "r3b-inputs")
    bundle = bundle_dir(manifest_path)
    poisoned = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    poisoned["dataset_manifests"][0]["sha256"] = "0" * 64
    (bundle / "manifest.json").write_text(json.dumps(poisoned), encoding="utf-8")
    backend.push(bundle, "bundle")
    code = run_bundle(Path(backend.remote_dir) / "bundle", repo=REPO)
    state = json.loads((Path(backend.remote_dir) / "state.json").read_text(encoding="utf-8"))
    assert code == 4
    assert (state["state"], state["exit_reason"]) == ("failed", "inputs")
    assert "sha256" in state["error"]
    assert not (Path(backend.remote_dir) / "runs").exists(), "학습을 시작하지 않았어야 한다"


@pytest.mark.parametrize(
    "wall, deadline, term, verdict",
    [
        (0.5, 1.0, False, None),
        (1.0, 1.0, False, "budget"),
        (1.5, 1.0, False, "budget"),
        (0.1, None, False, None),
        (0.1, None, True, "cancelled"),
        (0.1, 1.0, True, "cancelled"),
        # 돈이 떨어진 것과 정지가 같이 오면 **예산**으로 적는다 — 돈 때문에 끝난 run을 취소로 기록하지 않는다.
        (2.0, 1.0, True, "budget"),
    ],
)
def test_the_step_boundary_rule_is_one_function(wall, deadline, term, verdict):
    assert should_stop(wall, deadline, term=term) == verdict


def test_a_term_signal_saves_a_checkpoint_and_ends_as_cancelled(tmp_path):
    """`systemctl --user stop`이 보내는 SIGTERM — 러너가 다음 step 경계에서 checkpoint를 쓰고 `cancelled`로 끝난다."""
    import os
    import signal as signal_module
    import time

    manifest, backend, manifest_path = prepared(tmp_path, "r3b-term", max_steps=40)
    backend.push(bundle_dir(manifest_path), "bundle")
    remote = Path(backend.remote_dir)
    process = subprocess.Popen(
        [sys.executable, "-m", "robo_jev.launch.runner", "--bundle", str(remote / "bundle"), "--repo", str(REPO)],
        cwd=str(REPO), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )  # fmt: skip
    try:
        deadline = time.monotonic() + 120
        while time.monotonic() < deadline:
            state_path = remote / "state.json"
            if state_path.is_file() and (json.loads(state_path.read_text(encoding="utf-8")).get("step") or 0) >= 1:
                break
            time.sleep(0.25)
        else:  # pragma: no cover - 느린 상자
            pytest.fail("러너가 첫 step을 못 돌았다")
        os.kill(process.pid, signal_module.SIGTERM)
        assert process.wait(timeout=120) == 3
    finally:
        if process.poll() is None:  # pragma: no cover
            process.kill()
    state = json.loads((remote / "state.json").read_text(encoding="utf-8"))
    assert (state["state"], state["exit_reason"]) == ("cancelled", "cancelled")
    assert state["step"] >= 1 and Path(state["checkpoint"]).is_file()


def test_two_dataset_manifests_with_the_same_basename_are_told_apart(tmp_path):
    """로봇과 비로봇의 manifest는 **둘 다 `manifest.json`**이다 — 이름으로 짝지으면 서로의 해시와 견주게 된다."""
    from robo_jev.launch.manifest import sha256_of

    paths = []
    for index, body in enumerate(('{"a": 1}', '{"b": 2}')):
        directory = tmp_path / "repo" / "data" / f"set{index}"
        directory.mkdir(parents=True)
        (directory / "manifest.json").write_text(body, encoding="utf-8")
        paths.append(f"data/set{index}/manifest.json")
    repo = tmp_path / "repo"
    manifest = {"dataset_manifests": [{"path": path, "sha256": sha256_of(repo / path)} for path in paths]}
    config = {"dataset_manifests": [{"path": path} for path in paths]}
    assert check_inputs(manifest, config, repo) == []

    manifest["dataset_manifests"][0]["sha256"] = "0" * 64
    problems = check_inputs(manifest, config, repo)
    assert len(problems) == 1 and paths[0] in problems[0] and "sha256이 다르다" in problems[0]


def test_inventory_hashes_what_it_finds_and_nothing_else(tmp_path):
    (tmp_path / "runs" / "a").mkdir(parents=True)
    (tmp_path / "runs" / "a" / "checkpoint.pt").write_bytes(b"x" * 10)
    (tmp_path / "runs" / "a" / "notes.txt").write_text("skip", encoding="utf-8")
    found = inventory(tmp_path, ["runs/*/checkpoint.pt"])
    assert [entry["path"] for entry in found] == ["runs/a/checkpoint.pt"]
    assert found[0]["bytes"] == 10 and len(found[0]["sha256"]) == 64


# --------------------------------------------------------------------------
# B1 — 왕복 (local 백엔드: SSH 경로와 같은 코드)
# --------------------------------------------------------------------------


def test_the_round_trip_runs_end_to_end_and_every_fetched_file_matches_its_remote_hash(tmp_path):
    manifest, backend, manifest_path = prepared(tmp_path, "r3b-trip")
    manifest = launch(manifest, backend, manifest_path=manifest_path)
    assert manifest["state"] == "running"
    assert manifest["backend"]["launcher"] in ("systemd-run", "nohup")

    import time

    deadline = time.monotonic() + 180
    while manifest["state"] == "running" and time.monotonic() < deadline:
        time.sleep(1.0)
        manifest = status(manifest, backend, manifest_path=manifest_path)
    assert manifest["state"] == "completed", manifest["history"]
    assert manifest["exit_reason"] == "completed"
    assert manifest["progress"]["step"] == FAST["max_steps"]
    assert manifest["progress"]["estimated_usd"] is not None
    assert manifest["finished_at"]

    manifest = fetch(manifest, backend, dest=tmp_path / "fetched", manifest_path=manifest_path)
    got = {entry["kind"]: entry for entry in manifest["artifacts"] if entry.get("local")}
    assert {"checkpoint", "metrics", "config", "state", "log"} <= set(got)
    for entry in got.values():
        assert entry["sha256_match"] is True, entry
        assert Path(entry["local"]).is_file() and entry["bytes"] > 0
    assert validate(load_manifest(manifest_path)) == []


def test_a_killed_run_is_failed_not_completed(tmp_path):
    """상태 파일이 `running`인데 프로세스가 없으면 그 run은 **강제 종료된 것**이고 성공이 아니다."""
    manifest, backend, manifest_path = prepared(tmp_path, "r3b-gone")
    Path(backend.remote_dir).mkdir(parents=True, exist_ok=True)
    (Path(backend.remote_dir) / "state.json").write_text(json.dumps({"state": "running", "step": 3, "max_steps": 8}), encoding="utf-8")
    manifest["backend"].update({"launcher": "nohup", "pid": 2 ** 22 - 1, "unit": None})  # 없는 pid
    manifest["state"] = "running"
    manifest = status(manifest, backend, manifest_path=manifest_path)
    assert (manifest["state"], manifest["exit_reason"]) == ("failed", "process_vanished")
    assert "강제 종료" in manifest["history"][-1]["note"]


def test_fetch_refuses_a_file_whose_hash_moved_in_flight(tmp_path):
    manifest, backend, manifest_path = prepared(tmp_path, "r3b-corrupt")
    remote = Path(backend.remote_dir) / "runs" / "r3b-corrupt"
    remote.mkdir(parents=True)
    for name in ("checkpoint.pt", "metrics.json", "config.yaml"):
        (remote / name).write_text("ok", encoding="utf-8")
    (Path(backend.remote_dir) / "state.json").write_text("{}", encoding="utf-8")
    (Path(backend.remote_dir) / "logs").mkdir(exist_ok=True)
    (Path(backend.remote_dir) / "logs" / "runner.log").write_text("ok", encoding="utf-8")

    real_inventory = backend.inventory

    def lying_inventory(patterns):
        items = real_inventory(patterns)
        for item in items:
            if item["path"].endswith("checkpoint.pt"):
                item["sha256"] = "f" * 64
        return items

    backend.inventory = lying_inventory  # type: ignore[method-assign]
    with pytest.raises(ValueError, match="sha256이 원격과 다르다"):
        fetch(manifest, backend, dest=tmp_path / "fetched", manifest_path=manifest_path)


def test_a_completed_run_must_have_left_everything(tmp_path):
    manifest, backend, manifest_path = prepared(tmp_path, "r3b-empty")
    backend.mkdir()
    manifest["state"] = "completed"
    with pytest.raises(FileNotFoundError, match="원격에 없는 산출물"):
        fetch(manifest, backend, dest=tmp_path / "fetched", manifest_path=manifest_path)


def test_a_killed_run_is_fetched_in_part_and_what_is_missing_is_written_down(tmp_path):
    """강제 종료된 run은 **부분적인 것이 정상**이다 — 있는 것을 가져오고 없는 것의 이름을 남긴다."""
    manifest, backend, manifest_path = prepared(tmp_path, "r3b-partial")
    remote = Path(backend.remote_dir) / "runs" / "r3b-partial"
    remote.mkdir(parents=True)
    (remote / "checkpoint.pt").write_text("weights", encoding="utf-8")  # metrics.json·config.yaml은 없다
    (Path(backend.remote_dir) / "logs").mkdir(exist_ok=True)
    (Path(backend.remote_dir) / "logs" / "runner.log").write_text("log", encoding="utf-8")
    (Path(backend.remote_dir) / "state.json").write_text('{"state": "running"}', encoding="utf-8")
    manifest["state"] = "failed"
    manifest["exit_reason"] = "process_vanished"

    manifest = fetch(manifest, backend, dest=tmp_path / "fetched", manifest_path=manifest_path)
    got = {entry["kind"]: entry for entry in manifest["artifacts"] if entry.get("local")}
    assert got["checkpoint"]["sha256_match"] is True
    assert "metrics" not in got and "config" not in got
    assert "원격에 없던 것" in manifest["history"][-1]["note"]
    assert "metrics.json" in manifest["history"][-1]["note"]
    # 그런데도 **일부러 엄격하게** 부르면 실패한다.
    with pytest.raises(FileNotFoundError):
        fetch(manifest, backend, dest=tmp_path / "fetched2", strict=True)


# --------------------------------------------------------------------------
# B2 (iv) — 취소의 확인 실패는 `unknown`이다
# --------------------------------------------------------------------------


class StubBackend(Backend):
    """생사와 정지 결과를 시나리오로 주는 백엔드 (취소 규칙만 본다)."""

    kind = "local"

    def __init__(self, tmp_path: Path, *, alive: list[bool], stop_ok: bool = True) -> None:
        self.remote_dir = str(tmp_path / "remote")
        self.remote_repo = str(REPO)
        self.remote_python = sys.executable
        self.gpus = 0
        self._alive = list(alive)
        self._stop_ok = stop_ok
        self.stopped: list[str] = []
        Path(self.remote_dir).mkdir(parents=True, exist_ok=True)

    def _exec(self, argv, *, timeout=None):  # pragma: no cover - 쓰이지 않는다
        return Exec(argv=argv, returncode=0, stdout="", stderr="")

    def push(self, local_dir, remote_relpath):  # pragma: no cover
        return Exec(argv=[], returncode=0, stdout="", stderr="")

    def push_file(self, local_file, remote_relpath):  # pragma: no cover
        return Exec(argv=[], returncode=0, stdout="", stderr="")

    def fetch(self, relpaths, dest):  # pragma: no cover
        return Exec(argv=[], returncode=0, stdout="", stderr="")

    def read_text(self, relpath):
        path = Path(self.remote_dir) / relpath
        return path.read_text(encoding="utf-8") if path.is_file() else None

    def alive(self, handle):
        return self._alive.pop(0) if self._alive else False

    def stop(self, handle, *, signal="TERM"):
        self.stopped.append(signal)
        return Exec(argv=["stop"], returncode=0 if self._stop_ok else 1, stdout="", stderr="" if self._stop_ok else "API가 500을 냈다")


def _launched(tmp_path: Path, backend: Backend) -> tuple[dict, Path]:
    manifest, _, manifest_path = prepared(tmp_path, "r3b-cancel", backend=backend)
    manifest["backend"].update({"launcher": "systemd-run", "unit": "r3b-cancel", "pid": None})
    manifest["state"] = "running"
    return manifest, manifest_path


def test_a_stop_we_cannot_confirm_leaves_unknown_and_shouts(tmp_path):
    """docs/06 Task 6: 종료 API 실패는 **미종료 상태로 명시**한다. 돈이 계속 나갈 수 있기 때문이다."""
    backend = StubBackend(tmp_path, alive=[True, True, True, True])
    manifest, manifest_path = _launched(tmp_path, backend)
    with pytest.raises(CancelNotConfirmed, match="확인하지 못했다"):
        cancel(manifest, backend, manifest_path=manifest_path, confirm_timeout=0.0, sleep=lambda _: None)
    saved = load_manifest(manifest_path)
    assert saved["state"] == "unknown"
    assert saved["exit_reason"] == "cancel-unconfirmed"
    assert "사람이 확인한다" in saved["history"][-1]["note"]


def test_a_stop_command_that_fails_leaves_unknown_too(tmp_path):
    backend = StubBackend(tmp_path, alive=[True], stop_ok=False)
    manifest, manifest_path = _launched(tmp_path, backend)
    with pytest.raises(CancelNotConfirmed, match="정지 명령이 실패했다"):
        cancel(manifest, backend, manifest_path=manifest_path, confirm_timeout=30.0, sleep=lambda _: None)
    saved = load_manifest(manifest_path)
    assert (saved["state"], saved["exit_reason"]) == ("unknown", "cancel-command-failed")


def test_a_confirmed_stop_is_cancelled_and_reads_the_remote_reason(tmp_path):
    backend = StubBackend(tmp_path, alive=[True, False])
    manifest, manifest_path = _launched(tmp_path, backend)
    (Path(backend.remote_dir) / "state.json").write_text(json.dumps({"state": "cancelled", "exit_reason": "cancelled", "step": 2}), encoding="utf-8")
    manifest = cancel(manifest, backend, manifest_path=manifest_path, confirm_timeout=5.0, sleep=lambda _: None)
    assert (manifest["state"], manifest["exit_reason"]) == ("cancelled", "cancelled")
    assert backend.stopped == ["TERM"]


# --------------------------------------------------------------------------
# 재개는 등록된 범위에서만
# --------------------------------------------------------------------------


def test_resume_refuses_a_checkpoint_that_is_not_registered(tmp_path):
    manifest, backend, manifest_path = prepared(tmp_path, "r3b-resume")
    stray = tmp_path / "stray.pt"
    stray.write_bytes(b"not ours")
    with pytest.raises(ValueError, match="등록된 checkpoint가 아니다"):
        registered_checkpoint(manifest, stray)


def test_resume_refuses_a_registered_checkpoint_whose_bytes_moved(tmp_path):
    manifest, backend, manifest_path = prepared(tmp_path, "r3b-resume2")
    local = tmp_path / "checkpoint.pt"
    local.write_bytes(b"first")
    manifest["artifacts"] = [{"name": "checkpoint", "kind": "checkpoint", "remote": "runs/x/checkpoint.pt",
                              "local": str(local), "sha256": "a" * 64, "sha256_match": True, "bytes": 5}]  # fmt: skip
    with pytest.raises(ValueError, match="등록된 sha256과 지금 파일이 다르다"):
        registered_checkpoint(manifest, local)
    manifest["artifacts"][0]["sha256_match"] = False
    with pytest.raises(ValueError, match="대조되지 않았다"):
        registered_checkpoint(manifest, local)


def test_the_parents_spend_is_read_from_the_runs_own_state_file_not_from_the_last_poll(tmp_path):
    """`fetch`는 `progress`를 쓰지 않는다 — `launch → fetch`만 한 run은 `progress`가 **비어 있다**.

    그때 지출을 0으로 치면 자식이 부모의 상한을 통째로 다시 받는다(리뷰 1의 I4: 0.25 h를 그대로 물려받았다).
    권위 있는 값은 원격 러너가 쓰고 `fetch`가 sha256으로 대조한 `state.json`이다.
    """
    state = tmp_path / "fetched" / "state.json"
    state.parent.mkdir(parents=True, exist_ok=True)
    state.write_text(json.dumps({"state": "failed", "exit_reason": "budget", "elapsed_seconds": 27.866}), encoding="utf-8")
    manifest = new_manifest(run_id="r3b-observed")
    manifest["budget"] = {"hourly_usd": 2.0, "gpus": 1, "max_wall_hours": 0.25, "max_gpu_hours": None, "max_usd": None}
    manifest["artifacts"] = [{"name": "state", "kind": "state", "remote": "state.json", "local": str(state),
                              "sha256": "a" * 64, "sha256_match": True, "bytes": state.stat().st_size}]  # fmt: skip

    elapsed, source = observed_spend_seconds(manifest)
    assert (elapsed, "state.json" in source) == (27.866, True)
    # 27.866 s = 0.0077406 h → 0.25 − 0.0077406 = 0.242259 h. 옛 코드는 여기서 0.25(부모의 상한 전부)를 줬다.
    assert remaining_budget(manifest).max_wall_hours == pytest.approx(0.242259, abs=5e-7)

    # 마지막 `status`가 더 나중이면 그쪽이 더 크다 — 둘 다 있으면 **적게 세지 않는 쪽**을 쓴다.
    manifest["progress"] = {"elapsed_seconds": 40.0}
    assert observed_spend_seconds(manifest)[0] == 40.0
    manifest["progress"] = {"elapsed_seconds": 10.0}
    assert observed_spend_seconds(manifest)[0] == 27.866


def test_resume_refuses_a_parent_whose_spend_was_never_observed(tmp_path):
    """관측이 하나도 없으면 '안 썼다'가 아니라 '모른다'다 — 0으로 치지 않고 `status`를 먼저 하라고 말한다."""
    from robo_jev.launch.manifest import sha256_of

    budget = Budget(hourly_usd=2.0, gpus=1, max_wall_hours=0.25)
    parent, backend, _ = prepared(tmp_path, "r3b-unobserved", budget=budget)
    checkpoint = tmp_path / "fetched" / "checkpoint.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    checkpoint.write_bytes(b"weights")
    parent["artifacts"] = [{"name": "checkpoint", "kind": "checkpoint", "remote": "runs/p/checkpoint.pt",
                            "local": str(checkpoint), "sha256": sha256_of(checkpoint), "sha256_match": True, "bytes": 7}]  # fmt: skip
    parent["progress"] = {}  # launch → fetch만 했다: 합법적인 순서이고 progress는 비어 있다

    with pytest.raises(SpendNotObserved, match="한 번도 관측되지 않았다"):
        remaining_budget(parent)
    with pytest.raises(SpendNotObserved, match="status"):
        resume(parent, backend, checkpoint=checkpoint, run_id="r3b-unobserved-r1",
               manifest_path=tmp_path / "child.json", config=tiny_config(), config_path=TINY_CONFIG)  # fmt: skip
    assert not (tmp_path / "child.json").exists()
    assert not Path(backend.remote_dir).with_name("r3b-unobserved-r1").exists(), "거절 전에 원격에 아무것도 올리지 않는다"


def test_the_child_run_gets_its_own_remote_dir_and_the_remaining_budget(tmp_path):
    """자식은 부모의 `state.json`을 덮어쓰지 않고, 예산을 새로 시작하지도 않는다."""
    budget = Budget(hourly_usd=2.0, gpus=1, max_wall_hours=1.0, max_usd=2.0)
    parent, backend, manifest_path = prepared(tmp_path, "r3b-parent", budget=budget)
    checkpoint = tmp_path / "fetched" / "checkpoint.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    checkpoint.write_bytes(b"weights")
    from robo_jev.launch.manifest import sha256_of

    parent["artifacts"] = [{"name": "checkpoint", "kind": "checkpoint", "remote": "runs/r3b-parent/checkpoint.pt",
                            "local": str(checkpoint), "sha256": sha256_of(checkpoint), "sha256_match": True, "bytes": 7}]  # fmt: skip
    parent["progress"] = {"elapsed_seconds": 1800.0}

    child = resume(parent, backend, checkpoint=checkpoint, run_id="r3b-child", manifest_path=tmp_path / "child.json",
                   config=tiny_config(), config_path=TINY_CONFIG)  # fmt: skip
    assert validate(child) == []
    assert child["parent_run_id"] == "r3b-parent"
    assert child["backend"]["remote_dir"] != parent["backend"]["remote_dir"]
    assert child["backend"]["remote_dir"].endswith("r3b-child")
    assert child["budget"]["max_wall_hours"] == pytest.approx(0.5)
    assert child["budget"]["max_usd"] == pytest.approx(1.0)
    assert child["checkpoint"]["resume_from"].endswith("inbox/checkpoint.pt")
    assert Path(child["checkpoint"]["resume_from"]).is_file()
    assert "이어간다" in " ".join(child["notes"])


def test_re_authorising_the_budget_is_written_down(tmp_path):
    """사람이 예산을 다시 승인해 상한을 올릴 수 있다 — 다만 **조용히** 올라가지 않는다."""
    from robo_jev.launch.manifest import sha256_of

    budget = Budget(hourly_usd=2.0, gpus=1, max_wall_hours=1.0, max_usd=2.0)
    parent, backend, manifest_path = prepared(tmp_path, "r3b-parent2", budget=budget)
    checkpoint = tmp_path / "fetched" / "checkpoint.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    checkpoint.write_bytes(b"weights")
    parent["artifacts"] = [{"name": "checkpoint", "kind": "checkpoint", "remote": "runs/p/checkpoint.pt",
                            "local": str(checkpoint), "sha256": sha256_of(checkpoint), "sha256_match": True, "bytes": 7}]  # fmt: skip
    parent["progress"] = {"elapsed_seconds": 3400.0}
    child = resume(parent, backend, checkpoint=checkpoint, run_id="r3b-child2", manifest_path=tmp_path / "child2.json",
                   config=tiny_config(), config_path=TINY_CONFIG, budget=Budget(hourly_usd=2.0, gpus=1, max_wall_hours=6.0))  # fmt: skip
    assert child["budget"]["max_wall_hours"] == 6.0
    assert any("다시 승인했다" in note for note in child["notes"]), child["notes"]


# --------------------------------------------------------------------------
# A4 — SSH 백엔드 (가짜 전송으로 명령 문장을 본다)
# --------------------------------------------------------------------------


class FakeTransport:
    def __init__(self, *results: Exec) -> None:
        self.calls: list[list[str]] = []
        self.results = list(results)

    def __call__(self, argv, *, timeout=None):
        self.calls.append(list(argv))
        if self.results:
            return self.results.pop(0)
        return Exec(argv=list(argv), returncode=0, stdout="", stderr="")


def ssh_backend(transport: FakeTransport, **kwargs) -> SshBackend:
    return SshBackend(host="box.example", user="rj", port=2222, remote_dir="/w/run", remote_repo="/w/repo",
                      remote_python="/w/repo/.venv/bin/python", gpus=8, identity_path="/keys/id_ed25519",
                      known_hosts="/keys/known_hosts", transport=transport, **kwargs)  # fmt: skip


def test_the_ssh_backend_builds_the_command_we_think_it_does():
    transport = FakeTransport()
    backend = ssh_backend(transport)
    backend.shell("echo hi")
    argv = transport.calls[-1]
    assert argv[0] == "ssh"
    # 비대화식이 기본이다 — 암호를 묻는 자리가 생기면 분리 실행이 조용히 멈춘다.
    assert argv[1 : 1 + len(DEFAULT_SSH_OPTIONS)] == list(DEFAULT_SSH_OPTIONS)
    assert "BatchMode=yes" in argv and "-p" in argv and "2222" in argv
    assert argv[argv.index("-i") + 1] == "/keys/id_ed25519"
    assert "IdentitiesOnly=yes" in argv
    assert "UserKnownHostsFile=/keys/known_hosts" in argv
    assert argv[-2] == "rj@box.example"
    assert shlex.split(argv[-1]) == ["sh", "-c", "echo hi"]


def test_rsync_goes_over_the_same_ssh_options_not_a_second_path(tmp_path):
    transport = FakeTransport()
    backend = ssh_backend(transport)
    (tmp_path / "bundle").mkdir()
    backend.push(tmp_path / "bundle", "bundle")
    rsync = [call for call in transport.calls if call[0] == "rsync"][-1]
    rsh = rsync[rsync.index("-e") + 1]
    assert "-i /keys/id_ed25519" in rsh and "-p 2222" in rsh and "BatchMode=yes" in rsh
    assert rsync[-1] == "rj@box.example:/w/run/bundle/"


def test_the_detached_launch_prefers_systemd_and_falls_back_to_nohup():
    ok = FakeTransport()
    backend = ssh_backend(ok)
    handle = backend.start(["/w/repo/.venv/bin/python", "-m", "robo_jev.launch.runner", "--bundle", "/w/run/bundle"], unit="r3b-x", log_relpath="logs/runner.log")
    line = shlex.split(ok.calls[-1][-1])[-1]
    assert handle == Handle(launcher="systemd-run", unit="r3b-x", pid=None)
    assert "systemd-run --user --unit=r3b-x --collect" in line
    assert "StandardOutput=append:/w/run/logs/runner.log" in line

    broken = FakeTransport(Exec(argv=[], returncode=1, stdout="", stderr="no user bus"), Exec(argv=[], returncode=0, stdout="4242\n", stderr=""))
    handle = ssh_backend(broken).start(["python", "-m", "robo_jev.launch.runner"], unit="r3b-x", log_relpath="logs/runner.log")
    assert handle == Handle(launcher="nohup", unit=None, pid=4242)
    assert "nohup" in shlex.split(broken.calls[-1][-1])[-1]


def test_liveness_and_stop_use_the_units_own_words():
    transport = FakeTransport(Exec(argv=[], returncode=0, stdout="active\n", stderr=""))
    backend = ssh_backend(transport)
    assert backend.alive(Handle(launcher="systemd-run", unit="r3b-x")) is True
    backend.stop(Handle(launcher="systemd-run", unit="r3b-x"))
    # 정지는 **요청**이다 — `--no-block`이라 돌아왔다고 멈춘 것이 아니다 (확인은 실행기가 한다).
    assert "systemctl --user stop --no-block r3b-x" in shlex.split(transport.calls[-1][-1])[-1]
    backend.stop(Handle(launcher="nohup", pid=99), signal="KILL")
    assert "kill -KILL 99" in shlex.split(transport.calls[-1][-1])[-1]


def test_paths_inside_the_repo_are_written_relative_so_another_box_can_resolve_them(tmp_path):
    """명세와 묶음은 **다른 상자**에서 읽힌다 — 이 체크아웃의 절대 경로를 적으면 거기서 풀리지 않는다."""
    manifest, _, manifest_path = prepared(tmp_path, "r3b-portable")
    assert manifest["train_config"]["path"] == "configs/train/tiny_cpu.yaml"
    for entry in manifest["dataset_manifests"]:
        assert not Path(entry["path"]).is_absolute(), entry["path"]
    config = yaml.safe_load((bundle_dir(manifest_path) / "config.yaml").read_text(encoding="utf-8"))
    assert [entry["path"] for entry in config["dataset_manifests"]] == [entry["path"] for entry in manifest["dataset_manifests"]]
    assert not Path(config["model_config"]).is_absolute()
    # 저장소 **밖**의 경로(영속 볼륨 위의 데이터)는 그대로 둔다.
    assert portable_path("/mnt/persist/data/manifest.json") == "/mnt/persist/data/manifest.json"


def test_a_symlinked_directory_inside_the_repo_still_comes_out_relative(tmp_path, monkeypatch):
    """`artifacts`는 worktree에서 본 저장소를 가리키는 symlink다 — `resolve()`만 쓰면 저장소 밖으로 나가
    다른 상자에서 못 푸는 절대 경로가 명세에 남는다."""
    repo, outside = tmp_path / "repo", tmp_path / "outside"
    (outside / "datasets").mkdir(parents=True)
    (outside / "datasets" / "manifest.json").write_text("{}", encoding="utf-8")
    repo.mkdir()
    (repo / "artifacts").symlink_to(outside)
    monkeypatch.setattr(launcher_module, "REPO", repo)
    assert portable_path(repo / "artifacts" / "datasets" / "manifest.json") == "artifacts/datasets/manifest.json"


def test_the_backend_block_round_trips_through_the_manifest(tmp_path):
    """두 번째 명령이 첫 번째와 **같은 상자**를 봐야 한다 — known_hosts·키 경로·포트가 명세에 남는다."""
    import importlib.util

    backend = SshBackend(host="box", user="rj", port=2222, remote_dir="/w/run", remote_repo="/w/repo",
                         remote_python="uv run python", gpus=4, identity_path="/keys/id", known_hosts="/keys/known_hosts",
                         transport=FakeTransport())  # fmt: skip
    manifest, _, manifest_path = prepared(tmp_path, "r3b-roundtrip-backend", backend=backend)
    spec = importlib.util.spec_from_file_location("launch_run", REPO / "scripts" / "launch_run.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["launch_run"] = module
    spec.loader.exec_module(module)
    again = module.backend_from_manifest(load_manifest(manifest_path), transport=FakeTransport())
    assert (again.host, again.user, again.port) == ("box", "rj", 2222)
    assert (again.identity_path, again.known_hosts) == ("/keys/id", "/keys/known_hosts")
    assert (again.remote_python, again.gpus) == ("uv run python", 4)


def test_no_key_material_reaches_the_manifest(tmp_path):
    """자격 증명은 저장소에도 명세에도 들어가지 않는다 — 키는 **경로**로만 가리킨다."""
    key = tmp_path / "id_ed25519"
    key.write_text("-----BEGIN OPENSSH PRIVATE KEY-----\nSECRETSECRET\n", encoding="utf-8")
    backend = SshBackend(host="box", user="rj", remote_dir="/w/run", remote_repo="/w/repo", remote_python="python",
                         identity_path=str(key), transport=FakeTransport())  # fmt: skip
    described = backend.describe()
    assert described["identity_path"] == str(key)
    assert "SECRETSECRET" not in json.dumps(described)
    manifest, _, manifest_path = prepared(tmp_path, "r3b-keys", backend=backend)
    assert "SECRETSECRET" not in manifest_path.read_text(encoding="utf-8")


def test_the_ssh_and_local_backends_share_one_implementation_of_the_risky_parts():
    """분리 실행·생사·정지·목록이 백엔드마다 따로 쓰여 있으면 `local`로 통과한 인수가 SSH를 보증하지 못한다."""
    for name in ("start", "alive", "stop", "inventory", "read_text", "mkdir"):
        assert getattr(SshBackend, name) is getattr(Backend, name), f"{name}이 SshBackend에서 갈라졌다"
        assert getattr(LocalBackend, name) is getattr(Backend, name), f"{name}이 LocalBackend에서 갈라졌다"


def test_the_provider_interface_is_there_but_not_implemented():
    """인스턴스 생성·종료는 인터페이스만 두고 이월한다 (docs/06 Task 6)."""
    from robo_jev.launch.providers.base import Provider

    assert {"create", "destroy", "describe"} <= set(dir(Provider))


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def test_the_cli_offers_exactly_the_six_commands_task_6_named():
    import importlib.util

    spec = importlib.util.spec_from_file_location("launch_run", REPO / "scripts" / "launch_run.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["launch_run"] = module
    spec.loader.exec_module(module)
    parser = module.build_parser()
    actions = [action for action in parser._actions if hasattr(action, "choices") and action.dest == "command"]
    assert set(actions[0].choices) == {"prepare", "launch", "status", "fetch", "cancel", "resume"}
    assert module.COMMANDS == ("prepare", "launch", "status", "fetch", "cancel", "resume")


def test_a_config_with_modes_will_not_be_sent_without_a_mode():
    """`extends`/`modes`를 쓰는 설정은 어떤 판(t0/lora/t1)을 보낼지 사람이 골라야 한다 — 조용히 하나를 택하지 않는다."""
    result = subprocess.run(
        [sys.executable, str(REPO / "scripts" / "launch_run.py"), "prepare", "--config",
         str(REPO / "configs" / "train" / "qwen35-2b-r2.yaml"), "--run-id", "x", "--manifest", "/tmp/nope.json",
         "--backend", "local", "--hourly-usd", "0"],
        capture_output=True, text=True, check=False,
    )  # fmt: skip
    assert result.returncode != 0 and "--mode" in (result.stderr + result.stdout)


def test_the_cli_refuses_ssh_without_a_host():
    result = subprocess.run(
        [sys.executable, str(REPO / "scripts" / "launch_run.py"), "prepare", "--config", str(TINY_CONFIG),
         "--run-id", "x", "--manifest", "/tmp/nope.json", "--backend", "ssh", "--hourly-usd", "0"],
        capture_output=True, text=True, check=False,
    )  # fmt: skip
    assert result.returncode != 0 and "--host" in (result.stderr + result.stdout)
