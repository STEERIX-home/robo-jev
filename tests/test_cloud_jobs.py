"""판단 학습 job (`kind: decisions`) — 정의 검사, 원격 러너의 단계·마감·정지, local 백엔드 끝까지 (Task cloud1).

러너 쪽(:func:`robo_jev.launch.jobs.run_job`)은 진짜 하위 프로세스를 돈다. 끝까지 검사 둘:

* **평가만** — 학습 없는 소형 난수 Qwen(`--model tiny`)으로 `scripts/diagnose_decisions.py`를 job의 한 단계로 돌리고 `fetch`가 보고서·예측을
  sha256으로 가져온다 (이 브랜치에 있는 도구만 쓴다).
* **학습 + 평가** — Task v0d의 `scripts/train_decisions.py`가 있으면(병합 뒤) 소형 모델로 LoRA 몇 step → `--checkpoint`로 진단 →
  checkpoint·metrics·보고서를 가져온다. 학습기가 아직 없으면 그 이유로 건너뛴다.

CPU에서 소형 모델을 하위 프로세스로 돌리면 transformers가 fla(Triton) 커널을 골라 "0 active drivers"로 멈춘다 — 검사 전용 감싸개
(`tests/fixtures/launch/with_reference_kernels.py`)가 torch 참조 구현으로 바꿔 스크립트를 돌린다. 실제 job(GPU)에는 쓰지 않는다.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest
import yaml
from helpers import FIXTURES, REPO

from robo_jev.launch.jobs import JobError, decisions_artifact_plan, expand, load_job, resolve_job, run_job
from robo_jev.launch.launcher import fetch, launch, prepare_job, status
from robo_jev.launch.manifest import load_manifest, now, sha256_of, validate
from robo_jev.launch.providers.local import LocalBackend

REQUESTS = "tests/fixtures/decisions/requests.jsonl"
WRAPPER = FIXTURES / "launch" / "with_reference_kernels.py"
TRAIN_SCRIPT = REPO / "scripts" / "train_decisions.py"


def base(**job) -> dict:
    data = {"kind": "robojev-cloud-job-v0", "name": "t", "job": {"kind": "decisions", "model": "tiny", "inputs": [REQUESTS],
            "steps": [{"name": "s", "argv": ["true"]}], **job}, "budget": {"hourly_usd": 0.0, "max_wall_hours": 0.5}}  # fmt: skip
    return data


# --------------------------------------------------------------------------
# 정의
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "mutate, needle",
    [
        (lambda d: d.update(kind="other"), "kind"),
        (lambda d: d.update(surprise=1), "모르는 최상위 키"),
        (lambda d: d["job"].update(kind="train"), "job.kind"),
        (lambda d: d["job"].pop("model"), "job.model"),
        (lambda d: d["job"].update(inputs=["/etc/passwd"]), "상대 경로"),
        (lambda d: d["job"].update(inputs=["../outside.jsonl"]), "상대 경로"),
        (lambda d: d["job"].update(steps=[]), "단계가 하나도 없다"),
        (lambda d: d["job"].update(steps=[{"name": "a b", "argv": ["x"]}]), "name"),
        (lambda d: d["job"].update(steps=[{"name": "a", "argv": []}]), "argv"),
        (lambda d: d["job"].update(steps=[{"name": "a", "argv": ["x"]}, {"name": "a", "argv": ["y"]}]), "겹친다"),
        (lambda d: d["job"].update(steps=[{"name": "a", "argv": ["x"], "max_minute": 3}]), "모르는 키"),
    ],
)
def test_a_job_definition_is_checked_before_anything_is_sent(mutate, needle):
    data = base()
    mutate(data)
    with pytest.raises(JobError, match=needle):
        resolve_job(data)


def test_overrides_reach_nested_keys(tmp_path):
    path = tmp_path / "job.yaml"
    path.write_text(yaml.safe_dump(base()), encoding="utf-8")
    job = load_job(path, {"budget.max_usd": 3.5, "job.model": "Qwen/Qwen3.5-2B"})
    assert job["budget"]["max_usd"] == 3.5 and job["job"]["model"] == "Qwen/Qwen3.5-2B"


def test_placeholders_expand_and_a_several_word_python_is_split():
    context = {"python": "env PYTHONPATH=/x /usr/bin/python3", "runs": "/r/runs", "step": "t1", "repo": "/r", "model": "m"}
    argv = expand(["{python}", "-m", "x", "--set", "out_dir={runs}", "--set", "run_name={step}"], context)
    assert argv == ["env", "PYTHONPATH=/x", "/usr/bin/python3", "-m", "x", "--set", "out_dir=/r/runs", "--set", "run_name=t1"]
    with pytest.raises(JobError, match="모르는 자리 표시"):
        expand(["{nope}"], context)


def test_other_braces_are_left_alone_and_unknown_placeholders_are_refused_up_front():
    """`--set lora={r: 4, …}`처럼 YAML 값의 중괄호는 자리 표시가 아니다 — 그대로 간다 (Task v0d 학습기의 실제 인자로 드러난 결함)."""
    context = {"python": "/py", "runs": "/r/runs", "step": "s", "repo": "/r", "model": "m"}
    assert expand(["--set", "lora={r: 4, alpha: 8, targets: [q_proj]}", "--set", "out={runs}"], context) == [
        "--set", "lora={r: 4, alpha: 8, targets: [q_proj]}", "--set", "out=/r/runs"]
    with pytest.raises(JobError, match="모르는 자리 표시"):
        resolve_job(base(steps=[{"name": "a", "argv": ["{pyhton}", "x"]}]))  # 오타는 원격에 보내기 전에 걸린다


def test_the_fetch_plan_always_brings_back_the_state_the_report_and_the_logs():
    plan = decisions_artifact_plan({"fetch": ["runs/*/checkpoint.pt", "reports/*"]})
    remotes = {entry["remote"]: entry for entry in plan}
    assert {"state.json", "job-report.json", "logs/*", "runs/*/checkpoint.pt", "reports/*"} <= set(remotes)
    assert remotes["runs/*/checkpoint.pt"]["kind"] == "checkpoint" and remotes["state.json"]["required_"] is True


# --------------------------------------------------------------------------
# 원격 러너 — 단계·마감·정지
# --------------------------------------------------------------------------


def _bundle(tmp_path: Path, steps: list[dict], *, deadline_epoch: float | None = None, uploads: list | None = None,
            budget: dict | None = None, setup: list | None = None) -> tuple[Path, Path]:  # fmt: skip
    run_dir = tmp_path / "run"
    (run_dir / "bundle").mkdir(parents=True)
    repo = tmp_path / "repo"
    repo.mkdir()
    job = resolve_job(base(steps=steps, setup=setup or []))["job"]
    job["runtime"]["python"] = sys.executable
    job["uploads"] = uploads or []
    if deadline_epoch is not None:
        job["deadline_epoch"] = deadline_epoch
    manifest = {"run_id": "t", "job": job, "budget": {"hourly_usd": 1.0, "gpus": 0, "max_wall_hours": 0.5, "max_gpu_hours": None,
                                                     "max_usd": None, **(budget or {})}}  # fmt: skip
    (run_dir / "bundle" / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return run_dir, repo


def _state(run_dir: Path) -> dict:
    return json.loads((run_dir / "state.json").read_text(encoding="utf-8"))


def test_steps_run_in_order_and_the_runner_writes_its_ledger(tmp_path):
    steps = [
        {"name": "one", "argv": ["{python}", "-c", "open('{reports}/one.txt','w').write('1')"]},
        {"name": "two", "argv": ["{python}", "-c", "import sys; print('two', sys.argv[1:])", "{run_id}", "{model}"]},
    ]
    run_dir, repo = _bundle(tmp_path, steps)
    assert run_job(run_dir / "bundle", repo=repo) == 0
    state = _state(run_dir)
    assert (state["state"], state["exit_reason"], state["step"]) == ("completed", "completed", 2)
    report = json.loads((run_dir / "job-report.json").read_text(encoding="utf-8"))
    assert [(s["name"], s["status"]) for s in report["steps"]] == [("one", "ok"), ("two", "ok")]
    assert (run_dir / "reports" / "one.txt").read_text() == "1" and (run_dir / "started").is_file() and (run_dir / "finished").is_file()
    assert "two ['t', 'tiny']" in (run_dir / "logs" / "01-two.log").read_text(encoding="utf-8")


def test_a_required_step_failure_stops_the_job_and_an_optional_one_does_not(tmp_path):
    steps = [
        {"name": "maybe", "argv": ["{python}", "-c", "raise SystemExit(3)"], "optional": True},
        {"name": "must", "argv": ["{python}", "-c", "raise SystemExit(4)"]},
        {"name": "never", "argv": ["{python}", "-c", "open('never','w')"]},
    ]
    run_dir, repo = _bundle(tmp_path, steps)
    assert run_job(run_dir / "bundle", repo=repo) == 1
    state = _state(run_dir)
    assert (state["state"], state["exit_reason"]) == ("failed", "step:must:failed")
    report = json.loads((run_dir / "job-report.json").read_text(encoding="utf-8"))
    assert [(s["name"], s["status"], s["returncode"]) for s in report["steps"]] == [("maybe", "failed", 3), ("must", "failed", 4), ("never", "skipped", None)]
    assert not (repo / "never").exists()


def test_a_fallback_command_runs_when_the_first_one_fails(tmp_path):
    setup = [{"name": "sync", "argv": ["{python}", "-c", "raise SystemExit(1)"], "fallback": ["{python}", "-c", "open('fallback-ran','w')"]}]
    run_dir, repo = _bundle(tmp_path, [{"name": "s", "argv": ["true"]}], setup=setup)
    assert run_job(run_dir / "bundle", repo=repo) == 0
    report = json.loads((run_dir / "job-report.json").read_text(encoding="utf-8"))
    assert report["steps"][0]["fallback_used"] is True and (repo / "fallback-ran").is_file()


def test_a_step_that_overruns_its_own_limit_is_killed_with_its_children(tmp_path):
    marker = tmp_path / "grandchild-alive"
    code = (
        "import subprocess, sys, time; "
        f"subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(2.5); open(r\"{marker}\", \"w\")']); time.sleep(60)"
    )
    steps = [{"name": "hang", "argv": ["{python}", "-c", code], "max_minutes": 0.02, "optional": True}, {"name": "after", "argv": ["true"]}]
    run_dir, repo = _bundle(tmp_path, steps)
    import robo_jev.launch.jobs as jobs_module

    original = jobs_module.KILL_GRACE_SECONDS
    jobs_module.KILL_GRACE_SECONDS = 1.0
    try:
        started = time.monotonic()
        assert run_job(run_dir / "bundle", repo=repo) == 0
    finally:
        jobs_module.KILL_GRACE_SECONDS = original
    assert time.monotonic() - started < 20
    report = json.loads((run_dir / "job-report.json").read_text(encoding="utf-8"))
    assert report["steps"][0]["status"] == "timeout" and report["steps"][1]["status"] == "ok"
    time.sleep(3.0)
    assert not marker.exists(), "단계의 손자 프로세스까지 묶음째 죽었다"


def test_the_job_stops_at_its_deadline_and_says_budget(tmp_path):
    steps = [{"name": "first", "argv": ["{python}", "-c", "import time; time.sleep(30)"], "max_minutes": 5}, {"name": "second", "argv": ["true"]}]
    run_dir, repo = _bundle(tmp_path, steps, deadline_epoch=time.time() + 25.0)
    import robo_jev.launch.jobs as jobs_module

    original = jobs_module.KILL_GRACE_SECONDS
    jobs_module.KILL_GRACE_SECONDS = 1.0
    try:
        code = run_job(run_dir / "bundle", repo=repo)
    finally:
        jobs_module.KILL_GRACE_SECONDS = original
    state = _state(run_dir)
    assert code == 2 and (state["state"], state["exit_reason"], state["budget_limit"]) == ("failed", "budget", "pod-lifetime")
    report = json.loads((run_dir / "job-report.json").read_text(encoding="utf-8"))
    assert [s["status"] for s in report["steps"]] == ["timeout", "skipped"]
    assert state["elapsed_seconds"] < 40


def test_a_step_that_cannot_start_before_the_deadline_is_not_started(tmp_path):
    run_dir, repo = _bundle(tmp_path, [{"name": "late", "argv": ["{python}", "-c", "open('ran','w')"]}], deadline_epoch=time.time() + 5.0)
    assert run_job(run_dir / "bundle", repo=repo) == 2
    assert _state(run_dir)["exit_reason"] == "budget" and not (repo / "ran").exists()


def test_a_train_step_is_told_its_wall_clock_so_the_trainer_stops_itself(tmp_path):
    steps = [{"name": "train", "argv": ["{python}", "-c", "import sys; open('argv.json','w').write(__import__('json').dumps(sys.argv))"],
              "wall_set": "max_wall_hours", "max_minutes": 10}]  # fmt: skip
    run_dir, repo = _bundle(tmp_path, steps)
    assert run_job(run_dir / "bundle", repo=repo) == 0
    argv = json.loads((repo / "argv.json").read_text(encoding="utf-8"))
    assert argv[-2] == "--set" and argv[-1].startswith("max_wall_hours=")
    assert float(argv[-1].split("=", 1)[1]) == pytest.approx(10 / 60 * 0.9, rel=1e-3), "준 시간의 90 %"


def test_children_do_not_inherit_the_pod_api_key(tmp_path, monkeypatch):
    monkeypatch.setenv("RUNPOD_API_KEY", "pod-scoped-should-not-leak")
    steps = [{"name": "env", "argv": ["{python}", "-c", "import os; open('env.txt','w').write(os.environ.get('RUNPOD_API_KEY', 'absent'))"]}]
    run_dir, repo = _bundle(tmp_path, steps)
    assert run_job(run_dir / "bundle", repo=repo) == 0
    assert (repo / "env.txt").read_text() == "absent"


def test_a_runner_failure_outside_any_step_is_written_down_not_left_running(tmp_path):
    """러너 자신이 죽으면 `state.json`이 `running`으로 남는다 — 실행기는 마감까지 기다리고, pod에서는 그것이 돈이다."""
    run_dir, repo = _bundle(tmp_path, [{"name": "s", "argv": ["true"]}])
    manifest = json.loads((run_dir / "bundle" / "manifest.json").read_text(encoding="utf-8"))
    manifest["job"]["steps"][0]["argv"] = ["{not_a_placeholder}"]  # 정의 검사를 건너뛴 묶음 (손으로 고친 명세)
    (run_dir / "bundle" / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    assert run_job(run_dir / "bundle", repo=repo) == 1
    state = _state(run_dir)
    assert (state["state"], state["exit_reason"]) == ("failed", "error") and "JobError" in state["error"]
    assert (run_dir / "finished").is_file(), "끝 표지가 있어야 pod 안의 감시가 '끝난 뒤 유예'를 센다"


def test_inputs_whose_hash_moved_stop_the_job_before_any_step(tmp_path):
    run_dir, repo = _bundle(tmp_path, [{"name": "s", "argv": ["{python}", "-c", "open('ran','w')"]}],
                            uploads=[{"path": "data.jsonl", "sha256": "0" * 64, "bytes": 3}])  # fmt: skip
    (repo / "data.jsonl").write_text("abc", encoding="utf-8")
    assert run_job(run_dir / "bundle", repo=repo) == 4
    state = _state(run_dir)
    assert (state["state"], state["exit_reason"]) == ("failed", "inputs") and "sha256" in state["error"]
    assert not (repo / "ran").exists()


def test_a_term_signal_stops_the_current_step_and_ends_cancelled(tmp_path):
    steps = [{"name": "long", "argv": ["{python}", "-c", "import time; time.sleep(60)"], "max_minutes": 5}, {"name": "after", "argv": ["true"]}]
    run_dir, repo = _bundle(tmp_path, steps)
    process = subprocess.Popen([sys.executable, "-m", "robo_jev.launch.runner", "--bundle", str(run_dir / "bundle"), "--repo", str(repo)],
                               cwd=str(REPO), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env={**os.environ, "PYTHONPATH": str(REPO / "src")})  # fmt: skip
    try:
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            if (run_dir / "state.json").is_file() and _state(run_dir).get("current") == "long":
                break
            time.sleep(0.1)
        process.send_signal(signal.SIGTERM)
        assert process.wait(timeout=60) == 3
    finally:
        if process.poll() is None:  # pragma: no cover
            process.kill()
            process.wait()
    state = _state(run_dir)
    assert (state["state"], state["exit_reason"]) == ("cancelled", "cancelled")
    assert [s["status"] for s in json.loads((run_dir / "job-report.json").read_text())["steps"]] == ["cancelled", "skipped"]


def test_the_runner_job_mode_needs_neither_torch_nor_yaml():
    """원격 러너는 venv가 생기기 **전에** 시스템 파이썬으로 뜬다 — job 경로가 torch·yaml을 부르면 거기서 죽는다."""
    code = (
        "import sys, builtins; real = builtins.__import__\n"
        "def guard(name, *a, **k):\n"
        "    if name.split('.')[0] in ('torch', 'yaml', 'numpy'): raise ImportError('blocked ' + name)\n"
        "    return real(name, *a, **k)\n"
        "builtins.__import__ = guard\n"
        "import robo_jev.launch.runner, robo_jev.launch.jobs; print('ok')"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=False, env={**os.environ, "PYTHONPATH": str(REPO / "src")})
    assert result.returncode == 0 and result.stdout.strip() == "ok", result.stderr


# --------------------------------------------------------------------------
# local 백엔드로 끝까지
# --------------------------------------------------------------------------

TINY_ENV = {"CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2"}


def _local_job(tmp_path: Path, steps: list[dict], fetch_patterns: list[str]) -> Path:
    data = {
        "kind": "robojev-cloud-job-v0", "name": "cloud1-local",
        "job": {"kind": "decisions", "model": "tiny", "inputs": [REQUESTS],
                "runtime": {"python": f"{sys.executable} {WRAPPER}", "env": {**TINY_ENV, "PYTHONPATH": str(REPO / "src")}},
                "steps": steps, "fetch": fetch_patterns},
        "budget": {"hourly_usd": 0.0, "max_wall_hours": 0.5},
    }  # fmt: skip
    path = tmp_path / "job.yaml"
    path.write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")
    return path


def _round_trip(tmp_path: Path, job_path: Path, run_id: str, timeout: float = 600) -> dict:
    backend = LocalBackend(remote_dir=str(tmp_path / "remote" / run_id), remote_repo=str(REPO), remote_python=sys.executable, gpus=0)
    manifest_path = tmp_path / f"{run_id}.json"
    manifest = prepare_job(job_path=job_path, run_id=run_id, manifest_path=manifest_path, backend=backend)
    assert validate(manifest) == []
    manifest = launch(manifest, backend, manifest_path=manifest_path)
    deadline = time.monotonic() + timeout
    while manifest["state"] == "running" and time.monotonic() < deadline:
        time.sleep(1.0)
        manifest = status(manifest, backend, manifest_path=manifest_path)
    assert manifest["state"] == "completed", (manifest["history"], (Path(backend.remote_dir) / "logs").glob("*"))
    return fetch(manifest, backend, dest=tmp_path / "fetched", manifest_path=manifest_path)


def test_a_tiny_decisions_evaluation_job_runs_end_to_end_on_the_local_backend(tmp_path):
    """학습 없는 소형 모델의 진단을 job 단계로 — 보고서·예측·로그·상태가 sha256 대조를 거쳐 돌아온다."""
    steps = [{"name": "diagnose-base", "argv": ["{python}", "scripts/diagnose_decisions.py", "--model", "tiny", "--input", REQUESTS,
                                                "--out", "{reports}/base.json", "--n-boot", "10", "--batch", "8", "--orders", "seeded"],
              "max_minutes": 8}]  # fmt: skip
    run_id = f"cloud1-eval-{os.getpid()}"
    manifest = _round_trip(tmp_path, _local_job(tmp_path, steps, ["reports/*"]), run_id)
    got = {Path(e["remote"]).name: e for e in manifest["artifacts"] if e.get("local")}
    assert {"base.json", "base.predictions.jsonl", "base.md", "state.json", "job-report.json"} <= set(got)
    assert all(entry["sha256_match"] for entry in got.values())
    summary = json.loads(Path(got["base.json"]["local"]).read_text(encoding="utf-8"))
    assert summary["config"]["model"] == "tiny" and summary["config"]["records"] == 8
    report = json.loads(Path(got["job-report.json"]["local"]).read_text(encoding="utf-8"))
    assert report["state"] == "completed" and report["steps"][0]["status"] == "ok"
    assert validate(load_manifest(tmp_path / f"{run_id}.json")) == []


@pytest.mark.skipif(not TRAIN_SCRIPT.is_file(), reason="Task v0d의 판단 학습기(scripts/train_decisions.py)가 아직 이 브랜치에 병합되지 않았다")
def test_a_tiny_decisions_training_job_trains_evaluates_and_brings_back_the_adapter(tmp_path):
    tiny = ["--set", "model_id=tiny", "--set", "device=cpu", "--set", "dtype=float32", "--set", "min_free_disk_gb=0", "--set", "batch_examples=4",
            "--set", "lora={r: 4, alpha: 8, dropout: 0.0, targets: [q_proj, v_proj, in_proj_qkv, down_proj]}", "--set", "lr=0.01"]  # fmt: skip
    steps = [
        {"name": "train", "wall_set": "max_wall_hours", "max_minutes": 10,
         "argv": ["{python}", "scripts/train_decisions.py", "train", *tiny, "--set", f"train_input={REQUESTS}", "--set", "out_dir={runs}",
                  "--set", "run_name=tiny", "--set", "dev_fraction=0.3", "--set", "dev_every=2", "--set", "max_steps=3"]},
        {"name": "diagnose-after", "max_minutes": 8,
         "argv": ["{python}", "scripts/diagnose_decisions.py", "--model", "tiny", "--checkpoint", "{runs}/tiny/checkpoint.pt", "--input", REQUESTS,
                  "--out", "{reports}/after.json", "--n-boot", "10", "--batch", "8", "--orders", "seeded"]},
    ]  # fmt: skip
    run_id = f"cloud1-train-{os.getpid()}"
    manifest = _round_trip(tmp_path, _local_job(tmp_path, steps, ["runs/*/checkpoint.pt", "runs/*/metrics.json", "runs/*/config.yaml", "reports/*"]), run_id)
    got = {e["remote"]: e for e in manifest["artifacts"] if e.get("local")}
    assert {"runs/tiny/checkpoint.pt", "runs/tiny/metrics.json", "reports/after.json"} <= set(got)
    assert all(entry["sha256_match"] for entry in got.values())
    metrics = json.loads(Path(got["runs/tiny/metrics.json"]["local"]).read_text(encoding="utf-8"))
    assert metrics["status"] == "completed" and metrics["step"] == 3
    assert metrics["config"]["max_wall_hours"] is not None, "러너가 준 벽시계 상한이 학습기에 닿았다"
    after = json.loads(Path(got["reports/after.json"]["local"]).read_text(encoding="utf-8"))
    assert after["config"]["checkpoint"]["run_id"] == "tiny"


# --------------------------------------------------------------------------
# 실어 보내는 job 정의가 병합된 CLI와 맞는가 (수정 1: v0 본 run 둘)
# --------------------------------------------------------------------------

SHIPPED_JOBS = sorted((REPO / "configs" / "cloud").glob("*.yaml"))


def _script(name: str):
    import functools
    import importlib.util

    @functools.lru_cache(maxsize=None)
    def load(path: str):
        spec = importlib.util.spec_from_file_location(f"shipped_{Path(path).stem}", path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module

    return load(str(REPO / "scripts" / name))


@pytest.mark.parametrize("job_path", SHIPPED_JOBS, ids=[p.stem for p in SHIPPED_JOBS])
def test_every_shipped_step_parses_with_the_merged_tools_and_refers_only_to_shipped_inputs(job_path):
    """configs/cloud/*.yaml의 단계마다: 병합된 학습기·진단기·가중치 받기의 **자기 argparse**로 인자를 읽고, 학습 설정은 학습기의 resolve_config로
    풀어 본다(모르는 키·범위 밖 값이면 여기서 걸린다). 학습·진단의 입력 파일은 job의 inputs에 있어야 한다(올라가지 않으면 원격에 없다).
    모델을 싣지도 데이터를 읽지도 않는다."""
    import json as json_module

    from robo_jev.decisions.train import resolve_config

    spec = load_job(job_path)
    job = spec["job"]
    inputs = set(job["inputs"])
    models = json_module.loads((REPO / "artifacts" / "models" / "manifest.json").read_text(encoding="utf-8"))["models"]
    candidates = yaml.safe_load((REPO / "configs" / "model" / "candidates.yaml").read_text(encoding="utf-8"))
    candidate_ids = {entry["id"] for entry in candidates["candidates"]}
    trained: set[str] = set()
    for step in job["setup"] + job["steps"]:
        argv = expand(step["argv"], {"python": "python", "uv": "uv", "repo": "/r", "run_dir": "/r/run", "runs": "/r/run/runs",
                                     "reports": "/r/run/reports", "model": job["model"], "step": step["name"], "run_id": "x"})  # fmt: skip
        if argv[0] == "uv" or argv[1:3] == ["-m", "robo_jev.launch.envprobe"]:
            continue
        script = argv[1]
        if script == "scripts/train_decisions.py":
            args = _script("train_decisions.py").build_parser().parse_args(argv[2:])
            assert args.command == "train", step["name"]
            config = _script("train_decisions.py")._read_config(REPO / args.config, args.set)
            resolved = resolve_config(config)
            assert resolved["train_input"] in inputs, f"{step['name']}: 학습 입력 {resolved['train_input']}이 job inputs에 없다"
            assert resolved["out_dir"] == "/r/run/runs", f"{step['name']}: out_dir는 {{runs}}여야 가져온다"
            assert step.get("wall_set") == "max_wall_hours", f"{step['name']}: 학습 단계는 러너의 벽시계를 받아야 스스로 멈춘다"
            trained.add(resolved["run_name"])
            assert resolved["model_id"] in models, f"{step['name']}: {resolved['model_id']}의 고정 revision이 모델 manifest에 없다"
        elif script == "scripts/diagnose_decisions.py":
            args = _script("diagnose_decisions.py")._parser().parse_args(argv[2:])
            assert str(args.input) in inputs, f"{step['name']}: 진단 입력 {args.input}이 job inputs에 없다"
            assert args.model in models, f"{step['name']}: {args.model}의 고정 revision이 모델 manifest에 없다"
            if args.checkpoint is not None:
                run_name = Path(args.checkpoint).parent.name
                produced = str(args.checkpoint).startswith("/r/run/runs/") and run_name in trained
                uploaded = str(args.checkpoint) in inputs  # 올려 보낸 체크포인트 — 예: v0.1 main의 같은 하드웨어 기준(v0 adapter, Task v0.1b)
                assert produced or uploaded, f"{step['name']}: 앞 단계가 만든 체크포인트도, job inputs로 올려 보낸 체크포인트도 아니다"
            assert not args.preview or "simgen/v0b/tests" not in str(args.input), f"{step['name']}: 봉인 시험 세트에 '미리보기' 머리를 달지 않는다"
        elif script == "scripts/fetch_backbone.py":
            assert argv[2:4] == ["--from-manifest", "artifacts/models/manifest.json"] and argv[4] == "--id", argv
            assert argv[5] in models and argv[5] in candidate_ids, f"{step['name']}: {argv[5]}은 manifest·후보 목록에 있어야 한다"
        else:  # pragma: no cover - 새 도구가 생기면 여기서 알린다
            pytest.fail(f"{job_path.name}:{step['name']}: 모르는 도구 {script}")
    assert set(job["fetch"]) >= {"reports/*"}, "보고서는 언제나 가져온다"
