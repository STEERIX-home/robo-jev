"""RunPod 공급자·단계 장부·pod 수명 — 돈 안전 장치 각각을 가짜 API로 (Task cloud1).

네트워크는 쓰지 않는다: `RunPodClient`에 가짜 전송(`tests/runpod_fake.py`)을 넣고, "pod"는 이 상자의 임시 디렉터리다(`LocalBackend` + 진짜
컨테이너 시작 명령을 `sh`로 돌려 감시를 띄운다). 그래서 실행기의 실제 흐름 — 단가 대조 → 장부 예약 → 만들기 → 준비 → 감시 확인 → 올리기
(sha256) → 러너 → 지켜보기 → 가져오기(sha256) → **지우기와 확인** → 장부 닫기 — 가 그대로 돈다.

돈 안전 장치 (브리프 §2.2)와 이 파일의 검사:

* (i) 상한: `test_the_pod_deadline_*`, 러너의 마감 멈춤은 `tests/test_cloud_jobs.py`
* (ii) 끝·실패 뒤 반드시 지움: `test_a_full_pod_round_trip_*`, `test_a_failed_fetch_still_deletes_*`, `test_a_failing_job_*`,
  `test_a_pod_without_an_armed_watchdog_*`, `test_a_boot_that_never_becomes_ready_*`, `test_a_delete_that_cannot_be_confirmed_*`
* (iii) 실행기가 죽어도: 감시 자체는 `tests/test_cloud_watchdog.py`; 여기서는 만들기 본문에 감시가 실리는 것, 중단 경로
* (iv) 비용 출력 + `--yes`: `test_launch_without_yes_*`, CLI 검사
* (v) 키: `test_the_key_*`, `test_a_key_file_*`, `test_a_missing_key_*`
* 단가 불일치: `test_a_catalog_price_*`, `test_a_created_pod_billed_*` · 단계 장부 $150: `test_the_phase_*`
"""

from __future__ import annotations

import base64
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest
import yaml
from helpers import REPO
from runpod_fake import CATALOG, FakeRunPodApi

from robo_jev.launch import cloud as cloud_module
from robo_jev.launch.cloud import LaunchNotApproved, PodContext, down, finish, launch_pod, pod_lifetime, run_cap_usd, start_script, watchdog_script
from robo_jev.launch.jobs import format_plan, plan_job
from robo_jev.launch.launcher import prepare_job
from robo_jev.launch.ledger import DEFAULT_PHASE, Ledger, LedgerError, PhaseCapExceeded
from robo_jev.launch.manifest import Budget, load_manifest, validate
from robo_jev.launch.providers.local import LocalBackend
from robo_jev.launch.providers.runpod import (
    MissingApiKey,
    PriceMismatch,
    ReadOnlyViolation,
    RunPodClient,
    RunPodError,
    RunPodProvider,
    Secret,
    UnsafeKeyFile,
    load_api_key,
    pod_request,
)

KEY = "rpa_TESTKEY_0123456789abcdef"
REQUESTS = "tests/fixtures/decisions/requests.jsonl"


# --------------------------------------------------------------------------
# (v) 키
# --------------------------------------------------------------------------


@pytest.mark.parametrize("text", [f"RUNPOD_API_KEY={KEY}\n", f"{KEY}\n", f'export RUNPOD_API_KEY="{KEY}"\n', f"# 주석\n\nRUNPOD_API_KEY='{KEY}'\n"])
def test_the_key_is_read_from_the_file_in_either_shape(tmp_path, text):
    path = tmp_path / "runpod.env"
    path.write_text(text, encoding="utf-8")
    path.chmod(0o600)
    assert load_api_key(path).reveal() == KEY


@pytest.mark.parametrize("mode", [0o640, 0o604, 0o644])
def test_a_key_file_readable_by_group_or_others_is_refused(tmp_path, mode):
    path = tmp_path / "runpod.env"
    path.write_text(f"RUNPOD_API_KEY={KEY}\n", encoding="utf-8")
    path.chmod(mode)
    with pytest.raises(UnsafeKeyFile, match="chmod 600"):
        load_api_key(path)


def test_a_missing_key_file_is_refused_with_how_to_make_it(tmp_path):
    with pytest.raises(MissingApiKey, match="RUNPOD_API_KEY=<키>"):
        load_api_key(tmp_path / "nope.env")
    empty = tmp_path / "empty.env"
    empty.write_text("\n# 없음\n", encoding="utf-8")
    empty.chmod(0o600)
    with pytest.raises(MissingApiKey, match="비었다"):
        load_api_key(empty)


def test_the_key_is_not_taken_from_the_environment(tmp_path, monkeypatch):
    """환경 변수에 키가 있어도 파일이 없으면 거절한다 — 키의 출처는 파일 하나다."""
    monkeypatch.setenv("RUNPOD_API_KEY", KEY)
    with pytest.raises(MissingApiKey):
        load_api_key(tmp_path / "nope.env")


def test_the_key_never_shows_in_repr_str_format_or_pickle():
    secret = Secret(KEY)
    assert KEY not in repr(secret) and KEY not in str(secret) and KEY not in f"{secret}" and KEY not in f"{secret!r}"
    with pytest.raises(TypeError):
        import pickle

        pickle.dumps(secret)


def test_the_key_travels_only_in_the_authorization_header():
    api = FakeRunPodApi()
    client = RunPodClient(Secret(KEY), transport=api)
    client.gpu_types()
    client.list_pods()
    for request in api.requests:
        assert request["headers"]["Authorization"] == f"Bearer {KEY}"
        assert KEY not in request["url"], "키를 URL에 싣지 않는다 (GraphQL의 ?api_key= 방식 금지)"
        assert KEY not in json.dumps(request["body"])


def test_every_request_names_the_tool_in_its_user_agent():
    """urllib의 기본 User-Agent(`Python-urllib/3.x`)는 api.runpod.io 앞의 Cloudflare가 Error 1010으로 막는다 (2026-10-11 실측 403)."""
    api = FakeRunPodApi()
    RunPodClient(Secret(KEY), transport=api).gpu_types()
    agent = api.requests[0]["headers"]["User-Agent"]
    assert agent.startswith("robo-jev-launcher/") and "Python-urllib" not in agent


def test_an_api_error_that_echoes_the_key_is_redacted():
    def echo(method, url, headers, body, timeout):
        return 401, {}, json.dumps({"title": "Unauthorized", "status": 401, "detail": f"bad token {headers['Authorization']}"}).encode()

    client = RunPodClient(Secret(KEY), transport=echo)
    with pytest.raises(RunPodError) as caught:
        client.list_pods()
    assert KEY not in str(caught.value) and "***" in str(caught.value)
    assert KEY not in repr(client)


# --------------------------------------------------------------------------
# 클라이언트: 읽기 전용·재시도
# --------------------------------------------------------------------------


def test_a_read_only_client_refuses_writes_before_sending_anything():
    api = FakeRunPodApi()
    client = RunPodClient(Secret(KEY), transport=api, read_only=True)
    with pytest.raises(ReadOnlyViolation):
        client.create_pod({"name": "x"})
    with pytest.raises(ReadOnlyViolation):
        client.delete_pod("abc")
    assert api.requests == [], "거절은 보내기 **전에** 일어난다"
    client.gpu_types()
    assert [r["method"] for r in api.requests] == ["GET"]


def test_create_is_never_retried_but_reads_and_deletes_are():
    """응답 없는 POST를 다시 보내면 pod가 둘이다 — 만들기는 한 번만 보낸다."""
    attempts = {"POST": 0, "GET": 0}

    def flaky(method, url, headers, body, timeout):
        attempts[method] = attempts.get(method, 0) + 1
        raise TimeoutError("read timed out")

    client = RunPodClient(Secret(KEY), transport=flaky, sleep=lambda _: None, max_retries=3)
    with pytest.raises(RunPodError) as caught:
        client.create_pod({"name": "x"})
    assert attempts["POST"] == 1 and caught.value.ambiguous, "POST의 전송 실패는 '보냈는지 모른다'다"
    with pytest.raises(RunPodError) as caught:
        client.list_pods()
    assert attempts["GET"] == 4 and not caught.value.ambiguous


def test_a_429_is_waited_out_even_for_create():
    calls = []

    def limited(method, url, headers, body, timeout):
        calls.append(method)
        if len(calls) == 1:
            return 429, {"Retry-After": "2"}, b'{"title":"Too Many","status":429,"detail":"slow down"}'
        return 201, {}, b'{"id": "p1", "status": "PROVISIONING", "cost": 2.69}'

    slept = []
    client = RunPodClient(Secret(KEY), transport=limited, sleep=slept.append)
    assert client.create_pod({"name": "x"})["id"] == "p1"
    assert calls == ["POST", "POST"] and slept == [2.0]


# --------------------------------------------------------------------------
# 만들기 본문 — GPU·클라우드·이미지·볼륨·지역은 명세에서
# --------------------------------------------------------------------------


def test_the_pod_request_carries_every_choice_from_the_manifest():
    spec = {"gpu_type": "NVIDIA H100 PCIe", "gpu_count": 2, "cloud": "community", "image": "runpod/pytorch:x", "disk_gb": 120, "volume_gb": 50,
            "volume_path": "/workspace", "data_centers": ["US-TX-3"], "min_cuda_version": "13.0", "min_ram_per_gpu": 64}  # fmt: skip
    body = pod_request(spec, name="run-1", env={"A": "1"}, start_script="echo hi")
    assert body["gpu"] == {"id": "NVIDIA H100 PCIe", "count": 2, "minCudaVersion": "13.0", "minRamPerGpu": 64}
    assert (body["cloud"], body["image"], body["disk"], body["dataCenterIds"]) == ("COMMUNITY", "runpod/pytorch:x", 120, ["US-TX-3"])
    assert body["mounts"] == {"persistent": {"size": 50, "path": "/workspace"}}
    assert body["ports"] == ["22/tcp"] and body["env"] == {"A": "1"} and body["startSsh"] is True
    assert body["entrypoint"] == ["/bin/sh", "-c"] and body["cmd"] == ["echo hi"]
    network = pod_request({**spec, "network_volume_id": "vol-1"}, name="run-1", env={}, start_script=None)
    assert network["mounts"] == {"network": [{"volumeId": "vol-1", "path": "/workspace"}]} and "cmd" not in network


# --------------------------------------------------------------------------
# 단가 대조
# --------------------------------------------------------------------------


def _provider(api: FakeRunPodApi) -> RunPodProvider:
    return RunPodProvider(RunPodClient(Secret(KEY), transport=api, sleep=lambda _: None), sleep=lambda _: None)


def test_a_catalog_price_that_differs_from_the_manifest_refuses_before_creating():
    api = FakeRunPodApi()
    provider = _provider(api)
    spec = {"gpu_type": "NVIDIA H100 80GB HBM3", "gpu_count": 1, "cloud": "SECURE"}
    assert provider.check_price(spec, 2.69)["catalog_hourly_usd"] == pytest.approx(2.69)
    with pytest.raises(PriceMismatch, match="카탈로그 단가"):
        provider.check_price(spec, 2.39)
    with pytest.raises(PriceMismatch):
        provider.check_price({**spec, "gpu_count": 2}, 2.69)  # 노드 단가 = GPU 단가 × 수
    assert api.calls("POST") == [], "단가가 다르면 아무것도 만들지 않는다"


def test_a_gpu_the_cloud_does_not_offer_is_refused():
    provider = _provider(FakeRunPodApi())
    with pytest.raises(RunPodError, match="COMMUNITY 클라우드"):
        provider.check_price({"gpu_type": "NVIDIA H200", "gpu_count": 1, "cloud": "COMMUNITY"}, 0.0)
    with pytest.raises(RunPodError, match="카탈로그"):
        provider.check_price({"gpu_type": "NVIDIA Imaginary", "gpu_count": 1, "cloud": "SECURE"}, 1.0)


# --------------------------------------------------------------------------
# 단계 장부 — $150
# --------------------------------------------------------------------------


def test_the_default_phase_is_150_dollars_and_the_shipped_configs_agree():
    assert DEFAULT_PHASE["cap_usd"] == 150.0
    for path in sorted((REPO / "configs" / "cloud").glob("*.yaml")):
        phase = (yaml.safe_load(path.read_text(encoding="utf-8")).get("budget") or {}).get("phase")
        assert phase == {"name": "cloud-phase-1", "cap_usd": 150}, f"{path.name}: 단계 상한은 $150"


def test_the_phase_cap_refuses_a_run_that_would_cross_it_and_writes_nothing(tmp_path):
    ledger = Ledger(tmp_path / "ledger.json")
    ledger.reserve("r1", cap_usd=100.0, hourly_usd=2.0, manifest=None)
    before = (tmp_path / "ledger.json").read_text(encoding="utf-8")
    with pytest.raises(PhaseCapExceeded, match="남은 것은 \\$50.00"):
        ledger.reserve("r2", cap_usd=60.0, hourly_usd=2.0, manifest=None)
    assert (tmp_path / "ledger.json").read_text(encoding="utf-8") == before
    ledger.reserve("r2", cap_usd=50.0, hourly_usd=2.0, manifest=None)  # 딱 맞으면 된다
    with pytest.raises(PhaseCapExceeded):
        ledger.check("r3", 0.01)


def test_open_runs_count_their_whole_cap_and_closed_runs_what_they_spent(tmp_path):
    ledger = Ledger(tmp_path / "ledger.json")
    ledger.reserve("r1", cap_usd=40.0, hourly_usd=2.0, manifest=None)
    ledger.activate("r1", pod_id="p1")
    assert ledger.committed_usd() == 40.0, "지웠는지 모르는 pod는 상한까지 썼다고 본다"
    ledger.close("r1", spent_usd_estimate=3.25)
    assert ledger.committed_usd() == 3.25
    ledger.record_billed("r1", billed_usd=3.5, source="billing")
    assert ledger.committed_usd() == 3.5, "청구와 추정 중 큰 쪽"
    ledger.reserve("r2", cap_usd=10.0, hourly_usd=2.0, manifest=None)
    ledger.release("r2", reason="단가 불일치")
    assert ledger.committed_usd() == 3.5, "만들지 않은 run은 세지 않는다"
    ledger.reserve("r3", cap_usd=5.0, hourly_usd=2.0, manifest=None)
    ledger.activate("r3", pod_id="p3")
    ledger.mark_unconfirmed("r3", note="DELETE 실패")
    assert ledger.committed_usd() == 8.5 and ledger.summary()["open_runs"] == ["r3"]


def test_a_run_id_is_used_once_and_the_phase_must_match(tmp_path):
    ledger = Ledger(tmp_path / "ledger.json")
    ledger.reserve("r1", cap_usd=1.0, hourly_usd=2.0, manifest=None)
    with pytest.raises(LedgerError, match="이미 장부에"):
        ledger.reserve("r1", cap_usd=1.0, hourly_usd=2.0, manifest=None)
    with pytest.raises(LedgerError, match="단계"):
        ledger.check("r2", 1.0, phase={"name": "cloud-phase-1", "cap_usd": 300})


def test_raising_the_phase_cap_needs_a_reason_and_is_written_down(tmp_path):
    ledger = Ledger(tmp_path / "ledger.json")
    with pytest.raises(LedgerError, match="이유"):
        ledger.set_phase_cap(300.0, reason=" ")
    ledger.set_phase_cap(300.0, reason="사용자 재승인 2026-10-12")
    data = ledger.read()
    assert data["phase"]["cap_usd"] == 300.0 and "사용자 재승인" in data["history"][-1]["note"]


# --------------------------------------------------------------------------
# pod 수명 — 가짜 API + 임시 디렉터리 "pod"
# --------------------------------------------------------------------------

FAKE_BOOTSTRAP = """R={root}
mkdir -p "$R/run/logs" "$R/repo"
echo "UV=uv"
echo "PY311=%s"
echo "NOW=$(date +%%s)"
echo "GPU=fake H100"
if [ -f "$R/watchdog.pid" ] && kill -0 "$(cat "$R/watchdog.pid")" 2>/dev/null; then echo "WATCHDOG=armed"; else echo "WATCHDOG=missing"; fi
""" % sys.executable


def write_job(tmp_path: Path, *, steps: list | None = None, budget: dict | None = None, provider: dict | None = None, fetch: list | None = None) -> Path:
    root = tmp_path / "pod"
    data = {
        "kind": "robojev-cloud-job-v0",
        "name": "cloud1-test",
        "job": {
            "kind": "decisions", "model": "tiny", "inputs": [REQUESTS], "runtime": {"python": sys.executable},
            "steps": steps or [{"name": "hello", "argv": ["{python}", "-c", "import pathlib,sys; pathlib.Path(sys.argv[1]).write_text('hi')", "{reports}/hello.txt"],
                                "max_minutes": 2, "est_minutes": 0.1}],
            "fetch": fetch if fetch is not None else ["reports/*"],
        },
        "provider": {"name": "runpod", "gpu_type": "NVIDIA H100 80GB HBM3", "gpu_count": 1, "cloud": "SECURE", "image": "runpod/pytorch:test",
                     "disk_gb": 60, "boot_timeout_minutes": 0.2, "bootstrap": FAKE_BOOTSTRAP, "remote_root": str(root),
                     "ssh_key": str(tmp_path / "keys" / "id_ed25519"), **(provider or {})},
        "budget": {"hourly_usd": 2.69, "max_wall_hours": 0.5, "max_usd": 5, "phase": {"name": "cloud-phase-1", "cap_usd": 150}, **(budget or {})},
        "estimate": {"boot_minutes": 3, "assumption": "검사"},
    }  # fmt: skip
    path = tmp_path / "job.yaml"
    path.write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")
    return path


class Container:
    """"pod"의 컨테이너 — 가짜 API에 실린 env로 **진짜 시작 명령**을 sh로 돌린다 (감시가 뜨고 pid 파일이 생긴다)."""

    def __init__(self, tmp_path: Path) -> None:
        self.tmp = tmp_path
        self.process: subprocess.Popen | None = None
        self.calls = tmp_path / "runpodctl.log"

    def boot(self, pod_env: dict[str, str], pod_id: str) -> None:
        if self.process is not None:
            return
        bin_dir = self.tmp / "podbin"
        bin_dir.mkdir(exist_ok=True)
        fake = bin_dir / "runpodctl"
        fake.write_text(f'#!/bin/sh\necho "runpodctl $*" >> "{self.calls}"\nexit 0\n', encoding="utf-8")
        fake.chmod(0o755)
        env = {"PATH": f"{bin_dir}:/usr/bin:/bin", "HOME": str(self.tmp), "RUNPOD_POD_ID": pod_id, "RUNPOD_API_KEY": "pod-scoped", **pod_env}
        self.process = subprocess.Popen(["/bin/sh", "-c", start_script()], env=env, start_new_session=True,
                                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)  # fmt: skip
        root = Path(pod_env["ROBOJEV_ROOT"])
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline and not (root / "watchdog.pid").is_file():
            time.sleep(0.05)

    def stop(self) -> None:
        if self.process is not None:
            os.killpg(self.process.pid, signal.SIGKILL)
            self.process.wait()


class PinnedLocal(LocalBackend):
    """원격 파이썬을 이 worktree의 소스로 고정한 local 백엔드 — 커밋 전의 코드로도 러너가 돈다(묶음 풀기·대조는 그대로 시험된다)."""

    @property
    def remote_python(self) -> str:
        return f"env PYTHONPATH={REPO / 'src'} CUDA_VISIBLE_DEVICES= {sys.executable}"

    @remote_python.setter
    def remote_python(self, value: str) -> None:  # 부트스트랩이 정한 값은 받지 않는다
        pass


@pytest.fixture
def pod_world(tmp_path):
    api = FakeRunPodApi()
    container = Container(tmp_path)
    ledger = Ledger(tmp_path / "ledger.json")
    lines: list[str] = []

    def factory(view, manifest):
        pod = api.pods[view.pod_id]
        container.boot(pod["env"], view.pod_id)
        root = manifest["provider"]["remote_root"]
        return PinnedLocal(remote_dir=f"{root}/run", remote_repo=f"{root}/repo", remote_python="python3", gpus=1)

    ctx = PodContext(provider=_provider(api), ledger=ledger, backend_factory=factory, out=lines.append, poll_seconds=0.3)
    yield {"api": api, "ctx": ctx, "ledger": ledger, "container": container, "lines": lines, "tmp": tmp_path}
    container.stop()
    for pod in api.pods.values():  # "pod"를 지워도 이 상자의 러너 유닛은 남는다 — 진짜 pod라면 함께 사라질 것을 여기서 멈춘다
        subprocess.run(["systemctl", "--user", "stop", pod["name"]], capture_output=True, check=False)


def _prepared(world, run_id: str, **job_kwargs):
    tmp = world["tmp"]
    job = write_job(tmp, **job_kwargs)
    manifest_path = tmp / "runs" / f"{run_id}.json"
    manifest = prepare_job(job_path=job, run_id=run_id, manifest_path=manifest_path, allow_dirty=True, ledger=world["ledger"])
    return manifest, manifest_path


def _unique(stem: str) -> str:
    return f"{stem}-{os.getpid()}-{time.monotonic_ns() % 10**8}"


def test_launch_without_yes_prints_the_cost_and_creates_nothing(pod_world):
    manifest, path = _prepared(pod_world, "cloud1-noyes")
    with pytest.raises(LaunchNotApproved):
        launch_pod(manifest, manifest_path=path, ctx=pod_world["ctx"], yes=False)
    assert pod_world["api"].requests == [], "--yes 없이는 API를 한 번도 부르지 않는다 (가격 조회도)"
    printed = "\n".join(pod_world["lines"])
    assert "dry-run" in printed and "$" in printed and "단계" in printed
    assert not (pod_world["tmp"] / "ledger.json").exists() or pod_world["ledger"].committed_usd() == 0.0


def test_a_full_pod_round_trip_creates_runs_fetches_deletes_and_closes_the_ledger(pod_world):
    run_id = _unique("cloud1-trip")
    manifest, path = _prepared(pod_world, run_id)
    dest = pod_world["tmp"] / "fetched"
    manifest = launch_pod(manifest, manifest_path=path, ctx=pod_world["ctx"], yes=True, dest=dest)
    api = pod_world["api"]
    assert len(api.created) == 1 and api.deleted == ["fakepod001"], "만들기 한 번, 지우기 한 번"
    assert api.live_pods() == []
    saved = load_manifest(path)
    assert validate(saved) == []
    assert saved["state"] == "completed" and saved["exit_reason"] == "completed"
    pod = saved["provider"]["pod"]
    assert pod["delete_confirmed"] is True and pod["spent_usd_estimate"] <= run_cap_usd(Budget.from_manifest(saved))
    got = {Path(e["remote"]).name: e for e in saved["artifacts"] if e.get("local")}
    assert got["hello.txt"]["sha256_match"] is True and (dest / "reports" / "hello.txt").read_text(encoding="utf-8") == "hi"
    assert got["state.json"]["sha256_match"] and got["job-report.json"]["sha256_match"]
    ledger = pod_world["ledger"].summary()
    assert ledger["open_runs"] == [] and ledger["runs"][0]["status"] == "closed"
    # 만들기 본문: 감시가 실렸고 마감이 상한 안이다 (실행기가 그 뒤 죽어도 pod는 마감을 넘지 못한다)
    body = api.created[0]
    assert body["cmd"] == [start_script()]
    assert base64.b64decode(body["env"]["ROBOJEV_WATCHDOG_B64"]).decode() == watchdog_script()
    assert int(body["env"]["ROBOJEV_TERMINATE_AT"]) <= pod["create_requested_epoch"] + 0.5 * 3600 - 60
    assert "RUNPOD_API_KEY" not in body["env"] and KEY not in json.dumps(body), "계정 키는 pod에 가지 않는다"
    assert KEY not in path.read_text(encoding="utf-8")


def test_the_pod_deadline_and_the_runner_deadline_fit_inside_the_cap(pod_world):
    manifest, path = _prepared(pod_world, _unique("cloud1-caps"), budget={"max_wall_hours": 2.0, "max_usd": 1.0})
    budget = Budget.from_manifest(manifest)
    hours, binding = pod_lifetime(budget)
    assert (binding, hours) == ("max_usd", pytest.approx(1.0 / 2.69)), "달러 상한이 먼저 구속한다"
    assert run_cap_usd(budget) == pytest.approx(1.0)
    manifest = launch_pod(manifest, manifest_path=path, ctx=pod_world["ctx"], yes=True, dest=pod_world["tmp"] / "f")
    pod = load_manifest(path)["provider"]["pod"]
    requested = pod["create_requested_epoch"]
    assert pod["terminate_at_epoch"] <= requested + hours * 3600 - 120 + 1, "감시 마감 = 수명 − 지울 여유"
    assert pod["runner_deadline_epoch"] == pod["terminate_at_epoch"] - 4 * 60, "러너 마감 = 감시 마감 − 가져갈 몫"
    assert pod_world["ledger"].summary()["runs"][0]["cap_usd"] == pytest.approx(1.0)


def test_a_failing_job_is_fetched_and_the_pod_is_still_deleted(pod_world):
    steps = [{"name": "boom", "argv": ["{python}", "-c", "import sys; sys.exit(7)"], "max_minutes": 1}]
    manifest, path = _prepared(pod_world, _unique("cloud1-boom"), steps=steps)
    launch_pod(manifest, manifest_path=path, ctx=pod_world["ctx"], yes=True, dest=pod_world["tmp"] / "f")
    saved = load_manifest(path)
    assert (saved["state"], saved["exit_reason"]) == ("failed", "step:boom:failed")
    assert pod_world["api"].live_pods() == [] and saved["provider"]["pod"]["delete_confirmed"] is True
    assert any(e.get("local") and e["remote"].startswith("logs/") for e in saved["artifacts"]), "실패한 run의 로그도 가져온다"


def test_the_launcher_does_not_wait_for_the_deadline_when_the_runner_dies(pod_world):
    """러너가 SIGKILL로 죽어 상태가 `running`에 멈추면 — 실행기는 프로세스가 없는 것을 보고 곧바로 가져오고 지운다(마감까지 기다리지 않는다)."""
    steps = [{"name": "kill-runner", "argv": ["{python}", "-c", "import os, signal, time; os.kill(os.getppid(), signal.SIGKILL); time.sleep(20)"],
              "max_minutes": 2}]  # fmt: skip
    manifest, path = _prepared(pod_world, _unique("cloud1-dead"), steps=steps)
    started = time.monotonic()
    launch_pod(manifest, manifest_path=path, ctx=pod_world["ctx"], yes=True, dest=pod_world["tmp"] / "f")
    assert time.monotonic() - started < 60, "가져오기 마감(≈ 25분)까지 기다리지 않았다"
    saved = load_manifest(path)
    assert (saved["state"], saved["exit_reason"]) == ("failed", "process_vanished")
    assert pod_world["api"].live_pods() == [] and saved["provider"]["pod"]["delete_confirmed"] is True


def test_a_failed_fetch_still_deletes_the_pod_after_trying_until_its_deadline(pod_world, monkeypatch):
    manifest, path = _prepared(pod_world, _unique("cloud1-nofetch"))
    attempts = []

    def broken_fetch(*args, **kwargs):
        attempts.append(1)
        raise OSError("rsync: connection unexpectedly closed")

    import robo_jev.launch.launcher as launcher_module

    monkeypatch.setattr(launcher_module, "fetch", broken_fetch)
    real_watch = cloud_module._watch
    ctx = pod_world["ctx"]

    def watch_then_fast_clock(*args, **kwargs):
        # run은 진짜 시간으로 끝까지 보고, 가져오기부터는 가짜 시계 — 한 번 쉴 때마다 5분이 흘러 가져오기 마감을 넘긴다
        state = real_watch(*args, **kwargs)
        clock = {"t": time.time()}
        ctx.clock = lambda: clock["t"]
        ctx.sleep = lambda s: clock.__setitem__("t", clock["t"] + 300.0)
        return state

    monkeypatch.setattr(cloud_module, "_watch", watch_then_fast_clock)
    launch_pod(manifest, manifest_path=path, ctx=pod_world["ctx"], yes=True, dest=pod_world["tmp"] / "f")
    assert len(attempts) >= 2, "가져오기는 마감까지 다시 한다"
    assert pod_world["api"].live_pods() == [] and load_manifest(path)["provider"]["pod"]["delete_confirmed"] is True
    assert any("결과 없이 지운다" in line for line in pod_world["lines"])


def test_an_interrupted_launcher_deletes_the_pod_before_the_runner_starts(pod_world, monkeypatch):
    manifest, path = _prepared(pod_world, _unique("cloud1-int"))

    def interrupt(*args, **kwargs):
        raise KeyboardInterrupt()

    monkeypatch.setattr(cloud_module, "_push_payload", interrupt)
    with pytest.raises(KeyboardInterrupt):
        launch_pod(manifest, manifest_path=path, ctx=pod_world["ctx"], yes=True)
    assert pod_world["api"].live_pods() == [] and pod_world["api"].deleted == ["fakepod001"]
    assert pod_world["ledger"].summary()["open_runs"] == []


def test_a_sigterm_to_the_launcher_takes_the_same_cleanup_path(pod_world, monkeypatch):
    manifest, path = _prepared(pod_world, _unique("cloud1-term"))

    def terminate(*args, **kwargs):
        os.kill(os.getpid(), signal.SIGTERM)
        time.sleep(1)
        raise AssertionError("SIGTERM이 중단으로 바뀌지 않았다")

    monkeypatch.setattr(cloud_module, "_push_payload", terminate)
    with pytest.raises(KeyboardInterrupt):
        launch_pod(manifest, manifest_path=path, ctx=pod_world["ctx"], yes=True)
    assert pod_world["api"].live_pods() == []


def test_an_interrupt_after_the_runner_started_deletes_unless_told_to_keep_and_finish_cleans_up(pod_world, monkeypatch):
    steps = [{"name": "slow", "argv": ["{python}", "-c", "import time; time.sleep(30)"], "max_minutes": 2}]
    manifest, path = _prepared(pod_world, _unique("cloud1-keep"), steps=steps)

    def interrupt(*args, **kwargs):
        raise KeyboardInterrupt()

    monkeypatch.setattr(cloud_module, "_watch", interrupt)
    with pytest.raises(KeyboardInterrupt):
        launch_pod(manifest, manifest_path=path, ctx=pod_world["ctx"], yes=True, keep_on_interrupt=True)
    saved = load_manifest(path)
    assert len(pod_world["api"].live_pods()) == 1, "--keep-on-interrupt: 남긴다 (감시가 마감에 지운다)"
    assert saved["provider"]["pod"]["id"] == "fakepod001" and saved["provider"]["pod"]["terminate_at_epoch"]
    monkeypatch.undo()
    finish(saved, manifest_path=path, ctx=pod_world["ctx"], dest=pod_world["tmp"] / "f", wait=False)
    assert pod_world["api"].live_pods() == [] and load_manifest(path)["provider"]["pod"]["delete_confirmed"] is True
    assert pod_world["ledger"].summary()["open_runs"] == []


def test_an_ambiguous_create_is_swept_by_name_and_counted(pod_world):
    pod_world["api"].create_mode = "timeout_after_create"
    manifest, path = _prepared(pod_world, _unique("cloud1-ambig"))
    with pytest.raises(RunPodError):
        launch_pod(manifest, manifest_path=path, ctx=pod_world["ctx"], yes=True)
    api = pod_world["api"]
    assert len(api.calls("POST", "pods")) == 1, "애매한 POST를 다시 보내지 않는다"
    assert api.live_pods() == [] and api.deleted == ["fakepod001"], "응답을 못 받았어도 생긴 pod는 이름으로 찾아 지운다"
    run = pod_world["ledger"].summary()["runs"][0]
    assert run["status"] == "closed", "만들어졌던 pod는 장부에서 풀지 않고 쓴 만큼으로 닫는다"


def test_a_create_that_runpod_refused_releases_the_reservation(pod_world):
    pod_world["api"].create_mode = "http_409"
    manifest, path = _prepared(pod_world, _unique("cloud1-409"))
    with pytest.raises(RunPodError, match="409"):
        launch_pod(manifest, manifest_path=path, ctx=pod_world["ctx"], yes=True)
    assert pod_world["api"].pods == {}
    assert pod_world["ledger"].summary()["runs"][0]["status"] == "released"
    assert pod_world["ledger"].committed_usd() == 0.0


def test_a_created_pod_billed_at_another_rate_is_deleted_at_once(pod_world):
    pod_world["api"].cost_override = 3.99
    manifest, path = _prepared(pod_world, _unique("cloud1-cost"))
    with pytest.raises(PriceMismatch, match="실제 청구 단가"):
        launch_pod(manifest, manifest_path=path, ctx=pod_world["ctx"], yes=True)
    assert pod_world["api"].live_pods() == [] and pod_world["api"].deleted == ["fakepod001"]


def test_a_catalog_price_mismatch_at_launch_creates_nothing_and_reserves_nothing(pod_world):
    manifest, path = _prepared(pod_world, _unique("cloud1-price"), budget={"hourly_usd": 2.49})
    with pytest.raises(PriceMismatch):
        launch_pod(manifest, manifest_path=path, ctx=pod_world["ctx"], yes=True)
    assert pod_world["api"].calls("POST") == [] and pod_world["ledger"].summary()["runs"] == []


def test_the_phase_cap_stops_a_launch_before_any_pod_exists(pod_world):
    pod_world["ledger"].reserve("earlier", cap_usd=149.0, hourly_usd=2.69, manifest=None)
    manifest, path = _prepared(pod_world, _unique("cloud1-phase"), budget={"max_usd": 0.9})
    pod_world["ledger"].reserve("sneaky", cap_usd=0.5, hourly_usd=2.69, manifest=None)  # prepare 뒤에 남은 돈이 줄었다
    with pytest.raises(PhaseCapExceeded):
        launch_pod(manifest, manifest_path=path, ctx=pod_world["ctx"], yes=True)
    assert pod_world["api"].calls("POST") == []


def test_prepare_defaults_the_dollar_cap_to_what_remains_of_the_phase_and_refuses_more(pod_world):
    pod_world["ledger"].reserve("earlier", cap_usd=140.0, hourly_usd=2.69, manifest=None)
    manifest, _ = _prepared(pod_world, _unique("cloud1-rest"), budget={"max_usd": None, "max_wall_hours": 10})
    assert manifest["budget"]["max_usd"] == pytest.approx(10.0), "달러 상한을 적지 않으면 단계에 남은 돈"
    with pytest.raises(PhaseCapExceeded):
        _prepared(pod_world, _unique("cloud1-over"), budget={"max_usd": 20.0, "max_wall_hours": 10})


def test_a_pod_without_an_armed_watchdog_is_deleted_without_working(pod_world):
    bootstrap = FAKE_BOOTSTRAP.replace('echo "WATCHDOG=armed"', 'echo "WATCHDOG=missing"')
    manifest, path = _prepared(pod_world, _unique("cloud1-nowd"), provider={"bootstrap": bootstrap})
    with pytest.raises(RuntimeError, match="감시가 살아 있지 않다"):
        launch_pod(manifest, manifest_path=path, ctx=pod_world["ctx"], yes=True)
    assert pod_world["api"].live_pods() == []
    assert not (pod_world["tmp"] / "pod" / "run" / "state.json").exists(), "러너를 띄우지 않았다"


def test_a_boot_that_never_becomes_ready_is_deleted(pod_world):
    pod_world["api"].boot_status = "ERROR"
    manifest, path = _prepared(pod_world, _unique("cloud1-boot"))
    with pytest.raises(RunPodError, match="ERROR"):
        launch_pod(manifest, manifest_path=path, ctx=pod_world["ctx"], yes=True)
    assert pod_world["api"].live_pods() == [] and pod_world["api"].deleted


def test_a_delete_that_cannot_be_confirmed_leaves_unknown_and_keeps_the_whole_cap_counted(pod_world):
    pod_world["api"].delete_never = True
    manifest, path = _prepared(pod_world, _unique("cloud1-stuck"))
    clock = {"t": time.time()}
    pod_world["ctx"].provider._clock = lambda: clock["t"]
    pod_world["ctx"].provider._sleep = lambda s: clock.__setitem__("t", clock["t"] + 60)
    launch_pod(manifest, manifest_path=path, ctx=pod_world["ctx"], yes=True, dest=pod_world["tmp"] / "f")
    saved = load_manifest(path)
    assert (saved["state"], saved["exit_reason"]) == ("unknown", "pod-delete-unconfirmed")
    assert saved["provider"]["pod"]["delete_confirmed"] is False
    run = pod_world["ledger"].summary()["runs"][0]
    assert run["status"] == "delete-unconfirmed" and pod_world["ledger"].committed_usd() == pytest.approx(run["cap_usd"])
    assert any("콘솔에서 확인" in line for line in pod_world["lines"])


def test_down_deletes_now_and_closes_the_ledger(pod_world, monkeypatch):
    manifest, path = _prepared(pod_world, _unique("cloud1-down"), steps=[{"name": "slow", "argv": ["{python}", "-c", "import time; time.sleep(30)"], "max_minutes": 2}])
    monkeypatch.setattr(cloud_module, "_watch", lambda *a, **k: (_ for _ in ()).throw(KeyboardInterrupt()))
    with pytest.raises(KeyboardInterrupt):
        launch_pod(manifest, manifest_path=path, ctx=pod_world["ctx"], yes=True, keep_on_interrupt=True)
    assert down(load_manifest(path), manifest_path=path, ctx=pod_world["ctx"]) is True
    assert pod_world["api"].live_pods() == [] and pod_world["ledger"].summary()["open_runs"] == []


def test_one_manifest_creates_at_most_one_pod(pod_world):
    run_id = _unique("cloud1-once")
    manifest, path = _prepared(pod_world, run_id)
    launch_pod(manifest, manifest_path=path, ctx=pod_world["ctx"], yes=True, dest=pod_world["tmp"] / "f")
    with pytest.raises(RuntimeError, match="두 번"):
        launch_pod(load_manifest(path), manifest_path=path, ctx=pod_world["ctx"], yes=True)
    assert len(pod_world["api"].created) == 1


def test_the_dry_run_shows_steps_time_cost_uploads_and_fetches_without_any_request(pod_world):
    manifest, _ = _prepared(pod_world, _unique("cloud1-plan"))
    plan = plan_job(manifest, ledger=pod_world["ledger"])
    text = format_plan(plan)
    assert pod_world["api"].requests == []
    assert plan["cost"]["worst_case_usd"] == pytest.approx(0.5 * 2.69) and plan["caps"]["binding"] == "max_wall_hours"
    assert [row["name"] for row in plan["steps"]] == ["hello"]
    assert any(item["path"] == REQUESTS for item in plan["uploads"]["files"]) and plan["uploads"]["code"]["payload_sha256"]
    assert "reports/*" in plan["fetch"] and "state.json" in plan["fetch"]
    assert plan["ledger"]["phase"]["cap_usd"] == 150.0
    for needle in ("dry-run", "단가", "상한", "장부", "시간", "비용", "올리기", "가져오기"):
        assert needle in text


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def _cli(*args: str, env: dict | None = None) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(REPO / "scripts" / "launch_run.py"), *args], capture_output=True, text=True, check=False,
                          env={**os.environ, "CUDA_VISIBLE_DEVICES": "", **(env or {})})  # fmt: skip


def test_the_cli_launch_without_yes_prints_the_plan_and_exits_2(pod_world):
    manifest, path = _prepared(pod_world, _unique("cloud1-cli"))
    result = _cli("launch", "--manifest", str(path), "--ledger", str(pod_world["tmp"] / "ledger.json"), "--key-file", str(pod_world["tmp"] / "none.env"))
    assert result.returncode == 2, result.stderr
    assert "dry-run" in result.stdout and "--yes" in result.stderr
    assert load_manifest(path)["state"] == "prepared"


def test_the_cli_refuses_a_missing_or_open_key_file_before_any_request(tmp_path):
    result = _cli("prices", "--key-file", str(tmp_path / "missing.env"))
    assert result.returncode != 0 and "키 거절" in result.stderr and "RUNPOD_API_KEY=<키>" in result.stderr
    open_file = tmp_path / "open.env"
    open_file.write_text(f"RUNPOD_API_KEY={KEY}\n", encoding="utf-8")
    open_file.chmod(0o644)
    result = _cli("pods", "--key-file", str(open_file))
    assert result.returncode != 0 and "chmod 600" in result.stderr and KEY not in result.stderr + result.stdout


def test_the_cli_dry_run_reads_only_the_manifest(pod_world):
    manifest, path = _prepared(pod_world, _unique("cloud1-cli-dry"))
    result = _cli("dry-run", "--manifest", str(path), "--ledger", str(pod_world["tmp"] / "ledger.json"))
    assert result.returncode == 0, result.stderr
    assert "dry-run" in result.stdout and "NVIDIA H100 80GB HBM3" in result.stdout


def test_the_cli_ledger_shows_the_150_dollar_phase(tmp_path):
    result = _cli("ledger", "--ledger", str(tmp_path / "ledger.json"))
    assert result.returncode == 0 and json.loads(result.stdout)["phase"]["cap_usd"] == 150.0


def test_the_fake_catalog_matches_the_real_catalog_shape():
    """가짜 카탈로그는 v2 `GpuType`의 필수 키를 갖는다 (openapi 기준)."""
    for gpu in CATALOG:
        assert {"id", "name", "memory", "secure", "community", "price", "maxCount"} <= set(gpu)
        assert {"secure", "community"} <= set(gpu["price"])
