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
from robo_jev.launch.ledger import DEFAULT_PHASE, Ledger, LedgerError, PhaseCapExceeded, reconcile
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


#: Jobs written after the user raised the phase cap to $250 (2026-10-13, recorded in the ledger's history with
#: `launch_run.py ledger --set-phase-cap 250`) carry the raised cap — the ledger refuses a job whose phase differs from its own.
RAISED_PHASE_JOBS = ("decisions-v04-s0-h100.yaml", "decisions-v04-s1-h100.yaml", "decisions-v05-s0-h100.yaml", "decisions-v05-s1-h100.yaml")


def test_the_default_phase_is_150_dollars_and_the_shipped_configs_agree():
    assert DEFAULT_PHASE["cap_usd"] == 150.0
    for path in sorted((REPO / "configs" / "cloud").glob("*.yaml")):
        phase = (yaml.safe_load(path.read_text(encoding="utf-8")).get("budget") or {}).get("phase")
        cap = 250 if path.name in RAISED_PHASE_JOBS else 150
        assert phase == {"name": "cloud-phase-1", "cap_usd": cap}, f"{path.name}: 단계 상한은 ${cap}"


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
i=0; while [ ! -s "$R/watchdog-auth" ] && [ "$i" -lt 30 ]; do sleep 1; i=$((i + 1)); done
echo "WATCHDOG_AUTH=$(cat "$R/watchdog-auth" 2>/dev/null || echo none)"
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
    """"pod"의 컨테이너 — 가짜 API에 실린 env로 **진짜 시작 명령**을 sh로 돌린다 (감시가 뜨고 pid 파일이 생긴다).

    가짜 `runpodctl`·`curl`이 PATH 앞에 있다: 읽기(`pod get`, `-w` 상태 코드, GraphQL 읽기)는 `routes_ok`대로 인증되거나 거절되고, 지우기
    (`pod delete`·`remove pod`·REST DELETE·GraphQL podTerminate)가 `delete_ok`면 표지 파일(`marker`)을 만든다 — 가짜 API는 그 표지를 보고
    pod를 TERMINATED로 둔다(pod 안의 길이 자기 pod를 지운 것).
    """

    def __init__(self, tmp_path: Path, *, routes_ok: bool = True, delete_ok: bool = True) -> None:
        self.tmp = tmp_path
        self.process: subprocess.Popen | None = None
        self.calls = tmp_path / "runpodctl.log"
        self.marker = tmp_path / "deleted-from-inside-the-pod"
        self.routes_ok = routes_ok
        self.delete_ok = delete_ok

    def _fakes(self, bin_dir: Path) -> None:
        delete = f'touch "{self.marker}"; exit 0' if self.delete_ok else "exit 1"
        read = "exit 0" if self.routes_ok else "exit 1"
        (bin_dir / "runpodctl").write_text(
            f'#!/bin/sh\necho "runpodctl $*" >> "{self.calls}"\ncase "$1 $2" in\n'
            f'  "pod delete"|"remove pod") {delete} ;;\n  "pod get"|"get pod") {read} ;;\nesac\nexit 0\n', encoding="utf-8")
        code = "200" if self.routes_ok else "401"
        body = '{"data":{"pod":{"id":"%s"}}}' if self.routes_ok else '{"errors":[{"message":"unauthorized"}]}'
        (bin_dir / "curl").write_text(
            f'#!/bin/sh\necho "curl $*" >> "{self.calls}"\ncase "$*" in\n'
            f'  *"-X DELETE"*|*podTerminate*) {delete if self.delete_ok else "exit 22"} ;;\n'
            f'  *" -w "*) printf {code}; exit 0 ;;\n'
            f'  *"query {{ pod"*) printf \'{body}\' "$RUNPOD_POD_ID"; exit 0 ;;\nesac\nexit 0\n', encoding="utf-8")
        for name in ("runpodctl", "curl"):
            (bin_dir / name).chmod(0o755)

    def boot(self, pod_env: dict[str, str], pod_id: str) -> None:
        if self.process is not None:
            return
        bin_dir = self.tmp / "podbin"
        bin_dir.mkdir(exist_ok=True)
        self._fakes(bin_dir)
        env = {"PATH": f"{bin_dir}:/usr/bin:/bin", "HOME": str(self.tmp), "RUNPOD_POD_ID": pod_id, "RUNPOD_API_KEY": "pod-scoped",
               "ROBOJEV_WATCHDOG_POLL": "0.5", **pod_env}  # fmt: skip
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


class FakeFence:
    """셋째 울타리(이 상자의 systemd 타이머)의 기록용 가짜 — 검사가 진짜 타이머를 남기지 않게."""

    def __init__(self, *, fail: bool = False) -> None:
        self.armed: list[dict] = []
        self.disarmed: list[str] = []
        self.fail = fail

    def arm(self, *, manifest_path, run_id, at_epoch, now_epoch):
        if self.fail:
            raise RuntimeError("셋째 울타리(이 상자의 systemd 타이머)를 걸지 못했다: no user bus")
        record = {"unit": f"cloud-fence-{run_id}", "fires_in_seconds": int(at_epoch - now_epoch), "fires_at_epoch": at_epoch, "armed": True,
                  "manifest": str(manifest_path)}  # fmt: skip
        self.armed.append(record)
        return dict(record)

    def disarm(self, *, run_id):
        self.disarmed.append(run_id)
        return True


@pytest.fixture
def pod_world(tmp_path, request):
    marker = getattr(request, "param", {}) or {}
    api = FakeRunPodApi()
    container = Container(tmp_path, routes_ok=marker.get("routes_ok", True), delete_ok=marker.get("delete_ok", True))
    api.external_delete_marker = str(container.marker)
    ledger = Ledger(tmp_path / "ledger.json")
    lines: list[str] = []

    def factory(view, manifest):
        pod = api.pods[view.pod_id]
        container.boot(pod["env"], view.pod_id)
        root = manifest["provider"]["remote_root"]
        return PinnedLocal(remote_dir=f"{root}/run", remote_repo=f"{root}/repo", remote_python="python3", gpus=1)

    fence = FakeFence()
    ctx = PodContext(provider=_provider(api), ledger=ledger, backend_factory=factory, out=lines.append, poll_seconds=0.3, fence=fence,
                     ambiguous_sweep_seconds=2.0, ambiguous_sweep_poll=0.2, self_delete_timeout=20.0)  # fmt: skip
    yield {"api": api, "ctx": ctx, "ledger": ledger, "container": container, "lines": lines, "tmp": tmp_path, "fence": fence}
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


def test_the_billed_amount_is_read_back_into_the_ledger_and_the_larger_figure_counts(pod_world):
    """돈을 쓴 뒤 장부에 실제 청구액을 적는다 — 읽기 전용 청구 기록에서. 추정보다 많이 청구됐으면 그쪽을 센다."""
    manifest, path = _prepared(pod_world, _unique("cloud1-bill"))
    launch_pod(manifest, manifest_path=path, ctx=pod_world["ctx"], yes=True, dest=pod_world["tmp"] / "f")
    estimate = pod_world["ledger"].summary()["runs"][0]["spent_usd_estimate"]
    pod_world["api"].billed["fakepod001"] = round(estimate + 0.25, 4)
    reader = RunPodClient(Secret(KEY), transport=pod_world["api"], read_only=True)
    rows = reconcile(pod_world["ledger"], reader)
    assert rows == [{"run_id": manifest["run_id"], "pod_id": "fakepod001", "billed_usd": round(estimate + 0.25, 4), "estimate_usd": estimate}]
    assert pod_world["ledger"].committed_usd() == pytest.approx(round(estimate + 0.25, 4)), "추정보다 많이 청구됐다 — 청구액을 센다"
    assert pod_world["api"].calls("GET", "billing/pods")[0]["query"]["podId"] == "fakepod001"


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


def test_a_late_sshd_is_waited_for_not_mistaken_for_a_dead_pod(pod_world, monkeypatch):
    """RUNNING 직후엔 sshd가 아직 없다 — 처음 몇 번의 접속 실패로 pod를 지우지 않는다."""
    manifest, path = _prepared(pod_world, _unique("cloud1-ssh"))
    real_shell = LocalBackend.shell
    refused = {"n": 0}

    def flaky_shell(self, line, *, timeout=None):
        if line == "true" and refused["n"] < 2:
            refused["n"] += 1
            from robo_jev.launch.providers.base import Exec

            return Exec(argv=["ssh"], returncode=255, stdout="", stderr="ssh: connect to host port 40022: Connection refused")
        return real_shell(self, line, timeout=timeout)

    monkeypatch.setattr(LocalBackend, "shell", flaky_shell)
    pod_world["ctx"].sleep = lambda s: None
    launch_pod(manifest, manifest_path=path, ctx=pod_world["ctx"], yes=True, dest=pod_world["tmp"] / "f")
    saved = load_manifest(path)
    assert saved["state"] == "completed" and saved["provider"]["pod"]["ssh_attempts"] == 3


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


# --------------------------------------------------------------------------
# 수정 1 (리뷰 1): 울타리 셋·증명 모드·애매한 만들기·장부의 실제 지출
# --------------------------------------------------------------------------


@pytest.mark.parametrize("pod_world", [{"routes_ok": False}], indirect=True)
def test_a_pod_whose_delete_routes_do_not_authenticate_is_deleted_without_working(pod_world):
    """감시가 같은 키로 자기 pod를 **읽어 봤더니** 어느 길도 인증되지 않았다 — 실행기가 죽으면 남을 pod이므로 일하지 않고 지운다 (I1 c)."""
    manifest, path = _prepared(pod_world, _unique("cloud1-noroute"))
    with pytest.raises(RuntimeError, match="지울 길이 하나도 인증되지 않았다"):
        launch_pod(manifest, manifest_path=path, ctx=pod_world["ctx"], yes=True)
    saved = load_manifest(path)
    assert pod_world["api"].live_pods() == [] and saved["provider"]["pod"]["delete_confirmed"] is True
    assert "runpodctl=fail" in saved["provider"]["pod"]["bootstrap"]["WATCHDOG_AUTH"]
    assert not (pod_world["tmp"] / "pod" / "run" / "state.json").exists(), "러너를 띄우지 않았다"
    calls = pod_world["container"].calls.read_text(encoding="utf-8")
    assert "pod get" in calls and "pod delete" not in calls, "점검은 읽기만 했다 (지운 것은 실행기다)"


def test_the_third_fence_is_armed_past_the_watchdog_deadline_and_taken_down_after_a_confirmed_delete(pod_world):
    """셋째 울타리 — 이 상자의 타이머가 감시 마감 + 5분에 `down --fence` (I1 d). 지움이 확인되면 내린다."""
    run_id = _unique("cloud1-fence")
    manifest, path = _prepared(pod_world, run_id)
    launch_pod(manifest, manifest_path=path, ctx=pod_world["ctx"], yes=True, dest=pod_world["tmp"] / "f")
    pod = load_manifest(path)["provider"]["pod"]
    (armed,) = pod_world["fence"].armed
    assert armed["fires_at_epoch"] == pytest.approx(pod["terminate_at_epoch"] + 300.0)
    assert armed["manifest"] == str(path) and pod["fence"]["unit"] == f"cloud-fence-{run_id}"
    assert pod_world["fence"].disarmed == [run_id] and pod["fence"]["disarmed"] is True


def test_a_fence_that_cannot_be_armed_stops_the_launch_and_deletes_the_pod(pod_world):
    pod_world["ctx"].fence = FakeFence(fail=True)
    manifest, path = _prepared(pod_world, _unique("cloud1-nofence"))
    with pytest.raises(RuntimeError, match="셋째 울타리"):
        launch_pod(manifest, manifest_path=path, ctx=pod_world["ctx"], yes=True)
    assert pod_world["api"].live_pods() == [] and pod_world["ledger"].summary()["open_runs"] == []


def test_the_systemd_fence_runs_down_with_the_fence_flag_at_the_right_time():
    from robo_jev.launch.cloud import SystemdFence

    seen = []

    def runner(argv, **kwargs):
        seen.append(argv)
        return subprocess.CompletedProcess(argv, 0, "", "")

    fence = SystemdFence(python="/py", repo=Path("/repo"), runner=runner)
    armed = fence.arm(manifest_path="/runs/x.json", run_id="run-1", at_epoch=1_000_900.0, now_epoch=1_000_000.0)
    argv = seen[0]
    assert argv[:2] == ["systemd-run", "--user"] and "--unit=cloud-fence-run-1" in argv and "--on-active=900s" in argv
    assert argv[-6:] == ["/py", "/repo/scripts/launch_run.py", "down", "--manifest", str(Path("/runs/x.json").resolve()), "--fence"]
    assert armed["fires_in_seconds"] == 900 and armed["armed"] is True
    fence.disarm(run_id="run-1")
    assert seen[1] == ["systemctl", "--user", "stop", "cloud-fence-run-1.timer"]


def test_down_from_the_fence_does_nothing_once_the_delete_was_confirmed(pod_world):
    manifest, path = _prepared(pod_world, _unique("cloud1-fence-late"))
    launch_pod(manifest, manifest_path=path, ctx=pod_world["ctx"], yes=True, dest=pod_world["tmp"] / "f")
    before = pod_world["ledger"].read()
    calls = len(pod_world["api"].requests)
    assert down(load_manifest(path), manifest_path=path, ctx=pod_world["ctx"], fence=True) is True
    assert pod_world["ledger"].read()["runs"] == before["runs"], "장부를 두 번 닫지 않는다"
    assert len(pod_world["api"].requests) == calls, "API도 부르지 않는다"


def test_prove_self_delete_lets_the_in_pod_route_delete_and_names_the_route(pod_world):
    """증명 모드 (I1 a): job과 가져오기가 끝난 뒤 감시가 **자기 길로** 지운다 — 실행기는 DELETE를 보내지 않고 404만 확인한다."""
    manifest, path = _prepared(pod_world, _unique("cloud1-prove"))
    launch_pod(manifest, manifest_path=path, ctx=pod_world["ctx"], yes=True, dest=pod_world["tmp"] / "f", prove_self_delete=True)
    saved = load_manifest(path)
    proof = saved["provider"]["pod"]["self_delete"]
    assert proof["proved"] is True and proof["route"] == "runpodctl pod delete" and proof["deleted_by"] == "watchdog:runpodctl pod delete"
    assert pod_world["api"].deleted == [], "실행기는 DELETE를 보내지 않았다 — pod 안의 길이 지웠다"
    assert pod_world["api"].live_pods() == [] and saved["provider"]["pod"]["delete_confirmed"] is True
    assert pod_world["ledger"].summary()["runs"][0]["status"] == "closed"


@pytest.mark.parametrize("pod_world", [{"delete_ok": False}], indirect=True)
def test_prove_self_delete_falls_back_to_the_launcher_when_the_in_pod_route_fails(pod_world):
    pod_world["ctx"].self_delete_timeout = 4.0
    manifest, path = _prepared(pod_world, _unique("cloud1-prove-fail"))
    launch_pod(manifest, manifest_path=path, ctx=pod_world["ctx"], yes=True, dest=pod_world["tmp"] / "f", prove_self_delete=True)
    proof = load_manifest(path)["provider"]["pod"]["self_delete"]
    assert proof["proved"] is False and proof["deleted_by"] == "launcher"
    assert pod_world["api"].deleted == ["fakepod001"], "기한 안에 안 지워지면 실행기가 지운다"


def test_a_pod_created_after_the_client_gave_up_is_found_by_the_repeated_sweep(pod_world):
    """POST가 끊긴 뒤에 RunPod가 pod를 만든다 — 한 번 찾고 끝내면 놓친다(I2). 몇 번 다시 찾아 지운다."""
    pod_world["api"].create_mode = "late"
    pod_world["api"].late_after = 4
    manifest, path = _prepared(pod_world, _unique("cloud1-late"))
    with pytest.raises(RunPodError):
        launch_pod(manifest, manifest_path=path, ctx=pod_world["ctx"], yes=True)
    pod = load_manifest(path)["provider"]["pod"]
    assert pod["ambiguous_sweeps"] >= 2 and pod["id"] == "fakepod001"
    assert pod_world["api"].live_pods() == [] and pod_world["ledger"].summary()["runs"][0]["status"] == "closed"


def test_an_ambiguous_create_that_is_never_found_keeps_the_reservation_and_blocks_a_relaunch(pod_world):
    pod_world["api"].create_mode = "never"
    manifest, path = _prepared(pod_world, _unique("cloud1-never"))
    with pytest.raises(RunPodError):
        launch_pod(manifest, manifest_path=path, ctx=pod_world["ctx"], yes=True)
    saved = load_manifest(path)
    assert (saved["state"], saved["exit_reason"]) == ("unknown", "create-ambiguous")
    run = pod_world["ledger"].summary()["runs"][0]
    assert run["status"] == "create-ambiguous" and pod_world["ledger"].committed_usd() == pytest.approx(run["cap_usd"]), "'모른다'는 상한 전부"
    with pytest.raises(RuntimeError, match="두 번"):
        launch_pod(saved, manifest_path=path, ctx=pod_world["ctx"], yes=True)
    with pytest.raises(LedgerError, match="애매"):
        pod_world["ledger"].release(saved["run_id"], reason="사람이 지웠다고 믿는다")
    assert down(saved, manifest_path=path, ctx=pod_world["ctx"]) is True
    resolved = load_manifest(path)
    assert resolved["exit_reason"] == "create-ambiguous-resolved" and pod_world["ledger"].committed_usd() == 0.0


def test_a_pod_still_alive_past_its_deadline_is_counted_until_it_is_actually_deleted(pod_world):
    """감시가 실패해 pod가 마감을 넘겨 살아 있었다 — 장부는 상한에서 자르지 않고 실제로 지운 때까지 센다 (I3)."""
    manifest, path = _prepared(pod_world, _unique("cloud1-overrun"))
    api = pod_world["api"]
    status, body = api(("POST"), "https://api.runpod.io/v2/pods", {}, json.dumps({"name": manifest["run_id"], "image": "x", "cloud": "SECURE",
                       "gpu": {"id": "NVIDIA H100 80GB HBM3", "count": 1}, "env": {}}).encode(), 30)[0], None  # fmt: skip
    pod_world["ledger"].reserve(manifest["run_id"], cap_usd=1.345, hourly_usd=2.69, manifest=str(path), phase=manifest["budget"]["phase"])
    pod_world["ledger"].activate(manifest["run_id"], pod_id="fakepod001")
    api.pods["fakepod001"]["status"] = "RUNNING"
    now_epoch = time.time()
    manifest["provider"]["pod"] = {"id": "fakepod001", "create_requested_epoch": now_epoch - 3 * 3600, "terminate_at_epoch": now_epoch - 2.5 * 3600}
    assert down(manifest, manifest_path=path, ctx=pod_world["ctx"]) is True
    spent = load_manifest(path)["provider"]["pod"]["spent_usd_estimate"]
    assert spent == pytest.approx(3 * 2.69, rel=0.01), "3시간 × 단가 — 감시 마감(0.5 h)으로 자르지 않는다"
    assert pod_world["ledger"].committed_usd() == pytest.approx(spent)


def test_billed_amounts_are_reconciled_before_every_launch_and_can_refuse_it(pod_world):
    """닫힌 run의 실제 청구가 추정보다 많았다 — launch는 예약 전에 청구 기록을 읽고(읽기 전용) 그 숫자로 $150을 잰다 (I3)."""
    first, path1 = _prepared(pod_world, _unique("cloud1-billed"))
    launch_pod(first, manifest_path=path1, ctx=pod_world["ctx"], yes=True, dest=pod_world["tmp"] / "f")
    pod_world["api"].billed["fakepod001"] = 149.5  # 추정보다 훨씬 많이 청구됐다
    second, path2 = _prepared(pod_world, _unique("cloud1-after"))  # prepare는 아직 대조 전의 장부를 본다
    with pytest.raises(PhaseCapExceeded):
        launch_pod(second, manifest_path=path2, ctx=pod_world["ctx"], yes=True)
    assert len(pod_world["api"].created) == 1, "둘째 pod는 만들지 않았다"
    billing = [i for i, r in enumerate(pod_world["api"].requests) if r["path"] == "billing/pods"]
    assert billing, "청구 기록을 읽었다"
    assert pod_world["ledger"].committed_usd() == pytest.approx(149.5)


def test_the_cli_relaunches_itself_under_systemd_and_returns_at_once(pod_world, tmp_path):
    """systemd 유닛 밖에서 `launch --yes`를 부르면 유닛(cloud-<run-id>, TimeoutStopSec=600)으로 다시 띄우고 곧바로 돌아온다 — 도구의
    시간 초과가 실행기를 SIGKILL하지 못하게 (I1). 유닛 안에서는(`--foreground`) 그대로 돈다."""
    manifest, path = _prepared(pod_world, _unique("cloud1-detach"))
    fake = tmp_path / "fake-systemd-run"
    record = tmp_path / "systemd-run.args"
    fake.write_text(f'#!/bin/sh\nprintf "%s\\n" "$@" > "{record}"\nexit 0\n', encoding="utf-8")
    fake.chmod(0o755)
    env = {k: v for k, v in os.environ.items() if k != "INVOCATION_ID"}
    env.update(ROBOJEV_SYSTEMD_RUN=str(fake), CUDA_VISIBLE_DEVICES="")
    started = time.monotonic()
    result = subprocess.run([sys.executable, str(REPO / "scripts" / "launch_run.py"), "launch", "--manifest", str(path), "--yes",
                             "--prove-self-delete", "--key-file", str(tmp_path / "no-key.env"), "--ledger", str(tmp_path / "ledger.json")],
                            capture_output=True, text=True, check=False, env=env)  # fmt: skip
    assert result.returncode == 0, result.stderr
    assert time.monotonic() - started < 30
    out = json.loads(result.stdout)
    assert out["detached"] is True and out["unit"] == f"cloud-{manifest['run_id']}" and out["log"].endswith(f"{manifest['run_id']}-launch.log")
    args = record.read_text(encoding="utf-8").splitlines()
    assert args[:2] == ["--user", f"--unit=cloud-{manifest['run_id']}"] and "TimeoutStopSec=600" in args
    assert args[-1] == "--foreground" and "--prove-self-delete" in args and "--yes" in args and str(path) in args
    assert pod_world["api"].requests == [] and load_manifest(path)["state"] == "prepared"


def test_the_cli_exits_3_when_a_failed_launch_left_a_pod_unconfirmed(pod_world, monkeypatch, tmp_path):
    """예외 길에서도 명세가 '지움 미확인'이면 종료 코드 3 — 감싸는 쪽이 코드만 봐도 알게 (리뷰 1의 Minor 1)."""
    import importlib.util

    import robo_jev.launch.cloud as cloud

    manifest, path = _prepared(pod_world, _unique("cloud1-exit3"))
    spec = importlib.util.spec_from_file_location("launch_run_exit3", REPO / "scripts" / "launch_run.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["launch_run_exit3"] = module
    spec.loader.exec_module(module)

    def failing_launch(manifest, *, manifest_path, **kwargs):
        manifest["provider"]["pod"].update(id="fakepod009", delete_confirmed=False)
        manifest["exit_reason"] = "pod-delete-unconfirmed"
        from robo_jev.launch.manifest import note_state, save_manifest

        note_state(manifest, "unknown", note="검사")
        save_manifest(manifest_path, manifest)
        raise PriceMismatch("pod fakepod009: 실제 청구 단가 다름")

    monkeypatch.setattr(cloud, "launch_pod", failing_launch)
    monkeypatch.setattr(module, "_pod_context", lambda args, read_only=False: None)
    code = module.main(["launch", "--manifest", str(path), "--yes", "--foreground"])
    assert code == 3


def test_a_stale_reservation_can_be_released_from_the_cli_but_never_one_with_a_pod(tmp_path):
    ledger = Ledger(tmp_path / "ledger.json")
    ledger.reserve("stale", cap_usd=2.0, hourly_usd=4.0, manifest=None)
    ledger.reserve("real", cap_usd=2.0, hourly_usd=4.0, manifest=None)
    ledger.activate("real", pod_id="p1")
    result = _cli("ledger", "--ledger", str(tmp_path / "ledger.json"), "--release", "stale")
    assert result.returncode != 0 and "이유" in result.stderr
    result = _cli("ledger", "--ledger", str(tmp_path / "ledger.json"), "--release", "real", "--reason", "x")
    assert result.returncode != 0 and "pod p1" in result.stderr
    result = _cli("ledger", "--ledger", str(tmp_path / "ledger.json"), "--release", "stale", "--reason", "실행기가 만들기 전에 죽었다")
    assert result.returncode == 0 and ledger.summary()["open_runs"] == ["real"]
