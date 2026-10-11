"""pod 안의 자기 종료 감시 (`infra/runpod-watchdog.sh`)와 컨테이너 시작 명령 — **진짜 sh**로 돈다 (Task cloud1 돈 안전 (iii)).

실행기가 죽어도 pod가 남지 않게 하는 것은 이 스크립트다. 그래서 흉내가 아니라 그 파일 자체를 돌린다: PATH 앞에 가짜 `runpodctl`·`curl`을
두어 무엇이 불렸는지 기록하고(진짜 API는 부르지 않는다), 마감·시작 유예·끝난 뒤 유예 각각에 지우기가 불리는지, 실패하면 다른 길로
넘어가는지, 키가 로그에 남지 않는지를 본다.
"""

from __future__ import annotations

import base64
import os
import signal
import subprocess
import time
from pathlib import Path

import pytest

from robo_jev.launch.cloud import DEFAULT_BOOTSTRAP, UV_VERSION, _parse_markers, start_script, watchdog_script
from robo_jev.launch.manifest import REPO

WATCHDOG = REPO / "infra" / "runpod-watchdog.sh"
POD_KEY = "pod-scoped-SECRET-key-123"


def _fakes(tmp_path: Path, *, runpodctl_ok: bool = True, curl_ok: bool = True) -> tuple[Path, Path]:
    """가짜 runpodctl·curl — argv를 한 줄씩 기록하고 정해진 코드로 끝난다."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    record = tmp_path / "calls.log"
    for name, ok in (("runpodctl", runpodctl_ok), ("curl", curl_ok)):
        script = bin_dir / name
        script.write_text(f'#!/bin/sh\necho "{name} $*" >> "{record}"\nexit {0 if ok else 1}\n', encoding="utf-8")
        script.chmod(0o755)
    return bin_dir, record


def _env(tmp_path: Path, bin_dir: Path, **extra: str) -> dict[str, str]:
    run_dir = tmp_path / "run"
    run_dir.mkdir(exist_ok=True)
    env = {
        "PATH": f"{bin_dir}:/usr/bin:/bin", "HOME": str(tmp_path), "RUNPOD_POD_ID": "pod-abc", "RUNPOD_API_KEY": POD_KEY,
        "ROBOJEV_RUN_DIR": str(run_dir), "ROBOJEV_TERMINATE_AT": str(int(time.time()) + 3600), "ROBOJEV_START_GRACE": "3600",
        "ROBOJEV_DONE_GRACE": "3600", "ROBOJEV_WATCHDOG_POLL": "0.2", "ROBOJEV_ROOT": str(tmp_path / "root"),
    }  # fmt: skip
    env.update(extra)
    return env


def _run_until(process: subprocess.Popen, record: Path, needle: str, timeout: float = 15.0) -> str:
    deadline = time.monotonic() + timeout
    try:
        while time.monotonic() < deadline:
            text = record.read_text(encoding="utf-8") if record.is_file() else ""
            if needle in text:
                return text
            time.sleep(0.1)
        return record.read_text(encoding="utf-8") if record.is_file() else ""
    finally:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait()


def _start(env: dict[str, str], tmp_path: Path) -> tuple[subprocess.Popen, Path]:
    log = tmp_path / "watchdog.log"
    handle = log.open("ab")
    process = subprocess.Popen(["sh", str(WATCHDOG)], env=env, stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)
    handle.close()
    return process, log


def test_the_watchdog_deletes_its_own_pod_at_the_deadline(tmp_path):
    bin_dir, record = _fakes(tmp_path)
    env = _env(tmp_path, bin_dir, ROBOJEV_TERMINATE_AT=str(int(time.time()) + 2))
    (tmp_path / "run" / "started").write_text("x", encoding="utf-8")  # run이 살아 있어도 마감은 마감이다
    process, log = _start(env, tmp_path)
    started = time.monotonic()
    calls = _run_until(process, record, "runpodctl pod delete pod-abc")
    assert "runpodctl pod delete pod-abc" in calls, calls
    assert time.monotonic() - started >= 1.0, "마감 전에 지우면 안 된다"
    assert "이유: deadline" in log.read_text(encoding="utf-8")


def test_the_watchdog_waits_while_a_live_run_is_inside_its_deadline(tmp_path):
    bin_dir, record = _fakes(tmp_path)
    env = _env(tmp_path, bin_dir)
    (tmp_path / "run" / "started").write_text("x", encoding="utf-8")
    process, log = _start(env, tmp_path)
    time.sleep(2.0)
    os.killpg(process.pid, signal.SIGKILL)
    process.wait()
    calls = record.read_text(encoding="utf-8") if record.exists() else ""
    deletes = [line for line in calls.splitlines() if any(word in line for word in ("pod delete", "remove pod", "-X DELETE", "podTerminate"))]
    assert deletes == [], f"마감 전·시작 표지 있음·끝 표지 없음 — 아무것도 지우지 않는다 (읽기 점검만): {deletes}"
    assert "무장" in log.read_text(encoding="utf-8")


def test_the_watchdog_falls_back_to_the_rest_api_when_runpodctl_fails_and_keeps_the_key_out_of_its_log(tmp_path):
    bin_dir, record = _fakes(tmp_path, runpodctl_ok=False)
    env = _env(tmp_path, bin_dir, ROBOJEV_TERMINATE_AT=str(int(time.time()) - 5), ROBOJEV_API_BASE="https://api.example.invalid")
    process, log = _start(env, tmp_path)
    calls = _run_until(process, record, "curl ")
    assert "runpodctl pod delete pod-abc" in calls and "runpodctl remove pod pod-abc" in calls, "두 runpodctl 문법을 먼저 해 본다"
    assert "-X DELETE" in calls and "https://api.example.invalid/v2/pods/pod-abc" in calls, calls
    assert f"Authorization: Bearer {POD_KEY}" in calls, "pod 범위 키는 헤더로만 간다"
    assert POD_KEY not in log.read_text(encoding="utf-8"), "키가 감시 로그에 남으면 안 된다"


def test_the_watchdog_keeps_trying_when_every_route_fails(tmp_path):
    bin_dir, record = _fakes(tmp_path, runpodctl_ok=False, curl_ok=False)
    env = _env(tmp_path, bin_dir, ROBOJEV_TERMINATE_AT=str(int(time.time()) - 5))
    process, log = _start(env, tmp_path)
    calls = _run_until(process, record, "graphql")
    assert "podTerminate" in calls and "pod-abc" in calls, "넷째 길(GraphQL)까지 간다"
    assert "runpodctl pod stop pod-abc" in calls or "runpodctl stop pod pod-abc" in calls, "지우지 못하면 멈추기라도 한다"
    assert "실패" in log.read_text(encoding="utf-8")


def test_the_watchdog_deletes_a_pod_whose_run_never_started(tmp_path):
    bin_dir, record = _fakes(tmp_path)
    env = _env(tmp_path, bin_dir, ROBOJEV_START_GRACE="1")
    process, log = _start(env, tmp_path)
    calls = _run_until(process, record, "runpodctl pod delete")
    assert "runpodctl pod delete pod-abc" in calls
    assert "never-started" in log.read_text(encoding="utf-8")


def test_the_watchdog_deletes_a_finished_run_nobody_came_to_fetch(tmp_path):
    bin_dir, record = _fakes(tmp_path)
    env = _env(tmp_path, bin_dir, ROBOJEV_DONE_GRACE="1")
    (tmp_path / "run" / "started").write_text("x", encoding="utf-8")
    (tmp_path / "run" / "finished").write_text("x", encoding="utf-8")
    process, log = _start(env, tmp_path)
    calls = _run_until(process, record, "runpodctl pod delete")
    assert "runpodctl pod delete pod-abc" in calls
    assert "done-unfetched" in log.read_text(encoding="utf-8")


def test_a_malformed_deadline_deletes_at_once(tmp_path):
    bin_dir, record = _fakes(tmp_path)
    env = _env(tmp_path, bin_dir, ROBOJEV_TERMINATE_AT="tomorrow")
    process, log = _start(env, tmp_path)
    calls = _run_until(process, record, "runpodctl pod delete")
    assert "runpodctl pod delete pod-abc" in calls
    assert "bad-deadline" in log.read_text(encoding="utf-8")


def test_the_start_script_arms_the_watchdog_and_a_second_fence_before_handing_over(tmp_path):
    """컨테이너 시작 명령: 감시를 풀어 띄우고(pid 파일), 감시와 따로 마감에 지우는 둘째 울타리를 두고, 이미지의 시작으로 넘어간다."""
    bin_dir, record = _fakes(tmp_path)
    root = tmp_path / "root"
    env = _env(tmp_path, bin_dir, ROBOJEV_ROOT=str(root), ROBOJEV_TERMINATE_AT=str(int(time.time()) + 3),
               ROBOJEV_WATCHDOG_B64=base64.b64encode(watchdog_script().encode()).decode(), ROBOJEV_RUN_DIR=str(root / "run"))  # fmt: skip
    (root / "run").mkdir(parents=True)
    (root / "run" / "started").write_text("x", encoding="utf-8")
    process = subprocess.Popen(["/bin/sh", "-c", start_script()], env=env, start_new_session=True,
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)  # fmt: skip
    try:
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline and not (root / "watchdog.pid").is_file():
            time.sleep(0.05)
        pid = int((root / "watchdog.pid").read_text(encoding="utf-8").strip())
        os.kill(pid, 0)  # 살아 있다
        assert (root / "robojev-watchdog.sh").read_text(encoding="utf-8") == watchdog_script()
        assert process.poll() is None, "/start.sh가 없으면 sleep infinity로 컨테이너를 붙든다"
        calls = ""
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            calls = record.read_text(encoding="utf-8") if record.is_file() else ""
            if calls.count("runpodctl pod delete pod-abc") >= 2:
                break
            time.sleep(0.1)
        assert calls.count("runpodctl pod delete pod-abc") >= 2, f"감시와 둘째 울타리 둘 다 지운다: {calls}"
    finally:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait()


def test_the_watchdog_is_posix_sh_and_names_its_four_routes():
    text = WATCHDOG.read_text(encoding="utf-8")
    assert subprocess.run(["sh", "-n", str(WATCHDOG)], check=False).returncode == 0
    for route in ("runpodctl pod delete", "runpodctl remove pod", "-X DELETE", "podTerminate"):
        assert route in text
    code = [line for line in text.splitlines() if not line.lstrip().startswith("#")]
    assert not any("set -x" in line or "set -o xtrace" in line for line in code), "키가 셸 추적으로 새지 않게"


@pytest.mark.parametrize("name", ["ROBOJEV_TERMINATE_AT", "ROBOJEV_START_GRACE", "ROBOJEV_DONE_GRACE", "RUNPOD_POD_ID"])
def test_the_watchdog_reads_its_rules_from_the_pod_environment(name):
    assert name in WATCHDOG.read_text(encoding="utf-8")


def test_the_default_bootstrap_reports_tools_the_pod_clock_and_whether_the_watchdog_is_armed(tmp_path):
    """pod에서 처음 도는 부트스트랩(진짜 스크립트) — 가짜 uv·nvidia-smi로: uv·python 3.11·시계·GPU·감시 표지를 낸다(`set -e`에서도 끝까지)."""
    import shlex

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "uv").write_text('#!/bin/sh\nif [ "$1 $2" = "python find" ]; then echo /usr/bin/python3; exit 0; fi\nexit 1\n', encoding="utf-8")
    (bin_dir / "nvidia-smi").write_text('#!/bin/sh\necho "NVIDIA H100 80GB HBM3, 580.65.06, 81559 MiB"\n', encoding="utf-8")
    for name in ("uv", "nvidia-smi"):
        (bin_dir / name).chmod(0o755)
    root = tmp_path / "root"
    script = DEFAULT_BOOTSTRAP.replace("{root}", shlex.quote(str(root))).replace("{uv_version}", UV_VERSION).replace("{auth_wait}", "1")
    env = {"PATH": f"{bin_dir}:/usr/bin:/bin", "HOME": str(tmp_path)}
    sleeper = subprocess.Popen(["sleep", "30"])
    try:
        missing = subprocess.run(["sh", "-c", script], env=env, capture_output=True, text=True, check=False)
        root.mkdir(exist_ok=True)
        (root / "watchdog.pid").write_text(str(sleeper.pid), encoding="utf-8")
        (root / "watchdog-auth").write_text("runpodctl=ok rest=http-403 graphql=fail key=set pod=set\n", encoding="utf-8")
        armed = subprocess.run(["sh", "-c", script], env=env, capture_output=True, text=True, check=False)
    finally:
        sleeper.kill()
        sleeper.wait()
    assert missing.returncode == 0 and _parse_markers(missing.stdout)["WATCHDOG"] == "missing"
    markers = _parse_markers(armed.stdout)
    assert armed.returncode == 0, armed.stderr
    assert markers["UV"] == str(bin_dir / "uv") and markers["PY311"] == "/usr/bin/python3" and markers["WATCHDOG"] == "armed"
    assert markers["WATCHDOG_AUTH"] == "runpodctl=ok rest=http-403 graphql=fail key=set pod=set" and _parse_markers(missing.stdout)["WATCHDOG_AUTH"] == "none"
    assert abs(int(markers["NOW"]) - time.time()) < 30 and markers["GPU"].startswith("NVIDIA H100")
    assert (root / "run" / "logs").is_dir() and (root / "repo").is_dir()


def _fake_reader(tmp_path: Path, *, ok: bool) -> tuple[Path, Path]:
    """읽기 점검을 흉내 내는 가짜 — runpodctl `pod get`은 ok면 0, curl `-w`는 200/401, GraphQL 읽기는 pod id를 담은 본문/오류."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    record = tmp_path / "calls.log"
    (bin_dir / "runpodctl").write_text(
        f'#!/bin/sh\necho "runpodctl $*" >> "{record}"\ncase "$1 $2" in "pod get"|"get pod") exit {0 if ok else 1};; esac\nexit 0\n', encoding="utf-8")
    body = '{"data":{"pod":{"id":"pod-abc"}}}' if ok else '{"errors":[1]}'
    (bin_dir / "curl").write_text(
        f'#!/bin/sh\necho "curl $*" >> "{record}"\ncase "$*" in *" -w "*) printf {200 if ok else 403}; exit 0;; *"query {{ pod"*) printf \'{body}\'; exit 0;; esac\nexit 0\n',
        encoding="utf-8")
    for name in ("runpodctl", "curl"):
        (bin_dir / name).chmod(0o755)
    return bin_dir, record


@pytest.mark.parametrize("ok", [True, False])
def test_the_watchdog_checks_its_delete_routes_by_reading_its_own_pod_without_deleting(tmp_path, ok):
    """시작하자마자 같은 키·같은 주소로 자기 pod를 **읽기만** 해 지울 길이 인증되는지 적는다 (리뷰 1의 I1 c) — 지우지 않고, 키를 적지 않는다."""
    bin_dir, record = _fake_reader(tmp_path, ok=ok)
    env = _env(tmp_path, bin_dir)
    (tmp_path / "run" / "started").write_text("x", encoding="utf-8")
    process, log = _start(env, tmp_path)
    auth_file = tmp_path / "root" / "watchdog-auth"
    try:
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline and not auth_file.is_file():
            time.sleep(0.05)
    finally:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait()
    auth = auth_file.read_text(encoding="utf-8").strip()
    expected = "runpodctl=ok rest=ok graphql=ok key=set pod=set" if ok else "runpodctl=fail rest=http-403 graphql=fail key=set pod=set"
    assert auth == expected
    calls = record.read_text(encoding="utf-8")
    assert "pod get pod-abc" in calls and "/v2/pods/pod-abc" in calls and "query { pod" in calls
    assert not any(word in calls for word in ("pod delete", "remove pod", "-X DELETE", "podTerminate")), "점검은 읽기만 한다"
    assert POD_KEY not in auth and POD_KEY not in log.read_text(encoding="utf-8")


def test_every_curl_in_the_watchdog_and_the_second_fence_names_itself_and_retries():
    """기본 User-Agent는 api.runpod.io 앞의 Cloudflare가 막는다(Error 1010) — 감시와 둘째 울타리의 curl은 모두 `-A`와 `--retry`를 쓴다."""
    import re

    for text in (WATCHDOG.read_text(encoding="utf-8"), start_script()):
        for line in re.findall(r"curl [^;\n]*", text):
            if line.startswith("curl >/dev/null") or "command -v" in line:
                continue
            assert "-A " in line and "--retry" in line, line


def test_a_terminate_request_makes_the_watchdog_delete_and_record_the_route(tmp_path):
    """`--prove-self-delete`의 길: 실행기가 `terminate-now`를 두면 감시가 **자기 환경**으로 지우고 어느 길이 됐는지 적는다."""
    bin_dir, record = _fakes(tmp_path)
    env = _env(tmp_path, bin_dir)
    (tmp_path / "run" / "started").write_text("x", encoding="utf-8")
    (tmp_path / "root").mkdir(exist_ok=True)
    (tmp_path / "root" / "terminate-now").write_text("", encoding="utf-8")
    process, log = _start(env, tmp_path)
    calls = _run_until(process, record, "runpodctl pod delete pod-abc")
    assert "runpodctl pod delete pod-abc" in calls
    assert "이유: requested" in log.read_text(encoding="utf-8")
    assert (tmp_path / "root" / "watchdog-route").read_text(encoding="utf-8").startswith("ok:runpodctl pod delete")


def test_the_second_fence_falls_back_to_curl_and_keeps_retrying(tmp_path):
    """둘째 울타리(감시와 따로 도는 sleep): runpodctl이 안 되면 User-Agent를 밝힌 REST DELETE로, 그것도 안 되면 다시 (한 번에 끝나지 않는다)."""
    bin_dir, record = _fakes(tmp_path, runpodctl_ok=False, curl_ok=False)
    root = tmp_path / "root"
    root.mkdir()
    env = _env(tmp_path, bin_dir, ROBOJEV_TERMINATE_AT=str(int(time.time()) + 1), ROBOJEV_FENCE_RETRY="0.3",
               ROBOJEV_WATCHDOG_B64="", ROBOJEV_ROOT=str(root))  # 감시는 비어 있다 — 둘째 울타리만 본다
    process = subprocess.Popen(["/bin/sh", "-c", start_script()], env=env, start_new_session=True,
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)  # fmt: skip
    try:
        deadline = time.monotonic() + 15
        calls = ""
        while time.monotonic() < deadline:
            calls = record.read_text(encoding="utf-8") if record.is_file() else ""
            if calls.count("-X DELETE") >= 2:
                break
            time.sleep(0.1)
    finally:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait()
    deletes = [line for line in calls.splitlines() if "-X DELETE" in line]
    assert len(deletes) >= 2, calls
    assert all("-A robo-jev-fence/" in line and "/v2/pods/pod-abc" in line for line in deletes)
