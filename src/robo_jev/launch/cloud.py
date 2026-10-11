"""RunPod pod 한 대의 수명 — 만들기 → 접속 → 올리기 → 돌리기 → 가져오기 → **지우기** (Task cloud1).

`scripts/launch_run.py launch --yes`가 부른다. 돈 안전 장치는 다섯이고 이 모듈이 그 순서를 지킨다::

    (iv) 예상 비용·상한·장부를 찍는다 → --yes가 없으면 여기서 끝 (아무것도 만들지 않는다)
    (v)  키는 git 밖 파일에서만 (providers.runpod.load_api_key) — 명세·로그·예외에 남지 않는다
         카탈로그 단가 × GPU 수 ≠ 명세의 hourly_usd → 거절 (만들지 않는다)
         단계 장부 예약: 누적 + 이 run의 상한 ≤ 단계 상한($150) — 넘으면 거절 (만들지 않는다)
         만들기 (POST는 재시도하지 않는다; 응답이 애매하면 같은 이름의 pod를 찾아 지운다)
    (iii) 만들 때 **컨테이너 시작 명령에 자기 종료 감시를 심는다** — 절대 마감(ROBOJEV_TERMINATE_AT)에 runpodctl/API로 자기 pod를 지운다.
          실행기가 그 뒤 어디서 죽어도 pod는 마감을 넘지 못한다. 접속하면 감시가 살아 있는지 먼저 본다 — 없으면 곧바로 지운다
         만든 pod의 cost ≠ hourly_usd → 곧바로 지운다
    (i)  원격 러너가 마감(= 감시 마감 − 가져갈 몫)에 스스로 멈추고, 실행기도 그 시각을 잰다
    (ii) 끝나거나 실패하면 결과를 가져오고(sha256 대조, 실패하면 마감까지 다시) **반드시 지운다** — finally. 지우기가 확인되지 않으면
         명세는 `unknown`, 장부는 열린 채(상한 전부를 센다), 종료 코드 3

시계: 마감은 실행기의 epoch 초로 정해 pod의 환경 변수로 넘긴다. 접속하면 pod의 시계를 재서 어긋남이 크면(pod가 늦으면 감시가 늦게 지운다)
멈춘다.
"""

from __future__ import annotations

import base64
import contextlib
import gzip
import io
import json
import shlex
import signal
import subprocess
import tarfile
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from robo_jev.launch.manifest import REPO, Budget, budget_deadline_hours, note_state, now, save_manifest, sha256_of
from robo_jev.launch.providers.base import Backend, Handle
from robo_jev.launch.providers.runpod import PodView, RunPodError, pod_request

__all__ = [
    "DEFAULT_BOOTSTRAP",
    "DEFAULT_SSH_KEY",
    "LaunchNotApproved",
    "LauncherInterrupted",
    "PodContext",
    "build_payload",
    "down",
    "ensure_ssh_key",
    "finish",
    "launch_pod",
    "pod_lifetime",
    "run_cap_usd",
    "start_script",
    "watchdog_script",
]

WATCHDOG = REPO / "infra" / "runpod-watchdog.sh"
DEFAULT_SSH_KEY = Path("~/.config/robo-jev/runpod_ed25519")
DEFAULT_ROOT = "/root/robojev"
UV_VERSION = "0.11.21"
#: pod의 부트스트랩 — 도구(rsync·uv·python 3.11)를 맞추고, 시계와 GPU와 **감시가 살아 있는지**를 알린다. `{root}`만 채운다.
DEFAULT_BOOTSTRAP = r"""set -e
R={root}
mkdir -p "$R/run/logs" "$R/repo"
if ! command -v rsync >/dev/null 2>&1; then (apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq rsync) >/dev/null 2>&1 || true; fi
command -v rsync >/dev/null 2>&1 || { echo "BOOTSTRAP_FAIL=rsync"; exit 3; }
UV=$(command -v uv || true)
for c in /usr/local/bin/uv "$HOME/.local/bin/uv"; do [ -z "$UV" ] && [ -x "$c" ] && UV="$c"; done
if [ -z "$UV" ]; then
  (curl -LsSf https://astral.sh/uv/{uv_version}/install.sh | env UV_INSTALL_DIR=/usr/local/bin INSTALLER_NO_MODIFY_PATH=1 sh) >/dev/null 2>&1 \
    || python3 -m pip install -q "uv=={uv_version}" >/dev/null 2>&1 || true
  UV=$(command -v uv || true)
  for c in /usr/local/bin/uv "$HOME/.local/bin/uv"; do [ -z "$UV" ] && [ -x "$c" ] && UV="$c"; done
fi
[ -n "$UV" ] || { echo "BOOTSTRAP_FAIL=uv"; exit 3; }
PY=$("$UV" python find 3.11 2>/dev/null || true)
if [ -z "$PY" ]; then "$UV" python install 3.11 >/dev/null 2>&1 || true; PY=$("$UV" python find 3.11 2>/dev/null || true); fi
[ -n "$PY" ] || { echo "BOOTSTRAP_FAIL=python3.11"; exit 3; }
echo "UV=$UV"
echo "PY311=$PY"
echo "NOW=$(date +%s)"
echo "GPU=$(nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader 2>/dev/null | head -n1)"
if [ -f "$R/watchdog.pid" ] && kill -0 "$(cat "$R/watchdog.pid")" 2>/dev/null; then echo "WATCHDOG=armed"; else echo "WATCHDOG=missing"; fi
"""


class LaunchNotApproved(RuntimeError):
    """`--yes`가 없다 — 예상 비용만 찍고 아무것도 만들지 않았다."""


class LauncherInterrupted(KeyboardInterrupt):
    """실행기가 SIGTERM·SIGHUP을 받았다 — Ctrl-C와 같은 정리 길을 탄다."""


# --------------------------------------------------------------------------
# 상한의 산술
# --------------------------------------------------------------------------


def pod_lifetime(budget: Budget) -> tuple[float, str]:
    """pod가 살아 있어도 되는 시간(h)과 그것을 정한 상한. 구속하는 상한이 없으면 거절 — pod에는 `--no-cap`이 없다."""
    hours, binding = budget_deadline_hours(budget)
    if hours is None or binding is None:
        raise ValueError("pod를 띄우려면 구속하는 상한이 있어야 한다 (max_wall_hours / max_usd / max_gpu_hours) — 상한 없는 pod는 만들지 않는다")
    return float(hours), str(binding)


def run_cap_usd(budget: Budget) -> float:
    """이 run이 쓸 수 있는 최대 달러 = 단가 × 수명. 장부는 이것을 예약한다."""
    hours, _ = pod_lifetime(budget)
    return round(hours * float(budget.hourly_usd), 6)


# --------------------------------------------------------------------------
# 감시와 시작 명령
# --------------------------------------------------------------------------


def watchdog_script() -> str:
    return WATCHDOG.read_text(encoding="utf-8")


def start_script() -> str:
    """컨테이너 시작 명령 (`/bin/sh -c`). 감시를 먼저 띄우고(pid 파일), 감시가 못 떠도 마감에 지우는 둘째 울타리(sleep)를 두고, 이미지의
    원래 시작(``/start.sh`` — sshd가 PUBLIC_KEY로 뜬다)을 `exec`한다."""
    return (
        'R="${ROBOJEV_ROOT:-/root/robojev}"; mkdir -p "$R"; '
        'printf "%s" "$ROBOJEV_WATCHDOG_B64" | base64 -d > "$R/robojev-watchdog.sh" && chmod 700 "$R/robojev-watchdog.sh"; '
        'nohup sh "$R/robojev-watchdog.sh" >> "$R/watchdog.log" 2>&1 & echo $! > "$R/watchdog.pid"; '
        '( S=$(( ${ROBOJEV_TERMINATE_AT:-0} - $(date +%s) )); [ "$S" -gt 0 ] && sleep "$S"; '
        'runpodctl pod delete "$RUNPOD_POD_ID" || runpodctl remove pod "$RUNPOD_POD_ID" ) >> "$R/watchdog.log" 2>&1 & '
        'if [ -x /start.sh ]; then exec /start.sh; else exec sleep infinity; fi'
    )


def pod_env(*, run_id: str, root: str, public_key: str, terminate_at: int, start_grace: float, done_grace: float) -> dict[str, str]:
    """pod의 환경 변수 — 감시의 마감·유예와 SSH 공개 키. **API 키는 넣지 않는다**(pod에는 RunPod가 pod 범위 키를 따로 넣는다)."""
    return {
        "PUBLIC_KEY": public_key.strip(),
        "ROBOJEV_ROOT": root,
        "ROBOJEV_RUN_ID": run_id,
        "ROBOJEV_RUN_DIR": f"{root}/run",
        "ROBOJEV_TERMINATE_AT": str(int(terminate_at)),
        "ROBOJEV_START_GRACE": str(int(start_grace)),
        "ROBOJEV_DONE_GRACE": str(int(done_grace)),
        "ROBOJEV_WATCHDOG_B64": base64.b64encode(watchdog_script().encode("utf-8")).decode("ascii"),
    }


def ensure_ssh_key(path: str | Path | None = None) -> tuple[Path, str]:
    """pod 전용 SSH 키(없으면 만든다 — git 밖, 600). 돌려주는 것: (개인키 경로, 공개키 글)."""
    private = Path(path if path is not None else DEFAULT_SSH_KEY).expanduser()
    public = private.with_name(private.name + ".pub")
    if not private.is_file():
        private.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["ssh-keygen", "-q", "-t", "ed25519", "-N", "", "-C", "robojev-runpod", "-f", str(private)], check=True)
    private.chmod(0o600)
    return private, public.read_text(encoding="utf-8").strip()


# --------------------------------------------------------------------------
# 코드 묶음
# --------------------------------------------------------------------------


def _git(*args: str, cwd: Path = REPO) -> str:
    return subprocess.run(["git", "-C", str(cwd), *args], capture_output=True, text=True, check=True).stdout.strip()


def build_payload(ref: str, out: Path, *, repo: Path = REPO) -> dict[str, Any]:
    """`git archive <ref>`을 gzip(mtime 0 — 같은 커밋이면 같은 바이트)으로 묶는다. 커밋된 것만 실린다 — 고치는 중인 파일은 가지 않는다."""
    commit = _git("rev-parse", "--verify", f"{ref}^{{commit}}", cwd=repo)
    tree = _git("rev-parse", f"{commit}^{{tree}}", cwd=repo)
    raw = subprocess.run(["git", "-C", str(repo), "archive", "--format=tar", commit], capture_output=True, check=True).stdout
    out.parent.mkdir(parents=True, exist_ok=True)
    buffer = io.BytesIO()
    with gzip.GzipFile(fileobj=buffer, mode="wb", mtime=0) as handle:
        handle.write(raw)
    out.write_bytes(buffer.getvalue())
    with tarfile.open(out, "r:gz") as archive:
        names = archive.getnames()
    parents = _git("rev-list", "--parents", "-n", "1", commit, cwd=repo).split()[1:]
    return {"ref": ref, "commit": commit, "tree": tree, "parents": parents, "payload": out.name, "payload_sha256": sha256_of(out),
            "payload_bytes": out.stat().st_size, "files": len(names)}  # fmt: skip


# --------------------------------------------------------------------------
# 실행기 쪽 흐름
# --------------------------------------------------------------------------


@dataclass
class PodContext:
    """주입할 수 있는 것들 — 검사는 가짜 공급자·local 백엔드·가짜 시계를 넣는다."""

    provider: Any  # RunPodProvider (또는 같은 메서드를 가진 가짜)
    ledger: Any  # Ledger
    backend_factory: Callable[[PodView, dict[str, Any]], Backend]
    clock: Callable[[], float] = time.time
    sleep: Callable[[float], None] = time.sleep
    out: Callable[[str], None] = print
    ssh_key: str | Path | None = None
    poll_seconds: float = 20.0
    bootstrap_timeout: float = 900.0
    log: list[str] = field(default_factory=list)

    def say(self, message: str) -> None:
        line = f"[cloud {now()}] {message}"
        self.log.append(line)
        self.out(line)


@contextlib.contextmanager
def _signals_as_interrupt() -> Iterator[None]:
    """SIGTERM·SIGHUP을 Ctrl-C처럼 — 정리(finally: 지우기)가 돈다. 메인 스레드가 아니면 그대로 둔다."""
    def _raise(signum: int, _frame: Any) -> None:
        raise LauncherInterrupted(f"신호 {signum}")

    saved = {}
    try:
        for sig in (signal.SIGTERM, signal.SIGHUP):
            saved[sig] = signal.signal(sig, _raise)
    except ValueError:  # pragma: no cover - 메인 스레드가 아니다
        saved = {}
    try:
        yield
    finally:
        for sig, handler in saved.items():
            signal.signal(sig, handler)


def _pod_block(manifest: dict[str, Any]) -> dict[str, Any]:
    return manifest["provider"].setdefault("pod", {})


def _save(manifest: dict[str, Any], manifest_path: str | Path) -> None:
    save_manifest(manifest_path, manifest)


def _parse_markers(text: str) -> dict[str, str]:
    out = {}
    for line in text.splitlines():
        key, sep, value = line.partition("=")
        if sep and key.isupper() and key.replace("_", "").isalnum():
            out[key] = value.strip()
    return out


def _wait_ssh(backend: Backend, ctx: PodContext, *, deadline: float, poll: float = 5.0) -> int:
    """RUNNING이 된 뒤에도 sshd는 몇 초 늦게 뜬다(이미지의 /start.sh가 PUBLIC_KEY로 띄운다) — 첫 접속 실패로 pod를 버리지 않게 기다린다.

    돌려주는 것은 시도 횟수. 기한까지 안 되면 RuntimeError (호출자의 finally가 지운다)."""
    attempt = 0
    while True:
        attempt += 1
        result = backend.shell("true", timeout=30)
        if result.ok:
            return attempt
        if ctx.clock() >= deadline:
            raise RuntimeError(f"SSH가 {attempt}번 시도 동안 열리지 않았다 ({result.stderr.strip()[:200]})")
        ctx.sleep(poll)


def _bootstrap(backend: Backend, manifest: dict[str, Any], ctx: PodContext) -> dict[str, str]:
    provider = manifest["provider"]
    script = provider.get("bootstrap") or DEFAULT_BOOTSTRAP
    script = script.replace("{root}", shlex.quote(provider["remote_root"])).replace("{uv_version}", UV_VERSION)
    result = backend.shell(script, timeout=ctx.bootstrap_timeout)
    markers = _parse_markers(result.stdout)
    if not result.ok or "BOOTSTRAP_FAIL" in markers:
        raise RuntimeError(f"부트스트랩 실패 (코드 {result.returncode}): {markers.get('BOOTSTRAP_FAIL') or result.stderr.strip()[:400]}")
    return markers


def _push_payload(backend: Backend, manifest: dict[str, Any], bundle: Path, ctx: PodContext) -> None:
    """코드 묶음 → 원격 repo에 풀기, 입력 파일 → 원격 repo의 같은 자리, 묶음(명세) → run/bundle. 그다음 원격에서 sha256을 다시 잰다."""
    job = manifest["job"]
    repo_backend = backend.with_remote_dir(backend.remote_repo)
    code = job["code"]
    repo_backend.push_file(bundle / code["payload"], f".robojev/{code['payload']}")
    extract = (
        f"cd {shlex.quote(backend.remote_repo)} && tar -xzf .robojev/{shlex.quote(code['payload'])} && "
        f"sha256sum .robojev/{shlex.quote(code['payload'])}"
    )
    result = backend.shell(extract).require("코드 묶음 풀기")
    if code["payload_sha256"] not in result.stdout:
        raise RuntimeError(f"코드 묶음의 원격 sha256이 다르다 ({result.stdout.strip()[:120]})")
    for item in job.get("uploads") or []:
        repo_backend.push_file(REPO / item["path"], item["path"])
    check = backend.shell(f"cd {shlex.quote(backend.remote_repo)} && sha256sum " + " ".join(shlex.quote(i["path"]) for i in job.get("uploads") or []))
    remote = {}
    for line in check.stdout.splitlines():
        digest, _, path = line.partition("  ")
        remote[path.strip()] = digest.strip()
    bad = [item["path"] for item in job.get("uploads") or [] if remote.get(item["path"]) != item["sha256"]]
    if bad:
        raise RuntimeError(f"올린 파일의 원격 sha256이 다르다: {bad}")
    ctx.say(f"올림: 코드 {code['commit'][:12]} ({code['payload_bytes']:,} B) + 입력 {len(job.get('uploads') or [])}개 — 원격 sha256 대조 통과")
    (bundle / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    backend.push(bundle, "bundle")


def _start_runner(backend: Backend, manifest: dict[str, Any]) -> Handle:
    argv = [*shlex.split(backend.remote_python), "-m", "robo_jev.launch.runner", "--bundle", backend.path("bundle"), "--repo", backend.remote_repo]
    unit = str(manifest["run_id"])
    backend.reset_unit(unit)
    backend.shell(f"rm -f {shlex.quote(backend.path('state.json'))} {shlex.quote(backend.path('finished'))}")
    return backend.start(argv, unit=unit, log_relpath="logs/runner.log", env=manifest["job"]["runtime"].get("env") or None)


def _remote_state(backend: Backend) -> dict[str, Any] | None:
    try:
        text = backend.read_text("state.json")
    except Exception:  # noqa: BLE001 — 끊긴 연결도 '모른다'다
        return None
    if not text or not text.strip():
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def _watch(backend: Backend, handle: Handle, manifest: dict[str, Any], manifest_path: str | Path, ctx: PodContext) -> dict[str, Any] | None:
    """러너가 끝날 때까지 본다 (실행기 쪽 상한: 러너 마감 + 유예에 정지 요청, 가져오기 마감에 손을 뗀다)."""
    pod = _pod_block(manifest)
    runner_deadline = float(pod["runner_deadline_epoch"])
    fetch_deadline = float(pod["fetch_deadline_epoch"])
    stop_sent = False
    last_line = None
    misses = 0
    gone = 0
    while True:
        state = _remote_state(backend)
        if state is None:
            misses += 1
            if misses >= 3 and ctx.provider.describe(pod["id"]) is None:
                ctx.say("pod가 없어졌다 (404) — 감시나 사람이 지웠다")
                pod["vanished_at"] = now()
                return None
        else:
            misses = 0
            line = f"{state.get('state')} · 단계 {state.get('step')}/{state.get('max_steps')} {state.get('current') or ''} · {state.get('elapsed_seconds')} s · ${state.get('estimated_usd')}"
            if line != last_line:
                ctx.say(line)
                last_line = line
            manifest["progress"] = {k: state.get(k) for k in ("step", "max_steps", "loss", "elapsed_seconds", "gpu_hours", "estimated_usd", "seconds_per_step")}
            manifest["progress"].update(updated_at=state.get("updated_at"), observed_at=now(), alive=True)
            if state.get("state") in ("completed", "failed", "cancelled"):
                return state
            # 상태는 running인데 러너 프로세스가 없다 — 두 번 연속이면 죽은 것이다 (마감까지 기다리지 않는다)
            try:
                alive = backend.alive(handle)
            except Exception:  # noqa: BLE001 — 끊긴 연결은 '모른다'
                alive = True
            gone = 0 if alive else gone + 1
            if gone >= 2:
                ctx.say("상태는 running인데 러너가 없다 — 러너가 죽었다. 있는 것을 가져오고 지운다")
                return {**state, "state": "failed", "exit_reason": "process_vanished"}
        moment = ctx.clock()
        if moment >= runner_deadline + 120 and not stop_sent:
            ctx.say("러너가 마감을 2분 넘겼다 — 정지를 요청한다 (실행기 쪽 상한)")
            backend.stop(handle)
            stop_sent = True
        if moment >= fetch_deadline:
            ctx.say("가져오기 마감 — 더 기다리지 않고 있는 것을 가져온다")
            return state
        ctx.sleep(ctx.poll_seconds)


def _fetch_until(backend: Backend, manifest: dict[str, Any], manifest_path: str | Path, dest: Path, ctx: PodContext, *, deadline: float) -> bool:
    """가져오기를 마감까지 다시 한다. 성공(모든 sha256 일치)이면 참."""
    from robo_jev.launch.launcher import fetch

    attempt = 0
    while True:
        attempt += 1
        try:
            fetch(manifest, backend, dest=dest, manifest_path=manifest_path, strict=False)
            got = [entry for entry in manifest["artifacts"] if entry.get("local")]
            ctx.say(f"가져옴 {len(got)}개 — sha256 일치 {sum(1 for e in got if e.get('sha256_match'))}/{len(got)} → {dest}")
            return True
        except Exception as error:  # noqa: BLE001 — 실패해도 지우기는 간다
            ctx.say(f"가져오기 {attempt}번째 실패: {type(error).__name__}: {str(error)[:300]}")
            if ctx.clock() + 15 >= deadline:
                ctx.say("가져오기 마감 — 결과 없이 지운다 (돈이 먼저다)")
                return False
            ctx.sleep(15)


def _teardown(manifest: dict[str, Any], manifest_path: str | Path, ctx: PodContext, *, reason: str) -> bool:
    """pod를 지우고 **확인한다**. 확인되면 장부를 닫고 참, 아니면 명세 `unknown`·장부 열린 채로 두고 거짓."""
    pod = _pod_block(manifest)
    pod_id = pod.get("id")
    if not pod_id:
        return True
    ctx.say(f"pod {pod_id}를 지운다 — {reason}")
    confirmed = bool(ctx.provider.destroy(pod_id))
    pod["deleted_at"] = now()
    pod["delete_confirmed"] = confirmed
    # 끝 시각: 지금, 다만 감시 마감(+2분)보다 늦지 않게 — `finish`를 며칠 뒤에 돌려도 감시가 그 전에 지웠다(그래서 상한까지만 센다)
    end = ctx.clock()
    if pod.get("terminate_at_epoch") is not None:
        end = min(end, float(pod["terminate_at_epoch"]) + 120.0)
    hours = max(end - float(pod.get("create_requested_epoch") or end), 0.0) / 3600.0
    spent = round(hours * float(manifest["budget"]["hourly_usd"]), 6)
    pod["spent_usd_estimate"] = spent
    if confirmed:
        ctx.ledger.close(manifest["run_id"], spent_usd_estimate=spent, note=f"지움 확인 ({reason})")
        ctx.say(f"지움 확인 · 쓴 돈 추정 ${spent:.4f} (단가 × 만든 때부터 지운 때까지)")
    else:
        manifest["exit_reason"] = "pod-delete-unconfirmed"
        note_state(manifest, "unknown", note=f"pod {pod_id} 지우기를 확인하지 못했다 — 돈이 계속 나갈 수 있다. 사람이 콘솔에서 확인한다")
        ctx.ledger.mark_unconfirmed(manifest["run_id"], note=f"pod {pod_id} 지우기 미확인 ({reason})")
        ctx.say(f"⚠ pod {pod_id} 지우기를 확인하지 못했다 — 콘솔에서 확인하고 `launch_run.py down --manifest {manifest_path}`")
    _save(manifest, manifest_path)
    return confirmed


def launch_pod(
    manifest: dict[str, Any],
    *,
    manifest_path: str | Path,
    ctx: PodContext,
    yes: bool,
    dest: str | Path | None = None,
    watch: bool = True,
    keep_on_interrupt: bool = False,
) -> dict[str, Any]:
    """dry-run을 찍고, `--yes`면 pod를 만들어 job을 돌리고 결과를 가져온 뒤 지운다. 돌려주는 것은 명세.

    **지우기의 규칙** (finally): pod를 만들었으면 지운다. 예외는 둘뿐이다 — 러너가 뜬 뒤 `--no-watch`로 손을 뗀 경우와, 러너가 뜬 뒤
    `--keep-on-interrupt`로 중단한 경우. 둘 다 pod 안의 감시가 마감에 지우고, `finish`가 다시 붙는다.
    """
    from robo_jev.launch.jobs import format_plan, plan_job
    from robo_jev.launch.launcher import bundle_dir

    provider_spec = manifest["provider"]
    budget = Budget.from_manifest(manifest)
    hours, binding = pod_lifetime(budget)
    cap_usd = run_cap_usd(budget)
    ctx.out(format_plan(plan_job(manifest, ledger=ctx.ledger)))
    if not yes:
        raise LaunchNotApproved("--yes가 없다 — 위의 예상 비용을 보고 승인할 때만 띄운다. 아무것도 만들지 않았다")
    if manifest["state"] != "prepared" or (manifest.get("provider") or {}).get("pod", {}).get("id"):
        raise RuntimeError(f"{manifest['run_id']}: 상태가 {manifest['state']}다 — 새 run id로 다시 prepare한다 (한 명세로 pod를 두 번 만들지 않는다)")
    run_id = manifest["run_id"]
    bundle = bundle_dir(manifest_path)
    dest = Path(dest) if dest is not None else Path(manifest_path).parent / f"{Path(manifest_path).stem}-fetched"

    # 1) 단가 — 읽기만. 다르면 거절 (장부도 건드리지 않는다)
    price = ctx.provider.check_price(provider_spec, budget.hourly_usd)
    ctx.say(f"단가 대조 통과: 카탈로그 ${price['catalog_hourly_usd']:.4f}/h = 명세 ${budget.hourly_usd:.4f}/h")
    # 2) 장부 — 넘으면 거절
    admitted = ctx.ledger.reserve(run_id, cap_usd=cap_usd, hourly_usd=budget.hourly_usd, manifest=str(manifest_path), phase=manifest["budget"].get("phase"))
    ctx.say(f"장부 예약: ${cap_usd:.2f} (단계 누적 {admitted['after_usd']:.2f} / {admitted['phase']['cap_usd']:.2f})")
    try:
        key_path, public_key = ensure_ssh_key(ctx.ssh_key or provider_spec.get("ssh_key"))
    except BaseException:
        ctx.ledger.release(run_id, reason="SSH 키를 준비하지 못했다")
        raise

    pod = _pod_block(manifest)
    created = runner_started = leave_pod = finished = False
    view: PodView | None = None
    with _signals_as_interrupt():
        try:
            requested = ctx.clock()
            terminate_at = int(requested + hours * 3600.0 - float(provider_spec.get("delete_margin_minutes") or 2.0) * 60.0)
            fetch_reserve = float(provider_spec.get("fetch_reserve_minutes") or 4.0) * 60.0
            pod.update(create_requested_epoch=requested, terminate_at_epoch=terminate_at, lifetime_hours=hours, binding=binding,
                       runner_deadline_epoch=terminate_at - fetch_reserve, fetch_deadline_epoch=terminate_at - 60.0, price_check=price,
                       ssh_key=str(key_path))  # fmt: skip
            env = pod_env(run_id=run_id, root=provider_spec["remote_root"], public_key=public_key, terminate_at=terminate_at,
                          start_grace=float(provider_spec.get("start_grace_minutes") or 45.0) * 60.0,
                          done_grace=float(provider_spec.get("done_grace_minutes") or 20.0) * 60.0)  # fmt: skip
            body = pod_request(provider_spec, name=run_id, env=env, start_script=start_script())
            try:
                view = ctx.provider.create(body)
            except BaseException as error:
                ambiguous = not isinstance(error, RunPodError) or error.ambiguous
                if not ambiguous:
                    ctx.ledger.release(run_id, reason=f"만들기 거절: {str(error)[:200]}")
                    raise
                ctx.say(f"만들기 응답이 애매하다 ({type(error).__name__}: {str(error)[:200]}) — 같은 이름의 pod를 찾아 지운다")
                try:
                    swept = ctx.provider.sweep_name(run_id)
                except BaseException as sweep_error:  # noqa: BLE001
                    pod["sweep_error"] = f"{type(sweep_error).__name__}: {str(sweep_error)[:200]}"
                    note_state(manifest, "unknown", note="만들기 응답도, 같은 이름 pod 찾기도 실패했다 — 콘솔에서 확인한다 (장부는 상한 전부를 센 채로 둔다)")
                    manifest["exit_reason"] = "create-ambiguous"
                    _save(manifest, manifest_path)
                    raise error from sweep_error
                pod["swept_after_ambiguous_create"] = [[pod_id, ok] for pod_id, ok in swept]
                if not swept:
                    ctx.ledger.release(run_id, reason="만들기 응답이 애매했지만 같은 이름의 pod가 없었다")
                    raise
                created = True
                pod["id"] = swept[0][0]
                ctx.ledger.activate(run_id, pod_id=pod["id"])
                raise
            created = True
            pod.update(id=view.pod_id, created_at=now(), status=view.status, cost_per_hour=view.cost)
            manifest["backend"]["instance_id"] = view.pod_id
            manifest["backend"]["provider"] = "runpod"
            ctx.ledger.activate(run_id, pod_id=view.pod_id)
            _save(manifest, manifest_path)  # pod id를 디스크에 먼저 — 실행기가 여기서 죽어도 `down`이 찾는다
            ctx.say(f"pod {view.pod_id} 생성 · 감시 마감 {time.strftime('%H:%M:%S', time.localtime(terminate_at))} ({hours * 60:.0f}분, {binding})")
            ctx.provider.check_created_price(view, budget.hourly_usd)
            boot_timeout = float(provider_spec.get("boot_timeout_minutes") or 15.0) * 60.0
            view = ctx.provider.wait_ready(view.pod_id, timeout=boot_timeout, expected_hourly_usd=budget.hourly_usd)
            pod.update(status=view.status, cost_per_hour=view.cost, host=view.host, port=view.port, cuda_version=view.cuda_version,
                       data_center=view.data_center, ready_at=now(), ready_seconds=round(ctx.clock() - requested, 1))  # fmt: skip
            ctx.say(f"준비됨 {view.host}:{view.port} · CUDA {view.cuda_version} · {view.data_center} · 부팅 {pod['ready_seconds']} s")
            backend = ctx.backend_factory(view, manifest)
            for key in ("host", "user", "port"):
                manifest["backend"][key] = getattr(backend, key, None)
            pod["ssh_attempts"] = _wait_ssh(backend, ctx, deadline=requested + boot_timeout + 300.0)
            markers = _bootstrap(backend, manifest, ctx)
            pod["bootstrap"] = {k: markers.get(k) for k in ("UV", "PY311", "NOW", "GPU", "WATCHDOG")}
            pod["bootstrap_seconds"] = round(ctx.clock() - requested - pod["ready_seconds"], 1)
            if markers.get("WATCHDOG") != "armed":
                raise RuntimeError("pod 안의 자기 종료 감시가 살아 있지 않다 — 이 pod로는 일하지 않는다(실행기가 죽으면 남는다)")
            skew = float(markers.get("NOW") or ctx.clock()) - ctx.clock()
            pod["clock_skew_seconds"] = round(skew, 1)
            if skew < -300:
                raise RuntimeError(f"pod 시계가 {-skew:.0f} s 늦다 — 감시가 그만큼 늦게 지운다. 일하지 않는다")
            if markers.get("PY311"):
                backend.remote_python = f"env PYTHONPATH={shlex.quote(backend.remote_repo + '/src')} {shlex.quote(markers['PY311'])}"
                manifest["backend"]["remote_python"] = backend.remote_python
            if markers.get("UV"):
                manifest["job"]["runtime"]["uv"] = markers["UV"]
            # 러너의 마감은 pod의 시계로 — pod가 앞서 있으면 그만큼 옮긴다(늦으면 그대로 — 일찍 멈추는 쪽)
            manifest["job"]["deadline_epoch"] = pod["runner_deadline_epoch"] + max(skew, 0.0)
            manifest["job"]["deadline_limit"] = f"pod-lifetime({binding})"
            _push_payload(backend, manifest, bundle, ctx)
            pod["uploaded_seconds"] = round(ctx.clock() - requested, 1)
            handle = _start_runner(backend, manifest)
            runner_started = True
            manifest["backend"].update(handle.as_dict())
            note_state(manifest, "running", note=f"pod {view.pod_id} · {handle.launcher} · 감시 마감 {terminate_at}")
            _save(manifest, manifest_path)
            ctx.say(f"러너 시작 ({handle.launcher} {handle.unit or handle.pid}) · 부팅부터 {pod['uploaded_seconds']} s")
            if not watch:
                ctx.say(f"지켜보지 않는다(--no-watch) — pod는 감시 마감 {terminate_at}에 스스로 지워진다. 결과는 `finish --manifest {manifest_path}`")
                leave_pod = finished = True
                return manifest
            remote = _watch(backend, handle, manifest, manifest_path, ctx)
            if remote is not None:
                manifest["exit_reason"] = remote.get("exit_reason")
                if remote.get("budget_limit"):
                    manifest["budget_limit"] = remote["budget_limit"]
                note_state(manifest, str(remote.get("state")), note=f"원격이 끝냈다: {remote.get('state')}({remote.get('exit_reason')})")
            _save(manifest, manifest_path)
            if pod.get("vanished_at") is None:
                _fetch_until(backend, manifest, manifest_path, dest, ctx, deadline=float(pod["fetch_deadline_epoch"]))
            finished = True
            return manifest
        except KeyboardInterrupt:
            if runner_started and keep_on_interrupt:
                ctx.say(f"중단 — pod {pod.get('id')}를 남긴다(--keep-on-interrupt). 감시 마감 {pod.get('terminate_at_epoch')}에 스스로 지워진다; "
                        f"`finish --manifest {manifest_path}`로 가져오고 지운다")  # fmt: skip
                leave_pod = True
                raise
            ctx.say("중단 — 지운다 (러너가 떴으면 그 전에 잠깐 가져와 본다)")
            if runner_started and view is not None:
                with contextlib.suppress(Exception):
                    _fetch_until(ctx.backend_factory(view, manifest), manifest, manifest_path, dest, ctx, deadline=ctx.clock() + 60.0)
            raise
        finally:
            if created and not leave_pod:
                _teardown(manifest, manifest_path, ctx, reason="끝" if finished else "실패·중단")


def finish(manifest: dict[str, Any], *, manifest_path: str | Path, ctx: PodContext, dest: str | Path, wait: bool = True) -> dict[str, Any]:
    """다시 붙기 — (기다리고) 가져오고 **지운다**. `--no-watch`나 중단 뒤에 쓴다."""
    pod = _pod_block(manifest)
    if not pod.get("id"):
        raise RuntimeError(f"{manifest['run_id']}: 이 명세에는 pod가 없다")
    view = None
    try:
        raw = ctx.provider.describe(pod["id"])
        if raw is None:
            ctx.say(f"pod {pod['id']}는 이미 없다 (감시가 지웠거나 사람이 지웠다)")
        else:
            view = PodView.of(raw)
            if view.status == "RUNNING" and view.host:
                backend = ctx.backend_factory(view, manifest)
                if manifest["backend"].get("remote_python"):
                    backend.remote_python = manifest["backend"]["remote_python"]
                handle = Handle.from_manifest(manifest["backend"]) or Handle(launcher="nohup")
                if wait:
                    remote = _watch(backend, handle, manifest, manifest_path, ctx)
                    if remote is not None:
                        manifest["exit_reason"] = remote.get("exit_reason")
                        note_state(manifest, str(remote.get("state")), note=f"원격이 끝냈다: {remote.get('state')}({remote.get('exit_reason')})")
                _fetch_until(backend, manifest, manifest_path, Path(dest), ctx, deadline=float(pod.get("fetch_deadline_epoch") or ctx.clock() + 600))
    finally:
        _teardown(manifest, manifest_path, ctx, reason="finish")
    return manifest


def down(manifest: dict[str, Any], *, manifest_path: str | Path, ctx: PodContext) -> bool:
    """지금 지운다 (가져오지 않는다) — 비상용."""
    return _teardown(manifest, manifest_path, ctx, reason="down (사람의 요청)")
