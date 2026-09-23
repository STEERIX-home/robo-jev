"""백엔드와 공급자의 경계 (docs/06 Task 6).

**백엔드**(:class:`Backend`)는 "이미 있는 상자"에 대고 하는 일이다 — 묶음을 올리고, 명령을 돌리고, 분리 실행을
띄우고, 살아 있는지 보고, 멈추고, 파일을 해시해 가져온다. 이번 판의 구현은 둘: :mod:`~robo_jev.launch.providers.ssh`
(rsync + ssh)와 :mod:`~robo_jev.launch.providers.local`(같은 상자). **분리 실행·생사·정지·목록은 여기 한 번만**
쓰여 있고 두 백엔드가 그대로 쓴다 — `local`로 통과한 경로와 `ssh`로 통과한 경로가 같은 코드라는 뜻이다.
다른 점은 :meth:`Backend._exec`(어디서 셸을 여는가)와 :meth:`Backend.push`/:meth:`Backend.fetch`(복사냐 rsync냐)
둘뿐이다.

**공급자**(:class:`Provider`)는 인스턴스를 **만들고 없애는** 일이다 — `create`/`destroy`/`describe`. 이번 과제는
인터페이스만 두고 구현을 이월한다(docs/06 Task 6의 "공급자 instance ID·시간 단가·artifact 위치·종료 결과"를
기록할 자리는 manifest의 `backend.provider`·`backend.instance_id`·`budget.hourly_usd`·`exit_reason`이다).
자격 증명은 이 계층에도 저장하지 않는다 — 키는 **경로**로, 토큰은 환경 변수 **이름**으로만 받는다.
"""

from __future__ import annotations

import copy
import json
import shlex
import subprocess
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

__all__ = ["Backend", "Exec", "Handle", "Instance", "Provider", "RemoteError"]


class RemoteError(RuntimeError):
    """원격 명령이 실패했다 (전송 자체의 실패 포함)."""


@dataclass(frozen=True)
class Exec:
    """명령 하나의 결과. `ok`는 종료 코드 0."""

    argv: list[str]
    returncode: int
    stdout: str
    stderr: str

    @property
    def ok(self) -> bool:
        return self.returncode == 0

    def require(self, what: str) -> Exec:
        if not self.ok:
            raise RemoteError(f"{what}: 종료 코드 {self.returncode}\n  $ {shlex.join(self.argv)}\n  {self.stderr.strip()[:600]}")
        return self


@dataclass
class Handle:
    """띄운 run을 가리키는 것. `unit`이 있으면 systemd, 없으면 `pid`다."""

    launcher: str  # systemd-run | nohup
    unit: str | None = None
    pid: int | None = None

    def as_dict(self) -> dict[str, Any]:
        return {"launcher": self.launcher, "unit": self.unit, "pid": self.pid}

    @classmethod
    def from_manifest(cls, backend: dict[str, Any]) -> Handle | None:
        if not backend.get("launcher"):
            return None
        return cls(launcher=str(backend["launcher"]), unit=backend.get("unit"), pid=backend.get("pid"))


@dataclass
class Instance:
    """공급자가 만든 상자 (이월). 실행기는 이 정보를 manifest의 `backend`에 옮겨 적는다."""

    instance_id: str
    host: str
    user: str
    gpus: int
    hourly_usd: float
    provider: str
    extra: dict[str, Any] = field(default_factory=dict)


class Provider(Protocol):
    """인스턴스 수명 — **인터페이스만**이다 (이번 판은 "이미 있는 상자에 SSH"가 전부).

    구현할 때의 계약: `destroy`는 **확인된 종료**만 참을 돌려준다. 확인하지 못하면 거짓이고, 실행기는 그때
    run 상태를 `unknown`으로 두고 사람에게 알린다 (docs/06 Task 6: 종료 API 실패는 미종료 상태로 명시).
    """

    def create(self, spec: dict[str, Any]) -> Instance: ...

    def destroy(self, instance_id: str) -> bool: ...

    def describe(self, instance_id: str) -> dict[str, Any]: ...


class Backend(ABC):
    """이미 있는 상자에 대고 하는 일. 하위 클래스는 :meth:`_exec`·:meth:`push`·:meth:`fetch`만 구현한다."""

    #: 원격 작업 디렉터리 (run 하나의 뿌리).
    remote_dir: str
    #: 저장소 체크아웃과 파이썬 — 원격에서 코드를 찾는 곳.
    remote_repo: str
    remote_python: str
    gpus: int
    kind: str = "base"

    # -- 하위 클래스가 채우는 것 --

    @abstractmethod
    def _exec(self, argv: list[str], *, timeout: float | None = None) -> Exec:
        """대상 상자에서 명령 하나를 돌린다."""

    @abstractmethod
    def push(self, local_dir: str | Path, remote_relpath: str) -> Exec:
        """디렉터리를 원격 run 디렉터리 아래로 올린다."""

    @abstractmethod
    def push_file(self, local_file: str | Path, remote_relpath: str) -> Exec:
        """파일 하나를 원격 run 디렉터리 아래로 올린다 (resume이 checkpoint를 되돌려 보낼 때)."""

    @abstractmethod
    def fetch(self, relpaths: list[str], dest: str | Path) -> Exec:
        """원격 run 디렉터리의 파일들을 `dest` 아래 같은 상대 경로로 가져온다."""

    # -- 두 백엔드가 그대로 쓰는 것 --

    def shell(self, line: str, *, timeout: float | None = None) -> Exec:
        """셸 한 줄. `sh -c`로 돌린다 (원격에서도 로컬에서도 같은 문장)."""
        return self._exec(["sh", "-c", line], timeout=timeout)

    def with_remote_dir(self, remote_dir: str) -> Backend:
        """작업 디렉터리만 바꾼 사본 — 자식 run(resume)이 부모의 `state.json`을 덮어쓰지 않게."""
        clone = copy.copy(self)
        clone.remote_dir = remote_dir
        return clone

    def path(self, relpath: str = "") -> str:
        base = self.remote_dir.rstrip("/")
        return f"{base}/{relpath.lstrip('/')}" if relpath else base

    def mkdir(self, relpath: str = "") -> Exec:
        return self.shell(f"mkdir -p {shlex.quote(self.path(relpath))}").require("mkdir")

    def read_text(self, relpath: str) -> str | None:
        """원격 파일의 내용. 없으면 None (오류가 아니다 — 아직 안 쓰였을 수 있다)."""
        result = self.shell(f"cat {shlex.quote(self.path(relpath))} 2>/dev/null")
        return result.stdout if result.ok else None

    def exists(self, relpath: str) -> bool:
        return self.shell(f"test -e {shlex.quote(self.path(relpath))}").ok

    # -- 분리 실행 --

    def start(self, argv: list[str], *, unit: str, log_relpath: str, env: dict[str, str] | None = None, workdir: str | None = None) -> Handle:
        """원격에서 **분리 실행**한다 — 실행기가 죽어도, ssh가 끊겨도 run은 살고 로그는 남는다.

        먼저 `systemd-run --user --unit=… --collect`를 시도하고(로그는 파일로 덧붙이게 `StandardOutput=append:`),
        그 상자에 사용자 systemd가 없으면 `nohup … &`로 내려간다. 어느 쪽이 됐는지는 :class:`Handle`에 남고
        manifest의 `backend.launcher`에 적힌다.
        """
        work = workdir or self.remote_repo
        log = self.path(log_relpath)
        exports = " ".join(f"--setenv={shlex.quote(f'{key}={value}')}" for key, value in (env or {}).items())
        systemd = (
            "export XDG_RUNTIME_DIR=${XDG_RUNTIME_DIR:-/run/user/$(id -u)}; "
            f"mkdir -p {shlex.quote(str(Path(log).parent))} && "
            f"systemd-run --user --unit={shlex.quote(unit)} --collect "
            f"--working-directory={shlex.quote(work)} "
            f"--property=StandardOutput=append:{shlex.quote(log)} "
            f"--property=StandardError=append:{shlex.quote(log)} "
            f"{exports} -- {shlex.join(argv)}"
        )
        attempt = self.shell(systemd)
        if attempt.ok:
            return Handle(launcher="systemd-run", unit=unit, pid=None)
        prefix = "".join(f"{key}={shlex.quote(value)} " for key, value in (env or {}).items())
        fallback = (
            f"mkdir -p {shlex.quote(str(Path(log).parent))} && cd {shlex.quote(work)} && "
            f"{prefix}nohup {shlex.join(argv)} >> {shlex.quote(log)} 2>&1 < /dev/null & echo $!"
        )
        result = self.shell(fallback).require(f"분리 실행 (systemd-run도 실패했다: {attempt.stderr.strip()[:200]})")
        pid = int(result.stdout.strip().splitlines()[-1])
        return Handle(launcher="nohup", unit=None, pid=pid)

    def alive(self, handle: Handle) -> bool:
        """원격의 그 run이 아직 돌고 있는가."""
        if handle.unit:
            line = (
                "export XDG_RUNTIME_DIR=${XDG_RUNTIME_DIR:-/run/user/$(id -u)}; "
                f"systemctl --user is-active {shlex.quote(handle.unit)}"
            )
            return self.shell(line).stdout.strip() in ("active", "activating", "deactivating", "reloading")
        if handle.pid:
            return self.shell(f"kill -0 {int(handle.pid)} 2>/dev/null").ok
        return False

    def stop(self, handle: Handle, *, signal: str = "TERM") -> Exec:
        """정지를 **요청**한다. 확인은 호출자(실행기)의 일이다 — 이 함수가 참을 돌려줘도 멈췄다는 뜻이 아니다."""
        if handle.unit:
            # `--no-block`: 정지 **요청**만 보내고 기다리지 않는다 — 공급자의 종료 API와 같은 모양이라
            # "멈췄는지"는 호출자가 따로 확인해야 한다 (docs/06 Task 6).
            verb = f"systemctl --user stop --no-block {shlex.quote(handle.unit)}" if signal == "TERM" else f"systemctl --user kill -s {shlex.quote(signal)} {shlex.quote(handle.unit)}"
            return self.shell("export XDG_RUNTIME_DIR=${XDG_RUNTIME_DIR:-/run/user/$(id -u)}; " + verb)
        if handle.pid:
            return self.shell(f"kill -{shlex.quote(signal)} {int(handle.pid)}")
        return Exec(argv=["stop"], returncode=1, stdout="", stderr="멈출 대상이 없다 (unit도 pid도 없다)")

    def reset_unit(self, unit: str) -> Exec:
        """끝난 유닛의 찌꺼기를 지운다 (같은 이름으로 다시 띄울 수 있게)."""
        line = (
            "export XDG_RUNTIME_DIR=${XDG_RUNTIME_DIR:-/run/user/$(id -u)}; "
            f"systemctl --user reset-failed {shlex.quote(unit)} 2>/dev/null; "
            f"systemctl --user stop {shlex.quote(unit)} 2>/dev/null; true"
        )
        return self.shell(line)

    # -- 목록과 해시 --

    def inventory(self, patterns: list[str]) -> list[dict[str, Any]]:
        """원격에서 파일을 찾아 **원격이 계산한** 바이트·sha256을 돌려준다 (가져온 뒤 여기서 다시 해시해 대조한다)."""
        argv = [*shlex.split(self.remote_python), "-m", "robo_jev.launch.runner", "--inventory", self.path(), "--patterns", *patterns]
        line = f"cd {shlex.quote(self.remote_repo)} && {shlex.join(argv)}"
        result = self.shell(line).require("inventory")
        return list(json.loads(result.stdout or "[]"))

    def describe(self) -> dict[str, Any]:
        """manifest의 `backend` 블록에 적을 것. **자격 증명은 담지 않는다.**"""
        return {"kind": self.kind, "remote_dir": self.remote_dir, "remote_repo": self.remote_repo, "remote_python": self.remote_python, "gpus": int(self.gpus)}


def _run_local(argv: list[str], *, timeout: float | None = None) -> Exec:
    """로컬 프로세스 하나 — ssh 백엔드도 `ssh …`를 이걸로 돌린다."""
    try:
        done = subprocess.run(argv, capture_output=True, text=True, timeout=timeout, check=False)
    except subprocess.TimeoutExpired as exc:
        return Exec(argv=argv, returncode=124, stdout=exc.stdout or "", stderr=f"시간 초과 {timeout}s")
    except OSError as exc:
        return Exec(argv=argv, returncode=127, stdout="", stderr=str(exc))
    return Exec(argv=argv, returncode=done.returncode, stdout=done.stdout, stderr=done.stderr)
