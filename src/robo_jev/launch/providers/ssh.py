"""`ssh` 백엔드 — 호스트·사용자·작업 디렉터리·GPU 수를 받는 **일반** 백엔드 (docs/06 Task 6).

공급자 SDK를 의존성에 넣지 않는다: 여기 있는 것은 `ssh`와 `rsync` 두 명령뿐이고, 어떤 상자든 그 둘이 되면
쓸 수 있다. 자격 증명은 저장소에 들어가지 않는다 — 개인키는 **경로**(`identity_path`, 기본은 ssh의 기본 키)로
받고 manifest에도 경로만 적힌다.

원격의 파이썬은 미리 준비된 venv(`--remote-python`)를 쓰거나, `--remote-python 'uv run python'`처럼 uv에 맡긴다
(이 클래스는 그 문자열을 셸에 그대로 넘긴다 — 저장소 디렉터리에서 돌기 때문에 `uv run`이 `uv sync`를 겸한다).

전송(`transport`)은 주입할 수 있다 — 단위 시험은 sshd 없이 가짜 전송으로 명령 문장 자체를 검사하고,
인수(Stage B)는 진짜 `ssh`로 돈다.
"""

from __future__ import annotations

import shlex
from collections.abc import Callable
from pathlib import Path

from robo_jev.launch.providers.base import Backend, Exec, _run_local

__all__ = ["DEFAULT_SSH_OPTIONS", "RSYNC_RESUME_OPTIONS", "SshBackend"]

Transport = Callable[..., Exec]

#: 비대화식 기본값. 암호를 묻지 않고(BatchMode), 끊긴 연결을 오래 붙들지 않는다.
DEFAULT_SSH_OPTIONS = ("-o", "BatchMode=yes", "-o", "ConnectTimeout=10", "-o", "ServerAliveInterval=15")

#: 끊긴 전송을 **이어서** 받기 위한 rsync 인자. checkpoint 하나가 26 GB이므로(docs/05 §6) 한 번 끊겼다고
#: 처음부터 다시 받으면 그 시간이 그대로 돈이다. `--partial-dir`은 반쯤 받은 조각을 옆 디렉터리에 남겨
#: 다음 호출이 그것을 바탕으로 이어 받게 하고(목적지에는 **완성된 파일만** 나타난다), `--timeout`은 죽은
#: 연결을 영원히 붙들고 있지 않게 한다. rsync는 상대 경로 `--partial-dir`을 스스로 전송 목록에서 뺀다.
RSYNC_RESUME_OPTIONS = ("--partial", "--partial-dir=.rsync-partial", "--timeout=600")


def _reject_option_shaped(field: str, value: str | None) -> None:
    """`-`로 시작하는 호스트·사용자를 거절한다 — argv의 맨 앞 글자가 `ssh`에게는 **옵션**이다.

    대상(`user@host`)은 `ssh`에 맨 인자로 붙으므로 `-oProxyCommand=…` 같은 값은 셸을 거치지 않고도
    ssh 자신의 옵션으로 해석된다(경로는 전부 `shlex.quote` 되지만 이것은 따옴표가 막아 주지 못한다).
    값은 사람의 플래그나 명세에서 오므로 구멍이라기보다 울타리다 — 그래도 울타리는 여기 있어야 한다.
    """
    if value is None:
        return
    if not str(value).strip():
        raise ValueError(f"{field}: 비어 있다 — ssh 대상이 되지 못한다")
    if str(value).startswith("-"):
        raise ValueError(
            f"{field}: `-`로 시작하는 값({value!r})은 ssh가 **옵션**으로 읽는다 (예: -oProxyCommand=…) — "
            "호스트·사용자 이름으로 받지 않는다"
        )


class SshBackend(Backend):
    kind = "ssh"

    def __init__(
        self,
        *,
        host: str,
        user: str | None = None,
        port: int | None = None,
        remote_dir: str,
        remote_repo: str,
        remote_python: str,
        gpus: int = 0,
        identity_path: str | None = None,
        ssh_options: tuple[str, ...] | list[str] = DEFAULT_SSH_OPTIONS,
        known_hosts: str | None = None,
        transport: Transport | None = None,
    ) -> None:
        _reject_option_shaped("host", host)
        _reject_option_shaped("user", user)
        self.host = host
        self.user = user
        self.port = port
        self.remote_dir = remote_dir
        self.remote_repo = remote_repo
        self.remote_python = remote_python
        self.gpus = int(gpus)
        self.identity_path = identity_path
        self.known_hosts = known_hosts
        self.ssh_options = list(ssh_options)
        self._transport: Transport = transport or _run_local

    # -- 명령 문장 --

    @property
    def target(self) -> str:
        return f"{self.user}@{self.host}" if self.user else self.host

    def ssh_argv(self) -> list[str]:
        argv = ["ssh", *self.ssh_options]
        if self.port:
            argv += ["-p", str(int(self.port))]
        if self.identity_path:
            argv += ["-i", str(self.identity_path), "-o", "IdentitiesOnly=yes"]
        if self.known_hosts:
            argv += ["-o", f"UserKnownHostsFile={self.known_hosts}"]
        return argv

    def _rsh(self) -> str:
        """rsync의 `-e` 인자 — ssh와 **같은** 옵션으로 붙는다 (키가 다른 두 경로가 생기지 않게)."""
        return shlex.join(self.ssh_argv())

    def _exec(self, argv: list[str], *, timeout: float | None = None) -> Exec:
        return self._transport([*self.ssh_argv(), self.target, shlex.join(argv)], timeout=timeout)

    # -- 전송 --

    def push(self, local_dir: str | Path, remote_relpath: str) -> Exec:
        source = str(Path(local_dir)).rstrip("/") + "/"
        target = self.path(remote_relpath).rstrip("/") + "/"
        self.shell(f"mkdir -p {shlex.quote(target)}").require("원격 디렉터리 준비")
        argv = ["rsync", "-a", "--delete", *RSYNC_RESUME_OPTIONS, "-e", self._rsh(), source, f"{self.target}:{target}"]
        return self._transport(argv, timeout=None).require("rsync push")

    def push_file(self, local_file: str | Path, remote_relpath: str) -> Exec:
        target = self.path(remote_relpath)
        self.shell(f"mkdir -p {shlex.quote(str(Path(target).parent))}").require("원격 디렉터리 준비")
        argv = ["rsync", "-a", *RSYNC_RESUME_OPTIONS, "-e", self._rsh(), str(Path(local_file)), f"{self.target}:{target}"]
        return self._transport(argv, timeout=None).require(f"rsync push {remote_relpath}")

    def fetch(self, relpaths: list[str], dest: str | Path) -> Exec:
        root = Path(dest)
        root.mkdir(parents=True, exist_ok=True)
        if not relpaths:
            return Exec(argv=["rsync"], returncode=0, stdout="", stderr="")
        sources = [f"{self.target}:{self.path(relpath)}" for relpath in relpaths]
        # `--relative`가 아니라 상대 경로를 우리가 만든다 — 원격 뿌리의 절대 경로를 로컬에 재현하지 않기 위해서다.
        last: Exec | None = None
        for relpath, source in zip(relpaths, sources, strict=True):
            target = root / relpath
            target.parent.mkdir(parents=True, exist_ok=True)
            argv = ["rsync", "-a", *RSYNC_RESUME_OPTIONS, "-e", self._rsh(), source, str(target)]
            last = self._transport(argv, timeout=None).require(f"rsync fetch {relpath}")
        assert last is not None
        return last

    def describe(self) -> dict:
        return {
            **super().describe(),
            "host": self.host,
            "user": self.user,
            "port": self.port,
            # 키의 **경로**만. 내용도, 암호도, 토큰도 적지 않는다.
            "identity_path": self.identity_path,
            "known_hosts": self.known_hosts,
            "ssh_options": list(self.ssh_options),
        }
