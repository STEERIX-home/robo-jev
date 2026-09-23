"""`local` 백엔드 — 같은 상자. SSH 경로와 **같은 코드**를 돌리되 전송만 없앤다.

왜 있는가: 인수(Stage B)는 sshd·키가 있어야 도는 것이 아니라 **실행기의 논리**(묶음 → 분리 실행 → 상태 →
회수 → 상한 → 취소)가 도는 것이 목표다. 그래서 분리 실행·생사·정지·목록은 :class:`~robo_jev.launch.providers.base.Backend`
에 한 번만 쓰여 있고, 이 클래스는 셸을 **로컬에서** 열고 복사로 올리고 내린다. sshd가 없는 상자에서도 같은
검사가 돌고, 있는 상자에서는 두 백엔드가 같은 검사를 통과한다.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from robo_jev.launch.providers.base import Backend, Exec, _run_local

__all__ = ["LocalBackend"]


class LocalBackend(Backend):
    kind = "local"

    def __init__(self, *, remote_dir: str | Path, remote_repo: str | Path, remote_python: str, gpus: int = 0) -> None:
        self.remote_dir = str(Path(remote_dir))
        self.remote_repo = str(Path(remote_repo))
        self.remote_python = str(remote_python)
        self.gpus = int(gpus)

    def _exec(self, argv: list[str], *, timeout: float | None = None) -> Exec:
        return _run_local(argv, timeout=timeout)

    def push(self, local_dir: str | Path, remote_relpath: str) -> Exec:
        source = Path(local_dir)
        target = Path(self.path(remote_relpath))
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            shutil.rmtree(target)  # ssh 백엔드의 `rsync --delete`와 같은 뜻 — 지난 묶음의 찌꺼기를 남기지 않는다
        shutil.copytree(source, target)
        return Exec(argv=["copytree", str(source), str(target)], returncode=0, stdout="", stderr="")

    def push_file(self, local_file: str | Path, remote_relpath: str) -> Exec:
        source = Path(local_file)
        target = Path(self.path(remote_relpath))
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        return Exec(argv=["copy", str(source), str(target)], returncode=0, stdout="", stderr="")

    def fetch(self, relpaths: list[str], dest: str | Path) -> Exec:
        root = Path(dest)
        copied = []
        for relpath in relpaths:
            source = Path(self.path(relpath))
            target = root / relpath
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            copied.append(relpath)
        return Exec(argv=["copy", *copied], returncode=0, stdout="\n".join(copied), stderr="")

    def describe(self) -> dict:
        return {**super().describe(), "host": None, "user": None, "port": None}
