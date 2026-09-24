"""checkpoint — atomic 저장과 재개 상태 (docs/03 §5 "저장 단위", docs/06 Task 5).

한 번의 저장 단위는 model·optimizer·scheduler·RNG(torch CPU + Python + numpy + CUDA)·sampler 위치·config·
manifest(데이터 manifest 참조, serializer·질문 세트 버전, git SHA)와, 진행 중이던 step이 있으면 그
위치(accumulation 단위·구간 index, 누적 gradient, 이어 붙일 스트림 상태)다. 배포용 가중치는 이
파일이 아니라 `model`만 따로 내보내는 것으로 구분한다(docs/03 §5) — 여기서는 재개용만 다룬다.

:func:`save_checkpoint` 는 **atomic**이다: 같은 디렉터리의 임시 파일에 쓰고 fsync한 뒤 rename한다.
임시 파일을 쓰는 도중이나 rename 직전에 프로세스가 죽어도 이전 checkpoint는 그대로 남고, 실패한
임시 파일은 치운다. :func:`load_checkpoint` 는 `weights_only` 로 읽는다 — 저장 단위에 tensor·기본
자료형 이외의 객체를 넣지 않는다(경로는 문자열, numpy 상태는 정수 목록).

스트림 상태(:class:`robo_jev.model.stream.StreamState`)는 :func:`stream_state_to_dict` /
:func:`stream_state_from_dict` 로 tensor dict와 오간다. 저장되는 것은 **detach된** 공통 상태
(분기 이전)다 — truncated BPTT의 구간 경계에서 넘기는 것과 같은 것이다.

이 모듈은 generator·simulator·하네스를 import하지 않는다 (docs/06 §1).
"""

from __future__ import annotations

import os
import random
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch

from robo_jev.model.stream import StreamState, stream_state_class

__all__ = [
    "CHECKPOINT_FORMAT",
    "REQUIRED_KEYS",
    "collect_rng_state",
    "load_checkpoint",
    "restore_rng_state",
    "save_checkpoint",
    "stream_state_from_dict",
    "stream_state_to_dict",
]

#: 저장 형식 표지. 다른 파일(가중치만 있는 것 등)을 재개용으로 잘못 읽지 않게 한다.
CHECKPOINT_FORMAT = "robo-jev-checkpoint-v0"

#: 저장 단위에 반드시 있어야 하는 키 (docs/03 §5). `progress`는 step 사이에서는 `None`이다.
REQUIRED_KEYS = (
    "format", "run_id", "step", "model", "optimizer", "scheduler", "rng", "sampler", "progress",
    "config", "manifest",
)  # fmt: skip


# --------------------------------------------------------------------------
# atomic 저장 / 읽기
# --------------------------------------------------------------------------


def _fsync_directory(directory: Path) -> None:
    """rename이 디렉터리 항목까지 내려앉게 한다 (POSIX; 안 되는 파일 시스템이면 조용히 넘어간다)."""
    try:
        fd = os.open(directory, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


def save_checkpoint(path: str | Path, state: dict) -> None:
    """저장 단위를 `path`에 atomic하게 쓴다 (임시 파일 → fsync → rename).

    필요한 키(:data:`REQUIRED_KEYS`)가 빠지면 아무것도 쓰지 않고 `ValueError`다.
    """
    if not isinstance(state, dict):
        raise ValueError(f"state: dict여야 한다 (받은 값: {type(state).__name__})")
    missing = [key for key in REQUIRED_KEYS if key not in state]
    if missing:
        raise ValueError(f"state: 저장 단위에 필요한 키가 없다: {missing} (필요: {list(REQUIRED_KEYS)})")
    if state["format"] != CHECKPOINT_FORMAT:
        raise ValueError(f"state.format: {CHECKPOINT_FORMAT!r}여야 한다 (받은 값: {state['format']!r})")

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=target.name + ".", suffix=".tmp", dir=target.parent)
    temp = Path(temp_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            torch.save(state, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, target)
    except BaseException:
        try:
            temp.unlink()
        except OSError:
            pass
        raise
    _fsync_directory(target.parent)


def load_checkpoint(path: str | Path) -> dict:
    """저장 단위를 읽는다. 없으면 `FileNotFoundError`, 이 형식이 아니면 `ValueError`."""
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"checkpoint가 없다: {source}")
    state = torch.load(source, map_location="cpu", weights_only=True)
    if not isinstance(state, dict) or state.get("format") != CHECKPOINT_FORMAT:
        raise ValueError(
            f"{source}: 재개용 checkpoint(format={CHECKPOINT_FORMAT!r})가 아니다 "
            f"(받은 format: {state.get('format') if isinstance(state, dict) else type(state).__name__!r})"
        )
    missing = [key for key in REQUIRED_KEYS if key not in state]
    if missing:
        raise ValueError(f"{source}: 저장 단위에 필요한 키가 없다: {missing}")
    return state


# --------------------------------------------------------------------------
# RNG
# --------------------------------------------------------------------------


def collect_rng_state() -> dict[str, Any]:
    """torch CPU RNG + Python `random` + numpy `np.random` + **CUDA RNG**의 현재 상태 (tensor·기본 자료형만).

    `cuda`는 **이 프로세스가 CUDA를 켠 적이 있을 때만** 담는다(:func:`torch.cuda.is_initialized`).
    `get_rng_state_all()` 자체가 CUDA를 초기화하므로, 켜지 않은 프로세스(CPU 학습)에서 부르면 쓰지도 않을
    context를 만든다 — 그리고 켠 적이 없으면 뽑은 것도 없으므로 저장할 상태가 없다. 장치마다 한 tensor다.

    **이 설정에서는 CUDA generator에서 뽑는 것이 없다**(dropout 0, Qwen3.5 backbone에 확률 연산 없음, 모델
    초기화는 CPU generator를 명시적으로 받는다) — 그래서 이 키가 R2까지의 결과를 바꾸지 않는다. 담는 이유는
    앞으로다: CUDA generator를 한 번이라도 쓰는 run(dropout을 켜거나 표집을 넣는 순간)은 이 키가 없으면
    재개가 재현되지 않고, 8 GPU 클라우드 재시작이 그 위에 선다.
    """
    kind, keys, position, has_gauss, cached = np.random.get_state()
    state: dict[str, Any] = {
        "torch": torch.get_rng_state(),
        "python": random.getstate(),
        "numpy": {
            "kind": str(kind),
            "keys": [int(key) for key in keys],
            "position": int(position),
            "has_gauss": int(has_gauss),
            "cached_gaussian": float(cached),
        },
    }
    if torch.cuda.is_available() and torch.cuda.is_initialized():
        # 복사본을 담는다 — 저장 뒤의 뽑기가 이 tensor를 바꾸지 않게.
        state["cuda"] = [device.clone() for device in torch.cuda.get_rng_state_all()]
    return state


def restore_rng_state(state: dict[str, Any]) -> None:
    """:func:`collect_rng_state`가 돌려준 상태를 네 RNG에 되돌린다.

    `cuda`는 **있으면** 되돌린다 — R2까지의 checkpoint에는 그 키가 없고(그 run들은 CUDA generator에서 뽑지
    않았다) 그것들은 계속 이어갈 수 있어야 한다. 반대로 키가 **있는데** 이 상자가 받을 수 없으면(CUDA가 없거나
    장치 수가 다르면) 조용히 넘기지 않고 이름으로 거절한다 — 조용히 넘기면 "재개했는데 같은 뽑기가 아니다"가
    아무 데도 적히지 않은 채 지나간다.
    """
    for key in ("torch", "python", "numpy"):
        if key not in state:
            raise ValueError(f"rng.{key}: 없다 (필요: torch, python, numpy)")
    torch.set_rng_state(torch.as_tensor(state["torch"], dtype=torch.uint8))
    version, internal, gauss_next = state["python"]
    random.setstate((int(version), tuple(int(v) for v in internal), gauss_next))
    numpy_state = state["numpy"]
    np.random.set_state(
        (
            numpy_state["kind"],
            np.asarray(numpy_state["keys"], dtype=np.uint32),
            int(numpy_state["position"]),
            int(numpy_state["has_gauss"]),
            float(numpy_state["cached_gaussian"]),
        )
    )
    cuda = state.get("cuda")
    if cuda is None:
        return
    if not torch.cuda.is_available():
        raise ValueError(
            "rng.cuda: CUDA generator 상태가 저장돼 있는데 이 상자에는 CUDA가 없다 — 같은 뽑기로 이어갈 수 없다"
        )
    if len(cuda) != torch.cuda.device_count():
        raise ValueError(
            f"rng.cuda: 저장된 장치 수({len(cuda)})와 이 상자의 장치 수({torch.cuda.device_count()})가 다르다 — "
            "장치마다 generator가 하나이므로 같은 뽑기로 이어갈 수 없다"
        )
    torch.cuda.set_rng_state_all([torch.as_tensor(device, dtype=torch.uint8) for device in cuda])


# --------------------------------------------------------------------------
# 스트림 상태 ↔ dict
# --------------------------------------------------------------------------


def stream_state_to_dict(state: Any) -> dict[str, Any]:
    """분기 이전 공통 상태를 detach된 tensor dict로 (구간 경계에서 넘기는 것과 같은 것). 상태 클래스의 ``to_dict``."""
    if getattr(state, "is_branch", False):
        raise ValueError("branch 상태는 저장하지 않는다 — 구간 경계의 상태는 분기 이전 공통 상태다")
    return state.to_dict()


def stream_state_from_dict(packed: dict[str, Any], backbone: Any) -> Any:
    """:func:`stream_state_to_dict`의 역 — 주어진 backbone의 상태 클래스(fixture는 :class:`StreamState`)에 붙인 공통 상태."""
    cls = stream_state_class(backbone)
    kind = packed.get("kind", "tiny")
    expected = "tiny" if cls is StreamState else "qwen"
    if kind != expected:
        raise ValueError(f"carried_state.kind: {expected!r} 상태여야 한다 (저장된 것: {kind!r}) — 다른 종류의 backbone에서 저장한 상태다")
    return cls.from_dict(packed, backbone)
