"""GPU 메모리 울타리 — DGX Spark(GB10)의 통합 메모리에서 프로세스가 장치 메모리를 다 먹어 커널이 세션을 죽이는 일을 막는다.

GB10은 CPU와 GPU가 121 GB를 **같이** 쓴다. PyTorch의 caching allocator는 그 전부를 장치 메모리로 보고 한도 없이
자라므로(nvidia-smi는 N/A), 큰 forward+backward가 통합 메모리를 다 채우면 커널이 데스크톱 세션(과 그 안의 모든 것)을
죽인다 — 2026-09-20 01:00·01:43·02:08 KST의 `NVRM NV_ERR_NO_MEMORY`가 그것이다. 이 모듈은 세 가지를 준다.

* :func:`limit_gpu_memory` — `torch.cuda.set_per_process_memory_fraction(fraction)`(기본 0.6 = ≈73 GB)으로 allocator의 상한을
  두어 넘치는 할당이 커널 OOM 대신 **프로세스 안의** `torch.OutOfMemoryError`가 되게 하고, `PYTORCH_CUDA_ALLOC_CONF`가 비어
  있으면 `expandable_segments:True`를 넣는다(조각화로 reserved가 allocated의 2~4×가 되던 G0a 현상의 완화). **첫 CUDA 할당
  전에** 불러야 한다 — 모든 GPU 진입점(`scripts/measure_candidates.py`·`adapt_readout.py`·`attribution.py`, GPU 테스트의
  fixture)이 맨 처음에 부른다(`--gpu-memory-fraction`으로 바꾼다).
* :func:`require_free` — `torch.cuda.mem_get_info()`의 여유가 모자라면 할당해 보는 대신 분명한 오류를 낸다.
* :func:`memory_report` — free/total/allocated/reserved(바이트)를 돌려준다(작업의 처음과 끝에 로그로 남긴다).
"""

from __future__ import annotations

import os
from typing import Any

__all__ = ["DEFAULT_FRACTION", "EXPANDABLE_SEGMENTS", "limit_gpu_memory", "memory_report", "require_free"]

#: 프로세스가 쓸 수 있는 장치(통합) 메모리의 몫 — 121 GB의 0.6 ≈ 73 GB; 나머지는 CPU·데스크톱·다른 프로세스의 것.
DEFAULT_FRACTION = 0.6
EXPANDABLE_SEGMENTS = "expandable_segments:True"


def limit_gpu_memory(fraction: float = DEFAULT_FRACTION, *, device: int = 0) -> dict[str, Any]:
    """allocator 상한을 `fraction`으로 두고(첫 CUDA 할당 전에), `PYTORCH_CUDA_ALLOC_CONF`가 비어 있으면 expandable segments를 켠다.

    CUDA가 없으면 env만 만지고 `{"cuda": False}`를 돌려준다. `fraction`은 (0, 1]이어야 한다.
    """
    if not (isinstance(fraction, (int, float)) and 0.0 < float(fraction) <= 1.0):
        raise ValueError(f"gpu memory fraction: (0, 1] 안의 수여야 한다 (받은 값: {fraction!r})")
    if not os.environ.get("PYTORCH_CUDA_ALLOC_CONF"):
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = EXPANDABLE_SEGMENTS
    import torch

    if not torch.cuda.is_available():
        return {"cuda": False, "fraction": float(fraction), "alloc_conf": os.environ["PYTORCH_CUDA_ALLOC_CONF"]}
    torch.cuda.set_per_process_memory_fraction(float(fraction), device)
    total = int(torch.cuda.get_device_properties(device).total_memory)
    return {"cuda": True, "fraction": float(fraction), "limit_bytes": int(total * float(fraction)), "total_bytes": total, "alloc_conf": os.environ["PYTORCH_CUDA_ALLOC_CONF"]}


def memory_report(device: int = 0) -> dict[str, int | None]:
    """`{"free_bytes", "total_bytes", "allocated_bytes", "reserved_bytes", "peak_allocated_bytes"}` — CUDA가 없으면 값이 None."""
    import torch

    if not torch.cuda.is_available():
        return {"free_bytes": None, "total_bytes": None, "allocated_bytes": None, "reserved_bytes": None, "peak_allocated_bytes": None}
    free, total = torch.cuda.mem_get_info(device)
    return {
        "free_bytes": int(free), "total_bytes": int(total), "allocated_bytes": int(torch.cuda.memory_allocated(device)),
        "reserved_bytes": int(torch.cuda.memory_reserved(device)), "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
    }


def require_free(required_bytes: int, *, what: str = "this step", device: int = 0) -> dict[str, int | None]:
    """장치(통합) 메모리의 여유가 `required_bytes`보다 작으면 할당해 보지 않고 `MemoryError`를 낸다. 돌려주는 것은 :func:`memory_report`."""
    report = memory_report(device)
    free = report["free_bytes"]
    if free is not None and free < int(required_bytes):
        raise MemoryError(
            f"{what}: 장치 메모리 여유 {free / 2**30:.1f} GiB < 필요 {int(required_bytes) / 2**30:.1f} GiB "
            f"(total {report['total_bytes'] / 2**30:.1f} GiB, reserved {report['reserved_bytes'] / 2**30:.1f} GiB) — 할당하지 않는다"
        )
    return report
