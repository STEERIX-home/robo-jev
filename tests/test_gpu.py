"""`robo_jev.gpu` — 메모리 울타리 (CPU: 검증·env; GPU: 넘치는 할당이 프로세스 안의 OutOfMemoryError가 된다)."""

from __future__ import annotations

import pytest
import torch

from robo_jev.gpu import DEFAULT_FRACTION, EXPANDABLE_SEGMENTS, limit_gpu_memory, memory_report, require_free


def test_limit_gpu_memory_validates_the_fraction_and_sets_the_alloc_conf_when_unset(monkeypatch):
    for bad in (0.0, -0.1, 1.5, "0.5", None):
        with pytest.raises(ValueError):
            limit_gpu_memory(bad)  # type: ignore[arg-type]
    monkeypatch.delenv("PYTORCH_CUDA_ALLOC_CONF", raising=False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    out = limit_gpu_memory()
    assert out == {"cuda": False, "fraction": DEFAULT_FRACTION, "alloc_conf": EXPANDABLE_SEGMENTS}
    assert 0.0 < DEFAULT_FRACTION < 1.0
    # 사용자가 이미 정한 alloc conf는 덮어쓰지 않는다
    monkeypatch.setenv("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128")
    assert limit_gpu_memory(0.5)["alloc_conf"] == "max_split_size_mb:128"
    report = memory_report()
    assert set(report) == {"free_bytes", "total_bytes", "allocated_bytes", "reserved_bytes", "peak_allocated_bytes"}
    assert all(value is None for value in report.values())
    assert require_free(10**12) == report  # CUDA가 없으면 여유를 알 수 없어 통과시킨다


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA가 있어야 한다")
def test_oversized_allocation_raises_inside_the_process_under_the_fraction_limit():
    out = limit_gpu_memory(0.6)
    assert out["cuda"] and out["limit_bytes"] < out["total_bytes"]
    report = memory_report()
    assert report["total_bytes"] == out["total_bytes"] and report["free_bytes"] > 0
    with pytest.raises(torch.OutOfMemoryError):
        torch.empty(int(out["total_bytes"] * 0.8), dtype=torch.uint8, device="cuda")
    with pytest.raises(MemoryError):
        require_free(out["total_bytes"] * 2, what="검사")
    assert memory_report()["allocated_bytes"] == report["allocated_bytes"]
