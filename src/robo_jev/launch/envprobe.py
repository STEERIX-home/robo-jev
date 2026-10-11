"""원격 환경 기록 — GPU·드라이버·CUDA·torch·transformers·커널(fla·causal_conv1d)이 무엇이었나 (Task cloud1, docs/05 §8 1번).

``python -m robo_jev.launch.envprobe --out <reports>/env.json``. job의 첫 단계로 venv 파이썬에서 돈다. 속도 배수(H100 ÷ GB10)는
**같은 커널 묶음**일 때만 하드웨어의 배수다 — GB10의 기준(Task v0d §4)은 fla Triton + causal_conv1d **없음**(torch 대체)이었으므로
여기서 어느 구현이 골라졌는지 적는다.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

__all__ = ["collect", "main"]


def _version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _run(argv: list[str]) -> str | None:
    try:
        return subprocess.run(argv, capture_output=True, text=True, timeout=30, check=False).stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def _kernels() -> dict[str, Any]:
    """transformers의 qwen3_5가 고른 구현 — 함수의 closure(`implementation`)가 어느 모듈의 것인가."""
    out: dict[str, Any] = {}
    for package in ("fla", "causal_conv1d"):
        try:
            importlib.import_module(package)
            out[f"{package}_importable"] = True
        except Exception as error:  # noqa: BLE001
            out[f"{package}_importable"] = False
            out[f"{package}_error"] = f"{type(error).__name__}: {str(error)[:200]}"
    try:
        modeling = importlib.import_module("transformers.models.qwen3_5.modeling_qwen3_5")
        chosen = {}
        for name in ("torch_chunk_gated_delta_rule", "torch_recurrent_gated_delta_rule", "causal_conv1d_fn", "causal_conv1d_update"):
            fn = getattr(modeling, name, None)
            cells = dict(zip(getattr(fn, "__code__", None).co_freevars, fn.__closure__ or ())) if fn is not None and getattr(fn, "__code__", None) else {}
            impl = cells.get("implementation")
            value = getattr(impl, "cell_contents", None)
            chosen[name] = None if value is None else f"{getattr(value, '__module__', '?')}.{getattr(value, '__name__', '?')}"
        out["qwen3_5_implementations"] = chosen
    except Exception as error:  # noqa: BLE001
        out["qwen3_5_error"] = f"{type(error).__name__}: {str(error)[:200]}"
    return out


def collect() -> dict[str, Any]:
    info: dict[str, Any] = {
        "python": sys.version.split()[0], "platform": platform.platform(), "machine": platform.machine(),
        "cpu_count": os.cpu_count(), "packages": {name: _version(name) for name in ("torch", "triton", "transformers", "peft", "flash-linear-attention",
                                                                                  "fla-core", "causal-conv1d", "tokenizers", "huggingface-hub", "numpy")},
        "nvidia_smi": _run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total,pci.bus_id", "--format=csv,noheader"]),
        "disk_free_bytes": shutil.disk_usage("/").free,
        "env": {key: os.environ.get(key) for key in ("RUNPOD_POD_ID", "RUNPOD_DC_ID", "RUNPOD_GPU_COUNT", "CUDA_VERSION", "PYTORCH_CUDA_ALLOC_CONF")},
    }  # fmt: skip
    try:
        import torch

        info["torch"] = {"version": torch.__version__, "cuda": torch.version.cuda, "cuda_available": torch.cuda.is_available(),
                         "cudnn": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None}  # fmt: skip
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            info["torch"]["device"] = {"name": props.name, "capability": f"{props.major}.{props.minor}", "total_memory": int(props.total_memory),
                                       "count": torch.cuda.device_count(), "bf16": torch.cuda.is_bf16_supported()}  # fmt: skip
    except Exception as error:  # noqa: BLE001
        info["torch_error"] = f"{type(error).__name__}: {str(error)[:300]}"
    info["kernels"] = _kernels()
    return info


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m robo_jev.launch.envprobe", description=__doc__.splitlines()[0])
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    info = collect()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(info, ensure_ascii=False, indent=1, default=str) + "\n", encoding="utf-8")
    print(json.dumps({k: info.get(k) for k in ("python", "machine", "nvidia_smi")}, ensure_ascii=False))
    print(json.dumps(info.get("torch"), ensure_ascii=False, default=str))
    print(json.dumps(info.get("kernels"), ensure_ascii=False))
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
