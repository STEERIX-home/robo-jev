"""Task R8 A2 — 끝난 run의 체크포인트 슬림화 (사용자 승인 2026-09-29: **정확히 아래 아홉 파일**의 optimizer 상태만 지운다).

    CUDA_VISIBLE_DEVICES= uv run --no-sync python scripts/slim_checkpoint.py run --out artifacts/reports/r8-a2-slim.json
    CUDA_VISIBLE_DEVICES= uv run --no-sync python scripts/slim_checkpoint.py run --files artifacts/runs/r7-t1-fp32-2b-s18/checkpoint.pt --out …
    CUDA_VISIBLE_DEVICES= uv run --no-sync python scripts/slim_checkpoint.py run --dry-run --out …   # 검증만, 원본은 그대로

파일 하나는 26.35 GB = `model` 3.76 GB + `optimizer` 22.58 GB(AdamW 모멘트와 fp32 master 사본) + 작은 키들이다. 파일마다 차례로:

1. 원본의 크기와 sha256을 잰다(끝까지 읽는다 — 보고서의 "전"). 원본을 **mmap으로** 싣는다(`model`만 페이지가 올라온다).
2. `optimizer`만 뺀 model-only 상태(:func:`robo_jev.checkpoint.model_only_state` — 형식 표지 `robo-jev-checkpoint-v0-model-only`와 `slimmed`
   기록: 원본 sha256·크기·시각·도구)를 **같은 디렉터리의 임시 파일**에 쓰고 fsync한다(아직 아무것도 대체하지 않는다).
3. 검사 셋. (a) 임시 파일의 `model` tensor가 원본과 **비트 단위로** 같다 — 키·모양·dtype·바이트(:func:`robo_jev.checkpoint.compare_model_tensors`);
   (b) optimizer 밖의 키(run_id·step·status·scheduler·rng·sampler·progress·config·manifest·history)가 원본과 같다; (c) **평가 경로가 그 파일을
   받는다** — C2·D가 쓰는 :func:`robo_jev.harness.model_policy.load_serving_judge` 를 CPU에서 그 임시 파일로 부른다(계약 digest 대조 →
   rank → `load_readout_checkpoint`의 `state["model"]` 적재 — backbone은 실제 2B bf16을 CPU에 싣는다) 그리고 실린 파라미터가 파일의 tensor와
   비트 단위로 같은지 본다. GPU forward는 A3의 GPU 적재가 겸한다.
4. 셋 다 통과할 때만 ``os.replace``로 원본을 대체하고 디렉터리를 fsync한다 → 새 파일의 크기·sha256("후"). 하나라도 떨어지면 임시 파일을 지우고
   **원본은 건드리지 않은 채** 이유를 적는다.

대체할 수 있는 경로는 :data:`APPROVED` 뿐이다 — 목록 밖의 경로는 아무것도 읽기 전에 거절한다. 이미 model-only인 파일은 건너뛴다(다시 돌려도 된다).
보고서(JSON)에는 파일별 표와 `/`의 여유 전후가 든다. GPU를 쓰지 않는다 — `CUDA_VISIBLE_DEVICES`를 비우고 돌린다.
"""

from __future__ import annotations

import argparse
import datetime as dt
import gc
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import torch  # noqa: E402

from robo_jev.checkpoint import (  # noqa: E402
    CHECKPOINT_FORMAT,
    MODEL_ONLY_FORMAT,
    compare_model_tensors,
    load_model_checkpoint,
    model_only_state,
    write_temporary,
)

#: 사용자가 승인한 아홉 파일 (브리프 A2의 목록 그대로, 저장소 기준 경로). 이 밖의 파일은 이 도구가 대체하지 않는다.
APPROVED = (
    "artifacts/runs/r2-t1-fp32-2b/checkpoint.pt",
    "artifacts/runs/r2-t1-fp32-2b/checkpoint-step40.pt",
    "artifacts/runs/r3a-t1-fp32-2b-s18/checkpoint.pt",
    "artifacts/runs/r3a-t1-fp32-2b-s19/checkpoint.pt",
    "artifacts/runs/r3a-t1-fp32-2b-s19/checkpoint-step40.pt",
    "artifacts/runs/r3a-t1-466/r3a-t1-fp32-2b-466/checkpoint.pt",
    "artifacts/runs/r5-t1-fp32-2b-s18/checkpoint.pt",
    "artifacts/runs/r6-t1-fp32-2b-s18/checkpoint.pt",
    "artifacts/runs/r7-t1-fp32-2b-s18/checkpoint.pt",
)
SCRIPT_VERSION = "r8-slim-checkpoint-1.0"
#: sha256을 읽는 묶음 크기.
_CHUNK = 64 * 2**20


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def _log(message: str) -> None:
    print(f"[slim {time.strftime('%H:%M:%S')}] {message}", file=sys.stderr, flush=True)


def sha256_file(path: Path) -> str:
    """파일 전체의 sha256 — 다 읽은 뒤 그 파일의 페이지 캐시를 내보낸다(26 GB 원본이 통합 메모리의 캐시를 채우지 않게)."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            block = handle.read(_CHUNK)
            if not block:
                break
            digest.update(block)
        try:
            os.posix_fadvise(handle.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
        except (AttributeError, OSError):
            pass
    return digest.hexdigest()


def disk_free(path: Path) -> int:
    return int(shutil.disk_usage(path).free)


def _bits_equal(a: torch.Tensor, b: torch.Tensor) -> bool:
    if a.dtype != b.dtype or tuple(a.shape) != tuple(b.shape):
        return False
    return torch.equal(a.detach().contiguous().reshape(-1).view(torch.uint8), b.detach().contiguous().reshape(-1).view(torch.uint8))


def same_tree(a: Any, b: Any) -> bool:
    """두 값이 같은가 — tensor는 비트 단위, dict는 키 순서까지, 목록·tuple은 타입까지, float NaN은 NaN끼리 같다."""
    if isinstance(a, torch.Tensor) or isinstance(b, torch.Tensor):
        return isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor) and _bits_equal(a, b)
    if isinstance(a, dict):
        return isinstance(b, dict) and list(a) == list(b) and all(same_tree(a[key], b[key]) for key in a)
    if isinstance(a, (list, tuple)):
        return type(a) is type(b) and len(a) == len(b) and all(same_tree(x, y) for x, y in zip(a, b))
    if isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b):
        return True
    return type(a) is type(b) and a == b


def other_keys_check(original: dict[str, Any], written: dict[str, Any]) -> dict[str, Any]:
    """optimizer·형식 표지·`model`·`slimmed` 밖의 키가 원본과 같은가 — 슬림 파일이 **optimizer만** 뺐다는 확인."""
    skip = {"optimizer", "format", "model", "slimmed"}
    keys = [key for key in original if key not in skip]
    missing = [key for key in keys if key not in written]
    extra = [key for key in written if key not in skip and key not in original]
    different = [key for key in keys if key in written and not same_tree(original[key], written[key])]
    return {"equal": not (missing or extra or different), "keys": keys, "missing": missing, "unexpected": extra, "different": different}


def loaded_state_check(loaded: dict[str, torch.Tensor], model: dict[str, torch.Tensor], *, contract_sha256: Any) -> dict[str, Any]:
    """평가 경로가 모델에 **실제로 실은 값**(`judge.state_dict()`)이 파일의 `model` tensor와 비트 단위로 같은가."""
    mismatches = [key for key, tensor in model.items() if key not in loaded or not _bits_equal(loaded[key], tensor)]
    return {"loaded": True, "contract_sha256": contract_sha256, "tensors": len(model), "params_equal": not mismatches, "mismatches": mismatches[:10]}


def serving_load_check(path: Path, model: dict[str, torch.Tensor]) -> dict[str, Any]:
    """C2·D의 적재 함수(:func:`robo_jev.harness.model_policy.load_serving_judge`)를 CPU에서 이 파일로 부른다 — 계약 digest가 지금 체크아웃과
    다르거나 rank·키가 맞지 않으면 거기서 거절된다. 실린 파라미터를 파일과 비트 단위로 견준 뒤 모델을 놓는다."""
    from robo_jev.harness.model_policy import load_serving_judge

    manifest = load_model_checkpoint(path, mmap=True)["manifest"]
    model_id = str(((manifest.get("model") or {}).get("id")) or "Qwen/Qwen3.5-2B")
    started = time.perf_counter()
    bundle = load_serving_judge(path, model_id=model_id, device="cpu", compile_dense=False)
    try:
        out = loaded_state_check(bundle["judge"].state_dict(), model, contract_sha256=bundle["manifest"].get("contract_sha256"))
        out.update({"function": "robo_jev.harness.model_policy.load_serving_judge (device=cpu)", "model_id": model_id,
                    "tokenizer": bundle["tokenizer_name"], "seconds": round(time.perf_counter() - started, 1)})  # fmt: skip
        return out
    finally:
        del bundle
        gc.collect()


def _fsync_directory(directory: Path) -> None:
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


def slim_one(path: Path, *, evaluate: Callable[[Path, dict[str, torch.Tensor]], dict[str, Any]] | None, dry_run: bool = False) -> dict[str, Any]:
    """파일 하나를 슬림한다 (모듈 설명의 1~4). 돌려주는 행: 크기·sha256 전후, 검사 셋의 결과, 대체 여부, 확보한 바이트, 실패 이유."""
    path = Path(path)
    started = time.perf_counter()
    row: dict[str, Any] = {
        "path": str(path), "dry_run": bool(dry_run), "replaced": False, "skipped": None, "error": None,
        "bytes_before": path.stat().st_size, "sha256_before": None, "bytes_after": None, "sha256_after": None, "freed_bytes": 0,
        "checks": {"model_bits": None, "other_keys": None, "evaluation": None},
    }  # fmt: skip
    probe = load_model_checkpoint(path, mmap=True)
    fmt = probe.get("format")
    row.update({"run_id": probe.get("run_id"), "step": probe.get("step"), "status": probe.get("status"), "format_before": fmt})
    del probe
    if fmt == MODEL_ONLY_FORMAT:
        row["skipped"] = "already model-only"
        row["seconds"] = round(time.perf_counter() - started, 1)
        return row
    if fmt != CHECKPOINT_FORMAT:  # load_model_checkpoint가 이미 거절했다 — 방어
        raise ValueError(f"{path}: 알 수 없는 형식 {fmt!r}")
    row["sha256_before"] = sha256_file(path)
    temp: Path | None = None
    try:
        original = load_model_checkpoint(path, mmap=True)
        row["model_tensors"] = len(original["model"])
        row["model_bytes"] = int(sum(t.numel() * t.element_size() for t in original["model"].values()))
        state = model_only_state(original, slimmed={
            "source_sha256": row["sha256_before"], "source_bytes": row["bytes_before"], "at": _now(), "tool": f"scripts/slim_checkpoint.py {SCRIPT_VERSION}",
        })  # fmt: skip
        temp = write_temporary(state, path.parent, prefix=path.name)
        del state
        written = load_model_checkpoint(temp, mmap=True)
        row["checks"]["model_bits"] = compare_model_tensors(original["model"], written["model"])
        row["checks"]["other_keys"] = other_keys_check(original, written)
        del original
        gc.collect()
        if evaluate is not None:
            row["checks"]["evaluation"] = evaluate(temp, written["model"])
        del written
        gc.collect()
        passed = bool(row["checks"]["model_bits"]["equal"] and row["checks"]["other_keys"]["equal"]
                      and (evaluate is None or (row["checks"]["evaluation"] or {}).get("params_equal")))  # fmt: skip
        if not passed:
            row["error"] = "a check failed — the original is untouched"
        elif not dry_run:
            os.replace(temp, path)
            temp = None
            _fsync_directory(path.parent)
            row["replaced"] = True
            row["bytes_after"] = path.stat().st_size
            row["sha256_after"] = sha256_file(path)
            row["freed_bytes"] = row["bytes_before"] - row["bytes_after"]
    except Exception as exc:  # 어떤 실패도 원본을 건드리지 않고 이유로 남긴다
        row["error"] = f"{type(exc).__name__}: {exc}"[:2000]
    finally:
        if temp is not None:
            try:
                temp.unlink()
            except OSError:
                pass
        gc.collect()
    row["seconds"] = round(time.perf_counter() - started, 1)
    return row


def _resolve_approved(files: list[str] | None) -> list[Path]:
    """요청한 경로를 승인 목록과 대조한다 — 저장소 기준 상대 경로나 그 절대 경로(심볼릭 링크를 푼 것)만 받는다."""
    approved = {(REPO / name).resolve(): REPO / name for name in APPROVED}
    wanted = files or list(APPROVED)
    out: list[Path] = []
    for name in wanted:
        candidate = Path(name) if Path(name).is_absolute() else REPO / name
        resolved = candidate.resolve()
        if resolved not in approved:
            raise SystemExit(f"slim_checkpoint: {name}는 사용자가 승인한 아홉 파일이 아니다 — 대체하지 않는다 (승인 목록: scripts/slim_checkpoint.py APPROVED)")
        out.append(approved[resolved])
    return out


def cmd_run(args: argparse.Namespace) -> int:
    paths = _resolve_approved(args.files)
    root = REPO / "artifacts"
    report: dict[str, Any] = {
        "script": SCRIPT_VERSION, "generated_at": _now(), "dry_run": bool(args.dry_run), "approved": list(APPROVED),
        "disk_free_before_bytes": disk_free(root), "files": [],
    }  # fmt: skip
    try:
        report["git"] = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        report["git"] = None
    _log(f"{len(paths)} file(s) · / free {report['disk_free_before_bytes'] / 1e9:.2f} GB · dry run {args.dry_run}")
    out = Path(args.out)
    for path in paths:
        if not path.is_file():
            report["files"].append({"path": str(path.relative_to(REPO)), "replaced": False, "error": "missing"})
            continue
        row = slim_one(path, evaluate=None if args.no_evaluation else serving_load_check, dry_run=args.dry_run)
        row["path"] = str(path.relative_to(REPO))
        row["disk_free_after_bytes"] = disk_free(root)
        report["files"].append(row)
        _log(f"{row['path']}: replaced {row['replaced']} · {row['bytes_before'] / 1e9:.2f} → {(row['bytes_after'] or 0) / 1e9:.2f} GB · "
             f"freed {row['freed_bytes'] / 1e9:.2f} GB · skipped {row['skipped']} · error {row['error']} · {row['seconds']} s")  # fmt: skip
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")  # 파일마다 — 도중에 죽어도 남는다
    report["disk_free_after_bytes"] = disk_free(root)
    report["freed_bytes"] = sum(int(row.get("freed_bytes") or 0) for row in report["files"])
    report["replaced"] = sum(1 for row in report["files"] if row.get("replaced"))
    report["failed"] = [row["path"] for row in report["files"] if row.get("error")]
    out.write_text(json.dumps(report, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    _log(f"done: replaced {report['replaced']} · freed {report['freed_bytes'] / 1e9:.2f} GB · / free {report['disk_free_after_bytes'] / 1e9:.2f} GB · failed {report['failed']}")
    print(f"→ {out}")
    return 0 if not report["failed"] else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="승인한 파일(기본: 아홉 전부)을 차례로 슬림한다")
    run.add_argument("--files", nargs="*", default=None, help="승인 목록 가운데 일부 (저장소 기준 경로)")
    run.add_argument("--dry-run", dest="dry_run", action="store_true", help="검사만 하고 대체하지 않는다")
    run.add_argument("--no-evaluation", dest="no_evaluation", action="store_true", help="평가 경로 적재 검사를 건너뛴다 (시험·진단용 — 이것으로 대체하지 않는다)")
    run.add_argument("--out", required=True)
    run.set_defaults(func=cmd_run)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if getattr(args, "no_evaluation", False) and not args.dry_run:
        raise SystemExit("slim_checkpoint: --no-evaluation은 --dry-run과 함께만 — 평가 경로 검사 없이 원본을 대체하지 않는다")
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
