#!/usr/bin/env python
"""Task iface (M1) — score one model on the many-option set with the model's own prompt and readout (GPU unit).

    artifacts/scratch/iface/gpu.sh m1-base scripts/iface_m1_score.py --model Qwen/Qwen3.5-2B --lift-option-cap \\
        --input artifacts/iface/m1/m1-set.jsonl --out artifacts/iface/m1/pred/M1-base-2b.json
    … --checkpoint artifacts/cloud/runs/<run>-fetched/runs/<name>/checkpoint.pt (a trained LoRA; the same checks as diagnose_decisions)

**The cap.** Decisions records allow 2–8 choice options (:data:`robo_jev.decisions.records.OPTION_RANGE`, unchanged). This script refuses
any record that breaks it **unless** ``--lift-option-cap`` is given; with the flag it allows 2–26 options (A–Z, each a verified single token
at the answer boundary — :func:`robo_jev.model.zero_shot.verify_codes`) and every other check of
:func:`robo_jev.decisions.records.validate_record` still applies (:func:`robo_jev.bench.iface.m1_problems`). Nothing else is lifted, and no
default anywhere changes.

**Prompt and readout** — exactly the model's: :func:`robo_jev.decisions.zero_shot.diagnose_records` with prompt v0c-p3, chat format, direct
answer, batch 1 (no padding), one listing per record shown in the record's own order (``seed=None``, ``orders="seeded"`` — the M1 set fixes the
positions; nothing is reshuffled), every prompt's shown option lines checked against the record
(:func:`robo_jev.decisions.prompt.check_displayed_lines`), and the code probabilities = softmax over the shown codes of the LM-head rows at the
last position (fp32; :func:`robo_jev.decisions.zero_shot.score_code_prompts`; the code mass is the codes' share of the full vocabulary).

GPU rules (GB10 unified memory): the fence (``--gpu-memory-fraction``, default 0.6, :func:`robo_jev.gpu.limit_gpu_memory`) before the first
CUDA allocation and ``require_free`` before loading (2B ≥ 20 GiB, :data:`robo_jev.decisions.zero_shot.MIN_FREE_BYTES`). ``--model tiny`` is the
CPU test model. Writes ``<out>.json`` (config, run, a per-N accuracy glance) and ``<out>.predictions.jsonl`` (one line per record).
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from robo_jev.gpu import DEFAULT_FRACTION, limit_gpu_memory, memory_report, require_free  # noqa: E402

SCRIPT_VERSION = "iface-m1-score-1"
TINY = "tiny"
FALLBACK_MIN_FREE = 30 * 2**30
SEALED = (REPO / "artifacts" / "t9", REPO / "artifacts" / "v06" / "a4")


def _log(message: str) -> None:
    print(f"[iface-m1 {dt.datetime.now().strftime('%H:%M:%S')}] {message}", file=sys.stderr, flush=True)


def _git_sha() -> str | None:
    try:
        return subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True).stdout.strip() or None
    except (OSError, subprocess.CalledProcessError):
        return None


def _refuse_sealed(path: Path) -> None:
    resolved = Path(path).resolve()
    for place in SEALED:
        place = place.resolve()
        if resolved == place or place in resolved.parents:
            raise SystemExit(f"{path}: a sealed gate ({place}) — never opened by this task")


def load_checked(path: Path, *, lift_cap: bool, limit_items: int | None = None) -> list[dict[str, Any]]:
    """The M1 records, every one checked (:func:`robo_jev.bench.iface.m1_problems`); refuses the file if any record fails (with the cap
    in force, every listing with more than 8 options fails). `limit_items` keeps the listings of the first items (pilots and tests)."""
    from robo_jev.bench.iface import m1_problems
    from robo_jev.decisions.records import read_records

    _refuse_sealed(path)
    records = read_records(path)
    if limit_items is not None:
        keep = []
        for record in records:
            item = record["meta"]["item"]
            if item not in keep:
                keep.append(item)
        keep = set(keep[:limit_items])
        records = [r for r in records if r["meta"]["item"] in keep]
    problems = {str(r.get("id")): p for r in records if (p := m1_problems(r, lift_cap=lift_cap))}
    if problems:
        first = list(problems.items())[:3]
        hint = "" if lift_cap else " — the M1 set has 9-26 options: pass --lift-option-cap (the only place the 2-8 cap is lifted)"
        raise SystemExit(f"{len(problems)} of {len(records)} records fail their checks{hint}; first: {first}")
    ids = [str(r["id"]) for r in records]
    if len(set(ids)) != len(ids):
        raise SystemExit("duplicate record ids")
    return records


def _glance(records: list[dict[str, Any]], predictions: list[dict[str, Any]]) -> dict[str, Any]:
    """Accuracy and p(correct) per N — a glance for the log, not the verdict (scripts/iface_verdict.py reads the predictions)."""
    from robo_jev.bench.iface import m1_rows

    sums: dict[int, list[float]] = defaultdict(lambda: [0.0, 0.0, 0])
    for row in m1_rows(records, predictions):
        entry = sums[row["n"]]
        entry[0] += row["correct"]
        entry[1] += row["p_correct"]
        entry[2] += 1
    return {str(n): {"accuracy": a / k, "p_correct": p / k, "listings": k} for n, (a, p, k) in sorted(sums.items())}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", required=True, help="Qwen/Qwen3.5-2B (weights in artifacts/models/) | tiny (tests)")
    parser.add_argument("--checkpoint", type=Path, default=None, help="a trained checkpoint (LoRA or full) applied on --model")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--lift-option-cap", action="store_true", help="allow 2-26 options (this script only); without it > 8 options are refused")
    parser.add_argument("--batch", type=int, default=1, help="scoring batch (1 = no padding, the default of every diagnosis)")
    parser.add_argument("--limit-items", type=int, default=None, help="pilots and tests: only the first K items")
    parser.add_argument("--gpu-memory-fraction", type=float, default=DEFAULT_FRACTION)
    parser.add_argument("--title", default=None)
    args = parser.parse_args(argv)
    guard = limit_gpu_memory(args.gpu_memory_fraction)  # before the first CUDA allocation
    memory_start = memory_report()
    _log(f"gpu guard {guard} · memory at start {memory_start}")
    from robo_jev.decisions.prompt import DEFAULT_FORMAT, PROMPT_VERSION, fixed_text_digest
    from robo_jev.decisions.records import file_sha256
    from robo_jev.decisions.zero_shot import MIN_FREE_BYTES, diagnose_records
    from robo_jev.model.backbone_qwen import QwenBackbone
    from robo_jev.model.tokenizer import available_tokenizer, describe_tokenizer, load_tokenizer

    records = load_checked(args.input, lift_cap=args.lift_option_cap, limit_items=args.limit_items)
    _log(f"{len(records)} records checked (cap lifted: {args.lift_option_cap})")
    tokenizer_id = available_tokenizer()[0]
    tokenizer = load_tokenizer(tokenizer_id)
    if args.model != TINY:
        require_free(MIN_FREE_BYTES.get(args.model, FALLBACK_MIN_FREE), what=f"load {args.model}")
    started = time.perf_counter()
    backbone = QwenBackbone.tiny(seed=4, vocab_size=tokenizer.get_vocab_size()) if args.model == TINY else QwenBackbone.load(args.model)
    load_seconds = time.perf_counter() - started
    model_label, checkpoint = args.model, None
    if args.checkpoint is not None:
        from robo_jev.decisions.train import apply_checkpoint

        _refuse_sealed(args.checkpoint)
        checkpoint = apply_checkpoint(backbone, args.checkpoint, tokenizer_sha256=describe_tokenizer(tokenizer_id)["sha256"], prompt_version=PROMPT_VERSION)
        checkpoint["sha256"] = file_sha256(args.checkpoint)
        model_label = f"{args.model}+{checkpoint['mode']}:{checkpoint['run_id']}@{checkpoint['step']}"
    _log(f"loaded {model_label} in {load_seconds:.1f} s · memory {memory_report()}")
    started = time.perf_counter()
    predictions = diagnose_records(records, backbone, tokenizer, model_id=model_label, mode="direct", fmt="chat", seed=None, orders="seeded",
                                   batch=args.batch, log=_log, version=PROMPT_VERSION)
    seconds = time.perf_counter() - started
    import torch

    peak = int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else None
    listings = [l for p in predictions for q in p["questions"].values() for l in q["listings"]]
    config = {
        "model": model_label, "checkpoint": checkpoint, "input": str(args.input), "input_sha256": file_sha256(args.input), "records": len(records),
        "limit_items": args.limit_items, "lift_option_cap": bool(args.lift_option_cap), "mode": "direct", "format": "chat",
        "format_is_default": DEFAULT_FORMAT == "chat", "prompt_version": PROMPT_VERSION, "prompt_digest": fixed_text_digest(PROMPT_VERSION),
        "orders": "controlled: one listing per record in the record's own order (seed None, orders 'seeded'); no reshuffle", "batch": args.batch,
        "tokenizer": describe_tokenizer(tokenizer_id), "backbone_manifest": getattr(backbone, "manifest", None), "script_version": SCRIPT_VERSION,
        "git": _git_sha(), "created": dt.datetime.now().astimezone().isoformat(timespec="seconds"), "title": args.title,
    }
    run = {"seconds": round(seconds, 2), "load_seconds": round(load_seconds, 2), "prompts": len(listings),
           "prompt_tokens": sum(int(l["prompt_tokens"]) for l in listings),
           "gpu": {"guard": guard, "memory_at_start": memory_start, "memory_at_end": memory_report(), "peak_allocated_bytes": peak}}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.with_suffix(".predictions.jsonl").open("w", encoding="utf-8") as handle:
        for prediction in predictions:
            handle.write(json.dumps(prediction, ensure_ascii=False) + "\n")
    payload = {"config": config, "run": run, "glance_by_n": _glance(records, predictions)}
    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=1, default=str) + "\n", encoding="utf-8")
    _log(f"{len(predictions)} records, {len(listings)} prompts in {seconds:.1f} s → {args.out} · glance {payload['glance_by_n']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
