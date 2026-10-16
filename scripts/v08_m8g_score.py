#!/usr/bin/env python
"""Task v0.8b — score one model on gate M8g (many options, up to 255) with the package's many-option scoring path (GPU unit).

    artifacts/scratch/v08b/gpu.sh m8g-base scripts/v08_m8g_score.py --model Qwen/Qwen3.5-2B --input artifacts/v08/gates/m8g/M8g.jsonl \\
        --out artifacts/reports/v08/spark/M8g-base-2b.json
    … --checkpoint <run>-fetched/runs/<name>/checkpoint.pt          (a trained LoRA; the checks of diagnose_decisions)
    … --listing 0 --order-seed 2 --out …/M8g-seed2-<tag>.json         (the second order seed, for flip rates)
    … --listing 0 --max-n 60 --items 30 --orders counterbalanced --out …/M8g-balanced-<tag>.json   (single against balanced, bins ≤ 60)

**The path.** Exactly the diagnosis path (:func:`robo_jev.decisions.zero_shot.diagnose_records`, prompt v0c-p3, chat, direct, batch 1): the
package's prompt builder shows the options with the answer codes in order — A–Z, then the two-letter codes (task v0.8a's change, on main
before this runs; the runner's ``CODES[:255]``) — every prompt's option lines are checked against the record
(:func:`robo_jev.decisions.prompt.check_displayed_lines`), and the code probabilities are the softmax over the shown codes of the LM-head rows
at the last position (fp32). **Nothing is lifted here**: every record must pass :func:`robo_jev.decisions.records.validate_record` as main
has it — the records carry the 255-option opt-in of task v0.8a (``meta.option_range``), so before v0.8a is on main the script refuses them
(the smoke test uses ≤ 26-option copies carrying v0.7's opt-in).

**Orders.** Default: the record's own order (``--order-seed none`` → seed None; the gate fixes the positions). ``--order-seed K``: the package's
seeded shuffle with seed K (key = the record id) — a second order of the same records, for flip rates. ``--orders counterbalanced``: every
cyclic shift of the shown order (n listings per record; the answer is their mean).

GPU rules (GB10 unified memory): the fence (``--gpu-memory-fraction``, default 0.6) before the first CUDA allocation, ``require_free`` before
loading (2B ≥ 20 GiB). ``--model tiny`` is the CPU test model. Writes ``<out>.json`` (config, run, a per-bin glance) and
``<out>.predictions.jsonl`` (one line per record).
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

SCRIPT_VERSION = "v08-m8g-score-1"
TINY = "tiny"
FALLBACK_MIN_FREE = 30 * 2**30


def _log(message: str) -> None:
    print(f"[v08-m8g {dt.datetime.now().strftime('%H:%M:%S')}] {message}", file=sys.stderr, flush=True)


def _git_sha() -> str | None:
    try:
        return subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True).stdout.strip() or None
    except (OSError, subprocess.CalledProcessError):
        return None


def order_seed(text: str) -> int | None:
    return None if str(text).lower() == "none" else int(text)


def select(records: list[dict[str, Any]], *, listing: int | None, max_n: int | None, items: int | None) -> list[dict[str, Any]]:
    """The records to score: those of `listing` (``meta.listing``), with at most `max_n` options, of the first `items` items by rank."""
    out = [r for r in records if (listing is None or int(r["meta"]["listing"]) == listing) and (max_n is None or int(r["meta"]["n"]) <= max_n)]
    if items is not None:
        ranks = sorted({int(r["meta"]["rank"]) for r in out})[: int(items)]
        keep = set(ranks)
        out = [r for r in out if int(r["meta"]["rank"]) in keep]
    return out


def load_checked(path: Path) -> list[dict[str, Any]]:
    """Every record must pass main's :func:`validate_record` (the opt-in decides the range — nothing lifted); ids unique."""
    from robo_jev.decisions.records import read_records, validate_record

    records = read_records(path)
    problems = {str(r.get("id")): p for r in records if (p := validate_record(r))}
    if problems:
        first = list(problems.items())[:3]
        raise SystemExit(f"{len(problems)} of {len(records)} records fail main's checks — is task v0.8a's 255-option opt-in on main? first: {first}")
    ids = [str(r["id"]) for r in records]
    if len(set(ids)) != len(ids):
        raise SystemExit("duplicate record ids")
    return records


def glance(records: list[dict[str, Any]], predictions: list[dict[str, Any]]) -> dict[str, Any]:
    """Top-1 accuracy per bin and code kind — a glance for the log; the verdict reads the predictions."""
    from robo_jev.bench.gates_v08 import M8G_QUESTION, bin_of

    by_id = {str(p["id"]): p for p in predictions}
    sums: dict[str, list[float]] = defaultdict(lambda: [0.0, 0])
    for r in records:
        p = by_id.get(str(r["id"]))
        if p is None:
            continue
        probs = p["questions"][M8G_QUESTION]["answer"]["probabilities"]
        keys = list(r["request"]["questions"][M8G_QUESTION]["criteria"])
        top = max(keys, key=lambda k: (probs[k], -keys.index(k)))
        key = f"{bin_of(r['meta']['n'])}/{'two' if r['meta']['two_letter'] else 'one'}"
        sums[key][0] += float(top == r["meta"]["correct"])
        sums[key][1] += 1
    return {k: {"accuracy": a / n, "listings": n} for k, (a, n) in sorted(sums.items())}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", required=True, help="Qwen/Qwen3.5-2B (weights in artifacts/models/) | tiny (tests)")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--order-seed", type=order_seed, default=None, help="none (the record's order, default) or an int (the package's seeded shuffle)")
    parser.add_argument("--orders", choices=("seeded", "counterbalanced"), default="seeded")
    parser.add_argument("--listing", type=int, default=None, help="only records of this meta.listing")
    parser.add_argument("--max-n", type=int, default=None, help="only records with at most this many options")
    parser.add_argument("--items", type=int, default=None, help="only the first K items by meta.rank")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--gpu-memory-fraction", type=float, default=DEFAULT_FRACTION)
    parser.add_argument("--title", default=None)
    args = parser.parse_args(argv)
    guard = limit_gpu_memory(args.gpu_memory_fraction)  # before the first CUDA allocation
    memory_start = memory_report()
    _log(f"gpu guard {guard} · memory at start {memory_start}")
    from robo_jev.decisions.prompt import PROMPT_VERSION, fixed_text_digest
    from robo_jev.decisions.records import file_sha256
    from robo_jev.decisions.zero_shot import MIN_FREE_BYTES, diagnose_records
    from robo_jev.model.backbone_qwen import QwenBackbone
    from robo_jev.model.tokenizer import available_tokenizer, describe_tokenizer, load_tokenizer

    records = select(load_checked(args.input), listing=args.listing, max_n=args.max_n, items=args.items)
    if not records:
        raise SystemExit("no record selected")
    _log(f"{len(records)} records selected (listing {args.listing}, max n {args.max_n}, items {args.items}); order seed {args.order_seed}, {args.orders}")
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

        checkpoint = apply_checkpoint(backbone, args.checkpoint, tokenizer_sha256=describe_tokenizer(tokenizer_id)["sha256"], prompt_version=PROMPT_VERSION)
        checkpoint["sha256"] = file_sha256(args.checkpoint)
        model_label = f"{args.model}+{checkpoint['mode']}:{checkpoint['run_id']}@{checkpoint['step']}"
    _log(f"loaded {model_label} in {load_seconds:.1f} s · memory {memory_report()}")
    started = time.perf_counter()
    predictions = diagnose_records(records, backbone, tokenizer, model_id=model_label, mode="direct", fmt="chat", seed=args.order_seed,
                                   orders=args.orders, batch=args.batch, log=_log, version=PROMPT_VERSION)
    seconds = time.perf_counter() - started
    import torch

    peak = int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else None
    listings = [l for p in predictions for q in p["questions"].values() for l in q["listings"]]
    config = {
        "model": model_label, "checkpoint": checkpoint, "input": str(args.input), "input_sha256": file_sha256(args.input), "records": len(records),
        "record_ids_sha256": __import__("hashlib").sha256("\n".join(str(r["id"]) for r in records).encode()).hexdigest(),
        "listing": args.listing, "max_n": args.max_n, "items": args.items, "order_seed": args.order_seed, "orders": args.orders,
        "mode": "direct", "format": "chat", "prompt_version": PROMPT_VERSION, "prompt_digest": fixed_text_digest(PROMPT_VERSION), "batch": args.batch,
        "option_range": (records[0].get("meta") or {}).get("option_range"), "tokenizer": describe_tokenizer(tokenizer_id),
        "backbone_manifest": getattr(backbone, "manifest", None), "script_version": SCRIPT_VERSION, "git": _git_sha(),
        "created": dt.datetime.now().astimezone().isoformat(timespec="seconds"), "title": args.title,
    }
    run = {"seconds": round(seconds, 2), "load_seconds": round(load_seconds, 2), "prompts": len(listings),
           "prompt_tokens": sum(int(l["prompt_tokens"]) for l in listings), "ms_per_token": round(1000 * seconds / max(1, sum(int(l["prompt_tokens"]) for l in listings)), 4),
           "gpu": {"guard": guard, "memory_at_start": memory_start, "memory_at_end": memory_report(), "peak_allocated_bytes": peak}}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.with_suffix(".predictions.jsonl").open("w", encoding="utf-8") as handle:
        for prediction in predictions:
            handle.write(json.dumps(prediction, ensure_ascii=False) + "\n")
    payload = {"config": config, "run": run, "glance": glance(records, predictions)}
    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=1, default=str) + "\n", encoding="utf-8")
    _log(f"{len(predictions)} records, {len(listings)} prompts, {run['prompt_tokens']:,} tokens in {seconds:.1f} s ({run['ms_per_token']} ms/token) "
         f"→ {args.out} · glance {payload['glance']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
