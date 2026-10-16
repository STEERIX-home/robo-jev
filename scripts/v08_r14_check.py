#!/usr/bin/env python
"""Task v0.8b — R14's pipeline check: count v0.7's answers on robo-jev-serve's own sample exactly as serve's HANDOFF.md appendix B did and
compare them with the numbers serve published for v0.7 (CPU; reads one or more ``R14serve-<tag>.json`` score files).

    python scripts/v08_r14_check.py --scores artifacts/scratch/v08b/smoke/R14serve-v07s0-2b.json --out artifacts/scratch/v08b/smoke/r14-check.json

Serve's counting (appendix B, robo-jev-serve @ 7b54004): per environment, the sampled requests sent with 0, 1 and 2 trailing spaces; a request
counts only if all three runs answered; per question of the ORIGINAL request with type ``choice`` and ≥ 2 options (scores asked as level
choices are not counted, nor 1-option choices), ``counted`` += 1, ``flips`` += (more than one distinct top-1 over the three runs), and the
top-1 of each run is tallied. Serve's v0.7 numbers (the merged model on the HF endpoint, L4, the fused path for requests with 2+ questions)
are :data:`SERVE_V07`; this check runs the bundle's runner on the Spark (LoRA checkpoint or the merged folder), so near-tied answers may
differ — the comparison lists every difference, it does not judge.
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

ORDERS = ("spaces-0", "spaces-1", "spaces-2")
#: serve's published v0.7 results on its sample (HANDOFF.md §3.3 and the controller's relay, 2026-10-16): per (env, question) the requests
#: counted and the requests whose top-1 changed with the order; for the stable ones also the top-1 and how often (over the three orders).
SERVE_V07: dict[tuple[str, str], dict[str, Any]] = {
    ("lab", "reader"): {"counted": 20, "flips": 0, "top": "read", "top_count": 60},
    ("humanoid", "advance"): {"counted": 20, "flips": 0, "top": "proceed", "top_count": 60},
    ("delivery", "manoeuvre"): {"counted": 19, "flips": 1, "top": "stop", "top_count": 47},
    ("forklift", "advance"): {"counted": 20, "flips": 1, "top": "proceed", "top_count": 59},
    ("delivery", "announce"): {"counted": 17, "flips": 0, "top": "none", "top_count": 51},
    ("lab", "correction"): {"counted": 14, "flips": 0, "top": "taught", "top_count": 42},
    ("lab", "move"): {"counted": 14, "flips": 11},
    ("delivery", "offset"): {"counted": 17, "flips": 13},
    ("humanoid", "gaze"): {"counted": 20, "flips": 12},
    ("cafe", "say"): {"counted": 20, "flips": 10},
    ("forklift", "maneuver"): {"counted": 5, "flips": 5},
    ("forklift", "mast"): {"counted": 20, "flips": 5},
    ("cafe", "advance"): {"counted": 20, "flips": 3},
    ("cafe", "attend"): {"counted": 20, "flips": 3},
    ("lab", "advance"): {"counted": 20, "flips": 2},
    ("delivery", "pace"): {"counted": 17, "flips": 2},
    ("humanoid", "maneuver"): {"counted": 17, "flips": 2},
    ("cafe", "intent"): {"counted": 17, "flips": 2},
}


def count(sample: list[dict[str, Any]], scores: dict[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    """Serve's appendix-B tallies of one score file on the sample (module docstring)."""
    answers = scores.get("answers") or {}
    out: dict[tuple[str, str], dict[str, Any]] = {}
    for record in sample:
        env, rid = record["meta"]["env"], str(record["id"])
        got = answers.get(rid)
        if got is None:
            continue
        for name, question in record["request"]["questions"].items():
            if question.get("type") != "choice" or len(question.get("criteria") or {}) < 2:
                continue
            runs = got.get(name) or {}
            if not all(o in runs for o in ORDERS):
                continue
            tops = [runs[o]["choice"] for o in ORDERS]
            entry = out.setdefault((env, name), {"counted": 0, "flips": 0, "tops": collections.Counter()})
            entry["counted"] += 1
            entry["flips"] += len(set(tops)) > 1
            entry["tops"].update(tops)
    return out


def compare(ours: dict[tuple[str, str], dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for key in sorted(set(ours) | set(SERVE_V07)):
        mine, ref = ours.get(key), SERVE_V07.get(key)
        row: dict[str, Any] = {"env": key[0], "question": key[1], "ours": None, "serve_v07": ref}
        if mine is not None:
            row["ours"] = {"counted": mine["counted"], "flips": mine["flips"], "tops": dict(mine["tops"].most_common())}
            if ref and ref.get("top"):
                row["ours"]["top_count"] = mine["tops"].get(ref["top"], 0)
        if mine is not None and ref is not None:
            row["same_counted"] = mine["counted"] == ref["counted"]
            row["flips_diff"] = mine["flips"] - ref["flips"]
            if ref.get("top"):
                row["top_count_diff"] = mine["tops"].get(ref["top"], 0) - ref["top_count"]
        rows.append(row)
    return rows


def main(argv: list[str] | None = None) -> int:
    from robo_jev.bench.r14_v08 import R14_DIR
    from robo_jev.decisions.records import read_records

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sample", type=Path, default=R14_DIR / "R14-serve-sample.jsonl")
    parser.add_argument("--scores", type=Path, nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    sample = read_records(args.sample)
    merged: dict[str, Any] = {"answers": {}}
    for path in args.scores:
        merged["answers"].update(json.loads(path.read_text(encoding="utf-8"))["answers"])
    rows = compare(count(sample, merged))
    envs = sorted({r["meta"]["env"] for r in sample if str(r["id"]) in merged["answers"]})
    compared = [r for r in rows if r["ours"] is not None and r["serve_v07"] is not None]
    payload = {"kind": "v08-r14-pipeline-check", "sample": str(args.sample), "scores": [str(p) for p in args.scores], "envs_scored": envs,
               "rows": rows, "compared": len(compared), "same_counted": sum(bool(r.get("same_counted")) for r in compared),
               "exact_flips": sum(r.get("flips_diff") == 0 for r in compared), "max_abs_flips_diff": max((abs(r["flips_diff"]) for r in compared), default=None)}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=1, default=str) + "\n", encoding="utf-8")
    lines = [f"R14 pipeline check (envs scored: {', '.join(envs)}): {payload['compared']} readings compared, counted equal {payload['same_counted']}, "
             f"flips equal {payload['exact_flips']}, max |flips diff| {payload['max_abs_flips_diff']}", "",
             "| env | question | counted ours / serve | flips ours / serve | stable top: ours / serve |", "|---|---|---|---|---|"]
    for r in rows:
        if r["ours"] is None:
            continue
        ref = r["serve_v07"] or {}
        top = f"{ref.get('top')} {r['ours'].get('top_count')} / {ref.get('top_count')}" if ref.get("top") else ", ".join(f"{k} {v}" for k, v in list(r["ours"]["tops"].items())[:3])
        lines.append(f"| {r['env']} | {r['question']} | {r['ours']['counted']} / {ref.get('counted', '—')} | {r['ours']['flips']} / {ref.get('flips', '—')} | {top} |")
    args.out.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
