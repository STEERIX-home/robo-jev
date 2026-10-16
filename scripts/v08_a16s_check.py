#!/usr/bin/env python
"""Task v0.8d — A16s's pipeline check and curves: one or more A16s score files → the curves table, serve's pattern lines and the SVG (CPU).

    python scripts/v08_a16s_check.py --scores v07=<A16s-v07s0-2b.json> --out artifacts/scratch/v08d/smoke/a16s-check.json

Each ``--scores`` item is ``<series>=<file>`` with series one of F, FP, v07, base (several files of one series are averaged as an arm:
seed means per axis question). Writes ``<out>.json`` (per series: :func:`robo_jev.bench.a16s_v08.sweep_summary` and the curves), ``<out>.md``
(accuracy / P(correct) / stay per |d| by sign, and serve's pattern per axis: the stay rate over −20…+15 cm, accuracy at |d| ≥ 25 cm,
P(correct) at −25…−40 cm — serve's v0.7 y question: mostly stay over −20…+15, right only at ≥ 25, P(correct) 0.26–0.31 at −25…−40) and
``<out>.svg`` (:func:`robo_jev.bench.a16s_v08.curves_svg`).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

SERIES = ("F", "FP", "v07", "base")


def serve_pattern(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Per axis: the stay rate over −20…+15 cm, accuracy at |d| ≥ 25 cm, mean P(correct) at −25…−40 cm."""
    out = {}
    for axis in ("x", "y", "z"):
        r = [x for x in rows if x["axis"] == axis]
        mid = [x for x in r if -20 <= x["distance"] <= 15]
        far = [x for x in r if abs(x["distance"]) >= 25]
        neg = [x for x in r if x["distance"] <= -25]
        out[axis] = {"stay_-20..+15": sum(x["stay"] for x in mid) / len(mid) if mid else None,
                     "accuracy_>=25": sum(x["correct"] for x in far) / len(far) if far else None,
                     "p_correct_-25..-40": sum(x["p_correct"] for x in neg) / len(neg) if neg else None, "n": len(r)}
    return out


def main(argv: list[str] | None = None) -> int:
    from robo_jev.bench import a16s_v08 as S
    from robo_jev.decisions.records import read_records

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--records", type=Path, default=S.A16S_DIR / "A16s.jsonl")
    parser.add_argument("--scores", nargs="+", required=True, help="<series>=<A16s score file> (series: F, FP, v07, base)")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    problems = S.check_frozen(args.records.parent)
    if problems:
        raise SystemExit("A16s changed after it was frozen: " + "; ".join(problems))
    records = read_records(args.records)
    files: dict[str, list[Path]] = defaultdict(list)
    for item in args.scores:
        series, _, path = item.partition("=")
        if series not in SERIES or not path:
            raise SystemExit(f"--scores {item!r}: <series>=<file> with series in {SERIES}")
        files[series].append(Path(path))
    rows_by: dict[str, list] = {}
    seeds: dict[str, list] = {}
    for series, paths in files.items():
        per_file = [S.sweep_rows(records, json.loads(p.read_text(encoding="utf-8"))["answers"]) for p in paths]
        if len(per_file) > 2:
            raise SystemExit(f"series {series}: at most two files (an arm's two seeds)")
        rows_by[series] = S.seed_mean_rows(per_file[0], per_file[1]) if len(per_file) == 2 else per_file[0]  # an arm: seed means
        seeds[series] = [S.curves(r) for r in per_file] if len(per_file) > 1 else []
    out = {"kind": "v08-a16s-check", "files": {k: [str(p) for p in v] for k, v in files.items()},
           "series": {k: {"summary": S.sweep_summary(v), "curves": S.curves(v), "serve_pattern": serve_pattern(v)} for k, v in rows_by.items()}}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str) + "\n", encoding="utf-8")
    svg = S.curves_svg({k: v["curves"] for k, v in out["series"].items()}, seeds=seeds)
    args.out.with_suffix(".svg").write_text(svg, encoding="utf-8")
    lines = ["# A16s — the arm's distance sweep (reported)", ""]
    for series, entry in out["series"].items():
        lines += [f"## {series}", "", "serve's pattern per axis (stay over −20…+15 cm · accuracy at |d| ≥ 25 · P(correct) at −25…−40):", ""]
        for axis, p in entry["serve_pattern"].items():
            lines.append(f"- {axis}: stay {p['stay_-20..+15']:.2f} · accuracy {p['accuracy_>=25']:.2f} · P(correct) {p['p_correct_-25..-40']:.2f}")
        lines += ["", "| \\|d\\| cm | accuracy + | accuracy − | P(correct) + | P(correct) − | stay + | stay − |", "|---|---|---|---|---|---|---|"]
        c = {p["distance"]: p for p in entry["curves"]["points"]}
        for m in S.MAGNITUDES:
            a, b = c.get(m), c.get(-m)
            if a and b:
                lines.append(f"| {m:g} | {a['accuracy']:.2f} | {b['accuracy']:.2f} | {a['p_correct']:.2f} | {b['p_correct']:.2f} | {a['stay']:.2f} | {b['stay']:.2f} |")
        lines.append("")
    args.out.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines[:12]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
