#!/usr/bin/env python
"""Task v0.8c — A16's pipeline check: v0.7's arm answers as robo-jev-serve stored them, against the rule and serve's published numbers, then
our own scoring of v0.7-s0 on the same requests and the y-phrase ablation (CPU; reads the frozen A16 files and the given score files).

    python scripts/v08_a16_check.py --out artifacts/scratch/v08c/smoke/a16-check.json \\
        [--scores <A16-v07s0-2b.json>] [--ablation <A16y-v07s0-2b.json>]

1. *Stored answers* (``A16-stored-v07.json``, from ``arm-servo.jsonl``): rule agreement total and per axis, sign errors, the stay rate on
   small distances — compared with serve's v0.7 numbers as relayed (:data:`SERVE_V07`): agreement 116 / 270 (x 1, y 28, z 87 of 90).
2. *Our scoring* (``--scores``, the bundle's runner under serve's conditions: default key, single order seed 1, the bundle's prompt): the same
   readings, and per question the share of the 540 questions whose top option equals the stored one, with the largest probability difference.
3. *The ablation* (``--ablation``: the y question with " (up is plus)" removed): y agreement and sign errors before and after, against serve's
   v0.7 report (sign errors 27 → 4, agreement 27 → 50 of 90 — serve's own re-run; the stored answers give 28 and 26 before).
The boundary sensitivity (±0.2 read as stay, "within 0.2 cm") is reported beside every agreement.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

#: serve's v0.7 numbers on these requests (relayed by the controller, 2026-10-16).
SERVE_V07 = {"agreement": {"total": [116, 270], "x": [1, 90], "y": [28, 90], "z": [87, 90]},
             "ablation_y": {"sign_errors": {"before": 27, "after": 4}, "agreement": {"before": [27, 90], "after": [50, 90]}}}


def _counts(summary: dict[str, Any]) -> dict[str, Any]:
    agree = summary["agreement_by_axis"]
    return {"total": [int(summary["agreement"]["count"]), summary["agreement"]["n"]],
            **{axis: [int(agree[axis]["count"]), agree[axis]["n"]] for axis in agree},
            "sign_errors": {axis: int(v["count"]) for axis, v in summary["sign_errors_by_axis"].items()},
            "stay_rate_small": summary["stay_rate_small"]["value"]}


def compare_tops(records: list[dict[str, Any]], ours: dict[str, Any], stored: dict[str, Any], *, reading: str) -> dict[str, Any]:
    """Per question name: how often our top option equals the stored one, and the largest |Δp| over the options."""
    out: dict[str, Any] = {}
    for record in records:
        rid = str(record["id"])
        for name in record["request"]["questions"]:
            a = (((ours.get(rid) or {}).get(name) or {}).get(reading)) or None
            b = (((stored.get(rid) or {}).get(name) or {}).get("stored")) or None
            if not a or not b:
                continue
            entry = out.setdefault(name, {"n": 0, "same_top": 0, "max_abs_dp": 0.0})
            entry["n"] += 1
            entry["same_top"] += a["choice"] == b["choice"]
            pa, pb = a.get("probabilities") or {}, b.get("probabilities") or {}
            entry["max_abs_dp"] = max(entry["max_abs_dp"], max((abs(float(pa.get(k, 0.0)) - float(pb.get(k, 0.0))) for k in set(pa) | set(pb)), default=0.0))
    total = {"n": sum(e["n"] for e in out.values()), "same_top": sum(e["same_top"] for e in out.values()),
             "max_abs_dp": max((e["max_abs_dp"] for e in out.values()), default=None)}
    return {"by_question": out, "total": total}


def main(argv: list[str] | None = None) -> int:
    from robo_jev.bench import r16_v08 as R
    from robo_jev.decisions.records import read_records

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--records", type=Path, default=R.R16_DIR / "A16.jsonl")
    parser.add_argument("--stored", type=Path, default=R.R16_DIR / "A16-stored-v07.json")
    parser.add_argument("--scores", type=Path, default=None, help="our A16 score file (reading single-s1)")
    parser.add_argument("--ablation", type=Path, default=None, help="our A16y score file (the y question without ' (up is plus)')")
    parser.add_argument("--reading", default="single-s1")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    problems = R.check_frozen(args.records.parent)
    if problems:
        raise SystemExit("A16 files changed after they were frozen: " + "; ".join(problems))
    records = read_records(args.records)
    stored = json.loads(args.stored.read_text(encoding="utf-8"))["answers"]
    out: dict[str, Any] = {"kind": "v08-a16-pipeline-check", "serve_v07": SERVE_V07, "records": len(records)}
    s_main = R.arm_summary(R.arm_rows(records, stored, reading="stored"))
    s_alt = R.arm_summary(R.arm_rows(records, stored, reading="stored", boundary_stay=True))
    out["stored"] = {"main": _counts(s_main), "boundary_stay": _counts(s_alt), "summary": s_main}
    got = out["stored"]["main"]
    out["stored"]["reproduces_serve"] = all(got[k] == v for k, v in SERVE_V07["agreement"].items())
    lines = [f"A16 pipeline check — serve's v0.7 agreement {SERVE_V07['agreement']}", "",
             f"stored answers: agreement {got['total'][0]}/{got['total'][1]} (x {got['x'][0]}, y {got['y'][0]}, z {got['z'][0]}) — reproduces serve: "
             f"{out['stored']['reproduces_serve']}; sign errors {got['sign_errors']}; stay rate on 0.2-3 cm {got['stay_rate_small']:.3f}; "
             f"with +-0.2 read as stay: {out['stored']['boundary_stay']['total'][0]}/270 (x {out['stored']['boundary_stay']['x'][0]})"]
    if args.scores is not None:
        scores = json.loads(args.scores.read_text(encoding="utf-8"))
        ours = scores["answers"]
        o_main = R.arm_summary(R.arm_rows(records, ours, reading=args.reading))
        o_alt = R.arm_summary(R.arm_rows(records, ours, reading=args.reading, boundary_stay=True))
        out["ours"] = {"file": str(args.scores), "model": scores.get("model"), "main": _counts(o_main), "boundary_stay": _counts(o_alt), "summary": o_main,
                       "tops_vs_stored": compare_tops(records, ours, stored, reading=args.reading)}
        o = out["ours"]["main"]
        t = out["ours"]["tops_vs_stored"]["total"]
        lines.append(f"ours ({scores.get('model')}): agreement {o['total'][0]}/{o['total'][1]} (x {o['x'][0]}, y {o['y'][0]}, z {o['z'][0]}); sign errors "
                     f"{o['sign_errors']}; stay rate on 0.2-3 cm {o['stay_rate_small']:.3f}; top option equal to the stored one on {t['same_top']}/{t['n']} "
                     f"questions, max |dp| {t['max_abs_dp']:.4f}")
        for name, e in out["ours"]["tops_vs_stored"]["by_question"].items():
            lines.append(f"  {name}: same top {e['same_top']}/{e['n']}, max |dp| {e['max_abs_dp']:.4f}")
    if args.ablation is not None:
        ablated = json.loads(args.ablation.read_text(encoding="utf-8"))["answers"]
        a_main = R.arm_summary(R.arm_rows(records, ablated, reading=args.reading, axes=("y",)))
        before = out.get("ours", {}).get("main") or got
        out["ablation_y"] = {"file": str(args.ablation), "after": _counts(a_main), "summary": a_main,
                             "before_source": "ours" if "ours" in out else "stored"}
        a = out["ablation_y"]["after"]
        lines.append(f"y without ' (up is plus)': agreement {before['y'][0]} -> {a['y'][0]}/90, sign errors {before['sign_errors']['y']} -> "
                     f"{a['sign_errors']['y']} (serve's v0.7: agreement 27 -> 50, sign errors 27 -> 4)")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str) + "\n", encoding="utf-8")
    args.out.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
