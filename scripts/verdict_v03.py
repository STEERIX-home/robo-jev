#!/usr/bin/env python
"""Apply the PRE-DECLARED v0.3 VERDICT RULE (progress.md, written 2026-10-12 ~20:30 KST) mechanically to fetched predictions.

    python scripts/verdict_v03.py --v03-dir artifacts/cloud/runs/<run>-fetched/reports --out artifacts/reports/v03/verdict.json

The rule, verbatim in what it reads (System-1 metrics of :mod:`robo_jev.bench.system1`, single-order reading, paired cluster bootstrap
95 %, 2,000 draws, seed 0, cluster = record):

* **(O) OUTSIDE, primary** — T8b vs the untrained base: balanced band accuracy Δ interval lower bound > 0 AND severe-error rate point ≤
  the base's AND direction point ≥ 0.70.
* **(H) HOME non-regression** — vs v0.1-mix on T2–T4 (pooled): balanced band Δ interval lower bound > −0.05 AND severe-error rate point
  ≤ v0.1-mix + 0.01.
* **(G) GENERAL** — G2 accuracy v0.3 − base (pod) ≥ −0.02 point AND lower bound > −0.045 (the v0.2 rule's reading:
  :func:`robo_jev.decisions.single_probe.compare_probe`, cluster = record).
* v0.3 **PASSES** iff O and H and G.

Inputs (defaults): the v0.3 job's own predictions ``<v03-dir>/after-v03-2b-<set>.predictions.jsonl``; v0.1-mix's T2–T4 from its pod;
the untrained base on T8b from the T8 task (**Spark GB10, not the H100 pod** — the job has no same-pod base on T8b; the note is in the
output); the untrained base on G2 from the v0.1 main pod (same H100 hardware). Every prediction file is checked against the set it
scores — its diagnosis summary (``<name>.json``) must record the sha256 of the sealed file used here, and its scoring conditions (prompt
version, format, orders, mode, shuffle seed) must match across the compared files — and the coverage of each set is reported.
Comparisons are paired on the items both answered.

Secondary readings (reported, never gating): T8b comparison, outcome, plain band accuracy and Brier skill vs the base and vs v0.1-mix
(its Spark T8b predictions), T8a accuracy vs the base, and — with ``--with-diagnose-compare`` — T1–T7 Brier and default-criterion regret
vs v0.1-mix through ``scripts/diagnose_decisions.py --compare`` (the v0.2 verdict's reading). T6 fact order and bench-v0 Brier skill
are left to their existing tools.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

SCRIPT_VERSION = "v03-verdict-1"
RULE = {
    "written": "2026-10-12 ~20:30 KST, before any v0.3 training input existed (progress.md: PRE-DECLARED v0.3 VERDICT RULE)",
    "O": "T8b vs the untrained base: balanced band accuracy delta interval lower bound > 0 AND severe-error rate point <= the base's AND "
         "direction point >= 0.70",
    "H": "T2-T4 vs v0.1-mix: balanced band delta interval lower bound > -0.05 AND severe-error rate point <= v0.1-mix + 0.01",
    "G": "G2 accuracy v0.3 - base (pod) >= -0.02 point AND interval lower bound > -0.045",
    "verdict": "PASSES iff O and H and G -> release candidate (user decides)",
}
#: Thresholds of the rule.
O_BAND_LOW = 0.0
O_DIRECTION = 0.70
H_BAND_LOW = -0.05
H_SEVERE_SLACK = 0.01
G_POINT = -0.02
G_LOW = -0.045
HOME = ("T2", "T3", "T4")
#: T8b split by physical law (controller, 2026-10-12): the MuJoCo training families tilt and tower share their laws with T8b's deck and
#: stack families, so push is the only family whose law training never shows. Reported next to the gate; the gate is unchanged.
T8B_GROUPS = {"seen_law (deck_brake, deck_turn, stack)": ("deck_brake", "deck_turn", "stack"), "unseen_law (push)": ("push",)}
SCORING_KEYS = ("mode", "format", "prompt_version", "orders", "shuffle_seed")

TESTS = REPO / "artifacts" / "simgen" / "v0b" / "tests"
SET_PATHS = {**{s: TESTS / f"{s}.jsonl" for s in ("T1", "T2", "T3", "T4", "T5", "T6", "T7")}, "G2": REPO / "artifacts" / "simgen" / "probes" / "G2.jsonl",
             "T8a": REPO / "artifacts" / "t8" / "T8a.jsonl", "T8b": REPO / "artifacts" / "t8" / "T8b.jsonl"}  # fmt: skip
MIX_DIR = REPO / "artifacts" / "cloud" / "runs" / "v01-mix-01-fetched" / "reports"
BASE_T8 = REPO / "artifacts" / "reports" / "t8"
BASE_G2_POD = REPO / "artifacts" / "cloud" / "runs" / "v01-main-01-fetched" / "reports" / "base-2b-G2.predictions.jsonl"


def _log(message: str) -> None:
    print(f"[verdict-v03 {dt.datetime.now().strftime('%H:%M:%S')}] {message}", file=sys.stderr, flush=True)


# --------------------------------------------------------------------------
# The rule (pure)
# --------------------------------------------------------------------------


def rule_checks(o: Mapping[str, Any], h: Mapping[str, Any], g: Mapping[str, Any]) -> dict[str, Any]:
    """The rule's comparisons on already computed readings → ``{"O", "H", "G", "verdict"}`` (each part with its sub-checks).

    `o`: ``band_low`` (lower bound of v0.3 − base balanced band), ``severe`` / ``severe_base`` (points), ``direction`` (point);
    `h`: ``band_low`` (lower bound of v0.3 − v0.1-mix), ``severe`` / ``severe_mix``; `g`: ``diff`` / ``low`` (v0.3 − base accuracy)."""
    def none_safe(check: bool | None) -> bool:
        return bool(check) if check is not None else False

    O = {"band_low_gt_0": none_safe(o.get("band_low") is not None and o["band_low"] > O_BAND_LOW),
         "severe_le_base": none_safe(o.get("severe") is not None and o.get("severe_base") is not None and o["severe"] <= o["severe_base"]),
         "direction_ge_0.70": none_safe(o.get("direction") is not None and o["direction"] >= O_DIRECTION)}
    H = {"band_low_gt_-0.05": none_safe(h.get("band_low") is not None and h["band_low"] > H_BAND_LOW),
         "severe_le_mix_plus_0.01": none_safe(h.get("severe") is not None and h.get("severe_mix") is not None and h["severe"] <= h["severe_mix"] + H_SEVERE_SLACK)}
    G = {"point_ge_-0.02": none_safe(g.get("diff") is not None and g["diff"] >= G_POINT),
         "low_gt_-0.045": none_safe(g.get("low") is not None and g["low"] > G_LOW)}
    out = {"O": {"checks": O, "pass": all(O.values())}, "H": {"checks": H, "pass": all(H.values())}, "G": {"checks": G, "pass": all(G.values())}}
    out["verdict"] = "PASSES" if out["O"]["pass"] and out["H"]["pass"] and out["G"]["pass"] else "DOES NOT PASS"
    return out


def outside_reading(t8b: Sequence[dict[str, Any]], v03: Sequence[dict[str, Any]], base: Sequence[dict[str, Any]], *, n_boot: int = 2000,
                    seed: int = 0) -> dict[str, Any]:
    """(O)'s readings on T8b: balanced band Δ (paired, stratified by label band), severe-error points (paired items), direction point."""
    from robo_jev.bench import system1

    a = system1.set_items("T8b", t8b, v03, reading="single_order")
    b = system1.set_items("T8b", t8b, base, reading="single_order")
    band = system1.paired_difference(a.get("band_balanced") or {}, b.get("band_balanced") or {}, n_boot=n_boot, seed=seed, stratified=True)
    severe = system1.paired_difference(a.get("severe_error") or {}, b.get("severe_error") or {}, n_boot=n_boot, seed=seed)
    direction = system1.summarize(a.get("direction") or {}, n_boot=n_boot, seed=seed)
    return {"band_balanced": band, "severe_error": severe, "direction": direction,
            "band_low": (band or {}).get("low"), "severe": (severe or {}).get("a"), "severe_base": (severe or {}).get("b"),
            "direction_value": direction.get("value")}


def outside_by_group(t8b: Sequence[dict[str, Any]], v03: Sequence[dict[str, Any]], base: Sequence[dict[str, Any]], *, n_boot: int = 2000,
                     seed: int = 0) -> dict[str, Any]:
    """T8b System-1 metrics per family group (:data:`T8B_GROUPS`) — reported next to the gate, never gating: v0.3 and the base, the paired
    Δ with its interval for balanced band accuracy, severe errors, direction, comparison and outcome (all families)."""
    from robo_jev.bench import system1

    out = {}
    for group, scenes in T8B_GROUPS.items():
        records = [r for r in t8b if (r.get("meta") or {}).get("scene") in scenes]
        a = system1.set_items("T8b", records, v03, reading="single_order")
        b = system1.set_items("T8b", records, base, reading="single_order")
        entry = {"records": len(records)}
        for metric in ("band_balanced", "severe_error", "severe_miss", "direction", "comparison", "outcome:all_families"):
            if metric in a and metric in b:
                entry[metric] = system1.paired_difference(a[metric], b[metric], n_boot=n_boot, seed=seed, stratified=metric in system1.STRATIFIED)
        out[group] = entry
    return out


def home_reading(sets: Mapping[str, Sequence[dict[str, Any]]], v03: Mapping[str, Sequence[dict[str, Any]]], mix: Mapping[str, Sequence[dict[str, Any]]],
                 *, n_boot: int = 2000, seed: int = 0) -> dict[str, Any]:
    """(H)'s readings on T2–T4 pooled: balanced band Δ vs v0.1-mix (paired, stratified) and severe-error points (paired items)."""
    from robo_jev.bench import system1

    a = system1.group_items({s: system1.set_items(s, sets[s], v03[s]) for s in HOME}, HOME)
    b = system1.group_items({s: system1.set_items(s, sets[s], mix[s]) for s in HOME}, HOME)
    band = system1.paired_difference(a.get("band_balanced") or {}, b.get("band_balanced") or {}, n_boot=n_boot, seed=seed, stratified=True)
    severe = system1.paired_difference(a.get("severe_error") or {}, b.get("severe_error") or {}, n_boot=n_boot, seed=seed)
    per_set = {}
    for s in HOME:
        sa, sb = system1.set_items(s, sets[s], v03[s]), system1.set_items(s, sets[s], mix[s])
        per_set[s] = {"band_balanced": system1.paired_difference(sa["band_balanced"], sb["band_balanced"], n_boot=0, stratified=True),
                      "severe_error": system1.paired_difference(sa["severe_error"], sb["severe_error"], n_boot=0)}
    return {"band_balanced": band, "severe_error": severe, "per_set_points": per_set,
            "band_low": (band or {}).get("low"), "severe": (severe or {}).get("a"), "severe_mix": (severe or {}).get("b")}


def general_reading(g2: Sequence[dict[str, Any]], v03: Sequence[dict[str, Any]], base: Sequence[dict[str, Any]], *, n_boot: int = 2000,
                    seed: int = 0) -> dict[str, Any]:
    """(G)'s reading on G2: accuracy v0.3 − base, paired, cluster = record (the v0.2 verdict's reading)."""
    from robo_jev.decisions.single_probe import compare_probe

    comparison = compare_probe(list(g2), list(v03), list(base), dimensions=(), cluster="record", n_boot=n_boot, seed=seed)
    single = comparison["readings"]["single_order"]
    accuracy = single["table"]["_all"]["accuracy"]
    return {"accuracy": accuracy, "questions": single["questions"], "only_v03": single["only_a"], "only_base": single["only_b"],
            "diff": accuracy.get("diff"), "low": accuracy.get("low")}


def secondary(sets: Mapping[str, Sequence[dict[str, Any]]], models: Mapping[str, Mapping[str, Sequence[dict[str, Any]]]], *, n_boot: int = 2000,
              seed: int = 0) -> dict[str, Any]:
    """Reported, never gating: T8b comparison / outcome / plain band accuracy / severe misses vs the base and v0.1-mix, T8b and T2–T4
    Brier skill (reference), T8a accuracy vs the base."""
    from robo_jev.bench import system1
    from robo_jev.outside.score import pooled_skill, skill_rows

    out: dict[str, Any] = {}
    if "T8b" in sets and "T8b" in models.get("v03", {}):
        v = system1.set_items("T8b", sets["T8b"], models["v03"]["T8b"])
        for rival in ("base", "mix"):
            preds = models.get(rival, {}).get("T8b")
            if not preds:
                continue
            r = system1.set_items("T8b", sets["T8b"], preds)
            out[f"T8b_vs_{rival}"] = {m: system1.paired_difference(v[m], r[m], n_boot=n_boot, seed=seed, stratified=m in system1.STRATIFIED)
                                      for m in ("band_accuracy", "band_balanced", "severe_error", "severe_miss", "direction", "comparison", "outcome",
                                                "outcome:all_families") if m in v and m in r}
        out["T8b_skill"] = {name: (pooled_skill(skill_rows(list(sets["T8b"]), list(m["T8b"]), reading="single_order")) or {}).get("skill")
                            for name, m in models.items() if m.get("T8b")}
    if all(s in sets for s in HOME):
        out["T2-T4_skill"] = {name: (pooled_skill([row for s in HOME for row in skill_rows(list(sets[s]), list(m[s]), reading="single_order")]) or {}).get("skill")
                              for name, m in models.items() if all(m.get(s) for s in HOME)}
    if "T8a" in sets and models.get("v03", {}).get("T8a") and models.get("base", {}).get("T8a"):
        v = system1.set_items("T8a", sets["T8a"], models["v03"]["T8a"])
        r = system1.set_items("T8a", sets["T8a"], models["base"]["T8a"])
        out["T8a_vs_base"] = {"choice_accuracy": system1.paired_difference(v["choice_accuracy"], r["choice_accuracy"], n_boot=n_boot, seed=seed)}
    return out


# --------------------------------------------------------------------------
# Files
# --------------------------------------------------------------------------


def _summary_of(predictions: Path) -> Path:
    return Path(str(predictions).replace(".predictions.jsonl", ".json"))


def check_source(predictions: Path, set_path: Path, records: Sequence[dict[str, Any]], preds: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Was this prediction file scored on this set file? (summary's input sha256), scoring conditions, coverage of the set's ids."""
    from robo_jev.decisions.records import file_sha256

    summary_path = _summary_of(predictions)
    recorded = None
    if summary_path.is_file():
        recorded = (json.loads(summary_path.read_text(encoding="utf-8")).get("config") or {}).get("input_sha256")
    ids = {str(r["id"]) for r in records}
    predicted = {str(p["id"]) for p in preds}
    conditions = {key: sorted({json.dumps(p.get(key)) for p in preds}) for key in SCORING_KEYS}
    return {"predictions": str(predictions), "summary": str(summary_path) if summary_path.is_file() else None, "set": str(set_path),
            "set_sha256": file_sha256(set_path), "recorded_input_sha256": recorded,
            "same_set": (recorded == file_sha256(set_path)) if recorded else None,
            "coverage": len(ids & predicted) / len(ids) if ids else None, "extra_ids": len(predicted - ids),
            "conditions": {k: (json.loads(v[0]) if len(v) == 1 else [json.loads(x) for x in v]) for k, v in conditions.items()}}


def _read(path: Path) -> list[dict[str, Any]]:
    from robo_jev.decisions.records import read_records

    return read_records(path)


def _fmt(value: Any, digits: int = 3, signed: bool = False) -> str:
    if value is None:
        return "—"
    return f"{value:+.{digits}f}" if signed else f"{value:.{digits}f}"


def render(result: Mapping[str, Any]) -> str:
    o, h, g, r = result["readings"]["O"], result["readings"]["H"], result["readings"]["G"], result["rule"]
    band_o, band_h, acc = o["band_balanced"] or {}, h["band_balanced"] or {}, g["accuracy"] or {}
    lines = [f"# v0.3 pre-declared rule — {r['verdict']}", "",
             f"Run: `{result['inputs']['v03_dir']}` · {result['n_boot']} paired bootstrap draws (seed {result['seed']}, cluster = record) · single-order reading.", "",
             "| part | reading | value | check | pass |", "|---|---|---|---|---|",
             f"| O | T8b balanced band, v0.3 − base | {_fmt(band_o.get('value'), signed=True)} [{_fmt(band_o.get('low'), signed=True)}, {_fmt(band_o.get('high'), signed=True)}] "
             f"(v0.3 {_fmt(band_o.get('a'))}, base {_fmt(band_o.get('b'))}) | lower bound > 0 | {r['O']['checks']['band_low_gt_0']} |",
             f"| O | T8b severe-error rate | v0.3 {_fmt(o['severe'])} vs base {_fmt(o['severe_base'])} | ≤ base | {r['O']['checks']['severe_le_base']} |",
             f"| O | T8b direction | {_fmt(o['direction_value'])} [{_fmt(o['direction'].get('low'))}, {_fmt(o['direction'].get('high'))}] (n {o['direction'].get('n')}) | ≥ 0.70 | {r['O']['checks']['direction_ge_0.70']} |",
             f"| H | T2–T4 balanced band, v0.3 − v0.1-mix | {_fmt(band_h.get('value'), signed=True)} [{_fmt(band_h.get('low'), signed=True)}, {_fmt(band_h.get('high'), signed=True)}] "
             f"(v0.3 {_fmt(band_h.get('a'))}, v0.1-mix {_fmt(band_h.get('b'))}) | lower bound > −0.05 | {r['H']['checks']['band_low_gt_-0.05']} |",
             f"| H | T2–T4 severe-error rate | v0.3 {_fmt(h['severe'])} vs v0.1-mix {_fmt(h['severe_mix'])} | ≤ v0.1-mix + 0.01 | {r['H']['checks']['severe_le_mix_plus_0.01']} |",
             f"| G | G2 accuracy, v0.3 − base (pod) | {_fmt(acc.get('diff'), signed=True)} [{_fmt(acc.get('low'), signed=True)}, {_fmt(acc.get('high'), signed=True)}] "
             f"(v0.3 {_fmt(acc.get('a'))}, base {_fmt(acc.get('b'))}) | ≥ −0.02 and lower bound > −0.045 | "
             f"{r['G']['checks']['point_ge_-0.02'] and r['G']['checks']['low_gt_-0.045']} |", "",
             f"O {'PASS' if r['O']['pass'] else 'FAIL'} · H {'PASS' if r['H']['pass'] else 'FAIL'} · G {'PASS' if r['G']['pass'] else 'FAIL'} → **{r['verdict']}**", ""]
    groups = o.get("by_family_group") or {}
    if groups:
        lines += ["T8b by family group (reported next to the gate, not gating — tilt and tower share laws with deck and stack; push is the "
                  "only unseen law). v0.3 vs base, Δ [95 %]:", "",
                  "| group | records | balanced band | severe errors ↓ | direction | comparison | outcome (all families) |", "|---|---|---|---|---|---|---|"]
        for group, entry in groups.items():
            cells = []
            for metric in ("band_balanced", "severe_error", "direction", "comparison", "outcome:all_families"):
                d = entry.get(metric)
                cells.append("—" if not d else f"{_fmt(d.get('a'))} vs {_fmt(d.get('b'))}, {_fmt(d.get('value'), signed=True)} "
                                                f"[{_fmt(d.get('low'), signed=True)}, {_fmt(d.get('high'), signed=True)}]")
            lines.append(f"| {group} | {entry['records']} | " + " | ".join(cells) + " |")
        lines.append("")
    lines += ["## Notes", ""] + [f"- {note}" for note in result["notes"]] + [""]
    lines += ["## Sources", "", "| reading | predictions | same set file | coverage | conditions |", "|---|---|---|---|---|"]
    for name, source in result["sources"].items():
        lines.append(f"| {name} | `{source['predictions']}` | {source['same_set']} | {_fmt(source['coverage'])} | {json.dumps(source['conditions'])} |")
    if result.get("secondary"):
        lines += ["", "## Secondary (reported, not gating)", "", "```json", json.dumps(result["secondary"], indent=1, ensure_ascii=False)[:20000], "```"]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--v03-dir", type=Path, required=True, help="the fetched reports directory of the v0.3 job")
    parser.add_argument("--tag", default="after-v03-2b", help="prediction file prefix in --v03-dir (default after-v03-2b)")
    parser.add_argument("--mix-dir", type=Path, default=MIX_DIR, help="v0.1-mix's fetched reports (after-v01mix-2b-<set>)")
    parser.add_argument("--base-t8b", type=Path, default=BASE_T8 / "T8b-base-2b.predictions.jsonl", help="untrained 2B on T8b (Spark, T8 task)")
    parser.add_argument("--base-t8a", type=Path, default=BASE_T8 / "T8a-base-2b.predictions.jsonl")
    parser.add_argument("--mix-t8b", type=Path, default=BASE_T8 / "T8b-v01mix-2b.predictions.jsonl", help="v0.1-mix on T8b (secondary)")
    parser.add_argument("--base-g2", type=Path, default=BASE_G2_POD, help="untrained 2B on G2 (v0.1 main pod, H100)")
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--with-diagnose-compare", action="store_true", help="also T1–T7 Brier / regret vs v0.1-mix (diagnose_decisions --compare)")
    parser.add_argument("--allow-mismatch", action="store_true", help="read even if a prediction file was scored on another set file or conditions differ")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    v03 = {s: args.v03_dir / f"{args.tag}-{s}.predictions.jsonl" for s in ("T1", "T2", "T3", "T4", "T5", "T6", "T7", "G2", "T8a", "T8b")}
    needed = {"O:v03 T8b": (v03["T8b"], "T8b"), "O:base T8b (Spark)": (args.base_t8b, "T8b"), "G:v03 G2": (v03["G2"], "G2"),
              "G:base G2 (pod)": (args.base_g2, "G2"),
              **{f"H:v03 {s}": (v03[s], s) for s in HOME}, **{f"H:v0.1-mix {s}": (args.mix_dir / f"after-v01mix-2b-{s}.predictions.jsonl", s) for s in HOME}}  # fmt: skip
    missing = [f"{name}: {path}" for name, (path, _) in needed.items() if not path.is_file()]
    if missing:
        raise SystemExit("missing prediction files for the rule:\n  " + "\n  ".join(missing))
    sets = {s: _read(SET_PATHS[s]) for s in ("T2", "T3", "T4", "T8b", "G2")}
    for extra in ("T8a",):
        if SET_PATHS[extra].is_file():
            sets[extra] = _read(SET_PATHS[extra])
    loaded = {name: _read(path) for name, (path, _) in needed.items()}
    sources = {name: check_source(path, SET_PATHS[s], sets[s], loaded[name]) for name, (path, s) in needed.items()}
    problems = [f"{name}: scored on another file (recorded {src['recorded_input_sha256']}, set {src['set_sha256']})" for name, src in sources.items()
                if src["same_set"] is False]
    for left, right in (("O:v03 T8b", "O:base T8b (Spark)"), ("G:v03 G2", "G:base G2 (pod)"), *((f"H:v03 {s}", f"H:v0.1-mix {s}") for s in HOME)):
        if sources[left]["conditions"] != sources[right]["conditions"]:
            problems.append(f"{left} vs {right}: scoring conditions differ {sources[left]['conditions']} / {sources[right]['conditions']}")
    if problems and not args.allow_mismatch:
        raise SystemExit("refusing (use --allow-mismatch to read anyway):\n  " + "\n  ".join(problems))
    o = outside_reading(sets["T8b"], loaded["O:v03 T8b"], loaded["O:base T8b (Spark)"], n_boot=args.n_boot, seed=args.seed)
    h = home_reading(sets, {s: loaded[f"H:v03 {s}"] for s in HOME}, {s: loaded[f"H:v0.1-mix {s}"] for s in HOME}, n_boot=args.n_boot, seed=args.seed)
    g = general_reading(sets["G2"], loaded["G:v03 G2"], loaded["G:base G2 (pod)"], n_boot=args.n_boot, seed=args.seed)
    o["by_family_group"] = outside_by_group(sets["T8b"], loaded["O:v03 T8b"], loaded["O:base T8b (Spark)"], n_boot=args.n_boot, seed=args.seed)
    rule = rule_checks({"band_low": o["band_low"], "severe": o["severe"], "severe_base": o["severe_base"], "direction": o["direction_value"]},
                       {"band_low": h["band_low"], "severe": h["severe"], "severe_mix": h["severe_mix"]}, {"diff": g["diff"], "low": g["low"]})
    models: dict[str, dict[str, list[dict[str, Any]]]] = {
        "v03": {"T8b": loaded["O:v03 T8b"], **{s: loaded[f"H:v03 {s}"] for s in HOME}},
        "base": {"T8b": loaded["O:base T8b (Spark)"]}, "mix": {s: loaded[f"H:v0.1-mix {s}"] for s in HOME},
    }
    if v03["T8a"].is_file() and args.base_t8a.is_file() and "T8a" in sets:
        models["v03"]["T8a"], models["base"]["T8a"] = _read(v03["T8a"]), _read(args.base_t8a)
    if args.mix_t8b.is_file():
        models["mix"]["T8b"] = _read(args.mix_t8b)
    result: dict[str, Any] = {
        "kind": "v03-verdict", "script_version": SCRIPT_VERSION, "created": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "rule": rule, "rule_text": RULE, "n_boot": args.n_boot, "seed": args.seed, "reading": "single_order",
        "inputs": {"v03_dir": str(args.v03_dir), "tag": args.tag, "mix_dir": str(args.mix_dir), "base_t8b": str(args.base_t8b), "base_g2": str(args.base_g2)},
        "readings": {"O": o, "H": h, "G": g}, "sources": sources, "problems": problems,
        "notes": ["The untrained base on T8b was scored on the Spark (GB10) by the T8 task, the v0.3 model on the H100 pod: same prompt, "
                  "format, orders and shuffle seed, different hardware (bf16 kernels). The v0.1 pods matched Spark on G1/G2 within 0.003.",
                  "G2's base is the v0.1 main pod's untrained 2B (same H100 type as the v0.3 job).",
                  "Pairs and items are those both models answered; coverage per file is in `sources`."],
        "secondary": secondary({**sets}, models, n_boot=args.n_boot, seed=args.seed),
    }  # fmt: skip
    if args.with_diagnose_compare:
        result["secondary"]["T1-T7_vs_v0.1-mix"] = _diagnose_compare(v03, args)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    args.out.with_suffix(".md").write_text(render(result), encoding="utf-8")
    print(render(result).split("## Notes")[0])
    return 0


def _diagnose_compare(v03: Mapping[str, Path], args: argparse.Namespace) -> dict[str, Any]:
    """T1–T7: noul / score Brier and default-criterion regret, v0.3 − v0.1-mix (scripts/diagnose_decisions.py --compare, single order)."""
    out: dict[str, Any] = {}
    keys = ("noul.brier", "score.brier", "choice/action.regret_mean[default]")
    for s in ("T1", "T2", "T3", "T4", "T5", "T6", "T7"):
        mine, theirs = v03[s], args.mix_dir / f"after-v01mix-2b-{s}.predictions.jsonl"
        if not (mine.is_file() and theirs.is_file()):
            out[s] = "missing"
            continue
        target = args.out.with_name(f"compare-{s}.json")
        subprocess.run([sys.executable, str(REPO / "scripts" / "diagnose_decisions.py"), "--compare", str(mine), str(theirs), "--labels", "v0.3", "v0.1-mix",
                        "--input", str(SET_PATHS[s]), "--out", str(target)], check=True, cwd=REPO, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)  # fmt: skip
        diffs = json.loads(target.read_text(encoding="utf-8"))["comparison"]["readings"]["single_order"]["differences"]
        out[s] = {k: diffs[k] for k in keys if k in diffs}
    return out


if __name__ == "__main__":
    raise SystemExit(main())
