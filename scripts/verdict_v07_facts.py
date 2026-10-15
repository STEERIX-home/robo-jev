#!/usr/bin/env python
"""Apply the PRE-DECLARED v0.7 SECONDARY READING — FACTS IN INSTRUCTIONS, gate M2g-F (progress.md, written 2026-10-15 ~19:57 KST, before the
gate was built and before any v0.7 gate prediction existed) mechanically. The reading enters none of the v0.7 rule's parts (S-a as amended,
S-b, the guards O / N / R / G / F, the release line); ``scripts/verdict_v07.py`` is not changed.

    python scripts/verdict_v07_facts.py --s0-dir artifacts/cloud/runs/<s0 run>-fetched/reports --s1-dir artifacts/cloud/runs/<s1 run>-fetched/reports
        # → artifacts/reports/v07/verdict-facts.{json,md}   (after artifacts/scratch/v07f/score_facts.sh has scored M2g-F)

**Inputs**: the sealed M2g-short and M2g-F files (``artifacts/v07/gates/m2g/M2g-short.jsonl``, ``artifacts/v07/gates/m2gf/M2g-facts.jsonl`` —
the same ids, questions and labels) and the Spark predictions of the five models on both (``M2g-short-<tag>`` from the v0.7 chain,
``M2g-facts-<tag>`` from the M2g-F chain; tags v07s0-2b, v07s1-2b, v06s0-2b, v06s1-2b, base-2b), each checked for its scoring conditions
(M2g's), scored file and checkpoint; ``artifacts/v07/gates/overlap-v07.json``'s M2g exclusions (task v0.7b's, pre-declared) leave those moments
out of both forms.

**Reading** (call view, seed means, paired cluster bootstrap 2,000 draws seed 0, as S-a — the call items, the seed means, the bootstrap and the
exclusions are ``scripts/verdict_v07.py``'s, imported; v0.5's truth classes and 1/L call weights):

* gap_F = DC(F) − DC(short) per arm, short = M2g-short on the same questions; ΔCA_F = CA(F) − CA(short) per arm (CA = mean of P(safe |
  clear-safe) and P(risky | clear-risky)). gap_F, Δgap_F = gap_F(v0.7) − gap_F(v0.6) and ΔCA_F are read with one bootstrap over the four
  sources (v0.7 F, v0.7 short, v0.6 F, v0.6 short) on the questions all four answered.
* F is ACTIVE iff v0.6's gap_F ≥ +0.05 (point).
* READS FACTS (active) iff Δgap_F ≤ −0.05 with upper bound < 0 AND v0.7's gap_F ≤ +0.03 (point) AND v0.7's ΔCA_F ≥ −0.05 (point).
* NO HARM (inactive) iff v0.7's gap_F ≤ +0.03 (point) AND v0.7's ΔCA_F ≥ −0.05 (point).
* Otherwise NOT SHOWN; if v0.7's ΔCA_F < −0.05 the facts are taken as not used (calls carried by the bare status), so a small gap_F is then not
  read as robustness.
* Also reported, not interpreted: LU / LC / GAP F − short per arm, the same for the untrained base, per template, seed spread (and by source
  set).
* Pre-declared consequence: READS FACTS or NO HARM → the facts transform stays in the recipe; NOT SHOWN → the next version's request-shape
  work gives the facts form its own primary gate before more data.
"""

from __future__ import annotations

import argparse
import datetime as dt
import functools
import importlib.util
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

SCRIPT_VERSION = "v07f-verdict-facts-1"
READING = {
    "written": "2026-10-15 ~19:57 KST, BEFORE the gate is built and BEFORE any v0.7 gate prediction exists (progress.md: PRE-DECLARED v0.7 "
               "SECONDARY READING - FACTS IN INSTRUCTIONS, gate M2g-F)",
    "scope": "the v0.7 rule - S-a as amended, S-b, guards O / N / R / G / F, the release line - is unchanged, and this reading enters none of them",
    "gate": "the same 460 M2g moments, the same 1,098 call-view questions, labels and ids as M2g-short (so the seeded option order is M2g-short's), "
            "with the state replaced by a minimal status line that carries no fact and every original state sentence moved verbatim, in order, "
            "into each question's instructions, in >= 3 templates; scored for v0.7-s0, v0.7-s1, v0.6-s0, v0.6-s1 and the untrained base with "
            "M2g's settings (diagnose_decisions.py --orders seeded) after the main chain",
    "reading": "call view, seed means, paired cluster bootstrap 2000 draws seed 0, as S-a: gap_F = DC(F) - DC(short) per arm, short = M2g-short "
               "on the same questions; dCA_F = CA(F) - CA(short) per arm (CA = mean of P(safe | clear-safe) and P(risky | clear-risky))",
    "interpretation": "F is ACTIVE iff v0.6's gap_F >= +0.05 (point). READS FACTS (active) iff v0.7 - v0.6 dgap_F <= -0.05 with upper bound < 0 "
                      "AND v0.7's gap_F <= +0.03 (point) AND v0.7's dCA_F >= -0.05 (point). NO HARM (inactive) iff v0.7's gap_F <= +0.03 "
                      "(point) AND v0.7's dCA_F >= -0.05 (point). Otherwise NOT SHOWN; if v0.7's dCA_F < -0.05 the facts are taken as not "
                      "used (calls carried by the bare status), so a small gap_F is then not read as robustness.",
    "also_reported": "not interpreted: LU / LC / GAP F - short per arm, the same for the untrained base, per template, seed spread",
    "consequence": "READS FACTS or NO HARM -> the facts transform stays in the recipe; NOT SHOWN -> the next version's request-shape work gives the "
                   "facts form its own primary gate before more data",
}
#: The reading's bounds: activity (v0.6's gap_F ≥), Δgap_F (≤) and its upper bound (<), v0.7's gap_F (≤), v0.7's ΔCA_F (≥).
F_ACTIVE, F_DGAP, F_HIGH, F_GAP, F_DCA = 0.05, -0.05, 0.0, 0.03, -0.05
READS_FACTS, NO_HARM, NOT_SHOWN = "READS FACTS", "NO HARM", "NOT SHOWN"
CONSEQUENCE = {READS_FACTS: "the facts transform stays in the recipe", NO_HARM: "the facts transform stays in the recipe",
               NOT_SHOWN: "the next version's request-shape work gives the facts form its own primary gate before more data"}
#: The two forms read (``short`` = M2g-short, ``facts`` = M2g-F) and the call metrics reported.
FORMS = ("short", "facts")
METRICS = ("DC", "CA", "LU", "LC", "GAP")

ARTIFACTS = REPO / "artifacts"
FACTS_DIR = ARTIFACTS / "v07" / "gates" / "m2gf"
FACTS_FILE = FACTS_DIR / "M2g-facts.jsonl"
FACTS_MANIFEST = FACTS_DIR / "manifest.json"
OUT = ARTIFACTS / "reports" / "v07" / "verdict-facts.json"


@functools.lru_cache(maxsize=None)
def V7():
    """scripts/verdict_v07.py — the call items (:func:`m2g_calls`), the seed means (``V6().arm_calls``), the paired cluster bootstrap
    (:func:`dgap_reading`, :func:`gap_reading`, the v0.5 call-view machinery ``V5()``), the gate-file checks and the exclusions' file."""
    spec = importlib.util.spec_from_file_location("v07f_verdict_v07", REPO / "scripts" / "verdict_v07.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _log(message: str) -> None:
    print(f"[verdict-v07-facts {dt.datetime.now().strftime('%H:%M:%S')}] {message}", file=sys.stderr, flush=True)


def _git_sha() -> str | None:
    try:
        return subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True).stdout.strip() or None
    except (OSError, subprocess.CalledProcessError):
        return None


# --------------------------------------------------------------------------
# The reading (pure — readings in, outcome out)
# --------------------------------------------------------------------------


def _le(value: float | None, bound: float) -> bool:
    return value is not None and value <= bound


def _lt(value: float | None, bound: float) -> bool:
    return value is not None and value < bound


def _ge(value: float | None, bound: float) -> bool:
    return value is not None and value >= bound


def facts_outcome(entry: Mapping[str, Any] | None) -> dict[str, Any]:
    """The pre-declared reading on its four readings (``gap_v06``, ``gap_v07``, ``dgap`` with its interval, ``dca_v07``): ACTIVE iff v0.6's
    gap_F ≥ +0.05; READS FACTS (active) iff Δgap_F ≤ −0.05, its upper bound < 0, v0.7's gap_F ≤ +0.03 and v0.7's ΔCA_F ≥ −0.05; NO HARM
    (inactive) iff v0.7's gap_F ≤ +0.03 and v0.7's ΔCA_F ≥ −0.05; otherwise NOT SHOWN. ``facts_not_used`` iff v0.7's ΔCA_F < −0.05 (then a small
    gap_F is not read as robustness). Without v0.6's gap_F the status is ``missing`` and the outcome NOT SHOWN."""
    entry = entry or {}
    gap6, gap7 = (entry.get("gap_v06") or {}).get("value"), (entry.get("gap_v07") or {}).get("value")
    dgap, dca = entry.get("dgap") or {}, (entry.get("dca_v07") or {}).get("value")
    not_used = dca is not None and dca < F_DCA
    if gap6 is None:
        return {"status": "missing", "checks": {}, "outcome": NOT_SHOWN, "facts_not_used": not_used, "consequence": CONSEQUENCE[NOT_SHOWN]}
    active = gap6 >= F_ACTIVE
    checks = {"gap_v07_le_+0.03": _le(gap7, F_GAP), "dca_v07_ge_-0.05": _ge(dca, F_DCA)}
    if active:
        checks = {"dgap_le_-0.05": _le(dgap.get("value"), F_DGAP), "dgap_high_lt_0": _lt(dgap.get("high"), F_HIGH), **checks}
    outcome = (READS_FACTS if active else NO_HARM) if all(checks.values()) else NOT_SHOWN
    note = None
    if not_used:
        note = "v0.7's dCA_F < -0.05: the facts are taken as not used (calls carried by the bare status), so a small gap_F is not read as robustness"
    return {"status": "active" if active else "inactive", "gap_v06": gap6, "checks": checks, "outcome": outcome, "facts_not_used": not_used,
            "note": note, "consequence": CONSEQUENCE[outcome]}


# --------------------------------------------------------------------------
# Readings from predictions
# --------------------------------------------------------------------------


def four_sources(a_f: Mapping, a_s: Mapping, b_f: Mapping, b_s: Mapping, *, n_boot: int, seed: int, keys: set | None = None,
                 metrics: Sequence[str] = METRICS) -> dict[str, Any] | None:
    """Per call metric: arm a's and arm b's F − short and their difference, on the questions all four sources answered (∩ `keys`), with one
    paired cluster bootstrap over the four (``scripts/verdict_v07.py``'s :func:`dc_sources`, extended from DC to every call metric — the same
    clusters, class sums and draws)."""
    v5 = V7().V5()
    common = set(a_f) & set(a_s) & set(b_f) & set(b_s)
    if keys is not None:
        common &= set(keys)
    common = sorted(common)
    if not common:
        return None
    sums, counts, clusters = v5._cluster_arrays(common, [a_f, a_s, b_f, b_s])
    points = [v5._metrics(s.sum(0), counts.sum(0)) for s in sums]
    w = v5._draw_weights(clusters, n_boot, seed)
    n = w @ counts
    boots = [v5._metrics(np.einsum("bc,ctk->btk", w, s), n) for s in sums]
    out: dict[str, Any] = {"n": len(common), "clusters": clusters}
    reading = V7()._reading
    for metric in metrics:
        p = [float(x[metric]) for x in points]
        b = [np.asarray(x[metric], dtype=float) for x in boots]
        out[metric] = {"a": reading(p[0] - p[1], b[0] - b[1], facts=p[0], short=p[1]), "b": reading(p[2] - p[3], b[2] - b[3], facts=p[2], short=p[3]),
                       "a_minus_b": reading((p[0] - p[1]) - (p[2] - p[3]), (b[0] - b[1]) - (b[2] - b[3]))}
    return out


def facts_entry(calls: Mapping[tuple[str, str], Mapping], *, n_boot: int, seed: int, keys: set | None = None) -> dict[str, Any] | None:
    """The reading's four inputs on (a subset `keys` of) the questions: gap_F per arm and Δgap_F from ``verdict_v07.dgap_reading`` (imported,
    one bootstrap over the four sources) and v0.7's / v0.6's ΔCA_F from :func:`four_sources` (the same questions and draws)."""
    if not all((arm, form) in calls for arm in ("v07", "v06") for form in FORMS):
        return None
    sources = (calls[("v07", "facts")], calls[("v07", "short")], calls[("v06", "facts")], calls[("v06", "short")])
    d = V7().dgap_reading(*sources, n_boot=n_boot, seed=seed, keys=keys)
    if d is None:
        return None
    rest = four_sources(*sources, n_boot=n_boot, seed=seed, keys=keys)
    return {**d, "dca_v07": rest["CA"]["a"], "dca_v06": rest["CA"]["b"], "ddca": rest["CA"]["a_minus_b"],
            "also": {m: rest[m] for m in ("LU", "LC", "GAP")}, "n": d["dgap"]["n"]}


def template_of(record: Mapping[str, Any]) -> str | None:
    return ((record.get("meta") or {}).get("gate") or {}).get("template")


def facts_readings(short: Sequence[Mapping[str, Any]], facts: Sequence[Mapping[str, Any]], preds: Mapping[str, Mapping[str, Sequence[Mapping[str, Any]]]],
                   *, n_boot: int = 2000, seed: int = 0, exclude: set[str] | None = None) -> dict[str, Any]:
    """Every reading. `preds[model][form]` = predictions for ``short`` and ``facts`` of the models present (the reading needs v07s0, v07s1,
    v06s0, v06s1; the base is reported). Records in `exclude` are left out of both forms."""
    v7 = V7()
    exclude = set(exclude or ())
    records = {"short": [r for r in short if str(r["id"]) not in exclude], "facts": [r for r in facts if str(r["id"]) not in exclude]}
    if [str(r["id"]) for r in records["short"]] != [str(r["id"]) for r in records["facts"]]:
        raise ValueError("M2g-short and M2g-F must hold the same records in the same order")
    calls: dict[tuple[str, str], dict] = {}
    for model, by_form in preds.items():
        for form in FORMS:
            if by_form.get(form) is not None:
                calls[(model, form)] = v7.m2g_calls(records[form], by_form[form])
    for arm, (first, second) in v7.ARMS.items():
        for form in FORMS:
            if (first, form) in calls and (second, form) in calls:
                calls[(arm, form)] = v7.V6().arm_calls(calls[(first, form)], calls[(second, form)])
    models = [m for m in (*v7.GATE_MODELS, *v7.ARMS) if all((m, form) in calls for form in FORMS)]
    out: dict[str, Any] = {"records": len(records["facts"]), "excluded_ids": sorted(exclude), "models": models}
    entry = facts_entry(calls, n_boot=n_boot, seed=seed)
    out["rule"] = entry
    out["outcome"] = facts_outcome(entry)
    # levels (every call metric, per model and arm, each form) and F − short per model and arm (paired, the questions both forms answered)
    out["levels"] = {m: {form: v7.V5().call_metrics(calls[(m, form)], n_boot=n_boot, seed=seed) for form in FORMS} for m in models}
    out["facts_minus_short"] = {m: v7.V5().call_difference(calls[(m, "facts")], calls[(m, "short")], n_boot=n_boot, seed=seed) for m in models}
    # per template and by source set (the arms' rule readings on the subset; reported, not interpreted)
    templates: dict[str, set] = defaultdict(set)
    sets: dict[str, set] = defaultdict(set)
    for r in records["facts"]:
        for q in r["request"]["questions"]:
            key = (v7.source_of(r), str(r["id"]), q)
            templates[str(template_of(r))].add(key)
            sets[v7.source_of(r)].add(key)
    out["by_template"] = {t: _subset(calls, models, keys, n_boot=n_boot, seed=seed) for t, keys in sorted(templates.items())}
    out["by_set"] = {s: _subset(calls, models, keys, n_boot=n_boot, seed=seed) for s, keys in sorted(sets.items())}
    # seed spread: |s0 − s1| of each arm's gap_F and ΔCA_F (per-model F − short)
    spread = {}
    for arm, (first, second) in v7.ARMS.items():
        a, b = out["facts_minus_short"].get(first), out["facts_minus_short"].get(second)
        if a and b:
            spread[arm] = {f"{metric}_F_minus_short": abs(a[metric]["value"] - b[metric]["value"])
                           for metric in METRICS if a.get(metric, {}).get("value") is not None and b.get(metric, {}).get("value") is not None}
    out["seed_spread"] = spread
    return out


def _subset(calls: Mapping, models: Sequence[str], keys: set, *, n_boot: int, seed: int) -> dict[str, Any]:
    v5 = V7().V5()
    return {"rule": facts_entry(calls, n_boot=n_boot, seed=seed, keys=keys),
            "facts_minus_short": {m: v5.call_difference({k: v for k, v in calls[(m, "facts")].items() if k in keys},
                                                         {k: v for k, v in calls[(m, "short")].items() if k in keys}, n_boot=n_boot, seed=seed)
                                  for m in models}}


# --------------------------------------------------------------------------
# Files
# --------------------------------------------------------------------------


def prediction_paths(spark_dir: Path) -> dict[str, dict[str, Path]]:
    """``M2g-short-<tag>`` (the v0.7 chain) and ``M2g-facts-<tag>`` (the M2g-F chain) for the five models in `spark_dir`."""
    v7 = V7()
    return {m: {form: Path(spark_dir) / f"M2g-{form}-{v7.SPARK_TAGS[m]}.predictions.jsonl" for form in FORMS} for m in v7.GATE_MODELS}


def refuse_write(path: Path) -> None:
    """``scripts/verdict_v07.py``'s write guard (never into earlier reports, the sealed gates or the sets), and never onto the v0.7 verdict."""
    V7().refuse_write(path)
    if Path(path).resolve() in {(ARTIFACTS / "reports" / "v07" / "verdict.json").resolve(), (ARTIFACTS / "reports" / "v07" / "verdict.md").resolve()}:
        raise SystemExit(f"{path}: refusing to overwrite the v0.7 verdict")


def _exclusions(args: argparse.Namespace, notes: list[str]) -> set[str]:
    """The pre-declared M2g exclusions of ``overlap-v07.json`` (``scripts/verdict_v07.py``'s file and keys)."""
    v7 = V7()
    if v7.OVERLAP_V07.is_file():
        recheck = json.loads(v7.OVERLAP_V07.read_text(encoding="utf-8"))
        exclude = set(recheck.get("exclude", {}).get("M2g_ids") or [])
        notes.append(f"overlap-v07.json (train-v07 sha256 {str(recheck.get('train_v07', {}).get('sha256'))[:12]}): {len(exclude)} M2g moments left "
                     "out of both forms.")
        return exclude
    if not args.no_v07_recheck:
        raise SystemExit(f"{v7.OVERLAP_V07} is missing — run scripts/v07_gates_build.py recheck-v07 on v0.7's training input first")
    notes.append("overlap-v07.json absent: no exclusions (--no-v07-recheck)")
    return set()


def load_inputs(args: argparse.Namespace, problems: list[str], notes: list[str]) -> tuple[list, list, dict, dict]:
    """(M2g-short records, M2g-F records, predictions, sources): every prediction file checked against its gate file, M2g's scoring conditions
    and its checkpoint (``verdict_v07.check_gate_source``) and for completeness; the facts file against its sealed manifest."""
    from robo_jev.decisions.records import file_sha256, read_records

    v7 = V7()
    short_path = v7.M2G_DIR / "M2g-short.jsonl"
    if not FACTS_MANIFEST.is_file() or not FACTS_FILE.is_file():
        raise SystemExit(f"{FACTS_FILE} or its manifest is missing — build M2g-F first (scripts/v07_gates_build.py build-facts)")
    manifest = json.loads(FACTS_MANIFEST.read_text(encoding="utf-8"))
    if file_sha256(FACTS_FILE) != manifest["file"]["sha256"]:
        problems.append(f"{FACTS_FILE.name}: not the sealed file (sha256 {manifest['file']['sha256'][:12]})")
    short, facts = read_records(short_path), read_records(FACTS_FILE)
    paths = prediction_paths(args.spark_dir)
    missing = [str(p) for m in v7.GATE_MODELS for p in paths[m].values() if not p.is_file()]
    if missing:
        raise SystemExit("missing prediction files:\n  " + "\n  ".join(missing))
    checkpoints = v7.v07_checkpoints(args)
    preds: dict[str, dict[str, list]] = {}
    sources: dict[str, Any] = {}
    for m in v7.GATE_MODELS:
        preds[m] = {}
        for form, gate_file, records in (("short", short_path, short), ("facts", FACTS_FILE, facts)):
            path = paths[m][form]
            src = v7.check_gate_source(path, gate_file, v7.M2G_CONDITIONS, checkpoints[m])
            sources[f"{m} {form}"] = src
            problems += [f"{m} M2g-{form}: {p}" for p in src["problems"]]
            preds[m][form] = v7._read(path)
            absent = {str(r["id"]) for r in records} - {str(x["id"]) for x in preds[m][form]}
            if absent:
                problems.append(f"{m} M2g-{form}: {len(absent)} records without a prediction (first {sorted(absent)[:3]})")
    return short, facts, preds, sources


# --------------------------------------------------------------------------
# Markdown
# --------------------------------------------------------------------------


def _fmt(value: Any, digits: int = 3, signed: bool = False) -> str:
    if value is None:
        return "—"
    return f"{value:+.{digits}f}" if signed else f"{value:.{digits}f}"


def _cid(d: Mapping[str, Any] | None) -> str:
    if not d or d.get("value") is None:
        return "—"
    return f"{_fmt(d['value'], signed=True)} [{_fmt(d.get('low'), signed=True)}, {_fmt(d.get('high'), signed=True)}]"


def render(result: Mapping[str, Any]) -> str:
    r, o = result["readings"], result["readings"]["outcome"]
    e = r.get("rule") or {}
    names = V7().MODELS
    lines = [f"# v0.7 secondary reading — facts in instructions (M2g-F): {o['outcome']}", "",
             f"Reading written {READING['written']}. Applied by `scripts/verdict_v07_facts.py` ({SCRIPT_VERSION}, git {result.get('git')}); "
             f"{result['n_boot']} paired cluster-bootstrap draws (seed {result['seed']}); call view; **seed means** (arm v0.7 = s0 + s1 vs arm v0.6 = "
             f"s0 + s1). {READING['scope'].capitalize()}.", "",
             "## The reading", "",
             "| reading | value [95 %] | check |", "|---|---|---|",
             f"| v0.6 gap_F = DC(F) − DC(short) (ACTIVE iff ≥ +0.05) | {_cid(e.get('gap_v06'))} | {o['status']} |"]
    checks = o.get("checks") or {}
    lines += [f"| Δgap_F v0.7 − v0.6 (≤ −0.05, upper < 0; active only) | {_cid(e.get('dgap'))} | {checks.get('dgap_le_-0.05', '—')} / {checks.get('dgap_high_lt_0', '—')} |",
              f"| v0.7 gap_F (≤ +0.03) | {_cid(e.get('gap_v07'))} | {checks.get('gap_v07_le_+0.03', '—')} |",
              f"| v0.7 ΔCA_F = CA(F) − CA(short) (≥ −0.05) | {_cid(e.get('dca_v07'))} | {checks.get('dca_v07_ge_-0.05', '—')} |",
              f"| v0.6 ΔCA_F (reported) | {_cid(e.get('dca_v06'))} | |", "",
              f"**{o['outcome']}** ({o['status']}; questions {e.get('n')}) → {o['consequence']}."]
    if o.get("note"):
        lines += ["", f"Note: {o['note']}."]
    lines += ["", "## Also reported (not interpreted)", "", "### F − short per model and arm (paired, the questions both forms answered)", "",
              "| model / arm | DC short | DC F | ΔDC | ΔCA | ΔLU | ΔLC | ΔGAP |", "|---|---|---|---|---|---|---|---|"]
    for m, d in (r.get("facts_minus_short") or {}).items():
        lv = r["levels"][m]
        lines.append(f"| {names.get(m, m)} | {_fmt(lv['short'].get('DC', {}).get('value'))} | {_fmt(lv['facts'].get('DC', {}).get('value'))} | "
                     + " | ".join(_cid(d.get(k)) for k in METRICS) + " |")
    for title, group in (("per template", "by_template"), ("by source set", "by_set")):
        lines += ["", f"### {title.capitalize()} (arms)", "", "| group | n | v0.6 gap_F | v0.7 gap_F | Δgap_F | v0.7 ΔCA_F | base ΔDC |", "|---|---|---|---|---|---|---|"]
        for name, entry in (r.get(group) or {}).items():
            rule = entry.get("rule") or {}
            base = ((entry.get("facts_minus_short") or {}).get("base") or {}).get("DC")
            lines.append(f"| {name} | {rule.get('n')} | {_cid(rule.get('gap_v06'))} | {_cid(rule.get('gap_v07'))} | {_cid(rule.get('dgap'))} | "
                         f"{_cid(rule.get('dca_v07'))} | {_cid(base)} |")
    lines += ["", "Seed spread (|s0 − s1| of F − short per arm): " + "; ".join(f"{arm} " + ", ".join(f"{k} {_fmt(v)}" for k, v in e2.items())
                                                                         for arm, e2 in (r.get("seed_spread") or {}).items()) + "."]
    lines += ["", "## Notes", ""] + [f"- {note}" for note in result["notes"]] + ["", "## Reading (verbatim parts)", ""] + [f"- **{k}** — {t}" for k, t in READING.items()]
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--s0-dir", type=Path, default=None, help="fetched reports of the v0.7 seed-0 job (locates its checkpoint)")
    parser.add_argument("--s1-dir", type=Path, default=None, help="fetched reports of the v0.7 seed-1 job")
    parser.add_argument("--spark-dir", type=Path, default=None, help="the Spark predictions (M2g-short-<tag> and M2g-facts-<tag>)")
    parser.add_argument("--no-v07-recheck", action="store_true", help="read without overlap-v07.json (never for the reading itself)")
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--allow-mismatch", action="store_true", help="read even if a prediction file fails its checks")
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args(argv)
    args.spark_dir = args.spark_dir or V7().SPARK_DIR
    refuse_write(args.out)
    if args.s0_dir is None or args.s1_dir is None:
        raise SystemExit("--s0-dir and --s1-dir are required (they locate the v0.7 checkpoints the predictions must have been scored with)")
    notes: list[str] = []
    problems: list[str] = []
    exclude = _exclusions(args, notes)
    short, facts, preds, sources = load_inputs(args, problems, notes)
    if problems and not args.allow_mismatch:
        raise SystemExit("prediction files fail their checks:\n  " + "\n  ".join(problems[:30]))
    _log("readings")
    readings = facts_readings(short, facts, preds, n_boot=args.n_boot, seed=args.seed, exclude=exclude)
    notes += ["Seed means: an arm's question = the mean of its two seeds' call distributions (each 1/L-weighted over its listings), on the "
              "questions both seeds answered; gap_F, dgap_F and dCA_F read the four sources (each arm's F and short) on the questions all four "
              "answered, with one cluster bootstrap (cluster = the moment's source set and generator group, a one-fact pair together).",
              "M2g-short was scored by the v0.7 chain (artifacts/scratch/v07b/score_spark.sh) and M2g-F by artifacts/scratch/v07f/score_facts.sh, both "
              "with M2g's settings (scripts/diagnose_decisions.py --orders seeded: one listing per question, the same order in both forms — the ids "
              "are identical)."]
    result = {"kind": "v07-verdict-facts", "script_version": SCRIPT_VERSION, "git": _git_sha(), "created": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
              "reading_text": READING, "n_boot": args.n_boot, "seed": args.seed, "outcome": readings["outcome"]["outcome"], "readings": readings,
              "sources": sources, "problems": problems, "notes": notes, "inputs": {k: str(v) for k, v in vars(args).items() if isinstance(v, Path)}}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=1, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    md = render(result)
    args.out.with_suffix(".md").write_text(md, encoding="utf-8")
    print(md.split("## Also reported")[0])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
