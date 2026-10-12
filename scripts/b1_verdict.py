#!/usr/bin/env python
"""Task B1 — apply the PRE-DECLARED CAUSE CHECK B1 (progress.md, written 2026-10-13 ~03:40 KST, before any B1 data existed) mechanically.

    python scripts/b1_verdict.py --input artifacts/b1/b1-infamily.jsonl --out artifacts/reports/b1/verdict.json

Question: why did v0.3 learn direction and comparison outside (T8b) but not the absolute risk level? On a fresh in-family set built by the
v0.3b generator (B1-infamily: families tilt, grip, tower; new seed and group indices; never trained on), three models scored on the Spark
GPU under the T8 conditions (v0c-p3, chat, counterbalanced orders, shuffle seed 1): the untrained Qwen3.5-2B, v0.1-mix (engine only)
and v0.3.

Readings (System-1 metrics of :mod:`robo_jev.bench.system1`, mapped onto mjgen records by :mod:`robo_jev.bench.infamily`): single order;
cluster bootstrap 95 % (percentile), 2,000 draws, seed 0, cluster = the generator group (a one-fact pair's two records together).

* **PRIMARY** — balanced band accuracy on the *robust* yes/no + score questions (band identical under all three priors,
  ``meta.robustness[q].robust``), v0.3 − base (paired, stratified by the label's band).
* **Rule** — R1 "levels learned in-family" (→ H-transfer) iff Δ lower bound > +0.10 AND v0.3 ≥ 0.55; R2 "levels not learned in-family"
  (→ H-signal) iff Δ upper bound < +0.10; else R3 "partial" (both).
* **Positive control** — v0.3's in-family direction (one-fact pairs, ``meta.pair``) and comparison (in-record comparison questions)
  ≥ 0.80 each (points); otherwise the instrument is suspect and no interpretation is made.
* **Secondary** (reported, not deciding) — severe errors on robust questions; per family; answers on prior-ambiguous questions (spread,
  hedge share); v0.1-mix vs base; the 3×3 band confusion per model; plus band metrics over all yes/no + score questions (the T8b
  population), coverage, an oracle check and the native choice questions.

Every prediction file is checked against the set it scores: its diagnosis summary (``<name>.json``) must record the sha256 of the B1 file
and the T8 scoring conditions. The set's manifest must record a passed zero-overlap check with the training records. Writes
``verdict.json`` and ``verdict.md``.
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

SCRIPT_VERSION = "b1-verdict-1"
RULE = {
    "written": "2026-10-13 ~03:40 KST, before any B1 data or result existed (progress.md: PRE-DECLARED CAUSE CHECK B1)",
    "primary": "balanced band accuracy on robust yes/no+score questions, v0.3 - base",
    "R1": "'levels learned in-family' (-> H-transfer) iff delta lower bound > +0.10 AND v0.3 >= 0.55",
    "R2": "'levels not learned in-family' (-> H-signal) iff delta upper bound < +0.10",
    "R3": "else 'partial' (both)",
    "positive_control": "v0.3 in-family direction and comparison >= 0.80 each, else the instrument is suspect and no interpretation is made",
    "next": "B2 runs only under R1 or R3",
}
R1_LOW = 0.10
R1_MIN = 0.55
R2_HIGH = 0.10
CONTROL_MIN = 0.80
MEANING = {
    "R1": "levels learned in-family -> H-transfer: next = B2 (T8b band robustness under the same alternative-prior rule as v0.3b) and level diversity",
    "R2": "levels not learned in-family -> H-signal: the data fix is the level signal itself",
    "R3": "partial -> both H-signal and H-transfer: B2 runs, and the level signal needs fixing",
}
#: The T8 scoring conditions (artifacts/reports/t8/T8b-*.json).
CONDITIONS = {"mode": "direct", "format": "chat", "prompt_version": "v0c-p3", "orders": "counterbalanced", "shuffle_seed": 1}
SET = "B1"
REPORTS = REPO / "artifacts" / "reports" / "b1"
#: (tag, label, role, default prediction stem) in report order; exactly one baseline.
MODELS = (("base", "untrained Qwen3.5-2B", "baseline", "B1-base-2b"), ("v01mix", "v0.1-mix (engine only)", "ours", "B1-v01mix-2b"),
          ("v03", "v0.3", "ours", "B1-v03-2b"))
FAMILIES = ("tilt", "grip", "tower")
NATIVE_KINDS = ("direction", "compare", "outcome")


def _log(message: str) -> None:
    print(f"[b1-verdict {dt.datetime.now().strftime('%H:%M:%S')}] {message}", file=sys.stderr, flush=True)


def _git_sha() -> str | None:
    try:
        return subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True).stdout.strip() or None
    except (OSError, subprocess.CalledProcessError):
        return None


# --------------------------------------------------------------------------
# The rule (pure)
# --------------------------------------------------------------------------


def rule_outcome(delta_low: float | None, delta_high: float | None, v03_value: float | None) -> str:
    """R1 iff Δ lower bound > +0.10 AND v0.3 ≥ 0.55; R2 iff Δ upper bound < +0.10; else R3. Raises when a reading is missing (the rule
    cannot be computed as written)."""
    if delta_low is None or delta_high is None or v03_value is None:
        raise ValueError("the rule cannot be computed: a reading is missing (delta interval or v0.3's primary value)")
    if delta_low > R1_LOW and v03_value >= R1_MIN:
        return "R1"
    if delta_high < R2_HIGH:
        return "R2"
    return "R3"


def control_outcome(direction: float | None, comparison: float | None) -> dict[str, Any]:
    """Positive control: v0.3's direction and comparison points ≥ 0.80 each."""
    if direction is None or comparison is None:
        raise ValueError("the positive control cannot be computed: direction or comparison has no items")
    checks = {"direction_ge_0.80": direction >= CONTROL_MIN, "comparison_ge_0.80": comparison >= CONTROL_MIN}
    return {"direction": direction, "comparison": comparison, "checks": checks, "pass": all(checks.values())}


def decide(delta: Mapping[str, Any], v03_primary: Mapping[str, Any], direction: Mapping[str, Any], comparison: Mapping[str, Any]) -> dict[str, Any]:
    """The rule and the positive control on computed readings → the verdict (``outcome`` R1/R2/R3, ``interpreted``, ``meaning``)."""
    outcome = rule_outcome(delta.get("low"), delta.get("high"), v03_primary.get("value"))
    control = control_outcome(direction.get("value"), comparison.get("value"))
    checks = {"R1": {"delta_low_gt_0.10": delta["low"] > R1_LOW, "v03_ge_0.55": v03_primary["value"] >= R1_MIN},
              "R2": {"delta_high_lt_0.10": delta["high"] < R2_HIGH}}
    return {"outcome": outcome, "checks": checks, "positive_control": control, "interpreted": control["pass"],
            "meaning": MEANING[outcome] if control["pass"] else "instrument suspect (positive control failed): no interpretation is made",
            "b2_runs": bool(control["pass"] and outcome in ("R1", "R3"))}


# --------------------------------------------------------------------------
# Readings
# --------------------------------------------------------------------------


def _summary(items: Mapping[Any, Any], *, n_boot: int, seed: int, stratified: bool = False) -> dict[str, Any]:
    from robo_jev.bench import system1

    return system1.summarize(dict(items), n_boot=n_boot, seed=seed, stratified=stratified)


def _paired(a: Mapping[Any, Any], b: Mapping[Any, Any], *, n_boot: int, seed: int, stratified: bool = False) -> dict[str, Any] | None:
    from robo_jev.bench import system1

    return system1.paired_difference(dict(a), dict(b), n_boot=n_boot, seed=seed, stratified=stratified)


def model_items(records: Sequence[dict[str, Any]], predictions: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Every item a model has on B1, clustered by group: question items (all / robust / prior-ambiguous yes/no + score), direction and
    comparison."""
    from robo_jev.bench import infamily, system1

    questions = {metric: infamily.regroup(items, records) for metric, items in system1.question_items(SET, records, predictions).items()}
    robust, ambiguous = infamily.robust_keys(records), infamily.ambiguous_keys(records)
    return {"all": questions, "robust": {m: infamily.keep(v, robust) for m, v in questions.items()},
            "ambiguous": {m: infamily.keep(v, ambiguous) for m, v in questions.items()},
            "direction": infamily.direction_items(SET, records, predictions), "comparison": infamily.comparison_items(SET, records, predictions)}


BAND_METRICS = ("band_balanced", "band_accuracy", "band_recall:low", "band_recall:middle", "band_recall:high", "severe_error", "severe_miss",
                "severe_false_alarm")


def _band_block(items: Mapping[str, Any], *, n_boot: int, seed: int) -> dict[str, Any]:
    return {m: _summary(items.get(m) or {}, n_boot=n_boot if m in ("band_balanced", "severe_error") else 0, seed=seed, stratified=m == "band_balanced")
            for m in BAND_METRICS}


def _native(records: Sequence[dict[str, Any]], predictions: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Native choice questions of the mjgen records (single order): the share of listings whose argmax is in the label's argmax set, per
    question kind (direction / compare / outcome) — what the model answers in the format v0.3 was trained on (not a System-1 metric)."""
    from robo_jev.bench import system1
    from robo_jev.decisions.metrics import choice_metrics

    predicted = {str(p["id"]): p for p in predictions}
    totals: dict[str, list[float]] = {kind: [0.0, 0.0] for kind in NATIVE_KINDS}
    for record in records:
        kinds = record["meta"].get("question_kind") or {}
        answers = (predicted.get(str(record["id"])) or {}).get("questions") or {}
        for name, kind in kinds.items():
            if kind not in totals or name not in answers:
                continue
            label = record["labels"][name]
            for listing, weight in system1._answers(answers[name], "single_order"):
                totals[kind][0] += weight * float(choice_metrics(listing["probabilities"], label)["match"])
            totals[kind][1] += 1.0
    return {kind: {"match": (hit / n if n else None), "questions": int(n)} for kind, (hit, n) in totals.items()}


KINDS = ("key", "severe", "score_cf")


def instrument_checks(records: Sequence[dict[str, Any]], predictions: Mapping[str, Sequence[dict[str, Any]]], *, n_boot: int, seed: int) -> dict[str, Any]:
    """Non-gating checks of the instrument: the primary reading per question kind (key / severe / score_cf) and on clear labels only (more
    than 0.05 from a band edge), and T8b's within-record comparison structure (plan key question vs the counterfactual score question,
    labels ≥ 0.15 apart, :func:`robo_jev.bench.infamily.plan_counterfactual_pairs`)."""
    from robo_jev.bench import infamily, system1

    robust = infamily.robust_keys(records)
    kind_of = {(str(r["id"]), q): r["meta"]["question_kind"][q] for r in records for q in r["request"]["questions"]}
    labels = {(str(r["id"]), q): float(r["labels"][q].get("p_yes", r["labels"][q].get("value"))) for r in records for q in r["request"]["questions"]
              if r["request"]["questions"][q]["type"] in ("noul", "score")}
    clear = {k for k in robust if system1.is_clear(labels[k])}
    pairs = infamily.plan_counterfactual_pairs(records)
    out: dict[str, Any] = {"primary_by_kind": {}, "primary_clear": {}, "plan_vs_counterfactual": {}}
    for tag, preds in predictions.items():
        items = infamily.regroup(system1.question_items(SET, records, preds).get("band_balanced") or {}, records)
        out["primary_by_kind"][tag] = {kind: _summary(infamily.keep(items, {k for k in robust if kind_of[k] == kind}), n_boot=n_boot, seed=seed, stratified=True)
                                       for kind in KINDS}
        out["primary_clear"][tag] = _summary(infamily.keep(items, clear), n_boot=n_boot, seed=seed, stratified=True)
        ordered = infamily.regroup(system1.order_items(SET, records, preds, pairs, min_move=system1.COMPARISON_MIN), records)
        out["plan_vs_counterfactual"][tag] = _summary(ordered, n_boot=n_boot, seed=seed)
    out["robust_label_bands_by_kind"] = {kind: {b: sum(1 for k in robust if kind_of[k] == kind and system1.band(labels[k]) == b) for b in infamily.BANDS}
                                         for kind in KINDS}
    return out


def readings(records: Sequence[dict[str, Any]], predictions: Mapping[str, Sequence[dict[str, Any]]], *, n_boot: int = 2000, seed: int = 0) -> dict[str, Any]:
    """Every reading of the verdict for the models in `predictions` (tags of :data:`MODELS`)."""
    from robo_jev.bench import infamily, system1

    items = {tag: model_items(records, preds) for tag, preds in predictions.items()}
    oracle = model_items(records, system1.oracle_predictions(records))
    robust, ambiguous = infamily.robust_keys(records), infamily.ambiguous_keys(records)
    scalar = robust | ambiguous
    out: dict[str, Any] = {"models": {}, "paired": {}, "per_family": {}, "oracle": {}, "coverage": {}}
    for tag, found in items.items():
        out["models"][tag] = {
            "primary": _summary(found["robust"].get("band_balanced") or {}, n_boot=n_boot, seed=seed, stratified=True),
            "robust": _band_block(found["robust"], n_boot=n_boot, seed=seed),
            "all": _band_block(found["all"], n_boot=n_boot, seed=seed),
            "ambiguous": _band_block(found["ambiguous"], n_boot=0, seed=seed),
            "direction": _summary(found["direction"], n_boot=n_boot, seed=seed),
            "comparison": _summary(found["comparison"], n_boot=n_boot, seed=seed),
            "safe_choice": _summary(found["all"].get("safe_choice") or {}, n_boot=0, seed=seed),
            "catastrophic_choice": _summary(found["all"].get("catastrophic_choice") or {}, n_boot=0, seed=seed),
            "confusion": {"robust": infamily.band_confusion(records, predictions[tag], robust), "all": infamily.band_confusion(records, predictions[tag], scalar)},
            "answers": {"ambiguous": infamily.answer_profile(records, predictions[tag], ambiguous), "robust": infamily.answer_profile(records, predictions[tag], robust)},
            "native": _native(records, predictions[tag]),
        }
        out["coverage"][tag] = {"band": len(found["all"].get("band_accuracy") or {}) / max(1, len(oracle["all"]["band_accuracy"])),
                                "direction": len(found["direction"]) / max(1, len(oracle["direction"])),
                                "comparison": len(found["comparison"]) / max(1, len(oracle["comparison"]))}
    out["checks"] = instrument_checks(records, predictions, n_boot=n_boot, seed=seed)
    out["oracle"] = {"primary": _summary(oracle["robust"]["band_balanced"], n_boot=0, seed=seed, stratified=True)["value"],
                     "direction": _summary(oracle["direction"], n_boot=0, seed=seed)["value"],
                     "comparison": _summary(oracle["comparison"], n_boot=0, seed=seed)["value"],
                     "severe_error_robust": _summary(oracle["robust"]["severe_error"], n_boot=0, seed=seed)["value"],
                     "items": {"robust_band": len(oracle["robust"]["band_accuracy"]), "all_band": len(oracle["all"]["band_accuracy"]),
                               "ambiguous_band": len(oracle["ambiguous"]["band_accuracy"]), "direction": len(oracle["direction"]),
                               "comparison": len(oracle["comparison"])}}
    pairs = [(a, "base") for a in ("v03", "v01mix") if a in items and "base" in items] + ([("v03", "v01mix")] if {"v03", "v01mix"} <= set(items) else [])
    for a, b in pairs:
        ia, ib = items[a], items[b]
        out["paired"][f"{a}-{b}"] = {
            "primary": _paired(ia["robust"].get("band_balanced") or {}, ib["robust"].get("band_balanced") or {}, n_boot=n_boot, seed=seed, stratified=True),
            "severe_error_robust": _paired(ia["robust"].get("severe_error") or {}, ib["robust"].get("severe_error") or {}, n_boot=n_boot, seed=seed),
            "band_balanced_all": _paired(ia["all"].get("band_balanced") or {}, ib["all"].get("band_balanced") or {}, n_boot=n_boot, seed=seed, stratified=True),
            "severe_error_all": _paired(ia["all"].get("severe_error") or {}, ib["all"].get("severe_error") or {}, n_boot=n_boot, seed=seed),
            "direction": _paired(ia["direction"], ib["direction"], n_boot=n_boot, seed=seed),
            "comparison": _paired(ia["comparison"], ib["comparison"], n_boot=n_boot, seed=seed),
        }
    for family in FAMILIES:
        entry: dict[str, Any] = {"models": {}, "paired": {}}
        split = {tag: {"primary": infamily.by_family(found["robust"].get("band_balanced") or {}, records).get(family, {}),
                       "severe": infamily.by_family(found["robust"].get("severe_error") or {}, records).get(family, {}),
                       "direction": infamily.by_family(found["direction"], records).get(family, {}),
                       "comparison": infamily.by_family(found["comparison"], records).get(family, {})} for tag, found in items.items()}
        for tag, parts in split.items():
            entry["models"][tag] = {"primary": _summary(parts["primary"], n_boot=n_boot, seed=seed, stratified=True),
                                    "severe_error_robust": _summary(parts["severe"], n_boot=0, seed=seed),
                                    "direction": _summary(parts["direction"], n_boot=0, seed=seed),
                                    "comparison": _summary(parts["comparison"], n_boot=0, seed=seed)}
        for a, b in pairs:
            entry["paired"][f"{a}-{b}"] = {"primary": _paired(split[a]["primary"], split[b]["primary"], n_boot=n_boot, seed=seed, stratified=True)}
        fam_records = [r for r in records if infamily.family_of(r) == family]
        entry["robust_questions"] = len(infamily.robust_keys(fam_records))
        entry["ambiguous_questions"] = len(infamily.ambiguous_keys(fam_records))
        out["per_family"][family] = entry
    return out


# --------------------------------------------------------------------------
# Inputs and checks
# --------------------------------------------------------------------------


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    from robo_jev.decisions.records import read_records

    return read_records(path)


def check_prediction(path: Path, input_sha256: str) -> dict[str, Any]:
    """The diagnosis summary next to a prediction file: same B1 file (sha256) and the T8 scoring conditions; returns its identity."""
    summary_path = path.with_name(path.name.replace(".predictions.jsonl", ".json"))
    if not summary_path.is_file():
        raise SystemExit(f"{path}: no diagnosis summary {summary_path.name} next to it")
    config = json.loads(summary_path.read_text(encoding="utf-8"))["config"]
    problems = []
    if config.get("input_sha256") != input_sha256:
        problems.append(f"input sha256 {str(config.get('input_sha256'))[:12]} != B1 {input_sha256[:12]}")
    for key, want in CONDITIONS.items():
        if config.get(key) != want:
            problems.append(f"{key} = {config.get(key)!r}, want {want!r}")
    if problems:
        raise SystemExit(f"{path}: " + "; ".join(problems))
    checkpoint = config.get("checkpoint") or {}
    return {"predictions": str(path), "summary": str(summary_path), "model": config.get("model"), "records": config.get("records"),
            "conditions": {k: config.get(k) for k in CONDITIONS}, "checkpoint_sha256": checkpoint.get("sha256"),
            "checkpoint_run": checkpoint.get("run_id"), "checkpoint_step": checkpoint.get("step"), "git": config.get("git"), "created": config.get("created")}


def set_profile(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Composition of B1: records, groups, questions by type and kind, robust share per family and per kind (yes/no + score and all)."""
    from collections import Counter

    from robo_jev.bench import infamily
    from robo_jev.bench.system1 import band

    out: dict[str, Any] = {"records": len(records), "groups": len({infamily.group_of(r) for r in records}), "by_family": {}}
    for family in FAMILIES + ("all",):
        rows = [r for r in records if family == "all" or infamily.family_of(r) == family]
        kinds: Counter = Counter()
        robust: Counter = Counter()
        types: Counter = Counter()
        for r in rows:
            for name, question in r["request"]["questions"].items():
                kind = r["meta"]["question_kind"][name]
                kinds[kind] += 1
                types[question["type"]] += 1
                robust[kind] += int(bool(r["meta"]["robustness"][name]["robust"]))
        robust_scalar = infamily.robust_keys(rows)
        scalar = len(robust_scalar) + len(infamily.ambiguous_keys(rows))
        bands: Counter = Counter()
        for r in rows:
            for name in r["request"]["questions"]:
                if (str(r["id"]), name) in robust_scalar:
                    label = r["labels"][name]
                    bands[band(float(label.get("p_yes", label.get("value"))))] += 1
        out["by_family"][family] = {
            "records": len(rows), "groups": len({infamily.group_of(r) for r in rows}), "pairs": sum(1 for r in rows if (r["meta"].get("pair") or {}).get("role") == "base"),
            "questions": sum(kinds.values()), "types": dict(sorted(types.items())),
            "robust_share": sum(robust.values()) / max(1, sum(kinds.values())),
            "robust_by_kind": {k: {"questions": kinds[k], "robust": robust[k], "share": robust[k] / kinds[k]} for k in sorted(kinds)},
            "scalar_questions": scalar, "robust_scalar": len(robust_scalar),
            "robust_scalar_share": len(robust_scalar) / max(1, scalar), "robust_scalar_label_bands": dict(bands),
        }
    return out


# --------------------------------------------------------------------------
# Markdown
# --------------------------------------------------------------------------


def _f(value: Any, digits: int = 3, signed: bool = False) -> str:
    if value is None:
        return "—"
    return f"{value:+.{digits}f}" if signed else f"{value:.{digits}f}"


def _ci(summary: Mapping[str, Any] | None, signed: bool = False) -> str:
    if not summary or summary.get("value") is None:
        return "—"
    text = _f(summary["value"], signed=signed)
    if summary.get("low") is not None:
        text += f" [{_f(summary['low'], signed=signed)}, {_f(summary['high'], signed=signed)}]"
    return text


def render_markdown(result: Mapping[str, Any]) -> str:
    labels = {m["tag"]: m["label"] for m in result["models"]}
    tags = [m["tag"] for m in result["models"]]
    r = result["readings"]
    v = result["verdict"]
    lines = ["# B1 — does v0.3 judge risk levels inside its own MuJoCo families? (pre-declared cause check)", "",
             f"Rule written {RULE['written']}. Applied mechanically by `scripts/b1_verdict.py` ({SCRIPT_VERSION}, git {result['git']}). "
             "Single-order reading; cluster bootstrap 95 % (2,000 draws, seed 0, cluster = generator group: a one-fact pair together).", ""]
    lines += ["## Verdict", "",
              f"- **Outcome: {v['outcome']}** — " + ("; ".join(f"{k} {'yes' if ok else 'no'}" for part in v["checks"].values() for k, ok in part.items())) + ".",
              f"- **Positive control: {'PASS' if v['positive_control']['pass'] else 'FAIL'}** — v0.3 direction {_f(v['positive_control']['direction'])}, "
              f"comparison {_f(v['positive_control']['comparison'])} (each ≥ 0.80).",
              f"- **Reading:** {v['meaning']}.", f"- B2 runs: {'yes' if v['b2_runs'] else 'no'}.", ""]
    primary = r["paired"].get("v03-base", {}).get("primary") or {}
    lines += ["## Primary — balanced band accuracy on robust yes/no + score questions", "",
              "| model | balanced band [95 %] | recall low | recall middle | recall high | n questions | clusters |", "|---|---|---|---|---|---|---|"]
    for tag in tags:
        m = r["models"][tag]
        lines.append(f"| {labels[tag]} | {_ci(m['primary'])} | {_f(m['robust']['band_recall:low']['value'])} | {_f(m['robust']['band_recall:middle']['value'])} | "
                     f"{_f(m['robust']['band_recall:high']['value'])} | {m['primary']['n']} | {m['primary']['clusters']} |")
    lines += ["", "| paired Δ | primary [95 %] |", "|---|---|"]
    for key, entry in r["paired"].items():
        a, b = key.split("-")
        lines.append(f"| {labels[a]} − {labels[b]} | {_ci(entry['primary'], signed=True)} |")
    lines += ["", f"Rule inputs: Δ(v0.3 − base) = {_ci(primary, signed=True)}; v0.3 = {_f(r['models']['v03']['primary']['value'])}.", ""]
    lines += ["## Positive control and other System-1 readings", "",
              "| model | direction [95 %] | comparison [95 %] | severe, robust [95 %] | band balanced, all yes/no+score [95 %] | severe, all | safe choice | catastrophic choice |",
              "|---|---|---|---|---|---|---|---|"]
    for tag in tags:
        m = r["models"][tag]
        lines.append(f"| {labels[tag]} | {_ci(m['direction'])} | {_ci(m['comparison'])} | {_ci(m['robust']['severe_error'])} | {_ci(m['all']['band_balanced'])} | "
                     f"{_f(m['all']['severe_error']['value'])} | {_f(m['safe_choice']['value'])} | {_f(m['catastrophic_choice']['value'])} |")
    lines += ["", "| paired Δ | severe, robust | band balanced, all | direction | comparison |", "|---|---|---|---|---|"]
    for key, entry in r["paired"].items():
        a, b = key.split("-")
        lines.append(f"| {labels[a]} − {labels[b]} | {_ci(entry['severe_error_robust'], signed=True)} | {_ci(entry['band_balanced_all'], signed=True)} | "
                     f"{_ci(entry['direction'], signed=True)} | {_ci(entry['comparison'], signed=True)} |")
    lines += ["", "Severe errors on robust questions split into misses (label high, answer low) and false alarms (label low, answer high):", "",
              "| model | severe | miss | false alarm |", "|---|---|---|---|"]
    for tag in tags:
        m = r["models"][tag]["robust"]
        lines.append(f"| {labels[tag]} | {_f(m['severe_error']['value'])} | {_f(m['severe_miss']['value'])} | {_f(m['severe_false_alarm']['value'])} |")
    lines += ["", "## Per family", "", "| family | robust / ambiguous yes-no+score | model | primary [95 %] | severe, robust | direction | comparison |",
              "|---|---|---|---|---|---|---|"]
    for family, entry in r["per_family"].items():
        for i, tag in enumerate(tags):
            m = entry["models"][tag]
            head = f"{family} | {entry['robust_questions']} / {entry['ambiguous_questions']}" if i == 0 else " | "
            lines.append(f"| {head} | {labels[tag]} | {_ci(m['primary'])} | {_f(m['severe_error_robust']['value'])} | {_f(m['direction']['value'])} | "
                         f"{_f(m['comparison']['value'])} |")
        for key, paired in entry["paired"].items():
            a, b = key.split("-")
            lines.append(f"|  |  | Δ {labels[a]} − {labels[b]} | {_ci(paired['primary'], signed=True)} |  |  |  |")
    lines += ["", "## Answers on prior-ambiguous questions (band depends on the prior)", "",
              "Spread = weighted SD and 10th–90th percentile of the answers; hedge = answers in the middle band; 'in prior bands' = the answer's band "
              "is one the label takes under some prior; 'in prior range' = inside the label's min–max over the priors. Robust questions for contrast.", "",
              "| model | questions | mean | SD | p10–p90 | low / middle / high | hedge share | in prior bands | in prior range | robust: hedge share | robust: SD |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
    for tag in tags:
        a = r["models"][tag]["answers"]["ambiguous"]
        b = r["models"][tag]["answers"]["robust"]
        if not a.get("questions"):
            continue
        shares = a["band_shares"]
        lines.append(f"| {labels[tag]} | {a['questions']:.0f} | {_f(a['mean'])} | {_f(a['sd'])} | {_f(a['p10'], 2)}–{_f(a['p90'], 2)} | "
                     f"{_f(shares['low'], 2)} / {_f(shares['middle'], 2)} / {_f(shares['high'], 2)} | {_f(a['hedge_share'])} | {_f(a['in_prior_bands'])} | "
                     f"{_f(a['in_prior_range'])} | {_f(b.get('hedge_share'))} | {_f(b.get('sd'))} |")
    lines += ["", "## 3×3 band confusion (robust yes/no + score questions; rows = label band, shares of the row)", ""]
    for tag in tags:
        conf = r["models"][tag]["confusion"]["robust"]
        lines += [f"**{labels[tag]}**", "", "| label \\ answer | low | middle | high | questions |", "|---|---|---|---|---|"]
        for row in ("low", "middle", "high"):
            shares = conf["rows"][row] or {}
            lines.append(f"| {row} | {_f(shares.get('low'), 2)} | {_f(shares.get('middle'), 2)} | {_f(shares.get('high'), 2)} | {sum(conf['counts'][row].values()):.0f} |")
        lines.append("")
    checks = r.get("checks") or {}
    if checks:
        bands = checks["robust_label_bands_by_kind"]
        lines += ["## Instrument checks (not deciding)", "",
                  "The primary reading per question kind and on clear labels (more than 0.05 from a band edge); and T8b's within-record "
                  "structure — the plan's key question against the score question at the counterfactual value, labels ≥ 0.15 apart, answers "
                  "ordered like the labels.", "",
                  "| model | key [95 %] | severe [95 %] | score_cf [95 %] | clear labels [95 %] | plan vs counterfactual [95 %] |", "|---|---|---|---|---|---|"]
        for tag in tags:
            by_kind = checks["primary_by_kind"][tag]
            lines.append(f"| {labels[tag]} | {_ci(by_kind['key'])} | {_ci(by_kind['severe'])} | {_ci(by_kind['score_cf'])} | {_ci(checks['primary_clear'][tag])} | "
                         f"{_ci(checks['plan_vs_counterfactual'][tag])} |")
        lines += ["", "Robust label bands per kind (low / middle / high): " + "; ".join(
            f"{kind} {b['low']} / {b['middle']} / {b['high']}" for kind, b in bands.items()) + ".", ""]
    lines += ["## Native choice questions (not System-1 metrics)", "",
              "Share of listings whose argmax is in the label's argmax set (single order), per kind of the records' own choice questions.", "",
              "| model | direction (in-record) | comparison (3-way) | outcome |", "|---|---|---|---|"]
    for tag in tags:
        n = r["models"][tag]["native"]
        lines.append(f"| {labels[tag]} | {_f(n['direction']['match'])} | {_f(n['compare']['match'])} | {_f(n['outcome']['match'])} |")
    lines += ["", "## Checks", "",
              f"- Oracle (labels as answers): primary {_f(r['oracle']['primary'])}, direction {_f(r['oracle']['direction'])}, comparison {_f(r['oracle']['comparison'])}, "
              f"severe {_f(r['oracle']['severe_error_robust'])}; items: " + ", ".join(f"{k} {v}" for k, v in r["oracle"]["items"].items()) + ".",
              "- Coverage (answered items / oracle items): " + "; ".join(f"{labels[t]} " + ", ".join(f"{k} {_f(c, 3)}" for k, c in r["coverage"][t].items()) for t in tags) + ".",
              f"- Set: {result['set']['file']} (sha256 {result['set']['sha256'][:16]}…), {result['profile']['records']} records, {result['profile']['groups']} groups; "
              f"overlap with training: {'passed' if result['set']['overlap_passed'] else 'NOT PASSED'}.",
              "- Predictions: " + "; ".join(f"{labels[t]} `{Path(p['predictions']).name}` ({p['model']})" for t, p in result["inputs"].items()) + ".", ""]
    lines += ["## Set composition", "", "| family | records | groups | pairs | questions | robust (all kinds) | yes/no+score | robust yes/no+score | robust label bands low/middle/high |",
              "|---|---|---|---|---|---|---|---|---|"]
    for family, p in result["profile"]["by_family"].items():
        b = p["robust_scalar_label_bands"]
        lines.append(f"| {family} | {p['records']} | {p['groups']} | {p['pairs']} | {p['questions']} | {_f(p['robust_share'])} | {p['scalar_questions']} | "
                     f"{p['robust_scalar']} ({_f(p['robust_scalar_share'])}) | {b.get('low', 0)} / {b.get('middle', 0)} / {b.get('high', 0)} |")
    kinds = sorted({k for p in result["profile"]["by_family"].values() for k in p["robust_by_kind"]})
    lines += ["", "Robust share by question kind:", "", "| family | " + " | ".join(kinds) + " |", "|---|" + "---|" * len(kinds)]
    for family, p in result["profile"]["by_family"].items():
        lines.append(f"| {family} | " + " | ".join(f"{_f(p['robust_by_kind'][k]['share'], 2)} ({p['robust_by_kind'][k]['questions']})" if k in p["robust_by_kind"] else "—" for k in kinds) + " |")
    lines += ["", "## Rule (verbatim)", ""] + [f"- **{k}** — {v}" for k, v in RULE.items()]
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    from robo_jev.decisions.records import file_sha256

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input", type=Path, default=REPO / "artifacts" / "b1" / "b1-infamily.jsonl")
    for tag, label, _, stem in MODELS:
        parser.add_argument(f"--{tag}", type=Path, default=REPORTS / f"{stem}.predictions.jsonl", help=f"{label} predictions")
    parser.add_argument("--out", type=Path, default=REPORTS / "verdict.json")
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    records = _read_jsonl(args.input)
    sha = file_sha256(args.input)
    manifest_path = args.input.with_suffix(".manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else {}
    overlap = manifest.get("overlap_with_train") or {}
    if manifest.get("sha256") != sha:
        raise SystemExit(f"{manifest_path}: missing or its sha256 does not match {args.input}")
    if not (overlap.get("verdict") or {}).get("passed"):
        raise SystemExit(f"{manifest_path}: no passed zero-overlap check with the training records (run mjgen_v03.py train-overlap)")
    inputs, predictions = {}, {}
    for tag, _, _, _ in MODELS:
        path = getattr(args, tag)
        inputs[tag] = check_prediction(path, sha)
        predictions[tag] = _read_jsonl(path)
    _log(f"{len(records)} records; scoring {', '.join(f'{t} ({len(p)})' for t, p in predictions.items())}")
    found = readings(records, predictions, n_boot=args.n_boot, seed=args.seed)
    verdict = decide(found["paired"]["v03-base"]["primary"], found["models"]["v03"]["primary"], found["models"]["v03"]["direction"],
                     found["models"]["v03"]["comparison"])
    result = {
        "kind": "b1-verdict", "rule": RULE, "verdict": verdict, "readings": found, "profile": set_profile(records),
        "set": {"file": str(args.input), "sha256": sha, "manifest": str(manifest_path), "overlap_passed": True, "overlap": overlap,
                "master_seed": manifest.get("master_seed"), "group_indices": manifest.get("group_indices")},
        "inputs": inputs, "models": [{"tag": t, "label": label, "role": role} for t, label, role, _ in MODELS],
        "conditions": CONDITIONS, "n_boot": args.n_boot, "seed": args.seed, "script_version": SCRIPT_VERSION, "git": _git_sha(),
        "created": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=1, default=str) + "\n", encoding="utf-8")
    args.out.with_suffix(".md").write_text(render_markdown(result), encoding="utf-8")
    _log(f"verdict {verdict['outcome']} · positive control {'pass' if verdict['positive_control']['pass'] else 'FAIL'} → {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
