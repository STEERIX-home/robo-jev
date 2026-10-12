#!/usr/bin/env python
"""Task B2 — apply the PRE-DECLARED CAUSE CHECK B2 (progress.md, written 2026-10-13 ~05:20 KST, before any B2 label or result existed) mechanically.

    python scripts/b2_verdict.py --out artifacts/reports/b2/verdict.json

Question: on T8b, is v0.3's level failure confined to questions whose band depends on the hidden-parameter prior (the T8 author's
word→range choices), or does it also fail where the text decides the band?

* **Labels** — T8b's own labeller relabels the same 480 moments (same scenes, facts and hidden-draw points, K = 64) under two
  alternative priors (``scripts/t8.py relabel``, ``robo_jev.outside.moments.PRIOR_SETTINGS``). A yes/no or score question is
  *robust* iff its band is identical under the original prior (the stored labels), alt_a and alt_b
  (:func:`robo_jev.outside.robustness.band_robustness`).
* **Instrument check** — relabelling under the original prior reproduces the stored labels (≥ 99.5 % of questions identical within
  1e-9), and the robust low and high bands each hold ≥ 100 questions; otherwise no interpretation.
* **PRIMARY** — v0.3 − base balanced band accuracy on robust T8b yes/no + score questions (original labels), single order, cluster =
  record, 2,000 draws, seed 0 (System-1 definitions of :mod:`robo_jev.bench.system1`, unchanged; robust is the only added filter). If
  the robust middle band has < 50 questions, the primary is balanced over the low and high bands only and P1's level threshold is 0.65.
* **Rule** — P1 "levels carry where the text decides the band" iff Δ lower bound > +0.10 AND v0.3 ≥ 0.55 (0.65 two-band); P2 "levels
  do not carry even where the text decides the band" iff Δ upper bound < +0.10; else P3 (both).
* **Secondary** (not deciding) — robust share by family and kind, severe errors on robust questions, shared-law families vs push,
  v0.1-mix, the 3×3 confusion on robust questions, v0.3 vs base on prior-dependent questions; plus the all-question reading of the v0.3
  verdict as a reproduction check of the scoring path.

Predictions are the existing ones (no new GPU run): base and v0.1-mix scored on Spark (T8 task), v0.3 on its H100 pod. Each file's
diagnosis summary must record T8b's sha256 and the T8 scoring conditions. Writes ``verdict.json`` and ``verdict.md``.
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

SCRIPT_VERSION = "b2-verdict-1"
RULE = {
    "written": "2026-10-13 ~05:20 KST, before any B2 label or result existed (progress.md: PRE-DECLARED CAUSE CHECK B2)",
    "instrument": "relabelling under the original prior reproduces the stored T8b labels (>= 99.5 % of questions identical within 1e-9), and "
                  "the robust low and high bands each hold >= 100 questions; otherwise no interpretation",
    "robust": "band identical under the original prior, alt_a and alt_b",
    "primary": "v0.3 - base balanced band accuracy on robust T8b yes/no+score questions (original labels), single order, cluster = record, "
               "2000 draws, seed 0; if the robust middle band has < 50 questions, balanced over low and high only and P1's level threshold is 0.65",
    "P1": "'levels carry where the text decides the band' iff delta lower bound > +0.10 AND v0.3 >= 0.55 (0.65 two-band) -> T8b level failure "
          "sits in prior-dependent questions -> data priority: word->range knowledge and hedged / 'look again' labels on prior-dependent moments",
    "P2": "'levels do not carry even where the text decides the band' iff delta upper bound < +0.10 -> genuine transfer failure -> data "
          "priority: level supervision across many scenes and wordings",
    "P3": "else (both)",
}
P1_LOW = 0.10
P1_MIN = 0.55
P1_MIN_TWO_BAND = 0.65
P2_HIGH = 0.10
REPRODUCE_MIN = 0.995
BAND_MIN = 100
MIDDLE_MIN = 50
MEANING = {
    "P1": "levels carry where the text decides the band: the T8b level failure sits in prior-dependent questions -> data priority: "
          "word->range knowledge and hedged / 'look again' labels on prior-dependent moments",
    "P2": "levels do not carry even where the text decides the band: a genuine transfer failure -> data priority: level supervision across "
          "many scenes and wordings",
    "P3": "both: levels carry only partly where the text decides the band -> both data priorities",
}
CONDITIONS = {"mode": "direct", "format": "chat", "prompt_version": "v0c-p3", "orders": "counterbalanced", "shuffle_seed": 1}
SET = "T8b"
T8B = REPO / "artifacts" / "t8" / "T8b.jsonl"
LABELS = REPO / "artifacts" / "b2"
REPORTS = REPO / "artifacts" / "reports" / "b2"
#: (tag, label, role, default predictions) in report order; exactly one baseline.
MODELS = (
    ("base", "untrained Qwen3.5-2B (Spark)", "baseline", REPO / "artifacts" / "reports" / "t8" / "T8b-base-2b.predictions.jsonl"),
    ("v01mix", "v0.1-mix (Spark)", "ours", REPO / "artifacts" / "reports" / "t8" / "T8b-v01mix-2b.predictions.jsonl"),
    ("v03", "v0.3 (H100 pod)", "ours", REPO / "artifacts" / "cloud" / "runs" / "v03-01-fetched" / "reports" / "after-v03-2b-T8b.predictions.jsonl"),
)
FAMILIES = ("deck_brake", "deck_turn", "stack", "push")
#: The law split of the v0.3 verdict: the MuJoCo training families share their laws with deck and stack; push is the unseen law.
GROUPS = {"shared_law (deck_brake, deck_turn, stack)": ("deck_brake", "deck_turn", "stack"), "push (unseen law)": ("push",)}
PRIORS = ("original", "alt_a", "alt_b")


def _log(message: str) -> None:
    print(f"[b2-verdict {dt.datetime.now().strftime('%H:%M:%S')}] {message}", file=sys.stderr, flush=True)


def _git_sha() -> str | None:
    try:
        return subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True).stdout.strip() or None
    except (OSError, subprocess.CalledProcessError):
        return None


# --------------------------------------------------------------------------
# The rule (pure)
# --------------------------------------------------------------------------


def instrument_check(reproduced_share: float | None, robust_low: int, robust_high: int) -> dict[str, Any]:
    checks = {"reproduces_stored_labels_ge_0.995": reproduced_share is not None and reproduced_share >= REPRODUCE_MIN,
              "robust_low_ge_100": robust_low >= BAND_MIN, "robust_high_ge_100": robust_high >= BAND_MIN}
    return {"reproduced_share": reproduced_share, "robust_low": robust_low, "robust_high": robust_high, "checks": checks, "pass": all(checks.values())}


def rule_outcome(delta_low: float | None, delta_high: float | None, v03_value: float | None, *, two_band: bool) -> str:
    """P1 iff Δ lower bound > +0.10 AND v0.3 ≥ 0.55 (0.65 two-band); P2 iff Δ upper bound < +0.10; else P3. Raises when a reading is
    missing (the rule cannot be computed as written)."""
    if delta_low is None or delta_high is None or v03_value is None:
        raise ValueError("the rule cannot be computed: a reading is missing (delta interval or v0.3's primary value)")
    if delta_low > P1_LOW and v03_value >= (P1_MIN_TWO_BAND if two_band else P1_MIN):
        return "P1"
    if delta_high < P2_HIGH:
        return "P2"
    return "P3"


def decide(instrument: Mapping[str, Any], delta: Mapping[str, Any], v03_primary: Mapping[str, Any], *, two_band: bool) -> dict[str, Any]:
    outcome = rule_outcome(delta.get("low"), delta.get("high"), v03_primary.get("value"), two_band=two_band)
    threshold = P1_MIN_TWO_BAND if two_band else P1_MIN
    checks = {"P1": {"delta_low_gt_0.10": delta["low"] > P1_LOW, f"v03_ge_{threshold:.2f}": v03_primary["value"] >= threshold},
              "P2": {"delta_high_lt_0.10": delta["high"] < P2_HIGH}}
    return {"outcome": outcome, "two_band": two_band, "level_threshold": threshold, "checks": checks, "instrument": instrument,
            "interpreted": bool(instrument["pass"]),
            "meaning": MEANING[outcome] if instrument["pass"] else "instrument check failed: no interpretation is made"}


# --------------------------------------------------------------------------
# Readings
# --------------------------------------------------------------------------


def _summary(items: Mapping[Any, Any], *, n_boot: int, seed: int, stratified: bool = False) -> dict[str, Any]:
    from robo_jev.bench import system1

    return system1.summarize(dict(items), n_boot=n_boot, seed=seed, stratified=stratified)


def _paired(a: Mapping[Any, Any], b: Mapping[Any, Any], *, n_boot: int, seed: int, stratified: bool = False) -> dict[str, Any] | None:
    from robo_jev.bench import system1

    return system1.paired_difference(dict(a), dict(b), n_boot=n_boot, seed=seed, stratified=stratified)


def primary_items(band_items: Mapping[Any, Any], robust: set[tuple[str, str]], *, two_band: bool) -> dict[Any, Any]:
    """The primary's items: band items (keys ``(label band, set, record, question)``) of robust questions; two-band drops the middle
    stratum (balanced over low and high only)."""
    from robo_jev.bench import infamily

    kept = infamily.keep(dict(band_items), robust)
    return {k: v for k, v in kept.items() if not (two_band and k[0] == "middle")}


def _family_items(items: Mapping[Any, Any], families_of: Mapping[str, str], members: Sequence[str]) -> dict[Any, Any]:
    return {k: v for k, v in items.items() if families_of[k[-2]] in members}


def readings(records: Sequence[dict[str, Any]], predictions: Mapping[str, Sequence[dict[str, Any]]], robust: set[tuple[str, str]],
             ambiguous: set[tuple[str, str]], *, two_band: bool, n_boot: int = 2000, seed: int = 0) -> dict[str, Any]:
    """Every reading for the models in `predictions` (tags of :data:`MODELS`); `records` carry ``meta.robustness``."""
    from robo_jev.bench import infamily, system1

    families_of = {str(r["id"]): str(r["meta"]["scene"]) for r in records}
    items = {tag: system1.question_items(SET, records, preds) for tag, preds in predictions.items()}
    oracle = system1.question_items(SET, records, system1.oracle_predictions(records))
    prim = {tag: primary_items(found.get("band_balanced") or {}, robust, two_band=two_band) for tag, found in items.items()}
    out: dict[str, Any] = {"models": {}, "paired": {}, "groups": {}, "per_family": {}}
    for tag, found in items.items():
        robust_items = {m: infamily.keep(found.get(m) or {}, robust) for m in ("band_accuracy", "severe_error", "severe_miss", "severe_false_alarm",
                                                                                "band_recall:low", "band_recall:middle", "band_recall:high")}
        out["models"][tag] = {
            "primary": _summary(prim[tag], n_boot=n_boot, seed=seed, stratified=True),
            "robust": {m: _summary(v, n_boot=n_boot if m == "severe_error" else 0, seed=seed) for m, v in robust_items.items()},
            "robust_band_balanced_three": _summary(infamily.keep(found.get("band_balanced") or {}, robust), n_boot=n_boot, seed=seed, stratified=True),
            "ambiguous": {"band_balanced": _summary(infamily.keep(found.get("band_balanced") or {}, ambiguous), n_boot=n_boot, seed=seed, stratified=True),
                          "severe_error": _summary(infamily.keep(found.get("severe_error") or {}, ambiguous), n_boot=0, seed=seed),
                          "answers": infamily.answer_profile(records, predictions[tag], ambiguous)},
            "all": {"band_balanced": _summary(found.get("band_balanced") or {}, n_boot=0, seed=seed, stratified=True),
                    "severe_error": _summary(found.get("severe_error") or {}, n_boot=0, seed=seed)},
            "confusion_robust": infamily.band_confusion(records, predictions[tag], robust),
            "answers_robust": infamily.answer_profile(records, predictions[tag], robust),
            "coverage": len(found.get("band_accuracy") or {}) / max(1, len(oracle.get("band_accuracy") or {})),
        }
    pairs = [(a, "base") for a in ("v03", "v01mix") if a in items and "base" in items] + ([("v03", "v01mix")] if {"v03", "v01mix"} <= set(items) else [])
    for a, b in pairs:
        ia, ib = items[a], items[b]
        out["paired"][f"{a}-{b}"] = {
            "primary": _paired(prim[a], prim[b], n_boot=n_boot, seed=seed, stratified=True),
            "severe_error_robust": _paired(infamily.keep(ia.get("severe_error") or {}, robust), infamily.keep(ib.get("severe_error") or {}, robust),
                                           n_boot=n_boot, seed=seed),
            "band_balanced_ambiguous": _paired(infamily.keep(ia.get("band_balanced") or {}, ambiguous), infamily.keep(ib.get("band_balanced") or {}, ambiguous),
                                               n_boot=n_boot, seed=seed, stratified=True),
            "band_balanced_all": _paired(ia.get("band_balanced") or {}, ib.get("band_balanced") or {}, n_boot=n_boot, seed=seed, stratified=True),
        }
    for group, members in GROUPS.items():
        entry: dict[str, Any] = {"models": {}, "paired": {}}
        for tag in items:
            entry["models"][tag] = {
                "primary": _summary(_family_items(prim[tag], families_of, members), n_boot=n_boot, seed=seed, stratified=True),
                "severe_error_robust": _summary(_family_items(infamily.keep(items[tag].get("severe_error") or {}, robust), families_of, members), n_boot=0, seed=seed)}
        for a, b in pairs:
            entry["paired"][f"{a}-{b}"] = _paired(_family_items(prim[a], families_of, members), _family_items(prim[b], families_of, members),
                                                  n_boot=n_boot, seed=seed, stratified=True)
        entry["robust_questions"] = sum(1 for rid, _ in robust if families_of[rid] in members)
        entry["primary_questions"] = len(_family_items(prim[next(iter(items))], families_of, members)) if items else 0
        out["groups"][group] = entry
    for family in FAMILIES:
        out["per_family"][family] = {tag: _summary(_family_items(prim[tag], families_of, (family,)), n_boot=0, seed=seed, stratified=True)
                                     for tag in items}
    out["oracle"] = {"primary": _summary(primary_items(oracle["band_balanced"], robust, two_band=two_band), n_boot=0, seed=seed, stratified=True)["value"],
                     "severe_error_robust": _summary(infamily.keep(oracle["severe_error"], robust), n_boot=0, seed=seed)["value"]}
    return out


def sensitivity(records: Sequence[dict[str, Any]], predictions: Mapping[str, Sequence[dict[str, Any]]], robust: set[tuple[str, str]], *,
                two_band: bool, flagged: set[str], n_boot: int = 2000, seed: int = 0) -> dict[str, Any]:
    """Not deciding: the primary and the rule under the other band choice (two-band ↔ three-band), and with the questions of moments that
    had a rollout flagged by the instability guard under some prior (`flagged` record ids) left out."""
    from robo_jev.bench import system1

    items = {tag: system1.question_items(SET, records, preds).get("band_balanced") or {} for tag, preds in predictions.items()}
    out: dict[str, Any] = {}
    for name, keys, bands in (("other_band_choice", robust, not two_band),
                              ("without_flagged_moments", {k for k in robust if k[0] not in flagged}, two_band)):
        prim = {tag: primary_items(found, keys, two_band=bands) for tag, found in items.items()}
        delta = _paired(prim["v03"], prim["base"], n_boot=n_boot, seed=seed, stratified=True)
        values = {tag: _summary(found, n_boot=n_boot, seed=seed, stratified=True) for tag, found in prim.items()}
        out[name] = {"two_band": bands, "questions": len(prim["v03"]), "models": values, "delta_v03_base": delta,
                     "outcome_if_applied": rule_outcome(delta["low"], delta["high"], values["v03"]["value"], two_band=bands)}
    out["without_flagged_moments"]["flagged_moments"] = sorted(flagged)
    return out


def composition(records: Sequence[dict[str, Any]], robustness: Mapping[tuple[str, str], Mapping[str, Any]]) -> dict[str, Any]:
    """Robust share by family and by kind (T8b's question names, :data:`robo_jev.outside.robustness.KINDS`), and the robust label bands."""
    from collections import Counter

    from robo_jev.bench.system1 import band
    from robo_jev.outside.robustness import KINDS

    out: dict[str, Any] = {"by_family": {}, "by_kind": {}, "by_question": {}}
    totals: dict[str, Counter] = {"kind": Counter(), "kind_robust": Counter(), "q": Counter(), "q_robust": Counter()}
    for family in FAMILIES + ("all",):
        rows = [r for r in records if family == "all" or r["meta"]["scene"] == family]
        scalar = [(str(r["id"]), q, r["labels"][q]) for r in rows for q in r["request"]["questions"] if r["labels"][q]["type"] in ("noul", "score")]
        robust = [(rid, q, l) for rid, q, l in scalar if robustness[(rid, q)]["robust"]]
        choices = [(str(r["id"]), q) for r in rows for q in r["request"]["questions"] if r["labels"][q]["type"] == "choice"]
        bands = Counter(band(float(l.get("p_yes", l.get("value")))) for _, _, l in robust)
        out["by_family"][family] = {"moments": len(rows), "scalar_questions": len(scalar), "robust": len(robust),
                                    "robust_share": len(robust) / len(scalar) if scalar else None,
                                    "robust_bands": {b: bands.get(b, 0) for b in ("low", "middle", "high")},
                                    "all_bands": dict(Counter(band(float(l.get("p_yes", l.get("value")))) for _, _, l in scalar)),
                                    "choice_questions": len(choices), "choice_argmax_robust": sum(1 for key in choices if robustness[key]["robust"])}
        if family == "all":
            continue
        for rid, q, _ in scalar:
            totals["kind"][(family, KINDS[q])] += 1
            totals["kind_robust"][(family, KINDS[q])] += int(robustness[(rid, q)]["robust"])
            totals["q"][q] += 1
            totals["q_robust"][q] += int(robustness[(rid, q)]["robust"])
    for (family, kind), n in sorted(totals["kind"].items()):
        out["by_kind"].setdefault(family, {})[kind] = {"questions": n, "robust": totals["kind_robust"][(family, kind)],
                                                       "share": totals["kind_robust"][(family, kind)] / n}
    kinds = Counter()
    kinds_robust = Counter()
    for (family, kind), n in totals["kind"].items():
        kinds[kind] += n
        kinds_robust[kind] += totals["kind_robust"][(family, kind)]
    out["by_kind"]["all"] = {kind: {"questions": n, "robust": kinds_robust[kind], "share": kinds_robust[kind] / n} for kind, n in sorted(kinds.items())}
    out["by_question"] = {q: {"questions": n, "robust": totals["q_robust"][q], "share": totals["q_robust"][q] / n} for q, n in sorted(totals["q"].items())}
    return out


def shifts(records: Sequence[dict[str, Any]], robustness: Mapping[tuple[str, str], Mapping[str, Any]]) -> dict[str, Any]:
    """How the alternative priors move the yes/no and score labels: the mean value per prior and family, and where non-robust bands go."""
    from collections import Counter

    out: dict[str, Any] = {}
    for family in FAMILIES:
        keys = [(str(r["id"]), q) for r in records if r["meta"]["scene"] == family for q in r["request"]["questions"]
                if r["labels"][q]["type"] in ("noul", "score")]
        if not keys:
            continue
        out[family] = {"mean_by_prior": {p: sum(robustness[k]["values"][p] for k in keys) / len(keys) for p in PRIORS},
                       "band_patterns_non_robust": dict(Counter("/".join(robustness[k]["bands"][p] for p in PRIORS) for k in keys
                                                                if not robustness[k]["robust"]).most_common(6))}
    return out


# --------------------------------------------------------------------------
# Inputs and checks
# --------------------------------------------------------------------------


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    from robo_jev.decisions.records import read_records

    return read_records(path)


def check_prediction(path: Path, input_sha256: str) -> dict[str, Any]:
    """The diagnosis summary next to a prediction file: the T8b file it scored (sha256) and the T8 scoring conditions."""
    summary_path = path.with_name(path.name.replace(".predictions.jsonl", ".json"))
    if not summary_path.is_file():
        raise SystemExit(f"{path}: no diagnosis summary {summary_path.name} next to it")
    config = json.loads(summary_path.read_text(encoding="utf-8"))["config"]
    problems = []
    if config.get("input_sha256") != input_sha256:
        problems.append(f"input sha256 {str(config.get('input_sha256'))[:12]} != T8b {input_sha256[:12]}")
    for key, want in CONDITIONS.items():
        if config.get(key) != want:
            problems.append(f"{key} = {config.get(key)!r}, want {want!r}")
    if problems:
        raise SystemExit(f"{path}: " + "; ".join(problems))
    checkpoint = config.get("checkpoint") or {}
    return {"predictions": str(path), "summary": str(summary_path), "model": config.get("model"), "records": config.get("records"),
            "conditions": {k: config.get(k) for k in CONDITIONS}, "checkpoint_sha256": checkpoint.get("sha256"), "checkpoint_step": checkpoint.get("step"),
            "created": config.get("created")}


def load_labels(directory: Path, input_sha256: str) -> tuple[dict[str, dict[str, dict[str, Any]]], dict[str, Any]]:
    """The relabelled rows per prior and their manifest; refuses rows made from another T8b file or whose scenes, requests or draw seeds
    did not match the stored moments."""
    from robo_jev.decisions.records import file_sha256

    manifest_path = directory / "T8b-labels.manifest.json"
    if not manifest_path.is_file():
        raise SystemExit(f"{manifest_path}: missing — run scripts/t8.py relabel first")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("input_sha256") != input_sha256:
        raise SystemExit(f"{manifest_path}: made from another T8b file ({str(manifest.get('input_sha256'))[:12]})")
    rows: dict[str, dict[str, dict[str, Any]]] = {}
    for prior in PRIORS:
        entry = (manifest.get("files") or {}).get(prior)
        if entry is None:
            raise SystemExit(f"{manifest_path}: no {prior} labels")
        path = Path(entry["file"]) if Path(entry["file"]).is_absolute() else REPO / entry["file"]
        if file_sha256(path) != entry["sha256"]:
            raise SystemExit(f"{path}: sha256 differs from its manifest")
        found = {str(r["id"]): r for r in _read_jsonl(path)}
        bad = [rid for rid, r in found.items() if not (r["scene_matches_generator"] and r["request_matches"] and r["draws_seed_matches"])]
        if bad:
            raise SystemExit(f"{path}: {len(bad)} rows whose scene, request or draw seed does not match the stored moment (e.g. {bad[:3]})")
        rows[prior] = found
    return rows, manifest


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
    r, v, comp = result["readings"], result["verdict"], result["composition"]
    ins = v["instrument"]
    band_note = "low and high only (two-band: the robust middle band has < 50 questions)" if v["two_band"] else "low, middle and high"
    lines = ["# B2 — on T8b, does v0.3's level failure sit only where the prior decides the band? (pre-declared cause check)", "",
             f"Rule written {RULE['written']}. Applied mechanically by `scripts/b2_verdict.py` ({SCRIPT_VERSION}, git {result['git']}). Single-order "
             "reading; cluster bootstrap 95 % (2,000 draws, seed 0, cluster = record). Existing predictions (base and v0.1-mix on Spark, v0.3 on its "
             "H100 pod).", ""]
    lines += ["## Verdict", "",
              f"- **Instrument check: {'PASS' if ins['pass'] else 'FAIL'}** — original-prior relabelling reproduces {_f(ins['reproduced_share'], 4)} of the "
              f"stored labels (≥ 0.995); robust low {ins['robust_low']}, robust high {ins['robust_high']} (each ≥ 100).",
              f"- **Outcome: {v['outcome']}** — " + "; ".join(f"{k} {'yes' if ok else 'no'}" for part in v["checks"].values() for k, ok in part.items())
              + f" (primary balanced over {band_note}; level threshold {v['level_threshold']:.2f}).",
              f"- **Reading:** {v['meaning']}.", ""]
    lines += ["## Primary — balanced band accuracy on robust yes/no + score questions (original labels)", "",
              f"Balanced over {band_note}.", "",
              "| model | primary [95 %] | recall low | recall middle | recall high | n | clusters |", "|---|---|---|---|---|---|---|"]
    for tag in tags:
        m = r["models"][tag]
        rec = m["robust"]
        lines.append(f"| {labels[tag]} | {_ci(m['primary'])} | {_f(rec['band_recall:low']['value'])} | {_f(rec['band_recall:middle']['value'])} | "
                     f"{_f(rec['band_recall:high']['value'])} | {m['primary']['n']} | {m['primary']['clusters']} |")
    lines += ["", "| paired Δ | primary [95 %] | severe, robust | balanced band, prior-dependent | balanced band, all (v0.3 verdict reading) |",
              "|---|---|---|---|---|"]
    for key, entry in r["paired"].items():
        a, b = key.split("-")
        lines.append(f"| {labels[a]} − {labels[b]} | {_ci(entry['primary'], signed=True)} | {_ci(entry['severe_error_robust'], signed=True)} | "
                     f"{_ci(entry['band_balanced_ambiguous'], signed=True)} | {_ci(entry['band_balanced_all'], signed=True)} |")
    primary = r["paired"]["v03-base"]["primary"]
    lines += ["", f"Rule inputs: Δ(v0.3 − base) = {_ci(primary, signed=True)}; v0.3 = {_f(r['models']['v03']['primary']['value'])}.", ""]
    sens = r.get("sensitivity") or {}
    if sens:
        lines += ["## Sensitivity (not deciding)", "",
                  "The primary and the rule's outcome computed another way: with the other band choice, and without the moments that had a rollout "
                  "flagged by the instability guard under some prior.", "",
                  "| reading | bands | questions | base | v0.1-mix | v0.3 | Δ v0.3 − base [95 %] | outcome if it decided |", "|---|---|---|---|---|---|---|---|"]
        for name, entry in sens.items():
            m = entry["models"]
            lines.append(f"| {name.replace('_', ' ')} | {'low + high' if entry['two_band'] else 'three'} | {entry['questions']} | {_f(m['base']['value'])} | "
                         f"{_f(m['v01mix']['value'])} | {_f(m['v03']['value'])} | {_ci(entry['delta_v03_base'], signed=True)} | {entry['outcome_if_applied']} |")
        flagged = sens.get("without_flagged_moments", {}).get("flagged_moments") or []
        lines += ["", f"Flagged moments left out: {len(flagged)}.", ""]
    lines += ["## Severe errors and other readings", "",
              "| model | severe, robust [95 %] | miss | false alarm | balanced band (3 bands), robust | balanced band, prior-dependent [95 %] | "
              "balanced band, all | severe, all | coverage |", "|---|---|---|---|---|---|---|---|---|"]
    for tag in tags:
        m = r["models"][tag]
        lines.append(f"| {labels[tag]} | {_ci(m['robust']['severe_error'])} | {_f(m['robust']['severe_miss']['value'])} | {_f(m['robust']['severe_false_alarm']['value'])} | "
                     f"{_f(m['robust_band_balanced_three']['value'])} | {_ci(m['ambiguous']['band_balanced'])} | {_f(m['all']['band_balanced']['value'])} | "
                     f"{_f(m['all']['severe_error']['value'])} | {_f(m['coverage'])} |")
    lines += ["", "## Shared-law families vs push", "", "| group | robust questions (primary) | model | primary [95 %] | severe, robust |", "|---|---|---|---|---|"]
    for group, entry in r["groups"].items():
        for i, tag in enumerate(tags):
            head = f"{group} | {entry['robust_questions']} ({entry['primary_questions']})" if i == 0 else " | "
            lines.append(f"| {head} | {labels[tag]} | {_ci(entry['models'][tag]['primary'])} | {_f(entry['models'][tag]['severe_error_robust']['value'])} |")
        for key, paired in entry["paired"].items():
            a, b = key.split("-")
            lines.append(f"|  |  | Δ {labels[a]} − {labels[b]} | {_ci(paired, signed=True)} |  |")
    lines += ["", "Per family (primary, points): " + "; ".join(
        f"{family} " + ", ".join(f"{labels[t].split(' (')[0]} {_f(s['value'])}" for t, s in models.items()) for family, models in r["per_family"].items()) + ".", ""]
    lines += ["## Answers on prior-dependent questions (band depends on the prior)", "",
              "| model | questions | mean | SD | low / middle / high | hedge share | answer band among the prior bands | inside the prior range | "
              "robust: hedge share |", "|---|---|---|---|---|---|---|---|---|"]
    for tag in tags:
        a = r["models"][tag]["ambiguous"]["answers"]
        b = r["models"][tag]["answers_robust"]
        if not a.get("questions"):
            continue
        s = a["band_shares"]
        lines.append(f"| {labels[tag]} | {a['questions']:.0f} | {_f(a['mean'])} | {_f(a['sd'])} | {_f(s['low'], 2)} / {_f(s['middle'], 2)} / {_f(s['high'], 2)} | "
                     f"{_f(a['hedge_share'])} | {_f(a['in_prior_bands'])} | {_f(a['in_prior_range'])} | {_f(b.get('hedge_share'))} |")
    lines += ["", "## 3×3 band confusion on robust questions (rows = label band, shares of the row)", ""]
    for tag in tags:
        conf = r["models"][tag]["confusion_robust"]
        lines += [f"**{labels[tag]}**", "", "| label \\ answer | low | middle | high | questions |", "|---|---|---|---|---|"]
        for row in ("low", "middle", "high"):
            shares = conf["rows"][row] or {}
            lines.append(f"| {row} | {_f(shares.get('low'), 2)} | {_f(shares.get('middle'), 2)} | {_f(shares.get('high'), 2)} | {sum(conf['counts'][row].values()):.0f} |")
        lines.append("")
    lines += ["## Robust composition", "", "| family | moments | yes/no + score | robust | share | robust low / middle / high | outcome choice: argmax robust |",
              "|---|---|---|---|---|---|---|"]
    for family, p in comp["by_family"].items():
        b = p["robust_bands"]
        choice = f"{p['choice_argmax_robust']} / {p['choice_questions']}" if p["choice_questions"] else "—"
        lines.append(f"| {family} | {p['moments']} | {p['scalar_questions']} | {p['robust']} | {_f(p['robust_share'])} | {b['low']} / {b['middle']} / {b['high']} | {choice} |")
    kinds = sorted({k for fam, d in comp["by_kind"].items() if fam != "all" for k in d})
    lines += ["", "Robust share by kind:", "", "| family | " + " | ".join(kinds) + " |", "|---|" + "---|" * len(kinds)]
    for family, d in comp["by_kind"].items():
        lines.append(f"| {family} | " + " | ".join(f"{_f(d[k]['share'], 2)} ({d[k]['questions']})" if k in d else "—" for k in kinds) + " |")
    lines += ["", "How the priors move the labels (mean yes/no + score label per prior; the most common band patterns original/alt_a/alt_b of "
              "prior-dependent questions):", ""]
    for family, s in result["shifts"].items():
        lines.append(f"- {family}: mean " + " / ".join(f"{p} {_f(m)}" for p, m in s["mean_by_prior"].items()) + "; " +
                     ", ".join(f"{pattern} {n}" for pattern, n in s["band_patterns_non_robust"].items()))
    rep = result["instrument"]["reproduction"]
    lines += ["", "## Checks", "",
              f"- Reproduction (original prior vs the stored labels): {rep['identical']} / {rep['questions']} identical within 1e-9 "
              "(" + ", ".join(f"{t} {c['identical']}/{c['questions']}" for t, c in rep["by_type"].items()) + ").",
              "- Relabelled rows: " + "; ".join(f"{p} — scenes = generator {f['scene_matches_generator']}/{f['rows']}, requests {f['request_matches']}/{f['rows']}, "
                                                f"draw seeds {f['draws_seed_matches']}/{f['rows']}, moments with an unstable rollout {f['moments_with_unstable_rollouts']}, "
                                                f"off deck {f['moments_off_deck']}" for p, f in result["labels"]["files"].items()) + ".",
              f"- Oracle (labels as answers): primary {_f(r['oracle']['primary'])}, severe {_f(r['oracle']['severe_error_robust'])}.",
              "- Predictions: " + "; ".join(f"{labels[t]} `{Path(p['predictions']).name}` ({p['model']})" for t, p in result["inputs"].items()) + ".",
              f"- T8b: {result['set']['file']} (sha256 {result['set']['sha256'][:16]}…), never modified.", ""]
    lines += ["## Rule (verbatim)", ""] + [f"- **{k}** — {v_}" for k, v_ in RULE.items()]
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    from robo_jev.bench import infamily
    from robo_jev.decisions.records import file_sha256
    from robo_jev.outside.robustness import band_robustness, reproduction, with_robustness

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input", type=Path, default=T8B)
    parser.add_argument("--labels", type=Path, default=LABELS, help="directory with T8b-labels-<prior>.jsonl and T8b-labels.manifest.json")
    for tag, label, _, path in MODELS:
        parser.add_argument(f"--{tag}", type=Path, default=path, help=f"{label} predictions")
    parser.add_argument("--out", type=Path, default=REPORTS / "verdict.json")
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    stored = _read_jsonl(args.input)
    sha = file_sha256(args.input)
    rows, label_manifest = load_labels(args.labels, sha)
    repro = reproduction(stored, rows["original"])
    robustness = band_robustness(stored, {p: rows[p] for p in PRIORS if p != "original"})
    records = with_robustness(stored, robustness)
    robust, ambiguous = infamily.robust_keys(records), infamily.ambiguous_keys(records)
    bands = {b: sum(1 for key in robust if robustness[key]["bands"]["original"] == b) for b in ("low", "middle", "high")}
    two_band = bands["middle"] < MIDDLE_MIN
    instrument = instrument_check(repro["share"], bands["low"], bands["high"])
    inputs, predictions = {}, {}
    for tag, _, _, _ in MODELS:
        path = getattr(args, tag)
        inputs[tag] = check_prediction(path, sha)
        predictions[tag] = _read_jsonl(path)
    _log(f"{len(stored)} moments; robust {len(robust)} (low {bands['low']} · middle {bands['middle']} · high {bands['high']}); "
         f"reproduction {repro['share']:.4f}; two-band {two_band}")
    found = readings(records, predictions, robust, ambiguous, two_band=two_band, n_boot=args.n_boot, seed=args.seed)
    verdict = decide(instrument, found["paired"]["v03-base"]["primary"], found["models"]["v03"]["primary"], two_band=two_band)
    flagged = {rid for prior in PRIORS for rid, row in rows[prior].items() if row.get("unstable") or row.get("off_deck")}
    found["sensitivity"] = sensitivity(records, predictions, robust, two_band=two_band, flagged=flagged, n_boot=args.n_boot, seed=args.seed)
    result = {
        "kind": "b2-verdict", "rule": RULE, "verdict": verdict, "readings": found, "composition": composition(stored, robustness),
        "shifts": shifts(stored, robustness), "instrument": {"reproduction": repro, "robust_bands": bands, "two_band": two_band},
        "labels": {"manifest": str(args.labels / "T8b-labels.manifest.json"), "priors": label_manifest["priors"], "files": label_manifest["files"],
                   "git": label_manifest.get("git"), "mujoco": label_manifest.get("mujoco")},
        "set": {"file": str(args.input), "sha256": sha}, "inputs": inputs,
        "models": [{"tag": t, "label": label, "role": role} for t, label, role, _ in MODELS], "conditions": CONDITIONS, "n_boot": args.n_boot,
        "seed": args.seed, "script_version": SCRIPT_VERSION, "git": _git_sha(), "created": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=1, default=str) + "\n", encoding="utf-8")
    args.out.with_suffix(".md").write_text(render_markdown(result), encoding="utf-8")
    _log(f"instrument {'pass' if instrument['pass'] else 'FAIL'} · outcome {verdict['outcome']} → {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
