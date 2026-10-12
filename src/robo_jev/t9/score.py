"""T9 instrument sanity scoring (Task T9 §1-4) — System-1 metrics of :mod:`robo_jev.bench.system1` with T8b's definitions, plus the
robust reading, for the references (calculator, constants) and the untrained base.

* Metrics: every metric :func:`robo_jev.bench.system1.set_items` produces on T9 (band accuracy and balanced band accuracy, severe
  errors, direction, comparison, safe / catastrophic choice, outcome, direction choice, and the robust reading
  ``band_balanced_robust``, ``severe_error_robust``, ``direction_robust``), single-order reading; the direction is also split into the
  in-record pairs (planned → counterfactual questions, as T8b) and the one-fact record pairs (as T5). Brier physics skill (bench-v0's
  definition) as a reference.
* Intervals: cluster bootstrap 95 % (2,000 draws) with the **pair group** as the cluster (both records of a one-fact pair share the
  scene), i.e. the record id without its last letter.
* **Format validity**: every question of every record answered, every listing parsed (an answer with a probability / value).
* Per-family values only for the tags named in ``per_family_tags`` (the references): models are reported in aggregate only (pre-declared
  usage of T9).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

from robo_jev.bench import system1 as S1
from robo_jev.t9 import text as T

__all__ = ["MAIN_METRICS", "instrument_report", "render_markdown"]

MAIN_METRICS = ("band_accuracy", "band_balanced", "severe_error", "direction", "comparison", "safe_choice", "catastrophic_choice", "outcome",
                "direction_choice", "band_balanced_robust", "severe_error_robust", "direction_robust")
DETAIL = ("band_accuracy:noul", "band_accuracy:score", "band_recall:low", "band_recall:middle", "band_recall:high", "severe_miss", "severe_false_alarm",
          "band_accuracy_robust", "band_accuracy_clear", "severe_error_clear")


def _group(cluster: str) -> str:
    cluster = cluster[5:] if cluster.startswith("pair:") else cluster
    return cluster[:-1] if cluster.startswith("t9-") and cluster[-1:] in ("a", "b") else cluster


def _regroup(items: dict[tuple[str, ...], tuple[str, float]]) -> dict[tuple[str, ...], tuple[str, float]]:
    return {key: (_group(cluster), value) for key, (cluster, value) in items.items()}


def _read(path: Path) -> list[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _validity(records: Sequence[dict[str, Any]], predictions: Sequence[dict[str, Any]]) -> dict[str, Any]:
    by_id = {str(p["id"]): p for p in predictions}
    total = answered = listings = parsed = 0
    for record in records:
        pred = (by_id.get(str(record["id"])) or {}).get("questions") or {}
        for name, question in record["request"]["questions"].items():
            total += 1
            q = pred.get(name)
            if not isinstance(q, dict):
                continue
            answered += 1
            for listing in q.get("listings") or [q]:
                listings += 1
                answer = listing.get("answer") or {}
                ok = (answer.get("p_yes") is not None if question["type"] == "noul" else answer.get("value") is not None if question["type"] == "score"
                      else isinstance(answer.get("probabilities"), dict) and set(answer["probabilities"]) == set(question["criteria"]))
                parsed += int(bool(ok))
    return {"questions": total, "answered": answered, "listings": listings, "parsed_listings": parsed,
            "valid": total == answered and listings == parsed and total > 0}


def _skill(records: Sequence[dict[str, Any]], predictions: Sequence[dict[str, Any]]) -> float | None:
    from robo_jev.outside.score import pooled_skill, skill_rows

    pooled = pooled_skill(skill_rows(list(records), list(predictions), reading="single_order"))
    return pooled["skill"] if pooled else None


def _split_direction(records: Sequence[dict[str, Any]], predictions: Sequence[dict[str, Any]]) -> dict[str, Any]:
    out = {}
    for name, pairs in (("direction:in_record", S1.t8b_direction_pairs(records)), ("direction:record_pairs", S1.t5_direction_pairs(records))):
        items = S1.order_items("T9", records, predictions, pairs, min_move=S1.DIRECTION_MIN)
        out[name] = _regroup(items)
    return out


def _summaries(records: Sequence[dict[str, Any]], predictions: Sequence[dict[str, Any]], *, n_boot: int, seed: int) -> dict[str, Any]:
    items = {m: _regroup(v) for m, v in S1.set_items("T9", records, predictions, reading="single_order").items()}
    items.update(_split_direction(records, predictions))
    out = {}
    for metric in MAIN_METRICS + ("direction:in_record", "direction:record_pairs") + DETAIL:
        if metric in items:
            out[metric] = S1.summarize(items[metric], n_boot=n_boot if metric in MAIN_METRICS else 0, seed=seed, stratified=metric in S1.STRATIFIED)
    averaged = {m: _regroup(v) for m, v in S1.set_items("T9", records, predictions, reading="order_averaged").items()}
    out["order_averaged"] = {m: S1.summarize(averaged[m], stratified=m in S1.STRATIFIED)["value"] for m in MAIN_METRICS if m in averaged}
    out["physics_skill"] = _skill(records, predictions)
    return out


def instrument_report(records: Sequence[dict[str, Any]], models: Sequence[dict[str, Any]], *, n_boot: int = 2000, seed: int = 0,
                      per_family_tags: set[str] | None = None) -> dict[str, Any]:
    per_family_tags = per_family_tags or set()
    families = sorted({r["meta"]["family"] for r in records}, key=lambda f: list(T.NAMES).index(f))
    oracle = S1.oracle_predictions(records)
    report: dict[str, Any] = {"kind": "t9-instrument", "definitions": S1.DEFINITIONS, "n_boot": n_boot, "seed": seed,
                              "cluster": "pair group (record id without its last letter)", "label_profile": S1.label_profile(records),
                              "oracle": {m: v["value"] for m, v in _summaries(records, oracle, n_boot=0, seed=seed).items()
                                         if isinstance(v, dict) and "value" in v},
                              "models": {}}
    for model in models:
        predictions = _read(Path(model["path"]))
        entry: dict[str, Any] = {"path": str(model["path"]), "validity": _validity(records, predictions),
                                 "overall": _summaries(records, predictions, n_boot=n_boot, seed=seed)}
        groups = {"unseen_law": [r for r in records if r["meta"]["law_group"] == "unseen_law"],
                  "seen_law": [r for r in records if r["meta"]["law_group"] != "unseen_law"]}
        if model["tag"] in per_family_tags:
            entry["by_law_group"] = {g: _summaries(rs, predictions, n_boot=0, seed=seed) for g, rs in groups.items() if rs}
            entry["by_family"] = {f: _summaries([r for r in records if r["meta"]["family"] == f], predictions, n_boot=0, seed=seed) for f in families}
        report["models"][model["tag"]] = entry
    return report


def _f(x: Any) -> str:
    return "—" if x is None else f"{x:.3f}"


def _cell(summary: dict[str, Any] | None) -> str:
    if not summary or summary.get("value") is None:
        return "—"
    text = _f(summary["value"])
    if summary.get("low") is not None:
        text += f" [{_f(summary['low'])}, {_f(summary['high'])}]"
    return text


def render_markdown(report: dict[str, Any]) -> str:
    profile = report["label_profile"]
    lines = ["# T9 — instrument sanity (System-1 metrics)", "",
             f"Single-order reading; cluster bootstrap 95 % ({report['n_boot']} draws), cluster = {report['cluster']}. Labels: {profile['questions']} "
             f"yes/no and score questions — low {_f(profile['low'])}, middle {_f(profile['middle'])}, high {_f(profile['high'])}.", "",
             "| model | valid | Brier skill | " + " | ".join(MAIN_METRICS) + " | direction in-record | direction record pairs |",
             "|---|---|---|" + "---|" * (len(MAIN_METRICS) + 2)]
    for tag, entry in report["models"].items():
        o = entry["overall"]
        lines.append(f"| {tag} | {entry['validity']['valid']} | {_f(o.get('physics_skill'))} | " + " | ".join(_cell(o.get(m)) for m in MAIN_METRICS)
                     + f" | {_cell(o.get('direction:in_record'))} | {_cell(o.get('direction:record_pairs'))} |")
    lines += ["", "Oracle check (labels as answers): " + ", ".join(f"{m} {_f(v)}" for m, v in report["oracle"].items() if m in MAIN_METRICS), ""]
    for tag, entry in report["models"].items():
        if "by_family" not in entry:
            continue
        lines += [f"## {tag} by family and law group (reference only)", "", "| scope | Brier skill | " + " | ".join(MAIN_METRICS) + " |",
                  "|---|---|" + "---|" * len(MAIN_METRICS)]
        for scope, s in list(entry["by_law_group"].items()) + list(entry["by_family"].items()):
            lines.append(f"| {scope} | {_f(s.get('physics_skill'))} | " + " | ".join(_f((s.get(m) or {}).get("value")) for m in MAIN_METRICS) + " |")
        lines.append("")
    return "\n".join(lines) + "\n"
