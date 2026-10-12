"""T8 scoring (Task T8 §1-3) — the same metric code as the diagnosis tool and bench-v0, on the outside sets.

* **T8a (PIQA)** — accuracy (argmax on the correct solution) and choice Brier per model through
  :func:`robo_jev.decisions.single_probe.probe_values` (as G1/G2 in bench-v0); paired differences against the untrained base
  through :func:`robo_jev.decisions.single_probe.compare_probe` (record bootstrap — each record is its own cluster).
* **T8b (MuJoCo moments)** — physics skill = 1 − Σ Brier / Σ Brier(always 0.5) over the yes/no and score questions (the
  bench-v0 definition), from :func:`robo_jev.decisions.metrics.question_rows` (single-order reading: every listing weighted
  1/L; order-averaged: one row per question with the averaged answer). Also the yes/no and score Brier, the accuracy of the
  outcome question (stays / slides / topples), skill per family and for planned vs counterfactual questions, and record
  bootstrap 95 % intervals. Paired differences against the base use the same record draws on the questions both answered.
* **Coverage** — answered questions / questions (OpenRouter models may decline; their scores cover what they answered).
* **Home ground vs outside** — each model's bench-v0 physics skill (T2+T3+T4, pooled) next to its T8 numbers.

Our models' numbers are expectations over balanced option orders (single-order reading, the deployment condition); OpenRouter
models were asked once in the stored order, so their two readings coincide.
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Iterable, Sequence

import numpy as np

from robo_jev.decisions.metrics import READINGS, aggregate, question_rows
from robo_jev.decisions.single_probe import compare_probe, probe_values

__all__ = ["FAMILY_ORDER", "bootstrap_skill", "paired_skill", "piqa_metrics", "piqa_paired", "pooled_skill", "skill_rows", "t8b_metrics"]

FAMILY_ORDER = ("deck_brake", "deck_turn", "stack", "push")


def _interval(samples: np.ndarray) -> tuple[float | None, float | None]:
    samples = samples[np.isfinite(samples)]
    if not samples.size:
        return None, None
    return float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))


# --------------------------------------------------------------------------
# T8a
# --------------------------------------------------------------------------


def _first_option_share(predictions: Sequence[dict[str, Any]]) -> float | None:
    """Share of listings whose argmax is the first listed option (position bias; 0.5 = none on a balanced set)."""
    hits = total = 0.0
    for prediction in predictions:
        for question in prediction["questions"].values():
            listings = question.get("listings") or []
            for listing in listings:
                options = listing.get("options")
                probabilities = (listing.get("answer") or {}).get("probabilities")
                if not options or not probabilities:
                    continue
                best = max(options, key=lambda k: (float(probabilities.get(k, 0.0)), -options.index(k)))
                hits += (1.0 if best == options[0] else 0.0) / len(listings)
                total += 1.0 / len(listings)
    return hits / total if total else None


def piqa_metrics(records: Sequence[dict[str, Any]], predictions: Sequence[dict[str, Any]], *, n_boot: int = 2000, seed: int = 0) -> dict[str, Any]:
    """Accuracy, Brier and coverage per reading, with a record-bootstrap 95 % interval of the accuracy; first-option share."""
    out: dict[str, Any] = {"records": len(records), "answered": len({str(p["id"]) for p in predictions}), "first_option_share": _first_option_share(predictions)}
    rng = np.random.default_rng(seed)
    for reading in READINGS:
        values = probe_values(records, predictions, reading=reading)
        acc = np.array([v["accuracy"] for v in values.values()], dtype=float)
        brier = np.array([v["brier"] for v in values.values()], dtype=float)
        entry: dict[str, Any] = {"questions": len(values), "coverage": len(values) / len(records) if records else None,
                                 "accuracy": float(acc.mean()) if acc.size else None, "brier": float(brier.mean()) if brier.size else None}
        if acc.size:
            draws = rng.integers(0, acc.size, size=(int(n_boot), acc.size))
            entry["accuracy_ci"] = _interval(acc[draws].mean(1))
        out[reading] = entry
    return out


def piqa_paired(records: Sequence[dict[str, Any]], predictions_a: Sequence[dict[str, Any]], predictions_b: Sequence[dict[str, Any]], *,
                n_boot: int = 2000, seed: int = 0) -> dict[str, Any]:
    """a − b on the questions both answered: accuracy and Brier with record-bootstrap 95 % intervals, per reading."""
    comparison = compare_probe(records, predictions_a, predictions_b, dimensions=(), cluster="record", n_boot=n_boot, seed=seed)
    out = {}
    for reading, block in comparison["readings"].items():
        cell = block["table"]["_all"]
        out[reading] = {"questions": cell["questions"], "accuracy": cell.get("accuracy"), "brier": cell.get("brier")}
    return out


# --------------------------------------------------------------------------
# T8b
# --------------------------------------------------------------------------


def skill_rows(records: Sequence[dict[str, Any]], predictions: Sequence[dict[str, Any]], *, reading: str) -> list[dict[str, Any]]:
    """Scored yes/no and score rows: record, question, weight, Brier, Brier(0.5), block, family, planned/counterfactual."""
    by_id = {str(r["id"]): r for r in records}
    out = []
    for row in question_rows(list(records), list(predictions), reading=reading):
        if row.get("m") is None or row["block"] not in ("noul", "score"):
            continue
        meta = by_id[row["record_id"]].get("meta") or {}
        variant = (meta.get("question_variant") or {}).get(row["question"], "planned")
        out.append({"record": row["record_id"], "question": row["question"], "w": float(row["w"]), "brier": float(row["m"]["brier"]),
                    "uniform": float(row["u"]["brier"]), "block": row["block"], "family": str(meta.get("scene")),
                    "variant": "planned" if variant == "planned" else "counterfactual"})
    return out


def pooled_skill(rows: Iterable[dict[str, Any]]) -> dict[str, Any] | None:
    rows = list(rows)
    if not rows:
        return None
    b = sum(r["w"] * r["brier"] for r in rows)
    u = sum(r["w"] * r["uniform"] for r in rows)
    n = len({(r["record"], r["question"]) for r in rows})
    return {"questions": n, "brier": b / sum(r["w"] for r in rows), "uniform_brier": u / sum(r["w"] for r in rows), "skill": 1.0 - b / u if u > 0 else None}


def _record_sums(rows: Sequence[dict[str, Any]], ids: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
    index = {rid: i for i, rid in enumerate(ids)}
    brier, uniform = np.zeros(len(ids)), np.zeros(len(ids))
    for r in rows:
        i = index[r["record"]]
        brier[i] += r["w"] * r["brier"]
        uniform[i] += r["w"] * r["uniform"]
    return brier, uniform


def bootstrap_skill(rows: Sequence[dict[str, Any]], *, n_boot: int = 2000, seed: int = 0) -> tuple[float | None, float | None]:
    """Record-bootstrap 95 % interval of the pooled skill."""
    ids = sorted({r["record"] for r in rows})
    if not ids:
        return None, None
    brier, uniform = _record_sums(rows, ids)
    draws = np.random.default_rng(seed).integers(0, len(ids), size=(int(n_boot), len(ids)))
    with np.errstate(invalid="ignore", divide="ignore"):
        return _interval(1.0 - brier[draws].sum(1) / uniform[draws].sum(1))


def paired_skill(rows_a: Sequence[dict[str, Any]], rows_b: Sequence[dict[str, Any]], *, n_boot: int = 2000, seed: int = 0) -> dict[str, Any] | None:
    """Skill a − skill b on the (record, question) pairs both answered, with a paired record-bootstrap 95 % interval."""
    common = {(r["record"], r["question"]) for r in rows_a} & {(r["record"], r["question"]) for r in rows_b}
    a = [r for r in rows_a if (r["record"], r["question"]) in common]
    b = [r for r in rows_b if (r["record"], r["question"]) in common]
    ids = sorted({r["record"] for r in a})
    if not ids:
        return None
    ba, ua = _record_sums(a, ids)
    bb, ub = _record_sums(b, ids)
    draws = np.random.default_rng(seed).integers(0, len(ids), size=(int(n_boot), len(ids)))
    with np.errstate(invalid="ignore", divide="ignore"):
        samples = (1.0 - ba[draws].sum(1) / ua[draws].sum(1)) - (1.0 - bb[draws].sum(1) / ub[draws].sum(1))
    low, high = _interval(samples)
    value = (1.0 - ba.sum() / ua.sum()) - (1.0 - bb.sum() / ub.sum())
    return {"value": float(value), "low": low, "high": high, "questions": len(common), "records": len(ids),
            "excludes_zero": bool(low is not None and (low > 0 or high < 0))}


def t8b_metrics(records: Sequence[dict[str, Any]], predictions: Sequence[dict[str, Any]], *, n_boot: int = 2000, seed: int = 0) -> dict[str, Any]:
    """Per reading: pooled skill with interval, yes/no and score Brier, outcome-choice accuracy, coverage, skill per family and per
    question kind (planned vs counterfactual), yes/no ECE."""
    total_questions = sum(len(r["request"]["questions"]) for r in records)
    out: dict[str, Any] = {"records": len(records), "questions": total_questions}
    for reading in READINGS:
        rows = question_rows(list(records), list(predictions), reading=reading)
        scored = [r for r in rows if r.get("m") is not None]
        answered = len({(r["record_id"], r["question"]) for r in scored})
        srows = skill_rows(records, predictions, reading=reading)
        pooled = pooled_skill(srows)
        overall = aggregate(scored)
        entry: dict[str, Any] = {
            "coverage": answered / total_questions if total_questions else None, "pooled": pooled,
            "skill_ci": bootstrap_skill(srows, n_boot=n_boot, seed=seed) if srows else (None, None),
            "noul_brier": (overall.get("noul") or {}).get("brier"), "score_brier": (overall.get("score") or {}).get("brier"),
            "noul_ece": (overall.get("noul") or {}).get("ece"),
            "outcome_accuracy": (overall.get("choice/state") or {}).get("accuracy"),
            "outcome_brier": (overall.get("choice/state") or {}).get("brier"),
            "by_family": {f: pooled_skill(r for r in srows if r["family"] == f) for f in FAMILY_ORDER},
            "by_kind": {k: pooled_skill(r for r in srows if r["variant"] == k) for k in ("planned", "counterfactual")},
            "by_block": {b: pooled_skill(r for r in srows if r["block"] == b) for b in ("noul", "score")},
            "by_question": {q: pooled_skill(r for r in srows if r["question"] == q) for q in sorted({r["question"] for r in srows})},
            "mean_answer": {q: _mean_answer(scored, q) for q in sorted({r["question"] for r in srows})},
        }
        out[reading] = entry
    return out


def _mean_answer(rows: Sequence[dict[str, Any]], question: str) -> dict[str, float] | None:
    """Mean model value and mean label of one yes/no or score question name (weighted rows) — shows a yes / danger lean."""
    sel = [r for r in rows if r["question"] == question and r["block"] in ("noul", "score")]
    if not sel:
        return None
    w = sum(r["w"] for r in sel)
    return {"p": sum(r["w"] * r["m"]["p"] for r in sel) / w, "y": sum(r["w"] * r["m"]["y"] for r in sel) / w}


def composition(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    by_family: dict[str, dict[str, Any]] = defaultdict(lambda: {"moments": 0, "questions": 0, "majority": defaultdict(int)})
    for r in records:
        meta = r.get("meta") or {}
        entry = by_family[str(meta.get("scene"))]
        entry["moments"] += 1
        entry["questions"] += len(r["request"]["questions"])
        entry["majority"][str(meta.get("majority"))] += 1
    return {k: {"moments": v["moments"], "questions": v["questions"], "majority": dict(v["majority"])} for k, v in by_family.items()}


def fmt(value: Any, digits: int = 3, signed: bool = False) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "—"
    return f"{value:+.{digits}f}" if signed else f"{value:.{digits}f}"


def fmt_ci(ci: Sequence[Any] | None, digits: int = 3, signed: bool = False) -> str:
    if not ci or ci[0] is None:
        return ""
    return f" [{fmt(ci[0], digits, signed)}, {fmt(ci[1], digits, signed)}]"
