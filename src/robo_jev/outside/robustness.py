"""Task B2 — prior robustness of T8b's labels: the stored labels (T8b's own priors) next to the labels of the same moments relabelled
under alternative priors (:data:`robo_jev.outside.moments.PRIOR_SETTINGS`, ``scripts/t8.py relabel``).

* :func:`reproduction` — the instrument check's first half: relabelling under the original prior must reproduce the stored labels
  (a question is identical when every number of its label is within 1e-9: ``p_yes``, ``value`` or each choice probability).
* :func:`band_robustness` — per question, the label under each prior setting. Yes/no and score: the value, its risk band (low < 1/3 ≤
  middle ≤ 2/3 < high, :func:`robo_jev.bench.system1.band`), the min and max and ``robust`` = the band is identical under every
  setting. Choice: the argmax per setting and ``robust`` = the argmax is the same. The original setting is read from the stored
  records (the labels every model is scored against), the others from the relabelled rows.
* :func:`with_robustness` — copies of the records with ``meta.robustness`` set (the format of the v0.3b generator), so the in-family
  adapters (:mod:`robo_jev.bench.infamily`: robust keys, band confusion, answer profiles) read T8b the same way. In memory only.
"""

from __future__ import annotations

import copy
from typing import Any, Mapping, Sequence

from robo_jev.bench.system1 import band

__all__ = ["KINDS", "ORIGINAL", "band_robustness", "label_numbers", "reproduction", "with_robustness"]

ORIGINAL = "original"
TOLERANCE = 1e-9
#: T8b's questions by kind (the question names of its four families).
KINDS = {
    "falls_over": "planned yes/no", "falls_off": "planned yes/no", "tips_over": "planned yes/no",
    "keeps_place": "stays-put yes/no",
    "falls_if_further_in": "counterfactual yes/no", "tips_at_other_height": "counterfactual yes/no",
    "falls_other_braking": "counterfactual score", "falls_other_speed": "counterfactual score", "falls_if_further_out": "counterfactual score",
    "tip_risk_other_height": "counterfactual score",
    "what_happens": "outcome choice",
}


def label_numbers(label: Mapping[str, Any]) -> dict[str, float]:
    """The numbers of one label: ``{"p_yes": …}``, ``{"value": …}`` or the choice probabilities."""
    if isinstance(label.get("probabilities"), Mapping):
        return {str(k): float(v) for k, v in label["probabilities"].items()}
    for key in ("p_yes", "value"):
        if label.get(key) is not None:
            return {key: float(label[key])}
    raise ValueError(f"label without numbers: {label!r}")


def _identical(a: Mapping[str, Any], b: Mapping[str, Any]) -> bool:
    na, nb = label_numbers(a), label_numbers(b)
    return set(na) == set(nb) and all(abs(na[k] - nb[k]) <= TOLERANCE for k in na)


def reproduction(records: Sequence[Mapping[str, Any]], rows: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """Share of the stored questions whose relabelled label (``rows[id]["labels"]``, original prior) is identical within 1e-9. A record
    without a row counts every question as not reproduced."""
    total = same = 0
    by_type: dict[str, list[int]] = {}
    mismatches = []
    for record in records:
        row = rows.get(str(record["id"]))
        for name, label in (record.get("labels") or {}).items():
            qtype = str(label.get("type"))
            ok = bool(row is not None and name in row["labels"] and _identical(label, row["labels"][name]))
            total += 1
            same += ok
            counts = by_type.setdefault(qtype, [0, 0])
            counts[0] += ok
            counts[1] += 1
            if not ok and len(mismatches) < 20:
                mismatches.append({"id": record["id"], "question": name, "stored": label_numbers(label),
                                   "relabelled": label_numbers(row["labels"][name]) if row and name in row["labels"] else None})
    return {"questions": total, "identical": same, "share": same / total if total else None,
            "by_type": {t: {"identical": c[0], "questions": c[1]} for t, c in sorted(by_type.items())}, "mismatches": mismatches}


def band_robustness(records: Sequence[Mapping[str, Any]], rows_by_prior: Mapping[str, Mapping[str, Mapping[str, Any]]]) -> dict[tuple[str, str], dict[str, Any]]:
    """(record id, question) → its robustness over the original setting (the stored labels) and every setting in `rows_by_prior`
    (``{prior: {id: row}}``). Raises when a relabelled label is missing — the robust set must not silently shrink."""
    settings = [ORIGINAL] + [p for p in rows_by_prior if p != ORIGINAL]
    out: dict[tuple[str, str], dict[str, Any]] = {}
    for record in records:
        rid = str(record["id"])
        for name, label in (record.get("labels") or {}).items():
            by_prior = {ORIGINAL: label}
            for prior in settings[1:]:
                row = rows_by_prior[prior].get(rid)
                if row is None or name not in row["labels"]:
                    raise ValueError(f"{rid}/{name}: no label under {prior}")
                by_prior[prior] = row["labels"][name]
            if label.get("type") in ("noul", "score"):
                values = {p: next(iter(label_numbers(l).values())) for p, l in by_prior.items()}
                bands = {p: band(v) for p, v in values.items()}
                out[(rid, name)] = {"type": label["type"], "values": values, "min": min(values.values()), "max": max(values.values()),
                                    "bands": bands, "band_same": len(set(bands.values())) == 1, "robust": len(set(bands.values())) == 1}
            else:
                argmax = {p: max(label_numbers(l).items(), key=lambda kv: (kv[1], kv[0]))[0] for p, l in by_prior.items()}
                out[(rid, name)] = {"type": "choice", "argmax": argmax, "argmax_same": len(set(argmax.values())) == 1,
                                    "robust": len(set(argmax.values())) == 1}
    return out


def with_robustness(records: Sequence[Mapping[str, Any]], robustness: Mapping[tuple[str, str], Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Deep copies of `records` with ``meta.robustness[q]`` from `robustness` (in memory; the stored files are never touched)."""
    out = []
    for record in records:
        copied = copy.deepcopy(dict(record))
        rid = str(copied["id"])
        copied.setdefault("meta", {})["robustness"] = {name: dict(robustness[(rid, name)]) for name in copied["request"]["questions"]
                                                       if (rid, name) in robustness}
        out.append(copied)
    return out
