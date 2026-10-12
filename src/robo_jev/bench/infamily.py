"""In-family System-1 readings on records of the v0.3b MuJoCo generator (task B1) — small adapters from :mod:`robo_jev.mjgen` records to
:mod:`robo_jev.bench.system1`. The System-1 definitions used for T8b are not changed; this module only says which items of an mjgen record
they are computed on, and how the bootstrap clusters are formed.

**Mappings** (each one is used by ``scripts/b1_verdict.py``):

* **Band items** — :func:`robo_jev.bench.system1.question_items` on the records as they are: every yes/no question (the key event under
  the plan, e.g. ``moves_at_plan``; the severe outcome, e.g. ``topples_at_plan``) and every score question (the key event at the
  counterfactual value, e.g. ``risk_at_other_pitch``). Label = ``labels`` (the base prior). No adapter.
* **Robust** — a yes/no or score question is robust iff ``meta.robustness[q].robust`` (its band is the same under the three prior
  settings base, alt_a and alt_b); the others are *prior-ambiguous*. :func:`robust_keys`, :func:`ambiguous_keys`, :func:`keep`.
* **Cluster** — the generator group ``<family>:<group index>`` (``generator.family``, ``generator.group``): a one-fact pair's two records
  share it. :func:`group_of`, :func:`regroup` (system1 clusters by record, and a T5-style pair by ``pair:<base id>``).
* **Direction** — :func:`robo_jev.bench.system1.t5_direction_pairs` + :func:`robo_jev.bench.system1.order_items` with
  :data:`~robo_jev.bench.system1.DIRECTION_MIN` on ``meta.pair`` (kind ``numeric``; every v0.3b pair changes exactly one stated fact): every
  yes/no or score question both records ask, base → partner. No adapter beyond the cluster. :func:`direction_items`.
* **Comparison** — the in-record comparison question (``meta.question_kind[q] == "compare"``, e.g. ``riskier_orientation``: "which is
  more likely to fail: the plan as it stands, the other action, or about the same?"). Its two labels are the key-event shares of the
  plan and of the compared action under the base prior (``meta.outcomes[<variant>]["base"]`` summed over the family's key events; the
  compared variant is ``meta.plan.options[meta.compare_option]``). Items whose two labels differ by at least
  :data:`~robo_jev.bench.system1.COMPARISON_MIN`; the answer orders the two actions by P(compared action riskier) − P(plan riskier)
  (option keys ``meta.compare_option`` and :data:`PLAN_KEY`); it is right iff that difference is non-zero and has the sign of the label
  difference (an equal answer misses, as in T8b). Single order: each listing weighted 1/L (as system1). :func:`comparison_items`.

**Descriptive readings** (secondary): the 3×3 band confusion (:func:`band_confusion`) and the answer profile of a question set — mean,
spread, band shares, the hedge share (answers in the middle band) and the share whose band is one the label takes under some prior
(:func:`answer_profile`).
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from robo_jev.bench import system1
from robo_jev.bench.system1 import COMPARISON_MIN, DIRECTION_MIN, Items, band

__all__ = [
    "BANDS",
    "EVENT",
    "PLAN_KEY",
    "ambiguous_keys",
    "answer_profile",
    "band_confusion",
    "by_family",
    "comparison_items",
    "direction_items",
    "family_of",
    "group_of",
    "keep",
    "regroup",
    "robust_keys",
]

BANDS = ("low", "middle", "high")
#: The key event of each family (any failure) — :data:`robo_jev.mjgen.families.EVENT` (a test checks they agree).
EVENT = {"tilt": ("slide", "tip"), "grip": ("slip", "drop"), "tower": ("top", "more")}
#: The plan's option key in the comparison question — :data:`robo_jev.mjgen.generate.PLANNED_KEY` (a test checks they agree).
PLAN_KEY = {"tilt": "as_it_stands", "grip": "as_planned", "tower": "as_planned"}
_EPS = 1e-9


def group_of(record: Mapping[str, Any]) -> str:
    """The bootstrap cluster of a record: its generator group ``<family>:<group index>`` (shared by a pair's two records)."""
    gen = record.get("generator") or {}
    return f"{gen['family']}:{int(gen['group'])}"


def family_of(record: Mapping[str, Any]) -> str:
    return str((record.get("meta") or {}).get("scene"))


def _scalar_questions(record: Mapping[str, Any]) -> Iterable[str]:
    for name, question in record["request"]["questions"].items():
        if question.get("type") in ("noul", "score"):
            yield name


def robust_keys(records: Sequence[Mapping[str, Any]]) -> set[tuple[str, str]]:
    """(record id, question) of every yes/no or score question whose band is the same under all three priors (``meta.robustness``)."""
    out = set()
    for record in records:
        robustness = (record.get("meta") or {}).get("robustness") or {}
        for name in _scalar_questions(record):
            if bool((robustness.get(name) or {}).get("robust")):
                out.add((str(record["id"]), name))
    return out


def ambiguous_keys(records: Sequence[Mapping[str, Any]]) -> set[tuple[str, str]]:
    """(record id, question) of every yes/no or score question whose band depends on the prior (not robust)."""
    every = {(str(record["id"]), name) for record in records for name in _scalar_questions(record)}
    return every - robust_keys(records)


def keep(items: Items, keys: set[tuple[str, str]]) -> Items:
    """The question items whose (record id, question) — the last two parts of the item key — is in `keys`."""
    return {key: value for key, value in items.items() if tuple(key[-2:]) in keys}


def regroup(items: Items, records: Sequence[Mapping[str, Any]]) -> Items:
    """Items with their cluster replaced by the record's generator group (:func:`group_of`). system1 clusters a question by its record
    id and a T5-style pair by ``pair:<base id>``; both map to the group."""
    groups = {str(record["id"]): group_of(record) for record in records}
    out: Items = {}
    for key, (cluster, value) in items.items():
        rid = cluster[len("pair:"):] if cluster.startswith("pair:") else cluster
        out[key] = (groups.get(rid, cluster), value)
    return out


def direction_items(set_name: str, records: Sequence[Mapping[str, Any]], predictions: Sequence[Mapping[str, Any]], *,
                    reading: str = "single_order") -> Items:
    """Direction on the one-fact pairs (``meta.pair``): system1's T5 definition, clustered by group."""
    pairs = system1.t5_direction_pairs(records)
    return regroup(system1.order_items(set_name, records, predictions, pairs, min_move=DIRECTION_MIN, reading=reading), records)


def _event_share(meta: Mapping[str, Any], variant: str, family: str) -> float:
    freq = meta["outcomes"][variant]["base"]
    return float(sum(float(freq[o]) for o in EVENT[family]))


def comparison_items(set_name: str, records: Sequence[Mapping[str, Any]], predictions: Sequence[Mapping[str, Any]], *,
                     reading: str = "single_order", min_move: float = COMPARISON_MIN) -> Items:
    """Comparison on the in-record comparison questions (module docstring): value = the weight of the listings whose answer orders the
    plan and the compared action like their labels. Items keyed ``(set, record id, question)``, clustered by group; questions without an
    answer are left out."""
    predicted = {str(p["id"]): p for p in predictions}
    out: Items = {}
    for record in records:
        meta = record.get("meta") or {}
        family = family_of(record)
        rid = str(record["id"])
        for name, kind in (meta.get("question_kind") or {}).items():
            if kind != "compare" or name not in record["request"]["questions"]:
                continue
            option = meta.get("compare_option")
            variant = (meta.get("plan") or {}).get("options", {}).get(option)
            if option is None or variant is None:
                raise ValueError(f"{rid}/{name}: comparison question without meta.compare_option / meta.plan.options")
            moved = _event_share(meta, variant, family) - _event_share(meta, "planned", family)
            if abs(moved) < min_move - _EPS or moved == 0:
                continue
            answer = ((predicted.get(rid) or {}).get("questions") or {}).get(name)
            if not isinstance(answer, Mapping):
                continue
            value = 0.0
            for listing, weight in system1._answers(answer, reading):  # the same listing weights as every system1 metric
                probs = listing.get("probabilities") or {}
                order = float(probs.get(option, 0.0)) - float(probs.get(PLAN_KEY[family], 0.0))
                if order != 0 and (order > 0) == (moved > 0):
                    value += weight
            out[(set_name, rid, name)] = (group_of(record), value)
    return out


def _scalar(answer: Mapping[str, Any] | None) -> float | None:
    if not isinstance(answer, Mapping):
        return None
    for key in ("p_yes", "value"):
        if answer.get(key) is not None:
            return float(answer[key])
    return None


def _weighted_answers(records: Sequence[Mapping[str, Any]], predictions: Sequence[Mapping[str, Any]], keys: set[tuple[str, str]],
                      reading: str) -> list[tuple[Mapping[str, Any], str, float, float]]:
    """(record, question, answer value, weight) over the listings of every answered question in `keys` (each question weighs 1)."""
    predicted = {str(p["id"]): p for p in predictions}
    out = []
    for record in records:
        rid = str(record["id"])
        for name in _scalar_questions(record):
            if (rid, name) not in keys:
                continue
            answer = ((predicted.get(rid) or {}).get("questions") or {}).get(name)
            if not isinstance(answer, Mapping):
                continue
            parts = [(_scalar(a), w) for a, w in system1._answers(answer, reading)]
            if any(p is None for p, _ in parts):
                continue
            out += [(record, name, float(p), float(w)) for p, w in parts]
    return out


def band_confusion(records: Sequence[Mapping[str, Any]], predictions: Sequence[Mapping[str, Any]], keys: set[tuple[str, str]], *,
                   reading: str = "single_order") -> dict[str, Any]:
    """3×3 band confusion over the questions in `keys`: rows = the label's band (base prior), columns = the answer's band; ``counts`` are
    question weights (single order: each listing 1/L) and ``rows`` the row shares."""
    counts = {row: {col: 0.0 for col in BANDS} for row in BANDS}
    for record, name, value, weight in _weighted_answers(records, predictions, keys, reading):
        label = _scalar((record.get("labels") or {}).get(name))
        counts[band(label)][band(value)] += weight
    rows = {row: ({col: counts[row][col] / sum(counts[row].values()) for col in BANDS} if sum(counts[row].values()) > 0 else None) for row in BANDS}
    return {"counts": counts, "rows": rows, "questions": sum(sum(r.values()) for r in counts.values())}


def _quantile(values: np.ndarray, weights: np.ndarray, q: float) -> float:
    order = np.argsort(values, kind="stable")
    cum = np.cumsum(weights[order])
    return float(values[order][min(len(values) - 1, int(np.searchsorted(cum, q * cum[-1])))])


def answer_profile(records: Sequence[Mapping[str, Any]], predictions: Sequence[Mapping[str, Any]], keys: set[tuple[str, str]], *,
                   reading: str = "single_order") -> dict[str, Any]:
    """How a model answers the yes/no and score questions in `keys` (each question weighs 1, split over its listings): mean and spread
    (weighted SD, 10th and 90th percentile) of the answers, the share of answers per band, the hedge share (= the middle-band share),
    the share whose band is one the label takes under some prior (``meta.robustness[q].bands``) and the share inside the label's range
    over the priors (``min`` ≤ answer ≤ ``max``)."""
    rows = _weighted_answers(records, predictions, keys, reading)
    if not rows:
        return {"questions": 0}
    values = np.array([v for _, _, v, _ in rows])
    weights = np.array([w for _, _, _, w in rows])
    total = float(weights.sum())
    mean = float((values * weights).sum() / total)
    shares = {b: float(sum(w for _, _, v, w in rows if band(v) == b) / total) for b in BANDS}
    in_bands = in_range = 0.0
    for record, name, value, weight in rows:
        entry = ((record.get("meta") or {}).get("robustness") or {}).get(name) or {}
        if band(value) in set((entry.get("bands") or {}).values()):
            in_bands += weight
        if entry.get("min") is not None and float(entry["min"]) - _EPS <= value <= float(entry["max"]) + _EPS:
            in_range += weight
    return {"questions": round(total, 6), "mean": mean, "sd": float(np.sqrt((weights * (values - mean) ** 2).sum() / total)),
            "p10": _quantile(values, weights, 0.10), "p90": _quantile(values, weights, 0.90), "band_shares": shares, "hedge_share": shares["middle"],
            "in_prior_bands": in_bands / total, "in_prior_range": in_range / total}


def by_family(items: Items, records: Sequence[Mapping[str, Any]]) -> dict[str, Items]:
    """Items split by the family of their record (the record id is the second part of every item key)."""
    families = {str(record["id"]): family_of(record) for record in records}
    out: dict[str, Items] = defaultdict(dict)
    for key, value in items.items():
        rid = key[-2] if key[0] in BANDS else key[1]
        out[families[rid]][key] = value
    return dict(out)
