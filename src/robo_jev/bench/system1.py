"""System-1 metrics (Task s1score) — re-score existing predictions with judgments that fit a fast, integrated decision layer.

robo-jev is a System 1: it makes coarse, integrated judgments — *this is risky, slow down gently, look again, call a person*.
Getting the exact physical numbers right is System 0's job (the execution and control layer below). Brier against one
simulator's posterior (19 % vs 28 %) asks System 1 to do System 0's job and penalises sensible judgments that rest on other
priors, so these metrics ask coarser questions of the **same predictions**. Nothing here calls a model or an API.

**Metrics.** Every rate is a share of *items*; an item is one question, or one pair of questions.

* **Risk bands** — each yes/no (``noul``) or ``score`` label y and answer p falls in a band: low (x < 1/3), middle
  (1/3 ≤ x ≤ 2/3), high (x > 2/3).

  - ``band_accuracy``: band(p) = band(y); ``band_balanced``: the mean over the three label bands of the band accuracy within the
    band (``band_recall:<band>``) — free of base rates (answering "low" every time scores 1/3 whatever the share of low labels).
  - ``severe_error``: label high and answer low, or label low and answer high — a share of *all* questions; it splits into
    ``severe_miss`` (label high, answer low: an under-warning) and ``severe_false_alarm`` (label low, answer high).
  - ``…_clear``: the same over the questions whose label is more than 0.05 away from both band edges
    (|y − 1/3| > 0.05 and |y − 2/3| > 0.05) — the prior-robust reading.
* **Direction** (``direction``) — one-fact pairs: does the answer move the same way as the label? Only pairs whose label moves
  by at least 0.1; an unchanged answer counts as a miss.

  - home ground: T5 numeric pairs — every yes/no and score question both records of the pair ask (same name, base → variant);
  - outside: T8b, the planned question ↔ each counterfactual question about the same event (the T8 direction diagnostic).
* **Comparison** (``comparison``) — two questions in one record about the same event under different hypothetical actions
  (contact if I keep my speed vs contact if I brake hard): does the model order them like the labels? Only pairs whose labels
  differ by at least 0.15; equal answers count as a miss.

  - home ground: grouped by the recipe identity the engine records (``meta.basis``, as :func:`robo_jev.simgen.testsets.recipe_of`
    and :func:`robo_jev.simgen.questions.identity` read it) — yes/no event questions and event-probability score questions with
    the same source and event and different options; a negated yes/no is turned back into the event's probability.
    Quantity scores (delay under two actions) are kept apart as ``comparison:quantity``;
  - outside: every pair among a T8b record's planned and counterfactual questions about the same event.
* **Action choice** — action and look-again questions with expected costs, through the existing scale-free regret
  ``regret_norm`` = (E[cost(model argmax)] − min E[cost]) / (max − min) (:func:`robo_jev.decisions.metrics.choice_metrics`).

  - ``safe_choice``: regret_norm ≤ 0.1 (the top action is within a tenth of the cost range of the best);
  - ``catastrophic_choice``: regret_norm ≥ 0.5 (the worst option, or one at least half the range above the best).
  Split by criterion: ``default`` (no stated criterion — costs from our value table, which the API models never saw),
  ``stated`` (the question states it: avoid contact / spill / falls, time critical, no defect) and ``look_again``.
* **Outcome category** (``outcome``) — T8b's choice questions (stays / slides / topples): the answer's argmax is in the label's
  argmax set. ``outcome:all_families`` adds the release and push families through their planned yes/no question (falls vs stays,
  tips vs upright: the answer's side of 0.5 against the label's; an answer of exactly 0.5 is a miss, and a label of exactly 0.5
  accepts every answer).
* **Choice accuracy** (``choice_accuracy``) — T8a (PIQA): the argmax is the right solution.

**Readings.** ``single_order`` (deployment condition): a question answered under several balanced listings counts each listing
with weight 1/L — the expected value when asked in one order. A pair is read in one *fixed format*: listing pairs matched by
option order when both questions show the same option orders (two yes/no questions), otherwise every combination with weight
1/(La·Lb) (a yes/no and a score question). ``order_averaged``: the answer averaged over listings. OpenRouter models were asked
once in the stored order, so their two readings agree.

**Intervals.** Cluster bootstrap 95 % (percentile, 2,000 draws by default): the cluster is the record (a T5 pair spans two records
and is its own cluster). Paired differences use the items both models answered and the same draws.
"""

from __future__ import annotations

import itertools
from collections import defaultdict
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from robo_jev.decisions.metrics import average_answers, choice_metrics, listings_of
from robo_jev.decisions.records import pair_of

__all__ = [
    "BAND_EDGES",
    "DETAIL_METRICS",
    "SCORECARD_METRICS",
    "STRATIFIED",
    "CATASTROPHIC_REGRET",
    "CLEAR_MARGIN",
    "COMPARISON_MIN",
    "DEFINITIONS",
    "DIRECTION_MIN",
    "LOWER_IS_BETTER",
    "READINGS",
    "SAFE_REGRET",
    "T8B_EVENTS",
    "band",
    "basis_event_pairs",
    "basis_quantity_pairs",
    "group_items",
    "is_clear",
    "label_profile",
    "oracle_predictions",
    "order_items",
    "paired_difference",
    "question_items",
    "render_markdown",
    "score",
    "set_items",
    "summarize",
    "t5_direction_pairs",
    "t8b_direction_pairs",
    "t8b_event_pairs",
]

#: Band edges: low < 1/3 ≤ middle ≤ 2/3 < high.
BAND_EDGES = (1.0 / 3.0, 2.0 / 3.0)
#: Clear cases skip labels within this distance of a band edge (inclusive).
CLEAR_MARGIN = 0.05
#: Direction counts only pairs whose label moves at least this much.
DIRECTION_MIN = 0.1
#: Comparison counts only pairs whose labels differ at least this much.
COMPARISON_MIN = 0.15
#: Safe choice: normalised regret at most this.
SAFE_REGRET = 0.1
#: Catastrophic choice: normalised regret at least this.
CATASTROPHIC_REGRET = 0.5
#: The T8 report's direction diagnostic used this threshold (and order-averaged answers) — kept as a reproduction check.
T8_REPORT_MOVE = 0.05
READINGS = ("single_order", "order_averaged")
_EPS = 1e-9

#: T8b: the questions of one record about the same event (tipping / falling off), planned question first — the pairs of the
#: T8 direction diagnostic (``artifacts/scratch/t8/analysis/direction.py``); ``keeps_place`` asks about a different event.
T8B_EVENTS: dict[str, tuple[str, ...]] = {
    "deck_brake": ("falls_over", "falls_other_braking"),
    "deck_turn": ("falls_over", "falls_other_speed"),
    "stack": ("falls_off", "falls_if_further_out", "falls_if_further_in"),
    "push": ("tips_over", "tips_at_other_height", "tip_risk_other_height"),
}
#: T8b's outcome-category choice question (deck families).
T8B_OUTCOME = "what_happens"

#: Metrics where lower is better (the rest: higher is better).
LOWER_IS_BETTER = frozenset({
    "severe_error", "severe_error_clear", "severe_miss", "severe_false_alarm", "severe_miss_clear", "severe_error:event", "catastrophic_choice",
    "catastrophic_choice:default", "catastrophic_choice:stated", "catastrophic_choice:look_again",
})

DEFINITIONS = {
    "readings": "single_order = each listing of a question weighted 1/L (expected value when asked in one order; deployment condition); a pair "
                "is read in one fixed format (listing pairs matched by option order when both questions show the same option orders, otherwise "
                "every combination weighted 1/(La*Lb)); order_averaged = the answer averaged over listings",
    "band": "low if x < 1/3, high if x > 2/3, else middle; applied to the label probability y and to the answer p of every yes/no and score question",
    "band_accuracy": "share of yes/no and score questions with band(answer) = band(label)",
    "band_balanced": "balanced band accuracy: the mean over the three label bands of the band accuracy within that band (band_recall:low / "
                     "middle / high) — base-rate free: answering 'low' every time scores 1/3 whatever the share of low labels",
    "severe_error": "share of yes/no and score questions with label high and answer low, or label low and answer high (= severe_miss + severe_false_alarm)",
    "severe_miss": "share of yes/no and score questions with label high and answer low (an under-warning)",
    "severe_false_alarm": "share of yes/no and score questions with label low and answer high",
    "event": "band_balanced:event / severe_error:event leave out the home sets' quantity scores (time costs such as a detour's delay, "
             "meta.basis source 'quantity'); every T8b question is an event probability",
    "clear": "the _clear variants skip questions whose label lies within 0.05 of a band edge (|y - 1/3| <= 0.05 or |y - 2/3| <= 0.05)",
    "direction": "one-fact pairs (T5 numeric pairs: same yes/no or score question in base and variant; T8b: planned vs each counterfactual "
                 "question about the same event) whose label moves >= 0.1: share where the answer moves the same way (an unchanged answer is a miss)",
    "comparison": "pairs of questions in one record about the same event under different hypothetical actions (home: same meta.basis source and "
                  "event, different options, negated yes/no turned back into the event probability; T8b: planned and counterfactual questions "
                  "about the same event) whose labels differ by >= 0.15: share where the answers are ordered like the labels (equal answers miss)",
    "comparison:quantity": "as comparison, for quantity scores (same quantity and transform under two actions; home ground only)",
    "safe_choice": "action / look-again questions with expected costs: share where the model's top action has normalised regret <= 0.1 "
                   "(regret / (max - min expected cost))",
    "catastrophic_choice": "share where the model's top action has normalised regret >= 0.5 (includes the worst option)",
    "criterion_classes": "default = action question without a stated criterion (costs from our value table, unseen by the API models); stated = "
                         "the question states the criterion (avoid_* / time_critical / no_defect); look_again = act now vs look first",
    "outcome": "T8b deck choice question (stays / slides / topples): the answer's argmax is in the label's argmax set",
    "outcome:all_families": "outcome plus the release and push families through their planned yes/no question (event vs no event by the side of 0.5; "
                            "an answer of exactly 0.5 is a miss unless the label is exactly 0.5, when every answer is right)",
    "choice_accuracy": "T8a (PIQA): the answer's argmax is the right solution",
    "physics_skill": "reference (the Brier metric these metrics replace): 1 - sum Brier / sum Brier(always 0.5) over yes/no and score questions",
    "intervals": "cluster bootstrap 95 % percentile intervals (cluster = record; a T5 pair is one cluster); paired differences on the items both "
                 "models answered, same draws",
}


# --------------------------------------------------------------------------
# Bands and answers
# --------------------------------------------------------------------------


def band(x: float) -> str:
    """``low`` (x < 1/3), ``high`` (x > 2/3) or ``middle``."""
    x = float(x)
    if x < BAND_EDGES[0]:
        return "low"
    if x > BAND_EDGES[1]:
        return "high"
    return "middle"


def is_clear(y: float) -> bool:
    """A label more than :data:`CLEAR_MARGIN` away from both band edges."""
    return all(abs(float(y) - edge) > CLEAR_MARGIN for edge in BAND_EDGES)


def _scalar(answer: Mapping[str, Any] | None) -> float | None:
    """A yes/no or score answer (or label) → its probability / value."""
    if not isinstance(answer, Mapping):
        return None
    for key in ("p_yes", "value"):
        if answer.get(key) is not None:
            return float(answer[key])
    return None


def _answers(question: Mapping[str, Any], reading: str) -> list[tuple[dict[str, Any], float]]:
    """(answer, weight) of one predicted question in a reading (weights sum to 1)."""
    listings = listings_of(dict(question))
    if reading == "order_averaged":
        return [(average_answers(str(question["type"]), [l["answer"] for l in listings]), 1.0)]
    if reading != "single_order":
        raise ValueError(f"reading: one of {list(READINGS)} (got {reading!r})")
    return [(l["answer"], 1.0 / len(listings)) for l in listings]


def _joint(qa: Mapping[str, Any], qb: Mapping[str, Any], reading: str) -> list[tuple[dict[str, Any], dict[str, Any], float]]:
    """(answer a, answer b, weight) of two predicted questions read together (module docstring: one fixed format)."""
    if reading == "order_averaged":
        return [(_answers(qa, reading)[0][0], _answers(qb, reading)[0][0], 1.0)]
    la, lb = listings_of(dict(qa)), listings_of(dict(qb))
    ka = [tuple(l.get("options") or ()) for l in la]
    kb = [tuple(l.get("options") or ()) for l in lb]
    if reading != "single_order":
        raise ValueError(f"reading: one of {list(READINGS)} (got {reading!r})")
    if len(ka) == len(kb) and len(set(ka)) == len(ka) and set(ka) == set(kb):
        index = {key: i for i, key in enumerate(kb)}
        return [(la[i]["answer"], lb[index[key]]["answer"], 1.0 / len(la)) for i, key in enumerate(ka)]
    return [(x["answer"], y["answer"], 1.0 / (len(la) * len(lb))) for x in la for y in lb]


def _by_id(items: Iterable[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    return {str(item["id"]): item for item in items}


# --------------------------------------------------------------------------
# Items: questions
# --------------------------------------------------------------------------

#: An item: key → (cluster, value). Keys are tuples of strings (set, record, question[, record, question]).
Items = dict[tuple[str, ...], tuple[str, float]]


def _criterion_class(record: Mapping[str, Any], name: str) -> str:
    basis = ((record.get("meta") or {}).get("basis") or {}).get(name)
    if isinstance(basis, Mapping):
        if basis.get("kind") == "voi":
            return "look_again"
        if basis.get("kind") == "action":
            return "default" if not basis.get("criterion") else "stated"
    criteria = (record.get("meta") or {}).get("criteria")
    if isinstance(criteria, Mapping) and name in criteria:
        return "default" if not criteria[name] else "stated"
    return "default"


def _implied_category(value: float) -> str | None:
    """Event / no event by the side of 0.5 (exactly 0.5: no category)."""
    if value > 0.5:
        return "event"
    if value < 0.5:
        return "none"
    return None


def question_items(set_name: str, records: Sequence[Mapping[str, Any]], predictions: Sequence[Mapping[str, Any]], *,
                   reading: str = "single_order") -> dict[str, Items]:
    """Per-question metrics of one set: bands (yes/no and score), action choice, T8b outcome category, choice accuracy (PIQA-style
    ``probe`` choices). Questions without a prediction are left out (coverage is read against :func:`oracle_predictions`)."""
    out: dict[str, Items] = defaultdict(dict)
    predicted = _by_id(predictions)
    for record in records:
        rid = str(record["id"])
        meta = record.get("meta") or {}
        questions = ((predicted.get(rid) or {}).get("questions")) or {}
        labels = record.get("labels") or {}
        t8b = meta.get("set") == "T8b"
        for name, question in record["request"]["questions"].items():
            label, answer = labels.get(name), questions.get(name)
            if not isinstance(label, Mapping) or not isinstance(answer, Mapping):
                continue
            qtype = str(question.get("type"))
            key = (set_name, rid, name)
            if qtype in ("noul", "score"):
                y = _scalar(label)
                if y is None:
                    continue
                parts = [(_scalar(a), w) for a, w in _answers(answer, reading)]
                if any(p is None for p, _ in parts):
                    continue
                label_band = band(y)
                correct = sum(w for p, w in parts if band(p) == label_band)
                miss = sum(w for p, w in parts if label_band == "high" and band(p) == "low")
                alarm = sum(w for p, w in parts if label_band == "low" and band(p) == "high")
                out["band_accuracy"][key] = (rid, correct)
                out[f"band_accuracy:{qtype}"][key] = (rid, correct)
                out["band_balanced"][(label_band, *key)] = (rid, correct)
                out[f"band_recall:{label_band}"][key] = (rid, correct)
                recipe = ((meta.get("basis") or {}).get(name)) or {}
                if not (isinstance(recipe, Mapping) and recipe.get("source") == "quantity"):  # event probabilities only (no time costs)
                    out["band_balanced:event"][(label_band, *key)] = (rid, correct)
                    out["severe_error:event"][key] = (rid, miss + alarm)
                out["severe_error"][key] = (rid, miss + alarm)
                out["severe_miss"][key] = (rid, miss)
                out["severe_false_alarm"][key] = (rid, alarm)
                if is_clear(y):
                    out["band_accuracy_clear"][key] = (rid, correct)
                    out["severe_error_clear"][key] = (rid, miss + alarm)
                    out["severe_miss_clear"][key] = (rid, miss)
                if t8b and name == T8B_EVENTS.get(str(meta.get("scene")), ("",))[0] and meta.get("scene") in ("stack", "push"):
                    if y == 0.5:  # both outcomes equally likely: any answer is in the label's argmax set
                        value = 1.0
                    else:
                        value = sum(w for p, w in parts if _implied_category(p) == ("event" if y > 0.5 else "none"))
                    out["outcome:all_families"][key] = (rid, value)
                continue
            if qtype != "choice" or not isinstance(label.get("probabilities"), Mapping):
                continue
            parts = [(a.get("probabilities"), w) for a, w in _answers(answer, reading)]
            if any(not isinstance(p, Mapping) for p, _ in parts):
                continue
            metrics = [(choice_metrics(p, dict(label)), w) for p, w in parts]
            kind = label.get("kind")
            if isinstance(label.get("expected_cost"), Mapping):
                regrets = [(m["regret_norm"], w) for m, w in metrics if m["regret_norm"] is not None]
                if not regrets:
                    continue
                costs = [float(v) for v in label["expected_cost"].values()]
                spread = max(costs) - min(costs) > 0
                safe = sum(w for r, w in regrets if r <= SAFE_REGRET + _EPS)
                bad = sum(w for r, w in regrets if spread and r >= CATASTROPHIC_REGRET - _EPS)
                cls = _criterion_class(record, name)
                for metric, value in (("safe_choice", safe), ("catastrophic_choice", bad)):
                    out[metric][key] = (rid, value)
                    out[f"{metric}:{cls}"][key] = (rid, value)
            elif t8b and name == T8B_OUTCOME:
                value = sum(w for m, w in metrics if m["match"])
                out["outcome"][key] = (rid, value)
                out["outcome:all_families"][key] = (rid, value)
            elif kind == "probe":
                out["choice_accuracy"][key] = (rid, sum(w for m, w in metrics if m["match"]))
    return dict(out)


# --------------------------------------------------------------------------
# Items: pairs
# --------------------------------------------------------------------------

#: A pair: (cluster, (record a, question a, negated a), (record b, question b, negated b)); the label moves a → b.
Pair = tuple[str, tuple[str, str, bool], tuple[str, str, bool]]


def t5_direction_pairs(records: Sequence[Mapping[str, Any]]) -> list[Pair]:
    """T5 numeric one-fact pairs: every yes/no or score question both records ask (same name and type), base → variant; the pair is
    its own cluster. Base and variant follow ``meta.pair.role`` (else ``pair.of`` points at the base), as
    :func:`robo_jev.decisions.metrics.pair_deltas`."""
    by_id = _by_id(records)
    seen: set[tuple[str, str]] = set()
    out: list[Pair] = []
    for record in records:
        partner, kind = pair_of(dict(record))
        rid = str(record["id"])
        if kind != "numeric" or partner is None or partner not in by_id:
            continue
        key = (min(rid, partner), max(rid, partner))
        if key in seen:
            continue
        seen.add(key)
        role = ((record.get("meta") or {}).get("pair") or {}).get("role")
        base, variant = (record, by_id[partner]) if role == "base" else (by_id[partner], record)
        for name, question in variant["request"]["questions"].items():
            other = base["request"]["questions"].get(name)
            if question.get("type") not in ("noul", "score") or other is None or other.get("type") != question.get("type"):
                continue
            out.append((f"pair:{base['id']}", (str(base["id"]), name, False), (str(variant["id"]), name, False)))
    return out


def _basis_groups(record: Mapping[str, Any], *, quantity: bool) -> dict[tuple[Any, ...], list[tuple[str, Any, bool]]]:
    basis = (record.get("meta") or {}).get("basis") or {}
    groups: dict[tuple[Any, ...], list[tuple[str, Any, bool]]] = defaultdict(list)
    for name, question in record["request"]["questions"].items():
        recipe = basis.get(name)
        if not isinstance(recipe, Mapping) or recipe.get("option") is None:
            continue
        kind, qtype = recipe.get("kind"), question.get("type")
        if quantity:
            if kind == "score" and qtype == "score" and recipe.get("source") == "quantity":
                groups[("quantity", recipe.get("name"), recipe.get("transform"))].append((name, recipe["option"], False))
            continue
        if kind == "noul" and qtype == "noul":
            groups[("event", recipe.get("source"), recipe.get("event"))].append((name, recipe["option"], bool(recipe.get("negate"))))
        elif kind == "score" and qtype == "score" and recipe.get("source") == "event":
            groups[("event", "event", recipe.get("name"))].append((name, recipe["option"], False))
    return groups


def _within_pairs(record: Mapping[str, Any], groups: Mapping[tuple[Any, ...], list[tuple[str, Any, bool]]]) -> list[Pair]:
    rid = str(record["id"])
    out: list[Pair] = []
    for members in groups.values():
        for (n1, o1, g1), (n2, o2, g2) in itertools.combinations(members, 2):
            if o1 != o2:
                out.append((rid, (rid, n1, g1), (rid, n2, g2)))
    return out


def basis_event_pairs(records: Sequence[Mapping[str, Any]]) -> list[Pair]:
    """Home ground: pairs of questions in one record about the same event under different options (actions), by the engine's
    recipe (``meta.basis``) — yes/no (source event or world) and event-probability scores; a negated yes/no is flagged."""
    return [pair for record in records for pair in _within_pairs(record, _basis_groups(record, quantity=False))]


def basis_quantity_pairs(records: Sequence[Mapping[str, Any]]) -> list[Pair]:
    """Home ground: pairs of quantity scores in one record (same quantity and transform, different options)."""
    return [pair for record in records for pair in _within_pairs(record, _basis_groups(record, quantity=True))]


def _t8b_names(record: Mapping[str, Any]) -> tuple[str, ...]:
    meta = record.get("meta") or {}
    if meta.get("set") != "T8b":
        return ()
    names = T8B_EVENTS.get(str(meta.get("scene")), ())
    return tuple(n for n in names if n in record["request"]["questions"])


def t8b_direction_pairs(records: Sequence[Mapping[str, Any]]) -> list[Pair]:
    """T8b: the planned question → each counterfactual question about the same event (the T8 direction diagnostic's pairs)."""
    out: list[Pair] = []
    for record in records:
        names = _t8b_names(record)
        rid = str(record["id"])
        planned = T8B_EVENTS.get(str((record.get("meta") or {}).get("scene")), ("",))[0]
        if planned not in names:
            continue
        out += [(rid, (rid, planned, False), (rid, other, False)) for other in names if other != planned]
    return out


def t8b_event_pairs(records: Sequence[Mapping[str, Any]]) -> list[Pair]:
    """T8b: every pair among a record's planned and counterfactual questions about the same event."""
    out: list[Pair] = []
    for record in records:
        rid = str(record["id"])
        out += [(rid, (rid, a, False), (rid, b, False)) for a, b in itertools.combinations(_t8b_names(record), 2)]
    return out


def _flip(value: float, negated: bool) -> float:
    return 1.0 - value if negated else value


def order_items(set_name: str, records: Sequence[Mapping[str, Any]], predictions: Sequence[Mapping[str, Any]], pairs: Iterable[Pair], *,
                min_move: float, reading: str = "single_order") -> Items:
    """Pairs whose label moves at least `min_move` (a → b, on the event's probability): value = the weight of the readings in which the
    answer moves the same way (strictly; an unchanged answer is a miss). Pairs with a missing answer are left out."""
    by_id = _by_id(records)
    predicted = _by_id(predictions)
    out: Items = {}
    for cluster, (ra, qa, na), (rb, qb, nb) in pairs:
        ya = _scalar((by_id[ra].get("labels") or {}).get(qa))
        yb = _scalar((by_id[rb].get("labels") or {}).get(qb))
        if ya is None or yb is None:
            continue
        moved = _flip(yb, nb) - _flip(ya, na)
        if abs(moved) < min_move - _EPS or moved == 0:
            continue
        pa = ((predicted.get(ra) or {}).get("questions") or {}).get(qa)
        pb = ((predicted.get(rb) or {}).get("questions") or {}).get(qb)
        if not isinstance(pa, Mapping) or not isinstance(pb, Mapping):
            continue
        value = 0.0
        complete = True
        for xa, xb, weight in _joint(pa, pb, reading):
            sa, sb = _scalar(xa), _scalar(xb)
            if sa is None or sb is None:
                complete = False
                break
            change = _flip(sb, nb) - _flip(sa, na)
            if change != 0 and (change > 0) == (moved > 0):
                value += weight
        if complete:
            out[(set_name, ra, qa, rb, qb)] = (cluster, value)
    return out


def set_items(set_name: str, records: Sequence[Mapping[str, Any]], predictions: Sequence[Mapping[str, Any]], *,
              reading: str = "single_order") -> dict[str, Items]:
    """Every System-1 metric of one set → its items (metrics without items are absent)."""
    out = dict(question_items(set_name, records, predictions, reading=reading))
    direction = t5_direction_pairs(records) + t8b_direction_pairs(records)
    comparison = basis_event_pairs(records) + t8b_event_pairs(records)
    quantity = basis_quantity_pairs(records)
    for metric, pairs, threshold in (("direction", direction, DIRECTION_MIN), ("direction@0.05", direction, T8_REPORT_MOVE),
                                     ("comparison", comparison, COMPARISON_MIN), ("comparison:quantity", quantity, COMPARISON_MIN)):
        if pairs:
            items = order_items(set_name, records, predictions, pairs, min_move=threshold, reading=reading)
            if items:
                out[metric] = items
    return out


def group_items(per_set: Mapping[str, Mapping[str, Items]], sets: Iterable[str]) -> dict[str, Items]:
    """Pool the items of several sets (keys carry the set name)."""
    out: dict[str, Items] = defaultdict(dict)
    for set_name in sets:
        for metric, items in (per_set.get(set_name) or {}).items():
            out[metric].update(items)
    return dict(out)


# --------------------------------------------------------------------------
# Aggregation and intervals
# --------------------------------------------------------------------------


def _interval(samples: np.ndarray) -> tuple[float | None, float | None]:
    samples = samples[np.isfinite(samples)]
    if not samples.size:
        return None, None
    return float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))


def _draws(clusters: int, n_boot: int, seed: int) -> np.ndarray:
    return np.random.default_rng(int(seed)).integers(0, clusters, size=(int(n_boot), clusters))


def _arrays(keys: Sequence[tuple[str, ...]], sources: Sequence[Items], stratified: bool) -> tuple[list[np.ndarray], np.ndarray, int]:
    """Cluster × stratum sums of each source's values over `keys`, the item counts, and the number of clusters (the stratum is
    ``key[0]`` when `stratified`, else one stratum)."""
    strata = sorted({key[0] for key in keys}) if stratified else [None]
    s_index = {stratum: i for i, stratum in enumerate(strata)}
    clusters = sorted({sources[0][key][0] for key in keys})
    c_index = {cluster: i for i, cluster in enumerate(clusters)}
    sums = [np.zeros((len(clusters), len(strata))) for _ in sources]
    counts = np.zeros((len(clusters), len(strata)))
    for key in keys:
        i, j = c_index[sources[0][key][0]], s_index[key[0] if stratified else None]
        counts[i, j] += 1
        for total, items in zip(sums, sources):
            total[i, j] += items[key][1]
    return sums, counts, len(clusters)


def _mean_over_strata(sums: np.ndarray, counts: np.ndarray) -> np.ndarray:
    """Σ over the leading cluster axis is done by the caller: arrays [..., strata] → the mean over strata of sum / count (strata with
    no items in a draw are left out)."""
    valid = counts > 0
    ratio = np.where(valid, sums / np.where(valid, counts, 1.0), 0.0)
    return ratio.sum(-1) / np.maximum(valid.sum(-1), 1)


def summarize(items: Items, *, n_boot: int = 0, seed: int = 0, stratified: bool = False) -> dict[str, Any]:
    """Mean value over items (each item counts once) with its cluster-bootstrap 95 % interval when `n_boot` > 0. `stratified`: the
    mean over strata (``key[0]``) of the stratum means — balanced accuracy."""
    if not items:
        return {"value": None, "n": 0, "clusters": 0}
    (sums,), counts, clusters = _arrays(list(items), [items], stratified)
    out: dict[str, Any] = {"value": float(_mean_over_strata(sums.sum(0), counts.sum(0))), "n": len(items), "clusters": clusters}
    if n_boot:
        draws = _draws(clusters, n_boot, seed)
        out["low"], out["high"] = _interval(_mean_over_strata(sums[draws].sum(1), counts[draws].sum(1)))
    return out


def paired_difference(items_a: Items, items_b: Items, *, n_boot: int = 2000, seed: int = 0, stratified: bool = False) -> dict[str, Any] | None:
    """a − b on the items both have (same keys), with a paired cluster-bootstrap 95 % interval (same draws for both)."""
    keys = sorted(set(items_a) & set(items_b))
    if not keys:
        return None
    (sa, sb), counts, clusters = _arrays(keys, [items_a, items_b], stratified)
    a, b = float(_mean_over_strata(sa.sum(0), counts.sum(0))), float(_mean_over_strata(sb.sum(0), counts.sum(0)))
    out: dict[str, Any] = {"value": a - b, "a": a, "b": b, "n": len(keys), "clusters": clusters, "low": None, "high": None}
    if n_boot:
        draws = _draws(clusters, n_boot, seed)
        n = counts[draws].sum(1)
        out["low"], out["high"] = _interval(_mean_over_strata(sa[draws].sum(1), n) - _mean_over_strata(sb[draws].sum(1), n))
    out["excludes_zero"] = bool(out["low"] is not None and (out["low"] > 0 or out["high"] < 0))
    return out


# --------------------------------------------------------------------------
# References
# --------------------------------------------------------------------------


def oracle_predictions(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """The labels as answers (one listing per question) — every positive metric is 1 and every error rate 0; its item counts are the
    denominators of coverage."""
    out = []
    for record in records:
        questions = {}
        for name, question in record["request"]["questions"].items():
            label = (record.get("labels") or {}).get(name)
            if not isinstance(label, Mapping):
                continue
            qtype = str(question.get("type"))
            if qtype == "choice":
                answer: dict[str, Any] = {"probabilities": {k: float(v) for k, v in label["probabilities"].items()}}
                options = list(question.get("criteria") or label["probabilities"])
            else:
                answer = {"p_yes": _scalar(label)} if qtype == "noul" else {"value": _scalar(label), "bins": None}
                options = None
            questions[name] = {"type": qtype, "n_listings": 1, "answer": answer,
                               "listings": [{"options": options, "values": None, "answer": answer}]}
        out.append({"id": str(record["id"]), "model": "oracle", "questions": questions})
    return out


def label_profile(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Yes/no and score labels of a set of records: count, share per band, share of clear labels — the base rates behind band accuracy
    (answering "low" every time scores the low share)."""
    bands: dict[str, int] = defaultdict(int)
    clear = total = 0
    for record in records:
        for name, question in record["request"]["questions"].items():
            if question.get("type") not in ("noul", "score"):
                continue
            y = _scalar((record.get("labels") or {}).get(name))
            if y is None:
                continue
            total += 1
            bands[band(y)] += 1
            clear += is_clear(y)
    return {"questions": total, "low": bands["low"] / total if total else None, "middle": bands["middle"] / total if total else None,
            "high": bands["high"] / total if total else None, "clear": clear / total if total else None}


# --------------------------------------------------------------------------
# Scoring every model
# --------------------------------------------------------------------------

#: Metrics shown in the scorecard and paired (report order); a group shows those that have items.
SCORECARD_METRICS = (
    "band_accuracy", "band_balanced", "band_accuracy_clear", "severe_error", "severe_error_clear", "direction", "comparison", "safe_choice",
    "catastrophic_choice", "outcome", "outcome:all_families", "choice_accuracy",
)
#: Breakdowns kept in the JSON (and a detail table).
DETAIL_METRICS = (
    "band_accuracy:noul", "band_accuracy:score", "band_recall:low", "band_recall:middle", "band_recall:high", "band_balanced:event",
    "severe_error:event", "severe_miss", "severe_false_alarm", "severe_miss_clear", "comparison:quantity",
    "safe_choice:default", "safe_choice:stated", "safe_choice:look_again", "catastrophic_choice:default", "catastrophic_choice:stated",
    "catastrophic_choice:look_again", "direction@0.05",
)
ROLES = ("baseline", "ours", "api", "reference")
#: Metrics averaged over label bands (key[0] = the label's band): balanced accuracy.
STRATIFIED = frozenset({"band_balanced", "band_balanced:event"})


def _better(metric: str, a: float, b: float) -> bool:
    return a < b if metric in LOWER_IS_BETTER else a > b


def _physics_skill(records_by_set: Mapping[str, Sequence[Mapping[str, Any]]], predictions: Mapping[str, Sequence[Mapping[str, Any]]],
                   sets: Iterable[str], reading: str) -> float | None:
    """Reference: the Brier physics skill (bench-v0 / T8 definition) over the sets this model answered — the metric being replaced."""
    from robo_jev.outside.score import pooled_skill, skill_rows

    rows = []
    for set_name in sets:
        if set_name in predictions and set_name in records_by_set:
            rows += skill_rows(list(records_by_set[set_name]), list(predictions[set_name]), reading=reading)
    pooled = pooled_skill(rows)
    return pooled["skill"] if pooled else None


def score(sets: Mapping[str, Sequence[Mapping[str, Any]]], models: Sequence[Mapping[str, Any]], groups: Mapping[str, Sequence[str]], *,
          n_boot: int = 2000, seed: int = 0) -> dict[str, Any]:
    """Every model on every set and group.

    `models`: ``{"tag", "label", "role" (baseline | ours | api | reference), "predictions": {set: [prediction, …]}}`` in report order;
    exactly one ``baseline``. Returns per-set values, the group scorecard (single-order values with cluster-bootstrap intervals,
    order-averaged values, coverage against the oracle, the Brier physics skill as a reference), the best API model per group and
    metric, and paired differences (ours and the baseline vs the baseline and vs the best API model)."""
    oracle_items = {name: set_items(name, records, oracle_predictions(records)) for name, records in sets.items()}
    items: dict[str, dict[str, dict[str, dict[str, Items]]]] = {}
    for model in models:
        items[model["tag"]] = {}
        for reading in READINGS:
            items[model["tag"]][reading] = {name: set_items(name, sets[name], preds, reading=reading)
                                            for name, preds in model["predictions"].items() if name in sets}
    results: dict[str, Any] = {
        "definitions": DEFINITIONS, "groups": {g: list(s) for g, s in groups.items()}, "n_boot": int(n_boot), "seed": int(seed),
        "constants": {"band_edges": list(BAND_EDGES), "clear_margin": CLEAR_MARGIN, "direction_min": DIRECTION_MIN, "comparison_min": COMPARISON_MIN,
                      "safe_regret": SAFE_REGRET, "catastrophic_regret": CATASTROPHIC_REGRET},
        "label_profile": {g: label_profile([r for s in members if s in sets for r in sets[s]]) for g, members in groups.items()},
        "models": [{k: model[k] for k in ("tag", "label", "role")} | {"sets": sorted(model["predictions"])} for model in models],
        "per_set": {}, "scorecard": {}, "best_api": {}, "paired": {}, "checks": {"oracle": {}},
    }
    metrics_all = SCORECARD_METRICS + DETAIL_METRICS
    for set_name in sets:
        entry = {}
        for model in models:
            found = items[model["tag"]]["single_order"].get(set_name)
            if found is None:
                continue
            averaged = items[model["tag"]]["order_averaged"].get(set_name) or {}
            entry[model["tag"]] = {metric: {"value": summarize(found[metric], stratified=metric in STRATIFIED)["value"], "n": len(found[metric]),
                                            "order_averaged": summarize(averaged.get(metric) or {}, stratified=metric in STRATIFIED)["value"]}
                                   for metric in metrics_all if metric in found}
        results["per_set"][set_name] = entry
    for group, members in groups.items():
        oracle = group_items(oracle_items, members)
        results["checks"]["oracle"][group] = {metric: summarize(found, stratified=metric in STRATIFIED)["value"] for metric, found in oracle.items()}
        card: dict[str, Any] = {}
        pooled: dict[str, dict[str, dict[str, Items]]] = {}
        for model in models:
            if not any(s in model["predictions"] for s in members):
                continue
            single = group_items(items[model["tag"]]["single_order"], members)
            averaged = group_items(items[model["tag"]]["order_averaged"], members)
            pooled[model["tag"]] = {"single_order": single, "order_averaged": averaged}
            card[model["tag"]] = {
                "single_order": {m: summarize(single[m], n_boot=n_boot if m in SCORECARD_METRICS else 0, seed=seed, stratified=m in STRATIFIED)
                                 for m in metrics_all if m in single},
                "order_averaged": {m: summarize(averaged[m], stratified=m in STRATIFIED)["value"] for m in metrics_all if m in averaged},
                "coverage": {m: len(single.get(m) or {}) / len(oracle[m]) for m in metrics_all if oracle.get(m)},
                "physics_skill": {r: _physics_skill(sets, model["predictions"], members, r) for r in READINGS},
            }
        results["scorecard"][group] = card
        best: dict[str, str] = {}
        for metric in SCORECARD_METRICS:
            rivals = [(tag, card[tag]["single_order"][metric]["value"]) for tag in card
                      if next(m for m in models if m["tag"] == tag)["role"] == "api" and metric in card[tag]["single_order"]
                      and card[tag]["single_order"][metric]["value"] is not None]
            if rivals:
                top = rivals[0]
                for tag, value in rivals[1:]:
                    if _better(metric, value, top[1]):
                        top = (tag, value)
                best[metric] = top[0]
        results["best_api"][group] = best
        base = next((m["tag"] for m in models if m["role"] == "baseline" and m["tag"] in pooled), None)
        paired: dict[str, Any] = {}
        for metric in SCORECARD_METRICS:
            entry: dict[str, Any] = {"vs_base": {}, "vs_best_api": {}, "best_api": best.get(metric)}
            for model in models:
                tag = model["tag"]
                if model["role"] not in ("ours", "baseline") or tag not in pooled or metric not in pooled[tag]["single_order"]:
                    continue
                mine = pooled[tag]["single_order"][metric]
                if model["role"] == "ours" and base and metric in pooled[base]["single_order"]:
                    entry["vs_base"][tag] = paired_difference(mine, pooled[base]["single_order"][metric], n_boot=n_boot, seed=seed,
                                                              stratified=metric in STRATIFIED)
                rival = best.get(metric)
                if rival and metric in pooled[rival]["single_order"]:
                    entry["vs_best_api"][tag] = paired_difference(mine, pooled[rival]["single_order"][metric], n_boot=n_boot, seed=seed,
                                                                  stratified=metric in STRATIFIED)
            if entry["vs_base"] or entry["vs_best_api"]:
                paired[metric] = entry
        results["paired"][group] = paired
    return results


# --------------------------------------------------------------------------
# Markdown
# --------------------------------------------------------------------------

METRIC_LABELS = {
    "physics_skill": "Brier skill (ref.)", "band_accuracy": "band acc ↑", "band_balanced": "band acc, balanced ↑", "band_accuracy_clear": "band acc, clear ↑", "severe_error": "severe ↓",
    "severe_error_clear": "severe, clear ↓", "direction": "direction ↑", "comparison": "comparison ↑", "safe_choice": "safe choice ↑",
    "catastrophic_choice": "catastrophic ↓", "outcome": "outcome (deck, 3-way) ↑", "outcome:all_families": "outcome (all families) ↑",
    "choice_accuracy": "accuracy ↑",
}
GROUP_TITLES = {"home_T2-T4": "Home ground — T2–T4 (new style / ranges / combination)", "home_T5": "Home ground — T5 (one-fact pairs)",
                "outside_T8b": "Outside — T8b (MuJoCo-labelled robot moments)", "outside_T8a": "Outside — T8a (PIQA)"}


def _fmt(value: Any, digits: int = 3, signed: bool = False) -> str:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "—"
    return f"{value:+.{digits}f}" if signed else f"{value:.{digits}f}"


def _cell(summary: Mapping[str, Any] | None, *, ci: bool = False) -> str:
    if not summary or summary.get("value") is None:
        return "—"
    text = _fmt(summary["value"])
    if ci and summary.get("low") is not None:
        text += f" [{_fmt(summary['low'])}, {_fmt(summary['high'])}]"
    return text


def _diff(cell: Mapping[str, Any] | None) -> str:
    if not cell:
        return "—"
    mark = "" if cell.get("excludes_zero") else " ·"
    return f"{_fmt(cell['value'], signed=True)} [{_fmt(cell['low'], signed=True)}, {_fmt(cell['high'], signed=True)}]{mark}"


def render_markdown(results: Mapping[str, Any], *, title: str = "System-1 scorecard — bench-v0 and T8 re-scored") -> str:
    """Compact report: label profile, scorecard per group, paired intervals, breakdowns, checks."""
    labels = {m["tag"]: m["label"] for m in results["models"]}
    roles = {m["tag"]: m["role"] for m in results["models"]}
    lines = [f"# {title}", "",
             "Existing predictions only (no model or API call). Single-order reading (deployment condition): our models' values are expectations "
             "over balanced option orders; OpenRouter models were asked once in the stored order. Intervals are cluster-bootstrap 95 % "
             f"({results['n_boot']} draws; cluster = record, a T5 pair is one cluster). Definitions at the end.", ""]
    lines += ["## Label profile (yes/no and score questions)", "",
              "Base rates behind band accuracy: answering \"low\" every time scores the low share.", "",
              "| group | questions | low | middle | high | clear |", "|---|---|---|---|---|---|"]
    for group, profile in results["label_profile"].items():
        if profile["questions"]:
            lines.append(f"| {group} | {profile['questions']} | {_fmt(profile['low'], 2)} | {_fmt(profile['middle'], 2)} | {_fmt(profile['high'], 2)} | "
                         f"{_fmt(profile['clear'], 2)} |")
    lines.append("")
    for group, card in results["scorecard"].items():
        if not card:
            continue
        metrics = [m for m in SCORECARD_METRICS if any(m in entry["single_order"] for entry in card.values())]
        skill = group != "outside_T8a"
        lines += [f"## {GROUP_TITLES.get(group, group)}", ""]
        header = ["model", "answered"] + (["Brier skill (ref.)"] if skill else []) + [METRIC_LABELS.get(m, m) for m in metrics]
        lines += ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
        for tag, entry in card.items():
            coverage = entry["coverage"].get("band_accuracy", entry["coverage"].get("choice_accuracy"))
            name = f"**{labels[tag]}**" if roles[tag] in ("ours", "baseline") else (f"_{labels[tag]}_" if roles[tag] == "reference" else labels[tag])
            cells = [name, _fmt(coverage, 2)] + ([_fmt(entry["physics_skill"]["single_order"])] if skill else [])
            cells += [_cell(entry["single_order"].get(m), ci=m in ("band_balanced", "direction", "comparison", "safe_choice", "outcome", "choice_accuracy"))
                      for m in metrics]
            lines.append("| " + " | ".join(cells) + " |")
        best = results["best_api"].get(group) or {}
        if best:
            lines += ["", "Best OpenRouter model per metric: " + "; ".join(f"{METRIC_LABELS.get(m, m)} {labels.get(t, t)}" for m, t in best.items() if m in metrics) + "."]
        ours = [t for t in card if roles[t] in ("ours", "baseline")]
        averaged_metrics = [m for m in metrics if any(m in card[t]["order_averaged"] for t in ours)]
        if ours and averaged_metrics:
            lines += ["", "Order-averaged reading (ours; what the model knows once position bias is averaged out):", "",
                      "| model | " + " | ".join(METRIC_LABELS.get(m, m) for m in averaged_metrics) + " |", "|---|" + "---|" * len(averaged_metrics)]
            for tag in ours:
                lines.append(f"| {labels[tag]} | " + " | ".join(_fmt(card[tag]["order_averaged"].get(m)) for m in averaged_metrics) + " |")
        lines.append("")
    per_set = results.get("per_set") or {}
    shown_sets = [name for name in per_set if per_set[name] and any("band_balanced" in v for v in per_set[name].values())]
    if shown_sets:
        lines += ["## Per set (single order)", "", "Every bench-v0 set and T8b; the groups above pool T2–T4.", ""]
        for metric in ("band_balanced", "severe_error", "comparison", "safe_choice"):
            lines += [f"**{METRIC_LABELS.get(metric, metric)}**", "", "| model | " + " | ".join(shown_sets) + " |", "|---|" + "---|" * len(shown_sets)]
            for model in results["models"]:
                cells = [_fmt(((per_set[name].get(model["tag"]) or {}).get(metric) or {}).get("value")) for name in shown_sets]
                if any(c != "—" for c in cells):
                    lines.append(f"| {model['label']} | " + " | ".join(cells) + " |")
            lines.append("")
    lines += ["## Paired intervals", "", "Δ = model − rival on the items both answered, cluster-bootstrap 95 %; `·` marks an interval that includes 0. "
              "For error rates (severe, catastrophic) a negative Δ is better.", ""]
    for group, paired in results["paired"].items():
        if not paired:
            continue
        tags = [m["tag"] for m in results["models"] if any(m["tag"] in (p["vs_base"] | p["vs_best_api"]) for p in paired.values())]
        lines += [f"### {GROUP_TITLES.get(group, group)}", "", "| metric | comparison | " + " | ".join(labels[t] for t in tags) + " |",
                  "|---|---|" + "---|" * len(tags)]
        for metric, entry in paired.items():
            if entry["vs_base"]:
                lines.append(f"| {METRIC_LABELS.get(metric, metric)} | vs untrained base | " + " | ".join(_diff(entry["vs_base"].get(t)) for t in tags) + " |")
            if entry["vs_best_api"]:
                rival = labels.get(entry.get("best_api"), entry.get("best_api"))
                lines.append(f"| {METRIC_LABELS.get(metric, metric)} | vs best API ({rival}) | " + " | ".join(_diff(entry["vs_best_api"].get(t)) for t in tags) + " |")
        lines.append("")
    lines += ["## Breakdowns (single order)", ""]
    for group, card in results["scorecard"].items():
        details = [m for m in DETAIL_METRICS if any(m in entry["single_order"] for entry in card.values())]
        if not details:
            continue
        lines += [f"### {group}", "", "| model | " + " | ".join(details) + " |", "|---|" + "---|" * len(details)]
        for tag, entry in card.items():
            lines.append(f"| {labels[tag]} | " + " | ".join(
                f"{_fmt((entry['single_order'].get(m) or {}).get('value'))} (n={(entry['single_order'].get(m) or {}).get('n', 0)})" for m in details) + " |")
        lines.append("")
    lines += ["## Checks", "", "Oracle (labels as answers) per group — positive metrics must be 1 and error rates 0:", ""]
    for group, values in results["checks"]["oracle"].items():
        shown = {m: v for m, v in values.items() if m in SCORECARD_METRICS}
        lines.append(f"- {group}: " + ", ".join(f"{m} {_fmt(v)}" for m, v in shown.items()))
    lines += ["", "## Definitions", ""] + [f"- **{k}** — {v}" for k, v in results["definitions"].items()]
    return "\n".join(lines) + "\n"
