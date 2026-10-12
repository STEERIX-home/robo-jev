"""System-1 metrics (Task s1score): every metric on fake predictions whose value is worked out by hand, the readings, the intervals,
and the `scripts/bench.py system1` wiring on a tiny definition (no model or API call)."""

import json

import pytest
import yaml

from robo_jev.bench import system1 as s1
from robo_jev.outside import moments as mo

# --------------------------------------------------------------------------
# fake records and predictions
# --------------------------------------------------------------------------


def _q(qtype, **extra):
    question = {"type": qtype, "instructions": "?"}
    if qtype == "choice":
        question["criteria"] = {k: k for k in extra.pop("keys")}
    return question | extra


def _label(qtype, value=None, **extra):
    if qtype == "noul":
        return {"type": "noul", "p_yes": value}
    if qtype == "score":
        return {"type": "score", "value": value}
    return {"type": "choice", **extra}


def _record(rid, questions, labels, meta=None):
    return {"id": rid, "request": {"state": "s", "questions": questions}, "labels": labels, "meta": meta or {}}


def _listing(qtype, answer, options=None):
    return {"options": options, "values": None, "answer": answer}


def _answer(qtype, value):
    if qtype == "noul":
        return {"p_yes": value}
    if qtype == "score":
        return {"value": value, "bins": None}
    return {"probabilities": value}


def _prediction(rid, answers):
    """answers: name → (type, value) for one listing, or (type, [(options, value), …]) for several listings."""
    questions = {}
    for name, (qtype, value) in answers.items():
        listings = [_listing(qtype, _answer(qtype, v), o) for o, v in value] if isinstance(value, list) else [_listing(qtype, _answer(qtype, value))]
        questions[name] = {"type": qtype, "n_listings": len(listings), "answer": listings[0]["answer"], "listings": listings}
    return {"id": rid, "model": "test", "questions": questions}


def _value(items):
    return s1.summarize(items)["value"]


# --------------------------------------------------------------------------
# bands
# --------------------------------------------------------------------------


def test_band_edges_and_clear_labels():
    assert [s1.band(x) for x in (0.0, 0.33, 1 / 3, 0.5, 2 / 3, 0.67, 1.0)] == ["low", "low", "middle", "middle", "middle", "high", "high"]
    assert not s1.is_clear(0.30) and not s1.is_clear(0.62) and not s1.is_clear(0.70)
    assert s1.is_clear(0.2) and s1.is_clear(0.5) and s1.is_clear(0.9) and s1.is_clear(0.0)


def _band_record():
    questions = {f"q{i}": _q(t) for i, t in enumerate(("noul", "noul", "score", "noul", "score"), start=1)}
    labels = {"q1": _label("noul", 0.1), "q2": _label("noul", 0.9), "q3": _label("score", 0.5), "q4": _label("noul", 0.05), "q5": _label("score", 0.30)}
    return _record("r1", questions, labels)


def test_band_accuracy_severe_errors_clear_cases_and_balance_by_hand():
    record = _band_record()
    # q1 low → low (right) · q2 high → low (severe miss) · q3 middle → high · q4 low → high (severe false alarm) · q5 low (not clear) → middle
    prediction = _prediction("r1", {"q1": ("noul", 0.2), "q2": ("noul", 0.1), "q3": ("score", 0.9), "q4": ("noul", 0.8), "q5": ("score", 0.35)})
    items = s1.question_items("T2", [record], [prediction])
    assert _value(items["band_accuracy"]) == pytest.approx(1 / 5)
    assert _value(items["severe_error"]) == pytest.approx(2 / 5)
    assert _value(items["severe_miss"]) == pytest.approx(1 / 5) and _value(items["severe_false_alarm"]) == pytest.approx(1 / 5)
    assert len(items["band_accuracy_clear"]) == 4 and _value(items["band_accuracy_clear"]) == pytest.approx(1 / 4)
    assert _value(items["severe_error_clear"]) == pytest.approx(2 / 4)
    # balanced: low band {q1 right, q4, q5} = 1/3, middle {q3} = 0, high {q2} = 0
    assert s1.summarize(items["band_balanced"], stratified=True)["value"] == pytest.approx(1 / 9)
    assert _value(items["band_recall:low"]) == pytest.approx(1 / 3) and _value(items["band_recall:high"]) == 0.0
    assert _value(items["band_accuracy:noul"]) == pytest.approx(1 / 3) and _value(items["band_accuracy:score"]) == 0.0


def test_single_order_weights_each_listing_and_order_averaged_reads_the_mean_answer():
    record = _record("r1", {"q": _q("noul")}, {"q": _label("noul", 0.1)})
    prediction = _prediction("r1", {"q": ("noul", [(["yes", "no"], 0.2), (["no", "yes"], 0.5)])})
    single = s1.question_items("T2", [record], [prediction], reading="single_order")
    averaged = s1.question_items("T2", [record], [prediction], reading="order_averaged")
    assert _value(single["band_accuracy"]) == pytest.approx(0.5)  # one listing low, one middle
    assert _value(averaged["band_accuracy"]) == 0.0  # mean 0.35 is middle


def test_answering_low_every_time_scores_the_low_share_but_one_third_balanced():
    record = _band_record()
    prediction = _prediction("r1", {name: (q["type"], 0.0) for name, q in record["request"]["questions"].items()})
    items = s1.question_items("T2", [record], [prediction])
    assert _value(items["band_accuracy"]) == pytest.approx(3 / 5)
    assert s1.summarize(items["band_balanced"], stratified=True)["value"] == pytest.approx(1 / 3)
    assert s1.label_profile([record]) == {"questions": 5, "low": 0.6, "middle": 0.2, "high": 0.2, "clear": 0.8}


# --------------------------------------------------------------------------
# direction (T5 numeric pairs)
# --------------------------------------------------------------------------


def _t5_pair(kind="numeric"):
    questions = {"n1": _q("noul"), "n2": _q("score"), "n3": _q("noul"), "c": _q("choice", keys=["a", "b"])}
    base = _record("p-base", questions, {"n1": _label("noul", 0.2), "n2": _label("score", 0.7), "n3": _label("noul", 0.50),
                                         "c": _label("choice", probabilities={"a": 1.0, "b": 0.0})},
                   {"pair": {"of": "p-var", "kind": kind, "role": "base"}})
    variant = _record("p-var", questions, {"n1": _label("noul", 0.6), "n2": _label("score", 0.3), "n3": _label("noul", 0.55),
                                           "c": _label("choice", probabilities={"a": 0.0, "b": 1.0})},
                      {"pair": {"of": "p-base", "kind": kind, "role": "variant"}})
    return [base, variant]


def test_direction_counts_moved_labels_only_and_ties_miss():
    records = _t5_pair()
    predictions = [_prediction("p-base", {"n1": ("noul", 0.3), "n2": ("score", 0.6), "n3": ("noul", 0.5), "c": ("choice", {"a": 0.9, "b": 0.1})}),
                   _prediction("p-var", {"n1": ("noul", 0.5), "n2": ("score", 0.6), "n3": ("noul", 0.9), "c": ("choice", {"a": 0.1, "b": 0.9})})]
    pairs = s1.t5_direction_pairs(records)
    assert [(a[1], b[1]) for _, a, b in pairs] == [("n1", "n1"), ("n2", "n2"), ("n3", "n3")]  # choice questions are not direction items
    assert {cluster for cluster, _, _ in pairs} == {"pair:p-base"}
    items = s1.order_items("T5", records, predictions, pairs, min_move=s1.DIRECTION_MIN)
    # n1 moves up with the label (right) · n2 label down, answer unchanged (tie = miss) · n3 label moves 0.05 < 0.1 (left out)
    assert len(items) == 2 and _value(items) == pytest.approx(0.5)
    assert s1.t5_direction_pairs(_t5_pair("wording")) == []


def test_direction_single_order_matches_listings_by_option_order():
    records = _t5_pair()
    # the variant lists [no, yes] first; matched by option order both listings move up (+0.02) → 1.0; every combination would give 0.75
    predictions = [_prediction("p-base", {"n1": ("noul", [(["yes", "no"], 0.30), (["no", "yes"], 0.50)])}),
                   _prediction("p-var", {"n1": ("noul", [(["no", "yes"], 0.52), (["yes", "no"], 0.32)])})]
    pairs = [p for p in s1.t5_direction_pairs(records) if p[1][1] == "n1"]
    assert _value(s1.order_items("T5", records, predictions, pairs, min_move=0.1)) == pytest.approx(1.0)
    assert _value(s1.order_items("T5", records, predictions, pairs, min_move=0.1, reading="order_averaged")) == pytest.approx(1.0)
    # a yes/no and a score question show different option orders: every combination, 1/(2·2) each
    mixed = [_prediction("p-base", {"n1": ("noul", [(["yes", "no"], 0.30), (["no", "yes"], 0.50)])}),
             _prediction("p-var", {"n1": ("score", [(["0.0", "1.0"], 0.40), (["1.0", "0.0"], 0.60)])})]
    assert _value(s1.order_items("T5", records, mixed, pairs, min_move=0.1)) == pytest.approx(0.75)


# --------------------------------------------------------------------------
# comparison (same event, different actions, one record)
# --------------------------------------------------------------------------


def _comparison_record():
    basis = {
        "a": {"kind": "noul", "source": "event", "event": "contact", "option": "continue"},
        "b": {"kind": "noul", "source": "event", "event": "contact", "option": "slow"},
        "c": {"kind": "score", "source": "event", "name": "contact", "option": "stop_hard"},
        "d": {"kind": "noul", "source": "event", "event": "contact", "option": "detour", "negate": True},
        "e": {"kind": "noul", "source": "world", "event": "present"},
        "f": {"kind": "noul", "source": "event", "event": "spill", "option": "slow"},
        "g": {"kind": "score", "source": "quantity", "name": "delay", "option": "ramp_slow", "transform": "per_30s"},
        "h": {"kind": "score", "source": "quantity", "name": "delay", "option": "detour", "transform": "per_30s"},
    }
    types = {"a": "noul", "b": "noul", "c": "score", "d": "noul", "e": "noul", "f": "noul", "g": "score", "h": "score"}
    labels = {"a": 0.9, "b": 0.2, "c": 0.0, "d": 0.97, "e": 0.5, "f": 0.4, "g": 0.2, "h": 1.0}
    return _record("r-cmp", {n: _q(t) for n, t in types.items()}, {n: _label(types[n], labels[n]) for n in types}, {"basis": basis})


def test_comparison_pairs_by_recipe_identity_with_negation():
    record = _comparison_record()
    pairs = s1.basis_event_pairs([record])
    assert sorted((a[1], b[1]) for _, a, b in pairs) == [("a", "b"), ("a", "c"), ("a", "d"), ("b", "c"), ("b", "d"), ("c", "d")]
    assert {(a[1], a[2], b[1], b[2]) for _, a, b in pairs if "d" in (a[1], b[1])} >= {("a", False, "d", True)}
    answers = {"a": ("noul", 0.8), "b": ("noul", 0.3), "c": ("score", 0.4), "d": ("noul", 0.9), "e": ("noul", 0.5), "f": ("noul", 0.5),
               "g": ("score", 0.6), "h": ("score", 0.5)}
    items = s1.order_items("T2", [record], [_prediction("r-cmp", answers)], pairs, min_move=s1.COMPARISON_MIN)
    # event probabilities: d's label 0.97 is "no contact" → 0.03, its answer 0.9 → 0.1.
    # (a,b) right · (a,c) right · (b,c) labels 0.2 → 0.0 but answers 0.3 → 0.4 (wrong) · (a,d) right · (b,d) 0.2 → 0.03 vs 0.3 → 0.1 (right)
    # (c,d) labels 0.0 vs 0.03 differ by less than 0.15 (left out)
    assert len(items) == 5 and _value(items) == pytest.approx(4 / 5)
    quantity = s1.order_items("T2", [record], [_prediction("r-cmp", answers)], s1.basis_quantity_pairs([record]), min_move=s1.COMPARISON_MIN)
    assert len(quantity) == 1 and _value(quantity) == 0.0  # delay 0.2 → 1.0, answers 0.6 → 0.5
    bands = s1.question_items("T2", [record], [_prediction("r-cmp", answers)])
    assert len(bands["band_accuracy"]) == 8 and len(bands["band_balanced:event"]) == 6 and len(bands["severe_error:event"]) == 6  # no time costs


# --------------------------------------------------------------------------
# action choice
# --------------------------------------------------------------------------


def test_safe_and_catastrophic_choices_by_hand():
    keys = ["a1", "a2", "a3"]
    narrow = {"probabilities": {"a1": 0.8, "a2": 0.15, "a3": 0.05}, "expected_cost": {"a1": 0.0, "a2": 1.0, "a3": 10.0}, "kind": "action"}
    wide = {"probabilities": {"a1": 0.8, "a2": 0.15, "a3": 0.05}, "expected_cost": {"a1": 0.0, "a2": 6.0, "a3": 10.0}, "kind": "action"}
    look = {"probabilities": {"look_first": 0.3, "decide_now": 0.7}, "expected_cost": {"look_first": 1.5, "decide_now": 0.0}, "kind": "action"}
    questions = {"q1": _q("choice", keys=keys), "q2": _q("choice", keys=keys), "q3": _q("choice", keys=keys), "q4": _q("choice", keys=keys),
                 "v": _q("choice", keys=["look_first", "decide_now"])}
    labels = {"q1": _label("choice", **narrow), "q2": _label("choice", **narrow), "q3": _label("choice", **wide), "q4": _label("choice", **narrow),
              "v": _label("choice", **look)}
    basis = {name: {"kind": "action", "criterion": None} for name in ("q1", "q2", "q3")} | {"q4": {"kind": "action", "criterion": "avoid_contact"},
                                                                                            "v": {"kind": "voi", "criterion": None}}
    record = _record("r-act", questions, labels, {"basis": basis})
    pick = lambda k, ks=keys: {key: (0.9 if key == k else 0.1 / (len(ks) - 1)) for key in ks}  # noqa: E731
    prediction = _prediction("r-act", {
        "q1": ("choice", pick("a2")),                      # regret 1/10 = 0.1 → safe
        "q2": ("choice", pick("a3")),                      # the worst option → catastrophic
        "q3": ("choice", pick("a2")),                      # 6/10 → catastrophic, not safe
        "q4": ("choice", [(keys, pick("a1")), (list(reversed(keys)), pick("a3"))]),  # one listing safe, one catastrophic
        "v": ("choice", pick("look_first", ["look_first", "decide_now"])),           # the costlier of two → catastrophic
    })
    items = s1.question_items("T2", [record], [prediction])
    assert _value(items["safe_choice"]) == pytest.approx((1 + 0 + 0 + 0.5 + 0) / 5)
    assert _value(items["catastrophic_choice"]) == pytest.approx((0 + 1 + 1 + 0.5 + 1) / 5)
    assert _value(items["safe_choice:default"]) == pytest.approx(1 / 3) and _value(items["safe_choice:stated"]) == pytest.approx(0.5)
    assert _value(items["catastrophic_choice:look_again"]) == 1.0


# --------------------------------------------------------------------------
# T8b: outcome category, direction and comparison pairs
# --------------------------------------------------------------------------


def _t8b_records():
    deck_q = {"falls_over": _q("noul"), "keeps_place": _q("noul"), "what_happens": _q("choice", keys=["stays", "slides", "topples"]),
              "falls_other_braking": _q("score")}

    def deck(rid, outcome):
        labels = {"falls_over": _label("noul", 0.1), "keeps_place": _label("noul", 0.2),
                  "what_happens": _label("choice", kind="state", probabilities=outcome), "falls_other_braking": _label("score", 0.6)}
        return _record(rid, deck_q, labels, {"set": "T8b", "scene": "deck_brake"})

    stack_q = {"falls_off": _q("noul"), "falls_if_further_out": _q("score"), "falls_if_further_in": _q("noul")}
    push_q = {"tips_over": _q("noul"), "tips_at_other_height": _q("noul"), "tip_risk_other_height": _q("score")}
    records = [deck("d1", {"stays": 0.2, "slides": 0.7, "topples": 0.1}), deck("d2", {"stays": 0.2, "slides": 0.7, "topples": 0.1}),
               _record("s1", stack_q, {"falls_off": _label("noul", 0.7), "falls_if_further_out": _label("score", 0.9),
                                       "falls_if_further_in": _label("noul", 0.3)}, {"set": "T8b", "scene": "stack"}),
               _record("p1", push_q, {"tips_over": _label("noul", 0.2), "tips_at_other_height": _label("noul", 0.25),
                                      "tip_risk_other_height": _label("score", 0.8)}, {"set": "T8b", "scene": "push"}),
               _record("p2", push_q, {"tips_over": _label("noul", 0.5), "tips_at_other_height": _label("noul", 0.5),
                                      "tip_risk_other_height": _label("score", 0.5)}, {"set": "T8b", "scene": "push"})]
    deck_answer = {"falls_over": ("noul", 0.1), "keeps_place": ("noul", 0.9), "falls_other_braking": ("score", 0.3)}
    predictions = [_prediction("d1", deck_answer | {"what_happens": ("choice", {"stays": 0.1, "slides": 0.6, "topples": 0.3})}),
                   _prediction("d2", deck_answer | {"what_happens": ("choice", {"stays": 0.6, "slides": 0.3, "topples": 0.1})}),
                   _prediction("s1", {"falls_off": ("noul", 0.6), "falls_if_further_out": ("score", 0.8), "falls_if_further_in": ("noul", 0.2)}),
                   _prediction("p1", {"tips_over": ("noul", 0.5), "tips_at_other_height": ("noul", 0.5), "tip_risk_other_height": ("score", 0.5)}),
                   _prediction("p2", {"tips_over": ("noul", 0.5), "tips_at_other_height": ("noul", 0.5), "tip_risk_other_height": ("score", 0.5)})]
    return records, predictions


def test_t8b_outcome_category_by_hand():
    records, predictions = _t8b_records()
    items = s1.question_items("T8b", records, predictions)
    assert _value(items["outcome"]) == pytest.approx(1 / 2)  # d1 slides (right), d2 stays (wrong)
    # all families: d1 right, d2 wrong, s1 0.6 vs 0.7 (falls, right), p1 an answer of 0.5 (miss), p2 label 0.5 (any answer is right)
    assert _value(items["outcome:all_families"]) == pytest.approx(3 / 5)


def test_t8b_pairs_follow_the_t8_diagnostic():
    records, predictions = _t8b_records()
    direction = s1.t8b_direction_pairs(records)
    assert [(c, a[1], b[1]) for c, a, b in direction][:3] == [("d1", "falls_over", "falls_other_braking"), ("d2", "falls_over", "falls_other_braking"),
                                                              ("s1", "falls_off", "falls_if_further_out")]
    assert len(direction) == 1 + 1 + 2 + 2 + 2 and len(s1.t8b_event_pairs(records)) == 1 + 1 + 3 + 3 + 3
    items = s1.order_items("T8b", records, predictions, direction, min_move=s1.DIRECTION_MIN)
    # d1/d2 label 0.1 → 0.6, answers 0.1 → 0.3 (right, right) · s1 0.7 → 0.9 (0.6 → 0.8 right), 0.7 → 0.3 (0.6 → 0.2 right)
    # p1 0.2 → 0.25 (moves 0.05: left out), 0.2 → 0.8 (answers 0.5 → 0.5: miss) · p2 labels do not move (left out)
    assert len(items) == 5 and _value(items) == pytest.approx(4 / 5)
    assert all(name in mo.QUESTION_VARIANT for names in s1.T8B_EVENTS.values() for name in names)
    assert all(mo.QUESTION_VARIANT[names[0]] == "planned" and all(mo.QUESTION_VARIANT[n] != "planned" for n in names[1:])
               for names in s1.T8B_EVENTS.values())


# --------------------------------------------------------------------------
# aggregation, intervals, oracle, scoring
# --------------------------------------------------------------------------


def test_summary_and_paired_difference():
    a = {("T2", f"r{i}", "q"): (f"r{i}", 1.0) for i in range(20)}
    b = {("T2", f"r{i}", "q"): (f"r{i}", 0.0) for i in range(20)}
    b_more = b | {("T2", "x", "q"): ("x", 1.0)}
    summary = s1.summarize(a | {("T2", "r0", "q2"): ("r0", 0.0)}, n_boot=200)
    assert summary["value"] == pytest.approx(20 / 21) and summary["n"] == 21 and summary["clusters"] == 20 and summary["low"] <= summary["high"]
    diff = s1.paired_difference(a, b_more, n_boot=200)
    assert diff["value"] == 1.0 and diff["low"] == 1.0 and diff["high"] == 1.0 and diff["n"] == 20 and diff["excludes_zero"]
    same = s1.paired_difference(a, a, n_boot=200)
    assert same["value"] == 0.0 and same["low"] == 0.0 and not same["excludes_zero"]
    assert s1.paired_difference(a, {}, n_boot=10) is None and s1.summarize({})["value"] is None
    strata = {("low", "T2", "r1", "q"): ("r1", 1.0), ("low", "T2", "r2", "q"): ("r2", 1.0), ("low", "T2", "r3", "q"): ("r3", 1.0),
              ("high", "T2", "r4", "q"): ("r4", 0.0)}
    assert s1.summarize(strata, stratified=True)["value"] == pytest.approx(0.5) and s1.summarize(strata)["value"] == pytest.approx(0.75)


def test_oracle_is_perfect_on_every_metric():
    records = [_band_record(), _comparison_record(), *_t5_pair(), *_t8b_records()[0]]
    items = {}
    for set_name, recs in (("T2", records[:2]), ("T5", records[2:4]), ("T8b", records[4:])):
        items |= {(set_name, m): v for m, v in s1.set_items(set_name, recs, s1.oracle_predictions(recs)).items()}
    for (set_name, metric), found in items.items():
        expected = 0.0 if metric in s1.LOWER_IS_BETTER else 1.0
        assert s1.summarize(found, stratified=metric in s1.STRATIFIED)["value"] == pytest.approx(expected), (set_name, metric)
    assert {"band_accuracy", "comparison", "direction", "outcome", "outcome:all_families"} <= {m for _, m in items}


def test_score_picks_the_best_api_model_and_pairs_ours_with_it():
    records, predictions = _t8b_records()
    oracle = s1.oracle_predictions(records)
    half = [_prediction(r["id"], {n: (q["type"], 0.5 if q["type"] != "choice" else {k: 1 / 3 for k in q["criteria"]})
                                  for n, q in r["request"]["questions"].items()}) for r in records]
    models = [{"tag": "base", "label": "base", "role": "baseline", "predictions": {"T8b": half}},
              {"tag": "ours", "label": "ours", "role": "ours", "predictions": {"T8b": predictions}},
              {"tag": "api-good", "label": "api good", "role": "api", "predictions": {"T8b": oracle}},
              {"tag": "api-half", "label": "api half", "role": "api", "predictions": {"T8b": half[:2]}}]
    results = s1.score({"T8b": records}, models, {"outside_T8b": ["T8b"]}, n_boot=100)
    card = results["scorecard"]["outside_T8b"]
    assert card["api-good"]["single_order"]["band_accuracy"]["value"] == 1.0 and card["api-half"]["coverage"]["band_accuracy"] < 1.0
    assert results["best_api"]["outside_T8b"]["band_accuracy"] == "api-good" and results["best_api"]["outside_T8b"]["severe_error"] == "api-good"
    paired = results["paired"]["outside_T8b"]["direction"]
    assert paired["vs_best_api"]["ours"]["value"] == pytest.approx(4 / 5 - 1.0) and "ours" in paired["vs_base"]
    assert "base" in paired["vs_best_api"] and "base" not in paired["vs_base"]
    text = s1.render_markdown(results)
    assert "Outside — T8b" in text and "| **ours** |" in text and "vs best API (api good)" in text


# --------------------------------------------------------------------------
# scripts/bench.py system1
# --------------------------------------------------------------------------


def test_bench_script_system1_end_to_end(tmp_path):
    from test_bench_script import script

    from robo_jev.bench.openrouter import safe_name

    def write(path, rows):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")

    t5, t8b = _t5_pair(), _t8b_records()[0]
    write(tmp_path / "src" / "T5.jsonl", t5)
    write(tmp_path / "src" / "T8b.jsonl", t8b)
    bench = {"version": "bench-test", "seed": 1, "sets": {"T5": {"source": "src/T5.jsonl", "unit": "all"}},
             "models": {"default": [{"id": "vendor/a"}]},
             "ours": [{"tag": "base", "label": "base 2B", "role": "baseline", "T": ["preds/base-{set}.jsonl"]},
                      {"tag": "mine", "label": "mine 2B", "role": "ours", "T": ["preds/mine-{set}.jsonl"]}]}
    t8 = {"version": "t8-test", "seed": 1, "sets": {"T8b": {"source": "src/T8b.jsonl", "unit": "all"}}, "models": {"default": [{"id": "vendor/a"}]}}
    spec = {"version": "system1-test", "bench": "bench.yaml", "t8": "t8.yaml", "sets": ["T5", "T8b"], "groups": {"home_T5": ["T5"], "outside_T8b": ["T8b"]},
            "ours": [{"tag": "base", "T8": ["preds/base-{set}.jsonl"]}, {"tag": "mine"}, {"tag": "later", "label": "later", "T": ["missing/{set}.jsonl"]}],
            "reference": False}
    for name, payload in (("bench.yaml", bench), ("t8.yaml", t8), ("system1.yaml", spec)):
        (tmp_path / name).write_text(yaml.safe_dump(payload), encoding="utf-8")
    common = ["--config", str(tmp_path / "bench.yaml"), "--root", str(tmp_path)]
    assert script().main(common + ["build", "--out", str(tmp_path / "v0")]) == 0
    assert script().main(["--config", str(tmp_path / "t8.yaml"), "--root", str(tmp_path), "build", "--out", str(tmp_path / "t8")]) == 0
    write(tmp_path / "preds" / "base-T5.jsonl", s1.oracle_predictions(t5))
    write(tmp_path / "preds" / "mine-T5.jsonl", s1.oracle_predictions(t5))
    write(tmp_path / "preds" / "base-T8b.jsonl", s1.oracle_predictions(t8b))
    write(tmp_path / "v0" / "api" / safe_name("vendor/a") / "T5.predictions.jsonl", s1.oracle_predictions(t5))
    write(tmp_path / "t8" / "api" / safe_name("vendor/a") / "T8b.predictions.jsonl", s1.oracle_predictions(t8b)[:3])
    out = tmp_path / "s1" / "results.json"
    assert script().main(common + ["system1", "--spec", str(tmp_path / "system1.yaml"), "--bench-out", str(tmp_path / "v0"), "--t8-out",
                                   str(tmp_path / "t8"), "--results", str(out), "--n-boot", "50"]) == 0
    results = json.loads(out.read_text(encoding="utf-8"))
    assert [m["tag"] for m in results["models"]] == ["base", "mine", safe_name("vendor/a")]
    assert results["models"][1]["sets"] == ["T5"] and results["models"][0]["sets"] == ["T5", "T8b"]
    assert results["scorecard"]["home_T5"]["mine"]["single_order"]["direction"]["value"] == 1.0
    assert results["scorecard"]["outside_T8b"][safe_name("vendor/a")]["coverage"]["band_accuracy"] < 1.0
    assert results["kind"] == "system1-results" and set(results["set_ids_sha256"]) == {"T5", "T8b"}
    assert "Home ground — T5" in out.with_suffix(".md").read_text(encoding="utf-8")
