"""T8 scoring (Task T8 §1-3): pooled physics skill, paired intervals, PIQA accuracy, the static reference."""

import pytest

from robo_jev.outside import moments as mo
from robo_jev.outside.piqa import piqa_record
from robo_jev.outside.reference import static_outcome, static_prediction
from robo_jev.outside.score import paired_skill, piqa_metrics, piqa_paired, pooled_skill, skill_rows, t8b_metrics


def _prediction(record, value_fn):
    questions = {}
    for name, q in record["request"]["questions"].items():
        label = record["labels"][name]
        if q["type"] == "choice":
            answer = {"probabilities": dict(label["probabilities"])}
            options = list(q["criteria"])
        else:
            y = label.get("p_yes", label.get("value"))
            answer = {"p_yes": value_fn(y)} if q["type"] == "noul" else {"value": value_fn(y), "bins": None}
            options = None
        listing = {"options": options, "values": None, "codes": None, "code_probs": None, "code_mass": None, "answer": answer}
        questions[name] = {"type": q["type"], "n_listings": 1, "answer": answer, "code_mass": None, "listings": [listing]}
    return {"id": record["id"], "model": "test", "questions": questions}


@pytest.fixture(scope="module")
def t8b_records():
    out = []
    for family in mo.FAMILIES:
        for index in range(2):
            moment = mo.candidate(family, index, 20261012)
            out.append(mo.moment_record(moment, mo.label_moment(moment, k=4), number=index, mujoco_version="test"))
    return out


def test_skill_is_one_for_the_labels_and_zero_for_one_half(t8b_records):
    perfect = [_prediction(r, lambda y: y) for r in t8b_records]
    half = [_prediction(r, lambda y: 0.5) for r in t8b_records]
    assert pooled_skill(skill_rows(t8b_records, perfect, reading="single_order"))["skill"] == pytest.approx(1.0)
    assert pooled_skill(skill_rows(t8b_records, half, reading="single_order"))["skill"] == pytest.approx(0.0)
    diff = paired_skill(skill_rows(t8b_records, perfect, reading="single_order"), skill_rows(t8b_records, half, reading="single_order"), n_boot=200)
    assert diff["value"] == pytest.approx(1.0) and diff["low"] == pytest.approx(1.0) and diff["excludes_zero"]
    metrics = t8b_metrics(t8b_records, perfect, n_boot=200)
    assert metrics["single_order"]["coverage"] == 1.0 and metrics["single_order"]["outcome_accuracy"] == 1.0
    assert set(metrics["single_order"]["by_family"]) == set(mo.FAMILIES)


def test_missing_answers_lower_coverage_not_skill(t8b_records):
    perfect = [_prediction(r, lambda y: y) for r in t8b_records]
    partial = perfect[: len(perfect) // 2]
    metrics = t8b_metrics(t8b_records, partial, n_boot=100)
    assert metrics["single_order"]["coverage"] < 1.0
    assert metrics["single_order"]["pooled"]["skill"] == pytest.approx(1.0)


def test_piqa_accuracy_and_paired_difference():
    rows = [{"goal": f"goal {i}", "sol1": "a", "sol2": "b"} for i in range(40)]
    records = [piqa_record(i, row, i % 2) for i, row in enumerate(rows)]

    def answer(record, right):
        correct = max(record["labels"]["solution"]["probabilities"], key=record["labels"]["solution"]["probabilities"].get)
        pick = correct if right else ("sol2" if correct == "sol1" else "sol1")
        probs = {"sol1": 0.9 if pick == "sol1" else 0.1, "sol2": 0.9 if pick == "sol2" else 0.1}
        listing = {"options": ["sol1", "sol2"], "values": None, "codes": None, "code_probs": None, "code_mass": None, "answer": {"probabilities": probs}}
        return {"id": record["id"], "model": "t", "questions": {"solution": {"type": "choice", "n_listings": 1, "answer": {"probabilities": probs},
                                                                           "code_mass": None, "listings": [listing]}}}

    good = [answer(r, True) for r in records]
    mixed = [answer(r, i % 4 != 0) for i, r in enumerate(records)]
    assert piqa_metrics(records, good, n_boot=100)["single_order"]["accuracy"] == 1.0
    assert piqa_metrics(records, mixed, n_boot=100)["single_order"]["accuracy"] == pytest.approx(0.75)
    diff = piqa_paired(records, good, mixed, n_boot=200)["single_order"]["accuracy"]
    assert diff["diff"] == pytest.approx(0.25) and diff["low"] > 0


def test_static_reference_follows_the_closed_form_rules(t8b_records):
    moment = mo.Moment("deck_brake", 0, 0, "tip", {"dims": [0.2, 0.3, 0.6]}, {}, {"planned": {"kind": "brake", "speed": 2.0, "decel": 3.0}})
    # a_tip = g·0.1/0.3 = 3.27 > 3.0 → stays with grip; tips at 3.5 with grip; slides when μg < a_tip and below the brake rate
    assert static_outcome(moment, "planned", {"mu": 1.0, "kh": 0.5, "cx": 0.0}) == "stay"
    moment.variants["planned"]["decel"] = 3.5
    assert static_outcome(moment, "planned", {"mu": 1.0, "kh": 0.5, "cx": 0.0}) == "tip"
    assert static_outcome(moment, "planned", {"mu": 0.2, "kh": 0.5, "cx": 0.0}) == "slide"
    for record in t8b_records:
        prediction = static_prediction(record)
        assert set(prediction["questions"]) == set(record["request"]["questions"])
        for name, q in prediction["questions"].items():
            answer = q["answer"]
            if "p_yes" in answer:
                assert 0.0 <= answer["p_yes"] <= 1.0
