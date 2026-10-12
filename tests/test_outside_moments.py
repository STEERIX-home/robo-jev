"""T8b moments (Task T8 §1-2): records, text, independence from the simgen engine, balance."""

import ast
import json
from pathlib import Path

import pytest

from robo_jev.decisions.records import validate_record
from robo_jev.outside import moments as mo

ROOT = Path(__file__).resolve().parents[1] / "src" / "robo_jev" / "outside"


@pytest.fixture(scope="module")
def records():
    out = []
    for family in mo.FAMILIES:
        moment = mo.candidate(family, 0, 20261012)
        label = mo.label_moment(moment, k=4)
        out.append((moment, label, mo.moment_record(moment, label, number=0, mujoco_version="test")))
    return out


def test_records_are_valid_decision_requests(records):
    for moment, label, record in records:
        assert validate_record(record) == []
        assert record["meta"]["scene"] == moment.family and record["meta"]["set"] == "T8b"
        types = {q["type"] for q in record["request"]["questions"].values()}
        assert "noul" in types and "score" in types
        for name, q in record["request"]["questions"].items():
            if q["type"] == "score":
                assert len(q["criteria"]) == 2  # the API's two-anchor scale
            assert record["meta"]["question_variant"][name] in moment.variants


def test_labels_are_outcome_frequencies(records):
    for moment, label, record in records:
        planned = label["variants"]["planned"]["freq"]
        assert abs(sum(planned.values()) - 1.0) < 1e-9
        labels = record["labels"]
        if moment.family.startswith("deck"):
            assert labels["falls_over"]["p_yes"] == planned["tip"]
            assert labels["keeps_place"]["p_yes"] == planned["stay"]
            assert labels["what_happens"]["probabilities"] == {"stays": planned["stay"], "slides": planned["slide"], "topples": planned["tip"]}
        elif moment.family == "stack":
            assert labels["falls_off"]["p_yes"] == planned["fall"]
        else:
            assert labels["tips_over"]["p_yes"] == planned["tip"]


def test_text_states_the_visible_numbers_and_hides_the_hidden_ones(records):
    for moment, label, record in records:
        state = record["request"]["state"]
        assert len(state) < 2000
        if moment.family == "deck_brake":
            assert f"{moment.visible['speed']:.2f} m/s" in state and f"{moment.visible['decel']:.2f} m/s²" in state
        for word in ("friction", "coefficient", "centre of mass", "hidden", "probability"):
            assert word not in state.lower()


def test_same_visible_scene_gives_the_same_text():
    moment = mo.candidate("stack", 3, 20261012)
    a = mo.label_moment(moment, k=4)
    b = dict(a, variants={name: dict(v, freq={o: 1.0 - p for o, p in v["freq"].items()}) for name, v in a["variants"].items()})
    ra = mo.moment_record(moment, a, number=0, mujoco_version="test")
    rb = mo.moment_record(moment, b, number=0, mujoco_version="test")
    assert ra["request"] == rb["request"] and ra["labels"] != rb["labels"]


def test_candidates_are_deterministic_and_cycle_their_targets():
    for family in mo.FAMILIES:
        a, b = mo.candidate(family, 5, 7), mo.candidate(family, 5, 7)
        assert a.visible == b.visible and a.seed == b.seed
        assert [mo.candidate(family, i, 7).target for i in range(len(mo.OUTCOMES[family]))] == list(mo.OUTCOMES[family])


def test_hidden_draws_are_stratified_and_seeded():
    ranges = {"mu": (0.2, 0.4), "eps": ("normal", 0.0, 0.01)}
    draws = mo.hidden_draws(ranges, 8, 123)
    assert draws == mo.hidden_draws(ranges, 8, 123)
    mus = sorted(d["mu"] for d in draws)
    for i, mu in enumerate(mus):  # Latin hypercube: one draw in each eighth of the range
        assert 0.2 + 0.025 * i <= mu <= 0.2 + 0.025 * (i + 1)


def test_t8_code_does_not_use_the_simgen_engine():
    for path in ROOT.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                assert not (node.module or "").startswith("robo_jev.simgen"), path
            elif isinstance(node, ast.Import):
                assert not any(alias.name.startswith("robo_jev.simgen") for alias in node.names), path


def test_balanced_selection_drops_unstable_moments():
    def fake(family, index, majority, unstable=0):
        return mo.Moment(family, index, index, majority, {}, {}, {"planned": {}}), {"majority": majority, "unstable": unstable}

    items = [fake("stack", i, "stay") for i in range(5)] + [fake("stack", 10 + i, "fall") for i in range(2)] + [fake("stack", 20, "fall", unstable=1)]
    kept, report = mo.select_balanced(items, {"stack": 4})
    assert [m.index for m, _ in kept] == [0, 1, 10, 11]
    assert report["stack"]["dropped_unstable"] == 1 and report["stack"]["kept"] == {"stay": 2, "fall": 2}
    kept, report = mo.select_balanced(items, {"stack": 6})  # falls run short → the slot goes to the other outcome
    assert report["stack"]["kept"] == {"stay": 4, "fall": 2} and report["stack"]["shortfall"] == 0
    assert json.dumps(report)
