"""레코드 형식 `simgen-request/0` (Task v0a 지시서 §2·§5 '형식'): Decisions 제약, 정답 합, 열쇠 일치, 검사기가 위반을 잡는지."""

import copy

import pytest
from simgen_helpers import sample_records

from robo_jev.simgen import SCHEMA
from robo_jev.simgen.record import NAME_RE, TOP_KEYS, dumps, validate_record


@pytest.fixture(scope="module")
def records():
    return sample_records()


@pytest.fixture
def good(records):
    return copy.deepcopy(next(r for r in records if any(q["type"] == "score" for q in r["request"]["questions"].values())))


def test_every_generated_record_passes_the_validator(records):
    failures = {r["id"]: validate_record(r) for r in records if validate_record(r)}
    assert failures == {}


def test_records_have_the_documented_top_level_shape(records):
    for r in records[:200]:
        assert tuple(r) == TOP_KEYS
        assert r["schema"] == SCHEMA
        assert set(r["generator"]) >= {"version", "catalog", "values", "seed", "git"}
        assert set(r["meta"]) >= {"scene", "phenomena", "domain", "body", "sensor_suite", "style", "language", "units", "hint", "recommendation", "margins", "pair"}
        assert set(r["truth"]) == {"world", "realized"} and set(r["truth"]["realized"]) == set(r["request"]["questions"])
        assert isinstance(r["rationale"], str) and len(r["rationale"]) > 100


def test_decisions_constraints_hold_on_every_request(records):
    for r in records:
        state, questions = r["request"]["state"], r["request"]["questions"]
        assert 0 < len(state) <= 16_000
        assert 1 <= len(questions) <= 12
        for name, q in questions.items():
            assert NAME_RE.match(name)
            if q["type"] == "choice":
                assert 2 <= len(q["criteria"]) <= 8 and all(NAME_RE.match(k) for k in q["criteria"])
            if q["type"] == "score":
                assert isinstance(q["criteria"], list) and all(isinstance(c, str) for c in q["criteria"])


def test_label_probabilities_sum_to_one_and_keys_match(records):
    for r in records:
        for name, label in r["labels"].items():
            q = r["request"]["questions"][name]
            assert label["type"] == q["type"]
            if label["type"] == "choice":
                assert set(label["probabilities"]) == set(q["criteria"])
                assert sum(label["probabilities"].values()) == pytest.approx(1.0, abs=1e-6)
                if label["kind"] == "action":
                    assert set(label["expected_cost"]) == set(q["criteria"])
            elif label["type"] == "noul":
                assert 0.0 <= label["p_yes"] <= 1.0
            else:
                assert 0.0 <= label["value"] <= 1.0


def test_jsonl_line_is_single_line_utf8(records):
    for r in records[:200]:
        line = dumps(r)
        assert "\n" not in line and line.encode("utf-8")


@pytest.mark.parametrize(
    ("breakage", "message"),
    [
        (lambda r: r["request"].__setitem__("state", "x" * 16_001), "16,000"),
        (lambda r: r["request"]["questions"].update({f"extra_{i}": {"type": "noul", "instructions": "?"} for i in range(13)}), "1–12"),
        (lambda r: r["request"]["questions"].update({"bad-name": {"type": "noul", "instructions": "?"}}), "bad name"),
        (lambda r: r["labels"][next(iter(r["labels"]))].update({"type": "nonsense"}), "type"),
        (lambda r: r.__setitem__("schema", "simgen-request/9"), "schema"),
    ],
)
def test_validator_catches_broken_records(good, breakage, message):
    breakage(good)
    errors = validate_record(good)
    assert any(message in e for e in errors), errors


def test_validator_catches_probabilities_that_do_not_sum_to_one(good):
    name = next(n for n, q in good["request"]["questions"].items() if q["type"] == "choice")
    first = next(iter(good["labels"][name]["probabilities"]))
    good["labels"][name]["probabilities"][first] += 0.01
    assert any("sum to 1" in e for e in validate_record(good))


def test_validator_catches_a_choice_with_too_many_options(good):
    name = next(n for n, q in good["request"]["questions"].items() if q["type"] == "choice")
    good["request"]["questions"][name]["criteria"] = {f"o{i}": "x" for i in range(9)}
    assert any("2–8" in e for e in validate_record(good))


def test_validator_catches_score_criteria_that_are_not_a_list(good):
    name = next(n for n, q in good["request"]["questions"].items() if q["type"] == "score")
    good["request"]["questions"][name]["criteria"] = {"0": "low", "1": "high"}
    assert any("list of strings" in e for e in validate_record(good))


def test_validator_catches_an_action_argmax_that_is_not_the_cheapest(records):
    record, name = next(
        (copy.deepcopy(r), n) for r in records for n, lab in r["labels"].items()
        if lab.get("kind") == "action" and len(set(lab["expected_cost"].values())) > 1
    )  # fmt: skip
    label = record["labels"][name]
    worst = max(label["expected_cost"], key=label["expected_cost"].get)
    label["probabilities"] = {k: (1.0 if k == worst else 0.0) for k in label["probabilities"]}
    assert any("argmax" in e for e in validate_record(record))
