"""`robo_jev.decisions.records` — 판단 요청 기록(`simgen-request/0`)의 읽기·형식 검사·쌍을 지키는 표본 고르기 (Task v0c).

진단 도구는 v0a 엔진 코드에 기대지 않고 JSONL 형식에만 기댄다. 형식 검사는 Decisions 제약(v0a 지시서 §2)을 그대로 보고,
표본 고르기는 사실 하나만 바꾼 쌍(`meta.pair`)을 깨지 않으며 같은 seed의 작은 표본이 큰 표본에 들어간다(2B reasoning 100 ⊂ direct 300).
"""

import copy
import json

import pytest
from helpers import FIXTURES

from robo_jev.decisions.records import (
    SCHEMA,
    pair_groups,
    pair_of,
    read_records,
    select_records,
    shuffle_key,
    validate_record,
)

REQUESTS = FIXTURES / "decisions" / "requests.jsonl"


@pytest.fixture(scope="module")
def fixtures() -> list[dict]:
    return read_records(REQUESTS)


def test_the_handwritten_fixture_is_valid_simgen_request_0(fixtures):
    assert len(fixtures) == 8 and all(record["schema"] == SCHEMA for record in fixtures)
    assert {record["id"]: validate_record(record) for record in fixtures} == {record["id"]: [] for record in fixtures}


def test_the_fixture_file_is_what_its_builder_writes(tmp_path):
    """fixture는 손으로 쓴 builder의 출력 그대로다 (행동 choice 확률은 기대 비용 softmax로 계산)."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("build_requests", FIXTURES / "decisions" / "build_requests.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    built = [json.dumps(record, ensure_ascii=False) for record in module.records()]
    assert REQUESTS.read_text(encoding="utf-8").splitlines() == built


def test_validation_names_each_broken_constraint(fixtures):
    record = copy.deepcopy(fixtures[0])
    record["request"]["state"] = "x" * 16_001
    record["request"]["questions"]["bad-name"] = {"type": "noul", "instructions": "?"}
    record["request"]["questions"]["action"]["criteria"] = {"only": "one option"}
    record["labels"]["contact_if_continue"]["p_yes"] = 1.2
    record["labels"]["person_next"]["probabilities"]["stays"] = 0.9  # 합이 1이 아니다
    problems = "\n".join(validate_record(record))
    assert "state" in problems and "16000" in problems
    assert "bad-name" in problems
    assert "action" in problems and "2..8" in problems
    assert "contact_if_continue" in problems and "[0, 1]" in problems
    assert "person_next" in problems and "sum" in problems


def test_validation_requires_labels_to_cover_exactly_the_option_keys(fixtures):
    record = copy.deepcopy(fixtures[0])
    del record["labels"]["action"]["probabilities"]["continue"]  # 확률 ≈ 0인 선택지라 합은 그대로 1이다
    del record["labels"]["action"]["expected_cost"]["continue"]
    problems = validate_record(record)
    assert any("action" in p and "keys" in p for p in problems)
    other = copy.deepcopy(fixtures[0])
    other["labels"]["action"]["expected_cost"]["mystery"] = 1.0
    assert any("action" in p and "expected_cost" in p for p in validate_record(other))


def test_validation_rejects_a_wrong_schema_and_too_many_questions(fixtures):
    record = copy.deepcopy(fixtures[0])
    record["schema"] = "simgen-request/9"
    for i in range(12):
        record["request"]["questions"][f"extra_{i}"] = {"type": "noul", "instructions": "?"}
    problems = "\n".join(validate_record(record))
    assert "simgen-request/9" in problems and "1..12" in problems


def test_pair_partner_and_shuffle_key_are_shared_within_a_pair(fixtures):
    by_id = {record["id"]: record for record in fixtures}
    assert pair_of(by_id["fx-002"]) == ("fx-001", "numeric") and pair_of(by_id["fx-003"]) == ("fx-001", "wording")
    assert pair_of(by_id["fx-001"]) == (None, None)
    assert shuffle_key(by_id["fx-001"]) == shuffle_key(by_id["fx-002"]) == shuffle_key(by_id["fx-003"]) == "fx-001"
    assert shuffle_key(by_id["fx-004"]) == "fx-004"


def test_pair_groups_join_records_linked_by_meta_pair(fixtures):
    groups = pair_groups(fixtures)
    assert sorted(len(group) for group in groups) == [1, 1, 1, 1, 1, 3]
    assert sorted(r["id"] for r in next(g for g in groups if len(g) == 3)) == ["fx-001", "fx-002", "fx-003"]


def _synthetic(n: int, pairs: int) -> list[dict]:
    out = [{"id": f"r{i:04d}", "meta": {"pair": None}} for i in range(n)]
    for i in range(pairs):  # 뒤쪽 레코드가 앞쪽 레코드의 짝
        out[n - 1 - i]["meta"]["pair"] = {"of": f"r{i:04d}", "kind": "numeric" if i % 2 else "wording"}
    return out


def test_selection_is_reproducible_keeps_pairs_whole_and_nests_smaller_in_larger():
    records = _synthetic(2000, 100)
    a = select_records(records, 300, seed=7)
    assert [r["id"] for r in a] == [r["id"] for r in select_records(records, 300, seed=7)]
    assert [r["id"] for r in a] != [r["id"] for r in select_records(records, 300, seed=8)]
    assert 300 <= len(a) <= 301  # 마지막 묶음이 쌍이면 하나 넘칠 수 있다(쌍을 깨지 않는다)
    chosen = {r["id"] for r in a}
    for record in a:
        partner = (record["meta"]["pair"] or {}).get("of")
        if partner is not None:
            assert partner in chosen
    small = select_records(records, 100, seed=7)
    assert {r["id"] for r in small} <= chosen
    assert sum(1 for r in a if r["meta"]["pair"]) >= 5  # 쌍 단위로 뽑아 짝이 함께 들어온다


def test_selection_without_a_limit_returns_everything_in_file_order():
    records = _synthetic(10, 2)
    assert [r["id"] for r in select_records(records, None, seed=1)] == [r["id"] for r in records]


def test_read_records_skips_blank_lines(tmp_path):
    path = tmp_path / "x.jsonl"
    path.write_text('{"id": "a"}\n\n{"id": "b"}\n', encoding="utf-8")
    assert [r["id"] for r in read_records(path)] == ["a", "b"]
