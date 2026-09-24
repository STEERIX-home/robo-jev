"""증분 스트림 직렬화 검사 — 틱마다 새 토큰만 만드는 직렬화가 전체 직렬화(`serialize_request`)와 토큰 단위로 같다 (Task R4 A1).

루프 안의 정책은 에피소드 끝을 모른 채 틱마다 토큰을 붙여야 한다. 그 토큰이 학습·평가가 쓴 전체 직렬화(ts0.6)의 그 틱
구간과 **글자 그대로** 같아야 "루프 안의 모델이 우리가 평가한 그 모델"이라는 말이 성립한다 — 변화분은 입력의 정의이므로
(docs/08 §3.1) 앞 n틱의 직렬화는 전체의 접두이고, 증분 직렬화는 그 접두를 틱 단위로 잘라 낸 것이어야 한다.
"""

import copy

import pytest
from test_serialize import synthetic_stream

from robo_jev.contracts import model_input
from robo_jev.model.incremental import IncrementalStreamSerializer, project_stream_tick
from robo_jev.model.serialize import serialize_request
from robo_jev.model.tokenizer import FETCH_SCRIPT, WhitespaceTokenizer, available_tokenizer, load_tokenizer

#: 전체 직렬화의 틱 항목 가운데 증분 직렬화가 같은 값으로 내야 하는 키.
TICK_KEYS = ("index", "t", "start", "body_end", "end", "posed", "decision_positions", "candidate_boundaries", "candidate_mapping")


@pytest.fixture(scope="module")
def tokenizer() -> WhitespaceTokenizer:
    return WhitespaceTokenizer()


def _real_tokenizer():
    found = available_tokenizer()
    if found is None:
        pytest.skip(f"실제 tokenizer가 없다 — `uv run python {FETCH_SCRIPT}`")
    return load_tokenizer(found[0])


def _feed(record: dict, tokenizer, *, window_ticks: int = 30, instructions: list[dict] | None = None) -> tuple[IncrementalStreamSerializer, list[dict]]:
    """레코드의 틱을 차례로 먹인다 — 정책이 루프에서 하는 것과 같은 순서(prefix는 첫 틱의 상태로, 지시는 주어진 것만)."""
    projected = model_input(record)
    first = projected["ticks"][0]["request"]["state"]
    serializer = IncrementalStreamSerializer(
        tokenizer, question_set=projected["prefix"]["question_set"],
        instructions=instructions if instructions is not None else projected["prefix"]["instructions"],
        first_state=first, window_ticks=window_ticks,
    )
    entries = [serializer.tick(project_stream_tick(tick)) for tick in record["ticks"]]
    return serializer, entries


def _assert_same_as_whole(record: dict, tokenizer, **options) -> None:
    whole = serialize_request(record, tokenizer, layout="stream_l1a")
    serializer, entries = _feed(record, tokenizer, **options)
    assert serializer.prefix_tokens == whole["tokens"][: whole["prefix_end"]]
    assert serializer.prefix_end == whole["prefix_end"]
    assert serializer.static_candidate_boundaries == whole["static_candidate_boundaries"]
    assert serializer.static_candidate_mapping == whole["static_candidate_mapping"]
    assert serializer.decision_markers == whole["decision_markers"]
    assert len(entries) == len(whole["ticks"])
    for entry, expected in zip(entries, whole["ticks"]):
        assert {key: entry[key] for key in TICK_KEYS} == {key: expected[key] for key in TICK_KEYS}, entry["index"]
        assert entry["tokens"] == whole["tokens"][expected["start"] : expected["end"]], entry["index"]
        assert entry["body_tokens"] == whole["tokens"][expected["start"] : expected["body_end"]], entry["index"]
        assert entry["decision_tokens"] == whole["tokens"][expected["body_end"] : expected["end"]], entry["index"]
    assert serializer.cursor == len(whole["tokens"])


def test_incremental_ticks_are_the_whole_serialization_cut_at_tick_boundaries_on_the_fixtures(streams, tokenizer):
    """D0 fixture 네 편(둘은 지시 변경이 있다) 전부 — 공백 tokenizer."""
    for record in streams:
        _assert_same_as_whole(copy.deepcopy(record), tokenizer)


def test_incremental_ticks_match_the_whole_serialization_with_the_real_tokenizer(streams):
    """조각 단위 토큰화는 실제 BPE에서도 같아야 한다 — 줄 경계에서 조각을 따로 토큰화하는 것이 전체 직렬화의 규칙이다."""
    real = _real_tokenizer()
    record = copy.deepcopy(streams[0])
    record["ticks"] = record["ticks"][:40]
    _assert_same_as_whole(record, real)


def test_incremental_ticks_follow_the_delta_rules_over_the_refresh_periods(tokenizer):
    """35틱(소개 주기 30·동적 주기 10·목표 텍스트 주기 10을 다 지난다)에서 물체가 움직이는 합성 스트림."""
    record = synthetic_stream(35, mutate=lambda i, s: s["objects"][0].__setitem__("pose_mm", [300 + 5 * i, 0, -80]))
    _assert_same_as_whole(record, tokenizer)


def _with_instruction_change(ticks: int, *, at: int, version: int = 2) -> dict:
    """틱 `at`에서 지시가 `version`으로 바뀌는 합성 스트림 — 레코드의 prefix에는 생성기(`_extend_instructions`)의 규칙대로
    그 틱의 `sim_ms`를 `t_ms`로 한 지시가 덧붙어 있다."""
    text = "o1 상자를 zoneL로 옮겨라"

    def mutate(index: int, state: dict) -> None:
        if index >= at:
            state["goal"]["version"] = version
            state["goal"]["text"] = text
            state["goal"]["target_ref"] = "o1"
            state["goal"].pop("t_ms", None)  # 하네스의 goal에는 t_ms가 없다 — 생성기는 그 틱의 sim_ms를 쓴다

    record = synthetic_stream(ticks, mutate=mutate)
    record["prefix"]["instructions"].append({"version": version, "t_ms": int(record["ticks"][at]["sim_ms"]), "text": text})
    return record


def test_a_mid_episode_instruction_is_placed_at_the_tick_that_first_carries_its_version(tokenizer):
    record = _with_instruction_change(24, at=12)
    _assert_same_as_whole(record, tokenizer)
    _, entries = _feed(record, tokenizer)
    assert entries[12]["instructions_placed"] == [2] and entries[12]["carries_instruction"] is True
    assert all(entry["instructions_placed"] == [] for entry in entries if entry["index"] != 12)


def test_the_serializer_extends_the_instructions_from_the_goal_when_it_was_given_only_the_first_one(tokenizer):
    """루프 안에서는 뒤에 올 지시를 모른다: 목표 버전이 오르면 생성기의 규칙(버전·그 틱의 `sim_ms`·목표 텍스트)으로
    지시를 만들어 붙인다 — 전체 직렬화가 레코드의 prefix에서 읽는 것과 같은 줄이다."""
    record = _with_instruction_change(24, at=12)
    whole = serialize_request(record, tokenizer, layout="stream_l1a")
    serializer, entries = _feed(record, tokenizer, instructions=record["prefix"]["instructions"][:1])
    assert [ins["version"] for ins in serializer.instructions] == [1, 2]
    assert serializer.instructions[1] == record["prefix"]["instructions"][1]
    assert entries[12]["tokens"] == whole["tokens"][whole["ticks"][12]["start"] : whole["ticks"][12]["end"]]
    assert serializer.cursor == len(whole["tokens"])


def test_two_versions_arriving_in_one_tick_are_both_placed_there_in_order(tokenizer):
    """버전이 1 → 3으로 뛰면 알고 있는 v2·v3 지시가 그 틱 앞에 차례로 놓인다 (`_instruction_slots`와 같은 규칙)."""
    record = _with_instruction_change(20, at=8, version=3)
    record["prefix"]["instructions"].insert(1, {"version": 2, "t_ms": 350, "text": "o1 상자를 zoneL로 옮겨라 (v2)"})
    _assert_same_as_whole(record, tokenizer)
    _, entries = _feed(record, tokenizer)
    assert entries[8]["instructions_placed"] == [2, 3]


def test_the_serializer_refuses_ticks_out_of_order_and_an_unknown_question_set(streams, tokenizer):
    record = copy.deepcopy(streams[1])
    projected = model_input(record)
    with pytest.raises(ValueError, match="question_set"):
        IncrementalStreamSerializer(tokenizer, question_set="qs-none", instructions=projected["prefix"]["instructions"], first_state=projected["ticks"][0]["request"]["state"])
    serializer = IncrementalStreamSerializer(tokenizer, question_set="qs-v0", instructions=projected["prefix"]["instructions"], first_state=projected["ticks"][0]["request"]["state"])
    serializer.tick(project_stream_tick(record["ticks"][1]))
    with pytest.raises(ValueError, match="t"):
        serializer.tick(project_stream_tick(record["ticks"][0]))  # 틱 번호는 증가만 한다 (제어 스텝 수라 +1은 아니다)


def test_project_stream_tick_keeps_only_the_model_input_fields_and_rejects_leaks():
    tick = {
        "t": 3, "sim_ms": 300, "observed_at_ms": 290, "obs_age_ms": {"geom": 10, "proprio": 0},
        "request": {"state": {"goal": {"version": 1, "text": "x"}}, "exec_history": "none", "commitment": None, "candidates": {"q_main": [{"id": "c1", "key": "hold"}]}},
        "harness": {"candidates": {}}, "labels": [{"question_id": "q_main", "kind": "single", "answer": "c1"}], "adopted": {"main": "c1"},
    }
    projected = project_stream_tick(tick)
    assert set(projected) == {"t", "sim_ms", "observed_at_ms", "obs_age_ms", "request"}
    assert set(projected["request"]) == {"state", "exec_history", "commitment", "candidates"}
    assert projected["request"]["state"] is not tick["request"]["state"]  # 깊은 복사 — 원본은 그대로
    leaking = copy.deepcopy(tick)
    leaking["request"]["state"]["labels"] = [{"question_id": "q_main", "kind": "single", "answer": "c1"}]
    with pytest.raises(ValueError, match="labels"):
        project_stream_tick(leaking)
