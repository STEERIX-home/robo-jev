"""직렬화 검사 — 상태 선행(L0) `state_first`와 스트림(L1-a) `stream_l1a` (docs/08 §3, docs/03 §3).

구조 검사는 공백 tokenizer로 언제나 돈다. 실제 backbone tokenizer가 필요한 검사는 받아 둔
것이 없으면 이유를 적고 건너뛴다 (`scripts/fetch_tokenizer.py`).
"""

import copy
import json

import pytest
import yaml
from helpers import SIM_CONFIG

from robo_jev.contracts import QUESTION_SET_V0, model_input
from robo_jev.model.serialize import (
    DECISION_MARKERS,
    LAYOUTS,
    POSITION_UNIT,
    QUATERNION_DECIMALS,
    SERIALIZER_VERSION,
    TIME_UNIT,
    WINDOW_TICKS,
    serialize_request,
)
from robo_jev.model.tokenizer import FETCH_SCRIPT, WhitespaceTokenizer, available_tokenizer, load_tokenizer


@pytest.fixture
def tokenizer() -> WhitespaceTokenizer:
    return WhitespaceTokenizer()


@pytest.fixture
def single(singles) -> dict:
    return copy.deepcopy(singles[0])  # choice + valid_set (Task 1의 고정)


@pytest.fixture
def three_questions(singles) -> dict:
    """세 타입(choice·boolean·ordinal)이 한 요청에 있는 D0 레코드."""
    record = next(r for r in singles if len(r["request"]["questions"]) == 3)
    return copy.deepcopy(record)


@pytest.fixture
def stream(streams) -> dict:
    """첫 에피소드의 앞 6틱 (지시 변경 전)."""
    record = copy.deepcopy(streams[0])
    record["ticks"] = record["ticks"][:6]
    return record


def segments_of(out: dict, kind: str) -> list[dict]:
    return [segment for segment in out["segments"] if segment["kind"] == kind]


# --------------------------------------------------------------------------
# 입력 경계
# --------------------------------------------------------------------------


def test_forbidden_keys_in_the_input_area_are_rejected(single, tokenizer):
    single["request"]["state"]["evidence"] = {"future_success": True}
    with pytest.raises(ValueError, match="request.state.evidence"):
        serialize_request(single, tokenizer)


def test_labels_and_evidence_outside_the_input_never_reach_the_tokens(single, tokenizer):
    stripped = copy.deepcopy(single)
    for field in ("labels", "evidence", "provenance", "usage", "split", "origin_group"):
        stripped.pop(field, None)
    assert serialize_request(single, tokenizer) == serialize_request(stripped, tokenizer)

    text = serialize_request(single, tokenizer)["text"]
    assert "candidate_ids" not in text
    assert "rule_trace" not in text
    assert "true_state" not in text
    assert single["request"]["request_id"] not in text  # id는 입출력 연결용이다 (docs/03 §2)


def test_stream_labels_and_outputs_never_reach_the_tokens(stream, tokenizer):
    out = serialize_request(stream, tokenizer, layout="stream_l1a")
    stripped = copy.deepcopy(stream)
    for tick in stripped["ticks"]:
        for field in ("labels", "model_output", "adopted", "ack"):
            tick.pop(field, None)
    for field in ("evidence", "provenance", "versions", "split", "origin_group"):
        stripped.pop(field, None)
    assert serialize_request(stripped, tokenizer, layout="stream_l1a") == out
    assert "model_output" not in out["text"] and "adopted" not in out["text"]


def test_serialization_uses_only_the_model_input(single, tokenizer):
    """직렬화는 `model_input`의 투영 위에서만 일어난다 — 같은 투영이면 같은 결과다."""
    projected = model_input(single)
    assert serialize_request(projected, tokenizer) == serialize_request(single, tokenizer)


# --------------------------------------------------------------------------
# 서식 규칙 (docs/08 §3.2)
# --------------------------------------------------------------------------


def test_format_rules_match_the_serializer_config():
    config = yaml.safe_load(SIM_CONFIG.read_text(encoding="utf-8"))
    rules = config["serialization"]
    assert (POSITION_UNIT, QUATERNION_DECIMALS, TIME_UNIT) == (
        rules["position_unit"],
        rules["quaternion_decimals"],
        rules["time_unit"],
    )
    assert SERIALIZER_VERSION == config["version"]


def test_positions_are_mm_integers_quaternions_two_decimals_times_ms_integers(single, tokenizer):
    state = single["request"]["state"]
    state["objects"][0]["pose_mm"] = [-10.4, 255.5, 741.6]
    state["objects"][0]["quat"] = [0.0, 0.70710678, 0.0, 0.70710678]
    state["observed_at_ms"] = 1200.7
    state["robot"] = {"ee_pose_mm": [0.49, 0.5, 1002.0], "contact_n": 0.125}
    text = serialize_request(single, tokenizer)["text"]
    assert "pose_mm=-10,256,742" in text
    assert "quat=0.00,0.71,0.00,0.71" in text
    assert "observed_at_ms=1201" in text
    assert "ee_pose_mm=0,0,1002" in text or "ee_pose_mm=0,1,1002" in text
    assert "contact_n=0.12" in text or "contact_n=0.13" in text
    assert "0.70710678" not in text and "255.5" not in text


def test_one_item_per_line_in_a_fixed_field_order(single, tokenizer):
    state = single["request"]["state"]
    # 필드 순서를 뒤섞어도 출력 순서는 같다.
    shuffled = {key: state["objects"][0][key] for key in reversed(list(state["objects"][0]))}
    shuffled_record = copy.deepcopy(single)
    shuffled_record["request"]["state"]["objects"][0] = shuffled
    assert serialize_request(shuffled_record, tokenizer)["text"] == serialize_request(single, tokenizer)["text"]

    lines = serialize_request(single, tokenizer)["text"].splitlines()
    object_lines = [line for line in lines if line.startswith("object ")]
    zone_lines = [line for line in lines if line.startswith("zone ")]
    assert len(object_lines) == len(state["objects"])
    assert len(zone_lines) == len(state["zones"])
    assert object_lines[0].startswith("object id=o1 desc=")  # id·설명이 앞, 나머지는 고정 순서
    assert lines.index("goal=" + state["goal"]) < lines.index(object_lines[0]) < lines.index(zone_lines[0])


def test_empty_lists_and_none_are_explicit(single, tokenizer):
    state = single["request"]["state"]
    state["objects"][0]["attributes"] = []
    state["robot"] = {"holding": None}
    text = serialize_request(single, tokenizer)["text"]
    assert "attributes=-" in text
    assert "robot holding=none" in text


# --------------------------------------------------------------------------
# state_first (L0)
# --------------------------------------------------------------------------


def test_state_first_puts_the_shared_state_before_every_question(three_questions, tokenizer):
    out = serialize_request(three_questions, tokenizer)
    questions = three_questions["request"]["questions"]
    n = len(out["tokens"])
    assert out["layout"] == "state_first"
    assert len(out["kind"]) == len(out["question"]) == len(out["candidate"]) == len(out["position"]) == n
    assert out["state"] == [0] * n
    assert out["question_ids"] == [question["id"] for question in questions]

    state_end = out["state_end"]
    assert all(kind == "state" for kind in out["kind"][:state_end])
    assert all(branch == -1 for branch in out["question"][:state_end])
    assert out["position"][:state_end] == list(range(state_end))

    # 각 질문 T_i: 질문 텍스트 → 전체 후보(경계 포함) → 결정 위치 하나. position은 len(S)부터.
    for branch, question in enumerate(questions):
        indices = [index for index in range(n) if out["question"][index] == branch]
        assert indices == list(range(indices[0], indices[-1] + 1))  # 연속 구간
        assert out["position"][indices[0]] == state_end
        assert out["position"][indices[0] : indices[-1] + 1] == list(
            range(state_end, state_end + len(indices))
        )
        kinds = [out["kind"][index] for index in indices]
        assert kinds[0] == "question"
        assert kinds[-1] == "decision"
        assert kinds.count("decision") == 1

        boundaries = out["candidate_boundaries"][question["id"]]
        assert len(boundaries) == len(question["criteria"])
        assert boundaries == sorted(boundaries)
        assert all(out["kind"][index] == "candidate" for index in boundaries)
        assert [out["candidate"][index] for index in boundaries] == list(range(len(boundaries)))
        assert out["decision_positions"][question["id"]] == indices[-1]
        assert boundaries[-1] < out["decision_positions"][question["id"]]
        assert out["candidate_mapping"][question["id"]] == [c["id"] for c in question["criteria"]]


def test_boolean_and_ordinal_candidates_come_from_the_question(three_questions, tokenizer):
    out = serialize_request(three_questions, tokenizer)
    by_type = {q["type"]: q for q in three_questions["request"]["questions"]}
    boolean = by_type["boolean"]
    ordinal = by_type["ordinal"]
    assert out["candidate_mapping"][boolean["id"]] == ["true", "false"]
    assert out["candidate_mapping"][ordinal["id"]] == [c["id"] for c in ordinal["criteria"]]
    text = out["text"]
    for criterion in ordinal["criteria"]:
        assert f"value={criterion['value']:g}" in text  # 수준의 수치도 모델이 본다 (docs/03 §2)


def test_decision_token_is_the_documented_reserved_token_per_question(three_questions, tokenizer):
    out = serialize_request(three_questions, tokenizer)
    for branch, question_id in enumerate(out["question_ids"]):
        marker = DECISION_MARKERS[branch]
        assert out["decision_markers"][question_id] == marker
        position = out["decision_positions"][question_id]
        assert out["tokens"][position] == tokenizer.encode(marker).ids[-1]
        # 질문 머리에도 같은 표지가 있어 결정 토큰이 질문과 이어진다.
        header = next(s for s in out["segments"] if s["name"] == f"question:{question_id}")
        assert tokenizer.decode(out["tokens"][header["start"] : header["end"]]).startswith(
            f"{marker} {question_id} "
        )


def test_candidate_boundary_is_the_last_token_of_the_candidate_line(single, tokenizer):
    out = serialize_request(single, tokenizer)
    question = single["request"]["questions"][0]
    for boundary in out["candidate_boundaries"][question["id"]]:
        assert tokenizer.decode([out["tokens"][boundary]]) == "\n"
        assert out["kind"][boundary + 1] in ("candidate", "decision")


def test_state_first_segments_cover_the_sequence_in_order(three_questions, tokenizer):
    out = serialize_request(three_questions, tokenizer)
    segments = out["segments"]
    assert segments[0]["start"] == 0 and segments[-1]["end"] == len(out["tokens"])
    for previous, current in zip(segments, segments[1:]):
        assert previous["end"] == current["start"]
    assert [segment["id"] for segment in segments] == list(range(len(segments)))
    assert out["segment"] == [
        segment["id"] for segment in segments for _ in range(segment["end"] - segment["start"])
    ]
    assert segments_of(out, "decision") and all(
        segment["end"] - segment["start"] == 1 for segment in segments_of(out, "decision")
    )


def test_state_first_rejects_oversized_requests(single, tokenizer):
    single["request"]["state"]["padding"] = "x " * 3000
    with pytest.raises(ValueError, match="2048"):
        serialize_request(single, tokenizer)
    serialize_request(single, tokenizer, max_state_tokens=None, max_total_tokens=None)


def test_decision_markers_can_be_pinned_when_reserializing_part_of_a_request(three_questions, tokenizer):
    """요청의 일부(질문 하나)를 다시 직렬화해도 표지는 전체 요청의 것으로 고정할 수 있다 (P0 microbatch)."""
    whole = serialize_request(three_questions, tokenizer)
    second = copy.deepcopy(three_questions)
    second["request"]["questions"] = [second["request"]["questions"][1]]
    second["labels"] = [l for l in second["labels"] if l["question_id"] == second["request"]["questions"][0]["id"]]
    second["usage"]["questions_used"] = [second["request"]["questions"][0]["id"]]
    question_id = second["request"]["questions"][0]["id"]
    default = serialize_request(second, tokenizer)
    assert default["decision_markers"][question_id] == DECISION_MARKERS[0]  # 기본은 요청 순서
    pinned = serialize_request(second, tokenizer, decision_markers=whole["decision_markers"])
    assert pinned["decision_markers"][question_id] == whole["decision_markers"][question_id] == DECISION_MARKERS[1]
    # T_i의 토큰이 전체 요청 안의 것과 정확히 같다
    header = next(s for s in whole["segments"] if s["name"] == f"question:{question_id}")
    t_i = whole["tokens"][header["start"] : whole["decision_positions"][question_id] + 1]
    assert pinned["tokens"][pinned["state_end"] :] == t_i
    with pytest.raises(ValueError, match="decision_markers"):
        serialize_request(second, tokenizer, decision_markers={question_id: "a"})


def test_decision_markers_cannot_override_the_streams_fixed_order(stream, tokenizer):
    with pytest.raises(ValueError, match="decision_markers"):
        serialize_request(stream, tokenizer, layout="stream_l1a", decision_markers={"q_main": "B"})


def test_too_many_questions_for_the_reserved_markers_is_an_error(single, tokenizer):
    question = single["request"]["questions"][0]
    single["request"]["questions"] = [
        {**copy.deepcopy(question), "id": f"q{index}"} for index in range(len(DECISION_MARKERS) + 1)
    ]
    single.pop("labels")
    with pytest.raises(ValueError, match="결정 위치"):
        serialize_request(single, tokenizer, max_total_tokens=None)


def test_layout_must_match_the_record_kind(single, stream, tokenizer):
    assert LAYOUTS == ("state_first", "stream_l1a")
    with pytest.raises(ValueError, match="layout"):
        serialize_request(single, tokenizer, layout="stream_l1a")
    with pytest.raises(ValueError, match="layout"):
        serialize_request(stream, tokenizer, layout="state_first")
    with pytest.raises(ValueError, match="layout"):
        serialize_request(single, tokenizer, layout="question_first")


# --------------------------------------------------------------------------
# stream_l1a (L1-a)
# --------------------------------------------------------------------------


def test_stream_prefix_precedes_the_ticks_and_holds_the_question_set(stream, tokenizer):
    out = serialize_request(stream, tokenizer, layout="stream_l1a")
    n = len(out["tokens"])
    assert out["layout"] == "stream_l1a"
    assert out["window_ticks"] == WINDOW_TICKS == 30
    prefix_end = out["prefix_end"]
    assert 0 < prefix_end < n
    assert all(kind == "prefix" for kind in out["kind"][:prefix_end])
    assert all(tick == -1 for tick in out["tick"][:prefix_end])
    assert out["position"][:prefix_end] == list(range(prefix_end))
    assert out["question_ids"] == list(QUESTION_SET_V0)

    text = out["text"][: len("".join(out["text"].splitlines(keepends=True)[:60]))]
    assert stream["prefix"]["instructions"][0]["text"] in text
    for question_id, spec in QUESTION_SET_V0.items():
        assert spec["instructions"] in out["text"]
        if spec["criteria"]:  # 정적 후보는 prefix에 한 번만 — mask 역할은 prefix, 경계는 candidate 색인
            boundaries = out["static_candidate_boundaries"][question_id]
            assert len(boundaries) == len(spec["criteria"])
            assert all(index < prefix_end for index in boundaries)
            assert all(out["kind"][index] == "prefix" for index in boundaries)
            assert [out["candidate"][index] for index in boundaries] == list(range(len(boundaries)))
            assert out["static_candidate_mapping"][question_id] == [c["id"] for c in spec["criteria"]]
        else:
            assert question_id not in out["static_candidate_boundaries"]


def test_stream_ticks_are_contiguous_and_end_with_decision_branches(stream, tokenizer):
    out = serialize_request(stream, tokenizer, layout="stream_l1a")
    ticks = out["ticks"]
    assert len(ticks) == len(stream["ticks"])
    assert out["tick_boundaries"] == [[tick["start"], tick["end"]] for tick in ticks]
    assert ticks[0]["start"] == out["prefix_end"]
    assert ticks[-1]["end"] == len(out["tokens"])
    for index, (tick, source) in enumerate(zip(ticks, stream["ticks"])):
        assert tick["index"] == index and tick["t"] == source["t"]
        if index:
            assert tick["start"] == ticks[index - 1]["end"]
        body = out["kind"][tick["start"] : tick["body_end"]]
        assert "decision" not in body
        assert body[0] == "state"
        assert "exec" in body
        assert all(kind == "decision" for kind in out["kind"][tick["body_end"] : tick["end"]])
        assert all(value == index for value in out["tick"][tick["start"] : tick["end"]])
        assert all(branch == -1 for branch in out["question"][tick["start"] : tick["body_end"]])

        posed = [qid for qid, spec in QUESTION_SET_V0.items() if spec["criteria"] or qid in source["request"]["candidates"]]
        assert tick["posed"] == posed
        assert list(tick["decision_positions"]) == posed
        for branch, question_id in enumerate(out["question_ids"]):
            if question_id in posed:
                position = tick["decision_positions"][question_id]
                assert out["question"][position] == branch
                assert out["tokens"][position] == tokenizer.encode(DECISION_MARKERS[branch]).ids[-1]


def test_stream_decision_branches_share_the_tick_end_position_and_the_next_tick_continues_from_it(stream, tokenizer):
    out = serialize_request(stream, tokenizer, layout="stream_l1a")
    for index, tick in enumerate(out["ticks"]):
        body_positions = out["position"][tick["start"] : tick["body_end"]]
        assert body_positions == list(range(body_positions[0], body_positions[0] + len(body_positions)))
        branch_position = body_positions[-1] + 1
        assert set(out["position"][tick["body_end"] : tick["end"]]) == {branch_position}
        if index + 1 < len(out["ticks"]):
            assert out["position"][out["ticks"][index + 1]["start"]] == branch_position
    assert out["position"][out["ticks"][0]["start"]] == out["prefix_end"]


def test_stream_dynamic_candidates_live_in_the_tick_and_static_ones_in_the_prefix(stream, tokenizer):
    out = serialize_request(stream, tokenizer, layout="stream_l1a")
    for tick, source in zip(out["ticks"], stream["ticks"]):
        candidates = source["request"]["candidates"]
        for question_id in tick["posed"]:
            boundaries = tick["candidate_boundaries"][question_id]
            mapping = tick["candidate_mapping"][question_id]
            assert len(boundaries) == len(mapping)
            assert [out["candidate"][index] for index in boundaries] == list(range(len(boundaries)))
            if question_id in candidates:
                assert mapping == [entry["id"] for entry in candidates[question_id]]
                assert all(tick["start"] <= index < tick["body_end"] for index in boundaries)
                assert all(out["kind"][index] == "candidate" for index in boundaries)
            else:
                assert mapping == [c["id"] for c in QUESTION_SET_V0[question_id]["criteria"]]
                assert boundaries == out["static_candidate_boundaries"][question_id]
            assert all(index < tick["decision_positions"][question_id] for index in boundaries)


def test_stream_exec_history_is_the_actual_history_line(stream, tokenizer):
    stream["ticks"][2]["request"]["exec_history"] = "main=c1 phase=approach path=p0 speed=1 force=0 gripper=open stop=0 ack=ok"
    out = serialize_request(stream, tokenizer, layout="stream_l1a")
    tick = out["ticks"][2]
    exec_segments = [s for s in out["segments"] if s["kind"] == "exec" and s["tick"] == 2]
    assert len(exec_segments) == 1
    chunk = tokenizer.decode(out["tokens"][exec_segments[0]["start"] : exec_segments[0]["end"]])
    assert chunk == "exec_history main=c1 phase=approach path=p0 speed=1 force=0 gripper=open stop=0 ack=ok\n"
    assert tick["start"] <= exec_segments[0]["start"] < tick["body_end"]


def test_an_instruction_change_is_appended_before_the_tick_that_first_carries_it(streams, tokenizer):
    record = copy.deepcopy(streams[0])
    assert len(record["prefix"]["instructions"]) == 2
    second = record["prefix"]["instructions"][1]
    first_tick = next(
        index
        for index, tick in enumerate(record["ticks"])
        if int(tick["request"]["state"]["goal"]["version"]) >= second["version"]
    )
    record["ticks"] = record["ticks"][: first_tick + 2]
    out = serialize_request(record, tokenizer, layout="stream_l1a")

    # 도중 지시는 그 틱의 **첫 토큰들**이다 — prefix가 아니라 틱 토큰이라 윈도우 밖으로 나간다 (docs/08 §3.1).
    change = [s for s in out["segments"] if s["name"] == f"instruction:{second['version']}"]
    assert len(change) == 1
    assert change[0]["kind"] == "state" and change[0]["tick"] == first_tick
    assert change[0]["start"] == out["ticks"][first_tick]["start"] == out["ticks"][first_tick - 1]["end"]
    assert all(kind != "prefix" for kind in out["kind"][out["prefix_end"] :])
    assert all(value == first_tick for value in out["tick"][change[0]["start"] : change[0]["end"]])
    assert second["text"] in tokenizer.decode(out["tokens"][change[0]["start"] : change[0]["end"]])
    assert out["instruction_positions"] == [0, change[0]["start"]]
    assert out["unplaced_instructions"] == []
    # 처음 prefix는 에피소드 시작 시의 것으로 고정이다: 지시 변경이 있어도 같은 길이·내용이다.
    assert second["text"] not in tokenizer.decode(out["tokens"][: out["prefix_end"]])
    unchanged = copy.deepcopy(record)
    unchanged["prefix"]["instructions"] = unchanged["prefix"]["instructions"][:1]
    unchanged["ticks"] = unchanged["ticks"][:first_tick]
    same_prefix = serialize_request(unchanged, tokenizer, layout="stream_l1a")
    assert same_prefix["prefix_end"] == out["prefix_end"]
    assert same_prefix["tokens"][: out["prefix_end"]] == out["tokens"][: out["prefix_end"]]

    # 그 틱 앞에서 잘린 레코드에는 지시 v2를 실을 틱이 없다 — 넣지 않고 버전만 알린다.
    record["ticks"] = record["ticks"][:first_tick]
    truncated = serialize_request(record, tokenizer, layout="stream_l1a")
    assert truncated["unplaced_instructions"] == [second["version"]]
    assert truncated["instruction_positions"] == [0]
    assert truncated["ticks"][-1]["end"] == len(truncated["tokens"])


def test_tick_header_does_not_repeat_the_states_own_t_line(stream, tokenizer):
    """틱 머리는 `[tick t]`뿐이다. 상태에 `t` 구간이 없는 레코드(D0 fixture)에서만 틱 겉봉투의
    시각 값을 머리에 실어 정보를 잃지 않는다."""
    out = serialize_request(stream, tokenizer, layout="stream_l1a")
    lines = out["text"].splitlines()
    assert "[tick 0] sim_ms=0 observed_at_ms=0 obs_age_ms=geom:100,proprio:20" in lines

    harness_like = copy.deepcopy(stream)
    for tick in harness_like["ticks"]:
        tick["request"]["state"] = {
            "t": {"tick": tick["t"], "sim_ms": tick["sim_ms"], "observed_at_ms": tick["observed_at_ms"],
                  "age_ms": tick["obs_age_ms"], "seq": tick["t"] + 1},
            **tick["request"]["state"],
        }
    out = serialize_request(harness_like, tokenizer, layout="stream_l1a")
    lines = out["text"].splitlines()
    assert "[tick 0]" in lines
    assert "t tick=0 sim_ms=0 observed_at_ms=0 age_ms=geom:100,proprio:20 seq=1" in lines
    assert not any(line.startswith("[tick ") and "sim_ms" in line for line in lines)
    assert sum(line.startswith("[tick 0]") for line in lines) == 1


def test_unknown_question_set_is_an_error(stream, tokenizer):
    stream["prefix"]["question_set"] = "qs-v9"
    with pytest.raises(ValueError, match="question_set"):
        serialize_request(stream, tokenizer, layout="stream_l1a")


def test_stream_output_is_deterministic(stream, tokenizer):
    first = serialize_request(stream, tokenizer, layout="stream_l1a")
    second = serialize_request(stream, WhitespaceTokenizer(), layout="stream_l1a")
    assert first["text"] == second["text"]
    assert first["segments"] == second["segments"]


# --------------------------------------------------------------------------
# 실제 tokenizer
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def real_tokenizer():
    found = available_tokenizer()
    if found is None:
        pytest.skip(f"실제 tokenizer가 없다 — `uv run python {FETCH_SCRIPT}`로 받는다")
    return load_tokenizer(found[0])


def test_real_tokenizer_markers_are_single_tokens(real_tokenizer):
    """결정 위치는 1토큰 분기다 (docs/08 §3.1) — 예약 표지가 실제 어휘에서 한 토큰이어야 한다."""
    for marker in DECISION_MARKERS:
        assert len(real_tokenizer.encode(marker, add_special_tokens=False).ids) == 1, marker


def encode(tokenizer, text: str) -> list[int]:
    return tokenizer.encode(text, add_special_tokens=False).ids


def test_real_tokenizer_round_trips_the_stream_text(stream, real_tokenizer):
    """조각 단위 토큰화는 줄 경계에서 통째 토큰화와 같다 (분기점인 결정 표지 뒤는 제외)."""
    out = serialize_request(stream, real_tokenizer, layout="stream_l1a")
    assert real_tokenizer.decode(out["tokens"]) == out["text"]
    prefix_text = real_tokenizer.decode(out["tokens"][: out["prefix_end"]])
    assert encode(real_tokenizer, prefix_text) == out["tokens"][: out["prefix_end"]]
    for tick in out["ticks"]:
        assert tick["end"] - tick["body_end"] == len(tick["posed"])  # 결정 위치 = 질문당 1토큰
        body = real_tokenizer.decode(out["tokens"][tick["start"] : tick["body_end"]])
        assert encode(real_tokenizer, body) == out["tokens"][tick["start"] : tick["body_end"]]


def test_real_tokenizer_round_trips_the_single_request_text(three_questions, real_tokenizer):
    out = serialize_request(three_questions, real_tokenizer)
    assert real_tokenizer.decode(out["tokens"]) == out["text"]
    state_text = real_tokenizer.decode(out["tokens"][: out["state_end"]])
    assert encode(real_tokenizer, state_text) == out["tokens"][: out["state_end"]]
    for question_id in out["question_ids"]:
        branch = out["question_ids"].index(question_id)
        start = next(index for index, value in enumerate(out["question"]) if value == branch)
        decision = out["decision_positions"][question_id]
        branch_text = real_tokenizer.decode(out["tokens"][start:decision])
        assert encode(real_tokenizer, branch_text) == out["tokens"][start:decision]
    assert len(json.dumps(out["tokens"])) > 0
