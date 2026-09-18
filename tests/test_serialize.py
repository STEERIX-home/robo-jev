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
from robo_jev.model import serialize as serialize_module
from robo_jev.model.serialize import (
    DECISION_MARKERS,
    LAYOUTS,
    POSITION_UNIT,
    QUATERNION_DECIMALS,
    STATE_FIRST_MARKER,
    TIME_UNIT,
    TOKEN_SERIALIZER_VERSION,
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


def test_the_token_serializer_has_its_own_version_apart_from_the_record_serializer(single, stream, tokenizer):
    """토큰 직렬화(텍스트·표지·position 규칙)의 버전은 `TOKEN_SERIALIZER_VERSION`(`ts…`)이고, 레코드의
    `versions.serializer`(장면 설정의 `version`, `s…` — 상태 스키마·서식의 버전)와는 다른 것이다. 4b가 토큰 형식을
    바꿨을 때(표지 선언 줄, L0 표지 A) 두 형식이 한 문자열을 나눠 갖는 일이 다시 없도록 둘을 묶지 않는다."""
    from robo_jev.model import serialize as serialize_module

    config = yaml.safe_load(SIM_CONFIG.read_text(encoding="utf-8"))
    assert TOKEN_SERIALIZER_VERSION == "ts0.3"
    assert TOKEN_SERIALIZER_VERSION.startswith("ts") and str(config["version"]).startswith("s")
    assert TOKEN_SERIALIZER_VERSION != config["version"]
    assert not hasattr(serialize_module, "SERIALIZER_VERSION")  # 옛 이름은 뜻이 둘이라 없앴다
    assert serialize_request(single, tokenizer)["serializer"] == TOKEN_SERIALIZER_VERSION
    assert serialize_request(stream, tokenizer, layout="stream_l1a")["serializer"] == TOKEN_SERIALIZER_VERSION
    assert stream["versions"]["serializer"] != TOKEN_SERIALIZER_VERSION  # 레코드의 값은 레코드 직렬화의 것


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


def test_state_first_decision_token_is_one_fixed_marker_for_every_question(three_questions, tokenizer):
    """docs/03 §3(0d89e27): L0에서는 모든 질문이 같은 고정 표지 토큰을 쓴다 — 질문의 정체는 T_i 문맥이 준다."""
    out = serialize_request(three_questions, tokenizer)
    assert STATE_FIRST_MARKER in DECISION_MARKERS
    for question_id in out["question_ids"]:
        assert out["decision_markers"][question_id] == STATE_FIRST_MARKER
        position = out["decision_positions"][question_id]
        assert out["tokens"][position] == tokenizer.encode(STATE_FIRST_MARKER).ids[-1]
        # 질문 머리에도 같은 표지가 있어 결정 토큰이 질문과 이어진다.
        header = next(s for s in out["segments"] if s["name"] == f"question:{question_id}")
        assert tokenizer.decode(out["tokens"][header["start"] : header["end"]]).startswith(
            f"{STATE_FIRST_MARKER} {question_id} "
        )
    assert len(set(out["decision_markers"].values())) == 1


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


def test_state_first_reserializing_one_question_keeps_its_tokens(three_questions, tokenizer):
    """질문 하나만 다시 직렬화해도 T_i 토큰이 전체 요청 안의 것과 같다 — 표지가 순서에 묶이지 않으므로
    질문 추가·삭제·재배열 불변이 구조로 성립한다 (docs/03 §3, P0 microbatch docs/03 §5)."""
    whole = serialize_request(three_questions, tokenizer)
    for index, question in enumerate(three_questions["request"]["questions"]):
        alone = copy.deepcopy(three_questions)
        alone["request"]["questions"] = [copy.deepcopy(question)]
        alone["labels"] = [l for l in alone["labels"] if l["question_id"] == question["id"]]
        alone["usage"]["questions_used"] = [question["id"]]
        out = serialize_request(alone, tokenizer)
        header = next(s for s in whole["segments"] if s["name"] == f"question:{question['id']}")
        t_i = whole["tokens"][header["start"] : whole["decision_positions"][question["id"]] + 1]
        assert out["tokens"][out["state_end"] :] == t_i
        assert out["decision_markers"][question["id"]] == whole["decision_markers"][question["id"]]
    reordered = copy.deepcopy(three_questions)
    reordered["request"]["questions"] = list(reversed(reordered["request"]["questions"]))
    back = serialize_request(reordered, tokenizer)
    for question_id in whole["question_ids"]:
        header_a = next(s for s in whole["segments"] if s["name"] == f"question:{question_id}")
        header_b = next(s for s in back["segments"] if s["name"] == f"question:{question_id}")
        span_a = whole["tokens"][header_a["start"] : whole["decision_positions"][question_id] + 1]
        span_b = back["tokens"][header_b["start"] : back["decision_positions"][question_id] + 1]
        assert span_a == span_b


def test_state_first_has_no_question_cap_from_the_reserved_markers(single, tokenizer):
    """표지가 하나이므로 예약 토큰 수가 L0의 질문 수를 제한하지 않는다 (프로파일 상한은 별도)."""
    question = single["request"]["questions"][0]
    single["request"]["questions"] = [
        {**copy.deepcopy(question), "id": f"q{index}"} for index in range(len(DECISION_MARKERS) + 1)
    ]
    single.pop("labels")
    out = serialize_request(single, tokenizer, max_total_tokens=None)
    assert len(out["decision_positions"]) == len(DECISION_MARKERS) + 1
    assert set(out["decision_markers"].values()) == {STATE_FIRST_MARKER}


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


def test_every_question_set_id_the_harness_can_emit_is_serializable(stream, tokenizer):
    """하네스 설정의 `question_set_id`(언어별)는 전부 `QUESTION_SETS`에 있어야 한다 — 영어 세트는 한국어 세트와 같은
    id·타입·후보 id·표지에 설정의 `questions.*.en` 문구다. 그래서 언어를 바꿔도 결정 토큰과 layout 구조가 같다."""
    from helpers import HARNESS_CONFIG

    from robo_jev.model.serialize import QUESTION_SETS

    config = yaml.safe_load(HARNESS_CONFIG.read_text(encoding="utf-8"))
    ids = config["question_set_id"]
    assert set(ids.values()) <= set(QUESTION_SETS)
    assert ids["ko"] == "qs-v0" and QUESTION_SETS["qs-v0"] is QUESTION_SET_V0
    outputs = {}
    for language, question_set_id in ids.items():
        question_set = QUESTION_SETS[question_set_id]
        assert list(question_set) == list(QUESTION_SET_V0)
        for question_id, spec in question_set.items():
            reference = QUESTION_SET_V0[question_id]
            assert spec["instructions"] == config["questions"][question_id][language]
            assert spec["type"] == reference["type"] and spec["marker"] == reference["marker"]
            assert [c["id"] for c in spec["criteria"]] == [c["id"] for c in reference["criteria"]]
            assert [c.get("value") for c in spec["criteria"]] == [c.get("value") for c in reference["criteria"]]
        record = copy.deepcopy(stream)
        record["prefix"]["question_set"] = question_set_id
        outputs[language] = serialize_request(record, tokenizer, layout="stream_l1a")
        assert outputs[language]["question_set"] == question_set_id
        assert f"[questions {question_set_id}]" in outputs[language]["text"]
    ko, en = outputs["ko"], outputs["en"]
    assert en["decision_markers"] == ko["decision_markers"] and en["question_ids"] == ko["question_ids"]
    assert en["static_candidate_mapping"] == ko["static_candidate_mapping"]
    assert [t["posed"] for t in en["ticks"]] == [t["posed"] for t in ko["ticks"]]
    assert "Choose the action to execute now." in en["text"] and "지금 실행할 행동을 고르라." not in en["text"]
    assert en["text"] != ko["text"]


def test_stream_markers_are_declared_by_the_question_set_and_written_in_the_prefix(stream, tokenizer):
    """docs/08 §3.1(0d89e27): 표지는 질문 세트 버전이 id마다 고정해 정적 prefix에 선언한다."""
    out = serialize_request(stream, tokenizer, layout="stream_l1a")
    declared = {question_id: spec["marker"] for question_id, spec in QUESTION_SET_V0.items()}
    assert out["decision_markers"] == declared
    assert len(set(declared.values())) == len(declared)  # id마다 다른 예약 토큰
    assert all(marker in DECISION_MARKERS for marker in declared.values())
    prefix_text = tokenizer.decode(out["tokens"][: out["prefix_end"]])
    line = "markers " + " ".join(f"{question_id}={marker}" for question_id, marker in declared.items())
    assert line in prefix_text.splitlines()
    segment = next(s for s in out["segments"] if s["name"] == "markers")
    assert segment["kind"] == "prefix" and segment["end"] <= out["prefix_end"]
    for tick in out["ticks"]:
        for question_id, position in tick["decision_positions"].items():
            assert out["tokens"][position] == tokenizer.encode(declared[question_id]).ids[-1]


def test_stream_marker_of_a_question_does_not_depend_on_the_posed_subset(streams, tokenizer):
    """q_path가 없는 틱과 있는 틱에서 다른 질문의 표지 토큰이 같다 (묻는 부분집합·순서와 무관)."""
    record = copy.deepcopy(streams[0])
    record["ticks"] = record["ticks"][:8]
    out = serialize_request(record, tokenizer, layout="stream_l1a")
    with_path = [tick for tick in out["ticks"] if "q_path" in tick["decision_positions"]]
    without = [tick for tick in out["ticks"] if "q_path" not in tick["decision_positions"]]
    assert with_path and without
    for question_id, marker in out["decision_markers"].items():
        ids = {
            out["tokens"][tick["decision_positions"][question_id]]
            for tick in out["ticks"]
            if question_id in tick["decision_positions"]
        }
        assert ids == {tokenizer.encode(marker).ids[-1]}, question_id


def test_stream_question_set_can_gain_an_id_without_renumbering_the_others(stream, tokenizer, monkeypatch):
    """세트에 질문을 끼워 넣어도(선언된 map) 기존 id의 표지·결정 토큰은 그대로다."""
    before = serialize_request(stream, tokenizer, layout="stream_l1a")
    grown: dict = {}
    for question_id, spec in QUESTION_SET_V0.items():
        grown[question_id] = copy.deepcopy(spec)
        if question_id == "q_main":  # 두 번째 자리에 새 질문
            grown["q_new"] = {
                "type": "boolean",
                "instructions": "새 질문인가.",
                "criteria": [{"id": "true", "description": "예"}, {"id": "false", "description": "아니오"}],
                "marker": "K",
            }
    monkeypatch.setitem(serialize_module.QUESTION_SETS, "qs-v0", grown)
    after = serialize_request(stream, tokenizer, layout="stream_l1a")
    assert after["question_ids"] == list(grown) and after["question_ids"][1] == "q_new"
    assert after["decision_markers"]["q_new"] == "K"
    for question_id, marker in before["decision_markers"].items():
        assert after["decision_markers"][question_id] == marker
    for tick_before, tick_after in zip(before["ticks"], after["ticks"]):
        for question_id, position in tick_before["decision_positions"].items():
            assert after["tokens"][tick_after["decision_positions"][question_id]] == before["tokens"][position]
        assert "q_new" in tick_after["decision_positions"]
    assert "q_new=K" in tokenizer.decode(after["tokens"][: after["prefix_end"]])


@pytest.mark.parametrize(
    "mutate, message",
    [
        (lambda s: s["q_done"].pop("marker"), "marker"),
        (lambda s: s["q_done"].__setitem__("marker", "A"), "marker"),  # q_main과 중복
        (lambda s: s["q_done"].__setitem__("marker", "b"), "marker"),  # 예약 토큰 밖
    ],
)
def test_stream_question_set_markers_must_be_declared_unique_reserved_tokens(stream, tokenizer, monkeypatch, mutate, message):
    broken = copy.deepcopy(QUESTION_SET_V0)
    mutate(broken)
    monkeypatch.setitem(serialize_module.QUESTION_SETS, "qs-v0", broken)
    with pytest.raises(ValueError, match=message):
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
