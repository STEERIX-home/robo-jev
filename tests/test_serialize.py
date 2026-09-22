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
    assert TOKEN_SERIALIZER_VERSION == "ts0.6"
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
    state["objects"][0]["graspable_faces"] = []  # `attributes`는 서식 v0.4에서 모델 입력 밖이다 (Task R1 A1)
    state["robot"] = {"holding": None}
    text = serialize_request(single, tokenizer)["text"]
    assert "graspable_faces=-" in text
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
    # 프로파일 상한(Q≤16)은 표지 수와 무관한 **별개**의 제약이다 — 이 검사는 표지 쪽만 보므로 상한을 넓혀 준다.
    wide = {"max_questions": 64, "max_candidates": 32}
    out = serialize_request(single, tokenizer, max_total_tokens=None, limits=wide)
    assert len(out["decision_positions"]) == len(DECISION_MARKERS) + 1
    assert set(out["decision_markers"].values()) == {STATE_FIRST_MARKER}
    with pytest.raises(ValueError, match=r"^request\.questions: 질문 수가 프로파일 상한을 넘는다"):
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


def test_stream_exec_history_is_the_actual_history_line_in_short_form(stream, tokenizer):
    """실행 이력 줄(서식 v0.3): 하네스의 `main=… phase=…` 줄을 짧은 이름으로 다시 적고 기본값(stop=0·gate=none·fails=0)은 뺀다."""
    stream["ticks"][2]["request"]["exec_history"] = "main=c1 phase=approach path=p0 speed=1 force=0 gripper=open stop=0 gate=none ack=ok fails=0"
    stream["ticks"][3]["request"]["exec_history"] = "main=c1 phase=grasp path=via:w2@10,20,30 speed=2 force=1 gripper=closed stop=1 gate=observe ack=collision fails=2"
    out = serialize_request(stream, tokenizer, layout="stream_l1a")
    tick = out["ticks"][2]
    exec_segments = [s for s in out["segments"] if s["kind"] == "exec" and s["tick"] == 2]
    assert len(exec_segments) == 1
    chunk = tokenizer.decode(out["tokens"][exec_segments[0]["start"] : exec_segments[0]["end"]])
    assert chunk == "hist main=c1 ph=approach path=p0 v=1 f=0 g=open ack=ok\n"
    assert tick["start"] <= exec_segments[0]["start"] < tick["body_end"]
    third = next(s for s in out["segments"] if s["kind"] == "exec" and s["tick"] == 3)
    assert tokenizer.decode(out["tokens"][third["start"] : third["end"]]) == "hist main=c1 ph=grasp path=via:w2@10,20,30 v=2 f=1 g=closed stop gate=observe ack=collision fails=2\n"
    first = next(s for s in out["segments"] if s["kind"] == "exec" and s["tick"] == 0)
    assert tokenizer.decode(out["tokens"][first["start"] : first["end"]]) == "hist none\n"


def test_the_history_line_reads_the_fields_the_harness_writes():
    """직렬화의 이력 파서는 하네스의 `parse_exec_history`와 같은 규칙이다 (모델 코드가 하네스를 import하지 않으므로 대조)."""
    from robo_jev.harness.robot import parse_exec_history

    for text in ("none", "", "main=c1 phase=approach path=p0 speed=1 force=0 gripper=open stop=0 gate=none ack=ok fails=0", "odd tokens x=1"):
        assert serialize_module._history_fields(text) == parse_exec_history(text)


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


def test_the_tick_header_is_the_t_line_without_the_envelope_values(stream, tokenizer):
    """틱 머리는 `t <tick> age g<ms> p<ms> seq <n>` 한 줄이다 (서식 v0.3). 모의 시각·관측 시각·후보 집합 해시는 겉봉투
    (레코드·계약 검사)에만 있고 모델은 읽지 않는다. 상태에 `t` 구간이 없는 레코드(D0 fixture)는 겉봉투의 나이로 채운다."""
    out = serialize_request(stream, tokenizer, layout="stream_l1a")
    lines = out["text"].splitlines()
    assert "t 0 age g100 p20" in lines
    assert not any(line.startswith("[tick ") for line in lines)

    harness_like = copy.deepcopy(stream)
    for tick in harness_like["ticks"]:
        tick["request"]["state"] = {
            "t": {"tick": tick["t"], "sim_ms": tick["sim_ms"], "observed_at_ms": tick["observed_at_ms"],
                  "age_ms": tick["obs_age_ms"], "seq": tick["t"] + 1, "candidate_set_version": "cs-abcdef12"},
            **tick["request"]["state"],
        }
    out = serialize_request(harness_like, tokenizer, layout="stream_l1a")
    # 결정 표지 뒤에는 줄바꿈이 없으므로(1토큰 분기) 틱 몸통을 따로 읽는다.
    assert tick_text(out, tokenizer, 0).splitlines()[0] == "t 0 age g100 p20 seq 1"
    assert tick_text(out, tokenizer, 3).splitlines()[0] == "t 3 age g100 p20 seq 4"
    assert "cs-abcdef12" not in out["text"] and "sim_ms" not in out["text"] and "observed_at" not in out["text"]


# --------------------------------------------------------------------------
# 서식 v0.3 — 짧은 이름, 물체 소개/동적 분리, 변화분 틱 (docs/08 §3.1·§3.2)
# --------------------------------------------------------------------------


def synthetic_stream(ticks: int, *, mutate=None) -> dict:
    """하네스 상태 모양의 손으로 만든 스트림: 물체 둘, 영역 하나, `t` 구간과 물체별 나이·정밀도가 있다.

    `mutate(index, state)`가 틱마다 상태를 바꾼다(자세·가시성·관측 정지 …)."""

    def obj(object_id: str, pose, *, attributes=()):
        return {
            "id": object_id, "desc": f"{object_id} 상자", "class": "box", "pose_mm": list(pose), "quat": [0.0, 0.0, 0.0, 1.0],
            "precision_mm": 3, "pose_source": "geom", "obb_mm": [60, 60, 64], "top_mm": -48, "graspable_faces": ["top", "side"],
            "surface_conf": 0.92, "visible_ratio": 1.0, "last_seen_ms": 0, "age_ms": 0, "reid": [], "attributes": list(attributes),
        }

    record = {
        "schema_version": "stream-v0", "episode_id": "ep-v03",
        "prefix": {"instructions": [{"version": 1, "t_ms": 0, "text": "o0 상자를 zoneL로 옮겨라"}], "question_set": "qs-v0"},
        "ticks": [],
    }
    for index in range(ticks):
        now = 100 * index
        state = {
            "t": {"tick": index, "sim_ms": now, "observed_at_ms": now, "age_ms": {"geom": 0, "proprio": 0}, "seq": index + 1, "candidate_set_version": "cs-1"},
            "goal": {"text": "o0 상자를 zoneL로 옮겨라", "version": 1, "t_ms": 0, "target_ref": "o0", "target_desc": "o0 상자", "target_zone": "zoneL", "forbidden_contact": [], "fragile": ["o1"]},
            "objects": [obj("o0", (300, 0, -80)), obj("o1", (200, 220, -80), attributes=("fragile",))],
            "scene": {"work_surface_mm": -112, "free_width_mm": 300, "corridor_mm": 100, "clearance_mm": 40},
            "zones": [{"id": "zoneL", "desc": "왼쪽 정리 영역", "bounds_mm": [-120, 150, 180, 330]}],
            "robot": {"ee_pose_mm": [0, 0, 200], "ee_quat": [0.0, 0.0, 0.0, 1.0], "gripper_mm": 80, "holding": None, "contact_n": 0.0, "speed_mm_s": 0},
            "exec": {"seq": index, "action_ref": None, "phase": None, "path": None, "speed_level": 0, "force_level": None, "gripper": None, "stop": False, "progress": None, "applied": None, "reject": None, "gripper_wait": None, "events": [], "executor": "HOLD"},
            "events": [], "derived": [
                {"object": "o0", "relative_mm": [300, 0, -280], "clearance_mm": 40, "corridor_mm": 100, "age_ms": 0},
                {"object": "o1", "relative_mm": [200, 220, -280], "clearance_mm": 40, "corridor_mm": 100, "age_ms": 0},
            ],
            "commitment": None, "image": [], "geom": [], "extractor": "pw0.1",
        }
        for entry in state["objects"]:
            entry["last_seen_ms"] = now
        if mutate is not None:
            mutate(index, state)
        record["ticks"].append({
            "t": index, "sim_ms": now, "observed_at_ms": now, "obs_age_ms": {"geom": 0, "proprio": 0},
            "request": {
                "state": state, "exec_history": "none", "commitment": None,
                "candidates": {"q_main": [
                    {"id": "c1", "action_ref": "c1", "key": "grasp:o0:top:zoneL", "d": 381, "clr": 40, "path": "ok", "g": 0},
                    {"id": "c2", "action_ref": "c2", "key": "push:o0:-x:none", "d": 417, "clr": 40, "path": "blocked", "g": 0},
                    {"id": "ch", "action_ref": "ch", "key": "hold"},
                ], "q_path": [{"id": "p0", "kind": "direct", "action_ref": "ch"}, {"id": "p1", "kind": "via", "ref": "w1", "action_ref": "ch"}, {"id": "ph", "kind": "hold", "action_ref": "ch"}]},
            },
        })
    return record


def tick_text(out: dict, tokenizer, index: int) -> str:
    tick = out["ticks"][index]
    return tokenizer.decode(out["tokens"][tick["start"] : tick["body_end"]])


def section_text(out: dict, tokenizer, index: int, name: str) -> str:
    return "".join(
        tokenizer.decode(out["tokens"][segment["start"] : segment["end"]])
        for segment in out["segments"] if segment["tick"] == index and segment["name"] == name
    )


def test_stream_format_v03_uses_short_names_and_key_based_candidate_lines(tokenizer):
    out = serialize_request(synthetic_stream(1), tokenizer, layout="stream_l1a")
    assert out["format"] == serialize_module.STREAM_FORMAT == "v0.4"
    assert out["delta_rules"] == serialize_module.DELTA_RULES
    text = tick_text(out, tokenizer, 0)
    lines = text.splitlines()
    assert lines[0] == "t 0 age g0 p0 seq 1"
    assert lines[1] == "goal v1"  # 서식 v0.4: 버전뿐이다 — 텍스트는 지시 조각(prefix)과 주기적 재적재에, 목표는 없다
    assert "obj o0 o0 상자 obb=60,60,64 top=-48 faces=top,side" in lines
    assert "obj o1 o1 상자 obb=60,60,64 top=-48 faces=top,side" in lines  # 속성(`attr=`)은 없다 (서식 v0.4)
    # 동적 줄: 회전 없음(yaw=0)·다 보임(vis=1)은 기본값이라 없고, 말단 기준 상대 벡터는 싣지 않는다 (ee와 p로 정해진다).
    assert "o0 p=300,0,-80 s=3 conf=0.92 clr=40 corr=100" in lines
    assert "robot ee=0,0,200 eq=0.00,0.00,0.00,1.00 grip=80" in lines  # holding·contact_n·speed의 기본값은 뺀다
    assert "exec ex=HOLD" in lines
    assert "commitment none" in lines and "hist none" in lines
    # 후보 줄: 대상 기하 나이 g는 틱의 기하 나이(여기 0)와 같으면, `path=ok`는 기본값이라, `clr`는 대상 물체 동적 줄의 `clr`(40)와
    # 같으면(하네스가 같은 값에서 채운다) 뺀다 — 막힌 경로만 `path=blocked`.
    assert "c1: grasp o0 top→zoneL d=381" in lines
    assert "c2: push o0 -x d=417 path=blocked" in lines
    assert "ch: hold" in lines and "p0: direct" in lines and "p1: via w1" in lines and "ph: hold" in lines
    for absent in ("pose_mm", "visible_ratio", "graspable_faces", "extractor", "image", "geom", "candidate_set_version", "target_desc", "action_ref", "key=", "rel=", "g=0", "path=ok", "attr=", "target=", "forbid="):
        assert absent not in text, absent
    stale = synthetic_stream(1)
    stale["ticks"][0]["request"]["candidates"]["q_main"][0]["g"] = 200
    stale["ticks"][0]["request"]["candidates"]["q_main"][0]["clr"] = 25  # 대상 물체의 여유(40)와 다르면 남는다
    assert "c1: grasp o0 top→zoneL d=381 clr=25 g=200" in tick_text(serialize_request(stale, tokenizer, layout="stream_l1a"), tokenizer, 0)
    # 영역·장면 요약은 prefix에 있다.
    prefix_text = tokenizer.decode(out["tokens"][: out["prefix_end"]])
    assert "zone zoneL 왼쪽 정리 영역 b=-120,150,180,330" in prefix_text.splitlines()
    assert "scene surf=-112 free=300 corr=100 clr=40" in prefix_text.splitlines()
    assert "zone " not in text and "scene " not in text


def test_the_goal_line_carries_the_version_and_nothing_the_instruction_text_should_say(tokenizer):
    """서식 v0.4 (Task R1 A1): `goal` 줄은 **버전뿐**이고, 풀어 놓은 목표는 대상이 추적되든 말든, 금지 물체가 생기든
    한 글자도 나가지 않는다. 대상·목적지·제약은 지시 문장과 `obj`·`zone` 줄에서 모델이 스스로 풀어야 하는 것이다."""

    def mutate(index, state):
        if index == 1:
            state["goal"]["target_ref"] = None  # 아직 관측되지 않은 대상 — 옛 서식은 여기서 `desc=`를 실었다
        if index == 2:
            state["goal"]["forbidden_contact"] = ["o1"]

    out = serialize_request(synthetic_stream(3, mutate=mutate), tokenizer, layout="stream_l1a")
    for index in range(3):
        assert section_text(out, tokenizer, index, "state:goal") == "goal v1\n", index


#: 모델이 보는 텍스트 **어디에도** 있으면 안 되는 표지 — 풀어 놓은 목표와 물체 속성, 두 layout의 이름 모두 (Task R1 A1).
LEAKED_GOAL_MARKERS = (
    "target=", "target_ref=", "target_desc=", "zone=", "forbid=", "forbidden_contact=",
    "fragile=", "prio=", "priority=", "attr=", "attributes=",
)  # fmt: skip
#: `goal` 줄에만 걸리는 표지 — 물체·영역 줄의 `desc=`(설명)는 **남는다**: 그것이 모델이 지시 문장과 맞춰야 하는 것이다.
LEAKED_GOAL_LINE_MARKERS = ("desc=",)


def _model_text(out: dict, tokenizer) -> str:
    return tokenizer.decode(out["tokens"])


def assert_no_resolved_goal(text: str, where: str) -> None:
    for marker in LEAKED_GOAL_MARKERS:
        assert marker not in text, (where, marker)
    for line in text.splitlines():
        if line.startswith("goal"):
            for marker in LEAKED_GOAL_LINE_MARKERS:
                assert marker not in line, (where, marker, line)


def test_no_resolved_goal_field_reaches_the_model_in_either_layout(tokenizer, single, stream, three_questions):
    """**누출 봉쇄 (Task R1 A1·A2).** 풀어 놓은 목표(`target=`·`desc=`·`zone=`·`forbid=`·`fragile=`·`prio=`)와 물체
    속성(`attr=`)은 두 layout 어느 쪽의 모델 입력에도 나오지 않는다 — D0 fixture와 **실제로 생성한 에피소드 한 편**에서.
    레코드에는 그대로 있다(전문가·라벨·규칙 판정기가 읽는다); 빠지는 것은 모델이 보는 텍스트뿐이다."""
    for record, layout in ((single, "state_first"), (three_questions, "state_first"), (stream, "stream_l1a")):
        assert_no_resolved_goal(_model_text(serialize_request(record, tokenizer, layout=layout), tokenizer), layout)
    # 구조화된 목표는 레코드에 남아 있다 — 이 검사가 "그냥 목표가 없는 레코드"를 재는 것이 아니다.
    goal = stream["ticks"][0]["request"]["state"]["goal"]
    assert set(goal) & set(serialize_module.HIDDEN_GOAL_FIELDS)


@pytest.mark.parametrize("profile", ["E1"])
def test_no_resolved_goal_field_reaches_the_model_in_a_real_episode(tokenizer, profile):
    """실제 에피소드 한 편(생성기로 8틱)을 두 layout으로 직렬화해 같은 것을 확인한다 — 합성 fixture가 아니라
    하네스·전문가·앞단이 실제로 채운 상태에서."""
    from robo_jev.data.robot_contrast import default_question_texts, single_request_from_tick
    from robo_jev.data.robot_episodes import generate_episode, load_generator_config
    from robo_jev.sim.expert import Expert

    config = load_generator_config()
    expert = Expert()
    record = generate_episode(profile, 11, policy=expert, expert=expert, config=config, max_ticks=8)
    text = _model_text(serialize_request(record, tokenizer, layout="stream_l1a"), tokenizer)
    assert_no_resolved_goal(text, "stream_l1a")
    assert "goal v1\n" in text  # 버전은 남는다
    # 같은 틱을 `judgment-v0`(state_first)로 낸 대조 레코드도 같다.
    single, _ = single_request_from_tick(
        record,
        record["ticks"][0],
        record["ticks"][0]["request"]["state"],
        request_id="r1-leak-0",
        expert=expert,
        question_texts=default_question_texts(),
        shuffle_seed="r1-leak",
    )
    assert_no_resolved_goal(_model_text(serialize_request(single, tokenizer), tokenizer), "state_first")
    # 그런데 지시 **문장**은 있어야 한다 — 그것이 이제 유일한 단서다.
    assert record["prefix"]["instructions"][0]["text"] in text


def test_the_commitment_line_drops_the_key_the_candidate_line_carries(tokenizer):
    record = synthetic_stream(2)
    record["ticks"][1]["request"]["commitment"] = {"action_ref": "c1", "key": "grasp:o0:top:zoneL", "phase": "approach", "held_ticks": 0, "last_switch_tick": 0}
    out = serialize_request(record, tokenizer, layout="stream_l1a")
    assert "commitment a=c1 ph=approach held=0" in tick_text(out, tokenizer, 1).splitlines()
    assert "grasp:o0:top:zoneL" not in tick_text(out, tokenizer, 1)


def test_objects_are_introduced_once_and_dynamic_lines_follow_changes(tokenizer):
    """소개 줄은 시작·처음 관측·정적 필드 변경 때, 동적 줄은 변화(자세 > 정밀도, 회전, 가시 비율, 관측 정지, 재식별) 때만.
    바뀌지 않은 물체는 틱에서 빠진다."""

    def mutate(index, state):
        o0, o1 = state["objects"]
        if index >= 2:
            o0["pose_mm"] = [300, 0 + (2 if index == 2 else 20), -80]  # 틱 2: 정밀도 안(2 < 3), 틱 3부터 20mm
        if index >= 4:
            o0["quat"] = [0.0, 0.0, 0.38, 0.92]  # 틱 4에 회전, 그 뒤로 그대로
        if index in (5, 6):
            o1["visible_ratio"] = 0.3  # 가시 비율 변화 (틱 5), 틱 6은 같음
        if index == 7:
            o1["visible_ratio"] = 0.3
            o1["age_ms"] = 200  # 관측 정지: 기하 나이 0인데 이 물체는 200ms 전 것
            o1["last_seen_ms"] = 500
        if index == 8:
            o1["visible_ratio"] = 1.0
            o1["reid"] = ["merge:o9"]
        if index == 9:
            o1["graspable_faces"] = ["top"]  # 정적 필드 변경 → 소개 줄
            o1["attributes"] = ["fragile", "forbidden"]  # 서식 v0.4에서 모델 입력 밖 — 이것만으로는 소개 줄이 안 난다
            o1["visible_ratio"] = 1.0

    out = serialize_request(synthetic_stream(10, mutate=mutate), tokenizer, layout="stream_l1a")
    intro = [section_text(out, tokenizer, i, "state:objects_intro") for i in range(10)]
    dynamic = [section_text(out, tokenizer, i, "state:objects_dynamic") for i in range(10)]
    assert intro[0].count("obj ") == 2 and dynamic[0].count("\n") == 2
    assert intro[1] == "" and dynamic[1] == ""  # 아무것도 바뀌지 않았다
    assert dynamic[2] == ""  # 정밀도(3mm) 안의 흔들림은 변화가 아니다
    assert dynamic[3] == "o0 p=300,20,-80 s=3 clr=40 corr=100\n"  # 바뀐 것은 자세뿐: 선택 필드(방향·신뢰도·가시 비율)는 없다
    assert dynamic[4] == "o0 p=300,20,-80 yaw=45 s=3 clr=40 corr=100\n"  # z축 회전은 yaw(도)로
    assert dynamic[5] == "o1 p=200,220,-80 s=3 vis=0.3 clr=40 corr=100\n"
    assert dynamic[6] == ""
    assert dynamic[7] == "o1 p=200,220,-80 s=3 seen=500 clr=40 corr=100\n"  # 관측이 끊겼다 (가시 비율은 그대로라 없다)
    assert dynamic[8] == "o1 p=200,220,-80 s=3 vis=1 clr=40 corr=100 reid=merge:o9\n"  # 다시 보인다: 변화분 줄은 기본값(vis=1)도 적는다, 재식별
    assert "seen=" not in dynamic[8]
    assert intro[9] == "obj o1 o1 상자 obb=60,60,64 top=-48 faces=top\n"  # 속성(`attr=`)은 서식 v0.4에서 빠졌다
    assert all(intro[i] == "" for i in range(1, 9))


def test_a_change_of_object_attributes_alone_does_not_reach_the_model(tokenizer):
    """서식 v0.4 (Task R1 A1): 물체의 금지·취약 표지가 바뀌어도 모델의 입력은 한 글자도 바뀌지 않는다 — 그 사실은
    지시 문장이 말한다. (대조 쌍 `forbidden`의 sibling이 텍스트 없이 풀리지 않는다는 뜻이기도 하다.)"""

    def mutate(index, state):
        if index >= 1:
            state["objects"][1]["attributes"] = ["fragile", "forbidden"]

    plain = serialize_request(synthetic_stream(3), tokenizer, layout="stream_l1a")
    flipped = serialize_request(synthetic_stream(3, mutate=mutate), tokenizer, layout="stream_l1a")
    assert flipped["tokens"] == plain["tokens"]


def test_intro_and_dynamic_lines_refresh_on_their_periods_and_the_goal_text_on_its_own(tokenizer):
    out = serialize_request(synthetic_stream(31), tokenizer, layout="stream_l1a")
    rules = serialize_module.DELTA_RULES
    for index in range(31):
        intro = section_text(out, tokenizer, index, "state:objects_intro")
        dynamic = section_text(out, tokenizer, index, "state:objects_dynamic")
        goal = section_text(out, tokenizer, index, "state:goal")
        robot = section_text(out, tokenizer, index, "state:robot")
        assert (intro != "") == (index % rules["object_intro_period_ticks"] == 0), index
        assert (dynamic != "") == (index % rules["object_dynamic_period_ticks"] == 0), index
        assert ("text=" in goal) == (index > 0 and index % rules["goal_text_period_ticks"] == 0), index
        assert ("eq=" in robot) == (index % rules["object_intro_period_ticks"] == 0), index  # 말단 자세는 소개 틱과 변화 때만
    assert section_text(out, tokenizer, 30, "state:objects_intro").count("obj ") == 2
    assert section_text(out, tokenizer, 10, "state:objects_dynamic").count("\n") == 2
    # 갱신 틱의 동적 줄은 자세·정밀도·여유·통로뿐, 소개 틱의 동적 줄은 전체(표면 신뢰도까지)다.
    assert "conf=" not in section_text(out, tokenizer, 10, "state:objects_dynamic")
    assert section_text(out, tokenizer, 30, "state:objects_dynamic").count("conf=0.92") == 2
    assert "text=o0 상자를 zoneL로 옮겨라" in section_text(out, tokenizer, 10, "state:goal")


def test_delta_rules_can_be_overridden_but_only_by_known_keys(tokenizer):
    record = synthetic_stream(6)
    out = serialize_request(record, tokenizer, layout="stream_l1a", delta_rules={"object_dynamic_period_ticks": 2})
    assert out["delta_rules"]["object_dynamic_period_ticks"] == 2 and out["delta_rules"]["object_intro_period_ticks"] == 30
    assert [section_text(out, tokenizer, i, "state:objects_dynamic") != "" for i in range(6)] == [True, False, True, False, True, False]
    with pytest.raises(ValueError, match="delta_rules.bogus"):
        serialize_request(record, tokenizer, layout="stream_l1a", delta_rules={"bogus": 1})


def test_delta_rules_match_the_harness_config():
    """문턱·주기의 단일 출처는 하네스 설정의 `serialization:` 블록이다(레코드 지문에 든다); 모듈 상수가 그것을 비춘다."""
    from helpers import HARNESS_CONFIG

    config = yaml.safe_load(HARNESS_CONFIG.read_text(encoding="utf-8"))
    assert config["serialization"] == serialize_module.DELTA_RULES


def test_zones_and_scene_return_to_a_tick_only_when_they_change(tokenizer):
    def mutate(index, state):
        if index == 3:
            state["scene"]["clearance_mm"] = 12
        if index >= 5:
            state["zones"].append({"id": "zoneR", "desc": "오른쪽 정리 영역", "bounds_mm": [-120, -330, 180, -150]})

    out = serialize_request(synthetic_stream(7, mutate=mutate), tokenizer, layout="stream_l1a")
    scene = [section_text(out, tokenizer, i, "state:scene") for i in range(7)]
    zones = [section_text(out, tokenizer, i, "state:zones") for i in range(7)]
    assert scene[3] == "scene surf=-112 free=300 corr=100 clr=12\n" and scene[4] == "scene surf=-112 free=300 corr=100 clr=40\n"
    assert all(scene[i] == "" for i in (0, 1, 2, 5, 6))
    assert zones[5].count("zone ") == 2 and all(zones[i] == "" for i in (0, 1, 2, 3, 4, 6))


def test_events_and_waypoints_are_short_lines(tokenizer):
    def mutate(index, state):
        if index == 1:
            state["events"] = [{"kind": "object_moved", "sim_ms": 100, "object": "o1", "displacement_mm": 42}, {"kind": "reflex_force", "sim_ms": 100}]
            state["derived"].append({"waypoint": "w1", "kind": "over", "pos_mm": [150, 100, 60], "around": "o1", "extra_mm": 120})

    out = serialize_request(synthetic_stream(2, mutate=mutate), tokenizer, layout="stream_l1a")
    assert section_text(out, tokenizer, 1, "state:events") == "ev object_moved o=o1 d=42\nev reflex_force\n"
    assert section_text(out, tokenizer, 1, "state:waypoints") == "wp w1 over p=150,100,60 around=o1 extra=120\n"
    assert section_text(out, tokenizer, 0, "state:events") == "" and section_text(out, tokenizer, 0, "state:waypoints") == ""


def test_serializing_a_prefix_of_the_ticks_gives_a_prefix_of_the_tokens(streams, tokenizer):
    """변화분은 입력의 정의다: 틱 k의 토큰은 틱 0..k의 레코드만으로 정해지므로 앞 n틱의 직렬화는 전체의 접두다 —
    틱마다 새 토큰만 붙이는 증분 계산과 처음부터 계산이 같다 (docs/08 §3.1)."""
    for record in (copy.deepcopy(streams[0]), synthetic_stream(35, mutate=lambda i, s: s["objects"][0].__setitem__("pose_mm", [300 + 5 * i, 0, -80]))):
        record["ticks"] = record["ticks"][:35]
        whole = serialize_request(record, tokenizer, layout="stream_l1a")
        for n in (1, 2, 9, 10, 11, 30, 31, 34):
            head = copy.deepcopy(record)
            head["ticks"] = head["ticks"][:n]
            head["prefix"]["instructions"] = [ins for ins in head["prefix"]["instructions"] if int(ins["version"]) <= max(int((t["request"]["state"].get("goal") or {}).get("version", 1)) if isinstance(t["request"]["state"].get("goal"), dict) else 1 for t in head["ticks"])]
            part = serialize_request(head, tokenizer, layout="stream_l1a")
            end = whole["ticks"][n - 1]["end"]
            assert part["tokens"] == whole["tokens"][:end], n
            assert part["ticks"] == whole["ticks"][:n], n
            assert part["prefix_end"] == whole["prefix_end"]


def test_the_contract_doc_lists_every_short_field_name_of_format_v03():
    """docs/08 §3.2의 서식 v0.3 표는 `STREAM_FIELDS`의 짧은 이름을 전부 적어야 한다 — 문서가 코드와 같도록."""
    from pathlib import Path

    doc = (Path(__file__).resolve().parents[1] / "docs" / "08-streaming-io-and-data-contract.md").read_text(encoding="utf-8")
    section = doc[doc.index("### 3.2") : doc.index("### 3.3")]
    assert "v0.4" in section and "계약 v0.3" in doc[:2000]
    for group, table in serialize_module.STREAM_FIELDS.items():
        for short in table.values():
            if short:
                assert f"{short}=" in section or f" {short}" in section or f"[{short}]" in section, (group, short)
    for token in ("obj ", "commitment a=", "hist main=", "goal v", "t <tick> age g<ms> p<ms> seq <n>", "yaw", "seen", "prefix"):
        assert token in section, token


def test_legacy_candidate_entries_lose_their_prose_but_keep_the_key(tokenizer, stream):
    """다른 도구가 만든 틱(D0 fixture)의 옛 항목: 설명·산문 파생 값은 버리고 키(5조각이면 뒤 조각은 그대로)와 그 밖의 필드만."""
    from robo_jev.model.serialize import stream_candidate_line

    assert stream_candidate_line({"id": "c1", "action_ref": "c1", "key": "grasp:o7:top:zoneL:slow", "desc": "…", "derived": "reach ok"}) == "c1: grasp o7 top→zoneL slow\n"
    assert stream_candidate_line({"id": "c9", "action_ref": "c9", "key": "push:o4:+x:none"}) == "c9: push o4 +x\n"
    assert stream_candidate_line({"id": "c3", "action_ref": "c3", "key": "place:o0:release:zoneR", "d": 5, "clr": 6, "path": "ok", "g": 7}) == "c3: place o0 release→zoneR d=5 clr=6 g=7\n"
    assert stream_candidate_line({"id": "c3", "action_ref": "c3", "key": "place:o0:release:zoneR", "d": 5, "clr": 6, "path": "blocked"}, object_clearance={"o0": 6}) == "c3: place o0 release→zoneR d=5 path=blocked\n"
    assert stream_candidate_line({"id": "c3", "action_ref": "c3", "key": "place:o0:release:zoneR", "d": 5, "clr": 6}, object_clearance={"o0": 9}) == "c3: place o0 release→zoneR d=5 clr=6\n"
    assert stream_candidate_line({"id": "p1", "kind": "via", "ref": "w1", "action_ref": "c3", "desc": "경유"}) == "p1: via w1\n"
    assert stream_candidate_line({"id": "cx", "action_ref": "c3", "key": "hold"}) == "cx: hold action_ref=c3\n"  # 다른 후보를 가리키면 남는다
    out = serialize_request(stream, tokenizer, layout="stream_l1a")
    assert "c1: grasp o7 top→zoneL slow" in out["text"] and "빨간 컵을 top 면으로" not in out["text"]


def test_a_candidate_line_carries_the_clearance_when_a_neighbour_moved_but_the_target_did_not(tokenizer):
    """리뷰 2 C1: 후보 줄의 `clr` 생략은 **모델이 마지막으로 본** 대상 동적 줄의 값에 댄다. 이웃 o1이 정지한 대상 o0 쪽으로
    움직이면 o0의 여유가 40 → 12로 바뀌지만 o0의 동적 줄은 나가지 않는다(여유 변화는 촉발 조건이 아니다) — 그 틱에 후보 줄이
    새 값을 나르고, 다음 갱신 틱에 o0 줄이 12를 실은 뒤에야 다시 뺀다."""

    def mutate(index, state):
        if index >= 1:
            state["objects"][1]["pose_mm"] = [200, 120, -80]  # o1이 o0 쪽으로 100mm
            for item in state["derived"]:
                item["clearance_mm"] = 12

    record = synthetic_stream(12, mutate=mutate)
    for tick in record["ticks"][1:]:
        for entry in tick["request"]["candidates"]["q_main"]:
            if "clr" in entry:
                entry["clr"] = 12
    out = serialize_request(record, tokenizer, layout="stream_l1a")
    assert section_text(out, tokenizer, 1, "state:objects_dynamic") == "o1 p=200,120,-80 s=3 clr=12 corr=100\n"  # o0 줄은 없다
    assert "c1: grasp o0 top→zoneL d=381 clr=12" in tick_text(out, tokenizer, 1).splitlines()
    assert "c2: push o0 -x d=417 clr=12 path=blocked" in tick_text(out, tokenizer, 1).splitlines()
    for index in range(2, 10):  # 갱신 틱 전까지 틱마다 후보 줄이 새 값을 나른다
        assert "clr=12" in tick_text(out, tokenizer, index) and section_text(out, tokenizer, index, "state:objects_dynamic") == ""
    assert "o0 p=300,0,-80 s=3 clr=12 corr=100" in section_text(out, tokenizer, 10, "state:objects_dynamic")  # 갱신 틱: o0 줄이 12를 싣는다
    assert "c1: grasp o0 top→zoneL d=381" in tick_text(out, tokenizer, 10).splitlines() and "clr=12" not in tick_text(out, tokenizer, 11)
    # 대상 자신이 움직여 동적 줄이 나간 틱은 전처럼 뺀다.
    assert "c1: grasp o0 top→zoneL d=381" in tick_text(out, tokenizer, 0).splitlines()


def test_a_partial_dynamic_line_says_when_a_field_returned_to_its_default(tokenizer):
    """리뷰 1 I3: 변화분 줄은 바뀐 필드를 값이 기본값이라도 적는다 — 가시 비율 1→0.5→1과 1→0.5→0.5, 요 0→90→0과 90→90이
    같은 줄이 되면 안 된다. 기본값은 소개 틱의 전체 줄에서만 뺀다."""

    def visibility(back_to_one):
        def mutate(index, state):
            o2 = state["objects"][0]
            if index == 1:
                o2["visible_ratio"] = 0.5
            if index >= 2:
                o2["visible_ratio"] = 1.0 if back_to_one else 0.5
                o2["pose_mm"] = [320, 0, -80]  # 둘 다 같은 이동 — 가시 비율만 다르다
        return mutate

    returned = serialize_request(synthetic_stream(3, mutate=visibility(True)), tokenizer, layout="stream_l1a")
    stayed = serialize_request(synthetic_stream(3, mutate=visibility(False)), tokenizer, layout="stream_l1a")
    assert section_text(returned, tokenizer, 1, "state:objects_dynamic") == "o0 p=300,0,-80 s=3 vis=0.5 clr=40 corr=100\n"
    assert section_text(returned, tokenizer, 2, "state:objects_dynamic") == "o0 p=320,0,-80 s=3 vis=1 clr=40 corr=100\n"
    assert section_text(stayed, tokenizer, 2, "state:objects_dynamic") == "o0 p=320,0,-80 s=3 clr=40 corr=100\n"
    assert section_text(returned, tokenizer, 2, "state:objects_dynamic") != section_text(stayed, tokenizer, 2, "state:objects_dynamic")

    def yaw(back_to_zero):
        def mutate(index, state):
            o2 = state["objects"][0]
            if index == 1:
                o2["quat"] = [0.0, 0.0, 0.7071, 0.7071]  # z축 90도
            if index >= 2:
                o2["quat"] = [0.0, 0.0, 0.0, 1.0] if back_to_zero else [0.0, 0.0, 0.7071, 0.7071]
                o2["pose_mm"] = [320, 0, -80]
        return mutate

    returned = serialize_request(synthetic_stream(3, mutate=yaw(True)), tokenizer, layout="stream_l1a")
    stayed = serialize_request(synthetic_stream(3, mutate=yaw(False)), tokenizer, layout="stream_l1a")
    assert section_text(returned, tokenizer, 1, "state:objects_dynamic") == "o0 p=300,0,-80 yaw=90 s=3 clr=40 corr=100\n"
    assert section_text(returned, tokenizer, 2, "state:objects_dynamic") == "o0 p=320,0,-80 yaw=0 s=3 clr=40 corr=100\n"
    assert section_text(stayed, tokenizer, 2, "state:objects_dynamic") == "o0 p=320,0,-80 s=3 clr=40 corr=100\n"
    # 전체 줄(첫 틱·소개 틱)은 기본값을 뺀다.
    assert "yaw=" not in section_text(returned, tokenizer, 0, "state:objects_dynamic") and "vis=" not in section_text(returned, tokenizer, 0, "state:objects_dynamic")


def test_a_tilted_object_carries_its_quaternion_in_the_stream(tokenizer):
    """z축 회전이 아닌 자세는 `yaw`로 줄일 수 없어 `q=<quaternion>`(소수 2자리)으로 싣는다 — 바뀔 때와 소개 틱에."""

    def mutate(index, state):
        if index >= 1:
            state["objects"][0]["quat"] = [0.2588, 0.0, 0.0, 0.9659]  # x축 30도로 기울어짐
        if index >= 2:
            state["objects"][0]["pose_mm"] = [320, 0, -80]

    out = serialize_request(synthetic_stream(3, mutate=mutate), tokenizer, layout="stream_l1a")
    assert section_text(out, tokenizer, 1, "state:objects_dynamic") == "o0 p=300,0,-80 q=0.26,0.00,0.00,0.97 s=3 clr=40 corr=100\n"
    assert section_text(out, tokenizer, 2, "state:objects_dynamic") == "o0 p=320,0,-80 s=3 clr=40 corr=100\n"  # 기울기는 그대로: 이동만
    assert "yaw=" not in out["text"]


def test_an_object_that_leaves_the_tracker_is_announced_once_with_a_gone_line(tokenizer):
    """리뷰 1 M7: 변화분 틱에서 물체가 `objects[]`에서 빠지면 `<id> gone` 한 줄로 한 번 알린다; 다시 나타나면 처음 관측처럼 전체 줄."""

    def mutate(index, state):
        if 2 <= index <= 3:
            state["objects"] = [entry for entry in state["objects"] if entry["id"] != "o1"]
            state["derived"] = [item for item in state["derived"] if item["object"] != "o1"]

    out = serialize_request(synthetic_stream(6, mutate=mutate), tokenizer, layout="stream_l1a")
    assert section_text(out, tokenizer, 2, "state:objects_dynamic") == "o1 gone\n"
    assert section_text(out, tokenizer, 3, "state:objects_dynamic") == ""
    assert section_text(out, tokenizer, 4, "state:objects_intro").startswith("obj o1 ")
    assert section_text(out, tokenizer, 4, "state:objects_dynamic") == "o1 p=200,220,-80 s=3 conf=0.92 clr=40 corr=100\n"
    assert section_text(out, tokenizer, 5, "state:objects_dynamic") == ""
    assert out["text"].count(" gone") == 1


def test_non_empty_soft_token_slots_are_refused_instead_of_dropped(tokenizer):
    """리뷰 1 M6: `image`·`geom`은 soft token 슬롯이라 값이 있으면 조용히 버리지 않고 거절한다; `extractor`는 앞단 버전 문자열(겉봉투)만."""
    record = synthetic_stream(2)
    record["ticks"][1]["request"]["state"]["image"] = [{"camera": "cam0", "tokens": 16}]
    with pytest.raises(ValueError, match="state.image"):
        serialize_request(record, tokenizer, layout="stream_l1a")
    record = synthetic_stream(2)
    record["ticks"][0]["request"]["state"]["geom"] = [{"kind": "tsdf"}]
    with pytest.raises(ValueError, match="state.geom"):
        serialize_request(record, tokenizer, layout="stream_l1a")
    record = synthetic_stream(2)
    record["ticks"][0]["request"]["state"]["extractor"] = {"version": "pw0.1"}
    with pytest.raises(ValueError, match="state.extractor"):
        serialize_request(record, tokenizer, layout="stream_l1a")
    fine = serialize_request(synthetic_stream(2), tokenizer, layout="stream_l1a")  # extractor "pw0.1", 빈 슬롯
    assert "pw0.1" not in fine["text"]


def test_the_serializers_joint_key_parser_matches_the_harness():
    from robo_jev.harness.robot import joint_key_parts as harness_parts
    from robo_jev.model.serialize import joint_key_parts

    for key in ("grasp:o7:top:zoneL", "push:o4:+x:none", "place:o0:release:zoneR:slow", "hold", "observe", "", None, "grasp:o7"):
        assert joint_key_parts(key) == harness_parts(key), key


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
