"""입력·라벨 계약 검사.

거절 검사를 먼저 쓰고(RED) 구현을 붙인다(GREEN). 모든 거절은 필드 경로가 들어간
`ValueError`여야 하므로, 검사마다 경로 조각을 `match`로 확인한다.
"""

import copy
import json
from pathlib import Path

import pytest
from helpers import all_keys

from robo_jev.contracts import (
    FORBIDDEN_REQUEST_KEYS,
    NON_INPUT_FIELDS,
    PROFILE_LIMITS,
    QUESTION_SET_V0,
    model_input,
    validate_record,
)

# --------------------------------------------------------------------------
# 최소 레코드 (doc 04 §1 / doc 08 §8 예시 축약)
# --------------------------------------------------------------------------


def single_record() -> dict:
    """단일 요청 레코드 한 건. 호출할 때마다 새 dict를 만든다."""
    return {
        "schema_version": "judgment-v0",
        "origin_group": "scene-family-018",
        "split": "train",
        "request": {
            "request_id": "r018-07",
            "state": {
                "goal": "빨간 물체를 왼쪽 영역으로 옮긴다",
                "observed_at_ms": 1200,
                "objects": [
                    {"id": "o7", "color": "red", "visible": True},
                    {"id": "o2", "color": "blue", "visible": True},
                ],
            },
            "questions": [
                {
                    "id": "q_target",
                    "type": "choice",
                    "instructions": "현재 목표에서 옮겨야 하는 물체를 고르라.",
                    "criteria": [
                        {"id": "c91", "description": "관측된 빨간 물체 o7", "ref": "o7"},
                        {"id": "c14", "description": "관측된 파란 물체 o2", "ref": "o2"},
                        {"id": "c62", "description": "제공된 정보로 대상을 결정할 수 없음"},
                    ],
                }
            ],
        },
        "labels": [
            {
                "question_id": "q_target",
                "kind": "valid_set",
                "candidate_ids": ["c91"],
                "source": "goal-object-rule-v0",
                "mask": True,
            }
        ],
        "provenance": {"origin_group": "scene-family-018", "rules": "r0.4"},
        "evidence": {"rule_trace": "goal-object-rule-v0", "true_state": {"o7": [120, -40, 742]}},
        "usage": {"questions_used": ["q_target"]},
    }


def ordinal_question() -> dict:
    return {
        "id": "q_speed",
        "type": "ordinal",
        "instructions": "속도 수준을 고르라.",
        "criteria": [
            {"id": "0", "description": "정지", "value": 0.0},
            {"id": "1", "description": "저속", "value": 0.1},
            {"id": "2", "description": "중속", "value": 0.25},
            {"id": "3", "description": "고속", "value": 0.5},
        ],
    }


def stream_record() -> dict:
    """에피소드 스트림 레코드 (2틱). 호출할 때마다 새 dict를 만든다."""
    q_main = [
        {"id": "c3", "action_ref": "c3", "key": "grasp:o7:top:zoneL", "desc": "o7 윗면 파지"},
        {"id": "c5", "action_ref": "c5", "key": "grasp:o2:side:zoneL", "desc": "o2 측면 파지"},
        {"id": "c9", "action_ref": "c9", "key": "push:o4:+x:none", "desc": "o4 밀기"},
        {"id": "c0", "action_ref": "c0", "key": "hold", "desc": "현 상태 유지"},
    ]
    q_path = [
        {"id": "p0", "kind": "direct", "action_ref": "c3"},
        {"id": "p1", "kind": "via", "ref": "w2", "action_ref": "c3"},
    ]
    state = {
        "goal": {"text": "빨간 컵을 왼쪽 영역으로", "version": 1},
        "objects": [{"id": "o7", "desc": "빨간 컵", "pose_mm": [310, -40, 742], "visible_ratio": 1.0}],
        "robot": {"ee_pose_mm": [200, 0, 900], "gripper_mm": 80, "holding": None},
    }
    tick = {
        "t": 137,
        "sim_ms": 13700,
        "observed_at_ms": 13680,
        "obs_age_ms": 20,
        "request": {
            "state": state,
            "exec_history": "main=c3 phase=approach path=direct speed=2 force=0 gripper=open stop=0 ack=ok",
            "commitment": {"action_ref": "c3", "key": "grasp:o7:top:zoneL", "phase": "approach", "held_ticks": 12},
            "candidates": {"q_main": q_main, "q_path": q_path},
        },
        "model_output": {"q_main": {"c3": 0.71, "c5": 0.22, "c9": 0.07}, "q_stop": 0.01},
        "adopted": {"main": "c3", "switch": False, "path": "p0", "speed": 2, "force": 0, "gripper": "open", "stop": False},
        "ack": {"seq": 137, "applied": True, "gripper_event": None},
        "labels": [
            {
                "question_id": "q_main",
                "kind": "valid_set",
                "candidate_ids": ["c3"],
                "semantic_admissible": ["c3", "c5"],
                "unknown": ["c9"],
                "event_results": {
                    "c3": {"s": 7, "f": 1, "seeds": [1, 1, 1, 1, 0, 1, 1, 1]},
                    "c5": {"s": 4, "f": 4, "seeds": [1, 0, 0, 1, 1, 0, 1, 0]},
                },
                "label_confidence": "high",
                "rule": "admissible-then-performance-v1+commitment-v1",
            },
            {"question_id": "q_gripper", "kind": "valid_set", "candidate_ids": ["open"], "conditioned_on": "c3/approach"},
            {"question_id": "q_stop", "kind": "single", "answer": False},
            {"question_id": "q_speed", "kind": "valid_set", "candidate_ids": ["2", "1"], "conditioned_on": "c3/approach"},
        ],
    }
    second = copy.deepcopy(tick)
    second["t"] = 138
    second["sim_ms"] = 13800
    second["observed_at_ms"] = 13780
    return {
        "schema_version": "stream-v0",
        "episode_id": "ep-0412",
        "origin_group": "scene-family-031",
        "split": "train",
        "versions": {"harness": "h0.3", "controller": "c0.2", "expert": "e0.1"},
        "prefix": {
            "instructions": [{"version": 1, "t_ms": 0, "text": "빨간 컵을 왼쪽 영역으로 옮겨라"}],
            "question_set": "qs-v0",
        },
        "ticks": [tick, second],
        "evidence": {"expert_log": "…", "rollouts": "…"},
    }


def first_record():
    return json.loads(Path("tests/fixtures/d0.jsonl").read_text().splitlines()[0])


# --------------------------------------------------------------------------
# 계획서에 명시된 두 검사 (원문 그대로)
# --------------------------------------------------------------------------


def test_future_label_cannot_change_model_input():
    a = first_record()
    b = copy.deepcopy(a)
    b["evidence"] = {"future_success": True}
    b["labels"] = []
    assert model_input(a) == model_input(b)


def test_unknown_answer_is_rejected():
    r = first_record()  # 첫 fixture는 choice + valid_set으로 고정
    r["labels"][0]["candidate_ids"] = ["absent-candidate"]
    with pytest.raises(ValueError, match="candidate_ids"):
        validate_record(r)


# --------------------------------------------------------------------------
# 통과해야 하는 레코드
# --------------------------------------------------------------------------


def test_single_request_example_is_valid():
    validate_record(single_record())


def test_stream_example_is_valid():
    validate_record(stream_record())


def test_missing_label_is_loss_mask():
    """라벨이 없는 질문은 허용(loss mask)이며, mask=false 라벨도 허용한다."""
    record = single_record()
    record["request"]["questions"].append(ordinal_question())
    validate_record(record)  # q_speed 라벨 없음

    record["labels"][0]["mask"] = False
    validate_record(record)

    del record["labels"]
    validate_record(record)  # labels 필드 자체가 없어도 된다


def test_first_fixture_line_is_choice_with_valid_set():
    record = first_record()
    assert record["request"]["questions"][0]["type"] == "choice"
    assert record["labels"][0]["kind"] == "valid_set"


# --------------------------------------------------------------------------
# 거절: 단일 요청
# --------------------------------------------------------------------------


def test_unknown_schema_version_is_rejected():
    record = single_record()
    record["schema_version"] = "judgment-v9"
    with pytest.raises(ValueError, match="schema_version"):
        validate_record(record)


def test_nonexistent_candidate_id_is_rejected():
    record = single_record()
    record["labels"][0]["candidate_ids"] = ["c91", "c00"]
    with pytest.raises(ValueError, match=r"labels\[0\]\.candidate_ids"):
        validate_record(record)


def test_duplicate_candidate_ids_are_rejected():
    record = single_record()
    record["request"]["questions"][0]["criteria"][1]["id"] = "c91"
    with pytest.raises(ValueError, match=r"request\.questions\[0\]\.criteria\[1\]\.id"):
        validate_record(record)


def test_duplicate_ids_inside_a_valid_set_are_rejected():
    record = single_record()
    record["labels"][0]["candidate_ids"] = ["c91", "c91"]
    with pytest.raises(ValueError, match=r"labels\[0\]\.candidate_ids\[1\]"):
        validate_record(record)


def test_empty_candidate_list_is_rejected():
    record = single_record()
    record["request"]["questions"][0]["criteria"] = []
    with pytest.raises(ValueError, match=r"request\.questions\[0\]\.criteria"):
        validate_record(record)


def test_empty_valid_set_is_rejected():
    record = single_record()
    record["labels"][0]["candidate_ids"] = []
    with pytest.raises(ValueError, match=r"labels\[0\]\.candidate_ids"):
        validate_record(record)


def test_ordinal_criterion_without_value_is_rejected():
    record = single_record()
    question = ordinal_question()
    del question["criteria"][2]["value"]
    record["request"]["questions"] = [question]
    record["labels"] = []
    with pytest.raises(ValueError, match=r"request\.questions\[0\]\.criteria\[2\]\.value"):
        validate_record(record)


def test_ordinal_values_must_increase():
    record = single_record()
    question = ordinal_question()
    question["criteria"][2]["value"] = 0.05
    record["request"]["questions"] = [question]
    record["labels"] = []
    with pytest.raises(ValueError, match=r"request\.questions\[0\]\.criteria\[2\]\.value"):
        validate_record(record)


def test_distribution_sum_mismatch_is_rejected():
    record = single_record()
    record["labels"] = [
        {
            "question_id": "q_target",
            "kind": "distribution",
            "probabilities": {"c91": 0.5, "c14": 0.2},
        }
    ]
    with pytest.raises(ValueError, match=r"labels\[0\]\.probabilities"):
        validate_record(record)


def test_distribution_probability_out_of_range_is_rejected():
    record = single_record()
    record["labels"] = [
        {
            "question_id": "q_target",
            "kind": "distribution",
            "probabilities": {"c91": 1.4, "c14": -0.4},
        }
    ]
    with pytest.raises(ValueError, match=r"labels\[0\]\.probabilities\.c91"):
        validate_record(record)


def test_distribution_over_unknown_candidate_is_rejected():
    record = single_record()
    record["labels"] = [
        {
            "question_id": "q_target",
            "kind": "distribution",
            "probabilities": {"c91": 0.5, "nope": 0.5},
        }
    ]
    with pytest.raises(ValueError, match=r"labels\[0\]\.probabilities\.nope"):
        validate_record(record)


def test_distribution_over_a_subset_is_allowed():
    record = single_record()
    record["labels"] = [
        {"question_id": "q_target", "kind": "distribution", "probabilities": {"c91": 0.7, "c14": 0.3}}
    ]
    validate_record(record)


def test_single_answer_must_be_an_existing_candidate_id():
    record = single_record()
    record["labels"] = [{"question_id": "q_target", "kind": "single", "answer": "c00"}]
    with pytest.raises(ValueError, match=r"labels\[0\]\.answer"):
        validate_record(record)


def test_boolean_single_answer_must_be_a_bool():
    record = single_record()
    record["request"]["questions"] = [
        {
            "id": "q_done",
            "type": "boolean",
            "instructions": "현재 목표를 이미 만족했는가.",
            "criteria": [{"id": "true", "description": "예"}, {"id": "false", "description": "아니오"}],
        }
    ]
    record["labels"] = [{"question_id": "q_done", "kind": "single", "answer": "true"}]
    with pytest.raises(ValueError, match=r"labels\[0\]\.answer"):
        validate_record(record)

    record["labels"][0]["answer"] = True
    validate_record(record)


def test_event_counts_must_be_non_negative_ints():
    record = single_record()
    record["labels"] = [
        {
            "question_id": "q_target",
            "kind": "event",
            "event_id": "ev-1",
            "successes": 7,
            "failures": -1,
            "censored": 0,
        }
    ]
    with pytest.raises(ValueError, match=r"labels\[0\]\.failures"):
        validate_record(record)


def test_event_label_needs_an_event_reference():
    record = single_record()
    record["labels"] = [
        {"question_id": "q_target", "kind": "event", "successes": 7, "failures": 1, "censored": 0}
    ]
    with pytest.raises(ValueError, match=r"labels\[0\]\.event_id"):
        validate_record(record)


def test_label_for_unknown_question_is_rejected():
    record = single_record()
    record["labels"][0]["question_id"] = "q_nope"
    with pytest.raises(ValueError, match=r"labels\[0\]\.question_id"):
        validate_record(record)


def test_unknown_label_kind_is_rejected():
    record = single_record()
    record["labels"][0]["kind"] = "ranking"
    with pytest.raises(ValueError, match=r"labels\[0\]\.kind"):
        validate_record(record)


def test_unknown_question_type_is_rejected():
    record = single_record()
    record["request"]["questions"][0]["type"] = "freeform"
    with pytest.raises(ValueError, match=r"request\.questions\[0\]\.type"):
        validate_record(record)


def test_missing_required_request_field_is_rejected():
    record = single_record()
    del record["request"]["state"]
    with pytest.raises(ValueError, match=r"request\.state"):
        validate_record(record)


def test_duplicate_question_ids_are_rejected():
    record = single_record()
    record["request"]["questions"].append(copy.deepcopy(record["request"]["questions"][0]))
    with pytest.raises(ValueError, match=r"request\.questions\[1\]\.id"):
        validate_record(record)


def test_label_structure_inside_request_is_rejected():
    """정답 구조가 입력 영역에 섞이면 거절한다."""
    record = single_record()
    record["request"]["state"]["hint"] = {"question_id": "q_target", "kind": "valid_set", "candidate_ids": ["c91"]}
    with pytest.raises(ValueError, match=r"request\.state\.hint"):
        validate_record(record)


@pytest.mark.parametrize(
    "payload",
    [
        {"question_id": "q_target", "candidate_ids": ["c91"]},
        {"question_id": "q_target", "answer": "c91"},
        {"question_id": "q_target", "probabilities": {"c91": 1.0}},
        {"question_id": "q_target", "successes": 7},
        {"question_id": "q_target", "kind": "valid_set"},
    ],
    ids=["candidate_ids", "answer", "probabilities", "successes", "kind"],
)
def test_partial_label_structure_inside_request_is_rejected(payload):
    """`kind`가 없어도 `question_id` + 정답 모양이면 라벨 유출이다."""
    record = single_record()
    record["request"]["state"]["hint"] = payload
    with pytest.raises(ValueError, match=r"request\.state\.hint"):
        validate_record(record)


def test_non_input_field_inside_request_is_rejected():
    record = single_record()
    record["request"]["evidence"] = {"rule_trace": "…"}
    with pytest.raises(ValueError, match=r"request\.evidence"):
        validate_record(record)


@pytest.mark.parametrize(
    "field",
    ["labels", "provenance", "evidence", "split", "usage", "model_output", "adopted", "ack", "true_state", "occluded_true_poses"],
)
def test_nested_non_input_field_inside_request_is_rejected(field):
    """허용 목록은 재귀여야 한다: request 안쪽 어디에 숨겨도 걸러야 한다."""
    record = single_record()
    record["request"]["state"][field] = {"future_success": True}
    with pytest.raises(ValueError, match=rf"request\.state\.{field}"):
        validate_record(record)


def test_non_input_field_deep_inside_request_is_rejected():
    record = single_record()
    record["request"]["questions"][0]["criteria"][1]["evidence"] = {"rule_trace": "…"}
    with pytest.raises(ValueError, match=r"request\.questions\[0\]\.criteria\[1\]\.evidence"):
        validate_record(record)


# --------------------------------------------------------------------------
# 거절: 스트림
# --------------------------------------------------------------------------


def test_commitment_action_ref_must_exist():
    record = stream_record()
    record["ticks"][0]["request"]["commitment"]["action_ref"] = "c99"
    with pytest.raises(ValueError, match=r"ticks\[0\]\.request\.commitment\.action_ref"):
        validate_record(record)


def test_commitment_may_be_null():
    record = stream_record()
    tick = record["ticks"][0]
    tick["request"]["commitment"] = None
    tick["labels"] = [label for label in tick["labels"] if label["question_id"] in {"q_main", "q_stop"}]
    validate_record(record)


def test_unknown_phase_is_rejected():
    record = stream_record()
    record["ticks"][0]["request"]["commitment"]["phase"] = "hover"
    with pytest.raises(ValueError, match=r"ticks\[0\]\.request\.commitment\.phase"):
        validate_record(record)


def test_candidate_action_ref_must_exist():
    record = stream_record()
    record["ticks"][0]["request"]["candidates"]["q_path"][1]["action_ref"] = "c77"
    with pytest.raises(ValueError, match=r"ticks\[0\]\.request\.candidates\.q_path\[1\]\.action_ref"):
        validate_record(record)


def test_adopted_action_ref_must_exist():
    record = stream_record()
    record["ticks"][0]["adopted"]["main"] = "c77"
    with pytest.raises(ValueError, match=r"ticks\[0\]\.adopted\.main"):
        validate_record(record)


def test_label_action_ref_must_exist():
    record = stream_record()
    record["ticks"][0]["labels"].append(
        {
            "question_id": "q_main",
            "kind": "event",
            "action_ref": "c77",
            "successes": 5,
            "failures": 3,
            "censored": 0,
        }
    )
    with pytest.raises(ValueError, match=r"ticks\[0\]\.labels\[4\]\.action_ref"):
        validate_record(record)


def test_conditioned_on_must_match_the_commitment():
    record = stream_record()
    record["ticks"][1]["labels"][3]["conditioned_on"] = "c3/transport"
    with pytest.raises(ValueError, match=r"ticks\[1\]\.labels\[3\]\.conditioned_on"):
        validate_record(record)


def test_aux_label_needs_conditioned_on():
    record = stream_record()
    del record["ticks"][0]["labels"][1]["conditioned_on"]
    with pytest.raises(ValueError, match=r"ticks\[0\]\.labels\[1\]\.conditioned_on"):
        validate_record(record)


def test_aux_label_without_commitment_is_rejected():
    record = stream_record()
    record["ticks"][0]["request"]["commitment"] = None
    with pytest.raises(ValueError, match=r"ticks\[0\]\.labels\[1\]\.conditioned_on"):
        validate_record(record)


def test_label_copied_into_exec_history_is_rejected():
    """실행 이력과 라벨의 혼동."""
    record = stream_record()
    record["ticks"][0]["request"]["exec_history"] = copy.deepcopy(record["ticks"][0]["labels"][0])
    with pytest.raises(ValueError, match=r"ticks\[0\]\.request\.exec_history"):
        validate_record(record)


def test_label_copied_into_candidates_is_rejected():
    record = stream_record()
    record["ticks"][0]["request"]["candidates"]["q_main"][0]["answer"] = copy.deepcopy(
        record["ticks"][0]["labels"][0]
    )
    with pytest.raises(ValueError, match=r"ticks\[0\]\.request\.candidates\.q_main\[0\]\.answer"):
        validate_record(record)


def test_label_copied_into_adopted_is_rejected():
    record = stream_record()
    record["ticks"][0]["adopted"] = copy.deepcopy(record["ticks"][0]["labels"][0])
    with pytest.raises(ValueError, match=r"ticks\[0\]\.adopted"):
        validate_record(record)


def test_tick_numbers_must_strictly_increase():
    record = stream_record()
    record["ticks"][1]["t"] = record["ticks"][0]["t"]
    with pytest.raises(ValueError, match=r"ticks\[1\]\.t"):
        validate_record(record)


def test_unknown_set_must_be_disjoint_from_valid_set():
    record = stream_record()
    record["ticks"][0]["labels"][0]["unknown"] = ["c3", "c9"]
    with pytest.raises(ValueError, match=r"ticks\[0\]\.labels\[0\]\.unknown"):
        validate_record(record)


def test_semantic_admissible_must_reference_candidates():
    record = stream_record()
    record["ticks"][0]["labels"][0]["semantic_admissible"] = ["c3", "c88"]
    with pytest.raises(ValueError, match=r"ticks\[0\]\.labels\[0\]\.semantic_admissible\[1\]"):
        validate_record(record)


def test_event_results_must_reference_candidates():
    record = stream_record()
    record["ticks"][0]["labels"][0]["event_results"]["c88"] = {"s": 1, "f": 1}
    with pytest.raises(ValueError, match=r"ticks\[0\]\.labels\[0\]\.event_results\.c88"):
        validate_record(record)


def test_event_results_counts_must_be_non_negative():
    record = stream_record()
    record["ticks"][0]["labels"][0]["event_results"]["c3"]["f"] = -2
    with pytest.raises(ValueError, match=r"ticks\[0\]\.labels\[0\]\.event_results\.c3\.f"):
        validate_record(record)


def test_label_confidence_domain_is_checked():
    record = stream_record()
    record["ticks"][0]["labels"][0]["label_confidence"] = "very-high"
    with pytest.raises(ValueError, match=r"ticks\[0\]\.labels\[0\]\.label_confidence"):
        validate_record(record)


def test_missing_dynamic_candidate_list_is_rejected():
    record = stream_record()
    del record["ticks"][0]["request"]["candidates"]["q_path"]
    record["ticks"][0]["labels"].append(
        {"question_id": "q_path", "kind": "valid_set", "candidate_ids": ["p0"], "conditioned_on": "c3/approach"}
    )
    with pytest.raises(ValueError, match=r"ticks\[0\]\.request\.candidates\.q_path"):
        validate_record(record)


def test_candidates_for_unknown_question_are_rejected():
    record = stream_record()
    record["ticks"][0]["request"]["candidates"]["q_dance"] = [{"id": "x"}]
    with pytest.raises(ValueError, match=r"ticks\[0\]\.request\.candidates\.q_dance"):
        validate_record(record)


def test_instruction_versions_must_increase():
    record = stream_record()
    record["prefix"]["instructions"].append({"version": 1, "t_ms": 5500, "text": "유리잔은 건드리지 마라"})
    with pytest.raises(ValueError, match=r"prefix\.instructions\[1\]\.version"):
        validate_record(record)


def test_unknown_instruction_field_is_rejected():
    record = stream_record()
    record["prefix"]["instructions"][0]["note"] = "검수자가 남긴 메모"
    with pytest.raises(ValueError, match=r"prefix\.instructions\[0\]\.note"):
        validate_record(record)


def test_unknown_prefix_field_is_rejected():
    record = stream_record()
    record["prefix"]["question_bank"] = "qs-v9"
    with pytest.raises(ValueError, match=r"prefix\.question_bank"):
        validate_record(record)


@pytest.mark.parametrize(
    "field",
    ["labels", "provenance", "evidence", "split", "usage", "model_output", "adopted", "ack", "origin_group", "versions", "true_state", "occluded_true_poses"],
)
def test_non_input_field_inside_prefix_instructions_is_rejected(field):
    """prefix도 모델 입력이다: 지시 안에 비입력 필드를 숨길 수 없다."""
    record = stream_record()
    record["prefix"]["instructions"][0][field] = {"future_success": True}
    with pytest.raises(ValueError, match=rf"prefix\.instructions\[0\]\.{field}"):
        validate_record(record)


@pytest.mark.parametrize(
    "field",
    ["labels", "provenance", "evidence", "split", "usage", "model_output", "adopted", "ack", "origin_group", "versions", "true_state", "occluded_true_poses"],
)
def test_non_input_field_inside_prefix_is_rejected(field):
    record = stream_record()
    record["prefix"][field] = {"future_success": True}
    with pytest.raises(ValueError, match=rf"prefix\.{field}"):
        validate_record(record)


def test_label_structure_inside_prefix_is_rejected():
    record = stream_record()
    record["prefix"]["hint"] = {"question_id": "q_main", "candidate_ids": ["c3"]}
    with pytest.raises(ValueError, match=r"prefix\.hint"):
        validate_record(record)


def test_label_structure_inside_an_instruction_is_rejected():
    record = stream_record()
    record["prefix"]["instructions"][0]["hint"] = {"question_id": "q_main", "answer": "c3"}
    with pytest.raises(ValueError, match=r"prefix\.instructions\[0\]\.hint"):
        validate_record(record)


def test_unknown_key_inside_a_tick_request_is_rejected():
    record = stream_record()
    record["ticks"][0]["request"]["hint"] = {"best": "c3"}
    with pytest.raises(ValueError, match=r"ticks\[0\]\.request\.hint"):
        validate_record(record)


@pytest.mark.parametrize(
    "field",
    ["labels", "provenance", "evidence", "adopted", "ack", "true_state", "occluded_true_poses"],
)
def test_nested_non_input_field_inside_a_tick_request_is_rejected(field):
    record = stream_record()
    record["ticks"][1]["request"]["state"][field] = {"o7": [310, -40, 742]}
    with pytest.raises(ValueError, match=rf"ticks\[1\]\.request\.state\.{field}"):
        validate_record(record)


def test_non_input_field_inside_tick_candidates_is_rejected():
    record = stream_record()
    record["ticks"][0]["request"]["candidates"]["q_main"][2]["evidence"] = {"future_success": True}
    with pytest.raises(ValueError, match=r"ticks\[0\]\.request\.candidates\.q_main\[2\]\.evidence"):
        validate_record(record)


def test_partial_label_structure_inside_a_tick_request_is_rejected():
    record = stream_record()
    record["ticks"][0]["request"]["state"]["robot"]["hint"] = {
        "question_id": "q_main",
        "candidate_ids": ["c3"],
    }
    with pytest.raises(ValueError, match=r"ticks\[0\]\.request\.state\.robot\.hint"):
        validate_record(record)


# --------------------------------------------------------------------------
# 정보 경계
# --------------------------------------------------------------------------


def test_model_input_of_single_request_is_allow_listed():
    record = single_record()
    got = model_input(record)
    assert set(got) == {"schema_version", "request"}
    assert set(got["request"]) == {"request_id", "state", "questions"}
    assert set(got["request"]["questions"][0]) == {"id", "type", "instructions", "criteria"}
    assert set(got["request"]["questions"][0]["criteria"][0]) == {"id", "description", "ref"}
    assert set(got["request"]["questions"][0]["criteria"][2]) == {"id", "description"}


def test_model_input_of_stream_is_allow_listed():
    record = stream_record()
    got = model_input(record)
    assert set(got) == {"schema_version", "prefix", "ticks"}
    assert set(got["prefix"]) == {"instructions", "question_set"}
    assert set(got["prefix"]["instructions"][0]) == {"version", "t_ms", "text"}
    assert set(got["ticks"][0]) == {"t", "sim_ms", "observed_at_ms", "obs_age_ms", "request"}
    assert set(got["ticks"][0]["request"]) == {"state", "exec_history", "commitment", "candidates"}


@pytest.mark.parametrize("factory", [single_record, stream_record], ids=["single", "stream"])
def test_model_input_excludes_non_input_fields(factory):
    record = factory()
    if "prefix" in record:
        # prefix도 투영 대상이다: 지시에 비입력 필드를 섞어도 입력에 남지 않는다.
        record["prefix"]["provenance"] = {"rules": "r0.4"}
        record["prefix"]["instructions"][0]["evidence"] = {"future_success": True}
    got = model_input(record)
    assert all_keys(got).isdisjoint(NON_INPUT_FIELDS)
    assert all_keys(got).isdisjoint(FORBIDDEN_REQUEST_KEYS)  # 가려진 참값 키까지


@pytest.mark.parametrize("factory", [single_record, stream_record], ids=["single", "stream"])
def test_model_input_does_not_mutate_the_record(factory):
    record = factory()
    before = copy.deepcopy(record)
    got = model_input(record)
    assert record == before

    # 반환값은 새 객체다: 고쳐도 원본이 흔들리지 않는다.
    got["schema_version"] = "tampered"
    if "request" in got:
        got["request"]["state"]["goal"] = "tampered"
    else:
        got["ticks"][0]["request"]["state"]["goal"] = "tampered"
    assert record == before


@pytest.mark.parametrize("factory", [single_record, stream_record], ids=["single", "stream"])
def test_hidden_truth_change_keeps_model_input_byte_identical(factory):
    """가려진 참값은 evidence에만 있으므로 바꿔도 입력이 변하지 않는다."""
    record = factory()
    record["evidence"] = {
        "true_state": {"o7": [310, -40, 742]},
        "occluded_true_poses": {"o5": [-220, 190, 741]},
    }
    other = copy.deepcopy(record)
    other["evidence"]["true_state"]["o7"] = [-999, 999, 700]
    other["evidence"]["occluded_true_poses"]["o5"] = [1, 2, 3]

    dumped = json.dumps(model_input(record), ensure_ascii=False, sort_keys=False)
    assert dumped == json.dumps(model_input(other), ensure_ascii=False, sort_keys=False)


def test_model_input_rejects_unknown_schema_version():
    with pytest.raises(ValueError, match="schema_version"):
        model_input({"schema_version": "nope"})


@pytest.mark.parametrize(
    "record",
    [
        {"schema_version": "judgment-v0"},
        {"schema_version": "judgment-v0", "request": "nope"},
        {"schema_version": "judgment-v0", "request": None},
    ],
    ids=["missing", "string", "null"],
)
def test_model_input_rejects_a_record_without_a_usable_request(record):
    """request가 없거나 dict가 아니면 조용히 빈 입력을 만들지 않고 경로가 붙은 오류를 낸다."""
    with pytest.raises(ValueError, match=r"^request: "):
        model_input(record)


@pytest.mark.parametrize(
    "record",
    [
        {"schema_version": "stream-v0", "prefix": {}},
        {"schema_version": "stream-v0", "prefix": {}, "ticks": {"t": 0}},
    ],
    ids=["missing", "dict"],
)
def test_model_input_rejects_a_stream_without_usable_ticks(record):
    with pytest.raises(ValueError, match=r"^ticks: "):
        model_input(record)


def test_model_input_rejects_a_request_without_questions():
    record = single_record()
    del record["request"]["questions"]
    with pytest.raises(ValueError, match=r"^request\.questions: "):
        model_input(record)


def test_model_input_rejects_a_question_without_criteria():
    record = single_record()
    del record["request"]["questions"][0]["criteria"]
    with pytest.raises(ValueError, match=r"^request\.questions\[0\]\.criteria: "):
        model_input(record)


def test_model_input_rejects_a_tick_without_a_usable_request():
    record = {"schema_version": "stream-v0", "prefix": {}, "ticks": [{"t": 0}]}
    with pytest.raises(ValueError, match=r"^ticks\[0\]\.request: "):
        model_input(record)


def test_question_set_v0_has_the_fixed_ids():
    assert list(QUESTION_SET_V0) == [
        "q_main",
        "q_done",
        "q_instr",
        "q_observe",
        "q_retry",
        "q_stop",
        "q_gripper",
        "q_path",
        "q_speed",
        "q_force",
    ]


# --------------------------------------------------------------------------
# 프로파일 상한 Q≤16 · K≤32 (docs/06 Task 5 전제 5)
# --------------------------------------------------------------------------


def test_profile_limits_reject_too_many_questions_and_too_many_candidates():
    """`validate_record`가 질문 수(Q)와 후보 수(K)의 프로파일 상한을 강제한다 — 넘는 레코드는 경로와 실제 수를 말하며 거절된다."""
    record = single_record()
    template = record["request"]["questions"][0]
    record["request"]["questions"] = [template] + [
        {**copy.deepcopy(template), "id": f"q{index}"} for index in range(PROFILE_LIMITS["max_questions"] - 1)
    ]
    validate_record(record)  # 상한과 같은 수는 통과한다

    record["request"]["questions"].append({**copy.deepcopy(template), "id": "q_over"})
    with pytest.raises(ValueError, match=r"^request\.questions: 질문 수가 프로파일 상한을 넘는다: 17 > Q≤16"):
        validate_record(record)

    wide = single_record()
    wide["request"]["questions"][0]["criteria"] = [
        {"id": f"c{index}", "description": f"후보 {index}"} for index in range(PROFILE_LIMITS["max_candidates"] + 1)
    ]
    wide["labels"][0]["candidate_ids"] = ["c0"]
    with pytest.raises(ValueError, match=r"^request\.questions\[0\]\.criteria: 후보 수가 프로파일 상한을 넘는다: 33 > K≤32"):
        validate_record(wide)


def test_profile_limits_apply_to_stream_tick_candidates_and_are_profile_arguments():
    """스트림은 틱의 후보 목록에 걸리고, `limits`로 다른 프로파일을 줄 수 있다(상한을 넓히는 것은 호출자의 명시적 선택이다)."""
    record = stream_record()
    entries = record["ticks"][0]["request"]["candidates"]["q_main"]
    base = entries[0]
    entries.extend({**copy.deepcopy(base), "id": f"x{index}", "action_ref": f"x{index}"} for index in range(29))
    assert len(entries) == 33
    with pytest.raises(ValueError, match=r"^ticks\[0\]\.request\.candidates\.q_main: 후보 수가 프로파일 상한을 넘는다: 33 > K≤32"):
        validate_record(record)
    validate_record(record, limits={"max_questions": 16, "max_candidates": 64})
    with pytest.raises(ValueError, match=r"prefix\.question_set: 질문 세트 v0의 질문 수가 프로파일 상한을 넘는다: 10 > Q≤4"):
        validate_record(stream_record(), limits={"max_questions": 4, "max_candidates": 64})


def test_d1_records_stay_inside_the_profile_limits(streams, singles):
    """D0 fixture(계약이 같은 꼴)의 레코드는 상한 안에 있다 — D1 실측 최댓값은 보고서에 있다(스트림 K≤12, 비로봇 Q≤16·K≤14)."""
    for record in [*streams, *singles]:
        validate_record(record)
        questions = (
            record["request"]["questions"] if record["schema_version"] == "judgment-v0"
            else [entries for tick in record["ticks"] for entries in tick["request"]["candidates"].values()]
        )
        assert len(questions) <= PROFILE_LIMITS["max_questions"] or record["schema_version"] == "stream-v0"
