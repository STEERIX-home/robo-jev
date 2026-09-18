"""규칙 기반 판단기 기준군 검사 (docs/02 §9, docs/08 §5).

기준군은 **모델 자리**에 들어간다. 그래서 두 가지를 본다: 모델과 같은 요청을 받아 같은
형식으로 답하는가, 그 답이 같은 조합 규칙을 그대로 통과하는가. 답은 결정적이어야 한다.
"""

import json

import pytest
import yaml
from helpers import D0_STREAMS, HARNESS_CONFIG, RULE_JUDGE_CONFIG, read_jsonl
from test_harness import GRASP, answers, blocked_scene, candidate_for, harness, keys_of, obj, observation

from robo_jev.contracts import QUESTION_SET_V0
from robo_jev.harness.robot import candidate_id
from robo_jev.harness.rule_judge import RULE_JUDGE_VERSION, RuleJudge, rule_judge

CONFIG = yaml.safe_load(RULE_JUDGE_CONFIG.read_text(encoding="utf-8"))
HARNESS = yaml.safe_load(HARNESS_CONFIG.read_text(encoding="utf-8"))
CONFIDENCE = CONFIG["confidence"]
PROFILES = CONFIG["profiles"]

BOOLEANS = ("q_done", "q_instr", "q_observe", "q_retry", "q_stop")


def request_for(scene=None, commitment=None, hrn=None):
    hrn = hrn or harness()
    return hrn.build_request(scene if scene is not None else observation(), None, commitment)


def committed_request(key: str = GRASP, scene=None):
    hrn = harness()
    scene = scene if scene is not None else observation()
    first = hrn.build_request(scene, None, None)
    geometry = first["harness"]["candidates"][candidate_id(key)]
    commitment = {
        "action_ref": candidate_id(key),
        "key": key,
        "phase": geometry["phase"],
        "held_ticks": 2,
        "last_switch_tick": 0,
        "goal_version": 1,
    }
    return hrn.build_request(scene, None, commitment), commitment


# --------------------------------------------------------------------------
# 형식
# --------------------------------------------------------------------------


def test_the_rule_judge_answers_every_question_in_the_model_format():
    request = request_for()
    result = rule_judge(request)

    assert set(result) == set(QUESTION_SET_V0)
    candidates = {entry["id"] for entry in request["request"]["candidates"]["q_main"]}
    assert set(result["q_main"]) == candidates
    assert sum(result["q_main"].values()) == pytest.approx(1.0)
    for question_id in BOOLEANS:
        assert 0.0 <= result[question_id] <= 1.0
    assert set(result["q_gripper"]) == {"open", "closed"}
    assert set(result["q_speed"]) == {"0", "1", "2", "3"}
    assert set(result["q_force"]) == {"0", "1", "2"}
    assert set(result["q_path"]) == {entry["id"] for entry in request["request"]["candidates"]["q_path"]}
    for question_id in ("q_main", "q_gripper", "q_path", "q_speed", "q_force"):
        assert sum(result[question_id].values()) == pytest.approx(1.0)


def test_the_answers_are_deterministic():
    request = request_for()
    assert rule_judge(request) == rule_judge(request)


def test_it_answers_a_d0_tick_in_the_same_shape_as_the_model_output():
    """D0 fixture의 틱을 그대로 받아 그 `model_output`과 같은 모양으로 답한다."""
    record = read_jsonl(D0_STREAMS)[0]
    tick = record["ticks"][0]
    result = rule_judge(tick)

    reference = tick["model_output"]
    assert isinstance(result["q_main"], dict) and isinstance(reference["q_main"], dict)
    assert set(result["q_main"]) == {entry["id"] for entry in tick["request"]["candidates"]["q_main"]}
    assert isinstance(result["q_stop"], type(reference["q_stop"]))
    assert set(result["q_gripper"]) == set(reference["q_gripper"])
    assert json.dumps(result)  # 직렬화 가능한 순수 값이다


def test_the_version_is_fixed_in_config():
    assert RULE_JUDGE_VERSION == CONFIG["version"]
    assert RuleJudge(CONFIG).version == CONFIG["version"]


# --------------------------------------------------------------------------
# 적합성 필터와 고정 가중 비용
# --------------------------------------------------------------------------


def test_the_goal_target_carries_the_mass_and_the_rest_keeps_only_the_floor():
    """적합 후보가 질량을 갖고, 부적합 후보는 설정된 바닥만 나눠 갖는다."""
    request = request_for()
    result = rule_judge(request)
    keys = {entry["id"]: entry["key"] for entry in request["request"]["candidates"]["q_main"]}

    on_goal = sum(p for cid, p in result["q_main"].items() if keys[cid].startswith("grasp:o0:"))
    inadmissible = sum(
        p
        for cid, p in result["q_main"].items()
        if not keys[cid].startswith("grasp:o0:") and keys[cid] not in ("observe", "hold", "replan")
    )
    assert on_goal > 0.8
    assert inadmissible == pytest.approx(CONFIG["main"]["inadmissible_mass"], abs=1e-5)
    assert max(result["q_main"], key=result["q_main"].get) in [
        cid for cid, key in keys.items() if key.startswith("grasp:o0:")
    ]


def test_a_forbidden_object_is_never_preferred():
    scene = observation()
    scene["objects"][1]["attributes"] = ["forbidden"]
    result = rule_judge(request_for(scene))
    request = request_for(scene)
    forbidden = [
        entry["id"] for entry in request["request"]["candidates"]["q_main"] if ":o1:" in entry["key"]
    ]
    admissible = [
        entry["id"]
        for entry in request["request"]["candidates"]["q_main"]
        if entry["key"].startswith("grasp:o0:")
    ]
    assert max(result["q_main"][cid] for cid in forbidden) < min(
        result["q_main"][cid] for cid in admissible
    )


def test_a_blocking_object_may_be_pushed_out_of_the_way():
    request = request_for(blocked_scene())
    result = rule_judge(request)
    pushes = {
        entry["key"]: result["q_main"][entry["id"]]
        for entry in request["request"]["candidates"]["q_main"]
        if entry["key"].startswith("push:o5:")
    }
    others = [
        result["q_main"][entry["id"]]
        for entry in request["request"]["candidates"]["q_main"]
        if entry["key"].startswith("push:o0:")
    ]
    assert pushes and max(pushes.values()) > max(others or [0.0])


def test_cheaper_geometry_gets_more_mass():
    """같은 대상의 두 접근 중 거리가 가까운 쪽이 더 높은 확률을 받는다."""
    request = request_for()
    result = rule_judge(request)
    geometry = request["harness"]["candidates"]
    top = candidate_id("grasp:o0:top:zoneL:slow")
    side = candidate_id("grasp:o0:side:zoneL:slow")
    nearer, farther = (
        (top, side) if geometry[top]["distance_mm"] < geometry[side]["distance_mm"] else (side, top)
    )
    assert result["q_main"][nearer] > result["q_main"][farther]


def test_a_failed_approach_is_penalised_from_the_execution_history():
    hrn = harness()
    failed = {
        "adopted": {
            "main": candidate_id(GRASP), "phase": "approach", "path": "p0", "speed": 1,
            "force": 0, "gripper": "open", "stop": False,
        },
        "ack": {"seq": 1, "applied": False, "reason": "collision", "rejected": True},
    }
    plain = hrn.build_request(observation(), None, None)
    after = hrn.build_request(observation(tick=1, sim_time_ms=100), failed, None)

    before_p = rule_judge(plain)["q_main"][candidate_id(GRASP)]
    after_p = rule_judge(after)["q_main"][candidate_id(GRASP)]
    assert after_p < before_p


# --------------------------------------------------------------------------
# 게이팅·정지·부가 답
# --------------------------------------------------------------------------


def test_the_goal_is_reported_satisfied_by_zone_containment():
    scene = observation()
    result = rule_judge(request_for(scene))
    assert result["q_done"] == CONFIDENCE["low"]

    scene["objects"][0]["pos_mm"] = [30, 240, -80]  # zoneL 안
    assert rule_judge(request_for(scene))["q_done"] == CONFIDENCE["high"]


def test_an_incomplete_instruction_is_reported():
    scene = observation(instruction={"version": 1, "t_ms": 0, "text": "저것 좀 치워"})
    assert rule_judge(request_for(scene))["q_instr"] == CONFIDENCE["low"]
    assert rule_judge(request_for())["q_instr"] == CONFIDENCE["high"]


def test_an_occluded_target_asks_for_observation():
    hrn = harness()
    hrn.build_request(observation(), None, None)
    scene = observation(tick=1, sim_time_ms=100)
    scene["objects"][0].update(visible=False, visible_ratio=0.2)
    assert rule_judge(hrn.build_request(scene, None, None))["q_observe"] == CONFIDENCE["high"]
    assert rule_judge(request_for())["q_observe"] == CONFIDENCE["low"]


def test_contact_force_over_the_limit_asks_for_a_stop():
    scene = observation()
    scene["robot"]["contact_force_n"] = CONFIG["thresholds"]["stop_force_n"] + 5
    assert rule_judge(request_for(scene))["q_stop"] == CONFIDENCE["high"]
    assert rule_judge(request_for())["q_stop"] == CONFIDENCE["low"]


def test_retry_is_only_appropriate_after_a_first_failure():
    hrn = harness()
    failed = {
        "adopted": {"main": candidate_id(GRASP), "phase": "approach", "path": "p0", "speed": 1,
                    "force": 0, "gripper": "open", "stop": False},
        "ack": {"seq": 1, "applied": False, "reason": "collision"},
    }
    assert rule_judge(hrn.build_request(observation(), failed, None))["q_retry"] == CONFIDENCE["high"]
    assert rule_judge(request_for())["q_retry"] == CONFIDENCE["low"]


def test_aux_answers_follow_the_commitment_phase():
    request, commitment = committed_request()
    result = rule_judge(request)
    phase = commitment["phase"]
    assert max(result["q_speed"], key=result["q_speed"].get) == str(PROFILES["speed_by_phase"][phase])
    assert max(result["q_force"], key=result["q_force"].get) == str(PROFILES["force_by_phase"][phase])
    assert max(result["q_gripper"], key=result["q_gripper"].get) == PROFILES["gripper_by_phase"][phase]


def test_speed_is_lowered_next_to_a_fragile_object():
    scene = observation()
    scene["objects"][2]["pos_mm"] = [340, 40, -80]
    request, _ = committed_request(scene=scene)
    result = rule_judge(request)
    assert max(result["q_speed"], key=result["q_speed"].get) == str(PROFILES["fragile_speed_cap"])


def test_force_follows_the_function():
    push_key = "push:o0:+x:none:slow"
    request, _ = committed_request(push_key)
    result = rule_judge(request)
    assert max(result["q_force"], key=result["q_force"].get) == str(
        PROFILES["force_by_phase"][request["request"]["commitment"]["phase"]]
    )


def test_the_path_is_direct_unless_it_is_blocked():
    request, _ = committed_request()
    assert max(rule_judge(request)["q_path"], key=rule_judge(request)["q_path"].get) == "p0"

    blocked, _ = committed_request("grasp:o0:side:zoneL:slow", scene=blocked_scene())
    choice = max(rule_judge(blocked)["q_path"], key=rule_judge(blocked)["q_path"].get)
    kinds = {entry["id"]: entry["kind"] for entry in blocked["request"]["candidates"]["q_path"]}
    assert kinds[choice] == "via"


# --------------------------------------------------------------------------
# 같은 조합 규칙을 지난다
# --------------------------------------------------------------------------


def test_the_rule_answers_go_through_the_same_composition_rules():
    hrn = harness()
    commitment = None
    history = None
    for tick in range(4):
        scene = observation(tick=tick, sim_time_ms=tick * 100)
        request = hrn.build_request(scene, history, commitment)
        out = hrn.compose(request, rule_judge(request), commitment, tick * 100)
        assert out["command"] is not None
        assert out["adopted"]["main"] in {
            entry["id"] for entry in request["request"]["candidates"]["q_main"]
        }
        commitment = out["commitment"]
        history = {"adopted": out["adopted"], "ack": {"applied": True}, "gate": out["gate"]}
    assert commitment is not None and commitment["key"].startswith("grasp:o0:")


def test_the_rule_baseline_is_stable_under_the_hysteresis():
    """같은 장면이 이어지면 기준군도 결정을 바꾸지 않는다 (결정 안정성 지표의 바닥)."""
    hrn = harness()
    commitment = None
    switches = 0
    for tick in range(8):
        scene = observation(tick=tick, sim_time_ms=tick * 100)
        request = hrn.build_request(scene, None, commitment)
        out = hrn.compose(request, rule_judge(request), commitment, tick * 100)
        switches += int(out["switch"])
        commitment = out["commitment"]
    assert switches == 1  # 첫 틱의 채택뿐이다
