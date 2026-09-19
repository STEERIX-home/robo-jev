"""규칙 기반 판단기 기준군 검사 (docs/02 §9, docs/08 §5).

기준군은 **모델 자리**에 들어간다. 그래서 두 가지를 본다: 모델과 같은 요청을 받아 같은
형식으로 답하는가, 그 답이 같은 조합 규칙을 그대로 통과하는가. 답은 결정적이어야 한다.
"""

import copy
import json

import pytest
import yaml
from helpers import CONTROLLER_CONFIG, D0_STREAMS, HARNESS_CONFIG, RULE_JUDGE_CONFIG, SIM_CONFIG, read_jsonl
from test_harness import (
    BLOCKED,
    GRASP,
    PERIOD_MS,
    answers,
    blocked_scene,
    candidate_for,
    harness,
    keys_of,
    obj,
    observation,
)

from robo_jev.contracts import QUESTION_SET_V0
from robo_jev.harness.robot import candidate_id
from robo_jev.harness.rule_judge import RULE_JUDGE_VERSION, RuleJudge, load_rule_judge_config, rule_judge

CONFIG = yaml.safe_load(RULE_JUDGE_CONFIG.read_text(encoding="utf-8"))
HARNESS = yaml.safe_load(HARNESS_CONFIG.read_text(encoding="utf-8"))
CONTROLLER = yaml.safe_load(CONTROLLER_CONFIG.read_text(encoding="utf-8"))
SIM = yaml.safe_load(SIM_CONFIG.read_text(encoding="utf-8"))
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
    assert set(result["q_speed"]) == {str(index) for index in range(len(CONTROLLER["speed_levels_m_s"]))}
    assert set(result["q_force"]) == {str(index) for index in range(len(CONTROLLER["force_levels"]))}
    assert set(result["q_path"]) == {entry["id"] for entry in request["request"]["candidates"]["q_path"]}
    for question_id in ("q_main", "q_gripper", "q_path", "q_speed", "q_force"):
        assert sum(result[question_id].values()) == pytest.approx(1.0)


def test_speed_and_force_levels_come_from_the_controller_config():
    """수준의 개수는 실행기 설정이 단일 출처다 — 코드의 `range(4)`가 아니다."""
    controller = copy.deepcopy(CONTROLLER)
    controller["speed_levels_m_s"] = [0.0, 0.05, 0.1, 0.25, 0.5]
    controller["force_levels"]["crush"] = {"impedance_kp": 30.0, "contact_allowance_n": 80.0}
    judge = RuleJudge(CONFIG, controller_config=controller)
    result = judge(request_for())
    assert set(result["q_speed"]) == {"0", "1", "2", "3", "4"}
    assert set(result["q_force"]) == {"0", "1", "2", "3"}


def test_the_answers_are_deterministic():
    """새로 만든 기준군(설정 재적재)에 직렬화를 거친 같은 요청을 주면 같은 답이다."""
    request = request_for()
    fresh = RuleJudge(load_rule_judge_config())
    reloaded = json.loads(json.dumps(request, ensure_ascii=False))
    assert rule_judge(request) == fresh(reloaded)
    assert json.dumps(rule_judge(request), sort_keys=True) == json.dumps(fresh(reloaded), sort_keys=True)


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
    """같은 대상의 적합한 접근들 중 거리가 가까운 쪽이 더 높은 확률을 받는다.

    길을 막는 물체의 밀기는 네 방향이 모두 적합하다. 말단에 가까운 접촉점이 이긴다.
    """
    # 밀기는 영역 쪽 축만 만들어지므로(계약 v0.3) 영역을 둘 둔다: zoneL(+y 쪽)과 +x 쪽 영역. 말단(0, 0)에서 +x
    # 밀기의 접촉점(128, 0)이 +y 밀기의 접촉점(200, −72)보다 가깝다.
    scene = blocked_scene()
    scene["zones"].append({"id": "zoneX", "desc": "앞쪽 영역", "bounds_mm": [400, -80, 600, 80]})
    request = request_for(scene)
    result = rule_judge(request)
    geometry = request["harness"]["candidates"]
    pushes = {
        entry["id"]: geometry[entry["id"]]["distance_mm"]
        for entry in request["request"]["candidates"]["q_main"]
        if entry["key"].startswith("push:o5:")
    }
    assert len(pushes) >= 2
    nearer = min(pushes, key=pushes.get)
    farther = max(pushes, key=pushes.get)
    assert pushes[nearer] < pushes[farther]
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


@pytest.mark.parametrize(
    ("text", "complete"),
    [
        ("빨간 상자를 왼쪽 정리 영역으로 옮기고 파란 원통은 건드리지 마라", True),
        ("빨간 상자 대신 파란 원통를 왼쪽 정리 영역으로 먼저 옮겨라", True),
        ("빨간 상자를 옮겨라", False),  # 목적지가 없다
        ("왼쪽 정리 영역으로 옮겨라", False),  # 대상이 없다
        ("빨간 상자를 왼쪽 정리 영역으로 옮기고 그것은 건드리지 마라", False),  # 제약을 어휘로 풀 수 없다
    ],
)
def test_instruction_completeness_is_about_the_text_not_the_scene(text, complete):
    """`q_instr`은 지시 텍스트의 완결성이다: 어휘로 풀리는 대상·목적지·제약 (docs/08 §4).

    텍스트 근사 경로(구조화된 목표가 없는 틱)의 검사다. 아무 물체도 보이지 않으면 상태에 설명이
    없으므로 어휘를 **따로 준다** — 기본 기준군은 어휘 설정을 요구하지 않는다(3c-1, 아래 검사).
    """
    scene = observation(instruction={"version": 1, "t_ms": 0, "text": text})
    for entry in scene["objects"]:
        entry.update(visible=False, visible_ratio=0.0)  # 아무것도 보이지 않아도 답은 같다
    judge = RuleJudge(CONFIG, vocabulary_config=SIM)
    result = judge(request_for(scene))
    assert result["q_instr"] == (CONFIDENCE["high"] if complete else CONFIDENCE["low"])


# --------------------------------------------------------------------------
# 구조화된 목표 (3c-1): 텍스트 파싱 없이 판정하고, 어휘 설정은 필요 없다
# --------------------------------------------------------------------------


def structured_scene(*, target_ref="o0", target_desc="red 상자", zone_ref="zoneL", forbidden_refs=(), **over):
    scene = observation(**over)
    scene["goal"] = {
        "target_ref": target_ref,
        "target_desc": target_desc,
        "zone_ref": zone_ref,
        "forbidden_refs": list(forbidden_refs),
        "fragile_refs": ["o2"],
        "version": 1,
        "text": scene["instruction"]["text"],
    }
    return scene


def test_the_judge_config_no_longer_requires_a_vocabulary():
    assert "vocabulary_config" not in load_rule_judge_config()
    assert RuleJudge(load_rule_judge_config()).vocabulary_phrases == ()


@pytest.mark.parametrize(
    ("goal", "complete"),
    [
        ({}, True),
        ({"zone_ref": None}, False),  # 목적지가 없다
        ({"target_ref": None, "target_desc": None}, False),  # 대상이 없다
        ({"forbidden_refs": ["o0"]}, False),  # 대상이 금지 물체다 — 모순
        ({"zone_ref": "zoneX"}, False),  # 상태에 없는 영역이다
    ],
)
def test_structured_goal_fields_decide_instruction_completeness(goal, complete):
    """구조화된 목표가 있으면 `q_instr`은 그 필드로만 판정한다 — 어휘가 아니다."""
    scene = structured_scene(**goal)
    result = rule_judge(request_for(scene))
    assert result["q_instr"] == (CONFIDENCE["high"] if complete else CONFIDENCE["low"])


def test_a_complete_structured_goal_is_complete_whatever_is_visible():
    """가시성은 `q_instr`의 근거가 아니다 (docs/08 §4). 아무것도 안 보여도 지시는 완결됐다."""
    scene = structured_scene()
    for entry in scene["objects"]:
        entry.update(visible=False, visible_ratio=0.0)
    result = rule_judge(request_for(scene))
    assert result["q_instr"] == CONFIDENCE["high"]
    assert result["q_observe"] == CONFIDENCE["high"]  # 대상을 아직 못 봤으니 관측이다


def test_a_structured_target_that_is_not_yet_tracked_asks_for_observation_not_a_text_guess():
    """구조화된 대상이 아직 보이지 않으면 텍스트가 부르는 다른 물체로 바꿔 타지 않는다."""
    scene = structured_scene(target_ref="o1", target_desc="blue 상자")
    scene["instruction"]["text"] = "red 상자 대신 blue 상자를 왼쪽 정리 영역으로 먼저 옮겨라"
    scene["goal"]["text"] = scene["instruction"]["text"]
    scene["objects"][1].update(visible=False, visible_ratio=0.0)  # blue 상자를 본 적 없다
    request = request_for(scene)
    assert request["request"]["state"]["goal"]["target_ref"] is None
    result = rule_judge(request)
    assert result["q_instr"] == CONFIDENCE["high"]
    assert result["q_observe"] == CONFIDENCE["high"]
    keys = {entry["id"]: entry["key"] for entry in request["request"]["candidates"]["q_main"]}
    on_red = sum(p for cid, p in result["q_main"].items() if ":o0:" in keys[cid])
    assert on_red < 0.1  # red 상자는 대상이 아니다


def test_the_structured_goal_drives_the_main_decision():
    scene = structured_scene(target_ref="o1", target_desc="blue 상자")
    request = request_for(scene)
    result = rule_judge(request)
    keys = {entry["id"]: entry["key"] for entry in request["request"]["candidates"]["q_main"]}
    best = max(result["q_main"], key=result["q_main"].get)
    assert keys[best].startswith("grasp:o1:top:zoneL")


def test_the_text_fallback_still_resolves_the_d0_fixture():
    """구조화된 목표가 없는 틱(D0 fixture)은 상태의 물체 설명으로 텍스트를 푼다 — 어휘 설정 없이."""
    record = read_jsonl(D0_STREAMS)[0]
    tick = record["ticks"][0]
    assert "target_desc" not in tick["request"]["state"]["goal"]
    judge = RuleJudge(load_rule_judge_config())
    result = judge(tick)
    assert result["q_instr"] == CONFIDENCE["high"]
    keys = {entry["id"]: entry["key"] for entry in tick["request"]["candidates"]["q_main"]}
    best = max(result["q_main"], key=result["q_main"].get)
    assert ":o7:" in keys[best]  # "빨간 컵"


def test_an_unseen_target_asks_for_observation_not_for_a_replan():
    """docs/10 검토 1: 대상이 아직 관측되지 않은 것은 지시의 문제가 아니라 관측의 문제다.

    구조화된 목표에서는 `target_desc`가 대상을 나르므로 어휘 없이 판정된다. 텍스트 근사 경로는
    본 적 없는 물체의 이름을 상태에서 얻을 수 없으므로 어휘를 따로 줘야 같은 답이 나온다.
    """
    scene = observation()
    scene["objects"][0].update(visible=False, visible_ratio=0.0)  # 지시의 대상(red 상자)을 본 적 없다
    structured = copy.deepcopy(scene)
    structured["goal"] = {
        "target_ref": "o0", "target_desc": "red 상자", "zone_ref": "zoneL", "forbidden_refs": [],
        "fragile_refs": ["o2"], "version": 1, "text": scene["instruction"]["text"],
    }
    result = rule_judge(request_for(structured))
    assert result["q_instr"] == CONFIDENCE["high"]
    assert result["q_observe"] == CONFIDENCE["high"]

    result = RuleJudge(CONFIG, vocabulary_config=SIM)(request_for(scene))
    assert result["q_instr"] == CONFIDENCE["high"]
    assert result["q_observe"] == CONFIDENCE["high"]


def test_observation_follows_geometry_age_not_the_visible_ratio():
    """가시 비율만으로는 관측을 요구하지 않는다. 기하가 문턱보다 늙어야 한다."""
    hrn = harness()
    hrn.build_request(observation(), None, None)
    covered = observation(tick=1, sim_time_ms=100)
    covered["objects"][0].update(visible=False, visible_ratio=0.2)
    assert rule_judge(hrn.build_request(covered, None, None))["q_observe"] == CONFIDENCE["low"]

    stale_ms = CONFIG["thresholds"]["observe_geom_age_ms"] + PERIOD_MS
    stale = observation(tick=stale_ms // PERIOD_MS, sim_time_ms=stale_ms)
    stale["objects"][0].update(visible=False, visible_ratio=0.2)
    assert rule_judge(hrn.build_request(stale, None, None))["q_observe"] == CONFIDENCE["high"]
    assert rule_judge(request_for())["q_observe"] == CONFIDENCE["low"]


def test_the_arm_hiding_the_target_in_the_grasp_phase_is_not_a_lack_of_observation():
    """파지·놓기 국면과 파지 중에는 `max_geometry_age_ms` 안이면 관측을 요구하지 않는다 (docs/08 §4)."""
    hrn = harness()
    first = hrn.build_request(observation(), None, None)
    commitment = {
        "action_ref": candidate_id(GRASP), "key": GRASP, "phase": "grasp", "held_ticks": 4,
        "last_switch_tick": 0, "goal_version": 1,
    }
    stale_ms = CONFIG["thresholds"]["observe_geom_age_ms"] + PERIOD_MS
    assert stale_ms <= HARNESS["candidates"]["max_geometry_age_ms"]
    descending = observation(tick=stale_ms // PERIOD_MS, sim_time_ms=stale_ms)
    descending["robot"]["ee_pos_mm"] = [300, 0, -30]
    descending["objects"][0].update(visible=False, visible_ratio=0.0)
    request = hrn.build_request(descending, None, commitment)
    assert request["request"]["commitment"]["phase"] == "grasp"
    assert rule_judge(request)["q_observe"] == CONFIDENCE["low"]

    carrying = observation(tick=stale_ms // PERIOD_MS + 1, sim_time_ms=stale_ms + PERIOD_MS)
    carrying["robot"].update(ee_pos_mm=[300, 0, 40], holding="o0")
    carrying["objects"][0].update(visible=False, visible_ratio=0.0)
    request = hrn.build_request(carrying, None, {**commitment, "phase": "lift"})
    assert rule_judge(request)["q_observe"] == CONFIDENCE["low"]
    assert first["request"]["state"]["goal"]["target_ref"] == "o0"


def test_a_push_whose_hand_occludes_the_target_does_not_gate_to_observe():
    """밀기(접촉) 국면도 팔이 대상을 가리는 국면이다 — 하네스의 `CONTACT_PHASES`(grasp·place·push)가 단일 출처다."""
    from robo_jev.harness import rule_judge as rule_judge_module
    from robo_jev.harness.robot import CONTACT_PHASES

    assert "push" in CONTACT_PHASES and rule_judge_module._CONTACT_PHASES is CONTACT_PHASES

    hrn = harness()
    hrn.build_request(observation(), None, None)
    key = "push:o0:-x:none"  # o0(300, 0) → zoneL 쪽은 −x; 접촉점은 (372, 0)
    commitment = {
        "action_ref": candidate_id(key), "key": key, "phase": "push", "held_ticks": 4,
        "last_switch_tick": 0, "goal_version": 1,
    }
    stale_ms = CONFIG["thresholds"]["observe_geom_age_ms"] + PERIOD_MS
    pushing = observation(tick=stale_ms // PERIOD_MS, sim_time_ms=stale_ms)
    pushing["robot"]["ee_pos_mm"] = [400, 0, -80]  # 접촉점 40mm 안
    pushing["objects"][0].update(visible=False, visible_ratio=0.0)
    request = hrn.build_request(pushing, None, commitment)
    assert request["request"]["commitment"]["phase"] == "push"
    target = next(entry for entry in request["request"]["state"]["objects"] if entry["id"] == "o0")
    assert target["age_ms"] > CONFIG["thresholds"]["observe_geom_age_ms"]
    assert rule_judge(request)["q_observe"] == CONFIDENCE["low"]
    # 접근 국면(손이 아직 가리지 않는다)이면 같은 나이에 관측을 요구한다.
    request = hrn.build_request(pushing, None, {**commitment, "phase": "approach"})
    assert rule_judge(request)["q_observe"] == CONFIDENCE["high"]


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


def test_retry_stops_after_the_configured_number_of_same_way_failures():
    """`thresholds.retry_max_same_approach`가 실패 이력의 연속 횟수를 실제로 다스린다."""
    limit = CONFIG["thresholds"]["retry_max_same_approach"]
    hrn = harness()
    failed = {
        "adopted": {"main": candidate_id(GRASP), "phase": "approach", "path": "p0", "speed": 1,
                    "force": 0, "gripper": "open", "stop": False},
        "ack": {"seq": 1, "applied": False, "reason": "collision"},
    }
    verdicts = []
    for tick in range(limit + 2):
        request = hrn.build_request(observation(tick=tick, sim_time_ms=tick * PERIOD_MS), failed, None)
        assert f"fails={tick + 1}" in request["request"]["exec_history"]
        verdicts.append(rule_judge(request)["q_retry"])
    assert verdicts[:limit] == [CONFIDENCE["high"]] * limit
    assert verdicts[limit:] == [CONFIDENCE["low"]] * 2

    # 한 번 성공하면 횟수는 처음으로 돌아간다.
    succeeded = {**failed, "ack": {"seq": 9, "applied": True}}
    request = hrn.build_request(observation(tick=9, sim_time_ms=900), succeeded, None)
    assert "fails=0" in request["request"]["exec_history"]
    request = hrn.build_request(observation(tick=10, sim_time_ms=1000), failed, None)
    assert rule_judge(request)["q_retry"] == CONFIDENCE["high"]

    # 다른 방식의 실패는 따로 센다.
    other = copy.deepcopy(failed)
    other["adopted"]["main"] = candidate_id("push:o0:+x:none")
    request = hrn.build_request(observation(tick=11, sim_time_ms=1100), other, None)
    assert "fails=1" in request["request"]["exec_history"]


def test_aux_answers_follow_the_commitment_phase():
    request, commitment = committed_request()
    result = rule_judge(request)
    phase = commitment["phase"]
    assert max(result["q_speed"], key=result["q_speed"].get) == str(PROFILES["speed_by_phase"][phase])
    assert max(result["q_force"], key=result["q_force"].get) == str(PROFILES["force_by_phase"][phase])
    assert max(result["q_gripper"], key=result["q_gripper"].get) == PROFILES["gripper_by_phase"][phase]


def test_the_gripper_closes_only_at_the_grasp_point():
    """파지 국면이라도 말단이 파지점에 `grasp_ready_mm` 안으로 와야 닫는다 — 내려가는 중에 닫으면
    패드가 물체 윗면을 잡고 팔이 물체를 작업면에 누른다 (E0 폐루프에서 관측된 정지 반복)."""
    ready = CONFIG["thresholds"]["grasp_ready_mm"]
    hrn = harness()
    first = hrn.build_request(observation(), None, None)
    geometry = first["harness"]["candidates"][candidate_id(GRASP)]
    grasp_point = geometry["action_mm"]
    commitment = {
        "action_ref": candidate_id(GRASP), "key": GRASP, "phase": "grasp", "held_ticks": 4,
        "last_switch_tick": 0, "goal_version": 1,
    }

    descending = observation(tick=1, sim_time_ms=100)
    descending["robot"]["ee_pos_mm"] = [grasp_point[0], grasp_point[1], grasp_point[2] + ready + 20]
    request = hrn.build_request(descending, None, commitment)
    assert request["harness"]["candidates"][candidate_id(GRASP)]["phase"] == "grasp"
    result = rule_judge(request)
    assert max(result["q_gripper"], key=result["q_gripper"].get) == "open"

    arrived = observation(tick=2, sim_time_ms=200)
    arrived["robot"]["ee_pos_mm"] = [grasp_point[0], grasp_point[1], grasp_point[2] + ready - 3]
    request = hrn.build_request(arrived, None, commitment)
    result = rule_judge(request)
    assert max(result["q_gripper"], key=result["q_gripper"].get) == "closed"


def test_speed_is_lowered_next_to_a_fragile_object():
    scene = observation()
    scene["objects"][2]["pos_mm"] = [340, 40, -80]
    request, _ = committed_request(scene=scene)
    result = rule_judge(request)
    assert max(result["q_speed"], key=result["q_speed"].get) == str(PROFILES["fragile_speed_cap"])


def test_force_follows_the_function():
    push_key = "push:o0:-x:none"
    request, _ = committed_request(push_key)
    result = rule_judge(request)
    assert max(result["q_force"], key=result["q_force"].get) == str(
        PROFILES["force_by_phase"][request["request"]["commitment"]["phase"]]
    )


def test_the_path_is_direct_unless_it_is_blocked():
    request, _ = committed_request()
    assert max(rule_judge(request)["q_path"], key=rule_judge(request)["q_path"].get) == "p0"

    blocked, _ = committed_request(BLOCKED, scene=blocked_scene())
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


def test_instruction_wording_round_trips_to_the_resolved_target():
    """3a의 `desc`: 지시문이 부르는 이름 ↔ 상태의 물체 설명 ↔ 풀린 대상이 한 바퀴 맞아야 한다.

    장면 설명·지시 템플릿·앞단 설명이 같은 어휘를 쓰는지 실제 장면 생성기로 본다.
    """
    from robo_jev.sim.scene import build_plan

    labels = dict(SIM["objects"]["shape_labels"])
    # 텍스트 근사 경로다(구조화된 목표 없음). "건드리지 마라"의 주어가 보이지 않는 틱도 있으므로
    # 장면 어휘를 준다 — 실제 파이프라인은 구조화된 목표를 쓴다 (3c-1).
    judge = RuleJudge(CONFIG, vocabulary_config=SIM)
    for seed in range(1, 9):
        plan = build_plan(SIM, seed, "E1")
        text = plan.instructions[0].text
        described = {obj.describe(labels): obj.id for obj in plan.objects}
        named = [obj for obj in plan.objects if text.startswith(obj.describe(labels))]
        assert len(named) == 1, f"seed {seed}: 지시가 부르는 물체가 하나가 아니다: {text}"

        # 상한에 걸리지 않도록 지시가 부르는 물체와 속성 물체만 놓는다 — 보는 것은 어휘의 왕복이다.
        shown = [named[0]] + [o for o in plan.objects if o.attributes][:2]
        scene = observation(
            instruction={"version": 1, "t_ms": 0, "text": text},
            objects=[
                obj(o.id, (300 + 40 * index, -200 + 120 * index, -80), colour=o.colour,
                    desc=o.describe(labels), attributes=list(o.attributes), shape=o.shape,
                    obb_mm=list(o.obb_mm))
                for index, o in enumerate(shown)
            ],
            zones=[{"id": z.id, "desc": z.desc, "bounds_mm": list(z.bounds_mm)} for z in plan.zones],
        )
        request = request_for(scene)
        goal = request["request"]["state"]["goal"]
        assert goal["target_ref"] == named[0].id == described[named[0].describe(labels)]
        assert goal["target_zone"] in {z.id for z in plan.zones}
        result = judge(request)
        assert result["q_instr"] == CONFIDENCE["high"]
        best = max(result["q_main"], key=result["q_main"].get)
        key = candidate_for(request, next(k for k in keys_of(request) if candidate_id(k) == best))["key"]
        assert key.split(":")[1] == named[0].id, f"seed {seed}: 기준군이 지시의 대상을 고르지 않았다: {key}"


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
