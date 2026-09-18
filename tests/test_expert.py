"""스크립트 전문가 v0 검사 (docs/08 §7·§9, docs/04 §3, Task 3c-1).

전문가는 모델 자리에 서서 10개 답을 내고, 그 답이 라벨의 원천이 된다. 그래서 세 가지를 본다.

* **형식** — 규칙 기준군·모델과 같은 출력 형식(choice·ordinal은 후보 분포, boolean은 확률).
* **정보 경계** — 답은 하네스 요청(모델 입력)과 commitment의 함수다. 가려진 물체의 참값을 바꿔도,
  하네스 블록을 떼어도 답은 같다.
* **행동** — 구조화된 목표를 실현하는 결합 후보를 고르고 국면을 따라 commitment를 지키며,
  게이팅·정지·부가 답을 규칙대로 낸다. 마지막으로 실제 환경에서 E0·E1 에피소드를 완료한다.
"""

import copy
import math

import pytest
import yaml
from helpers import HARNESS_CONFIG, SIM_CONFIG
from test_harness import GRASP, PERIOD_MS, harness, obj, observation

from robo_jev.contracts import QUESTION_SET_V0
from robo_jev.harness.robot import candidate_id
from robo_jev.sim.expert import EXPERT_VERSION, Expert, load_expert_config

CONFIG = load_expert_config()
HARNESS = yaml.safe_load(HARNESS_CONFIG.read_text(encoding="utf-8"))
SIM = yaml.safe_load(SIM_CONFIG.read_text(encoding="utf-8"))
CONFIDENCE = CONFIG["confidence"]
THRESHOLDS = CONFIG["thresholds"]
PROFILES = CONFIG["profiles"]
BOOLEANS = ("q_done", "q_instr", "q_observe", "q_retry", "q_stop")


def structured(scene: dict, *, target_ref="o0", target_desc="red 상자", zone_ref="zoneL", forbidden_refs=(), fragile_refs=("o2",)) -> dict:
    """관측에 구조화된 목표를 붙인다 (`Environment._structured_goal`의 형태)."""
    scene["goal"] = {
        "target_ref": target_ref,
        "target_desc": target_desc,
        "zone_ref": zone_ref,
        "forbidden_refs": list(forbidden_refs),
        "fragile_refs": list(fragile_refs),
        "version": int(scene["instruction"]["version"]),
        "text": scene["instruction"]["text"],
    }
    return scene


def scene(**over) -> dict:
    return structured(observation(**over))


def expert() -> Expert:
    return Expert(CONFIG)


def request_for(obs=None, commitment=None, hrn=None, history=None) -> dict:
    hrn = hrn or harness()
    return hrn.build_request(obs if obs is not None else scene(), history, commitment)


def top(distribution: dict) -> str:
    return max(distribution, key=lambda key: (distribution[key], key))


def key_of(request: dict, candidate: str) -> str:
    return next(entry["key"] for entry in request["request"]["candidates"]["q_main"] if entry["id"] == candidate)


def committed_request(key: str = GRASP, obs=None, hrn=None, **over):
    """`key`에 commitment가 걸린 요청과 그 commitment (하네스 장부 포함)."""
    hrn = hrn or harness()
    obs = obs if obs is not None else scene()
    first = hrn.build_request(obs, None, None)
    geometry = first["harness"]["candidates"][candidate_id(key)]
    commitment = {
        "action_ref": candidate_id(key), "key": key, "phase": geometry["phase"], "held_ticks": 2,
        "last_switch_tick": 0, "goal_version": 1, "challenger": None, "challenger_ticks": 0, "stop_ticks": 0,
        "start_pose_mm": None, "start_clearance_mm": None,
    }
    commitment.update(over)
    return hrn.build_request(obs, None, commitment), commitment


# --------------------------------------------------------------------------
# 형식과 버전
# --------------------------------------------------------------------------


def test_the_expert_answers_every_question_in_the_model_format():
    request = request_for()
    out = expert().act(request, None, scene())
    assert set(out) == set(QUESTION_SET_V0) | {"phase", "expert_meta"}
    candidates = {entry["id"] for entry in request["request"]["candidates"]["q_main"]}
    assert set(out["q_main"]) == candidates
    for question_id in BOOLEANS:
        assert 0.0 <= out[question_id] <= 1.0
    assert set(out["q_gripper"]) == {"open", "closed"}
    assert set(out["q_path"]) == {entry["id"] for entry in request["request"]["candidates"]["q_path"]}
    assert set(out["q_speed"]) == {"0", "1", "2", "3"} and set(out["q_force"]) == {"0", "1", "2"}
    for question_id in ("q_main", "q_gripper", "q_path", "q_speed", "q_force"):
        assert sum(out[question_id].values()) == pytest.approx(1.0, abs=1e-6)
    assert out["phase"] == "none"
    assert out["expert_meta"]["version"] == EXPERT_VERSION


def test_the_expert_is_its_own_class_with_its_own_version():
    from robo_jev.harness.rule_judge import RULE_JUDGE_VERSION, RuleJudge

    assert not issubclass(Expert, RuleJudge)
    assert EXPERT_VERSION == CONFIG["version"] != RULE_JUDGE_VERSION
    assert expert().version == EXPERT_VERSION


# --------------------------------------------------------------------------
# 정보 경계 — 답은 모델 입력과 commitment의 함수다
# --------------------------------------------------------------------------


def test_hidden_world_perturbation_does_not_change_the_answers():
    """가려진 물체의 참값(자세·속성)을 바꿔도 요청이 같으므로 답·국면·근거가 모두 같다."""
    plain, tampered = harness(), harness()
    first = scene()
    plain.build_request(first, None, None)
    tampered.build_request(first, None, None)

    hidden = scene(tick=1, sim_time_ms=PERIOD_MS)
    hidden["objects"][1].update(visible=False, visible_ratio=0.0)
    changed = copy.deepcopy(hidden)
    changed["objects"][1].update(pos_mm=[650, 220, -80], attributes=["forbidden"], obb_mm=[90, 90, 90])
    changed["events"] = [{"kind": "disturbance_applied", "object": "o1", "sim_ms": PERIOD_MS}]

    expected = expert().act(plain.build_request(hidden, None, None), None, hidden)
    actual = expert().act(tampered.build_request(changed, None, None), None, changed)
    assert actual == expected


def test_the_expert_reads_only_the_model_input_not_the_harness_block():
    request = request_for()
    stripped = {key: copy.deepcopy(value) for key, value in request.items() if key != "harness"}
    assert expert().act(request, None, scene()) == expert().act(stripped, None, scene())


def test_the_answers_are_deterministic_and_serialisable():
    import json

    request = request_for()
    once = expert().act(request, None, scene())
    again = Expert(load_expert_config()).act(json.loads(json.dumps(request, ensure_ascii=False)), None, scene())
    assert json.dumps(once, sort_keys=True) == json.dumps(again, sort_keys=True)


# --------------------------------------------------------------------------
# 주 결정 — 구조화된 목표를 실현하는 결합 후보
# --------------------------------------------------------------------------


def test_the_goal_grasp_to_the_goal_zone_is_chosen():
    request = request_for()
    out = expert().act(request, None, scene())
    chosen = key_of(request, top(out["q_main"]))
    assert chosen.startswith("grasp:o0:top:zoneL:")
    assert chosen.endswith(":" + CONFIG["goal"]["profile_preference"][0])
    assert out["q_main"][top(out["q_main"])] == pytest.approx(CONFIDENCE["choice_mass"], abs=1e-4)
    assert out["expert_meta"]["main"]["reason"] == "goal_grasp"


def test_a_different_structured_target_changes_the_choice_without_text_parsing():
    obs = scene()
    obs["goal"].update(target_ref="o1", target_desc="blue 상자")
    request = request_for(obs)
    out = expert().act(request, None, obs)
    assert key_of(request, top(out["q_main"])).startswith("grasp:o1:top:zoneL:")


def test_a_forbidden_target_is_never_chosen():
    obs = scene()
    obs["objects"][0]["attributes"] = ["forbidden"]
    obs["goal"]["forbidden_refs"] = ["o0"]
    request = request_for(obs)
    out = expert().act(request, None, obs)
    assert not key_of(request, top(out["q_main"])).startswith("grasp:o0:")
    assert out["q_instr"] == CONFIDENCE["low"]  # 모순된 지시


def test_the_commitment_is_kept_through_its_phases():
    """같은 목표 아래에서는 국면이 바뀌어도 commitment를 바꾸지 않는다 (approach → grasp → lift …)."""
    hrn = harness()
    first = request_for(hrn=hrn)
    chosen = candidate_id(GRASP)
    assert top(expert().act(first, None, scene())["q_main"]) == chosen

    for tick, (ee, holding) in enumerate([([300, 0, 0], None), ([300, 0, 40], "o0"), ([300, 0, 150], "o0"), ([30, 240, 150], "o0")], start=1):
        obs = scene(tick=tick, sim_time_ms=tick * PERIOD_MS)
        obs["robot"].update(ee_pos_mm=ee, holding=holding)
        request, commitment = committed_request(GRASP, obs=obs, hrn=hrn)
        out = expert().act(request, commitment, obs)
        assert top(out["q_main"]) == chosen, (tick, out["expert_meta"]["main"])
        assert out["phase"] == request["request"]["commitment"]["phase"]
        assert out["expert_meta"]["main"]["reason"] == "keep_commitment"


def test_a_commitment_on_the_wrong_goal_is_not_kept_after_the_instruction_changes():
    obs = scene(instruction={"version": 2, "t_ms": 500, "text": "red 상자 대신 blue 상자를 왼쪽 정리 영역으로 먼저 옮겨라"})
    obs["goal"].update(target_ref="o1", target_desc="blue 상자", version=2)
    request, commitment = committed_request(GRASP, obs=obs)
    out = expert().act(request, commitment, obs)
    assert key_of(request, top(out["q_main"])).startswith("grasp:o1:top:zoneL:")


def test_place_is_chosen_while_holding_without_a_commitment():
    obs = scene()
    obs["robot"].update(ee_pos_mm=[300, 0, 150], holding="o0")
    request = request_for(obs)
    out = expert().act(request, None, obs)
    assert key_of(request, top(out["q_main"])).startswith("place:o0:release:zoneL:")


def two_object_scene(target_pos) -> dict:
    """대상과 취약 물체 둘뿐인 장면 — 후보가 상한(29) 안에 들어 어느 것도 잘리지 않는다."""
    return scene(objects=[obj("o0", target_pos), obj("o2", (120, -200, -80), colour="green", attributes=["fragile"])])


def without_goal_grasp(request: dict) -> dict:
    """요청에서 목표 대상→목표 영역의 파지 후보를 뺀 사본 (상한에 걸려 빠진 틱을 흉내 낸다)."""
    trimmed = copy.deepcopy(request)
    trimmed.pop("harness", None)
    trimmed["request"]["candidates"]["q_main"] = [
        entry for entry in trimmed["request"]["candidates"]["q_main"]
        if not entry["key"].startswith("grasp:o0:top:zoneL:")
    ]
    return trimmed


def test_push_toward_the_zone_when_the_goal_grasp_is_unavailable():
    """파지가 목록에 없으면(실행 불가·상한) 대상을 목표 영역 쪽으로 미는 후보를 고른다."""
    obs = two_object_scene((250, 240, -80))  # zoneL(x∈[-120,180], y∈[150,330])까지 −x로 70mm
    request = without_goal_grasp(request_for(obs))
    assert any(entry["key"].startswith("push:o0:-x:") for entry in request["request"]["candidates"]["q_main"])
    out = expert().act(request, None, obs)
    chosen = key_of(request, top(out["q_main"]))
    assert chosen.startswith("push:o0:-x:none:")
    assert out["expert_meta"]["main"]["reason"] == "push_toward_zone"


def test_hold_when_no_candidate_can_realise_the_goal():
    """목표를 실현할 후보가 하나도 없으면 hold다 — 다른 물체를 집지 않는다. 근거는 낮은 신뢰도로 남는다."""
    obs = two_object_scene((300, -100, -80))
    request = without_goal_grasp(request_for(obs))
    request["request"]["candidates"]["q_main"] = [
        entry for entry in request["request"]["candidates"]["q_main"] if not entry["key"].startswith("push:o0:")
    ]
    out = expert().act(request, None, obs)
    assert key_of(request, top(out["q_main"])) == "hold"
    assert out["expert_meta"]["main"]["reason"] == "goal_candidate_missing"
    labels = {label["question_id"]: label for label in expert().labels(out, request)}
    assert labels["q_main"]["label_confidence"] == "low"


def test_push_is_not_chosen_when_it_would_not_bring_the_target_closer():
    obs = two_object_scene((100, 0, -80))  # +y만이 영역에 가까워지는 방향이다
    request = without_goal_grasp(request_for(obs))
    request["request"]["candidates"]["q_main"] = [
        entry for entry in request["request"]["candidates"]["q_main"] if not entry["key"].startswith("push:o0:+y")
    ]
    out = expert().act(request, None, obs)
    assert key_of(request, top(out["q_main"])) == "hold"


def blocked_descent_scene(**neighbour_over) -> dict:
    """대상 o0 곁(60mm)에 이웃 o1이 있어 파지 하강 구간이 외접 구 + 여유에 걸린다. 말단은 파지 국면."""
    obs = scene(objects=[obj("o0", (300, 0, -80)), obj("o1", (300, 60, -80), colour="blue", **neighbour_over)])
    obs["robot"]["ee_pos_mm"] = [300, 0, 0]
    return obs


def test_a_plain_neighbour_blocking_the_grasp_with_no_detour_is_pushed_away():
    """파지가 막혔고 경유점도 없으면(하강 구간의 이웃) 그 이웃을 대상에서 멀어지는 방향으로 민다 —
    "밀기는 파지가 불가능할 때만". 물러났다 다시 다가가는 반복(E1 seed 37)을 끊는다."""
    obs = blocked_descent_scene()
    request, commitment = committed_request(GRASP, obs=obs)
    assert request["request"]["commitment"]["phase"] == "grasp"
    assert "path blocked" in next(e["derived"] for e in request["request"]["candidates"]["q_main"] if e["key"] == GRASP)
    assert not [e for e in request["request"]["candidates"]["q_path"] if e["kind"] == "via"]
    out = expert().act(request, commitment, obs)
    chosen = key_of(request, top(out["q_main"]))
    # +y가 가장 멀리 밀지만 그 접촉점은 대상의 구 안이라 어떤 경로로도 닿을 수 없다 → 옆에서 미는 ±x (키 순).
    assert chosen.startswith("push:o1:+x:none:"), out["expert_meta"]["main"]
    assert out["expert_meta"]["main"]["reason"] == "push_blocker"
    assert candidate_id(GRASP) in out["expert_meta"]["main"]["admissible"]


def test_the_blocker_push_is_kept_while_it_still_blocks():
    obs = blocked_descent_scene()
    push = "push:o1:+x:none:slow"
    request, commitment = committed_request(push, obs=obs)
    out = expert().act(request, commitment, obs)
    assert top(out["q_main"]) == candidate_id(push)
    assert out["expert_meta"]["main"]["reason"] == "keep_commitment"

    cleared = blocked_descent_scene()
    cleared["objects"][1]["pos_mm"] = [300, 160, -80]  # 밀려서 더는 막지 않는다
    request, commitment = committed_request(push, obs=cleared)
    out = expert().act(request, commitment, cleared)
    assert top(out["q_main"]) == candidate_id(GRASP)


@pytest.mark.parametrize("attributes", [["fragile"], ["forbidden"]])
def test_a_protected_blocker_is_never_pushed(attributes):
    obs = blocked_descent_scene(attributes=attributes)
    obs["goal"]["forbidden_refs" if attributes == ["forbidden"] else "fragile_refs"] = ["o1"]
    request, commitment = committed_request(GRASP, obs=obs)
    out = expert().act(request, commitment, obs)
    assert top(out["q_main"]) == candidate_id(GRASP)
    assert out["expert_meta"]["main"]["reason"] == "keep_commitment"


def test_push_directions_are_restricted_by_config_to_what_the_open_gripper_can_do():
    """v0 실행기는 밀 때 그리퍼가 열려 있고 손가락이 y축으로 벌어진다 — ±y 접근은 손가락이 물체를
    친다(측정: 접촉력 45~78N). 설정의 방향만 고른다."""
    assert set(CONFIG["goal"]["push_directions"]) == {"+x", "-x"}
    obs = two_object_scene((100, 120, -80))  # +y가 영역 쪽이지만 허용된 방향이 아니다
    request = without_goal_grasp(request_for(obs))
    out = expert().act(request, None, obs)
    assert key_of(request, top(out["q_main"])) == "hold"

    sideways = two_object_scene((250, 240, -80))  # zoneL(x ≤ 180)까지 −x로 70mm
    request = without_goal_grasp(request_for(sideways))
    out = expert().act(request, None, sideways)
    assert key_of(request, top(out["q_main"])).startswith("push:o0:-x:none:")


def test_a_held_non_target_is_put_down_in_the_goal_zone_first():
    """지시가 바뀌었는데 손에 옛 대상이 있으면 그것을 먼저 놓는다 — 새 대상은 손이 비어야 집는다."""
    obs = scene(instruction={"version": 2, "t_ms": 500, "text": "red 상자 대신 blue 상자를 왼쪽 정리 영역으로 먼저 옮겨라"})
    obs["goal"].update(target_ref="o1", target_desc="blue 상자", version=2)
    obs["robot"].update(ee_pos_mm=[300, 0, 150], holding="o0")
    request = request_for(obs)
    assert not any(e["key"].startswith("grasp:o1:") for e in request["request"]["candidates"]["q_main"])
    out = expert().act(request, None, obs)
    assert key_of(request, top(out["q_main"])).startswith("place:o0:release:zoneL:")
    assert out["expert_meta"]["main"]["reason"] == "release_held_object"

    request, commitment = committed_request("place:o0:release:zoneL:slow", obs=obs)
    out = expert().act(request, commitment, obs)
    assert out["expert_meta"]["main"]["reason"] == "keep_commitment"


def test_a_way_the_harness_just_retry_blocked_is_not_proposed_again():
    """직전 틱에 같은 방식이 한계 횟수를 넘겨 실패했으면(`q_retry` 거짓) 하네스가 그 방식을 막는다 —
    그때 같은 답을 내면 하네스는 남은 후보 중 임의의 것을 채택한다. 전문가가 먼저 비켜 준다."""
    hrn = harness()
    failed = {
        "adopted": {"main": candidate_id(GRASP), "phase": "approach", "path": "p0", "speed": 1, "force": 0, "gripper": "open", "stop": False},
        "ack": {"seq": 1, "applied": False, "reason": "transition_collision"},
    }
    limit = THRESHOLDS["retry_max_same_approach"]
    for tick in range(limit + 1):
        request = hrn.build_request(scene(tick=tick, sim_time_ms=tick * PERIOD_MS), failed, None)
    assert f"fails={limit + 1}" in request["request"]["exec_history"]
    out = expert().act(request, None, scene())
    assert out["q_retry"] == CONFIDENCE["low"]
    chosen = key_of(request, top(out["q_main"]))
    assert not chosen.startswith("grasp:o0:top:")
    assert out["expert_meta"]["main"]["blocked_ways"] == ["grasp:o0:top"]
    assert not any(key.startswith("grasp:o0:top:") for key in
                   (key_of(request, c) for c in out["expert_meta"]["main"]["admissible"]))


# --------------------------------------------------------------------------
# 게이팅
# --------------------------------------------------------------------------


def test_done_when_the_target_rests_inside_the_zone():
    obs = scene()
    assert expert().act(request_for(obs), None, obs)["q_done"] == CONFIDENCE["low"]
    obs["objects"][0]["pos_mm"] = [30, 240, -80]
    request = request_for(obs)
    out = expert().act(request, None, obs)
    assert out["q_done"] == CONFIDENCE["high"]
    assert key_of(request, top(out["q_main"])) == "hold"

    held = scene()
    held["objects"][0]["pos_mm"] = [30, 240, 40]
    held["robot"].update(ee_pos_mm=[30, 240, 100], holding="o0")
    assert expert().act(request_for(held), None, held)["q_done"] == CONFIDENCE["low"]


@pytest.mark.parametrize(
    ("goal", "complete"),
    [
        ({}, True),
        ({"zone_ref": None}, False),
        ({"target_ref": None, "target_desc": None}, False),
        ({"zone_ref": "zoneX"}, False),
    ],
)
def test_instruction_is_insufficient_only_for_unparseable_or_contradictory_goals(goal, complete):
    obs = scene()
    obs["goal"].update(goal)
    out = expert().act(request_for(obs), None, obs)
    assert out["q_instr"] == (CONFIDENCE["high"] if complete else CONFIDENCE["low"])


def test_an_unseen_target_asks_for_observation():
    obs = scene()
    obs["objects"][0].update(visible=False, visible_ratio=0.0)
    request = request_for(obs)
    out = expert().act(request, None, obs)
    assert out["q_instr"] == CONFIDENCE["high"]
    assert out["q_observe"] == CONFIDENCE["high"]
    assert key_of(request, top(out["q_main"])) == "observe"
    assert out["expert_meta"]["main"]["reason"] == "observe_target"


def test_observation_follows_geometry_age_except_when_the_arm_hides_the_target():
    hrn = harness()
    hrn.build_request(scene(), None, None)
    stale_ms = THRESHOLDS["observe_geom_age_ms"] + PERIOD_MS
    assert stale_ms <= HARNESS["candidates"]["max_geometry_age_ms"]
    stale = scene(tick=stale_ms // PERIOD_MS, sim_time_ms=stale_ms)
    stale["objects"][0].update(visible=False, visible_ratio=0.0)
    assert expert().act(hrn.build_request(stale, None, None), None, stale)["q_observe"] == CONFIDENCE["high"]

    descending = copy.deepcopy(stale)
    descending["robot"]["ee_pos_mm"] = [300, 0, -30]
    request, commitment = committed_request(GRASP, obs=descending, hrn=hrn)
    assert request["request"]["commitment"]["phase"] == "grasp"
    assert expert().act(request, commitment, descending)["q_observe"] == CONFIDENCE["low"]


def test_retry_is_bounded_by_the_configured_same_way_failure_count():
    hrn = harness()
    failed = {
        "adopted": {"main": candidate_id(GRASP), "phase": "approach", "path": "p0", "speed": 1, "force": 0, "gripper": "open", "stop": False},
        "ack": {"seq": 1, "applied": False, "reason": "collision"},
    }
    limit = THRESHOLDS["retry_max_same_approach"]
    verdicts = []
    for tick in range(limit + 2):
        request = hrn.build_request(scene(tick=tick, sim_time_ms=tick * PERIOD_MS), failed, None)
        verdicts.append(expert().act(request, None, scene())["q_retry"])
    assert verdicts[:limit] == [CONFIDENCE["high"]] * limit
    assert verdicts[limit:] == [CONFIDENCE["low"]] * 2
    assert expert().act(request_for(), None, scene())["q_retry"] == CONFIDENCE["low"]


def test_stop_on_force_reflex_forbidden_proximity_and_protected_contact():
    quiet = scene()
    assert expert().act(request_for(quiet), None, quiet)["q_stop"] == CONFIDENCE["low"]

    forceful = scene()
    forceful["robot"]["contact_force_n"] = THRESHOLDS["stop_force_n"] + 1
    assert expert().act(request_for(forceful), None, forceful)["q_stop"] == CONFIDENCE["high"]

    # 파지 중의 정상 하중(쥐는 힘)은 충돌이 아니다: 들고 있는 동안은 반사 한계 위의 문턱이다.
    squeezing = scene()
    squeezing["robot"].update(holding="o0", ee_pos_mm=[300, 0, 40], contact_force_n=THRESHOLDS["stop_force_n"] + 1)
    assert THRESHOLDS["stop_force_holding_n"] > THRESHOLDS["stop_force_n"]
    assert expert().act(request_for(squeezing), None, squeezing)["q_stop"] == CONFIDENCE["low"]
    squeezing["robot"]["contact_force_n"] = THRESHOLDS["stop_force_holding_n"] + 1
    assert expert().act(request_for(squeezing), None, squeezing)["q_stop"] == CONFIDENCE["high"]

    reflex = scene(events=[{"kind": "reflex_force", "sim_ms": 0}])
    assert expert().act(request_for(reflex), None, reflex)["q_stop"] == CONFIDENCE["high"]

    near = scene()
    near["objects"][1].update(attributes=["forbidden"], pos_mm=[0, 30, 200])
    near["goal"]["forbidden_refs"] = ["o1"]
    assert expert().act(request_for(near), None, near)["q_stop"] == CONFIDENCE["high"]

    touched = scene(events=[{"kind": "contact_onset", "object": "o2", "sim_ms": 0}])  # 취약 물체
    assert expert().act(request_for(touched), None, touched)["q_stop"] == CONFIDENCE["high"]

    brushed = scene(events=[{"kind": "contact_onset", "object": "o1", "sim_ms": 0}])  # 평범한 이웃
    assert expert().act(request_for(brushed), None, brushed)["q_stop"] == CONFIDENCE["low"]


# --------------------------------------------------------------------------
# 부가 답 — commitment의 국면 기준
# --------------------------------------------------------------------------


def test_aux_answers_follow_the_commitment_phase():
    request, commitment = committed_request()
    out = expert().act(request, commitment, scene())
    phase = commitment["phase"]
    assert out["phase"] == phase == "approach"
    assert top(out["q_speed"]) == str(PROFILES["speed_by_phase"][phase])
    assert top(out["q_force"]) == str(PROFILES["force_by_phase"][phase])
    assert top(out["q_gripper"]) == PROFILES["gripper_by_phase"][phase]


def test_the_gripper_closes_only_at_the_grasp_point_and_stays_closed_until_place():
    ready = THRESHOLDS["grasp_ready_mm"]
    hrn = harness()
    first = hrn.build_request(scene(), None, None)
    grasp_point = first["harness"]["candidates"][candidate_id(GRASP)]["action_mm"]

    descending = scene(tick=1, sim_time_ms=PERIOD_MS)
    descending["robot"]["ee_pos_mm"] = [grasp_point[0], grasp_point[1], grasp_point[2] + ready + 20]
    request, commitment = committed_request(GRASP, obs=descending, hrn=hrn)
    assert request["request"]["commitment"]["phase"] == "grasp"
    assert top(expert().act(request, commitment, descending)["q_gripper"]) == "open"

    arrived = scene(tick=2, sim_time_ms=2 * PERIOD_MS)
    arrived["robot"]["ee_pos_mm"] = [grasp_point[0], grasp_point[1], grasp_point[2] + ready - 3]
    request, commitment = committed_request(GRASP, obs=arrived, hrn=hrn)
    assert top(expert().act(request, commitment, arrived)["q_gripper"]) == "closed"

    for ee, phase in (([300, 0, 40], "lift"), ([200, 120, 150], "transport")):
        carrying = scene(tick=3, sim_time_ms=3 * PERIOD_MS)
        carrying["robot"].update(ee_pos_mm=ee, holding="o0")
        request, commitment = committed_request(GRASP, obs=carrying, hrn=hrn)
        assert request["request"]["commitment"]["phase"] == phase
        assert top(expert().act(request, commitment, carrying)["q_gripper"]) == "closed"

    placing = scene(tick=4, sim_time_ms=4 * PERIOD_MS)
    placing["robot"].update(ee_pos_mm=[30, 240, -40], holding="o0")
    request, commitment = committed_request(GRASP, obs=placing, hrn=hrn)
    assert request["request"]["commitment"]["phase"] == "place"
    assert top(expert().act(request, commitment, placing)["q_gripper"]) == "open"


def test_the_gripper_is_open_when_idle_and_closed_when_holding_without_a_commitment():
    idle = scene()
    assert top(expert().act(request_for(idle), None, idle)["q_gripper"]) == "open"
    holding = scene()
    holding["robot"].update(ee_pos_mm=[300, 0, 150], holding="o0", gripper_mm=30)
    assert top(expert().act(request_for(holding), None, holding)["q_gripper"]) == "closed"


def test_the_path_is_direct_unless_blocked_then_via():
    from test_harness import BLOCKED, blocked_scene

    request, commitment = committed_request()
    assert top(expert().act(request, commitment, scene())["q_path"]) == "p0"

    blocked = structured(blocked_scene())
    request, commitment = committed_request(BLOCKED, obs=blocked)
    kinds = {entry["id"]: entry["kind"] for entry in request["request"]["candidates"]["q_path"]}
    out = expert().act(request, commitment, blocked)
    assert kinds[top(out["q_path"])] == "via"


def test_blocked_without_a_detour_holds_unless_the_hand_itself_is_inside_an_obstacle():
    """물러나기(`retreat`)는 명령마다 120mm를 올린다 — 매 틱 답하면 팔이 한계까지 올라간다(E1 seed 37).
    경유점이 없으면 기다리고(`hold`), 말단 자체가 장애물의 넓힌 구 안일 때만 물러난다."""
    obs = blocked_descent_scene(attributes=["fragile"])  # 밀 수도 없다
    obs["goal"]["fragile_refs"] = ["o1"]
    request, commitment = committed_request(GRASP, obs=obs)
    kinds = {entry["id"]: entry["kind"] for entry in request["request"]["candidates"]["q_path"]}
    out = expert().act(request, commitment, obs)
    assert kinds[top(out["q_path"])] == "hold"
    assert out["expert_meta"]["aux"]["path"]["reason"] == "blocked_no_detour"

    stuck = blocked_descent_scene(attributes=["fragile"])
    stuck["goal"]["fragile_refs"] = ["o1"]
    stuck["robot"]["ee_pos_mm"] = [300, 40, -40]  # 손이 이웃의 외접 구 + 여유 안에 있다
    request, commitment = committed_request(GRASP, obs=stuck)
    kinds = {entry["id"]: entry["kind"] for entry in request["request"]["candidates"]["q_path"]}
    out = expert().act(request, commitment, stuck)
    assert kinds[top(out["q_path"])] == "retreat"
    assert out["expert_meta"]["aux"]["path"]["reason"] == "hand_inside_obstacle"


def test_a_clear_approach_outranks_a_larger_gain_for_a_blocker_push():
    """접촉점이 닿을 수 있는 방향들 가운데서는 접근이 비어 있는 쪽이 먼저다(이득은 그다음). 후보 설명의
    `path`만 바꿔 순위 규칙을 본다: −x의 접근만 비어 있으면 키 순으로 앞서는 +x 대신 −x다."""
    obs = blocked_descent_scene()
    request, commitment = committed_request(GRASP, obs=obs)
    request = copy.deepcopy(request)
    request.pop("harness")
    for entry in request["request"]["candidates"]["q_main"]:
        if entry["key"].startswith("push:o1:-x:"):
            entry["derived"] = entry["derived"].replace("path blocked", "path clear")
    out = expert().act(request, commitment, obs)
    assert key_of(request, top(out["q_main"])).startswith("push:o1:-x:none:")


def test_a_push_whose_contact_point_sits_inside_another_object_is_never_chosen():
    """대상에서 곧장 멀어지는 +y 밀기는 접촉점이 대상의 외접 구 안이다 — 어떤 경로로도 닿을 수 없다."""
    obs = blocked_descent_scene()
    request, commitment = committed_request(GRASP, obs=obs)
    out = expert().act(request, commitment, obs)
    admissible = {key_of(request, candidate) for candidate in out["expert_meta"]["main"]["admissible"]}
    assert not any(key.startswith("push:o1:+y:") for key in admissible)
    assert any(key.startswith("push:o1:+x:") for key in admissible)


def test_speed_is_reduced_next_to_a_fragile_object_and_in_place():
    obs = scene()
    obs["objects"][2]["pos_mm"] = [340, 40, -80]  # 취약 물체가 대상 곁
    request, commitment = committed_request(obs=obs)
    assert top(expert().act(request, commitment, obs)["q_speed"]) == str(PROFILES["fragile_speed_cap"])

    placing = scene()
    placing["robot"].update(ee_pos_mm=[30, 240, -40], holding="o0")
    request, commitment = committed_request(GRASP, obs=placing)
    assert request["request"]["commitment"]["phase"] == "place"
    assert top(expert().act(request, commitment, placing)["q_speed"]) == str(PROFILES["speed_by_phase"]["place"])


def test_force_is_push_only_for_push_candidates():
    request, commitment = committed_request()
    assert top(expert().act(request, commitment, scene())["q_force"]) == "0"

    pushing = scene()
    pushing["robot"]["ee_pos_mm"] = [240, 0, -80]  # 접촉점 40mm 안 → push 국면
    request, commitment = committed_request("push:o0:+x:none:slow", obs=pushing)
    assert request["request"]["commitment"]["phase"] == "push"
    assert top(expert().act(request, commitment, pushing)["q_force"]) == "2"


# --------------------------------------------------------------------------
# 라벨 — 전문가 답이 라벨이 된다 (docs/08 §7)
# --------------------------------------------------------------------------


def test_labels_carry_the_source_and_the_rule_grounds():
    request, commitment = committed_request()
    out = expert().act(request, commitment, scene())
    labels = {label["question_id"]: label for label in expert().labels(out, request)}
    assert set(labels) == set(QUESTION_SET_V0)
    assert all(label["source"] == "expert_v0" for label in labels.values())
    assert labels["q_main"]["kind"] == "valid_set" and labels["q_main"]["candidate_ids"] == [candidate_id(GRASP)]
    assert candidate_id(GRASP) in labels["q_main"]["semantic_admissible"]
    for question_id in BOOLEANS:
        assert labels[question_id]["kind"] == "single" and isinstance(labels[question_id]["answer"], bool)
        assert labels[question_id]["rule"]
    expected = f"{commitment['action_ref']}/{commitment['phase']}"
    for question_id in ("q_gripper", "q_path", "q_speed", "q_force"):
        assert labels[question_id]["conditioned_on"] == expected


def test_aux_labels_are_masked_without_a_commitment():
    request = request_for()
    out = expert().act(request, None, scene())
    labels = {label["question_id"] for label in expert().labels(out, request)}
    assert labels == set(BOOLEANS) | {"q_main"}
