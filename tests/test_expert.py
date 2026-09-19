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
from helpers import CONTROLLER_CONFIG, HARNESS_CONFIG, SIM_CONFIG
from test_harness import GRASP, PERIOD_MS, harness, obj, observation

from robo_jev.contracts import QUESTION_SET_V0
from robo_jev.harness.robot import candidate_id
from robo_jev.harness.rule_judge import candidate_values, read_goal
from robo_jev.sim.expert import EXPERT_VERSION, Expert, load_expert_config

CONFIG = load_expert_config()
HARNESS = yaml.safe_load(HARNESS_CONFIG.read_text(encoding="utf-8"))
CONTROLLER = yaml.safe_load(CONTROLLER_CONFIG.read_text(encoding="utf-8"))
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
    assert chosen == "grasp:o0:top:zoneL"
    assert out["q_main"][top(out["q_main"])] == pytest.approx(CONFIDENCE["choice_mass"], abs=1e-4)
    assert out["expert_meta"]["main"]["reason"] == "goal_grasp"


def test_a_different_structured_target_changes_the_choice_without_text_parsing():
    obs = scene()
    obs["goal"].update(target_ref="o1", target_desc="blue 상자")
    request = request_for(obs)
    out = expert().act(request, None, obs)
    assert key_of(request, top(out["q_main"])).startswith("grasp:o1:top:zoneL")


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
    assert key_of(request, top(out["q_main"])).startswith("grasp:o1:top:zoneL")


def test_place_is_chosen_while_holding_without_a_commitment():
    obs = scene()
    obs["robot"].update(ee_pos_mm=[300, 0, 150], holding="o0")
    request = request_for(obs)
    out = expert().act(request, None, obs)
    assert key_of(request, top(out["q_main"])).startswith("place:o0:release:zoneL")


def two_object_scene(target_pos) -> dict:
    """대상과 취약 물체 둘뿐인 장면 — 후보가 상한(29) 안에 들어 어느 것도 잘리지 않는다."""
    return scene(objects=[obj("o0", target_pos), obj("o2", (120, -200, -80), colour="green", attributes=["fragile"])])


def without_goal_grasp(request: dict) -> dict:
    """요청에서 목표 대상→목표 영역의 파지 후보를 뺀 사본 (상한에 걸려 빠진 틱을 흉내 낸다)."""
    trimmed = copy.deepcopy(request)
    trimmed.pop("harness", None)
    trimmed["request"]["candidates"]["q_main"] = [
        entry for entry in trimmed["request"]["candidates"]["q_main"]
        if not entry["key"].startswith("grasp:o0:top:zoneL")
    ]
    return trimmed


def test_push_toward_the_zone_when_the_goal_grasp_is_unavailable():
    """파지가 목록에 없으면(실행 불가·상한) 대상을 목표 영역 쪽으로 미는 후보를 고른다."""
    obs = two_object_scene((250, 240, -80))  # zoneL(x∈[-120,180], y∈[150,330])까지 −x로 70mm
    request = without_goal_grasp(request_for(obs))
    assert any(entry["key"].startswith("push:o0:-x:") for entry in request["request"]["candidates"]["q_main"])
    out = expert().act(request, None, obs)
    chosen = key_of(request, top(out["q_main"]))
    assert chosen.startswith("push:o0:-x:none")
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


def test_low_confidence_labels_are_down_weighted_from_the_config():
    """퇴화 틱(목표 후보 없음·재시도 차단·실행기 사정)의 낮은 신뢰도 라벨은 설정의 `labels.low_confidence_weight`
    (시작값 0.25)를 `weight`로 달고 나간다 — 손실(`loss.py`)이 weight를 그대로 쓰므로 그 틱이 학습을 덜 끈다.
    후보 공간 자체의 결정(I4)은 사람의 몫이고 이것은 그때까지의 완화다."""
    import torch

    from robo_jev.loss import question_losses

    weight = CONFIG["labels"]["low_confidence_weight"]
    assert weight == 0.25
    obs = two_object_scene((300, -100, -80))
    request = without_goal_grasp(request_for(obs))
    request["request"]["candidates"]["q_main"] = [
        entry for entry in request["request"]["candidates"]["q_main"] if not entry["key"].startswith("push:o0:")
    ]
    out = expert().act(request, None, obs)
    labels = {label["question_id"]: label for label in expert().labels(out, request)}
    main = labels["q_main"]
    assert main["label_confidence"] == "low" and main["weight"] == weight
    assert all("weight" not in labels[question_id] for question_id in BOOLEANS)  # 신뢰도 표지가 없는 라벨은 그대로

    # 손실이 그 weight를 쓴다.
    ids = [entry["id"] for entry in request["request"]["candidates"]["q_main"]]
    outputs = {"logits": [{"q_main": torch.zeros(len(ids))}], "candidates": [{"q_main": ids}]}
    entries = question_losses(outputs, {"labels": [[main]]})[0]
    assert entries["q_main"]["weight"] == weight

    # 설정값을 바꾸면 그 값이다. 정상 틱(높은 신뢰도)에는 weight가 없다.
    heavier = Expert({**CONFIG, "labels": {**CONFIG["labels"], "low_confidence_weight": 0.5}})
    assert {l["question_id"]: l for l in heavier.labels(out, request)}["q_main"]["weight"] == 0.5
    normal = request_for()
    normal_labels = {l["question_id"]: l for l in expert().labels(expert().act(normal, None, scene()), normal)}
    assert normal_labels["q_main"]["label_confidence"] == "medium" and "weight" not in normal_labels["q_main"]  # 비용 근거

    # 실행기 사정(`not_executable`)의 낮은 신뢰도도 같은 weight다.
    blocked = two_object_scene((100, 120, -80))
    request = without_goal_grasp(request_for(blocked))
    out = expert().act(request, None, blocked)
    assert out["expert_meta"]["main"]["reason"] == "not_executable"
    assert {l["question_id"]: l for l in expert().labels(out, request)}["q_main"]["weight"] == weight


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
    assert next(e["path"] for e in request["request"]["candidates"]["q_main"] if e["key"] == GRASP) == "blocked"
    assert not [e for e in request["request"]["candidates"]["q_path"] if e["kind"] == "via"]
    out = expert().act(request, commitment, obs)
    chosen = key_of(request, top(out["q_main"]))
    # o1(300, 60)의 영역 쪽 밀기는 −x뿐이다(zoneL 중심까지 −x 성분이 더 크다, 계약 v0.3의 영역 방향 열거) — 대상에서
    # 40mm 멀어지므로 이득 문턱(20)을 넘는다.
    assert chosen == "push:o1:-x:none", out["expert_meta"]["main"]
    assert out["expert_meta"]["main"]["reason"] == "push_blocker"
    assert candidate_id(GRASP) in out["expert_meta"]["main"]["admissible"]


def test_the_blocker_push_is_kept_while_it_still_blocks():
    obs = blocked_descent_scene()
    push = "push:o1:-x:none"
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


def test_push_directions_are_restricted_by_config_to_what_the_executor_can_push():
    """밀기는 닫힌 손가락으로 하지만 ±y는 원통에서 손가락 옆면이 곡면을 비껴 타 반사 정지가 난다(3c-2 실측
    45~90N, 8건 중 6건). 설정의 방향만 고른다 — 실행기 역량이지 의미 적합성이 아니다."""
    assert set(CONFIG["goal"]["push_directions"]) == {"+x", "-x"}
    obs = two_object_scene((100, 120, -80))  # +y가 영역 쪽이지만 허용된 방향이 아니다
    request = without_goal_grasp(request_for(obs))
    out = expert().act(request, None, obs)
    assert key_of(request, top(out["q_main"])) == "hold"

    sideways = two_object_scene((250, 240, -80))  # zoneL(x ≤ 180)까지 −x로 70mm
    request = without_goal_grasp(request_for(sideways))
    out = expert().act(request, None, sideways)
    assert key_of(request, top(out["q_main"])).startswith("push:o0:-x:none")


def test_semantic_admissibility_comes_from_the_goal_not_from_executor_capability():
    """`semantic_admissible`은 지시·목적지·금지 조건에서만 나온다 (docs/08 §7 (1)). 실행기 역량(밀기 방향
    설정)은 **선택**(`q_main`)만 거른다 — 영역 쪽으로 미는 +y 밀기는 고를 수 없어도 여전히 적합하다."""
    obs = two_object_scene((100, 120, -80))  # +y만 영역 쪽 — 설정의 방향(±x)이 아니다
    request = without_goal_grasp(request_for(obs))
    out = expert().act(request, None, obs)
    main = out["expert_meta"]["main"]
    assert key_of(request, top(out["q_main"])) == "hold"
    assert main["reason"] == "not_executable" and main["confidence"] == "low"
    admissible = [key_of(request, candidate) for candidate in main["admissible"]]
    assert admissible and all(key.startswith("push:o0:+y:none") for key in admissible)
    assert set(main["excluded"].values()) == {"push_direction"}
    labels = {label["question_id"]: label for label in expert().labels(out, request)}
    assert labels["q_main"]["candidate_ids"] == [top(out["q_main"])]
    assert labels["q_main"]["semantic_admissible"] == main["admissible"]

    # 목표에 맞는 후보가 하나도 없을 때만 적합 집합이 빈다.
    request["request"]["candidates"]["q_main"] = [
        entry for entry in request["request"]["candidates"]["q_main"] if not entry["key"].startswith("push:o0:")
    ]
    out = expert().act(request, None, obs)
    assert out["expert_meta"]["main"]["reason"] == "goal_candidate_missing"
    assert out["expert_meta"]["main"]["admissible"] == []


def test_a_held_non_target_is_put_down_in_the_goal_zone_first():
    """지시가 바뀌었는데 손에 옛 대상이 있으면 그것을 먼저 놓는다 — 새 대상은 손이 비어야 집는다."""
    obs = scene(instruction={"version": 2, "t_ms": 500, "text": "red 상자 대신 blue 상자를 왼쪽 정리 영역으로 먼저 옮겨라"})
    obs["goal"].update(target_ref="o1", target_desc="blue 상자", version=2)
    obs["robot"].update(ee_pos_mm=[300, 0, 150], holding="o0")
    request = request_for(obs)
    assert not any(e["key"].startswith("grasp:o1:") for e in request["request"]["candidates"]["q_main"])
    out = expert().act(request, None, obs)
    assert key_of(request, top(out["q_main"])).startswith("place:o0:release:zoneL")
    assert out["expert_meta"]["main"]["reason"] == "release_held_object"

    request, commitment = committed_request("place:o0:release:zoneL", obs=obs)
    out = expert().act(request, commitment, obs)
    assert out["expert_meta"]["main"]["reason"] == "keep_commitment"


def test_a_way_the_harness_just_retry_blocked_is_not_proposed_again():
    """직전 틱에 같은 방식이 한계 횟수를 넘겨 실패했으면(`q_retry` 거짓) 하네스가 그 방식을 막는다 —
    그때 같은 답을 내면 하네스는 남은 후보 중 임의의 것을 채택한다. 전문가가 먼저 비켜 준다(hold).
    차단은 하네스의 일시적 상태이지 의미 적합성이 아니다 — `semantic_admissible`에는 그 방식이 남는다."""
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
    assert chosen == "hold"
    main = out["expert_meta"]["main"]
    assert main["reason"] == "way_retry_blocked" and main["confidence"] == "low"
    assert main["blocked_ways"] == ["grasp:o0:top"]
    admissible = [key_of(request, candidate) for candidate in main["admissible"]]
    assert admissible and all(key.startswith("grasp:o0:top:zoneL") for key in admissible)
    assert main["excluded"] == {candidate: "retry_blocked" for candidate in main["admissible"]}
    labels = {label["question_id"]: label for label in expert().labels(out, request)}
    assert labels["q_main"]["candidate_ids"] == [candidate_id("hold")]
    assert candidate_id(GRASP) in labels["q_main"]["semantic_admissible"]
    # hold∉A 규칙: 적합 후보는 판단하지 않는다 (`unknown`) — 손실이 그것을 누르지 않는다.
    assert labels["q_main"]["unknown"] == labels["q_main"]["semantic_admissible"]
    assert labels["q_main"]["weight"] == CONFIG["labels"]["low_confidence_weight"]


# --------------------------------------------------------------------------
# 비키프레임 허용 집합 (docs/08 §7, 계약 v0.3)
# --------------------------------------------------------------------------


def test_the_allowed_set_holds_admissible_candidates_within_the_cost_tolerance():
    """A = {선택} ∪ {적합·실행 가능 후보 : cost ≤ cost(선택) × (1 + τ)}, 신뢰도 medium, unknown 없음.
    들고 있는 대상을 두 영역 중 어디에 놓아도 되는 지시(구조화된 목표의 영역은 하나지만 놓기 후보는 목표 영역
    것만 적합)와, 파지가 없어 밀기 두 방향이 적합한 장면으로 본다."""
    from robo_jev.sim.expert import DEGENERATE_REASONS, GATE_REASONS

    assert CONFIG["labels"]["cost_tolerance"] == 0.15 and expert().cost_tolerance == 0.15
    # 대상 o0(0, 0)은 zoneL(+y)·zoneF(+x) 두 영역 쪽 밀기가 있고, 파지 후보를 빼면 둘 다 적합·실행 가능(±x·±y 중 +x는
    # 실행기 역량 안, +y는 밖)이다 — 실행 가능한 밀기가 하나뿐이면 A는 그것뿐이다.
    zones = [
        {"id": "zoneL", "desc": "왼쪽 정리 영역", "bounds_mm": [-120, 150, 180, 330]},
        {"id": "zoneF", "desc": "앞쪽 보관 영역", "bounds_mm": [100, -80, 300, 80]},
    ]
    obs = scene(objects=[obj("o0", (0, 0, -80)), obj("o2", (-200, -200, -80), colour="green", attributes=["fragile"])], zones=zones)
    obs["goal"].update(zone_ref="zoneF")
    request = without_goal_grasp(request_for(obs))
    request["request"]["candidates"]["q_main"] = [
        e for e in request["request"]["candidates"]["q_main"] if not e["key"].startswith("grasp:o0:top:zoneF")
    ]
    out = expert().act(request, None, obs)
    main = out["expert_meta"]["main"]
    assert main["reason"] == "push_toward_zone" and main["confidence"] == "medium"
    assert main["reason"] not in GATE_REASONS + DEGENERATE_REASONS
    assert set(main["costs_mm"]) == {c for c in main["admissible"] if c not in main["excluded"]}
    assert main["allowed"] == [main["choice"]] and main["unknown"] == []
    labels = {label["question_id"]: label for label in expert().labels(out, request)}
    assert labels["q_main"]["candidate_ids"] == [main["choice"]] and "unknown" not in labels["q_main"]
    assert labels["q_main"]["label_confidence"] == "medium" and "weight" not in labels["q_main"]

    # 비용이 τ 안으로 비슷한 적합 후보 둘: 대상 바로 위에서 두 영역으로 가는 파지 — 같은 거리면 둘 다 허용 집합이다.
    equidistant = scene(objects=[obj("o0", (0, 0, -80))], zones=[
        {"id": "zoneL", "desc": "왼쪽 정리 영역", "bounds_mm": [-120, 150, 180, 330]},
        {"id": "zoneR", "desc": "오른쪽 정리 영역", "bounds_mm": [-120, -330, 180, -150]},
    ])
    equidistant["goal"].update(zone_ref="zoneL", fragile_refs=[])
    request = request_for(equidistant)
    out = expert().act(request, None, equidistant)
    main = out["expert_meta"]["main"]
    assert main["reason"] == "goal_grasp" and main["allowed"] == [candidate_id("grasp:o0:top:zoneL")]
    # 적합 집합은 목표 영역의 파지뿐이므로 zoneR 파지는 I다 — 허용 집합은 적합 집합 안에서만 넓어진다.
    assert candidate_id("grasp:o0:top:zoneR") not in main["admissible"]

    # τ를 키우면 더 비싼 적합 후보도 들어온다: 놓기 후보 둘(영역 중심까지의 xy 거리가 다르다).
    holding = scene(objects=[obj("o0", (60, 100, -40))], zones=[
        {"id": "zoneL", "desc": "왼쪽 정리 영역", "bounds_mm": [-120, 150, 180, 330]},
        {"id": "zoneF", "desc": "앞쪽 보관 영역", "bounds_mm": [100, -80, 300, 80]},
    ])
    holding["goal"].update(zone_ref="zoneL", fragile_refs=[])
    holding["robot"].update(holding="o0", ee_pos_mm=[60, 100, -40])
    request = request_for(holding)
    request["request"]["candidates"]["q_main"] = [
        e for e in request["request"]["candidates"]["q_main"] if not e["key"].startswith("grasp:")
    ]
    out = expert().act(request, None, holding)
    main = out["expert_meta"]["main"]
    assert main["reason"] == "goal_place" and main["allowed"] == [candidate_id("place:o0:release:zoneL")]
    assert set(main["costs_mm"]) == {candidate_id("place:o0:release:zoneL")}  # 적합 = 목표 영역의 놓기뿐


def test_a_kept_commitment_within_the_cost_tolerance_is_the_unique_answer():
    """commitment 규칙 (3): 지킨 commitment가 최선 비용의 (1 + τ) 안이면 그것만 정답이다 — 비용이 같은 대안이 있어도."""
    request, commitment = committed_request()
    out = expert().act(request, commitment, scene())
    main = out["expert_meta"]["main"]
    assert main["reason"] == "keep_commitment" and main["allowed"] == [candidate_id(GRASP)]
    assert main["confidence"] == "medium"
    labels = {label["question_id"]: label for label in expert().labels(out, request)}
    assert labels["q_main"]["candidate_ids"] == [candidate_id(GRASP)] and labels["q_main"]["label_confidence"] == "medium"


def test_plan_cost_is_the_remaining_command_path_length():
    """플래너 비용 = 국면 목표점까지의 거리 + 남은 국면의 구간 (하네스와 같은 수치, 새 물리 없음)."""
    ex = expert()
    request = request_for()
    state = request["request"]["state"]
    goal = read_goal(state)
    entry = next(e for e in request["request"]["candidates"]["q_main"] if e["key"] == GRASP)
    value = candidate_values(entry)
    centre = ((-120 + 180) / 2, (150 + 330) / 2)
    expected = (
        entry["d"] + HARNESS["candidates"]["approach_clearance_mm"] + HARNESS["candidates"]["grasp_depth_mm"]
        + HARNESS["phases"]["lift_height_mm"] + math.dist((300, 0), centre) + HARNESS["candidates"]["approach_clearance_mm"]
    )
    assert ex.plan_cost_mm(value, state, goal) == pytest.approx(expected)
    blocked = dict(value, path_clear=False)
    assert ex.plan_cost_mm(blocked, state, goal) == pytest.approx(expected + 2 * HARNESS["planner"]["side_offset_mm"])
    push = candidate_values(next(e for e in request["request"]["candidates"]["q_main"] if e["key"] == "push:o0:-x:none"))
    remaining = math.hypot(300 - 180, 150 - 0)  # o0(300, 0)에서 zoneL 사각형까지
    segments = math.ceil(remaining / HARNESS["candidates"]["push_segment_mm"])
    assert ex.plan_cost_mm(push, state, goal) == pytest.approx(
        push["distance_mm"] + segments * HARNESS["candidates"]["push_segment_mm"]
        + (segments - 1) * (HARNESS["candidates"]["approach_clearance_mm"] + HARNESS["candidates"]["push_contact_mm"])
    )
    assert ex.plan_cost_mm(candidate_values({"id": "x", "key": "hold"}), state, goal) == 0.0


def test_degenerate_ticks_follow_the_hold_not_in_a_rule():
    """실행기 사정의 `hold`(not_executable·goal_candidate_missing): A = {hold}, unknown = 적합 집합, 신뢰도 low."""
    obs = two_object_scene((100, 120, -80))  # +y만 영역 쪽 — 실행기 역량 밖
    request = without_goal_grasp(request_for(obs))
    out = expert().act(request, None, obs)
    main = out["expert_meta"]["main"]
    assert main["reason"] == "not_executable" and main["allowed"] == [candidate_id("hold")]
    assert main["unknown"] == main["admissible"] and main["admissible"]
    labels = {label["question_id"]: label for label in expert().labels(out, request)}
    assert labels["q_main"]["candidate_ids"] == [candidate_id("hold")]
    assert labels["q_main"]["unknown"] == main["admissible"] and labels["q_main"]["label_confidence"] == "low"
    assert labels["q_main"]["weight"] == CONFIG["labels"]["low_confidence_weight"]
    # 적합 후보가 아예 없으면 unknown도 없다 (I = hold 밖의 전부).
    request["request"]["candidates"]["q_main"] = [
        entry for entry in request["request"]["candidates"]["q_main"] if not entry["key"].startswith("push:o0:")
    ]
    out = expert().act(request, None, obs)
    labels = {label["question_id"]: label for label in expert().labels(out, request)}
    assert out["expert_meta"]["main"]["reason"] == "goal_candidate_missing"
    assert labels["q_main"]["candidate_ids"] == [candidate_id("hold")] and "unknown" not in labels["q_main"]


def test_contact_phases_gate_observation_on_readiness_not_on_geometry_age():
    """접촉 국면·파지 중에는 대상 기하가 하네스의 실행 가능성 문턱보다 늙어도 관측을 요구하지 않는다 — 실행기의
    readiness가 시점을 정한다(하네스 §5.0과 같은 면제; 계약 v0.3 이월 항목)."""
    hrn = harness()
    hrn.build_request(scene(), None, None)
    stale_ms = HARNESS["candidates"]["max_geometry_age_ms"] + 3 * PERIOD_MS
    descending = scene(tick=stale_ms // PERIOD_MS, sim_time_ms=stale_ms)
    descending["robot"]["ee_pos_mm"] = [300, 0, -30]
    descending["objects"][0].update(visible=False, visible_ratio=0.0)
    request, commitment = committed_request(GRASP, obs=descending, hrn=hrn)
    assert request["request"]["commitment"]["phase"] == "grasp"
    target = next(entry for entry in request["request"]["state"]["objects"] if entry["id"] == "o0")
    assert target["age_ms"] > HARNESS["candidates"]["max_geometry_age_ms"]
    out = expert().act(request, commitment, descending)
    assert out["q_observe"] == CONFIDENCE["low"]
    assert out["expert_meta"]["gates"]["q_observe"]["reason"] == "contact_phase"
    from robo_jev.harness.rule_judge import rule_judge

    assert rule_judge(request)["q_observe"] == CONFIDENCE["low"]


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


def test_a_push_whose_hand_occludes_the_target_does_not_gate_to_observe():
    """밀기(접촉) 국면도 팔이 대상을 가리는 국면이다 (docs/08 §5.0·§6): 하네스의 `CONTACT_PHASES`가 단일 출처이고
    전문가는 그것을 읽는다 — 밀리는 물체의 기하 나이가 관측 문턱을 넘어도 `max_geometry_age_ms` 안이면 관측이 아니다."""
    from robo_jev.harness.robot import CONTACT_PHASES
    from robo_jev.sim import expert as expert_module

    assert "push" in CONTACT_PHASES
    assert expert_module._CONTACT_PHASES is CONTACT_PHASES

    hrn = harness()
    hrn.build_request(scene(), None, None)
    stale_ms = THRESHOLDS["observe_geom_age_ms"] + PERIOD_MS
    pushing = scene(tick=stale_ms // PERIOD_MS, sim_time_ms=stale_ms)
    pushing["robot"]["ee_pos_mm"] = [400, 0, -80]  # −x 밀기의 접촉점(372, 0) 40mm 안 → push 국면
    pushing["objects"][0].update(visible=False, visible_ratio=0.0)  # 손이 대상을 가린다
    request, commitment = committed_request("push:o0:-x:none", obs=pushing, hrn=hrn)
    assert request["request"]["commitment"]["phase"] == "push"
    target = next(entry for entry in request["request"]["state"]["objects"] if entry["id"] == "o0")
    assert THRESHOLDS["observe_geom_age_ms"] < target["age_ms"] <= HARNESS["candidates"]["max_geometry_age_ms"]
    out = expert().act(request, commitment, pushing)
    assert out["q_observe"] == CONFIDENCE["low"]
    assert out["expert_meta"]["gates"]["q_observe"]["reason"] == "contact_phase"
    assert out["expert_meta"]["main"]["reason"] != "observe_target"
    assert key_of(request, top(out["q_main"])) != "observe"

    # 같은 나이의 대상을 접근 국면에서 밀러 가는 중이면(손이 아직 가리지 않는다) 관측이 맞다.
    approaching = copy.deepcopy(pushing)
    approaching["robot"]["ee_pos_mm"] = [100, 0, 100]
    request, commitment = committed_request("push:o0:-x:none", obs=approaching, hrn=hrn)
    assert request["request"]["commitment"]["phase"] == "approach"
    assert expert().act(request, commitment, approaching)["q_observe"] == CONFIDENCE["high"]


def test_observation_waits_while_another_object_is_held():
    """지시가 바뀌어 새 대상이 안 보이는데 손에 옛 대상이 있으면 관측은 지금 할 수 있는 일이 아니다 — 운반 중의
    관측은 제자리 hold이고 가리는 것은 팔 자신이라 영영 풀리지 않는다(E1 seed 2, 244틱). 먼저 놓는다."""
    obs = scene(instruction={"version": 2, "t_ms": 500, "text": "red 상자 대신 blue 상자를 왼쪽 정리 영역으로 먼저 옮겨라"})
    obs["goal"].update(target_ref="o1", target_desc="blue 상자", version=2)
    obs["objects"][1].update(visible=False, visible_ratio=0.0)
    obs["robot"].update(ee_pos_mm=[300, 0, 150], holding="o0")
    request = request_for(obs)
    out = expert().act(request, None, obs)
    assert out["q_observe"] == CONFIDENCE["low"]
    assert out["expert_meta"]["gates"]["q_observe"]["reason"] == "holding_other_first"
    assert key_of(request, top(out["q_main"])).startswith("place:o0:release:zoneL")
    assert out["expert_meta"]["main"]["reason"] == "release_held_object"

    # 손이 비면 다시 관측을 요구한다.
    obs["robot"].update(holding=None)
    request = request_for(obs)
    out = expert().act(request, None, obs)
    assert out["q_observe"] == CONFIDENCE["high"]


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


def test_the_gripper_stays_closed_on_a_place_tick_whose_path_answer_is_hold_or_retreat():
    """리뷰 2 C2: 막힌 하강(경유점 없음)의 놓기 틱에 `open`을 답하면 하네스의 hold 경로 + 운반 높이 open이 라벨이 된다.
    전문가의 경로 답이 내려가는 경로(direct·via)가 아니면 그리퍼 답은 `closed`(이유 `place_blocked`)다 — 모델 입력만 본다."""
    hrn = harness()
    placing = scene(tick=4, sim_time_ms=4 * PERIOD_MS)
    placing["robot"].update(ee_pos_mm=[30, 240, -40], holding="o0")
    request, commitment = committed_request(GRASP, obs=placing, hrn=hrn)
    assert request["request"]["commitment"]["phase"] == "place"
    assert top(expert().act(request, commitment, placing)["q_gripper"]) == "open"

    blocked = copy.deepcopy(request)
    entry = next(item for item in blocked["request"]["candidates"]["q_main"] if item["id"] == candidate_id(GRASP))
    entry["path"] = "blocked"
    blocked["request"]["candidates"]["q_path"] = [item for item in blocked["request"]["candidates"]["q_path"] if item["kind"] != "via"]
    out = expert().act(blocked, commitment, placing)
    kinds = {item["id"]: item["kind"] for item in blocked["request"]["candidates"]["q_path"]}
    assert kinds[top(out["q_path"])] == "hold"
    assert top(out["q_gripper"]) == "closed"
    assert out["expert_meta"]["aux"]["gripper"]["reason"] == "place_blocked"
    labels = {label["question_id"]: label for label in expert().labels(out, blocked)}
    assert labels["q_gripper"]["candidate_ids"] == ["closed"]
    assert labels["q_gripper"]["rule"].endswith("/place_blocked")


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


def test_retreat_when_the_hand_holding_the_target_sits_inside_a_forbidden_objects_inflated_sphere():
    """금지 물체는 하네스에 `forbidden_margin_mm`만큼 더 큰 장애물이다. 파지 뒤 물체가 그쪽으로 밀려
    들기 구간의 시작점이 그 넓힌 구 안에 들면 어떤 경유점도 없다(E0 seed 110의 들기 정지). 손이 그
    안에 있으면 물러난다 — 기본 여유로만 보면 '안'이 아니어서 영영 기다린다."""
    obs = scene(objects=[
        obj("o0", (474, 97, -85), obb_mm=[54, 54, 32]),
        obj("o2", (505, -23, -84), colour="green", obb_mm=[42, 44, 56], attributes=["forbidden"]),
    ])
    obs["goal"].update(forbidden_refs=["o2"], fragile_refs=[])
    obs["robot"].update(ee_pos_mm=[474, 97, -80], holding="o0", gripper_mm=30)
    request, commitment = committed_request(GRASP.replace("o0:top:zoneL", "o0:top:zoneL"), obs=obs)
    assert request["request"]["commitment"]["phase"] == "lift"
    assert next(e["path"] for e in request["request"]["candidates"]["q_main"] if e["key"] == GRASP) == "blocked"
    kinds = {entry["id"]: entry["kind"] for entry in request["request"]["candidates"]["q_path"]}
    out = expert().act(request, commitment, obs)
    assert kinds[top(out["q_path"])] == "retreat"
    assert out["expert_meta"]["aux"]["path"]["reason"] == "hand_inside_obstacle"


def test_a_clear_approach_outranks_a_larger_gain_for_a_blocker_push():
    """접촉점이 닿을 수 있는 방향들 가운데서는 접근이 비어 있는 쪽이 먼저다(이득은 그다음). 후보 설명의
    `path`만 바꿔 순위 규칙을 본다: −x의 접근만 비어 있으면 키 순으로 앞서는 +x 대신 −x다. 밀기는 영역 쪽 축만
    만들어지므로(계약 v0.3) +x 쪽에 영역을 하나 더 둔다."""
    obs = blocked_descent_scene()
    obs["zones"].append({"id": "zoneX", "desc": "앞쪽 영역", "bounds_mm": [500, -80, 700, 80]})
    request, commitment = committed_request(GRASP, obs=obs)
    request = copy.deepcopy(request)
    request.pop("harness")
    keys = {entry["key"] for entry in request["request"]["candidates"]["q_main"]}
    assert {"push:o1:-x:none", "push:o1:+x:none"} <= keys
    for entry in request["request"]["candidates"]["q_main"]:
        if entry["key"] == "push:o1:-x:none":
            entry["path"] = "ok"
        elif entry["key"] == "push:o1:+x:none":
            entry["path"] = "blocked"
    out = expert().act(request, commitment, obs)
    assert key_of(request, top(out["q_main"])) == "push:o1:-x:none"


def test_a_push_whose_contact_point_sits_inside_another_object_is_never_chosen():
    """대상에서 곧장 멀어지는 +y 밀기는 접촉점이 대상의 외접 구 안이다 — 어떤 경로로도 닿을 수 없다.
    접촉점 도달성은 실행기 쪽 조건이므로 선택에서만 뺀다: 목표에는 맞으니(막는 이웃을 대상에서
    멀리 민다) `semantic_admissible`에는 남고, 뺀 이유는 근거에 적힌다."""
    obs = blocked_descent_scene()
    # 밀기는 영역 쪽 축만 만들어지므로(계약 v0.3) +y 쪽과 +x 쪽에 영역을 둔다 — o1의 밀기는 −x·+y·+x다.
    obs["zones"].extend([
        {"id": "zoneU", "desc": "위쪽 영역", "bounds_mm": [200, 300, 400, 500]},
        {"id": "zoneX", "desc": "앞쪽 영역", "bounds_mm": [500, -80, 700, 80]},
    ])
    request, commitment = committed_request(GRASP, obs=obs)
    every_direction = copy.deepcopy(CONFIG)
    every_direction["goal"]["push_directions"] = None  # 방향 설정과 무관하게 접촉점만 본다
    out = Expert(every_direction).act(request, commitment, obs)
    main = out["expert_meta"]["main"]
    assert not key_of(request, main["choice"]).startswith("push:o1:+y:")
    admissible = {key_of(request, candidate) for candidate in main["admissible"]}
    assert "push:o1:+y:none" in admissible and "push:o1:+x:none" in admissible
    assert candidate_id(GRASP) in main["admissible"]
    excluded = {key_of(request, candidate): reason for candidate, reason in main["excluded"].items()}
    assert excluded == {"push:o1:+y:none": "contact_point"}


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
    pushing["robot"]["ee_pos_mm"] = [400, 0, -80]  # −x 밀기의 접촉점(372, 0) 40mm 안 → push 국면
    request, commitment = committed_request("push:o0:-x:none", obs=pushing)
    assert request["request"]["commitment"]["phase"] == "push"
    assert top(expert().act(request, commitment, pushing)["q_force"]) == "2"


def test_push_candidates_close_the_fingers_before_contact():
    """밀기는 닫힌 손가락(주먹)으로 한다: 접촉점으로 가는 접근 국면부터 `closed`다. 열린 손가락은 y축으로
    벌어져 ±y 접근에서 물체를 치고(3c-1 측정 45~78N), ±x에서는 물체가 손가락 사이로 빠진다."""
    approaching = scene()
    approaching["robot"]["ee_pos_mm"] = [100, 0, 100]
    request, commitment = committed_request("push:o0:-x:none", obs=approaching)
    assert request["request"]["commitment"]["phase"] == "approach"
    out = expert().act(request, commitment, approaching)
    assert top(out["q_gripper"]) == "closed"
    assert out["expert_meta"]["aux"]["gripper"]["reason"] == "push_with_closed_fingers"

    pushing = scene()
    pushing["robot"]["ee_pos_mm"] = [400, 0, -80]
    request, commitment = committed_request("push:o0:-x:none", obs=pushing)
    assert request["request"]["commitment"]["phase"] == "push"
    assert top(expert().act(request, commitment, pushing)["q_gripper"]) == "closed"

    # 파지 후보의 접근 국면은 여전히 열려 있다.
    request, commitment = committed_request(GRASP, obs=approaching)
    assert request["request"]["commitment"]["phase"] == "approach"
    assert top(expert().act(request, commitment, approaching)["q_gripper"]) == "open"


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


def test_a_label_whose_answer_set_is_empty_is_masked_by_omission():
    """허용 집합이 빈 라벨은 근거가 없는 질문이다 (docs/08 §7 "근거가 없는 질문은 loss mask"). 계약은 `valid_set`의
    `candidate_ids`가 비는 것을 거절하므로 mask는 라벨을 아예 적지 않는 것이다 — 경로 후보가 없는 틱의 `q_path`."""
    from robo_jev.contracts import validate_record

    request, commitment = committed_request()
    request["request"]["candidates"].pop("q_path")
    out = expert().act(request, commitment, scene())
    assert out["q_path"] == {}
    labels = expert().labels(out, request)
    ids = {label["question_id"] for label in labels}
    assert "q_path" not in ids and {"q_gripper", "q_speed", "q_force"} <= ids
    assert all(label["candidate_ids"] for label in labels if label["kind"] == "valid_set")
    tick = {key: value for key, value in request.items() if key != "harness"}
    validate_record({
        "schema_version": "stream-v0", "episode_id": "ep-x",
        "prefix": {"instructions": [{"version": 1, "t_ms": 0, "text": "x"}], "question_set": "qs-v0"},
        "ticks": [{**tick, "labels": labels}],
    })


# --------------------------------------------------------------------------
# 폐루프 인수 — 실제 환경에서 E0·E1을 완료한다
# --------------------------------------------------------------------------


def run_episode(profile: str, seed: int) -> dict:
    from robo_jev.data.robot_episodes import generate_episode, load_generator_config

    policy = Expert()
    record = generate_episode(profile, seed, policy=policy, expert=policy, config=load_generator_config())
    return record["provenance"]["outcome"]


def assert_completed(outcome: dict) -> None:
    assert outcome["done"] is True, outcome
    assert outcome["done_tick"] is not None and outcome["done_tick"] < 300, outcome
    assert outcome["target_inside_zone"] is True, outcome
    assert outcome["holding"] is None, outcome
    # 놓았다는 것은 손이 비었다는 판정만이 아니라 손가락이 실제로 열렸다는 것이다.
    gripper = CONTROLLER["gripper"]
    assert outcome["gripper_mm"] > (gripper["open_mm"] + gripper["closed_mm"]) / 2, outcome


@pytest.mark.parametrize("seed", [17, 29, 43, 101])
def test_the_expert_completes_an_e0_episode(seed):
    """101은 3c-1의 파지 결함(키 큰 원통, 대각선 하강 → 정지 76틱) seed — 파지 진입 조건(xy 정렬·잦아들기)이 고쳤다."""
    assert_completed(run_episode("E0", seed))


class ForcedMain:
    """전문가의 답 위에 주 결정만 한 후보로 못박고 게이팅을 끈 정책 — 밀기 실측용."""

    version = "forced-main"

    def __init__(self, expert: Expert, key: str) -> None:
        self.expert, self.key = expert, key

    def act(self, request, commitment, observation=None):
        out = self.expert.act(request, commitment, observation)
        ids = [entry["id"] for entry in request["request"]["candidates"]["q_main"]]
        if candidate_id(self.key) in ids:
            out["q_main"] = self.expert._spread(candidate_id(self.key), ids)
        out.update(q_done=CONFIDENCE["low"], q_instr=CONFIDENCE["high"], q_observe=CONFIDENCE["low"], q_retry=CONFIDENCE["high"])
        return out


def test_a_push_moves_the_object_with_closed_fingers_at_mid_height():
    """3c-1 측정에서 열린 손가락의 ±x 밀기는 120틱에 ≤5mm였다(물체가 손가락 사이로 빠진다). 주먹으로 물체
    중간 높이를 밀면 E0 seed 43의 상자 o1(34×56×42)이 4초 안에 한 구간(80mm)의 절반 이상 움직이고 접촉력은
    반사 한계 아래다."""
    from robo_jev.harness.robot import RobotHarness, load_harness_config
    from robo_jev.sim.environment import Environment

    policy_expert = Expert()
    env = Environment(config_path=str(SIM_CONFIG), profile="E0")
    try:
        scene = env.reset(seed=43)
        target = next(entry for entry in scene["objects"] if entry["id"] == "o1")
        assert target["shape"] == "box"
        start = list(target["pos_mm"])
        hrn = RobotHarness(load_harness_config())
        policy = ForcedMain(policy_expert, "push:o1:+x:none")
        commitment = history = None
        max_force, closed_ticks = 0.0, 0
        for _ in range(40):
            request = hrn.build_request(scene, history, commitment)
            answers = policy.act(request, commitment, scene)
            out = hrn.compose(request, {q: answers[q] for q in QUESTION_SET_V0}, commitment, int(scene["sim_time_ms"]))
            ack = None
            for step in range(5):
                scene = env.step(out["command"] if step == 0 else None)
                ack = scene["ack"] or ack
                max_force = max(max_force, scene["robot"]["contact_force_n"])
            committed = request["request"]["commitment"]
            if committed and committed["phase"] == "push" and not out["adopted"]["stop"]:  # 부가 답은 틱 시작 시 commitment 기준
                assert out["command"]["force_level"] == "push" and out["command"]["gripper"] == "closed"
                surface = target["pos_mm"][2] - target["obb_mm"][2] // 2  # 놓인 물체의 바닥 = 작업면
                height = max(start[2], surface + HARNESS["candidates"]["push_height_min_mm"])
                assert abs(out["command"]["path"]["target_mm"][2] - height) <= 2  # 중간 높이 (작업면 위 최소 높이)
                closed_ticks += int(scene["robot"]["gripper_mm"] < 20)
            commitment, history = out["commitment"], {"adopted": out["adopted"], "ack": ack, "gate": out["gate"]}
        end = next(entry for entry in scene["objects"] if entry["id"] == "o1")["pos_mm"]
    finally:
        env.close()
    assert end[0] - start[0] >= HARNESS["candidates"]["push_segment_mm"] * 0.5, (start, end)
    assert abs(end[1] - start[1]) < 40
    assert max_force < CONTROLLER["reflex"]["force_limit_n"], max_force
    assert closed_ticks >= 10


@pytest.mark.parametrize("seed", [17, 43, 11, 29])
def test_the_expert_completes_an_e1_episode(seed):
    """E1: 물체 6~10개, 취약·금지, 지시 변경(5~15초), 외란. 300틱 안에 대상이 영역 안에 놓이고 손은 빈다.
    seed 29는 옛 후보 상한(29 + 고정 3)이 지시의 대상×목표 영역 파지를 목록에서 빼 hold하던 seed다 — 계약 v0.3의
    지시 조합 예약(K ≤ 12 = 9 + 3, 대상×영역 조합 먼저)이 고쳤다."""
    assert_completed(run_episode("E1", seed))
