"""로봇 스트림 하네스 검사 — 요청 구성과 조합 규칙 v0 (docs/08 §3~§6).

하네스는 순수 논리다. 물리는 `tests/test_sim_replay.py`가 본다. 여기서는 합성 관측과
합성 시각으로 몰아 후보 생성·경유점·조합 규칙을 하나씩 확인하고, 마지막에 **실제
컨트롤러**가 하네스의 명령을 받아들이는지 본다. 수치는 전부 설정에서 읽는다.
"""

import copy
import math

import pytest
import yaml
from helpers import CONTROLLER_CONFIG, D0_STREAMS, HARNESS_CONFIG, SIM_CONFIG, read_jsonl

from robo_jev.contracts import AUX_QUESTIONS, QUESTION_SET_V0, model_input, validate_record
from robo_jev.data.episode import aggregate, append_tick, finalize, new_episode
from robo_jev.data.split import assign_split
from robo_jev.data.validate import validate_dataset
from robo_jev.harness.robot import HARNESS_VERSION, RobotHarness, candidate_id, count_records
from robo_jev.harness.rule_judge import rule_judge
from robo_jev.sim.controller import Controller

CONFIG = yaml.safe_load(HARNESS_CONFIG.read_text(encoding="utf-8"))
CONTROLLER = yaml.safe_load(CONTROLLER_CONFIG.read_text(encoding="utf-8"))
COMPOSE = CONFIG["compose"]
CANDIDATES = CONFIG["candidates"]
PLANNER = CONFIG["planner"]
DELTA = COMPOSE["delta"]
M = COMPOSE["m"]
PERIOD_MS = 100  # 판단 틱 10Hz (docs/08 §1)


# --------------------------------------------------------------------------
# 합성 관측
# --------------------------------------------------------------------------


def obj(object_id: str, pos_mm, **over) -> dict:
    entry = {
        "id": object_id,
        "class": "box",
        "shape": "box",
        "colour": "red",
        "pos_mm": list(pos_mm),
        "quat": [0.0, 0.0, 0.0, 1.0],
        "obb_mm": [60, 60, 64],
        "visible": True,
        "visible_ratio": 1.0,
        "attributes": [],
        # `None`이면 관측 시각으로 채운다 — 보이는 물체의 `last_seen_ms`는 지금이다
        # (`Environment._observation`과 같은 규칙).
        "last_seen_ms": None,
        # 설명은 지시문이 물체를 부르는 이름과 같은 말이다 (`Environment._observation`).
        "desc": None,
    }
    entry.update(over)
    return entry


def observation(objects=None, **over) -> dict:
    """`Environment.step`이 내는 관측의 최소 형태."""
    base = {
        "tick": 0,
        "sim_time_ms": 0,
        "instruction": {"version": 1, "t_ms": 0, "text": "red 상자를 왼쪽 정리 영역으로 옮겨라"},
        "objects": objects
        if objects is not None
        else [
            obj("o0", (300, 0, -80)),
            obj("o1", (200, 220, -80), colour="blue"),
            obj("o2", (120, -200, -80), colour="green", attributes=["fragile"]),
        ],
        "zones": [{"id": "zoneL", "desc": "왼쪽 정리 영역", "bounds_mm": [-120, 150, 180, 330]}],
        "robot": {
            "ee_pos_mm": [0, 0, 200],
            "ee_quat": [0.0, 0.0, 0.0, 1.0],
            "gripper_mm": 80,
            "holding": None,
            "contact_force_n": 0.0,
            "speed_mm_s": 0,
        },
        "events": [],
        "exec": {"seq": 0, "executor": "HOLD", "action_ref": None, "phase": None},
        "ack": None,
    }
    base.update(over)
    for entry in base["objects"]:
        if entry.get("last_seen_ms") is None:
            entry["last_seen_ms"] = base["sim_time_ms"] if entry["visible"] else 0
        if entry.get("desc") is None:
            entry["desc"] = f"{entry['colour']} 상자"
    return base


def harness() -> RobotHarness:
    return RobotHarness.from_config_path(HARNESS_CONFIG)


def tick_of(request: dict) -> dict:
    """요청 하나를 계약 검사가 받는 틱으로 옮긴다 (하네스 블록은 레코드에 가지 않는다)."""
    return {key: value for key, value in request.items() if key != "harness"}


def stream(*ticks: dict) -> dict:
    return {
        "schema_version": "stream-v0",
        "episode_id": "ep-test",
        "prefix": {
            "instructions": [{"version": 1, "t_ms": 0, "text": "red 상자를 옮겨라"}],
            "question_set": "qs-v0",
        },
        "ticks": [tick_of(tick) for tick in ticks],
    }


def keys_of(request: dict) -> list[str]:
    return [entry["key"] for entry in request["request"]["candidates"]["q_main"]]


def candidate_for(request: dict, key: str) -> dict:
    return next(entry for entry in request["request"]["candidates"]["q_main"] if entry["key"] == key)


# --------------------------------------------------------------------------
# 요청 구성
# --------------------------------------------------------------------------


def test_request_is_a_valid_tick():
    request = harness().build_request(observation(), None, None)
    validate_record(stream(request))


def test_question_wording_matches_the_contract():
    """설정의 ko 문구는 계약의 질문 세트 v0와 같아야 한다 (docs/08 §4)."""
    texts = harness().question_texts()
    assert set(texts) == set(QUESTION_SET_V0)
    for question_id, spec in QUESTION_SET_V0.items():
        assert texts[question_id] == spec["instructions"]


def test_english_variant_is_selectable_by_config():
    english = copy.deepcopy(CONFIG)
    english["language"] = "en"
    hrn = RobotHarness(english)
    assert hrn.question_set_id() == CONFIG["question_set_id"]["en"]
    assert hrn.question_texts()["q_main"] == CONFIG["questions"]["q_main"]["en"]
    request = hrn.build_request(observation(), None, None)
    assert "Grasp" in candidate_for(request, keys_of(request)[0])["desc"]


def test_candidates_combine_function_target_approach_destination_profile():
    request = harness().build_request(observation(), None, None)
    keys = keys_of(request)
    assert "grasp:o0:top:zoneL:slow" in keys
    assert "grasp:o0:top:zoneL:fast" in keys
    assert "push:o0:+x:none:slow" in keys
    for key in keys:
        if key in ("observe", "hold", "replan"):
            continue
        function, target, approach, destination, profile = key.split(":")
        assert function in ("grasp", "place", "push")
        assert target.startswith("o")
        assert approach and destination and profile in CANDIDATES["profiles"]


def test_observe_hold_replan_are_always_offered():
    request = harness().build_request(observation(), None, None)
    assert {"observe", "hold", "replan"} <= set(keys_of(request))


def test_candidate_ids_are_derived_from_the_key_not_from_the_position():
    """후보 id는 의미 키에서 나온다 — 목록이 바뀌어도 같은 행동은 같은 id다.

    같은 인덱스를 같은 후보로 해석하는 것이 docs/02 §6이 금지한 실수다.
    """
    hrn = harness()
    first = hrn.build_request(observation(), None, None)
    fewer = observation(objects=[obj("o1", (200, 220, -80), colour="blue"), obj("o0", (300, 0, -80))])
    second = hrn.build_request(fewer, None, None)
    key = "grasp:o0:top:zoneL:slow"
    assert candidate_for(first, key)["id"] == candidate_for(second, key)["id"]


def test_place_candidates_appear_only_while_holding():
    scene = observation()
    assert not any(key.startswith("place:") for key in keys_of(harness().build_request(scene, None, None)))

    scene["robot"]["holding"] = "o0"
    keys = keys_of(harness().build_request(scene, None, None))
    assert "place:o0:release:zoneL:slow" in keys
    # 들고 있는 물체의 파지 후보는 **진행 중인 결합 행동**이라 남고, 다른 물체는 잡을 수 없다.
    assert "grasp:o0:top:zoneL:slow" in keys
    assert not any(key.startswith("grasp:o1") or key.startswith("push:") for key in keys)


def test_an_in_progress_joint_action_aims_at_its_destination():
    """`grasp:o0:…:zoneL:…`은 "집어서 zoneL로 옮긴다" 하나다 — 집은 뒤에는 목적지를 향한다."""
    scene = observation()
    scene["robot"].update(holding="o0", ee_pos_mm=[300, 0, -40])
    request = harness().build_request(scene, None, None)
    geometry = request["harness"]["candidates"][candidate_id("grasp:o0:top:zoneL:slow")]
    assert geometry["phase"] in ("lift", "transport", "place")
    assert geometry["action_mm"][:2] == [30, 240]  # zoneL 중심


def test_unreachable_combinations_are_removed():
    far = observation(objects=[obj("o0", (300, 0, -80)), obj("o9", (2000, 0, -80), colour="blue")])
    request = harness().build_request(far, None, None)
    assert not any(":o9:" in key for key in keys_of(request))
    assert request["harness"]["accounting"]["dropped"]["unreachable"] > 0


def test_feasibility_follows_geometry_age_not_visibility():
    """가시 비율은 정보이지 실행 가능성의 기준이 아니다 — 기하 나이가 기준이다 (docs/08 §3.2)."""
    hrn = harness()
    hrn.build_request(observation(), None, None)  # 처음에는 다 보인다

    covered = observation(tick=2, sim_time_ms=200)
    covered["objects"][1].update(visible=False, visible_ratio=0.1)
    request = hrn.build_request(covered, None, None)
    assert any(":o1:" in key for key in keys_of(request)), "가려졌을 뿐 기하는 새것이다"
    assert "occluded" not in request["harness"]["accounting"]["dropped"]
    assert request["harness"]["accounting"]["dropped"]["stale"] == 0

    stale_ms = CANDIDATES["max_geometry_age_ms"] + 300
    stale = observation(tick=stale_ms // PERIOD_MS, sim_time_ms=stale_ms)
    stale["objects"][1].update(visible=False, visible_ratio=0.1)
    request = hrn.build_request(stale, None, None)
    assert not any(":o1:" in key for key in keys_of(request))
    assert request["harness"]["accounting"]["dropped"]["stale"] > 0


def test_a_held_target_is_never_stale():
    """들고 있는 물체의 자세는 말단에서 온다(나이 0) — 팔이 가려도 후보가 남는다."""
    hrn = harness()
    hrn.build_request(observation(), None, None)
    stale_ms = CANDIDATES["max_geometry_age_ms"] + 300
    carrying = observation(tick=stale_ms // PERIOD_MS, sim_time_ms=stale_ms)
    carrying["robot"].update(holding="o0", ee_pos_mm=[300, 0, -40])
    carrying["objects"][0].update(visible=False, visible_ratio=0.0)
    request = hrn.build_request(carrying, None, None)
    assert "grasp:o0:top:zoneL:slow" in keys_of(request)
    assert request["harness"]["candidates"][candidate_id("grasp:o0:top:zoneL:slow")]["geometry_age_ms"] == 0


def test_a_target_in_the_grasp_phase_is_kept_while_the_arm_hides_it():
    """파지 국면에서 팔이 대상을 가려 기하가 늙어도 후보는 사라지지 않는다 (docs/08 §3.2)."""
    hrn = harness()
    hrn.build_request(observation(), None, None)
    stale_ms = CANDIDATES["max_geometry_age_ms"] + 300
    descending = observation(tick=stale_ms // PERIOD_MS, sim_time_ms=stale_ms)
    descending["robot"].update(ee_pos_mm=[300, 0, -20])  # 접근 지점 아래, 파지 지점 근처
    descending["objects"][0].update(visible=False, visible_ratio=0.0)
    request = hrn.build_request(descending, None, None)
    geometry = request["harness"]["candidates"].get(candidate_id("grasp:o0:top:zoneL:slow"))
    assert geometry is not None and geometry["phase"] == "grasp"
    assert not any(key.startswith("push:o0:") for key in keys_of(request)), "접촉 전의 밀기는 늙은 기하로 만들지 않는다"


def test_candidate_cap_and_kept_ratio():
    """상한을 넘으면 정답을 모르는 채로 줄이고 목록 보존 비율을 기록한다 (docs/02 §3).

    `kept_ratio`는 목록 보존 비율이다. 정답 포함률은 라벨과 대조해 오프라인으로 잰다.
    """
    crowd = [obj(f"o{index}", (140 + 30 * index, -300 + 70 * index, -80)) for index in range(10)]
    request = harness().build_request(observation(objects=crowd), None, None)
    accounting = request["harness"]["accounting"]

    assert len(request["request"]["candidates"]["q_main"]) <= CANDIDATES["max"]
    assert accounting["kept"] < accounting["feasible"]
    assert accounting["capped"] is True
    assert "inclusion_rate" not in accounting
    assert accounting["kept_ratio"] == pytest.approx(accounting["kept"] / accounting["feasible"])
    assert accounting["dropped"]["cap"] == accounting["feasible"] - accounting["kept"]
    assert set(accounting["spread"]) == {"targets", "functions", "approaches", "destinations"}
    for dimension in accounting["spread"].values():
        assert 0 < dimension["kept"] <= dimension["feasible"]


def test_the_cap_keeps_a_spread_of_targets():
    """상한 안에서 대상 다양성을 지킨다 — 한 물체가 목록을 독식하지 않는다."""
    crowd = [obj(f"o{index}", (140 + 30 * index, -300 + 70 * index, -80)) for index in range(10)]
    request = harness().build_request(observation(objects=crowd), None, None)
    targets = {key.split(":")[1] for key in keys_of(request) if ":" in key}
    assert len(targets) >= 6


def crowded_scene(**over) -> dict:
    """docs/10 I3의 장면: 보이는 물체 8개, 영역 3개, 지시는 o7 → z2."""
    objects = [
        obj(f"o{i}", (100 + (i % 4) * 120, -180 + (i // 4) * 360, -80), colour=f"c{i}")
        for i in range(8)
    ]
    zones = [
        {"id": f"z{j}", "desc": f"zone{j}", "bounds_mm": [-150, j * 100, 100, j * 100 + 80]}
        for j in range(3)
    ]
    return observation(
        objects=objects,
        zones=zones,
        instruction={"version": 1, "t_ms": 0, "text": "c7 상자를 zone2로 옮겨라"},
        **over,
    )


def test_the_cap_keeps_every_function_and_destination():
    """docs/10 I3: 상한이 기능·목적지를 통째로 지우면 안 된다 — 각 차원의 값이 하나는 남는다."""
    request = harness().build_request(crowded_scene(), None, None)
    accounting = request["harness"]["accounting"]
    assert accounting["capped"] is True
    parts = [key.split(":") for key in keys_of(request) if ":" in key]

    feasible_functions = {"grasp", "push"}
    assert {part[0] for part in parts} == feasible_functions
    assert {part[3] for part in parts if part[0] == "grasp"} == {"z0", "z1", "z2"}
    assert {part[2] for part in parts if part[0] == "push"} == set(CANDIDATES["push_directions"])
    assert len({part[1] for part in parts}) == 8
    spread = accounting["spread"]
    assert spread["functions"]["kept"] == spread["functions"]["feasible"]
    assert spread["destinations"]["kept"] == spread["destinations"]["feasible"]


def test_the_accounting_counts_destinations_per_target():
    """상한의 붕괴는 **대상별**로 일어난다(3c-1 보고서 E1 미완료의 원인): 전체로는 모든 목적지가 남아도 한 대상의
    목적지가 하나만 남을 수 있다. 회계는 대상마다 남은/실행 가능한 목적지 수를 적는다."""
    request = harness().build_request(crowded_scene(), None, None)
    destinations = request["harness"]["accounting"]["spread"]["destinations"]
    by_target = destinations["by_target"]
    parts = [key.split(":") for key in keys_of(request) if ":" in key]
    assert set(by_target) == {f"o{i}" for i in range(8)}
    for target, entry in by_target.items():
        kept = {part[3] for part in parts if part[1] == target and part[0] == "grasp"}
        assert entry["kept"] == len(kept)
        assert entry["feasible"] == 3  # 세 영역 모두 실행 가능하다
        assert entry["kept"] <= entry["feasible"]
    collapsed = [target for target, entry in by_target.items() if entry["kept"] < entry["feasible"]]
    assert collapsed, "이 장면에서는 상한이 대상별 목적지를 줄인다"
    assert destinations["targets_collapsed"] == len(collapsed)

    # 상한에 걸리지 않으면 대상마다 목적지가 온전하다.
    request = harness().build_request(observation(), None, None)
    destinations = request["harness"]["accounting"]["spread"]["destinations"]
    assert destinations["targets_collapsed"] == 0
    assert all(entry["kept"] == entry["feasible"] for entry in destinations["by_target"].values())


def test_the_cap_reserves_the_committed_candidate():
    """현재 commitment의 후보는 실행 가능하기만 하면 상한과 무관하게 남는다."""
    key = "grasp:o7:top:z2:slow"
    hrn = harness()
    without = hrn.build_request(crowded_scene(), None, None)
    commitment = {
        "action_ref": candidate_id(key), "key": key, "phase": "approach", "held_ticks": 2,
        "last_switch_tick": 0, "goal_version": 1, "challenger": None, "challenger_ticks": 0,
        "stop_ticks": 0,
    }
    with_commitment = hrn.build_request(crowded_scene(tick=1, sim_time_ms=PERIOD_MS), None, commitment)
    assert key in keys_of(with_commitment)
    assert with_commitment["request"]["commitment"]["action_ref"] == candidate_id(key)
    assert with_commitment["harness"]["commitment_invalid"] is False
    assert len(with_commitment["request"]["candidates"]["q_main"]) <= CANDIDATES["max"]
    # 예약은 답을 모르는 채로 한다: commitment 없이 만든 목록과 같은 크기다.
    assert len(keys_of(with_commitment)) == len(keys_of(without))


def test_pruning_is_deterministic_and_answer_agnostic():
    first = harness().build_request(crowded_scene(), None, None)
    second = harness().build_request(crowded_scene(), None, None)
    assert keys_of(first) == keys_of(second)


def test_candidate_generation_does_not_use_hidden_truth():
    """가려진 물체의 참값이 달라져도 후보와 파생 값은 그대로다 (docs/08 §3.2).

    두 하네스에 같은 첫 관측을 주고, 둘째 하네스에만 "가려진 동안 물체가 옮겨진" 참값을
    준다. 요청이 한 글자도 달라지면 안 된다 — 달라진다면 하네스가 근거를 본 것이다.
    """
    first, second = harness(), harness()
    first.build_request(observation(), None, None)
    second.build_request(observation(), None, None)

    hidden = observation(tick=2, sim_time_ms=200)
    hidden["objects"][1].update(visible=False, visible_ratio=0.1)
    moved = copy.deepcopy(hidden)
    moved["objects"][1]["pos_mm"] = [-900, 900, -80]

    assert second.build_request(moved, None, None) == first.build_request(hidden, None, None)


def test_derived_values_are_attached_to_every_joint_candidate():
    request = harness().build_request(observation(), None, None)
    entry = candidate_for(request, "grasp:o0:top:zoneL:slow")
    assert "reach ok" in entry["derived"]
    assert "clr" in entry["derived"] and "d " in entry["derived"]
    geometry = request["harness"]["candidates"][entry["id"]]
    assert geometry["reach_ok"] is True
    assert geometry["distance_mm"] > 0
    assert geometry["target_ref"] == "o0"


def test_path_candidates_are_direct_retreat_and_hold_when_nothing_blocks():
    request = harness().build_request(observation(), None, None)
    kinds = [entry["kind"] for entry in request["request"]["candidates"]["q_path"]]
    assert kinds == ["direct", "retreat", "hold"]


def blocked_scene() -> dict:
    """말단과 목표 사이에 물체 하나가 놓인 장면. 말단이 테이블 가까이 있어 넘어갈 수 없다."""
    return observation(
        objects=[obj("o0", (400, 0, -80)), obj("o5", (200, 0, -80), colour="blue")],
        robot={
            "ee_pos_mm": [0, 0, -60],
            "ee_quat": [0.0, 0.0, 0.0, 1.0],
            "gripper_mm": 80,
            "holding": None,
            "contact_force_n": 0.0,
            "speed_mm_s": 0,
        },
    )


BLOCKED = "grasp:o0:top:zoneL:slow"


def test_the_local_planner_produces_real_waypoints_around_a_blocker():
    """`via`는 플래너가 실제로 만든 경유점이다 — 컨트롤러는 좌표 없는 via를 거절한다."""
    blocked = blocked_scene()
    commitment = {"action_ref": candidate_id(BLOCKED), "key": BLOCKED, "phase": "approach"}
    request = harness().build_request(blocked, None, commitment)

    vias = [entry for entry in request["request"]["candidates"]["q_path"] if entry["kind"] == "via"]
    assert 0 < len(vias) <= PLANNER["max_waypoints"]
    waypoints = request["harness"]["waypoints"]
    for entry in vias:
        assert entry["ref"] in waypoints
        assert len(waypoints[entry["ref"]]["pos_mm"]) == 3
    # 경유점은 상태에도 있어야 모델이 참조를 풀 수 있다.
    refs = {item.get("waypoint") for item in request["request"]["state"]["derived"]}
    assert {entry["ref"] for entry in vias} <= refs


def test_blocked_direct_path_is_reported_in_the_candidate_values():
    request = harness().build_request(blocked_scene(), None, None)
    geometry = request["harness"]["candidates"][candidate_for(request, BLOCKED)["id"]]
    assert geometry["path_clear"] is False
    assert geometry["blocker"] == "o5"
    assert "path blocked" in candidate_for(request, BLOCKED)["derived"]


def test_path_clear_describes_the_segment_of_the_candidates_phase():
    """`path_clear`는 그 후보의 국면에서 **실제로 명령할 구간**을 말한다 (docs/08 §3.2 `derived[]`).

    접근 지점까지는 막혔어도, 이미 파지 국면에 들어온 후보의 구간(말단→파지점)은 비어 있다.
    """
    scene = blocked_scene()
    scene["robot"]["ee_pos_mm"] = [400, 0, 20]  # 대상 바로 위, 파지 지점 근처
    request = harness().build_request(scene, None, None)
    geometry = request["harness"]["candidates"][candidate_id(BLOCKED)]
    assert geometry["phase"] == "grasp"
    assert geometry["path_clear"] is True and geometry["blocker"] is None
    assert geometry["target_mm"] == geometry["action_mm"]


def test_aux_questions_reference_the_tick_start_commitment():
    hrn = harness()
    request = hrn.build_request(observation(), None, None)
    hold = candidate_for(request, "hold")["id"]
    # commitment가 없는 틱은 hold 기준으로 묻는다 (docs/08 §4).
    assert {entry["action_ref"] for entry in request["request"]["candidates"]["q_path"]} == {hold}

    grasp = candidate_for(request, "grasp:o0:top:zoneL:slow")["id"]
    committed = hrn.build_request(
        observation(),
        None,
        {"action_ref": grasp, "key": "grasp:o0:top:zoneL:slow", "phase": "approach"},
    )
    assert {entry["action_ref"] for entry in committed["request"]["candidates"]["q_path"]} == {grasp}
    assert committed["request"]["commitment"]["action_ref"] == grasp


def test_the_request_commitment_carries_only_model_visible_fields():
    """도전자 카운터 같은 하네스 장부는 모델 입력에 넣지 않는다."""
    hrn = harness()
    first = hrn.build_request(observation(), None, None)
    grasp = candidate_for(first, "grasp:o0:top:zoneL:slow")["id"]
    commitment = {
        "action_ref": grasp,
        "key": "grasp:o0:top:zoneL:slow",
        "phase": "approach",
        "held_ticks": 4,
        "last_switch_tick": 1,
        "challenger": "c-other",
        "challenger_ticks": 1,
        "goal_version": 1,
        "stop_ticks": 0,
    }
    request = hrn.build_request(observation(), None, commitment)
    assert set(request["request"]["commitment"]) == {
        "action_ref",
        "key",
        "phase",
        "held_ticks",
        "last_switch_tick",
    }


def test_exec_history_is_a_short_line_of_what_actually_ran():
    hrn = harness()
    previous = {
        "adopted": {
            "main": "c1",
            "phase": "approach",
            "path": "p0",
            "speed": 2,
            "force": 0,
            "gripper": "open",
            "stop": False,
        },
        "ack": {"seq": 12, "applied": True, "gripper_event": None},
        "gate": "none",
    }
    request = hrn.build_request(observation(), previous, None)
    history = request["request"]["exec_history"]
    assert history.startswith("main=c1 phase=approach")
    assert "stop=0" in history and "ack=ok" in history and "gate=none" in history
    assert len(history.split()) <= 12
    assert hrn.build_request(observation(), None, None)["request"]["exec_history"] == "none"


def test_a_rejected_command_is_reported_in_the_exec_history():
    previous = {
        "adopted": {"main": "c1", "phase": "approach", "path": "p0", "speed": 2, "force": 0,
                    "gripper": "open", "stop": False},
        "ack": {"seq": 12, "applied": False, "reason": "unreachable", "rejected": True},
    }
    history = harness().build_request(observation(), previous, None)["request"]["exec_history"]
    assert "ack=unreachable" in history


def test_the_state_never_carries_a_forbidden_field_name():
    request = harness().build_request(observation(), None, None)
    served = model_input(stream(request))
    state = served["ticks"][0]["request"]["state"]
    assert "ack" not in state["exec"]
    assert state["image"] == [] and state["geom"] == []


def test_tick_header_carries_source_wise_ages():
    hrn = harness()
    hrn.build_request(observation(), None, None)
    later = observation(tick=3, sim_time_ms=300)
    request = hrn.build_request(later, None, None)
    assert set(request["obs_age_ms"]) == {"geom", "proprio"}
    assert request["obs_age_ms"]["geom"] >= request["obs_age_ms"]["proprio"]
    assert request["t"] == 3 and request["sim_ms"] == 300


def test_aux_candidates_follow_the_question_set():
    """정적 후보를 가진 질문은 틱마다 다시 싣지 않는다 (계약이 세트에서 푼다)."""
    request = harness().build_request(observation(), None, None)
    assert set(request["request"]["candidates"]) == {"q_main", "q_path"}
    assert set(AUX_QUESTIONS) - {"q_path"} <= set(QUESTION_SET_V0)


def test_harness_version_is_reported():
    request = harness().build_request(observation(), None, None)
    assert request["harness"]["version"] == HARNESS_VERSION == CONFIG["version"]


def test_the_harness_reads_lifetime_numbers_from_the_controller_config():
    hrn = harness()
    assert hrn.lease_ms == CONTROLLER["lifetime"]["lease_ms"]
    assert hrn.observation_deadline_ms == CONTROLLER["lifetime"]["observation_deadline_ms"]
    assert hrn.speed_levels_m_s == CONTROLLER["speed_levels_m_s"]
    assert hrn.force_level_names == list(CONTROLLER["force_levels"])


def test_controller_accepts_a_command_for_every_candidate():
    """하네스가 만든 목표는 실행기의 도달·충돌 검사를 통과한다 (docs/08 §6 "거절")."""
    request = harness().build_request(observation(), None, None)
    ctrl = Controller.from_config_path(CONTROLLER_CONFIG)
    ctrl.reset(ee_pos_mm=[0, 0, 200], ee_quat=[0.0, 0.0, 0.0, 1.0], gripper_mm=80, now_ms=0)
    for seq, (candidate_id, geometry) in enumerate(request["harness"]["candidates"].items(), start=1):
        if geometry["function"] is None:
            continue
        for point in ("approach_mm", "action_mm"):
            command = {
                "seq": seq * 10 + (point == "action_mm"),
                "observed_at": 0,
                "issued_at": 0,
                "action_ref": candidate_id,
                "phase": "approach",
                "path": {"kind": "direct", "target_ref": geometry["target_ref"],
                         "target_mm": geometry[point]},
                "speed_level": 1,
                "force_level": "avoid",
                "gripper": "open",
                "stop": False,
            }
            ack = ctrl.apply(command, now_ms=0)
            assert ack["applied"] is True, (candidate_id, point, ack["reason"])


# --------------------------------------------------------------------------
# 조합 규칙 v0 (docs/08 §5)
# --------------------------------------------------------------------------


def probabilities(**by_key: float) -> dict[str, float]:
    """의미 키로 쓴 `q_main` 분포를 후보 id 분포로 옮긴다."""
    return {candidate_id(key.replace("__", ":")): value for key, value in by_key.items()}


def answers(main: dict[str, float] | None = None, **over) -> dict:
    """모델 출력 형식 (D0 fixture의 `model_output`과 같은 모양)."""
    results: dict = {}
    if main is not None:
        results["q_main"] = main
    results.update(over)
    return results


def step(
    hrn: RobotHarness,
    scene: dict,
    results: dict,
    commitment: dict | None = None,
    *,
    now_ms: int | None = None,
    exec_history: dict | None = None,
) -> tuple[dict, dict]:
    request = hrn.build_request(scene, exec_history, commitment)
    now = int(scene["sim_time_ms"] if now_ms is None else now_ms)
    return request, hrn.compose(request, results, commitment, now)


def committed(hrn: RobotHarness, key: str, scene: dict | None = None, **over) -> dict:
    """그 행동에 이미 commitment가 잡힌 상태를 만든다."""
    scene = scene if scene is not None else observation()
    request = hrn.build_request(scene, None, None)
    geometry = request["harness"]["candidates"][candidate_id(key)]
    target = next(item for item in request["request"]["state"]["objects"] if item["id"] == geometry["target_ref"])
    commitment = {
        "action_ref": candidate_id(key),
        "key": key,
        "phase": geometry["phase"],
        "held_ticks": 3,
        "last_switch_tick": 0,
        "goal_version": 1,
        "challenger": None,
        "challenger_ticks": 0,
        "stop_ticks": 0,
        "start_pose_mm": list(target["pose_mm"]),
        "start_clearance_mm": geometry["clearance_mm"],
    }
    commitment.update(over)
    return commitment


GRASP = "grasp:o0:top:zoneL:slow"
OTHER = "grasp:o1:top:zoneL:slow"
THIRD = "grasp:o2:top:zoneL:slow"


# -- 0. 유효성 검사 ---------------------------------------------------------


@pytest.mark.parametrize(
    ("meta", "reason"),
    [
        ({"observed_at": -10_000}, "observation_late"),
        ({"goal_version": 99}, "goal_version_mismatch"),
        ({"candidate_set_version": "cs-stale"}, "candidate_set_version_mismatch"),
        ({"seq": 0}, "seq_regression"),
    ],
)
def test_an_invalid_response_is_discarded_and_recorded(meta, reason):
    hrn = harness()
    commitment = committed(hrn, GRASP)
    _, out = step(hrn, observation(), answers(probabilities(**{OTHER.replace(":", "__"): 1.0}), meta=meta), commitment)

    assert out["command"] is None
    assert out["adopted"] is None
    assert out["commitment"] == commitment  # 폐기는 상태를 바꾸지 않는다
    assert [record["reason"] for record in out["records"]] == [reason]
    assert count_records(out["records"]) == {"discarded": 1}


def test_a_response_that_echoes_the_request_passes_the_validity_check():
    hrn = harness()
    request = hrn.build_request(observation(), None, None)
    meta = {
        "observed_at": request["observed_at_ms"],
        "goal_version": request["harness"]["goal_version"],
        "candidate_set_version": request["harness"]["candidate_set_version"],
        "seq": request["harness"]["seq"],
    }
    out = hrn.compose(request, answers(probabilities(grasp__o0__top__zoneL__slow=1.0), meta=meta), None, 0)
    assert out["command"] is not None


# -- 1. 반사·정지 -----------------------------------------------------------


def test_stop_bypasses_every_other_answer():
    """정지 틱에는 다른 답을 적용하지 않는다 (docs/08 §5.1)."""
    hrn = harness()
    commitment = committed(hrn, GRASP)
    _, out = step(
        hrn,
        observation(),
        answers(
            probabilities(**{OTHER.replace(":", "__"): 1.0}),
            q_stop=0.99,
            q_speed={"3": 1.0},
            q_gripper={"closed": 1.0},
        ),
        commitment,
    )

    assert out["command"]["stop"] is True
    assert out["command"]["speed_level"] == 0
    assert out["command"]["path"]["kind"] == "hold"
    assert out["command"]["gripper"] == "open"  # 현재 상태 그대로
    assert out["adopted"]["main"] == commitment["action_ref"]  # 전환하지 않는다
    assert out["adopted"]["stop"] is True
    assert "stop" in count_records(out["records"])


def test_an_executor_reflex_is_a_stop_even_without_the_model_answer():
    hrn = harness()
    scene = observation(events=[{"kind": "reflex_stop", "sim_ms": 0, "force_n": 40.0}])
    _, out = step(hrn, scene, answers(probabilities(grasp__o0__top__zoneL__slow=1.0)), None)
    assert out["command"]["stop"] is True
    assert [record["cause"] for record in out["records"] if record["kind"] == "stop"] == ["reflex"]


def test_stop_keeps_the_commitment_for_less_than_m_ticks_then_releases():
    hrn = harness()
    commitment = committed(hrn, GRASP)
    for tick in range(1, M):
        _, out = step(hrn, observation(tick=tick, sim_time_ms=tick * PERIOD_MS), answers(q_stop=1.0), commitment)
        assert out["commitment"] is not None
        assert out["commitment"]["stop_ticks"] == tick
        commitment = out["commitment"]

    _, out = step(hrn, observation(tick=M, sim_time_ms=M * PERIOD_MS), answers(q_stop=1.0), commitment)
    assert out["commitment"] is None
    assert [record["reason"] for record in out["records"] if record["kind"] == "release"] == ["stop"]


# -- 2. 게이팅 --------------------------------------------------------------


@pytest.mark.parametrize(
    ("results", "gate", "key"),
    [
        (answers(q_done=1.0, q_instr=0.0, q_observe=1.0), "done", "hold"),
        (answers(q_done=0.0, q_instr=0.0, q_observe=1.0), "instr", "replan"),
        (answers(q_done=0.0, q_instr=1.0, q_observe=1.0), "observe", "observe"),
    ],
)
def test_gating_follows_the_documented_precedence(results, gate, key):
    hrn = harness()
    commitment = committed(hrn, GRASP)
    _, out = step(hrn, observation(), {**results, **answers(probabilities(grasp__o0__top__zoneL__slow=1.0))}, commitment)

    assert out["gate"] == gate
    assert out["adopted"]["main"] == candidate_id(key)
    assert out["commitment"] is None  # 게이팅이 발동하면 commitment를 해제한다
    assert out["switch"] is True      # 즉시 전환된다
    assert out["adopted"]["speed"] == 0


def test_observe_while_carrying_holds_first():
    """운반 중이면 보류 후 관측한다 (docs/08 §5.2)."""
    hrn = harness()
    scene = observation()
    scene["robot"]["holding"] = "o0"
    _, out = step(hrn, scene, answers(q_observe=1.0), None)
    assert out["adopted"]["main"] == candidate_id("hold")
    assert out["gate"] == "observe"


def test_gating_discards_the_aux_answers():
    hrn = harness()
    _, out = step(hrn, observation(), answers(q_done=1.0, q_speed={"3": 1.0}, q_path={"p0": 1.0}), committed(hrn, GRASP))
    assert out["adopted"]["speed"] == 0
    assert out["adopted"]["path"] == "ph"
    assert "aux_discarded" in count_records(out["records"])


# -- 3. 주 결정과 결정 유지 -------------------------------------------------


def test_a_challenger_below_delta_never_switches():
    hrn = harness()
    commitment = committed(hrn, GRASP)
    results = answers(probabilities(**{
        GRASP.replace(":", "__"): 0.40,
        OTHER.replace(":", "__"): 0.40 + DELTA / 2,
    }))
    for tick in range(6):
        _, out = step(hrn, observation(tick=tick, sim_time_ms=tick * PERIOD_MS), results, commitment)
        commitment = out["commitment"]
        assert out["switch"] is False
        assert commitment["action_ref"] == candidate_id(GRASP)
    assert commitment["held_ticks"] > 3


def test_the_same_challenger_over_delta_switches_after_m_ticks():
    hrn = harness()
    commitment = committed(hrn, GRASP)
    results = answers(probabilities(**{
        GRASP.replace(":", "__"): 0.30,
        OTHER.replace(":", "__"): 0.30 + DELTA + 0.01,
    }))
    for tick in range(M - 1):
        _, out = step(hrn, observation(tick=tick, sim_time_ms=tick * PERIOD_MS), results, commitment)
        assert out["switch"] is False, "δ를 넘어도 m 틱 전에는 바뀌지 않는다"
        assert out["commitment"]["challenger_ticks"] == tick + 1
        commitment = out["commitment"]

    _, out = step(hrn, observation(tick=M, sim_time_ms=M * PERIOD_MS), results, commitment)
    assert out["switch"] is True
    assert out["commitment"]["action_ref"] == candidate_id(OTHER)
    assert out["commitment"]["challenger"] is None
    assert out["commitment"]["held_ticks"] == 0


def test_a_changed_challenger_resets_the_counter():
    hrn = harness()
    commitment = committed(hrn, GRASP)
    first = answers(probabilities(**{GRASP.replace(":", "__"): 0.2, OTHER.replace(":", "__"): 0.9}))
    second = answers(probabilities(**{GRASP.replace(":", "__"): 0.2, THIRD.replace(":", "__"): 0.9}))

    _, out = step(hrn, observation(), first, commitment)
    assert out["commitment"]["challenger_ticks"] == 1

    _, out = step(hrn, observation(tick=1, sim_time_ms=PERIOD_MS), second, out["commitment"])
    assert out["switch"] is False, "도전자가 바뀌면 카운터가 초기화된다"
    assert out["commitment"]["challenger"] == candidate_id(THIRD)
    assert out["commitment"]["challenger_ticks"] == 1
    assert "challenger_reset" in count_records(out["records"])


def test_a_goal_version_change_releases_the_commitment():
    hrn = harness()
    commitment = committed(hrn, GRASP)
    scene = observation(instruction={"version": 2, "t_ms": 5000, "text": "blue 상자를 왼쪽 정리 영역으로 옮겨라"})
    _, out = step(hrn, scene, answers(probabilities(**{GRASP.replace(":", "__"): 1.0})), commitment)

    assert [r["reason"] for r in out["records"] if r["kind"] == "release"] == ["goal_version"]
    assert out["switch"] is True
    assert out["commitment"]["held_ticks"] == 0


def test_an_invalidated_commitment_is_released():
    """후보 목록에서 사라진 행동은 무효다 (대상의 기하가 허용치 너머로 늙은 경우)."""
    hrn = harness()
    commitment = committed(hrn, GRASP)
    stale_ms = CANDIDATES["max_geometry_age_ms"] + PERIOD_MS
    gone = observation(tick=stale_ms // PERIOD_MS, sim_time_ms=stale_ms)
    gone["objects"][0].update(visible=False, visible_ratio=0.0)
    request, out = step(hrn, gone, answers(probabilities(**{OTHER.replace(":", "__"): 1.0})), commitment)

    assert request["request"]["commitment"] is None  # 모델에게도 보이지 않는다
    assert [r["reason"] for r in out["records"] if r["kind"] == "release"] == ["invalidated"]
    assert out["commitment"]["action_ref"] == candidate_id(OTHER)


def test_a_completed_action_releases_the_commitment():
    hrn = harness()
    scene = observation()
    scene["objects"][0]["pos_mm"] = [30, 240, -80]  # 목적지 영역 안 (zoneL 중심)
    commitment = committed(hrn, GRASP, scene)
    _, out = step(hrn, scene, answers(probabilities(**{GRASP.replace(":", "__"): 1.0})), commitment)
    assert [r["reason"] for r in out["records"] if r["kind"] == "release"] == ["completed"]


def test_without_a_commitment_the_top_candidate_is_adopted_at_once():
    hrn = harness()
    _, out = step(hrn, observation(), answers(probabilities(**{OTHER.replace(":", "__"): 1.0})), None)
    assert out["switch"] is True
    assert out["commitment"]["action_ref"] == candidate_id(OTHER)
    assert out["commitment"]["key"] == OTHER


def test_a_missing_main_answer_keeps_the_commitment():
    hrn = harness()
    commitment = committed(hrn, GRASP)
    _, out = step(hrn, observation(), answers(), commitment)
    assert out["commitment"]["action_ref"] == commitment["action_ref"]
    assert "missing_answer" in count_records(out["records"])


def failed_history(key: str = GRASP, reason: str = "collision") -> dict:
    """직전 틱에 그 행동이 실패한 실행 이력 (docs/08 §3.3)."""
    return {
        "adopted": {
            "main": candidate_id(key), "phase": "approach", "path": "p0", "speed": 1,
            "force": 0, "gripper": "open", "stop": False,
        },
        "ack": {"seq": 1, "applied": False, "reason": reason, "rejected": True},
    }


def test_a_refused_retry_blocks_the_approach_that_just_failed():
    """`q_retry`가 거짓이면 같은 방식의 후보는 이 틱에 실행하지 않는다 (docs/08 §4)."""
    hrn = harness()
    request = hrn.build_request(observation(), failed_history(), None)
    out = hrn.compose(
        request,
        answers(probabilities(**{GRASP.replace(":", "__"): 0.9}), q_retry=0.0),
        None,
        0,
    )
    assert out["adopted"]["main"] != candidate_id(GRASP)
    assert "retry_blocked" in count_records(out["records"])


def test_an_accepted_retry_leaves_the_same_approach_available():
    hrn = harness()
    request = hrn.build_request(observation(), failed_history(), None)
    out = hrn.compose(
        request,
        answers(probabilities(**{GRASP.replace(":", "__"): 0.9}), q_retry=1.0),
        None,
        0,
    )
    assert out["adopted"]["main"] == candidate_id(GRASP)
    assert "retry_blocked" not in count_records(out["records"])


def test_a_refused_retry_releases_a_commitment_on_the_failed_approach():
    hrn = harness()
    commitment = committed(hrn, GRASP)
    request = hrn.build_request(observation(), failed_history(), commitment)
    out = hrn.compose(
        request,
        answers(probabilities(**{GRASP.replace(":", "__"): 0.9}), q_retry=0.0),
        commitment,
        0,
    )
    assert [r["reason"] for r in out["records"] if r["kind"] == "release"] == ["retry_blocked"]
    assert out["commitment"]["action_ref"] != candidate_id(GRASP)


def test_without_a_failure_the_retry_answer_changes_nothing():
    hrn = harness()
    request = hrn.build_request(observation(), None, None)
    out = hrn.compose(
        request,
        answers(probabilities(**{GRASP.replace(":", "__"): 0.9}), q_retry=0.0),
        None,
        0,
    )
    assert out["adopted"]["main"] == candidate_id(GRASP)


# -- 4~6. 부가 답·그리퍼·명령 ----------------------------------------------


def test_aux_answers_apply_while_the_commitment_holds():
    hrn = harness()
    commitment = committed(hrn, GRASP)
    _, out = step(
        hrn,
        observation(),
        answers(
            probabilities(**{GRASP.replace(":", "__"): 1.0}),
            q_speed={"0": 0.1, "2": 0.9},
            q_force={"0": 0.2, "1": 0.8},
            q_gripper={"open": 0.1, "closed": 0.9},
            q_path={"p0": 0.9, "ph": 0.1},
        ),
        commitment,
    )
    assert out["adopted"]["speed"] == 2
    assert out["adopted"]["force"] == 1
    assert out["adopted"]["gripper"] == "closed"
    assert out["command"]["force_level"] == "light"
    assert out["command"]["speed_level"] == 2


def test_aux_answers_are_discarded_on_the_switch_tick():
    """전환 틱에는 네 답을 버리고 새 후보의 초기 프로파일을 쓴다 (docs/08 §5.4).

    그리퍼도 예외가 아니다(docs/10 I4): 현재 닫혀 있고 이전 commitment 기준의 답이 `open`이면
    명령은 `closed`(현재 상태)이고 실제 실행기도 개방 이벤트를 내지 않는다.
    """
    hrn = harness()
    scene = observation()
    scene["robot"].update(gripper_mm=CONTROLLER["gripper"]["closed_mm"])
    scene["exec"] = {"seq": 3, "gripper": "closed"}
    commitment = committed(hrn, GRASP, scene, challenger=candidate_id(OTHER), challenger_ticks=M - 1)
    _, out = step(
        hrn,
        scene,
        answers(
            probabilities(**{GRASP.replace(":", "__"): 0.2, OTHER.replace(":", "__"): 0.9}),
            q_speed={"3": 1.0},
            q_force={"2": 1.0},
            q_path={"ph": 1.0},
            q_gripper={"open": 1.0},
        ),
        commitment,
    )
    assert out["switch"] is True
    assert out["adopted"]["speed"] == COMPOSE["initial_profile"]["speed_level"]
    assert out["adopted"]["force"] == COMPOSE["initial_profile"]["force_level"]
    assert out["adopted"]["path"] == "p0"
    assert out["adopted"]["gripper"] == out["command"]["gripper"] == "closed"
    assert count_records(out["records"])["aux_discarded"] == 1
    assert "gripper_change" not in count_records(out["records"])

    ctrl = Controller.from_config_path(CONTROLLER_CONFIG)
    ctrl.reset(ee_pos_mm=[0, 0, 200], ee_quat=[0.0, 0.0, 0.0, 1.0],
               gripper_mm=CONTROLLER["gripper"]["closed_mm"], now_ms=0)
    ctrl.observe({"holding": None, "gripper_load_n": 0.0, "target_distance_mm": 0.0, "contact_force_n": 0.0})
    ack = ctrl.apply(out["command"], now_ms=0)
    assert ack["applied"] is True
    assert ack["gripper_event"] is None
    assert ctrl.gripper_desired == "closed"


def test_an_aux_answer_outside_the_candidates_is_recorded_as_missing():
    """후보 밖의 부가 답은 무시하되 `missing_answer`로 적는다 (q_main과 같은 규칙)."""
    hrn = harness()
    commitment = committed(hrn, GRASP)
    _, out = step(
        hrn,
        observation(),
        answers(
            probabilities(**{GRASP.replace(":", "__"): 1.0}),
            q_path={"p9": 1.0},
            q_speed={"7": 1.0},
            q_force={"0": 1.0},
            q_gripper={"open": 1.0},
        ),
        commitment,
    )
    missing = [record["question"] for record in out["records"] if record["kind"] == "missing_answer"]
    assert missing == ["q_path", "q_speed"]
    assert out["adopted"]["speed"] == COMPOSE["initial_profile"]["speed_level"]
    assert out["adopted"]["path"] == "p0"


def test_speed_is_capped_next_to_a_fragile_object():
    hrn = harness()
    scene = observation()
    scene["objects"][2]["pos_mm"] = [340, 40, -80]  # 취약 물체가 대상 바로 옆
    commitment = committed(hrn, GRASP, scene)
    _, out = step(hrn, scene, answers(probabilities(**{GRASP.replace(":", "__"): 1.0}), q_speed={"3": 1.0}), commitment)
    assert out["adopted"]["speed"] == COMPOSE["fragile_speed_cap"]
    assert "speed_cap" in count_records(out["records"])


def test_force_is_capped_next_to_a_forbidden_object():
    hrn = harness()
    scene = observation()
    scene["objects"][1].update(pos_mm=[330, 30, -80], attributes=["forbidden"])
    commitment = committed(hrn, GRASP, scene)
    _, out = step(hrn, scene, answers(probabilities(**{GRASP.replace(":", "__"): 1.0}), q_force={"2": 1.0}), commitment)
    assert out["adopted"]["force"] == 0
    assert "force_cap" in count_records(out["records"])


def test_the_desired_gripper_state_goes_into_the_command_and_readiness_is_the_controllers_job():
    hrn = harness()
    commitment = committed(hrn, GRASP)
    _, out = step(hrn, observation(), answers(probabilities(**{GRASP.replace(":", "__"): 1.0}), q_gripper={"closed": 1.0}), commitment)
    assert out["command"]["gripper"] == "closed"
    assert out["command"]["gripper_ref"] == "o0"
    assert "gripper_change" in count_records(out["records"])


def test_a_pending_gripper_wait_is_recorded():
    hrn = harness()
    scene = observation()
    scene["exec"] = {"seq": 3, "gripper_wait": "readiness", "gripper": "open"}
    commitment = committed(hrn, GRASP, scene)
    _, out = step(hrn, scene, answers(probabilities(**{GRASP.replace(":", "__"): 1.0}), q_gripper={"closed": 1.0}), commitment)
    assert "gripper_wait" in count_records(out["records"])


def test_a_via_answer_carries_the_planner_waypoint_into_the_command():
    hrn = harness()
    scene = blocked_scene()
    commitment = committed(hrn, BLOCKED, scene)
    request, out = step(hrn, scene, answers(probabilities(**{BLOCKED.replace(":", "__"): 1.0}), q_path={"p1": 1.0}), commitment)

    assert out["command"]["path"]["kind"] == "via"
    assert out["command"]["path"]["waypoint_ref"] == "w1"
    assert out["command"]["path"]["waypoint_mm"] == request["harness"]["waypoints"]["w1"]["pos_mm"]
    assert out["adopted"]["path"] == "p1"

    # 실제 실행기는 그 경유점으로 움직인다.
    ctrl = Controller.from_config_path(CONTROLLER_CONFIG)
    start = scene["robot"]["ee_pos_mm"]
    ctrl.reset(ee_pos_mm=start, ee_quat=[0.0, 0.0, 0.0, 1.0], gripper_mm=80, now_ms=0)
    ctrl.observe({"ee_pos_mm": list(start), "nearest_obstacle_mm": 300.0, "holding": None})
    # 하네스는 10Hz로 다시 발행한다. 여기서는 lease를 길게 줘 한 명령으로 도착까지 본다.
    ack = ctrl.apply({**out["command"], "lease_until": 5000}, now_ms=0)
    assert ack["applied"] is True and ack["path"] == "via" and ack["executor"] == "MOVE_EE"
    setpoint = None
    for step_ms in range(20, 3000, 20):
        setpoint = ctrl.advance(step_ms)
    waypoint = out["command"]["path"]["waypoint_mm"]
    assert setpoint["ee_pos_mm"] == pytest.approx(waypoint, abs=1.0)


def moved_target_scene(hrn: RobotHarness, *, key: str = GRASP) -> tuple[dict, dict]:
    """대상이 관측된 변위로 '이동 대상'이 된 뒤, 기하가 갱신되지 않은 틱.

    기하 갱신 주기(200ms)마다 자세를 두 번 보고(0ms, 200ms — 40mm 이동), 390ms의 틱은 아직 새
    기하가 없어 대상 기하의 나이가 190ms다 (docs/10 I6의 값).
    """
    hrn.build_request(observation(), None, None)
    period = CONFIG["perception"]["geom_period_ms"]
    shifted = observation(tick=period // PERIOD_MS, sim_time_ms=period)
    shifted["objects"][0]["pos_mm"] = [340, 0, -80]
    hrn.build_request(shifted, None, None)

    request_ms = 2 * period - 10
    scene = observation(tick=request_ms // PERIOD_MS, sim_time_ms=request_ms)
    scene["objects"][0]["pos_mm"] = [340, 0, -80]
    commitment = {
        "action_ref": candidate_id(key), "key": key, "phase": "approach", "held_ticks": 2,
        "last_switch_tick": 0, "goal_version": 1, "challenger": None, "challenger_ticks": 0,
        "stop_ticks": 0, "start_pose_mm": [340, 0, -80],
    }
    return scene, commitment


def test_stale_geometry_sends_the_chosen_candidate_to_the_observe_branch():
    """선택된 후보의 기하가 허용치를 넘으면 적용하지 않고 관측 분기로 보낸다 (docs/08 §5.0).

    관측된 변위로 '이동 대상'이 된 물체의 허용치는 200ms다. 요청 시점의 나이는 190ms지만
    답이 100ms 뒤에 오면 적용 시각의 나이는 290ms다 (docs/10 I6).
    """
    hrn = harness()
    scene, commitment = moved_target_scene(hrn)
    request = hrn.build_request(scene, None, commitment)
    geometry = request["harness"]["candidates"][candidate_id(GRASP)]
    assert geometry["geometry_age_ms"] == 190 and geometry["moving"] is True

    out = hrn.compose(request, answers(probabilities(**{GRASP.replace(":", "__"): 1.0})), commitment, scene["sim_time_ms"] + 100)
    assert [r for r in out["records"] if r["kind"] == "geometry_age"][0]["age_ms"] == 290
    assert out["gate"] == "observe"
    assert out["adopted"]["main"] == candidate_id("observe")
    assert out["commitment"] is None
    assert count_records(out["records"]).get("switch") is None, "관측으로 보낸 틱에 전환 기록이 남았다"
    assert count_records(out["records"])["release"] == 1

    prompt = hrn.compose(request, answers(probabilities(**{GRASP.replace(":", "__"): 1.0})), commitment, scene["sim_time_ms"])
    assert prompt["gate"] is None and prompt["command"]["geometry_observed_at"] == 200
    assert prompt["command"]["geometry_age_ms"] == 190


def test_the_real_controller_measures_geometry_age_at_apply_time():
    """docs/10 I6: 요청 시 190ms였던 기하는 100ms 뒤에 적용하면 290ms다 — 실행기가 관측을 요청한다."""
    hrn = harness()
    scene, commitment = moved_target_scene(hrn)
    request = hrn.build_request(scene, None, commitment)
    out = hrn.compose(request, answers(probabilities(**{GRASP.replace(":", "__"): 1.0})), commitment, scene["sim_time_ms"])
    assert out["command"]["target_moving"] is True

    ctrl = Controller.from_config_path(CONTROLLER_CONFIG)
    ctrl.reset(ee_pos_mm=[0, 0, 200], ee_quat=[0.0, 0.0, 0.0, 1.0], gripper_mm=80, now_ms=0)
    ctrl.observe({"holding": None})
    late = ctrl.apply(out["command"], now_ms=scene["sim_time_ms"] + 100)
    assert late["applied"] is False
    assert late["reason"] == "geometry_age" and late["request_observation"] is True

    ctrl = Controller.from_config_path(CONTROLLER_CONFIG)
    ctrl.reset(ee_pos_mm=[0, 0, 200], ee_quat=[0.0, 0.0, 0.0, 1.0], gripper_mm=80, now_ms=0)
    ctrl.observe({"holding": None})
    assert ctrl.apply(out["command"], now_ms=scene["sim_time_ms"])["applied"] is True


def test_the_grasp_phase_is_exempt_from_the_geometry_age_gate():
    """파지·놓기 국면과 파지 중에는 기하 나이 대신 readiness가 시점을 정한다 (docs/08 §5.0)."""
    hrn = harness()
    scene, commitment = moved_target_scene(hrn)
    scene["robot"]["ee_pos_mm"] = [340, 0, -30]  # 파지 지점 바로 위
    request = hrn.build_request(scene, None, commitment)
    geometry = request["harness"]["candidates"][candidate_id(GRASP)]
    assert geometry["phase"] == "grasp" and geometry["geometry_age_ms"] == 190

    out = hrn.compose(request, answers(probabilities(**{GRASP.replace(":", "__"): 1.0})), commitment, scene["sim_time_ms"] + 100)
    assert out["gate"] is None
    assert "geometry_age" not in count_records(out["records"])
    assert out["command"]["phase"] == "grasp"

    ctrl = Controller.from_config_path(CONTROLLER_CONFIG)
    ctrl.reset(ee_pos_mm=[340, 0, -30], ee_quat=[0.0, 0.0, 0.0, 1.0], gripper_mm=80, now_ms=0)
    ctrl.observe({"holding": None, "target_distance_mm": 10.0})
    ack = ctrl.apply(out["command"], now_ms=scene["sim_time_ms"] + 100)
    assert ack["applied"] is True and ack["request_observation"] is False


def test_a_redirect_to_observe_leaves_no_phantom_switch_records():
    """관측으로 보낸 틱에 주 결정이 남긴 전환·유지 기록이 섞이면 안 된다 (해제는 한 번)."""
    hrn = harness()
    scene, _ = moved_target_scene(hrn)
    request = hrn.build_request(scene, None, None)
    out = hrn.compose(request, answers(probabilities(**{GRASP.replace(":", "__"): 1.0})), None, scene["sim_time_ms"] + 100)
    counts = count_records(out["records"])
    assert out["gate"] == "observe"
    assert "switch" not in counts and "hold" not in counts
    assert "release" not in counts  # commitment가 없었으니 해제할 것도 없다
    assert counts["geometry_age"] == 1 and counts["gate"] == 1 and counts["aux_discarded"] == 1


@pytest.mark.parametrize(("key", "executor"), [("observe", "OBSERVE"), ("replan", "REQUEST_REPLAN"), ("hold", "HOLD")])
def test_a_chosen_observe_or_replan_candidate_reaches_its_executor(key, executor):
    """후보를 고른 것이 게이팅이 아니어도 실행기 대응은 같다 (docs/02 §4의 대응표)."""
    hrn = harness()
    _, out = step(hrn, observation(), answers({candidate_id(key): 1.0}), None)
    assert out["adopted"]["main"] == candidate_id(key)
    assert out["command"]["speed_level"] == 0

    ctrl = Controller.from_config_path(CONTROLLER_CONFIG)
    ctrl.reset(ee_pos_mm=[0, 0, 200], ee_quat=[0.0, 0.0, 0.0, 1.0], gripper_mm=80, now_ms=0)
    ack = ctrl.apply(out["command"], now_ms=0)
    assert ack["applied"] is True
    assert ack["executor"] == executor


def test_the_command_carries_the_documented_contract_fields():
    hrn = harness()
    commitment = committed(hrn, GRASP)
    request, out = step(hrn, observation(), answers(probabilities(**{GRASP.replace(":", "__"): 1.0})), commitment)
    command = out["command"]
    assert set(command) >= {
        "seq", "request_id", "observed_at", "issued_at", "lease_until", "goal_version",
        "candidate_set_version", "action_ref", "phase", "path", "speed_level", "force_level",
        "gripper", "stop", "gripper_ref", "frame", "constraints",
    }
    assert command["lease_until"] == command["issued_at"] + hrn.lease_ms
    assert command["seq"] == request["harness"]["seq"]
    assert command["goal_version"] == 1
    assert command["frame"] == CONFIG["command"]["frame"]


def test_records_count_every_composition_event():
    """대본대로 돌린 여섯 틱의 기록 종류별 건수가 정확히 맞아야 한다 (docs/08 §5 "충돌 건수")."""
    hrn = harness()
    totals: dict[str, int] = {}
    commitment = None
    history = None
    script = [
        # 틱 0: commitment 없음 → 즉시 채택 (switch 1, aux_discarded 1)
        (answers(probabilities(**{GRASP.replace(":", "__"): 1.0})), None),
        # 틱 1: 같은 답 → 유지 (hold 1)
        (answers(probabilities(**{GRASP.replace(":", "__"): 1.0})), None),
        # 틱 2: 도전자 δ 초과 1틱째 → 유지 (hold 1)
        (answers(probabilities(**{GRASP.replace(":", "__"): 0.2, OTHER.replace(":", "__"): 0.8})), None),
        # 틱 3: 같은 도전자 2틱째 → 전환 (switch 1, aux_discarded 1)
        (answers(probabilities(**{GRASP.replace(":", "__"): 0.2, OTHER.replace(":", "__"): 0.8})), None),
        # 틱 4: 직전 실패 + 재시도 거부 → 같은 방식 차단·해제·재채택 (retry_blocked 1, release 1, switch 1, aux_discarded 1)
        (answers(probabilities(**{OTHER.replace(":", "__"): 0.9, THIRD.replace(":", "__"): 0.1}), q_retry=0.0),
         failed_history(OTHER)),
        # 틱 5: 늦은 응답 → 폐기 (discarded 1)
        (answers(probabilities(**{THIRD.replace(":", "__"): 1.0}), meta={"observed_at": -10_000}), None),
    ]
    for tick, (results, forced_history) in enumerate(script):
        scene = observation(tick=tick, sim_time_ms=tick * PERIOD_MS)
        request = hrn.build_request(scene, forced_history or history, commitment)
        out = hrn.compose(request, results, commitment, tick * PERIOD_MS)
        for kind, count in count_records(out["records"]).items():
            totals[kind] = totals.get(kind, 0) + count
        for record in out["records"]:
            assert isinstance(record["kind"], str) and set(record) >= {"kind"}
        commitment = out["commitment"]
        history = {"adopted": out["adopted"], "ack": {"applied": True}, "gate": out["gate"]}

    assert totals["switch"] == 3
    assert totals["hold"] == 2
    assert totals["release"] == 1
    assert totals["retry_blocked"] == 1
    assert totals["discarded"] == 1
    assert totals["aux_discarded"] == 3
    assert "gate" not in totals and "stop" not in totals and "conflict" not in totals


def test_a_commitment_without_a_goal_version_is_released_as_stale():
    """목표 버전을 모르는 commitment는 지금 목표에 대한 것이라고 볼 수 없다 — 해제한다."""
    hrn = harness()
    commitment = committed(hrn, GRASP)
    del commitment["goal_version"]
    _, out = step(hrn, observation(), answers(probabilities(**{GRASP.replace(":", "__"): 1.0})), commitment)
    assert [r["reason"] for r in out["records"] if r["kind"] == "release"] == ["goal_version"]
    assert out["switch"] is True
    assert out["commitment"]["goal_version"] == 1


def test_a_drifted_target_releases_the_commitment():
    """같은 의미 키라도 연속 파라미터가 허용 오차를 넘으면 같은 후보가 아니다 (docs/02 §4).

    대상의 자세가 채택 시점에서 `tolerance.distance_mm`보다 멀어졌거나(외란), 대상 주변의
    여유가 `tolerance.clearance_mm`보다 달라졌으면 commitment를 해제하고 다시 고른다.
    """
    hrn = harness()
    commitment = committed(hrn, GRASP)
    tolerance = COMPOSE["tolerance"]
    period = CONFIG["perception"]["geom_period_ms"]  # 자세는 기하 갱신 틱에만 바뀐다

    # 가장 가까운 이웃(o1)을 도는 접선 방향으로 허용 오차 안만큼 움직인다 — 여유는 그대로다.
    nudged = observation(tick=period // PERIOD_MS, sim_time_ms=period)
    nudged["objects"][0]["pos_mm"] = [341, 19, -80]
    _, out = step(hrn, nudged, answers(probabilities(**{GRASP.replace(":", "__"): 1.0})), commitment)
    assert "release" not in count_records(out["records"])

    shoved = observation(tick=2 * period // PERIOD_MS, sim_time_ms=2 * period)
    shoved["objects"][0]["pos_mm"] = [300 + tolerance["distance_mm"] + 5, 0, -80]
    _, out = step(hrn, shoved, answers(probabilities(**{GRASP.replace(":", "__"): 1.0})), commitment)
    release = [r for r in out["records"] if r["kind"] == "release"]
    assert [r["reason"] for r in release] == ["drifted"]
    assert release[0]["parameter"] == "distance_mm"
    assert out["commitment"]["held_ticks"] == 0  # 다시 채택했다

    crowded = observation(tick=3 * period // PERIOD_MS, sim_time_ms=3 * period)
    crowded["objects"][1]["pos_mm"] = [300, 100, -80]  # 이웃이 대상 옆으로 왔다 → 여유가 줄었다
    _, out = step(hrn, crowded, answers(probabilities(**{GRASP.replace(":", "__"): 1.0})), commitment)
    release = [r for r in out["records"] if r["kind"] == "release"]
    assert [r["reason"] for r in release] == ["drifted"]
    assert release[0]["parameter"] == "clearance_mm"


def test_a_missing_destination_zone_is_a_conflict_not_a_crash():
    """다른 도구의 틱에서 목적지 영역이 없으면 `StopIteration`이 아니라 충돌 기록이다 (docs/10 검토 6)."""
    record = copy.deepcopy(read_jsonl(D0_STREAMS)[0])
    tick = next(item for item in record["ticks"] if item["request"]["state"]["robot"].get("holding"))
    commitment = tick["request"]["commitment"]
    for zone in tick["request"]["state"]["zones"]:
        zone["id"] = "zoneX"
    out = harness().compose(
        tick, answers({commitment["action_ref"]: 1.0}), {**commitment, "goal_version": 1}, tick["sim_ms"]
    )
    assert out["command"] is not None
    assert out["command"]["path"]["kind"] == "hold"
    conflicts = [r for r in out["records"] if r["kind"] == "conflict"]
    assert [r["reason"] for r in conflicts] == ["destination_missing"]


# --------------------------------------------------------------------------
# docs/10 회귀 — 실제로 명령할 구간과 국면별 목표점
# --------------------------------------------------------------------------


def probe_answers(request: dict, key: str, **extra) -> dict:
    """docs/10 probe.py의 답: 규칙 답 위에 `key`를 확정하고 부가 답은 direct·1·회피·open이다."""
    result = rule_judge(request)
    result.update(
        q_main={candidate_id(key): 1.0}, q_done=0.0, q_instr=1.0, q_observe=0.0, q_stop=0.0,
        q_retry=1.0, q_path={"p0": 1.0}, q_speed={"1": 1.0}, q_force={"0": 1.0},
        q_gripper={"open": 1.0},
    )
    result.update(extra)
    return result


def i1_scene(**blocker_over) -> dict:
    """docs/10 I1: 말단 [0,0,200], 대상 o0 [400,0,0], 장애물 o1 [200,0,140]."""
    return observation(objects=[obj("o0", (400, 0, 0)), obj("o1", (200, 0, 140), colour="blue", **blocker_over)])


def controller_at(scene: dict, *, closed: bool = False) -> Controller:
    ctrl = Controller.from_config_path(CONTROLLER_CONFIG)
    robot = scene["robot"]
    ctrl.reset(
        ee_pos_mm=robot["ee_pos_mm"], ee_quat=robot["ee_quat"],
        gripper_mm=CONTROLLER["gripper"]["closed_mm"] if closed else robot["gripper_mm"],
        now_ms=scene["sim_time_ms"],
    )
    ctrl.observe({"ee_pos_mm": list(robot["ee_pos_mm"]), "holding": robot.get("holding"),
                  "gripper_load_n": 0.0, "target_distance_mm": 0.0, "contact_force_n": 0.0,
                  "nearest_obstacle_mm": 300.0})
    return ctrl


def test_a_known_blocker_is_never_sent_as_direct():
    """docs/10 I1: 막힌 direct는 보내지 않는다 — 첫 경유 경로로 바꾸고 이유를 적는다 (docs/08 §5.4, §5.6).

    채택 틱(commitment 없음)에는 요청에 경유점이 없으므로 하네스가 즉석에서 계획한다. 그 경우에도
    `adopted`와 다음 틱의 실행 이력은 **실제로 명령한 경로**(via + 경유점 id·좌표)를 말해야 하고
    (docs/08 §3.3 — 이력은 학습 입력이다), 레코드(`append_tick`)만으로 경유점이 풀려야 한다.
    """
    scene = i1_scene()
    hrn = harness()
    request = hrn.build_request(scene, None, None)
    geometry = request["harness"]["candidates"][candidate_id(GRASP)]
    assert geometry["path_clear"] is False and geometry["blocker"] == "o1"
    assert not [entry for entry in request["request"]["candidates"]["q_path"] if entry["kind"] == "via"]

    # 채택 틱: 초기 프로파일의 direct도 같은 규칙을 따른다.
    out = hrn.compose(request, probe_answers(request, GRASP), None, 0)
    command = out["command"]
    assert command["path"]["kind"] == "via"
    assert command["path"]["waypoint_mm"] is not None
    assert command["path"]["target_mm"] == [400, 0, 92]
    assert command["path"]["waypoint_mm"] != [400, 0, 92]
    conflicts = [r for r in out["records"] if r["kind"] == "conflict"]
    assert conflicts and conflicts[0]["reason"] == "path_blocked" and conflicts[0]["blocker"] == "o1"
    assert conflicts[0]["resolution"] == "via"
    ack = controller_at(scene).apply(command, 0)
    assert ack["applied"] is True and ack["path"] == "via"

    # 채택 결과는 답한 direct(p0)가 아니라 실제로 명령한 경로다.
    adopted = out["adopted"]
    assert adopted["path"] is None  # 이 틱의 경로 후보 어느 것도 아니다
    assert adopted["path_kind"] == "via"
    assert adopted["waypoint"] == {"ref": command["path"]["waypoint_ref"], "pos_mm": command["path"]["waypoint_mm"]}
    assert adopted["waypoint"]["ref"] == "w1"

    # 레코드만으로 경유점이 풀린다 (하네스 블록은 레코드에 가지 않는다).
    record = new_episode("ep-i1", "scene-family-i1", instructions=[scene["instruction"]])
    tick = append_tick(record, request, model_output=probe_answers(request, GRASP), adopted=adopted, ack=ack)
    assert "harness" not in tick
    assert tick["adopted"]["waypoint"]["pos_mm"] == command["path"]["waypoint_mm"]
    finalize(record)

    # 다음 틱의 실행 이력은 경유점 이름과 좌표를 말한다.
    commitment = out["commitment"]
    later = i1_scene(tick=1, sim_time_ms=PERIOD_MS)
    request = hrn.build_request(later, {"adopted": adopted, "ack": ack, "gate": None}, commitment)
    history = request["request"]["exec_history"]
    x, y, z = command["path"]["waypoint_mm"]
    assert f"path=via:w1@{x},{y},{z}" in history, history
    assert "path=p0" not in history
    assert len(history.split()) <= 12

    # 유지 틱: 모델이 direct라고 답해도 마찬가지고, 요청의 경유 경로가 이 후보 것이면 그 id를 적는다.
    out = hrn.compose(request, probe_answers(request, GRASP), commitment, PERIOD_MS)
    assert out["command"]["path"]["kind"] == "via"
    assert out["adopted"]["path"] == "p1"
    assert out["adopted"]["path_kind"] == "via"
    assert out["adopted"]["waypoint"]["ref"] == "w1"
    assert [r["reason"] for r in out["records"] if r["kind"] == "conflict"] == ["path_blocked"]
    following = hrn.build_request(i1_scene(tick=2, sim_time_ms=2 * PERIOD_MS),
                                  {"adopted": out["adopted"], "ack": ack, "gate": None}, out["commitment"])
    assert "path=p1" in following["request"]["exec_history"]


def test_the_exec_history_says_what_the_executor_actually_did_on_an_observe_tick():
    """게이팅 관측 틱의 채택 기록은 hold·속도 0이지만 실행기는 관측 자세로 움직인다 — 이력은 실행된 것을 말한다."""
    hrn = harness()
    scene = observation()
    request = hrn.build_request(scene, None, None)
    out = hrn.compose(request, answers(q_observe=1.0), None, 0)
    assert out["gate"] == "observe"
    assert out["adopted"]["path_kind"] == "observe" and out["adopted"]["path"] is None
    assert out["adopted"]["speed"] == 0  # 게이팅 틱의 부가 답은 버린다

    ctrl = controller_at(scene)
    ack = ctrl.apply(out["command"], 0)
    assert ack["applied"] is True and ack["executor"] == "OBSERVE" and ack["path"] == "observe"
    assert ack["speed_level"] == CONTROLLER["observe"]["speed_level"]

    following = hrn.build_request(observation(tick=1, sim_time_ms=PERIOD_MS),
                                  {"adopted": out["adopted"], "ack": ack, "gate": out["gate"]}, None)
    history = following["request"]["exec_history"]
    assert "path=observe" in history and f"speed={CONTROLLER['observe']['speed_level']}" in history
    assert "gate=observe" in history and "ack=ok" in history

    # 운반 중의 관측은 실제로 hold다.
    carrying = observation()
    carrying["robot"].update(holding="o0", ee_pos_mm=[300, 0, 40], gripper_mm=0)
    carrying["exec"] = {"seq": 1, "gripper": "closed"}
    request = hrn.build_request(carrying, None, None)
    out = hrn.compose(request, answers(q_observe=1.0), None, 0)
    ack = controller_at(carrying, closed=True).apply(out["command"], 0)
    assert ack["path"] == "hold" and ack["speed_level"] == 0
    following = hrn.build_request(observation(tick=1, sim_time_ms=PERIOD_MS),
                                  {"adopted": out["adopted"], "ack": ack, "gate": out["gate"]}, None)
    assert "path=ph" in following["request"]["exec_history"] and "speed=0" in following["request"]["exec_history"]


def test_unsupported_faces_are_counted_not_silently_dropped():
    """앞단이 낸 파지면 중 실행기가 못 쓰는 면(`candidates.faces` 밖)은 회계에 남는다 (실행기 역량이지 물체의 실행 가능성이 아니다)."""
    request = harness().build_request(observation(), None, None)
    accounting = request["harness"]["accounting"]
    faces = {face for entry in request["request"]["state"]["objects"] for face in entry["graspable_faces"]}
    assert "side" in faces and CANDIDATES["faces"] == ["top"]
    zones, profiles = len(request["request"]["state"]["zones"]), len(CANDIDATES["profiles"])
    assert accounting["dropped"]["unsupported_face"] == 3 * zones * profiles
    assert accounting["enumerated"] == accounting["feasible"] + sum(
        accounting["dropped"][reason] for reason in ("unsupported_face", "stale", "unreachable")
    )

    wrist = copy.deepcopy(CONFIG)
    wrist["candidates"]["faces"] = ["top", "side"]
    request = RobotHarness(wrist).build_request(observation(), None, None)
    assert request["harness"]["accounting"]["dropped"]["unsupported_face"] == 0
    assert "grasp:o0:side:zoneL:slow" in keys_of(request)


def test_a_forbidden_blocker_is_rerouted_with_extra_margin():
    """docs/10 I1 (금지 물체): 금지 접촉 물체는 더 큰 여유의 장애물이다 — 경유 경로가 있으면 그리로 간다."""
    scene = i1_scene(attributes=["forbidden"])
    hrn = harness()
    request = hrn.build_request(scene, None, None)
    assert request["request"]["state"]["goal"]["forbidden_contact"] == ["o1"]
    out = hrn.compose(request, probe_answers(request, GRASP), None, 0)
    command = out["command"]

    assert command["path"]["kind"] == "via"
    assert command["constraints"]["forbidden_segment"] is False
    conflicts = [r for r in out["records"] if r["kind"] == "conflict"]
    assert [r["reason"] for r in conflicts] == ["forbidden_reroute"] and conflicts[0]["blocker"] == "o1"
    assert out["adopted"]["path_kind"] == "via" and out["adopted"]["waypoint"]["ref"] == "w1"

    # 경유점의 두 소구간은 금지 물체를 추가 여유만큼 더 멀리 비껴간다.
    from robo_jev.perception.pointworld import circumradius_mm, segment_point_distance_mm

    blocker = next(entry for entry in scene["objects"] if entry["id"] == "o1")
    clearance = circumradius_mm(blocker["obb_mm"]) + PLANNER["margin_mm"] + PLANNER["forbidden_margin_mm"]
    ee = scene["robot"]["ee_pos_mm"]
    waypoint = command["path"]["waypoint_mm"]
    for start, end in ((ee, waypoint), (waypoint, command["path"]["target_mm"])):
        assert segment_point_distance_mm(start, end, blocker["pos_mm"]) >= clearance - 1e-6

    ack = controller_at(scene).apply(command, 0)
    assert ack["applied"] is True and ack["path"] == "via" and ack["stop_transition"] is False


def test_a_target_beside_a_forbidden_object_is_still_reachable_directly():
    """추가 여유는 피할 수 있는 곳에만 적용한다 — 목표점이 금지 물체의 넓힌 구 안이면 기본 여유로 본다.

    최소 간격 85mm의 장면에서 넓힌 여유를 목표점에도 적용하면 금지 물체 곁의 대상은 어떤 경로로도
    닿을 수 없다(E0 seed 17에서 파지 직전에 retreat로 빠지던 원인).
    """
    scene = observation(objects=[
        obj("o0", (300, 0, -80)),
        obj("o1", (300, 120, -80), colour="blue", attributes=["forbidden"]),
    ])
    hrn = harness()
    request = hrn.build_request(scene, None, None)
    geometry = request["harness"]["candidates"][candidate_id(GRASP)]
    assert geometry["path_clear"] is True and geometry["blocker"] is None
    out = hrn.compose(request, probe_answers(request, GRASP), None, 0)
    assert out["command"]["path"]["kind"] == "direct"
    assert out["command"]["constraints"]["forbidden_segment"] is False
    assert not [r for r in out["records"] if r["kind"] == "conflict"]

    # 그립 국면(수직 하강)도 마찬가지다.
    descending = observation(objects=copy.deepcopy(scene["objects"]), tick=1, sim_time_ms=PERIOD_MS)
    descending["robot"]["ee_pos_mm"] = [300, 0, 0]
    request = hrn.build_request(descending, None, out["commitment"])
    assert request["harness"]["candidates"][candidate_id(GRASP)]["phase"] == "grasp"
    out = hrn.compose(request, probe_answers(request, GRASP), out["commitment"], PERIOD_MS)
    assert out["command"]["path"]["kind"] == "direct" and out["command"]["phase"] == "grasp"


def test_a_grasp_point_inside_the_base_sphere_of_a_forbidden_object_still_holds():
    """끝점 예외의 거울 검사 (3b 리뷰 (b)): 예외는 **넓힌** 여유만 벗긴다. 목표점이 금지 물체의 기본
    구(외접 반지름 + margin) 안이면 어떤 경로로도 갈 수 없으므로 hold + `forbidden_segment`이고
    실행기는 정지 전이한다 — 파지점 63.9mm vs 한계 78.1mm."""
    from robo_jev.perception.pointworld import circumradius_mm

    scene = observation(objects=[
        obj("o0", (300, 0, -80)),
        obj("o1", (300, 60, -80), colour="blue", attributes=["forbidden"]),
    ])
    scene["robot"]["ee_pos_mm"] = [300, 0, 0]  # 파지점 80mm 안 → grasp 국면
    hrn = harness()
    request = hrn.build_request(scene, None, None)
    geometry = request["harness"]["candidates"][candidate_id(GRASP)]
    assert geometry["phase"] == "grasp"
    forbidden = next(entry for entry in scene["objects"] if entry["id"] == "o1")
    base_limit = circumradius_mm(forbidden["obb_mm"]) + PLANNER["margin_mm"]
    inside = math.dist(geometry["target_mm"], forbidden["pos_mm"])
    assert inside < base_limit, (inside, base_limit)

    out = hrn.compose(request, probe_answers(request, GRASP), None, 0)
    command = out["command"]
    assert command["path"]["kind"] == "hold"
    assert command["constraints"]["forbidden_segment"] is True
    assert [r["reason"] for r in out["records"] if r["kind"] == "conflict"] == ["forbidden_segment"]
    assert out["adopted"]["path_kind"] == "hold" and out["adopted"]["waypoint"] is None
    ack = controller_at(scene).apply(command, 0)
    assert ack["stop_transition"] is True and ack["reason"] == "transition_collision"


def walled_scene(**blocker_over) -> dict:
    """유일한 직선을 o1이 막고 위·양옆도 막힌 장면 — 경유점이 없다."""
    return observation(objects=[
        obj("o0", (400, 0, 0)),
        obj("o1", (200, 0, 140), colour="blue", **blocker_over),
        obj("o2", (200, 160, 140), colour="green"),
        obj("o3", (200, -160, 140), colour="grey"),
        obj("o4", (200, 0, 300), colour="pink"),
    ])


def test_a_blocked_direct_with_no_detour_holds():
    """경유점도 없으면 hold와 충돌 기록이다 (docs/08 §5.4)."""
    walled = walled_scene()
    hrn = harness()
    request = hrn.build_request(walled, None, None)
    out = hrn.compose(request, probe_answers(request, GRASP), None, 0)
    assert out["command"]["path"]["kind"] == "hold"
    assert out["command"]["path"].get("target_mm") is None
    conflicts = [r for r in out["records"] if r["kind"] == "conflict"]
    assert conflicts[0]["reason"] == "path_blocked" and conflicts[0]["resolution"] == "hold"
    assert out["adopted"]["path"] == "ph" and out["adopted"]["path_kind"] == "hold"
    assert controller_at(walled).apply(out["command"], 0)["executor"] == "HOLD"


def test_a_forbidden_blocker_with_no_route_holds_then_releases_after_m_ticks():
    """경유 경로가 없으면 hold+`forbidden_segment`(실행기 정지 전이)이고, `m` 틱 이어지면 commitment를 풀어
    모델이 다른 행동을 고를 수 있게 한다."""
    hrn = harness()
    scene = walled_scene(attributes=["forbidden"])
    request = hrn.build_request(scene, None, None)
    out = hrn.compose(request, probe_answers(request, GRASP), None, 0)
    command = out["command"]
    assert command["path"]["kind"] == "hold"
    assert command["constraints"]["forbidden_segment"] is True
    assert [r["reason"] for r in out["records"] if r["kind"] == "conflict"] == ["forbidden_segment"]
    assert out["commitment"] is not None and out["commitment"]["forbidden_ticks"] == 1

    ctrl = controller_at(scene)
    ack = ctrl.apply(command, 0)
    assert ack["stop_transition"] is True and ack["reason"] == "transition_collision" and ack["applied"] is False
    assert ctrl.advance(20)["executor"] == "HOLD"

    commitment = out["commitment"]
    for tick in range(1, M):
        request = hrn.build_request(walled_scene(attributes=["forbidden"], tick=tick, sim_time_ms=tick * PERIOD_MS),
                                    {"adopted": out["adopted"], "ack": ack, "gate": None}, commitment)
        out = hrn.compose(request, probe_answers(request, GRASP), commitment, tick * PERIOD_MS)
        commitment = out["commitment"]
    reasons = [r["reason"] for r in out["records"] if r["kind"] in ("conflict", "release")]
    assert "forbidden_blocked" in reasons
    assert [r["reason"] for r in out["records"] if r["kind"] == "release"] == ["forbidden_blocked"]
    assert out["commitment"] is None
    assert out["command"]["constraints"]["forbidden_segment"] is True  # 이 틱도 보내지 않는다


def i2_scene(ee=(300, 0, 0)) -> dict:
    """docs/10 I2: o0을 든 말단 [300,0,0], 목적지 영역은 다른 곳. 작업면은 o1이 말해 준다."""
    scene = observation(objects=[obj("o0", (300, 0, -80)), obj("o1", (200, 220, -80), colour="blue")])
    scene["robot"].update(ee_pos_mm=list(ee), holding="o0", gripper_mm=20)
    scene["exec"] = {"seq": 1, "gripper": "closed"}
    return scene


def test_lift_rises_vertically_then_transport_keeps_the_height_then_place_descends():
    """docs/10 I2: 국면별 목표점 (docs/08 §5.6) — lift는 현재 XY에서 수직 상승, transport는 높이 유지, place는 하강."""
    hrn = harness()
    lift = i2_scene()
    request = hrn.build_request(lift, None, None)
    out = hrn.compose(request, probe_answers(request, GRASP), None, 0)
    command = out["command"]
    assert command["phase"] == "lift"
    assert command["path"]["target_mm"][:2] == [300, 0]
    transport_z = command["path"]["target_mm"][2]
    assert transport_z > 0
    assert command["gripper"] == "closed"  # 전환 틱: 현재 상태 유지
    assert controller_at(lift, closed=True).apply(command, 0)["applied"] is True

    commitment = out["commitment"]
    raised = i2_scene(ee=(300, 0, transport_z))
    raised.update(tick=1, sim_time_ms=PERIOD_MS)
    request = hrn.build_request(raised, None, commitment)
    out = hrn.compose(request, probe_answers(request, GRASP), commitment, PERIOD_MS)
    assert out["command"]["phase"] == "transport"
    assert out["command"]["path"]["target_mm"] == [30, 240, transport_z]

    commitment = out["commitment"]
    nearby = i2_scene(ee=(30 + CONFIG["phases"]["place_tolerance_mm"] + 10, 240, transport_z))
    nearby.update(tick=2, sim_time_ms=2 * PERIOD_MS)
    request = hrn.build_request(nearby, None, commitment)
    out = hrn.compose(request, probe_answers(request, GRASP), commitment, 2 * PERIOD_MS)
    assert out["command"]["phase"] == "transport", "허용 오차 밖에서는 아직 내려가지 않는다"
    assert out["command"]["path"]["target_mm"][2] == transport_z

    commitment = out["commitment"]
    above = i2_scene(ee=(40, 235, transport_z))
    above.update(tick=3, sim_time_ms=3 * PERIOD_MS)
    request = hrn.build_request(above, None, commitment)
    out = hrn.compose(request, probe_answers(request, GRASP), commitment, 3 * PERIOD_MS)
    assert out["command"]["phase"] == "place"
    place_target = out["command"]["path"]["target_mm"]
    assert place_target[:2] == [30, 240] and place_target[2] < transport_z
    assert controller_at(above, closed=True).apply(out["command"], 3 * PERIOD_MS)["applied"] is True


def test_the_grasp_phase_also_needs_xy_alignment_on_a_tall_cylinder():
    """3c-1의 E0 seed 101: 접근점(윗면+60)과 파지점(윗면−10)은 어떤 물체든 70mm 떨어져 있어
    `grasp_distance_mm`(80)만으로는 xy가 23mm 벗어난 채 파지 국면에 들어가 대각선으로 내려온다. 국면 전환은
    xy 정렬(`phases.grasp_xy_tolerance_mm`)도 요구한다."""
    tolerance = CONFIG["phases"]["grasp_xy_tolerance_mm"]
    assert 0 < tolerance < CONFIG["phases"]["grasp_distance_mm"]
    tall = obj("o0", (300, 0, -74), obb_mm=[48, 48, 76], **{"class": "cylinder"}, shape="cylinder")
    approach_z = -74 + 38 + CANDIDATES["approach_clearance_mm"]
    grasp_point = [300, 0, -74 + 38 - CANDIDATES["grasp_depth_mm"]]

    def phase_at(ee, executing=None, speed=0):
        scene = observation(objects=[tall], robot={**observation()["robot"], "ee_pos_mm": list(ee), "speed_mm_s": speed})
        if executing is not None:
            scene["exec"] = {**scene["exec"], **executing}
        request = harness().build_request(scene, None, None)
        geometry = request["harness"]["candidates"][candidate_id(GRASP)]
        assert math.dist(ee, grasp_point) <= CONFIG["phases"]["grasp_distance_mm"]
        return geometry["phase"], geometry["target_mm"]

    drifted = (300 + tolerance + 13, 0, approach_z)
    phase, target = phase_at(drifted)
    assert phase == "approach" and target[:2] == [300, 0], "xy가 벗어나 있으면 접근점으로 먼저 간다"
    aligned = (300 + tolerance - 5, 0, approach_z)
    phase, target = phase_at(aligned)
    assert phase == "grasp" and target == grasp_point
    # 정렬됐어도 접근 속도를 안고 있으면 아직이다 — 관성이 하강 초기에 가로로 흘러 테두리를 짚는다.
    phase, _ = phase_at(aligned, speed=CONFIG["phases"]["grasp_entry_speed_mm_s"] + 40)
    assert phase == "approach"

    # 들어간 뒤의 가로 흔들림은 되돌리지 않는다: 실행기가 이미 이 후보의 파지 국면이면 거리 조건만 본다.
    # 다른 후보(빠른 프로파일)의 파지 국면이었다면 이 후보에는 해당하지 않는다.
    phase, target = phase_at(drifted, {"phase": "grasp", "action_ref": candidate_id(GRASP), "executor": "MOVE_EE"})
    assert phase == "grasp" and target == grasp_point
    phase, _ = phase_at(drifted, {"phase": "grasp", "action_ref": candidate_id("grasp:o0:top:zoneL:fast"), "executor": "MOVE_EE"})
    assert phase == "approach"
    phase, _ = phase_at(drifted, {"phase": "approach", "action_ref": candidate_id(GRASP), "executor": "MOVE_EE"})
    assert phase == "approach"


def test_the_controller_opens_only_at_the_place_point():
    """놓기 국면의 open readiness는 말단이 놓기점에 와야 한다 (docs/08 §4 "release readiness")."""
    ctrl = Controller.from_config_path(CONTROLLER_CONFIG)
    ctrl.reset(ee_pos_mm=[30, 240, 60], ee_quat=[0.0, 0.0, 0.0, 1.0],
               gripper_mm=CONTROLLER["gripper"]["closed_mm"], now_ms=0)
    ctrl.observe({"ee_pos_mm": [30, 240, 60], "holding": "o0", "gripper_load_n": 0.0})
    place = {
        "seq": 1, "observed_at": 0, "issued_at": 0, "action_ref": "c1", "phase": "place",
        "path": {"kind": "direct", "target_ref": "o0", "target_mm": [30, 240, -48]},
        "speed_level": 1, "force_level": "light", "gripper": "open", "stop": False,
    }
    far = ctrl.apply(place, now_ms=0)
    assert far["applied"] is True and far["gripper_event"] is None
    assert far["gripper_wait"] == "readiness"

    ctrl.observe({"ee_pos_mm": [30, 240, -48 + CONTROLLER["gripper"]["open_readiness_distance_mm"] - 1]})
    near = ctrl.apply({**place, "seq": 2, "observed_at": 20, "issued_at": 20}, now_ms=20)
    assert near["gripper_event"] is not None


@pytest.mark.parametrize("boolean", [1.0, {"true": 0.9, "false": 0.1}, {"false": 0.1}])
def test_boolean_answers_accept_the_two_output_shapes(boolean):
    hrn = harness()
    _, out = step(hrn, observation(), answers(q_stop=boolean), None)
    assert out["command"]["stop"] is True


# -- 컨트롤러와의 통합 -------------------------------------------------------


def test_the_composed_commands_are_accepted_by_the_real_controller():
    """하네스가 낸 명령을 Task 3a의 실행기가 그대로 받는다 (합성 시각, 10Hz)."""
    hrn = harness()
    ctrl = Controller.from_config_path(CONTROLLER_CONFIG)
    ctrl.reset(ee_pos_mm=[0, 0, 200], ee_quat=[0.0, 0.0, 0.0, 1.0], gripper_mm=80, now_ms=0)

    commitment = None
    history = None
    applied = 0
    for tick in range(6):
        now_ms = tick * PERIOD_MS
        scene = observation(tick=tick, sim_time_ms=now_ms)
        request = hrn.build_request(scene, history, commitment)
        out = hrn.compose(request, answers(probabilities(**{GRASP.replace(":", "__"): 1.0})), commitment, now_ms)

        ctrl.observe({
            "ee_pos_mm": [0, 0, 200],
            "ee_quat": [0.0, 0.0, 0.0, 1.0],
            "gripper_mm": 80,
            "gripper_load_n": 0.0,
            "contact_force_n": 0.0,
            "nearest_obstacle_mm": 300.0,
            "target_distance_mm": 400.0,
            "holding": None,
            "slip_mm": 0.0,
            "speed_mm_s": 0.0,
        })
        ack = ctrl.apply(out["command"], now_ms=now_ms)
        assert ack["applied"] is True, ack["reason"]
        assert ack["seq"] == out["command"]["seq"]
        applied += 1
        commitment = out["commitment"]
        history = {"adopted": out["adopted"], "ack": ack, "gate": out["gate"]}
    assert applied == 6


def test_a_composed_tick_validates_as_a_stream_record():
    hrn = harness()
    commitment = committed(hrn, GRASP)
    request, out = step(hrn, observation(), answers(probabilities(**{GRASP.replace(":", "__"): 1.0})), commitment)
    tick = tick_of(request)
    tick["adopted"] = out["adopted"]
    tick["model_output"] = answers(probabilities(**{GRASP.replace(":", "__"): 1.0}))
    validate_record(stream(request) | {"ticks": [tick]})


# --------------------------------------------------------------------------
# 에피소드 레코드 (docs/08 §8)
# --------------------------------------------------------------------------


INSTRUCTION = {"version": 1, "t_ms": 0, "text": "red 상자를 왼쪽 정리 영역으로 옮겨라"}


def test_new_episode_assigns_the_split_from_the_scene_family():
    """분할은 장면 계열 단위로 **생성 전에** 배정한다 (docs/08 §8)."""
    record = new_episode("ep-0001", "scene-family-031", instructions=[INSTRUCTION])
    assert record["origin_group"] == "scene-family-031"
    assert record["split"] == assign_split("scene-family-031")
    assert record["schema_version"] == "stream-v0"
    assert record["prefix"]["question_set"] == harness().question_set_id()
    assert record["ticks"] == []


def test_new_episode_reads_the_question_set_without_building_a_harness(monkeypatch):
    """질문 세트 id는 설정에서 읽는다 — 앞단까지 딸린 하네스를 만들 이유가 없다."""
    from robo_jev.data import episode as episode_module
    from robo_jev.harness import robot as robot_module

    def refuse(*args, **kwargs):
        raise AssertionError("new_episode가 RobotHarness를 만들었다")

    monkeypatch.setattr(robot_module.RobotHarness, "__init__", refuse)
    record = new_episode("ep-0001b", "scene-family-031", instructions=[INSTRUCTION])
    assert record["prefix"]["question_set"] == CONFIG["question_set_id"][CONFIG["language"]]
    assert episode_module.default_question_set() == record["prefix"]["question_set"]


def test_default_versions_follow_the_config_on_disk(monkeypatch):
    """`default_versions`는 캐시하지 않는다 — 설정이 바뀌면 다음 호출이 그것을 적는다."""
    from robo_jev.data import episode as episode_module
    from robo_jev.sim import controller as controller_module

    before = episode_module.default_versions()
    original = controller_module.load_controller_config

    def patched(path):
        return {**original(path), "version": "c-test"}

    monkeypatch.setattr(controller_module, "load_controller_config", patched)
    after = episode_module.default_versions()
    assert before["controller"] != "c-test"
    assert after["controller"] == "c-test"


def test_append_tick_keeps_the_four_outputs_separate_and_drops_the_harness_block():
    hrn = harness()
    record = new_episode("ep-0002", "scene-family-031", instructions=[INSTRUCTION])
    request, out = step(hrn, observation(), answers(probabilities(**{GRASP.replace(":", "__"): 1.0})), None)
    model_output = answers(probabilities(**{GRASP.replace(":", "__"): 1.0}))

    tick = append_tick(
        record,
        request,
        model_output=model_output,
        adopted=out["adopted"],
        ack={"seq": 1, "applied": True, "gripper_event": None},
        labels=[{"question_id": "q_stop", "kind": "single", "answer": False}],
    )
    assert "harness" not in tick
    assert tick["model_output"] == model_output
    assert tick["adopted"] == out["adopted"]
    assert tick["ack"]["applied"] is True
    assert tick["labels"][0]["question_id"] == "q_stop"
    assert record["ticks"] == [tick]
    # 원본 요청은 건드리지 않는다.
    assert "harness" in request and "model_output" not in request


def test_a_new_instruction_version_is_appended_to_the_prefix():
    """지시가 바뀌면 리셋 없이 prefix 뒤에 붙는다 (docs/08 §3.1)."""
    hrn = harness()
    record = new_episode("ep-0003", "scene-family-031", instructions=[INSTRUCTION])
    append_tick(record, hrn.build_request(observation(), None, None))

    changed = observation(tick=1, sim_time_ms=100,
                          instruction={"version": 2, "t_ms": 100, "text": "blue 상자를 먼저 옮겨라"})
    append_tick(record, hrn.build_request(changed, None, None))

    assert [item["version"] for item in record["prefix"]["instructions"]] == [1, 2]
    assert record["prefix"]["instructions"][1]["text"] == "blue 상자를 먼저 옮겨라"


def test_finalize_records_every_version_and_validates_the_record():
    hrn = harness()
    record = new_episode("ep-0004", "scene-family-031", instructions=[INSTRUCTION])
    append_tick(record, hrn.build_request(observation(), None, None))
    finalize(record, provenance={"generator": "test", "seed": 1})

    assert set(record["versions"]) >= {"harness", "controller", "rules", "serializer", "extractor"}
    assert record["versions"]["harness"] == HARNESS_VERSION
    assert record["provenance"]["generator"] == "test"
    validate_record(record)


def test_finalize_refuses_a_record_that_breaks_the_contract():
    hrn = harness()
    record = new_episode("ep-0005", "scene-family-031", instructions=[INSTRUCTION])
    append_tick(
        record,
        hrn.build_request(observation(), None, None),
        labels=[{"question_id": "q_main", "kind": "single", "answer": "c-does-not-exist"}],
    )
    with pytest.raises(ValueError, match="존재하지 않는 후보"):
        finalize(record)


def test_aggregate_counts_what_the_qa_report_counts():
    hrn = harness()
    record = new_episode("ep-0006", "scene-family-031", instructions=[INSTRUCTION])
    for tick in range(3):
        append_tick(record, hrn.build_request(observation(tick=tick, sim_time_ms=tick * PERIOD_MS), None, None))
    finalize(record)

    counts = aggregate([record])
    report = validate_dataset([record])
    assert report["invalid_records"] == 0
    assert counts["episodes"] == report["episodes"] == 1
    assert counts["ticks"] == report["ticks"] == 3
    assert counts["questions"] == report["questions"]
    assert counts["splits"] == {record["split"]: 1}


def test_an_episode_of_composed_ticks_passes_the_automatic_qa():
    """하네스 + 규칙 기준군 + 실행기로 만든 에피소드가 계약·경계 검사를 통과한다."""
    hrn = harness()
    ctrl = Controller.from_config_path(CONTROLLER_CONFIG)
    ctrl.reset(ee_pos_mm=[0, 0, 200], ee_quat=[0.0, 0.0, 0.0, 1.0], gripper_mm=80, now_ms=0)
    record = new_episode("ep-0007", "scene-family-042", instructions=[INSTRUCTION])

    commitment = None
    history = None
    for tick in range(10):
        now_ms = tick * PERIOD_MS
        scene = observation(tick=tick, sim_time_ms=now_ms)
        request = hrn.build_request(scene, history, commitment)
        results = rule_judge(request)
        out = hrn.compose(request, results, commitment, now_ms)
        ctrl.observe({
            "ee_pos_mm": [0, 0, 200], "ee_quat": [0.0, 0.0, 0.0, 1.0], "gripper_mm": 80,
            "gripper_load_n": 0.0, "contact_force_n": 0.0, "nearest_obstacle_mm": 300.0,
            "target_distance_mm": 400.0, "holding": None, "slip_mm": 0.0, "speed_mm_s": 0.0,
        })
        ack = ctrl.apply(out["command"], now_ms=now_ms)
        append_tick(record, request, model_output=results, adopted=out["adopted"], ack=ack)
        commitment = out["commitment"]
        history = {"adopted": out["adopted"], "ack": ack, "gate": out["gate"]}

    finalize(record, evidence={"expert_log": "규칙 기준군", "rollouts": "없음"})
    report = validate_dataset([record])
    assert report["errors"] == []
    assert report["ticks"] == 10
    served = model_input(record)
    assert "harness" not in str(served)


# --------------------------------------------------------------------------
# 실제 시뮬레이터와의 폐루프 (docs/08 §9의 제작 파이프라인 한 조각)
# --------------------------------------------------------------------------


def test_the_harness_drives_a_real_e0_episode_and_records_it():
    """관측 → 요청 → 규칙 답 → 조합 → 명령 → 실행을 실제 환경에서 10틱 돈다.

    앞단 어댑터가 진짜 관측 스키마를 받는지, 하네스 명령이 실제 실행기에 받아들여지는지,
    그 기록이 계약과 정보 경계 검사를 통과하는지를 한 번에 본다.
    """
    from robo_jev.sim.environment import Environment

    env = Environment(config_path=str(SIM_CONFIG), profile="E0")
    try:
        scene = env.reset(seed=17)
        hrn = harness()
        record = new_episode(
            "ep-sim-017",
            "scene-family-e0-017",
            instructions=[scene["instruction"]],
        )
        assert scene["objects"][0]["desc"], "상태 스키마의 물체 설명이 없다"

        commitment = None
        history = None
        applied = 0
        for _ in range(10):
            request = hrn.build_request(scene, history, commitment)
            results = rule_judge(request)
            out = hrn.compose(request, results, commitment, int(scene["sim_time_ms"]))

            ack = None
            for control_step in range(PERIOD_MS // (1000 // CONTROLLER["timing"]["control_hz"])):
                scene = env.step(out["command"] if control_step == 0 else None)
                ack = scene["ack"] or ack
            applied += int(bool(ack and ack["applied"]))

            append_tick(record, request, model_output=results, adopted=out["adopted"], ack=ack)
            commitment = out["commitment"]
            history = {"adopted": out["adopted"], "ack": ack, "gate": out["gate"]}

        finalize(record, evidence=hrn.adapter.evidence())
        report = validate_dataset([record])
        assert report["errors"] == []
        assert report["ticks"] == 10
        assert applied == 10, "실행기가 하네스 명령을 받아들이지 않았다"
        assert aggregate([record])["questions"] == report["questions"]
    finally:
        env.close()


def run_e0_episode(seed: int, *, max_ticks: int = 300) -> dict:
    """규칙 기준군 + 하네스 + 실제 실행기로 E0 에피소드 하나를 완료(done 게이트)까지 돈다.

    판단 틱마다 50Hz 제어 5회를 진행한다 (docs/10 closed_loop_probe.py와 같은 구성).
    """
    from robo_jev.harness.rule_judge import RuleJudge
    from robo_jev.sim.environment import Environment

    env = Environment(config_path=str(SIM_CONFIG), profile="E0")
    try:
        scene = env.reset(seed=seed)
        hrn = harness()
        judge = RuleJudge()
        text = scene["instruction"]["text"]
        target = next(entry for entry in scene["objects"] if text.startswith(entry["desc"]))
        zone = next(entry for entry in scene["zones"] if entry["desc"] in text)

        commitment = None
        history = None
        summary = {
            "seed": seed, "first_joint_tick": None, "holding_tick": None, "done_tick": None,
            "gates": {}, "acks": {}, "ticks": 0, "final": None,
        }
        for tick in range(max_ticks):
            request = hrn.build_request(scene, history, commitment)
            results = judge(request)
            out = hrn.compose(request, results, commitment, int(scene["sim_time_ms"]))
            key = next(
                (entry["key"] for entry in request["request"]["candidates"]["q_main"] if entry["id"] == out["adopted"]["main"]),
                None,
            )
            if summary["first_joint_tick"] is None and key and ":" in key:
                summary["first_joint_tick"] = tick
            summary["gates"][str(out["gate"])] = summary["gates"].get(str(out["gate"]), 0) + 1

            ack = None
            for control_step in range(PERIOD_MS // (1000 // CONTROLLER["timing"]["control_hz"])):
                scene = env.step(out["command"] if control_step == 0 else None)
                ack = scene["ack"] or ack
                if summary["holding_tick"] is None and scene["robot"]["holding"] is not None:
                    summary["holding_tick"] = tick
            result = "applied" if ack and ack["applied"] else str((ack or {}).get("reason"))
            summary["acks"][result] = summary["acks"].get(result, 0) + 1
            summary["ticks"] = tick + 1
            commitment = out["commitment"]
            history = {"adopted": out["adopted"], "ack": ack, "gate": out["gate"]}
            if out["gate"] == "done":
                summary["done_tick"] = tick
                break

        placed = next(entry for entry in scene["objects"] if entry["id"] == target["id"])
        x0, y0, x1, y1 = zone["bounds_mm"]
        summary["final"] = {
            "target_pos_mm": placed["pos_mm"],
            "inside_zone": x0 <= placed["pos_mm"][0] <= x1 and y0 <= placed["pos_mm"][1] <= y1,
            "gripper_mm": scene["robot"]["gripper_mm"],
            "holding": scene["robot"]["holding"],
        }
        return summary
    finally:
        env.close()


@pytest.mark.parametrize("seed", [17, 29, 43])
def test_the_rule_baseline_completes_an_e0_episode(seed):
    """E0 완료 검증 (docs/10 §3, Part D3): 20틱 안에 결합 행동을 채택하고, 잡고, 목표 영역에 놓고 연다."""
    summary = run_e0_episode(seed)
    assert summary["first_joint_tick"] is not None and summary["first_joint_tick"] < 20, summary
    assert summary["holding_tick"] is not None, summary
    assert summary["done_tick"] is not None, summary
    assert summary["final"]["inside_zone"] is True, summary
    assert summary["final"]["holding"] is None, summary
    midpoint = (CONTROLLER["gripper"]["open_mm"] + CONTROLLER["gripper"]["closed_mm"]) / 2
    assert summary["final"]["gripper_mm"] > midpoint, summary
    assert set(summary["acks"]) == {"applied"}, summary


def test_hidden_world_changes_do_not_change_the_request():
    """docs/10 I5의 대조: 가려진 물체의 자세·사건·이동·파생 값이 요청 어디에도 새지 않는다."""
    first, second = harness(), harness()
    scene = observation(objects=[obj("o0", (300, 0, -80)), obj("o1", (200, 220, -80), colour="blue")])
    first.build_request(scene, None, None)
    second.build_request(scene, None, None)

    quiet = observation(objects=[obj("o0", (300, 0, -80)), obj("o1", (200, 220, -80), colour="blue")],
                        tick=1, sim_time_ms=100)
    quiet["objects"][1].update(visible=False, visible_ratio=0.0)
    moved = copy.deepcopy(quiet)
    moved["objects"][1]["pos_mm"] = [650, 220, -80]
    moved["events"] = [{"kind": "disturbance_applied", "object": "o1", "sim_ms": 100}]

    expected = first.build_request(quiet, None, None)
    actual = second.build_request(moved, None, None)
    state = actual["request"]["state"]
    assert next(entry for entry in state["objects"] if entry["id"] == "o1")["pose_mm"] == [200, 220, -80]
    assert state["events"] == []
    assert state["derived"] == expected["request"]["state"]["derived"]
    assert not any(geometry["moving"] for geometry in actual["harness"]["candidates"].values())
    assert actual == expected
    assert second.adapter.evidence()["simulator_events"] == moved["events"]


def test_compose_accepts_a_tick_built_by_another_tool():
    """다른 도구가 만든 틱(D0 fixture)에도 조합 규칙이 그대로 걸린다.

    후보·경로의 id 규칙은 하네스 버전마다 다를 수 있으므로, 채택 결과는 **그 틱의 목록에
    있는 id**를 가리켜야 한다 (docs/02 §6: 만료된 후보를 인덱스로 해석하지 않는다).
    """
    record = copy.deepcopy(read_jsonl(D0_STREAMS)[0])
    hrn = harness()
    kept = []
    for tick in record["ticks"][:60]:
        results = rule_judge(tick)
        out = hrn.compose(tick, results, tick["request"]["commitment"], tick["sim_ms"])
        ids = {entry["id"] for entry in tick["request"]["candidates"]["q_main"]}
        path_ids = {entry["id"] for entry in tick["request"]["candidates"].get("q_path", [])}
        assert out["adopted"]["main"] in ids
        assert out["adopted"]["path"] in path_ids or out["adopted"]["path"] is None
        tick["adopted"] = out["adopted"]
        tick["model_output"] = results
        kept.append(tick)

    record["ticks"] = kept
    record.pop("labels", None)
    validate_record(record)
