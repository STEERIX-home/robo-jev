"""로봇 스트림 하네스 검사 — 요청 구성과 조합 규칙 v0 (docs/08 §3~§6).

하네스는 순수 논리다. 물리는 `tests/test_sim_replay.py`가 본다. 여기서는 합성 관측과
합성 시각으로 몰아 후보 생성·경유점·조합 규칙을 하나씩 확인하고, 마지막에 **실제
컨트롤러**가 하네스의 명령을 받아들이는지 본다. 수치는 전부 설정에서 읽는다.
"""

import copy

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
    assert "grasp:o0:side:zoneL:fast" in keys
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


def test_objects_that_are_not_observed_enough_are_not_candidate_targets():
    hrn = harness()
    hrn.build_request(observation(), None, None)  # 처음에는 다 보인다

    hidden = observation(tick=2, sim_time_ms=200)
    hidden["objects"][1].update(visible=False, visible_ratio=0.1)
    request = hrn.build_request(hidden, None, None)
    assert not any(":o1:" in key for key in keys_of(request))
    assert request["harness"]["accounting"]["dropped"]["occluded"] > 0


def test_candidate_cap_and_inclusion_rate():
    """상한을 넘으면 정답을 모르는 채로 줄이고 포함률을 기록한다 (docs/02 §3)."""
    crowd = [obj(f"o{index}", (140 + 30 * index, -300 + 70 * index, -80)) for index in range(10)]
    request = harness().build_request(observation(objects=crowd), None, None)
    accounting = request["harness"]["accounting"]

    assert len(request["request"]["candidates"]["q_main"]) <= CANDIDATES["max"]
    assert accounting["kept"] < accounting["feasible"]
    assert accounting["capped"] is True
    assert accounting["inclusion_rate"] == pytest.approx(accounting["kept"] / accounting["feasible"])
    assert accounting["dropped"]["cap"] == accounting["feasible"] - accounting["kept"]


def test_the_cap_keeps_a_spread_of_targets():
    """상한 안에서 대상 다양성을 지킨다 — 한 물체가 목록을 독식하지 않는다."""
    crowd = [obj(f"o{index}", (140 + 30 * index, -300 + 70 * index, -80)) for index in range(10)]
    request = harness().build_request(observation(objects=crowd), None, None)
    targets = {key.split(":")[1] for key in keys_of(request) if ":" in key}
    assert len(targets) >= 6


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


def test_the_local_planner_produces_real_waypoints_around_a_blocker():
    """`via`는 플래너가 실제로 만든 경유점이다 — 컨트롤러는 좌표 없는 via를 거절한다."""
    blocked = blocked_scene()
    key = "grasp:o0:side:zoneL:slow"
    commitment = {"action_ref": candidate_id(key), "key": key, "phase": "approach"}
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
    geometry = request["harness"]["candidates"][candidate_for(request, "grasp:o0:side:zoneL:slow")["id"]]
    assert geometry["path_clear"] is False
    assert geometry["blocker"] == "o5"


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
        "start_pose_mm": [300, 0, -80],
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
    """후보 목록에서 사라진 행동은 무효다 (대상이 안 보이게 된 경우)."""
    hrn = harness()
    commitment = committed(hrn, GRASP)
    gone = observation(tick=1, sim_time_ms=PERIOD_MS)
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
    """전환 틱에는 네 답을 버리고 새 후보의 초기 프로파일을 쓴다 (docs/08 §5.4)."""
    hrn = harness()
    commitment = committed(hrn, GRASP, challenger=candidate_id(OTHER), challenger_ticks=M - 1)
    _, out = step(
        hrn,
        observation(),
        answers(
            probabilities(**{GRASP.replace(":", "__"): 0.2, OTHER.replace(":", "__"): 0.9}),
            q_speed={"3": 1.0},
            q_force={"2": 1.0},
            q_path={"ph": 1.0},
        ),
        commitment,
    )
    assert out["switch"] is True
    assert out["adopted"]["speed"] == COMPOSE["initial_profile"]["speed_level"]
    assert out["adopted"]["force"] == COMPOSE["initial_profile"]["force_level"]
    assert out["adopted"]["path"] == "p0"
    assert "aux_discarded" in count_records(out["records"])


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
    key = "grasp:o0:side:zoneL:slow"
    commitment = committed(hrn, key, scene)
    request, out = step(hrn, scene, answers(probabilities(**{key.replace(":", "__"): 1.0}), q_path={"p1": 1.0}), commitment)

    assert out["command"]["path"]["kind"] == "via"
    assert out["command"]["path"]["waypoint_ref"] == "w1"
    assert out["command"]["path"]["waypoint_mm"] == request["harness"]["waypoints"]["w1"]["pos_mm"]


def test_stale_geometry_sends_the_chosen_candidate_to_the_observe_branch():
    """선택된 후보의 기하가 허용치를 넘으면 적용하지 않고 관측 분기로 보낸다 (docs/08 §5.0).

    외란으로 움직인 대상의 허용치는 200ms다. 후보로 남을 만큼은 보이지만(옆면) 기하가
    그보다 오래된 틱을 만든다.
    """
    hrn = harness()
    hrn.build_request(
        observation(events=[{"kind": "disturbance_applied", "object": "o0", "sim_ms": 0}]), None, None
    )
    stale_ms = CONTROLLER["lifetime"]["geometry_age_moving_ms"] + PERIOD_MS
    scene = observation(tick=stale_ms // PERIOD_MS, sim_time_ms=stale_ms)
    scene["objects"][0].update(visible=False, visible_ratio=0.5)

    key = "grasp:o0:side:zoneL:slow"
    commitment = {
        "action_ref": candidate_id(key), "key": key, "phase": "approach", "held_ticks": 2,
        "last_switch_tick": 0, "goal_version": 1, "challenger": None, "challenger_ticks": 0,
        "stop_ticks": 0,
    }
    request, out = step(hrn, scene, answers(probabilities(**{key.replace(":", "__"): 1.0})), commitment)

    assert request["harness"]["candidates"][candidate_id(key)]["geometry_age_ms"] == stale_ms
    assert [r["kind"] for r in out["records"] if r["kind"] == "geometry_age"] == ["geometry_age"]
    assert out["gate"] == "observe"
    assert out["adopted"]["main"] == candidate_id("observe")
    assert out["commitment"] is None


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
    hrn = harness()
    commitment = committed(hrn, GRASP)
    _, out = step(hrn, observation(), answers(probabilities(**{GRASP.replace(":", "__"): 1.0})), commitment)
    counts = count_records(out["records"])
    assert counts.get("hold") == 1
    for record in out["records"]:
        assert set(record) >= {"kind"}
        assert isinstance(record["kind"], str)


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
