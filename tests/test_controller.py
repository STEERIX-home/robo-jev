"""컨트롤러 계약 검사 (docs/08 §6).

컨트롤러는 순수 논리다. 물리는 `tests/test_sim_replay.py`가 본다. 여기서는 합성 관측과
합성 시각으로 몰아 수명·혼합·정지 전이·반사·그리퍼 이벤트·거절을 하나씩 확인한다.
수치는 전부 `configs/controller/osc_v0.yaml`에서 읽는다 — 검사도 상수를 쓰지 않는다.
"""

import pytest
import yaml
from helpers import CONTROLLER_CONFIG

from robo_jev.sim.controller import EXECUTORS, Controller

CONFIG = yaml.safe_load(CONTROLLER_CONFIG.read_text(encoding="utf-8"))
LIFETIME = CONFIG["lifetime"]
BLEND_MS = CONFIG["blend"]["blend_ms"]
PERIOD_MS = 1000 // CONFIG["timing"]["control_hz"]
SPEEDS = CONFIG["speed_levels_m_s"]
GRIPPER = CONFIG["gripper"]
REFLEX = CONFIG["reflex"]
REACH = CONFIG["reach"]

START_MM = [400, 0, 200]
START_QUAT = [1.0, 0.0, 0.0, 0.0]  # xyzw: 말단이 아래를 보는 초기 자세


def sensors(**over) -> dict:
    """계약 검사의 기본 관측. 위험 신호는 전부 꺼진 상태다."""
    base = {
        "ee_pos_mm": list(START_MM),
        "ee_quat": list(START_QUAT),
        "gripper_mm": GRIPPER["open_mm"],
        "gripper_load_n": 0.0,
        "contact_force_n": 0.0,
        "nearest_obstacle_mm": 500.0,
        "target_distance_mm": None,
        "holding": None,
        "slip_mm": 0.0,
        "speed_mm_s": 0.0,
    }
    base.update(over)
    return base


def controller(**over) -> Controller:
    ctrl = Controller.from_config_path(CONTROLLER_CONFIG)
    ctrl.reset(ee_pos_mm=START_MM, ee_quat=START_QUAT, gripper_mm=GRIPPER["open_mm"], now_ms=0)
    ctrl.observe(sensors(**over))
    return ctrl


def move(seq: int = 1, now_ms: int = 0, target_mm=(600, 0, 200), **over) -> dict:
    """docs/08 §6 형식의 명령. `lease_until`은 생략하면 발행 시각 + lease다."""
    command = {
        "seq": seq,
        "request_id": f"r{seq}",
        "observed_at": now_ms,
        "issued_at": now_ms,
        "goal_version": 1,
        "candidate_set_version": 1,
        "action_ref": "c1",
        "phase": "approach",
        "path": {"kind": "direct", "target_ref": "o1", "target_mm": list(target_mm)},
        "speed_level": 2,
        "force_level": "light",
        "gripper": "open",
        "stop": False,
        "geometry_age_ms": 40,
        "target_moving": False,
    }
    command.update(over)
    return command


# --------------------------------------------------------------------------
# 수명
# --------------------------------------------------------------------------


def test_late_response_gets_no_fresh_lease():
    """관측 deadline을 넘긴 응답은 적용되지 않고 lease도 새로 붙지 않는다."""
    ctrl = controller()
    first = ctrl.apply(move(seq=1, now_ms=0), now_ms=0)
    assert first["applied"] is True
    lease = ctrl.state_dict()["lease_until"]
    assert lease == LIFETIME["lease_ms"]

    late_at = 400
    stale_observation = late_at - LIFETIME["observation_deadline_ms"] - 1
    ack = ctrl.apply(move(seq=2, now_ms=late_at, observed_at=stale_observation), now_ms=late_at)

    assert ack["applied"] is False
    assert ack["stale"] is True
    assert ack["reason"] == "observation_deadline"
    assert ctrl.state_dict()["lease_until"] == lease, "늦은 응답이 lease를 늘렸다"


def test_seq_regression_is_discarded():
    ctrl = controller()
    ctrl.apply(move(seq=7, now_ms=0), now_ms=0)
    ack = ctrl.apply(move(seq=6, now_ms=20), now_ms=20)

    assert ack["applied"] is False
    assert ack["rejected"] is True
    assert ack["reason"] == "seq_regression"
    assert ctrl.state_dict()["last_seq"] == 7


def test_geometry_age_over_tolerance_requests_observation():
    """움직이는 대상은 200ms, 정지 대상은 500ms가 허용치다."""
    moving_tolerance = LIFETIME["geometry_age_moving_ms"]
    static_tolerance = LIFETIME["geometry_age_static_ms"]
    assert moving_tolerance < static_tolerance

    ctrl = controller()
    age = moving_tolerance + 1
    ack = ctrl.apply(move(seq=1, geometry_age_ms=age, target_moving=True), now_ms=0)
    assert ack["applied"] is False
    assert ack["reason"] == "geometry_age"
    assert ack["request_observation"] is True

    ctrl = controller()
    ok = ctrl.apply(move(seq=1, geometry_age_ms=age, target_moving=False), now_ms=0)
    assert ok["applied"] is True, "같은 나이도 정지 대상에서는 허용치 안이다"


def test_lease_expiry_decelerates_then_holds_after_one_second():
    lease_ms = LIFETIME["lease_ms"]
    hold_after = LIFETIME["hold_after_stale_ms"]
    ctrl = controller()
    ctrl.apply(move(seq=1, now_ms=0), now_ms=0)
    for now in range(0, lease_ms + 1, PERIOD_MS):
        ctrl.advance(now)
    moving = ctrl.advance(lease_ms)
    assert moving["stale"] is False
    assert moving["speed_mm_s"] > 0.0

    just_after = ctrl.advance(lease_ms + PERIOD_MS)
    assert just_after["stale"] is True
    assert just_after["speed_mm_s"] < moving["speed_mm_s"], "lease가 끝났는데 감속하지 않는다"
    assert just_after["gripper"] == "open", "정지 중에 그리퍼 상태를 바꿨다"

    held = ctrl.advance(lease_ms + hold_after)
    assert held["executor"] == "HOLD"
    assert held["speed_mm_s"] == pytest.approx(0.0)


# --------------------------------------------------------------------------
# 혼합과 정지 전이
# --------------------------------------------------------------------------


def test_blend_takes_100ms():
    ctrl = controller()
    ctrl.apply(move(seq=1, now_ms=0), now_ms=0)

    steps = BLEND_MS // PERIOD_MS
    alphas = [ctrl.advance(i * PERIOD_MS)["blend_alpha"] for i in range(steps + 1)]

    assert alphas[0] == pytest.approx(0.0)
    assert alphas == pytest.approx([i / steps for i in range(steps + 1)])
    assert ctrl.advance(BLEND_MS)["blending"] is False
    assert ctrl.advance(BLEND_MS - PERIOD_MS)["blending"] is True


def test_stop_bypasses_blend_with_a_fixed_deceleration():
    ctrl = controller()
    ctrl.apply(move(seq=1, now_ms=0), now_ms=0)
    for i in range(BLEND_MS // PERIOD_MS + 1):
        ctrl.advance(i * PERIOD_MS)
    cruising = ctrl.advance(BLEND_MS)
    assert cruising["speed_mm_s"] == pytest.approx(SPEEDS[2] * 1000.0)

    ack = ctrl.apply(move(seq=2, now_ms=BLEND_MS, stop=True), now_ms=BLEND_MS)
    assert ack["applied"] is True
    assert ack["stop_transition"] is True

    decel = CONFIG["stop"]["decel_mm_s2"]
    after = ctrl.advance(BLEND_MS + PERIOD_MS)
    assert after["blending"] is False, "정지가 혼합을 거쳤다"
    assert after["stopping"] is True
    expected = cruising["speed_mm_s"] - decel * PERIOD_MS / 1000.0
    assert after["speed_mm_s"] == pytest.approx(expected)

    stopped = ctrl.advance(BLEND_MS + 1000)
    assert stopped["speed_mm_s"] == pytest.approx(0.0)


def test_speed_level_sets_the_cruise_cap():
    for level, metres_per_second in enumerate(SPEEDS):
        ctrl = controller()
        ctrl.apply(move(seq=1, speed_level=level), now_ms=0)
        for i in range(BLEND_MS // PERIOD_MS + 1):
            ctrl.advance(i * PERIOD_MS)
        assert ctrl.advance(BLEND_MS)["speed_cap_mm_s"] == pytest.approx(metres_per_second * 1000.0)


def test_force_level_selects_impedance_and_contact_allowance():
    for name, values in CONFIG["force_levels"].items():
        ctrl = controller()
        ctrl.apply(move(seq=1, force_level=name), now_ms=0)
        setpoint = ctrl.advance(0)
        assert setpoint["impedance_kp"] == pytest.approx(values["impedance_kp"])
        assert setpoint["contact_allowance_n"] == pytest.approx(values["contact_allowance_n"])


# --------------------------------------------------------------------------
# 반사
# --------------------------------------------------------------------------


def test_force_limit_reflex_stops_without_a_policy_command():
    ctrl = controller()
    ctrl.apply(move(seq=1, now_ms=0), now_ms=0)
    for i in range(BLEND_MS // PERIOD_MS + 1):
        ctrl.advance(i * PERIOD_MS)

    ctrl.observe(sensors(contact_force_n=REFLEX["force_limit_n"] + 1.0))
    reflex = ctrl.advance(BLEND_MS + PERIOD_MS)  # 명령 없이 주기만 돈다

    assert reflex["stopping"] is True
    kinds = [event["kind"] for event in ctrl.drain_events()]
    assert "reflex_stop" in kinds


def test_proximity_reflex_slows_without_stopping():
    ctrl = controller()
    ctrl.apply(move(seq=1, now_ms=0), now_ms=0)
    for i in range(BLEND_MS // PERIOD_MS + 1):
        ctrl.advance(i * PERIOD_MS)
    full = ctrl.advance(BLEND_MS)["speed_cap_mm_s"]

    ctrl.observe(sensors(nearest_obstacle_mm=REFLEX["proximity_mm"] - 1.0))
    slowed = ctrl.advance(BLEND_MS + PERIOD_MS)

    assert slowed["stopping"] is False
    assert slowed["speed_cap_mm_s"] == pytest.approx(full * REFLEX["proximity_speed_factor"])


def test_slip_reflex_keeps_the_grip_and_reports_an_event():
    ctrl = controller(gripper_mm=GRIPPER["closed_mm"], holding="o3")
    ctrl.apply(move(seq=1, gripper="closed", target_distance_mm=0.0), now_ms=0)
    ctrl.drain_events()

    ctrl.observe(
        sensors(
            gripper_mm=GRIPPER["closed_mm"],
            holding="o3",
            slip_mm=REFLEX["slip_detect_mm"] + 1.0,
        )
    )
    setpoint = ctrl.advance(PERIOD_MS)

    assert setpoint["gripper"] == "closed", "미끄러짐에 파지를 놓았다"
    events = ctrl.drain_events()
    assert [event["kind"] for event in events].count("reflex_slip") == 1


# --------------------------------------------------------------------------
# 그리퍼 이벤트
# --------------------------------------------------------------------------


def test_repeated_gripper_command_emits_exactly_one_event():
    ctrl = controller(target_distance_mm=0.0)
    ids = []
    for seq in range(1, 6):
        ack = ctrl.apply(move(seq=seq, gripper="closed"), now_ms=seq * PERIOD_MS)
        assert ack["applied"] is True
        if ack["gripper_event"] is not None:
            ids.append(ack["gripper_event"])
        ctrl.advance(seq * PERIOD_MS)

    assert len(ids) == 1, f"같은 그리퍼 상태에 이벤트가 여러 번 났다: {ids}"
    assert ids[0].startswith(GRIPPER["event_id_prefix"])
    gripper_events = [e for e in ctrl.drain_events() if e["kind"] == "gripper"]
    assert [e["id"] for e in gripper_events] == ids


def test_gripper_readiness_failure_waits_without_an_event():
    too_far = GRIPPER["close_readiness_distance_mm"] + 10.0
    ctrl = controller(target_distance_mm=too_far)

    waiting = ctrl.apply(move(seq=1, gripper="closed"), now_ms=0)
    assert waiting["gripper_event"] is None
    assert waiting["gripper_wait"] == "readiness"
    assert [e["kind"] for e in ctrl.drain_events()].count("gripper_wait") == 1

    ctrl.observe(sensors(target_distance_mm=0.0))
    ready = ctrl.apply(move(seq=2, gripper="closed"), now_ms=PERIOD_MS)
    assert ready["gripper_event"] is not None


def test_open_waits_while_the_gripper_is_loaded():
    loaded = GRIPPER["open_readiness_force_n"] + 1.0
    ctrl = controller(gripper_mm=GRIPPER["closed_mm"], gripper_load_n=loaded, holding="o3")
    ctrl.apply(move(seq=1, gripper="closed", target_distance_mm=0.0), now_ms=0)

    ack = ctrl.apply(move(seq=2, gripper="open"), now_ms=PERIOD_MS)
    assert ack["gripper_event"] is None
    assert ack["gripper_wait"] == "readiness"


# --------------------------------------------------------------------------
# 거절
# --------------------------------------------------------------------------


def test_rejection_returns_a_reason():
    ctrl = controller()
    far = REACH["workspace_radius_mm"] + 100
    ack = ctrl.apply(move(seq=1, target_mm=(far, 0, 200)), now_ms=0)

    assert ack["applied"] is False
    assert ack["rejected"] is True
    assert ack["reason"] == "unreachable"
    assert [e["kind"] for e in ctrl.drain_events()].count("rejected") == 1


def test_below_the_table_is_rejected():
    ctrl = controller()
    ack = ctrl.apply(move(seq=1, target_mm=(500, 0, REACH["min_height_mm"] - 10)), now_ms=0)
    assert ack["rejected"] is True
    assert ack["reason"] == "collision"


def test_transition_collision_becomes_a_stop_transition():
    """전환 구간이 걸리면 명령을 그대로 쓰지 않고 정지 전이로 바꾼다 (docs/08 §6 "혼합")."""
    ctrl = controller()
    ctrl.apply(move(seq=1, now_ms=0), now_ms=0)
    for i in range(BLEND_MS // PERIOD_MS + 1):
        ctrl.advance(i * PERIOD_MS)

    # 전환 구간이 금지 영역을 지난다고 알려 준다.
    blocked = move(seq=2, now_ms=BLEND_MS, target_mm=(600, 300, 200))
    blocked["constraints"] = {"forbidden_segment": True}
    ack = ctrl.apply(blocked, now_ms=BLEND_MS)

    assert ack["applied"] is False
    assert ack["stop_transition"] is True
    assert ack["reason"] == "transition_collision"


# --------------------------------------------------------------------------
# ACK 형식·실행기·상태 직렬화
# --------------------------------------------------------------------------


def test_ack_carries_the_contract_fields():
    ctrl = controller()
    ack = ctrl.apply(move(seq=1), now_ms=0)
    required = {
        "seq",
        "applied",
        "reason",
        "gripper_event",
        "stale",
        "rejected",
        "stop_transition",
        "request_observation",
        "gripper_wait",
        "executor",
        "lease_until",
    }
    assert required <= set(ack)
    assert ack["seq"] == 1
    assert isinstance(ack["applied"], bool)
    assert ack["reason"] is None
    assert ack["executor"] == "MOVE_EE"


def test_executors_match_the_plan_table():
    assert list(EXECUTORS) == CONFIG["executors"]
    assert set(EXECUTORS) == {
        "MOVE_EE",
        "SET_GRIPPER",
        "PUSH_SEGMENT",
        "OBSERVE",
        "HOLD",
        "REQUEST_REPLAN",
    }


@pytest.mark.parametrize(
    ("command", "executor"),
    [
        ({"kind": "HOLD", "duration_ms": 100}, "HOLD"),
        ({"kind": "OBSERVE", "duration_ms": 200}, "OBSERVE"),
        ({"kind": "REQUEST_REPLAN", "reason": "지시가 모호하다"}, "REQUEST_REPLAN"),
        ({"kind": "SET_GRIPPER", "gripper": "closed"}, "SET_GRIPPER"),
        ({"kind": "MOVE_EE", "target_mm": [500, 0, 200]}, "MOVE_EE"),
        ({"kind": "PUSH_SEGMENT", "target_mm": [500, 40, 200]}, "PUSH_SEGMENT"),
    ],
)
def test_primitive_commands_name_an_executor(command, executor):
    """계획서의 `{"kind": "HOLD", "duration_ms": 100}` 형식은 실행기를 직접 부른다."""
    ctrl = controller(target_distance_mm=0.0)
    ack = ctrl.apply(command, now_ms=0)
    assert ack["applied"] is True
    assert ack["executor"] == executor
    assert ack["seq"] == 1, "순번이 없는 명령은 컨트롤러가 이어 붙인다"


def test_hold_command_freezes_the_setpoint():
    ctrl = controller()
    ctrl.apply({"kind": "HOLD", "duration_ms": 100}, now_ms=0)
    for i in range(1, 6):
        setpoint = ctrl.advance(i * PERIOD_MS)
    assert setpoint["speed_mm_s"] == pytest.approx(0.0)
    assert setpoint["ee_pos_mm"] == pytest.approx(START_MM)


def test_unknown_command_is_rejected_with_a_reason():
    ctrl = controller()
    ack = ctrl.apply({"kind": "TELEPORT"}, now_ms=0)
    assert ack["applied"] is False
    assert ack["rejected"] is True
    assert ack["reason"] == "unknown_executor"


def test_state_dict_round_trips():
    ctrl = controller(target_distance_mm=0.0)
    ctrl.apply(move(seq=3, gripper="closed"), now_ms=0)
    ctrl.advance(PERIOD_MS)
    saved = ctrl.state_dict()

    times = range(2 * PERIOD_MS, 200, PERIOD_MS)
    later = [value for now in times for value in ctrl.advance(now)["ee_pos_mm"]]

    restored = Controller.from_config_path(CONTROLLER_CONFIG)
    restored.reset(ee_pos_mm=START_MM, ee_quat=START_QUAT, gripper_mm=GRIPPER["open_mm"], now_ms=0)
    restored.observe(sensors(target_distance_mm=0.0))
    restored.load_state_dict(saved)
    again = [value for now in times for value in restored.advance(now)["ee_pos_mm"]]

    assert again == pytest.approx(later)
    assert len(set(later)) > 1, "말단이 움직이지 않으면 재현을 확인한 것이 아니다"
    assert restored.state_dict() == ctrl.state_dict()


def test_state_dict_is_json_safe():
    import json

    ctrl = controller(target_distance_mm=0.0)
    ctrl.apply(move(seq=1, gripper="closed"), now_ms=0)
    ctrl.advance(PERIOD_MS)
    assert json.loads(json.dumps(ctrl.state_dict())) == ctrl.state_dict()
