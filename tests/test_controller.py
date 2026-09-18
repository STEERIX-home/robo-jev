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
RETREAT = CONFIG["retreat"]
DEFAULTS = CONFIG["defaults"]

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
    # lease 만료(`lease_expired`)와 이름으로 구분한다 — 둘 다 "stale"이지만 원인이 다르다.
    assert ack["reason"] == "observation_late"
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
    """감속 구간과 HOLD 절차는 다른 상태다 — 실행 기록에서 구분돼야 한다 (docs/08 §6 "stale")."""
    lease_ms = LIFETIME["lease_ms"]
    hold_after = LIFETIME["hold_after_stale_ms"]
    ctrl = controller()
    ctrl.apply(move(seq=1, now_ms=0), now_ms=0)
    for now in range(0, lease_ms + 1, PERIOD_MS):
        ctrl.advance(now)
    moving = ctrl.advance(lease_ms)
    assert moving["stale"] is False
    assert moving["speed_mm_s"] > 0.0
    ctrl.drain_events()

    just_after = ctrl.advance(lease_ms + PERIOD_MS)
    assert just_after["stale"] is True
    assert just_after["speed_mm_s"] < moving["speed_mm_s"], "lease가 끝났는데 감속하지 않는다"
    assert just_after["gripper"] == "open", "정지 중에 그리퍼 상태를 바꿨다"
    # 감속은 아직 HOLD 절차가 아니다. 진행 중이던 행동이 무엇이었는지 남아 있어야 한다.
    assert just_after["executor"] == "MOVE_EE"
    assert just_after["holding_after_stale"] is False
    assert "hold_entered" not in [event["kind"] for event in ctrl.drain_events()]

    edge = ctrl.advance(lease_ms + hold_after - PERIOD_MS)
    assert edge["executor"] == "MOVE_EE", "1초가 되기 전에 HOLD로 넘어갔다"

    held = ctrl.advance(lease_ms + hold_after)
    assert held["executor"] == "HOLD"
    assert held["holding_after_stale"] is True
    assert held["speed_mm_s"] == pytest.approx(0.0)
    assert [event["kind"] for event in ctrl.drain_events()].count("hold_entered") == 1

    later = ctrl.advance(lease_ms + hold_after + PERIOD_MS)
    assert later["executor"] == "HOLD"
    assert [event["kind"] for event in ctrl.drain_events()].count("hold_entered") == 0, (
        "HOLD 진입 사건이 주기마다 반복된다"
    )


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


def test_hold_procedure_ends_when_a_command_is_accepted():
    """HOLD 절차는 걸쇠가 아니다. 새 명령을 받으면 풀리고, 다음 만료는 다시 사건을 낸다."""
    lease_ms = LIFETIME["lease_ms"]
    hold_after = LIFETIME["hold_after_stale_ms"]
    ctrl = controller()
    ctrl.apply(move(seq=1, now_ms=0), now_ms=0)
    for now in range(0, lease_ms + hold_after + 1, PERIOD_MS):
        held = ctrl.advance(now)
    assert held["executor"] == "HOLD"
    assert held["holding_after_stale"] is True
    assert [event["kind"] for event in ctrl.drain_events()].count("hold_entered") == 1

    resumed_at = lease_ms + hold_after + PERIOD_MS
    assert ctrl.apply(move(seq=2, now_ms=resumed_at), now_ms=resumed_at)["applied"] is True
    moving = ctrl.advance(resumed_at)
    assert moving["executor"] == "MOVE_EE"
    assert moving["holding_after_stale"] is False, "HOLD 표시가 팔이 움직이는데도 남아 있다"

    second_lease = resumed_at + lease_ms
    for now in range(resumed_at, second_lease + hold_after + 1, PERIOD_MS):
        again = ctrl.advance(now)
    assert again["executor"] == "HOLD"
    assert [event["kind"] for event in ctrl.drain_events()].count("hold_entered") == 1, (
        "두 번째 HOLD 진입이 조용히 일어났다"
    )


def test_stop_during_deceleration_records_its_own_transition():
    """이미 감속 중이어도 새 정지는 그 자체로 전이다 — 사건과 ACK가 남아야 한다."""
    lease_ms = LIFETIME["lease_ms"]
    ctrl = controller()
    ctrl.apply(move(seq=1, now_ms=0), now_ms=0)
    for now in range(0, lease_ms + PERIOD_MS + 1, PERIOD_MS):
        decelerating = ctrl.advance(now)
    assert decelerating["stopping"] is True
    assert decelerating["executor"] == "MOVE_EE"
    ctrl.drain_events()

    stop_at = lease_ms + 2 * PERIOD_MS
    ack = ctrl.apply(move(seq=2, now_ms=stop_at, stop=True), now_ms=stop_at)
    events = ctrl.drain_events()

    transitions = [event for event in events if event["kind"] == "stop_transition"]
    assert len(transitions) == 1, f"감속 중 정지가 사건을 남기지 않았다: {events}"
    assert transitions[0]["cause"] == "command"
    assert [event["kind"] for event in events].count("hold_entered") == 1

    assert ack["executor"] == ctrl.advance(stop_at)["executor"] == "HOLD"


def test_stop_ack_matches_the_exec_record():
    """ACK가 말하는 실행기·경로는 실행 기록이 말할 것과 같아야 한다."""
    ctrl = controller()
    ack = ctrl.apply(move(seq=1, stop=True), now_ms=0)
    setpoint = ctrl.advance(0)
    assert ack["executor"] == setpoint["executor"]
    assert ack["path"] == ctrl.state_dict()["path_kind"]


def test_transition_collision_stop_records_its_event():
    ctrl = controller()
    ctrl.apply(move(seq=1, now_ms=0), now_ms=0)
    for i in range(BLEND_MS // PERIOD_MS + 1):
        ctrl.advance(i * PERIOD_MS)
    ctrl.drain_events()

    blocked = move(seq=2, now_ms=BLEND_MS, target_mm=(600, 300, 200))
    blocked["constraints"] = {"forbidden_segment": True}
    ack = ctrl.apply(blocked, now_ms=BLEND_MS)
    events = ctrl.drain_events()

    transitions = [event for event in events if event["kind"] == "stop_transition"]
    assert len(transitions) == 1
    assert transitions[0]["cause"] == "transition_collision"
    assert ack["executor"] == ctrl.advance(BLEND_MS)["executor"] == "HOLD"


def test_force_reflex_during_deceleration_escalates_to_hold():
    """lease 만료 감속 중에 힘 한계를 넘으면 그 자리에서 HOLD다 — 1초를 기다리지 않는다."""
    lease_ms = LIFETIME["lease_ms"]
    ctrl = controller()
    ctrl.apply(move(seq=1, now_ms=0), now_ms=0)
    for now in range(0, lease_ms + PERIOD_MS + 1, PERIOD_MS):
        ctrl.advance(now)
    ctrl.drain_events()

    ctrl.observe(sensors(contact_force_n=REFLEX["force_limit_n"] + 5.0))
    reflex = ctrl.advance(lease_ms + 2 * PERIOD_MS)

    assert reflex["executor"] == "HOLD"
    kinds = [event["kind"] for event in ctrl.drain_events()]
    assert kinds.count("reflex_stop") == 1
    assert kinds.count("hold_entered") == 1


def test_stop_survives_a_late_observation():
    """정지는 반사와 같은 우선순위다 — 수명 검사가 그것을 버리면 안 된다 (docs/08 §6)."""
    ctrl = controller()
    ctrl.apply(move(seq=1, now_ms=0), now_ms=0)
    for i in range(BLEND_MS // PERIOD_MS + 1):
        ctrl.advance(i * PERIOD_MS)

    late_at = 400
    stale_observation = late_at - LIFETIME["observation_deadline_ms"] - 1
    ack = ctrl.apply(
        move(seq=2, now_ms=late_at, observed_at=stale_observation, stop=True), now_ms=late_at
    )

    assert ack["applied"] is True
    assert ack["stop_transition"] is True
    assert ack["stale"] is True, "늦게 온 정지라는 사실은 기록돼야 한다"
    assert ack["reason"] == "observation_late"
    assert ctrl.advance(late_at)["stopping"] is True


def test_stop_survives_stale_geometry():
    ctrl = controller()
    ctrl.apply(move(seq=1, now_ms=0), now_ms=0)
    for i in range(BLEND_MS // PERIOD_MS + 1):
        ctrl.advance(i * PERIOD_MS)

    age = LIFETIME["geometry_age_static_ms"] + 1
    ack = ctrl.apply(move(seq=2, now_ms=BLEND_MS, geometry_age_ms=age, stop=True), now_ms=BLEND_MS)

    assert ack["applied"] is True
    assert ack["stop_transition"] is True
    assert ack["reason"] == "geometry_age"
    assert ctrl.advance(BLEND_MS)["stopping"] is True


def test_a_late_stop_does_not_refresh_the_lease():
    ctrl = controller()
    ctrl.apply(move(seq=1, now_ms=0), now_ms=0)
    lease = ctrl.state_dict()["lease_until"]

    late_at = 400
    ctrl.apply(
        move(seq=2, now_ms=late_at, observed_at=late_at - 300, stop=True), now_ms=late_at
    )
    assert ctrl.state_dict()["lease_until"] == lease


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


def test_force_reflex_during_a_stop_still_reports_the_event():
    """이미 멈추는 중이어도 힘 한계 초과는 사건이다 — 상태만 보고 기록을 건너뛰지 않는다."""
    ctrl = controller()
    ctrl.apply(move(seq=1, now_ms=0), now_ms=0)
    for i in range(BLEND_MS // PERIOD_MS + 1):
        ctrl.advance(i * PERIOD_MS)
    ctrl.apply(move(seq=2, now_ms=BLEND_MS, stop=True), now_ms=BLEND_MS)
    ctrl.drain_events()
    assert ctrl.advance(BLEND_MS)["stopping"] is True

    ctrl.observe(sensors(contact_force_n=REFLEX["force_limit_n"] + 5.0))
    ctrl.advance(BLEND_MS + PERIOD_MS)
    assert [event["kind"] for event in ctrl.drain_events()].count("reflex_stop") == 1


def test_force_reflex_reports_once_per_onset():
    ctrl = controller()
    ctrl.observe(sensors(contact_force_n=REFLEX["force_limit_n"] + 5.0))
    for i in range(4):
        ctrl.advance(i * PERIOD_MS)
    assert [event["kind"] for event in ctrl.drain_events()].count("reflex_stop") == 1


def test_stop_holds_the_grasp():
    """docs/08 §6 "정지 전이 … 파지 중이면 유지" — 설정의 stop.hold_grasp를 실제로 따른다."""
    assert CONFIG["stop"]["hold_grasp"] is True
    ctrl = controller(gripper_mm=GRIPPER["closed_mm"], holding="o3")
    ctrl.apply(move(seq=1, gripper="closed", target_distance_mm=0.0), now_ms=0)

    # 감속하면서 물체를 놓으라는 명령은 그대로 떨어뜨리는 일이다.
    stopping = ctrl.apply(move(seq=2, now_ms=PERIOD_MS, stop=True, gripper="open"), now_ms=PERIOD_MS)

    assert stopping["applied"] is True
    assert stopping["stop_transition"] is True
    assert stopping["gripper_event"] is None
    assert stopping["gripper_wait"] == "stop_holds_grasp"
    assert ctrl.advance(PERIOD_MS)["gripper"] == "closed"


def test_stop_tick_does_not_apply_the_gripper_answer():
    """docs/08 §5 1항: 정지가 참이면 **다른 답은 이 틱에 적용하지 않는다**.

    파지 중이 아니어서 `hold_grasp`가 걸리지 않을 때도 마찬가지다 — 정지 틱의 그리퍼 답은
    다음 틱에 다시 판단한다.
    """
    ctrl = controller(target_distance_mm=0.0)
    ack = ctrl.apply(move(seq=1, stop=True, gripper="closed"), now_ms=0)

    assert ack["applied"] is True
    assert ack["stop_transition"] is True
    assert ack["gripper_event"] is None
    assert ack["gripper_wait"] == "stop_tick"
    assert ctrl.advance(0)["gripper"] == "open", "정지 틱에 그리퍼가 움직였다"
    assert [event["kind"] for event in ctrl.drain_events()].count("gripper_wait") == 1


def test_the_tick_after_a_stop_applies_the_gripper_answer():
    """보류지 폐기가 아니다 — 정지 틱이 지나면 같은 답이 그대로 실행된다."""
    ctrl = controller(target_distance_mm=0.0)
    ctrl.apply(move(seq=1, stop=True, gripper="closed"), now_ms=0)
    ctrl.advance(0)

    resumed = ctrl.apply(move(seq=2, now_ms=PERIOD_MS, gripper="closed"), now_ms=PERIOD_MS)
    assert resumed["gripper_event"] is not None
    assert ctrl.advance(PERIOD_MS)["gripper"] == "closed"


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


def test_via_without_a_waypoint_is_rejected():
    """경유점 없는 via를 직선으로 바꾸지 않는다 — 하네스가 피하려던 바로 그 경로다."""
    ctrl = controller()
    command = move(seq=1)
    command["path"] = {"kind": "via", "target_ref": "o1", "target_mm": [600, 0, 200]}

    ack = ctrl.apply(command, now_ms=0)

    assert ack["applied"] is False
    assert ack["rejected"] is True
    assert ack["reason"] == "waypoint_missing"
    assert ctrl.advance(PERIOD_MS)["ee_pos_mm"] == pytest.approx(START_MM)


def test_via_with_a_waypoint_moves_to_the_waypoint():
    ctrl = controller()
    command = move(seq=1)
    command["path"] = {
        "kind": "via",
        "target_ref": "o1",
        "target_mm": [600, 0, 200],
        "waypoint_mm": [450, 120, 260],
    }
    ack = ctrl.apply(command, now_ms=0)
    assert ack["applied"] is True

    for i in range(1, 40):
        setpoint = ctrl.advance(i * PERIOD_MS)
    assert setpoint["ee_pos_mm"][1] > START_MM[1], "경유점 쪽으로 가지 않았다"


def test_retreat_moves_along_the_configured_vector():
    """retreat은 제자리 유지가 아니라 실제 후퇴다. 기록된 실행기가 움직임과 맞아야 한다."""
    ctrl = controller()
    # 후퇴가 끝날 때까지 도는 것을 보려고 lease를 길게 준다. 10Hz 재발행은
    # 하네스(3b)의 몫이고, 여기서 보는 것은 "기록된 실행기가 실제 움직임과 맞는가"다.
    command = move(seq=1, speed_level=2, lease_until=5000)
    command["path"] = {"kind": "retreat"}
    ack = ctrl.apply(command, now_ms=0)

    assert ack["applied"] is True
    assert ack["executor"] == "MOVE_EE", "움직이는데 HOLD라고 기록했다"
    assert ack["path"] == "retreat"

    for i in range(1, 60):
        setpoint = ctrl.advance(i * PERIOD_MS)

    moved = [after - before for after, before in zip(setpoint["ee_pos_mm"], START_MM)]
    distance = sum(value * value for value in moved) ** 0.5
    assert distance == pytest.approx(RETREAT["distance_mm"], rel=1e-6)

    vector = RETREAT["vector_mm"]
    length = sum(value * value for value in vector) ** 0.5
    expected = [value / length * RETREAT["distance_mm"] for value in vector]
    assert moved == pytest.approx(expected)


def test_retreat_out_of_reach_is_rejected_not_silently_frozen():
    """후퇴가 도달 범위를 벗어나면 거절한다. 조용히 제자리에 두지 않는다."""
    ctrl = controller()
    # 후퇴는 위로 드는 방향이므로, 이미 높은 곳에서는 도달 반경을 넘는다.
    ctrl.observe(sensors(ee_pos_mm=[0.0, 0.0, REACH["workspace_radius_mm"] - 20.0]))
    command = move(seq=1)
    command["path"] = {"kind": "retreat"}

    ack = ctrl.apply(command, now_ms=0)
    assert ack["applied"] is False
    assert ack["rejected"] is True
    assert ack["reason"] == "unreachable"


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


@pytest.mark.parametrize(
    ("override", "reason"),
    [
        ({"force_level": "crush"}, "invalid_force_level"),
        ({"speed_level": 9}, "invalid_speed_level"),
        ({"speed_level": "fast"}, "invalid_speed_level"),
        ({"gripper": "ajar"}, "invalid_gripper"),
    ],
)
def test_invalid_fields_get_their_own_reason(override, reason):
    """무엇이 틀렸는지 사유로 구분한다 — 전부 `unknown_executor`로 뭉뚱그리지 않는다."""
    ctrl = controller()
    ack = ctrl.apply(move(seq=1, **override), now_ms=0)
    assert ack["rejected"] is True
    assert ack["reason"] == reason


def test_invalid_path_kind_gets_its_own_reason():
    ctrl = controller()
    command = move(seq=1)
    command["path"] = {"kind": "teleport"}
    assert ctrl.apply(command, now_ms=0)["reason"] == "invalid_path"


def test_primitive_defaults_come_from_config():
    """원시 명령의 기본 속도 수준은 코드가 아니라 설정이 정한다."""
    ctrl = controller()
    ctrl.apply({"kind": "MOVE_EE", "target_mm": [600, 0, 200]}, now_ms=0)
    for i in range(BLEND_MS // PERIOD_MS + 1):
        ctrl.advance(i * PERIOD_MS)
    moving = SPEEDS[DEFAULTS["speed_level_moving"]] * 1000.0
    assert ctrl.advance(BLEND_MS)["speed_cap_mm_s"] == pytest.approx(moving)

    still = controller()
    still.apply({"kind": "HOLD", "duration_ms": 100}, now_ms=0)
    still_cap = SPEEDS[DEFAULTS["speed_level_still"]] * 1000.0
    assert still.advance(PERIOD_MS)["speed_cap_mm_s"] == pytest.approx(still_cap)


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


def test_state_dict_holds_no_infinities():
    """`Infinity`는 표준 JSON이 아니다. "장애물 없음"은 null로 적는다."""
    import json

    ctrl = Controller.from_config_path(CONTROLLER_CONFIG)
    ctrl.reset(ee_pos_mm=START_MM, ee_quat=START_QUAT, gripper_mm=GRIPPER["open_mm"], now_ms=0)
    assert ctrl.state_dict()["sensors"]["nearest_obstacle_mm"] is None

    ctrl.observe(sensors(nearest_obstacle_mm=None))
    ctrl.advance(PERIOD_MS)
    text = json.dumps(ctrl.state_dict(), allow_nan=False)

    def reject(constant):
        raise AssertionError(f"JSON에 {constant}가 들어 있다")

    restored = json.loads(text, parse_constant=reject)
    assert restored["sensors"]["nearest_obstacle_mm"] is None


def test_quaternion_order_is_declared_and_checked():
    """설정의 frame.quaternion_order를 코드가 실제로 확인한다 (죽은 키를 두지 않는다)."""
    import copy as copy_module

    assert CONFIG["frame"]["quaternion_order"] == "xyzw"
    broken = copy_module.deepcopy(CONFIG)
    broken["frame"]["quaternion_order"] = "wxyz"
    with pytest.raises(ValueError, match="quaternion"):
        Controller(broken)
