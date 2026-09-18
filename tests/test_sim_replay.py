"""E0/E1 환경 검사 — reset·step·snapshot·restore (docs/05 §2, docs/08 §3.2).

계획서의 snapshot 검사를 그대로 두고, 그 둘레에 "같은 seed면 같은 장면", "외란은
모의 시간으로 정해진다", "지시는 예정된 시각에 바뀐다", "관측은 mm 정수 스키마다"를
더한다. 마지막 하나는 실제 물리에서 짧은 E0 에피소드를 돌린다.
"""

import math

import numpy as np
import pytest
import yaml
from helpers import SIM_CONFIG

from robo_jev.sim.environment import Environment

CONFIG = yaml.safe_load(SIM_CONFIG.read_text(encoding="utf-8"))
PERIOD_MS = 1000 // CONFIG["simulator"]["control_hz"]
HOLD = {"kind": "HOLD", "duration_ms": 100}


@pytest.fixture
def env():
    environment = Environment(config_path=str(SIM_CONFIG))
    yield environment
    environment.close()


@pytest.fixture
def e0():
    environment = Environment(config_path=str(SIM_CONFIG), profile="E0")
    yield environment
    environment.close()


# --------------------------------------------------------------------------
# 계획서의 검사 (task-3-brief.md). 문구를 바꾸지 않는다.
# --------------------------------------------------------------------------


def test_snapshot_restores_controller_and_rng():
    env = Environment(config_path="configs/sim/tidy_clutter.yaml")
    env.reset(seed=17)
    state = env.snapshot()
    command = {"kind": "HOLD", "duration_ms": 100}
    a = env.step(command)
    env.restore(state)
    b = env.step(command)
    np.testing.assert_allclose(a["qpos"], b["qpos"], atol=1e-8)
    assert a["sim_time_ms"] == b["sim_time_ms"]
    assert a["disturbance_log"] == b["disturbance_log"]


# --------------------------------------------------------------------------
# reset — seed가 장면과 일정을 정한다
# --------------------------------------------------------------------------


def test_reset_is_deterministic_for_a_seed(env):
    first = env.reset(seed=41)
    first_plan = env.plan.to_json()
    second = env.reset(seed=41)

    assert env.plan.to_json() == first_plan
    assert second["objects"] == first["objects"]
    assert second["zones"] == first["zones"]
    assert second["instruction"] == first["instruction"]
    np.testing.assert_allclose(second["qpos"], first["qpos"], atol=1e-12)


def test_different_seeds_give_different_scenes(env):
    env.reset(seed=41)
    one = env.plan.to_json()
    env.reset(seed=42)
    assert env.plan.to_json() != one


def test_scene_matches_the_task_description(env):
    """docs/02 §1: 물체 6~10개, 취약·금지 속성, 목표 영역."""
    observation = env.reset(seed=5)
    objects = observation["objects"]
    assert CONFIG["objects"]["count_min"] <= len(objects) <= CONFIG["objects"]["count_max"]

    attributes = [attribute for obj in objects for attribute in obj["attributes"]]
    assert attributes.count("fragile") >= CONFIG["objects"]["fragile_count"][0]
    assert attributes.count("forbidden") >= CONFIG["objects"]["forbidden_count"][0]

    assert len({obj["id"] for obj in objects}) == len(objects)
    assert len({obj["colour"] for obj in objects}) == len(objects)
    assert CONFIG["zones"]["count_min"] <= len(observation["zones"]) <= CONFIG["zones"]["count_max"]


def test_instruction_change_is_scheduled_inside_the_window(env):
    """docs/02 §1: 에피소드 도중(5~15초)에 지시가 바뀐다."""
    low, high = CONFIG["instruction"]["change_window_ms"]
    for seed in range(8):
        env.reset(seed=seed)
        changes = [step for step in env.plan.instructions if step.version > 1]
        assert len(changes) == 1
        assert low <= changes[0].sim_ms <= high


def test_e0_is_the_static_reference(e0):
    """docs/05 §2 표: E0은 정적 장면이고 지시 변경·외란이 없다."""
    observation = e0.reset(seed=3)
    assert len(observation["objects"]) == CONFIG["profiles"]["E0"]["objects"]["count_min"]
    assert env_instruction_versions(e0) == [1]
    assert e0.plan.disturbances == ()

    for _ in range(50):
        observation = e0.step(HOLD)
    assert observation["disturbance_log"] == []
    assert observation["instruction"]["version"] == 1


def env_instruction_versions(environment) -> list[int]:
    return [step.version for step in environment.plan.instructions]


# --------------------------------------------------------------------------
# 외란과 지시 변경은 모의 시간으로 정해진다 (docs/05 §2)
# --------------------------------------------------------------------------


def test_disturbances_fire_by_sim_time_not_by_policy_calls(env):
    """정책 호출 횟수가 달라도 외란은 같은 모의 시각에 난다."""
    seed = 11
    env.reset(seed=seed)
    horizon = env.plan.disturbances[0].sim_ms + 400
    ticks = horizon // PERIOD_MS

    env.reset(seed=seed)
    for _ in range(ticks):
        held = env.step(HOLD)

    env.reset(seed=seed)
    for index in range(ticks):
        # 명령을 매 틱 내지 않고, 일부는 명령 없이 주기만 돌린다.
        if index % 3 == 0:
            varied = env.step(None)
        elif index % 3 == 1:
            varied = env.step({"kind": "OBSERVE", "duration_ms": 100})
        else:
            varied = env.step(HOLD)

    assert held["disturbance_log"], "검사 구간에 외란이 하나도 없다"
    assert schedule_of(varied) == schedule_of(held)


def schedule_of(observation: dict) -> list[tuple[int, str]]:
    return [(entry["sim_ms"], entry["object"]) for entry in observation["disturbance_log"]]


def test_disturbance_moves_the_object_and_reports_an_event(env):
    env.reset(seed=11)
    first = env.plan.disturbances[0]
    before = None
    observation = None
    while (observation or {}).get("sim_time_ms", 0) < first.sim_ms:
        if observation is not None:
            before = pose_of(observation, first.object)
        observation = env.step(HOLD)

    assert observation["sim_time_ms"] == first.sim_ms
    entry = observation["disturbance_log"][0]
    assert entry["sim_ms"] == first.sim_ms
    assert entry["object"] == first.object
    assert [event["kind"] for event in observation["events"]].count("disturbance_applied") == 1

    after = pose_of(observation, first.object)
    assert abs(after[0] - before[0]) + abs(after[1] - before[1]) > 1


def pose_of(observation: dict, object_id: str) -> list[int]:
    return next(obj["pos_mm"] for obj in observation["objects"] if obj["id"] == object_id)


def test_instruction_changes_at_the_scheduled_time(env):
    env.reset(seed=6)
    change = next(step for step in env.plan.instructions if step.version > 1)

    observation = env.step(HOLD)
    while observation["sim_time_ms"] < change.sim_ms:
        assert observation["instruction"]["version"] == 1
        observation = env.step(HOLD)

    assert observation["sim_time_ms"] == change.sim_ms
    assert observation["instruction"]["version"] == 2
    assert observation["instruction"]["text"] == change.text
    assert [event["kind"] for event in observation["events"]].count("instruction_changed") == 1


# --------------------------------------------------------------------------
# 관측 스키마 (docs/08 §3.2)
# --------------------------------------------------------------------------


def test_observation_follows_the_state_schema(env):
    observation = env.reset(seed=5)

    assert isinstance(observation["sim_time_ms"], int)
    assert isinstance(observation["tick"], int)
    assert observation["episode_over"] is False
    assert observation["instruction"]["version"] == 1
    assert observation["disturbance_log"] == []

    for obj in observation["objects"]:
        assert all(isinstance(value, int) for value in obj["pos_mm"])
        assert all(isinstance(value, int) for value in obj["obb_mm"])
        assert len(obj["quat"]) == 4
        assert all(value == round(value, 2) for value in obj["quat"])
        assert isinstance(obj["visible"], bool)
        assert 0.0 <= obj["visible_ratio"] <= 1.0
        assert obj["shape"] in CONFIG["objects"]["shapes"]
        assert set(obj["attributes"]) <= {"fragile", "forbidden"}
        assert isinstance(obj["last_seen_ms"], int)

    for zone in observation["zones"]:
        assert len(zone["bounds_mm"]) == 4
        assert all(isinstance(value, int) for value in zone["bounds_mm"])
        assert zone["desc"]

    robot = observation["robot"]
    assert all(isinstance(value, int) for value in robot["ee_pos_mm"])
    assert len(robot["ee_quat"]) == 4
    assert isinstance(robot["gripper_mm"], int)
    assert robot["holding"] is None
    assert isinstance(robot["contact_force_n"], float)
    assert isinstance(robot["speed_mm_s"], int)


def test_visibility_is_measured_not_assumed(env):
    """가림은 참값이 아니라 시점에서 실제로 쏜 광선으로 정한다."""
    ratios = [obj["visible_ratio"] for seed in range(6) for obj in env.reset(seed=seed)["objects"]]
    assert {ratio == 1.0 for ratio in ratios} == {True, False}
    partial = [ratio for ratio in ratios if 0.0 < ratio < 1.0]
    assert partial, "부분 가림이 한 번도 없다 — 광선이 아니라 상수를 쓰고 있을 수 있다"


def test_moving_the_arm_away_uncovers_what_it_hid(env):
    """가시성이 팔의 실제 자세에 달렸는지 본다 — 팔을 치우면 가렸던 물체가 보인다."""
    observation = env.reset(seed=0)
    hidden = [obj["id"] for obj in observation["objects"] if obj["visible_ratio"] == 0.0]
    assert hidden, "seed 0에서 팔에 가려진 물체가 없다"

    aside = {"kind": "MOVE_EE", "target_mm": [320, 340, 260], "speed_level": 3}
    for tick in range(80):
        observation = env.step(aside if tick % 5 == 0 else None)

    uncovered = {obj["id"]: obj["visible_ratio"] for obj in observation["objects"]}
    assert any(uncovered[object_id] > 0.0 for object_id in hidden), (
        f"팔을 치웠는데 가렸던 물체가 그대로다: {[(i, uncovered[i]) for i in hidden]}"
    )


def test_ack_is_reported_for_every_command(env):
    env.reset(seed=5)
    observation = env.step(HOLD)
    assert observation["ack"]["applied"] is True
    assert observation["ack"]["executor"] == "HOLD"
    assert observation["exec"]["seq"] == observation["ack"]["seq"]
    assert env.step(None)["ack"] is None


def test_gripper_readiness_uses_the_real_distance_to_the_target(env):
    """close readiness는 명령이 가리키는 물체까지의 실제 거리로 판정한다 (docs/08 §4)."""
    observation = env.reset(seed=5)
    far = max(
        observation["objects"],
        key=lambda obj: math.dist(obj["pos_mm"], observation["robot"]["ee_pos_mm"]),
    )
    close = {"kind": "SET_GRIPPER", "gripper": "closed", "target_ref": far["id"]}

    waiting = env.step(close)
    assert waiting["ack"]["gripper_event"] is None
    assert waiting["ack"]["gripper_wait"] == "readiness"
    assert [event["kind"] for event in waiting["events"]].count("gripper_wait") == 1


# --------------------------------------------------------------------------
# snapshot / restore
# --------------------------------------------------------------------------


def test_snapshot_restores_mid_episode_with_events_and_schedules(env):
    env.reset(seed=11)
    first = env.plan.disturbances[0]
    ticks_to_disturbance = first.sim_ms // PERIOD_MS
    for _ in range(max(0, ticks_to_disturbance - 4)):
        env.step(HOLD)

    state = env.snapshot()
    forward = [env.step(HOLD) for _ in range(12)]
    env.restore(state)
    again = [env.step(HOLD) for _ in range(12)]

    for left, right in zip(forward, again):
        np.testing.assert_allclose(left["qpos"], right["qpos"], atol=1e-12)
        assert left["sim_time_ms"] == right["sim_time_ms"]
        assert left["disturbance_log"] == right["disturbance_log"]
        assert left["events"] == right["events"]
        assert left["objects"] == right["objects"]
        assert left["robot"] == right["robot"]
    assert forward[-1]["disturbance_log"], "외란이 없는 구간을 검사했다"


def test_restore_rewinds_the_controller_lease(env):
    env.reset(seed=5)
    env.step({"kind": "MOVE_EE", "target_mm": [500, 40, 200]})
    state = env.snapshot()
    before = env.controller.state_dict()

    for _ in range(40):
        env.step(None)
    assert env.controller.state_dict()["stale"] is True

    env.restore(state)
    assert env.controller.state_dict() == before


def test_a_fresh_environment_can_restore_without_reset(env):
    """rollout은 snapshot만 받아 이어 달린다 — reset을 먼저 부르지 않는다."""
    env.reset(seed=23)
    for _ in range(20):
        env.step(HOLD)
    state = env.snapshot()
    forward = [env.step(HOLD) for _ in range(5)]

    cold = Environment(config_path=str(SIM_CONFIG))
    try:
        cold.restore(state)
        again = [cold.step(HOLD) for _ in range(5)]
    finally:
        cold.close()

    for left, right in zip(forward, again):
        np.testing.assert_allclose(left["qpos"], right["qpos"], atol=1e-12)
        assert left["objects"] == right["objects"]
        assert left["disturbance_log"] == right["disturbance_log"]


def test_snapshot_is_bytes_and_carries_every_rng(env):
    env.reset(seed=5)
    state = env.snapshot()
    assert isinstance(state, bytes)
    parts = env.describe_snapshot(state)
    assert {"mujoco", "controller", "schedules", "wrapper", "rng", "osc"} <= set(parts)
    assert {"scene", "numpy_global", "python"} <= set(parts["rng"])


# --------------------------------------------------------------------------
# 실제 물리에서의 짧은 E0 에피소드
# --------------------------------------------------------------------------


def test_short_e0_episode_reaches_the_target_and_replays(e0):
    observation = e0.reset(seed=2)
    start = list(observation["robot"]["ee_pos_mm"])
    target = [start[0] + 60, start[1] + 40, start[2] - 30]

    def command(seq: int, now_ms: int) -> dict:
        return {
            "seq": seq,
            "observed_at": now_ms,
            "issued_at": now_ms,
            "action_ref": "c1",
            "phase": "approach",
            "path": {"kind": "direct", "target_ref": "o0", "target_mm": target},
            "speed_level": 2,
            "force_level": "light",
            "gripper": "open",
            "stop": False,
            "geometry_age_ms": 40,
            "target_moving": False,
        }

    # 하네스는 10Hz로 명령을 낸다 — 다섯 주기마다 하나이고 lease 300ms가 그것을 덮는다.
    seq = 0
    state = None
    trace = []
    for tick in range(60):
        issued = None
        if tick % 5 == 0:
            seq += 1
            issued = command(seq, observation["sim_time_ms"])
        observation = e0.step(issued)
        trace.append(list(observation["robot"]["ee_pos_mm"]))
        if tick == 29:
            state = e0.snapshot()

    reached = observation["robot"]["ee_pos_mm"]
    error = max(abs(a - b) for a, b in zip(reached, target))
    assert error < 15, f"말단이 목표에 닿지 않았다: {reached} vs {target}"
    assert observation["ack"] is None or observation["ack"]["rejected"] is False

    # 같은 snapshot에서 같은 명령을 다시 내면 같은 궤적이 나온다.
    e0.restore(state)
    replay = []
    seq = 6
    resumed = {"sim_time_ms": 30 * PERIOD_MS}
    for tick in range(30, 60):
        issued = None
        if tick % 5 == 0:
            seq += 1
            issued = command(seq, resumed["sim_time_ms"])
        resumed = e0.step(issued)
        replay.append(list(resumed["robot"]["ee_pos_mm"]))

    assert replay == trace[30:]
