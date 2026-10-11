"""갈라 굴리기: 선택지마다 × 세계 K개마다 3초 굴림 → 결과 (docs/12 §2 ④, Task v0a 지시서 §3.3·§3.4).

입력은 관측(로봇 자신의 사양·알려진 기하)과 세계 batch(사후 표본 K개, 또는 참 세계 1개)다. 세계 모듈을 import하지 않는다 — 참 세계도
:meth:`World.as_batch` 로 같은 batch가 되어 같은 함수를 지난다(그래서 `truth`의 결과와 정답의 결과가 같은 물리에서 나온다).

선택지(정식 열쇠):
``continue`` 등속 · ``slow`` gentle로 slow_ratio까지 · ``stop_gentle|moderate|hard`` 그 감속으로 멈춤 · ``sidestep_left|right`` gentle로
속도를 줄이며 옆으로 비킴 · ``retract``(팔) moderate로 멈춘 뒤 뒤로 물러남 · ``go_now`` / ``creep`` / ``wait``(멈춘 로봇) ·
``turn_100`` … ``turn_30``(회전 속도 = 지금 속도의 비율).

결과: 사건(닿음 contact, 옆 장애물 닿음 side_contact, 쏟아짐 spill, 장애물 넘어섬 overrun)은 (A, K) bool, 양(진행 손실 delay 초,
가까움 closeness, 빈 높이 사용 headroom)은 (A, K) float, 선택지와 무관한 세계 사건(world_events)과 상태(states)는 (K,).
쏟아짐: 정적 액면 근사로 넘치거나, 닿음·넘어섬의 충격이 있고 뚜껑이 없으면 쏟아진 것으로 센다(v0 근사).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from robo_jev.simgen import physics as ph
from robo_jev.simgen.catalog import SimgenConfig
from robo_jev.simgen.observation import Observation
from robo_jev.simgen.spec import Batch, RobotSpec, fill_bins

__all__ = ["STOP_LEVELS", "TURN_RATIOS", "Outcomes", "available_options", "max_accel", "rollout", "time_grid", "turn_profile"]

STOP_LEVELS = {"stop_gentle": "gentle", "stop_moderate": "moderate", "stop_hard": "hard"}
TURN_RATIOS = {"turn_100": 1.0, "turn_80": 0.8, "turn_60": 0.6, "turn_45": 0.45, "turn_30": 0.3}
_WALL_GAP = 0.05


@dataclass
class Outcomes:
    options: tuple[str, ...]
    events: dict[str, np.ndarray] = field(default_factory=dict)
    quantities: dict[str, np.ndarray] = field(default_factory=dict)
    world_events: dict[str, np.ndarray] = field(default_factory=dict)
    states: dict[str, np.ndarray] = field(default_factory=dict)
    cost_items: dict[str, str] = field(default_factory=dict)   # 사건 → 가치표의 비용 항목

    def index(self, option: str) -> int:
        return self.options.index(option)


def time_grid(cfg: SimgenConfig) -> np.ndarray:
    h, dt = cfg.catalog["horizon_s"], cfg.catalog["dt_s"]
    return np.arange(0.0, h + dt / 2.0, dt)


def available_options(obs: Observation) -> tuple[str, ...]:
    """그 장면·몸·관측에서 가능한 선택지(정식 열쇠)."""
    r = obs.robot
    arm = obs.body == "arm"
    base = ("continue", "slow", "stop_gentle", "stop_moderate", "stop_hard")
    if obs.scene in ("p1_step_in", "p1p5_step_in_carrying"):
        opts = list(base) + (["retract"] if arm else [])
        if r.can_sidestep and obs.env.sides is not None:
            opts += [f"sidestep_{side}" for side, s in zip(("left", "right"), obs.env.sides, strict=True) if s.status != "blocked"]
        return tuple(opts)
    if obs.scene == "p1_crossing":
        if obs.variant == "stopped":
            return ("go_now", "creep", "wait")
        return base + (("retract",) if arm else ())
    if obs.scene == "p1_head_on":
        return base + (("sidestep_left", "sidestep_right") if r.can_sidestep else ())
    if obs.scene == "p5_braking":
        return ("continue", "stop_gentle", "stop_moderate", "stop_hard")
    if obs.scene == "p5_turning":
        return tuple(TURN_RATIOS)
    raise ValueError(obs.scene)


def max_accel(option: str, r: RobotSpec) -> float:
    """선택지가 쓰는 가장 큰 가속(쏟아짐 판정용, m/s²)."""
    if r.speed <= 0.0 and option in ("go_now", "creep", "wait"):
        return {"go_now": r.accel, "creep": min(r.accel, r.decel_gentle), "wait": 0.0}[option]
    if option == "continue":
        return 0.0
    if option == "slow":
        return r.decel_gentle
    if option in STOP_LEVELS:
        return r.decel(STOP_LEVELS[option])
    if option == "retract":
        return r.decel_moderate
    if option.startswith("sidestep"):
        return float(np.hypot(r.decel_gentle, ph.smoothstep_peak_accel(r.sidestep_offset, r.sidestep_duration)))
    raise ValueError(option)


def _longitudinal(option: str, r: RobotSpec, t: np.ndarray) -> np.ndarray:
    """로봇의 진행 방향 위치 x(t) (T,)."""
    v0, lat = r.speed, r.latency
    if option == "continue":
        return v0 * t
    if option == "slow":
        return ph.ramp_position(t, v0, lat, r.decel_gentle, r.slow_ratio * v0)
    if option in STOP_LEVELS:
        return ph.ramp_position(t, v0, lat, r.decel(STOP_LEVELS[option]), 0.0)
    if option == "retract":
        t_stop = lat + v0 / r.decel_moderate
        return ph.ramp_position(t, v0, lat, r.decel_moderate, 0.0) + ph.ramp_position(t, 0.0, t_stop, r.decel_gentle, -r.retract_speed)
    if option.startswith("sidestep"):
        return ph.ramp_position(t, v0, lat, r.decel_gentle, r.sidestep_speed_ratio * v0)
    if option == "go_now":
        return ph.ramp_position(t, 0.0, lat, r.accel, r.cruise)
    if option == "creep":
        return ph.ramp_position(t, 0.0, lat, min(r.accel, r.decel_gentle), r.creep_ratio * r.cruise)
    if option == "wait":
        return np.zeros_like(t)
    raise ValueError(option)


def _delay(x_end: float | np.ndarray, r: RobotSpec, horizon: float) -> np.ndarray:
    """진행 손실(초): 기준(등속 또는 바로 출발)보다 지평 끝에서 덜 간 거리 / 기준 속도."""
    if r.speed > 0:
        return (r.speed * horizon - x_end) / r.speed
    reference = ph.ramp_position(np.array([horizon]), 0.0, r.latency, r.accel, r.cruise)[0]
    return (reference - x_end) / r.cruise


def _closeness(min_dist: np.ndarray, r_sum: np.ndarray, arm: bool) -> np.ndarray:
    scale = 0.2 if arm else 1.0
    return np.clip(1.0 - (min_dist - r_sum) / scale, 0.0, 1.0)


def _container_outcomes(obs: Observation, batch: Batch, options, impact: np.ndarray | None) -> tuple[np.ndarray, np.ndarray]:
    """(spill (A, K), headroom (A, K)) — 정적 근사 + 충격(닿음·넘어섬)."""
    c, r = obs.container, obs.robot
    spill, headroom = [], []
    for i, option in enumerate(options):
        a = max_accel(option, r)
        rise = ph.liquid_rise(c.diameter, a, batch["k"], batch["tilt"])
        static = ph.spills(batch["fill"], c.height, c.diameter, a, batch["k"], batch["tilt"], batch["lid"] > 0.5)
        hit = impact[i] & (batch["lid"] < 0.5) if impact is not None else False
        spill.append(static | hit)
        headroom.append(ph.headroom_used(batch["fill"], c.height, rise))
    return np.array(spill), np.array(headroom)


def _fill_states(batch: Batch) -> dict[str, np.ndarray]:
    return {"fill_bin": fill_bins(batch["fill"]), "lid": (batch["lid"] > 0.5).astype(float)}


# ----------------------------------------------------------------------------------------------------------------


def _step_in(cfg: SimgenConfig, obs: Observation, batch: Batch, options) -> Outcomes:
    r, t = obs.robot, time_grid(cfg)
    horizon = float(t[-1])
    b = batch.values
    col = lambda name: b[name][:, None]  # noqa: E731
    tt = t[None, :]
    y_stay = col("lat0") + col("dir") * (
        ph.ramp_position(tt, col("u0"), 0.0, col("stop_decel"), 0.0) + ph.ramp_position(tt, 0.0, col("stay_s"), col("cross_accel"), col("cross_speed"))
    )
    y_cross = col("lat0") + col("dir") * ph.ramp_position(tt, col("u0"), 0.0, col("cross_accel"), col("cross_speed"))
    y_back = col("lat0") + col("dir") * ph.ramp_position(tt, col("u0"), 0.0, col("back_decel"), -col("back_speed"))
    roll = np.where(col("roll_decel") > 0, col("roll_decel"), 1.0)
    y_ball = col("lat0") + col("dir") * ph.ramp_position(tt, col("u0"), 0.0, roll, 0.0)
    intent = col("intent")
    y_a = np.where(intent == 0, y_stay, np.where(intent == 1, y_cross, np.where(intent == 2, y_back, y_ball)))
    x_a = col("gap")
    r_sum = r.radius + b["ra"]
    contact, side_contact, delay, closeness, impact = [], [], [], [], []
    for option in options:
        x_r = _longitudinal(option, r, t)[None, :]
        y_r = np.zeros_like(x_r)
        side_hit = np.zeros(batch.size, dtype=bool)
        if option.startswith("sidestep"):
            sign = 1.0 if option.endswith("left") else -1.0
            y_r = ph.smoothstep(t, r.latency, r.sidestep_duration, sign * r.sidestep_offset)[None, :]
            side_hit = b["side_left" if sign > 0 else "side_right"] > 0.5
        d_min = ph.min_distance(x_r, y_r, x_a, y_a)
        hit = d_min < r_sum
        contact.append(hit)
        side_contact.append(side_hit)
        delay.append(np.full(batch.size, _delay(float(x_r[0, -1]), r, horizon)))
        closeness.append(_closeness(d_min, r_sum, obs.body == "arm"))
        impact.append(hit | side_hit)
    out = Outcomes(tuple(options))
    out.events = {"contact": np.array(contact), "side_contact": np.array(side_contact)}
    out.quantities = {"delay": np.array(delay), "closeness": np.array(closeness)}
    out.cost_items = {"contact": cfg.agent(obs.agent.kind)["contact"], "side_contact": "contact_object"}
    stop_mod = ph.stopping_distance(r.speed, r.latency, r.decel_moderate)
    i2 = int(round(2.0 / cfg.catalog["dt_s"]))
    out.world_events = {
        "stops_short": b["gap"] - r_sum - stop_mod > 0.0,
        "in_lane_2s": np.abs(y_a[:, i2]) < r_sum,
        "side_free_left": b["side_left"] < 0.5,
        "side_free_right": b["side_right"] < 0.5,
    }
    if obs.agent.group == "ball":
        rest = b["lat0"] + b["dir"] * b["u0"] ** 2 / (2.0 * roll[:, 0])
        out.world_events["ball_stops_in_lane"] = np.abs(rest) < r_sum
    out.states = {"intent": b["intent"]}
    if obs.container is not None:
        spill, headroom = _container_outcomes(obs, batch, options, np.array(impact))
        out.events["spill"] = spill
        out.quantities["headroom"] = headroom
        out.cost_items["spill"] = f"spill_{obs.container.content}"
        out.states.update(_fill_states(batch))
    return out


def _crossing(cfg: SimgenConfig, obs: Observation, batch: Batch, options) -> Outcomes:
    r, t = obs.robot, time_grid(cfg)
    horizon = float(t[-1])
    b = batch.values
    col = lambda name: b[name][:, None]  # noqa: E731
    tt = t[None, :]
    r_sum = r.radius + b["ra"]
    stop_at = np.maximum(col("dist") - r_sum[:, None] - col("yield_standoff"), 0.02)
    yield_decel = np.minimum(col("va") ** 2 / (2.0 * stop_at), col("agent_decel"))
    s_cross = col("va") * tt
    s_yield = ph.ramp_position(tt, col("va"), 0.0, yield_decel, 0.0)
    s_hurry = ph.ramp_position(tt, col("va"), 0.0, col("hurry_accel"), col("hurry_factor") * col("va"))
    roll = np.where(col("roll_decel") > 0, col("roll_decel"), 1.0)
    s_ball = ph.ramp_position(tt, col("va"), 0.0, roll, 0.0)
    intent = col("intent")
    s = np.where(intent == 0, s_cross, np.where(intent == 1, s_yield, np.where(intent == 2, s_hurry, s_ball)))
    y_a = col("from_sign") * (col("dist") - s)
    x_a = col("ahead")
    contact, delay, closeness = [], [], []
    for option in options:
        x_r = _longitudinal(option, r, t)[None, :]
        d_min = ph.min_distance(x_r, 0.0, x_a, y_a)
        contact.append(d_min < r_sum)
        delay.append(np.full(batch.size, _delay(float(x_r[0, -1]), r, horizon)))
        closeness.append(_closeness(d_min, r_sum, obs.body == "arm"))
    out = Outcomes(tuple(options))
    out.events = {"contact": np.array(contact)}
    out.quantities = {"delay": np.array(delay), "closeness": np.array(closeness)}
    out.cost_items = {"contact": cfg.agent(obs.agent.kind)["contact"]}
    safety = cfg.catalog["margins"]["safety_s"]
    t_arr = (b["dist"] - r_sum) / b["va"]
    if obs.variant == "stopped":
        t_clear = ph.clear_time_from_rest(b["ahead"] + r_sum, r.latency, r.accel, r.cruise)
    else:
        t_clear = (b["ahead"] + r_sum) / r.speed
    out.world_events = {"clears_first": t_arr - (t_clear + safety) > 0.0}
    if obs.agent.group == "ball":
        out.world_events["ball_reaches_path"] = b["va"] ** 2 / (2.0 * roll[:, 0]) >= b["dist"] - r_sum
    out.states = {"intent": b["intent"]}
    return out


def _head_on(cfg: SimgenConfig, obs: Observation, batch: Batch, options) -> Outcomes:
    r, t = obs.robot, time_grid(cfg)
    dt = float(t[1] - t[0])
    horizon = float(t[-1])
    b = batch.values
    k = batch.size
    r_sum = r.radius + b["ra"]
    y0 = obs.env.robot_lateral
    width_obs = obs.env.corridor_width.value
    # 상대의 옆 위치 (K, T)
    target = np.where(b["intent"] == 1, 1.0, -1.0) * (b["width"] / 2.0 - b["ra"] - _WALL_GAP)
    step = np.minimum(np.abs(target - b["lat_a"])[:, None], b["aside_speed"][:, None] * t[None, :])
    aside = b["lat_a"][:, None] + np.sign(target - b["lat_a"])[:, None] * step
    is_aside = ((b["intent"] == 1) | (b["intent"] == 2))[:, None]
    y_a = np.where(is_aside, aside, b["lat_a"][:, None])
    ball = b["roll_decel"] > 0
    decel = np.where(ball, b["roll_decel"], b["agent_decel"])
    contact, delay, closeness, blocked = [], [], [], []
    for option in options:
        x_r = _longitudinal(option, r, t)
        y_r = np.full((k, len(t)), y0)
        if option.startswith("sidestep"):
            sign = 1.0 if option.endswith("left") else -1.0
            planned = sign * (width_obs / 2.0 - r.radius - _WALL_GAP)
            wall = b["width"] / 2.0 - r.radius - 0.02
            goal = np.clip(planned, -wall, wall)
            y_r = y0 + ph.smoothstep(t[None, :], r.latency, r.sidestep_duration, (goal - y0)[:, None])
        x_a = np.empty((k, len(t)))
        x_a[:, 0] = b["dist"]
        speed = b["va"].copy()
        braking = (b["intent"] == 3) | ball
        for i in range(1, len(t)):
            overlap = np.abs(y_r[:, i - 1] - y_a[:, i - 1]) < r_sum + 0.1
            gap = x_a[:, i - 1] - x_r[i - 1] - r_sum
            trigger = (b["reactive"] > 0.5) & overlap & (gap <= b["keep_standoff"] + speed**2 / (2.0 * decel))
            braking = braking | trigger
            new_speed = np.where(braking, np.maximum(speed - decel * dt, 0.0), speed)
            x_a[:, i] = x_a[:, i - 1] - 0.5 * (speed + new_speed) * dt
            speed = new_speed
        d_min = ph.min_distance(x_r[None, :], y_r, x_a, y_a)
        contact.append(d_min < r_sum)
        delay.append(np.full(k, _delay(float(x_r[-1]), r, horizon)))
        closeness.append(_closeness(d_min, r_sum, False))
        robot_still = abs(x_r[-1] - x_r[-2]) < 1e-9
        blocked.append(robot_still & (speed <= 1e-9) & (np.abs(y_r[:, -1] - y_a[:, -1]) < r_sum))
    out = Outcomes(tuple(options))
    out.events = {"contact": np.array(contact), "blocked": np.array(blocked)}
    out.quantities = {"delay": np.array(delay), "closeness": np.array(closeness)}
    out.cost_items = {"contact": cfg.agent(obs.agent.kind)["contact"]}
    out.world_events = {"passable": b["width"] - 2.0 * r_sum - r.passing_clearance >= 0.0}
    out.states = {"intent": b["intent"]}
    return out


def _braking(cfg: SimgenConfig, obs: Observation, batch: Batch, options) -> Outcomes:
    r = obs.robot
    b = batch.values
    k = batch.size
    overrun, delay = [], []
    for option in options:
        if option == "continue":
            overrun.append(np.ones(k, dtype=bool))
            delay.append(np.zeros(k))
            continue
        stop = ph.stopping_distance(r.speed, r.latency, r.decel(STOP_LEVELS[option]))
        overrun.append(stop + r.radius > b["obstacle"])
        delay.append(np.maximum(0.0, (b["obstacle"] - r.radius - 0.1) - stop) / r.speed)
    out = Outcomes(tuple(options))
    impact = np.array(overrun)
    spill, headroom = _container_outcomes(obs, batch, options, impact)
    out.events = {"overrun": impact, "spill": spill}
    out.quantities = {"delay": np.array(delay), "headroom": headroom}
    out.cost_items = {"overrun": "contact_object", "spill": f"spill_{obs.container.content}"}
    out.states = _fill_states(batch)
    return out


def turn_profile(r: RobotSpec, option: str, radius: float, angle_deg: float, approach: float) -> tuple[float, float]:
    """(지연 s, 쏟아짐 판정용 가장 큰 가속 m/s²) — 회전 속도 선택 하나."""
    v = TURN_RATIOS[option] * r.speed
    floor = float(np.sqrt(max(0.0, r.speed**2 - 2.0 * r.decel_hard * approach)))
    v_entry = max(v, floor)
    delay, a_long = ph.turn_delay(r.speed, v_entry, approach, radius, np.radians(angle_deg), r.decel_gentle, r.accel)
    return delay, max(a_long, v_entry**2 / radius)


def _turning(cfg: SimgenConfig, obs: Observation, batch: Batch, options) -> Outcomes:
    r, env, c = obs.robot, obs.env, obs.container
    b = batch.values
    k = batch.size
    spill, headroom, delay = [], [], []
    for option in options:
        d, a = turn_profile(r, option, env.turn_radius, env.turn_angle_deg, env.turn_approach.value)
        rise = ph.liquid_rise(c.diameter, a, b["k"], b["tilt"])
        spill.append(ph.spills(b["fill"], c.height, c.diameter, a, b["k"], b["tilt"], b["lid"] > 0.5))
        headroom.append(ph.headroom_used(b["fill"], c.height, rise))
        delay.append(np.full(k, d))
    out = Outcomes(tuple(options))
    out.events = {"spill": np.array(spill)}
    out.quantities = {"delay": np.array(delay), "headroom": np.array(headroom)}
    out.cost_items = {"spill": f"spill_{c.content}"}
    out.states = _fill_states(batch)
    return out


def rollout(cfg: SimgenConfig, obs: Observation, batch: Batch, options=None) -> Outcomes:
    """선택지 × 세계의 결과. options가 없으면 가능한 선택지 전부."""
    options = tuple(options or available_options(obs))
    if obs.scene in ("p1_step_in", "p1p5_step_in_carrying"):
        return _step_in(cfg, obs, batch, options)
    if obs.scene == "p1_crossing":
        return _crossing(cfg, obs, batch, options)
    if obs.scene == "p1_head_on":
        return _head_on(cfg, obs, batch, options)
    if obs.scene == "p5_braking":
        return _braking(cfg, obs, batch, options)
    if obs.scene == "p5_turning":
        return _turning(cfg, obs, batch, options)
    raise ValueError(obs.scene)
