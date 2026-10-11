"""갈라 굴리기: 선택지마다 × 세계 K개마다 3초 굴림 → 결과 (docs/12 §2 ④, Task v0a 지시서 §3.3·§3.4, 리뷰 1 반영).

입력은 관측(로봇 자신의 사양·알려진 기하)과 세계 batch(사후 표본 K개, 또는 참 세계 1개)다. 세계 모듈을 import하지 않는다 — 참 세계도
같은 batch가 되어 같은 함수를 지난다(그래서 `truth`의 결과와 정답의 결과가 같은 물리에서 나온다).

선택지(정식 열쇠):
``continue`` 등속 · ``slow`` gentle로 slow_ratio까지 · ``stop_gentle|moderate|hard`` 그 감속으로 멈춤 · ``sidestep_left|right`` gentle로
속도를 줄이며 옆으로 비킴 · ``retract``(팔) moderate로 멈춘 뒤 뒤로 물러남 · ``go_now`` / ``creep`` / ``wait``(멈춘 로봇) ·
``turn_100`` … ``turn_30``(회전 속도 = 지금 속도의 비율).

결과: 사건(닿음 contact, 옆 장애물 닿음 side_contact, 쏟아짐 spill, 장애물 넘어섬 overrun, 마주 섬 blocked, 액면이 림·뚜껑에 닿음
rim_reached)은 (A, K) bool, 양(진행 손실 delay 초, 가까움 closeness, 빈 높이 사용 headroom)은 (A, K) float, 닿음·넘어섬의 충격 속도
(impact, (A, K) m/s — 충돌 속도에 따른 닿음 비용 기제용), 급제동을 쓰나(hard_brake, (A,)), 선택지와 무관한 세계 사건(world_events)과
상태(states)는 (K,).

* 가장 가까운 거리는 표본 사이를 직선으로 이은 구간마다 찾는다(0.05 s 간격이 팔·손에서 지나가는 가까움을 놓치지 않게, 리뷰 1 m5).
* 가로지르는 상대(사람·로봇·손)는 로봇이 자기 줄을 막고 있으면 늦게라도 선다 — 숨은 `reacts`(단서에 묶인 확률)일 때(리뷰 1 질문 4).
* 쏟아짐: 정적 액면 근사로 넘치거나, 충격 — 상대와의 닿음(contact), 비켜서다 옆의 숨은 장애물에 닿음(side_contact), 장애물 넘어섬
  (overrun) — 이 있고 뚜껑이 없으면 쏟아진 것으로 센다(v0 근사; 옆 장애물 닿음은 Task v0b에서 되살렸다 — 리뷰 1 재검증 m-b).
* 회전: 접근 구간 전체에 걸친 필요한 감속으로 줄인다 — 편안함 하한과 최대 감속 사이(리뷰 1 I2).
* p5_braking의 지연: 목표(장애물 앞, 몸별 허용치) 앞에 일찍 서면 남은 거리를 gentle로 다시 다가가는 시간(삼각 속도 꼴 2·√(d/a))이다 —
  급제동이 '공짜'가 아니다(리뷰 1 I3).
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
    impact: dict[str, np.ndarray] = field(default_factory=dict)  # 사건 → 충격 속도 (A, K)
    hard_brake: np.ndarray | None = None                        # (A,) 급제동을 쓰는 선택지
    body: str = ""                                              # 비용표의 몸별 값(급제동 비용·v_ref)을 고르려고
    loaded: bool = False

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


def _hard_brake(options) -> np.ndarray:
    return np.array([option == "stop_hard" for option in options], dtype=bool)


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
    """진행 손실(초): 기준(등속 또는 바로 출발)보다 지평 끝에서 덜 간 거리 / 기준 속도 — 앞으로 3초 동안 잃는 진행이다."""
    if r.speed > 0:
        return (r.speed * horizon - x_end) / r.speed
    reference = ph.ramp_position(np.array([horizon]), 0.0, r.latency, r.accel, r.cruise)[0]
    return (reference - x_end) / r.cruise


def _closeness(min_dist: np.ndarray, r_sum: np.ndarray, arm: bool) -> np.ndarray:
    scale = 0.2 if arm else 1.0
    return np.clip(1.0 - (min_dist - r_sum) / scale, 0.0, 1.0)


def _container_outcomes(obs: Observation, batch: Batch, options, impact: np.ndarray | None, accels=None):
    """(spill, headroom, rim_reached) 각각 (A, K) — 정적 근사 + 충격(닿음·넘어섬). rim_reached는 뚜껑과 무관하게 액면이 림에 닿나."""
    c, r = obs.container, obs.robot
    spill, headroom, rim = [], [], []
    for i, option in enumerate(options):
        a = max_accel(option, r) if accels is None else accels[i]
        rise = ph.liquid_rise(c.diameter, a, batch["k"], batch["tilt"])
        reached = np.asarray(batch["fill"]) * c.height + rise > c.height
        hit = impact[i] & (batch["lid"] < 0.5) if impact is not None else False
        spill.append((reached & (batch["lid"] < 0.5)) | hit)
        rim.append(reached)
        headroom.append(ph.headroom_used(batch["fill"], c.height, rise))
    return np.array(spill), np.array(headroom), np.array(rim)


def _fill_states(batch: Batch) -> dict[str, np.ndarray]:
    return {"fill_bin": fill_bins(batch["fill"]), "lid": (batch["lid"] > 0.5).astype(float)}


def _add_container(cfg: SimgenConfig, obs: Observation, batch: Batch, options, out: Outcomes, impact, accels=None) -> None:
    spill, headroom, rim = _container_outcomes(obs, batch, options, impact, accels)
    out.events["spill"] = spill
    out.events["rim_reached"] = rim
    out.quantities["headroom"] = headroom
    out.cost_items["spill"] = f"spill_{obs.container.content}"
    out.states.update(_fill_states(batch))


# ----------------------------------------------------------------------------------------------------------------


def _step_in(cfg: SimgenConfig, obs: Observation, batch: Batch, options) -> Outcomes:
    r, t = obs.robot, time_grid(cfg)
    dt = float(t[1] - t[0])
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
    x_a = col("gap") + 0.0 * tt
    r_sum = r.radius + b["ra"]
    contact, side_contact, delay, closeness, speed = [], [], [], [], []
    for option in options:
        x_r = _longitudinal(option, r, t)[None, :]
        y_r = np.zeros_like(x_r)
        side_hit = np.zeros(batch.size, dtype=bool)
        if option.startswith("sidestep"):
            sign = 1.0 if option.endswith("left") else -1.0
            y_r = ph.smoothstep(t, r.latency, r.sidestep_duration, sign * r.sidestep_offset)[None, :]
            side_hit = b["side_left" if sign > 0 else "side_right"] > 0.5
        hit, v_imp, d_min = ph.first_contact(x_r, y_r, x_a, y_a, r_sum, dt)
        contact.append(hit)
        speed.append(v_imp)
        side_contact.append(side_hit)
        delay.append(np.full(batch.size, _delay(float(x_r[0, -1]), r, horizon)))
        closeness.append(_closeness(d_min, r_sum, obs.body == "arm"))
    out = Outcomes(tuple(options), hard_brake=_hard_brake(options))
    out.events = {"contact": np.array(contact), "side_contact": np.array(side_contact)}
    out.impact = {"contact": np.array(speed)}
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
        _add_container(cfg, obs, batch, options, out, np.array(contact) | np.array(side_contact))  # 옆 장애물 닿음도 충격(m-b)
    return out


def _crossing(cfg: SimgenConfig, obs: Observation, batch: Batch, options) -> Outcomes:
    r, t = obs.robot, time_grid(cfg)
    dt = float(t[1] - t[0])
    horizon = float(t[-1])
    b = batch.values
    k = batch.size
    col = lambda name: b[name][:, None]  # noqa: E731
    tt = t[None, :]
    r_sum = r.radius + b["ra"]
    # 의도별 '자유' 진행(로봇이 길을 막지 않을 때): 진로 쪽으로 간 거리 s(t) (K, T)
    stop_at = np.maximum(col("dist") - r_sum[:, None] - col("yield_standoff"), 0.02)
    yield_decel = np.minimum(col("va") ** 2 / (2.0 * stop_at), col("agent_decel"))
    s_cross = col("va") * tt
    s_yield = ph.ramp_position(tt, col("va"), 0.0, yield_decel, 0.0)
    s_hurry = ph.ramp_position(tt, col("va"), 0.0, col("hurry_accel"), col("hurry_factor") * col("va"))
    roll = np.where(col("roll_decel") > 0, col("roll_decel"), 1.0)
    s_ball = ph.ramp_position(tt, col("va"), 0.0, roll, 0.0)
    intent = col("intent")
    s_free = np.where(intent == 0, s_cross, np.where(intent == 1, s_yield, np.where(intent == 2, s_hurry, s_ball)))
    v_free = np.diff(s_free, axis=1) / dt
    reacts = b["reacts"] > 0.5
    decel = b["agent_decel"]
    contact, delay, closeness, speed = [], [], [], []
    for option in options:
        x_r = _longitudinal(option, r, t)
        # 늦은 반응: 로봇 몸이 상대의 줄을 덮고 있고(앞뒤로 겹침) 상대가 멈춤 거리 + 여유 안에 들어오면 agent_decel로 선다.
        s = np.zeros((k, len(t)))
        v = v_free[:, 0].copy()
        braking = np.zeros(k, dtype=bool)
        for i in range(1, len(t)):
            on_line = np.abs(x_r[i - 1] - b["ahead"]) < r_sum + 0.1
            remaining = b["dist"] - s[:, i - 1] - r_sum
            trigger = reacts & on_line & (remaining > -b["ra"]) & (remaining <= b["react_standoff"] + v**2 / (2.0 * decel))
            braking = braking | trigger
            v_next = np.where(braking, np.minimum(np.maximum(v - decel * dt, 0.0), v_free[:, i - 1]), v_free[:, i - 1])
            s[:, i] = s[:, i - 1] + 0.5 * (v + v_next) * dt
            v = v_next
        y_a = col("from_sign") * (col("dist") - s)
        hit, v_imp, d_min = ph.first_contact(x_r[None, :], 0.0, col("ahead"), y_a, r_sum, dt)
        contact.append(hit)
        speed.append(v_imp)
        delay.append(np.full(k, _delay(float(x_r[-1]), r, horizon)))
        closeness.append(_closeness(d_min, r_sum, obs.body == "arm"))
    out = Outcomes(tuple(options), hard_brake=_hard_brake(options))
    out.events = {"contact": np.array(contact)}
    out.impact = {"contact": np.array(speed)}
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
    contact, delay, closeness, blocked, speed = [], [], [], [], []
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
        agent_speed = b["va"].copy()
        braking = (b["intent"] == 3) | ball
        for i in range(1, len(t)):
            overlap = np.abs(y_r[:, i - 1] - y_a[:, i - 1]) < r_sum + 0.1
            gap = x_a[:, i - 1] - x_r[i - 1] - r_sum
            trigger = (b["reactive"] > 0.5) & overlap & (gap <= b["keep_standoff"] + agent_speed**2 / (2.0 * decel))
            braking = braking | trigger
            new_speed = np.where(braking, np.maximum(agent_speed - decel * dt, 0.0), agent_speed)
            x_a[:, i] = x_a[:, i - 1] - 0.5 * (agent_speed + new_speed) * dt
            agent_speed = new_speed
        hit, v_imp, d_min = ph.first_contact(x_r[None, :], y_r, x_a, y_a, r_sum, dt)
        contact.append(hit)
        speed.append(v_imp)
        delay.append(np.full(k, _delay(float(x_r[-1]), r, horizon)))
        closeness.append(_closeness(d_min, r_sum, False))
        robot_still = abs(x_r[-1] - x_r[-2]) < 1e-9
        blocked.append(robot_still & (agent_speed <= 1e-9) & (np.abs(y_r[:, -1] - y_a[:, -1]) < r_sum))
    out = Outcomes(tuple(options), hard_brake=_hard_brake(options))
    out.events = {"contact": np.array(contact), "blocked": np.array(blocked)}
    out.impact = {"contact": np.array(speed)}
    out.quantities = {"delay": np.array(delay), "closeness": np.array(closeness)}
    out.cost_items = {"contact": cfg.agent(obs.agent.kind)["contact"]}
    out.world_events = {"passable": b["width"] - 2.0 * r_sum - r.passing_clearance >= 0.0}
    out.states = {"intent": b["intent"]}
    return out


def _braking(cfg: SimgenConfig, obs: Observation, batch: Batch, options) -> Outcomes:
    r = obs.robot
    b = batch.values
    k = batch.size
    tolerance = cfg.body(obs.body)["stop_tolerance"]
    surface = b["obstacle"] - r.radius
    overrun, delay, speed = [], [], []
    for option in options:
        if option == "continue":
            overrun.append(np.ones(k, dtype=bool))
            delay.append(np.zeros(k))
            speed.append(np.full(k, r.speed))
            continue
        a = r.decel(STOP_LEVELS[option])
        stop = ph.stopping_distance(r.speed, r.latency, a)
        # 장애물 면에 닿는 속도: 반응 거리 안이면 v0, 아니면 v0² − 2a(면 − 반응 거리)
        braking_room = np.maximum(surface - r.speed * r.latency, 0.0)
        v_imp = np.where(surface <= r.speed * r.latency, r.speed, np.sqrt(np.maximum(r.speed**2 - 2.0 * a * braking_room, 0.0)))
        hit = stop > surface
        overrun.append(hit)
        speed.append(np.where(hit, v_imp, 0.0))
        short = np.maximum(0.0, (surface - tolerance) - stop)
        delay.append(np.where(hit, 0.0, 2.0 * np.sqrt(short / r.decel_gentle)))
    out = Outcomes(tuple(options), hard_brake=_hard_brake(options))
    impact = np.array(overrun)
    out.events = {"overrun": impact}
    out.impact = {"overrun": np.array(speed)}
    out.quantities = {"delay": np.array(delay)}
    out.cost_items = {"overrun": "contact_object"}
    _add_container(cfg, obs, batch, options, out, impact)
    return out


def turn_profile(cfg: SimgenConfig, r: RobotSpec, option: str, radius: float, angle_deg: float, approach: float) -> tuple[float, float, bool]:
    """(지연 s, 쏟아짐 판정용 가장 큰 가속 m/s², 최대 감속을 썼나) — 회전 속도 선택 하나."""
    comfort = cfg.body(r.body)["comfort_decel"]
    v_entry, a_long = ph.turn_slowdown(r.speed, TURN_RATIOS[option] * r.speed, approach, comfort, r.decel_hard)
    if a_long <= 0.0:
        return 0.0, v_entry**2 / radius, False
    delay = ph.turn_delay(r.speed, v_entry, approach, radius, np.radians(angle_deg), a_long, r.accel)
    return delay, max(a_long, v_entry**2 / radius), a_long >= r.decel_hard


def _turning(cfg: SimgenConfig, obs: Observation, batch: Batch, options) -> Outcomes:
    r, env = obs.robot, obs.env
    k = batch.size
    delays, accels, hard = [], [], []
    for option in options:
        d, a, h = turn_profile(cfg, r, option, env.turn_radius, env.turn_angle_deg, env.turn_approach.value)
        delays.append(np.full(k, d))
        accels.append(a)
        hard.append(h)
    out = Outcomes(tuple(options), hard_brake=np.array(hard, dtype=bool))
    out.quantities = {"delay": np.array(delays)}
    _add_container(cfg, obs, batch, options, out, None, accels)
    return out


def rollout(cfg: SimgenConfig, obs: Observation, batch: Batch, options=None) -> Outcomes:
    """선택지 × 세계의 결과. options가 없으면 가능한 선택지 전부."""
    options = tuple(options or available_options(obs))
    builders = {
        "p1_step_in": _step_in, "p1p5_step_in_carrying": _step_in, "p1_crossing": _crossing, "p1_head_on": _head_on,
        "p5_braking": _braking, "p5_turning": _turning,
    }  # fmt: skip
    if obs.scene not in builders:
        raise ValueError(obs.scene)
    out = builders[obs.scene](cfg, obs, batch, options)
    out.body, out.loaded = obs.robot.body, obs.robot.loaded
    return out
