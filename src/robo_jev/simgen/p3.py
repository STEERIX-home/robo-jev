"""P3 받침 위 물건 — 관측 쪽: 사후, 굴리기, 닫힌 꼴, 무게 확인(다시 보기) (Task v0b 지시서 §1 P3, P3+P5, P1+P3).

**사후.** 잰 높이·바닥 폭·가장자리 거리는 보고 주변 정규(국소적으로 평평한 사전), 무게중심 높이 비율·앞뒤 치우침은 물건 종류의 속성 목록 범위
균등, 마찰 μ는 받침 표면(글이 말하는 종류)의 범위 균등 — 설계와 무관한 사전. 쌓기의 무게 치우침 c는 ±c_max 균등에 들어 올릴 때의 쏠림 단서
(맞을 확률 0.8)를 곱한 조각 균등, 놓기 오차 ε ~ N(0, σ_place)(글에 적힌 정확도).

**굴리기(준정적, physics의 식).** 선택지마다 받침의 앞뒤 가속(감속·출발, 경사로면 기울기를 더한 쏠림 비)과 회전의 옆 가속에서
넘어짐(tip)·미끄러짐(slide)·떨어짐(fall = 넘어짐 또는 가장자리를 넘는 미끄러짐 또는 충돌)과 미끄러진 거리의 비율(slide_frac)을 센다.
감속 갈래는 p5_braking처럼 장애물 넘어섬(충돌 → 떨어짐)·지연을, 회전은 p5_turning처럼 감속·지연을, 끼어듦(P1+P3)은 P1 굴리기의 닿음(충돌
→ 떨어짐)을 그대로 쓴다. 실은 용기(P3+P5)는 같은 가속으로 쏟아짐을 센다.

세계 모듈을 import하지 않는다.
"""

from __future__ import annotations

import math

import numpy as np

from robo_jev.simgen import physics as ph
from robo_jev.simgen.observation import Observation
from robo_jev.simgen.posterior import _container, _normal, _std_cdf, _uniform
from robo_jev.simgen.rollout import STOP_LEVELS, TURN_RATIOS, Outcomes, _add_container, _braking, _step_in, time_grid
from robo_jev.simgen.spec import Batch
from robo_jev.simgen.voi import DiscreteSignal

__all__ = ["available", "exact", "load_events", "rollout", "sample_posterior"]

G = ph.G


def available(obs: Observation) -> tuple[str, ...]:
    if obs.scene == "p3_ramp":
        return ("ramp_normal", "ramp_slow", "secure_first", "detour")
    if obs.scene == "p3_place_stack":
        return ("place_now", "recentre", "place_elsewhere", "check_weight")
    maneuver = obs.env.maneuver
    if maneuver == "brake":
        return ("continue", "stop_gentle", "stop_moderate", "stop_hard")
    if maneuver == "turn":
        return tuple(TURN_RATIOS)
    return ("start_gentle", "start_normal", "start_quick", "secure_first")


# ----------------------------------------------------------------------------------------------------------------
# 사후
# ----------------------------------------------------------------------------------------------------------------


def _load_posterior(cfg, obs: Observation, rng, k: int) -> dict:
    load = obs.load
    spec = cfg.catalog["loads"][load.kind]
    return {
        "height": np.maximum(_normal(rng, load.height, k, 0.01), 0.01), "base": np.maximum(_normal(rng, load.base, k, 0.005), 0.005),
        "edge_f": np.maximum(_normal(rng, load.edge_front, k, 0.0), 0.0), "edge_b": np.maximum(_normal(rng, load.edge_back, k, 0.0), 0.0),
        "edge_s": np.maximum(_normal(rng, load.edge_side, k, 0.0), 0.0),
        "com": _uniform(rng, spec["com_ratio"], k), "off": _uniform(rng, (-spec["offset"], spec["offset"]), k),
        "mu": _uniform(rng, cfg.catalog["decks"][load.deck]["mu"], k),
    }  # fmt: skip


def _stack_posterior(cfg, obs: Observation, rng, k: int) -> dict:
    st = obs.stack
    c_max = cfg.catalog["stack_items"][st.kind]["c_max"] * st.length_top / 2.0
    rel = cfg.catalog["stack"]["cue_reliability"]
    if st.cue is None:
        c = rng.uniform(-c_max, c_max, k)
    else:
        toward = 1.0 if st.cue == "left" else -1.0
        side = np.where(rng.random(k) < rel, toward, -toward)
        c = side * rng.uniform(0.0, c_max, k)
    return {"c": c, "eps": st.sigma_place * rng.standard_normal(k)}


def sample_posterior(cfg, obs: Observation, k: int, rng: np.random.Generator) -> Batch:
    if obs.scene == "p3_place_stack":
        return Batch(_stack_posterior(cfg, obs, rng, k))
    if obs.scene == "p1p3_step_in_carrying_tall":
        from robo_jev.simgen.posterior import _step_in

        values = _step_in(cfg, obs, rng, k)
    else:
        values = {}
        if obs.env.maneuver == "brake":
            surface = np.maximum(_normal(rng, obs.env.obstacle_distance, k, 0.0), 0.005)
            values["obstacle"] = surface + obs.robot.radius
        if obs.container is not None:
            values.update(_container(cfg, obs, rng, k))
    values.update(_load_posterior(cfg, obs, rng, k))
    return Batch({key: np.asarray(v, dtype=float) for key, v in values.items()})


# ----------------------------------------------------------------------------------------------------------------
# 물건의 사건
# ----------------------------------------------------------------------------------------------------------------


def load_events(batch: Batch, long_acc, dv, direction: str, lat_acc=None, lat_time=None, impact=None, incline: float = 0.0, slow_static=None):
    """선택지마다 (넘어짐, 미끄러짐, 떨어짐, 미끄러진 거리 비율) 각각 (A, K).

    long_acc (A,): 앞뒤 가속의 크기(감속은 앞으로, 출발·오르막은 뒤로 쏠림 — ``direction``), dv (A,): 그 가속으로 바뀌는 속도,
    lat_acc·lat_time (A,): 회전의 옆 가속과 그 지속 시간, impact (A, K): 충돌(그러면 떨어진다), incline: 경사로 기울기(라디안)."""
    b = batch.values
    h = b["com"] * b["height"]
    b_long = b["base"] / 2.0 * (1.0 - b["off"]) if direction == "front" else b["base"] / 2.0 * (1.0 + b["off"])
    edge_long = b["edge_f"] if direction == "front" else b["edge_b"]
    tip, slide, fall, frac = [], [], [], []
    for i in range(len(long_acc)):
        ratio = float(ph.incline_ratio(long_acc[i], incline)) if (long_acc[i] > 0 or incline > 0) else 0.0
        t_l, s_l = ph.tip_slide(ratio, b["mu"], b_long, h)
        d_l = np.where(s_l, ph.slide_distance(dv[i], long_acc[i], b["mu"], incline), 0.0) if (long_acc[i] > 0 or incline > 0) else np.zeros(batch.size)
        off_l = s_l & (d_l > edge_long)
        t_s = s_s = off_s = np.zeros(batch.size, dtype=bool)
        d_s = np.zeros(batch.size)
        if lat_acc is not None and lat_acc[i] > 0:
            t_s, s_s = ph.tip_slide(lat_acc[i] / G, b["mu"], b["base"] / 2.0, h)
            d_s = np.where(s_s, ph.turn_slide_distance(lat_acc[i], b["mu"], lat_time[i]), 0.0)
            off_s = s_s & (d_s > b["edge_s"])
        hit = np.asarray(impact[i], dtype=bool) if impact is not None else np.zeros(batch.size, dtype=bool)
        tip.append(t_l | t_s)
        slide.append(s_l | s_s)
        fall.append(t_l | t_s | off_l | off_s | hit)
        with np.errstate(divide="ignore", invalid="ignore"):
            f_l = np.where(s_l, np.minimum(1.0, d_l / np.maximum(edge_long, 1e-6)), 0.0)
            f_s = np.where(s_s, np.minimum(1.0, d_s / np.maximum(b["edge_s"], 1e-6)), 0.0)
        frac.append(np.maximum(f_l, f_s))
    return np.array(tip), np.array(slide), np.array(fall), np.array(frac)


def _add_load(cfg, obs: Observation, out: Outcomes, events) -> None:
    tip, slide, fall, frac = events
    out.events.update(tip=tip, slide=slide, fall=fall)
    out.quantities["slide_frac"] = frac
    out.cost_items["fall"] = cfg.catalog["loads"][obs.load.kind]["fall"]


# ----------------------------------------------------------------------------------------------------------------
# 굴리기
# ----------------------------------------------------------------------------------------------------------------


def _brake(cfg, obs: Observation, batch: Batch, options) -> Outcomes:
    """감속: p5_braking의 넘어섬·지연(용기가 있으면 쏟아짐까지) 위에 물건의 사건."""
    out = _braking(cfg, obs, batch, options) if obs.container is not None else _braking_without_container(cfg, obs, batch, options)
    r = obs.robot
    acc = np.array([0.0 if o == "continue" else r.decel(STOP_LEVELS[o]) for o in options])
    dv = np.full(len(options), r.speed)
    _add_load(cfg, obs, out, load_events(batch, acc, dv, "front", impact=out.events["overrun"]))
    return out


def _braking_without_container(cfg, obs: Observation, batch: Batch, options) -> Outcomes:
    """p5_braking의 넘어섬·지연만(용기 없이)."""
    r = obs.robot
    b = batch.values
    k = batch.size
    tolerance = cfg.body(obs.body)["stop_tolerance"]
    surface = b["obstacle"] - r.radius
    overrun, delay, speed = [], [], []
    for option in options:
        if option == "continue":
            overrun.append(np.ones(k, dtype=bool)), delay.append(np.zeros(k)), speed.append(np.full(k, r.speed))
            continue
        a = r.decel(STOP_LEVELS[option])
        stop = ph.stopping_distance(r.speed, r.latency, a)
        braking_room = np.maximum(surface - r.speed * r.latency, 0.0)
        v_imp = np.where(surface <= r.speed * r.latency, r.speed, np.sqrt(np.maximum(r.speed**2 - 2.0 * a * braking_room, 0.0)))
        hit = stop > surface
        overrun.append(hit), speed.append(np.where(hit, v_imp, 0.0))
        short = np.maximum(0.0, (surface - tolerance) - stop)
        delay.append(np.where(hit, 0.0, 2.0 * np.sqrt(short / r.decel_gentle)))
    out = Outcomes(tuple(options), hard_brake=np.array([o == "stop_hard" for o in options]))
    out.events = {"overrun": np.array(overrun)}
    out.impact = {"overrun": np.array(speed)}
    out.quantities = {"delay": np.array(delay)}
    out.cost_items = {"overrun": "contact_object"}
    return out


def _turn(cfg, obs: Observation, batch: Batch, options) -> Outcomes:
    """회전: p5_turning의 감속(접근 구간의 필요한 감속)·지연 + 물건의 앞뒤(감속)·옆(회전) 사건; 용기가 있으면 쏟아짐."""
    from robo_jev.simgen.rollout import turn_profile

    r, env = obs.robot, obs.env
    k = batch.size
    comfort = cfg.body(r.body)["comfort_decel"]
    delays, accels, hard, long_acc, dv, lat_acc, lat_time = [], [], [], [], [], [], []
    for option in options:
        d, a, h = turn_profile(cfg, r, option, env.turn_radius, env.turn_angle_deg, env.turn_approach.value)
        v_entry, a_long = ph.turn_slowdown(r.speed, TURN_RATIOS[option] * r.speed, env.turn_approach.value, comfort, r.decel_hard)
        delays.append(np.full(k, d)), accels.append(a), hard.append(h)
        long_acc.append(a_long), dv.append(r.speed - v_entry)
        lat_acc.append(v_entry**2 / env.turn_radius), lat_time.append(env.turn_radius * math.radians(env.turn_angle_deg) / max(v_entry, 1e-6))
    out = Outcomes(tuple(options), hard_brake=np.array(hard, dtype=bool))
    out.quantities = {"delay": np.array(delays)}
    if obs.container is not None:
        _add_container(cfg, obs, batch, options, out, None, accels)
    _add_load(cfg, obs, out, load_events(batch, np.array(long_acc), np.array(dv), "front", np.array(lat_acc), np.array(lat_time)))
    return out


def _start(cfg, obs: Observation, batch: Batch, options) -> Outcomes:
    """출발(멈춘 로봇): 출발 가속(부드럽게·보통·빠르게)으로 순항 속도까지 — 물건은 뒤로 쏠린다. 고정 먼저: 고정 시간 뒤 빠르게(위험 없음).
    지연 = 앞으로 3초 동안 '빠르게 출발'보다 덜 간 진행(초); 고정 먼저는 고정 시간."""
    r, t = obs.robot, time_grid(cfg)
    k = batch.size
    horizon = float(t[-1])
    accels = obs.env.start_accels
    ref = ph.ramp_position(np.array([horizon]), 0.0, r.latency, accels["start_quick"], r.cruise)[0]
    long_acc, dv, delay = [], [], []
    for option in options:
        if option == "secure_first":
            long_acc.append(0.0), dv.append(0.0), delay.append(np.full(k, float(obs.env.secure_s)))
            continue
        a = accels[option]
        x = ph.ramp_position(np.array([horizon]), 0.0, r.latency, a, r.cruise)[0]
        long_acc.append(a), dv.append(r.cruise), delay.append(np.full(k, (ref - x) / r.cruise))
    out = Outcomes(tuple(options), hard_brake=np.zeros(len(options), dtype=bool))
    out.quantities = {"delay": np.array(delay)}
    _add_load(cfg, obs, out, load_events(batch, np.array(long_acc), np.array(dv), "back"))
    return out


def _ramp(cfg, obs: Observation, batch: Batch, options) -> Outcomes:
    """경사로: 보통(지금 속도, 단단한 멈춤 / 보통 출발)·천천히(절반 속도, 부드러운 멈춤 / 절반 출발)·고정 먼저·돌아가기.
    내려가며 멈추면 앞으로, 올라가며 출발하면 뒤로 쏠린다(쏠림 비 = (sin φ + a/g)/cos φ)."""
    r, env = obs.robot, obs.env
    k = batch.size
    phi = math.radians(env.ramp_angle_deg)
    down = env.ramp_direction == "down"
    v = r.speed if down else r.cruise
    slow = cfg.catalog["p3"]["ramp_slow_ratio"]
    long_acc, dv, delay = [], [], []
    for option in options:
        if option in ("secure_first", "detour"):
            long_acc.append(0.0), dv.append(0.0)
            delay.append(np.full(k, float(env.secure_s if option == "secure_first" else env.detour_s)))
            continue
        if option == "ramp_normal":
            a = r.decel_moderate if down else r.accel
            long_acc.append(a), dv.append(v), delay.append(np.zeros(k))
        else:
            a = r.decel_gentle if down else 0.5 * r.accel
            long_acc.append(a), dv.append(slow * v), delay.append(np.full(k, env.ramp_length * (1.0 / (slow * v) - 1.0 / v)))
    out = Outcomes(tuple(options), hard_brake=np.zeros(len(options), dtype=bool))
    out.quantities = {"delay": np.array(delay)}
    events = load_events(batch, np.array(long_acc), np.array(dv), "front" if down else "back", incline=phi)
    # 고정 먼저·돌아가기는 물건이 넘어지지 않는다
    tip, slide, fall, frac = events
    for i, option in enumerate(options):
        if option in ("secure_first", "detour"):
            tip[i] = slide[i] = fall[i] = False
            frac[i] = 0.0
    _add_load(cfg, obs, out, (tip, slide, fall, frac))
    return out


def _step_in_tall(cfg, obs: Observation, batch: Batch, options) -> Outcomes:
    """P1+P3: P1 끼어듦의 굴리기(닿음·지연·가까움) + 물건 — 멈춤은 그 감속, 느려지기는 gentle, 비켜서기는 gentle + 옆 가속, 닿으면 떨어진다."""
    out = _step_in(cfg, obs, batch, options)
    r = obs.robot
    long_acc, dv, lat_acc, lat_time = [], [], [], []
    for option in options:
        if option in STOP_LEVELS:
            long_acc.append(r.decel(STOP_LEVELS[option])), dv.append(r.speed), lat_acc.append(0.0), lat_time.append(0.0)
        elif option == "slow":
            long_acc.append(r.decel_gentle), dv.append(r.speed * (1.0 - r.slow_ratio)), lat_acc.append(0.0), lat_time.append(0.0)
        elif option.startswith("sidestep"):
            long_acc.append(r.decel_gentle), dv.append(r.speed * (1.0 - r.sidestep_speed_ratio))
            lat_acc.append(ph.smoothstep_peak_accel(r.sidestep_offset, r.sidestep_duration)), lat_time.append(r.sidestep_duration / 2.0)
        else:  # continue
            long_acc.append(0.0), dv.append(0.0), lat_acc.append(0.0), lat_time.append(0.0)
    impact = out.events["contact"] | out.events["side_contact"]
    _add_load(cfg, obs, out, load_events(batch, np.array(long_acc), np.array(dv), "front", np.array(lat_acc), np.array(lat_time), impact=impact))
    return out


def _stack_topple(batch: Batch, obs: Observation, offset: float) -> np.ndarray:
    st = obs.stack
    return np.abs(offset + batch["eps"] + batch["c"]) >= st.length_low / 2.0


def _stack(cfg, obs: Observation, batch: Batch, options) -> Outcomes:
    st = obs.stack
    k = batch.size
    topple, delay = [], []
    for option in options:
        if option == "place_now":
            topple.append(_stack_topple(batch, obs, st.offset)), delay.append(np.zeros(k))
        elif option == "recentre":
            topple.append(_stack_topple(batch, obs, 0.0)), delay.append(np.full(k, st.recentre_s))
        elif option == "place_elsewhere":
            topple.append(np.zeros(k, dtype=bool)), delay.append(np.full(k, st.elsewhere_s))
        else:  # check_weight — 다시 보기 줄(voi)
            topple.append(np.zeros(k, dtype=bool)), delay.append(np.zeros(k))
    out = Outcomes(tuple(options), hard_brake=np.zeros(len(options), dtype=bool))
    out.events = {"topple": np.array(topple)}
    out.quantities = {"delay": np.array(delay)}
    out.cost_items = {"topple": cfg.catalog["stack_items"][st.kind]["fall"]}
    if st.offset != 0.0:
        out.states = {"weight_toward": (np.sign(batch["c"]) == np.sign(st.offset)).astype(float)}
    if "check_weight" in options:
        out.voi = _weight_signal(cfg, obs, batch, tuple(o for o in options if o != "check_weight"))
    return out


def _weight_signal(cfg, obs: Observation, batch: Batch, base: tuple[str, ...]) -> DiscreteSignal:
    """무게 확인: c를 σ_check로 잰 값을 구간(24칸 + 양 끝)으로 — 신호 s의 세계별 확률은 정규 누적분포의 차(닫힌 꼴)."""
    st = obs.stack
    c_max = cfg.catalog["stack_items"][st.kind]["c_max"] * st.length_top / 2.0
    sigma = cfg.catalog["stack"]["sigma_check"] * c_max
    edges = np.linspace(-c_max - 2 * sigma, c_max + 2 * sigma, 25)
    cdf = _std_cdf((edges[:, None] - batch["c"][None, :]) / sigma)               # (25, K)
    probs = np.vstack([cdf[:1], np.diff(cdf, axis=0), 1.0 - cdf[-1:]])         # (26, K)
    return DiscreteSignal(base, float(st.check_s), probs, option="check_weight")


def rollout(cfg, obs: Observation, batch: Batch, options) -> Outcomes:
    if obs.scene == "p3_place_stack":
        return _stack(cfg, obs, batch, options)
    if obs.scene == "p3_ramp":
        return _ramp(cfg, obs, batch, options)
    if obs.scene == "p1p3_step_in_carrying_tall":
        return _step_in_tall(cfg, obs, batch, options)
    maneuver = obs.env.maneuver
    if maneuver == "brake":
        return _brake(cfg, obs, batch, options)
    if maneuver == "turn":
        return _turn(cfg, obs, batch, options)
    return _start(cfg, obs, batch, options)


# ----------------------------------------------------------------------------------------------------------------
# 닫힌 꼴 — 쌓기의 넘어짐
# ----------------------------------------------------------------------------------------------------------------


def _int_phi(a: float, b: float) -> float:
    """∫_a^b Φ(x) dx = [xΦ(x) + φ(x)]_a^b."""

    def F(x: float) -> float:
        return x * float(_std_cdf(np.array([x]))[0]) + math.exp(-0.5 * x * x) / math.sqrt(2 * math.pi)

    return F(b) - F(a)


def _p_stable(offset: float, half: float, sigma: float, pieces: list[tuple[float, float, float]]) -> float:
    """P(|offset + ε + c| < half), ε ~ N(0, σ), c ~ 조각 균등(pieces = (하한, 상한, 무게))."""
    total = 0.0
    for lo, hi, w in pieces:
        if hi <= lo or w <= 0:
            continue
        # E_c[Φ((half − offset − c)/σ) − Φ((−half − offset − c)/σ)] = σ/(hi − lo)·[∫Φ(u)du]
        upper = sigma / (hi - lo) * _int_phi((half - offset - hi) / sigma, (half - offset - lo) / sigma)
        lower = sigma / (hi - lo) * _int_phi((-half - offset - hi) / sigma, (-half - offset - lo) / sigma)
        total += w * (upper - lower)
    return float(min(max(total, 0.0), 1.0))


def exact(cfg, obs: Observation) -> dict:
    if obs.scene != "p3_place_stack":
        return {}
    st = obs.stack
    c_max = cfg.catalog["stack_items"][st.kind]["c_max"] * st.length_top / 2.0
    rel = cfg.catalog["stack"]["cue_reliability"]
    if st.cue is None:
        pieces = [(-c_max, c_max, 1.0)]
    else:
        pos = rel if st.cue == "left" else 1.0 - rel
        pieces = [(0.0, c_max, pos), (-c_max, 0.0, 1.0 - pos)]
    half = st.length_low / 2.0
    out = {
        "topple|place_now": 1.0 - _p_stable(st.offset, half, st.sigma_place, pieces),
        "topple|recentre": 1.0 - _p_stable(0.0, half, st.sigma_place, pieces),
    }
    if st.offset != 0.0:
        # 무게가 튀어나올 쪽(계획 어긋남과 같은 부호)에 있을 확률 = 조각마다 그 쪽에 놓인 비율 × 무게(단서가 없으면 반반)
        def share(lo: float, hi: float) -> float:
            part = (max(hi, 0.0) - max(lo, 0.0)) if st.offset > 0 else (min(hi, 0.0) - min(lo, 0.0))
            return part / (hi - lo)

        toward = sum(w * share(lo, hi) for lo, hi, w in pieces)
        out["weight_toward"] = np.array([1.0 - toward, toward])
    return out
