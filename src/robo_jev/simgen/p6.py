"""P6 오래된 관측·가림·센서 불일치 — 관측 쪽: 사후, 굴리기, 닫힌 꼴, 다시 보기 (Task v0b 지시서 §1 P6, P1+P6).

* ``p6_stale_track`` — 진로에 서 있던 것(사람·카트·로봇·지게차)을 a초 전에 봤고 지금은 가려졌다. 떠날 위험률 λ(숨은 값, 종류별 범위
  균등)로 P(아직 있음) = E_λ[e^{−λa}](닫힌 꼴). 있으면 지금부터 Exp(λ) 뒤에 떠난다. 로봇은 가림이 걷히는 거리 d_vis에 와서야 보고
  최대로 선다 — 반응 거리 + 최대 제동 거리 > d_vis면 닿는다(충격 속도도).
* ``p6_blind_corner`` — 모퉁이 뒤에서 누가 나올 위험: 출현 위험률 λ(분야 카드의 범위 균등), 3초 안 출현 확률 1 − E[e^{−3λ}](닫힌 꼴).
  나오면 그때 보이고 로봇은 반응 뒤 최대로 선다; 나온 상대는 진로 쪽으로 걷고(종류는 분야의 가로지름 상대 섞기), 로봇이 자기 줄을 막으면
  늦게라도 선다(가로지름과 같은 반응).
* ``p6_sensor_conflict`` — 라이다·카메라 중 하나만 '있다'. 단서(유리·먼지·햇빛·검은 표면·김)가 정한 신뢰도 표로 P(있음 | 두 보고) 닫힌 꼴.
  있으면 가까운 거리(d_late, 숨은 값)에서야 확실히 보고 최대로 선다.
* ``p1p6_emerging_step_in`` — 가림 뒤에서 막 나와 차선으로 들어오는 상대(P1 끼어듦) + 그 뒤를 따라 나올 수 있는 둘째 상대(있음 확률 —
  종류별, 나오는 시각·속도 사전); 닿음은 둘 중 누구와든.

있음/없음·출현/없음·따라옴/아님은 사후 세계를 둘로 나눠 **정확한 무게**로 센다(나머지 숨은 값은 표본). '다시 보기'는 이산 신호
(:class:`~robo_jev.simgen.voi.DiscreteSignal`): 잠깐 서서 다시 살핌(있으면 detect, 없으면 false_alarm), 모퉁이에서 내다봄(window_s 안에
나올 상대가 있으면 detect), 다른 각도로 다시 잼(두 센서의 새 보고, 같은 신뢰도 표). 보는 동안 상황은 그대로라고 둔다(멈춰서 본다).

세계 모듈을 import하지 않는다.
"""

from __future__ import annotations

import math

import numpy as np

from robo_jev.simgen import physics as ph
from robo_jev.simgen.observation import Observation
from robo_jev.simgen.posterior import _normal, _uniform
from robo_jev.simgen.rollout import Outcomes, _delay, _longitudinal, time_grid
from robo_jev.simgen.spec import Batch, RobotSpec
from robo_jev.simgen.voi import DiscreteSignal

__all__ = ["available", "conflict_posterior", "emerge_probability", "exact", "rollout", "sample_posterior", "stay_probability"]

OPTIONS = {
    "p6_stale_track": ("continue", "slow", "stop_moderate", "look_again"),
    "p6_blind_corner": ("continue", "slow", "creep", "stop_moderate", "look_again"),
    "p6_sensor_conflict": ("continue", "slow", "stop_moderate", "detour", "look_again"),
}


def available(obs: Observation) -> tuple[str, ...]:
    return OPTIONS[obs.scene]


def _uniform_exp_mean(lo: float, hi: float, t: float) -> float:
    """E_λ[e^{−λt}], λ ~ U(lo, hi)."""
    if t <= 0:
        return 1.0
    if hi - lo < 1e-12:
        return math.exp(-lo * t)
    return (math.exp(-lo * t) - math.exp(-hi * t)) / ((hi - lo) * t)


def stay_probability(cfg, kind: str, age: float) -> float:
    lo, hi = cfg.catalog["p6"]["stale"]["leave_rate"][kind]
    return _uniform_exp_mean(lo, hi, age)


def emerge_probability(cfg, domain: str, horizon: float = 3.0) -> float:
    lo, hi = cfg.domain(domain)["emerge_rate"]
    return 1.0 - _uniform_exp_mean(lo, hi, horizon)


def conflict_posterior(cfg, cue: str, lidar: bool, camera: bool) -> float:
    """P(있음 | 라이다·카메라 보고, 단서)."""
    table = cfg.catalog["p6"]["conflict"]["cues"][cue]
    prior = cfg.catalog["p6"]["conflict"]["prior_present"]

    def lik(sensor: str, says: bool, present: bool) -> float:
        p_yes = table[sensor][0 if present else 1]
        return p_yes if says else 1.0 - p_yes

    with_obj = prior * lik("lidar", lidar, True) * lik("camera", camera, True)
    without = (1.0 - prior) * lik("lidar", lidar, False) * lik("camera", camera, False)
    return with_obj / (with_obj + without)


# ----------------------------------------------------------------------------------------------------------------
# 사후
# ----------------------------------------------------------------------------------------------------------------


def _split(values: dict, p_yes: np.ndarray, flag: str) -> Batch:
    """세계 k를 둘로: flag = 1(무게 p_k/K)과 0(무게 (1 − p_k)/K)."""
    k = len(p_yes)
    doubled = {key: np.concatenate([np.asarray(v, dtype=float)] * 2) for key, v in values.items()}
    doubled[flag] = np.concatenate([np.ones(k), np.zeros(k)])
    weights = np.concatenate([p_yes, 1.0 - p_yes]) / k
    return Batch(doubled, weights=weights)


def _pick_world(batch: Batch, rng: np.random.Generator) -> Batch:
    """무게대로 세계 하나(참 세계)."""
    cdf = np.cumsum(batch.weights)
    i = min(int(np.searchsorted(cdf / cdf[-1], rng.random(), side="right")), batch.size - 1)
    return Batch({key: v[i : i + 1] for key, v in batch.values.items()})


def sample_posterior(cfg, obs: Observation, k: int, rng: np.random.Generator) -> Batch:
    if obs.scene == "p1p6_emerging_step_in":
        return _emerging(cfg, obs, k, rng)
    m = max(k, 64) if k == 1 else k
    if obs.scene == "p6_stale_track":
        s = obs.stale
        lo, hi = cfg.catalog["p6"]["stale"]["leave_rate"][s.kind]
        lam = _uniform(rng, (lo, hi), m)
        values = {"lam": lam, "t_leave": rng.exponential(1.0, m) / lam, "spot": np.maximum(_normal(rng, s.ahead, m, 0.0), 0.02)}
        batch = _split(values, np.exp(-lam * s.age), "present")
    elif obs.scene == "p6_blind_corner":
        c = obs.corner
        horizon = cfg.catalog["horizon_s"]
        lo, hi = cfg.domain(obs.domain)["emerge_rate"]
        lam = _uniform(rng, (lo, hi), m)
        p_emerge = 1.0 - np.exp(-lam * horizon)
        # 출현 시각: [0, H]에 잘린 Exp(λ)(역누적분포)
        t_e = -np.log(1.0 - rng.random(m) * p_emerge) / lam
        kinds = cfg.domain(obs.domain)["agents"]["crossing"]
        names = list(kinds)
        which = np.searchsorted(np.cumsum([kinds[n] for n in names]) / sum(kinds.values()), rng.random(m), side="right")
        which = np.minimum(which, len(names) - 1)
        spec = [cfg.agent(n) for n in names]
        lo_u = np.array([s["speed"][0] for s in spec])[which]
        hi_u = np.array([s["speed"][1] for s in spec])[which]
        lo_r = np.array([s["radius"][0] for s in spec])[which]
        hi_r = np.array([s["radius"][1] for s in spec])[which]
        lo_d = np.array([s.get("decel", [1.0, 2.0])[0] for s in spec])[which]
        hi_d = np.array([s.get("decel", [1.0, 2.0])[1] for s in spec])[which]
        react = np.array([cfg.catalog["crossing_reaction"]["base"][s["group"]] for s in spec])[which]
        u = lo_u + (hi_u - lo_u) * rng.random(m)
        values = {
            "lam": lam, "t_e": t_e, "u": u, "ra": lo_r + (hi_r - lo_r) * rng.random(m), "agent_decel": lo_d + (hi_d - lo_d) * rng.random(m),
            "reacts": (rng.random(m) < react).astype(float), "react_standoff": _uniform(rng, cfg.catalog["crossing_reaction"]["standoff"], m),
            "ahead": np.maximum(_normal(rng, c.ahead, m, 0.0), 0.0), "lateral": np.maximum(_normal(rng, c.lateral, m, 0.0), 0.0),
            "kind": which.astype(float),
        }  # fmt: skip
        batch = _split(values, p_emerge, "emerges")
    elif obs.scene == "p6_sensor_conflict":
        c = obs.conflict
        late = _uniform(rng, cfg.catalog["p6"]["conflict"]["cues"][c.cue]["late"], m)
        values = {"late": late, "spot": np.maximum(_normal(rng, c.distance, m, 0.0), 0.02)}
        p = conflict_posterior(cfg, c.cue, c.lidar, c.camera)
        batch = _split(values, np.full(m, p), "present")
    else:
        raise ValueError(obs.scene)
    return _pick_world(batch, rng) if k == 1 else batch


def _emerging(cfg, obs: Observation, k: int, rng: np.random.Generator) -> Batch:
    from robo_jev.simgen.posterior import _step_in

    m = max(k, 64) if k == 1 else k
    values = _step_in(cfg, obs, rng, m)
    spec = cfg.agent(obs.agent.kind)
    values.update({
        "t2": _uniform(rng, cfg.catalog["p6"]["follow_delay"], m), "u2": _uniform(rng, spec["speed"], m), "ra2": _uniform(rng, spec["radius"], m),
    })  # fmt: skip
    p_follow = cfg.catalog["p6"]["follow"].get(obs.agent.kind, 0.0)
    batch = _split(values, np.full(m, p_follow), "follow")
    return _pick_world(batch, rng) if k == 1 else batch


# ----------------------------------------------------------------------------------------------------------------
# 굴리기
# ----------------------------------------------------------------------------------------------------------------


def _profile(option: str, r: RobotSpec, t: np.ndarray) -> np.ndarray:
    if option == "creep":
        return ph.ramp_position(t, r.speed, r.latency, r.decel_moderate, r.creep_ratio * r.speed)
    return _longitudinal(option, r, t)


def _speed_at(option: str, r: RobotSpec, t) -> np.ndarray:
    if option == "continue":
        return np.full_like(np.asarray(t, dtype=float), r.speed)
    if option == "slow":
        return ph.ramp_velocity(t, r.speed, r.latency, r.decel_gentle, r.slow_ratio * r.speed)
    if option == "creep":
        return ph.ramp_velocity(t, r.speed, r.latency, r.decel_moderate, r.creep_ratio * r.speed)
    if option == "stop_moderate":
        return ph.ramp_velocity(t, r.speed, r.latency, r.decel_moderate, 0.0)
    raise ValueError(option)


def _detection_contact(r: RobotSpec, option: str, t: np.ndarray, reach: np.ndarray, present: np.ndarray, remaining: np.ndarray, gone_before=None):
    """로봇 앞면이 ``reach``(앞으로 간 거리)에 닿는 순간 보고, 반응 뒤 최대로 선다. 그때 남은 거리 ``remaining``보다 멈춤 거리가 길면 닿는다.
    돌려준다: (닿음, 충격 속도, 3초 끝 위치)."""
    x = _profile(option, r, t)
    horizon = float(t[-1])
    k = len(reach)
    hit = np.zeros(k, dtype=bool)
    speed = np.zeros(k)
    x_end = np.full(k, float(x[-1]))
    # 처음 닿는 시각(선형 보간)
    idx = np.searchsorted(x, reach, side="left")
    seen = (idx < len(t)) & present
    i = np.clip(idx, 1, len(t) - 1)
    frac = np.clip((reach - x[i - 1]) / np.maximum(x[i] - x[i - 1], 1e-12), 0.0, 1.0)
    t_seen = np.where(idx == 0, 0.0, t[i - 1] + frac * (t[i] - t[i - 1]))
    if gone_before is not None:
        seen &= ~(gone_before < t_seen)
    v_seen = _speed_at(option, r, t_seen)
    stop = v_seen * r.latency + v_seen**2 / (2.0 * r.decel_hard)
    hit = seen & (stop > remaining)
    room = np.maximum(remaining - v_seen * r.latency, 0.0)
    v_imp = np.where(remaining <= v_seen * r.latency, v_seen, np.sqrt(np.maximum(v_seen**2 - 2.0 * r.decel_hard * room, 0.0)))
    speed = np.where(hit, v_imp, 0.0)
    # 3초 끝 위치: 보았으면 그 자리에서 선 데까지(또는 닿은 데까지)
    braked_end = reach + np.minimum(stop, remaining)
    x_end = np.where(seen & (t_seen <= horizon), np.minimum(x_end, braked_end), x_end)
    return hit, speed, x_end


def _stale(cfg, obs: Observation, batch: Batch, options) -> Outcomes:
    r, t = obs.robot, time_grid(cfg)
    s = obs.stale
    b = batch.values
    k = batch.size
    present = b["present"] > 0.5
    reach = np.maximum(b["spot"] - s.sight, 0.0)           # 로봇 앞면이 이만큼 가면 그 자리가 보인다
    out = Outcomes(tuple(options))
    contact, impact, delay = [], [], []
    for option in options:
        if option == "look_again":
            contact.append(np.zeros(k, dtype=bool)), impact.append(np.zeros(k)), delay.append(np.zeros(k))
            continue
        remaining = b["spot"] - reach
        hit, speed, x_end = _detection_contact(r, option, t, reach, present, remaining, gone_before=b["t_leave"])
        contact.append(hit), impact.append(speed), delay.append(_delay(x_end, r, float(t[-1])))
    out.events = {"contact": np.array(contact)}
    out.impact = {"contact": np.array(impact)}
    out.quantities = {"delay": np.array(delay)}
    out.cost_items = {"contact": cfg.agent(s.kind)["contact"]}
    out.world_events = {"present": present.astype(float)}
    out.hard_brake = np.zeros(len(options), dtype=bool)
    look = cfg.catalog["p6"]["stale"]["look"]
    if "look_again" in options:
        still = present & (b["t_leave"] > obs.env.look_s)
        p_seen = np.where(still, look["detect"], look["false_alarm"])
        out.voi = DiscreteSignal(tuple(o for o in options if o != "look_again"), float(obs.env.look_s), np.vstack([p_seen, 1.0 - p_seen]))
    return out


def _corner(cfg, obs: Observation, batch: Batch, options) -> Outcomes:
    """모퉁이: 출현한 세계에서 로봇은 출현 시각 + 반응 뒤 최대로 서고, 상대는 진로 쪽으로 걷다가(반응하면) 로봇 앞에서 선다."""
    r, t = obs.robot, time_grid(cfg)
    dt = float(t[1] - t[0])
    b = batch.values
    k = batch.size
    c = obs.corner
    emerges = b["emerges"] > 0.5
    line = b["ahead"] + r.radius                              # 로봇 중심 → 상대가 지나는 줄
    r_sum = r.radius + b["ra"]
    start = r.radius + b["lateral"] + b["ra"]                # 출현 때 상대 중심의 진로 중심선에서의 거리
    sign = 1.0 if c.side == "left" else -1.0
    out = Outcomes(tuple(options))
    contact, impact, delay = [], [], []
    for option in options:
        if option == "look_again":
            contact.append(np.zeros(k, dtype=bool)), impact.append(np.zeros(k)), delay.append(np.zeros(k))
            continue
        x_opt = _profile(option, r, t)
        v_opt = np.concatenate([[r.speed], np.diff(x_opt) / dt])
        # 로봇: 출현 + 반응 뒤 최대 제동 — 세계마다 (K, T)
        t_b = np.where(emerges, b["t_e"] + r.latency, np.inf)
        x_r = np.empty((k, len(t)))
        for j, tj in enumerate(t):
            before = tj <= t_b
            tb = np.minimum(t_b, t[-1])
            jb = np.minimum((tb / dt).astype(int), len(t) - 1)
            xb, vb = x_opt[jb], v_opt[jb]
            tau = np.clip(tj - tb, 0.0, None)
            stop_t = vb / r.decel_hard
            tau1 = np.minimum(tau, stop_t)
            braked = xb + vb * tau1 - 0.5 * r.decel_hard * tau1 * tau1
            x_r[:, j] = np.where(before, x_opt[j], braked)
        # 상대: 출현 전에도 걷고 있다(가려졌을 뿐) — 진로 쪽 거리 d(t) = start + u·(t_e − t); 반응하면 로봇이 줄을 막을 때 선다
        dist = np.empty((k, len(t)))
        v = b["u"].copy()
        s_pos = start + b["u"] * b["t_e"]                   # t = 0의 진로 쪽 거리
        braking = np.zeros(k, dtype=bool)
        dist[:, 0] = s_pos
        for j in range(1, len(t)):
            visible = t[j - 1] >= b["t_e"]
            on_line = np.abs(x_r[:, j - 1] - line) < r_sum + 0.1
            remaining_d = dist[:, j - 1] - r_sum
            trigger = visible & (b["reacts"] > 0.5) & on_line & (remaining_d <= b["react_standoff"] + v**2 / (2.0 * b["agent_decel"]))
            braking = braking | trigger
            v_next = np.where(braking, np.maximum(v - b["agent_decel"] * dt, 0.0), b["u"])
            dist[:, j] = dist[:, j - 1] - 0.5 * (v + v_next) * dt
            v = v_next
        y_a = sign * dist
        hit, v_imp, _ = ph.first_contact(x_r, 0.0, line[:, None], y_a, r_sum, dt)
        hit &= emerges
        contact.append(hit), impact.append(np.where(hit, v_imp, 0.0))
        delay.append(_delay(x_r[:, -1], r, float(t[-1])))
    out.events = {"contact": np.array(contact)}
    out.impact = {"contact": np.array(impact)}
    out.quantities = {"delay": np.array(delay)}
    kinds = list(cfg.domain(obs.domain)["agents"]["crossing"])
    items = [cfg.agent(n)["contact"] for n in kinds]
    # 닿음 비용은 나올 수 있는 상대 종류마다 다르다 — 세계마다의 종류로 센다(가장 흔한 비용 항목으로 묶지 않는다)
    out.cost_items = {}
    for item in sorted(set(items)):
        mask = np.isin(b["kind"].astype(int), [i for i, it in enumerate(items) if it == item])
        out.events[f"contact__{item}"] = out.events["contact"] & mask[None, :]
        out.impact[f"contact__{item}"] = out.impact["contact"]
        out.cost_items[f"contact__{item}"] = item
    horizon = float(t[-1])
    # 결정 여유의 사건: 지금 누가 나오면 모퉁이에 닿기 전에 설 수 있나(반응 + 최대 제동 < 모퉁이까지 거리 — 나옴과 무관한 세계 사건)
    d_hard = ph.stopping_distance(r.speed, r.latency, r.decel_hard)
    out.world_events = {"emerge_3s": (emerges & (b["t_e"] <= horizon)).astype(float), "stops_before_corner": (b["ahead"] > d_hard).astype(float)}
    out.hard_brake = np.zeros(len(options), dtype=bool)
    if "look_again" in options:
        look = cfg.catalog["p6"]["corner"]["look"]
        coming = emerges & (b["t_e"] <= look["window_s"])
        p_seen = np.where(coming, look["detect"], look["false_alarm"])
        out.voi = DiscreteSignal(tuple(o for o in options if o != "look_again"), float(obs.env.look_s), np.vstack([p_seen, 1.0 - p_seen]))
    return out


def _conflict(cfg, obs: Observation, batch: Batch, options) -> Outcomes:
    r, t = obs.robot, time_grid(cfg)
    b = batch.values
    k = batch.size
    c = obs.conflict
    present = b["present"] > 0.5
    reach = np.maximum(b["spot"] - b["late"], 0.0)
    out = Outcomes(tuple(options))
    contact, impact, delay = [], [], []
    for option in options:
        if option == "look_again":
            contact.append(np.zeros(k, dtype=bool)), impact.append(np.zeros(k)), delay.append(np.zeros(k))
            continue
        if option == "detour":
            contact.append(np.zeros(k, dtype=bool)), impact.append(np.zeros(k)), delay.append(np.full(k, float(obs.env.detour_s)))
            continue
        hit, speed, x_end = _detection_contact(r, option, t, reach, present, b["spot"] - reach)
        contact.append(hit), impact.append(speed), delay.append(_delay(x_end, r, float(t[-1])))
    out.events = {"contact": np.array(contact)}
    out.impact = {"contact": np.array(impact)}
    out.quantities = {"delay": np.array(delay)}
    out.cost_items = {"contact": "contact_object"}
    out.world_events = {"present": present.astype(float)}
    out.states = {"sensor_present": present.astype(float)}
    out.hard_brake = np.zeros(len(options), dtype=bool)
    if "look_again" in options:
        table = cfg.catalog["p6"]["conflict"]["cues"][c.cue]
        rows = []
        for lidar in (True, False):
            for camera in (True, False):
                pl = np.where(present, table["lidar"][0], table["lidar"][1])
                pc = np.where(present, table["camera"][0], table["camera"][1])
                rows.append((pl if lidar else 1.0 - pl) * (pc if camera else 1.0 - pc))
        out.voi = DiscreteSignal(tuple(o for o in options if o != "look_again"), float(obs.env.look_s), np.vstack(rows))
    return out


def _emerging_rollout(cfg, obs: Observation, batch: Batch, options) -> Outcomes:
    """P1+P6: 첫 상대는 P1 끼어듦 그대로, 둘째(따라옴)는 첫 상대가 나온 자리(첫 보고 위치)에서 t2에 나와 같은 쪽으로 u2로 가로지른다."""
    from robo_jev.simgen.rollout import _step_in

    out = _step_in(cfg, obs, batch, options)
    r, t = obs.robot, time_grid(cfg)
    dt = float(t[1] - t[0])
    b = batch.values
    a = obs.agent
    follow = b["follow"] > 0.5
    # 첫 상대가 나온 자리: 가장 오래된 위치 보고(나이 동안 옆으로 움직였다)
    y_occ = b["lat0"] - b["dir"] * b["u0"] * a.track_ages[0]
    tt = t[None, :]
    y2 = y_occ[:, None] + b["dir"][:, None] * b["u2"][:, None] * np.clip(tt - b["t2"][:, None], 0.0, None)
    active = tt >= b["t2"][:, None]
    y2 = np.where(active, y2, y_occ[:, None] - b["dir"][:, None] * 5.0)   # 나오기 전에는 가림 뒤(멀리)
    x2 = b["gap"][:, None] + 0.0 * tt
    r_sum2 = r.radius + b["ra2"]
    contact2, speed2 = [], []
    for i, option in enumerate(options):
        from robo_jev.simgen.rollout import _longitudinal as longitudinal

        x_r = longitudinal(option, r, t)[None, :]
        y_r = np.zeros_like(x_r)
        if option.startswith("sidestep"):
            sgn = 1.0 if option.endswith("left") else -1.0
            y_r = ph.smoothstep(t, r.latency, r.sidestep_duration, sgn * r.sidestep_offset)[None, :]
        hit, v_imp, _ = ph.first_contact(x_r, y_r, x2, y2, r_sum2, dt)
        contact2.append(hit & follow), speed2.append(np.where(hit & follow, v_imp, 0.0))
    first = out.events["contact"]
    second = np.array(contact2)
    out.events["contact"] = first | second
    out.impact["contact"] = np.where(first, out.impact["contact"], np.array(speed2))
    out.world_events["follower"] = follow.astype(float)
    return out


def rollout(cfg, obs: Observation, batch: Batch, options) -> Outcomes:
    if obs.scene == "p6_stale_track":
        return _stale(cfg, obs, batch, options)
    if obs.scene == "p6_blind_corner":
        return _corner(cfg, obs, batch, options)
    if obs.scene == "p6_sensor_conflict":
        return _conflict(cfg, obs, batch, options)
    if obs.scene == "p1p6_emerging_step_in":
        return _emerging_rollout(cfg, obs, batch, options)
    raise ValueError(obs.scene)


def exact(cfg, obs: Observation) -> dict:
    out: dict = {}
    if obs.scene == "p6_stale_track":
        out["present"] = stay_probability(cfg, obs.stale.kind, obs.stale.age)
    elif obs.scene == "p6_blind_corner":
        out["emerge_3s"] = emerge_probability(cfg, obs.domain, cfg.catalog["horizon_s"])
        a = obs.corner.ahead
        if not a.censored:
            # 모퉁이까지 참 거리의 사후 = N(보고, σ_eff)(0에서 자름 — d > 0이라 영향 없음) → P(참 거리 > 멈춤 거리) = Φ((보고 − d)/σ)
            d_hard = ph.stopping_distance(obs.robot.speed, obs.robot.latency, obs.robot.decel_hard)
            out["stops_before_corner"] = 0.5 * (1.0 + math.erf((a.value - d_hard) / (a.sigma_eff * math.sqrt(2.0))))
    elif obs.scene == "p6_sensor_conflict":
        p = conflict_posterior(cfg, obs.conflict.cue, obs.conflict.lidar, obs.conflict.camera)
        out["present"] = p
        out["sensor_present"] = np.array([1.0 - p, p])
    elif obs.scene == "p1p6_emerging_step_in":
        out["follower"] = cfg.catalog["p6"]["follow"].get(obs.agent.kind, 0.0)
    return out
