"""사후 표본: 관측이 주어졌을 때 가능한 세계 K개 (docs/12 §5 '사후 표본 뽑기', Task v0a 지시서 §3.5, 리뷰 1 반영).

**관측만 받는다** — 세계 모듈을 import하지 않는다(누출 금지; 시험이 이 모듈의 import를 검사한다). 사전은 설계(여유 표본기)와 무관한
속성 목록의 사전이다 — 그래서 정답이 표본 설계를 배우게 하지 않는다(리뷰 1 질문 3).

* 관측된 수치는 관측값 주변 정규분포(감각 잡음 σ와 보고 반올림을 합친 ``Reading.sigma_eff``; 국소적으로 평평한 사전).
  바닥에 붙은 보고(``Reading.censored`` — 날 값이 바닥보다 작았다)는 검열된 관측이다: 사후 ∝ Φ((바닥 + 해상도/2 − x)/σ)(물리적 하한 위).
* 속도(≥ 0)는 잘린 정규분포(역누적분포 — 같은 난수면 보고가 클수록 표본이 크다).
* 채움은 보고가 거친 구간(말 구간 또는 5 % 반올림)이고 날 추정값이 [0, 1]로 잘려 보고되므로, **속성 목록의 채움 범위 위 평평한 사전** ×
  정확한 구간 가능도로 뽑는다(:func:`_fill_posterior`).
* 관측 안 된 값(상대의 반지름, 의도별 움직임 매개변수, 출렁임 계수, 바닥 기울기, 공의 구름 저항, 가려진 쪽의 장애물, 보이지 않는 뚜껑,
  가로지르는 상대가 반응하나)은 속성 목록·분야 카드의 사전 분포에서 뽑는다.
* 이산 의도는 단서 가능도로 베이즈 갱신한다: P(지금 의도 | 단서, 나이) = (1 − q)·P(의도 | 단서) + q·사전, q = 1 − e^{−λ·나이}.
* 위치를 몇 초 전에 봤으면 그동안 관측 속도로 움직였다고 지금 위치를 민다. 끼어듦의 지금 옆 위치는 글의 전제("당신의 차선으로 들어오고
  있다" — 몸 중심이 들어오는 쪽에서 중심선까지 (로봇 + 상대 반지름) × step_in_band 안)로 자른다(Task v0b — 생성기와 같은 띠). 마주 옴(head-on)의 위치 보고는 **그 시각의 둘 사이 간격**이라
  지금 간격 = 보고 − (상대 속도 + 로봇 속도)·나이이고, 위치 두 개의 차이 / 간격 시간은 다가오는 속도(상대 + 로봇)다(리뷰 1 C1).

정답에는 사후 표본의 빈도 대신 정확한 값을 쓸 수 있는 곳(의도·뚜껑·채움 구간의 상태 정답, Φ로 닫히는 세계 사건과 감속의 넘어섬)이 있다 —
:func:`exact_labels` (리뷰 1 m2).
"""

from __future__ import annotations

import math
from statistics import NormalDist

import numpy as np

from robo_jev.simgen.catalog import INTENTS, SimgenConfig, reaction_probability
from robo_jev.simgen.observation import Observation
from robo_jev.simgen.spec import FILL_BINS, Batch, Reading, fill_bin_index

__all__ = ["exact_labels", "intent_posterior", "lid_probability", "sample_posterior"]

G = 9.81
_STD = NormalDist()


def _std_cdf(x: np.ndarray) -> np.ndarray:
    """표준정규 누적분포(벡터) — erf의 Abramowitz–Stegun 7.1.26 근사(오차 < 1.5e-7)."""
    x = np.asarray(x, dtype=float)
    z = np.abs(x) / math.sqrt(2.0)
    t = 1.0 / (1.0 + 0.3275911 * z)
    poly = t * (0.254829592 + t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429))))
    erf = 1.0 - poly * np.exp(-z * z)
    return 0.5 * (1.0 + np.sign(x) * erf)


def _grid_sample(rng: np.random.Generator, grid: np.ndarray, weight: np.ndarray, k: int) -> np.ndarray:
    """격자 위 가중치로 역누적분포 표본(같은 난수면 단조)."""
    cdf = np.cumsum(np.clip(weight, 0.0, None))
    if cdf[-1] <= 0:
        return np.full(k, float(grid[0]))
    return np.interp(rng.random(k), cdf / cdf[-1], grid)


def _censored(rng: np.random.Generator, reading: Reading, k: int, lo: float) -> np.ndarray:
    """바닥에 붙은 보고의 사후: 평평한 사전(x ≥ lo) × P(날 값 < 바닥 + 해상도/2 | x)."""
    cut = reading.floor + reading.resolution / 2.0
    grid = np.linspace(lo, cut + 6.0 * reading.sigma_eff, 801)
    return _grid_sample(rng, grid, _std_cdf((cut - grid) / max(reading.sigma, 1e-6)), k)


def _normal(rng: np.random.Generator, reading: Reading, k: int, lo: float = -math.inf) -> np.ndarray:
    if reading.censored:
        return _censored(rng, reading, k, 0.0 if not math.isfinite(lo) else lo)
    return reading.value + reading.sigma_eff * rng.standard_normal(k)


def _truncated_normal(rng: np.random.Generator, mean: float, sd: float, k: int, lo: float = -math.inf, hi: float = math.inf) -> np.ndarray:
    """잘린 정규분포 — 역누적분포로 뽑는다(균등 난수 k개 → 분위수). 같은 난수면 평균이 클수록 모든 표본이 커진다(단조 결합)."""
    u = rng.random(k)
    a = _STD.cdf((lo - mean) / sd) if math.isfinite(lo) else 0.0
    b = _STD.cdf((hi - mean) / sd) if math.isfinite(hi) else 1.0
    if b - a < 1e-12:
        return np.full(k, min(max(mean, lo), hi))
    p = np.clip(a + u * (b - a), 1e-12, 1.0 - 1e-12)
    out = mean + sd * np.array([_STD.inv_cdf(float(x)) for x in p])
    return np.clip(out, lo, hi)


def _speed(rng: np.random.Generator, reading: Reading, k: int, lo: float) -> np.ndarray:
    if reading.censored:
        return _censored(rng, reading, k, lo)
    return _truncated_normal(rng, reading.value, reading.sigma_eff, k, lo=lo)


def _uniform(rng: np.random.Generator, lo_hi, k: int) -> np.ndarray:
    return rng.uniform(lo_hi[0], lo_hi[1], k)


def intent_posterior(cfg: SimgenConfig, kind: str, family: str, cue: str, age: float) -> np.ndarray:
    """지금 의도의 사후 확률(INTENTS[family] 순서)."""
    agent = cfg.agent(kind)
    intents = INTENTS[family]
    prior = np.array([agent["intent_prior"][family][i] for i in intents], dtype=float)
    prior = prior / prior.sum()
    row = cfg.catalog["cues"][agent["group"]][family][cue]
    lik = np.array([row[i] for i in intents], dtype=float)
    given_cue = prior * lik / float(np.sum(prior * lik))
    q = 1.0 - math.exp(-agent["hazard_per_s"] * age)
    return (1.0 - q) * given_cue + q * prior


def lid_probability(cfg: SimgenConfig, obs: Observation) -> float:
    """뚜껑이 있을 사후 확률(보임 상태에 따라 0 / 1 / 베이즈)."""
    c = obs.container
    if c.lid == "open":
        return 0.0
    if c.lid == "lid":
        return 1.0
    prior = cfg.container(c.kind)["lid_prior"]
    views = cfg.catalog["lid_view"]
    with_lid = prior * views["given_lid"]["unknown"]
    return float(with_lid / (with_lid + (1.0 - prior) * views["given_open"]["unknown"]))


def _categorical(rng: np.random.Generator, probs: np.ndarray, k: int) -> np.ndarray:
    cdf = np.cumsum(probs / probs.sum())
    return np.minimum(np.searchsorted(cdf, rng.random(k), side="right"), len(probs) - 1).astype(float)


def _roll_decel(cfg: SimgenConfig, obs: Observation, rng: np.random.Generator, k: int) -> np.ndarray:
    floors = cfg.domain(obs.domain)["floor"]
    names = list(floors)
    which = _categorical(rng, np.array([floors[n] for n in names], dtype=float), k).astype(int)
    lo = np.array([cfg.catalog["floors"][n][0] for n in names])[which]
    hi = np.array([cfg.catalog["floors"][n][1] for n in names])[which]
    return (lo + (hi - lo) * rng.random(k)) * G


def _track(rng, obs: Observation, k: int, direction: float, speed_lo: float, offset: float = 0.0, lo: float = -math.inf):
    """(관측 축의 '지금' 값, 상대 자신의 속도). 관측 축 값은 시간이 흐를 때 direction·(상대 속도 + offset)으로 변한다 —
    offset은 로봇 쪽 기여(마주 옴에서 로봇 속도: 둘 사이 간격은 둘이 함께 줄인다). 위치 두 개면 차이가 (상대 속도 + offset)이다."""
    agent = obs.agent
    if agent.speed is not None:
        speed = _speed(rng, agent.speed, k, speed_lo)
        last = _normal(rng, agent.track[-1], k, lo)
    else:
        first = _normal(rng, agent.track[0], k, lo)
        last = _normal(rng, agent.track[1], k, lo)
        dt = agent.track_ages[0] - agent.track_ages[1]
        speed = np.maximum(direction * (last - first) / dt - offset, speed_lo)
    now = last + direction * (speed + offset) * agent.age
    return now, speed


def _intents(cfg, obs: Observation, family: str, rng, k: int) -> np.ndarray:
    agent = obs.agent
    if agent.cue is None:  # 공
        return np.full(k, -1.0)
    return _categorical(rng, intent_posterior(cfg, agent.kind, family, agent.cue, agent.age), k)


_FILL_POINTS = 981


def _fill_interval(c) -> tuple[float, float]:
    """보고된 채움이 뜻하는 날 추정값의 구간. 날 추정값은 [0, 1]로 잘려 보고되므로 맨 위·맨 아래 구간은 끝이 열려 있다(검열)."""
    if c.fill_mode == "phrase":
        lo, hi, _ = FILL_BINS[fill_bin_index(c.fill.value)]
    else:
        lo, hi = c.fill.value - c.fill.resolution / 2.0, c.fill.value + c.fill.resolution / 2.0
    return (-math.inf if lo <= 0.0 else lo), (math.inf if hi >= 1.0 else hi)


def _fill_weights(cfg: SimgenConfig, c) -> tuple[np.ndarray, np.ndarray]:
    """(채움 격자, 사후 가중치): 속성 목록의 채움 범위 위 평평한 사전 × 정확한 구간 가능도 P(f + σz ∈ [lo, hi))."""
    f_lo, f_hi = cfg.container(c.kind)["fill"]
    grid = np.linspace(f_lo, f_hi, _FILL_POINTS)
    lo, hi = _fill_interval(c)
    sigma = c.fill.sigma
    upper = np.ones_like(grid) if math.isinf(hi) else _std_cdf((hi - grid) / sigma)
    lower = np.zeros_like(grid) if math.isinf(lo) else _std_cdf((lo - grid) / sigma)
    return grid, np.clip(upper - lower, 0.0, None)


def _fill_posterior(rng: np.random.Generator, cfg: SimgenConfig, c, k: int) -> np.ndarray:
    """채움의 사후 표본(격자 역누적분포 — 같은 난수면 보고가 클수록 표본이 크다)."""
    grid, weight = _fill_weights(cfg, c)
    return _grid_sample(rng, grid, weight, k)


def _container(cfg: SimgenConfig, obs: Observation, rng, k: int) -> dict[str, np.ndarray]:
    c = obs.container
    fill = _fill_posterior(rng, cfg, c, k)
    slosh = _uniform(rng, cfg.catalog["slosh_k"], k)
    tilt = np.radians(rng.normal(0.0, cfg.domain(obs.domain)["tilt_sd_deg"], k))
    lid = (rng.random(k) < lid_probability(cfg, obs)).astype(float)
    return {"fill": fill, "k": slosh, "tilt": tilt, "lid": lid}


_PREMISE_OVERSAMPLE = 8


def _step_in_track(cfg, obs: Observation, rng, k: int, direction: float, spec: dict):
    """(상대 반지름, 지금 옆 위치, 옆 속도) — 글의 전제 "당신의 차선으로 들어오고 있다"로 자른 사후(Task v0b; 리뷰 1 재검증의 끼어듦
    잔여 — 자르지 않으면 오래된 추적에서 사후가 상대를 차선 밖에 두어 닿음을 약 0.05 낮게 잡았다).

    전제: 몸 중심이 들어오는 쪽에서 진로 중심선까지 (로봇 반지름 + 상대 반지름) × ``intent_motion.step_in_band`` 안에 있고 아직 중심선을
    넘지 않았다 — 생성기가 상대를 두는 띠와 같다. 후보 8K개(반지름·추적 보고의 사후)를 뽑아 전제를 지키는 것만 받는다: 앞에서부터 K개,
    모자라면 받은 것에서 복원 추출, 하나도 없으면(관측이 전제와 크게 어긋날 때) 자르지 않은 앞 K개."""
    band = cfg.catalog["intent_motion"]["step_in_band"]
    m = _PREMISE_OVERSAMPLE * k
    ra = _uniform(rng, spec["radius"], m)
    lat0, u0 = _track(rng, obs, m, direction, 0.0)
    entering = -direction * lat0  # 들어오는 쪽에서 중심선까지(+ = 아직 넘지 않음)
    ok = np.flatnonzero((entering >= 0.0) & (entering <= band * (obs.robot.radius + ra)))
    if len(ok) >= k:
        idx = ok[:k]
    elif len(ok) > 0:
        idx = np.concatenate([ok, rng.choice(ok, k - len(ok))])
    else:
        idx = np.arange(k)
    return ra[idx], lat0[idx], u0[idx]


def _step_in(cfg, obs: Observation, rng, k: int) -> dict[str, np.ndarray]:
    agent, robot = obs.agent, obs.robot
    spec = cfg.agent(agent.kind)
    motion = cfg.catalog["intent_motion"]
    direction = 1.0 if agent.heading == "left" else -1.0
    # 옆 위치 축: 시간이 흐르면 direction 쪽으로 움직인다. 상대 반지름(숨은 값)과 함께 글의 전제 띠로 자른다.
    ra, lat0, u0 = _step_in_track(cfg, obs, rng, k, direction, spec)
    # 보고는 면 사이 간격 — 중심 거리 = 간격 + 로봇 반지름 + 상대 반지름.
    gap = np.maximum(_normal(rng, agent.ahead, k, 0.0), 0.005) + robot.radius + ra
    decel = _uniform(rng, spec.get("decel", [1.0, 2.0]), k)
    cross_speed = np.maximum(_uniform(rng, spec["speed"], k), u0)
    values = {
        "gap": gap, "lat0": lat0, "dir": np.full(k, direction), "u0": u0, "ra": ra,
        "intent": _intents(cfg, obs, "step_in", rng, k),
        "stop_decel": decel, "stay_s": _uniform(rng, motion["stay"]["stay_s"], k), "cross_speed": cross_speed,
        "cross_accel": _uniform(rng, motion["cross"]["accel"], k), "back_decel": decel,
        "back_speed": _uniform(rng, motion["back"]["speed_factor"], k) * cross_speed,
        "roll_decel": _roll_decel(cfg, obs, rng, k) if spec["group"] == "ball" else np.zeros(k),
    }  # fmt: skip
    sides = obs.env.sides
    prior = cfg.domain(obs.domain)["side_obstacle_prior"]
    for name, i in (("side_left", 0), ("side_right", 1)):
        draw = rng.random(k)
        status = sides[i].status if sides is not None else "none"
        values[name] = {"clear": np.zeros(k), "blocked": np.ones(k), "none": np.zeros(k)}.get(status, (draw < prior).astype(float))
    if obs.container is not None:
        values.update(_container(cfg, obs, rng, k))
    return values


def _crossing(cfg, obs: Observation, rng, k: int) -> dict[str, np.ndarray]:
    agent = obs.agent
    spec = cfg.agent(agent.kind)
    motion = cfg.catalog["intent_motion"]
    scale = motion["arm_scale"] if obs.body == "arm" else 1.0
    # 교차선 보고는 로봇 앞면에서 상대가 지나는 줄(상대 중심)까지 — 중심 기준으로 되돌린다.
    ahead = np.maximum(_normal(rng, agent.ahead, k, 0.0), 0.0) + obs.robot.radius
    # 진로까지 거리 축(상대의 가까운 면 → 로봇 진로 가장자리): 시간이 흐르면 줄어든다(로봇의 전진은 이 옆 거리를 바꾸지 않는다).
    surface, va = _track(rng, obs, k, -1.0, 0.05, lo=0.0)
    ra = _uniform(rng, spec["radius"], k)
    dist = surface + obs.robot.radius + ra
    react = reaction_probability(cfg, agent.kind, agent.cue)
    return {
        "ahead": ahead, "dist": dist, "from_sign": np.full(k, 1.0 if agent.heading == "from_left" else -1.0), "va": va,
        "ra": ra, "intent": _intents(cfg, obs, "crossing", rng, k),
        "yield_standoff": _uniform(rng, motion["yield"]["standoff"], k) * scale,
        "agent_decel": _uniform(rng, spec.get("decel", [1.0, 2.0]), k),
        "hurry_factor": _uniform(rng, motion["hurry"]["factor"], k), "hurry_accel": _uniform(rng, motion["hurry"]["accel"], k),
        "roll_decel": _roll_decel(cfg, obs, rng, k) if spec["group"] == "ball" else np.zeros(k),
        "reacts": (rng.random(k) < react).astype(float),
        "react_standoff": _uniform(rng, cfg.catalog["crossing_reaction"]["standoff"], k) * scale,
    }  # fmt: skip


def _head_on(cfg, obs: Observation, rng, k: int) -> dict[str, np.ndarray]:
    agent = obs.agent
    spec = cfg.agent(agent.kind)
    motion = cfg.catalog["intent_motion"]
    # 위치 보고는 그 시각의 간격 — 지금 간격 = 보고 − (상대 속도 + 로봇 속도)·나이(리뷰 1 C1).
    surface, va = _track(rng, obs, k, -1.0, 0.05, offset=obs.robot.speed, lo=0.0)
    ra = np.maximum(_normal(rng, agent.size, k, 0.05) / 2.0, 0.03)
    intent = _intents(cfg, obs, "head_on", rng, k)
    return {
        "dist": surface + obs.robot.radius + ra, "lat_a": _normal(rng, agent.lateral, k),
        "width": np.maximum(_normal(rng, obs.env.corridor_width, k, 0.1), 0.1),
        "va": va, "ra": ra, "intent": np.maximum(intent, 0.0),
        "aside_speed": _uniform(rng, motion["aside"]["lateral_speed"], k), "keep_standoff": _uniform(rng, motion["keep"]["standoff"], k),
        "agent_decel": _uniform(rng, spec.get("decel", [1.0, 2.0]), k), "reactive": np.full(k, float(spec["reactive"])),
        "roll_decel": _roll_decel(cfg, obs, rng, k) if spec["group"] == "ball" else np.zeros(k),
    }  # fmt: skip


def sample_posterior(cfg: SimgenConfig, obs: Observation, k: int, rng: np.random.Generator) -> Batch:
    """관측에서 사후 세계 K개(P4·P6의 이산 부분은 정확한 무게로 나열 — 그 Batch는 weights를 갖는다)."""
    from robo_jev.simgen.catalog import scene_kind

    kind = scene_kind(obs.scene)
    if kind is not None:
        from robo_jev.simgen import p3, p4, p6

        return {"p3": p3, "p4": p4, "p6": p6}[kind].sample_posterior(cfg, obs, k, rng)
    if obs.scene in ("p1_step_in", "p1p5_step_in_carrying"):
        values = _step_in(cfg, obs, rng, k)
    elif obs.scene == "p1_crossing":
        values = _crossing(cfg, obs, rng, k)
    elif obs.scene == "p1_head_on":
        values = _head_on(cfg, obs, rng, k)
    elif obs.scene == "p5_braking":
        surface = np.maximum(_normal(rng, obs.env.obstacle_distance, k, 0.0), 0.005)
        values = {"obstacle": surface + obs.robot.radius, **_container(cfg, obs, rng, k)}
    elif obs.scene == "p5_turning":
        values = _container(cfg, obs, rng, k)
    else:
        raise ValueError(obs.scene)
    return Batch({name: np.asarray(value, dtype=float) for name, value in values.items()})


# ----------------------------------------------------------------------------------------------------------------
# 정확한 정답(표본 빈도 대신) — 리뷰 1 m2
# ----------------------------------------------------------------------------------------------------------------


def _upper_tail(reading: Reading, threshold: float, lo: float) -> float | None:
    """P(사후의 그 값 > threshold) — 검열이 아니면 1 − Φ((threshold − 보고)/σ_eff), 검열이면 None(표본으로)."""
    if reading.censored:
        return None
    if threshold <= lo:
        return 1.0
    return float(1.0 - _STD.cdf((threshold - reading.value) / reading.sigma_eff))


def exact_labels(cfg: SimgenConfig, obs: Observation) -> dict:
    """닫힌 꼴로 계산되는 정답: 상태(의도·뚜껑·채움 구간), 세계 사건(stops_short, passable), 선택지 사건(감속의 넘어섬 —
    열쇠 'overrun|stop_moderate'). 계산할 수 없는 것(검열된 보고 등)은 넣지 않는다."""
    out: dict = {}
    a, r = obs.agent, obs.robot
    from robo_jev.simgen.catalog import scene_kind

    kind = scene_kind(obs.scene)
    if kind is not None:
        from robo_jev.simgen import p3, p4, p6

        out.update({"p3": p3, "p4": p4, "p6": p6}[kind].exact(cfg, obs))
        if obs.scene not in ("p1p6_emerging_step_in", "p1p3_step_in_carrying_tall"):
            return out
    family = {"p1_step_in": "step_in", "p1p5_step_in_carrying": "step_in", "p1_crossing": "crossing", "p1_head_on": "head_on",
              "p1p6_emerging_step_in": "step_in", "p1p3_step_in_carrying_tall": "step_in"}.get(obs.scene)  # fmt: skip
    if a is not None and a.cue is not None and family is not None:
        out["intent"] = intent_posterior(cfg, a.kind, family, a.cue, a.age)
    if obs.container is not None:
        out["lid"] = lid_probability(cfg, obs)
        grid, weight = _fill_weights(cfg, obs.container)
        bins = np.searchsorted(np.array([hi for _, hi, _ in FILL_BINS[:-1]]), grid, side="right")
        total = float(weight.sum())
        out["fill_bin"] = np.array([float(weight[bins == b].sum()) / total for b in range(len(FILL_BINS))])
    if obs.scene in ("p1_step_in", "p1p5_step_in_carrying", "p1p6_emerging_step_in", "p1p3_step_in_carrying_tall"):
        stop = r.speed * r.latency + r.speed**2 / (2.0 * r.decel_moderate)
        p = _upper_tail(a.ahead, stop, 0.005)
        if p is not None:
            out["stops_short"] = p
    if obs.scene == "p5_braking":
        # 넘어섬: 멈춤 거리 > 장애물 면 간격 — 선택지마다 P(면 < 멈춤) (사건 열쇠 '사건|선택지')
        for option, level in (("stop_gentle", "gentle"), ("stop_moderate", "moderate"), ("stop_hard", "hard")):
            stop = r.speed * r.latency + r.speed**2 / (2.0 * r.decel(level))
            p = _upper_tail(obs.env.obstacle_distance, stop, 0.005)
            if p is not None:
                out[f"overrun|{option}"] = 1.0 - p
    if obs.scene == "p1_head_on" and not obs.env.corridor_width.censored and not a.size.censored:
        spare = obs.env.corridor_width.value - a.size.value - 2.0 * r.radius - r.passing_clearance
        sd = math.hypot(obs.env.corridor_width.sigma_eff, a.size.sigma_eff)
        out["passable"] = float(_STD.cdf(spare / sd))
    return out
