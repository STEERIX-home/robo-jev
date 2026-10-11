"""사후 표본: 관측이 주어졌을 때 가능한 세계 K개 (docs/12 §5 '사후 표본 뽑기', Task v0a 지시서 §3.5).

**관측만 받는다** — 세계 모듈을 import하지 않는다(누출 금지; 시험이 이 모듈의 import를 검사한다).

* 관측된 수치는 관측값 주변 정규분포(감각 잡음 σ와 보고 반올림을 합친 ``Reading.sigma_eff``; 국소적으로 평평한 사전).
  값의 범위가 물리적으로 막힌 속도(≥ 0)는 잘린 정규분포(역누적분포)로 뽑는다. 채움은 보고가 거친 구간(말 구간 또는 5 % 반올림)이고
  날 추정값이 [0, 1]로 잘려 보고되므로, 평평한 사전 × 정확한 구간 가능도로 뽑는다(:func:`_fill_posterior`).
* 관측 안 된 값(상대의 반지름, 의도별 움직임 매개변수, 출렁임 계수, 바닥 기울기, 공의 구름 저항, 가려진 쪽의 장애물, 보이지 않는 뚜껑)은
  속성 목록·분야 카드의 사전 분포에서 뽑는다.
* 이산 의도는 단서 가능도로 베이즈 갱신한다: P(지금 의도 | 단서, 나이) = (1 − q)·P(의도 | 단서) + q·사전, q = 1 − e^{−λ·나이}
  (관측 뒤 의도가 바뀌었을 확률 q — 바뀌면 사전에서 다시 뽑힌다; scenes._intent 의 생성 모형과 같다).
* 위치를 몇 초 전에 봤으면 그동안 관측 속도로 움직였다고 지금 위치를 민다(장면 모듈의 v0 근사와 같다). 위치 두 개만 주어지면
  속도는 두 위치의 차이로 정해진다(두 위치를 각각 뽑으므로 속도 불확실성이 저절로 따라온다).
"""

from __future__ import annotations

import math
from statistics import NormalDist

import numpy as np

from robo_jev.simgen.catalog import INTENTS, SimgenConfig
from robo_jev.simgen.observation import Observation
from robo_jev.simgen.spec import FILL_BINS, Batch, Reading, fill_bin_index

__all__ = ["intent_posterior", "sample_posterior"]

G = 9.81
_STD = NormalDist()


def _normal(rng: np.random.Generator, reading: Reading, k: int) -> np.ndarray:
    return reading.value + reading.sigma_eff * rng.standard_normal(k)


def _truncated_normal(rng: np.random.Generator, mean: float, sd: float, k: int, lo: float = -math.inf, hi: float = math.inf) -> np.ndarray:
    """잘린 정규분포 — 역누적분포로 뽑는다(균등 난수 k개 → 분위수). 같은 난수면 평균이 클수록 모든 표본이 커진다(단조 결합):
    사실 하나만 바꾼 쌍과 단조성 시험이 같은 난수 위에서 비교된다. 거절 표본은 받아들인 수가 평균에 따라 달라져 이 결합을 깬다."""
    u = rng.random(k)
    a = _STD.cdf((lo - mean) / sd) if math.isfinite(lo) else 0.0
    b = _STD.cdf((hi - mean) / sd) if math.isfinite(hi) else 1.0
    if b - a < 1e-12:
        return np.full(k, min(max(mean, lo), hi))
    p = np.clip(a + u * (b - a), 1e-12, 1.0 - 1e-12)
    out = mean + sd * np.array([_STD.inv_cdf(float(x)) for x in p])
    return np.clip(out, lo, hi)


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


def _track(rng, obs: Observation, k: int, direction: float, speed_lo: float) -> tuple[np.ndarray, np.ndarray]:
    """(관측 축의 '지금' 위치, 그 축을 따른 속도). direction = 시간이 흐를 때 그 축 값이 변하는 부호."""
    agent = obs.agent
    if agent.speed is not None:
        speed = _truncated_normal(rng, agent.speed.value, agent.speed.sigma_eff, k, lo=speed_lo)
        last = _normal(rng, agent.track[-1], k)
    else:
        first = _normal(rng, agent.track[0], k)
        last = _normal(rng, agent.track[1], k)
        dt = agent.track_ages[0] - agent.track_ages[1]
        speed = np.maximum(direction * (last - first) / dt, speed_lo)
    now = last + direction * speed * agent.age
    return now, speed


def _intents(cfg, obs: Observation, family: str, rng, k: int) -> np.ndarray:
    agent = obs.agent
    if agent.cue is None:  # 공
        return np.full(k, -1.0)
    return _categorical(rng, intent_posterior(cfg, agent.kind, family, agent.cue, agent.age), k)


_FILL_GRID = np.linspace(0.02, 1.0, 981)


def _std_cdf(x: np.ndarray) -> np.ndarray:
    """표준정규 누적분포(벡터) — erf의 Abramowitz–Stegun 7.1.26 근사(오차 < 1.5e-7)."""
    z = np.abs(x) / math.sqrt(2.0)
    t = 1.0 / (1.0 + 0.3275911 * z)
    poly = t * (0.254829592 + t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429))))
    erf = 1.0 - poly * np.exp(-z * z)
    return 0.5 * (1.0 + np.sign(x) * erf)


def _fill_interval(c) -> tuple[float, float]:
    """보고된 채움이 뜻하는 날 추정값의 구간. 날 추정값은 [0, 1]로 잘려 보고되므로 맨 위·맨 아래 구간은 끝이 열려 있다(검열)."""
    if c.fill_mode == "phrase":
        lo, hi, _ = FILL_BINS[fill_bin_index(c.fill.value)]
    else:
        lo, hi = c.fill.value - c.fill.resolution / 2.0, c.fill.value + c.fill.resolution / 2.0
    return (-math.inf if lo <= 0.0 else lo), (math.inf if hi >= 1.0 else hi)


def _fill_posterior(rng: np.random.Generator, c, k: int) -> np.ndarray:
    """채움의 사후 표본: 평평한 사전 × 정확한 구간 가능도 P(f + σz ∈ [lo, hi)) — 격자 위 역누적분포.

    보고를 구간 가운데의 정규 관측으로 보면(σ_eff) 잘린 맨 위('full to the brim', 100 %)에서 사후가 아래로 끌린다 — 가득 찬 용기의 쏟아짐
    확률이 체계적으로 낮게 나온다(스모크 실측: 열린 용기 정답 평균 0.39 대 실제 0.52). 같은 난수 u면 보고가 클수록 표본이 크다(단조 결합)."""
    lo, hi = _fill_interval(c)
    sigma = c.fill.sigma
    upper = np.ones_like(_FILL_GRID) if math.isinf(hi) else _std_cdf((hi - _FILL_GRID) / sigma)
    lower = np.zeros_like(_FILL_GRID) if math.isinf(lo) else _std_cdf((lo - _FILL_GRID) / sigma)
    weight = np.clip(upper - lower, 0.0, None)
    cdf = np.cumsum(weight)
    cdf = cdf / cdf[-1]
    return np.interp(rng.random(k), cdf, _FILL_GRID)


def _container(cfg: SimgenConfig, obs: Observation, rng, k: int) -> dict[str, np.ndarray]:
    c = obs.container
    fill = _fill_posterior(rng, c, k)
    slosh = _uniform(rng, cfg.catalog["slosh_k"], k)
    tilt = np.radians(rng.normal(0.0, cfg.domain(obs.domain)["tilt_sd_deg"], k))
    lid_draw = rng.random(k)
    if c.lid == "open":
        lid = np.zeros(k)
    elif c.lid == "lid":
        lid = np.ones(k)
    else:
        prior = cfg.container(c.kind)["lid_prior"]
        views = cfg.catalog["lid_view"]
        with_lid = prior * views["given_lid"]["unknown"]
        lid = (lid_draw < with_lid / (with_lid + (1.0 - prior) * views["given_open"]["unknown"])).astype(float)
    return {"fill": fill, "k": slosh, "tilt": tilt, "lid": lid}


def _step_in(cfg, obs: Observation, rng, k: int) -> dict[str, np.ndarray]:
    agent, robot = obs.agent, obs.robot
    spec = cfg.agent(agent.kind)
    motion = cfg.catalog["intent_motion"]
    direction = 1.0 if agent.heading == "left" else -1.0
    # 보고는 면 사이 간격 — 중심 거리 = 간격 + 로봇 반지름 + 상대 반지름(숨은 값, 사전에서).
    ra = _uniform(rng, spec["radius"], k)
    gap = np.maximum(_normal(rng, agent.ahead, k), 0.005) + robot.radius + ra
    # 옆 위치 축: 시간이 흐르면 direction 쪽으로 움직인다.
    lat0, u0 = _track(rng, obs, k, direction, 0.0)
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
    ahead = np.maximum(_normal(rng, agent.ahead, k), 0.0) + obs.robot.radius
    # 진로까지 거리 축(상대의 가까운 면 → 로봇 진로 가장자리): 시간이 흐르면 줄어든다.
    surface, va = _track(rng, obs, k, -1.0, 0.05)
    ra = _uniform(rng, spec["radius"], k)
    dist = surface + obs.robot.radius + ra
    return {
        "ahead": ahead, "dist": dist, "from_sign": np.full(k, 1.0 if agent.heading == "from_left" else -1.0), "va": va,
        "ra": ra, "intent": _intents(cfg, obs, "crossing", rng, k),
        "yield_standoff": _uniform(rng, motion["yield"]["standoff"], k) * scale,
        "agent_decel": _uniform(rng, spec.get("decel", [1.0, 2.0]), k),
        "hurry_factor": _uniform(rng, motion["hurry"]["factor"], k), "hurry_accel": _uniform(rng, motion["hurry"]["accel"], k),
        "roll_decel": _roll_decel(cfg, obs, rng, k) if spec["group"] == "ball" else np.zeros(k),
    }  # fmt: skip


def _head_on(cfg, obs: Observation, rng, k: int) -> dict[str, np.ndarray]:
    agent = obs.agent
    spec = cfg.agent(agent.kind)
    motion = cfg.catalog["intent_motion"]
    surface, va = _track(rng, obs, k, -1.0, 0.05)
    ra = np.maximum(_normal(rng, agent.size, k) / 2.0, 0.03)
    intent = _intents(cfg, obs, "head_on", rng, k)
    return {
        "dist": surface + obs.robot.radius + ra, "lat_a": _normal(rng, agent.lateral, k),
        "width": np.maximum(_normal(rng, obs.env.corridor_width, k), 0.1),
        "va": va, "ra": ra, "intent": np.maximum(intent, 0.0),
        "aside_speed": _uniform(rng, motion["aside"]["lateral_speed"], k), "keep_standoff": _uniform(rng, motion["keep"]["standoff"], k),
        "agent_decel": _uniform(rng, spec.get("decel", [1.0, 2.0]), k), "reactive": np.full(k, float(spec["reactive"])),
        "roll_decel": _roll_decel(cfg, obs, rng, k) if spec["group"] == "ball" else np.zeros(k),
    }  # fmt: skip


def sample_posterior(cfg: SimgenConfig, obs: Observation, k: int, rng: np.random.Generator) -> Batch:
    """관측에서 사후 세계 K개."""
    if obs.scene in ("p1_step_in", "p1p5_step_in_carrying"):
        values = _step_in(cfg, obs, rng, k)
    elif obs.scene == "p1_crossing":
        values = _crossing(cfg, obs, rng, k)
    elif obs.scene == "p1_head_on":
        values = _head_on(cfg, obs, rng, k)
    elif obs.scene == "p5_braking":
        surface = np.maximum(_normal(rng, obs.env.obstacle_distance, k), 0.005)
        values = {"obstacle": surface + obs.robot.radius, **_container(cfg, obs, rng, k)}
    elif obs.scene == "p5_turning":
        values = _container(cfg, obs, rng, k)
    else:
        raise ValueError(obs.scene)
    return Batch({name: np.asarray(value, dtype=float) for name, value in values.items()})
