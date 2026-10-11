"""P4 안착·정렬 확인 — 관측 쪽: 상태 모형, 닫힌 꼴 사후, 행동 결과, 다시 재기의 가치 (Task v0b 지시서 §1 P4, P4+P6).

**상태와 측정.** 놓기(place): seated(제대로) / perched(가이드·턱 위에 걸침) / tilted(기울어 끼임); 끼우기(insert): inserted / stuck(덜
들어가 걸림) / misaligned(어긋나 끼임). 셋째 상태는 부호 있는 측정(기울기·옆 힘)의 평균이 ±라 속으로 두 하위 상태(+, −)로 나눈다(사전은 반반).
하위 상태 s마다 참 측정값 x_j ~ N(μ_sj, σ_sj), 보고 m_j = x_j + 감각 잡음(σ_ej, 보고 반올림 포함 — 글에 정밀도를 적는다). 그래서

    P(s | m) ∝ 사전(s) × Π_j N(m_j; μ_sj, √(σ_sj² + σ_ej²))     — 닫힌 꼴(표본 없음).

**오래된 카메라(P4+P6).** 카메라 측정(높이·기울기, 또는 깊이)은 a초 전 상태의 것이고 힘은 지금 것이다. 그동안 상태는 위험률 λ(숨은 값,
범위 균등)로 바뀌었을 수 있다: 바뀌면 사전에서 새로(같은 상태도 나올 수 있다 — 그때 x도 새로). 세계 = (그때 하위 상태, 바뀜, 지금 하위 상태)
— 무게는 사전(그때) × p(카메라 | 그때) × 전이 × p(힘 | 지금), 전이: 그대로 (1 − q̄)·[같음], 바뀜 q̄·사전(지금), q̄ = 1 − E_λ[e^{−λa}].

**행동의 결과**(catalog p4.actions — 하위 상태마다 확률): 그대로 둠(continue), 눌러 넣기·더 세게 밀기(fix·damage), 다시 하기(success),
사람 부르기(언제나 바로잡힘 + 사람 비용), 다시 재기(look_again — :class:`~robo_jev.simgen.voi.GaussianSignal`(준난수 적분): 같은 x를 새 잡음으로; 바뀐
세계는 지금 상태에서 새 x). 사건은 세계마다의 확률이다(ok_after, damage, unseated, human_call) — 정답은 그 무게 평균(정확).

세계 모듈을 import하지 않는다.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from robo_jev.simgen.observation import Observation
from robo_jev.simgen.spec import Batch
from robo_jev.simgen.voi import GaussianSignal

__all__ = ["DIMS", "STATES", "StateModel", "action_outcomes", "available", "logit_good", "posterior_worlds", "state_model", "stale_q"]

STATES = {"place": ("seated", "perched", "tilted"), "insert": ("inserted", "stuck", "misaligned")}
DIMS = {"place": ("height_mm", "tilt_deg", "force_n"), "insert": ("depth_mm", "force_n", "lateral_n")}
SIGNED = {"place": "tilt_deg", "insert": "lateral_n"}
CAMERA_DIMS = {"place": ("height_mm", "tilt_deg"), "insert": ("depth_mm",)}
OPTIONS = {"place": ("continue", "press_down", "reseat", "call_human", "look_again"), "insert": ("continue", "push_harder", "reseat", "call_human", "look_again")}


@dataclass(frozen=True)
class StateModel:
    """하위 상태 4개(좋음, 둘째, 셋째+, 셋째−)의 사전·평균·표준편차 — 차원별."""

    family: str
    dims: tuple[str, ...]
    prior: np.ndarray            # (4,)
    mean: np.ndarray             # (4, D)
    sd: np.ndarray               # (4, D)
    label_state: np.ndarray      # (4,) 상태 번호 0/1/2(셋째 ±는 2)


def state_model(cfg, family: str, cls_name: str, dims: tuple[str, ...]) -> StateModel:
    cls = cfg.catalog["p4"][family]["classes"][cls_name]
    names = STATES[family]
    prior3 = np.array([cls["prior"][s] for s in names], dtype=float)
    prior = np.array([prior3[0], prior3[1], prior3[2] / 2.0, prior3[2] / 2.0])
    mean = np.zeros((4, len(dims)))
    sd = np.zeros((4, len(dims)))
    for j, d in enumerate(dims):
        for i, s in enumerate((names[0], names[1], names[2], names[2])):
            mu, sigma = cls[d][s]
            sign = -1.0 if (i == 3 and d == SIGNED[family]) else 1.0
            mean[i, j], sd[i, j] = sign * mu, sigma
    return StateModel(family, dims, prior / prior.sum(), mean, sd, np.array([0, 1, 2, 2]))


def _logpdf(x, mean, sd):
    return -0.5 * ((x - mean) / sd) ** 2 - np.log(sd) - 0.5 * math.log(2 * math.pi)


def stale_q(cfg, age: float) -> float:
    """오래된 카메라 동안 상태가 바뀌었을 확률 q̄ = 1 − E_λ[e^{−λa}], λ ~ U(범위)."""
    lo, hi = cfg.catalog["p4"]["stale"]["hazard_per_s"]
    if age <= 0:
        return 0.0
    return 1.0 - (math.exp(-lo * age) - math.exp(-hi * age)) / ((hi - lo) * age)


@dataclass(frozen=True)
class Worlds:
    """사후 세계(정확한 무게): 지금 하위 상태, (오래된 카메라면) 그때 하위 상태·바뀜, 무게."""

    now: np.ndarray        # (W,) 하위 상태 번호
    then: np.ndarray       # (W,)
    changed: np.ndarray    # (W,) bool
    weight: np.ndarray     # (W,) 합 1


def posterior_worlds(model: StateModel, readings: dict, sigma_e: dict, camera_age: float = 0.0, q: float = 0.0) -> Worlds:
    """측정(이름 → 보고값)과 감각 잡음(이름 → σ_eff)에서 사후 세계와 무게. 오래된 카메라면(camera_age > 0) 카메라 차원은 그때, 나머지는 지금."""
    dims = list(model.dims)
    cam = set(CAMERA_DIMS[model.family]) if camera_age > 0 else set()
    def loglik(states: np.ndarray, which: set) -> np.ndarray:
        out = np.zeros(len(states))
        for name, value in readings.items():
            if (name in cam) != (which == "camera"):
                continue
            j = dims.index(name)
            sd = np.sqrt(model.sd[states, j] ** 2 + sigma_e[name] ** 2)
            out += _logpdf(value, model.mean[states, j], sd)
        return out
    subs = np.arange(4)
    if camera_age <= 0:
        logw = np.log(model.prior) + loglik(subs, "now")
        w = np.exp(logw - logw.max())
        return Worlds(subs, subs, np.zeros(4, dtype=bool), w / w.sum())
    then, now, changed, logw = [], [], [], []
    cam_ll = loglik(subs, "camera")
    now_ll = loglik(subs, "now")
    for a in subs:
        then.append(a), now.append(a), changed.append(False)
        logw.append(math.log(model.prior[a]) + cam_ll[a] + math.log(max(1.0 - q, 1e-300)) + now_ll[a])
        for b in subs:
            then.append(a), now.append(b), changed.append(True)
            logw.append(math.log(model.prior[a]) + cam_ll[a] + math.log(max(q, 1e-300)) + math.log(model.prior[b]) + now_ll[b])
    logw = np.array(logw)
    w = np.exp(logw - logw.max())
    return Worlds(np.array(now), np.array(then), np.array(changed), w / w.sum())


def logit_good(worlds: Worlds) -> float:
    p = float(np.sum(worlds.weight[worlds.now == 0]))
    p = min(max(p, 1e-9), 1 - 1e-9)
    return math.log(p / (1.0 - p))


def available(obs: Observation) -> tuple[str, ...]:
    return OPTIONS[obs.place.family]


def _model_of(cfg, obs: Observation) -> StateModel:
    place = obs.place
    return state_model(cfg, place.family, place.cls, tuple(d for d in DIMS[place.family] if d in place.readings))


def _sigma_e(obs: Observation) -> dict:
    return {name: r.sigma_eff for name, r in obs.place.readings.items()}


def _readings(obs: Observation) -> dict:
    return {name: r.value for name, r in obs.place.readings.items()}


def _camera_age(obs: Observation) -> float:
    return float(max(obs.place.ages.values())) if obs.place.ages else 0.0


def sample_posterior(cfg, obs: Observation, k: int, rng: np.random.Generator) -> Batch:
    """사후 세계를 정확한 무게로 나열한다(표본이 아니다). ``k = 1``이면(참 세계) 그 무게로 하나를 뽑고 참 측정값 x도 뽑는다."""
    model = _model_of(cfg, obs)
    age = _camera_age(obs)
    worlds = posterior_worlds(model, _readings(obs), _sigma_e(obs), age, stale_q(cfg, age))
    values = {"state": worlds.now.astype(float), "state_then": worlds.then.astype(float), "changed": worlds.changed.astype(float)}
    if k == 1:
        i = int(np.searchsorted(np.cumsum(worlds.weight), rng.random(), side="right"))
        i = min(i, len(worlds.weight) - 1)
        values = {key: value[i : i + 1] for key, value in values.items()}
        # 참 측정값 x: 같은 세계의 조건부(카메라 차원은 그때 상태, 그대로면 지금도 같은 x)
        for j, name in enumerate(model.dims):
            if name not in obs.place.readings:
                continue
            cam = name in CAMERA_DIMS[model.family] and age > 0
            s = int(values["state_then"][0]) if cam else int(values["state"][0])
            mu, sd = model.mean[s, j], model.sd[s, j]
            se = obs.place.readings[name].sigma_eff
            post_var = 1.0 / (1.0 / sd**2 + 1.0 / se**2)
            post_mean = post_var * (mu / sd**2 + obs.place.readings[name].value / se**2)
            x = post_mean + math.sqrt(post_var) * rng.standard_normal()
            if cam and values["changed"][0] > 0.5:   # 바뀌었으면 지금의 x는 지금 상태에서 새로
                s_now = int(values["state"][0])
                x = model.mean[s_now, j] + model.sd[s_now, j] * rng.standard_normal()
            values[f"x_{name}"] = np.array([x])
        return Batch({key: np.asarray(v, dtype=float) for key, v in values.items()})
    return Batch({key: np.asarray(v, dtype=float) for key, v in values.items()}, weights=worlds.weight)


def action_outcomes(cfg, family: str, option: str, sub_state: np.ndarray) -> dict[str, np.ndarray]:
    """하위 상태마다 행동 결과의 확률: ok(제대로 되고 멀쩡), damage, unseated(제대로 안 되고 멀쩡), human(사람 부름)."""
    spec = cfg.catalog["p4"]["actions"][family][option]
    states = STATES[family]
    label = np.array([0, 1, 2, 2])[sub_state.astype(int)]
    ok = np.zeros(len(sub_state))
    damage = np.zeros(len(sub_state))
    human = np.zeros(len(sub_state))
    for i, s in enumerate(label):
        name = states[s]
        if option == "continue":
            ok[i] = 1.0 if s == 0 else 0.0
        elif option in ("press_down", "push_harder"):
            if s == 0:
                ok[i] = 1.0
            else:
                d = spec.get("damage", {}).get(name, 0.0)
                ok[i] = (1.0 - d) * spec.get("fix", {}).get(name, 0.0)
                damage[i] = d
        elif option == "reseat":
            ok[i] = spec["success"]
        elif option == "call_human":
            ok[i] = 1.0
            human[i] = 1.0
    return {"ok": ok, "damage": damage, "unseated": np.clip(1.0 - ok - damage, 0.0, 1.0), "human": human}


def rollout(cfg, obs: Observation, batch: Batch, options, rng: np.random.Generator | None = None):
    """선택지 × 세계의 결과(확률 사건). ``rng``가 있으면(참 세계) 결과를 뽑아 0/1로."""
    from robo_jev.simgen.rollout import Outcomes

    place = obs.place
    family = place.family
    sub = batch["state"].astype(int)
    k = batch.size
    out = Outcomes(tuple(options))
    ok, damage, unseated, human, delay = [], [], [], [], []
    draws = rng.random((len(options), k)) if rng is not None else None
    for i, option in enumerate(options):
        if option == "look_again":
            for lst in (ok, damage, unseated, human):
                lst.append(np.zeros(k))
            delay.append(np.zeros(k))
            continue
        res = action_outcomes(cfg, family, option, sub)
        if draws is not None:  # 참 세계: 결과 하나를 뽑는다(ok / damage / unseated)
            u = draws[i]
            o = (u < res["ok"]).astype(float)
            d = ((u >= res["ok"]) & (u < res["ok"] + res["damage"])).astype(float)
            res = {"ok": o, "damage": d, "unseated": 1.0 - o - d, "human": res["human"]}
        ok.append(res["ok"]), damage.append(res["damage"]), unseated.append(res["unseated"]), human.append(res["human"])
        delay.append(np.full(k, place.action_times.get(option, 0.0)))
    out.events = {"ok_after": np.array(ok), "damage": np.array(damage), "unseated": np.array(unseated), "human_call": np.array(human)}
    out.quantities = {"delay": np.array(delay), "bad_after": 1.0 - np.array(ok)}
    fail_item = cfg.catalog["p4"]["items"][place.kind]["fail"]
    out.cost_items = {"unseated": fail_item, "damage": "damage", "human_call": "human_call"}
    out.world_events = {"ok_now": (sub == 0).astype(float)}
    out.states = {"p4_state": np.array([0, 1, 2, 2])[sub].astype(float)}
    out.hard_brake = np.zeros(len(options), dtype=bool)
    if "look_again" in options:
        out.voi = _signal(cfg, obs, batch, tuple(o for o in options if o != "look_again"))
    return out


def _signal(cfg, obs: Observation, batch: Batch, base: tuple[str, ...]) -> GaussianSignal:
    """다시 재기의 신호 모형: 세계마다 새 보고의 분포. 사후 세계(무게 있음)면 지금 보고에 조건부; 참 세계(x_ 값이 있음)면 N(x, σ_e)."""
    model = _model_of(cfg, obs)
    place = obs.place
    names = [n for n in model.dims if n in place.readings]
    age = _camera_age(obs)
    k = batch.size
    means = np.zeros((k, len(names)))
    sds = np.zeros((k, len(names)))
    true_world = f"x_{names[0]}" in batch.values
    for j, name in enumerate(names):
        jj = model.dims.index(name)
        se = place.readings[name].sigma_eff
        m = place.readings[name].value
        cam = name in CAMERA_DIMS[model.family] and age > 0
        for i in range(k):
            if true_world:
                means[i, j], sds[i, j] = float(batch[f"x_{name}"][i]), se
                continue
            s_now, s_then, changed = int(batch["state"][i]), int(batch["state_then"][i]), batch["changed"][i] > 0.5
            if cam and changed:  # 바뀐 세계: 지금 상태에서 새 x
                means[i, j], sds[i, j] = model.mean[s_now, jj], math.sqrt(model.sd[s_now, jj] ** 2 + se**2)
                continue
            s = s_then if cam else s_now
            mu, sd = model.mean[s, jj], model.sd[s, jj]
            post_var = 1.0 / (1.0 / sd**2 + 1.0 / se**2)
            post_mean = post_var * (mu / sd**2 + m / se**2)
            means[i, j], sds[i, j] = post_mean, math.sqrt(post_var + se**2)
    return GaussianSignal(base, float(place.action_times["look_again"]), means, sds)


def exact(cfg, obs: Observation) -> dict:
    """닫힌 꼴 정답(사후 세계의 무게로 정확): 지금 제대로인가, 상태 분포."""
    model = _model_of(cfg, obs)
    age = _camera_age(obs)
    worlds = posterior_worlds(model, _readings(obs), _sigma_e(obs), age, stale_q(cfg, age))
    dist = np.array([float(np.sum(worlds.weight[np.array([0, 1, 2, 2])[worlds.now] == s])) for s in range(3)])
    return {"ok_now": dist[0], "p4_state": dist}
