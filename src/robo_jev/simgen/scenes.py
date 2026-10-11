"""장면: 결정 여유에서 참 세계를 풀고(②), 감각 모형으로 관측을 만든다(③) (docs/12 §2·§8, Task v0a 지시서 §3.3–§3.7).

이 모듈만 세계(:class:`~robo_jev.simgen.world.World`)와 관측(:class:`~robo_jev.simgen.observation.Observation`)을 함께 본다.

**결정 여유로 뽑기(지시서 §3.7).** 장면마다 목표 여유 m을 혼합분포 0.6·N(0, σ_m) + 0.4·U(−M, M)에서 뽑고(:func:`draw_margins`),
표면 변수(거리·속도·채움·통로 폭 …)를 그 여유에 **정확히** 맞춰 푼다. 여유의 정의는 catalog의 `margins` 주석, 계산은
:func:`margins_of` (참 세계에서 다시 잰다 — 시험이 목표와 대조한다). 풀 수 없으면 :class:`Infeasible`; 생성기가 다른 갈래로 다시 뽑는다.

**관측(지시서 §3.5).** 감각 구성별 잡음 σ, 보고 해상도(반올림), 관측 나이, 속도를 직접 주기/위치 두 개만 주기, 단서, 옆 공간의 보임 상태,
채움의 거친 말 표현. 잡음은 고정된 순서로 미리 뽑는다(:func:`observe` 의 ``z``·``u`` 풀) — 사실 하나만 바꾼 쌍(:func:`flip_numeric`)이 같은
잡음을 써서 바뀐 사실만 글에서 바뀐다.

**닻 세계.** 여기서 푼 세계는 관측을 만드는 '닻'이다 — 레코드의 참 결과는 정답의 사후에서 따로 뽑은 세계로 만든다(생성기, 리뷰 1 I4).
그래서 여유 집중은 관측(보고된 명목 여유)에 걸리고, 닻을 풀 때의 결합(채움을 감속으로 맞춘 것 등)은 참값에 남지 않는다.

v0 근사(보고서에 적음): 관측 나이 동안 상대는 관측 시점의 속도로 움직였다고 둔다(의도의 효과는 '지금'부터); step_in의 상대는 옆으로만
움직인다(앞뒤 위치 고정). 보고의 기준 시각: step_in·crossing·p5_braking의 앞 거리(간격·교차선·장애물)는 **지금** 값이고(로봇 자신의
이동이 보정된 값), 옆 위치·진로까지 거리(step_in·crossing의 track)는 로봇의 전진과 무관한 축이다. 마주 옴(head-on)의 track은 **그 시각의
둘 사이 간격**이다 — 글이 그렇게 말한다("as of 1.15 s ago the gap between you was …"; 리뷰 1 C1). 그래서 그 시각의 간격 = 지금 간격 +
(상대 속도 + 로봇 속도)·나이이고, 위치 두 개의 차이 / 간격 시간은 다가오는 속도다.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

import numpy as np

from robo_jev.simgen import physics as ph
from robo_jev.simgen.catalog import INTENTS, SCENE_MARGINS, SimgenConfig, reaction_probability
from robo_jev.simgen.observation import (
    AgentObs,
    ContainerObs,
    Descriptor,
    EnvObs,
    Formatting,
    Observation,
    SideObs,
    SkinText,
)
from robo_jev.simgen.skins import compatible, entry_text, setting_allows, setting_tags
from robo_jev.simgen.spec import FILL_BINS, Reading, RobotSpec, fill_bin_index, round_to
from robo_jev.simgen.world import World

__all__ = [
    "FILL_BINS",
    "Context",
    "Infeasible",
    "build_world",
    "draw_context",
    "draw_margins",
    "fill_bin_index",
    "flip_numeric",
    "margins_of",
    "observe",
]

G = ph.G

class Infeasible(Exception):
    """목표 여유를 이 표면 변수 뽑기로는 맞출 수 없다(다른 갈래로 다시 뽑는다)."""


@dataclass(frozen=True)
class Context:
    """레코드의 틀: 분야·몸·장면·변형·상대 종류·감각 구성·서술 문체."""

    domain: str
    body: str
    scene: str
    variant: str
    agent_kind: str | None
    sensor_suite: str
    style: str
    setting: int = 0                  # 분야 카드의 상황 번호(그 태그가 용기·장소·과제·문구를 고른다 — 의미 일관성)


# ----------------------------------------------------------------------------------------------------------------
# 작은 도우미
# ----------------------------------------------------------------------------------------------------------------


def _u(rng: np.random.Generator, lo_hi) -> float:
    return float(rng.uniform(lo_hi[0], lo_hi[1]))


def _lerp(lo_hi, t: float) -> float:
    return float(lo_hi[0] + (lo_hi[1] - lo_hi[0]) * t)


def _pick(rng: np.random.Generator, weights: dict) -> str:
    keys = list(weights)
    p = np.array([weights[k] for k in keys], dtype=float)
    return keys[int(rng.choice(len(keys), p=p / p.sum()))]


def _pick_from(items, u: float):
    """균등 난수 u ∈ [0, 1)로 목록에서 하나(갈래의 소비 수를 고정하려고)."""
    return items[min(int(u * len(items)), len(items) - 1)]


def _speed_for_stop(stop_dist: float, latency: float, decel: float) -> float:
    """멈춤 거리 S = v·t + v²/(2a)를 v에 대해 푼 값."""
    return decel * (-latency + math.sqrt(latency * latency + 2.0 * max(stop_dist, 0.0) / decel))


# ----------------------------------------------------------------------------------------------------------------
# 틀과 여유
# ----------------------------------------------------------------------------------------------------------------


def draw_context(cfg: SimgenConfig, rng: np.random.Generator, domain: str, forced: dict | None = None) -> Context:
    """분야는 주어진 것(층화), 나머지는 분야 카드와 속성 목록의 가중치로 뽑는다. ``forced``는 예시용(body·scene·style·agent_kind)."""
    forced = dict(forced or {})
    card = cfg.domain(domain)
    scene_forced = forced.get("scene")
    bodies = dict(card["bodies"])
    if scene_forced:
        bodies = {b: w for b, w in bodies.items() if scene_forced in cfg.catalog["scenes_by_body"][b]} or bodies
    body = forced.get("body") or _pick(rng, bodies)
    scene = scene_forced or _pick(rng, cfg.catalog["scenes_by_body"][body])
    variant = "moving"
    stopped_draw = rng.random()
    if scene == "p1_crossing" and stopped_draw < cfg.catalog["crossing_stopped_share"]:
        variant = "stopped"
    variant = forced.get("variant", variant)
    family = cfg.agent_family(scene, body)
    agent_kind = None
    if family is not None:
        agent_kind = forced.get("agent_kind") or _pick(rng, card["agents"][family])
    suite = _pick(rng, cfg.body(body)["sensor_suites"])
    style = forced.get("style") or ("facts", "prose", "planner")[int(rng.integers(3))]
    allowed = [i for i, entry in enumerate(card["settings"]) if setting_allows(entry, body)]
    setting = int(forced.get("setting", allowed[int(rng.integers(len(allowed)))]))
    return Context(domain, body, scene, variant, agent_kind, suite, style, setting)


def draw_margin(cfg: SimgenConfig, name: str, body: str, rng: np.random.Generator) -> float:
    """혼합분포 0.6·N(0, σ_m) + 0.4·U(−M, M) 한 번."""
    spec = cfg.catalog["margins"][name]
    sigma, half = spec["sigma"][body], spec["half_range"][body]
    use_normal = rng.random() < cfg.catalog["margins"]["normal_weight"]
    normal, uniform = rng.normal(0.0, sigma), rng.uniform(-half, half)
    return float(normal if use_normal else uniform)


def draw_margins(cfg: SimgenConfig, ctx: Context, rng: np.random.Generator) -> dict[str, float]:
    return {name: draw_margin(cfg, name, ctx.body, rng) for name in SCENE_MARGINS[ctx.scene]}


# ----------------------------------------------------------------------------------------------------------------
# 로봇·용기·상대
# ----------------------------------------------------------------------------------------------------------------


def _robot_params(cfg: SimgenConfig, body: str, rng: np.random.Generator, loaded: bool) -> dict:
    spec = cfg.body(body)
    factor = _u(rng, spec["loaded_decel_factor"]) if loaded and "loaded_decel_factor" in spec else 1.0
    p = {
        "body": body,
        "latency": round_to(_u(rng, spec["latency"]), 0.01),
        "decel_gentle": round_to(_u(rng, spec["decel"]["gentle"]) * factor, 0.1),
        "decel_moderate": round_to(_u(rng, spec["decel"]["moderate"]) * factor, 0.1),
        "decel_hard": round_to(_u(rng, spec["decel"]["hard"]) * factor, 0.1),
        "accel": round_to(_u(rng, spec["accel"]), 0.1),
        "radius": round_to(_u(rng, spec["radius"]), 0.01),
        "can_sidestep": bool(spec["can_sidestep"]),
        "slow_ratio": float(spec["slow_ratio"]),
        "creep_ratio": float(spec["creep_ratio"]),
        "passing_clearance": float(spec.get("passing_clearance", 0.2)),
        "loaded": bool(loaded and "loaded_decel_factor" in spec),
    }
    if spec["can_sidestep"]:
        p["sidestep_offset"] = round_to(_u(rng, spec["sidestep"]["offset"]), 0.05)
        p["sidestep_duration"] = round_to(_u(rng, spec["sidestep"]["duration"]), 0.1)
        p["sidestep_speed_ratio"] = float(spec["sidestep"]["speed_ratio"])
    if "retract" in spec:
        p["retract_speed"] = round_to(_u(rng, spec["retract"]["speed"]), 0.05)
    lo, hi = spec["decel"]["moderate"]
    p["_moderate_range"] = (lo * factor, hi * factor)
    return p


def _make_robot(p: dict, speed: float, cruise: float | None = None) -> RobotSpec:
    fields = {k: v for k, v in p.items() if not k.startswith("_")}
    return RobotSpec(speed=float(speed), cruise=float(cruise if cruise is not None else speed), **fields)


def _carried(cfg: SimgenConfig, ctx: Context, rng: np.random.Generator) -> list[dict]:
    """그 상황·몸·장면에 맞는 용기 항목(분야 카드의 제약 + 속성 목록의 몸 제약)."""
    card = cfg.domain(ctx.domain)
    usable = compatible(card["containers"], setting_tags(card, ctx.setting), ctx.body, ctx.scene)
    return [e for e in usable if ctx.body in cfg.container(e["kind"])["bodies"]]


def _spill_coef(cfg: SimgenConfig, kind: str) -> float:
    """k_ref·(D/2)/(g·H): 가속 1 m/s²당 림 높이 비율로 잰 상승."""
    c = cfg.container(kind)
    return cfg.catalog["margins"]["spill_k_ref"] * (c["diameter"] / 2.0) / (G * c["height"])


def _choose_carry_for_decel(cfg, ctx, m_spill: float, p: dict, rng) -> tuple[dict, float, float]:
    """기준 감속(moderate)에서 쏟아짐 여유가 m_spill이 되도록 (용기 항목, moderate 감속, 채움)을 고른다."""
    entries = _carried(cfg, ctx, rng)
    lo, hi = p["_moderate_range"]
    for idx in rng.permutation(len(entries)):
        entry = entries[int(idx)]
        spec = cfg.container(entry["kind"])
        c = _spill_coef(cfg, entry["kind"])
        f_lo, f_hi = spec["fill"]
        d_a, d_b = max(lo, (1.0 - m_spill - f_hi) / c), min(hi, (1.0 - m_spill - f_lo) / c)
        grid = [round_to(x, 0.1) for x in np.arange(math.ceil(d_a / 0.1 - 1e-9) * 0.1, d_b + 1e-9, 0.1)]
        grid = [d for d in grid if d_a - 1e-9 <= d <= d_b + 1e-9]
        if not grid:
            continue
        d_m = grid[int(rng.integers(len(grid)))]
        fill = 1.0 - m_spill - c * d_m
        if f_lo - 1e-9 <= fill <= f_hi + 1e-9:
            return entry, d_m, float(min(max(fill, f_lo), f_hi))
    raise Infeasible("no container realises the spill margin")


def _container_hidden(cfg, ctx, entry: dict, rng) -> tuple[dict, dict]:
    """출렁임 계수 k, 바닥 기울기, 뚜껑(보임 상태 → 참값)."""
    spec = cfg.container(entry["kind"])
    k = _u(rng, cfg.catalog["slosh_k"])
    tilt = math.radians(float(rng.normal(0.0, cfg.domain(ctx.domain)["tilt_sd_deg"])))
    lid = bool(rng.random() < spec["lid_prior"])
    views = cfg.catalog["lid_view"]["given_lid" if lid else "given_open"]
    view = _pick(rng, views)
    return {"k": k, "tilt": tilt, "lid": float(lid)}, {"lid_view": view}


def _intent(cfg, kind: str, family: str, age: float, rng) -> tuple[int, str | None, int]:
    """관측 시점 의도 ~ 사전, 단서 ~ P(단서 | 그 의도), 관측 뒤 바뀜(위험률) → 지금 의도. 공은 (−1, None, −1)."""
    agent = cfg.agent(kind)
    draws = rng.random(4)
    if agent["group"] == "ball":
        return -1, None, -1
    intents = INTENTS[family]
    prior = np.array([agent["intent_prior"][family][i] for i in intents])
    prior = prior / prior.sum()
    i_obs = int(np.searchsorted(np.cumsum(prior), draws[0] * prior.sum(), side="right"))
    i_obs = min(i_obs, len(intents) - 1)
    table = cfg.catalog["cues"][agent["group"]][family]
    cues = list(table)
    lik = np.array([table[c][intents[i_obs]] for c in cues])
    cue_idx = min(int(np.searchsorted(np.cumsum(lik / lik.sum()), draws[1], side="right")), len(cues) - 1)
    changed = draws[2] < 1.0 - math.exp(-agent["hazard_per_s"] * age)
    i_alt = min(int(np.searchsorted(np.cumsum(prior), draws[3], side="right")), len(intents) - 1)
    return i_obs, cues[cue_idx], (i_alt if changed else i_obs)


def _roll_decel(cfg, ctx, rng) -> float:
    floor = _pick(rng, cfg.domain(ctx.domain)["floor"])
    return _u(rng, cfg.catalog["floors"][floor]) * G


def _age(cfg, rng) -> float:
    spec = cfg.catalog["observation"]["age"]
    fresh = rng.random() < spec["fresh_share"]
    a, b = _u(rng, spec["fresh"]), _u(rng, spec["stale"])
    return round_to(a if fresh else b, 0.05)


def _sides(cfg, ctx, can_sidestep: bool, rng) -> tuple[tuple[str, str], tuple[float, float]]:
    """옆 공간 (보임 상태, 참 장애물) — 막힘은 장애물, 가림은 분야 사전 확률로 장애물."""
    if not can_sidestep:
        return ("none", "none"), (0.0, 0.0)
    views, truth = [], []
    prior = cfg.domain(ctx.domain)["side_obstacle_prior"]
    for _ in range(2):
        view = _pick(rng, cfg.catalog["observation"]["side_view"])
        hidden = rng.random() < prior
        views.append(view)
        truth.append(float(view == "blocked" or (view == "occluded" and hidden)))
    return (views[0], views[1]), (truth[0], truth[1])


# ----------------------------------------------------------------------------------------------------------------
# 장면별 참 세계
# ----------------------------------------------------------------------------------------------------------------


def _solve_speed_for_gap(cfg, ctx, p: dict, r_sum: float, m: float, d_m: float, rng) -> tuple[float, float]:
    """step_in 여유 m = gap − 멈춤 거리(v0, moderate) − 반지름 합을 맞추는 (v0, gap)."""
    body = cfg.body(ctx.body)
    t = p["latency"]
    s_lo = max(0.0, body["gap_min"] - m)
    s_hi = body["range_max"] - r_sum - m
    if s_hi <= s_lo:
        raise Infeasible("gap range empty")
    v_lo = max(body["speed"][0], _speed_for_stop(s_lo, t, d_m))
    v_hi = min(body["speed"][1], _speed_for_stop(s_hi, t, d_m))
    if v_lo > v_hi:
        raise Infeasible("no speed realises the step-in margin")
    v0 = round_to(rng.uniform(v_lo, v_hi), body["speed_resolution"])
    v0 = min(max(v0, body["speed"][0]), body["speed"][1])
    gap = m + ph.stopping_distance(v0, t, d_m) + r_sum
    if gap - r_sum < 0.5 * body["gap_min"] or gap > body["range_max"] + 0.5:
        raise Infeasible("rounded speed broke the gap")
    return v0, gap


def _step_in_agent(cfg, ctx, ra: float, r_sum: float, rng) -> tuple[dict, dict]:
    agent = cfg.agent(ctx.agent_kind)
    motion = cfg.catalog["intent_motion"]
    direction = 1.0 if rng.random() < 0.5 else -1.0
    offset = rng.uniform(0.0, cfg.catalog["intent_motion"]["step_in_band"]) * r_sum  # 글의 전제 띠(사후와 같은 띠)
    lat0 = -direction * offset
    if agent["group"] == "ball":
        u0 = _u(rng, motion["ball_lateral_speed"])
    else:
        u0 = _u(rng, motion["step_in_lateral_speed"]) * agent["speed"][1]
    age = _age(cfg, rng)
    # 마지막 위치 보고가 진로 중심선에서 last_fix_max보다 멀지 않게 나이를 줄인다(리뷰 1 질문 5: '끼어든다'는 글과 2 m 밖의 보고가 어긋나 보인다).
    age_max = max(0.0, (cfg.body(ctx.body)["last_fix_max"] - offset) / u0)
    if age > age_max:
        age = round_to(math.floor(age_max / 0.05) * 0.05, 0.05)
    i_obs, cue, i_now = _intent(cfg, ctx.agent_kind, "step_in", age, rng)
    decel = _u(rng, agent.get("decel", [1.0, 2.0]))
    cross_speed = max(_u(rng, agent["speed"]), u0)
    values = {
        "lat0": lat0, "dir": direction, "u0": u0, "ra": ra, "intent": float(i_now),
        "stop_decel": decel, "stay_s": _u(rng, motion["stay"]["stay_s"]), "cross_speed": cross_speed,
        "cross_accel": _u(rng, motion["cross"]["accel"]), "back_decel": decel,
        "back_speed": _u(rng, motion["back"]["speed_factor"]) * cross_speed,
        "roll_decel": _roll_decel(cfg, ctx, rng) if agent["group"] == "ball" else 0.0,
    }  # fmt: skip
    return values, {"intent_obs": i_obs, "cue": cue, "age": age}


def _build_step_in(cfg, ctx, margins, rng, *, carrying: bool) -> World:
    p = _robot_params(cfg, ctx.body, rng, loaded=carrying)
    agent = cfg.agent(ctx.agent_kind)
    ra = _u(rng, agent["radius"])
    r_sum = p["radius"] + ra
    entry, hidden_c, at_c = None, {}, {}
    if carrying:
        entry, p["decel_moderate"], fill = _choose_carry_for_decel(cfg, ctx, margins["spill"], p, rng)
        hidden_c, at_c = _container_hidden(cfg, ctx, entry, rng)
        hidden_c["fill"] = fill
    v0, gap = _solve_speed_for_gap(cfg, ctx, p, r_sum, margins["step_in"], p["decel_moderate"], rng)
    values, at_obs = _step_in_agent(cfg, ctx, ra, r_sum, rng)
    views, sides = _sides(cfg, ctx, p["can_sidestep"], rng)
    values.update({"gap": gap, "side_left": sides[0], "side_right": sides[1]}, **hidden_c)
    at_obs.update(sides=views, **at_c)
    world = World(ctx.scene, ctx.domain, ctx.variant, _make_robot(p, v0), ctx.agent_kind, entry, values, at_obs)
    return _with_margins(cfg, world)


def _build_crossing(cfg, ctx, margins, rng) -> World:
    p = _robot_params(cfg, ctx.body, rng, loaded=False)
    body, agent = cfg.body(ctx.body), cfg.agent(ctx.agent_kind)
    motion = cfg.catalog["intent_motion"]
    ra = _u(rng, agent["radius"])
    r_sum = p["radius"] + ra
    m = margins["crossing"]
    safety = cfg.catalog["margins"]["safety_s"]
    scale = motion["arm_scale"] if ctx.body == "arm" else 1.0
    # 상대가 지나는 줄(교차선)은 로봇 앞면 너머에 둔다 — 로봇이 이미 그 줄을 막고 있지 않게(멈춤이 길을 막는 경우를 만들지 않는다).
    ahead_min = r_sum + body["gap_min"]
    if ctx.variant == "stopped":
        cruise = round_to(_u(rng, body["speed"]), body["speed_resolution"])
        ahead = round_to(ahead_min + rng.uniform(0.0, 0.8) * scale, 0.01)
        t_clear = ph.clear_time_from_rest(ahead + r_sum, p["latency"], p["accel"], cruise)
        v0 = 0.0
    else:
        v0 = round_to(_u(rng, body["speed"]), body["speed_resolution"])
        cruise = v0
        t_lo = max(0.8, 0.3 - m - safety, (ahead_min + r_sum) / v0)
        if t_lo > 2.6:
            raise Infeasible("too slow to clear the crossing within the horizon")
        t_clear = rng.uniform(t_lo, 2.6)
        ahead = v0 * t_clear - r_sum
    t_arr = m + t_clear + safety
    if t_arr < 0.3:
        raise Infeasible("agent would already be at the path")
    va_cap = (body["range_max"] - r_sum) / t_arr
    if va_cap < agent["speed"][0]:
        raise Infeasible("agent beyond sensor range")
    va = rng.uniform(agent["speed"][0], min(agent["speed"][1], va_cap))
    dist = r_sum + va * t_arr
    age = _age(cfg, rng)
    i_obs, cue, i_now = _intent(cfg, ctx.agent_kind, "crossing", age, rng)
    react_draw = rng.random()
    values = {
        "ahead": ahead, "dist": dist, "from_sign": 1.0 if rng.random() < 0.5 else -1.0, "va": va, "ra": ra,
        "intent": float(i_now), "yield_standoff": _u(rng, motion["yield"]["standoff"]) * scale,
        "agent_decel": _u(rng, agent.get("decel", [1.0, 2.0])), "hurry_factor": _u(rng, motion["hurry"]["factor"]),
        "hurry_accel": _u(rng, motion["hurry"]["accel"]),
        "roll_decel": _roll_decel(cfg, ctx, rng) if agent["group"] == "ball" else 0.0,
        # 로봇이 자기 줄을 막고 있으면 늦게라도 서나(단서에 묶인 확률; 리뷰 1 질문 4) — 숨은 값
        "reacts": float(react_draw < reaction_probability(cfg, ctx.agent_kind, cue)),
        "react_standoff": _u(rng, cfg.catalog["crossing_reaction"]["standoff"]) * scale,
    }  # fmt: skip
    world = World(ctx.scene, ctx.domain, ctx.variant, _make_robot(p, v0, cruise), ctx.agent_kind, None, values, {"intent_obs": i_obs, "cue": cue, "age": age})
    return _with_margins(cfg, world)


def _build_head_on(cfg, ctx, margins, rng) -> World:
    p = _robot_params(cfg, ctx.body, rng, loaded=False)
    body, agent = cfg.body(ctx.body), cfg.agent(ctx.agent_kind)
    motion = cfg.catalog["intent_motion"]
    ra = _u(rng, agent["radius"])
    r_sum = p["radius"] + ra
    v0 = round_to(_u(rng, body["speed"]), body["speed_resolution"])
    width = margins["head_on"] + p["passing_clearance"] + 2.0 * r_sum
    t_meet = rng.uniform(1.2, 2.6)
    va_cap = (body["range_max"] - r_sum) / t_meet - v0
    if va_cap < agent["speed"][0]:
        raise Infeasible("oncoming agent beyond range")
    va = rng.uniform(agent["speed"][0], min(agent["speed"][1], va_cap))
    dist = r_sum + (v0 + va) * t_meet
    robot_lateral = round_to(rng.uniform(-0.1, 0.1) * (width / 2.0 - p["radius"]), 0.01)
    lat_a = rng.uniform(-0.15, 0.15) * max(width / 2.0 - ra, 0.0)
    age = _age(cfg, rng)
    i_obs, cue, i_now = _intent(cfg, ctx.agent_kind, "head_on", age, rng)
    values = {
        "dist": dist, "lat_a": lat_a, "width": width, "va": va, "ra": ra,
        "intent": float(i_now if i_now >= 0 else 0), "aside_speed": _u(rng, motion["aside"]["lateral_speed"]),
        "keep_standoff": _u(rng, motion["keep"]["standoff"]), "agent_decel": _u(rng, agent.get("decel", [1.0, 2.0])),
        "reactive": float(agent["reactive"]), "roll_decel": _roll_decel(cfg, ctx, rng) if agent["group"] == "ball" else 0.0,
    }  # fmt: skip
    world = World(
        ctx.scene, ctx.domain, ctx.variant, _make_robot(p, v0), ctx.agent_kind, None, values,
        {"intent_obs": i_obs, "cue": cue, "age": age}, geometry={"robot_lateral": robot_lateral},
    )  # fmt: skip
    return _with_margins(cfg, world)


def _build_braking(cfg, ctx, margins, rng) -> World:
    p = _robot_params(cfg, ctx.body, rng, loaded=True)
    body = cfg.body(ctx.body)
    entry, p["decel_moderate"], fill = _choose_carry_for_decel(cfg, ctx, margins["spill"], p, rng)
    hidden_c, at_c = _container_hidden(cfg, ctx, entry, rng)
    m = margins["stop"]
    t = p["latency"]
    s_lo = max(0.0, body["gap_min"] - m)
    s_hi = body["range_max"] - p["radius"] - m
    v_lo = max(body["speed"][0], _speed_for_stop(s_lo, t, p["decel_moderate"]))
    v_hi = min(body["speed"][1], _speed_for_stop(s_hi, t, p["decel_moderate"]))
    if v_lo > v_hi:
        raise Infeasible("no speed realises the stop margin")
    v0 = min(max(round_to(rng.uniform(v_lo, v_hi), body["speed_resolution"]), body["speed"][0]), body["speed"][1])
    obstacle = m + p["radius"] + ph.stopping_distance(v0, t, p["decel_moderate"])
    if obstacle - p["radius"] < 0.5 * body["gap_min"]:
        raise Infeasible("rounded speed broke the stop distance")
    values = {"obstacle": obstacle, "fill": fill, **hidden_c}
    world = World(ctx.scene, ctx.domain, ctx.variant, _make_robot(p, v0), None, entry, values, dict(at_c))
    return _with_margins(cfg, world)


def _build_turning(cfg, ctx, margins, rng) -> World:
    p = _robot_params(cfg, ctx.body, rng, loaded=True)
    body = cfg.body(ctx.body)
    m = margins["spill"]
    radius_res = 0.01 if ctx.body == "arm" else 0.05
    entries = _carried(cfg, ctx, rng)
    for idx in rng.permutation(len(entries)):
        entry = entries[int(idx)]
        f_lo, f_hi = cfg.container(entry["kind"])["fill"]
        c = _spill_coef(cfg, entry["kind"])
        for _ in range(40):
            v0 = min(max(round_to(_u(rng, body["speed"]), body["speed_resolution"]), body["speed"][0]), body["speed"][1])
            radius = round_to(_u(rng, body["turn_radius"]), radius_res)
            fill = 1.0 - m - c * v0 * v0 / radius
            if f_lo <= fill <= f_hi:
                hidden_c, at_c = _container_hidden(cfg, ctx, entry, rng)
                angle = float((45.0, 90.0, 135.0)[int(rng.choice(3, p=[0.2, 0.6, 0.2]))])
                approach = round_to(_u(rng, body["turn_approach"]), 0.01 if ctx.body == "arm" else 0.05)
                values = {"fill": fill, **hidden_c}
                geometry = {"radius": radius, "angle_deg": angle, "approach": approach}
                world = World(ctx.scene, ctx.domain, ctx.variant, _make_robot(p, v0), None, entry, values, dict(at_c), geometry=geometry)
                return _with_margins(cfg, world)
    raise Infeasible("no container/turn realises the spill margin")


def build_world(cfg: SimgenConfig, ctx: Context, margins: dict[str, float], rng: np.random.Generator) -> World:
    """목표 여유에 맞춘 참 세계 하나."""
    if ctx.scene == "p1_step_in":
        return _build_step_in(cfg, ctx, margins, rng, carrying=False)
    if ctx.scene == "p1p5_step_in_carrying":
        return _build_step_in(cfg, ctx, margins, rng, carrying=True)
    if ctx.scene == "p1_crossing":
        return _build_crossing(cfg, ctx, margins, rng)
    if ctx.scene == "p1_head_on":
        return _build_head_on(cfg, ctx, margins, rng)
    if ctx.scene == "p5_braking":
        return _build_braking(cfg, ctx, margins, rng)
    if ctx.scene == "p5_turning":
        return _build_turning(cfg, ctx, margins, rng)
    raise ValueError(ctx.scene)


# ----------------------------------------------------------------------------------------------------------------
# 여유 재기(참 세계에서)
# ----------------------------------------------------------------------------------------------------------------


def margins_of(cfg: SimgenConfig, world: World) -> dict[str, float]:
    """참 세계의 결정 여유(정의는 catalog `margins` 주석). 풀이와 따로 계산한다."""
    r, v = world.robot, world.values
    out: dict[str, float] = {}
    for name in SCENE_MARGINS[world.scene]:
        if name == "step_in":
            out[name] = v["gap"] - ph.stopping_distance(r.speed, r.latency, r.decel_moderate) - (r.radius + v["ra"])
        elif name == "crossing":
            r_sum = r.radius + v["ra"]
            t_arr = ph.arrival_time(v["dist"], r_sum, v["va"])
            if world.variant == "stopped":
                t_clear = ph.clear_time_from_rest(v["ahead"] + r_sum, r.latency, r.accel, r.cruise)
            else:
                t_clear = ph.clear_time(v["ahead"], r_sum, r.speed)
            out[name] = t_arr - (t_clear + cfg.catalog["margins"]["safety_s"])
        elif name == "head_on":
            out[name] = v["width"] - 2.0 * (r.radius + v["ra"]) - r.passing_clearance
        elif name == "spill":
            c = _spill_coef(cfg, world.container["kind"])
            accel = r.speed**2 / world.geometry["radius"] if world.scene == "p5_turning" else r.decel_moderate
            out[name] = 1.0 - v["fill"] - c * accel
        elif name == "stop":
            out[name] = v["obstacle"] - r.radius - ph.stopping_distance(r.speed, r.latency, r.decel_moderate)
    return out


def _with_margins(cfg: SimgenConfig, world: World) -> World:
    return replace(world, margins=margins_of(cfg, world))


# ----------------------------------------------------------------------------------------------------------------
# 사실 하나만 바꾼 쌍(numeric)
# ----------------------------------------------------------------------------------------------------------------


def flip_numeric(cfg: SimgenConfig, world: World, rng: np.random.Generator):
    """결정 변수 하나만 경계 너머로 옮긴 세계와 그 여유 이름의 후보들(앞의 것부터 — 생성기가 관측이 실제로 바뀌는 첫 후보를 고른다).
    다른 값은 그대로다. 옮길 수 없으면 아무것도 내지 않는다."""
    names = list(SCENE_MARGINS[world.scene])
    name = names[int(rng.integers(len(names)))]
    m = world.margins[name]
    sigma = cfg.catalog["margins"][name]["sigma"][world.robot.body]
    sign = 1.0 if m >= 0 else -1.0
    r, v = world.robot, world.values
    body = cfg.body(r.body)
    targets = (-sign * max(abs(m), sigma), -sign * 0.5 * sigma, -sign * max(abs(m), 0.25 * sigma), -sign * max(abs(m), 2.0 * sigma), -sign * max(abs(m), 3.0 * sigma))
    for target in targets:
        delta = target - m
        changes: dict[str, float] = {}
        if name == "step_in":
            changes["gap"] = v["gap"] + delta
            ok = changes["gap"] - (r.radius + v["ra"]) >= 0.5 * body["gap_min"]
        elif name == "stop":
            changes["obstacle"] = v["obstacle"] + delta
            ok = changes["obstacle"] - r.radius >= 0.5 * body["gap_min"]
        elif name == "crossing":
            changes["dist"] = v["dist"] + v["va"] * delta
            ok = ph.arrival_time(changes["dist"], r.radius + v["ra"], v["va"]) >= 0.3 and changes["dist"] <= body["range_max"]
        elif name == "head_on":
            changes["width"] = v["width"] + delta
            ok = changes["width"] >= 2.0 * max(r.radius, v["ra"]) + 0.1
        else:  # spill: 채움만 옮긴다
            changes["fill"] = v["fill"] - delta
            f_lo, f_hi = cfg.container(world.container["kind"])["fill"]
            ok = f_lo <= changes["fill"] <= f_hi
        if ok:
            flipped = world.with_values(**changes)
            yield _with_margins(cfg, flipped), name


# ----------------------------------------------------------------------------------------------------------------
# 관측(감각 모형)
# ----------------------------------------------------------------------------------------------------------------


def _distance_sigma(cfg, suite: str, distance: float, u: float) -> float:
    spec = cfg.catalog["sensor_suites"][suite]["distance_sigma"]
    if isinstance(spec, dict):
        per_m = _lerp(spec["per_m"], u)
        return float(min(max(per_m * distance, spec["min"]), spec["max"]))
    return _lerp(spec, u)


def _reading(true_value: float, sigma: float, resolution: float, z: float, floor: float | None = None) -> Reading:
    """날 값 = 참값 + σ·z를 해상도로 반올림. 바닥이 있으면 바닥 + 해상도/2보다 작은 날 값은 바닥으로 보고한다(검열 — Reading.censored)."""
    value = true_value + sigma * z
    if floor is not None and value < floor + resolution / 2.0:
        return Reading(round_to(floor, resolution), float(sigma), float(resolution), float(floor))
    return Reading(round_to(value, resolution), float(sigma), float(resolution), None if floor is None else float(floor))


def _formatting(ctx: Context, rng: np.random.Generator) -> Formatting:
    u = rng.random(3)
    if ctx.body == "arm":
        distance = _pick_from(("cm", "cm", "mm", "m"), u[0])
        speed = _pick_from(("m/s", "mm/s"), u[1])
    else:
        distance = _pick_from(("m", "m", "m", "cm"), u[0])
        speed = _pick_from(("m/s", "m/s", "km/h"), u[1]) if ctx.body == "forklift_amr" else _pick_from(("m/s",) * 6 + ("km/h",), u[1])
    time = _pick_from(("s", "s", "s", "ms"), u[2])
    return Formatting(distance, speed, time)


def _skin(cfg, ctx: Context, world: World, rng: np.random.Generator) -> dict:
    """상황 옷 문구 고르기(쌍의 'wording' 변형은 이 갈래만 바꾼다). 상황은 틀이 정했다 — 목록마다 그 상황·몸·장면에 맞는 항목만 쓴다."""
    card = cfg.domain(ctx.domain)
    tags = setting_tags(card, ctx.setting)

    def usable(entries):
        return compatible(entries, tags, ctx.body, ctx.scene)

    u = rng.random(16)
    names = usable(card.get("body_names", {}).get(ctx.body) or cfg.skins["body_names"][ctx.body])
    name = _pick_from(names, u[1])
    carry_scene = world.container is not None
    own_tasks = name.get("carry_tasks" if carry_scene else "tasks") if isinstance(name, dict) else None
    tasks = usable(own_tasks) if own_tasks else usable(card["carry_tasks" if carry_scene else "tasks"][ctx.body])
    places = card["places"]
    key = "bench" if ctx.body == "arm" else {"p1_crossing": "crossing", "p1_head_on": "corridor", "p5_turning": "turn"}.get(ctx.scene, "floor")
    out = {
        "setting": entry_text(card["settings"][ctx.setting]),
        "body_name": entry_text(name),
        "carry_where": (name.get("carry") if isinstance(name, dict) else None) or cfg.skins["carry_default"][ctx.body],
        "place": entry_text(_pick_from(usable(places[key]), u[2])),
    }
    container_phrase = None
    task = entry_text(_pick_from(tasks, u[4]))
    if carry_scene:
        container_phrase = _pick_from(world.container["phrases"], u[3])
        task = task.format(container=container_phrase)
    out["task"] = task
    out["container_phrase"] = container_phrase
    if ctx.agent_kind is not None:
        d = _pick_from(usable(card["descriptors"][ctx.agent_kind]), u[5])
        out["descriptor"] = Descriptor(d["np"], d["ref"])
        cue = world.at_observation.get("cue")
        if cue is not None:
            group = cfg.agent(ctx.agent_kind)["group"]
            template = _pick_from(cfg.skins["cue_phrases"][group][cue], u[6])
            out["cue_phrase"] = template.format(Ref=_cap(d["ref"]), ref=d["ref"])
    out["obstacle"] = entry_text(_pick_from(usable(card["bench_obstacles"] if ctx.body == "arm" else card["obstacles"]), u[7]))
    occluders = [entry_text(e) for e in usable(card["occluders"])]
    blockers = [entry_text(e) for e in usable(card["blockers"])]
    occluders = [occluders[int(i)] for i in rng.permutation(len(occluders))]
    blockers = [blockers[int(i)] for i in rng.permutation(len(blockers))]
    out["occluders"] = (occluders[0], occluders[1 % len(occluders)])
    out["blockers"] = (blockers[0], blockers[1 % len(blockers)])
    distractors = [entry_text(e) for e in usable(card["distractors"])]
    n_distract = _pick_from((0, 1, 1, 2, 2, 3), u[12])
    order = rng.permutation(len(distractors))
    out["distractors"] = tuple(distractors[int(i)] for i in order[:n_distract])
    return out


def _cap(text: str) -> str:
    return text[:1].upper() + text[1:]


def observe(
    cfg: SimgenConfig,
    world: World,
    ctx: Context,
    obs_rng: np.random.Generator,
    fmt_rng: np.random.Generator,
    skin_rng: np.random.Generator,
) -> Observation:
    """참 세계 → 관측. 잡음 풀(z, u)을 고정된 순서로 먼저 뽑는다."""
    suite = cfg.catalog["sensor_suites"][ctx.sensor_suite]
    obs_spec = cfg.catalog["observation"]
    u = obs_rng.random(16)
    z = obs_rng.standard_normal(16)
    skin = _skin(cfg, ctx, world, skin_rng)
    fmt = _formatting(ctx, fmt_rng)
    arm = ctx.body == "arm"
    dist_res = _pick_from((0.005, 0.01), u[4]) if arm else _pick_from((0.01, 0.01, 0.05, 0.1), u[4])
    lat_res = _pick_from((0.005, 0.01), u[9]) if arm else _pick_from((0.01, 0.05), u[9])
    speed_res = _pick_from((0.01, 0.05), u[5]) if arm else _pick_from((0.05, 0.05, 0.1), u[5])
    two_fix = u[6] < obs_spec["two_fix_share"]
    fix_dt = round_to(_lerp(obs_spec["two_fix_dt"], u[7]), 0.1)
    lat_sigma = _lerp(suite["lateral_sigma"], u[1])
    speed_sigma = _lerp(suite["speed_sigma"], u[2])
    width_sigma = _lerp(suite["width_sigma"], u[3])
    v, r = world.values, world.robot

    agent = None
    if ctx.agent_kind is not None:
        group = cfg.agent(ctx.agent_kind)["group"]
        age = world.at_observation["age"]
        ages = (round_to(age + fix_dt, 0.05), age) if two_fix else (age,)
        # 거리는 감각이 재는 면 사이 간격으로 보고한다(로봇 앞면 → 상대의 가까운 면; 진로 가장자리 → 상대의 가까운 면).
        # 중심 거리로 되돌리려면 상대의 반지름이 필요하다 — step_in·crossing에서는 숨은 값(사전 분포), head_on에서는 폭 보고에서.
        r_sum = r.radius + v["ra"]
        if ctx.scene in ("p1_step_in", "p1p5_step_in_carrying"):
            coord_now, rate = v["lat0"], -v["dir"] * v["u0"]
            track_sigma, track_res = lat_sigma, lat_res
            surface = v["gap"] - r_sum
            ahead = _reading(surface, _distance_sigma(cfg, ctx.sensor_suite, surface, u[0]), dist_res, z[0], floor=dist_res)
            heading = "left" if v["dir"] > 0 else "right"
            speed_true = v["u0"]
            extra = {"ahead": ahead}
        elif ctx.scene == "p1_crossing":
            coord_now, rate = v["dist"] - r_sum, v["va"]
            track_sigma, track_res = _distance_sigma(cfg, ctx.sensor_suite, coord_now, u[0]), dist_res
            line = v["ahead"] - r.radius
            ahead = _reading(line, _distance_sigma(cfg, ctx.sensor_suite, line, u[0]), dist_res, z[0], floor=dist_res)
            heading = "from_left" if v["from_sign"] > 0 else "from_right"
            speed_true = v["va"]
            extra = {"ahead": ahead}
        else:  # p1_head_on
            # 리뷰 1 C1: 글은 "a초 전 둘 사이 간격이 d였다"고 말한다 — d는 **그 시각의** 간격이다. 그동안 상대도 로봇도 다가왔으므로
            # 그 시각의 간격 = 지금 간격 + (상대 속도 + 로봇 속도)·a. 위치 두 개의 차이 / 간격 시간 = 다가오는 속도(상대 + 로봇).
            coord_now, rate = v["dist"] - r_sum, v["va"] + r.speed
            track_sigma, track_res = _distance_sigma(cfg, ctx.sensor_suite, coord_now, u[0]), dist_res
            heading = "toward"
            speed_true = v["va"]
            extra = {
                "lateral": _reading(v["lat_a"], lat_sigma, lat_res, z[6]),
                "size": _reading(2.0 * v["ra"], width_sigma, lat_res, z[5], floor=0.05),
            }
        fixes = []
        noise = (z[1], z[2]) if two_fix else (z[2],)
        lateral_track = ctx.scene in ("p1_step_in", "p1p5_step_in_carrying")  # 옆 위치는 부호가 있는 값 — 바닥이 없다
        for a, zz in zip(ages, noise, strict=True):
            fixes.append(_reading(coord_now + rate * a, track_sigma, track_res, zz, floor=None if lateral_track else track_res))
        speed = None if two_fix else _reading(speed_true, speed_sigma, speed_res, z[3], floor=speed_res)
        agent = AgentObs(
            kind=ctx.agent_kind, group=group, descriptor=skin["descriptor"], track=tuple(fixes), track_ages=ages,
            speed=speed, heading=heading, cue=world.at_observation.get("cue"), cue_phrase=skin.get("cue_phrase"), **extra,
        )  # fmt: skip

    container = None
    if world.container is not None:
        spec = cfg.container(world.container["kind"])
        sigma_f = cfg.catalog["fill_sigma"]["transparent" if spec["transparent"] else "opaque"]
        raw = min(max(v["fill"] + sigma_f * z[7], 0.0), 1.0)
        if u[8] < 0.5:
            idx = fill_bin_index(raw)
            lo, hi, _ = FILL_BINS[idx]
            fill = Reading(round((lo + hi) / 2.0, 4), sigma_f, round(hi - lo, 4))
            mode = "phrase"
        else:
            fill = Reading(round_to(raw, 0.05), sigma_f, 0.05)
            mode = "percent"
        container = ContainerObs(
            kind=world.container["kind"], content=world.container["content"], phrase=skin["container_phrase"],
            diameter=spec["diameter"], height=spec["height"], fill=fill, fill_mode=mode, lid=world.at_observation["lid_view"],
        )  # fmt: skip

    env_kwargs: dict = {"place": skin["place"]}
    if ctx.scene in ("p1_step_in", "p1p5_step_in_carrying") and r.can_sidestep:
        views = world.at_observation["sides"]
        sides = []
        for i, view in enumerate(views):
            phrase = {"occluded": skin["occluders"][i], "blocked": skin["blockers"][i]}.get(view)
            sides.append(SideObs(view, phrase))
        env_kwargs["sides"] = (sides[0], sides[1])
    if ctx.scene == "p1_head_on":
        env_kwargs["corridor_width"] = _reading(v["width"], _lerp(suite["width_sigma"], u[3]), lat_res, z[4], floor=0.1)
        env_kwargs["robot_lateral"] = world.geometry["robot_lateral"]
    if ctx.scene == "p5_braking":
        env_kwargs["obstacle"] = skin["obstacle"]
        surface = v["obstacle"] - r.radius
        env_kwargs["obstacle_distance"] = _reading(surface, _distance_sigma(cfg, ctx.sensor_suite, surface, u[0]), dist_res, z[8], floor=dist_res)
    if ctx.scene == "p5_turning":
        g = world.geometry
        env_kwargs.update(turn_radius=g["radius"], turn_angle_deg=g["angle_deg"])
        env_kwargs["turn_approach"] = Reading(g["approach"], 0.0, 0.01 if arm else 0.05)

    skin_text = SkinText(ctx.domain, skin["setting"], skin["body_name"], skin["task"], skin["distractors"], skin["carry_where"])
    return Observation(
        scene=ctx.scene, domain=ctx.domain, body=ctx.body, sensor_suite=ctx.sensor_suite, variant=ctx.variant,
        robot=world.robot, agent=agent, container=container, env=EnvObs(**env_kwargs), skin=skin_text, fmt=fmt,
    )  # fmt: skip
