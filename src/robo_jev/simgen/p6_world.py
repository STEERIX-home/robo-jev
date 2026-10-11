"""P6 오래된 관측·가림·센서 불일치의 세계 쪽: 닻 세계, 여유, 사실 하나 바꾸기, 감각 모형 (Task v0b 지시서 §1 P6, P1+P6).

결정 여유(catalog margins):
* sight(오래된 추적) = d_vis − (반응 거리 + 최대 제동 거리) — 다시 보이는 거리 안에 설 수 있나(지금 속도로). d_vis를 여유에 맞춰 푼다.
* corner(모퉁이) = d_corner − (반응 + 최대 제동) — 지금 누가 나오면 보이는 거리(모퉁이까지) 안에 설 수 있나(지시서 §1 '보이는 거리 안에
  설 수 있는 속도'). 모퉁이까지 거리 d_corner를 여유에 맞춰 푼다; 가림 가장자리 → 진로 가장자리 옆 거리 w는 범위에서(빼 둔 칸: 학습
  w ≥ 0.5 m, T3 < 0.5 m).
* late(센서 불일치) = d_late − (반응 + 최대 제동) — 있다면 늦게 보고도 서나. 숨은 d_late에 맞춰 로봇 속도를 푼다.
* step_in(P1+P6) = P1과 같다(첫 상대); 둘째 상대는 사전(따라옴 확률·시각·속도)에서.

세계 쪽이다(scenes와 같은 층).
"""

from __future__ import annotations

import math

import numpy as np

from robo_jev.simgen import physics as ph
from robo_jev.simgen.catalog import SimgenConfig
from robo_jev.simgen.observation import ConflictObs, CornerObs, Descriptor, StaleObs
from robo_jev.simgen.scenes import (
    Context,
    Infeasible,
    _make_robot,
    _reading,
    _robot_params,
    _sides,
    _solve_speed_for_gap,
    _speed_for_stop,
    _step_in_agent,
    _u,
    speed_range,
)
from robo_jev.simgen.skins import compatible, setting_tags
from robo_jev.simgen.spec import round_to
from robo_jev.simgen.world import World

__all__ = ["build", "flip", "follower_reaction", "margins_of", "observe_parts", "prior_redraw", "truth_parts"]


def truth_parts(cfg: SimgenConfig, obs) -> tuple[dict, dict, dict]:
    """참 세계의 (parts, 더할 관측값, 기하)."""
    if obs.stale is not None:
        return {"sight": obs.stale.sight, "age": obs.stale.age}, {}, {}
    if obs.corner is not None:
        return {"side": obs.corner.side}, {}, {}
    return {}, {}, {}



def _hard_stop(r) -> float:
    return ph.stopping_distance(r.speed, r.latency, r.decel_hard)


def _speed(cfg: SimgenConfig, ctx: Context, rng) -> float:
    body = cfg.body(ctx.body)
    return min(max(round_to(rng.uniform(*speed_range(cfg, ctx)), body["speed_resolution"]), body["speed"][0]), body["speed"][1])


def _speed_grid(cfg: SimgenConfig, ctx: Context, p: dict, d_lo: float, d_hi: float) -> list[float]:
    """최대 제동 멈춤 거리(반응 포함)가 [d_lo, d_hi] 안이 되는 속도들(글의 해상도 배수, 속도 범위 ∩ 빼 둔 칸)."""
    body = cfg.body(ctx.body)
    sp_lo, sp_hi = speed_range(cfg, ctx)
    v_lo = max(sp_lo, body["speed"][0], _speed_for_stop(max(d_lo, 0.0), p["latency"], p["decel_hard"]))
    v_hi = min(sp_hi, body["speed"][1], _speed_for_stop(max(d_hi, 0.0), p["latency"], p["decel_hard"]))
    step = body["speed_resolution"]
    return [float(v) for v in np.round(np.arange(math.ceil(v_lo / step - 1e-9), math.floor(v_hi / step + 1e-9) + 1) * step, 6) if v > 0.0]


def _build_stale(cfg: SimgenConfig, ctx: Context, margins: dict, rng) -> World:
    p = _robot_params(cfg, ctx.body, rng, loaded=False)
    spec = cfg.catalog["p6"]["stale"]
    lo, hi = ctx.range("stale_age", spec["age"])
    if hi < lo:
        raise Infeasible("stale age range empty")
    age = round_to(rng.uniform(lo, hi), 0.1)
    # 다시 보이는 거리 = 멈춤 거리 + 여유가 지도 범위 안이 되는 속도에서 고른다(여유 분포를 자르지 않게)
    s_lo, s_hi = spec["sight"][ctx.body]
    speeds = _speed_grid(cfg, ctx, p, s_lo + 0.03 - margins["sight"], s_hi - 0.03 - margins["sight"])
    if not speeds:
        raise Infeasible("no speed puts the sight distance in range")
    robot = _make_robot(p, speeds[int(rng.integers(len(speeds)))])
    sight = round_to(_hard_stop(robot) + margins["sight"], 0.05)
    if not s_lo <= sight <= s_hi:
        raise Infeasible("sight distance out of range")
    spot = sight + _u(rng, spec["spot_ahead"])
    lam = _u(rng, spec["leave_rate"][ctx.agent_kind])
    present = float(rng.random() < math.exp(-lam * age))
    values = {"present": present, "t_leave": float(rng.exponential(1.0 / lam)), "spot": spot, "lam": lam}
    parts = {"sight": sight, "age": age, "look_s": round_to(_u(rng, spec["look"]["time"]), 0.1)}
    return World(ctx.scene, ctx.domain, ctx.variant, robot, ctx.agent_kind, None, values, {}, parts=parts)


def _build_corner(cfg: SimgenConfig, ctx: Context, margins: dict, rng) -> World:
    p = _robot_params(cfg, ctx.body, rng, loaded=False)
    spec = cfg.catalog["p6"]["corner"]
    # 모퉁이는 0.3 m보다 앞에 있어야 한다 — 음의 여유는 그만큼 빠른 속도가 필요하다(속도 범위를 그 아래로 자른다)
    speeds = _speed_grid(cfg, ctx, p, 0.3 - margins["corner"], 1e9)
    if not speeds:
        raise Infeasible("no speed leaves the corner ahead")
    v0 = float(speeds[int(rng.integers(len(speeds)))])
    robot = _make_robot(p, v0)
    ahead = margins["corner"] + _hard_stop(robot)
    if ahead < 0.3 - 1e-9:
        raise Infeasible("corner margin puts the corner inside 0.3 m")
    lo, hi = ctx.range("corner_lateral", spec["lateral"][ctx.body])
    if hi < lo:
        raise Infeasible("corner lateral range empty")
    w = float(rng.uniform(lo + 0.1, hi) if hi - lo > 0.2 else (lo + hi) / 2.0)   # 보고 잡음(σ 0.05)이 칸 경계를 넘지 않게 0.1 띄운다
    side = "left" if rng.random() < 0.5 else "right"
    lam = _u(rng, cfg.domain(ctx.domain)["emerge_rate"])
    t_e = float(rng.exponential(1.0 / lam))
    agent = cfg.agent(ctx.agent_kind)
    values = {
        "lam": lam, "t_e": t_e, "emerges": float(t_e <= cfg.catalog["horizon_s"]), "u": _u(rng, agent["speed"]), "ra": _u(rng, agent["radius"]),
        "agent_decel": _u(rng, agent.get("decel", [1.0, 2.0])), "reacts": float(rng.random() < cfg.catalog["crossing_reaction"]["base"][agent["group"]]),
        "react_standoff": _u(rng, cfg.catalog["crossing_reaction"]["standoff"]), "ahead": ahead, "lateral": w,
        "kind": float(list(cfg.domain(ctx.domain)["agents"]["crossing"]).index(ctx.agent_kind)),   # 사후와 같은 번호(분야의 가로지름 섞기)
        "distracted": float(rng.random() < cfg.catalog["static_avoidance"]["distracted_prior"][agent["group"]]), "static_u": float(rng.random()),
    }  # fmt: skip
    parts = {"side": side, "look_s": round_to(_u(rng, spec["look"]["time"]), 0.1)}
    return World(ctx.scene, ctx.domain, ctx.variant, robot, ctx.agent_kind, None, values, {}, parts=parts)


def _conflict_entries(cfg: SimgenConfig, ctx: Context) -> list[dict]:
    card = cfg.domain(ctx.domain)
    return compatible(card.get("conflicts", []), setting_tags(card, ctx.setting), ctx.body, ctx.scene)


def _build_conflict(cfg: SimgenConfig, ctx: Context, margins: dict, rng) -> World:
    p = _robot_params(cfg, ctx.body, rng, loaded=False)
    spec = cfg.catalog["p6"]["conflict"]
    entries = _conflict_entries(cfg, ctx)
    entry = entries[int(rng.integers(len(entries)))]
    table = spec["cues"][entry["cue"]]
    for _ in range(60):
        present = rng.random() < spec["prior_present"]
        lidar = rng.random() < table["lidar"][0 if present else 1]
        camera = rng.random() < table["camera"][0 if present else 1]
        if lidar != camera:
            break
    else:
        raise Infeasible("sensors did not disagree")
    # 속도(글에 보이는 해상도의 값)를 정하고 숨은 늦은 감지 거리 d_late = 여유 + 최대 제동 멈춤 거리로 여유를 정확히 맞춘다(그 단서의 범위 안)
    late_lo, late_hi = table["late"]
    body = cfg.body(ctx.body)
    sp_lo, sp_hi = speed_range(cfg, ctx)
    v_lo = max(sp_lo, _speed_for_stop(max(late_lo - margins["late"], 0.0), p["latency"], p["decel_hard"]))
    v_hi = min(sp_hi, _speed_for_stop(max(late_hi - margins["late"], 0.0), p["latency"], p["decel_hard"]))
    speeds = [v for v in np.round(np.arange(math.ceil(v_lo / body["speed_resolution"] - 1e-9), math.floor(v_hi / body["speed_resolution"] + 1e-9) + 1)
                                  * body["speed_resolution"], 6) if v > 0.0]  # fmt: skip
    if not speeds:
        raise Infeasible("no speed realises the late margin")
    v0 = float(speeds[int(rng.integers(len(speeds)))])
    robot = _make_robot(p, v0)
    late = margins["late"] + _hard_stop(robot)
    if not late_lo - 1e-9 <= late <= late_hi + 1e-9:
        raise Infeasible("late distance out of the cue's range")
    distance = _u(rng, spec["distance"][ctx.body])
    if distance < late + 0.3:
        raise Infeasible("reported distance inside the late-detection range")
    values = {"present": float(present), "late": late, "spot": distance}
    parts = {"entry": entry, "lidar": bool(lidar), "camera": bool(camera), "bearing": ("ahead", "left", "right")[int(rng.integers(3))],
             "look_s": round_to(_u(rng, spec["look"]["time"]), 0.1), "detour_s": round_to(_u(rng, spec["detour_s"]), 1.0)}  # fmt: skip
    return World(ctx.scene, ctx.domain, ctx.variant, robot, None, None, values, {}, parts=parts)


def _build_emerging(cfg: SimgenConfig, ctx: Context, margins: dict, rng) -> World:
    """P1+P6: 막 가림 뒤에서 나온 상대(나이 ≤ 0.3 s — 빼 둔 칸 '오래된 추적'과 겹치지 않는다) + 따라 나올 수 있는 둘째."""
    p = _robot_params(cfg, ctx.body, rng, loaded=False)
    agent = cfg.agent(ctx.agent_kind)
    ra = _u(rng, agent["radius"])
    r_sum = p["radius"] + ra
    v0, gap = _solve_speed_for_gap(cfg, ctx, p, r_sum, margins["step_in"], p["decel_moderate"], rng)
    young = Context(**{**ctx.__dict__, "ranges": ctx.ranges + (("track_age", 0.0, 0.3),)})
    values, at_obs = _step_in_agent(cfg, young, ra, r_sum, rng)
    views, sides = _sides(cfg, ctx, p["can_sidestep"], rng)
    follow_p = cfg.catalog["p6"]["follow"].get(ctx.agent_kind, 0.0)
    values.update({
        "gap": gap, "side_left": sides[0], "side_right": sides[1], "follow": float(rng.random() < follow_p),
        "t2": _u(rng, cfg.catalog["p6"]["follow_delay"]), "u2": _u(rng, agent["speed"]), "ra2": _u(rng, agent["radius"]),
        **follower_reaction(cfg, ctx.agent_kind, rng, 1, scalar=True),
    })  # fmt: skip
    at_obs.update(sides=views)
    return World(ctx.scene, ctx.domain, ctx.variant, _make_robot(p, v0), ctx.agent_kind, None, values, at_obs, parts={})


def follower_reaction(cfg: SimgenConfig, kind: str, rng, k: int, scalar: bool = False) -> dict:
    """따라 나오는 둘째 상대의 반응(수정 1 I3 — 가로지름과 같은 규칙): 움직이는 로봇에 늦게라도 서나(갈래의 늦은 반응 확률), 감속·여유,
    주의가 흐트러졌나(갈래 사전), 멈춘 로봇 앞 실패를 정하는 균등 난수. 사후(p6._emerging)와 닻이 같은 사전을 쓴다."""
    agent = cfg.agent(kind)
    group = agent["group"]
    react = cfg.catalog["crossing_reaction"]["base"][group]
    distracted = cfg.catalog["static_avoidance"]["distracted_prior"][group]
    lo_d, hi_d = agent.get("decel", [1.0, 2.0])
    lo_s, hi_s = cfg.catalog["crossing_reaction"]["standoff"]
    out = {"reacts2": (rng.random(k) < react).astype(float), "decel2": lo_d + (hi_d - lo_d) * rng.random(k), "standoff2": lo_s + (hi_s - lo_s) * rng.random(k),
           "distracted2": (rng.random(k) < distracted).astype(float), "static_u2": rng.random(k)}  # fmt: skip
    return {key: float(v[0]) for key, v in out.items()} if scalar else out


def build(cfg: SimgenConfig, ctx: Context, margins: dict, rng) -> World:
    from robo_jev.simgen.scenes import _with_margins

    builder = {"p6_stale_track": _build_stale, "p6_blind_corner": _build_corner, "p6_sensor_conflict": _build_conflict,
               "p1p6_emerging_step_in": _build_emerging}[ctx.scene]  # fmt: skip
    return _with_margins(cfg, builder(cfg, ctx, margins, rng))


def prior_redraw(cfg: SimgenConfig, world: World, rng) -> World:
    """순방향 SBC용: 여유에 맞춰 푼 숨은 값을 정답의 사전에서 다시 뽑는다 — 센서 불일치의 늦은 감지 거리(단서의 범위 균등). 나머지(떠남·출현·
    상대 종류와 매개변수·따라옴·있음과 두 센서의 보고)는 닻이 이미 사전에서 뽑는다."""
    if world.scene != "p6_sensor_conflict":
        return world
    late = cfg.catalog["p6"]["conflict"]["cues"][world.parts["entry"]["cue"]]["late"]
    return world.with_values(late=float(rng.uniform(*late)))


def margins_of(cfg: SimgenConfig, world: World) -> dict[str, float]:
    r, v = world.robot, world.values
    if world.scene == "p6_stale_track":
        return {"sight": world.parts["sight"] - _hard_stop(r)}
    if world.scene == "p6_blind_corner":
        return {"corner": v["ahead"] - _hard_stop(r)}
    if world.scene == "p6_sensor_conflict":
        return {"late": v["late"] - _hard_stop(r)}
    return {"step_in": v["gap"] - ph.stopping_distance(r.speed, r.latency, r.decel_moderate) - (r.radius + v["ra"])}


def _age_for(cfg: SimgenConfig, kind: str, target: float) -> float | None:
    """'아직 그 자리에 있을 확률'이 target이 되는 나이(초) — 확률은 나이에 따라 줄어든다(이분법)."""
    from robo_jev.simgen.p6 import stay_probability

    lo, hi = 0.0, 600.0
    if not stay_probability(cfg, kind, hi) < target < stay_probability(cfg, kind, 1e-6):
        return None
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if stay_probability(cfg, kind, mid) > target else (lo, mid)
    return 0.5 * (lo + hi)


def flip(cfg: SimgenConfig, world: World, name: str, rng):
    """보이는 거리(sight), 모퉁이까지 거리(corner), 로봇 속도(late — 숨은 d_late는 그대로), 간격(step_in), 본 지 지난 시간(age — 오래된
    추적의 핵심 '아직 있나'를 정한다, 수정 1 I4: 그 확률이 0.5의 반대쪽 같은 거리(0.15~0.85로 자름)가 되는 나이, 0.1 s 단위) 하나만 경계 너머로."""
    from dataclasses import replace

    from robo_jev.simgen.scenes import _with_margins

    if name == "age":
        from robo_jev.simgen.p6 import stay_probability

        p = stay_probability(cfg, world.agent_kind, world.parts["age"])
        far = 1.0 - p if abs(p - 0.5) >= 0.1 else (0.4 if p > 0.5 else 0.6)
        for target in (min(max(far, 0.15), 0.85), 0.35 if p > 0.5 else 0.65, 0.25 if p > 0.5 else 0.75):
            age = _age_for(cfg, world.agent_kind, target)
            if age is None:
                continue
            age = round_to(age, 0.1)
            if age > 0.05 and (stay_probability(cfg, world.agent_kind, age) - 0.5) * (p - 0.5) < 0:
                yield _with_margins(cfg, replace(world, parts={**world.parts, "age": age})), name
        return
    m = world.margins[name]
    sigma = cfg.catalog["margins"][name]["sigma"][world.robot.body]
    sign = 1.0 if m >= 0 else -1.0
    for target in (-sign * max(abs(m), sigma), -sign * 0.5 * sigma, -sign * max(abs(m), 2.0 * sigma)):
        delta = target - m
        if name == "sight":
            parts = {**world.parts, "sight": round_to(world.parts["sight"] + delta, 0.05)}
            moved = replace(world, parts=parts)
            ok = parts["sight"] > 0.1 and world.values["spot"] > parts["sight"] + 0.2
        elif name == "corner":
            ahead = world.values["ahead"] + delta
            moved = world.with_values(ahead=ahead)
            ok = ahead > 0.3
        elif name == "late":
            stop = world.values["late"] - target
            if stop <= 0.02:
                continue
            v = round_to(_speed_for_stop(stop, world.robot.latency, world.robot.decel_hard), 0.05)
            moved = replace(world, robot=replace(world.robot, speed=v, cruise=v))
            ok = v > 0.1
        else:
            moved = world.with_values(gap=world.values["gap"] + delta)
            ok = moved.values["gap"] - (world.robot.radius + world.values["ra"]) >= 0.5 * cfg.body(world.robot.body)["gap_min"]
        if ok:
            yield _with_margins(cfg, moved), name


def observe_parts(cfg: SimgenConfig, world: World, ctx: Context, u, z, skin: dict) -> tuple[dict, dict]:
    from robo_jev.simgen.scenes import _distance_sigma

    v, r = world.values, world.robot
    out: dict = {}
    env: dict = {"look_s": world.parts.get("look_s")}
    if world.scene == "p6_stale_track":
        res = (0.01, 0.05, 0.1)[min(int(u[0] * 3), 2)]
        ahead = _reading(v["spot"], 0.05, res, z[0], floor=res)
        out["stale"] = StaleObs(
            kind=world.agent_kind, descriptor=Descriptor(skin["descriptor"].np, skin["descriptor"].ref), ref=skin["descriptor"].ref,
            pronoun=skin["pronoun"], spot=skin["spot"], age=world.parts["age"], ahead=ahead, sight=world.parts["sight"], occluder=skin["occluder"],
        )  # fmt: skip
    elif world.scene == "p6_blind_corner":
        res = (0.01, 0.05, 0.1)[min(int(u[0] * 3), 2)]
        out["corner"] = CornerObs(
            phrase=skin["corner"], side=world.parts["side"], ahead=_reading(v["ahead"], _distance_sigma(cfg, ctx.sensor_suite, v["ahead"], u[1]), res, z[0], floor=res),
            lateral=_reading(v["lateral"], 0.05, 0.05, z[1], floor=0.05), busy="normal",
        )  # fmt: skip
    elif world.scene == "p6_sensor_conflict":
        parts = world.parts
        sigma = 0.03 if parts["lidar"] else 0.12
        res = (0.01, 0.05, 0.1)[min(int(u[0] * 3), 2)]
        out["conflict"] = ConflictObs(cue=parts["entry"]["cue"], phrase=skin["conflict_phrase"], lidar=parts["lidar"], camera=parts["camera"],
                                      distance=_reading(v["spot"], sigma, res, z[0], floor=res), bearing=parts["bearing"])  # fmt: skip
        env.update(detour=skin["detour_generic"], detour_s=parts["detour_s"])
    elif world.scene == "p1p6_emerging_step_in":
        env["occluder"] = skin["occluder"]
    return out, env
