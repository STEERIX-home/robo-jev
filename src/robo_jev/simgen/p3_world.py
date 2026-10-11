"""P3 받침 위 물건의 세계 쪽: 결정 여유에 맞춘 닻 세계, 여유 재기, 사실 하나 바꾸기, 감각 모형 (Task v0b 지시서 §1·§2).

장면: ``p3_carrying_tall``(키 큰 물건을 싣고 감속 brake·회전 turn·출발 start), ``p3_ramp``(경사로를 내려가며 멈춤 down·올라가며 출발 up),
``p3_place_stack``(쌓기 — 무게중심이 받침면 안에 있나), 조합 ``p3p5_carrying_mixed``(키 큰 물건과 열린 용기를 함께 실음 — 감속·회전),
``p1p3_step_in_carrying_tall``(사람이 끼어들 때 키 큰 물건을 실음 — 시험 세트 T4 전용).

**결정 여유**(catalog margins): tip = b_eff/h − r_ref(기준 움직임의 쏠림 비 — 단단한 제동, 지금 속도의 회전, 보통 출발, 보통 경사로),
slide = μ − r_ref, stop(감속 갈래) = 장애물 면 − 단단한 제동의 멈춤 거리, stack = L_low/2 − |d|(명목 — 무게가 가운데이고 놓기 오차가 없을 때
받침 가장자리까지; 숨은 무게 치우침 c와 놓기 오차 ε가 그 둘레의 불확실성), spill·step_in은 P5·P1과 같다.
숨은 값(무게중심 높이 비율 com, 앞뒤 치우침 off, 마찰 μ, 쌓을 물건의 무게 치우침 c)을 여유에 맞춰 풀되 속성 목록의 범위 안에서만(안 되면
:class:`~robo_jev.simgen.scenes.Infeasible` — 다른 갈래로 다시 뽑는다). 닻 세계는 관측만 만든다(참 세계는 정답의 사후에서 — 생성기).

이 모듈은 세계 쪽이다(scenes와 같은 층).
"""

from __future__ import annotations

import math

import numpy as np

from robo_jev.simgen import physics as ph
from robo_jev.simgen.catalog import SimgenConfig
from robo_jev.simgen.observation import LoadObs, StackObs
from robo_jev.simgen.scenes import (
    Context,
    Infeasible,
    _carried,
    _container_hidden,
    _make_robot,
    _reading,
    _robot_params,
    _sides,
    _solve_speed_for_gap,
    _speed_for_stop,
    _spill_coef,
    _step_in_agent,
    _u,
    fill_range,
    speed_range,
)
from robo_jev.simgen.skins import compatible, setting_tags
from robo_jev.simgen.spec import round_to
from robo_jev.simgen.world import World

__all__ = ["b_eff", "build", "flip", "load_entries", "margins_of", "observe_parts", "prior_redraw", "reference_ratio", "truth_parts"]


def truth_parts(cfg: SimgenConfig, obs) -> tuple[dict, dict, dict]:
    """참 세계(정답의 사후에서 뽑은 세계)의 (parts, 더할 관측값, 기하) — 여유를 다시 재려고(관측에서 그대로)."""
    parts: dict = {"maneuver": obs.env.maneuver}
    extra: dict = {}
    geometry: dict = {}
    if obs.load is not None:
        parts["load"] = {"kind": obs.load.kind}
    if obs.env.turn_radius is not None:
        geometry = {"radius": obs.env.turn_radius, "angle_deg": obs.env.turn_angle_deg, "approach": obs.env.turn_approach.value}
    if obs.scene == "p3_ramp":
        parts["maneuver"] = obs.env.ramp_direction
        geometry = {"angle_deg": obs.env.ramp_angle_deg, "length": obs.env.ramp_length}
    if obs.scene == "p1p3_step_in_carrying_tall":
        parts["maneuver"] = "brake"
    if obs.stack is not None:
        parts["stack_item"] = {"kind": obs.stack.kind}
        extra = {"length_low": obs.stack.length_low, "length_top": obs.stack.length_top, "offset": obs.stack.offset}
    return parts, extra, geometry

G = ph.G
_CARRY = ("p3_carrying_tall", "p3p5_carrying_mixed", "p1p3_step_in_carrying_tall")


def load_entries(cfg: SimgenConfig, ctx: Context) -> list[dict]:
    card = cfg.domain(ctx.domain)
    usable = compatible(card.get("loads", []), setting_tags(card, ctx.setting), ctx.body, ctx.scene)
    return [e for e in usable if ctx.body in cfg.catalog["loads"][e["kind"]]["bodies"]]


def b_eff(base, off, direction: str):
    """쏠리는 쪽 가장자리까지 무게중심의 수평 거리: 앞(감속·내리막 멈춤) base/2·(1 − off), 뒤(출발·오르막) base/2·(1 + off), 옆 base/2."""
    base = np.asarray(base, dtype=float)
    if direction == "front":
        return base / 2.0 * (1.0 - np.asarray(off, dtype=float))
    if direction == "back":
        return base / 2.0 * (1.0 + np.asarray(off, dtype=float))
    return base / 2.0


def _direction(scene: str, maneuver: str | None) -> str:
    if maneuver in ("start", "up"):
        return "back"
    if maneuver == "turn":
        return "side"
    return "front"


def reference_ratio(robot, scene: str, maneuver: str | None, geometry: dict) -> float:
    """결정 여유의 기준 움직임의 쏠림 비 r_ref."""
    if maneuver == "turn":
        return robot.speed**2 / geometry["radius"] / G
    if maneuver == "start":
        return robot.accel / G
    if scene == "p3_ramp":
        a = robot.decel_moderate if maneuver == "down" else robot.accel
        return float(ph.incline_ratio(a, math.radians(geometry["angle_deg"])))
    return robot.decel_moderate / G


# ----------------------------------------------------------------------------------------------------------------
# 닻 세계
# ----------------------------------------------------------------------------------------------------------------


_ASPECT_PAD = 0.15     # 높이/바닥 폭 칸 경계에서 띄우는 여유(보고 잡음 σ 0.01/0.005가 경계를 넘지 않게)


def _geometry_samples(cfg: SimgenConfig, ctx: Context, kind: str, rng, n: int = 256) -> dict[str, np.ndarray]:
    """그 물건 종류의 기하 n개(속성 목록 범위에서 균등) — 높이/바닥 폭은 빼 둔 칸 범위 안(학습 ≤ 4.0, T3 > 4.0; 여유 0.15)."""
    spec = cfg.catalog["loads"][kind]
    edges = cfg.catalog["p3"]["edge"][ctx.body]
    a_lo, a_hi = ctx.range("load_aspect", (0.0, 1e9))
    base = rng.uniform(*spec["base"], n)
    h_lo = np.maximum(spec["height"][0], (a_lo + _ASPECT_PAD) * base if a_lo > 0 else np.zeros(n))
    h_hi = np.minimum(spec["height"][1], (a_hi - _ASPECT_PAD) * base)
    ok = h_hi >= h_lo
    out = {
        "base": base, "height": h_lo + rng.random(n) * (h_hi - h_lo), "h_lo": h_lo, "h_hi": h_hi,
        "off": rng.uniform(-spec["offset"], spec["offset"], n), "com": rng.uniform(*spec["com_ratio"], n),
        "edge_f": rng.uniform(*edges, n), "edge_b": rng.uniform(*edges, n), "edge_s": rng.uniform(*edges, n),
    }  # fmt: skip
    return {k: v[ok] for k, v in out.items()}


def _fit_ratio(spec: dict, geo: dict[str, np.ndarray], b: np.ndarray, q: float, rng) -> dict | None:
    """b_eff/(com·H) = q를 정확히 맞추는 기하 하나: 표본마다 com을 고쳐(범위 안이면) 또는 H를 고쳐(높이 범위 안이면) 맞춘다."""
    if q <= 0.0 or len(b) == 0:
        return None
    com_need = b / (q * geo["height"])
    h_need = b / (q * geo["com"])
    fit_c = (com_need >= spec["com_ratio"][0]) & (com_need <= spec["com_ratio"][1])
    fit_h = (h_need >= geo["h_lo"]) & (h_need <= geo["h_hi"])
    choices = [(int(i), "com") for i in np.flatnonzero(fit_c)] + [(int(i), "height") for i in np.flatnonzero(fit_h)]
    if not choices:
        return None
    i, which = choices[int(rng.integers(len(choices)))]
    out = {k: float(v[i]) for k, v in geo.items() if k not in ("h_lo", "h_hi")}
    if which == "com":
        out["com"] = float(com_need[i])
    else:
        out["height"] = float(h_need[i])
    return out


def _nearest_band(lo: float, hi: float, target: float, width: float, rng) -> float:
    """닿을 수 있는 구간 [lo, hi] 밖의 목표를 가장 가까운 끝에서 안쪽으로 U(0, width)만큼 들인 값으로(구간 안이면 그대로)."""
    if lo <= target <= hi:
        return target
    step = float(rng.uniform(0.0, width))
    return min(lo + step, hi) if target < lo else max(hi - step, lo)


def _grid(lo: float, hi: float, step: float) -> np.ndarray:
    """[lo, hi] 안의 step 배수(글에 그대로 보이는 값들)."""
    k0, k1 = math.ceil(lo / step - 1e-9), math.floor(hi / step + 1e-9)
    return np.round(np.arange(k0, k1 + 1) * step, 6)


def _solve_load(cfg: SimgenConfig, ctx: Context, r_values, m_tip: float, m_slide: float | None, rng, entries=None,
                strict: bool = False) -> tuple[dict, dict, dict, int, tuple[str, ...]]:  # fmt: skip
    """(물건 항목, 받침 표면 항목, 숨은 값, 고른 r의 번호, 정확히 맞추지 못한 여유) — tip 여유 b_eff/(com·H) − r = m_tip을 물건 기하와 r로,
    slide 여유 μ − r = m_slide를 마찰로(그 μ를 범위에 담는 표면을 고른다) 맞춘다.

    r(기준 움직임의 쏠림 비)은 로봇 쪽 결정 변수가 정한다(단단한 제동의 감속/g, 회전 속도의 v²/(R·g), 출발 가속/g, 경사로의 쏠림 비).
    ``r_values``는 그 변수가 글에 보이는 값(해상도의 배수)일 때의 r들이다 — 그중 하나를 고르고 기하를 그 r에 정확히 맞춘다(그래서 닻의
    여유가 목표와 정확히 같다). 물건 종류마다 b_eff/h가 닿는 구간이 정해져 있어(예: 사람형 로봇의 제동에서 물병은 늘 넘어지지 않는 쪽
    멀리) 뽑은 여유에 닿는 조합이 그 상황에 없으면 가장 가까이 닿는 물건의 구간 끝에서 안쪽으로 σ 폭 안의 값을 쓰고 그 여유를 '정확하지
    않음'으로 돌려준다 — 그런 순간은 본래 여유가 큰 순간이고, 레코드의 meta.margins는 실제 세계에서 다시 잰 여유다(정답은 관측의 사후라
    닻의 여유 분포와 무관하다). ``strict``면 그 대신 Infeasible(호출한 쪽이 다른 기하로 다시). slide가 None이면 μ는 고른 표면의 사전에서."""
    entries = entries if entries is not None else load_entries(cfg, ctx)
    if not entries:
        raise Infeasible("no load for this context")
    r_values = np.asarray(r_values, dtype=float)
    if len(r_values) == 0:
        raise Infeasible("no reference ratio to choose from")
    direction = _direction(ctx.scene, ctx.maneuver)
    order_r = rng.permutation(len(r_values))[:8]
    pool = []
    fit, i_r, inexact = None, None, []
    for idx in rng.permutation(len(entries)):
        entry = entries[int(idx)]
        spec = cfg.catalog["loads"][entry["kind"]]
        geo = _geometry_samples(cfg, ctx, entry["kind"], rng)
        if len(geo["base"]) == 0:
            continue
        b = b_eff(geo["base"], geo["off"], direction)
        for j in order_r:
            fit = _fit_ratio(spec, geo, b, float(r_values[j]) + m_tip, rng)
            if fit is not None:
                i_r = int(j)
                break
        if fit is not None:
            break
        ratios = b / (geo["com"] * geo["height"])
        pool.append((entry, spec, geo, b, float(ratios.min()), float(ratios.max())))
    if fit is None:
        if not pool:
            raise Infeasible("no load geometry fits the aspect range")
        if strict:
            raise Infeasible("no load reaches the tip margin with these reference ratios")
        # 닿을 수 있는 r 구간 [q_min − m, q_max − m]에 가장 가까운 (물건, r) → 쏠림 비는 그 물건 구간의 안쪽 띠(σ 폭)
        sigma = cfg.catalog["margins"]["tip"]["sigma"][ctx.body]

        def gap(t, r):
            return max(t[4] - m_tip - r, r - (t[5] - m_tip), 0.0)

        best = min(((t, j) for t in pool for j in range(len(r_values))), key=lambda tj: gap(tj[0], float(r_values[tj[1]])))
        (entry, spec, geo, b, q_min, q_max), i_r = best
        q = _nearest_band(q_min, q_max, float(r_values[i_r]) + m_tip, sigma, rng)
        fit = _fit_ratio(spec, geo, b, q, rng)
        if fit is None:
            ratios = b / (geo["com"] * geo["height"])
            i = int(np.argmin(np.abs(ratios - q)))
            fit = {k: float(v[i]) for k, v in geo.items() if k not in ("h_lo", "h_hi")}
        inexact.append("tip")
    r = float(r_values[i_r])
    decks = cfg.skins["decks"][ctx.body]
    order = [decks[int(i)] for i in rng.permutation(len(decks))]
    if m_slide is None:
        deck = order[0]
        mu = _u(rng, cfg.catalog["decks"][deck["kind"]]["mu"])
    else:
        mu = r + m_slide
        fits = [d for d in order if cfg.catalog["decks"][d["kind"]]["mu"][0] <= mu <= cfg.catalog["decks"][d["kind"]]["mu"][1]]
        if fits:
            deck = fits[0]
        else:
            # 그 마찰을 담는 표면이 이 몸에 없으면 가장 가까운 표면의 범위 끝에서 안쪽으로(σ 폭)
            sigma = cfg.catalog["margins"]["slide"]["sigma"][ctx.body]
            deck = min(order, key=lambda d: max(cfg.catalog["decks"][d["kind"]]["mu"][0] - mu, mu - cfg.catalog["decks"][d["kind"]]["mu"][1]))
            mu = _nearest_band(*cfg.catalog["decks"][deck["kind"]]["mu"], mu, sigma, rng)
            inexact.append("slide")
    return entry, deck, {**fit, "mu": float(mu)}, i_r, tuple(inexact)


def _brake_layout(cfg: SimgenConfig, ctx: Context, p: dict, m_stop: float, rng) -> tuple[float, float]:
    """감속 갈래: 단단한 제동의 멈춤 여유를 맞추는 (속도, 장애물 중심 거리) — p5_braking과 같다."""
    body = cfg.body(ctx.body)
    t = p["latency"]
    s_lo = max(0.0, body["gap_min"] - m_stop)
    s_hi = body["range_max"] - p["radius"] - m_stop
    sp_lo, sp_hi = speed_range(cfg, ctx)
    v_lo = max(sp_lo, _speed_for_stop(s_lo, t, p["decel_moderate"]))
    v_hi = min(sp_hi, _speed_for_stop(s_hi, t, p["decel_moderate"]))
    if v_lo > v_hi:
        raise Infeasible("no speed realises the stop margin")
    v0 = min(max(round_to(rng.uniform(v_lo, v_hi), body["speed_resolution"]), body["speed"][0]), body["speed"][1])
    obstacle = m_stop + p["radius"] + ph.stopping_distance(v0, t, p["decel_moderate"])
    if obstacle - p["radius"] < 0.5 * body["gap_min"]:
        raise Infeasible("rounded speed broke the stop distance")
    return v0, obstacle


def _turn_geometry(cfg: SimgenConfig, ctx: Context, rng) -> dict:
    body = cfg.body(ctx.body)
    radius = round_to(_u(rng, body["turn_radius"]), 0.05)
    angle = float((45.0, 90.0, 135.0)[int(rng.choice(3, p=[0.2, 0.6, 0.2]))])
    approach = round_to(_u(rng, body["turn_approach"]), 0.05)
    side = "left" if rng.random() < 0.5 else "right"
    return {"radius": radius, "angle_deg": angle, "approach": approach, "direction": side}


def _start_accels(cfg: SimgenConfig, p: dict) -> dict:
    levels = cfg.catalog["p3"]["start_levels"]
    return {name: round_to(min(p["accel"] * f, p["decel_hard"]), 0.05) for name, f in levels.items()}


def _times(cfg: SimgenConfig, ctx: Context, rng) -> dict:
    return {"secure_s": round_to(_u(rng, cfg.catalog["p3"]["secure_s"][ctx.body]), 0.5)}


def _build_carrying(cfg: SimgenConfig, ctx: Context, margins: dict, rng) -> World:
    """키 큰 물건을 싣고 감속·회전·출발. 넘어짐 여유는 물건 기하와 로봇 쪽 결정 변수(감속·회전 속도·출발 가속 — 글에 보이는 해상도의
    값)를 함께 풀어 맞춘다; P3+P5는 그 가속에서 용기의 쏟아짐 여유를 채움으로 맞춘다."""
    p = _robot_params(cfg, ctx.body, rng, loaded=True)
    body = cfg.body(ctx.body)
    geometry: dict = {}
    parts: dict = {"maneuver": ctx.maneuver}
    values: dict = {}
    entry_c = None
    if ctx.maneuver == "brake":
        lo, hi = p["_moderate_range"]
        decels = _grid(max(lo, p["decel_gentle"] + 0.1), min(hi, p["decel_hard"] - 0.1), 0.1)
        entry, deck, hidden, i_r, inexact = _solve_load(cfg, ctx, decels / G, margins["tip"], margins.get("slide"), rng)
        p["decel_moderate"] = float(decels[i_r])
        if ctx.scene == "p3p5_carrying_mixed":
            entry_c, fill = _spill_fill(cfg, ctx, margins["spill"], p["decel_moderate"], rng)
            hidden_c, at_c = _container_hidden(cfg, ctx, entry_c, rng)
            values.update(hidden_c, fill=fill)
            parts["at_observation"] = at_c
        v0, obstacle = _brake_layout(cfg, ctx, p, margins["stop"], rng)
        values.update(hidden, obstacle=obstacle)
        robot = _make_robot(p, v0)
    elif ctx.maneuver == "turn":
        # 회전 반경이 닿을 수 있는 옆 쏠림 비를 정한다(큰 반경에서는 키 큰 물건도 경계 근처에 오지 않는다) — 반경을 몇 번 다시 뽑아 정확히
        # 맞는 조합을 찾고, 끝내 없으면 마지막 반경에서 가장 가까운 띠로.
        speeds = _grid(*speed_range(cfg, ctx), body["speed_resolution"])
        speeds = speeds[(speeds >= body["speed"][0]) & (speeds <= body["speed"][1])]
        solved = None
        for _ in range(6):
            geometry = _turn_geometry(cfg, ctx, rng)
            try:
                solved = _solve_load(cfg, ctx, speeds**2 / (G * geometry["radius"]), margins["tip"], margins.get("slide"), rng, strict=True)
                break
            except Infeasible:
                continue
        if solved is None:
            solved = _solve_load(cfg, ctx, speeds**2 / (G * geometry["radius"]), margins["tip"], margins.get("slide"), rng)
        entry, deck, hidden, i_r, inexact = solved
        v0 = float(speeds[i_r])
        if ctx.scene == "p3p5_carrying_mixed":
            entry_c, fill = _spill_fill(cfg, ctx, margins["spill"], v0 * v0 / geometry["radius"], rng)
            hidden_c, at_c = _container_hidden(cfg, ctx, entry_c, rng)
            values.update(hidden_c, fill=fill)
            parts["at_observation"] = at_c
        values.update(hidden)
        robot = _make_robot(p, v0)
    else:  # start
        cruise = round_to(rng.uniform(*speed_range(cfg, ctx)), body["speed_resolution"])
        accels = _grid(*cfg.catalog["p3"]["start_accel"][ctx.body], 0.05)
        entry, deck, hidden, i_r, inexact = _solve_load(cfg, ctx, accels / G, margins["tip"], margins["slide"], rng)
        p["accel"] = float(accels[i_r])
        values.update(hidden)
        robot = _make_robot(p, 0.0, cruise)
        parts["start_accels"] = _start_accels(cfg, p)
    parts.update(load=entry, deck=deck, inexact=inexact, **_times(cfg, ctx, rng))
    if entry_c is not None:
        parts["container_entry"] = entry_c
    at_obs = dict(parts.pop("at_observation", {}))
    world = World(ctx.scene, ctx.domain, ctx.variant, robot, None, entry_c, values, at_obs, geometry=geometry, parts=parts)
    return world


def _spill_fill(cfg: SimgenConfig, ctx: Context, m_spill: float, accel: float, rng) -> tuple[dict, float]:
    """P3+P5: 정해진 가속(감속 또는 옆 가속)에서 용기의 쏟아짐 여유를 맞추는 (용기 항목, 채움) — 채움 = 1 − m − c·a가 채움 범위 안인 용기."""
    entries = _carried(cfg, ctx, rng)
    for idx in rng.permutation(len(entries)):
        entry = entries[int(idx)]
        f_lo, f_hi = fill_range(ctx, cfg.container(entry["kind"])["fill"])
        fill = 1.0 - m_spill - _spill_coef(cfg, entry["kind"]) * accel
        if f_lo <= fill <= f_hi:
            return entry, float(fill)
    raise Infeasible("no container realises the spill margin")


def _build_ramp(cfg: SimgenConfig, ctx: Context, margins: dict, rng) -> World:
    p = _robot_params(cfg, ctx.body, rng, loaded=True)
    body = cfg.body(ctx.body)
    spec = cfg.catalog["p3"]
    lo, hi = ctx.range("ramp_angle", spec["ramp_angle_deg"][ctx.body])
    if hi < lo:
        raise Infeasible("ramp angle range empty")
    angle = round_to(rng.uniform(lo + 0.25, hi - 0.25) if hi - lo > 0.5 else (lo + hi) / 2.0, 0.5)
    length = round_to(_u(rng, spec["ramp_length"]), 0.5)
    speed = round_to(rng.uniform(*speed_range(cfg, ctx)), body["speed_resolution"])
    robot = _make_robot(p, speed) if ctx.maneuver == "down" else _make_robot(p, 0.0, speed)
    geometry = {"angle_deg": angle, "length": length}
    ratio_ref = reference_ratio(robot, ctx.scene, ctx.maneuver, geometry)
    entry, deck, hidden, _, inexact = _solve_load(cfg, ctx, [ratio_ref], margins["tip"], margins["slide"], rng)
    parts = {"maneuver": ctx.maneuver, "load": entry, "deck": deck, "detour_s": round_to(_u(rng, spec["detour_s"]), 1.0), "inexact": inexact,
             **_times(cfg, ctx, rng)}  # fmt: skip
    return World(ctx.scene, ctx.domain, ctx.variant, robot, None, None, dict(hidden), {}, geometry=geometry, parts=parts)


def _build_step_in_tall(cfg: SimgenConfig, ctx: Context, margins: dict, rng) -> World:
    """P1+P3(T4 전용): 끼어듦 + 키 큰 물건. step_in 여유는 P1처럼, tip 여유는 단단한 제동의 쏠림 비에서."""
    p = _robot_params(cfg, ctx.body, rng, loaded=True)
    agent = cfg.agent(ctx.agent_kind)
    ra = _u(rng, agent["radius"])
    r_sum = p["radius"] + ra
    lo, hi = p["_moderate_range"]
    decels = _grid(max(lo, p["decel_gentle"] + 0.1), min(hi, p["decel_hard"] - 0.1), 0.1)
    entry, deck, hidden, i_r, inexact = _solve_load(cfg, ctx, decels / G, margins["tip"], None, rng)
    p["decel_moderate"] = float(decels[i_r])
    v0, gap = _solve_speed_for_gap(cfg, ctx, p, r_sum, margins["step_in"], p["decel_moderate"], rng)
    values, at_obs = _step_in_agent(cfg, ctx, ra, r_sum, rng)
    views, sides = _sides(cfg, ctx, p["can_sidestep"], rng)
    values.update({"gap": gap, "side_left": sides[0], "side_right": sides[1]}, **hidden)
    at_obs.update(sides=views)
    parts = {"maneuver": "brake", "load": entry, "deck": deck, "inexact": inexact, **_times(cfg, ctx, rng)}
    return World(ctx.scene, ctx.domain, ctx.variant, _make_robot(p, v0), ctx.agent_kind, None, values, at_obs, parts=parts)


def _stack_entries(cfg: SimgenConfig, ctx: Context) -> list[dict]:
    card = cfg.domain(ctx.domain)
    usable = compatible(card.get("stack_items", []), setting_tags(card, ctx.setting), ctx.body, ctx.scene)
    return [e for e in usable if ctx.body in cfg.catalog["stack_items"][e["kind"]]["bodies"]]


def _build_stack(cfg: SimgenConfig, ctx: Context, margins: dict, rng) -> World:
    p = _robot_params(cfg, ctx.body, rng, loaded=False)
    entries = _stack_entries(cfg, ctx)
    entry = entries[int(rng.integers(len(entries)))]
    spec = cfg.catalog["stack_items"][entry["kind"]]
    st = cfg.catalog["stack"]
    l_top = round_to(_u(rng, spec["length"]), 0.01)
    l_low = round_to(_u(rng, spec["length"]), 0.01)
    c_max = spec["c_max"] * l_top / 2.0
    m = margins["stack"]
    # 계획 위치 |d| = L_low/2 − 여유(글에 보이는 0.005 m 배수로 — 여유는 ±0.0025 안), 숨은 무게 치우침 c는 사전(±c_max 균등)에서
    want = round_to(l_low / 2.0 - m, 0.005)
    if want < 0.01 or want > l_low / 2.0 + cfg.catalog["margins"]["stack"]["half_range"][ctx.body]:
        raise Infeasible("planned offset out of range")
    d = float(want if rng.random() < 0.5 else -want)
    c = float(rng.uniform(-c_max, c_max))
    cue = None
    if rng.random() < st["cue_share"]:
        true_side = "left" if c > 0 else "right"
        cue = true_side if rng.random() < st["cue_reliability"] else ("right" if true_side == "left" else "left")
    values = {"c": c, "eps": 0.0, "length_top": l_top, "length_low": l_low, "offset": d}
    parts = {
        "stack_item": entry, "sigma_place": round_to(_u(rng, st["sigma_place"][ctx.body]), 0.001), "cue": cue,
        "recentre_s": round_to(_u(rng, st["recentre_s"]), 0.5), "elsewhere_s": round_to(_u(rng, st["elsewhere_s"]), 1.0),
        "check_s": round_to(_u(rng, st["check_s"]), 0.5), "c_max": c_max,
    }  # fmt: skip
    return World(ctx.scene, ctx.domain, ctx.variant, _make_robot(p, 0.0, 0.0), None, None, values, {}, parts=parts)


def build(cfg: SimgenConfig, ctx: Context, margins: dict, rng) -> World:
    from robo_jev.simgen.scenes import _with_margins

    if ctx.scene in ("p3_carrying_tall", "p3p5_carrying_mixed"):
        world = _build_carrying(cfg, ctx, margins, rng)
    elif ctx.scene == "p3_ramp":
        world = _build_ramp(cfg, ctx, margins, rng)
    elif ctx.scene == "p3_place_stack":
        world = _build_stack(cfg, ctx, margins, rng)
    elif ctx.scene == "p1p3_step_in_carrying_tall":
        world = _build_step_in_tall(cfg, ctx, margins, rng)
    else:
        raise ValueError(ctx.scene)
    return _with_margins(cfg, world)


def prior_redraw(cfg: SimgenConfig, world: World, rng) -> World:
    """순방향 SBC용: 여유에 맞춰 푼 숨은 값을 정답의 사전(p3.sample_posterior의 사전)에서 다시 뽑는다 — 무게중심 높이 비율·앞뒤 치우침
    (물건 종류의 범위 균등), 마찰(받침 표면의 범위 균등), 쌓기의 무게 치우침 c(±c_max 균등)와 그 c에서 다시 뽑은 쏠림 단서, 놓기 오차
    ε ~ N(0, σ_place). 관측은 이 세계에서 새로 만든다."""
    from dataclasses import replace

    v = dict(world.values)
    parts = dict(world.parts)
    if "load" in parts:
        spec = cfg.catalog["loads"][parts["load"]["kind"]]
        v.update(com=float(rng.uniform(*spec["com_ratio"])), off=float(rng.uniform(-spec["offset"], spec["offset"])),
                 mu=float(rng.uniform(*cfg.catalog["decks"][parts["deck"]["kind"]]["mu"])))  # fmt: skip
    if "stack_item" in parts:
        st = cfg.catalog["stack"]
        c = float(rng.uniform(-parts["c_max"], parts["c_max"]))
        cue = None
        if rng.random() < st["cue_share"]:
            true_side = "left" if c > 0 else "right"
            cue = true_side if rng.random() < st["cue_reliability"] else ("right" if true_side == "left" else "left")
        v.update(c=c, eps=float(parts["sigma_place"] * rng.standard_normal()))
        parts["cue"] = cue
    return replace(world, values=v, parts=parts)


# ----------------------------------------------------------------------------------------------------------------
# 여유 재기·사실 하나 바꾸기
# ----------------------------------------------------------------------------------------------------------------


def margins_of(cfg: SimgenConfig, world: World) -> dict[str, float]:
    from robo_jev.simgen.catalog import scene_margins
    from robo_jev.simgen.scenes import _spill_coef as spill_coef

    r, v = world.robot, world.values
    maneuver = world.parts.get("maneuver")
    out: dict[str, float] = {}
    for name in scene_margins(world.scene, maneuver if world.scene != "p1p3_step_in_carrying_tall" else None):
        if name == "tip":
            ratio = reference_ratio(r, world.scene, maneuver, world.geometry)
            h = v["com"] * v["height"]
            out[name] = float(b_eff(v["base"], v["off"], _direction(world.scene, maneuver))) / h - ratio
        elif name == "slide":
            out[name] = v["mu"] - reference_ratio(r, world.scene, maneuver, world.geometry)
        elif name == "stop":
            out[name] = v["obstacle"] - r.radius - ph.stopping_distance(r.speed, r.latency, r.decel_moderate)
        elif name == "spill":
            c = spill_coef(cfg, world.container["kind"])
            accel = r.speed**2 / world.geometry["radius"] if maneuver == "turn" else r.decel_moderate
            out[name] = 1.0 - v["fill"] - c * accel
        elif name == "step_in":
            out[name] = v["gap"] - ph.stopping_distance(r.speed, r.latency, r.decel_moderate) - (r.radius + v["ra"])
        elif name == "stack":
            out[name] = v["length_low"] / 2.0 - abs(v["offset"])
    return out


def flip(cfg: SimgenConfig, world: World, name: str, rng):
    """결정 변수 하나만 경계 너머로: tip → 물건 높이(관측), stop → 장애물 거리, stack → 계획 위치, slide → (관측 변수가 없어) tip으로,
    step_in·spill → P1·P5와 같다."""
    from robo_jev.simgen.scenes import _with_margins

    if name == "slide":
        name = "tip"
    m = world.margins[name]
    sigma = cfg.catalog["margins"][name]["sigma"][world.robot.body]
    sign = 1.0 if m >= 0 else -1.0
    r, v = world.robot, world.values
    maneuver = world.parts.get("maneuver")
    targets = (-sign * max(abs(m), sigma), -sign * 0.5 * sigma, -sign * max(abs(m), 2.0 * sigma))
    for target in targets:
        changes: dict[str, float] = {}
        if name == "tip":
            ratio = reference_ratio(r, world.scene, maneuver, world.geometry)
            b = float(b_eff(v["base"], v["off"], _direction(world.scene, maneuver)))
            if ratio + target <= 0.02:
                continue
            height = b / (v["com"] * (ratio + target))
            spec = cfg.catalog["loads"][world.parts["load"]["kind"]]["height"]
            changes["height"] = height
            ok = 0.5 * spec[0] <= height <= 1.5 * spec[1]
        elif name == "stop":
            changes["obstacle"] = v["obstacle"] + (target - m)
            ok = changes["obstacle"] - r.radius >= 0.5 * cfg.body(r.body)["gap_min"]
        elif name == "stack":
            want = round_to(v["length_low"] / 2.0 - target, 0.005)      # |d'| (같은 쪽)
            changes["offset"] = want if v["offset"] >= 0 else -want
            ok = 0.01 <= want <= v["length_low"] / 2.0 + 2.0 * cfg.catalog["margins"]["stack"]["half_range"][world.robot.body]
        elif name == "step_in":
            changes["gap"] = v["gap"] + (target - m)
            ok = changes["gap"] - (r.radius + v["ra"]) >= 0.5 * cfg.body(r.body)["gap_min"]
        else:  # spill: 채움만
            changes["fill"] = v["fill"] - (target - m)
            f_lo, f_hi = cfg.container(world.container["kind"])["fill"]
            ok = f_lo <= changes["fill"] <= f_hi
        if ok:
            yield _with_margins(cfg, world.with_values(**changes)), name


# ----------------------------------------------------------------------------------------------------------------
# 감각 모형
# ----------------------------------------------------------------------------------------------------------------


def observe_parts(cfg: SimgenConfig, world: World, ctx: Context, u: np.ndarray, z: np.ndarray, skin: dict) -> tuple[dict, dict]:
    """(Observation에 더할 칸, EnvObs에 더할 칸)."""
    obs_kwargs: dict = {}
    env: dict = {"maneuver": None if world.scene == "p1p3_step_in_carrying_tall" else world.parts.get("maneuver"), "secure_s": world.parts.get("secure_s")}
    v, r = world.values, world.robot
    forklift = ctx.body == "forklift_amr"
    if "load" in world.parts:
        sig = cfg.catalog["p3"]["measure_sigma"]
        scale = cfg.catalog["p3"]["forklift_scale"] if forklift else 1.0
        edge_res = 0.01
        obs_kwargs["load"] = LoadObs(
            kind=world.parts["load"]["kind"], phrase=skin["load_phrase"], ref=skin["load_ref"],
            height=_reading(v["height"], sig["height"] * scale, 0.01, z[0], floor=0.01),
            base=_reading(v["base"], sig["base"] * scale, 0.005, z[1], floor=0.005),
            edge_front=_reading(v["edge_f"], sig["edge"] * scale, edge_res, z[2], floor=edge_res),
            edge_back=_reading(v["edge_b"], sig["edge"] * scale, edge_res, z[3], floor=edge_res),
            edge_side=_reading(v["edge_s"], sig["edge"] * scale, edge_res, z[4], floor=edge_res),
            deck=world.parts["deck"]["kind"], deck_phrase=skin["deck_phrase"],
        )  # fmt: skip
    maneuver = world.parts.get("maneuver")
    if world.scene in ("p3_carrying_tall", "p3p5_carrying_mixed") and maneuver == "brake":
        from robo_jev.simgen.scenes import _distance_sigma

        surface = v["obstacle"] - r.radius
        res = (0.01, 0.01, 0.05, 0.1)[min(int(u[0] * 4), 3)]
        env.update(obstacle=skin["obstacle"], obstacle_distance=_reading(surface, _distance_sigma(cfg, ctx.sensor_suite, surface, u[1]), res, z[5], floor=res))
    if maneuver == "turn":
        g = world.geometry
        from robo_jev.simgen.spec import Reading

        env.update(turn_radius=g["radius"], turn_angle_deg=g["angle_deg"], turn_approach=Reading(g["approach"], 0.0, 0.05), turn_direction=g["direction"])
    if maneuver == "start":
        env["start_accels"] = dict(world.parts["start_accels"])
    if world.scene == "p3_ramp":
        g = world.geometry
        env.update(ramp=skin["ramp"], ramp_angle_deg=g["angle_deg"], ramp_length=g["length"], ramp_direction=maneuver, ramp_end=skin["ramp_end"],
                   detour=skin["detour"], detour_s=world.parts["detour_s"])  # fmt: skip
    if world.scene == "p3_place_stack":
        parts = world.parts
        obs_kwargs["stack"] = StackObs(
            kind=parts["stack_item"]["kind"], phrase=skin["stack_phrase"], ref=skin["stack_ref"], spot=skin["stack_spot"],
            length_top=v["length_top"], length_low=v["length_low"], offset=v["offset"], sigma_place=parts["sigma_place"], cue=parts["cue"],
            recentre_s=parts["recentre_s"], elsewhere_s=parts["elsewhere_s"], check_s=parts["check_s"],
        )  # fmt: skip
    return obs_kwargs, env
