"""P4 안착·정렬 확인의 세계 쪽: 닻 세계(상태·참 측정값·보고), 여유, 사실 하나 바꾸기, 감각 모형 (Task v0b 지시서 §1 P4, P4+P6).

**닻과 결정 여유.** 여유 seat = logit P(제대로 | 보고)(속성 목록 사전으로 계산한 정답의 로그 오즈). 목표 여유를 혼합분포에서 뽑고, 설계 섞기
(좋음 ½, 둘째 ¼, 셋째 ¼)에서 상태·참 값·보고를 뽑아 로그 오즈가 목표 ±0.35 안에 들 때까지 거절 표본한다(최대 300번, 안 되면 Infeasible).
보고(잡음 낀 측정)도 닻 세계가 만든다 — 관측은 그 값을 옮겨 적기만 한다(숨은 값만 바꾼 세계는 같은 관측을 낸다).

**오래된 카메라(P4+P6).** 카메라 측정은 a초 전(빼 둔 칸: 학습 ≤ 3 s), 그 뒤 위험률 λ로 상태가 바뀌었을 수 있다(숨은 값); 힘은 지금 값.
세계 쪽이다(scenes와 같은 층) — 관측 쪽 :mod:`~robo_jev.simgen.p4` 의 상태 모형과 사후를 쓴다.
"""

from __future__ import annotations

import math

import numpy as np

from robo_jev.simgen import p4
from robo_jev.simgen.catalog import SimgenConfig
from robo_jev.simgen.observation import PlacementObs
from robo_jev.simgen.scenes import Context, Infeasible, _make_robot, _robot_params, _u
from robo_jev.simgen.skins import compatible, setting_tags
from robo_jev.simgen.spec import Reading, round_to
from robo_jev.simgen.world import World

__all__ = ["build", "flip", "item_entries", "margins_of", "observe_parts", "prior_redraw", "truth_parts"]


def truth_parts(cfg: SimgenConfig, obs) -> tuple[dict, dict, dict]:
    """참 세계의 (parts, 보고값, 기하) — 여유(사후 로그 오즈)는 관측의 보고로 다시 잰다."""
    place = obs.place
    parts = {"family": place.family, "cls": place.cls, "item": {"kind": place.kind},
             "sigma_e": {n: r.sigma for n, r in place.readings.items()}, "res": {n: r.resolution for n, r in place.readings.items()},
             "age": float(max(place.ages.values())) if place.ages else 0.0}  # fmt: skip
    return parts, {f"m_{n}": r.value for n, r in place.readings.items()}, {}

_RES = {"fine": {"height_mm": 0.05, "depth_mm": 0.05, "tilt_deg": 0.1, "force_n": 0.1, "lateral_n": 0.1},
        "medium": {"height_mm": 0.1, "depth_mm": 0.1, "tilt_deg": 0.1, "force_n": 0.1, "lateral_n": 0.1},
        "coarse": {"height_mm": 0.5, "depth_mm": 0.5, "tilt_deg": 0.1, "force_n": 0.5, "lateral_n": 0.5}}  # fmt: skip
_DESIGN = np.array([0.5, 0.25, 0.125, 0.125])


def item_entries(cfg: SimgenConfig, ctx: Context, family: str) -> list[dict]:
    card = cfg.domain(ctx.domain)
    usable = compatible(card.get("p4_items", []), setting_tags(card, ctx.setting), ctx.body, ctx.scene)
    items = cfg.catalog["p4"]["items"]
    return [e for e in usable if items[e["kind"]]["family"] == family and ctx.body in items[e["kind"]]["bodies"]]


def _family(scene: str) -> str:
    return "insert" if scene == "p4_insert" else "place"


def build(cfg: SimgenConfig, ctx: Context, margins: dict, rng) -> World:
    from robo_jev.simgen.scenes import _with_margins

    family = _family(ctx.scene)
    entries = item_entries(cfg, ctx, family)
    entry = entries[int(rng.integers(len(entries)))]
    item = cfg.catalog["p4"]["items"][entry["kind"]]
    cls_name = item["class"]
    cls = cfg.catalog["p4"][family]["classes"][cls_name]
    spec = cfg.catalog["p4"]
    dims = [d for d in p4.DIMS[family] if d in cls]
    primary = dims[0]
    measured = [primary] + [d for d in dims[1:] if rng.random() < spec["measured_share"].get(d, 1.0)]
    # 감각 잡음: 주 측정은 빼 둔 칸 '거친 게이지'(잡음 / 범위 상한) 범위 안
    sigma_e, res = {}, {}
    for d in measured:
        lo, hi = cls["sensor_sigma"][d]
        if d == primary:
            r_lo, r_hi = ctx.range("gauge_ratio", (lo / hi, 1.0))
            if r_hi < r_lo:
                raise Infeasible("gauge range empty")
            sigma_e[d] = round_to(hi * rng.uniform(r_lo, r_hi), 0.01)
        else:
            sigma_e[d] = round_to(_u(rng, (lo, hi)), 0.01)
        res[d] = _RES[cls_name][d]
    times = {name: (0.0 if name == "continue" else round_to(_u(rng, a["time"]), 0.5)) for name, a in spec["actions"][family].items()}
    age = 0.0
    if ctx.scene == "p4p6_stale_check":
        lo, hi = ctx.range("image_age", spec["stale"]["age"])
        if hi < lo:
            raise Infeasible("image age range empty")
        age = round_to(rng.uniform(lo, hi), 0.1)
        if not lo - 1e-9 <= age <= hi + 1e-9:
            raise Infeasible("rounded image age left the range")
    model = p4.state_model(cfg, family, cls_name, tuple(d for d in dims if d in measured))
    q = p4.stale_q(cfg, age)
    target = margins["seat"]
    for _ in range(300):
        then = int(rng.choice(4, p=_DESIGN))
        changed = bool(age > 0 and rng.random() < q)
        now = int(rng.choice(4, p=model.prior)) if changed else then
        x, m = {}, {}
        for j, d in enumerate(model.dims):
            cam = d in p4.CAMERA_DIMS[family] and age > 0
            s = then if cam else now
            x[d] = float(model.mean[s, j] + model.sd[s, j] * rng.standard_normal())
            m[d] = round_to(x[d] + sigma_e[d] * rng.standard_normal(), res[d])
        se = {d: math.sqrt(sigma_e[d] ** 2 + res[d] ** 2 / 12.0) for d in model.dims}
        worlds = p4.posterior_worlds(model, m, se, age, q)
        if abs(p4.logit_good(worlds) - target) < 0.35:
            values = {"state": float(now), "state_then": float(then), "changed": float(changed)}
            values.update({f"x_{d}": x[d] for d in model.dims})
            values.update({f"m_{d}": m[d] for d in model.dims})
            parts = {"item": entry, "family": family, "cls": cls_name, "sigma_e": sigma_e, "res": res, "times": times, "age": age,
                     "tolerance": dict(cls["tolerance"]), "gauge_ratio": round(sigma_e[primary] / cls["sensor_sigma"][primary][1], 4)}  # fmt: skip
            p = _robot_params(cfg, ctx.body, rng, loaded=False)
            world = World(ctx.scene, ctx.domain, ctx.variant, _make_robot(p, 0.0, 0.0), None, None, values, {}, parts=parts)
            return _with_margins(cfg, world)
    raise Infeasible("no measurement reached the seat margin")


def prior_redraw(cfg: SimgenConfig, world: World, rng) -> World:
    """순방향 SBC용: 상태(그때·바뀜·지금)와 참 측정값·보고를 정답의 사전(클래스의 상태 사전, 위험률 λ의 바뀜, 상태 모형, 감각 잡음)에서
    다시 뽑는다 — 닻의 설계 섞기·거절 표본을 쓰지 않는다. 관측(보고)은 이 세계의 값이다."""
    parts = world.parts
    family, age = parts["family"], parts["age"]
    dims = tuple(d for d in p4.DIMS[family] if f"m_{d}" in world.values)
    model = p4.state_model(cfg, family, parts["cls"], dims)
    q = p4.stale_q(cfg, age)
    then = int(rng.choice(len(model.prior), p=model.prior))
    changed = bool(age > 0 and rng.random() < q)
    now = int(rng.choice(len(model.prior), p=model.prior)) if changed else then
    values = {"state": float(now), "state_then": float(then), "changed": float(changed)}
    for j, d in enumerate(model.dims):
        s = then if (d in p4.CAMERA_DIMS[family] and age > 0) else now
        x = float(model.mean[s, j] + model.sd[s, j] * rng.standard_normal())
        values[f"x_{d}"] = x
        values[f"m_{d}"] = round_to(x + parts["sigma_e"][d] * rng.standard_normal(), parts["res"][d])
    return world.with_values(**values)


def _readings_of(world: World) -> tuple[dict, dict]:
    parts = world.parts
    m = {k[2:]: v for k, v in world.values.items() if k.startswith("m_")}
    se = {d: math.sqrt(parts["sigma_e"][d] ** 2 + parts["res"][d] ** 2 / 12.0) for d in m}
    return m, se


def margins_of(cfg: SimgenConfig, world: World) -> dict[str, float]:
    parts = world.parts
    m, se = _readings_of(world)
    dims = tuple(d for d in p4.DIMS[parts["family"]] if d in m)
    model = p4.state_model(cfg, parts["family"], parts["cls"], dims)
    worlds = p4.posterior_worlds(model, m, se, parts["age"], p4.stale_q(cfg, parts["age"]))
    return {"seat": p4.logit_good(worlds)}


def flip(cfg: SimgenConfig, world: World, name: str, rng):
    """주 측정(높이·깊이) 보고 하나만 옮겨 로그 오즈의 부호를 바꾼다 — 같은 다른 보고·같은 숨은 값."""
    from robo_jev.simgen.scenes import _with_margins

    parts = world.parts
    primary = p4.DIMS[parts["family"]][0]
    m0 = world.values[f"m_{primary}"]
    res = parts["res"][primary]
    sign = 1.0 if world.margins["seat"] >= 0 else -1.0
    for step in np.arange(1, 400) * res:
        for candidate in (m0 + step, m0 - step):
            moved = world.with_values(**{f"m_{primary}": round_to(candidate, res)})
            margins = margins_of(cfg, moved)
            if np.sign(margins["seat"]) == -sign and abs(margins["seat"]) >= 0.5:
                yield _with_margins(cfg, moved), name
                return


def observe_parts(cfg: SimgenConfig, world: World, ctx: Context, u, z, skin: dict) -> tuple[dict, dict]:
    parts = world.parts
    m, _ = _readings_of(world)
    readings = {d: Reading(m[d], parts["sigma_e"][d], parts["res"][d]) for d in m}
    ages = {d: parts["age"] for d in m if d in p4.CAMERA_DIMS[parts["family"]]} if parts["age"] > 0 else {}
    place = PlacementObs(
        family=parts["family"], kind=parts["item"]["kind"], cls=parts["cls"], item=skin["item"], target=skin["target"], readings=readings,
        ages=ages, tolerance=parts["tolerance"], action_times=dict(parts["times"]), stale_reason=skin.get("stale_reason"), gauge_ratio=parts["gauge_ratio"],
    )  # fmt: skip
    return {"place": place}, {}
