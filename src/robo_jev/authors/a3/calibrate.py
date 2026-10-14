"""Calibrate A3's analytic aim against MuJoCo (no labels come from here — the proxy only aims the decisive number).

For random states and single hidden draws (base prior) the MuJoCo flip point of the key event is found by bisection on the decisive
number and compared with the analytic threshold:

* **index dial** — the acceleration at the flip, a(T*) = r·c(Θ)/T*², over the proxy's threshold min(μ g, g b/h_c): the sliding factor
  from draws whose sliding threshold is well below the tipping one (ratio < 0.8), the tipping factor from the opposite draws (> 1.25);
* **instrument stand** — the factor on the release impulse that puts the proxy's flip at MuJoCo's flip set-down (draws whose release
  height is > 0 and whose flip is not static, i.e. the combined centre of mass is still inside the feet);
* **hold-down clamp** — the press peak at the flip over the static holding capacity.

The medians are written into :data:`robo_jev.authors.a3.scenes` ``calib`` (fixed constants, so planning is deterministic); this
module re-measures them.
"""

from __future__ import annotations

import math
from typing import Any, Callable

import numpy as np

from robo_jev.authors.a3 import catalog as cat
from robo_jev.authors.a3 import scenes, sim

__all__ = ["calibrate_scene"]


def _bisect(event: Callable[[float], bool], lo: float, hi: float, *, log: bool, steps: int = 11) -> float | None:
    """The flip point of a monotone event between `lo` and `hi` (None when it does not flip inside)."""
    e_lo, e_hi = event(lo), event(hi)
    if e_lo == e_hi:
        return None
    a, b = (math.log(lo), math.log(hi)) if log else (lo, hi)
    for _ in range(steps):
        mid = (a + b) / 2
        if event(math.exp(mid) if log else mid) == e_lo:
            a = mid
        else:
            b = mid
    m = (a + b) / 2
    return math.exp(m) if log else m


def _one_draw(sc: scenes.Scene, state: dict, rng: np.random.Generator) -> dict[str, float]:
    points = {q: np.array([rng.random()]) for q in sc.hidden_names}
    values = sc.values(state, "base", points)
    return {q: float(v[0]) for q, v in values.items()}


def _key(sc: scenes.Scene, state: dict, values: dict[str, float]) -> bool:
    return sc.simulate(state, values)["outcome"] in sc.key_events


def calibrate_dial(n: int, seed: int) -> dict[str, Any]:
    sc = scenes.SCENES["indexdial"]
    rng = np.random.default_rng(seed)
    rows: dict[str, list[float]] = {"slide": [], "tip": []}
    while len(rows["slide"]) < n or len(rows["tip"]) < n:
        state = sc.sample_state(rng)
        values = _one_draw(sc, state, rng)
        length, width, height = (v / 1000 for v in state["dims_mm"])
        half = max(min(width / 2, length / 2 - abs(values["ex"]) * length), 1e-3)
        a_slide, a_tip = values["mu"] * sim.GRAVITY, sim.GRAVITY * half / (values["kh"] * height)
        ratio = a_slide / a_tip
        mode = "slide" if ratio < 0.8 else ("tip" if ratio > 1.25 else None)
        if mode is None or len(rows[mode]) >= n:
            continue
        radius = state["radius_mm"] / 1000 + values["dr"]
        coef = sim.dial_peak_accel(state["angle_deg"], 1.0, 1.0)
        a_proxy = min(a_slide, a_tip)
        t_proxy = math.sqrt(coef * radius / a_proxy)
        lo, hi = max(0.2, t_proxy / 2.5), min(8.0, t_proxy * 2.5)

        def event(t: float) -> bool:
            st = dict(state)
            st["index_s"] = t
            return _key(sc, st, values)

        flip = _bisect(event, lo, hi, log=True)
        if flip is None:
            continue
        if mode == "slide":
            # the μ scale that puts the point-mass slip model's 10 mm flip at MuJoCo's flip time
            def slips(scale: float) -> bool:
                return bool(sim.dial_slip(state["angle_deg"], flip, np.array([radius]), np.array([values["mu"] * scale]))[0] >= sim.DIAL_SHIFT_M)

            scale = _bisect(slips, 0.3, 3.0, log=True, steps=14)
            if scale is not None:
                rows[mode].append(scale)
        else:
            rows[mode].append((t_proxy / flip) ** 2)  # a(T*) / a_proxy
    return {mode: _summary(v) for mode, v in rows.items()}


def calibrate_stand(n: int, seed: int) -> dict[str, Any]:
    sc = scenes.SCENES["mayostand"]
    rng = np.random.default_rng(seed)
    factors: list[float] = []
    static_ok = 0
    tries = 0
    while len(factors) < n and tries < 40 * n:
        tries += 1
        state = sc.sample_state(rng)
        if state["release_mm"] == 0:
            continue
        values = _one_draw(sc, state, rng)
        lo, hi = sc.decisive_bounds(state)

        def event(d: float) -> bool:
            st = dict(state)
            st["setdown_mm"] = d
            return _key(sc, st, values)

        flip = _bisect(event, lo, hi, log=False, steps=9)
        if flip is None:
            continue
        st = dict(state)
        st["setdown_mm"] = flip
        arrays = {q: np.array([v]) for q, v in values.items()}
        saved = dict(sc.calib)
        try:
            sc.calib["drop"] = 1.0
            parts = _stand_terms(sc, st, arrays)
        finally:
            sc.calib.update(saved)
        if parts["margin"] <= 0:
            static_ok += 1
            continue
        if parts["energy"] <= 0:
            continue
        factors.append(math.sqrt(parts["barrier"] / parts["energy"]))
    return {"drop": _summary(factors), "static_flips": static_ok}


def _stand_terms(sc: scenes.MayoStand, state: dict, values: dict[str, np.ndarray]) -> dict[str, float]:
    reach, tray_len, tray_top = state["reach_mm"] / 1000, state["tray_mm"] / 1000, state["height_mm"] / 1000
    length, _, height = (v / 1000 for v in state["dims_mm"])
    base, load = float(values["base_kg"][0]), float(values["load_kg"][0])
    parts = {nm: (np.array(c), np.array(h)) for nm, c, h in sim._stand_parts(reach, tray_len, tray_top)}
    masses = [(base, parts["base"]), (sim.POST_MASS, parts["post"]), (sim.TRAY_MASS, parts["tray"])]
    x_load = state["setdown_mm"] / 1000 + float(values["dx"][0]) + float(values["com_frac"][0]) * length
    z_load = tray_top + height / 2
    total = base + sim.POST_MASS + sim.TRAY_MASS + load
    x_com = (sum(m * c[0] for m, (c, _) in masses) + load * x_load) / total
    margin = reach - x_com
    inertia = load * ((x_load - reach) ** 2 + z_load**2) + load * (length**2 + height**2) / 12
    for m, (c, h) in masses:
        inertia += m * ((c[0] - reach) ** 2 + c[2] ** 2) + m * ((2 * h[0]) ** 2 + (2 * h[2]) ** 2) / 12
    lever = max(x_load - reach, 0.0)
    impulse = load * math.sqrt(2 * sim.GRAVITY * state["release_mm"] / 1000)
    energy = (impulse * lever) ** 2 / (2 * inertia)
    barrier = total * sim.GRAVITY * max(margin, 0.0) * math.radians(sc.calib["rock_deg"])
    return {"margin": margin, "energy": energy, "barrier": barrier}


def calibrate_clamp(n: int, seed: int) -> dict[str, Any]:
    sc = scenes.SCENES["holddown"]
    rng = np.random.default_rng(seed)
    ratios: list[float] = []
    while len(ratios) < n:
        state = sc.sample_state(rng)
        values = _one_draw(sc, state, rng)
        capacity = sim.clamp_capacity(mu_nest=values["mu_nest"], mu_pad=values["mu_pad"],
                                      clamp_force=values["force_factor"] * state["clamp_n"], mass=state["mass_kg"])

        def event(f: float) -> bool:
            st = dict(state)
            st["press_n"] = f
            return _key(sc, st, values)

        flip = _bisect(event, capacity * 0.6, capacity * 1.8, log=True, steps=10)
        if flip is not None:
            ratios.append(flip / capacity)
    return {"press": _summary(ratios)}


def _summary(values: list[float]) -> dict[str, Any]:
    arr = np.array(values, dtype=float)
    if not arr.size:
        return {"n": 0}
    return {"n": int(arr.size), "median": float(np.median(arr)), "p10": float(np.percentile(arr, 10)), "p90": float(np.percentile(arr, 90)),
            "mean": float(arr.mean())}


def calibrate_scene(name: str, n: int = 40, seed: int = 0) -> dict[str, Any]:
    if name == "indexdial":
        return calibrate_dial(n, seed)
    if name == "mayostand":
        return calibrate_stand(n, seed)
    if name == "holddown":
        return calibrate_clamp(n, seed)
    raise ValueError(f"unknown scene {name!r}")
