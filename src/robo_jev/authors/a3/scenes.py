"""A3's three scenes — what the work order states, what stays hidden, the analytic aim, the MuJoCo rollout and the alternatives.

Each scene (one per law) defines:

* the **visible state** — catalog keys (words) and stated numbers (:meth:`Scene.sample_state`);
* the **hidden specs** of a state under the base prior (:meth:`Scene.hidden_specs`) — every range comes from a named catalog option
  (:mod:`robo_jev.authors.a3.catalog`);
* an **analytic proxy** of the key event per hidden draw (:meth:`Scene.proxy_events`) — the sliding (μ g), tipping (g b/h), support
  (centre of mass over the front feet, plus the energy of the release impulse) and friction-capacity thresholds with calibration
  factors measured against MuJoCo (``scripts/author_a3.py calibrate``). It only **aims** the decisive stated number so the key labels
  spread flat over 0–1 (:meth:`Scene.aim`); every label comes from MuJoCo (:meth:`Scene.simulate`);
* the **alternative actions** a reviewer could choose instead of the plan (:meth:`Scene.alternatives`) and the **one-factor changes**
  of a pair (:meth:`Scene.one_factor_changes`).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from robo_jev.authors.a3 import catalog as cat
from robo_jev.authors.a3 import sim

__all__ = ["SCENES", "Alternative", "HoldDown", "IndexDial", "MayoStand", "Scene", "scene"]

G = sim.GRAVITY


@dataclass(frozen=True)
class Alternative:
    """An action the reviewer may pick instead of the plan: its option key, its work-order wording, whether it was written as the
    riskier kind (selection balance only — labels come from MuJoCo), its visible changes and when it applies."""

    key: str
    words: str
    riskier: bool
    apply: Callable[[Mapping[str, Any]], dict[str, Any]]
    valid: Callable[[Mapping[str, Any]], bool]


def _round(value: float, step: float) -> float:
    return round(round(float(value) / step) * step, 6)


class Scene:
    """Common interface (module docstring)."""

    name: str
    law: str
    phenomena: tuple[str, ...]
    domain: str
    outcomes: tuple[str, ...]
    key_events: tuple[str, ...]
    severe: str
    costs: Mapping[str, float]
    decisive: str
    risk_up: bool  # True: a larger decisive value raises the key risk
    step: float
    hidden_names: tuple[str, ...]
    #: Calibration factors of the analytic proxy (``scripts/author_a3.py calibrate``).
    calib: dict[str, float]

    # --- per scene ---
    def sample_state(self, rng: np.random.Generator) -> dict[str, Any]:
        raise NotImplementedError

    def hidden_specs(self, state: Mapping[str, Any]) -> dict[str, list]:
        raise NotImplementedError

    def decisive_bounds(self, state: Mapping[str, Any]) -> tuple[float, float]:
        raise NotImplementedError

    def proxy_events(self, state: Mapping[str, Any], values: Mapping[str, np.ndarray]) -> np.ndarray:
        raise NotImplementedError

    def simulate(self, state: Mapping[str, Any], values: Mapping[str, float]) -> dict[str, Any]:
        raise NotImplementedError

    def alternatives(self) -> list[Alternative]:
        raise NotImplementedError

    def one_factor_changes(self, state: Mapping[str, Any], rng: np.random.Generator) -> list[tuple[str, Any, str, float]]:
        """Candidate one-fact changes of `state`: (field, new value, ``word`` | ``number``, weight)."""
        raise NotImplementedError

    # --- shared ---
    def values(self, state: Mapping[str, Any], prior: str, points: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Hidden values of the K shared draw points under `prior`."""
        specs = cat.prior_ranges(self.hidden_specs(state))[prior]
        return {q: cat.draw_values(specs[q], points[q]) for q in self.hidden_names}

    def proxy(self, state: Mapping[str, Any], points: Mapping[str, np.ndarray], prior: str = "base") -> float:
        return float(np.mean(self.proxy_events(state, self.values(state, prior, points))))

    def with_value(self, state: Mapping[str, Any], value: float) -> dict[str, Any]:
        out = dict(state)
        out[self.decisive] = _round(value, self.step)
        return out

    def aim(self, state: Mapping[str, Any], points: Mapping[str, np.ndarray], target: float) -> tuple[float, float] | None:
        """The decisive value (on its display step) whose proxy key share under the base prior is closest to `target`, and that share;
        None when the bounds cannot bracket the target within 0.08."""
        lo, hi = self.decisive_bounds(state)
        p_lo, p_hi = self.proxy(self.with_value(state, lo), points), self.proxy(self.with_value(state, hi), points)
        p_min, p_max = min(p_lo, p_hi), max(p_lo, p_hi)
        if target < p_min - 0.08 or target > p_max + 0.08:
            return None
        a, b = lo, hi
        for _ in range(40):
            mid = (a + b) / 2
            p = self.proxy(self.with_value(state, mid), points)
            rising = p_hi >= p_lo
            if (p < target) == rising:
                a = mid
            else:
                b = mid
            if b - a < self.step / 4:
                break
        best = None
        for v in (a, b, _round(a, self.step), _round(b, self.step)):
            v = min(max(_round(v, self.step), _round(lo, self.step)), _round(hi, self.step))
            p = self.proxy(self.with_value(state, v), points)
            if best is None or abs(p - target) < abs(best[1] - target):
                best = (v, p)
        return best

    def cost(self, freq: Mapping[str, float]) -> float:
        return float(sum(float(freq[o]) * float(self.costs[o]) for o in self.outcomes))

    def apply_change(self, state: Mapping[str, Any], field: str, value: Any) -> dict[str, Any]:
        """`state` with one stated fact changed (a pair's one-fact change)."""
        out = dict(state)
        out[field] = value
        return out

    def changed_value(self, state: Mapping[str, Any], field: str) -> Any:
        """The value of a changeable field (the inverse of :meth:`apply_change`)."""
        return state[field]


# ==========================================================================
# Law A — index dial (light manufacturing)
# ==========================================================================

#: Parts: key → (words, length range, width range, height range (mm), mass range (kg)).
DIAL_PARTS = {
    "valve_body": ("valve body casting", (90, 140), (70, 110), (120, 220), (2.5, 6.0)),
    "end_cover": ("gearbox end cover", (140, 220), (120, 180), (50, 90), (1.2, 3.0)),
    "servo": ("servo motor with flange", (80, 110), (80, 110), (160, 280), (2.0, 4.5)),
    "filter": ("filter housing", (90, 130), (90, 130), (180, 320), (0.8, 2.2)),
    "enclosure": ("assembled sensor enclosure", (100, 160), (70, 110), (60, 110), (0.5, 1.4)),
    "battery": ("battery module", (150, 230), (90, 140), (90, 150), (3.0, 7.5)),
    "manifold": ("pump manifold block", (100, 160), (80, 120), (80, 140), (3.5, 8.0)),
    "reservoir": ("sealed coolant reservoir", (120, 180), (90, 130), (200, 300), (1.5, 4.0)),
    "terminal": ("terminal block stack", (60, 90), (50, 80), (140, 240), (0.6, 1.5)),
    "bearing": ("bearing housing", (110, 160), (110, 160), (70, 120), (2.5, 6.0)),
}
DIAL_ANGLES = (45, 60, 90, 120)
DIAL_STATIONS = {45: 8, 60: 6, 90: 4, 120: 3}
_DIAL_PLATE_WEIGHTS = {"dry": 0.34, "coolant": 0.2, "oil": 0.14, "dust": 0.16, "mat": 0.16}


class IndexDial(Scene):
    name = "indexdial"
    law = "A"
    phenomena = ("tip_or_slide_under_acceleration",)
    domain = "light_manufacturing"
    outcomes = sim.OUTCOMES["indexdial"]
    key_events = ("slides_off_mark", "rocks_back", "falls_over")
    severe = "falls_over"
    costs = {"stays_put": 0.0, "slides_off_mark": 1.0, "rocks_back": 1.5, "falls_over": 2.5}
    decisive = "index_s"
    risk_up = False
    step = 0.01
    hidden_names = ("mu", "kh", "ex", "dr")
    #: Sliding: a point mass with Coulomb friction integrated over the cycloidal profile (:func:`robo_jev.authors.a3.sim.dial_slip`,
    #: event = slides ≥ 10 mm); ``slide`` scales μ (MuJoCo agrees at μ × 0.96, p10–p90 0.93–0.98). Tipping: the static g b/h_c
    #: threshold times ``tip``, measured 2026-10-14
    #: (``author_a3.py calibrate``, 40 draws): MuJoCo lifts an edge 10° (or tips) at 1.27 g b/h_c (p10–p90 1.13–1.51).
    calib = {"slide": 0.96, "tip": 1.27}

    def sample_state(self, rng: np.random.Generator) -> dict[str, Any]:
        part = str(rng.choice(sorted(DIAL_PARTS)))
        _, lr, wr, hr, mr = DIAL_PARTS[part]
        length, width = sorted((float(rng.uniform(*lr)), float(rng.uniform(*wr))), reverse=True)
        plates = sorted(_DIAL_PLATE_WEIGHTS)
        weights = np.array([_DIAL_PLATE_WEIGHTS[p] for p in plates])
        return {
            "part": part,
            "dims_mm": [int(_round(length, 5)), int(_round(width, 5)), int(_round(rng.uniform(*hr), 5))],
            "mass_kg": _round(rng.uniform(*mr), 0.1),
            "build": str(rng.choice(sorted(cat.DIAL_BUILD))),
            "base": str(rng.choice(sorted(cat.DIAL_BASE))),
            "plate": str(rng.choice(plates, p=weights / weights.sum())),
            "placement": str(rng.choice(sorted(cat.DIAL_PLACEMENT))),
            "orient": str(rng.choice(["radial", "travel"])),
            "radius_mm": int(_round(rng.uniform(180, 460), 5)),
            "angle_deg": int(rng.choice(DIAL_ANGLES)),
            "index_s": 1.0,
        }

    def hidden_specs(self, state: Mapping[str, Any]) -> dict[str, list]:
        _, kh, ex = cat.DIAL_BUILD[state["build"]]
        return {"mu": ["u", *cat.dial_mu_range(state["base"], state["plate"])], "kh": ["u", *kh], "ex": ["u", *ex],
                "dr": ["n", 0.0, cat.DIAL_PLACEMENT[state["placement"]].sigma]}

    def decisive_bounds(self, state: Mapping[str, Any]) -> tuple[float, float]:
        return (0.25, 4.0)

    @staticmethod
    def _coefficient(angle_deg: float, index_s: float) -> float:
        return sim.dial_peak_accel(angle_deg, index_s, 1.0)

    def proxy_events(self, state: Mapping[str, Any], values: Mapping[str, np.ndarray]) -> np.ndarray:
        length, width, height = (v / 1000 for v in state["dims_mm"])
        radius = state["radius_mm"] / 1000 + values["dr"]
        accel = self._coefficient(state["angle_deg"], state["index_s"]) * radius
        half_base = np.maximum(np.minimum(width / 2, length / 2 - np.abs(values["ex"]) * length), 1e-3)
        slide = sim.dial_slip(state["angle_deg"], state["index_s"], radius, values["mu"] * self.calib["slide"]) >= sim.DIAL_SHIFT_M
        tip = accel > G * half_base / (values["kh"] * height) * self.calib["tip"]
        return slide | tip

    def apply_change(self, state: Mapping[str, Any], field: str, value: Any) -> dict[str, Any]:
        out = dict(state)
        if field == "part_height_mm":
            out["dims_mm"] = [state["dims_mm"][0], state["dims_mm"][1], int(value)]
        else:
            out[field] = value
        return out

    def changed_value(self, state: Mapping[str, Any], field: str) -> Any:
        return state["dims_mm"][2] if field == "part_height_mm" else state[field]

    def geometry(self, state: Mapping[str, Any], values: Mapping[str, float]) -> dict[str, Any]:
        length, width, height = (v / 1000 for v in state["dims_mm"])
        offset = float(values["ex"]) * length
        com_z = float(values["kh"]) * height - height / 2
        if state["orient"] == "radial":
            extents, com = (length, width, height), (offset, 0.0, com_z)
        else:
            extents, com = (width, length, height), (0.0, offset, com_z)
        return {"extents": extents, "com": com, "radius": state["radius_mm"] / 1000 + float(values["dr"])}

    def simulate(self, state: Mapping[str, Any], values: Mapping[str, float]) -> dict[str, Any]:
        geo = self.geometry(state, values)
        model = sim.DialModel.build(*geo["extents"], float(state["mass_kg"]), geo["com"])
        profile = _dial_profile(float(state["angle_deg"]), float(state["index_s"]))
        return sim.run_dial(model, profile, radius=geo["radius"], mu=float(values["mu"]))

    def alternatives(self) -> list[Alternative]:
        s = self.step
        return [
            Alternative("slow_index", "stretch the index time x1.5", False,
                        lambda st: {"index_s": _round(st["index_s"] * 1.5, s)}, lambda st: st["index_s"] * 1.5 <= 6.0),
            Alternative("inner_spot", "move the part 120 mm inward", False,
                        lambda st: {"radius_mm": int(st["radius_mm"] - 120)}, lambda st: st["radius_mm"] - 120 >= 150),
            Alternative("outer_spot", "move the part 100 mm outward", True,
                        lambda st: {"radius_mm": int(st["radius_mm"] + 100)}, lambda st: st["radius_mm"] + 100 <= 480),
            Alternative("fit_mat", "put an anti-slip mat under it", False,
                        lambda st: {"plate": "mat"}, lambda st: st["plate"] != "mat"),
            Alternative("quarter_turn", "turn the part a quarter turn", False,
                        lambda st: {"orient": "travel" if st["orient"] == "radial" else "radial"},
                        lambda st: abs(st["dims_mm"][0] - st["dims_mm"][1]) >= 20),
            Alternative("quick_index", "cut the index time to x0.8", True,
                        lambda st: {"index_s": _round(st["index_s"] * 0.8, s)}, lambda st: st["index_s"] * 0.8 >= 0.25),
        ]

    def one_factor_changes(self, state: Mapping[str, Any], rng: np.random.Generator) -> list[tuple[str, Any, str, float]]:
        out: list[tuple[str, Any, str, float]] = []
        out += [("base", k, "word", 0.15 / (len(cat.DIAL_BASE) - 1)) for k in cat.DIAL_BASE if k != state["base"]]
        out += [("plate", k, "word", 0.15 / (len(cat.DIAL_PLATE) - 1)) for k in cat.DIAL_PLATE if k != state["plate"]]
        out += [("build", k, "word", 0.12 / (len(cat.DIAL_BUILD) - 1)) for k in cat.DIAL_BUILD if k != state["build"]]
        out += [("placement", k, "word", 0.08 / (len(cat.DIAL_PLACEMENT) - 1)) for k in cat.DIAL_PLACEMENT if k != state["placement"]]
        r = state["radius_mm"]
        for delta in (-150, -100, -60, 60, 100, 150):
            if 150 <= r + delta <= 480:
                out.append(("radius_mm", int(r + delta), "number", 0.15 / 6))
        out += [("angle_deg", a, "number", 0.10 / 3) for a in DIAL_ANGLES if a != state["angle_deg"]]
        height = state["dims_mm"][2]
        for factor in (0.6, 0.75, 1.3, 1.5):
            new = int(_round(height * factor, 5))
            if 40 <= new <= 400 and new != height:
                out.append(("part_height_mm", new, "number", 0.10 / 4))
        mass = state["mass_kg"]
        for factor in (0.5, 0.7, 1.4, 2.0):
            new = _round(mass * factor, 0.1)
            if new >= 0.2 and new != mass:
                out.append(("mass_kg", new, "number", 0.07 / 4))
        for factor in (0.85, 0.9, 1.1, 1.15):
            new = _round(state["index_s"] * factor, self.step)
            if new != state["index_s"] and 0.25 <= new <= 4.0:
                out.append(("index_s", new, "number", 0.08 / 4))
        return out


_PROFILE_CACHE: dict[tuple[float, float], Any] = {}


def _dial_profile(angle_deg: float, index_s: float) -> Any:
    key = (round(angle_deg, 6), round(index_s, 6))
    if key not in _PROFILE_CACHE:
        if len(_PROFILE_CACHE) > 256:
            _PROFILE_CACHE.clear()
        _PROFILE_CACHE[key] = sim.index_profile(angle_deg, index_s)
    return _PROFILE_CACHE[key]


# ==========================================================================
# Law B — cantilever instrument stand (hospital sterile services / theatre set-up)
# ==========================================================================

#: Container dims per set: (length along the tray, width, height) ranges in mm.
STAND_CASES = {
    "dressing": ((220, 300), (180, 240), (60, 100)),
    "basic": ((260, 360), (200, 260), (70, 110)),
    "lap": ((300, 400), (220, 280), (90, 140)),
    "major": ((320, 400), (240, 300), (100, 150)),
    "ortho": ((300, 400), (240, 300), (110, 160)),
    "power": ((260, 360), (220, 280), (110, 160)),
}
STAND_RELEASES = (0, 5, 10, 15, 20, 30, 40)
STAND_RELEASE_WEIGHTS = (0.18, 0.14, 0.18, 0.14, 0.14, 0.12, 0.10)
#: The container's rear end may sit at most this far behind the post axis, its far end exactly at the tray's end.
STAND_REAR_LIMIT = 0.03


class MayoStand(Scene):
    name = "mayostand"
    law = "B"
    phenomena = ("centre_of_mass_past_support_edge",)
    domain = "hospital"
    outcomes = sim.OUTCOMES["mayostand"]
    key_events = ("rocks", "tips_over")
    severe = "tips_over"
    costs = {"stays_level": 0.0, "rocks": 0.6, "tips_over": 3.0}
    decisive = "setdown_mm"
    risk_up = True
    step = 5.0
    hidden_names = ("load_kg", "com_frac", "base_kg", "dx")
    #: Measured 2026-10-14 (40 draws with a release height > 0): the release-impulse factor that puts the proxy's flip at MuJoCo's,
    #: median 1.66 (p10–p90 1.22–2.75); static flips follow the support rule exactly.
    calib = {"drop": 1.66, "rock_deg": sim.STAND_ROCK_DEG}

    def sample_state(self, rng: np.random.Generator) -> dict[str, Any]:
        kit = str(rng.choice(sorted(cat.STAND_SET)))
        lr, wr, hr = STAND_CASES[kit]
        return {
            "set": kit,
            "dims_mm": [int(_round(rng.uniform(*lr), 10)), int(_round(rng.uniform(*wr), 10)), int(_round(rng.uniform(*hr), 10))],
            "packing": str(rng.choice(sorted(cat.STAND_PACKING))),
            "stand": str(rng.choice(sorted(cat.STAND_MODEL))),
            "method": str(rng.choice(sorted(cat.STAND_METHOD))),
            "reach_mm": int(_round(rng.uniform(120, 280), 10)),
            "tray_mm": int(_round(rng.uniform(500, 640), 10)),
            "height_mm": int(_round(rng.uniform(880, 1180), 10)),
            "release_mm": int(rng.choice(STAND_RELEASES, p=np.array(STAND_RELEASE_WEIGHTS) / sum(STAND_RELEASE_WEIGHTS))),
            "setdown_mm": 250.0,
            "split": False,
            "ballast_kg": 0.0,
        }

    def hidden_specs(self, state: Mapping[str, Any]) -> dict[str, list]:
        kit = cat.STAND_SET[state["set"]]
        scale = 0.55 if state.get("split") else 1.0
        model = cat.STAND_MODEL[state["stand"]]
        ballast = float(state.get("ballast_kg") or 0.0)
        packing = cat.STAND_PACKING[state["packing"]]
        return {"load_kg": ["u", round(kit.lo * scale, 4), round(kit.hi * scale, 4)], "com_frac": ["u", packing.lo, packing.hi],
                "base_kg": ["u", model.lo + ballast, model.hi + ballast], "dx": ["n", 0.0, cat.STAND_METHOD[state["method"]].sigma]}

    def decisive_bounds(self, state: Mapping[str, Any]) -> tuple[float, float]:
        length = state["dims_mm"][0]
        lo = length / 2 - STAND_REAR_LIMIT * 1000
        hi = state["tray_mm"] - sim.TRAY_BACK * 1000 - length / 2
        return (math.ceil(lo / self.step) * self.step, math.floor(hi / self.step) * self.step)

    def proxy_events(self, state: Mapping[str, Any], values: Mapping[str, np.ndarray]) -> np.ndarray:
        reach, tray_len, tray_top = state["reach_mm"] / 1000, state["tray_mm"] / 1000, state["height_mm"] / 1000
        length, _, height = (v / 1000 for v in state["dims_mm"])
        base = np.asarray(values["base_kg"], dtype=float)
        load = np.asarray(values["load_kg"], dtype=float)
        parts = {n: (np.array(c), np.array(h)) for n, c, h in sim._stand_parts(reach, tray_len, tray_top)}
        masses = [(base, parts["base"]), (np.full_like(base, sim.POST_MASS), parts["post"]), (np.full_like(base, sim.TRAY_MASS), parts["tray"])]
        x_load = state["setdown_mm"] / 1000 + values["dx"] + values["com_frac"] * length
        z_load = tray_top + height / 2
        total = base + sim.POST_MASS + sim.TRAY_MASS + load
        x_com = (sum(m * c[0] for m, (c, _) in masses) + load * x_load) / total
        z_com = (sum(m * c[2] for m, (c, _) in masses) + load * z_load) / total
        margin = reach - x_com
        # inertia about the front-edge axis (y through x = reach on the floor): point masses + own box terms
        inertia = load * ((x_load - reach) ** 2 + z_load**2) + load * (length**2 + height**2) / 12
        for m, (c, h) in masses:
            inertia = inertia + m * ((c[0] - reach) ** 2 + c[2] ** 2) + m * ((2 * h[0]) ** 2 + (2 * h[2]) ** 2) / 12
        lever = np.maximum(x_load - reach, 0.0)
        impulse = load * math.sqrt(2 * G * state["release_mm"] / 1000) * self.calib["drop"]
        energy = (impulse * lever) ** 2 / (2 * inertia)
        lift_barrier = total * G * np.maximum(margin, 0.0) * math.radians(self.calib["rock_deg"])
        return (margin < 0) | (energy > lift_barrier)

    def simulate(self, state: Mapping[str, Any], values: Mapping[str, float]) -> dict[str, Any]:
        length, width, height = (v / 1000 for v in state["dims_mm"])
        model = sim.StandModel.build(state["reach_mm"] / 1000, state["tray_mm"] / 1000, state["height_mm"] / 1000, (length, width, height),
                                     base_mass=float(values["base_kg"]), load_mass=float(values["load_kg"]),
                                     load_com_x=float(values["com_frac"]) * length)
        return sim.run_stand(model, setdown_x=state["setdown_mm"] / 1000 + float(values["dx"]), release_h=state["release_mm"] / 1000)

    def alternatives(self) -> list[Alternative]:
        def lo(st: Mapping[str, Any]) -> float:
            return self.decisive_bounds(st)[0]

        def hi(st: Mapping[str, Any]) -> float:
            return self.decisive_bounds(st)[1]

        return [
            Alternative("nearer_post", "set it 150 mm nearer the post", False,
                        lambda st: {"setdown_mm": float(st["setdown_mm"] - 150)}, lambda st: st["setdown_mm"] - 150 >= lo(st)),
            Alternative("lower_to_touch", "lower it to touch before letting go", False,
                        lambda st: {"release_mm": 0}, lambda st: st["release_mm"] >= 5),
            Alternative("far_end", "slide it to the far end of the tray", True,
                        lambda st: {"setdown_mm": float(hi(st))}, lambda st: hi(st) - st["setdown_mm"] >= 40),
            Alternative("split_set", "take the heavy layer out first (half weight)", False,
                        lambda st: {"split": True}, lambda st: not st.get("split")),
            Alternative("ballast_plate", "put a 4 kg ballast plate on the base", False,
                        lambda st: {"ballast_kg": 4.0}, lambda st: not st.get("ballast_kg")),
            Alternative("higher_release", "let go 25 mm higher to clear the drape", True,
                        lambda st: {"release_mm": int(st["release_mm"] + 25)}, lambda st: st["release_mm"] + 25 <= 70),
        ]

    def one_factor_changes(self, state: Mapping[str, Any], rng: np.random.Generator) -> list[tuple[str, Any, str, float]]:
        out: list[tuple[str, Any, str, float]] = []
        out += [("set", k, "word", 0.14 / (len(cat.STAND_SET) - 1)) for k in cat.STAND_SET if k != state["set"]]
        out += [("packing", k, "word", 0.14 / (len(cat.STAND_PACKING) - 1)) for k in cat.STAND_PACKING if k != state["packing"]]
        out += [("stand", k, "word", 0.14 / (len(cat.STAND_MODEL) - 1)) for k in cat.STAND_MODEL if k != state["stand"]]
        out += [("method", k, "word", 0.08 / (len(cat.STAND_METHOD) - 1)) for k in cat.STAND_METHOD if k != state["method"]]
        reach = state["reach_mm"]
        for delta in (-80, -50, -30, 30, 50, 80):
            if 100 <= reach + delta <= 320:
                out.append(("reach_mm", int(reach + delta), "number", 0.16 / 6))
        out += [("release_mm", h, "number", 0.12 / (len(STAND_RELEASES) - 1)) for h in STAND_RELEASES if h != state["release_mm"]]
        for delta in (-200, -120, 120, 200):
            if 850 <= state["height_mm"] + delta <= 1250:
                out.append(("height_mm", int(state["height_mm"] + delta), "number", 0.08 / 4))
        lo, hi = self.decisive_bounds(state)
        for delta in (-40, -25, 25, 40):
            new = float(state["setdown_mm"] + delta)
            if lo <= new <= hi:
                out.append(("setdown_mm", new, "number", 0.14 / 4))
        return out


# ==========================================================================
# Law C — hold-down clamp (light manufacturing / biomedical engineering workshop)
# ==========================================================================

#: Parts: key → (words, length, width, height ranges (mm), mass range (kg)).
CLAMP_PARTS = {
    "pump_housing": ("infusion pump housing", (110, 170), (80, 120), (40, 70), (0.6, 1.6)),
    "valve_manifold": ("valve manifold", (90, 150), (60, 100), (35, 60), (0.8, 2.5)),
    "luer_block": ("luer manifold block", (60, 100), (40, 70), (25, 45), (0.15, 0.5)),
    "end_bell": ("motor end bell", (80, 120), (80, 120), (30, 50), (0.4, 1.2)),
    "carrier_plate": ("sensor carrier plate", (120, 200), (80, 140), (12, 25), (0.3, 1.0)),
    "hinge_bracket": ("hinge bracket", (80, 140), (40, 80), (20, 40), (0.2, 0.8)),
    "gear_half": ("gear housing half", (120, 180), (90, 140), (45, 80), (1.5, 4.0)),
}
CLAMP_TIMES = (0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6)
CLAMP_SITES = ("biomedical engineering workshop", "device assembly cell", "light-assembly cobot cell", "prototype lab bench cell")


class HoldDown(Scene):
    name = "holddown"
    law = "C"
    phenomena = ("friction_limited_holding",)
    domain = "light_manufacturing"
    outcomes = sim.OUTCOMES["holddown"]
    key_events = ("creeps", "breaks_free")
    severe = "breaks_free"
    costs = {"stays_put": 0.0, "creeps": 1.0, "breaks_free": 2.2}
    decisive = "press_n"
    risk_up = True
    step = 5.0
    hidden_names = ("mu_nest", "mu_pad", "force_factor")
    #: Measured 2026-10-14 (40 draws): the press peak at MuJoCo's ≥ 1 mm flip is 1.009 × the static capacity (p10–p90 1.004–1.036).
    calib = {"press": 1.01}

    def sample_state(self, rng: np.random.Generator) -> dict[str, Any]:
        part = str(rng.choice(sorted(CLAMP_PARTS)))
        _, lr, wr, hr, mr = CLAMP_PARTS[part]
        length, width = sorted((float(rng.uniform(*lr)), float(rng.uniform(*wr))), reverse=True)
        return {
            "part": part,
            "dims_mm": [int(_round(length, 5)), int(_round(width, 5)), int(_round(rng.uniform(*hr), 1))],
            "mass_kg": _round(rng.uniform(*mr), 0.05),
            "nest": str(rng.choice(sorted(cat.CLAMP_NEST))),
            "pad": str(rng.choice(sorted(cat.CLAMP_PAD))),
            "condition": str(rng.choice(sorted(cat.CLAMP_CONDITION))),
            "clamp_n": int(_round(rng.uniform(60, 400), 10)),
            "press_s": float(rng.choice(CLAMP_TIMES)),
            "press_n": 100.0,
        }

    def hidden_specs(self, state: Mapping[str, Any]) -> dict[str, list]:
        nest, pad, cond = cat.CLAMP_NEST[state["nest"]], cat.CLAMP_PAD[state["pad"]], cat.CLAMP_CONDITION[state["condition"]]
        return {"mu_nest": ["u", nest.lo, nest.hi], "mu_pad": ["u", pad.lo, pad.hi], "force_factor": ["u", cond.lo, cond.hi]}

    def decisive_bounds(self, state: Mapping[str, Any]) -> tuple[float, float]:
        return (5.0, 2500.0)

    def proxy_events(self, state: Mapping[str, Any], values: Mapping[str, np.ndarray]) -> np.ndarray:
        force = values["force_factor"] * state["clamp_n"]
        down = force + sim.CLAMP_MASS * G
        capacity = values["mu_nest"] * (state["mass_kg"] * G + down) + values["mu_pad"] * down
        return state["press_n"] > capacity * self.calib["press"]

    def simulate(self, state: Mapping[str, Any], values: Mapping[str, float]) -> dict[str, Any]:
        model = _clamp_model(tuple(v / 1000 for v in state["dims_mm"]), float(state["mass_kg"]))
        return sim.run_clamp(model, press_force=float(state["press_n"]), press_time=float(state["press_s"]),
                             clamp_force=float(values["force_factor"]) * float(state["clamp_n"]), mu_nest=float(values["mu_nest"]),
                             mu_pad=float(values["mu_pad"]))

    def alternatives(self) -> list[Alternative]:
        return [
            Alternative("raise_clamp", "raise the clamp setting by half", False,
                        lambda st: {"clamp_n": int(_round(st["clamp_n"] * 1.5, 10))}, lambda st: st["clamp_n"] * 1.5 <= 650),
            Alternative("lube_fitting", "grease the fitting (peak about 30 % lower)", False,
                        lambda st: {"press_n": _round(st["press_n"] * 0.7, 5)}, lambda st: st["press_n"] * 0.7 >= 5),
            Alternative("wipe_nest", "wipe the nest clean and dry", False,
                        lambda st: {"nest": "steel_dry"}, lambda st: st["nest"] in ("steel_oil", "coolant")),
            Alternative("pu_pad", "swap to a polyurethane pad", False,
                        lambda st: {"pad": "pu"}, lambda st: st["pad"] not in ("pu", "serrated")),
            Alternative("soft_clamp", "cut the clamp setting by 40 % to spare the finish", True,
                        lambda st: {"clamp_n": int(_round(st["clamp_n"] * 0.6, 10))}, lambda st: st["clamp_n"] * 0.6 >= 30),
            Alternative("quick_stroke", "quick stroke: half the time, peak +20 %", True,
                        lambda st: {"press_n": _round(st["press_n"] * 1.2, 5), "press_s": _round(st["press_s"] * 0.5, 0.05)},
                        lambda st: st["press_s"] * 0.5 >= 0.075),
        ]

    def one_factor_changes(self, state: Mapping[str, Any], rng: np.random.Generator) -> list[tuple[str, Any, str, float]]:
        out: list[tuple[str, Any, str, float]] = []
        out += [("nest", k, "word", 0.20 / (len(cat.CLAMP_NEST) - 1)) for k in cat.CLAMP_NEST if k != state["nest"]]
        out += [("pad", k, "word", 0.17 / (len(cat.CLAMP_PAD) - 1)) for k in cat.CLAMP_PAD if k != state["pad"]]
        out += [("condition", k, "word", 0.13 / (len(cat.CLAMP_CONDITION) - 1)) for k in cat.CLAMP_CONDITION if k != state["condition"]]
        for factor in (0.6, 0.75, 1.3, 1.6):
            new = int(_round(state["clamp_n"] * factor, 10))
            if 30 <= new <= 650 and new != state["clamp_n"]:
                out.append(("clamp_n", new, "number", 0.20 / 4))
        out += [("press_s", t, "number", 0.08 / (len(CLAMP_TIMES) - 1)) for t in CLAMP_TIMES if t != state["press_s"]]
        for factor in (0.5, 0.7, 1.4, 2.0):
            new = _round(state["mass_kg"] * factor, 0.05)
            if new >= 0.1 and new != state["mass_kg"]:
                out.append(("mass_kg", new, "number", 0.08 / 4))
        for factor in (0.85, 0.9, 1.1, 1.15):
            new = _round(state["press_n"] * factor, self.step)
            if new != state["press_n"] and new >= 5:
                out.append(("press_n", new, "number", 0.14 / 4))
        return out


_CLAMP_CACHE: dict[tuple, Any] = {}


def _clamp_model(dims: Sequence[float], mass: float) -> Any:
    key = (tuple(round(v, 6) for v in dims), round(mass, 6))
    if key not in _CLAMP_CACHE:
        if len(_CLAMP_CACHE) > 64:
            _CLAMP_CACHE.clear()
        _CLAMP_CACHE[key] = sim.ClampModel.build(dims, mass)
    return _CLAMP_CACHE[key]


SCENES: dict[str, Scene] = {s.name: s for s in (IndexDial(), MayoStand(), HoldDown())}


def scene(name: str) -> Scene:
    if name not in SCENES:
        raise ValueError(f"scene: one of {sorted(SCENES)} (got {name!r})")
    return SCENES[name]
