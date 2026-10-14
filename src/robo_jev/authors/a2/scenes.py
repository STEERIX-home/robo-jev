"""Author A2 — the three scene specifications: what a briefing states (``facts``), which hidden quantities its words pick and their
ranges under each prior setting, the physical setup of a variant, the static aim model, the alternatives and the one-factor
partner changes. Physics lives in :mod:`robo_jev.authors.a2.physics`, catalogs in :mod:`robo_jev.authors.a2.catalog`.

A **variant** is the facts with a few fields changed (``planned`` = none; ``cf`` = the decisive number at its counterfactual
value; ``opt_*`` = one alternative action). Hidden draws are shared: K uniform numbers (or standard normals for placement) per hidden
quantity and group, mapped into each prior's range, the same for every variant and both members of a pair.
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np

from robo_jev.authors.a2 import catalog as C
from robo_jev.authors.a2 import physics as PH

__all__ = ["SCENES", "Scene", "Turntable", "Tiltback", "Wallpress", "hidden_values"]

GRAVITY = PH.GRAVITY


def _u(rng: np.random.Generator, lo: float, hi: float) -> float:
    return float(lo + (hi - lo) * rng.random())


def hidden_values(spec: "Scene", ranges: Mapping[str, tuple[float, float]], draws: Mapping[str, np.ndarray], prior: str) -> dict[str, np.ndarray]:
    """Hidden values under one prior: ``u`` draws mapped into the prior's range, ``place`` draws (standard normals) times its σ."""
    out = {}
    for name, kind in spec.hidden:
        lo, hi = C.prior_range(kind, *ranges[name], prior)
        out[name] = draws[name] * hi if kind == "place" else lo + draws[name] * (hi - lo)
    return out


class Scene:
    """Common interface (see the subclasses)."""

    name: str
    law: str
    outcomes: tuple[str, ...]
    key: tuple[str, ...]
    severe: tuple[str, ...]
    decisive: str
    hidden: tuple[tuple[str, str], ...]
    #: Fields that decide the physics (the cache key of a setup).
    physical_fields: tuple[str, ...]
    plan_key: str
    fallback: str
    costs: Mapping[str, float]
    action_costs: Mapping[str, float]

    def draws(self, rng: np.random.Generator, k: int, extra: int = 0) -> dict[str, np.ndarray]:
        out = {}
        for name, kind in self.hidden:
            out[name] = rng.standard_normal(k + extra) if kind == "place" else rng.random(k + extra)
        return out

    def setup_key(self, facts: Mapping[str, Any]) -> str:
        import json

        return json.dumps({f: facts[f] for f in self.physical_fields if f in facts}, sort_keys=True)

    def hidden_prior(self, facts: Mapping[str, Any]) -> dict[str, dict[str, list[Any]]]:
        """``meta.hidden_prior``: per prior, per hidden quantity ``["u", lo, hi]`` or ``["n", 0, σ]``."""
        ranges = self.ranges(facts)
        out: dict[str, dict[str, list[Any]]] = {}
        for prior in C.PRIORS:
            row = {}
            for name, kind in self.hidden:
                lo, hi = C.prior_range(kind, *ranges[name], prior)
                row[name] = ["n", 0.0, round(hi, 6)] if kind == "place" else ["u", round(lo, 6), round(hi, 6)]
            out[prior] = row
        return out

    # subclasses: sample, ranges, run, static_key_draws, solve, round_decisive, alternatives, apply_alternative, one_factor_options,
    # change_one_factor, decisive_bounds


# --------------------------------------------------------------------------
# A — turntable
# --------------------------------------------------------------------------


class Turntable(Scene):
    name = "turntable"
    law = "A"
    outcomes = C.TT_OUTCOMES
    key = C.TT_KEY
    severe = C.TT_SEVERE
    decisive = "duration"
    hidden = (("mu", "friction"), ("kh", "spread"), ("dr", "place"))
    physical_fields = ("item", "side_cm", "ratio", "height_cm", "orient", "contents", "base", "top", "condition", "radius_cm", "turn",
                       "duration")
    plan_key = "turn_as_planned"
    fallback = "carry_it_round"
    costs = C.TT_COSTS
    action_costs = C.TT_ACTION_COSTS
    #: Static aim: the dynamic thresholds sit this much above the static μg (a 2 cm skid) and g·b/h (a full topple) — fitted on a
    #: 48-group pilot (95.7 % of planned rollouts called right; κ = 1 gave 87.2 %).
    kappa_slide = 1.10
    kappa_tip = 1.15

    def sample(self, rng: np.random.Generator, flavour: np.random.Generator) -> dict[str, Any]:
        domain = str(flavour.choice(sorted(C.TT_VENUES)))
        item = str(rng.choice(sorted(C.TT_ITEMS)))
        spec = C.TT_ITEMS[item]
        side = round(_u(rng, *spec["side"]) * 2) / 2
        ratio = round(_u(rng, *spec["ratio"]), 2)
        height = round(_u(rng, *spec["height"]) * 2) / 2
        facts = {
            "domain": domain,
            "venue": str(flavour.choice(C.TT_VENUES[domain])),
            "item": item,
            "side_cm": side,
            "ratio": ratio,
            "height_cm": height,
            "orient": "square" if ratio >= 0.99 else str(rng.choice(["radial", "along"])),
            "contents": str(rng.choice(sorted(C.TT_CONTENTS[spec["family"]]))),
            "base": str(rng.choice(spec["bases"])),
            "top": str(rng.choice(sorted(C.TT_TOPS))),
            "condition": str(rng.choice(sorted(C.TT_CONDITIONS))),
            "radius_cm": int(rng.integers(C.TT_RADIUS_CM[0], C.TT_RADIUS_CM[1] + 1)),
            "turn": str(rng.choice(sorted(C.TT_TURNS))),
            "spin_dir": str(flavour.choice(["clockwise", "anticlockwise"])),
            "seat": int(flavour.integers(2, 11)),
        }
        return facts

    def dims(self, facts: Mapping[str, Any]) -> tuple[float, float, float]:
        """(radial, tangential, height) in m."""
        long_side, short_side = facts["side_cm"] / 100, facts["side_cm"] * facts["ratio"] / 100
        if facts["orient"] == "along":
            return (short_side, long_side, facts["height_cm"] / 100)
        return (long_side, short_side, facts["height_cm"] / 100)

    def mass(self, facts: Mapping[str, Any]) -> float:
        spec = C.TT_ITEMS[facts["item"]]
        a, b, h = self.dims(facts)
        fill = C.TT_CONTENTS[spec["family"]][facts["contents"]][2]
        return round(spec["shell"] + fill * a * b * h * 0.75 * 1000.0, 4)

    def ranges(self, facts: Mapping[str, Any]) -> dict[str, tuple[float, float]]:
        spec = C.TT_ITEMS[facts["item"]]
        lo, hi = C.TT_BASES[facts["base"]][1]
        f = C.TT_TOPS[facts["top"]][1] * C.TT_CONDITIONS[facts["condition"]][1]
        return {"mu": (lo * f, hi * f), "kh": C.TT_CONTENTS[spec["family"]][facts["contents"]][1], "dr": (0.0, C.TT_PLACE_SIGMA)}

    def run(self, scene_cache: dict, facts: Mapping[str, Any], values: Mapping[str, float]) -> dict[str, Any]:
        dims = self.dims(facts)
        key = ("tt", dims, self.mass(facts))
        if key not in scene_cache:
            scene_cache[key] = PH.TurntableScene(dims, self.mass(facts))
        sc = scene_cache[key]
        radius = max(0.03, facts["radius_cm"] / 100 + float(values["dr"]))
        kh = min(0.92, max(0.08, float(values["kh"])))
        return sc.run(angle=math.radians(self.turn_deg(facts)), duration=float(facts["duration"]), radius=radius,
                      mu=max(0.02, float(values["mu"])), com=(0.0, 0.0, (kh - 0.5) * dims[2]))

    @staticmethod
    def turn_deg(facts: Mapping[str, Any]) -> float:
        return {**C.TT_TURNS, **C.TT_LONG_TURNS}[facts["turn"]][0]

    def _crit(self, facts: Mapping[str, Any], values: Mapping[str, np.ndarray]) -> np.ndarray:
        """Per draw: the turn time below which the static model calls the key event (s): peak acceleration above κ_slide·μg or
        κ_tip·g·b/h (b = half the narrower base side, h = centre-of-mass height)."""
        a, b, h = self.dims(facts)
        c = PH.turntable_peak_accel(math.radians(self.turn_deg(facts)), 1.0, 1.0)
        radius = np.maximum(0.03, facts["radius_cm"] / 100 + values["dr"])
        kh = np.clip(values["kh"], 0.08, 0.92)
        limit = np.minimum(self.kappa_slide * np.maximum(values["mu"], 0.02), self.kappa_tip * (min(a, b) / 2) / (kh * h))
        return np.sqrt(c * radius / (GRAVITY * limit))

    def static_key(self, facts: Mapping[str, Any], values: Mapping[str, np.ndarray]) -> float:
        return float(np.mean(self._crit(facts, values) > float(facts["duration"])))

    def solve(self, facts: Mapping[str, Any], values: Mapping[str, np.ndarray], target: float) -> float:
        return float(np.quantile(self._crit(facts, values), 1.0 - target))

    def round_decisive(self, value: float) -> float:
        return round(min(max(value, C.TT_DURATION[0]), C.TT_DURATION[1]), 1)

    def decisive_ok(self, value: float) -> bool:
        return C.TT_DURATION[0] <= value <= C.TT_DURATION[1]

    def risk_up(self, value: float, factor: float) -> float:
        """The decisive value moved by `factor` (> 1) towards more risk (a quicker turn)."""
        return value / factor

    def risk_down(self, value: float, factor: float) -> float:
        return value * factor

    def alternatives(self, facts: Mapping[str, Any]) -> list[str]:
        out = ["turn_slower", "go_the_other_way"]
        if facts["radius_cm"] >= 15:
            out.append("move_it_inward")
        return out

    def apply_alternative(self, facts: Mapping[str, Any], alt: str) -> dict[str, Any]:
        if alt == "turn_slower":
            return {"duration": round(float(facts["duration"]) * C.TT_SLOWER, 1)}
        if alt == "move_it_inward":
            return {"radius_cm": C.TT_INWARD_CM}
        if alt == "go_the_other_way":  # the same seat the long way round, in the same time
            return {"turn": C.TT_OTHER_WAY[facts["turn"]], "spin_dir": "anticlockwise" if facts["spin_dir"] == "clockwise" else "clockwise"}
        raise ValueError(alt)

    def one_factor_options(self, facts: Mapping[str, Any]) -> list[tuple[str, str, str, float]]:
        """(field, fact kind, hidden factor, weight)."""
        return [("base", "word", "mu", 0.14), ("condition", "word", "mu", 0.14), ("top", "word", "mu", 0.07), ("contents", "word", "kh", 0.30),
                ("radius_cm", "number", "geometry", 0.20), ("duration", "number", "decisive", 0.15)]

    def change_one_factor(self, facts: Mapping[str, Any], field: str, rng: np.random.Generator) -> Any:
        spec = C.TT_ITEMS[facts["item"]]
        if field == "base":
            return str(rng.choice([b for b in spec["bases"] if b != facts["base"]]))
        if field == "condition":
            return str(rng.choice([c for c in sorted(C.TT_CONDITIONS) if c != facts["condition"]]))
        if field == "top":
            return str(rng.choice([t for t in sorted(C.TT_TOPS) if t != facts["top"]]))
        if field == "contents":
            return str(rng.choice([c for c in sorted(C.TT_CONTENTS[spec["family"]]) if c != facts["contents"]]))
        if field == "radius_cm":
            r = facts["radius_cm"]
            choices = [v for v in range(C.TT_RADIUS_CM[0], C.TT_RADIUS_CM[1] + 1) if abs(v - r) >= max(5, 0.3 * r)]
            return int(rng.choice(choices))
        raise ValueError(field)


# --------------------------------------------------------------------------
# B — tiltback
# --------------------------------------------------------------------------


class Tiltback(Scene):
    name = "tiltback"
    law = "B"
    outcomes = C.TB_OUTCOMES
    key = C.TB_KEY
    severe = C.TB_SEVERE
    decisive = "tilt_deg"
    hidden = (("mu", "friction"), ("kh", "spread"), ("cx", "spread"), ("mass", "spread"), ("dtilt", "place"))
    physical_fields = ("piece", "build", "depth_cm", "width_cm", "height_cm", "contents", "floor", "limit_n", "axis", "tilt_deg")
    plan_key = "tip_as_planned"
    fallback = "use_the_lifter"
    costs = C.TB_COSTS
    action_costs = C.TB_ACTION_COSTS

    def sample(self, rng: np.random.Generator, flavour: np.random.Generator) -> dict[str, Any]:
        domain = str(flavour.choice(sorted(C.TB_VENUES)))
        piece = str(rng.choice(sorted(C.TB_PIECES)))
        spec = C.TB_PIECES[piece]
        return {
            "domain": domain,
            "venue": str(flavour.choice(C.TB_VENUES[domain])),
            "piece": piece,
            "build": str(rng.choice(spec["builds"])),
            "depth_cm": int(round(_u(rng, *spec["depth"]) * 100)),
            "width_cm": int(round(_u(rng, *spec["width"]) * 100)),
            "height_cm": int(round(_u(rng, *spec["height"]) * 20)) * 5,
            "contents": str(rng.choice(sorted(C.TB_CONTENTS))),
            "floor": str(rng.choice(sorted(C.TB_FLOORS))),
            "limit_n": float(rng.choice(C.TB_LIMITS)),
            "axis": "back",
            "task": str(flavour.choice(C.TB_TASKS)),
        }

    def dims(self, facts: Mapping[str, Any]) -> tuple[float, float, float]:
        """(depth along the tipping direction, width, height) in m — swapped when tipping onto a side edge."""
        d, w, h = facts["depth_cm"] / 100, facts["width_cm"] / 100, facts["height_cm"] / 100
        return (w, d, h) if facts.get("axis") == "side" else (d, w, h)

    def ranges(self, facts: Mapping[str, Any]) -> dict[str, tuple[float, float]]:
        spec = C.TB_PIECES[facts["piece"]]
        table = C.TB_FRIDGE_CONTENTS if facts["piece"] == "fridge" else C.TB_CONTENTS
        kh, add = table[facts["contents"]]
        volume = facts["depth_cm"] * facts["width_cm"] * facts["height_cm"] / 1e6
        blo, bhi = C.TB_BUILDS[facts["build"]][1]
        cx = (-0.05, 0.05) if facts.get("axis") == "side" else spec["cx"]
        return {"mu": C.TB_FLOORS[facts["floor"]][1], "kh": kh, "cx": cx, "mass": (volume * (blo + add[0]), volume * (bhi + add[1])),
                "dtilt": (0.0, C.TB_PLACE_SIGMA)}

    def run(self, scene_cache: dict, facts: Mapping[str, Any], values: Mapping[str, float]) -> dict[str, Any]:
        dims = self.dims(facts)
        key = ("tb", dims)
        if key not in scene_cache:
            scene_cache[key] = PH.TiltbackScene(dims, 50.0)
        sc = scene_cache[key]
        kh = min(0.92, max(0.08, float(values["kh"])))
        return sc.run(tilt_deg=max(0.2, float(facts["tilt_deg"]) + float(values["dtilt"])), limit=float(facts["limit_n"]),
                      mu=max(0.02, float(values["mu"])), com=(float(values["cx"]) * dims[0], 0.0, (kh - 0.5) * dims[2]),
                      mass=max(2.0, float(values["mass"])))

    def _crit(self, facts: Mapping[str, Any], values: Mapping[str, np.ndarray]) -> np.ndarray:
        """Per draw: the planned tilt above which the centre of mass is behind the pivot edge (degrees)."""
        d, _, h = self.dims(facts)
        kh = np.clip(values["kh"], 0.08, 0.92)
        return np.degrees(np.arctan2(d / 2 + values["cx"] * d, kh * h)) - values["dtilt"]

    def static_key(self, facts: Mapping[str, Any], values: Mapping[str, np.ndarray]) -> float:
        return float(np.mean(self._crit(facts, values) < float(facts["tilt_deg"])))

    def solve(self, facts: Mapping[str, Any], values: Mapping[str, np.ndarray], target: float) -> float:
        return float(np.quantile(self._crit(facts, values), target))

    def round_decisive(self, value: float) -> float:
        return round(min(max(value, C.TB_TILT[0]), C.TB_TILT[1]) * 2) / 2

    def decisive_ok(self, value: float) -> bool:
        return C.TB_TILT[0] <= value <= C.TB_TILT[1]

    def risk_up(self, value: float, factor: float) -> float:
        return value * factor

    def risk_down(self, value: float, factor: float) -> float:
        return value / factor

    def alternatives(self, facts: Mapping[str, Any]) -> list[str]:
        out = ["tip_it_sideways"]
        if facts["contents"] in ("top", "even"):
            out.append("empty_top_first")
        if facts["contents"] in ("bottom", "empty", "even"):
            out.append("load_top_first")
        return out

    def apply_alternative(self, facts: Mapping[str, Any], alt: str) -> dict[str, Any]:
        if alt == "tip_it_sideways":
            return {"axis": "side"}
        if alt == "empty_top_first":
            return {"contents": "empty"}
        if alt == "load_top_first":  # put the boxes from the floor up top before tipping (saves a trip)
            return {"contents": "top"}
        raise ValueError(alt)

    def one_factor_options(self, facts: Mapping[str, Any]) -> list[tuple[str, str, str, float]]:
        return [("floor", "word", "mu", 0.10), ("contents", "word", "kh", 0.30), ("build", "word", "mass", 0.08),
                ("depth_cm", "number", "geometry", 0.22), ("limit_n", "number", "severe", 0.15), ("tilt_deg", "number", "decisive", 0.15)]

    def change_one_factor(self, facts: Mapping[str, Any], field: str, rng: np.random.Generator) -> Any:
        spec = C.TB_PIECES[facts["piece"]]
        if field == "floor":
            return str(rng.choice([f for f in sorted(C.TB_FLOORS) if f != facts["floor"]]))
        if field == "contents":
            return str(rng.choice([c for c in sorted(C.TB_CONTENTS) if c != facts["contents"]]))
        if field == "build":
            others = [b for b in spec["builds"] if b != facts["build"]]
            if not others:
                raise LookupError("no other build")
            return str(rng.choice(others))
        if field == "depth_cm":
            d = facts["depth_cm"]
            lo, hi = int(round(spec["depth"][0] * 100)) - 6, int(round(spec["depth"][1] * 100)) + 8
            choices = [v for v in range(lo, hi + 1) if abs(v - d) >= max(6, 0.2 * d)]
            if not choices:
                raise LookupError("no depth far enough")
            return int(rng.choice(choices))
        if field == "limit_n":
            return float(rng.choice([v for v in C.TB_LIMITS if v != facts["limit_n"]]))
        raise ValueError(field)


# --------------------------------------------------------------------------
# C — wallpress
# --------------------------------------------------------------------------


class Wallpress(Scene):
    name = "wallpress"
    law = "C"
    outcomes = C.WP_OUTCOMES
    key = C.WP_KEY
    severe = C.WP_SEVERE
    decisive = "force_n"
    hidden = (("mu", "friction"), ("mu_pad", "friction"), ("mass", "spread"), ("dy", "place"), ("dz", "place"))
    physical_fields = ("item", "thick_cm", "width_cm", "height_cm", "build", "back", "wall", "pad", "pads", "reflex", "force_n")
    plan_key = "hold_as_planned"
    fallback = "batten_first"
    costs = C.WP_COSTS
    action_costs = C.WP_ACTION_COSTS
    kappa = 1.0

    def sample(self, rng: np.random.Generator, flavour: np.random.Generator) -> dict[str, Any]:
        domain = str(flavour.choice(sorted(C.WP_VENUES)))
        item = str(rng.choice(sorted(C.WP_ITEMS)))
        spec = C.WP_ITEMS[item]
        thick = _u(rng, *spec["thick"]) * 100
        return {
            "domain": domain,
            "venue": str(flavour.choice(C.WP_VENUES[domain])),
            "item": item,
            "thick_cm": round(thick, 1) if thick < 3 else float(int(round(thick))),
            "width_cm": int(round(_u(rng, *spec["width"]) * 20)) * 5,
            "height_cm": int(round(_u(rng, *spec["height"]) * 20)) * 5,
            "build": str(rng.choice(spec["builds"])),
            "back": str(rng.choice(spec["backs"])),
            "wall": str(rng.choice(sorted(C.WP_WALLS))),
            "pad": str(rng.choice(sorted(C.WP_PADS))),
            "pads": 1,
            "reflex": float(rng.choice(C.WP_REFLEX)),
            "helper": str(flavour.choice(C.WP_HELPERS)),
        }

    def dims(self, facts: Mapping[str, Any]) -> tuple[float, float, float]:
        return (facts["thick_cm"] / 100, facts["width_cm"] / 100, facts["height_cm"] / 100)

    def ranges(self, facts: Mapping[str, Any]) -> dict[str, tuple[float, float]]:
        area = facts["width_cm"] * facts["height_cm"] / 1e4
        mlo, mhi = C.WP_BUILDS[facts["build"]][1]
        back = C.WP_RUBBER_SHEET if facts["back"] == "rubbersheet" else C.WP_BACKS[facts["back"]][1]
        f = C.WP_WALLS[facts["wall"]][1]
        return {"mu": (back[0] * f, back[1] * f), "mu_pad": C.WP_PADS[facts["pad"]][1], "mass": (area * mlo, area * mhi),
                "dy": (0.0, C.WP_PLACE_SIGMA), "dz": (0.0, C.WP_PLACE_SIGMA)}

    def run(self, scene_cache: dict, facts: Mapping[str, Any], values: Mapping[str, float]) -> dict[str, Any]:
        dims = self.dims(facts)
        key = ("wp", dims)
        if key not in scene_cache:
            scene_cache[key] = PH.WallpressScene(dims, 5.0, 0.5)
        sc = scene_cache[key]
        return sc.run(force=float(facts["force_n"]) * int(facts.get("pads", 1)), reflex=float(facts["reflex"]),
                      mu=max(0.02, float(values["mu"])), offset=(float(values["dy"]), float(values["dz"])),
                      mass=max(0.2, float(values["mass"])), pad_mu=max(0.02, float(values["mu_pad"])))

    def _crit(self, facts: Mapping[str, Any], values: Mapping[str, np.ndarray]) -> np.ndarray:
        """Per draw: the total push (N) below which the item slips (static: the weight against both faces' friction)."""
        mass = np.maximum(0.2, values["mass"])
        return self.kappa * mass * GRAVITY / (np.maximum(0.02, values["mu"]) + np.maximum(0.02, values["mu_pad"]))

    def static_key(self, facts: Mapping[str, Any], values: Mapping[str, np.ndarray]) -> float:
        return float(np.mean(self._crit(facts, values) > float(facts["force_n"]) * int(facts.get("pads", 1))))

    def solve(self, facts: Mapping[str, Any], values: Mapping[str, np.ndarray], target: float) -> float:
        return float(np.quantile(self._crit(facts, values), 1.0 - target)) / int(facts.get("pads", 1))

    def round_decisive(self, value: float) -> float:
        value = min(max(value, C.WP_FORCE[0]), C.WP_FORCE[1])
        return float(round(value)) if value < 30 else float(int(round(value / 5)) * 5)

    def decisive_ok(self, value: float) -> bool:
        return C.WP_FORCE[0] <= value <= C.WP_FORCE[1]

    def risk_up(self, value: float, factor: float) -> float:
        return value / factor

    def risk_down(self, value: float, factor: float) -> float:
        return value * factor

    def alternatives(self, facts: Mapping[str, Any]) -> list[str]:
        out = ["use_both_pads"]
        if facts["back"] != "rubber":
            out.append("rubber_sheet_behind")
        if facts["pad"] != "feltpad":
            out.append("felt_pad_instead")
        return out

    def apply_alternative(self, facts: Mapping[str, Any], alt: str) -> dict[str, Any]:
        if alt == "use_both_pads":
            return {"pads": 2}
        if alt == "rubber_sheet_behind":
            return {"back": "rubbersheet"}
        if alt == "felt_pad_instead":  # kinder to the glass and paint, same push
            return {"pad": "feltpad"}
        raise ValueError(alt)

    def one_factor_options(self, facts: Mapping[str, Any]) -> list[tuple[str, str, str, float]]:
        return [("back", "word", "mu", 0.13), ("wall", "word", "mu", 0.13), ("pad", "word", "mu_pad", 0.09), ("build", "word", "mass", 0.25),
                ("height_cm", "number", "geometry", 0.15), ("reflex", "number", "severe", 0.10), ("force_n", "number", "decisive", 0.15)]

    def change_one_factor(self, facts: Mapping[str, Any], field: str, rng: np.random.Generator) -> Any:
        spec = C.WP_ITEMS[facts["item"]]
        if field == "back":
            return str(rng.choice([b for b in spec["backs"] if b != facts["back"]]))
        if field == "wall":
            return str(rng.choice([w for w in sorted(C.WP_WALLS) if w != facts["wall"]]))
        if field == "pad":
            return str(rng.choice([p for p in sorted(C.WP_PADS) if p != facts["pad"]]))
        if field == "build":
            return str(rng.choice([b for b in spec["builds"] if b != facts["build"]]))
        if field == "height_cm":
            h = facts["height_cm"]
            lo, hi = int(round(spec["height"][0] * 100)), int(round(spec["height"][1] * 100)) + 20
            choices = [v for v in range(lo - lo % 5, hi + 1, 5) if abs(v - h) >= max(15, 0.3 * h) and v >= 35]
            if not choices:
                raise LookupError("no height far enough")
            return int(rng.choice(choices))
        if field == "reflex":
            return float(rng.choice([v for v in C.WP_REFLEX if v != facts["reflex"]]))
        raise ValueError(field)


SCENES: dict[str, Scene] = {"turntable": Turntable(), "tiltback": Tiltback(), "wallpress": Wallpress()}
