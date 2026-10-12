"""T9 families — catalogs (word → range priors), visible scenes, aiming, one-fact changes and the textbook calculators (Task T9 §1-1).

A *moment* is a visible configuration ``vis`` (a dict of stated numbers and catalog words). Its questions ask about **variants** of it,
each a copy of ``vis`` with one entry changed:

* ``plan`` — the moment as stated (the planned action);
* ``cf`` — one stated number changed (asked as a 0–1 rating: "how likely … if <number> were <value>");
* ``alt`` — an alternative action (a gentler version of the plan, with a stated time cost);
* ``dirv`` — one number or word changed for the more / less / about-the-same question.

Every family declares its hidden quantities (word ranges, :meth:`Family.words`; stated tolerances, :meth:`Family.fixed`), the
physics of a variant (:meth:`Family.scene` + :meth:`Family.action` → :mod:`robo_jev.t9.physics`) and a **textbook calculator**
(:meth:`Family.calc`): closed-form statics or a small ODE on the same hidden draws, used to aim the decisive number (the base-prior key
rate hits a target drawn uniformly) and, later, as the analysis reference. The calculator never sees MuJoCo.

Laws and groups (relative to the training data as of v0.3 — the engine plus v0.3b's tilt, grip and tower):

========== ============================================================================== =======================
family     law                                                                            group
========== ============================================================================== =======================
roll       rolling (no static-friction threshold; lip height against radius and speed)   unseen_law
swing      pendulum (swing ≈ v·√(L/g) after a stop; mass barely matters)                  unseen_law
knock      impulse at a height (rotate about the far edge vs slide; energy to tip)       unseen_law
conveyor   tip or slide under acceleration                                                seen_law_new_scene
overhang   centre of mass past an edge                                                    seen_law_new_scene
suction    friction-limited holding                                                       seen_law_new_mechanism
========== ============================================================================== =======================
"""

from __future__ import annotations

import copy
import math
import random
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

import numpy as np

from robo_jev.t9 import physics as ph

__all__ = ["FAMILIES", "Change", "Family", "family"]

G = ph.G


@dataclass(frozen=True)
class Change:
    """One fact of a moment changed: ``field`` of ``vis`` set to ``value`` (a number or a catalog word)."""

    field: str
    value: Any
    kind: str  # "num" | "word"

    def apply(self, vis: dict[str, Any]) -> dict[str, Any]:
        out = copy.deepcopy(vis)
        out[self.field] = self.value
        return out


@dataclass
class Factor:
    """A number or word of a moment that a one-fact change may touch. ``sign``: +1 if raising it (or the listed word order) makes
    the key event likelier, −1 if rarer, 0 if it should not matter (an "about the same" fact); ``lo``/``hi`` bound numbers;
    ``scale`` = True draws the new value as a ratio of the old one."""

    field: str
    kind: str
    sign: int
    lo: float = 0.0
    hi: float = 0.0
    scale: bool = False
    digits: int = 3
    same_ratio: tuple[float, ...] = ()


class Family:
    """Base class: catalogs and hooks of one family (module docstring)."""

    name = ""
    law = ""
    group = ""
    key: tuple[str, ...] = ()  # outcomes that count as the key event
    decisive = ""  # the number aimed at a target key rate
    decisive_range = (0.0, 1.0)
    decisive_sign = 1
    cf_factors: tuple[Factor, ...] = ()
    partner_factors: tuple[Factor, ...] = ()
    dir_factors: tuple[Factor, ...] = ()
    alt_field = ""  # the number the alternative action changes
    alt_ratio = (0.3, 0.6)

    # ---- hooks -------------------------------------------------------------------------------------------------------------
    def sample(self, rng: random.Random) -> dict[str, Any]:  # pragma: no cover - abstract
        raise NotImplementedError

    def words(self, vis: dict[str, Any]) -> dict[str, tuple[float, float]]:  # pragma: no cover
        raise NotImplementedError

    def fixed(self, vis: dict[str, Any]) -> dict[str, tuple[float, float]]:
        return {}

    def scene(self, vis: dict[str, Any]) -> dict[str, Any]:  # pragma: no cover
        raise NotImplementedError

    def action(self, vis: dict[str, Any]) -> dict[str, Any]:  # pragma: no cover
        raise NotImplementedError

    def physics_draw(self, vis: dict[str, Any], draw: dict[str, float]) -> dict[str, float]:
        return dict(draw)

    def calc(self, vis: dict[str, Any], draws: Sequence[dict[str, float]]) -> list[str]:  # pragma: no cover
        raise NotImplementedError

    def word_choices(self, field_name: str) -> list[str]:  # pragma: no cover
        raise NotImplementedError

    def valid(self, vis: dict[str, Any]) -> bool:
        return True

    def aim_spec(self, vis: dict[str, Any]) -> tuple[str, float, float, int]:
        """(field, lo, hi, sign) of the decisive number of this moment."""
        return (self.decisive, *self.decisive_range, self.decisive_sign)

    def alt_param(self, rng: random.Random) -> float:
        """The random part of the gentler action (a ratio of the plan's number by default)."""
        return round(rng.uniform(*self.alt_ratio), 4)

    def alt_from(self, vis: dict[str, Any], param: float) -> Any:
        """The gentler action's number for this moment."""
        value = float(vis[self.alt_field]) * float(param)
        return round(value, 3 if value < 1 else 2)

    # ---- shared ------------------------------------------------------------------------------------------------------------
    def hidden_names(self) -> tuple[str, ...]:
        raise NotImplementedError

    def rate(self, vis: dict[str, Any], draws: Sequence[dict[str, float]]) -> float:
        outcomes = self.calc(vis, draws)
        return sum(o in self.key for o in outcomes) / len(outcomes)

    def aim(self, vis: dict[str, Any], field_name: str, lo: float, hi: float, sign: int, target: float,
            draws_of: Callable[[dict[str, Any]], Sequence[dict[str, float]]], digits: int = 3) -> dict[str, Any]:
        """Bisection on ``vis[field_name]`` in [lo, hi] so the calculator's base-prior key rate is as close to `target` as the step
        function allows (`sign` = +1 if the rate rises with the number)."""
        a, b = float(lo), float(hi)
        best = None
        for _ in range(22):
            mid = (a + b) / 2
            trial = dict(vis)
            trial[field_name] = round(mid, digits)
            r = self.rate(trial, draws_of(trial))
            gap = abs(r - target)
            if best is None or gap < best[0] - 1e-12:
                best = (gap, trial[field_name])
            if (r < target) == (sign > 0):
                a = mid
            else:
                b = mid
        out = dict(vis)
        out[field_name] = best[1]
        return out


def _u(rng: random.Random, lo: float, hi: float, digits: int = 3) -> float:
    return round(rng.uniform(lo, hi), digits)


# ==========================================================================================================================
# roll
# ==========================================================================================================================

ROLL_ITEMS: dict[str, dict[str, Any]] = {
    # name, shape, radius, length (capsules), mass, spin-inertia ratio k behind the contents words
    "soup": {"name": "tin of tomato soup", "contents": "unopened", "shape": "capsule", "r": (0.036, 0.038), "len": (0.100, 0.112), "mass": (0.40, 0.45), "k": (0.15, 0.40), "twin": "soup_empty"},
    "soup_empty": {"name": "tin of tomato soup", "contents": "rinsed out and empty", "shape": "capsule", "r": (0.036, 0.038), "len": (0.100, 0.112), "mass": (0.05, 0.07), "k": (0.85, 0.96), "twin": "soup"},
    "pin": {"name": "beech rolling pin", "contents": "solid wood", "shape": "capsule", "r": (0.024, 0.029), "len": (0.32, 0.44), "mass": (0.45, 0.80), "k": (0.47, 0.52)},
    "tube": {"name": "cardboard poster tube", "contents": "empty", "shape": "capsule", "r": (0.034, 0.042), "len": (0.50, 0.70), "mass": (0.14, 0.24), "k": (0.82, 0.96)},
    "honey": {"name": "glass jar of honey", "contents": "full to the lid", "shape": "capsule", "r": (0.040, 0.044), "len": (0.115, 0.135), "mass": (0.60, 0.78), "k": (0.28, 0.48)},
    "water": {"name": "litre bottle of still water", "contents": "full", "shape": "capsule", "r": (0.040, 0.043), "len": (0.27, 0.31), "mass": (1.02, 1.08), "k": (0.10, 0.30), "twin": "water_empty"},
    "water_empty": {"name": "litre bottle of still water", "contents": "drunk empty, cap on", "shape": "capsule", "r": (0.040, 0.043), "len": (0.27, 0.31), "mass": (0.03, 0.05), "k": (0.80, 0.95), "twin": "water"},
    "pipe": {"name": "offcut of copper pipe", "contents": "hollow", "shape": "capsule", "r": (0.011, 0.016), "len": (0.20, 0.40), "mass": (0.12, 0.30), "k": (0.80, 0.95)},
    "candle": {"name": "pillar candle", "contents": "solid wax", "shape": "capsule", "r": (0.030, 0.036), "len": (0.12, 0.18), "mass": (0.28, 0.50), "k": (0.47, 0.52)},
    "tennis": {"name": "tennis ball", "contents": "hollow and pressurised", "shape": "sphere", "r": (0.033, 0.034), "mass": (0.056, 0.059), "k": (0.55, 0.67)},
    "rubber": {"name": "rubber dog ball", "contents": "solid rubber", "shape": "sphere", "r": (0.030, 0.036), "mass": (0.12, 0.19), "k": (0.38, 0.42)},
    "orange": {"name": "orange", "contents": "fresh", "shape": "sphere", "r": (0.036, 0.042), "mass": (0.15, 0.22), "k": (0.34, 0.42)},
    "toyball": {"name": "plastic toy ball", "contents": "thin-walled and hollow", "shape": "sphere", "r": (0.040, 0.050), "mass": (0.03, 0.05), "k": (0.60, 0.67)},
}
ROLL_SURFACES: dict[str, dict[str, Any]] = {
    "steel": {"text": "bare stainless steel", "mu_roll": (0.0002, 0.0008), "mu": (0.30, 0.45)},
    "melamine": {"text": "smooth white melamine", "mu_roll": (0.0003, 0.0010), "mu": (0.30, 0.45)},
    "cork": {"text": "a cork sheet", "mu_roll": (0.0010, 0.0025), "mu": (0.45, 0.65)},
    "silicone": {"text": "a silicone baking sheet", "mu_roll": (0.0012, 0.0030), "mu": (0.60, 0.90)},
    "rubber_mat": {"text": "a dimpled rubber liner", "mu_roll": (0.0015, 0.0040), "mu": (0.70, 1.00)},
    "cardboard": {"text": "a flattened piece of cardboard", "mu_roll": (0.0020, 0.0050), "mu": (0.40, 0.55)},
    "felt": {"text": "a thin felt liner", "mu_roll": (0.0030, 0.0070), "mu": (0.40, 0.60)},
}
ROLL_CONTEXTS = ("ramp", "tray")
#: Share of the plastic-pivot impact energy the rim impact keeps: 1.0, the textbook formula (checked against MuJoCo on 599 draws: 99.3 %
#: agreement on over / caught; the best constant, 0.97, gains 0.4 points).
ROLL_IMPACT_KEEP = 1.0


class Roll(Family):
    name = "roll"
    law = "rolling: a round item rolls at any tilt steeper than its small rolling-resistance angle (no static-friction threshold); a lip stops it only if it is tall enough for the item's radius and arrival speed"
    group = "unseen_law"
    key = ("over",)
    decisive = "tilt"
    decisive_range = (0.8, 26.0)
    decisive_sign = 1
    cf_factors = (Factor("tilt", "num", +1, 0.8, 26.0, digits=1), Factor("lip", "num", -1, 0.003, 0.032, digits=3),
                  Factor("dist", "num", +1, 0.04, 0.40, digits=2))
    partner_factors = (Factor("lip", "num", -1, 0.003, 0.032, digits=3), Factor("dist", "num", +1, 0.04, 0.40, digits=2),
                       Factor("surface", "word", 0), Factor("item", "word", 0))
    dir_factors = (Factor("mass", "num", 0, scale=True, digits=3, same_ratio=(2.0, 0.5)), Factor("lip", "num", -1, 0.003, 0.032, digits=3),
                   Factor("dist", "num", +1, 0.04, 0.40, digits=2), Factor("tilt", "num", +1, 0.8, 26.0, digits=1),
                   Factor("length", "num", 0, scale=True, digits=3, same_ratio=(1.5, 0.7)))
    alt_field = "tilt"
    alt_ratio = (0.55, 0.92)

    def hidden_names(self) -> tuple[str, ...]:
        return ("k", "mu", "mu_roll")

    def sample(self, rng: random.Random) -> dict[str, Any]:
        item_key = rng.choice(sorted(ROLL_ITEMS))
        item = ROLL_ITEMS[item_key]
        r = _u(rng, *item["r"], 4)
        vis = {
            "context": rng.choice(ROLL_CONTEXTS), "item": item_key, "radius": r, "mass": _u(rng, *item["mass"], 3),
            "length": _u(rng, *item["len"], 3) if item["shape"] == "capsule" else None,
            "surface": rng.choice(sorted(ROLL_SURFACES)), "lip": round(min(0.8 * r, rng.uniform(0.003, 0.028)), 3),
            "dist": _u(rng, 0.06, 0.34, 2), "tilt": 8.0, "ramp": _u(rng, 0.5, 0.9, 1), "hold": float(rng.choice((4, 5, 6))),
        }
        return vis

    def word_choices(self, field_name: str) -> list[str]:
        return sorted(ROLL_SURFACES) if field_name == "surface" else sorted(ROLL_ITEMS)

    def valid(self, vis: dict[str, Any]) -> bool:
        return 0.003 <= vis["lip"] <= 0.8 * vis["radius"] and vis["dist"] > math.sqrt(2 * vis["radius"] * vis["lip"]) + 0.02

    def words(self, vis: dict[str, Any]) -> dict[str, tuple[float, float]]:
        s = ROLL_SURFACES[vis["surface"]]
        return {"k": ROLL_ITEMS[vis["item"]]["k"], "mu": s["mu"], "mu_roll": s["mu_roll"]}

    def scene(self, vis: dict[str, Any]) -> dict[str, Any]:
        item = ROLL_ITEMS[vis["item"]]
        return {"shape": item["shape"], "radius": vis["radius"], "length": vis["length"], "mass": vis["mass"], "lip": vis["lip"], "dist": vis["dist"]}

    def action(self, vis: dict[str, Any]) -> dict[str, Any]:
        return {"tilt": vis["tilt"], "ramp": vis["ramp"], "hold": vis["hold"]}

    def calc(self, vis: dict[str, Any], draws: Sequence[dict[str, float]]) -> list[str]:
        """Rolling: stays put if tan θ ≤ μ_r/r; else rolls at g(sin θ − (μ_r/r) cos θ)/(1 + k) and reaches the lip (still rolling at
        the end of the hold → caught); over if the kept share of the plastic-pivot impact energy, ½u²(kr + z0)²/(kr² + R²), beats the
        climb over the rounded lip in the tilted gravity field (R = r + h/2, z0 = r − h/2)."""
        r, h, d = vis["radius"], vis["lip"], vis["dist"]
        th = math.radians(vis["tilt"])
        big_r, z0 = r + h / 2, r - h / 2
        travel = d - math.sqrt(2 * r * h)
        barrier = G * (big_r - math.sin(th) * math.sqrt(max(big_r**2 - z0**2, 0.0)) - math.cos(th) * z0)
        out = []
        for dr in draws:
            k, c = dr["k"], dr["mu_roll"] / r
            if math.tan(th) <= c:
                out.append("put")
                continue
            a_rel = G * (math.sin(th) - c * math.cos(th)) / (1 + k)
            t_arrive = math.sqrt(2 * max(travel, 0.0) / a_rel)
            if t_arrive > vis["hold"] + vis["ramp"] / 2:
                out.append("caught")
                continue
            if barrier <= 0:
                out.append("over")
                continue
            u2 = 2 * a_rel * max(travel, 0.0)
            ke = 0.5 * u2 * (k * r + z0) ** 2 / (k * r * r + big_r**2)
            out.append("over" if ROLL_IMPACT_KEEP * ke > barrier else "caught")
        return out

    def apply_word(self, vis: dict[str, Any], field_name: str, word: str) -> dict[str, Any] | None:
        out = dict(vis)
        if field_name == "item":
            twin = ROLL_ITEMS[vis["item"]].get("twin")
            if word != twin:
                return None
            out["item"] = word
            out["mass"] = round(sum(ROLL_ITEMS[word]["mass"]) / 2, 3)
            return out
        out[field_name] = word
        return out


# ==========================================================================================================================
# swing
# ==========================================================================================================================

SWING_LOADS: dict[str, dict[str, Any]] = {
    # dims: depth along the motion, width, height; com: centre-of-mass depth below the top, fraction of the height
    "bolts": {"name": "plastic crate of loose bolts", "dims": (0.40, 0.30, 0.22), "mass": (8.0, 14.0), "com": (0.60, 0.78)},
    "paint": {"name": "lidded bucket of wall paint", "dims": (0.29, 0.29, 0.33), "mass": (10.0, 14.0), "com": (0.52, 0.66)},
    "toolbox": {"name": "cantilever tool chest", "dims": (0.50, 0.22, 0.24), "mass": (6.0, 12.0), "com": (0.48, 0.68)},
    "soil": {"name": "sack of potting soil", "dims": (0.45, 0.30, 0.15), "mass": (15.0, 22.0), "com": (0.45, 0.60)},
    "tote": {"name": "tote of mixed spare parts", "dims": (0.40, 0.30, 0.25), "mass": (4.0, 9.0), "com": (0.40, 0.78)},
    "drum": {"name": "small drum of floor cleaner", "dims": (0.27, 0.27, 0.36), "mass": (11.0, 13.0), "com": (0.50, 0.58)},
}
SWING_STRAPS: dict[str, dict[str, Any]] = {
    "wire": {"text": "a steel wire rope", "stretch": (0.0000, 0.0006), "slop": (-0.005, 0.005)},
    "chain": {"text": "a short run of chain", "stretch": (0.0000, 0.0004), "slop": (-0.020, 0.020)},
    "sling": {"text": "a flat polyester sling", "stretch": (0.0020, 0.0060), "slop": (-0.010, 0.010)},
    "cotton": {"text": "a braided cotton rope", "stretch": (0.0060, 0.0150), "slop": (-0.015, 0.015)},
    "bungee": {"text": "a rubber bungee cord", "stretch": (0.0100, 0.0250), "slop": (-0.010, 0.010)},
}
SWING_OBSTACLES = ("steel upright of a pallet rack", "glass-fronted display cabinet", "stack of plastic crates", "frame of a doorway",
                   "parked trolley")
SWING_CONTEXTS = ("gantry", "arm", "rail")


class Swing(Family):
    name = "swing"
    law = "pendulum: a load hanging on a strap swings forward when its hook stops, reaching about v*sqrt(L/g) for a sudden stop (more for a faster move or a longer strap, less for a gentle stop); its mass barely matters unless the strap stretches"
    group = "unseen_law"
    key = ("brush", "strike")
    decisive = "gap"
    decisive_range = (0.01, 0.80)
    decisive_sign = -1
    cf_factors = (Factor("speed", "num", +1, 0.2, 1.6, digits=2), Factor("strap_len", "num", +1, 0.2, 1.4, digits=2),
                  Factor("gap", "num", -1, 0.01, 0.80, digits=2))
    partner_factors = (Factor("speed", "num", +1, 0.2, 1.6, digits=2), Factor("strap_len", "num", +1, 0.2, 1.4, digits=2),
                       Factor("gap", "num", -1, 0.01, 0.80, digits=2), Factor("decel", "num", +1, 0.8, 9.0, digits=1),
                       Factor("strap", "word", 0))
    dir_factors = (Factor("mass", "num", 0, scale=True, digits=1, same_ratio=(2.0, 0.5)), Factor("strap_len", "num", +1, 0.2, 1.4, digits=2),
                   Factor("speed", "num", +1, 0.2, 1.6, digits=2), Factor("gap", "num", -1, 0.01, 0.80, digits=2))
    alt_field = "speed"
    alt_ratio = (0.5, 0.85)

    def hidden_names(self) -> tuple[str, ...]:
        return ("com", "slop", "stretch")

    def sample(self, rng: random.Random) -> dict[str, Any]:
        load = rng.choice(sorted(SWING_LOADS))
        spec = SWING_LOADS[load]
        return {
            "context": rng.choice(SWING_CONTEXTS), "load": load, "mass": _u(rng, *spec["mass"], 1), "strap": rng.choice(sorted(SWING_STRAPS)),
            "strap_len": _u(rng, 0.25, 1.2, 2), "speed": _u(rng, 0.35, 1.4, 2), "decel": _u(rng, 1.5, 7.5, 1), "ramp": 0.1,
            "obstacle": rng.choice(SWING_OBSTACLES), "gap": 0.2,
        }

    def word_choices(self, field_name: str) -> list[str]:
        return sorted(SWING_STRAPS)

    def apply_word(self, vis: dict[str, Any], field_name: str, word: str) -> dict[str, Any] | None:
        out = dict(vis)
        out[field_name] = word
        return out

    def words(self, vis: dict[str, Any]) -> dict[str, tuple[float, float]]:
        s = SWING_STRAPS[vis["strap"]]
        return {"com": SWING_LOADS[vis["load"]]["com"], "slop": s["slop"], "stretch": s["stretch"]}

    def scene(self, vis: dict[str, Any]) -> dict[str, Any]:
        return {"dims": SWING_LOADS[vis["load"]]["dims"], "mass": vis["mass"], "strap": vis["strap_len"], "gap": vis["gap"]}

    def action(self, vis: dict[str, Any]) -> dict[str, Any]:
        return {"speed": vis["speed"], "decel": vis["decel"], "ramp": vis["ramp"]}

    def calc(self, vis: dict[str, Any], draws: Sequence[dict[str, float]]) -> list[str]:
        """Rigid physical pendulum (hook → strap → load) driven by the hook's planned deceleration (the same profile as the rollout),
        integrated for all draws at once (semi-implicit Euler, 1 ms) until every swing turns back; touch if the load's bottom front
        edge reaches the gap, strike if it is still moving at ≥ 0.25 m/s toward the upright there."""
        b, w, h = SWING_LOADS[vis["load"]]["dims"]
        m = vis["mass"]
        lengths = np.array([vis["strap_len"] * (1 + d["stretch"] * m) + d["slop"] for d in draws])
        com = np.array([d["com"] for d in draws])
        lc = lengths + com * h
        inertia = m * lc**2 + m * (b * b + h * h) / 12
        acc = ph.speed_change(-vis["speed"], vis["decel"], vis["ramp"])
        dt = ph.TIMESTEP
        phi = np.zeros(len(draws))
        omega = np.zeros(len(draws))
        reach = np.zeros(len(draws))
        speed_at = np.full(len(draws), np.nan)
        done = np.zeros(len(draws), dtype=bool)
        gap = vis["gap"]
        arm = lengths + h
        for i in range(acc.size + 6000):
            a = -acc[i] if i < acc.size else 0.0  # forward pseudo-acceleration while braking
            torque = m * lc * (a * np.cos(phi) - G * np.sin(phi))
            omega = omega + torque / inertia * dt
            phi = phi + omega * dt
            x = arm * np.sin(phi) + (b / 2) * np.cos(phi) - b / 2
            vx = omega * (arm * np.cos(phi) - (b / 2) * np.sin(phi))
            newly = (~done) & (x >= gap)
            speed_at[newly] = vx[newly]
            done |= newly
            reach = np.maximum(reach, x)
            if i >= acc.size and np.all(done | (omega < 0)):
                break
        out = []
        for j in range(len(draws)):
            if not done[j]:
                out.append("clear")
            else:
                out.append("strike" if speed_at[j] >= ph.STRIKE_SPEED else "brush")
        return out


# ==========================================================================================================================
# knock
# ==========================================================================================================================

KNOCK_OBJECTS: dict[str, dict[str, Any]] = {
    # dims: depth along the strike, width, height; hc: centre-of-mass height fraction; cx: offset along the strike (fraction of the depth)
    "oil": {"name": "square glass bottle of olive oil", "contents": "full", "dims": (0.065, 0.065, 0.29), "mass": (0.95, 1.10), "hc": (0.40, 0.47), "cx": (-0.02, 0.02), "twin": "oil_low"},
    "oil_low": {"name": "square glass bottle of olive oil", "contents": "down to its last few centimetres", "dims": (0.065, 0.065, 0.29), "mass": (0.45, 0.55), "hc": (0.30, 0.42), "cx": (-0.02, 0.02), "twin": "oil"},
    "oatmilk": {"name": "carton of oat milk", "contents": "unopened", "dims": (0.065, 0.095, 0.24), "mass": (1.02, 1.08), "hc": (0.46, 0.50), "cx": (-0.01, 0.01), "twin": "oatmilk_low"},
    "oatmilk_low": {"name": "carton of oat milk", "contents": "about a quarter left", "dims": (0.065, 0.095, 0.24), "mass": (0.28, 0.34), "hc": (0.18, 0.30), "cx": (-0.01, 0.01), "twin": "oatmilk"},
    "cereal": {"name": "box of cereal", "contents": "opened, contents settled", "dims": (0.070, 0.19, 0.30), "mass": (0.35, 0.55), "hc": (0.28, 0.45), "cx": (-0.03, 0.03)},
    "wine": {"name": "boxed bottle of wine", "contents": "sealed", "dims": (0.090, 0.090, 0.34), "mass": (1.30, 1.50), "hc": (0.38, 0.46), "cx": (-0.02, 0.02)},
    "vase": {"name": "square ceramic vase", "contents": "empty", "dims": (0.10, 0.10, 0.28), "mass": (0.80, 1.20), "hc": (0.36, 0.46), "cx": (-0.02, 0.02)},
    "cleaner": {"name": "spray bottle of glass cleaner", "contents": "about half full", "dims": (0.08, 0.05, 0.27), "mass": (0.42, 0.55), "hc": (0.26, 0.40), "cx": (-0.06, 0.06)},
    "lamp": {"name": "small table lamp", "contents": "with a heavy cast base", "dims": (0.12, 0.12, 0.35), "mass": (1.0, 1.4), "hc": (0.16, 0.26), "cx": (-0.02, 0.02)},
    "reams": {"name": "stack of printer paper reams", "contents": "five reams", "dims": (0.21, 0.30, 0.25), "mass": (11.5, 12.8), "hc": (0.49, 0.51), "cx": (-0.01, 0.01)},
}
KNOCK_SURFACES: dict[str, dict[str, Any]] = {
    "oak": {"text": "a varnished oak shelf", "mu": (0.30, 0.45)},
    "glass": {"text": "a glass shelf", "mu": (0.15, 0.30)},
    "barmat": {"text": "a rubber bar mat", "mu": (0.70, 1.00)},
    "laminate": {"text": "a laminate countertop", "mu": (0.25, 0.40)},
    "steel": {"text": "a brushed steel worktop", "mu": (0.20, 0.35)},
    "linen": {"text": "a linen tablecloth", "mu": (0.40, 0.60)},
}
KNOCK_CONTEXTS = ("forearm", "gripper", "elbow")
#: Share of the bare impulse energy needed to topple: the arm keeps pressing through its stop distance, so less than the impulse
#: energy is enough (fitted once against the T9 knock labels: Brier 0.117 against 0.201 for the plain impulse rule).
KNOCK_ENERGY = 0.6


class Knock(Family):
    name = "knock"
    law = "impulse at a height: a short strike makes a standing object rotate about its far bottom edge (struck high) or slide (struck low, slippery base); it topples when the strike's energy lifts its centre of mass over that edge — easiest for narrow, tall, top-heavy objects"
    group = "unseen_law"
    key = ("topple",)
    decisive = "speed"
    decisive_range = (0.03, 1.6)
    decisive_sign = 1
    cf_factors = (Factor("height", "num", +1, 0.04, 0.34, digits=3), Factor("stop", "num", +1, 0.004, 0.04, digits=3))
    partner_factors = (Factor("height", "num", +1, 0.04, 0.34, digits=3), Factor("speed", "num", +1, 0.03, 1.6, digits=2),
                       Factor("surface", "word", 0), Factor("object", "word", 0), Factor("stop", "num", +1, 0.004, 0.04, digits=3))
    dir_factors = (Factor("mass", "num", 0, scale=True, digits=2, same_ratio=(2.0, 0.5)), Factor("height", "num", +1, 0.04, 0.34, digits=3),
                   Factor("speed", "num", +1, 0.03, 1.6, digits=2), Factor("surface", "word", 0), Factor("stop", "num", +1, 0.004, 0.04, digits=3))
    alt_field = "speed"
    alt_ratio = (0.45, 0.85)

    def hidden_names(self) -> tuple[str, ...]:
        return ("cx", "hc", "mu")

    def sample(self, rng: random.Random) -> dict[str, Any]:
        obj = rng.choice(sorted(KNOCK_OBJECTS))
        spec = KNOCK_OBJECTS[obj]
        height = spec["dims"][2]
        return {
            "context": rng.choice(KNOCK_CONTEXTS), "object": obj, "mass": _u(rng, *spec["mass"], 2), "surface": rng.choice(sorted(KNOCK_SURFACES)),
            "height": round(height * rng.uniform(0.25, 0.92), 3), "speed": 0.5, "stop": _u(rng, 0.006, 0.030, 3),
        }

    def valid(self, vis: dict[str, Any]) -> bool:
        return 0.03 <= vis["height"] <= 0.95 * KNOCK_OBJECTS[vis["object"]]["dims"][2]

    def word_choices(self, field_name: str) -> list[str]:
        return sorted(KNOCK_SURFACES) if field_name == "surface" else sorted(KNOCK_OBJECTS)

    def apply_word(self, vis: dict[str, Any], field_name: str, word: str) -> dict[str, Any] | None:
        out = dict(vis)
        if field_name == "object":
            if word != KNOCK_OBJECTS[vis["object"]].get("twin"):
                return None
            out["object"] = word
            out["mass"] = round(sum(KNOCK_OBJECTS[word]["mass"]) / 2, 2)
            return out
        out[field_name] = word
        return out

    def words(self, vis: dict[str, Any]) -> dict[str, tuple[float, float]]:
        spec = KNOCK_OBJECTS[vis["object"]]
        return {"cx": spec["cx"], "hc": spec["hc"], "mu": KNOCK_SURFACES[vis["surface"]]["mu"]}

    def scene(self, vis: dict[str, Any]) -> dict[str, Any]:
        return {"dims": KNOCK_OBJECTS[vis["object"]]["dims"], "mass": vis["mass"]}

    def action(self, vis: dict[str, Any]) -> dict[str, Any]:
        return {"speed": vis["speed"], "height": vis["height"], "stop": vis["stop"]}

    def calc(self, vis: dict[str, Any], draws: Sequence[dict[str, float]]) -> list[str]:
        """Impulsive strike: the struck point takes the arm's speed, so pivoting about the far bottom edge starts at ω = v/h with energy
        ½·I_P·ω². Below the no-slip height h* = 2(b² + H²)/(3(H + μb)) the pivot's impulsive reaction leaves the friction cone and the
        base slides; only a share (h/h*)² of that energy goes into the rotation. Topple if the rotation energy beats
        :data:`KNOCK_ENERGY` × the lift of the centre of mass over the edge, or (no slip) if the arm's stop distance alone pushes the
        face past the tipping angle (stop/h ≥ atan(d/h_c)); else slide if the base slips and v²/(2μg) ≥ 1 cm; else stay."""
        b, w, H = KNOCK_OBJECTS[vis["object"]]["dims"]
        m, v, h, stop = vis["mass"], vis["speed"], vis["height"], vis["stop"]
        out = []
        for d in draws:
            dx, dz = b / 2 - d["cx"] * b, d["hc"] * H
            ip = m * (b * b + H * H) / 12 + m * (dx * dx + dz * dz)
            omega = v / h
            barrier = m * G * (math.hypot(dx, dz) - dz)
            ke = 0.5 * ip * omega * omega
            jx = ip * omega / h
            rx, rz = m * omega * dz - jx, m * omega * dx
            slips = abs(rx) > d["mu"] * rz
            h_star = 2 * (b * b + H * H) / (3 * (H + d["mu"] * b))
            rotating = ke if not slips else ke * min(1.0, h / h_star) ** 2
            if rotating > KNOCK_ENERGY * barrier or (not slips and stop / h >= math.atan2(dx, dz)):
                out.append("topple")
            elif slips and v * v / (2 * d["mu"] * G) >= 0.01:
                out.append("slide")
            else:
                out.append("stay")
        return out


# ==========================================================================================================================
# conveyor
# ==========================================================================================================================

CONVEYOR_ITEMS: dict[str, dict[str, Any]] = {
    # dims: depth along the belt (narrow side forward), width, height
    "soap": {"name": "tall bottle of dish soap", "contents": "full", "dims": (0.060, 0.090, 0.26), "mass": (0.72, 0.80), "hc": (0.40, 0.48), "cx": (-0.02, 0.02)},
    "pods": {"name": "tub of laundry pods", "contents": "nearly full", "dims": (0.15, 0.20, 0.21), "mass": (1.0, 1.3), "hc": (0.38, 0.48), "cx": (-0.03, 0.03)},
    "basil": {"name": "potted basil plant", "contents": "freshly watered", "dims": (0.11, 0.11, 0.24), "mass": (0.6, 0.9), "hc": (0.22, 0.34), "cx": (-0.03, 0.03)},
    "tins": {"name": "stack of four tins taped together", "contents": "all full", "dims": (0.075, 0.075, 0.42), "mass": (1.6, 1.8), "hc": (0.48, 0.52), "cx": (-0.01, 0.01)},
    "lampbox": {"name": "boxed floor lamp", "contents": "packed in a way I cannot see", "dims": (0.12, 0.16, 0.50), "mass": (2.0, 3.2), "hc": (0.25, 0.60), "cx": (-0.06, 0.06)},
    "kettlebox": {"name": "boxed rice cooker", "contents": "with its paperwork on top", "dims": (0.17, 0.20, 0.25), "mass": (1.1, 1.4), "hc": (0.38, 0.50), "cx": (-0.04, 0.04)},
    "vasebox": {"name": "tall gift box holding a vase", "contents": "padded with paper", "dims": (0.10, 0.10, 0.36), "mass": (0.9, 1.3), "hc": (0.32, 0.46), "cx": (-0.03, 0.03)},
}
CONVEYOR_BELTS: dict[str, dict[str, Any]] = {
    "pvc": {"text": "a smooth PVC belt", "mu": (0.25, 0.40)},
    "roughtop": {"text": "a rough-top rubber belt", "mu": (0.70, 1.00)},
    "modular": {"text": "a modular plastic belt", "mu": (0.15, 0.28)},
    "felt": {"text": "a felt-covered belt", "mu": (0.40, 0.55)},
}
CONVEYOR_CONTEXTS = ("packing", "sorting")


class Conveyor(Family):
    name = "conveyor"
    law = "tip or slide under acceleration: an item standing on a belt slides when the belt's start or stop needs more than mu*g, and tips when it exceeds g*(half base - offset)/(centre-of-mass height), whichever comes first, if the push lasts long enough"
    group = "seen_law_new_scene"
    key = ("topple",)
    decisive = "rate"
    decisive_range = (0.3, 14.0)
    decisive_sign = 1
    cf_factors = (Factor("rate", "num", +1, 0.3, 14.0, digits=1), Factor("speed", "num", +1, 0.15, 1.6, digits=2))
    partner_factors = (Factor("rate", "num", +1, 0.3, 14.0, digits=1), Factor("speed", "num", +1, 0.15, 1.6, digits=2),
                       Factor("belt", "word", 0), Factor("turned", "word", 0))
    dir_factors = (Factor("mass", "num", 0, scale=True, digits=2, same_ratio=(2.0, 0.5)), Factor("rate", "num", +1, 0.3, 14.0, digits=1),
                   Factor("speed", "num", +1, 0.15, 1.6, digits=2), Factor("belt", "word", 0), Factor("turned", "word", 0))
    alt_field = "rate"
    alt_ratio = (0.4, 0.8)

    def hidden_names(self) -> tuple[str, ...]:
        return ("cx", "hc", "mu")

    def sample(self, rng: random.Random) -> dict[str, Any]:
        """A moment where tipping can happen at all: the static tipping threshold at the middle of the centre-of-mass range lies below
        the belt's highest sliding threshold (else the item always slides first and toppling is out of reach)."""
        for _ in range(200):
            item = rng.choice(sorted(CONVEYOR_ITEMS))
            vis = {
                "context": rng.choice(CONVEYOR_CONTEXTS), "item": item, "mass": _u(rng, *CONVEYOR_ITEMS[item]["mass"], 2),
                "belt": rng.choice(sorted(CONVEYOR_BELTS)), "event": rng.choice(("stop", "start")), "speed": _u(rng, 0.3, 1.3, 2), "rate": 3.0,
                "ramp": _u(rng, 0.04, 0.12, 2), "turned": rng.choice(("narrow", "wide")),
            }
            if self.tippable(vis):
                return vis
        return vis

    def tippable(self, vis: dict[str, Any]) -> bool:
        b, w, h = self.dims(vis)
        hc = sum(CONVEYOR_ITEMS[vis["item"]]["hc"]) / 2
        return G * (b / 2) / (hc * h) < 0.9 * CONVEYOR_BELTS[vis["belt"]]["mu"][1] * G


    def dims(self, vis: dict[str, Any]) -> tuple[float, float, float]:
        b, w, h = CONVEYOR_ITEMS[vis["item"]]["dims"]
        return (b, w, h) if vis["turned"] == "narrow" else (w, b, h)

    def word_choices(self, field_name: str) -> list[str]:
        return sorted(CONVEYOR_BELTS) if field_name == "belt" else ["narrow", "wide"]

    def apply_word(self, vis: dict[str, Any], field_name: str, word: str) -> dict[str, Any] | None:
        out = dict(vis)
        out[field_name] = word
        return out

    def words(self, vis: dict[str, Any]) -> dict[str, tuple[float, float]]:
        spec = CONVEYOR_ITEMS[vis["item"]]
        return {"cx": spec["cx"], "hc": spec["hc"], "mu": CONVEYOR_BELTS[vis["belt"]]["mu"]}

    def scene(self, vis: dict[str, Any]) -> dict[str, Any]:
        return {"dims": self.dims(vis), "mass": vis["mass"]}

    def action(self, vis: dict[str, Any]) -> dict[str, Any]:
        return {"speed": vis["speed"], "rate": vis["rate"], "ramp": vis["ramp"]}

    def calc(self, vis: dict[str, Any], draws: Sequence[dict[str, float]]) -> list[str]:
        """Rocking block under the belt's pulse (the same profile as the rollout), all draws at once: the item rotates about its edge on
        the thrown side once a·h_c > g·d (d = half base − offset toward that side), I_edge·φ'' = m[a(d sin φ + h_c cos φ) − g(d cos φ −
        h_c sin φ)], and topples once φ passes atan(d/h_c); while flat it slides when a > μg (relative acceleration a − μg, then μg
        braking after the pulse) — slide if it moves ≥ 2 cm, else stay."""
        b, w, h = self.dims(vis)
        m = vis["mass"]
        hc = np.array([d["hc"] for d in draws]) * h
        dd = b / 2 - np.array([d["cx"] for d in draws]) * b
        mu = np.array([d["mu"] for d in draws])
        inertia = m * (b * b + h * h) / 12 + m * (dd * dd + hc * hc)
        crit = np.arctan2(dd, hc)
        pulse = ph.speed_change(vis["speed"], vis["rate"], vis["ramp"])
        dt = ph.TIMESTEP
        phi = np.zeros(len(draws))
        omega = np.zeros(len(draws))
        slide_v = np.zeros(len(draws))
        slide_x = np.zeros(len(draws))
        fell = np.zeros(len(draws), dtype=bool)
        for i in range(pulse.size + 1500):
            a = pulse[i] if i < pulse.size else 0.0
            torque = m * (a * (dd * np.sin(phi) + hc * np.cos(phi)) - G * (dd * np.cos(phi) - hc * np.sin(phi)))
            rocking = (phi > 0) | (torque > 0)
            omega = np.where(rocking, omega + torque / inertia * dt, 0.0)
            phi = np.maximum(phi + omega * dt, 0.0)
            omega = np.where(phi <= 0, 0.0, omega)
            fell |= phi > crit
            flat = ~rocking
            drive = np.where(flat & (slide_v > 0), a - mu * G, np.where(flat & (a > mu * G), a - mu * G, 0.0))
            slide_v = np.maximum(slide_v + drive * dt, 0.0)
            slide_x += slide_v * dt
            if i >= pulse.size and np.all((slide_v <= 0) & ((phi <= 0) | fell)):
                break
        return ["topple" if fell[j] else ("slide" if slide_x[j] >= 0.02 else "stay") for j in range(len(draws))]


# ==========================================================================================================================
# overhang
# ==========================================================================================================================

OVERHANG_BOARDS: dict[str, dict[str, Any]] = {
    "pine": {"text": "pine board", "density": (420.0, 540.0)},
    "birch": {"text": "birch plywood board", "density": (620.0, 720.0)},
    "mdf": {"text": "MDF panel", "density": (700.0, 820.0)},
    "bamboo": {"text": "bamboo chopping board", "density": (650.0, 780.0)},
    "oak": {"text": "oak plank", "density": (650.0, 760.0)},
}
OVERHANG_ITEMS: dict[str, dict[str, Any]] = {
    # dims: depth along the board, width, height; mass: the stated "about" mass; scale: hidden factor behind the contents words
    "paint": {"name": "tin of emulsion paint", "contents": "partly used", "dims": (0.17, 0.17, 0.19), "mass": 3.4, "scale": (0.35, 0.85), "icx": (-0.02, 0.02), "stated": False},
    "flour": {"name": "bag of flour", "contents": "unopened", "dims": (0.11, 0.17, 0.20), "mass": 1.5, "scale": (0.97, 1.03), "icx": (-0.03, 0.03), "stated": True},
    "kettle": {"name": "electric kettle", "contents": "with some water left in it", "dims": (0.20, 0.15, 0.24), "mass": 1.0, "scale": (1.10, 1.90), "icx": (-0.10, 0.10), "stated": False},
    "books": {"name": "stack of three hardback books", "contents": "", "dims": (0.24, 0.17, 0.12), "mass": 2.4, "scale": (0.85, 1.15), "icx": (-0.02, 0.02), "stated": True},
    "cactus": {"name": "small potted cactus", "contents": "in damp soil", "dims": (0.12, 0.12, 0.16), "mass": 0.9, "scale": (0.75, 1.25), "icx": (-0.03, 0.03), "stated": True},
    "toolbag": {"name": "canvas tool bag", "contents": "loaded with hand tools", "dims": (0.34, 0.18, 0.20), "mass": 4.0, "scale": (0.60, 1.30), "icx": (-0.15, 0.15), "stated": True},
    "pan": {"name": "cast-iron frying pan", "contents": "", "dims": (0.26, 0.26, 0.05), "mass": 2.6, "scale": (0.95, 1.05), "icx": (-0.05, 0.05), "stated": True},
}
OVERHANG_CONTEXTS = ("pantry", "workshop")


class Overhang(Family):
    name = "overhang"
    law = "centre of mass past an edge: a board sticking out past a shelf edge tips when the combined centre of mass of the board and what stands on it passes the edge; an item set near the board's end falls when its own centre of mass passes the end"
    group = "seen_law_new_scene"
    key = ("item_falls", "board_tips")
    decisive = "over"
    decisive_range = (0.02, 0.45)
    decisive_sign = 1
    cf_factors = (Factor("over", "num", +1, 0.02, 0.45, digits=3), Factor("inset", "num", -1, -0.03, 0.22, digits=3))
    partner_factors = (Factor("over", "num", +1, 0.02, 0.45, digits=3), Factor("inset", "num", -1, -0.03, 0.22, digits=3),
                       Factor("board", "word", 0), Factor("place_tol", "num", +1, 0.003, 0.03, digits=3),
                       Factor("board_t", "num", -1, 0.008, 0.040, digits=3))
    dir_factors = (Factor("item_height", "num", 0, scale=True, digits=3, same_ratio=(1.5, 0.7)), Factor("over", "num", +1, 0.02, 0.45, digits=3),
                   Factor("inset", "num", -1, -0.03, 0.22, digits=3), Factor("board", "word", 0), Factor("board_t", "num", -1, 0.008, 0.040, digits=3),
                   Factor("board_len", "num", -1, 0.35, 1.0, digits=2))
    alt_field = "inset"
    alt_ratio = (0.0, 0.0)

    def hidden_names(self) -> tuple[str, ...]:
        return ("board_density", "err", "icx", "item_scale")

    def sample(self, rng: random.Random) -> dict[str, Any]:
        item = rng.choice(sorted(OVERHANG_ITEMS))
        length = _u(rng, 0.40, 0.90, 2)
        return {
            "context": rng.choice(OVERHANG_CONTEXTS), "board": rng.choice(sorted(OVERHANG_BOARDS)), "board_len": length,
            "board_w": _u(rng, 0.20, 0.34, 2), "board_t": _u(rng, 0.012, 0.024, 3), "item": item,
            "item_height": OVERHANG_ITEMS[item]["dims"][2], "over": round(0.3 * length, 3),
            "inset": _u(rng, 0.0, 0.14, 3) if rng.random() < 0.7 else _u(rng, -0.02, 0.04, 3),
            "place_tol": _u(rng, 0.004, 0.02, 3), "mode": rng.choice(("board", "board", "item")),
        }

    def aim_spec(self, vis: dict[str, Any]) -> tuple[str, float, float, int]:
        if vis.get("mode") == "item":
            return ("inset", -0.03, 0.10, -1)
        return ("over", 0.02, round(0.48 * vis["board_len"], 3), +1)

    def alt_param(self, rng: random.Random) -> float:
        return round(rng.uniform(0.02, 0.12), 4)

    def alt_from(self, vis: dict[str, Any], param: float) -> Any:
        return round(vis["inset"] + float(param), 3)

    def valid(self, vis: dict[str, Any]) -> bool:
        d = OVERHANG_ITEMS[vis["item"]]["dims"][0]
        return (0.02 <= vis["over"] <= 0.48 * vis["board_len"] and vis["inset"] <= vis["over"] - d / 2 + 0.06
                and vis["inset"] >= -0.03 and vis["over"] - vis["inset"] > -0.05)

    def word_choices(self, field_name: str) -> list[str]:
        return sorted(OVERHANG_BOARDS)

    def apply_word(self, vis: dict[str, Any], field_name: str, word: str) -> dict[str, Any] | None:
        out = dict(vis)
        out[field_name] = word
        return out

    def words(self, vis: dict[str, Any]) -> dict[str, tuple[float, float]]:
        spec = OVERHANG_ITEMS[vis["item"]]
        return {"board_density": OVERHANG_BOARDS[vis["board"]]["density"], "icx": spec["icx"], "item_scale": spec["scale"]}

    def fixed(self, vis: dict[str, Any]) -> dict[str, tuple[float, float]]:
        return {"err": (-vis["place_tol"], vis["place_tol"])}

    def item_dims(self, vis: dict[str, Any]) -> tuple[float, float, float]:
        d, w, _ = OVERHANG_ITEMS[vis["item"]]["dims"]
        return (d, w, vis["item_height"])

    def scene(self, vis: dict[str, Any]) -> dict[str, Any]:
        return {"board": (vis["board_len"], vis["board_w"], vis["board_t"]), "item_dims": self.item_dims(vis)}

    def action(self, vis: dict[str, Any]) -> dict[str, Any]:
        return {"over": vis["over"], "inset": vis["inset"]}

    def physics_draw(self, vis: dict[str, Any], draw: dict[str, float]) -> dict[str, float]:
        volume = vis["board_len"] * vis["board_w"] * vis["board_t"]
        return {"board_mass": draw["board_density"] * volume, "item_mass": OVERHANG_ITEMS[vis["item"]]["mass"] * draw["item_scale"],
                "icx": draw["icx"], "err": draw["err"]}

    def calc(self, vis: dict[str, Any], draws: Sequence[dict[str, float]]) -> list[str]:
        """Statics: the item falls if its centre of mass lands past the board's end; else the board tips if the summed moment about the
        shelf edge, m_item·x_item + m_board·(over − L/2), is positive."""
        d = self.item_dims(vis)[0]
        out = []
        for raw in draws:
            dr = self.physics_draw(vis, raw)
            x_item = vis["over"] - vis["inset"] + dr["err"] + dr["icx"] * d
            if x_item > vis["over"]:
                out.append("item_falls")
            elif dr["item_mass"] * x_item + dr["board_mass"] * (vis["over"] - vis["board_len"] / 2) > 0:
                out.append("board_tips")
            else:
                out.append("stays")
        return out


# ==========================================================================================================================
# suction
# ==========================================================================================================================

SUCTION_PANELS: dict[str, dict[str, Any]] = {
    "glass": {"name": "tempered glass shelf", "dims": (0.50, 0.30, 0.006), "mass": 2.3},
    "acrylic": {"name": "acrylic sheet", "dims": (0.60, 0.40, 0.005), "mass": 1.4},
    "tile": {"name": "ceramic floor tile", "dims": (0.30, 0.30, 0.009), "mass": 1.9},
    "chipboard": {"name": "laminated chipboard panel", "dims": (0.60, 0.30, 0.016), "mass": 2.0},
    "composite": {"name": "aluminium composite sheet", "dims": (0.50, 0.40, 0.004), "mass": 1.2},
}
SUCTION_FACES: dict[str, dict[str, Any]] = {
    "clean": {"text": "clean and dry", "mu": (0.65, 0.95)},
    "dusty": {"text": "slightly dusty", "mu": (0.40, 0.60)},
    "damp": {"text": "still damp from washing", "mu": (0.25, 0.45)},
    "oily": {"text": "filmed with cutting oil", "mu": (0.10, 0.22)},
    "film": {"text": "still in its protective plastic film", "mu": (0.45, 0.65)},
}
SUCTION_PUMPS: dict[str, dict[str, Any]] = {
    "strong": {"text": "my vacuum gauge sits high in its green band", "vacuum": (60000.0, 75000.0)},
    "normal": {"text": "my vacuum gauge reads in the middle of its green band", "vacuum": (45000.0, 60000.0)},
    "worn": {"text": "the cup's rubber lip is worn and nicked", "vacuum": (35000.0, 50000.0)},
    "weak": {"text": "my pump has been running weak all shift", "vacuum": (25000.0, 40000.0)},
}
SUCTION_CONTEXTS = ("glazing", "kitting")


class Suction(Family):
    name = "suction"
    law = "friction-limited holding: a vacuum cup holds a panel by friction from its net clamping force (vacuum force minus weight); the panel slips when a move's acceleration asks more than mu*(F - m*g)/m, and off-centre weight adds a peeling moment"
    group = "seen_law_new_mechanism"
    key = ("drop",)
    decisive = "rate"
    decisive_range = (0.3, 30.0)
    decisive_sign = 1
    cf_factors = (Factor("rate", "num", +1, 0.3, 30.0, digits=1), Factor("speed", "num", +1, 0.2, 2.0, digits=2),
                  Factor("offset", "num", +1, 0.0, 0.03, digits=3))
    partner_factors = (Factor("rate", "num", +1, 0.3, 30.0, digits=1), Factor("speed", "num", +1, 0.2, 2.0, digits=2),
                       Factor("face", "word", 0), Factor("pump", "word", 0), Factor("cup", "num", -1, 0.03, 0.07, digits=3))
    dir_factors = (Factor("carry", "num", 0, scale=True, digits=1, same_ratio=(2.0, 0.5)), Factor("rate", "num", +1, 0.3, 30.0, digits=1),
                   Factor("cup", "num", -1, 0.03, 0.07, digits=3), Factor("face", "word", 0))
    alt_field = "rate"
    alt_ratio = (0.4, 0.8)

    def hidden_names(self) -> tuple[str, ...]:
        return ("mass_scale", "mu", "vacuum")

    def sample(self, rng: random.Random) -> dict[str, Any]:
        panel = rng.choice(sorted(SUCTION_PANELS))
        return {
            "context": rng.choice(SUCTION_CONTEXTS), "panel": panel, "face": rng.choice(sorted(SUCTION_FACES)), "pump": rng.choice(sorted(SUCTION_PUMPS)),
            "cup": rng.choice((0.035, 0.040, 0.045, 0.050, 0.060)), "offset": _u(rng, 0.0, 0.012, 3), "speed": _u(rng, 0.5, 1.5, 2),
            "rate": 4.0, "ramp": 0.05, "carry": float(rng.choice((2, 3, 4, 5))),
        }

    def valid(self, vis: dict[str, Any]) -> bool:
        """The panel must hang on the cup at rest under the base reading's middle: vacuum force ≥ 1.25 × weight."""
        spec = SUCTION_PANELS[vis["panel"]]
        lo, hi = SUCTION_PUMPS[vis["pump"]]["vacuum"]
        force = (lo + hi) / 2 * math.pi * (vis["cup"] / 2) ** 2
        return force >= 1.25 * spec["mass"] * G and vis["offset"] <= 0.03

    def word_choices(self, field_name: str) -> list[str]:
        return sorted(SUCTION_FACES) if field_name == "face" else sorted(SUCTION_PUMPS)

    def apply_word(self, vis: dict[str, Any], field_name: str, word: str) -> dict[str, Any] | None:
        out = dict(vis)
        out[field_name] = word
        return out

    def words(self, vis: dict[str, Any]) -> dict[str, tuple[float, float]]:
        return {"mu": SUCTION_FACES[vis["face"]]["mu"], "vacuum": SUCTION_PUMPS[vis["pump"]]["vacuum"]}

    def fixed(self, vis: dict[str, Any]) -> dict[str, tuple[float, float]]:
        return {"mass_scale": (0.94, 1.06)}

    def scene(self, vis: dict[str, Any]) -> dict[str, Any]:
        return {"panel": SUCTION_PANELS[vis["panel"]]["dims"], "cup": vis["cup"], "offset": vis["offset"]}

    def action(self, vis: dict[str, Any]) -> dict[str, Any]:
        return {"rate": vis["rate"], "speed": vis["speed"], "ramp": vis["ramp"], "carry": vis["carry"]}

    def physics_draw(self, vis: dict[str, Any], draw: dict[str, float]) -> dict[str, float]:
        return {"vacuum": draw["vacuum"], "mu": draw["mu"], "mass": SUCTION_PANELS[vis["panel"]]["mass"] * draw["mass_scale"]}

    def calc(self, vis: dict[str, Any], draws: Sequence[dict[str, float]]) -> list[str]:
        """Clamping force N = ΔP·A − m·g; drop if N ≤ 0 or the weight's moment m·g·e exceeds N·s (s = half the pad's side); slip
        acceleration a_s = μ·N/m; each speed change of duration V/a slips ½·(a − a_s)·(V/a)² when a > a_s; drop if that slip moves the
        cup far enough off-centre to peel (e + slip > N·s/(m·g)), shift if the slip is ≥ 3 mm, else hold."""
        side_half = math.sqrt(math.pi) * vis["cup"] / 4
        area = math.pi * (vis["cup"] / 2) ** 2
        out = []
        for raw in draws:
            dr = self.physics_draw(vis, raw)
            m = dr["mass"]
            n = dr["vacuum"] * area - m * G
            if n <= 0 or m * G * vis["offset"] >= n * side_half:
                out.append("drop")
                continue
            a_s = dr["mu"] * n / m
            slip = 0.0
            if vis["rate"] > a_s:
                t = vis["speed"] / vis["rate"]
                slip = 0.5 * (vis["rate"] - a_s) * t * t
            if vis["offset"] + slip > n * side_half / (m * G):
                out.append("drop")
            elif slip >= ph.SHIFT_M:
                out.append("shift")
            else:
                out.append("hold")
        return out


FAMILIES: dict[str, Family] = {f.name: f for f in (Roll(), Swing(), Knock(), Conveyor(), Overhang(), Suction())}


def family(name: str) -> Family:
    return FAMILIES[name]
