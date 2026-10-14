"""Author A4 — catalogs, priors, scene sampling, aiming and the physics of each variant (v0.6 held-out field-robot test).

Three scenes, one per law, all outdoors (orchards, farms, building sites, marinas, parks, footpath verges):

* **bank** (A) — a robot sets a load down on sloping ground and lets go. Decisive stated number: the slope (degrees). Hidden: the
  friction of the load on the ground (ground word), the centre-of-mass height and downhill shift (contents word), and the local slope
  deviation from the stated reading (ground word: σ in degrees). Actions: the plan, turned a quarter turn first, carried on to a
  flatter patch (its slope stated).
* **ledge** (B) — the robot creeps forward until its nose sticks out past the edge of a drop. Decisive stated number: the overhang
  (cm). Hidden: the robot's centre of mass along its length and its height (payload word), the friction of the ground (ground word)
  and the error of the stopping point (ground word: σ in m). Actions: the plan, stop shorter by a stated distance, turn round and
  reverse to the same overhang (the centre of mass mirrored).
* **anchor** (C) — the parked robot holds a winch line hauling a load up a ramp. Decisive stated number: the load's estimated mass
  (kg). Hidden: the robot's friction on the ramp (ramp word), the load's drag coefficient (load word), the ratio of the true to the
  stated mass (word for how the mass is known). Actions: the plan, a soft start (its time to line speed stated), haul from the flat
  apron behind (the line runs over the crest).

**Priors.** ``base`` = the catalog ranges; ``alt_a`` = friction-type ranges × 0.7 (friction and drag), centre-of-mass and mass ranges
1.5 × wider around the same centre, placement σ × 1.5; ``alt_b`` = friction × 1.3, those ranges half as wide, σ × 0.75. Every hidden
quantity is drawn from the same K = 32 uniform points per group (columns of ``U``), mapped through each prior — the draws are shared by
the priors, the variants and both records of a pair.

**Aiming.** The decisive number of the base record is set so that the analytic key probability over the group's 32 base-prior draws
hits a target drawn uniformly in [0.03, 0.97] (bisection on the closed-form threshold, then rounded to the stated resolution), so the
MuJoCo key labels are flat across 0–1.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from statistics import NormalDist
from typing import Any

import numpy as np

__all__ = [
    "ACTION_COST",
    "AUTHOR",
    "DOMAINS",
    "EVENT",
    "K",
    "OUTCOMES",
    "PLAN_KEY",
    "PRIORS",
    "SCENES",
    "SEVERE",
    "Scene",
    "aim",
    "hidden_values",
    "key_analytic",
    "prior_specs",
]

AUTHOR = "a4"
K = 32
PRIORS = ("base", "alt_a", "alt_b")
SCENES = ("bank", "ledge", "anchor")
LAW = {"bank": "A", "ledge": "B", "anchor": "C"}
PHENOMENA = {"bank": ["tip_or_slide_under_tilt"], "ledge": ["centre_of_mass_past_edge"], "anchor": ["friction_limited_holding"]}
#: Outcome categories of each scene (the outcome question's option keys), the key event and the severe event.
OUTCOMES = {"bank": ("stays_put", "slides_down", "tips_over"), "ledge": ("holds", "noses_down", "goes_over"),
            "anchor": ("holds_fast", "creeps", "dragged")}
EVENT = {"bank": ("slides_down", "tips_over"), "ledge": ("noses_down", "goes_over"), "anchor": ("creeps", "dragged")}
SEVERE = {"bank": ("tips_over",), "ledge": ("goes_over",), "anchor": ("dragged",)}
#: The plan's option key in the comparison and action questions (every scene).
PLAN_KEY = {scene: "the_plan" for scene in SCENES}
#: Cost of each outcome and of each action (the action question's expected costs = outcome frequencies · cost + action cost).
OUTCOME_COST = {"bank": {"stays_put": 0.0, "slides_down": 1.0, "tips_over": 2.5}, "ledge": {"holds": 0.0, "noses_down": 1.5, "goes_over": 4.0},
                "anchor": {"holds_fast": 0.0, "creeps": 1.0, "dragged": 4.0}}
ACTION_COST = {"bank": {"the_plan": 0.0, "turned_first": 0.05, "flatter_spot": 0.3}, "ledge": {"the_plan": 0.0, "stop_short": 0.2, "reverse_in": 0.35},
               "anchor": {"the_plan": 0.0, "soft_start": 0.1, "from_apron": 0.35}}
#: Option key → variant name.
OPTIONS = {"bank": {"the_plan": "planned", "turned_first": "opt_turned", "flatter_spot": "opt_flatter"},
           "ledge": {"the_plan": "planned", "stop_short": "opt_short", "reverse_in": "opt_reverse"},
           "anchor": {"the_plan": "planned", "soft_start": "opt_soft", "from_apron": "opt_apron"}}
DOMAINS = ("orchard", "farm", "construction", "marina", "park", "sidewalk")

_PHI_INV = NormalDist().inv_cdf

# --------------------------------------------------------------------------
# Catalogs (word → range). Friction ranges are the pair's coefficient; centre-of-mass ranges are fractions of the box dimension.
# --------------------------------------------------------------------------

#: bank — ground: (mu_lo, mu_hi, local slope deviation σ in degrees, domains).
BANK_GROUND = {
    "dry clipped turf": (0.45, 0.65, 1.5, ("orchard", "park", "farm", "sidewalk")),
    "wet grass": (0.25, 0.42, 1.5, ("orchard", "park", "farm", "sidewalk", "marina")),
    "packed dry earth": (0.55, 0.75, 1.0, ("orchard", "farm", "construction")),
    "loose gravel": (0.40, 0.58, 2.0, ("construction", "marina", "park", "farm")),
    "sticky clay after rain": (0.30, 0.52, 2.0, ("construction", "farm", "orchard")),
    "smooth poured concrete": (0.50, 0.70, 0.5, ("construction", "marina", "sidewalk")),
    "dew-soaked timber decking": (0.20, 0.38, 0.7, ("marina", "park")),
    "dry sand": (0.45, 0.62, 2.5, ("marina", "construction", "park")),
    "mossy stone flags": (0.22, 0.40, 1.0, ("park", "sidewalk", "marina")),
}
#: bank — loads: (long, short, height ranges in m; mass range kg; filler noun; domains).
BANK_LOADS = {
    "picking bin": ((0.60, 0.80), (0.40, 0.50), (0.30, 0.42), (12, 32), "pears", ("orchard", "farm")),
    "jerrycan": ((0.30, 0.36), (0.15, 0.19), (0.40, 0.48), (6, 22), "diesel", ("farm", "construction", "marina")),
    "tool chest": ((0.50, 0.70), (0.25, 0.35), (0.25, 0.36), (15, 40), "hand tools", ("construction", "park", "sidewalk")),
    "cool box": ((0.50, 0.65), (0.35, 0.42), (0.35, 0.46), (8, 30), "bait and ice", ("marina", "park")),
    "hive box": ((0.48, 0.52), (0.38, 0.42), (0.24, 0.50), (15, 45), "honey frames", ("orchard", "farm")),
    "instrument case": ((0.45, 0.55), (0.30, 0.36), (0.50, 0.65), (10, 26), "survey gear", ("construction", "park", "sidewalk")),
    "seed tote": ((0.55, 0.65), (0.35, 0.45), (0.28, 0.36), (10, 26), "seed potatoes", ("farm", "orchard")),
    "battery crate": ((0.40, 0.50), (0.30, 0.36), (0.30, 0.40), (25, 45), "spare cells", ("construction", "marina", "sidewalk")),
}
#: bank — contents: (centre-of-mass height above the box centre, fraction of H; downhill shift, fraction of the downslope extent).
BANK_CONTENTS = {
    "packed full, level with the top": ((-0.04, 0.04), (-0.03, 0.03)),
    "half full, settled at the bottom": ((-0.24, -0.12), (-0.05, 0.05)),
    "nearly empty, one heavy offcut on the floor": ((-0.38, -0.24), (-0.12, 0.12)),
    "heaped up over the rim": ((0.04, 0.16), (-0.04, 0.04)),
    "full, with a coil of chain strapped on the lid": ((0.16, 0.30), (-0.05, 0.05)),
    "half full, all shoved to one side": ((-0.16, -0.04), (-0.14, 0.14)),
}
#: bank — the contents that make sense for each load (a lidded case is never heaped over its rim, a jerrycan holds liquid).
_LIDDED = tuple(c for c in BANK_CONTENTS if c != "heaped up over the rim")
BANK_CONTENTS_FOR = {
    "picking bin": tuple(BANK_CONTENTS), "seed tote": tuple(BANK_CONTENTS), "battery crate": tuple(BANK_CONTENTS),
    "jerrycan": ("packed full, level with the top", "half full, settled at the bottom", "full, with a coil of chain strapped on the lid"),
    "tool chest": _LIDDED, "cool box": _LIDDED, "hive box": _LIDDED, "instrument case": _LIDDED,
}

#: ledge — ground at the top: (mu_lo, mu_hi, stopping-point σ in m, domains).
LEDGE_GROUND = {
    "dry concrete": (0.60, 0.80, 0.010, ("construction", "marina", "sidewalk", "park")),
    "wet timber boards": (0.30, 0.48, 0.025, ("marina", "park")),
    "loose gravel": (0.45, 0.62, 0.035, ("construction", "park", "farm", "marina")),
    "wet grass": (0.25, 0.45, 0.040, ("orchard", "park", "farm", "sidewalk")),
    "soft mud": (0.20, 0.40, 0.050, ("farm", "orchard", "construction")),
    "steel chequer plate": (0.35, 0.55, 0.015, ("construction", "marina")),
    "dry turf": (0.45, 0.65, 0.025, ("orchard", "park", "farm", "sidewalk")),
}
#: ledge — robot bodies: (length, width, height ranges in m; mass range kg; domains).
LEDGE_ROBOTS = {
    "tracked site rover": ((0.90, 1.30), (0.60, 0.80), (0.45, 0.65), (120, 260), ("construction",)),
    "mower bot": ((0.70, 0.95), (0.55, 0.70), (0.30, 0.42), (45, 110), ("park", "farm", "orchard")),
    "dock runner": ((0.80, 1.10), (0.55, 0.70), (0.40, 0.55), (70, 160), ("marina",)),
    "survey crawler": ((0.60, 0.85), (0.45, 0.60), (0.35, 0.50), (35, 90), ("construction", "park")),
    "sprayer rover": ((1.10, 1.50), (0.70, 0.90), (0.60, 0.85), (180, 420), ("orchard", "farm")),
    "kerb inspector": ((0.75, 1.00), (0.50, 0.65), (0.40, 0.55), (50, 120), ("sidewalk", "park")),
}
#: ledge — payload: (centre of mass forward of the middle, fraction of the length; above the middle, fraction of the height).
LEDGE_PAYLOAD = {
    "front tank full": ((0.08, 0.16), (-0.05, 0.10)),
    "front tank half full": ((0.03, 0.09), (-0.10, 0.04)),
    "running empty, battery pack in the back": ((-0.10, -0.03), (-0.16, -0.04)),
    "front tool tray loaded": ((0.05, 0.13), (0.00, 0.15)),
    "counterweight plates on the rear": ((-0.16, -0.08), (-0.12, 0.00)),
    "load spread evenly": ((-0.03, 0.03), (-0.08, 0.06)),
    "camera mast folded forward": ((0.02, 0.08), (0.10, 0.24)),
}
#: ledge — edges: (drop range in m, domains).
LEDGE_EDGES = {
    "terrace wall": ((0.25, 0.60), ("orchard", "park", "farm")),
    "loading-bay lip": ((0.30, 0.60), ("construction", "marina")),
    "kerb step": ((0.12, 0.25), ("sidewalk", "park")),
    "quay step": ((0.30, 0.60), ("marina",)),
    "trench edge": ((0.30, 0.60), ("construction", "farm")),
    "raised bed edge": ((0.15, 0.35), ("park", "farm", "orchard")),
    "pontoon step": ((0.20, 0.40), ("marina",)),
    "retaining wall": ((0.35, 0.60), ("sidewalk", "construction", "park")),
}

#: anchor — ramp under the robot: (mu_lo, mu_hi, domains).
ANCHOR_RAMP = {
    "dry ridged concrete": (0.60, 0.85, ("marina", "construction")),
    "wet concrete": (0.45, 0.65, ("marina", "construction")),
    "weed-slick concrete": (0.18, 0.35, ("marina",)),
    "packed gravel": (0.50, 0.70, ("construction", "farm", "park", "orchard")),
    "wet clay track": (0.30, 0.50, ("farm", "orchard", "construction")),
    "dry grass bank": (0.40, 0.60, ("park", "farm", "orchard")),
    "steel mesh plates": (0.35, 0.55, ("construction", "marina")),
    "damp sand": (0.40, 0.58, ("marina", "park")),
}
#: anchor — robots: (length, width, height ranges in m; mass range kg; domains).
ANCHOR_ROBOTS = {
    "slipway tug": ((1.20, 1.80), (0.80, 1.10), (0.60, 0.90), (300, 700), ("marina",)),
    "tracked winch rover": ((1.00, 1.50), (0.70, 1.00), (0.50, 0.80), (250, 600), ("construction", "park", "farm")),
    "farm tug": ((1.40, 2.00), (0.90, 1.20), (0.70, 1.00), (400, 900), ("farm", "orchard")),
    "yard hauler": ((1.00, 1.40), (0.70, 0.90), (0.50, 0.70), (200, 500), ("construction", "marina", "orchard", "park")),
}
#: anchor — loads on the line: (drag coefficient range, domains).
ANCHOR_LOADS = {
    "boat on a roller trailer": ((0.02, 0.05), ("marina",)),
    "boat on a bunk trailer": ((0.04, 0.08), ("marina",)),
    "jet-ski trolley": ((0.03, 0.07), ("marina",)),
    "loaded farm cart": ((0.05, 0.10), ("farm", "orchard")),
    "skip on steel skids": ((0.25, 0.45), ("construction",)),
    "log on a drag sled": ((0.30, 0.55), ("park", "farm")),
    "pallet on a sled": ((0.25, 0.40), ("construction", "orchard", "park")),
}
#: anchor — how the load's mass is known: (true / stated ratio range).
ANCHOR_MASS_SOURCE = {
    "weighed on the scale this morning": (0.97, 1.03),
    "going by the paperwork": (0.92, 1.10),
    "the owner's estimate": (0.85, 1.20),
    "my rough guess, and there may be gear aboard": (0.95, 1.35),
}

#: Hidden quantities per scene: (name, type) — type u_fric (friction-like, scaled), u_mass (centre-of-mass / mass, widened or
#: narrowed around its centre), n_place (placement σ, scaled). The order is the column of U.
HIDDEN = {
    "bank": (("mu", "u_fric"), ("com_z", "u_mass"), ("com_down", "u_mass"), ("slope_dev", "n_place")),
    "ledge": (("mu", "u_fric"), ("com_x", "u_mass"), ("com_z", "u_mass"), ("stop_dev", "n_place")),
    "anchor": (("mu", "u_fric"), ("drag", "u_fric"), ("mass_ratio", "u_mass")),
}


def _r(rng: np.random.Generator, lo: float, hi: float, nd: int = 2) -> float:
    return round(float(rng.uniform(lo, hi)), nd)


def _pick(rng: np.random.Generator, names: list[str]) -> str:
    return names[int(rng.integers(len(names)))]


# --------------------------------------------------------------------------
# Priors and draws
# --------------------------------------------------------------------------


def base_ranges(scene: str, vis: dict[str, Any]) -> dict[str, tuple[str, float, float]]:
    """The base prior of a record's visible facts: {quantity: ("u", lo, hi) | ("n", 0, σ)} in physical units (friction coefficient,
    metres for centre-of-mass offsets and stop error, degrees for the slope deviation, a ratio for the mass)."""
    if scene == "bank":
        mu_lo, mu_hi, sigma, _ = BANK_GROUND[vis["ground"]]
        (z_lo, z_hi), (s_lo, s_hi) = BANK_CONTENTS[vis["contents"]]
        h = vis["height"]
        return {"mu": ("u", mu_lo, mu_hi), "com_z": ("u", z_lo * h, z_hi * h), "com_down": ("u", s_lo, s_hi), "slope_dev": ("n", 0.0, sigma)}
    if scene == "ledge":
        mu_lo, mu_hi, sigma, _ = LEDGE_GROUND[vis["ground"]]
        (x_lo, x_hi), (z_lo, z_hi) = LEDGE_PAYLOAD[vis["payload"]]
        return {"mu": ("u", mu_lo, mu_hi), "com_x": ("u", x_lo * vis["length"], x_hi * vis["length"]),
                "com_z": ("u", z_lo * vis["height"], z_hi * vis["height"]), "stop_dev": ("n", 0.0, sigma)}
    mu_lo, mu_hi, _ = ANCHOR_RAMP[vis["ramp"]]
    (d_lo, d_hi), _ = ANCHOR_LOADS[vis["load"]]
    m_lo, m_hi = ANCHOR_MASS_SOURCE[vis["mass_source"]]
    return {"mu": ("u", mu_lo, mu_hi), "drag": ("u", d_lo, d_hi), "mass_ratio": ("u", m_lo, m_hi)}


def prior_specs(scene: str, vis: dict[str, Any]) -> dict[str, dict[str, list[Any]]]:
    """Base, alt_a and alt_b priors (the module rule) as JSON-able ``{prior: {quantity: ["u", lo, hi] | ["n", 0, σ]}}``."""
    base = base_ranges(scene, vis)
    kinds = dict(HIDDEN[scene])
    out: dict[str, dict[str, list[Any]]] = {}
    for prior, (fric, widen, place) in (("base", (1.0, 1.0, 1.0)), ("alt_a", (0.7, 1.5, 1.5)), ("alt_b", (1.3, 0.5, 0.75))):
        specs = {}
        for name, (dist, a, b) in base.items():
            kind = kinds[name]
            if kind == "u_fric":
                specs[name] = ["u", round(a * fric, 6), round(b * fric, 6)]
            elif kind == "u_mass":
                centre, half = (a + b) / 2, (b - a) / 2 * widen
                specs[name] = ["u", round(centre - half, 6), round(centre + half, 6)]
            else:
                specs[name] = ["n", 0.0, round(b * place, 6)]
        out[prior] = specs
    return out


def draw_points(draws_seed: int, scene: str) -> np.ndarray:
    """The group's K × D uniform points (shared by the priors, the variants and both records of a pair)."""
    rng = np.random.default_rng(int(draws_seed))
    return rng.uniform(1e-6, 1 - 1e-6, size=(K, len(HIDDEN[scene])))


def hidden_values(scene: str, specs: dict[str, list[Any]], points: np.ndarray) -> dict[str, np.ndarray]:
    """Map the uniform points through one prior's specs → {quantity: array of K values}."""
    out = {}
    for column, (name, _) in enumerate(HIDDEN[scene]):
        dist, a, b = specs[name]
        u = points[:, column]
        out[name] = a + u * (b - a) if dist == "u" else np.array([a + b * _PHI_INV(float(x)) for x in u])
    return out


# --------------------------------------------------------------------------
# Variants: the physical inputs of one hypothetical
# --------------------------------------------------------------------------


def variant_facts(scene: str, vis: dict[str, Any], variant: str, variants: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """The visible facts of `variant` (the record's facts with the variant's overrides) → the physical inputs (no hidden values)."""
    facts = {**vis, **(variants.get(variant) or {})}
    if scene == "bank":
        across = facts["orientation"] == "across"
        down, side = (facts["short"], facts["long"]) if across else (facts["long"], facts["short"])
        return {"dims": (down, side, facts["height"]), "mass": facts["mass"], "slope": facts["slope"]}
    if scene == "ledge":
        return {"dims": (facts["length"], facts["width"], facts["height"]), "mass": facts["mass"], "drop": facts["drop"],
                "overhang": facts["overhang"], "mirror": bool(facts.get("reverse", False))}
    return {"dims": (facts["length"], facts["width"], facts["height"]), "mass": facts["mass"], "robot_slope": 0.0 if facts.get("apron") else facts["slope"],
            "load_slope": facts["slope"], "load_mass": facts["load_mass"], "accel": facts["line_speed"] / facts["start_time"],
            "line_speed": facts["line_speed"]}


def key_analytic(scene: str, phys: dict[str, Any], hid: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Closed-form key and severe events over the hidden draws (aiming and the sanity report; labels come from MuJoCo).

    bank: moves iff tan(slope + dev) ≥ min(mu, b/h) (b = half the downslope extent minus the downhill shift, h = height of the centre
    of mass); tips iff tan(slope + dev) ≥ b/h and mu ≥ b/h (static tipping, or tipping while sliding). ledge: the nose drops iff the
    centre of mass is past the edge: overhang + dev > length/2 − com_x (mirrored for the reverse); severe is not closed-form (None →
    the key). anchor: slips iff the peak tension + m g sin θ > mu m g cos θ; dragged iff the steady tension does."""
    if scene == "bank":
        down, _, height = phys["dims"]
        b = down / 2 - hid["com_down"] * down
        h = height / 2 + hid["com_z"]
        ratio = np.maximum(b, 1e-6) / np.maximum(h, 1e-6)
        tan = np.tan(np.radians(np.clip(phys["slope"] + hid["slope_dev"], 0.0, 89.0)))
        moves = tan >= np.minimum(hid["mu"], ratio)
        tips = (tan >= ratio) & (hid["mu"] >= ratio)
        return moves, tips
    if scene == "ledge":
        length = phys["dims"][0]
        cx = -hid["com_x"] if phys["mirror"] else hid["com_x"]
        drops = phys["overhang"] + hid["stop_dev"] > length / 2 - cx
        return drops, drops
    g = 9.81
    th_r, th_l = math.radians(phys["robot_slope"]), math.radians(phys["load_slope"])
    capacity = (hid["mu"] * math.cos(th_r) - math.sin(th_r)) * phys["mass"] * g
    load = phys["load_mass"] * hid["mass_ratio"]
    steady = load * g * (math.sin(th_l) + hid["drag"] * math.cos(th_l))
    peak = steady + load * phys["accel"]
    return peak > capacity, steady > capacity


# --------------------------------------------------------------------------
# Scene specs: sampling the visible facts, the decisive number, options and one-factor changes
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Scene:
    name: str
    decisive: str
    lo: float
    hi: float
    resolution: float
    #: near counterfactual step range (absolute, or relative for the anchor's mass)
    near: tuple[float, float]
    relative: bool = False
    factors: tuple[tuple[str, float], ...] = field(default_factory=tuple)

    def round(self, value: float) -> float:
        step = self.resolution
        out = round(round(float(value) / step) * step, 6)
        return float(min(max(out, self.lo), self.hi))


SCENE_SPECS = {
    "bank": Scene("bank", "slope", 2.0, 42.0, 0.5, (1.0, 2.0), factors=(("ground", 0.35), ("contents", 0.35), ("height", 0.30))),
    "ledge": Scene("ledge", "overhang", 0.03, 0.95, 0.01, (0.03, 0.05),
                   factors=(("payload", 0.35), ("ground", 0.25), ("drop", 0.20), ("length", 0.20))),
    "anchor": Scene("anchor", "load_mass", 20.0, 4000.0, 10.0, (0.06, 0.10), relative=True,
                    factors=(("ramp", 0.30), ("load", 0.20), ("mass_source", 0.15), ("slope", 0.20), ("start_time", 0.15))),
}


def sample_visible(scene: str, rng: np.random.Generator) -> dict[str, Any]:
    """The visible facts of a new group (without the decisive number), in one domain."""
    if scene == "bank":
        load = _pick(rng, sorted(BANK_LOADS))
        (l_lo, l_hi), (s_lo, s_hi), (h_lo, h_hi), (m_lo, m_hi), filler, domains = BANK_LOADS[load]
        domain = _pick(rng, sorted(domains))
        grounds = sorted(g for g, v in BANK_GROUND.items() if domain in v[3])
        return {"domain": domain, "load": load, "filler": filler, "long": _r(rng, l_lo, l_hi), "short": _r(rng, s_lo, s_hi),
                "height": _r(rng, h_lo, h_hi), "mass": float(int(rng.integers(m_lo, m_hi + 1))), "contents": _pick(rng, sorted(BANK_CONTENTS_FOR[load])),
                "ground": _pick(rng, grounds), "orientation": "across" if rng.random() < 0.5 else "along"}
    if scene == "ledge":
        robot = _pick(rng, sorted(LEDGE_ROBOTS))
        (l_lo, l_hi), (w_lo, w_hi), (h_lo, h_hi), (m_lo, m_hi), domains = LEDGE_ROBOTS[robot]
        domain = _pick(rng, sorted(domains))
        edges = sorted(e for e, v in LEDGE_EDGES.items() if domain in v[1])
        edge = _pick(rng, edges)
        d_lo, d_hi = LEDGE_EDGES[edge][0]
        grounds = sorted(g for g, v in LEDGE_GROUND.items() if domain in v[3])
        return {"domain": domain, "robot": robot, "length": _r(rng, l_lo, l_hi), "width": _r(rng, w_lo, w_hi), "height": _r(rng, h_lo, h_hi),
                "mass": float(int(rng.integers(m_lo, m_hi + 1))), "payload": _pick(rng, sorted(LEDGE_PAYLOAD)), "ground": _pick(rng, grounds),
                "edge": edge, "drop": _r(rng, d_lo, d_hi)}
    robot = _pick(rng, sorted(ANCHOR_ROBOTS))
    (l_lo, l_hi), (w_lo, w_hi), (h_lo, h_hi), (m_lo, m_hi), domains = ANCHOR_ROBOTS[robot]
    domain = _pick(rng, sorted(domains))
    ramps = sorted(r for r, v in ANCHOR_RAMP.items() if domain in v[2])
    loads = sorted(x for x, v in ANCHOR_LOADS.items() if domain in v[1]) or sorted(ANCHOR_LOADS)
    return {"domain": domain, "robot": robot, "length": _r(rng, l_lo, l_hi), "width": _r(rng, w_lo, w_hi), "height": _r(rng, h_lo, h_hi),
            "mass": float(int(rng.integers(m_lo // 10, m_hi // 10 + 1)) * 10), "ramp": _pick(rng, ramps), "load": _pick(rng, loads),
            "mass_source": _pick(rng, sorted(ANCHOR_MASS_SOURCE)), "slope": float(int(rng.integers(4, 17))),
            "line_speed": _r(rng, 0.2, 0.5, 1), "start_time": _r(rng, 0.3, 1.2, 1)}


def options_values(scene: str, vis: dict[str, Any], rng: np.random.Generator) -> dict[str, dict[str, Any]]:
    """The stated parameters of the two alternative actions (shared by both records of a pair)."""
    if scene == "bank":
        return {"opt_turned": {"orientation": "along" if vis["orientation"] == "across" else "across"},
                "opt_flatter": {"flat_drop": _r(rng, 4.0, 10.0, 0), "flat_distance": float(int(rng.integers(4, 16)))}}
    if scene == "ledge":
        return {"opt_short": {"short_by": float(int(rng.integers(2, 6)) * 0.05)}, "opt_reverse": {"reverse": True}}
    return {"opt_soft": {"start_time": _r(rng, 3.0, 5.0, 1)}, "opt_apron": {"apron": True, "apron_distance": float(int(rng.integers(3, 12)))}}


def resolve_variants(scene: str, vis: dict[str, Any], option_values: dict[str, dict[str, Any]], cf_value: float) -> dict[str, dict[str, Any]]:
    """The record's ``meta.variants``: {planned: {}, cf: {decisive: value}, option variants: the overrides on this record's facts}."""
    spec = SCENE_SPECS[scene]
    out: dict[str, dict[str, Any]] = {"planned": {}, "cf": {spec.decisive: cf_value}}
    for variant, values in option_values.items():
        if scene == "bank" and variant == "opt_flatter":
            out[variant] = {"slope": values["flat_slope"]}
        elif scene == "ledge" and variant == "opt_short":
            out[variant] = {"overhang": round(max(0.0, vis["overhang"] - values["short_by"]), 6)}
        elif scene == "anchor" and variant == "opt_apron":
            out[variant] = {"apron": True}
        else:
            out[variant] = {k: v for k, v in values.items() if k in ("orientation", "reverse", "start_time")}
    return out


def aim(scene: str, vis: dict[str, Any], hid_base: dict[str, np.ndarray], target: float, *, variants: dict | None = None) -> tuple[float | None, float]:
    """The decisive number (rounded) whose analytic base-prior key share over the group's draws is nearest `target`, and that share.
    None if the target cannot be reached inside the scene's range."""
    spec = SCENE_SPECS[scene]

    def share(value: float) -> float:
        facts = {**vis, spec.decisive: value}
        phys = variant_facts(scene, facts, "planned", {})
        return float(np.mean(key_analytic(scene, phys, hid_base)[0]))

    lo, hi = spec.lo, spec.hi
    if scene == "ledge":
        hi = min(hi, 0.95 * vis["length"])
    s_lo, s_hi = share(lo), share(hi)
    if not (s_lo <= target <= s_hi):
        return None, float("nan")
    for _ in range(60):
        mid = (lo + hi) / 2
        if share(mid) < target:
            lo = mid
        else:
            hi = mid
    candidates = sorted({spec.round(lo), spec.round(hi), spec.round(lo - spec.resolution), spec.round(hi + spec.resolution)})
    best = min(candidates, key=lambda v: (abs(share(v) - target), v))
    return best, share(best)


def factor_change(scene: str, vis: dict[str, Any], factor: str, rng: np.random.Generator) -> tuple[Any, str] | None:
    """A new value for one non-decisive stated fact (`factor`) and its kind (word / number); None if nothing else fits."""
    domain = vis["domain"]
    if scene == "bank":
        if factor == "ground":
            others = sorted(g for g, v in BANK_GROUND.items() if domain in v[3] and g != vis["ground"])
            return (_pick(rng, others), "word") if others else None
        if factor == "contents":
            return _pick(rng, sorted(c for c in BANK_CONTENTS_FOR[vis["load"]] if c != vis["contents"])), "word"
        if factor == "height":
            h = vis["height"] * (1 + (1 if rng.random() < 0.5 else -1) * float(rng.uniform(0.2, 0.4)))
            return round(min(max(h, 0.15), 0.85), 2), "number"
    if scene == "ledge":
        if factor == "payload":
            return _pick(rng, sorted(p for p in LEDGE_PAYLOAD if p != vis["payload"])), "word"
        if factor == "ground":
            others = sorted(g for g, v in LEDGE_GROUND.items() if domain in v[3] and g != vis["ground"])
            return (_pick(rng, others), "word") if others else None
        if factor == "drop":
            lo, hi = LEDGE_EDGES[vis["edge"]][0]
            for _ in range(20):
                d = _r(rng, lo, hi)
                if abs(d - vis["drop"]) >= 0.08:
                    return d, "number"
            return None
        if factor == "length":
            length = vis["length"] * (1 + (1 if rng.random() < 0.5 else -1) * float(rng.uniform(0.12, 0.25)))
            length = round(length, 2)
            return (length, "number") if length > vis["overhang"] / 0.95 + 0.02 else None
    if scene == "anchor":
        if factor == "ramp":
            others = sorted(r for r, v in ANCHOR_RAMP.items() if domain in v[2] and r != vis["ramp"])
            return (_pick(rng, others), "word") if others else None
        if factor == "load":
            others = sorted(x for x, v in ANCHOR_LOADS.items() if domain in v[1] and x != vis["load"])
            return (_pick(rng, others), "word") if others else None
        if factor == "mass_source":
            return _pick(rng, sorted(m for m in ANCHOR_MASS_SOURCE if m != vis["mass_source"])), "word"
        if factor == "slope":
            for _ in range(20):
                s = float(int(rng.integers(3, 19)))
                if abs(s - vis["slope"]) >= 3:
                    return s, "number"
            return None
        if factor == "start_time":
            for _ in range(20):
                t = _r(rng, 0.3, 1.2, 1)
                if abs(t - vis["start_time"]) >= 0.3:
                    return t, "number"
            return None
    raise ValueError(f"{scene}: unknown factor {factor!r}")


def choose(rng: np.random.Generator, weighted: tuple[tuple[str, float], ...]) -> str:
    names = [n for n, _ in weighted]
    weights = np.array([w for _, w in weighted], dtype=float)
    return names[int(rng.choice(len(names), p=weights / weights.sum()))]

