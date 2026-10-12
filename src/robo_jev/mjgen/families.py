"""v0.3b families — catalogs, priors, candidate scenes, variants and MuJoCo labels (Task v0.3b §1, controller update 2026-10-12).

A **group** is one scene plus, for about half of them, a **paired** scene that differs in exactly one stated number. Every group has
one set of K hidden-draw points (a scrambled Latin hypercube in the unit cube, seeded per group); each prior setting maps the same
points to values (common random numbers across priors, variants and the two members of a pair), so differences between variants,
priors and pair members are measured on the same hidden worlds.

**Priors.** ``base`` = the ranges behind the words in the text (catalogs below). Two alternative but reasonable settings
(:data:`PRIOR_SETTINGS`): ``alt_a`` — friction ranges × 0.7, centre-of-mass / mass ranges 1.5 × wider around the same centre,
placement errors × 1.5; ``alt_b`` — friction × 1.3, ranges half as wide, placement errors × 0.75. Every variant is rolled out
under all three, so each record carries the raw outcome frequencies per prior (``meta.outcomes``) and the question labels per prior
(``meta.labels_by_prior``); ``labels`` = the ``base`` labels.

**Aiming.** The planned value of each scene's decisive number (deck pitch, squeeze force, top-box offset) is chosen so the base
prior's **static** (closed-form) answer to the key question equals a target drawn uniformly (inverse-CDF aiming: the quantile of
the per-draw static thresholds, :func:`solve_value`), plus some clear-cut cases beyond either end. The MuJoCo label then lands
near the target, so key labels cover [0, 1] roughly evenly before the balanced selection (:mod:`robo_jev.mjgen.generate`). The
counterfactual value (questions inside a record) and the paired value are aimed the same way at a second target, usually far
from the first (so the label moves) and sometimes close (so "about the same" is a real answer).
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import random
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

import numpy as np
from scipy.stats import norm, qmc

from robo_jev.mjgen import physics as ph

__all__ = [
    "DEFAULT_K",
    "FAMILIES",
    "GENERATOR",
    "OUTCOMES",
    "PRIORS",
    "PRIOR_SETTINGS",
    "Group",
    "FACTORS",
    "candidate",
    "draw_values",
    "factor_pair",
    "group_seed",
    "label_group",
    "prior_ranges",
    "solve_value",
    "static_outcome",
    "unit_draws",
]

GENERATOR = "v03b-mujoco-train-v1"
DEFAULT_K = 32
FAMILIES = ("tilt", "grip", "tower")
OUTCOMES = {"tilt": ph.TILT_OUTCOMES, "grip": ph.GRIP_OUTCOMES, "tower": ph.TOWER_OUTCOMES}
#: the key event (a failure) and its severe part, per family
EVENT = {"tilt": ("slide", "tip"), "grip": ("slip", "drop"), "tower": ("top", "more")}
SEVERE = {"tilt": "tip", "grip": "drop", "tower": "more"}
PRIORS = ("base", "alt_a", "alt_b")
PRIOR_SETTINGS = {
    "base": {"friction": 1.0, "spread": 1.0, "placement": 1.0, "what": "the ranges behind the words in the text"},
    "alt_a": {"friction": 0.7, "spread": 1.5, "placement": 1.5,
              "what": "friction ranges x 0.7; centre-of-mass and mass ranges 1.5 x wider (same centre); placement errors x 1.5"},
    "alt_b": {"friction": 1.3, "spread": 0.5, "placement": 0.75,
              "what": "friction ranges x 1.3; centre-of-mass and mass ranges half as wide (same centre); placement errors x 0.75"},
}
G = ph.GRAVITY
#: a wet surface (deck, or the held item's outside): friction × a value in this range (hidden), on top of the material's range
WET_FACTOR = (0.55, 0.75)
#: grip: time for the hand's acceleration to build up to its stated value (and to fall back), unless a record states another
GRIP_JERK = 0.04
DOMAINS = ("warehouse", "hospital", "lab", "retail", "office", "home", "factory", "kitchen")

# --------------------------------------------------------------------------
# Catalogs — the priors behind the words (new for v0.3b; none of T8b's objects, surfaces or wording)
# --------------------------------------------------------------------------

#: tilt: deck surfaces → friction prior (uniform) with the boxes in this family.
TILT_SURFACES = {
    "aluminium": {"text": "anodised aluminium plate", "mu": (0.22, 0.34)},
    "tread": {"text": "diamond-tread steel", "mu": (0.38, 0.55)},
    "neoprene": {"text": "a neoprene liner", "mu": (0.75, 1.05)},
    "melamine": {"text": "a melamine panel", "mu": (0.17, 0.27)},
    "cork": {"text": "cork-faced board", "mu": (0.45, 0.65)},
    "pvc": {"text": "a PVC sheet", "mu": (0.30, 0.44)},
}
#: tilt loads: L along the deck's length (fore-aft), W across, H up (ranges), mass (kg), allowed contents words.
TILT_LOADS = (
    {"key": "archive", "name": "archive box of files", "dims": ((0.36, 0.44), (0.28, 0.34), (0.24, 0.32)), "mass": (5.0, 12.0),
     "fills": ("level", "unsure", "fore", "aft")},
    {"key": "fridge", "name": "boxed mini fridge", "dims": ((0.40, 0.50), (0.40, 0.50), (0.70, 0.90)), "mass": (14.0, 24.0),
     "fills": ("level", "low", "unsure")},
    {"key": "speaker", "name": "speaker cabinet", "dims": ((0.24, 0.34), (0.24, 0.34), (0.55, 0.85)), "mass": (7.0, 15.0),
     "fills": ("level", "low", "high", "unsure")},
    {"key": "monitor", "name": "boxed monitor", "dims": ((0.14, 0.22), (0.55, 0.75), (0.40, 0.55)), "mass": (4.0, 9.0),
     "fills": ("level", "low", "unsure")},
    {"key": "tower_pc", "name": "computer tower", "dims": ((0.38, 0.48), (0.18, 0.24), (0.38, 0.50)), "mass": (6.0, 14.0),
     "fills": ("level", "low", "unsure", "fore", "aft")},
    {"key": "jars", "name": "crate of glass jars", "dims": ((0.30, 0.42), (0.24, 0.34), (0.20, 0.32)), "mass": (6.0, 13.0),
     "fills": ("level", "unsure", "high")},
    {"key": "bolts", "name": "bin of loose bolts", "dims": ((0.30, 0.45), (0.20, 0.30), (0.18, 0.30)), "mass": (8.0, 20.0),
     "fills": ("low", "level", "fore", "aft")},
    {"key": "flatpack", "name": "flat-pack shelf carton", "dims": ((0.10, 0.16), (0.40, 0.60), (0.80, 1.20)), "mass": (9.0, 18.0),
     "fills": ("level", "unsure")},
)
#: contents words → centre-of-mass prior: kh = height ratio (0 = base, 1 = top), cx = fore-aft offset as a fraction of L
#: (+ = toward the deck's front end).
TILT_FILLS = {
    "level": {"kh": (0.46, 0.54), "cx": (-0.03, 0.03)},
    "low": {"kh": (0.24, 0.38), "cx": (-0.04, 0.04)},
    "high": {"kh": (0.58, 0.72), "cx": (-0.04, 0.04)},
    "unsure": {"kh": (0.32, 0.68), "cx": (-0.12, 0.12)},
    "fore": {"kh": (0.42, 0.58), "cx": (0.06, 0.16)},
    "aft": {"kh": (0.42, 0.58), "cx": (-0.16, -0.06)},
}
TILT_ROBOTS = {
    "bed": ("courier robot with a tilting cargo bed", "yard robot with a dump bed", "lab cart with a tilting tray deck"),
    "ramp": ("hospital supply cart", "delivery robot", "low-deck inventory robot"),
}

#: grip: finger pads → friction prior with an ordinary matte surface; item surfaces scale it.
GRIP_PADS = {
    "silicone": {"text": "silicone pads", "mu": (0.80, 1.10)},
    "urethane": {"text": "polyurethane pads", "mu": (0.60, 0.85)},
    "aluminium": {"text": "bare aluminium jaws", "mu": (0.20, 0.32)},
    "nylon": {"text": "nylon fingertips", "mu": (0.25, 0.38)},
    "foam": {"text": "foam-lined pads", "mu": (0.45, 0.65)},
}
GRIP_SURFACES = {
    "matte": {"text": "matte cardboard", "factor": (0.95, 1.05)},
    "glossy": {"text": "glossy printed card", "factor": (0.75, 0.88)},
    "plastic": {"text": "smooth hard plastic", "factor": (0.62, 0.78)},
    "metal": {"text": "painted metal", "factor": (0.55, 0.70)},
    "rubber": {"text": "a rubberised coating", "factor": (1.20, 1.40)},
}
#: grip items: L along the pads, W between the pads (grip width), H up; mass statement: ("about", lo, hi) = a stated mass drawn in
#: [lo, hi] (hidden ±5 % around it) or ("range", lo, hi) = "between lo and hi, contents unknown" (hidden uniform).
GRIP_ITEMS = (
    {"key": "drive", "name": "boxed hard drive", "dims": ((0.10, 0.16), (0.04, 0.07), (0.14, 0.20)), "mass": ("about", 0.3, 0.9),
     "surfaces": ("matte", "glossy")},
    {"key": "tin", "name": "square tea tin", "dims": ((0.08, 0.11), (0.08, 0.11), (0.12, 0.18)), "mass": ("range", 0.15, 0.6),
     "surfaces": ("metal",)},
    {"key": "kettle", "name": "boxed kettle", "dims": ((0.20, 0.26), (0.16, 0.22), (0.24, 0.30)), "mass": ("about", 1.0, 2.0),
     "surfaces": ("glossy", "matte")},
    {"key": "battery", "name": "battery pack", "dims": ((0.10, 0.18), (0.05, 0.08), (0.12, 0.20)), "mass": ("about", 0.8, 2.5),
     "surfaces": ("plastic", "rubber")},
    {"key": "lunch", "name": "lunch box", "dims": ((0.16, 0.22), (0.10, 0.14), (0.08, 0.12)), "mass": ("range", 0.2, 1.4),
     "surfaces": ("plastic",)},
    {"key": "cartridges", "name": "pack of ink cartridges", "dims": ((0.12, 0.18), (0.05, 0.08), (0.10, 0.15)), "mass": ("about", 0.2, 0.6),
     "surfaces": ("glossy", "matte")},
    {"key": "sample", "name": "sample case", "dims": ((0.18, 0.26), (0.08, 0.12), (0.14, 0.20)), "mass": ("range", 0.6, 2.8),
     "surfaces": ("plastic", "metal", "rubber")},
    {"key": "bulbs", "name": "carton of light bulbs", "dims": ((0.14, 0.20), (0.08, 0.12), (0.12, 0.18)), "mass": ("about", 0.3, 0.8),
     "surfaces": ("matte",)},
)
GRIP_ROBOTS = ("ceiling-mounted picking arm", "mobile picker with a parallel gripper", "bench-top arm with a two-finger gripper")
GRIP_MOVES = ("lift", "lower", "swing")

#: tower boxes: L fore-aft, W, H; mass; contents words → fore-aft centre-of-mass prior (fraction of L, + = toward the front).
TOWER_BOXES = (
    {"key": "banker", "name": "banker's box", "dims": ((0.38, 0.44), (0.30, 0.34), (0.25, 0.29)), "mass": (3.0, 9.0)},
    {"key": "soup", "name": "case of canned soup", "dims": ((0.26, 0.32), (0.18, 0.22), (0.12, 0.16)), "mass": (5.0, 9.0)},
    {"key": "toner", "name": "toner carton", "dims": ((0.40, 0.50), (0.18, 0.24), (0.16, 0.22)), "mass": (2.0, 5.0)},
    {"key": "wine", "name": "wine case", "dims": ((0.30, 0.36), (0.24, 0.28), (0.30, 0.36)), "mass": (9.0, 16.0)},
    {"key": "tiles", "name": "carton of floor tiles", "dims": ((0.30, 0.62), (0.30, 0.40), (0.08, 0.14)), "mass": (10.0, 22.0)},
    {"key": "cereal", "name": "case of cereal boxes", "dims": ((0.36, 0.48), (0.26, 0.34), (0.28, 0.36)), "mass": (2.5, 6.0)},
    {"key": "cooler", "name": "foam cooler box", "dims": ((0.40, 0.55), (0.30, 0.38), (0.28, 0.36)), "mass": (1.0, 6.0)},
    {"key": "drawers", "name": "boxed set of plastic drawers", "dims": ((0.34, 0.44), (0.28, 0.36), (0.40, 0.55)), "mass": (3.0, 7.0)},
)
TOWER_CONTENTS = {
    "even": {"cx": (-0.02, 0.02)},
    "shifted": {"cx": (-0.12, 0.12)},
    "front": {"cx": (0.05, 0.15)},
    "back": {"cx": (-0.15, -0.05)},
}
TOWER_ROBOTS = ("palletising arm", "stockroom robot with a suction gripper", "wheeled manipulator")
#: names changed after the first labelling run because the overlap check found them in T8b's catalogs (a name is not physics;
#: records are written with the new name, :func:`robo_jev.mjgen.generate.group_records`)
ROBOT_RENAMES = {"mobile manipulator": "wheeled manipulator"}
TOWER_MU = (0.35, 0.65)

# --------------------------------------------------------------------------
# Seeds, hidden draws, priors
# --------------------------------------------------------------------------


def group_seed(master: int, family: str, index: int) -> int:
    """Seed of one group: the first 8 bytes of sha256("v03b|master|family|index")."""
    return int.from_bytes(hashlib.sha256(f"v03b|{int(master)}|{family}|{int(index)}".encode("utf-8")).digest()[:8], "big")


def unit_draws(names: Sequence[str], k: int, seed: int) -> np.ndarray:
    """K points of a scrambled Latin hypercube over the hidden names (sorted), seeded — shared by every prior and variant."""
    return qmc.LatinHypercube(d=len(names), seed=np.random.default_rng(seed % (2**63))).random(int(k))


def _scale(span: tuple[float, float], factor: float, *, lo: float = -math.inf, hi: float = math.inf) -> tuple[float, float]:
    """Widen/narrow a range around its centre by `factor`, clipped to [lo, hi]."""
    centre, half = (span[0] + span[1]) / 2, (span[1] - span[0]) / 2 * factor
    return (max(lo, centre - half), min(hi, centre + half))


def prior_ranges(hidden: dict[str, Any], setting: str) -> dict[str, Any]:
    """Base hidden specs → the specs of one prior setting. Spec kinds: ``("mu", lo, hi)`` friction (scaled), ``("u", lo, hi)``
    centre of mass / mass (spread around its centre), ``("n", mean, sd)`` placement error (sd scaled), ``("fixed", value)``."""
    s = PRIOR_SETTINGS[setting]
    out: dict[str, Any] = {}
    for name, spec in hidden.items():
        kind = spec[0]
        if kind == "mu":
            out[name] = ("u", spec[1] * s["friction"], spec[2] * s["friction"])
        elif kind == "u":
            lo, hi = _scale((spec[1], spec[2]), s["spread"], lo=spec[3] if len(spec) > 3 else -math.inf, hi=spec[4] if len(spec) > 4 else math.inf)
            out[name] = ("u", lo, hi)
        elif kind == "n":
            out[name] = ("n", spec[1], spec[2] * s["placement"])
        elif kind == "fixed":
            out[name] = ("fixed", spec[1])
        else:
            raise ValueError(f"hidden spec kind {kind!r}")
    return out


def draw_values(ranges: dict[str, Any], unit: np.ndarray, names: Sequence[str]) -> list[dict[str, float]]:
    """Map the shared unit points through one prior's specs (uniform: lo + (hi − lo)·u; normal: mean + sd·Φ⁻¹(u))."""
    out = []
    for row in unit:
        draw = {}
        for name, u in zip(names, row):
            spec = ranges[name]
            if spec[0] == "u":
                draw[name] = float(spec[1] + (spec[2] - spec[1]) * u)
            elif spec[0] == "n":
                draw[name] = float(spec[1] + spec[2] * norm.ppf(min(max(float(u), 1e-6), 1 - 1e-6)))
            else:
                draw[name] = float(spec[1])
        out.append(draw)
    return out


# --------------------------------------------------------------------------
# Groups and variants
# --------------------------------------------------------------------------


@dataclass
class Group:
    """One scene (``visible`` = everything the text states, ``hidden`` = base prior specs) with its ``variants`` (name → overrides of
    the visible facts to roll out), the decisive-number plan (``plan``: targets, the counterfactual, the action options), an optional
    one-number ``pair`` (``{"field", "value", "aim"}``) and the text choices (``style``, ``wording`` seed, ``mix`` of question
    kinds)."""

    family: str
    index: int
    seed: int
    visible: dict[str, Any]
    hidden: dict[str, Any]
    variants: dict[str, dict[str, Any]]
    plan: dict[str, Any]
    style: str
    wording: int
    mix: list[str]
    pair: dict[str, Any] | None = None
    notes: dict[str, Any] = field(default_factory=dict)

    @property
    def hidden_names(self) -> list[str]:
        return sorted(self.hidden)

    def visible_for(self, member: str) -> dict[str, Any]:
        """The stated facts of the base (``base``) or the paired (``pair``) record."""
        if member == "base" or self.pair is None:
            return dict(self.visible)
        out = dict(self.visible)
        _set_field(out, self.pair["field"], self.pair["value"])
        return out


def _set_field(visible: dict[str, Any], name: str, value: Any) -> None:
    """Set a visible fact; ``dims.<i>`` and ``boxes.<j>.<key>`` address nested values."""
    parts = name.split(".")
    if parts[0] == "dims":
        dims = list(visible["dims"])
        dims[int(parts[1])] = value
        visible["dims"] = dims
    elif parts[0] == "boxes":
        boxes = [dict(b) for b in visible["boxes"]]
        j, key = int(parts[1]), parts[2]
        if key.startswith("dims"):
            dims = list(boxes[j]["dims"])
            dims[int(key.split("_")[1])] = value
            boxes[j]["dims"] = dims
        else:
            boxes[j][key] = value
        visible["boxes"] = boxes
    else:
        visible[name] = value


def _get_field(visible: dict[str, Any], name: str) -> Any:
    parts = name.split(".")
    if parts[0] == "dims":
        return visible["dims"][int(parts[1])]
    if parts[0] == "boxes":
        box = visible["boxes"][int(parts[1])]
        key = parts[2]
        return box["dims"][int(key.split("_")[1])] if key.startswith("dims") else box[key]
    return visible[name]


def hidden_for(family: str, visible: dict[str, Any]) -> dict[str, Any]:
    """Base hidden specs implied by the stated facts (the words → ranges)."""
    if family == "tilt":
        fill = TILT_FILLS[visible["fill"]]
        mu = TILT_SURFACES[visible["surface"]]["mu"]
        if visible.get("wet"):
            mu = (round(mu[0] * WET_FACTOR[0], 4), round(mu[1] * WET_FACTOR[1], 4))
        return {"mu": ("mu",) + tuple(mu), "kh": ("u",) + fill["kh"] + (0.08, 0.92), "cx": ("u",) + fill["cx"] + (-0.40, 0.40)}
    if family == "grip":
        pad = GRIP_PADS[visible["pads"]]["mu"]
        factor = GRIP_SURFACES[visible["surface"]]["factor"]
        if visible.get("wet"):
            factor = (factor[0] * WET_FACTOR[0], factor[1] * WET_FACTOR[1])
        mass = (visible["mass"] * 0.95, visible["mass"] * 1.05) if visible["mass_kind"] == "about" else (visible["mass_lo"], visible["mass_hi"])
        return {"mu": ("mu", round(pad[0] * factor[0], 4), round(pad[1] * factor[1], 4)), "mass": ("u",) + tuple(mass) + (0.05, 50.0)}
    out: dict[str, Any] = {"mu": ("mu",) + TOWER_MU}
    for j, box in enumerate(visible["boxes"]):
        out[f"cx{j}"] = ("u",) + TOWER_CONTENTS[box["contents"]]["cx"] + (-0.40, 0.40)
        if j > 0:
            out[f"eps{j}"] = ("n", 0.0, visible["sigma"])
    return out


# --------------------------------------------------------------------------
# Static (closed-form) outcomes — used to aim the sampled values, never as labels
# --------------------------------------------------------------------------


def _tilt_geometry(visible: dict[str, Any], draw: dict[str, float]) -> tuple[float, float]:
    """(distance from the centre of mass to the downhill tipping edge, centre-of-mass height above the deck), in the deck frame."""
    frame = ph.orientation_frame(visible.get("orientation", "upright"))
    length, width, height = visible["dims"]
    com = frame @ np.array([draw["cx"] * length, 0.0, (draw["kh"] - 0.5) * height])
    half = np.abs(frame) @ (np.array([length, width, height]) / 2)
    downhill = -1.0 if visible["direction"] == "up" else 1.0  # nose up → the load's rear (−x) is downhill
    return float(half[0] - downhill * com[0]), float(half[2] + com[2])


def static_outcome(family: str, visible: dict[str, Any], draw: dict[str, float]) -> str:
    """Closed-form outcome of one hidden draw (quasi-static textbook rules; no dynamics)."""
    if family == "tilt":
        edge, height = _tilt_geometry(visible, draw)
        theta = math.radians(visible["angle"])
        tip_at = math.atan2(max(edge, 1e-6), height)
        slide_at = math.atan(draw["mu"])
        if theta <= min(tip_at, slide_at):
            return "stay"
        return "tip" if tip_at < slide_at else "slide"
    if family == "grip":
        need = draw["mass"] * _grip_load(visible["move"], visible["accel"])
        hold = 2 * draw["mu"] * visible["squeeze"]
        if hold >= need:
            return "hold"
        return "slip" if hold >= draw["mass"] * G else "drop"
    centres = tower_centres(visible, draw)
    return _tower_static(visible, draw, centres)


def _grip_load(move: str, accel: float) -> float:
    """Acceleration the grip must carry per kg at the move's critical phase: lift (speeding up) and lower (stopping) g + a,
    swing √(g² + a²)."""
    return math.hypot(G, accel) if move == "swing" else G + accel


def tower_centres(visible: dict[str, Any], draw: dict[str, float] | None) -> list[float]:
    """Fore-aft centres of the boxes (bottom at 0): each box at the stated offset from the one below plus its placement error."""
    centres = [0.0]
    for j in range(1, len(visible["boxes"])):
        eps = 0.0 if draw is None else draw.get(f"eps{j}", 0.0)
        centres.append(centres[-1] + visible["boxes"][j]["offset"] + eps)
    return centres


def _turn(box: dict[str, Any]) -> float:
    return -1.0 if box.get("turned") else 1.0


def _tower_static(visible: dict[str, Any], draw: dict[str, float], centres: list[float]) -> str:
    boxes = visible["boxes"]
    n = len(boxes)
    coms = [centres[j] + _turn(boxes[j]) * draw[f"cx{j}"] * boxes[j]["dims"][0] for j in range(n)]
    masses = [boxes[j]["mass"] for j in range(n)]
    failed = []
    for j in range(n - 1):  # interface between box j (below) and box j + 1
        lo = max(centres[j] - boxes[j]["dims"][0] / 2, centres[j + 1] - boxes[j + 1]["dims"][0] / 2)
        hi = min(centres[j] + boxes[j]["dims"][0] / 2, centres[j + 1] + boxes[j + 1]["dims"][0] / 2)
        above = range(j + 1, n)
        com = sum(coms[i] * masses[i] for i in above) / sum(masses[i] for i in above)
        failed.append(not (lo <= com <= hi))
    if not any(failed):
        return "stands"
    return "top" if failed[-1] and not any(failed[:-1]) else "more"


def static_event_rate(family: str, visible: dict[str, Any], draws: list[dict[str, float]]) -> float:
    """Share of draws whose static outcome is the key event (any failure)."""
    events = EVENT[family]
    return sum(1 for d in draws if static_outcome(family, visible, d) in events) / len(draws)


def solve_value(rate: Callable[[float], float], lo: float, hi: float, target: float, *, increasing: bool, iterations: int = 18) -> float:
    """Bisection for the value whose static event rate first reaches `target` (rate monotone in the value)."""
    a, b = float(lo), float(hi)
    for _ in range(iterations):
        mid = (a + b) / 2
        reached = rate(mid) >= target
        if increasing:
            a, b = (a, mid) if reached else (mid, b)
        else:
            a, b = (mid, b) if reached else (a, mid)
    return (a + b) / 2


# --------------------------------------------------------------------------
# Candidate groups
# --------------------------------------------------------------------------

STYLES = ("log", "brief", "sheet", "self")
#: question kinds per record — every record has the key yes/no and a counterfactual (direction or score); the rest is mixed
OPTIONAL_KINDS = ("outcome", "action", "compare", "score_cf", "severe")


def _target(rng: random.Random) -> float:
    """Aim of the key label: uniform on [0, 1] with 12 % clear-cut cases just beyond either end."""
    return rng.uniform(-0.06, 1.06)


def _second_target(rng: random.Random, first: float, *, near_share: float = 0.2) -> tuple[float, str]:
    """Aim of a counterfactual / paired value: usually far from `first` (|Δ| ≥ 0.3), sometimes close (|Δ| ≤ 0.06)."""
    first = min(1.0, max(0.0, first))
    if rng.random() < near_share:
        return min(1.0, max(0.0, first + rng.uniform(-0.06, 0.06))), "near"
    options = [t for t in (rng.uniform(0.0, first - 0.3) if first >= 0.3 else None, rng.uniform(first + 0.3, 1.0) if first <= 0.7 else None) if t is not None]
    return rng.choice(options), "far"


def _mix(rng: random.Random) -> list[str]:
    """Question kinds of one record: key + counterfactual direction + 2–3 optional kinds (seeded)."""
    picks = rng.sample(OPTIONAL_KINDS, rng.choice((2, 3)))
    return ["key", "direction"] + sorted(picks, key=OPTIONAL_KINDS.index)


def _round(value: float, step: float) -> float:
    return round(round(value / step) * step, 6)


def _tilt_candidate(group: dict[str, Any], rng: random.Random) -> dict[str, Any] | None:
    load = rng.choice(TILT_LOADS)
    storyline = rng.choice(("bed", "ramp"))
    dims = [round(rng.uniform(*span), 2) for span in load["dims"]]
    visible = {
        "robot": rng.choice(TILT_ROBOTS[storyline]), "storyline": storyline, "load": load["key"], "load_name": load["name"], "dims": dims,
        "mass": round(rng.uniform(*load["mass"]), 1), "fill": rng.choice(load["fills"]), "surface": rng.choice(sorted(TILT_SURFACES)),
        "direction": rng.choice(("up", "down")), "duration": round(rng.uniform(0.8, 2.5), 1), "orientation": "upright", "angle": 0.0,
        "domain": rng.choice(DOMAINS),
    }
    group["visible"] = visible
    return {"field": "angle", "lo": 0.5, "hi": 45.0, "step": 0.5, "increasing": True}


def _grip_candidate(group: dict[str, Any], rng: random.Random) -> dict[str, Any] | None:
    item = rng.choice(GRIP_ITEMS)
    dims = [round(rng.uniform(*span), 2) for span in item["dims"]]
    kind, lo, hi = item["mass"]
    visible = {
        "robot": rng.choice(GRIP_ROBOTS), "item": item["key"], "item_name": item["name"], "dims": dims, "surface": rng.choice(item["surfaces"]),
        "pads": rng.choice(sorted(GRIP_PADS)), "mass_kind": kind, "move": rng.choice(GRIP_MOVES), "distance": round(rng.uniform(0.2, 0.5), 2),
        "peak": round(rng.uniform(0.4, 1.0), 1), "accel": round(rng.uniform(1.0, 8.0), 1), "squeeze": 0.0, "domain": rng.choice(DOMAINS),
    }
    if kind == "about":
        visible["mass"] = round(rng.uniform(lo, hi), 2)
    else:
        visible["mass_lo"], visible["mass_hi"] = round(lo, 2), round(hi, 2)
    if visible["distance"] < visible["peak"] ** 2 / visible["accel"] + 0.02:
        visible["peak"] = round(max(0.2, math.sqrt(max(0.01, (visible["distance"] - 0.02) * visible["accel"]))), 1)
    group["visible"] = visible
    return {"field": "squeeze", "lo": 0.5, "hi": 120.0, "step": 0.5, "increasing": False}


def _tower_candidate(group: dict[str, Any], rng: random.Random) -> dict[str, Any] | None:
    n = rng.choice((2, 3, 3))
    picks = rng.sample(TOWER_BOXES, n)
    boxes = []
    for j, box in enumerate(picks):
        dims = [round(rng.uniform(*span), 2) for span in box["dims"]]
        boxes.append({"key": box["key"], "name": box["name"], "dims": dims, "mass": round(rng.uniform(*box["mass"]), 1),
                      "contents": rng.choice(sorted(TOWER_CONTENTS)), "offset": 0.0})
    # the middle box (three-box towers) sits at a stated offset (either way); the top box's offset is the aimed number
    if n == 3:
        reach = min(boxes[0]["dims"][0], boxes[1]["dims"][0]) / 2
        boxes[1]["offset"] = _round(rng.uniform(-0.3, 0.6) * reach, 0.005)
    visible = {"robot": rng.choice(TOWER_ROBOTS), "boxes": boxes, "sigma": rng.choice((0.005, 0.01, 0.015, 0.02)),
               "drop": rng.choice((0.005, 0.01, 0.02)), "domain": rng.choice(DOMAINS)}
    group["visible"] = visible
    top = boxes[-1]["dims"][0] / 2 + boxes[-2]["dims"][0] / 2
    return {"field": f"boxes.{n - 1}.offset", "lo": 0.005, "hi": 0.95 * top, "step": 0.005, "increasing": True}


def candidate(family: str, index: int, master: int, *, k: int = DEFAULT_K, pair_share: float = 0.5) -> Group:
    """Candidate group `index` of `family` (deterministic in (master, family, index, k))."""
    seed = group_seed(master, family, index)
    rng = random.Random(seed)
    builder = {"tilt": _tilt_candidate, "grip": _grip_candidate, "tower": _tower_candidate}[family]
    for _ in range(200):
        holder: dict[str, Any] = {}
        decisive = builder(holder, rng)
        if decisive is None:
            continue
        visible = holder["visible"]
        hidden = hidden_for(family, visible)
        names = sorted(hidden)
        unit = unit_draws(names, k, seed ^ 0xD3A5)
        draws = draw_values(prior_ranges(hidden, "base"), unit, names)

        def rate_at(value: float, fieldname: str = decisive["field"], base: dict[str, Any] = visible, fam: str = family) -> float:
            trial = dict(base)
            _set_field(trial, fieldname, value)
            if fam == "tower":
                trial["boxes"] = [dict(b) for b in trial["boxes"]]
            return static_event_rate(fam, trial, draws)

        aim_key = _target(rng)
        value = solve_value(rate_at, decisive["lo"], decisive["hi"], min(max(aim_key, 1e-3), 1.0), increasing=decisive["increasing"])
        if aim_key > 1.0:  # clear-cut: push past the last threshold
            value = value * (1.15 if decisive["increasing"] else 0.85)
        elif aim_key < 0.0:
            value = value * (0.85 if decisive["increasing"] else 1.15)
        value = _round(value, decisive["step"])
        if not decisive["lo"] <= value <= decisive["hi"]:
            continue
        _set_field(visible, decisive["field"], value)
        if family == "grip" and value < 1.0:
            continue
        aim_cf, cf_kind = _second_target(rng, aim_key)
        cf_value = _round(solve_value(rate_at, decisive["lo"], decisive["hi"], max(aim_cf, 1e-3), increasing=decisive["increasing"]), decisive["step"])
        if abs(cf_value - value) < decisive["step"] / 2:
            cf_value = _round(value + decisive["step"] * (1 if rng.random() < 0.5 else -1) * rng.choice((1, 2, 3)), decisive["step"])
        if not decisive["lo"] <= cf_value <= decisive["hi"] or (family == "grip" and cf_value < 1.0):
            continue
        group_out = Group(family=family, index=index, seed=seed, visible=visible, hidden=hidden, variants={}, plan={}, style=rng.choice(STYLES),
                          wording=rng.getrandbits(32), mix=_mix(rng))
        group_out.plan = {"decisive": decisive["field"], "aim_key": aim_key, "aim_cf": aim_cf, "cf_kind": cf_kind, "cf_value": cf_value}
        group_out.variants = {"planned": {}, "cf": {decisive["field"]: cf_value}}
        _options(group_out, rng)
        if rng.random() < pair_share:
            pair = _pair(group_out, rng, rate_at, decisive, draws)
            if pair is not None:
                group_out.pair = pair
        group_out.notes = {"static_key": rate_at(value), "static_cf": rate_at(cf_value)}
        return group_out
    raise RuntimeError(f"{family} #{index}: no valid scene in 200 tries")


def _options(group: Group, rng: random.Random) -> None:
    """Action options (a 'which is safest?' question and the comparisons): the planned action plus two alternatives."""
    v = group.visible
    if group.family == "tilt":
        group.variants.update({"opt_along": {"orientation": "along"}, "opt_across": {"orientation": "across"}})
        group.plan["options"] = {"upright": "planned", "lengthwise": "opt_along", "crosswise": "opt_across"}
    elif group.family == "grip":
        squeeze, accel = v["squeeze"], v["accel"]
        firmer = _round(min(120.0, squeeze * rng.uniform(1.3, 1.9)), 0.5)
        faster = _round(min(12.0, accel * rng.uniform(1.6, 2.6) + 1.0), 0.1)
        gentler = _round(max(0.5, accel * rng.uniform(0.25, 0.6)), 0.1)
        lighter = _round(max(1.0, squeeze * rng.uniform(0.6, 0.85)), 0.5)
        group.variants.update({"opt_firm_fast": {"squeeze": firmer, "accel": faster}, "opt_light_slow": {"squeeze": lighter, "accel": gentler}})
        group.plan["options"] = {"as_planned": "planned", "firmer_faster": "opt_firm_fast", "lighter_slower": "opt_light_slow"}
    else:
        boxes = v["boxes"]
        n = len(boxes)
        top_field = f"boxes.{n - 1}.offset"
        back = _round(boxes[-1]["offset"] - rng.uniform(0.3, 0.8) * boxes[-1]["dims"][0] / 2, 0.005)
        if n == 3:
            mid = _round(boxes[1]["offset"] + rng.choice((-1, 1)) * rng.uniform(0.25, 0.6) * boxes[1]["dims"][0] / 2, 0.005)
            out = _round(boxes[-1]["offset"] + rng.uniform(0.05, 0.3) * boxes[-1]["dims"][0] / 2, 0.005)
            group.variants.update({"opt_back": {top_field: back}, "opt_shift": {"boxes.1.offset": mid, top_field: out}})
        else:
            group.variants.update({"opt_back": {top_field: back}, "opt_shift": {"turn": True}})
        group.plan["options"] = {"as_planned": "planned", "top_back": "opt_back", "rearranged": "opt_shift"}
    group.plan["option_values"] = {name: dict(group.variants[name]) for name in group.plan["options"].values() if name != "planned"}


#: per family: the numbers a pair may change (field, share, kind) — ``aim`` = solved for a second target, ``free`` = a plain
#: different value (a number the outcome barely depends on).
PAIR_FIELDS = {
    "tilt": (("angle", 0.6, "aim"), ("dims.2", 0.25, "aim"), ("mass", 0.15, "free")),
    "grip": (("squeeze", 0.4, "aim"), ("accel", 0.35, "aim"), ("distance", 0.1, "free"), ("mass", 0.15, "aim")),
    "tower": (("top_offset", 0.6, "aim"), ("sigma", 0.15, "free"), ("top_mass", 0.25, "free")),
}


def _pair(group: Group, rng: random.Random, rate_at: Callable[..., float], decisive: dict[str, Any], draws: list[dict[str, float]]) -> dict[str, Any] | None:
    """A paired record's one changed number (``meta.pair``); None when no valid value is found."""
    fields = PAIR_FIELDS[group.family]
    roll = rng.random()
    acc = 0.0
    chosen = fields[-1]
    for entry in fields:
        acc += entry[1]
        if roll < acc:
            chosen = entry
            break
    name, _, how = chosen
    v = group.visible
    if group.family == "grip" and name == "mass" and v["mass_kind"] != "about":
        name, how = "accel", "aim"
    if group.family == "tower":
        n = len(v["boxes"])
        name = {"top_offset": f"boxes.{n - 1}.offset", "top_mass": f"boxes.{n - 1}.mass", "sigma": "sigma"}[name]
    current = _get_field(v, name)
    aim_key = min(1.0, max(0.0, group.plan["aim_key"]))
    aim_pair, closeness = _second_target(rng, aim_key)
    if how == "aim":
        spans = {"angle": (0.5, 45.0, 0.5, True), "dims.2": (max(0.08, current * 0.4), current * 2.2, 0.01, True), "squeeze": (1.0, 120.0, 0.5, False),
                 "accel": (0.3, 12.0, 0.1, True), "mass": (max(0.05, current * 0.3), current * 3.0, 0.01, True)}
        if name.startswith("boxes."):
            lo, hi = decisive["lo"], decisive["hi"]
            span = (lo, hi, 0.005, True)
        else:
            span = spans[name]
        lo, hi, step, increasing = span

        def rate(value: float) -> float:
            trial = dict(v)
            _set_field(trial, name, value)
            if name == "mass":  # a stated mass moves the hidden mass range with it
                trial_hidden = hidden_for(group.family, trial)
                trial_draws = draw_values(prior_ranges(trial_hidden, "base"), unit_draws(sorted(trial_hidden), len(draws), group.seed ^ 0xD3A5), sorted(trial_hidden))
                return static_event_rate(group.family, trial, trial_draws)
            return static_event_rate(group.family, trial, draws)

        value = _round(solve_value(rate, lo, hi, max(aim_pair, 1e-3), increasing=increasing), step)
        if abs(value - current) < step / 2:
            value = _round(current + step * rng.choice((-2, -1, 1, 2)), step)
        if not lo <= value <= hi:
            return None
    else:
        factor = rng.choice((0.6, 0.75, 1.3, 1.6))
        if name == "sigma":
            value = rng.choice([s for s in (0.005, 0.01, 0.015, 0.02, 0.03) if s != current])
        elif name == "distance":
            value = _round(min(0.8, max(0.1, current * factor)), 0.01)
            if value < v["peak"] ** 2 / v["accel"] + 0.02 or value == current:
                return None
        else:
            value = _round(current * factor, 0.1 if name == "mass" or name.endswith("mass") else 0.01)
        if value == current:
            return None
        closeness = "free"
    # the changed number must not collide with the counterfactual value the record asks about
    if name == group.plan["decisive"] and abs(value - group.plan["cf_value"]) < 1e-9:
        return None
    if group.family == "grip" and name in ("squeeze",) and value < 1.0:
        return None
    return {"field": name, "from": current, "value": value, "aim": aim_pair if how == "aim" else None, "closeness": closeness}


# --------------------------------------------------------------------------
# Factor pairs (controller update 2026-10-12, after E1): one physical factor changed
# --------------------------------------------------------------------------

#: the physical factor types each family varies in a one-factor pair (field, fact kind): friction as a named material or wet
#: surface, mass distribution, ramp-up time of the motion, drop height, footprint in the direction of the force
FACTORS = {
    "tilt": {"surface": ("surface", "word"), "wet": ("wet", "word"), "mass_distribution": ("fill", "word"), "footprint": ("dims.0", "number"),
             "ramp_time": ("duration", "number")},
    "grip": {"pads": ("pads", "word"), "item_surface": ("surface", "word"), "wet": ("wet", "word"), "ramp_time": ("jerk_time", "number")},
    "tower": {"mass_distribution": ("top.contents", "word"), "drop_height": ("drop", "number"), "footprint": ("below.dims_0", "number")},
}


def _factor_field(group: Group, field: str) -> str:
    n = len(group.visible["boxes"]) if group.family == "tower" else 0
    if field == "top.contents":
        return f"boxes.{n - 1}.contents"
    if field == "below.dims_0":
        return f"boxes.{n - 2}.dims_0"
    return field


def _static_rate_for(group: Group, visible: dict[str, Any]) -> float:
    """Static key-event rate of a stated scene under its own base prior, on the group's shared unit draws."""
    hidden = hidden_for(group.family, visible)
    names = sorted(hidden)
    draws = draw_values(prior_ranges(hidden, "base"), unit_draws(names, DEFAULT_K, group.seed ^ 0xD3A5), names)
    return static_event_rate(group.family, visible, draws)


def factor_pair(group: Group, rng: random.Random) -> dict[str, Any] | None:
    """A paired record that changes exactly one physical factor of the base scene (:data:`FACTORS`). Word factors (a material,
    wetness, a contents description) take the alternative whose static key rate moves furthest (a random one among those moving ≥ 0.25
    when there are several); number factors in the static model (footprint) are aimed at a far target; number factors outside it
    (ramp-up time, drop height) take a large change and MuJoCo decides how much it matters. None when the family offers no valid change."""
    family, v = group.family, group.visible
    names = list(FACTORS[family])
    rng.shuffle(names)
    base_rate = _static_rate_for(group, v)
    for factor in names:
        field_spec, fact = FACTORS[family][factor]
        field = _factor_field(group, field_spec)
        current = bool(v.get("wet")) if field == "wet" else (v.get("jerk_time", GRIP_JERK) if field == "jerk_time" else _get_field(v, field))
        candidates: list[Any] = []
        if field == "surface" and family == "tilt":
            candidates = [k for k in sorted(TILT_SURFACES) if k != current]
        elif field == "surface":
            item = next(i for i in GRIP_ITEMS if i["key"] == v["item"])
            candidates = [k for k in sorted(GRIP_SURFACES) if k != current and k in item["surfaces"] + ("rubber", "glossy", "plastic")]
        elif field == "pads":
            candidates = [k for k in sorted(GRIP_PADS) if k != current]
        elif field == "wet":
            candidates = [not current]
        elif field == "fill":
            load = next(item for item in TILT_LOADS if item["key"] == v["load"])
            candidates = [k for k in load["fills"] if k != current]
        elif field.endswith(".contents"):
            candidates = [k for k in sorted(TOWER_CONTENTS) if k != current]
        if candidates:
            scored = []
            for value in candidates:
                trial = copy.deepcopy(v)
                _set_field(trial, field, value)
                scored.append((abs(_static_rate_for(group, trial) - base_rate), value))
            movers = [value for delta, value in scored if delta >= 0.25]
            value = rng.choice(movers) if movers else max(scored, key=lambda dv: dv[0])[1]
        elif field == "dims.0":  # tilt footprint along the slope: aim the static rate at a far target
            aim, _ = _second_target(rng, base_rate, near_share=0.0)

            def rate(length: float) -> float:
                trial = copy.deepcopy(v)
                _set_field(trial, field, length)
                return _static_rate_for(group, trial)

            lo, hi = max(0.06, current * 0.4), current * 2.5
            increasing = rate(lo) <= rate(hi)
            value = _round(solve_value(rate, lo, hi, max(aim, 1e-3), increasing=increasing), 0.01)
            if abs(value - current) < 0.015:
                continue
        elif field.endswith(".dims_0"):  # tower: length of the box under the top box (its edge carries the top box)
            def rate(length: float) -> float:
                trial = copy.deepcopy(v)
                _set_field(trial, field, length)
                return _static_rate_for(group, trial)

            aim, _ = _second_target(rng, base_rate, near_share=0.0)
            lo, hi = max(0.12, current * 0.6), current * 1.6
            value = _round(solve_value(rate, lo, hi, max(aim, 1e-3), increasing=False), 0.01)
            if abs(value - current) < 0.015:
                continue
        elif field == "duration":  # tilt: how fast the pitch is reached
            value = _round(min(6.0, max(0.3, current * rng.choice((0.3, 0.4, 2.5, 3.0)))), 0.1)
        elif field == "jerk_time":  # grip: how fast the acceleration builds up (the profile honours it up to peak / accel)
            longest = v["peak"] / v["accel"]
            if longest < 0.1:
                continue
            value = _round(longest * rng.uniform(0.6, 1.0), 0.01)
        elif field == "drop":
            value = rng.choice([d for d in (0.04, 0.06, 0.08) if d != current])
        else:
            continue
        if value == current:
            continue
        return {"field": field, "from": current, "value": value, "aim": None, "closeness": "factor", "factor": factor, "fact": fact}
    return None


# --------------------------------------------------------------------------
# MuJoCo labels
# --------------------------------------------------------------------------


def _effective(group: Group, member: str, overrides: dict[str, Any]) -> dict[str, Any]:
    visible = group.visible_for(member)
    for name, value in overrides.items():
        if name == "turn":
            boxes = [dict(b) for b in visible["boxes"]]
            boxes[-1]["turned"] = bool(value)  # the top box placed end for end: its fore-aft centre-of-mass offset changes sign
            visible["boxes"] = boxes
            continue
        _set_field(visible, name, value)
    return visible


def _physics_key(family: str, visible: dict[str, Any]) -> str:
    keys = {"tilt": ("dims", "mass", "orientation", "angle", "duration", "direction"),
            "grip": ("dims", "move", "distance", "peak", "accel", "squeeze", "jerk_time"),
            "tower": ("boxes", "drop")}[family]
    payload = {k: visible.get(k) for k in keys}
    if family == "tower":
        payload["boxes"] = [{k: b.get(k) for k in ("dims", "mass", "offset", "turned")} for b in visible["boxes"]]
    return json.dumps(payload, sort_keys=True)


def roll(family: str, visible: dict[str, Any], draw: dict[str, float], rigs: dict[str, Any]) -> dict[str, Any]:
    """One MuJoCo rollout of a stated scene under one hidden draw."""
    if family == "tilt":
        key = json.dumps(["tilt", visible["dims"], visible["mass"], visible["orientation"]])
        if key not in rigs:
            rigs[key] = ph.tilt_rig(visible["dims"], visible["mass"], visible["orientation"])
        sign = -1.0 if visible["direction"] == "up" else 1.0
        path = ph.pitch_profile(sign * visible["angle"], visible["duration"])
        length, _, height = visible["dims"]
        com = (draw["cx"] * length, 0.0, (draw["kh"] - 0.5) * height)
        return ph.run_tilt(rigs[key], path, mu=draw["mu"], com=com, orientation=visible["orientation"])
    if family == "grip":
        key = json.dumps(["grip", visible["dims"]])
        if key not in rigs:
            rigs[key] = ph.grip_rig(visible["dims"], 1.0)
        path, axis = ph.move_profile(visible["move"], visible["distance"], visible["peak"], visible["accel"], jerk_time=visible.get("jerk_time", GRIP_JERK))
        return ph.run_grip(rigs[key], path, axis, mu=draw["mu"], mass=draw["mass"], squeeze=visible["squeeze"])
    boxes = visible["boxes"]
    key = json.dumps(["tower", [(b["dims"], b["mass"]) for b in boxes]])
    if key not in rigs:
        rigs[key] = ph.tower_rig([(b["dims"], b["mass"]) for b in boxes])
    centres = tower_centres(visible, draw)
    coms = [(_turn(boxes[j]) * draw[f"cx{j}"] * boxes[j]["dims"][0], 0.0, 0.0) for j in range(len(boxes))]
    return ph.run_tower(rigs[key], centres=centres, drop=visible["drop"], mu=draw["mu"], coms=coms)


def label_group(group: Group, k: int = DEFAULT_K, *, rigs: dict[str, Any] | None = None, only: Sequence[str] | None = None) -> dict[str, Any]:
    """Roll out every variant of both members under all three priors on the group's shared unit draws.

    Returns ``{"k", "draws_seed", "members": {member: {"variants": {name: {"outcomes": {prior: [outcome per draw]}, "freq": {prior:
    {outcome: share}}}}}}, "rollouts", "flags"}``. Identical (scene, prior) specs are rolled once (the pair's counterfactual usually
    equals the base's)."""
    rigs = {} if rigs is None else rigs
    family = group.family
    draws_seed = group.seed ^ 0xD3A5
    outcomes_list = OUTCOMES[family]
    cache: dict[str, list[dict[str, Any]]] = {}
    members = ("base", "pair") if group.pair is not None else ("base",)
    if only is not None:  # e.g. only the new factor partner (the base member's rollouts are already labelled)
        members = tuple(m for m in members if m in only)
    out: dict[str, Any] = {"k": int(k), "draws_seed": int(draws_seed), "members": {}, "rollouts": 0, "flags": {"unstable": 0}}
    for member in members:
        visible_member = group.visible_for(member)
        hidden = hidden_for(family, visible_member)
        names = sorted(hidden)
        unit = unit_draws(names, k, draws_seed)
        entry: dict[str, Any] = {"hidden": {p: prior_ranges(hidden, p) for p in PRIORS}, "variants": {}}
        for name, overrides in group.variants.items():
            visible = _effective(group, member, overrides)
            per_prior: dict[str, list[str]] = {}
            freq: dict[str, dict[str, float]] = {}
            for prior in PRIORS:
                ranges = entry["hidden"][prior]
                key = _physics_key(family, visible) + "|" + json.dumps(ranges, sort_keys=True)
                if key not in cache:
                    draws = draw_values(ranges, unit, names)
                    cache[key] = [roll(family, visible, d, rigs) for d in draws]
                    out["rollouts"] += len(draws)
                    out["flags"]["unstable"] += sum(1 for r in cache[key] if r.get("unstable"))
                rolls = cache[key]
                per_prior[prior] = [r["outcome"] for r in rolls]
                freq[prior] = {o: sum(1 for r in rolls if r["outcome"] == o) / len(rolls) for o in outcomes_list}
            entry["variants"][name] = {"outcomes": per_prior, "freq": freq}
        out["members"][member] = entry
    return out
