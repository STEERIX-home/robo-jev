"""T8b — robot moments labelled by MuJoCo (Task T8 §1-2).

A *moment* is one decision request: what the robot can see (the visible scene) written as text, plus 3–4 questions.
What the robot cannot see — friction, where the weight sits inside the load, how exactly a release lands — is drawn
**K times** from a prior that matches the words in the text, every draw is rolled out in MuJoCo
(:mod:`robo_jev.outside.physics`), and each label is the outcome frequency over the K rollouts: the posterior idea of our
engine, but with real rigid-body dynamics. Hidden draws are a Latin hypercube (stratified, seeded per moment) and all the
questions of a moment share the same K draws.

Families (``meta.scene``):

* ``deck_brake`` / ``deck_turn`` — a load standing on the robot's deck while it brakes to a stop or takes a 90° curve.
  Questions: falls over? (noul), stays put? (noul), what happens — stays / slides / topples (choice, a posterior over
  outcomes), and how likely it falls over under a different braking rate or curve speed (score).
* ``stack`` — a box released on top of a fixed support with its far end sticking out past the support's edge.
  Questions: falls off? (noul), how likely it falls off if it stuck out further (score), would it fall off if it stuck out
  less (noul).
* ``push`` — a box on the floor pushed by the robot's bumper bar at a chosen height. Questions: tips over instead of
  sliding? (noul), would it tip at another height? (noul), how likely it tips at a third height (score).

**Text.** Written for T8 in a first-person robot voice with its own wording; nothing is imported or copied from
:mod:`robo_jev.simgen` (phrase banks, narrators, skins). Every number the physics needs is in the text; the hidden values
are not (only words that set their prior: the deck surface, "evenly packed", "heavier near the top", …).

**Sampling and balance.** Visible values are drawn around the quasi-static decision boundary of each family (so many
moments are uncertain), candidates are labelled, and the final set keeps equal numbers of moments per majority outcome
of the planned action within each family (:func:`select_balanced`).
"""

from __future__ import annotations

import hashlib
import math
import random
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import numpy as np
from scipy.stats import qmc

from robo_jev.decisions.records import SCHEMA, validate_record
from robo_jev.outside import physics as ph

__all__ = [
    "DEFAULT_K",
    "FAMILIES",
    "GENERATOR",
    "OUTCOMES",
    "Moment",
    "candidate",
    "hidden_draws",
    "label_moment",
    "moment_record",
    "moment_seed",
    "select_balanced",
]

GENERATOR = "t8b-mujoco-v1"
DEFAULT_K = 64
FAMILIES = ("deck_brake", "deck_turn", "stack", "push")
OUTCOMES = {"deck_brake": ("stay", "slide", "tip"), "deck_turn": ("stay", "slide", "tip"), "stack": ("stay", "fall"), "push": ("slide", "tip")}
G = ph.GRAVITY

# --------------------------------------------------------------------------
# Catalogs (T8's own; the ranges are the priors behind the words)
# --------------------------------------------------------------------------

#: Deck surfaces → friction prior (uniform) between the surface and the loads in this set.
DECK_SURFACES = {
    "rubber_mat": {"text": "a ribbed rubber mat", "mu": (0.65, 0.95)},
    "painted_steel": {"text": "bare painted steel", "mu": (0.25, 0.40)},
    "plastic_liner": {"text": "a smooth plastic liner", "mu": (0.15, 0.28)},
    "plywood": {"text": "plain plywood", "mu": (0.35, 0.50)},
    "foam_pad": {"text": "an anti-slip foam pad", "mu": (0.90, 1.20)},
}
#: Deck loads (all box-shaped — see physics: no cylinders): size ranges L × W × H (along travel × across × up; ``square`` = D × D × H
#: from ranges (D, H)), mass range (kg), allowed fills.
DECK_LOADS = (
    {"key": "carton", "name": "cardboard carton", "dims": ((0.15, 0.45), (0.15, 0.45), (0.20, 0.80)), "mass": (1.0, 9.0),
     "fills": ("even", "unknown", "bottom_heavy", "top_heavy")},
    {"key": "tote", "name": "plastic tote", "dims": ((0.30, 0.55), (0.25, 0.40), (0.15, 0.40)), "mass": (1.5, 12.0),
     "fills": ("even", "unknown", "bottom_heavy")},
    {"key": "crate", "name": "slatted wooden crate", "dims": ((0.25, 0.50), (0.25, 0.50), (0.25, 0.60)), "mass": (3.0, 15.0),
     "fills": ("even", "unknown", "top_heavy")},
    {"key": "parts_bin", "name": "tall parts bin", "dims": ((0.12, 0.25), (0.12, 0.25), (0.45, 0.90)), "mass": (2.0, 8.0),
     "fills": ("unknown", "bottom_heavy", "top_heavy")},
    {"key": "tin", "name": "square metal tin", "square": True, "dims": ((0.10, 0.30), (0.20, 0.60)), "mass": (1.0, 8.0),
     "fills": ("even", "unknown", "bottom_heavy")},
    {"key": "planter", "name": "potted plant in a square planter", "square": True, "dims": ((0.18, 0.35), (0.40, 0.90)), "mass": (3.0, 12.0),
     "fills": ("planter",)},
    {"key": "bottle", "name": "tall square glass bottle, full and capped", "square": True, "dims": ((0.06, 0.10), (0.25, 0.34)),
     "mass": (0.8, 1.6), "fills": ("even",)},
    {"key": "jerrycan", "name": "full rectangular jerrycan", "dims": ((0.14, 0.20), (0.28, 0.36), (0.35, 0.47)), "mass": (12.0, 22.0),
     "fills": ("even",)},
)
#: Mass distribution → prior of the centre-of-mass height ratio ``kh`` (0 = base, 1 = top) and of the horizontal offset ``cx`` as a
#: fraction of the base dimension toward the tipping edge (deck) — symmetric, so the direction of the move does not matter.
FILLS = {
    "even": {"kh": (0.45, 0.55), "cx": (-0.04, 0.04),
             "text": ("Its weight looks evenly spread.", "It appears evenly packed.")},
    "unknown": {"kh": (0.30, 0.70), "cx": (-0.15, 0.15),
                "text": ("I cannot tell how its contents are arranged.", "Where the weight sits inside it is not something I can see.")},
    "bottom_heavy": {"kh": (0.20, 0.35), "cx": (-0.05, 0.05),
                     "text": ("Its heaviest contents sit near the bottom.", "Most of its weight is packed low down.")},
    "top_heavy": {"kh": (0.60, 0.78), "cx": (-0.05, 0.05),
                  "text": ("It is heavier near the top than near the bottom.", "Its heavier contents ride up high.")},
    "planter": {"kh": (0.22, 0.38), "cx": (-0.05, 0.05),
                "text": ("The pot is full of damp soil, so most of the weight is low down.",)},
}
DECK_ROBOTS = ("flatbed delivery cart", "warehouse tugger with an open load deck", "hospital supply robot with a flat cargo deck",
               "low mobile platform")

#: Stack: the box being placed, its size range (L along the placement × W × H) and mass; supports with top length and friction prior.
STACK_ITEMS = (
    {"key": "toolbox", "name": "steel toolbox", "dims": ((0.35, 0.55), (0.18, 0.25), (0.15, 0.25)), "mass": (3.0, 9.0)},
    {"key": "shoebox", "name": "shoebox", "dims": ((0.28, 0.36), (0.17, 0.22), (0.10, 0.14)), "mass": (0.6, 1.6)},
    {"key": "parcel", "name": "taped parcel", "dims": ((0.25, 0.60), (0.20, 0.40), (0.10, 0.30)), "mass": (0.8, 6.0)},
    {"key": "paper", "name": "box of printer paper", "dims": ((0.42, 0.46), (0.28, 0.31), (0.24, 0.27)), "mass": (10.0, 12.5)},
    {"key": "storage", "name": "lidded plastic storage box", "dims": ((0.40, 0.60), (0.28, 0.40), (0.15, 0.30)), "mass": (1.0, 7.0)},
)
STACK_SPREADS = {
    "even": {"cx": (-0.03, 0.03), "text": "Its weight looks evenly spread from end to end."},
    "outer_heavy": {"cx": (0.06, 0.22), "text": "It is heavier at the end that will stick out."},
    "inner_heavy": {"cx": (-0.22, -0.06), "text": "It is heavier at the end that will rest on top."},
    "unknown": {"cx": (-0.15, 0.15), "text": "Its contents slide around, so the weight could sit toward either end."},
}
STACK_SUPPORTS = (
    {"key": "cabinet", "name": "a steel filing cabinet", "short": "cabinet", "top": (0.45, 0.65), "height": (0.70, 1.35), "mu": (0.28, 0.42)},
    {"key": "pallets", "name": "a stack of wooden pallets", "short": "pallets", "top": (0.80, 1.20), "height": (0.30, 0.60), "mu": (0.40, 0.60)},
    {"key": "bench", "name": "a wooden workbench", "short": "workbench", "top": (0.60, 1.20), "height": (0.75, 0.95), "mu": (0.35, 0.50)},
    {"key": "carton", "name": "a closed cardboard carton", "short": "carton", "top": (0.40, 0.60), "height": (0.30, 0.60), "mu": (0.45, 0.60)},
)
STACK_ROBOTS = ("pick-and-place arm", "shelf-stocking robot arm", "mobile manipulator")

#: Push: boxes (L along the push × W × H), floors (friction prior with these boxes), mass distribution (cx toward the far side, kh).
PUSH_BOXES = (
    {"key": "moving_box", "name": "taped moving box", "dims": ((0.35, 0.55), (0.30, 0.50), (0.30, 0.75)), "mass": (4.0, 15.0)},
    {"key": "bottle_crate", "name": "plastic crate of bottles", "dims": ((0.30, 0.40), (0.25, 0.35), (0.25, 0.35)), "mass": (6.0, 14.0)},
    {"key": "cabinet", "name": "small wooden cabinet", "dims": ((0.30, 0.45), (0.40, 0.60), (0.60, 1.00)), "mass": (10.0, 25.0)},
    {"key": "carton_stack", "name": "two cartons taped one on top of the other", "dims": ((0.30, 0.45), (0.30, 0.45), (0.60, 0.90)),
     "mass": (5.0, 16.0)},
)
PUSH_FLOORS = {
    "concrete": {"text": "polished concrete", "mu": (0.35, 0.50)},
    "vinyl": {"text": "smooth vinyl flooring", "mu": (0.25, 0.40)},
    "rubber": {"text": "rubber floor matting", "mu": (0.70, 1.00)},
    "carpet": {"text": "low-pile carpet", "mu": (0.50, 0.70)},
    "epoxy": {"text": "an epoxy-coated floor", "mu": (0.30, 0.45)},
}
PUSH_SPREADS = {
    "even": {"cx": (-0.03, 0.03), "kh": (0.45, 0.55), "text": "Its weight seems evenly spread."},
    "unknown": {"cx": (-0.12, 0.12), "kh": (0.30, 0.70), "text": "I do not know how the weight is arranged inside."},
    "far_heavy": {"cx": (0.05, 0.18), "kh": (0.40, 0.60), "text": "It is heavier on the side facing away from me."},
    "near_heavy": {"cx": (-0.18, -0.05), "kh": (0.40, 0.60), "text": "It is heavier on the side facing me."},
    "bottom_heavy": {"cx": (-0.03, 0.03), "kh": (0.20, 0.35), "text": "Its heaviest part is at the bottom."},
}
PUSH_ROBOTS = ("warehouse robot", "floor-cleaning robot", "service robot")
PUSH_TRAVEL = 0.40
DOMAINS = ("warehouse", "hospital", "office", "home", "retail", "lab")


# --------------------------------------------------------------------------
# Seeds and hidden draws
# --------------------------------------------------------------------------


def moment_seed(master: int, family: str, index: int) -> int:
    """Seed of one candidate moment: the first 8 bytes of sha256("t8b|master|family|index")."""
    return int.from_bytes(hashlib.sha256(f"t8b|{int(master)}|{family}|{int(index)}".encode("utf-8")).digest()[:8], "big")


def hidden_draws(ranges: dict[str, tuple[float, float] | tuple[str, float, float]], k: int, seed: int) -> list[dict[str, float]]:
    """K stratified draws (Latin hypercube, scrambled, seeded): ``(lo, hi)`` = uniform; ``("normal", mean, sd)`` = normal."""
    names = sorted(ranges)
    unit = qmc.LatinHypercube(d=len(names), seed=np.random.default_rng(seed % (2**63))).random(int(k))
    out = []
    for row in unit:
        draw = {}
        for name, u in zip(names, row):
            spec = ranges[name]
            if spec[0] == "normal":
                from scipy.stats import norm

                draw[name] = float(spec[1] + spec[2] * norm.ppf(min(max(u, 1e-6), 1 - 1e-6)))
            else:
                lo, hi = float(spec[0]), float(spec[1])
                draw[name] = lo + (hi - lo) * float(u)
        out.append(draw)
    return out


def _uniform(rng: random.Random, span: Sequence[float], digits: int = 2) -> float:
    return round(rng.uniform(float(span[0]), float(span[1])), digits)


def _mid(span: Sequence[float]) -> float:
    return (float(span[0]) + float(span[1])) / 2


# --------------------------------------------------------------------------
# Candidates (visible scene + hidden priors + variants to roll out)
# --------------------------------------------------------------------------


@dataclass
class Moment:
    """One candidate moment: ``family``, ``seed``, ``target`` (the outcome the sampler aimed at), ``visible`` (everything the text
    states), ``hidden`` (prior ranges) and ``variants`` (name → what to roll out; ``planned`` is the planned action)."""

    family: str
    index: int
    seed: int
    target: str
    visible: dict[str, Any]
    hidden: dict[str, Any]
    variants: dict[str, dict[str, Any]]


def _deck_candidate(family: str, index: int, seed: int, target: str, rng: random.Random) -> Moment | None:
    load = rng.choice(DECK_LOADS)
    fill = rng.choice(load["fills"])
    surface_key = rng.choice(sorted(DECK_SURFACES))
    surface = DECK_SURFACES[surface_key]
    if load.get("square"):
        side, height = (_uniform(rng, span) for span in load["dims"])
        dims = (side, side, height)
    else:
        dims = tuple(_uniform(rng, span) for span in load["dims"])
    base = dims[1] if family == "deck_turn" else dims[0]
    height = dims[2]
    prior = FILLS[fill]
    a_tip = G * (base / 2 - _mid(prior["cx"]) * base) / (_mid(prior["kh"]) * height)
    a_slide = G * _mid(surface["mu"])
    if target == "stay":
        accel = rng.uniform(0.3, 0.9) * min(a_tip, a_slide)
    elif target == "slide":
        if a_slide > 0.85 * a_tip:
            return None
        accel = rng.uniform(1.1, 2.2) * a_slide
    else:
        if a_tip > 0.85 * a_slide:
            return None
        accel = rng.uniform(1.15, 2.6) * a_tip
    if not 0.3 <= accel <= 7.0:
        return None
    mass = _uniform(rng, load["mass"], 1)
    visible: dict[str, Any] = {"robot": rng.choice(DECK_ROBOTS), "load": load["key"], "load_name": load["name"], "shape": "box",
                               "dims": list(dims), "mass": mass, "fill": fill, "surface": surface_key, "domain": rng.choice(DOMAINS)}
    hidden = {"mu": surface["mu"], "kh": prior["kh"], "cx": prior["cx"]}
    if family == "deck_brake":
        speed = round(rng.uniform(1.0, 2.2) if target == "tip" else rng.uniform(0.6, 2.0), 2)
        decel = round(accel, 2)
        factor = rng.choice((0.5, 0.7, 1.4, 1.8))
        other = round(min(7.5, max(0.3, decel * factor)), 2)
        visible.update({"speed": speed, "decel": decel, "jerk_time": round(min(0.15, 0.5 * speed / decel), 3), "other_decel": other})
        variants = {"planned": {"kind": "brake", "speed": speed, "decel": decel}, "other": {"kind": "brake", "speed": speed, "decel": other}}
    else:
        speed = round(rng.uniform(0.6, 2.0), 2)
        radius = speed**2 / accel
        if not 0.25 <= radius <= 4.0:
            speed = round(min(2.2, max(0.5, math.sqrt(accel * rng.uniform(0.4, 2.5)))), 2)
            radius = speed**2 / accel
            if not 0.25 <= radius <= 4.0:
                return None
        radius = round(radius, 2)
        direction = rng.choice((1, -1))
        factor = rng.choice((0.75, 0.85, 1.15, 1.3))
        other = round(min(2.6, max(0.3, speed * factor)), 2)
        # steering ramp: 0.2 s, shortened for tight fast curves so the 90° turn still has a constant-rate part (both variants alike)
        ramp = round(min(0.2, 0.4 * (math.pi / 2) * radius / max(speed, other)), 2)
        visible.update({"speed": speed, "radius": radius, "direction": "left" if direction == 1 else "right", "angle_deg": 90,
                        "steer_ramp": ramp, "other_speed": other})
        variants = {"planned": {"kind": "turn", "speed": speed, "radius": radius, "direction": direction, "ramp": ramp},
                    "other": {"kind": "turn", "speed": other, "radius": radius, "direction": direction, "ramp": ramp}}
    return Moment(family, index, seed, target, visible, hidden, variants)


def _stack_candidate(index: int, seed: int, target: str, rng: random.Random) -> Moment | None:
    item = rng.choice(STACK_ITEMS)
    spread_key = rng.choice(sorted(STACK_SPREADS))
    spread = STACK_SPREADS[spread_key]
    support = rng.choice(STACK_SUPPORTS)
    length, width, height = (_uniform(rng, span) for span in item["dims"])
    top = _uniform(rng, support["top"])
    support_height = _uniform(rng, support["height"])
    # nominal position of the centre of mass beyond the edge: e = overhang − L/2 + cx·L
    e = rng.uniform(-0.18, -0.01) * length if target == "stay" else rng.uniform(0.0, 0.12) * length
    overhang = round(e + length / 2 - _mid(spread["cx"]) * length, 3)
    if overhang > length - 0.04 or overhang < -0.05 or overhang < length - top + 0.02:
        return None
    sigma = rng.choice((0.005, 0.01, 0.015))
    drop = rng.choice((0.005, 0.01, 0.02, 0.03))
    delta_out = rng.choice((0.02, 0.03, 0.04, 0.06))
    delta_in = rng.choice((0.02, 0.03, 0.04, 0.06))
    out_more = round(overhang + delta_out, 3)
    in_more = round(overhang - delta_in, 3)
    if out_more > length - 0.02:
        return None
    visible = {"robot": rng.choice(STACK_ROBOTS), "item": item["key"], "item_name": item["name"], "dims": [length, width, height],
               "mass": _uniform(rng, item["mass"], 1), "spread": spread_key, "support": support["key"], "support_name": support["name"],
               "support_short": support["short"], "support_top": top, "support_height": support_height, "overhang": overhang,
               "sigma": sigma, "drop": drop, "overhang_out": out_more, "overhang_in": in_more, "domain": rng.choice(DOMAINS)}
    hidden = {"mu": support["mu"], "cx": spread["cx"], "kh": (0.42, 0.58), "eps": ("normal", 0.0, sigma), "tilt0": ("normal", 0.0, 1.0)}
    variants = {"planned": {"overhang": overhang}, "out": {"overhang": out_more}, "in": {"overhang": in_more}}
    return Moment("stack", index, seed, target, visible, hidden, variants)


def _push_candidate(index: int, seed: int, target: str, rng: random.Random) -> Moment | None:
    box = rng.choice(PUSH_BOXES)
    floor_key = rng.choice(sorted(PUSH_FLOORS))
    floor = PUSH_FLOORS[floor_key]
    spread_key = rng.choice(sorted(PUSH_SPREADS))
    spread = PUSH_SPREADS[spread_key]
    length, width, height = (_uniform(rng, span) for span in box["dims"])
    threshold = (length / 2 - _mid(spread["cx"]) * length) / _mid(floor["mu"])
    planned = threshold * (rng.uniform(1.15, 1.8) if target == "tip" else rng.uniform(0.35, 0.87))
    planned = round(planned, 2)
    low, high = 0.05, round(height - 0.04, 2)
    if not low <= planned <= high:
        return None

    def other(step_choices: Sequence[float]) -> float | None:
        for _ in range(8):
            value = round(planned + rng.choice((-1, 1)) * rng.choice(step_choices), 2)
            if low <= value <= high and value != planned:
                return value
        return None

    alt_noul, alt_score = other((0.05, 0.08, 0.12, 0.18)), other((0.05, 0.08, 0.12, 0.18))
    if alt_noul is None or alt_score is None:
        return None
    speed = round(rng.uniform(0.10, 0.35), 2)
    visible = {"robot": rng.choice(PUSH_ROBOTS), "box": box["key"], "box_name": box["name"], "dims": [length, width, height],
               "mass": _uniform(rng, box["mass"], 1), "floor": floor_key, "spread": spread_key, "speed": speed, "travel": PUSH_TRAVEL,
               "push_height": planned, "push_height_noul": alt_noul, "push_height_score": alt_score, "domain": rng.choice(DOMAINS)}
    hidden = {"mu": floor["mu"], "cx": spread["cx"], "kh": spread["kh"]}
    variants = {"planned": {"height": planned}, "alt_noul": {"height": alt_noul}, "alt_score": {"height": alt_score}}
    return Moment("push", index, seed, target, visible, hidden, variants)


def candidate(family: str, index: int, master: int) -> Moment:
    """Candidate `index` of `family`: the target outcome cycles through the family's outcomes (balanced aim), and the sampler retries
    (same seed stream) until the visible values satisfy the family's constraints."""
    seed = moment_seed(master, family, index)
    rng = random.Random(seed)
    target = OUTCOMES[family][index % len(OUTCOMES[family])]
    for _ in range(200):
        if family in ("deck_brake", "deck_turn"):
            found = _deck_candidate(family, index, seed, target, rng)
        elif family == "stack":
            found = _stack_candidate(index, seed, target, rng)
        elif family == "push":
            found = _push_candidate(index, seed, target, rng)
        else:
            raise ValueError(f"family: {list(FAMILIES)} (got {family!r})")
        if found is not None:
            return found
    raise RuntimeError(f"{family} #{index}: no valid visible scene in 200 tries")


# --------------------------------------------------------------------------
# Labels (MuJoCo)
# --------------------------------------------------------------------------


def _com(family: str, visible: dict[str, Any], draw: dict[str, float]) -> tuple[float, float, float]:
    """Hidden draw → centre-of-mass offset from the load's geometric centre (body frame, metres)."""
    dims = visible["dims"]
    height = dims[-1]
    z = (draw["kh"] - 0.5) * height
    if family == "deck_brake":
        return (draw["cx"] * dims[0], 0.0, z)  # + = toward the front (the tipping edge when braking)
    if family == "deck_turn":
        base = dims[1]
        outward = -1.0 if visible["direction"] == "left" else 1.0  # a left curve throws the load to the right (−y)
        return (0.0, outward * draw["cx"] * base, z)
    return (draw["cx"] * dims[0], 0.0, z)  # stack: + = toward the outer end; push: + = toward the far side (the tipping edge)


def _roll_variant(moment: Moment, name: str, draws: list[dict[str, float]], scenes: dict[Any, ph.Scene]) -> list[dict[str, Any]]:
    v, spec = moment.visible, moment.variants[name]
    out = []
    if moment.family in ("deck_brake", "deck_turn"):
        key = "deck"
        if key not in scenes:
            scenes[key] = ph.Scene(ph.deck_xml(v["shape"], v["dims"], v["mass"]))
        if spec["kind"] == "brake":
            profile = ph.brake_profile(spec["speed"], spec["decel"])
        else:
            profile = ph.turn_profile(spec["speed"], spec["radius"], direction=spec["direction"], ramp=spec["ramp"])
        for draw in draws:
            out.append(ph.run_deck(scenes[key], profile, mu=draw["mu"], com=_com(moment.family, v, draw)))
    elif moment.family == "stack":
        key = "stack"
        if key not in scenes:
            scenes[key] = ph.Scene(ph.stack_xml(v["dims"], v["mass"], (v["support_top"], max(0.3, v["dims"][1] + 0.1), v["support_height"])))
        length = v["dims"][0]
        for draw in draws:
            centre = v["support_top"] / 2 + spec["overhang"] - length / 2 + draw["eps"]
            out.append(ph.run_stack(scenes[key], x_centre=centre, drop=v["drop"], tilt0_deg=draw["tilt0"], mu=draw["mu"],
                                    com=_com("stack", v, draw), support_height=v["support_height"], box_height=v["dims"][2]))
    else:
        key = ("push", spec["height"])
        if key not in scenes:
            scenes[key] = ph.Scene(ph.push_xml(v["dims"], v["mass"], spec["height"]))
        profile = ph.push_profile(v["speed"], v["travel"])
        for draw in draws:
            out.append(ph.run_push(scenes[key], profile, mu=draw["mu"], com=_com("push", v, draw)))
    return out


def label_moment(moment: Moment, k: int = DEFAULT_K) -> dict[str, Any]:
    """Roll every variant out for the same K hidden draws → ``{"k", "draws_seed", "variants": {name: {"freq": {outcome: share},
    "n": {...}, "max_tilt_mean", "slide_mean"?}}, "majority"}``."""
    draws_seed = moment.seed ^ 0x5EED
    draws = hidden_draws(moment.hidden, k, draws_seed)
    scenes: dict[Any, ph.Scene] = {}
    outcomes = OUTCOMES[moment.family]
    variants = {}
    for name in moment.variants:
        rolls = _roll_variant(moment, name, draws, scenes)
        counts = {o: sum(1 for r in rolls if r["outcome"] == o) for o in outcomes}
        entry: dict[str, Any] = {"n": counts, "freq": {o: counts[o] / len(rolls) for o in outcomes},
                                 "max_tilt_mean": float(np.mean([r["max_tilt"] for r in rolls])),
                                 "unstable": sum(1 for r in rolls if r.get("unstable"))}
        if "slide_m" in rolls[0]:
            entry["slide_mean"] = float(np.mean([r["slide_m"] for r in rolls]))
            entry["off_deck"] = sum(1 for r in rolls if r.get("off_deck"))
        variants[name] = entry
    planned = variants["planned"]["freq"]
    majority = max(outcomes, key=lambda o: (planned[o], -outcomes.index(o)))
    unstable = sum(entry["unstable"] for entry in variants.values())
    return {"k": int(k), "draws_seed": int(draws_seed), "variants": variants, "majority": majority, "unstable": int(unstable)}


# --------------------------------------------------------------------------
# Text and questions (T8's own wording, first-person robot voice)
# --------------------------------------------------------------------------


def _m(value: float) -> str:
    return f"{value:.2f} m"


def _cm(value: float) -> str:
    return f"{value * 100:.0f} cm" if abs(value * 100 - round(value * 100)) < 1e-6 else f"{value * 100:.1f} cm"


def _deck_state(moment: Moment, rng: random.Random) -> str:
    v = moment.visible
    name = v["load_name"]
    article = "an" if name[0] in "aeiou" else "a"
    intro = rng.choice((f"I am a {v['robot']} moving straight ahead at {v['speed']:.2f} m/s.",
                        f"I'm a {v['robot']}, driving forward in a straight line at {v['speed']:.2f} m/s."))
    length, width, height = v["dims"]
    size = rng.choice((f"{_m(length)} front to back, {_m(width)} side to side and {_m(height)} tall",
                       f"{_m(length)} along my direction of travel, {_m(width)} across and {_m(height)} high"))
    load = rng.choice((f"Standing upright in the middle of my deck is {article} {name}, {size}, about {v['mass']:.1f} kg.",
                       f"In the middle of my deck, upright, I carry {article} {name}: {size}, roughly {v['mass']:.1f} kg."))
    fill = rng.choice(FILLS[v["fill"]]["text"])
    surface = rng.choice((f"Under it, my deck is {DECK_SURFACES[v['surface']]['text']}.",
                          f"The deck surface it stands on is {DECK_SURFACES[v['surface']]['text']}."))
    room = "There is at least 0.8 m of free deck around it on every side, and nothing it could lean against."
    if moment.family == "deck_brake":
        stop_time = v["speed"] / v["decel"]
        plan = rng.choice((
            f"I am about to stop: braking builds up to {v['decel']:.2f} m/s² over {v['jerk_time']:.2f} s and stays there until I stand still, "
            f"about {stop_time:.2f} s after it begins.",
            f"Next I brake to a standstill at {v['decel']:.2f} m/s² (reached within {v['jerk_time']:.2f} s); the stop takes about {stop_time:.2f} s.",
        ))
    else:
        plan = rng.choice((
            f"Next I follow a 90° curve to the {v['direction']} with a radius of {_m(v['radius'])}, keeping my speed; the steering eases in "
            f"over the first {v['steer_ramp']:.2f} s and eases out over the last {v['steer_ramp']:.2f} s.",
            f"Coming up is a {v['direction']}-hand 90° curve of radius {_m(v['radius'])}, which I take without slowing down (steering eases "
            f"in and out over {v['steer_ramp']:.2f} s).",
        ))
    return " ".join((intro, load, fill, surface, room, plan))


def _deck_questions(moment: Moment) -> dict[str, dict[str, Any]]:
    v = moment.visible
    name = v["load_name"]
    plan = "brake as planned" if moment.family == "deck_brake" else "take the curve as planned"
    during = "the planned stop" if moment.family == "deck_brake" else "the planned curve"
    if moment.family == "deck_brake":
        other = (f"Say I braked from the same speed at {v['other_decel']:.2f} m/s² instead. How likely is the {name} to end up "
                 "knocked over then?")
        other_name = "falls_other_braking"
    else:
        other = f"Say I took the same curve at {v['other_speed']:.2f} m/s instead. How likely is the {name} to end up knocked over then?"
        other_name = "falls_other_speed"
    return {
        "falls_over": {"type": "noul", "instructions": f"If I {plan}, does the {name} end up knocked over?"},
        "keeps_place": {"type": "noul", "instructions": f"If I {plan}, does the {name} stay put, upright and within 2 cm of where it stands now?"},
        "what_happens": {"type": "choice", "instructions": f"During {during}, which of these happens to the {name}?",
                         "criteria": {"stays": "it stays upright and shifts less than 2 cm on the deck",
                                      "slides": "it slides across the deck but stays upright",
                                      "topples": "it tips over (whether or not it slides first)"}},
        other_name: {"type": "score", "instructions": other, "criteria": ["0 = no chance it gets knocked over", "1 = it certainly gets knocked over"]},
    }


def _overhang_phrase(overhang: float, item: str) -> str:
    if overhang > 0.0005:
        return f"its far end sticking out {_cm(overhang)} past the edge"
    if overhang < -0.0005:
        return f"its far end {_cm(-overhang)} short of the edge"
    return "its far end exactly at the edge"


def _stack_state(moment: Moment, rng: random.Random) -> str:
    v = moment.visible
    length, width, height = v["dims"]
    intro = rng.choice((f"I am a {v['robot']} holding a {v['item_name']} by its top.", f"I'm a {v['robot']}, and my gripper holds a {v['item_name']}."))
    size = f"It is {_m(length)} long, {_m(width)} wide and {_m(height)} tall, about {v['mass']:.1f} kg."
    spread = STACK_SPREADS[v["spread"]]["text"]
    target = (f"I am about to set it down on {v['support_name']} whose flat top is {_m(v['support_top'])} long and {_m(v['support_height'])} above "
              "the floor.")
    plan = (f"The plan lays the {v['item_name']} lengthwise along that top, square to the edge, with {_overhang_phrase(v['overhang'], v['item_name'])}; "
            "the rest of it lies on the top.")
    release = rng.choice((f"My gripper opens when its underside is about {_cm(v['drop'])} above the top, and my placements usually land within "
                          f"about {_cm(v['sigma'])} of the plan.",
                          f"I let go about {_cm(v['drop'])} above the top; my placement error is typically around {_cm(v['sigma'])}."))
    return " ".join((intro, size, spread, target, plan, release))


def _stack_questions(moment: Moment) -> dict[str, dict[str, Any]]:
    v = moment.visible
    item, support = v["item_name"], v["support_short"]
    return {
        "falls_off": {"type": "noul", "instructions": f"Once I let go, does the {item} fall off the {support}?"},
        "falls_if_further_out": {"type": "score", "instructions": f"If I placed the {item} with {_overhang_phrase(v['overhang_out'], item)} instead, "
                                 "how likely is it to fall off once I let go?",
                                 "criteria": ["0 = it certainly stays on top", "1 = it certainly falls off"]},
        "falls_if_further_in": {"type": "noul", "instructions": f"Would the {item} fall off if I placed it with "
                                f"{_overhang_phrase(v['overhang_in'], item)} instead?"},
    }


def _push_state(moment: Moment, rng: random.Random) -> str:
    v = moment.visible
    length, width, height = v["dims"]
    intro = rng.choice((f"I am a {v['robot']} with a smooth plastic bumper bar across my front.",
                        f"I'm a {v['robot']}; across my front runs a smooth, rigid plastic bumper bar."))
    box = (f"Ahead of me on {PUSH_FLOORS[v['floor']]['text']} stands a {v['box_name']}: {_m(length)} deep in the direction I push, "
           f"{_m(width)} wide and {_m(height)} tall, about {v['mass']:.1f} kg.")
    spread = PUSH_SPREADS[v["spread"]]["text"]
    plan = rng.choice((f"I plan to push it {_m(v['travel'])} forward at {v['speed']:.2f} m/s, with the bar pressing on its near face "
                       f"{_m(v['push_height'])} above the floor.",
                       f"My plan: nudge it forward by {_m(v['travel'])} at {v['speed']:.2f} m/s, the bar touching its near face at a height of "
                       f"{_m(v['push_height'])}."))
    return " ".join((intro, box, spread, plan))


def _push_questions(moment: Moment) -> dict[str, dict[str, Any]]:
    v = moment.visible
    box = v["box_name"]
    return {
        "tips_over": {"type": "noul", "instructions": f"If I push as planned, does the {box} tip over instead of sliding along?"},
        "tips_at_other_height": {"type": "noul", "instructions": f"Would the {box} tip over if the bar met it {_m(v['push_height_noul'])} above the "
                                 "floor instead?"},
        "tip_risk_other_height": {"type": "score", "instructions": f"If the bar met the {box} {_m(v['push_height_score'])} above the floor, how "
                                  "likely is it to tip over?",
                                  "criteria": ["0 = it certainly slides along upright", "1 = it certainly tips over"]},
    }


def _labels(moment: Moment, label: dict[str, Any], questions: dict[str, Any]) -> dict[str, Any]:
    var = label["variants"]
    if moment.family in ("deck_brake", "deck_turn"):
        planned = var["planned"]["freq"]
        other_name = next(name for name, q in questions.items() if q["type"] == "score")
        return {
            "falls_over": {"type": "noul", "p_yes": planned["tip"]},
            "keeps_place": {"type": "noul", "p_yes": planned["stay"]},
            "what_happens": {"type": "choice", "kind": "state",
                             "probabilities": {"stays": planned["stay"], "slides": planned["slide"], "topples": planned["tip"]}},
            other_name: {"type": "score", "value": var["other"]["freq"]["tip"]},
        }
    if moment.family == "stack":
        return {"falls_off": {"type": "noul", "p_yes": var["planned"]["freq"]["fall"]},
                "falls_if_further_out": {"type": "score", "value": var["out"]["freq"]["fall"]},
                "falls_if_further_in": {"type": "noul", "p_yes": var["in"]["freq"]["fall"]}}
    return {"tips_over": {"type": "noul", "p_yes": var["planned"]["freq"]["tip"]},
            "tips_at_other_height": {"type": "noul", "p_yes": var["alt_noul"]["freq"]["tip"]},
            "tip_risk_other_height": {"type": "score", "value": var["alt_score"]["freq"]["tip"]}}


#: Which variant each question asks about (for the analysis of the planned vs counterfactual questions).
QUESTION_VARIANT = {
    "falls_over": "planned", "keeps_place": "planned", "what_happens": "planned", "falls_other_braking": "other", "falls_other_speed": "other",
    "falls_off": "planned", "falls_if_further_out": "out", "falls_if_further_in": "in",
    "tips_over": "planned", "tips_at_other_height": "alt_noul", "tip_risk_other_height": "alt_score",
}


def moment_record(moment: Moment, label: dict[str, Any], *, number: int, mujoco_version: str, git: str | None = None) -> dict[str, Any]:
    """Candidate + labels → one ``simgen-request/0`` record (``id = t8b-<family>-<number>``). Text wording is drawn from a stream of
    the moment's seed, so the same candidate always gives the same text."""
    rng = random.Random(moment.seed ^ 0x7E47)
    if moment.family in ("deck_brake", "deck_turn"):
        state, questions = _deck_state(moment, rng), _deck_questions(moment)
    elif moment.family == "stack":
        state, questions = _stack_state(moment, rng), _stack_questions(moment)
    else:
        state, questions = _push_state(moment, rng), _push_questions(moment)
    labels = _labels(moment, label, questions)
    short = {"deck_brake": "brake", "deck_turn": "turn", "stack": "stack", "push": "push"}[moment.family]
    record = {
        "schema": SCHEMA,
        "id": f"t8b-{short}-{number:04d}",
        "generator": {"name": GENERATOR, "family": moment.family, "candidate": moment.index, "seed": moment.seed, "draws_seed": label["draws_seed"],
                      "k": label["k"], "mujoco": mujoco_version, "git": git},
        "meta": {
            "set": "T8b", "scene": moment.family, "group": moment.family, "phenomena": [moment.family], "language": "en",
            "domain": moment.visible.get("domain"), "target": moment.target, "majority": label["majority"],
            "visible": moment.visible, "hidden_prior": {k: list(v) for k, v in moment.hidden.items()}, "variants": moment.variants,
            "outcomes": {name: entry["freq"] for name, entry in label["variants"].items()},
            "rollout_stats": {name: {k: v for k, v in entry.items() if k not in ("freq",)} for name, entry in label["variants"].items()},
            "question_variant": {name: QUESTION_VARIANT[name] for name in questions},
        },
        "request": {"state": state, "questions": questions},
        "labels": labels,
    }
    problems = validate_record(record)
    if problems:
        raise ValueError(f"{record['id']}: {problems}")
    return record


# --------------------------------------------------------------------------
# Balanced selection
# --------------------------------------------------------------------------


def select_balanced(labelled: Iterable[tuple[Moment, dict[str, Any]]], targets: dict[str, int]) -> tuple[list[tuple[Moment, dict[str, Any]]], dict[str, Any]]:
    """Keep, per family, ``targets[family] / n_outcomes`` moments of each majority outcome (candidate order). If an outcome runs short,
    the remaining slots go to the other outcomes in turn (recorded in the report)."""
    by_family: dict[str, list[tuple[Moment, dict[str, Any]]]] = {}
    unstable: dict[str, int] = {}
    for moment, label in labelled:
        if label.get("unstable"):
            unstable[moment.family] = unstable.get(moment.family, 0) + 1  # a rollout left the physically possible range — drop the moment
            continue
        by_family.setdefault(moment.family, []).append((moment, label))
    kept: list[tuple[Moment, dict[str, Any]]] = []
    report: dict[str, Any] = {}
    for family, total in targets.items():
        items = sorted(by_family.get(family, []), key=lambda ml: ml[0].index)
        outcomes = OUTCOMES[family]
        pools = {o: [ml for ml in items if ml[1]["majority"] == o] for o in outcomes}
        share = {o: total // len(outcomes) + (1 if i < total % len(outcomes) else 0) for i, o in enumerate(outcomes)}
        taken = {o: pools[o][: share[o]] for o in outcomes}
        missing = total - sum(len(v) for v in taken.values())
        while missing > 0:
            spare = [o for o in outcomes if len(pools[o]) > len(taken[o])]
            if not spare:
                break
            for o in spare:
                if missing <= 0:
                    break
                taken[o].append(pools[o][len(taken[o])])
                missing -= 1
        chosen = sorted((ml for o in outcomes for ml in taken[o]), key=lambda ml: ml[0].index)
        kept.extend(chosen)
        report[family] = {"candidates": len(items) + unstable.get(family, 0), "dropped_unstable": unstable.get(family, 0),
                          "pool": {o: len(pools[o]) for o in outcomes}, "kept": {o: len(taken[o]) for o in outcomes},
                          "target": total, "shortfall": max(0, total - len(chosen))}
    return kept, report
