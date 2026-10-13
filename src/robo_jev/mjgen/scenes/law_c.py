"""Law C — friction-limited holding: three new scenes on the squeeze engine (:mod:`robo_jev.mjgen.scenes.rigs`).

* ``plate`` — a lab arm's gripper holding a microplate, tube rack or tip box by two sides during a fast transfer, a lift or a lowering.
* ``clamp`` — a forklift's carton clamp holding an appliance carton between two big pads while it lifts, lowers, or the truck stops.
* ``palms`` — a humanoid carrying a box squeezed between its two palms while it lifts it or walks (held-out scene).

The item is held only by friction: it slips when 2·μ·F < m·(g + a) (a lift's speed-up, a lowering's stop, the bottom of each walking
bob) or m·√(g² + a²) (a sideways transfer or a braking carry), and falls out when 2·μ·F < m·g. Hidden values (never written): friction
(pad material × the item's surface, × 0.55–0.75 when wet) and, when the text gives only a range, the mass. The decisive number is the
squeeze force.
"""

from __future__ import annotations

import json
import math
import random
from typing import Any

from robo_jev.mjgen import families as fam
from robo_jev.mjgen import physics as ph
from robo_jev.mjgen.scenes import rigs
from robo_jev.mjgen.scenes.base import Scene, a_or_an, cm, num

__all__ = ["ClampScene", "PalmsScene", "PlateScene"]

G = rigs.GRAVITY
WET = fam.WET_FACTOR
SLIP_M = 0.01


class SqueezeScene(Scene):
    """Shared physics of the law-C scenes."""

    law = "friction_hold"
    outcomes = ("hold", "slip", "drop")
    event = ("slip", "drop")
    severe = "drop"
    pad_half: tuple[float, float, float] = (0.03, 0.006, 0.035)
    hand_mass = 400.0
    pads: dict[str, Any] = {}
    surfaces: dict[str, Any] = {}
    physics_fields = ("dims", "move", "distance", "peak", "accel", "ramp", "squeeze", "rise", "hz", "steps", "speed")

    def pad_for(self, dims: list[float]) -> tuple[float, float, float]:
        length, _, height = dims
        return (min(self.pad_half[0], length / 2 - 0.002), self.pad_half[1], min(self.pad_half[2], height / 2 - 0.002))

    def load(self, v: dict[str, Any]) -> float:
        """Acceleration (per kg) the grip must carry at the move's critical phase."""
        if v["move"] in ("swing", "brake"):
            return math.hypot(G, v["accel"])
        if v["move"] == "walk":
            return G + v["rise"] / 2 * (2 * math.pi * v["hz"]) ** 2
        return G + v["accel"]

    def hidden_for(self, v: dict[str, Any]) -> dict[str, Any]:
        mu = self.pads[v["pads"]]["mu"]
        factor = self.surfaces[v["surface"]]["factor"]
        if v.get("wet"):
            factor = (factor[0] * WET[0], factor[1] * WET[1])
        mass = (v["mass"] * 0.95, v["mass"] * 1.05) if v["mass_kind"] == "about" else (v["mass_lo"], v["mass_hi"])
        return {"mu": ("mu", round(mu[0] * factor[0], 4), round(mu[1] * factor[1], 4)), "mass": ("u",) + tuple(mass) + (0.01, 500.0)}

    def static_outcome(self, v: dict[str, Any], draw: dict[str, float]) -> str:
        hold = 2 * draw["mu"] * v["squeeze"]
        if hold >= draw["mass"] * self.load(v):
            return "hold"
        return "slip" if hold >= draw["mass"] * G else "drop"

    def path(self, v: dict[str, Any]) -> tuple[Any, int]:
        if v["move"] in ("lift", "lower", "swing"):
            return ph.move_profile(v["move"], v["distance"], v["peak"], v["accel"], jerk_time=v["ramp"])
        if v["move"] == "brake":
            return rigs.brake_carry_profile(v["speed"], v["accel"], jerk_time=v["ramp"]), 0
        return rigs.walk_profile(v["rise"], v["hz"], int(v["steps"])), 1

    def accept(self, v: dict[str, Any]) -> bool:
        if v["squeeze"] <= 0:
            return False
        if v["move"] == "walk":
            return v["steps"] / v["hz"] <= 8.0
        if v["move"] == "brake":  # the stated deceleration is held for most of the stop (the ramp-up is at most a third of it)
            return v["speed"] / v["accel"] <= 5.0 and v["ramp"] <= v["speed"] / v["accel"] / 3
        return True

    def roll(self, v: dict[str, Any], draw: dict[str, float], rigs_cache: dict[str, Any]) -> dict[str, Any]:
        key = json.dumps([self.name, v["dims"]])
        if key not in rigs_cache:
            rigs_cache[key] = rigs.squeeze_rig(self.name, v["dims"], pad_half=self.pad_for(v["dims"]), hand_mass=self.hand_mass)
        path, axis = self.path(v)
        pre = ph.GRIP_PRE if v["move"] in ("lift", "lower", "swing") else 0.4
        return rigs.run_squeeze(rigs_cache[key], path, axis, mu=draw["mu"], mass=draw["mass"], squeeze=v["squeeze"], slip_m=SLIP_M, pre=pre)

    def xml_example(self, v: dict[str, Any]) -> str:
        return rigs.squeeze_xml(self.name, v["dims"], pad_half=self.pad_for(v["dims"]), hand_mass=self.hand_mass)

    mass_digits = 2

    def mass_text(self, v: dict[str, Any], unknown: str) -> str:
        d = self.mass_digits
        if v["mass_kind"] == "about":
            return f"about {v['mass']:.{d}f} kg"
        return f"between {v['mass_lo']:.{d}f} and {v['mass_hi']:.{d}f} kg, {unknown}"

    def options(self, group: Any, rng: random.Random) -> None:
        v = group.visible
        step = self.force_step
        firmer = fam._round(min(self.force_max, v["squeeze"] * rng.uniform(1.3, 1.9)), step)
        lighter = fam._round(max(step, v["squeeze"] * rng.uniform(0.6, 0.85)), step)
        if v["move"] == "walk":
            harsh = {"squeeze": firmer, "hz": round(min(2.6, v["hz"] * rng.uniform(1.2, 1.4)), 2)}
            gentle = {"squeeze": lighter, "rise": round(max(0.01, v["rise"] * rng.uniform(0.4, 0.65)), 3)}
        else:
            harsh = {"squeeze": firmer, "accel": fam._round(min(self.accel_max, v["accel"] * rng.uniform(1.6, 2.6) + self.accel_add), 0.1)}
            gentle = {"squeeze": lighter, "accel": fam._round(max(0.2, v["accel"] * rng.uniform(0.25, 0.6)), 0.1)}
        group.variants.update({"opt_firm": harsh, "opt_gentle": gentle})
        group.plan["options"] = {"as_planned": "planned", "firmer_harsher": "opt_firm", "lighter_gentler": "opt_gentle"}

    force_step = 0.5
    force_max = 120.0
    accel_max = 15.0
    accel_add = 1.0
    dominant_option = "firmer_harsher"

    def factor_choices(self, group: Any, factor: str, current: Any) -> list[Any]:
        v = group.visible
        if factor == "pads":
            return sorted(self.pads)
        if factor == "item_surface":
            return sorted(self.surfaces)
        if factor == "wet":
            return [not current]
        if factor == "ramp_time":
            if v["move"] not in ("lift", "lower", "swing"):
                return [r for r in (0.02, 0.05, 0.25, 0.5) if abs(r - current) > 0.04]
            longest = v["peak"] / v["accel"]
            return [round(x, 2) for x in (0.01, longest * 0.7, longest) if longest >= 0.08 and abs(x - current) > 0.03]
        if factor == "steps":
            return [s for s in (3, 6, 10, 14) if s != current and s / v["hz"] <= 8.0]
        return []


# ======================================================================================================================================
# plate — a lab gripper holding a microplate, rack or box
# ======================================================================================================================================

PLATE_TIPS = {
    "peek": {"text": "PEEK fingertips", "mu": (0.25, 0.35)},
    "nitrile": {"text": "nitrile-sleeved fingers", "mu": (0.70, 0.95)},
    "knurled": {"text": "knurled steel fingertips", "mu": (0.40, 0.55)},
    "acetal": {"text": "acetal fingertips", "mu": (0.18, 0.26)},
    "tpe": {"text": "soft TPE fingertip pads", "mu": (0.80, 1.05)},
}
PLATE_SURFACES = {
    "polystyrene": {"text": "smooth polystyrene", "factor": (0.95, 1.05)},
    "polypropylene": {"text": "waxy polypropylene", "factor": (0.65, 0.80)},
    "frosted": {"text": "frosted, matt plastic", "factor": (1.05, 1.20)},
    "label": {"text": "plastic with paper barcode labels on the gripped sides", "factor": (1.00, 1.15)},
}
PLATE_ITEMS = (
    {"key": "microplate", "name": "96-well microplate", "dims": ((0.126, 0.128), (0.084, 0.086), (0.014, 0.016)), "mass": ("about", 0.06, 0.25),
     "surfaces": ("polystyrene", "label")},
    {"key": "deepwell", "name": "deep-well block", "dims": ((0.126, 0.128), (0.084, 0.086), (0.040, 0.046)), "mass": ("range", 0.15, 1.6),
     "surfaces": ("polypropylene", "label")},
    {"key": "tuberack", "name": "rack of capped sample tubes", "dims": ((0.120, 0.130), (0.080, 0.090), (0.045, 0.060)), "mass": ("range", 0.10, 0.80),
     "surfaces": ("polypropylene", "frosted")},
    {"key": "tipbox", "name": "box of pipette tips", "dims": ((0.120, 0.125), (0.080, 0.085), (0.060, 0.090)), "mass": ("about", 0.10, 0.30),
     "surfaces": ("polypropylene", "frosted")},
    {"key": "reservoir", "name": "filled reagent reservoir", "dims": ((0.126, 0.128), (0.084, 0.086), (0.030, 0.044)), "mass": ("range", 0.06, 0.45),
     "surfaces": ("polystyrene", "polypropylene")},
    {"key": "slidebox", "name": "box of glass slides", "dims": ((0.10, 0.12), (0.08, 0.09), (0.030, 0.040)), "mass": ("about", 0.12, 0.40),
     "surfaces": ("frosted", "label")},
)
PLATE_ROBOTS = ("plate-handling arm", "liquid-handler gripper arm", "benchtop lab automation arm")
PLATE_DOMAINS = ("genomics lab", "hospital pathology lab", "vaccine plant QC lab", "university biology lab", "drug-screening facility",
                 "food-testing lab", "forensics lab", "blood bank")


class PlateScene(SqueezeScene):
    name = "plate"
    pad_half = (0.012, 0.004, 0.012)
    pads = PLATE_TIPS
    surfaces = PLATE_SURFACES
    force_step = 0.1
    force_max = 60.0
    accel_max = 20.0
    accel_add = 1.5
    question_names = {"key": "plate_slips", "direction": "grip_force_effect", "outcome": "plate_after_transfer", "action": "safest_transfer",
                      "compare": "riskier_transfer", "score_cf": "slip_risk_other_grip", "severe": "plate_dropped"}
    outcome_keys = {"held_firm": "hold", "slid_in_grip": "slip", "dropped": "drop"}
    factors = {
        "pads": {"field": "pads", "fact": "word", "how": "word"},
        "item_surface": {"field": "surface", "fact": "word", "how": "word"},
        "wet": {"field": "wet", "fact": "word", "how": "word"},
        "ramp_time": {"field": "ramp", "fact": "number", "how": "choice"},
    }
    pair_fields = (("decisive", 0.5), ("accel", 0.5))
    pair_spans = {"accel": lambda c, v: (0.5, 20.0, 0.1)}
    domains = PLATE_DOMAINS

    def sample(self, rng: random.Random) -> tuple[dict[str, Any], dict[str, Any]]:
        item = rng.choice(PLATE_ITEMS)
        dims = [round(rng.uniform(*span), 3) for span in item["dims"]]
        kind, lo, hi = item["mass"]
        v = {"robot": rng.choice(PLATE_ROBOTS), "domain": rng.choice(PLATE_DOMAINS), "item": item["key"], "item_name": item["name"], "dims": dims,
             "surface": rng.choice(item["surfaces"]), "pads": rng.choice(sorted(PLATE_TIPS)), "wet": rng.random() < 0.15, "mass_kind": kind,
             "move": rng.choice(("swing", "swing", "lift", "lower")), "distance": round(rng.uniform(0.2, 0.6), 2), "peak": round(rng.uniform(0.4, 1.5), 1),
             "accel": round(rng.uniform(1.5, 14.0), 1), "ramp": 0.03, "squeeze": 0.0}
        if kind == "about":
            v["mass"] = round(rng.uniform(lo, hi), 3)
        else:
            v["mass_lo"], v["mass_hi"] = round(lo, 2), round(hi, 2)
        if v["distance"] < v["peak"] ** 2 / v["accel"] + 0.02:
            v["peak"] = round(max(0.2, math.sqrt(max(0.01, (v["distance"] - 0.02) * v["accel"]))), 1)
        return v, {"field": "squeeze", "lo": 0.2, "hi": 60.0, "step": 0.1, "increasing": False}

    def _move(self, v: dict[str, Any], person: str) -> str:
        subject = {"third": "The arm", "second": "You", "first": "I"}[person]
        verb = {"lift": "raise it straight up", "lower": "lower it straight down", "swing": "carry it sideways to the next instrument"}[v["move"]]
        return (f"{subject} will {verb}, {cm(v['distance'])} in all; peak speed {v['peak']:.1f} m/s, speeding up and slowing at {v['accel']:.1f} m/s², "
                f"a rate reached {v['ramp']:.2f} s into each change.")

    def state_text(self, v: dict[str, Any], style: str, rng: random.Random) -> str:
        length, width, height = v["dims"]
        tips = PLATE_TIPS[v["pads"]]["text"]
        outside = PLATE_SURFACES[v["surface"]]["text"] + (", beaded with condensation from the cold room" if v.get("wet") else "")
        size = f"{num(length, 3)} m in length and {num(height, 3)} m in height, gripped across a width of {num(width, 3)} m"
        force = f"{v['squeeze']:.1f} N"
        mass = self.mass_text(v, "fill level not logged")
        if style == "log":
            return (f"Run log, {v['domain']}. Unit — {a_or_an(v['robot'])} with {tips}. In the gripper — one {v['item_name']} ({size}), {outside}, "
                    f"weighing {mass}. Grip — the two long sides, {force} of closing force. Next step — {self._move(v, 'third')}")
        if style == "brief":
            return (f"You are {a_or_an(v['robot'])} with {tips}, holding {a_or_an(v['item_name'])} by its two long sides at {force}. It is {size}; "
                    f"its sides are {outside}; it weighs {mass}. {self._move(v, 'second')}")
        if style == "sheet":
            return "\n".join((f"robot — {v['robot']} ({v['domain']})", f"fingers — {tips}", f"labware — {v['item_name']}", f"size — {size}",
                              f"gripped sides — {outside}", f"weight — {mass}", f"closing force — {force}", f"motion — {self._move(v, 'third')}"))
        return (f"As {a_or_an(v['robot'])} in the {v['domain']}, I am holding {a_or_an(v['item_name'])} between my {tips}, pressing its long sides "
                f"with {force}. It is {size}, its sides are {outside}, and it weighs {mass}. {self._move(v, 'first')}")

    def _options(self, v: dict[str, Any], plan: dict[str, Any]) -> dict[str, str]:
        opts = plan["option_values"]
        return {"as_planned": f"grip at {v['squeeze']:.1f} N and move at {v['accel']:.1f} m/s², as planned",
                "firmer_harsher": f"grip at {opts['opt_firm']['squeeze']:.1f} N but move at {opts['opt_firm']['accel']:.1f} m/s²",
                "lighter_gentler": f"grip at {opts['opt_gentle']['squeeze']:.1f} N and move at a gentle {opts['opt_gentle']['accel']:.1f} m/s²"}

    def questions(self, v: dict[str, Any], mix: list[str], plan: dict[str, Any], person: str, rng: random.Random, cf_option: str,
                  action_keys: list[str]) -> dict[str, Any]:
        item = v["item_name"]
        what = {"lift": "lift", "lower": "lowering", "swing": "transfer"}[v["move"]]
        fingers = {"third": "the fingers", "second": "your fingers", "first": "my fingers"}[person]
        now, other = f"{v['squeeze']:.1f} N", f"{plan['cf_value']:.1f} N"
        out: dict[str, Any] = {}
        out["plate_slips"] = {"type": "noul", "instructions": rng.choice((
            f"During the {what}, does the {item} slide 1 cm or more between {fingers}, or drop out?",
            f"Will the {item} slip at least 1 cm in {fingers} (or fall) during the {what}?"))}
        out["grip_force_effect"] = {"type": "choice", "instructions": rng.choice((
            f"Suppose the closing force were {other} rather than {now}. Would the {item} be more likely, less likely or about as likely to slip or drop?",
            f"Against gripping at {now}, what would gripping at {other} do to the chance that the {item} slips or drops in the {what}?")),
            "criteria": dict(PLATE_DIRECTION)}
        if "outcome" in mix:
            out["plate_after_transfer"] = {"type": "choice", "instructions": f"Where is the {item} at the end of the {what}?",
                                           "criteria": {"held_firm": "held, shifted less than 1 cm", "slid_in_grip": "slid 1 cm or more but still held",
                                                        "dropped": "out of the gripper"}}
        options = self._options(v, plan)
        if "action" in mix:
            out["safest_transfer"] = {"type": "choice", "instructions": f"Which way of moving the {item} is safest?",
                                      "criteria": {k: options[k] for k in action_keys}}
        if "compare" in mix:
            out["riskier_transfer"] = {"type": "choice", "instructions": f"Which makes a slip or drop of the {item} more likely: sticking to the plan "
                                       f"({options['as_planned']}), or the other way ({options[cf_option]})?",
                                       "criteria": {"as_planned": "sticking to the plan", cf_option: "the other way",
                                                    "about_same": "roughly equal (within 0.1)"}}
        if "score_cf" in mix:
            out["slip_risk_other_grip"] = {"type": "score", "instructions": f"Rate the chance that the {item} slips or drops if the closing force were "
                                           f"{other} rather than {now}.", "criteria": ["0 = the fingers will not lose it at all", "1 = it is certain to slip or drop"]}
        if "severe" in mix:
            out["plate_dropped"] = {"type": "noul", "instructions": f"Does the {item} drop out of {fingers} during the {what}?"}
        return out

    def catalogs(self) -> dict[str, Any]:
        return {"fingertips": PLATE_TIPS, "surfaces": PLATE_SURFACES, "items": PLATE_ITEMS, "robots": PLATE_ROBOTS, "domains": PLATE_DOMAINS}

    def catalog_names(self) -> set[str]:
        out = {item["name"].lower() for item in PLATE_ITEMS} | {r.lower() for r in PLATE_ROBOTS}
        out |= {x["text"].lower() for x in list(PLATE_TIPS.values()) + list(PLATE_SURFACES.values())}
        return out


PLATE_DIRECTION = {"more_likely": "a higher chance than in the plan", "less_likely": "a lower chance than in the plan",
                   "about_same": "roughly unchanged (it moves by less than 0.1)"}


# ======================================================================================================================================
# clamp — a forklift carton clamp holding an appliance carton
# ======================================================================================================================================

CLAMP_PADS = {
    "rubber": {"text": "rubber-faced clamp arms", "mu": (0.60, 0.85)},
    "ribbed": {"text": "ribbed steel clamp plates", "mu": (0.35, 0.50)},
    "foam": {"text": "foam-faced clamp arms", "mu": (0.45, 0.62)},
    "smooth": {"text": "smooth steel clamp plates", "mu": (0.22, 0.32)},
}
CLAMP_SURFACES = {
    "corrugated": {"text": "plain corrugated board", "factor": (0.95, 1.05)},
    "waxed": {"text": "a waxed carton finish", "factor": (0.60, 0.75)},
    "stretch": {"text": "stretch-film wrapping", "factor": (0.70, 0.85)},
    "printed": {"text": "glossy printed board", "factor": (0.80, 0.92)},
}
CLAMP_ITEMS = (
    {"key": "fridge", "name": "boxed fridge-freezer", "dims": ((0.65, 0.80), (0.65, 0.80), (1.70, 1.95)), "mass": ("about", 60.0, 95.0),
     "surfaces": ("corrugated", "printed", "stretch")},
    {"key": "washer", "name": "boxed washing machine", "dims": ((0.60, 0.68), (0.62, 0.70), (0.85, 0.92)), "mass": ("about", 60.0, 82.0),
     "surfaces": ("corrugated", "printed")},
    {"key": "tv", "name": "boxed 75-inch television", "dims": ((1.75, 1.90), (0.20, 0.28), (1.05, 1.20)), "mass": ("about", 30.0, 48.0),
     "surfaces": ("printed", "corrugated")},
    {"key": "dishwasher", "name": "boxed dishwasher", "dims": ((0.62, 0.68), (0.62, 0.70), (0.88, 0.95)), "mass": ("about", 42.0, 58.0),
     "surfaces": ("corrugated", "waxed")},
    {"key": "aircon", "name": "boxed air-conditioning unit", "dims": ((0.90, 1.05), (0.38, 0.45), (0.65, 0.80)), "mass": ("range", 35.0, 70.0),
     "surfaces": ("corrugated", "stretch")},
    {"key": "returns", "name": "unlabelled returns carton", "dims": ((0.70, 1.00), (0.60, 0.90), (0.70, 1.10)), "mass": ("range", 15.0, 110.0),
     "surfaces": ("corrugated", "waxed", "stretch")},
)
CLAMP_ROBOTS = ("carton-clamp forklift", "appliance clamp truck", "driverless clamp truck")
CLAMP_DOMAINS = ("appliance warehouse", "retail distribution hub", "returns centre", "container yard", "electronics depot", "furniture store backroom",
                 "rail freight shed", "cross-dock terminal")


class ClampScene(SqueezeScene):
    name = "clamp"
    pad_half = (0.30, 0.02, 0.35)
    hand_mass = 4000.0
    mass_digits = 1
    pads = CLAMP_PADS
    surfaces = CLAMP_SURFACES
    force_step = 10.0
    force_max = 12000.0
    accel_max = 6.0
    accel_add = 0.3
    question_names = {"key": "carton_slips_in_clamp", "direction": "clamp_force_effect", "outcome": "carton_after_move", "action": "safest_clamp_plan",
                      "compare": "riskier_clamp_plan", "score_cf": "slip_risk_other_clamp", "severe": "carton_dropped"}
    outcome_keys = {"held_firm": "hold", "slid_down": "slip", "dropped": "drop"}
    factors = {
        "pads": {"field": "pads", "fact": "word", "how": "word"},
        "item_surface": {"field": "surface", "fact": "word", "how": "word"},
        "wet": {"field": "wet", "fact": "word", "how": "word"},
        "ramp_time": {"field": "ramp", "fact": "number", "how": "choice"},
    }
    pair_fields = (("decisive", 0.5), ("accel", 0.3), ("mass", 0.2))
    pair_spans = {"accel": lambda c, v: (0.2, 6.0, 0.1), "mass": lambda c, v: (max(5.0, c * 0.4), c * 2.5, 1.0)}
    domains = CLAMP_DOMAINS

    def pad_for(self, dims: list[float]) -> tuple[float, float, float]:
        length, _, height = dims
        return (min(self.pad_half[0], length / 2 - 0.02), self.pad_half[1], min(self.pad_half[2], height / 2 - 0.02))

    def sample(self, rng: random.Random) -> tuple[dict[str, Any], dict[str, Any]]:
        item = rng.choice(CLAMP_ITEMS)
        dims = [round(rng.uniform(*span), 2) for span in item["dims"]]
        kind, lo, hi = item["mass"]
        move = rng.choice(("lift", "lower", "brake", "brake"))
        v = {"robot": rng.choice(CLAMP_ROBOTS), "domain": rng.choice(CLAMP_DOMAINS), "item": item["key"], "item_name": item["name"], "dims": dims,
             "surface": rng.choice(item["surfaces"]), "pads": rng.choice(sorted(CLAMP_PADS)), "wet": rng.random() < 0.15, "mass_kind": kind,
             "move": move, "squeeze": 0.0, "ramp": round(rng.uniform(0.05, 0.3), 2)}
        if move == "brake":
            v.update({"speed": round(rng.uniform(1.5, 3.5), 1), "accel": round(rng.uniform(0.6, 4.5), 1)})
        else:
            v.update({"distance": round(rng.uniform(0.3, 1.2), 2), "peak": round(rng.uniform(0.2, 0.6), 1), "accel": round(rng.uniform(0.4, 3.0), 1),
                      "ramp": 0.08})
            if v["distance"] < v["peak"] ** 2 / v["accel"] + 0.02:
                v["peak"] = round(max(0.1, math.sqrt(max(0.01, (v["distance"] - 0.02) * v["accel"]))), 1)
        if kind == "about":
            v["mass"] = round(rng.uniform(lo, hi), 1)
        else:
            v["mass_lo"], v["mass_hi"] = round(lo, 1), round(hi, 1)
        return v, {"field": "squeeze", "lo": 50.0, "hi": 12000.0, "step": 10.0, "increasing": False}

    def _move(self, v: dict[str, Any], person: str) -> str:
        subject = {"third": "The truck", "second": "You", "first": "I"}[person]
        if v["move"] == "brake":
            be = {"third": "is", "second": "are", "first": "am"}[person]
            return (f"{subject} {be} carrying it across the floor at {v['speed']:.1f} m/s; a halt follows, decelerating at {v['accel']:.1f} m/s², full braking "
                    f"reached after {v['ramp']:.2f} s.")
        verb = "raise" if v["move"] == "lift" else "set down"
        return (f"{subject} will {verb} it {num(v['distance'])} m; top speed {v['peak']:.1f} m/s, speeding up and slowing at {v['accel']:.1f} m/s² "
                f"(built up over {v['ramp']:.2f} s).")

    def state_text(self, v: dict[str, Any], style: str, rng: random.Random) -> str:
        length, width, height = v["dims"]
        pads = CLAMP_PADS[v["pads"]]["text"]
        outside = CLAMP_SURFACES[v["surface"]]["text"] + (", rain-soaked on the outside" if v.get("wet") else "")
        size = f"{num(height)} m tall, {num(length)} m along the arms, {num(width)} m between them"
        force = f"{v['squeeze']:.0f} N"
        mass = self.mass_text(v, "model not on the label")
        if style == "log":
            return (f"Dock log, {v['domain']}. Unit — {a_or_an(v['robot'])} with {pads}. Clamped — one {v['item_name']} ({size}), {outside}, weighing "
                    f"{mass}. Clamp force — {force}, arms centred on the two side faces; nothing under the carton. {self._move(v, 'third')}")
        if style == "brief":
            return (f"You run {a_or_an(v['robot'])} with {pads}. You have {a_or_an(v['item_name'])} squeezed between the arms at {force}, centred on its "
                    f"sides, with nothing underneath. It is {size}, wrapped in {outside}, and weighs {mass}. {self._move(v, 'second')}")
        if style == "sheet":
            return "\n".join((f"truck — {v['robot']} ({v['domain']})", f"clamp arms — {pads}", f"carton — {v['item_name']}", f"carton size — {size}",
                              f"carton outside — {outside}", f"carton weight — {mass}", f"clamp force — {force}, on the side faces only",
                              f"move — {self._move(v, 'third')}"))
        return (f"I am {a_or_an(v['robot'])} with {pads} in the {v['domain']}. Between my arms, pressed at {force} on its side faces and resting on "
                f"nothing else, is {a_or_an(v['item_name'])}: {size}, finished in {outside}, weighing {mass}. {self._move(v, 'first')}")

    def _options(self, v: dict[str, Any], plan: dict[str, Any]) -> dict[str, str]:
        opts = plan["option_values"]
        return {"as_planned": f"clamp at {v['squeeze']:.0f} N and go at {v['accel']:.1f} m/s², as planned",
                "firmer_harsher": f"clamp at {opts['opt_firm']['squeeze']:.0f} N but go at {opts['opt_firm']['accel']:.1f} m/s²",
                "lighter_gentler": f"clamp at {opts['opt_gentle']['squeeze']:.0f} N and go at only {opts['opt_gentle']['accel']:.1f} m/s²"}

    def questions(self, v: dict[str, Any], mix: list[str], plan: dict[str, Any], person: str, rng: random.Random, cf_option: str,
                  action_keys: list[str]) -> dict[str, Any]:
        item = v["item_name"]
        what = {"lift": "lift", "lower": "set-down", "brake": "stop"}[v["move"]]
        arms = {"third": "the clamp", "second": "your clamp", "first": "my clamp"}[person]
        now, other = f"{v['squeeze']:.0f} N", f"{plan['cf_value']:.0f} N"
        out: dict[str, Any] = {}
        out["carton_slips_in_clamp"] = {"type": "noul", "instructions": rng.choice((
            f"During the {what}, does the {item} slide 1 cm or more in {arms}, or fall out?",
            f"Will the {item} shift at least 1 cm between the arms (or drop) during the {what}?"))}
        out["clamp_force_effect"] = {"type": "choice", "instructions": rng.choice((
            f"Suppose the clamp force were {other} rather than {now}. Would the {item} be more likely, less likely or about as likely to slip or drop?",
            f"Against clamping at {now}, what would {other} do to the chance that the {item} slips or drops during the {what}?")),
            "criteria": dict(PLATE_DIRECTION)}
        if "outcome" in mix:
            out["carton_after_move"] = {"type": "choice", "instructions": f"How does the {item} come through the {what}?",
                                        "criteria": {"held_firm": "held, moved less than 1 cm", "slid_down": "slid 1 cm or more but still clamped",
                                                     "dropped": "fell out of the clamp"}}
        options = self._options(v, plan)
        if "action" in mix:
            out["safest_clamp_plan"] = {"type": "choice", "instructions": f"Which plan is safest for the {item}?",
                                        "criteria": {k: options[k] for k in action_keys}}
        if "compare" in mix:
            out["riskier_clamp_plan"] = {"type": "choice", "instructions": f"Which makes a slip or drop of the {item} more likely: sticking to the plan "
                                         f"({options['as_planned']}), or the other plan ({options[cf_option]})?",
                                         "criteria": {"as_planned": "sticking to the plan", cf_option: "the alternative plan",
                                                      "about_same": "roughly equal (within 0.1)"}}
        if "score_cf" in mix:
            out["slip_risk_other_clamp"] = {"type": "score", "instructions": f"Rate the chance that the {item} slips or drops if the clamp force were "
                                            f"{other} rather than {now}.", "criteria": ["0 = the clamp will not give at all", "1 = it is certain to slip or drop"]}
        if "severe" in mix:
            out["carton_dropped"] = {"type": "noul", "instructions": f"Does the {item} fall out of {arms} during the {what}?"}
        return out

    def catalogs(self) -> dict[str, Any]:
        return {"pads": CLAMP_PADS, "surfaces": CLAMP_SURFACES, "items": CLAMP_ITEMS, "robots": CLAMP_ROBOTS, "domains": CLAMP_DOMAINS}

    def catalog_names(self) -> set[str]:
        out = {item["name"].lower() for item in CLAMP_ITEMS} | {r.lower() for r in CLAMP_ROBOTS}
        out |= {x["text"].lower() for x in list(CLAMP_PADS.values()) + list(CLAMP_SURFACES.values())}
        return out


# ======================================================================================================================================
# palms — a humanoid carrying a box between its palms (held-out scene: never in the training file)
# ======================================================================================================================================

PALM_COVERS = {
    "rubber": {"text": "grippy rubber palms", "mu": (0.75, 1.00)},
    "fabric": {"text": "fabric work gloves", "mu": (0.45, 0.62)},
    "hard": {"text": "hard plastic palms", "mu": (0.25, 0.36)},
    "leather": {"text": "leather-palmed gloves", "mu": (0.50, 0.68)},
    "metal": {"text": "bare aluminium palms", "mu": (0.18, 0.28)},
}
PALM_SURFACES = {
    "card": {"text": "dry cardboard", "factor": (0.95, 1.05)},
    "shrink": {"text": "shrink-wrap", "factor": (0.65, 0.80)},
    "plywood": {"text": "rough plywood", "factor": (1.00, 1.15)},
    "crate": {"text": "smooth moulded plastic", "factor": (0.65, 0.78)},
}
PALM_ITEMS = (
    {"key": "books", "name": "moving box of books", "dims": ((0.30, 0.40), (0.28, 0.36), (0.25, 0.32)), "mass": ("range", 6.0, 16.0), "surfaces": ("card",)},
    {"key": "tote", "name": "lidded storage tote", "dims": ((0.35, 0.45), (0.28, 0.36), (0.22, 0.30)), "mass": ("range", 1.5, 9.0), "surfaces": ("crate",)},
    {"key": "water", "name": "shrink-wrapped tray of water bottles", "dims": ((0.36, 0.40), (0.24, 0.27), (0.20, 0.23)), "mass": ("about", 8.5, 9.5),
     "surfaces": ("shrink",)},
    {"key": "toolchest", "name": "plywood tool chest", "dims": ((0.40, 0.50), (0.25, 0.32), (0.22, 0.30)), "mass": ("range", 4.0, 14.0),
     "surfaces": ("plywood",)},
    {"key": "copier", "name": "carton of copier paper", "dims": ((0.42, 0.44), (0.28, 0.30), (0.22, 0.25)), "mass": ("about", 11.0, 12.5),
     "surfaces": ("card",)},
    {"key": "planter", "name": "boxed ceramic planter", "dims": ((0.30, 0.38), (0.30, 0.38), (0.30, 0.40)), "mass": ("about", 3.0, 7.0),
     "surfaces": ("card", "shrink")},
)
PALM_ROBOTS = ("G1 humanoid", "bipedal helper humanoid", "two-armed walking robot")
PALM_DOMAINS = ("apartment hallway", "office move", "school storeroom", "community hall", "garden shed", "library basement", "hotel corridor",
                "clinic supply room")


class PalmsScene(SqueezeScene):
    name = "palms"
    verify_aim = True
    pad_half = (0.06, 0.012, 0.07)
    mass_digits = 1
    pads = PALM_COVERS
    surfaces = PALM_SURFACES
    force_step = 1.0
    force_max = 400.0
    accel_max = 8.0
    accel_add = 0.5
    question_names = {"key": "box_slips_between_palms", "direction": "press_change_effect", "outcome": "box_after_carry", "action": "safest_carry",
                      "compare": "riskier_carry", "score_cf": "slip_risk_other_press", "severe": "box_slips_free"}
    outcome_keys = {"secure": "hold", "slid_lower": "slip", "on_the_floor": "drop"}
    factors = {
        "palms": {"field": "pads", "fact": "word", "how": "word"},
        "item_surface": {"field": "surface", "fact": "word", "how": "word"},
        "wet": {"field": "wet", "fact": "word", "how": "word"},
        "steps": {"field": "steps", "fact": "number", "how": "choice"},
        "bob": {"field": "rise", "fact": "number", "how": "aim", "span": lambda c, v: (0.01, 0.10, 0.005)},
    }
    pair_fields = (("decisive", 0.7), ("mass", 0.3))
    pair_spans = {"mass": lambda c, v: (max(0.5, c * 0.4), c * 2.5, 0.1)}
    domains = PALM_DOMAINS

    def factor_choices(self, group: Any, factor: str, current: Any) -> list[Any]:
        if factor == "palms":
            return sorted(PALM_COVERS)
        if factor in ("steps", "bob") and group.visible["move"] != "walk":
            return []
        return super().factor_choices(group, factor, current)

    def sample(self, rng: random.Random) -> tuple[dict[str, Any], dict[str, Any]]:
        item = rng.choice(PALM_ITEMS)
        dims = [round(rng.uniform(*span), 2) for span in item["dims"]]
        kind, lo, hi = item["mass"]
        move = rng.choice(("walk", "walk", "lift"))
        v = {"robot": rng.choice(PALM_ROBOTS), "domain": rng.choice(PALM_DOMAINS), "item": item["key"], "item_name": item["name"], "dims": dims,
             "surface": rng.choice(item["surfaces"]), "pads": rng.choice(sorted(PALM_COVERS)), "wet": rng.random() < 0.15, "mass_kind": kind,
             "move": move, "squeeze": 0.0, "rise": 0.0, "hz": 0.0, "steps": 0, "accel": 0.0}
        if move == "walk":
            v.update({"rise": round(rng.uniform(0.02, 0.08), 3), "hz": round(rng.uniform(1.4, 2.2), 2), "steps": rng.choice((4, 6, 8, 10, 12))})
        else:
            v.update({"distance": round(rng.uniform(0.2, 0.5), 2), "peak": round(rng.uniform(0.3, 0.9), 1), "accel": round(rng.uniform(1.0, 6.0), 1),
                      "ramp": 0.05})
            if v["distance"] < v["peak"] ** 2 / v["accel"] + 0.02:
                v["peak"] = round(max(0.2, math.sqrt(max(0.01, (v["distance"] - 0.02) * v["accel"]))), 1)
        if kind == "about":
            v["mass"] = round(rng.uniform(lo, hi), 1)
        else:
            v["mass_lo"], v["mass_hi"] = round(lo, 1), round(hi, 1)
        return v, {"field": "squeeze", "lo": 2.0, "hi": 400.0, "step": 1.0, "increasing": False}

    def _move(self, v: dict[str, Any], person: str) -> str:
        subject = {"third": "The robot", "second": "You", "first": "I"}[person]
        if v["move"] == "walk":
            return (f"{subject} will walk {v['steps']} steps at {v['hz']:.2f} steps per second; with each step the hands bob up and down by "
                    f"{cm(v['rise'])} in all.")
        return (f"{subject} will lift it {cm(v['distance'])} off the bench; top speed {v['peak']:.1f} m/s, accelerating upward at {v['accel']:.1f} m/s².")

    def state_text(self, v: dict[str, Any], style: str, rng: random.Random) -> str:
        length, width, height = v["dims"]
        palms = PALM_COVERS[v["pads"]]["text"]
        outside = PALM_SURFACES[v["surface"]]["text"] + (", slick with drizzle" if v.get("wet") else "")
        size = f"{num(width)} m across between the palms, {num(length)} m deep and {num(height)} m high"
        press = f"{v['squeeze']:.0f} N"
        mass = self.mass_text(v, "fill unknown")
        if style == "log":
            return (f"Task note, {v['domain']}. Unit — {a_or_an(v['robot'])} wearing {palms}. Carrying — one {v['item_name']}, {size}, sides of {outside}, "
                    f"weight {mass}. Hold — one palm flat on each side face, pressing {press} each, no hand underneath. {self._move(v, 'third')}")
        if style == "brief":
            return (f"You are {a_or_an(v['robot'])} with {palms}. You hold {a_or_an(v['item_name'])} ({size}; sides of {outside}; weight {mass}) with "
                    f"one palm flat on each side, each pressing {press}, and nothing beneath it. {self._move(v, 'second')}")
        if style == "sheet":
            return "\n".join((f"walker — {v['robot']} ({v['domain']})", f"palm covers — {palms}", f"box carried — {v['item_name']}", f"box size — {size}",
                              f"box sides — {outside}", f"box weight — {mass}", f"grip — palms flat on the two side faces, {press} each, no support below",
                              f"walk or lift — {self._move(v, 'third')}"))
        return (f"I am {a_or_an(v['robot'])} with {palms}, working in the {v['domain']}. I'm holding {a_or_an(v['item_name'])} pressed between my "
                f"flat palms, {press} each and nothing underneath; it is {size}, its sides are {outside}, and it weighs {mass}. {self._move(v, 'first')}")

    def _options(self, v: dict[str, Any], plan: dict[str, Any]) -> dict[str, str]:
        opts = plan["option_values"]
        if v["move"] == "walk":
            return {"as_planned": "keep the planned press and walk as planned",
                    "firmer_harsher": f"press {opts['opt_firm']['squeeze']:.0f} N each but walk faster, {opts['opt_firm']['hz']:.2f} steps per second",
                    "lighter_gentler": f"press only {opts['opt_gentle']['squeeze']:.0f} N each but walk smoothly, the hands bobbing {cm(opts['opt_gentle']['rise'])}"}
        return {"as_planned": "keep the planned press and lift as planned",
                "firmer_harsher": f"press {opts['opt_firm']['squeeze']:.0f} N each but lift at {opts['opt_firm']['accel']:.1f} m/s²",
                "lighter_gentler": f"press only {opts['opt_gentle']['squeeze']:.0f} N each and lift gently at {opts['opt_gentle']['accel']:.1f} m/s²"}

    def questions(self, v: dict[str, Any], mix: list[str], plan: dict[str, Any], person: str, rng: random.Random, cf_option: str,
                  action_keys: list[str]) -> dict[str, Any]:
        item = v["item_name"]
        what = "walk" if v["move"] == "walk" else "lift"
        hands = {"third": "its palms", "second": "your palms", "first": "my palms"}[person]
        now, other = f"{v['squeeze']:.0f} N", f"{plan['cf_value']:.0f} N"
        out: dict[str, Any] = {}
        out["box_slips_between_palms"] = {"type": "noul", "instructions": rng.choice((
            f"Over the {what}, does the {item} sink 1 cm or more between {hands}, or slip out?",
            f"Does the {item} creep down at least 1 cm between {hands} (or fall) over the {what}?"))}
        out["press_change_effect"] = {"type": "choice", "instructions": rng.choice((
            f"What if each palm pressed {other} in place of {now}: would a slip or fall of the {item} become more likely, less likely, or stay about as "
            "likely?",
            f"Pressing {other} per palm instead of {now} — how does that shift the odds that the {item} slips or falls?")),
            "criteria": dict(PALMS_DIRECTION)}
        if "outcome" in mix:
            out["box_after_carry"] = {"type": "choice", "instructions": f"After the {what}, where is the {item}?",
                                      "criteria": {"secure": "still where it was between the palms (under 1 cm)",
                                                   "slid_lower": "sunk 1 cm or more but still held", "on_the_floor": "slipped free and fell"}}
        options = self._options(v, plan)
        if "action" in mix:
            out["safest_carry"] = {"type": "choice", "instructions": f"For the {item}, which way to go is safest?",
                                   "criteria": {k: options[k] for k in action_keys}}
        if "compare" in mix:
            out["riskier_carry"] = {"type": "choice", "instructions": f"Is a slip or fall of the {item} more likely if you {options['as_planned']}, or if "
                                    f"you {options[cf_option]}?",
                                    "criteria": {"as_planned": "carrying on as planned", cf_option: "the alternative",
                                                 "about_same": "no real difference (under 0.1 apart)"}}
        if "score_cf" in mix:
            out["slip_risk_other_press"] = {"type": "score", "instructions": f"Give the chance, from 0 to 1, that the {item} slips or falls if each palm "
                                            f"pressed {other} rather than {now}.", "criteria": ["0 = no chance it moves", "1 = it will surely slip or fall"]}
        if "severe" in mix:
            out["box_slips_free"] = {"type": "noul", "instructions": f"Does the {item} slip free and fall during the {what}?"}
        return out

    def catalogs(self) -> dict[str, Any]:
        return {"palms": PALM_COVERS, "surfaces": PALM_SURFACES, "items": PALM_ITEMS, "robots": PALM_ROBOTS, "domains": PALM_DOMAINS}

    def catalog_names(self) -> set[str]:
        out = {item["name"].lower() for item in PALM_ITEMS} | {r.lower() for r in PALM_ROBOTS}
        out |= {x["text"].lower() for x in list(PALM_COVERS.values()) + list(PALM_SURFACES.values())}
        return out


#: direction-option wording of the held-out palms scene (as the tray's: no text unit shared with the training scenes)
PALMS_DIRECTION = {"more_likely": "more likely than as planned", "less_likely": "less likely than as planned",
                   "about_same": "about as likely (a change smaller than 0.1)"}
