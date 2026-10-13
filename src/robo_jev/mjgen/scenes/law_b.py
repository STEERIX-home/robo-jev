"""Law B — centre of mass past an edge: three new scenes (:mod:`robo_jev.mjgen.scenes.rigs`).

* ``table`` — an arm sets an item down partly over a table's edge and lets go; it falls when its centre of mass (placement error and
  where its weight sits inside included) is past the edge (ledge engine).
* ``reach`` — a wheeled base with an arm reaches out holding a payload; the base tips over its front wheels when the centre of mass of
  base and payload passes them, earlier when the arm stops hard at the end of a fast reach (shift engine).
* ``cabinet`` — a robot pulls a loaded drawer out of a narrow filing cabinet; the cabinet tips forward when the centre of mass of cabinet
  and drawer passes its front feet, earlier when a quick pull stops the loaded drawer on the cabinet's rails (shift engine).

Hidden values (never written): where the weight sits inside (contents or ballast words), placement error (table), the payload's or the
drawer contents' mass when only a range is known, and the floor or table friction. The decisive number is the overhang, the reach or the
drawer's extension.
"""

from __future__ import annotations

import json
import math
import random
from typing import Any

from robo_jev.mjgen import families as fam
from robo_jev.mjgen.scenes import rigs
from robo_jev.mjgen.scenes.base import Scene, a_or_an, cm, num

__all__ = ["CabinetScene", "ReachScene", "TableScene"]

G = rigs.GRAVITY


# ======================================================================================================================================
# table — an item set down partly over a table's edge
# ======================================================================================================================================

TABLE_TOPS = {
    "laminate": {"text": "laminate worktop", "mu": (0.30, 0.42)},
    "steel": {"text": "stainless-steel bench top", "mu": (0.20, 0.30)},
    "oak": {"text": "oiled oak tabletop", "mu": (0.38, 0.52)},
    "resin": {"text": "epoxy-resin lab bench", "mu": (0.28, 0.40)},
    "mat": {"text": "bench covered by a rubber work mat", "mu": (0.70, 0.95)},
}
TABLE_ITEMS = (
    {"key": "laptop", "name": "laptop in a padded case", "dims": ((0.36, 0.42), (0.26, 0.30), (0.04, 0.06)), "mass": (1.8, 3.2),
     "contents": ("even", "unknown")},
    {"key": "reagents", "name": "carton of reagent bottles", "dims": ((0.28, 0.36), (0.20, 0.26), (0.18, 0.26)), "mass": (3.0, 7.0),
     "contents": ("even", "out", "in", "unknown")},
    {"key": "casserole", "name": "lidded cast-iron casserole", "dims": ((0.30, 0.36), (0.24, 0.28), (0.14, 0.18)), "mass": (4.0, 7.0),
     "contents": ("even",)},
    {"key": "psu", "name": "bench power supply", "dims": ((0.24, 0.32), (0.20, 0.26), (0.10, 0.16)), "mass": (3.0, 8.0),
     "contents": ("in", "out", "even")},
    {"key": "books", "name": "stack of hardback books", "dims": ((0.24, 0.30), (0.17, 0.22), (0.12, 0.24)), "mass": (2.0, 6.0),
     "contents": ("even",)},
    {"key": "cooler", "name": "sealed specimen cooler", "dims": ((0.34, 0.44), (0.24, 0.30), (0.24, 0.32)), "mass": (2.5, 8.0),
     "contents": ("even", "unknown", "out", "in")},
    {"key": "oranges", "name": "crate of oranges", "dims": ((0.40, 0.50), (0.30, 0.36), (0.18, 0.26)), "mass": (6.0, 14.0),
     "contents": ("even", "unknown")},
    {"key": "kit", "name": "first-aid kit case", "dims": ((0.30, 0.38), (0.10, 0.16), (0.22, 0.28)), "mass": (1.5, 4.0),
     "contents": ("even", "out", "in", "unknown")},
)
#: contents words → fore-aft centre-of-mass offset (fraction of the item's length; + = toward the end that sticks out)
TABLE_CONTENTS = {
    "even": {"cx": (-0.02, 0.02), "text": "its weight is spread evenly along it"},
    "out": {"cx": (0.06, 0.16), "text": "its heavy end is the end that sticks out"},
    "in": {"cx": (-0.16, -0.06), "text": "its heavy end stays over the table"},
    "unknown": {"cx": (-0.12, 0.12), "text": "which end is heavier is not known"},
}
TABLE_ROBOTS = ("bench-top cobot", "kitchen assistant arm", "rail-mounted workshop arm")
TABLE_DOMAINS = ("test kitchen", "electronics workshop", "chemistry lab", "school canteen", "repair bench", "bakery", "dental lab", "print shop")


class TableScene(Scene):
    name = "table"
    law = "com_edge"
    outcomes = ("stays", "rocks", "falls")
    event = ("rocks", "falls")
    severe = "falls"
    question_names = {"key": "item_goes_over_edge", "direction": "overhang_change_effect", "outcome": "item_after_release", "action": "safest_placement",
                      "compare": "riskier_placement", "score_cf": "fall_risk_other_overhang", "severe": "item_falls_off"}
    outcome_keys = {"stays_flat": "stays", "rocks_back": "rocks", "falls_off": "falls"}
    dominant_option = "further_in"
    factors = {
        "mass_distribution": {"field": "contents", "fact": "word", "how": "word"},
        "footprint": {"field": "dims.0", "fact": "number", "how": "aim", "span": lambda c, v: (max(0.08, c * 0.6), c * 1.6, 0.01)},
        "release_height": {"field": "drop", "fact": "number", "how": "choice"},
        "placement_scatter": {"field": "sigma", "fact": "number", "how": "choice"},
    }
    pair_fields = (("decisive", 0.7), ("dims.0", 0.3))
    pair_spans = {"dims.0": lambda c, v: (max(0.08, c * 0.6), c * 1.6, 0.01)}
    domains = TABLE_DOMAINS
    physics_fields = ("dims", "mass", "overhang", "drop", "turned")

    def sample(self, rng: random.Random) -> tuple[dict[str, Any], dict[str, Any]]:
        item = rng.choice(TABLE_ITEMS)
        dims = [round(rng.uniform(*span), 2) for span in item["dims"]]
        v = {"robot": rng.choice(TABLE_ROBOTS), "domain": rng.choice(TABLE_DOMAINS), "item": item["key"], "item_name": item["name"], "dims": dims,
             "mass": round(rng.uniform(*item["mass"]), 1), "contents": rng.choice(item["contents"]), "top": rng.choice(sorted(TABLE_TOPS)),
             "drop": rng.choice((0.0, 0.005, 0.01, 0.02)), "sigma": rng.choice((0.003, 0.005, 0.01, 0.015)), "overhang": 0.0, "turned": False}
        return v, {"field": "overhang", "lo": 0.005, "hi": round(0.95 * dims[0], 3), "step": 0.005, "increasing": True}

    def hidden_for(self, v: dict[str, Any]) -> dict[str, Any]:
        return {"mu": ("mu",) + TABLE_TOPS[v["top"]]["mu"], "cx": ("u",) + TABLE_CONTENTS[v["contents"]]["cx"] + (-0.40, 0.40), "eps": ("n", 0.0, v["sigma"])}

    def accept(self, v: dict[str, Any]) -> bool:
        return 0.0 < v["overhang"] < v["dims"][0]

    def _centre(self, v: dict[str, Any], draw: dict[str, float]) -> tuple[float, float]:
        """(x of the item's centre, x of its centre of mass) relative to the edge (+ = beyond it)."""
        x = v["overhang"] - v["dims"][0] / 2 + draw["eps"]
        sign = -1.0 if v.get("turned") else 1.0
        return x, x + sign * draw["cx"] * v["dims"][0]

    def static_outcome(self, v: dict[str, Any], draw: dict[str, float]) -> str:
        return "falls" if self._centre(v, draw)[1] > 0 else "stays"

    def roll(self, v: dict[str, Any], draw: dict[str, float], rigs_cache: dict[str, Any]) -> dict[str, Any]:
        key = json.dumps(["table", v["dims"], v["mass"]])
        if key not in rigs_cache:
            rigs_cache[key] = rigs.ledge_rig(v["dims"], v["mass"])
        x, _ = self._centre(v, draw)
        sign = -1.0 if v.get("turned") else 1.0
        return rigs.run_ledge(rigs_cache[key], x_centre=x, drop=v["drop"], mu=draw["mu"], com=(sign * draw["cx"] * v["dims"][0], 0.0, 0.0))

    def xml_example(self, v: dict[str, Any]) -> str:
        return rigs.ledge_xml(v["dims"], v["mass"])

    def options(self, group: Any, rng: random.Random) -> None:
        v = group.visible
        back = fam._round(max(0.005, v["overhang"] - rng.uniform(0.25, 0.6) * v["dims"][0] / 2), 0.005)
        group.variants.update({"opt_in": {"overhang": back}, "opt_turned": {"turned": True}})
        group.plan["options"] = {"as_planned": "planned", "further_in": "opt_in", "turned_round": "opt_turned"}

    def factor_choices(self, group: Any, factor: str, current: Any) -> list[Any]:
        v = group.visible
        item = next(i for i in TABLE_ITEMS if i["key"] == v["item"])
        if factor == "mass_distribution":
            return [c for c in sorted(TABLE_CONTENTS) if c != current and (c in item["contents"] or c == "unknown")]
        if factor == "release_height":
            return [d for d in (0.0, 0.02, 0.04, 0.06) if abs(d - current) >= 0.02]
        if factor == "placement_scatter":
            return [s for s in (0.002, 0.005, 0.01, 0.02, 0.03) if abs(s - current) >= 0.004]
        return []

    def _place(self, v: dict[str, Any], person: str) -> str:
        subject = {"third": "The arm", "second": "You", "first": "I"}[person]
        drop = "touching the surface" if v["drop"] == 0 else f"{cm(v['drop'])} above the surface"
        return (f"{subject} will line it up at right angles to the edge, leaving {cm(v['overhang'])} of its length hanging out over the floor, and "
                f"open the gripper with it {drop}; the placement error is typically up to {cm(v['sigma'])} either way.")

    def state_text(self, v: dict[str, Any], style: str, rng: random.Random) -> str:
        length, width, height = v["dims"]
        top = TABLE_TOPS[v["top"]]["text"]
        inside = TABLE_CONTENTS[v["contents"]]["text"]
        size = f"{num(length)} m long (the direction it will stick out), {num(width)} m wide, {num(height)} m high"
        if style == "log":
            return (f"Bench log, {v['domain']}. Unit: {a_or_an(v['robot'])}. Holding: one {v['item_name']}, {size}, {v['mass']:.1f} kg; {inside}. "
                    f"Target: the straight edge of a {top}, nothing below it but the floor. {self._place(v, 'third')}")
        if style == "brief":
            return (f"You are {a_or_an(v['robot'])} in {a_or_an(v['domain'])}, holding {a_or_an(v['item_name'])} ({size}; {v['mass']:.1f} kg); {inside}. "
                    f"You are placing it at the edge of a {top}, with only the floor below. {self._place(v, 'second')}")
        if style == "sheet":
            return "\n".join((f"robot: {v['robot']} ({v['domain']})", f"item: {v['item_name']}", f"item size: {size}", f"item weight: {v['mass']:.1f} kg",
                              f"weight inside: {inside}", f"surface: {top}, straight edge, floor below", f"placement: {self._place(v, 'third')}"))
        return (f"I'm {a_or_an(v['robot'])} working in the {v['domain']} with {a_or_an(v['item_name'])} in my gripper: {size}, {v['mass']:.1f} kg; "
                f"{inside}. I'm putting it on a {top}, right at the edge, with the floor below. {self._place(v, 'first')}")

    def _options(self, v: dict[str, Any], plan: dict[str, Any]) -> dict[str, str]:
        back = plan["option_values"]["opt_in"]["overhang"]
        return {"as_planned": f"place it with {cm(v['overhang'])} sticking out, as planned",
                "further_in": f"place it with only {cm(back)} sticking out",
                "turned_round": "turn it end for end, then place it with the same length sticking out"}

    def questions(self, v: dict[str, Any], mix: list[str], plan: dict[str, Any], person: str, rng: random.Random, cf_option: str,
                  action_keys: list[str]) -> dict[str, Any]:
        item = v["item_name"]
        now, other = cm(v["overhang"]), cm(plan["cf_value"])
        let = {"third": "the gripper opens", "second": "your gripper opens", "first": "my gripper opens"}[person]
        out: dict[str, Any] = {}
        out["item_goes_over_edge"] = {"type": "noul", "instructions": rng.choice((
            f"When {let}, does the {item} tilt out over the edge by 4° or more, or drop to the floor?",
            f"After the release, does the {item} rock out over the edge (4° or more) or go over it?"))}
        out["overhang_change_effect"] = {"type": "choice", "instructions": rng.choice((
            f"Suppose {other} stuck out instead of {now}. Would the {item} be more likely, less likely or about as likely to tip up or fall?",
            f"Against the planned overhang of {now}, what would an overhang of {other} do to the chance that the {item} tips up or falls?")),
            "criteria": dict(TABLE_DIRECTION)}
        if "outcome" in mix:
            out["item_after_release"] = {"type": "choice", "instructions": f"What does the {item} do in the first seconds after the release?",
                                         "criteria": {"stays_flat": "it stays flat on the surface", "rocks_back": "it tips up 4° or more, then settles back",
                                                      "falls_off": "it goes over the edge"}}
        options = self._options(v, plan)
        if "action" in mix:
            out["safest_placement"] = {"type": "choice", "instructions": f"Which way of placing the {item} is safest?",
                                       "criteria": {k: options[k] for k in action_keys}}
        if "compare" in mix:
            out["riskier_placement"] = {"type": "choice", "instructions": f"Which is more likely to end with the {item} tipping up or falling: the plan as "
                                        f"it stands, or to {options[cf_option]}?",
                                        "criteria": {"as_planned": "sticking to the plan", cf_option: "the other placement",
                                                     "about_same": "roughly equal (within 0.1)"}}
        if "score_cf" in mix:
            out["fall_risk_other_overhang"] = {"type": "score", "instructions": f"Rate the chance that the {item} tips up or falls if the overhang were "
                                               f"{other} rather than {now}.", "criteria": ["0 = it will sit there flat", "1 = it is certain to tip up or fall"]}
        if "severe" in mix:
            out["item_falls_off"] = {"type": "noul", "instructions": f"Does the {item} end up falling off the edge?"}
        return out

    def catalogs(self) -> dict[str, Any]:
        return {"tops": TABLE_TOPS, "items": TABLE_ITEMS, "contents": TABLE_CONTENTS, "robots": TABLE_ROBOTS, "domains": TABLE_DOMAINS}

    def catalog_names(self) -> set[str]:
        out = {item["name"].lower() for item in TABLE_ITEMS} | {r.lower() for r in TABLE_ROBOTS}
        out |= {x["text"].lower() for x in TABLE_TOPS.values()}
        return out


TABLE_DIRECTION = {"more_likely": "a higher chance than in the plan", "less_likely": "a lower chance than in the plan",
                   "about_same": "roughly unchanged (it moves by less than 0.1)"}


# ======================================================================================================================================
# shift scenes — a base that tips when a mover inside it runs out
# ======================================================================================================================================


class ShiftScene(Scene):
    law = "com_edge"
    outcomes = ("steady", "lifts", "tips")
    event = ("lifts", "tips")
    severe = "tips"
    physics_fields = ("geom", "reach", "extension", "duration")

    def geometry(self, v: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError

    def travel(self, v: dict[str, Any]) -> float:
        raise NotImplementedError

    def mover_mass(self, v: dict[str, Any], draw: dict[str, float]) -> float:
        raise NotImplementedError

    def front_edge(self, v: dict[str, Any]) -> float:
        g = self.geometry(v)
        return g["feet_x"][1] + rigs.FOOT_HALF[0]

    def static_outcome(self, v: dict[str, Any], draw: dict[str, float]) -> str:
        """``tips`` when the resting centre of mass of base and mover is past the front feet; ``lifts`` when it is not, but the mover's
        stop at the end of the run (peak deceleration 2π·travel/duration², at the mover's height) briefly turns the moment about the
        front feet forward; else ``steady``. Aiming only — the labels are MuJoCo's."""
        g = self.geometry(v)
        m = self.mover_mass(v, draw)
        x_m = g["mover_at"][0] + self.travel(v)
        x_b = draw["cx"] * g["base_dims"][0]
        edge = self.front_edge(v)
        moment = G * (g["base_mass"] * (x_b - edge) + m * (x_m - edge))
        if moment > 0:
            return "tips"
        height = g["base_z"] + g["base_dims"][2] / 2 + g["mover_at"][2]
        jolt = m * (2 * math.pi * self.travel(v) / v["duration"] ** 2) * height
        return "lifts" if moment + jolt > 0 else "steady"

    def static_travel(self, v: dict[str, Any], mover_mass: float) -> float:
        """The travel at which the centre of mass of base (centred) and mover passes the front feet's edge."""
        g = self.geometry(v)
        return ((g["base_mass"] + mover_mass) * self.front_edge(v)) / mover_mass - g["mover_at"][0]

    def rig_for(self, v: dict[str, Any], rigs_cache: dict[str, Any]) -> Any:
        g = self.geometry(v)
        key = json.dumps([self.name, g], sort_keys=True)
        if key not in rigs_cache:
            rigs_cache[key] = rigs.shift_rig(self.name, **{k: (tuple(x) if isinstance(x, list) else x) for k, x in g.items()})
        return rigs_cache[key]

    def roll(self, v: dict[str, Any], draw: dict[str, float], rigs_cache: dict[str, Any]) -> dict[str, Any]:
        rig = self.rig_for(v, rigs_cache)
        g = self.geometry(v)
        path = rigs.shift_profile(self.travel(v), v["duration"])
        base_com = (draw["cx"] * g["base_dims"][0], 0.0, -0.15 * g["base_dims"][2])
        return rigs.run_shift(rig, path, mu=draw["mu"], base_com=base_com, mover_mass=self.mover_mass(v, draw))

    def physics_key(self, v: dict[str, Any]) -> str:
        return json.dumps({"geom": self.geometry(v), "travel": self.travel(v), "duration": v["duration"]}, sort_keys=True)

    def xml_example(self, v: dict[str, Any]) -> str:
        return rigs.shift_xml(self.name, **self.geometry(v))


# --------------------------------------------------------------------------
# reach — a wheeled base reaching out with a payload
# --------------------------------------------------------------------------

REACH_BALLAST = {
    "centred": {"cx": (-0.03, 0.03), "text": "its batteries sit in the middle of the chassis"},
    "rear": {"cx": (-0.16, -0.08), "text": "its battery pack is mounted at the back of the chassis"},
    "front": {"cx": (0.06, 0.14), "text": "its battery pack is mounted at the front of the chassis"},
    "unknown": {"cx": (-0.12, 0.12), "text": "the layout of its batteries is not documented"},
}
REACH_FLOORS = {
    "concrete": {"text": "sealed concrete", "mu": (0.70, 0.90)},
    "epoxy": {"text": "glossy epoxy floor paint", "mu": (0.45, 0.60)},
    "rubber": {"text": "studded rubber flooring", "mu": (0.85, 1.10)},
}
REACH_PAYLOADS = (
    {"key": "jerrycan", "name": "jerrycan of coolant", "dims": ((0.30, 0.36), (0.16, 0.20), (0.36, 0.44)), "mass": ("range", 3.0, 22.0)},
    {"key": "gearbox", "name": "spare gearbox", "dims": ((0.30, 0.40), (0.24, 0.30), (0.24, 0.30)), "mass": ("about", 14.0, 30.0)},
    {"key": "batteries", "name": "crate of replacement batteries", "dims": ((0.36, 0.44), (0.28, 0.34), (0.20, 0.26)), "mass": ("range", 8.0, 32.0)},
    {"key": "sandbag", "name": "bag of potting soil", "dims": ((0.40, 0.50), (0.28, 0.34), (0.12, 0.16)), "mass": ("about", 10.0, 25.0)},
    {"key": "valve", "name": "brass gate valve", "dims": ((0.22, 0.30), (0.18, 0.22), (0.20, 0.28)), "mass": ("about", 9.0, 20.0)},
    {"key": "tote", "name": "returns tote, contents unknown", "dims": ((0.40, 0.50), (0.30, 0.36), (0.22, 0.28)), "mass": ("range", 2.0, 28.0)},
)
REACH_ROBOTS = ("long-reach arm on a wheeled base", "autonomous base carrying a six-axis arm", "inspection rover with a manipulator arm")
REACH_DOMAINS = ("loading bay", "greenhouse", "hospital store room", "parcel hub", "data-centre aisle", "brewery cellar", "machine shop", "seed bank")
X0 = 0.25


class ReachScene(ShiftScene):
    name = "reach"
    question_names = {"key": "base_tips_or_lifts", "direction": "reach_change_effect", "outcome": "base_after_reach", "action": "safest_reach_plan",
                      "compare": "riskier_reach_plan", "score_cf": "tip_risk_other_reach", "severe": "base_goes_over"}
    outcome_keys = {"steady": "steady", "wheels_lift": "lifts", "tips_over": "tips"}
    factors = {
        "ballast": {"field": "ballast", "fact": "word", "how": "word"},
        "floor": {"field": "floor", "fact": "word", "how": "word"},
        "wheelbase": {"field": "wheelbase", "fact": "number", "how": "aim", "span": lambda c, v: (max(0.3, c * 0.6), c * 1.5, 0.01)},
        "reach_time": {"field": "duration", "fact": "number", "how": "choice"},
        "arm_height": {"field": "arm_h", "fact": "number", "how": "choice"},
    }
    pair_fields = (("decisive", 0.6), ("base_mass", 0.4))
    pair_spans = {"base_mass": lambda c, v: (max(15.0, c * 0.5), c * 2.2, 1.0)}
    domains = REACH_DOMAINS

    def geometry(self, v: dict[str, Any]) -> dict[str, Any]:
        length, width, height = v["base_dims"]
        half = v["wheelbase"] / 2
        base_z = 0.06
        centre_z = base_z + height / 2
        return {"base_dims": [length, width, height], "base_mass": v["base_mass"], "base_z": base_z, "feet_x": [-half, half], "feet_y": width / 2 - 0.05,
                "mover_dims": list(v["payload_dims"]), "mover_mass": 10.0, "mover_at": [X0, 0.0, round(v["arm_h"] - centre_z, 4)], "travel": 2.0,
                "link": [[0.6, 0.02, 0.02], [-0.6, 0.0, 0.0]]}

    def travel(self, v: dict[str, Any]) -> float:
        return max(0.0, v["reach"] - X0)

    def mover_mass(self, v: dict[str, Any], draw: dict[str, float]) -> float:
        return draw["mass"]

    def sample(self, rng: random.Random) -> tuple[dict[str, Any], dict[str, Any]]:
        payload = rng.choice(REACH_PAYLOADS)
        kind, lo, hi = payload["mass"]
        length = round(rng.uniform(0.50, 0.90), 2)
        v = {"robot": rng.choice(REACH_ROBOTS), "domain": rng.choice(REACH_DOMAINS), "payload": payload["key"], "payload_name": payload["name"],
             "payload_dims": [round(rng.uniform(*span), 2) for span in payload["dims"]], "mass_kind": kind,
             "base_dims": [length, round(rng.uniform(0.40, 0.70), 2), round(rng.uniform(0.25, 0.45), 2)], "base_mass": float(round(rng.uniform(30.0, 90.0))),
             "wheelbase": round(length - rng.uniform(0.06, 0.14), 2), "ballast": rng.choice(sorted(REACH_BALLAST)), "floor": rng.choice(sorted(REACH_FLOORS)),
             "arm_h": round(rng.uniform(0.6, 1.2), 2), "duration": round(rng.uniform(1.0, 4.0), 1), "reach": 0.0}
        if kind == "about":
            v["mass"] = round(rng.uniform(lo, hi), 1)
        else:
            v["mass_lo"], v["mass_hi"] = round(lo, 1), round(hi, 1)
        return v, {"field": "reach", "lo": 0.30, "hi": 2.20, "step": 0.01, "increasing": True}

    def hidden_for(self, v: dict[str, Any]) -> dict[str, Any]:
        mass = (v["mass"] * 0.95, v["mass"] * 1.05) if v["mass_kind"] == "about" else (v["mass_lo"], v["mass_hi"])
        return {"mu": ("mu",) + REACH_FLOORS[v["floor"]]["mu"], "cx": ("u",) + REACH_BALLAST[v["ballast"]]["cx"] + (-0.40, 0.40),
                "mass": ("u",) + tuple(mass) + (0.5, 200.0)}

    def accept(self, v: dict[str, Any]) -> bool:
        return v["reach"] > X0 + 0.05 and v["wheelbase"] < v["base_dims"][0]

    def options(self, group: Any, rng: random.Random) -> None:
        v = group.visible
        closer = fam._round(max(X0 + 0.06, v["reach"] - rng.uniform(0.15, 0.30)), 0.01)
        group.variants.update({"opt_closer": {"reach": closer, "duration": round(max(0.4, v["duration"] * rng.uniform(0.3, 0.5)), 1)},
                               "opt_slow": {"duration": round(min(8.0, v["duration"] * rng.uniform(2.0, 3.0)), 1)}})
        group.plan["options"] = {"as_planned": "planned", "closer_quicker": "opt_closer", "slower_reach": "opt_slow"}

    def factor_choices(self, group: Any, factor: str, current: Any) -> list[Any]:
        if factor == "ballast":
            return sorted(REACH_BALLAST)
        if factor == "floor":
            return sorted(REACH_FLOORS)
        if factor == "reach_time":
            return [d for d in (0.5, 0.8, 2.5, 5.0) if abs(d - current) >= 0.4]
        if factor == "arm_height":
            return [h for h in (0.5, 0.8, 1.2, 1.5) if abs(h - current) >= 0.25]
        return []

    def _plan(self, v: dict[str, Any], person: str) -> str:
        subject = {"third": "The robot", "second": "You", "first": "I"}[person]
        return (f"{subject} will stay parked and swing the arm straight out ahead, keeping the payload {num(v['arm_h'])} m off the floor, until it is "
                f"{num(v['reach'])} m in front of the middle of the base, taking {v['duration']:.1f} s to get there.")

    def _mass(self, v: dict[str, Any]) -> str:
        if v["mass_kind"] == "about":
            return f"about {v['mass']:.1f} kg"
        return f"anywhere from {v['mass_lo']:.1f} to {v['mass_hi']:.1f} kg (its fill is unknown)"

    def state_text(self, v: dict[str, Any], style: str, rng: random.Random) -> str:
        length, width, height = v["base_dims"]
        ballast = REACH_BALLAST[v["ballast"]]["text"]
        floor = REACH_FLOORS[v["floor"]]["text"]
        base = (f"a footprint of {num(length)} × {num(width)} m (length × width), {v['base_mass']:.0f} kg with the arm, axles {num(v['wheelbase'])} m "
                "apart")
        load = f"{v['payload_name']} ({self._mass(v)})"
        if style == "log":
            return (f"Ops log, {v['domain']}. Unit: {a_or_an(v['robot'])}; base {base}; {ballast}. Floor: {floor}, level. In the gripper: one {load}. "
                    f"{self._plan(v, 'third')}")
        if style == "brief":
            return (f"You are {a_or_an(v['robot'])} in {a_or_an(v['domain'])}: your base is {base}, and {ballast}. You stand on level {floor}, gripping "
                    f"{a_or_an(load)}. {self._plan(v, 'second')}")
        if style == "sheet":
            return "\n".join((f"robot: {v['robot']} ({v['domain']})", f"base: {base}", f"batteries: {ballast}", f"floor: {floor}, level",
                              f"payload: {load}", f"plan: {self._plan(v, 'third')}"))
        return (f"I'm {a_or_an(v['robot'])} at work in the {v['domain']}. My base is {base}, and {ballast}. I'm parked on level {floor} with "
                f"{a_or_an(load)} in my gripper. {self._plan(v, 'first')}")

    def _options(self, v: dict[str, Any], plan: dict[str, Any]) -> dict[str, str]:
        opts = plan["option_values"]
        return {"as_planned": f"reach out to {num(v['reach'])} m over {v['duration']:.1f} s, as planned",
                "closer_quicker": f"drive closer so the reach is only {num(opts['opt_closer']['reach'])} m, but swing out in {opts['opt_closer']['duration']:.1f} s",
                "slower_reach": f"reach out to the same {num(v['reach'])} m, taking {opts['opt_slow']['duration']:.1f} s"}

    def questions(self, v: dict[str, Any], mix: list[str], plan: dict[str, Any], person: str, rng: random.Random, cf_option: str,
                  action_keys: list[str]) -> dict[str, Any]:
        base = {"third": "the base", "second": "your base", "first": "my base"}[person]
        now, other = f"{num(v['reach'])} m", f"{num(plan['cf_value'])} m"
        load = v["payload_name"]
        out: dict[str, Any] = {}
        out["base_tips_or_lifts"] = {"type": "noul", "instructions": rng.choice((
            f"As the arm reaches out with the {load}, do the rear wheels of {base} lift off the floor (a tilt of 1.5° or more), or does it tip over?",
            f"Will {base} rock up onto its front wheels (1.5° or more) or go over during the reach?"))}
        out["reach_change_effect"] = {"type": "choice", "instructions": rng.choice((
            f"Suppose the reach were {other} rather than {now}. Would a lift or tip-over of {base} be more likely, less likely or about as likely?",
            f"Against the planned reach of {now}, what would a reach of {other} do to the chance that {base} lifts or tips?")),
            "criteria": dict(TABLE_DIRECTION)}
        if "outcome" in mix:
            out["base_after_reach"] = {"type": "choice", "instructions": f"What does {base} do during the reach?",
                                       "criteria": {"steady": "all four wheels stay down", "wheels_lift": "the rear wheels lift, then come back down",
                                                    "tips_over": "it tips over forward"}}
        options = self._options(v, plan)
        if "action" in mix:
            out["safest_reach_plan"] = {"type": "choice", "instructions": f"Which plan is safest for placing the {load}?",
                                        "criteria": {k: options[k] for k in action_keys}}
        if "compare" in mix:
            out["riskier_reach_plan"] = {"type": "choice", "instructions": f"Which is more likely to lift or tip {base}: sticking to the plan, or to "
                                         f"{options[cf_option]}?",
                                         "criteria": {"as_planned": "sticking to the plan", cf_option: "the alternative plan",
                                                      "about_same": "roughly equal (within 0.1)"}}
        if "score_cf" in mix:
            out["tip_risk_other_reach"] = {"type": "score", "instructions": f"Rate the chance that {base} lifts or tips if the reach were {other} rather "
                                           f"than {now}.", "criteria": ["0 = it will not budge", "1 = it is certain to lift or tip"]}
        if "severe" in mix:
            out["base_goes_over"] = {"type": "noul", "instructions": f"Does {base} actually tip over?"}
        return out

    def sanity_cases(self) -> list[tuple[dict[str, Any], float]]:
        base = {"base_dims": [0.6, 0.5, 0.3], "base_mass": 40.0, "wheelbase": 0.5, "payload_dims": [0.3, 0.25, 0.25], "arm_h": 0.9, "duration": 6.0}
        return [(base, 10.0), ({**base, "base_mass": 60.0, "wheelbase": 0.4}, 25.0)]

    def catalogs(self) -> dict[str, Any]:
        return {"ballast": REACH_BALLAST, "floors": REACH_FLOORS, "payloads": REACH_PAYLOADS, "robots": REACH_ROBOTS, "domains": REACH_DOMAINS}

    def catalog_names(self) -> set[str]:
        out = {item["name"].lower() for item in REACH_PAYLOADS} | {r.lower() for r in REACH_ROBOTS}
        out |= {x["text"].lower() for x in REACH_FLOORS.values()}
        return out


# --------------------------------------------------------------------------
# cabinet — a loaded drawer pulled out of a narrow filing cabinet
# --------------------------------------------------------------------------

CABINET_CONTENTS = {
    "files": {"mass": (18.0, 32.0), "text": "packed tight with hanging files"},
    "tools": {"mass": (22.0, 40.0), "text": "full of hand tools"},
    "half": {"mass": (8.0, 16.0), "text": "about half full of binders"},
    "light": {"mass": (1.5, 5.0), "text": "holding only a few folders"},
    "unknown": {"mass": (1.5, 36.0), "text": "holding contents nobody has listed"},
}
CABINET_BALLAST = {
    "none": {"cx": (-0.03, 0.03), "text": "the other drawers are empty"},
    "back": {"cx": (-0.16, -0.08), "text": "a ballast plate is bolted at the back of its base"},
    "unknown": {"cx": (-0.10, 0.10), "text": "what is in the other drawers is not known"},
}
CABINET_FLOORS = {
    "carpet": {"text": "cord carpet tiles", "mu": (0.60, 0.85)},
    "vinyl": {"text": "polished vinyl", "mu": (0.35, 0.50)},
    "tile": {"text": "ceramic tiles", "mu": (0.45, 0.60)},
}
CABINET_KINDS = (
    {"key": "filing4", "name": "four-drawer filing cabinet", "depth": (0.60, 0.70), "width": (0.45, 0.50), "height": (1.30, 1.40), "mass": (28.0, 45.0), "drawers": 4},
    {"key": "filing3", "name": "three-drawer filing cabinet", "depth": (0.60, 0.68), "width": (0.45, 0.50), "height": (1.00, 1.05), "mass": (22.0, 36.0), "drawers": 3},
    {"key": "toolchest", "name": "tall mechanic's tool chest", "depth": (0.45, 0.55), "width": (0.65, 0.80), "height": (0.95, 1.10), "mass": (35.0, 60.0), "drawers": 5},
    {"key": "plan", "name": "tall sample-storage cabinet", "depth": (0.50, 0.60), "width": (0.40, 0.50), "height": (1.50, 1.80), "mass": (25.0, 40.0), "drawers": 5},
)
CABINET_ROBOTS = ("office helper robot with an arm", "pharmacy picking robot", "workshop rover with a gripper arm")
CABINET_DOMAINS = ("records office", "school office", "hospital records room", "machine shop", "law firm archive", "architect's studio", "garage workshop",
                   "university department")
DRAWER_SHELL = 3.0


class CabinetScene(ShiftScene):
    name = "cabinet"
    question_names = {"key": "cabinet_tips_or_rocks", "direction": "extension_change_effect", "outcome": "cabinet_after_pull",
                      "action": "safest_pull_plan", "compare": "riskier_pull_plan", "score_cf": "tip_risk_other_extension", "severe": "cabinet_falls"}
    outcome_keys = {"steady": "steady", "rocks_forward": "lifts", "falls_forward": "tips"}
    factors = {
        "drawer_contents": {"field": "contents", "fact": "word", "how": "word"},
        "ballast": {"field": "ballast", "fact": "word", "how": "word"},
        "floor": {"field": "floor", "fact": "word", "how": "word"},
        "pull_time": {"field": "duration", "fact": "number", "how": "choice"},
        "drawer_level": {"field": "level", "fact": "word", "how": "choice"},
    }
    pair_fields = (("decisive", 0.7), ("cab_mass", 0.3))
    pair_spans = {"cab_mass": lambda c, v: (max(12.0, c * 0.5), c * 2.2, 1.0)}
    domains = CABINET_DOMAINS

    def geometry(self, v: dict[str, Any]) -> dict[str, Any]:
        depth, width, height = v["cab_dims"]
        base_z = 0.03
        centre_z = base_z + height / 2
        drawers = v["drawers"]
        slot = (height - 0.06) / drawers
        index = {"top": drawers - 1, "middle": drawers // 2, "bottom": 0}[v["level"]]
        z = base_z + 0.05 + slot * (index + 0.5)
        drawer = [round(depth - 0.04, 3), round(width - 0.06, 3), round(slot * 0.8, 3)]
        return {"base_dims": [depth, width, height], "base_mass": v["cab_mass"], "base_z": base_z, "feet_x": [-(depth / 2 - 0.04), depth / 2 - 0.04],
                "feet_y": width / 2 - 0.04, "mover_dims": drawer, "mover_mass": 10.0, "mover_at": [0.0, 0.0, round(z - centre_z, 4)],
                "travel": round(depth, 3), "link": None}

    def travel(self, v: dict[str, Any]) -> float:
        return v["extension"]

    def mover_mass(self, v: dict[str, Any], draw: dict[str, float]) -> float:
        return DRAWER_SHELL + draw["mass"]

    def sample(self, rng: random.Random) -> tuple[dict[str, Any], dict[str, Any]]:
        kind = rng.choice(CABINET_KINDS)
        depth = round(rng.uniform(*kind["depth"]), 2)
        v = {"robot": rng.choice(CABINET_ROBOTS), "domain": rng.choice(CABINET_DOMAINS), "cabinet": kind["key"], "cabinet_name": kind["name"],
             "cab_dims": [depth, round(rng.uniform(*kind["width"]), 2), round(rng.uniform(*kind["height"]), 2)],
             "cab_mass": float(round(rng.uniform(*kind["mass"]))), "drawers": kind["drawers"], "level": rng.choice(("top", "top", "middle", "bottom")),
             "contents": rng.choice(sorted(CABINET_CONTENTS)), "ballast": rng.choice(sorted(CABINET_BALLAST)), "floor": rng.choice(sorted(CABINET_FLOORS)),
             "duration": round(rng.uniform(0.6, 2.5), 1), "extension": 0.0}
        return v, {"field": "extension", "lo": 0.05, "hi": round(0.95 * (depth - 0.04), 2), "step": 0.01, "increasing": True}

    def hidden_for(self, v: dict[str, Any]) -> dict[str, Any]:
        return {"mu": ("mu",) + CABINET_FLOORS[v["floor"]]["mu"], "cx": ("u",) + CABINET_BALLAST[v["ballast"]]["cx"] + (-0.40, 0.40),
                "mass": ("u",) + CABINET_CONTENTS[v["contents"]]["mass"] + (0.1, 200.0)}

    def accept(self, v: dict[str, Any]) -> bool:
        return 0.0 < v["extension"] <= 0.96 * (v["cab_dims"][0] - 0.04)

    def options(self, group: Any, rng: random.Random) -> None:
        v = group.visible
        part = fam._round(max(0.05, v["extension"] * rng.uniform(0.45, 0.7)), 0.01)
        group.variants.update({"opt_part": {"extension": part, "duration": round(max(0.2, v["duration"] * rng.uniform(0.3, 0.5)), 1)},
                               "opt_slow": {"duration": round(min(6.0, v["duration"] * rng.uniform(2.0, 3.0)), 1)}})
        group.plan["options"] = {"as_planned": "planned", "part_way_quick": "opt_part", "slower_pull": "opt_slow"}

    def factor_choices(self, group: Any, factor: str, current: Any) -> list[Any]:
        if factor == "drawer_contents":
            return sorted(CABINET_CONTENTS)
        if factor == "ballast":
            return sorted(CABINET_BALLAST)
        if factor == "floor":
            return sorted(CABINET_FLOORS)
        if factor == "pull_time":
            return [d for d in (0.3, 0.5, 2.0, 4.0) if abs(d - current) >= 0.4]
        if factor == "drawer_level":
            return [x for x in ("top", "middle", "bottom") if x != current]
        return []

    def _plan(self, v: dict[str, Any], person: str) -> str:
        subject = {"third": "The robot", "second": "You", "first": "I"}[person]
        return (f"{subject} will pull the {v['level']} drawer straight out by {cm(v['extension'])}, taking {v['duration']:.1f} s from start to stop; "
                "the drawer runs on the cabinet's own rails, which take its push as it speeds up and slows down.")

    def state_text(self, v: dict[str, Any], style: str, rng: random.Random) -> str:
        depth, width, height = v["cab_dims"]
        contents = CABINET_CONTENTS[v["contents"]]["text"]
        ballast = CABINET_BALLAST[v["ballast"]]["text"]
        floor = CABINET_FLOORS[v["floor"]]["text"]
        cab = (f"{v['cabinet_name']}, {num(height)} m tall, {num(depth)} m deep, {num(width)} m wide, {v['cab_mass']:.0f} kg empty, {v['drawers']} "
               "drawers, standing free (not fixed to the wall)")
        if style == "log":
            return (f"Task log, {v['domain']}. Unit: {a_or_an(v['robot'])}. Cabinet: {cab}, on {floor}; {ballast}. The {v['level']} drawer is "
                    f"{contents}. {self._plan(v, 'third')}")
        if style == "brief":
            return (f"You are {a_or_an(v['robot'])} in {a_or_an(v['domain'])}, facing a {cab}, on {floor}. {ballast.capitalize()}. Its {v['level']} "
                    f"drawer is {contents}. {self._plan(v, 'second')}")
        if style == "sheet":
            return "\n".join((f"robot: {v['robot']} ({v['domain']})", f"cabinet: {cab}", f"floor: {floor}", f"rest of the cabinet: {ballast}",
                              f"{v['level']} drawer: {contents}", f"plan: {self._plan(v, 'third')}"))
        return (f"I'm {a_or_an(v['robot'])} in the {v['domain']}, in front of a {cab}, on {floor}. {ballast.capitalize()}, and the {v['level']} "
                f"drawer is {contents}. {self._plan(v, 'first')}")

    def _options(self, v: dict[str, Any], plan: dict[str, Any]) -> dict[str, str]:
        opts = plan["option_values"]
        return {"as_planned": f"pull it out {cm(v['extension'])} over {v['duration']:.1f} s, as planned",
                "part_way_quick": f"pull it out only {cm(opts['opt_part']['extension'])}, but in {opts['opt_part']['duration']:.1f} s",
                "slower_pull": f"pull it out the full {cm(v['extension'])}, taking {opts['opt_slow']['duration']:.1f} s"}

    def questions(self, v: dict[str, Any], mix: list[str], plan: dict[str, Any], person: str, rng: random.Random, cf_option: str,
                  action_keys: list[str]) -> dict[str, Any]:
        cab = f"the {v['cabinet_name']}"
        now, other = cm(v["extension"]), cm(plan["cf_value"])
        out: dict[str, Any] = {}
        out["cabinet_tips_or_rocks"] = {"type": "noul", "instructions": rng.choice((
            f"As the drawer comes out and stops, does {cab} rock forward off its back feet (1.5° or more), or fall over?",
            f"Will pulling the drawer make {cab} tilt forward by 1.5° or more, or topple?"))}
        out["extension_change_effect"] = {"type": "choice", "instructions": rng.choice((
            f"Suppose the drawer came out {other} instead of {now}. Would a rock or fall of {cab} be more likely, less likely or about as likely?",
            f"Against pulling the drawer out {now}, what would pulling it {other} do to the chance that {cab} rocks or falls?")),
            "criteria": dict(TABLE_DIRECTION)}
        if "outcome" in mix:
            out["cabinet_after_pull"] = {"type": "choice", "instructions": f"What does {cab} do as the drawer comes out?",
                                         "criteria": {"steady": "it stays flat on its feet", "rocks_forward": "it rocks forward, then settles back",
                                                      "falls_forward": "it falls forward"}}
        options = self._options(v, plan)
        if "action" in mix:
            out["safest_pull_plan"] = {"type": "choice", "instructions": f"Which way of opening the drawer is safest for {cab}?",
                                       "criteria": {k: options[k] for k in action_keys}}
        if "compare" in mix:
            out["riskier_pull_plan"] = {"type": "choice", "instructions": f"Which is more likely to rock or topple {cab}: sticking to the plan, or to "
                                        f"{options[cf_option]}?",
                                        "criteria": {"as_planned": "sticking to the plan", cf_option: "the other way",
                                                     "about_same": "roughly equal (within 0.1)"}}
        if "score_cf" in mix:
            out["tip_risk_other_extension"] = {"type": "score", "instructions": f"Rate the chance that {cab} rocks or falls if the drawer came out {other} "
                                               f"rather than {now}.", "criteria": ["0 = it will not budge", "1 = it is certain to rock or fall"]}
        if "severe" in mix:
            out["cabinet_falls"] = {"type": "noul", "instructions": f"Does {cab} actually fall over?"}
        return out

    def sanity_cases(self) -> list[tuple[dict[str, Any], float]]:
        base = {"cab_dims": [0.62, 0.47, 1.32], "cab_mass": 30.0, "drawers": 4, "level": "top", "duration": 6.0}
        return [(base, 33.0), ({**base, "level": "bottom", "cab_mass": 24.0}, 25.0)]

    def catalogs(self) -> dict[str, Any]:
        return {"contents": CABINET_CONTENTS, "ballast": CABINET_BALLAST, "floors": CABINET_FLOORS, "cabinets": CABINET_KINDS, "robots": CABINET_ROBOTS,
                "domains": CABINET_DOMAINS, "drawer_shell_kg": DRAWER_SHELL}

    def catalog_names(self) -> set[str]:
        out = {item["name"].lower() for item in CABINET_KINDS} | {r.lower() for r in CABINET_ROBOTS}
        out |= {x["text"].lower() for x in CABINET_FLOORS.values()}
        return out
