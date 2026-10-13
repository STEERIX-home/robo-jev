"""Law A — tip or slide under acceleration or tilt: three new scenes on the platform engine (:mod:`robo_jev.mjgen.scenes.rigs`).

* ``forks`` — a boxed load resting across a forklift's two tines while the truck brakes or takes a turn; the mast may be tilted back
  (fork tips up), which resists a forward slide; in a turn the load tips about the outer tine's edge when it is wider than the forks.
* ``bin`` — an item standing in a sidewalk delivery robot's cargo bin while the robot brakes; the bin's front wall, a stated gap
  ahead of the item, stops a slide and catches a tipping item unless the gap is wide.
* ``tray`` — a drink or pot standing on a café robot's hand-held tray while the robot starts off, stops or turns (held-out scene).

Hidden values (never written): friction (support material × the item's base material, × 0.55–0.75 when wet), the centre-of-mass height
ratio and its fore-aft and lateral offsets (contents or fill words). The decisive number is the deceleration (or acceleration) of a
stop or start, or the speed of a turn.
"""

from __future__ import annotations

import json
import math
import random
from typing import Any

from robo_jev.mjgen import families as fam
from robo_jev.mjgen.scenes import rigs
from robo_jev.mjgen.scenes.base import Scene, a_or_an, cm, num

__all__ = ["BinScene", "ForksScene", "TrayScene"]

G = rigs.GRAVITY
WET = fam.WET_FACTOR


def _com_specs(word: dict[str, Any]) -> dict[str, Any]:
    return {"kh": ("u",) + tuple(word["kh"]) + (0.08, 0.92), "cx": ("u",) + tuple(word.get("cx", (-0.03, 0.03))) + (-0.40, 0.40),
            "cy": ("u",) + tuple(word.get("cy", (-0.03, 0.03))) + (-0.40, 0.40)}


class PlatformScene(Scene):
    """Shared physics of the law-A scenes: a load on a support that is pushed by an acceleration (and tilted by the mast)."""

    law = "tip_slide"
    outcomes = ("stay", "slide", "tip")
    event = ("slide", "tip")
    severe = "tip"
    slide_m = 0.03
    physics_fields = ("dims", "mass", "move", "speed", "accel", "radius", "ramp", "tilt", "spread", "tine_w", "gap", "wall_h", "tray", "laid")

    # -- geometry -------------------------------------------------------------------------------------------------------------------

    def support(self, visible: dict[str, Any]) -> list[tuple[str, tuple[float, float, float], tuple[float, float, float], bool]]:
        raise NotImplementedError

    def load_dims(self, visible: dict[str, Any]) -> list[float]:
        return list(visible["dims"])

    def lateral_edge(self, visible: dict[str, Any]) -> float:
        """Half-width of the support under the load across the travel direction (its tipping edge in a turn)."""
        return self.load_dims(visible)[1] / 2

    def front_room(self, visible: dict[str, Any]) -> float:
        """How far the load can slide forward before something stops it (inf: nothing)."""
        return math.inf

    # -- static model ---------------------------------------------------------------------------------------------------------------

    def push(self, visible: dict[str, Any]) -> tuple[float, int]:
        """(magnitude of the inertial push per kg, direction code: +1 = toward the front, −1 = toward the rear, 2 = sideways)."""
        if visible["move"] == "turn":
            return visible["speed"] ** 2 / visible["radius"], 2
        return visible["accel"], (1 if visible["move"] in ("brake", "stop") else -1)

    def static_outcome(self, visible: dict[str, Any], draw: dict[str, float]) -> str:
        length, width, height = self.load_dims(visible)
        a, way = self.push(visible)
        beta = math.radians(visible.get("tilt", 0.0))
        com_x, com_y, h_c = draw["cx"] * length, draw["cy"] * width, draw["kh"] * height
        if way == 2:
            tangential = math.hypot(a, G * math.sin(beta))
            normal = G * math.cos(beta)
            edge = self.lateral_edge(visible) - abs(com_y)
        elif way == 1:
            tangential = a * math.cos(beta) - G * math.sin(beta)
            normal = G * math.cos(beta) + a * math.sin(beta)
            edge = length / 2 - com_x
        else:
            tangential = a * math.cos(beta) + G * math.sin(beta)
            normal = G * math.cos(beta) - a * math.sin(beta)
            edge = length / 2 + com_x
        ratio = tangential / max(normal, 1e-6)
        tip_at = max(edge, 1e-6) / max(h_c, 1e-6)
        if ratio <= min(tip_at, draw["mu"]):
            return "stay"
        if way == 1 and self.front_room(visible) < self.slide_m and draw["mu"] <= tip_at:
            return "stay"  # a slide that the wall stops before it counts
        return "tip" if tip_at < draw["mu"] else "slide"

    # -- rollout --------------------------------------------------------------------------------------------------------------------

    def profile(self, visible: dict[str, Any]) -> tuple[Any, int]:
        ramp = visible["ramp"]
        if visible["move"] == "turn":
            a = visible["speed"] ** 2 / visible["radius"]
            duration = (math.pi / 2) * visible["radius"] / visible["speed"]
            return rigs.linear_profile(a * duration, a, jerk_time=ramp, pre=0.3, post=1.0, sign=-1.0), 1
        sign = -1.0 if visible["move"] in ("brake", "stop") else 1.0
        return rigs.linear_profile(visible["speed"], visible["accel"], jerk_time=ramp, pre=0.3, post=1.0, sign=sign), 0

    def duration(self, visible: dict[str, Any]) -> float:
        if visible["move"] == "turn":
            return (math.pi / 2) * visible["radius"] / visible["speed"]
        return visible["speed"] / visible["accel"]

    def accept(self, visible: dict[str, Any]) -> bool:
        """The speed change lasts 0.05–6 s, and the stated rate is held for most of it: the ramp-up time is at most a third of the change
        (a stop shorter than its own ramp never reaches the stated rate except for an instant, and nothing moves)."""
        duration = self.duration(visible)
        return 0.05 <= duration <= 6.0 and visible["ramp"] <= duration / 3

    def rig_for(self, visible: dict[str, Any], rigs_cache: dict[str, Any]) -> Any:
        key = json.dumps([self.name, self.load_dims(visible), visible["mass"], visible.get("tilt", 0.0), self.support(visible)])
        if key not in rigs_cache:
            rigs_cache[key] = rigs.platform_rig(self.name, self.support(visible), tilt_deg=visible.get("tilt", 0.0), load_dims=self.load_dims(visible),
                                                load_mass=visible["mass"])
        return rigs_cache[key]

    def roll(self, visible: dict[str, Any], draw: dict[str, float], rigs_cache: dict[str, Any]) -> dict[str, Any]:
        rig = self.rig_for(visible, rigs_cache)
        path, axis = self.profile(visible)
        length, width, height = self.load_dims(visible)
        com = (draw["cx"] * length, draw["cy"] * width, (draw["kh"] - 0.5) * height)
        out = rigs.run_platform(rig, path, axis, mu=draw["mu"], com=com)
        out["outcome"] = self.classify(out)
        return out

    def classify(self, result: dict[str, Any]) -> str:
        if result["fell"] or result["early"] == "tip" or result["tilt_deg"] > 45.0:
            return "tip"
        if result["early"] == "slide" or result["slide_m"] >= self.slide_m:
            return "slide"
        return "stay"

    def xml_example(self, visible: dict[str, Any]) -> str:
        return rigs.platform_xml(self.name, self.support(visible), tilt_deg=visible.get("tilt", 0.0), load_dims=self.load_dims(visible),
                                 load_mass=visible["mass"])


# ======================================================================================================================================
# forks — a boxed load on a forklift's tines
# ======================================================================================================================================

FORK_TINES = {
    "steel": {"text": "bare steel tines", "mu": (0.30, 0.42)},
    "painted": {"text": "freshly painted tines", "mu": (0.24, 0.34)},
    "sleeved": {"text": "tines in rubber fork sleeves", "mu": (0.70, 0.95)},
    "worn": {"text": "polished, worn-smooth tines", "mu": (0.17, 0.25)},
    "capped": {"text": "tines with plywood caps", "mu": (0.45, 0.60)},
}
FORK_BASES = {
    "skid": {"text": "a wooden skid", "factor": (0.95, 1.05)},
    "card": {"text": "a corrugated cardboard bottom", "factor": (0.85, 1.00)},
    "slip": {"text": "a plastic slip sheet", "factor": (0.62, 0.78)},
    "frame": {"text": "a steel-framed base", "factor": (0.58, 0.72)},
    "feet": {"text": "rubber-footed runners", "factor": (1.20, 1.40)},
}
FORK_LOADS = (
    {"key": "heater", "name": "boxed water heater", "dims": ((0.50, 0.60), (0.55, 0.65), (1.20, 1.50)), "mass": (40.0, 70.0),
     "contents": ("level", "low", "unsure"), "bases": ("card", "skid")},
    {"key": "generator", "name": "crated generator", "dims": ((0.70, 0.90), (0.55, 0.70), (0.55, 0.75)), "mass": (60.0, 120.0),
     "contents": ("low", "level", "fore", "aft", "unsure"), "bases": ("skid", "frame")},
    {"key": "reams", "name": "shrink-wrapped bundle of copier paper", "dims": ((0.60, 0.80), (0.80, 1.00), (0.50, 0.90)), "mass": (150.0, 350.0),
     "contents": ("level",), "bases": ("slip", "card", "skid")},
    {"key": "tiles", "name": "crate of porcelain tiles", "dims": ((0.60, 0.80), (0.60, 0.80), (0.35, 0.55)), "mass": (200.0, 400.0),
     "contents": ("level", "low"), "bases": ("skid",)},
    {"key": "freezer", "name": "boxed upright freezer", "dims": ((0.55, 0.70), (0.60, 0.75), (1.40, 1.80)), "mass": (50.0, 80.0),
     "contents": ("level", "high", "unsure"), "bases": ("card", "feet")},
    {"key": "jugs", "name": "banded carton of motor-oil jugs", "dims": ((0.70, 0.90), (0.90, 1.10), (0.60, 0.80)), "mass": (180.0, 300.0),
     "contents": ("level", "fore", "aft"), "bases": ("card", "slip")},
    {"key": "rack", "name": "crated server cabinet", "dims": ((0.80, 1.00), (0.60, 0.70), (1.20, 1.60)), "mass": (90.0, 160.0),
     "contents": ("low", "high", "unsure"), "bases": ("frame", "feet", "skid")},
    {"key": "panes", "name": "timber crate of window panes", "dims": ((0.30, 0.45), (1.00, 1.30), (0.90, 1.20)), "mass": (80.0, 160.0),
     "contents": ("level",), "bases": ("skid",)},
)
FORK_CONTENTS = {
    "level": {"kh": (0.45, 0.55), "text": ("its weight is spread evenly through it", "it is evenly packed")},
    "low": {"kh": (0.26, 0.38), "cx": (-0.04, 0.04), "text": ("its heaviest parts sit near its base", "it is bottom-heavy")},
    "high": {"kh": (0.58, 0.72), "cx": (-0.04, 0.04), "text": ("its heavy compressor sits near the top", "it is top-heavy")},
    "unsure": {"kh": (0.30, 0.70), "cx": (-0.12, 0.12), "cy": (-0.08, 0.08), "text": ("nobody knows how it is packed inside",
                                                                                     "its packing inside is unknown")},
    "fore": {"kh": (0.42, 0.58), "cx": (0.06, 0.16), "text": ("its heavy end faces the fork tips", "it is heavier toward the fork tips")},
    "aft": {"kh": (0.42, 0.58), "cx": (-0.16, -0.06), "text": ("its heavy end faces the mast", "it is heavier toward the mast")},
}
FORK_ROBOTS = ("counterbalance forklift", "autonomous reach truck", "electric pallet stacker")
FORK_DOMAINS = ("distribution centre", "cold store", "lumber yard", "brewery warehouse", "port shed", "garden-centre yard", "appliance depot",
                "print works")
TINE_W = (0.10, 0.15)


class ForksScene(PlatformScene):
    name = "forks"
    question_names = {"key": "load_shifts_on_forks", "direction": "push_change_effect", "outcome": "load_after_manoeuvre",
                      "action": "safest_fork_plan", "compare": "riskier_fork_plan", "score_cf": "shift_risk_other_rate", "severe": "load_tips_off"}
    outcome_keys = {"stays_put": "stay", "slides": "slide", "tips_over": "tip"}
    dominant_option = "tilt_back"
    factors = {
        "tine_surface": {"field": "surface", "fact": "word", "how": "word"},
        "load_base": {"field": "base", "fact": "word", "how": "word"},
        "wet": {"field": "wet", "fact": "word", "how": "word"},
        "mass_distribution": {"field": "contents", "fact": "word", "how": "word"},
        "footprint": {"field": "dims.0", "fact": "number", "how": "aim", "span": lambda c, v: (max(0.2, c * 0.5), c * 1.8, 0.01)},
        "mast_tilt": {"field": "tilt", "fact": "number", "how": "aim", "span": lambda c, v: (0.0, 8.0, 0.5)},
        "fork_spread": {"field": "spread", "fact": "number", "how": "aim",
                        "span": lambda c, v: (0.30, max(0.31, min(0.80, v["dims"][1] - v["tine_w"] - 0.02)), 0.01)},
        "ramp_time": {"field": "ramp", "fact": "number", "how": "choice"},
    }
    pair_fields = (("decisive", 0.65), ("dims.2", 0.35))
    pair_spans = {"dims.2": lambda c, v: (max(0.2, c * 0.4), c * 2.0, 0.01)}
    domains = FORK_DOMAINS

    def support(self, v: dict[str, Any]) -> list[tuple[str, tuple[float, float, float], tuple[float, float, float], bool]]:
        """Two tines from the heel forward, and the load backrest (1.2 m high) at the heel, 5 mm behind the load: a load on forks sits
        back against it, so the mast tilt never slides or tips it backward (without it, a load on worn tines slid back under a steep
        mast tilt and the tilt sweep was not monotone)."""
        length = self.load_dims(v)[0]
        tine_len = max(1.07, length + 0.20)
        heel = -(length / 2 + 0.005)
        cx = heel + tine_len / 2
        half = (tine_len / 2, v["tine_w"] / 2, 0.025)
        width = v["spread"] / 2 + v["tine_w"] / 2 + 0.10
        return [("tine_l", (cx, v["spread"] / 2, -0.025), half, True), ("tine_r", (cx, -v["spread"] / 2, -0.025), half, True),
                ("backrest", (heel - 0.02, 0.0, 0.60), (0.02, width, 0.60), False)]

    def lateral_edge(self, v: dict[str, Any]) -> float:
        return min(self.load_dims(v)[1] / 2, v["spread"] / 2 + v["tine_w"] / 2)

    def sample(self, rng: random.Random) -> tuple[dict[str, Any], dict[str, Any]]:
        load = rng.choice(FORK_LOADS)
        dims = [round(rng.uniform(*span), 2) for span in load["dims"]]
        tine_w = round(rng.uniform(*TINE_W), 2)
        spread = round(rng.uniform(0.30, max(0.31, min(0.80, dims[1] - tine_w - 0.02))), 2)
        move = rng.choice(("brake", "brake", "turn"))
        v = {"robot": rng.choice(FORK_ROBOTS), "domain": rng.choice(FORK_DOMAINS), "load": load["key"], "load_name": load["name"], "dims": dims,
             "mass": round(rng.uniform(*load["mass"]), 0), "contents": rng.choice(load["contents"]), "surface": rng.choice(sorted(FORK_TINES)),
             "base": rng.choice(load["bases"]), "wet": rng.random() < 0.15, "tilt": rng.choice((0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0)),
             "spread": spread, "tine_w": tine_w, "move": move, "ramp": round(rng.uniform(0.04, 0.25), 2), "speed": 0.0, "accel": 0.0, "radius": 0.0}
        if move == "brake":
            v["speed"] = round(rng.uniform(1.5, 3.5), 1)
            return v, {"field": "accel", "lo": 0.3, "hi": 9.0, "step": 0.1, "increasing": True}
        v["radius"] = round(rng.uniform(2.0, 6.0), 1)
        v["ramp"] = round(rng.uniform(0.10, 0.40), 2)
        return v, {"field": "speed", "lo": 0.5, "hi": 7.0, "step": 0.1, "increasing": True}

    def hidden_for(self, v: dict[str, Any]) -> dict[str, Any]:
        mu = FORK_TINES[v["surface"]]["mu"]
        factor = FORK_BASES[v["base"]]["factor"]
        lo, hi = mu[0] * factor[0], mu[1] * factor[1]
        if v.get("wet"):
            lo, hi = lo * WET[0], hi * WET[1]
        return {"mu": ("mu", round(lo, 4), round(hi, 4)), **_com_specs(FORK_CONTENTS[v["contents"]])}

    def options(self, group: Any, rng: random.Random) -> None:
        v = group.visible
        back = min(8.0, v["tilt"] + rng.choice((2.0, 3.0, 4.0)))
        if v["move"] == "brake":
            gentler = {"accel": fam._round(max(0.2, v["accel"] * rng.uniform(0.55, 0.8)), 0.1), "tilt": 0.0}
        else:
            gentler = {"speed": fam._round(max(0.3, v["speed"] * rng.uniform(0.7, 0.88)), 0.1), "tilt": 0.0}
        group.variants.update({"opt_back": {"tilt": back}, "opt_gentle": gentler})
        group.plan["options"] = {"as_planned": "planned", "tilt_back": "opt_back", "gentler_level": "opt_gentle"}

    def factor_choices(self, group: Any, factor: str, current: Any) -> list[Any]:
        v = group.visible
        load = next(item for item in FORK_LOADS if item["key"] == v["load"])
        if factor == "tine_surface":
            return sorted(FORK_TINES)
        if factor == "load_base":
            return sorted(FORK_BASES)
        if factor == "wet":
            return [not current]
        if factor == "mass_distribution":
            return [c for c in load["contents"] + ("unsure",) if c != current]
        if factor == "ramp_time":
            return [r for r in (0.02, 0.05, 0.3, 0.6) if abs(r - current) > 0.04]
        return []

    # -- text -----------------------------------------------------------------------------------------------------------------------

    def _manoeuvre(self, v: dict[str, Any], person: str) -> str:
        subject = {"third": "The truck", "second": "You", "first": "I"}[person]
        if v["move"] == "brake":
            return (f"{subject} {'are' if person == 'second' else ('am' if person == 'first' else 'is')} travelling at {v['speed']:.1f} m/s and will brake "
                    f"to a stop at {v['accel']:.1f} m/s²; full braking is reached {v['ramp']:.2f} s after the brakes come on.")
        verb = {"third": "will swing", "second": "will swing", "first": "will swing"}[person]
        return (f"{subject} {verb} through a quarter circle of {v['radius']:.1f} m radius, holding {v['speed']:.1f} m/s throughout; the sideways "
                f"pull builds up over {v['ramp']:.2f} s.")

    def state_text(self, v: dict[str, Any], style: str, rng: random.Random) -> str:
        length, width, height = v["dims"]
        contents = rng.choice(FORK_CONTENTS[v["contents"]]["text"])
        tines = FORK_TINES[v["surface"]]["text"] + (", wet from rain" if v.get("wet") else "")
        base = FORK_BASES[v["base"]]["text"]
        size = f"{num(length)} m deep on the forks, {num(width)} m across its face, {num(height)} m from base to top"
        tilt = "level (no tilt)" if v["tilt"] == 0 else f"tilted back {v['tilt']:.1f}°, tips raised"
        spread = f"{num(v['spread'])} m apart centre to centre, each {cm(v['tine_w'])} wide"
        if style == "log":
            return (f"Yard log, {v['domain']}. Unit: {a_or_an(v['robot'])}. On the forks: one {v['load_name']}, {size}, {v['mass']:.0f} kg, "
                    f"resting on {base} with its back against the load backrest, not tied down. Load note: {contents}. Forks: {tines}, {spread}; mast {tilt}. {self._manoeuvre(v, 'third')}")
        if style == "brief":
            return (f"You drive {a_or_an(v['robot'])} in {a_or_an(v['domain'])}. Your forks carry {a_or_an(v['load_name'])} ({size}; {v['mass']:.0f} kg) "
                    f"on {base}, back against the load backrest and not tied down; {contents}. Your forks are {tines}, {spread}, and your mast is {tilt}. {self._manoeuvre(v, 'second')}")
        if style == "sheet":
            return "\n".join((
                f"truck: {v['robot']} ({v['domain']})", f"load on forks: {v['load_name']}, against the load backrest, not tied down", f"load size: {size}",
                f"load weight: {v['mass']:.0f} kg", f"load base: {base}", f"inside the load: {contents}", f"forks: {tines}; {spread}",
                f"mast: {tilt}", f"manoeuvre: {self._manoeuvre(v, 'third')}",
            ))
        return (f"I am {a_or_an(v['robot'])} at work in the {v['domain']}. I have {a_or_an(v['load_name'])} on my forks, {size}, weighing "
                f"{v['mass']:.0f} kg; it stands on {base}, back against my load backrest, loose. As for the inside: {contents}. My forks are {tines}, {spread}, and my mast is "
                f"{tilt}. {self._manoeuvre(v, 'first')}")

    def _option_text(self, v: dict[str, Any], plan: dict[str, Any]) -> dict[str, str]:
        opts = plan["option_values"]
        manoeuvre = "stop" if v["move"] == "brake" else "turn"
        gentle = opts["opt_gentle"]
        if v["move"] == "brake":
            soft = f"keep the mast level and brake at only {gentle['accel']:.1f} m/s²"
        else:
            soft = f"keep the mast level and take the turn at only {gentle['speed']:.1f} m/s"
        return {"as_planned": f"{manoeuvre} as planned with the mast at {v['tilt']:.1f}°",
                "tilt_back": f"tilt the mast back to {opts['opt_back']['tilt']:.1f}° first, then {manoeuvre} as planned",
                "gentler_level": soft}

    def questions(self, v: dict[str, Any], mix: list[str], plan: dict[str, Any], person: str, rng: random.Random, cf_option: str,
                  action_keys: list[str]) -> dict[str, Any]:
        load = v["load_name"]
        manoeuvre = "stop" if v["move"] == "brake" else "turn"
        truck = {"third": "the truck", "second": "you", "first": "I"}[person]
        if v["move"] == "brake":
            now, other = f"{v['accel']:.1f} m/s²", f"{plan['cf_value']:.1f} m/s²"
            change = rng.choice((f"Suppose {truck} braked at {other} rather than {now}. Would the {load} be more likely, less likely or about as likely "
                                 "to slide or tip?",
                                 f"Against the planned {now} of braking, what would {other} of braking do to the chance that the {load} slides or tips?"))
            rate_text = f"if the braking were {other}, not the planned {now}"
        else:
            now, other = f"{v['speed']:.1f} m/s", f"{plan['cf_value']:.1f} m/s"
            change = rng.choice((f"Suppose the turning speed were {other} (the plan says {now}). Would the {load} be more likely, less likely or "
                                 "about as likely to slide or tip?",
                                 f"Against the planned turning speed of {now}, what would {other} do to the chance that the {load} slides or tips?"))
            rate_text = f"if the turning speed were {other}, not the planned {now}"
        out: dict[str, Any] = {}
        out["load_shifts_on_forks"] = {"type": "noul", "instructions": rng.choice((
            f"During this {manoeuvre}, does the {load} slide 3 cm or more along the tines, or tip over?",
            f"Will the {load} shift at least 3 cm on the forks, or topple, during the {manoeuvre}?"))}
        out["push_change_effect"] = {"type": "choice", "instructions": change, "criteria": dict(FORK_DIRECTION)}
        if "outcome" in mix:
            out["load_after_manoeuvre"] = {"type": "choice", "instructions": f"How does the {load} end up after the {manoeuvre}?",
                                           "criteria": {"stays_put": "still in place on the tines (under 3 cm of movement)",
                                                        "slides": "slid 3 cm or more along the tines, still upright", "tips_over": "tipped over"}}
        options = self._option_text(v, plan)
        if "action" in mix:
            out["safest_fork_plan"] = {"type": "choice", "instructions": f"Which of these plans keeps the {load} safest?",
                                       "criteria": {k: options[k] for k in action_keys}}
        if "compare" in mix:
            out["riskier_fork_plan"] = {"type": "choice", "instructions": f"Which is more likely to end with the {load} slid or tipped: sticking to the "
                                        f"plan, or to {options[cf_option]}?",
                                        "criteria": {"as_planned": "sticking to the plan", cf_option: "the alternative plan",
                                                     "about_same": "roughly equal (within 0.1)"}}
        if "score_cf" in mix:
            out["shift_risk_other_rate"] = {"type": "score", "instructions": f"Rate the chance that the {load} slides or tips {rate_text}.",
                                            "criteria": ["0 = it stays exactly where it is", "1 = it is certain to slide or tip"]}
        if "severe" in mix:
            out["load_tips_off"] = {"type": "noul", "instructions": f"Does the {load} actually tip over during the {manoeuvre}?"}
        return out

    def catalogs(self) -> dict[str, Any]:
        return {"tines": FORK_TINES, "bases": FORK_BASES, "loads": FORK_LOADS, "contents": FORK_CONTENTS, "robots": FORK_ROBOTS,
                "domains": FORK_DOMAINS, "tine_width": TINE_W}

    def catalog_names(self) -> set[str]:
        out = {item["name"].lower() for item in FORK_LOADS} | {r.lower() for r in FORK_ROBOTS}
        out |= {x["text"].lower() for x in list(FORK_TINES.values()) + list(FORK_BASES.values())}
        return out


#: wording of the direction options (training scenes)
FORK_DIRECTION = {"more_likely": "a higher chance than in the plan", "less_likely": "a lower chance than in the plan",
                  "about_same": "roughly unchanged (it moves by less than 0.1)"}


# ======================================================================================================================================
# bin — an item in a sidewalk delivery robot's cargo bin
# ======================================================================================================================================

BIN_LINERS = {
    "ribbed": {"text": "a grooved rubber liner", "mu": (0.75, 1.00)},
    "plastic": {"text": "its bare moulded-plastic floor", "mu": (0.25, 0.38)},
    "felt": {"text": "a felt liner", "mu": (0.45, 0.60)},
    "alu": {"text": "a brushed aluminium floor", "mu": (0.22, 0.32)},
    "foam": {"text": "a closed-cell foam pad", "mu": (0.60, 0.85)},
}
BIN_SURFACES = {
    "brown": {"text": "plain brown card", "factor": (0.95, 1.05)},
    "gloss": {"text": "printed gloss card", "factor": (0.75, 0.88)},
    "film": {"text": "shrink-wrapped plastic film", "factor": (0.62, 0.78)},
    "kraft": {"text": "a kraft-paper wrap", "factor": (0.88, 1.00)},
}
BIN_ITEMS = (
    {"key": "cake", "name": "boxed layer cake", "dims": ((0.24, 0.30), (0.24, 0.30), (0.14, 0.20)), "mass": (1.0, 2.5),
     "contents": ("level", "low"), "surfaces": ("brown", "gloss"), "lay": False},
    {"key": "oil", "name": "gift box of olive oil", "dims": ((0.08, 0.10), (0.08, 0.10), (0.28, 0.34)), "mass": (1.0, 1.6),
     "contents": ("low", "level"), "surfaces": ("gloss", "kraft"), "lay": True},
    {"key": "flowers", "name": "boxed flower arrangement", "dims": ((0.18, 0.24), (0.18, 0.24), (0.30, 0.42)), "mass": (0.8, 2.5),
     "contents": ("low", "unsure"), "surfaces": ("brown", "kraft"), "lay": False},
    {"key": "tubs", "name": "carton of soup tubs", "dims": ((0.20, 0.28), (0.16, 0.22), (0.14, 0.20)), "mass": (1.5, 3.0),
     "contents": ("level", "unsure"), "surfaces": ("brown", "film"), "lay": True},
    {"key": "wine", "name": "two-bottle wine carton", "dims": ((0.17, 0.20), (0.09, 0.11), (0.32, 0.36)), "mass": (2.6, 3.4),
     "contents": ("level", "low"), "surfaces": ("brown", "gloss"), "lay": True},
    {"key": "pharmacy", "name": "sealed pharmacy parcel", "dims": ((0.15, 0.25), (0.10, 0.18), (0.10, 0.25)), "mass": (0.3, 1.5),
     "contents": ("unsure", "level"), "surfaces": ("kraft", "film"), "lay": True},
    {"key": "lamp", "name": "boxed desk lamp", "dims": ((0.20, 0.26), (0.20, 0.26), (0.40, 0.50)), "mass": (1.5, 3.0),
     "contents": ("low", "level", "unsure"), "surfaces": ("gloss", "brown"), "lay": True},
)
BIN_CONTENTS = {
    "level": {"kh": (0.45, 0.55), "text": ("the order inside fills it evenly", "its contents are evenly spread")},
    "low": {"kh": (0.25, 0.38), "cx": (-0.04, 0.04), "text": ("the heavy part rests at the bottom", "it is weighted at its base")},
    "unsure": {"kh": (0.30, 0.68), "cx": (-0.12, 0.12), "cy": (-0.08, 0.08), "text": ("the shop did not say how it is packed",
                                                                                     "how it is packed inside is not known")},
}
BIN_ROBOTS = ("six-wheeled sidewalk courier", "kerbside parcel rover", "campus food-delivery robot")
BIN_DOMAINS = ("residential street", "university campus", "shopping district", "business park", "hospital grounds", "seaside promenade",
               "suburban cul-de-sac", "city square")


class BinScene(PlatformScene):
    name = "bin"
    outcomes = ("stays", "shifts", "falls")
    event = ("shifts", "falls")
    severe = "falls"
    slide_m = 0.02
    question_names = {"key": "parcel_moves_in_bin", "direction": "stop_change_effect", "outcome": "parcel_after_stop", "action": "safest_bin_plan",
                      "compare": "riskier_bin_plan", "score_cf": "move_risk_other_stop", "severe": "parcel_falls_over"}
    outcome_keys = {"stays_put": "stays", "shifts": "shifts", "falls_over": "falls"}
    dominant_option = "against_wall"
    factors = {
        "liner": {"field": "surface", "fact": "word", "how": "word"},
        "item_surface": {"field": "base", "fact": "word", "how": "word"},
        "wet": {"field": "wet", "fact": "word", "how": "word"},
        "mass_distribution": {"field": "contents", "fact": "word", "how": "word"},
        "footprint": {"field": "dims.0", "fact": "number", "how": "aim", "span": lambda c, v: (max(0.06, c * 0.5), c * 1.8, 0.01)},
        "ramp_time": {"field": "ramp", "fact": "number", "how": "choice"},
        "wall_gap": {"field": "gap", "fact": "number", "how": "choice"},
    }
    pair_fields = (("decisive", 0.65), ("dims.2", 0.35))
    pair_spans = {"dims.2": lambda c, v: (max(0.06, c * 0.4), min(0.6, c * 2.0), 0.01)}
    domains = BIN_DOMAINS

    def load_dims(self, v: dict[str, Any]) -> list[float]:
        length, width, height = v["dims"]
        return [height, width, length] if v.get("laid") else [length, width, height]

    def support(self, v: dict[str, Any]) -> list[tuple[str, tuple[float, float, float], tuple[float, float, float], bool]]:
        length, width, height = self.load_dims(v)
        gap, wall_h = v["gap"], v["wall_h"]
        front = length / 2 + gap
        rear = -(length / 2 + 0.15)
        side = width / 2 + 0.12
        mid = (front + rear) / 2
        half_x = (front - rear) / 2
        t = 0.01
        return [
            ("bin_floor", (mid, 0.0, -0.01), (half_x + t, side + t, 0.01), True),
            ("wall_front", (front + t, 0.0, wall_h / 2), (t, side + t, wall_h / 2), False),
            ("wall_rear", (rear - t, 0.0, wall_h / 2), (t, side + t, wall_h / 2), False),
            ("wall_left", (mid, side + t, wall_h / 2), (half_x + t, t, wall_h / 2), False),
            ("wall_right", (mid, -side - t, wall_h / 2), (half_x + t, t, wall_h / 2), False),
        ]

    def front_room(self, v: dict[str, Any]) -> float:
        return v["gap"]

    def classify(self, result: dict[str, Any]) -> str:
        if result["fell"] or result["early"] == "tip" or result["tilt_deg"] > 45.0:
            return "falls"
        if result["early"] == "slide" or result["slide_m"] >= self.slide_m or result["tilt_deg"] >= 8.0:
            return "shifts"
        return "stays"

    def static_outcome(self, visible: dict[str, Any], draw: dict[str, float]) -> str:
        raw = super().static_outcome(visible, draw)
        if raw == "stay":
            return "stays"
        if raw == "slide":
            return "shifts"
        # a tipping item is caught by the front wall unless the gap lets it lean past 45°; a wall right in front holds it upright
        height = self.load_dims(visible)[2]
        if visible["gap"] > 0.7 * height or visible["wall_h"] < 0.5 * height:
            return "falls"
        return "shifts" if math.degrees(math.asin(min(1.0, visible["gap"] / height))) >= 8.0 else "stays"

    def sample(self, rng: random.Random) -> tuple[dict[str, Any], dict[str, Any]]:
        item = rng.choice(BIN_ITEMS)
        dims = [round(rng.uniform(*span), 2) for span in item["dims"]]
        v = {"robot": rng.choice(BIN_ROBOTS), "domain": rng.choice(BIN_DOMAINS), "item": item["key"], "item_name": item["name"], "dims": dims,
             "mass": round(rng.uniform(*item["mass"]), 1), "contents": rng.choice(item["contents"]), "surface": rng.choice(sorted(BIN_LINERS)),
             "base": rng.choice(item["surfaces"]), "wet": rng.random() < 0.15, "move": "brake", "speed": round(rng.uniform(1.2, 2.5), 1),
             "accel": 0.0, "ramp": round(rng.uniform(0.03, 0.15), 2), "gap": round(rng.choice((0.03, 0.05, 0.08, 0.12, 0.18, 0.25)), 2),
             "wall_h": round(rng.uniform(0.30, 0.45), 2), "laid": False, "tilt": 0.0, "radius": 0.0}
        return v, {"field": "accel", "lo": 0.3, "hi": 9.0, "step": 0.1, "increasing": True}

    def hidden_for(self, v: dict[str, Any]) -> dict[str, Any]:
        mu = BIN_LINERS[v["surface"]]["mu"]
        factor = BIN_SURFACES[v["base"]]["factor"]
        lo, hi = mu[0] * factor[0], mu[1] * factor[1]
        if v.get("wet"):
            lo, hi = lo * WET[0], hi * WET[1]
        return {"mu": ("mu", round(lo, 4), round(hi, 4)), **_com_specs(BIN_CONTENTS[v["contents"]])}

    def options(self, group: Any, rng: random.Random) -> None:
        v = group.visible
        item = next(i for i in BIN_ITEMS if i["key"] == v["item"])
        group.variants["opt_wall"] = {"gap": 0.005}
        if item["lay"]:
            group.variants["opt_other"] = {"laid": True}
            group.plan["other_option"] = "laid_down"
        else:
            group.variants["opt_other"] = {"accel": fam._round(max(0.2, v["accel"] * rng.uniform(0.55, 0.75)), 0.1), "gap": min(0.30, v["gap"] + 0.10)}
            group.plan["other_option"] = "gentler_further"
        group.plan["options"] = {"as_planned": "planned", "against_wall": "opt_wall", group.plan["other_option"]: "opt_other"}

    def factor_choices(self, group: Any, factor: str, current: Any) -> list[Any]:
        v = group.visible
        item = next(i for i in BIN_ITEMS if i["key"] == v["item"])
        if factor == "liner":
            return sorted(BIN_LINERS)
        if factor == "item_surface":
            return sorted(BIN_SURFACES)
        if factor == "wet":
            return [not current]
        if factor == "mass_distribution":
            return [c for c in item["contents"] + ("unsure",) if c != current]
        if factor == "ramp_time":
            return [r for r in (0.01, 0.03, 0.2, 0.4) if abs(r - current) > 0.04]
        if factor == "wall_gap":
            return [g for g in (0.01, 0.03, 0.08, 0.15, 0.25, 0.35) if abs(g - current) >= 0.04]
        return []

    # -- text -----------------------------------------------------------------------------------------------------------------------

    def _stop(self, v: dict[str, Any], person: str) -> str:
        subject = {"third": "The rover", "second": "You", "first": "I"}[person]
        be = {"third": "is", "second": "are", "first": "am"}[person]
        return (f"{subject} {be} cruising at {v['speed']:.1f} m/s and must stop for a crossing: it stops at {v['accel']:.1f} m/s², braking that "
                f"builds up over {v['ramp']:.2f} s.")

    def state_text(self, v: dict[str, Any], style: str, rng: random.Random) -> str:
        length, width, height = v["dims"]
        contents = rng.choice(BIN_CONTENTS[v["contents"]]["text"])
        liner = BIN_LINERS[v["surface"]]["text"] + (", damp after a shower" if v.get("wet") else "")
        outside = BIN_SURFACES[v["base"]]["text"]
        size = f"{num(height)} m tall on a {num(length)} × {num(width)} m base (the first figure is its depth toward the front wall)"
        place = f"its front face {cm(v['gap'])} behind the bin's front wall, which is {cm(v['wall_h'])} high"
        if style == "log":
            return (f"Route log, {v['domain']}. Unit: {a_or_an(v['robot'])}. In the cargo bin: one {v['item_name']} standing on its base, {size}, "
                    f"{v['mass']:.1f} kg, wrapped in {outside}; {contents}. It stands loose on {liner}, {place}; the other walls are well clear. "
                    f"{self._stop(v, 'third')}")
        if style == "brief":
            return (f"You are {a_or_an(v['robot'])} out on {a_or_an(v['domain'])}. Standing loose in your bin is {a_or_an(v['item_name'])} ({size}, "
                    f"{v['mass']:.1f} kg; outside: {outside}); {contents}. Your bin floor is {liner}. The item sits with {place}, and nothing else is "
                    f"near it. {self._stop(v, 'second')}")
        if style == "sheet":
            return "\n".join((
                f"robot: {v['robot']} ({v['domain']})", f"cargo: {v['item_name']}, standing, not secured", f"cargo size: {size}",
                f"cargo weight: {v['mass']:.1f} kg", f"cargo outside: {outside}", f"packing: {contents}", f"bin floor: {liner}",
                f"position: {place}; side and rear walls well clear", f"event: {self._stop(v, 'third')}",
            ))
        return (f"I'm {a_or_an(v['robot'])} on my route through the {v['domain']}. In my bin stands {a_or_an(v['item_name'])}, {size}, "
                f"{v['mass']:.1f} kg, its outside {outside}; {contents}. It is not held by anything, just standing on {liner}, with {place}. "
                f"{self._stop(v, 'first')}")

    def _option_text(self, v: dict[str, Any], plan: dict[str, Any]) -> dict[str, str]:
        opts = plan["option_values"]
        out = {"as_planned": f"leave it where it is and stop at {v['accel']:.1f} m/s²",
               "against_wall": "push it forward until it touches the front wall, then stop as planned"}
        if plan["other_option"] == "laid_down":
            out["laid_down"] = "lay it down with its top facing the front wall, then stop as planned"
        else:
            other = opts["opt_other"]
            out["gentler_further"] = f"stop more softly, at {other['accel']:.1f} m/s², but with the item set {cm(other['gap'])} from the front wall"
        return out

    def questions(self, v: dict[str, Any], mix: list[str], plan: dict[str, Any], person: str, rng: random.Random, cf_option: str,
                  action_keys: list[str]) -> dict[str, Any]:
        item = v["item_name"]
        now, other = f"{v['accel']:.1f} m/s²", f"{plan['cf_value']:.1f} m/s²"
        robot = {"third": "the rover", "second": "you", "first": "I"}[person]
        out: dict[str, Any] = {}
        out["parcel_moves_in_bin"] = {"type": "noul", "instructions": rng.choice((
            f"When {robot} stop{'s' if person == 'third' else ''}, does the {item} shift 2 cm or more, end up leaning, or fall over?",
            f"Will the stop move the {item} (2 cm or more, or leaning against something) or tip it onto its side?"))}
        out["stop_change_effect"] = {"type": "choice", "instructions": rng.choice((
            f"Imagine stopping with a deceleration of {other} in place of {now}. Is the {item} then more likely, less likely or about as likely to "
            "move or fall?",
            f"How would stopping at {other}, not {now}, change the chance that the {item} moves or falls over?")),
            "criteria": dict(FORK_DIRECTION)}
        if "outcome" in mix:
            out["parcel_after_stop"] = {"type": "choice", "instructions": f"Where is the {item} once the rover has come to rest?",
                                        "criteria": {"stays_put": "where it was, upright (moved less than 2 cm)",
                                                     "shifts": "moved 2 cm or more, or leaning, but not on its side",
                                                     "falls_over": "lying on its side"}}
        options = self._option_text(v, plan)
        if "action" in mix:
            out["safest_bin_plan"] = {"type": "choice", "instructions": f"Which choice keeps the {item} most secure in the stop?",
                                      "criteria": {k: options[k] for k in action_keys}}
        if "compare" in mix:
            out["riskier_bin_plan"] = {"type": "choice", "instructions": f"Which leaves the {item} more likely to move or fall: sticking to the plan, "
                                       f"or to {options[cf_option]}?",
                                       "criteria": {"as_planned": "sticking to the plan", cf_option: "the other choice",
                                                    "about_same": "roughly equal (within 0.1)"}}
        if "score_cf" in mix:
            out["move_risk_other_stop"] = {"type": "score", "instructions": f"Rate the chance that the {item} moves or falls if the stop were made "
                                           f"at {other} rather than {now}.", "criteria": ["0 = it will not budge", "1 = it is certain to move or fall"]}
        if "severe" in mix:
            out["parcel_falls_over"] = {"type": "noul", "instructions": f"Does the {item} end up lying on its side after the stop?"}
        return out

    def catalogs(self) -> dict[str, Any]:
        return {"liners": BIN_LINERS, "surfaces": BIN_SURFACES, "items": BIN_ITEMS, "contents": BIN_CONTENTS, "robots": BIN_ROBOTS, "domains": BIN_DOMAINS}

    def catalog_names(self) -> set[str]:
        out = {item["name"].lower() for item in BIN_ITEMS} | {r.lower() for r in BIN_ROBOTS}
        out |= {x["text"].lower() for x in list(BIN_LINERS.values()) + list(BIN_SURFACES.values())}
        return out


# ======================================================================================================================================
# tray — a drink or pot on a café robot's hand-held tray (held-out scene: never in the training file)
# ======================================================================================================================================

TRAY_SURFACES = {
    "cork": {"text": "cork-lined", "mu": (0.55, 0.75)},
    "rubber": {"text": "non-slip rubberised", "mu": (0.80, 1.05)},
    "steel": {"text": "plain stainless steel", "mu": (0.18, 0.28)},
    "lacquer": {"text": "lacquered beech", "mu": (0.28, 0.40)},
    "linen": {"text": "covered with a linen napkin", "mu": (0.40, 0.55)},
}
TRAY_ITEMS = (
    {"key": "icedtea", "name": "tall glass of iced tea", "dims": ((0.065, 0.075), (0.065, 0.075), (0.15, 0.18)), "mass": (0.35, 0.50), "base": "glass"},
    {"key": "sparkling", "name": "bottle of sparkling water", "dims": ((0.060, 0.075), (0.060, 0.075), (0.24, 0.30)), "mass": (0.6, 0.9), "base": "glass"},
    {"key": "carafe", "name": "carafe of tap water", "dims": ((0.09, 0.11), (0.09, 0.11), (0.20, 0.26)), "mass": (0.9, 1.5), "base": "glass"},
    {"key": "cappuccino", "name": "cappuccino in a cup and saucer", "dims": ((0.12, 0.14), (0.12, 0.14), (0.07, 0.09)), "mass": (0.30, 0.45),
     "base": "porcelain"},
    {"key": "pint", "name": "pint of lager", "dims": ((0.08, 0.09), (0.08, 0.09), (0.15, 0.17)), "mass": (0.7, 0.9), "base": "glass"},
    {"key": "milkshake", "name": "milkshake in a sundae glass", "dims": ((0.075, 0.085), (0.075, 0.085), (0.18, 0.22)), "mass": (0.5, 0.7), "base": "glass"},
    {"key": "teapot", "name": "pot of green tea", "dims": ((0.14, 0.18), (0.14, 0.18), (0.12, 0.16)), "mass": (0.8, 1.4), "base": "porcelain"},
)
TRAY_BASES = {"glass": (0.95, 1.05), "porcelain": (0.88, 1.00)}
TRAY_FILLS = {
    "full": {"kh": (0.42, 0.50), "text": "filled almost to the brim"},
    "half": {"kh": (0.28, 0.40), "text": "about half full"},
    "low": {"kh": (0.24, 0.36), "text": "nearly finished, only a little left"},
    "unknown": {"kh": (0.24, 0.50), "cx": (-0.06, 0.06), "cy": (-0.06, 0.06), "text": "filled to a level the robot cannot see"},
}
TRAY_ROBOTS = ("wheeled table waiter with a carrying arm", "café runner robot", "barista assistant on wheels")
TRAY_DOMAINS = ("café", "hotel lounge", "airport bar", "hospital canteen", "rooftop bistro", "museum café", "office pantry", "food court")
TRAY_SIZE = ((0.35, 0.50), (0.25, 0.38))


class TrayScene(PlatformScene):
    name = "tray"
    slide_m = 0.02
    question_names = {"key": "drink_moves_on_tray", "direction": "pace_change_effect", "outcome": "drink_after_move", "action": "safest_tray_plan",
                      "compare": "riskier_tray_plan", "score_cf": "spill_risk_other_pace", "severe": "drink_topples"}
    outcome_keys = {"unmoved": "stay", "skids": "slide", "topples": "tip"}
    dominant_option = "slower"
    factors = {
        "tray_surface": {"field": "surface", "fact": "word", "how": "word"},
        "wet": {"field": "wet", "fact": "word", "how": "word"},
        "fill_level": {"field": "fill", "fact": "word", "how": "word"},
        "footprint": {"field": "dims.0", "fact": "number", "how": "aim", "span": lambda c, v: (max(0.04, c * 0.5), c * 1.8, 0.005)},
        "ramp_time": {"field": "ramp", "fact": "number", "how": "choice"},
    }
    pair_fields = (("decisive", 0.65), ("dims.2", 0.35))
    pair_spans = {"dims.2": lambda c, v: (max(0.05, c * 0.5), min(0.45, c * 2.0), 0.005)}
    domains = TRAY_DOMAINS

    def load_dims(self, v: dict[str, Any]) -> list[float]:
        """A round or square base: the stated width is both the depth and the width."""
        return [v["dims"][0], v["dims"][0], v["dims"][2]]

    def support(self, v: dict[str, Any]) -> list[tuple[str, tuple[float, float, float], tuple[float, float, float], bool]]:
        return [("tray_top", (0.0, 0.0, -0.006), (v["tray"][0] / 2, v["tray"][1] / 2, 0.006), True)]

    def sample(self, rng: random.Random) -> tuple[dict[str, Any], dict[str, Any]]:
        item = rng.choice(TRAY_ITEMS)
        width = round(rng.uniform(*item["dims"][0]), 3)
        dims = [width, width, round(rng.uniform(*item["dims"][2]), 3)]
        move = rng.choice(("start", "stop", "turn"))
        v = {"robot": rng.choice(TRAY_ROBOTS), "domain": rng.choice(TRAY_DOMAINS), "item": item["key"], "item_name": item["name"], "dims": dims,
             "mass": round(rng.uniform(*item["mass"]), 2), "fill": rng.choice(sorted(TRAY_FILLS)), "surface": rng.choice(sorted(TRAY_SURFACES)),
             "base": item["base"], "wet": rng.random() < 0.2, "move": move, "speed": 0.0, "accel": 0.0, "radius": 0.0,
             "ramp": round(rng.uniform(0.03, 0.15), 2), "tray": [round(rng.uniform(*TRAY_SIZE[0]), 2), round(rng.uniform(*TRAY_SIZE[1]), 2)],
             "tilt": 0.0}
        if move == "turn":
            v["radius"] = round(rng.uniform(0.5, 2.0), 1)
            v["ramp"] = round(rng.uniform(0.10, 0.30), 2)
            return v, {"field": "speed", "lo": 0.2, "hi": 4.0, "step": 0.05, "increasing": True}
        v["speed"] = round(rng.uniform(0.8, 1.5), 1)
        return v, {"field": "accel", "lo": 0.2, "hi": 8.0, "step": 0.1, "increasing": True}

    def hidden_for(self, v: dict[str, Any]) -> dict[str, Any]:
        mu = TRAY_SURFACES[v["surface"]]["mu"]
        factor = TRAY_BASES[v["base"]]
        lo, hi = mu[0] * factor[0], mu[1] * factor[1]
        if v.get("wet"):
            lo, hi = lo * WET[0], hi * WET[1]
        return {"mu": ("mu", round(lo, 4), round(hi, 4)), **_com_specs(TRAY_FILLS[v["fill"]])}

    def options(self, group: Any, rng: random.Random) -> None:
        v = group.visible
        if v["move"] == "turn":
            slower = {"speed": fam._round(max(0.15, v["speed"] * rng.uniform(0.6, 0.8)), 0.05)}
        else:
            slower = {"accel": fam._round(max(0.1, v["accel"] * rng.uniform(0.5, 0.75)), 0.1)}
        group.variants.update({"opt_slower": slower, "opt_rubber": {"surface": "rubber", "wet": False}})
        if v["surface"] == "rubber":
            group.variants["opt_rubber"] = {"surface": "cork", "wet": False}
        group.plan["options"] = {"as_planned": "planned", "slower": "opt_slower", "swap_surface": "opt_rubber"}

    def factor_choices(self, group: Any, factor: str, current: Any) -> list[Any]:
        if factor == "tray_surface":
            return sorted(TRAY_SURFACES)
        if factor == "wet":
            return [not current]
        if factor == "fill_level":
            return sorted(TRAY_FILLS)
        if factor == "ramp_time":
            return [r for r in (0.01, 0.03, 0.2, 0.4) if abs(r - current) > 0.04]
        return []

    def _move(self, v: dict[str, Any], person: str) -> str:
        subject = {"third": "The robot", "second": "You", "first": "I"}[person]
        if v["move"] == "turn":
            return (f"{subject} will round a table on a curve of {v['radius']:.1f} m radius, moving {v['speed']:.2f} m/s; its sideways push grows "
                    f"over {v['ramp']:.2f} s.")
        if v["move"] == "start":
            return (f"{subject} will set off from standing, speeding up at {v['accel']:.1f} m/s² to a walking pace of {v['speed']:.1f} m/s; "
                    f"the push builds over {v['ramp']:.2f} s.")
        return (f"{subject} will come to a halt from a walking pace of {v['speed']:.1f} m/s, slowing at {v['accel']:.1f} m/s² (a rate it builds up "
                f"to over {v['ramp']:.2f} s).")

    def state_text(self, v: dict[str, Any], style: str, rng: random.Random) -> str:
        width, _, height = v["dims"]
        fill = TRAY_FILLS[v["fill"]]["text"]
        tray = f"a {num(v['tray'][0])} × {num(v['tray'][1])} m tray, {TRAY_SURFACES[v['surface']]['text']}"
        wet = ", and it stands in a ring of spilt water" if v.get("wet") else ""
        drink = f"{v['item_name']} ({num(height, 3)} m high, base {num(width, 3)} m across, {v['mass']:.2f} kg; {fill})"
        if style == "log":
            return (f"Service log, {v['domain']}. Unit: {a_or_an(v['robot'])}, carrying {tray}, held level. On it, alone in the middle: one {drink}{wet}. "
                    f"{self._move(v, 'third')}")
        if style == "brief":
            return (f"You are {a_or_an(v['robot'])} in {a_or_an(v['domain'])}, holding {tray}, perfectly level. The only thing on it is {a_or_an(drink)}, "
                    f"right in the middle{wet}. {self._move(v, 'second')}")
        if style == "sheet":
            return "\n".join((f"server: {v['robot']} ({v['domain']})", f"tray: {tray}, held level", f"on the tray (centre): {drink}",
                              f"under it: {'a ring of spilt water' if v.get('wet') else 'dry'}", f"next move: {self._move(v, 'third')}"))
        return (f"I am {a_or_an(v['robot'])} serving in the {v['domain']}, and I hold {tray} level in front of me. On it, in the middle and on its "
                f"own, is {a_or_an(drink)}{wet}. {self._move(v, 'first')}")

    def _option_text(self, v: dict[str, Any], plan: dict[str, Any]) -> dict[str, str]:
        opts = plan["option_values"]
        if v["move"] == "turn":
            slower = f"round the table at only {opts['opt_slower']['speed']:.2f} m/s"
        else:
            slower = f"change speed at only {opts['opt_slower']['accel']:.1f} m/s²"
        surface = opts["opt_rubber"]["surface"]
        swap = "move it onto a dry rubber mat on the tray first" if surface == "rubber" else "move it onto a dry cork coaster on the tray first"
        return {"as_planned": "carry on exactly as planned", "slower": slower, "swap_surface": swap}

    def questions(self, v: dict[str, Any], mix: list[str], plan: dict[str, Any], person: str, rng: random.Random, cf_option: str,
                  action_keys: list[str]) -> dict[str, Any]:
        item = v["item_name"]
        if v["move"] == "turn":
            now, other = f"{v['speed']:.2f} m/s", f"{plan['cf_value']:.2f} m/s"
            what = "curve"
        else:
            now, other = f"{v['accel']:.1f} m/s²", f"{plan['cf_value']:.1f} m/s²"
            what = "start" if v["move"] == "start" else "halt"
        out: dict[str, Any] = {}
        out["drink_moves_on_tray"] = {"type": "noul", "instructions": rng.choice((
            f"Through this {what}, does the {item} skid 2 cm or more across the tray, or topple?",
            f"Does the {what} send the {item} skidding (2 cm or more) or over on its side?"))}
        out["pace_change_effect"] = {"type": "choice", "instructions": rng.choice((
            f"What if the {what} used {other} in place of the planned {now}: would a skid or topple of the {item} become more likely, less likely, "
            "or stay about as likely?",
            f"With {other} for the {what} (planned: {now}), how do the odds that the {item} skids or topples shift?")),
            "criteria": dict(TRAY_DIRECTION)}
        if "outcome" in mix:
            out["drink_after_move"] = {"type": "choice", "instructions": f"After the {what}, what has happened to the {item}?",
                                       "criteria": {"unmoved": "nothing: it is where it was", "skids": "it skidded 2 cm or more but is upright",
                                                    "topples": "it went over"}}
        options = self._option_text(v, plan)
        if "action" in mix:
            out["safest_tray_plan"] = {"type": "choice", "instructions": f"For the {item}, which way to go is safest?",
                                       "criteria": {k: options[k] for k in action_keys}}
        if "compare" in mix:
            out["riskier_tray_plan"] = {"type": "choice", "instructions": f"Is a skid or topple of the {item} more likely if you carry on as planned, "
                                        f"or if you {options[cf_option]}?",
                                        "criteria": {"as_planned": "carrying on as planned", cf_option: "the alternative",
                                                     "about_same": "no real difference (under 0.1 apart)"}}
        if "score_cf" in mix:
            out["spill_risk_other_pace"] = {"type": "score", "instructions": f"Give the chance, from 0 to 1, that the {item} skids or topples if the "
                                            f"{what} used {other} (planned: {now}).",
                                            "criteria": ["0 = no chance it moves", "1 = it will surely skid or topple"]}
        if "severe" in mix:
            out["drink_topples"] = {"type": "noul", "instructions": f"Does the {item} go over during the {what}?"}
        return out

    def catalogs(self) -> dict[str, Any]:
        return {"surfaces": TRAY_SURFACES, "items": TRAY_ITEMS, "bases": TRAY_BASES, "fills": TRAY_FILLS, "robots": TRAY_ROBOTS,
                "domains": TRAY_DOMAINS, "tray_size": TRAY_SIZE}

    def catalog_names(self) -> set[str]:
        out = {item["name"].lower() for item in TRAY_ITEMS} | {r.lower() for r in TRAY_ROBOTS}
        out |= {f"{x['text']} tray".lower() for x in TRAY_SURFACES.values()}
        return out


#: direction-option wording of the held-out tray scene (its own sentences: no text unit shared with the training scenes)
TRAY_DIRECTION = {"more_likely": "more likely than as planned", "less_likely": "less likely than as planned",
                  "about_same": "about as likely (a change smaller than 0.1)"}
