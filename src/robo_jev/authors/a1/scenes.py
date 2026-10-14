"""Author A1 — the three scenes: stated facts, hidden priors, variants (actions), a closed-form surrogate used only to aim the
decisive number, and the telemetry-block wording.

A scene object never runs MuJoCo; :mod:`robo_jev.authors.a1.build` does. Visible facts are a flat dict (``visible``); the
decisive number of each scene is ``decel`` (mast, m/s²), ``overhang`` (dock, m) or ``grade`` (ramp, %). Hidden values come
from the word → range catalogs (:mod:`robo_jev.authors.a1.catalog`) under three priors (:func:`priors`).
"""

from __future__ import annotations

import math
import random
from statistics import NormalDist
from typing import Any, Mapping

import numpy as np

from robo_jev.authors.a1 import catalog as C
from robo_jev.authors.a1 import sims

PRIORS = ("base", "alt_a", "alt_b")
G = 9.81
_ND = NormalDist()
#: Question option keys shared by the three scenes (direction and comparison).
DIR_KEYS = ("higher", "lower", "unchanged")
PLAN_KEY = "current_plan"
TIE_KEY = "no_difference"


# --------------------------------------------------------------------------
# Priors and draws
# --------------------------------------------------------------------------


def priors(roles: Mapping[str, tuple[str, float, float]]) -> dict[str, dict[str, list[Any]]]:
    """Three priors from base entries ``name → (role, a, b)``: role ``friction`` (uniform [a, b]; × 0.7 / × 1.3), ``range``
    (uniform [a, b]; 1.5 × wider / half as wide around the same centre) or ``placement`` (normal, mean a, σ b; σ × 1.5 / × 0.75).
    Entries are ``["u", lo, hi]`` or ``["n", mean, sd]``."""
    out: dict[str, dict[str, list[Any]]] = {p: {} for p in PRIORS}
    for name, (role, a, b) in roles.items():
        a, b = float(a), float(b)
        if role == "friction":
            out["base"][name], out["alt_a"][name], out["alt_b"][name] = ["u", a, b], ["u", 0.7 * a, 0.7 * b], ["u", 1.3 * a, 1.3 * b]
        elif role == "range":
            c, w = (a + b) / 2, (b - a) / 2
            out["base"][name], out["alt_a"][name], out["alt_b"][name] = ["u", a, b], ["u", c - 1.5 * w, c + 1.5 * w], ["u", c - 0.5 * w, c + 0.5 * w]
        elif role == "placement":
            out["base"][name], out["alt_a"][name], out["alt_b"][name] = ["n", a, b], ["n", a, 1.5 * b], ["n", a, 0.75 * b]
        else:
            raise ValueError(f"unknown prior role {role!r}")
    return out


def draw_value(entry: list[Any], u: float) -> float:
    if entry[0] == "u":
        return float(entry[1]) + float(u) * (float(entry[2]) - float(entry[1]))
    if entry[0] == "n":
        return float(entry[1]) + float(entry[2]) * _ND.inv_cdf(min(max(float(u), 1e-9), 1 - 1e-9))
    raise ValueError(f"unknown prior entry {entry!r}")


def lhs(n: int, d: int, seed: int) -> np.ndarray:
    """Latin-hypercube points in (0, 1)^d: one point per 1/n stratum in every dimension (the shared draw points)."""
    rng = np.random.default_rng(int(seed))
    out = np.empty((n, d))
    for j in range(d):
        out[:, j] = (rng.permutation(n) + rng.random(n)) / n
    return out


def _r(x: float, nd: int) -> float:
    return round(float(x) + 0.0, nd)


# --------------------------------------------------------------------------
# Scene base
# --------------------------------------------------------------------------


class Scene:
    name = ""
    law = ""
    phenomena: tuple[str, ...] = ()
    decisive = ""
    decimals = 1
    hidden_names: tuple[str, ...] = ()
    #: question names by kind
    q: dict[str, str] = {}
    actions: tuple[str, str, str] = ("", "", "")
    compare_option = ""
    outcome_cost: dict[str, float] = {}
    action_cost: dict[str, float] = {}
    key_events: tuple[str, ...] = ()
    severe: str = ""
    templates = 2
    variant_transforms: dict[str, str] = {}

    @property
    def outcomes(self) -> tuple[str, ...]:
        return sims.OUTCOMES[self.name]

    # --- to override -------------------------------------------------------
    def sample_visible(self, rng: random.Random) -> dict[str, Any]: ...
    def roles(self, visible: Mapping[str, Any]) -> dict[str, tuple[str, float, float]]: ...
    def spec(self, visible: Mapping[str, Any]) -> dict[str, Any]: ...
    def state(self, visible: Mapping[str, Any]) -> dict[str, Any]: ...
    def bounds(self, visible: Mapping[str, Any]) -> tuple[float, float]: ...
    def surrogate(self, visible: Mapping[str, Any], value: float, hidden: Mapping[str, np.ndarray]) -> float: ...
    def action_overrides(self, visible: Mapping[str, Any]) -> dict[str, dict[str, Any]]: ...
    def factor_change(self, visible: Mapping[str, Any], rng: random.Random) -> tuple[str, Any, str]: ...
    def render(self, visible: Mapping[str, Any], tpl: int) -> str: ...
    def questions(self, visible: Mapping[str, Any], cf_value: float, tpl: int) -> dict[str, Any]: ...

    # --- shared ------------------------------------------------------------
    def priors(self, visible: Mapping[str, Any]) -> dict[str, dict[str, list[Any]]]:
        return priors(self.roles(visible))

    def hidden(self, visible: Mapping[str, Any], prior: str, u: np.ndarray) -> list[dict[str, float]]:
        """One hidden-value dict per draw row of `u` under `prior` (the variant's own facts, e.g. a turned pallet)."""
        entries = self.priors(visible)[prior]
        rows = []
        for row in np.atleast_2d(u):
            rows.append({name: draw_value(entries[name], row[j]) for j, name in enumerate(self.hidden_names)})
        return [self.transform_hidden(visible, h) for h in rows]

    def transform_hidden(self, visible: Mapping[str, Any], hidden: dict[str, float]) -> dict[str, float]:
        return hidden

    def quantize(self, x: float) -> float:
        return _r(x, self.decimals)

    def aim(self, visible: Mapping[str, Any], target: float, u: np.ndarray) -> tuple[float, float]:
        """The decisive value (quantized) whose surrogate key probability under the base prior is closest to `target`
        (bisection; the surrogate is non-decreasing in the decisive value), and that probability."""
        hidden = self._hidden_arrays(visible, u)
        lo, hi = self.bounds(visible)
        p_lo, p_hi = self.surrogate(visible, lo, hidden), self.surrogate(visible, hi, hidden)
        if target <= p_lo:
            x = lo
        elif target >= p_hi:
            x = hi
        else:
            a, b = lo, hi
            for _ in range(40):
                mid = (a + b) / 2
                if self.surrogate(visible, mid, hidden) < target:
                    a = mid
                else:
                    b = mid
            x = (a + b) / 2
        x = min(max(self.quantize(x), self.quantize(lo)), self.quantize(hi))
        return x, self.surrogate(visible, x, hidden)

    def surrogate_at(self, visible: Mapping[str, Any], value: float, u: np.ndarray) -> float:
        return self.surrogate(visible, value, self._hidden_arrays(visible, u))

    def _hidden_arrays(self, visible: Mapping[str, Any], u: np.ndarray) -> dict[str, np.ndarray]:
        rows = self.hidden(visible, "base", u)
        return {name: np.array([r[name] for r in rows]) for name in self.hidden_names}

    def apply(self, visible: Mapping[str, Any], overrides: Mapping[str, Any]) -> dict[str, Any]:
        out = dict(visible)
        out.update(overrides)
        return out


# --------------------------------------------------------------------------
# mast — law A
# --------------------------------------------------------------------------


class MastScene(Scene):
    name, law, phenomena = "mast", "A", ("tip_or_slide", "braking", "weight_transfer")
    decisive, decimals = "decel", 1
    hidden_names = ("mu", "tote_mass", "tote_x")
    q = {"key": "skid_or_tipup", "severe": "rear_lifts", "score_cf": "risk_at_alt_decel", "direction": "decel_shift",
         "compare": "plan_vs_low_lift", "action": "stop_choice", "outcome": "stop_result"}
    actions = ("brake_as_planned", "soft_brake", "drop_lift_first")
    compare_option = "drop_lift_first"
    outcome_cost = {"clean": 0.0, "skid": 0.6, "lift": 1.5}
    #: own costs of the alternatives (delay: a soft stop needs more aisle; lowering the lift takes a cycle) in outcome-cost units
    action_cost = {"brake_as_planned": 0.0, "soft_brake": 0.35, "drop_lift_first": 0.30}
    key_events, severe = ("skid", "lift"), "lift"
    SOFT, LOW_LIFT = 0.55, 0.40

    def sample_visible(self, rng: random.Random) -> dict[str, Any]:
        unit = rng.choice(sorted(C.MAST_UNITS))
        tote = rng.choice(sorted(C.MAST_TOTES))
        return {"unit": unit, **C.MAST_UNITS[unit], "tote": tote, "tote_mm": list(C.MAST_TOTES[tote]),
                "contents": rng.choice(sorted(C.MAST_CONTENTS)), "seat": rng.choice(sorted(C.MAST_SEATS)),
                "floor": rng.choice(sorted(C.MAST_FLOORS)), "lift_h": _r(rng.uniform(0.70, 2.40), 2),
                "speed": _r(rng.uniform(0.8, 2.2), 1), "reason": rng.choice(C.MAST_REASONS)}

    def roles(self, visible):
        return {"mu": ("friction", *C.MAST_FLOORS[visible["floor"]]), "tote_mass": ("range", *C.MAST_CONTENTS[visible["contents"]]),
                "tote_x": ("placement", 0.0, C.MAST_SEATS[visible["seat"]])}

    def spec(self, visible):
        return {k: visible[k] for k in ("wheelbase", "track", "base_mass", "base_cg_h", "lift_h")} | {
            "tote_dims": [v / 1000.0 for v in visible["tote_mm"]]}

    def state(self, visible):
        return {"speed": float(visible["speed"]), "decel": float(visible["decel"])}

    def bounds(self, visible):
        return 0.4, 7.0

    def surrogate(self, visible, value, hidden):
        a = float(value)
        mu, mt, xt = hidden["mu"], hidden["tote_mass"], hidden["tote_x"]
        mb, hb, wb = float(visible["base_mass"]), float(visible["base_cg_h"]), float(visible["wheelbase"])
        th = visible["tote_mm"][2] / 1000.0
        zt = float(visible["lift_h"]) + th / 2
        m = mb + mt
        xc, h = mt * xt / m, (mb * hb + mt * zt) / m
        d = wb / 2 - xc
        iyy = mb * ((wb + 0.12) ** 2 + (2 * hb) ** 2) / 12 + mb * (hb - h) ** 2 + mt * (zt - h) ** 2 + mt * (xt - xc) ** 2
        ip = iyy + m * (d**2 + h**2)
        frac = (wb / 2 + xc) / wb
        denom = 1 - mu * h / wb
        a_lock = np.where(denom > 0.02, mu * G * frac / np.maximum(denom, 0.02), np.inf)
        v = float(visible["speed"])
        with np.errstate(divide="ignore", invalid="ignore"):
            overrun = np.where(a > a_lock, v**2 / 2 * (1 / a_lock - 1 / a), 0.0)
        skid = overrun > sims.SKID_OVERRUN
        a_eff = np.minimum(a, a_lock)
        alpha = np.maximum(m * (a_eff * h - G * d) / ip, 0.0)
        t = v / a_eff
        phi = alpha * t**2 / 2 + (alpha * t) ** 2 * ip / (2 * m * G * d)
        lift = phi > math.radians(sims.LIFT_DEG)
        return float(np.mean(skid | lift))

    def action_overrides(self, visible):
        return {"brake_as_planned": {}, "soft_brake": {"decel": _r(self.SOFT * float(visible["decel"]), 1)},
                "drop_lift_first": {"lift_h": self.LOW_LIFT}}

    def factor_change(self, visible, rng):
        field = rng.choices(("floor", "contents", "lift_h", "speed"), weights=(0.4, 0.3, 0.2, 0.1))[0]
        if field in ("floor", "contents"):
            pool = C.MAST_FLOORS if field == "floor" else C.MAST_CONTENTS
            return field, rng.choice(sorted(k for k in pool if k != visible[field])), "word"
        if field == "lift_h":
            new = visible["lift_h"]
            while abs(new - visible["lift_h"]) < 0.4:
                new = _r(rng.uniform(0.70, 2.40), 2)
            return field, new, "number"
        new = visible["speed"]
        while abs(new - visible["speed"]) < 0.4:
            new = _r(rng.uniform(0.8, 2.2), 1)
        return field, new, "number"

    def render(self, visible, tpl):
        tl, tw, th = visible["tote_mm"]
        if tpl == 0:
            return "\n".join([
                "== TELEMETRY | mast picker ==",
                f"unit: {visible['unit']}",
                f"chassis: wb {visible['wheelbase']:.2f} m, track {visible['track']:.2f} m, {visible['base_mass']:.0f} kg, front drive, rear casters",
                f"lift_h: {visible['lift_h']:.2f} m",
                f"tote: {tl}x{tw}x{th} mm | {visible['contents']} | seat: {visible['seat']}",
                f"v: {visible['speed']:.1f} m/s",
                f"floor: {visible['floor']}",
                f"cmd: stop @ {visible['reason']}, decel {visible['decel']:.1f} m/s2",
            ])
        return "\n".join([
            f"[mast-pick {visible['unit']}]",
            f"base_geom: wheelbase {visible['wheelbase']:.2f} m / track {visible['track']:.2f} m / {visible['base_mass']:.0f} kg (drive fwd, casters aft)",
            f"tray_height: {visible['lift_h']:.2f} m",
            f"payload: tote {tl}x{tw}x{th} mm, {visible['contents']}, {visible['seat']}",
            f"surface: {visible['floor']}",
            f"cruise_v: {visible['speed']:.1f} m/s",
            f"next: brake to stop for {visible['reason']}, target {visible['decel']:.1f} m/s2",
        ])

    def questions(self, visible, cf_value, tpl):
        dec = float(visible["decel"])
        soft = _r(self.SOFT * dec, 1)
        crit = (["0.0 = certain clean stop", "1.0 = certain skid or lift"] if tpl == 0
                else ["0.0: stops clean, no doubt", "1.0: skids or lifts, no doubt"])
        return {
            self.q["key"]: {"type": "noul", "instructions": ("Will this stop end in a skid (over 6 cm past the mark) or a caster lift?" if tpl == 0
                                                              else "Skid past the mark (over 6 cm) or rear casters lifting: yes or no?")},
            self.q["severe"]: {"type": "noul", "instructions": ("Will the rear casters lift off the floor during this stop?" if tpl == 0
                                                                 else "Rear casters leave the floor on this stop?")},
            self.q["score_cf"]: {"type": "score", "instructions": (f"Skid-or-lift risk if braking at {cf_value:.1f} m/s2 instead." if tpl == 0
                                                                    else f"Skid/lift risk score for a {cf_value:.1f} m/s2 stop."), "criteria": crit},
            self.q["direction"]: {"type": "choice", "instructions": f"Decel {cf_value:.1f} m/s2 instead of {dec:.1f}: skid-or-lift risk?",
                                  "criteria": {"higher": "goes up", "lower": "goes down", "unchanged": "about the same"}},
            self.q["compare"]: {"type": "choice", "instructions": ("Riskier (skid or lift): commanded stop or lift at 0.40 m first?" if tpl == 0
                                                                    else "More skid/lift risk: commanded stop or lowered-lift stop?"),
                                "criteria": {PLAN_KEY: "as commanded", "drop_lift_first": "lift at 0.40 m first", TIE_KEY: "about equal"}},
            self.q["action"]: {"type": "choice", "instructions": "Pick the stop to run (risk against delay)." if tpl == 0 else "Operator: choose stop mode, weighing risk and delay.",
                               "criteria": {"brake_as_planned": f"{dec:.1f} m/s2, lift as is", "soft_brake": f"{soft:.1f} m/s2, lift as is",
                                            "drop_lift_first": f"lift to 0.40 m, then {dec:.1f} m/s2"}},
            self.q["outcome"]: {"type": "choice", "instructions": "Most likely result of the commanded stop?" if tpl == 0 else "Expected stop result?",
                                "criteria": {"clean": "stops near the mark", "skid": "slides past the mark", "lift": "rear casters lift"}},
        }


# --------------------------------------------------------------------------
# dock — law B
# --------------------------------------------------------------------------


class DockScene(Scene):
    name, law, phenomena = "dock", "B", ("centre_of_mass_past_edge", "overhang", "support_edge")
    decisive, decimals = "overhang", 2
    hidden_names = ("mu", "cg_x", "cg_z", "place")
    q = {"key": "goes_over_edge", "severe": "drops_off_dock", "score_cf": "risk_at_alt_overhang", "direction": "overhang_shift",
         "compare": "plan_vs_turned", "action": "setdown_choice", "outcome": "setdown_result"}
    actions = ("set_down_as_planned", "back_off_25cm", "turn_end_for_end")
    compare_option = "turn_end_for_end"
    outcome_cost = {"level": 0.0, "lean": 0.9, "drop": 2.0}
    #: backing off leaves the pallet in the dock lane (re-handling later); turning costs a manoeuvre
    action_cost = {"set_down_as_planned": 0.0, "back_off_25cm": 0.45, "turn_end_for_end": 0.20}
    key_events, severe = ("lean", "drop"), "drop"
    BACK = 0.25
    #: how a variant changes the hidden draws (recorded in the records' meta)
    variant_transforms = {"opt_turn_end_for_end": "cg_x -> -cg_x for every draw (the same pallet turned end for end)"}

    def sample_visible(self, rng):
        pallet = rng.choice(sorted(C.DOCK_PALLETS))
        length, width = C.DOCK_PALLETS[pallet]
        return {"mover": rng.choice(C.DOCK_MOVERS), "pallet": pallet, "length": length, "width": width,
                "height": _r(rng.uniform(0.9, 1.9), 2), "mass": float(rng.randrange(180, 920, 10)),
                "load": rng.choice(sorted(C.DOCK_LOADS)), "surface": rng.choice(sorted(C.DOCK_SURFACES)),
                "guidance": rng.choice(sorted(C.DOCK_GUIDANCE)), "step_h": _r(rng.uniform(0.15, 0.60), 2),
                "door": rng.randrange(1, 19), "turned": False}

    def roles(self, visible):
        load = C.DOCK_LOADS[visible["load"]]
        return {"mu": ("friction", *C.DOCK_SURFACES[visible["surface"]]), "cg_x": ("range", *load["cg_x"]), "cg_z": ("range", *load["cg_z"]),
                "place": ("placement", 0.0, C.DOCK_GUIDANCE[visible["guidance"]])}

    def transform_hidden(self, visible, hidden):
        if visible.get("turned"):
            hidden = dict(hidden)
            hidden["cg_x"] = -hidden["cg_x"]
        return hidden

    def spec(self, visible):
        return {"step_h": float(visible["step_h"]), "pallet_dims": [float(visible["length"]), float(visible["width"]), float(visible["height"])],
                "mass": float(visible["mass"])}

    def state(self, visible):
        return {"overhang": float(visible["overhang"])}

    def bounds(self, visible):
        return 0.02, round(float(visible["length"]) - 0.12, 2)

    def surrogate(self, visible, value, hidden):
        return float(np.mean(float(value) + hidden["place"] + hidden["cg_x"] > float(visible["length"]) / 2))

    def action_overrides(self, visible):
        return {"set_down_as_planned": {}, "back_off_25cm": {"overhang": _r(float(visible["overhang"]) - self.BACK, 2)},
                "turn_end_for_end": {"turned": True}}

    def factor_change(self, visible, rng):
        field = rng.choices(("load", "surface", "step_h", "guidance"), weights=(0.4, 0.2, 0.25, 0.15))[0]
        if field in ("load", "surface", "guidance"):
            pool = {"load": C.DOCK_LOADS, "surface": C.DOCK_SURFACES, "guidance": C.DOCK_GUIDANCE}[field]
            return field, rng.choice(sorted(k for k in pool if k != visible[field])), "word"
        new = visible["step_h"]
        while abs(new - visible["step_h"]) < 0.15:
            new = _r(rng.uniform(0.15, 0.60), 2)
        return field, new, "number"

    def render(self, visible, tpl):
        if tpl == 0:
            return "\n".join([
                "== TELEMETRY | pallet mover ==",
                f"unit: {visible['mover']} @ dock door {visible['door']}",
                f"pallet: {visible['pallet']}",
                f"load_h: {visible['height']:.2f} m | gross: {visible['mass']:.0f} kg",
                f"load_layout: {visible['load']}",
                f"dock_surface: {visible['surface']}",
                f"dock_drop_to_yard: {visible['step_h']:.2f} m",
                f"guidance: {visible['guidance']}",
                f"setdown_plan: lead end {visible['overhang']:.2f} m past dock edge",
            ])
        return "\n".join([
            f"[mover {visible['mover']} | door {visible['door']} | open dock, no trailer]",
            f"carried: {visible['pallet']}, {visible['height']:.2f} m high, {visible['mass']:.0f} kg",
            f"stacking: {visible['load']}",
            f"floor@edge: {visible['surface']}; drop below edge {visible['step_h']:.2f} m",
            f"nav: {visible['guidance']}",
            f"release_point: lead end overhangs edge by {visible['overhang']:.2f} m",
        ])

    def questions(self, visible, cf_value, tpl):
        o = float(visible["overhang"])
        back = _r(o - self.BACK, 2)
        back_text = f"lead end {back:.2f} m past edge" if back > 0 else (f"lead end {-back:.2f} m short of edge" if back < 0 else "lead end flush with edge")
        crit = (["0.0 = stays put for sure", "1.0 = goes over for sure"] if tpl == 0 else ["0.0: certain to stay level", "1.0: certain to tip over"])
        return {
            self.q["key"]: {"type": "noul", "instructions": ("Will the pallet tip over the dock edge once released?" if tpl == 0
                                                              else "Pallet pivots over the edge after release: yes or no?")},
            self.q["severe"]: {"type": "noul", "instructions": ("Will the pallet end up off the dock, down in the yard?" if tpl == 0
                                                                 else "Pallet leaves the dock completely?")},
            self.q["score_cf"]: {"type": "score", "instructions": (f"Rate the tip-over risk at {cf_value:.2f} m overhang." if tpl == 0
                                                                    else f"Tip-over risk score with the lead end {cf_value:.2f} m out."), "criteria": crit},
            self.q["direction"]: {"type": "choice", "instructions": f"Overhang {cf_value:.2f} m instead of {o:.2f} m: tip-over risk?",
                                  "criteria": {"higher": "goes up", "lower": "goes down", "unchanged": "about the same"}},
            self.q["compare"]: {"type": "choice", "instructions": ("Riskier set-down: as planned, or pallet turned end for end?" if tpl == 0
                                                                    else "More tip-over risk: planned set-down or turned-around set-down?"),
                                "criteria": {PLAN_KEY: "as planned", "turn_end_for_end": "turned end for end", TIE_KEY: "about equal"}},
            self.q["action"]: {"type": "choice", "instructions": "Pick the set-down (risk against re-handling)." if tpl == 0 else "Operator: choose release point, weighing risk and rework.",
                               "criteria": {"set_down_as_planned": f"lead end {o:.2f} m past edge",
                                            "back_off_25cm": back_text,
                                            "turn_end_for_end": f"turned, {o:.2f} m past edge"}},
            self.q["outcome"]: {"type": "choice", "instructions": "Expected pallet state after release?" if tpl == 0 else "Release result?",
                                "criteria": {"level": "sits level on the dock", "lean": "tips and leans on the edge", "drop": "goes off the dock"}},
        }


# --------------------------------------------------------------------------
# ramp — law C
# --------------------------------------------------------------------------


class RampScene(Scene):
    name, law, phenomena = "ramp", "C", ("friction_limited_holding", "braking_on_slope", "towing")
    decisive, decimals = "grade", 1
    hidden_names = ("mu", "payload")
    q = {"key": "fails_to_hold", "severe": "train_runs_away", "score_cf": "risk_at_alt_grade", "direction": "grade_shift",
         "compare": "ramp_a_vs_b", "action": "descent_choice", "outcome": "hold_result"}
    actions = ("stop_on_ramp", "use_ramp_b", "crawl_down")
    compare_option = "use_ramp_b"
    outcome_cost = {"held": 0.0, "overrun": 0.8, "runaway": 2.0}
    #: ramp B is a detour; crawling halves the speed on the ramp
    action_cost = {"stop_on_ramp": 0.0, "use_ramp_b": 0.25, "crawl_down": 0.20}
    key_events, severe = ("overrun", "runaway"), "runaway"
    CRAWL = 0.5

    def sample_visible(self, rng):
        tug, cart = rng.choice(sorted(C.RAMP_TUGS)), rng.choice(sorted(C.RAMP_CARTS))
        surface = rng.choice(sorted(C.RAMP_SURFACES))
        return {"tug": tug, "tug_mass": C.RAMP_TUGS[tug], "cart": cart, "cart_tare": C.RAMP_CARTS[cart],
                "contents": rng.choice(sorted(C.RAMP_CONTENTS)), "surface": surface, "speed": _r(rng.uniform(0.25, 0.80), 2),
                "place": rng.choice(C.RAMP_PLACES), "b_surface": rng.choice(sorted(k for k in C.RAMP_SURFACES if k != surface))}

    def roles(self, visible):
        return {"mu": ("friction", *C.RAMP_SURFACES[visible["surface"]]), "payload": ("range", *C.RAMP_CONTENTS[visible["contents"]])}

    def spec(self, visible):
        return {"tug_mass": float(visible["tug_mass"]), "cart_tare": float(visible["cart_tare"])}

    def state(self, visible):
        return {"grade": float(visible["grade"]), "speed": float(visible["speed"])}

    def bounds(self, visible):
        return 0.0, 32.0

    def surrogate(self, visible, value, hidden):
        theta = math.atan(float(value) / 100.0)
        mt = float(visible["tug_mass"])
        m = mt + float(visible["cart_tare"]) + hidden["payload"]
        a_net = G * (hidden["mu"] * mt * math.cos(theta) / m - math.sin(theta))
        v = float(visible["speed"])
        with np.errstate(divide="ignore", invalid="ignore"):
            dist = np.where(a_net > 0, v**2 / (2 * np.maximum(a_net, 1e-9)), np.inf)
        return float(np.mean(dist > sims.STOP_WINDOW_M))

    def action_overrides(self, visible):
        return {"stop_on_ramp": {}, "use_ramp_b": {"grade": float(visible["b_grade"]), "surface": visible["b_surface"]},
                "crawl_down": {"speed": _r(self.CRAWL * float(visible["speed"]), 2)}}

    def factor_change(self, visible, rng):
        field = rng.choices(("contents", "surface", "speed"), weights=(0.35, 0.35, 0.3))[0]
        if field == "contents":
            return field, rng.choice(sorted(k for k in C.RAMP_CONTENTS if k != visible[field])), "word"
        if field == "surface":
            return field, rng.choice(sorted(k for k in C.RAMP_SURFACES if k not in (visible["surface"], visible["b_surface"]))), "word"
        new = visible["speed"]
        while abs(new - visible["speed"]) < 0.15:
            new = _r(rng.uniform(0.25, 0.80), 2)
        return field, new, "number"

    def render(self, visible, tpl):
        if tpl == 0:
            return "\n".join([
                "== TELEMETRY | tugger train ==",
                f"tug: {visible['tug']}, {visible['tug_mass']:.0f} kg, wheels braked at stop",
                f"cart: {visible['cart']}, tare {visible['cart_tare']:.0f} kg, free casters | load: {visible['contents']}",
                f"route: {visible['place']} (ramp A), grade {visible['grade']:.1f} %, {visible['surface']}",
                f"v_approach: {visible['speed']:.2f} m/s",
                "stop: on the ramp, window 0.30 m",
                f"alt: ramp B, grade {visible['b_grade']:.1f} %, {visible['b_surface']}",
            ])
        return "\n".join([
            f"[tugger {visible['tug']} | {visible['tug_mass']:.0f} kg]",
            f"towing: {visible['cart']} ({visible['cart_tare']:.0f} kg tare, casters unbraked), {visible['contents']}",
            f"ramp_A: {visible['place']}, {visible['grade']:.1f} % grade, {visible['surface']}",
            f"ramp_B: {visible['b_grade']:.1f} % grade, {visible['b_surface']}",
            f"descent_v: {visible['speed']:.2f} m/s",
            "plan: stop mid-ramp with wheel brakes; stop window 0.30 m",
        ])

    def questions(self, visible, cf_value, tpl):
        g, v = float(visible["grade"]), float(visible["speed"])
        crawl = _r(self.CRAWL * v, 2)
        crit = (["0.0 = certain stop in window", "1.0 = certain miss"] if tpl == 0
                else ["0.0: certain clean stop", "1.0: certain missed stop"])
        return {
            self.q["key"]: {"type": "noul", "instructions": ("Will the train miss the 0.30 m stop window or fail to hold?" if tpl == 0
                                                              else "Stop window (0.30 m) missed or brakes not holding: yes or no?")},
            self.q["severe"]: {"type": "noul", "instructions": ("Will the train run away down the ramp, brakes unable to hold?" if tpl == 0
                                                                 else "Brakes lose the train entirely?")},
            self.q["score_cf"]: {"type": "score", "instructions": (f"Rate the missed-stop risk on a {cf_value:.1f} % grade." if tpl == 0
                                                                    else f"Missed-stop risk score at {cf_value:.1f} % grade."), "criteria": crit},
            self.q["direction"]: {"type": "choice", "instructions": f"Grade {cf_value:.1f} % instead of {g:.1f} %: missed-stop risk?",
                                  "criteria": {"higher": "goes up", "lower": "goes down", "unchanged": "about the same"}},
            self.q["compare"]: {"type": "choice", "instructions": ("Riskier stop: ramp A as planned, or ramp B?" if tpl == 0
                                                                    else "More missed-stop risk: planned ramp A or ramp B?"),
                                "criteria": {PLAN_KEY: "ramp A as planned", "use_ramp_b": "ramp B, same speed", TIE_KEY: "about equal"}},
            self.q["action"]: {"type": "choice", "instructions": "Pick the descent (risk against delay)." if tpl == 0 else "Operator: choose descent, weighing risk and delay.",
                               "criteria": {"stop_on_ramp": f"ramp A at {v:.2f} m/s", "use_ramp_b": f"ramp B at {v:.2f} m/s",
                                            "crawl_down": f"ramp A at {crawl:.2f} m/s"}},
            self.q["outcome"]: {"type": "choice", "instructions": "Expected outcome at the stop?" if tpl == 0 else "Stop result?",
                                "criteria": {"held": "stops inside the window", "overrun": "stops past the window", "runaway": "keeps rolling"}},
        }


SCENES: dict[str, Scene] = {"mast": MastScene(), "dock": DockScene(), "ramp": RampScene()}
