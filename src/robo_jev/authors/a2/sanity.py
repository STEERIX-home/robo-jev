"""Author A2 — physics sanity: analytic thresholds per law, monotonicity sweeps at the label level, determinism, and the agreement
of every labelled rollout with the static (closed-form) reading of its law.

* **Analytic thresholds** (:func:`threshold_checks`) — each law's textbook threshold, read in MuJoCo just below and just above it:
  turntable sliding (peak acceleration vs μg, a low wide item that cannot tip) and tipping (vs g·b/h, a tall narrow item on a grippy
  top); tiltback (planned tilt vs atan(d/h), the centre of mass against the rear edge; the hand's pull against m·g·e/lever; the arm
  limit decides ``over``); wallpress (push vs m·g/(μ_wall + μ_pad); the reflex multiple decides ``drops``).
* **Monotonicity sweeps** (:func:`sweep`) — for real group plans, one stated number swept over a grid with the group's K shared draws
  under the base prior; the key-event frequency must move one way only (0 label-level violations allowed).
* **Determinism** (:func:`determinism`) — the same rollout twice gives identical outcomes and readings.
* **Static agreement** (:func:`static_agreement`) — over labelled groups, the share of rollouts whose key outcome equals the static
  model's call, by distance from the static threshold.
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np

from robo_jev.authors.a2 import catalog as C
from robo_jev.authors.a2 import physics as PH
from robo_jev.authors.a2.scenes import SCENES, hidden_values

__all__ = ["determinism", "static_agreement", "sweep", "threshold_checks", "SWEEPS"]

G = PH.GRAVITY

#: Sweeps per scene: (field, grid factors around the plan's value, expected direction of the key frequency as the value grows).
SWEEPS = {
    "turntable": (("duration", (0.5, 0.65, 0.8, 1.0, 1.25, 1.6, 2.0), "down"), ("radius_cm", (0.5, 0.7, 0.85, 1.0, 1.2, 1.4, 1.7), "up")),
    "tiltback": (("tilt_deg", (0.5, 0.7, 0.85, 1.0, 1.15, 1.35, 1.6), "up"), ("depth_cm", (0.7, 0.8, 0.9, 1.0, 1.1, 1.25, 1.4), "down")),
    "wallpress": (("force_n", (0.4, 0.6, 0.8, 1.0, 1.25, 1.6, 2.2), "down"), ("height_cm", (0.6, 0.75, 0.9, 1.0, 1.15, 1.3, 1.5), "up")),
}


def _round_field(scene: str, field: str, value: float) -> Any:
    spec = SCENES[scene]
    if field == spec.decisive:
        return spec.round_decisive(value)
    if field == "radius_cm":
        return int(min(40, max(4, round(value))))
    if field in ("depth_cm", "height_cm"):
        return int(max(5, round(value)))
    raise ValueError(field)


def sweep(plan: Mapping[str, Any], field: str, factors: Sequence[float], *, k: int = 32, prior: str = "base") -> dict[str, Any]:
    """Key-event frequency of the plan's base facts as `field` takes `factors` × its value (K shared draws, one prior)."""
    spec = SCENES[plan["scene"]]
    draws = spec.draws(np.random.default_rng(plan["draws_seed"]), k)
    cache: dict = {}
    values, freqs = [], []
    for f in factors:
        facts = dict(plan["facts"])
        facts[field] = _round_field(plan["scene"], field, float(plan["facts"][field]) * f)
        vals = hidden_values(spec, spec.ranges(facts), draws, prior)
        hits = 0
        for i in range(k):
            out = spec.run(cache, facts, {n: float(vals[n][i]) for n in vals})["outcome"]
            hits += out in spec.key
        values.append(facts[field])
        freqs.append(hits / k)
    return {"field": field, "values": values, "freqs": freqs}


def violations(freqs: Sequence[float], direction: str) -> int:
    """Adjacent grid steps where the frequency moves against `direction` (``up`` = non-decreasing expected)."""
    sign = 1.0 if direction == "up" else -1.0
    return int(sum(1 for a, b in zip(freqs[:-1], freqs[1:]) if sign * (b - a) < -1e-12))


def determinism() -> dict[str, bool]:
    """Each scene's rollout run twice on a fresh and a reused model: identical outcome and reading."""
    out = {}
    tt = PH.TurntableScene((0.065, 0.065, 0.24), 0.9)
    args = dict(angle=2 * math.pi / 3, duration=1.1, radius=0.27, mu=0.38, com=(0.0, 0.0, -0.01))
    a, b, c = tt.run(**args), tt.run(**args), PH.TurntableScene((0.065, 0.065, 0.24), 0.9).run(**args)
    out["turntable"] = a == b == c
    tb = PH.TiltbackScene((0.32, 0.8, 1.8), 50.0)
    args = dict(tilt_deg=11.0, limit=40.0, mu=0.4, com=(0.0, 0.0, 0.1), mass=55.0)
    a, b, c = tb.run(**args), tb.run(**args), PH.TiltbackScene((0.32, 0.8, 1.8), 50.0).run(**args)
    out["tiltback"] = a == b == c
    wp = PH.WallpressScene((0.02, 0.5, 0.6), 5.0, 0.5)
    args = dict(force=40.0, reflex=2.0, mu=0.35, offset=(0.01, -0.02), mass=4.0, pad_mu=0.6)
    a, b, c = wp.run(**args), wp.run(**args), PH.WallpressScene((0.02, 0.5, 0.6), 5.0, 0.5).run(**args)
    out["wallpress"] = a == b == c
    return out


def threshold_checks() -> list[dict[str, Any]]:
    """Each law's analytic threshold read just below and above in MuJoCo (module docstring). Every row has ``ok``."""
    rows: list[dict[str, Any]] = []
    # A — sliding: a low, wide item (b/h = 1.25: cannot tip at these accelerations), μ = 0.3
    angle, radius, mu = 2 * math.pi / 3, 0.25, 0.30
    low = PH.TurntableScene((0.16, 0.16, 0.08), 0.8)
    c = PH.turntable_peak_accel(angle, 1.0, radius)
    for ratio, want in ((0.75, "stays"), (1.35, "skids")):
        duration = math.sqrt(c / (ratio * mu * G))  # peak acceleration = ratio · μg
        got = low.run(angle=angle, duration=duration, radius=radius, mu=mu, com=(0.0, 0.0, 0.0))["outcome"]
        rows.append({"law": "A", "check": f"slide: peak accel = {ratio} × μg", "expect": want, "got": got, "ok": got == want})
    # A — tipping: tall narrow item (b/h = 0.025/0.14 ≈ 0.18) on a grippy top (μ = 0.9)
    tall = PH.TurntableScene((0.05, 0.05, 0.28), 0.6)
    bh = 0.025 / 0.14
    for ratio, want in ((0.75, "stays"), (1.5, "topples")):
        duration = math.sqrt(c / (ratio * bh * G))
        got = tall.run(angle=angle, duration=duration, radius=radius, mu=0.9, com=(0.0, 0.0, 0.0))["outcome"]
        rows.append({"law": "A", "check": f"tip: peak accel = {ratio} × g·b/h", "expect": want, "got": got, "ok": got == want})
    # B — centre of mass past the rear edge: θc = atan(d/h)
    dims, mass = (0.34, 0.8, 1.8), 60.0
    tb = PH.TiltbackScene(dims, mass)
    for kh in (0.40, 0.55, 0.65):
        h = kh * dims[2]
        crit = math.degrees(math.atan2(dims[0] / 2, h))
        for delta, want in ((-1.0, "rests"), (1.0, "hangs")):
            r = tb.run(tilt_deg=crit + delta, limit=500.0, mu=0.5, com=(0.0, 0.0, (kh - 0.5) * dims[2]), mass=mass)
            rows.append({"law": "B", "check": f"kh {kh}: tilt = atan(d/h) {delta:+.0f}°", "expect": want, "got": r["outcome"], "ok": r["outcome"] == want})
        # the hand's pull vs statics m·g·e / lever, then the arm limit decides "over"
        tilt = crit + 5.0
        radius_c = math.hypot(dims[0] / 2, h)
        e = radius_c * math.sin(math.radians(tilt - crit))
        lever = dims[2] * math.cos(math.radians(tilt)) + dims[0] * math.sin(math.radians(tilt))
        static = mass * G * e / lever
        r = tb.run(tilt_deg=tilt, limit=500.0, mu=0.5, com=(0.0, 0.0, (kh - 0.5) * dims[2]), mass=mass)
        rows.append({"law": "B", "check": f"kh {kh}: pull at θc+5° vs m·g·e/lever = {static:.1f} N", "expect": "within 10 %",
                     "got": round(r["pull_n"], 1), "ok": abs(r["pull_n"] - static) <= 0.1 * static})
        for frac, want in ((1.3, "hangs"), (0.7, "over")):
            r = tb.run(tilt_deg=tilt, limit=frac * static, mu=0.5, com=(0.0, 0.0, (kh - 0.5) * dims[2]), mass=mass)
            rows.append({"law": "B", "check": f"kh {kh}: arm limit = {frac} × needed pull", "expect": want, "got": r["outcome"], "ok": r["outcome"] == want})
    # C — friction limit: push vs m·g / (μ_wall + μ_pad); the reflex multiple decides "drops"
    wp = PH.WallpressScene((0.02, 0.5, 0.6), 5.0, 0.5)
    for mass, mu_w, mu_p in ((4.0, 0.35, 0.6), (9.0, 0.25, 0.9)):
        need = mass * G / (mu_w + mu_p)
        for frac, reflex, want in ((1.1, 1.5, "holds"), (0.9, 2.0, "slips"), (0.4, 1.5, "drops")):
            r = wp.run(force=frac * need, reflex=reflex, mu=mu_w, offset=(0.0, 0.0), mass=mass, pad_mu=mu_p)
            rows.append({"law": "C", "check": f"m {mass} kg, μ {mu_w}+{mu_p}: push = {frac} × m·g/Σμ, reflex {reflex}×", "expect": want,
                         "got": r["outcome"], "ok": r["outcome"] == want})
    return rows


def static_agreement(plan: Mapping[str, Any], labelled: Mapping[str, Any]) -> list[tuple[float, bool]]:
    """(signed static margin, MuJoCo key outcome == static call) for every rollout of the group's planned setups (both members, three
    priors). The margin is the static model's distance from its threshold, as a share of the threshold (+ = the event side)."""
    from robo_jev.authors.a2.generate import member_variants

    spec = SCENES[plan["scene"]]
    draws = spec.draws(np.random.default_rng(plan["draws_seed"]), len(next(iter(labelled["setups"].values()))["codes"]["base"]))
    out: list[tuple[float, bool]] = []
    for member in ("base", "pair"):
        facts = member_variants(plan, member)["planned"]
        codes = labelled["setups"][spec.setup_key(facts)]["codes"]
        for prior in C.PRIORS:
            vals = hidden_values(spec, spec.ranges(facts), draws, prior)
            crit = spec._crit(facts, vals)
            value = float(facts[spec.decisive])
            if spec.name == "tiltback":
                margin = (value - crit) / np.maximum(np.abs(crit), 1e-6)
            elif spec.name == "turntable":
                margin = (crit - value) / value
            else:
                margin = (crit - value * int(facts.get("pads", 1))) / (value * int(facts.get("pads", 1)))
            for i, code in enumerate(codes[prior]):
                event = spec.outcomes[int(code)] in spec.key
                out.append((float(margin[i]), bool(event == (margin[i] > 0))))
    return out
