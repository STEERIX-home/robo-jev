"""T8b physics sanity checks (Task T8 §1-2) — the report behind ``scripts/t8.py sanity``.

* **Monotonicity** — for candidate moments of each family, the planned action is swept along its physical axis with the
  moment's own hidden draws held fixed (common random numbers): harder braking (deck_brake), a faster curve (deck_turn),
  more overhang (stack), a higher push (push). The outcome frequency must never go down along the sweep (label level);
  the report also counts single draws whose outcome flips back (draw level).
* **Analytic thresholds** — bisection on the action for cases with a closed-form answer:
  tipping onset on a braking deck, a* = g·(L/2)/h_c (no slip, μ = 1.5); sliding onset, a* = μ·g (wide low box, μ = 0.3);
  tipping over under a long hold (just above the onset); the push height where a pushed box tips instead of sliding,
  h* = μ_b·L + (L/2)(1 − μ·μ_b)/μ for a smooth bar with friction μ_b; a box released flat falls iff its centre of mass is
  past the support's edge (threshold 0).
* **Determinism** — labelling the same moment twice (and in another process) gives identical results.
"""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any

from robo_jev.outside import physics as ph
from robo_jev.outside.moments import _com, candidate, hidden_draws, label_moment

__all__ = ["SWEEPS", "analytic_push", "analytic_stack", "analytic_tip", "analytic_slide", "determinism", "jobs", "label_hash", "monotone", "summarize"]

#: Sweep factors (deck_brake: decel ×, deck_turn: speed ×, stack: overhang + metres, push: height ×).
SWEEPS = {
    "deck_brake": (0.4, 0.6, 0.8, 1.0, 1.25, 1.6, 2.0),
    "deck_turn": (0.6, 0.8, 1.0, 1.2, 1.4),
    "stack": (-0.06, -0.03, 0.0, 0.03, 0.06),
    "push": (0.5, 0.75, 1.0, 1.25, 1.5),
}
G = ph.GRAVITY


def label_hash(label: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(label, sort_keys=True).encode("utf-8")).hexdigest()


def monotone(family: str, index: int, master: int, k: int) -> dict[str, Any]:
    """One candidate's sweep: per step the event frequency (tip / fall) and per draw the outcome sequence."""
    moment = candidate(family, index, master)
    v = moment.visible
    draws = hidden_draws(moment.hidden, k, moment.seed ^ 0x5EED)
    steps: list[dict[str, Any]] = []
    if family == "deck_brake":
        scene = ph.Scene(ph.deck_xml(v["shape"], v["dims"], v["mass"]))
        for f in SWEEPS[family]:
            decel = min(9.0, v["decel"] * f)
            profile = ph.brake_profile(v["speed"], decel)
            events = [ph.run_deck(scene, profile, mu=d["mu"], com=_com(family, v, d))["outcome"] == "tip" for d in draws]
            steps.append({"x": decel, "events": events})
    elif family == "deck_turn":
        scene = ph.Scene(ph.deck_xml(v["shape"], v["dims"], v["mass"]))
        direction = 1 if v["direction"] == "left" else -1
        for f in SWEEPS[family]:
            speed = v["speed"] * f
            ramp = min(v["steer_ramp"], 0.4 * (math.pi / 2) * v["radius"] / speed)
            profile = ph.turn_profile(speed, v["radius"], direction=direction, ramp=ramp)
            events = [ph.run_deck(scene, profile, mu=d["mu"], com=_com(family, v, d))["outcome"] == "tip" for d in draws]
            steps.append({"x": speed, "events": events})
    elif family == "stack":
        scene = ph.Scene(ph.stack_xml(v["dims"], v["mass"], (v["support_top"], max(0.3, v["dims"][1] + 0.1), v["support_height"])))
        length = v["dims"][0]
        for delta in SWEEPS[family]:
            overhang = v["overhang"] + delta
            events = []
            for d in draws:
                centre = v["support_top"] / 2 + overhang - length / 2 + d["eps"]
                result = ph.run_stack(scene, x_centre=centre, drop=v["drop"], tilt0_deg=d["tilt0"], mu=d["mu"], com=_com("stack", v, d),
                                      support_height=v["support_height"], box_height=v["dims"][2])
                events.append(result["outcome"] == "fall")
            steps.append({"x": overhang, "events": events})
    else:
        profile = ph.push_profile(v["speed"], v["travel"])
        for f in SWEEPS[family]:
            height = v["push_height"] * f
            if not 0.03 <= height <= v["dims"][2] - 0.02:
                continue
            scene = ph.Scene(ph.push_xml(v["dims"], v["mass"], height))
            events = [ph.run_push(scene, profile, mu=d["mu"], com=_com("push", v, d))["outcome"] == "tip" for d in draws]
            steps.append({"x": height, "events": events})
    freqs = [sum(s["events"]) / len(s["events"]) for s in steps]
    label_violations = sum(1 for a, b in zip(freqs, freqs[1:]) if b < a - 1e-12)
    draw_violations = 0
    for j in range(len(draws)):
        seq = [s["events"][j] for s in steps]
        draw_violations += sum(1 for a, b in zip(seq, seq[1:]) if a and not b)
    return {"family": family, "index": index, "x": [s["x"] for s in steps], "freq": freqs, "label_violations": label_violations,
            "draw_violations": draw_violations, "draws": len(draws)}


def _bisect(event: Any, low: float, high: float, iterations: int = 9) -> tuple[float, float]:
    """`event(x)` is False at `low` and True at `high` → bracket of the switch after `iterations` halvings."""
    if event(low) or not event(high):
        raise ValueError("bisection bracket does not straddle the switch")
    for _ in range(iterations):
        mid = (low + high) / 2
        if event(mid):
            high = mid
        else:
            low = mid
    return low, high


def analytic_tip(length: float, height: float, kh: float, *, mode: str = "onset") -> dict[str, Any]:
    """Braking deck, μ = 1.5 (no slip), uniform box with the centre of mass at ``kh · height``: the deceleration where tipping starts
    (``onset``: peak tilt > 1°) or where the box goes over during a 6 s hold (``over``), vs a* = g·(L/2)/h_c."""
    expected = G * (length / 2) / (kh * height)
    scene = ph.Scene(ph.deck_xml("box", (length, 0.30, height), 4.0))
    com = (0.0, 0.0, (kh - 0.5) * height)
    hold = 2.0 if mode == "onset" else 6.0

    def event(decel: float) -> bool:
        profile = ph.brake_profile(decel * hold, decel, post=1.0)
        result = ph.run_deck(scene, profile, mu=1.5, com=com)
        return result["max_tilt"] > 1.0 if mode == "onset" else result["outcome"] == "tip"

    low, high = _bisect(event, 0.6 * expected, 1.6 * expected)
    found = (low + high) / 2
    return {"case": f"tip-{mode} L={length} H={height} kh={kh}", "expected": expected, "found": found, "ratio": found / expected,
            "bracket": [low, high]}


def analytic_slide(mu: float = 0.3) -> dict[str, Any]:
    """Braking deck, wide low box (cannot tip first), friction μ: the deceleration where it slides more than 1 cm during a 2 s hold,
    vs a* = μ·g."""
    expected = mu * G
    scene = ph.Scene(ph.deck_xml("box", (0.40, 0.40, 0.15), 3.0))

    def event(decel: float) -> bool:
        return ph.run_deck(scene, ph.brake_profile(decel * 2.0, decel, post=0.5), mu=mu, com=(0.0, 0.0, 0.0))["slide_m"] > 0.01

    low, high = _bisect(event, 0.6 * expected, 1.6 * expected)
    found = (low + high) / 2
    return {"case": f"slide-onset mu={mu}", "expected": expected, "found": found, "ratio": found / expected, "bracket": [low, high]}


def analytic_push(length: float = 0.40, height: float = 0.80, mu: float = 0.5) -> dict[str, Any]:
    """Pushed uniform box at 0.10 m/s: the bar height where it tips instead of sliding, vs h* = μ_b·L + (L/2)(1 − μ·μ_b)/μ."""
    mu_b = ph.BAR_FRICTION
    expected = mu_b * length + (length / 2) * (1 - mu * mu_b) / mu
    profile = ph.push_profile(0.10, 0.40)

    def event(h: float) -> bool:
        scene = ph.Scene(ph.push_xml((length, 0.40, height), 8.0, h))
        return ph.run_push(scene, profile, mu=mu, com=(0.0, 0.0, 0.0))["outcome"] == "tip"

    low, high = _bisect(event, 0.6 * expected, min(height - 0.03, 1.4 * expected))
    found = (low + high) / 2
    return {"case": f"push-tip L={length} H={height} mu={mu}", "expected": expected, "found": found, "ratio": found / expected, "bracket": [low, high]}


def analytic_stack(length: float = 0.40, height: float = 0.20) -> dict[str, Any]:
    """Box released flat (no drop, no tilt) on a support: position of its centre of mass past the edge where it starts to fall
    (expected 0). Reported in metres (``found``) — the check is |found| < 5 mm."""
    support = (0.60, 0.40, 0.50)
    scene = ph.Scene(ph.stack_xml((length, 0.25, height), 3.0, support))

    def event(offset: float) -> bool:
        centre = support[0] / 2 + offset
        return ph.run_stack(scene, x_centre=centre, drop=0.0, tilt0_deg=0.0, mu=0.4, com=(0.0, 0.0, 0.0), support_height=support[2],
                            box_height=height)["outcome"] == "fall"

    low, high = _bisect(event, -0.05, 0.05)
    return {"case": f"stack-edge L={length} H={height}", "expected": 0.0, "found": (low + high) / 2, "ratio": None, "bracket": [low, high]}


def determinism(family: str, index: int, master: int, k: int) -> dict[str, Any]:
    moment = candidate(family, index, master)
    first, second = label_moment(moment, k=k), label_moment(moment, k=k)
    return {"family": family, "index": index, "hash_1": label_hash(first), "hash_2": label_hash(second)}


def jobs(*, master: int, k: int, moments: int) -> list[tuple[str, dict[str, Any]]]:
    out: list[tuple[str, dict[str, Any]]] = []
    for family in SWEEPS:
        out += [("monotone", {"family": family, "index": i, "master": master, "k": k}) for i in range(moments)]
    out += [("analytic_tip", {"length": L, "height": H, "kh": kh, "mode": mode})
            for (L, H, kh) in ((0.20, 0.60, 0.5), (0.30, 0.60, 0.5), (0.20, 0.50, 0.35), (0.15, 0.60, 0.65)) for mode in ("onset", "over")]
    out += [("analytic_slide", {"mu": mu}) for mu in (0.2, 0.3, 0.45)]
    out += [("analytic_push", {"length": L, "height": H, "mu": mu}) for (L, H, mu) in ((0.40, 0.80, 0.5), (0.30, 0.90, 0.7), (0.50, 1.00, 0.45))]
    out += [("analytic_stack", {"length": L, "height": H}) for (L, H) in ((0.40, 0.20), (0.30, 0.12))]
    out += [("determinism", {"family": family, "index": 0, "master": master, "k": 8}) for family in SWEEPS]
    return out


def summarize(job_list: list[tuple[str, dict[str, Any]]], results: list[Any]) -> dict[str, Any]:
    """Results → report with a verdict: zero label-level monotonicity violations, analytic ratios within the tolerances (onset 0.95–1.05,
    over 0.98–1.25 — tipping over needs a little more than the onset —, slide 0.95–1.05, push 0.85–1.10, stack |offset| < 5 mm),
    identical determinism hashes (twice in a worker and once in this process)."""
    mono = [r for (kind, _), r in zip(job_list, results) if kind == "monotone"]
    analytic = [r for (kind, _), r in zip(job_list, results) if kind.startswith("analytic")]
    determ = [r for (kind, _), r in zip(job_list, results) if kind == "determinism"]
    for entry in determ:
        here = determinism(entry["family"], entry["index"], *_master_k(job_list, entry))
        entry["hash_here"] = here["hash_1"]
        entry["identical"] = entry["hash_1"] == entry["hash_2"] == entry["hash_here"]

    def within(entry: dict[str, Any]) -> bool:
        case = entry["case"]
        if case.startswith("stack"):
            return abs(entry["found"]) < 0.005
        lo, hi = ((0.95, 1.05) if "onset" in case and case.startswith("tip") else (0.98, 1.25) if case.startswith("tip-over")
                  else (0.95, 1.05) if case.startswith("slide") else (0.85, 1.10))
        return lo <= entry["ratio"] <= hi

    for entry in analytic:
        entry["within_tolerance"] = within(entry)
    by_family: dict[str, Any] = {}
    for entry in mono:
        fam = by_family.setdefault(entry["family"], {"moments": 0, "label_violations": 0, "draw_violations": 0, "draw_steps": 0})
        fam["moments"] += 1
        fam["label_violations"] += entry["label_violations"]
        fam["draw_violations"] += entry["draw_violations"]
        fam["draw_steps"] += entry["draws"] * max(0, len(entry["x"]) - 1)
    verdict = {
        "monotone_label_level": all(f["label_violations"] == 0 for f in by_family.values()),
        "analytic": all(e["within_tolerance"] for e in analytic),
        "deterministic": all(e["identical"] for e in determ),
    }
    verdict["passed"] = all(verdict.values())
    return {"kind": "t8b-sanity", "verdict": verdict, "monotonicity": {"by_family": by_family, "moments": mono}, "analytic": analytic,
            "determinism": determ}


def _master_k(job_list: list[tuple[str, dict[str, Any]]], entry: dict[str, Any]) -> tuple[int, int]:
    for kind, payload in job_list:
        if kind == "determinism" and payload["family"] == entry["family"] and payload["index"] == entry["index"]:
            return payload["master"], payload["k"]
    raise KeyError(entry["family"])
