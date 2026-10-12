"""v0.3b physics sanity (Task v0.3b §1.1) — the report behind ``scripts/mjgen_v03.py sanity``.

* **Monotonicity** — candidate scenes of each family, the planned number swept along its physical axis with the scene's own base
  hidden draws held fixed (common random numbers); the key-event frequency must move one way only (label level; the report also
  counts single draws that flip back): steeper pitch (tilt) → not rarer; firmer squeeze (grip) → not more frequent; harder
  acceleration (grip) → not rarer; top box further out (tower) → not rarer.
* **Analytic thresholds** — bisection on the action for cases with a closed form: a wide low box starts to slide at
  tan θ = μ; a tall box on a grippy deck tips at tan θ = b / h_c; a held box slips when 2·μ·F < m·(g + a) (lift, lowering stop) or
  2·μ·F < m·√(g² + a²) (sideways); it falls out when 2·μ·F < m·g; a top box falls when its centre of mass passes the edge of the
  box under it; two boxes on a third topple together when their joint centre of mass passes the bottom box's edge.
* **Determinism** — labelling the same group twice (and in another process) gives identical results.
"""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any

from robo_jev.mjgen import families as fam
from robo_jev.mjgen import physics as ph

__all__ = ["SWEEPS", "analytic_grip", "analytic_tilt_slide", "analytic_tilt_tip", "analytic_tower_pair", "analytic_tower_top", "creep_grip",
           "determinism", "jobs", "label_hash", "monotone", "summarize"]

G = ph.GRAVITY
#: sweep → (family, visible field, factors or offsets, direction the event rate must keep: +1 = never down, −1 = never up)
SWEEPS = {
    "tilt_angle": ("tilt", "angle", (0.6, 0.8, 1.0, 1.2, 1.4), +1),
    "grip_squeeze": ("grip", "squeeze", (0.6, 0.8, 1.0, 1.25, 1.5), -1),
    "grip_accel": ("grip", "accel", (0.5, 0.75, 1.0, 1.5, 2.0), +1),
    "tower_top": ("tower", "top", (-0.03, -0.015, 0.0, 0.015, 0.03), +1),
}


def label_hash(label: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(label, sort_keys=True).encode("utf-8")).hexdigest()


def monotone(sweep: str, index: int, master: int, k: int) -> dict[str, Any]:
    """One candidate scene's sweep: per step the key-event frequency and per draw the event sequence (base prior draws)."""
    family, name, steps, sign = SWEEPS[sweep]
    group = fam.candidate(family, index, master, k=k, pair_share=0.0)
    v = group.visible
    hidden = fam.hidden_for(family, v)
    names = sorted(hidden)
    draws = fam.draw_values(fam.prior_ranges(hidden, "base"), fam.unit_draws(names, k, group.seed ^ 0xD3A5), names)
    rigs: dict[str, Any] = {}
    xs, rows = [], []
    for step in steps:
        trial = dict(v)
        if family == "tower":
            n = len(v["boxes"])
            value = v["boxes"][-1]["offset"] + step
            fam._set_field(trial, f"boxes.{n - 1}.offset", value)
        else:
            value = v[name] * step
            if family == "tilt":
                value = min(value, 60.0)
            trial[name] = value
        events = [fam.roll(family, trial, d, rigs)["outcome"] in fam.EVENT[family] for d in draws]
        xs.append(value)
        rows.append(events)
    freqs = [sum(r) / len(r) for r in rows]
    label_violations = sum(1 for a, b in zip(freqs, freqs[1:]) if sign * (b - a) < -1e-12)
    draw_violations = 0
    for j in range(len(draws)):
        seq = [r[j] for r in rows]
        # a draw flips back: the event happened at one step and not at the next (or the reverse for a decreasing sweep)
        draw_violations += sum(1 for a, b in zip(seq, seq[1:]) if ((a and not b) if sign > 0 else (b and not a)))
    return {"sweep": sweep, "family": family, "index": index, "x": xs, "freq": freqs, "label_violations": label_violations,
            "draw_violations": draw_violations, "draws": len(draws)}


def _bisect(event: Any, low: float, high: float, iterations: int = 10) -> tuple[float, float]:
    """`event(x)` is False at `low` and True at `high` → bracket of the switch."""
    if event(low) or not event(high):
        raise ValueError(f"bisection bracket [{low}, {high}] does not straddle the switch")
    for _ in range(iterations):
        mid = (low + high) / 2
        if event(mid):
            high = mid
        else:
            low = mid
    return low, high


def analytic_tilt_slide(mu: float) -> dict[str, Any]:
    """Wide low uniform box: the pitch where it slides (outcome ``slide``), vs atan μ."""
    expected = math.degrees(math.atan(mu))
    rig = ph.tilt_rig((0.40, 0.40, 0.12), 3.0)

    def event(angle: float) -> bool:
        return ph.run_tilt(rig, ph.pitch_profile(angle, 3.0), mu=mu, com=(0.0, 0.0, 0.0))["outcome"] == "slide"

    low, high = _bisect(event, 0.7 * expected, 1.3 * expected)
    found = (low + high) / 2
    return {"case": f"tilt-slide mu={mu}", "expected": expected, "found": found, "ratio": found / expected, "bracket": [low, high]}


def analytic_tilt_tip(length: float, height: float, kh: float) -> dict[str, Any]:
    """Tall uniform box, grippy deck (μ = 1.5), nose up (the rear edge is downhill), slow pitch (4 s): the angle where it tips, vs
    atan((L/2) / (kh·H))."""
    expected = math.degrees(math.atan((length / 2) / (kh * height)))
    rig = ph.tilt_rig((length, 0.30, height), 4.0)
    com = (0.0, 0.0, (kh - 0.5) * height)

    def event(angle: float) -> bool:
        return ph.run_tilt(rig, ph.pitch_profile(-angle, 4.0), mu=1.5, com=com)["outcome"] == "tip"

    low, high = _bisect(event, 0.7 * expected, 1.3 * expected)
    found = (low + high) / 2
    return {"case": f"tilt-tip L={length} H={height} kh={kh}", "expected": expected, "found": found, "ratio": found / expected, "bracket": [low, high]}


def analytic_grip(move: str, accel: float, mu: float = 0.5, mass: float = 1.0, *, mode: str = "slip", peak: float = 1.5,
                  distance: float = 0.8) -> dict[str, Any]:
    """Held box, a move at `accel`: the squeeze below which it slips 1 cm or more (``slip``: vs m·load/(2μ)) or falls out
    (``drop`` with a gentle move: vs m·g/(2μ)). The default move has a long acceleration phase (1.5 m/s peak): with a short one
    the box cannot slip a full centimetre just below the static threshold, so the 1-cm threshold sits a few per cent lower
    (0.95 × for a 0.12 s phase — a real duration effect, reported by :func:`jobs` as its own case)."""
    load = math.hypot(G, accel) if move == "swing" else G + accel
    expected = mass * (G if mode == "drop" else load) / (2 * mu)
    rig = ph.grip_rig((0.10, 0.08, 0.16), mass)
    path, axis = ph.move_profile(move, distance, peak, accel)

    def event(force: float) -> bool:  # True when the grip holds (above the threshold)
        outcome = ph.run_grip(rig, path, axis, mu=mu, mass=mass, squeeze=force)["outcome"]
        return outcome != "drop" if mode == "drop" else outcome == "hold"

    low, high = _bisect(event, 0.7 * expected, 1.3 * expected)
    found = (low + high) / 2
    return {"case": f"grip-{mode} {move} a={accel} peak={peak} mu={mu}", "expected": expected, "found": found, "ratio": found / expected,
            "bracket": [low, high]}


def creep_grip(ratio: float = 1.2, seconds: float = 3.0) -> dict[str, Any]:
    """Held still with 2·μ·F = `ratio`·m·g for `seconds`: slip must stay below 2 mm (soft-contact creep check)."""
    rig = ph.grip_rig((0.10, 0.08, 0.16), 1.0)
    path, axis = ph.move_profile("lift", 0.01, 0.01, 0.5, post=seconds)
    result = ph.run_grip(rig, path, axis, mu=0.5, mass=1.0, squeeze=ratio * G / (2 * 0.5))
    return {"case": f"grip-creep 2muF={ratio} mg {seconds} s", "slip_m": result["slip_m"], "outcome": result["outcome"], "passed": result["slip_m"] < 0.002}


def analytic_tower_top(lower: float = 0.40, upper: float = 0.30) -> dict[str, Any]:
    """Two uniform boxes, no errors or drop: the top-box offset where it falls, vs the lower box's half length (centre of mass at
    the edge). Reported in metres; the check is |found − expected| < 5 mm."""
    rig = ph.tower_rig([((lower, 0.30, 0.20), 4.0), ((upper, 0.25, 0.15), 2.0)])

    def event(offset: float) -> bool:
        return ph.run_tower(rig, centres=(0.0, offset), drop=0.0, mu=0.5, coms=[(0, 0, 0)] * 2)["outcome"] != "stands"

    expected = lower / 2
    low, high = _bisect(event, expected - 0.05, expected + 0.05)
    return {"case": f"tower-top L={lower}/{upper}", "expected": expected, "found": (low + high) / 2, "ratio": None, "bracket": [low, high]}


def analytic_tower_pair(top_offset: float = 0.10) -> dict[str, Any]:
    """Three uniform boxes: the top box rests safely on the middle one (offset `top_offset`); the middle box's offset where the
    middle and top boxes go over together, vs L0/2 − m2·d2/(m1 + m2)."""
    l0, l1, l2, m1, m2 = 0.40, 0.40, 0.40, 3.0, 3.0
    rig = ph.tower_rig([((l0, 0.30, 0.20), 4.0), ((l1, 0.30, 0.20), m1), ((l2, 0.30, 0.20), m2)])
    expected = l0 / 2 - m2 * top_offset / (m1 + m2)

    def event(offset: float) -> bool:
        return ph.run_tower(rig, centres=(0.0, offset, offset + top_offset), drop=0.0, mu=0.5, coms=[(0, 0, 0)] * 3)["outcome"] == "more"

    low, high = _bisect(event, expected - 0.05, expected + 0.05)
    return {"case": f"tower-pair d2={top_offset}", "expected": expected, "found": (low + high) / 2, "ratio": None, "bracket": [low, high]}


def determinism(family: str, index: int, master: int, k: int) -> dict[str, Any]:
    group = fam.candidate(family, index, master, k=k)
    first, second = fam.label_group(group, k=k), fam.label_group(group, k=k)
    return {"family": family, "index": index, "hash_1": label_hash(first), "hash_2": label_hash(second)}


def jobs(*, master: int, k: int, moments: int) -> list[tuple[str, dict[str, Any]]]:
    out: list[tuple[str, dict[str, Any]]] = []
    for sweep in SWEEPS:
        out += [("monotone", {"sweep": sweep, "index": i, "master": master, "k": k}) for i in range(moments)]
    out += [("analytic_tilt_slide", {"mu": mu}) for mu in (0.25, 0.4, 0.6)]
    out += [("analytic_tilt_tip", {"length": L, "height": H, "kh": kh}) for (L, H, kh) in ((0.20, 0.60, 0.5), (0.30, 0.70, 0.5), (0.25, 0.60, 0.35))]
    out += [("analytic_grip", {"move": move, "accel": a}) for move, a in (("lift", 3.0), ("lower", 4.0), ("swing", 5.0))]
    out += [("analytic_grip", {"move": "lift", "accel": 0.5, "mode": "drop"})]
    out += [("analytic_grip", {"move": "swing", "accel": 5.0, "peak": 0.6, "distance": 0.3, "mode": "short"})]
    out += [("creep_grip", {})]
    out += [("analytic_tower_top", {"lower": lo, "upper": up}) for lo, up in ((0.40, 0.30), (0.30, 0.45))]
    out += [("analytic_tower_pair", {"top_offset": d}) for d in (0.05, 0.12)]
    out += [("determinism", {"family": family, "index": 0, "master": master, "k": 4}) for family in fam.FAMILIES]
    return out


def summarize(job_list: list[tuple[str, dict[str, Any]]], results: list[Any]) -> dict[str, Any]:
    """Results → report with a verdict: zero label-level monotonicity violations, analytic ratios within tolerance (slide and slip
    0.95–1.05, tip 0.90–1.05 — the pitch motion itself helps a tall box over a little early —, drop 0.90–1.10, tower |found −
    expected| < 5 mm; the short-phase slip 0.90–1.05), creep < 2 mm, identical determinism hashes (twice in a worker and once in this process)."""
    mono = [r for (kind, _), r in zip(job_list, results) if kind == "monotone"]
    analytic = [r for (kind, _), r in zip(job_list, results) if kind.startswith("analytic")]
    creep = [r for (kind, _), r in zip(job_list, results) if kind == "creep_grip"]
    determ = [r for (kind, _), r in zip(job_list, results) if kind == "determinism"]
    for entry, (kind, payload) in zip(determ, [(k, p) for k, p in job_list if k == "determinism"]):
        here = determinism(**payload)
        entry["hash_here"] = here["hash_1"]
        entry["identical"] = entry["hash_1"] == entry["hash_2"] == entry["hash_here"]

    def within(entry: dict[str, Any]) -> bool:
        case = entry["case"]
        if case.startswith("tower"):
            return abs(entry["found"] - entry["expected"]) < 0.005
        lo, hi = ((0.95, 1.05) if case.startswith(("tilt-slide", "grip-slip")) else (0.90, 1.05) if case.startswith(("tilt-tip", "grip-short"))
                  else (0.90, 1.10))
        return lo <= entry["ratio"] <= hi

    for entry in analytic:
        entry["within_tolerance"] = within(entry)
    by_sweep: dict[str, Any] = {}
    for entry in mono:
        row = by_sweep.setdefault(entry["sweep"], {"moments": 0, "label_violations": 0, "draw_violations": 0, "draw_steps": 0})
        row["moments"] += 1
        row["label_violations"] += entry["label_violations"]
        row["draw_violations"] += entry["draw_violations"]
        row["draw_steps"] += entry["draws"] * max(0, len(entry["x"]) - 1)
    verdict = {
        "monotone_label_level": all(row["label_violations"] == 0 for row in by_sweep.values()),
        "analytic": all(e["within_tolerance"] for e in analytic),
        "no_creep": all(e["passed"] for e in creep),
        "deterministic": all(e["identical"] for e in determ),
    }
    verdict["passed"] = all(verdict.values())
    return {"kind": "v03b-sanity", "verdict": verdict, "monotonicity": {"by_sweep": by_sweep, "moments": mono}, "analytic": analytic, "creep": creep,
            "determinism": determ}
