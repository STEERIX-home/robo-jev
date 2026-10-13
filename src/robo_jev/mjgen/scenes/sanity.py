"""v0.4a physics sanity per new scene (Task v0.4a §1.4) — the report behind ``scripts/mjgen_v04.py sanity``.

* **Monotonicity** — candidate scenes, one stated number swept along its physical axis with the scene's own base-prior hidden draws
  held fixed (common random numbers); the key-event frequency must move one way only (label level; draws that flip back are counted):
  a harder stop or a faster turn → not rarer (law A); a firmer squeeze → not more frequent, a harder move → not rarer (law C); a larger
  overhang, reach or drawer extension → not rarer (law B); a mast tilted further back → not more frequent in a stop (forks).
* **Analytic thresholds** — bisection on the action for cases with a closed form, in each scene's own geometry: a wide low load slides at
  a = μg (forks, tray; with a mast tilt β, a = g(μ + tan β)/(1 − μ tan β)); a tall load tips at a = g·b/h_c (forks, bin, tray), in a turn
  about the outer tine edge when the load is wider than the forks; a bin wall right ahead keeps a tipping item upright; a held item slips
  at 2μF = m(g + a) (lift), m·√(g² + a²) (a braking carry); an item falls off a table edge when its centre of mass passes the edge;
  a base tips when the system's centre of mass passes its front feet.
* **Determinism** — labelling the same group twice in a worker and once in the parent gives identical results.
"""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any

from robo_jev.mjgen import families as fam
from robo_jev.mjgen.scenes import rigs
from robo_jev.mjgen.scenes import build as B

__all__ = ["SWEEPS", "analytic", "determinism", "factor_sweep", "jobs", "label_hash", "monotone", "summarize", "summarize_sweeps"]

G = rigs.GRAVITY

#: per scene: (sweep name, visible field — "decisive" for the scene's decisive number —, multipliers, direction the event rate must keep)
SWEEPS = {
    "forks": (("decisive", "decisive", (0.6, 0.8, 1.0, 1.2, 1.4), +1), ("mast_tilt_brake", "tilt", (0.0, 2.0, 4.0, 6.0, 8.0), -1)),
    "bin": (("decisive", "decisive", (0.6, 0.8, 1.0, 1.2, 1.4), +1),),
    "tray": (("decisive", "decisive", (0.6, 0.8, 1.0, 1.2, 1.4), +1),),
    "plate": (("decisive", "decisive", (0.6, 0.8, 1.0, 1.25, 1.5), -1), ("accel", "accel", (0.5, 0.75, 1.0, 1.5, 2.0), +1)),
    "clamp": (("decisive", "decisive", (0.6, 0.8, 1.0, 1.25, 1.5), -1), ("accel", "accel", (0.5, 0.75, 1.0, 1.5, 2.0), +1)),
    "palms": (("decisive", "decisive", (0.6, 0.8, 1.0, 1.25, 1.5), -1),),
    "table": (("decisive", "decisive", (-0.03, -0.015, 0.0, 0.015, 0.03), +1),),
    "reach": (("decisive", "decisive", (-0.10, -0.05, 0.0, 0.05, 0.10), +1),),
    "cabinet": (("decisive", "decisive", (-0.08, -0.04, 0.0, 0.04, 0.08), +1),),
}
#: law-B sweeps add metres to the decisive number; the others multiply it (or set the level directly: mast tilt)
ADDITIVE = {"table", "reach", "cabinet"}


def label_hash(label: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(label, sort_keys=True).encode("utf-8")).hexdigest()


def monotone(scene: str, sweep: str, index: int, master: int, k: int) -> dict[str, Any]:
    """One candidate scene's sweep: per step the key-event frequency and per draw the event sequence (base prior draws)."""
    spec = B.spec(scene)
    entry = next(s for s in SWEEPS[scene] if s[0] == sweep)
    _, field, steps, sign = entry
    group = None
    for offset in range(40):  # a brake for the mast-tilt sweep; any move otherwise
        trial = B.candidate(scene, index + 1000 * offset, master, k=k, pair_share=0.0)
        if sweep != "mast_tilt_brake" or trial.visible["move"] == "brake":
            group = trial
            break
    assert group is not None
    v = group.visible
    name = group.plan["decisive"] if field == "decisive" else field
    lo, hi, step_size, _ = group.plan["decisive_span"] if field == "decisive" else (None, None, None, None)
    draws = B._draws(spec, v, group.seed ^ 0xD3A5, k)
    rigs_cache: dict[str, Any] = {}
    xs, rows = [], []
    for step in steps:
        trial = json.loads(json.dumps(v))
        if sweep == "mast_tilt_brake":
            value = step
        elif scene in ADDITIVE and field == "decisive":
            value = min(hi, max(lo, fam._get_field(v, name) + step))
        else:
            value = fam._get_field(v, name) * step
            if field == "decisive":
                value = min(hi, max(lo, value))
        fam._set_field(trial, name, value)
        if not spec.accept(trial):
            continue
        events = [spec.roll(trial, d, rigs_cache)["outcome"] in spec.event for d in draws]
        xs.append(value)
        rows.append(events)
    freqs = [sum(r) / len(r) for r in rows]
    label_violations = sum(1 for a, b in zip(freqs, freqs[1:]) if sign * (b - a) < -1e-12)
    draw_violations = 0
    for j in range(len(draws)):
        seq = [r[j] for r in rows]
        draw_violations += sum(1 for a, b in zip(seq, seq[1:]) if ((a and not b) if sign > 0 else (b and not a)))
    return {"scene": scene, "sweep": sweep, "index": index, "field": name, "x": xs, "freq": freqs, "label_violations": label_violations,
            "draw_violations": draw_violations, "draws": len(draws)}


def _bisect(event: Any, low: float, high: float, iterations: int = 11) -> tuple[float, float]:
    if event(low) or not event(high):
        raise ValueError(f"bisection bracket [{low}, {high}] does not straddle the switch")
    for _ in range(iterations):
        mid = (low + high) / 2
        if event(mid):
            high = mid
        else:
            low = mid
    return low, high


def _platform(scene: str, visible: dict[str, Any]) -> tuple[Any, Any]:
    spec = B.spec(scene)
    return spec, spec.rig_for(visible, {})


def analytic(case: str, **p: Any) -> dict[str, Any]:
    """One analytic case → ``{"case", "expected", "found", "ratio" | "diff_m", "bracket"}`` (cases listed in :func:`jobs`)."""
    if case in ("forks_slide", "forks_tip", "forks_turn_tip", "tray_slide", "tray_tip", "bin_tip"):
        scene = case.split("_")[0]
        spec = B.spec(scene)
        mu, tilt = p.get("mu", 1.5), p.get("tilt", 0.0)
        dims = p["dims"]
        v = {"dims": list(dims), "mass": 20.0, "tilt": tilt, "spread": p.get("spread", 0.5), "tine_w": p.get("tine_w", 0.12), "gap": p.get("gap", 0.6),
             "wall_h": 0.5, "laid": False, "tray": [0.6, 0.6], "ramp": 0.05}
        rig = spec.rig_for(v, {})
        turn = case.endswith("turn_tip")
        backward = case == "tray_tip"

        def event(a: float) -> bool:
            if turn:
                path = rigs.linear_profile(a * 3.0, a, jerk_time=0.3, pre=0.3, post=1.0, sign=-1.0)
            else:
                path = rigs.linear_profile(a * 3.0, a, jerk_time=0.1 if case.endswith("slide") else 0.4, pre=0.3, post=1.0, sign=1.0 if backward else -1.0)
            out = rigs.run_platform(rig, path, 1 if turn else 0, mu=mu, com=(0.0, 0.0, 0.0))
            return spec.classify(out) in spec.event if case.endswith("slide") else (out["tilt_deg"] > 45.0 or out["fell"] or out["max_tilt_deg"] > 30.0)

        beta = math.radians(tilt)
        length, width, height = spec.load_dims(v)
        if case.endswith("slide"):
            expected = G * (mu + math.tan(beta)) / (1 - mu * math.tan(beta))
        elif turn:
            expected = G * min(width / 2, v["spread"] / 2 + v["tine_w"] / 2) / (height / 2)
        else:
            expected = G * (length / 2) / (height / 2)
        low, high = _bisect(event, 0.6 * expected, 1.6 * expected)
        found = (low + high) / 2
        return {"case": f"{case} {p}", "expected": expected, "found": found, "ratio": found / expected, "bracket": [low, high]}
    if case == "bin_wall_holds":
        spec = B.spec("bin")
        v = {"dims": [0.10, 0.10, 0.40], "mass": 2.0, "tilt": 0.0, "gap": 0.015, "wall_h": 0.35, "laid": False, "ramp": 0.05}
        rig = spec.rig_for(v, {})
        path = rigs.linear_profile(3.0, 6.0, jerk_time=0.05, pre=0.3, post=1.0, sign=-1.0)
        out = rigs.run_platform(rig, path, 0, mu=1.5, com=(0.0, 0.0, 0.0))
        return {"case": "bin_wall_holds a=6 (tip threshold 2.45) gap 1.5 cm", "expected": "stays", "found": spec.classify(out), "max_tilt_deg": out["max_tilt_deg"],
                "passed": spec.classify(out) == "stays"}
    if case in ("squeeze_lift", "squeeze_brake", "squeeze_walk"):
        mass, mu, a = p.get("mass", 3.0), p.get("mu", 0.6), p.get("accel", 3.0)
        dims, pad, hand = p["dims"], p["pad"], p.get("hand", 400.0)
        rig = rigs.squeeze_rig("sanity", dims, pad_half=pad, hand_mass=hand)
        if case == "squeeze_lift":
            from robo_jev.mjgen import physics as ph

            path, axis = ph.move_profile("lift", 0.8, 1.5, a, jerk_time=0.04)
            load = G + a
        elif case == "squeeze_brake":
            path, axis = rigs.brake_carry_profile(a * 0.8, a, jerk_time=0.04), 0
            load = math.hypot(G, a)
        else:
            path, axis = rigs.walk_profile(p["rise"], p["hz"], 8), 1
            load = G + p["rise"] / 2 * (2 * math.pi * p["hz"]) ** 2
        expected = mass * load / (2 * mu)

        def held(force: float) -> bool:
            return rigs.run_squeeze(rig, path, axis, mu=mu, mass=mass, squeeze=force, slip_m=0.01, pre=0.4)["outcome"] == "hold"

        low, high = _bisect(held, 0.7 * expected, 1.3 * expected)
        found = (low + high) / 2
        return {"case": f"{case} {p}", "expected": expected, "found": found, "ratio": found / expected, "bracket": [low, high]}
    if case == "table_edge":
        rig = rigs.ledge_rig(p["dims"], 3.0)
        cx = p.get("cx", 0.0)

        def falls(x: float) -> bool:
            return rigs.run_ledge(rig, x_centre=x, drop=0.0, mu=0.5, com=(cx * p["dims"][0], 0.0, 0.0))["outcome"] == "falls"

        expected = -cx * p["dims"][0]
        low, high = _bisect(falls, expected - 0.05, expected + 0.05)
        return {"case": f"table_edge {p}", "expected": expected, "found": (low + high) / 2, "diff_m": (low + high) / 2 - expected, "bracket": [low, high]}
    if case in ("reach_tip", "cabinet_tip"):
        scene = case.split("_")[0]
        spec = B.spec(scene)
        v = dict(p["visible"])
        rig = spec.rig_for(v, {})
        mover_mass = p["mover_mass"]

        def tips(travel: float) -> bool:
            path = rigs.shift_profile(travel, 6.0)
            return rigs.run_shift(rig, path, mu=0.9, base_com=(0.0, 0.0, 0.0), mover_mass=mover_mass)["outcome"] == "tips"

        expected = spec.static_travel(v, mover_mass)
        low, high = _bisect(tips, max(0.0, expected - 0.25), expected + 0.25)
        return {"case": f"{case} m={mover_mass}", "expected": expected, "found": (low + high) / 2, "diff_m": (low + high) / 2 - expected, "bracket": [low, high]}
    raise ValueError(case)


def determinism(scene: str, index: int, master: int, k: int) -> dict[str, Any]:
    group = B.candidate(scene, index, master, k=k)
    first, second = B.label_group(group, k=k), B.label_group(group, k=k)
    return {"scene": scene, "index": index, "hash_1": label_hash(first), "hash_2": label_hash(second)}


def jobs(*, scenes: list[str] | None, master: int, k: int, moments: int) -> list[tuple[str, dict[str, Any]]]:
    from robo_jev.mjgen.scenes import SCENES

    names = list(scenes or SCENES)
    out: list[tuple[str, dict[str, Any]]] = []
    for scene in names:
        for sweep in SWEEPS.get(scene, ()):
            out += [("monotone", {"scene": scene, "sweep": sweep[0], "index": 960_000 + i, "master": master, "k": k}) for i in range(moments)]
        out.append(("determinism", {"scene": scene, "index": 0, "master": master, "k": 4}))
    cases = {
        "forks": [("analytic", {"case": "forks_slide", "mu": 0.4, "dims": (0.6, 0.9, 0.3)}), ("analytic", {"case": "forks_slide", "mu": 0.3, "tilt": 4.0, "dims": (0.6, 0.9, 0.3)}),
                  ("analytic", {"case": "forks_tip", "dims": (0.4, 0.9, 1.4)}),
                  ("analytic", {"case": "forks_turn_tip", "dims": (0.8, 1.0, 1.2), "spread": 0.4, "tine_w": 0.12})],
        "bin": [("analytic", {"case": "bin_tip", "dims": (0.10, 0.10, 0.40), "gap": 0.6}), ("analytic", {"case": "bin_wall_holds"})],
        "tray": [("analytic", {"case": "tray_slide", "mu": 0.3, "dims": (0.12, 0.12, 0.05)}), ("analytic", {"case": "tray_tip", "dims": (0.07, 0.07, 0.25)})],
        "plate": [("analytic", {"case": "squeeze_lift", "mass": 0.3, "mu": 0.4, "accel": 6.0, "dims": (0.128, 0.086, 0.04), "pad": (0.015, 0.004, 0.012)}),
                  ("analytic", {"case": "squeeze_brake", "mass": 0.3, "mu": 0.4, "accel": 8.0, "dims": (0.128, 0.086, 0.04), "pad": (0.015, 0.004, 0.012)})],
        "clamp": [("analytic", {"case": "squeeze_lift", "mass": 80.0, "mu": 0.5, "accel": 1.5, "dims": (0.8, 0.7, 1.2), "pad": (0.35, 0.02, 0.4), "hand": 3000.0}),
                  ("analytic", {"case": "squeeze_brake", "mass": 80.0, "mu": 0.5, "accel": 3.0, "dims": (0.8, 0.7, 1.2), "pad": (0.35, 0.02, 0.4), "hand": 3000.0})],
        "palms": [("analytic", {"case": "squeeze_lift", "mass": 6.0, "mu": 0.6, "accel": 2.0, "dims": (0.35, 0.30, 0.30), "pad": (0.06, 0.012, 0.07)}),
                  ("analytic", {"case": "squeeze_walk", "mass": 6.0, "mu": 0.6, "rise": 0.05, "hz": 1.8, "dims": (0.35, 0.30, 0.30), "pad": (0.06, 0.012, 0.07)})],
        "table": [("analytic", {"case": "table_edge", "dims": (0.30, 0.20, 0.15)}), ("analytic", {"case": "table_edge", "dims": (0.40, 0.25, 0.10), "cx": 0.1})],
    }
    for scene in names:
        out += cases.get(scene, [])
        if scene in ("reach", "cabinet"):
            from robo_jev.mjgen.scenes import SCENES as REG

            for visible, mass in REG[scene].sanity_cases():
                out.append(("analytic", {"case": f"{scene}_tip", "visible": visible, "mover_mass": mass}))
    return out


def summarize(job_list: list[tuple[str, dict[str, Any]]], results: list[Any]) -> dict[str, Any]:
    """Results → report with a verdict: zero label-level monotonicity violations; analytic ratios within tolerance (slide and slip
    0.95–1.06 — a short push needs a little more than the static threshold to move an item 2–3 cm; tip 0.90–1.06; the walking bob
    0.85–1.05 — a brief peak needs a little less than its static load); table edge and tip-over |found − expected| within 8 mm and 25 mm;
    a bin wall right ahead keeps the item upright; identical determinism hashes (twice in a worker and once in this process)."""
    mono = [r for (kind, _), r in zip(job_list, results) if kind == "monotone"]
    anal = [r for (kind, _), r in zip(job_list, results) if kind == "analytic"]
    determ = [r for (kind, _), r in zip(job_list, results) if kind == "determinism"]
    for entry, (kind, payload) in zip(determ, [(k, p) for k, p in job_list if k == "determinism"]):
        here = determinism(**payload)
        entry["hash_here"] = here["hash_1"]
        entry["identical"] = entry["hash_1"] == entry["hash_2"] == entry["hash_here"]

    def within(entry: dict[str, Any]) -> bool:
        case = entry["case"]
        if "passed" in entry:
            return bool(entry["passed"])
        if case.startswith("table_edge"):
            return abs(entry["diff_m"]) < 0.008
        if case.startswith(("reach_tip", "cabinet_tip")):
            return abs(entry["diff_m"]) < 0.025
        if "slide" in case or "squeeze_lift" in case or "squeeze_brake" in case:
            return 0.95 <= entry["ratio"] <= 1.06
        if "squeeze_walk" in case:
            return 0.85 <= entry["ratio"] <= 1.05
        return 0.90 <= entry["ratio"] <= 1.06

    for entry in anal:
        entry["within_tolerance"] = within(entry)
    by_sweep: dict[str, Any] = {}
    for entry in mono:
        row = by_sweep.setdefault(f"{entry['scene']}/{entry['sweep']}", {"moments": 0, "label_violations": 0, "draw_violations": 0, "draw_steps": 0})
        row["moments"] += 1
        row["label_violations"] += entry["label_violations"]
        row["draw_violations"] += entry["draw_violations"]
        row["draw_steps"] += entry["draws"] * max(0, len(entry["x"]) - 1)
    verdict = {"monotone_label_level": all(row["label_violations"] == 0 for row in by_sweep.values()),
               "analytic": all(e["within_tolerance"] for e in anal), "deterministic": all(e["identical"] for e in determ)}
    verdict["passed"] = all(verdict.values())
    return {"kind": "v04a-sanity", "verdict": verdict, "monotonicity": {"by_sweep": by_sweep, "moments": mono}, "analytic": anal, "determinism": determ}


# --------------------------------------------------------------------------
# One-factor sweeps: label sensitivity to each physical factor
# --------------------------------------------------------------------------


def factor_sweep(scene: str, index: int, master: int, k: int) -> dict[str, Any]:
    """One candidate scene: every factor of the scene swept on its own (the others as stated), the key-event rate per level on the
    scene's base-prior draws (re-mapped when a word changes the prior). Word and choice factors take every alternative; number factors
    take × 0.6, 0.8, 1, 1.25, 1.5 of the stated value (clipped to the factor's span)."""
    spec = B.spec(scene)
    group = B.candidate(scene, index, master, k=k, pair_share=0.0)
    rigs_cache: dict[str, Any] = {}
    out: dict[str, Any] = {"scene": scene, "index": index, "factors": {}}
    v0 = group.visible
    for factor, entry in spec.factors.items():
        field = entry["field"]
        current = fam._get_field(v0, field)
        if entry["how"] in ("word", "choice"):
            levels = [current] + [c for c in spec.factor_choices(group, factor, current) if c != current]
        else:
            lo, hi, step = entry["span"](current, v0)
            levels = sorted({fam._round(min(hi, max(lo, current * m)), step) for m in (0.6, 0.8, 1.0, 1.25, 1.5)})
        rows = []
        for level in levels:
            v = json.loads(json.dumps(v0))
            fam._set_field(v, field, level)
            if not spec.accept(v):
                continue
            draws = B._draws(spec, v, group.seed ^ 0xD3A5, k)
            events = [spec.roll(v, d, rigs_cache)["outcome"] in spec.event for d in draws]
            rows.append({"level": level, "rate": sum(events) / len(events)})
        rates = [r["rate"] for r in rows]
        out["factors"][factor] = {"levels": rows, "range": (max(rates) - min(rates)) if rates else 0.0}
    return out


def summarize_sweeps(results: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for scene in sorted({r["scene"] for r in results}):
        rows = [r for r in results if r["scene"] == scene]
        out[scene] = {}
        for factor in rows[0]["factors"]:
            ranges = [r["factors"][factor]["range"] for r in rows if factor in r["factors"]]
            out[scene][factor] = {"scenes": len(ranges), "range_mean": sum(ranges) / len(ranges), "range_median": sorted(ranges)[len(ranges) // 2],
                                  "moves_ge_0.1": sum(1 for x in ranges if x >= 0.1) / len(ranges)}
            if B.spec(scene).factors[factor]["how"] == "word":
                per_level: dict[str, list[float]] = {}
                for r in rows:
                    for level in r["factors"][factor]["levels"]:
                        per_level.setdefault(str(level["level"]), []).append(level["rate"])
                out[scene][factor]["mean_rate_by_level"] = {k: round(sum(v) / len(v), 3) for k, v in sorted(per_level.items())}
    return out
