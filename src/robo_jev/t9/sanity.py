"""T9 physics sanity (Task T9 §1-2) — the report behind ``scripts/t9.py sanity`` (MuJoCo only; no labels or models).

**Monotonicity.** Candidate moments of each family (built with the calculator, as for T9 but from another master seed) have one
number swept with the moment's own K base-prior draws held fixed (common random numbers); the key-event share along the sweep must
never move against the physics (label level), and single draws that flip back are counted. Required sweeps (the physics guarantees
the order):

* roll — a steeper tilt never lowers the roll-over share; a taller lip never raises it;
* swing — a faster stop never lowers the touch share (and never lowers a draw's swing reach); a faster move never lowers it;
* knock — a faster strike never lowers toppling; a narrower base (same height, weight layout and strike height) never lowers it;
* conveyor — a longer pulse (higher belt speed at the same rate) never lowers toppling;
* overhang — a board sticking out further, or an item set further out, never lowers the falling share;
* suction — a longer pulse (higher carry speed at the same rate) never lowers the loss share.

Reported, not required: knock strike height (a low strike slides the base instead of rotating it, so higher usually topples more —
but not as a law), conveyor rate and suction rate at a fixed speed change (a very abrupt, very short pulse can let the belt slip under
the item or end before the panel slips far — the share need not rise with the rate).

**Analytic checks** (bisection in MuJoCo against a closed form):

* roll — rolling acceleration down an incline (no rolling resistance) against g·sin θ/(1 + k): solid cylinder (2/3), thin tube (1/2),
  solid ball (5/7); the rolling-resistance angle (rolls iff tan θ > μ_r/r);
* swing — the swing reach of a compact load after a near-instant stop against the physical pendulum v·√(ℓ/g)·√(mℓ²/I_hook)
  (→ v·√(L/g) for a point mass);
* knock — the strike height that separates sliding from toppling for a struck block on a slippery surface against the impulsive
  no-slip height h* = 2(b² + H²)/(3(H + μb)); the toppling speed on a high-friction surface against ½·I_P·(v/h)² = m·g·Δz;
* conveyor — tipping onset under a long pulse against a* = g·(b/2)/h_c, sliding onset against μ·g;
* overhang — the overhang at which the board tips against o* = (m_i·p + M·L/2)/(m_i + M) (item centre p in from the end; the item
  set down 0.3 mm above the board, so the static rule applies — T9's 5 mm release adds an impact that tips a little earlier);
* suction — slip onset (centred cup, long pulse) against a* = μ·(F − m·g)/m.

**Determinism.** One group per family labelled twice in one process (fresh scenes each time) and once in a fresh process: identical
outcome codes.
"""

from __future__ import annotations

import hashlib
import json
import math
import multiprocessing as mp
import random
import subprocess
import sys
from typing import Any

from robo_jev.t9 import physics as ph
from robo_jev.t9 import priors as P
from robo_jev.t9.families import FAMILIES, KNOCK_OBJECTS

__all__ = ["ANALYTIC", "SWEEPS", "analytic", "determinism", "monotone", "run_all"]

G = ph.G
#: name → (family, field, kind, values or factors, expected sign of the key share along the sweep, required)
SWEEPS: dict[str, tuple[str, str, str, tuple[float, ...], int, bool]] = {
    "roll_tilt": ("roll", "tilt", "scale", (0.6, 0.8, 1.0, 1.25, 1.5), +1, True),
    "roll_lip": ("roll", "lip", "scale", (0.5, 0.75, 1.0, 1.25, 1.5), -1, True),
    "swing_decel": ("swing", "decel", "abs", (1.0, 2.0, 4.0, 8.0, 16.0), +1, True),
    "swing_speed": ("swing", "speed", "scale", (0.6, 0.8, 1.0, 1.25, 1.5), +1, True),
    "knock_speed": ("knock", "speed", "scale", (0.5, 0.75, 1.0, 1.5, 2.0), +1, True),
    "knock_narrower": ("knock", "depth", "scale", (1.4, 1.2, 1.0, 0.8, 0.6), +1, True),
    "knock_height": ("knock", "height", "frac", (0.3, 0.45, 0.6, 0.75, 0.9), +1, False),
    "conveyor_longer": ("conveyor", "speed", "scale", (0.5, 0.75, 1.0, 1.5, 2.0), +1, True),
    "conveyor_rate": ("conveyor", "rate", "scale", (0.5, 0.75, 1.0, 1.5, 2.0), +1, False),
    "overhang_over": ("overhang", "over", "add", (-0.06, -0.03, 0.0, 0.03, 0.06), +1, True),
    "overhang_out": ("overhang", "inset", "add", (0.06, 0.03, 0.0, -0.03, -0.06), +1, True),
    "suction_longer": ("suction", "speed", "scale", (0.5, 0.75, 1.0, 1.5, 2.0), +1, True),
    "suction_rate": ("suction", "rate", "scale", (0.5, 0.75, 1.0, 1.5, 2.0), +1, False),
}


def _sample(family_name: str, index: int, master: int, k: int) -> tuple[dict[str, Any], list[dict[str, float]]]:
    from robo_jev.t9 import build as B

    group = B.candidate(family_name, index, master, k=k)
    fam = FAMILIES[family_name]
    units = P.unit_draws(fam.hidden_names(), k, int(group["draws_seed"]))
    vis = group["base"]
    return vis, P.draws_for(fam.words(vis), fam.fixed(vis), "base", units, fam.hidden_names())


class _KnockDepth:
    """knock with the object's depth scaled (narrower or wider base; height, mass, weight layout and strike height unchanged)."""

    def __init__(self, vis: dict[str, Any], factor: float):
        dims = list(KNOCK_OBJECTS[vis["object"]]["dims"])
        dims[0] *= factor
        self.scene = ph.make_scene("knock", {"dims": tuple(dims), "mass": vis["mass"]})


def monotone(name: str, index: int, master: int, k: int) -> dict[str, Any]:
    family_name, field, kind, steps, sign, required = SWEEPS[name]
    fam = FAMILIES[family_name]
    vis, draws = _sample(family_name, index, master, k)
    rows = []
    for step in steps:
        v = dict(vis)
        if field == "depth":
            scene = _KnockDepth(vis, step).scene
        else:
            if kind == "scale":
                v[field] = vis[field] * step
            elif kind == "abs":
                v[field] = step
            elif kind == "frac":
                v[field] = KNOCK_OBJECTS[vis["object"]]["dims"][2] * step
            else:
                v[field] = vis[field] + step
            if field == "tilt":
                v[field] = min(v[field], 32.0)
            if field == "lip":
                v[field] = min(max(v[field], 0.003), 0.8 * vis["radius"])
            scene = ph.make_scene(family_name, fam.scene(v))
        outcomes = [scene.run(fam.action(v), fam.physics_draw(v, d)) for d in draws]
        rows.append({"x": step, "events": [o["outcome"] in fam.key for o in outcomes], "unstable": sum(bool(o.get("unstable")) for o in outcomes),
                     "reach": [o.get("reach") for o in outcomes] if family_name == "swing" else None})
    shares = [sum(r["events"]) / len(r["events"]) for r in rows]
    label_violations = sum(1 for a, b in zip(shares, shares[1:]) if (b - a) * sign < -1e-12)
    draw_flips = 0
    for j in range(len(draws)):
        seq = [r["events"][j] for r in rows]
        draw_flips += sum(1 for a, b in zip(seq, seq[1:]) if (int(b) - int(a)) * sign < 0)
    out = {"sweep": name, "family": family_name, "index": index, "shares": shares, "label_violations": label_violations, "draw_flips": draw_flips,
           "steps": len(rows) - 1, "draws": len(draws), "unstable": sum(r["unstable"] for r in rows), "required": required}
    if family_name == "swing" and field == "decel":
        reach_drops = 0
        for j in range(len(draws)):
            seq = [r["reach"][j] for r in rows]
            reach_drops += sum(1 for a, b in zip(seq, seq[1:]) if b < a - 0.002)
        out["reach_drops"] = reach_drops
    return out


# --------------------------------------------------------------------------
# Analytic checks
# --------------------------------------------------------------------------


def _bisect(fn: Any, lo: float, hi: float, steps: int = 14) -> float:
    """Smallest x in [lo, hi] with fn(x) True (fn monotone False → True)."""
    for _ in range(steps):
        mid = (lo + hi) / 2
        if fn(mid):
            hi = mid
        else:
            lo = mid
    return (lo + hi) / 2


def _roll_accel(k: float, shape: str) -> dict[str, Any]:
    vis = {"shape": shape, "radius": 0.04, "length": 0.2, "mass": 0.5, "lip": 0.004, "dist": 3.0}
    scene = ph.make_scene("roll", vis)
    theta = 10.0
    scene.prepare({"k": k, "mu": 0.9, "mu_roll": 0.0})
    model, data = scene.model, scene.data
    th = math.radians(theta)
    model.opt.gravity[:] = (G * math.sin(th), 0.0, -G * math.cos(th))
    item = scene.body("item")
    mujoco_step = __import__("mujoco").mj_step
    for _ in range(100):
        mujoco_step(model, data)
    v0, t0 = float(data.qvel[0]), data.time
    for _ in range(500):
        mujoco_step(model, data)
    a = (float(data.qvel[0]) - v0) / (data.time - t0)
    model.opt.gravity[:] = (0.0, 0.0, -G)
    expected = G * math.sin(th) / (1 + k)
    return {"check": f"roll acceleration, k = {k} ({shape})", "mujoco": a, "closed_form": expected, "ratio": a / expected,
            "pass": abs(a / expected - 1) < 0.03, "item_x": float(data.xpos[item][0])}


def _roll_threshold() -> dict[str, Any]:
    r, mu_roll = 0.04, 0.002
    vis = {"shape": "capsule", "radius": r, "length": 0.2, "mass": 0.5, "lip": 0.004, "dist": 0.5}
    scene = ph.make_scene("roll", vis)

    def rolls(tilt: float) -> bool:
        out = scene.run({"tilt": tilt, "ramp": 0.2, "hold": 3.0}, {"k": 0.5, "mu": 0.9, "mu_roll": mu_roll})
        return out["outcome"] != "put"

    found = _bisect(rolls, 0.5, 8.0)
    expected = math.degrees(math.atan(mu_roll / r))
    return {"check": "rolling-resistance angle (rolls iff tan θ > μ_r/r)", "mujoco_deg": found, "closed_form_deg": expected, "ratio": found / expected,
            "pass": abs(found / expected - 1) < 0.05}


def _swing_reach(length: float, speed: float) -> dict[str, Any]:
    dims, mass = (0.06, 0.06, 0.06), 3.0
    vis = {"dims": dims, "mass": mass, "strap": length, "gap": 5.0}
    scene = ph.make_scene("swing", vis)
    out = scene.run({"speed": speed, "decel": 80.0, "ramp": 0.002}, {"stretch": 0.0, "slop": 0.0, "com": 0.5})
    ell = length + 0.5 * dims[2]
    inertia = mass * ell * ell + mass * (dims[0] ** 2 + dims[2] ** 2) / 12
    phys = speed * math.sqrt(ell / G) * math.sqrt(mass * ell * ell / inertia)
    point = speed * math.sqrt(ell / G)
    # the reach is the bottom front edge's excursion: the box tilts with the strap, its corner sits (L + H) from the hook
    corner = phys * (length + dims[2]) / ell
    return {"check": f"swing reach after a near-instant stop (L = {length} m, v = {speed} m/s)", "mujoco_reach": out["reach"],
            "closed_form_corner": corner, "point_mass_com": point, "ratio": out["reach"] / corner, "pass": abs(out["reach"] / corner - 1) < 0.08}


def _knock_slide_topple(mu: float) -> dict[str, Any]:
    b, w, height, mass = 0.10, 0.10, 0.30, 1.0
    scene = ph.make_scene("knock", {"dims": (b, w, height), "mass": mass})
    speed = 1.2

    def topples(h: float) -> bool:
        return scene.run({"speed": speed, "height": h, "stop": 0.003}, {"mu": mu, "hc": 0.5, "cx": 0.0})["outcome"] == "topple"

    found = _bisect(topples, 0.03, 0.29, steps=12)
    expected = 2 * (b * b + height * height) / (3 * (height + mu * b))
    return {"check": f"knock: slide-to-topple strike height, μ = {mu}", "mujoco_m": found, "closed_form_m": expected, "ratio": found / expected,
            "pass": abs(found / expected - 1) < 0.15}


def _knock_speed() -> dict[str, Any]:
    b, w, height, mass, h = 0.08, 0.08, 0.30, 1.0, 0.24
    scene = ph.make_scene("knock", {"dims": (b, w, height), "mass": mass})

    def topples(v: float) -> bool:
        return scene.run({"speed": v, "height": h, "stop": 0.002}, {"mu": 1.4, "hc": 0.5, "cx": 0.0})["outcome"] == "topple"

    found = _bisect(topples, 0.02, 2.0)
    dx, dz = b / 2, height / 2
    ip = mass * (b * b + height * height) / 12 + mass * (dx * dx + dz * dz)
    expected = h * math.sqrt(2 * mass * G * (math.hypot(dx, dz) - dz) / ip)
    return {"check": "knock: toppling speed on a high-friction surface vs ½·I_P·(v/h)² = m·g·Δz", "mujoco": found, "closed_form": expected,
            "ratio": found / expected, "pass": abs(found / expected - 1) < 0.25}


def _conveyor_onsets() -> list[dict[str, Any]]:
    out = []
    for dims, hc in (((0.08, 0.10, 0.30), 0.5), ((0.12, 0.12, 0.30), 0.4)):
        scene = ph.make_scene("conveyor", {"dims": dims, "mass": 1.0})

        def tips(a: float) -> bool:
            return scene.run({"speed": 6.0, "rate": a, "ramp": 0.05}, {"mu": 1.4, "hc": hc, "cx": 0.0})["outcome"] == "topple"

        found = _bisect(tips, 0.2, 9.0)
        expected = G * (dims[0] / 2) / (hc * dims[2])
        out.append({"check": f"conveyor: tipping onset, base {dims[0]} m, centre of mass {hc * dims[2]:.3f} m high", "mujoco": found, "closed_form": expected,
                    "ratio": found / expected, "pass": abs(found / expected - 1) < 0.05})
    scene = ph.make_scene("conveyor", {"dims": (0.30, 0.30, 0.06), "mass": 1.0})
    for mu in (0.2, 0.4):
        def slides(a: float, mu: float = mu) -> bool:
            return scene.run({"speed": 6.0, "rate": a, "ramp": 0.05}, {"mu": mu, "hc": 0.5, "cx": 0.0})["outcome"] != "stay"

        found = _bisect(slides, 0.2, 9.0)
        out.append({"check": f"conveyor: sliding onset, μ = {mu}", "mujoco": found, "closed_form": mu * G, "ratio": found / (mu * G),
                    "pass": abs(found / (mu * G) - 1) < 0.05})
    return out


def _overhang_tip() -> list[dict[str, Any]]:
    out = []
    for board_mass, item_mass, inset in ((1.6, 1.0, 0.06), (2.4, 2.0, 0.10)):
        board, item = (0.60, 0.25, 0.018), (0.10, 0.10, 0.12)
        scene = ph.make_scene("overhang", {"board": board, "item_dims": item, "drop": 0.0003})  # set down gently: the static rule

        def tips(over: float) -> bool:
            return scene.run({"over": over, "inset": inset}, {"board_mass": board_mass, "item_mass": item_mass, "icx": 0.0, "err": 0.0})["outcome"] != "stays"

        found = _bisect(tips, 0.05, 0.29)
        expected = (item_mass * inset + board_mass * board[0] / 2) / (item_mass + board_mass)
        out.append({"check": f"overhang: tipping overhang (M = {board_mass} kg, m = {item_mass} kg, p = {inset} m)", "mujoco_m": found,
                    "closed_form_m": expected, "ratio": found / expected, "pass": abs(found - expected) < 0.005})
    return out


def _suction_slip() -> list[dict[str, Any]]:
    out = []
    for vacuum, mu, mass in ((40000.0, 0.5, 2.4), (55000.0, 0.3, 1.5)):
        scene = ph.make_scene("suction", {"panel": (0.40, 0.30, 0.008), "cup": 0.04, "offset": 0.0})

        def slips(a: float, vacuum: float = vacuum, mu: float = mu, mass: float = mass) -> bool:
            return scene.run({"rate": a, "speed": 3.0, "ramp": 0.03, "carry": 6.0}, {"vacuum": vacuum, "mu": mu, "mass": mass})["outcome"] != "hold"

        found = _bisect(slips, 0.5, 30.0)
        force = vacuum * math.pi * 0.02**2
        expected = mu * (force - mass * G) / mass
        out.append({"check": f"suction: slip onset (ΔP = {vacuum:.0f} Pa, μ = {mu}, m = {mass} kg)", "mujoco": found, "closed_form": expected,
                    "ratio": found / expected, "pass": abs(found / expected - 1) < 0.08})
    return out


ANALYTIC = ("roll_accel", "roll_threshold", "swing_reach", "knock_slide_topple", "knock_speed", "conveyor", "overhang", "suction")


def analytic(name: str) -> list[dict[str, Any]]:
    if name == "roll_accel":
        return [_roll_accel(0.5, "capsule"), _roll_accel(1.0, "capsule"), _roll_accel(0.4, "sphere")]
    if name == "roll_threshold":
        return [_roll_threshold()]
    if name == "swing_reach":
        return [_swing_reach(0.6, 0.3), _swing_reach(1.0, 0.2), _swing_reach(0.4, 0.4)]
    if name == "knock_slide_topple":
        return [_knock_slide_topple(0.15), _knock_slide_topple(0.3)]
    if name == "knock_speed":
        return [_knock_speed()]
    if name == "conveyor":
        return _conveyor_onsets()
    if name == "overhang":
        return _overhang_tip()
    return _suction_slip()


# --------------------------------------------------------------------------
# Determinism
# --------------------------------------------------------------------------


def label_hash(family_name: str, index: int, master: int, k: int) -> str:
    from robo_jev.t9 import build as B

    group = B.candidate(family_name, index, master, k=k)
    labelled = B.label_group(group)
    codes = {m: {v: {p: labelled["labels"][m][v][p]["codes"] for p in P.PRIORS} for v in labelled["labels"][m]} for m in labelled["labels"]}
    return hashlib.sha256(json.dumps(codes, sort_keys=True).encode()).hexdigest()


def determinism(family_name: str, master: int, k: int = 8) -> dict[str, Any]:
    first = label_hash(family_name, 0, master, k)
    second = label_hash(family_name, 0, master, k)
    code = ("import sys; sys.path.insert(0, %r); from robo_jev.t9.sanity import label_hash; print(label_hash(%r, 0, %d, %d))"
            % (str(__import__("pathlib").Path(__file__).resolve().parents[2]), family_name, master, k))
    other = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout.strip()
    return {"family": family_name, "same_process": first == second, "other_process": first == other, "sha256": first}


# --------------------------------------------------------------------------
# All
# --------------------------------------------------------------------------


def _job(job: tuple[str, Any]) -> Any:
    kind, payload = job
    if kind == "monotone":
        return ("monotone", monotone(*payload))
    if kind == "analytic":
        return ("analytic", analytic(payload))
    return ("determinism", determinism(*payload))


def run_all(*, workers: int, k: int, moments: int, master: int) -> dict[str, Any]:
    jobs: list[tuple[str, Any]] = [("monotone", (name, i, master, k)) for name in SWEEPS for i in range(moments)]
    jobs += [("analytic", name) for name in ANALYTIC]
    jobs += [("determinism", (f, master, 8)) for f in FAMILIES]
    results: dict[str, list[Any]] = {"monotone": [], "analytic": [], "determinism": []}
    with mp.get_context("fork").Pool(workers) as pool:
        for kind, result in pool.imap_unordered(_job, jobs):
            if kind == "analytic":
                results[kind] += result
            else:
                results[kind].append(result)
    by_sweep: dict[str, dict[str, Any]] = {}
    for row in results["monotone"]:
        entry = by_sweep.setdefault(row["sweep"], {"family": row["family"], "required": row["required"], "moments": 0, "label_violations": 0,
                                                   "draw_flips": 0, "draw_steps": 0, "unstable": 0, "reach_drops": 0})
        entry["moments"] += 1
        entry["label_violations"] += row["label_violations"]
        entry["draw_flips"] += row["draw_flips"]
        entry["draw_steps"] += row["steps"] * row["draws"]
        entry["unstable"] += row["unstable"]
        entry["reach_drops"] += row.get("reach_drops", 0)
    verdict = {
        "monotone_required": all(v["label_violations"] == 0 for v in by_sweep.values() if v["required"]),
        "analytic": all(r["pass"] for r in results["analytic"]),
        "determinism": all(r["same_process"] and r["other_process"] for r in results["determinism"]),
    }
    verdict["passed"] = all(verdict.values())
    return {"kind": "t9-sanity", "k": k, "moments_per_sweep": moments, "master": master, "sweeps": {n: dict(SWEEPS[n][:2]) | {"required": SWEEPS[n][5]} for n in SWEEPS},
            "monotone": by_sweep, "monotone_rows": sorted(results["monotone"], key=lambda r: (r["sweep"], r["index"])),
            "analytic": results["analytic"], "determinism": results["determinism"], "verdict": verdict}
