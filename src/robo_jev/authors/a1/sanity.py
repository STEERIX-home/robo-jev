"""Author A1 — physics sanity (MuJoCo; run inside a capped unit): monotone sweeps at the label level, one analytic threshold per
law, and determinism.

* **Monotone sweeps** — for sampled plans, the K = 32 shared base-prior draws are rolled out along a grid of one quantity with
  everything else fixed; the key-event share must never move against physics (a *label-level violation* is any step in the
  wrong direction; per-draw flips are reported too). Mast: deceleration ↑ ⇒ key ↑; friction ↑ ⇒ key ↓ (and the tip-up share ↑:
  a slippery floor skids instead). Dock: overhang ↑ ⇒ key ↑; centre of mass toward the lead end ↑ ⇒ key ↑. Ramp: grade ↑ ⇒ key ↑;
  payload ↑ ⇒ key ↑; friction ↑ ⇒ key ↓.
* **Analytic thresholds** — MuJoCo bisection against the closed form: sliding (mast front-wheel lock and the overrun it causes;
  ramp runaway at tan θ = μ·M_tug / (M_tug + m_cart)), tipping (mast rear-caster lift-off onset at a = g·d / h), centre of mass
  past the support (dock tips once overhang = L/2 − c_x) and the friction limit against m·a (ramp stopping distance v² / 2a_net).
* **Determinism** — the same rollout twice (and its MJCF string) is identical; cached rollouts re-run identically.
"""

from __future__ import annotations

import math
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from robo_jev.authors.a1 import build, sims
from robo_jev.authors.a1.scenes import G, SCENES, lhs


def _key(scene: str, outcome: str) -> int:
    return int(outcome in SCENES[scene].key_events)


def _sweep(scene: str, visible: Mapping[str, Any], hidden_rows: Sequence[Mapping[str, float]],
           make: Callable[[float], tuple[dict[str, Any], list[dict[str, float]]]], grid: Sequence[float], sign: int) -> dict[str, Any]:
    """Key share along `grid` (make(x) → (visible, hidden rows)); `sign` +1 = must not fall, −1 = must not rise."""
    sc = SCENES[scene]
    per_draw = []
    for x in grid:
        v, rows = make(x)
        per_draw.append([_key(scene, sims.rollout(scene, sc.spec(v), h, sc.state(v))["outcome"]) for h in rows])
    arr = np.array(per_draw)  # grid × draws
    shares = arr.mean(1)
    steps = np.diff(shares) * sign
    flips = int(((np.diff(arr, axis=0) * sign) < 0).sum())
    return {"grid": [float(g) for g in grid], "shares": [round(float(s), 4) for s in shares], "label_violations": int((steps < -1e-12).sum()),
            "draw_flips": flips}


def monotone_sweeps(n_plans: int = 6, points: int = 8, start: int = 50_000) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for scene, sc in SCENES.items():
        results: dict[str, list[Any]] = {}
        for i in range(start, start + n_plans):
            plan = build.plan_group(scene, i)
            visible = dict(plan["visible"])
            u = lhs(build.K, len(sc.hidden_names), plan["draws_seed"])
            rows = sc.hidden(visible, "base", u)
            lo, hi = sc.bounds(visible)
            grid = [sc.quantize(lo + (hi - lo) * k / (points - 1)) for k in range(points)]

            def decisive(x: float, visible=visible, rows=rows) -> tuple[dict[str, Any], list[dict[str, float]]]:
                v = dict(visible)
                v[sc.decisive] = x
                return v, rows

            results.setdefault(f"{sc.decisive}_up", []).append(_sweep(scene, visible, rows, decisive, grid, +1))
            scales = [0.6 + 0.8 * k / (points - 1) for k in range(points)]

            def scaled(name: str) -> Callable[[float], tuple[dict[str, Any], list[dict[str, float]]]]:
                def make(s: float, visible=visible, rows=rows) -> tuple[dict[str, Any], list[dict[str, float]]]:
                    return dict(visible), [dict(h, **{name: h[name] * s}) for h in rows]
                return make

            if scene in ("mast", "ramp"):
                results.setdefault("friction_up", []).append(_sweep(scene, visible, rows, scaled("mu"), scales, -1))
            if scene == "ramp":
                results.setdefault("payload_up", []).append(_sweep(scene, visible, rows, scaled("payload"), scales, +1))
            if scene == "dock":
                shifts = [-0.15 + 0.30 * k / (points - 1) for k in range(points)]

                def shifted(dx: float, visible=visible, rows=rows) -> tuple[dict[str, Any], list[dict[str, float]]]:
                    return dict(visible), [dict(h, cg_x=h["cg_x"] + dx) for h in rows]

                results.setdefault("cg_toward_lead_up", []).append(_sweep(scene, visible, rows, shifted, shifts, +1))
        out[scene] = {name: {"sweeps": len(v), "label_violations": sum(s["label_violations"] for s in v),
                             "draw_flips": sum(s["draw_flips"] for s in v), "examples": v[:2]} for name, v in results.items()}
    return out


def _bisect(f: Callable[[float], bool], lo: float, hi: float, iters: int = 14) -> float:
    """Smallest x in [lo, hi] with f(x) True, assuming f is monotone (False below, True above)."""
    for _ in range(iters):
        mid = (lo + hi) / 2
        if f(mid):
            hi = mid
        else:
            lo = mid
    return (lo + hi) / 2


def analytic_thresholds() -> dict[str, Any]:
    out: dict[str, Any] = {}
    # mast — sliding: front-wheel lock limits the deceleration to a_lock; the stop then overruns by v²/2·(1/a_lock − 1/a).
    unit = {"wheelbase": 0.58, "track": 0.46, "base_mass": 68.0, "base_cg_h": 0.18}
    rows = []
    for mu in (0.20, 0.30, 0.40):
        spec = {**unit, "lift_h": 0.70, "tote_dims": [0.4, 0.3, 0.25]}
        hidden = {"mu": mu, "tote_mass": 5.0, "tote_x": 0.0}
        props = sims.mast_mass_properties(spec, hidden)
        wb, h, xc = unit["wheelbase"], props["cg_h"], props["cg_x"]
        a_lock = mu * G * (wb / 2 + xc) / wb / (1 - mu * h / wb)
        v = 1.6
        a_skid = 1.0 / (1.0 / a_lock - 2 * sims.SKID_OVERRUN / v**2)
        sim = _bisect(lambda a: sims.rollout("mast", spec, hidden, {"speed": v, "decel": a})["outcome"] == "skid", 0.3, 8.0)
        rows.append({"mu": mu, "cg_h": round(h, 3), "analytic_lock": round(a_lock, 3), "analytic_skid": round(a_skid, 3), "mujoco_skid": round(sim, 3),
                     "ratio": round(sim / a_skid, 3)})
    out["mast_sliding"] = rows
    # mast — tipping: rear casters start to lift once a·h > g·d (onset: pitch > 0.5° on a long, grippy stop).
    rows = []
    for lift, mass in ((1.6, 20.0), (2.2, 30.0), (2.4, 38.0)):
        spec = {**unit, "lift_h": lift, "tote_dims": [0.6, 0.4, 0.32]}
        hidden = {"mu": 1.2, "tote_mass": mass, "tote_x": 0.0}
        props = sims.mast_mass_properties(spec, hidden)
        d = unit["wheelbase"] / 2 - props["cg_x"]
        a_tip = G * d / props["cg_h"]
        sim = _bisect(lambda a: sims.rollout("mast", spec, hidden, {"speed": 2.5, "decel": a})["max_tilt"] > 0.5, 0.5, 9.0)
        rows.append({"lift_h": lift, "tote_mass": mass, "cg_h": round(props["cg_h"], 3), "analytic_lift_onset": round(a_tip, 3),
                     "mujoco_onset": round(sim, 3), "ratio": round(sim / a_tip, 3)})
    out["mast_tipping"] = rows
    # dock — centre of mass past the edge: tips once overhang + c_x > L/2.
    rows = []
    for length, cx in ((1.2, 0.0), (1.2, 0.10), (0.8, -0.06), (1.0, 0.05)):
        spec = {"step_h": 0.35, "pallet_dims": [length, 0.8, 1.4], "mass": 500.0}
        hidden = {"mu": 0.5, "cg_x": cx, "cg_z": 0.45, "place": 0.0}
        sim = _bisect(lambda o: sims.rollout("dock", spec, hidden, {"overhang": o})["outcome"] != "level", 0.0, length - 0.05)
        rows.append({"length": length, "cg_x": cx, "analytic": round(length / 2 - cx, 4), "mujoco": round(sim, 4), "error_m": round(sim - (length / 2 - cx), 4)})
    out["dock_com_past_edge"] = rows
    # ramp — friction limit: runaway once tan θ > μ·M / (M + m); stopping distance v² / 2a_net below it.
    rows = []
    for mu, payload in ((0.4, 300.0), (0.7, 500.0), (0.25, 100.0)):
        spec = {"tug_mass": 330.0, "cart_tare": 70.0}
        hidden = {"mu": mu, "payload": payload}
        m_cart = spec["cart_tare"] + payload
        grade_star = 100 * mu * spec["tug_mass"] / (spec["tug_mass"] + m_cart)
        sim = _bisect(lambda g: sims.rollout("ramp", spec, hidden, {"grade": g, "speed": 0.05})["outcome"] == "runaway", 0.0, 60.0)
        theta = math.atan(0.5 * grade_star / 100)
        a_net = G * (mu * spec["tug_mass"] * math.cos(theta) / (spec["tug_mass"] + m_cart) - math.sin(theta))
        stop = sims.rollout("ramp", spec, hidden, {"grade": 0.5 * grade_star, "speed": 0.6})
        rows.append({"mu": mu, "cart_mass": m_cart, "analytic_runaway_grade": round(grade_star, 3), "mujoco_runaway_grade": round(sim, 3),
                     "ratio": round(sim / grade_star, 3), "stop_at_half_grade_analytic_m": round(0.36 / (2 * a_net), 4),
                     "stop_at_half_grade_mujoco_m": round(stop["travel"], 4)})
    out["ramp_friction_limit"] = rows
    return out


def determinism(n: int = 3) -> dict[str, Any]:
    same = total = 0
    xml_same = True
    for scene, sc in SCENES.items():
        for i in range(n):
            plan = build.plan_group(scene, 60_000 + i)
            v = plan["visible"]
            u = lhs(build.K, len(sc.hidden_names), plan["draws_seed"])
            for h in sc.hidden(v, "alt_a", u)[:4]:
                a = sims.rollout(scene, sc.spec(v), h, sc.state(v))
                b = sims.rollout(scene, sc.spec(v), h, sc.state(v))
                xml_same &= sims.scene_xml(scene, sc.spec(v), h, sc.state(v)) == sims.scene_xml(scene, sc.spec(v), h, sc.state(v))
                total += 1
                same += int(a == b)
    return {"rollouts": total, "identical": same, "xml_identical": bool(xml_same)}


def cached_rerun(root, per_scene: int = 2) -> dict[str, Any]:
    """Re-run the planned / base-prior rollouts of a few cached groups and compare with the cache."""
    same = total = 0
    for scene, sc in SCENES.items():
        for path in sorted((root / scene).glob("*.json"))[:per_scene]:
            import json

            cached = json.loads(path.read_text())
            plan = cached["plan"]
            u = lhs(build.K, len(sc.hidden_names), plan["draws_seed"])
            v = plan["visible"]
            outs = [sims.rollout(scene, sc.spec(v), h, sc.state(v))["outcome"] for h in sc.hidden(v, "base", u)]
            total += len(outs)
            same += sum(int(a == b) for a, b in zip(outs, cached["members"]["base"]["planned"]["base"]))
    return {"rollouts": total, "identical": same}
