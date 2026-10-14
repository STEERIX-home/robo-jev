#!/usr/bin/env python
"""Physics audit (2026-10-14): a hidden centre of mass written into a compiled MuJoCo model at run time (``body_ipos``) leaves the
body's stored bounding volumes (``bvh_aabb``, kept in the compile-time inertial frame) behind; MuJoCo's midphase reads them. CPU only;
run every command in a capped unit (``systemd-run --user … -p MemoryMax=… choom -n 1000``), at most 4 workers. Nothing outside
artifacts/physaudit/ is written; no stored dataset is read for anything but comparison.

    python scripts/physaudit.py repro                      # minimal reproduction and candidate fixes → repro.json
    python scripts/physaudit.py exposure                   # which scene builders collide a moved body through the midphase → exposure.json
    python scripts/physaudit.py t8b --workers 4            # the 480 stored T8b moments (original prior) + stack under B2's alt priors
    python scripts/physaudit.py t8b-candidates --workers 4 # all 288 stack candidates both ways → what T8b's balanced selection would keep
    python scripts/physaudit.py t8b-draws --workers 4      # per draw: centre-of-mass shift, deepest box–support contact, outcomes
    python scripts/physaudit.py t8b-fixcheck --workers 4   # refresh_bvh vs compiled centre of mass vs own support body vs midphase off
    python scripts/physaudit.py mjgen --set v03 --family tower --groups 300 --workers 4   # stored v0.3b / B1 / v0.4a groups, refreshed
    python scripts/physaudit.py t9-physics --family overhang roll --workers 4   # T9's physics code on our own random scenes
    python scripts/physaudit.py t9-relabel --family roll overhang --workers 4   # T9's candidate groups (read in code; counts only)
    python scripts/physaudit.py rescore                    # every stored T8b prediction file under stored vs refreshed stack labels
    python scripts/physaudit.py b2                         # B2's rule with the stored and the refreshed stack labels

The "refreshed" labelling is the stored labeller with ``refresh_bvh=True`` (robo_jev.outside.physics.refresh_bvh): the only change to
the compiled model is the moved body's bounding volumes, which then equal those of a model compiled with that centre of mass (tested).
``t8b`` must run before ``rescore`` and ``b2`` (they read its rows).
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Sequence

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

OUT = REPO / "artifacts" / "physaudit"
T8_DIR = REPO / "artifacts" / "t8"
B2_DIR = REPO / "artifacts" / "b2"
T8B_MASTER = 20261012
SETS = {
    "v03": {"candidates": [REPO / "artifacts" / "simgen" / "v03" / "mujoco-candidates.jsonl"], "kind": "v03"},
    "b1": {"candidates": [REPO / "artifacts" / "b1" / "b1-candidates.jsonl"], "kind": "v03"},
    "v04": {"candidates": sorted((REPO / "artifacts" / "simgen" / "v04" / "train" / "candidates").glob("*.jsonl"))
            + sorted((REPO / "artifacts" / "simgen" / "v04" / "holdout" / "candidates").glob("*.jsonl")), "kind": "v04"},
}


def _log(message: str) -> None:
    print(f"[physaudit {dt.datetime.now().strftime('%H:%M:%S')}] {message}", file=sys.stderr, flush=True)


def _git_sha() -> str | None:
    try:
        return subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True).stdout.strip() or None
    except (OSError, subprocess.CalledProcessError):
        return None


def _guard(path: Path) -> Path:
    path = Path(path).resolve()
    if OUT.resolve() not in (path, *path.parents):
        raise SystemExit(f"{path}: the audit writes only under {OUT}")
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _write_json(path: Path, payload: Any) -> None:
    path = _guard(path)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(payload, ensure_ascii=False, indent=1, default=str) + "\n", encoding="utf-8")
    temp.replace(path)


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path = _guard(path)
    temp = path.with_name(path.name + ".tmp")
    with temp.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
    temp.replace(path)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _pool_map(func: Any, jobs: list[Any], workers: int) -> list[Any]:
    workers = max(1, min(int(workers), 4))  # another task labels MuJoCo on this box: never more than 4 workers
    if workers == 1:
        return [func(job) for job in jobs]
    import multiprocessing

    with multiprocessing.get_context("fork").Pool(workers) as pool:
        return pool.map(func, jobs, chunksize=1)


def _meta(**extra: Any) -> dict[str, Any]:
    import mujoco

    return {"created": dt.datetime.now().astimezone().isoformat(timespec="seconds"), "git": _git_sha(), "mujoco": mujoco.__version__, **extra}


# --------------------------------------------------------------------------
# label comparison helpers
# --------------------------------------------------------------------------


def band(value: float) -> str:
    """v0.3b's thirds (robo_jev.mjgen.generate.band)."""
    from robo_jev.mjgen.generate import band as _band

    return _band(float(value))


def side(value: float) -> int:
    """Which side of 0.5 a probability label is on (−1, 0, +1): a yes/no answer flips when this changes."""
    value = float(value)
    return 0 if abs(value - 0.5) < 1e-12 else (1 if value > 0.5 else -1)


def compare_labels(stored: dict[str, Any], fixed: dict[str, Any]) -> list[dict[str, Any]]:
    """Per question: the stored and refreshed values (noul p_yes / score value; choice: probabilities), |Δ|, band and 0.5-side change,
    and for choices the largest |Δ| of a probability and whether the argmax changed."""
    out = []
    for name, a in stored.items():
        b = fixed[name]
        if a["type"] in ("noul", "score"):
            x, y = float(a.get("p_yes", a.get("value"))), float(b.get("p_yes", b.get("value")))
            out.append({"q": name, "type": a["type"], "stored": x, "fixed": y, "abs_delta": abs(y - x), "band_change": band(x) != band(y),
                        "side_flip": side(x) != side(y)})
        else:
            pa, pb = a["probabilities"], b["probabilities"]
            delta = max(abs(float(pb[k]) - float(pa[k])) for k in pa)
            arg_a = max(pa, key=lambda k: (float(pa[k]), k))
            arg_b = max(pb, key=lambda k: (float(pb[k]), k))
            out.append({"q": name, "type": "choice", "abs_delta": delta, "argmax_flip": arg_a != arg_b, "stored_argmax": arg_a, "fixed_argmax": arg_b})
    return out


def summarise_questions(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    binary = [r for r in rows if r["type"] in ("noul", "score")]
    choice = [r for r in rows if r["type"] == "choice"]
    out: dict[str, Any] = {"questions": len(rows), "binary_questions": len(binary)}
    if binary:
        deltas = sorted(r["abs_delta"] for r in binary)
        out.update({"mean_abs_delta": sum(deltas) / len(deltas), "max_abs_delta": deltas[-1],
                    "p90_abs_delta": deltas[int(0.9 * (len(deltas) - 1))],
                    "share_changed": sum(1 for d in deltas if d > 0) / len(deltas),
                    "share_band_change": sum(r["band_change"] for r in binary) / len(binary),
                    "share_side_flip": sum(r["side_flip"] for r in binary) / len(binary),
                    "band_changes": sum(r["band_change"] for r in binary), "side_flips": sum(r["side_flip"] for r in binary)})
    if choice:
        out.update({"choice_questions": len(choice), "choice_mean_max_abs_delta": sum(r["abs_delta"] for r in choice) / len(choice),
                    "choice_share_argmax_flip": sum(r["argmax_flip"] for r in choice) / len(choice)})
    return out


# --------------------------------------------------------------------------
# repro
# --------------------------------------------------------------------------

_REPRO_HEADER = """<mujoco model="repro">
  <option timestep="0.001" gravity="0 0 -9.81" integrator="implicitfast" cone="elliptic" impratio="10">
    <flag multiccd="enable"{extra}/>
  </option>
  <default><geom solref="0.004 1" condim="3" friction="0.5 0.005 0.0001"/></default>
  <worldbody>
"""
_L, _W, _H, _TOP = 0.40, 0.25, 0.20, 0.50


def _repro_xml(ipos: Sequence[float], layout: str, *, midphase: bool = True) -> str:
    """A 3 kg box (0.40 × 0.25 × 0.20 m) resting on a support whose top is 0.50 m up. Layouts: ``world_box_floor`` (support box and
    floor plane both geoms of the world body — T8b's stack), ``world_box`` (support box the world's only geom), ``own_box_floor``
    (support box in its own static body), ``world_plane_floor`` (a plane at the support height plus the floor, both in the world body),
    ``world_two_boxes`` (the support split into two boxes of the world body, no plane)."""
    out = _REPRO_HEADER.format(extra="" if midphase else ' midphase="disable"')
    box = f'<geom name="surface" type="box" pos="0 0 {_TOP / 2}" size="0.4 0.3 {_TOP / 2}" priority="1"/>'
    floor = '<geom name="floor" type="plane" size="3 3 0.1"/>'
    if layout == "world_box_floor":
        out += f"    {floor}\n    {box}\n"
    elif layout == "world_box":
        out += f"    {box}\n"
    elif layout == "own_box_floor":
        out += f"    {floor}\n    <body name=\"support\">{box}</body>\n"
    elif layout == "world_plane_floor":
        out += f'    {floor}\n    <geom name="surface" type="plane" pos="0 0 {_TOP}" size="3 3 0.1" priority="1"/>\n'
    elif layout == "world_two_boxes":
        out += (f'    <geom name="surface" type="box" pos="-0.2 0 {_TOP / 2}" size="0.2 0.3 {_TOP / 2}" priority="1"/>\n'
                f'    <geom name="surface2" type="box" pos="0.2 0 {_TOP / 2}" size="0.2 0.3 {_TOP / 2}" priority="1"/>\n')
    else:
        raise ValueError(layout)
    m = 3.0
    ixx, iyy, izz = m * (_W**2 + _H**2) / 12, m * (_L**2 + _H**2) / 12, m * (_L**2 + _W**2) / 12
    out += (f'    <body name="load" pos="0 0 {_TOP + _H / 2 + 0.0005}">\n      <freejoint/>\n'
            f'      <inertial pos="{ipos[0]!r} {ipos[1]!r} {ipos[2]!r}" mass="{m}" diaginertia="{ixx} {iyy} {izz}"/>\n'
            f'      <geom name="load_geom" type="box" size="{_L / 2} {_W / 2} {_H / 2}"/>\n    </body>\n')
    return out + "  </worldbody>\n</mujoco>\n"


def _repro_model(method: str, com: Sequence[float], layout: str) -> tuple[Any, Any]:
    import mujoco
    import numpy as np

    from robo_jev.outside import physics as ph

    nominal = (0.001, 0.001, 0.001)
    if method == "compiled_com":  # the centre of mass in the XML
        model = mujoco.MjModel.from_xml_string(_repro_xml(com, layout))
        return model, mujoco.MjData(model)
    if method == "mjspec_recompile":  # documented model editing: change the spec, recompile (state kept)
        spec = mujoco.MjSpec.from_string(_repro_xml(nominal, layout))
        model = spec.compile()
        data = mujoco.MjData(model)
        spec.body("load").ipos = list(com)
        model, data = spec.recompile(model, data)
        mujoco.mj_resetData(model, data)
        return model, data
    model = mujoco.MjModel.from_xml_string(_repro_xml(nominal, layout, midphase=method != "midphase_off"))
    data = mujoco.MjData(model)
    load = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "load")
    reference = ph.bvh_reference(model, load)
    model.body_ipos[load] = np.asarray(com, dtype=float)
    if method == "refresh_bvh":
        ph.refresh_bvh(model, load, reference)
    # "runtime" (T8b's set_hidden) and "midphase_off": write body_ipos, mj_setConst, reset
    mujoco.mj_resetData(model, data)
    mujoco.mj_setConst(model, data)
    mujoco.mj_resetData(model, data)
    return model, data


def _repro_run(model: Any, data: Any, seconds: float = 1.0) -> dict[str, Any]:
    import mujoco

    load = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "load")
    mujoco.mj_forward(model, data)
    z0 = float(data.xpos[load][2])
    low, high, first = z0, z0, None
    for i in range(int(round(seconds / model.opt.timestep))):
        mujoco.mj_step(model, data)
        z = float(data.xpos[load][2])
        low, high = min(low, z), max(high, z)
        if first is None and data.ncon > 0:
            first = i + 1
    return {"sink_mm": round((z0 - low) * 1000, 2), "rise_mm": round((high - z0) * 1000, 2), "first_contact_step": first,
            "final_z_mm": round((float(data.xpos[load][2]) - z0) * 1000, 2), "ncon_end": int(data.ncon)}


def _stack_scene_repro(refresh: bool, com: Sequence[float]) -> dict[str, Any]:
    """T8b's own stack scene and run_stack: a box released 5 mm above the support's middle (nothing sticks out), no tilt."""
    from robo_jev.outside import physics as ph

    dims, support = (0.40, 0.25, 0.20), (0.60, 0.35, 0.90)
    scene = ph.Scene(ph.stack_xml(dims, 3.0, support), refresh_bvh=refresh)
    result = ph.run_stack(scene, x_centre=0.0, drop=0.005, tilt0_deg=0.0, mu=0.4, com=com, support_height=support[2], box_height=dims[2])
    return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in result.items()}


def cmd_repro(args: argparse.Namespace) -> int:
    layouts = ("world_box_floor", "world_box", "own_box_floor", "world_plane_floor", "world_two_boxes")
    methods = ("runtime", "refresh_bvh", "compiled_com", "mjspec_recompile", "midphase_off")
    coms = ((0.0, 0.0, 0.0), (0.0, 0.0, 0.005), (0.0, 0.0, 0.01), (0.0, 0.0, 0.02), (0.0, 0.0, 0.04), (0.0, 0.0, -0.02), (0.08, 0.0, 0.0),
            (-0.08, 0.0, 0.0))
    rows = []
    for layout in layouts:
        for com in coms:
            for method in methods:
                rows.append({"layout": layout, "com": list(com), "method": method, **_repro_run(*_repro_model(method, com, layout))})
    stack = [{"com": list(com), "refresh_bvh": refresh, **_stack_scene_repro(refresh, com)}
             for com in ((0.0, 0.0, 0.0), (0.0, 0.0, 0.008), (0.0, 0.0, 0.016), (0.0, 0.0, 0.024), (0.0, 0.0, -0.016))
             for refresh in (False, True)]
    _write_json(Path(args.out), {**_meta(kind="physaudit-repro"), "rows": rows, "t8b_stack_scene": stack})
    for row in rows:
        if row["method"] in ("runtime", "refresh_bvh") and row["com"][2] in (0.02, 0.04):
            _log(f"{row['layout']:18s} com={row['com']} {row['method']:12s} sink {row['sink_mm']:7.2f} mm rise {row['rise_mm']:7.2f} mm "
                 f"first contact step {row['first_contact_step']}")
    for row in stack:
        _log(f"T8b stack scene com={row['com']} refresh={row['refresh_bvh']}: {row}")
    return 0


# --------------------------------------------------------------------------
# exposure
# --------------------------------------------------------------------------


def _pairs_desc(model: Any, pairs: Sequence[tuple[int, int]]) -> list[dict[str, Any]]:
    import mujoco

    out = []
    for body, other in pairs:
        geoms = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or f"geom{g}" for g in range(int(model.body_geomadr[other]),
                                                                                                    int(model.body_geomadr[other] + model.body_geomnum[other]))]
        types = [int(model.geom_type[g]) for g in range(int(model.body_geomadr[other]), int(model.body_geomadr[other] + model.body_geomnum[other]))]
        out.append({"body": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body) or "world",
                    "other": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, other) or "world", "other_geoms": geoms,
                    "other_geom_types": [mujoco.mjtGeom(t).name for t in types], "body_geoms": int(model.body_geomnum[body])})
    return out


def cmd_exposure(args: argparse.Namespace) -> int:
    """Every scene builder whose bodies get a runtime ``body_ipos``: the pairs that go through the midphase, and the size and direction of
    the runtime move (hidden centre of mass − compiled inertial position) the scene can apply."""
    import mujoco

    from robo_jev.mjgen import physics as mph
    from robo_jev.mjgen.scenes import SCENES
    from robo_jev.mjgen.scenes import build as B
    from robo_jev.mjgen.scenes import rigs
    from robo_jev.outside import moments as M
    from robo_jev.outside import physics as ph
    from robo_jev.t9 import families as t9fam
    from robo_jev.t9 import physics as t9ph

    entries = []

    def add(package: str, scene: str, model: Any, moved: Sequence[int], note: str) -> None:
        pairs = ph.midphase_pairs(model, moved)
        entries.append({"package": package, "scene": scene, "moved_bodies": [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b) for b in moved],
                        "world_geoms": int(model.body_geomnum[0]), "midphase_pairs": _pairs_desc(model, pairs), "exposed": bool(pairs), "move": note})

    # T8b (outside): deck, stack, push
    for family, index in (("deck_brake", 0), ("deck_turn", 1), ("stack", 0), ("push", 0)):
        moment = M.candidate(family, index, T8B_MASTER)
        v = moment.visible
        if family.startswith("deck"):
            xml = ph.deck_xml(v["shape"], v["dims"], v["mass"])
        elif family == "stack":
            xml = ph.stack_xml(v["dims"], v["mass"], (v["support_top"], max(0.3, v["dims"][1] + 0.1), v["support_height"]))
        else:
            xml = ph.push_xml(v["dims"], v["mass"], v["push_height"])
        scene = ph.Scene(xml)
        note = {"deck_brake": "cx·L along travel, (kh−0.5)·H up", "deck_turn": "cx·W sideways, (kh−0.5)·H up",
                "stack": "cx·L outward (±0.22 L), (kh−0.5)·H up with kh ∈ (0.42, 0.58): up to +0.08·H, ×1.5 under B2's alt_a",
                "push": "cx·L, (kh−0.5)·H"}[family]
        add("outside (T8b)", family, scene.model, [scene.load], note)
    # v0.3b (mjgen): tilt, grip, tower
    for orientation in mph.ORIENTATIONS:
        add("mjgen (v0.3b)", f"tilt/{orientation}", mph.tilt_rig((0.3, 0.2, 0.5), 5.0, orientation).model, [1], "cx·L, (kh−0.5)·H")
    grip = mph.grip_rig((0.1, 0.1, 0.15), 1.0)
    add("mjgen (v0.3b)", "grip", grip.model, [grip.boxes["load"]["body"]], "none (centre of mass at the centre; 1 mm from the nominal)")
    for n in (2, 3):
        rig = mph.tower_rig([((0.40, 0.30, 0.25), 6.0), ((0.35, 0.30, 0.20), 4.0), ((0.30, 0.25, 0.20), 3.0)][:n])
        add("mjgen (v0.3b)", f"tower/{n} boxes", rig.model, [b["body"] for b in rig.boxes.values()], "cx·L fore-aft only (z −1 mm from the nominal)")
    # v0.4a (mjgen.scenes): nine scenes, rigs from candidate 0 of each
    for name in sorted(SCENES):
        scene = SCENES[name]
        v = B.candidate(name, 0, 20261040, k=2, pair_share=0.0).visible
        if hasattr(scene, "rig_for"):
            rig = scene.rig_for(v, {})
        elif name == "table":
            rig = rigs.ledge_rig(v["dims"], v["mass"])
        else:
            rig = rigs.squeeze_rig(name, v["dims"], pad_half=scene.pad_for(v["dims"]), hand_mass=scene.hand_mass)
        add("mjgen.scenes (v0.4a)", name, rig.model, [b["body"] for b in rig.boxes.values()], "see the scene's roll()")
    # T9: its own random scenes from T9's samplers (our seeds; no T9 file is read)
    moved_t9 = {"roll": ["item"], "swing": ["load"], "knock": ["obj"], "conveyor": ["item"], "overhang": ["board", "item"], "suction": ["panel"]}
    notes_t9 = {"roll": "item: (0.001, 0, 0) → 0 (x only, rotates with the item)", "swing": "load: along the strap (geom_pos is moved too)",
                "knock": "cx·b, (hc−0.5)·h", "conveyor": "cx·b, (hc−0.5)·h", "overhang": "board: −1 mm each axis; item: (icx·d − 0.001, −0.001, −0.001)",
                "suction": "panel: −1 mm each axis"}
    for name in t9ph.FAMILIES:
        family = t9fam.family(name)
        rng = random.Random(1234)
        vis = family.sample(rng)
        for _ in range(50):
            if family.valid(vis):
                break
            vis = family.sample(rng)
        scene = t9ph.make_scene(name, family.scene(vis))
        add("t9", name, scene.model, [scene.body(b) for b in moved_t9[name]], notes_t9[name])
    _write_json(Path(args.out), {**_meta(kind="physaudit-exposure"), "entries": entries})
    for e in entries:
        _log(f"{e['package']:22s} {e['scene']:16s} exposed={e['exposed']!s:5s} world geoms={e['world_geoms']} pairs="
             f"{[(p['body'], p['other'], p['other_geom_types']) for p in e['midphase_pairs']]}")
    return 0


# --------------------------------------------------------------------------
# T8b
# --------------------------------------------------------------------------


def _t8b_rolls(moment: Any, k: int, prior: str, refresh: bool) -> dict[str, list[dict[str, Any]]]:
    from robo_jev.outside import moments as M

    draws = M.hidden_draws(M.prior_ranges(moment.hidden, prior), k, moment.seed ^ 0x5EED)
    scenes: dict[Any, Any] = {}
    return {name: M._roll_variant(moment, name, draws, scenes, refresh_bvh=refresh) for name in moment.variants}


def _t8b_label(moment: Any, k: int, rolls: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    """label_moment's result from per-draw rolls (same arithmetic)."""
    import numpy as np

    from robo_jev.outside import moments as M

    outcomes = M.OUTCOMES[moment.family]
    variants = {}
    for name, rr in rolls.items():
        counts = {o: sum(1 for r in rr if r["outcome"] == o) for o in outcomes}
        entry = {"n": counts, "freq": {o: counts[o] / len(rr) for o in outcomes}, "max_tilt_mean": float(np.mean([r["max_tilt"] for r in rr])),
                 "unstable": sum(1 for r in rr if r.get("unstable"))}
        if "slide_m" in rr[0]:
            entry["slide_mean"] = float(np.mean([r["slide_m"] for r in rr]))
            entry["off_deck"] = sum(1 for r in rr if r.get("off_deck"))
        variants[name] = entry
    planned = variants["planned"]["freq"]
    majority = max(outcomes, key=lambda o: (planned[o], -outcomes.index(o)))
    return {"k": int(k), "draws_seed": int(moment.seed ^ 0x5EED), "variants": variants, "majority": majority,
            "unstable": int(sum(e["unstable"] for e in variants.values())), "off_deck": int(sum(e.get("off_deck", 0) for e in variants.values()))}


def _draw_diff(a: dict[str, list[dict[str, Any]]], b: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    """Per variant: rollouts whose result differs at all (a contact was missed somewhere in the stored roll) and whose outcome flipped."""
    out = {}
    for name in a:
        diff = [ra != rb for ra, rb in zip(a[name], b[name])]
        flip = [ra["outcome"] != rb["outcome"] for ra, rb in zip(a[name], b[name])]
        out[name] = {"rolls": len(diff), "changed": sum(diff), "outcome_flips": sum(flip),
                     "flips": [[ra["outcome"], rb["outcome"]] for ra, rb in zip(a[name], b[name]) if ra["outcome"] != rb["outcome"]]}
    return out


def _t8b_job(job: tuple[dict[str, Any], str, dict[str, Any] | None]) -> dict[str, Any]:
    """One stored T8b record under one prior: the stored labeller (must reproduce the stored labels) and the refreshed one."""
    from robo_jev.outside import moments as M

    record, prior, b2_row = job
    started = time.perf_counter()
    moment = M.moment_from_record(record)
    k = int(record["generator"]["k"])
    questions = record["request"]["questions"]
    rolls = {refresh: _t8b_rolls(moment, k, prior, refresh) for refresh in (False, True)}
    labels = {refresh: _t8b_label(moment, k, rolls[refresh]) for refresh in (False, True)}
    answers = {refresh: M._labels(moment, labels[refresh], questions) for refresh in (False, True)}
    stored = record["labels"] if prior == "original" else (b2_row or {}).get("labels")
    stored_unstable = None if prior == "original" else (b2_row or {}).get("unstable")
    return {"id": record["id"], "family": moment.family, "prior": prior, "k": k,
            "reproduces_stored": stored is not None and answers[False] == stored,
            "stored_unstable_matches": stored_unstable is None or stored_unstable == labels[False]["unstable"],
            "labels_stored_method": answers[False], "labels_refreshed": answers[True],
            "majority": {"stored_method": labels[False]["majority"], "refreshed": labels[True]["majority"], "record": record["meta"]["majority"]},
            "unstable": {"stored_method": labels[False]["unstable"], "refreshed": labels[True]["unstable"]},
            "unstable_by_variant": {name: [labels[False]["variants"][name]["unstable"], labels[True]["variants"][name]["unstable"]] for name in rolls[False]},
            "outcomes": {"stored_method": {n: e["freq"] for n, e in labels[False]["variants"].items()},
                         "refreshed": {n: e["freq"] for n, e in labels[True]["variants"].items()}},
            "draws": _draw_diff(rolls[False], rolls[True]), "questions": compare_labels(answers[False], answers[True]),
            "seconds": round(time.perf_counter() - started, 2)}


def _t8b_summary(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    keys = sorted({(r["prior"], r["family"]) for r in rows})
    for prior, family in keys:
        mine = [r for r in rows if r["prior"] == prior and r["family"] == family]
        questions = [q for r in mine for q in r["questions"]]
        rolls = sum(v["rolls"] for r in mine for v in r["draws"].values())
        out[f"{prior}/{family}"] = {
            "moments": len(mine), "reproduces_stored": sum(r["reproduces_stored"] for r in mine), "stored_unstable_matches": sum(r["stored_unstable_matches"] for r in mine),
            "rollouts": rolls, "rollouts_changed": sum(v["changed"] for r in mine for v in r["draws"].values()),
            "rollout_outcome_flips": sum(v["outcome_flips"] for r in mine for v in r["draws"].values()),
            "moments_with_a_changed_rollout": sum(1 for r in mine if any(v["changed"] for v in r["draws"].values())),
            "majority_flips": sum(1 for r in mine if r["majority"]["stored_method"] != r["majority"]["refreshed"]),
            "unstable_rollouts": {"stored_method": sum(r["unstable"]["stored_method"] for r in mine), "refreshed": sum(r["unstable"]["refreshed"] for r in mine)},
            "moments_with_unstable": {"stored_method": sum(1 for r in mine if r["unstable"]["stored_method"]), "refreshed": sum(1 for r in mine if r["unstable"]["refreshed"])},
            **summarise_questions(questions),
            "by_question": {name: summarise_questions([q for q in questions if q["q"] == name]) for name in sorted({q["q"] for q in questions})},
        }
    return out


def cmd_t8b(args: argparse.Namespace) -> int:
    from robo_jev.decisions.records import read_records

    records = read_records(T8_DIR / "T8b.jsonl")
    jobs: list[tuple[dict[str, Any], str, dict[str, Any] | None]] = []
    families = set(args.families)
    for record in records:
        if record["meta"]["scene"] in families:
            jobs.append((record, "original", None))
    for prior in args.alt_priors:
        b2 = {r["id"]: r for r in _read_jsonl(B2_DIR / f"T8b-labels-{prior}.jsonl")}
        for record in records:
            if record["meta"]["scene"] == "stack":
                jobs.append((record, prior, b2[record["id"]]))
    _log(f"T8b: {len(jobs)} jobs (families {sorted(families)}, alt priors {args.alt_priors} on stack) on {args.workers} workers")
    started = time.perf_counter()
    rows = _pool_map(_t8b_job, jobs, args.workers)
    wall = time.perf_counter() - started
    out = Path(args.out_dir)
    _write_jsonl(out / "t8b-moments.jsonl", rows)
    summary = _t8b_summary(rows)
    _write_json(out / "t8b-summary.json", {**_meta(kind="physaudit-t8b", wall_s=round(wall, 1), worker_s=round(sum(r["seconds"] for r in rows), 1)),
                                           "input": "artifacts/t8/T8b.jsonl", "b2": [f"artifacts/b2/T8b-labels-{p}.jsonl" for p in args.alt_priors],
                                           "summary": summary})
    for key, entry in summary.items():
        _log(f"{key}: {json.dumps({k: v for k, v in entry.items() if k != 'by_question'})}")
    return 0


def _t8b_candidate_job(job: tuple[str, int, int, int]) -> dict[str, Any]:
    from robo_jev.outside import moments as M

    family, index, master, k = job
    started = time.perf_counter()
    moment = M.candidate(family, index, master)
    rolls = {refresh: _t8b_rolls(moment, k, "original", refresh) for refresh in (False, True)}
    labels = {refresh: _t8b_label(moment, k, rolls[refresh]) for refresh in (False, True)}
    return {"family": family, "index": index, "seed": moment.seed, "target": moment.target,
            "label_stored_method": labels[False], "label_refreshed": labels[True], "draws": _draw_diff(rolls[False], rolls[True]),
            "seconds": round(time.perf_counter() - started, 2)}


def cmd_t8b_candidates(args: argparse.Namespace) -> int:
    """All stack candidates of T8b's generator, both ways, and T8b's balanced selection run on each: which moments would be kept."""
    from robo_jev.outside import moments as M

    stored = [r for r in _read_jsonl(T8_DIR / "T8b.candidates.jsonl") if r["family"] == args.family]
    indices = sorted(int(r["index"]) for r in stored)
    jobs = [(args.family, i, T8B_MASTER, 64) for i in indices]
    _log(f"T8b candidates: {len(jobs)} {args.family} candidates on {args.workers} workers")
    started = time.perf_counter()
    rows = _pool_map(_t8b_candidate_job, jobs, args.workers)
    wall = time.perf_counter() - started
    by_index = {int(r["index"]): r for r in stored}
    for row in rows:
        s = by_index[row["index"]]
        row["reproduces_stored"] = ({n: v["freq"] for n, v in row["label_stored_method"]["variants"].items()} == s["outcomes"]
                                    and row["label_stored_method"]["majority"] == s["majority"])
        row["stored_kept"] = bool(s["kept"])
    targets = {"stack": 120, "push": 120, "deck_brake": 150, "deck_turn": 90}
    selections = {}
    for which, key in (("stored_method", "label_stored_method"), ("refreshed", "label_refreshed")):
        labelled = [(M.candidate(args.family, r["index"], T8B_MASTER), r[key]) for r in sorted(rows, key=lambda r: r["index"])]
        kept, report = M.select_balanced(labelled, {args.family: targets[args.family]})
        selections[which] = {"kept": sorted(m.index for m, _ in kept), "report": report}
    kept_stored = sorted(r["index"] for r in rows if r["stored_kept"])
    common = set(selections["refreshed"]["kept"]) & set(kept_stored)
    summary = {
        "candidates": len(rows), "reproduces_stored": sum(r["reproduces_stored"] for r in rows),
        "stored_selection_reproduced": selections["stored_method"]["kept"] == kept_stored,
        "selection_report": {k: v["report"] for k, v in selections.items()},
        "kept_by_both": len(common), "kept_only_stored": len(set(kept_stored) - common), "kept_only_refreshed": len(set(selections["refreshed"]["kept"]) - common),
        "majority_flips": sum(1 for r in rows if r["label_stored_method"]["majority"] != r["label_refreshed"]["majority"]),
        "candidates_unstable": {"stored_method": sum(1 for r in rows if r["label_stored_method"]["unstable"]),
                                "refreshed": sum(1 for r in rows if r["label_refreshed"]["unstable"])},
        "rollouts_changed": sum(v["changed"] for r in rows for v in r["draws"].values()), "rollouts": sum(v["rolls"] for r in rows for v in r["draws"].values()),
        "rollout_outcome_flips": sum(v["outcome_flips"] for r in rows for v in r["draws"].values()),
    }
    out = Path(args.out_dir)
    _write_jsonl(out / f"t8b-candidates-{args.family}.jsonl", rows)
    _write_json(out / f"t8b-candidates-{args.family}.summary.json", {**_meta(kind="physaudit-t8b-candidates", wall_s=round(wall, 1)), "summary": summary,
                                                                     "selections": selections, "kept_stored": kept_stored})
    _log(json.dumps({k: v for k, v in summary.items() if k != "selection_report"}))
    return 0


def _stack_draw_job(job: tuple[dict[str, Any], str]) -> dict[str, Any]:
    """Per draw of one stored stack moment's planned variant: the hidden centre of mass, how far the stored rollout's bounding volume sat
    above the box (the compiled offset subtracted), the deepest box–support contact MuJoCo reported (max −dist over the load–support
    contacts, every step; a contact the midphase skipped is reported only once the box is that deep) both ways, and the outcomes."""
    import mujoco

    from robo_jev.outside import moments as M
    from robo_jev.outside import physics as ph

    record, variant = job
    moment = M.moment_from_record(record)
    v = moment.visible
    k = int(record["generator"]["k"])
    draws = M.hidden_draws(moment.hidden, k, moment.seed ^ 0x5EED)
    spec = moment.variants[variant]
    rows = []
    for refresh in (False, True):
        scene = ph.Scene(ph.stack_xml(v["dims"], v["mass"], (v["support_top"], max(0.3, v["dims"][1] + 0.1), v["support_height"])), refresh_bvh=refresh)
        load_geom = mujoco.mj_name2id(scene.model, mujoco.mjtObj.mjOBJ_GEOM, "load_geom")
        original_step = mujoco.mj_step

        for j, d in enumerate(draws):
            centre = v["support_top"] / 2 + spec["overhang"] - v["dims"][0] / 2 + d["eps"]
            low = [0.0]
            count = [0]

            def step(model: Any, data: Any, *a: Any, low: list = low, count: list = count, scene: Any = scene, geom: int = load_geom) -> None:
                original_step(model, data, *a)
                count[0] += 1
                for c in range(int(data.ncon)):
                    contact = data.contact[c]
                    if {int(contact.geom1), int(contact.geom2)} == {geom, int(scene.surface)}:
                        low[0] = min(low[0], float(contact.dist))

            mujoco.mj_step = step  # run_stack calls mujoco.mj_step(model, data); restored below
            try:
                result = ph.run_stack(scene, x_centre=centre, drop=v["drop"], tilt0_deg=d["tilt0"], mu=d["mu"], com=M._com("stack", v, d),
                                      support_height=v["support_height"], box_height=v["dims"][2])
            finally:
                mujoco.mj_step = original_step
            com = M._com("stack", v, d)
            if not refresh:
                rows.append({"draw": j, "kh": d["kh"], "cx": d["cx"], "com_z_minus_nominal": com[2] - ph.NOMINAL_IPOS,
                             "stored": {"outcome": result["outcome"], "unstable": result["unstable"], "deepest_contact_mm": -1000 * low[0],
                                        "settle_steps": result["settle_steps"], "final_tilt": result["final_tilt"]}})
            else:
                rows[j]["refreshed"] = {"outcome": result["outcome"], "unstable": result["unstable"], "deepest_contact_mm": -1000 * low[0],
                                        "settle_steps": result["settle_steps"], "final_tilt": result["final_tilt"]}
    return {"id": record["id"], "variant": variant, "dims": v["dims"], "overhang": spec["overhang"], "rows": rows}


def cmd_t8b_draws(args: argparse.Namespace) -> int:
    from robo_jev.decisions.records import read_records

    records = [r for r in read_records(T8_DIR / "T8b.jsonl") if r["meta"]["scene"] == "stack"]
    chosen = random.Random(f"physaudit-draws|{args.seed}").sample(records, min(args.moments, len(records)))
    jobs = [(r, "planned") for r in sorted(chosen, key=lambda r: r["id"])]
    _log(f"stack draws: {len(jobs)} moments (planned variant) on {args.workers} workers")
    rows = _pool_map(_stack_draw_job, jobs, args.workers)
    flat = [dict(row, id=m["id"]) for m in rows for row in m["rows"]]
    up = [r for r in flat if r["com_z_minus_nominal"] > 0]
    down = [r for r in flat if r["com_z_minus_nominal"] <= 0]

    def frac(rows: list[dict[str, Any]], pred: Any) -> float | None:
        return sum(1 for r in rows if pred(r)) / len(rows) if rows else None

    summary = {
        "draws": len(flat),
        "raised (com z above the compiled offset)": {"draws": len(up), "outcome_flips": frac(up, lambda r: r["stored"]["outcome"] != r["refreshed"]["outcome"]),
                                                    "stored_fall": frac(up, lambda r: r["stored"]["outcome"] == "fall"),
                                                    "refreshed_fall": frac(up, lambda r: r["refreshed"]["outcome"] == "fall"),
                                                    "mean_deepest_contact_mm": {"stored": sum(r["stored"]["deepest_contact_mm"] for r in up) / max(1, len(up)),
                                                                            "refreshed": sum(r["refreshed"]["deepest_contact_mm"] for r in up) / max(1, len(up))},
                                                    "mean_com_shift_mm": 1000 * sum(r["com_z_minus_nominal"] for r in up) / max(1, len(up)),
                                                    "stored_unsettled": frac(up, lambda r: r["stored"]["settle_steps"] >= 4000)},
        "lowered or level": {"draws": len(down), "outcome_flips": frac(down, lambda r: r["stored"]["outcome"] != r["refreshed"]["outcome"]),
                             "stored_fall": frac(down, lambda r: r["stored"]["outcome"] == "fall"),
                             "refreshed_fall": frac(down, lambda r: r["refreshed"]["outcome"] == "fall"),
                             "mean_deepest_contact_mm": {"stored": sum(r["stored"]["deepest_contact_mm"] for r in down) / max(1, len(down)),
                                                     "refreshed": sum(r["refreshed"]["deepest_contact_mm"] for r in down) / max(1, len(down))}},
    }
    _write_jsonl(Path(args.out_dir) / "t8b-stack-draws.jsonl", flat)
    _write_json(Path(args.out_dir) / "t8b-stack-draws.summary.json", {**_meta(kind="physaudit-t8b-stack-draws"), "moments": [m["id"] for m in rows],
                                                                       "summary": summary})
    _log(json.dumps(summary))
    return 0


def _fixcheck_job(job: dict[str, Any]) -> dict[str, Any]:
    """One stored stack moment rolled with four fixes: ``refresh_bvh`` (the audit's labeller), ``compiled_com`` (a model compiled per draw
    with the hidden centre of mass in the XML), ``own_body`` (the support box in its own static body: no midphase pair) and
    ``midphase_off`` (MuJoCo's midphase disabled); per draw, whether each result equals the refresh_bvh one bit for bit, and the
    outcomes."""
    from robo_jev.outside import moments as M
    from robo_jev.outside import physics as ph

    record = job
    moment = M.moment_from_record(record)
    v = moment.visible
    draws = M.hidden_draws(moment.hidden, int(record["generator"]["k"]), moment.seed ^ 0x5EED)
    support = (v["support_top"], max(0.3, v["dims"][1] + 0.1), v["support_height"])
    base_xml = ph.stack_xml(v["dims"], v["mass"], support)
    nominal = f'<inertial pos="{ph.NOMINAL_IPOS} {ph.NOMINAL_IPOS} {ph.NOMINAL_IPOS}"'
    surface_line = next(line for line in base_xml.splitlines() if 'name="surface"' in line)
    variants = {
        "own_body": base_xml.replace(surface_line, f'    <body name="support_body">{surface_line.strip()}</body>'),
        "midphase_off": base_xml.replace('<flag multiccd="enable"/>', '<flag multiccd="enable" midphase="disable"/>'),
    }
    assert all(x != base_xml for x in variants.values())
    out: dict[str, Any] = {"id": record["id"], "methods": {}}
    reference: dict[str, list[dict[str, Any]]] = {}
    for method in ("refresh_bvh", "compiled_com", "own_body", "midphase_off"):
        rolls: dict[str, list[dict[str, Any]]] = {}
        scenes: dict[str, Any] = {}
        for name, spec in moment.variants.items():
            rolls[name] = []
            for d in draws:
                com = M._com("stack", v, d)
                if method == "refresh_bvh":
                    scene = scenes.setdefault("s", ph.Scene(base_xml, refresh_bvh=True))
                elif method == "compiled_com":
                    scene = ph.Scene(base_xml.replace(nominal, f'<inertial pos="{com[0]!r} {com[1]!r} {com[2]!r}"'))
                else:
                    scene = scenes.setdefault(method, ph.Scene(variants[method]))
                centre = v["support_top"] / 2 + spec["overhang"] - v["dims"][0] / 2 + d["eps"]
                rolls[name].append(ph.run_stack(scene, x_centre=centre, drop=v["drop"], tilt0_deg=d["tilt0"], mu=d["mu"], com=com,
                                                support_height=v["support_height"], box_height=v["dims"][2]))
        if method == "refresh_bvh":
            reference = rolls
        flat_a = [r for name in rolls for r in reference[name]]
        flat_b = [r for name in rolls for r in rolls[name]]
        out["methods"][method] = {"rolls": len(flat_b), "identical": sum(1 for a, b in zip(flat_a, flat_b) if a == b),
                                  "outcome_flips": sum(1 for a, b in zip(flat_a, flat_b) if a["outcome"] != b["outcome"]),
                                  "fall": {name: sum(1 for r in rr if r["outcome"] == "fall") / len(rr) for name, rr in rolls.items()},
                                  "unstable": sum(1 for r in flat_b if r["unstable"])}
    return out


def cmd_t8b_fixcheck(args: argparse.Namespace) -> int:
    from robo_jev.decisions.records import read_records

    records = [r for r in read_records(T8_DIR / "T8b.jsonl") if r["meta"]["scene"] == "stack"]
    chosen = sorted(random.Random(f"physaudit-fixcheck|{args.seed}").sample(records, min(args.moments, len(records))), key=lambda r: r["id"])
    _log(f"fix check: {len(chosen)} stack moments × 4 methods on {args.workers} workers")
    rows = _pool_map(_fixcheck_job, chosen, args.workers)
    summary: dict[str, Any] = {}
    for method in ("refresh_bvh", "compiled_com", "own_body", "midphase_off"):
        entries = [r["methods"][method] for r in rows]
        deltas = [abs(e["fall"][name] - r["methods"]["refresh_bvh"]["fall"][name]) for r, e in zip(rows, entries) for name in e["fall"]]
        summary[method] = {"rolls": sum(e["rolls"] for e in entries), "identical_to_refresh_bvh": sum(e["identical"] for e in entries),
                           "outcome_flips_vs_refresh_bvh": sum(e["outcome_flips"] for e in entries),
                           "mean_abs_label_delta_vs_refresh_bvh": sum(deltas) / len(deltas), "max_abs_label_delta_vs_refresh_bvh": max(deltas),
                           "unstable": sum(e["unstable"] for e in entries)}
    _write_jsonl(Path(args.out_dir) / "t8b-fixcheck.jsonl", rows)
    _write_json(Path(args.out_dir) / "t8b-fixcheck.summary.json", {**_meta(kind="physaudit-t8b-fixcheck"), "moments": [r["id"] for r in rows],
                                                                    "summary": summary})
    _log(json.dumps(summary))
    return 0


# --------------------------------------------------------------------------
# mjgen (v0.3b, B1, v0.4a)
# --------------------------------------------------------------------------


class _RefreshRigs(dict):
    """A rig cache that turns on ``refresh_bvh`` for every rig the labeller builds."""

    def __setitem__(self, key: Any, rig: Any) -> None:
        rig.refresh_bvh = True
        super().__setitem__(key, rig)


def _mjgen_job(job: tuple[str, dict[str, Any]]) -> dict[str, Any]:
    from robo_jev.mjgen import families as fam
    from robo_jev.mjgen.scenes import build as B

    kind, row = job
    started = time.perf_counter()
    group = fam.Group(**row["group"])
    stored = row["labelled"]
    rigs = _RefreshRigs()
    if kind == "v03":
        fresh = fam.label_group(group, k=int(stored["k"]), rigs=rigs)
    else:
        fresh = B.label_group(group, k=int(stored["k"]), rigs=rigs)
    assert rigs and all(getattr(r, "refresh_bvh", False) for r in rigs.values())
    compared = flips = draws = 0
    mismatch: list[list[str]] = []
    for member, entry in stored["members"].items():
        for name, variant in entry["variants"].items():
            for prior, outcomes in variant["outcomes"].items():
                other = fresh["members"][member]["variants"][name]["outcomes"][prior]
                compared += 1
                draws += len(outcomes)
                f = sum(1 for a, b in zip(outcomes, other) if a != b)
                if f or len(outcomes) != len(other):
                    flips += f
                    mismatch.append([member, name, prior])
    return {"family": group.family, "index": group.index, "k": int(stored["k"]), "variants_compared": compared, "draws": draws,
            "draw_outcome_flips": flips, "mismatched": mismatch, "identical": not mismatch,
            "unstable": {"stored": int(stored.get("flags", {}).get("unstable", 0)), "refreshed": int(fresh["flags"]["unstable"])},
            "midphase_pairs": sum(len(r.midphase_pairs()) for r in rigs.values()), "seconds": round(time.perf_counter() - started, 2)}


def cmd_mjgen(args: argparse.Namespace) -> int:
    spec = SETS[args.set]
    families = set(args.family)
    rows = []
    for path in spec["candidates"]:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                # cheap family filter before parsing the whole row
                head = line[:200]
                if not any(f'"family": "{f}"' in head for f in families):
                    continue
                rows.append(json.loads(line))
    rows.sort(key=lambda r: int(r["group"]["index"]))
    by_family: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_family.setdefault(row["group"]["family"], []).append(row)
    rng = random.Random(f"physaudit|{args.set}|{args.seed}")
    jobs = []
    for family in sorted(by_family):
        pool = by_family[family]
        chosen = pool if len(pool) <= args.groups else rng.sample(pool, args.groups)
        jobs += [(spec["kind"], row) for row in sorted(chosen, key=lambda r: int(r["group"]["index"]))]
    _log(f"{args.set}: {len(jobs)} groups ({ {f: min(len(p), args.groups) for f, p in by_family.items()} }) of {len(rows)} stored, on {args.workers} workers")
    started = time.perf_counter()
    out_rows = _pool_map(_mjgen_job, jobs, args.workers)
    wall = time.perf_counter() - started
    summary = {}
    for family in sorted({r["family"] for r in out_rows}):
        mine = [r for r in out_rows if r["family"] == family]
        summary[family] = {"groups": len(mine), "identical_groups": sum(r["identical"] for r in mine), "draws": sum(r["draws"] for r in mine),
                           "draw_outcome_flips": sum(r["draw_outcome_flips"] for r in mine), "midphase_pairs": sum(r["midphase_pairs"] for r in mine),
                           "unstable_rollouts": {"stored": sum(r["unstable"]["stored"] for r in mine), "refreshed": sum(r["unstable"]["refreshed"] for r in mine)},
                           "worker_s": round(sum(r["seconds"] for r in mine), 1)}
    tag = f"mjgen-{args.set}-{'-'.join(sorted(families))}"
    out = Path(args.out_dir)
    _write_jsonl(out / f"{tag}.jsonl", out_rows)
    _write_json(out / f"{tag}.summary.json", {**_meta(kind="physaudit-mjgen", set=args.set, wall_s=round(wall, 1)),
                                              "candidates": [str(p.relative_to(REPO)) for p in spec["candidates"]], "summary": summary})
    _log(json.dumps(summary))
    return 0


# --------------------------------------------------------------------------
# T9 physics code on our own random scenes (no T9 file is read)
# --------------------------------------------------------------------------


def _t9_job(job: tuple[str, int, int]) -> dict[str, Any]:
    import mujoco

    from robo_jev.outside import physics as ph
    from robo_jev.t9 import families as t9fam
    from robo_jev.t9 import physics as t9ph
    from robo_jev.t9 import priors as P

    name, seed, k = job
    started = time.perf_counter()
    family = t9fam.family(name)
    rng = random.Random(f"physaudit-t9|{name}|{seed}")
    vis = family.sample(rng)
    for _ in range(200):
        field, lo, hi, _sign = family.aim_spec(vis)
        trial = dict(vis)
        trial[field] = round(rng.uniform(lo, hi), 3)
        if family.valid(trial):
            vis = trial
            break
        vis = family.sample(rng)
    names = family.hidden_names()
    units = P.unit_draws(names, k, seed)
    draws = P.draws_for(family.words(vis), family.fixed(vis), "base", units, names)
    results = {}
    for refresh in (False, True):
        scene = t9ph.make_scene(name, family.scene(vis))
        if refresh:
            refs = {b: ph.bvh_reference(scene.model, b) for b in range(scene.model.nbody)}
            original = scene._refresh

            def _refresh(scene: Any = scene, refs: dict = refs, original: Any = original) -> None:
                for body, reference in refs.items():
                    ph.refresh_bvh(scene.model, body, reference)
                original()

            scene._refresh = _refresh  # type: ignore[method-assign]
        results[refresh] = [scene.run(family.action(vis), family.physics_draw(vis, d)) for d in draws]
    changed = sum(1 for a, b in zip(results[False], results[True]) if a != b)
    flips = [[a["outcome"], b["outcome"]] for a, b in zip(results[False], results[True]) if a["outcome"] != b["outcome"]]
    return {"family": name, "seed": seed, "k": k, "rolls": len(draws), "changed": changed, "outcome_flips": len(flips), "flips": flips,
            "unstable": [sum(1 for r in results[x] if r.get("unstable")) for x in (False, True)],
            "outcomes": [sorted({r["outcome"] for r in results[x]}) for x in (False, True)], "seconds": round(time.perf_counter() - started, 2),
            "mujoco": mujoco.__version__}


def cmd_t9_physics(args: argparse.Namespace) -> int:
    jobs = [(name, seed, args.k) for name in args.family for seed in range(args.moments)]
    _log(f"T9 physics on our own scenes: {len(jobs)} moments × K={args.k} on {args.workers} workers")
    started = time.perf_counter()
    rows = _pool_map(_t9_job, jobs, args.workers)
    wall = time.perf_counter() - started
    summary = {name: {"moments": sum(1 for r in rows if r["family"] == name), "rolls": sum(r["rolls"] for r in rows if r["family"] == name),
                      "rolls_changed": sum(r["changed"] for r in rows if r["family"] == name),
                      "outcome_flips": sum(r["outcome_flips"] for r in rows if r["family"] == name),
                      "unstable": [sum(r["unstable"][i] for r in rows if r["family"] == name) for i in (0, 1)]} for name in args.family}
    tag = "t9-physics-" + "-".join(args.family)
    _write_jsonl(Path(args.out_dir) / f"{tag}.jsonl", rows)
    _write_json(Path(args.out_dir) / f"{tag}.summary.json", {**_meta(kind="physaudit-t9-physics", wall_s=round(wall, 1)), "summary": summary,
                                                             "note": "T9's scene code on scenes drawn by T9's samplers with our own seeds; no T9 file is read"})
    _log(json.dumps(summary))
    return 0


T9_CANDIDATES = REPO / "artifacts" / "t9" / "T9.candidates.jsonl"
#: candidate-row fields that are bookkeeping of the labelling run, not part of the group the labeller reads
T9_RESULT_FIELDS = ("labelled", "unstable", "cpu_s", "key_a", "key_b", "wall_s")


def _t9_refreshing_scenes() -> Any:
    """A context manager: while active, every T9 scene re-expresses its bodies' bounding volumes after each ``prepare`` (the hidden
    values are written, then ``_refresh`` runs) — T9's labeller with the audit's fix, without touching T9's code."""
    import contextlib

    from robo_jev.outside import physics as ph
    from robo_jev.t9 import physics as t9ph

    @contextlib.contextmanager
    def active() -> Any:
        original = t9ph.make_scene

        def make_scene(family: str, visible: dict[str, Any]) -> Any:
            scene = original(family, visible)
            refs = {b: ph.bvh_reference(scene.model, b) for b in range(scene.model.nbody)}
            plain = scene._refresh

            def _refresh() -> None:
                for body, reference in refs.items():
                    ph.refresh_bvh(scene.model, body, reference)
                plain()

            scene._refresh = _refresh  # type: ignore[method-assign]
            return scene

        t9ph.make_scene = make_scene
        try:
            yield
        finally:
            t9ph.make_scene = original

    return active()


def _t9_codes(labels: dict[str, Any]) -> dict[tuple[str, str, str], str]:
    return {(m, v, p): entry["codes"] for m, per_variant in labels.items() for v, per_prior in per_variant.items() for p, entry in per_prior.items()}


def _t9_relabel_job(row: dict[str, Any]) -> dict[str, Any]:
    """One stored T9 candidate group relabelled by T9's own labeller twice — as stored, and with the bounding-volume refresh — and the
    per-draw outcome codes compared. Only counts leave this function (no T9 text or catalog word is written)."""
    from robo_jev.t9 import build as T9B

    started = time.perf_counter()
    group = {k: v for k, v in row.items() if k not in T9_RESULT_FIELDS}
    plain = T9B.label_group(group)
    with _t9_refreshing_scenes():
        fixed = T9B.label_group(group)
    stored, a, b = _t9_codes(row["labelled"]), _t9_codes(plain["labels"]), _t9_codes(fixed["labels"])
    draws = sum(len(c) for c in stored.values())
    reproduced = sum(sum(1 for x, y in zip(stored[k], a[k]) if x == y) for k in stored)
    flips = sum(sum(1 for x, y in zip(a[k], b[k]) if x != y) for k in a)
    return {"family": row["family"], "index": int(row["index"]), "draws": draws, "reproduced_draws": reproduced, "refresh_flips": flips,
            "unstable": [int(plain["unstable"]), int(fixed["unstable"])], "seconds": round(time.perf_counter() - started, 2)}


def cmd_t9_relabel(args: argparse.Namespace) -> int:
    """T9's stored candidate groups of the structurally exposed families (roll, overhang) relabelled with and without the fix; whether a
    group is in T9's selection comes from T9's own ``select`` run in code. Reads T9's candidate file in code only; prints counts only."""
    from robo_jev.t9 import build as T9B

    rows = [r for r in _read_jsonl(T9_CANDIDATES) if not r.get("failed")]
    by_key = {(r["family"], int(r["index"])): r for r in rows}
    chosen, _ = T9B.select(list(by_key.values()), args.per_family)
    selected = {(g["family"], int(g["index"])) for g in chosen}
    jobs = [r for key, r in sorted(by_key.items()) if key[0] in set(args.family)]
    _log(f"T9 relabel: {len(jobs)} candidate groups of {args.family} ({sum(1 for r in jobs if (r['family'], int(r['index'])) in selected)} in T9's "
         f"selection) on {args.workers} workers")
    started = time.perf_counter()
    out = _pool_map(_t9_relabel_job, jobs, args.workers)
    wall = time.perf_counter() - started
    for r in out:
        r["selected"] = (r["family"], r["index"]) in selected
    summary = {}
    for family in args.family:
        mine = [r for r in out if r["family"] == family]
        for scope, part in (("all candidates", mine), ("in T9", [r for r in mine if r["selected"]])):
            summary[f"{family} / {scope}"] = {"groups": len(part), "draws": sum(r["draws"] for r in part),
                                              "reproduced_draws": sum(r["reproduced_draws"] for r in part),
                                              "refresh_outcome_flips": sum(r["refresh_flips"] for r in part),
                                              "groups_with_a_flip": sum(1 for r in part if r["refresh_flips"]),
                                              "unstable": [sum(r["unstable"][i] for r in part) for i in (0, 1)]}
    tag = "t9-relabel-" + "-".join(args.family)
    _write_jsonl(Path(args.out_dir) / f"{tag}.jsonl", out)
    _write_json(Path(args.out_dir) / f"{tag}.summary.json", {**_meta(kind="physaudit-t9-relabel", wall_s=round(wall, 1)), "summary": summary,
                                                             "note": "T9's candidate file read in code only; counts only"})
    _log(json.dumps(summary))
    return 0


# --------------------------------------------------------------------------
# rescore T8b
# --------------------------------------------------------------------------

PREDICTIONS = {
    "base-2b": "artifacts/reports/t8/T8b-base-2b.predictions.jsonl",
    "v0-2b": "artifacts/reports/t8/T8b-v0-2b.predictions.jsonl",
    "v01mix-2b": "artifacts/reports/t8/T8b-v01mix-2b.predictions.jsonl",
    "v02a-2b": "artifacts/reports/t8/T8b-v02a-2b.predictions.jsonl",
    "v02b-2b": "artifacts/reports/t8/T8b-v02b-2b.predictions.jsonl",
    "v03-2b (cloud)": "artifacts/cloud/runs/v03-01-fetched/reports/after-v03-2b-T8b.predictions.jsonl",
    "v03-2b (spark)": "artifacts/reports/v04/spark/T8b-v03-2b.predictions.jsonl",
    "v04s0-2b": "artifacts/reports/v04/spark/T8b-v04s0-2b.predictions.jsonl",
    "v04s1-2b": "artifacts/reports/v04/spark/T8b-v04s1-2b.predictions.jsonl",
    "v05s0-2b": "artifacts/reports/v05/spark/T8b-v05s0-2b.predictions.jsonl",
    "v05s1-2b": "artifacts/reports/v05/spark/T8b-v05s1-2b.predictions.jsonl",
}


def _load_script(name: str) -> Any:
    import importlib.util

    spec = importlib.util.spec_from_file_location(f"_physaudit_{name}", REPO / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def cmd_rescore(args: argparse.Namespace) -> int:
    """Every stored T8b prediction file scored under the stored labels and under the refreshed stack labels (same moments and questions;
    only the stack labels change; nothing is predicted again): Brier skill, the System-1 readings (balanced band accuracy, severe errors,
    direction, comparison) on all of T8b and on the stack family, paired against the untrained base, and v0.3's (O) gate reading."""
    import copy

    from robo_jev.bench import system1
    from robo_jev.decisions.records import read_records
    from robo_jev.outside.score import paired_skill, skill_rows, t8b_metrics

    records = read_records(T8_DIR / "T8b.jsonl")
    fixed_rows = {r["id"]: r for r in _read_jsonl(Path(args.out_dir) / "t8b-moments.jsonl") if r["prior"] == "original"}
    if any(not r["reproduces_stored"] for r in fixed_rows.values()):
        raise SystemExit("the stored labeller did not reproduce every stored label — refusing to rescore")
    refreshed = copy.deepcopy(records)
    swapped = 0
    for record in refreshed:
        row = fixed_rows.get(record["id"])
        if row is not None and record["meta"]["scene"] == "stack":
            record["labels"] = row["labels_refreshed"]
            swapped += 1
    sets = {"stored": records, "refreshed": refreshed}
    files = dict(PREDICTIONS)
    for path in sorted((REPO / "artifacts" / "bench" / "t8" / "api").glob("*/T8b.predictions.jsonl")):
        files[f"api:{path.parent.name}"] = str(path.relative_to(REPO))
    out: dict[str, Any] = {}
    base_pred = read_records(REPO / PREDICTIONS["base-2b"])
    metrics_s1 = ("band_balanced", "severe_error", "direction", "comparison")

    def s1(recs: Sequence[dict[str, Any]], predictions: Sequence[dict[str, Any]]) -> dict[str, Any]:
        return system1.set_items("T8b", recs, predictions, reading="single_order")

    for tag, rel in files.items():
        path = REPO / rel
        if not path.is_file():
            continue
        predictions = read_records(path)
        entry: dict[str, Any] = {"file": rel}
        for which, recs in sets.items():
            single = t8b_metrics(recs, predictions, n_boot=0)["single_order"]
            row: dict[str, Any] = {"pooled_skill": (single.get("pooled") or {}).get("skill"),
                                   "stack_skill": ((single.get("by_family") or {}).get("stack") or {}).get("skill")}
            stack_recs = [r for r in recs if r["meta"]["scene"] == "stack"]
            for scope, scoped in (("all", recs), ("stack", stack_recs)):
                items = s1(scoped, predictions)
                row[f"s1_{scope}"] = {m: system1.summarize(items.get(m) or {}, stratified=m in system1.STRATIFIED).get("value") for m in metrics_s1}
                if tag != "base-2b":
                    base_items = s1(scoped, base_pred)
                    row[f"s1_{scope}_vs_base"] = {m: system1.paired_difference(items.get(m) or {}, base_items.get(m) or {}, n_boot=args.n_boot,
                                                                                stratified=m in system1.STRATIFIED) for m in metrics_s1}
            if tag != "base-2b":
                row["skill_vs_base"] = paired_skill(skill_rows(recs, predictions, reading="single_order"), skill_rows(recs, base_pred, reading="single_order"),
                                                    n_boot=args.n_boot)
            entry[which] = row
        out[tag] = entry
        _log(f"{tag:45s} skill {entry['stored']['pooled_skill']:.4f} → {entry['refreshed']['pooled_skill']:.4f} · stack skill "
             f"{entry['stored']['stack_skill']:.4f} → {entry['refreshed']['stack_skill']:.4f} · band_balanced(all) "
             f"{entry['stored']['s1_all']['band_balanced']} → {entry['refreshed']['s1_all']['band_balanced']}")
    # v0.3's pre-declared (O) gate on T8b, read with both label sets (v0.3 = the cloud run's predictions, base = the Spark base)
    gate = {}
    v03_pred = read_records(REPO / PREDICTIONS["v03-2b (cloud)"]) if (REPO / PREDICTIONS["v03-2b (cloud)"]).is_file() else None
    if v03_pred is not None:
        verdict = _load_script("verdict_v03")
        for which, recs in sets.items():
            reading = verdict.outside_reading(recs, v03_pred, base_pred, n_boot=args.n_boot)
            gate[which] = {"band_balanced_delta": reading["band_balanced"], "band_low": reading["band_low"], "severe": reading["severe"],
                           "severe_base": reading["severe_base"], "direction": reading["direction_value"],
                           "checks": {"band_low_gt_0": bool(reading["band_low"] is not None and reading["band_low"] > 0),
                                      "severe_le_base": bool(reading["severe"] is not None and reading["severe_base"] is not None and reading["severe"] <= reading["severe_base"]),
                                      "direction_ge_0.70": bool(reading["direction_value"] is not None and reading["direction_value"] >= 0.70)}}
            _log(f"v0.3 (O) with {which} labels: {json.dumps(gate[which]['checks'])} band Δ {reading['band_balanced']['value']:+.4f} "
                 f"[{reading['band_balanced']['low']:+.4f}, {reading['band_balanced']['high']:+.4f}] severe {reading['severe']:.4f} vs {reading['severe_base']:.4f} "
                 f"direction {reading['direction_value']:.4f}")
    # the (N) non-regression guards of the v0.4 and v0.5 rules on T8b: direction and comparison, lower bound > −0.05, each seed
    # (scripts/verdict_v04.py n_checks; 2,000 draws, seed 0, single order, cluster = record)
    guards: dict[str, Any] = {}
    pairs = {"v0.4 (N) vs v0.3": [("v04s0-2b", "v03-2b (spark)"), ("v04s1-2b", "v03-2b (spark)")],
             "v0.5 (N) vs v0.4 same seed": [("v05s0-2b", "v04s0-2b"), ("v05s1-2b", "v04s1-2b")]}
    cache: dict[tuple[str, str], Any] = {}
    for rule, members in pairs.items():
        for a, b in members:
            if not ((REPO / PREDICTIONS[a]).is_file() and (REPO / PREDICTIONS[b]).is_file()):
                continue
            entry = {}
            for which, recs in sets.items():
                for tag in (a, b):
                    if (tag, which) not in cache:
                        cache[(tag, which)] = s1(recs, read_records(REPO / PREDICTIONS[tag]))
                entry[which] = {m: system1.paired_difference(cache[(a, which)].get(m) or {}, cache[(b, which)].get(m) or {}, n_boot=2000, seed=0)
                                for m in ("direction", "comparison")}
                entry[which]["checks"] = {f"{m}_low_gt_-0.05": bool(entry[which][m] and entry[which][m]["low"] is not None and entry[which][m]["low"] > -0.05)
                                          for m in ("direction", "comparison")}
            guards[f"{rule}: {a} − {b}"] = entry
            _log(f"{rule}: {a} − {b}: " + " · ".join(f"{which} {m} {entry[which][m]['value']:+.4f} [{entry[which][m]['low']:+.4f}, {entry[which][m]['high']:+.4f}]"
                                                     for which in sets for m in ("direction", "comparison") if entry[which][m]))
    _write_json(Path(args.out_dir) / "t8b-rescore.json", {**_meta(kind="physaudit-t8b-rescore"), "stack_records_swapped": swapped, "models": out,
                                                          "v03_outside_gate": gate, "t8b_guards": guards})
    return 0


def cmd_b2(args: argparse.Namespace) -> int:
    """Task B2's pre-declared rule (scripts/b2_verdict.py, its own functions) read twice: with the stored labels (must give B2's published
    outcome) and with the refreshed stack labels under all three priors (the other families are unchanged). No MuJoCo; predictions are
    the stored ones."""
    import copy

    from robo_jev.bench import infamily
    from robo_jev.decisions.records import read_records
    from robo_jev.outside.robustness import band_robustness, with_robustness

    b2 = _load_script("b2_verdict")
    stored = read_records(T8_DIR / "T8b.jsonl")
    alt = {p: {str(r["id"]): r for r in _read_jsonl(B2_DIR / f"T8b-labels-{p}.jsonl")} for p in ("alt_a", "alt_b")}
    original_rows = {str(r["id"]): r for r in _read_jsonl(B2_DIR / "T8b-labels-original.jsonl")}
    mine = {(r["id"], r["prior"]): r for r in _read_jsonl(Path(args.out_dir) / "t8b-moments.jsonl") if r["family"] == "stack"}
    if any(not r["reproduces_stored"] for r in mine.values()):
        raise SystemExit("the stored labeller did not reproduce every stored stack label — refusing")
    refreshed_records = copy.deepcopy(stored)
    for record in refreshed_records:
        if record["meta"]["scene"] == "stack":
            record["labels"] = mine[(record["id"], "original")]["labels_refreshed"]
    refreshed_alt = copy.deepcopy(alt)
    for prior, rows in refreshed_alt.items():
        for rid, row in rows.items():
            if (rid, prior) in mine:
                row["labels"] = mine[(rid, prior)]["labels_refreshed"]
                row["unstable"] = mine[(rid, prior)]["unstable"]["refreshed"]
    predictions = {tag: read_records(path) for tag, _, _, path in b2.MODELS}
    flagged_stored = {rid for rows in (original_rows, *alt.values()) for rid, row in rows.items() if row.get("unstable") or row.get("off_deck")}
    flagged_refreshed = {rid for rid, row in original_rows.items() if (row.get("unstable") or row.get("off_deck")) and not (rid, "original") in mine}
    flagged_refreshed |= {rid for rows in refreshed_alt.values() for rid, row in rows.items() if row.get("unstable") or row.get("off_deck")}
    out: dict[str, Any] = {}
    for which, recs, alt_rows, flagged in (("stored", stored, alt, flagged_stored), ("refreshed", refreshed_records, refreshed_alt, flagged_refreshed)):
        robustness = band_robustness(recs, alt_rows)
        records = with_robustness(recs, robustness)
        robust, ambiguous = infamily.robust_keys(records), infamily.ambiguous_keys(records)
        bands = {b: sum(1 for key in robust if robustness[key]["bands"]["original"] == b) for b in ("low", "middle", "high")}
        two_band = bands["middle"] < b2.MIDDLE_MIN
        instrument = b2.instrument_check(1.0, bands["low"], bands["high"])
        found = b2.readings(records, predictions, robust, ambiguous, two_band=two_band, n_boot=args.n_boot, seed=0)
        verdict = b2.decide(instrument, found["paired"]["v03-base"]["primary"], found["models"]["v03"]["primary"], two_band=two_band)
        sens = b2.sensitivity(records, predictions, robust, two_band=two_band, flagged=flagged, n_boot=args.n_boot, seed=0)
        by_family = {f: {b: sum(1 for key in robust if robustness[key]["bands"]["original"] == b and key[0].startswith(f"t8b-{ {'deck_brake': 'brake', 'deck_turn': 'turn'}.get(f, f)}-"))
                         for b in ("low", "middle", "high")} for f in b2.FAMILIES}
        out[which] = {"robust_questions": len(robust), "robust_bands": bands, "robust_bands_by_family": by_family, "two_band": two_band,
                      "outcome": verdict["outcome"], "checks": verdict["checks"], "instrument": instrument,
                      "delta_v03_base_primary": found["paired"]["v03-base"]["primary"], "v03_primary": found["models"]["v03"]["primary"],
                      "base_primary": found["models"]["base"]["primary"], "flagged_moments": len(flagged),
                      "sensitivity": {k: {"outcome_if_applied": v["outcome_if_applied"], "questions": v["questions"],
                                          "delta": v["delta_v03_base"]} for k, v in sens.items()}}
        d = out[which]["delta_v03_base_primary"]
        _log(f"B2 with {which} labels: robust {len(robust)} {bands} two-band {two_band} → {verdict['outcome']} · Δ(v0.3 − base) "
             f"{d['value']:+.4f} [{d['low']:+.4f}, {d['high']:+.4f}] · v0.3 {out[which]['v03_primary']['value']:.4f}")
    _write_json(Path(args.out_dir) / "b2-reread.json", {**_meta(kind="physaudit-b2-reread"), "rule": b2.RULE, "readings": out})
    return 0


# --------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("repro")
    p.add_argument("--out", type=Path, default=OUT / "repro.json")
    p = sub.add_parser("exposure")
    p.add_argument("--out", type=Path, default=OUT / "exposure.json")
    p = sub.add_parser("t8b")
    p.add_argument("--families", nargs="+", default=["deck_brake", "deck_turn", "stack", "push"])
    p.add_argument("--alt-priors", nargs="*", default=["alt_a", "alt_b"])
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--out-dir", type=Path, default=OUT)
    p = sub.add_parser("t8b-candidates")
    p.add_argument("--family", default="stack")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--out-dir", type=Path, default=OUT)
    p = sub.add_parser("t8b-fixcheck")
    p.add_argument("--moments", type=int, default=30)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--out-dir", type=Path, default=OUT)
    p = sub.add_parser("t8b-draws")
    p.add_argument("--moments", type=int, default=24)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--out-dir", type=Path, default=OUT)
    p = sub.add_parser("mjgen")
    p.add_argument("--set", choices=sorted(SETS), required=True)
    p.add_argument("--family", nargs="+", required=True, help="families (v03, b1) or scenes (v04)")
    p.add_argument("--groups", type=int, default=300, help="groups per family / scene (seeded sample)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--out-dir", type=Path, default=OUT)
    p = sub.add_parser("t9-physics")
    p.add_argument("--family", nargs="+", default=["overhang", "roll"])
    p.add_argument("--moments", type=int, default=60)
    p.add_argument("--k", type=int, default=32)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--out-dir", type=Path, default=OUT)
    p = sub.add_parser("t9-relabel")
    p.add_argument("--family", nargs="+", default=["roll", "overhang"])
    p.add_argument("--per-family", type=int, default=50, help="T9's selection size per family (scripts/t9.py build --per-family)")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--out-dir", type=Path, default=OUT)
    p = sub.add_parser("rescore")
    p.add_argument("--n-boot", type=int, default=1000)
    p.add_argument("--out-dir", type=Path, default=OUT)
    p = sub.add_parser("b2")
    p.add_argument("--n-boot", type=int, default=2000)
    p.add_argument("--out-dir", type=Path, default=OUT)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    handler = {"repro": cmd_repro, "exposure": cmd_exposure, "t8b": cmd_t8b, "t8b-candidates": cmd_t8b_candidates, "t8b-draws": cmd_t8b_draws,
               "t8b-fixcheck": cmd_t8b_fixcheck, "t9-relabel": cmd_t9_relabel,
               "mjgen": cmd_mjgen, "t9-physics": cmd_t9_physics, "rescore": cmd_rescore, "b2": cmd_b2}[args.command]
    return int(handler(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
