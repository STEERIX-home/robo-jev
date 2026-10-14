"""Author A1 (warehouse and logistics) — the three MuJoCo scenes, one per law.

Every scene is my own MJCF. Only the low-level helpers of :mod:`robo_jev.outside.physics` are reused: the option / contact
header (1 ms step, ``implicitfast``, elliptic cones, ``impratio`` 10, contact ``solref`` 0.004 s, ``condim`` 3) and the
box-inertia formula. **Every rollout compiles its own model** with the draw's hidden values written into the MJCF (compiling
takes ≈ 0.3 ms; changing ``body_ipos`` of a body hung on slide / hinge joints after compiling left the model inconsistent in a
test, so nothing is patched after compiling). Same stated values + same hidden values → the same MJCF string → a
bit-identical rollout (checked by the tests).

* **mast** (law A — tip or slide under acceleration): a mast picker AMR (front drive wheels, free rear casters, a tote
  raised on its mast) brakes from cruise speed at a planned deceleration. The wheels are sphere bodies on hinge joints; the
  two front ones are driven by torque-limited velocity servos and braking sets their target to zero, so each gets a constant
  brake torque sized for the planned deceleration (from the true total mass). Tyre friction comes from the floor (hidden).
  On a grippy floor the weight transfer lets the front wheels hold and the body pitches over them (the rear casters lift
  once a·h > g·d); on a slippery one the front wheels lock first (skid once the braking demand exceeds μ·N_front).
  Outcome: ``lift`` (the body pitched forward more than :data:`LIFT_DEG` at some point — the rear casters left the floor;
  this includes falling over), ``skid`` (stopping distance more than :data:`SKID_OVERRUN` beyond the planned v²/2a) or
  ``clean``.
* **dock** (law B — centre of mass past an edge): an autonomous pallet mover sets a wrapped pallet load down on a raised
  dock floor with its lead end overhanging the dock edge (a drop to the yard below). Hidden: the load's centre-of-mass
  offset along the pallet and its height, the dock surface friction, the set-down position error. Outcome: ``drop`` (no
  contact with the dock any more — it went over), ``lean`` (still touching the dock but tilted more than :data:`LEAN_DEG`:
  it pivoted over the edge and rests on the edge and the yard) or ``level``.
* **ramp** (law C — friction-limited holding): a tugger AMR towing a roll cart comes down a ramp and stops on it with its
  wheels braked (locked sphere pads carry the hidden floor friction), while the cart rolls freely (frictionless casters) and
  pushes through its drawbar (a child body at the hitch point with a vertical slide and a pitch hinge: the bar passes only
  the push along the ramp). Gravity is tilted by the ramp angle. Outcome: ``runaway`` (moved more than :data:`RUNAWAY_M`, or
  still moving after :data:`RAMP_SECONDS`), ``overrun`` (stopped beyond :data:`STOP_WINDOW_M`) or ``held``.
"""

from __future__ import annotations

import math
from typing import Any, Mapping

import mujoco
import numpy as np

from robo_jev.outside.physics import GRAVITY, TIMESTEP, _header, inertia

__all__ = [
    "LEAN_DEG",
    "LIFT_DEG",
    "OUTCOMES",
    "RAMP_SECONDS",
    "RUNAWAY_M",
    "SKID_OVERRUN",
    "STOP_WINDOW_M",
    "WHEEL_R",
    "rollout",
    "scene_xml",
]

#: Outcome categories per scene, mildest first (the last one is the severe outcome).
OUTCOMES = {"mast": ("clean", "skid", "lift"), "dock": ("level", "lean", "drop"), "ramp": ("held", "overrun", "runaway")}
WHEEL_R = 0.075
WHEEL_MASS = 2.0
SERVO_KV = 60.0
CASTER_DAMPING = 0.003
#: mast: a stop that ends more than this beyond the planned stopping distance v²/2a is a skid (m).
SKID_OVERRUN = 0.06
#: mast: the body pitching forward more than this at any time = the rear casters lifted off (a tip-up; degrees).
LIFT_DEG = 4.0
#: dock: a pallet still on the dock but tilted more than this has pivoted over the edge (degrees).
LEAN_DEG = 4.0
#: ramp: the stop window and the runaway distance (m), and the longest rollout (s).
STOP_WINDOW_M = 0.30
RUNAWAY_M = 1.50
RAMP_SECONDS = 4.0
SETTLE_LIN, SETTLE_ANG = 0.02, 0.05


def _f(x: float) -> str:
    return f"{float(x):.6f}"


def _box_inertia(mass: float, dims: tuple[float, float, float]) -> tuple[float, float, float]:
    return inertia("box", dims, mass)


def _tilt(data: Any, body: int) -> float:
    return math.degrees(math.acos(max(-1.0, min(1.0, float(data.xmat[body][8])))))


def _settled(data: Any, adr: int) -> bool:
    v = np.asarray(data.qvel[adr : adr + 6], dtype=float)
    return bool(np.linalg.norm(v[:3]) < SETTLE_LIN and np.linalg.norm(v[3:]) < SETTLE_ANG)


_BAD = (mujoco.mjtWarning.mjWARN_BADQACC, mujoco.mjtWarning.mjWARN_BADQPOS, mujoco.mjtWarning.mjWARN_BADQVEL)


def _unstable(data: Any) -> bool:
    """A numerical blow-up: non-finite state or MuJoCo's bad-qacc / bad-qpos / bad-qvel warnings."""
    return bool(not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all() or any(data.warning[int(w)].number > 0 for w in _BAD))


def _compile(xml: str) -> tuple[Any, Any]:
    model = mujoco.MjModel.from_xml_string(xml)
    return model, mujoco.MjData(model)


def _body(model: Any, name: str) -> int:
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)


def _geom(model: Any, name: str) -> int:
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)


# --------------------------------------------------------------------------
# mast — a mast picker AMR braking (law A)
# --------------------------------------------------------------------------


def mast_mass_properties(spec: Mapping[str, Any], hidden: Mapping[str, float]) -> dict[str, Any]:
    """Total mass, centre of mass and inertia (about the centre) of the robot body with its tote (wheels excluded)."""
    mb, hb, xb = float(spec["base_mass"]), float(spec["base_cg_h"]), float(spec.get("base_cg_x", 0.0))
    mt, xt = float(hidden["tote_mass"]), float(hidden["tote_x"])
    tl, tw, th = (float(v) for v in spec["tote_dims"])
    zt = float(spec["lift_h"]) + th / 2
    total = mb + mt
    xc, zc = (mb * xb + mt * xt) / total, (mb * hb + mt * zt) / total
    ib = np.array(_box_inertia(mb, (float(spec["wheelbase"]) + 0.12, float(spec["track"]) + 0.06, 2 * hb)))
    it = np.array(_box_inertia(mt, (tl, tw, th)))

    def shift(m: float, dx: float, dz: float) -> np.ndarray:
        return m * np.array([dz**2, dx**2 + dz**2, dx**2])

    inert = ib + it + shift(mb, xb - xc, hb - zc) + shift(mt, xt - xc, zt - zc)
    return {"mass": total, "cg_x": xc, "cg_h": zc, "inertia": tuple(float(v) for v in inert)}


def mast_brake_torque(total_mass: float, decel: float) -> float:
    """Brake torque per front wheel for the planned deceleration: the body, all four wheels and the rear casters' spin."""
    wheel_inertia = 0.4 * WHEEL_MASS * WHEEL_R**2
    effective = float(total_mass) + 4 * WHEEL_MASS + 2 * wheel_inertia / WHEEL_R**2
    return (effective * float(decel) / 2.0) * WHEEL_R + wheel_inertia * float(decel) / WHEEL_R


def _mast_xml(spec: Mapping[str, Any], hidden: Mapping[str, float], decel: float) -> str:
    wb, track = float(spec["wheelbase"]), float(spec["track"])
    lift = float(spec["lift_h"])
    tl, tw, th = (float(v) for v in spec["tote_dims"])
    props = mast_mass_properties(spec, hidden)
    torque = mast_brake_torque(props["mass"], decel)
    base_hx, base_hy, base_hz = wb / 2 + 0.06, track / 2 + 0.03, 0.14
    base_z = 0.05 + base_hz
    mast_hz = (max(lift, 0.36) - 0.33) / 2
    wheels = ""
    for name, sx, sy in (("fl", 1, 1), ("fr", 1, -1), ("rl", -1, 1), ("rr", -1, -1)):
        damping = "" if sx > 0 else f' damping="{CASTER_DAMPING}"'  # rear casters: bearing drag, so a lifted caster stops spinning
        wheels += (
            f'      <body name="w_{name}" pos="{_f(sx * wb / 2)} {_f(sy * track / 2)} {_f(WHEEL_R)}">\n'
            f'        <joint name="j_{name}" type="hinge" axis="0 1 0"{damping}/>\n'
            f'        <geom name="g_{name}" type="sphere" size="{WHEEL_R}" mass="{WHEEL_MASS}"/>\n'
            "      </body>\n"
        )
    actuators = "".join(
        f'    <velocity name="b_{name}" joint="j_{name}" kv="{SERVO_KV}" forcelimited="true" forcerange="{_f(-torque)} {_f(torque)}"/>\n'
        for name in ("fl", "fr")
    )
    ix, iy, iz = props["inertia"]
    return (
        _header("a1-mast")
        + "  <worldbody>\n"
        + f'    <geom name="surface" type="plane" size="40 6 0.1" priority="1" friction="{_f(hidden["mu"])} 0.005 0.0001"/>\n'
        + '    <body name="load" pos="0 0 0.0005">\n'
        + "      <freejoint/>\n"
        + f'      <inertial pos="{_f(props["cg_x"])} 0.0001 {_f(props["cg_h"])}" mass="{_f(props["mass"])}" diaginertia="{_f(ix)} {_f(iy)} {_f(iz)}"/>\n'
        + f'      <geom name="base" type="box" size="{_f(base_hx)} {_f(base_hy)} {_f(base_hz)}" pos="0 0 {_f(base_z)}"/>\n'
        + f'      <geom name="mast" type="box" size="0.04 0.06 {_f(mast_hz)}" pos="-0.14 0 {_f(0.33 + mast_hz)}"/>\n'
        + f'      <geom name="tote" type="box" size="{_f(tl / 2)} {_f(tw / 2)} {_f(th / 2 + 0.01)}" pos="{_f(hidden["tote_x"])} 0 {_f(lift + th / 2)}"/>\n'
        + wheels
        + "    </body>\n"
        + "  </worldbody>\n"
        + "  <actuator>\n"
        + actuators
        + "  </actuator>\n"
        + "</mujoco>\n"
    )


def _mast_rollout(spec: Mapping[str, Any], hidden: Mapping[str, float], *, speed: float, decel: float) -> dict[str, Any]:
    model, data = _compile(_mast_xml(spec, hidden, decel))
    body = _body(model, "load")
    q = int(model.jnt_qposadr[model.body_jntadr[body]])
    qv = int(model.jnt_dofadr[model.body_jntadr[body]])
    omega = float(speed) / WHEEL_R
    data.qvel[qv] = float(speed)
    for name in ("fl", "fr", "rl", "rr"):
        data.qvel[int(model.jnt_dofadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"j_{name}")])] = omega
    data.ctrl[:] = omega
    mujoco.mj_forward(model, data)
    mujoco.mj_step(model, data, nstep=50)  # 0.05 s of cruise to seat the contacts
    x_brake = float(data.qpos[q])
    data.ctrl[:] = 0.0
    planned = float(speed) ** 2 / (2.0 * float(decel))
    stop_steps = int(math.ceil(float(speed) / float(decel) / TIMESTEP))
    max_tilt, done, furthest = 0.0, 0, x_brake
    limit = stop_steps + int(3.0 / TIMESTEP)
    while done < limit:
        for _ in range(10):
            mujoco.mj_step(model, data, nstep=10)
            max_tilt = max(max_tilt, _tilt(data, body))
            furthest = max(furthest, float(data.qpos[q]))
        done += 100
        if max_tilt > 70.0:
            break
        if done >= stop_steps and _settled(data, qv):
            break
    # the furthest point reached (a body rocking back after the stop does not shorten it)
    overrun = furthest - x_brake - planned
    outcome = "lift" if max_tilt > LIFT_DEG else ("skid" if overrun > SKID_OVERRUN else "clean")
    return {"outcome": outcome, "max_tilt": max_tilt, "tilt": _tilt(data, body), "overrun": overrun, "planned": planned, "steps": done,
            "unstable": _unstable(data)}


# --------------------------------------------------------------------------
# dock — a pallet load set down overhanging the dock edge (law B)
# --------------------------------------------------------------------------


def _dock_xml(spec: Mapping[str, Any], hidden: Mapping[str, float], overhang: float) -> str:
    step = float(spec["step_h"])
    length, width, height = (float(v) for v in spec["pallet_dims"])
    mass = float(spec["mass"])
    ix, iy, iz = _box_inertia(mass, (length, width, height))
    x0 = -length / 2 + float(overhang) + float(hidden["place"])
    cz = float(hidden["cg_z"]) * height - height / 2
    return (
        _header("a1-dock")
        + "  <worldbody>\n"
        + '    <geom name="yard" type="plane" size="8 8 0.1" friction="0.6 0.005 0.0001"/>\n'
        + f'    <geom name="surface" type="box" pos="-2 0 {_f(step / 2)}" size="2 2.5 {_f(step / 2)}" priority="1" friction="{_f(hidden["mu"])} 0.005 0.0001"/>\n'
        + f'    <body name="load" pos="{_f(x0)} 0 {_f(step + height / 2 + 0.002)}">\n'
        + "      <freejoint/>\n"
        + f'      <inertial pos="{_f(hidden["cg_x"])} 0.0001 {_f(cz)}" mass="{_f(mass)}" diaginertia="{_f(ix)} {_f(iy)} {_f(iz)}"/>\n'
        + f'      <geom name="load_geom" type="box" size="{_f(length / 2)} {_f(width / 2)} {_f(height / 2)}"/>\n'
        + "    </body>\n"
        + "  </worldbody>\n"
        + "</mujoco>\n"
    )


def _dock_rollout(spec: Mapping[str, Any], hidden: Mapping[str, float], *, overhang: float, seconds: float = 1.5) -> dict[str, Any]:
    model, data = _compile(_dock_xml(spec, hidden, overhang))
    body = _body(model, "load")
    qv = int(model.jnt_dofadr[model.body_jntadr[body]])
    surface, load_geom = _geom(model, "surface"), _geom(model, "load_geom")
    mujoco.mj_forward(model, data)
    steps = int(round(seconds / TIMESTEP))
    mujoco.mj_step(model, data, nstep=steps)
    done, limit = steps, steps + int(3.0 / TIMESTEP)
    while done < limit and not _settled(data, qv):
        mujoco.mj_step(model, data, nstep=100)
        done += 100
    tilt = _tilt(data, body)
    on_dock = any({int(data.contact[i].geom1), int(data.contact[i].geom2)} == {surface, load_geom} for i in range(data.ncon))
    outcome = "level" if on_dock and tilt <= LEAN_DEG else ("lean" if on_dock else "drop")
    return {"outcome": outcome, "tilt": tilt, "on_dock": on_dock, "steps": done, "unstable": _unstable(data)}


# --------------------------------------------------------------------------
# ramp — a tugger and its cart stopping on a ramp (law C)
# --------------------------------------------------------------------------

TUG_DIMS = (1.00, 0.62, 0.46)
TUG_CG_H = 0.26
CART_DIMS = (1.10, 0.75, 1.50)
CART_CLEAR = 0.14
CART_CG_H = 0.62
HITCH_H = 0.30
HITCH_GAP = 0.40
PAD_R = 0.03
CASTER_R = 0.04


def _ramp_xml(spec: Mapping[str, Any], hidden: Mapping[str, float], grade: float) -> str:
    """The tugger is a free body on four sphere pads (its locked wheels; they carry the hidden friction). The cart hangs on the
    tugger's drawbar: a child body whose frame is the hitch point, with a vertical slide and a pitch hinge there — the drawbar
    passes the push along the ramp and nothing else — and it stands on four frictionless sphere casters."""
    tl, tw, th = TUG_DIMS
    cl, cw, ch = CART_DIMS
    theta = math.atan(float(grade) / 100.0)
    tug_mass = float(spec["tug_mass"])
    cart_mass = float(spec["cart_tare"]) + float(hidden["payload"])
    ti, ci = _box_inertia(tug_mass, TUG_DIMS), _box_inertia(cart_mass, CART_DIMS)
    mu = _f(hidden["mu"])
    pads = "".join(
        f'      <geom name="pad_{i}" type="sphere" size="{PAD_R}" pos="{_f(sx * (tl / 2 - 0.08))} {_f(sy * (tw / 2 - 0.06))} {PAD_R}" '
        f'priority="1" friction="{mu} 0.005 0.0001"/>\n'
        for i, (sx, sy) in enumerate(((1, 1), (1, -1), (-1, 1), (-1, -1)))
    )
    hitch_x = -(tl / 2 + HITCH_GAP / 2)
    cart_cx = -(HITCH_GAP / 2 + cl / 2)  # cart centre relative to the hitch point
    casters = "".join(
        f'        <geom name="caster_{i}" type="sphere" size="{CASTER_R}" pos="{_f(cart_cx + sx * (cl / 2 - 0.08))} {_f(sy * (cw / 2 - 0.06))} '
        f'{_f(CASTER_R - HITCH_H)}" priority="1" condim="1"/>\n'
        for i, (sx, sy) in enumerate(((1, 1), (1, -1), (-1, 1), (-1, -1)))
    )
    flat = f'gravity="0 0 {-GRAVITY}"'
    header = _header("a1-ramp")
    if flat not in header:
        raise RuntimeError("could not tilt gravity in the ramp header")
    header = header.replace(flat, f'gravity="{_f(GRAVITY * math.sin(theta))} 0 {_f(-GRAVITY * math.cos(theta))}"')
    return (
        header
        + "  <worldbody>\n"
        + '    <geom name="floor" type="plane" size="60 8 0.1"/>\n'
        + '    <body name="load" pos="0 0 0.0005">\n'
        + "      <freejoint/>\n"
        + f'      <inertial pos="0.001 0.0001 {TUG_CG_H}" mass="{_f(tug_mass)}" diaginertia="{_f(ti[0])} {_f(ti[1])} {_f(ti[2])}"/>\n'
        + f'      <geom name="tug_body" type="box" size="{_f(tl / 2)} {_f(tw / 2)} {_f((th - 0.08) / 2)}" pos="0 0 {_f(0.08 + (th - 0.08) / 2)}"/>\n'
        + pads
        + f'      <body name="cart" pos="{_f(hitch_x)} 0 {HITCH_H}">\n'
        + '        <joint name="hitch_z" type="slide" axis="0 0 1"/>\n'
        + '        <joint name="hitch_pitch" type="hinge" axis="0 1 0"/>\n'
        + f'        <inertial pos="{_f(cart_cx)} 0.0001 {_f(CART_CG_H - HITCH_H)}" mass="{_f(cart_mass)}" diaginertia="{_f(ci[0])} {_f(ci[1])} {_f(ci[2])}"/>\n'
        + f'        <geom name="cart_body" type="box" size="{_f(cl / 2)} {_f(cw / 2)} {_f((ch - CART_CLEAR) / 2)}" '
        f'pos="{_f(cart_cx)} 0 {_f(CART_CLEAR + (ch - CART_CLEAR) / 2 - HITCH_H)}"/>\n'
        + casters
        + "      </body>\n"
        + "    </body>\n"
        + "  </worldbody>\n"
        + "</mujoco>\n"
    )


def _ramp_rollout(spec: Mapping[str, Any], hidden: Mapping[str, float], *, grade: float, speed: float) -> dict[str, Any]:
    model, data = _compile(_ramp_xml(spec, hidden, grade))
    body = _body(model, "load")
    q = int(model.jnt_qposadr[model.body_jntadr[body]])
    qt = int(model.jnt_dofadr[model.body_jntadr[body]])
    data.qvel[qt] = float(speed)
    mujoco.mj_forward(model, data)
    x0 = float(data.qpos[q])
    done, limit, moving = 0, int(round(RAMP_SECONDS / TIMESTEP)), True
    while done < limit:
        mujoco.mj_step(model, data, nstep=100)
        done += 100
        if float(data.qpos[q]) - x0 > RUNAWAY_M:
            break
        if _settled(data, qt) and float(np.abs(data.qvel[qt + 6 : qt + 8]).max()) < SETTLE_ANG:
            moving = False
            break
    travel = float(data.qpos[q]) - x0
    outcome = "runaway" if (travel > RUNAWAY_M or moving) else ("overrun" if travel > STOP_WINDOW_M else "held")
    return {"outcome": outcome, "travel": travel, "moving": moving, "steps": done, "unstable": _unstable(data)}


# --------------------------------------------------------------------------
# Entry points
# --------------------------------------------------------------------------


def scene_xml(scene: str, spec: Mapping[str, Any], hidden: Mapping[str, float], state: Mapping[str, Any]) -> str:
    """The MJCF one rollout compiles (for the determinism check and for inspection)."""
    if scene == "mast":
        return _mast_xml(spec, hidden, float(state["decel"]))
    if scene == "dock":
        return _dock_xml(spec, hidden, float(state["overhang"]))
    if scene == "ramp":
        return _ramp_xml(spec, hidden, float(state["grade"]))
    raise ValueError(f"unknown scene {scene!r}")


def rollout(scene: str, spec: Mapping[str, Any], hidden: Mapping[str, float], state: Mapping[str, Any]) -> dict[str, Any]:
    """One rollout: `spec` = the stated geometry, `hidden` = one draw of the hidden values, `state` = the variant's stated motion
    values (mast: speed, decel; dock: overhang; ramp: grade, speed)."""
    if scene == "mast":
        return _mast_rollout(spec, hidden, speed=float(state["speed"]), decel=float(state["decel"]))
    if scene == "dock":
        return _dock_rollout(spec, hidden, overhang=float(state["overhang"]))
    if scene == "ramp":
        return _ramp_rollout(spec, hidden, grade=float(state["grade"]), speed=float(state["speed"]))
    raise ValueError(f"unknown scene {scene!r}")
