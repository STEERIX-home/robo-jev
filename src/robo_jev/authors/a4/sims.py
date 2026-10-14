"""Author A4 — the three MuJoCo scenes of the held-out field-robot test (v0.6, one scene per law).

The contact model is :mod:`robo_jev.outside.physics`'s (its MJCF header: timestep 1 ms, ``implicitfast``, elliptic cones with
``impratio`` 10, contact ``solref`` 0.004 s, ``condim`` 3; its rest test and rise guard); the geometry, the motions and the outcome rules
below are this author's own. **The hidden centre of mass is compiled into each draw's model** (``<inertial pos>``), not written into
``body_ipos`` afterwards: MuJoCo keeps a body's broadphase box (``bvh_aabb``) in its inertial frame and does not rebuild it, so moving
``body_ipos`` at run time shifts that box away from the geometry and box-on-box contacts are missed until the penetration exceeds the
shift (a 3 cm shift gave no contact at 2 mm penetration; planes are unaffected). A compile takes ≈ 0.25 ms. Every rollout steps in
chunks (``mj_step(..., nstep)``) and is deterministic: the same model, hidden values and plan give bit-identical results.

* **bank** (law A, tip or slide under tilt) — a field robot sets a box-shaped load down on sloping ground and lets go. The ground is a
  level plane and gravity is turned by the local slope (stated slope + a hidden local deviation), so the plane's normal stays the world
  z axis. Outcome: ``tips_over`` (the load's tilt passed 45° at any time), ``slides_down`` (it moved more than 3 cm) or ``stays_put``.
* **ledge** (law B, centre of mass past an edge) — the robot itself, a box with a hidden centre of mass, has crept forward until its
  nose sticks out past the edge of a drop (a step, terrace wall, quay step); it is released at rest. The upper level is a box whose
  edge is at x = 0, the lower level a plane at z = 0, both carrying the hidden friction. Outcome after settling: ``goes_over`` (no part
  of the robot is above the upper level any more), ``noses_down`` (tilted more than 8° but still partly on the upper level) or ``holds``.
* **anchor** (law C, friction-limited holding) — the robot is parked with its brakes set (a box on a plane, gravity turned by the ramp
  angle; 0 for the flat apron) and holds a winch line that hauls a load up the ramp. The line pulls the robot downhill at a low hitch
  with the tension of the haul: ``T = m_load · (g · (sin θ + c · cos θ) + a(t))`` with the start acceleration ``a(t)`` ramped in and out
  (holding before the start: ``m_load · g · (sin θ − c · cos θ)``). The robot's slip does not feed back into the tension (the winch
  follows its haul profile). Outcome: ``dragged`` (slid ≥ 40 cm or still sliding at the end), ``creeps`` (slid ≥ 5 mm, then held) or
  ``holds_fast``. Under static friction the contact model shows no creep (< 0.5 mm at 96 % of the slip load), so 5 mm marks a real slip.
"""

from __future__ import annotations

import math
from typing import Any, Sequence

import mujoco
import numpy as np

from robo_jev.outside import physics as P

__all__ = [
    "ANCHOR_CREEP_M",
    "ANCHOR_DRAG_M",
    "BANK_SLIDE_M",
    "BANK_TIP_DEG",
    "LEDGE_TILT_DEG",
    "anchor_scene",
    "bank_scene",
    "ledge_scene",
    "run_anchor",
    "run_bank",
    "run_ledge",
]

BANK_TIP_DEG = 45.0
BANK_SLIDE_M = 0.03
LEDGE_TILT_DEG = 8.0
ANCHOR_CREEP_M = 0.005
ANCHOR_DRAG_M = 0.40
#: Steps per chunk between checks (10 ms).
CHUNK = 10
#: Hitch height above the robot's underside (m) — low, so the line slides the robot rather than rocking it.
HITCH_Z = 0.12


class Rig:
    """A compiled scene: the ``load`` body (one box geom ``load_geom``, the hidden centre of mass compiled in) and the ``surface`` geom
    that carries the hidden friction (priority 1)."""

    def __init__(self, xml: str):
        self.xml = xml
        self.model = mujoco.MjModel.from_xml_string(xml)
        self.data = mujoco.MjData(self.model)
        self.load = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "load")
        self.surface = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "surface")
        self.load_qpos = int(self.model.jnt_qposadr[self.model.body_jntadr[self.load]])
        self.load_qvel = int(self.model.jnt_dofadr[self.model.body_jntadr[self.load]])
        geom = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "load_geom")
        self.dims = tuple(float(2 * v) for v in self.model.geom_size[geom])

    def tilt_deg(self) -> float:
        return math.degrees(math.acos(max(-1.0, min(1.0, float(self.data.xmat[self.load][8])))))

    def com_offset(self) -> np.ndarray:
        """Centre of mass minus geometric centre (world frame) — a check that the compiled centre of mass is in force."""
        return np.array(self.data.xipos[self.load]) - np.array(self.data.xpos[self.load])


def _load(dims: Sequence[float], mass: float, pos: Sequence[float], com: Sequence[float]) -> str:
    """The load body: a free box with its centre of mass at `com` (body frame) and the inertia of a uniform box about it."""
    ixx, iyy, izz = P.inertia("box", dims, mass)
    return (f'    <body name="load" pos="{pos[0]:.6f} {pos[1]:.6f} {pos[2]:.6f}">\n'
            "      <freejoint/>\n"
            f'      <inertial pos="{com[0]:.6f} {com[1]:.6f} {com[2]:.6f}" mass="{mass:.6f}" diaginertia="{ixx:.8f} {iyy:.8f} {izz:.8f}"/>\n'
            f'      <geom name="load_geom" type="box" size="{dims[0] / 2:.6f} {dims[1] / 2:.6f} {dims[2] / 2:.6f}"/>\n'
            "    </body>\n")


def _box_corners(scene: Rig) -> np.ndarray:
    """World coordinates of the load box's eight corners."""
    half = np.asarray(scene.dims, dtype=float) / 2.0
    signs = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], dtype=float)
    rot = np.asarray(scene.data.xmat[scene.load], dtype=float).reshape(3, 3)
    return np.asarray(scene.data.xpos[scene.load], dtype=float) + (signs * half) @ rot.T


def _settle(scene: Rig, step, max_seconds: float) -> int:
    """Step in chunks until the load is at rest (physics' rest test, checked every 0.1 s) or `max_seconds` pass; returns the steps."""
    steps = 0
    limit = int(round(max_seconds / P.TIMESTEP))
    while steps < limit:
        if steps % 100 == 0 and P._settled(scene):
            return steps
        step()
        steps += CHUNK
    return steps


# --------------------------------------------------------------------------
# Scene A — bank (a load set down on sloping ground)
# --------------------------------------------------------------------------


def bank_scene(dims: Sequence[float], mass: float, com: Sequence[float]) -> Rig:
    """A level plane (the ground, hidden friction) and the load standing on it; ``dims`` = (downslope extent, across, height), ``com``
    = the centre of mass in the load's frame (+x downhill)."""
    dims = tuple(float(v) for v in dims)
    xml = (P._header("a4-bank") + "  <worldbody>\n"
           '    <geom name="surface" type="plane" size="60 60 0.1" priority="1"/>\n'
           + _load(dims, float(mass), (0.0, 0.0, dims[2] / 2 + 0.0005), com)
           + "  </worldbody>\n</mujoco>\n")
    return Rig(xml)


def run_bank(scene: Rig, *, slope_deg: float, mu: float, seconds: float = 2.5) -> dict[str, Any]:
    """One release on the slope (downhill = +x) → ``{"outcome", "moved_m", "max_tilt", "unstable", "steps"}``."""
    model, data = scene.model, scene.data
    mujoco.mj_resetData(model, data)
    model.geom_friction[scene.surface, 0] = float(mu)
    theta = math.radians(max(0.0, float(slope_deg)))
    model.opt.gravity[:] = (P.GRAVITY * math.sin(theta), 0.0, -P.GRAVITY * math.cos(theta))
    mujoco.mj_forward(model, data)
    start = np.array(data.xpos[scene.load], dtype=float)
    z0 = top = float(start[2])
    max_tilt, steps, outcome = 0.0, 0, None
    for _ in range(int(round(seconds / (P.TIMESTEP * CHUNK)))):
        mujoco.mj_step(model, data, nstep=CHUNK)
        steps += CHUNK
        max_tilt = max(max_tilt, scene.tilt_deg())
        top = max(top, float(data.xpos[scene.load][2]))
        if max_tilt > BANK_TIP_DEG:
            outcome = "tips_over"
            break
        if float(np.hypot(*(data.xpos[scene.load][:2] - start[:2]))) > 0.5:
            outcome = "slides_down"
            break
    if outcome is None:
        def step() -> None:
            mujoco.mj_step(model, data, nstep=CHUNK)

        steps += _settle(scene, step, 2.0)
        max_tilt = max(max_tilt, scene.tilt_deg())
    moved = float(np.hypot(*(data.xpos[scene.load][:2] - start[:2])))
    if outcome is None:
        outcome = "tips_over" if max_tilt > BANK_TIP_DEG else ("slides_down" if moved > BANK_SLIDE_M else "stays_put")
    return {"outcome": outcome, "moved_m": moved, "max_tilt": max_tilt, "unstable": bool(top - z0 > P.max_rise(scene.dims)), "steps": steps}


# --------------------------------------------------------------------------
# Scene B — ledge (the robot's nose past the edge of a drop)
# --------------------------------------------------------------------------


def ledge_scene(dims: Sequence[float], mass: float, drop: float, com: Sequence[float], overhang: float) -> Rig:
    """Lower level (plane at z = 0) and upper level (a box whose top is at ``drop`` and whose edge is at x = 0); both carry the hidden
    friction (priority 1). The robot body is the load, compiled at rest with its front `overhang` m past the edge (front = +x)."""
    dims = tuple(float(v) for v in dims)
    drop = float(drop)
    xml = (P._header("a4-ledge") + "  <worldbody>\n"
           '    <geom name="floor" type="plane" size="30 30 0.1" priority="1"/>\n'
           f'    <geom name="surface" type="box" pos="-2 0 {drop / 2:.6f}" size="2 2 {drop / 2:.6f}" priority="1"/>\n'
           + _load(dims, float(mass), (float(overhang) - dims[0] / 2, 0.0, drop + dims[2] / 2 + 0.0005), com)
           + "  </worldbody>\n</mujoco>\n")
    scene = Rig(xml)
    scene.drop = drop  # type: ignore[attr-defined]
    scene.floor = mujoco.mj_name2id(scene.model, mujoco.mjtObj.mjOBJ_GEOM, "floor")  # type: ignore[attr-defined]
    return scene


def run_ledge(scene: Rig, *, mu: float, seconds: float = 3.0) -> dict[str, Any]:
    """The robot released at rest where it was compiled → ``{"outcome", "final_tilt", "max_tilt", "unstable", "steps"}``."""
    model, data = scene.model, scene.data
    mujoco.mj_resetData(model, data)
    model.geom_friction[scene.surface, 0] = float(mu)
    model.geom_friction[scene.floor, 0] = float(mu)  # type: ignore[attr-defined]
    mujoco.mj_forward(model, data)
    z0 = top = float(data.xpos[scene.load][2])
    max_tilt, steps = 0.0, 0
    for _ in range(int(round(seconds / (P.TIMESTEP * CHUNK)))):
        mujoco.mj_step(model, data, nstep=CHUNK)
        steps += CHUNK
        max_tilt = max(max_tilt, scene.tilt_deg())
        top = max(top, float(data.xpos[scene.load][2]))

    def step() -> None:
        mujoco.mj_step(model, data, nstep=CHUNK)

    steps += _settle(scene, step, 3.0)
    tilt = scene.tilt_deg()
    corners = _box_corners(scene)
    off = bool(corners[:, 0].min() > -0.005)
    outcome = "goes_over" if off else ("noses_down" if tilt > LEDGE_TILT_DEG else "holds")
    return {"outcome": outcome, "final_tilt": tilt, "max_tilt": max(max_tilt, tilt), "unstable": bool(top - z0 > P.max_rise(scene.dims)),
            "steps": steps}


# --------------------------------------------------------------------------
# Scene C — anchor (the parked robot holds a winch line)
# --------------------------------------------------------------------------


def anchor_scene(dims: Sequence[float], mass: float) -> Rig:
    """A level plane (the ramp under the robot, hidden friction) and the robot body standing on it, front = +x (uphill); its centre of
    mass at the geometric centre."""
    dims = tuple(float(v) for v in dims)
    xml = (P._header("a4-anchor") + "  <worldbody>\n"
           '    <geom name="surface" type="plane" size="60 60 0.1" priority="1"/>\n'
           + _load(dims, float(mass), (0.0, 0.0, dims[2] / 2 + 0.0005), (0.0, 0.0, 0.0))
           + "  </worldbody>\n</mujoco>\n")
    return Rig(xml)


def haul_tension(t: np.ndarray | float, *, load_mass: float, load_slope_deg: float, drag: float, start_accel: float, line_speed: float,
                 jerk_time: float = 0.15, pre: float = 0.3) -> np.ndarray:
    """Line tension (N) over time: holding before ``pre``; then the haul — the acceleration ramps up over `jerk_time`, holds until the
    line reaches `line_speed`, ramps down over `jerk_time`; then the steady haul."""
    t = np.asarray(t, dtype=float)
    theta = math.radians(float(load_slope_deg))
    g = P.GRAVITY
    hold = max(0.0, load_mass * g * (math.sin(theta) - drag * math.cos(theta)))
    steady = load_mass * g * (math.sin(theta) + drag * math.cos(theta))
    a = float(start_accel)
    t_acc = max(float(line_speed) / a, 2 * jerk_time) if a > 0 else 2 * jerk_time
    rel = t - pre
    accel = np.where(rel < 0, 0.0, np.where(rel < jerk_time, a * rel / jerk_time,
                                            np.where(rel < t_acc - jerk_time, a,
                                                     np.where(rel < t_acc, a * (t_acc - rel) / jerk_time, 0.0))))
    return np.where(rel < 0, hold, steady + load_mass * accel)


def run_anchor(scene: Rig, *, robot_slope_deg: float, mu: float, load_mass: float, load_slope_deg: float, drag: float,
               start_accel: float, line_speed: float, cruise: float = 2.0) -> dict[str, Any]:
    """One haul → ``{"outcome", "slip_m", "end_speed", "peak_tension", "unstable", "steps"}``. Downhill = −x for the robot. The
    rollout stops early once the robot has slid 1 m (dragged — the outcome is decided)."""
    model, data = scene.model, scene.data
    mujoco.mj_resetData(model, data)
    model.geom_friction[scene.surface, 0] = float(mu)
    theta = math.radians(max(0.0, float(robot_slope_deg)))
    model.opt.gravity[:] = (-P.GRAVITY * math.sin(theta), 0.0, -P.GRAVITY * math.cos(theta))
    mujoco.mj_forward(model, data)
    length, _, height = scene.dims
    x0 = float(data.xpos[scene.load][0])
    z0 = top = float(data.xpos[scene.load][2])
    a = float(start_accel)
    t_acc = max(float(line_speed) / a, 0.3) if a > 0 else 0.3
    seconds = 0.3 + t_acc + cruise
    hitch_body = np.array([-length / 2, 0.0, -height / 2 + HITCH_Z])
    peak, steps = 0.0, 0
    n = int(round(seconds / (P.TIMESTEP * CHUNK)))
    times = (np.arange(n) * CHUNK + CHUNK / 2) * P.TIMESTEP
    tensions = haul_tension(times, load_mass=load_mass, load_slope_deg=load_slope_deg, drag=drag, start_accel=start_accel, line_speed=line_speed)
    for i in range(n):
        tension = float(tensions[i])
        peak = max(peak, tension)
        rot = np.asarray(data.xmat[scene.load], dtype=float).reshape(3, 3)
        hitch = np.asarray(data.xpos[scene.load], dtype=float) + rot @ hitch_body
        force = np.array([-tension, 0.0, 0.0])
        torque = np.cross(hitch - np.asarray(data.xipos[scene.load], dtype=float), force)
        data.xfrc_applied[scene.load, :3] = force
        data.xfrc_applied[scene.load, 3:] = torque
        mujoco.mj_step(model, data, nstep=CHUNK)
        steps += CHUNK
        top = max(top, float(data.xpos[scene.load][2]))
        if x0 - float(data.xpos[scene.load][0]) > 1.0:
            break
    data.xfrc_applied[scene.load] = 0.0
    slip = x0 - float(data.xpos[scene.load][0])
    speed = float(np.linalg.norm(data.qvel[scene.load_qvel: scene.load_qvel + 3]))
    if slip >= ANCHOR_DRAG_M or speed > 0.05:
        outcome = "dragged"
    elif slip >= ANCHOR_CREEP_M:
        outcome = "creeps"
    else:
        outcome = "holds_fast"
    return {"outcome": outcome, "slip_m": slip, "end_speed": speed, "peak_tension": peak, "unstable": bool(top - z0 > P.max_rise(scene.dims)),
            "steps": steps}
