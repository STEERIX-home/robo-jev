"""T8b physics — rigid-body rollouts in MuJoCo (Task T8 §1-2). Labels come from these rollouts, not from the closed-form engine.

Three scene families, each a small MJCF model compiled once per moment; the hidden parameters (friction, mass
distribution, placement error) are written into the compiled model before every rollout (:meth:`Scene.set_hidden`):

* **deck** (:func:`deck_xml`, :func:`run_deck`) — a load standing on a robot deck. The deck is a heavy body (5 t) on two
  slide joints and a yaw hinge that follows a planned trajectory exactly (feed-forward of the planned acceleration plus a
  stiff PD correction, :func:`_track`): straight braking with a jerk-limited deceleration (:func:`brake_profile`) or a
  constant-speed curve whose curvature ramps in and out (:func:`turn_profile`). The load starts at rest on the deck at
  the deck's centre (on the yaw axis). Outcome after the manoeuvre, a fixed settle time and until the load is at rest: ``tip`` (final tilt > 45°),
  ``slide`` (upright, moved more than 2 cm relative to the deck) or ``stay``.
* **stack** (:func:`stack_xml`, :func:`run_stack`) — a box released by a gripper on top of a fixed support with part of it
  sticking out past the support's edge. Outcome after 2 s (and settling): ``fall`` (its centre ended below the support's top, or tilt
  > 30°) or ``stay``.
* **push** (:func:`push_xml`, :func:`run_push`) — a box on the floor pushed by a horizontal bumper bar at a chosen height
  (a heavy body on a slide joint following a smooth velocity profile: forward by the planned travel, then back away 0.30 m so
  nothing holds the box). Outcome: ``tip`` (final tilt > 45°) or ``upright``. The push is displacement-limited: the higher the
  bar, the more travel it takes to pivot the box to its critical angle, so tipping is not monotone in the bar height.

Contact model (all families): timestep 1 ms, ``implicitfast`` integrator, elliptic friction cones with ``impratio`` 10
(no measurable creep under static friction — checked in the tests), contact ``solref`` 0.004 s, ``condim`` 3. The surface
that carries the hidden friction has geom priority 1, so its friction is the pair's friction. Loads are **boxes only**, a
single collision geom with an explicit body inertial: mass, centre of mass at the hidden offset, and the inertia of a uniform
solid box (an approximation: the mass distribution moves the centre of mass, not the inertia tensor). Box-on-box and
box-on-plane use MuJoCo's dedicated colliders; a cylinder standing on the (box) deck goes through the general convex collider,
which launched a sliding canister metres into the air when the deck stopped (with or without ``multiccd``), and plane geoms
are allowed only in static bodies — so T8b has no cylinders. A guard flags any rollout where the load's centre rises more
than geometry allows (``unstable``, :func:`max_rise`); moments with a flagged rollout are dropped.

The load's inertial frame is compiled with a small non-zero offset (:data:`NOMINAL_IPOS`): MuJoCo treats a body whose inertial
frame coincides with its body frame as "same frame" and then ignores later changes of ``body_ipos``.

Every rollout continues after its planned motion until the load is at rest relative to its support (|v| < 2 cm/s, |ω| < 0.05 rad/s,
checked every 0.1 s, at most :data:`SETTLE_MAX` s more), so no outcome is read mid-pivot. Everything is deterministic: the same model,
hidden values and profile give bit-identical rollouts.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence

import mujoco
import numpy as np

__all__ = [
    "DECK_TOP",
    "max_rise",
    "GRAVITY",
    "SLIDE_M",
    "TIMESTEP",
    "TIP_DEG",
    "Scene",
    "brake_profile",
    "deck_xml",
    "inertia",
    "push_profile",
    "push_xml",
    "run_deck",
    "run_push",
    "run_stack",
    "stack_xml",
    "turn_profile",
]

TIMESTEP = 0.001
GRAVITY = 9.81
SOLREF = "0.004 1"
#: Deck: a heavy body (it follows its trajectory regardless of the load) with a 20 m × 20 m top at this height — wide enough that a
#: sliding load never reaches an edge (a long slide is a slide, not a fall; ``off_deck`` stays as a guard and is counted).
DECK_MASS = 5000.0
DECK_IZZ = 4000.0
DECK_HALF = (10.0, 10.0, 0.02)
DECK_Z = 0.30
DECK_TOP = DECK_Z + DECK_HALF[2]
PUSHER_MASS = 800.0
#: PD on top of the feed-forward (acceleration units): ω_n = 20 rad/s, critically damped.
KP, KD = 400.0, 40.0
TIP_DEG = 45.0
FALL_TILT_DEG = 30.0
SLIDE_M = 0.02
#: Bumper bar half thickness (x) and half height (z) — a 2 cm square bar.
BAR_HALF = 0.01
BAR_FRICTION = 0.1
#: Compile-time inertial offset of the load (m) — any non-zero value; replaced by the hidden centre of mass before each rollout.
NOMINAL_IPOS = 0.001


def inertia(shape: str, dims: Sequence[float], mass: float) -> tuple[float, float, float]:
    """Principal inertia of a uniform solid box ``dims = (L, W, H)`` (x, y, z). Loads are boxes only (module docstring)."""
    if shape != "box":
        raise ValueError(f"loads are boxes only (got {shape!r})")
    length, width, height = (float(v) for v in dims)
    return (mass * (width**2 + height**2) / 12, mass * (length**2 + height**2) / 12, mass * (length**2 + width**2) / 12)


def _height(shape: str, dims: Sequence[float]) -> float:
    if shape != "box":
        raise ValueError(f"loads are boxes only (got {shape!r})")
    return float(dims[2])


def _geom(shape: str, dims: Sequence[float], name: str = "load_geom") -> str:
    if shape != "box":
        raise ValueError(f"loads are boxes only (got {shape!r})")
    return f'<geom name="{name}" type="box" size="{dims[0] / 2:.6f} {dims[1] / 2:.6f} {dims[2] / 2:.6f}"/>'


def _header(name: str) -> str:
    return (
        f'<mujoco model="{name}">\n'
        f'  <option timestep="{TIMESTEP}" gravity="0 0 {-GRAVITY}" integrator="implicitfast" cone="elliptic" impratio="10">\n'
        '    <flag multiccd="enable"/>\n'
        "  </option>\n"
        f'  <default><geom solref="{SOLREF}" condim="3" friction="0.5 0.005 0.0001"/></default>\n'
    )


def _load_body(shape: str, dims: Sequence[float], mass: float, pos: Sequence[float]) -> str:
    ixx, iyy, izz = inertia(shape, dims, mass)
    return (
        f'    <body name="load" pos="{pos[0]:.6f} {pos[1]:.6f} {pos[2]:.6f}">\n'
        "      <freejoint/>\n"
        # a non-zero nominal inertial offset: MuJoCo marks a body whose inertial frame equals its body frame as "same frame"/"simple" at
        # compile time and then ignores later changes of body_ipos — the hidden centre of mass is written per rollout (Scene.set_hidden)
        f'      <inertial pos="{NOMINAL_IPOS} {NOMINAL_IPOS} {NOMINAL_IPOS}" mass="{mass:.6f}" diaginertia="{ixx:.8f} {iyy:.8f} {izz:.8f}"/>\n'
        f"      {_geom(shape, dims)}\n"
        "    </body>\n"
    )


def deck_xml(shape: str, dims: Sequence[float], mass: float) -> str:
    """MJCF of the deck scene: the deck (slide x, slide y, yaw hinge) and the load standing at its centre."""
    height = _height(shape, dims)
    return (
        _header("t8b-deck")
        + "  <worldbody>\n"
        f'    <body name="deck" pos="0 0 {DECK_Z}">\n'
        '      <joint name="deck_x" type="slide" axis="1 0 0"/>\n'
        '      <joint name="deck_y" type="slide" axis="0 1 0"/>\n'
        '      <joint name="deck_yaw" type="hinge" axis="0 0 1"/>\n'
        f'      <inertial pos="0 0 0" mass="{DECK_MASS}" diaginertia="{DECK_IZZ / 2} {DECK_IZZ / 2} {DECK_IZZ}"/>\n'
        f'      <geom name="surface" type="box" size="{DECK_HALF[0]} {DECK_HALF[1]} {DECK_HALF[2]}" priority="1"/>\n'
        "    </body>\n"
        + _load_body(shape, dims, mass, (0.0, 0.0, DECK_TOP + height / 2 + 0.0005))
        + "  </worldbody>\n</mujoco>\n"
    )


def stack_xml(dims: Sequence[float], mass: float, support: Sequence[float]) -> str:
    """MJCF of the stack scene: a fixed support box ``support = (length, width, height)`` centred at x = 0 (its right edge at
    x = length / 2), a floor, and the box being placed (position set per rollout)."""
    sl, sw, sh = (float(v) for v in support)
    height = _height("box", dims)
    return (
        _header("t8b-stack")
        + "  <worldbody>\n"
        '    <geom name="floor" type="plane" size="3 3 0.1"/>\n'
        f'    <geom name="surface" type="box" pos="0 0 {sh / 2:.6f}" size="{sl / 2:.6f} {sw / 2:.6f} {sh / 2:.6f}" priority="1"/>\n'
        + _load_body("box", dims, mass, (0.0, 0.0, sh + height / 2 + 0.01))
        + "  </worldbody>\n</mujoco>\n"
    )


def push_xml(dims: Sequence[float], mass: float, push_height: float) -> str:
    """MJCF of the push scene: floor (hidden friction), the box at the origin, and a bumper bar at `push_height` behind it
    on a slide joint along +x."""
    length, width, height = (float(v) for v in dims)
    x0 = -length / 2 - BAR_HALF - 0.002
    return (
        _header("t8b-push")
        + "  <worldbody>\n"
        '    <geom name="surface" type="plane" size="5 5 0.1" priority="1"/>\n'
        f'    <body name="pusher" pos="{x0:.6f} 0 {push_height:.6f}">\n'
        '      <joint name="push_x" type="slide" axis="1 0 0"/>\n'
        f'      <inertial pos="0 0 0" mass="{PUSHER_MASS}" diaginertia="10 10 10"/>\n'
        f'      <geom name="bar" type="box" size="{BAR_HALF} {width / 2 + 0.1:.6f} {BAR_HALF}" priority="2" friction="{BAR_FRICTION} 0.005 0.0001"/>\n'
        "    </body>\n"
        + _load_body("box", dims, mass, (0.0, 0.0, height / 2 + 0.0005))
        + "  </worldbody>\n</mujoco>\n"
    )


class Scene:
    """A compiled MJCF scene with a ``load`` body and a ``surface`` geom (the one that carries the hidden friction)."""

    def __init__(self, xml: str):
        self.xml = xml
        self.model = mujoco.MjModel.from_xml_string(xml)
        self.data = mujoco.MjData(self.model)
        self.load = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "load")
        self.surface = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "surface")
        self.load_qpos = int(self.model.jnt_qposadr[self.model.body_jntadr[self.load]])
        self.load_qvel = int(self.model.jnt_dofadr[self.model.body_jntadr[self.load]])
        load_geom = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "load_geom")
        if int(self.model.geom_type[load_geom]) != int(mujoco.mjtGeom.mjGEOM_BOX):
            raise ValueError("loads are boxes only (MuJoCo's general convex cylinder-on-box contact launched loads in tests)")
        self.dims = tuple(float(2 * v) for v in self.model.geom_size[load_geom])
        if int(self.model.body_sameframe[self.load]) == int(mujoco.mjtSameFrame.mjSAMEFRAME_BODY) or int(self.model.body_simple[self.load]) != 0:
            raise ValueError("the load must compile with an inertial offset (non-zero inertial pos) so the hidden centre of mass applies")

    def set_hidden(self, mu: float, com: Sequence[float]) -> None:
        """Friction of the surface (the pair's friction — the surface has priority) and the load's centre-of-mass offset from its
        geometric centre (body frame), then refresh the constants that depend on them."""
        self.model.geom_friction[self.surface, 0] = float(mu)
        self.model.body_ipos[self.load] = np.asarray(com, dtype=float)
        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_setConst(self.model, self.data)
        mujoco.mj_resetData(self.model, self.data)

    def com_offset(self) -> np.ndarray:
        """World-frame vector from the load's geometric centre to its centre of mass in the current state (a check that the hidden
        centre of mass is in force: equals ``R · body_ipos`` after :func:`mujoco.mj_forward`)."""
        return np.array(self.data.xipos[self.load]) - np.array(self.data.xpos[self.load])

    def tilt_deg(self) -> float:
        """Angle between the load's z axis and the world z axis (degrees)."""
        return math.degrees(math.acos(max(-1.0, min(1.0, float(self.data.xmat[self.load][8])))))


# --------------------------------------------------------------------------
# Planned motions (arrays per step; positions integrate like MuJoCo: v += a·dt, then q += v·dt)
# --------------------------------------------------------------------------


@dataclass
class Profile:
    """Planned trajectory per step: positions ``q`` (n, k), velocities ``v`` (n, k), accelerations ``a`` (n, k), the step where
    the manoeuvre starts and ends."""

    q: np.ndarray
    v: np.ndarray
    a: np.ndarray
    start: int
    end: int

    @property
    def steps(self) -> int:
        return int(self.a.shape[0])


def _integrate(acc: np.ndarray, v0: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray]:
    n, k = acc.shape
    vel = np.empty((n + 1, k))
    pos = np.empty((n + 1, k))
    vel[0], pos[0] = v0, 0.0
    for i in range(n):
        vel[i + 1] = vel[i] + acc[i] * dt
        pos[i + 1] = pos[i] + vel[i + 1] * dt
    return pos, vel


def brake_profile(v0: float, decel: float, *, jerk_time: float = 0.15, pre: float = 0.3, post: float = 2.0, dt: float = TIMESTEP) -> Profile:
    """Straight braking from `v0` (m/s, +x) to rest: deceleration ramps linearly to `decel` (m/s²) over `jerk_time` (shortened to half
    the stop time for very short stops), holds, and ramps back to 0 as the speed reaches 0. `pre` s of steady driving before, `post` s
    stationary after. Columns: x, y, yaw."""
    v0, decel = float(v0), float(decel)
    if v0 <= 0 or decel <= 0:
        raise ValueError("brake profile needs v0 > 0 and decel > 0")
    tj = min(jerk_time, 0.5 * v0 / decel)
    th = v0 / decel - tj
    n_pre, n_j, n_h, n_post = (int(round(t / dt)) for t in (pre, tj, th, post))
    ramp = (np.arange(n_j) + 0.5) / max(n_j, 1)
    acc_x = np.concatenate([np.zeros(n_pre), -decel * ramp, -decel * np.ones(n_h), -decel * ramp[::-1], np.zeros(n_post)])
    acc = np.zeros((acc_x.size, 3))
    acc[:, 0] = acc_x
    # remove the tiny rounding remainder so the deck ends exactly at rest
    stop = n_pre + 2 * n_j + n_h
    residual = v0 + acc[:stop, 0].sum() * dt
    acc[n_pre:stop, 0] -= residual / max(stop - n_pre, 1) / dt
    pos, vel = _integrate(acc, np.array([v0, 0.0, 0.0]), dt)
    vel[stop + 1 :, 0] = 0.0
    return Profile(q=pos[:-1], v=vel[:-1], a=acc, start=n_pre, end=stop)


def turn_profile(v0: float, radius: float, *, angle_deg: float = 90.0, direction: int = 1, ramp: float = 0.2, pre: float = 0.3, post: float = 1.5,
                 dt: float = TIMESTEP) -> Profile:
    """Constant-speed curve: yaw rate ramps linearly to ``direction · v0 / radius`` over `ramp` s (a clothoid entry), holds, and ramps
    out so the heading turns by `angle_deg`; then straight again for `post` s. ``direction`` +1 = left, −1 = right. Columns: x, y, yaw."""
    v0, radius = float(v0), float(radius)
    if v0 <= 0 or radius <= 0 or direction not in (1, -1):
        raise ValueError("turn profile needs v0 > 0, radius > 0, direction ±1")
    w = v0 / radius
    phi = math.radians(angle_deg)
    tr = min(ramp, 0.4 * phi / w)
    th = phi / w - tr
    n_pre, n_r, n_h, n_post = (int(round(t / dt)) for t in (pre, tr, th, post))
    up = (np.arange(n_r) + 0.5) / max(n_r, 1)
    rate = direction * w * np.concatenate([np.zeros(n_pre), up, np.ones(n_h), up[::-1], np.zeros(n_post)])
    n = rate.size
    heading = np.concatenate([[0.0], np.cumsum(rate) * dt])  # heading after each step
    vel = np.zeros((n + 1, 3))
    vel[:, 0] = v0 * np.cos(heading)
    vel[:, 1] = v0 * np.sin(heading)
    vel[1:, 2] = rate
    vel[0, 2] = 0.0
    acc = (vel[1:] - vel[:-1]) / dt
    pos, vel2 = _integrate(acc, vel[0], dt)
    return Profile(q=pos[:-1], v=vel2[:-1], a=acc, start=n_pre, end=n_pre + 2 * n_r + n_h)


#: After the push the robot backs the bar away by this much (m) at :data:`RETREAT_SPEED`, so a box left leaning on the bar settles on its
#: own — the outcome is where it ends up once nothing holds it.
RETREAT = 0.30
RETREAT_SPEED = 0.25


def push_profile(speed: float, travel: float, *, ramp: float = 0.3, pre: float = 0.2, pause: float = 0.3, retreat: float = RETREAT,
                 post: float = 1.0, dt: float = TIMESTEP) -> Profile:
    """Bumper motion along x: rest for `pre` s, smooth (raised-cosine) speed-up to `speed` over `ramp` s, cruise, smooth stop, so the
    bar moves `travel` m forward; `pause` s still; then back by `retreat` m the same way at :data:`RETREAT_SPEED`; then `post` s at rest.
    One column (x)."""
    speed, travel = float(speed), float(travel)
    if speed <= 0 or travel <= speed * ramp:
        raise ValueError("push profile needs speed > 0 and travel longer than the two half-ramps")

    def leg(v: float, distance: float) -> np.ndarray:
        hold = (distance - v * ramp) / v
        n_r, n_h = int(round(ramp / dt)), int(round(hold / dt))
        s = (np.arange(n_r) + 0.5) / max(n_r, 1)
        up = 0.5 * (1 - np.cos(np.pi * s))
        return v * np.concatenate([up, np.ones(n_h), up[::-1]])

    n_pre, n_pause, n_post = (int(round(t / dt)) for t in (pre, pause, post))
    forward = leg(speed, travel)
    back = -leg(RETREAT_SPEED, retreat) if retreat > 0 else np.zeros(0)
    vel_target = np.concatenate([np.zeros(n_pre), forward, np.zeros(n_pause), back, np.zeros(n_post)])
    acc = np.diff(np.concatenate([[0.0], vel_target])) / dt
    pos, vel = _integrate(acc[:, None], np.zeros(1), dt)
    return Profile(q=pos[:-1], v=vel[:-1], a=acc[:, None], start=n_pre, end=n_pre + forward.size)


def _track(data: Any, profile: Profile, i: int, dofs: Sequence[int], masses: Sequence[float]) -> None:
    """Feed-forward of the planned acceleration plus PD on the planned position/velocity, as generalized forces on `dofs`."""
    for column, (dof, mass) in enumerate(zip(dofs, masses)):
        error = profile.q[i, column] - data.qpos[dof]
        rate = profile.v[i, column] - data.qvel[dof]
        data.qfrc_applied[dof] = mass * (profile.a[i, column] + KP * error + KD * rate)


# --------------------------------------------------------------------------
# Rollouts
# --------------------------------------------------------------------------

#: After the planned motion, keep simulating until the load is at rest (checked every 0.1 s), at most this long.
SETTLE_MAX = 4.0
SETTLE_LIN = 0.02
SETTLE_ANG = 0.05


def _settled(scene: "Scene", frame_velocity: Sequence[float] = (0.0, 0.0, 0.0)) -> bool:
    """Load at rest relative to its support: linear speed (minus the support's) < 2 cm/s and angular speed < 0.05 rad/s."""
    v = np.asarray(scene.data.qvel[scene.load_qvel : scene.load_qvel + 6], dtype=float)
    return bool(np.linalg.norm(v[:3] - np.asarray(frame_velocity)) < SETTLE_LIN and np.linalg.norm(v[3:]) < SETTLE_ANG)


def _settle(scene: "Scene", step: Any, frame_velocity: Any = lambda: (0.0, 0.0, 0.0)) -> tuple[int, float]:
    """Step (``step()`` applies any drive and advances one step) until :func:`_settled` or :data:`SETTLE_MAX`; returns the extra steps and the
    highest load centre seen."""
    top = float(scene.data.xpos[scene.load][2])
    for i in range(int(round(SETTLE_MAX / TIMESTEP))):
        if i % 100 == 0 and _settled(scene, frame_velocity()):
            return i, top
        step()
        if i % 10 == 0:
            top = max(top, float(scene.data.xpos[scene.load][2]))
    return int(round(SETTLE_MAX / TIMESTEP)), top


def max_rise(dims: Sequence[float]) -> float:
    """Largest physically possible rise of a box's centre when it rotates onto an edge, plus 5 cm — anything above it is a numerical
    artefact (``unstable``)."""
    length, width, height = (float(v) for v in dims)
    return math.hypot(max(length, width) / 2, height / 2) - height / 2 + 0.05


def run_deck(scene: Scene, profile: Profile, *, mu: float, com: Sequence[float]) -> dict[str, Any]:
    """One deck rollout → ``{"outcome": stay|slide|tip, "slide_m", "final_tilt", "max_tilt", "off_deck", "unstable"}``."""
    scene.set_hidden(mu, com)
    model, data = scene.model, scene.data
    data.qvel[0:3] = profile.v[0]
    data.qvel[scene.load_qvel : scene.load_qvel + 3] = (profile.v[0, 0], profile.v[0, 1], 0.0)
    mujoco.mj_forward(model, data)
    start_rel = data.xpos[scene.load][:2] - data.qpos[0:2]
    z0, top_z = float(data.xpos[scene.load][2]), float(data.xpos[scene.load][2])
    max_tilt = 0.0
    for i in range(profile.steps):
        _track(data, profile, i, (0, 1, 2), (DECK_MASS, DECK_MASS, DECK_IZZ))
        mujoco.mj_step(model, data)
        if i % 10 == 0:
            max_tilt = max(max_tilt, scene.tilt_deg())
            top_z = max(top_z, float(data.xpos[scene.load][2]))
    last_q, last_v, n = profile.q[-1] + profile.v[-1] * TIMESTEP, profile.v[-1].copy(), [0]
    last_v[2] = 0.0 if abs(last_v[2]) < 1e-9 else last_v[2]

    def hold() -> None:
        n[0] += 1
        for column in range(3):
            error = last_q[column] + last_v[column] * TIMESTEP * (n[0] - 1) - data.qpos[column]
            data.qfrc_applied[column] = (DECK_IZZ if column == 2 else DECK_MASS) * (KP * error + KD * (last_v[column] - data.qvel[column]))
        mujoco.mj_step(model, data)

    extra, top = _settle(scene, hold, lambda: (float(data.qvel[0]), float(data.qvel[1]), 0.0))
    top_z = max(top_z, top)
    tilt = scene.tilt_deg()
    max_tilt = max(max_tilt, tilt)
    yaw = float(data.qpos[2])
    rel = data.xpos[scene.load][:2] - data.qpos[0:2]
    c, s = math.cos(-yaw), math.sin(-yaw)
    rel_deck = np.array([c * rel[0] - s * rel[1], s * rel[0] + c * rel[1]]) - start_rel
    slide = float(np.hypot(*rel_deck))
    off_deck = bool(data.xpos[scene.load][2] < DECK_TOP - 0.05)
    outcome = "tip" if (tilt > TIP_DEG or off_deck) else ("slide" if slide > SLIDE_M else "stay")
    unstable = bool(top_z - z0 > max_rise(scene.dims))
    return {"outcome": outcome, "slide_m": slide, "final_tilt": tilt, "max_tilt": max_tilt, "off_deck": off_deck, "unstable": unstable,
            "settle_steps": extra}


def run_stack(scene: Scene, *, x_centre: float, drop: float, tilt0_deg: float, mu: float, com: Sequence[float], support_height: float,
              box_height: float, seconds: float = 2.0) -> dict[str, Any]:
    """One release on the support → ``{"outcome": stay|fall, "final_tilt", "final_z", "max_tilt"}``. The box starts at rest with its
    centre at `x_centre`, its bottom `drop` m above the support's top, pitched by `tilt0_deg` about y."""
    scene.set_hidden(mu, com)
    model, data = scene.model, scene.data
    q = scene.load_qpos
    half = math.radians(float(tilt0_deg)) / 2
    data.qpos[q : q + 3] = (float(x_centre), 0.0, support_height + box_height / 2 + float(drop))
    data.qpos[q + 3 : q + 7] = (math.cos(half), 0.0, math.sin(half), 0.0)
    mujoco.mj_forward(model, data)
    z0 = top_z = float(data.xpos[scene.load][2])
    max_tilt = 0.0
    for i in range(int(round(seconds / TIMESTEP))):
        mujoco.mj_step(model, data)
        if i % 10 == 0:
            max_tilt = max(max_tilt, scene.tilt_deg())
            top_z = max(top_z, float(data.xpos[scene.load][2]))
    extra, top = _settle(scene, lambda: mujoco.mj_step(model, data))
    top_z = max(top_z, top)
    tilt = scene.tilt_deg()
    z = float(data.xpos[scene.load][2])
    fell = z < support_height - 0.01 or tilt > FALL_TILT_DEG
    # pivoting on the support's edge can lift the box's centre by up to its half-diagonal minus half-height (more when the weight sits at
    # the outer end); anything above that bound is a numerical artefact
    return {"outcome": "fall" if fell else "stay", "final_tilt": tilt, "final_z": z, "max_tilt": max(max_tilt, tilt),
            "unstable": bool(top_z - z0 > max_rise(scene.dims)), "settle_steps": extra}


def run_push(scene: Scene, profile: Profile, *, mu: float, com: Sequence[float]) -> dict[str, Any]:
    """One push (forward, pause, back away) → ``{"outcome": upright|tip, "final_tilt", "max_tilt", "moved_m", "unstable", "settle_steps"}``."""
    scene.set_hidden(mu, com)
    model, data = scene.model, scene.data
    mujoco.mj_forward(model, data)
    x0 = float(data.xpos[scene.load][0])
    z0 = top_z = float(data.xpos[scene.load][2])
    max_tilt = 0.0
    for i in range(profile.steps):
        _track(data, profile, i, (0,), (PUSHER_MASS,))
        mujoco.mj_step(model, data)
        if i % 10 == 0:
            max_tilt = max(max_tilt, scene.tilt_deg())
            top_z = max(top_z, float(data.xpos[scene.load][2]))
    last = float(profile.q[-1, 0] + profile.v[-1, 0] * TIMESTEP)

    def hold() -> None:
        data.qfrc_applied[0] = PUSHER_MASS * (KP * (last - data.qpos[0]) - KD * data.qvel[0])
        mujoco.mj_step(model, data)

    extra, top = _settle(scene, hold)
    top_z = max(top_z, top)
    tilt = scene.tilt_deg()
    return {"outcome": "tip" if tilt > TIP_DEG else "upright", "final_tilt": tilt, "max_tilt": max(max_tilt, tilt),
            "moved_m": float(data.xpos[scene.load][0]) - x0, "unstable": bool(top_z - z0 > max_rise(scene.dims)), "settle_steps": extra}
