"""Author A2 — MuJoCo scenes for the three laws (v0.6 author-diversity experiment). Geometry and drives are this author's own; the
contact defaults, the load body with a writable centre of mass, the tracking helper and the settle check come from
:mod:`robo_jev.outside.physics` (the shared low-level helpers the brief allows).

Three scenes, one per law:

* **turntable** (law A, tip or slide under acceleration) — a box-shaped item (a square bottle, a carafe, a boxy vase ...) stands on
  a lazy Susan at a stated distance from its centre. The robot turns the top through a stated angle in a stated time with a smooth
  (raised-cosine) speed profile; the item feels the tangential and the centripetal acceleration of its spot. The top is a heavy body
  on a yaw hinge that follows the planned angle exactly (feed-forward of the planned angular acceleration plus a stiff PD). Its
  collision top is a 2 m square, so an item that skids keeps skidding on the top instead of dropping off an edge. Outcome after the
  turn and settling: ``topples`` (final tilt > 45°), ``skids`` (upright, moved more than 2 cm relative to the top) or ``stays``.
* **tiltback** (law B, centre of mass past an edge) — a tall piece of furniture (bookcase, wardrobe, drinks fridge ...) standing on
  the floor is tipped back about its rear bottom edge by the planned angle and held there by the robot's hand on the middle of its
  top front edge: a weightless point tied to the piece by a ``connect`` constraint, moved along x by a position servo whose force is
  limited to the arm's stated limit (either way). The hold starts in contact at the planned tilt, pre-loaded with the static holding
  force, and settles for 0.3 s with an unlimited arm; then the stated limit applies for 3 s. Outcome: ``over`` (tilt > 45°, or the
  hand saturated with the tilt already 2° past the plan), ``hangs`` (the hand pulls: the centre of mass is behind the rear edge and
  the arm holds it) or ``rests`` (the hand pushes, or the piece came back onto its feet).
* **wallpress** (law C, friction-limited holding; rolled with ``refresh_bvh`` since the 2026-10-14 physics audit) — a flat item
  (mirror, framed print, pinboard ...) is pressed flat against a wall
  by a padded paddle with a stated force; nothing else holds it up. The item's back carries the hidden wall friction, the paddle's
  face its own friction. The paddle is on a slide joint normal to the wall (it cannot move up or down) and is pushed with the planned
  force. A slip reflex is part of the plan: once the item has dropped 4 mm, the push ramps to a stated multiple of the planned force
  after 60 ms latency over 150 ms. Outcome after 2 s: ``drops`` (still sliding down faster than 5 cm/s, or slid off the pad),
  ``slips`` (dropped 4 mm or more, or turned more than 3°, then caught) or ``holds``.

Contact model (all scenes): :func:`robo_jev.outside.physics._header` — 1 ms steps, ``implicitfast``, elliptic cones, ``impratio``
10, ``solref`` 0.004 s, ``condim`` 3. The geom named ``surface`` carries the hidden friction (priority 1). Everything is
deterministic: the same inputs give bit-identical rollouts (tested).
"""

from __future__ import annotations

import math
from typing import Any, Sequence

import mujoco
import numpy as np

from robo_jev.outside import physics as P

__all__ = [
    "SLIDE_M",
    "TIP_DEG",
    "TiltbackScene",
    "TurntableScene",
    "WallpressScene",
    "set_mass",
    "spin_profile",
    "turntable_peak_accel",
]

TIMESTEP = P.TIMESTEP
GRAVITY = P.GRAVITY
TIP_DEG = 45.0
SLIDE_M = 0.02

# --------------------------------------------------------------------------
# A — turntable
# --------------------------------------------------------------------------

#: The turntable top: a heavy body on a yaw hinge; its collision top is a 2 m square (see module docstring).
TT_TOP_HALF = (1.0, 1.0, 0.012)
TT_Z = 0.75
TT_TOP = TT_Z + TT_TOP_HALF[2]
TT_MASS = 60.0
TT_IZZ = 20.0
#: PD on top of the feed-forward (angular acceleration units, rad/s² per rad and per rad/s).
TT_KP, TT_KD = 900.0, 60.0
#: Steps at rest before the turn starts.
TT_PRE = 0.15


def spin_profile(angle: float, duration: float, *, pre: float = TT_PRE, dt: float = TIMESTEP) -> P.Profile:
    """Raised-cosine speed profile turning the top by `angle` rad in `duration` s: ω(t) = (Φ/T)(1 − cos 2πt/T), so the speed and the
    acceleration both start and end at zero. One column (the hinge angle). Positions integrate like MuJoCo (v += a·dt, q += v·dt)."""
    angle, duration = float(angle), float(duration)
    if duration <= 0 or angle == 0:
        raise ValueError("spin profile needs a non-zero angle and a positive duration")
    n_pre, n_turn = int(round(pre / dt)), int(round(duration / dt))
    t = (np.arange(n_turn) + 0.5) * dt
    acc_turn = (2 * math.pi * angle / duration**2) * np.sin(2 * math.pi * t / duration)
    acc = np.concatenate([np.zeros(n_pre), acc_turn])[:, None]
    vel = np.empty((acc.shape[0] + 1, 1))
    pos = np.empty((acc.shape[0] + 1, 1))
    vel[0], pos[0] = 0.0, 0.0
    for i in range(acc.shape[0]):
        vel[i + 1] = vel[i] + acc[i] * dt
        pos[i + 1] = pos[i] + vel[i + 1] * dt
    return P.Profile(q=pos[:-1], v=vel[:-1], a=acc, start=n_pre, end=n_pre + n_turn)


def turntable_peak_accel(angle: float, duration: float, radius: float) -> float:
    """Largest horizontal acceleration (m/s²) of a point at `radius` under :func:`spin_profile` — the tangential α·r and the centripetal
    ω²·r together (the static model the aim uses)."""
    s = np.linspace(0.0, 1.0, 2001)
    w = (angle / duration) * (1 - np.cos(2 * math.pi * s))
    a = (2 * math.pi * angle / duration**2) * np.sin(2 * math.pi * s)
    return float(np.max(np.hypot(a * radius, w**2 * radius)))


def turntable_xml(dims: Sequence[float], mass: float, radius: float) -> str:
    """MJCF: the turntable (yaw hinge, a 2 m square collision top at :data:`TT_TOP`), a floor, and the item standing at (`radius`, 0)
    with its sides along the radial (x) and tangential (y) directions."""
    height = float(dims[2])
    return (
        P._header("a2-turntable")
        + "  <worldbody>\n"
        '    <geom name="floor" type="plane" size="4 4 0.1"/>\n'
        f'    <body name="top" pos="0 0 {TT_Z}">\n'
        '      <joint name="spin" type="hinge" axis="0 0 1"/>\n'
        f'      <inertial pos="0 0 0" mass="{TT_MASS}" diaginertia="{TT_IZZ / 2} {TT_IZZ / 2} {TT_IZZ}"/>\n'
        f'      <geom name="surface" type="box" size="{TT_TOP_HALF[0]} {TT_TOP_HALF[1]} {TT_TOP_HALF[2]}" priority="1"/>\n'
        "    </body>\n"
        + P._load_body("box", dims, mass, (float(radius), 0.0, TT_TOP + height / 2 + 0.0005))
        + "  </worldbody>\n</mujoco>\n"
    )


class TurntableScene:
    """One compiled turntable model per (item dims, mass); the radius is set per rollout (the item's start position)."""

    def __init__(self, dims: Sequence[float], mass: float):
        self.dims = tuple(float(v) for v in dims)
        self.scene = P.Scene(turntable_xml(self.dims, mass, 0.3))
        self.hinge = int(self.scene.model.jnt_dofadr[mujoco.mj_name2id(self.scene.model, mujoco.mjtObj.mjOBJ_JOINT, "spin")])
        self.hinge_q = int(self.scene.model.jnt_qposadr[mujoco.mj_name2id(self.scene.model, mujoco.mjtObj.mjOBJ_JOINT, "spin")])

    def run(self, *, angle: float, duration: float, radius: float, mu: float, com: Sequence[float]) -> dict[str, Any]:
        sc = self.scene
        sc.set_hidden(mu, com)
        model, data = sc.model, sc.data
        q = sc.load_qpos
        data.qpos[q : q + 3] = (float(radius), 0.0, TT_TOP + self.dims[2] / 2 + 0.0005)
        data.qpos[q + 3 : q + 7] = (1.0, 0.0, 0.0, 0.0)
        mujoco.mj_forward(model, data)
        profile = spin_profile(angle, duration)
        z0 = top_z = float(data.xpos[sc.load][2])
        max_tilt = 0.0
        h, hq = self.hinge, self.hinge_q
        for i in range(profile.steps):
            error = profile.q[i, 0] - data.qpos[hq]
            rate = profile.v[i, 0] - data.qvel[h]
            data.qfrc_applied[h] = TT_IZZ * (profile.a[i, 0] + TT_KP * error + TT_KD * rate)
            mujoco.mj_step(model, data)
            if i % 10 == 0:
                max_tilt = max(max_tilt, sc.tilt_deg())
                top_z = max(top_z, float(data.xpos[sc.load][2]))
        final = float(profile.q[-1, 0] + profile.v[-1, 0] * TIMESTEP)

        def hold() -> None:
            data.qfrc_applied[h] = TT_IZZ * (TT_KP * (final - data.qpos[hq]) - TT_KD * data.qvel[h])
            mujoco.mj_step(model, data)

        extra, top = P._settle(sc, hold)
        top_z = max(top_z, top)
        tilt = sc.tilt_deg()
        yaw = float(data.qpos[hq])
        x, y = float(data.xpos[sc.load][0]), float(data.xpos[sc.load][1])
        c, s = math.cos(-yaw), math.sin(-yaw)
        rel = (c * x - s * y - float(radius), s * x + c * y)
        moved = float(math.hypot(*rel))
        outcome = "topples" if tilt > TIP_DEG else ("skids" if moved > SLIDE_M else "stays")
        return {"outcome": outcome, "moved_m": moved, "final_tilt": tilt, "max_tilt": max(max_tilt, tilt),
                "unstable": bool(top_z - z0 > P.max_rise(self.dims)), "settle_steps": extra}


# --------------------------------------------------------------------------
# B — tilt back and hold
# --------------------------------------------------------------------------

#: The hold lasts this long; the outcome is read at its end.
TB_SECONDS = 3.0
#: The hand's position servo along x (N/m, N·s/m), its force limited to the arm's stated limit.
TB_KP, TB_KV = 4000.0, 600.0
#: The force the hand must exert to call the piece "past its balance point": a pull towards the robot, averaged over the last 0.5 s.
TB_PULL_MIN = 0.5
#: A saturated hand with the tilt this far (degrees) past the plan at the end counts as "over" (it is still going).
TB_AWAY_DEG = 2.0
#: Contact settling before the hold (s): the arm is unlimited, so the hold starts from equilibrium at the planned tilt.
TB_SETTLE = 0.3


def tiltback_xml(dims: Sequence[float], mass: float, limit: float) -> str:
    """MJCF: the floor (``surface``, hidden friction), the piece (``load``, a box: depth x, width y, height z) and the robot's hand — a
    weightless point (gravity compensated) on two slide joints (x, z) tied to the middle of the piece's top front edge by a ``connect``
    constraint; a position servo on the hand's x joint holds the planned tilt with its force limited to `limit` N either way. The hand
    is compiled at that edge of the upright piece (a ``connect`` constraint is satisfied in the compiled configuration)."""
    depth, _, height = (float(v) for v in dims)
    return (
        P._header("a2-tiltback")
        + "  <worldbody>\n"
        '    <geom name="surface" type="plane" size="6 6 0.1" priority="1"/>\n'
        f'    <body name="hand" pos="{depth / 2:.6f} 0 {height + 0.0005:.6f}" gravcomp="1">\n'
        '      <joint name="hand_x" type="slide" axis="1 0 0"/>\n'
        '      <joint name="hand_z" type="slide" axis="0 0 1"/>\n'
        '      <inertial pos="0 0 0" mass="0.01" diaginertia="1e-6 1e-6 1e-6"/>\n'
        "    </body>\n"
        + P._load_body("box", dims, mass, (0.0, 0.0, height / 2 + 0.0005))
        + "  </worldbody>\n"
        "  <equality>\n"
        f'    <connect body1="load" body2="hand" anchor="{depth / 2:.6f} 0 {height / 2:.6f}" solref="{P.SOLREF}"/>\n'
        "  </equality>\n"
        "  <actuator>\n"
        f'    <position name="hold" joint="hand_x" kp="{TB_KP}" kv="{TB_KV}" forcelimited="true" forcerange="{-limit:.6f} {limit:.6f}"/>\n'
        "  </actuator>\n"
        "</mujoco>\n"
    )


def _quat_y(angle: float) -> tuple[float, float, float, float]:
    half = angle / 2
    return (math.cos(half), 0.0, math.sin(half), 0.0)


class TiltbackScene:
    """One compiled model per (dims, mass); the arm's force limit is written per rollout. The piece is tipped back by the planned
    angle about its rear bottom edge (the edge at x = −depth/2; the top moves towards −x, away from the robot standing at +x) and held
    there by the robot's hand on the middle of its top front edge (module docstring). Outcome after :data:`TB_SECONDS`: ``over``
    (tilt > 45°: it got away), ``hangs`` (the hand is pulling: the centre of mass is behind the rear edge and the arm holds it) or
    ``rests`` (the hand pushes, or the piece came back down onto its feet)."""

    def __init__(self, dims: Sequence[float], mass: float):
        self.dims = tuple(float(v) for v in dims)
        self.scene = P.Scene(tiltback_xml(self.dims, mass, 100.0))
        model = self.scene.model
        self.hand_x = int(model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "hand_x")])
        self.hand_z = int(model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "hand_z")])
        self.actuator = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "hold")

    def pose(self, tilt_deg: float) -> tuple[np.ndarray, tuple[float, float, float, float], np.ndarray]:
        """Centre position, orientation and grip point (middle of the top front edge) of the piece tipped back by `tilt_deg` about its
        rear bottom edge (resting on that edge)."""
        depth, _, height = self.dims
        phi = -math.radians(float(tilt_deg))  # rotation about +y; negative moves the top towards −x
        c, s = math.cos(phi), math.sin(phi)
        pivot = np.array([-depth / 2, 0.0, -0.0001])  # the rear edge already bears on the floor (0.1 mm into the soft contact)

        def place(rel: Sequence[float]) -> np.ndarray:
            return pivot + np.array([c * rel[0] + s * rel[2], 0.0, -s * rel[0] + c * rel[2]])

        return place((depth / 2, 0.0, height / 2)), _quat_y(phi), place((depth, 0.0, height))

    def run(self, *, tilt_deg: float, limit: float, mu: float, com: Sequence[float], mass: float | None = None,
            seconds: float = TB_SECONDS) -> dict[str, Any]:
        """Hold the piece at `tilt_deg` (the actual angle: plan plus error) with at most `limit` N at the hand; `mass` (kg) replaces the
        compiled mass (a hidden quantity)."""
        sc = self.scene
        if mass is not None:
            set_mass(sc, self.dims, mass)
        sc.set_hidden(mu, com)
        model, data = sc.model, sc.data
        q = sc.load_qpos
        centre, quat, grip = self.pose(tilt_deg)
        data.qpos[q : q + 3] = centre
        data.qpos[q + 3 : q + 7] = quat
        # the hand's joints measure from its compiled place, the middle of the top front edge of the upright piece
        data.qpos[self.hand_x], data.qpos[self.hand_z] = grip[0] - self.dims[0] / 2, grip[2] - (self.dims[2] + 0.0005)
        mujoco.mj_forward(model, data)
        # the robot has tipped it there and is already holding it: the servo starts loaded with the static holding force (moment
        # balance about the rear edge) and the contact settles for TB_SETTLE s with an unlimited arm; then the arm's limit applies, so
        # the hold starts from equilibrium at the planned tilt (no start-up transient)
        pivot = np.array([-self.dims[0] / 2, 0.0, 0.0])
        com_w = np.array(data.xipos[sc.load])
        weight = float(model.body_mass[sc.load]) * GRAVITY
        hold = -weight * (com_w[0] - pivot[0]) / max(grip[2] - pivot[2], 1e-6)
        data.ctrl[self.actuator] = data.qpos[self.hand_x] + hold / TB_KP
        model.actuator_forcerange[self.actuator] = (-1e5, 1e5)
        mujoco.mj_step(model, data, nstep=int(round(TB_SETTLE / TIMESTEP)))
        model.actuator_forcerange[self.actuator] = (-float(limit), float(limit))
        mujoco.mj_forward(model, data)
        z0 = top_z = float(data.xpos[sc.load][2])
        chunk = 10
        n = int(round(seconds / TIMESTEP)) // chunk
        tail = int(round(0.5 / TIMESTEP)) // chunk
        pulls: list[float] = []
        max_tilt = 0.0
        for i in range(n):
            mujoco.mj_step(model, data, nstep=chunk)
            if i >= n - tail:
                pulls.append(float(data.actuator_force[self.actuator]))
            max_tilt = max(max_tilt, sc.tilt_deg())
            top_z = max(top_z, float(data.xpos[sc.load][2]))
        tilt = sc.tilt_deg()
        pull = float(np.mean(pulls)) if pulls else 0.0
        # got away: past 45°, or the hand is at its limit and the tilt has already grown 2° past the plan (it is on its way over)
        saturated = pull >= 0.98 * float(limit) and tilt > float(tilt_deg) + TB_AWAY_DEG
        if tilt > TIP_DEG or saturated:
            outcome = "over"
        elif pull > TB_PULL_MIN:
            outcome = "hangs"
        else:
            outcome = "rests"
        return {"outcome": outcome, "pull_n": pull, "final_tilt": tilt, "max_tilt": max(max_tilt, tilt),
                "unstable": bool(top_z - z0 > P.max_rise(self.dims))}


# --------------------------------------------------------------------------
# C — press a flat item against a wall
# --------------------------------------------------------------------------

#: Paddle: a square pad (half sizes x, y, z) on a slide joint normal to the wall, this heavy.
PAD_HALF = (0.01, 0.06, 0.06)
PAD_MASS = 1.5
WP_Z = 1.4
WP_SECONDS = 2.0
#: Slip reflex: triggers once the item has dropped this far, acts after the latency, ramps over the ramp time.
REFLEX_DROP = 0.004
REFLEX_LATENCY = 0.06
REFLEX_RAMP = 0.15
SLIP_DROP = 0.004
SLIP_TURN_DEG = 3.0
#: "drops": still sliding down faster than this (m/s) when the hold ends, or slid off the pad.
DROP_SPEED = 0.05


def wallpress_xml(dims: Sequence[float], mass: float, pad_mu: float) -> str:
    """MJCF: the wall (``surface``: a box whose face is the plane x = 0, hidden friction), the item (``load``: thickness x, width y,
    height z) with its back on the wall, and the paddle (slide joint along x, priority 2, its own friction) on the item's face."""
    thick, width, height = (float(v) for v in dims)
    return (
        P._header("a2-wallpress")
        + "  <worldbody>\n"
        '    <geom name="floor" type="plane" size="4 4 0.1" pos="0 0 0"/>\n'
        f'    <geom name="surface" type="box" pos="-0.05 0 {WP_Z}" size="0.05 2 1.4" priority="1"/>\n'
        f'    <body name="paddle" pos="{thick + PAD_HALF[0] + 0.0002:.6f} 0 {WP_Z}">\n'
        '      <joint name="press" type="slide" axis="1 0 0"/>\n'
        f'      <inertial pos="0 0 0" mass="{PAD_MASS}" diaginertia="0.01 0.01 0.01"/>\n'
        f'      <geom name="pad" type="box" size="{PAD_HALF[0]} {PAD_HALF[1]} {PAD_HALF[2]}" priority="2" friction="{pad_mu:.6f} 0.005 0.0001"/>\n'
        "    </body>\n"
        + P._load_body("box", dims, mass, (thick / 2 + 0.0002, 0.0, WP_Z))
        + "  </worldbody>\n</mujoco>\n"
    )


class WallpressScene:
    """One compiled model per (dims, mass, pad friction); the paddle offset (placement) and the force are set per rollout.

    ``refresh_bvh`` (default on since the physics audit of 2026-10-14): the world body here holds two geoms (floor and wall), so the
    load–world pair goes through MuJoCo's midphase, which reads the load's bounding volumes in its compile-time inertial frame; every
    hidden centre-of-mass write therefore also re-expresses them (:func:`robo_jev.outside.physics.refresh_bvh`). ``False`` reproduces the
    labels stored before the fix (``artifacts/v06/a2/work/labels-wallpress.jsonl``). The turntable and tiltback scenes have no
    midphase pair (:meth:`robo_jev.outside.physics.Scene.midphase_pairs` is empty) and are left as they were."""

    def __init__(self, dims: Sequence[float], mass: float, pad_mu: float, *, refresh_bvh: bool = True):
        self.dims = tuple(float(v) for v in dims)
        self.mass = float(mass)
        self.scene = P.Scene(wallpress_xml(self.dims, mass, pad_mu), refresh_bvh=refresh_bvh)
        model = self.scene.model
        self.press = int(model.jnt_dofadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "press")])
        self.press_q = int(model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "press")])
        self.paddle = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "paddle")
        self.pad = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "pad")

    def run(self, *, force: float, reflex: float, mu: float, offset: Sequence[float] = (0.0, 0.0),
            com: Sequence[float] = (0.0, 0.0, 0.0), mass: float | None = None, pad_mu: float | None = None,
            seconds: float = WP_SECONDS) -> dict[str, Any]:
        """`offset` = (y, z) of the item's centre relative to the paddle's centre (the paddle stays at the wall position it was given;
        the item is placed off it). The push starts at the planned `force` (N) and the reflex can raise it to `reflex` × force. `mass`
        (kg) and `pad_mu` replace the compiled values (hidden quantities)."""
        sc = self.scene
        if mass is not None:
            set_mass(sc, self.dims, mass)
        if pad_mu is not None:
            sc.model.geom_friction[self.pad, 0] = float(pad_mu)
        sc.set_hidden(mu, com)
        model, data = sc.model, sc.data
        q = sc.load_qpos
        thick = self.dims[0]
        data.qpos[q : q + 3] = (thick / 2 + 0.0002, float(offset[0]), WP_Z + float(offset[1]))
        data.qpos[q + 3 : q + 7] = (1.0, 0.0, 0.0, 0.0)
        mujoco.mj_forward(model, data)
        z0 = float(data.xipos[sc.load][2])
        trigger = None
        max_drop = 0.0
        n = int(round(seconds / TIMESTEP))
        for i in range(n):
            t = i * TIMESTEP
            drop = z0 - float(data.xipos[sc.load][2])
            max_drop = max(max_drop, drop)
            if trigger is None and drop >= REFLEX_DROP:
                trigger = t
            push = float(force)
            if trigger is not None and t > trigger + REFLEX_LATENCY:
                push *= 1.0 + (float(reflex) - 1.0) * min(1.0, (t - trigger - REFLEX_LATENCY) / REFLEX_RAMP)
            data.qfrc_applied[self.press] = -push
            mujoco.mj_step(model, data)
        drop = z0 - float(data.xipos[sc.load][2])
        rot = np.array(data.xmat[sc.load]).reshape(3, 3)
        turn = math.degrees(math.acos(max(-1.0, min(1.0, float(rot[2, 2])))))  # z axis away from vertical (in-plane swing or peel)
        falling = float(data.qvel[sc.load_qvel + 2]) < -DROP_SPEED
        off_pad = drop > self.dims[2] / 2 + PAD_HALF[2]
        if falling or off_pad:
            outcome = "drops"
        elif drop >= SLIP_DROP or turn > SLIP_TURN_DEG:
            outcome = "slips"
        else:
            outcome = "holds"
        return {"outcome": outcome, "drop_m": drop, "turn_deg": turn, "reflex_at": trigger, "max_drop": max(max_drop, drop), "falling": falling}


def set_mass(scene: P.Scene, dims: Sequence[float], mass: float) -> None:
    """Write a new mass (and the matching uniform-box inertia) into the compiled load; :meth:`robo_jev.outside.physics.Scene.set_hidden`
    then refreshes the derived constants (``mj_setConst``)."""
    scene.model.body_mass[scene.load] = float(mass)
    scene.model.body_inertia[scene.load] = P.inertia("box", dims, float(mass))
