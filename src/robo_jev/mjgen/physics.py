"""v0.3b physics — rigid-body rollouts in MuJoCo for three **training** families (Task v0.3b §1).

The contact model, time step, tracking controller and settling rule are T8's (:mod:`robo_jev.outside.physics`: 1 ms step,
``implicitfast``, elliptic cones with ``impratio`` 10, ``solref`` 0.004 s, boxes only, loads compiled with a non-zero nominal
inertial offset so a hidden centre of mass applies). The scenes are new:

* **tilt** (:func:`tilt_xml`, :func:`run_tilt`) — a box on a flat deck that pitches about a level axis under the box
  (a tilting cargo bed, or a deck pitching onto a ramp). The deck (hinge, heavy) follows a smooth pitch profile to the planned
  angle and holds it (:func:`pitch_profile`). The box may stand upright or lie on a side (:data:`ORIENTATIONS`). Outcome once the
  deck holds still and the box is at rest relative to it: ``tip`` (the box's original up axis more than 45° from the deck
  normal), ``slide`` (upright, moved more than :data:`TILT_SLIDE_M` along the deck) or ``stay``.
* **grip** (:func:`grip_xml`, :func:`run_grip`) — a box squeezed between two finger pads (each finger on a slide joint pushed
  inward with the planned squeeze force), the hand on two slide joints following a lift, a lowering or a sideways swing
  (:func:`move_profile`). Outcome: ``drop`` (the box left the pads), ``slip`` (still held, moved at least
  :data:`GRIP_SLIP_M` against the pads) or ``hold``.
* **tower** (:func:`tower_xml`, :func:`run_tower`) — two or three boxes stacked on the floor with planned fore-aft offsets;
  the top box is released a little above the box under it. Outcome: ``stands`` (no box fell), ``top`` (only the top box
  fell) or ``more`` (a lower box went over too). A box has fallen when it tilted more than 10° (a box resting flat on the one below stays under 1°; one that went
  over the edge may come to rest leaning on the floor at ≈ 20°) or dropped by half its
  height.

Everything is deterministic: the same model, hidden values and profile give bit-identical rollouts (tested).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence

import mujoco
import numpy as np

from robo_jev.outside import physics as t8

__all__ = [
    "GRAVITY",
    "GRIP_OUTCOMES",
    "GRIP_SLIP_M",
    "ORIENTATIONS",
    "TILT_OUTCOMES",
    "TILT_SLIDE_M",
    "TOWER_OUTCOMES",
    "Rig",
    "grip_xml",
    "move_profile",
    "orientation_frame",
    "pitch_profile",
    "run_grip",
    "run_tilt",
    "run_tower",
    "tilt_xml",
    "tower_xml",
]

TIMESTEP = t8.TIMESTEP
GRAVITY = t8.GRAVITY
NOMINAL_IPOS = t8.NOMINAL_IPOS
KP, KD = t8.KP, t8.KD

TILT_OUTCOMES = ("stay", "slide", "tip")
GRIP_OUTCOMES = ("hold", "slip", "drop")
TOWER_OUTCOMES = ("stands", "top", "more")

#: tilt: deck half-size (long along the slope so a slide never reaches an edge before it is read) and the pivot height.
DECK_HALF = (4.0, 1.5, 0.02)
DECK_PIVOT_Z = 0.40
DECK_MASS = 3000.0
DECK_INERTIA = 6000.0
TILT_SLIDE_M = 0.03
TILT_TIP_DEG = 45.0
#: a box that has slid this far while still upright is sliding for good (it keeps going on a held slope; a sliding box only tips
#: when μ·h_c > b, and then it tips at once): stop reading early. After the hold, rollouts continue until the box is at rest.
TILT_SLIDE_STOP = 0.10
#: … but only while the box is within this many degrees of upright: a box that starts to tip moves its centre by rotation
#: (≈ (H/2)·sin φ — 0.10 m at 19.5° for a 0.6 m box), which is not a slide.
TILT_SLIDE_UPRIGHT = 5.0
TILT_HOLD = 0.8

#: grip: the hand (heavy, two slide joints x/z, gravity compensated), finger pads (half sizes: along x, thickness y, height z).
HAND_Z = 1.0
HAND_MASS = 400.0
FINGER_MASS = 0.25
PAD_HALF = (0.03, 0.006, 0.035)
GRIP_SLIP_M = 0.01
GRIP_PRE = 0.4
GRIP_POST = 0.6

#: tower: a box has fallen when it tilts more than this or its centre drops by half its height.
TOWER_FALL_DEG = 10.0
TOWER_SECONDS = 1.2
SETTLE_LIN, SETTLE_ANG = t8.SETTLE_LIN, t8.SETTLE_ANG
SETTLE_MAX = 3.0

#: tilt: box orientations — body rotation (w, x, y, z) taking the box's own axes (L along, W across, H up) to the deck frame.
#: ``upright`` = as listed; ``along`` = laid down with its top end pointing down the slope's +x axis; ``across`` = laid on a side
#: with its top end pointing across the deck (+y).
ORIENTATIONS = {
    "upright": (1.0, 0.0, 0.0, 0.0),
    "along": (math.cos(math.pi / 4), 0.0, math.sin(math.pi / 4), 0.0),
    "across": (math.cos(math.pi / 4), -math.sin(math.pi / 4), 0.0, 0.0),
}


def orientation_frame(name: str) -> np.ndarray:
    """Rotation matrix (body → deck) of an orientation in :data:`ORIENTATIONS`."""
    out = np.zeros(9)
    mujoco.mju_quat2Mat(out, np.asarray(ORIENTATIONS[name], dtype=float))
    return out.reshape(3, 3)


def _box_body(name: str, dims: Sequence[float], mass: float, pos: Sequence[float], quat: Sequence[float] = (1.0, 0.0, 0.0, 0.0),
              *, free: bool = True) -> str:
    """A box body (``dims`` = full sizes along its own x, y, z) with an explicit inertial at a small nominal offset — the hidden
    centre of mass is written per rollout (:meth:`Rig.set_com`; MuJoCo ignores runtime ``body_ipos`` of a 'same frame' body)."""
    ixx, iyy, izz = t8.inertia("box", dims, mass)
    joint = "      <freejoint/>\n" if free else ""
    return (
        f'    <body name="{name}" pos="{pos[0]:.6f} {pos[1]:.6f} {pos[2]:.6f}" quat="{quat[0]:.9f} {quat[1]:.9f} {quat[2]:.9f} {quat[3]:.9f}">\n'
        + joint
        + f'      <inertial pos="{NOMINAL_IPOS} {NOMINAL_IPOS} {NOMINAL_IPOS}" mass="{mass:.6f}" diaginertia="{ixx:.8f} {iyy:.8f} {izz:.8f}"/>\n'
        f'      <geom name="{name}_geom" type="box" size="{dims[0] / 2:.6f} {dims[1] / 2:.6f} {dims[2] / 2:.6f}"/>\n'
        "    </body>\n"
    )


class Rig:
    """A compiled MJCF scene whose hidden values (friction of named geoms, centre of mass and mass of named boxes) are written
    before every rollout. Refuses non-box loads and loads compiled without an inertial offset (T8 lessons 4 and 5)."""

    def __init__(self, xml: str, *, boxes: Sequence[str], dims: dict[str, Sequence[float]]):
        self.xml = xml
        self.model = mujoco.MjModel.from_xml_string(xml)
        self.data = mujoco.MjData(self.model)
        self.boxes = {}
        for name in boxes:
            body = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
            geom = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, f"{name}_geom")
            if body < 0 or geom < 0:
                raise ValueError(f"box {name!r} missing from the scene")
            if int(self.model.geom_type[geom]) != int(mujoco.mjtGeom.mjGEOM_BOX):
                raise ValueError("loads are boxes only (MuJoCo's general convex cylinder contact launched loads in T8)")
            if int(self.model.body_sameframe[body]) == int(mujoco.mjtSameFrame.mjSAMEFRAME_BODY) or int(self.model.body_simple[body]) != 0:
                raise ValueError(f"{name}: compile with a non-zero inertial pos so the hidden centre of mass applies")
            jnt = int(self.model.body_jntadr[body])
            self.boxes[name] = {"body": body, "geom": geom, "dims": tuple(float(v) for v in dims[name]),
                                "qpos": int(self.model.jnt_qposadr[jnt]), "qvel": int(self.model.jnt_dofadr[jnt])}

    def geom(self, name: str) -> int:
        out = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, name)
        if out < 0:
            raise KeyError(name)
        return out

    def set_hidden(self, *, friction: dict[str, float], com: dict[str, Sequence[float]], mass: dict[str, float] | None = None) -> None:
        """Write the hidden values and refresh the derived constants; the state is reset to the compiled one."""
        for geom, mu in friction.items():
            self.model.geom_friction[self.geom(geom), 0] = float(mu)
        for name, offset in com.items():
            self.model.body_ipos[self.boxes[name]["body"]] = np.asarray(offset, dtype=float)
        for name, value in (mass or {}).items():
            box = self.boxes[name]
            self.model.body_mass[box["body"]] = float(value)
            self.model.body_inertia[box["body"]] = t8.inertia("box", box["dims"], float(value))
        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_setConst(self.model, self.data)
        mujoco.mj_resetData(self.model, self.data)

    def com_offset(self, name: str) -> np.ndarray:
        body = self.boxes[name]["body"]
        return np.array(self.data.xipos[body]) - np.array(self.data.xpos[body])

    def velocity(self, name: str) -> np.ndarray:
        box = self.boxes[name]
        return np.asarray(self.data.qvel[box["qvel"] : box["qvel"] + 6], dtype=float)


# --------------------------------------------------------------------------
# tilt — a box on a pitching deck
# --------------------------------------------------------------------------


def tilt_xml(dims: Sequence[float], mass: float, orientation: str = "upright") -> str:
    """MJCF of the tilt scene: the deck (hinge about y through the point under the box, top at the pivot) and the box resting at
    its centre in the given orientation."""
    frame = orientation_frame(orientation)
    half_up = float(np.abs(frame[2]) @ (np.asarray(dims, dtype=float) / 2))
    return (
        t8._header("v03-tilt")
        + "  <worldbody>\n"
        f'    <body name="deck" pos="0 0 {DECK_PIVOT_Z}">\n'
        '      <joint name="pitch" type="hinge" axis="0 1 0"/>\n'
        f'      <inertial pos="0 0 0" mass="{DECK_MASS}" diaginertia="{DECK_INERTIA} {DECK_INERTIA} {DECK_INERTIA}"/>\n'
        f'      <geom name="deck_geom" type="box" pos="0 0 {-DECK_HALF[2]}" size="{DECK_HALF[0]} {DECK_HALF[1]} {DECK_HALF[2]}" priority="1"/>\n'
        "    </body>\n"
        + _box_body("load", dims, mass, (0.0, 0.0, DECK_PIVOT_Z + half_up + 0.0005), ORIENTATIONS[orientation])
        + "  </worldbody>\n</mujoco>\n"
    )


def tilt_rig(dims: Sequence[float], mass: float, orientation: str = "upright") -> Rig:
    return Rig(tilt_xml(dims, mass, orientation), boxes=("load",), dims={"load": dims})


@dataclass
class Path1D:
    """Planned 1-D motion per step: position, velocity, acceleration; the step where the manoeuvre ends."""

    q: np.ndarray
    v: np.ndarray
    a: np.ndarray
    end: int


def _from_velocity(vel: np.ndarray, dt: float = TIMESTEP) -> Path1D:
    acc = np.diff(np.concatenate([[0.0], vel])) / dt
    q = np.cumsum(vel) * dt
    return Path1D(q=q, v=vel, a=acc, end=int(vel.size))


def leg_reaches_peak(peak: float, accel: float, distance: float, *, jerk_time: float, dt: float = TIMESTEP) -> bool:
    """Whether a move can reach `peak` at `accel` (with its ramps) within `distance`; if not, :func:`_smooth_leg` lowers the top speed."""
    up = _speed_up(float(peak), float(accel), jerk_time, dt)
    return bool(2 * up.sum() * dt <= float(distance) + 1e-12)


def _speed_up(peak: float, accel: float, jerk_time: float, dt: float) -> np.ndarray:
    tj = min(jerk_time, peak / accel)
    t_hold = max(0.0, peak / accel - tj)
    n_j, n_h = int(round(tj / dt)), int(round(t_hold / dt))
    ramp = (np.arange(n_j) + 0.5) / max(n_j, 1)
    acc = np.concatenate([accel * ramp, accel * np.ones(n_h), accel * ramp[::-1]])
    up = np.cumsum(acc) * dt
    return up * (peak / up[-1])


def _smooth_leg(peak: float, accel: float, distance: float, *, jerk_time: float, dt: float = TIMESTEP) -> np.ndarray:
    """Velocity samples of one move: acceleration ramps (linearly, over `jerk_time`) to `accel`, holds until `peak` speed, cruises,
    and comes back down the same way, covering `distance`. If the move is too short to reach `peak`, the **top speed** is lowered
    (bisection) so the stated acceleration and ramp time are kept — an earlier version scaled the whole profile instead, which lowered
    the acceleration below the stated one (≈ 15 % of the first grip variants; repaired, see the v0.3b report)."""
    peak, accel, distance = float(peak), float(accel), float(distance)
    if peak <= 0 or accel <= 0 or distance <= 0:
        raise ValueError("a move needs a positive peak speed, acceleration and distance")
    if not leg_reaches_peak(peak, accel, distance, jerk_time=jerk_time, dt=dt):
        low, high = 1e-4, peak
        for _ in range(40):
            mid = (low + high) / 2
            low, high = (mid, high) if leg_reaches_peak(mid, accel, distance, jerk_time=jerk_time, dt=dt) else (low, mid)
        peak = low
    tj = min(jerk_time, peak / accel)
    # speed gained over the ramp-up + hold + ramp-down of the acceleration = accel · (t_hold + tj) = peak
    t_hold = max(0.0, peak / accel - tj)
    n_j, n_h = int(round(tj / dt)), int(round(t_hold / dt))
    ramp = (np.arange(n_j) + 0.5) / max(n_j, 1)
    acc = np.concatenate([accel * ramp, accel * np.ones(n_h), accel * ramp[::-1]])
    up = np.cumsum(acc) * dt
    up *= peak / up[-1]
    down = up[::-1]
    covered = (up.sum() + down.sum()) * dt
    cruise = max(0, int(round((distance - covered) / (peak * dt))))
    vel = np.concatenate([up, peak * np.ones(cruise), down[1:], [0.0]])
    return vel * (distance / (vel.sum() * dt))


def pitch_profile(angle_deg: float, duration: float, *, pre: float = 0.2, hold: float = TILT_HOLD, dt: float = TIMESTEP) -> Path1D:
    """Deck pitch (rad): rest for `pre` s, a raised-cosine pitch rate that reaches `angle_deg` after `duration` s, then hold."""
    angle = math.radians(float(angle_deg))
    if duration <= 0:
        raise ValueError("pitch duration must be positive")
    n_pre, n_move, n_hold = int(round(pre / dt)), int(round(duration / dt)), int(round(hold / dt))
    s = (np.arange(n_move) + 0.5) / n_move
    rate = (1 - np.cos(2 * np.pi * s)) / n_move / dt  # integrates to 1 over the move
    vel = np.concatenate([np.zeros(n_pre), angle * rate, np.zeros(n_hold)])
    path = _from_velocity(vel, dt)
    path.end = n_pre + n_move
    return path


def _deck_state(rig: Rig, deck: int, up_body: np.ndarray, start: np.ndarray) -> tuple[float, float]:
    """(tilt of the box's original up axis from the deck normal in degrees, slide along the deck in metres)."""
    box = rig.boxes["load"]
    deck_rot = np.asarray(rig.data.xmat[deck]).reshape(3, 3)
    box_rot = np.asarray(rig.data.xmat[box["body"]]).reshape(3, 3)
    up_world = box_rot @ up_body
    cosine = float(np.clip(up_world @ deck_rot[:, 2], -1.0, 1.0))
    rel = deck_rot.T @ (np.asarray(rig.data.xpos[box["body"]]) - np.asarray(rig.data.xpos[deck]))
    return math.degrees(math.acos(cosine)), float(np.hypot(*(rel[:2] - start[:2])))


def run_tilt(rig: Rig, path: Path1D, *, mu: float, com: Sequence[float], orientation: str = "upright") -> dict[str, Any]:
    """One tilt rollout → ``{"outcome": stay|slide|tip, "slide_m", "tilt_deg", "max_tilt_deg", "unstable", "steps"}``."""
    rig.set_hidden(friction={"deck_geom": mu}, com={"load": com})
    model, data = rig.model, rig.data
    deck = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "deck")
    up_body = orientation_frame(orientation).T @ np.array([0.0, 0.0, 1.0])  # the box axis that points up at the start
    mujoco.mj_forward(model, data)
    start = np.asarray(data.xmat[deck]).reshape(3, 3).T @ (np.asarray(data.xpos[rig.boxes["load"]["body"]]) - np.asarray(data.xpos[deck]))
    z0 = top = float(start[2])
    # largest possible rise of the centre when the box rotates onto an edge, in its actual orientation (+ 5 cm)
    half = np.abs(orientation_frame(orientation)) @ (np.asarray(rig.boxes["load"]["dims"], dtype=float) / 2)
    rise_limit = math.hypot(max(half[0], half[1]), half[2]) - half[2] + 0.05
    max_tilt, steps, outcome = 0.0, 0, None
    total = int(path.q.size)
    settle_cap = total + int(round(SETTLE_MAX / TIMESTEP))
    i = 0
    while i < settle_cap:
        k = min(i, total - 1)
        target_q, target_v, target_a = (path.q[k], path.v[k], path.a[k]) if i < total else (path.q[-1], 0.0, 0.0)
        data.qfrc_applied[0] = DECK_INERTIA * (target_a + KP * (target_q - data.qpos[0]) + KD * (target_v - data.qvel[0]))
        mujoco.mj_step(model, data)
        i += 1
        if i % 10:
            continue
        tilt, slide = _deck_state(rig, deck, up_body, start)
        rel_z = float((np.asarray(data.xmat[deck]).reshape(3, 3).T @ (np.asarray(data.xpos[rig.boxes["load"]["body"]]) - np.asarray(data.xpos[deck])))[2])
        top = max(top, rel_z)
        max_tilt = max(max_tilt, tilt)
        if tilt > TILT_TIP_DEG:
            outcome = "tip"
            break
        if slide > TILT_SLIDE_STOP and tilt < TILT_SLIDE_UPRIGHT:
            outcome = "slide"
            break
        if i >= total and i % 100 == 0:
            v = rig.velocity("load")
            if np.linalg.norm(v[:3]) < SETTLE_LIN and np.linalg.norm(v[3:]) < SETTLE_ANG:
                break
    steps = i
    tilt, slide = _deck_state(rig, deck, up_body, start)
    if outcome is None:
        outcome = "tip" if tilt > TILT_TIP_DEG else ("slide" if slide > TILT_SLIDE_M else "stay")
    return {"outcome": outcome, "slide_m": slide, "tilt_deg": tilt, "max_tilt_deg": max(max_tilt, tilt), "unstable": bool(top - z0 > rise_limit),
            "steps": steps}


# --------------------------------------------------------------------------
# grip — a box squeezed between two pads while the arm moves
# --------------------------------------------------------------------------


def grip_xml(dims: Sequence[float], mass: float) -> str:
    """MJCF of the grip scene. ``dims`` = (L along the pads, W between the pads, H up). The pads touch the box's side faces at its
    mid-height; each finger is a slide joint (limited so the pads never meet) pushed inward by the squeeze force."""
    length, width, height = (float(v) for v in dims)
    y_pad = width / 2 + PAD_HALF[1]
    close = y_pad - PAD_HALF[1] - 0.004  # how far a finger can travel inward before its stop (pads stay 8 mm apart)
    finger = (
        '    <body name="finger_{side}" pos="0 {y:.6f} 0">\n'
        '      <joint name="finger_{side}" type="slide" axis="0 1 0" limited="true" range="{lo:.6f} {hi:.6f}"/>\n'
        f'      <inertial pos="0 0 0" mass="{FINGER_MASS}" diaginertia="1e-4 1e-4 1e-4"/>\n'
        f'      <geom name="pad_{{side}}" type="box" size="{PAD_HALF[0]} {PAD_HALF[1]} {PAD_HALF[2]}" priority="1"/>\n'
        "    </body>\n"
    )
    return (
        t8._header("v03-grip")
        + "  <worldbody>\n"
        f'    <body name="hand" pos="0 0 {HAND_Z}">\n'
        '      <joint name="hand_x" type="slide" axis="1 0 0"/>\n'
        '      <joint name="hand_z" type="slide" axis="0 0 1"/>\n'
        f'      <inertial pos="0 0 0" mass="{HAND_MASS}" diaginertia="10 10 10"/>\n'
        + "".join("  " + line + "\n" for line in finger.format(side="l", y=y_pad, lo=-close, hi=0.02).splitlines())
        + "".join("  " + line + "\n" for line in finger.format(side="r", y=-y_pad, lo=-0.02, hi=close).splitlines())
        + "    </body>\n"
        + _box_body("load", (length, width, height), mass, (0.0, 0.0, HAND_Z))
        + "  </worldbody>\n</mujoco>\n"
    )


def grip_rig(dims: Sequence[float], mass: float) -> Rig:
    return Rig(grip_xml(dims, mass), boxes=("load",), dims={"load": dims})


def move_profile(kind: str, distance: float, peak: float, accel: float, *, jerk_time: float = 0.04, pre: float = GRIP_PRE,
                 post: float = GRIP_POST, dt: float = TIMESTEP) -> tuple[Path1D, int]:
    """Hand motion: ``lift`` (up along z), ``lower`` (down along z) or ``swing`` (sideways along x) by `distance` m, speeding up at
    `accel` m/s² to `peak` m/s and braking at the same rate. Returns the path and the axis (0 = x, 1 = z)."""
    if kind not in ("lift", "lower", "swing"):
        raise ValueError(f"move kind: lift | lower | swing (got {kind!r})")
    leg = _smooth_leg(peak, accel, distance, jerk_time=jerk_time, dt=dt)
    sign = -1.0 if kind == "lower" else 1.0
    vel = np.concatenate([np.zeros(int(round(pre / dt))), sign * leg, np.zeros(int(round(post / dt)))])
    path = _from_velocity(vel, dt)
    path.end = int(round(pre / dt)) + leg.size
    return path, (0 if kind == "swing" else 1)


def run_grip(rig: Rig, path: Path1D, axis: int, *, mu: float, mass: float, squeeze: float, com: Sequence[float] = (0.0, 0.0, 0.0)) -> dict[str, Any]:
    """One grip rollout → ``{"outcome": hold|slip|drop, "slip_m", "unstable", "steps"}``. The box starts held at rest; the slip is
    measured against its position in the hand at the end of the still phase (``GRIP_PRE`` s, while the squeeze settles)."""
    rig.set_hidden(friction={"pad_l": mu, "pad_r": mu}, com={"load": com}, mass={"load": mass})
    model, data = rig.model, rig.data
    hand = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "hand")
    box = rig.boxes["load"]
    dof_x, dof_z = 0, 1
    fl = int(model.jnt_dofadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "finger_l")])
    fr = int(model.jnt_dofadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "finger_r")])
    carried = HAND_MASS + 2 * FINGER_MASS
    mujoco.mj_forward(model, data)
    height = box["dims"][2]
    drop_at = PAD_HALF[2] + height / 2 - 0.002  # the box's top edge has passed the pads' lower edge
    reference = None
    slip, steps, outcome = 0.0, 0, None
    n = int(path.q.size)
    for i in range(n):
        target = [0.0, 0.0]
        target_v = [0.0, 0.0]
        target_a = [0.0, 0.0]
        target[axis], target_v[axis], target_a[axis] = path.q[i], path.v[i], path.a[i]
        for dof, column in ((dof_x, 0), (dof_z, 1)):
            data.qfrc_applied[dof] = carried * (target_a[column] + KP * (target[column] - data.qpos[dof]) + KD * (target_v[column] - data.qvel[dof]))
        data.qfrc_applied[dof_z] += carried * GRAVITY
        data.qfrc_applied[fl] = -float(squeeze)
        data.qfrc_applied[fr] = float(squeeze)
        mujoco.mj_step(model, data)
        steps = i + 1
        if steps == int(round(GRIP_PRE / TIMESTEP)):
            reference = np.asarray(data.xpos[box["body"]]) - np.asarray(data.xpos[hand])
        if steps % 10:
            continue
        rel = np.asarray(data.xpos[box["body"]]) - np.asarray(data.xpos[hand])
        if reference is not None:
            slip = float(np.linalg.norm(rel - reference))
        if rel[2] < -drop_at or abs(rel[0]) > PAD_HALF[0] + box["dims"][0] / 2:
            outcome = "drop"
            break
    if outcome is None:
        outcome = "slip" if slip >= GRIP_SLIP_M else "hold"
    return {"outcome": outcome, "slip_m": slip, "unstable": False, "steps": steps}


# --------------------------------------------------------------------------
# tower — two or three boxes stacked on the floor
# --------------------------------------------------------------------------


#: tower: full ``implicit`` integration, a 0.5 ms step and softer contacts (time constant 8 ms). With T8's settings (``implicitfast``,
#: 1 ms, 4 ms) a heavy box dropping onto the edge of the box below, or tumbling off it, sometimes gained energy and launched itself
#: and the box below metres away: 10.7 % of the first 449 tower groups had a rollout flagged by the rise guard, and the same
#: rollouts turned "top" into "more". ``implicit`` alone (keeps the gyroscopic terms), a 0.5 ms or 0.25 ms step alone, or 8 ms
#: contacts alone each left 10–20 % of the flagged rollouts unstable; the three together left none (0 of 192 on the worst groups),
#: at about the same cost per rollout. The other families have no impacts or tumbling and keep T8's settings.
TOWER_INTEGRATOR = "implicit"
TOWER_TIMESTEP = 0.0005
TOWER_SOLREF = "0.008 1"


def tower_xml(boxes: Sequence[tuple[Sequence[float], float]]) -> str:
    """MJCF of the tower scene: a floor and the boxes (``(dims, mass)`` from the bottom up, dims = L fore-aft × W × H); the
    positions are written per rollout."""
    header = (t8._header("v03-tower").replace('integrator="implicitfast"', f'integrator="{TOWER_INTEGRATOR}"')
              .replace(f'timestep="{TIMESTEP}"', f'timestep="{TOWER_TIMESTEP}"').replace(f'solref="{t8.SOLREF}"', f'solref="{TOWER_SOLREF}"'))
    if f'timestep="{TOWER_TIMESTEP}"' not in header or f'solref="{TOWER_SOLREF}"' not in header:
        raise AssertionError("tower header: T8's option line changed")
    out = header + "  <worldbody>\n" + '    <geom name="floor" type="plane" size="3 3 0.1" priority="1"/>\n'
    z = 0.0
    for i, (dims, mass) in enumerate(boxes):
        out += _box_body(f"box{i}", dims, mass, (0.0, 0.0, z + dims[2] / 2 + 0.0005))
        z += dims[2]
    return out + "  </worldbody>\n</mujoco>\n"


def tower_rig(boxes: Sequence[tuple[Sequence[float], float]]) -> Rig:
    return Rig(tower_xml(boxes), boxes=tuple(f"box{i}" for i in range(len(boxes))), dims={f"box{i}": dims for i, (dims, _) in enumerate(boxes)})


def run_tower(rig: Rig, *, centres: Sequence[float], drop: float, mu: float, coms: Sequence[Sequence[float]]) -> dict[str, Any]:
    """One release → ``{"outcome": stands|top|more, "fallen": [bool per box], "tilts", "unstable", "steps"}``. ``centres`` = the
    boxes' fore-aft centre positions (bottom up); the lower boxes start resting on each other, the top box `drop` m above."""
    names = sorted(rig.boxes, key=lambda b: int(b[3:]))
    friction = {f"{name}_geom": mu for name in names}
    rig.set_hidden(friction=friction, com={name: com for name, com in zip(names, coms)})
    model, data = rig.model, rig.data
    z = 0.0
    start_z = []
    for i, name in enumerate(names):
        box = rig.boxes[name]
        height = box["dims"][2]
        lift = float(drop) if i == len(names) - 1 and i > 0 else 0.0
        centre_z = z + height / 2 + 0.0005 * (i + 1) + lift
        data.qpos[box["qpos"] : box["qpos"] + 7] = (float(centres[i]), 0.0, centre_z, 1.0, 0.0, 0.0, 0.0)
        start_z.append(centre_z)
        z += height + 0.0005
    mujoco.mj_forward(model, data)
    tops = list(start_z)
    steps = 0
    dt = float(model.opt.timestep)
    chunk = int(round(0.05 / dt))  # check every 50 ms
    limit = int(round((TOWER_SECONDS + SETTLE_MAX) / dt))
    while steps < limit:
        mujoco.mj_step(model, data, nstep=chunk)
        steps += chunk
        for i, name in enumerate(names):
            tops[i] = max(tops[i], float(data.xpos[rig.boxes[name]["body"]][2]))
        if steps >= int(round(TOWER_SECONDS / dt)) and steps % (2 * chunk) == 0:
            if all(np.linalg.norm(rig.velocity(n)[:3]) < SETTLE_LIN and np.linalg.norm(rig.velocity(n)[3:]) < SETTLE_ANG for n in names):
                break
    tilts, fallen = [], []
    for i, name in enumerate(names):
        body = rig.boxes[name]["body"]
        tilt = math.degrees(math.acos(max(-1.0, min(1.0, float(data.xmat[body][8])))))
        tilts.append(tilt)
        fallen.append(bool(tilt > TOWER_FALL_DEG or float(data.xpos[body][2]) < start_z[i] - rig.boxes[name]["dims"][2] / 2))
    if not any(fallen):
        outcome = "stands"
    elif fallen[-1] and not any(fallen[:-1]):
        outcome = "top"
    else:
        outcome = "more"
    unstable = any(tops[i] - start_z[i] > t8.max_rise(rig.boxes[name]["dims"]) for i, name in enumerate(names))
    return {"outcome": outcome, "fallen": fallen, "tilts": tilts, "unstable": bool(unstable), "steps": steps}
