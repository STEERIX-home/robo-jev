"""v0.4a scene physics — new MuJoCo geometry for the nine new scenes of three laws (Task v0.4a §1.1).

Built on v0.3b's rig (:class:`robo_jev.mjgen.physics.Rig`: the hidden friction, centre of mass and mass are written before every
rollout; loads are boxes only) and on T8's contact settings and tracking gains (:mod:`robo_jev.outside.physics`: 1 ms step,
``implicitfast``, elliptic cones, ``impratio`` 10, 4 ms contacts). Four engines:

* **platform** (law A — tip or slide under acceleration or tilt; scenes ``forks``, ``bin``, ``tray``): a heavy carriage on two slide
  joints carries a support (two fork tines; a cargo-bin floor with four walls; a hand-held tray), fixed at a tilt about the lateral
  axis (the forks' mast tilt); a free box load rests on it. The carriage follows a planned acceleration profile along x (braking or
  starting off) or y (a turn, as its lateral acceleration — a pure translation, so the load feels the same inertial push v²/R as in
  the turn). Only relative motion matters, so a stop from speed v is rolled as the carriage speeding up to v the other way from rest.
  Read in the support frame: the load's slide along the support plane and the tilt of its up axis from the support normal.
* **squeeze** (law C — friction-limited holding; ``plate``, ``clamp``, ``palms``): v0.3b's two-pad grip with pad size and hand mass
  as parameters, and two new motions: a braking carry (horizontal, one-sided) and a walking bob (vertical, periodic).
* **ledge** (law B — centre of mass past an edge; ``table``): a fixed table top with a straight edge at x = 0 and a free box released
  partly over it (the tower's integrator settings: ``implicit``, 0.5 ms, 8 ms contacts — a box landing on an edge).
* **shift** (law B; ``reach``, ``cabinet``): a free base on four feet (a wheeled robot base; a cabinet) and a mover on a slide joint
  inside it (an arm carrying a payload; a drawer and its contents) that runs out along +x under a planned profile. The drive force acts
  between base and mover, so the base takes the reaction (the stop at the end of the run is a forward jolt). Read: the base's tilt.

Everything is deterministic: the same model, hidden values and profile give bit-identical rollouts (tested).
"""

from __future__ import annotations

import math
from typing import Any, Sequence

import mujoco
import numpy as np

from robo_jev.mjgen import physics as ph
from robo_jev.outside import physics as t8

__all__ = [
    "CART_MASS",
    "LEDGE_TOP",
    "PLATFORM_Z",
    "brake_carry_profile",
    "ledge_rig",
    "ledge_xml",
    "linear_profile",
    "platform_rig",
    "platform_xml",
    "run_ledge",
    "run_platform",
    "run_shift",
    "run_squeeze",
    "shift_profile",
    "shift_rig",
    "shift_xml",
    "squeeze_rig",
    "squeeze_xml",
    "walk_profile",
]

TIMESTEP = t8.TIMESTEP
GRAVITY = t8.GRAVITY
KP, KD = t8.KP, t8.KD
SETTLE_LIN, SETTLE_ANG = t8.SETTLE_LIN, t8.SETTLE_ANG


def _quat_y(angle_rad: float) -> tuple[float, float, float, float]:
    """Unit quaternion of a rotation by `angle_rad` about +y."""
    return (math.cos(angle_rad / 2), 0.0, math.sin(angle_rad / 2), 0.0)


def _mat(quat: Sequence[float]) -> np.ndarray:
    out = np.zeros(9)
    mujoco.mju_quat2Mat(out, np.asarray(quat, dtype=float))
    return out.reshape(3, 3)


def _quat_mul(a: Sequence[float], b: Sequence[float]) -> np.ndarray:
    out = np.zeros(4)
    mujoco.mju_mulQuat(out, np.asarray(a, dtype=float), np.asarray(b, dtype=float))
    return out


# --------------------------------------------------------------------------
# Profiles
# --------------------------------------------------------------------------


def linear_profile(speed: float, accel: float, *, jerk_time: float, pre: float = 0.3, post: float = 1.2, sign: float = 1.0,
                   dt: float = TIMESTEP) -> ph.Path1D:
    """One-sided speed change along an axis: rest for `pre` s, the acceleration ramps (over `jerk_time`) to `accel`, holds until the
    speed reaches `speed`, ramps back to zero, then cruises at that speed for `post` s. A stop from `speed` at deceleration `accel` is
    the same motion seen from the vehicle (``sign`` = the direction the vehicle speeds up in this frame)."""
    if speed <= 0 or accel <= 0:
        raise ValueError("a speed change needs a positive speed and acceleration")
    up = ph._speed_up(float(speed), float(accel), float(jerk_time), dt)
    vel = sign * np.concatenate([np.zeros(int(round(pre / dt))), up, float(speed) * np.ones(int(round(post / dt)))])
    path = ph._from_velocity(vel, dt)
    path.end = int(round(pre / dt)) + up.size
    return path


def brake_carry_profile(speed: float, decel: float, *, jerk_time: float, pre: float = 0.4, post: float = 0.8, dt: float = TIMESTEP) -> ph.Path1D:
    """A carried item, the carrier stopping from `speed` at `decel` (the item is pushed toward +x): :func:`linear_profile` toward −x."""
    return linear_profile(speed, decel, jerk_time=jerk_time, pre=pre, post=post, sign=-1.0, dt=dt)


def walk_profile(rise: float, step_hz: float, steps: int, *, pre: float = 0.4, post: float = 0.6, dt: float = TIMESTEP) -> ph.Path1D:
    """Vertical bob of the hands while walking: z = (rise/2)·(1 − cos 2πft) for `steps` steps (one bob per step), at rest before
    and after. Peak upward acceleration (rise/2)·(2πf)², at the bottom of each bob."""
    omega = 2 * math.pi * float(step_hz)
    n_pre, n_walk, n_post = int(round(pre / dt)), int(round(steps / step_hz / dt)), int(round(post / dt))
    t = np.arange(n_walk) * dt
    q = np.concatenate([np.zeros(n_pre), rise / 2 * (1 - np.cos(omega * t)), np.zeros(n_post)])
    v = np.concatenate([np.zeros(n_pre), rise / 2 * omega * np.sin(omega * t), np.zeros(n_post)])
    a = np.concatenate([np.zeros(n_pre), rise / 2 * omega**2 * np.cos(omega * t), np.zeros(n_post)])
    return ph.Path1D(q=q, v=v, a=a, end=n_pre + n_walk)


def shift_profile(travel: float, duration: float, *, pre: float = 0.3, post: float = 0.2, dt: float = TIMESTEP) -> ph.Path1D:
    """A mover running out by `travel` m in `duration` s with a sine-squared speed: it starts and stops smoothly, and its peak
    deceleration near the end of the run, 2π·travel/duration², is the forward push the base feels as the run ends."""
    n_pre, n_move, n_post = int(round(pre / dt)), int(round(duration / dt)), int(round(post / dt))
    s = (np.arange(n_move) + 0.5) / n_move
    rate = np.sin(np.pi * s) ** 2
    vel = travel * rate / (rate.sum() * dt)
    vel = np.concatenate([np.zeros(n_pre), vel, np.zeros(n_post)])
    path = ph._from_velocity(vel, dt)
    path.end = n_pre + n_move
    return path


# --------------------------------------------------------------------------
# platform — a load on an accelerating (and possibly tilted) support
# --------------------------------------------------------------------------

#: height of the support's top plane (at its pivot) above the floor; the carriage's mass (it follows its trajectory regardless of the load)
PLATFORM_Z = 0.80
CART_MASS = 5000.0


def platform_xml(name: str, support: Sequence[tuple[str, Sequence[float], Sequence[float], bool]], *, tilt_deg: float, load_dims: Sequence[float],
                 load_mass: float, load_xy: Sequence[float] = (0.0, 0.0)) -> str:
    """MJCF of a platform scene. ``support`` = [(geom name, centre in the support frame, half sizes, carries the hidden friction)]; the
    support frame has its top plane at z = 0 and is tilted by `tilt_deg` about y so that its +x end rises (fork tips up). The load
    stands upright on the support plane at `load_xy` (support frame)."""
    tilt = -math.radians(float(tilt_deg))  # a rotation about +y by a negative angle raises the +x end
    q_support = _quat_y(tilt)
    rot = _mat(q_support)
    pos_support = np.array([float(load_xy[0]), float(load_xy[1]), float(load_dims[2]) / 2 + 0.0005])
    load_world = np.array([0.0, 0.0, PLATFORM_Z]) + rot @ pos_support
    # one body per support geom: MuJoCo's midphase places a body's bounding box in its compile-time inertial frame, so a load whose
    # hidden centre of mass is moved at runtime can miss (or half-detect) contacts with a support body holding several geoms (the load
    # fell through two fork tines or was launched off them); bodies of one geom each are collided without that bounding box (tested)
    geoms = ""
    for geom, centre, half, _ in support:
        geoms += (f'        <body name="{geom}_body"><geom name="{geom}" type="box" pos="{centre[0]:.6f} {centre[1]:.6f} {centre[2]:.6f}" '
                  f'size="{half[0]:.6f} {half[1]:.6f} {half[2]:.6f}" priority="1"/></body>\n')
    return (
        t8._header(f"v04-{name}")
        + "  <worldbody>\n"
        '    <geom name="floor" type="plane" size="20 20 0.1"/>\n'
        f'    <body name="cart" pos="0 0 {PLATFORM_Z}">\n'
        '      <joint name="cart_x" type="slide" axis="1 0 0"/>\n'
        '      <joint name="cart_y" type="slide" axis="0 1 0"/>\n'
        f'      <inertial pos="0 0 0" mass="{CART_MASS}" diaginertia="1e4 1e4 1e4"/>\n'
        f'      <body name="support" pos="0 0 0" quat="{q_support[0]:.9f} {q_support[1]:.9f} {q_support[2]:.9f} {q_support[3]:.9f}">\n'
        + geoms
        + "      </body>\n"
        "    </body>\n"
        + ph._box_body("load", load_dims, load_mass, tuple(load_world), tuple(q_support))
        + "  </worldbody>\n</mujoco>\n"
    )


def platform_rig(name: str, support: Sequence[tuple[str, Sequence[float], Sequence[float], bool]], *, tilt_deg: float, load_dims: Sequence[float],
                 load_mass: float, load_xy: Sequence[float] = (0.0, 0.0)) -> ph.Rig:
    rig = ph.Rig(platform_xml(name, support, tilt_deg=tilt_deg, load_dims=load_dims, load_mass=load_mass, load_xy=load_xy), boxes=("load",),
                 dims={"load": load_dims})
    rig.friction_geoms = tuple(g for g, _, _, hidden in support if hidden)  # type: ignore[attr-defined]
    return rig


def run_platform(rig: ph.Rig, path: ph.Path1D, axis: int, *, mu: float, com: Sequence[float], stop_slide: float = 0.10,
                 settle: float = 2.0) -> dict[str, Any]:
    """One platform rollout → ``{"slide_m", "tilt_deg", "max_tilt_deg", "fell", "early", "unstable", "steps"}`` (the scene classifies).

    The carriage tracks `path` along `axis` (0 = x, 1 = y) and holds the other axis; after the path it keeps its final speed (a
    vehicle cruising on) until the load is at rest relative to it (checked every 0.1 s, at most `settle` s). Early stops: the load tilts
    past 45° (``early = "tip"``), or it has slid `stop_slide` m while within 5° of upright (``early = "slide"``: a held push keeps it
    going). ``fell`` = the load dropped below the support plane (it went off an edge)."""
    rig.set_hidden(friction={g: mu for g in rig.friction_geoms}, com={"load": com})  # type: ignore[attr-defined]
    model, data = rig.model, rig.data
    support = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "support")
    load = rig.boxes["load"]
    dofs = (int(model.jnt_dofadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "cart_x")]),
            int(model.jnt_dofadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "cart_y")]))
    mujoco.mj_forward(model, data)
    rot_s = np.asarray(data.xmat[support]).reshape(3, 3)
    rot_l = np.asarray(data.xmat[load["body"]]).reshape(3, 3)
    up_body = rot_l.T @ rot_s[:, 2]
    start = rot_s.T @ (np.asarray(data.xpos[load["body"]]) - np.asarray(data.xpos[support]))
    half_up = float(start[2])
    rise_limit = t8.max_rise(load["dims"])
    n = int(path.q.size)
    total = n + int(round(settle / TIMESTEP))
    slide = tilt = max_tilt = 0.0
    top = half_up
    fell, early, i = False, None, 0
    while i < total:
        k = min(i, n - 1)
        if i < n:
            q_t, v_t, a_t = path.q[k], path.v[k], path.a[k]
        else:
            q_t, v_t, a_t = path.q[-1] + path.v[-1] * (i - n + 1) * TIMESTEP, path.v[-1], 0.0
        for column, dof in enumerate(dofs):
            target = (q_t, v_t, a_t) if column == axis else (0.0, 0.0, 0.0)
            data.qfrc_applied[dof] = CART_MASS * (target[2] + KP * (target[0] - data.qpos[dof]) + KD * (target[1] - data.qvel[dof]))
        mujoco.mj_step(model, data)
        i += 1
        if i % 10:
            continue
        rot_s = np.asarray(data.xmat[support]).reshape(3, 3)
        rel = rot_s.T @ (np.asarray(data.xpos[load["body"]]) - np.asarray(data.xpos[support]))
        up_world = np.asarray(data.xmat[load["body"]]).reshape(3, 3) @ up_body
        tilt = math.degrees(math.acos(float(np.clip(up_world @ rot_s[:, 2], -1.0, 1.0))))
        slide = float(np.hypot(*(rel[:2] - start[:2])))
        max_tilt = max(max_tilt, tilt)
        top = max(top, float(rel[2]))
        if rel[2] < -0.02:
            fell, early = True, "fell"
            break
        if tilt > 45.0:
            early = "tip"
            break
        if slide > stop_slide and tilt < 5.0:
            early = "slide"
            break
        if i >= n and i % 100 == 0:
            v = rig.velocity("load")
            frame = np.zeros(3)
            frame[axis] = path.v[-1]
            if np.linalg.norm(v[:3] - frame) < SETTLE_LIN and np.linalg.norm(v[3:]) < SETTLE_ANG:
                break
    return {"slide_m": slide, "tilt_deg": tilt, "max_tilt_deg": max_tilt, "fell": fell, "early": early,
            "unstable": bool(top - half_up > rise_limit), "steps": i}


# --------------------------------------------------------------------------
# squeeze — an item held between two pads while the holder moves
# --------------------------------------------------------------------------


def squeeze_xml(name: str, dims: Sequence[float], *, pad_half: Sequence[float], hand_mass: float, finger_mass: float = 0.25) -> str:
    """MJCF of a squeeze scene (v0.3b's grip with the pad size and hand mass as parameters). ``dims`` = (L along the pads, W between
    the pads, H up); each pad touches a side face at the item's mid-height; each finger is a slide joint (limited so the pads never
    meet) pushed inward by the squeeze force."""
    length, width, height = (float(v) for v in dims)
    y_pad = width / 2 + pad_half[1]
    close = y_pad - pad_half[1] - 0.004
    finger = (
        '    <body name="finger_{side}" pos="0 {y:.6f} 0">\n'
        '      <joint name="finger_{side}" type="slide" axis="0 1 0" limited="true" range="{lo:.6f} {hi:.6f}"/>\n'
        f'      <inertial pos="0 0 0" mass="{finger_mass}" diaginertia="1e-3 1e-3 1e-3"/>\n'
        f'      <geom name="pad_{{side}}" type="box" size="{pad_half[0]:.6f} {pad_half[1]:.6f} {pad_half[2]:.6f}" priority="1"/>\n'
        "    </body>\n"
    )
    return (
        t8._header(f"v04-{name}")
        + "  <worldbody>\n"
        f'    <body name="hand" pos="0 0 {ph.HAND_Z}">\n'
        '      <joint name="hand_x" type="slide" axis="1 0 0"/>\n'
        '      <joint name="hand_z" type="slide" axis="0 0 1"/>\n'
        f'      <inertial pos="0 0 0" mass="{hand_mass}" diaginertia="10 10 10"/>\n'
        + "".join("  " + line + "\n" for line in finger.format(side="l", y=y_pad, lo=-close, hi=0.02).splitlines())
        + "".join("  " + line + "\n" for line in finger.format(side="r", y=-y_pad, lo=-0.02, hi=close).splitlines())
        + "    </body>\n"
        + ph._box_body("load", (length, width, height), 1.0, (0.0, 0.0, ph.HAND_Z))
        + "  </worldbody>\n</mujoco>\n"
    )


def squeeze_rig(name: str, dims: Sequence[float], *, pad_half: Sequence[float], hand_mass: float, finger_mass: float = 0.25) -> ph.Rig:
    rig = ph.Rig(squeeze_xml(name, dims, pad_half=pad_half, hand_mass=hand_mass, finger_mass=finger_mass), boxes=("load",), dims={"load": dims})
    rig.pad_half = tuple(float(v) for v in pad_half)  # type: ignore[attr-defined]
    rig.carried = float(hand_mass) + 2 * float(finger_mass)  # type: ignore[attr-defined]
    return rig


def run_squeeze(rig: ph.Rig, path: ph.Path1D, axis: int, *, mu: float, mass: float, squeeze: float, slip_m: float, pre: float) -> dict[str, Any]:
    """One squeeze rollout → ``{"outcome": hold|slip|drop, "slip_m", "unstable", "steps"}`` (v0.3b's :func:`physics.run_grip` with the pad
    size as a parameter). The slip is measured against the item's place in the hand after `pre` s (while the squeeze settles); ``drop``
    = the item's top edge passed the pads' lower edge, or it left the pads sideways."""
    rig.set_hidden(friction={"pad_l": mu, "pad_r": mu}, com={"load": (0.0, 0.0, 0.0)}, mass={"load": mass})
    model, data = rig.model, rig.data
    hand = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "hand")
    box = rig.boxes["load"]
    pad_half = rig.pad_half  # type: ignore[attr-defined]
    carried = rig.carried  # type: ignore[attr-defined]
    fl = int(model.jnt_dofadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "finger_l")])
    fr = int(model.jnt_dofadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "finger_r")])
    mujoco.mj_forward(model, data)
    drop_at = pad_half[2] + box["dims"][2] / 2 - 0.002
    reference = None
    slip, steps, outcome = 0.0, 0, None
    n = int(path.q.size)
    ref_step = int(round(pre / TIMESTEP))
    for i in range(n):
        target = [0.0, 0.0]
        target_v = [0.0, 0.0]
        target_a = [0.0, 0.0]
        target[axis], target_v[axis], target_a[axis] = path.q[i], path.v[i], path.a[i]
        for dof in (0, 1):
            data.qfrc_applied[dof] = carried * (target_a[dof] + KP * (target[dof] - data.qpos[dof]) + KD * (target_v[dof] - data.qvel[dof]))
        data.qfrc_applied[1] += carried * GRAVITY
        data.qfrc_applied[fl] = -float(squeeze)
        data.qfrc_applied[fr] = float(squeeze)
        mujoco.mj_step(model, data)
        steps = i + 1
        if steps == ref_step:
            reference = np.asarray(data.xpos[box["body"]]) - np.asarray(data.xpos[hand])
        if steps % 10:
            continue
        rel = np.asarray(data.xpos[box["body"]]) - np.asarray(data.xpos[hand])
        if reference is not None:
            slip = float(np.linalg.norm(rel - reference))
        if rel[2] < -drop_at or abs(rel[0]) > pad_half[0] + box["dims"][0] / 2:
            outcome = "drop"
            break
    if outcome is None:
        outcome = "slip" if slip >= slip_m else "hold"
    return {"outcome": outcome, "slip_m": slip, "unstable": False, "steps": steps}


# --------------------------------------------------------------------------
# ledge — a box released partly over a table edge
# --------------------------------------------------------------------------

LEDGE_TOP = 0.75
LEDGE_HALF = (0.8, 0.6, 0.02)
LEDGE_SECONDS = 1.5
LEDGE_SETTLE = 2.0


def ledge_xml(dims: Sequence[float], mass: float) -> str:
    """MJCF of the ledge scene: a floor, a fixed table top whose edge runs along y at x = 0 (the table extends toward −x) and a free
    box (L along x, W, H). The tower's integrator, step and contact softness (a box landing on an edge)."""
    header = (t8._header("v04-ledge").replace('integrator="implicitfast"', f'integrator="{ph.TOWER_INTEGRATOR}"')
              .replace(f'timestep="{TIMESTEP}"', f'timestep="{ph.TOWER_TIMESTEP}"').replace(f'solref="{t8.SOLREF}"', f'solref="{ph.TOWER_SOLREF}"'))
    if f'timestep="{ph.TOWER_TIMESTEP}"' not in header or f'solref="{ph.TOWER_SOLREF}"' not in header:
        raise AssertionError("ledge header: T8's option line changed")
    return (
        header
        + "  <worldbody>\n"
        '    <geom name="floor" type="plane" size="5 5 0.1"/>\n'
        f'    <body name="table"><geom name="table_geom" type="box" pos="{-LEDGE_HALF[0]} 0 {LEDGE_TOP - LEDGE_HALF[2]}" '
        f'size="{LEDGE_HALF[0]} {LEDGE_HALF[1]} {LEDGE_HALF[2]}" priority="1"/></body>\n'
        + ph._box_body("load", dims, mass, (0.0, 0.0, LEDGE_TOP + float(dims[2]) / 2 + 0.05))
        + "  </worldbody>\n</mujoco>\n"
    )


def ledge_rig(dims: Sequence[float], mass: float) -> ph.Rig:
    return ph.Rig(ledge_xml(dims, mass), boxes=("load",), dims={"load": dims})


def run_ledge(rig: ph.Rig, *, x_centre: float, drop: float, mu: float, com: Sequence[float], rock_deg: float = 4.0) -> dict[str, Any]:
    """One release → ``{"outcome": stays|rocks|falls, "max_tilt_deg", "tilt_deg", "unstable", "steps"}``. The box starts level with its
    centre at `x_centre` (positive = beyond the edge), its bottom `drop` m above the top. ``falls`` = it went over the edge (its centre
    dropped below the top, or it tilted past 45°); ``rocks`` = it stayed on the table but tilted at least `rock_deg` on the way."""
    rig.set_hidden(friction={"table_geom": mu}, com={"load": com})
    model, data = rig.model, rig.data
    box = rig.boxes["load"]
    height = box["dims"][2]
    z0 = LEDGE_TOP + height / 2 + 0.0005 + float(drop)
    data.qpos[box["qpos"] : box["qpos"] + 7] = (float(x_centre), 0.0, z0, 1.0, 0.0, 0.0, 0.0)
    mujoco.mj_forward(model, data)
    dt = float(model.opt.timestep)
    chunk = int(round(0.01 / dt))
    limit = int(round((LEDGE_SECONDS + LEDGE_SETTLE) / dt))
    steps, max_tilt, tilt, top = 0, 0.0, 0.0, z0
    fell = False
    while steps < limit:
        mujoco.mj_step(model, data, nstep=chunk)
        steps += chunk
        z = float(data.xpos[box["body"]][2])
        top = max(top, z)
        tilt = math.degrees(math.acos(max(-1.0, min(1.0, float(data.xmat[box["body"]][8])))))
        max_tilt = max(max_tilt, tilt)
        if z < LEDGE_TOP - 0.01 or tilt > 45.0:
            fell = True
            break
        if steps >= int(round(0.3 / dt)) and steps % (10 * chunk) == 0:
            v = rig.velocity("load")
            if np.linalg.norm(v[:3]) < SETTLE_LIN and np.linalg.norm(v[3:]) < SETTLE_ANG:
                break
    outcome = "falls" if fell else ("rocks" if max_tilt >= rock_deg else "stays")
    return {"outcome": outcome, "max_tilt_deg": max_tilt, "tilt_deg": tilt, "unstable": bool(top - z0 > t8.max_rise(box["dims"])), "steps": steps}


# --------------------------------------------------------------------------
# shift — a base that may tip when a mover inside it runs out
# --------------------------------------------------------------------------

FOOT_HALF = (0.03, 0.03, 0.015)
SHIFT_SETTLE = 2.0


def shift_xml(name: str, *, base_dims: Sequence[float], base_mass: float, base_z: float, feet_x: Sequence[float], feet_y: float,
              mover_dims: Sequence[float], mover_mass: float, mover_at: Sequence[float], travel: float,
              link: tuple[Sequence[float], Sequence[float]] | None = None) -> str:
    """MJCF of a shift scene. The base (a box ``base_dims`` whose bottom is `base_z` above the floor) stands on four feet at x ∈
    `feet_x` (rear, front), y = ±`feet_y`, each reaching down to the floor. The mover (a box ``mover_dims`` centred at `mover_at` in
    the base frame when retracted) slides along +x by up to `travel` m. ``link`` = (half sizes, centre in the mover frame) of a
    collision-free arm link drawn between base and payload (no mass: the bodies carry explicit inertials)."""
    bl, bw, bh = (float(v) for v in base_dims)
    centre_z = base_z + bh / 2 + 0.0005
    # each foot reaches from the floor up into the base's bottom face (2 mm overlap)
    foot_half = (FOOT_HALF[0], FOOT_HALF[1], base_z / 2 + 0.001)
    foot_z = -bh / 2 - base_z / 2 + 0.001
    feet = ""
    for i, (fx, fy) in enumerate((x, y) for x in feet_x for y in (-feet_y, feet_y)):
        feet += (f'      <body name="foot{i}_body" pos="{fx:.6f} {fy:.6f} {foot_z:.6f}"><inertial pos="0 0 0" mass="1e-6" diaginertia="1e-9 1e-9 1e-9"/>'
                 f'<geom name="foot{i}" type="box" size="{foot_half[0]:.6f} {foot_half[1]:.6f} {foot_half[2]:.6f}" priority="1"/></body>\n')
    ixx, iyy, izz = t8.inertia("box", base_dims, base_mass)
    mxx, myy, mzz = t8.inertia("box", mover_dims, mover_mass)
    link_geom = ""
    if link is not None:
        half, centre = link
        link_geom = (f'        <geom name="link" type="box" pos="{centre[0]:.6f} {centre[1]:.6f} {centre[2]:.6f}" '
                     f'size="{half[0]:.6f} {half[1]:.6f} {half[2]:.6f}" contype="0" conaffinity="0"/>\n')
    return (
        t8._header(f"v04-{name}")
        + "  <worldbody>\n"
        '    <geom name="floor_geom" type="plane" size="10 10 0.1" priority="2"/>\n'
        f'    <body name="base" pos="0 0 {centre_z:.6f}">\n'
        "      <freejoint/>\n"
        f'      <inertial pos="{t8.NOMINAL_IPOS} {t8.NOMINAL_IPOS} {t8.NOMINAL_IPOS}" mass="{base_mass:.6f}" diaginertia="{ixx:.8f} {iyy:.8f} {izz:.8f}"/>\n'
        f'      <geom name="base_geom" type="box" size="{bl / 2:.6f} {bw / 2:.6f} {bh / 2:.6f}"/>\n'
        + feet
        + f'      <body name="mover" pos="{mover_at[0]:.6f} {mover_at[1]:.6f} {mover_at[2]:.6f}">\n'
        f'        <joint name="run" type="slide" axis="1 0 0" limited="true" range="-0.001 {travel + 0.01:.6f}" damping="0"/>\n'
        f'        <inertial pos="{t8.NOMINAL_IPOS} {t8.NOMINAL_IPOS} {t8.NOMINAL_IPOS}" mass="{mover_mass:.6f}" diaginertia="{mxx:.8f} {myy:.8f} {mzz:.8f}"/>\n'
        f'        <geom name="mover_geom" type="box" size="{mover_dims[0] / 2:.6f} {mover_dims[1] / 2:.6f} {mover_dims[2] / 2:.6f}" contype="0" conaffinity="0"/>\n'
        + link_geom
        + "      </body>\n"
        "    </body>\n"
        "  </worldbody>\n</mujoco>\n"
    )


def shift_rig(name: str, **kwargs: Any) -> ph.Rig:
    rig = ph.Rig(shift_xml(name, **kwargs), boxes=("base", "mover"), dims={"base": kwargs["base_dims"], "mover": kwargs["mover_dims"]})
    return rig


def run_shift(rig: ph.Rig, path: ph.Path1D, *, mu: float, base_com: Sequence[float], mover_mass: float, mover_com: Sequence[float] = (0.0, 0.0, 0.0),
              lift_deg: float = 1.5, tip_deg: float = 30.0) -> dict[str, Any]:
    """One shift rollout → ``{"outcome": steady|lifts|tips, "max_tilt_deg", "tilt_deg", "slide_m", "unstable", "steps"}``. The mover
    tracks `path` (internal drive), then holds; the base is read until at rest (at most :data:`SHIFT_SETTLE` s after the path).
    ``tips`` = the base tilted past `tip_deg` (it is going over); ``lifts`` = it tilted at least `lift_deg` (feet left the floor) and
    came back down."""
    feet = [name for name in (f"foot{i}" for i in range(4))]
    rig.set_hidden(friction={f: mu for f in feet}, com={"base": base_com, "mover": mover_com}, mass={"mover": mover_mass})
    model, data = rig.model, rig.data
    base, mover = rig.boxes["base"], rig.boxes["mover"]
    dof = mover["qvel"]
    qadr = mover["qpos"]
    mujoco.mj_forward(model, data)
    x0 = np.array(data.xpos[base["body"]])
    z0 = float(x0[2])
    n = int(path.q.size)
    total = n + int(round(SHIFT_SETTLE / TIMESTEP))
    max_tilt = tilt = 0.0
    top = z0
    i, early = 0, False
    while i < total:
        k = min(i, n - 1)
        q_t, v_t, a_t = (path.q[k], path.v[k], path.a[k]) if i < n else (path.q[-1], 0.0, 0.0)
        data.qfrc_applied[dof] = mover_mass * (a_t + KP * (q_t - data.qpos[qadr]) + KD * (v_t - data.qvel[dof]))
        mujoco.mj_step(model, data)
        i += 1
        if i % 10:
            continue
        tilt = math.degrees(math.acos(max(-1.0, min(1.0, float(data.xmat[base["body"]][8])))))
        max_tilt = max(max_tilt, tilt)
        top = max(top, float(data.xpos[base["body"]][2]))
        if tilt > tip_deg:
            early = True
            break
        if i >= n and i % 100 == 0:
            v = rig.velocity("base")
            if np.linalg.norm(v[:3]) < SETTLE_LIN and np.linalg.norm(v[3:]) < SETTLE_ANG:
                break
    slide = float(np.hypot(*(np.asarray(data.xpos[base["body"]])[:2] - x0[:2])))
    outcome = "tips" if early or tilt > tip_deg else ("lifts" if max_tilt >= lift_deg else "steady")
    bl, bw, bh = base["dims"]
    rise_limit = math.hypot(max(bl, bw) / 2, bh / 2 + 0.3) + 0.1
    return {"outcome": outcome, "max_tilt_deg": max_tilt, "tilt_deg": tilt, "slide_m": slide, "unstable": bool(top - z0 > rise_limit), "steps": i}
