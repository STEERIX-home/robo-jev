"""A3 MuJoCo scenes — one per law (author A3: laboratory, hospital and light manufacturing).

The contact defaults (timestep 1 ms, ``implicitfast``, elliptic cones with ``impratio`` 10, ``solref`` 0.004 s, ``condim`` 3), the box
inertia, the planned-motion integrator, the feed-forward + PD tracking of a driven body and the at-rest criterion are the shared
low-level helpers of :mod:`robo_jev.outside.physics`. The geometry, the outcomes and their thresholds are A3's own:

* **Law A — index dial** (:func:`dial_xml`, :func:`run_dial`): a square fixture plate on a rotary indexer (1.12 m square, top at
  0.82 m) turns by a stated angle in a stated time on a cycloidal cam profile (rest to rest), then dwells. A part stands free on the
  plate at a stated radius (no nest). The plate is a heavy driven body (hinge about z, feed-forward + PD on the planned angle). Hidden:
  the part-on-plate friction, the centre of mass (height fraction, offset along the long axis) and the radial placement error.
  Outcome once the plate has stopped and the part is at rest: ``falls_over`` (tilt > 45°, or off the plate), ``rocks_back`` (an
  edge lifted ≥ 10° at some moment, ends upright), ``slides_off_mark`` (moved ≥ 10 mm on the plate, never lifted 10°) or
  ``stays_put``. Soft contacts let a stuck part creep outward by ≈ 1.4 mm per (m · rad²/s) of r·∫ω² dt (≤ 5 mm for every no-slip
  case the catalogs reach), so the shift threshold is 10 mm. Why ``rocks_back`` is its own outcome: the cam's tangential
  acceleration reverses mid-index, so whether a rocking part falls depends on timing — falling alone is not monotone in the index
  time near its threshold, while "disturbed at all" (slides 10 mm, lifts 10° or falls) is (sweeps in ``author_a3.py sanity``).
* **Law B — cantilever instrument stand** (:func:`stand_xml`, :func:`run_stand`): a Mayo-type stand on four feet (a post at x = 0, a
  tray cantilevered along +x, the base reaching a stated distance ahead of the post). A robot releases an instrument container at rest
  a stated height above the tray, its centre a stated distance from the post. Hidden: the container mass and its centre-of-mass offset
  along the tray, the stand's base mass, the set-down error along the tray. Outcome: ``tips_over`` (the stand ends tilted > 20°),
  ``rocks`` (the stand tilted ≥ 0.5° at some moment — rear feet lifted about 3 mm or more — and came back) or ``stays_level``.
* **Law C — hold-down clamp** (:func:`clamp_xml`, :func:`run_clamp`): a part lies on a fixture nest under a pneumatic hold-down pad
  (a body on a vertical slide pressed down with the clamp force); a cobot presses a fitting into its side — a horizontal half-sine
  force pulse through the part's centre of mass. Hidden: the part-on-nest friction, the pad-on-part friction and the clamp's force
  factor (actual / set). Outcome: ``breaks_free`` (moved ≥ 8 mm), ``creeps`` (moved 1–8 mm) or ``stays_put``.

Every rollout is deterministic: the same model, state and hidden values give bit-identical results (a test checks).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import mujoco
import numpy as np

from robo_jev.outside import physics

__all__ = [
    "CLAMP_BREAK_M",
    "CLAMP_CREEP_M",
    "DIAL_SHIFT_M",
    "DIAL_TOP",
    "OUTCOMES",
    "STAND_ROCK_DEG",
    "STAND_TIP_DEG",
    "ClampModel",
    "DialModel",
    "StandModel",
    "clamp_capacity",
    "clamp_xml",
    "dial_peak_accel",
    "dial_slip",
    "dial_xml",
    "index_profile",
    "run_clamp",
    "run_dial",
    "run_stand",
    "stand_xml",
]

TIMESTEP = physics.TIMESTEP
GRAVITY = physics.GRAVITY
#: Outcome categories per scene, mildest first (the key event is every category but the first).
OUTCOMES = {
    "indexdial": ("stays_put", "slides_off_mark", "rocks_back", "falls_over"),
    "mayostand": ("stays_level", "rocks", "tips_over"),
    "holddown": ("stays_put", "creeps", "breaks_free"),
}


def _quat_tilt_deg(xmat: np.ndarray) -> float:
    return math.degrees(math.acos(max(-1.0, min(1.0, float(xmat[8])))))


def _box_inertia(dims: Sequence[float], mass: float) -> tuple[float, float, float]:
    return physics.inertia("box", dims, mass)


# --------------------------------------------------------------------------
# Law A — index dial
# --------------------------------------------------------------------------

DIAL_Z = 0.80
DIAL_HALF = (0.56, 0.56, 0.02)
DIAL_TOP = DIAL_Z + DIAL_HALF[2]
DIAL_MASS = 300.0
DIAL_IZZ = 72.0
DIAL_SHIFT_M = 0.010
DIAL_FALL_DEG = 45.0
DIAL_ROCK_DEG = 10.0
DIAL_PRE = 0.1
DIAL_DWELL = 0.3


def dial_xml(radial: float, tangential: float, height: float, mass: float, com: Sequence[float] = (0.0, 0.0, 0.0)) -> str:
    """MJCF of the index dial: the plate (hinge about z through its centre) and the part (box with ``radial`` × ``tangential`` ×
    ``height`` extents along the plate's radial, tangential and vertical directions at its start position on +x; centre of mass at
    `com` from its geometric centre — compiled in, never changed afterwards: MuJoCo keeps a body's collision bounding boxes in its
    compile-time inertial frame, so moving ``body_ipos`` later loses contacts on multi-geom bodies)."""
    ixx, iyy, izz = _box_inertia((radial, tangential, height), mass)
    return (
        physics._header("a3-indexdial")
        + "  <worldbody>\n"
        f'    <body name="dial" pos="0 0 {DIAL_Z}">\n'
        '      <joint name="dial_yaw" type="hinge" axis="0 0 1"/>\n'
        f'      <inertial pos="0 0 0" mass="{DIAL_MASS}" diaginertia="{DIAL_IZZ / 2} {DIAL_IZZ / 2} {DIAL_IZZ}"/>\n'
        f'      <geom name="plate" type="box" size="{DIAL_HALF[0]} {DIAL_HALF[1]} {DIAL_HALF[2]}" priority="1"/>\n'
        "    </body>\n"
        f'    <body name="part" pos="0.3 0 {DIAL_TOP + height / 2 + 0.0005:.6f}">\n'
        "      <freejoint/>\n"
        f'      <inertial pos="{com[0]:.6f} {com[1]:.6f} {com[2]:.6f}" mass="{mass:.6f}" diaginertia="{ixx:.8f} {iyy:.8f} {izz:.8f}"/>\n'
        f'      <geom name="part_geom" type="box" size="{radial / 2:.6f} {tangential / 2:.6f} {height / 2:.6f}"/>\n'
        "    </body>\n"
        "  </worldbody>\n</mujoco>\n"
    )


def index_profile(angle_deg: float, index_time: float, *, pre: float = DIAL_PRE, dwell: float = DIAL_DWELL, dt: float = TIMESTEP) -> physics.Profile:
    """Cycloidal index (rest to rest): θ(t) = Θ (t/T − sin(2πt/T) / 2π); the angular acceleration is Θ/T² · 2π sin(2πt/T). `pre` s at
    rest before, `dwell` s at rest after. One column (the dial angle)."""
    theta, T = math.radians(float(angle_deg)), float(index_time)
    if theta <= 0 or T <= 0:
        raise ValueError("index needs angle > 0 and time > 0")
    n_pre, n_t, n_d = (int(round(x / dt)) for x in (pre, T, dwell))
    t = (np.arange(n_t) + 0.5) * dt
    acc = theta / T**2 * 2 * math.pi * np.sin(2 * math.pi * t / T)
    a = np.concatenate([np.zeros(n_pre), acc, np.zeros(n_d)])
    # remove the rounding remainder so the plate ends exactly at rest
    a[n_pre : n_pre + n_t] -= a[n_pre : n_pre + n_t].sum() / max(n_t, 1)
    pos, vel = physics._integrate(a[:, None], np.zeros(1), dt)
    return physics.Profile(q=pos[:-1], v=vel[:-1], a=a[:, None], start=n_pre, end=n_pre + n_t)


def dial_peak_accel(angle_deg: float, index_time: float, radius: float) -> float:
    """Peak in-plane acceleration (m/s²) of a point at `radius` on the plate during the cycloidal index: r · max_t √(α² + ω⁴)."""
    theta, T = math.radians(float(angle_deg)), float(index_time)
    s = np.linspace(0.0, 1.0, 401)
    alpha = theta / T**2 * 2 * math.pi * np.sin(2 * math.pi * s)
    omega = theta / T * (1 - np.cos(2 * math.pi * s))
    return float(radius * np.max(np.sqrt(alpha**2 + omega**4)))


#: Soft-contact creep of a stuck part (m per m · rad²/s of r ∫ω² dt), measured with μ = 1.15 on the dial (module docstring).
DIAL_CREEP = 0.00141


def dial_slip(angle_deg: float, index_time: float, radius: np.ndarray, mu: np.ndarray, *, steps: int = 160) -> np.ndarray:
    """Analytic aid (not a label): how far a point mass with Coulomb friction `mu` slides on the plate (m, in the plate frame) during the
    cycloidal index and its run-out, starting at rest at `radius` — vectorised over draws; plus the soft-contact creep MuJoCo shows."""
    theta, T = math.radians(float(angle_deg)), float(index_time)
    r = np.asarray(radius, dtype=float)
    mu = np.asarray(mu, dtype=float) * np.ones_like(r)
    dt = T / steps
    px, py = r.copy(), np.zeros_like(r)
    vx, vy = np.zeros_like(r), np.zeros_like(r)
    g_mu = mu * GRAVITY
    for i in range(steps):
        s = (i + 0.5) / steps
        om = theta / T * (1 - math.cos(2 * math.pi * s))
        al = theta / T**2 * 2 * math.pi * math.sin(2 * math.pi * s)
        ux, uy = -om * py, om * px
        wx, wy = vx - ux, vy - uy
        speed = np.hypot(wx, wy)
        ax_s, ay_s = -al * py - om**2 * px, al * px - om**2 * py
        need = np.hypot(ax_s, ay_s)
        stuck = speed <= g_mu * dt
        scale = np.where(need > g_mu, g_mu / np.maximum(need, 1e-12), 1.0)
        ax = np.where(stuck, ax_s * scale, -g_mu * wx / np.maximum(speed, 1e-12))
        ay = np.where(stuck, ay_s * scale, -g_mu * wy / np.maximum(speed, 1e-12))
        # a stuck part that can hold keeps the plate's velocity exactly
        hold = stuck & (need <= g_mu)
        vx = np.where(hold, ux + ax * dt, vx + ax * dt)
        vy = np.where(hold, uy + ay * dt, vy + ay * dt)
        px, py = px + vx * dt, py + vy * dt
    c, sn = math.cos(-theta), math.sin(-theta)
    rx, ry = c * px - sn * py, sn * px + c * py
    run_out = np.hypot(vx, vy) ** 2 / (2 * np.maximum(g_mu, 1e-9))  # the plate has stopped; the part slides until friction stops it
    omega_sq = 1.5 * theta**2 / T  # ∫ω² dt of the cycloid
    return np.hypot(rx - r, ry) + run_out + DIAL_CREEP * r * omega_sq


@dataclass
class DialModel:
    xml: str
    model: Any
    data: Any
    dial: int
    plate: int
    part: int
    qpos: int
    height: float

    @classmethod
    def build(cls, radial: float, tangential: float, height: float, mass: float, com: Sequence[float] = (0.0, 0.0, 0.0)) -> "DialModel":
        xml = dial_xml(radial, tangential, height, mass, com)
        model = mujoco.MjModel.from_xml_string(xml)
        data = mujoco.MjData(model)
        part = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "part")
        return cls(xml, model, data, mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "dial"),
                   mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "plate"), part,
                   int(model.jnt_qposadr[model.body_jntadr[part]]), float(height))


def _part_velocity_small(data: Any, qvel: int) -> bool:
    v = np.asarray(data.qvel[qvel : qvel + 6], dtype=float)
    return bool(np.linalg.norm(v[:3]) < physics.SETTLE_LIN and np.linalg.norm(v[3:]) < physics.SETTLE_ANG)


def run_dial(m: DialModel, profile: physics.Profile, *, radius: float, mu: float) -> dict[str, Any]:
    """One index → ``{"outcome", "moved_m", "tilt", "max_tilt", "fell"}``. `radius` is where the part's centre starts (stated radius
    plus the placement error); the centre of mass is compiled into `m`."""
    model, data = m.model, m.data
    model.geom_friction[m.plate, 0] = float(mu)
    mujoco.mj_resetData(model, data)
    mujoco.mj_setConst(model, data)
    mujoco.mj_resetData(model, data)
    q = m.qpos
    data.qpos[q : q + 3] = (float(radius), 0.0, DIAL_TOP + m.height / 2 + 0.0005)
    data.qpos[q + 3 : q + 7] = (1.0, 0.0, 0.0, 0.0)
    mujoco.mj_forward(model, data)
    start = np.array(data.xpos[m.part][:2])
    qvel = int(model.jnt_dofadr[model.body_jntadr[m.part]])
    max_tilt, early = 0.0, False
    for i in range(profile.steps):
        physics._track(data, profile, i, (0,), (DIAL_IZZ,))
        mujoco.mj_step(model, data)
        if i % 10 == 0:
            tilt = _quat_tilt_deg(data.xmat[m.part])
            max_tilt = max(max_tilt, tilt)
            if tilt > 60.0 or data.xpos[m.part][2] < DIAL_TOP - 0.10:
                early = True
                break
    if not early:
        last = float(profile.q[-1, 0] + profile.v[-1, 0] * TIMESTEP)
        for i in range(int(round(physics.SETTLE_MAX / TIMESTEP))):
            if i % 100 == 0 and _part_velocity_small(data, qvel):
                break
            data.qfrc_applied[0] = DIAL_IZZ * (physics.KP * (last - data.qpos[0]) - physics.KD * data.qvel[0])
            mujoco.mj_step(model, data)
            if i % 10 == 0:
                tilt = _quat_tilt_deg(data.xmat[m.part])
                max_tilt = max(max_tilt, tilt)
                if tilt > 60.0 or data.xpos[m.part][2] < DIAL_TOP - 0.10:
                    early = True
                    break
    tilt = _quat_tilt_deg(data.xmat[m.part])
    max_tilt = max(max_tilt, tilt)
    yaw = float(data.qpos[0])
    c, s = math.cos(-yaw), math.sin(-yaw)
    rel = np.array(data.xpos[m.part][:2])
    rel_plate = np.array([c * rel[0] - s * rel[1], s * rel[0] + c * rel[1]])
    moved = float(np.hypot(*(rel_plate - start)))
    fell = bool(data.xpos[m.part][2] < DIAL_TOP - 0.05)
    if tilt > DIAL_FALL_DEG or fell:
        outcome = "falls_over"
    elif max_tilt >= DIAL_ROCK_DEG:
        outcome = "rocks_back"
    elif moved >= DIAL_SHIFT_M:
        outcome = "slides_off_mark"
    else:
        outcome = "stays_put"
    return {"outcome": outcome, "moved_m": moved, "tilt": tilt, "max_tilt": max_tilt, "fell": fell}


# --------------------------------------------------------------------------
# Law B — cantilever instrument stand
# --------------------------------------------------------------------------

STAND_REAR_X = -0.14
STAND_HALF_Y = 0.21
FOOT_HALF = (0.025, 0.025, 0.015)
BASE_THICK = 0.025
POST_HALF = 0.0175
POST_MASS = 1.2
TRAY_MASS = 1.8
TRAY_WIDTH = 0.38
TRAY_THICK = 0.012
TRAY_BACK = 0.05  # the tray starts this far behind the post axis
STAND_TIP_DEG = 20.0
STAND_ROCK_DEG = 0.5
STAND_RUN = 2.0


def _stand_parts(reach: float, tray_len: float, tray_top: float) -> list[tuple[str, tuple[float, float, float], tuple[float, float, float]]]:
    """(name, centre, half size) of the stand's boxes in the stand frame (origin on the floor under the post)."""
    base_len = reach - STAND_REAR_X
    base_cx = (reach + STAND_REAR_X) / 2
    feet_top = 2 * FOOT_HALF[2]
    post_bottom = feet_top + BASE_THICK
    tray_mid_z = tray_top - TRAY_THICK / 2
    post_h = tray_mid_z - TRAY_THICK / 2 - post_bottom
    parts = [("base", (base_cx, 0.0, feet_top + BASE_THICK / 2), (base_len / 2, STAND_HALF_Y, BASE_THICK / 2)),
             ("post", (0.0, 0.0, post_bottom + post_h / 2), (POST_HALF, POST_HALF, post_h / 2)),
             ("tray", (tray_len / 2 - TRAY_BACK, 0.0, tray_mid_z), (tray_len / 2, TRAY_WIDTH / 2, TRAY_THICK / 2))]
    for i, (fx, fy) in enumerate(((STAND_REAR_X, -1), (STAND_REAR_X, 1), (reach, -1), (reach, 1))):
        x = fx + (FOOT_HALF[0] if fx == STAND_REAR_X else -FOOT_HALF[0])
        parts.append((f"foot{i}", (x, fy * (STAND_HALF_Y - FOOT_HALF[1]), FOOT_HALF[2]), FOOT_HALF))
    return parts


def stand_inertial(reach: float, tray_len: float, tray_top: float, base_mass: float) -> tuple[float, np.ndarray, np.ndarray]:
    """Mass, centre of mass and the diagonal of the inertia tensor about it (stand frame axes) of the stand: base box (with its feet)
    of `base_mass`, post and tray of fixed masses, each a uniform box. The y-symmetric stand has I_xy = I_yz = 0, so tipping about y is
    exact with the diagonal; I_xz is dropped."""
    parts = {name: (np.array(c), np.array(h)) for name, c, h in _stand_parts(reach, tray_len, tray_top)}
    masses = {"base": float(base_mass), "post": POST_MASS, "tray": TRAY_MASS}
    total = sum(masses.values())
    com = sum(masses[n] * parts[n][0] for n in masses) / total
    diag = np.zeros(3)
    for name, mass in masses.items():
        centre, half = parts[name]
        own = np.array(_box_inertia(2 * half, mass))
        d = centre - com
        diag += own + mass * np.array([d[1] ** 2 + d[2] ** 2, d[0] ** 2 + d[2] ** 2, d[0] ** 2 + d[1] ** 2])
    return total, com, diag


def stand_xml(reach: float, tray_len: float, tray_top: float, box: Sequence[float], *, base_mass: float, load_mass: float,
              load_com_x: float) -> str:
    """MJCF of the stand scene: the floor, the stand (free body: base, post, tray, four feet — only the feet touch the floor) and the
    container (free box `box` = length along the tray, width, height). The hidden masses and centres of mass are compiled in (one
    model per draw — see :func:`dial_xml` on why ``body_ipos`` is never changed after compiling)."""
    mass, com, diag = stand_inertial(reach, tray_len, tray_top, base_mass)
    cl, cw, ch = (float(v) for v in box)
    cixx, ciyy, cizz = _box_inertia((cl, cw, ch), load_mass)
    geoms = []
    for name, centre, half in _stand_parts(reach, tray_len, tray_top):
        extra = ' contype="0" conaffinity="0"' if name in ("base", "post") else ""
        geoms.append(f'      <geom name="{name}" type="box" pos="{centre[0]:.6f} {centre[1]:.6f} {centre[2]:.6f}" '
                     f'size="{half[0]:.6f} {half[1]:.6f} {half[2]:.6f}"{extra}/>\n')
    return (
        physics._header("a3-mayostand")
        + "  <worldbody>\n"
        '    <geom name="floor" type="plane" size="4 4 0.1" friction="0.8 0.005 0.0001"/>\n'
        '    <body name="stand" pos="0 0 0.0005">\n'
        "      <freejoint/>\n"
        f'      <inertial pos="{com[0]:.6f} {com[1]:.6f} {com[2]:.6f}" mass="{mass:.6f}" diaginertia="{diag[0]:.8f} {diag[1]:.8f} {diag[2]:.8f}"/>\n'
        + "".join(geoms)
        + "    </body>\n"
        f'    <body name="load" pos="0.3 0 {tray_top + ch / 2 + 0.01:.6f}">\n'
        "      <freejoint/>\n"
        f'      <inertial pos="{float(load_com_x):.6f} 0 0" mass="{float(load_mass):.6f}" diaginertia="{cixx:.8f} {ciyy:.8f} {cizz:.8f}"/>\n'
        f'      <geom name="load_geom" type="box" size="{cl / 2:.6f} {cw / 2:.6f} {ch / 2:.6f}" friction="0.6 0.005 0.0001"/>\n'
        "    </body>\n"
        "  </worldbody>\n</mujoco>\n"
    )


@dataclass
class StandModel:
    xml: str
    model: Any
    data: Any
    stand: int
    load: int
    reach: float
    tray_len: float
    tray_top: float
    box: tuple[float, float, float]

    @classmethod
    def build(cls, reach: float, tray_len: float, tray_top: float, box: Sequence[float], *, base_mass: float, load_mass: float,
              load_com_x: float) -> "StandModel":
        xml = stand_xml(reach, tray_len, tray_top, box, base_mass=base_mass, load_mass=load_mass, load_com_x=load_com_x)
        model = mujoco.MjModel.from_xml_string(xml)
        data = mujoco.MjData(model)
        ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, n) for n in ("stand", "load")]
        return cls(xml, model, data, ids[0], ids[1], float(reach), float(tray_len), float(tray_top), tuple(float(v) for v in box))


def run_stand(m: StandModel, *, setdown_x: float, release_h: float, seconds: float = STAND_RUN) -> dict[str, Any]:
    """One set-down → ``{"outcome", "max_tilt", "tilt", "lift_m"}``. The container starts at rest with its centre at `setdown_x` from
    the post axis and its bottom `release_h` above the tray (masses and centres of mass are compiled into `m`)."""
    model, data = m.model, m.data
    mujoco.mj_resetData(model, data)
    lq = int(model.jnt_qposadr[model.body_jntadr[m.load]])
    data.qpos[lq : lq + 3] = (float(setdown_x), 0.0, m.tray_top + m.box[2] / 2 + float(release_h) + 0.0005)
    data.qpos[lq + 3 : lq + 7] = (1.0, 0.0, 0.0, 0.0)
    mujoco.mj_forward(model, data)
    z0 = float(data.xpos[m.stand][2])
    max_tilt, lift = 0.0, 0.0
    sv = int(model.jnt_dofadr[model.body_jntadr[m.stand]])
    lv = int(model.jnt_dofadr[model.body_jntadr[m.load]])
    n_run = int(round(seconds / TIMESTEP))
    n_max = n_run + int(round(physics.SETTLE_MAX / TIMESTEP))
    for i in range(n_max):
        if i >= n_run and i % 100 == 0 and _part_velocity_small(data, sv) and _part_velocity_small(data, lv):
            break
        mujoco.mj_step(model, data)
        if i % 5 == 0:
            tilt = _quat_tilt_deg(data.xmat[m.stand])
            max_tilt = max(max_tilt, tilt)
            lift = max(lift, float(data.xpos[m.stand][2]) - z0)
            if tilt > 35.0:
                break
    tilt = _quat_tilt_deg(data.xmat[m.stand])
    max_tilt = max(max_tilt, tilt)
    if tilt > STAND_TIP_DEG:
        outcome = "tips_over"
    elif max_tilt >= STAND_ROCK_DEG:
        outcome = "rocks"
    else:
        outcome = "stays_level"
    return {"outcome": outcome, "max_tilt": max_tilt, "tilt": tilt, "lift_m": lift}


# --------------------------------------------------------------------------
# Law C — hold-down clamp
# --------------------------------------------------------------------------

NEST_TOP = 0.90
NEST_HALF = (0.22, 0.16, 0.02)
PAD_HALF = (0.014, 0.014, 0.006)
CLAMP_MASS = 0.4
CLAMP_CREEP_M = 0.001
CLAMP_BREAK_M = 0.008
CLAMP_PRE = 0.20
CLAMP_RAMP = 0.06
CLAMP_POST = 0.5


def clamp_xml(dims: Sequence[float], mass: float) -> str:
    """MJCF of the clamp scene: the nest (static box, priority 1), the part (free box centred at the origin on the nest) and the
    hold-down (a body on a vertical slide carrying the pad, priority 2, centred over the part)."""
    length, width, height = (float(v) for v in dims)
    ixx, iyy, izz = _box_inertia((length, width, height), mass)
    n = physics.NOMINAL_IPOS
    pad_z = NEST_TOP + height + PAD_HALF[2] + 0.0003
    return (
        physics._header("a3-holddown")
        + "  <worldbody>\n"
        f'    <geom name="nest" type="box" pos="0 0 {NEST_TOP - NEST_HALF[2]:.6f}" size="{NEST_HALF[0]} {NEST_HALF[1]} {NEST_HALF[2]}" priority="1"/>\n'
        f'    <body name="part" pos="0 0 {NEST_TOP + height / 2 + 0.0003:.6f}">\n'
        "      <freejoint/>\n"
        f'      <inertial pos="{n} {n} {n}" mass="{mass:.6f}" diaginertia="{ixx:.8f} {iyy:.8f} {izz:.8f}"/>\n'
        f'      <geom name="part_geom" type="box" size="{length / 2:.6f} {width / 2:.6f} {height / 2:.6f}"/>\n'
        "    </body>\n"
        f'    <body name="clamp" pos="0 0 {pad_z:.6f}">\n'
        '      <joint name="clamp_z" type="slide" axis="0 0 1"/>\n'
        f'      <inertial pos="0 0 0" mass="{CLAMP_MASS}" diaginertia="0.001 0.001 0.001"/>\n'
        f'      <geom name="pad" type="box" size="{PAD_HALF[0]} {PAD_HALF[1]} {PAD_HALF[2]}" priority="2"/>\n'
        "    </body>\n"
        "  </worldbody>\n</mujoco>\n"
    )


def clamp_capacity(*, mu_nest: float, mu_pad: float, clamp_force: float, mass: float) -> float:
    """Static holding capacity (N) against a horizontal push through the centre of mass: μ_nest (m g + F + m_clamp g) + μ_pad (F + m_clamp g)."""
    down = float(clamp_force) + CLAMP_MASS * GRAVITY
    return float(mu_nest) * (float(mass) * GRAVITY + down) + float(mu_pad) * down


@dataclass
class ClampModel:
    xml: str
    model: Any
    data: Any
    nest: int
    pad: int
    part: int
    clamp_dof: int

    @classmethod
    def build(cls, dims: Sequence[float], mass: float) -> "ClampModel":
        xml = clamp_xml(dims, mass)
        model = mujoco.MjModel.from_xml_string(xml)
        data = mujoco.MjData(model)
        clamp = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "clamp")
        return cls(xml, model, data, mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "nest"),
                   mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "pad"), mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "part"),
                   int(model.jnt_dofadr[model.body_jntadr[clamp]]))


def run_clamp(m: ClampModel, *, press_force: float, press_time: float, clamp_force: float, mu_nest: float, mu_pad: float) -> dict[str, Any]:
    """One press → ``{"outcome", "moved_m"}``: the clamp force ramps in over :data:`CLAMP_RAMP` s and holds; after :data:`CLAMP_PRE` s a
    half-sine push of peak `press_force` (N, along −x) and duration `press_time` (s) acts through the part's centre of mass; then
    :data:`CLAMP_POST` s and until the part is at rest."""
    model, data = m.model, m.data
    model.geom_friction[m.nest, 0] = float(mu_nest)
    model.geom_friction[m.pad, 0] = float(mu_pad)
    mujoco.mj_resetData(model, data)
    mujoco.mj_setConst(model, data)
    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)
    n_pre, n_press, n_post = (int(round(x / TIMESTEP)) for x in (CLAMP_PRE, press_time, CLAMP_POST))
    n_ramp = int(round(CLAMP_RAMP / TIMESTEP))
    pv = int(model.jnt_dofadr[model.body_jntadr[m.part]])
    start = None
    total = n_pre + n_press + n_post
    i = 0
    while True:
        if i < total:
            data.qfrc_applied[m.clamp_dof] = -float(clamp_force) * min(1.0, (i + 1) / n_ramp)
        if i == n_pre:
            start = float(data.xpos[m.part][0])
        if n_pre <= i < n_pre + n_press:
            phase = (i - n_pre + 0.5) / n_press
            data.xfrc_applied[m.part, 0] = -float(press_force) * math.sin(math.pi * phase)
        else:
            data.xfrc_applied[m.part, 0] = 0.0
        mujoco.mj_step(model, data)
        i += 1
        if i >= total and (i - total) % 100 == 0 and (_part_velocity_small(data, pv) or i - total >= int(round(physics.SETTLE_MAX / TIMESTEP))):
            break
        if start is not None and abs(float(data.xpos[m.part][0]) - start) > 0.05:
            break
    moved = abs(float(data.xpos[m.part][0]) - (start if start is not None else 0.0))
    if moved >= CLAMP_BREAK_M:
        outcome = "breaks_free"
    elif moved >= CLAMP_CREEP_M:
        outcome = "creeps"
    else:
        outcome = "stays_put"
    return {"outcome": outcome, "moved_m": moved}
