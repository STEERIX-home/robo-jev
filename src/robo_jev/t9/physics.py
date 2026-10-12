"""T9 physics — six small MuJoCo scenes, one rollout per hidden draw (Task T9 §1-1).

Every scene is a short MJCF written here (the geometry is T9's own); only the integrator constants (time step, gravity, PD gains)
come from :mod:`robo_jev.outside.physics`. A scene compiles once per moment; the hidden values of a draw are written into the
compiled model before each rollout (:meth:`_Scene.prepare`), and every rollout starts from ``mj_resetData`` — the same model,
action and draw give a bit-identical rollout.

Shared contact model: 1 ms step, ``implicitfast``, elliptic cones, ``impratio`` 10, contact ``solref`` 0.004 s (damping ratio 1),
the surface that carries a hidden friction has geom priority 1. Loads that stand or slide are boxes (MuJoCo's dedicated box
colliders; T8 found the general convex cylinder collider unstable); round items are capsules lying on their side or spheres,
which meet the floor (a plane) and the rounded lip (a capsule) through dedicated colliders.

* **roll** (unseen law: rolling) — a round item on a tray that tilts. Simulated in the tray's frame: the tray floor is a plane, the
  lip at its low edge a capsule bead of height ``h`` (radius h/2), and the tilt enters as the direction of gravity, ramped in over
  ``ramp`` s and held (the rotational pseudo-forces of the slow tilt, ≤ ω²·r ≈ 0.03 m/s² and α·r ≈ 0.15 m/s² during the ramp, are
  left out). Rolling resistance is MuJoCo's rolling friction (``condim`` 6, a length: resistive torque ≤ μ_r·N) of the tray's
  lining (the lip is a hard rim with negligible rolling resistance), the item's spin
  inertia is ``k·m·r²`` (k from the contents). Outcome: ``over`` (the centre passed the bead's axis), ``caught`` (rolled ≥ 1 cm and
  stopped on the tray), ``put`` (moved < 1 cm).
* **swing** (unseen law: pendulum) — a box hanging rigidly below a strap from a hook on a heavy carriage that moves at ``v`` and
  stops at ``a`` (jerk-limited); a fixed upright stands ``gap`` ahead of the load's front face at its rest position under the stopped
  hook. Strap length = nominal × (1 + compliance × mass) + hook slop (hidden). Outcome at the first touch: ``strike`` (the touching
  point moves toward the upright at ≥ :data:`STRIKE_SPEED`), ``brush`` (slower), else ``clear`` once the swing turns back.
* **knock** (unseen law: impulse at a height) — a standing box; an arm link (horizontal capsule) at height ``h`` moves toward it at
  ``v``; at the first touch the arm's reaction brakes it to rest over ``stop`` m, holds, and backs off 10 cm. Outcome after settling:
  ``topple`` (tilt > 45°), ``slide`` (upright, moved ≥ 1 cm), ``stay``.
* **conveyor** (seen law: tip / slide under acceleration, new scene) — a box on a belt that stops or starts; simulated in the belt's
  frame (exact for a translating frame: the belt's acceleration enters as a horizontal gravity pulse). Outcome after settling:
  ``topple`` (tilt > 45°), ``slide`` (upright, moved ≥ 2 cm), ``stay``.
* **overhang** (seen law: centre of mass past an edge, new scene) — a board lies on a shelf with its end sticking out by ``over``;
  an item is set down on it (released 5 mm above, centre ``inset`` in from the board's outer end). Outcome after settling:
  ``board_tips`` (the board tilted > 10° or dropped > 2 cm), ``item_falls`` (the item ended below the board's top), ``stays``.
* **suction** (seen law: friction-limited holding, new mechanism) — a panel held under a vacuum cup (a square pad of the cup's area)
  on a carriage that accelerates to ``speed`` at ``a``, carries, and stops at ``a``. The vacuum is a force ΔP·A pulling the panel
  up at the cup's centre while the seal holds; the seal is lost for good when the panel tilts > 3° against the pad, gaps > 3 mm,
  or the cup's centre comes within a cup radius of the panel's edge. Friction pad/panel is hidden. Outcome: ``drop`` (seal lost),
  ``shift`` (held, slipped ≥ 3 mm at some point of the move), ``hold``.
"""

from __future__ import annotations

import math
from typing import Any, Callable, Sequence

import mujoco
import numpy as np

from robo_jev.outside.physics import GRAVITY, KD, KP, TIMESTEP

__all__ = [
    "FAMILIES",
    "OUTCOMES",
    "STRIKE_SPEED",
    "box_inertia",
    "make_scene",
    "speed_change",
]

G = GRAVITY
SOLREF = "0.004 1"
#: Outcome categories per family (the key event is the last one).
OUTCOMES: dict[str, tuple[str, ...]] = {
    "roll": ("put", "caught", "over"),
    "swing": ("clear", "brush", "strike"),
    "knock": ("stay", "slide", "topple"),
    "conveyor": ("stay", "slide", "topple"),
    "overhang": ("stays", "item_falls", "board_tips"),
    "suction": ("hold", "shift", "drop"),
}
FAMILIES = tuple(OUTCOMES)
#: swing: a touch at this speed or more (m/s, toward the upright) is a strike, below it a brush.
STRIKE_SPEED = 0.25
TOPPLE_DEG = 45.0
#: roll: rolling resistance of the hard lip (m).
LIP_ROLL = 0.00001
NOMINAL_IPOS = 0.001


def box_inertia(dims: Sequence[float], mass: float) -> tuple[float, float, float]:
    """Principal inertia of a uniform solid box ``dims = (x, y, z)`` about its centre."""
    a, b, c = (float(v) for v in dims)
    return (mass * (b * b + c * c) / 12, mass * (a * a + c * c) / 12, mass * (a * a + b * b) / 12)


def _header(name: str, condim: int = 3) -> str:
    return (
        f'<mujoco model="{name}">\n'
        f'  <option timestep="{TIMESTEP}" gravity="0 0 {-G}" integrator="implicitfast" cone="elliptic" impratio="10"/>\n'
        f'  <default><geom solref="{SOLREF}" condim="{condim}" friction="0.5 0.005 0.0001"/></default>\n'
    )


def speed_change(dv: float, rate: float, ramp: float, dt: float = TIMESTEP) -> np.ndarray:
    """Accelerations (one per step) that change a speed by exactly `dv` at a level of ``rate`` (same sign as dv): a linear ramp up
    over `ramp` s, a hold, a ramp down (a triangle when the change is too small for the full ramps)."""
    dv, rate = float(dv), abs(float(rate))
    if dv == 0 or rate <= 0:
        return np.zeros(0)
    sign = 1.0 if dv > 0 else -1.0
    total = abs(dv)
    ramp = min(float(ramp), total / rate)  # triangle if the change is small: the peak stays below `rate`
    peak = rate if ramp * rate <= total else total / ramp
    hold = max(0.0, total / peak - ramp)
    n_r, n_h = int(round(ramp / dt)), int(round(hold / dt))
    up = (np.arange(n_r) + 0.5) / max(n_r, 1)
    acc = peak * np.concatenate([up, np.ones(n_h), up[::-1]])
    acc *= total / (acc.sum() * dt)  # exact speed change
    return sign * acc


def _rest(qvel: np.ndarray, lin: float = 0.02, ang: float = 0.05) -> bool:
    return bool(np.linalg.norm(qvel[:3]) < lin and np.linalg.norm(qvel[3:6]) < ang)


def _tilt_deg(data: Any, body: int) -> float:
    return math.degrees(math.acos(max(-1.0, min(1.0, float(data.xmat[body][8])))))


class _Scene:
    """A compiled MJCF scene. Subclasses define :meth:`xml`, :meth:`prepare` (write a draw's hidden values) and :meth:`run`."""

    family = ""

    def __init__(self, visible: dict[str, Any]):
        self.visible = dict(visible)
        self.xml_text = self.xml()
        self.model = mujoco.MjModel.from_xml_string(self.xml_text)
        self.data = mujoco.MjData(self.model)

    def xml(self) -> str:  # pragma: no cover - abstract
        raise NotImplementedError

    def body(self, name: str) -> int:
        return mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)

    def geom(self, name: str) -> int:
        return mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, name)

    def _refresh(self) -> None:
        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_setConst(self.model, self.data)
        mujoco.mj_resetData(self.model, self.data)

    def unstable(self) -> bool:
        """The rollout left the finite range or MuJoCo flagged a bad acceleration (the moment is then dropped)."""
        bad = int(self.data.warning[int(mujoco.mjtWarning.mjWARN_BADQACC)].number)
        return bool(bad > 0 or not np.all(np.isfinite(self.data.qpos)) or not np.all(np.isfinite(self.data.qvel)))

    def touching(self, a: int, b: int) -> int | None:
        """Index of a contact between geoms `a` and `b`, else None."""
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            if (c.geom1 == a and c.geom2 == b) or (c.geom1 == b and c.geom2 == a):
                return i
        return None


# --------------------------------------------------------------------------
# roll
# --------------------------------------------------------------------------


class RollScene(_Scene):
    """visible: ``shape`` (capsule | sphere), ``radius``, ``length`` (capsule), ``mass``, ``lip`` (height), ``dist`` (contact line to
    the lip's foot). Draw: ``mu_roll`` (m), ``k`` (spin inertia / m r²), ``mu`` (sliding friction)."""

    family = "roll"

    def xml(self) -> str:
        v = self.visible
        r, h = float(v["radius"]), float(v["lip"])
        rho = h / 2
        item = (f'<geom name="item" type="capsule" fromto="0 {-v["length"] / 2 + r:.6f} 0 0 {v["length"] / 2 - r:.6f} 0" size="{r:.6f}" condim="6"/>'
                if v["shape"] == "capsule" else f'<geom name="item" type="sphere" size="{r:.6f}" condim="6"/>')
        return (
            _header("t9-roll", condim=6)
            + "  <worldbody>\n"
            '    <geom name="floor" type="plane" size="4 4 0.1" priority="1" condim="6"/>\n'
            f'    <geom name="lip" type="capsule" fromto="0 -1 {rho:.6f} 0 1 {rho:.6f}" size="{rho:.6f}" priority="1" condim="6"/>\n'
            f'    <body name="item" pos="{-float(v["dist"]):.6f} 0 {r + 0.0001:.6f}">\n'
            "      <freejoint/>\n"
            f'      <inertial pos="{NOMINAL_IPOS} 0 0" mass="{float(v["mass"]):.6f}" diaginertia="0.001 0.001 0.001"/>\n'
            f"      {item}\n"
            "    </body>\n"
            "  </worldbody>\n</mujoco>\n"
        )

    def prepare(self, draw: dict[str, float]) -> None:
        v = self.visible
        m, r = float(v["mass"]), float(v["radius"])
        spin = float(draw["k"]) * m * r * r
        if v["shape"] == "sphere":
            inertia = (spin, spin, spin)
        else:
            across = m * (3 * r * r + float(v["length"]) ** 2) / 12
            inertia = (across, spin, across)  # capsule axis along y
        item = self.body("item")
        self.model.body_inertia[item] = inertia
        self.model.body_ipos[item] = (NOMINAL_IPOS, 0.0, 0.0)
        # the hidden rolling resistance belongs to the tray's lining; the lip is a hard rim (negligible rolling resistance)
        self.model.geom_friction[self.geom("floor")] = (float(draw["mu"]), 0.005, float(draw["mu_roll"]))
        self.model.geom_friction[self.geom("lip")] = (float(draw["mu"]), 0.005, LIP_ROLL)
        self._refresh()

    def run(self, action: dict[str, Any], draw: dict[str, float]) -> dict[str, Any]:
        """action: ``tilt`` (deg, toward the lip), ``ramp`` (s), ``hold`` (s)."""
        self.prepare(draw)
        model, data = self.model, self.data
        item = self.body("item")
        theta = math.radians(float(action["tilt"]))
        n_pre, n_ramp, n_hold = int(round(0.1 / TIMESTEP)), int(round(float(action["ramp"]) / TIMESTEP)), int(round(float(action["hold"]) / TIMESTEP))
        mujoco.mj_forward(model, data)
        x0 = float(data.xpos[item][0])
        still = 0
        outcome = None
        steps = n_pre + n_ramp + n_hold
        for i in range(steps):
            ang = 0.0 if i < n_pre else theta * min(1.0, (i - n_pre + 1) / max(n_ramp, 1))
            model.opt.gravity[:] = (G * math.sin(ang), 0.0, -G * math.cos(ang))
            mujoco.mj_step(model, data)
            if data.xpos[item][0] > 0.0:
                outcome = "over"
                break
            if i > n_pre + n_ramp and i % 50 == 0:
                still = still + 1 if _rest(data.qvel[0:6], lin=0.003, ang=0.08) else 0
                if still >= 6:
                    break
        model.opt.gravity[:] = (0.0, 0.0, -G)
        moved = float(data.xpos[item][0]) - x0
        if outcome is None:
            outcome = "caught" if abs(moved) >= 0.01 else "put"
        return {"outcome": outcome, "unstable": self.unstable(), "moved": moved, "t": data.time}


# --------------------------------------------------------------------------
# swing
# --------------------------------------------------------------------------

CARRIAGE_MASS = 400.0
SWING_DAMPING = 0.002


class SwingScene(_Scene):
    """visible: ``dims`` (load: depth along the motion, width, height), ``mass``, ``strap`` (nominal length hook → load top), ``gap``
    (load front face at rest under the stopped hook → upright face). The upright's position depends on the action (where the hook
    stops), so it is placed per rollout. Draw: ``stretch`` (strap compliance per kg), ``slop`` (m), ``com`` (centre-of-mass depth
    below the load's top, fraction of its height)."""

    family = "swing"
    TOP = 2.5  # hook height (m)

    def xml(self) -> str:
        b, w, h = (float(x) for x in self.visible["dims"])
        m = float(self.visible["mass"])
        ixx, iyy, izz = box_inertia((b, w, h), m)
        return (
            _header("t9-swing")
            + "  <worldbody>\n"
            f'    <body name="carriage" pos="0 0 {self.TOP}">\n'
            '      <joint name="run" type="slide" axis="1 0 0"/>\n'
            f'      <inertial pos="0 0 0" mass="{CARRIAGE_MASS}" diaginertia="10 10 10"/>\n'
            '      <body name="load" pos="0 0 0">\n'
            f'        <joint name="hinge" type="hinge" axis="0 1 0" damping="{SWING_DAMPING}"/>\n'
            f'        <inertial pos="{NOMINAL_IPOS} 0 -1" mass="{m:.6f}" diaginertia="{ixx:.8f} {iyy:.8f} {izz:.8f}"/>\n'
            f'        <geom name="load" type="box" pos="0 0 -1" size="{b / 2:.6f} {w / 2:.6f} {h / 2:.6f}"/>\n'
            "      </body>\n"
            "    </body>\n"
            '    <body name="upright" pos="5 0 1" mocap="true">\n'
            '      <geom name="upright" type="box" size="0.03 0.6 1.2"/>\n'
            "    </body>\n"
            "  </worldbody>\n</mujoco>\n"
        )

    def length(self, draw: dict[str, float]) -> float:
        return float(self.visible["strap"]) * (1 + float(draw["stretch"]) * float(self.visible["mass"])) + float(draw["slop"])

    def prepare(self, draw: dict[str, float]) -> None:
        b, w, h = (float(x) for x in self.visible["dims"])
        length = self.length(draw)
        load = self.body("load")
        self.model.geom_pos[self.geom("load")] = (0.0, 0.0, -(length + h / 2))
        self.model.body_ipos[load] = (NOMINAL_IPOS, 0.0, -(length + float(draw["com"]) * h))
        self._refresh()

    def run(self, action: dict[str, Any], draw: dict[str, float]) -> dict[str, Any]:
        """action: ``speed`` (m/s), ``decel`` (m/s²), ``ramp`` (s, jerk ramp of the stop)."""
        self.prepare(draw)
        model, data = self.model, self.data
        b, w, h = (float(x) for x in self.visible["dims"])
        speed = float(action["speed"])
        acc = np.concatenate([np.zeros(int(round(0.2 / TIMESTEP))), speed_change(-speed, float(action["decel"]), float(action["ramp"]))])
        vel = speed + np.cumsum(acc) * TIMESTEP
        vel[-1] = 0.0
        pos = np.cumsum(vel) * TIMESTEP  # carriage x after each step (start at 0)
        stop_x = float(pos[-1])
        upright_face = stop_x + b / 2 + float(self.visible["gap"])
        model.body_pos[self.body("upright")] = (upright_face + 0.03, 0.0, 1.0)
        data.mocap_pos[0] = (upright_face + 0.03, 0.0, 1.0)
        data.qvel[0] = speed
        mujoco.mj_forward(model, data)
        load_geom, upright_geom = self.geom("load"), self.geom("upright")
        load_body = self.body("load")
        reach = -1.0
        outcome, contact_speed = "clear", 0.0
        n_drive = acc.size
        peak_seen = False
        for i in range(n_drive + int(round(6.0 / TIMESTEP))):
            target_a = acc[i] if i < n_drive else 0.0
            target_v = vel[i] if i < n_drive else 0.0
            target_x = pos[i] if i < n_drive else stop_x
            data.qfrc_applied[0] = CARRIAGE_MASS * (target_a + KP * (target_x - data.qpos[0]) + KD * (target_v - data.qvel[0]))
            mujoco.mj_step(model, data)
            hit = self.touching(load_geom, upright_geom)
            if hit is not None:
                point = np.array(data.contact[hit].pos)
                hinge = np.array(data.xpos[load_body])
                omega = float(data.qvel[1])
                v_x = float(data.qvel[0]) + omega * float(point[2] - hinge[2])
                contact_speed = max(0.0, v_x)
                outcome = "strike" if contact_speed >= STRIKE_SPEED else "brush"
                break
            if i % 5 == 0:
                front = self._front(data, b, h)
                reach = max(reach, front - (stop_x + b / 2))
                if i >= n_drive and data.qvel[1] > 0:  # swinging back after the forward peak (forward swing has θ̇ < 0)
                    peak_seen = True
                if peak_seen and i >= n_drive + 200:
                    break
        return {"outcome": outcome, "unstable": self.unstable(), "contact_speed": contact_speed, "reach": reach, "length": self.length(draw), "t": data.time}

    def _front(self, data: Any, depth: float, height: float) -> float:
        """Largest x of the load box (its front edges) in the current pose."""
        g = self.geom("load")
        centre = np.array(data.geom_xpos[g])
        rot = np.array(data.geom_xmat[g]).reshape(3, 3)
        return float(max((centre + rot @ np.array([sx * depth / 2, 0.0, sz * height / 2]))[0] for sx in (-1, 1) for sz in (-1, 1)))


# --------------------------------------------------------------------------
# knock
# --------------------------------------------------------------------------

ARM_MASS = 60.0
ARM_RADIUS = 0.02
BACK_OFF = 0.10


class KnockScene(_Scene):
    """visible: ``dims`` (object: depth along the strike, width, height), ``mass``. Draw: ``mu`` (base friction), ``hc`` (centre of
    mass height, fraction of the height), ``cx`` (centre-of-mass offset along the strike, fraction of the depth; + = away from the
    arm)."""

    family = "knock"

    def xml(self) -> str:
        b, w, h = (float(x) for x in self.visible["dims"])
        m = float(self.visible["mass"])
        ixx, iyy, izz = box_inertia((b, w, h), m)
        return (
            _header("t9-knock")
            + "  <worldbody>\n"
            '    <geom name="surface" type="plane" size="4 4 0.1" priority="1"/>\n'
            f'    <body name="arm" pos="{-b / 2 - ARM_RADIUS - 0.03:.6f} 0 0.5">\n'
            '      <joint name="arm_x" type="slide" axis="1 0 0"/>\n'
            f'      <inertial pos="0 0 0" mass="{ARM_MASS}" diaginertia="1 1 1"/>\n'
            f'      <geom name="link" type="capsule" fromto="0 -0.2 0 0 0.2 0" size="{ARM_RADIUS}" friction="0.4 0.005 0.0001"/>\n'
            "    </body>\n"
            f'    <body name="obj" pos="0 0 {h / 2 + 0.0005:.6f}">\n'
            "      <freejoint/>\n"
            f'      <inertial pos="{NOMINAL_IPOS} {NOMINAL_IPOS} {NOMINAL_IPOS}" mass="{m:.6f}" diaginertia="{ixx:.8f} {iyy:.8f} {izz:.8f}"/>\n'
            f'      <geom name="obj" type="box" size="{b / 2:.6f} {w / 2:.6f} {h / 2:.6f}"/>\n'
            "    </body>\n"
            "  </worldbody>\n</mujoco>\n"
        )

    def prepare(self, draw: dict[str, float], height: float) -> None:
        b, w, h = (float(x) for x in self.visible["dims"])
        obj = self.body("obj")
        self.model.body_ipos[obj] = (float(draw["cx"]) * b, 0.0, (float(draw["hc"]) - 0.5) * h)
        self.model.geom_friction[self.geom("surface")] = (float(draw["mu"]), 0.005, 0.0001)
        self.model.body_pos[self.body("arm")][2] = float(height)
        self._refresh()

    def run(self, action: dict[str, Any], draw: dict[str, float]) -> dict[str, Any]:
        """action: ``speed`` (m/s), ``height`` (strike height above the surface, m), ``stop`` (m of travel after the first touch)."""
        self.prepare(draw, float(action["height"]))
        model, data = self.model, self.data
        obj, arm_geom, obj_geom = self.body("obj"), self.geom("link"), self.geom("obj")
        speed, stop = float(action["speed"]), float(action["stop"])
        data.qvel[0] = speed
        mujoco.mj_forward(model, data)
        x0 = np.array(data.xipos[obj][:2])
        plan: Callable[[float], tuple[float, float, float]] | None = None
        t_touch = None
        steps = int(round(4.0 / TIMESTEP))
        start_pos = float(data.qpos[0])
        for i in range(steps):
            t = i * TIMESTEP
            if plan is None:
                target = (0.0, speed, start_pos + speed * (t + TIMESTEP))
            else:
                target = plan(t)
            a_ff, v_ref, x_ref = target
            data.qfrc_applied[0] = ARM_MASS * (a_ff + KP * (x_ref - data.qpos[0]) + KD * (v_ref - data.qvel[0]))
            mujoco.mj_step(model, data)
            if plan is None and self.touching(arm_geom, obj_geom) is not None:
                t_touch, x_touch = data.time, float(data.qpos[0])
                plan = _stop_then_back(t_touch, x_touch, speed, stop)
            if plan is None and t > 1.5:
                break  # never reached (should not happen)
            if t_touch is not None and t > t_touch + 1.2 and i % 50 == 0 and _rest(self._obj_qvel()):
                break
        tilt = _tilt_deg(data, obj)
        moved = float(np.linalg.norm(np.array(data.xipos[obj][:2]) - x0))
        outcome = "topple" if tilt > TOPPLE_DEG else ("slide" if moved >= 0.01 else "stay")
        return {"outcome": outcome, "unstable": self.unstable(), "tilt": tilt, "moved": moved, "touched": t_touch is not None, "t": data.time}

    def _obj_qvel(self) -> np.ndarray:
        adr = int(self.model.jnt_dofadr[self.model.body_jntadr[self.body("obj")]])
        return np.array(self.data.qvel[adr : adr + 6])


def _stop_then_back(t0: float, x0: float, speed: float, stop: float) -> Callable[[float], tuple[float, float, float]]:
    """Arm reference after the first touch: constant deceleration to rest over `stop` m, 0.2 s still, back off :data:`BACK_OFF` m at
    0.2 m/s (raised-cosine speed), then still. Returns t → (feed-forward acceleration, reference speed, reference position)."""
    decel = speed * speed / (2 * max(stop, 1e-4))
    t_stop = speed / decel
    t_wait = t_stop + 0.2
    back_v, ramp = 0.2, 0.1
    t_back = BACK_OFF / back_v + ramp

    def ref(t: float) -> tuple[float, float, float]:
        s = t - t0
        if s < t_stop:
            return (-decel, speed - decel * s, x0 + speed * s - 0.5 * decel * s * s)
        x_stop = x0 + stop
        if s < t_wait:
            return (0.0, 0.0, x_stop)
        u = s - t_wait
        if u < t_back:
            # raised-cosine ramps of `ramp` s at both ends
            if u < ramp:
                f = 0.5 * (1 - math.cos(math.pi * u / ramp))
                vel = -back_v * f
                pos = x_stop - back_v * (u / 2 - ramp / (2 * math.pi) * math.sin(math.pi * u / ramp))
                acc = -back_v * 0.5 * math.pi / ramp * math.sin(math.pi * u / ramp)
                return (acc, vel, pos)
            if u < t_back - ramp:
                return (0.0, -back_v, x_stop - back_v * (ramp / 2) - back_v * (u - ramp))
            w = t_back - u
            f = 0.5 * (1 - math.cos(math.pi * w / ramp))
            pos = x_stop - BACK_OFF + back_v * (w / 2 - ramp / (2 * math.pi) * math.sin(math.pi * w / ramp))
            return (back_v * 0.5 * math.pi / ramp * math.sin(math.pi * w / ramp), -back_v * f, pos)
        return (0.0, 0.0, x_stop - BACK_OFF)

    return ref


# --------------------------------------------------------------------------
# conveyor
# --------------------------------------------------------------------------


class ConveyorScene(_Scene):
    """visible: ``dims`` (item: depth along the belt, width, height), ``mass``. Draw: ``mu``, ``hc`` (centre-of-mass height
    fraction), ``cx`` (offset along the belt, fraction of the depth; + = toward the side the item is thrown — the front for a stop,
    the back for a start)."""

    family = "conveyor"

    def xml(self) -> str:
        b, w, h = (float(x) for x in self.visible["dims"])
        m = float(self.visible["mass"])
        ixx, iyy, izz = box_inertia((b, w, h), m)
        return (
            _header("t9-conveyor")
            + "  <worldbody>\n"
            '    <geom name="belt" type="plane" size="6 6 0.1" priority="1"/>\n'
            f'    <body name="item" pos="0 0 {h / 2 + 0.0005:.6f}">\n'
            "      <freejoint/>\n"
            f'      <inertial pos="{NOMINAL_IPOS} {NOMINAL_IPOS} {NOMINAL_IPOS}" mass="{m:.6f}" diaginertia="{ixx:.8f} {iyy:.8f} {izz:.8f}"/>\n'
            f'      <geom name="item" type="box" size="{b / 2:.6f} {w / 2:.6f} {h / 2:.6f}"/>\n'
            "    </body>\n"
            "  </worldbody>\n</mujoco>\n"
        )

    def prepare(self, draw: dict[str, float]) -> None:
        b, w, h = (float(x) for x in self.visible["dims"])
        item = self.body("item")
        self.model.body_ipos[item] = (float(draw["cx"]) * b, 0.0, (float(draw["hc"]) - 0.5) * h)
        self.model.geom_friction[self.geom("belt")] = (float(draw["mu"]), 0.005, 0.0001)
        self._refresh()

    def run(self, action: dict[str, Any], draw: dict[str, float]) -> dict[str, Any]:
        """action: ``speed`` (belt speed, m/s), ``rate`` (m/s² of the stop or start), ``ramp`` (s). The pulse throws the item toward +x
        (for a start the scene is mirrored: the item's back is its +x side — ``cx`` is read toward the thrown side)."""
        self.prepare(draw)
        model, data = self.model, self.data
        item = self.body("item")
        pulse = speed_change(float(action["speed"]), float(action["rate"]), float(action["ramp"]))
        n_pre = int(round(0.1 / TIMESTEP))
        mujoco.mj_forward(model, data)
        x0 = np.array(data.xipos[item][:2])
        adr = int(model.jnt_dofadr[model.body_jntadr[item]])
        total = n_pre + pulse.size + int(round(4.0 / TIMESTEP))
        for i in range(total):
            j = i - n_pre
            a = pulse[j] if 0 <= j < pulse.size else 0.0
            model.opt.gravity[:] = (a, 0.0, -G)
            mujoco.mj_step(model, data)
            if j > pulse.size + 100 and i % 50 == 0 and _rest(data.qvel[adr : adr + 6]):
                break
        model.opt.gravity[:] = (0.0, 0.0, -G)
        tilt = _tilt_deg(data, item)
        moved = float(np.linalg.norm(np.array(data.xipos[item][:2]) - x0))
        outcome = "topple" if tilt > TOPPLE_DEG else ("slide" if moved >= 0.02 else "stay")
        return {"outcome": outcome, "unstable": self.unstable(), "tilt": tilt, "moved": moved, "t": data.time}


# --------------------------------------------------------------------------
# overhang
# --------------------------------------------------------------------------

SHELF_TOP = 0.9
#: overhang: the item is let go this far above the board (m).
RELEASE_DROP = 0.005


class OverhangScene(_Scene):
    """visible: ``board`` (length, width, thickness), ``item_dims`` (depth along the board, width, height). The board's mass, the
    item's mass and centre of mass and the placement error are hidden. Draw: ``board_mass`` (kg), ``item_mass`` (kg), ``icx``
    (item centre-of-mass offset along the board, fraction of its depth; + = outward), ``err`` (placement error, m; + = outward)."""

    family = "overhang"

    def xml(self) -> str:
        bl, bw, bt = (float(x) for x in self.visible["board"])
        d, w, h = (float(x) for x in self.visible["item_dims"])
        ib = box_inertia((bl, bw, bt), 1.0)
        ii = box_inertia((d, w, h), 1.0)
        return (
            _header("t9-overhang")
            + "  <worldbody>\n"
            '    <geom name="floor" type="plane" size="4 4 0.1"/>\n'
            f'    <geom name="shelf" type="box" pos="-0.4 0 {SHELF_TOP - 0.02:.6f}" size="0.4 0.3 0.02" priority="1"/>\n'
            f'    <body name="board" pos="0 0 {SHELF_TOP + bt / 2 + 0.0002:.6f}">\n'
            "      <freejoint/>\n"
            f'      <inertial pos="{NOMINAL_IPOS} {NOMINAL_IPOS} {NOMINAL_IPOS}" mass="1" diaginertia="{ib[0]:.8f} {ib[1]:.8f} {ib[2]:.8f}"/>\n'
            f'      <geom name="board" type="box" size="{bl / 2:.6f} {bw / 2:.6f} {bt / 2:.6f}" priority="1"/>\n'
            "    </body>\n"
            f'    <body name="item" pos="0 0 {SHELF_TOP + bt + h / 2 + 0.005:.6f}">\n'
            "      <freejoint/>\n"
            f'      <inertial pos="{NOMINAL_IPOS} {NOMINAL_IPOS} {NOMINAL_IPOS}" mass="1" diaginertia="{ii[0]:.8f} {ii[1]:.8f} {ii[2]:.8f}"/>\n'
            f'      <geom name="item" type="box" size="{d / 2:.6f} {w / 2:.6f} {h / 2:.6f}"/>\n'
            "    </body>\n"
            "  </worldbody>\n</mujoco>\n"
        )

    def prepare(self, draw: dict[str, float], over: float, inset: float) -> None:
        bl, bw, bt = (float(x) for x in self.visible["board"])
        d, w, h = (float(x) for x in self.visible["item_dims"])
        board, item = self.body("board"), self.body("item")
        mb, mi = float(draw["board_mass"]), float(draw["item_mass"])
        self.model.body_mass[board] = mb
        self.model.body_inertia[board] = box_inertia((bl, bw, bt), mb)
        self.model.body_mass[item] = mi
        self.model.body_inertia[item] = box_inertia((d, w, h), mi)
        self.model.body_ipos[board] = (NOMINAL_IPOS, 0.0, 0.0)
        self.model.body_ipos[item] = (float(draw["icx"]) * d, 0.0, 0.0)
        # board: outer end at x = over (the shelf's edge is x = 0); item centre `inset` in from the outer end, plus the placement error
        # free bodies start from qpos0 (compiled from the body positions), so the placement is written there
        for body, pos in ((board, (over - bl / 2, 0.0, SHELF_TOP + bt / 2 + 0.0002)),
                          (item, (over - inset + float(draw["err"]), 0.0, SHELF_TOP + bt + h / 2 + float(self.visible.get("drop", RELEASE_DROP))))):
            adr = int(self.model.jnt_qposadr[self.model.body_jntadr[body]])
            self.model.qpos0[adr : adr + 7] = (*pos, 1.0, 0.0, 0.0, 0.0)
        self.model.geom_friction[self.geom("board")] = (0.5, 0.005, 0.0001)
        self._refresh()

    def run(self, action: dict[str, Any], draw: dict[str, float]) -> dict[str, Any]:
        """action: ``over`` (m the board's end sticks out past the shelf edge), ``inset`` (item centre in from the board's outer end)."""
        self.prepare(draw, float(action["over"]), float(action["inset"]))
        model, data = self.model, self.data
        board, item = self.body("board"), self.body("item")
        bl, bw, bt = (float(x) for x in self.visible["board"])
        mujoco.mj_forward(model, data)
        z_board0 = float(data.xipos[board][2])
        a_b = int(model.jnt_dofadr[model.body_jntadr[board]])
        a_i = int(model.jnt_dofadr[model.body_jntadr[item]])
        for i in range(int(round(3.5 / TIMESTEP))):
            mujoco.mj_step(model, data)
            if i > 400 and i % 50 == 0 and _rest(data.qvel[a_b : a_b + 6]) and _rest(data.qvel[a_i : a_i + 6]):
                break
            if i % 50 == 0 and data.xipos[board][2] < z_board0 - 0.2:
                break  # the board is on its way down
        tilt_b = _tilt_deg(data, board)
        dropped = z_board0 - float(data.xipos[board][2])
        if tilt_b > 10.0 or dropped > 0.02:
            outcome = "board_tips"
        elif float(data.xipos[item][2]) < SHELF_TOP + bt - 0.01:
            outcome = "item_falls"
        else:
            outcome = "stays"
        return {"outcome": outcome, "unstable": self.unstable(), "board_tilt": tilt_b, "t": data.time}


# --------------------------------------------------------------------------
# suction
# --------------------------------------------------------------------------

CARRIER_MASS = 80.0
PAD_HALF_T = 0.005
PAD_Z = 1.2
SEAL_TILT_DEG = 3.0
SEAL_GAP = 0.003
SHIFT_M = 0.003
#: suction: longest simulated cruise between the start and the stop (s).
CRUISE_MAX = 0.6


class SuctionScene(_Scene):
    """visible: ``panel`` (length along the motion, width, thickness), ``cup`` (diameter, m), ``offset`` (cup centre ahead of the
    panel's centre along the motion, m). Draw: ``vacuum`` (Pa), ``mu`` (pad/panel friction), ``mass`` (panel mass, kg)."""

    family = "suction"

    def xml(self) -> str:
        pl, pw, pt = (float(x) for x in self.visible["panel"])
        side = math.sqrt(math.pi) * float(self.visible["cup"]) / 2  # square of the cup's area
        ip = box_inertia((pl, pw, pt), 1.0)
        off = float(self.visible["offset"])
        return (
            _header("t9-suction")
            + "  <worldbody>\n"
            '    <geom name="floor" type="plane" size="6 6 0.1"/>\n'
            f'    <body name="carrier" pos="0 0 {PAD_Z}">\n'
            '      <joint name="carry" type="slide" axis="1 0 0"/>\n'
            f'      <inertial pos="0 0 0.05" mass="{CARRIER_MASS}" diaginertia="2 2 2"/>\n'
            f'      <geom name="pad" type="box" size="{side / 2:.6f} {side / 2:.6f} {PAD_HALF_T}" priority="1"/>\n'
            "    </body>\n"
            f'    <body name="panel" pos="{-off:.6f} 0 {PAD_Z - PAD_HALF_T - pt / 2:.6f}">\n'
            "      <freejoint/>\n"
            f'      <inertial pos="{NOMINAL_IPOS} {NOMINAL_IPOS} {NOMINAL_IPOS}" mass="1" diaginertia="{ip[0]:.8f} {ip[1]:.8f} {ip[2]:.8f}"/>\n'
            f'      <geom name="panel" type="box" size="{pl / 2:.6f} {pw / 2:.6f} {pt / 2:.6f}"/>\n'
            "    </body>\n"
            "  </worldbody>\n</mujoco>\n"
        )

    def prepare(self, draw: dict[str, float]) -> None:
        pl, pw, pt = (float(x) for x in self.visible["panel"])
        panel = self.body("panel")
        m = float(draw["mass"])
        self.model.body_mass[panel] = m
        self.model.body_inertia[panel] = box_inertia((pl, pw, pt), m)
        self.model.body_ipos[panel] = (NOMINAL_IPOS, 0.0, 0.0)
        self.model.geom_friction[self.geom("pad")] = (float(draw["mu"]), 0.005, 0.0001)
        self._refresh()

    def run(self, action: dict[str, Any], draw: dict[str, float]) -> dict[str, Any]:
        """action: ``rate`` (m/s² of the start and of the stop), ``speed`` (m/s), ``ramp`` (s), ``carry`` (m, the whole move)."""
        self.prepare(draw)
        model, data = self.model, self.data
        panel, carrier = self.body("panel"), self.body("carrier")
        pl, pw, pt = (float(x) for x in self.visible["panel"])
        radius = float(self.visible["cup"]) / 2
        force = float(draw["vacuum"]) * math.pi * radius * radius
        speed, rate, ramp = float(action["speed"]), float(action["rate"]), float(action["ramp"])
        up = speed_change(speed, rate, ramp)
        # cruise at constant speed for the rest of the carry; only its first CRUISE_MAX s are simulated (no inertial load while
        # cruising — the panel has settled on the pad long before)
        cruise = max(0.0, (float(action.get("carry", 0.0)) - 2 * speed * up.size * TIMESTEP / 2) / speed)
        acc = np.concatenate([np.zeros(int(round(0.25 / TIMESTEP))), up, np.zeros(int(round(min(max(cruise, 0.3), CRUISE_MAX) / TIMESTEP))),
                              speed_change(-speed, rate, ramp)])
        vel = np.cumsum(acc) * TIMESTEP
        vel[-1] = 0.0
        pos = np.cumsum(vel) * TIMESTEP
        mujoco.mj_forward(model, data)
        sealed = True
        start_rel = None
        slip = 0.0
        a_p = int(model.jnt_dofadr[model.body_jntadr[panel]])
        n = acc.size + int(round(2.0 / TIMESTEP))
        for i in range(n):
            if sealed:
                # the vacuum pulls the panel up at the cup's centre (the pad centre's footprint on the panel's top face)
                point = np.array(data.xpos[carrier]) + np.array([0.0, 0.0, -PAD_HALF_T])
                com = np.array(data.xipos[panel])
                lever = point - com
                f = np.array([0.0, 0.0, force])
                data.xfrc_applied[panel, :3] = f
                data.xfrc_applied[panel, 3:] = np.cross(lever, f)
            else:
                data.xfrc_applied[panel] = 0.0
            target_a = acc[i] if i < acc.size else 0.0
            target_v = vel[i] if i < acc.size else 0.0
            target_x = pos[i] if i < acc.size else float(pos[-1])
            data.qfrc_applied[0] = CARRIER_MASS * (target_a + KP * (target_x - data.qpos[0]) + KD * (target_v - data.qvel[0]))
            mujoco.mj_step(model, data)
            rel = self._relative(data, panel, carrier)
            if i == int(round(0.2 / TIMESTEP)):
                start_rel = rel
            if start_rel is not None and sealed and i % 5 == 0:
                slip = max(slip, float(np.linalg.norm(rel[:2] - start_rel[:2])))
            if sealed and i % 5 == 0:
                tilt = _tilt_deg(data, panel)
                gap = (float(data.xpos[carrier][2]) - PAD_HALF_T) - (float(data.xpos[panel][2]) + pt / 2)
                inside = min(pl / 2 - abs(rel[0]), pw / 2 - abs(rel[1]))
                if tilt > SEAL_TILT_DEG or gap > SEAL_GAP or inside < radius:
                    sealed = False
            if not sealed and float(data.xpos[panel][2]) < PAD_Z - 0.15:
                break
            if i > acc.size + 200 and i % 50 == 0 and _rest(data.qvel[a_p : a_p + 6]):
                break
        data.xfrc_applied[panel] = 0.0
        outcome = "drop" if not sealed else ("shift" if slip >= SHIFT_M else "hold")
        return {"outcome": outcome, "unstable": self.unstable(), "slip": slip, "t": data.time}

    @staticmethod
    def _relative(data: Any, panel: int, carrier: int) -> np.ndarray:
        """Pad centre relative to the panel's centre, in the panel's frame."""
        rot = np.array(data.xmat[panel]).reshape(3, 3)
        return rot.T @ (np.array(data.xpos[carrier]) - np.array(data.xpos[panel]))


SCENES = {"roll": RollScene, "swing": SwingScene, "knock": KnockScene, "conveyor": ConveyorScene, "overhang": OverhangScene, "suction": SuctionScene}


def make_scene(family: str, visible: dict[str, Any]) -> _Scene:
    return SCENES[family](visible)

