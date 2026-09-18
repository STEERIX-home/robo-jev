"""E0/E1 환경 — reset·step·snapshot·restore (docs/05 §2, docs/08 §3.2).

한 `step`은 정확히 한 제어 주기(20ms = 2ms 물리 10회)다. 그 주기 동안

1. 실행기가 아는 자기 상태를 `Controller.observe`에 넣고,
2. 명령이 있으면 `Controller.apply`로 계약 검사를 거쳐 ACK를 받고,
3. 명령이 없어도 `Controller.advance`로 반사·혼합·감속을 진행시켜 말단 목표 하나를 얻고,
4. 그 목표를 robosuite OSC_POSE에 **절대 자세**로 넘겨 물리를 20ms 돌리고,
5. 새 모의 시각에서 일정(외란·지시 변경)을 적용한 뒤 관측을 만든다.

`snapshot`은 "다시 이어 붙일 수 있는 모든 것"을 담는다: MuJoCo 적분 상태(qpos·qvel·
act·ctrl·mocap·userdata·warm start), robosuite OSC/그리퍼 내부 목표, 우리 컨트롤러의
전체 상태, 일정과 그 진행도, wrapper 상태, 그리고 모든 RNG. 바이트는 gzip한 JSON이라
사람이 열어 볼 수 있고 pickle 호환성 문제가 없다.
"""

from __future__ import annotations

import base64
import gzip
import json
import math
import random
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
import robosuite.macros

from robo_jev.sim.controller import (
    Controller,
    load_controller_config,
    resolve_config_path,
)
from robo_jev.sim.scene import ScenePlan, build_plan, merge_profile
from robo_jev.sim.tidy_clutter import TidyClutter

__all__ = ["Environment"]

_STATE_SPEC = mujoco.mjtState.mjSTATE_INTEGRATION
_WARMSTART_SPEC = mujoco.mjtState.mjSTATE_WARMSTART

#: 이 환경이 만드는 사건 종류 (docs/08 §3.2 `events[]`).
_EVENT_KINDS = (
    "disturbance_applied",
    "contact_onset",
    "instruction_changed",
)

#: 로봇 base가 world에 대해 회전이 없다고 보고 좌표를 평행이동만으로 바꾼다.
#: 설정의 `frame.assert_base_orientation_identity`가 참이면 reset마다 확인한다.
_BASE_ORIENTATION_TOLERANCE = 1e-9

#: 설정의 `reach.min_height_mm`가 실제 테이블 윗면과 이만큼 넘게 어긋나면 실패한다.
_TABLE_HEIGHT_TOLERANCE_MM = 2.0

#: 물리 timestep 비교 허용 오차(초). 설정은 ms 정수이므로 부동소수 표현 오차만 흡수한다.
_TIMESTEP_TOLERANCE_S = 1e-12

#: 모의 시각과 MuJoCo 시간의 허용 차이(ms). 누적 부동소수 오차만 흡수한다.
_CLOCK_TOLERANCE_MS = 1e-3


def _json_constant(name: str) -> float:
    """표준 JSON이 아닌 토큰(`Infinity`·`NaN`)을 만나면 그 자리에서 막는다."""
    raise ValueError(f"snapshot에 표준 JSON이 아닌 값이 들어 있다: {name}")


def _encode_array(array: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(array, dtype=np.float64).tobytes()).decode("ascii")


def _decode_array(text: str) -> np.ndarray:
    return np.frombuffer(base64.b64decode(text.encode("ascii")), dtype=np.float64).copy()


def _maybe_array(value: Any) -> str | None:
    """robosuite의 목표 값은 첫 주기 전까지 `None`이다. 그 상태도 그대로 담는다."""
    return None if value is None else _encode_array(np.asarray(value, dtype=np.float64))


def _maybe_decode(text: str | None, shape: tuple[int, ...] | None = None) -> np.ndarray | None:
    if text is None:
        return None
    array = _decode_array(text)
    return array if shape is None else array.reshape(shape)


def _target_ref(command: dict[str, Any] | None) -> str | None:
    """명령이 가리키는 물체 id. 두 명령 형식 모두에서 같은 자리를 본다."""
    if command is None:
        return None
    path = command.get("path") or {}
    return command.get("target_ref") or path.get("target_ref") or command.get("gripper_ref")


def _top_face_samples(plan_object, inflate_m: float, samples: int) -> list[np.ndarray]:
    """물체 좌표계에서 윗면 표본점을 만든다 — 중심 + 안쪽으로 `inflate`만큼 들인 네 점.

    표본을 면 안쪽으로 들이는 이유는 모서리를 스치는 광선이 물체를 빗맞고 뒤의 지오메트리를
    맞히는 것을 막기 위해서다. 원통은 모서리가 없으므로 반지름 위의 네 점을 쓴다.
    """
    half = [value / 1000.0 for value in plan_object.half_size_mm]
    top = half[2]
    points = [np.array([0.0, 0.0, top])]
    if plan_object.shape == "box":
        x_in = max(half[0] - inflate_m, half[0] * 0.5)
        y_in = max(half[1] - inflate_m, half[1] * 0.5)
        corners = [(x_in, y_in), (x_in, -y_in), (-x_in, y_in), (-x_in, -y_in)]
    else:
        radius = max(half[0] - inflate_m, half[0] * 0.5)
        diagonal = radius / math.sqrt(2.0)
        corners = [
            (diagonal, diagonal),
            (diagonal, -diagonal),
            (-diagonal, diagonal),
            (-diagonal, -diagonal),
        ]
    points.extend(np.array([x, y, top]) for x, y in corners)
    return points[:samples]


def _mat_to_quat_xyzw(mat: np.ndarray) -> list[float]:
    quat_wxyz = np.empty(4)
    mujoco.mju_mat2Quat(quat_wxyz, np.ascontiguousarray(mat, dtype=np.float64).reshape(9))
    return [float(quat_wxyz[1]), float(quat_wxyz[2]), float(quat_wxyz[3]), float(quat_wxyz[0])]


def _quat_xyzw_to_rotvec(quat: list[float]) -> list[float]:
    """xyzw quaternion → 축각(rotvec). robosuite absolute 입력이 이 형식을 받는다."""
    x, y, z, w = quat
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if norm == 0.0:
        return [0.0, 0.0, 0.0]
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    if w < 0.0:  # 항상 짧은 쪽 회전을 고른다
        x, y, z, w = -x, -y, -z, -w
    sin_half = math.sqrt(x * x + y * y + z * z)
    if sin_half < 1e-12:
        return [0.0, 0.0, 0.0]
    angle = 2.0 * math.atan2(sin_half, w)
    scale = angle / sin_half
    return [x * scale, y * scale, z * scale]


class Environment:
    """docs/05 §2의 E0/E1. 물리·일정·컨트롤러를 하나의 재현 가능한 단위로 묶는다."""

    def __init__(self, config_path: str | Path, profile: str | None = None) -> None:
        self.config_path = resolve_config_path(config_path)
        self.config = json.loads(
            json.dumps(_load_yaml(self.config_path))
        )  # 설정은 값이다. 아래에서 프로파일 병합으로만 바뀐다.
        self.profile = profile or str(self.config["default_profile"])
        self.settings = merge_profile(self.config, self.profile)
        self.serializer_version = str(self.config.get("version", "s0"))
        # docs/08 §3.2의 직렬화 규약. 자릿수는 serializer 버전과 함께 고정한다.
        self.quaternion_decimals = int(self.settings["serialization"]["quaternion_decimals"])

        simulator = self.settings["simulator"]
        self.control_hz = int(simulator["control_hz"])
        self.period_ms = 1000 // self.control_hz
        self.physics_dt_ms = int(simulator["physics_dt_ms"])
        if self.period_ms % self.physics_dt_ms:
            raise ValueError(
                f"제어 주기가 물리 timestep의 배수가 아니다: {self.period_ms}ms / "
                f"{self.physics_dt_ms}ms"
            )
        self.substeps = self.period_ms // self.physics_dt_ms
        self.max_ms = int(self.settings["episode"]["max_ms"])

        self.controller_config = load_controller_config(simulator["controller_config"])
        self.controller = Controller(self.controller_config)
        # 두 설정이 같은 주기를 말해야 한다. 컨트롤러는 혼합·감속을 시각으로 계산하고
        # 환경은 그 시각만큼 물리를 돌리므로, 어긋나면 계약의 100ms가 100ms가 아니게 된다.
        if self.controller.period_ms != self.period_ms:
            raise ValueError(
                f"제어 주기가 설정 둘에서 다르다: sim {self.period_ms}ms, controller "
                f"{self.controller.period_ms}ms"
            )
        if self.controller.physics_dt_ms != self.physics_dt_ms:
            raise ValueError(
                f"물리 timestep이 설정 둘에서 다르다: sim {self.physics_dt_ms}ms, controller "
                f"{self.controller.physics_dt_ms}ms"
            )

        self.plan: ScenePlan | None = None
        self._env: TidyClutter | None = None
        self._model_signature: tuple | None = None
        self.seed: int | None = None
        # 에피소드 RNG. 3a는 여기서 뽑지 않지만(장면·일정은 reset seed가 통째로 정한다)
        # snapshot에 담아 둔다. 뒤 slice가 에피소드 안에서 난수를 써야 할 때 **이 두 개**를
        # 쓰라는 자리다 — `numpy.random`·`random`의 전역 상태는 이 과정 밖에서도 바뀌므로
        # 재현을 보장할 수 없고, 그래서 reset이 전역을 건드리지도 않는다.
        self.rng = np.random.default_rng(0)
        self.py_rng = random.Random(0)

    # ------------------------------------------------------------------
    # reset
    # ------------------------------------------------------------------

    def reset(self, seed: int) -> dict[str, Any]:
        """seed 하나에서 장면·일정·물리·컨트롤러·RNG를 전부 다시 세운다."""
        self.seed = int(seed)
        self.plan = build_plan(self.config, self.seed, self.profile)

        # 장면 난수와 별개로, 에피소드 안에서 쓰는 난수는 파생 seed로 둔다. 파생이므로
        # reset seed 하나가 여전히 에피소드 전체를 정한다.
        self.rng = np.random.default_rng([self.seed, 0xB0B0])
        self.py_rng = random.Random(self.seed)

        self._build_sim()
        self._reset_episode_state()
        self._apply_schedules()
        return self._observation(ack=None)

    def _build_sim(self) -> None:
        assert self.plan is not None
        signature = self.plan.model_signature()
        simulator = self.settings["simulator"]
        # robosuite는 물리 timestep을 **모델 XML을 쓸 때** `macros.SIMULATION_TIMESTEP`에서
        # 읽고(`models/world.py`), 하위 스텝 수도 거기서 온 `env.model_timestep`으로 센다
        # (`environments/base.py`). 그래서 `sim.model.opt.timestep`에 나중에 값을 넣어 봐야
        # hard reset이 XML을 다시 쓰면서 지워진다. 모델을 짓기 **전에** macro를 맞춰야 한다.
        robosuite.macros.SIMULATION_TIMESTEP = self.physics_dt_ms / 1000.0
        if self._env is not None and signature == self._model_signature:
            # 모델이 같으면 다시 짓지 않는다. 자세·일정은 새 plan이 정한다.
            self._env.plan = self.plan
        else:
            self.close()
            self._env = TidyClutter(
                plan=self.plan,
                settings=self.settings,
                robots=str(simulator["robot"]),
                gripper_types=str(simulator["gripper"]),
                controller_configs=self._composite_controller_config(),
                has_renderer=bool(simulator["has_renderer"]),
                has_offscreen_renderer=bool(simulator["has_offscreen_renderer"]),
                use_camera_obs=False,
                control_freq=self.control_hz,
                horizon=10**9,
                ignore_done=True,
                renderer="mujoco",
                seed=self.seed,
            )
            self._model_signature = signature
        # robosuite는 생성자에서 `_reset_internal`을 부르지 않는다. 물체 자세는 거기서
        # 놓이므로 새로 지었든 재사용하든 여기서 한 번 reset한다.
        self._env.reset()
        self._check_timing()
        self._check_frame_assumptions()

    def _check_timing(self) -> None:
        """설정의 주기가 실제 모델에 걸렸는지 reset마다 확인한다 (docs/05 §2).

        `physics_dt_ms`는 조용히 무시되기 쉬운 값이다 — macro를 거쳐 XML로 가므로
        어느 한 군데만 어긋나도 물리는 기본값(2ms)으로 돌면서 설정만 4ms라고 말한다.
        그래서 모델·robosuite·우리 설정 셋을 매번 맞대 본다.
        """
        expected_s = self.physics_dt_ms / 1000.0
        actual_s = float(self._env.sim.model.opt.timestep)
        if abs(actual_s - expected_s) > _TIMESTEP_TOLERANCE_S:
            raise RuntimeError(
                f"모델의 물리 timestep이 설정과 다르다: {actual_s}s vs {expected_s}s"
            )
        if abs(float(self._env.model_timestep) - expected_s) > _TIMESTEP_TOLERANCE_S:
            raise RuntimeError(
                "robosuite가 세는 하위 스텝의 기준이 설정과 다르다: "
                f"{self._env.model_timestep}s vs {expected_s}s"
            )
        actual_substeps = round(self._env.control_timestep / self._env.model_timestep)
        if actual_substeps != self.substeps:
            raise RuntimeError(
                f"제어 주기당 물리 스텝 수가 다르다: {actual_substeps} vs {self.substeps}"
            )

    def _check_sim_clock(self) -> None:
        """모의 시각이 실제 물리 시간과 같이 흐르는지 본다. 일정이 여기에 걸리기 때문이다."""
        drift_ms = abs(float(self._env.sim.data.time) * 1000.0 - self.sim_time_ms)
        if drift_ms > _CLOCK_TOLERANCE_MS:
            raise RuntimeError(
                f"모의 시각이 물리 시간과 어긋났다: {self._env.sim.data.time * 1000.0}ms "
                f"vs {self.sim_time_ms}ms"
            )

    def _composite_controller_config(self) -> dict[str, Any]:
        """`configs/controller/osc_v0.yaml`의 `osc` 블록을 robosuite 형식으로 옮긴다."""
        osc = dict(self.controller_config["osc"])
        part = {
            "type": str(osc["part_controller"]),
            "input_type": str(osc["input_type"]),
            "input_ref_frame": str(osc["input_ref_frame"]),
            "input_max": osc["input_max"],
            "input_min": osc["input_min"],
            "output_max": list(osc["output_max"]),
            "output_min": list(osc["output_min"]),
            "kp": float(osc["kp"]),
            "damping_ratio": float(osc["damping_ratio"]),
            "impedance_mode": str(osc["impedance_mode"]),
            "uncouple_pos_ori": bool(osc["uncouple_pos_ori"]),
            "interpolation": None,
            "gripper": {"type": "GRIP"},
        }
        return {"type": "BASIC", "body_parts": {"right": part}}

    def _check_frame_assumptions(self) -> None:
        """좌표계 가정을 reset마다 확인한다 (docs/05 §2 "변환 검사")."""
        frame = self.controller_config["frame"]
        if frame.get("assert_base_orientation_identity"):
            error = np.max(np.abs(np.array(self._osc.origin_ori) - np.eye(3)))
            if error > _BASE_ORIENTATION_TOLERANCE:
                raise RuntimeError(f"로봇 base가 world에 대해 회전해 있다 (오차 {error})")
        table_top_mm = (self._env.table_offset[2] - self._base_pos[2]) * 1000.0
        configured = float(self.controller_config["reach"]["min_height_mm"])
        if abs(table_top_mm - configured) > _TABLE_HEIGHT_TOLERANCE_MM:
            raise RuntimeError(
                "컨트롤러 설정의 reach.min_height_mm이 실제 테이블 윗면과 다르다: "
                f"{configured}mm vs {table_top_mm:.1f}mm"
            )

    def _reset_episode_state(self) -> None:
        assert self.plan is not None
        self.sim_time_ms = 0
        self.tick = 0
        self.instruction_version = 1
        self._next_disturbance = 0
        self._next_instruction = 1  # v1은 t=0에 이미 걸려 있다
        self.disturbance_log: list[dict[str, Any]] = []
        self.event_log: list[dict[str, Any]] = []
        self._step_events: list[dict[str, Any]] = []
        self._contacting: dict[str, bool] = {obj.id: False for obj in self.plan.objects}
        self._last_seen_ms: dict[str, int] = {obj.id: 0 for obj in self.plan.objects}
        self._holding: str | None = None
        self._grasp_pose_mm: dict[str, list[float]] = {}

        pose = self._ee_pose_mm()
        self.controller.reset(
            ee_pos_mm=pose[0],
            ee_quat=pose[1],
            gripper_mm=self._gripper_mm(),
            now_ms=0,
        )

    # ------------------------------------------------------------------
    # step
    # ------------------------------------------------------------------

    def step(self, command: dict[str, Any] | None = None) -> dict[str, Any]:
        """제어 주기 하나를 진행한다. 명령이 없어도 세계와 반사는 계속 돈다."""
        if self._env is None:
            raise RuntimeError("reset(seed)를 먼저 불러야 한다")
        self._step_events = []

        self.controller.observe(self._sensors(_target_ref(command)))
        ack = self.controller.apply(command, self.sim_time_ms) if command is not None else None
        setpoint = self.controller.advance(self.sim_time_ms)
        self._drain_controller_events()

        self._set_impedance(setpoint["impedance_kp"])
        self._env.step(self._action_from(setpoint))

        self.sim_time_ms += self.period_ms
        self.tick += 1
        self._check_sim_clock()
        self._apply_schedules()
        self._track_contacts()
        self._track_holding()
        return self._observation(ack=ack)

    def _action_from(self, setpoint: dict[str, Any]) -> np.ndarray:
        """말단 목표를 robosuite OSC_POSE의 절대 입력(로봇 기준 좌표계)으로 옮긴다."""
        action = np.zeros(self._env.action_dim)
        action[0:3] = [value / 1000.0 for value in setpoint["ee_pos_mm"]]
        action[3:6] = _quat_xyzw_to_rotvec(setpoint["ee_quat"])
        # PandaGripper: +1이 닫기, -1이 열기.
        action[6] = 1.0 if setpoint["gripper"] == "closed" else -1.0
        return action

    def _set_impedance(self, kp: float) -> None:
        """force_level → 임피던스 (docs/08 §6 "힘 수준")."""
        osc = self._osc
        damping = float(self.controller_config["osc"]["damping_ratio"])
        osc.kp = osc.nums2array(kp, 6)
        osc.kd = 2 * np.sqrt(osc.kp) * damping

    # ------------------------------------------------------------------
    # 일정 — 모의 시간으로만 걸린다 (docs/05 §2)
    # ------------------------------------------------------------------

    def _apply_schedules(self) -> None:
        assert self.plan is not None
        while self._next_disturbance < len(self.plan.disturbances):
            item = self.plan.disturbances[self._next_disturbance]
            if item.sim_ms > self.sim_time_ms:
                break
            self._apply_disturbance(item)
            self._next_disturbance += 1

        while self._next_instruction < len(self.plan.instructions):
            step = self.plan.instructions[self._next_instruction]
            if step.sim_ms > self.sim_time_ms:
                break
            self.instruction_version = step.version
            self._emit("instruction_changed", version=step.version, text=step.text)
            self._next_instruction += 1

    def _apply_disturbance(self, item) -> None:
        joint = self._env.object_joints[item.object]
        qpos = np.array(self._env.sim.data.get_joint_qpos(joint))
        qpos[0] += item.delta_mm[0] / 1000.0
        qpos[1] += item.delta_mm[1] / 1000.0
        half = math.radians(item.delta_yaw_deg) / 2.0
        delta_quat = np.array([math.cos(half), 0.0, 0.0, math.sin(half)])
        rotated = np.empty(4)
        mujoco.mju_mulQuat(rotated, delta_quat, qpos[3:7])
        qpos[3:7] = rotated
        self._env.sim.data.set_joint_qpos(joint, qpos)
        self._env.sim.data.set_joint_qvel(joint, np.zeros(6))
        self._env.sim.forward()

        self.disturbance_log.append(
            {
                # `sim_ms`는 **예정** 시각이다. seed가 정한 값이므로 제어 주기를 바꿔도 같다.
                "sim_ms": int(item.sim_ms),
                # `applied_ms`는 실제로 적용된 주기 경계다. 제어 주기가 일정 격자보다 굵으면
                # 예정 시각 **직후의** 경계가 되므로 둘이 달라질 수 있다.
                "applied_ms": int(self.sim_time_ms),
                "object": item.object,
                "delta_mm": [int(item.delta_mm[0]), int(item.delta_mm[1])],
                "delta_yaw_deg": round(float(item.delta_yaw_deg), 2),
            }
        )
        self._emit("disturbance_applied", object=item.object, scheduled_ms=int(item.sim_ms))

    def _emit(self, kind: str, **fields: Any) -> None:
        if kind not in _EVENT_KINDS:
            raise ValueError(f"환경이 만들 수 없는 사건이다: {kind!r}")
        event = {"kind": kind, "sim_ms": int(self.sim_time_ms), **fields}
        self.event_log.append(event)
        self._step_events.append(event)

    def _drain_controller_events(self) -> None:
        """실행기 사건(반사·그리퍼·거절)도 같은 사건 목록에 담는다 (docs/08 §3.2)."""
        for event in self.controller.drain_events():
            entry = {"kind": event["kind"], "sim_ms": int(self.sim_time_ms)}
            entry.update({key: value for key, value in event.items() if key != "kind"})
            entry.pop("at_ms", None)
            self.event_log.append(entry)
            self._step_events.append(entry)

    # ------------------------------------------------------------------
    # 관측
    # ------------------------------------------------------------------

    @property
    def _osc(self):
        return self._env.robots[0].composite_controller.part_controllers["right"]

    @property
    def _gripper_controller(self):
        return self._env.robots[0].composite_controller.part_controllers["right_gripper"]

    @property
    def _gripper_model(self):
        return self._env.robots[0].gripper["right"]

    @property
    def _base_pos(self) -> np.ndarray:
        return np.array(self._osc.origin_pos)

    def _to_base_mm(self, world: np.ndarray) -> list[float]:
        return [float(value) * 1000.0 for value in (np.asarray(world) - self._base_pos)]

    def _ee_pose_mm(self) -> tuple[list[float], list[float]]:
        position = self._to_base_mm(self._osc.ref_pos)
        return position, _mat_to_quat_xyzw(np.array(self._osc.ref_ori_mat))

    def _gripper_mm(self) -> float:
        indexes = self._env.robots[0]._ref_gripper_joint_pos_indexes["right"]
        qpos = self._env.sim.data.qpos[indexes]
        return float(np.sum(np.abs(qpos)) * 1000.0)

    def _contact_force_n(self) -> float:
        return float(np.linalg.norm(self._env.robots[0].ee_force["right"]))

    def _ee_speed_mm_s(self) -> float:
        return float(np.linalg.norm(self._osc.ref_pos_vel) * 1000.0)

    def _object_pose(self, object_id: str) -> tuple[list[float], list[float]]:
        body = self._env.object_body_ids[object_id]
        position = self._to_base_mm(self._env.sim.data.body_xpos[body])
        quat_wxyz = self._env.sim.data.body_xquat[body]
        quat = [float(quat_wxyz[1]), float(quat_wxyz[2]), float(quat_wxyz[3]), float(quat_wxyz[0])]
        return position, quat

    def _round_quat(self, quat: list[float]) -> list[float]:
        """docs/08 §3.2의 직렬화 규약. 자릿수는 설정이 정한다."""
        return [round(float(value), self.quaternion_decimals) for value in quat]

    def _visibility(self) -> dict[str, float]:
        """고정 시점에서 물체 윗면 표본으로 광선을 쏴 가시 비율을 잰다.

        참값을 읽지 않는다 — 실제로 막히는지 MuJoCo에 물어본다. 앞단(3D 재구성)이
        채울 수 있는 값만 만든다는 docs/08 §3.2의 조건을 지키는 근사다.
        """
        spec = self.settings["visibility"]
        offset = [float(value) / 1000.0 for value in spec["viewpoint_mm"]]
        viewpoint = np.array(
            [
                self._env.table_offset[0] + offset[0],
                self._env.table_offset[1] + offset[1],
                self._env.table_offset[2] + offset[2],
            ]
        )
        inflate = float(spec["inflate_mm"]) / 1000.0
        samples = int(spec["samples"])
        model, data = self._env.sim.model._model, self._env.sim.data._data
        geom_id = np.zeros(1, dtype=np.int32)

        ratios: dict[str, float] = {}
        for plan_object in self.plan.objects:
            body = self._env.object_body_ids[plan_object.id]
            centre = np.array(self._env.sim.data.body_xpos[body])
            rotation = np.array(self._env.sim.data.body_xmat[body]).reshape(3, 3)
            hits = 0
            offsets = _top_face_samples(plan_object, inflate, samples)
            for offset in offsets:
                target = centre + rotation @ offset
                direction = target - viewpoint
                distance = float(np.linalg.norm(direction))
                if distance == 0.0:
                    continue
                mujoco.mj_ray(
                    model, data, viewpoint, direction / distance, None, 1, -1, geom_id
                )
                if geom_id[0] >= 0 and int(model.geom_bodyid[geom_id[0]]) == body:
                    hits += 1
            ratios[plan_object.id] = hits / max(1, len(offsets))
        return ratios

    def _sensors(self, target_ref: str | None = None) -> dict[str, Any]:
        """실행기가 아는 자기 상태. 컨트롤러의 반사·readiness가 이것만 본다.

        `target_ref`는 이번 명령이 가리키는 물체다. 그리퍼 close readiness가 "대상까지의
        거리"를 보므로(docs/08 §4) 명령에서 대상을 받아 그 거리만 채운다.
        """
        position, quat = self._ee_pose_mm()
        clearances: list[float] = []
        target_distance_mm = None
        for plan_object in self.plan.objects:
            object_position, _ = self._object_pose(plan_object.id)
            distance = math.dist(position, object_position)
            if plan_object.id == target_ref:
                target_distance_mm = distance
            if plan_object.id == self._holding:
                continue  # 들고 있는 물체는 근접 반사의 장애물이 아니다
            # 중심 거리에서 물체의 외접 반지름을 빼 표면까지의 여유로 본다.
            radius = math.dist((0.0, 0.0, 0.0), plan_object.half_size_mm)
            clearances.append(distance - radius)
        # 볼 장애물이 하나도 없으면 "없음"은 `None`이다. `inf`는 표준 JSON으로 적을 수 없다.
        nearest = min(clearances) if clearances else None
        slip_mm = 0.0
        if self._holding is not None and self._holding in self._grasp_pose_mm:
            current, _ = self._object_pose(self._holding)
            reference = self._grasp_pose_mm[self._holding]
            slip_mm = math.dist(
                [current[i] - position[i] for i in range(3)],
                reference,
            )
        return {
            "ee_pos_mm": position,
            "ee_quat": quat,
            "gripper_mm": self._gripper_mm(),
            # 말단 힘 센서를 그리퍼 하중의 대리값으로 쓴다. 손가락 하중 센서는 이 모델에
            # 없고, open readiness는 "하중이 실린 채로 놓지 않는다"만 보면 된다.
            "gripper_load_n": self._contact_force_n(),
            "contact_force_n": self._contact_force_n(),
            "nearest_obstacle_mm": nearest,
            "target_distance_mm": target_distance_mm,
            "holding": self._holding,
            "slip_mm": slip_mm,
            "speed_mm_s": self._ee_speed_mm_s(),
        }

    def _track_contacts(self) -> None:
        gripper = self._env.robots[0].gripper["right"]
        for plan_object in self.plan.objects:
            geoms = self._env.object_geom_names[plan_object.id]
            touching = bool(self._env.check_contact(gripper, geoms))
            if touching and not self._contacting[plan_object.id]:
                self._emit("contact_onset", object=plan_object.id)
            self._contacting[plan_object.id] = touching

    def _track_holding(self) -> None:
        gripper = self._env.robots[0].gripper["right"]
        held = None
        for plan_object in self.plan.objects:
            geoms = self._env.object_geom_names[plan_object.id]
            if self._env._check_grasp(gripper=gripper, object_geoms=geoms):
                held = plan_object.id
                break
        if held != self._holding:
            self._grasp_pose_mm.pop(self._holding, None)
            if held is not None:
                ee, _ = self._ee_pose_mm()
                position, _ = self._object_pose(held)
                self._grasp_pose_mm[held] = [position[i] - ee[i] for i in range(3)]
        self._holding = held

    def _observation(self, ack: dict[str, Any] | None) -> dict[str, Any]:
        assert self.plan is not None
        ratios = self._visibility()
        threshold = float(self.settings["visibility"]["visible_ratio_threshold"])

        objects = []
        for plan_object in self.plan.objects:
            position, quat = self._object_pose(plan_object.id)
            ratio = ratios[plan_object.id]
            visible = ratio >= threshold
            if visible:
                self._last_seen_ms[plan_object.id] = self.sim_time_ms
            objects.append(
                {
                    "id": plan_object.id,
                    "class": plan_object.shape,
                    "shape": plan_object.shape,
                    "colour": plan_object.colour,
                    "pos_mm": [round(value) for value in position],
                    "quat": self._round_quat(quat),
                    "obb_mm": [int(value) for value in plan_object.obb_mm],
                    "visible": visible,
                    "visible_ratio": round(ratio, 2),
                    "attributes": list(plan_object.attributes),
                    "last_seen_ms": int(self._last_seen_ms[plan_object.id]),
                }
            )

        ee_position, ee_quat = self._ee_pose_mm()
        instruction = self.plan.instructions[self.instruction_version - 1]
        return {
            "tick": int(self.tick),
            "sim_time_ms": int(self.sim_time_ms),
            "episode_over": self.sim_time_ms >= self.max_ms,
            "qpos": self._env.sim.data.qpos.copy(),
            "instruction": {
                "version": int(instruction.version),
                "t_ms": int(instruction.sim_ms),
                "text": instruction.text,
            },
            "objects": objects,
            "zones": [
                {"id": zone.id, "desc": zone.desc, "bounds_mm": list(zone.bounds_mm)}
                for zone in self.plan.zones
            ],
            "robot": {
                "ee_pos_mm": [round(value) for value in ee_position],
                "ee_quat": self._round_quat(ee_quat),
                "gripper_mm": round(self._gripper_mm()),
                "holding": self._holding,
                "contact_force_n": round(self._contact_force_n(), 2),
                "speed_mm_s": round(self._ee_speed_mm_s()),
            },
            "events": list(self._step_events),
            "disturbance_log": [dict(entry) for entry in self.disturbance_log],
            "ack": ack,
            "exec": {
                "seq": int(self.controller.last_seq),
                "executor": self.controller.executor,
                "action_ref": self.controller.action_ref,
                "phase": self.controller.phase,
                "path": self.controller.path_kind,
                "speed_mm_s": round(self.controller.commanded_speed_mm_s),
                "force_level": self.controller.force_level,
                "gripper": self.controller.gripper_desired,
                "stop": bool(self.controller.stopping),
                "stale": bool(self.controller.stale),
                # 감속 구간과 HOLD 절차를 실행 이력에서 구분한다 (docs/08 §6 "stale").
                "hold_after_stale": bool(self.controller.holding_after_stale),
            },
            "versions": {
                "serializer": self.serializer_version,
                "controller": self.controller.version,
            },
        }

    # ------------------------------------------------------------------
    # snapshot / restore
    # ------------------------------------------------------------------

    def _snapshot_dict(self) -> dict[str, Any]:
        assert self.plan is not None and self._env is not None
        model, data = self._env.sim.model._model, self._env.sim.data._data
        buffer = np.empty(mujoco.mj_stateSize(model, _STATE_SPEC))
        mujoco.mj_getState(model, data, buffer, _STATE_SPEC)
        osc = self._osc

        return {
            "format": "robojev-sim-snapshot-v0",
            "seed": self.seed,
            "profile": self.profile,
            "plan": self.plan.to_json(),
            # 1. 물리 — 적분에 필요한 전부(qpos·qvel·act·ctrl·mocap·userdata·warm start).
            "mujoco": {"integration": _encode_array(buffer), "time": float(data.time)},
            # 2. robosuite 내부 목표. 말단 목표는 절대 입력이라 매 주기 덮어쓰지만,
            #    그리퍼는 다르다 — `PandaGripper.format_action`이 정책 주기마다
            #    `current_action`에 ±speed를 **누적**하므로 그 값을 담지 않으면 복원 직후
            #    손가락이 다른 속도로 움직인다.
            "osc": {
                "goal_pos": _maybe_array(osc.goal_pos),
                "goal_ori": _maybe_array(osc.goal_ori),
                "relative_ori": _maybe_array(osc.relative_ori),
                "ori_ref": _maybe_array(osc.ori_ref),
                "kp": _encode_array(np.asarray(osc.kp, dtype=np.float64)),
                "kd": _encode_array(np.asarray(osc.kd, dtype=np.float64)),
                "new_update": bool(osc.new_update),
                "gripper_goal_qvel": _maybe_array(self._gripper_controller.goal_qvel),
                "gripper_current_action": _encode_array(
                    np.asarray(self._gripper_model.current_action, dtype=np.float64)
                ),
            },
            # 3. 컨트롤러 계약의 내부 상태 전부.
            "controller": self.controller.state_dict(),
            # 4. 일정과 진행도.
            "schedules": {
                "next_disturbance": int(self._next_disturbance),
                "next_instruction": int(self._next_instruction),
                "instruction_version": int(self.instruction_version),
                "disturbance_log": [dict(entry) for entry in self.disturbance_log],
            },
            # 5. wrapper 상태.
            "wrapper": {
                "sim_time_ms": int(self.sim_time_ms),
                "tick": int(self.tick),
                "holding": self._holding,
                "grasp_pose_mm": {key: list(value) for key, value in self._grasp_pose_mm.items()},
                "contacting": dict(self._contacting),
                "last_seen_ms": dict(self._last_seen_ms),
                "event_log": [dict(event) for event in self.event_log],
                "step_events": [dict(event) for event in self._step_events],
                "robosuite_timestep": int(self._env.timestep),
                "robosuite_cur_time": float(self._env.cur_time),
                "robosuite_done": bool(self._env.done),
            },
            # 6. 모든 RNG. 전역(`numpy.random`·`random`)은 담지 않는다 — 이 과정 밖에서도
            #    바뀌므로 담아도 재현을 보장하지 못하고, 담으면 복원이 남의 상태를 덮는다.
            #    대신 에피소드 RNG 두 개를 환경이 들고 있고 그것만 담는다.
            "rng": {
                "scene": self.rng.bit_generator.state,
                "robosuite": self._env.rng.bit_generator.state,
                "python": _encode_python_state(self.py_rng.getstate()),
            },
        }

    def snapshot(self) -> bytes:
        """이어 붙일 수 있는 모든 상태 (docs/05 §2). gzip한 JSON이라 열어 볼 수 있다."""
        # `allow_nan=False`: 파이썬의 json은 기본으로 `Infinity`·`NaN`을 적는데 그것은 표준
        # JSON이 아니다. 다른 언어의 파서가 거절하므로 여기서 먼저 막는다.
        payload = json.dumps(
            self._snapshot_dict(), ensure_ascii=False, sort_keys=True, allow_nan=False
        )
        return gzip.compress(payload.encode("utf-8"), mtime=0)

    @staticmethod
    def describe_snapshot(snapshot: bytes) -> dict[str, Any]:
        """snapshot 바이트를 dict로 편다. 검사와 진단이 쓴다."""
        return json.loads(
            gzip.decompress(snapshot).decode("utf-8"), parse_constant=_json_constant
        )

    def restore(self, snapshot: bytes) -> None:
        state = self.describe_snapshot(snapshot)
        if state.get("format") != "robojev-sim-snapshot-v0":
            raise ValueError(f"모르는 snapshot 형식이다: {state.get('format')!r}")

        plan = ScenePlan.from_json(state["plan"])
        if self.plan is None or plan.model_signature() != self.plan.model_signature():
            self.seed = int(state["seed"])
            self.profile = str(state["profile"])
            self.settings = merge_profile(self.config, self.profile)
            self.plan = plan
            self._build_sim()
            self._reset_episode_state()
        self.plan = plan
        self._env.plan = plan
        self.seed = int(state["seed"])

        model, data = self._env.sim.model._model, self._env.sim.data._data
        buffer = _decode_array(state["mujoco"]["integration"])
        mujoco.mj_setState(model, data, buffer, _STATE_SPEC)
        # 적분 상태에 든 시각과 따로 적어 둔 시각이 어긋나면 snapshot이 깨진 것이다.
        if abs(float(data.time) - float(state["mujoco"]["time"])) > _TIMESTEP_TOLERANCE_S:
            raise ValueError(
                f"snapshot의 물리 시각이 적분 상태와 다르다: {state['mujoco']['time']}s "
                f"vs {data.time}s"
            )

        osc_state = state["osc"]
        osc = self._osc
        osc.goal_pos = _maybe_decode(osc_state["goal_pos"])
        osc.goal_ori = _maybe_decode(osc_state["goal_ori"], shape=(3, 3))
        osc.relative_ori = _maybe_decode(osc_state["relative_ori"])
        osc.ori_ref = _maybe_decode(osc_state["ori_ref"], shape=(3, 3))
        osc.kp = _decode_array(osc_state["kp"])
        osc.kd = _decode_array(osc_state["kd"])
        self._gripper_controller.goal_qvel = _maybe_decode(osc_state["gripper_goal_qvel"])
        self._gripper_model.current_action = _decode_array(osc_state["gripper_current_action"])

        # 부품 컨트롤러가 들고 있는 파생 캐시(말단 자세·관절·질량 행렬)를 복원한 물리에서
        # 다시 계산한다. `force=True`는 안에서 `mj_forward`를 부르므로 warm start가 바뀐다 —
        # 그래서 캐시를 먼저 새로 쓰고, 그 다음에 적분 상태를 통째로 되돌려 놓는다.
        for part in self._env.robots[0].composite_controller.part_controllers.values():
            part.update(force=True)
            part.new_update = False
        osc.new_update = bool(osc_state["new_update"])
        mujoco.mj_setState(model, data, buffer, _WARMSTART_SPEC | _STATE_SPEC)

        self.controller.load_state_dict(state["controller"])

        schedules = state["schedules"]
        self._next_disturbance = int(schedules["next_disturbance"])
        self._next_instruction = int(schedules["next_instruction"])
        self.instruction_version = int(schedules["instruction_version"])
        self.disturbance_log = [dict(entry) for entry in schedules["disturbance_log"]]

        wrapper = state["wrapper"]
        self.sim_time_ms = int(wrapper["sim_time_ms"])
        self.tick = int(wrapper["tick"])
        self._holding = wrapper["holding"]
        self._grasp_pose_mm = {key: list(value) for key, value in wrapper["grasp_pose_mm"].items()}
        self._contacting = dict(wrapper["contacting"])
        self._last_seen_ms = dict(wrapper["last_seen_ms"])
        self.event_log = [dict(event) for event in wrapper["event_log"]]
        self._step_events = [dict(event) for event in wrapper["step_events"]]
        self._env.timestep = int(wrapper["robosuite_timestep"])
        self._env.cur_time = float(wrapper["robosuite_cur_time"])
        self._env.done = bool(wrapper["robosuite_done"])

        rng = state["rng"]
        self.rng = np.random.default_rng()
        self.rng.bit_generator.state = rng["scene"]
        self._env.rng = np.random.default_rng()
        self._env.rng.bit_generator.state = rng["robosuite"]
        self.py_rng = random.Random()
        self.py_rng.setstate(_decode_python_state(rng["python"]))

    def close(self) -> None:
        if self._env is not None:
            self._env.close()
            self._env = None
            self._model_signature = None


# --------------------------------------------------------------------------
# RNG 상태 직렬화
# --------------------------------------------------------------------------


def _encode_python_state(state: tuple) -> dict[str, Any]:
    version, internal, gauss = state
    return {"version": int(version), "internal": list(internal), "gauss": gauss}


def _decode_python_state(state: dict[str, Any]) -> tuple:
    return (int(state["version"]), tuple(int(value) for value in state["internal"]), state["gauss"])


def _load_yaml(path: Path) -> dict[str, Any]:
    import yaml

    return yaml.safe_load(path.read_text(encoding="utf-8"))
