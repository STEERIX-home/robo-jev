"""컨트롤러 계약 v0 — docs/08 §6.

`Controller`는 하네스와 OSC 사이의 실행기다. 물리를 모르고, 오직

* 명령의 **수명**(관측 deadline·기하 나이·발행 lease·순번),
* 새 명령으로의 **혼합**(100ms)과 전환 구간 검사,
* **정지 전이**(혼합 우회, 고정 감속),
* **반사**(힘 한계·근접·미끄러짐 — 모델 응답을 기다리지 않는다),
* **그리퍼 이벤트**(readiness·ID·ACK·멱등),
* 국소 도달·충돌 **거절**

만 다룬다. 한 제어 주기의 결과는 로봇 기준 좌표계의 절대 말단 목표 하나이고,
`sim/environment.py`가 그것을 robosuite OSC_POSE에 넘긴다.

세 가지를 나눠 부른다. `observe(sensors)`는 실행기가 아는 자기 상태를 갱신하고,
`apply(command, now_ms)`는 명령 하나를 계약대로 검사해 ACK를 돌려주며,
`advance(now_ms)`는 명령이 없어도 매 주기 돌면서 반사·혼합·감속을 진행시킨다.
반사가 명령과 무관하게 동작해야 하므로 이 셋을 합치지 않는다.

계약의 수치는 전부 `configs/controller/osc_v0.yaml`에 있다. 이 파일에는 상수가 없다.
"""

from __future__ import annotations

import copy
import math
from pathlib import Path
from typing import Any

import yaml

__all__ = ["EXECUTORS", "Controller", "load_controller_config", "resolve_config_path"]

#: 실행기 등록 목록 (docs/02 §4 표). 명령은 이 원시 기능 하나로 사상된다.
EXECUTORS = (
    "MOVE_EE",
    "SET_GRIPPER",
    "PUSH_SEGMENT",
    "OBSERVE",
    "HOLD",
    "REQUEST_REPLAN",
)

#: 말단을 움직이는 실행기. 나머지는 자세를 그대로 두고 제자리에서 감속한다.
_MOVING_EXECUTORS = ("MOVE_EE", "PUSH_SEGMENT")

#: 경로 종류 (docs/08 §6 `path`).
_PATH_KINDS = ("direct", "via", "retreat", "hold")

_GRIPPER_STATES = ("open", "closed")

#: 이 모듈이 쓰는 quaternion 순서. 설정의 `frame.quaternion_order`와 대조한다.
_QUATERNION_ORDER = "xyzw"

_PACKAGE_ROOT = Path(__file__).resolve().parents[3]


def resolve_config_path(path: str | Path) -> Path:
    """설정 경로를 푼다.

    계획서의 검사는 `configs/sim/tidy_clutter.yaml`처럼 저장소 뿌리 기준의 상대
    경로를 쓴다. 현재 작업 디렉터리에 없으면 설치된 패키지 위치에서 한 번 더 찾아,
    검사를 어디서 돌리든 같은 설정을 읽게 한다.
    """
    candidate = Path(path)
    if candidate.is_file():
        return candidate
    if not candidate.is_absolute():
        fallback = _PACKAGE_ROOT / candidate
        if fallback.is_file():
            return fallback
    raise FileNotFoundError(f"설정 파일을 찾을 수 없다: {path}")


def load_controller_config(path: str | Path) -> dict[str, Any]:
    return yaml.safe_load(resolve_config_path(path).read_text(encoding="utf-8"))


def _norm(vector: list[float]) -> float:
    return math.sqrt(sum(value * value for value in vector))


def _quat_normalise(quat: list[float]) -> list[float]:
    length = _norm(quat)
    if length == 0.0:
        return [0.0, 0.0, 0.0, 1.0]
    return [value / length for value in quat]


def _quat_blend(start: list[float], end: list[float], alpha: float) -> list[float]:
    """부호를 맞춘 nlerp.

    slerp를 쓰지 않는 이유는 이 구간이 100ms(5주기)뿐이고 두 자세의 각 차이가 작아
    nlerp와 slerp의 차이가 각속도 프로파일의 미세한 비선형뿐이기 때문이다. 정지·거절
    판정은 위치로 하므로 여기에 의존하지 않는다.
    """
    if sum(a * b for a, b in zip(start, end)) < 0.0:
        end = [-value for value in end]
    return _quat_normalise([a + (b - a) * alpha for a, b in zip(start, end)])


class Controller:
    """명령 하나를 주기마다의 말단 목표로 바꾸는 실행기 (docs/08 §6)."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = copy.deepcopy(config)
        self.version = str(config.get("version", "c0"))

        # 자세 표현 규약. 코드가 xyzw를 가정하므로 설정이 다른 순서를 말하면 그 자리에서 막는다.
        order = str(config["frame"]["quaternion_order"])
        if order != _QUATERNION_ORDER:
            raise ValueError(
                f"이 컨트롤러는 quaternion {_QUATERNION_ORDER} 순서만 쓴다 (설정: {order!r})"
            )

        timing = config["timing"]
        self.control_hz: int = int(timing["control_hz"])
        self.period_ms: int = 1000 // self.control_hz
        # 물리 timestep은 컨트롤러가 직접 쓰지 않는다. 환경이 자기 설정과 대조하는
        # 값이며(`Environment._check_timing`), 두 설정이 어긋나면 reset이 실패한다.
        self.physics_dt_ms: int = int(timing["physics_dt_ms"])

        lifetime = config["lifetime"]
        self.observation_deadline_ms = int(lifetime["observation_deadline_ms"])
        self.lease_ms = int(lifetime["lease_ms"])
        self.geometry_age_static_ms = int(lifetime["geometry_age_static_ms"])
        self.geometry_age_moving_ms = int(lifetime["geometry_age_moving_ms"])
        self.hold_after_stale_ms = int(lifetime["hold_after_stale_ms"])

        self.blend_ms = int(config["blend"]["blend_ms"])
        self.decel_mm_s2 = float(config["stop"]["decel_mm_s2"])
        self.hold_grasp_on_stop = bool(config["stop"]["hold_grasp"])

        self.speed_levels_mm_s = [float(value) * 1000.0 for value in config["speed_levels_m_s"]]
        self.force_levels = {
            name: {
                "impedance_kp": float(values["impedance_kp"]),
                "contact_allowance_n": float(values["contact_allowance_n"]),
            }
            for name, values in config["force_levels"].items()
        }

        defaults = config["defaults"]
        self.default_speed_level_moving = int(defaults["speed_level_moving"])
        self.default_speed_level_still = int(defaults["speed_level_still"])
        self.default_force_level = str(defaults["force_level"])
        self.push_force_level = str(defaults["push_force_level"])
        for name in (self.default_force_level, self.push_force_level):
            if name not in self.force_levels:
                raise ValueError(f"defaults가 없는 force_level을 가리킨다: {name!r}")

        retreat = config["retreat"]
        vector = [float(value) for value in retreat["vector_mm"]]
        length = _norm(vector)
        if length == 0.0:
            raise ValueError("retreat.vector_mm이 길이 0이다 — 후퇴 방향을 정할 수 없다")
        self.retreat_direction = [value / length for value in vector]
        self.retreat_distance_mm = float(retreat["distance_mm"])

        reflex = config["reflex"]
        self.force_limit_n = float(reflex["force_limit_n"])
        self.proximity_mm = float(reflex["proximity_mm"])
        self.proximity_speed_factor = float(reflex["proximity_speed_factor"])
        self.slip_detect_mm = float(reflex["slip_detect_mm"])

        gripper = config["gripper"]
        self.gripper_open_mm = float(gripper["open_mm"])
        self.gripper_closed_mm = float(gripper["closed_mm"])
        self.close_readiness_distance_mm = float(gripper["close_readiness_distance_mm"])
        self.open_readiness_force_n = float(gripper["open_readiness_force_n"])
        self.gripper_event_prefix = str(gripper["event_id_prefix"])

        reach = config["reach"]
        self.workspace_radius_mm = float(reach["workspace_radius_mm"])
        self.min_height_mm = float(reach["min_height_mm"])
        self.clearance_mm = float(reach["clearance_mm"])

        declared = list(config["executors"])
        if declared != list(EXECUTORS):
            raise ValueError(f"설정의 실행기 목록이 계약과 다르다: {declared} != {list(EXECUTORS)}")

        self.reset(ee_pos_mm=[0.0, 0.0, 0.0], ee_quat=[0.0, 0.0, 0.0, 1.0], gripper_mm=self.gripper_open_mm)

    @classmethod
    def from_config_path(cls, path: str | Path) -> Controller:
        return cls(load_controller_config(path))

    # ------------------------------------------------------------------
    # 수명 주기
    # ------------------------------------------------------------------

    def reset(
        self,
        *,
        ee_pos_mm: list[float],
        ee_quat: list[float],
        gripper_mm: float,
        now_ms: int = 0,
    ) -> None:
        """에피소드 시작 상태. 목표는 현재 자세이고 명령은 아직 없다."""
        self.setpoint_mm = [float(value) for value in ee_pos_mm]
        self.setpoint_quat = _quat_normalise([float(value) for value in ee_quat])
        self.goal_mm = list(self.setpoint_mm)
        self.goal_quat = list(self.setpoint_quat)
        self.blend_from_quat = list(self.setpoint_quat)

        self.executor = "HOLD"
        self.action_ref: str | None = None
        self.phase: str | None = None
        self.path_kind = "hold"

        self.last_seq = 0
        self.lease_until = now_ms
        self.issued_at = now_ms
        self.observed_at = now_ms

        self.blend_start_ms = now_ms
        self.blend_from_speed_mm_s = 0.0
        self.commanded_speed_mm_s = 0.0
        self.speed_mm_s = 0.0

        self.force_level = self.default_force_level
        self.stopping = False
        self.holding_after_stale = False
        self.force_reflex_active = False
        self.decel_start_ms = now_ms
        self.decel_from_speed_mm_s = 0.0

        self.gripper_desired = "closed" if float(gripper_mm) <= self.gripper_closed_mm else "open"
        self.gripper_event_count = 0
        self.gripper_event_id: str | None = None

        self.slip_active = False
        self.now_ms = now_ms
        self.last_advance_ms = now_ms
        self.stale = False
        self.stale_since_ms: int | None = None

        self.sensors: dict[str, Any] = {
            "ee_pos_mm": list(self.setpoint_mm),
            "ee_quat": list(self.setpoint_quat),
            "gripper_mm": float(gripper_mm),
            "gripper_load_n": 0.0,
            "contact_force_n": 0.0,
            # "장애물 없음"은 `None`이다. `math.inf`는 표준 JSON으로 적을 수 없어
            # snapshot이 `Infinity`라는 비표준 토큰을 뱉게 된다.
            "nearest_obstacle_mm": None,
            "target_distance_mm": None,
            "holding": None,
            "slip_mm": 0.0,
            "speed_mm_s": 0.0,
        }
        self.events: list[dict[str, Any]] = []

    def observe(self, sensors: dict[str, Any]) -> None:
        """실행기가 아는 자기 상태를 갱신한다. 반사와 readiness가 이것만 본다."""
        self.sensors = dict(self.sensors)
        self.sensors.update(sensors)

    def drain_events(self) -> list[dict[str, Any]]:
        """마지막 호출 이후의 실행기 사건을 꺼낸다 (docs/08 §3.2 `events`)."""
        events, self.events = self.events, []
        return events

    def _record(self, kind: str, **fields: Any) -> dict[str, Any]:
        event = {"kind": kind, "at_ms": int(self.now_ms), **fields}
        self.events.append(event)
        return event

    # ------------------------------------------------------------------
    # 명령 해석
    # ------------------------------------------------------------------

    def _normalise_command(
        self, command: dict[str, Any], now_ms: int
    ) -> tuple[dict[str, Any] | None, str | None]:
        """명령을 실행기 하나 + 수명 필드로 편다. 실패하면 `(None, 사유)`.

        두 형식을 받는다. docs/08 §6의 전체 명령과, 계획서의 `{"kind": "HOLD", …}`처럼
        실행기를 직접 부르는 원시 형식이다. 원시 형식에는 수명 필드가 없으므로
        "지금 관측해 지금 발행한" 명령으로 채운다 — 값이 `now_ms`와 순번에서만 오므로
        snapshot 복원 뒤에도 같은 명령이 같은 수명을 갖는다. 두 형식이 다른 것은
        실행기·경로·목표를 어디서 읽느냐뿐이므로 그 셋만 갈라 읽고 나머지는 한 군데서 만든다.
        """
        kind = command.get("kind")
        if kind is not None:
            if kind not in EXECUTORS:
                return None, "unknown_executor"
            moving = kind in _MOVING_EXECUTORS
            shape = {
                "executor": kind,
                "path_kind": "direct" if moving else "hold",
                "target_mm": command.get("target_mm"),
                "target_quat": command.get("target_quat"),
                "speed_level": command.get(
                    "speed_level",
                    self.default_speed_level_moving if moving else self.default_speed_level_still,
                ),
                "force_level": command.get(
                    "force_level",
                    self.push_force_level if kind == "PUSH_SEGMENT" else self.default_force_level,
                ),
                "forbidden_segment": bool(command.get("forbidden_segment", False)),
            }
        else:
            path = command.get("path") or {}
            path_kind = path.get("kind", "hold")
            if path_kind not in _PATH_KINDS:
                return None, "invalid_path"
            if path_kind == "via" and path.get("waypoint_mm") is None:
                # 경유점 없는 via를 직선으로 바꾸지 않는다. 하네스가 그 직선을 피하려고
                # 경유점을 고른 것이므로, 대신 직진하면 계약을 어기는 쪽이 더 위험하다.
                return None, "waypoint_missing"
            shape = {
                "executor": self._executor_for(command, path_kind),
                "path_kind": path_kind,
                "target_mm": path.get("waypoint_mm") if path_kind == "via" else path.get("target_mm"),
                "target_quat": path.get("target_quat"),
                "speed_level": command.get("speed_level", self.default_speed_level_still),
                "force_level": command.get("force_level", self.default_force_level),
                "forbidden_segment": bool(
                    (command.get("constraints") or {}).get("forbidden_segment", False)
                ),
            }

        normalised = {
            **shape,
            "seq": command.get("seq", self.last_seq + 1),
            "observed_at": command.get("observed_at", now_ms),
            "issued_at": command.get("issued_at", now_ms),
            "lease_until": command.get("lease_until"),
            "gripper": command.get("gripper"),
            "stop": bool(command.get("stop", False)),
            "geometry_age_ms": command.get("geometry_age_ms"),
            "target_moving": bool(command.get("target_moving", False)),
            "action_ref": command.get("action_ref"),
            "phase": command.get("phase"),
        }

        if normalised["force_level"] not in self.force_levels:
            return None, "invalid_force_level"
        level = normalised["speed_level"]
        if isinstance(level, bool) or not isinstance(level, int):
            return None, "invalid_speed_level"
        if not 0 <= level < len(self.speed_levels_mm_s):
            return None, "invalid_speed_level"
        if normalised["gripper"] is not None and normalised["gripper"] not in _GRIPPER_STATES:
            return None, "invalid_gripper"
        return normalised, None

    def _executor_for(self, command: dict[str, Any], path_kind: str) -> str:
        """docs/02 §4의 대응표. 명령 하나는 원시 기능 하나로 간다."""
        if command.get("stop"):
            return "HOLD"
        if command.get("replan"):
            return "REQUEST_REPLAN"
        if command.get("observe"):
            return "OBSERVE"
        if path_kind == "hold":
            # 자세를 두고 그리퍼만 바꾸는 명령은 SET_GRIPPER다.
            if command.get("gripper") is not None and command["gripper"] != self.gripper_desired:
                return "SET_GRIPPER"
            return "HOLD"
        if str(command.get("phase", "")) == "push":
            return "PUSH_SEGMENT"
        return "MOVE_EE"

    # ------------------------------------------------------------------
    # apply
    # ------------------------------------------------------------------

    def _ack(self, seq: int, executor: str | None, **fields: Any) -> dict[str, Any]:
        ack = {
            "seq": int(seq),
            "applied": False,
            "reason": None,
            "gripper_event": None,
            "gripper_wait": None,
            "stale": False,
            "rejected": False,
            "stop_transition": False,
            "request_observation": False,
            "executor": executor,
            "path": None,
            "lease_until": int(self.lease_until),
        }
        ack.update(fields)
        return ack

    def _lifetime_fault(self, normalised: dict[str, Any]) -> str | None:
        """수명 검사의 결과 사유. 걸리는 것이 없으면 `None`.

        관측 deadline과 기하 나이를 한 군데서 본다. 정지는 이 사유를 **기록만** 하고
        진행하며, 나머지 명령은 여기서 멈춘다.
        """
        if self.now_ms - int(normalised["observed_at"]) > self.observation_deadline_ms:
            return "observation_late"
        age = normalised["geometry_age_ms"]
        if age is not None and normalised["target_mm"] is not None:
            tolerance = (
                self.geometry_age_moving_ms
                if normalised["target_moving"]
                else self.geometry_age_static_ms
            )
            if float(age) > tolerance:
                return "geometry_age"
        return None

    def apply(self, command: dict[str, Any], now_ms: int) -> dict[str, Any]:
        """명령 하나를 계약대로 검사하고 ACK를 돌려준다 (docs/08 §6)."""
        self.now_ms = int(now_ms)
        normalised, fault = self._normalise_command(command, self.now_ms)
        if normalised is None:
            seq = command.get("seq", self.last_seq + 1)
            self._record("rejected", seq=seq, reason=fault)
            return self._ack(seq, None, rejected=True, reason=fault)

        seq = int(normalised["seq"])
        executor = normalised["executor"]

        # 1. 순번 역행은 폐기한다. lease도 last_seq도 건드리지 않는다. 정지도 예외가 아니다 —
        #    역행한 명령은 지금 상태에 대한 판단이 아니다.
        if seq <= self.last_seq:
            self._record("discarded", seq=seq, reason="seq_regression")
            return self._ack(seq, executor, rejected=True, reason="seq_regression")
        # 검사를 통과한 순번은 적용 여부와 무관하게 기록한다. 그래야 폐기된 명령보다
        # 오래된 명령이 나중에 되살아나지 않는다.
        self.last_seq = seq

        lifetime_fault = self._lifetime_fault(normalised)

        # 2. 정지는 반사와 같은 우선순위다(docs/08 §5 1항: "허용 지연이 없다"). 관측이 늦었거나
        #    기하가 오래됐다는 이유로 **버리지 않는다** — 늦은 정지도 정지다. 사유는 ACK에
        #    남겨 하네스가 집계할 수 있게 하고, 늦은 명령이므로 lease만 새로 붙이지 않는다.
        if normalised["stop"]:
            self._begin_stop("command")
            if lifetime_fault is None:
                self._refresh_lease(normalised)
            else:
                self._record("stop_late", seq=seq, reason=lifetime_fault)
            return self._ack(
                seq,
                # 하드코딩하지 않는다. ACK가 말하는 실행기·경로는 실행 기록이 말할 것과
                # 같은 값이어야 하고, 그것은 `_begin_stop`이 방금 정했다.
                self.executor,
                applied=True,
                stop_transition=True,
                reason=lifetime_fault,
                stale=lifetime_fault == "observation_late",
                gripper_wait=self._stop_tick_gripper(normalised["gripper"]),
                path=self.path_kind,
            )

        # 3. 나머지 명령은 수명 검사에서 멈춘다. 늦은 응답에는 lease가 새로 붙지 않고,
        #    기하가 오래됐으면 관측 분기로 보낸다.
        if lifetime_fault == "observation_late":
            self._record("discarded", seq=seq, reason=lifetime_fault)
            return self._ack(seq, executor, stale=True, reason=lifetime_fault)
        if lifetime_fault == "geometry_age":
            self._record(
                "discarded", seq=seq, reason=lifetime_fault, age_ms=normalised["geometry_age_ms"]
            )
            return self._ack(seq, executor, reason=lifetime_fault, request_observation=True)

        # 4. 국소 도달·충돌 검사.
        target = self._resolve_target(normalised)
        rejection = self._check_reach(target)
        if rejection is not None:
            self._record("rejected", seq=seq, reason=rejection)
            return self._ack(seq, executor, rejected=True, reason=rejection)

        # 5. 전환 구간 검사. 걸리면 명령 대신 정지 전이를 한다.
        if normalised["forbidden_segment"]:
            self._begin_stop("transition_collision")
            self._record("rejected", seq=seq, reason="transition_collision")
            return self._ack(
                seq,
                self.executor,
                stop_transition=True,
                reason="transition_collision",
                path=self.path_kind,
            )

        # 6. 채택. 목표 속도를 100ms 동안 보간한다.
        self._adopt(normalised, target)
        self._refresh_lease(normalised)

        # 7. 그리퍼. 상태가 바뀔 때만 readiness를 보고 이벤트를 한 번 낸다.
        event_id, wait = self._set_gripper(normalised["gripper"])
        return self._ack(
            seq,
            executor,
            applied=True,
            gripper_event=event_id,
            gripper_wait=wait,
            path=normalised["path_kind"],
        )

    def _refresh_lease(self, normalised: dict[str, Any]) -> None:
        issued_at = int(normalised["issued_at"])
        lease_until = normalised["lease_until"]
        self.issued_at = issued_at
        self.observed_at = int(normalised["observed_at"])
        self.lease_until = int(lease_until) if lease_until is not None else issued_at + self.lease_ms
        self.stale = False
        self.stale_since_ms = None

    def _resolve_target(self, normalised: dict[str, Any]) -> dict[str, Any]:
        """경로 종류를 말단 목표로 바꾼다."""
        kind = normalised["path_kind"]
        if kind == "retreat":
            # 설정된 후퇴 방향으로 실제로 물러난다. 제자리에 두고 MOVE_EE라고 기록하면
            # 실행 이력이 하지 않은 일을 말하게 된다.
            here = [float(value) for value in self.sensors["ee_pos_mm"]]
            return {
                "pos_mm": [
                    value + direction * self.retreat_distance_mm
                    for value, direction in zip(here, self.retreat_direction)
                ],
                "quat": list(self.setpoint_quat),
            }
        if kind == "hold" or normalised["target_mm"] is None:
            return {"pos_mm": list(self.sensors["ee_pos_mm"]), "quat": list(self.setpoint_quat)}
        quat = normalised["target_quat"]
        return {
            "pos_mm": [float(value) for value in normalised["target_mm"]],
            "quat": _quat_normalise([float(v) for v in quat]) if quat else list(self.setpoint_quat),
        }

    def _check_reach(self, target: dict[str, Any]) -> str | None:
        """국소 도달·충돌 검사 (docs/08 §6 "거절")."""
        position = target["pos_mm"]
        if _norm(position) > self.workspace_radius_mm:
            return "unreachable"
        if position[2] < self.min_height_mm + self.clearance_mm:
            return "collision"
        return None

    def _adopt(self, normalised: dict[str, Any], target: dict[str, Any]) -> None:
        self.executor = normalised["executor"]
        self.action_ref = normalised["action_ref"]
        self.phase = normalised["phase"]
        self.path_kind = normalised["path_kind"]
        self.force_level = normalised["force_level"]
        self.goal_mm = list(target["pos_mm"])
        self.goal_quat = list(target["quat"])
        self.blend_from_quat = list(self.setpoint_quat)
        self.blend_from_speed_mm_s = self.speed_mm_s
        self.blend_start_ms = self.now_ms
        self.commanded_speed_mm_s = self.speed_levels_mm_s[normalised["speed_level"]]
        # 명령을 받아들이면 정지도 HOLD 절차도 끝난다. 표시를 내려야 다음 만료의 진입이
        # 다시 사건으로 보인다 — 걸쇠로 남으면 팔이 움직이는데도 HOLD라고 기록된다.
        self.stopping = False
        self.holding_after_stale = False

    # ------------------------------------------------------------------
    # 그리퍼
    # ------------------------------------------------------------------

    def _stop_tick_gripper(self, desired: str | None) -> str | None:
        """정지 틱의 그리퍼 답은 적용하지 않는다. 기다린 사유를 돌려준다.

        docs/08 §5 1항은 정지 틱에 "다른 답은 이 틱에 적용하지 않는다"고 못박는다. 폐기가
        아니라 보류이므로 다음 틱에 같은 답이 오면 그때 실행된다. 파지 중이면 사유를
        `stop_holds_grasp`로 구분해, docs/08 §6의 "정지 전이 … 파지 중이면 유지"가 실제로
        걸렸다는 것을 기록에 남긴다(설정 `stop.hold_grasp`).
        """
        if desired is None or desired == self.gripper_desired:
            return None
        holding = self.sensors.get("holding") is not None
        reason = "stop_holds_grasp" if self.hold_grasp_on_stop and holding else "stop_tick"
        self._record("gripper_wait", desired=desired, reason=reason)
        return reason

    def _set_gripper(self, desired: str | None) -> tuple[str | None, str | None]:
        """원하는 상태가 바뀔 때만 이벤트 하나. 같은 상태가 반복돼도 다시 나지 않는다."""
        if desired is None or desired == self.gripper_desired:
            return None, None

        reason = self._gripper_readiness(desired)
        if reason is not None:
            self._record("gripper_wait", desired=desired, reason=reason)
            return None, reason

        self.gripper_event_count += 1
        event_id = f"{self.gripper_event_prefix}-{self.gripper_event_count:04d}"
        self.gripper_desired = desired
        self.gripper_event_id = event_id
        self._record("gripper", id=event_id, desired=desired)
        return event_id, None

    def _gripper_readiness(self, desired: str) -> str | None:
        """close는 대상 접촉·도달, open은 해제 readiness를 본다 (docs/08 §4)."""
        if desired == "closed":
            distance = self.sensors.get("target_distance_mm")
            # 대상을 참조하지 않는 close에는 거리 조건이 없다.
            if distance is not None and float(distance) > self.close_readiness_distance_mm:
                return "readiness"
            return None
        load = float(self.sensors.get("gripper_load_n") or 0.0)
        if load > self.open_readiness_force_n:
            return "readiness"
        return None

    # ------------------------------------------------------------------
    # advance — 명령이 없어도 매 주기 돈다
    # ------------------------------------------------------------------

    def _begin_stop(
        self, cause: str, at_ms: int | None = None, *, hold: bool = True, fresh: bool = True
    ) -> None:
        """감속 프로파일을 건다.

        `at_ms`는 감속이 **시작된 시각**이다. lease 만료는 주기가 그것을 알아차린
        시각이 아니라 만료 시각부터 감속한다. 그래야 주기 경계가 어디에 놓이든
        같은 모의 시각에 같은 속도가 된다.

        `hold`는 이것이 곧 HOLD 절차인지를 가른다. 명령 정지·반사·전환 구간 충돌은
        그 자리에서 HOLD다. lease 만료는 아니다 — docs/08 §6은 "감속 정지"와 "1초 이상
        지속되면 HOLD 절차"를 나눠 적으므로, 감속 중에는 하던 행동이 실행 기록에 남는다.

        `fresh`는 이 호출이 **새 전이**인지, 이미 걸린 조건을 다시 알아차린 것인지를
        가른다. lease 만료는 stale인 동안 주기마다 다시 오므로 새 전이가 아니다(사건을
        되풀이하면 안 된다). 명령 정지·전환 구간 충돌·반사의 발생은 감속 중에 와도
        그 자체로 새 전이다 — 사건을 남기고 HOLD로 올린다.
        """
        if self.stopping and not fresh:
            return
        if not self.stopping:
            # 감속의 시작점. 이미 감속 중이면 프로파일을 새로 깔지 않는다 —
            # 그러면 현재 속도에서 다시 시작해 정지가 느려진다.
            self.decel_start_ms = self.now_ms if at_ms is None else int(at_ms)
            self.decel_from_speed_mm_s = self.speed_mm_s
        self.stopping = True
        if hold:
            self._enter_hold()
        self.path_kind = "hold"
        self.goal_mm = list(self.setpoint_mm)
        self.goal_quat = list(self.setpoint_quat)
        self._record("stop_transition", cause=cause)

    def _enter_hold(self) -> None:
        """HOLD 절차에 든다. 사건은 **들어갈 때마다** 한 번씩 난다.

        나가는 자리는 `_adopt`다 — 새 명령을 받으면 절차가 끝나므로 표시를 내린다.
        그래야 다음 만료의 진입이 다시 사건으로 보인다.
        """
        self.executor = "HOLD"
        if not self.holding_after_stale:
            self.holding_after_stale = True
            self._record("hold_entered")

    def _reflexes(self) -> float:
        """반사는 모델 응답을 기다리지 않는다 (docs/08 §6 "반사"). 속도 계수를 돌려준다."""
        factor = 1.0
        over_force = float(self.sensors.get("contact_force_n") or 0.0) > self.force_limit_n
        if over_force:
            # 이미 멈추는 중이어도 힘 한계 초과는 그 자체로 사건이다. 다만 주기마다
            # 되풀이하지 않도록 발생(onset)에서만 적는다.
            if not self.force_reflex_active:
                self._record("reflex_stop", force_n=float(self.sensors["contact_force_n"]))
            # 힘 한계의 **발생**은 감속 중에 와도 새 전이다 — 그 자리에서 HOLD로 올린다.
            # 그 뒤로 힘이 계속 걸려 있는 동안은 같은 조건을 다시 알아차린 것뿐이다.
            self._begin_stop("reflex_force", fresh=not self.force_reflex_active)
            factor = 0.0
        self.force_reflex_active = over_force
        nearest = self.sensors.get("nearest_obstacle_mm")
        if nearest is not None and float(nearest) < self.proximity_mm:
            factor = min(factor, self.proximity_speed_factor)

        slipping = (
            self.sensors.get("holding") is not None
            and float(self.sensors.get("slip_mm") or 0.0) > self.slip_detect_mm
        )
        if slipping and not self.slip_active:
            # 파지는 유지하고 사건만 보고한다.
            self._record("reflex_slip", slip_mm=float(self.sensors["slip_mm"]))
        self.slip_active = slipping
        return factor

    def advance(self, now_ms: int) -> dict[str, Any]:
        """제어 주기 하나. 반사 → 수명 → 혼합·감속 → 말단 목표 순이다."""
        self.now_ms = int(now_ms)
        dt_s = max(0, self.now_ms - self.last_advance_ms) / 1000.0
        self.last_advance_ms = self.now_ms

        reflex_factor = self._reflexes()

        # lease 만료: 감속 정지하고 그리퍼 상태를 유지한다. 1초 이상 이어지면 HOLD 절차.
        # 두 단계는 실행 기록에서 구분된다 — 감속 중에는 하던 실행기가 그대로 남고,
        # HOLD로 넘어갈 때 `hold_entered` 사건이 한 번 난다.
        if self.now_ms > self.lease_until:
            if not self.stale:
                self.stale = True
                self.stale_since_ms = self.lease_until
                self._record("stale", lease_until=int(self.lease_until), reason="lease_expired")
            # stale인 동안 주기마다 다시 온다. 새 전이가 아니므로 사건을 되풀이하지 않는다.
            self._begin_stop(
                "lease_expired", at_ms=self.stale_since_ms, hold=False, fresh=False
            )
            since = int(self.stale_since_ms if self.stale_since_ms is not None else self.lease_until)
            if self.now_ms - since >= self.hold_after_stale_ms:
                self._enter_hold()

        if self.stopping:
            elapsed_s = max(0, self.now_ms - self.decel_start_ms) / 1000.0
            speed = max(0.0, self.decel_from_speed_mm_s - self.decel_mm_s2 * elapsed_s)
            blend_alpha = 1.0
            blending = False
            speed_cap = speed
        else:
            span = max(1, self.blend_ms)
            blend_alpha = min(1.0, max(0.0, (self.now_ms - self.blend_start_ms) / span))
            blending = blend_alpha < 1.0
            speed_cap = self.commanded_speed_mm_s * reflex_factor
            speed = (
                self.blend_from_speed_mm_s * (1.0 - blend_alpha) + speed_cap * blend_alpha
            )

        self.speed_mm_s = speed
        self._advance_setpoint(speed, dt_s, blend_alpha)

        return {
            "executor": self.executor,
            "action_ref": self.action_ref,
            "phase": self.phase,
            "ee_pos_mm": list(self.setpoint_mm),
            "ee_quat": list(self.setpoint_quat),
            "gripper": self.gripper_desired,
            "gripper_mm": self.gripper_open_mm
            if self.gripper_desired == "open"
            else self.gripper_closed_mm,
            "speed_mm_s": speed,
            "speed_cap_mm_s": speed_cap,
            "impedance_kp": self.force_levels[self.force_level]["impedance_kp"],
            "contact_allowance_n": self.force_levels[self.force_level]["contact_allowance_n"],
            "force_level": self.force_level,
            "blending": blending,
            "blend_alpha": blend_alpha,
            "stopping": self.stopping,
            "stale": self.stale,
            "holding_after_stale": self.holding_after_stale,
        }

    def _advance_setpoint(self, speed_mm_s: float, dt_s: float, blend_alpha: float) -> None:
        if dt_s <= 0.0 or speed_mm_s <= 0.0:
            return
        delta = [goal - current for goal, current in zip(self.goal_mm, self.setpoint_mm)]
        distance = _norm(delta)
        if distance <= 1e-12:
            self.setpoint_quat = _quat_blend(self.blend_from_quat, self.goal_quat, blend_alpha)
            return
        travel = min(distance, speed_mm_s * dt_s)
        scale = travel / distance
        self.setpoint_mm = [
            current + component * scale for current, component in zip(self.setpoint_mm, delta)
        ]
        self.setpoint_quat = _quat_blend(self.blend_from_quat, self.goal_quat, blend_alpha)

    # ------------------------------------------------------------------
    # 직렬화 — snapshot이 그대로 담는다 (docs/05 §2 "controller 내부 상태")
    # ------------------------------------------------------------------

    _STATE_FIELDS = (
        "setpoint_mm",
        "setpoint_quat",
        "goal_mm",
        "goal_quat",
        "blend_from_quat",
        "executor",
        "action_ref",
        "phase",
        "path_kind",
        "last_seq",
        "lease_until",
        "issued_at",
        "observed_at",
        "blend_start_ms",
        "blend_from_speed_mm_s",
        "commanded_speed_mm_s",
        "speed_mm_s",
        "force_level",
        "stopping",
        "holding_after_stale",
        "force_reflex_active",
        "decel_start_ms",
        "decel_from_speed_mm_s",
        "gripper_desired",
        "gripper_event_count",
        "gripper_event_id",
        "slip_active",
        "now_ms",
        "last_advance_ms",
        "stale",
        "stale_since_ms",
    )

    def state_dict(self) -> dict[str, Any]:
        state = {field: copy.deepcopy(getattr(self, field)) for field in self._STATE_FIELDS}
        state["sensors"] = copy.deepcopy(self.sensors)
        state["events"] = copy.deepcopy(self.events)
        state["version"] = self.version
        return state

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if state.get("version") != self.version:
            raise ValueError(
                f"컨트롤러 버전이 다르다: {state.get('version')!r} != {self.version!r}"
            )
        for field in self._STATE_FIELDS:
            setattr(self, field, copy.deepcopy(state[field]))
        self.sensors = copy.deepcopy(state["sensors"])
        self.events = copy.deepcopy(state["events"])
