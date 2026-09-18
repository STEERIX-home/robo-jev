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

        timing = config["timing"]
        self.control_hz: int = int(timing["control_hz"])
        self.period_ms: int = 1000 // self.control_hz
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
        self.default_force_level = next(iter(self.force_levels))

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
            "nearest_obstacle_mm": math.inf,
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

    def _normalise_command(self, command: dict[str, Any], now_ms: int) -> dict[str, Any] | None:
        """명령을 실행기 하나 + 수명 필드로 편다.

        두 형식을 받는다. docs/08 §6의 전체 명령과, 계획서의 `{"kind": "HOLD", …}`처럼
        실행기를 직접 부르는 원시 형식이다. 원시 형식에는 수명 필드가 없으므로
        "지금 관측해 지금 발행한" 명령으로 채운다 — 값이 `now_ms`와 순번에서만 오므로
        snapshot 복원 뒤에도 같은 명령이 같은 수명을 갖는다.
        """
        kind = command.get("kind")
        if kind is not None:
            if kind not in EXECUTORS:
                return None
            executor = kind
            path_kind = "hold" if executor not in _MOVING_EXECUTORS else "direct"
            target = command.get("target_mm")
            speed_level = command.get("speed_level")
            if speed_level is None:
                speed_level = 0 if executor not in _MOVING_EXECUTORS else 2
            force_level = command.get("force_level")
            if force_level is None:
                force_level = "push" if executor == "PUSH_SEGMENT" else self.default_force_level
            normalised = {
                "executor": executor,
                "seq": command.get("seq", self.last_seq + 1),
                "observed_at": command.get("observed_at", now_ms),
                "issued_at": command.get("issued_at", now_ms),
                "lease_until": command.get("lease_until"),
                "path_kind": path_kind,
                "target_mm": target,
                "target_quat": command.get("target_quat"),
                "speed_level": speed_level,
                "force_level": force_level,
                "gripper": command.get("gripper"),
                "stop": bool(command.get("stop", False)),
                "geometry_age_ms": command.get("geometry_age_ms"),
                "target_moving": bool(command.get("target_moving", False)),
                "forbidden_segment": bool(command.get("forbidden_segment", False)),
                "action_ref": command.get("action_ref"),
                "phase": command.get("phase"),
            }
        else:
            path = command.get("path") or {}
            path_kind = path.get("kind", "hold")
            if path_kind not in _PATH_KINDS:
                return None
            target = path.get("target_mm") if path_kind != "via" else path.get("waypoint_mm")
            if path_kind == "via" and target is None:
                target = path.get("target_mm")
            executor = self._executor_for(command, path_kind)
            constraints = command.get("constraints") or {}
            normalised = {
                "executor": executor,
                "seq": command.get("seq", self.last_seq + 1),
                "observed_at": command.get("observed_at", now_ms),
                "issued_at": command.get("issued_at", now_ms),
                "lease_until": command.get("lease_until"),
                "path_kind": path_kind,
                "target_mm": target,
                "target_quat": path.get("target_quat"),
                "speed_level": command.get("speed_level", 0),
                "force_level": command.get("force_level", self.default_force_level),
                "gripper": command.get("gripper"),
                "stop": bool(command.get("stop", False)),
                "geometry_age_ms": command.get("geometry_age_ms"),
                "target_moving": bool(command.get("target_moving", False)),
                "forbidden_segment": bool(constraints.get("forbidden_segment", False)),
                "action_ref": command.get("action_ref"),
                "phase": command.get("phase"),
            }

        if normalised["force_level"] not in self.force_levels:
            return None
        level = normalised["speed_level"]
        if not isinstance(level, int) or isinstance(level, bool):
            return None
        if not 0 <= level < len(self.speed_levels_mm_s):
            return None
        if normalised["gripper"] is not None and normalised["gripper"] not in _GRIPPER_STATES:
            return None
        return normalised

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
            "lease_until": int(self.lease_until),
        }
        ack.update(fields)
        return ack

    def apply(self, command: dict[str, Any], now_ms: int) -> dict[str, Any]:
        """명령 하나를 계약대로 검사하고 ACK를 돌려준다 (docs/08 §6)."""
        self.now_ms = int(now_ms)
        normalised = self._normalise_command(command, self.now_ms)
        if normalised is None:
            seq = command.get("seq", self.last_seq + 1)
            self._record("rejected", seq=seq, reason="unknown_executor")
            return self._ack(seq, None, rejected=True, reason="unknown_executor")

        seq = int(normalised["seq"])
        executor = normalised["executor"]

        # 1. 순번 역행은 폐기한다. lease도 last_seq도 건드리지 않는다.
        if seq <= self.last_seq:
            self._record("discarded", seq=seq, reason="seq_regression")
            return self._ack(seq, executor, rejected=True, reason="seq_regression")
        # 검사를 통과한 순번은 적용 여부와 무관하게 기록한다. 그래야 폐기된 명령보다
        # 오래된 명령이 나중에 되살아나지 않는다.
        self.last_seq = seq

        # 2. 관측 deadline. 늦은 응답에는 lease가 새로 붙지 않는다.
        if self.now_ms - int(normalised["observed_at"]) > self.observation_deadline_ms:
            self._record("discarded", seq=seq, reason="observation_deadline")
            return self._ack(seq, executor, stale=True, reason="observation_deadline")

        # 3. 기하 나이 허용치. 넘으면 적용하지 않고 관측을 요청한다.
        age = normalised["geometry_age_ms"]
        if age is not None and normalised["target_mm"] is not None:
            tolerance = (
                self.geometry_age_moving_ms
                if normalised["target_moving"]
                else self.geometry_age_static_ms
            )
            if float(age) > tolerance:
                self._record("discarded", seq=seq, reason="geometry_age", age_ms=age)
                return self._ack(
                    seq, executor, reason="geometry_age", request_observation=True
                )

        # 4. 정지는 혼합을 우회한다. 반사와 같은 우선순위이므로 다른 검사보다 앞선다.
        if normalised["stop"]:
            self._begin_stop("command")
            self._refresh_lease(normalised)
            return self._ack(seq, "HOLD", applied=True, stop_transition=True)

        # 5. 국소 도달·충돌 검사.
        target = self._resolve_target(normalised)
        rejection = self._check_reach(target)
        if rejection is not None:
            self._record("rejected", seq=seq, reason=rejection)
            return self._ack(seq, executor, rejected=True, reason=rejection)

        # 6. 전환 구간 검사. 걸리면 명령 대신 정지 전이를 한다.
        if normalised["forbidden_segment"]:
            self._begin_stop("transition_collision")
            self._record("rejected", seq=seq, reason="transition_collision")
            return self._ack(
                seq, executor, stop_transition=True, reason="transition_collision"
            )

        # 7. 채택. 목표 속도를 100ms 동안 보간한다.
        self._adopt(normalised, target)
        self._refresh_lease(normalised)

        # 8. 그리퍼. 상태가 바뀔 때만 readiness를 보고 이벤트를 한 번 낸다.
        event_id, wait = self._set_gripper(normalised["gripper"])
        return self._ack(
            seq,
            executor,
            applied=True,
            gripper_event=event_id,
            gripper_wait=wait,
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
        if kind in ("hold", "retreat") or normalised["target_mm"] is None:
            # retreat은 국소 플래너(3b)가 경유점을 채우기 전까지 제자리 유지다.
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
        self.stopping = False

    # ------------------------------------------------------------------
    # 그리퍼
    # ------------------------------------------------------------------

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

    def _begin_stop(self, cause: str, at_ms: int | None = None) -> None:
        """감속 프로파일을 건다.

        `at_ms`는 감속이 **시작된 시각**이다. lease 만료는 주기가 그것을 알아차린
        시각이 아니라 만료 시각부터 감속한다. 그래야 주기 경계가 어디에 놓이든
        같은 모의 시각에 같은 속도가 된다.
        """
        if self.stopping:
            return
        self.stopping = True
        self.decel_start_ms = self.now_ms if at_ms is None else int(at_ms)
        self.decel_from_speed_mm_s = self.speed_mm_s
        self.executor = "HOLD"
        self.path_kind = "hold"
        self.goal_mm = list(self.setpoint_mm)
        self.goal_quat = list(self.setpoint_quat)
        self._record("stop_transition", cause=cause)

    def _reflexes(self) -> float:
        """반사는 모델 응답을 기다리지 않는다 (docs/08 §6 "반사"). 속도 계수를 돌려준다."""
        factor = 1.0
        if float(self.sensors.get("contact_force_n") or 0.0) > self.force_limit_n:
            if not self.stopping:
                self._record("reflex_stop", force_n=float(self.sensors["contact_force_n"]))
                self._begin_stop("reflex_force")
            factor = 0.0
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
        holding_after_stale = False
        if self.now_ms > self.lease_until:
            if not self.stale:
                self.stale = True
                self.stale_since_ms = self.lease_until
                self._record("stale", lease_until=int(self.lease_until))
            self._begin_stop("lease_expired", at_ms=self.stale_since_ms)
            if self.now_ms - int(self.stale_since_ms or self.lease_until) >= self.hold_after_stale_ms:
                holding_after_stale = True
                self.executor = "HOLD"

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
            "holding_after_stale": holding_after_stale,
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
