"""3D 재구성 → 공통 구조화 상태 (docs/08 §3.2).

robojev의 입력은 앞단이 무엇이든 **같은 상태 스키마**다. 이 모듈은 그 경계를 고정한다.

* :class:`Reconstruction` — 앞단이 내는 값. 추적된 인스턴스(자세·정밀도·OBB·가시성·
  점 수·파지 가능 면), 자유 공간 요약, 소스별 시각.
* :func:`extract` — 재구성 + 로봇 고유 감각 → docs/08 §3.2의 상태 dict. **버전이 있고**
  (:data:`EXTRACTOR_VERSION`) 모든 비교군이 같은 것을 쓴다.
* :class:`GroundTruthAdapter` — D1의 앞단. 시뮬레이터 관측(:meth:`Environment.step`의
  반환)을 같은 :class:`Reconstruction`으로 옮긴다.

**정보 경계.** 가려진 물체의 참값은 상태에 들어가지 않는다. 어댑터는 마지막으로
**관측된** 자세와 그 시각만 넘기고, 지금의 참값은 :meth:`GroundTruthAdapter.evidence`로만
꺼낼 수 있다 — 그 자리는 레코드의 `evidence`이고 모델 입력이 아니다(docs/08 §3.2).

E2의 시뮬 3D 카메라와 재구성 결함 모델(표면 결손·추적 id 흔들림·지연)은 같은
:class:`Reconstruction`을 만드는 다른 어댑터로 붙는다. `extract`는 바뀌지 않는다.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "EXTRACTOR_VERSION",
    "GroundTruthAdapter",
    "Reconstruction",
    "SceneSummary",
    "TrackedInstance",
    "circumradius_mm",
    "extract",
    "named_target",
    "segment_point_distance_mm",
]

#: 추출 모듈의 버전. 레코드의 `versions.extractor`에 들어간다 (docs/08 §3.2 "앞단 조건").
EXTRACTOR_VERSION = "pw0.1"


# --------------------------------------------------------------------------
# 기하 원시 함수 — 하네스도 같은 것을 쓴다 (계산 규칙이 하나여야 한다)
# --------------------------------------------------------------------------


def circumradius_mm(obb_mm) -> float:
    """OBB의 외접 반지름. yaw를 모르고도 보수적으로 쓸 수 있는 반지름이다."""
    return math.dist((0.0, 0.0, 0.0), [value / 2.0 for value in obb_mm])


def named_target(objects, text: str) -> str | None:
    """지시문이 부르는 대상 물체의 id.

    물체 설명이 지시문에 나오는 것 중 **마지막으로 불린 평범한 물체**를 고른다. 지시가
    바뀌면 새 대상이 뒤에 오고("A 대신 B를 먼저 옮겨라"), 취약·금지 물체는 "건드리지
    마라" 쪽이므로 뺀다. 구조화된 목표(`goal.target_ref`)가 있으면 그것이 먼저다 — 이
    함수는 텍스트밖에 없을 때의 근사이며, 상태와 지시가 **같은 이름**으로 물체를 부른다는
    조건에 기댄다(docs/08 §3.2의 `objects[].설명`).
    """
    mentioned: list[tuple[int, str]] = []
    for entry in objects or ():
        if entry.get("attributes"):
            continue
        description = str(entry.get("desc") or "")
        position = text.find(description) if description else -1
        if position >= 0:
            mentioned.append((position, str(entry["id"])))
    return max(mentioned)[1] if mentioned else None


def _tracked_only(refs, tracked_ids: set[str], by_attribute: list[str]) -> list[str]:
    """구조화된 참조 목록을 추적 중인 물체로 제한하고, 속성으로 아는 것을 합친다 (순서 유지)."""
    kept = [str(ref) for ref in (refs or ()) if str(ref) in tracked_ids]
    kept.extend(ref for ref in by_attribute if ref not in kept)
    return kept


def segment_point_distance_mm(start, end, point) -> float:
    """선분과 점 사이의 최단 거리."""
    segment = [b - a for a, b in zip(start, end)]
    length_sq = sum(value * value for value in segment)
    if length_sq <= 1e-9:
        return math.dist(start, point)
    t = sum((p - a) * s for a, s, p in zip(start, segment, point)) / length_sq
    t = min(1.0, max(0.0, t))
    closest = [a + s * t for a, s in zip(start, segment)]
    return math.dist(closest, point)


# --------------------------------------------------------------------------
# 앞단이 내는 값
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class TrackedInstance:
    """틱 사이에 안정된 추적 id를 가진 인스턴스 하나 (docs/08 §3.2 `objects[]`).

    `pose_mm`은 **마지막으로 관측된** 자세다. 지금 가려져 있으면 `observed_now`가 거짓이고
    `last_seen_ms`가 그 자세의 시각을 말한다. 지금의 참값은 여기 들어오지 않는다.
    """

    track_id: str
    desc: str
    cls: str
    pose_mm: tuple[int, int, int]
    quat: tuple[float, float, float, float]
    precision_mm: int
    obb_mm: tuple[int, int, int]
    top_mm: int
    graspable_faces: tuple[str, ...]
    surface_conf: float
    visible_ratio: float
    points: int
    last_seen_ms: int
    observed_now: bool
    reid: tuple[str, ...] = ()
    attributes: tuple[str, ...] = ()
    moving: bool = False
    #: 자세의 출처. `geom`은 3D 재구성, `proprio`는 파지 중인 물체를 말단 자세에서 채운 것이다
    #: (docs/08 §3.2 "파지 중인 물체의 자세는 말단 자세에서 채운다").
    pose_source: str = "geom"

    @property
    def radius_mm(self) -> float:
        return circumradius_mm(self.obb_mm)


@dataclass(frozen=True)
class SceneSummary:
    """자유 공간·통로·작업면 요약 (docs/08 §3.2 `scene`)."""

    work_surface_mm: int
    free_width_mm: int
    corridor_mm: int
    clearance_mm: int


@dataclass(frozen=True)
class Reconstruction:
    """한 틱의 재구성 결과. 값이고, 앞단의 내부 상태를 들고 있지 않다."""

    instances: tuple[TrackedInstance, ...]
    scene: SceneSummary
    zones: tuple[dict[str, Any], ...] = ()
    goal: dict[str, Any] = field(default_factory=dict)
    events: tuple[dict[str, Any], ...] = ()
    geom_ms: int = 0
    tick: int = 0
    sim_ms: int = 0
    source: str = "ground-truth"


# --------------------------------------------------------------------------
# 추출 — 공통 스키마 (docs/08 §3.2)
# --------------------------------------------------------------------------

#: `exec`의 기본값. 이름에 `ack`를 쓰지 않는다 — `ack`는 레코드의 비입력 필드라서
#: 상태 안에 같은 이름이 있으면 계약 검사가 요청 전체를 거절한다(contracts).
_EXEC_DEFAULT: dict[str, Any] = {
    "seq": 0,
    "action_ref": None,
    "phase": None,
    "path": None,
    "speed_level": 0,
    "force_level": None,
    "gripper": None,
    "stop": False,
    "progress": None,
    "applied": None,
    "reject": None,
    "gripper_wait": None,
    "events": [],
}


def _round_mm(values) -> list[int]:
    return [int(round(float(value))) for value in values]


def extract(recon: Reconstruction, robot: dict[str, Any], now_ms: int) -> dict[str, Any]:
    """재구성 + 로봇 고유 감각 → docs/08 §3.2의 공통 상태.

    `robot`은 로봇 쪽이 아는 것이다: 말단 자세, 그리퍼 폭, 파지 중인 물체, 접촉력, 속도와
    그 관측 시각(`observed_at_ms`), 실행기의 자기 보고(`exec`), 틱 시작 시 확정된
    `commitment`. 지각 모듈은 그 셋을 만들지 않고 그대로 옮긴다 — 만드는 것은 로봇과
    하네스의 몫이고, 이 함수의 몫은 **스키마**다.

    소스별 나이는 여기서 계산한다. 기하는 10Hz보다 느리게 갱신될 수 있으므로 로봇
    고유 감각과 한 숫자로 합치지 않는다.
    """
    proprio_ms = int(robot.get("observed_at_ms", now_ms))
    if now_ms < recon.geom_ms or now_ms < proprio_ms:
        raise ValueError(
            f"관측 시각이 현재보다 뒤에 있다: geom {recon.geom_ms}ms, proprio {proprio_ms}ms "
            f"> now {now_ms}ms"
        )

    ee = [float(value) for value in robot["ee_pose_mm"]]
    objects = [_object_entry(instance, now_ms) for instance in recon.instances]
    derived = [_object_derived(instance, recon.instances, ee, recon.scene, now_ms) for instance in recon.instances]

    return {
        "t": {
            "tick": int(recon.tick),
            "sim_ms": int(recon.sim_ms),
            "observed_at_ms": proprio_ms,
            "age_ms": {"geom": now_ms - int(recon.geom_ms), "proprio": now_ms - proprio_ms},
        },
        "goal": copy.deepcopy(recon.goal),
        "objects": objects,
        "scene": {
            "work_surface_mm": int(recon.scene.work_surface_mm),
            "free_width_mm": int(recon.scene.free_width_mm),
            "corridor_mm": int(recon.scene.corridor_mm),
            "clearance_mm": int(recon.scene.clearance_mm),
        },
        "zones": [copy.deepcopy(zone) for zone in recon.zones],
        "robot": {
            "ee_pose_mm": _round_mm(ee),
            "ee_quat": [float(value) for value in robot["ee_quat"]],
            "gripper_mm": int(round(float(robot["gripper_mm"]))),
            "holding": robot.get("holding"),
            "contact_n": round(float(robot.get("contact_n") or 0.0), 2),
            "speed_mm_s": int(round(float(robot.get("speed_mm_s") or 0.0))),
        },
        "exec": {**copy.deepcopy(_EXEC_DEFAULT), **copy.deepcopy(robot.get("exec") or {})},
        "events": [copy.deepcopy(event) for event in recon.events],
        "derived": derived,
        "commitment": copy.deepcopy(robot.get("commitment")),
        # 영상·기하 soft token 슬롯은 예약만 한다 (docs/08 §3.2, §12).
        "image": [],
        "geom": [],
        "extractor": EXTRACTOR_VERSION,
    }


def _object_entry(instance: TrackedInstance, now_ms: int) -> dict[str, Any]:
    return {
        "id": instance.track_id,
        "desc": instance.desc,
        "class": instance.cls,
        "pose_mm": _round_mm(instance.pose_mm),
        "quat": [float(value) for value in instance.quat],
        "precision_mm": int(instance.precision_mm),
        "pose_source": str(instance.pose_source),
        "obb_mm": [int(value) for value in instance.obb_mm],
        "top_mm": int(instance.top_mm),
        "graspable_faces": list(instance.graspable_faces),
        "surface_conf": round(float(instance.surface_conf), 2),
        "visible_ratio": round(float(instance.visible_ratio), 2),
        "last_seen_ms": int(instance.last_seen_ms),
        "age_ms": now_ms - int(instance.last_seen_ms),
        "reid": list(instance.reid),
        "attributes": list(instance.attributes),
    }


def _object_derived(
    instance: TrackedInstance,
    instances: tuple[TrackedInstance, ...],
    ee: list[float],
    scene: SceneSummary,
    now_ms: int,
) -> dict[str, Any]:
    """물체별 파생 값 (docs/08 §3.2 `derived[]`).

    **관측된 자세로만** 계산한다. 가려진 물체는 마지막으로 본 자세를 쓰고 그 나이를 함께
    낸다 — 그것이 하네스가 아는 전부이기 때문이다.
    """
    pose = [float(value) for value in instance.pose_mm]
    others = [other for other in instances if other.track_id != instance.track_id]

    clearance = min(
        (math.dist(pose, other.pose_mm) - instance.radius_mm - other.radius_mm for other in others),
        default=float(scene.clearance_mm),
    )
    corridor = min(
        (
            2.0 * (segment_point_distance_mm(ee, pose, other.pose_mm) - other.radius_mm)
            for other in others
        ),
        default=float(scene.free_width_mm),
    )
    return {
        "object": instance.track_id,
        "relative_mm": _round_mm([p - e for p, e in zip(pose, ee)]),
        "clearance_mm": int(round(clearance)),
        "corridor_mm": max(0, min(int(round(corridor)), int(scene.free_width_mm))),
        "age_ms": now_ms - int(instance.last_seen_ms),
    }


# --------------------------------------------------------------------------
# D1 어댑터 — 시뮬레이터 참값
# --------------------------------------------------------------------------


class GroundTruthAdapter:
    """시뮬레이터 관측 → :class:`Reconstruction` (D1의 앞단).

    참값을 그대로 흘리지 않는다. 네 가지를 앞단처럼 흉내 낸다.

    1. **가시성.** 시뮬레이터의 광선 검사(`visible`·`visible_ratio`)를 그대로 쓴다.
       안 보이는 물체는 자세·크기·속성을 갱신하지 않고 마지막으로 본 것과 그 시각만 남긴다.
    2. **정밀도.** 자세 오차 범위는 설정의 공칭값이다(참값과 무관).
    3. **갱신 주기.** 기하는 `geom_period_ms`마다만 갱신된다. 로봇 고유 감각은 매 틱이다.
       파지 중인 물체의 자세는 말단 자세에서 채운다(고유 감각, 기하 나이 0).
    4. **사건.** 물체 이동 사건은 두 기하 갱신 사이의 **관측된 변위**에서만 만든다.
       시뮬레이터의 외란 발생 사건(`evidence_only_events`)은 상태에 넣지 않고 근거로만 남긴다.

    에피소드마다 하나를 만든다 — 마지막으로 본 자세를 기억하기 때문이다.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = copy.deepcopy(config)
        self.geom_period_ms = int(config["geom_period_ms"])
        self.visible_ratio_threshold = float(config["visible_ratio_threshold"])
        self.precision_mm = dict(config["precision_mm"])
        self.surface_conf = dict(config["surface_conf"])
        self.points_visible = int(config["points_visible"])
        self.points_occluded = int(config["points_occluded"])
        self.moving_window_ms = int(config["moving_window_ms"])
        self.moved_threshold_mm = float(config["moved_threshold_mm"])
        if self.moved_threshold_mm < float(self.precision_mm["visible"]):
            raise ValueError(
                "moved_threshold_mm는 자세 정밀도 이상이어야 한다: "
                f"{self.moved_threshold_mm} < precision_mm.visible {self.precision_mm['visible']}"
            )
        self.evidence_only_events = tuple(str(kind) for kind in config.get("evidence_only_events") or ())
        self.faces_by_class = {key: tuple(value) for key, value in config["graspable_faces"].items()}
        self.max_graspable_width_mm = float(config["max_graspable_width_mm"])

        self._tracks: dict[str, dict[str, Any]] = {}
        self._geom_ms: int | None = None
        self._sim_ms: int = 0
        self._moved_ms: dict[str, int] = {}
        self._hidden: dict[str, list[int]] = {}
        self._sim_events: list[dict[str, Any]] = []
        self._surface_mm: int | None = None

    # -- 재구성 -------------------------------------------------------------

    def reconstruct(self, observation: dict[str, Any]) -> Reconstruction:
        """관측 하나를 재구성 결과로 옮긴다. 가려진 물체의 참값은 버린다."""
        sim_ms = int(observation["sim_time_ms"])
        self._sim_ms = sim_ms
        fresh = self._geom_ms is None or sim_ms - self._geom_ms >= self.geom_period_ms
        if fresh:
            self._geom_ms = sim_ms

        # 시뮬레이터만 아는 사건은 근거로 가고, 나머지(실행기 사건·지시 변경·접촉)는 그대로 지난다.
        events: list[dict[str, Any]] = []
        self._sim_events = []
        for event in observation.get("events") or ():
            if str(event.get("kind", "")) in self.evidence_only_events:
                self._sim_events.append(copy.deepcopy(event))
            else:
                events.append(copy.deepcopy(event))

        robot = observation["robot"]
        holding = robot.get("holding")
        ee = [float(value) for value in robot["ee_pos_mm"]]
        self._hidden = {}
        instances = []
        moved: list[dict[str, Any]] = []
        for entry in observation["objects"]:
            instance = self._track(
                entry, sim_ms, fresh=fresh, held=(holding == str(entry["id"])), ee=ee, moved=moved
            )
            if instance is not None:
                instances.append(instance)
        events.extend(moved)

        return Reconstruction(
            instances=tuple(instances),
            scene=self._scene(observation, instances, holding),
            zones=tuple(copy.deepcopy(zone) for zone in observation.get("zones") or ()),
            goal=self._goal(observation, instances),
            events=tuple(events),
            geom_ms=int(self._geom_ms or 0),
            tick=int(observation.get("tick", 0)),
            sim_ms=sim_ms,
            source="ground-truth",
        )

    def _track(
        self,
        entry: dict[str, Any],
        sim_ms: int,
        *,
        fresh: bool,
        held: bool,
        ee: list[float],
        moved: list[dict[str, Any]],
    ) -> TrackedInstance | None:
        """물체 하나의 추적 상태를 갱신한다. 한 번도 본 적 없으면 `None`.

        자세·크기·속성·파지면은 **보이는 동안 기하가 갱신된 틱에만** 새로 쓴다. 그래서 본 적
        없는 물체는 앞단이 아무것도 모르고(상태에서 빠지고), 가려진 물체는 마지막으로 본
        것에 머문다. 추적 상태의 `visible`은 "마지막 기하 갱신에서 보였는가"이며 갱신 사이의
        틱은 그것을 바꾸지 않는다 — 그래야 재식별이 다음 갱신 틱에 난다. 지금의 참값은
        :meth:`evidence`로만 나간다.

        파지 중인 물체는 말단 자세에서 채운다: 파지가 시작된 틱의 (마지막 관측 자세 − 말단)
        오프셋을 붙들고 매 틱 말단에 더한다. 들고 움직인 것은 이동 사건이 아니다.
        """
        object_id = str(entry["id"])
        visible = bool(entry.get("visible", True))
        ratio = float(entry.get("visible_ratio", 1.0))
        known = self._tracks.get(object_id)

        if not visible:
            # 가려진 물체의 지금 참값은 근거로만 남기고 상태로는 보내지 않는다.
            self._hidden[object_id] = [int(value) for value in entry["pos_mm"]]

        if known is None and not (visible and fresh):
            return None  # 앞단이 모르는 물체다. 참값으로 채우지 않는다

        reid: tuple[str, ...] = ()
        source = "geom"
        if held and known is not None:
            offset = known.get("held_offset")
            if offset is None:
                offset = [float(value) - e for value, e in zip(known["pose_mm"], ee)]
            pose = tuple(int(round(e + o)) for e, o in zip(ee, offset))
            quat = tuple(known["quat"])
            last_seen = sim_ms
            source = "proprio"
            appearance = known["appearance"]
            track = {**known, "pose_mm": pose, "last_seen_ms": last_seen, "held_offset": offset}
        elif visible and fresh:
            pose = tuple(int(value) for value in entry["pos_mm"])
            quat = tuple(float(value) for value in entry["quat"])
            last_seen = min(int(entry.get("last_seen_ms", sim_ms)), sim_ms)
            appearance = self._appearance(entry, ratio)
            if known is not None:
                if not known["visible"]:
                    reid = (f"reacquired:{sim_ms}",)
                displacement = math.dist(pose, known["pose_mm"])
                if known.get("held_offset") is None and displacement >= self.moved_threshold_mm:
                    self._moved_ms[object_id] = sim_ms
                    moved.append(
                        {
                            "kind": "object_moved",
                            "sim_ms": sim_ms,
                            "object": object_id,
                            "displacement_mm": int(round(displacement)),
                        }
                    )
            track = {
                "pose_mm": pose,
                "quat": quat,
                "last_seen_ms": last_seen,
                "visible": True,
                "appearance": appearance,
                "held_offset": None,
            }
        else:
            pose = tuple(known["pose_mm"])
            quat = tuple(known["quat"])
            last_seen = int(known["last_seen_ms"])
            appearance = known["appearance"]
            track = {**known, "held_offset": None}
            if fresh:
                track["visible"] = False
        self._tracks[object_id] = track

        obb = appearance["obb_mm"]
        state = "occluded" if (not visible and source == "geom") else "visible"
        return TrackedInstance(
            track_id=object_id,
            desc=appearance["desc"],
            cls=appearance["cls"],
            pose_mm=pose,
            quat=quat,
            precision_mm=int(self.precision_mm[state]),
            obb_mm=obb,
            top_mm=int(pose[2]) + obb[2] // 2,
            graspable_faces=appearance["faces"],
            surface_conf=float(self.surface_conf[state]),
            visible_ratio=ratio,
            points=self.points_visible if visible else self.points_occluded,
            last_seen_ms=last_seen,
            observed_now=visible or source == "proprio",
            reid=reid,
            attributes=appearance["attributes"],
            moving=self.moving(object_id),
            pose_source=source,
        )

    def _appearance(self, entry: dict[str, Any], ratio: float) -> dict[str, Any]:
        """관측된 틱에 앞단이 읽는 외양. 가려진 동안은 이 사본이 그대로 남는다."""
        obb = tuple(int(value) for value in entry["obb_mm"])
        return {
            "desc": self._describe(entry),
            "cls": str(entry.get("class") or entry.get("shape") or "object"),
            "obb_mm": obb,
            "faces": self._faces(entry, obb, ratio),
            "attributes": tuple(entry.get("attributes") or ()),
        }

    def _faces(self, entry: dict[str, Any], obb: tuple[int, int, int], ratio: float) -> tuple[str, ...]:
        """파지 가능 면. 표면을 못 본 물체의 윗면은 파지면으로 내지 않는다.

        v0에서는 두 면 모두 **수평 폭**으로 판정한다 — 평행 그리퍼가 닫히는 방향이
        어느 면에서든 수평이기 때문이다. 관측된 틱의 값이며 가려진 동안은 유지된다
        (가시 비율은 실행 가능성의 기준이 아니다, docs/08 §3.2).
        """
        faces = self.faces_by_class.get(str(entry.get("class") or entry.get("shape")), ())
        if min(obb[0], obb[1]) > self.max_graspable_width_mm:
            return ()
        if ratio < self.visible_ratio_threshold:
            return tuple(face for face in faces if face != "top")
        return faces

    def _describe(self, entry: dict[str, Any]) -> str:
        """물체 설명. 장면이 준 설명이 있으면 그것을 쓴다 — 지시문이 부르는 이름이다."""
        if entry.get("desc"):
            return str(entry["desc"])
        colour = entry.get("colour")
        cls = entry.get("class") or entry.get("shape") or "물체"
        return f"{colour} {cls}" if colour else str(cls)

    def _scene(
        self, observation: dict[str, Any], instances: list[TrackedInstance], holding: str | None
    ) -> SceneSummary:
        """자유 공간 요약. 관측된 자세에서만 계산한다. 들고 있는 물체는 작업면을 말하지 않는다."""
        resting = [inst for inst in instances if inst.track_id != holding]
        if resting:
            self._surface_mm = min(int(inst.pose_mm[2]) - inst.obb_mm[2] // 2 for inst in resting)
        elif self._surface_mm is None and instances:
            # 아는 물체가 들고 있는 것뿐이면 그 첫 관측(아직 놓여 있던 자세)의 바닥이 작업면이다.
            self._surface_mm = min(int(inst.pose_mm[2]) - inst.obb_mm[2] // 2 for inst in instances)
        surface = self._surface_mm if self._surface_mm is not None else 0
        gaps = [
            math.dist(a.pose_mm, b.pose_mm) - a.radius_mm - b.radius_mm
            for index, a in enumerate(instances)
            for b in instances[index + 1 :]
        ]
        clearance = min(gaps) if gaps else 0.0
        zones = observation.get("zones") or ()
        widths = [abs(zone["bounds_mm"][2] - zone["bounds_mm"][0]) for zone in zones]
        free_width = max(widths) if widths else 0
        return SceneSummary(
            work_surface_mm=int(surface),
            free_width_mm=int(free_width),
            corridor_mm=int(round(max(0.0, clearance))),
            clearance_mm=int(round(clearance)),
        )

    def _goal(self, observation: dict[str, Any], instances: list[TrackedInstance]) -> dict[str, Any]:
        """지시와 구조화된 제약 (docs/08 §3.2 `goal`).

        구조화된 목표(`observation["goal"]`: `target_ref`·`target_desc`·`zone_ref`·
        `forbidden_refs`·`fragile_refs`·`version`·`text`)가 오면 그것을 쓴다 — 실제 경로에서는
        상위 작업 지능(L3)이나 장면 명세가 대상·목적지·금지를 넘긴다. 참조는 **추적 중인 물체로
        제한한다**: 아직 본 적 없는 id는 상태에 없으므로 가리킬 수 없고, `target_desc`와 텍스트가
        그것을 나르다가 물체가 보이면 채워진다. 상태의 `goal`에 `target_desc`가 있으면 구조화된
        경로다(판단기·전문가가 이것으로 두 경로를 가른다).

        없으면 D1 어댑터가 관측 가능한 정보로 푼다: 추적 중인 물체 가운데 금지 속성이 붙은 것이
        금지 접촉이고, 목표 영역은 지시문이 부르는 영역이며, 대상은 지시문이 **마지막으로 부른
        평범한 물체**다(지시가 바뀌면 새 대상이 뒤에 온다: "A 대신 B를 먼저 옮겨라").
        """
        instruction = observation.get("instruction") or {}
        text = str(instruction.get("text", ""))
        zones = observation.get("zones") or ()
        given = dict(observation.get("goal") or {})

        tracked = [
            {"id": inst.track_id, "desc": inst.desc, "attributes": list(inst.attributes)}
            for inst in instances
        ]
        tracked_ids = {entry["id"] for entry in tracked}
        forbidden = [entry["id"] for entry in tracked if "forbidden" in entry["attributes"]]
        fragile = [entry["id"] for entry in tracked if "fragile" in entry["attributes"]]

        if "target_desc" in given or "zone_ref" in given:
            target = given.get("target_ref")
            return {
                "text": str(given.get("text", text)),
                "version": int(given.get("version", instruction.get("version", 1))),
                "t_ms": int(instruction.get("t_ms", 0)),
                "target_ref": str(target) if target is not None and str(target) in tracked_ids else None,
                "target_desc": given.get("target_desc"),
                "target_zone": given.get("zone_ref"),
                "forbidden_contact": _tracked_only(given.get("forbidden_refs"), tracked_ids, forbidden),
                "fragile": _tracked_only(given.get("fragile_refs"), tracked_ids, fragile),
            }

        goal = {
            "text": text,
            "version": int(instruction.get("version", 1)),
            "t_ms": int(instruction.get("t_ms", 0)),
            "target_ref": given.get("target_ref") or named_target(tracked, text),
            "target_zone": given.get("target_zone")
            or next((zone["id"] for zone in zones if str(zone.get("desc", "")) in text), None),
            "forbidden_contact": given.get("forbidden_contact") or forbidden,
            "fragile": given.get("fragile") or fragile,
        }
        if "priority" in given:
            goal["priority"] = given["priority"]
        return goal

    # -- 로봇 고유 감각 -----------------------------------------------------

    def robot(
        self, observation: dict[str, Any], commitment: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """관측의 로봇 부분을 `extract`가 받는 형태로 옮긴다."""
        robot = observation["robot"]
        return {
            "ee_pose_mm": list(robot["ee_pos_mm"]),
            "ee_quat": list(robot["ee_quat"]),
            "gripper_mm": robot["gripper_mm"],
            "holding": robot.get("holding"),
            "contact_n": robot.get("contact_force_n", 0.0),
            "speed_mm_s": robot.get("speed_mm_s", 0),
            "observed_at_ms": int(observation["sim_time_ms"]),
            "exec": copy.deepcopy(observation.get("exec") or {}),
            "commitment": copy.deepcopy(commitment),
        }

    # -- 근거 ---------------------------------------------------------------

    def evidence(self) -> dict[str, Any]:
        """마지막 재구성에서 **버린** 참값과 시뮬레이터 사건. 레코드의 `evidence` 자리에만 쓴다."""
        return {
            "occluded_true_poses": copy.deepcopy(self._hidden),
            "simulator_events": copy.deepcopy(self._sim_events),
            "source": "ground-truth",
        }

    def moving(self, object_id: str) -> bool:
        """관측된 이동이 `moving_window_ms` 안에 있는 대상인가 (기하 나이 허용치를 가른다)."""
        moved_ms = self._moved_ms.get(object_id)
        return moved_ms is not None and self._sim_ms - moved_ms <= self.moving_window_ms
