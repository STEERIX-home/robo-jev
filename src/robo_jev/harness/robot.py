"""로봇 스트림 하네스 v0 — 요청 구성과 조합 규칙 (docs/08 §3~§6, docs/02 §3~§4).

하네스가 하는 일은 둘이다.

* :meth:`RobotHarness.build_request` — 관측 하나를 **틱 요청**으로 만든다. 상태는
  :mod:`robo_jev.perception.pointworld`의 추출 인터페이스가 채우고, 하네스는 거기에
  결합 행동 후보(기능×대상×접근×목적지), 관측·보류·재계획 후보, 국소 플래너가 만든
  경유점 후보, 실행 이력을 붙인다. **정답을 알고 후보를 줄이거나 정렬하지 않는다**
  (docs/02 §2): 걸러내는 것은 "실행 가능한가"뿐이고, 상한을 넘으면 지시(입력)의 대상×목표
  영역 조합을 먼저 남긴 뒤 기하적 다양성 표본으로 줄이면서 **후보 회계**를 함께 기록한다.
* :meth:`RobotHarness.compose` — 모델(또는 규칙 기준군)의 답에 조합 규칙 v0을 순서대로
  적용해 명령·채택 결과·commitment·전환 기록을 낸다.

**id는 의미 키에서 나온다.** 후보 id는 `기능:대상:접근:목적지`의 해시다(계약 v0.3, docs/08
§5-3 — 프로파일은 키에 없다; 속도는 `q_speed`가 commitment 기준으로 답하고 전환 틱의 초기값은
`compose.initial_profile`이다). 그래서 틱마다 후보 목록이 바뀌어도 같은 행동은 같은 id를 갖고,
"만료된 후보를 최신 목록의 같은 인덱스로 해석"하는 사고(docs/02 §6)가 구조적으로 불가능하다.

**후보 줄은 키가 설명이다.** 모델이 보는 후보 항목은 `id`·`key`와 기하 네 값(`d` 거리, `clr` 여유,
`path` ok/blocked, `g` 대상 기하 나이)뿐이다(docs/08 §3.2 서식 v0.3). 자연어 설명·프로파일 표지는 없다.

**commitment는 두 겹이다.** 하네스가 들고 있는 commitment에는 도전자 카운터·정지 틱 수
같은 장부가 붙어 있고, 요청에 싣는 것은 모델이 볼 필드(`action_ref`·의미 키·국면·유지
틱 수·마지막 전환 틱)뿐이다(docs/08 §3.2).

수치는 전부 설정에 있다. 실행기와 공유하는 값(수명·속도·힘·그리퍼·도달 범위)은
`configs/controller/osc_v0.yaml`이 단일 출처이고 여기서 그것을 읽는다.
"""

from __future__ import annotations

import copy
import hashlib
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from robo_jev.perception.pointworld import (
    EXTRACTOR_VERSION,
    GroundTruthAdapter,
    circumradius_mm,
    extract,
    segment_point_distance_mm,
)
from robo_jev.sim.controller import load_controller_config, resolve_config_path

__all__ = [
    "CONTACT_PHASES",
    "DEFAULT_CONFIG_PATH",
    "FIXED_KEYS",
    "GATE_QUESTIONS",
    "HARNESS_VERSION",
    "JOINT_FUNCTIONS",
    "RobotHarness",
    "build_request",
    "candidate_id",
    "compose",
    "count_records",
    "joint_key_parts",
    "load_harness_config",
    "parse_exec_history",
    "push_contact_class",
    "push_contact_offset_mm",
    "push_standoff_slack_mm",
    "push_directions_toward_zones",
]

#: 하네스 버전. 질문 세트·후보 형식·조합 규칙의 묶음을 가리킨다 (docs/08 §3.1). h0.4 = 계약 v0.3(결합 키에서
#: 프로파일 제거, 영역 방향 밀기, K≤12와 지시 조합 예약, 키 기반 후보 줄). h0.5 = 놓기점이 영역 안의 빈 자리(관측된
#: 바닥 높이), 명령의 `place_mm`, hold·retreat 경로의 놓기 틱에는 open 없음(리뷰 2 C2). h0.6 = 축·손몸통별 밀기 접촉 거리,
#: 놓기 정체 감시(`place_stalled`), 빈 자리 없는 영역의 후보는 `path=blocked`(D1-prep 리뷰 1). h0.7 = 밀기 명령 구간을
#: stand-off 여유(접촉 거리 − 실측 도달, `push_reach_mm`)만큼 늘림, 놓기 정체 감시의 절대 상한(`m_place_total`), 경로 답과
#: 무관한 `conflict{zone_full}` 기록(D1-prep 리뷰 2 N7·N2·N4).
HARNESS_VERSION = "h0.8"

#: 결합 행동의 기능. 이 셋만 `기능:대상:접근:목적지` 키를 갖는다.
JOINT_FUNCTIONS = ("grasp", "place", "push")

DEFAULT_CONFIG_PATH = "configs/harness/robot.yaml"

#: 게이팅 질문과 그 순서 (docs/08 §5.2). 앞의 것이 먼저 발동한다.
GATE_QUESTIONS = ("q_done", "q_instr", "q_observe")

#: 후보 없이 언제나 제시하는 세 가지 (docs/08 §4). 규칙 기준군도 이 이름을 여기서 읽는다.
FIXED_KEYS = ("observe", "hold", "replan")

#: 팔이 대상을 가리는 국면. 기하 나이 대신 실행기의 readiness가 시점을 정한다 (docs/08 §5.0).
CONTACT_PHASES = ("grasp", "place", "push")

#: 밀기 방향 벡터 (로봇 기준 좌표계, xy 평면).
_PUSH_VECTORS = {"+x": (1.0, 0.0), "-x": (-1.0, 0.0), "+y": (0.0, 1.0), "-y": (0.0, -1.0)}

#: 상한 안에서 돌아가며 고를 때 보는 차원과 그 순서 (docs/10 I3). 앞의 차원이 먼저 고르게 퍼진다.
_SPREAD_DIMENSIONS = ("target_ref", "function", "approach", "destination")


def load_harness_config(path: str | Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    return yaml.safe_load(resolve_config_path(path).read_text(encoding="utf-8"))


def candidate_id(key: str) -> str:
    """의미 키 → 후보 id. 같은 의미의 행동은 언제나 같은 id다."""
    return "c" + hashlib.sha1(key.encode("utf-8")).hexdigest()[:6]


def joint_key_parts(key: Any) -> tuple[str, str, str, str] | None:
    """결합 키 `기능:대상:접근:목적지` → (기능, 대상, 접근, 목적지). 결합 후보가 아니면 `None`.

    전문가·라벨·규칙 기준군이 같은 함수로 키를 읽는다. 다른 도구가 만든 틱(D0 fixture)의 옛 5조각 키
    `…:프로파일`은 앞 네 조각으로 읽는다 — 프로파일은 계약 v0.3부터 키의 일부가 아니다.
    """
    parts = str(key or "").split(":")
    if len(parts) < 4 or parts[0] not in JOINT_FUNCTIONS:
        return None
    return parts[0], parts[1], parts[2], parts[3]


def push_directions_toward_zones(
    pose_mm, zones: list[dict[str, Any]], allowed: list[str]
) -> dict[str, list[str]]:
    """물체 자세에서 각 영역 쪽으로 미는 축 방향 → 그 방향이 향하는 영역 id들 (계약 v0.3, docs/08 §4).

    영역마다 **영역 중심까지의 거리를 가장 줄이는 축**(성분이 큰 축; 같으면 `allowed`의 앞 축; 허용 집합 밖이면
    다른 축) 하나를 고르고 영역들에 걸쳐 합친다 — 물체당 방향은 영역 수 이하다. 관측된 자세와 영역 경계만 쓴다.
    """
    x, y = float(pose_mm[0]), float(pose_mm[1])
    found: dict[str, list[str]] = {}
    for zone in zones:
        cx, cy = _zone_centre(zone)
        dx, dy = cx - x, cy - y
        options = [(abs(dx), "+x" if dx > 0 else "-x"), (abs(dy), "+y" if dy > 0 else "-y")]
        options.sort(key=lambda item: (-item[0], allowed.index(item[1]) if item[1] in allowed else len(allowed)))
        for magnitude, direction in options:
            if magnitude > 0.0 and direction in allowed:
                found.setdefault(direction, []).append(str(zone["id"]))
                break
    return {direction: found[direction] for direction in allowed if direction in found}


def push_contact_offset_mm(spec: dict[str, Any], direction: str, obb_mm, top_mm: float, push_z: float) -> float:
    """밀기 접촉점이 물체 표면 밖으로 떨어지는 거리 (h0.6): 닫힌 그리퍼가 접근 축으로 뻗은 길이 + 여유.

    `candidates.push_contact_mm`이 `{fingers: {…}, hand: {…}}`이면 물체 윗면이 손몸통 바닥(`push_z` + `push_hand_bottom_mm`)보다
    `push_hand_overlap_mm`보다 더 높을 때 손몸통의 값을, 아니면 손가락의 값을 쓴다(방향 키가 있으면 그것, 없으면 축 키) — 겹침이
    작은 물체는 손가락이 밀어도 충돌이 드물고 잘 미끄러진다(실측, configs/harness/robot.yaml). 옛 설정(하나의 수)은 축과 무관하게
    그대로다. 전문가·rollout 후보 선택도 같은 함수로 접촉점을 계산한다.
    """
    contact = spec["push_contact_mm"]
    if not isinstance(contact, dict):
        return float(contact)
    return _push_table_value(contact, "push_contact_mm", push_contact_class(spec, top_mm, push_z), direction)


def push_standoff_slack_mm(spec: dict[str, Any], direction: str, obb_mm, top_mm: float, push_z: float) -> float:
    """밀기의 stand-off 여유 (h0.7, D1-prep 리뷰 2 N7): 접촉 거리(`push_contact_mm`) − 그 축으로 닫힌 그리퍼가 **실제로** 뻗은 길이
    (`push_reach_mm`, 실측 `Environment.gripper_extent_mm`). 명령한 밀기 구간은 접촉점에서 시작하므로 손가락·손몸통은 이만큼 간 뒤에야
    물체에 닿고, 물체는 구간에서 이만큼 덜 간다(y 손가락 48 − 30 = 18, 손몸통 111 − 106 = 5, x 손가락 33 − 12 = 21·52 − 17 = 35).
    `_geometry_for`는 명령 구간을 `push_segment_mm + slack`으로 늘린다 — 성공 기준(사건의 `segment_fraction × push_segment_mm`)과
    완료 판정(변위 ≥ `push_segment_mm`)은 절대값 그대로다. 도달 표가 없는 설정(h0.6 이전, 스칼라 접촉 거리)은 0이다.
    """
    contact, reach = spec["push_contact_mm"], spec.get("push_reach_mm")
    if not isinstance(contact, dict) or not isinstance(reach, dict):
        return 0.0
    klass = push_contact_class(spec, top_mm, push_z)
    return max(0.0, _push_table_value(contact, "push_contact_mm", klass, direction) - _push_table_value(reach, "push_reach_mm", klass, direction))


def push_contact_class(spec: dict[str, Any], top_mm: float, push_z: float) -> str:
    """밀기의 접촉 부위 — 무엇이 먼저 닿는가: 물체 윗면이 손몸통 바닥(`push_z` + `push_hand_bottom_mm`)보다 `push_hand_overlap_mm` 넘게 높으면 `hand`, 아니면 `fingers`."""
    hand_bottom = float(push_z) + float(spec.get("push_hand_bottom_mm", 0.0))
    overlap = float(top_mm) - hand_bottom
    return "hand" if overlap > float(spec.get("push_hand_overlap_mm", 0.0)) else "fingers"


def _push_table_value(table: dict[str, Any], name: str, klass: str, direction: str) -> float:
    axis = direction[-1]
    row = table[klass]
    value = row.get(direction, row.get(axis))
    if value is None:
        raise ValueError(f"{name}에 {direction!r}(축 {axis!r})의 값이 없다: {table}")
    return float(value)


def parse_exec_history(text: Any) -> dict[str, str]:
    """실행 이력 한 줄(`main=c1 phase=approach … ack=ok`)을 필드로 푼다.

    이 줄을 쓰는 쪽(:meth:`RobotHarness._exec_history_text`)과 읽는 쪽(조합 규칙의 재시도
    조건, 규칙 기준군의 실패 감점)이 같은 형식을 봐야 하므로 파서를 한 군데 둔다.
    """
    if not isinstance(text, str) or text in ("", "none"):
        return {}
    fields: dict[str, str] = {}
    for token in text.split():
        if "=" in token:
            key, value = token.split("=", 1)
            fields[key] = value
    return fields


def count_records(records: list[dict[str, Any]]) -> dict[str, int]:
    """조합 기록을 종류별로 센다 (docs/08 §5 "충돌 건수를 기록한다")."""
    counts: dict[str, int] = {}
    for record in records:
        counts[record["kind"]] = counts.get(record["kind"], 0) + 1
    return counts


# --------------------------------------------------------------------------
# 후보
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class _Candidate:
    """결합 행동 후보 하나. 모델이 보는 부분과 하네스만 쓰는 기하를 함께 들고 있다."""

    key: str
    function: str | None
    target_ref: str | None
    approach: str | None
    destination: str | None
    approach_mm: list[float] = field(default_factory=list)
    action_mm: list[float] = field(default_factory=list)
    #: 이 후보를 지금 실행하면 **실제로 명령할** 목표점 (국면별, docs/08 §5.6).
    target_mm: list[float] = field(default_factory=list)
    distance_mm: int = 0
    clearance_mm: int = 0
    reach_ok: bool = True
    path_clear: bool = True
    blocker: str | None = None
    geometry_age_ms: int = 0
    moving: bool = False
    phase: str = "none"
    #: 밀기 후보가 향하는 영역 id들 (영역 방향 열거의 근거; 지시 조합 예약이 읽는다).
    toward_zones: tuple[str, ...] = ()
    #: 놓기 후보의 자리 기록 (`_place_points`): `{"free", "offset_mm", "point"}`. 놓기 후보가 아니면 `None`.
    place: dict[str, Any] | None = None

    @property
    def id(self) -> str:
        return candidate_id(self.key)

    def model_entry(self) -> dict[str, Any]:
        """모델이 보는 후보 (docs/08 §3.2 서식 v0.3, §8의 틱 예시).

        결합 후보는 키와 기하 네 값(`d` 말단→국면 목표점 거리 mm, `clr` 최근접 여유 mm, `path` ok/blocked, `g` 대상
        기하 나이 ms)이고, 고정 후보(observe·hold·replan)는 키뿐이다. `action_ref`는 id와 같다(계약의 참조 필드).
        """
        entry = {"id": self.id, "action_ref": self.id, "key": self.key}
        if self.function is not None:
            entry.update(
                {
                    "d": int(self.distance_mm),
                    "clr": int(self.clearance_mm),
                    "path": "ok" if self.path_clear else "blocked",
                    "g": int(self.geometry_age_ms),
                }
            )
        return entry

    def geometry(self) -> dict[str, Any]:
        geometry = {
            "key": self.key,
            "function": self.function,
            "target_ref": self.target_ref,
            "approach": self.approach,
            "destination": self.destination,
            "approach_mm": [int(round(value)) for value in self.approach_mm],
            "action_mm": [int(round(value)) for value in self.action_mm],
            "target_mm": [int(round(value)) for value in self.target_mm],
            "distance_mm": self.distance_mm,
            "clearance_mm": self.clearance_mm,
            "reach_ok": self.reach_ok,
            "path_clear": self.path_clear,
            "blocker": self.blocker,
            "geometry_age_ms": self.geometry_age_ms,
            "moving": self.moving,
            "phase": self.phase,
            "toward_zones": list(self.toward_zones),
        }
        if self.place is not None:
            # 놓기점은 행동점이다. 명령이 `place_mm`으로 따로 나른다 (docs/08 §6; 실행기의 open readiness 기준).
            geometry["place_mm"] = geometry["action_mm"]
            geometry["place"] = dict(self.place)
        return geometry


# --------------------------------------------------------------------------
# 하네스
# --------------------------------------------------------------------------


class RobotHarness:
    """한 에피소드의 하네스. 앞단(추적 기억)을 들고 있으므로 에피소드마다 하나를 만든다."""

    def __init__(
        self,
        config: dict[str, Any],
        controller_config: dict[str, Any] | None = None,
        adapter: Any | None = None,
    ) -> None:
        self.config = copy.deepcopy(config)
        self.version = str(config.get("version", HARNESS_VERSION))
        self.language = str(config.get("language", "ko"))

        controller = controller_config or load_controller_config(config["controller_config"])
        self.controller_config = copy.deepcopy(controller)
        self.controller_version = str(controller.get("version", "c0"))
        lifetime = controller["lifetime"]
        self.observation_deadline_ms = int(lifetime["observation_deadline_ms"])
        self.lease_ms = int(lifetime["lease_ms"])
        self.geometry_age_static_ms = int(lifetime["geometry_age_static_ms"])
        self.geometry_age_moving_ms = int(lifetime["geometry_age_moving_ms"])
        self.speed_levels_m_s = list(controller["speed_levels_m_s"])
        self.force_level_names = list(controller["force_levels"])
        self.gripper_open_mm = float(controller["gripper"]["open_mm"])
        self.gripper_closed_mm = float(controller["gripper"]["closed_mm"])
        reach = controller["reach"]
        self.workspace_radius_mm = float(reach["workspace_radius_mm"])
        self.min_height_mm = float(reach["min_height_mm"])
        self.reach_clearance_mm = float(reach["clearance_mm"])

        self.candidates_config = config["candidates"]
        self.planner_config = config["planner"]
        self.compose_config = config["compose"]
        self.phases_config = config["phases"]
        self.command_config = config["command"]

        self.adapter = adapter if adapter is not None else GroundTruthAdapter(config["perception"])
        #: 같은 방식의 연속 실패 횟수 (실행 이력의 `fails=`). 에피소드 안에서만 센다.
        self._failure_streak: dict[str, Any] | None = None
        #: 놓기 정체(`place_stalled`)로 이 에피소드 동안 제외한 놓기점 (영역 id → xy 목록). 에피소드 안에서만 든다.
        self._stalled_place_points: dict[str, list[tuple[float, float]]] = {}
        self._reset_stall_ledger()

    @classmethod
    def from_config_path(
        cls, path: str | Path = DEFAULT_CONFIG_PATH, adapter: Any | None = None
    ) -> RobotHarness:
        return cls(load_harness_config(path), adapter=adapter)

    def reset(self) -> None:
        """새 에피소드. 앞단의 추적 기억과 실패 횟수를 버린다."""
        self.adapter = GroundTruthAdapter(self.config["perception"])
        self._failure_streak = None
        self._stalled_place_points = {}
        self._reset_stall_ledger()

    def _reset_stall_ledger(self) -> None:
        """정체 장부 (h0.8, Task R1 B2-v). 에피소드 안에서만 든다 — 놓기 정체 장부와 같은 수명이다."""
        #: 기하가 젊어지지 않은 채 이어진 관측 게이트 틱 수 (진행 기준, `m_observe`).
        self._observe_ticks = 0
        #: 진행과 무관하게 이어진 관측 게이트 틱 수 (절대 상한, `m_observe_total`) — `m_place_total`과 대칭.
        self._observe_total = 0
        #: 그 구간에서 본 대상 기하의 가장 젊은 나이. 이보다 젊어지면 진행이다.
        self._observe_age: int | None = None
        #: 이어진 hold·정지 틱 수 (`m_hold`). 완료 꼬리(`gate=done`)는 세지 않는다.
        self._hold_ticks = 0

    def _stall_reason(self) -> dict[str, Any] | None:
        """상한에 이른 정체가 있으면 그 기록, 없으면 `None`. 기록을 내면 장부를 비운다."""
        caps = self.compose_config
        limit_observe = int(caps.get("m_observe", 0) or 10**9)
        limit_total = int(caps.get("m_observe_total", 0) or 10**9)
        if self._observe_ticks >= limit_observe or self._observe_total >= limit_total:
            reason = {
                "kind": "observe_stalled",
                "ticks": int(self._observe_ticks),
                "total_ticks": int(self._observe_total),
                "cap": "progress" if self._observe_ticks >= limit_observe else "total",
            }
            self._reset_stall_ledger()
            return reason
        if self._hold_ticks >= int(caps.get("m_hold", 0) or 10**9):
            reason = {"kind": "hold_stalled", "ticks": int(self._hold_ticks)}
            self._reset_stall_ledger()
            return reason
        return None

    def _note_stall_outcome(self, out: dict[str, Any], request: dict[str, Any]) -> None:
        """이 틱의 결과로 정체 장부를 갱신한다 (:meth:`compose`가 부른다)."""
        if out.get("command") is None:
            return  # 폐기된 요청은 이 에피소드의 진행이 아니다
        gate = out.get("gate")
        adopted = out.get("adopted") or {}
        if gate == "observe":
            self._observe_total += 1
            age = self._target_geometry_age((request.get("request", request)).get("state") or {})
            if age is not None and (self._observe_age is None or age < self._observe_age):
                self._observe_ticks = 0  # 진행: 대상 기하가 실제로 젊어졌다
            else:
                self._observe_ticks += 1
            self._observe_age = age if age is not None else self._observe_age
        else:
            self._observe_ticks = self._observe_total = 0
            self._observe_age = None
        if gate != "done" and str(adopted.get("path_kind")) == "hold":
            self._hold_ticks += 1
        else:
            self._hold_ticks = 0

    @staticmethod
    def _target_geometry_age(state: dict[str, Any]) -> int | None:
        """지시 대상의 기하 나이(ms). 대상이 없거나 상태에 없으면 `None`."""
        target = str((state.get("goal") or {}).get("target_ref") or "")
        entry = next((item for item in state.get("objects") or () if str(item.get("id")) == target), None)
        return int(entry.get("age_ms", 0)) if isinstance(entry, dict) else None

    # ------------------------------------------------------------------
    # 질문 세트 (docs/08 §4)
    # ------------------------------------------------------------------

    def question_texts(self) -> dict[str, str]:
        return {
            question_id: str(texts[self.language])
            for question_id, texts in self.config["questions"].items()
        }

    def question_set_id(self) -> str:
        return str(self.config["question_set_id"][self.language])

    # ------------------------------------------------------------------
    # 요청 구성
    # ------------------------------------------------------------------

    def build_request(
        self,
        observation: dict[str, Any],
        exec_history: dict[str, Any] | None = None,
        commitment: dict[str, Any] | None = None,
        *,
        keep_key: str | None = None,
    ) -> dict[str, Any]:
        """관측 하나 → 틱 요청 (docs/08 §3).

        돌려주는 dict는 계약의 틱(`t`·`sim_ms`·`observed_at_ms`·`obs_age_ms`·`request`)에
        하네스 블록(`harness`)을 더한 것이다. 하네스 블록은 기하·회계·경유점 좌표처럼
        **명령을 만들 때 쓰는 값**이며 레코드에는 들어가지 않는다
        (:func:`robo_jev.data.episode.append_tick`이 떼어낸다).

        `keep_key`는 commitment가 없을 때 commitment처럼 예약할 결합 키다 — 키프레임 rollout이 실행할 후보를
        첫 틱의 목록에 남기는 데 쓴다(자세 흔들기로 영역 쪽 축이 바뀌거나 상한에 밀려도 실행 가능하면 남는다).
        """
        now_ms = int(observation["sim_time_ms"])
        recon = self.adapter.reconstruct(observation)
        robot = self.adapter.robot(observation)
        state = extract(recon, robot, now_ms)

        if commitment:
            reserved, reserved_key = str(commitment["action_ref"]), str(commitment.get("key", ""))
        elif keep_key:
            reserved, reserved_key = candidate_id(str(keep_key)), str(keep_key)
        else:
            reserved = reserved_key = None
        candidates, accounting = self._candidates(state, reserved=reserved, reserved_key=reserved_key)
        by_id = {candidate.id: candidate for candidate in candidates}

        # commitment는 이 틱의 후보 목록 안에서만 모델에게 보일 수 있다. 사라진 행동을
        # 가리키는 commitment는 요청에서 비우고 하네스 쪽에만 남긴다 — compose가 그것을
        # "무효화"로 기록하고 해제한다.
        committed = commitment if commitment and commitment.get("action_ref") in by_id else None
        # commitment가 없으면 부가 질문은 `hold` 기준으로 묻는다 (docs/08 §4). `hold`는
        # `FIXED_KEYS`라 언제나 목록에 있다.
        reference = committed["action_ref"] if committed else candidate_id("hold")
        paths, waypoints = self._path_candidates(state, by_id.get(reference), reference)

        state["derived"].extend(
            {"waypoint": name, **values} for name, values in sorted(waypoints.items())
        )
        state["t"]["seq"] = int(state["t"]["tick"]) + 1
        state["t"]["candidate_set_version"] = accounting["candidate_set_version"]
        state["commitment"] = self._project_commitment(committed)

        request = {
            "state": state,
            "exec_history": self._exec_history_text(exec_history, candidates=by_id),
            "commitment": self._project_commitment(committed),
            "candidates": {
                "q_main": [candidate.model_entry() for candidate in candidates],
                "q_path": paths,
            },
        }
        return {
            "t": int(state["t"]["tick"]),
            "sim_ms": now_ms,
            "observed_at_ms": int(state["t"]["observed_at_ms"]),
            "obs_age_ms": dict(state["t"]["age_ms"]),
            "request": request,
            "harness": {
                "version": self.version,
                "extractor": EXTRACTOR_VERSION,
                "controller": self.controller_version,
                "request_id": f"{self.command_config['request_id_prefix']}{state['t']['tick']}",
                "seq": int(state["t"]["seq"]),
                "goal_version": int(state["goal"].get("version", 1)),
                "candidate_set_version": accounting["candidate_set_version"],
                "candidates": {
                    candidate.id: candidate.geometry() for candidate in candidates
                },
                "waypoints": waypoints,
                "accounting": accounting,
                "commitment_invalid": bool(commitment) and committed is None,
            },
        }

    # -- 결합 후보 ---------------------------------------------------------

    def _candidates(
        self, state: dict[str, Any], *, reserved: str | None = None, reserved_key: str | None = None
    ) -> tuple[list[_Candidate], dict[str, Any]]:
        """실행 가능한 결합 후보를 만들고 상한 안으로 줄인다 (docs/02 §3).

        거르는 기준은 **실행 가능성**뿐이다: 도달할 수 있고 대상 기하가 `max_geometry_age_ms`
        안이다. 가시 비율은 기준이 아니다(docs/08 §3.2). 팔이 대상을 가리는 접촉 국면(파지·놓기·
        밀기)과 들고 있는 물체는 나이 문턱을 적용하지 않는다 — 그러지 않으면 파지 도중에
        후보가 사라진다. 의미 적합성(금지 물체, 지시가 가리키는 대상)은 여기서 쓰지 않는다 —
        적합하지 않은 실행 가능 후보도 그대로 제시한다(docs/08 §7).

        `reserved`는 현재 commitment의 후보 id, `reserved_key`는 그 의미 키다. 실행 가능하면 상한과 무관하게
        남긴다 — 밀기 commitment는 대상이 밀려 영역 쪽 축이 바뀌어도 그 방향을 계속 만든다(진행 중인 행동이 목록에서
        사라져 매 틱 무효가 되지 않게). 상한을 넘을 때는
        **지시(입력)의 대상×목표 영역 조합**(파지→영역, 영역 쪽 밀기)도 먼저 남긴다(계약 v0.3, docs/08 §4) —
        상태의 `goal`만 읽고 라벨은 읽지 않으므로 적합성 규칙을 후보 생성기에 넣는 것이 아니다(적합하지 않은
        후보도 계속 제시된다). 지시가 대상을 풀지 못하면(`q_instr` 거짓인 경우) 예약은 없다.
        """
        spec = self.candidates_config
        # `unsupported_face`는 앞단이 낸 파지면 중 실행기가 못 쓰는 면(`faces` 밖)의 조합이다 —
        # 물체의 실행 가능성이 아니라 실행기 역량이며, 조용히 빠지지 않고 여기 남는다.
        dropped = {"unsupported_face": 0, "stale": 0, "unreachable": 0, "cap": 0}
        enumerated = 0
        feasible: list[_Candidate] = []

        objects = {entry["id"]: entry for entry in state["objects"]}
        zones = state["zones"]
        holding = state["robot"].get("holding")
        ee = [float(value) for value in state["robot"]["ee_pose_mm"]]
        ages = {entry["object"]: entry["age_ms"] for entry in state["derived"] if "object" in entry}
        margins = self._margins(state)

        committed_push = joint_key_parts(reserved_key) if reserved_key else None
        if committed_push is not None and committed_push[0] != "push":
            committed_push = None
        for object_id, entry in objects.items():
            age = int(ages.get(object_id, 0))
            keep_push = committed_push[2] if committed_push is not None and committed_push[1] == object_id else None
            combos, unsupported = self._combinations(entry, zones, holding, keep_push=keep_push)
            enumerated += len(combos) + unsupported
            dropped["unsupported_face"] += unsupported
            for combo, toward in combos:
                candidate = self._geometry_for(combo, entry, objects, state, ee, age, margins, toward=toward)
                if candidate is None:
                    dropped["unreachable"] += 1
                    continue
                exempt = candidate.phase in CONTACT_PHASES or holding == object_id
                if age > int(spec["max_geometry_age_ms"]) and not exempt:
                    dropped["stale"] += 1
                    continue
                feasible.append(candidate)

        goal_reserved = self._goal_reserved(feasible, state)
        kept = self._prune(feasible, objects, ee, reserved=reserved, goal_reserved=goal_reserved)
        dropped["cap"] = len(feasible) - len(kept)
        fixed = [self._fixed_candidate(key) for key in FIXED_KEYS]
        candidates = sorted(kept + fixed, key=lambda candidate: candidate.key)

        accounting = {
            "enumerated": enumerated,
            "feasible": len(feasible),
            "kept": len(kept),
            # 목록 보존 비율이다. 정답 포함률은 라벨과 대조해 오프라인으로 잰다 (docs/10 I3).
            "kept_ratio": (len(kept) / len(feasible)) if feasible else 1.0,
            "capped": len(kept) < len(feasible),
            "cap": int(spec["max"]),
            "reserved": int(spec["reserved"]),
            # 지시의 대상×목표 영역 조합의 예약 (계약 v0.3). 상한에 걸리지 않은 틱에도 어느 조합이 지시의 것인지 적는다.
            "reserved_goal": {
                "count": len(goal_reserved),
                "keys": sorted(candidate.key for candidate in goal_reserved),
                "target": state["goal"].get("target_ref") if isinstance(state.get("goal"), dict) else None,
                "zone": state["goal"].get("target_zone") if isinstance(state.get("goal"), dict) else None,
            },
            "by_function": {
                function: sum(1 for c in kept if c.function == function)
                for function in ("grasp", "place", "push")
            },
            "spread": {
                name: {
                    "kept": len({getattr(c, dimension) for c in kept}),
                    "feasible": len({getattr(c, dimension) for c in feasible}),
                }
                for name, dimension in (
                    ("targets", "target_ref"),
                    ("functions", "function"),
                    ("approaches", "approach"),
                    ("destinations", "destination"),
                )
            },
            "dropped": dropped,
            "candidate_set_version": "cs-" + hashlib.sha1(
                "|".join(candidate.key for candidate in candidates).encode("utf-8")
            ).hexdigest()[:8],
        }
        # 상한의 붕괴는 대상별이다(목적지는 돌아가며 고르는 차원 중 넷째): 전체로는 모든 목적지가 남아도
        # 한 대상의 목적지가 하나만 남을 수 있고, 그것이 지시의 대상×목표 영역이면 정답이 목록에서 빠진다.
        # 밀기의 `none`은 목적지가 아니므로 세지 않는다.
        by_target = {
            target: {
                "kept": len({c.destination for c in kept if c.target_ref == target and c.destination != "none"}),
                "feasible": len({c.destination for c in feasible if c.target_ref == target and c.destination != "none"}),
            }
            for target in sorted({str(c.target_ref) for c in feasible})
        }
        accounting["spread"]["destinations"]["by_target"] = by_target
        accounting["spread"]["destinations"]["targets_collapsed"] = sum(
            1 for entry in by_target.values() if entry["kept"] < entry["feasible"]
        )
        return candidates, accounting

    def _combinations(
        self,
        entry: dict[str, Any],
        zones: list[dict[str, Any]],
        holding: str | None,
        *,
        keep_push: str | None = None,
    ) -> tuple[list[tuple[tuple[str, str, str, str], tuple[str, ...]]], int]:
        """한 물체가 낳는 ((기능, 대상, 접근, 목적지), 밀기가 향하는 영역들) 조합과, 실행기가 못 쓰는 면의 조합 수.

        밀기는 고정 축 열거가 아니라 **영역 쪽 축 방향**만 만든다(:func:`push_directions_toward_zones`;
        설정 `push_directions`는 허용 축 집합이다). `keep_push`는 이 물체에 대한 현재 밀기 commitment의 방향 —
        영역 쪽이 아니게 됐어도(대상이 밀려서) 허용 축이면 계속 만든다.
        """
        spec = self.candidates_config
        object_id = str(entry["id"])
        combos: list[tuple[tuple[str, str, str, str], tuple[str, ...]]] = []
        unsupported = 0
        # 들고 있는 물체의 파지 후보는 **진행 중인 결합 행동**이다. 목적지가 의미 키에
        # 들어 있으므로 `grasp:o7:top:zoneL`은 "집어서 zoneL로 옮긴다" 하나이고,
        # 집은 순간에 사라지면 commitment가 매번 무효가 된다.
        if holding is None or holding == object_id:
            for face in entry["graspable_faces"]:
                if face not in spec["faces"]:
                    unsupported += len(zones)
                    continue
                for zone in zones:
                    combos.append((("grasp", object_id, face, str(zone["id"])), ()))
        if holding == object_id:
            for zone in zones:
                combos.append((("place", object_id, "release", str(zone["id"])), ()))
        elif holding is None:
            allowed = [str(direction) for direction in spec["push_directions"]]
            directions = push_directions_toward_zones(entry["pose_mm"], zones, allowed)
            if keep_push is not None and keep_push in allowed and keep_push not in directions:
                directions[keep_push] = []
            for direction, toward in directions.items():
                combos.append((("push", object_id, direction, "none"), tuple(toward)))
        return combos, unsupported

    def _goal_reserved(self, feasible: list[_Candidate], state: dict[str, Any]) -> list[_Candidate]:
        """지시의 대상×목표 영역 조합 — 상한과 무관하게 남길 후보 (계약 v0.3, docs/08 §4).

        상태의 `goal`(입력)만 읽는다: 추적 중인 대상 id(`target_ref`)와 목표 영역(`target_zone`)이 둘 다 있고
        대상이 금지 접촉 물체가 아닐 때만(아니면 지시가 풀리지 않거나 모순이라 `q_instr`이 거짓이다) 대상→영역의
        파지·놓기와 대상을 그 영역 쪽으로 미는 후보를 고른다.
        """
        goal = state.get("goal")
        if not isinstance(goal, dict):
            return []
        target, zone = goal.get("target_ref"), goal.get("target_zone")
        if target is None or zone is None:
            return []
        target, zone = str(target), str(zone)
        if target in {str(item) for item in goal.get("forbidden_contact") or ()}:
            return []
        return [
            candidate
            for candidate in feasible
            if candidate.target_ref == target
            and (
                (candidate.function in ("grasp", "place") and candidate.destination == zone)
                or (candidate.function == "push" and zone in candidate.toward_zones)
            )
        ]

    def _geometry_for(
        self,
        combo: tuple[str, str, str, str],
        entry: dict[str, Any],
        objects: dict[str, dict[str, Any]],
        state: dict[str, Any],
        ee: list[float],
        age_ms: int,
        margins: dict[str, float] | None = None,
        *,
        toward: tuple[str, ...] = (),
    ) -> _Candidate | None:
        """조합 하나의 기하. 도달 불가하거나 목적지가 없으면 `None` (실행 가능성 기준의 제거).

        국면별 목표점(docs/08 §5.6)까지 여기서 정한다. `path_clear`·`blocker`는 **그 국면에서
        실제로 명령할 구간**(말단→목표점)을 말한다 — 접근점까지의 직선이 아니다.
        """
        function, object_id, approach, destination = combo
        spec = self.candidates_config
        pose = [float(value) for value in entry["pose_mm"]]
        obb = [float(value) for value in entry["obb_mm"]]
        radius_xy = math.dist((0.0, 0.0), (obb[0] / 2.0, obb[1] / 2.0))
        holding = state["robot"].get("holding")

        place: dict[str, Any] | None = None
        if function == "place" or (function == "grasp" and holding == object_id):
            # 이미 집은 결합 행동은 남은 절반(목적지로 옮기기)이 목표다.
            points = self._place_points(state, destination, entry)
            if points is None:
                return None
            approach_mm, action_mm, place = points
        elif function == "grasp":
            top = float(entry["top_mm"])
            approach_mm = [pose[0], pose[1], top + float(spec["approach_clearance_mm"])]
            action_mm = [pose[0], pose[1], top - float(spec["grasp_depth_mm"])]
        else:  # push
            vector = _PUSH_VECTORS[approach]
            # 물체 중간 높이로 밀되 작업면 위 최소 높이는 지킨다 — 닫힌 손가락 끝이 그립 사이트 아래 4mm에 있고
            # 하강 끝의 추종 오차가 10mm쯤이라, 낮은 상자(32mm)의 중간 높이(16mm)에서는 손가락이 작업면을 친다.
            surface = float(state["scene"].get("work_surface_mm", 0.0))
            height = max(pose[2], surface + float(spec["push_height_min_mm"]))
            # 접촉 거리는 접근 축과 손몸통 겹침에 따른다 (h0.6): ±y는 닫힌 손가락이 30mm, 손몸통이 106mm 뻗어 있다.
            contact = radius_xy + push_contact_offset_mm(spec, approach, obb, float(entry["top_mm"]), height)
            approach_mm = [pose[0] - vector[0] * contact, pose[1] - vector[1] * contact, height]
            # 명령 구간은 접촉점에서 시작하므로 stand-off 여유(접촉 거리 − 실측 도달)만큼 더 간다 (h0.7) — 그래야 물체가
            # `push_segment_mm`을 간다. 성공 기준·완료 판정은 물체 변위의 절대값이라 여기 늘린 만큼 자라지 않는다.
            segment = float(spec["push_segment_mm"]) + push_standoff_slack_mm(spec, approach, obb, float(entry["top_mm"]), height)
            action_mm = [approach_mm[0] + vector[0] * segment, approach_mm[1] + vector[1] * segment, height]

        if not (self._reachable(approach_mm) and self._reachable(action_mm)):
            return None

        key = f"{function}:{object_id}:{approach}:{destination}"
        phase, target_mm = self._phase_target(
            function, object_id, entry, approach_mm, action_mm, state, action_ref=candidate_id(key)
        )
        if not self._reachable(target_mm):
            return None

        blockers = [
            other
            for other in objects.values()
            if str(other["id"]) != object_id and str(other["id"]) != holding
        ]
        blocker = self._first_blocker(ee, target_mm, blockers, margins)
        if place is not None and not place["free"]:
            # 빈 자리 없는 영역 (h0.6, 리뷰 1 M2): 후보 줄이 `path=blocked`를 말해야 전문가의 경로 답이 hold, 그리퍼 답이
            # `closed`가 되어 하네스가 실제로 명령하는 hold + `conflict{zone_full}`과 라벨이 맞는다.
            blocker = "zone_full"
        clearance = min(
            (
                math.dist(pose, [float(v) for v in other["pose_mm"]])
                - circumradius_mm(obb)
                - circumradius_mm(other["obb_mm"])
                for other in blockers
            ),
            default=float(state["scene"].get("clearance_mm", 0.0)),
        )
        return _Candidate(
            key=key,
            function=function,
            target_ref=object_id,
            approach=approach,
            destination=destination,
            approach_mm=approach_mm,
            action_mm=action_mm,
            target_mm=target_mm,
            distance_mm=int(round(math.dist(ee, target_mm))),
            clearance_mm=int(round(clearance)),
            reach_ok=True,
            path_clear=blocker is None,
            blocker=blocker,
            geometry_age_ms=age_ms,
            moving=bool(self.adapter.moving(object_id)),
            phase=phase,
            toward_zones=tuple(toward),
            place=place,
        )

    @staticmethod
    def _fixed_candidate(key: str) -> _Candidate:
        return _Candidate(key=key, function=None, target_ref=None, approach=None, destination=None)

    def _reachable(self, point: list[float]) -> bool:
        """실행기의 도달·충돌 검사와 **같은 기준**이다 (docs/08 §6 "거절")."""
        if math.dist((0.0, 0.0, 0.0), point) > self.workspace_radius_mm:
            return False
        return point[2] >= self.min_height_mm + self.reach_clearance_mm

    def _first_blocker(
        self,
        start: list[float],
        end: list[float],
        objects: list[dict[str, Any]],
        margins: dict[str, float] | None = None,
    ) -> str | None:
        """직선 구간을 막는 첫 물체. OBB의 외접 구 + 여유로 보수적으로 본다.

        `margins`는 물체별 추가 여유다(금지 접촉 물체는 `planner.forbidden_margin_mm`만큼 더 큰
        장애물이다). 구간의 끝점(명령할 목표점)이 그 물체의 넓힌 구 안에 있으면 그 물체는 추가
        여유 없이 본다 — 목표점 자체가 그 안이면 피할 길이 없고, 그때의 안전은 실행기의 근접
        반사와 하네스의 힘 상한·정지 규칙이 맡는다. 그러지 않으면 금지 물체 곁의 대상은 어떤
        경로로도 닿을 수 없다(최소 간격 85mm의 장면에서 외접 구 + 여유가 이미 대상에 걸친다).
        """
        margin = float(self.planner_config["margin_mm"])
        extra = margins or {}
        hits: list[tuple[float, dict[str, Any]]] = []
        for other in objects:
            centre = [float(v) for v in other["pose_mm"]]
            limit = circumradius_mm(other["obb_mm"]) + margin
            inflation = float(extra.get(str(other["id"]), 0.0))
            if inflation > 0.0 and math.dist(end, centre) >= limit + inflation:
                limit += inflation
            distance = segment_point_distance_mm(start, end, centre)
            if distance < limit:
                hits.append((distance, other))
        if not hits:
            return None
        return str(min(hits, key=lambda item: item[0])[1]["id"])

    def _forbidden_ids(self, state: dict[str, Any]) -> set[str]:
        """금지 접촉 물체: 구조화된 목표의 목록과 `forbidden` 속성."""
        return set(str(item) for item in state["goal"].get("forbidden_contact") or ()) | {
            str(entry["id"]) for entry in state["objects"] if "forbidden" in (entry.get("attributes") or ())
        }

    def _margins(self, state: dict[str, Any]) -> dict[str, float]:
        """물체별 추가 여유. 금지 접촉 물체만 `forbidden_margin_mm`을 갖는다."""
        extra = float(self.planner_config["forbidden_margin_mm"])
        return {object_id: extra for object_id in self._forbidden_ids(state)}

    def _prune(
        self,
        candidates: list[_Candidate],
        objects: dict[str, dict[str, Any]],
        ee: list[float],
        *,
        reserved: str | None = None,
        goal_reserved: list[_Candidate] | None = None,
    ) -> list[_Candidate]:
        """상한 안으로 줄인다. **정답을 모르는 채로** 적용할 수 있는 규칙만 쓴다.

        현재 commitment의 후보(`reserved`)와 지시의 대상×목표 영역 조합(`goal_reserved`, 상태의 `goal`에서 —
        입력이지 라벨이 아니다)은 실행 가능하면 먼저 남긴다. 나머지는 대상 × 기능 × 접근 × 목적지를
        **돌아가며** 고른다: 매번 지금까지 가장 적게 고른 값을 가진 후보를 (대상, 기능, 접근, 목적지의
        순서로) 택하고, 같은 값이면 말단에서 가까운 대상부터 시작하는 farthest-point 순서와 의미 키로
        가른다. 그래서 한 기능이나 한 목적지가 통째로 잘리지 않는다(docs/10 I3). 후보의 확률·비용·의미
        적합성은 보지 않는다.
        """
        budget = int(self.candidates_config["max"]) - int(self.candidates_config["reserved"])
        if len(candidates) <= budget:
            return list(candidates)

        positions = {
            str(candidate.target_ref): [float(value) for value in objects[candidate.target_ref]["pose_mm"]]
            for candidate in candidates
            if candidate.target_ref in objects
        }
        order = _farthest_point_order(positions, ee)
        rank = {object_id: index for index, object_id in enumerate(order)}

        remaining = sorted(candidates, key=lambda item: item.key)
        kept: list[_Candidate] = []
        if reserved is not None:
            held = next((candidate for candidate in remaining if candidate.id == reserved), None)
            if held is not None:
                kept.append(held)
                remaining.remove(held)
        for candidate in sorted(goal_reserved or (), key=lambda item: item.key):
            if len(kept) >= budget:
                break
            if candidate in remaining:
                kept.append(candidate)
                remaining.remove(candidate)

        counts: dict[str, dict[Any, int]] = {dimension: {} for dimension in _SPREAD_DIMENSIONS}
        for candidate in kept:
            for dimension in _SPREAD_DIMENSIONS:
                value = getattr(candidate, dimension)
                counts[dimension][value] = counts[dimension].get(value, 0) + 1

        while remaining and len(kept) < budget:
            chosen = min(
                remaining,
                key=lambda item: (
                    *(counts[dimension].get(getattr(item, dimension), 0) for dimension in _SPREAD_DIMENSIONS),
                    rank.get(str(item.target_ref), len(rank)),
                    item.key,
                ),
            )
            kept.append(chosen)
            remaining.remove(chosen)
            for dimension in _SPREAD_DIMENSIONS:
                value = getattr(chosen, dimension)
                counts[dimension][value] = counts[dimension].get(value, 0) + 1
        return kept

    def _phase_target(
        self,
        function: str,
        object_id: str,
        entry: dict[str, Any],
        approach_mm: list[float],
        action_mm: list[float],
        state: dict[str, Any],
        action_ref: str | None = None,
    ) -> tuple[str, list[float]]:
        """이 후보를 지금 실행하면 어느 국면이고 어디로 가는가 (docs/08 §3.2 `commitment`, §5.6).

        관측으로만 정한다: 무엇을 들고 있는가, 말단이 어디인가, 실행기가 지금 무엇을 어느 국면으로
        실행 중인가(상태의 `exec`). 국면별 목표점은 approach → 접근점, grasp → 파지점, lift → 현재
        XY에서 이동 높이, transport → 목적지 위 접근점(이동 높이), place → 놓기점, push → 접촉점 뒤
        밀기 구간이다.
        """
        spec = self.phases_config
        ee = [float(value) for value in state["robot"]["ee_pose_mm"]]
        holding = state["robot"].get("holding")
        executing = state.get("exec") or {}

        if function == "push":
            # 밀기 구간은 접촉점에서 시작해 정해진 길이만큼 간다. 이미 이 후보의 밀기 국면을 실행 중이면
            # 구간을 이어 간다 — 손이 물체를 가리면(시점 쪽에서 미는 −x 방향) 상태의 자세가 마지막 관측에
            # 머물러 손이 "접촉점을 떠난" 것처럼 보이고, 그러면 후보가 늙은 기하로 빠져 밀기가 끊긴다.
            already = (
                action_ref is not None
                and executing.get("phase") == "push"
                and executing.get("action_ref") == action_ref
            )
            contact_mm = float(spec["push_contact_mm"])
            # 밀기 국면에는 접촉점에서 **멈춘 뒤** 들어간다 — 하강 속도를 안은 채 임피던스가 밀기(부드러움)로
            # 바뀌면 z가 15mm 처져 손가락이 작업면을 친다(3c-2 실측 113~144N).
            settled = float(state["robot"].get("speed_mm_s") or 0.0) <= float(spec["push_entry_speed_mm_s"])
            if already or (math.dist(ee, approach_mm) <= contact_mm and settled):
                return "push", list(action_mm)
            # 접촉점은 물체 옆 중간 높이다. 거기로 곧장 가면 직선이 물체 자신을 지난다(구간 대조는 대상을 뺀다 —
            # 위에서 내려오는 파지에는 맞지만 옆 접촉에는 아니다; 첫 100 rollout에서 밀기 71건 중 59건이 접근 중
            # 34~100N 충돌). 파지처럼 두 단계로 간다: 접촉점 **위**(윗면 + 접근 여유)로 먼저, xy가 맞으면 내려온다.
            above = [approach_mm[0], approach_mm[1], float(entry["top_mm"]) + float(self.candidates_config["approach_clearance_mm"])]
            if math.dist(ee[:2], approach_mm[:2]) <= contact_mm and ee[2] > approach_mm[2]:
                return "approach", list(approach_mm)
            return "approach", above
        if function == "place" or holding == object_id:
            if math.dist(ee[:2], action_mm[:2]) <= float(spec["place_tolerance_mm"]):
                return "place", list(action_mm)
            transport_z = self._transport_height(entry, approach_mm, state)
            if ee[2] < transport_z - float(spec["height_tolerance_mm"]):
                return "lift", [ee[0], ee[1], max(ee[2], transport_z)]
            return "transport", [approach_mm[0], approach_mm[1], max(ee[2], transport_z)]
        # 파지 국면에 **들어가는** 조건은 파지점에 가깝고 xy가 정렬된 것이다. 접근점과 파지점은 언제나
        # 70mm(접근 여유 + 파지 깊이) 떨어져 있어 거리 조건만으로는 xy가 벗어난 채 대각선으로 내려온다
        # (키 큰 원통에서 패드가 윗면 모서리를 누른다 — 3c-1 E0 seed 101). 이미 이 후보의 파지 국면을
        # 실행 중이면 거리 조건만 본다 — 하강 초기의 가로 흔들림(임피던스 전환·관성)이 xy 허용 오차를
        # 잠깐 넘겨도 접근점으로 되돌아가지 않는다(파지점의 xy가 대상이므로 하강하며 바로잡힌다).
        if math.dist(ee, action_mm) <= float(spec["grasp_distance_mm"]):
            already = (
                action_ref is not None
                and executing.get("phase") == "grasp"
                and executing.get("action_ref") == action_ref
            )
            aligned = math.dist(ee[:2], action_mm[:2]) <= float(spec["grasp_xy_tolerance_mm"])
            # 접근의 관성을 안고 내려가지 않는다: 말단이 접근점에서 멈춘 뒤에 하강한다.
            settled = float(state["robot"].get("speed_mm_s") or 0.0) <= float(spec["grasp_entry_speed_mm_s"])
            if already or (aligned and settled):
                return "grasp", list(action_mm)
        return "approach", list(approach_mm)

    def _transport_height(
        self, entry: dict[str, Any], place_approach_mm: list[float], state: dict[str, Any]
    ) -> float:
        """이동 높이: 목적지 접근점 높이와 "작업면에 놓인 물체를 파지점에서 `lift_height_mm`만큼
        든 높이" 중 큰 것. 든 물체의 바닥이 작업면 위의 다른 물체를 넘도록 한다."""
        spec = self.candidates_config
        surface = float(state["scene"].get("work_surface_mm", 0.0))
        obb_z = float(entry["obb_mm"][2])
        lifted = surface + obb_z - float(spec["grasp_depth_mm"]) + float(self.phases_config["lift_height_mm"])
        return max(float(place_approach_mm[2]), lifted)

    # -- 경유점 -------------------------------------------------------------

    def _path_candidates(
        self, state: dict[str, Any], candidate: _Candidate | None, reference: str
    ) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
        """경로 후보와 그 경유점 (docs/08 §4 `q_path`).

        `via`는 **국소 플래너가 실제로 만든 경유점**에만 붙는다. 좌표 없는 `via`는
        실행기가 거절하므로(Task 3a) 만들지 않는다.
        """
        entries = [{"id": "p0", "kind": "direct", "action_ref": reference}]
        waypoints: dict[str, dict[str, Any]] = {}
        if candidate is not None and candidate.function is not None:
            ee = [float(value) for value in state["robot"]["ee_pose_mm"]]
            blockers = self._obstacles(state, candidate.target_ref)
            for index, waypoint in enumerate(
                self._plan_waypoints(ee, candidate.target_mm, blockers, self._margins(state)), start=1
            ):
                name = f"w{index}"
                waypoints[name] = waypoint
                entries.append({"id": f"p{index}", "kind": "via", "ref": name, "action_ref": reference})
        entries.append({"id": "pr", "kind": "retreat", "action_ref": reference})
        entries.append({"id": "ph", "kind": "hold", "action_ref": reference})
        return entries, waypoints

    def _plan_waypoints(
        self,
        start: list[float],
        target: list[float],
        blockers: list[dict[str, Any]],
        margins: dict[str, float] | None = None,
    ) -> list[dict[str, Any]]:
        """국소 경유점 플래너.

        직선 구간을 물체의 외접 구 + 여유로 검사하고, 막혔으면 막은 물체를 **넘어가거나
        돌아가는** 경유점을 만든다. 두 소구간(시작→경유점, 경유점→목표)이 모두 비어 있고
        도달 가능한 것만 남기고, 늘어나는 경로 길이가 짧은 순으로 `n≤3`개를 낸다.
        금지 접촉 물체(`margins`)는 그만큼 더 멀리 돈다.
        """
        spec = self.planner_config
        blocker_id = self._first_blocker(start, target, blockers, margins)
        if blocker_id is None:
            return []

        blocker = next(entry for entry in blockers if str(entry["id"]) == blocker_id)
        pose = [float(value) for value in blocker["pose_mm"]]
        radius = circumradius_mm(blocker["obb_mm"])
        margin = float(spec["margin_mm"]) + float((margins or {}).get(blocker_id, 0.0))
        direct = math.dist(start, target)

        closest = _closest_point(start, target, pose)
        along = _unit_xy([target[0] - start[0], target[1] - start[1]])
        perpendicular = (-along[1], along[0])
        offset = radius + margin + float(spec["side_offset_mm"])

        proposals = [
            (
                "over",
                [pose[0], pose[1], float(blocker["top_mm"]) + float(spec["over_clearance_mm"]) + margin - float(spec["margin_mm"])],
            ),
            ("side+", [closest[0] + perpendicular[0] * offset, closest[1] + perpendicular[1] * offset, closest[2]]),
            ("side-", [closest[0] - perpendicular[0] * offset, closest[1] - perpendicular[1] * offset, closest[2]]),
        ]

        planned: list[dict[str, Any]] = []
        for kind, point in proposals:
            if not self._reachable(point):
                continue
            if self._first_blocker(start, point, blockers, margins) is not None:
                continue
            if self._first_blocker(point, target, blockers, margins) is not None:
                continue
            extra = math.dist(start, point) + math.dist(point, target) - direct
            planned.append(
                {
                    "kind": kind,
                    "pos_mm": [int(round(value)) for value in point],
                    "around": blocker_id,
                    "extra_mm": int(round(extra)),
                }
            )
        planned.sort(key=lambda item: (item["extra_mm"], item["kind"]))
        return planned[: int(spec["max_waypoints"])]

    def _place_points(
        self, state: dict[str, Any], destination: str, entry: dict[str, Any]
    ) -> tuple[list[float], list[float], dict[str, Any]] | None:
        """목적지 영역의 접근 지점·놓기 지점과 자리 기록. 영역을 모르면 `None`.

        놓기점은 고정된 영역 중심이 아니라 **영역 안의 빈 자리**다 (h0.5, 리뷰 2 C2 — E1에서 같은 영역에 두 번째로
        놓는 물체의 하강이 먼저 놓인 물체에 막혀 hold 경로 + 운반 높이 open으로 떨어졌고, seed 13에서는 중심 곁 42mm의
        낮은 상자가 구간 대조에는 걸리지 않으면서 든 원통의 바닥을 받쳐 하강이 멈췄다). 중심에서 가까운 격자점부터
        (`place_grid_mm`; 같은 거리면 x·y 순) 든 물체의 바닥 자국(OBB xy 외접 반지름)이 관측된 다른 물체의 자국과
        `place_margin_mm` 이상 떨어지고, 물체 중심이 영역 경계에서 `place_zone_inset_mm` 안쪽이며(완료 판정은 중심이 영역
        안인 것이다), 그 자리로의 수직 하강 구간이 구간 대조(:meth:`_first_blocker`)에 걸리지 않는 첫 자리를 고른다 —
        결정적이고 관측된 자세만 쓴다. 빈 자리가 없으면 중심을 목표로 두되 `free: False`(zone_full)로 표시한다.

        놓기 높이는 **든 물체의 관측된 바닥**(앞단이 말단 + 파지 오프셋으로 채운다)이 작업면 위 `place_clearance_mm`에
        오는 말단 높이다: 파지가 공칭(윗면 아래 `grasp_depth_mm`)보다 깊거나 얕아도 바닥이 작업면에 닿는 높이에서 열되,
        공칭에서 `place_height_slack_mm`까지만 벗어난다(오프셋 추정이 틀렸을 때의 안전). 들고 있지 않은 대상(합성 상태)은
        공칭이다: 그립 = 작업면 + 높이 − 깊이 + 여유.
        """
        spec = self.candidates_config
        zone = next((item for item in state["zones"] if str(item["id"]) == destination), None)
        if zone is None:
            return None
        centre = _zone_centre(zone)
        x0, y0, x1, y1 = [float(value) for value in zone["bounds_mm"]]
        x0, x1 = min(x0, x1), max(x0, x1)
        y0, y1 = min(y0, y1), max(y0, y1)
        surface = float(state["scene"].get("work_surface_mm", 0.0))
        clearance = float(spec["place_clearance_mm"])
        obb = [float(value) for value in entry["obb_mm"]]
        nominal = surface + obb[2] - float(spec["grasp_depth_mm"]) + clearance
        grip = nominal
        object_id = str(entry["id"])
        holding = state["robot"].get("holding")
        if holding == object_id:
            ee_z = float(state["robot"]["ee_pose_mm"][2])
            bottom = float(entry["pose_mm"][2]) - obb[2] / 2.0
            slack = float(spec.get("place_height_slack_mm", 0.0))
            grip = min(max(ee_z + (surface + clearance) - bottom, nominal - slack), nominal + slack)

        footprint = math.hypot(obb[0] / 2.0, obb[1] / 2.0)
        inset = float(spec["place_zone_inset_mm"])
        others = [
            other for other in state["objects"] if str(other["id"]) not in (object_id, holding)
        ]
        margins = self._margins(state)
        rise = float(spec["approach_clearance_mm"])
        stalled = self._stalled_place_points.get(destination) or []
        exclusion = float(self.compose_config.get("place_stall_exclusion_mm", 0.0))

        def free(px: float, py: float) -> bool:
            if not (x0 + inset <= px <= x1 - inset and y0 + inset <= py <= y1 - inset):
                return False
            if any(math.hypot(px - sx, py - sy) <= exclusion for sx, sy in stalled):
                return False  # 이 에피소드에서 정체한 자리 (h0.6, `place_stalled`)
            return self._place_point_free(px, py, grip, rise, footprint, others, margins)

        grid = float(spec["place_grid_mm"])
        span_x = int((x1 - x0) / (2.0 * grid)) + 1
        span_y = int((y1 - y0) / (2.0 * grid)) + 1
        points = sorted(
            (
                (math.hypot(i * grid, j * grid), i * grid, j * grid)
                for i in range(-span_x, span_x + 1)
                for j in range(-span_y, span_y + 1)
            ),
        )
        for offset, dx, dy in points:
            px, py = centre[0] + dx, centre[1] + dy
            if free(px, py):
                record = {"free": True, "offset_mm": int(round(offset)), "point": "centre" if offset == 0.0 else "offset"}
                return [px, py, grip + rise], [px, py, grip], record
        record = {"free": False, "offset_mm": 0, "point": "zone_full"}
        return [centre[0], centre[1], grip + rise], [centre[0], centre[1], grip], record

    def _place_point_free(
        self,
        px: float,
        py: float,
        grip: float,
        rise: float,
        footprint: float,
        others: list[dict[str, Any]],
        margins: dict[str, float],
    ) -> bool:
        """놓기 자리의 자국·구간 검사 (:meth:`_place_points`의 규칙 (a)·(c)): 든 물체의 자국이 이웃 자국과 `place_margin_mm`
        이상 떨어지고 수직 하강 구간이 구간 대조에 걸리지 않는다. 영역 경계·정체 제외는 부르는 쪽이 본다."""
        margin = float(self.candidates_config["place_margin_mm"])
        for other in others:
            pose = [float(value) for value in other["pose_mm"]]
            reach = footprint + math.hypot(float(other["obb_mm"][0]) / 2.0, float(other["obb_mm"][1]) / 2.0) + margin
            if math.hypot(px - pose[0], py - pose[1]) < reach:
                return False
        return self._first_blocker([px, py, grip + rise], [px, py, grip], others, margins) is None

    def _obstacles(self, state: dict[str, Any], target_ref: str | None) -> list[dict[str, Any]]:
        """구간 검사의 장애물: 관측된 물체 가운데 대상 자신과 들고 있는 물체를 뺀 것."""
        holding = state["robot"].get("holding")
        return [
            entry
            for entry in state["objects"]
            if str(entry["id"]) != target_ref and str(entry["id"]) != holding
        ]

    # -- 실행 이력 ----------------------------------------------------------

    def _exec_history_text(
        self, exec_history: dict[str, Any] | None, *, candidates: dict[str, _Candidate] | None = None
    ) -> str:
        """틱 t−1에서 **실제로 채택·실행된** 답 한 줄 (docs/08 §3.3).

        라벨이 아니라 실행 결과다. 전문가 에피소드에서는 전문가의 실행이, DAgger
        에피소드에서는 모델 답을 하네스가 채택한 결과가 여기로 온다. 경로와 속도는 실행기의
        ACK가 말하는 것(관측 이동 `observe`, 적용한 속도 수준)을 채택 결과보다 앞세운다 — 게이팅
        관측 틱의 채택 결과는 hold·속도 0이지만 실행기는 관측 자세로 움직인다. 요청의 경로 후보가
        아닌 즉석 경유 경로는 `via:<이름>@x,y,z`로 좌표까지 적어 참조 없이 풀리게 한다. `fails`는
        같은 방식(기능·대상·접근)이 **연속으로** 실패한 횟수다 — 하네스가 에피소드 안에서 센다.
        """
        if not exec_history:
            self._failure_streak = None
            return "none"
        adopted = exec_history.get("adopted") if "adopted" in exec_history else exec_history
        if not adopted:
            self._failure_streak = None
            return "none"
        ack = exec_history.get("ack") or {}
        if not ack:
            result = "none"
        elif ack.get("applied"):
            result = "ok"
        else:
            result = str(ack.get("reason") or "discarded")

        main = str(adopted.get("main"))
        if result in ("ok", "none"):
            self._failure_streak = None
            fails = 0
        else:
            entry = (candidates or {}).get(main)
            way = (joint_key_parts(entry.key) or (main,))[:3] if entry is not None else (main,)
            streak = self._failure_streak
            count = int(streak["count"]) + 1 if streak and streak["way"] == way else 1
            self._failure_streak = {"way": way, "count": count}
            fails = count

        applied = bool(ack.get("applied"))
        speed = adopted.get("speed")
        if applied and ack.get("speed_level") is not None:
            speed = int(ack["speed_level"])
        return (
            f"main={main} phase={adopted.get('phase')} "
            f"path={_executed_path_text(adopted, ack if applied else {})} speed={speed} "
            f"force={adopted.get('force')} gripper={adopted.get('gripper')} "
            f"stop={int(bool(adopted.get('stop')))} gate={exec_history.get('gate') or 'none'} "
            f"ack={result} fails={fails}"
        )

    # ------------------------------------------------------------------
    # 조합 규칙 v0 (docs/08 §5)
    # ------------------------------------------------------------------

    def compose(self, request: dict[str, Any], results: dict[str, Any], commitment: dict[str, Any] | None, now_ms: int) -> dict[str, Any]:
        """한 틱의 조합. 정체 감시(h0.8)가 앞뒤를 감싼다 — 안쪽은 :meth:`_compose`다.

        감시는 **틱 사이에 남는 장부**라서 여기 있다: 연속 관측 게이트(기하가 젊어지지 않는)와 연속 hold·정지 틱을
        세고, 상한에 이르면 :meth:`_compose`가 그 틱을 재계획 게이트로 내보낸다 (docs/04 §3, Task R1 B2-v).
        """
        out = self._compose(request, results, commitment, now_ms)
        self._note_stall_outcome(out, request)
        return out

    def _compose(
        self,
        request: dict[str, Any],
        results: dict[str, Any],
        commitment: dict[str, Any] | None,
        now_ms: int,
    ) -> dict[str, Any]:
        """답 묶음 → 명령·채택·commitment (docs/08 §5의 0~6단계를 그 순서로).

        `request`는 :meth:`build_request`가 낸 틱이다. `results`는 모델(또는 규칙 기준군)의
        출력이고 형식은 D0 fixture의 `model_output`과 같다: choice·ordinal은
        `{후보 id: 확률}`, boolean은 `p_true` 또는 `{"true": p, "false": p}`. 응답이 어느
        요청에 대한 것인지 말하는 `meta`(`observed_at`·`goal_version`·`candidate_set_version`
        ·`seq`)는 선택이며, 없으면 이 요청에 대한 답으로 본다.

        돌려주는 것은 `{command, adopted, commitment, switch, gate, records}`다. `records`는
        폐기·정지·게이팅·전환·유지·해제·부가 답 폐기·그리퍼 대기·충돌 사건의 목록이고
        :func:`count_records`가 종류별로 센다.
        """
        model = request.get("request", request)
        block = request.get("harness") or {}
        state = model["state"]
        candidates = {entry["id"]: entry for entry in model["candidates"]["q_main"]}
        paths = {entry["id"]: entry for entry in model["candidates"].get("q_path") or []}
        geometry = dict(block.get("candidates") or {})
        waypoints = block.get("waypoints") or {}
        records: list[dict[str, Any]] = []

        goal_version = int(state.get("goal", {}).get("version", 1))
        seq = int(block.get("seq") or state.get("t", {}).get("seq") or int(request.get("t", 0)) + 1)
        observed_at = int(
            request.get("observed_at_ms", state.get("t", {}).get("observed_at_ms", now_ms))
        )
        set_version = block.get("candidate_set_version") or state.get("t", {}).get(
            "candidate_set_version"
        )

        # 0. 유효성 검사 -------------------------------------------------
        fault = self._validity_fault(
            results.get("meta") or {}, now_ms, observed_at, goal_version, set_version, seq
        )
        if fault is not None:
            records.append({"kind": "discarded", "reason": fault, "seq": seq})
            return {
                "command": None,
                "adopted": None,
                "commitment": commitment,
                "switch": False,
                "gate": None,
                "records": records,
            }

        gripper_now = self._current_gripper(state)
        header = {
            "seq": seq,
            "request_id": block.get("request_id", f"r{request.get('t', 0)}"),
            "observed_at": observed_at,
            "issued_at": int(now_ms),
            "goal_version": goal_version,
            "candidate_set_version": set_version,
        }

        # 1a. 정체 감시 (h0.8, Task R1 B2-v) --------------------------------
        # 스스로 풀 수 없는 반복 — 팔이 카메라를 가린 채 관측을 되풀이하거나(ep-E1-000235), 금지 물체 곁에서
        # 정지가 풀리지 않거나 경로가 막힌 채 hold를 되풀이하는(ep-E1-000244) 구간 — 을 재계획 게이트로 끊는다.
        # **반사·정지보다 먼저** 본다: 풀리지 않는 정지가 바로 그 정체이기 때문이다. 물러남 경로로 나가 팔이
        # 실제로 그 자리를 뜬다.
        stall = self._stall_reason()
        if stall is not None:
            records.append(stall)
            return self._compose_gate(
                header, state, candidates, paths, commitment, gripper_now, records, gate="instr", escape=True
            )

        # 1. 반사·정지 ---------------------------------------------------
        reflex = any(
            str(event.get("kind", "")).startswith("reflex") for event in state.get("events") or []
        )
        if reflex or self._boolean(results, "q_stop", "stop", default=False):
            return self._compose_stop(
                header, state, candidates, paths, commitment, gripper_now, records,
                cause="reflex" if reflex else "q_stop",
            )

        # 2. 게이팅 ------------------------------------------------------
        gate = self._gate(results, state)
        if gate is not None:
            return self._compose_gate(
                header, state, candidates, paths, commitment, gripper_now, records, gate=gate
            )

        # 3. 주 결정과 결정 유지 ------------------------------------------
        blocked = self._retry_blocked(results, model.get("exec_history"), candidates, records)
        current = self._track_place_stall(commitment, state, geometry) if commitment is not None else None
        if current is not None:
            release = self._release_reason(current, state, candidates, geometry)
            if release is None and current["action_ref"] in blocked:
                release = {"reason": "retry_blocked"}
            if release is not None:
                if release["reason"] == "place_stalled":
                    records.append(self._exclude_place_point(current))
                records.append({"kind": "release", **release, "action_ref": current["action_ref"]})
                current = None

        # 주 결정은 따로 적어 두었다가 이 틱이 관측으로 방향을 바꾸면 버린다 — 실행되지 않은
        # 전환·유지가 기록에 남으면 안 된다(docs/10 검토 5). 해제는 위에서 한 번만 적힌다.
        decision_records: list[dict[str, Any]] = []
        chosen, decided, switch = self._main_decision(
            results,
            candidates,
            current,
            decision_records,
            state=state,
            goal_version=goal_version,
            blocked=blocked,
            geometry=geometry,
        )

        # 0의 후속: 선택된 후보의 기하가 **적용 시각 기준**으로 허용치를 넘으면 적용하지 않고
        # 관측 분기로 보낸다 (docs/08 §5.0). 접촉 국면과 파지 중에는 readiness가 시점을 정한다.
        info = geometry.get(chosen) or self._geometry_fallback(candidates.get(chosen), state)
        stale = self._geometry_fault(info, now_ms, observed_at, state)
        if stale is not None:
            records.append({"kind": "geometry_age", "action_ref": chosen, "age_ms": stale})
            return self._compose_gate(
                header, state, candidates, paths, current, gripper_now, records, gate="observe"
            )
        records.extend(decision_records)
        current = decided

        chosen_key = str((candidates.get(chosen) or {}).get("key", ""))
        if info is None and chosen_key not in FIXED_KEYS and chosen_key:
            # 다른 도구의 틱에서 기하를 다시 만들 수 없다(목적지 영역이 없는 등). 명령을 만들지
            # 않고 충돌로 적는다 — `StopIteration`으로 죽는 것도, 모르는 채로 움직이는 것도 아니다.
            records.append({"kind": "conflict", "reason": self._geometry_missing_reason(chosen_key, state), "action_ref": chosen})
        phase = (info or {}).get("phase", "none")
        if current is not None:
            current = {**current, "phase": phase}

        # 4. 부가 답의 적용 ------------------------------------------------
        aux = self._aux(results, paths, state, info, switch=switch, gripper=gripper_now, records=records)

        # 6. 명령 생성 ------------------------------------------------------
        # 실행기 대응은 **채택된 후보**를 따른다 (docs/02 §4 표). 관측·재계획을 게이팅이
        # 아니라 주 결정으로 고른 틱도 같은 원시 기능으로 가야 한다. 5(그리퍼)보다 먼저 만드는 것은 실제로
        # 명령한 경로가 그리퍼 답의 하향(아래)을 정하기 때문이다.
        command, executed = self._command(
            header,
            action_ref=chosen,
            phase=phase,
            info=info,
            path_entry=paths.get(aux["path"]),
            paths=paths,
            waypoints=waypoints,
            speed_level=aux["speed"],
            force_level=aux["force"],
            gripper=aux["gripper"],
            stop=False,
            state=state,
            records=records,
            branch=chosen_key if chosen_key in FIXED_KEYS else None,
        )
        if phase == "place" and executed["kind"] in ("hold", "retreat") and aux["gripper"] == "open" != gripper_now:
            # 놓기 국면인데 실제 명령 경로가 내려가지 않는다(막힌 하강·모델의 경로 답). 운반 높이에서 열면 떨어뜨리는
            # 일이므로 규칙 4의 경로 하향처럼 그리퍼 답도 현재 상태로 하향한다 (h0.5, 리뷰 2 C2). 전문가의 라벨도
            # 같은 틱에 `closed`다(`place_blocked`).
            records.append({"kind": "gripper_downgraded", "desired": "open", "reason": f"place_{executed['kind']}"})
            aux["gripper"] = gripper_now
            command["gripper"] = gripper_now

        # 5. 그리퍼 --------------------------------------------------------
        if aux["gripper"] != gripper_now:
            records.append({"kind": "gripper_change", "desired": aux["gripper"]})
            if state.get("exec", {}).get("gripper_wait"):
                records.append(
                    {"kind": "gripper_wait", "reason": str(state["exec"]["gripper_wait"])}
                )
        current = self._count_forbidden(current, command, records)
        adopted = {
            "main": chosen,
            "switch": switch,
            # 채택 결과는 **실제로 명령한 경로**다 (docs/08 §3.3 — 다음 틱의 실행 이력이 된다).
            # 요청의 경로 후보가 아니면(즉석 경유점) `path`는 없고 종류·경유점이 말한다.
            "path": executed["id"],
            "path_kind": executed["kind"],
            "waypoint": executed["waypoint"],
            "speed": aux["speed"],
            "force": aux["force"],
            "gripper": aux["gripper"],
            "stop": False,
            "phase": phase,
        }
        return {
            "command": command,
            "adopted": adopted,
            "commitment": current,
            "switch": switch,
            "gate": None,
            "records": records,
        }

    def _count_forbidden(
        self, commitment: dict[str, Any] | None, command: dict[str, Any], records: list[dict[str, Any]]
    ) -> dict[str, Any] | None:
        """금지 구간으로 보내지 못한 틱을 센다. `m` 틱 이어지면 commitment를 풀어 다른 행동을 고르게 한다."""
        if commitment is None:
            return None
        if not command["constraints"].get("forbidden_segment"):
            return {**commitment, "forbidden_ticks": 0} if commitment.get("forbidden_ticks") else commitment
        ticks = int(commitment.get("forbidden_ticks", 0)) + 1
        if ticks >= int(self.compose_config["m"]):
            records.append(
                {"kind": "conflict", "reason": "forbidden_blocked", "action_ref": commitment["action_ref"], "ticks": ticks}
            )
            records.append(
                {"kind": "release", "reason": "forbidden_blocked", "action_ref": commitment["action_ref"]}
            )
            return None
        return {**commitment, "forbidden_ticks": ticks}

    @staticmethod
    def _geometry_missing_reason(key: str, state: dict[str, Any]) -> str:
        parts = joint_key_parts(key)
        if parts is not None and parts[3] != "none":
            if not any(str(zone.get("id")) == parts[3] for zone in state.get("zones") or ()):
                return "destination_missing"
        return "geometry_unavailable"

    # -- 0. 유효성 ----------------------------------------------------------

    def _validity_fault(
        self,
        meta: dict[str, Any],
        now_ms: int,
        observed_at: int,
        goal_version: int,
        set_version: str | None,
        seq: int,
    ) -> str | None:
        """응답을 적용해도 되는가 (docs/08 §5.0). 걸리는 것이 없으면 `None`."""
        if now_ms - int(meta.get("observed_at", observed_at)) > self.observation_deadline_ms:
            return "observation_late"
        if int(meta.get("goal_version", goal_version)) != goal_version:
            return "goal_version_mismatch"
        if set_version is not None and meta.get("candidate_set_version", set_version) != set_version:
            return "candidate_set_version_mismatch"
        answered_seq = int(meta.get("seq", seq))
        if answered_seq < seq:
            return "seq_regression"
        if answered_seq > seq:
            return "seq_mismatch"
        return None

    def _geometry_fault(
        self, info: dict[str, Any] | None, now_ms: int, observed_at: int, state: dict[str, Any]
    ) -> int | None:
        """대상 기하가 **적용 시각 기준**으로 허용치를 넘었으면 그 나이를 돌려준다 (docs/08 §5.0, §6).

        요청 시점의 나이에 모델·전송 경과 시간(`now_ms − observed_at`)을 더한다. 말단이 대상의
        행동점에 들어온 파지·놓기 국면과 대상을 들고 있는 동안에는 실행기의 readiness가
        시점을 정하므로 이 검사를 하지 않는다.
        """
        if not info or info.get("function") is None:
            return None
        # 접촉 국면(파지·놓기·밀기)과 파지 중에는 실행기의 접촉이 시점을 정한다. 밀리는 물체는 실행기가
        # 밀어서 움직이므로 이동 대상의 짧은 문턱으로 밀기 자체를 끊으면 안 된다.
        if info.get("phase") in CONTACT_PHASES or state["robot"].get("holding") == info.get("target_ref"):
            return None
        tolerance = (
            self.geometry_age_moving_ms if info.get("moving") else self.geometry_age_static_ms
        )
        age = int(now_ms) - self._geometry_observed_at(info, observed_at)
        return age if age > tolerance else None

    @staticmethod
    def _geometry_observed_at(info: dict[str, Any], observed_at: int) -> int:
        """명령이 참조하는 대상 기하의 관측 시각 (docs/08 §6 `geometry_observed_at`)."""
        return int(observed_at) - int(info.get("geometry_age_ms", 0))

    # -- 1. 정지 -------------------------------------------------------------

    def _compose_stop(
        self,
        header: dict[str, Any],
        state: dict[str, Any],
        candidates: dict[str, Any],
        paths: dict[str, Any],
        commitment: dict[str, Any] | None,
        gripper: str,
        records: list[dict[str, Any]],
        *,
        cause: str,
    ) -> dict[str, Any]:
        """정지 전이. 혼합을 우회하고 **다른 답은 이 틱에 적용하지 않는다** (docs/08 §5.1)."""
        main = self._fallback_main(commitment, candidates)
        records.append({"kind": "stop", "cause": cause, "action_ref": main})

        updated = commitment
        if commitment is not None:
            stop_ticks = int(commitment.get("stop_ticks", 0)) + 1
            if stop_ticks >= int(self.compose_config["m"]):
                records.append(
                    {"kind": "release", "reason": "stop", "action_ref": commitment["action_ref"]}
                )
                updated = None
            else:
                updated = {
                    **commitment,
                    "stop_ticks": stop_ticks,
                    "held_ticks": int(commitment.get("held_ticks", 0)) + 1,
                }

        phase = (commitment or {}).get("phase", "none")
        hold_path = _path_of_kind(paths, "hold")
        command, _ = self._command(
            header,
            action_ref=main,
            phase=phase,
            info=None,
            path_entry=paths.get(hold_path),
            paths=paths,
            waypoints={},
            speed_level=0,
            force_level=0,
            gripper=gripper,
            stop=True,
            state=state,
            records=records,
        )
        adopted = {
            "main": main,
            "switch": False,
            "path": hold_path,
            "path_kind": "hold",
            "waypoint": None,
            "speed": 0,
            "force": 0,
            "gripper": gripper,
            "stop": True,
            "phase": phase,
        }
        return {
            "command": command,
            "adopted": adopted,
            "commitment": updated,
            "switch": False,
            "gate": None,
            "records": records,
        }

    # -- 2. 게이팅 -----------------------------------------------------------

    def _gate(self, results: dict[str, Any], state: dict[str, Any]) -> str | None:
        """완료 → 재계획 → 관측의 순서로 본다 (docs/08 §5.2)."""
        if self._boolean(results, "q_done", "done", default=False):
            return "done"
        if not self._boolean(results, "q_instr", "instr", default=True):
            return "instr"
        if self._boolean(results, "q_observe", "observe", default=False):
            return "observe"
        return None

    def _compose_gate(
        self,
        header: dict[str, Any],
        state: dict[str, Any],
        candidates: dict[str, Any],
        paths: dict[str, Any],
        commitment: dict[str, Any] | None,
        gripper: str,
        records: list[dict[str, Any]],
        *,
        gate: str,
        escape: bool = False,
    ) -> dict[str, Any]:
        """게이팅 분기. commitment를 해제하고 부가 답을 버린다 (docs/08 §5.2, §5.4).

        `escape`는 정체 감시가 부른 재계획이다(h0.8): hold가 아니라 **물러남** 경로를 0이 아닌 속도로 명령해 팔이
        실제로 그 자리를 뜬다 — 금지 물체 곁의 정지처럼 제자리에서는 풀 수 없는 조건이 있기 때문이다.
        """
        carrying = state["robot"].get("holding") is not None
        key = {"done": "hold", "instr": "replan", "observe": "hold" if carrying else "observe"}[gate]
        main = _id_for_key(key, candidates)
        records.append({"kind": "gate", "gate": gate, "action_ref": main})
        if commitment is not None:
            records.append(
                {"kind": "release", "reason": gate, "action_ref": commitment["action_ref"]}
            )
        records.append({"kind": "aux_discarded", "reason": gate})

        escape_speed = int(self.compose_config.get("stall_escape_speed_level", 1)) if escape else 0
        hold_path = (_path_of_kind(paths, "retreat") if escape else None) or _path_of_kind(paths, "hold")
        command, executed = self._command(
            header,
            action_ref=main,
            phase="none",
            info=None,
            path_entry=paths.get(hold_path),
            paths=paths,
            waypoints={},
            speed_level=escape_speed,
            force_level=0,
            gripper=gripper,
            stop=False,
            state=state,
            records=records,
            branch=None if escape else key,
        )
        adopted = {
            "main": main,
            "switch": True,
            # 관측 게이트의 실행기는 관측 자세로 움직인다 — 채택 결과도 그것을 말한다.
            "path": executed["id"],
            "path_kind": executed["kind"],
            "waypoint": executed["waypoint"],
            "speed": escape_speed,
            "force": 0,
            "gripper": gripper,
            "stop": False,
            "phase": "none",
        }
        return {
            "command": command,
            "adopted": adopted,
            "commitment": None,
            "switch": True,
            "gate": gate,
            "records": records,
        }

    # -- 3. 주 결정 ----------------------------------------------------------

    def _release_reason(
        self,
        commitment: dict[str, Any],
        state: dict[str, Any],
        candidates: dict[str, Any],
        geometry: dict[str, Any],
    ) -> dict[str, Any] | None:
        """해제 조건 (docs/08 §5.3): 목표 버전 변경, 무효화, 행동 완료, 연속 파라미터의 표류.

        목표 버전을 모르는 commitment는 지금 목표에 대한 것이라고 볼 수 없으므로 오래된 것으로
        본다. 같은 의미 키라도 연속 파라미터(대상 자세·대상 주변 여유)가 채택 시점에서
        `compose.tolerance`를 넘게 달라졌으면 같은 후보가 아니다(docs/02 §4).
        """
        goal_version = int(state["goal"].get("version", 1))
        if commitment.get("goal_version") is None or int(commitment["goal_version"]) != goal_version:
            return {"reason": "goal_version"}
        if commitment["action_ref"] not in candidates:
            return {"reason": "invalidated"}
        if self._completed(commitment, state):
            return {"reason": "completed"}
        drift = self._drift(commitment, state, geometry.get(commitment["action_ref"]))
        if drift is not None:
            return {"reason": "drifted", **drift}
        waited = int(commitment.get("place_wait_ticks", 0))
        total = int(commitment.get("place_wait_total", 0))
        if waited >= int(self.compose_config.get("m_place", 0) or 10**9):
            return {"reason": "place_stalled", "ticks": waited, "total_ticks": total, "cap": "progress"}
        cap = self.compose_config.get("m_place_total")
        if cap is not None and total >= int(cap):
            # 절대 상한 (h0.7, 리뷰 2 N2): 접촉 중 creep(틱마다 0.1~0.4mm 하강)은 진행 장부를 계속 새로 세게 해 `m_place`를
            # 영원히 미룰 수 있다 — 연속 readiness 대기 틱 자체가 이 수에 이르면 진행과 무관하게 푼다.
            return {"reason": "place_stalled", "ticks": waited, "total_ticks": total, "cap": "total"}
        return None

    def _track_place_stall(
        self, commitment: dict[str, Any], state: dict[str, Any], geometry: dict[str, Any]
    ) -> dict[str, Any]:
        """놓기 정체 장부 (h0.6, 리뷰 1 I6): 놓기 국면에서 실행기가 `gripper_wait: readiness`로 기다린 연속 틱 수와 그 구간의
        시작 높이. 말단이 `place_progress_mm` 이상 더 내려가면 구간을 새로 센다 — 접촉 진동은 진행이 아니다. `m_place`에
        이르면 :meth:`_release_reason`이 `place_stalled`로 commitment를 푼다. `place_wait_total`은 진행과 무관한 연속 대기 틱
        수다(h0.7, 절대 상한 `m_place_total`의 장부)."""
        info = geometry.get(commitment["action_ref"]) or {}
        waiting = str((state.get("exec") or {}).get("gripper_wait") or "") == "readiness"
        if commitment.get("phase") != "place" or not info.get("place_mm") or not waiting:
            return {**commitment, "place_wait_ticks": 0, "place_wait_total": 0, "place_wait_z": None}
        z = float(state["robot"]["ee_pose_mm"][2])
        start = commitment.get("place_wait_z")
        progressed = start is None or float(start) - z >= float(self.compose_config.get("place_progress_mm", 0.0))
        return {
            **commitment,
            "place_wait_ticks": 1 if progressed else int(commitment.get("place_wait_ticks", 0)) + 1,
            "place_wait_total": int(commitment.get("place_wait_total", 0)) + 1,
            "place_wait_z": z if progressed else float(start),
            "place_point_mm": [float(value) for value in info["place_mm"]],
            "place_destination": info.get("destination"),
        }

    def _exclude_place_point(self, commitment: dict[str, Any]) -> dict[str, Any]:
        """정체한 놓기점을 이 에피소드에서 제외한다. 다음 :meth:`_place_points`는 그 둘레 `place_stall_exclusion_mm`를 건넌다."""
        point = [float(value) for value in commitment.get("place_point_mm") or (0.0, 0.0, 0.0)]
        destination = str(commitment.get("place_destination"))
        self._stalled_place_points.setdefault(destination, []).append((point[0], point[1]))
        return {
            "kind": "place_stalled",
            "action_ref": commitment["action_ref"],
            "destination": destination,
            "point_mm": [int(round(value)) for value in point],
            "ticks": int(commitment.get("place_wait_ticks", 0)),
            "total_ticks": int(commitment.get("place_wait_total", 0)),
            "cap": "total" if int(commitment.get("place_wait_ticks", 0)) < int(self.compose_config.get("m_place", 0) or 10**9) else "progress",
        }

    def _drift(
        self, commitment: dict[str, Any], state: dict[str, Any], info: dict[str, Any] | None
    ) -> dict[str, Any] | None:
        """연속 파라미터를 허용 오차로 비교한다. 들고 있는 대상과 밀기는 움직이는 것이 행동이다."""
        parts = joint_key_parts(commitment.get("key", ""))
        if parts is None or parts[0] == "push" or state["robot"].get("holding") == parts[1]:
            return None
        tolerance = self.compose_config["tolerance"]
        entry = next((item for item in state["objects"] if str(item["id"]) == parts[1]), None)
        start = commitment.get("start_pose_mm")
        if entry is not None and start is not None:
            moved = math.dist([float(v) for v in entry["pose_mm"]], [float(v) for v in start])
            if moved > float(tolerance["distance_mm"]):
                return {"parameter": "distance_mm", "delta_mm": int(round(moved))}
        start_clearance = commitment.get("start_clearance_mm")
        if info is not None and start_clearance is not None:
            changed = abs(float(info.get("clearance_mm", 0)) - float(start_clearance))
            if changed > float(tolerance["clearance_mm"]):
                return {"parameter": "clearance_mm", "delta_mm": int(round(changed))}
        return None

    def _completed(self, commitment: dict[str, Any], state: dict[str, Any]) -> bool:
        """행동이 끝났는가. 관측으로만 판정한다."""
        key = str(commitment.get("key", ""))
        held = int(commitment.get("held_ticks", 0))
        parts = joint_key_parts(key)
        if parts is None:
            limit = self.compose_config["observe_ticks" if key == "observe" else "hold_ticks"]
            return key in FIXED_KEYS and held >= int(limit)

        function, target, _approach, destination = parts
        entry = next((item for item in state["objects"] if str(item["id"]) == target), None)
        if entry is None:
            return False
        if function == "push":
            start = commitment.get("start_pose_mm")
            if start is None:
                return False
            moved = math.dist(
                [float(value) for value in entry["pose_mm"]], [float(value) for value in start]
            )
            return moved >= float(self.candidates_config["push_segment_mm"])
        zone = next((item for item in state["zones"] if str(item["id"]) == destination), None)
        if zone is None:
            return False
        return state["robot"].get("holding") != target and _inside(entry["pose_mm"], zone)

    def _retry_blocked(
        self,
        results: dict[str, Any],
        exec_history: Any,
        candidates: dict[str, Any],
        records: list[dict[str, Any]],
    ) -> set[str]:
        """`q_retry`가 거짓이면 직전에 실패한 것과 **같은 방식**의 후보를 막는다.

        docs/08 §4의 `q_retry` 사용처("같은 방식 후보의 실행 조건")다. 같은 방식은
        기능·대상·접근 유형이 같은 것을 말한다(목적지·프로파일은 다를 수 있다). 실패가
        없으면 이 답은 아무것도 바꾸지 않는다 — 근거가 없는 답으로 후보를 지우지 않는다.
        """
        if "q_retry" not in results or self._boolean(results, "q_retry", "retry", default=True):
            return set()
        history = parse_exec_history(exec_history)
        result = history.get("ack")
        if not history or result in (None, "ok", "none"):
            return set()
        failed = candidates.get(str(history.get("main")))
        parts = joint_key_parts((failed or {}).get("key", ""))
        if parts is None:
            return set()
        same = parts[:3]
        blocked = {
            candidate
            for candidate, entry in candidates.items()
            if (joint_key_parts(entry.get("key", "")) or ())[:3] == same
        }
        if blocked:
            records.append(
                {"kind": "retry_blocked", "reason": result, "action_ref": str(history["main"])}
            )
        return blocked

    def _main_decision(
        self,
        results: dict[str, Any],
        candidates: dict[str, Any],
        current: dict[str, Any] | None,
        records: list[dict[str, Any]],
        *,
        state: dict[str, Any],
        goal_version: int,
        blocked: set[str] | None = None,
        geometry: dict[str, Any] | None = None,
    ) -> tuple[str, dict[str, Any] | None, bool]:
        """히스테리시스 (docs/08 §5.3). 돌려주는 것은 (채택 후보, commitment, 전환 여부)."""
        probabilities = results.get("q_main")
        allowed = [
            candidate for candidate in candidates if candidate not in (blocked or set())
        ] or list(candidates)
        ranked = sorted(
            allowed,
            key=lambda cid: (-float((probabilities or {}).get(cid, 0.0)), cid),
        )
        top = ranked[0] if ranked else _id_for_key("hold", candidates)
        answered = bool(probabilities) and float(probabilities.get(top, 0.0)) > 0.0
        if not answered:
            records.append({"kind": "missing_answer", "question": "q_main"})
            if current is not None:
                return current["action_ref"], self._hold(current, clear_challenger=True), False
            top = _id_for_key("hold", candidates)

        if current is None:
            adopted = self._adopt(
                top, candidates, records, state, goal_version, reason="no_commitment", geometry=geometry
            )
            return top, adopted, True

        if top == current["action_ref"]:
            if current.get("challenger"):
                records.append({"kind": "challenger_reset", "reason": "current_is_top"})
            records.append({"kind": "hold", "action_ref": current["action_ref"]})
            return current["action_ref"], self._hold(current, clear_challenger=True), False

        delta = float(self.compose_config["delta"])
        margin = float((probabilities or {}).get(top, 0.0)) - float(
            (probabilities or {}).get(current["action_ref"], 0.0)
        )
        if margin < delta:
            if current.get("challenger"):
                records.append({"kind": "challenger_reset", "reason": "below_delta"})
            records.append(
                {"kind": "hold", "action_ref": current["action_ref"], "reason": "below_delta"}
            )
            return current["action_ref"], self._hold(current, clear_challenger=True), False

        ticks = int(current.get("challenger_ticks", 0))
        if current.get("challenger") != top:
            if current.get("challenger"):
                records.append({"kind": "challenger_reset", "reason": "challenger_changed"})
            ticks = 0
        ticks += 1
        if ticks < int(self.compose_config["m"]):
            records.append(
                {
                    "kind": "hold",
                    "action_ref": current["action_ref"],
                    "reason": "hysteresis",
                    "challenger": top,
                    "challenger_ticks": ticks,
                }
            )
            held = self._hold(current)
            held.update({"challenger": top, "challenger_ticks": ticks})
            return current["action_ref"], held, False

        records.append(
            {
                "kind": "switch",
                "from": current["action_ref"],
                "action_ref": top,
                "margin": round(margin, 4),
            }
        )
        return (
            top,
            self._adopt(top, candidates, records, state, goal_version, reason="hysteresis", geometry=geometry),
            True,
        )

    def _adopt(
        self,
        action_ref: str,
        candidates: dict[str, Any],
        records: list[dict[str, Any]],
        state: dict[str, Any],
        goal_version: int,
        *,
        reason: str,
        geometry: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """새 commitment. 해제 조건이 보는 값(목표 버전·시작 자세·시작 여유)을 여기서 못박는다."""
        if reason == "no_commitment":
            records.append({"kind": "switch", "from": None, "action_ref": action_ref})
        entry = candidates.get(action_ref) or {}
        key = str(entry.get("key", ""))
        parts = joint_key_parts(key)
        start = None
        if parts is not None:
            target = next(
                (item for item in state["objects"] if str(item["id"]) == parts[1]), None
            )
            start = [float(value) for value in target["pose_mm"]] if target else None
        info = (geometry or {}).get(action_ref)
        return {
            "action_ref": action_ref,
            "key": key,
            "phase": "none",
            "held_ticks": 0,
            "last_switch_tick": int(state.get("t", {}).get("tick", 0)),
            "goal_version": int(goal_version),
            "challenger": None,
            "challenger_ticks": 0,
            "stop_ticks": 0,
            "start_pose_mm": start,
            "start_clearance_mm": int(info["clearance_mm"]) if info else None,
            "place_wait_ticks": 0,
            "place_wait_z": None,
        }

    @staticmethod
    def _hold(commitment: dict[str, Any], *, clear_challenger: bool = False) -> dict[str, Any]:
        held = {
            **commitment,
            "held_ticks": int(commitment.get("held_ticks", 0)) + 1,
            "stop_ticks": 0,
        }
        if clear_challenger:
            held.update({"challenger": None, "challenger_ticks": 0})
        return held

    @staticmethod
    def _fallback_main(commitment: dict[str, Any] | None, candidates: dict[str, Any]) -> str:
        if commitment and commitment["action_ref"] in candidates:
            return str(commitment["action_ref"])
        return _id_for_key("hold", candidates)

    # -- 4. 부가 답 ----------------------------------------------------------

    def _aux(
        self,
        results: dict[str, Any],
        paths: dict[str, Any],
        state: dict[str, Any],
        info: dict[str, Any] | None,
        *,
        switch: bool,
        gripper: str,
        records: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """경로·속도·힘·그리퍼 (docs/08 §5.4).

        전환 틱에는 **네 답을 모두** 버리고 새 후보의 초기 프로파일(경로 direct — 막혔으면
        명령 생성이 경유 경로로 바꾼다, 속도 1단계, 힘 회피, 그리퍼 현재 상태)을 쓴다. 유지
        틱에는 답을 쓰되 commitment의 제약(취약 물체 근접, 금지 물체 근접) 안으로 묶는다.
        """
        initial = self.compose_config["initial_profile"]
        is_joint_action = bool(info and info.get("function"))
        # 경로 후보의 id는 틱마다 다를 수 있으므로 **종류**로 고른다.
        hold_path = _path_of_kind(paths, "hold")
        direct_path = _path_of_kind(paths, str(initial["path"])) or hold_path
        fallback = direct_path if is_joint_action else hold_path
        if switch:
            records.append({"kind": "aux_discarded", "reason": "switch"})
            path = fallback
            speed = int(initial["speed_level"]) if is_joint_action else 0
            force = int(initial["force_level"])
            desired = gripper
        else:
            path = self._argmax(results.get("q_path"), paths, default=fallback, question="q_path", records=records)
            speed = int(
                self._argmax(
                    results.get("q_speed"),
                    {str(index): None for index in range(len(self.speed_levels_m_s))},
                    default=str(initial["speed_level"] if is_joint_action else 0),
                    question="q_speed",
                    records=records,
                )
            )
            force = int(
                self._argmax(
                    results.get("q_force"),
                    {str(index): None for index in range(len(self.force_level_names))},
                    default=str(initial["force_level"]),
                    question="q_force",
                    records=records,
                )
            )
            desired = self._argmax(
                results.get("q_gripper"), {"open": None, "closed": None}, default=gripper,
                question="q_gripper", records=records,
            )

        capped = self._cap_speed(speed, state, info)
        if capped != speed:
            records.append({"kind": "speed_cap", "from": speed, "to": capped})
            speed = capped
        capped = self._cap_force(force, state, info)
        if capped != force:
            records.append({"kind": "force_cap", "from": force, "to": capped})
            force = capped
        return {"path": path, "speed": speed, "force": force, "gripper": desired}

    def _cap_speed(self, speed: int, state: dict[str, Any], info: dict[str, Any] | None) -> int:
        if self._near(state, info, "fragile", float(self.compose_config["fragile_proximity_mm"])):
            return min(speed, int(self.compose_config["fragile_speed_cap"]))
        return speed

    def _cap_force(self, force: int, state: dict[str, Any], info: dict[str, Any] | None) -> int:
        if self._near(state, info, "forbidden", float(self.compose_config["forbidden_proximity_mm"])):
            return 0
        return force

    def _near(
        self, state: dict[str, Any], info: dict[str, Any] | None, attribute: str, limit: float
    ) -> bool:
        """말단이나 대상 근처에 그 속성의 물체가 있는가. 관측된 자세로만 본다."""
        points = [[float(value) for value in state["robot"]["ee_pose_mm"]]]
        if info and info.get("target_ref"):
            entry = next(
                (item for item in state["objects"] if str(item["id"]) == info["target_ref"]), None
            )
            if entry is not None:
                points.append([float(value) for value in entry["pose_mm"]])
        for entry in state["objects"]:
            if attribute not in (entry.get("attributes") or ()):
                continue
            if info and str(entry["id"]) == info.get("target_ref"):
                continue
            pose = [float(value) for value in entry["pose_mm"]]
            if any(math.dist(pose, point) <= limit for point in points):
                return True
        return False

    # -- 6. 명령 -------------------------------------------------------------

    def _command(
        self,
        header: dict[str, Any],
        *,
        action_ref: str,
        phase: str,
        info: dict[str, Any] | None,
        path_entry: dict[str, Any] | None,
        paths: dict[str, Any],
        waypoints: dict[str, Any],
        speed_level: int,
        force_level: int,
        gripper: str,
        stop: bool,
        state: dict[str, Any],
        records: list[dict[str, Any]],
        branch: str | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """docs/08 §6의 명령. 실행기가 받는 필드만 만든다.

        돌려주는 것은 (명령, 실제로 명령한 경로의 기술 `{id, kind, waypoint}`). 막힌 direct를
        경유 경로로 바꾸면 채택 결과가 그 경유 경로를 말한다 — 요청의 경로 후보면 그 id로, 즉석
        계획이면 id 없이(`None`) 종류·경유점 이름·좌표로.
        """
        path, target_ref, forbidden, executed = self._path_block(
            path_entry, info, phase, paths, waypoints, state, records, action_ref=action_ref
        )
        command = {
            **header,
            "lease_until": int(header["issued_at"]) + self.lease_ms,
            "action_ref": action_ref,
            "phase": phase,
            "path": path,
            "speed_level": int(speed_level),
            "force_level": self.force_level_names[int(force_level)],
            "gripper": gripper,
            "stop": bool(stop),
            # 그리퍼가 작용하는 물체 (docs/08 §6). 밀기는 주먹으로 하므로 없다 — 실행기의 close readiness
            # (대상까지의 거리)가 걸리지 않아 접촉 전에 손가락이 닫힌다. 밀 물체는 `path.target_ref`가 가리킨다.
            "gripper_ref": None if (info or {}).get("function") == "push" else target_ref,
            "frame": str(self.command_config["frame"]),
            "constraints": {
                "forbidden_objects": list(state["goal"].get("forbidden_contact") or ()),
                "speed_limit_mm_s": int(self.speed_levels_m_s[int(speed_level)] * 1000),
                "forbidden_segment": forbidden,
            },
            "geometry_age_ms": int((info or {}).get("geometry_age_ms", 0)),
            "geometry_observed_at": (
                self._geometry_observed_at(info, int(header["observed_at"]))
                if info and info.get("function")
                else None
            ),
            "target_moving": bool((info or {}).get("moving", False)),
        }
        if phase == "place" and (info or {}).get("place_mm"):
            # 놓기점은 경로와 따로 나른다 (h0.5): 실행기의 open readiness가 경로 종류(hold 포함)와 무관하게 이 자리에 댄다.
            command["place_mm"] = _round_list(info["place_mm"])
        if branch == "observe":
            command["observe"] = True
            if state["robot"].get("holding") is None:
                # 실행기는 관측 자세로 **이동**한다 (docs/08 §6 관측). 운반 중이면 제자리다.
                executed = _executed(None, "observe")
        elif branch == "replan":
            command["replan"] = True
        return command, executed

    def _path_block(
        self,
        path_entry: dict[str, Any] | None,
        info: dict[str, Any] | None,
        phase: str,
        paths: dict[str, Any],
        waypoints: dict[str, Any],
        state: dict[str, Any],
        records: list[dict[str, Any]],
        *,
        action_ref: str,
    ) -> tuple[dict[str, Any], str | None, bool, dict[str, Any]]:
        """경로 블록. **실제로 명령할 구간**을 관측된 장애물에 대조한다 (docs/08 §5.4, §5.6).

        * 금지 접촉 물체는 `forbidden_margin_mm`만큼 더 큰 장애물이다. 구간에 걸리면 먼저 경유
          경로를 찾고(`conflict{forbidden_reroute}`), 없을 때만 그 구간을 보내지 않고 `hold`에
          `forbidden_segment`를 붙여 실행기가 정지 전이하게 한다(`conflict{forbidden_segment}`).
        * 다른 물체에 막힌 direct는 보내지 않는다 — 국소 플래너의 첫 경유 경로(요청의 경유점이
          이 후보의 것이면 그것, 아니면 지금 계산한 것), 그것도 없으면 `hold`. 둘 다 충돌로 적는다.
        * via는 말단→경유점, 경유점→목표점 두 구간을 본다.

        돌려주는 것은 (경로 블록, 대상 id, 금지 구간 여부, 실행 경로 기술). 실행 경로 기술은
        `{"id": 요청의 경로 후보 id 또는 None, "kind": 실제 종류, "waypoint": {"ref", "pos_mm"} | None}`
        이며 채택 결과·실행 이력이 **실제로 명령한 경로**를 말하게 한다 (docs/08 §3.3).
        """
        kind = (path_entry or {}).get("kind", "hold")
        target_ref = (info or {}).get("target_ref")
        joint = info is not None and info.get("function") is not None
        if joint and phase in ("transport", "place") and (info.get("place") or {}).get("free") is False:
            # 영역에 빈 자리가 없다 (h0.5). 중심으로 내려가 떨어뜨리지 않고 hold로 기다린다 — 그 사이 목표가 바뀌거나
            # 외란이 자리를 만든다. 충돌로 적는다 — 경로 답이 무엇이든(hold·retreat 포함, h0.7 리뷰 2 N4): 후보 줄의
            # `path=blocked`(blocker zone_full)로 전문가·규칙 기준군이 hold를 답하는 틱에도 기록은 남아야 한다.
            records.append({"kind": "conflict", "reason": "zone_full", "target_ref": target_ref, "destination": info.get("destination")})
            actual = "retreat" if kind == "retreat" else "hold"
            return {"kind": actual}, target_ref, False, _executed(_path_of_kind(paths, actual), actual)
        if not joint or kind in ("hold", "retreat"):
            actual = "hold" if kind != "retreat" else "retreat"
            return {"kind": actual}, target_ref, False, _executed(_path_of_kind(paths, actual), actual)

        point = [float(value) for value in (info.get("target_mm") or self._legacy_target(info, phase))]
        if not self._reachable(point):
            # 알면서 거절당할 명령을 내지 않는다 (docs/08 §5.6의 충돌 기록).
            records.append({"kind": "conflict", "reason": "unreachable", "target_ref": target_ref})
            return {"kind": "hold"}, target_ref, False, _executed(_path_of_kind(paths, "hold"), "hold")

        ee = [float(value) for value in state["robot"]["ee_pose_mm"]]
        obstacles = self._obstacles(state, target_ref)
        forbidden_ids = self._forbidden_ids(state)
        margins = self._margins(state)
        hold = _executed(_path_of_kind(paths, "hold"), "hold")

        def crossing(start: list[float], end: list[float]) -> tuple[str | None, str | None]:
            """(금지 물체, 다른 장애물) — 구간을 막는 첫 물체를 종류별로 (금지 물체는 추가 여유)."""
            forbidden_hit = self._first_blocker(
                start, end, [entry for entry in obstacles if str(entry["id"]) in forbidden_ids], margins
            )
            other_hit = self._first_blocker(start, end, obstacles, margins)
            return forbidden_hit, other_hit

        def reroute(blocker: str, reason: str) -> tuple[dict[str, Any], str | None, bool, dict[str, Any]]:
            """막힌 구간 대신 첫 경유 경로. 없으면 금지 물체는 정지 전이, 다른 물체는 hold."""
            detour = self._first_detour(paths, waypoints, action_ref, ee, point, obstacles, margins)
            if detour is None:
                if reason == "forbidden_segment":
                    records.append(
                        {"kind": "conflict", "reason": "forbidden_segment", "blocker": blocker, "target_ref": target_ref}
                    )
                    return {"kind": "hold"}, target_ref, True, hold
                records.append(
                    {"kind": "conflict", "reason": "path_blocked", "blocker": blocker,
                     "target_ref": target_ref, "resolution": "hold"}
                )
                return {"kind": "hold"}, target_ref, False, hold
            name, waypoint, path_id = detour
            records.append(
                {"kind": "conflict", "reason": "forbidden_reroute" if reason == "forbidden_segment" else "path_blocked",
                 "blocker": blocker, "target_ref": target_ref, "resolution": "via", "waypoint": name}
            )
            block = {
                "kind": "via",
                "target_ref": target_ref,
                "target_mm": _round_list(point),
                "waypoint_ref": name,
                "waypoint_mm": _round_list(waypoint["pos_mm"]),
            }
            return block, target_ref, False, _executed(path_id, "via", name, block["waypoint_mm"])

        if kind == "via":
            waypoint = waypoints.get(str(path_entry.get("ref")))
            if waypoint is None:
                records.append({"kind": "conflict", "reason": "waypoint_missing"})
                kind = "direct"
            else:
                waypoint_mm = [float(value) for value in waypoint["pos_mm"]]
                for start, end in ((ee, waypoint_mm), (waypoint_mm, point)):
                    forbidden_hit, other_hit = crossing(start, end)
                    if forbidden_hit is not None:
                        return reroute(forbidden_hit, "forbidden_segment")
                    if other_hit is not None:
                        records.append(
                            {"kind": "conflict", "reason": "path_blocked", "blocker": other_hit,
                             "target_ref": target_ref, "resolution": "hold"}
                        )
                        return {"kind": "hold"}, target_ref, False, hold
                block = {
                    "kind": "via",
                    "target_ref": target_ref,
                    "target_mm": _round_list(point),
                    "waypoint_ref": str(path_entry["ref"]),
                    "waypoint_mm": _round_list(waypoint_mm),
                }
                return block, target_ref, False, _executed(
                    str(path_entry["id"]), "via", str(path_entry["ref"]), block["waypoint_mm"]
                )

        forbidden_hit, other_hit = crossing(ee, point)
        if forbidden_hit is not None:
            return reroute(forbidden_hit, "forbidden_segment")
        if other_hit is not None:
            return reroute(other_hit, "path_blocked")
        block = {"kind": "direct", "target_ref": target_ref, "target_mm": _round_list(point)}
        path_id = str(path_entry["id"]) if path_entry else _path_of_kind(paths, "direct")
        return block, target_ref, False, _executed(path_id, "direct")

    def _first_detour(
        self,
        paths: dict[str, Any],
        waypoints: dict[str, Any],
        action_ref: str,
        ee: list[float],
        point: list[float],
        obstacles: list[dict[str, Any]],
        margins: dict[str, float] | None = None,
    ) -> tuple[str, dict[str, Any], str | None] | None:
        """막힌 direct를 대신할 첫 경유점: (이름, 경유점, 요청의 경로 후보 id).

        요청의 경유점이 이 후보의 것이면 그것을 쓴다(모델이 본 이름 그대로). 아니면 지금 계획하고
        요청의 경유점과 겹치지 않는 이름(`w<n+1>`)을 붙인다 — 이 경우 경로 후보 id는 없다.
        """
        for path_id, entry in paths.items():
            if entry.get("kind") == "via" and str(entry.get("action_ref")) == str(action_ref):
                waypoint = waypoints.get(str(entry.get("ref")))
                if waypoint is not None:
                    return str(entry["ref"]), waypoint, str(path_id)
        planned = self._plan_waypoints(ee, point, obstacles, margins)
        if not planned:
            return None
        return f"w{len(waypoints) + 1}", planned[0], None

    @staticmethod
    def _legacy_target(info: dict[str, Any], phase: str) -> list[float]:
        """`target_mm`이 없는 기하(다른 하네스 버전의 블록)에서 국면별 목표점을 고른다."""
        return info["approach_mm"] if phase == "approach" else info["action_mm"]

    # -- 답 읽기 -------------------------------------------------------------

    def _boolean(
        self, results: dict[str, Any], question_id: str, gate: str, *, default: bool
    ) -> bool:
        """boolean 답을 임계값으로 읽는다. 두 출력 형식을 모두 받는다."""
        value = results.get(question_id)
        if value is None:
            return default
        if isinstance(value, dict):
            if "true" in value:
                probability = float(value["true"])
            elif "false" in value:
                probability = 1.0 - float(value["false"])
            else:
                return default
        else:
            probability = float(value)
        return probability >= float(self.compose_config["gates"][gate])

    @staticmethod
    def _argmax(
        answer: Any,
        allowed: dict[str, Any],
        default: str,
        *,
        question: str | None = None,
        records: list[dict[str, Any]] | None = None,
    ) -> str:
        """choice·ordinal 답에서 최댓값. 후보 밖 답이나 없는 답은 기본값을 쓰고 `missing_answer`로 적는다."""
        usable = (
            {key: float(value) for key, value in answer.items() if key in allowed}
            if isinstance(answer, dict)
            else {}
        )
        if not usable:
            if records is not None and question is not None:
                records.append(
                    {
                        "kind": "missing_answer",
                        "question": question,
                        "reason": "absent" if answer is None else "outside_candidates",
                    }
                )
            return default
        return min(usable, key=lambda key: (-usable[key], key))

    def _current_gripper(self, state: dict[str, Any]) -> str:
        reported = state.get("exec", {}).get("gripper")
        if reported in ("open", "closed"):
            return str(reported)
        midpoint = (self.gripper_open_mm + self.gripper_closed_mm) / 2.0
        return "open" if float(state["robot"]["gripper_mm"]) > midpoint else "closed"

    def _geometry_fallback(
        self, entry: dict[str, Any] | None, state: dict[str, Any]
    ) -> dict[str, Any] | None:
        """하네스 블록 없이 들어온 요청(다른 도구가 만든 틱)의 기하를 다시 계산한다."""
        if not entry:
            return None
        parts = joint_key_parts(entry.get("key", ""))
        if parts is None:
            return None
        objects = {item["id"]: item for item in state["objects"]}
        target = objects.get(parts[1])
        if target is None:
            return None
        ee = [float(value) for value in state["robot"]["ee_pose_mm"]]
        ages = {
            item["object"]: item.get("age_ms", 0) for item in state.get("derived") or () if "object" in item
        }
        candidate = self._geometry_for(
            parts, target, objects, state, ee, int(ages.get(parts[1], 0)), self._margins(state)
        )
        return candidate.geometry() if candidate else None

    @staticmethod
    def _project_commitment(commitment: dict[str, Any] | None) -> dict[str, Any] | None:
        """모델이 보는 commitment (docs/08 §3.2). 하네스 장부는 빼고 보낸다."""
        if commitment is None:
            return None
        return {
            "action_ref": commitment["action_ref"],
            "key": commitment.get("key", ""),
            "phase": commitment.get("phase", "none"),
            "held_ticks": int(commitment.get("held_ticks", 0)),
            "last_switch_tick": int(commitment.get("last_switch_tick", 0)),
        }


def _round_list(values) -> list[int]:
    return [int(round(float(value))) for value in values]


def _executed_path_text(adopted: dict[str, Any], ack: dict[str, Any]) -> str:
    """실행 이력의 `path=` 값: 실행기가 관측으로 움직였으면 `observe`, 요청의 경로 후보면 그 id,
    즉석 경유 경로면 `via:<이름>@x,y,z`, 아니면 실제 종류."""
    if ack.get("path") == "observe":
        return "observe"
    if adopted.get("path") is not None:
        return str(adopted["path"])
    kind = str(adopted.get("path_kind") or "hold")
    waypoint = adopted.get("waypoint")
    if kind == "via" and waypoint and waypoint.get("pos_mm") is not None:
        x, y, z = (int(round(float(value))) for value in waypoint["pos_mm"])
        return f"via:{waypoint.get('ref') or 'w'}@{x},{y},{z}"
    return kind


def _executed(
    path_id: str | None, kind: str, waypoint_ref: str | None = None, waypoint_mm: list[int] | None = None
) -> dict[str, Any]:
    """실제로 명령한 경로의 기술 (채택 결과 `path`·`path_kind`·`waypoint`의 원천)."""
    return {
        "id": path_id,
        "kind": kind,
        "waypoint": {"ref": waypoint_ref, "pos_mm": list(waypoint_mm)} if waypoint_ref else None,
    }


def _unit_xy(vector) -> tuple[float, float]:
    length = math.dist((0.0, 0.0), vector)
    if length <= 1e-9:
        return (1.0, 0.0)
    return (vector[0] / length, vector[1] / length)


def _zone_centre(zone: dict[str, Any]) -> tuple[float, float]:
    x0, y0, x1, y1 = [float(value) for value in zone["bounds_mm"]]
    return ((x0 + x1) / 2.0, (y0 + y1) / 2.0)


def _path_of_kind(paths: dict[str, Any], kind: str) -> str | None:
    """그 틱의 경로 후보 중 이 종류의 첫 id. 없으면 `None`."""
    for path_id, entry in paths.items():
        if entry.get("kind") == kind:
            return str(path_id)
    return None


def _id_for_key(key: str, candidates: dict[str, Any]) -> str:
    """그 틱의 목록에서 의미 키가 같은 후보의 id.

    하네스가 만든 틱에서는 :func:`candidate_id`와 같은 값이지만, 다른 도구가 만든 틱(D0
    fixture 등)은 다른 id 규칙을 쓸 수 있다. **목록에 있는 id**를 골라야 채택 결과가
    그 틱의 후보를 가리킨다.
    """
    for candidate, entry in candidates.items():
        if entry.get("key") == key:
            return str(candidate)
    return candidate_id(key)


def _inside(pose_mm, zone: dict[str, Any]) -> bool:
    x0, y0, x1, y1 = [float(value) for value in zone["bounds_mm"]]
    x, y = float(pose_mm[0]), float(pose_mm[1])
    return min(x0, x1) <= x <= max(x0, x1) and min(y0, y1) <= y <= max(y0, y1)


def _closest_point(start, end, point) -> list[float]:
    segment = [b - a for a, b in zip(start, end)]
    length_sq = sum(value * value for value in segment)
    if length_sq <= 1e-9:
        return list(start)
    t = sum((p - a) * s for a, s, p in zip(start, segment, point)) / length_sq
    t = min(1.0, max(0.0, t))
    return [a + s * t for a, s in zip(start, segment)]


def _farthest_point_order(positions: dict[str, list[float]], ee: list[float]) -> list[str]:
    """말단에서 가까운 대상부터 시작하는 farthest-point 표본 순서."""
    remaining = dict(positions)
    if not remaining:
        return []
    first = min(remaining, key=lambda key: (math.dist(ee, remaining[key]), key))
    order = [first]
    chosen = [remaining.pop(first)]
    while remaining:
        nxt = max(
            remaining,
            key=lambda key: (min(math.dist(remaining[key], point) for point in chosen), key),
        )
        order.append(nxt)
        chosen.append(remaining.pop(nxt))
    return order


# --------------------------------------------------------------------------
# 모듈 수준 편의 함수 (docs/06의 인터페이스 이름)
# --------------------------------------------------------------------------


def build_request(
    observation: dict[str, Any],
    exec_history: dict[str, Any] | None = None,
    commitment: dict[str, Any] | None = None,
    *,
    harness: RobotHarness | None = None,
) -> dict[str, Any]:
    """틱 요청 하나. `harness`를 주지 않으면 **그 호출에만 쓰는** 하네스를 만든다.

    에피소드를 이어 만들 때는 :class:`RobotHarness`를 직접 들고 써야 한다 — 앞단의 추적
    기억(마지막으로 본 자세)이 하네스에 붙어 있기 때문이다.
    """
    return (harness or RobotHarness.from_config_path()).build_request(
        observation, exec_history, commitment
    )


def compose(
    request: dict[str, Any],
    results: dict[str, Any],
    commitment: dict[str, Any] | None,
    now_ms: int,
    *,
    harness: RobotHarness | None = None,
) -> dict[str, Any]:
    """조합 규칙 v0. `harness`를 주지 않으면 기본 설정의 하네스를 쓴다."""
    return (harness or RobotHarness.from_config_path()).compose(
        request, results, commitment, now_ms
    )
