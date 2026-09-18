"""규칙 기반 판단기 기준군 v0 — 모델 자리에 들어가는 결정적 규칙 (docs/02 §9).

모델과 **같은 요청**을 받고 **같은 형식**으로 10개 답을 낸다. 순서는 세 단계다.

1. **적합성 필터.** 상태의 구조화된 목표·제약(대상·목적지·금지 접촉)으로 적합 후보를
   고른다. 모델에게 주는 후보 목록은 그대로 두고(하네스는 부적합한 실행 가능 후보도
   제시한다, docs/08 §7) 여기서만 거른다.
2. **고정 가중 기하 비용.** 하네스가 계산한 값(여유·거리·도달·경로 막힘·직전에 실패한
   접근 유형)의 고정 가중합이다. 가중치는 `configs/harness/rule_judge_v0.yaml`에 있다.
3. **결정적 분포.** `softmax(-비용/온도)`를 적합 후보에 주고, 부적합 후보는 설정된 작은
   질량을 고르게 나눈다. 난수는 쓰지 않는다.

게이팅·정지·그리퍼·경로·속도·힘도 같은 설정의 임계값으로 답한다. 이 기준군이 어떤 조건
층에서 모델과 같은 성공률을 내면 그 층에서는 모델의 기여를 주장하지 않는다.

**입력 경계.** 요청 안의 것만 본다. 하네스 블록(`request["harness"]`)이 있으면 거기 있는
기하 값을 쓰고, 없으면 모델이 보는 후보 설명(`derived`)에서 같은 값을 읽는다 — 그래서
다른 도구가 만든 틱(D0 fixture 등)에도 그대로 답할 수 있다.
"""

from __future__ import annotations

import copy
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from robo_jev.contracts import PHASES
from robo_jev.harness.robot import FIXED_KEYS, load_harness_config, parse_exec_history
from robo_jev.perception.pointworld import named_target
from robo_jev.sim.controller import load_controller_config, resolve_config_path

__all__ = [
    "DEFAULT_CONFIG_PATH",
    "RULE_JUDGE_VERSION",
    "Goal",
    "RuleJudge",
    "candidate_values",
    "load_rule_judge_config",
    "normalise_distribution",
    "read_goal",
    "rule_judge",
]

#: 규칙 버전. 레코드의 `versions.rules`에 들어간다.
RULE_JUDGE_VERSION = "rj0.3"

DEFAULT_CONFIG_PATH = "configs/harness/rule_judge_v0.yaml"

#: 후보 설명의 기하 값 (`reach ok, clr 41mm, d 320mm, path clear, geom 120ms`).
_DERIVED = re.compile(
    r"reach (?P<reach>ok|no)|clr (?P<clearance>-?\d+)mm|d (?P<distance>-?\d+)mm|"
    r"path (?P<path>clear|blocked)|geom (?P<geom>-?\d+)ms"
)

#: 기하 나이 대신 하네스의 `max_geometry_age_ms`가 관측 문턱인 국면 (docs/08 §4 `q_observe`).
_CONTACT_PHASES = ("grasp", "place")


def load_rule_judge_config(path: str | Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    return yaml.safe_load(resolve_config_path(path).read_text(encoding="utf-8"))


# --------------------------------------------------------------------------
# 목표 읽기 — 기준군과 전문가가 같은 규칙으로 읽는다 (docs/08 §3.2 `goal`)
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Goal:
    """상태의 `goal`을 읽은 것.

    `structured`가 참이면 상태에 구조화된 목표(`target_desc`가 있는 형태 — 3c-1의 환경·어댑터
    경로)가 있어서 텍스트를 파싱하지 않았다. 거짓이면 텍스트 근사 경로다(D0 fixture 등):
    대상은 상태의 물체 설명이 지시문에 나오는 것 가운데 마지막으로 불린 평범한 물체다.
    `target_ref`는 **추적 중인** 물체만 가리킨다. 구조화된 목표의 대상이 아직 보이지 않으면
    `target_ref`는 없고 `target_desc`만 있다 — 그것은 관측의 문제이지 지시의 문제가 아니다.
    """

    text: str
    version: int
    structured: bool
    target_ref: str | None
    target_desc: str | None
    zone: str | None
    forbidden: tuple[str, ...]
    fragile: tuple[str, ...]


def read_goal(state: dict[str, Any]) -> Goal:
    goal = state.get("goal") or {}
    objects = state.get("objects") or ()
    text = str(goal.get("text") or "")
    structured = "target_desc" in goal
    target_ref = goal.get("target_ref")
    if not structured and not target_ref:
        target_ref = named_target(objects, text)
    if target_ref is not None and not any(str(entry["id"]) == str(target_ref) for entry in objects):
        target_ref = None
    return Goal(
        text=text,
        version=int(goal.get("version", 1)),
        structured=structured,
        target_ref=str(target_ref) if target_ref is not None else None,
        target_desc=str(goal["target_desc"]) if goal.get("target_desc") else None,
        zone=str(goal["target_zone"]) if goal.get("target_zone") else None,
        forbidden=tuple(str(item) for item in goal.get("forbidden_contact") or ()),
        fragile=tuple(str(item) for item in goal.get("fragile") or ()),
    )


def candidate_values(entry: dict[str, Any], request: dict[str, Any] | None = None) -> dict[str, Any]:
    """후보 하나의 의미 조각과 기하 값.

    하네스 블록(`request["harness"]`)이 있으면 그 값을, 없으면 모델이 보는 `derived` 문자열을
    읽는다 — 그래서 다른 도구가 만든 틱(D0 fixture 등)에도 답할 수 있다. 전문가는 블록을
    주지 않고 부른다(모델 입력만 본다).
    """
    parts = str(entry.get("key", "")).split(":")
    semantic = (
        {
            "function": parts[0],
            "target": parts[1],
            "approach": parts[2],
            "destination": parts[3],
            "profile": parts[4],
        }
        if len(parts) == 5
        else {
            "function": None,
            "target": None,
            "approach": None,
            "destination": None,
            "profile": None,
        }
    )
    block = ((request or {}).get("harness") or {}).get("candidates") or {}
    geometry = block.get(entry["id"])
    if geometry is not None:
        return {
            **semantic,
            "reach_ok": bool(geometry["reach_ok"]),
            "clearance_mm": float(geometry["clearance_mm"]),
            "distance_mm": float(geometry["distance_mm"]),
            "path_clear": bool(geometry["path_clear"]),
            "blocker": geometry.get("blocker"),
            "geometry_age_ms": float(geometry.get("geometry_age_ms", 0)),
            "action_mm": geometry.get("action_mm"),
        }

    text = str(entry.get("derived", ""))
    found: dict[str, str] = {}
    for match in _DERIVED.finditer(text):
        found.update({key: value for key, value in match.groupdict().items() if value})
    return {
        **semantic,
        "reach_ok": found.get("reach", "ok") == "ok",
        "clearance_mm": float(found.get("clearance", 0.0)),
        "distance_mm": float(found.get("distance", 0.0)),
        "path_clear": found.get("path", "clear") == "clear",
        "blocker": None,
        "geometry_age_ms": float(found.get("geom", 0.0)),
        "action_mm": None,
    }


class RuleJudge:
    """요청 하나 → 10개 답. 상태가 없고 결정적이다.

    실행기·하네스와 공유하는 값(속도·힘 수준 수, 기하 나이 문턱)은 복사하지 않고 그 설정
    파일을 읽는다. 검사에서 바꿔 끼울 수 있게 dict로도 받는다.

    지시의 대상·목적지·제약은 상태의 **구조화된 목표**로 읽는다(:func:`read_goal`). 텍스트를
    파싱하는 것은 구조화된 목표가 없는 틱(D0 fixture 등)의 근사이고, 그때의 어휘는 상태의 물체
    설명이다 — 장면 설정의 어휘(`vocabulary_config`)는 선택이며 기본 설정은 요구하지 않는다.
    """

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        *,
        controller_config: dict[str, Any] | None = None,
        harness_config: dict[str, Any] | None = None,
        vocabulary_config: dict[str, Any] | None = None,
    ) -> None:
        self.config = copy.deepcopy(config or load_rule_judge_config())
        self.version = str(self.config.get("version", RULE_JUDGE_VERSION))
        self.main = self.config["main"]
        self.confidence = self.config["confidence"]
        self.thresholds = self.config["thresholds"]
        self.profiles = self.config["profiles"]

        controller = controller_config or load_controller_config(self.config["controller_config"])
        self.speed_levels = [str(index) for index in range(len(controller["speed_levels_m_s"]))]
        self.force_levels = [str(index) for index in range(len(controller["force_levels"]))]
        harness = harness_config or load_harness_config(self.config["harness_config"])
        self.max_geometry_age_ms = float(harness["candidates"]["max_geometry_age_ms"])
        # 텍스트 근사 경로의 추가 어휘. 설정 파일이 `vocabulary_config`를 적었을 때만 읽는다.
        if vocabulary_config is None and self.config.get("vocabulary_config"):
            vocabulary_config = yaml.safe_load(
                resolve_config_path(self.config["vocabulary_config"]).read_text(encoding="utf-8")
            )
        self.vocabulary_phrases = self._object_phrases(vocabulary_config) if vocabulary_config else ()
        self.constraint_markers = [
            str(marker) for marker in (self.config.get("instruction") or {}).get("constraint_markers") or ()
        ]

    @staticmethod
    def _object_phrases(vocabulary: dict[str, Any]) -> tuple[str, ...]:
        """지시문이 물체를 부르는 "<색> <형상>" 구절 전부 (장면 설정의 palette × shape_labels)."""
        spec = vocabulary["objects"]
        colours: list[str] = []
        for entry in spec["palette"]:
            colours.extend(str(entry[key]) for key in ("ko", "name") if entry.get(key))
        shapes = [str(label) for label in dict(spec["shape_labels"]).values()]
        return tuple(f"{colour} {shape}" for colour in colours for shape in shapes)

    def _phrases(self, state: dict[str, Any]) -> tuple[str, ...]:
        """텍스트 근사 경로의 물체 구절: 상태의 물체 설명 + (있으면) 어휘 설정."""
        described = tuple(
            str(entry["desc"]) for entry in state.get("objects") or () if entry.get("desc")
        )
        return described + tuple(phrase for phrase in self.vocabulary_phrases if phrase not in described)

    @classmethod
    def from_config_path(cls, path: str | Path = DEFAULT_CONFIG_PATH) -> RuleJudge:
        return cls(load_rule_judge_config(path))

    # ------------------------------------------------------------------

    def __call__(self, request: dict[str, Any]) -> dict[str, Any]:
        model = request.get("request", request)
        state = model["state"]
        candidates = list(model["candidates"]["q_main"])
        paths = list(model["candidates"].get("q_path") or [])
        values = {entry["id"]: candidate_values(entry, request) for entry in candidates}
        goal = read_goal(state)
        commitment = model.get("commitment")
        phase = str((commitment or {}).get("phase", "none"))
        if phase not in PHASES:
            phase = "none"

        return {
            "q_main": self._main(candidates, values, state, goal, model),
            "q_done": self._truth(self._goal_satisfied(state, goal)),
            "q_instr": self._truth(self._instruction_complete(state, goal)),
            "q_observe": self._truth(self._needs_observation(state, goal, phase)),
            "q_retry": self._truth(self._retry_ok(model)),
            "q_stop": self._truth(self._must_stop(state, goal)),
            "q_gripper": self._gripper(state, phase, commitment, values),
            "q_path": self._path(paths, values, commitment),
            "q_speed": self._speed(state, phase, commitment, values),
            "q_force": self._force(phase),
        }

    # -- 주 결정 ------------------------------------------------------------

    def _main(
        self,
        candidates: list[dict[str, Any]],
        values: dict[str, dict[str, Any]],
        state: dict[str, Any],
        goal: Goal,
        model: dict[str, Any],
    ) -> dict[str, float]:
        goal_target = goal.target_ref
        blockers = {
            values[entry["id"]]["blocker"]
            for entry in candidates
            if values[entry["id"]]["target"] == goal_target
        } - {None}
        failed = self._failed_approach(model, values)

        admissible: list[str] = []
        costs: dict[str, float] = {}
        for entry in candidates:
            value = values[entry["id"]]
            if self._admissible(entry["key"], value, goal, goal_target, blockers):
                admissible.append(entry["id"])
            costs[entry["id"]] = self._cost(entry["key"], value, failed)

        ids = [entry["id"] for entry in candidates]
        if not admissible:
            return {candidate: 1.0 / len(ids) for candidate in ids}

        floor = float(self.main["inadmissible_mass"])
        others = [candidate for candidate in ids if candidate not in admissible]
        share = (floor / len(others)) if others else 0.0
        budget = 1.0 - share * len(others)

        temperature = float(self.main["temperature"])
        best = min(costs[candidate] for candidate in admissible)
        weights = {
            candidate: math.exp(-(costs[candidate] - best) / temperature) for candidate in admissible
        }
        total = sum(weights.values())
        distribution = {candidate: share for candidate in others}
        distribution.update(
            {candidate: budget * weight / total for candidate, weight in weights.items()}
        )
        return _normalise({candidate: distribution[candidate] for candidate in ids})

    def _admissible(
        self,
        key: str,
        value: dict[str, Any],
        goal: Goal,
        goal_target: str | None,
        blockers: set[str],
    ) -> bool:
        """의미 적합성 — 지시·목적지·금지 조건 (docs/08 §7)."""
        if key in FIXED_KEYS:
            return True
        target, function = value["target"], value["function"]
        if target in goal.forbidden:
            return False
        if function == "push":
            # 목표 대상으로 가는 길을 막는 물체만 치운다.
            return target in blockers
        if goal_target and target != goal_target:
            return False
        if goal_target is None and goal.structured:
            # 구조화된 대상이 아직 보이지 않는다 — 다른 물체로 바꿔 타지 않는다 (관측이 먼저다).
            return False
        if goal.zone and value["destination"] not in (goal.zone, "none"):
            return False
        return True

    def _cost(self, key: str, value: dict[str, Any], failed: str | None) -> float:
        """고정 가중 기하 비용. 가중치·기준값은 설정에 있다 (docs/02 §9)."""
        if key in FIXED_KEYS:
            return float(self.main["fixed_cost"][key])
        weights = self.main["weights"]
        references = self.main["references"]
        clearance_reference = float(references["clearance_mm"])
        clearance = max(0.0, clearance_reference - value["clearance_mm"]) / clearance_reference
        distance = value["distance_mm"] / float(references["distance_mm"])
        return (
            float(weights["clearance"]) * clearance
            + float(weights["distance"]) * distance
            + float(weights["reach"]) * (0.0 if value["reach_ok"] else 1.0)
            + float(weights["blocked_path"]) * (0.0 if value["path_clear"] else 1.0)
            + float(weights["failed_approach"]) * (1.0 if failed and value["approach"] == failed else 0.0)
        )

    def _failed_approach(
        self, model: dict[str, Any], values: dict[str, dict[str, Any]]
    ) -> str | None:
        """직전에 실패한 접근 유형 (docs/08 §3.3의 실행 이력에서 읽는다)."""
        history = parse_exec_history(model.get("exec_history"))
        if not history or history.get("ack") in (None, "ok", "none"):
            return None
        value = values.get(str(history.get("main")))
        return value["approach"] if value else None

    # -- 게이팅 --------------------------------------------------------------

    def _truth(self, answer: bool) -> float:
        return float(self.confidence["high" if answer else "low"])

    @staticmethod
    def _target(state: dict[str, Any], goal: Goal) -> dict[str, Any] | None:
        """지시가 가리키는 물체 (추적 중일 때만)."""
        objects = state.get("objects") or ()
        return next((entry for entry in objects if str(entry["id"]) == goal.target_ref), None)

    def _goal_satisfied(self, state: dict[str, Any], goal: Goal) -> bool:
        """목표 영역 포함으로 판정한다. 모델의 답이 아니라 관측으로 본다."""
        target = self._target(state, goal)
        zone = next(
            (entry for entry in state.get("zones") or () if str(entry["id"]) == goal.zone),
            None,
        )
        if target is None or zone is None:
            return False
        if state["robot"].get("holding") == target["id"]:
            return False
        x0, y0, x1, y1 = [float(value) for value in zone["bounds_mm"]]
        x, y = float(target["pose_mm"][0]), float(target["pose_mm"][1])
        return min(x0, x1) <= x <= max(x0, x1) and min(y0, y1) <= y <= max(y0, y1)

    def _instruction_complete(self, state: dict[str, Any], goal: Goal) -> bool:
        """지시가 완결됐는가 (docs/08 §4 `q_instr`) — 대상의 현재 가시성과 무관하다.

        구조화된 목표가 있으면 그 필드로만 본다: 대상이 지목됐고(`target_desc`나 `target_ref`),
        목적지가 상태의 영역이며, 대상이 금지 물체가 아니다(모순). 텍스트 근사 경로에서는
        대상이 이름으로 불렸는가(상태의 물체 설명·어휘 구절), 목적지가 있는가(`target_zone`이거나
        영역 설명이 텍스트에 있는가), 제약 표지 앞에 구절이 있는가("그것은 건드리지 마라"처럼
        풀 수 없는 제약이 아닌가)를 본다.
        """
        zones = state.get("zones") or ()
        if goal.structured:
            target_named = bool(goal.target_desc or goal.target_ref)
            destination_known = goal.zone is not None and any(
                str(zone.get("id")) == goal.zone for zone in zones
            )
            consistent = goal.target_ref is None or goal.target_ref not in goal.forbidden
            return target_named and destination_known and consistent

        text = goal.text
        if not text:
            return False
        mentions = sorted(
            (position, phrase)
            for phrase in self._phrases(state)
            if (position := text.find(phrase)) >= 0
        )
        target_named = bool(goal.target_ref) or bool(mentions)
        destination_named = bool(goal.zone) or any(
            str(zone.get("desc", "")) and str(zone["desc"]) in text for zone in zones
        )
        constraints_parseable = all(
            self._constraint_subject(text, marker, mentions) is not None
            for marker in self.constraint_markers
            if marker in text
        )
        return target_named and destination_named and constraints_parseable

    @staticmethod
    def _constraint_subject(
        text: str, marker: str, mentions: list[tuple[int, str]]
    ) -> str | None:
        """제약 표지 바로 앞의 어휘 구절. 사이에 조사 정도만 있어야 한다("그것은 …"은 풀 수 없다)."""
        at = text.find(marker)
        before = [(position, phrase) for position, phrase in mentions if position < at]
        if not before:
            return None
        position, phrase = before[-1]
        between = text[position + len(phrase) : at].strip()
        return phrase if len(between) <= 2 else None

    def _needs_observation(self, state: dict[str, Any], goal: Goal, phase: str) -> bool:
        """관측을 더 얻어야 하는가 (docs/08 §4 `q_observe`).

        지시의 대상이 아직 관측되지 않았거나 대상 기하가 문턱보다 오래됐을 때다. 가시 비율만으로는
        요구하지 않는다. 파지·놓기 국면과 파지 중(팔이 대상을 가린다)에는 하네스의 실행 가능성
        문턱(`max_geometry_age_ms`)이 기준이다 — 들고 있는 물체의 기하 나이는 0이다.
        """
        target = self._target(state, goal)
        if target is None:
            return True
        age = float(target.get("age_ms", 0))
        contact = phase in _CONTACT_PHASES or state["robot"].get("holding") == target["id"]
        limit = self.max_geometry_age_ms if contact else float(self.thresholds["observe_geom_age_ms"])
        return age > limit

    def _retry_ok(self, model: dict[str, Any]) -> bool:
        """직전 실패와 같은 방식의 재시도가 적절한가.

        기준군이 보는 것은 재입력된 실행 이력뿐이다(docs/08 §3.3). 같은 방식의 연속 실패
        (`fails=`)가 `retry_max_same_approach` 안이면 한 번 더 해 볼 수 있고, 실패가 없으면 이
        질문의 근거도 없다. 횟수를 적지 않은 이력(다른 도구의 틱)은 첫 실패로 본다.
        """
        history = parse_exec_history(model.get("exec_history"))
        if not history or history.get("ack") in (None, "ok", "none"):
            return False
        try:
            fails = max(1, int(history.get("fails", 1)))
        except ValueError:
            fails = 1
        return fails <= int(self.thresholds["retry_max_same_approach"])

    def _must_stop(self, state: dict[str, Any], goal: Goal) -> bool:
        if float(state["robot"].get("contact_n") or 0.0) > float(self.thresholds["stop_force_n"]):
            return True
        if any(
            str(event.get("kind", "")).startswith("reflex") for event in state.get("events") or ()
        ):
            return True
        ee = [float(value) for value in state["robot"]["ee_pose_mm"]]
        limit = float(self.thresholds["stop_forbidden_mm"])
        for entry in state.get("objects") or ():
            if str(entry["id"]) not in goal.forbidden:
                continue
            if math.dist(ee, [float(value) for value in entry["pose_mm"]]) <= limit:
                return True
        return False

    # -- 부가 답 -------------------------------------------------------------

    def _spread(self, chosen: str, options: list[str]) -> dict[str, float]:
        """고른 수준에 설정된 질량을 주고 나머지를 고르게 나눈다."""
        mass = float(self.confidence["choice_mass"])
        rest = [option for option in options if option != chosen]
        share = (1.0 - mass) / len(rest) if rest else 0.0
        return _normalise({option: mass if option == chosen else share for option in options})

    def _gripper(
        self, state: dict[str, Any], phase: str, commitment: dict[str, Any] | None, values: dict[str, dict[str, Any]]
    ) -> dict[str, float]:
        """국면 프로파일의 그리퍼 상태. 파지 국면에서는 말단이 파지점에 와야 닫는다."""
        desired = str(self.profiles["gripper_by_phase"][phase])
        if desired == "current":
            desired = "closed" if state["robot"].get("holding") else "open"
        if phase == "grasp" and desired == "closed" and not state["robot"].get("holding"):
            value = values.get(str((commitment or {}).get("action_ref"))) if commitment else None
            point = (value or {}).get("action_mm")
            if point is not None:
                ee = [float(item) for item in state["robot"]["ee_pose_mm"]]
                if math.dist(ee, [float(item) for item in point]) > float(self.thresholds["grasp_ready_mm"]):
                    desired = "open"
        return self._spread(desired, ["open", "closed"])

    def _path(
        self,
        paths: list[dict[str, Any]],
        values: dict[str, dict[str, Any]],
        commitment: dict[str, Any] | None,
    ) -> dict[str, float]:
        """직진이 기본이고 막혔으면 첫 번째 비어 있는 경유점이다 (docs/02 §9)."""
        options = [entry["id"] for entry in paths]
        if not options:
            return {}
        by_kind: dict[str, list[str]] = {}
        for entry in paths:
            by_kind.setdefault(str(entry["kind"]), []).append(str(entry["id"]))

        value = values.get(str((commitment or {}).get("action_ref"))) if commitment else None
        if commitment is None or value is None or value["function"] is None:
            chosen = (by_kind.get("hold") or options)[0]
        elif value["path_clear"]:
            chosen = (by_kind.get("direct") or options)[0]
        elif by_kind.get("via"):
            chosen = by_kind["via"][0]
        else:
            chosen = (by_kind.get("retreat") or by_kind.get("hold") or options)[0]
        return self._spread(chosen, options)

    def _speed(
        self,
        state: dict[str, Any],
        phase: str,
        commitment: dict[str, Any] | None,
        values: dict[str, dict[str, Any]],
    ) -> dict[str, float]:
        level = int(self.profiles["speed_by_phase"][phase])
        if level and self._near_fragile(state, commitment, values):
            level = min(level, int(self.profiles["fragile_speed_cap"]))
        return self._spread(str(min(level, len(self.speed_levels) - 1)), self.speed_levels)

    def _force(self, phase: str) -> dict[str, float]:
        level = int(self.profiles["force_by_phase"][phase])
        return self._spread(str(min(level, len(self.force_levels) - 1)), self.force_levels)

    def _near_fragile(
        self,
        state: dict[str, Any],
        commitment: dict[str, Any] | None,
        values: dict[str, dict[str, Any]],
    ) -> bool:
        limit = float(self.thresholds["fragile_proximity_mm"])
        points = [[float(value) for value in state["robot"]["ee_pose_mm"]]]
        value = values.get(str((commitment or {}).get("action_ref"))) if commitment else None
        target = value["target"] if value else None
        for entry in state.get("objects") or ():
            if str(entry["id"]) == target:
                points.append([float(item) for item in entry["pose_mm"]])
        for entry in state.get("objects") or ():
            if "fragile" not in (entry.get("attributes") or ()) or str(entry["id"]) == target:
                continue
            pose = [float(item) for item in entry["pose_mm"]]
            if any(math.dist(pose, point) <= limit for point in points):
                return True
        return False


def _normalise(distribution: dict[str, float]) -> dict[str, float]:
    """소수 6자리로 적되 합을 1로 맞춘다.

    자른 자리의 나머지는 가장 큰 후보가 받는다. 확률 합은 계약 검사가 보는 값이므로
    (`contracts.PROB_SUM_TOL`) 반올림 잔차를 그대로 두지 않는다.
    """
    rounded = {key: round(value, 6) for key, value in distribution.items()}
    if not rounded:
        return rounded
    top = max(rounded, key=lambda key: (rounded[key], key))
    rounded[top] = round(rounded[top] + (1.0 - sum(rounded.values())), 6)
    return rounded


#: 공개 이름. 전문가도 같은 반올림 규칙으로 분포를 낸다.
normalise_distribution = _normalise

_DEFAULT: RuleJudge | None = None


def rule_judge(request: dict[str, Any], *, judge: RuleJudge | None = None) -> dict[str, Any]:
    """요청 하나에 대한 규칙 기준군의 10개 답 (모델 출력과 같은 형식)."""
    global _DEFAULT
    if judge is None:
        if _DEFAULT is None:
            _DEFAULT = RuleJudge()
        judge = _DEFAULT
    return judge(request)
