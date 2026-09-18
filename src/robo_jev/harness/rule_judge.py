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
from pathlib import Path
from typing import Any

import yaml

from robo_jev.harness.robot import parse_exec_history
from robo_jev.perception.pointworld import named_target
from robo_jev.sim.controller import resolve_config_path

__all__ = [
    "DEFAULT_CONFIG_PATH",
    "RULE_JUDGE_VERSION",
    "RuleJudge",
    "load_rule_judge_config",
    "rule_judge",
]

#: 규칙 버전. 레코드의 `versions.rules`에 들어간다.
RULE_JUDGE_VERSION = "rj0.1"

DEFAULT_CONFIG_PATH = "configs/harness/rule_judge_v0.yaml"

_FIXED_KEYS = ("observe", "hold", "replan")

#: 후보 설명의 기하 값 (`reach ok, clr 41mm, d 320mm, path clear, geom 120ms`).
_DERIVED = re.compile(
    r"reach (?P<reach>ok|no)|clr (?P<clearance>-?\d+)mm|d (?P<distance>-?\d+)mm|"
    r"path (?P<path>clear|blocked)|geom (?P<geom>-?\d+)ms"
)

_PHASES = ("approach", "grasp", "lift", "transport", "place", "push", "none")


def load_rule_judge_config(path: str | Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    return yaml.safe_load(resolve_config_path(path).read_text(encoding="utf-8"))


class RuleJudge:
    """요청 하나 → 10개 답. 상태가 없고 결정적이다."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = copy.deepcopy(config or load_rule_judge_config())
        self.version = str(self.config.get("version", RULE_JUDGE_VERSION))
        self.main = self.config["main"]
        self.confidence = self.config["confidence"]
        self.thresholds = self.config["thresholds"]
        self.profiles = self.config["profiles"]

    @classmethod
    def from_config_path(cls, path: str | Path = DEFAULT_CONFIG_PATH) -> RuleJudge:
        return cls(load_rule_judge_config(path))

    # ------------------------------------------------------------------

    def __call__(self, request: dict[str, Any]) -> dict[str, Any]:
        model = request.get("request", request)
        state = model["state"]
        candidates = list(model["candidates"]["q_main"])
        paths = list(model["candidates"].get("q_path") or [])
        values = {
            entry["id"]: self._values(entry, request) for entry in candidates
        }
        goal = state.get("goal") or {}
        commitment = model.get("commitment")
        phase = str((commitment or {}).get("phase", "none"))
        if phase not in _PHASES:
            phase = "none"

        return {
            "q_main": self._main(candidates, values, state, goal, model),
            "q_done": self._truth(self._goal_satisfied(state, goal)),
            "q_instr": self._truth(self._instruction_complete(state, goal)),
            "q_observe": self._truth(self._needs_observation(state, goal, values)),
            "q_retry": self._truth(self._retry_ok(model)),
            "q_stop": self._truth(self._must_stop(state, goal)),
            "q_gripper": self._gripper(state, phase),
            "q_path": self._path(paths, values, commitment),
            "q_speed": self._speed(state, phase, commitment, values),
            "q_force": self._force(phase),
        }

    # -- 후보 값 ------------------------------------------------------------

    def _values(self, entry: dict[str, Any], request: dict[str, Any]) -> dict[str, Any]:
        """후보 하나의 기하 값과 의미 조각.

        하네스 블록이 있으면 그것을, 없으면 모델이 보는 `derived` 문자열을 읽는다.
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
        block = (request.get("harness") or {}).get("candidates") or {}
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
        }

    # -- 주 결정 ------------------------------------------------------------

    def _main(
        self,
        candidates: list[dict[str, Any]],
        values: dict[str, dict[str, Any]],
        state: dict[str, Any],
        goal: dict[str, Any],
        model: dict[str, Any],
    ) -> dict[str, float]:
        target = self._target(state, goal)
        goal_target = str(target["id"]) if target else None
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
        goal: dict[str, Any],
        goal_target: str | None,
        blockers: set[str],
    ) -> bool:
        """의미 적합성 — 지시·목적지·금지 조건 (docs/08 §7)."""
        if key in _FIXED_KEYS:
            return True
        target, function = value["target"], value["function"]
        if target in (goal.get("forbidden_contact") or ()):
            return False
        if function == "push":
            # 목표 대상으로 가는 길을 막는 물체만 치운다.
            return target in blockers
        if goal_target and target != goal_target:
            return False
        if goal.get("target_zone") and value["destination"] not in (goal["target_zone"], "none"):
            return False
        return True

    def _cost(self, key: str, value: dict[str, Any], failed: str | None) -> float:
        """고정 가중 기하 비용. 가중치·기준값은 설정에 있다 (docs/02 §9)."""
        if key in _FIXED_KEYS:
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

    def _target(self, state: dict[str, Any], goal: dict[str, Any]) -> dict[str, Any] | None:
        """지시가 가리키는 물체. 구조화된 `target_ref`가 먼저이고, 없으면 이름으로 푼다."""
        objects = state.get("objects") or ()
        reference = goal.get("target_ref") or named_target(objects, str(goal.get("text", "")))
        return next((entry for entry in objects if str(entry["id"]) == reference), None)

    def _goal_satisfied(self, state: dict[str, Any], goal: dict[str, Any]) -> bool:
        """목표 영역 포함으로 판정한다. 모델의 답이 아니라 관측으로 본다."""
        target = self._target(state, goal)
        zone = next(
            (
                entry
                for entry in state.get("zones") or ()
                if str(entry["id"]) == goal.get("target_zone")
            ),
            None,
        )
        if target is None or zone is None:
            return False
        if state["robot"].get("holding") == target["id"]:
            return False
        x0, y0, x1, y1 = [float(value) for value in zone["bounds_mm"]]
        x, y = float(target["pose_mm"][0]), float(target["pose_mm"][1])
        return min(x0, x1) <= x <= max(x0, x1) and min(y0, y1) <= y <= max(y0, y1)

    def _instruction_complete(self, state: dict[str, Any], goal: dict[str, Any]) -> bool:
        """대상·목적지가 하나로 풀리는가 (docs/08 §4 `q_instr`)."""
        return bool(goal.get("text")) and self._target(state, goal) is not None and bool(
            goal.get("target_zone")
        )

    def _needs_observation(
        self, state: dict[str, Any], goal: dict[str, Any], values: dict[str, dict[str, Any]]
    ) -> bool:
        target = self._target(state, goal)
        if target is None:
            return True
        if float(target.get("visible_ratio", 1.0)) < float(
            self.thresholds["observe_visible_ratio"]
        ):
            return True
        return float(target.get("age_ms", 0)) > float(self.thresholds["observe_geom_age_ms"])

    def _retry_ok(self, model: dict[str, Any]) -> bool:
        """직전 실패와 같은 방식의 재시도가 적절한가.

        기준군이 보는 것은 재입력된 실행 이력뿐이다(docs/08 §3.3). 실패가 한 번이면
        같은 방식을 한 번 더 해 볼 수 있고, 실패가 없으면 이 질문의 근거도 없다.
        """
        history = parse_exec_history(model.get("exec_history"))
        if not history or history.get("ack") in (None, "ok", "none"):
            return False
        return int(self.thresholds["retry_max_same_approach"]) >= 1

    def _must_stop(self, state: dict[str, Any], goal: dict[str, Any]) -> bool:
        if float(state["robot"].get("contact_n") or 0.0) > float(self.thresholds["stop_force_n"]):
            return True
        if any(
            str(event.get("kind", "")).startswith("reflex") for event in state.get("events") or ()
        ):
            return True
        ee = [float(value) for value in state["robot"]["ee_pose_mm"]]
        limit = float(self.thresholds["stop_forbidden_mm"])
        for entry in state.get("objects") or ():
            if str(entry["id"]) not in (goal.get("forbidden_contact") or ()):
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

    def _gripper(self, state: dict[str, Any], phase: str) -> dict[str, float]:
        desired = str(self.profiles["gripper_by_phase"][phase])
        if desired == "current":
            desired = "closed" if state["robot"].get("holding") else "open"
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
        return self._spread(str(level), [str(index) for index in range(4)])

    def _force(self, phase: str) -> dict[str, float]:
        level = int(self.profiles["force_by_phase"][phase])
        return self._spread(str(level), [str(index) for index in range(3)])

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


_DEFAULT: RuleJudge | None = None


def rule_judge(request: dict[str, Any], *, judge: RuleJudge | None = None) -> dict[str, Any]:
    """요청 하나에 대한 규칙 기준군의 10개 답 (모델 출력과 같은 형식)."""
    global _DEFAULT
    if judge is None:
        if _DEFAULT is None:
            _DEFAULT = RuleJudge()
        judge = _DEFAULT
    return judge(request)
