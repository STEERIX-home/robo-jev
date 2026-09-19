"""스크립트 전문가 v0 — D1 로봇 에피소드의 정책이자 라벨 원천 (docs/08 §7·§9, docs/04 §3).

전문가는 모델 자리에 서서 **같은 요청**을 받고 **같은 형식**으로 10개 답을 낸다. 규칙
기준군(:mod:`robo_jev.harness.rule_judge`)과 다른 점은 셋이다.

* 비용 순위가 아니라 **구조화된 목표를 실현하는 결합 후보**를 고른다: 대상 → 목표 영역의
  파지(들고 있으면 놓기)이고, 그 파지가 목록에 없을 때만 대상을 영역 쪽으로 미는 후보다.
  실현할 후보가 없으면 다른 물체를 집지 않고 `hold`한다(근거는 낮은 신뢰도로 남는다).
* 같은 목표 아래에서는 국면이 바뀌어도 commitment를 지킨다.
* 답마다 **근거 코드**(`expert_meta`)를 남겨 라벨의 `rule`이 된다.
* rollout이 없는 틱의 `q_main` 라벨은 단일 정답이 아니라 **비용 허용 집합**이다(docs/08 §7, 계약 v0.3):
  A = {선택} ∪ {적합·실행 가능 후보 중 플래너 비용이 선택의 (1 + τ) 안인 것}, τ = `labels.cost_tolerance`.
  플래너 비용(:meth:`Expert.plan_cost_mm`)은 남은 명령 경로 길이의 추정(mm)이며 새 물리는 없다. 실행기
  사정으로 `hold`로 물러난 퇴화 틱은 hold∉A 규칙 — A = {hold}, `unknown` = 적합 집합(판단하지 않음).

**정보 경계.** 답은 하네스 요청의 모델 입력(`request["request"]`: 상태·실행 이력·commitment·
후보)과 commitment의 함수다. 하네스 블록(`request["harness"]`)도, 시뮬레이터 관측도 읽지
않는다 — `observation` 인자는 인터페이스(docs/06)의 자리이며 v0는 쓰지 않는다. 가려진 물체의
참값을 바꿔도 답이 같다는 검사가 이를 고정한다(tests/test_expert.py).

수치는 전부 `configs/sim/expert_v0.yaml`에 있다.
"""

from __future__ import annotations

import copy
import math
from pathlib import Path
from typing import Any

import yaml

from robo_jev.contracts import AUX_QUESTIONS, PHASES
from robo_jev.harness.robot import CONTACT_PHASES, FIXED_KEYS, load_harness_config, parse_exec_history
from robo_jev.harness.rule_judge import Goal, candidate_values, normalise_distribution, read_goal
from robo_jev.perception.pointworld import circumradius_mm, segment_point_distance_mm
from robo_jev.sim.controller import load_controller_config, resolve_config_path

__all__ = [
    "DEFAULT_CONFIG_PATH",
    "DEGENERATE_REASONS",
    "EXPERT_VERSION",
    "GATE_REASONS",
    "Expert",
    "load_expert_config",
]

#: 전문가 버전. 레코드의 `versions.expert`에 들어간다. e0.3 = 계약 v0.3(비용 허용 집합, hold∉A, readiness 관측 게이트,
#: 4조각 결합 키).
EXPERT_VERSION = "e0.3"

DEFAULT_CONFIG_PATH = "configs/sim/expert_v0.yaml"

#: 게이팅 질문과 그 라벨 규칙 이름 (docs/08 §7 표).
_GATE_RULES = {
    "q_done": "goal-zone-containment-v0",
    "q_instr": "goal-completeness-v0",
    "q_observe": "target-tracked-and-geometry-age-v0",
    "q_retry": "same-way-failure-count-v0",
    "q_stop": "force-reflex-forbidden-contact-v0",
}

#: 게이트가 주 결정을 정한 이유 — 답은 규칙 후보(hold·replan·observe) 하나이고 허용 집합도 그것뿐이다.
GATE_REASONS = ("goal_done", "instruction_incomplete", "observe_target")

#: 실행기 사정으로 `hold`로 물러난 퇴화 틱의 이유 — hold∉A 규칙(docs/08 §7): A = {hold}, `unknown` = 적합 집합,
#: `label_confidence: low`(설정의 weight). 재시도 차단·실행 불가·목표 후보 없음.
DEGENERATE_REASONS = ("way_retry_blocked", "not_executable", "goal_candidate_missing")

#: 관측 게이트를 기하 나이로 보지 않는 국면 (docs/08 §4 `q_observe`, §5.0): 팔이 대상을 가리는 접촉 국면(파지·놓기·
#: 밀기)과 파지 중에는 실행기의 readiness가 시점을 정한다 — 하네스와 같은 면제이며 단일 출처는 하네스다.
_CONTACT_PHASES = CONTACT_PHASES

#: 밀기 방향 벡터 (로봇 기준 xy). 하네스의 의미 키와 같은 이름이다.
_PUSH_VECTORS = {"+x": (1.0, 0.0), "-x": (-1.0, 0.0), "+y": (0.0, 1.0), "-y": (0.0, -1.0)}


def load_expert_config(path: str | Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    return yaml.safe_load(resolve_config_path(path).read_text(encoding="utf-8"))


class Expert:
    """요청 하나 → 10개 답 + 국면 + 근거. 상태가 없고 결정적이다."""

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        *,
        controller_config: dict[str, Any] | None = None,
        harness_config: dict[str, Any] | None = None,
    ) -> None:
        self.config = copy.deepcopy(config or load_expert_config())
        self.version = str(self.config.get("version", EXPERT_VERSION))
        self.confidence = self.config["confidence"]
        self.goal_config = self.config["goal"]
        self.thresholds = self.config["thresholds"]
        self.profiles = self.config["profiles"]
        self.stop_config = self.config.get("stop") or {}
        self.label_config = self.config.get("labels") or {}
        self.label_source = str(self.label_config.get("source", "expert_v0"))

        controller = controller_config or load_controller_config(self.config["controller_config"])
        self.speed_levels = [str(index) for index in range(len(controller["speed_levels_m_s"]))]
        self.force_levels = [str(index) for index in range(len(controller["force_levels"]))]
        harness = harness_config or load_harness_config(self.config["harness_config"])
        self.grasp_depth_mm = float(harness["candidates"]["grasp_depth_mm"])
        self.approach_clearance_mm = float(harness["candidates"]["approach_clearance_mm"])
        self.push_segment_mm = float(harness["candidates"]["push_segment_mm"])
        self.push_contact_mm = float(harness["candidates"]["push_contact_mm"])
        self.lift_height_mm = float(harness["phases"]["lift_height_mm"])
        self.planner_margin_mm = float(harness["planner"]["margin_mm"])
        self.forbidden_margin_mm = float(harness["planner"]["forbidden_margin_mm"])
        self.side_offset_mm = float(harness["planner"]["side_offset_mm"])
        self.cost_tolerance = float(self.label_config.get("cost_tolerance", 0.15))
        if self.cost_tolerance < 0:
            raise ValueError(f"labels.cost_tolerance: 0 이상이어야 한다 (받은 값: {self.cost_tolerance})")

    @classmethod
    def from_config_path(cls, path: str | Path = DEFAULT_CONFIG_PATH) -> Expert:
        return cls(load_expert_config(path))

    # ------------------------------------------------------------------

    def act(
        self,
        request: dict[str, Any],
        commitment: dict[str, Any] | None,
        observation: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """10개 답(모델 출력 형식) + `phase` + `expert_meta`.

        `commitment`는 하네스가 든 commitment다. 결정은 요청에 실린 투영(그 틱의 목록에 있는
        것)을 기준으로 하고, 요청에 없으면 하네스 commitment의 후보가 목록에 있을 때만 쓴다.
        `observation`은 쓰지 않는다(위 모듈 설명).
        """
        del observation  # 정보 경계: 시뮬레이터 관측은 답의 입력이 아니다
        model = request.get("request", request)
        state = model["state"]
        candidates = list(model["candidates"]["q_main"])
        paths = list(model["candidates"].get("q_path") or [])
        values = {entry["id"]: candidate_values(entry) for entry in candidates}
        goal = read_goal(state)
        committed = self._committed(model, commitment, values)
        phase = str((committed or {}).get("phase", "none"))
        if phase not in PHASES:
            phase = "none"

        gates = self._gates(state, goal, phase, model)
        main = self._main(candidates, values, state, goal, committed, gates, paths, model)
        aux = self._aux(state, goal, phase, committed, values, paths) if committed else None

        answers: dict[str, Any] = {
            "q_main": self._spread(main["choice"], [entry["id"] for entry in candidates]),
            "q_done": self._truth(gates["q_done"]["value"]),
            "q_instr": self._truth(gates["q_instr"]["value"]),
            "q_observe": self._truth(gates["q_observe"]["value"]),
            "q_retry": self._truth(gates["q_retry"]["value"]),
            "q_stop": self._truth(gates["q_stop"]["value"]),
        }
        if aux is not None:
            answers["q_gripper"] = self._spread(aux["gripper"]["desired"], ["open", "closed"])
            answers["q_path"] = self._spread(aux["path"]["choice"], [entry["id"] for entry in paths])
            answers["q_speed"] = self._spread(aux["speed"]["level"], self.speed_levels)
            answers["q_force"] = self._spread(aux["force"]["level"], self.force_levels)
        else:
            # commitment가 없는 틱: 부가 질문은 hold 기준이다 (docs/08 §4). 그리퍼는 현재 상태.
            gripper = "closed" if state["robot"].get("holding") else "open"
            hold = next((entry["id"] for entry in paths if entry.get("kind") == "hold"), None)
            answers["q_gripper"] = self._spread(gripper, ["open", "closed"])
            answers["q_path"] = self._spread(hold, [entry["id"] for entry in paths])
            answers["q_speed"] = self._spread("0", self.speed_levels)
            answers["q_force"] = self._spread("0", self.force_levels)

        answers["phase"] = phase
        answers["expert_meta"] = {
            "version": self.version,
            "goal": {
                "structured": goal.structured,
                "target_ref": goal.target_ref,
                "target_desc": goal.target_desc,
                "zone": goal.zone,
                "version": goal.version,
            },
            "main": main,
            "gates": gates,
            "aux": aux,
        }
        return answers

    # -- commitment -----------------------------------------------------------

    @staticmethod
    def _committed(
        model: dict[str, Any], commitment: dict[str, Any] | None, values: dict[str, dict[str, Any]]
    ) -> dict[str, Any] | None:
        projected = model.get("commitment")
        if projected and projected.get("action_ref") in values:
            return projected
        if commitment and commitment.get("action_ref") in values:
            return {
                "action_ref": commitment["action_ref"],
                "key": commitment.get("key", ""),
                "phase": commitment.get("phase", "none"),
                "held_ticks": int(commitment.get("held_ticks", 0)),
                "last_switch_tick": int(commitment.get("last_switch_tick", 0)),
            }
        return None

    # -- 주 결정 ---------------------------------------------------------------

    def _main(
        self,
        candidates: list[dict[str, Any]],
        values: dict[str, dict[str, Any]],
        state: dict[str, Any],
        goal: Goal,
        committed: dict[str, Any] | None,
        gates: dict[str, dict[str, Any]],
        paths: list[dict[str, Any]],
        model: dict[str, Any],
    ) -> dict[str, Any]:
        """구조화된 목표를 실현하는 후보 (docs/08 §7의 의미 적합성 → 전문가 선택 → commitment).

        두 집합을 따로 든다. **`admissible`(의미 적합성)**은 지시·목적지·금지 조건에서만 나온다 — 목표
        대상을 목표 영역으로 옮기는 파지·놓기, 대상을 영역 쪽으로 미는 밀기, 파지 하강을 막는 평범한
        이웃을 대상에서 멀리 미는 밀기. **선택(`choice`)**은 그 안에서 실행기 역량(설정의 밀기 방향),
        접촉점 도달성, 하네스의 일시적 재시도 차단으로 거른 것 가운데 고른다. 거른 이유는 `excluded`에
        남는다. 걸러서 아무것도 남지 않으면 `hold`이되 적합 집합은 그대로다 — 라벨의
        `semantic_admissible`이 실행기 사정으로 줄면 안 되기 때문이다.
        """
        ids = [entry["id"] for entry in candidates]
        keys = {entry["id"]: str(entry.get("key", "")) for entry in candidates}
        fixed = {keys[candidate]: candidate for candidate in ids if keys[candidate] in FIXED_KEYS}
        holding = state["robot"].get("holding")

        blocked_ways = self._retry_blocked_ways(model, values)
        committed_id = committed["action_ref"] if committed else None

        def decision(
            choice: str,
            reason: str,
            admissible: list[str],
            confidence: str = "high",
            excluded: dict[str, str] | None = None,
        ) -> dict[str, Any]:
            """결정 하나 + 라벨의 허용 집합(docs/08 §7).

            게이트 이유는 규칙 후보 하나(A = {choice}), 퇴화 이유는 hold∉A 규칙(A = {hold}, unknown = 적합 집합),
            그 밖의 결합 결정은 비용 허용 집합 — A = {choice} ∪ {적합·실행 가능 후보 중 cost ≤ cost(choice)(1+τ)},
            단 지킨 commitment가 최선 비용의 (1+τ) 안이면 그것만(commitment 규칙 (3)); 신뢰도는 medium(비용 근거).
            """
            excluded = dict(excluded or {})
            usable = [candidate for candidate in admissible if candidate not in excluded]
            allowed, unknown, costs = [choice], [], {}
            if reason in DEGENERATE_REASONS:
                unknown = [candidate for candidate in admissible if candidate != choice]
                confidence = "low"
            elif reason not in GATE_REASONS:
                costs = {candidate: self.plan_cost_mm(values[candidate], state, goal) for candidate in usable}
                if choice not in costs:
                    costs[choice] = self.plan_cost_mm(values[choice], state, goal)
                tolerance = 1.0 + self.cost_tolerance
                kept_commitment = choice == committed_id and costs[choice] <= min(costs.values()) * tolerance
                if not kept_commitment:
                    allowed = sorted(
                        {choice} | {c for c in usable if costs[c] <= costs[choice] * tolerance}, key=ids.index
                    )
                confidence = "medium"
            return {
                "choice": choice,
                "key": keys.get(choice),
                "reason": reason,
                "admissible": list(admissible),
                "allowed": allowed,
                "unknown": unknown,
                "costs_mm": {candidate: round(cost, 1) for candidate, cost in costs.items()},
                "excluded": excluded,
                "confidence": confidence,
                "blocked_ways": sorted(blocked_ways),
            }

        def open_way(candidate: str) -> bool:
            value = values[candidate]
            return value["function"] is None or f"{value['function']}:{value['target']}:{value['approach']}" not in blocked_ways

        def executable(options: list[str]) -> tuple[list[str], dict[str, str]]:
            """적합 후보 가운데 지금 고를 수 있는 것과, 뺀 것의 이유. 선택에만 쓴다."""
            usable: list[str] = []
            excluded: dict[str, str] = {}
            for candidate in options:
                value = values[candidate]
                if not open_way(candidate):
                    excluded[candidate] = "retry_blocked"
                elif value["function"] == "push" and not self._push_allowed(value):
                    excluded[candidate] = "push_direction"
                elif value["function"] == "push" and not self._contact_point_free(
                    state, value["target"], _PUSH_VECTORS[str(value["approach"])]
                ):
                    excluded[candidate] = "contact_point"
                else:
                    usable.append(candidate)
            return usable, excluded

        def fallback(admissible: list[str], excluded: dict[str, str]) -> dict[str, Any]:
            """고를 수 있는 것이 없다: 적합 후보가 아예 없으면 `goal_candidate_missing`, 있는데 실행기
            사정이면 그 이유로 `hold` (신뢰도 low). 적합 집합은 줄이지 않는다."""
            choice = fixed.get("hold", ids[0])
            if not admissible:
                return decision(choice, "goal_candidate_missing", [], confidence="low")
            reason = "way_retry_blocked" if "retry_blocked" in excluded.values() else "not_executable"
            return decision(choice, reason, admissible, confidence="low", excluded=excluded)

        if gates["q_done"]["value"]:
            choice = fixed.get("hold", ids[0])
            return decision(choice, "goal_done", [choice])
        if not gates["q_instr"]["value"]:
            choice = fixed.get("replan", ids[0])
            return decision(choice, "instruction_incomplete", [choice])
        if gates["q_observe"]["value"]:
            choice = fixed.get("hold" if holding else "observe", ids[0])
            return decision(choice, "observe_target", [choice])

        target, zone = goal.target_ref, goal.zone
        has_via = any(entry.get("kind") == "via" for entry in paths)

        if holding is not None and holding != target:
            # 손에 든 것이 지금의 대상이 아니다(지시가 바뀌었다). 먼저 놓아야 새 대상을 집을 수 있다 —
            # 목표 영역이 있으면 거기에, 아니면 아무 영역에.
            places = [
                candidate for candidate in ids
                if values[candidate]["function"] == "place" and values[candidate]["target"] == holding
            ]
            if places:
                usable, excluded = executable(places)
                if committed_id in usable:
                    return decision(committed_id, "keep_commitment", places, excluded=excluded)
                if not usable:
                    return fallback(places, excluded)
                in_zone = [candidate for candidate in usable if values[candidate]["destination"] == zone]
                choice = self._preferred(in_zone or usable, values, keys)
                return decision(choice, "release_held_object", places, excluded=excluded)

        realising = [
            candidate
            for candidate in ids
            if values[candidate]["function"] in ("grasp", "place")
            and values[candidate]["target"] == target
            and values[candidate]["destination"] == zone
            and target not in goal.forbidden
        ]
        blockers = self._descent_blockers(state, goal) if holding is None else set()

        # 목표에 맞는 밀기(대상을 영역 쪽으로, 또는 막는 이웃을 대상에서 멀리)는 하네스가 완료를 판정할
        # 때까지 지킨다 — 상한 때문에 파지 후보가 틱마다 나타났다 사라져도 밀기와 파지를 오가지 않는다.
        # 그 밀기의 접근이 막혔고 경유점도 없으면 같은 목적의 다른 방향을 고른다.
        if committed_id is not None and values[committed_id]["function"] == "push":
            value = values[committed_id]
            if value["target"] == target:
                pushes = self._pushes_toward_zone(ids, values, state, target, zone)
            elif value["target"] in blockers:
                pushes = self._blocker_pushes(ids, values, state, goal)
            else:
                pushes = []
            usable, excluded = executable(pushes)
            if committed_id in usable:
                if value["path_clear"] or has_via:
                    return decision(committed_id, "keep_commitment", realising + pushes, excluded=excluded)
                choice = self._preferred(usable, values, keys, push_order=True)
                reason = "keep_commitment" if choice == committed_id else "push_redirect"
                return decision(choice, reason, realising + pushes, excluded=excluded)

        if realising:
            usable, excluded = executable(realising)
            if committed_id in usable:
                # 파지가 막혔고 경유점도 없으면(하강 구간의 이웃) 평범한 이웃을 대상에서 밀어낸다 —
                # "밀기는 파지가 불가능할 때만". 물러났다 다가가는 반복을 끊는다.
                value = values[committed_id]
                if not value["path_clear"] and not has_via and holding is None and self.goal_config.get("push_blockers", True):
                    pushes = self._blocker_pushes(ids, values, state, goal)
                    usable_pushes, push_excluded = executable(pushes)
                    excluded = {**excluded, **push_excluded}
                    if usable_pushes:
                        choice = self._preferred(usable_pushes, values, keys, push_order=True)
                        return decision(choice, "push_blocker", realising + pushes, excluded=excluded)
                    return decision(committed_id, "keep_commitment", realising + pushes, excluded=excluded)
                return decision(committed_id, "keep_commitment", realising, excluded=excluded)
            if not usable:
                # 실행기 사정(재시도 차단)으로 지금은 고를 수 없다. 차단은 한 틱이므로 기다린다 — 적합
                # 집합 밖의 밀기로 갈아타지 않는다(라벨의 정답이 적합 집합 밖에 서면 안 된다).
                return fallback(realising, excluded)
            preferred_function = "place" if holding == target else "grasp"
            choice = self._preferred(
                [c for c in usable if values[c]["function"] == preferred_function] or usable, values, keys
            )
            return decision(
                choice, "goal_place" if values[choice]["function"] == "place" else "goal_grasp", realising, excluded=excluded
            )

        pushes = self._pushes_toward_zone(ids, values, state, target, zone) if self.goal_config.get(
            "push_when_grasp_unavailable", True
        ) else []
        usable, excluded = executable(pushes)
        if usable:
            choice = self._preferred(usable, values, keys, push_order=True)
            return decision(choice, "push_toward_zone", pushes, excluded=excluded)
        return fallback(pushes, excluded)

    def _retry_blocked_ways(self, model: dict[str, Any], values: dict[str, dict[str, Any]]) -> set[str]:
        """직전 틱의 실패가 한계 횟수를 넘긴 방식(`기능:대상:접근`). 하네스가 이 틱에 그 방식을 막는다
        (`q_retry` 거짓) — 같은 답을 내면 남은 후보 중 임의의 것이 채택되므로 전문가가 먼저 비켜 준다."""
        history = parse_exec_history(model.get("exec_history"))
        if not history or history.get("ack") in (None, "ok", "none"):
            return set()
        try:
            fails = max(1, int(history.get("fails", 1)))
        except ValueError:
            fails = 1
        if fails <= int(self.thresholds["retry_max_same_approach"]):
            return set()
        value = values.get(str(history.get("main")))
        if not value or value["function"] is None:
            return set()
        return {f"{value['function']}:{value['target']}:{value['approach']}"}

    def _preferred(
        self,
        options: list[str],
        values: dict[str, dict[str, Any]],
        keys: dict[str, str],
        *,
        push_order: bool = False,
    ) -> str:
        """경로가 비어 있는 것 → (밀기는 이득이 큰 것) → 의미 키 순."""

        def rank(candidate: str) -> tuple:
            value = values[candidate]
            path = 0 if value["path_clear"] else 1
            gain = -float(value.get("push_gain_mm", 0.0)) if push_order else 0.0
            # 밀기는 접근이 비어 있는 것이 먼저다 — 대상에서 곧장 멀어지는 방향은 접촉점이 대상의 구 안이다.
            return (path, gain, keys[candidate])

        return min(options, key=rank)

    # -- 플래너 비용 (docs/08 §7 비키프레임 허용 집합) ---------------------------------

    def plan_cost_mm(self, value: dict[str, Any], state: dict[str, Any], goal: Goal) -> float:
        """결합 후보 하나의 플래너 비용 — **남은 명령 경로 길이의 추정(mm)**. 새 물리는 없다.

        하네스가 후보 줄에 실은 국면 목표점까지의 거리(`d`)에 그 뒤에 남는 국면의 구간을 하네스의 같은 수치로
        더한다: 파지(아직 안 들었으면) 하강(접근 여유 + 파지 깊이) + 들기(`lift_height_mm`) + 대상→목적지 영역 중심의
        xy 이동 + 놓기 하강(접근 여유); 들고 있는 대상의 파지·놓기는 말단→영역 중심 xy + 놓기 하강; 밀기는 접촉점까지
        `d` 뒤에 남은 영역 거리를 구간(`push_segment_mm`) 수로 환산하고 구간 사이의 재접근을 더한다(막는 이웃 밀기는
        한 구간). 막힌 직선 구간은 국소 플래너의 옆 우회 폭(`side_offset_mm`)의 두 배를 더한다. 고정 후보는 0이다.
        """
        function = value.get("function")
        if function is None:
            return 0.0
        cost = float(value.get("distance_mm", 0.0))
        if not value.get("path_clear", True):
            cost += 2.0 * self.side_offset_mm
        target = next((item for item in state.get("objects") or () if str(item["id"]) == value["target"]), None)
        pose = [float(item) for item in target["pose_mm"]] if target is not None else None
        ee = [float(item) for item in state["robot"]["ee_pose_mm"]]
        holding = state["robot"].get("holding")
        zone = next(
            (item for item in state.get("zones") or () if str(item["id"]) == str(value.get("destination"))), None
        )
        centre = _zone_centre(zone["bounds_mm"]) if zone is not None else None

        if function == "push":
            bounds = next((item["bounds_mm"] for item in state.get("zones") or () if str(item["id"]) == goal.zone), None)
            if value["target"] == goal.target_ref and pose is not None and bounds is not None:
                remaining = _zone_distance_mm(pose, bounds)
                segments = max(1, math.ceil(remaining / self.push_segment_mm)) if self.push_segment_mm > 0 else 1
            else:
                segments = 1  # 막는 이웃 밀기: 한 구간
            reapproach = (segments - 1) * (self.approach_clearance_mm + self.push_contact_mm)
            return cost + segments * self.push_segment_mm + reapproach

        if holding == value["target"]:
            # 들고 있는 대상(진행 중인 파지·놓기): 남은 것은 영역까지의 xy 이동과 놓기 하강이다.
            if centre is not None:
                cost += math.dist(ee[:2], centre)
            return cost + self.approach_clearance_mm

        # 아직 들지 않은 파지: 하강 → 들기 → 이동 → 놓기 하강.
        cost += self.approach_clearance_mm + self.grasp_depth_mm + self.lift_height_mm
        if pose is not None and centre is not None:
            cost += math.dist(pose[:2], centre)
        return cost + self.approach_clearance_mm

    def _pushes_toward_zone(
        self,
        ids: list[str],
        values: dict[str, dict[str, Any]],
        state: dict[str, Any],
        target: str | None,
        zone: str | None,
    ) -> list[str]:
        """대상을 목표 영역 쪽으로 미는 후보(의미 적합성: 목적지 조건). 한 구간을 민 예측 자세가 영역까지의
        거리를 문턱 이상 줄여야 한다. 실행기 역량·접촉점 도달성은 여기서 보지 않는다 — 선택의 몫이다."""
        if target is None or zone is None:
            return []
        entry = next((item for item in state.get("objects") or () if str(item["id"]) == target), None)
        bounds = next((item["bounds_mm"] for item in state.get("zones") or () if str(item["id"]) == zone), None)
        if entry is None or bounds is None:
            return []
        pose = [float(value) for value in entry["pose_mm"]]
        now = _zone_distance_mm(pose, bounds)
        gain_needed = float(self.goal_config.get("push_min_gain_mm", 0.0))
        improving: list[str] = []
        for candidate in ids:
            value = values[candidate]
            if value["function"] != "push" or value["target"] != target:
                continue
            vector = _PUSH_VECTORS.get(str(value["approach"]))
            if vector is None:
                continue
            predicted = [pose[0] + vector[0] * self.push_segment_mm, pose[1] + vector[1] * self.push_segment_mm]
            gain = now - _zone_distance_mm(predicted, bounds)
            if gain >= gain_needed:
                value["push_gain_mm"] = gain
                improving.append(candidate)
        return improving

    def _descent_blockers(self, state: dict[str, Any], goal: Goal) -> set[str]:
        """목표 대상의 접근점·파지점으로 가는 직선을 외접 구 + 여유로 막는 평범한 물체.

        하네스의 구간 대조(`_first_blocker`)와 같은 근사를 **모델 입력**(상태의 자세·OBB, 말단)으로
        다시 계산한다 — 후보 설명에는 막는 물체의 이름이 없기 때문이다. 금지·취약 물체는 밀 수
        없으므로 여기서 세지 않는다.
        """
        target = self._target(state, goal)
        if target is None:
            return set()
        pose = [float(value) for value in target["pose_mm"]]
        top = float(target["top_mm"])
        ee = [float(value) for value in state["robot"]["ee_pose_mm"]]
        holding = state["robot"].get("holding")
        endpoints = [
            [pose[0], pose[1], top + self.approach_clearance_mm],
            [pose[0], pose[1], top - self.grasp_depth_mm],
        ]
        blockers: set[str] = set()
        for entry in state.get("objects") or ():
            object_id = str(entry["id"])
            if object_id in (str(target["id"]), holding):
                continue
            attributes = set(entry.get("attributes") or ())
            if attributes & {"forbidden", "fragile"} or object_id in goal.forbidden or object_id in goal.fragile:
                continue
            centre = [float(value) for value in entry["pose_mm"]]
            limit = circumradius_mm(entry["obb_mm"]) + self.planner_margin_mm
            if math.dist(ee, centre) < limit:
                # 시작점(말단)이 그 물체의 구 안이다 — 방금 놓은 물체 위의 손. 미는 것이 아니라 물러날 일이다.
                continue
            if any(segment_point_distance_mm(ee, point, centre) < limit for point in endpoints):
                blockers.add(object_id)
        return blockers

    def _blocker_pushes(
        self,
        ids: list[str],
        values: dict[str, dict[str, Any]],
        state: dict[str, Any],
        goal: Goal,
    ) -> list[str]:
        """막는 이웃을 대상에서 멀어지게 미는 후보(의미 적합성: 지시를 위한 밀기). 한 구간을 민 예측 자세가
        거리를 문턱 이상 벌려야 한다. 실행기 역량·접촉점 도달성은 여기서 보지 않는다 — 선택의 몫이다."""
        target = self._target(state, goal)
        blockers = self._descent_blockers(state, goal)
        if target is None or not blockers:
            return []
        target_pose = [float(value) for value in target["pose_mm"]]
        poses = {str(entry["id"]): [float(value) for value in entry["pose_mm"]] for entry in state.get("objects") or ()}
        gain_needed = float(self.goal_config.get("push_min_gain_mm", 0.0))
        improving: list[str] = []
        for candidate in ids:
            value = values[candidate]
            if value["function"] != "push" or value["target"] not in blockers:
                continue
            vector = _PUSH_VECTORS.get(str(value["approach"]))
            if vector is None:
                continue
            pose = poses[value["target"]]
            predicted = [pose[0] + vector[0] * self.push_segment_mm, pose[1] + vector[1] * self.push_segment_mm]
            gain = math.dist(predicted, target_pose[:2]) - math.dist(pose[:2], target_pose[:2])
            if gain >= gain_needed:
                value["push_gain_mm"] = gain
                improving.append(candidate)
        return improving

    def _push_allowed(self, value: dict[str, Any]) -> bool:
        """설정이 허용한 밀기 방향인가 (실행기 역량: 열린 손가락이 치지 않는 방향)."""
        allowed = self.goal_config.get("push_directions")
        if allowed is None:
            return True
        return str(value["approach"]) in {str(direction) for direction in allowed}

    def _contact_point_free(self, state: dict[str, Any], pushed: str, vector: tuple[float, float]) -> bool:
        """밀기 접촉점(물체 표면 밖 `push_contact_mm`)이 다른 물체의 외접 구 + 여유 밖인가.

        끝점이 다른 물체의 구 안이면 어떤 경로로도 닿을 수 없다 — 그 방향은 고르지 않는다.
        """
        entry = next((item for item in state.get("objects") or () if str(item["id"]) == pushed), None)
        if entry is None:
            return False
        pose = [float(value) for value in entry["pose_mm"]]
        obb = [float(value) for value in entry["obb_mm"]]
        reach = math.hypot(obb[0] / 2.0, obb[1] / 2.0) + self.push_contact_mm
        contact = [pose[0] - vector[0] * reach, pose[1] - vector[1] * reach, pose[2]]
        holding = state["robot"].get("holding")
        for other in state.get("objects") or ():
            if str(other["id"]) in (pushed, holding):
                continue
            centre = [float(value) for value in other["pose_mm"]]
            if math.dist(contact, centre) < circumradius_mm(other["obb_mm"]) + self.planner_margin_mm:
                return False
        return True

    # -- 게이팅 -------------------------------------------------------------------

    def _gates(
        self, state: dict[str, Any], goal: Goal, phase: str, model: dict[str, Any]
    ) -> dict[str, dict[str, Any]]:
        done = self._goal_satisfied(state, goal)
        instr, instr_reason = self._instruction_complete(state, goal)
        observe, observe_reason = self._needs_observation(state, goal, phase)
        retry, retry_reason = self._retry_ok(model)
        stop, stop_reason = self._must_stop(state, goal)
        return {
            "q_done": {"value": done, "rule": _GATE_RULES["q_done"], "reason": "inside_zone" if done else "not_yet"},
            "q_instr": {"value": instr, "rule": _GATE_RULES["q_instr"], "reason": instr_reason},
            "q_observe": {"value": observe, "rule": _GATE_RULES["q_observe"], "reason": observe_reason},
            "q_retry": {"value": retry, "rule": _GATE_RULES["q_retry"], "reason": retry_reason},
            "q_stop": {"value": stop, "rule": _GATE_RULES["q_stop"], "reason": stop_reason},
        }

    @staticmethod
    def _target(state: dict[str, Any], goal: Goal) -> dict[str, Any] | None:
        return next((entry for entry in state.get("objects") or () if str(entry["id"]) == goal.target_ref), None)

    def _goal_satisfied(self, state: dict[str, Any], goal: Goal) -> bool:
        """목표 조건이 관측으로 성립하는가: 대상이 목표 영역 안에 놓여 있고 손에 없다."""
        target = self._target(state, goal)
        zone = next((entry for entry in state.get("zones") or () if str(entry["id"]) == goal.zone), None)
        if target is None or zone is None or state["robot"].get("holding") == target["id"]:
            return False
        return _inside(target["pose_mm"], zone["bounds_mm"])

    @staticmethod
    def _instruction_complete(state: dict[str, Any], goal: Goal) -> tuple[bool, str]:
        """풀 수 없거나(대상·목적지 없음) 모순인(대상이 금지 물체) 지시만 불충분하다 (docs/08 §4)."""
        zones = {str(entry.get("id")) for entry in state.get("zones") or ()}
        if goal.structured:
            if not (goal.target_desc or goal.target_ref):
                return False, "no_target"
            if goal.zone is None or goal.zone not in zones:
                return False, "no_destination"
            if goal.target_ref is not None and goal.target_ref in goal.forbidden:
                return False, "target_forbidden"
            return True, "structured"
        if goal.target_ref is None:
            return False, "no_target"
        if goal.zone is None or goal.zone not in zones:
            return False, "no_destination"
        if goal.target_ref in goal.forbidden:
            return False, "target_forbidden"
        return True, "text"

    def _needs_observation(self, state: dict[str, Any], goal: Goal, phase: str) -> tuple[bool, str]:
        """대상이 아직 추적되지 않았거나 대상 기하가 문턱보다 오래됐다. 팔이 대상을 가리는 접촉 국면(파지·놓기·
        밀기)과 파지 중에는 기하 나이가 아니라 **실행기의 readiness**가 시점을 정하므로 관측을 요구하지 않는다 —
        하네스(§5.0)·규칙 기준군과 같은 면제.

        손에 **다른** 물체가 있으면 관측은 지금 할 수 있는 일이 아니다: 운반 중의 관측은 제자리 hold이고
        (docs/08 §5.2) 가리는 것이 팔 자신이면 영영 풀리지 않는다. 먼저 놓는 것이 답이다(`_main`).
        """
        holding = state["robot"].get("holding")
        if holding is not None and goal.target_ref != holding and (goal.target_ref is not None or goal.target_desc):
            return False, "holding_other_first"
        target = self._target(state, goal)
        if target is None:
            return True, "target_untracked"
        if phase in _CONTACT_PHASES or holding == target["id"]:
            return False, "contact_readiness"
        age = float(target.get("age_ms", 0))
        limit = float(self.thresholds["observe_geom_age_ms"])
        return (age > limit), ("geometry_stale" if age > limit else "fresh")

    def _retry_ok(self, model: dict[str, Any]) -> tuple[bool, str]:
        history = parse_exec_history(model.get("exec_history"))
        if not history or history.get("ack") in (None, "ok", "none"):
            return False, "no_failure"
        try:
            fails = max(1, int(history.get("fails", 1)))
        except ValueError:
            fails = 1
        limit = int(self.thresholds["retry_max_same_approach"])
        return (fails <= limit), (f"fails={fails}" if fails <= limit else f"fails={fails}>{limit}")

    def _must_stop(self, state: dict[str, Any], goal: Goal) -> tuple[bool, str | None]:
        robot = state["robot"]
        # 파지 중의 정상 하중은 충돌이 아니다 — 들고 있는 동안은 더 높은 문턱(실행기 반사 한계 위)을 쓴다.
        force_limit = float(
            self.thresholds["stop_force_holding_n" if robot.get("holding") else "stop_force_n"]
        )
        if float(robot.get("contact_n") or 0.0) > force_limit:
            return True, "contact_force"
        attributes = {str(entry["id"]): set(entry.get("attributes") or ()) for entry in state.get("objects") or ()}
        for forbidden in goal.forbidden:
            attributes.setdefault(forbidden, set()).add("forbidden")
        watched = set(str(item) for item in self.stop_config.get("contact_attributes") or ())
        for event in state.get("events") or ():
            kind = str(event.get("kind", ""))
            if kind.startswith("reflex"):
                return True, "reflex"
            if kind == "contact_onset" and attributes.get(str(event.get("object")), set()) & watched:
                return True, "protected_contact"
        ee = [float(value) for value in robot["ee_pose_mm"]]
        limit = float(self.thresholds["stop_forbidden_mm"])
        for entry in state.get("objects") or ():
            if "forbidden" not in attributes[str(entry["id"])]:
                continue
            if math.dist(ee, [float(value) for value in entry["pose_mm"]]) <= limit:
                return True, "forbidden_proximity"
        return False, None

    # -- 부가 답 ------------------------------------------------------------------

    def _aux(
        self,
        state: dict[str, Any],
        goal: Goal,
        phase: str,
        committed: dict[str, Any],
        values: dict[str, dict[str, Any]],
        paths: list[dict[str, Any]],
    ) -> dict[str, Any]:
        value = values.get(str(committed["action_ref"]))
        return {
            "phase": phase,
            "gripper": self._gripper(state, phase, value),
            "path": self._path(paths, value, state),
            "speed": self._speed(state, goal, phase, value),
            "force": {"level": str(min(int(self.profiles["force_by_phase"][phase]), len(self.force_levels) - 1))},
        }

    def _gripper(self, state: dict[str, Any], phase: str, value: dict[str, Any] | None) -> dict[str, Any]:
        """국면 프로파일. 파지 국면에서는 말단이 파지점에 와야 닫는다 (내려가는 중에 닫으면 윗면을 누른다)."""
        holding = state["robot"].get("holding")
        desired = str(self.profiles["gripper_by_phase"][phase])
        reason = f"phase:{phase}"
        if value and value.get("function") == "push" and phase in ("approach", "push") and not holding:
            # 밀기는 주먹으로: 접촉점으로 가는 접근부터 닫는다 (설정 `gripper_for_push`).
            desired = str(self.profiles.get("gripper_for_push", desired))
            reason = "push_with_closed_fingers" if desired == "closed" else f"push:{desired}"
        if desired == "current":
            desired = "closed" if holding else "open"
            reason = "holding" if holding else "idle"
        if phase == "grasp" and desired == "closed" and not holding:
            point = self._grasp_point(state, value)
            ee = [float(item) for item in state["robot"]["ee_pose_mm"]]
            if point is None or math.dist(ee, point) > float(self.thresholds["grasp_ready_mm"]):
                desired, reason = "open", "not_at_grasp_point"
            else:
                reason = "at_grasp_point"
        return {"desired": desired, "reason": reason}

    def _grasp_point(self, state: dict[str, Any], value: dict[str, Any] | None) -> list[float] | None:
        """파지점 = 대상 윗면 아래 `grasp_depth_mm` (하네스와 같은 정의, 상태의 물체에서 계산)."""
        if not value or value.get("target") is None:
            return None
        entry = next((item for item in state.get("objects") or () if str(item["id"]) == value["target"]), None)
        if entry is None:
            return None
        pose = [float(item) for item in entry["pose_mm"]]
        return [pose[0], pose[1], float(entry["top_mm"]) - self.grasp_depth_mm]

    def _path(self, paths: list[dict[str, Any]], value: dict[str, Any] | None, state: dict[str, Any]) -> dict[str, Any]:
        """비어 있으면 direct, 막혔으면 첫 경유점(대안 경유점은 라벨 허용 집합), 그것도 없으면 기다린다.

        `retreat`은 명령마다 정해진 만큼 물러나므로 매 틱 답하면 팔이 한계까지 올라간다. 말단 자체가
        장애물의 넓힌 구 안에 있을 때만(어느 구간도 시작점에서 막힌다) 물러나고, 끝점이 막힌 것은
        기다린다(`hold`) — 그 사이 주 결정이 막는 이웃을 밀거나 목표가 바뀐다.
        """
        options = [entry["id"] for entry in paths]
        by_kind: dict[str, list[str]] = {}
        for entry in paths:
            by_kind.setdefault(str(entry.get("kind")), []).append(str(entry["id"]))
        if not options:
            return {"choice": None, "kind": None, "reason": "no_paths", "alternatives": []}
        if value is None or value.get("function") is None:
            choice = (by_kind.get("hold") or options)[0]
            return {"choice": choice, "kind": "hold", "reason": "fixed_candidate", "alternatives": []}
        if value["path_clear"]:
            return {"choice": (by_kind.get("direct") or options)[0], "kind": "direct", "reason": "clear", "alternatives": []}
        vias = by_kind.get("via") or []
        if vias:
            return {"choice": vias[0], "kind": "via", "reason": "blocked_via", "alternatives": self._via_alternatives(paths, vias, state)}
        if by_kind.get("retreat") and self._hand_inside_obstacle(state, value):
            return {"choice": by_kind["retreat"][0], "kind": "retreat", "reason": "hand_inside_obstacle", "alternatives": []}
        choice = (by_kind.get("hold") or options)[0]
        return {"choice": choice, "kind": "hold", "reason": "blocked_no_detour", "alternatives": []}

    def _hand_inside_obstacle(self, state: dict[str, Any], value: dict[str, Any]) -> bool:
        """말단이 대상·든 물체가 아닌 물체의 장애물 구 안에 있는가 (모든 구간이 시작점에서 막힌다).

        하네스와 같은 반지름이다: 외접 구 + 여유, 금지 접촉 물체는 `forbidden_margin_mm`만큼 더.
        """
        ee = [float(item) for item in state["robot"]["ee_pose_mm"]]
        skip = {str(value.get("target")), str(state["robot"].get("holding"))}
        forbidden = set(read_goal(state).forbidden)
        for entry in state.get("objects") or ():
            if str(entry["id"]) in skip:
                continue
            centre = [float(item) for item in entry["pose_mm"]]
            limit = circumradius_mm(entry["obb_mm"]) + self.planner_margin_mm
            if str(entry["id"]) in forbidden or "forbidden" in (entry.get("attributes") or ()):
                limit += self.forbidden_margin_mm
            if math.dist(ee, centre) < limit:
                return True
        return False

    def _via_alternatives(self, paths: list[dict[str, Any]], vias: list[str], state: dict[str, Any]) -> list[str]:
        """최선의 경유점보다 `path_alternative_extra_mm` 안에 드는 다른 경유점 (docs/08 §7 `q_path`)."""
        extra = {
            str(item["waypoint"]): float(item.get("extra_mm", 0))
            for item in state.get("derived") or ()
            if "waypoint" in item
        }
        refs = {str(entry["id"]): str(entry.get("ref")) for entry in paths if entry.get("kind") == "via"}
        best = extra.get(refs.get(vias[0], ""), 0.0)
        limit = float(self.label_config.get("path_alternative_extra_mm", 0.0))
        return [
            via for via in vias[1:] if extra.get(refs.get(via, ""), math.inf) - best <= limit
        ]

    def _speed(self, state: dict[str, Any], goal: Goal, phase: str, value: dict[str, Any] | None) -> dict[str, Any]:
        level = int(self.profiles["speed_by_phase"][phase])
        capped = False
        if level and self._near_fragile(state, goal, value):
            cap = int(self.profiles["fragile_speed_cap"])
            capped = cap < level
            level = min(level, cap)
        level = min(level, len(self.speed_levels) - 1)
        return {"level": str(level), "capped": capped}

    def _near_fragile(self, state: dict[str, Any], goal: Goal, value: dict[str, Any] | None) -> bool:
        limit = float(self.thresholds["fragile_proximity_mm"])
        points = [[float(item) for item in state["robot"]["ee_pose_mm"]]]
        target = value["target"] if value else None
        for entry in state.get("objects") or ():
            if str(entry["id"]) == target:
                points.append([float(item) for item in entry["pose_mm"]])
        for entry in state.get("objects") or ():
            fragile = "fragile" in (entry.get("attributes") or ()) or str(entry["id"]) in goal.fragile
            if not fragile or str(entry["id"]) == target:
                continue
            pose = [float(item) for item in entry["pose_mm"]]
            if any(math.dist(pose, point) <= limit for point in points):
                return True
        return False

    # -- 분포 ---------------------------------------------------------------------

    def _truth(self, answer: bool) -> float:
        return float(self.confidence["high" if answer else "low"])

    def _spread(self, chosen: str | None, options: list[str]) -> dict[str, float]:
        """고른 항목에 설정된 질량을 주고 나머지를 고르게 나눈다. 고른 것이 없으면 고른 분포다."""
        if not options:
            return {}
        if chosen not in options:
            return normalise_distribution({option: 1.0 / len(options) for option in options})
        mass = float(self.confidence["choice_mass"])
        rest = [option for option in options if option != chosen]
        share = (1.0 - mass) / len(rest) if rest else 0.0
        return normalise_distribution(
            {option: (mass if option == chosen else share) for option in options}
        )

    # -- 라벨 (docs/08 §7) ----------------------------------------------------------

    def labels(self, answers: dict[str, Any], request: dict[str, Any]) -> list[dict[str, Any]]:
        """전문가 답 → 그 틱의 라벨. 부가 질문은 요청의 commitment에 조건화되고, 없으면 마스킹한다.

        두 가지 완화 규칙을 :meth:`_finish_label` 이 건다. **낮은 신뢰도**(`label_confidence: low` — 목표를
        실현할 후보가 목록에 없거나 재시도가 막혔거나 실행기 사정으로 `hold`로 물러난 퇴화 틱)의 라벨은
        설정 `labels.low_confidence_weight`(시작값 0.25)를 `weight`로 달아 손실(docs/03 §4, `loss.py`)이
        덜 끌리게 한다. **허용 집합이 빈** `valid_set` 라벨은 근거가 없는 질문이라 마스크인데, 계약이 빈
        `candidate_ids`를 거절하므로 마스크 = 라벨을 적지 않는 것이다(경로 후보가 없는 틱의 `q_path`).
        """
        model = request.get("request", request)
        meta = answers["expert_meta"]
        main = meta["main"]
        source = self.label_source
        labels: list[dict[str, Any]] = []
        allowed = list(main.get("allowed") or ([main["choice"]] if main["choice"] is not None else []))
        label: dict[str, Any] = {
            "question_id": "q_main",
            "kind": "valid_set",
            "candidate_ids": allowed,
            "semantic_admissible": list(main["admissible"]),
            "source": source,
            "rule": f"expert-{self.version}/{main['reason']}",
            "label_confidence": main["confidence"],
        }
        unknown = [candidate for candidate in main.get("unknown") or () if candidate not in allowed]
        if unknown:
            label["unknown"] = unknown  # hold∉A 규칙: 적합 후보는 판단하지 않는다 (정규화에서 빠진다)
        self._finish_label(labels, label)
        for question_id, gate in meta["gates"].items():
            self._finish_label(
                labels,
                {
                    "question_id": question_id,
                    "kind": "single",
                    "answer": bool(gate["value"]),
                    "source": source,
                    "rule": gate["rule"],
                },
            )
        commitment = model.get("commitment")
        aux = meta.get("aux")
        if commitment is None or aux is None:
            return labels
        conditioned = f"{commitment['action_ref']}/{commitment['phase']}"
        path_ids = [aux["path"]["choice"]] + list(aux["path"]["alternatives"]) if aux["path"]["choice"] else []
        entries = {
            "q_gripper": {"kind": "valid_set", "candidate_ids": [aux["gripper"]["desired"]], "rule": f"phase-profile-v0/{aux['gripper']['reason']}"},
            "q_path": {"kind": "valid_set", "candidate_ids": path_ids, "rule": f"clear-direct-else-via-v0/{aux['path']['kind']}"},
            "q_speed": {"kind": "valid_set", "candidate_ids": [aux["speed"]["level"]], "rule": "phase-profile-fragile-cap-v0"},
            "q_force": {"kind": "single", "answer": aux["force"]["level"], "rule": "phase-contact-mode-v0"},
        }
        for question_id in AUX_QUESTIONS:
            entry = entries[question_id]
            self._finish_label(labels, {"question_id": question_id, **entry, "source": source, "conditioned_on": conditioned})
        return labels

    def _finish_label(self, labels: list[dict[str, Any]], label: dict[str, Any]) -> None:
        """라벨 하나를 목록에 넣기 전에 완화 규칙을 건다 (:meth:`labels` 참조).

        허용 집합이 빈 `valid_set`은 넣지 않고(마스크), 낮은 신뢰도에는 설정의 weight를 단다. weight는
        1이면 적지 않는다 — 기본값과 같은 값을 레코드마다 되풀이할 이유가 없다.
        """
        if label["kind"] == "valid_set" and not label["candidate_ids"]:
            return
        if label.get("label_confidence") == "low":
            weight = float(self.label_config.get("low_confidence_weight", 0.25))
            if weight < 0:
                raise ValueError(f"labels.low_confidence_weight: 0 이상이어야 한다 (받은 값: {weight})")
            if weight != 1.0:
                label["weight"] = weight
        labels.append(label)


# --------------------------------------------------------------------------


def _inside(pose_mm, bounds_mm) -> bool:
    x0, y0, x1, y1 = [float(value) for value in bounds_mm]
    x, y = float(pose_mm[0]), float(pose_mm[1])
    return min(x0, x1) <= x <= max(x0, x1) and min(y0, y1) <= y <= max(y0, y1)


def _zone_centre(bounds_mm) -> tuple[float, float]:
    x0, y0, x1, y1 = [float(value) for value in bounds_mm]
    return ((x0 + x1) / 2.0, (y0 + y1) / 2.0)


def _zone_distance_mm(point, bounds_mm) -> float:
    """점에서 영역 사각형까지의 xy 거리. 안이면 0."""
    x0, y0, x1, y1 = [float(value) for value in bounds_mm]
    dx = max(min(x0, x1) - float(point[0]), 0.0, float(point[0]) - max(x0, x1))
    dy = max(min(y0, y1) - float(point[1]), 0.0, float(point[1]) - max(y0, y1))
    return math.hypot(dx, dy)
