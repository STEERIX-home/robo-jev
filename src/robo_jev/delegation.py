"""실행 쪽 위임 — 그리퍼 닫기 시점(과 정지)을 실행 층의 기하 규칙에 맡기는 **정책 감싸개** (Task R10 A1; docs/08 §10).

왜. R4~R9의 폐루프 실패의 한 갈래는 실행 수준 head의 불안정이다 — 그리퍼 head는 모든 seed에서 ≈ 100 step 평탄 뒤 늦게
맞고, `q_stop`은 세 seed 모두 정지 사건을 하나도 잡지 못했다. 그런데 전문가의 그리퍼 결정은 **기하 규칙**이다(국면별 기본 상태 +
파지점까지 `grasp_ready_mm`; 놓기의 `place_blocked`). 가설: 그리퍼 시점(과 힘 정지)은 의미 층이 아니라 실행 층의 일이고, 그것을 실행
쪽 규칙에 맡기면 모델의 실패 상당 부분이 사라진다. 이 모듈은 그 가설을 **학습 없이** 잰다. 하네스·컨트롤러·직렬화·expert·`contracts.py`는
바꾸지 않는다 — 하네스 버전 문자열이 계약 digest에 들어가므로 하네스를 바꾸면 기존 체크포인트가 적재를 거절당한다. 위임은 정책
자리에서 한다: 감싸개가 정책의 답 가운데 실행 층으로 옮길 답만 바꿔 하네스에 넘긴다(R6 A3 `DoneGatePolicy`와 같은 꼴).

갈래 둘 (:data:`ARMS`).

* **G** — 감싼 정책의 답을 그대로 쓰되 `q_gripper`만 실행 쪽 규칙(:class:`ExecutionGripperRule`)의 답으로 바꾼다.
* **GS** — G에 더해 `q_stop`을 언제나 거짓(0.0)으로 둔다 — 정지는 실행기의 힘 반사뿐이다(하네스가 반사 사건을 보면 답과 무관하게
  정지 분기로 간다, `RobotHarness._compose` 1단계).

**실행 쪽 규칙** = 전문가 e0.4의 `Expert._gripper` — 국면별 기본 상태(`profiles.gripper_by_phase`), 파지 국면은 지금 commitment의 대상
파지점(윗면 아래 `grasp_depth_mm`)까지 `grasp_ready_mm` 안일 때만 `closed`, 놓기의 `open`은 내려가는 경로(direct·via)일 때만, 밀기는
접근·밀기 국면에서 주먹(`gripper_for_push`), commitment가 없으면 지금 쥐고 있는지. 그 규칙을 **지금 commitment와 관측 상태(요청의
`state`)·후보 목록만으로** 계산하도록 떼어 냈다 — 전문가 객체를 부르지 않고 그 설정의 수치만 든다. 목표·지시 문장·시뮬레이터
참값(`observation`)은 읽지 않는다. 같은 입력에서 전문가의 답과 같다는 것은 시험이 고정한다(`tests/test_delegation.py`: 합성 요청의 갈래마다
`Expert._gripper`와, 저장된 폐루프 기록의 틱마다 전문가의 `evidence.expert` 판단과, 전문가를 감싼 G가 감싸지 않은 전문가의 기록을 그대로
재현하는지).

**정보 경계.** 감싼 정책은 생성기가 주는 것을 그대로 받는다(`act(request, commitment, observation)` — 감싸지 않은 경로와 같은 인자);
`ModelPolicy`는 commitment·관측을 읽지 않는다(그 규칙 그대로). 감싼 정책은 실행 규칙의 답도 전문가의 답도 받지 않는다 — 감싸개가
그것들을 하네스 쪽으로만 넘긴다. 감싼 정책이 다음 틱에 보는 것은 하네스가 실제로 실행한 결과(`exec_history`의 그리퍼 등)뿐이고, 그것은
감싸지 않은 경로와 같은 되먹임이다.

**기록의 자리.** 생성기(:func:`robo_jev.data.robot_episodes.generate_episode`)는 정책이 돌려준 답을 `model_output`에 쓴다 — 이 정책의 답이므로
`model_output.q_gripper`(GS면 `q_stop`도)는 **실행 규칙의 값**(하네스가 실제로 받은 것)이다. 감싼 정책의 raw 답은 틱마다
`usage.delegation.inner`에, 실행 규칙의 판단(원하는 상태·이유·국면)은 `usage.delegation.rule`에, 편의 요약은 `evidence.delegation`에, 갈래·안쪽
정책은 `provenance.policy`에 남긴다(:meth:`DelegatedExecutionPolicy.attach`) — 셋 다 계약 검증이 받는 자리다(`usage`·`evidence`·`provenance`는
dict이면 된다; `model_output`의 라벨 유출 검사와 무관하다).
"""

from __future__ import annotations

import copy
import math
from typing import Any

from robo_jev.contracts import PHASES
from robo_jev.harness.robot import load_harness_config
from robo_jev.harness.rule_judge import candidate_values, normalise_distribution
from robo_jev.sim.expert import load_expert_config

__all__ = [
    "ARMS",
    "DELEGATION_VERSION",
    "EXECUTION_GRIPPER_VERSION",
    "DelegatedExecutionPolicy",
    "ExecutionGripperRule",
    "committed_candidate",
    "delegate_bundle",
    "descent_path_kind",
    "rule_agreement",
]

#: 감싸개 버전 — 레코드의 `provenance.policy.version`은 ``dx0.1/<갈래>/<안쪽 정책의 버전>``이다.
DELEGATION_VERSION = "dx0.1"
#: 실행 쪽 그리퍼 규칙의 버전 — 전문가 e0.4의 `_gripper`와 같은 규칙이다(시험이 고정).
EXECUTION_GRIPPER_VERSION = "xg0.1"
#: 갈래 — G: `q_gripper`만 실행 규칙; GS: G + `q_stop` 언제나 거짓(정지는 힘 반사만).
ARMS = ("G", "GS")
#: 그리퍼 답의 후보 순서 — 전문가 `_spread(desired, ["open", "closed"])`와 같은 순서라 직렬화한 바이트도 같다.
_GRIPPER_OPTIONS = ("open", "closed")


def committed_candidate(model: dict[str, Any], commitment: dict[str, Any] | None, values: dict[str, dict[str, Any]]) -> dict[str, Any] | None:
    """지금 commitment — 요청에 실린 투영(그 틱의 후보 목록에 있는 것)이 먼저, 없으면 하네스 commitment의 후보가 목록에 있을 때만
    (전문가 `Expert._committed`와 같은 규칙). 하네스는 투영을 하네스 commitment에서 만들므로(`_project_commitment`) 둘은 같은 후보를 가리킨다."""
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


def descent_path_kind(paths: list[dict[str, Any]], value: dict[str, Any] | None) -> str | None:
    """commitment 후보의 하강 경로 종류 — 전문가 `Expert._path`가 고르는 경로의 종류에서 그리퍼 규칙이 쓰는 몫만.

    경로 후보가 없으면 None, 고정 후보(기능 없음)면 `hold`, 경로가 비어 있으면 `direct`, 막혔지만 경유점 후보가 있으면 `via`, 그 밖은
    `blocked`(전문가는 손이 장애물 구 안이면 `retreat`, 아니면 `hold`를 고른다 — 그리퍼 규칙에는 둘 다 "내려가지 않는다"이다)."""
    if not paths:
        return None
    if value is None or value.get("function") is None:
        return "hold"
    if value["path_clear"]:
        return "direct"
    if any(str(entry.get("kind")) == "via" for entry in paths):
        return "via"
    return "blocked"


class ExecutionGripperRule:
    """실행 쪽 그리퍼 규칙 (모듈 설명) — 지금 commitment와 관측 상태만으로 전문가 e0.4의 `_gripper`를 계산한다.

    수치는 전문가 설정(`configs/sim/expert_v0.yaml`: `profiles.gripper_by_phase`·`gripper_for_push`, `thresholds.grasp_ready_mm`,
    `confidence.choice_mass`)과 그것이 가리키는 하네스 설정(`candidates.grasp_depth_mm`)에서 읽는다 — 단일 출처는 그 설정이다."""

    name = "ExecutionGripperRule"
    version = EXECUTION_GRIPPER_VERSION

    def __init__(self, expert_config: dict[str, Any] | None = None, *, harness_config: dict[str, Any] | None = None) -> None:
        config = copy.deepcopy(expert_config) if expert_config is not None else load_expert_config()
        profiles = config["profiles"]
        self.expert_version = str(config.get("version", "unknown"))
        self.gripper_by_phase = {str(phase): str(state) for phase, state in profiles["gripper_by_phase"].items()}
        self.profiles = copy.deepcopy(profiles)
        self.grasp_ready_mm = float(config["thresholds"]["grasp_ready_mm"])
        self.choice_mass = float(config["confidence"]["choice_mass"])
        harness = harness_config or load_harness_config(config["harness_config"])
        self.grasp_depth_mm = float(harness["candidates"]["grasp_depth_mm"])

    def describe(self) -> dict[str, Any]:
        return {"name": self.name, "version": self.version, "pinned_to": f"Expert._gripper ({self.expert_version})",
                "grasp_ready_mm": self.grasp_ready_mm, "grasp_depth_mm": self.grasp_depth_mm, "gripper_by_phase": dict(self.gripper_by_phase),
                "gripper_for_push": self.profiles.get("gripper_for_push")}

    def grasp_point(self, state: dict[str, Any], value: dict[str, Any] | None) -> list[float] | None:
        """파지점 = 대상 윗면 아래 `grasp_depth_mm` — 관측된 물체의 자세·윗면에서 (하네스·전문가와 같은 정의)."""
        if not value or value.get("target") is None:
            return None
        entry = next((item for item in state.get("objects") or () if str(item["id"]) == value["target"]), None)
        if entry is None:
            return None
        pose = [float(item) for item in entry["pose_mm"]]
        return [pose[0], pose[1], float(entry["top_mm"]) - self.grasp_depth_mm]

    def decide(self, request: dict[str, Any], commitment: dict[str, Any] | None = None) -> dict[str, Any]:
        """원하는 그리퍼 상태와 그 이유 — ``{"desired", "reason", "phase", "action_ref", "path_kind"}``.

        읽는 것: 요청의 commitment 투영(없으면 하네스 `commitment`), 후보 목록(`q_main`의 의미 키·경로 막힘, `q_path`의 종류), 관측 상태의
        `robot`(쥔 물체·말단 자세)과 `objects`(대상의 자세·윗면). 목표·지시·관측 원본은 읽지 않는다."""
        model = request.get("request", request)
        state = model["state"]
        holding = state["robot"].get("holding")
        values = {entry["id"]: candidate_values(entry) for entry in model["candidates"]["q_main"]}
        committed = committed_candidate(model, commitment, values)
        if committed is None:
            # commitment가 없는 틱: 부가 질문은 hold 기준 — 그리퍼는 현재 상태 (전문가 `act`의 `aux is None` 갈래)
            return {"desired": "closed" if holding else "open", "reason": "no_commitment", "phase": None, "action_ref": None, "path_kind": None}
        phase = str(committed.get("phase", "none"))
        if phase not in PHASES:
            phase = "none"
        value = values.get(str(committed["action_ref"]))
        path_kind = descent_path_kind(list(model["candidates"].get("q_path") or []), value)
        desired = self.gripper_by_phase[phase]
        reason = f"phase:{phase}"
        if phase == "place" and desired == "open" and path_kind not in ("direct", "via"):
            # 막힌 하강에서 열면 운반 높이에서 떨어뜨린다 — 하네스도 같은 틱에 open을 적용하지 않는다 (e0.4, 리뷰 2 C2)
            desired, reason = "closed", "place_blocked"
        if value and value.get("function") == "push" and phase in ("approach", "push") and not holding:
            # 밀기는 주먹으로: 접촉점으로 가는 접근부터 닫는다
            desired = str(self.profiles.get("gripper_for_push", desired))
            reason = "push_with_closed_fingers" if desired == "closed" else f"push:{desired}"
        if desired == "current":
            desired = "closed" if holding else "open"
            reason = "holding" if holding else "idle"
        if phase == "grasp" and desired == "closed" and not holding:
            point = self.grasp_point(state, value)
            ee = [float(item) for item in state["robot"]["ee_pose_mm"]]
            if point is None or math.dist(ee, point) > self.grasp_ready_mm:
                desired, reason = "open", "not_at_grasp_point"
            else:
                reason = "at_grasp_point"
        return {"desired": desired, "reason": reason, "phase": phase, "action_ref": str(committed["action_ref"]), "path_kind": path_kind}

    def distribution(self, desired: str) -> dict[str, float]:
        """원하는 상태 → 답의 분포 (전문가 `_spread(desired, ["open", "closed"])`와 같은 질량·반올림·키 순서)."""
        if desired not in _GRIPPER_OPTIONS:
            raise ValueError(f"desired: {list(_GRIPPER_OPTIONS)} 중 하나여야 한다 (받은 값: {desired!r})")
        share = 1.0 - self.choice_mass
        return normalise_distribution({option: (self.choice_mass if option == desired else share) for option in _GRIPPER_OPTIONS})

    def answer(self, request: dict[str, Any], commitment: dict[str, Any] | None = None) -> dict[str, float]:
        return self.distribution(self.decide(request, commitment)["desired"])


def _argmax(answer: Any) -> str | None:
    """choice 답의 최댓값 — 하네스 `_argmax`와 같은 동점 규칙(확률 내림차순 → id 오름차순). 빈 답·dict가 아니면 None."""
    if not isinstance(answer, dict) or not answer:
        return None
    usable = {str(key): float(value) for key, value in answer.items() if str(key) in _GRIPPER_OPTIONS}
    if not usable:
        return None
    return min(usable, key=lambda key: (-usable[key], key))


def _said_stop(value: Any) -> bool | None:
    """boolean 답 → 하네스 정지 문턱(0.5)에서의 참·거짓 (두 출력 꼴 모두). 답이 없으면 None."""
    if value is None:
        return None
    if isinstance(value, dict):
        if "true" in value:
            return float(value["true"]) >= 0.5
        if "false" in value:
            return 1.0 - float(value["false"]) >= 0.5
        return None
    return float(value) >= 0.5


class DelegatedExecutionPolicy:
    """감싼 정책의 답 + 실행 쪽 규칙의 `q_gripper`(GS면 `q_stop` = 0.0도) (모듈 설명). 평가 정책이다 — 수집 전용이 아니다."""

    name = "DelegatedExecutionPolicy"

    def __init__(self, inner: Any, *, arm: str, rule: ExecutionGripperRule | None = None) -> None:
        if arm not in ARMS:
            raise ValueError(f"arm: {list(ARMS)} 중 하나여야 한다 (받은 값: {arm!r})")
        self.inner = inner
        self.arm = str(arm)
        self.rule = rule if rule is not None else ExecutionGripperRule()
        self.inner_name = str(getattr(inner, "name", type(inner).__name__))
        self.inner_version = str(getattr(inner, "version", "unknown"))
        self.version = f"{DELEGATION_VERSION}/{self.arm}/{self.inner_version}"
        self.raw: list[dict[str, Any]] = []

    @property
    def timing(self) -> list[dict[str, Any]]:
        """안쪽 정책의 틱 지연 기록(모델이면 `ModelPolicy.timing`) — 폐루프 실행기가 지연 sidecar에 옮긴다."""
        return list(getattr(self.inner, "timing", []) or [])

    def reset(self) -> None:
        """새 에피소드 — 안쪽 정책을 되돌리고 raw 기록을 비운다."""
        if hasattr(self.inner, "reset"):
            self.inner.reset()
        self.raw = []

    def act(self, request: dict[str, Any], commitment: dict[str, Any] | None = None, observation: Any = None) -> dict[str, Any]:
        # 안쪽 정책은 감싸지 않은 경로와 같은 인자를 받는다 (ModelPolicy는 commitment·관측을 읽지 않는다)
        answers = self.inner.act(request, commitment, observation)
        # 실행 규칙: 지금 commitment와 관측 상태(요청의 state)만 — 시뮬레이터 관측 원본(`observation`)은 넘기지 않는다
        decision = self.rule.decide(request, commitment)
        delegated = dict(answers)
        delegated["q_gripper"] = self.rule.distribution(decision["desired"])
        inner = {"q_gripper": copy.deepcopy(answers.get("q_gripper"))}
        if self.arm == "GS":
            inner["q_stop"] = copy.deepcopy(answers.get("q_stop"))
            delegated["q_stop"] = 0.0  # 의미 정지를 버린다 — 정지는 실행기의 힘 반사뿐
        self.raw.append({"t": int(request["t"]), "inner": inner, "rule": decision})
        return delegated

    def attach(self, record: dict[str, Any]) -> dict[str, Any]:
        """틱마다 감싼 정책의 raw 답과 실행 규칙의 판단을 `usage.delegation`에, 편 요약을 `evidence.delegation`에, 갈래·안쪽 정책을
        `provenance.policy`에 적는다. 틱 번호가 어긋나면 거절한다(raw는 생성기가 부른 `act` 순서 그대로다)."""
        ticks = record["ticks"]
        if len(self.raw) != len(ticks) or any(int(entry["t"]) != int(tick["t"]) for entry, tick in zip(self.raw, ticks)):
            raise ValueError(f"{record.get('episode_id')}: 위임 raw 기록이 틱과 맞지 않는다 ({len(self.raw)}개 대 틱 {len(ticks)}개)")
        changed = inner_closed = rule_closed = stop_overridden = inner_silent = 0
        for tick, entry in zip(ticks, self.raw):
            usage = tick.setdefault("usage", {})
            usage["delegation"] = {"arm": self.arm, "inner": copy.deepcopy(entry["inner"]), "rule": copy.deepcopy(entry["rule"])}
            inner_choice = _argmax(entry["inner"]["q_gripper"])
            rule_choice = entry["rule"]["desired"]
            inner_silent += int(inner_choice is None)
            inner_closed += int(inner_choice == "closed")
            rule_closed += int(rule_choice == "closed")
            changed += int(inner_choice is not None and inner_choice != rule_choice)
            if self.arm == "GS":
                stop_overridden += int(bool(_said_stop(entry["inner"].get("q_stop"))))
        record["provenance"]["policy"] = {
            **record["provenance"]["policy"], "arm": self.arm, "delegation_version": DELEGATION_VERSION,
            "rule": {"name": self.rule.name, "version": self.rule.version}, "inner": {"name": self.inner_name, "version": self.inner_version},
        }
        summary: dict[str, Any] = {
            "version": DELEGATION_VERSION, "arm": self.arm, "rule": self.rule.describe(),
            "inner": {"name": self.inner_name, "version": self.inner_version}, "ticks": len(ticks),
            "q_gripper_changed_ticks": changed, "inner_q_gripper_closed_ticks": inner_closed, "inner_q_gripper_silent_ticks": inner_silent,
            "rule_closed_ticks": rule_closed,
            "note": "model_output.q_gripper is the execution rule's answer (what the harness received); the wrapped policy's raw answers are usage.delegation.inner",
        }
        if self.arm == "GS":
            summary["q_stop_overridden_ticks"] = stop_overridden
            summary["note"] += "; GS: model_output.q_stop is 0.0 on every tick (stopping is the executor's force reflex only)"
        record.setdefault("evidence", {})["delegation"] = summary
        return record


def delegate_bundle(bundle: dict[str, Any], *, arm: str, expert_config: dict[str, Any]) -> dict[str, Any]:
    """폐루프 정책 묶음(:func:`robo_jev.closed_loop.build_policy`)을 갈래 `arm`의 감싸개로 바꾼 새 묶음.

    참조(라벨·지연의 원천 `bundle["expert"]`)는 그대로다. 안쪽 정책이 그 참조 자신이면(전문가를 감쌀 때) 안쪽에는 **따로 만든** 전문가를
    둔다 — 같은 객체면 생성기가 참조를 정책의 답으로 대신하고(`policy is expert`), 참조 지연 계측도 두 번 쌓인다. 두 전문가는 상태가 없고
    결정적이므로 답은 같다."""
    from robo_jev.sim.expert import Expert

    inner = bundle["policy"]
    if inner is bundle["expert"]:
        inner = Expert(copy.deepcopy(expert_config))
    rule = ExecutionGripperRule(expert_config)
    policy = DelegatedExecutionPolicy(inner, arm=arm, rule=rule)
    describe = {
        **bundle["describe"], "name": policy.name, "version": policy.version, "arm": policy.arm, "delegation_version": DELEGATION_VERSION,
        "inner": {"name": policy.inner_name, "version": policy.inner_version}, "rule": rule.describe(),
    }
    return {**bundle, "policy": policy, "describe": describe, "arm": policy.arm}


def rule_agreement(records: list[dict[str, Any]], rule: ExecutionGripperRule) -> dict[str, Any]:
    """저장된 폐루프 기록의 틱마다 실행 규칙 대 **그 틱의 전문가 참조 판단**(`evidence.expert.ticks[i].aux.gripper` — 전문가가 그 요청·
    commitment에서 실제로 낸 원하는 상태·이유; commitment가 없으면 지금 쥐고 있는지)을 대조한다. 정책이 무엇이었든(모델·규칙 판정기·전문가)
    전문가 참조는 틱마다 같은 요청에서 계산됐으므로, 다른 정책이 만든 다양한 상태에서 규칙 = 전문가를 확인하는 재료다."""
    ticks = agree = 0
    by_reason: dict[str, int] = {}
    mismatches: list[dict[str, Any]] = []
    for record in records:
        meta = ((record.get("evidence") or {}).get("expert") or {}).get("ticks") or []
        if len(meta) != len(record["ticks"]):
            raise ValueError(f"{record.get('episode_id')}: evidence.expert.ticks({len(meta)})가 틱({len(record['ticks'])})과 맞지 않는다")
        for index, (tick, expert) in enumerate(zip(record["ticks"], meta)):
            if int(expert["t"]) != int(tick["t"]):
                raise ValueError(f"{record.get('episode_id')} 틱 {index}: 전문가 기록의 t가 다르다")
            mine = rule.decide(tick, None)
            aux = expert.get("aux")
            if aux is None:
                holding = tick["request"]["state"]["robot"].get("holding")
                expected = {"desired": "closed" if holding else "open", "reason": "no_commitment"}
            else:
                expected = {"desired": aux["gripper"]["desired"], "reason": aux["gripper"]["reason"]}
            ticks += 1
            same = mine["desired"] == expected["desired"] and mine["reason"] == expected["reason"]
            agree += int(same)
            by_reason[expected["reason"]] = by_reason.get(expected["reason"], 0) + 1
            if not same and len(mismatches) < 20:
                mismatches.append({"episode_id": record.get("episode_id"), "tick": index, "rule": mine, "expert": expected})
    return {"episodes": len(records), "ticks": ticks, "agree": agree, "disagree": ticks - agree, "by_expert_reason": dict(sorted(by_reason.items())),
            "mismatches": mismatches}
