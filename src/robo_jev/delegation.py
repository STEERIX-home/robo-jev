"""실행 쪽 위임 — 그리퍼 닫기 시점(과 정지, 경로·힘)을 실행 층의 기하 규칙에 맡기는 **정책 감싸개** (Task R10 A1, R13 A1; docs/08 §10).

왜. R4~R9의 폐루프 실패의 한 갈래는 실행 수준 head의 불안정이다 — 그리퍼 head는 모든 seed에서 ≈ 100 step 평탄 뒤 늦게
맞고, `q_stop`은 세 seed 모두 정지 사건을 하나도 잡지 못했다. 그런데 전문가의 그리퍼 결정은 **기하 규칙**이다(국면별 기본 상태 +
파지점까지 `grasp_ready_mm`; 놓기의 `place_blocked`). 가설: 그리퍼 시점(과 힘 정지)은 의미 층이 아니라 실행 층의 일이고, 그것을 실행
쪽 규칙에 맡기면 모델의 실패 상당 부분이 사라진다. 이 모듈은 그 가설을 **학습 없이** 잰다. 하네스·컨트롤러·직렬화·expert·`contracts.py`는
바꾸지 않는다 — 하네스 버전 문자열이 계약 digest에 들어가므로 하네스를 바꾸면 기존 체크포인트가 적재를 거절당한다. 위임은 정책
자리에서 한다: 감싸개가 정책의 답 가운데 실행 층으로 옮길 답만 바꿔 하네스에 넘긴다(R6 A3 `DoneGatePolicy`와 같은 꼴).

R12의 판정(규칙_G − r12s19G 엄격 +0.080)에서 규칙 판정기만 성공한 25 seed는 모두 모델의 정체였고 그 가운데 부가 판단 정체가 14(경로 연속
불일치 13 + 힘·그리퍼 1)였다. 전문가의 힘 규칙(국면별 `force_by_phase`)은 지시 없이 계산되고, 경로 규칙(`Expert._path` — 요청의 경로 후보,
지금 commitment 후보의 `path_clear`, 손이 장애물 구 안인지)도 한 자리(아래 — 금지 물체의 추가 여유)만 빼면 그렇다 — 그래서 R13은 그 둘도 실행
쪽에 맡기는 갈래 X를 더한다. 속도는 국면별
값에 깨지기 쉬운 물체 근처 상한(`_near_fragile`)이 붙는데 그 상한은 `goal.fragile`(지시에서 온 정보)을 쓰므로 실행 쪽이 가질 수 없다 —
속도는 감싼 정책의 답으로 둔다.

갈래 셋 (:data:`ARMS`).

* **G** — 감싼 정책의 답을 그대로 쓰되 `q_gripper`만 실행 쪽 규칙(:class:`ExecutionGripperRule`)의 답으로 바꾼다.
* **GS** — G에 더해 `q_stop`을 언제나 거짓(0.0)으로 둔다 — 정지는 실행기의 힘 반사뿐이다(하네스가 반사 사건을 보면 답과 무관하게
  정지 분기로 간다, `RobotHarness._compose` 1단계).
* **X** (Task R13) — G에 더해 `q_path`(:class:`ExecutionPathRule`)와 `q_force`(:class:`ExecutionForceRule`)도 실행 쪽 규칙의 답이다.
  `q_speed`·`q_stop`과 모든 의미 답(`q_main`·게이트)은 감싼 정책의 것이다.

**실행 쪽 그리퍼 규칙** = 전문가 e0.4의 `Expert._gripper` — 국면별 기본 상태(`profiles.gripper_by_phase`), 파지 국면은 지금 commitment의 대상
파지점(윗면 아래 `grasp_depth_mm`)까지 `grasp_ready_mm` 안일 때만 `closed`, 놓기의 `open`은 내려가는 경로(direct·via)일 때만, 밀기는
접근·밀기 국면에서 주먹(`gripper_for_push`), commitment가 없으면 지금 쥐고 있는지. 그 규칙을 **지금 commitment와 관측 상태(요청의
`state`)·후보 목록만으로** 계산하도록 떼어 냈다 — 전문가 객체를 부르지 않고 그 설정의 수치만 든다. 목표·지시 문장·시뮬레이터
참값(`observation`)은 읽지 않는다. 같은 입력에서 전문가의 답과 같다는 것은 시험이 고정한다(`tests/test_delegation.py`: 합성 요청의 갈래마다
`Expert._gripper`와, 저장된 폐루프 기록의 틱마다 전문가의 `evidence.expert` 판단과, 전문가를 감싼 G가 감싸지 않은 전문가의 기록을 그대로
재현하는지).

**실행 쪽 경로 규칙** (xp0.1) = 전문가 e0.4의 `Expert._path` — 경로 후보가 없으면 답 없음, 고정 후보 commitment면 hold, 하강 구간이 비었으면
direct, 막혔고 경유점이 있으면 첫 경유점, 경유점도 없으면 손이 장애물의 넓힌 구 안일 때만 retreat 아니면 hold; commitment가 없으면 hold
경로(전문가 `act`의 `aux is None` 갈래). **한 군데가 전문가와 다르다**: 전문가의 "손이 장애물 구 안"(`_hand_inside_obstacle`)은 금지 접촉
물체의 구를 `planner.forbidden_margin_mm`만큼 더 넓히는데, 어느 물체가 금지인가는 목표의 `forbidden_contact`·물체의 `forbidden` 속성 —
**지시 문장이 말하는 정보**(docs/08 §3.2 v0.4: 모델 입력에서 `attr=`가 빠진 까닭)다. 실행 규칙은 그것을 읽지 않으므로 모든 물체를 외접 구 +
`planner.margin_mm`로 본다. 그래서 손이 금지 물체의 그 추가 여유 띠 안에만 있고 경로가 막혔고 경유점이 없는 틱에서 전문가는 retreat,
규칙은 hold를 답한다 — 그 틱 수는 저장된 기록에서 센다(:func:`aux_rule_agreement`의 `forbidden_margin`).
**실행 쪽 힘 규칙** (xf0.1) = 전문가의 국면별 값(`profiles.force_by_phase`, 실행기 힘 수준 수 안으로); commitment가 없으면 `0`.
두 규칙의 답의 분포는 전문가 `_spread`와 같은 질량·반올림·키 순서다(:func:`spread`).

**정보 경계.** 감싼 정책은 생성기가 주는 것을 그대로 받는다(`act(request, commitment, observation)` — 감싸지 않은 경로와 같은 인자);
`ModelPolicy`는 commitment·관측을 읽지 않는다(그 규칙 그대로). 감싼 정책은 실행 규칙의 답도 전문가의 답도 받지 않는다 — 감싸개가
그것들을 하네스 쪽으로만 넘긴다. 감싼 정책이 다음 틱에 보는 것은 하네스가 실제로 실행한 결과(`exec_history`의 그리퍼·경로·힘 등)뿐이고,
그것은 감싸지 않은 경로와 같은 되먹임이다.

**기록의 자리.** 생성기(:func:`robo_jev.data.robot_episodes.generate_episode`)는 정책이 돌려준 답을 `model_output`에 쓴다 — 이 정책의 답이므로
`model_output.q_gripper`(GS면 `q_stop`도, X면 `q_path`·`q_force`도)는 **실행 규칙의 값**(하네스가 실제로 받은 것)이다. 감싼 정책의 raw 답은 틱마다
`usage.delegation.inner`에, 실행 그리퍼 규칙의 판단(원하는 상태·이유·국면)은 `usage.delegation.rule`에(X면 경로·힘 규칙의 판단이
`usage.delegation.path_rule`·`force_rule`에), 편의 요약은 `evidence.delegation`에, 갈래·안쪽 정책은 `provenance.policy`에 남긴다
(:meth:`DelegatedExecutionPolicy.attach`) — 셋 다 계약 검증이 받는 자리다(`usage`·`evidence`·`provenance`는 dict이면 된다; `model_output`의 라벨 유출
검사와 무관하다). 갈래 G·GS의 기록은 R10과 바이트 단위로 같다(시험).
"""

from __future__ import annotations

import copy
import math
from typing import Any

from robo_jev.contracts import PHASES
from robo_jev.harness.robot import load_harness_config
from robo_jev.harness.rule_judge import candidate_values, normalise_distribution, read_goal
from robo_jev.perception.pointworld import circumradius_mm
from robo_jev.sim.controller import load_controller_config
from robo_jev.sim.expert import load_expert_config

__all__ = [
    "ARMS",
    "ARM_DELEGATION_VERSIONS",
    "DELEGATION_VERSION",
    "EXECUTION_FORCE_VERSION",
    "EXECUTION_GRIPPER_VERSION",
    "EXECUTION_PATH_VERSION",
    "DelegatedExecutionPolicy",
    "ExecutionForceRule",
    "ExecutionGripperRule",
    "ExecutionPathRule",
    "aux_rule_agreement",
    "committed_candidate",
    "delegate_bundle",
    "descent_path_kind",
    "rule_agreement",
    "spread",
]

#: 감싸개 버전 — 레코드의 `provenance.policy.version`은 ``<갈래의 감싸개 버전>/<갈래>/<안쪽 정책의 버전>``이다. G·GS는 R10의 dx0.1 그대로다.
DELEGATION_VERSION = "dx0.1"
#: 갈래마다의 감싸개 버전 — dx0.2 = dx0.1 + 갈래 X(Task R13). G·GS의 행동은 바뀌지 않았으므로 그 기록은 dx0.1을 그대로 적는다(바이트 단위로 같다 — 시험).
ARM_DELEGATION_VERSIONS = {"G": DELEGATION_VERSION, "GS": DELEGATION_VERSION, "X": "dx0.2"}
#: 실행 쪽 그리퍼 규칙의 버전 — 전문가 e0.4의 `_gripper`와 같은 규칙이다(시험이 고정).
EXECUTION_GRIPPER_VERSION = "xg0.1"
#: 실행 쪽 경로 규칙의 버전 — 전문가 e0.4의 `_path`에서 금지 물체의 추가 여유만 뺀 규칙이다(시험이 고정; 모듈 설명).
EXECUTION_PATH_VERSION = "xp0.1"
#: 실행 쪽 힘 규칙의 버전 — 전문가 e0.4의 국면별 힘 수준과 같은 규칙이다(시험이 고정).
EXECUTION_FORCE_VERSION = "xf0.1"
#: 갈래 — G: `q_gripper`만 실행 규칙; GS: G + `q_stop` 언제나 거짓(정지는 힘 반사만); X: G + `q_path`·`q_force`도 실행 규칙(R13).
ARMS = ("G", "GS", "X")
#: 그리퍼 답의 후보 순서 — 전문가 `_spread(desired, ["open", "closed"])`와 같은 순서라 직렬화한 바이트도 같다.
_GRIPPER_OPTIONS = ("open", "closed")


def spread(chosen: Any, options: list[Any], mass: float) -> dict[Any, float]:
    """고른 항목에 `mass`를 주고 나머지를 고르게 나눈 분포 — 전문가 `Expert._spread`와 같은 규칙(후보가 없으면 빈 분포, 고른 것이 후보에
    없으면 고른 분포, 같은 반올림·키 순서)이다. 실행 경로·힘 규칙의 답이 전문가의 답과 바이트 단위로 같게 하는 단일 자리다."""
    if not options:
        return {}
    if chosen not in options:
        return normalise_distribution({option: 1.0 / len(options) for option in options})
    rest = [option for option in options if option != chosen]
    share = (1.0 - float(mass)) / len(rest) if rest else 0.0
    return normalise_distribution({option: (float(mass) if option == chosen else share) for option in options})


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


class ExecutionPathRule:
    """실행 쪽 경로 규칙 (모듈 설명, Task R13) — 요청의 경로 후보, 지금 commitment 후보의 기능·`path_clear`, 관측 상태(말단 자세·쥔 물체·
    물체의 관측 자세와 OBB)만으로 전문가 e0.4의 `_path`를 계산한다. 목표·지시·물체 속성·시뮬레이터 관측 원본은 읽지 않고 전문가 객체를
    들지 않는다 — 그래서 "손이 장애물 구 안"에서 금지 물체의 추가 여유(`planner.forbidden_margin_mm`)를 쓰지 않는다(전문가와 다른 유일한 곳).

    수치는 전문가 설정(`confidence.choice_mass`)과 그것이 가리키는 하네스 설정(`planner.margin_mm`)에서 읽는다."""

    name = "ExecutionPathRule"
    version = EXECUTION_PATH_VERSION

    def __init__(self, expert_config: dict[str, Any] | None = None, *, harness_config: dict[str, Any] | None = None) -> None:
        config = copy.deepcopy(expert_config) if expert_config is not None else load_expert_config()
        self.expert_version = str(config.get("version", "unknown"))
        self.choice_mass = float(config["confidence"]["choice_mass"])
        harness = harness_config or load_harness_config(config["harness_config"])
        self.planner_margin_mm = float(harness["planner"]["margin_mm"])

    def describe(self) -> dict[str, Any]:
        return {"name": self.name, "version": self.version, "pinned_to": f"Expert._path ({self.expert_version})", "planner_margin_mm": self.planner_margin_mm,
                "differs_from_expert": "hand-inside-obstacle uses circumradius + planner.margin_mm for every observed object; the expert also adds "
                                       "planner.forbidden_margin_mm for forbidden objects (instruction information the execution rule does not read)"}

    def hand_inside_obstacle(self, state: dict[str, Any], value: dict[str, Any] | None) -> bool:
        """말단이 commitment 대상·쥔 물체가 아닌 관측 물체의 장애물 구(외접 구 + `planner.margin_mm`) 안인가 — 전문가 `_hand_inside_obstacle`에서
        금지 물체의 추가 여유만 뺀 것(어느 물체가 금지인지는 지시가 말하는 정보라 읽지 않는다)."""
        ee = [float(item) for item in state["robot"]["ee_pose_mm"]]
        skip = {str((value or {}).get("target")), str(state["robot"].get("holding"))}
        for entry in state.get("objects") or ():
            if str(entry["id"]) in skip:
                continue
            centre = [float(item) for item in entry["pose_mm"]]
            if math.dist(ee, centre) < circumradius_mm(entry["obb_mm"]) + self.planner_margin_mm:
                return True
        return False

    def decide(self, request: dict[str, Any], commitment: dict[str, Any] | None = None) -> dict[str, Any]:
        """고른 경로 후보와 그 종류·이유 — ``{"choice", "kind", "reason", "action_ref"}`` (전문가 `_path`의 갈래와 이름 그대로; commitment가 없으면
        hold 경로와 이유 `no_commitment`). 읽는 것: 요청의 commitment 투영(없으면 하네스 `commitment`), `q_main` 후보의 의미 키·경로 막힘,
        `q_path` 후보의 id·종류, 관측 상태의 `robot`(말단·쥔 물체)과 `objects`(관측 자세·OBB)."""
        model = request.get("request", request)
        state = model["state"]
        paths = list(model["candidates"].get("q_path") or [])
        values = {entry["id"]: candidate_values(entry) for entry in model["candidates"]["q_main"]}
        committed = committed_candidate(model, commitment, values)
        if committed is None:
            # commitment가 없는 틱: 부가 질문은 hold 기준 (전문가 `act`의 `aux is None` 갈래 — 첫 hold 경로, 없으면 고른 것 없음)
            hold = next((entry["id"] for entry in paths if entry.get("kind") == "hold"), None)
            return {"choice": hold, "kind": "hold" if hold is not None else None, "reason": "no_commitment", "action_ref": None}
        action_ref = str(committed["action_ref"])
        value = values.get(action_ref)
        options = [entry["id"] for entry in paths]
        by_kind: dict[str, list[str]] = {}
        for entry in paths:
            by_kind.setdefault(str(entry.get("kind")), []).append(str(entry["id"]))
        if not options:
            return {"choice": None, "kind": None, "reason": "no_paths", "action_ref": action_ref}
        if value is None or value.get("function") is None:
            return {"choice": (by_kind.get("hold") or options)[0], "kind": "hold", "reason": "fixed_candidate", "action_ref": action_ref}
        if value["path_clear"]:
            return {"choice": (by_kind.get("direct") or options)[0], "kind": "direct", "reason": "clear", "action_ref": action_ref}
        vias = by_kind.get("via") or []
        if vias:
            return {"choice": vias[0], "kind": "via", "reason": "blocked_via", "action_ref": action_ref}
        if by_kind.get("retreat") and self.hand_inside_obstacle(state, value):
            # 손이 장애물 구 안이면 어느 구간도 시작점에서 막힌다 — 물러난다 (매 틱 물러나면 팔이 한계까지 올라가므로 이때만)
            return {"choice": by_kind["retreat"][0], "kind": "retreat", "reason": "hand_inside_obstacle", "action_ref": action_ref}
        return {"choice": (by_kind.get("hold") or options)[0], "kind": "hold", "reason": "blocked_no_detour", "action_ref": action_ref}

    def distribution(self, choice: Any, options: list[Any]) -> dict[Any, float]:
        """고른 경로 → 답의 분포 (전문가 `_spread(choice, 경로 후보 id)`와 같은 질량·반올림·키 순서)."""
        return spread(choice, list(options), self.choice_mass)

    def answer(self, request: dict[str, Any], commitment: dict[str, Any] | None = None) -> dict[Any, float]:
        model = request.get("request", request)
        options = [entry["id"] for entry in model["candidates"].get("q_path") or []]
        return self.distribution(self.decide(request, commitment)["choice"], options)


class ExecutionForceRule:
    """실행 쪽 힘 규칙 (모듈 설명, Task R13) — 지금 commitment 후보의 국면별 힘 수준(`profiles.force_by_phase`, 실행기 힘 수준 수 안으로;
    모르는 국면은 `none`), commitment가 없으면 `0` — 전문가 e0.4의 `_aux`·`act`와 같은 값. 목표·지시·관측 원본은 읽지 않는다.

    수치는 전문가 설정(`profiles.force_by_phase`, `confidence.choice_mass`)과 그것이 가리키는 실행기 설정(`force_levels`의 수)에서 읽는다."""

    name = "ExecutionForceRule"
    version = EXECUTION_FORCE_VERSION

    def __init__(self, expert_config: dict[str, Any] | None = None, *, controller_config: dict[str, Any] | None = None) -> None:
        config = copy.deepcopy(expert_config) if expert_config is not None else load_expert_config()
        self.expert_version = str(config.get("version", "unknown"))
        self.force_by_phase = {str(phase): int(level) for phase, level in config["profiles"]["force_by_phase"].items()}
        self.choice_mass = float(config["confidence"]["choice_mass"])
        controller = controller_config or load_controller_config(config["controller_config"])
        self.levels = [str(index) for index in range(len(controller["force_levels"]))]

    def describe(self) -> dict[str, Any]:
        return {"name": self.name, "version": self.version, "pinned_to": f"Expert force_by_phase ({self.expert_version})",
                "force_by_phase": dict(self.force_by_phase), "levels": list(self.levels)}

    def decide(self, request: dict[str, Any], commitment: dict[str, Any] | None = None) -> dict[str, Any]:
        """힘 수준과 그 이유 — ``{"level", "reason", "phase"}``. 읽는 것: 요청의 commitment 투영(없으면 하네스 `commitment`)과 `q_main` 후보 id."""
        model = request.get("request", request)
        values = {entry["id"]: candidate_values(entry) for entry in model["candidates"]["q_main"]}
        committed = committed_candidate(model, commitment, values)
        if committed is None:
            return {"level": "0", "reason": "no_commitment", "phase": None}
        phase = str(committed.get("phase", "none"))
        if phase not in PHASES:
            phase = "none"
        return {"level": str(min(self.force_by_phase[phase], len(self.levels) - 1)), "reason": f"phase:{phase}", "phase": phase}

    def distribution(self, level: str) -> dict[str, float]:
        """힘 수준 → 답의 분포 (전문가 `_spread(level, 힘 수준)`과 같은 질량·반올림·키 순서)."""
        return spread(str(level), list(self.levels), self.choice_mass)

    def answer(self, request: dict[str, Any], commitment: dict[str, Any] | None = None) -> dict[str, float]:
        return self.distribution(self.decide(request, commitment)["level"])


def _argmax(answer: Any) -> str | None:
    """choice 답의 최댓값 — 하네스 `_argmax`와 같은 동점 규칙(확률 내림차순 → id 오름차순). 빈 답·dict가 아니면 None."""
    if not isinstance(answer, dict) or not answer:
        return None
    usable = {str(key): float(value) for key, value in answer.items() if str(key) in _GRIPPER_OPTIONS}
    if not usable:
        return None
    return min(usable, key=lambda key: (-usable[key], key))


def _choice_argmax(answer: Any, allowed: list[str]) -> str | None:
    """choice·ordinal 답에서 `allowed` 안의 최댓값 — 하네스 `_argmax`와 같은 동점 규칙(확률 내림차순 → id 오름차순). 쓸 답이 없으면 None
    (하네스는 그때 기본값을 쓴다)."""
    if not isinstance(answer, dict) or not answer:
        return None
    names = {str(item) for item in allowed}
    usable = {str(key): float(value) for key, value in answer.items() if str(key) in names}
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
    """감싼 정책의 답 + 실행 쪽 규칙의 `q_gripper`(GS면 `q_stop` = 0.0도, X면 `q_path`·`q_force`도) (모듈 설명). 평가 정책이다 — 수집 전용이 아니다."""

    name = "DelegatedExecutionPolicy"

    def __init__(
        self, inner: Any, *, arm: str, rule: ExecutionGripperRule | None = None, path_rule: ExecutionPathRule | None = None,
        force_rule: ExecutionForceRule | None = None,
    ) -> None:
        if arm not in ARMS:
            raise ValueError(f"arm: {list(ARMS)} 중 하나여야 한다 (받은 값: {arm!r})")
        if arm != "X" and (path_rule is not None or force_rule is not None):
            raise ValueError(f"arm {arm}: 경로·힘 규칙은 갈래 X에서만 쓴다")
        self.inner = inner
        self.arm = str(arm)
        self.rule = rule if rule is not None else ExecutionGripperRule()
        # 갈래 X(R13)만 경로·힘도 실행 규칙에 맡긴다 — G·GS는 R10 그대로(규칙 하나)
        self.path_rule = (path_rule if path_rule is not None else ExecutionPathRule()) if self.arm == "X" else None
        self.force_rule = (force_rule if force_rule is not None else ExecutionForceRule()) if self.arm == "X" else None
        self.delegation_version = ARM_DELEGATION_VERSIONS[self.arm]
        self.inner_name = str(getattr(inner, "name", type(inner).__name__))
        self.inner_version = str(getattr(inner, "version", "unknown"))
        self.version = f"{self.delegation_version}/{self.arm}/{self.inner_version}"
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
        if self.path_rule is not None and self.force_rule is not None:
            # 갈래 X(R13; 두 규칙은 X에서만 있다): 경로·힘도 실행 규칙 — 같은 요청·commitment만 읽는다(관측 원본·목표·전문가 없음). 감싼 정책의
            # 답은 이미 받았다
            path = self.path_rule.decide(request, commitment)
            force = self.force_rule.decide(request, commitment)
            model = request.get("request", request)
            options = [entry["id"] for entry in model["candidates"].get("q_path") or []]
            inner["q_path"] = copy.deepcopy(answers.get("q_path"))
            inner["q_force"] = copy.deepcopy(answers.get("q_force"))
            delegated["q_path"] = self.path_rule.distribution(path["choice"], options)
            delegated["q_force"] = self.force_rule.distribution(force["level"])
            self.raw.append({"t": int(request["t"]), "inner": inner, "rule": decision, "path_rule": path, "force_rule": force})
            return delegated
        self.raw.append({"t": int(request["t"]), "inner": inner, "rule": decision})
        return delegated

    def attach(self, record: dict[str, Any]) -> dict[str, Any]:
        """틱마다 감싼 정책의 raw 답과 실행 규칙의 판단을 `usage.delegation`에, 편 요약을 `evidence.delegation`에, 갈래·안쪽 정책을
        `provenance.policy`에 적는다. 틱 번호가 어긋나면 거절한다(raw는 생성기가 부른 `act` 순서 그대로다). 갈래 G·GS의 필드는 R10과 같다 —
        갈래 X만 경로·힘 규칙의 판단(`usage.delegation.path_rule`·`force_rule`)과 그 요약·버전을 더한다."""
        ticks = record["ticks"]
        if len(self.raw) != len(ticks) or any(int(entry["t"]) != int(tick["t"]) for entry, tick in zip(self.raw, ticks)):
            raise ValueError(f"{record.get('episode_id')}: 위임 raw 기록이 틱과 맞지 않는다 ({len(self.raw)}개 대 틱 {len(ticks)}개)")
        changed = inner_closed = rule_closed = stop_overridden = inner_silent = 0
        path_changed = path_silent = force_changed = force_silent = 0
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
            if self.force_rule is not None:
                # 갈래 X(R13)
                usage["delegation"]["path_rule"] = copy.deepcopy(entry["path_rule"])
                usage["delegation"]["force_rule"] = copy.deepcopy(entry["force_rule"])
                # 하네스가 받았을 감싼 정책의 답 — 그 틱의 경로 후보·힘 수준 안의 최댓값(하네스 `_argmax`의 동점 규칙)
                options = [str(item["id"]) for item in tick["request"]["candidates"].get("q_path") or []]
                inner_path = _choice_argmax(entry["inner"].get("q_path"), options)
                path_silent += int(inner_path is None)
                path_changed += int(inner_path is not None and inner_path != str(entry["path_rule"]["choice"]))
                inner_force = _choice_argmax(entry["inner"].get("q_force"), self.force_rule.levels)
                force_silent += int(inner_force is None)
                force_changed += int(inner_force is not None and inner_force != str(entry["force_rule"]["level"]))
        record["provenance"]["policy"] = {
            **record["provenance"]["policy"], "arm": self.arm, "delegation_version": self.delegation_version,
            "rule": {"name": self.rule.name, "version": self.rule.version}, "inner": {"name": self.inner_name, "version": self.inner_version},
        }
        summary: dict[str, Any] = {
            "version": self.delegation_version, "arm": self.arm, "rule": self.rule.describe(),
            "inner": {"name": self.inner_name, "version": self.inner_version}, "ticks": len(ticks),
            "q_gripper_changed_ticks": changed, "inner_q_gripper_closed_ticks": inner_closed, "inner_q_gripper_silent_ticks": inner_silent,
            "rule_closed_ticks": rule_closed,
            "note": "model_output.q_gripper is the execution rule's answer (what the harness received); the wrapped policy's raw answers are usage.delegation.inner",
        }
        if self.arm == "GS":
            summary["q_stop_overridden_ticks"] = stop_overridden
            summary["note"] += "; GS: model_output.q_stop is 0.0 on every tick (stopping is the executor's force reflex only)"
        if self.path_rule is not None and self.force_rule is not None:
            # 갈래 X(R13)
            record["provenance"]["policy"]["path_rule"] = {"name": self.path_rule.name, "version": self.path_rule.version}
            record["provenance"]["policy"]["force_rule"] = {"name": self.force_rule.name, "version": self.force_rule.version}
            summary.update({"path_rule": self.path_rule.describe(), "force_rule": self.force_rule.describe(),
                            "q_path_changed_ticks": path_changed, "inner_q_path_silent_ticks": path_silent,
                            "q_force_changed_ticks": force_changed, "inner_q_force_silent_ticks": force_silent})
            summary["note"] += ("; X: model_output.q_path and q_force are the execution path and force rules' answers too (q_speed, q_stop and every "
                                "semantic answer are the wrapped policy's)")
        record.setdefault("evidence", {})["delegation"] = summary
        return record


def delegate_bundle(bundle: dict[str, Any], *, arm: str, expert_config: dict[str, Any]) -> dict[str, Any]:
    """폐루프 정책 묶음(:func:`robo_jev.closed_loop.build_policy`)을 갈래 `arm`의 감싸개로 바꾼 새 묶음.

    참조(라벨·지연의 원천 `bundle["expert"]`)는 그대로다. 안쪽 정책이 그 참조 자신이면(전문가를 감쌀 때) 안쪽에는 **따로 만든** 전문가를
    둔다 — 같은 객체면 생성기가 참조를 정책의 답으로 대신하고(`policy is expert`), 참조 지연 계측도 두 번 쌓인다. 두 전문가는 상태가 없고
    결정적이므로 답은 같다. 갈래 X(R13)면 경로·힘 규칙도 같은 전문가 설정의 수치로 만들고 서술에 그 둘을 더한다(G·GS의 서술은 R10 그대로)."""
    from robo_jev.sim.expert import Expert

    inner = bundle["policy"]
    if inner is bundle["expert"]:
        inner = Expert(copy.deepcopy(expert_config))
    rule = ExecutionGripperRule(expert_config)
    extra: dict[str, Any] = {}
    if arm == "X":
        extra = {"path_rule": ExecutionPathRule(expert_config), "force_rule": ExecutionForceRule(expert_config)}
    policy = DelegatedExecutionPolicy(inner, arm=arm, rule=rule, **extra)
    describe = {
        **bundle["describe"], "name": policy.name, "version": policy.version, "arm": policy.arm, "delegation_version": policy.delegation_version,
        "inner": {"name": policy.inner_name, "version": policy.inner_version}, "rule": rule.describe(),
    }
    if arm == "X":
        describe.update({"path_rule": extra["path_rule"].describe(), "force_rule": extra["force_rule"].describe()})
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


#: 기록의 정책이 전문가 자신(감싸지 않았거나 G로 감싼 것)이면 `model_output.q_path`·`q_force`가 그 틱의 전문가 답 그대로다 — 분포·키 순서까지 대조할 수 있다.
_EXPERT_POLICY_NAMES = ("Expert", "TimedExpert")


def _is_expert_policy(record: dict[str, Any]) -> bool:
    policy = (record.get("provenance") or {}).get("policy") or {}
    if str(policy.get("name")) in _EXPERT_POLICY_NAMES:
        return True
    # 전문가를 G·GS로 감싼 기록(R10 expertG): 경로·힘은 안쪽 전문가의 답이다 (X로 감싼 기록은 규칙의 답이라 대조 재료가 아니다)
    return (str(policy.get("name")) == "DelegatedExecutionPolicy" and str((policy.get("inner") or {}).get("name")) in _EXPERT_POLICY_NAMES
            and str(policy.get("arm")) in ("G", "GS"))


def _forbidden_band(state: dict[str, Any], value: dict[str, Any] | None, *, margin_mm: float, forbidden_margin_mm: float) -> bool:
    """**분석용**(실행 규칙이 아니다 — 그래서 목표를 읽는다): 말단이 금지 물체(그 틱 목표의 `forbidden_contact` 또는 `forbidden` 속성 — 전문가
    `_hand_inside_obstacle`과 같은 집합)의 넓힌 구(외접 구 + margin + forbidden_margin) 안이고, 어느 물체의 외접 구 + margin 안도 아닌가 —
    곧 전문가와 실행 경로 규칙이 갈리는 유일한 자리인가."""
    ee = [float(item) for item in state["robot"]["ee_pose_mm"]]
    skip = {str((value or {}).get("target")), str(state["robot"].get("holding"))}
    forbidden = set(read_goal(state).forbidden)
    inside_plain = inside_forbidden = False
    for entry in state.get("objects") or ():
        if str(entry["id"]) in skip:
            continue
        distance = math.dist(ee, [float(item) for item in entry["pose_mm"]])
        limit = circumradius_mm(entry["obb_mm"]) + margin_mm
        inside_plain = inside_plain or distance < limit
        if str(entry["id"]) in forbidden or "forbidden" in (entry.get("attributes") or ()):
            inside_forbidden = inside_forbidden or distance < limit + forbidden_margin_mm
    return inside_forbidden and not inside_plain


def aux_rule_agreement(
    records: list[dict[str, Any]], path_rule: ExecutionPathRule, force_rule: ExecutionForceRule, *, forbidden_margin_mm: float | None = None,
    max_mismatches: int = 20,
) -> dict[str, Any]:
    """저장된 폐루프 기록의 틱마다 실행 경로·힘 규칙(Task R13) 대 **그 틱의 전문가 참조 판단**(`evidence.expert.ticks[i].aux.path`의 고른 경로·종류·
    이유와 `aux.force`의 수준; commitment가 없으면 첫 hold 경로·이유 `no_commitment`와 수준 `0` — 전문가 `act`의 `aux is None` 갈래)을 대조한다
    (:func:`rule_agreement`의 경로·힘 판). 정책이 무엇이었든 전문가 참조는 틱마다 같은 요청에서 계산됐다.

    경로 불일치마다 원인을 가른다: `forbidden_margin` — 전문가는 retreat(`hand_inside_obstacle`)이고 규칙은 hold(`blocked_no_detour`)이며 말단이 금지
    물체의 추가 여유 띠 안에만 있다(:func:`_forbidden_band` — 규칙이 금지 정보를 읽지 않아서 생기는 차, 모듈 설명); 그 밖은 `other`. 원인 가르기는
    **분석**이라 목표를 읽는다 — 실행 규칙은 읽지 않는다. 기록의 정책이 전문가 자신이면(`model_output`이 그 틱의 전문가 답) 규칙의 답 분포가
    `model_output.q_path`·`q_force`와 키 순서까지 같은지도 센다(`expert_answers`)."""
    if forbidden_margin_mm is None:
        forbidden_margin_mm = float(load_harness_config(load_expert_config()["harness_config"])["planner"]["forbidden_margin_mm"])
    ticks = 0
    path_agree = force_agree = 0
    path_by_reason: dict[str, int] = {}
    force_by_level: dict[str, int] = {}
    causes: dict[str, int] = {"forbidden_margin": 0, "other": 0}
    path_mismatches: list[dict[str, Any]] = []
    force_mismatches: list[dict[str, Any]] = []
    expert_ticks = path_equal = force_equal = 0
    answer_mismatches: list[dict[str, Any]] = []
    for record in records:
        meta = ((record.get("evidence") or {}).get("expert") or {}).get("ticks") or []
        if len(meta) != len(record["ticks"]):
            raise ValueError(f"{record.get('episode_id')}: evidence.expert.ticks({len(meta)})가 틱({len(record['ticks'])})과 맞지 않는다")
        expert_policy = _is_expert_policy(record)
        for index, (tick, expert) in enumerate(zip(record["ticks"], meta)):
            if int(expert["t"]) != int(tick["t"]):
                raise ValueError(f"{record.get('episode_id')} 틱 {index}: 전문가 기록의 t가 다르다")
            model = tick["request"]
            paths = list(model["candidates"].get("q_path") or [])
            aux = expert.get("aux")
            if aux is None:
                hold = next((entry["id"] for entry in paths if entry.get("kind") == "hold"), None)
                expected_path = {"choice": hold, "kind": "hold" if hold is not None else None, "reason": "no_commitment"}
                expected_force = "0"
            else:
                expected_path = {"choice": aux["path"]["choice"], "kind": aux["path"]["kind"], "reason": aux["path"]["reason"]}
                expected_force = str(aux["force"]["level"])
            path = path_rule.decide(tick, None)
            force = force_rule.decide(tick, None)
            ticks += 1
            path_by_reason[expected_path["reason"]] = path_by_reason.get(expected_path["reason"], 0) + 1
            force_by_level[expected_force] = force_by_level.get(expected_force, 0) + 1
            same_path = all(path[key] == expected_path[key] for key in ("choice", "kind", "reason"))
            path_agree += int(same_path)
            if not same_path:
                values = {entry["id"]: candidate_values(entry) for entry in model["candidates"]["q_main"]}
                committed = committed_candidate(model, None, values)
                value = values.get(str(committed["action_ref"])) if committed else None
                band = (expected_path["reason"] == "hand_inside_obstacle" and path["reason"] == "blocked_no_detour"
                        and _forbidden_band(model["state"], value, margin_mm=path_rule.planner_margin_mm, forbidden_margin_mm=float(forbidden_margin_mm)))
                cause = "forbidden_margin" if band else "other"
                causes[cause] += 1
                if len(path_mismatches) < max_mismatches:
                    path_mismatches.append({"episode_id": record.get("episode_id"), "tick": index, "cause": cause, "rule": path, "expert": expected_path})
            same_force = force["level"] == expected_force
            force_agree += int(same_force)
            if not same_force and len(force_mismatches) < max_mismatches:
                force_mismatches.append({"episode_id": record.get("episode_id"), "tick": index, "rule": force, "expert": expected_force})
            if expert_policy:
                output = tick.get("model_output") or {}
                options = [entry["id"] for entry in paths]
                mine_path = list(path_rule.distribution(path["choice"], options).items())
                mine_force = list(force_rule.distribution(force["level"]).items())
                expert_ticks += 1
                equal_path = mine_path == list((output.get("q_path") or {}).items())
                equal_force = mine_force == list((output.get("q_force") or {}).items())
                path_equal += int(equal_path)
                force_equal += int(equal_force)
                if (not equal_path or not equal_force) and len(answer_mismatches) < max_mismatches:
                    answer_mismatches.append({"episode_id": record.get("episode_id"), "tick": index, "rule_q_path": dict(mine_path), "expert_q_path": output.get("q_path"),
                                              "rule_q_force": dict(mine_force), "expert_q_force": output.get("q_force")})
    return {
        "episodes": len(records), "ticks": ticks,
        "q_path": {"agree": path_agree, "disagree": ticks - path_agree, "by_expert_reason": dict(sorted(path_by_reason.items())), "by_cause": causes,
                   "mismatches": path_mismatches},
        "q_force": {"agree": force_agree, "disagree": ticks - force_agree, "by_expert_level": dict(sorted(force_by_level.items())), "mismatches": force_mismatches},
        "expert_answers": {"ticks": expert_ticks, "q_path_equal": path_equal, "q_force_equal": force_equal, "mismatches": answer_mismatches},
    }
