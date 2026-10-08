"""실행 쪽 위임 검사 (Task R10 A1·A2) — 그리퍼 닫기 시점(과 정지)을 실행 층의 기하 규칙에 맡기는 정책 감싸개.

고정하는 것: (1) 실행 규칙 = 전문가 `_gripper` (합성 요청의 갈래마다, 실제 틱마다 살아 있는 전문가와, 저장된 폐루프 기록의 전문가 판단과);
(2) 규칙은 commitment와 관측 상태만 읽는다; (3) 감싸개는 `q_gripper`(GS면 `q_stop`도)만 바꾸고 감싼 정책의 raw 답을 기록에 남긴다;
(4) 감싸지 않은 경로는 바이트 단위로 그대로다(R9 기록 한 편을 재현); (5) 전문가를 감싼 G는 감싸지 않은 전문가의 기록을 재현한다; (6) CLI의 갈래.
R13 (갈래 X): (7) 실행 경로·힘 규칙 = 전문가 `_path`·국면별 힘 — 갈래마다, 살아 있는 전문가와, 저장 기록과 — 단 경로는 금지 물체의 추가 여유
띠에서만 다르다(규칙은 금지 정보를 읽지 않는다); (8) X는 `q_gripper`·`q_path`·`q_force`만 바꾼다; (9) G의 경로는 R12 기록 한 편을 바이트 단위로
재현하고, 전문가를 감싼 X는 전문가를 재현한다; (10) CLI의 갈래 X.
"""

import copy
import json
import math

import pytest
from helpers import REPO

from robo_jev.closed_loop import TimedExpert, build_policy, load_closed_loop_config, run_condition
from robo_jev.contracts import PHASES, QUESTION_SET_V0, validate_record
from robo_jev.data.robot_episodes import QUESTIONS, config_paths, load_generator_config
from robo_jev.data.sealed import read_episode_ids, read_open_episodes
from robo_jev.delegation import (
    ARM_DELEGATION_VERSIONS,
    ARMS,
    DELEGATION_VERSION,
    EXECUTION_FORCE_VERSION,
    EXECUTION_GRIPPER_VERSION,
    EXECUTION_PATH_VERSION,
    DelegatedExecutionPolicy,
    ExecutionForceRule,
    ExecutionGripperRule,
    ExecutionPathRule,
    _forbidden_band,
    aux_rule_agreement,
    delegate_bundle,
    descent_path_kind,
    rule_agreement,
    spread,
)
from robo_jev.harness.robot import load_harness_config
from robo_jev.harness.rule_judge import candidate_values
from robo_jev.perception.pointworld import circumradius_mm
from robo_jev.sim.expert import Expert, load_expert_config

GENERATOR = load_generator_config("configs/data/r1_robot.yaml")
#: 판정 조건과 같은 seed 파일·생성 설정(r1_robot)의 폐루프 설정 — dev_new2 장면을 다시 돌린다
R6_CONFIG = REPO / "configs/eval/r6-closed-loop.yaml"
#: 재현할 R9 기록 — seed 17의 dev_new2 한 편(E1, 지시 변경 1회, 54틱, 엄격 성공, 그리퍼 26틱 닫힘)
R9_RECORD = (REPO / "artifacts/datasets/r9-closed-loop/r9s17/dev_new2", "ep-E1-990376-r9-r9s17")
#: 전문가를 감싼 G가 재현할 R6 전문가 기록 — dev_new2 한 편(E1, 51틱, 파지·놓기·완료)
R6_EXPERT_RECORD = (REPO / "artifacts/datasets/r6-closed-loop/expert/dev_new2", "ep-E1-990297-r6-expert")
#: 저장된 폐루프 기록 가운데 규칙의 모든 갈래(이유)가 나오는 디렉터리 — 전문가·규칙 판정기·모델 둘이 몬 상태
AGREEMENT_DIRS = (
    "artifacts/datasets/r6-closed-loop/expert/dev_new2",
    "artifacts/datasets/r6-closed-loop/rule/dev_new2",
    "artifacts/datasets/r9-closed-loop/r9s17/dev_new2",
    "artifacts/datasets/r9-closed-loop/r9s18c/ood_dev",
)
ALL_REASONS = {"no_commitment", "idle", "holding", "at_grasp_point", "not_at_grasp_point", "place_blocked", "push_with_closed_fingers",
               "phase:approach", "phase:grasp", "phase:lift", "phase:transport", "phase:place"}


def _stored(where):
    """저장된 폐루프 기록 한 편 — manifest로 먼저(봉인 편·manifest에 없는 편은 열지 않는다), 그다음 그 파일의 바이트."""
    directory, episode_id = where
    path = directory / "episodes" / episode_id / "streams.jsonl"
    if not path.is_file():
        pytest.fail(f"저장된 폐루프 기록이 없다: {path}")
    (record,) = read_episode_ids(directory, [episode_id])
    raw = path.read_bytes()
    assert json.loads(raw.decode("utf-8")) == record
    return raw, record


@pytest.fixture(scope="module")
def rule():
    return ExecutionGripperRule()


@pytest.fixture(scope="module")
def expert():
    return Expert()


@pytest.fixture(scope="module")
def expert_ticks():
    """R6 전문가 dev_new2 기록의 틱 — 파지·놓기·밀기·막힌 놓기가 다 든 실제 요청 (앞 여섯 편 + 밀기·막힌 놓기가 든 두 편)."""
    directory = REPO / AGREEMENT_DIRS[0]
    names = sorted(path.name for path in (directory / "episodes").iterdir())[:6] + ["ep-E1-990158-r6-expert", "ep-E2-1000321-r6-expert"]
    return [tick for record in read_episode_ids(directory, names) for tick in record["ticks"]]


# --------------------------------------------------------------------------
# (1) 실행 규칙 = 전문가 `_gripper`
# --------------------------------------------------------------------------


def _expert_gripper(expert, request, commitment=None):
    """전문가가 같은 입력에서 내는 그리퍼 판단 — `Expert._gripper`를 그 자신의 `_committed`·`_path`로 부른 값."""
    model = request.get("request", request)
    values = {entry["id"]: candidate_values(entry) for entry in model["candidates"]["q_main"]}
    committed = expert._committed(model, commitment, values)
    if committed is None:
        return None
    phase = str(committed.get("phase", "none"))
    phase = phase if phase in PHASES else "none"
    value = values.get(str(committed["action_ref"]))
    path = expert._path(list(model["candidates"].get("q_path") or []), value, model["state"])
    return expert._gripper(model["state"], phase, value, path_kind=path["kind"])


def _variant(tick, *, phase=None, ee=None, holding="keep", key=None, path_ok=None, q_path=None, commit=True):
    """실제 틱 하나를 고친 사본 — commitment를 그 틱의 첫 결합 후보로 세우고 국면·말단·쥔 물체·후보 키·경로 막힘·경로 후보를 바꾼다."""
    out = copy.deepcopy(tick)
    model = out["request"]
    joint = next(entry for entry in model["candidates"]["q_main"] if str(entry.get("key", "")).startswith("grasp:"))
    if key is not None:
        joint["key"] = key
    if path_ok is not None:
        joint["path"] = "ok" if path_ok else "blocked"
    if q_path is not None:
        model["candidates"]["q_path"] = q_path
    if ee is not None:
        model["state"]["robot"]["ee_pose_mm"] = list(ee)
    if holding != "keep":
        model["state"]["robot"]["holding"] = holding
    model["commitment"] = ({"action_ref": joint["id"], "key": joint["key"], "phase": phase, "held_ticks": 3, "last_switch_tick": 0} if commit else None)
    return out, joint


def _grasp_point(rule, tick, joint):
    state = tick["request"]["state"]
    return rule.grasp_point(state, candidate_values(joint))


def test_the_execution_rule_equals_the_expert_gripper_on_every_branch_of_hand_made_requests(rule, expert, expert_ticks):
    """합성 요청의 갈래마다(국면 기본·파지점 안팎과 문턱 경계·쥔 채 파지·놓기의 direct·via·막힘·경로 없음·밀기 주먹·none의 current·모르는 국면·
    commitment 없음) 실행 규칙의 (원하는 상태, 이유)가 `Expert._gripper`와 같고, 답의 분포가 `Expert.act`의 `q_gripper`와 키 순서까지 같다."""
    base = next(tick for tick in expert_ticks if tick["request"]["state"]["robot"].get("holding") is None
                and any(str(entry.get("key", "")).startswith("grasp:") for entry in tick["request"]["candidates"]["q_main"]))
    _, joint = _variant(base, phase="approach")
    target = joint["key"].split(":")[1]
    point = _grasp_point(rule, base, joint)
    assert point is not None
    far = [point[0] + 200.0, point[1], point[2] + 150.0]
    inside = [point[0] + rule.grasp_ready_mm - 0.1, point[1], point[2]]
    outside = [point[0] + rule.grasp_ready_mm + 0.1, point[1], point[2]]
    paths_with_via = [{"id": "p1", "kind": "direct"}, {"id": "p2", "kind": "via", "ref": "w1"}, {"id": "p3", "kind": "hold"}]
    paths_without_via = [{"id": "p1", "kind": "direct"}, {"id": "p3", "kind": "hold"}, {"id": "p4", "kind": "retreat"}]
    cases = {
        "approach": dict(phase="approach"),
        "grasp_far": dict(phase="grasp", ee=far),
        "grasp_inside_threshold": dict(phase="grasp", ee=inside),
        "grasp_outside_threshold": dict(phase="grasp", ee=outside),
        "grasp_at_point": dict(phase="grasp", ee=point),
        "grasp_while_holding": dict(phase="grasp", ee=far, holding=target),
        "lift": dict(phase="lift", holding=target),
        "transport": dict(phase="transport", holding=target),
        "place_clear": dict(phase="place", holding=target, path_ok=True),
        "place_blocked_with_via": dict(phase="place", holding=target, path_ok=False, q_path=paths_with_via),
        "place_blocked_without_via": dict(phase="place", holding=target, path_ok=False, q_path=paths_without_via),
        "place_no_paths": dict(phase="place", holding=target, path_ok=True, q_path=[]),
        "push_approach": dict(phase="approach", key=f"push:{target}:+x:none"),
        "push_push": dict(phase="push", key=f"push:{target}:-x:none"),
        "push_while_holding": dict(phase="push", key=f"push:{target}:+x:none", holding=target),
        "none_idle": dict(phase="none"),
        "none_holding": dict(phase="none", holding=target),
        "unknown_phase": dict(phase="hover"),
        "no_commitment": dict(commit=False),
        "no_commitment_holding": dict(commit=False, holding=target),
    }
    reasons = set()
    for name, change in cases.items():
        tick, _ = _variant(base, **change)
        mine = rule.decide(tick, None)
        expected = _expert_gripper(expert, tick)
        if expected is None:
            holding = tick["request"]["state"]["robot"].get("holding")
            expected = {"desired": "closed" if holding else "open", "reason": "no_commitment"}
        assert (mine["desired"], mine["reason"]) == (expected["desired"], expected["reason"]), name
        answers = expert.act(tick, None)
        assert list(rule.answer(tick, None).items()) == list(answers["q_gripper"].items()), name
        reasons.add(mine["reason"])
    assert {"not_at_grasp_point", "at_grasp_point", "place_blocked", "push_with_closed_fingers", "idle", "holding", "no_commitment"} <= reasons
    # 문턱 경계는 전문가처럼 "거리 > grasp_ready_mm이면 열기"다
    assert rule.decide(_variant(base, phase="grasp", ee=inside)[0])["desired"] == "closed"
    assert rule.decide(_variant(base, phase="grasp", ee=outside)[0])["desired"] == "open"


def test_the_execution_rule_equals_a_live_expert_on_real_requests_with_and_without_the_harness_commitment(rule, expert, expert_ticks):
    """R6 전문가 기록의 실제 요청마다: 규칙의 답 = 살아 있는 전문가 `act`의 `q_gripper`(분포·키 순서까지)와 그 판단(`aux.gripper`) — 요청의
    commitment 투영만 줄 때와, 같은 후보를 가리키는 하네스 commitment를 함께 줄 때 모두."""
    assert len(expert_ticks) > 500
    for index, tick in enumerate(expert_ticks):
        projected = tick["request"].get("commitment")
        harness_commitment = None if projected is None else {**projected, "goal_version": 1, "stop_ticks": 0}
        for commitment in (None, harness_commitment):
            mine = rule.decide(tick, commitment)
            answers = expert.act(tick, commitment)
            aux = answers["expert_meta"]["aux"]
            if aux is None:
                assert mine["reason"] == "no_commitment" and mine["desired"] == ("closed" if tick["request"]["state"]["robot"].get("holding") else "open"), index
            else:
                assert (mine["desired"], mine["reason"]) == (aux["gripper"]["desired"], aux["gripper"]["reason"]), index
            assert list(rule.distribution(mine["desired"]).items()) == list(answers["q_gripper"].items()), index


def test_the_execution_rule_agrees_with_the_experts_recorded_decision_on_stored_closed_loop_ticks(rule):
    """저장된 폐루프 기록(R6 전문가·규칙 판정기, R9 seed 17·seed 18 — 서로 다른 정책이 몬 상태)의 틱마다 규칙 = 그 틱의 전문가 참조 판단
    (`evidence.expert`), 그리고 그 틱들이 규칙의 모든 갈래(이유)를 지난다."""
    reasons: dict[str, int] = {}
    total = 0
    for directory in AGREEMENT_DIRS:
        records = read_open_episodes(REPO / directory)[0]
        if not records:
            pytest.fail(f"저장된 폐루프 기록이 없다: {directory}")
        out = rule_agreement(records, rule)
        assert out["disagree"] == 0 and out["agree"] == out["ticks"] > 0, (directory, out["mismatches"][:3])
        total += out["ticks"]
        for reason, count in out["by_expert_reason"].items():
            reasons[reason] = reasons.get(reason, 0) + count
    assert ALL_REASONS <= set(reasons) and total > 30_000


def test_rule_agreement_names_a_tick_where_the_rule_and_the_recorded_expert_differ(rule):
    """대조가 실제로 대조하는가 — 저장된 전문가 판단 하나를 뒤집으면 그 틱이 불일치로 적힌다(어긋난 틱 수가 다르면 거절)."""
    record = copy.deepcopy(_stored(R6_EXPERT_RECORD)[1])
    index = next(i for i, meta in enumerate(record["evidence"]["expert"]["ticks"]) if (meta.get("aux") or {}).get("gripper", {}).get("reason") == "at_grasp_point")
    record["evidence"]["expert"]["ticks"][index]["aux"]["gripper"]["desired"] = "open"
    out = rule_agreement([record], rule)
    assert out["disagree"] == 1 and out["mismatches"][0]["tick"] == index and out["mismatches"][0]["expert"]["desired"] == "open"
    record["evidence"]["expert"]["ticks"].pop()
    with pytest.raises(ValueError, match="evidence.expert"):
        rule_agreement([record], rule)


def test_descent_path_kind_is_the_experts_path_kind_where_the_gripper_rule_reads_it():
    """하강 경로 종류: 경로 후보 없음 → None, 고정 후보 → hold, 비었음 → direct, 막혔고 경유점 있음 → via, 막혔고 경유점 없음 → blocked."""
    clear = {"function": "grasp", "path_clear": True}
    blocked = {"function": "place", "path_clear": False}
    paths = [{"id": "p1", "kind": "direct"}, {"id": "p2", "kind": "via"}]
    assert descent_path_kind([], clear) is None
    assert descent_path_kind(paths, None) == "hold" and descent_path_kind(paths, {"function": None, "path_clear": True}) == "hold"
    assert descent_path_kind(paths, clear) == "direct" and descent_path_kind(paths, blocked) == "via"
    assert descent_path_kind([{"id": "p1", "kind": "direct"}, {"id": "p4", "kind": "retreat"}], blocked) == "blocked"


# --------------------------------------------------------------------------
# (2) 규칙은 commitment와 관측 상태만 읽는다
# --------------------------------------------------------------------------


def test_the_rule_reads_the_commitment_and_the_observed_state_but_not_the_goal_or_the_simulator_observation(rule, expert_ticks):
    """목표(문장·대상·영역·금지)·관측 원본을 바꿔도 같은 판단이고, commitment·말단 자세를 바꾸면 판단이 바뀐다. 규칙은 전문가 객체를 들지 않는다."""
    tick = next(t for t in expert_ticks if (t["request"].get("commitment") or {}).get("phase") == "grasp" and t["request"]["state"]["robot"].get("holding") is None)
    before = rule.decide(tick, None)
    changed = copy.deepcopy(tick)
    goal = changed["request"]["state"]["goal"]
    goal.update({"text": "put everything somewhere else", "target_ref": "nothing", "target_zone": "zoneZ", "forbidden_contact": ["o1", "o2", "o3"]})
    assert rule.decide(changed, None) == before
    assert rule.decide(tick, None) == rule.decide(tick, {"action_ref": "no-such-candidate", "phase": "place"})  # 투영이 우선이다
    joint = rule.grasp_point(tick["request"]["state"], candidate_values(next(e for e in tick["request"]["candidates"]["q_main"] if e["id"] == tick["request"]["commitment"]["action_ref"])))
    moved = copy.deepcopy(tick)
    moved["request"]["state"]["robot"]["ee_pose_mm"] = list(joint) if before["desired"] == "open" else [joint[0] + 300.0, joint[1], joint[2] + 300.0]
    assert rule.decide(moved, None)["desired"] != before["desired"]
    uncommitted = copy.deepcopy(tick)
    uncommitted["request"]["commitment"] = None
    assert rule.decide(uncommitted, None)["reason"] == "no_commitment"
    assert not any(isinstance(value, Expert) for value in vars(rule).values())
    # 감싸개는 시뮬레이터 관측 원본을 규칙에 넘기지 않는다 — 관측이 달라도 하네스가 받는 그리퍼 답과 규칙의 판단이 같다
    wrapper = DelegatedExecutionPolicy(Fixed(), arm="G", rule=rule)
    first = wrapper.act(tick, None, {"objects": [{"id": "o1", "pos_mm": [0, 0, 0]}]})["q_gripper"]
    second = wrapper.act(tick, None, {"objects": [], "anything": "else"})["q_gripper"]
    assert first == second == rule.answer(tick, None) and wrapper.raw[0]["rule"] == wrapper.raw[1]["rule"] == before


# --------------------------------------------------------------------------
# (3) 감싸개
# --------------------------------------------------------------------------


class Fixed:
    """정해진 답을 내는 가짜 정책 — 받은 인자를 기록한다."""

    name = "Fixed"
    version = "fixed-0"

    def __init__(self) -> None:
        self.calls: list[tuple] = []
        self.resets = 0
        self.timing = [{"t": 0, "model_ms": 1.0}]

    def reset(self) -> None:
        self.resets += 1

    def act(self, request, commitment=None, observation=None):
        self.calls.append((request, commitment, observation))
        answers = {qid: (0.2 if QUESTION_SET_V0[qid]["type"] == "boolean" else {}) for qid in QUESTION_SET_V0}
        answers.update({"q_gripper": {"open": 0.2, "closed": 0.8}, "q_stop": 0.9, "q_done": 0.1})
        return answers


def test_arm_g_replaces_only_q_gripper_and_arm_gs_also_sets_q_stop_false(rule, expert_ticks):
    """G: `q_gripper`만 실행 규칙의 답이고 나머지 아홉 답은 감싼 정책의 것 그대로; GS: 거기에 `q_stop` 0.0. raw 기록은 감싼 정책의 답과 규칙의 판단."""
    tick = expert_ticks[10]
    for arm in ("G", "GS"):  # R10의 두 갈래 (X는 아래 R13 시험)
        inner = Fixed()
        policy = DelegatedExecutionPolicy(inner, arm=arm, rule=rule)
        assert policy.version == f"{DELEGATION_VERSION}/{arm}/fixed-0" and policy.name == "DelegatedExecutionPolicy"
        out = policy.act(tick, None, {"sim": "ignored"})
        inner_answers = Fixed().act(tick)
        differs = {qid for qid in QUESTION_SET_V0 if out[qid] != inner_answers[qid]}
        assert differs <= {"q_gripper", "q_stop"} and ("q_stop" in differs) == (arm == "GS")
        assert out["q_gripper"] == rule.answer(tick, None) != inner_answers["q_gripper"]
        assert out["q_stop"] == (0.0 if arm == "GS" else 0.9)
        assert policy.raw == [{"t": int(tick["t"]), "inner": {"q_gripper": {"open": 0.2, "closed": 0.8}, **({"q_stop": 0.9} if arm == "GS" else {})}, "rule": rule.decide(tick, None)}]
        policy.reset()
        assert policy.raw == [] and inner.resets == 1 and policy.timing == inner.timing


def test_the_wrapped_policy_receives_exactly_the_generators_arguments_and_the_wrapper_never_calls_the_expert(rule, expert_ticks, monkeypatch):
    """정보 경계: 감싼 정책은 감싸지 않은 경로와 같은 (요청, commitment, 관측) 객체를 받고, 감싸개는 전문가를 부르지 않는다
    (전문가 `act`·`_gripper`를 막아도 돈다) — 규칙은 설정의 수치만 든다."""

    def forbidden(*args, **kwargs):
        raise AssertionError("감싸개가 전문가를 불렀다")

    monkeypatch.setattr(Expert, "act", forbidden)
    monkeypatch.setattr(Expert, "_gripper", forbidden)
    inner = Fixed()
    policy = DelegatedExecutionPolicy(inner, arm="GS", rule=rule)
    tick, commitment, observation = expert_ticks[20], {"action_ref": "x"}, {"sim": True}
    policy.act(tick, commitment, observation)
    received = inner.calls[0]
    assert received[0] is tick and received[1] is commitment and received[2] is observation


def test_an_unknown_arm_is_refused():
    # R13이 X를 갈래로 더했다 — 모르는 갈래의 예는 Z다
    with pytest.raises(ValueError, match="arm"):
        DelegatedExecutionPolicy(Fixed(), arm="Z")


def test_attach_refuses_a_raw_log_that_does_not_line_up_with_the_ticks(rule, expert_ticks):
    policy = DelegatedExecutionPolicy(Fixed(), arm="G", rule=rule)
    policy.act(expert_ticks[0])
    with pytest.raises(ValueError, match="틱"):
        policy.attach({"episode_id": "e", "ticks": [{"t": 0}, {"t": 5}], "provenance": {"policy": {}}})


def test_build_policy_delegates_with_a_separate_inner_expert_and_names_the_arm():
    """`build_policy(..., delegate=…)`: 전문가를 감싸면 안쪽은 참조와 **다른** 전문가 객체(생성기가 참조를 정책의 답으로 대신하지 않게);
    규칙 판정기를 감싸면 안쪽이 그 정책; 서술에 감싸개 이름·버전·갈래·안쪽 정책·규칙; 감싸지 않으면 그대로; 모르는 갈래는 거절."""
    plain = build_policy("expert", generator=GENERATOR)
    assert plain["policy"] is plain["expert"] and "arm" not in plain and plain["describe"] == {"name": "Expert", "version": "e0.4", "kind": "expert"}
    wrapped = build_policy("expert", generator=GENERATOR, delegate="G")
    assert isinstance(wrapped["policy"], DelegatedExecutionPolicy) and wrapped["arm"] == "G"
    assert isinstance(wrapped["expert"], TimedExpert) and wrapped["policy"].inner is not wrapped["expert"] and isinstance(wrapped["policy"].inner, Expert)
    rule_gs = build_policy("rule", generator=GENERATOR, delegate="GS")
    describe = rule_gs["describe"]
    assert (describe["name"], describe["version"], describe["arm"], describe["kind"]) == ("DelegatedExecutionPolicy", f"{DELEGATION_VERSION}/GS/rj0.5", "GS", "rule")
    assert describe["inner"] == {"name": "RuleJudge", "version": "rj0.5"} and describe["rule"]["version"] == EXECUTION_GRIPPER_VERSION
    assert describe["rule"]["pinned_to"] == "Expert._gripper (e0.4)" and describe["rule"]["grasp_ready_mm"] == 15.0
    with pytest.raises(ValueError, match="arm"):
        build_policy("rule", generator=GENERATOR, delegate="S")
    expert_config = load_expert_config(config_paths(GENERATOR)["expert_config"])
    again = delegate_bundle(build_policy("mechanical", generator=GENERATOR), arm="G", expert_config=expert_config)
    assert again["policy"].inner_name == "MechanicalPolicy" and again["describe"]["kind"] == "mechanical"


# --------------------------------------------------------------------------
# (4)·(5) 기록 — 감싸지 않은 경로는 그대로, 전문가를 감싼 G는 전문가를 재현한다
# --------------------------------------------------------------------------


class Replay:
    """저장된 기록의 `model_output`을 틱 순서대로 되돌려 주는 정책 — 기록이 쓴 정책 이름·버전 그대로."""

    def __init__(self, record) -> None:
        self.record = record
        self.name = record["provenance"]["policy"]["name"]
        self.version = record["provenance"]["policy"]["version"]
        self.index = 0

    def reset(self) -> None:
        self.index = 0

    def act(self, request, commitment=None, observation=None):
        tick = self.record["ticks"][self.index]
        assert int(request["t"]) == int(tick["t"])
        self.index += 1
        return {qid: copy.deepcopy(tick["model_output"][qid]) for qid in QUESTIONS}


def _bundle(policy, kind="model"):
    expert = TimedExpert(Expert(load_expert_config(config_paths(GENERATOR)["expert_config"])))
    return {"kind": kind, "policy": policy, "expert": expert, "describe": {"name": policy.name, "version": policy.version, "kind": kind}}


def _written(out, condition, episode_id):
    return (out / condition / "episodes" / episode_id / "streams.jsonl").read_bytes()


def _same_wall(raw: bytes, wall: float) -> bytes:
    record = json.loads(raw.decode("utf-8"))
    record["provenance"]["timing"]["wall_s"] = wall
    return (json.dumps(record, ensure_ascii=False, sort_keys=False, separators=(",", ":")) + "\n").encode("utf-8")


def test_the_unwrapped_path_reproduces_a_stored_r9_record_byte_for_byte(tmp_path):
    """감싸지 않은 경로(폐루프 실행기 → 생성기 → 하네스·컨트롤러·시뮬레이터 → 계약 검사 → 기록)가 R9 seed 17의 dev_new2 한 편을 그 기록의
    모델 답으로 다시 돌리면 **바이트 단위로** 같은 파일을 쓴다 — 다른 것은 벽시계(`provenance.timing.wall_s`) 하나다."""
    raw, stored = _stored(R9_RECORD)
    config = load_closed_loop_config(R6_CONFIG)
    profile, seed = stored["provenance"]["profile"], int(stored["provenance"]["seed"])
    run = run_condition(_bundle(Replay(stored)), [(profile, seed)], config=config, out=tmp_path / "dev_new2", condition="dev_new2", label="r9s17", id_tag="r9")
    written = _written(tmp_path, "dev_new2", stored["episode_id"])
    assert run["episodes"][0]["episode_id"] == stored["episode_id"]
    # 벽시계는 소수 셋째 자리까지라 자릿수가 달라질 수 있다 — 그 값 하나를 기록의 값으로 되돌린 바이트가 같아야 한다
    assert _same_wall(written, stored["provenance"]["timing"]["wall_s"]) == raw


def test_the_expert_wrapped_in_arm_g_reproduces_the_stored_unwrapped_expert_record(tmp_path):
    """전문가를 감싼 G는 감싸지 않은 전문가(R6 dev_new2 한 편)와 **같은 편**을 만든다: 실행 규칙의 답이 전문가의 `q_gripper`와 같으므로 명령이
    같고, 감싸개가 붙인 것(`provenance.policy`의 이름·버전·갈래·안쪽, `usage.delegation`, `evidence.delegation`)과 에피소드 id의 이름표·벽시계 말고는
    바이트 단위로 같다. 같은 장면의 감싸지 않은 전문가도 그 기록을 바이트 단위로 재현한다."""
    raw, stored = _stored(R6_EXPERT_RECORD)
    config = load_closed_loop_config(R6_CONFIG)
    schedule = [(stored["provenance"]["profile"], int(stored["provenance"]["seed"]))]
    plain = build_policy("expert", generator=config["generator"])
    run_condition(plain, schedule, config=config, out=tmp_path / "plain" / "dev_new2", condition="dev_new2", label="expert", id_tag="r6")
    assert _same_wall(_written(tmp_path / "plain", "dev_new2", stored["episode_id"]), stored["provenance"]["timing"]["wall_s"]) == raw
    wrapped = build_policy("expert", generator=config["generator"], delegate="G")
    run_condition(wrapped, schedule, config=config, out=tmp_path / "g" / "dev_new2", condition="dev_new2", label="expertG", id_tag="r6")
    record = json.loads(_written(tmp_path / "g", "dev_new2", stored["episode_id"].replace("-r6-expert", "-r6-expertG")))
    summary = record["evidence"].pop("delegation")
    assert summary["arm"] == "G" and summary["q_gripper_changed_ticks"] == 0 and summary["ticks"] == len(stored["ticks"])
    assert record["provenance"]["policy"] == {"name": "DelegatedExecutionPolicy", "version": f"{DELEGATION_VERSION}/G/e0.4", "arm": "G",
                                              "delegation_version": DELEGATION_VERSION, "rule": {"name": "ExecutionGripperRule", "version": EXECUTION_GRIPPER_VERSION},
                                              "inner": {"name": "Expert", "version": "e0.4"}}
    for tick in record["ticks"]:
        delegation = tick["usage"].pop("delegation")
        assert delegation["inner"]["q_gripper"] == tick["model_output"]["q_gripper"]
    record["provenance"]["policy"] = stored["provenance"]["policy"]
    record["episode_id"] = stored["episode_id"]
    record["provenance"]["timing"]["wall_s"] = stored["provenance"]["timing"]["wall_s"]
    assert (json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8") == raw


def test_a_wrapped_rule_judge_record_keeps_its_raw_answers_in_usage_and_the_harness_received_the_rules(tmp_path):
    """규칙 판정기를 G·GS로 감싼 짧은 run: 기록이 계약을 지나고, 틱마다 `model_output.q_gripper` = 그 틱 요청에서 다시 계산한 실행 규칙의 답,
    `usage.delegation.inner` = 규칙 판정기가 그 요청에 내는 raw 답(GS면 `q_stop`도), GS의 `model_output.q_stop`은 언제나 0.0; manifest·provenance·
    evidence에 갈래가 적힌다."""
    from robo_jev.data.dagger import rule_judge_policy

    config = load_closed_loop_config(R6_CONFIG)
    rule, judge = ExecutionGripperRule(), rule_judge_policy()
    for arm in ("G", "GS"):  # R10의 두 갈래 (X는 아래 R13 시험)
        bundle = build_policy("rule", generator=config["generator"], delegate=arm)
        run = run_condition(bundle, [("E0", 980108)], config=config, out=tmp_path / arm / "dev_new2", condition="dev_new2", label=f"rule{arm}", id_tag="r10", max_ticks=40)
        manifest = json.loads((tmp_path / arm / "dev_new2" / "manifest.json").read_text(encoding="utf-8"))
        assert manifest["closed_loop"]["arm"] == arm and manifest["closed_loop"]["policy"]["arm"] == arm
        record = json.loads(_written(tmp_path / arm, "dev_new2", run["episodes"][0]["episode_id"]))
        validate_record(record)
        assert record["provenance"]["policy"]["arm"] == arm and record["provenance"]["policy"]["inner"] == {"name": "RuleJudge", "version": "rj0.5"}
        assert record["evidence"]["delegation"]["arm"] == arm and record["evidence"]["delegation"]["ticks"] == len(record["ticks"]) == 40
        changed = 0
        for tick in record["ticks"]:
            delegation = tick["usage"]["delegation"]
            raw = judge.act(tick)
            assert tick["model_output"]["q_gripper"] == rule.answer(tick, None) and delegation["rule"] == rule.decide(tick, None)
            assert delegation["inner"]["q_gripper"] == raw["q_gripper"]
            inner_choice = min(raw["q_gripper"], key=lambda key: (-raw["q_gripper"][key], key))  # 하네스 `_argmax`의 동점 규칙
            changed += int(inner_choice != delegation["rule"]["desired"])
            if arm == "GS":
                assert tick["model_output"]["q_stop"] == 0.0 and delegation["inner"]["q_stop"] == raw["q_stop"]
            else:
                assert tick["model_output"]["q_stop"] == raw["q_stop"] and "q_stop" not in delegation["inner"]
            for qid in ("q_main", "q_done", "q_instr", "q_observe", "q_retry", "q_path", "q_speed", "q_force"):
                assert tick["model_output"][qid] == raw[qid]
        assert record["evidence"]["delegation"]["q_gripper_changed_ticks"] == changed


# --------------------------------------------------------------------------
# (6) CLI
# --------------------------------------------------------------------------


def _script():
    import importlib.util

    spec = importlib.util.spec_from_file_location("closed_loop_script_r10", REPO / "scripts" / "closed_loop.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_seed_files_are_merged_by_condition_and_a_repeated_condition_is_refused(tmp_path):
    module = _script()
    first, second, clash = tmp_path / "a.json", tmp_path / "b.json", tmp_path / "c.json"
    first.write_text(json.dumps({"conditions": {"ood_dev": {"seeds": [1]}}}), encoding="utf-8")
    second.write_text(json.dumps({"conditions": {"ood_dev_new": {"seeds": [2]}, "dev_new2": {"seeds": [3]}}}), encoding="utf-8")
    clash.write_text(json.dumps({"conditions": {"ood_dev": {"seeds": [4]}}}), encoding="utf-8")
    assert module.read_seed_files([str(first)]) == {"conditions": {"ood_dev": {"seeds": [1]}}}
    merged = module.read_seed_files([str(first), str(second)])
    assert merged["conditions"] == {"ood_dev": {"seeds": [1]}, "ood_dev_new": {"seeds": [2]}, "dev_new2": {"seeds": [3]}}
    with pytest.raises(ValueError, match="ood_dev"):
        module.read_seed_files([str(first), str(clash)])


def test_the_run_command_takes_an_arm_and_puts_it_in_the_label_and_the_run_file(tmp_path):
    """`closed_loop.py run --delegate G`: 이름표 = 정책 이름표 + 갈래(ruleG), 에피소드 id 꼬리 `-<tag>-ruleG`, run 파일의 `arm`; 감싸지 않은 run 파일에는
    `arm`이 없다. seed 파일을 둘 주면 두 파일의 조건을 한 프로세스에서 돈다."""
    module = _script()
    report = tmp_path / "run.json"
    args = ["run", "--policy", "rule", "--delegate", "G", "--id-tag", "r10t", "--condition", "dev_new2,ood_dev", "--config", str(R6_CONFIG),
            "--seeds", str(REPO / "artifacts/reports/r6-seeds.json"), str(REPO / "artifacts/reports/r4-seeds.json"),
            "--limit", "1", "--max-ticks", "6", "--out", str(tmp_path / "rule"), "--report", str(report)]
    assert module.main(args) == 0
    payload = json.loads(report.read_text(encoding="utf-8"))
    assert payload["label"] == "ruleG" and payload["arm"] == "G" and payload["policy"]["arm"] == "G" and set(payload["conditions"]) == {"dev_new2", "ood_dev"}
    assert payload["seeds_file"] == [str(REPO / "artifacts/reports/r6-seeds.json"), str(REPO / "artifacts/reports/r4-seeds.json")]
    for condition in ("dev_new2", "ood_dev"):
        assert all(row["episode_id"].endswith("-r10t-ruleG") for row in payload["conditions"][condition]["episodes"])
    plain = tmp_path / "plain.json"
    args = ["run", "--policy", "rule", "--id-tag", "r10t", "--condition", "dev_new2", "--config", str(R6_CONFIG), "--seeds", str(REPO / "artifacts/reports/r6-seeds.json"),
            "--limit", "1", "--max-ticks", "4", "--out", str(tmp_path / "plain"), "--report", str(plain)]
    assert module.main(args) == 0
    payload = json.loads(plain.read_text(encoding="utf-8"))
    assert payload["label"] == "rule" and "arm" not in payload and payload["seeds_file"] == str(REPO / "artifacts/reports/r6-seeds.json")


def test_the_grasp_point_is_below_the_observed_top_by_the_harness_grasp_depth(rule, expert_ticks):
    """파지점 = 관측된 대상 자세의 xy, 윗면 − `grasp_depth_mm`(하네스 설정); 대상이 관측에 없거나 후보에 대상이 없으면 None."""
    tick = expert_ticks[0]
    state = tick["request"]["state"]
    entry = state["objects"][0]
    point = rule.grasp_point(state, {"target": str(entry["id"])})
    assert point == [float(entry["pose_mm"][0]), float(entry["pose_mm"][1]), float(entry["top_mm"]) - rule.grasp_depth_mm]
    assert rule.grasp_point(state, {"target": "no-such-object"}) is None and rule.grasp_point(state, {"target": None}) is None
    harness = load_harness_config(load_expert_config()["harness_config"])
    assert rule.grasp_depth_mm == float(harness["candidates"]["grasp_depth_mm"]) and rule.grasp_ready_mm == float(load_expert_config()["thresholds"]["grasp_ready_mm"])


# ==========================================================================
# Task R13 — 갈래 X: 그리퍼에 더해 경로·힘도 실행 쪽 규칙 (expert `_path`·국면별 힘을 요청·commitment·관측 상태만으로)
# ==========================================================================

#: R12 G 갈래의 저장 기록 한 편 — r12s19G dev_new2(E1, 지시 변경 1회, 51틱, 엄격 성공; 실행 규칙이 모델의 그리퍼를 2틱 바꿨다)
R12_G_RECORD = (REPO / "artifacts/datasets/r12-closed-loop/r12s19G/dev_new2", "ep-E1-990297-r12-r12s19G")
#: 금지 물체의 추가 여유 띠에 말단이 들어간 틱이 있는 R6 전문가 기록 (전문가 retreat · 실행 경로 규칙 hold — 모듈 설명)
R6_EXPERT_BAND_RECORD = (REPO / "artifacts/datasets/r6-closed-loop/expert/dev_new2", "ep-E1-990132-r6-expert")
PATH_REASONS = {"no_commitment", "fixed_candidate", "clear", "blocked_via", "hand_inside_obstacle", "blocked_no_detour"}


@pytest.fixture(scope="module")
def path_rule():
    return ExecutionPathRule()


@pytest.fixture(scope="module")
def force_rule():
    return ExecutionForceRule()


def _expert_path(expert, request, commitment=None):
    """전문가가 같은 입력에서 고르는 경로 — `Expert._path`를 그 자신의 `_committed`로 부른 값(commitment가 없으면 None)."""
    model = request.get("request", request)
    values = {entry["id"]: candidate_values(entry) for entry in model["candidates"]["q_main"]}
    committed = expert._committed(model, commitment, values)
    if committed is None:
        return None
    return expert._path(list(model["candidates"].get("q_path") or []), values.get(str(committed["action_ref"])), model["state"])


def _plain_inside(state, value, margin):
    """말단이 대상·쥔 물체가 아닌 관측 물체의 외접 구 + margin 안인가 (시험의 전제 확인용)."""
    ee = [float(item) for item in state["robot"]["ee_pose_mm"]]
    skip = {str((value or {}).get("target")), str(state["robot"].get("holding"))}
    return any(math.dist(ee, [float(v) for v in entry["pose_mm"]]) < circumradius_mm(entry["obb_mm"]) + margin
               for entry in state["objects"] if str(entry["id"]) not in skip)


def test_spread_is_the_experts_spread_for_any_choice_and_option_list(expert):
    """`spread` = `Expert._spread`: 고른 것이 후보에 있을 때·없을 때(고른 분포)·후보가 하나일 때·없을 때(빈 분포) — 값과 키 순서까지."""
    mass = float(expert.confidence["choice_mass"])
    cases = [("ph", ["p0", "pr", "ph"]), ("p1", ["p0", "p1", "p2", "pr", "ph"]), (None, ["p0", "pr", "ph"]), ("zz", ["p0", "ph"]), ("p0", ["p0"]),
             (None, []), ("1", ["0", "1", "2"]), ("0", ["0", "1", "2"])]
    for chosen, options in cases:
        assert list(spread(chosen, options, mass).items()) == list(expert._spread(chosen, options).items()), (chosen, options)


def test_the_path_rule_equals_the_expert_path_on_every_branch_of_hand_made_requests(path_rule, expert, expert_ticks):
    """합성 요청의 갈래마다(commitment 없음 · hold 경로가 없는 commitment 없음 · 고정 후보 commitment · 비었음 · 막혔고 경유점 있음 · 막혔고 경유점 없음에
    손이 평범한 물체의 구 안 · 밖 · 구 안인데 retreat 경로 없음 · hold·retreat 둘 다 없음 · 경로 후보 없음) 실행 경로 규칙의 (고른 경로, 종류, 이유)가
    `Expert._path`(commitment가 없으면 `act`의 hold 갈래)와 같고, 답의 분포가 `Expert.act`의 `q_path`와 키 순서까지 같다."""
    base = next(tick for tick in expert_ticks if tick["request"]["state"]["robot"].get("holding") is None
                and any(str(entry.get("key", "")).startswith("grasp:") for entry in tick["request"]["candidates"]["q_main"]))
    _, joint = _variant(base, phase="approach")
    target = joint["key"].split(":")[1]
    other = next(entry for entry in base["request"]["state"]["objects"] if str(entry["id"]) != target)
    with_via = [{"id": "p0", "kind": "direct"}, {"id": "p1", "kind": "via", "ref": "w1"}, {"id": "p2", "kind": "via", "ref": "w2"},
                {"id": "pr", "kind": "retreat"}, {"id": "ph", "kind": "hold"}]
    without_via = [{"id": "p0", "kind": "direct"}, {"id": "pr", "kind": "retreat"}, {"id": "ph", "kind": "hold"}]
    no_retreat = [{"id": "p0", "kind": "direct"}, {"id": "ph", "kind": "hold"}]
    only_direct = [{"id": "p0", "kind": "direct"}]
    far = [0.0, 0.0, 900.0]  # 모든 물체의 구 밖(테이블 위 높이)
    inside = [float(value) for value in other["pose_mm"]]  # 대상이 아닌 물체의 중심 — 평범한 장애물 구 안
    cases = {
        "no_commitment": dict(commit=False, q_path=without_via),
        "no_commitment_without_hold": dict(commit=False, q_path=only_direct),
        "clear": dict(phase="approach", path_ok=True, q_path=without_via),
        "blocked_with_via": dict(phase="approach", path_ok=False, q_path=with_via),
        "blocked_hand_inside_obstacle": dict(phase="approach", path_ok=False, q_path=without_via, ee=inside),
        "blocked_hand_free": dict(phase="approach", path_ok=False, q_path=without_via, ee=far),
        "blocked_inside_without_retreat": dict(phase="approach", path_ok=False, q_path=no_retreat, ee=inside),
        "blocked_only_direct": dict(phase="approach", path_ok=False, q_path=only_direct, ee=far),
        "no_paths": dict(phase="approach", path_ok=True, q_path=[]),
    }
    reasons = set()
    for name, change in cases.items():
        tick, _ = _variant(base, **change)
        mine = path_rule.decide(tick, None)
        expected = _expert_path(expert, tick)
        answers = expert.act(tick, None)
        if expected is None:
            hold = next((entry["id"] for entry in tick["request"]["candidates"]["q_path"] if entry.get("kind") == "hold"), None)
            assert (mine["choice"], mine["reason"]) == (hold, "no_commitment") and answers["expert_meta"]["aux"] is None, name
        else:
            assert (mine["choice"], mine["kind"], mine["reason"]) == (expected["choice"], expected["kind"], expected["reason"]), name
        assert list(path_rule.answer(tick, None).items()) == list(answers["q_path"].items()), name
        reasons.add(mine["reason"])
    # 고정 후보(hold)에 걸린 commitment — 경로는 hold
    fixed, _ = _variant(base, phase="none", q_path=without_via)
    hold_candidate = next(entry for entry in fixed["request"]["candidates"]["q_main"] if entry.get("key") == "hold")
    fixed["request"]["commitment"] = {"action_ref": hold_candidate["id"], "key": "hold", "phase": "none", "held_ticks": 1, "last_switch_tick": 0}
    expected = _expert_path(expert, fixed)
    mine = path_rule.decide(fixed, None)
    assert (mine["choice"], mine["kind"], mine["reason"]) == (expected["choice"], "hold", "fixed_candidate")
    assert list(path_rule.answer(fixed, None).items()) == list(expert.act(fixed, None)["q_path"].items())
    reasons.add(mine["reason"])
    assert PATH_REASONS | {"no_paths"} <= reasons
    assert path_rule.decide(_variant(base, phase="approach", path_ok=False, q_path=with_via)[0])["choice"] == "p1"  # 첫 경유점
    assert path_rule.answer(_variant(base, phase="approach", path_ok=True, q_path=[])[0]) == {}  # 경로 후보가 없으면 답이 없다


def test_the_path_rule_reads_no_forbidden_information_so_it_holds_where_the_expert_retreats_inside_a_forbidden_objects_extra_margin(path_rule, expert, expert_ticks):
    """전문가와 다른 유일한 자리(모듈 설명): 경로가 막혔고 경유점이 없고 말단이 **금지 물체**의 추가 여유 띠(외접 구 + margin 밖, + forbidden_margin 안)에만
    있으면 전문가는 retreat(`hand_inside_obstacle`), 실행 경로 규칙은 hold(`blocked_no_detour`) — 규칙은 목표의 `forbidden_contact`·물체의 `forbidden` 속성을
    읽지 않는다. 그 표시를 지우면 전문가도 hold다(규칙은 그대로). `aux_rule_agreement`는 그 틱을 원인 `forbidden_margin`으로 적는다."""
    harness = load_harness_config(load_expert_config()["harness_config"])
    margin, extra = float(harness["planner"]["margin_mm"]), float(harness["planner"]["forbidden_margin_mm"])
    base = next(tick for tick in expert_ticks if tick["request"]["state"]["robot"].get("holding") is None
                and any(str(entry.get("key", "")).startswith("grasp:") for entry in tick["request"]["candidates"]["q_main"]))
    tick, joint = _variant(base, phase="approach", path_ok=False, q_path=[{"id": "p0", "kind": "direct"}, {"id": "pr", "kind": "retreat"}, {"id": "ph", "kind": "hold"}])
    target = joint["key"].split(":")[1]
    state = tick["request"]["state"]
    forbidden = next(entry for entry in state["objects"] if str(entry["id"]) != target)
    pose = [float(value) for value in forbidden["pose_mm"]]
    state["robot"]["ee_pose_mm"] = [pose[0], pose[1], pose[2] + circumradius_mm(forbidden["obb_mm"]) + margin + extra / 2.0]
    forbidden["attributes"] = ["forbidden"]
    state["goal"]["forbidden_contact"] = [str(forbidden["id"])]
    assert not _plain_inside(state, candidate_values(joint), margin)  # 전제: 어느 물체의 평범한 구 안도 아니다
    expert_answers = expert.act(tick, None)
    assert expert_answers["expert_meta"]["aux"]["path"]["reason"] == "hand_inside_obstacle" and path_rule.decide(tick)["reason"] == "blocked_no_detour"
    assert path_rule.answer(tick)["ph"] == 0.9 and expert_answers["q_path"]["pr"] == 0.9
    record = {"episode_id": "band", "ticks": [{**tick, "model_output": {key: expert_answers[key] for key in QUESTIONS}}],
              "evidence": {"expert": {"ticks": [{"t": tick["t"], **expert_answers["expert_meta"]}]}}, "provenance": {"policy": {"name": "TimedExpert"}}}
    out = aux_rule_agreement([record], path_rule, ExecutionForceRule())
    assert out["q_path"]["disagree"] == 1 and out["q_path"]["by_cause"] == {"forbidden_margin": 1, "other": 0} and out["q_force"]["disagree"] == 0
    assert out["expert_answers"] == {"ticks": 1, "q_path_equal": 0, "q_force_equal": 1, "mismatches": out["expert_answers"]["mismatches"]}
    # 금지 표시를 지우면 전문가도 hold — 규칙의 답은 그대로다(규칙은 그 표시를 읽은 적이 없다)
    plain = copy.deepcopy(tick)
    plain["request"]["state"]["goal"]["forbidden_contact"] = []
    next(e for e in plain["request"]["state"]["objects"] if e["id"] == forbidden["id"])["attributes"] = []
    assert _expert_path(expert, plain)["reason"] == "blocked_no_detour" and path_rule.decide(plain) == path_rule.decide(tick)


def test_the_force_rule_equals_the_expert_force_on_every_phase_and_without_a_commitment(force_rule, expert, expert_ticks):
    """국면마다(모르는 국면은 none) 실행 힘 규칙의 수준 = 전문가 `_aux`의 힘 수준, commitment가 없으면 `0`; 답의 분포 = `Expert.act`의 `q_force`(키 순서까지)."""
    base = next(tick for tick in expert_ticks if any(str(entry.get("key", "")).startswith("grasp:") for entry in tick["request"]["candidates"]["q_main"]))
    seen = set()
    for phase in (*PHASES, "hover", None):
        tick, _ = _variant(base, phase=phase, commit=phase is not None)
        answers = expert.act(tick, None)
        aux = answers["expert_meta"]["aux"]
        mine = force_rule.decide(tick, None)
        assert mine["level"] == (aux["force"]["level"] if aux is not None else "0"), phase
        assert list(force_rule.answer(tick, None).items()) == list(answers["q_force"].items()), phase
        seen.add(mine["level"])
    assert seen == {"0", "1", "2"} and force_rule.levels == ["0", "1", "2"]
    assert force_rule.decide(_variant(base, phase="push")[0])["level"] == "2" and force_rule.decide(_variant(base, phase="hover")[0])["reason"] == "phase:none"


def test_the_path_and_force_rules_equal_a_live_expert_on_real_requests_except_inside_a_forbidden_extra_margin(path_rule, force_rule, expert, expert_ticks):
    """R6 전문가 기록의 실제 요청마다(요청의 commitment 투영만 줄 때와 같은 후보를 가리키는 하네스 commitment를 함께 줄 때): 힘 규칙의 답 = 살아 있는
    전문가 `act`의 `q_force`, 경로 규칙의 답 = `q_path`(분포·키 순서까지) — 단 말단이 금지 물체의 추가 여유 띠에만 있는 틱(이 여덟 편 가운데
    ep-E1-990158에 있다)은 전문가 retreat · 규칙 hold이고, 그 밖의 불일치는 없다."""
    harness = load_harness_config(load_expert_config()["harness_config"])
    band = 0
    for index, tick in enumerate(expert_ticks):
        projected = tick["request"].get("commitment")
        harness_commitment = None if projected is None else {**projected, "goal_version": 1, "stop_ticks": 0}
        for commitment in (None, harness_commitment):
            answers = expert.act(tick, commitment)
            assert list(force_rule.answer(tick, commitment).items()) == list(answers["q_force"].items()), index
            mine = path_rule.decide(tick, commitment)
            if list(path_rule.answer(tick, commitment).items()) == list(answers["q_path"].items()):
                continue
            aux = answers["expert_meta"]["aux"]
            values = {entry["id"]: candidate_values(entry) for entry in tick["request"]["candidates"]["q_main"]}
            value = values.get(str(mine["action_ref"]))
            assert aux["path"]["reason"] == "hand_inside_obstacle" and mine["reason"] == "blocked_no_detour", index
            assert _forbidden_band(tick["request"]["state"], value, margin_mm=float(harness["planner"]["margin_mm"]),
                                   forbidden_margin_mm=float(harness["planner"]["forbidden_margin_mm"])), index
            band += 1
    assert band > 0 and band % 2 == 0  # 두 commitment 꼴에서 같은 틱이 둘씩


def test_the_path_and_force_rules_agree_with_the_experts_recorded_decisions_on_stored_closed_loop_ticks(path_rule, force_rule):
    """저장된 폐루프 기록(R6 전문가·규칙 판정기, R9 seed 17·seed 18 — 서로 다른 정책이 몬 상태)의 틱마다 힘 규칙 = 그 틱의 전문가 참조 판단, 경로 규칙
    = 전문가 판단이되 불일치는 모두 금지 물체의 추가 여유 띠(원인 `forbidden_margin`, 그 밖 0); 틱들이 경로의 모든 갈래(이유)와 힘 수준 셋을 지나고,
    전문가 기록에서는 답의 분포가 `model_output`과 띠의 틱 말고 같다."""
    reasons: dict[str, int] = {}
    levels: set[str] = set()
    total = band = 0
    for directory in AGREEMENT_DIRS:
        records = read_open_episodes(REPO / directory)[0]
        if not records:
            pytest.fail(f"저장된 폐루프 기록이 없다: {directory}")
        out = aux_rule_agreement(records, path_rule, force_rule)
        assert out["q_force"]["disagree"] == 0 and out["q_force"]["agree"] == out["ticks"] > 0, directory
        assert out["q_path"]["by_cause"]["other"] == 0 and out["q_path"]["disagree"] == out["q_path"]["by_cause"]["forbidden_margin"], directory
        if directory.endswith("expert/dev_new2"):
            answers = out["expert_answers"]
            assert answers["ticks"] == out["ticks"] and answers["q_force_equal"] == out["ticks"]
            assert answers["q_path_equal"] == out["ticks"] - out["q_path"]["disagree"]
        else:
            assert out["expert_answers"]["ticks"] == 0  # 전문가가 몬 기록이 아니면 model_output은 전문가의 답이 아니다
        total += out["ticks"]
        band += out["q_path"]["disagree"]
        levels |= set(out["q_force"]["by_expert_level"])
        for reason, count in out["q_path"]["by_expert_reason"].items():
            reasons[reason] = reasons.get(reason, 0) + count
    assert PATH_REASONS <= set(reasons) and levels == {"0", "1", "2"} and total > 30_000 and 0 < band < total / 100


def test_aux_rule_agreement_names_a_tick_where_the_rules_and_the_recorded_expert_differ(path_rule, force_rule):
    """대조가 실제로 대조하는가 — 저장된 전문가의 경로 판단(direct)을 hold로, 힘 수준을 바꾸면 그 틱이 불일치로 적히고(경로 원인 `other`), 금지 띠의
    틱은 `forbidden_margin`으로 적힌다; 전문가 기록의 틱 수가 다르면 거절한다."""
    record = copy.deepcopy(_stored(R6_EXPERT_RECORD)[1])
    clean = aux_rule_agreement([record], path_rule, force_rule)
    assert clean["q_path"]["disagree"] == 0 and clean["q_force"]["disagree"] == 0 and clean["expert_answers"]["q_path_equal"] == clean["ticks"]
    meta = record["evidence"]["expert"]["ticks"]
    index = next(i for i, entry in enumerate(meta) if (entry.get("aux") or {}).get("path", {}).get("reason") == "clear")
    meta[index]["aux"]["path"].update({"choice": "ph", "kind": "hold", "reason": "blocked_no_detour"})
    wrong = next(level for level in ("0", "1", "2") if level != force_rule.decide(record["ticks"][index])["level"])
    meta[index]["aux"]["force"]["level"] = wrong
    out = aux_rule_agreement([record], path_rule, force_rule)
    assert out["q_path"]["disagree"] == 1 and out["q_path"]["mismatches"][0]["tick"] == index and out["q_path"]["by_cause"] == {"forbidden_margin": 0, "other": 1}
    assert out["q_force"]["disagree"] == 1 and out["q_force"]["mismatches"][0]["expert"] == wrong
    band = aux_rule_agreement([_stored(R6_EXPERT_BAND_RECORD)[1]], path_rule, force_rule)
    assert band["q_path"]["disagree"] > 0 and band["q_path"]["by_cause"]["other"] == 0
    assert {item["expert"]["reason"] for item in band["q_path"]["mismatches"]} == {"hand_inside_obstacle"}
    record["evidence"]["expert"]["ticks"].pop()
    with pytest.raises(ValueError, match="evidence.expert"):
        aux_rule_agreement([record], path_rule, force_rule)


def test_the_path_and_force_rules_read_neither_the_goal_nor_object_attributes_nor_the_observation(path_rule, force_rule, expert_ticks):
    """목표(문장·대상·영역·금지·취약)와 물체 속성(금지·취약)을 바꿔도 두 규칙의 판단이 같고, commitment·말단·경로 막힘을 바꾸면 판단이 바뀐다.
    두 규칙은 전문가 객체를 들지 않는다."""
    tick = next(t for t in expert_ticks if (t["request"].get("commitment") or {}).get("phase") == "approach")
    before = (path_rule.decide(tick, None), force_rule.decide(tick, None))
    changed = copy.deepcopy(tick)
    state = changed["request"]["state"]
    state["goal"].update({"text": "anything", "target_ref": "nothing", "target_zone": "zoneZ", "forbidden_contact": [e["id"] for e in state["objects"]],
                          "fragile": [e["id"] for e in state["objects"]]})
    for entry in state["objects"]:
        entry["attributes"] = ["forbidden", "fragile"]
    assert (path_rule.decide(changed, None), force_rule.decide(changed, None)) == before
    uncommitted = copy.deepcopy(tick)
    uncommitted["request"]["commitment"] = None
    assert path_rule.decide(uncommitted)["reason"] == "no_commitment" and force_rule.decide(uncommitted)["reason"] == "no_commitment"
    flipped = copy.deepcopy(tick)
    entry = next(e for e in flipped["request"]["candidates"]["q_main"] if e["id"] == flipped["request"]["commitment"]["action_ref"])
    entry["path"] = "blocked" if entry.get("path", "ok") == "ok" else "ok"
    assert path_rule.decide(flipped)["reason"] != before[0]["reason"]
    pushed = copy.deepcopy(tick)
    pushed["request"]["commitment"]["phase"] = "push"
    assert force_rule.decide(pushed)["level"] == "2" != before[1]["level"]
    for rule in (path_rule, force_rule):
        assert not any(isinstance(value, Expert) for value in vars(rule).values())


class FixedX(Fixed):
    """갈래 X 시험용 — 경로·힘·속도에도 답하는 가짜 정책."""

    def act(self, request, commitment=None, observation=None):
        answers = super().act(request, commitment, observation)
        answers.update({"q_path": {"ph": 0.7, "p0": 0.3}, "q_force": {"2": 0.8, "0": 0.2}, "q_speed": {"3": 0.9, "0": 0.1}})
        return answers


def test_arm_x_replaces_q_gripper_q_path_and_q_force_and_leaves_speed_stop_and_every_semantic_answer(rule, path_rule, force_rule, expert_ticks):
    """X: 하네스가 받는 답 가운데 `q_gripper`·`q_path`·`q_force`만 실행 규칙의 답이고 `q_speed`·`q_stop`·`q_main`·게이트는 감싼 정책의 것 그대로; raw 기록은
    감싼 정책의 세 답과 세 규칙의 판단; 버전은 dx0.2(G·GS는 dx0.1 그대로)."""
    tick = next(t for t in expert_ticks if (t["request"].get("commitment") or {}).get("phase") == "approach")
    assert ARMS == ("G", "GS", "X")
    policy = DelegatedExecutionPolicy(FixedX(), arm="X", rule=rule, path_rule=path_rule, force_rule=force_rule)
    assert policy.version == "dx0.2/X/fixed-0" and policy.delegation_version == ARM_DELEGATION_VERSIONS["X"] == "dx0.2"
    out = policy.act(tick, None, {"sim": "ignored"})
    raw = FixedX().act(tick)
    assert {qid for qid in QUESTION_SET_V0 if out[qid] != raw[qid]} == {"q_gripper", "q_path", "q_force"}
    assert out["q_gripper"] == rule.answer(tick) and out["q_path"] == path_rule.answer(tick) and out["q_force"] == force_rule.answer(tick)
    assert out["q_speed"] == raw["q_speed"] and out["q_stop"] == 0.9 and out["q_main"] == raw["q_main"]
    assert policy.raw == [{"t": int(tick["t"]), "inner": {"q_gripper": raw["q_gripper"], "q_path": raw["q_path"], "q_force": raw["q_force"]},
                           "rule": rule.decide(tick), "path_rule": path_rule.decide(tick), "force_rule": force_rule.decide(tick)}]
    for arm in ("G", "GS"):
        assert DelegatedExecutionPolicy(Fixed(), arm=arm, rule=rule).version == f"dx0.1/{arm}/fixed-0" and ARM_DELEGATION_VERSIONS[arm] == DELEGATION_VERSION
    with pytest.raises(ValueError, match="X"):
        DelegatedExecutionPolicy(Fixed(), arm="G", rule=rule, path_rule=path_rule)
    assert DelegatedExecutionPolicy(Fixed(), arm="G").path_rule is None and isinstance(DelegatedExecutionPolicy(Fixed(), arm="X").path_rule, ExecutionPathRule)


def test_the_wrapped_policy_receives_the_generators_arguments_in_arm_x_and_the_wrapper_never_calls_the_expert(expert_ticks, monkeypatch):
    """정보 경계(X): 감싼 정책은 감싸지 않은 경로와 같은 (요청, commitment, 관측) 객체를 먼저 받고, 감싸개는 전문가를 부르지 않는다(전문가 `act`·`_path`·
    `_aux`·`_gripper`·`_hand_inside_obstacle`을 막아도 돈다)."""

    def forbidden(*args, **kwargs):
        raise AssertionError("감싸개가 전문가를 불렀다")

    for name in ("act", "_path", "_aux", "_gripper", "_hand_inside_obstacle"):
        monkeypatch.setattr(Expert, name, forbidden)
    inner = FixedX()
    policy = DelegatedExecutionPolicy(inner, arm="X")
    tick, commitment, observation = expert_ticks[30], {"action_ref": "x"}, {"sim": True}
    policy.act(tick, commitment, observation)
    received = inner.calls[0]
    assert received[0] is tick and received[1] is commitment and received[2] is observation


def test_build_policy_delegates_arm_x_with_the_three_rules_and_keeps_arm_gs_description():
    """`build_policy(..., delegate="X")`: 서술에 세 규칙(그리퍼 xg0.1·경로 xp0.1·힘 xf0.1)과 dx0.2; G의 서술은 R10 그대로(경로·힘 규칙 없음, dx0.1)."""
    rule_x = build_policy("rule", generator=GENERATOR, delegate="X")
    describe = rule_x["describe"]
    assert (describe["name"], describe["version"], describe["arm"], describe["delegation_version"]) == ("DelegatedExecutionPolicy", "dx0.2/X/rj0.5", "X", "dx0.2")
    assert describe["rule"]["version"] == EXECUTION_GRIPPER_VERSION and describe["path_rule"]["version"] == EXECUTION_PATH_VERSION == "xp0.1"
    assert describe["force_rule"]["version"] == EXECUTION_FORCE_VERSION == "xf0.1" and describe["path_rule"]["pinned_to"] == "Expert._path (e0.4)"
    assert isinstance(rule_x["policy"].path_rule, ExecutionPathRule) and isinstance(rule_x["policy"].force_rule, ExecutionForceRule)
    rule_g = build_policy("rule", generator=GENERATOR, delegate="G")["describe"]
    assert "path_rule" not in rule_g and "force_rule" not in rule_g and rule_g["delegation_version"] == "dx0.1" and rule_g["version"] == "dx0.1/G/rj0.5"
    expert_x = build_policy("expert", generator=GENERATOR, delegate="X")
    assert expert_x["policy"].inner is not expert_x["expert"] and isinstance(expert_x["policy"].inner, Expert)


class ReplayInner(Replay):
    """저장된 위임 기록의 **감싼 정책의 raw 답**을 돌려주는 정책 — `model_output`에서 감싸개가 바꾼 답만 `usage.delegation.inner`의 것으로."""

    def __init__(self, record, name, version) -> None:
        super().__init__(record)
        self.name, self.version = name, version

    def act(self, request, commitment=None, observation=None):
        tick = self.record["ticks"][self.index]
        answers = super().act(request, commitment, observation)
        answers.update(copy.deepcopy(tick["usage"]["delegation"]["inner"]))
        return answers


def test_the_g_path_reproduces_a_stored_r12_g_record_byte_for_byte(tmp_path):
    """갈래 G의 경로(폐루프 실행기 → 생성기 → 감싸개 G → 하네스·컨트롤러·시뮬레이터 → 계약 검사 → 기록)가 R12 r12s19G의 dev_new2 한 편을 그 기록의 감싼
    모델 raw 답으로 다시 돌리면 **바이트 단위로** 같은 파일을 쓴다 — 다른 것은 벽시계 하나다(갈래 X를 더해도 G의 기록은 R10·R12 그대로)."""
    raw, stored = _stored(R12_G_RECORD)
    inner = stored["provenance"]["policy"]["inner"]
    assert stored["provenance"]["policy"]["version"] == "dx0.1/G/mp0.1" and stored["evidence"]["delegation"]["q_gripper_changed_ticks"] > 0
    config = load_closed_loop_config(R6_CONFIG)
    expert_config = load_expert_config(config_paths(config["generator"])["expert_config"])
    policy = DelegatedExecutionPolicy(ReplayInner(stored, inner["name"], inner["version"]), arm="G", rule=ExecutionGripperRule(expert_config))
    bundle = {**_bundle(policy), "arm": "G"}
    profile, seed = stored["provenance"]["profile"], int(stored["provenance"]["seed"])
    run_condition(bundle, [(profile, seed)], config=config, out=tmp_path / "dev_new2", condition="dev_new2", label="r12s19G", id_tag="r12")
    assert _same_wall(_written(tmp_path, "dev_new2", stored["episode_id"]), stored["provenance"]["timing"]["wall_s"]) == raw


def test_the_expert_wrapped_in_arm_x_reproduces_the_stored_unwrapped_expert_record(tmp_path):
    """전문가를 감싼 X는 감싸지 않은 전문가(R6 dev_new2 한 편 — 금지 띠의 틱이 없는 편)와 **같은 편**을 만든다: 세 실행 규칙의 답이 그 편의 모든 틱에서
    전문가의 `q_gripper`·`q_path`·`q_force`와 같으므로 명령이 같고, 감싸개가 붙인 것과 에피소드 id의 이름표·벽시계 말고는 바이트 단위로 같다."""
    raw, stored = _stored(R6_EXPERT_RECORD)
    config = load_closed_loop_config(R6_CONFIG)
    schedule = [(stored["provenance"]["profile"], int(stored["provenance"]["seed"]))]
    wrapped = build_policy("expert", generator=config["generator"], delegate="X")
    run_condition(wrapped, schedule, config=config, out=tmp_path / "x" / "dev_new2", condition="dev_new2", label="expertX", id_tag="r6")
    record = json.loads(_written(tmp_path / "x", "dev_new2", stored["episode_id"].replace("-r6-expert", "-r6-expertX")))
    summary = record["evidence"].pop("delegation")
    assert summary["arm"] == "X" and summary["version"] == "dx0.2" and summary["ticks"] == len(stored["ticks"])
    assert (summary["q_gripper_changed_ticks"], summary["q_path_changed_ticks"], summary["q_force_changed_ticks"]) == (0, 0, 0)
    assert record["provenance"]["policy"] == {"name": "DelegatedExecutionPolicy", "version": "dx0.2/X/e0.4", "arm": "X", "delegation_version": "dx0.2",
                                              "rule": {"name": "ExecutionGripperRule", "version": EXECUTION_GRIPPER_VERSION},
                                              "inner": {"name": "Expert", "version": "e0.4"},
                                              "path_rule": {"name": "ExecutionPathRule", "version": EXECUTION_PATH_VERSION},
                                              "force_rule": {"name": "ExecutionForceRule", "version": EXECUTION_FORCE_VERSION}}
    for tick in record["ticks"]:
        delegation = tick["usage"].pop("delegation")
        for qid in ("q_gripper", "q_path", "q_force"):
            assert delegation["inner"][qid] == tick["model_output"][qid]
    record["provenance"]["policy"] = stored["provenance"]["policy"]
    record["episode_id"] = stored["episode_id"]
    record["provenance"]["timing"]["wall_s"] = stored["provenance"]["timing"]["wall_s"]
    assert (json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8") == raw


def test_a_wrapped_rule_judge_record_in_arm_x_keeps_its_raw_answers_and_the_harness_received_the_three_rules(tmp_path):
    """규칙 판정기를 X로 감싼 짧은 run: 기록이 계약을 지나고, 틱마다 `model_output`의 `q_gripper`·`q_path`·`q_force` = 그 틱 요청에서 다시 계산한 세 실행
    규칙의 답, `usage.delegation.inner` = 규칙 판정기가 그 요청에 내는 raw 세 답, 나머지 일곱 답(`q_speed`·`q_stop` 포함)은 규칙 판정기의 것 그대로;
    manifest·provenance·evidence에 갈래와 세 규칙이 적힌다."""
    from robo_jev.data.dagger import rule_judge_policy

    config = load_closed_loop_config(R6_CONFIG)
    gripper, path, force, judge = ExecutionGripperRule(), ExecutionPathRule(), ExecutionForceRule(), rule_judge_policy()
    bundle = build_policy("rule", generator=config["generator"], delegate="X")
    run = run_condition(bundle, [("E0", 980108)], config=config, out=tmp_path / "X" / "dev_new2", condition="dev_new2", label="ruleX", id_tag="r13", max_ticks=40)
    manifest = json.loads((tmp_path / "X" / "dev_new2" / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["closed_loop"]["arm"] == "X" and manifest["closed_loop"]["policy"]["path_rule"]["version"] == EXECUTION_PATH_VERSION
    record = json.loads(_written(tmp_path / "X", "dev_new2", run["episodes"][0]["episode_id"]))
    validate_record(record)
    assert record["provenance"]["policy"]["arm"] == "X" and record["provenance"]["policy"]["path_rule"]["name"] == "ExecutionPathRule"
    assert record["evidence"]["delegation"]["ticks"] == len(record["ticks"]) == 40
    changed_path = 0
    for tick in record["ticks"]:
        delegation = tick["usage"]["delegation"]
        raw = judge.act(tick)
        assert tick["model_output"]["q_gripper"] == gripper.answer(tick) and tick["model_output"]["q_path"] == path.answer(tick)
        assert tick["model_output"]["q_force"] == force.answer(tick)
        assert delegation["rule"] == gripper.decide(tick) and delegation["path_rule"] == path.decide(tick) and delegation["force_rule"] == force.decide(tick)
        assert delegation["inner"] == {"q_gripper": raw["q_gripper"], "q_path": raw["q_path"], "q_force": raw["q_force"]}
        for qid in ("q_main", "q_done", "q_instr", "q_observe", "q_retry", "q_stop", "q_speed"):
            assert tick["model_output"][qid] == raw[qid]
        inner_choice = min(raw["q_path"], key=lambda key: (-raw["q_path"][key], key))  # 하네스 `_argmax`의 동점 규칙
        changed_path += int(inner_choice != delegation["path_rule"]["choice"])
    assert record["evidence"]["delegation"]["q_path_changed_ticks"] == changed_path


def test_the_run_command_takes_arm_x_and_puts_it_in_the_label_and_the_run_file(tmp_path):
    """`closed_loop.py run --delegate X`: 이름표 = 정책 이름표 + X(ruleX), 에피소드 id 꼬리 `-<tag>-ruleX`, run 파일의 `arm` X."""
    module = _script()
    report = tmp_path / "run.json"
    args = ["run", "--policy", "rule", "--delegate", "X", "--id-tag", "r13t", "--condition", "dev_new2", "--config", str(R6_CONFIG),
            "--seeds", str(REPO / "artifacts/reports/r6-seeds.json"), "--limit", "1", "--max-ticks", "6", "--out", str(tmp_path / "rule"), "--report", str(report)]
    assert module.main(args) == 0
    payload = json.loads(report.read_text(encoding="utf-8"))
    assert payload["label"] == "ruleX" and payload["arm"] == "X" and payload["policy"]["arm"] == "X" and payload["policy"]["delegation_version"] == "dx0.2"
    assert all(row["episode_id"].endswith("-r13t-ruleX") for row in payload["conditions"]["dev_new2"]["episodes"])
