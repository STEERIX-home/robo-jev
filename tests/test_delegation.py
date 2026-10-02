"""실행 쪽 위임 검사 (Task R10 A1·A2) — 그리퍼 닫기 시점(과 정지)을 실행 층의 기하 규칙에 맡기는 정책 감싸개.

고정하는 것: (1) 실행 규칙 = 전문가 `_gripper` (합성 요청의 갈래마다, 실제 틱마다 살아 있는 전문가와, 저장된 폐루프 기록의 전문가 판단과);
(2) 규칙은 commitment와 관측 상태만 읽는다; (3) 감싸개는 `q_gripper`(GS면 `q_stop`도)만 바꾸고 감싼 정책의 raw 답을 기록에 남긴다;
(4) 감싸지 않은 경로는 바이트 단위로 그대로다(R9 기록 한 편을 재현); (5) 전문가를 감싼 G는 감싸지 않은 전문가의 기록을 재현한다; (6) CLI의 갈래.
"""

import copy
import json

import pytest
from helpers import REPO

from robo_jev.closed_loop import TimedExpert, build_policy, load_closed_loop_config, run_condition
from robo_jev.contracts import PHASES, QUESTION_SET_V0, validate_record
from robo_jev.data.robot_episodes import QUESTIONS, config_paths, load_generator_config
from robo_jev.data.sealed import read_episode_ids, read_open_episodes
from robo_jev.delegation import (
    ARMS,
    DELEGATION_VERSION,
    EXECUTION_GRIPPER_VERSION,
    DelegatedExecutionPolicy,
    ExecutionGripperRule,
    delegate_bundle,
    descent_path_kind,
    rule_agreement,
)
from robo_jev.harness.robot import load_harness_config
from robo_jev.harness.rule_judge import candidate_values
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
    for arm in ARMS:
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
    with pytest.raises(ValueError, match="arm"):
        DelegatedExecutionPolicy(Fixed(), arm="X")


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
    for arm in ARMS:
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
