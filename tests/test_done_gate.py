"""done 게이트 수집 정책 검사 (Task R6 A3) — 모델의 답을 그대로 쓰되 하네스에 넘기는 `q_done`만 expert의 것으로 바꾸는 **수집 전용** 감싸개."""

import json

import pytest
import yaml
from helpers import REPO

from robo_jev.closed_loop import build_policy, load_closed_loop_config, run_condition, select_seeds
from robo_jev.contracts import QUESTION_SET_V0, validate_record
from robo_jev.data.done_gate import (
    COLLECTION_NAME,
    DONE_GATE_VERSION,
    DoneGatePolicy,
    attach_raw_done,
    collect_done_gate,
    collection_seed_base,
)
from robo_jev.data.done_strata import false_done_events, model_q_done, reference_done
from robo_jev.data.robot_episodes import config_paths, load_generator_config, read_episodes
from robo_jev.sim.controller import resolve_config_path
from robo_jev.sim.expert import Expert

CONFIG = load_generator_config("configs/data/r1_robot.yaml")


class AlwaysDone:
    """전문가의 답에 `q_done` 0.9만 박은 가짜 모델 — 거짓 done을 틱마다 내는 정책."""

    name = "AlwaysDone"
    version = "always-done-0"

    def __init__(self) -> None:
        self.expert = Expert()
        self.timing: list[dict] = []
        self.resets = 0

    def reset(self) -> None:
        self.resets += 1

    def act(self, request, commitment=None, observation=None):
        answers = self.expert.act(request, commitment, observation)
        return {**{qid: answers[qid] for qid in QUESTION_SET_V0}, "q_done": 0.9}


def test_the_wrapper_passes_the_experts_done_to_the_harness_and_keeps_the_models_raw_answer():
    inner = AlwaysDone()
    policy = DoneGatePolicy(inner, Expert())
    assert policy.collection_only is True and policy.name == "DoneGatePolicy"
    assert policy.version == f"{DONE_GATE_VERSION}/{inner.version}"
    from robo_jev.harness.robot import RobotHarness, load_harness_config
    from robo_jev.sim.environment import Environment

    env = Environment(config_path=config_paths(CONFIG)["sim_config"], profile="E0")
    try:
        scene = env.reset(seed=400101)
        harness = RobotHarness(load_harness_config(config_paths(CONFIG)["harness_config"]))
        request = harness.build_request(scene, None, None)
        policy.reset()
        answers = policy.act(request, None, scene)
    finally:
        env.close()
    expert_done = Expert().act(request, None, scene)["q_done"]
    assert answers["q_done"] == pytest.approx(expert_done)  # 하네스가 받는 값
    assert policy.raw == [{"t": int(request["t"]), "model_q_done": 0.9, "expert_q_done": pytest.approx(expert_done)}]
    for qid in QUESTION_SET_V0:  # 나머지 답은 모델의 것 그대로
        if qid != "q_done":
            assert answers[qid] == inner.act(request, None, scene)[qid]
    assert inner.resets == 1


def test_attach_raw_done_puts_the_models_answer_in_usage_and_refuses_a_misaligned_list():
    record = {"ticks": [{"t": 0, "usage": {}}, {"t": 5, "usage": {"gate": None}}], "evidence": {}}
    attach_raw_done(record, [{"t": 0, "model_q_done": 0.1, "expert_q_done": 0.05}, {"t": 5, "model_q_done": 0.8, "expert_q_done": 0.05}], policy_version="v")
    assert [tick["usage"]["model_q_done"] for tick in record["ticks"]] == [0.1, 0.8]
    assert record["evidence"]["done_gate"]["collection"] == COLLECTION_NAME and record["evidence"]["done_gate"]["policy"] == "v"
    with pytest.raises(ValueError, match="틱"):
        attach_raw_done({"ticks": [{"t": 0, "usage": {}}]}, [{"t": 5, "model_q_done": 0.1, "expert_q_done": 0.0}], policy_version="v")


def test_the_collection_seed_base_is_the_dagger_cycle_start_and_overlaps_no_other_range():
    from robo_jev.data.dagger import dagger_seed_schedule

    assert collection_seed_base(CONFIG, cycle=1) == 600100 == dagger_seed_schedule(CONFIG, 1, cycle=1)[0][1]  # 사이클 1 일정의 첫 seed
    for taken in (400100, 900100, 950100, 980100):
        assert abs(collection_seed_base(CONFIG, cycle=1) - taken) >= 100_000 - 1


def test_seed_selection_can_ask_for_some_profiles_only_and_lands_in_the_train_split():
    generator = CONFIG
    sim = yaml.safe_load(resolve_config_path(config_paths(generator)["sim_config"]).read_text(encoding="utf-8"))
    block = select_seeds(generator, sim, split="train", count=4, base=600100, per_profile_max=200, quota={"E1": 2, "E2": 2})
    assert block["quota"] == {"E0": 0, "E1": 2, "E2": 2} and block["by_profile"] == {"E0": 0, "E1": 2, "E2": 2}
    assert [item["profile"] for item in block["seeds"]].count("E0") == 0 and len(block["seeds"]) == 4
    assert all(item["split"] == "train" and item["holdout"] == [] for item in block["seeds"])
    assert "ood_test" not in json.dumps(block["seeds"])
    with pytest.raises(ValueError, match="count"):
        select_seeds(generator, sim, split="train", count=5, base=600100, per_profile_max=200, quota={"E1": 2, "E2": 2})


@pytest.fixture(scope="module")
def collected(tmp_path_factory):
    out = tmp_path_factory.mktemp("donegate") / "train"
    policy = DoneGatePolicy(AlwaysDone(), Expert())
    summary = collect_done_gate(policy, [("E0", 400100)], config=CONFIG, out=out, label="always-done", id_tag="r6", max_ticks=40,
                                describe={"name": "AlwaysDone", "version": "always-done-0-inner", "kind": "model", "checkpoint": "x.pt"})
    return out, summary


def test_a_collected_episode_does_not_end_on_the_models_false_done_and_records_it(collected):
    out, summary = collected
    records = [record for _, record in read_episodes(out)]
    assert len(records) == 1 and summary["episodes"] == 1
    record = records[0]
    validate_record(record)
    assert record["episode_id"] == "ep-E0-400100-r6-always-done"
    assert record["provenance"]["policy"] == {"name": "DoneGatePolicy", "version": f"{DONE_GATE_VERSION}/always-done-0"}
    # 모델(가짜)은 틱마다 0.9를 말했다 — 그 답은 usage에, 하네스가 받은 expert의 답은 model_output에 있다
    assert all(tick["usage"]["model_q_done"] == pytest.approx(0.9) for tick in record["ticks"])
    first_true = next((index for index, tick in enumerate(record["ticks"]) if reference_done(tick)), None)
    # 모델 자신의 게이트였다면 4틱째에 끝났을 편이 이어진다: done 게이트는 참조가 참이 된 뒤에만 든다
    gates = [index for index, tick in enumerate(record["ticks"]) if tick["usage"]["gate"] == "done"]
    assert len(record["ticks"]) > 4
    assert not gates or (first_true is not None and min(gates) >= first_true)
    assert all(model_q_done(tick) == pytest.approx(0.9) for tick in record["ticks"])
    events = false_done_events(records)
    assert events["episodes"] == 1 and events["ticks"] == sum(1 for tick in record["ticks"] if reference_done(tick) is False)
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["closed_loop"]["collection"] == COLLECTION_NAME and manifest["closed_loop"]["condition"] == "train"
    assert manifest["closed_loop"]["collection_only"] is True
    assert manifest["closed_loop"]["policy"]["version"] == f"{DONE_GATE_VERSION}/always-done-0"  # 안쪽 서술이 감싸개의 버전을 덮지 않는다


def test_the_collection_refuses_a_scene_outside_the_train_split(tmp_path):
    policy = DoneGatePolicy(AlwaysDone(), Expert())
    with pytest.raises(ValueError, match="train"):
        collect_done_gate(policy, [("E0", 900101)], config=CONFIG, out=tmp_path / "x", label="x", id_tag="r6", max_ticks=5)  # dev 계열


def test_the_evaluation_runner_refuses_the_collection_policy(tmp_path):
    config = load_closed_loop_config(REPO / "configs/eval/r4-closed-loop.yaml")
    bundle = build_policy("expert", generator=config["generator"])
    bundle = {**bundle, "policy": DoneGatePolicy(AlwaysDone(), Expert()), "describe": {"name": "DoneGatePolicy"}}
    with pytest.raises(ValueError, match="평가"):
        run_condition(bundle, [("E0", 900101)], config=config, out=tmp_path / "dev", condition="dev", label="x", max_ticks=3)


# --------------------------------------------------------------------------
# Task R12 — DAgger 사이클 2: 배치 구성 그대로 위임 갈래 G로 감싼 모델 위에 done 게이트를 겹친 수집
# --------------------------------------------------------------------------


class AlwaysDoneAlwaysClosed(AlwaysDone):
    """거짓 done을 틱마다 내고 그리퍼도 언제나 `closed`(0.95)라고 말하는 가짜 모델 — 위임 갈래가 그 그리퍼 답을 실행 규칙의 답으로 바꾸는지 보려고."""

    version = "always-done-closed-0"

    def act(self, request, commitment=None, observation=None):
        return {**super().act(request, commitment, observation), "q_gripper": {"open": 0.05, "closed": 0.95}}


def _delegated_done_gate(inner):
    from robo_jev.delegation import DelegatedExecutionPolicy, ExecutionGripperRule

    return DoneGatePolicy(DelegatedExecutionPolicy(inner, arm="G", rule=ExecutionGripperRule()), Expert())


def test_wrapper_chain_names_each_wrapper_from_the_outside_in_and_stops_at_the_model():
    """감싸개 사슬: done 게이트(이름·버전 dg0.1, `q_done`을 expert의 것으로) → 위임(이름·버전 dx0.1·갈래 G, `q_gripper`를 실행 규칙의 것으로). 모델은 넣지 않는다."""
    from robo_jev.data.done_gate import wrapper_chain
    from robo_jev.delegation import DELEGATION_VERSION, EXECUTION_GRIPPER_VERSION

    policy = _delegated_done_gate(AlwaysDoneAlwaysClosed())
    assert wrapper_chain(policy) == [
        {"name": "DoneGatePolicy", "version": DONE_GATE_VERSION, "replaces": ["q_done"], "source": "expert"},
        {"name": "DelegatedExecutionPolicy", "version": DELEGATION_VERSION, "arm": "G", "replaces": ["q_gripper"],
         "rule": {"name": "ExecutionGripperRule", "version": EXECUTION_GRIPPER_VERSION}},
    ]
    assert policy.version == f"{DONE_GATE_VERSION}/{DELEGATION_VERSION}/G/always-done-closed-0" and policy.collection_only is True
    assert wrapper_chain(DoneGatePolicy(AlwaysDone(), Expert())) == [wrapper_chain(policy)[0]]  # R6 꼴: done 게이트 하나


@pytest.fixture(scope="module")
def collected_delegated(tmp_path_factory):
    out = tmp_path_factory.mktemp("dagger2") / "train"
    policy = _delegated_done_gate(AlwaysDoneAlwaysClosed())
    summary = collect_done_gate(policy, [("E0", 400100)], config=CONFIG, out=out, label="r6G", id_tag="r12", max_ticks=40,
                                describe={"name": "DelegatedExecutionPolicy", "version": "dx0.1/G/always-done-closed-0", "kind": "model", "checkpoint": "x.pt", "arm": "G"})
    return out, summary


def test_the_done_gate_over_arm_g_hands_the_harness_the_experts_done_and_the_rules_gripper_and_keeps_both_raw_answers(collected_delegated):
    """R12 A1: 하네스가 받은 것(`model_output`) — `q_done`은 그 틱 expert의 값(모델 자신의 게이트였다면 4틱째 끝났을 편이 이어진다), `q_gripper`는 실행 규칙이
    그 틱 요청에서 낸 분포. 모델의 raw 답 — `q_done`은 `usage.model_q_done`, `q_gripper`는 `usage.delegation.inner`. `provenance.policy`가 두 감싸개를 이름·버전으로
    적고(갈래 G), manifest도 갈래·감싸개 사슬·수집 전용을 적는다. 레코드는 계약 검증을 지난다."""
    from robo_jev.delegation import DELEGATION_VERSION, ExecutionGripperRule

    out, summary = collected_delegated
    (record,) = [record for _, record in read_episodes(out)]
    validate_record(record)
    assert record["episode_id"] == "ep-E0-400100-r12-r6G" and summary["episodes"] == 1 and len(record["ticks"]) > 4
    policy = record["provenance"]["policy"]
    assert policy["name"] == "DoneGatePolicy" and policy["version"] == f"{DONE_GATE_VERSION}/{DELEGATION_VERSION}/G/always-done-closed-0"
    assert [entry["name"] for entry in policy["wrappers"]] == ["DoneGatePolicy", "DelegatedExecutionPolicy"]
    assert [entry["version"] for entry in policy["wrappers"]] == [DONE_GATE_VERSION, DELEGATION_VERSION] and policy["arm"] == "G"
    rule = ExecutionGripperRule()
    for tick in record["ticks"]:
        assert tick["usage"]["model_q_done"] == pytest.approx(0.9)  # 모델의 raw done
        assert tick["usage"]["delegation"]["inner"]["q_gripper"] == {"open": 0.05, "closed": 0.95}  # 모델의 raw 그리퍼
        assert tick["model_output"]["q_gripper"] == rule.distribution(rule.decide(tick, None)["desired"])  # 하네스가 받은 그리퍼 = 실행 규칙
        assert (tick["model_output"]["q_done"] >= 0.5) == bool(reference_done(tick))  # 하네스가 받은 done = expert의 값
    first_true = next((index for index, tick in enumerate(record["ticks"]) if reference_done(tick)), None)
    gates = [index for index, tick in enumerate(record["ticks"]) if tick["usage"]["gate"] == "done"]
    assert not gates or (first_true is not None and min(gates) >= first_true)
    assert record["evidence"]["delegation"]["q_gripper_changed_ticks"] > 0 and record["evidence"]["done_gate"]["collection"] == COLLECTION_NAME
    assert false_done_events([record])["episodes"] == 1
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    block = manifest["closed_loop"]
    assert block["collection_only"] is True and block["arm"] == "G" and block["wrappers"] == policy["wrappers"] and block["condition"] == "train"
    assert block["policy"]["name"] == "DoneGatePolicy" and block["policy"]["inner_version"] == "dx0.1/G/always-done-closed-0"


def test_the_evaluation_runner_refuses_the_done_gate_over_arm_g(tmp_path):
    """수집 정책은 위임 갈래를 안에 두어도 평가에 쓰지 않는다 — 평가 실행기가 거절한다(평가는 G를 등록된 후보 갈래로만 쓴다)."""
    config = load_closed_loop_config(REPO / "configs/eval/r4-closed-loop.yaml")
    bundle = build_policy("expert", generator=config["generator"])
    bundle = {**bundle, "policy": _delegated_done_gate(AlwaysDone()), "describe": {"name": "DoneGatePolicy"}}
    with pytest.raises(ValueError, match="평가"):
        run_condition(bundle, [("E0", 900101)], config=config, out=tmp_path / "dev", condition="dev", label="x", max_ticks=3)


def test_the_done_gate_over_arm_g_refuses_a_scene_outside_the_train_split(tmp_path):
    policy = _delegated_done_gate(AlwaysDone())
    with pytest.raises(ValueError, match="train"):
        collect_done_gate(policy, [("E0", 900101)], config=CONFIG, out=tmp_path / "x", label="x", id_tag="r12", max_ticks=5)  # dev 계열


def _collect_script():
    import importlib.util

    spec = importlib.util.spec_from_file_location("donegate_collect", REPO / "scripts/donegate_collect.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_seeds_command_walks_the_cycles_slot_for_a_quota_and_writes_it_under_the_condition_name(tmp_path, capsys):
    """수집 seed 명령 (R12): `--cycle 2`의 자리(700100)에서 `--quota`의 몫만큼 train 장면을 걷고 `--condition`의 이름으로 적는다(몫도 적는다). 기본은 R6 그대로
    (E1·E2 각 100, 조건 train_donegate). 몫의 글꼴이 틀리면 거절한다."""
    module = _collect_script()
    out = tmp_path / "seeds.json"
    assert module.main(["seeds", "--cycle", "2", "--quota", "E1=2,E2=1,E0=1", "--condition", "train_dagger2", "--per-profile-max", "200", "--out", str(out)]) == 0
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["seeds_base"] == 700100 and payload["cycle"] == 2 and payload["quota"] == {"E1": 2, "E2": 1, "E0": 1}
    block = payload["conditions"]["train_dagger2"]
    assert block["by_profile"] == {"E0": 1, "E1": 2, "E2": 1} and all(item["split"] == "train" for item in block["seeds"])
    spans = {"E0": (700100, 704100), "E1": (710100, 718100), "E2": (720100, 728100)}  # 사이클 2의 자리 + 프로파일 × 10,000
    assert all(spans[item["profile"]][0] <= item["seed"] < spans[item["profile"]][1] for item in block["seeds"])
    assert "ood_test" not in json.dumps(block["seeds"]) and list(payload["conditions"]) == ["train_dagger2"]
    assert module.parse_quota(None) == {"E1": 100, "E2": 100} and module.CONDITION == "train_donegate"
    for bad in ("E1=0", "E1=2,E1=3", "E1", "=4"):
        with pytest.raises(ValueError, match="quota"):
            module.parse_quota(bad)
