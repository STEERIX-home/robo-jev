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
    assert collection_seed_base(CONFIG, cycle=1) == 600100
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
