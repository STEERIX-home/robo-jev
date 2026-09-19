"""DAgger 사이클 검사 — 실행된 것은 그대로, 전문가 답은 labels에만, 정책 오류·복구·commitment 변경 집계 (docs/04 §7)."""

import json

import pytest
from helpers import SIM_CONFIG
import yaml

from robo_jev.contracts import model_input, validate_record
from robo_jev.data.dagger import (
    DAGGER_VERSION,
    PolicyClient,
    count_policy_behaviour,
    dagger_seed_schedule,
    rule_judge_policy,
    run_cycle,
)
from robo_jev.data.robot_episodes import (
    build_manifest,
    episode_id,
    generate_episode,
    load_generator_config,
    origin_group,
    plan_tags,
    read_episodes,
    seed_schedule,
    write_episode,
)
from robo_jev.data.split import SplitPolicy, assign_split
from robo_jev.harness.robot import parse_exec_history
from robo_jev.sim.scene import build_plan

CONFIG = load_generator_config()
SIM = yaml.safe_load(SIM_CONFIG.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def cycle(tmp_path_factory):
    out = tmp_path_factory.mktemp("dagger")
    # cycle 1 → seed 200100부터: E0·E1 모두 20틱을 다 돈다 (cycle 3의 E1 seed 400100은 대상이 처음부터 영역 안이라 done 꼬리로 11틱에 끝난다).
    manifest = run_cycle(rule_judge_policy(), episodes=2, out=out, config=CONFIG, cycle=1, max_ticks=20)
    records = [record for _, record in read_episodes(out)]
    return {"out": out, "manifest": manifest, "records": records}


def test_dagger_seeds_and_ids_never_collide_with_the_expert_batch():
    """DAgger 사이클의 seed는 `seeds.base + (cycle + 1) × seeds.dagger_cycle_offset`부터이고 id에는 `-dagger{cycle}`이
    붙는다 — D1 전문가 에피소드(`ep-E0-000100`)와 같은 id로 파일·manifest·`by_id`를 덮어쓰지 않는다."""
    offset = CONFIG["seeds"]["dagger_cycle_offset"]
    base = CONFIG["seeds"]["base"]
    assert offset >= 100_000
    assert dagger_seed_schedule(CONFIG, 4, cycle=0) == [
        ("E0", base + offset), ("E1", base + offset), ("E0", base + offset + 1), ("E1", base + offset + 1),
    ]
    assert dagger_seed_schedule(CONFIG, 2, cycle=3) == [("E0", base + 4 * offset), ("E1", base + 4 * offset)]
    expert_seeds = {seed for _, seed in seed_schedule(CONFIG, offset)}
    for cycle_index in range(3):
        for _, seed in dagger_seed_schedule(CONFIG, offset, cycle=cycle_index):
            assert seed not in expert_seeds
    assert episode_id("E0", base + offset, "-dagger0") == f"ep-E0-{base + offset:06d}-dagger0"
    assert dagger_seed_schedule(CONFIG, 200, cycle=1)[:2] == dagger_seed_schedule(CONFIG, 2, cycle=1)  # prefix 성질 그대로
    without = {**CONFIG, "seeds": {"base": base}}
    assert dagger_seed_schedule(without, 1, cycle=0) == [("E0", base + 100_000)]  # 기본 offset


def test_a_dagger_cycle_written_next_to_an_expert_batch_shares_no_id_and_the_manifest_merges(cycle, tmp_path):
    import shutil

    from robo_jev.sim.expert import Expert

    out = tmp_path / "merged"
    shutil.copytree(cycle["out"] / "episodes", out / "episodes")  # DAgger 사이클의 파일을 전문가 배치 옆에 둔다
    expert = Expert()
    expert_ids = []
    for profile, seed in seed_schedule(CONFIG, 2):
        record = generate_episode(profile, seed, policy=expert, expert=expert, config=CONFIG, max_ticks=3)
        assert record["episode_id"] == episode_id(profile, seed) and "dagger" not in record["episode_id"]
        write_episode(record, out)
        expert_ids.append(record["episode_id"])
    dagger_ids = [record["episode_id"] for record in cycle["records"]]
    assert all(identifier.endswith("-dagger1") for identifier in dagger_ids)
    assert not set(expert_ids) & set(dagger_ids)
    merged = build_manifest(out, CONFIG, batch_wall_s=1.0)
    assert merged["episodes"] == 4
    assert set(merged["files"]) == {f"episodes/{identifier}/streams.jsonl" for identifier in expert_ids + dagger_ids}
    assert {entry["episode_id"] for entry in merged["files"].values()} == set(expert_ids + dagger_ids)
    assert len(read_episodes(out)) == 4
    policy = SplitPolicy.from_config(CONFIG["split"])
    for record in cycle["records"]:
        # 같은 장면 계열 → 같은 split. 봉인 태그(문구 변형·zoneF 목표)는 계획에서 나오므로 계획을 다시 지어 맞댄다.
        plan = build_plan(SIM, record["provenance"]["seed"], record["provenance"]["profile"])
        assert record["origin_group"] == origin_group(record["provenance"]["profile"], plan)
        assert record["split"] == assign_split(record["origin_group"], policy, plan_tags(plan))
        assert record["provenance"]["seed"] >= CONFIG["seeds"]["base"] + 2 * CONFIG["seeds"]["dagger_cycle_offset"]
        assert record["provenance"]["dagger"]["seed_base"] == CONFIG["seeds"]["base"] + 2 * CONFIG["seeds"]["dagger_cycle_offset"]
    assert cycle["manifest"]["dagger"]["seed_base"] == CONFIG["seeds"]["base"] + 2 * CONFIG["seeds"]["dagger_cycle_offset"]


def test_two_short_episodes_keep_the_executed_history_and_carry_expert_relabels(cycle):
    records = cycle["records"]
    assert len(records) == 2 and {record["provenance"]["profile"] for record in records} == {"E0", "E1"}
    for record in records:
        validate_record(record)
        assert record["provenance"]["policy"] == {"name": "RuleJudge", "version": rule_judge_policy().version}
        assert record["provenance"]["dagger"]["cycle"] == 1 and record["provenance"]["dagger"]["version"] == DAGGER_VERSION
        ticks = record["ticks"]
        assert len(ticks) == 20 and ticks[0]["request"]["exec_history"] == "none"
        for previous, tick in zip(ticks, ticks[1:]):
            # 실행 이력은 직전 틱에 실제로 채택·실행된 것이다 — 라벨(전문가 답)이 아니다.
            history = parse_exec_history(tick["request"]["exec_history"])
            assert history["main"] == previous["adopted"]["main"]
            assert history["gripper"] == previous["adopted"]["gripper"]
            assert history["ack"] == ("ok" if previous["ack"]["applied"] else str(previous["ack"]["reason"]))
        for tick in ticks:
            assert tick["model_output"] and tick["adopted"] and tick["ack"]
            labels = {label["question_id"]: label for label in tick["labels"]}
            assert "q_main" in labels and all(label["source"] == "expert_v0" and label["relabel"] is True for label in labels.values())
            assert tick["request"]["commitment"] is None or tick["request"]["commitment"]["action_ref"] in {
                entry["id"] for entry in tick["request"]["candidates"]["q_main"]
            }
        served = model_input(record)
        assert "relabel" not in json.dumps(served, ensure_ascii=False) and "labels" not in json.dumps(served, ensure_ascii=False)


def test_the_cycle_counts_policy_errors_recoveries_commitment_changes_and_gates(cycle):
    dagger = cycle["manifest"]["dagger"]
    assert dagger["episodes"] == 2 and dagger["policy"]["name"] == "RuleJudge" and dagger["relabel_source"] == "expert_v0"
    totals = dagger["totals"]
    assert set(totals) >= {"ticks", "policy_errors", "gate_disagreement", "main_disagreement", "recoveries", "commitment_changes", "gates", "stops"}
    assert totals["ticks"] == 40 and totals["policy_errors"] == totals["gate_disagreement"] + totals["main_disagreement"]
    assert 0 <= totals["recoveries"] <= totals["policy_errors"]
    assert 0.0 <= dagger["policy_error_rate"] <= 1.0
    assert len(dagger["per_episode"]) == 2
    for entry in dagger["per_episode"]:
        assert entry == count_policy_behaviour(next(record for record in cycle["records"] if record["episode_id"] == entry["episode_id"]))
    assert json.loads((cycle["out"] / "manifest.json").read_text(encoding="utf-8"))["dagger"] == dagger


def test_a_stand_in_policy_behind_the_callable_interface_is_recorded_as_the_executor():
    """`policy(request) -> results` 인터페이스: 언제나 hold를 고르는 정책은 전문가와 어긋나 정책 오류가 된다."""
    from robo_jev.sim.expert import Expert

    expert = Expert()

    def always_hold(request):
        answers = expert.act(request, None)
        candidates = request["request"]["candidates"]["q_main"]
        hold = next(entry["id"] for entry in candidates if entry["key"] == "hold")
        rest = [entry["id"] for entry in candidates if entry["id"] != hold]
        answers["q_main"] = {hold: 0.9, **{cid: round(0.1 / len(rest), 6) for cid in rest}}
        return {question: answers[question] for question in answers if question.startswith("q_")}

    from robo_jev.data.robot_episodes import generate_episode

    client = PolicyClient(always_hold, name="AlwaysHold", version="test")
    record = generate_episode("E0", 17, policy=client, expert=expert, config=CONFIG, max_ticks=12)
    behaviour = count_policy_behaviour(record)
    assert record["provenance"]["policy"] == {"name": "AlwaysHold", "version": "test"} and behaviour["ticks"] == 12
    assert behaviour["policy_errors"] >= 10 and behaviour["main_disagreement"] >= 10
    assert behaviour["commitment_changes"] == 0
