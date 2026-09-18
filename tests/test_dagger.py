"""DAgger 사이클 검사 — 실행된 것은 그대로, 전문가 답은 labels에만, 정책 오류·복구·commitment 변경 집계 (docs/04 §7)."""

import json

import pytest

from robo_jev.contracts import model_input, validate_record
from robo_jev.data.dagger import DAGGER_VERSION, PolicyClient, count_policy_behaviour, rule_judge_policy, run_cycle
from robo_jev.data.robot_episodes import load_generator_config, read_episodes
from robo_jev.harness.robot import parse_exec_history

CONFIG = load_generator_config()


@pytest.fixture(scope="module")
def cycle(tmp_path_factory):
    out = tmp_path_factory.mktemp("dagger")
    manifest = run_cycle(rule_judge_policy(), episodes=2, out=out, config=CONFIG, cycle=3, max_ticks=20)
    records = [record for _, record in read_episodes(out)]
    return {"out": out, "manifest": manifest, "records": records}


def test_two_short_episodes_keep_the_executed_history_and_carry_expert_relabels(cycle):
    records = cycle["records"]
    assert len(records) == 2 and {record["provenance"]["profile"] for record in records} == {"E0", "E1"}
    for record in records:
        validate_record(record)
        assert record["provenance"]["policy"] == {"name": "RuleJudge", "version": rule_judge_policy().version}
        assert record["provenance"]["dagger"]["cycle"] == 3 and record["provenance"]["dagger"]["version"] == DAGGER_VERSION
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
