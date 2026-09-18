"""키프레임 rollout 제작 파이프라인 검사 — 재생 충실도, 작업 순서, 출력 파일, 비용 산정 (Task 3c-2)."""

import json

import pytest

from robo_jev.contracts import validate_record
from robo_jev.data.robot_episodes import generate_episode, load_generator_config, write_episode
from robo_jev.data.rollouts import build_jobs, costing, replay_to_keyframes, run
from robo_jev.harness.robot import load_harness_config
from robo_jev.sim.expert import Expert
from robo_jev.sim.label import load_events_config, select_keyframes

CONFIG = load_generator_config()
EVENTS = load_events_config()


@pytest.fixture(scope="module")
def short_episode(tmp_path_factory):
    """E0 seed 5를 25틱만 돈 에피소드 (접근 중의 키프레임들)."""
    expert = Expert()
    record = generate_episode("E0", 5, policy=expert, expert=expert, config=CONFIG, max_ticks=25)
    out = tmp_path_factory.mktemp("batch")
    write_episode(record, out)
    return {"record": record, "out": out}


def test_replay_reaches_the_keyframe_with_the_recorded_request(short_episode):
    record = short_episode["record"]
    keyframes = select_keyframes(record, {"keyframes": {"per_episode": 5}})
    indices = [frame["index"] for frame in keyframes]
    found = replay_to_keyframes(
        record, indices, sim_config=CONFIG["sim_config"], harness_config=load_harness_config(),
        control_steps=CONFIG["episode"]["control_steps_per_tick"],
    )
    assert set(found) == set(indices)
    for index, entry in found.items():
        assert entry["exact"] is True, (index, entry["fidelity"])
        assert isinstance(entry["snapshot"], bytes) and len(entry["snapshot"]) > 1000
        assert entry["precision_mm"]
    committed = [index for index in indices if record["ticks"][index]["request"]["commitment"]]
    assert all(found[index]["commitment"]["action_ref"] == record["ticks"][index]["request"]["commitment"]["action_ref"] for index in committed)


def test_jobs_cover_every_candidate_with_paired_seeds_before_the_next_keyframe(short_episode):
    record = short_episode["record"]
    jobs, keyframes, summary = build_jobs(
        [record], EVENTS, limit=20, sim_config=CONFIG["sim_config"], harness_config=load_harness_config(),
        control_steps=CONFIG["episode"]["control_steps_per_tick"],
    )
    assert len(jobs) == 20 and summary["skipped_fidelity"] == 0
    first = jobs[0]["keyframe"]
    assert all(job["keyframe"] == first for job in jobs)
    candidates = keyframes[0]["candidates"]
    assert len(candidates) == 8
    assert [job["action"]["id"] for job in jobs[:8]] == candidates and [job["seed"] for job in jobs[:8]] == [0] * 8
    assert [job["seed"] for job in jobs[8:16]] == [1] * 8
    assert {job["event"]["event_id"] for job in jobs} >= {"grasp-lift-v0", "push-segment-v0"}
    assert keyframes[0]["expert_choice"] in candidates
    assert "snapshot" not in keyframes[0] and keyframes[0]["snapshot_bytes"] > 0


def test_the_pipeline_writes_rollouts_labels_and_a_costing_that_extrapolates(short_episode):
    out = short_episode["out"]
    outcome = run(out, limit=3, workers=1, per_episode=1)
    assert outcome["jobs"] == 3 and len(outcome["results"]) == 3
    paths = outcome["paths"]
    for path in paths.values():
        assert path.is_file()
    rollouts = [json.loads(line) for line in paths["rollouts"].read_text(encoding="utf-8").splitlines()]
    assert len(rollouts) == 3 and all(r["outcome"] in ("success", "failure", "censored") for r in rollouts)
    labels = [json.loads(line) for line in paths["labels"].read_text(encoding="utf-8").splitlines()]
    assert len(labels) == 1 and labels[0]["rollouts"] == 3
    label = labels[0]["label"]
    assert label["question_id"] == "q_main" and label["source"] == "rollout_v0" and label["event_results"]
    tick = dict(short_episode["record"]["ticks"][labels[0]["index"]])
    validate_record({"schema_version": "stream-v0", "episode_id": "ep-x",
                     "prefix": {"instructions": [{"version": 1, "t_ms": 0, "text": "x"}], "question_set": "qs-v0"},
                     "ticks": [{**tick, "labels": [label]}]})
    cost = json.loads(paths["costing"].read_text(encoding="utf-8"))
    assert cost["rollouts"] == 3 and cost["wall_s_per_rollout"]["mean"] > 0
    assert cost["restore_s_per_rollout"]["mean"] > 0 and cost["env_rebuild_s"]["count"] == 1
    assert cost["bytes_per_rollout"]["mean"] > 0
    for name in ("d1_128k", "d2_1_28m", "d2_2_56m"):
        projection = cost["projections"][name]
        assert projection["cpu_hours"] > 0 and projection["wall_hours_with_all_cores"] <= projection["cpu_hours"]
    assert cost["machine"]["cpu_count"] >= 1 and cost["keyframes"]["labelled"] == 1


def test_costing_counts_outcomes_by_function():
    def result(key, outcome, reason=None, wall=0.5):
        return {"outcome": outcome, "reason": reason, "evidence": {"wall_s": wall, "restore_s": 0.05, "ticks": 30},
                "job": {"key": key, "env_rebuild_s": None}}

    cost = costing([result("grasp:o0:top:zoneL:slow", "success"), result("push:o1:+x:none:slow", "failure", "horizon"),
                    result("push:o1:+x:none:fast", "censored", "wall_time_limit", wall=2.0)], batch_wall_s=3.0, replay_s_total=1.0, workers=1)
    assert cost["outcomes"] == {"censored": 1, "failure": 1, "success": 1}
    assert cost["by_function"]["push"] == {"rollouts": 2, "success": 0, "failure": 1, "censored": 1}
    assert cost["reasons"] == {"horizon": 1, "wall_time_limit": 1}
    assert cost["projections"]["d1_128k"]["cpu_hours"] == pytest.approx(128_000 * 1.0 / 3600, abs=0.01)
