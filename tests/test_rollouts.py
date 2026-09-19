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


def test_replay_refuses_a_record_made_by_other_configs_with_an_explicit_reason(short_episode):
    """레코드의 `versions`(config_digest와 하네스·컨트롤러·전문가 버전)가 지금 돌아가는 것과 다르면 재생은 그 레코드를
    거절한다 — 조용한 `skipped_fidelity`가 아니라 무엇이 다른지 말하는 예외다."""
    import copy

    from robo_jev.data.rollouts import ConfigMismatch, running_versions_for

    record = short_episode["record"]
    running = running_versions_for(sim_config=CONFIG["sim_config"], events=EVENTS)
    assert record["versions"]["config_digest"] == running["config_digest"]
    kwargs = dict(sim_config=CONFIG["sim_config"], harness_config=load_harness_config(), control_steps=CONFIG["episode"]["control_steps_per_tick"], running=running)

    stale = copy.deepcopy(record)
    stale["versions"]["config_digest"] = "0" * 64
    with pytest.raises(ConfigMismatch) as excinfo:
        replay_to_keyframes(stale, [0], **kwargs)
    message = str(excinfo.value)
    assert stale["episode_id"] in message and "config_digest" in message and "0" * 64 in message and running["config_digest"] in message

    older = copy.deepcopy(record)
    older["versions"]["harness"] = "h0.2"
    with pytest.raises(ConfigMismatch, match="harness"):
        replay_to_keyframes(older, [0], **kwargs)

    undated = copy.deepcopy(record)
    del undated["versions"]["config_digest"]
    with pytest.raises(ConfigMismatch, match="config_digest"):
        replay_to_keyframes(undated, [0], **kwargs)

    # 추출기·규칙 기준군·레코드 직렬화·장면 버전도 재생의 전제다 — 올라간 것이 조용한 fidelity skip이 되지 않는다.
    from robo_jev.data.rollouts import VERSION_KEYS

    assert set(VERSION_KEYS) == {"harness", "controller", "expert", "rules", "serializer", "extractor", "sim", "config_digest"}
    for key in ("extractor", "rules", "serializer", "sim"):
        assert record["versions"][key] == running[key]
        bumped = copy.deepcopy(record)
        bumped["versions"][key] = f"{record['versions'][key]}-next"
        with pytest.raises(ConfigMismatch, match=key):
            replay_to_keyframes(bumped, [0], **kwargs)

    out = short_episode["out"]
    (out / "episodes" / stale["episode_id"] / "streams.jsonl").write_text(json.dumps(stale, ensure_ascii=False) + "\n", encoding="utf-8")
    try:
        with pytest.raises(ConfigMismatch):
            run(out, limit=1, workers=1, per_episode=1, out=out / "refused")
    finally:
        (out / "episodes" / record["episode_id"] / "streams.jsonl").write_text(json.dumps(record, ensure_ascii=False) + "\n", encoding="utf-8")


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


def test_a_worker_pool_produces_the_same_evidence_as_the_serial_run(short_episode):
    """비용 산정이 기대는 성질: worker 풀(spawn, worker마다 환경 하나)의 결과가 직렬과 같다 — 시간·pid만 다르다."""
    from robo_jev.data.rollouts import run_jobs

    record = short_episode["record"]
    jobs, _, _ = build_jobs(
        [record], EVENTS, limit=2, sim_config=CONFIG["sim_config"], harness_config=load_harness_config(),
        control_steps=CONFIG["episode"]["control_steps_per_tick"],
    )
    assert len(jobs) == 2
    serial = run_jobs(jobs, sim_config=CONFIG["sim_config"], expert_config=EVENTS["followup"]["expert_config"], workers=1)
    pooled = run_jobs(jobs, sim_config=CONFIG["sim_config"], expert_config=EVENTS["followup"]["expert_config"], workers=2)
    pids = {result["job"]["worker_pid"] for result in pooled}
    assert 1 <= len(pids) <= 2 and all(pid != serial[0]["job"]["worker_pid"] for pid in pids)  # 다른 프로세스가 돌렸다
    timing = {"wall_s", "restore_s"}
    for left, right in zip(serial, pooled):
        assert (left["outcome"], left["reason"]) == (right["outcome"], right["reason"])
        assert {k: v for k, v in left["evidence"].items() if k not in timing} == {k: v for k, v in right["evidence"].items() if k not in timing}
        assert {k: v for k, v in left["job"].items() if k not in ("worker_pid", "env_rebuild_s")} == {
            k: v for k, v in right["job"].items() if k not in ("worker_pid", "env_rebuild_s")
        }


def test_summarise_sweep_conditions_push_success_on_approach_time_and_start_distance(short_episode):
    """128k 전 관문의 요약: 사건별 결과, 밀기 성공률을 접근 시간·시작 거리·방향·키프레임 종류로 조건화한 표(Wilson 구간),
    censoring 사유, 키프레임 종류 혼합, 산정. rollout 파일에서만 만들고 다시 돌리지 않는다."""
    from robo_jev.data.rollouts import summarise_sweep

    out = short_episode["out"] / "sweep"
    outcome = run(short_episode["out"], limit=16, workers=1, out=out)
    summary = summarise_sweep(out)
    assert (out / "sweep-summary.json").is_file()
    assert summary["rollouts"] == 16 == sum(summary["outcomes"].values())
    assert sum(sum(counts.values()) for counts in summary["by_event"].values()) == 16
    assert set(summary["keyframes"]["kinds"]) and summary["keyframes"]["count"] == len(outcome["keyframes"])
    for table in ("push_by_approach_s", "push_by_start_distance_mm", "push_by_direction", "push_by_keyframe_kind", "grasp_by_start_distance_mm"):
        for row in summary[table].values():
            assert row["rollouts"] == row["success"] + row["failure"] + row["censored"]
            assert row["wilson"][0] <= (row["rate"] if row["rate"] is not None else 1.0) <= row["wilson"][1] + 1e-9
    pushes = sum(row["rollouts"] for row in summary["push_by_direction"].values())
    assert pushes == summary["by_event"].get("push", {}).get("success", 0) + summary["by_event"].get("push", {}).get("failure", 0) + summary["by_event"].get("push", {}).get("censored", 0)
    assert summary["projections"]["d1_128k"]["cpu_hours"] > 0 and summary["wall_s_per_rollout"]["mean"] > 0


def test_costing_counts_outcomes_by_function():
    def result(key, outcome, reason=None, wall=0.5):
        return {"outcome": outcome, "reason": reason, "evidence": {"wall_s": wall, "restore_s": 0.05, "ticks": 30},
                "job": {"key": key, "env_rebuild_s": None}}

    cost = costing([result("grasp:o0:top:zoneL", "success"), result("push:o1:+x:none", "failure", "horizon"),
                    result("push:o1:+x:none", "censored", "wall_time_limit", wall=2.0)], batch_wall_s=3.0, replay_s_total=1.0, workers=1)
    assert cost["outcomes"] == {"censored": 1, "failure": 1, "success": 1}
    assert cost["by_function"]["push"] == {"rollouts": 2, "success": 0, "failure": 1, "censored": 1}
    assert cost["reasons"] == {"horizon": 1, "wall_time_limit": 1}
    assert cost["projections"]["d1_128k"]["cpu_hours"] == pytest.approx(128_000 * 1.0 / 3600, abs=0.01)
