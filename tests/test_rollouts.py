"""키프레임 rollout 제작 파이프라인 검사 — 재생 충실도, 작업 순서, 출력 파일, 비용 산정 (Task 3c-2)."""

import json
from pathlib import Path

import pytest
import yaml

from robo_jev.contracts import validate_record
from robo_jev.data.robot_episodes import generate_episode, load_generator_config, read_episodes, write_episode
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
    # 리뷰 1 M1: 에피소드 안의 키프레임 순서는 에피소드 id로 seed한 무작위다 — 첫 키프레임이 언제나 t=0 `switch`가 아니고,
    # 같은 레코드는 같은 순서다.
    import random

    from robo_jev.sim.label import select_keyframes

    expected = select_keyframes(record, {"keyframes": {"per_episode": 5}})
    random.Random(f"jobs:{record['episode_id']}").shuffle(expected)
    again, _, _ = build_jobs(
        [record], EVENTS, limit=20, sim_config=CONFIG["sim_config"], harness_config=load_harness_config(),
        control_steps=CONFIG["episode"]["control_steps_per_tick"],
    )
    assert keyframes[0]["index"] == expected[0]["index"] and again[0]["keyframe"] == first
    assert [frame["index"] for frame in expected] != sorted(frame["index"] for frame in expected) or len(expected) < 3
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
    # 리뷰 1 I5: 접근 구간마다·방향마다 실패 이유를 단계(접근 중 충돌·밀기 중 충돌·horizon)로 가른다.
    assert sum(sum(counts.values()) for counts in summary["push_reasons_by_approach_s"].values()) == pushes
    assert sum(sum(counts.values()) for counts in summary["push_stage_by_direction"].values()) == pushes
    allowed = {"success", "censored", "approach_contact_force", "push_contact_force", "approach_horizon", "push_horizon", "push_None", "approach_None"}
    assert all(set(counts) <= allowed for counts in summary["push_stage_by_direction"].values()), summary["push_stage_by_direction"]
    assert all(set(counts) <= allowed for counts in summary["push_reasons_by_approach_s"].values())
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


def test_the_push_contact_ab_driver_runs_the_same_push_jobs_under_each_arms_harness_override(short_episode, tmp_path):
    """D1-prep 리뷰 2 N3: 옛 A/B는 스크립트 없이 돌아 재현할 수 없었다. 드라이버는 배치의 밀기 job을 팔마다 override한 하네스
    yaml(+ 그것을 가리키는 전문가 yaml)로 돌리고, 명령줄·override·설정 sha256·방향별/접촉 부위별 단계 표를 JSON에 적는다.
    `null`은 키 삭제다(도달 표를 지운 팔은 h0.6의 구간, 스칼라 접촉 거리 팔은 옛 값)."""
    from robo_jev.data.push_ab import apply_override, run as run_ab

    config = load_harness_config()
    assert apply_override(config, {"candidates": {"push_reach_mm": None}})["candidates"].get("push_reach_mm") is None
    assert apply_override(config, {"candidates": {"push_contact_mm": 30}})["candidates"]["push_contact_mm"] == 30
    assert apply_override(config, {})["candidates"] == config["candidates"]

    out = tmp_path / "ab.json"
    report = run_ab(
        short_episode["out"], {"h0.6": {"candidates": {"push_reach_mm": None}}, "h0.7": {}}, workers=1, limit_keyframes=1,
        out=out, scratch=tmp_path / "arms", events_override={"seeds": 1}, command=["scripts/push_contact_ab.py", "--smoke"],
    )
    assert out.is_file() and json.loads(out.read_text(encoding="utf-8"))["command"] == ["scripts/push_contact_ab.py", "--smoke"]
    assert report["push_jobs"] >= 1 and report["seeds"] == 1
    assert set(report["arms"]) == {"h0.6", "h0.7"}
    old, new = report["arms"]["h0.6"], report["arms"]["h0.7"]
    assert old["harness_sha256"] != new["harness_sha256"] and old["override"] == {"candidates": {"push_reach_mm": None}}
    assert "push_reach_mm" not in yaml.safe_load(Path(old["harness_config"]).read_text(encoding="utf-8"))["candidates"]
    assert yaml.safe_load(Path(new["harness_config"]).read_text(encoding="utf-8"))["candidates"]["push_reach_mm"] == config["candidates"]["push_reach_mm"]
    assert yaml.safe_load(Path(new["expert_config"]).read_text(encoding="utf-8"))["harness_config"] == str(Path(new["harness_config"]).resolve())
    for arm in (old, new):
        assert sum(arm["outcomes"].values()) == report["push_jobs"]
        assert sum(row["rollouts"] for row in arm["by_direction"].values()) == report["push_jobs"]
        assert sum(row["rollouts"] for row in arm["by_class"].values()) == report["push_jobs"]
        assert all(name.split(":")[0] in ("fingers", "hand") for name in arm["by_class"])
    assert sum(report["jobs_by_direction"].values()) == report["push_jobs"]


@pytest.fixture(scope="module")
def two_episodes(tmp_path_factory):
    """E0 seed 5·6을 20틱씩 돈 배치 (재개 검사용)."""
    expert = Expert()
    out = tmp_path_factory.mktemp("batch2")
    records = []
    for seed in (5, 6):
        record = generate_episode("E0", seed, policy=expert, expert=expert, config=CONFIG, max_ticks=20)
        write_episode(record, out)
        records.append(record)
    return {"records": records, "out": out}


def test_the_rollout_run_is_resumable_per_episode_and_reproduces_a_fresh_run(two_episodes, tmp_path):
    """D1 128k: 묶음마다 증분 파일에 덧붙이고 완료 에피소드에 표지를 적는다. `--limit`에 잘린 묶음은 표지가 없어 재개가 다시 돌리고,
    표지가 있는 에피소드는 건너뛴다. 재개한 결과는 한 번에 돌린 것과 같다(같은 snapshot·seed는 같은 궤적)."""
    from robo_jev.data.rollouts import KEYFRAMES_INCREMENTAL, PROGRESS_FILE, ConfigMismatch

    dataset = two_episodes["out"]
    small = {"seeds": 1}
    common = dict(workers=1, per_episode=2, candidates_per_keyframe=2, chunk_episodes=1, events_override=small)
    # 한 번에 (기준).
    whole = run(dataset, limit=None, out=tmp_path / "whole", **common)
    key = lambda result: (result["job"]["keyframe"], result["job"]["candidate"], result["job"]["seed"])  # noqa: E731
    reference = {key(result): (result["outcome"], result["reason"]) for result in whole["results"]}
    assert len(reference) == len(whole["results"]) >= 4
    markers = [json.loads(line) for line in (tmp_path / "whole" / PROGRESS_FILE).read_text(encoding="utf-8").splitlines()]
    assert [marker["episode_id"] for marker in markers] == [record["episode_id"] for record in two_episodes["records"]]
    assert sum(marker["rollouts"] for marker in markers) == len(whole["results"])
    # 잘린 첫 실행: 첫 에피소드는 완료, 둘째는 limit에 잘려 표지가 없다.
    first_jobs = markers[0]["rollouts"]
    partial = run(dataset, limit=first_jobs + 1, out=tmp_path / "resumed", **common)
    assert len(partial["results"]) == first_jobs + 1
    progress = [json.loads(line) for line in (tmp_path / "resumed" / PROGRESS_FILE).read_text(encoding="utf-8").splitlines()]
    assert [marker["episode_id"] for marker in progress] == [markers[0]["episode_id"]]
    assert len((tmp_path / "resumed" / "rollouts.jsonl").read_text(encoding="utf-8").splitlines()) == first_jobs + 1
    # 재개: 첫 에피소드는 건너뛰고(결과 재사용) 둘째를 처음부터 돌린다; 부분 결과는 버린다.
    resumed = run(dataset, limit=None, out=tmp_path / "resumed", resume=True, **common)
    assert resumed["jobs"] == markers[1]["rollouts"]
    got = {key(result): (result["outcome"], result["reason"]) for result in resumed["results"]}
    assert got == reference
    lines = (tmp_path / "resumed" / "rollouts.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == len(reference)
    frames = [json.loads(line) for line in (tmp_path / "resumed" / KEYFRAMES_INCREMENTAL).read_text(encoding="utf-8").splitlines()]
    assert len(frames) == len(json.loads((tmp_path / "resumed" / "keyframes.json").read_text(encoding="utf-8"))) == len(whole["keyframes"])
    progress = [json.loads(line) for line in (tmp_path / "resumed" / PROGRESS_FILE).read_text(encoding="utf-8").splitlines()]
    assert [marker["episode_id"] for marker in progress] == [marker["episode_id"] for marker in markers]
    cost = json.loads((tmp_path / "resumed" / "costing.json").read_text(encoding="utf-8"))
    assert cost["rollouts"] == len(reference) and cost["keyframes"]["resumed"] is True and cost["keyframes"]["episodes_done"] == 2
    labels = [json.loads(line) for line in (tmp_path / "resumed" / "labels.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(labels) == len(whole["labels"])
    # 재개는 아무것도 다시 돌리지 않는다; 표지의 버전이 다르면 거절한다.
    again = run(dataset, limit=None, out=tmp_path / "resumed", resume=True, **common)
    assert again["jobs"] == 0 and len(again["results"]) == len(reference)
    poisoned = tmp_path / "resumed" / PROGRESS_FILE
    rows = [json.loads(line) for line in poisoned.read_text(encoding="utf-8").splitlines()]
    rows[0]["running_versions"]["harness"] = "h0.0"
    poisoned.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    with pytest.raises(ConfigMismatch):
        run(dataset, limit=None, out=tmp_path / "resumed", resume=True, **common)


def test_rollout_labels_are_attached_in_a_successor_dataset_version_that_keeps_the_lineage(two_episodes, tmp_path):
    """docs/04 §6: 키프레임 rollout 라벨은 계보를 유지한 후속 버전에 붙는다 — 라벨한 틱의 `q_main`만 rollout 라벨로 바뀌고(전문가
    라벨은 `expert`에 남는다), 다른 것은 그대로이며 `versions.labels`·`provenance.lineage`·manifest의 `lineage`·`rollout_labels`가
    계보와 집계를 말한다. QA를 지난다. rollout의 버전이 레코드와 다르면 거절한다."""
    from robo_jev.data.lineage import LABELS_VERSION, attach_rollout_labels
    from robo_jev.data.robot_episodes import build_manifest
    from robo_jev.data.rollouts import ConfigMismatch
    from robo_jev.data.validate import validate_dataset

    dataset = two_episodes["out"]
    build_manifest(dataset, CONFIG)
    rollouts = tmp_path / "rollouts"
    run(dataset, limit=None, out=rollouts, workers=1, per_episode=2, candidates_per_keyframe=2, chunk_episodes=8, events_override={"seeds": 1})
    labels = [json.loads(line) for line in (rollouts / "labels.jsonl").read_text(encoding="utf-8").splitlines()]
    assert labels
    out = tmp_path / "successor"
    manifest = attach_rollout_labels(dataset, rollouts, out)
    assert manifest["lineage"]["labels_version"] == LABELS_VERSION and manifest["lineage"]["parent_dataset"] == str(dataset)
    assert manifest["rollout_labels"]["ticks_labelled"] == len(labels) and manifest["rollout_labels"]["episodes_with_labels"] >= 1
    assert sum(manifest["rollout_labels"]["confidence"].values()) == len(labels)
    assert manifest["versions"]["labels"] == [LABELS_VERSION] and manifest["episodes"] == 2
    successors = {record["episode_id"]: record for _, record in read_episodes(out)}
    parents = {record["episode_id"]: record for record in two_episodes["records"]}
    labelled = {(entry["episode_id"], entry["index"]) for entry in labels}
    for episode_id, parent in parents.items():
        child = successors[episode_id]
        assert child["versions"]["labels"] == LABELS_VERSION and child["split"] == parent["split"]
        assert child["provenance"]["lineage"]["rollout_label_ticks"] == sorted(index for (eid, index) in labelled if eid == episode_id)
        for index, (old_tick, new_tick) in enumerate(zip(parent["ticks"], child["ticks"])):
            assert {k: v for k, v in old_tick.items() if k != "labels"} == {k: v for k, v in new_tick.items() if k != "labels"}
            old_main = next(l for l in old_tick["labels"] if l["question_id"] == "q_main")
            new_main = next(l for l in new_tick["labels"] if l["question_id"] == "q_main")
            others_old = [l for l in old_tick["labels"] if l["question_id"] != "q_main"]
            others_new = [l for l in new_tick["labels"] if l["question_id"] != "q_main"]
            assert others_old == others_new
            if (episode_id, index) in labelled:
                assert new_main["source"] == "rollout_v0" and new_main["expert"]["candidate_ids"] == old_main["candidate_ids"] and "event_results" in new_main
            else:
                assert new_main == old_main
    report = validate_dataset([record for _, record in read_episodes(out)])
    assert report["invalid_records"] == 0 and not report.get("errors")
    # 버전이 다른 rollout은 거절한다.
    cost_path = rollouts / "costing.json"
    cost = json.loads(cost_path.read_text(encoding="utf-8"))
    cost["versions"]["harness"] = "h0.0"
    cost_path.write_text(json.dumps(cost), encoding="utf-8")
    with pytest.raises(ConfigMismatch):
        attach_rollout_labels(dataset, rollouts, tmp_path / "rejected")


def test_holding_twin_statistics_pair_the_grasp_and_place_keys_of_one_action_by_zone_and_seed():
    """D1 리뷰 1 I2: 들고 있는 키프레임의 {grasp:held:zone, place:held:zone}는 같은 place-release 사건의 쌍둥이다 — (키프레임, 영역, seed)로
    짝지어 commitment 영역/나머지 영역의 일치·불일치, 불일치 사유·grasp 키 첫 성공 틱·place 키 holding_at_end, 라벨의 섞인 허용 집합을 센다."""
    from robo_jev.data.rollouts import holding_twin_statistics

    keys = {"g1": "grasp:o0:top:zoneL", "p1": "place:o0:release:zoneL", "g2": "grasp:o0:top:zoneF", "p2": "place:o0:release:zoneF"}
    keyframes = [
        {"episode_id": "ep", "t": 10, "holding": "o0", "commitment": "g1", "candidates": list(keys), "keys": keys, "kind": "random"},
        {"episode_id": "ep", "t": 3, "holding": None, "commitment": "x", "candidates": ["x", "y"], "keys": {"x": "grasp:o1:top:zoneL", "y": "push:o2:+x:none"}, "kind": "random"},
    ]

    def result(candidate, seed, outcome, reason=None, *, first_success=None, holding_at_end=None, keyframe="ep@10"):
        return {"outcome": outcome, "reason": reason, "job": {"keyframe": keyframe, "candidate": candidate, "key": keys.get(candidate, "grasp:o1:top:zoneL"), "seed": seed},
                "evidence": {"event_id": "place-release-v0", "first_success_tick": first_success, "holding_at_end": holding_at_end}}

    results = [
        result("g1", 0, "success", first_success=20), result("p1", 0, "success", first_success=21),       # commitment 영역, S/S
        result("g1", 1, "failure", "horizon"), result("p1", 1, "failure", "horizon"),                    # commitment 영역, F/F
        result("g2", 0, "success", first_success=41), result("p2", 0, "failure", "horizon", holding_at_end="o0"),  # 다른 영역, grasp 성공·place 실패
        result("g2", 1, "success", first_success=39), result("p2", 1, "failure", "horizon", holding_at_end=None),
        result("g2", 2, "censored", "candidate_unavailable"), result("p2", 2, "success"),
        result("x", 0, "success", keyframe="ep@3"),  # 들고 있지 않은 키프레임은 세지 않는다
    ]
    labels = [
        {"episode_id": "ep", "t": 10, "label": {"candidate_ids": ["g1", "p1"], "rollout_reason": "performance_allowed_set", "label_confidence": "high"}},
        {"episode_id": "ep", "t": 3, "label": {"candidate_ids": ["x"], "rollout_reason": "commitment_kept", "label_confidence": "high"}},
    ]
    stats = holding_twin_statistics(keyframes, results, labels)
    assert stats["keyframes"] == 1 and stats["candidates_per_keyframe"] == {"4": 1} and stats["keys_by_function"] == {"grasp": 2, "place": 2}
    assert stats["commitment_key_function"] == {"grasp": 1} and stats["rollouts"] == 10 and stats["event_ids"] == {"place-release-v0": 10}
    assert stats["pairs_by_zone"]["committed_zone"] == {"pairs": 2, "ss": 1, "ff": 1, "grasp_success_place_failure": 0, "place_success_grasp_failure": 0, "censored": 0}
    assert stats["pairs_by_zone"]["other_zones"] == {"pairs": 3, "ss": 0, "ff": 0, "grasp_success_place_failure": 2, "place_success_grasp_failure": 0, "censored": 1}
    assert stats["disagreement_reasons"] == {"place:horizon": 2}
    assert stats["grasp_key_first_success_tick_on_disagreement"] == {"n": 2, "p50": 40, "max": 41} and stats["place_key_holding_at_end_on_disagreement"] == 1
    assert stats["labels"] == {"rollout_reason": {"performance_allowed_set": 1}, "confidence": {"high": 1}, "allowed_sets_mixing_keys": 1}
    assert "open_question" in stats


def test_incremental_jsonl_tolerates_a_truncated_last_line_and_compacts_through_a_temp_file(tmp_path, capsys):
    """리뷰 1 M4: 덧붙이는 중에 죽은 실행이 남긴 잘린 마지막 줄은 버리고(부분 결과는 어차피 버린다) 가운데가 깨진 파일은 거절한다;
    다시 쓰기는 임시 파일 + os.replace다."""
    from robo_jev.data.rollouts import _read_jsonl, _write_jsonl

    path = tmp_path / "rollouts.jsonl"
    path.write_text('{"a": 1}\n{"a": 2}\n{"a": 3, "tru', encoding="utf-8")
    assert _read_jsonl(path) == [{"a": 1}, {"a": 2}]
    assert "잘린 마지막 줄" in capsys.readouterr().err
    path.write_text('{"a": 1}\n{"a": 2, "bro\n{"a": 3}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="손상"):
        _read_jsonl(path)
    assert _read_jsonl(tmp_path / "missing.jsonl") == []
    _write_jsonl(path, [{"b": 1}, {"b": 2}])
    assert _read_jsonl(path) == [{"b": 1}, {"b": 2}] and not (tmp_path / "rollouts.jsonl.tmp").exists()


def test_the_overlapped_pool_loop_persists_each_chunk_once_and_resumes_like_a_whole_run(two_episodes, tmp_path):
    """리뷰 1 M5: `workers=2, chunk_episodes=1`은 풀이 앞 묶음을 도는 동안 다음 묶음을 재생하는 `map_async` 경로다 — 표지는 에피소드마다
    한 번, 결과 수 = 파일 줄 수, 표지의 wall_s 합은 배치 벽시계를 넘지 않고(M3: 제출 → get 창), 잘린 실행의 재개는 한 번에 돈 것과 같다."""
    from robo_jev.data.rollouts import KEYFRAMES_INCREMENTAL, PROGRESS_FILE

    dataset = two_episodes["out"]
    common = dict(workers=2, per_episode=2, candidates_per_keyframe=2, chunk_episodes=1, events_override={"seeds": 1})
    whole = run(dataset, limit=None, out=tmp_path / "whole", **common)
    key = lambda result: (result["job"]["keyframe"], result["job"]["candidate"], result["job"]["seed"])  # noqa: E731
    reference = {key(result): (result["outcome"], result["reason"]) for result in whole["results"]}
    assert len(reference) == len(whole["results"]) >= 4
    markers = [json.loads(line) for line in (tmp_path / "whole" / PROGRESS_FILE).read_text(encoding="utf-8").splitlines()]
    assert [marker["episode_id"] for marker in markers] == [record["episode_id"] for record in two_episodes["records"]]
    assert sum(marker["rollouts"] for marker in markers) == len(whole["results"]) == len((tmp_path / "whole" / "rollouts.jsonl").read_text(encoding="utf-8").splitlines())
    cost = json.loads((tmp_path / "whole" / "costing.json").read_text(encoding="utf-8"))
    assert sum(marker["wall_s"] for marker in markers) <= cost["batch_wall_s"] + 1e-6 and cost["keyframes"]["prior_wall_s"] == 0.0 and "batch_wall_s_note" in cost
    assert all(marker["wall_s"] > 0 and marker["replay_s"] > 0 for marker in markers)
    # 잘린 실행 → 재개 (풀 경로): 결과는 한 번에 돈 것과 같고, 표지는 여전히 에피소드마다 하나.
    partial = run(dataset, limit=markers[0]["rollouts"] + 1, out=tmp_path / "resumed", **common)
    assert len(partial["results"]) == markers[0]["rollouts"] + 1
    resumed = run(dataset, limit=None, out=tmp_path / "resumed", resume=True, **common)
    assert resumed["jobs"] == markers[1]["rollouts"]
    assert {key(result): (result["outcome"], result["reason"]) for result in resumed["results"]} == reference
    progress = [json.loads(line) for line in (tmp_path / "resumed" / PROGRESS_FILE).read_text(encoding="utf-8").splitlines()]
    assert [marker["episode_id"] for marker in progress] == [marker["episode_id"] for marker in markers]
    assert len((tmp_path / "resumed" / KEYFRAMES_INCREMENTAL).read_text(encoding="utf-8").splitlines()) == len(whole["keyframes"])
    assert not (tmp_path / "resumed" / "rollouts.jsonl.tmp").exists()
    cost = json.loads((tmp_path / "resumed" / "costing.json").read_text(encoding="utf-8"))
    assert cost["keyframes"]["resumed"] is True and cost["keyframes"]["prior_wall_s"] == pytest.approx(progress[0]["wall_s"], abs=1e-3)
    assert cost["batch_wall_s"] == pytest.approx(cost["keyframes"]["prior_wall_s"] + cost["keyframes"]["this_run_wall_s"], abs=1e-2)
