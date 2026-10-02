"""폐루프 평가 검사 (Task R4 Stage B·C) — 장면 선택, 기계적 정책, run(짧게, 실제 시뮬레이터), 층·사건·그리퍼·정지·안전 지표, 짝지은 구간."""

import copy
import json

import pytest
import yaml
from helpers import REPO

from robo_jev.closed_loop import (
    LAYERS,
    MechanicalPolicy,
    TimedEnvironment,
    TimedExpert,
    build_policy,
    closed_loop_report,
    condition_layer,
    condition_metrics,
    episode_summary,
    failure_cause,
    gripper_event_metrics,
    load_closed_loop_config,
    paired_success,
    per_record_rows,
    print_report,
    quotas,
    run_condition,
    select_conditions,
    select_seeds,
)
from robo_jev.contracts import validate_record
from robo_jev.data.robot_episodes import config_paths, seed_schedule
from robo_jev.sim.controller import resolve_config_path

CONFIG_PATH = REPO / "configs/eval/r4-closed-loop.yaml"


@pytest.fixture(scope="module")
def config():
    return load_closed_loop_config(CONFIG_PATH)


# --------------------------------------------------------------------------
# 장면(seed) 선택
# --------------------------------------------------------------------------


def test_quotas_follow_the_profile_weights_with_the_largest_remainder():
    assert quotas([20, 40, 40], 100) == [20, 40, 40]
    assert quotas([20, 40, 40], 26) == [5, 11, 10]
    assert quotas([1, 1, 1], 4) == [2, 1, 1] and sum(quotas([3, 7], 11)) == 11


def test_the_config_names_the_generator_and_refuses_the_sealed_split(tmp_path, config):
    assert config["conditions"] == {"dev": {"split": "dev", "count": 100}, "ood_dev": {"split": "ood_dev", "count": 26}}
    assert config["seeds"]["base"] == 900100 and config["generator"]["version"] == "r1-robot-v0.2"
    raw = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    raw["conditions"]["sealed"] = {"split": "ood_test", "count": 1}
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(raw, allow_unicode=True), encoding="utf-8")
    with pytest.raises(ValueError, match="ood_test"):
        load_closed_loop_config(path)


def test_selected_seeds_are_new_land_in_the_asked_split_and_follow_the_profile_quota(config):
    generator = config["generator"]
    sim = yaml.safe_load(resolve_config_path(config_paths(generator)["sim_config"]).read_text(encoding="utf-8"))
    block = select_seeds(generator, sim, split="dev", count=10, base=config["seeds"]["base"], per_profile_max=400)
    assert block["quota"] == {"E0": 2, "E1": 4, "E2": 4} and block["by_profile"] == block["quota"] and block["shortfall"] == {}
    assert len(block["seeds"]) == 10 and all(item["split"] == "dev" and item["holdout"] == [] for item in block["seeds"])
    r1 = {(profile, seed) for profile, seed in seed_schedule(generator, 400)}
    assert not {(item["profile"], item["seed"]) for item in block["seeds"]} & r1
    assert all(item["seed"] >= 900100 for item in block["seeds"])
    assert "ood_test" not in json.dumps(block["seeds"])  # 봉인 split의 seed는 적히지 않는다 (세기만 한다)
    again = select_seeds(generator, sim, split="dev", count=10, base=config["seeds"]["base"], per_profile_max=400)
    assert again == block  # 결정적
    ood = select_seeds(generator, sim, split="ood_dev", count=4, base=config["seeds"]["base"], per_profile_max=800)
    assert all(item["split"] == "ood_dev" and item["holdout"] for item in ood["seeds"])


def test_select_conditions_counts_the_families_r1_already_used(config):
    small = copy.deepcopy(config)
    small["conditions"] = {"dev": {"split": "dev", "count": 5}}
    small["scan"]["per_profile_max"] = 200
    out = select_conditions(small, manifest=REPO / "artifacts/datasets/r1-robot/r1/manifest.json")
    block = out["conditions"]["dev"]
    assert len(block["seeds"]) == 5 and 0 <= block["families_in_r1"] <= len(block["families"])


# --------------------------------------------------------------------------
# 기계적 정책
# --------------------------------------------------------------------------


def _tick(commitment: str | None) -> dict:
    candidates = [{"id": "c1", "key": "grasp:o1:top:zoneL"}, {"id": "c2", "key": "observe"}, {"id": "c3", "key": "hold"}]
    return {"t": 0, "request": {"state": {}, "exec_history": "none", "commitment": ({"action_ref": commitment, "key": "grasp:o1:top:zoneL", "phase": "approach", "held_ticks": 1, "last_switch_tick": 0} if commitment else None), "candidates": {"q_main": candidates}}, "harness": {}}


def test_the_mechanical_policy_repeats_the_commitment_or_observes_and_says_only_the_harness_defaults():
    policy = MechanicalPolicy()
    committed = policy.act(_tick("c1"), {"action_ref": "c3"}, {"ignored": True})
    assert committed["q_main"] == {"c1": 1.0, "c2": 0.0, "c3": 0.0}
    idle = policy.act(_tick(None))
    assert idle["q_main"] == {"c1": 0.0, "c2": 1.0, "c3": 0.0}
    assert {k: v for k, v in idle.items() if k != "q_main"} == {"q_done": 0.0, "q_instr": 1.0, "q_observe": 0.0, "q_retry": 0.0, "q_stop": 0.0, "q_gripper": {}, "q_path": {}, "q_speed": {}, "q_force": {}}


# --------------------------------------------------------------------------
# run — 실제 시뮬레이터에서 짧게 (규칙 판정기·expert·기계적)
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def short_runs(tmp_path_factory, config):
    out = tmp_path_factory.mktemp("r4")
    schedule = [("E0", 900100), ("E1", 910100)]
    runs = {}
    tags = {"expert": "r4", "rule": "r4", "mechanical": "r5"}  # 기본 꼬리는 r4; R5의 run은 `id_tag`로 다른 라운드 표지를 단다
    for kind in ("expert", "rule", "mechanical"):
        bundle = build_policy(kind, generator=config["generator"])
        runs[kind] = run_condition(bundle, schedule, config=config, out=out / kind, condition="dev", label=kind, max_ticks=24, **({"id_tag": "r5"} if kind == "mechanical" else {}))
    return {"out": out, "runs": runs, "schedule": schedule, "tags": tags}


def test_a_short_run_writes_records_a_manifest_and_a_timing_sidecar(short_runs):
    for kind, run in short_runs["runs"].items():
        assert run["summary"]["episodes"] == 2 and run["condition"] == "dev" and run["label"] == kind
        assert [row["key"] for row in run["episodes"]] == ["E0:900100", "E1:910100"]
        manifest = json.loads((short_runs["out"] / kind / "manifest.json").read_text(encoding="utf-8"))
        assert manifest["episodes"] == 2 and manifest["closed_loop"]["policy"]["kind"] == kind
        rows = [json.loads(line) for line in (short_runs["out"] / kind / "timing.jsonl").read_text(encoding="utf-8").splitlines()]
        assert len(rows) == run["summary"]["ticks"] and all(row["obs_to_command_ms"] is not None and row["obs_to_command_ms"] >= 0 for row in rows)
        assert run["latency"]["obs_to_command_ms"]["n"] == run["summary"]["ticks"]
        assert manifest["closed_loop"]["id_tag"] == short_runs["tags"][kind]
        for episode in run["episodes"]:
            assert episode["layer"] in LAYERS and episode["ticks"] <= 24 and episode["episode_id"].endswith(f"-{short_runs['tags'][kind]}-{kind}")
    expert_rows = [json.loads(line) for line in (short_runs["out"] / "expert" / "timing.jsonl").read_text(encoding="utf-8").splitlines()]
    rule_rows = [json.loads(line) for line in (short_runs["out"] / "rule" / "timing.jsonl").read_text(encoding="utf-8").splitlines()]
    assert all(row["reference_ms"] == 0.0 for row in expert_rows)  # 정책이 expert면 참조 답은 따로 계산되지 않는다
    assert all(row["reference_ms"] > 0.0 for row in rule_rows)  # 대역 정책이면 참조(전문가 act + labels) 시간이 있다


def test_the_layers_are_read_from_instruction_changes_and_the_mechanical_policy_never_acts(short_runs):
    from robo_jev.data.robot_episodes import read_episodes

    records = {kind: {record["provenance"]["profile"]: record for _, record in read_episodes(short_runs["out"] / kind)} for kind in short_runs["runs"]}
    for kind in records:
        assert condition_layer(records[kind]["E0"]) == "no_instruction_change"
        validate_record(records[kind]["E0"])
    mechanical = records["mechanical"]["E0"]
    summary = episode_summary(mechanical)
    assert summary["done"] is False and summary["decisions"]["acted"] == 0 and summary["decisions"]["expert_joint"] > 0
    assert failure_cause(mechanical) == "semantic_main"  # 참조는 결합 후보를 허용하는데 정책은 끝까지 게이트·hold
    assert all(label["source"] == "expert_v0" for tick in mechanical["ticks"] for label in tick["labels"])
    expert = records["expert"]["E0"]
    assert episode_summary(expert)["decisions"]["wrong_action"] == 0  # 자기 참조와 같은 결정


def test_condition_metrics_and_the_paired_bootstrap_read_the_short_runs(short_runs):
    from robo_jev.data.robot_episodes import read_episodes

    tables = {}
    rows = {}
    for kind in short_runs["runs"]:
        records = [record for _, record in read_episodes(short_runs["out"] / kind)]
        rows[kind] = [episode_summary(record) for record in records]
        tables[kind] = condition_metrics(records, rows[kind])
    for kind, table in tables.items():
        assert table["episodes"] == 2 and set(table["layers"]) <= set(LAYERS) and table["success_ci"] is not None
        assert table["reaction_delay"]["adopted"]["horizon_ticks"] == 30 and "switch_rate" in table["stability"]["adopted"]
        assert table["stop_timing"] is not None and "unsafe_action_rate" in table["selective"] and "missing_rate" in table["gripper_events"]
        assert table["controller"]["commanded_ticks"] > 0 and table["cost"]["ticks"] == short_runs["runs"][kind]["summary"]["ticks"]
    pair = paired_success(rows["expert"], rows["mechanical"])
    assert pair["seeds"] == 2 and pair["a"] >= pair["b"] and pair["margin"] == pytest.approx(pair["a"] - pair["b"])
    assert paired_success(rows["expert"], []) is None


def test_closed_loop_report_joins_runs_and_prints_tables(short_runs, tmp_path, capsys):
    paths = []
    for kind, run in short_runs["runs"].items():
        path = tmp_path / f"run-{kind}.json"
        path.write_text(json.dumps({"label": kind, "policy": {"kind": kind, "name": kind}, "conditions": {"dev": run}}, ensure_ascii=False), encoding="utf-8")
        paths.append(path)
    report = closed_loop_report(paths)
    assert set(report["tables"]["dev"]) == {"expert", "rule", "mechanical"} and report["conditions"] == ["dev"]
    assert "expert - rule" in report["paired"]["dev"] and "expert - mechanical" in report["paired"]["dev"]
    print_report(report)
    printed = capsys.readouterr().out
    assert "### dev" in printed and "| expert |" in printed and "| expert - mechanical |" in printed


# --------------------------------------------------------------------------
# 지표 — 손으로 만든 레코드 위에서 정의를 고정한다
# --------------------------------------------------------------------------


def _record(ticks: list[dict], *, done: bool = True, profile: str = "E0", seed: int = 1, inside: bool | None = None) -> dict:
    out = {"schema_version": "stream-v0", "episode_id": f"ep-{profile}-{seed}", "ticks": [], "provenance": {"profile": profile, "seed": seed, "timing": {"wall_s": 1.0},
           "outcome": {"done": done, "done_tick": (len(ticks) - 1) if done else None, "first_done_tick": None, "sim_ms": 100 * len(ticks), "terminated": "done_tail" if done else "max_ms", "target_inside_zone": done if inside is None else inside}}}
    for index, spec in enumerate(ticks):
        candidates = [{"id": "c1", "key": "grasp:o1:top:zoneL"}, {"id": "c2", "key": "grasp:o2:top:zoneL"}, {"id": "c3", "key": "observe"}, {"id": "c4", "key": "hold"}]
        state = {"goal": {"version": spec.get("version", 1), "forbidden_contact": ["o9"]}, "events": spec.get("events", []), "robot": {}, "objects": []}
        labels = [{"question_id": "q_main", "kind": "valid_set", "candidate_ids": spec.get("allowed", ["c1"])},
                  {"question_id": "q_stop", "kind": "single", "answer": spec.get("stop_label", False)},
                  {"question_id": "q_gripper", "kind": "valid_set", "candidate_ids": spec.get("gripper_label", ["open"])}]
        out["ticks"].append({"t": index, "sim_ms": 100 * index, "request": {"state": state, "commitment": None, "exec_history": "none", "candidates": {"q_main": candidates}},
                             "adopted": {"main": spec.get("adopted", "c1"), "gripper": spec.get("gripper", "open"), "stop": spec.get("stop", False), "switch": False},
                             "model_output": {"q_main": spec.get("model", {"c1": 1.0}), "q_stop": spec.get("p_stop", 0.0)}, "ack": {"applied": True, "rejected": spec.get("rejected", False), "reason": spec.get("reason")},
                             "usage": {"gate": None, "records": {}}, "labels": labels})
    return out


def test_failure_cause_separates_a_wrong_decision_from_a_failed_execution():
    same = _record([{"adopted": "c1"}] * 5, done=False)
    assert failure_cause(same) == "geometric"
    wrong = _record([{"adopted": "c1"}, {"adopted": "c2"}, {"adopted": "c1"}], done=False)  # 다른 대상을 골랐다
    assert failure_cause(wrong) == "semantic_main"
    idle = _record([{"adopted": "c3"}] * 4, done=False)  # 참조는 파지를 허용하는데 관측만
    assert failure_cause(idle) == "semantic_main"
    hover = _record([{"adopted": "c1"}] * 2 + [{"adopted": "c1", "gripper_label": ["closed"], "gripper": "open"}] * 3, done=False)  # 파지점에서 닫지 않았다
    assert failure_cause(hover) == "semantic_aux"
    flicker = _record([{"adopted": "c1"}] * 2 + [{"adopted": "c1", "gripper_label": ["closed"], "gripper": "open"}] * 2 + [{"adopted": "c1", "gripper_label": ["closed"], "gripper": "closed"}], done=False)
    assert failure_cause(flicker) == "geometric"  # 2틱의 흔들림은 세지 않는다
    gate_allowed = _record([{"adopted": "c3", "allowed": ["c3"]}] * 4, done=False)  # 참조도 관측
    assert failure_cause(gate_allowed) == "geometric"
    assert failure_cause(_record([{"adopted": "c1"}], done=True)) is None
    assert condition_layer(_record([{"version": 1}, {"version": 2}])) == "instruction_changes"
    assert condition_layer(_record([{"version": 1}, {"version": 1}])) == "no_instruction_change"


def test_gripper_events_count_missing_duplicate_and_timing_error():
    ticks = [{"gripper_label": ["open"], "gripper": "open"}] * 3 + [{"gripper_label": ["closed"], "gripper": "open"}] * 2 + [{"gripper_label": ["closed"], "gripper": "closed"}] * 3
    ticks += [{"gripper_label": ["closed"], "gripper": "open"}] + [{"gripper_label": ["closed"], "gripper": "closed"}] * 22  # 참조 없는 정책 전환 둘(열고 다시 닫음), 그 뒤 20틱 조용
    ticks += [{"gripper_label": ["open"], "gripper": "closed"}] * 12  # 참조는 열라는데 정책은 끝까지 닫힘 → 누락 (앞의 열림은 지평선 밖)
    metrics = gripper_event_metrics([_record(ticks)])
    assert metrics["reference_transitions"] == 2 and metrics["matched"] == 1 and metrics["missing"] == 1
    assert metrics["duplicate"] == 2 and metrics["timing_error_ticks"]["median_abs"] == 2 and metrics["timing_error_ticks"]["mean_signed"] == 2.0
    early = [{"gripper_label": ["open"], "gripper": "open"}] * 6 + [{"gripper_label": ["open"], "gripper": "closed"}] * 3 + [{"gripper_label": ["closed"], "gripper": "closed"}] * 3
    assert gripper_event_metrics([_record(early)])["timing_error_ticks"]["mean_signed"] == -3.0  # 참조보다 3틱 먼저 닫아도 지평선 안이면 같은 사건이다


def test_per_record_rows_read_adopted_model_and_stop_answers():
    record = _record([{"adopted": "c1", "model": {"c1": 0.2, "c2": 0.8}, "p_stop": 0.7}, {"adopted": "c2", "model": {"c1": 0.9, "c2": 0.1}, "p_stop": {"true": 0.1, "false": 0.9}}])
    rows = per_record_rows([record])
    assert [row["predicted"] for row in rows["adopted"]] == ["c1", "c2"]
    assert [row["predicted"] for row in rows["model"]] == ["c2", "c1"]
    assert [row["predicted"] for row in rows["stop"]] == ["true", "false"]


def test_condition_metrics_on_hand_made_records_report_stop_reflex_rejections_and_layers():
    stop_ticks = [{"stop_label": False}] * 2 + [{"stop_label": True, "p_stop": 0.9, "stop": True}] + [{"stop_label": True, "events": [{"kind": "reflex_stop"}], "stop": True, "p_stop": 0.0}] + [{"stop_label": False, "rejected": True, "reason": "transition_collision"}]
    records = [_record(stop_ticks, done=True, seed=1), _record([{"adopted": "c2"}] * 3, done=False, seed=2, profile="E1"), _record([{"version": 2, "adopted": "c1"}] * 3, done=True, seed=3, profile="E2")]
    for record in records:
        record["ticks"][0]["request"]["state"]["goal"]["version"] = 1
    table = condition_metrics(records)
    assert table["episodes"] == 3 and table["done"] == 2 and table["failure_causes"] == {"semantic_main": 1}
    assert table["aux_agreement"]["q_gripper"]["n"] > 0 and table["aux_agreement"]["q_gripper"]["rate"] == 1.0
    assert table["layers"]["no_instruction_change"]["episodes"] == 2 and table["layers"]["instruction_changes"]["episodes"] == 1
    assert table["stop_vs_reflex"]["stop_ticks_by_q_stop"] == 1 and table["stop_vs_reflex"]["stop_ticks_by_reflex"] == 1 and table["stop_vs_reflex"]["reflex_ticks_where_model_had_said_stop"] == 1
    assert table["stop_timing"]["onsets"] == 1 and table["stop_timing"]["reacted"] == 1 and table["stop_timing"]["false_alarm_rate"] == 0.0
    assert table["controller"]["rejected_ticks"] == 1 and table["controller"]["transition_collisions"] == 1 and table["controller"]["rejection_rate"] == pytest.approx(1 / 11)
    assert table["by_profile"]["E1"]["success_rate"] == 0.0 and table["completion_ticks"]["median"] in (2, 4)


def test_timed_proxies_measure_without_changing_behaviour(config):
    from robo_jev.sim.environment import Environment
    from robo_jev.sim.expert import Expert, load_expert_config

    paths = config_paths(config["generator"])
    expert = TimedExpert(Expert(load_expert_config(paths["expert_config"])))
    assert expert.version == expert._expert.version and expert.label_source == "expert_v0"
    env = TimedEnvironment(Environment(config_path=paths["sim_config"], profile="E0"))
    try:
        observation = env.reset(900100)
        assert env.profile == "E0" and env.max_ms == 45000 and env.intervals_ms == []
        before = env._ready_at
        env.step(None)
        assert env.intervals_ms == [] and env._ready_at is not None and env._ready_at > before  # 명령 없는 스텝은 관측→명령 구간이 아니고, 관측 시각은 새로 찍힌다
        assert observation["tick"] == 0
    finally:
        env.close()


def test_offline_gripper_transitions_split_initiate_from_settled_ticks():
    """오프라인 `q_gripper` 예측을 전환 틱(닫아야 하는데 실행 그리퍼는 아직 열림)·정착 틱(이미 닫힘)·open 틱으로 나눠 채점한다."""
    from robo_jev.closed_loop import offline_gripper_transitions

    ticks = [{"gripper_label": ["open"]}] * 2 + [{"gripper_label": ["closed"], "exec_gripper": "open"}] * 2 + [{"gripper_label": ["closed"], "exec_gripper": "closed"}] * 3 + [{"gripper_label": ["open", "closed"]}]
    record = _record(ticks, seed=7)
    for tick, spec in zip(record["ticks"], ticks):
        tick["request"]["state"]["exec"] = {"gripper": spec.get("exec_gripper", "open")}
    predicted = ["open", "open", "open", "closed", "closed", "closed", "closed", "open"]  # 전환 틱 둘 중 하나만 맞힌다; 마지막(두 값 라벨·실행 open)은 window
    rows = [{"record_id": record["episode_id"], "tick": index, "question": "q_gripper", "predicted": value, "correct": None} for index, value in enumerate(predicted)]
    report = {"evaluation": {"splits": {"x": {"model": {"q_gripper": {"accuracy": 0.9, "per_record": rows}}}}}}
    out = offline_gripper_transitions(report, [record], split_name="x")
    assert out["initiate"] == {"n": 2, "correct": 1, "predicted_closed": 1, "accuracy": 0.5}
    assert out["settled"] == {"n": 3, "correct": 3, "predicted_closed": 3, "accuracy": 1.0}
    assert out["open"] == {"n": 2, "correct": 2, "predicted_closed": 0, "accuracy": 1.0}
    assert out["window"] == {"n": 1, "correct": 0, "predicted_closed": 0, "accuracy": None, "predicted_closed_rate": 0.0}  # 헤드라인(1/271)이 서는 칸
    assert out["episodes_with_initiate_ticks"] == 1 and out["episodes_where_every_initiate_tick_is_wrong"] == 0 and out["whole_question_accuracy"] == 0.9


def test_completed_without_a_single_close_is_listed():
    done_by_gate = _record([{"adopted": "c1", "gripper": "open"}] * 3, done=True, seed=11)
    grasped = _record([{"adopted": "c1", "gripper": "open"}, {"adopted": "c1", "gripper": "closed"}, {"adopted": "c1", "gripper": "closed"}], done=True, seed=12)
    table = condition_metrics([done_by_gate, grasped])
    assert table["completed_without_close"] == ["ep-E0-11"]


def test_a_policy_declared_done_with_the_target_outside_its_zone_is_a_false_done_and_the_strict_column_excludes_it():
    """`done`은 정책 자신의 `q_done`이 꼬리 동안 든 것이라 거짓으로 들 수 있다 — `done ∧ target_inside_zone`이 관측된 완료다 (리뷰 1 I1)."""
    real = _record([{"adopted": "c1"}] * 3, done=True, seed=21)
    false_done = _record([{"adopted": "c1"}] * 3, done=True, seed=22, inside=False)
    failed = _record([{"adopted": "c1"}] * 3, done=False, seed=23)
    rows = [episode_summary(record) for record in (real, false_done, failed)]
    assert [row["done_inside"] for row in rows] == [True, False, False]
    table = condition_metrics([real, false_done, failed], rows)
    assert table["done"] == 2 and table["strict_done"] == 1 and table["false_done"] == ["ep-E0-22"]
    assert table["strict_success_rate"] == pytest.approx(1 / 3) and table["strict_success_ci"] is not None
    other = [episode_summary(_record([{"adopted": "c1"}] * 3, done=True, seed=seed)) for seed in (21, 22, 23)]
    assert paired_success(rows, other)["margin"] == pytest.approx(2 / 3 - 1.0)
    assert paired_success(rows, other, strict=True)["margin"] == pytest.approx(1 / 3 - 1.0)


def test_disturbances_are_counted_from_the_applied_log_and_object_moved_events():
    record = _record([{"events": [{"kind": "object_moved", "object": "o1", "displacement_mm": 29}]}, {}, {"events": [{"kind": "object_moved", "object": "o0", "displacement_mm": 67}]}])
    record["evidence"] = {"disturbance_log": [{"sim_ms": 100, "object": "o1"}, {"sim_ms": 300, "object": "o0"}, {"sim_ms": 800, "object": "o1"}], "scene_plan": {"instructions": [{"version": 1}], "disturbances": [{}, {}, {}, {}]}}
    summary = episode_summary(record)
    assert summary["disturbances_applied"] == 3 and summary["object_moved_events"] == 2 and summary["scheduled"] == {"instruction_changes": 0, "disturbances": 4}
    events = condition_metrics([record], [summary])["events"]
    assert events == {"goal_changes": 0, "disturbances_applied": 3, "disturbances_scheduled": 4, "object_moved_events": 2, "reflex_ticks": 0}


def test_the_report_merges_run_files_that_share_a_label_across_conditions_and_refuses_an_overlap(short_runs, tmp_path):
    """Task R5 D2: R4의 `expert/rule/mechanical/s18`(dev·ood_dev)과 R5의 같은 정책(dev_new)은 run 파일이 다르다 — 같은 이름표의 파일은
    조건을 합쳐 한 정책으로 읽고, 같은 조건이 두 파일에 있으면 거절한다."""
    runs = short_runs["runs"]
    first = tmp_path / "expert-dev.json"
    second = tmp_path / "expert-dev_new.json"
    first.write_text(json.dumps({"label": "expert", "policy": {"kind": "expert"}, "conditions": {"dev": runs["expert"]}}, default=str), encoding="utf-8")
    moved = {**runs["expert"], "condition": "dev_new"}
    second.write_text(json.dumps({"label": "expert", "policy": {"kind": "expert"}, "conditions": {"dev_new": moved}}, default=str), encoding="utf-8")
    report = closed_loop_report([first, second])
    assert report["conditions"] == ["dev", "dev_new"] and set(report["tables"]["dev"]) == {"expert"} and set(report["tables"]["dev_new"]) == {"expert"}
    assert report["runs"]["expert"]["paths"] == [str(first), str(second)]
    clash = tmp_path / "expert-dev-again.json"
    clash.write_text(first.read_text(encoding="utf-8"), encoding="utf-8")
    with pytest.raises(ValueError, match="겹친다"):
        closed_loop_report([first, clash])


def test_family_overlap_counts_the_scene_groups_and_seeds_a_condition_shares_with_training_material(tmp_path):
    """Task R6 D1: "본 적 없는 seed·대부분 학습한 계열" — 조건의 장면 계열(origin group)이 학습 재료의 계열과 얼마나 겹치는가."""
    from robo_jev.closed_loop import family_overlap

    seeds = {"conditions": {"dev_new2": {"seeds": [
        {"profile": "E1", "seed": 1, "origin_group": "robot/E1/family-a/goal-zoneL", "family": "family-a"},
        {"profile": "E1", "seed": 2, "origin_group": "robot/E1/family-a/goal-zoneL", "family": "family-a"},
        {"profile": "E2", "seed": 3, "origin_group": "robot/E2/family-b/goal-zoneR", "family": "family-b"},
    ]}}}
    manifest = tmp_path / "m.json"
    manifest.write_text(json.dumps({"files": {
        "episodes/x/streams.jsonl": {"origin_group": "robot/E1/family-a/goal-zoneL"},
        "episodes/y/streams.jsonl": {"origin_group": "robot/E2/family-b/goal-zoneL"},  # 같은 구조 계열, 다른 목표 영역
        "episodes/z/streams.jsonl": {"origin_group": "robot/E2/family-b/goal-zoneR", "split": "dev"},  # train이 아닌 편은 재료가 아니다
        "contrast/records.jsonl": {"records": 3},
    }}), encoding="utf-8")
    out = family_overlap(seeds, {"dagger": manifest})
    block = out["dev_new2"]
    assert block["origin_groups"] == 2 and block["seeds"] == 3
    assert block["by_material"]["dagger"] == {"origin_groups": 1, "seeds": 2, "families": 2, "seeds_by_family": 3}
    assert block["any_material"] == {"origin_groups": 1, "seeds": 2, "families": 2, "seeds_by_family": 3}


def test_the_report_merges_two_conditions_into_one_table_and_pairs_false_dones(short_runs, tmp_path, config):
    """Task R6 D2: ood_dev 100 = R4의 ood_dev 26 seed + 새 ood_dev 74 seed — 두 조건의 편을 한 표로 합치고(같은 seed가 둘에 있으면 거절),
    짝지은 비교에 **거짓 done**(`done ∧ ¬target_inside_zone`, seed마다 0/1)의 차를 더한다. 조건 디렉터리는 manifest가 있어야 읽힌다
    (Task R7 A3: 보고서는 manifest의 split으로 먼저 거른다)."""
    from robo_jev.data.robot_episodes import build_manifest, read_episodes, write_episode

    paths = []
    for kind in ("expert", "rule"):
        records = [record for _, record in read_episodes(short_runs["out"] / kind)]
        parts = {}
        for name, record in zip(("part_a", "part_b"), records):
            directory = tmp_path / kind / name
            if kind == "expert" and name == "part_a":  # 한 편을 거짓 done으로 — 짝지은 거짓 done 차가 0이 아닌 값을 싣는지 본다
                record = copy.deepcopy(record)
                record["provenance"]["outcome"].update({"done": True, "target_inside_zone": False})
            write_episode(record, directory)
            (directory / "manifest.json").write_text(json.dumps(build_manifest(directory, config["generator"]), default=str), encoding="utf-8")
            parts[name] = {**short_runs["runs"][kind], "condition": name, "episodes_dir": str(directory)}
        path = tmp_path / f"run-{kind}.json"
        path.write_text(json.dumps({"label": kind, "policy": {"kind": kind}, "conditions": parts}, default=str), encoding="utf-8")
        paths.append(path)
    report = closed_loop_report(paths, merge={"both": ["part_a", "part_b"]}, only=["both"])
    assert report["conditions"] == ["both"] and set(report["tables"]["both"]) == {"expert", "rule"}
    assert report["tables"]["both"]["expert"]["episodes"] == 2 and report["tables"]["both"]["expert"]["merged_from"] == ["part_a", "part_b"]
    pair = report["paired"]["both"]["expert - rule (false done)"]
    rule_false = report["tables"]["both"]["rule"]["false_done"]
    assert pair["seeds"] == 2 and pair["a_count"] == 1 and pair["b_count"] == len(rule_false)
    assert pair["a"] == pytest.approx(0.5) and pair["margin"] == pytest.approx(0.5 - len(rule_false) / 2)
    # seed 단위 짝지은 지표와 판정의 견고성 블록 (리뷰 1 I-2·I-3) — 청한 쌍에만, 합친 조건에도
    seeded = closed_loop_report(paths, merge={"both": ["part_a", "part_b"]}, only=["both"], seed_pairs=[("expert", "rule")], alternative_seeds=range(1, 6))
    block = seeded["seed_pairs"]["both"]["expert - rule"]
    from robo_jev.closed_loop import SEED_METRICS

    assert set(block["metrics"]) == set(SEED_METRICS) and block["metrics"]["aux_failure"]["seeds"] == 2
    assert set(block["robustness"]) == {"strict", "false_done"} and block["robustness"]["false_done"]["discordant"]["a_only"] >= 1
    assert block["robustness"]["strict"]["alternative_seeds"]["count"] == 5
    assert report["seed_pairs"] == {}  # 청하지 않으면 비어 있다 (위의 `report`는 seed_pairs 없이 만들었다)
    # 같은 seed가 두 조건에 있으면 합칠 수 없다
    clash = tmp_path / "clash.json"
    run = short_runs["runs"]["expert"]
    clash.write_text(json.dumps({"label": "expert", "policy": {"kind": "expert"}, "conditions": {"x": run, "y": {**run, "condition": "y"}}}, default=str), encoding="utf-8")
    with pytest.raises(ValueError, match="seed"):
        closed_loop_report([clash], merge={"xy": ["x", "y"]})


def test_paired_false_done_is_the_difference_of_seed_level_false_done_indicators():
    from robo_jev.closed_loop import paired_false_done

    def row(key, done, inside):
        return {"key": key, "done": done, "done_inside": done and inside}

    a = [row("E1:1", True, False), row("E1:2", True, True), row("E1:3", False, False)]
    b = [row("E1:1", True, True), row("E1:2", True, True), row("E1:3", False, False)]
    out = paired_false_done(a, b)
    assert out["seeds"] == 3 and out["a"] == pytest.approx(1 / 3) and out["b"] == 0.0 and out["margin"] == pytest.approx(1 / 3)
    assert paired_false_done(a, []) is None


# --------------------------------------------------------------------------
# seed 단위 짝지은 지표와 판정의 견고성 (Task R6 수정 라운드 1, 리뷰 1 I-2·I-3)
# --------------------------------------------------------------------------


def test_paired_seed_ratio_pools_numerators_and_denominators_over_the_same_resampled_seeds():
    from robo_jev.closed_loop import paired_seed_ratio

    a = {"E1:1": (1.0, 1.0), "E1:2": (0.0, 1.0), "E1:3": (2.0, 4.0)}
    b = {"E1:1": (0.0, 1.0), "E1:2": (0.0, 1.0), "E1:3": (1.0, 4.0)}
    out = paired_seed_ratio(a, b, resamples=300)
    assert out["seeds"] == 3 and out["a"] == pytest.approx(3 / 6) and out["b"] == pytest.approx(1 / 6)
    assert out["difference"] == pytest.approx(2 / 6) and out["ci"][0] >= 0.0 and out["includes_zero"] is (out["ci"][0] <= 0.0 <= out["ci"][1])
    # 분모가 0인 재표집은 건너뛰고 그 수를 적는다 (정지 사건이 없는 seed만 뽑힌 경우)
    sparse = paired_seed_ratio({"E1:1": (1.0, 1.0), "E1:2": (0.0, 0.0)}, {"E1:1": (0.0, 1.0), "E1:2": (0.0, 0.0)}, resamples=300)
    assert sparse["difference"] == 1.0 and sparse["skipped_resamples"] > 0
    assert paired_seed_ratio(a, {}, resamples=10) is None


def test_seed_metrics_read_the_gripper_streak_the_wrong_action_and_the_stop_catch_of_one_record():
    from robo_jev.closed_loop import seed_metrics

    ticks = [{"gripper_label": ["closed"], "gripper": "open"} for _ in range(3)] + [
        {"adopted": "c2", "allowed": ["c1"]},  # 다른 대상을 채택 — 오행동 틱
        {"stop_label": True, "p_stop": 0.9, "stop": True},  # 정지가 필요해진 틱에 q_stop이 참 — 잡았다
    ]
    metrics = seed_metrics(_record(ticks, done=False))
    assert metrics["gripper_streak"] == (1.0, 1.0) and metrics["wrong_action"][0] == 1.0 and metrics["wrong_action"][1] == 5.0
    assert metrics["q_stop_caught"] == (1.0, 1.0) and metrics["aux_failure"][1] == 1.0 and metrics["gripper_duplicates"][1] == 1.0


def test_seed_metrics_count_missing_gripper_transitions_forbidden_contacts_and_reflex_ticks():
    """R10이 더한 셋: 누락된 참조 그리퍼 전환 / 참조 전환, 편당 금지 물체(goal.forbidden_contact)와의 접촉 시작 수(다른 물체와의 접촉은 세지 않는다),
    편당 반사 사건이 실린 틱 수."""
    from robo_jev.closed_loop import SEED_METRICS, seed_metrics

    ticks = [{"gripper_label": ["open"], "gripper": "open"}] * 2 + [{"gripper_label": ["closed"], "gripper": "open"}] * 12  # 닫으라는 전환 하나 — 끝까지 열림 → 누락
    closed = {"gripper_label": ["closed"], "gripper": "open"}  # 참조는 그대로 닫힘 — 둘째 전환을 만들지 않는다
    ticks += [{**closed, "events": [{"kind": "contact_onset", "object": "o9"}]}, {**closed, "events": [{"kind": "contact_onset", "object": "o2"}]}]
    ticks += [{**closed, "events": [{"kind": "reflex_stop"}], "stop": True}, {**closed, "events": [{"kind": "reflex_slip"}, {"kind": "contact_onset", "object": "o9"}]}]
    metrics = seed_metrics(_record(ticks, done=False))
    assert metrics["gripper_missing"] == (1.0, 1.0) and metrics["forbidden_contacts"] == (2.0, 1.0) and metrics["reflex_ticks"] == (2.0, 1.0)
    quiet = seed_metrics(_record([{"gripper_label": ["open"], "gripper": "open"}] * 3, done=True))
    assert quiet["gripper_missing"] == (0.0, 0.0) and quiet["forbidden_contacts"] == (0.0, 1.0) and quiet["reflex_ticks"] == (0.0, 1.0)
    assert set(metrics) == set(SEED_METRICS) and SEED_METRICS[-3:] == ("gripper_missing", "forbidden_contacts", "reflex_ticks")


def test_paired_robustness_counts_discordant_seeds_the_exact_mcnemar_p_and_other_bootstrap_seeds():
    from robo_jev.closed_loop import paired_robustness

    def row(key, strict):
        return {"key": key, "done": strict, "done_inside": strict}

    a = [row(f"E1:{i}", i < 5) for i in range(10)]  # a만 성공 3, b만 성공 1, 둘 다 2
    b = [row(f"E1:{i}", i in (0, 1, 7)) for i in range(10)]
    out = paired_robustness(a, b, metric="strict", alternative_seeds=range(1, 21))
    assert out["discordant"] == {"a_only": 3, "b_only": 1} and out["mcnemar_exact_p"] == pytest.approx(0.625)
    assert out["registered"]["margin"] == pytest.approx(0.2) and out["alternative_seeds"]["count"] == 20
    alternative = out["alternative_seeds"]
    assert 0 <= alternative["intervals_containing_zero"] <= 20 and alternative["lower_bound"]["min"] <= alternative["lower_bound"]["max"]
    assert alternative["lower_bound_at_or_below_zero"] + alternative["lower_bound_above_zero"] == 20
    with pytest.raises(ValueError, match="metric"):
        paired_robustness(a, b, metric="nonsense")


# --------------------------------------------------------------------------
# Task R7 사전 등록 — 등록한 판정 규칙을 보고서에 그대로 적용한다 (configs/eval/r7-registration.yaml)
# --------------------------------------------------------------------------

REGISTRATION_PATH = REPO / "configs/eval/r7-registration.yaml"


def _interval(low, high):
    return {"margin": (low + high) / 2.0, "margin_ci": [low, high], "margin_includes_zero": low <= 0.0 <= high, "seeds": 200, "a": 0.5, "b": 0.5}


def _ratio(low, high, *, resamples=2000, seed=20260921, level=0.95):
    return {"difference": (low + high) / 2.0, "ci": [low, high], "includes_zero": low <= 0.0 <= high, "seeds": 200, "a": 0.5, "b": 0.5,
            "resamples": resamples, "seed": seed, "level": level}


def _robust(a_only, b_only):
    return {"discordant": {"a_only": a_only, "b_only": b_only}, "mcnemar_exact_p": 0.5, "registered": {"seed": 20260921, "resamples": 2000},
            "alternative_seeds": {"count": 200, "lower_bound_at_or_below_zero": 3, "upper_bound_at_or_above_zero": 0}}


def _verdict_report(*, a=(-0.05, 0.10), b=(-0.20, -0.05), dup=(-0.10, 0.30), stop=(-0.20, 0.10), cause=None, condition="ood_dev200", secondary=True):
    """등록이 읽는 블록만 든 보고서 — `paired`(규칙 − r7 엄격·r7 − r5 거짓 done)와 `seed_pairs`(r7 − r5·r7 − r6·규칙 − r7)."""
    cause = cause or {"gripper_duplicates": (-0.1, 0.4), "gripper_streak": (-0.1, 0.2), "q_stop_caught": (-0.2, 0.3)}

    def block():
        metrics_r5 = {"gripper_duplicates": _ratio(*dup), "q_stop_caught": _ratio(*stop), "gripper_streak": _ratio(-0.1, 0.1)}
        metrics_r6 = {name: _ratio(*value) for name, value in cause.items()}
        return {
            "paired": {"rule - r7 (done ∧ inside)": _interval(*a), "r7 - r5 (false done)": _interval(*b)},
            "seed_pairs": {
                "r7 - r5": {"metrics": metrics_r5, "robustness": {"false_done": _robust(1, 12), "strict": _robust(20, 18)}},
                "r7 - r6": {"metrics": metrics_r6, "robustness": {"false_done": _robust(2, 2), "strict": _robust(10, 9)}},
                "rule - r7": {"metrics": {}, "robustness": {"strict": _robust(21, 15), "false_done": _robust(0, 3)}},
            },
        }

    names = [condition] + (["dev_new2"] if secondary else [])
    blocks = {name: block() for name in names}
    return {"conditions": names, "paired": {name: blocks[name]["paired"] for name in names}, "seed_pairs": {name: blocks[name]["seed_pairs"] for name in names}}


def test_the_r7_registration_is_the_briefs_rule_on_ood_dev200_with_the_registered_bootstrap():
    """등록 파일이 브리프의 규칙 그대로인가 — 주 집합·합칠 조건·부트스트랩·(a)(b)(c)의 쌍·지표·경계·원인 판정의 쌍과 세 지표."""
    from robo_jev.closed_loop import load_registration
    from robo_jev.evaluate import EPISODE_BOOTSTRAP

    registration = load_registration(REGISTRATION_PATH)
    assert registration["primary"] == "ood_dev200" and registration["secondary"] == ["dev_new2"]
    assert registration["merge"] == {"ood_dev200": ["ood_dev", "ood_dev_new", "ood_dev_new2"]}
    assert registration["bootstrap"] == EPISODE_BOOTSTRAP == {"resamples": 2000, "seed": 20260921, "level": 0.95}
    cloud = registration["cloud"]
    assert (cloud["a"]["pair"], cloud["a"]["metric"], cloud["a"]["holds_if"]) == ("rule - r7", "strict", "lower_le_zero")
    assert (cloud["b"]["pair"], cloud["b"]["metric"], cloud["b"]["holds_if"]) == ("r7 - r5", "false_done", "upper_lt_zero")
    assert (cloud["c_duplicates"]["pair"], cloud["c_duplicates"]["metric"], cloud["c_duplicates"]["holds_if"]) == ("r7 - r5", "gripper_duplicates", "lower_le_zero")
    assert (cloud["c_q_stop"]["pair"], cloud["c_q_stop"]["metric"], cloud["c_q_stop"]["holds_if"]) == ("r7 - r5", "q_stop_caught", "upper_ge_zero")
    assert registration["cause"]["pair"] == "r7 - r6" and set(registration["cause"]["metrics"]) == {"gripper_duplicates", "gripper_streak", "q_stop_caught"}
    assert registration["cause"]["calls"]["all"] == "R6의 퇴행은 expert 노출 비율 때문"
    assert registration["cause"]["calls"]["none"] == "노출 비율로 설명되지 않음 — DAgger 부가 라벨이 다음 표적"
    assert sorted(registration["seed_pairs"]) == sorted(["r7:r5", "r7:r6", "rule:r7"])


def test_each_cloud_condition_holds_exactly_at_its_registered_bound_and_the_cloud_needs_all_three():
    """(a) 하한 ≤ 0 · (b) 상한 < 0 · (c) 중복 하한 ≤ 0 그리고 q_stop 상한 ≥ 0 — 경계값에서 성립/불성립이 등록대로 갈린다."""
    from robo_jev.closed_loop import apply_registration, load_registration

    registration = load_registration(REGISTRATION_PATH)
    at_bounds = apply_registration(_verdict_report(a=(0.0, 0.2), b=(-0.2, -0.001), dup=(0.0, 1.0), stop=(-0.5, 0.0)), registration)
    conditions = at_bounds["primary"]["conditions"]
    assert all(conditions[name]["holds"] for name in ("a", "b", "c_duplicates", "c_q_stop"))
    assert at_bounds["primary"]["cloud"] is True and at_bounds["verdict"]["cloud"] is True and at_bounds["primary"]["failed"] == []
    just_past = apply_registration(_verdict_report(a=(0.001, 0.2), b=(-0.2, 0.0), dup=(0.001, 1.0), stop=(-0.5, -0.001)), registration)
    conditions = just_past["primary"]["conditions"]
    assert not any(conditions[name]["holds"] for name in ("a", "b", "c_duplicates", "c_q_stop"))
    assert just_past["verdict"]["cloud"] is False and just_past["primary"]["failed"] == ["a", "b", "c_duplicates", "c_q_stop"]
    only_c_fails = apply_registration(_verdict_report(dup=(0.2, 1.0)), registration)
    assert only_c_fails["primary"]["failed"] == ["c_duplicates"] and only_c_fails["verdict"]["cloud"] is False
    # 값과 구간은 보고서의 것 그대로 옮긴다
    assert conditions["a"]["ci"] == [0.001, 0.2] and conditions["c_q_stop"]["ci"] == [-0.5, -0.001]


def test_a_pair_written_the_other_way_round_is_read_with_its_sign_flipped():
    """보고서의 쌍 이름은 이름표 순서가 정한다 — `r7 - rule`로 적혔으면 구간을 뒤집어 `rule - r7`로 읽는다."""
    from robo_jev.closed_loop import apply_registration, load_registration

    report = _verdict_report()
    for name in report["conditions"]:
        paired = report["paired"][name]
        paired["r7 - rule (done ∧ inside)"] = _interval(-0.3, 0.1)  # = rule − r7 [−0.1, +0.3]
        del paired["rule - r7 (done ∧ inside)"]
    out = apply_registration(report, load_registration(REGISTRATION_PATH))
    assert out["primary"]["conditions"]["a"]["ci"] == pytest.approx([-0.1, 0.3]) and out["primary"]["conditions"]["a"]["holds"] is True
    assert out["primary"]["conditions"]["a"]["read_as"] == "r7 - rule (done ∧ inside), sign flipped"


def test_robustness_is_written_next_to_a_and_b_and_does_not_change_the_verdict():
    from robo_jev.closed_loop import apply_registration, load_registration

    out = apply_registration(_verdict_report(), load_registration(REGISTRATION_PATH))
    robust = out["primary"]["robustness"]
    assert robust["a"]["discordant"] == {"a_only": 21, "b_only": 15} and robust["b"]["discordant"] == {"a_only": 1, "b_only": 12}
    assert robust["a"]["mcnemar_exact_p"] == 0.5 and robust["a"]["alternative_seeds"]["lower_bound_at_or_below_zero"] == 3
    assert out["verdict"]["cloud"] is True  # 견고성 블록은 판정에 들어가지 않는다


def test_the_cause_call_is_all_none_or_mixed_and_an_interval_against_r7_is_named_a_new_regression():
    from robo_jev.closed_loop import apply_registration, load_registration

    registration = load_registration(REGISTRATION_PATH)
    calls = registration["cause"]["calls"]
    toward = {"gripper_duplicates": (-1.5, -0.5), "gripper_streak": (-0.7, -0.4), "q_stop_caught": (0.2, 0.5)}
    everything = apply_registration(_verdict_report(cause=toward), registration)["primary"]["cause"]
    assert everything["call"] == "all" and everything["text"] == calls["all"] and everything["new_regressions"] == []
    assert all(entry["status"] == "toward_r7" for entry in everything["metrics"].values())
    nothing = apply_registration(_verdict_report(cause={"gripper_duplicates": (-0.5, 0.5), "gripper_streak": (-0.1, 0.0), "q_stop_caught": (0.0, 0.3)}), registration)["primary"]["cause"]
    assert nothing["call"] == "none" and nothing["text"] == calls["none"]
    assert [entry["status"] for entry in nothing["metrics"].values()] == ["includes_zero"] * 3  # 경계 0은 "제외"가 아니다
    mixed = apply_registration(_verdict_report(cause={"gripper_duplicates": (-1.0, -0.2), "gripper_streak": (-0.1, 0.1), "q_stop_caught": (-0.4, -0.1)}), registration)["primary"]["cause"]
    assert mixed["call"] == "mixed" and mixed["text"] == calls["mixed"]
    assert mixed["metrics"]["gripper_duplicates"]["call"] == registration["cause"]["per_metric"]["explained"]
    assert mixed["metrics"]["gripper_streak"]["call"] == registration["cause"]["per_metric"]["not_explained"]
    assert mixed["metrics"]["q_stop_caught"]["status"] == "against_r7" and mixed["new_regressions"] == ["q_stop_caught"]


def test_the_secondary_set_is_written_beside_the_verdict_but_does_not_make_it():
    from robo_jev.closed_loop import apply_registration, load_registration

    registration = load_registration(REGISTRATION_PATH)
    report = _verdict_report()
    report["paired"]["dev_new2"]["rule - r7 (done ∧ inside)"] = _interval(0.1, 0.3)  # 둘째 근거에서는 (a)가 떨어져도
    out = apply_registration(report, registration)
    assert out["secondary"]["dev_new2"]["conditions"]["a"]["holds"] is False and out["verdict"]["cloud"] is True
    missing = apply_registration(_verdict_report(secondary=False), registration)
    assert missing["secondary"]["dev_new2"] == {"available": False, "reason": "condition dev_new2 is not in the report"}


def test_apply_registration_refuses_a_report_that_lacks_what_the_rule_reads():
    from robo_jev.closed_loop import apply_registration, load_registration

    registration = load_registration(REGISTRATION_PATH)
    with pytest.raises(ValueError, match="ood_dev200"):
        apply_registration(_verdict_report(condition="ood_dev100", secondary=False), registration)
    report = _verdict_report()
    del report["seed_pairs"]["ood_dev200"]["r7 - r6"]
    with pytest.raises(ValueError, match="r7 - r6"):
        apply_registration(report, registration)
    report = _verdict_report()
    report["seed_pairs"]["ood_dev200"]["r7 - r5"]["metrics"]["gripper_duplicates"]["seed"] = 1  # 등록하지 않은 부트스트랩
    with pytest.raises(ValueError, match="bootstrap"):
        apply_registration(report, registration)


def test_the_verdict_command_writes_the_applied_rule_and_prints_the_table(tmp_path, capsys):
    """`scripts/closed_loop.py verdict` — 보고서 JSON + 등록 파일 → 판정 JSON과 표 (E1이 쓰는 명령 그대로)."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("closed_loop_script", REPO / "scripts" / "closed_loop.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    report_path, out_path = tmp_path / "report.json", tmp_path / "verdict.json"
    report_path.write_text(json.dumps(_verdict_report(dup=(0.2, 1.0))), encoding="utf-8")
    assert module.main(["verdict", "--report", str(report_path), "--registration", str(REGISTRATION_PATH), "--out", str(out_path)]) == 0
    written = json.loads(out_path.read_text(encoding="utf-8"))
    assert written["verdict"]["cloud"] is False and written["verdict"]["failed"] == ["c_duplicates"] and written["report"] == str(report_path)
    printed = capsys.readouterr().out
    assert "| c_duplicates | r7 - r5 | gripper_duplicates | +0.600 [+0.200, +1.000] | lower_le_zero | no |" in printed
    assert "cloud: **closed** (failed: ['c_duplicates'])" in printed and "ood_dev200 (primary)" in printed and "dev_new2 (secondary)" in printed


def test_the_r7_seed_walk_is_r6s_generator_and_mix_on_a_range_no_earlier_walk_or_dagger_collection_touches():
    """Task R7 B1: 새 ood_dev 100 — R6와 같은 생성 설정·프로파일 혼합, seed base 1050100. 프로파일마다 걷는 구간(base + 프로파일 × 10,000,
    최대 4,000 seed)이 r1 · DAgger 사이클 0~5 · R4 · R5 · R6의 구간과 겹치지 않는다."""
    r7 = load_closed_loop_config(REPO / "configs/eval/r7-closed-loop.yaml")
    r6 = load_closed_loop_config(REPO / "configs/eval/r6-closed-loop.yaml")
    assert r7["conditions"] == {"ood_dev_new2": {"split": "ood_dev", "count": 100}} and r7["seeds"]["base"] == 1050100
    assert r7["generator"] == r6["generator"] and r7["generator_config_path"] == r6["generator_config_path"]
    assert r7["generator"]["profile_weights"] == [20, 40, 40] and r7["scan"] == r6["scan"] == {"per_profile_max": 4000}
    offset = int(r7["generator"]["seeds"]["profile_offset"])

    def ranges(base: int) -> set[tuple[int, int]]:
        return {(base + index * offset, base + index * offset + r7["scan"]["per_profile_max"]) for index in range(len(r7["generator"]["profiles"]))}

    def disjoint(a: set[tuple[int, int]], b: set[tuple[int, int]]) -> bool:
        return all(high_a <= low_b or high_b <= low_a for low_a, high_a in a for low_b, high_b in b)

    mine = ranges(1050100)
    earlier = {"r1": 400100, "R4": 900100, "R5": 950100, "R6": 980100}
    earlier.update({f"dagger cycle {cycle}": int(r7["generator"]["seeds"]["base"]) + (cycle + 1) * int(r7["generator"]["seeds"]["dagger_cycle_offset"]) for cycle in range(6)})
    assert earlier["dagger cycle 1"] == 600100 and earlier["r1"] == int(r7["generator"]["seeds"]["base"])
    for name, base in earlier.items():
        assert disjoint(mine, ranges(base)), name


# --------------------------------------------------------------------------
# Task R8 사전 등록 — R7의 (a)(b)(c)를 seed마다, 두 seed가 모두 통과해야 클라우드 (configs/eval/r8-registration.yaml)
# --------------------------------------------------------------------------

R8_REGISTRATION_PATH = REPO / "configs/eval/r8-registration.yaml"
PASSING = {"a": (-0.05, 0.10), "b": (-0.20, -0.05), "dup": (-0.10, 0.30), "stop": (-0.20, 0.10)}


def _r8_report(*, s18=None, s19=None, seeds=("r8s18", "r8s19"), between=None, secondary=True):
    """등록이 읽는 블록만 든 R8 보고서 — seed마다 `paired`(규칙 − r8sXX 엄격·r8sXX − r5 거짓 done)와 `seed_pairs`(r8sXX − r5·규칙 − r8sXX),
    그리고 두 seed가 다 있으면 seed 사이 쌍(r8s18 − r8s19)."""
    values = {"r8s18": {**PASSING, **(s18 or {})}, "r8s19": {**PASSING, **(s19 or {})}}
    between = between or {"strict": (-0.08, 0.06), "false_done": (-0.03, 0.02), "gripper_duplicates": (-0.4, 0.2)}

    def block():
        paired: dict = {}
        pairs: dict = {}
        for label in seeds:
            v = values[label]
            paired[f"rule - {label} (done ∧ inside)"] = _interval(*v["a"])
            paired[f"{label} - r5 (false done)"] = _interval(*v["b"])
            pairs[f"{label} - r5"] = {"metrics": {"gripper_duplicates": _ratio(*v["dup"]), "q_stop_caught": _ratio(*v["stop"]), "gripper_streak": _ratio(-0.1, 0.1)},
                                      "robustness": {"false_done": _robust(1, 12), "strict": _robust(20, 18)}}  # fmt: skip
            pairs[f"rule - {label}"] = {"metrics": {}, "robustness": {"strict": _robust(21, 15), "false_done": _robust(0, 3)}}
        if len(seeds) == 2:
            paired["r8s18 - r8s19 (done ∧ inside)"] = _interval(*between["strict"])
            paired["r8s18 - r8s19 (false done)"] = _interval(*between["false_done"])
            pairs["r8s18 - r8s19"] = {"metrics": {"gripper_duplicates": _ratio(*between["gripper_duplicates"])}, "robustness": {}}
        return {"paired": paired, "seed_pairs": pairs}

    names = ["ood_dev200"] + (["dev_new2"] if secondary else [])
    blocks = {name: block() for name in names}
    return {"conditions": names, "paired": {name: blocks[name]["paired"] for name in names}, "seed_pairs": {name: blocks[name]["seed_pairs"] for name in names}}


def test_the_r8_registration_is_r7s_three_conditions_for_each_seed_with_the_training_configs_monitor():
    """등록 파일 = 브리프: 조건 (a)(b)(c)를 seed 18·19 각각에, 주 집합 ood_dev200·부트스트랩 등록값, 감시(q_gripper · step 150 · 창 20 ·
    0.70 — 브리프의 0.95를 A3 뒤에 바꿨다)는 학습 설정(`qwen35-2b-r8.yaml`)의 `head_fit_monitor`와 같은 값이고, 원인 판정은 없다(이 라운드는 안정성을 잰다)."""
    import yaml

    from robo_jev.closed_loop import load_registration
    from robo_jev.evaluate import EPISODE_BOOTSTRAP

    registration = load_registration(R8_REGISTRATION_PATH)
    assert registration["primary"] == "ood_dev200" and registration["secondary"] == ["dev_new2"]
    assert registration["merge"] == {"ood_dev200": ["ood_dev", "ood_dev_new", "ood_dev_new2"]} and registration["bootstrap"] == EPISODE_BOOTSTRAP
    assert registration["groups"] == {"r8s18": {"seed": 18, "run": "artifacts/runs/r8-t1-fp32-2b-s18"}, "r8s19": {"seed": 19, "run": "artifacts/runs/r8-t1-fp32-2b-s19"}}
    r7 = load_registration(REGISTRATION_PATH)["cloud"]
    for label, prefix in (("r8s18", "s18"), ("r8s19", "s19")):
        for name in ("a", "b", "c_duplicates", "c_q_stop"):
            mine, theirs = registration["cloud"][f"{prefix}_{name}"], r7[name]
            assert mine["group"] == label and mine["pair"] == theirs["pair"].replace("r7", label)
            assert (mine["metric"], mine["source"], mine["holds_if"]) == (theirs["metric"], theirs["source"], theirs["holds_if"])
    assert len(registration["cloud"]) == 8 and registration.get("cause") is None
    training = yaml.safe_load((REPO / "configs/train/qwen35-2b-r8.yaml").read_text(encoding="utf-8"))["head_fit_monitor"]
    assert {key: registration["monitor"][key] for key in ("question", "step", "window", "ratio")} == training == {"question": "q_gripper", "step": 150, "window": 20, "ratio": 0.70}
    changed = registration["monitor"]["changed"]  # 브리프의 0.95를 A3 뒤·첫 학습 step 전에 바꿨다 — 까닭과 근거 파일이 등록에 있다
    assert (changed["from"], changed["to"]) == (0.95, 0.70) and changed["evidence"] == "artifacts/reports/r8-a3-teacher-forced.json" and "0.735" in changed["why"]
    assert sorted(registration["seed_pairs"]) == sorted(["r8s18:r5", "r8s19:r5", "rule:r8s18", "rule:r8s19", "r8s18:r8s19"])
    assert {entry["pair"] for entry in registration["stability"]} == {"r8s18 - r8s19"}


def test_the_cloud_is_recommended_only_when_both_seeds_pass_all_three_conditions():
    from robo_jev.closed_loop import apply_registration, load_registration

    registration = load_registration(R8_REGISTRATION_PATH)
    both = apply_registration(_r8_report(), registration)
    assert both["verdict"]["cloud"] is True and both["verdict"]["groups"] == {"r8s18": True, "r8s19": True}
    one = apply_registration(_r8_report(s19={"a": (0.01, 0.2)}), registration)
    assert one["verdict"]["cloud"] is False and one["verdict"]["failed"] == ["s19_a"]
    assert one["verdict"]["groups"] == {"r8s18": True, "r8s19": False} and one["primary"]["groups"]["r8s19"]["failed"] == ["s19_a"]
    edge = apply_registration(_r8_report(s18={"b": (-0.2, 0.0)}, s19={"stop": (-0.5, -0.001)}), registration)
    assert edge["verdict"]["failed"] == ["s18_b", "s19_c_q_stop"] and edge["verdict"]["groups"] == {"r8s18": False, "r8s19": False}
    assert edge["primary"]["cause"] is None and edge["verdict"]["cause"] is None


def test_a_seed_the_head_fit_monitor_stopped_fails_its_conditions_without_a_closed_loop():
    """감시가 멈춘 seed는 폐루프가 없다 — 그 seed의 조건 넷은 보고서를 읽지 않고 불성립(이유: 감시), 다른 seed는 그대로 읽는다; 클라우드는 닫힌다."""
    from robo_jev.closed_loop import apply_registration, load_registration

    registration = load_registration(R8_REGISTRATION_PATH)
    stopped = {"r8s19": "head-fit monitor: not_fitting (q_gripper 0.61 ≥ 0.95 × 0.60)"}
    out = apply_registration(_r8_report(seeds=("r8s18",)), registration, stopped=stopped)
    conditions = out["primary"]["conditions"]
    assert all(conditions[f"s19_{name}"]["holds"] is False and conditions[f"s19_{name}"]["stopped"] == stopped["r8s19"] for name in ("a", "b", "c_duplicates", "c_q_stop"))
    assert all(conditions[f"s18_{name}"]["holds"] for name in ("a", "b", "c_duplicates", "c_q_stop"))
    assert out["verdict"]["cloud"] is False and out["verdict"]["groups"] == {"r8s18": True, "r8s19": False}
    assert out["primary"]["groups"]["r8s19"]["stopped"] == stopped["r8s19"]
    assert out["primary"]["robustness"]["s19_a"]["available"] is False and out["primary"]["robustness"]["s18_a"]["discordant"] == {"a_only": 21, "b_only": 15}
    assert all(entry["available"] is False for entry in out["primary"]["stability"])  # seed 사이 쌍은 한 seed가 없으면 읽을 수 없다
    with pytest.raises(ValueError, match="r8s19"):
        apply_registration(_r8_report(seeds=("r8s18",)), registration)  # 멈췄다는 말 없이 쌍이 없으면 거절 — 조용히 넘기지 않는다


def test_the_stability_readings_sit_beside_the_verdict_and_do_not_make_it():
    from robo_jev.closed_loop import apply_registration, load_registration

    registration = load_registration(R8_REGISTRATION_PATH)
    out = apply_registration(_r8_report(between={"strict": (0.05, 0.3), "false_done": (-0.1, 0.1), "gripper_duplicates": (1.0, 3.0)}), registration)
    stability = {entry["name"]: entry for entry in out["primary"]["stability"]}
    assert stability["strict"]["ci"] == [0.05, 0.3] and stability["strict"]["includes_zero"] is False
    assert stability["gripper_duplicates"]["ci"] == [1.0, 3.0] and stability["false_done"]["includes_zero"] is True
    assert stability["q_stop_caught"]["available"] is False  # 보고서에 없는 지표는 없다고 적는다
    assert out["verdict"]["cloud"] is True  # seed 사이 차가 커도 판정은 조건만 정한다


def test_the_verdict_command_reads_each_seeds_monitor_and_refuses_one_that_is_not_the_registered_rule(tmp_path, capsys):
    """`verdict`는 등록의 `groups[].run/metrics.json`에서 감시 판정을 읽는다 — `fits`가 아니면 그 seed는 멈춘 것, 학습이 적용한 감시(질문·
    step·창·비율)가 등록값과 다르면 판정하지 않는다."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("closed_loop_script", REPO / "scripts" / "closed_loop.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    def write_run(name: str, verdict: str, *, ratio: float = 0.70) -> None:
        run = tmp_path / "artifacts" / "runs" / name
        run.mkdir(parents=True, exist_ok=True)
        monitor = {"question": "q_gripper", "step": 150, "window": [131, 150], "stop_ratio": ratio, "steps_used": 20, "loss_mean": 0.3,
                   "baseline_mean": 0.6, "ratio_to_baseline": 0.5, "threshold": ratio * 0.6, "verdict": verdict}  # fmt: skip
        status = "completed" if verdict == "fits" else "stopped_head_not_fitting"
        (run / "metrics.json").write_text(json.dumps({"status": status, "step": 233 if verdict == "fits" else 150, "summary": {"head_fit_monitor": monitor}}), encoding="utf-8")

    write_run("r8-t1-fp32-2b-s18", "fits")
    write_run("r8-t1-fp32-2b-s19", "not_fitting")
    report_path, out_path = tmp_path / "report.json", tmp_path / "verdict.json"
    report_path.write_text(json.dumps(_r8_report(seeds=("r8s18",))), encoding="utf-8")
    args = ["verdict", "--report", str(report_path), "--registration", str(R8_REGISTRATION_PATH), "--runs-root", str(tmp_path), "--out", str(out_path)]
    assert module.main(args) == 0
    written = json.loads(out_path.read_text(encoding="utf-8"))
    assert written["verdict"]["cloud"] is False and written["verdict"]["groups"] == {"r8s18": True, "r8s19": False}
    assert written["monitor"]["r8s19"]["verdict"] == "not_fitting" and written["monitor"]["r8s18"]["verdict"] == "fits"
    printed = capsys.readouterr().out
    assert "r8s19" in printed and "stopped" in printed and "cloud: **closed**" in printed
    write_run("r8-t1-fp32-2b-s19", "fits", ratio=0.95)  # 브리프의 옛 상수로 돈 run은 등록과 다르다
    with pytest.raises(ValueError, match="등록"):
        module.main(args)
    import shutil

    shutil.rmtree(tmp_path / "artifacts" / "runs" / "r8-t1-fp32-2b-s19")
    assert module.main(args) == 0  # 학습이 없던 seed도 멈춘 것으로 센다 (이유: run 없음)
    assert "no training run" in json.loads(out_path.read_text(encoding="utf-8"))["primary"]["groups"]["r8s19"]["stopped"]


# --------------------------------------------------------------------------
# Task R9 사전 등록 — 한 조리법의 seed 셋에서 검증 집합(dev_new2)으로 하나를 고르고 그 seed만 ood_dev 200에서 판정한다
# (configs/eval/r9-registration.yaml)
# --------------------------------------------------------------------------

R9_REGISTRATION_PATH = REPO / "configs/eval/r9-registration.yaml"
R9_SEEDS = ("r8s19", "r9s18c", "r9s17")
#: 판정 집합의 편 수·검증 집합의 편 수 (R7·R8과 같은 장면), 그리고 판정 집합을 이루는 조건 하나(부분으로 고르는 등록의 거절 시험)
R9_EPISODES = {"ood_dev200": 200, "dev_new2": 100, "ood_dev_new2": 100}


def _r9_interval(low, high, *, seeds):
    return {"margin": (low + high) / 2.0, "margin_ci": [low, high], "margin_includes_zero": low <= 0.0 <= high, "seeds": seeds, "a": 0.5, "b": 0.5}


def _r9_report(*, dev=None, primary=None, judged=None, conditions=("ood_dev200", "dev_new2"), labels=R9_SEEDS, shared=None):
    """R9 등록이 읽는 블록만 든 보고서. 조건마다 후보의 표(`strict_done`·`false_done`·`done`·그리퍼 중복·`q_stop`), 후보마다 판정의 쌍
    (규칙 − s 엄격, s − r5 거짓 done, s − r5 중복·q_stop, 견고성), 그리고 후보 사이의 짝지은 블록(공통 seed 수 = 편 수; `shared`로 바꾼다).
    `dev`·`primary`는 이름표 → (엄격 성공 수, 거짓 done 수) — 기본은 **판정 집합으로 고르면 r8s19가, 검증 집합으로 고르면 r9s18c가** 뽑히는 수다."""
    dev = {"r8s19": (26, 25), "r9s18c": (61, 9), "r9s17": (55, 4), **(dev or {})}
    primary = {"r8s19": (150, 2), "r9s18c": (40, 30), "r9s17": (35, 20), **(primary or {})}
    judged = judged or {}

    def block(condition):
        counts = dev if condition == "dev_new2" else primary
        episodes = R9_EPISODES[condition]
        tables, paired, pairs = {}, {}, {}
        for label in labels:
            strict, false_done = counts[label]
            tables[label] = {"episodes": episodes, "strict_done": strict, "done": strict + false_done, "false_done": [f"ep-{n}" for n in range(false_done)],
                             "gripper_events": {"duplicate": 40 + strict % 7}, "stop_timing": {"reacted": strict % 5, "onsets": 30}}
            v = {**PASSING, **judged.get(label, {})}
            paired[f"rule - {label} (done ∧ inside)"] = _r9_interval(*v["a"], seeds=episodes)
            paired[f"{label} - r5 (false done)"] = _r9_interval(*v["b"], seeds=episodes)
            pairs[f"{label} - r5"] = {"metrics": {"gripper_duplicates": _ratio(*v["dup"]), "q_stop_caught": _ratio(*v["stop"])},
                                      "robustness": {"false_done": _robust(3, 9), "strict": _robust(4, 30)}}
            pairs[f"rule - {label}"] = {"metrics": {}, "robustness": {"strict": _robust(40, 11), "false_done": _robust(0, 3)}}
        for index, a in enumerate(labels):
            for b in labels[index + 1 :]:
                seeds = (shared or {}).get((a, b), episodes)
                paired[f"{a} - {b} (done ∧ inside)"] = _r9_interval(-0.1, 0.1, seeds=seeds)
                paired[f"{a} - {b} (false done)"] = _r9_interval(-0.1, 0.1, seeds=seeds)
        for a, b in (("r9s18c", "r8s19"), ("r9s17", "r8s19"), ("r9s18c", "r9s17")):
            if a in labels and b in labels:
                pairs[f"{a} - {b}"] = {"metrics": {"gripper_duplicates": _ratio(-0.3, 0.2), "q_stop_caught": _ratio(-0.1, 0.4)}, "robustness": {}}
        tables["rule"] = {"episodes": episodes, "strict_done": 140, "done": 140, "false_done": [], "gripper_events": {"duplicate": 700}, "stop_timing": {"reacted": 30, "onsets": 43}}
        return tables, paired, pairs

    blocks = {name: block(name) for name in conditions}
    return {"conditions": list(conditions), "tables": {name: blocks[name][0] for name in conditions}, "paired": {name: blocks[name][1] for name in conditions},
            "seed_pairs": {name: blocks[name][2] for name in conditions}}


def test_the_r9_registration_selects_on_dev_new2_and_judges_only_the_selected_seed_by_r7s_conditions():
    """등록 파일 = 브리프 Stage A: 후보 셋(r8s19 그대로·r9s18c·r9s17)과 그 run 디렉터리, 고르기(dev_new2 엄격 성공 → 거짓 done 적은 쪽 → seed 번호 작은 쪽),
    판정(고른 seed만, ood_dev200에서 R7의 (a)(b)(c) 그대로 — 쌍의 r7 자리에 `{selected}`), 통과·불통과의 등록 문장, 감시는 기록 전용, 보고서 명령이 실을
    seed 쌍이 판정·견고성·분포 판독이 읽는 쌍을 **어느 seed가 뽑히든** 모두 담는다."""
    from robo_jev.closed_loop import load_registration
    from robo_jev.evaluate import EPISODE_BOOTSTRAP

    registration = load_registration(R9_REGISTRATION_PATH)
    assert registration["primary"] == "ood_dev200" and registration["secondary"] == ["dev_new2"]
    assert registration["merge"] == {"ood_dev200": ["ood_dev", "ood_dev_new", "ood_dev_new2"]} and registration["bootstrap"] == EPISODE_BOOTSTRAP
    assert registration["groups"] == {
        "r8s19": {"seed": 19, "run": "artifacts/runs/r8-t1-fp32-2b-s19"},
        "r9s18c": {"seed": 18, "run": "artifacts/runs/r9-t1-fp32-2b-s18c/r8-t1-fp32-2b-s18"},
        "r9s17": {"seed": 17, "run": "artifacts/runs/r9-t1-fp32-2b-s17"},
    }
    assert registration["selection"] == {"condition": "dev_new2", "candidates": list(R9_SEEDS), "metric": "strict", "ties": ["false_done", "seed"]}
    r7 = load_registration(REGISTRATION_PATH)["cloud"]
    assert set(registration["cloud"]) == set(r7) == {"a", "b", "c_duplicates", "c_q_stop"}
    for name, mine in registration["cloud"].items():
        assert mine["group"] == "selected" and mine["pair"] == r7[name]["pair"].replace("r7", "{selected}")
        assert (mine["metric"], mine["source"], mine["holds_if"]) == (r7[name]["metric"], r7[name]["source"], r7[name]["holds_if"])
    assert registration["calls"]["pass"].startswith("여러 seed를 돌려 검증 집합으로 고르는 절차가 통한다") and "사용자 결정" in registration["calls"]["pass"]
    assert "학습 안정성" in registration["calls"]["fail"] and "q_stop" in registration["calls"]["fail"]
    assert registration.get("monitor") is None  # 감시는 멈추지 않는다 — 기록 전용 통계만
    assert registration["run_readings"] == {"monitor_log": {"question": "q_gripper", "step": 150, "window": 20, "ratio": 0.70},
                                            "gripper_onset": {"stratum": "settled", "min_accuracy": 0.5, "min_ticks": 5}, "log_only": ["r9s18c", "r9s17"]}
    pairs = {tuple(pair.split(":")) for pair in registration["seed_pairs"]}
    needed = {pair for s in R9_SEEDS for pair in ((s, "r5"), ("rule", s))} | {("r9s18c", "r8s19"), ("r9s17", "r8s19"), ("r9s18c", "r9s17")}
    assert pairs == needed
    read_by_stability = {tuple(entry["pair"].split(" - ")) for entry in registration["stability"] if entry["source"] == "seed_pairs"}
    assert read_by_stability <= pairs and {entry["metric"] for entry in registration["stability"]} == {"strict", "false_done", "gripper_duplicates", "q_stop_caught"}
    assert [(entry["condition"], entry["pair"], entry["metric"]) for entry in registration["robustness"]] == [("a", "rule - {selected}", "strict"), ("b", "{selected} - r5", "false_done")]
    training = yaml.safe_load((REPO / "configs/train/qwen35-2b-r8.yaml").read_text(encoding="utf-8"))
    assert training["max_steps"] == 233  # 조리법은 R8의 것 그대로 — 두 새 run 모두 같은 일정


def test_the_seed_with_the_most_dev_new2_strict_successes_is_selected_and_ties_go_to_fewer_false_dones_then_the_smaller_seed():
    from robo_jev.closed_loop import load_registration, select_seed

    registration = load_registration(R9_REGISTRATION_PATH)
    plain = select_seed(_r9_report(), registration)
    assert plain["selected"] == "r9s18c" and plain["ranking"] == ["r9s18c", "r9s17", "r8s19"] and plain["decided_by"] == "strict"
    assert [(row["label"], row["strict"], row["false_done"], row["episodes"]) for row in plain["candidates"]] == [("r8s19", 26, 25, 100), ("r9s18c", 61, 9, 100), ("r9s17", 55, 4, 100)]
    tie_on_strict = select_seed(_r9_report(dev={"r9s18c": (55, 9), "r9s17": (55, 4)}), registration)
    assert tie_on_strict["selected"] == "r9s17" and tie_on_strict["decided_by"] == "false_done"
    full_tie = select_seed(_r9_report(dev={"r8s19": (55, 4), "r9s18c": (55, 4), "r9s17": (55, 4)}), registration)
    assert full_tie["selected"] == "r9s17" and full_tie["ranking"] == ["r9s17", "r9s18c", "r8s19"] and full_tie["decided_by"] == "seed"


def test_the_selection_never_reads_the_judged_set_and_a_registration_that_would_is_refused(tmp_path):
    """고르기는 dev_new2의 표만 읽는다 — 판정 집합(ood_dev200)의 표·쌍을 무엇으로 바꾸거나 지워도 같은 seed가 뽑힌다(기본 수는 판정 집합으로 고르면
    r8s19가 뽑히게 짜여 있다). 그리고 판정 집합이나 그것을 이루는 조건으로 고르겠다는 등록은 읽히지 않는다."""
    from robo_jev.closed_loop import load_registration, select_seed

    registration = load_registration(R9_REGISTRATION_PATH)
    report = _r9_report()
    chosen = select_seed(report, registration)
    assert chosen["selected"] == "r9s18c"
    report["tables"]["ood_dev200"] = {}
    report["paired"]["ood_dev200"] = {}
    report["seed_pairs"]["ood_dev200"] = {}
    assert select_seed(report, registration) == chosen
    raw = yaml.safe_load(R9_REGISTRATION_PATH.read_text(encoding="utf-8"))
    for condition in ("ood_dev200", "ood_dev", "ood_dev_new", "ood_dev_new2"):
        path = tmp_path / f"select-on-{condition}.yaml"
        path.write_text(yaml.safe_dump({**raw, "selection": {**raw["selection"], "condition": condition}}, allow_unicode=True), encoding="utf-8")
        with pytest.raises(ValueError, match="판정할 집합의 숫자로 고르지 않는다"):
            load_registration(path)
    bad_cloud = copy.deepcopy(raw)
    bad_cloud["cloud"]["a"]["pair"] = "rule - r9s17"  # 고른 seed 자리 없이 한 seed를 못 박은 조건
    path = tmp_path / "fixed-seed.yaml"
    path.write_text(yaml.safe_dump(bad_cloud, allow_unicode=True), encoding="utf-8")
    with pytest.raises(ValueError, match="고른 seed 하나"):
        load_registration(path)


def test_only_the_selected_seed_is_judged_on_ood_dev200_and_the_registered_sentence_follows_the_verdict():
    """(a)(b)(c)는 고른 seed의 쌍만 읽는다 — 다른 후보가 무엇이든 판정을 바꾸지 않는다; 넷이 모두 성립하면 등록한 통과 문장, 아니면 불통과 문장.
    견고성도 고른 seed의 쌍이고, dev_new2의 같은 조건은 옆에 적기만 한다."""
    from robo_jev.closed_loop import apply_registration, load_registration

    registration = load_registration(R9_REGISTRATION_PATH)
    failing = {"a": (0.2, 0.4), "b": (0.01, 0.2), "dup": (0.5, 2.0), "stop": (-0.8, -0.3)}
    passing = apply_registration(_r9_report(judged={"r8s19": failing, "r9s17": failing}), registration)
    assert passing["verdict"]["selected"] == "r9s18c" and passing["selection"]["selected"] == "r9s18c"
    conditions = passing["primary"]["conditions"]
    assert [conditions[name]["pair"] for name in ("a", "b", "c_duplicates", "c_q_stop")] == ["rule - r9s18c", "r9s18c - r5", "r9s18c - r5", "r9s18c - r5"]
    assert all(conditions[name]["holds"] for name in conditions) and all(entry["group"] == "r9s18c" for entry in conditions.values())
    assert passing["verdict"]["cloud"] is True and passing["verdict"]["groups"] == {"r9s18c": True}
    assert passing["verdict"]["call"] == "pass" and passing["verdict"]["call_text"] == registration["calls"]["pass"]
    assert passing["primary"]["robustness"]["a"]["pair"] == "rule - r9s18c" and passing["primary"]["robustness"]["a"]["discordant"] == {"a_only": 40, "b_only": 11}
    assert passing["secondary"]["dev_new2"]["conditions"]["a"]["pair"] == "rule - r9s18c"
    failed = apply_registration(_r9_report(judged={"r9s18c": {"a": (0.001, 0.2)}}), registration)
    assert failed["verdict"]["cloud"] is False and failed["verdict"]["failed"] == ["a"] and failed["verdict"]["call"] == "fail"
    assert failed["verdict"]["call_text"] == registration["calls"]["fail"]
    # 분포 판독: 셋 모두의 수를 표에서 옮기고(판정에 들어가지 않는다) seed 사이 쌍은 stability에 적는다
    distribution = passing["distribution"]
    assert set(distribution) == {"ood_dev200", "dev_new2"} and set(distribution["dev_new2"]) == set(R9_SEEDS)
    assert distribution["dev_new2"]["r9s17"] == {"available": True, "episodes": 100, "strict": 55, "done": 59, "false_done": 4, "gripper_duplicates": 46, "q_stop_caught": 0, "q_stop_onsets": 30}
    stability = {entry["name"]: entry for entry in passing["primary"]["stability"]}
    assert stability["r9s18c - r9s17 gripper_duplicates"]["ci"] == [-0.3, 0.2] and stability["r9s18c - r9s17 gripper_duplicates"]["includes_zero"] is True
    assert stability["r9s17 - r8s19 strict"]["available"] is True


def test_a_seed_whose_run_did_not_complete_cannot_be_selected_and_without_an_eligible_seed_every_condition_fails():
    from robo_jev.closed_loop import apply_registration, load_registration, select_seed

    registration = load_registration(R9_REGISTRATION_PATH)
    report = _r9_report(labels=("r8s19", "r9s17"))  # r9s18c의 폐루프가 없다
    unavailable = {"r9s18c": "the run did not complete (status interrupted, step 190 of max_steps 233)"}
    chosen = select_seed(report, registration, unavailable=unavailable)
    assert chosen["selected"] == "r9s17" and chosen["decided_by"] == "strict"
    assert chosen["candidates"][1] == {"label": "r9s18c", "seed": 18, "eligible": False, "reason": unavailable["r9s18c"]}
    out = apply_registration(report, registration, stopped=unavailable)
    assert out["verdict"]["selected"] == "r9s17" and out["primary"]["conditions"]["a"]["pair"] == "rule - r9s17"
    assert {entry["name"] for entry in out["primary"]["stability"] if not entry["available"]} == {
        f"{pair} {metric}" for pair in ("r9s18c - r8s19", "r9s18c - r9s17") for metric in ("strict", "false_done", "gripper_duplicates", "q_stop_caught")}
    with pytest.raises(ValueError, match="r9s18c"):
        select_seed(report, registration)  # 이유 없이 후보의 폐루프가 없으면 거절한다 — 조용히 건너뛰지 않는다
    nobody = {label: "no training run" for label in R9_SEEDS}
    empty = apply_registration(_r9_report(), registration, stopped=nobody)
    assert empty["verdict"]["selected"] is None and empty["verdict"]["cloud"] is False and empty["verdict"]["call"] == "fail"
    assert all(entry["holds"] is False and "no eligible seed" in entry["stopped"] for entry in empty["primary"]["conditions"].values())


def test_the_selection_refuses_candidates_that_did_not_run_the_same_scenes():
    from robo_jev.closed_loop import load_registration, select_seed

    registration = load_registration(R9_REGISTRATION_PATH)
    report = _r9_report(shared={("r8s19", "r9s17"): 99})
    with pytest.raises(ValueError, match="공통 seed"):
        select_seed(report, registration)
    report = _r9_report()
    report["tables"]["dev_new2"]["r9s17"]["episodes"] = 99
    with pytest.raises(ValueError, match="편 수가 다르다"):
        select_seed(report, registration)


def _probe_step(step, correct, n, *, loss=0.3, baseline=0.5):
    settled = {"n": n, "correct": correct} if n else None
    return {"step": step, "probes": {"q_gripper": ({"settled": settled} if settled else {})},
            "loss_by_question": {"q_gripper": {"loss": loss, "baseline": baseline}}}


def test_the_gripper_fit_onset_is_the_first_counted_step_after_the_last_failing_one():
    """등록한 정의: `settled` 틱이 5개 이상인 step만 세고, 그런 step마다 argmax 정답률 ≥ 0.5가 기록의 끝까지 이어지는 가장 이른 step. 틱이 적은 step은
    세지 않으며(3/3도, 0/3도), 한 번 맞았다가 다시 떨어지면 떨어진 뒤로 밀린다; 끝의 셀 수 있는 step이 떨어지면 None."""
    from robo_jev.closed_loop import gripper_fit_onset

    steps = [_probe_step(1, 0, 30), _probe_step(2, 20, 30), _probe_step(3, 1, 30), _probe_step(4, 0, 3), _probe_step(5, 16, 30),
             _probe_step(6, 3, 3), _probe_step(7, 0, 0), _probe_step(8, 30, 31)]
    onset = gripper_fit_onset(steps)
    assert onset["onset_step"] == 5 and onset["steps_counted"] == 5 and onset["last_step"] == 8  # step 2의 짧은 적합은 step 3에서 무너졌다
    assert gripper_fit_onset(steps, min_ticks=1)["onset_step"] == 5  # 4(0/3)가 세어지면 그 뒤인 5
    assert gripper_fit_onset([*steps, _probe_step(9, 2, 30)])["onset_step"] is None  # 기록의 끝에서 떨어졌다
    assert gripper_fit_onset([*steps, _probe_step(9, 2, 4)])["onset_step"] == 5  # 틱 4개짜리 step은 세지 않는다
    assert gripper_fit_onset([])["onset_step"] is None


def test_the_r8_curves_read_their_published_onsets_and_monitor_values_from_the_run_records():
    """등록의 정의와 기록 전용 감시를 R8의 두 run(숫자가 이미 공개된)에 적용하면 R8 보고서의 값이 나온다 — seed 19 시작 132·감시 0.592(fits),
    seed 18(step 150까지) 시작 142·감시 0.723(not_fitting)."""
    from robo_jev.closed_loop import load_registration, run_reading

    readings = load_registration(R9_REGISTRATION_PATH)["run_readings"]
    runs = REPO / "artifacts" / "runs"
    if not (runs / "r8-t1-fp32-2b-s19" / "metrics.json").is_file():
        pytest.fail("R8 run records are missing — artifacts/runs/r8-t1-fp32-2b-s{18,19}/metrics.json")
    s19 = run_reading(json.loads((runs / "r8-t1-fp32-2b-s19" / "metrics.json").read_text(encoding="utf-8")), readings)
    s18 = run_reading(json.loads((runs / "r8-t1-fp32-2b-s18" / "metrics.json").read_text(encoding="utf-8")), readings)
    assert s19["gripper_onset"]["onset_step"] == 132 and s18["gripper_onset"]["onset_step"] == 142
    assert s19["monitor_log"]["verdict"] == "fits" and round(s19["monitor_log"]["ratio_to_baseline"], 3) == 0.592
    assert s18["monitor_log"]["verdict"] == "not_fitting" and round(s18["monitor_log"]["ratio_to_baseline"], 3) == 0.723
    assert (s19["status"], s19["step"], s19["max_steps"], s19["seed"]) == ("completed", 233, 233, 19)
    assert (s18["status"], s18["step"], s18["applied_monitor"]["ratio"]) == ("stopped_head_not_fitting", 150, 0.7)


def test_the_verdict_command_reads_each_runs_completion_and_refuses_a_log_only_run_that_applied_a_monitor(tmp_path, capsys):
    """`verdict`(R9 등록): 묶음마다 run의 `metrics.json`에서 완주 여부를 읽는다 — `completed`로 `max_steps`에 닿지 않은 seed는 고를 수 없다. 기록 전용 감시와
    그리퍼 시작 step이 판정 옆 `runs`에 남고, 감시를 기록만 해야 하는 묶음의 학습이 감시를 적용했으면 판정하지 않는다."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("closed_loop_script", REPO / "scripts" / "closed_loop.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    def write_run(relative: str, *, status="completed", step=233, monitor=None, seed=17):
        run = tmp_path / relative
        run.mkdir(parents=True, exist_ok=True)
        steps = [_probe_step(n, 25 if n >= 130 else 0, 30, loss=0.2 if n >= 130 else 0.6, baseline=0.5) for n in range(1, step + 1)]
        payload = {"run_id": run.name, "status": status, "step": step, "config": {"max_steps": 233, "seed": seed, "head_fit_monitor": monitor, "resume": None},
                   "steps": steps, "summary": {"head_fit_monitor": None, "rescheduled": None, "sampler": {"units": {"robot/existing": 190}}}}
        (run / "metrics.json").write_text(json.dumps(payload), encoding="utf-8")

    write_run("artifacts/runs/r8-t1-fp32-2b-s19", seed=19, monitor={"question": "q_gripper", "step": 150, "window": 20, "ratio": 0.7})
    write_run("artifacts/runs/r9-t1-fp32-2b-s18c/r8-t1-fp32-2b-s18", seed=18, status="interrupted", step=190)
    write_run("artifacts/runs/r9-t1-fp32-2b-s17", seed=17)
    report_path, out_path = tmp_path / "report.json", tmp_path / "verdict.json"
    report_path.write_text(json.dumps(_r9_report(labels=("r8s19", "r9s17"))), encoding="utf-8")
    args = ["verdict", "--report", str(report_path), "--registration", str(R9_REGISTRATION_PATH), "--runs-root", str(tmp_path), "--out", str(out_path)]
    assert module.main(args) == 0
    written = json.loads(out_path.read_text(encoding="utf-8"))
    assert written["selection"]["selected"] == "r9s17" and written["verdict"]["selected"] == "r9s17"
    assert "did not complete" in written["selection"]["candidates"][1]["reason"] and "190" in written["selection"]["candidates"][1]["reason"]
    assert written["runs"]["r9s17"]["gripper_onset"]["onset_step"] == 130 and written["runs"]["r9s17"]["monitor_log"]["verdict"] == "fits"  # 창 131–150 = 0.2 / 0.5
    assert written["runs"]["r9s17"]["applied_monitor"] is None and written["runs"]["r8s19"]["applied_monitor"]["ratio"] == 0.7
    printed = capsys.readouterr().out
    assert "seed selection on dev_new2" in printed and "selected: **r9s17**" in printed and "registered call: **pass**" in printed
    write_run("artifacts/runs/r9-t1-fp32-2b-s17", seed=17, monitor={"question": "q_gripper", "step": 150, "window": 20, "ratio": 0.7})
    with pytest.raises(ValueError, match="기록만"):
        module.main(args)


def test_choosing_on_a_part_of_the_judged_set_is_refused_even_when_the_registration_does_not_declare_merge(tmp_path):
    """R9 리뷰 1 M1: `load_registration`은 등록 자신의 `merge`로만 판정 집합의 부분을 안다 — `merge`를 빼고 `ood_dev_new2`로 고르는 등록은 읽힌다.
    그래서 고르기는 **보고서가 실제로 합친 것**(`report["merge"][primary]`)도 본다: 판정 집합을 이룬 조건으로는 고르지 않는다(거절). 합치지 않은 보고서의
    다른 조건(dev_new2)으로는 그대로 고른다."""
    from robo_jev.closed_loop import apply_registration, load_registration, select_seed

    raw = yaml.safe_load(R9_REGISTRATION_PATH.read_text(encoding="utf-8"))
    no_merge = {key: value for key, value in raw.items() if key != "merge"}
    path = tmp_path / "no-merge-select-on-part.yaml"
    path.write_text(yaml.safe_dump({**no_merge, "selection": {**raw["selection"], "condition": "ood_dev_new2"}}, allow_unicode=True), encoding="utf-8")
    registration = load_registration(path)  # 등록만으로는 부분인지 알 수 없다
    report = _r9_report(conditions=("ood_dev200", "dev_new2", "ood_dev_new2"))
    report["merge"] = {"ood_dev200": ["ood_dev", "ood_dev_new", "ood_dev_new2"]}
    with pytest.raises(ValueError, match="판정할 집합의 숫자로 고르지 않는다"):
        select_seed(report, registration)
    with pytest.raises(ValueError, match="판정할 집합의 숫자로 고르지 않는다"):
        apply_registration(report, registration)
    fine = load_registration(R9_REGISTRATION_PATH)  # 등록된 R9 규칙(dev_new2)은 합친 기록이 있는 보고서에서도 그대로 고른다
    assert select_seed(report, fine)["selected"] == "r9s18c"


def test_a_seed_choosing_registration_refuses_robustness_pairs_that_do_not_name_the_selected_seed(tmp_path):
    """R9 리뷰 1 M1: 견고성(McNemar·RNG seed)은 판정 조건 옆의 것이라 고른 seed의 쌍이어야 한다 — 한 seed를 못 박은 견고성 쌍은 등록이 받지 않는다."""
    from robo_jev.closed_loop import load_registration

    raw = yaml.safe_load(R9_REGISTRATION_PATH.read_text(encoding="utf-8"))
    fixed = copy.deepcopy(raw)
    fixed["robustness"][0]["pair"] = "rule - r8s19"
    path = tmp_path / "fixed-robustness.yaml"
    path.write_text(yaml.safe_dump(fixed, allow_unicode=True), encoding="utf-8")
    with pytest.raises(ValueError, match="robustness"):
        load_registration(path)
    assert [entry["pair"] for entry in load_registration(R9_REGISTRATION_PATH)["robustness"]] == ["rule - {selected}", "{selected} - r5"]


# --------------------------------------------------------------------------
# Task R10 — 판정의 이름(decision), `conditions` 묶음, 등록한 예측(predictions), 두 효과의 차(effects)
# --------------------------------------------------------------------------


def _delegation_registration(tmp_path, **changes):
    """R10 꼴의 작은 등록 — 클라우드가 아닌 판정(`decision: delegation`), 조건 묶음은 `conditions`, 예측 둘(쌍 하나·효과의 차 하나)."""
    raw = {
        "version": "test-delegation", "decision": "delegation", "primary": "ood_dev200", "secondary": ["dev_new2"],
        "bootstrap": {"resamples": 2000, "seed": 20260921, "level": 0.95}, "seed_pairs": ["ruleG:r5G"],
        "conditions": {"a": {"what": "규칙_G − r5_G 엄격", "pair": "ruleG - r5G", "source": "paired", "metric": "strict", "holds_if": "lower_le_zero"}},
        "calls": {"pass": "실행을 맡기면 r5는 규칙 판정기보다 확실히 뒤지지 않는다", "fail": "실행을 맡겨도 규칙 판정기가 앞선다"},
        "robustness": [{"condition": "a", "pair": "ruleG - r5G", "metric": "strict"}],
        "predictions": [
            {"name": "r9s18c rises", "pair": "r9s18cG - r9s18c", "source": "paired", "metric": "strict", "holds_if": "lower_gt_zero"},
            {"name": "r8s19 rises less", "pair": "(r9s18cG - r9s18c) - (r8s19G - r8s19)", "source": "effects", "metric": "strict", "holds_if": "lower_gt_zero"},
        ],
    }
    raw.update(changes)
    raw = {key: value for key, value in raw.items() if value is not None}
    path = tmp_path / "delegation.yaml"
    path.write_text(yaml.safe_dump(raw, allow_unicode=True), encoding="utf-8")
    return path


def _delegation_report(*, a=(-0.05, 0.10), rise=(0.20, 0.40), effect=(0.05, 0.30), condition="ood_dev200"):
    def effect_block(low, high):
        return {"strict": {"metric": "strict", "seeds": 200, "a_effect": 0.3, "b_effect": 0.1, "difference": (low + high) / 2.0, "ci": [low, high],
                           "includes_zero": low <= 0.0 <= high, "resamples": 2000, "seed": 20260921, "level": 0.95}}

    names = [condition, "dev_new2"]
    return {
        "conditions": names,
        "paired": {name: {"ruleG - r5G (done ∧ inside)": _interval(*a), "r9s18cG - r9s18c (done ∧ inside)": _interval(*rise)} for name in names},
        "seed_pairs": {name: {"ruleG - r5G": {"metrics": {}, "robustness": {"strict": _robust(30, 25)}}} for name in names},
        "effects": {name: {"(r9s18cG - r9s18c) - (r8s19G - r8s19)": effect_block(*effect)} for name in names},
    }


def test_paired_effect_difference_is_the_seed_paired_difference_of_two_effects():
    """(a − a₀) − (b − b₀): a가 모든 seed에서 +1, b가 절반에서 +1이면 차 0.5이고 구간이 0을 제외한다; 네 정책이 모두 돈 seed만 쓴다; 등록 부트스트랩."""
    from robo_jev.closed_loop import paired_effect_difference

    def rows(values, *, inside=True):
        return [{"key": f"E0:{index}", "done": bool(value), "done_inside": bool(value) and inside} for index, value in enumerate(values)]

    out = paired_effect_difference(rows([1] * 10), rows([0] * 10), rows([1] * 10), rows([0] * 5 + [1] * 5), metric="strict")
    assert (out["seeds"], out["a_effect"], out["b_effect"], out["difference"]) == (10, 1.0, 0.5, 0.5)
    assert 0.0 < out["ci"][0] <= 0.5 <= out["ci"][1] <= 1.0 and out["includes_zero"] is False
    assert (out["resamples"], out["seed"], out["level"]) == (2000, 20260921, 0.95)
    common = paired_effect_difference(rows([1] * 10), rows([0] * 5), rows([1] * 10), rows([0] * 5 + [1] * 5), metric="strict")
    assert common["seeds"] == 5 and common["difference"] == 0.0 and common["ci"] == [0.0, 0.0] and common["includes_zero"] is True
    false_done = paired_effect_difference(rows([1] * 4, inside=False), rows([0] * 4), rows([0] * 4), rows([0] * 4), metric="false_done")
    assert false_done["difference"] == 1.0 and false_done["a_effect"] == 1.0
    with pytest.raises(ValueError, match="metric"):
        paired_effect_difference(rows([1]), rows([0]), rows([1]), rows([0]), metric="duplicates")
    assert paired_effect_difference(rows([1]), [], rows([1]), rows([0]), metric="strict") is None


def test_the_report_carries_the_effect_difference_of_the_four_labels_it_is_given(short_runs, tmp_path):
    """`closed_loop_report(..., effects=[(a, a0, b, b0)])`: 네 이름표가 모두 돈 조건에만 `effects[조건]["(a - a0) - (b - b0)"]`가 엄격·done·거짓 done으로
    실리고, 그 값이 같은 seed의 표에서 손으로 센 것과 같다; 청하지 않으면 블록이 없다."""
    paths = []
    for label, kind in (("expert", "expert"), ("rule", "rule"), ("mech", "mechanical"), ("expert2", "expert")):
        path = tmp_path / f"run-{label}.json"
        path.write_text(json.dumps({"label": label, "policy": {"kind": kind}, "conditions": {"dev": short_runs["runs"][kind]}}, default=str), encoding="utf-8")
        paths.append(path)
    report = closed_loop_report(paths, effects=[("expert", "rule", "expert2", "mech"), ("expert", "rule", "absent", "mech")])
    block = report["effects"]["dev"]
    assert list(block) == ["(expert - rule) - (expert2 - mech)"] and set(block["(expert - rule) - (expert2 - mech)"]) == {"strict", "done", "false_done"}
    done = {label: [int(row["done"]) for row in short_runs["runs"][kind]["episodes"]] for label, kind in (("expert", "expert"), ("rule", "rule"), ("mech", "mechanical"))}
    expected = sum((e - r) - (e - m) for e, r, m in zip(done["expert"], done["rule"], done["mech"])) / 2
    assert block["(expert - rule) - (expert2 - mech)"]["done"]["difference"] == pytest.approx(expected) and block["(expert - rule) - (expert2 - mech)"]["done"]["seeds"] == 2
    assert "effects" not in closed_loop_report(paths)


def test_a_registration_names_its_decision_and_may_write_its_conditions_as_conditions_but_not_both(tmp_path):
    """R10: `decision`(무엇을 정하는가의 이름, 없으면 cloud)·`conditions`(= cloud 자리) — 둘 다 있거나 둘 다 없으면 거절; 예측은 출처·지표·경계가 있어야 한다;
    R7 등록은 그대로 cloud로 읽힌다."""
    from robo_jev.closed_loop import load_registration

    registration = load_registration(_delegation_registration(tmp_path))
    assert registration["decision"] == "delegation" and registration["cloud"] == registration["conditions"] and set(registration["cloud"]) == {"a"}
    assert [entry["source"] for entry in registration["predictions"]] == ["paired", "effects"]
    assert load_registration(REGISTRATION_PATH)["decision"] == "cloud" and load_registration(REGISTRATION_PATH)["predictions"] == []
    both = load_registration(REGISTRATION_PATH)["cloud"]
    with pytest.raises(ValueError, match="cloud 또는 conditions"):
        load_registration(_delegation_registration(tmp_path, cloud=both))
    with pytest.raises(ValueError, match="cloud 또는 conditions"):
        load_registration(_delegation_registration(tmp_path, conditions=None))
    with pytest.raises(ValueError, match="predictions"):
        load_registration(_delegation_registration(tmp_path, predictions=[{"name": "x", "pair": "a - b", "source": "effects", "metric": "gripper_duplicates", "holds_if": "lower_gt_zero"}]))
    with pytest.raises(ValueError, match="predictions"):
        load_registration(_delegation_registration(tmp_path, predictions=[{"name": "x", "pair": "a - b", "source": "paired", "metric": "strict"}]))
    with pytest.raises(ValueError, match="decision"):
        load_registration(_delegation_registration(tmp_path, decision=3))


def test_a_non_cloud_decision_is_written_as_its_name_and_holds_and_the_registered_predictions_sit_beside_it(tmp_path):
    """`decision: delegation`이면 판정은 `verdict.decision`·`verdict.holds`(cloud 키 없음)와 등록 문장; 예측은 집합마다 성립 여부가 적히되 판정을 바꾸지
    않는다(예측이 틀려도 같은 판정); 읽을 수 없는 예측은 없다고 적는다; 견고성은 그대로 옆에."""
    from robo_jev.closed_loop import apply_registration, load_registration

    registration = load_registration(_delegation_registration(tmp_path))
    held = apply_registration(_delegation_report(), registration)
    assert "cloud" not in held["verdict"] and held["verdict"]["decision"] == "delegation" and held["verdict"]["holds"] is True
    assert held["verdict"]["call"] == "pass" and held["verdict"]["call_text"] == "실행을 맡기면 r5는 규칙 판정기보다 확실히 뒤지지 않는다"
    predictions = {entry["name"]: entry for entry in held["primary"]["predictions"]}
    assert predictions["r9s18c rises"]["holds"] is True and predictions["r8s19 rises less"]["holds"] is True
    assert predictions["r8s19 rises less"]["ci"] == [0.05, 0.30] and predictions["r8s19 rises less"]["a"] == 0.3
    assert held["primary"]["robustness"]["a"]["discordant"] == {"a_only": 30, "b_only": 25}
    wrong = apply_registration(_delegation_report(rise=(-0.1, 0.2), effect=(-0.2, 0.1)), registration)
    assert wrong["verdict"]["holds"] is True and [entry["holds"] for entry in wrong["primary"]["predictions"]] == [False, False]
    fails = apply_registration(_delegation_report(a=(0.01, 0.2)), registration)
    assert fails["verdict"]["holds"] is False and fails["verdict"]["failed"] == ["a"] and fails["verdict"]["call_text"] == "실행을 맡겨도 규칙 판정기가 앞선다"
    missing = _delegation_report()
    del missing["effects"]
    out = apply_registration(missing, registration)
    assert out["verdict"]["holds"] is True and out["primary"]["predictions"][1]["available"] is False and "effects" in out["primary"]["predictions"][1]["reason"]
    assert "predictions" not in apply_registration(_verdict_report(), load_registration(REGISTRATION_PATH))["primary"]


def test_the_verdict_command_prints_the_decisions_name_and_the_registered_predictions(tmp_path, capsys):
    import importlib.util

    spec = importlib.util.spec_from_file_location("closed_loop_script_r10v", REPO / "scripts" / "closed_loop.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    report_path, out_path = tmp_path / "report.json", tmp_path / "verdict.json"
    report_path.write_text(json.dumps(_delegation_report(a=(0.02, 0.2))), encoding="utf-8")
    assert module.main(["verdict", "--report", str(report_path), "--registration", str(_delegation_registration(tmp_path)), "--out", str(out_path)]) == 0
    written = json.loads(out_path.read_text(encoding="utf-8"))
    assert written["verdict"]["decision"] == "delegation" and written["verdict"]["holds"] is False
    printed = capsys.readouterr().out
    assert "- delegation: registered conditions **do not all hold** (failed: ['a'])" in printed and "cloud:" not in printed
    assert "- registered prediction r8s19 rises less ((r9s18cG - r9s18c) - (r8s19G - r8s19) strict): +0.175 [+0.050, +0.300], holds if lower_gt_zero → **yes**" in printed
    assert "registered call: **fail** — 실행을 맡겨도 규칙 판정기가 앞선다" in printed
