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


def test_the_report_merges_two_conditions_into_one_table_and_pairs_false_dones(short_runs, tmp_path):
    """Task R6 D2: ood_dev 100 = R4의 ood_dev 26 seed + 새 ood_dev 74 seed — 두 조건의 편을 한 표로 합치고(같은 seed가 둘에 있으면 거절),
    짝지은 비교에 **거짓 done**(`done ∧ ¬target_inside_zone`, seed마다 0/1)의 차를 더한다."""
    from robo_jev.data.robot_episodes import read_episodes, write_episode

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
