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
    for kind in ("expert", "rule", "mechanical"):
        bundle = build_policy(kind, generator=config["generator"])
        runs[kind] = run_condition(bundle, schedule, config=config, out=out / kind, condition="dev", label=kind, max_ticks=24)
    return {"out": out, "runs": runs, "schedule": schedule}


def test_a_short_run_writes_records_a_manifest_and_a_timing_sidecar(short_runs):
    for kind, run in short_runs["runs"].items():
        assert run["summary"]["episodes"] == 2 and run["condition"] == "dev" and run["label"] == kind
        assert [row["key"] for row in run["episodes"]] == ["E0:900100", "E1:910100"]
        manifest = json.loads((short_runs["out"] / kind / "manifest.json").read_text(encoding="utf-8"))
        assert manifest["episodes"] == 2 and manifest["closed_loop"]["policy"]["kind"] == kind
        rows = [json.loads(line) for line in (short_runs["out"] / kind / "timing.jsonl").read_text(encoding="utf-8").splitlines()]
        assert len(rows) == run["summary"]["ticks"] and all(row["obs_to_command_ms"] is not None and row["obs_to_command_ms"] >= 0 for row in rows)
        assert run["latency"]["obs_to_command_ms"]["n"] == run["summary"]["ticks"]
        for episode in run["episodes"]:
            assert episode["layer"] in LAYERS and episode["ticks"] <= 24 and episode["episode_id"].endswith(f"-r4-{kind}")
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
    assert failure_cause(mechanical) == "semantic"  # 참조는 결합 후보를 허용하는데 정책은 끝까지 게이트·hold
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


def _record(ticks: list[dict], *, done: bool = True, profile: str = "E0", seed: int = 1) -> dict:
    out = {"schema_version": "stream-v0", "episode_id": f"ep-{profile}-{seed}", "ticks": [], "provenance": {"profile": profile, "seed": seed, "timing": {"wall_s": 1.0},
           "outcome": {"done": done, "done_tick": (len(ticks) - 1) if done else None, "first_done_tick": None, "sim_ms": 100 * len(ticks), "terminated": "done_tail" if done else "max_ms", "target_inside_zone": done}}}
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
    assert failure_cause(wrong) == "semantic"
    idle = _record([{"adopted": "c3"}] * 4, done=False)  # 참조는 파지를 허용하는데 관측만
    assert failure_cause(idle) == "semantic"
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
    assert table["episodes"] == 3 and table["done"] == 2 and table["failure_causes"] == {"semantic": 1}
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
        env.step(None)
        assert env.intervals_ms == []  # 명령 없는 스텝은 관측→명령 구간이 아니다
        env.step({"seq": 1, "kind": "hold"} if False else None)
        assert observation["tick"] == 0
    finally:
        env.close()
