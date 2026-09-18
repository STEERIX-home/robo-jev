"""키프레임 counterfactual rollout 검사 — 사건 실행·재현·검열, 키프레임·후보 선택, 주 결정 라벨
(docs/04 §4, docs/08 §7, Task 3c-2).

물리가 도는 검사는 짧은 horizon 몇 개뿐이다. 라벨 규칙은 합성 결과표로 본다.
"""

import copy
import json

import numpy as np
import pytest
import yaml
from helpers import SIM_CONFIG

from robo_jev.contracts import validate_record
from robo_jev.harness.robot import candidate_id
from robo_jev.sim.environment import Environment
from robo_jev.sim.expert import Expert
from robo_jev.sim.label import (
    EVENT_FIELDS,
    FollowupPolicy,
    event_for,
    load_events_config,
    rollout_event,
    wilson_interval,
)

EVENTS = load_events_config()
GRASP = "grasp:o0:top:zoneL:slow"


# --------------------------------------------------------------------------
# 사건 정의 (configs/sim/events.yaml)
# --------------------------------------------------------------------------


def test_every_joint_function_has_a_versioned_event_with_the_documented_fields():
    """docs/04 §4의 기록 필드(event_id, followup_policy_version, horizon_seconds, success_rule,
    randomization_distribution)가 기능마다 정의돼 있고 후속 정책이 버전으로 고정돼 있다."""
    assert EVENTS["followup_policy_version"].startswith("followup-")
    for function, holding in (("grasp", None), ("place", "o0"), ("grasp", "o0"), ("push", None)):
        event = event_for(f"{function}:o0:top:zoneL:slow", holding=holding, config=EVENTS)
        assert event["event_id"] and event["horizon_seconds"] > 0 and event["success_rule"]["kind"]
        assert event["followup_policy_version"] == EVENTS["followup_policy_version"]
        assert set(event["randomization"]) == {"pose_jitter", "controller_noise"}
        assert event["function"] == function
    assert event_for("grasp:o0:top:zoneL:slow", holding=None, config=EVENTS)["event_id"] == "grasp-lift-v0"
    assert event_for("grasp:o0:top:zoneL:slow", holding="o0", config=EVENTS)["event_id"] == "place-release-v0"
    assert event_for("push:o0:+x:none:slow", holding=None, config=EVENTS)["event_id"] == "push-segment-v0"
    with pytest.raises(ValueError):
        event_for("hold", holding=None, config=EVENTS)
    # 후보는 같은 horizon 아래 비교한다 (docs/08 §7).
    horizons = {name: spec["horizon_seconds"] for name, spec in EVENTS["events"].items()}
    assert horizons["push"] == horizons["grasp"] == horizons["place"] == 5.0


# --------------------------------------------------------------------------
# rollout_event — 실제 환경
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def e0_seed5():
    """E0 seed 5의 reset snapshot과 그 첫 관측 (대상 o0 상자, 영역 zoneR)."""
    env = Environment(config_path=str(SIM_CONFIG), profile="E0")
    try:
        scene = env.reset(seed=5)
        snapshot = env.snapshot()
    finally:
        env.close()
    return {"snapshot": snapshot, "scene": scene}


def action_for(scene: dict, key: str) -> dict:
    return {"id": candidate_id(key), "action_ref": candidate_id(key), "key": key,
            "precision_mm": {entry["id"]: 3 for entry in scene["objects"]}}


def goal_grasp_key(scene: dict) -> str:
    return f"grasp:{scene['goal']['target_ref']}:top:{scene['goal']['zone_ref']}:slow"


def test_a_grasp_rollout_succeeds_and_records_the_event_fields(e0_seed5):
    scene = e0_seed5["scene"]
    key = goal_grasp_key(scene)
    event = event_for(key, holding=None, config=EVENTS)
    result = rollout_event(e0_seed5["snapshot"], action_for(scene, key), event, seed=0)
    assert result["outcome"] == "success", result["reason"]
    evidence = result["evidence"]
    assert set(EVENT_FIELDS) <= set(evidence)
    assert evidence["event_id"] == "grasp-lift-v0" and evidence["action_ref"] == candidate_id(key)
    assert evidence["followup_policy_version"] == EVENTS["followup_policy_version"]
    assert evidence["rollout_seed"] == 0 and evidence["horizon_seconds"] == event["horizon_seconds"]
    assert evidence["observation"]["sim_ms"] == scene["sim_time_ms"]
    assert evidence["randomization_distribution"]["applied"][scene["goal"]["target_ref"]]
    assert 0 < evidence["first_success_tick"] <= evidence["ticks"] <= event["horizon_seconds"] * 10
    assert evidence["wall_s"] > 0 and evidence["restore_s"] > 0
    assert len(evidence["trajectory"]) == evidence["ticks"]
    # 접근 시간은 구간 성과와 따로 읽힌다: 접근을 벗어난 틱, 대상과의 첫 접촉 틱.
    assert 0 < evidence["first_action_tick"] <= evidence["first_contact_tick"] <= evidence["first_success_tick"]
    assert evidence["approach_s"] == pytest.approx((evidence["first_action_tick"] - 1) * 0.1)
    assert evidence["contact_s"] == pytest.approx(evidence["first_contact_tick"] * 0.1)
    json.dumps(result)  # 저장 가능


def test_a_horizon_that_is_too_short_is_a_failure_not_a_censoring(e0_seed5):
    scene = e0_seed5["scene"]
    key = goal_grasp_key(scene)
    event = {**event_for(key, holding=None, config=EVENTS), "horizon_seconds": 1.0}
    result = rollout_event(e0_seed5["snapshot"], action_for(scene, key), event, seed=0)
    assert result["outcome"] == "failure" and result["reason"] == "horizon"
    assert result["evidence"]["ticks"] == 10


def test_the_same_snapshot_and_seed_reproduce_the_trajectory_on_a_fresh_environment(e0_seed5):
    """재현 허용 오차: 관측 해상도인 1mm — 같은 기계에서는 실제로 0이다. 새 `Environment` 인스턴스(모델을
    새로 짓고 reset 없이 restore)도 같은 궤적을 낸다 — 분산 rollout이 성립하는 조건이다."""
    scene = e0_seed5["scene"]
    key = goal_grasp_key(scene)
    event = {**event_for(key, holding=None, config=EVENTS), "horizon_seconds": 1.5}
    action = action_for(scene, key)
    shared = Environment(config_path=str(SIM_CONFIG), profile="E0")
    try:
        shared.reset(seed=11)  # 다른 에피소드를 돌린 뒤 복원한다
        for _ in range(7):
            shared.step({"kind": "MOVE_EE", "target_mm": [500, 40, 150]})
        first = rollout_event(e0_seed5["snapshot"], action, event, seed=3, env=shared)
        second = rollout_event(e0_seed5["snapshot"], action, event, seed=3, env=shared)
    finally:
        shared.close()
    fresh = rollout_event(e0_seed5["snapshot"], action, event, seed=3)  # 자기 환경을 새로 짓는다
    for other in (second, fresh):
        assert other["outcome"] == first["outcome"]
        left = np.array([row[:-1] for row in first["evidence"]["trajectory"]], dtype=float)
        right = np.array([row[:-1] for row in other["evidence"]["trajectory"]], dtype=float)
        assert left.shape == right.shape
        np.testing.assert_allclose(left, right, atol=1.0)
        assert [row[-1] for row in first["evidence"]["trajectory"]] == [row[-1] for row in other["evidence"]["trajectory"]]
        assert other["evidence"]["randomization_distribution"]["applied"] == first["evidence"]["randomization_distribution"]["applied"]


def test_a_different_seed_draws_a_different_jitter_within_the_reported_precision(e0_seed5):
    scene = e0_seed5["scene"]
    key = goal_grasp_key(scene)
    event = {**event_for(key, holding=None, config=EVENTS), "horizon_seconds": 0.5}
    action = action_for(scene, key)
    a = rollout_event(e0_seed5["snapshot"], action, event, seed=0)["evidence"]["randomization_distribution"]
    b = rollout_event(e0_seed5["snapshot"], action, event, seed=1)["evidence"]["randomization_distribution"]
    assert a["applied"] != b["applied"]
    for applied in (a["applied"], b["applied"]):
        for object_id, delta in applied.items():
            assert abs(delta["dx_mm"]) <= action["precision_mm"][object_id]
            assert abs(delta["dy_mm"]) <= action["precision_mm"][object_id]
            assert abs(delta["dyaw_deg"]) <= EVENTS["randomization"]["pose_jitter"]["yaw_deg"]
    assert a["controller_noise"]["setpoint_sigma_mm"] == EVENTS["randomization"]["controller_noise"]["setpoint_sigma_mm"]


def test_simulator_errors_and_wall_time_overruns_are_censored_with_the_reason(e0_seed5, monkeypatch):
    scene = e0_seed5["scene"]
    key = goal_grasp_key(scene)
    event = {**event_for(key, holding=None, config=EVENTS), "horizon_seconds": 1.0}
    action = action_for(scene, key)

    original = Environment.step

    def exploding(self, command=None):
        if self.tick >= 12:
            raise RuntimeError("모의 시각이 물리 시간과 어긋났다")
        return original(self, command)

    monkeypatch.setattr(Environment, "step", exploding)
    result = rollout_event(e0_seed5["snapshot"], action, event, seed=0)
    assert result["outcome"] == "censored" and result["reason"] == "simulator_error:RuntimeError"
    assert result["evidence"]["ticks"] >= 1 and "모의 시각" in result["evidence"]["error"]
    monkeypatch.setattr(Environment, "step", original)

    slow = {**event, "wall_time_limit_s": 0.0}
    result = rollout_event(e0_seed5["snapshot"], action, slow, seed=0)
    assert result["outcome"] == "censored" and result["reason"] == "wall_time_limit"
    assert result["evidence"]["ticks"] >= 1


def test_a_bug_outside_the_simulator_is_censored_as_a_pipeline_error(e0_seed5, monkeypatch):
    """simulator(환경·MuJoCo) 밖의 예외 — 하네스·전문가·후속 정책·성공 기준의 결함 — 는 `pipeline_error:<Type>`이다.
    코드 버그가 simulator 오류 통계에 섞이지 않고 제 이름으로 센다."""
    from robo_jev.harness.robot import RobotHarness

    scene = e0_seed5["scene"]
    key = goal_grasp_key(scene)
    event = {**event_for(key, holding=None, config=EVENTS), "horizon_seconds": 1.0}
    action = action_for(scene, key)
    original = RobotHarness.compose

    def broken(self, request, results, commitment, now_ms):
        if int(request["t"]) >= 3:
            raise KeyError("q_missing")
        return original(self, request, results, commitment, now_ms)

    monkeypatch.setattr(RobotHarness, "compose", broken)
    result = rollout_event(e0_seed5["snapshot"], action, event, seed=0)
    assert result["outcome"] == "censored" and result["reason"] == "pipeline_error:KeyError"
    assert "q_missing" in result["evidence"]["error"] and result["evidence"]["ticks"] >= 1

    # MuJoCo 자체의 오류도 simulator 오류다.
    import mujoco

    def fatal(self, command=None):
        raise mujoco.FatalError("mj_step: unstable simulation")

    monkeypatch.setattr(Environment, "step", fatal)
    result = rollout_event(e0_seed5["snapshot"], action, event, seed=0)
    assert result["outcome"] == "censored" and result["reason"] == "simulator_error:FatalError"


def test_a_candidate_the_followup_cannot_see_is_censored(e0_seed5):
    scene = e0_seed5["scene"]
    key = "grasp:o99:top:zoneL:slow"
    event = event_for(key, holding=None, config=EVENTS)
    result = rollout_event(e0_seed5["snapshot"], action_for(scene, key), event, seed=0)
    assert result["outcome"] == "censored" and result["reason"] == "candidate_unavailable"


def test_a_push_rollout_measures_displacement_along_the_direction():
    """E0 seed 43의 상자 o1(34×56×42)을 +x로 민다 — 시작 자세에서 3초 horizon 안에 반 구간(40mm)을 넘는다."""
    env = Environment(config_path=str(SIM_CONFIG), profile="E0")
    try:
        scene = env.reset(seed=43)
        snapshot = env.snapshot()
    finally:
        env.close()
    key = "push:o1:+x:none:slow"
    event = event_for(key, holding=None, config=EVENTS)
    result = rollout_event(snapshot, action_for(scene, key), event, seed=0)
    assert result["outcome"] == "success", result
    evidence = result["evidence"]
    assert evidence["event_id"] == "push-segment-v0"
    assert evidence["displacement_along_mm"] >= event["success_rule"]["segment_fraction"] * 80
    assert evidence["max_contact_n"] <= event["success_rule"]["max_contact_force_n"]
    assert evidence["first_contact_tick"] is not None and evidence["first_action_tick"] <= evidence["first_success_tick"]


def test_the_followup_policy_forces_the_main_decision_and_disables_the_gates(e0_seed5):
    from robo_jev.harness.robot import RobotHarness, load_harness_config

    scene = e0_seed5["scene"]
    key = "push:o0:+x:none:slow"
    request = RobotHarness(load_harness_config()).build_request(scene, None, None)
    policy = FollowupPolicy(Expert(), key)
    out = policy.act(request, None, scene)
    assert max(out["q_main"], key=out["q_main"].get) == candidate_id(key)
    assert out["q_done"] < 0.5 and out["q_instr"] > 0.5 and out["q_observe"] < 0.5 and out["q_retry"] > 0.5
    assert policy.version == EVENTS["followup_policy_version"]


# --------------------------------------------------------------------------
# 신뢰구간
# --------------------------------------------------------------------------


def test_the_wilson_interval_is_censoring_aware_and_shrinks_with_evidence():
    low8, high8 = wilson_interval(8, 8, z=1.0)
    low4, high4 = wilson_interval(4, 4, z=1.0)
    assert high8 == pytest.approx(1.0) and low8 > low4 > 0.5
    low, high = wilson_interval(0, 8, z=1.0)
    assert low == pytest.approx(0.0) and 0 < high < 0.3
    assert wilson_interval(0, 0, z=1.0) == (0.0, 1.0)


# --------------------------------------------------------------------------
# 키프레임 선택 — 라벨·결과를 보지 않는다
# --------------------------------------------------------------------------


def synthetic_record(mains, *, gates=None, versions=None, moved=(), stops=()) -> dict:
    """채택 주 결정·게이트·목표 버전·이동 사건·정지만 있는 레코드 (키프레임 선택이 보는 전부)."""
    ticks = []
    for index, main in enumerate(mains):
        ticks.append({
            "t": index, "sim_ms": index * 100,
            "request": {"state": {"goal": {"version": (versions or {}).get(index, 1)},
                                  "events": [{"kind": "object_moved", "object": "o1"}] if index in moved else []}},
            "adopted": {"main": main, "stop": index in stops},
            "usage": {"gate": (gates or {}).get(index)},
            "labels": [{"question_id": "q_main", "kind": "valid_set", "candidate_ids": ["SECRET"]}],
        })
    return {"episode_id": "ep-test", "ticks": ticks}


def test_keyframes_are_event_ticks_first_then_random_fill_with_inclusion_counts():
    from robo_jev.sim.label import select_keyframes

    mains = ["c1"] * 40
    mains[12] = "c2"  # 전환
    for index in range(13, 40):
        mains[index] = "c2"
    record = synthetic_record(mains, versions={index: (2 if index >= 25 else 1) for index in range(40)}, moved={30}, stops={18, 19})
    keyframes = select_keyframes(record, {"keyframes": {"per_episode": 5}})
    assert len(keyframes) == 5
    indices = {frame["index"]: frame for frame in keyframes}
    assert 0 in indices and indices[0]["kind"] == "switch"        # 첫 채택
    assert 12 in indices and indices[12]["kind"] == "switch"
    assert 25 in indices and indices[25]["kind"] == "instruction"
    assert 30 in indices and indices[30]["kind"] == "moved"
    assert 18 in indices and indices[18]["kind"] == "stop" and 19 not in indices  # 정지 구간의 첫 틱만
    inclusion = keyframes[0]["inclusion"]
    assert inclusion["switch"] == {"available": 2, "included": 2}
    assert inclusion["instruction"] == {"available": 1, "included": 1}
    assert inclusion["moved"] == {"available": 1, "included": 1}
    assert inclusion["stop"] == {"available": 1, "included": 1}
    assert inclusion["random"]["included"] == 0
    assert [frame["index"] for frame in keyframes] == sorted(frame["index"] for frame in keyframes)

    quiet = synthetic_record(["c1"] * 30)
    keyframes = select_keyframes(quiet, {"keyframes": {"per_episode": 5}})
    kinds = [frame["kind"] for frame in keyframes]
    assert kinds.count("switch") == 1 and kinds.count("random") == 4
    assert select_keyframes(quiet, {"keyframes": {"per_episode": 5}}) == keyframes  # 결정적


def test_gate_ticks_are_not_keyframes_and_labels_are_never_read():
    from robo_jev.sim.label import select_keyframes

    record = synthetic_record(["c1"] * 20 + ["hold"] * 11, gates={index: "done" for index in range(20, 31)})
    keyframes = select_keyframes(record, {"keyframes": {"per_episode": 5}})
    assert all(frame["index"] < 20 for frame in keyframes)
    stripped = copy.deepcopy(record)
    for tick in stripped["ticks"]:
        tick.pop("labels")
    assert select_keyframes(stripped, {"keyframes": {"per_episode": 5}}) == keyframes


# --------------------------------------------------------------------------
# 후보 선택 — commitment + 전문가 선택 + 기하 다양성
# --------------------------------------------------------------------------


def test_rollout_candidates_include_the_commitment_and_the_expert_choice_and_spread_geometrically():
    from test_harness import harness, obj, observation

    from robo_jev.sim.label import choose_rollout_candidates

    crowd = [obj(f"o{index}", (140 + 30 * index, -300 + 70 * index, -80)) for index in range(10)]
    request = harness().build_request(observation(objects=crowd), None, None)
    tick = {key: value for key, value in request.items() if key != "harness"}
    keys = {entry["id"]: entry["key"] for entry in tick["request"]["candidates"]["q_main"]}
    committed = next(cid for cid, key in keys.items() if key.startswith("push:o3:"))
    choice = next(cid for cid, key in keys.items() if key.startswith("grasp:o0:top:zoneL:slow"))
    commitment = {"action_ref": committed, "key": keys[committed], "phase": "approach", "held_ticks": 1, "last_switch_tick": 0}

    chosen = choose_rollout_candidates(tick, commitment, choice, k=8)
    assert len(chosen) == 8 and len(set(chosen)) == 8
    assert committed in chosen and choice in chosen
    assert all(keys[cid] not in ("observe", "hold", "replan") for cid in chosen)
    # 프로파일만 다른 후보(같은 접근점)보다 다른 대상·방향이 먼저다.
    stems = {keys[cid].rsplit(":", 1)[0] for cid in chosen}
    assert len(stems) == 8
    assert choose_rollout_candidates(tick, commitment, choice, k=8) == chosen  # 결정적
    # 라벨이 있어도 같다 — 정답을 보지 않는다.
    labelled = copy.deepcopy(tick)
    labelled["labels"] = [{"question_id": "q_main", "kind": "valid_set", "candidate_ids": [chosen[-1]]}]
    assert choose_rollout_candidates(labelled, commitment, choice, k=8) == chosen
    assert choose_rollout_candidates(tick, None, None, k=3)  # 강제 후보가 없어도 채운다


# --------------------------------------------------------------------------
# 주 결정 라벨 — 합성 결과표 (docs/08 §7)
# --------------------------------------------------------------------------


def labelled_tick(*, commitment=None, admissible=("c1", "c2", "c3"), choice="c1", rule="expert-e0.2/goal_grasp", confidence="high") -> dict:
    ids = [f"c{index}" for index in range(1, 10)] + ["ch", "co", "cr"]
    keys = {f"c{index}": f"grasp:o{index}:top:zoneL:slow" for index in range(1, 10)}
    keys.update(ch="hold", co="observe", cr="replan")
    return {
        "t": 7, "sim_ms": 700, "observed_at_ms": 700, "obs_age_ms": {"geom": 0, "proprio": 0},
        "request": {
            "state": {"robot": {"ee_pose_mm": [0, 0, 100]}, "objects": [], "zones": [], "scene": {}},
            "exec_history": "none",
            "commitment": {"action_ref": commitment, "key": keys[commitment], "phase": "approach", "held_ticks": 3} if commitment else None,
            "candidates": {"q_main": [{"id": cid, "action_ref": cid, "key": keys[cid], "desc": cid} for cid in ids]},
        },
        "labels": [{"question_id": "q_main", "kind": "valid_set", "candidate_ids": [choice],
                    "semantic_admissible": list(admissible), "source": "expert_v0", "rule": rule, "label_confidence": confidence}],
    }


def table(**per_candidate) -> dict:
    out = {}
    for cid, (s, f, censored) in per_candidate.items():
        out[cid] = {"s": s, "f": f, "censored": censored, "seeds": [1] * s + [0] * f + [-1] * censored, "rollout_seeds": list(range(s + f + censored))}
    return out


def check_contract(tick: dict, label: dict) -> None:
    record = {"schema_version": "stream-v0", "episode_id": "ep-x",
              "prefix": {"instructions": [{"version": 1, "t_ms": 0, "text": "x"}], "question_set": "qs-v0"},
              "ticks": [{**tick, "labels": [label]}]}
    validate_record(record)


def test_a_kept_commitment_within_epsilon_and_above_the_lower_bound_is_the_unique_answer():
    from robo_jev.sim.label import label_main_decision

    tick = labelled_tick(commitment="c1")
    label = label_main_decision(tick, table(c1=(7, 1, 0), c2=(8, 0, 0), c3=(2, 6, 0)), EVENTS)
    assert label["candidate_ids"] == ["c1"] and label["rollout_reason"] == "commitment_kept"
    assert label["label_confidence"] == "high"
    assert label["semantic_admissible"] == ["c1", "c2", "c3"]
    assert label["event_results"]["c1"] == {"s": 7, "f": 1, "censored": 0, "seeds": [1] * 7 + [0]}
    assert set(label["unknown"]) == {f"c{index}" for index in range(4, 10)}  # rollout하지 않은 결합 후보만
    assert "ch" not in label["unknown"] and label["source"] == "rollout_v0" and "wilson" in label["rule"]
    check_contract(tick, label)

    # commitment가 ε 밖이면 지키지 않는다: 최고 후보가 유일 정답이다.
    label = label_main_decision(tick, table(c1=(4, 4, 0), c2=(8, 0, 0)), EVENTS)
    assert label["candidate_ids"] == ["c2"] and label["rollout_reason"] == "performance_unique"


def test_a_tie_becomes_an_allowed_set_and_an_inadmissible_winner_is_ignored():
    from robo_jev.sim.label import label_main_decision

    tick = labelled_tick(commitment=None, admissible=("c1", "c2"))
    label = label_main_decision(tick, table(c1=(8, 0, 0), c2=(8, 0, 0), c3=(8, 0, 0), c4=(1, 7, 0)), EVENTS)
    assert label["candidate_ids"] == ["c1", "c2"] and label["rollout_reason"] == "performance_allowed_set"
    assert label["label_confidence"] == "high"
    assert "c3" not in label["candidate_ids"]  # 적합하지 않은 후보는 성과가 좋아도 정답이 아니다
    assert "c3" not in label["unknown"] and "c4" not in label["unknown"]  # rollout했으므로 미확인이 아니다
    check_contract(tick, label)


def test_all_failed_falls_back_to_the_expert_choice_with_low_confidence():
    from robo_jev.sim.label import label_main_decision

    tick = labelled_tick(commitment="c1")
    label = label_main_decision(tick, table(c1=(0, 8, 0), c2=(0, 8, 0)), EVENTS)
    assert label["candidate_ids"] == ["c1"] and label["label_confidence"] == "low" and label["rollout_reason"] == "all_failed"
    check_contract(tick, label)


def test_censoring_lowers_the_confidence_and_shrinks_the_effective_trials():
    from robo_jev.sim.label import label_main_decision

    tick = labelled_tick(commitment=None, admissible=("c1", "c2"))
    heavy = label_main_decision(tick, table(c1=(2, 0, 6)), EVENTS)
    assert heavy["candidate_ids"] == ["c1"] and heavy["label_confidence"] == "low"
    assert heavy["event_results"]["c1"]["censored"] == 6
    some = label_main_decision(tick, table(c2=(5, 1, 2)), EVENTS)
    assert some["candidate_ids"] == ["c2"] and some["label_confidence"] == "medium"
    check_contract(tick, heavy)


def test_weak_evidence_keeps_a_plural_set_with_low_confidence():
    from robo_jev.sim.label import label_main_decision

    tick = labelled_tick(commitment=None)
    label = label_main_decision(tick, table(c1=(3, 5, 0), c2=(3, 5, 0), c3=(1, 7, 0)), EVENTS)
    assert label["candidate_ids"] == ["c1", "c2"] and label["label_confidence"] == "low"
    assert label["rollout_reason"] == "weak_evidence"


def test_a_gate_tick_keeps_its_rule_label_but_carries_the_event_results():
    from robo_jev.sim.label import label_main_decision

    tick = labelled_tick(commitment=None, admissible=("ch",), choice="ch", rule="expert-e0.2/goal_done")
    label = label_main_decision(tick, table(c1=(8, 0, 0)), EVENTS)
    assert label["candidate_ids"] == ["ch"] and label["rollout_reason"] == "gate:goal_done"
    assert label["event_results"]["c1"]["s"] == 8 and "c1" not in label["unknown"]
    check_contract(tick, label)


@pytest.mark.parametrize(
    ("reason", "admissible"),
    [("way_retry_blocked", ("c1",)), ("not_executable", ("c1",)), ("goal_candidate_missing", ())],
)
def test_a_retry_blocked_or_degenerate_tick_keeps_its_rule_label_like_a_gate_tick(reason, admissible):
    """하네스가 한 틱 동안 재시도를 막았거나(`way_retry_blocked`) 실행기 사정으로 고를 수 없거나(`not_executable`)
    목표를 실현할 후보가 목록에 없는(`goal_candidate_missing`) 틱의 정답은 규칙의 `hold`다. rollout은 그 차단을
    모른 채(`rollout_event`는 `history=None`으로 시작한다) 후보를 성공시키므로 결과로 라벨을 바꾸면 안 된다."""
    from robo_jev.sim.label import _GATE_REASONS, label_main_decision

    assert reason in _GATE_REASONS
    tick = labelled_tick(commitment=None, admissible=admissible, choice="ch", rule=f"expert-e0.2/{reason}", confidence="low")
    label = label_main_decision(tick, table(c1=(8, 0, 0), c2=(7, 1, 0)), EVENTS)
    assert label["candidate_ids"] == ["ch"] and label["rollout_reason"] == f"gate:{reason}"
    assert label["label_confidence"] == "low"  # 규칙 라벨의 신뢰도 그대로
    assert label["semantic_admissible"] == list(admissible)
    assert label["event_results"]["c1"]["s"] == 8 and "c1" not in label["unknown"]
    check_contract(tick, label)


def test_without_rollout_evidence_the_label_is_the_expert_choice_marked_low():
    from robo_jev.sim.label import label_main_decision

    tick = labelled_tick(commitment="c1")
    label = label_main_decision(tick, {}, EVENTS)
    assert label["candidate_ids"] == ["c1"] and label["label_confidence"] == "low"
    assert set(label["unknown"]) == {f"c{index}" for index in range(2, 10)}


def test_summarise_results_keeps_per_seed_outcomes_and_censoring_reasons():
    from robo_jev.sim.label import summarise_results

    def result(cid, seed, outcome, reason=None):
        return {"outcome": outcome, "reason": reason, "evidence": {"action_ref": cid, "rollout_seed": seed, "event_id": "grasp-lift-v0"}}

    summary = summarise_results([
        result("c1", 1, "failure", "horizon"), result("c1", 0, "success"), result("c1", 2, "censored", "wall_time_limit"),
        result("c2", 0, "success"),
    ])
    assert summary["c1"] == {"s": 1, "f": 1, "censored": 1, "seeds": [1, 0, -1], "rollout_seeds": [0, 1, 2],
                             "censored_reasons": ["wall_time_limit"], "event_id": "grasp-lift-v0"}
    assert summary["c2"]["seeds"] == [1]
