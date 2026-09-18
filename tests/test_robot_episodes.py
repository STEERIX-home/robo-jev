"""로봇 에피소드 생성기 검사 (docs/08 §8·§9, docs/04 §5, Task 3c-1).

두 갈래다. 물리 없이 보는 것(장면 계열·split·seed 일정·집계·manifest 형식)과, 실제 환경에서 2편
(E0 seed 17 + E1 seed 29)을 만들어 보는 smoke — 레코드가 계약을 지키고, 실행 이력이 직전 틱에
실제로 적용된 것(라벨이 아니라)이며, 라벨이 입력 영역에 새지 않는지. 40편 배치는 검사가 아니라
스크립트다(`scripts/generate_episodes.py`).
"""

import copy
import json

import pytest
import yaml
from helpers import SIM_CONFIG, all_keys

from robo_jev.contracts import NON_INPUT_FIELDS, QUESTION_SET_V0, model_input, validate_record
from robo_jev.data.episode import aggregate
from robo_jev.data.robot_episodes import (
    GENERATOR_VERSION,
    MANIFEST_VERSION,
    build_manifest,
    episode_id,
    family_id,
    family_signature,
    generate_episode,
    load_generator_config,
    origin_group,
    run,
    seed_schedule,
    write_episode,
)
from robo_jev.data.split import SplitPolicy, assign_split
from robo_jev.data.validate import validate_dataset
from robo_jev.harness.robot import parse_exec_history
from robo_jev.sim.expert import Expert
from robo_jev.sim.scene import build_plan

CONFIG = load_generator_config()
SIM = yaml.safe_load(SIM_CONFIG.read_text(encoding="utf-8"))
SMOKE = (("E0", 17), ("E1", 29))


# --------------------------------------------------------------------------
# 장면 계열과 split — 생성 전에
# --------------------------------------------------------------------------


def test_the_family_is_the_structural_signature_of_the_scene_plan():
    plan = build_plan(SIM, 100, "E1")
    signature = family_signature(plan)
    assert signature["objects"] == len(plan.objects)
    assert sum(signature["shapes"].values()) == len(plan.objects)
    assert signature["zones"] == sorted(zone.id for zone in plan.zones)
    boxes = signature["shapes"].get("box", 0)
    cylinders = signature["shapes"].get("cylinder", 0)
    letters = "".join(zone[len("zone"):] for zone in signature["zones"])
    assert family_id(plan) == f"family-n{len(plan.objects)}-b{boxes}c{cylinders}-z{letters}"
    assert origin_group("E1", plan) == f"robot/E1/{family_id(plan)}"

    # 자세·색·일정이 달라도 구조가 같으면 같은 계열이다.
    same = [seed for seed in range(100, 140) if family_id(build_plan(SIM, seed, "E1")) == family_id(plan)]
    assert 100 in same
    for seed in same[1:3]:
        assert build_plan(SIM, seed, "E1").to_json() != plan.to_json()


def test_split_is_assigned_from_the_family_before_generation_and_the_holdout_is_ood():
    policy = SplitPolicy.from_config(CONFIG["split"])
    holdout = CONFIG["split"]["holdout_prefixes"]
    assert holdout == [origin_group("E1", build_plan(SIM, CONFIG["seeds"]["base"], "E1"))]
    assert assign_split(holdout[0], policy) == "ood"
    splits = {assign_split(origin_group(profile, build_plan(SIM, seed, profile)), policy) for profile, seed in seed_schedule(CONFIG, 40)}
    assert "ood" in splits and "train" in splits


def test_the_generator_config_names_every_config_it_digests_and_the_generator_requires_them():
    """생성 설정은 지문에 드는 설정 파일(장면·하네스·전문가·사건)의 경로를 전부 이름으로 적는다 — 키가 빠지면
    기본 경로로 조용히 대신하지 않고 생성기가 거절한다."""
    from robo_jev.data.robot_episodes import config_paths

    assert CONFIG["events_config"] == "configs/sim/events.yaml"
    paths = config_paths(CONFIG)
    assert paths == {
        "sim_config": CONFIG["sim_config"], "harness_config": CONFIG["harness_config"],
        "expert_config": CONFIG["expert_config"], "events_config": CONFIG["events_config"],
    }
    for key in ("sim_config", "harness_config", "expert_config", "events_config"):
        missing = {k: v for k, v in CONFIG.items() if k != key}
        with pytest.raises(ValueError, match=key):
            config_paths(missing)
        with pytest.raises(ValueError, match=key):
            generate_episode("E0", 17, policy=None, expert=None, config=missing, max_ticks=1)
        with pytest.raises(ValueError, match=key):
            run(missing, 1, None)


def test_the_seed_schedule_is_deterministic_and_a_prefix_of_a_longer_one():
    short = seed_schedule(CONFIG, 6)
    assert short == [("E0", 100), ("E1", 100), ("E0", 101), ("E1", 101), ("E0", 102), ("E1", 102)]
    assert seed_schedule(CONFIG, 40)[:6] == short
    assert len({episode_id(profile, seed) for profile, seed in seed_schedule(CONFIG, 40)}) == 40


# --------------------------------------------------------------------------
# smoke — E0 seed 17 + E1 seed 29
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def smoke(tmp_path_factory):
    """두 에피소드를 한 번 만들어 이 모듈의 검사가 나눠 본다 (합쳐서 20초 안)."""
    expert = Expert()
    out = tmp_path_factory.mktemp("d1-robot")
    records = {}
    for profile, seed in SMOKE:
        record = generate_episode(profile, seed, policy=expert, expert=expert, config=CONFIG)
        write_episode(record, out)
        records[(profile, seed)] = record
    manifest = build_manifest(out, CONFIG, batch_wall_s=1.0)
    return {"records": records, "out": out, "manifest": manifest, "expert": expert}


def test_smoke_records_pass_the_contract_and_the_automatic_qa(smoke):
    for record in smoke["records"].values():
        validate_record(record)
        assert record["schema_version"] == "stream-v0"
        assert len(record["ticks"]) >= 10
    report = validate_dataset(list(smoke["records"].values()))
    assert report["invalid_records"] == 0 and report["errors"] == []
    assert report["labels_without_source"] == 0


def test_smoke_e0_completes_with_a_one_second_tail(smoke):
    """꼬리는 안정된 완료 1초다: 마지막 11틱이 모두 done 게이트다."""
    record = smoke["records"][("E0", 17)]
    outcome = record["provenance"]["outcome"]
    assert outcome["done"] is True and outcome["target_inside_zone"] is True and outcome["holding"] is None
    # 놓았다 = 손가락이 실제로 열렸다. 마지막 관측의 그리퍼 폭을 결과에 적는다.
    assert isinstance(outcome["gripper_mm"], int) and outcome["gripper_mm"] > 40
    assert outcome["terminated"] == "done_tail"
    tail = CONFIG["episode"]["tail_ticks_after_done"]
    assert len(record["ticks"]) == outcome["done_tick"] + 1 + tail
    assert all(tick["usage"]["gate"] == "done" for tick in record["ticks"][-(tail + 1):])
    assert record["ticks"][-(tail + 2)]["usage"]["gate"] != "done"
    assert outcome["first_done_tick"] == outcome["done_tick"]


def test_smoke_records_are_split_by_family_and_carry_provenance_and_versions(smoke):
    policy = SplitPolicy.from_config(CONFIG["split"])
    for (profile, seed), record in smoke["records"].items():
        assert record["episode_id"] == episode_id(profile, seed)
        assert record["origin_group"].startswith(f"robot/{profile}/family-")
        assert record["split"] == assign_split(record["origin_group"], policy)
        provenance = record["provenance"]
        assert provenance["seed"] == seed and provenance["profile"] == profile
        assert provenance["generator"] == GENERATOR_VERSION and provenance["label_source"] == "expert_v0"
        assert provenance["policy"]["name"] == "Expert"
        for name in ("harness", "controller", "rules", "serializer", "extractor", "expert", "generator", "sim"):
            assert record["versions"][name]
        assert record["versions"]["expert"] == smoke["expert"].version
        assert record["prefix"]["instructions"][0]["version"] == 1


def test_smoke_model_output_is_the_policys_raw_answer_and_labels_are_the_experts(smoke):
    """D1에서는 정책이 전문가 자신이다: `model_output`은 10답 그대로이고 라벨은 그 답에서 나온다."""
    expert = smoke["expert"]
    for record in smoke["records"].values():
        for tick in record["ticks"]:
            assert set(tick["model_output"]) == set(QUESTION_SET_V0)
            again = expert.act(tick, None, None)  # 요청(모델 입력)만으로 같은 답이 다시 나온다
            assert {q: again[q] for q in QUESTION_SET_V0} == tick["model_output"]
            labels = {label["question_id"]: label for label in tick["labels"]}
            assert all(label["source"] == "expert_v0" for label in labels.values())
            assert labels["q_main"]["candidate_ids"] == [max(again["q_main"], key=again["q_main"].get)]
            for question_id in ("q_done", "q_instr", "q_observe", "q_retry", "q_stop"):
                assert labels[question_id]["answer"] is (again[question_id] >= 0.5)
                assert labels[question_id]["rule"]
            commitment = tick["request"]["commitment"]
            for question_id in ("q_gripper", "q_path", "q_speed", "q_force"):
                if commitment is None:
                    assert question_id not in labels  # 마스킹
                else:
                    assert labels[question_id]["conditioned_on"] == f"{commitment['action_ref']}/{commitment['phase']}"


def assert_history_follows_the_executed_tick(record: dict) -> int:
    """직전 틱의 채택 결과·ACK와 이력 한 줄을 대조한다. 라벨과 채택이 갈린 틱 수를 돌려준다."""
    differing = 0
    ticks = record["ticks"]
    assert ticks[0]["request"]["exec_history"] == "none"
    for previous, tick in zip(ticks, ticks[1:]):
        history = parse_exec_history(tick["request"]["exec_history"])
        adopted, ack = previous["adopted"], previous["ack"]
        assert history["main"] == adopted["main"]
        assert history["phase"] == str(adopted["phase"])
        assert history["gripper"] == adopted["gripper"]
        assert history["stop"] == str(int(bool(adopted["stop"])))
        assert history["gate"] == str(previous["usage"]["gate"] or "none")
        assert history["ack"] == ("ok" if ack["applied"] else str(ack["reason"]))
        if ack["applied"] and ack.get("path") == "observe":
            assert history["path"] == "observe"
        elif adopted["path"] is not None:
            assert history["path"] == adopted["path"]
        label = next(item for item in previous["labels"] if item["question_id"] == "q_main")
        if label["candidate_ids"][0] != adopted["main"]:
            differing += 1
            assert history["main"] != label["candidate_ids"][0]
    return differing


def test_smoke_exec_history_is_the_previous_ticks_applied_ack(smoke):
    """docs/08 §3.3: 실행 이력은 직전 틱에 실제로 채택·실행된 것이다."""
    for record in smoke["records"].values():
        assert_history_follows_the_executed_tick(record)


class HoldPolicy:
    """DAgger 모양의 대역 정책: 전문가와 같은 형식으로 답하되 주 결정은 언제나 hold다."""

    version = "stand-in-hold"

    def __init__(self, expert: Expert) -> None:
        self.expert = expert

    def act(self, request, commitment, observation=None):
        answers = self.expert.act(request, commitment, observation)
        candidates = request["request"]["candidates"]["q_main"]
        hold = next(entry["id"] for entry in candidates if entry["key"] == "hold")
        rest = [entry["id"] for entry in candidates if entry["id"] != hold]
        answers["q_main"] = {hold: 0.9, **{cid: round(0.1 / len(rest), 6) for cid in rest}}
        return answers


def test_a_stand_in_policy_keeps_its_own_executed_history_while_labels_come_from_the_expert(smoke):
    """DAgger의 형태(docs/08 §7): 모델 raw 출력·채택·ACK는 그대로, 전문가 답은 labels에만.
    실행 이력은 채택된 hold를 말하고 라벨의 파지를 말하지 않는다."""
    expert = smoke["expert"]
    record = generate_episode("E0", 17, policy=HoldPolicy(expert), expert=expert, config=CONFIG, max_ticks=20)
    validate_record(record)
    assert record["provenance"]["policy"] == {"name": "HoldPolicy", "version": "stand-in-hold"}
    assert record["provenance"]["outcome"]["done"] is False
    differing = assert_history_follows_the_executed_tick(record)
    assert differing >= 10
    for tick in record["ticks"]:
        hold = next(entry["id"] for entry in tick["request"]["candidates"]["q_main"] if entry["key"] == "hold")
        assert max(tick["model_output"]["q_main"], key=tick["model_output"]["q_main"].get) == hold
        assert tick["adopted"]["main"] == hold
        label = next(item for item in tick["labels"] if item["question_id"] == "q_main")
        assert label["candidate_ids"] != [hold]  # 전문가는 파지를 고른다
        assert label["source"] == "expert_v0"


def test_smoke_labels_never_reach_the_input_area(smoke):
    for record in smoke["records"].values():
        served = model_input(record)
        assert not (all_keys(served) & set(NON_INPUT_FIELDS))
        for tick in served["ticks"]:
            assert set(tick["request"]) == {"state", "exec_history", "commitment", "candidates"}
        assert "labels" not in json.dumps(served, ensure_ascii=False)


def test_smoke_gripper_labels_allow_both_states_around_a_transition(smoke):
    record = smoke["records"][("E0", 17)]
    desired = []
    for tick in record["ticks"]:
        label = next((item for item in tick["labels"] if item["question_id"] == "q_gripper"), None)
        desired.append(label["candidate_ids"] if label else None)
    transitions = [
        index for index in range(1, len(desired))
        if desired[index] and desired[index - 1] and desired[index] != desired[index - 1]
    ]
    assert transitions, "E0 에피소드에는 닫기·열기 전환이 있어야 한다"
    tolerant = [index for index, value in enumerate(desired) if value == ["open", "closed"]]
    assert tolerant
    assert any(abs(index - at) <= CONFIG["episode"].get("tail_ticks_after_done", 10) for index in tolerant for at in transitions)


def test_smoke_usage_and_evidence_stay_out_of_the_input(smoke):
    record = smoke["records"][("E1", 29)]
    tick = record["ticks"][0]
    assert set(tick["usage"]) >= {"gate", "switch", "records", "executor", "applied"}
    assert record["evidence"]["expert"]["version"] == smoke["expert"].version
    assert len(record["evidence"]["expert"]["ticks"]) == len(record["ticks"])
    assert record["evidence"]["scene_plan"]["seed"] == 29
    served = model_input(record)
    assert "usage" not in json.dumps(served, ensure_ascii=False)
    assert "expert_meta" not in json.dumps(served, ensure_ascii=False)


# --------------------------------------------------------------------------
# 집계와 manifest
# --------------------------------------------------------------------------


def test_aggregate_counts_gates_stops_and_switches_per_episode(smoke):
    records = list(smoke["records"].values())
    counts = aggregate(records)
    assert counts["episodes"] == 2
    assert counts["gates"].get("done", 0) >= CONFIG["episode"]["tail_ticks_after_done"] + 1
    assert set(counts) >= {"ticks", "questions", "labels", "splits", "gates", "stops", "switches", "main_changes", "conflicts", "per_episode"}
    per = {entry["episode_id"]: entry for entry in counts["per_episode"]}
    e0 = per[episode_id("E0", 17)]
    assert e0["switches"] >= 1 and e0["ticks"] == len(smoke["records"][("E0", 17)]["ticks"])
    # 채택된 주 결정이 실제로 바뀐 틱: 첫 채택 … 파지 → done 게이트의 hold — 게이트 꼬리는 한 번만 센다.
    assert 1 <= e0["main_changes"] <= 3
    assert sum(entry["switches"] for entry in per.values()) == counts["switches"]
    assert sum(entry["stops"] for entry in per.values()) == counts["stops"]
    assert sum(entry["main_changes"] for entry in per.values()) == counts["main_changes"]


def test_the_manifest_has_the_documented_schema(smoke):
    manifest = smoke["manifest"]
    out = smoke["out"]
    assert json.loads((out / "manifest.json").read_text(encoding="utf-8")) == manifest
    assert manifest["version"] == MANIFEST_VERSION and manifest["generator"] == GENERATOR_VERSION
    assert manifest["episodes"] == 2 and manifest["config_sha256"] and manifest["config"] == CONFIG
    assert set(manifest["ticks"]) == {"total", "mean", "p95", "max"}
    assert set(manifest["per_profile"]) == {"E0", "E1"}
    for entry in manifest["per_profile"].values():
        assert set(entry) == {"episodes", "done", "done_rate", "ticks_mean", "ticks_p95", "wall_s_mean"}
    assert sum(manifest["splits"].values()) == 2
    assert manifest["holdout_prefixes"] == CONFIG["split"]["holdout_prefixes"]
    assert manifest["versions"]["expert"] == [smoke["expert"].version]
    assert set(manifest["bytes"]) == {"total", "per_episode_mean_mb", "per_tick_mean_kb"}
    assert set(manifest["timing"]) >= {"wall_s_per_episode_mean", "episodes_per_hour", "projected_hours_for_target", "target_episodes"}
    assert manifest["timing"]["episodes_per_hour"] > 0
    assert set(manifest["decisions"]) == {"gates", "stops", "switches", "main_changes", "conflicts", "per_episode"}
    assert len(manifest["files"]) == 2
    for entry in manifest["files"]:
        assert set(entry) == {"path", "episode_id", "profile", "seed", "split", "origin_group", "done", "ticks", "bytes", "sha256"}
        assert (out / entry["path"]).is_file()


def test_resume_skips_seeds_that_already_have_an_episode(tmp_path, monkeypatch):
    """`--resume`는 이미 있는 seed를 건너뛴다. 새로 만드는 쪽만 환경을 돌린다."""
    import robo_jev.data.robot_episodes as module

    calls = []

    def fake_generate(profile, seed, **kwargs):
        calls.append((profile, seed))
        record = copy.deepcopy(_SMOKE_TEMPLATE)
        record["episode_id"] = episode_id(profile, seed)
        record["provenance"]["profile"], record["provenance"]["seed"] = profile, seed
        return record

    monkeypatch.setattr(module, "generate_episode", fake_generate)
    monkeypatch.setattr(module, "validate_record", lambda record: None)

    class FakeEnv:
        def __init__(self, *args, **kwargs):
            pass

        def close(self):
            pass

    import robo_jev.sim.environment as environment

    monkeypatch.setattr(environment, "Environment", FakeEnv)
    run(CONFIG, 2, tmp_path)
    assert calls == [("E0", 100), ("E1", 100)]
    calls.clear()
    manifest = run(CONFIG, 3, tmp_path, resume=True)
    assert calls == [("E0", 101)]
    assert manifest["run"] == {"requested": 3, "produced": 1, "skipped": 2, "resume": True}
    assert manifest["episodes"] == 3


_SMOKE_TEMPLATE = {
    "schema_version": "stream-v0",
    "episode_id": "ep-x",
    "origin_group": "robot/E0/family-n3-b2c1-zFL",
    "split": "train",
    "versions": {"harness": "h", "controller": "c", "rules": "r", "serializer": "s", "extractor": "p", "expert": "e"},
    "prefix": {"instructions": [{"version": 1, "t_ms": 0, "text": "x"}], "question_set": "qs-v0"},
    "ticks": [
        {
            "t": 0, "sim_ms": 0, "observed_at_ms": 0, "obs_age_ms": {"geom": 0, "proprio": 0},
            "request": {"state": {}, "exec_history": "none", "commitment": None, "candidates": {"q_main": [{"id": "c1", "key": "hold"}]}},
            "adopted": {"main": "c1", "switch": True, "stop": False}, "usage": {"gate": None, "records": {}},
        }
    ],
    "provenance": {"profile": "E0", "seed": 0, "timing": {"wall_s": 0.5}, "outcome": {"done": False}},
}
