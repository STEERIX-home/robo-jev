"""로봇 에피소드 생성기 검사 (docs/08 §8·§9, docs/04 §5, Task 3c-1).

두 갈래다. 물리 없이 보는 것(장면 계열·split·seed 일정·집계·manifest 형식)과, 실제 환경에서 2편
(E0 seed 17 + E1 seed 29)을 만들어 보는 smoke — 레코드가 계약을 지키고, 실행 이력이 직전 틱에
실제로 적용된 것(라벨이 아니라)이며, 라벨이 입력 영역에 새지 않는지. 40편 배치는 검사가 아니라
스크립트다(`scripts/generate_episodes.py`).
"""

import copy
import hashlib
import json
from collections import Counter

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
    plan_concepts,
    plan_tags,
    run,
    seed_schedule,
    write_episode,
)
from robo_jev.data.split import OOD_SPLITS, SplitPolicy, assign_split, ood_split
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
    # origin group = 장면 계열 + 목표 생성 계열(지시의 목표 영역) — 영역별 holdout이 한 group을 두 split에 걸치지 않는다.
    assert origin_group("E1", plan) == f"robot/E1/{family_id(plan)}/goal-{plan.instructions[0].zone}"

    # 자세·색·일정이 달라도 구조가 같으면 같은 계열이다.
    same = [seed for seed in range(100, 140) if family_id(build_plan(SIM, seed, "E1")) == family_id(plan)]
    assert 100 in same
    for seed in same[1:3]:
        assert build_plan(SIM, seed, "E1").to_json() != plan.to_json()


def test_split_is_assigned_from_the_family_before_generation_and_the_holdout_is_ood():
    policy = SplitPolicy.from_config(CONFIG["split"])
    holdout = CONFIG["split"]["holdout_prefixes"]
    first_e1 = origin_group("E1", build_plan(SIM, CONFIG["seeds"]["base"], "E1"))
    assert first_e1.startswith(holdout[0] + "/goal-")
    assert assign_split(first_e1, policy) == ood_split(first_e1) and assign_split(first_e1, policy) in OOD_SPLITS
    splits = {assign_split(origin_group(profile, build_plan(SIM, seed, profile)), policy) for profile, seed in seed_schedule(CONFIG, 40)}
    assert splits & set(OOD_SPLITS) and "train" in splits


def test_the_zone_f_goal_family_and_the_third_instruction_variant_are_sealed_before_generation():
    """docs/04 §5 봉인 표(로봇): 지시의 목표 영역이 zoneF인 계열과 지시 문구 변형 3번(v1#2·v2#2)은 생성 전에 OOD로 간다 —
    태그는 계획에서 나오므로 에피소드를 돌리기 전에 안다. train은 zoneL·zoneR만 목표로 하고 변형 1·2번만 본다."""
    policy = SplitPolicy.from_config(CONFIG["split"])
    assert CONFIG["split"]["holdout_concepts"] == ["robot:goal-zone:zoneF"] and CONFIG["split"]["holdout_templates"] == ["v1#2", "v2#2"]
    seen = {"ood": 0, "zoneF": 0, "variant3": 0, "train_like": 0}
    for profile, seed in seed_schedule(CONFIG, 60):
        plan = build_plan(SIM, seed, profile)
        group, tags = origin_group(profile, plan), plan_tags(plan)
        assert plan_concepts(plan) == sorted({f"robot:goal-zone:{step.zone}" for step in plan.instructions})
        assert all(step.template in ("v1#0", "v1#1", "v1#2", "v2#0", "v2#1", "v2#2") for step in plan.instructions)
        split = assign_split(group, policy, tags)
        reasons = policy.holdout_reasons(group, tags)
        zone_f = any(step.zone == "zoneF" for step in plan.instructions)
        variant3 = any(step.template.endswith("#2") for step in plan.instructions)
        if zone_f:
            seen["zoneF"] += 1
            assert "concept:robot:goal-zone:zoneF" in reasons and split in OOD_SPLITS
        if variant3:
            seen["variant3"] += 1
            assert any(reason.startswith("template:v") for reason in reasons) and split in OOD_SPLITS
        if split in OOD_SPLITS:
            seen["ood"] += 1
            assert reasons
        else:
            seen["train_like"] += 1
            assert not reasons and not zone_f and not variant3
    assert seen["zoneF"] and seen["variant3"] and seen["train_like"] and seen["ood"] < 60


def test_no_origin_group_straddles_two_splits_and_the_sealed_share_lands_in_the_decided_band():
    """리뷰 1 I1·I2: 문구 변형은 origin group의 해시가 정하므로 같은 group의 에피소드는 같은 변형·같은 split이다(400편 일정 모의,
    E1 seed 243 포함 — 예외 없이 지어진다). 봉인(zoneF 목표 계열 + 변형 3번 + E1 계열 하나)이 보내는 OOD는 사용자가 정한
    ≈10~15 %(D1 규모의 에피소드 기준; 생성 비중 `goal_zone_weights`·`template_weights`가 맞춘다)이고 ood_dev/ood_test는 둘 다 쓰인다."""
    policy = SplitPolicy.from_config(CONFIG["split"])
    groups: dict[str, set[str]] = {}
    variants: dict[tuple[str, int], set[str]] = {}
    splits = Counter()
    for profile, seed in seed_schedule(CONFIG, 400):
        plan = build_plan(SIM, seed, profile)
        group = origin_group(profile, plan)
        split = assign_split(group, policy, plan_tags(plan))
        groups.setdefault(group, set()).add(split)
        for step in plan.instructions:  # v1·v2는 따로 정해지고, v2가 없는 에피소드(영역 밖의 다른 대상이 없다)도 있다
            variants.setdefault((group, step.version), set()).add(str(step.template))
        splits[split] += 1
    assert all(len(seen) == 1 for seen in groups.values()), [g for g, seen in groups.items() if len(seen) > 1]
    assert all(len(seen) == 1 for seen in variants.values()), [g for g, seen in variants.items() if len(seen) > 1]
    ood = sum(splits[name] for name in OOD_SPLITS)
    assert 0.10 <= ood / 400 <= 0.15, dict(splits)
    assert splits["ood_dev"] > 0 and splits["ood_test"] > 0 and splits["train"] > 200


def test_the_goal_zone_is_reweighted_without_disturbing_the_scene_stream():
    """`goal_zone_weights`는 주 난수의 균등 추첨을 seed 해시의 기각 표본으로 다시 가중한다: 결과는 비중에 비례하고, 비중이 최대인
    영역을 뽑은 seed의 계획(장면·일정·대상)은 비중이 없을 때와 같다 — zoneF를 뽑은 seed만 일부 옮겨진다."""
    from robo_jev.sim import scene as scene_module

    zones = tuple(scene_module.Zone(name, name, (0, 0, 1, 1)) for name in ("zoneL", "zoneR", "zoneF"))
    counts = Counter(scene_module._reweight_zone(seed % 3, zones, [48.0, 48.0, 4.0], seed).id for seed in range(30000))
    assert abs(counts["zoneF"] / 30000 - 0.04) < 0.01 and abs(counts["zoneL"] / 30000 - 0.48) < 0.02
    assert all(scene_module._reweight_zone(index, zones, [48.0, 48.0, 4.0], seed).id == zones[index].id for seed in range(200) for index in (0, 1))
    assert scene_module._reweight_zone(2, zones, None, 7).id == "zoneF"  # 비중이 없으면 그대로

    flat = copy.deepcopy(SIM)
    flat["instruction"]["goal_zone_weights"] = {"zoneL": 1, "zoneR": 1, "zoneF": 1}
    kept = moved = 0
    for seed in range(100, 220):
        for profile in ("E0", "E1"):
            weighted, uniform = build_plan(SIM, seed, profile), build_plan(flat, seed, profile)
            if uniform.instructions[0].zone != "zoneF":
                assert weighted.to_json() == uniform.to_json()
                kept += 1
            elif weighted.instructions[0].zone != "zoneF":
                moved += 1  # 목표 영역이 옮겨졌다: 장면의 구조(물체·형상·속성·영역)는 같다 (대상·v2·일정은 영역에 따라 달라질 수 있다)
                assert [(o.id, o.shape, o.half_size_mm, o.attributes) for o in weighted.objects] == [(o.id, o.shape, o.half_size_mm, o.attributes) for o in uniform.objects]
                assert weighted.zones == uniform.zones
    assert kept > 100 and moved > 0
    with pytest.raises(ValueError, match="goal_zone_weights"):
        broken = copy.deepcopy(SIM)
        broken["instruction"]["goal_zone_weights"] = {"zoneL": 1, "zoneR": 1}
        build_plan(broken, 100, "E1")


def test_a_target_inside_its_goal_zone_is_swapped_relocated_or_rezoned_in_that_order():
    """A6 이월 제약의 해결 순서(리뷰 1 M8): 영역은 두고 (1) 영역 밖의 다른 대상, (2) 없으면 대상을 영역 밖의 빈자리로(E0의
    평범한 물체 하나), (3) 그것도 안 되면 대상을 담지 않는 다른 영역 — 목표 영역의 분포를 흔들지 않는다."""
    from robo_jev.sim import scene as scene_module

    zone = scene_module.Zone("zoneL", "왼쪽", (-120, 150, 180, 330))
    other = scene_module.Zone("zoneF", "앞", (100, -80, 300, 80))
    inside = scene_module.SceneObject("o0", "box", "red", "빨간", (1, 0, 0, 1), (20, 20, 20), (0, 200, 20), 0.0, ())
    outside = scene_module.SceneObject("o1", "box", "blue", "파란", (0, 0, 1, 1), (20, 20, 20), (-200, -200, 20), 0.0, ())
    fix = scene_module._target_outside_zone
    assert fix(outside, zone, [inside, outside], (zone, other)) == (outside, zone)  # 위반이 없으면 그대로
    assert fix(inside, zone, [inside, outside], (zone, other), unit=0.3) == (outside, zone)  # (1) 다른 대상
    relocated = scene_module.SceneObject(**{**inside.__dict__, "pos_mm": (-100, -100, 20)})
    assert fix(inside, zone, [inside], (zone, other), unit=0.3, relocate=lambda obj, area: relocated) == (relocated, zone)  # (2) 옮김
    assert fix(inside, zone, [inside], (zone, other), unit=0.3, weights=[48.0, 4.0], relocate=lambda obj, area: None) == (inside, other)  # (3) 다른 영역
    assert fix(inside, zone, [inside], (zone,), unit=0.3) is None

    # 실제 옮김: 영역 밖이고 배치 규칙(간격·금지 물체 여유)을 지킨다.
    spec = dict(SIM["objects"])
    forbidden = scene_module.SceneObject("o2", "cylinder", "green", "초록", (0, 1, 0, 1), (26, 26, 40), (-100, -100, 40), 0.0, ("forbidden",))
    moved = scene_module._relocated_outside_zone(inside, zone, (inside, outside, forbidden), spec, seed=5)
    assert moved is not None and moved.id == "o0" and not scene_module._inside_zone(moved, zone)
    assert scene_module._relocated_outside_zone(inside, zone, (inside, outside, forbidden), spec, seed=5) == moved  # 결정적
    limit = scene_module._required_separation_mm(moved, forbidden, float(spec["min_separation_mm"]), scene_module._forbidden_margins_mm(spec))
    import math

    assert math.dist(moved.pos_mm[:2], forbidden.pos_mm[:2]) >= limit and math.dist(moved.pos_mm[:2], outside.pos_mm[:2]) >= spec["min_separation_mm"]


def test_instruction_variants_keep_the_structured_goal_and_the_constraint_marker():
    """세 변형은 표현만 다르다: 대상·영역·보호 물체는 같고, v1의 모든 변형에 규칙 기준군의 제약 표지가 있다. 변형은 계획의
    난수 소비 맨 뒤에서 고르므로 장면·일정은 변형 도입 전과 같다."""
    spec = SIM["instruction"]
    assert len(spec["v1_templates"]) == 3 and len(spec["v2_templates"]) == 3 and spec["template_weights"] == [47.5, 47.5, 5]
    assert all("{color}" in t and "{shape}" in t and "{zone}" in t and "{fragile}" in t for t in spec["v1_templates"])
    assert all("{color2}" in t and "{shape2}" in t and "{zone}" in t for t in spec["v2_templates"])
    assert all("건드리지 마라" in t for t in spec["v1_templates"])
    variants = set()
    for seed in range(100, 160):
        plan = build_plan(SIM, seed, "E1")
        for step in plan.instructions:
            variants.add(step.template)
            assert step.text and step.target and step.zone
        assert plan.instructions[0].template.startswith("v1#")
        # 같은 seed에서 변형 비중을 바꿔도(group 해시의 선택) 장면·일정은 같다.
        other = copy.deepcopy(SIM)
        other["instruction"]["template_weights"] = [10, 45, 45]
        again = build_plan(other, seed, "E1")
        assert [o.pos_mm for o in again.objects] == [o.pos_mm for o in plan.objects]
        assert again.disturbances == plan.disturbances
        assert [(s.target, s.zone, s.protected, s.sim_ms) for s in again.instructions] == [(s.target, s.zone, s.protected, s.sim_ms) for s in plan.instructions]
    assert variants >= {"v1#0", "v1#1", "v1#2"}


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
        assert record["origin_group"].startswith(f"robot/{profile}/family-") and "/goal-zone" in record["origin_group"]
        provenance = record["provenance"]
        tags = [f"template:{item}" for item in provenance["phrasing"]] + [f"concept:{item}" for item in provenance["concepts"]]
        assert record["split"] == assign_split(record["origin_group"], policy, tags)
        assert provenance["holdout"] == policy.holdout_reasons(record["origin_group"], tags)
        assert provenance["instruction_templates"] == provenance["phrasing"] and provenance["concepts"]
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


# --------------------------------------------------------------------------
# 틱 대조 쌍 (docs/04 §3·§6, robo_jev.data.robot_contrast)
# --------------------------------------------------------------------------


def test_tick_contrast_pairs_flip_the_named_question_and_pass_the_deletion_analogue(smoke):
    """에피소드마다 종류별 ≤ 1의 대조 쌍: 기본 틱과 sibling이 같은 계열·split의 `judgment-v0` 레코드이고, 겨냥 질문의 라벨이
    실제로 다르며, 대상을 지우면 전문가가 관측·재계획 게이트로 간다(삭제 analogue). 세 종류뿐이다(`holding` 없음, 리뷰 1 I2);
    퇴화(hold∉A) 답이 나오는 쌍은 없고(I3), `instruction` sibling의 v2 조합은 목록에 있거나 로봇이 다른 물체를 들고 있다(I3)."""
    from robo_jev.data.robot_contrast import KINDS, build_pairs, deletion_outcome, flipped_answer
    from robo_jev.harness.robot import joint_key_parts
    from robo_jev.sim.expert import DEGENERATE_REASONS

    assert set(KINDS) == {"forbidden", "zone_boundary", "instruction"}
    expert = smoke["expert"]
    kinds_seen = set()
    path_orders: list[list[str]] = []
    for (profile, seed), record in smoke["records"].items():
        pairs, reasons = build_pairs(record, expert=expert)
        assert pairs, reasons
        assert len(pairs) % 2 == 0 and len(pairs) // 2 <= 3
        by_id = {row["request"]["request_id"]: row for row in pairs}
        for row in pairs:
            validate_record(row)
            assert row["schema_version"] == "judgment-v0"
            assert row["origin_group"] == record["origin_group"] and row["split"] == record["split"]
            assert row["provenance"]["domain"] == "robot" and row["provenance"]["episode_id"] == record["episode_id"]
            assert row["provenance"]["holdout"] == record["provenance"]["holdout"]
            assert {question["id"] for question in row["request"]["questions"]} >= {"q_main", "q_done", "q_gripper"}
            assert not any(key in row["request"] for key in NON_INPUT_FIELDS)
        for sibling in (row for row in pairs if row["provenance"].get("derivation") == "contrast"):
            contrast = sibling["provenance"]["contrast"]
            base = by_id[contrast["sibling_id"]]
            kind = sibling["provenance"]["kind"]
            kinds_seen.add(kind)
            assert contrast["flipped_question"] == KINDS[kind]
            assert base["provenance"]["contrast"] == {"role": "base", "sibling_id": sibling["request"]["request_id"], "focus_field": contrast["focus_field"], "flipped_question": contrast["flipped_question"]}
            assert flipped_answer(base, contrast["flipped_question"]) != flipped_answer(sibling, contrast["flipped_question"])
            assert contrast["deletion"]["outcome"] in ("observe_gate", "replan_gate")
            tick_request = sibling["evidence"]["contrast"]["tick_request"]
            assert deletion_outcome(sibling["request"]["state"], tick_request, kind, expert) == contrast["deletion"]["outcome"]
            # 기본 틱의 후보를 그대로 쓰되 q_main의 순서는 섞는다 (정답 위치 편향 방지).
            base_main = next(q for q in base["request"]["questions"] if q["id"] == "q_main")
            sibling_main = next(q for q in sibling["request"]["questions"] if q["id"] == "q_main")
            assert sorted(c["id"] for c in base_main["criteria"]) == sorted(c["id"] for c in sibling_main["criteria"])
            tick = record["ticks"][sibling["evidence"]["tick_index"]]
            assert sorted(c["id"] for c in base_main["criteria"]) == sorted(entry["id"] for entry in tick["request"]["candidates"]["q_main"])
            # 경로 후보도 섞는다 (gen-robot-contrast-v0.3): 하네스의 정식 순서(direct 첫 자리)를 그대로 두면 정답(거의 언제나 direct)이
            # 첫 자리에 몰려 D1 규모의 QA 정답 위치 검사가 걸렸다(400편: K=3 표본 1,214 중 0.993이 첫 자리).
            base_path = next((q for q in base["request"]["questions"] if q["id"] == "q_path"), None)
            if base_path is not None:
                assert sorted(c["id"] for c in base_path["criteria"]) == sorted(entry["id"] for entry in tick["request"]["candidates"]["q_path"])
                path_orders.append([c["id"] for c in base_path["criteria"]])
            main_reasons = sibling["evidence"]["contrast"]["main_reasons"]
            assert main_reasons["base"] not in DEGENERATE_REASONS and main_reasons["sibling"] not in DEGENERATE_REASONS
            assert sibling["provenance"]["contrast"]["deletion"]["expert_version"] == expert.version
            if kind == "instruction":
                goal = sibling["request"]["state"]["goal"]
                holding = sibling["request"]["state"]["robot"].get("holding")
                listed = any(
                    (parts := joint_key_parts(entry.get("key", ""))) is not None and parts[0] in ("grasp", "place")
                    and parts[1] == goal["target_ref"] and parts[3] == goal["target_zone"]
                    for entry in tick["request"]["candidates"]["q_main"]
                )
                assert listed or (holding is not None and holding != goal["target_ref"])
    assert {"forbidden", "zone_boundary"} <= kinds_seen, kinds_seen
    assert path_orders and any(order[0] != "p0" for order in path_orders), path_orders  # 경로 후보의 첫 자리가 언제나 direct가 아니다


def test_the_zone_boundary_flip_moves_the_placed_target_one_centimetre_out_and_forgetting_it_gates_to_observe(smoke):
    from robo_jev.data.robot_contrast import BOUNDARY_STEP_MM, flip, forget

    record = smoke["records"][("E0", 17)]
    done = next(tick for tick in record["ticks"] if tick["usage"]["gate"] == "done")
    state = done["request"]["state"]
    target = next(o for o in state["objects"] if o["id"] == state["goal"]["target_ref"])
    zone = next(z for z in state["zones"] if z["id"] == state["goal"]["target_zone"])
    flipped, field = flip(state, "zone_boundary")
    moved = next(o for o in flipped["objects"] if o["id"] == target["id"])
    x0, y0, x1, y1 = zone["bounds_mm"]
    assert not (min(x0, x1) <= moved["pose_mm"][0] <= max(x0, x1) and min(y0, y1) <= moved["pose_mm"][1] <= max(y0, y1))
    assert sum(abs(a - b) for a, b in zip(moved["pose_mm"], target["pose_mm"])) <= BOUNDARY_STEP_MM + max(
        abs(target["pose_mm"][0] - min(x0, x1)), 0
    )
    assert field.startswith(f"objects[{target['id']}].pose_mm")
    derived = next(item for item in flipped["derived"] if item["object"] == target["id"])
    assert derived["relative_mm"] == [moved["pose_mm"][i] - flipped["robot"]["ee_pose_mm"][i] for i in range(3)]
    gone = forget(flipped, "zone_boundary")
    assert all(o["id"] != target["id"] for o in gone["objects"]) and all(item["object"] != target["id"] for item in gone["derived"])
    out = smoke["expert"].act({"request": {**done["request"], "state": gone}}, None)
    assert out["expert_meta"]["main"]["reason"] == "observe_target"


def test_the_batch_writes_the_contrast_file_into_the_manifest_and_the_qa_recounts_it(smoke, tmp_path):
    """`run`은 배치 끝에 `contrast/records.jsonl`을 쓰고 manifest의 `files`·`contrast`에 적는다; 자동 QA가 쌍을 다시 검사하고
    (한 자리·뒤집힘·삭제) 깨진 sibling을 잡는다."""
    import shutil

    from robo_jev.data.robot_episodes import CONTRAST_PATH, write_contrast
    from robo_jev.data.validate import load_dataset

    out = tmp_path / "batch"
    shutil.copytree(smoke["out"] / "episodes", out / "episodes")
    records = list(smoke["records"].values())
    summary = write_contrast(records, out, CONFIG, expert=smoke["expert"])
    assert summary["pairs"] >= 3 and summary["records"] == 2 * summary["pairs"]
    assert set(summary["by_kind"]) >= {"forbidden", "zone_boundary"} and "holding" not in summary["by_kind"]
    assert set(summary["deletion_outcomes"]) <= {"observe_gate", "replan_gate"}
    manifest = build_manifest(out, CONFIG, batch_wall_s=1.0)
    entry = manifest["files"][CONTRAST_PATH]
    assert entry["kind"] == "contrast" and entry["records"] == summary["records"]
    assert entry["sha256"] == hashlib.sha256((out / CONTRAST_PATH).read_bytes()).hexdigest()

    rows, paths = load_dataset(out)
    assert len(rows) == 2 + summary["records"]
    report = validate_dataset(rows, holdouts=CONFIG["split"])
    assert report["errors"] == [] and report["holdouts"]["leaked_groups"] == 0
    contrast = report["contrast"]
    assert contrast["pairs"] == summary["pairs"] and contrast["by_domain"] == {"robot": summary["pairs"]}
    assert contrast["deletion_failures"] == contrast["one_field_failures"] == contrast["flip_failures"] == 0

    sibling = next(row for row in rows if (row.get("provenance") or {}).get("derivation") == "contrast" and row["provenance"]["kind"] == "forbidden")
    poisoned = copy.deepcopy(rows)
    target = next(row for row in poisoned if row.get("request", {}).get("request_id") == sibling["request"]["request_id"])
    target["request"]["state"]["robot"]["gripper_mm"] += 5  # 초점 사실이 아닌 곳이 달라졌다
    assert any(error["path"] == "request.state" for error in validate_dataset(poisoned, holdouts=CONFIG["split"])["errors"])
    # 리뷰 1 I1: 초점 물체가 아닌 **다른 물체**가 40mm 움직인 sibling도 잡는다 (접두 검사는 놓쳤다) — 두 종류 모두.
    for kind in ("forbidden", "zone_boundary"):
        sibling = next(row for row in rows if (row.get("provenance") or {}).get("derivation") == "contrast" and row["provenance"]["kind"] == kind)
        focus = sibling["request"]["state"]["goal"]["target_ref"]
        poisoned = copy.deepcopy(rows)
        target = next(row for row in poisoned if row.get("request", {}).get("request_id") == sibling["request"]["request_id"])
        other = next(entry for entry in target["request"]["state"]["objects"] if entry["id"] != focus)
        other["pose_mm"][1] += 40
        report_poisoned = validate_dataset(poisoned, holdouts=CONFIG["split"])
        assert report_poisoned["contrast"]["one_field_failures"] >= 1, kind
        assert any(error["path"] == "request.state" for error in report_poisoned["errors"]), kind
    poisoned = copy.deepcopy(rows)
    target = next(row for row in poisoned if row.get("request", {}).get("request_id") == sibling["request"]["request_id"])
    target["provenance"]["contrast"]["deletion"]["outcome"] = "replan_gate" if sibling["provenance"]["contrast"]["deletion"]["outcome"] == "observe_gate" else "observe_gate"
    assert any(error["path"] == "provenance.contrast.deletion" for error in validate_dataset(poisoned, holdouts=CONFIG["split"])["errors"])


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
    assert manifest["holdout_templates"] == CONFIG["split"]["holdout_templates"] and manifest["holdout_concepts"] == CONFIG["split"]["holdout_concepts"]
    assert manifest["versions"]["expert"] == [smoke["expert"].version]
    assert set(manifest["bytes"]) == {"total", "per_episode_mean_mb", "per_tick_mean_kb"}
    assert set(manifest["timing"]) >= {"wall_s_per_episode_mean", "episodes_per_hour", "projected_hours_for_target", "target_episodes"}
    assert manifest["timing"]["episodes_per_hour"] > 0
    assert set(manifest["decisions"]) == {"gates", "stops", "switches", "main_changes", "conflicts", "per_episode"}
    # `files`는 학습 적재기(`sampler.load_items`)와 비로봇 manifest(`data/generate.py`)가 읽는 꼴 — 경로 → {sha256, …}.
    assert isinstance(manifest["files"], dict) and len(manifest["files"]) == 2
    for path, entry in manifest["files"].items():
        assert path == f"episodes/{entry['episode_id']}/streams.jsonl"
        assert set(entry) == {"episode_id", "profile", "seed", "split", "origin_group", "done", "ticks", "bytes", "sha256", "records"}
        assert (out / path).is_file() and entry["records"] == 1
        assert entry["sha256"] == hashlib.sha256((out / path).read_bytes()).hexdigest()


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
    assert manifest["run"] == {"requested": 3, "produced": 1, "skipped": 2, "excluded": 0, "resume": True}
    assert manifest["episodes"] == 3


def test_exclude_groups_from_skips_the_origin_groups_a_previous_batch_used(tmp_path, monkeypatch):
    """D-OOD (docs/04 §6, `configs/data/d_ood.yaml`): `exclude_groups_from`의 manifest `families`(origin group)에 든 seed는 건너뛰고 일정을
    앞으로 늘려 `count`편을 채운다 — D1과 겹치지 않는 장면·목표 계열만 남는다. manifest `run.excluded`가 센다."""
    import robo_jev.data.robot_episodes as module
    from robo_jev.sim.scene import build_plan, origin_group

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
    sim = yaml.safe_load(SIM_CONFIG.read_text(encoding="utf-8"))
    used = {origin_group(profile, build_plan(sim, seed, profile)) for profile, seed in (("E0", 100), ("E1", 100), ("E0", 101))}
    previous = tmp_path / "previous-manifest.json"
    previous.write_text(json.dumps({"families": {group: 1 for group in used}}), encoding="utf-8")
    config = {**CONFIG, "exclude_groups_from": [str(previous)], "exclude_schedule_factor": 8}
    manifest = run(config, 2, tmp_path / "out")
    assert len(calls) == 2 and manifest["run"]["produced"] == 2 and manifest["run"]["excluded"] >= 1
    assert all(origin_group(profile, build_plan(sim, seed, profile)) not in used for profile, seed in calls)
    assert calls[0] != ("E0", 100)  # 첫 seed는 D1이 쓴 계열이라 건넌다
    assert module.excluded_groups(CONFIG) == set()


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
