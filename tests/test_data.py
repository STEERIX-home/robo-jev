"""생성기·그룹 분할·QA 검사 (docs/04 §2·§3·§5·§6).

`tests/test_contracts.py`가 계약 자체를 본다면, 여기서는 **생성된 데이터**가 그 계약과
생성 계획을 지키는지 본다. 거절·불변 검사를 먼저 쓰고(RED) 구현을 붙인다(GREEN).

핵심 불변:

* 같은 seed는 같은 레코드 목록을 낸다 (바이트 단위 재현).
* 라벨은 (사실, 질문 명세)의 순수 함수다 — 표현만 바꾸면 그대로, 사실을 바꾸면 따라 바뀐다.
* origin group 하나는 split 하나에만 들어간다. 파생본은 부모의 group·split을 승계한다.
* 정답 위치에 편향이 없다.
"""

import copy
import json
import os
import random
import subprocess
from pathlib import Path
import sys
from collections import Counter, defaultdict

import pytest
from helpers import D0, D0_STREAMS, read_jsonl

from robo_jev.contracts import model_input, validate_record
from robo_jev.data import domains
from robo_jev.data.generate import (
    DEFAULT_CONFIG,
    PILOT_CONFIG,
    generate_records,
    load_config,
)
from robo_jev.data.generate import main as generate_main
from robo_jev.data.split import CONCEPT_TAG, DEFAULT_WEIGHTS, OOD_SPLITS, TEMPLATE_TAG, SplitPolicy, assign_split, ood_split
from robo_jev.data.validate import POSITION_BIAS_MIN_SAMPLES, validate_dataset
from robo_jev.data.validate import main as validate_main

BATCH_COUNT = 500
BATCH_SEED = 17


@pytest.fixture(scope="module")
def batch() -> list[dict]:
    """검사 대부분이 함께 보는 비로봇 500상태 (docs/04 §6 smoke)."""
    return generate_records(count=BATCH_COUNT, seed=BATCH_SEED)


@pytest.fixture(scope="module")
def report(batch) -> dict:
    return validate_dataset(batch)


# --------------------------------------------------------------------------
# 계획서가 요구한 재현·계보 검사 (task-2-brief.md)
# --------------------------------------------------------------------------


def test_generation_is_reproducible_and_groups_do_not_leak():
    rows = generate_records(count=500, seed=17)
    assert rows == generate_records(count=500, seed=17)
    groups = defaultdict(set)
    for row in rows:
        groups[row["origin_group"]].add(row["split"])
    assert all(len(parts) == 1 for parts in groups.values())
    report = validate_dataset(rows, holdouts=DEFAULT_CONFIG["split"])
    assert report["invalid_records"] == 0 and report["errors"] == []
    assert report["states"] == 500


def test_a_different_seed_gives_different_states():
    other = generate_records(count=20, seed=18)
    assert other != generate_records(count=20, seed=17)
    assert len(other) == 20


def test_count_is_exact_for_odd_sizes():
    for count in (1, 3, 7, 33):
        assert len(generate_records(count=count, seed=3)) == count


# --------------------------------------------------------------------------
# 생성된 레코드가 계약을 지킨다
# --------------------------------------------------------------------------


def test_every_generated_record_is_valid(batch):
    for index, record in enumerate(batch):
        try:
            validate_record(record)
        except ValueError as error:  # pragma: no cover - 실패할 때만 본다
            pytest.fail(f"{index}번 레코드가 계약을 어긴다: {error}")


def test_generated_records_are_json_serialisable(batch):
    for record in batch:
        json.dumps(record, ensure_ascii=False)


def test_records_carry_group_split_provenance_and_evidence(batch):
    for record in batch:
        assert record["schema_version"] == "judgment-v0"
        assert record["origin_group"].count("/") == 2, record["origin_group"]
        provenance = record["provenance"]
        for field in ("generator", "domain", "template", "seed", "origin_group", "variants", "phrasing", "concepts", "holdout"):
            assert field in provenance, field
        # split은 계열의 해시 분할이거나(holdout 아님) 계열 해시로 반분한 OOD다(holdout 이유가 적힌다).
        if provenance["holdout"]:
            assert record["split"] == ood_split(record["origin_group"]) and record["split"] in OOD_SPLITS
        else:
            assert record["split"] == assign_split(record["origin_group"])
        assert record["evidence"]["rule_trace"], record["request"]["request_id"]


def test_request_ids_are_unique(batch):
    ids = [record["request"]["request_id"] for record in batch]
    assert len(set(ids)) == len(ids)


def test_geometry_records_never_use_physical_event_labels(batch):
    """기하·규칙 문제를 실제 물리 성공 라벨로 표시하지 않는다 (task-2-brief)."""
    kinds = {label["kind"] for record in batch for label in record["labels"]}
    assert kinds <= {"valid_set", "single"}, kinds


def test_every_label_names_its_rule(batch):
    for record in batch:
        for label in record["labels"]:
            assert label.get("source") or label.get("rule"), record["request"]["request_id"]


def test_masked_questions_have_no_label_but_keep_a_reason(batch):
    masked = 0
    for record in batch:
        labelled = {label["question_id"] for label in record["labels"]}
        for question in record["request"]["questions"]:
            if question["id"] in labelled:
                continue
            masked += 1
            assert question["id"] in record["evidence"]["masked"], question["id"]
    assert masked > 0, "근거가 없어 마스킹한 질문이 하나도 없다"


# --------------------------------------------------------------------------
# 분야·타입·난이도 구성 (docs/04 §2·§3)
# --------------------------------------------------------------------------


def test_all_four_non_robot_domains_are_generated(batch):
    counts = Counter(record["provenance"]["domain"] for record in batch)
    assert set(counts) == {"spatial", "dom", "workflow", "rules"}
    for domain, weight in DEFAULT_CONFIG["domains"].items():
        share = counts[domain] / len(batch)
        assert abs(share - weight / 100) <= 0.08, (domain, share)


def test_question_type_mix_follows_the_config(batch):
    counts = Counter(
        question["type"] for record in batch for question in record["request"]["questions"]
    )
    total = sum(counts.values())
    for question_type, weight in DEFAULT_CONFIG["question_types"].items():
        share = counts[question_type] / total
        assert abs(share - weight / 100) <= 0.06, (question_type, share)


def test_questions_per_state_uses_the_configured_sizes(batch):
    sizes = Counter(len(record["request"]["questions"]) for record in batch)
    assert set(sizes) == set(DEFAULT_CONFIG["questions_per_state"])
    for size, weight in DEFAULT_CONFIG["questions_per_state"].items():
        share = sizes[size] / len(batch)
        assert abs(share - weight / 100) <= 0.08, (size, share)


def test_both_languages_appear(batch):
    languages = Counter(record["provenance"]["language"] for record in batch)
    assert set(languages) == {"ko", "en"}
    assert min(languages.values()) / len(batch) > 0.1


def test_every_difficulty_variant_is_exercised(batch):
    """docs/04 §3의 변형이 500상태 안에 모두 들어 있다."""
    seen = Counter(tag for record in batch for tag in record["provenance"]["variants"])
    missing = sorted(set(domains.VARIANT_TAGS) - set(seen))
    assert not missing, missing


# --------------------------------------------------------------------------
# 수준 경계: 상태에서 답이 나와야 한다 (검토 1차 Important 1, Minor 5)
# --------------------------------------------------------------------------

#: 상태에 허용 오차를 적어 둔 규칙만 인접 수준을 함께 허용할 수 있다.
TOLERANCE_RULES = {
    "spatial/distance-level-v0",
    "workflow/urgency-level-v0",
    "workflow/resource-load-v0",
}

#: 정수·정확한 값을 세는 규칙. 경계에 정확히 걸려도 수준은 하나여야 한다.
EXACT_LEVEL_RULES = {"spatial/crowding-level-v0", "dom/progress-level-v0"}

#: 분야마다 상태에 있어야 하는 수준 경계 (생성기 상수가 아니라 상태에서 읽혀야 한다).
REQUIRED_THRESHOLDS = {
    "spatial": (
        "distance_edges_mm",
        "distance_tolerance_mm",
        "crowding_edges",
        "crowd_radius_mm",
    ),
    "dom": ("progress_edges", "progress_scale"),
    "workflow": ("urgency_edges_h", "schedule_tolerance_h", "load_edges", "load_tolerance"),
    "rules": (),  # 심각도는 지배 규칙의 `severity` 그 자체다
}


def test_bucket_keeps_exact_values_in_one_level():
    """허용 오차가 0이면 경계값은 위쪽 한 수준에만 속한다 (반열린 구간)."""
    edges = (1.0, 2.0, 3.0)
    assert domains._bucket(0.0, edges, margin=0.0) == ["0"]
    assert domains._bucket(1.0, edges, margin=0.0) == ["1"]
    assert domains._bucket(2.0, edges, margin=0.0) == ["2"]
    assert domains._bucket(3.0, edges, margin=0.0) == ["3"]
    # 실제 허용 오차가 있을 때만 인접 수준을 함께 허용한다.
    assert domains._bucket(1.0, edges, margin=0.2) == ["0", "1"]
    assert domains._bucket(1.5, edges, margin=0.2) == ["1"]


def test_exact_count_levels_never_straddle_two_levels(batch):
    for record in batch:
        for trace in record["evidence"]["rule_trace"]:
            if trace["rule"] in EXACT_LEVEL_RULES and isinstance(trace["answer"], list):
                assert len(trace["answer"]) == 1, (record["request"]["request_id"], trace)


def test_only_rules_with_a_stated_tolerance_give_two_level_answers(batch):
    two_level: Counter = Counter()
    for record in batch:
        by_id = {question["id"]: question for question in record["request"]["questions"]}
        for label in record["labels"]:
            if by_id[label["question_id"]]["type"] != "ordinal":
                continue
            if len(label.get("candidate_ids") or [label["answer"]]) > 1:
                two_level[label["source"]] += 1
    # 심각도만 예외다 — 지배 규칙이 여럿이라 수준이 여럿이지 경계 때문이 아니다.
    assert set(two_level) <= TOLERANCE_RULES | {"rules/severity-level-v0"}, two_level
    assert set(two_level) & TOLERANCE_RULES, two_level


def test_ordinal_cut_points_are_written_in_the_state(batch):
    for record in batch:
        thresholds = record["request"]["state"].get("thresholds", {})
        for key in REQUIRED_THRESHOLDS[record["provenance"]["domain"]]:
            assert key in thresholds, (record["request"]["request_id"], key)


# --------------------------------------------------------------------------
# 관측 한계: 관측으로 못 정하는 답을 정답이라 하지 않는다 (검토 1차 Important 2·3)
# --------------------------------------------------------------------------

#: 관측된 것만 두고 묻는 질문. 모든 표현에 "관측"이 들어가야 답이 상태에서 나온다.
OBSERVED_SCOPED = ("q_goal_met", "q_distance", "q_crowding")


def _spatial_pair(*, hidden: bool):
    """같은 색 물체 둘 — 하나는 가려 두거나(hidden) 둘 다 관측된 장면."""
    domain = domains.DOMAINS["spatial"]
    scene = domain.make_scene(random.Random("observability"), "zone-color")
    for obj in scene["objects"]:
        obj.update({"color": "blue", "visible": True, "age_ms": 40})
    scene["objects"][0].update({"color": "red", "x": 100, "y": 0})
    scene["objects"][1].update({"color": "red", "x": 300, "y": 0})
    if hidden:
        scene["objects"][1].update({"visible": False, "x": None, "y": None})
    return domain, scene


def _nearest_red(domain, scene):
    """`nearest` 규칙만 떼어 본다 (후보는 pool과 같게 전부 + "해당 없음")."""
    spec = domains.QuestionSpec(
        "q_nearest_0",
        "choice",
        "nearest",
        {
            "color": "red",
            "candidates": [obj["id"] for obj in scene["objects"]],
            "none": True,
        },
    )
    return domain.render(scene, spec, random.Random(3), "ko")


def _goal_met_red(domain, scene, zone: str):
    spec = domains.QuestionSpec("q_goal_met_0", "boolean", "goal_met", {"color": "red", "zone": zone})
    return domain.render(scene, spec, random.Random(3), "ko")


def test_nearest_is_undecidable_when_a_same_colour_object_is_unobserved():
    """가려진 같은 색 물체가 더 가까울 수 있으므로 '가장 가까운'은 정해지지 않는다."""
    domain, scene = _spatial_pair(hidden=True)
    rendered = _nearest_red(domain, scene)
    assert _answer_key(rendered) in (("<none>",), "masked"), rendered.label
    assert "missing_info" in rendered.variants


def test_nearest_names_the_object_when_every_same_colour_object_is_observed():
    domain, scene = _spatial_pair(hidden=False)
    assert _answer_key(_nearest_red(domain, scene)) == ("o1",)


def test_goal_satisfied_is_about_the_observation_only():
    """가려진 같은 색 물체가 있어도 '관측으로 확인되는가'는 답할 수 있다."""
    domain, scene = _spatial_pair(hidden=True)
    zone = domains._zone_of_x(scene["objects"][0]["x"])
    assert _answer_key(_goal_met_red(domain, scene, zone)) is True
    other = next(zone_id for zone_id, _, _ in domains._ZONES if zone_id != zone)
    assert _answer_key(_goal_met_red(domain, scene, other)) is False


def test_observation_scoped_questions_say_so_in_both_languages(batch):
    seen: Counter = Counter()
    for record in batch:
        marker = "관측" if record["provenance"]["language"] == "ko" else "observ"
        for question in record["request"]["questions"]:
            if question["id"].rsplit("_", 1)[0] in OBSERVED_SCOPED:
                seen[question["id"].rsplit("_", 1)[0]] += 1
                assert marker in question["instructions"], question["instructions"]
    assert set(seen) == set(OBSERVED_SCOPED), seen


def test_candidate_counts_and_multiple_answers_vary(batch):
    sizes = set()
    multiple = 0
    for record in batch:
        by_id = {question["id"]: question for question in record["request"]["questions"]}
        for question in record["request"]["questions"]:
            if question["type"] == "choice":
                sizes.add(len(question["criteria"]))
        for label in record["labels"]:
            if (
                label["kind"] == "valid_set"
                and len(label["candidate_ids"]) > 1
                and by_id[label["question_id"]]["type"] == "choice"
            ):
                multiple += 1
    assert len(sizes) >= 4, sizes
    assert multiple > 0


def test_answer_position_has_no_bias(batch):
    """정답이 유일한 choice 질문에서 후보 위치가 한쪽으로 쏠리지 않는다."""
    positions: dict[int, Counter] = defaultdict(Counter)
    for record in batch:
        labels = {label["question_id"]: label for label in record["labels"]}
        for question in record["request"]["questions"]:
            label = labels.get(question["id"])
            if question["type"] != "choice" or label is None:
                continue
            answers = label.get("candidate_ids") or [label.get("answer")]
            if len(answers) != 1:
                continue
            candidates = [criterion["id"] for criterion in question["criteria"]]
            if len(candidates) < 3:
                continue
            positions[len(candidates)][candidates.index(answers[0])] += 1

    assert positions, "위치 편향을 볼 choice 질문이 없다"
    checked = 0
    for size, counter in sorted(positions.items()):
        total = sum(counter.values())
        if total < POSITION_BIAS_MIN_SAMPLES:
            continue  # 표본이 적으면 균등해도 한 자리가 우연히 튄다
        checked += 1
        assert max(counter.values()) / total <= 1 / size + 0.15, (size, total, counter)
    assert checked >= 2, positions  # 표본이 충분한 층이 최소 둘은 있어야 검사가 의미 있다


# --------------------------------------------------------------------------
# 라벨은 사실의 함수다 (docs/04 §3: "표현 변형으로 사실이 바뀌면 정답도 다시 계산한다")
# --------------------------------------------------------------------------


def _semantic(question: dict) -> dict[str, str]:
    """후보 id → 표현·재배열과 무관한 의미 키.

    choice 후보의 id는 레코드마다 다시 붙으므로 상태 요소를 가리키는 `ref`로 본다.
    `ref`가 없는 choice 후보는 "해당 없음·정보 부족" 하나뿐이다. ordinal 수준의 id는
    수준 자체라서 그대로 쓴다.
    """
    if question["type"] == "choice":
        return {
            criterion["id"]: criterion.get("ref", "<none>") for criterion in question["criteria"]
        }
    return {criterion["id"]: criterion["id"] for criterion in question["criteria"]}


def _answer_key(rendered) -> object:
    """표현과 무관하게 비교할 수 있는 정답 값."""
    if rendered.label is None:
        return "masked"
    if rendered.question["type"] == "boolean":
        return rendered.label["answer"]
    keys = _semantic(rendered.question)
    chosen = rendered.label.get("candidate_ids") or [rendered.label["answer"]]
    return tuple(sorted(keys[candidate] for candidate in chosen))


def _render_all(domain, scene, specs, *, language: str, seed: int) -> list:
    rng = random.Random(seed)
    return [domain.render(scene, spec, rng, language) for spec in specs]


def _answers(rendered_list) -> dict:
    return {item.question["id"]: _answer_key(item) for item in rendered_list}


def _wording(rendered_list) -> list[str]:
    return [item.question["instructions"] for item in rendered_list]


#: 분야마다 "정답을 실제로 바꾸는" 사실 변경 (표현 변경과 구분한다).
FACT_CHANGES = {
    "spatial": lambda scene: scene["objects"][0].update(
        {"x": -scene["objects"][0]["x"], "color": "magenta"}
    ),
    "dom": lambda scene: [
        element.update({"visible": False, "enabled": False}) for element in scene["elements"]
    ],
    "workflow": lambda scene: [step.update({"done": not step["done"]}) for step in scene["steps"]],
    "rules": lambda scene: scene["situation"].update(
        {key: f"{value}-바뀜" for key, value in scene["situation"].items()}
    ),
}


@pytest.mark.parametrize("name", sorted(domains.DOMAINS))
def test_paraphrase_keeps_the_label_but_a_fact_change_moves_it(name):
    domain = domains.DOMAINS[name]
    scene = domain.make_scene(random.Random(f"fact:{name}"), domain.templates[0])
    specs = domain.pool(scene)
    assert len(specs) >= 16, (name, len(specs))

    korean = _render_all(domain, scene, specs, language="ko", seed=1)
    english = _render_all(domain, scene, specs, language="en", seed=2)
    assert _answers(korean) == _answers(english), name
    assert _wording(korean) != _wording(english), name

    changed = copy.deepcopy(scene)
    FACT_CHANGES[name](changed)
    moved = _render_all(domain, changed, specs, language="ko", seed=1)
    assert _answers(moved) != _answers(korean), name


@pytest.mark.parametrize("name", sorted(domains.DOMAINS))
def test_every_template_of_every_domain_builds_valid_questions(name):
    domain = domains.DOMAINS[name]
    for template in domain.templates:
        scene = domain.make_scene(random.Random(f"tpl:{name}:{template}"), template)
        rng = random.Random(7)
        for spec in domain.pool(scene):
            rendered = domain.render(scene, spec, rng, "ko")
            assert rendered.question["id"] == spec.id
            assert rendered.question["type"] == spec.type
            assert rendered.trace, spec.id


# --------------------------------------------------------------------------
# 분할 (docs/04 §5)
# --------------------------------------------------------------------------

#: stable hash를 얼려 둔다. 값이 바뀌면 이미 배포한 데이터의 split이 뒤집힌다.
FROZEN_SPLITS = {
    "spatial/zone-color/0000": "train",
    "spatial/zone-color/0001": "calibration",
    "spatial/zone-color/0011": "dev",
    "spatial/zone-color/0016": "test",
    "dom/checkout-form/0001": "test",
    "workflow/release-train/0002": "train",
    "rules/access-policy/0003": "train",
    "scene-family-018": "test",
}


def test_known_groups_keep_their_frozen_split():
    for group, split in FROZEN_SPLITS.items():
        assert assign_split(group) == split, group


def test_assign_split_does_not_depend_on_the_process_hash_seed():
    """`hash()`가 아니라 sha256을 써야 프로세스마다 같은 split이 나온다."""
    code = (
        "from robo_jev.data.split import assign_split;"
        "print(','.join(assign_split(f'g/{i}') for i in range(24)))"
    )
    runs = []
    for hash_seed in ("0", "1", "random"):
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            check=True,
            env={**os.environ, "PYTHONHASHSEED": hash_seed},
        )
        runs.append(result.stdout.strip())
    assert len(set(runs)) == 1, runs


def test_split_proportions_follow_the_weights():
    groups = [f"synthetic-family-{index:05d}" for index in range(10_000)]
    counts = Counter(assign_split(group) for group in groups)
    for split, weight in DEFAULT_WEIGHTS:
        assert abs(counts[split] / len(groups) - weight / 100) <= 0.03, (split, counts[split])


def test_holdouts_are_excluded_before_the_hashed_split():
    policy = SplitPolicy.from_config(
        {
            "holdout_groups": ["rules/access-policy/0003"],
            "holdout_prefixes": ["rules/energy-device/"],
            "holdout_domains": ["dom"],
        }
    )
    assert policy.assign("rules/access-policy/0003") in OOD_SPLITS
    assert policy.assign("rules/energy-device/0007") in OOD_SPLITS
    assert policy.assign("dom/checkout-form/0001") in OOD_SPLITS
    assert policy.holdout_reasons("rules/access-policy/0003") == ["group:rules/access-policy/0003"]
    assert policy.holdout_reasons("dom/checkout-form/0001") == ["domain:dom"]
    # holdout이 아닌 group은 해시 분할을 그대로 따른다.
    assert policy.assign("spatial/zone-color/0000") == assign_split("spatial/zone-color/0000")


def test_template_and_concept_holdouts_seal_by_tag_and_split_ood_into_dev_and_test():
    """계약 v0.3: 문구 템플릿 변형·개념은 태그(`template:<id>`·`concept:<id>`)로 맞대고, OOD는 계열 해시로
    ood_dev/ood_test로 반분한다 (docs/04 §5)."""
    policy = SplitPolicy.from_config({"holdout_templates": ["spatial.distance.ko#1"], "holdout_concepts": ["dom:reveal"]})
    group = "spatial/zone-color/0007"
    assert policy.assign(group) == assign_split(group)  # 태그가 없으면 holdout이 아니다
    assert policy.holdout_reasons(group, [TEMPLATE_TAG + "spatial.distance.ko#0"]) == []
    assert policy.holdout_reasons(group, [TEMPLATE_TAG + "spatial.distance.ko#1"]) == ["template:spatial.distance.ko#1"]
    assert policy.holdout_reasons(group, [CONCEPT_TAG + "dom:reveal", TEMPLATE_TAG + "spatial.distance.ko#1"]) == [
        "concept:dom:reveal", "template:spatial.distance.ko#1"
    ]
    assert policy.assign(group, [CONCEPT_TAG + "dom:reveal"]) == ood_split(group)
    assert assign_split(group, policy, tags=[CONCEPT_TAG + "dom:reveal"]) in OOD_SPLITS
    # OOD 반분은 결정적이고 두 쪽 다 쓰인다.
    halves = Counter(ood_split(f"g/{index}") for index in range(2000))
    assert set(halves) == set(OOD_SPLITS) and abs(halves["ood_dev"] - halves["ood_test"]) < 200
    assert ood_split(group) == ood_split(group)


def test_split_weights_must_be_positive_integers():
    with pytest.raises(ValueError, match="weights"):
        SplitPolicy.from_config({"weights": {"train": 0, "dev": 0}})
    with pytest.raises(ValueError, match="ood"):
        SplitPolicy.from_config({"weights": {"train": 70, "ood": 30}})


def test_derived_records_inherit_the_parent_group_and_split(batch):
    by_id = {record["request"]["request_id"]: record for record in batch}
    derived = [record for record in batch if record["provenance"].get("derived_from")]
    assert derived, "파생본이 하나도 없다"
    for record in derived:
        parent = by_id[record["provenance"]["derived_from"]]
        assert record["origin_group"] == parent["origin_group"]
        assert record["split"] == parent["split"]
        assert record["provenance"]["derivation"] in ("paraphrase", "reorder", "contrast")


# --------------------------------------------------------------------------
# 대조 sibling과 삭제 검사 (docs/04 §3·§6, analysis-nimble §3-2)
# --------------------------------------------------------------------------


def _pairs(batch):
    by_id = {record["request"]["request_id"]: record for record in batch}
    return [
        (by_id[record["provenance"]["contrast"]["sibling_id"]], record)
        for record in batch
        if (record["provenance"].get("contrast") or {}).get("role") == "sibling"
        and record["provenance"]["contrast"]["sibling_id"] in by_id
    ]


def test_every_domain_emits_contrast_siblings_that_share_group_split_wording_and_candidate_order(batch):
    """기본 레코드마다 사실 하나만 바꾼 sibling이 있다(삭제 검사를 지난 쌍만). 같은 계열·split, 같은 문구·질문 목록(같은
    wording seed; 후보 id는 정답 후보를 뺀 질문 뒤로는 같은 shuffle seed에서도 달라질 수 있다)이고 provenance가 쌍을 서로
    가리킨다."""
    pairs = _pairs(batch)
    assert len(pairs) >= 0.7 * sum(1 for record in batch if record["provenance"].get("derived_from") is None)
    assert {sibling["provenance"]["domain"] for _, sibling in pairs} == set(domains.DOMAINS)
    for base, sibling in pairs:
        assert sibling["provenance"]["derivation"] == "contrast" and sibling["provenance"]["derived_from"] == base["request"]["request_id"]
        assert sibling["origin_group"] == base["origin_group"] and sibling["split"] == base["split"]
        assert base["provenance"]["contrast"] == {
            "role": "base",
            "sibling_id": sibling["request"]["request_id"],
            "focus_field": sibling["provenance"]["contrast"]["focus_field"],
            "flipped_question": sibling["provenance"]["contrast"]["flipped_question"],
        }
        assert sibling["provenance"]["phrasing"] == base["provenance"]["phrasing"]
        assert [q["instructions"] for q in sibling["request"]["questions"]] == [q["instructions"] for q in base["request"]["questions"]]
        assert [q["id"] for q in sibling["request"]["questions"]] == [q["id"] for q in base["request"]["questions"]]


def test_a_contrast_sibling_changes_exactly_one_fact_and_flips_the_named_question(batch):
    from robo_jev.data.generate import semantic_answers
    from robo_jev.data.validate import _strip_wording, leaf_diff

    for base, sibling in _pairs(batch):
        contrast = sibling["provenance"]["contrast"]
        diff = leaf_diff(_strip_wording(base["request"]["state"]), _strip_wording(sibling["request"]["state"]))
        assert len(diff) == 1, (sibling["request"]["request_id"], diff)
        assert contrast["focus_field"].split(".")[0] in diff[0]
        before, after = semantic_answers(base), semantic_answers(sibling)
        question_id = contrast["flipped_question"]
        assert question_id in before and question_id in after and before[question_id] != after[question_id]
        assert question_id in contrast["flipped_questions"]
        assert sibling["evidence"]["contrast"]["spec"]["id"] == question_id
        assert contrast["deletion"]["outcome"] in ("masked", "none_candidate")


def test_deleting_the_focus_fact_makes_the_flipped_label_unknown_in_every_domain():
    """삭제 검사: 초점 사실을 지운 장면(분야의 `forget`)에서 뒤집힌 질문을 다시 그리면 라벨이 마스크거나 "해당 없음"이다 —
    규칙 코드가 사실의 부재를 "모른다"로 답한다는 것을 분야마다 확인한다."""
    from robo_jev.data.generate import deletion_outcome

    for name, domain in domains.DOMAINS.items():
        scene = domain.make_scene(random.Random(f"contrast:{name}"), domain.templates[0])
        specs = domain.pool(scene)
        by_id = {spec.id: spec for spec in specs}
        outcomes = Counter()
        for contrast in domain.contrasts(scene, specs):
            assert contrast.focus_field.count(".") in (1, 2)
            assert contrast.flipped != scene and contrast.deleted != contrast.flipped
            for question_id in contrast.question_ids:
                outcomes[deletion_outcome(domain, contrast.deleted, by_id[question_id], "ko")] += 1
        assert outcomes["masked"] + outcomes["none_candidate"] > 0, (name, outcomes)


def test_the_qa_reports_contrast_pairs_and_flags_a_broken_pair(batch, report):
    """QA는 쌍 수·split별 수·삭제 결과를 적고 세 검사(한 자리, 뒤집힘, 삭제)를 다시 돌린다."""
    contrast = report["contrast"]
    assert contrast["pairs"] == len(_pairs(batch)) and contrast["checked"] == contrast["pairs"]
    assert sum(contrast["by_split"].values()) == contrast["pairs"] and sum(contrast["by_domain"].values()) == contrast["pairs"]
    assert contrast["deletion_failures"] == contrast["one_field_failures"] == contrast["flip_failures"] == 0
    assert set(contrast["deletion_outcomes"]) <= {"masked", "none_candidate"}
    assert set(contrast["missing"]) <= {"no_flip", "deletion_failed", "no_contrast", "sealed_concept"}

    base, sibling = next(
        (base, sibling) for base, sibling in _pairs(batch) if sibling["provenance"]["domain"] == "rules"
    )
    two_fields = copy.deepcopy(batch)
    poisoned = next(record for record in two_fields if record["request"]["request_id"] == sibling["request"]["request_id"])
    other = next(key for key in poisoned["request"]["state"]["situation"] if key != sibling["provenance"]["contrast"]["focus_field"].split(".")[1])
    poisoned["request"]["state"]["situation"][other] = "bogus"
    paths = {error["path"] for error in validate_dataset(two_fields)["errors"]}
    assert "request.state" in paths

    no_flip = copy.deepcopy(batch)
    poisoned = next(record for record in no_flip if record["request"]["request_id"] == sibling["request"]["request_id"])
    poisoned["labels"] = copy.deepcopy(base["labels"])
    poisoned["request"]["questions"] = copy.deepcopy(base["request"]["questions"])
    assert any(error["path"] == "provenance.contrast.flipped_question" for error in validate_dataset(no_flip)["errors"])

    wrong_field = copy.deepcopy(batch)
    poisoned = next(record for record in wrong_field if record["request"]["request_id"] == sibling["request"]["request_id"])
    poisoned["provenance"]["contrast"]["focus_field"] = f"situation.{other}"
    report_wrong = validate_dataset(wrong_field)
    assert report_wrong["contrast"]["deletion_failures"] >= 1 or any(
        error["path"] == "provenance.contrast.deletion" for error in report_wrong["errors"]
    )


def test_contrast_siblings_never_change_their_familys_sealing_in_either_direction(batch):
    """리뷰 1 M3: sibling 자신의 봉인 근거(문구 변형·개념)는 기본 레코드의 것과 같다 — 봉인 개념을 새로 다루는 sibling도, OOD
    계열 안에서 봉인 개념을 잃는 sibling도 만들지 않는다."""
    policy = SplitPolicy.from_config(DEFAULT_CONFIG["split"])
    for base, sibling in _pairs(batch):
        tags = lambda record: sorted({TEMPLATE_TAG + p for p in record["provenance"]["phrasing"]} | {CONCEPT_TAG + c for c in record["provenance"]["concepts"]})
        assert policy.holdout_reasons(base["origin_group"], tags(base)) == policy.holdout_reasons(sibling["origin_group"], tags(sibling))


def test_leaf_diff_names_exactly_the_changed_leaves():
    from robo_jev.data.validate import leaf_diff

    left = {"a": 1, "b": [1, {"c": 2}], "d": {"e": None}}
    assert leaf_diff(left, copy.deepcopy(left)) == []
    assert leaf_diff(left, {"a": 1, "b": [1, {"c": 3}], "d": {"e": None}}) == ["state.b[1].c"]
    assert leaf_diff(left, {"a": 1, "b": [1], "d": {"e": None}}) == ["state.b"]
    assert leaf_diff(left, {"a": 1, "b": [1, {"c": 2}], "d": {}}) == ["state.d.e"]


def test_paraphrases_keep_the_answer_and_change_the_wording(batch):
    by_id = {record["request"]["request_id"]: record for record in batch}
    paraphrases = [
        record
        for record in batch
        if record["provenance"].get("derivation") == "paraphrase"
    ]
    assert paraphrases, "표현 변형본이 하나도 없다"
    for record in paraphrases:
        parent = by_id[record["provenance"]["derived_from"]]
        assert _record_answers(record) == _record_answers(parent)
        assert _record_wording(record) != _record_wording(parent)


def _record_answers(record: dict) -> dict:
    questions = {question["id"]: question for question in record["request"]["questions"]}
    labels = {label["question_id"]: label for label in record["labels"]}
    answers = {}
    for question_id, question in questions.items():
        label = labels.get(question_id)
        if label is None:
            answers[question_id] = "masked"
        elif question["type"] == "boolean":
            answers[question_id] = label["answer"]
        else:
            keys = _semantic(question)
            chosen = label.get("candidate_ids") or [label["answer"]]
            answers[question_id] = tuple(sorted(keys[candidate] for candidate in chosen))
    return answers


def _record_wording(record: dict) -> list[str]:
    return [question["instructions"] for question in record["request"]["questions"]]


# --------------------------------------------------------------------------
# QA (docs/04 §6)
# --------------------------------------------------------------------------


def test_report_counts_states_questions_and_groups(batch, report):
    assert report["version"]
    assert report["invalid_records"] == 0
    assert report["errors"] == []
    assert report["states"] == len(batch)
    assert report["episodes"] == 0
    assert report["ticks"] == 0
    assert report["questions"] == sum(
        len(record["request"]["questions"]) for record in batch
    )
    assert sum(report["domains"].values()) == len(batch)
    assert sum(report["question_types"].values()) == report["questions"]
    assert sum(report["label_kinds"].values()) == report["labels"]
    groups = {record["origin_group"] for record in batch}
    assert sum(report["split_groups"].values()) == len(groups)
    assert sum(report["split_records"].values()) == len(batch)


def test_report_counts_episodes_and_ticks_for_streams():
    """D0 fixture로 스트림 집계 경로를 함께 본다."""
    records = read_jsonl(D0) + read_jsonl(D0_STREAMS)
    report = validate_dataset(records)
    assert report["invalid_records"] == 0
    assert report["states"] == 64
    assert report["episodes"] == 4
    assert report["ticks"] == sum(len(record["ticks"]) for record in read_jsonl(D0_STREAMS))
    assert report["errors"] == []


def test_stream_questions_count_only_what_a_tick_actually_poses():
    """틱이 던지지 않은 동적 질문(후보 없음)은 질문 수에 넣지 않는다."""
    streams = read_jsonl(D0_STREAMS)[:1]
    full = validate_dataset(copy.deepcopy(streams))["questions"]

    trimmed = copy.deepcopy(streams)
    tick = next(
        tick for tick in trimmed[0]["ticks"] if "q_path" in tick["request"]["candidates"]
    )
    tick["request"]["candidates"].pop("q_path")
    tick["labels"] = [label for label in tick["labels"] if label["question_id"] != "q_path"]

    report = validate_dataset(trimmed)
    assert report["invalid_records"] == 0, report["errors"]
    assert report["questions"] == full - 1


def test_report_flags_a_split_conflict_in_one_group(batch):
    poisoned = copy.deepcopy(batch[:8])
    conflict = copy.deepcopy(poisoned[0])
    conflict["split"] = "dev" if conflict["split"] != "dev" else "test"
    conflict["request"]["request_id"] += "-conflict"
    poisoned.append(conflict)
    report = validate_dataset(poisoned)
    assert report["invalid_records"] == 0
    paths = [error["path"] for error in report["errors"]]
    assert "split" in paths, report["errors"]


def test_report_flags_duplicated_facts_across_groups(batch):
    poisoned = copy.deepcopy(batch[:8])
    leaked = copy.deepcopy(poisoned[0])
    leaked["origin_group"] = "spatial/leaked-family/9999"
    leaked["split"] = assign_split(leaked["origin_group"])
    leaked["request"]["request_id"] += "-leak"
    poisoned.append(leaked)
    report = validate_dataset(poisoned)
    messages = [error["message"] for error in report["errors"]]
    assert any("group" in message and "사실" in message for message in messages), report["errors"]
    assert report["duplicate_content"]["cross_group"] == 1


def test_report_flags_a_paraphrase_parent_in_another_group(batch):
    poisoned = copy.deepcopy(batch[:8])
    poisoned[1]["provenance"]["derived_from"] = poisoned[0]["request"]["request_id"]
    poisoned[1]["provenance"]["derivation"] = "paraphrase"
    poisoned[1]["origin_group"] = "spatial/other-family/4242"
    poisoned[1]["split"] = assign_split(poisoned[1]["origin_group"])
    report = validate_dataset(poisoned)
    paths = [error["path"] for error in report["errors"]]
    assert "provenance.derived_from" in paths, report["errors"]


@pytest.mark.parametrize("derivation", ["paraphrase", "reorder"])
def test_report_flags_a_derivation_whose_facts_moved(batch, derivation):
    """표현·순서만 바꿨다면서 사실이 다르면 계보가 거짓이다."""
    poisoned = copy.deepcopy(
        [record for record in batch if record["provenance"].get("derivation") == derivation][:1]
    )
    assert poisoned, derivation
    parent_id = poisoned[0]["provenance"]["derived_from"]
    poisoned.insert(0, copy.deepcopy(_find(batch, parent_id)))
    assert validate_dataset(poisoned)["errors"] == []

    poisoned[1]["request"]["state"]["observed_at_ms"] += 50
    report = validate_dataset(poisoned)
    messages = [error["message"] for error in report["errors"]]
    assert any("파생본" in message and "사실" in message for message in messages), report["errors"]


def _find(records: list[dict], request_id: str) -> dict:
    return next(
        record for record in records if record["request"]["request_id"] == request_id
    )


def test_report_flags_a_dangling_candidate_reference(batch):
    poisoned = copy.deepcopy(batch[:8])
    for record in poisoned:
        for question in record["request"]["questions"]:
            for criterion in question["criteria"]:
                if "ref" in criterion:
                    criterion["ref"] = "does-not-exist"
                    report = validate_dataset(poisoned)
                    assert any(
                        error["path"].endswith(".ref") for error in report["errors"]
                    ), report["errors"]
                    return
    pytest.fail("ref를 가진 후보가 없다")


def test_report_flags_a_missing_label_source(batch):
    poisoned = copy.deepcopy(batch[:8])
    poisoned[0]["labels"][0].pop("source", None)
    poisoned[0]["labels"][0].pop("rule", None)
    report = validate_dataset(poisoned)
    paths = [error["path"] for error in report["errors"]]
    assert "labels[0].source" in paths, report["errors"]


def test_report_flags_an_information_boundary_leak(batch):
    poisoned = copy.deepcopy(batch[:4])
    poisoned[0]["request"]["state"]["evidence"] = {"rule_trace": "정답이 새어 나간다"}
    report = validate_dataset(poisoned)
    assert report["invalid_records"] == 1
    paths = [error["path"] for error in report["errors"]]
    assert any(path.startswith("model_input") for path in paths), report["errors"]


def test_report_collects_every_invalid_record_instead_of_stopping(batch):
    poisoned = copy.deepcopy(batch[:6])
    for record in poisoned[:3]:
        record["labels"].append({"question_id": "q_missing", "kind": "single", "answer": True})
    report = validate_dataset(poisoned)
    assert report["invalid_records"] == 3
    assert len({error["index"] for error in report["errors"]}) == 3


def test_report_summarises_answer_positions(batch, report):
    summary = report["answer_position"]
    assert summary["questions"] > 0
    for size, entry in summary["by_candidate_count"].items():
        assert abs(sum(entry["shares"]) - 1.0) < 1e-6, size


def test_model_input_of_every_generated_record_hides_the_labels(batch):
    for record in batch:
        served = model_input(record)
        text = json.dumps(served, ensure_ascii=False)
        assert "rule_trace" not in text
        assert served["request"]["request_id"] == record["request"]["request_id"]


# --------------------------------------------------------------------------
# 설정과 CLI
# --------------------------------------------------------------------------


def test_pilot_config_file_matches_the_embedded_default():
    assert load_config(PILOT_CONFIG) == DEFAULT_CONFIG


def test_loading_a_partial_config_never_touches_the_embedded_default(tmp_path):
    """빠진 절은 기본값에서 오는데, 그 값을 고쳐도 모듈 전역이 따라 바뀌면 안 된다."""
    partial = tmp_path / "partial.yaml"
    partial.write_text("version: partial\n", encoding="utf-8")

    before = copy.deepcopy(DEFAULT_CONFIG)
    loaded = load_config(partial)
    assert loaded["version"] == "partial"
    loaded["split"]["holdout_domains"].append("spatial")
    loaded["domains"]["spatial"] = 1
    loaded["questions_per_state"].clear()
    assert DEFAULT_CONFIG == before


def test_config_changes_the_domain_mix():
    config = copy.deepcopy(DEFAULT_CONFIG)
    config["domains"] = {"rules": 100}
    records = generate_records(count=40, seed=4, config=config)
    assert {record["provenance"]["domain"] for record in records} == {"rules"}


def test_config_holdouts_move_a_domain_to_ood():
    config = copy.deepcopy(DEFAULT_CONFIG)
    config["split"]["holdout_domains"] = ["spatial"]
    records = generate_records(count=60, seed=4, config=config)
    splits = {
        record["split"] for record in records if record["provenance"]["domain"] == "spatial"
    }
    assert splits <= set(OOD_SPLITS) and splits


# --------------------------------------------------------------------------
# 봉인 holdout — 템플릿 변형·개념 계열 (docs/04 §5, 계약 v0.3)
# --------------------------------------------------------------------------


def test_the_pilot_holdouts_are_in_the_vocabularies_and_seal_whole_families(batch):
    split = DEFAULT_CONFIG["split"]
    vocabulary = domains.phrasing_vocabulary()
    known = {item for ids in vocabulary.values() for item in ids}
    assert set(split["holdout_templates"]) <= known and len(split["holdout_templates"]) == 4
    assert {item.split(".", 1)[0] for item in split["holdout_templates"]} == {"spatial", "dom", "workflow", "rules"}
    concepts = {item for ids in domains.CONCEPT_VOCABULARY.values() for item in ids}
    assert set(split["holdout_concepts"]) <= concepts and len(split["holdout_concepts"]) == 4
    assert {item.split(":", 1)[0] for item in split["holdout_concepts"]} == {"spatial", "dom", "workflow", "rules"}

    by_group = defaultdict(list)
    for record in batch:
        by_group[record["origin_group"]].append(record)
    sealed = 0
    for group, records in by_group.items():
        reasons = {tuple(record["provenance"]["holdout"]) for record in records}
        assert len(reasons) == 1  # 계열의 레코드는 같은 이유·같은 split
        tags = {TEMPLATE_TAG + p for r in records for p in r["provenance"]["phrasing"]} | {CONCEPT_TAG + c for r in records for c in r["provenance"]["concepts"]}
        expected = SplitPolicy.from_config(split).holdout_reasons(group, sorted(tags))
        assert list(next(iter(reasons))) == expected
        if expected:
            sealed += 1
            assert {record["split"] for record in records} == {ood_split(group)}
    assert 0 < sealed < len(by_group)
    # train 쪽에는 봉인 문구·개념이 하나도 없다.
    for record in batch:
        if record["split"] in OOD_SPLITS:
            continue
        assert not set(record["provenance"]["phrasing"]) & set(split["holdout_templates"])
        assert not set(record["provenance"]["concepts"]) & set(split["holdout_concepts"])


def test_the_pilot_sealed_share_lands_in_the_decided_band_per_domain_without_straddling():
    """리뷰 1 I2: 봉인 종류(분야마다 문구 변형 하나·개념 하나)는 그대로 두고 생성 비중으로 OOD를 분야마다 ≈10~15 %(계열 기준,
    2,000상태·seed 17)에 맞춘다 — 봉인 문구 변형은 표에서 `sealed_phrasing_share`(10 %)만큼만 뽑히고, 봉인 개념의 근원이 되는
    장면(가운데 영역 목표·접힌 구역·오프라인 자원·안내 요청 조치)은 장면 생성 비중이 드물게 둔다. 한 계열이 두 split에 걸치지
    않고, ood_dev/ood_test는 둘 다 쓰인다."""
    records = generate_records(2000, 17)
    by_group = defaultdict(list)
    for record in records:
        by_group[record["origin_group"]].append(record)
    assert all(len({record["split"] for record in records}) == 1 for records in by_group.values())
    for domain in DEFAULT_CONFIG["domains"]:
        groups = {group: rows for group, rows in by_group.items() if group.startswith(domain + "/")}
        sealed = [rows for rows in groups.values() if rows[0]["provenance"]["holdout"]]
        share = len(sealed) / len(groups)
        assert 0.10 <= share <= 0.15, (domain, share, len(sealed), len(groups))
        assert any(reason.startswith("template:") for rows in sealed for reason in rows[0]["provenance"]["holdout"])
        assert any(reason.startswith("concept:") for rows in sealed for reason in rows[0]["provenance"]["holdout"])
        halves = Counter(rows[0]["split"] for rows in sealed)
        assert halves["ood_dev"] > 0 and halves["ood_test"] > 0
    assert validate_dataset(records, holdouts=DEFAULT_CONFIG["split"])["holdouts"]["leaked_groups"] == 0


def test_sealed_phrasing_variants_are_drawn_with_the_configured_share_and_other_tables_are_untouched():
    """`_say`는 봉인 변형이 든 표에서만 비중을 쓴다(봉인 변형에 `share` %, 나머지에 그 나머지를 고르게); 봉인 변형이 없는 표는
    고르게 뽑고 난수 소비도 도입 전과 같다(`randrange`)."""
    import random

    table = {"_id": "spatial.distance", "ko": ("하나 {x}", "둘 {x}"), "en": ("one {x}", "two {x}")}
    domains.begin_phrasing(sealed=["spatial.distance.ko#1"], share=10)
    rng = random.Random(3)
    drawn = Counter(domains._say(rng, table, "ko", x=1) for _ in range(4000))
    assert abs(drawn["둘 1"] / 4000 - 0.10) < 0.02
    assert domains.take_phrasing() == ["spatial.distance.ko#0", "spatial.distance.ko#1"]
    # 봉인이 없는 언어·표는 고르게, 그리고 봉인 없이 부를 때와 같은 난수 흐름이다.
    domains.begin_phrasing(sealed=["spatial.distance.ko#1"], share=10)
    rng = random.Random(9)
    with_seal = [domains._say(rng, table, "en", x=i) for i in range(50)]
    domains.begin_phrasing()
    rng = random.Random(9)
    without = [domains._say(rng, table, "en", x=i) for i in range(50)]
    assert with_seal == without and 0.3 < sum(text.startswith("two") for text in without) / 50 < 0.7
    with pytest.raises(ValueError, match="sealed_phrasing_share"):
        domains.begin_phrasing(sealed=[], share=100)
    domains.begin_phrasing()
    with pytest.raises(ValueError, match="sealed_phrasing_share"):
        generate_records(1, 1, config={"sealed_phrasing_share": 0})


def test_every_record_names_its_phrasing_variants_and_concepts(batch):
    for record in batch:
        provenance = record["provenance"]
        assert provenance["phrasing"] and all("#" in item and "." in item for item in provenance["phrasing"])
        assert provenance["concepts"] and all(item.startswith(provenance["domain"] + ":") for item in provenance["concepts"])
        kinds = {item.split(":", 1)[1].split(":")[0] for item in provenance["concepts"]}
        assert kinds <= {c.split(":", 1)[1].split(":")[0] for c in domains.CONCEPT_VOCABULARY[provenance["domain"]]}
        assert provenance["phrasing"] == sorted(set(provenance["phrasing"]))


def test_the_qa_report_counts_holdouts_and_flags_leaks(batch):
    split = DEFAULT_CONFIG["split"]
    report = validate_dataset(batch, holdouts=split)
    holdouts = report["holdouts"]
    assert holdouts["policy"]["holdout_templates"] == sorted(split["holdout_templates"])
    assert set(holdouts["groups"]) <= set(OOD_SPLITS) and sum(holdouts["groups"].values()) > 0
    assert holdouts["leaked_groups"] == 0 and report["errors"] == []
    assert {reason.split(":", 1)[0] for reason in holdouts["by_reason"]} == {"template", "concept"}
    for entry in holdouts["by_reason"].values():
        assert entry["groups"] > 0 and set(entry["splits"]) <= set(OOD_SPLITS)
    # 누출: 봉인 계열의 레코드를 train으로 옮기면 위반이다.
    leaked = copy.deepcopy(batch)
    victim = next(record for record in leaked if record["provenance"]["holdout"])
    victim["split"] = "train"
    report = validate_dataset(leaked, holdouts=split)
    assert report["holdouts"]["leaked_groups"] == 1
    assert any("누출" in error["message"] and victim["origin_group"] in error["message"] for error in report["errors"])
    # 이유 없는 OOD도 위반이다.
    orphan = copy.deepcopy(batch)
    plain = next(record for record in orphan if not record["provenance"]["holdout"])
    plain["split"] = "ood_test"
    report = validate_dataset(orphan, holdouts=split)
    assert any("holdout 이유가 없다" in error["message"] for error in report["errors"])
    # 봉인 목록 없이 부르면 누출 검사를 하지 않는다 (옛 데이터셋).
    assert validate_dataset(batch)["holdouts"]["policy"] is None


def test_cli_generate_and_validate_round_trip(tmp_path):
    dataset = tmp_path / "d1"
    assert (
        generate_main(
            [
                "--config",
                str(PILOT_CONFIG),
                "--count",
                "40",
                "--seed",
                "11",
                "--output",
                str(dataset / "single"),
            ]
        )
        == 0
    )
    records = read_jsonl(dataset / "single" / "records.jsonl")
    manifest = json.loads((dataset / "single" / "manifest.json").read_text(encoding="utf-8"))
    assert len(records) == 40
    assert manifest["seed"] == 11
    assert manifest["counts"]["states"] == 40
    assert manifest["files"]["records.jsonl"]["records"] == 40
    assert manifest["generator"]
    assert manifest["config_sha256"]

    import hashlib

    digest = hashlib.sha256((dataset / "single" / "records.jsonl").read_bytes()).hexdigest()
    assert manifest["files"]["records.jsonl"]["sha256"] == digest

    report_path = tmp_path / "reports" / "d1-qa.json"
    assert validate_main(["--dataset", str(dataset), "--report", str(report_path)]) == 0
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["invalid_records"] == 0
    assert report["states"] == 40
    assert report["errors"] == []


def test_cli_generate_is_byte_reproducible(tmp_path):
    outputs = []
    for run in ("a", "b"):
        target = tmp_path / run
        assert (
            generate_main(
                ["--count", "24", "--seed", "9", "--output", str(target)]
            )
            == 0
        )
        outputs.append(
            (
                (target / "records.jsonl").read_bytes(),
                (target / "manifest.json").read_bytes(),
            )
        )
    assert outputs[0] == outputs[1]


def test_cli_validate_reports_failure_with_a_non_zero_exit(tmp_path):
    dataset = tmp_path / "broken"
    dataset.mkdir()
    broken = {"schema_version": "judgment-v0", "request": {}}
    (dataset / "records.jsonl").write_text(
        json.dumps(broken, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    report_path = tmp_path / "qa.json"
    assert validate_main(["--dataset", str(dataset), "--report", str(report_path)]) == 1
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["invalid_records"] == 1


def test_generate_module_runs_as_a_script(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "robo_jev.data.generate",
            "--count",
            "4",
            "--seed",
            "2",
            "--output",
            str(tmp_path / "tiny"),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert (tmp_path / "tiny" / "records.jsonl").exists()


def test_origin_prefix_renames_groups_and_request_ids_for_a_disjoint_dataset_lineage():
    """D-OOD (`configs/data/d_ood_single.yaml`): 다른 seed의 계열은 새 장면이지만 이름 `<분야>/<장면>/<번호>`는 D1과 겹친다 — `origin_prefix`가
    group과 요청 id에 접두사를 붙여 계보를 가른다. 접두사 없이는(기본 "") 레코드가 그대로다."""
    config = copy.deepcopy(DEFAULT_CONFIG)
    plain = generate_records(count=60, seed=9001, config=config)
    config["origin_prefix"] = "dood"
    prefixed = generate_records(count=60, seed=9001, config=config)
    assert len(plain) == len(prefixed) == 60
    for a, b in zip(plain, prefixed):
        assert b["origin_group"] == "dood/" + a["origin_group"] and b["provenance"]["origin_group"] == b["origin_group"]
        assert b["request"]["request_id"] == "dood-" + a["request"]["request_id"]
        derived = b["provenance"].get("derived_from")
        if derived is not None:
            assert derived.startswith("dood-") and derived == "dood-" + a["provenance"]["derived_from"]
        # 문구·후보 배열의 seed 문자열에 group이 들어가므로 후보 id·문장은 달라진다; 장면 계열(분야·장면·번호·언어·질문 종류)은 같다.
        for key in ("domain", "template", "family", "language", "derivation"):
            assert b["provenance"].get(key) == a["provenance"].get(key)
        assert [q["type"] for q in b["request"]["questions"]] == [q["type"] for q in a["request"]["questions"]]
        validate_record(b)
    assert generate_records(count=20, seed=9001, config={**DEFAULT_CONFIG, "origin_prefix": ""}) == plain[:20]
    assert DEFAULT_CONFIG["origin_prefix"] == "" and load_config(Path(__file__).resolve().parent.parent / "configs" / "data" / "pilot.yaml")["origin_prefix"] == ""
    ood = generate_records(count=40, seed=9001, config={**config, "split": {**config["split"], "holdout_prefixes": ["dood/"]}})
    assert {record["split"] for record in ood} <= {"ood_dev", "ood_test"} and len({record["split"] for record in ood}) == 2
