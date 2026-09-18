"""sampler 검사 — 읽기 전용 적재, 틱 종류, 두 sampler 축(로봇/비로봇 토큰, 70/20/10), 결정성·재개 (docs/04 §2·§7, docs/06 Task 5).

로봇/비로봇 비중은 상태 수가 아니라 **토큰**으로 관리하고(시작 60/40), 기존 자료/오류 계열/새 의미
계열의 70/20/10은 별개의 축이다(레코드의 provenance 태그로 나눈다; D0/D1에서는 뒤의 두 묶음이
비어 있을 수 있으므로 재정규화하고 실현 비중을 기록한다). 정상 유지 틱은 하향, 이벤트·목표 변경
틱은 상향 가중하며 틱의 종류는 그 틱의 레코드·라벨에서만 정한다(미래 틱을 보지 않는다).
"""

import copy
import json
import random

import pytest
from helpers import D0_MANIFEST

from robo_jev.model.tokenizer import WhitespaceTokenizer
from robo_jev.sampler import (
    MATERIALS,
    TICK_CLASSES,
    MixedSampler,
    load_items,
    tick_class,
    tick_weights,
    valid_label_ticks,
)

TICK_WEIGHTS = {"steady": 0.25, "event": 2.0, "goal_change": 2.0, "other": 1.0}


@pytest.fixture(scope="module")
def items():
    return load_items(D0_MANIFEST, tokenizer=WhitespaceTokenizer(), splits=("train",), stream_max_ticks=20)


# --------------------------------------------------------------------------
# 읽기 전용 적재
# --------------------------------------------------------------------------


def test_loader_reads_the_manifest_files_checks_hashes_and_serializes(items):
    kinds = [item.kind for item in items]
    assert kinds.count("single") == 32 and kinds.count("stream") == 2  # D0 train split
    assert all(item.split == "train" for item in items)
    assert [item.index for item in items] == list(range(len(items)))
    single = next(item for item in items if item.kind == "single")
    stream = next(item for item in items if item.kind == "stream")
    assert single.layout["layout"] == "state_first" and stream.layout["layout"] == "stream_l1a"
    assert single.tokens == len(single.layout["tokens"]) > 0
    assert stream.tokens == len(stream.layout["tokens"]) and len(stream.record["ticks"]) == 20
    assert single.domain == "non_robot" and stream.domain == "robot"  # 태그가 없으면 레코드 종류로
    assert single.material == stream.material == "existing"  # 태그가 없으면 기존 자료
    assert single.question_types == {q["id"]: q["type"] for q in single.record["request"]["questions"]}
    assert stream.question_types["q_main"] == "choice" and stream.question_types["q_speed"] == "ordinal"
    assert single.record_id.startswith("d0-") and stream.record_id.startswith("ep-d0-")


def test_loader_rejects_tampered_files_unknown_splits_and_missing_manifest(tmp_path):
    manifest = json.loads(D0_MANIFEST.read_text(encoding="utf-8"))
    tampered = tmp_path / "d0_manifest.json"
    (tmp_path / "d0.jsonl").write_text(D0_MANIFEST.with_name("d0.jsonl").read_text(encoding="utf-8") + "\n")
    (tmp_path / "d0_streams.jsonl").write_bytes(D0_MANIFEST.with_name("d0_streams.jsonl").read_bytes())
    tampered.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="sha256"):
        load_items(tampered, tokenizer=WhitespaceTokenizer())
    with pytest.raises(ValueError, match="split"):
        load_items(D0_MANIFEST, tokenizer=WhitespaceTokenizer(), splits=("validation",))
    with pytest.raises(FileNotFoundError):
        load_items(tmp_path / "nope.json", tokenizer=WhitespaceTokenizer())


def test_loader_reads_domain_and_material_tags_and_rejects_unknown_values(tmp_path):
    manifest = json.loads(D0_MANIFEST.read_text(encoding="utf-8"))
    records = [json.loads(line) for line in D0_MANIFEST.with_name("d0.jsonl").read_text(encoding="utf-8").splitlines()]
    records[0]["provenance"]["material"] = "error_family"
    records[1]["provenance"]["material"] = "new_semantic_family"
    records[2]["provenance"]["domain"] = "robot"
    text = "\n".join(json.dumps(r, ensure_ascii=False) for r in records) + "\n"
    (tmp_path / "d0.jsonl").write_text(text, encoding="utf-8")
    import hashlib

    manifest["files"] = {"d0.jsonl": {"sha256": hashlib.sha256(text.encode("utf-8")).hexdigest()}}
    (tmp_path / "m.json").write_text(json.dumps(manifest), encoding="utf-8")
    loaded = load_items(tmp_path / "m.json", tokenizer=WhitespaceTokenizer())
    assert [item.material for item in loaded[:3]] == ["error_family", "new_semantic_family", "existing"]
    assert [item.domain for item in loaded[:3]] == ["non_robot", "non_robot", "robot"]
    records[3]["provenance"]["material"] = "fresh"
    text = "\n".join(json.dumps(r, ensure_ascii=False) for r in records) + "\n"
    (tmp_path / "d0.jsonl").write_text(text, encoding="utf-8")
    manifest["files"]["d0.jsonl"]["sha256"] = hashlib.sha256(text.encode("utf-8")).hexdigest()
    (tmp_path / "m.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="material"):
        load_items(tmp_path / "m.json", tokenizer=WhitespaceTokenizer())


# --------------------------------------------------------------------------
# 틱 종류와 가중치
# --------------------------------------------------------------------------


def test_tick_classes_come_from_the_ticks_own_record(streams):
    first = next(s for s in streams if s["episode_id"] == "ep-d0-001")
    classes = [tick_class(first, i, steady_min_held_ticks=3) for i in range(len(first["ticks"]))]
    assert set(classes) <= set(TICK_CLASSES)
    assert classes[55] == "goal_change"  # 지시 v2가 goal.version을 올리는 틱
    assert classes[5] == classes[20] == "event"  # adopted.switch
    assert classes[0] == "other"  # commitment 없음
    assert classes[10] == "steady" and classes[6] == "other"  # held_ticks 5 vs 1
    assert classes.count("steady") > 40
    second = next(s for s in streams if s["episode_id"] == "ep-d0-002")
    assert tick_class(second, 40) == "event"  # state.events의 slip
    # 미래 틱을 보지 않는다: 뒤를 잘라도 앞 틱의 종류가 같다
    truncated = copy.deepcopy(first)
    truncated["ticks"] = truncated["ticks"][:56]
    assert [tick_class(truncated, i) for i in range(56)] == classes[:56]
    weights = tick_weights(first, weights=TICK_WEIGHTS)
    assert weights[55] == 2.0 and weights[10] == 0.25 and weights[0] == 1.0 and len(weights) == 100
    with pytest.raises(ValueError, match="tick_weights"):
        tick_weights(first, weights={"steady": 0.25})


def test_valid_label_ticks_mirror_the_loss_mask_rules(streams):
    record = copy.deepcopy(streams[0])
    record["ticks"] = record["ticks"][:4]
    record["ticks"][1]["labels"] = [dict(l, mask=False) for l in record["ticks"][1]["labels"]]
    record["ticks"][2]["labels"] = []
    record["ticks"][3]["labels"] = [
        {"question_id": "q_stop", "kind": "event", "successes": 0, "failures": 0, "censored": 0, "event_id": "e"}
    ]
    assert valid_label_ticks(record) == [True, False, False, False]


# --------------------------------------------------------------------------
# 두 축: 로봇/비로봇 토큰, 70/20/10 (재정규화)
# --------------------------------------------------------------------------


def ids(units, items):
    return [tuple(items[i].record_id for i in unit.items) for unit in units]


def test_sampler_is_deterministic_under_seed_and_resumable_from_a_saved_position(items):
    a = MixedSampler(items, seed=17)
    b = MixedSampler(items, seed=17)
    c = MixedSampler(items, seed=18)
    drawn_a = [a.draw() for _ in range(60)]
    drawn_b = [b.draw() for _ in range(60)]
    assert ids(drawn_a, items) == ids(drawn_b, items)
    assert ids([c.draw() for _ in range(60)], items) != ids(drawn_a, items)
    assert [u.index for u in drawn_a] == list(range(60))
    # 20개 뒤에 저장한 위치에서 새 sampler가 이어간다
    fresh = MixedSampler(items, seed=17)
    for _ in range(20):
        fresh.draw()
    position = fresh.state_dict()
    resumed = MixedSampler(items, seed=17)
    resumed.load_state_dict(position)
    assert resumed.state_dict() == position and position["drawn"] == 20
    assert ids([resumed.draw() for _ in range(40)], items) == ids(drawn_a[20:], items)
    assert resumed.state_dict() == a.state_dict()
    # 위치는 저장 단위에 들어가는 기본 자료형뿐이다
    json.dumps(position)


def test_robot_share_is_steered_by_tokens_and_realised_share_is_recorded(items):
    sampler = MixedSampler(items, seed=3, robot_token_share=0.6)
    first = sampler.draw()
    assert first.kind == "stream" and first.domain == "robot"  # 0 < 0.6 → 로봇부터
    units = [first] + [sampler.draw() for _ in range(400)]
    realized = sampler.realized()
    total = realized["tokens"]["robot"] + realized["tokens"]["non_robot"]
    assert total == sum(u.tokens for u in units)
    biggest = max(u.tokens for u in units)
    assert abs(realized["token_share"]["robot"] - 0.6) <= biggest / total  # 단위 하나의 크기 안에서 목표를 따른다
    assert realized["token_share"]["robot"] > 0.5
    kinds = [u.kind for u in units]
    assert kinds.count("stream") >= 2 and kinds.count("single") > 300
    # 두 에피소드가 한 epoch를 이루고 다시 섞인다
    assert realized["epochs"]["robot/existing"] >= 1


def test_material_axis_renormalises_over_empty_buckets_and_records_the_share(items):
    sampler = MixedSampler(items, seed=5)
    for _ in range(50):
        sampler.draw()
    realized = sampler.realized()
    assert set(realized["unit_share"]["material"]) == set(MATERIALS)
    assert realized["unit_share"]["material"]["existing"] == 1.0  # D0에는 기존 자료뿐 — 실패하지 않고 기록한다
    assert realized["unit_share"]["material"]["error_family"] == 0.0
    assert realized["effective_material_shares"]["non_robot"] == {"existing": 1.0, "error_family": 0.0, "new_semantic_family": 0.0}


def test_material_axis_follows_70_20_10_when_all_buckets_exist(items):
    tagged = copy.deepcopy(items)
    rng = random.Random(0)
    for item in tagged:
        if item.kind == "single":
            item.material = rng.choice(MATERIALS)
    sampler = MixedSampler(tagged, seed=7, robot_token_share=0.0)
    counts = {m: 0 for m in MATERIALS}
    for _ in range(3000):
        unit = sampler.draw()
        assert unit.kind == "single"
        counts[unit.materials[0]] += 1
    assert abs(counts["existing"] / 3000 - 0.7) < 0.03
    assert abs(counts["error_family"] / 3000 - 0.2) < 0.03
    assert abs(counts["new_semantic_family"] / 3000 - 0.1) < 0.03
    assert sampler.realized()["token_share"]["robot"] == 0.0


def test_sampler_handles_one_sided_data_and_microbatches(items):
    singles_only = [item for item in items if item.kind == "single"]
    sampler = MixedSampler(singles_only, seed=1, robot_token_share=0.6)
    assert all(sampler.draw().kind == "single" for _ in range(10))
    assert sampler.realized()["token_share"]["robot"] == 0.0
    streams_only = [item for item in items if item.kind == "stream"]
    sampler = MixedSampler(streams_only, seed=1, robot_token_share=0.0)
    assert sampler.draw().kind == "stream"
    with pytest.raises(ValueError, match="items"):
        MixedSampler([], seed=1)
    batched = MixedSampler(items, seed=2, microbatch=3)
    units = [batched.draw() for _ in range(5)]
    for unit in units:
        if unit.kind == "single":
            assert len(unit.items) == 3 and len(unit.materials) == 3
            assert unit.tokens == sum(items[i].tokens for i in unit.items)
        else:
            assert len(unit.items) == 1
    assert any(u.kind == "single" for u in units)


def test_every_item_of_a_bucket_is_drawn_once_per_epoch(items):
    sampler = MixedSampler(items, seed=9, robot_token_share=0.0)
    drawn = [sampler.draw().items[0] for _ in range(32)]
    assert sorted(drawn) == sorted(item.index for item in items if item.kind == "single")
    second = [sampler.draw().items[0] for _ in range(32)]
    assert sorted(second) == sorted(drawn) and second != drawn  # 다음 epoch는 다시 섞인다
    assert sampler.realized()["epochs"]["non_robot/existing"] == 2
