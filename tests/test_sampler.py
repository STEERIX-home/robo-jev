"""sampler 검사 — 읽기 전용 적재, 틱 종류, 두 sampler 축(로봇/비로봇 토큰, 70/20/10), 결정성·재개 (docs/04 §2·§7, docs/06 Task 5).

step마다 로봇 스트림 단위(에피소드 하나)와 비로봇 단위(토큰 예산까지 묶은 단일 요청 microbatch)를
**둘 다** 넣는다(docs/04 §2, 판정 e086c90) — 60/40은 학습 loop의 손실 비중이고 sampler는 두 종류를
번갈아 뽑을 뿐이다. 기존 자료/오류 계열/새 의미 계열의 70/20/10은 별개의 축이다(레코드의 provenance
태그로 나눈다; D0/D1에서는 뒤의 두 묶음이 비어 있을 수 있으므로 재정규화하고 실현 비중을 기록한다).
정상 유지 틱은 하향, 이벤트·목표 변경 틱은 상향 가중하며 틱의 종류는 그 틱의 레코드·라벨에서만
정한다(미래 틱을 보지 않는다).
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
    valid_single,
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


def test_loader_accepts_both_manifest_shapes_for_files(tmp_path):
    """`files`는 dict(`{경로: {sha256, …}}` — 비로봇·D0 manifest, 로봇 manifest도 쓰는 시점에 이 꼴로 맞춘다)이지만
    목록(`[{path, sha256, …}]` — 이전 로봇 manifest)도 같은 뜻으로 읽는다. 둘 다 sha256을 대조한다."""
    import hashlib

    manifest = json.loads(D0_MANIFEST.read_text(encoding="utf-8"))
    (tmp_path / "d0.jsonl").write_bytes(D0_MANIFEST.with_name("d0.jsonl").read_bytes())
    digest = hashlib.sha256((tmp_path / "d0.jsonl").read_bytes()).hexdigest()
    as_dict = {**manifest, "files": {"d0.jsonl": {"sha256": digest}}}
    as_list = {**manifest, "files": [{"path": "d0.jsonl", "sha256": digest, "episode_id": "x"}]}
    (tmp_path / "dict.json").write_text(json.dumps(as_dict), encoding="utf-8")
    (tmp_path / "list.json").write_text(json.dumps(as_list), encoding="utf-8")
    from_dict = load_items(tmp_path / "dict.json", tokenizer=WhitespaceTokenizer())
    from_list = load_items(tmp_path / "list.json", tokenizer=WhitespaceTokenizer())
    assert [item.record_id for item in from_dict] == [item.record_id for item in from_list]
    assert len(from_dict) == 32 and from_dict[0].source == from_list[0].source
    (tmp_path / "bad.json").write_text(json.dumps({**manifest, "files": [{"path": "d0.jsonl", "sha256": "0" * 64}]}), encoding="utf-8")
    with pytest.raises(ValueError, match="sha256"):
        load_items(tmp_path / "bad.json", tokenizer=WhitespaceTokenizer())
    (tmp_path / "nopath.json").write_text(json.dumps({**manifest, "files": [{"sha256": digest}]}), encoding="utf-8")
    with pytest.raises(ValueError, match="path"):
        load_items(tmp_path / "nopath.json", tokenizer=WhitespaceTokenizer())


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


def test_valid_single_mirrors_the_same_rules(singles):
    record = copy.deepcopy(singles[0])
    assert valid_single(record)
    record["labels"] = [dict(l, mask=False) for l in record["labels"]]
    assert not valid_single(record)
    record["labels"] = []
    assert not valid_single(record)
    # weight 0 라벨만 있는 상태는 손실이 세지 않는다 (judgment_loss의 total_weight <= 0) — 같은 규칙.
    record["labels"] = [dict(l, weight=0.0) for l in singles[0]["labels"]]
    assert not valid_single(record)
    record["labels"] = [dict(l, weight=0.25) for l in singles[0]["labels"]]  # 낮은 신뢰도의 내린 weight는 기여한다
    assert valid_single(record)


# --------------------------------------------------------------------------
# 두 축: 로봇/비로봇 토큰, 70/20/10 (재정규화)
# --------------------------------------------------------------------------


def ids(units, items):
    return [tuple(items[i].record_id for i in unit.items) for unit in units]


def test_sampler_is_deterministic_under_seed_and_resumable_from_a_saved_position(items):
    a = MixedSampler(items, seed=17, nonrobot_tokens_per_unit=400)
    b = MixedSampler(items, seed=17, nonrobot_tokens_per_unit=400)
    c = MixedSampler(items, seed=18, nonrobot_tokens_per_unit=400)
    drawn_a = [unit for _ in range(30) for unit in a.draw_step(2)]
    drawn_b = [unit for _ in range(30) for unit in b.draw_step(2)]
    assert ids(drawn_a, items) == ids(drawn_b, items)
    assert ids([unit for _ in range(30) for unit in c.draw_step(2)], items) != ids(drawn_a, items)
    assert [u.index for u in drawn_a] == list(range(60))
    # 10 step(20 단위) 뒤에 저장한 위치에서 새 sampler가 이어간다
    fresh = MixedSampler(items, seed=17, nonrobot_tokens_per_unit=400)
    for _ in range(10):
        fresh.draw_step(2)
    position = fresh.state_dict()
    resumed = MixedSampler(items, seed=17, nonrobot_tokens_per_unit=400)
    resumed.load_state_dict(position)
    assert resumed.state_dict() == position and position["drawn"] == 20
    assert ids([unit for _ in range(20) for unit in resumed.draw_step(2)], items) == ids(drawn_a[20:], items)
    assert resumed.state_dict() == a.state_dict()
    # 위치는 저장 단위에 들어가는 기본 자료형뿐이다
    json.dumps(position)


def test_every_step_alternates_robot_and_nonrobot_units_and_records_token_share(items):
    """step의 단위는 로봇(에피소드), 비로봇(묶음), 로봇, … — 누적 토큰 비중을 좇지 않는다 (판정 e086c90)."""
    sampler = MixedSampler(items, seed=3, nonrobot_tokens_per_unit=400)
    for count in (2, 3, 4):
        units = sampler.draw_step(count)
        assert [u.domain for u in units] == ["robot", "non_robot", "robot", "non_robot"][:count]
        assert [u.kind for u in units] == ["stream", "single", "stream", "single"][:count]
        assert all(len(u.items) == 1 for u in units if u.kind == "stream")
    with pytest.raises(ValueError, match="units"):
        sampler.draw_step(0)
    realized = sampler.realized()
    assert realized["tokens"]["robot"] > realized["tokens"]["non_robot"] > 0  # 기록만 한다
    assert realized["token_share"]["robot"] + realized["token_share"]["non_robot"] == pytest.approx(1.0)
    assert realized["drawn"] == 9 and realized["units"]["robot/existing"] == 5


def test_nonrobot_units_pack_singles_up_to_the_token_budget_without_losing_items(items):
    budget = 400
    sampler = MixedSampler(items, seed=5, nonrobot_tokens_per_unit=budget)
    packed = [sampler.draw("non_robot") for _ in range(12)]
    singles = {item.index: item for item in items if item.kind == "single"}
    for unit in packed:
        assert unit.kind == "single" and unit.domain == "non_robot" and len(unit.items) >= 2
        assert unit.tokens == sum(singles[i].tokens for i in unit.items) <= budget
        assert len(unit.materials) == len(unit.items)
    # 예산을 넘겨서 되돌려 둔 레코드는 다음 단위에 들어간다: 한 epoch 안에서 모든 단일 요청이 정확히 한 번
    drawn = [i for unit in packed for i in unit.items]
    first_epoch = drawn[: len(singles)]
    assert sorted(first_epoch) == sorted(singles)
    assert sampler.realized()["epochs"]["non_robot/existing"] >= 1
    # 예산이 레코드 하나보다 작으면 그 레코드 하나로 단위를 만든다 (빈 단위는 없다)
    tiny = MixedSampler(items, seed=5, nonrobot_tokens_per_unit=1)
    unit = tiny.draw("non_robot")
    assert len(unit.items) == 1 and unit.tokens > 1
    with pytest.raises(ValueError, match="nonrobot_tokens_per_unit"):
        MixedSampler(items, seed=5, nonrobot_tokens_per_unit=0)
    with pytest.raises(ValueError, match="domain"):
        sampler.draw("space")


def test_material_axis_renormalises_over_empty_buckets_and_records_the_share(items):
    sampler = MixedSampler(items, seed=5, nonrobot_tokens_per_unit=400)
    for _ in range(25):
        sampler.draw_step(2)
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
    sampler = MixedSampler(tagged, seed=7, nonrobot_tokens_per_unit=400)
    counts = {m: 0 for m in MATERIALS}
    total = 0
    while total < 3000:
        unit = sampler.draw("non_robot")
        assert unit.kind == "single"
        for material in unit.materials:
            counts[material] += 1
        total += len(unit.items)
    assert abs(counts["existing"] / total - 0.7) < 0.03
    assert abs(counts["error_family"] / total - 0.2) < 0.03
    assert abs(counts["new_semantic_family"] / total - 0.1) < 0.03


def test_sampler_handles_one_sided_data(items):
    singles_only = [item for item in items if item.kind == "single"]
    sampler = MixedSampler(singles_only, seed=1, nonrobot_tokens_per_unit=400)
    assert sampler.domains == ("non_robot",)
    assert all(u.kind == "single" for _ in range(3) for u in sampler.draw_step(2))  # 로봇이 없으면 전부 비로봇
    assert sampler.realized()["token_share"]["robot"] == 0.0
    with pytest.raises(ValueError, match="domain"):
        sampler.draw("robot")
    streams_only = [item for item in items if item.kind == "stream"]
    sampler = MixedSampler(streams_only, seed=1)
    assert sampler.domains == ("robot",)
    assert [u.kind for u in sampler.draw_step(2)] == ["stream", "stream"]
    with pytest.raises(ValueError, match="items"):
        MixedSampler([], seed=1)


def test_every_item_of_a_bucket_is_drawn_once_per_epoch(items):
    sampler = MixedSampler(items, seed=9)
    drawn = [sampler.draw("robot").items[0] for _ in range(2)]
    assert sorted(drawn) == sorted(item.index for item in items if item.kind == "stream")
    second = [sampler.draw("robot").items[0] for _ in range(2)]
    assert sorted(second) == sorted(drawn)
    assert sampler.realized()["epochs"]["robot/existing"] == 2
    many = MixedSampler(items, seed=9, nonrobot_tokens_per_unit=70)  # 대체로 하나씩
    drawn = []
    while len(drawn) < 64:
        drawn.extend(many.draw("non_robot").items)
    assert sorted(drawn[:32]) == sorted(item.index for item in items if item.kind == "single")
    assert sorted(drawn[32:64]) == sorted(drawn[:32]) and drawn[32:64] != drawn[:32]  # 다음 epoch는 다시 섞인다


def test_sampler_position_names_the_record_files_and_refuses_a_position_over_different_records(tmp_path, items):
    """저장 위치는 뽑은 순서·cursor뿐 아니라 **그 index가 가리키는 레코드의 출처**(파일별 sha256, 적재 순서, 레코드 수)를
    적는다. 같은 index가 다른 레코드를 가리키게 된 데이터(정답 하나를 바꾸고 manifest 해시를 맞춘 사본)에는 그 위치를
    싣지 않는다 (리뷰 11 S1)."""
    import hashlib

    manifest = json.loads(D0_MANIFEST.read_text(encoding="utf-8"))
    sampler = MixedSampler(items, seed=17, nonrobot_tokens_per_unit=400)
    sampler.draw_step(2)
    position = sampler.state_dict()
    assert position["sources"] == [
        {"file": "d0.jsonl", "sha256": manifest["files"]["d0.jsonl"]["sha256"], "first_index": 0, "items": 32},
        {"file": "d0_streams.jsonl", "sha256": manifest["files"]["d0_streams.jsonl"]["sha256"], "first_index": 32, "items": 2},
    ]
    assert all(item.file_sha256 == manifest["files"][item.file]["sha256"] for item in items)

    for name in ("d0.jsonl", "d0_streams.jsonl"):
        (tmp_path / name).write_bytes(D0_MANIFEST.with_name(name).read_bytes())
    rows = [json.loads(line) for line in (tmp_path / "d0.jsonl").read_text(encoding="utf-8").splitlines()]
    assert rows[0]["labels"][0]["candidate_ids"] == ["c0"]
    rows[0]["labels"][0]["candidate_ids"] = ["c1"]
    (tmp_path / "d0.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    changed = hashlib.sha256((tmp_path / "d0.jsonl").read_bytes()).hexdigest()
    manifest["files"]["d0.jsonl"]["sha256"] = changed
    (tmp_path / "d0_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    tampered = load_items(tmp_path / "d0_manifest.json", tokenizer=WhitespaceTokenizer(), splits=("train",), stream_max_ticks=20)
    assert [i.index for i in tampered] == [i.index for i in items]  # 같은 index·같은 묶음 — 내용만 다르다
    with pytest.raises(ValueError, match="sources") as excinfo:
        MixedSampler(tampered, seed=17, nonrobot_tokens_per_unit=400).load_state_dict(position)
    message = str(excinfo.value)
    assert "d0.jsonl" in message and changed[:12] in message and "d0_streams.jsonl" not in message
    # 같은 파일(다른 경로의 사본)이면 싣는다
    (tmp_path / "d0.jsonl").write_bytes(D0_MANIFEST.with_name("d0.jsonl").read_bytes())
    (tmp_path / "d0_manifest.json").write_bytes(D0_MANIFEST.read_bytes())
    same = load_items(tmp_path / "d0_manifest.json", tokenizer=WhitespaceTokenizer(), splits=("train",), stream_max_ticks=20)
    resumed = MixedSampler(same, seed=17, nonrobot_tokens_per_unit=400)
    resumed.load_state_dict(position)
    assert resumed.state_dict() == position
