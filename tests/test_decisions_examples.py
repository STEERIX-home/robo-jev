"""`robo_jev.decisions.examples` — 판단 요청 → 학습 예 (Task v0d §1).

학습 쪽 프롬프트가 진단 도구의 것과 **바이트 단위로 같은지**(같은 함수, 같은 순서면 같은 글·같은 토큰), 순서 증강에서 정답이
선택지와 함께 옮겨지는지, score 정답 → 6구간 분포의 기대값이 정답과 같은지(읽을 수 있는 범위 [0.05, 0.95] 안; 밖은 끝 구간),
질문 뽑기(요청마다 최대 q개, 종류별로 고르게), 레코드 id 해시의 dev 나눔(쌍은 함께), epoch 계획의 결정성, 프롬프트에 요청 말고는
아무것도 들어가지 않음을 본다. 모델은 쓰지 않는다(tokenizer만).

Task v0d fix round 1: 기본 판 v0c-p3에서는 score 정답이 [0, 1] 어디든 이웃한 두 기준점의 선형 보간으로 **그대로** 닿는다(잘림 없음);
v0c-p2(재현용)의 끝 구간 잘림은 그대로 남는다.
"""

import collections
import copy
import functools
import math
import random

import pytest
from helpers import FIXTURES

from robo_jev.decisions import examples as E
from robo_jev.decisions.prompt import SCORE_ANCHORS, SCORE_CENTERS, answer_from_codes, check_displayed_lines, record_prompts
from robo_jev.decisions.records import read_records, shuffle_key
from robo_jev.model.tokenizer import available_tokenizer, load_tokenizer

REQUESTS = FIXTURES / "decisions" / "requests.jsonl"


@functools.lru_cache(maxsize=1)
def tokenizer():
    return load_tokenizer(available_tokenizer()[0])


@pytest.fixture(scope="module")
def fixtures() -> list[dict]:
    return read_records(REQUESTS)


# -- score 정답 → 구간 분포 --------------------------------------------------------------------------


@pytest.mark.parametrize("value", [0.0, 0.001, 0.03, 0.05, 0.1, 0.27, 0.43, 0.5, 0.71, 0.9, 0.95, 0.97, 0.999, 1.0])
def test_a_p3_score_label_anywhere_in_zero_one_becomes_two_adjacent_anchors_whose_mean_is_the_label(value):
    probs, clamped = E.score_target(value)  # 기본 판 v0c-p3
    assert not clamped and len(probs) == len(SCORE_ANCHORS) == 11
    assert sum(probs) == pytest.approx(1.0, abs=1e-12) and min(probs) >= 0.0
    assert sum(p * c for p, c in zip(probs, SCORE_ANCHORS)) == pytest.approx(value, abs=1e-12)
    nonzero = [i for i, p in enumerate(probs) if p > 0]
    assert 1 <= len(nonzero) <= 2 and nonzero[-1] - nonzero[0] <= 1
    assert E.SCORE_READABLE["v0c-p3"] == (0.0, 1.0)


@pytest.mark.parametrize("value", [0.05, 0.051, 0.1, 0.2, 0.27, 0.4, 0.43, 0.5, 0.6, 0.71, 0.8, 0.9, 0.949, 0.95])
def test_a_score_inside_the_readable_range_becomes_two_adjacent_bins_whose_mean_is_the_label(value):
    probs, clamped = E.score_target(value, version="v0c-p2")
    assert not clamped and len(probs) == len(SCORE_CENTERS)
    assert sum(probs) == pytest.approx(1.0, abs=1e-12) and min(probs) >= 0.0
    assert sum(p * c for p, c in zip(probs, SCORE_CENTERS)) == pytest.approx(value, abs=1e-12)
    nonzero = [i for i, p in enumerate(probs) if p > 0]
    assert 1 <= len(nonzero) <= 2 and nonzero[-1] - nonzero[0] <= 1


def test_a_score_on_a_bin_centre_is_one_bin_and_scores_outside_the_range_go_to_the_end_bin():
    """v0c-p2(재현용): 구간 중앙값 위면 한 구간, [0.05, 0.95] 밖은 끝 구간 + clamped."""
    for i, centre in enumerate(SCORE_CENTERS):
        probs, clamped = E.score_target(centre, version="v0c-p2")
        assert probs == [1.0 if j == i else 0.0 for j in range(6)] and not clamped
    for value in (0.0, 0.01, 0.049):
        assert E.score_target(value, version="v0c-p2") == ([1.0, 0.0, 0.0, 0.0, 0.0, 0.0], True)
    for value in (0.951, 0.99, 1.0):
        assert E.score_target(value, version="v0c-p2") == ([0.0, 0.0, 0.0, 0.0, 0.0, 1.0], True)
    assert E.SCORE_READABLE["v0c-p2"] == (0.05, 0.95)
    for i, anchor in enumerate(SCORE_ANCHORS):  # p3: 기준점 위면 그 기준점 하나
        assert E.score_target(anchor) == ([1.0 if j == i else 0.0 for j in range(11)], False)


def test_a_score_outside_zero_one_is_refused():
    with pytest.raises(ValueError):
        E.score_target(1.2)
    with pytest.raises(ValueError):
        E.score_target(float("nan"))


# -- 순서 증강: 정답이 선택지와 함께 옮겨진다 ----------------------------------------------------------


def _examples(record, names, seed):
    return E.build_examples(record, names, tokenizer(), order_seed=seed, fmt="chat")


def test_order_augmentation_moves_the_label_with_the_options(fixtures):
    seen = collections.defaultdict(set)
    for record in fixtures:
        names = list(record["request"]["questions"])
        for seed in range(12):
            for example in _examples(record, names, seed):
                label = record["labels"][example["question"]]
                target = example["target"]
                assert sum(target) == pytest.approx(1.0, abs=1e-9) and len(target) == len(example["options"]) == len(example["code_ids"])
                seen[(record["id"], example["question"])].add(tuple(example["options"]))
                # 보이는 자리 j의 정답 몫 = 그 자리에 놓인 선택지의 정답 (어느 순서든)
                if example["type"] == "choice":
                    for key, p in zip(example["options"], target):
                        assert p == pytest.approx(label["probabilities"][key], abs=1e-6)
                elif example["type"] == "noul":
                    for key, p in zip(example["options"], target):
                        assert p == pytest.approx(label["p_yes"] if key == "yes" else 1.0 - label["p_yes"], abs=1e-12)
                else:  # v0c-p3: 잘림 없이 정답 그대로
                    assert sum(p * v for p, v in zip(target, example["values"])) == pytest.approx(label["value"], abs=1e-12)
                    assert example["clamped"] is False
                # 같은 대응을 진단 도구의 읽기(answer_from_codes)로 되읽으면 정답이 나온다
                read = answer_from_codes(example, target)
                if example["type"] == "choice":
                    assert read["probabilities"] == pytest.approx(label["probabilities"], abs=1e-6)
                elif example["type"] == "noul":
                    assert read["p_yes"] == pytest.approx(label["p_yes"], abs=1e-12)
                else:
                    assert read["value"] == pytest.approx(label["value"], abs=1e-12)
    # 순서는 seed마다 실제로 바뀐다 — noul은 둘, score는 두 방향, choice는 여럿(선택지 둘이면 둘)
    for record in fixtures:
        for name, question in record["request"]["questions"].items():
            distinct = len(seen[(record["id"], name)])
            if question["type"] == "choice":
                assert distinct >= min(3, math.factorial(len(question["criteria"])))
            else:
                assert distinct == 2


def test_the_score_example_flags_a_label_the_reading_cannot_reach(fixtures):
    """p2(재현용)에서만 0.05 밖의 정답이 잘린다; p3는 0.01도 그대로 닿는다(기대값 = 0.01)."""
    record = copy.deepcopy(fixtures[0])
    name = next(n for n, q in record["request"]["questions"].items() if q["type"] == "score")
    record["labels"][name]["value"] = 0.01
    (example,) = E.build_examples(record, [name], tokenizer(), order_seed=3, fmt="chat", version="v0c-p2")
    assert example["clamped"] is True and example["version"] == "v0c-p2"
    (example,) = _examples(record, [name], 3)
    assert example["clamped"] is False and example["version"] == "v0c-p3"
    assert sum(p * v for p, v in zip(example["target"], example["values"])) == pytest.approx(0.01, abs=1e-12)
    record["labels"][name]["value"] = 0.5
    (example,) = E.build_examples(record, [name], tokenizer(), order_seed=3, fmt="chat", version="v0c-p2")
    assert example["clamped"] is False


# -- 프롬프트 동일성 (학습 = 평가) ------------------------------------------------------------------------


@pytest.mark.parametrize("version", ["v0c-p2", "v0c-p3"])
def test_training_prompts_are_byte_identical_to_the_diagnosis_prompts_for_the_same_order(fixtures, version):
    for record in fixtures:
        names = list(record["request"]["questions"])
        for seed in (0, 1, 7, 123456789):
            evaluation = {p["question"]: p for p in record_prompts(record, tokenizer(), seed=seed, fmt="chat", mode="direct", orders="seeded", version=version)}
            for example in E.build_examples(record, names, tokenizer(), order_seed=seed, fmt="chat", version=version):
                twin = evaluation[example["question"]]
                assert example["text"].encode("utf-8") == twin["text"].encode("utf-8")
                assert example["tokens"] == twin["tokens"] and example["code_ids"] == twin["code_ids"] and example["options"] == twin["options"]
                assert check_displayed_lines(example) == []


def test_every_counterbalanced_diagnosis_listing_is_reached_by_some_training_order_with_the_same_bytes(fixtures):
    """0번 배치는 진단의 shuffle seed(1)로 바로 닿는다; 나머지 배치는 학습의 무작위 순서가 닿을 때까지 seed를 찾는다
    (선택지가 다섯 이상인 choice는 순열이 너무 많아 찾지 않는다 — 그 경우는 바로 위 검사가 seed 단위로 덮는다)."""
    checked = 0
    for record in fixtures:
        listings = record_prompts(record, tokenizer(), seed=1, fmt="chat", mode="direct", orders="counterbalanced")
        for listing in listings:
            if listing["listing"] == 0:
                seeds = [E.FIXED_ORDER_SEED]
            elif listing["type"] == "choice" and len(listing["options"]) > 4:
                continue
            else:
                seeds = range(400)
            found = None
            for seed in seeds:
                (example,) = _examples(record, [listing["question"]], seed)
                if example["options"] == listing["options"]:
                    found = example
                    break
            assert found is not None, (record["id"], listing["question"], listing["listing"])
            assert found["text"] == listing["text"] and found["tokens"] == listing["tokens"]
            checked += 1
    assert checked >= 60


def test_only_the_request_reaches_the_prompt(fixtures):
    record = copy.deepcopy(fixtures[4])
    secret = "ZZSECRETZZ"
    record["truth"] = {"note": secret}
    record["rationale"] = secret
    record["meta"]["hint_text"] = secret
    for label in record["labels"].values():
        label["note"] = secret
    for example in _examples(record, list(record["request"]["questions"]), 5):
        assert secret not in example["text"]
    plain = copy.deepcopy(fixtures[4])
    assert [e["tokens"] for e in _examples(plain, list(plain["request"]["questions"]), 5)] == [
        e["tokens"] for e in _examples(record, list(record["request"]["questions"]), 5)
    ]


def test_build_examples_refuses_the_reasoning_mode_and_unknown_questions(fixtures):
    with pytest.raises(ValueError):
        E.build_examples(fixtures[0], ["nope"], tokenizer(), order_seed=1, fmt="chat")
    with pytest.raises(ValueError):
        E.build_examples(fixtures[0], list(fixtures[0]["request"]["questions"])[:1], tokenizer(), order_seed=1, fmt="chat", mode="reasoning")


# -- 질문 뽑기 ------------------------------------------------------------------------------------------


def test_question_selection_takes_at_most_q_distinct_questions_spread_over_the_types(fixtures):
    counts = collections.Counter()
    draws = 0
    for record in fixtures:
        questions = record["request"]["questions"]
        types = {q["type"] for q in questions.values()}
        for trial in range(60):
            chosen = E.select_questions(questions, q=2, rng=random.Random(trial))
            assert len(chosen) == min(2, len(questions)) and len(set(chosen)) == len(chosen) and set(chosen) <= set(questions)
            chosen_types = [questions[n]["type"] for n in chosen]
            if len(types) >= 2:
                assert len(set(chosen_types)) == 2  # 종류가 둘 이상이면 서로 다른 종류에서
            if types == {"choice", "noul", "score"}:
                counts.update(chosen_types)
                draws += 1
    for kind in ("choice", "noul", "score"):
        assert counts[kind] / draws == pytest.approx(2 / 3, abs=0.12)
    everything = E.select_questions(fixtures[0]["request"]["questions"], q=99, rng=random.Random(0))
    assert sorted(everything) == sorted(fixtures[0]["request"]["questions"])
    assert E.select_questions(fixtures[0]["request"]["questions"], q=2, rng=random.Random(5)) == E.select_questions(
        fixtures[0]["request"]["questions"], q=2, rng=random.Random(5)
    )
    with pytest.raises(ValueError):
        E.select_questions(fixtures[0]["request"]["questions"], q=0, rng=random.Random(0))


# -- dev 나눔 -------------------------------------------------------------------------------------------


def test_the_dev_split_is_a_record_id_hash_that_keeps_pairs_together():
    records = [{"id": f"r-{i:05d}", "meta": {}} for i in range(20000)]
    for i in range(0, 400, 2):  # 쌍 200개: 짝의 열쇠는 작은 id
        records[i + 1]["meta"]["pair"] = {"of": records[i]["id"], "kind": "numeric"}
    train, dev = E.split_dev(records, fraction=0.02, salt="t")
    assert len(train) + len(dev) == len(records) and not ({r["id"] for r in train} & {r["id"] for r in dev})
    assert 0.015 <= len(dev) / len(records) <= 0.025
    dev_ids = {r["id"] for r in dev}
    for i in range(0, 400, 2):
        assert (records[i]["id"] in dev_ids) == (records[i + 1]["id"] in dev_ids)
    again_train, again_dev = E.split_dev(list(reversed(records)), fraction=0.02, salt="t")
    assert {r["id"] for r in again_dev} == dev_ids  # 순서와 무관 (열쇠의 해시만)
    _, other = E.split_dev(records, fraction=0.02, salt="other salt")
    assert {r["id"] for r in other} != dev_ids
    assert E.is_dev(records[0], fraction=0.0, salt="t") is False and E.is_dev(records[0], fraction=1.0, salt="t") is True
    assert all(E.is_dev(r, fraction=0.02, salt="t") == (r["id"] in dev_ids) for r in records[:500])
    assert shuffle_key(records[1]) == records[0]["id"]


# -- epoch 계획 -----------------------------------------------------------------------------------------


def test_the_epoch_plan_is_deterministic_per_seed_and_epoch_and_takes_q_per_request(fixtures):
    plan = E.epoch_plan(fixtures, seed=3, epoch=0, q=2, augment=True)
    assert plan == E.epoch_plan(fixtures, seed=3, epoch=0, q=2, augment=True)
    assert len(plan) == sum(min(2, len(r["request"]["questions"])) for r in fixtures) == E.examples_per_epoch(fixtures, q=2)
    per_record = collections.Counter(i for i, _, _ in plan)
    assert all(per_record[i] == min(2, len(r["request"]["questions"])) for i, r in enumerate(fixtures))
    assert all(name in fixtures[i]["request"]["questions"] for i, name, _ in plan)
    later = E.epoch_plan(fixtures, seed=3, epoch=1, q=2, augment=True)
    assert later != plan and len(later) == len(plan)
    assert E.epoch_plan(fixtures, seed=4, epoch=0, q=2, augment=True) != plan
    # 증강을 끄면 순서 seed가 진단의 0번 배치(shuffle seed 1)로 고정된다
    fixed = E.epoch_plan(fixtures, seed=3, epoch=0, q=2, augment=False)
    assert {s for _, _, s in fixed} == {E.FIXED_ORDER_SEED} and len({s for _, _, s in plan}) > 1


def test_dev_examples_cover_every_question_and_counterbalanced_listing_with_targets(fixtures):
    dev = E.dev_examples(fixtures[:3], tokenizer(), fmt="chat")
    listings = [p for r in fixtures[:3] for p in record_prompts(r, tokenizer(), seed=1, fmt="chat", orders="counterbalanced")]
    assert [(e["record_id"], e["question"], e["listing"]) for e in dev] == [(p["record_id"], p["question"], p["listing"]) for p in listings]
    assert all(e["tokens"] == p["tokens"] for e, p in zip(dev, listings))
    assert all(sum(e["target"]) == pytest.approx(1.0, abs=1e-9) for e in dev)


def test_length_summary_reports_the_distribution():
    summary = E.length_summary([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
    assert summary["count"] == 10 and summary["mean"] == 55.0 and summary["min"] == 10 and summary["max"] == 100
    assert summary["p50"] == 55.0 and summary["total"] == 550
    assert E.length_summary([])["count"] == 0
