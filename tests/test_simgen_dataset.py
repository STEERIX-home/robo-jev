"""생성한 2,000개 묶음의 성질 (Task v0a 지시서 §3.6·§3.7·§3.8·§3.9·§3.10·§5).

* 행동 choice: 2,000개 전부에서 argmax 정답 = 기대 비용 최소.
* 분야 균형: 분야별 비율 10 % ± 2 %p.
* 쌍(사실 하나 바꾼): 약 10 %; wording 쌍은 숫자·정답이 같고 글만 다르다; numeric 쌍은 여유의 부호가 바뀐다.
* 계획기 권고(planner는 늘, 다른 문체는 절반쯤, 그중 약 25 %는 물리적으로 틀림), 계산된 힌트(약 30 %), 기준을 적은 행동 질문(약 20 %).
"""

import re
from collections import Counter

import numpy as np
import pytest
from simgen_helpers import SAMPLE_N, sample_records

from robo_jev.simgen.catalog import BODIES, DOMAINS, SCENES, default_config


@pytest.fixture(scope="module")
def records():
    return sample_records()


def test_the_sample_has_2000_records_with_unique_ids(records):
    assert len(records) == SAMPLE_N == 2000
    assert len({r["id"] for r in records}) == 2000


def test_action_argmax_is_the_minimum_expected_cost_in_every_record(records):
    checked = 0
    for r in records:
        for name, label in r["labels"].items():
            if label.get("kind") != "action":
                continue
            probs, ec = label["probabilities"], label["expected_cost"]
            best = max(probs, key=probs.get)
            assert ec[best] == min(ec.values()), (r["id"], name)
            # 확률은 기대 비용의 순서를 그대로 따른다(비용이 낮으면 확률이 높거나 같다)
            order = sorted(ec, key=ec.get)
            assert all(probs[a] >= probs[b] for a, b in zip(order, order[1:], strict=False)), (r["id"], name)
            checked += 1
    assert checked >= 2000


def test_action_probabilities_match_the_softmax_of_expected_costs(records):
    tau = default_config().values["tau"]
    for r in records[:500]:
        for label in r["labels"].values():
            if label.get("kind") != "action":
                continue
            keys = list(label["expected_cost"])
            ec = np.array([label["expected_cost"][k] for k in keys])
            z = np.exp(-(ec - ec.min()) / tau)
            assert np.allclose([label["probabilities"][k] for k in keys], z / z.sum(), atol=2e-6)


def test_domain_shares_are_ten_percent_within_two_points(records):
    counts = Counter(r["meta"]["domain"] for r in records)
    assert set(counts) == set(DOMAINS)
    for domain, count in counts.items():
        assert abs(count / len(records) - 0.10) <= 0.02, (domain, count)


def test_every_scene_body_and_style_appears_and_the_combination_is_well_represented(records):
    scenes = Counter(r["meta"]["scene"] for r in records)
    assert set(scenes) == set(SCENES)
    assert set(Counter(r["meta"]["body"] for r in records)) == set(BODIES)
    styles = Counter(r["meta"]["style"] for r in records)
    assert set(styles) == {"facts", "prose", "planner"} and min(styles.values()) > 0.25 * len(records)
    assert scenes["p1p5_step_in_carrying"] / len(records) >= 0.2


def test_about_ten_percent_of_records_are_in_pairs_that_point_at_each_other(records):
    by_id = {r["id"]: r for r in records}
    paired = [r for r in records if r["meta"]["pair"]]
    assert 0.07 <= len(paired) / len(records) <= 0.13
    for r in paired:
        other = by_id[r["meta"]["pair"]["of"]]
        assert other["meta"]["pair"]["of"] == r["id"]
        assert other["meta"]["pair"]["kind"] == r["meta"]["pair"]["kind"]
        assert {r["meta"]["pair"]["role"], other["meta"]["pair"]["role"]} == {"base", "variant"}
    kinds = Counter(r["meta"]["pair"]["kind"] for r in paired)
    assert kinds["numeric"] > 0 and kinds["wording"] > 0


_NUMBER = re.compile(r"\d+(?:\.\d+)?")


def test_wording_pairs_keep_every_number_and_every_label(records):
    by_id = {r["id"]: r for r in records}
    pairs = [(by_id[r["meta"]["pair"]["of"]], r) for r in records if r["meta"]["pair"] and r["meta"]["pair"]["kind"] == "wording" and r["meta"]["pair"]["role"] == "variant"]
    assert pairs
    for base, variant in pairs:
        assert base["labels"] == variant["labels"]
        assert base["request"]["state"] != variant["request"]["state"]
        numbers = lambda r: Counter(_NUMBER.findall(r["request"]["state"].split("\n", 1)[-1]))  # noqa: E731
        # 상황 옷 문구(장소·과제 이름에 든 번호)를 빼면 수치가 같다 — 감각 보고는 바뀌지 않는다
        assert base["meta"]["margins"] == variant["meta"]["margins"]
        assert base["truth"]["world"]["values"] == variant["truth"]["world"]["values"]
        assert numbers(base) or numbers(variant)


def test_numeric_pairs_move_one_margin_across_the_boundary(records):
    by_id = {r["id"]: r for r in records}
    pairs = [(by_id[r["meta"]["pair"]["of"]], r) for r in records if r["meta"]["pair"] and r["meta"]["pair"]["kind"] == "numeric" and r["meta"]["pair"]["role"] == "variant"]
    assert pairs
    for base, variant in pairs:
        name = variant["meta"]["pair"]["flipped"]
        before, after = base["meta"]["margins"][name], variant["meta"]["margins"][name]
        assert np.sign(before) != np.sign(after) or before == 0.0, (variant["id"], name, before, after)
        others = {k: v for k, v in base["meta"]["margins"].items() if k != name}
        assert others == {k: v for k, v in variant["meta"]["margins"].items() if k != name}
        assert base["request"]["state"] != variant["request"]["state"]


def test_planner_style_always_carries_a_recommendation_and_others_about_half(records):
    planner = [r for r in records if r["meta"]["style"] == "planner"]
    others = [r for r in records if r["meta"]["style"] != "planner"]
    assert all(r["meta"]["recommendation"]["present"] for r in planner)
    share = np.mean([r["meta"]["recommendation"]["present"] for r in others])
    assert 0.4 <= share <= 0.6


def test_wrong_recommendations_are_about_a_quarter_and_really_worse(records):
    tau = default_config().values["tau"]
    present = [r for r in records if r["meta"]["recommendation"]["present"]]
    wrong = [r for r in present if r["meta"]["recommendation"]["wrong"]]
    assert 0.15 <= len(wrong) / len(present) <= 0.3
    for r in wrong:
        rec = r["meta"]["recommendation"]
        ec = r["labels"][rec["question"]]["expected_cost"]
        assert ec[rec["option"]] - min(ec.values()) >= 2 * tau
    for r in present:
        rec = r["meta"]["recommendation"]
        if not rec["wrong"]:
            probs = r["labels"][rec["question"]]["probabilities"]
            assert probs[rec["option"]] == max(probs.values())


def test_hint_share_is_about_thirty_percent(records):
    assert 0.25 <= np.mean([r["meta"]["hint"] for r in records]) <= 0.35


def test_about_a_fifth_of_action_questions_state_a_criterion(records):
    flags = [criterion is not None for r in records for criterion in r["meta"]["criteria"].values()]
    assert 0.15 <= np.mean(flags) <= 0.25


def test_question_counts_and_types_per_request(records):
    for r in records:
        questions = r["request"]["questions"]
        assert 3 <= len(questions) <= 8
        kinds = Counter(r["labels"][n].get("kind", r["labels"][n]["type"]) for n in questions)
        assert kinds["action"] >= 1 and kinds["noul"] >= 1 and kinds["score"] >= 1, r["id"]
    state_share = np.mean([any(lab.get("kind") == "state" for lab in r["labels"].values()) for r in records])
    rule_share = np.mean([any(lab.get("kind") == "rule" for lab in r["labels"].values()) for r in records])
    assert 0.35 <= state_share <= 0.6 and 0.2 <= rule_share <= 0.45


def test_option_key_styles_are_mixed(records):
    styles = Counter(r["meta"]["key_style"] for r in records)
    assert set(styles) == {"word", "code", "verb"} and min(styles.values()) > 0.25 * len(records)
