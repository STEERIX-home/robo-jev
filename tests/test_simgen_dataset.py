"""생성한 2,000개 묶음의 성질 (Task v0a 지시서 §3.6·§3.7·§3.8·§3.9·§3.10·§5).

* 행동 choice: 2,000개 전부에서 argmax 정답 = 기대 비용 최소.
* 분야 균형: 분야별 비율 10 % ± 2 %p.
* 쌍(사실 하나 바꾼): 약 10 %; wording 쌍은 관측의 모든 수치·정답이 같고(상황 옷 문구를 지운 관측이 같고, 글에서 그 문구를 지우면
  수의 목록이 같다 — 리뷰 1 I1) 글만 다르다; numeric 쌍은 여유의 부호가 바뀌고 글이 달라진다.
* 계획기 권고(planner는 늘, 다른 문체는 절반쯤, 그중 약 25 %는 물리적으로 틀림 — 그 순간 가능한 모든 행동으로 잰다, 리뷰 1 I5),
  계산된 힌트(약 30 %), 기준을 적은 행동 질문(약 20 %).
* 정규화한 결정 여유 meta.margin_norm = meta.margins / σ_m(여유 이름·몸) (Task v0c 요청).
"""

import re
from collections import Counter

import numpy as np
import pytest
from simgen_helpers import SAMPLE_N, numeric_view, sample_records, skin_strings

from robo_jev.simgen.catalog import BODIES, DOMAINS, SCENE_GROUP, SCENES, TRAIN_GROUPS, default_config
from robo_jev.simgen.generate import build_moment, split_from


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


def test_every_train_scene_body_and_style_appears_and_groups_are_even(records):
    """학습 표본: 빼 둔 조합(P1+P3)의 장면을 뺀 모든 장면·몸·학습 문체 셋이 나오고, 현상·조합 묶음은 고르다(각 1/9 ± 3점 — Task v0b §5)."""
    scenes = Counter(r["meta"]["scene"] for r in records)
    assert set(scenes) == {s for s in SCENES if SCENE_GROUP[s] in TRAIN_GROUPS}
    assert set(Counter(r["meta"]["body"] for r in records)) == set(BODIES)
    styles = Counter(r["meta"]["style"] for r in records)
    assert set(styles) == {"facts", "prose", "planner"} and min(styles.values()) > 0.25 * len(records)
    groups = Counter(r["meta"]["group"] for r in records)
    assert set(groups) == set(TRAIN_GROUPS)
    for group, count in groups.items():
        assert abs(count / len(records) - 1 / len(TRAIN_GROUPS)) <= 0.03, (group, count)


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


def _moment_of(record):
    g = record["generator"]
    forced = {k: v for k, v in g["forced"].items() if k != "domain"}
    return build_moment(default_config(), g["seed"], g["forced"]["domain"], forced or None, g["variant"], split_from(g["split"]))


def _numbers_without_skin(record, obs) -> Counter:
    text = record["request"]["state"]
    for phrase in skin_strings(obs):
        text = text.replace(phrase, " ")
    return Counter(_NUMBER.findall(text))


def test_wording_pairs_keep_every_number_and_every_label(records):
    by_id = {r["id"]: r for r in records}
    pairs = [(by_id[r["meta"]["pair"]["of"]], r) for r in records if r["meta"]["pair"] and r["meta"]["pair"]["kind"] == "wording" and r["meta"]["pair"]["role"] == "variant"]
    assert pairs
    for base, variant in pairs:
        assert base["labels"] == variant["labels"]
        assert base["request"]["state"] != variant["request"]["state"]
        m_base, m_variant = _moment_of(base), _moment_of(variant)
        # 관측에서 상황 옷 문구를 지우면 같다 — 모든 보고 수치·해상도·나이·단서·단위가 그대로다
        assert numeric_view(m_base.obs) == numeric_view(m_variant.obs), variant["id"]
        assert skin_strings(m_base.obs) != skin_strings(m_variant.obs)      # 상황 옷 문구(장면마다의 물건·장소·가림 문구 포함)가 다르다
        # 글에서도: 상황 옷 문구를 지운 나머지의 수(값과 순서 무관 목록)가 같고, 비어 있지 않다
        a, b = _numbers_without_skin(base, m_base.obs), _numbers_without_skin(variant, m_variant.obs)
        assert a == b and sum(a.values()) >= 3, (variant["id"], a, b)
        assert base["meta"]["margins"] == variant["meta"]["margins"]
        assert base["truth"]["world"]["values"] == variant["truth"]["world"]["values"]


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


def test_wrong_recommendations_are_about_a_quarter_and_judged_over_every_available_action(records):
    """권고는 그 순간 가능한 모든 행동(계획표의 '가능한 동작' 전부)을 기본 비용표로 비교해 정한다: 맞는 권고는 최선에서 τ 안쪽,
    틀린 권고는 2τ 이상 나쁘다(리뷰 1 I5 — 옛 판정은 기본 질문의 일부 선택지와 그 질문의 기준만 봤다)."""
    tau = default_config().values["tau"]
    present = [r for r in records if r["meta"]["recommendation"]["present"]]
    wrong = [r for r in present if r["meta"]["recommendation"]["wrong"]]
    assert 0.18 <= len(wrong) / len(present) <= 0.32
    for r in present:
        rec = r["meta"]["recommendation"]
        costs = rec["costs"]
        gap = costs[rec["canonical"]] - min(costs.values())
        assert gap == pytest.approx(rec["gap"], abs=1e-3)
        assert (gap >= 2 * tau - 1e-6) if rec["wrong"] else (gap <= tau + 1e-6), r["id"]
        assert set(r["meta"]["options"][rec["question"]].values()) <= set(costs)  # 기본 질문의 선택지는 모두 가능한 행동이다
        if rec["option"] is not None:
            assert r["meta"]["options"][rec["question"]][rec["option"]] == rec["canonical"]


def test_normalized_margins_are_the_margins_over_their_design_sigma(records):
    cfg = default_config()
    for r in records:
        meta = r["meta"]
        assert set(meta["margin_norm"]) == set(meta["margins"])
        for name, value in meta["margins"].items():
            sigma = cfg.catalog["margins"][name]["sigma"][meta["body"]]
            assert meta["margin_norm"][name] == pytest.approx(value / sigma, abs=2e-4 / sigma + 1e-4), (r["id"], name)


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
