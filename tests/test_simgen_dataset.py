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
    from robo_jev.simgen.generate import record_moment

    return record_moment(default_config(), record)       # 숫자 쌍의 변형은 받아들인 후보 번호까지(수정 1 N2)


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
        if name in base["meta"]["margins"]:     # 오래된 추적의 나이(age)는 설계 여유가 아니다 — 핵심 정답이 넘는 것으로 본다(아래 시험)
            before, after = base["meta"]["margins"][name], variant["meta"]["margins"][name]
            assert np.sign(before) != np.sign(after) or before == 0.0, (variant["id"], name, before, after)
        others = {k: v for k, v in base["meta"]["margins"].items() if k != name}
        assert others == {k: v for k, v in variant["meta"]["margins"].items() if k != name}
        assert base["request"]["state"] != variant["request"]["state"]


_SKELETON = re.compile(r"\d+(?:[.,]\d+)?")


def _skeleton(text: str) -> str:
    """수를 '#'로, 채움을 말로 보고한 구간(옮긴 사실 자체의 표시 — 영어·한국어)을 '~'로."""
    from robo_jev.simgen.narrate_ko import FILL_KO
    from robo_jev.simgen.spec import FILL_BINS

    for phrase in sorted([b[2] for b in FILL_BINS] + [p for pair in FILL_KO for p in pair], key=len, reverse=True):
        text = text.replace(phrase, "~")
    # 부호 있는 측정(P4 — 옮긴 주 측정 자체)의 방향 말: 기준보다 위/아래, 덜/더 들어감
    for phrase in ("above nominal", "below nominal", "short of fully home", "past the fully-home mark", "높음(", "낮음(", "모자람(", "더 들어감("):
        text = text.replace(phrase, "^" + ("(" if phrase.endswith("(") else ""))
    return _SKELETON.sub("#", text)


def _request_texts(record) -> dict[str, str]:
    out = {"state": record["request"]["state"]}
    for name, q in record["request"]["questions"].items():
        out[name] = q["instructions"] + " | " + repr(q.get("criteria"))
    return out


@pytest.fixture(scope="module")
def forced_pairs():
    """숫자 쌍 단위를 묶음마다 고르게(학습 세트의 쌍과 같은 경로 generate_unit)."""
    from robo_jev.simgen.generate import generate_unit

    cfg = default_config()
    out = []
    for i in range(48):
        unit = {"seed": 6_100_000 + i, "domain": DOMAINS[i % len(DOMAINS)], "group": TRAIN_GROUPS[i % len(TRAIN_GROUPS)], "indices": [2 * i, 2 * i + 1],
                "pair_kind": "numeric"}  # fmt: skip
        out.append(generate_unit(cfg, unit))
    return out


def test_numeric_pairs_change_only_the_flipped_fact_and_cross_the_key(forced_pairs):
    """수정 1 I4: 숫자 쌍은 핵심 질문이 묻는 결정 변수 하나만 옮기고(``meta.pair.target`` = 핵심 질문, ``flipped`` ∈ 그 장면의 쌍 변수),
    핵심 정답이 0.5를 넘으며, 변형은 기준의 계획기 권고를 그대로 보인다(사실 둘이 바뀌지 않게) — 두 요청은 수를 지운 뼈대가 글자 그대로
    같고(질문 이름·글·선택지 설명까지), 다른 수는 몇 개뿐이다(옮긴 값과 그에 딸린 표시값)."""
    from robo_jev.simgen.catalog import pair_targets

    numeric = 0
    for base, variant in forced_pairs:
        bm, vm = base["meta"], variant["meta"]
        assert bm["pair"]["kind"] == vm["pair"]["kind"] and bm["pair"]["of"] == variant["id"] and vm["pair"]["of"] == base["id"]
        if vm["pair"]["kind"] != "numeric":
            assert vm["pair"].get("target") is None
            continue
        numeric += 1
        key = bm["key_question"]
        assert vm["key_question"] == key and vm["pair"]["target"] == key
        assert vm["pair"]["flipped"] in pair_targets(bm["scene"]), (variant["id"], vm["pair"]["flipped"])
        p0, p1 = base["labels"][key]["p_yes"], variant["labels"][key]["p_yes"]
        assert (p0 - 0.5) * (p1 - 0.5) < 0, (variant["id"], p0, p1)
        rb, rv = bm["recommendation"], vm["recommendation"]
        assert (rv["present"], rv["canonical"], rv["option"]) == (rb["present"], rb["canonical"], rb["option"]) and rv.get("kept") is True
        # 질문은 글자 그대로 같다(규칙 질문의 문턱도 기준의 값에 — 변형 자신의 값에 놓으면 문턱이 달라진다)
        assert base["request"]["questions"] == variant["request"]["questions"], variant["id"]
        tb, tv = _request_texts(base), _request_texts(variant)
        assert tb.keys() == tv.keys(), variant["id"]
        changed = 0
        for part in tb:
            assert _skeleton(tb[part]) == _skeleton(tv[part]), (variant["id"], part, tb[part][:300], tv[part][:300])
            changed += sum(a != b for a, b in zip(_SKELETON.findall(tb[part]), _SKELETON.findall(tv[part]), strict=True))
        assert changed <= 8 and tb["state"] != tv["state"], (variant["id"], changed)     # 말로 보고한 채움은 수가 아니라 구간 말이 바뀐다
    assert numeric >= 30, numeric


@pytest.fixture(scope="module")
def skipped_pairs():
    """받아들인 후보가 첫째가 아닌(generator.pair_skip > 0) 숫자 쌍 단위 셋 — 드물다(약 2 %)라서 찾는다(학습 세트와 같은 경로 generate_unit)."""
    from robo_jev.simgen.generate import generate_unit

    cfg = default_config()
    out = []
    for i in range(600):
        unit = {"seed": 6_100_000 + i, "domain": DOMAINS[i % len(DOMAINS)], "group": TRAIN_GROUPS[i % len(TRAIN_GROUPS)], "indices": [2 * i, 2 * i + 1],
                "pair_kind": "numeric"}  # fmt: skip
        if i < 48:
            continue     # forced_pairs와 같은 단위는 건너뛴다
        records = generate_unit(cfg, unit)
        if (records[1]["generator"]["pair_skip"] or 0) > 0:
            out.append(records)
            if len(out) == 3:
                break
    return out


def test_every_record_and_pair_rebuilds_from_its_own_fields(records, forced_pairs, skipped_pairs):
    """수정 1 N2: 숫자 쌍의 변형은 받아들인 후보 번호(generator.pair_skip)를 적는다 — 그래서 학습 세트의 모든 레코드(쌍의 기준·변형, 단어
    쌍, 혼자인 순간)가 그 자신의 칸만으로 글자 그대로 다시 만들어진다(``rebuild_record``). 후보 0이 아닌 변형이 실제로 있고, 그 번호를 빼면
    다른 관측이 나온다(번호가 필요하다)."""
    from robo_jev.simgen.generate import build_moment, rebuild_record, record_moment, split_from

    cfg = default_config()
    pairs = [r for unit in forced_pairs + skipped_pairs for r in unit]
    checked = Counter()
    for r in pairs + [r for r in records[:240]]:
        g, pair = r["generator"], r["meta"]["pair"]
        numeric_variant = bool(pair) and pair["role"] == "variant" and pair["kind"] == "numeric"
        assert (g["pair_skip"] is not None) == numeric_variant, (r["id"], g["pair_skip"], pair)
        assert rebuild_record(cfg, r) == r, r["id"]
        checked["numeric variant" if numeric_variant else ("pair" if pair else "single")] += 1
        if numeric_variant and g["pair_skip"] > 0:
            checked["skip > 0"] += 1
            forced = {k: v for k, v in g["forced"].items() if k != "domain"}
            first = build_moment(cfg, g["seed"], g["forced"]["domain"], forced, "numeric", split_from(g["split"]))
            assert first.obs != record_moment(cfg, r).obs, r["id"]
    assert checked["numeric variant"] >= 30 and checked["skip > 0"] >= 3 and checked["single"] >= 100, checked


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
    present = [r for r in records if r["meta"]["recommendation"]["present"] and not r["meta"]["recommendation"].get("kept")]
    wrong = [r for r in present if r["meta"]["recommendation"]["wrong"]]
    assert 0.18 <= len(wrong) / len(present) <= 0.32
    kept = [r for r in records if r["meta"]["recommendation"]["present"] and r["meta"]["recommendation"].get("kept")]
    for r in present + kept:
        rec = r["meta"]["recommendation"]
        costs = rec["costs"]
        gap = costs[rec["canonical"]] - min(costs.values())
        assert gap == pytest.approx(rec["gap"], abs=1e-3)
        if rec.get("kept"):     # 쌍의 변형: 기준의 권고를 그대로 — 틀림은 이 순간의 비용으로(2τ 이상 나쁘면)
            assert rec["wrong"] == (gap >= 2 * tau - 1e-9), r["id"]
            continue
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
