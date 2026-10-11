"""검수용 예시 30개와 묶음 통계 (Task v0b 지시서 §7; v0a 지시서 §6의 모양).

예시: 분야 10개 × 3, 새 현상 P3·P4·P6과 새 조합 위주(현상마다 4개 이상 — 조합은 두 현상에 센다), 한국어 4개 이상, '다시 볼까' 질문 3개 이상,
문체 셋 다, numeric 쌍 하나(바뀐 것은 결정 변수 하나), 모두 형식 통과, 마크다운에 모델 입력이 그대로. 칸은 K = 2048로 다시 계산한 정답으로
고른다(리뷰 1 m8). 통계: 비율의 합, 생성 속도, 핵심 noul 평균, 여유 분포의 설계값 대조, 권고 판정.
"""

from collections import Counter

import pytest
from simgen_helpers import sample_records

from robo_jev.simgen.catalog import DOMAINS, default_config
from robo_jev.simgen.examples import PLAN, SELECTION_K, build_examples, render_markdown
from robo_jev.simgen.generate import assemble, build_moment
from robo_jev.simgen.record import validate_record
from robo_jev.simgen.stats import dataset_stats


@pytest.fixture(scope="module")
def examples():
    return build_examples(default_config(), git="test")


def test_the_plan_has_thirty_slots_three_per_domain():
    slots = [s.domain for s in PLAN] + [s.domain for s in PLAN if s.pair]   # 쌍의 칸은 두 레코드
    assert len(slots) == 30
    assert Counter(slots) == {d: 3 for d in DOMAINS}


def test_examples_cover_domains_phenomena_languages_and_look_again(examples):
    assert len(examples) == 30
    assert Counter(r["meta"]["domain"] for r in examples) == {d: 3 for d in DOMAINS}
    phenomena = Counter(p for r in examples for p in r["meta"]["phenomena"])
    assert all(phenomena[p] >= 4 for p in ("P1", "P3", "P4", "P5", "P6")), phenomena
    new = sum(any(p in ("P3", "P4", "P6") for p in r["meta"]["phenomena"]) for r in examples)
    assert new >= 24
    assert sum(r["meta"]["language"] == "ko" for r in examples) >= 4
    assert sum(any(b.get("kind") == "voi" for b in r["meta"]["basis"].values()) for r in examples) >= 3
    assert set(Counter(r["meta"]["style"] for r in examples)) == {"facts", "prose", "planner"}
    assert all(validate_record(r) == [] for r in examples)


def test_examples_contain_exactly_one_numeric_pair_that_flips_the_decision(examples):
    paired = [r for r in examples if r["meta"]["pair"]]
    assert len(paired) == 2
    base, variant = sorted(paired, key=lambda r: r["meta"]["pair"]["role"])
    assert base["meta"]["pair"]["of"] == variant["id"] and variant["meta"]["pair"]["of"] == base["id"]
    assert variant["meta"]["pair"]["kind"] == "numeric"
    a, b = (r["labels"][r["meta"]["recommendation"]["question"]]["probabilities"] for r in (base, variant))
    assert max(a, key=a.get) != max(b, key=b.get)
    # 글에서 바뀐 줄은 결정 변수(끼어든 상대와의 간격)를 담은 줄과, 정답이 바뀌었으니 계획기 권고 줄뿐이다
    changed = [(x, y) for x, y in zip(base["request"]["state"].split("\n"), variant["request"]["state"].split("\n"), strict=True) if x != y]
    assert 1 <= len(changed) <= 2 and "ahead" in changed[0][0]
    assert all("ahead" in x or "planner" in x.lower() or "recommendation" in x.lower() for x, _ in changed)


def test_searched_examples_are_informative_at_the_selection_k(examples):
    """칸마다 고른 레코드는 정답이 한쪽으로 다 쏠리지 않는다 — 고르는 판단은 K = 2048로 다시 계산한 정답으로 했다(보여 주는 K = 256
    정답의 잡음으로 고르지 않는다). 쌍의 변형 쪽은 고른 것이 아니라 결정 변수를 경계 너머로 옮겨 만든 것이라 뺀다."""
    cfg = default_config()
    for r in examples:
        if r["meta"]["pair"] and r["meta"]["pair"]["role"] == "variant":
            continue
        g = r["generator"]
        forced = {k: v for k, v in g["forced"].items() if k != "domain"}
        selected = assemble(cfg, build_moment(cfg, g["seed"], g["forced"]["domain"], forced), 0, forced=forced, k=SELECTION_K)
        assert selected["request"]["questions"] == r["request"]["questions"]  # 같은 순간·같은 질문(권고 줄만 K에 따라 다를 수 있다)
        primary = selected["labels"][selected["meta"]["recommendation"]["question"]]
        assert max(primary["probabilities"].values()) <= 0.97, r["id"]
        assert max(primary["expected_cost"].values()) - min(primary["expected_cost"].values()) >= 1.0, r["id"]
        assert 0.03 <= selected["labels"][selected["meta"]["key_question"]]["p_yes"] <= 0.97, r["id"]


def test_markdown_shows_the_model_input_verbatim_and_every_label(examples):
    text = render_markdown(examples, default_config())
    for r in examples:
        assert r["request"]["state"] in text
        for name, q in r["request"]["questions"].items():
            assert q["instructions"] in text and f"`{name}`" in text
        assert r["rationale"] in text
    assert text.count("### True world (a posterior draw) and what happened") == 30


def test_dataset_stats_report_shares_speed_and_design_checks():
    records = list(sample_records())
    stats = dataset_stats(records, default_config(), seconds=10.0)
    assert stats["records"] == 2000 and stats["generation"]["records_per_second"] == 200.0
    for table in ("scene", "domain", "body", "style"):
        assert sum(v["n"] for v in stats["shares"][table].values()) == 2000
        assert sum(v["share"] for v in stats["shares"][table].values()) == pytest.approx(1.0, abs=1e-3)
    assert 0.3 <= stats["labels"]["key_noul"]["summary"]["mean"] <= 0.7
    for name, margin in stats["margins"].items():
        # 표본 잡음만큼 허용(v0b는 여유 이름이 12개라 이름마다 레코드가 70~800개): |관측 − 설계| < 4·SE + 0.03
        design = margin["design_share_abs_below_1.5_sigma"]
        se = (design * (1.0 - design) / margin["n"]) ** 0.5
        assert abs(margin["share_abs_below_1.5_sigma"] - design) < 4.0 * se + 0.03, (name, margin["n"])
    assert set(stats["truth_check"]) >= {"noul_by_label_bin", "noul_brier_vs_truth", "action_label_best_is_cheapest_in_hindsight"}
    rec = stats["flags"]["recommendation"]
    assert rec["correct_within_tau"] == 1.0 and rec["wrong_at_least_2tau"] == 1.0 and 0.18 <= rec["wrong_share"] <= 0.32
