"""검수용 예시 20개와 묶음 통계 (Task v0a 지시서 §6).

예시: 분야 10개 × 2, P1 단독·P5 단독·조합(6개 이상), 문체 셋 다, numeric 쌍 하나(바뀐 것은 결정 변수 하나), 모두 형식 통과, 마크다운에 모델 입력이
그대로. 통계: 비율의 합, 생성 속도, 핵심 noul 평균, 여유 분포의 설계값 대조.
"""

from collections import Counter

import pytest
from simgen_helpers import sample_records

from robo_jev.simgen.catalog import DOMAINS, default_config
from robo_jev.simgen.examples import PLAN, build_examples, render_markdown
from robo_jev.simgen.record import validate_record
from robo_jev.simgen.stats import dataset_stats


@pytest.fixture(scope="module")
def examples():
    return build_examples(default_config(), git="test")


def test_the_plan_has_twenty_slots_two_per_domain():
    slots = [s.domain for s in PLAN] + [PLAN[-1].domain]  # 마지막 칸은 쌍 — 두 레코드
    assert Counter(slots) == {d: 2 for d in DOMAINS}


def test_examples_cover_domains_phenomena_and_styles(examples):
    assert len(examples) == 20
    assert Counter(r["meta"]["domain"] for r in examples) == {d: 2 for d in DOMAINS}
    phenomena = Counter("+".join(r["meta"]["phenomena"]) for r in examples)
    assert phenomena["P1+P5"] >= 6 and phenomena["P1"] >= 1 and phenomena["P5"] >= 1
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
    # 글에서 바뀐 줄은 결정 변수(끼어든 상대와의 간격)를 담은 줄뿐이다
    changed = [(x, y) for x, y in zip(base["request"]["state"].split("\n"), variant["request"]["state"].split("\n"), strict=True) if x != y]
    assert len(changed) == 1 and "ahead" in changed[0][0]


def test_searched_examples_are_informative(examples):
    """칸마다 고른 레코드는 정답이 한쪽으로 다 쏠리지 않는다. 쌍의 변형 쪽은 고른 것이 아니라 결정 변수를 경계 너머로 옮겨 만든 것이라 뺀다."""
    for r in examples:
        if r["meta"]["pair"] and r["meta"]["pair"]["role"] == "variant":
            continue
        primary = r["labels"][r["meta"]["recommendation"]["question"]]
        assert max(primary["probabilities"].values()) <= 0.97, r["id"]
        assert 0.03 <= r["labels"][r["meta"]["key_question"]]["p_yes"] <= 0.97, r["id"]


def test_markdown_shows_the_model_input_verbatim_and_every_label(examples):
    text = render_markdown(examples, default_config())
    for r in examples:
        assert r["request"]["state"] in text
        for name, q in r["request"]["questions"].items():
            assert q["instructions"] in text and f"`{name}`" in text
        assert r["rationale"] in text
    assert text.count("### Hidden truth") == 20


def test_dataset_stats_report_shares_speed_and_design_checks():
    records = list(sample_records())
    stats = dataset_stats(records, default_config(), seconds=10.0)
    assert stats["records"] == 2000 and stats["generation"]["records_per_second"] == 200.0
    for table in ("scene", "domain", "body", "style"):
        assert sum(v["n"] for v in stats["shares"][table].values()) == 2000
        assert sum(v["share"] for v in stats["shares"][table].values()) == pytest.approx(1.0, abs=1e-3)
    assert 0.3 <= stats["labels"]["key_noul"]["summary"]["mean"] <= 0.7
    for name, margin in stats["margins"].items():
        assert abs(margin["share_abs_below_1.5_sigma"] - margin["design_share_abs_below_1.5_sigma"]) < 0.08, name
    assert set(stats["truth_check"]) >= {"noul_by_label_bin", "noul_brier_vs_truth", "action_label_best_is_cheapest_in_hindsight"}
