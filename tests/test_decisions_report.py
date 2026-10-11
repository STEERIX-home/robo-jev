"""`robo_jev.decisions.report` — 요약 JSON → 사람이 읽는 markdown 표 (Task v0c §2.4; fix round 1).

보고서가 반드시 담아야 하는 것: 미리보기라는 경고(봉인된 시험 세트가 아님), 두 읽기(한 순서·순서 평균)의 질문 종류별 지표와 기준선
대비 짝지은 구간, 기준 계열별 후회, 장면별 신호, 어려움 축, 묶음 표, 권고(범위를 밝혀), 위치 편향·짝지은 순서 효과·서로 다른 배치
사이의 판정 변경률, 쌍, bootstrap 구간, (설명 뒤 답이면) 생성 통계와 완료율 문턱 — 미달이면 "판정 불가". 숫자는 요약의 값 그대로다.
"""

import copy

import pytest
from helpers import FIXTURES

from robo_jev.decisions import metrics as M
from robo_jev.decisions import report as R
from robo_jev.decisions.records import read_records

REQUESTS = FIXTURES / "decisions" / "requests.jsonl"


@pytest.fixture(scope="module")
def fixtures() -> list[dict]:
    return read_records(REQUESTS)


def _uniformish(record: dict, *, complete: bool | None = None) -> dict:
    questions = {}
    for name, question in record["request"]["questions"].items():
        if question["type"] == "choice":
            keys = list(question["criteria"])
            probs = {k: (0.6 if i == 0 else 0.4 / (len(keys) - 1)) for i, k in enumerate(keys)}
            answer, options = {"probabilities": probs}, keys
        elif question["type"] == "noul":
            answer, options = {"p_yes": 0.7}, ["yes", "no"]
        else:
            answer, options = {"value": 0.4}, ["0.0-0.1", "0.1-0.3", "0.3-0.5", "0.5-0.7", "0.7-0.9", "0.9-1.0"]
        q = {"type": question["type"], "options": options, "answer": answer, "code_mass": 0.75}
        if complete is not None:
            q["reasoning"] = {"text": "x", "stop": "eos" if complete else "cap", "generated_tokens": 150, "self_answer": complete, "complete": complete}
        questions[name] = q
    return {"id": record["id"], "questions": questions}


def test_the_markdown_carries_the_preview_warning_and_every_section(fixtures):
    preds = [_uniformish(r) for r in fixtures]
    summary = M.summarize_all(fixtures, preds, n_boot=50)
    config = {"model": "Qwen/Qwen3.5-2B", "mode": "direct", "format": "chat", "prompt_version": "v0c-p2", "orders": "counterbalanced", "input": "x.jsonl", "input_sha256": "ab" * 32}
    text = R.render_markdown(summary, config=config, title="v0c preview — 2B direct", note=R.PREVIEW_NOTE)
    assert text.startswith("# v0c preview — 2B direct")
    assert "not the sealed test sets" in text and R.PREVIEW_NOTE in text
    for heading in ("## By question type (both readings)", "## Action regret by criterion family", "## Signal by scene — single order",
                    "## Signal by scene — order-averaged", "## Difficulty axes", "## By phenomenon — single order", "## Planner recommendation",
                    "## Position bias", "### Paired order effects", "### Verdict changes between listings", "## Fact-changed pairs",
                    "## Bootstrap (record-level, 95 %) — single order", "## Bootstrap (record-level, 95 %) — order-averaged"):
        assert heading in text, heading
    single = summary["readings"]["single_order"]
    # 표의 숫자는 요약 그대로 (소수 셋째 자리)
    assert f"{single['overall']['noul']['brier']:.3f}" in text
    assert f"{single['recommendation']['wrong']['follow_rate']:.3f}" in text
    vs = single["bootstrap"]["noul.brier"]["vs_uniform"]
    assert f"{vs['value']:+.3f} [{vs['low']:+.3f}, {vs['high']:+.3f}]" in text
    assert "Qwen/Qwen3.5-2B" in text and "v0c-p2" in text and "INCONCLUSIVE" not in text


def test_explanation_runs_get_a_generation_section_and_an_inconclusive_banner_below_the_gate(fixtures):
    failing = M.summarize_all(fixtures, [_uniformish(r, complete=False) for r in fixtures], n_boot=20)
    text = R.render_markdown(failing, config={"mode": "reasoning"}, title="t")
    assert "## Explanation-first generation" in text and "INCONCLUSIVE — explanation-first completion rate 0.000 < 0.90" in text
    passing = M.summarize_all(fixtures, [_uniformish(r, complete=True) for r in fixtures], n_boot=20)
    text = R.render_markdown(passing, config={"mode": "reasoning"}, title="t")
    assert "INCONCLUSIVE" not in text and "the reasoning comparison can be read" in text


def test_missing_blocks_render_as_dashes_rather_than_failing(fixtures):
    only_noul = copy.deepcopy(fixtures[:1])
    for record in only_noul:
        record["request"]["questions"] = {k: v for k, v in record["request"]["questions"].items() if v["type"] == "noul"}
        record["labels"] = {k: v for k, v in record["labels"].items() if k in record["request"]["questions"]}
    summary = M.summarize_all(only_noul, [_uniformish(r) for r in only_noul], n_boot=10)
    text = R.render_markdown(summary, config={}, title="t")
    assert "—" in text and "**choice/action**" not in text


def test_a_comparison_renders_both_readings_with_their_intervals_and_the_gate(fixtures):
    a = [_uniformish(r) for r in fixtures]
    cmp = M.compare_runs(fixtures, a, a, n_boot=20)
    text = R.render_comparison(cmp, label_a="2B direct", label_b="2B reasoning", title="cmp")
    assert "2B direct" in text and "2B reasoning" in text and "+0.000 [+0.000, +0.000]" in text
    assert "## single order (deployment)" in text and "## order-averaged (what it knows)" in text and "INCONCLUSIVE" not in text
    gated = M.compare_runs(fixtures, [_uniformish(r, complete=False) for r in fixtures], a, n_boot=20)
    assert "INCONCLUSIVE" in R.render_comparison(gated, label_a="2B reasoning", label_b="2B direct", title="cmp")
