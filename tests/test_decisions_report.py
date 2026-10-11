"""`robo_jev.decisions.report` — 요약 JSON → 사람이 읽는 markdown 표 (Task v0c §2.4).

보고서가 반드시 담아야 하는 것: 미리보기라는 경고(봉인된 시험 세트가 아님), 질문 종류별 지표와 균등 기준선, 현상·질문 종류별 핵심
지표, 묶음 표, 틀린 권고를 따른 비율, 위치 편향, 쌍, bootstrap 구간, (reasoning이면) 생성 통계. 숫자는 요약의 값 그대로다.
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


def _uniformish(record: dict, *, reasoning: bool = False) -> dict:
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
        if reasoning:
            q["reasoning"] = {"text": "think", "stop": "cap", "generated_tokens": 256, "self_answer": False}
        questions[name] = q
    return {"id": record["id"], "questions": questions}


def test_the_markdown_carries_the_preview_warning_and_every_section(fixtures):
    preds = [_uniformish(r) for r in fixtures]
    summary = M.summarize(fixtures, preds, n_boot=50)
    config = {"model": "Qwen/Qwen3.5-2B", "mode": "direct", "format": "chat", "prompt_version": "v0c-p1", "records": 8, "input": "x.jsonl", "input_sha256": "ab" * 32}
    text = R.render_markdown(summary, config=config, title="v0c preview — 2B direct", note=R.PREVIEW_NOTE)
    assert text.startswith("# v0c preview — 2B direct")
    assert "not the sealed test sets" in text and R.PREVIEW_NOTE in text
    for heading in ("## By question type", "## By phenomenon", "## By scene", "## Planner recommendation", "## Position bias", "## Fact-changed pairs", "## Bootstrap"):
        assert heading in text, heading
    # 표의 숫자는 요약 그대로 (소수 셋째 자리)
    assert f"{summary['overall']['noul']['brier']:.3f}" in text
    assert f"{summary['recommendation']['wrong']['follow_rate']:.3f}" in text
    assert "Qwen/Qwen3.5-2B" in text and "v0c-p1" in text


def test_reasoning_runs_get_a_generation_section(fixtures):
    preds = [_uniformish(r, reasoning=True) for r in fixtures]
    summary = M.summarize(fixtures, preds, n_boot=20)
    summary["reasoning"] = R.reasoning_stats(preds)
    assert summary["reasoning"]["cap_rate"] == 1.0 and summary["reasoning"]["generated_tokens_mean"] == 256.0
    text = R.render_markdown(summary, config={"mode": "reasoning"}, title="t")
    assert "## Reasoning generation" in text and "1.000" in text


def test_missing_blocks_render_as_dashes_rather_than_failing(fixtures):
    only_noul = copy.deepcopy(fixtures[:1])
    for record in only_noul:
        record["request"]["questions"] = {k: v for k, v in record["request"]["questions"].items() if v["type"] == "noul"}
        record["labels"] = {k: v for k, v in record["labels"].items() if k in record["request"]["questions"]}
    summary = M.summarize(only_noul, [_uniformish(r) for r in only_noul], n_boot=10)
    text = R.render_markdown(summary, config={}, title="t")
    assert "| — |" in text or "—" in text


def test_a_comparison_renders_differences_with_their_intervals(fixtures):
    a = [_uniformish(r) for r in fixtures]
    cmp = M.compare_runs(fixtures, a, a, n_boot=20)
    text = R.render_comparison(cmp, label_a="2B direct", label_b="2B reasoning", title="cmp")
    assert "2B direct" in text and "2B reasoning" in text and "+0.000 [+0.000, +0.000]" in text
