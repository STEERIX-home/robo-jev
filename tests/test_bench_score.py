"""bench-v0 점수 (Task bench0 §3·§5) — 가짜 예측으로 지표가 기존 도구(진단 도구 ``--rescore``, ``convert_single_probe.py compare``,
v0 보고의 T6 스크립트)와 같은지 본다."""

import copy
import functools
import importlib.util
import json
import sys

import pytest
from helpers import FIXTURES, REPO

from robo_jev.bench.score import (
    fake_api_predictions,
    pilot_report,
    pooled_brier,
    probe_summary,
    readme_table,
    render_pilot,
    set_metrics,
    t6_invariance,
)
from robo_jev.decisions.single_probe import compare_probe

REQUESTS = FIXTURES / "decisions" / "requests.jsonl"


def _records() -> list[dict]:
    return [json.loads(line) for line in REQUESTS.read_text(encoding="utf-8").splitlines() if line.strip()]


@functools.lru_cache(maxsize=1)
def _diagnose():
    spec = importlib.util.spec_from_file_location("diagnose_decisions_for_bench", REPO / "scripts" / "diagnose_decisions.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _with_criteria(records: list[dict]) -> list[dict]:
    """행동 질문(기대 비용이 붙은 choice)에 기준 계열을 단다 — 첫 레코드의 것은 적힌 기준(time_critical), 나머지는 기본."""
    out = copy.deepcopy(records)
    for index, record in enumerate(out):
        actions = [n for n, label in record["labels"].items() if isinstance(label.get("expected_cost"), dict)]
        record["meta"]["criteria"] = {name: ("time_critical" if index == 0 else None) for name in actions}
    return out


def test_physics_metrics_equal_the_diagnosis_tools_rescore(tmp_path):
    records = _with_criteria(_records())
    source = tmp_path / "records.jsonl"
    source.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
    predictions = fake_api_predictions(records, "vendor/model", strength=0.6, seed=3)
    pred_path = tmp_path / "fake.predictions.jsonl"
    pred_path.write_text("".join(json.dumps(p) + "\n" for p in predictions), encoding="utf-8")
    out = tmp_path / "rescored.json"
    assert _diagnose().main(["--rescore", str(pred_path), "--input", str(source), "--out", str(out), "--n-boot", "50"]) == 0
    tool = json.loads(out.read_text(encoding="utf-8"))["summary"]["readings"]
    ours = set_metrics("T1", records, predictions, n_boot=50)["readings"]
    for reading in ("single_order", "order_averaged"):
        overall = tool[reading]["overall"]
        mine = ours[reading]
        assert mine["noul_brier"] == pytest.approx(overall["noul"]["brier"]) and mine["score_brier"] == pytest.approx(overall["score"]["brier"])
        assert mine["noul_brier_uniform"] == pytest.approx(overall["noul"]["uniform"]["brier"])
        assert mine["action_accuracy"] == pytest.approx(overall["choice/action"]["accuracy"])
        assert mine["state_accuracy"] == pytest.approx((overall.get("choice/state") or {}).get("accuracy")) if overall.get("choice/state") else True
        assert mine["rule_accuracy"] == pytest.approx(overall["choice/rule"]["accuracy"])
        assert mine["regret_default"] == pytest.approx(tool[reading]["regret_by_criterion"]["default"]["regret_mean"])
        assert mine["regret_stated"] == pytest.approx(tool[reading]["regret_by_criterion"]["stated"]["regret_mean"])
        assert mine["follow_right"] == pytest.approx(tool[reading]["recommendation"]["right"]["follow_rate"])
        assert mine["kl_all"] == pytest.approx(overall["_all"]["kl"])
        assert mine["direction_correct"] == pytest.approx(tool[reading]["pairs"]["numeric"]["direction_correct_rate"])
        assert mine["coverage"] == 1.0
    # API 예측은 질문마다 listing 하나 — 두 읽기가 같다
    assert ours["single_order"]["noul_brier"] == pytest.approx(ours["order_averaged"]["noul_brier"])


def test_missing_answers_lower_the_coverage_not_the_denominator():
    records = _records()
    predictions = fake_api_predictions(records, "m", strength=0.9, seed=1)
    del predictions[0]["questions"][next(iter(predictions[0]["questions"]))]
    predictions = predictions[1:] + predictions[:1]
    metrics = set_metrics("T2", records, predictions[:-2], n_boot=20)["readings"]["single_order"]
    total = sum(len(r["request"]["questions"]) for r in records)
    assert metrics["questions"] == total and metrics["scored"] < total and metrics["coverage"] == pytest.approx(metrics["scored"] / total)


def test_general_reading_accuracy_equals_the_probe_compare_tool():
    records = _records()
    a = fake_api_predictions(records, "a", strength=0.7, seed=1)
    b = fake_api_predictions(records, "b", strength=0.2, seed=2)
    mine = probe_summary(records, a, reading="single_order")
    for record in records:  # probe 도구는 분야·묶음 meta를 읽는다
        record.setdefault("meta", {}).setdefault("questions", {})
    tool = compare_probe(records, a, b, n_boot=20)["readings"]["single_order"]["table"]["_all"]
    assert mine["accuracy"] == pytest.approx(tool["accuracy"]["a"]) and mine["brier"] == pytest.approx(tool["brier"]["a"])
    assert mine["coverage"] == 1.0 and set(mine["by_domain"]) == {str(r["meta"].get("domain")) for r in records}


def _t6_family():
    base = {"id": "b0", "meta": {"t6": {"base": None, "variant": "base"}}, "request": {"state": "s", "questions": {
        "q": {"type": "noul", "instructions": "?"}, "c": {"type": "choice", "instructions": "?", "criteria": {"x": "x", "y": "y"}}}},
        "labels": {"q": {"type": "noul", "p_yes": 0.2}, "c": {"type": "choice", "kind": "state", "probabilities": {"x": 1.0, "y": 0.0}}}}
    same = copy.deepcopy(base)
    same.update({"id": "v1", "meta": {"t6": {"base": "b0", "variant": "state_paraphrase", "map": {"q": "q", "c": "c"}}}})
    flipped = copy.deepcopy(base)
    flipped.update({"id": "v2", "meta": {"t6": {"base": "b0", "variant": "key_change", "map": {"q": "q", "c": None}}}})
    flipped["labels"]["q"]["p_yes"] = 0.8
    return [base, same, flipped]


def _pred(rid: str, p: float, x: float) -> dict:
    return {"id": rid, "questions": {"q": {"type": "noul", "listings": [{"answer": {"p_yes": p}}, {"answer": {"p_yes": p + 0.02}}]},
                                     "c": {"type": "choice", "listings": [{"answer": {"probabilities": {"x": x, "y": 1 - x}}}]}}}


def test_t6_invariance_follows_the_v0_report_definition():
    records = _t6_family()
    predictions = [_pred("b0", 0.30, 0.9), _pred("v1", 0.50, 0.6), _pred("v2", 0.60, 0.9)]
    out = t6_invariance(records, predictions)
    paraphrase, key = out["kinds"]["state_paraphrase"], out["kinds"]["key_change"]
    assert paraphrase["n"] == 2 and paraphrase["label_change"] == pytest.approx(0.0)
    assert paraphrase["shift_mean"] == pytest.approx((0.2 + 0.3) / 2)  # noul |0.51 − 0.31|, choice 전변동 ½(|0.6−0.9| + |0.4−0.1|)
    assert key["n"] == 1 and key["label_change"] == pytest.approx(0.6) and key["excess_mean"] == pytest.approx(abs(0.3 - 0.6))
    assert out["invariant"]["n"] == 2 and out["invariant_kinds"] == ["state_paraphrase"]  # key_change는 불변 합에서 뺀다
    assert t6_invariance(records, predictions[:1])["invariant"]["n"] == 0  # 변형의 답이 없으면 세지 않는다


def test_pooled_brier_and_the_readme_table():
    def physics(noul_b, noul_n, score_b, score_n):
        return {"readings": {"single_order": {"noul_n": noul_n, "noul_brier": noul_b, "noul_brier_uniform": 0.2,
                                              "score_n": score_n, "score_brier": score_b, "score_brier_uniform": 0.25}}}

    per_set = {"T2": physics(0.1, 10, 0.05, 10), "T3": physics(0.2, 10, 0.05, 10), "T4": physics(0.1, 20, 0.15, 0),
               "G1": {"readings": {"single_order": {"accuracy": 0.5}}}, "G2": {"readings": {"single_order": {"accuracy": 0.7}}}}
    pooled = pooled_brier(per_set)
    assert pooled["questions"] == 60
    assert pooled["brier"] == pytest.approx((1.0 + 0.5 + 2.0 + 0.5 + 2.0) / 60)
    assert pooled["skill"] == pytest.approx(1 - pooled["brier"] / ((0.2 * 40 + 0.25 * 20) / 60))
    assert pooled_brier({"T2": per_set["T2"]}) is None  # 세트 하나라도 없으면 없다
    results = {"models": [{"label": "ours", "role": "ours", "sets": per_set, "price_per_m_input": None},
                          {"label": "vendor/model", "role": "api", "sets": per_set, "price_per_m_input": 0.042}]}
    table = readme_table(results, date="2026-10-12")
    assert "| **ours** |" in table and "| vendor/model |" in table and "0.042" in table and "self-hosted" in table
    assert "2026-10-12" in table and "home ground" in table and "stored option order" in table


def test_pilot_report_compares_the_two_score_formats():
    records = _records()
    eleven = fake_api_predictions(records, "m", strength=0.8, seed=1)
    two = fake_api_predictions(records, "m", strength=0.3, seed=2)
    for p in eleven + two:
        p["api"]["usage"] = {"input_tokens": 100, "cost": 4.2e-6}
    estimates = {levels: {str(r["id"]): {"low": 80, "high": 200} for r in records} for levels in (2, 11)}
    report = pilot_report(records, eleven, two, estimates=estimates, bench_tokens={"low": 1_000_000, "high": 2_000_000}, price={"prompt": 4.2e-8}, n_boot=50)
    assert report["variants"]["score11"]["brier"] < report["variants"]["score2"]["brier"]
    assert report["difference_11_minus_2"]["value"] < 0
    assert report["variants"]["score11"]["actual_over_high"] == pytest.approx(0.5)
    assert report["projected_bench_cost_usd"] == pytest.approx(0.5 * 2_000_000 * 4.2e-8)
    text = render_pilot({"records": len(records), "models": {"m": report}})
    assert "| m |" in text and "Δ 11 − 2" in text
