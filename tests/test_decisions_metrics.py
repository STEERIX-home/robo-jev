"""`robo_jev.decisions.metrics` — 사후 확률 정답에 대한 지표 (Task v0c §2.3).

손으로 계산한 작은 사례와 맞춘다: soft 이진 로그 손실·KL·Brier, ECE(10구간, 정답 확률 대비), choice 교차 엔트로피,
정답 argmax(= 기대 비용 최소) 일치, 기대 비용 후회, 쌍의 방향 맞음, 단어 쌍의 흔들림, 틀린 권고를 따른 비율, 위치 편향,
여유 구간, 레코드 단위 bootstrap, 같은 레코드의 두 run 비교.

Fix round 1(리뷰 1 I1–I4, 통제자 결정): 모델 − 기준선(언제나 0.5 / 무작위 고르기)의 짝지은 구간, 두 읽기(한 순서 = 배치별 지표의
평균, 순서 평균 = 배치 평균 답의 지표), 같은 배치를 한 번만 두는 병합, 서로 다른 배치 짝만 세는 판정 변경률, 기준 계열별 후회(평균·
중앙값·상위 10 비중·척도 없는 후회), 어려움 축(margin_norm·정답 엔트로피), 묶음별 신호, 설명 뒤 답의 완료율 문턱.
"""

import copy
import math

import pytest
from helpers import FIXTURES

from robo_jev.decisions import metrics as M
from robo_jev.decisions.records import read_records

REQUESTS = FIXTURES / "decisions" / "requests.jsonl"


@pytest.fixture(scope="module")
def fixtures() -> list[dict]:
    return read_records(REQUESTS)


def _prediction(record: dict, answers: dict[str, dict], *, orders: dict[str, list] | None = None, mass: float = 0.9) -> dict:
    """채점 결과와 같은 모양의 예측 (답은 손으로 준다). 보이는 순서는 요청 순서(또는 `orders`)."""
    questions = {}
    for name, question in record["request"]["questions"].items():
        if question["type"] == "choice":
            options = list(question["criteria"])
        elif question["type"] == "noul":
            options = ["yes", "no"]
        else:
            options = ["0.0-0.1", "0.1-0.3", "0.3-0.5", "0.5-0.7", "0.7-0.9", "0.9-1.0"]
        if orders and name in orders:
            options = orders[name]
        questions[name] = {"type": question["type"], "options": options, "answer": answers[name], "code_mass": mass}
    return {"id": record["id"], "questions": questions}


def _oracle(record: dict) -> dict:
    answers = {}
    for name, label in record["labels"].items():
        if label["type"] == "choice":
            answers[name] = {"probabilities": dict(label["probabilities"])}
        elif label["type"] == "noul":
            answers[name] = {"p_yes": label["p_yes"]}
        else:
            answers[name] = {"value": label["value"]}
    return _prediction(record, answers)


def _uniform(record: dict) -> dict:
    answers = {}
    for name, question in record["request"]["questions"].items():
        if question["type"] == "choice":
            answers[name] = {"probabilities": {k: 1.0 / len(question["criteria"]) for k in question["criteria"]}}
        elif question["type"] == "noul":
            answers[name] = {"p_yes": 0.5}
        else:
            answers[name] = {"value": 0.5}
    return _prediction(record, answers)


# -- 손 계산 사례 ---------------------------------------------------------------------------------------------------


def test_soft_binary_log_loss_kl_and_brier_match_the_hand_computation():
    m = M.binary_metrics(p=0.4, y=0.7)
    # −(0.7 ln 0.4 + 0.3 ln 0.6) = 0.7946512, H(0.7) = 0.6108643
    assert m["log_loss"] == pytest.approx(0.7946512, abs=1e-6)
    assert m["kl"] == pytest.approx(0.7946512 - 0.6108643, abs=1e-6)
    assert m["brier"] == pytest.approx(0.09) and m["abs_error"] == pytest.approx(0.3)
    exact = M.binary_metrics(p=0.25, y=0.25)
    assert exact["kl"] == pytest.approx(0.0, abs=1e-12) and exact["brier"] == 0.0
    hard = M.binary_metrics(p=0.0, y=1.0)  # 0 확률은 잘라서 유한한 손실
    assert math.isfinite(hard["log_loss"]) and hard["log_loss"] == pytest.approx(-math.log(M.EPS))


def test_ece_uses_ten_bins_and_compares_with_the_label_probability():
    pairs = [(0.05, 0.0), (0.08, 0.2), (0.55, 0.9), (0.95, 1.0)]
    assert M.ece(pairs) == pytest.approx(0.1175)
    assert M.ece([(1.0, 1.0), (1.0, 0.0)]) == pytest.approx(0.5)  # p = 1은 마지막 구간
    assert M.ece([]) is None


def test_choice_cross_entropy_argmax_match_and_cost_regret_match_the_hand_computation():
    label = {"type": "choice", "kind": "action", "probabilities": {"a": 0.6, "b": 0.3, "c": 0.1}, "expected_cost": {"a": 2.0, "b": 3.0, "c": 10.0}}
    m = M.choice_metrics({"a": 0.2, "b": 0.5, "c": 0.3}, label)
    assert m["ce"] == pytest.approx(1.294004, abs=1e-6) and m["kl"] == pytest.approx(1.294004 - 0.897946, abs=1e-6)
    assert m["predicted"] == "b" and m["best"] == ["a"] and m["match"] is False
    assert m["regret"] == pytest.approx(1.0) and m["sampled_regret"] == pytest.approx(2.9)
    assert m["regret_norm"] == pytest.approx(1.0 / 8.0)  # 후회 1 ÷ 비용 범위 (10 − 2)
    assert m["confidence"] == pytest.approx(0.5) and m["target"] == pytest.approx(0.3)
    assert m["brier"] == pytest.approx(0.16 + 0.04 + 0.04)


def test_ties_in_the_minimum_expected_cost_accept_any_minimiser_with_zero_regret():
    label = {"type": "choice", "kind": "action", "probabilities": {"a": 0.45, "b": 0.45, "c": 0.1}, "expected_cost": {"a": 1.0, "b": 1.0, "c": 5.0}}
    m = M.choice_metrics({"a": 0.1, "b": 0.8, "c": 0.1}, label)
    assert m["best"] == ["a", "b"] and m["match"] is True and m["regret"] == 0.0


def test_without_expected_costs_the_best_option_is_the_label_argmax_and_there_is_no_regret():
    label = {"type": "choice", "kind": "state", "probabilities": {"x": 0.2, "y": 0.8}}
    m = M.choice_metrics({"x": 0.9, "y": 0.1}, label)
    assert m["best"] == ["y"] and m["match"] is False and m["regret"] is None and m["sampled_regret"] is None and m["regret_norm"] is None


def _row(i: int, qtype: str, block: str, m: dict, label: dict) -> dict:
    """aggregate가 읽는 꼴의 행 하나 (손으로 만든 사례용)."""
    return {"record_id": f"r{i}", "question": "q", "type": qtype, "kind": None, "block": block, "w": 1.0, "code_mass": None,
            "m": m, "u": M.uniform_metrics(qtype, label, m), "criterion": "n/a"}


def test_state_choice_ece_is_soft_top_label_calibration():
    label = {"probabilities": {"x": 0.6, "y": 0.4}}
    rows = [
        _row(0, "choice", "choice/state", M.choice_metrics({"x": 0.8, "y": 0.2}, label), label),
        _row(1, "choice", "choice/state", M.choice_metrics({"x": 0.3, "y": 0.7}, label), label),
    ]
    block = M.aggregate(rows)["choice/state"]
    # 확신 0.8 대 표적 0.6 (구간 8), 확신 0.7 대 표적 q[y] = 0.4 (구간 7) → (0.2 + 0.3) / 2
    assert block["ece"] == pytest.approx(0.25) and block["n"] == 2 and block["accuracy"] == pytest.approx(0.5)


# -- 결합과 요약 ----------------------------------------------------------------------------------------------------


def test_an_oracle_scores_zero_kl_full_accuracy_and_zero_regret(fixtures):
    summary = M.summarize(fixtures, [_oracle(r) for r in fixtures], n_boot=50)
    overall = summary["overall"]
    assert overall["noul"]["kl"] == pytest.approx(0.0, abs=1e-9) and overall["noul"]["brier"] == pytest.approx(0.0)
    assert overall["noul"]["ece"] == pytest.approx(0.0, abs=1e-12)
    assert overall["score"]["kl"] == pytest.approx(0.0, abs=1e-9)
    assert overall["choice/action"]["accuracy"] == 1.0 and overall["choice/action"]["regret_mean"] == 0.0
    assert overall["choice/state"]["kl"] == pytest.approx(0.0, abs=1e-9) and overall["choice/rule"]["accuracy"] == 1.0
    assert summary["counts"]["questions"] == sum(len(r["request"]["questions"]) for r in fixtures) == overall["_all"]["n"]


def test_a_uniform_predictor_matches_its_closed_form(fixtures):
    summary = M.summarize(fixtures, [_uniform(r) for r in fixtures], n_boot=50)
    nouls = [r["labels"][n]["p_yes"] for r in fixtures for n in r["labels"] if r["labels"][n]["type"] == "noul"]
    assert summary["overall"]["noul"]["log_loss"] == pytest.approx(math.log(2))
    assert summary["overall"]["noul"]["brier"] == pytest.approx(sum((0.5 - y) ** 2 for y in nouls) / len(nouls))
    actions = [r["labels"][n] for r in fixtures for n in r["labels"] if r["labels"][n].get("kind") == "action"]
    assert summary["overall"]["choice/action"]["ce"] == pytest.approx(sum(math.log(len(l["probabilities"])) for l in actions) / len(actions))
    # 균등 예측의 argmax는 첫 선택지(보이는 순서) — 보고서의 균등 기준선은 argmax 대신 기대값으로 따로 계산한다
    reference = summary["reference"]["uniform"]
    regrets = []
    for l in actions:
        costs = l["expected_cost"]
        regrets.append(sum(costs.values()) / len(costs) - min(costs.values()))
    assert reference["choice/action"]["regret_mean"] == pytest.approx(sum(regrets) / len(regrets))
    assert summary["overall"]["choice/action"]["uniform_regret_mean"] == pytest.approx(sum(regrets) / len(regrets))
    assert reference["choice/action"]["accuracy"] == pytest.approx(sum(sum(1 for c in l["expected_cost"].values() if c == min(l["expected_cost"].values())) / len(l["expected_cost"]) for l in actions) / len(actions))


def test_rows_carry_the_groups_and_the_question_family(fixtures):
    rows = M.question_rows(fixtures, [_oracle(r) for r in fixtures])
    row = next(r for r in rows if r["record_id"] == "fx-005" and r["question"] == "action")
    g = row["groups"]
    assert (g["scene"], g["phenomena"], g["domain"], g["body"], g["style"], g["hint"], g["recommendation"]) == (
        "p1_head_on", "P1", "hospital", "humanoid", "planner", "no_hint", "wrong")
    assert next(r for r in rows if r["record_id"] == "fx-001")["groups"]["phenomena"] == "P1+P5"
    assert row["family"] == "choice/action" and next(r for r in rows if r["question"] == "zone")["family"] == "choice/rule"
    custom = copy.deepcopy(fixtures[3])
    custom["labels"]["conflict"]["family"] = "outcome"
    assert next(r for r in M.question_rows([custom], [_oracle(custom)]) if r["question"] == "conflict")["family"] == "outcome"
    assert {r["groups"]["pair"] for r in rows} == {"none", "numeric", "wording"}


def test_a_question_without_a_prediction_or_label_is_counted_but_not_scored(fixtures):
    record = copy.deepcopy(fixtures[3])
    pred = _oracle(record)
    del pred["questions"]["zone"]
    del record["labels"]["conflict"]
    summary = M.summarize([record], [pred], n_boot=20)
    assert summary["counts"]["questions"] == 4 and summary["counts"]["scored"] == 2
    assert summary["counts"]["missing_prediction"] == 1 and summary["counts"]["missing_label"] == 1


def test_by_phenomenon_and_type_tables_exist(fixtures):
    summary = M.summarize(fixtures, [_oracle(r) for r in fixtures], n_boot=20)
    by = summary["groups"]["phenomena"]
    assert set(by) == {"P1", "P5", "P1+P5"}
    assert by["P1+P5"]["choice/action"]["n"] == 3 and by["P5"]["noul"]["n"] == 2
    assert set(summary["groups"]) >= {"scene", "phenomena", "domain", "body", "style", "hint", "recommendation", "margin_band", "family", "pair", "label_position"}


# -- 여유 구간 ------------------------------------------------------------------------------------------------------


def test_margin_bands_are_tertiles_of_abs_margin_per_margin_name_and_take_the_tightest_margin():
    records = [{"id": f"r{i}", "meta": {"margins": {"gap_m": v}}} for i, v in enumerate([0.1, -0.2, 0.3, 0.9, -1.0, 1.1, 2.0, -2.5, 3.0])]
    records.append({"id": "both", "meta": {"margins": {"gap_m": 2.2, "spill_cm": 0.05}}})
    records.append({"id": "none", "meta": {}})
    records += [{"id": f"s{i}", "meta": {"margins": {"spill_cm": v}}} for i, v in enumerate([1.0, 2.0, 3.0])]
    bands = M.margin_bands(records)
    assert [bands[f"r{i}"] for i in range(9)] == ["small"] * 3 + ["medium"] * 3 + ["large"] * 3
    assert bands["both"] == "small" and bands["none"] == "unknown"
    lonely = M.margin_bands([{"id": "a", "meta": {"margins": {"x": 1.0}}}, {"id": "b", "meta": {"margins": {"x": 2.0}}}])
    assert lonely == {"a": "unknown", "b": "unknown"}  # 이름마다 셋 이상이어야 구간을 나눈다


# -- 권고 ------------------------------------------------------------------------------------------------------------


def test_the_wrong_recommendation_follow_rate_counts_model_argmax_on_the_recommended_option(fixtures):
    preds = [_oracle(r) for r in fixtures]
    fx5 = next(p for p in preds if p["id"] == "fx-005")
    fx5["questions"]["action"]["answer"] = {"probabilities": {"continue": 0.7, "sidestep_right": 0.1, "stop_gentle": 0.1, "stop_moderate": 0.1}}
    summary = M.summarize(fixtures, preds, n_boot=20)
    wrong = summary["recommendation"]["wrong"]
    assert wrong["n"] == 1 and wrong["follow_rate"] == 1.0 and wrong["model_p_recommended"] == pytest.approx(0.7)
    assert wrong["label_p_recommended"] == pytest.approx(next(r for r in fixtures if r["id"] == "fx-005")["labels"]["action"]["probabilities"]["continue"])
    right = summary["recommendation"]["right"]
    # 맞는 권고 넷(fx-001·002·003·008)의 행동 질문; 오라클은 정답 argmax를 고르므로 권고가 최선인 셋에서만 따른다(fx-002는 최선이 stop_gentle)
    assert right["n"] == 4 and right["follow_rate"] == pytest.approx(0.75) and right["label_best_rate"] == pytest.approx(0.75)
    assert summary["groups"]["recommendation"]["wrong"]["choice/action"]["n"] == 1


# -- 위치 편향 -------------------------------------------------------------------------------------------------------


def test_position_bias_counts_where_the_model_argmax_and_the_label_best_sit():
    record = {
        "id": "p", "meta": {}, "request": {"state": "s", "questions": {
            f"q{i}": {"type": "choice", "instructions": "?", "criteria": {"a": "", "b": "", "c": ""}} for i in range(4)}},
        "labels": {f"q{i}": {"type": "choice", "kind": "state", "probabilities": {"a": 0.7, "b": 0.2, "c": 0.1}} for i in range(4)},
    }
    orders = {"q0": ["a", "b", "c"], "q1": ["b", "a", "c"], "q2": ["c", "b", "a"], "q3": ["b", "c", "a"]}
    answers = {f"q{i}": {"probabilities": {orders[f"q{i}"][0]: 0.6, orders[f"q{i}"][1]: 0.3, orders[f"q{i}"][2]: 0.1}} for i in range(4)}
    summary = M.summarize([record], [_prediction(record, answers, orders=orders)], n_boot=10)
    bias = summary["position_bias"]["choice"]["3"]
    assert bias["model_argmax_position_rates"] == [1.0, 0.0, 0.0] and bias["label_best_position_rates"] == [0.25, 0.25, 0.5]
    assert summary["position_bias"]["choice"]["all"]["model_first_rate"] == 1.0 and summary["position_bias"]["choice"]["all"]["label_first_rate"] == 0.25
    assert bias["mean_probability_by_position"] == pytest.approx([0.6, 0.3, 0.1])
    assert bias["accuracy_by_label_position"] == [1.0, 0.0, 0.0]
    assert summary["groups"]["label_position"]["1"]["choice/state"]["accuracy"] == 1.0


def test_noul_and_score_position_bias_compare_listing_orders_against_the_label_means():
    record = {
        "id": "n", "meta": {}, "request": {"state": "s", "questions": {
            "y1": {"type": "noul", "instructions": "?"}, "y2": {"type": "noul", "instructions": "?"},
            "s1": {"type": "score", "instructions": "?", "criteria": []}, "s2": {"type": "score", "instructions": "?", "criteria": []}}},
        "labels": {"y1": {"type": "noul", "p_yes": 0.5}, "y2": {"type": "noul", "p_yes": 0.5},
                   "s1": {"type": "score", "value": 0.4}, "s2": {"type": "score", "value": 0.4}},
    }
    bins = ["0.0-0.1", "0.1-0.3", "0.3-0.5", "0.5-0.7", "0.7-0.9", "0.9-1.0"]
    orders = {"y1": ["yes", "no"], "y2": ["no", "yes"], "s1": bins, "s2": bins[::-1]}
    answers = {"y1": {"p_yes": 0.8}, "y2": {"p_yes": 0.6}, "s1": {"value": 0.3}, "s2": {"value": 0.5}}
    bias = M.summarize([record], [_prediction(record, answers, orders=orders)], n_boot=10)["position_bias"]
    assert bias["noul"]["yes_first"]["model_p_yes"] == pytest.approx(0.8) and bias["noul"]["yes_second"]["model_p_yes"] == pytest.approx(0.6)
    assert bias["noul"]["shift"] == pytest.approx(0.2) and bias["noul"]["label_shift"] == pytest.approx(0.0)
    assert bias["score"]["ascending"]["model_value"] == pytest.approx(0.3) and bias["score"]["descending"]["model_value"] == pytest.approx(0.5)
    assert bias["score"]["shift"] == pytest.approx(-0.2)


# -- 쌍 --------------------------------------------------------------------------------------------------------------


def test_numeric_pairs_check_that_the_model_moves_the_way_the_label_moved(fixtures):
    preds = {r["id"]: _oracle(r) for r in fixtures}
    # 기준 fx-001 → 숫자 쌍 fx-002: 정답은 contact_if_continue 0.86→0.71(−0.15), risk_gentle 0.07→0.01(−0.06), spill_if_hard 그대로
    preds["fx-001"]["questions"]["contact_if_continue"]["answer"] = {"p_yes": 0.60}
    preds["fx-002"]["questions"]["contact_if_continue"]["answer"] = {"p_yes": 0.50}  # 같은 방향
    preds["fx-001"]["questions"]["risk_gentle"]["answer"] = {"value": 0.30}
    preds["fx-002"]["questions"]["risk_gentle"]["answer"] = {"value": 0.35}  # 반대 방향
    pairs = M.summarize(fixtures, list(preds.values()), n_boot=10)["pairs"]["numeric"]
    assert pairs["pairs"] == 1 and pairs["questions"] == 5
    by = pairs["by_type"]
    assert by["noul"]["moved"] == 1 and by["noul"]["direction_correct"] == 1  # spill_if_hard는 움직이지 않아 빠진다
    assert by["score"]["moved"] == 1 and by["score"]["direction_correct"] == 0
    # 행동 choice: 오라클은 정답 그대로 움직여 내적 > 0
    assert by["choice"]["moved"] >= 1 and by["choice"]["direction_correct"] == by["choice"]["moved"]
    assert pairs["direction_correct_rate"] == pytest.approx((1 + 0 + by["choice"]["direction_correct"]) / (2 + by["choice"]["moved"]))


def test_wording_pairs_report_how_much_the_model_shook_when_only_words_changed(fixtures):
    preds = {r["id"]: _oracle(r) for r in fixtures}
    preds["fx-003"]["questions"]["contact_if_continue"]["answer"] = {"p_yes": 0.46}  # 0.86 → 0.46: 0.4 흔들림, 0.5 경계를 넘음
    wording = M.summarize(fixtures, list(preds.values()), n_boot=10)["pairs"]["wording"]
    assert wording["pairs"] == 1 and wording["questions"] == 5
    assert wording["by_type"]["noul"]["model_shift_mean"] == pytest.approx(0.2)  # (0.4 + 0) / 2
    assert wording["by_type"]["noul"]["flip_rate"] == pytest.approx(0.5)
    assert wording["label_shift_mean"] == pytest.approx(0.0)
    assert wording["by_type"]["choice"]["model_shift_mean"] == pytest.approx(0.0)


# -- bootstrap과 비교 ------------------------------------------------------------------------------------------------


def test_record_bootstrap_is_reproducible_and_brackets_the_point_estimate(fixtures):
    preds = [_uniform(r) for r in fixtures]
    a = M.summarize(fixtures, preds, n_boot=200, seed=3)["bootstrap"]
    b = M.summarize(fixtures, preds, n_boot=200, seed=3)["bootstrap"]
    assert a == b
    ll = a["noul.log_loss"]
    assert ll["low"] == pytest.approx(math.log(2)) and ll["high"] == pytest.approx(math.log(2))  # 상수 지표
    brier = a["noul.brier"]
    assert brier["low"] <= brier["value"] <= brier["high"] and brier["low"] < brier["high"]
    # 균등 예측 − 균등 기준선 = 0 (같은 뽑기라 구간도 0)
    assert brier["vs_uniform"] == {"value": pytest.approx(0.0, abs=1e-15), "low": pytest.approx(0.0, abs=1e-15), "high": pytest.approx(0.0, abs=1e-15)}


def test_comparing_a_run_with_itself_gives_zero_differences_in_both_readings(fixtures):
    preds = [_uniform(r) for r in fixtures]
    cmp = M.compare_runs(fixtures, preds, preds, n_boot=100, seed=1)
    assert set(cmp["readings"]) == set(M.READINGS) and cmp["records"] == len(fixtures) and cmp["gate"]["passed"] is True
    for reading in M.READINGS:
        for key, row in cmp["readings"][reading]["differences"].items():
            assert row["value"] == pytest.approx(0.0, abs=1e-12) and row["low"] == pytest.approx(0.0, abs=1e-12) and row["high"] == pytest.approx(0.0, abs=1e-12), key


def test_comparing_oracle_with_uniform_shows_the_oracle_ahead(fixtures):
    cmp = M.compare_runs(fixtures, [_oracle(r) for r in fixtures], [_uniform(r) for r in fixtures], n_boot=200, seed=1)
    diffs = cmp["readings"]["single_order"]["differences"]
    assert diffs["noul.log_loss"]["value"] < 0  # a − b: 오라클이 손실이 작다
    assert diffs["choice/action.accuracy"]["value"] > 0


def test_code_mass_is_averaged_by_question_type(fixtures):
    preds = [_prediction(r, {n: q["answer"] for n, q in _oracle(r)["questions"].items()}, mass=0.8) for r in fixtures]
    summary = M.summarize(fixtures, preds, n_boot=10)
    assert summary["overall"]["noul"]["code_mass"] == pytest.approx(0.8) and summary["overall"]["_all"]["code_mass"] == pytest.approx(0.8)


# -- 선택지 순서 교란 (T6 꼴의 판정 변경률) ---------------------------------------------------------------------------


def _reordered(prediction: dict, record: dict, answers: dict[str, dict]) -> dict:
    """같은 질문들을 보이는 순서만 뒤집어 다시 채점한 것 (배치 하나)."""
    out = copy.deepcopy(prediction)
    for name, q in out["questions"].items():
        q["options"] = list(reversed(q["options"]))
        if name in answers:
            q["answer"] = answers[name]
    return out


def test_permutation_sensitivity_counts_only_pairs_of_listings_whose_order_differs(fixtures):
    """리뷰 1 I4: 같은 보이는 순서의 짝은 판정이 늘 같으므로 세지 않는다(세면 비율이 절반쯤으로 희석된다)."""
    record = fixtures[3]  # fx-004: choice(행동) · noul · score · choice(규칙)
    a = _oracle(record)
    b = _reordered(a, record, {"decision": {"probabilities": {"proceed": 0.7, "slow_down": 0.1, "stop_moderate": 0.1, "stop_hard": 0.1}},
                                "conflict": {"p_yes": 0.61}})  # 0.41 → 0.61: 0.2 움직이고 0.5 경계를 넘는다
    c = copy.deepcopy(a)  # a와 같은 배치 — 병합에서 한 번만 남는다
    merged, stats = M.merge_runs([[a], [b], [c]])
    assert stats["duplicate_listings_skipped"] == {"choice": 2, "noul": 1, "score": 1}
    sens = M.permutation_sensitivity(merged)
    assert sens["noul"] == {"questions": 1, "pairs": 1, "shift_mean": pytest.approx(0.2), "flip_rate": 1.0, "identical_pairs_skipped": 0}
    label = record["labels"]["decision"]["probabilities"]
    tv = 0.5 * sum(abs(label[k] - v) for k, v in {"proceed": 0.7, "slow_down": 0.1, "stop_moderate": 0.1, "stop_hard": 0.1}.items())
    assert sens["choice"]["questions"] == 2 and sens["choice"]["shift_mean"] == pytest.approx(tv / 2) and sens["choice"]["flip_rate"] == 0.5
    assert sens["score"]["shift_mean"] == 0.0 and sens["_all"]["questions"] == 4
    # 병합하지 않은 채 같은 배치를 두 번 가진 예측: 그 짝은 건너뛰고 센다
    doubled = copy.deepcopy(merged)
    for q in doubled[0]["questions"].values():
        q["listings"].append(copy.deepcopy(q["listings"][0]))
    again = M.permutation_sensitivity(doubled)
    assert again["noul"]["pairs"] == 2 and again["noul"]["identical_pairs_skipped"] == 1
    assert M.permutation_sensitivity([a]) == {"_all": {"questions": 0, "pairs": 0, "shift_mean": None, "flip_rate": None, "identical_pairs_skipped": 0}}


def test_side_agreement_counts_the_same_side_of_one_half_and_skips_ambiguous_labels():
    rows = [_row(i, "noul", "noul", M.binary_metrics(p, y), {"p_yes": y}) for i, (p, y) in enumerate([(0.8, 0.9), (0.3, 0.7), (0.6, 0.55), (0.2, 0.1)])]
    block = M.aggregate(rows)["noul"]
    # (0.6, 0.55)은 |y − 0.5| < 0.1이라 빠진다; 나머지 셋 중 둘이 같은 쪽
    assert block["side_n"] == 3 and block["side_agreement"] == pytest.approx(2 / 3)


def _orders_record() -> dict:
    return {
        "id": "o", "meta": {}, "request": {"state": "s", "questions": {
            "y": {"type": "noul", "instructions": "?"}, "s": {"type": "score", "instructions": "?", "criteria": []},
            "c": {"type": "choice", "instructions": "?", "criteria": {"a": "", "b": "", "c": ""}}}},
        "labels": {"y": {"type": "noul", "p_yes": 0.5}, "s": {"type": "score", "value": 0.4},
                   "c": {"type": "choice", "kind": "state", "probabilities": {"a": 0.4, "b": 0.3, "c": 0.3}}},
    }


def _orders_runs():
    record = _orders_record()
    bins = ["0.0-0.1", "0.1-0.3", "0.3-0.5", "0.5-0.7", "0.7-0.9", "0.9-1.0"]
    one = _prediction(record, {"y": {"p_yes": 0.7}, "s": {"value": 0.3}, "c": {"probabilities": {"a": 0.6, "b": 0.3, "c": 0.1}}},
                      orders={"y": ["yes", "no"], "s": bins, "c": ["a", "b", "c"]})
    two = _prediction(record, {"y": {"p_yes": 0.3}, "s": {"value": 0.5}, "c": {"probabilities": {"a": 0.2, "b": 0.3, "c": 0.5}}},
                      orders={"y": ["no", "yes"], "s": bins[::-1], "c": ["c", "b", "a"]})
    return record, [[one], [two], [copy.deepcopy(one)]]  # 셋째는 첫째와 같은 배치(같은 답)


def test_merging_runs_keeps_each_distinct_listing_once_and_averages_them_equally():
    record, runs = _orders_runs()
    (merged,), stats = M.merge_runs(runs)
    q = merged["questions"]
    assert stats == {"runs": 3, "duplicate_listings_skipped": {"choice": 1, "noul": 1, "score": 1}}
    assert q["y"]["n_listings"] == 2 and q["y"]["answer"]["p_yes"] == pytest.approx(0.5)  # (0.7 + 0.3) / 2 — 같은 배치는 한 번만
    assert q["s"]["answer"]["value"] == pytest.approx(0.4)
    assert q["c"]["answer"]["probabilities"] == pytest.approx({"a": 0.4, "b": 0.3, "c": 0.3})
    assert M.summarize([record], [merged], reading="order_averaged", n_boot=10)["overall"]["noul"]["n"] == 1


def test_paired_order_effects_compare_the_same_question_under_different_listings():
    _, runs = _orders_runs()
    merged, _ = M.merge_runs(runs)
    effects = M.order_effects(merged)
    assert effects["noul"] == {"questions": 1, "mean_shift": pytest.approx(0.4), "positive_rate": 1.0}
    assert effects["score"]["mean_shift"] == pytest.approx(-0.2)
    # choice: 키별 배치 평균(a .4·b .3·c .3)에서 벗어난 정도를 자리별로 — 배치 (a,b,c)와 (c,b,a)
    assert effects["choice"]["3"]["mean_deviation_by_position"] == pytest.approx([0.2, 0.0, -0.2])


def test_the_two_readings_differ_as_defined():
    """한 순서 읽기 = 배치별 지표의 평균, 순서 평균 읽기 = 배치 평균 답의 지표 (통제자 결정)."""
    record = {
        "id": "r", "meta": {}, "request": {"state": "s", "questions": {
            "y": {"type": "noul", "instructions": "?"}, "c": {"type": "choice", "instructions": "?", "criteria": {"a": "", "b": ""}}}},
        "labels": {"y": {"type": "noul", "p_yes": 0.8}, "c": {"type": "choice", "kind": "state", "probabilities": {"a": 0.9, "b": 0.1}}},
    }
    first = _prediction(record, {"y": {"p_yes": 0.9}, "c": {"probabilities": {"a": 0.7, "b": 0.3}}}, orders={"y": ["yes", "no"], "c": ["a", "b"]})
    second = _prediction(record, {"y": {"p_yes": 0.5}, "c": {"probabilities": {"a": 0.4, "b": 0.6}}}, orders={"y": ["no", "yes"], "c": ["b", "a"]})
    merged, _ = M.merge_runs([[first], [second]])
    single = M.summarize([record], merged, reading="single_order", n_boot=10)["overall"]
    averaged = M.summarize([record], merged, reading="order_averaged", n_boot=10)["overall"]
    assert single["noul"]["brier"] == pytest.approx((0.01 + 0.09) / 2) and averaged["noul"]["brier"] == pytest.approx(0.01)  # p = 0.7
    assert single["choice/state"]["accuracy"] == pytest.approx(0.5) and averaged["choice/state"]["accuracy"] == 1.0  # {a .55, b .45}
    assert single["noul"]["n"] == averaged["noul"]["n"] == 1 and single["noul"]["rows"] == 2 and averaged["noul"]["rows"] == 1


# -- Fix round 1: 기준선과 짝지은 구간 (리뷰 1 I1) ---------------------------------------------------------------------


def _noul_records(pairs: list[tuple[float, float]]) -> tuple[list[dict], list[dict]]:
    records, preds = [], []
    for i, (p, y) in enumerate(pairs):
        record = {"id": f"r{i}", "meta": {}, "request": {"state": "s", "questions": {"y": {"type": "noul", "instructions": "?"}}},
                  "labels": {"y": {"type": "noul", "p_yes": y}}}
        records.append(record)
        preds.append(_prediction(record, {"y": {"p_yes": p}}))
    return records, preds


def test_the_paired_difference_against_always_half_matches_the_hand_computation():
    # r0: (p .7, y .9) → Brier .04 대 기준선 .16 (차 −.12); r1: (p .6, y .2) → .16 대 .09 (차 +.07)
    records, preds = _noul_records([(0.7, 0.9), (0.6, 0.2)])
    boot = M.summarize(records, preds, n_boot=4000, seed=0)["bootstrap"]["noul.brier"]
    assert boot["value"] == pytest.approx(0.10) and boot["uniform"] == pytest.approx(0.125)
    assert boot["vs_uniform"]["value"] == pytest.approx(-0.025)
    # 레코드 둘의 bootstrap: (r0, r0) 확률 1/4 → −.12, (r1, r1) 1/4 → +.07 — 2.5·97.5 백분위가 그 끝값이다
    assert boot["vs_uniform"]["low"] == pytest.approx(-0.12) and boot["vs_uniform"]["high"] == pytest.approx(0.07)


def test_oracle_and_uniform_differences_against_the_baseline_have_the_right_sign_and_size(fixtures):
    oracle = M.summarize(fixtures, [_oracle(r) for r in fixtures], n_boot=300)
    nouls = [r["labels"][n]["p_yes"] for r in fixtures for n in r["labels"] if r["labels"][n]["type"] == "noul"]
    brier = oracle["bootstrap"]["noul.brier"]["vs_uniform"]
    assert brier["value"] == pytest.approx(-sum((0.5 - y) ** 2 for y in nouls) / len(nouls)) and brier["high"] < 0
    assert oracle["bootstrap"]["choice/action.accuracy"]["vs_uniform"]["low"] > 0
    assert oracle["overall"]["noul"]["brier_vs_uniform"] == pytest.approx(brier["value"])


# -- Fix round 1: 기준 계열별 후회 (리뷰 1 I3) --------------------------------------------------------------------------


def _criterion_record() -> tuple[dict, dict]:
    qs, labels, basis, answers = {}, {}, {}, {}
    spec = {  # 이름: (기준, 기대 비용, 모델이 고르는 키)
        "default_q": (None, {"a": 1.0, "b": 3.0, "c": 5.0}, "b"),        # 후회 2, 범위 4 → .5; 무작위 2 (.5)
        "contact_q": ("avoid_contact", {"a": 0.0, "b": 1000.0}, "b"),     # 후회 1000, 척도 없는 1; 무작위 500 (.5)
        "time_q": ("time_critical", {"a": 2.0, "b": 2.5}, "a"),           # 후회 0
    }
    for name, (criterion, costs, pick) in spec.items():
        qs[name] = {"type": "choice", "instructions": "?", "criteria": {k: "" for k in costs}}
        labels[name] = {"type": "choice", "kind": "action", "probabilities": {k: 1.0 / len(costs) for k in costs}, "expected_cost": costs}
        basis[name] = {"kind": "action", "criterion": criterion}
        answers[name] = {"probabilities": {k: (0.9 if k == pick else 0.1 / (len(costs) - 1)) for k in costs}}
    record = {"id": "c", "meta": {"basis": basis}, "request": {"state": "s", "questions": qs}, "labels": labels}
    return record, _prediction(record, answers)


def test_regret_is_reported_per_criterion_family_with_a_scale_free_version():
    record, pred = _criterion_record()
    summary = M.summarize([record], [pred], n_boot=10)
    table = summary["regret_by_criterion"]
    assert set(table) == {"default", "avoid_contact", "time_critical", "stated", "all"}
    assert table["default"]["regret_mean"] == 2.0 and table["default"]["regret_norm_mean"] == 0.5 and table["default"]["uniform_regret_mean"] == 2.0
    assert table["avoid_contact"]["regret_mean"] == 1000.0 and table["avoid_contact"]["regret_norm_mean"] == 1.0
    stated = table["stated"]
    assert (stated["regret_n"], stated["regret_mean"], stated["regret_median"], stated["regret_max"]) == (2, 500.0, 500.0, 1000.0)
    assert stated["regret_positive_rate"] == 0.5 and stated["regret_top10_share"] == 1.0
    assert table["all"]["regret_mean"] == pytest.approx(1002.0 / 3) and table["all"]["regret_norm_mean"] == pytest.approx(1.5 / 3)
    boot = summary["bootstrap"]
    assert boot["choice/action.regret_mean[default]"]["value"] == 2.0 and boot["choice/action.regret_mean[stated]"]["value"] == 500.0
    assert boot["choice/action.regret_norm[stated]"]["value"] == 0.5 and boot["choice/action.regret_norm[stated]"]["uniform"] == pytest.approx(0.5)
    rows = M.question_rows([record], [pred])
    assert {r["question"]: r["groups"]["criterion"] for r in rows} == {"default_q": "default", "contact_q": "avoid_contact", "time_q": "time_critical"}


def test_the_top_ten_share_is_the_share_of_the_ten_largest_question_regrets():
    qs, labels, answers = {}, {}, {}
    for i in range(12):  # 후회 1, 2, …, 12 (질문마다 하나)
        costs = {"a": 0.0, "b": float(i + 1)}
        qs[f"q{i}"] = {"type": "choice", "instructions": "?", "criteria": {"a": "", "b": ""}}
        labels[f"q{i}"] = {"type": "choice", "kind": "action", "probabilities": {"a": 0.5, "b": 0.5}, "expected_cost": costs}
        answers[f"q{i}"] = {"probabilities": {"a": 0.1, "b": 0.9}}
    record = {"id": "t", "meta": {}, "request": {"state": "s", "questions": qs}, "labels": labels}
    block = M.summarize([record], [_prediction(record, answers)], n_boot=10)["overall"]["choice/action"]
    assert block["regret_top10_share"] == pytest.approx(sum(range(3, 13)) / sum(range(1, 13)))
    assert block["regret_median"] == pytest.approx(6.5) and block["regret_max"] == 12.0


# -- Fix round 1: 어려움 축 (통제자 결정) ---------------------------------------------------------------------------------


def test_margin_norm_bands_take_the_tightest_normalized_margin():
    assert M.margin_norm_band({"margin_norm": {"step_in": -0.3, "spill": 2.0}}) == "near"
    assert M.margin_norm_band({"margin_norm": 0.7}) == "mid" and M.margin_norm_band({"margin_norm": {"x": -1.6}}) == "far"
    assert M.margin_norm_band({}) == "absent" and M.margin_norm_band({"margin_norm": {"x": float("nan")}}) == "absent"


def test_label_entropy_bands_put_labels_near_one_half_in_the_high_band(fixtures):
    assert M.normalized_entropy("noul", {"p_yes": 0.5}) == pytest.approx(1.0) and M.normalized_entropy("noul", {"p_yes": 1.0}) == 0.0
    assert M.normalized_entropy("choice", {"probabilities": {"a": 0.5, "b": 0.5}}) == pytest.approx(1.0)
    assert M.entropy_band(0.1) == "low" and M.entropy_band(0.5) == "mid" and M.entropy_band(0.9) == "high" and M.entropy_band(None) == "absent"
    record = copy.deepcopy(fixtures[0])
    record["meta"]["key_question"] = "contact_if_continue"  # p_yes .86 → H ≈ 0.58 비트 → mid
    record["meta"]["margin_norm"] = {"stop_margin_m": 0.2, "spill_margin_cm": 3.0}
    rows = {r["question"]: r["groups"] for r in M.question_rows([record], [_oracle(record)])}
    assert rows["action"]["key_entropy_band"] == "mid" and rows["action"]["margin_norm_band"] == "near"
    # spill_if_hard .18 → H ≈ 0.68 비트(high), risk_gentle .07 → 0.37(mid), person_next (.52 .39 .09) → 0.84(high)
    assert rows["spill_if_hard"]["label_entropy_band"] == "high" and rows["risk_gentle"]["label_entropy_band"] == "mid"
    assert rows["person_next"]["label_entropy_band"] == "high"
    low = {r["question"]: r["groups"] for r in M.question_rows([fixtures[1]], [_oracle(fixtures[1])])}
    assert low["risk_gentle"]["label_entropy_band"] == "low"  # .01 → 0.08 비트
    plain = {r["question"]: r["groups"] for r in M.question_rows([fixtures[0]], [_oracle(fixtures[0])])}
    assert plain["action"]["key_entropy_band"] == "absent" and plain["action"]["margin_norm_band"] == "absent"


# -- Fix round 1: 묶음별 신호 (리뷰 1 I2) ---------------------------------------------------------------------------------


def test_group_signal_reports_correlation_and_paired_difference_per_scene(fixtures):
    signal = M.summarize(fixtures, [_oracle(r) for r in fixtures], n_boot=300)["signal"]
    cafe = signal["scene"]["p1p5_step_in_carrying"]["noul"]
    assert cafe["n"] == 6 and cafe["pearson"] == pytest.approx(1.0)  # 오라클: p = y
    labels = [0.86, 0.18, 0.71, 0.18, 0.86, 0.18]
    assert cafe["brier"]["vs_uniform"]["value"] == pytest.approx(-sum((0.5 - y) ** 2 for y in labels) / 6) and cafe["brier"]["vs_uniform"]["high"] < 0
    assert signal["scene"]["p1_crossing"]["choice/action"]["accuracy"]["vs_uniform"]["value"] == pytest.approx(0.75)  # 오라클 1 − 무작위 1/4
    assert set(signal) == set(M.SIGNAL_DIMENSIONS)


# -- Fix round 1: 설명 뒤 답의 완료율 문턱 -------------------------------------------------------------------------------


def _with_reasoning(prediction: dict, completes: list[bool]) -> dict:
    out = copy.deepcopy(prediction)
    flags = iter(completes)
    for q in out["questions"].values():
        q["reasoning"] = {"text": "x", "stop": "eos", "generated_tokens": 10, "self_answer": True, "complete": next(flags, True)}
    return out


def test_the_completion_gate_marks_explanation_runs_below_ninety_percent_as_inconclusive(fixtures):
    record = fixtures[7]  # 질문 넷
    good = M.reasoning_stats([_with_reasoning(_oracle(record), [True, True, True, True])])
    bad = M.reasoning_stats([_with_reasoning(_oracle(record), [True, False, True, True])])
    assert good["complete_rate"] == 1.0 and good["gate"] == {"threshold": 0.9, "passed": True}
    assert bad["complete_rate"] == 0.75 and bad["gate"]["passed"] is False
    legacy = copy.deepcopy(_oracle(record))  # v0c-p1 기록(완료 칸 없음): 상한에서 잘린 것은 미완료
    for q in legacy["questions"].values():
        q["reasoning"] = {"text": "x", "stop": "cap", "generated_tokens": 256, "self_answer": False}
    assert M.reasoning_stats([legacy])["complete_rate"] == 0.0
    cmp = M.compare_runs([record], [_with_reasoning(_oracle(record), [True, False, True, True])], [_oracle(record)], n_boot=20)
    assert cmp["gate"]["passed"] is False and cmp["gate"]["a"]["complete_rate"] == 0.75 and cmp["gate"]["b"] is None


def test_a_choice_label_without_kind_but_with_expected_costs_is_read_as_an_action_choice(fixtures):
    record = copy.deepcopy(fixtures[0])
    del record["labels"]["action"]["kind"]
    del record["labels"]["person_next"]["kind"]
    rows = M.question_rows([record], [_oracle(record)])
    blocks = {r["question"]: r["block"] for r in rows}
    assert blocks["action"] == "choice/action" and blocks["person_next"] == "choice/other"


# -- v0a 실제 형식의 메타 (basis·recommendation.question·key_style·agent·distractors·pair.role) ------------------------


def _v0a_style(record: dict) -> dict:
    out = copy.deepcopy(record)
    out["meta"].update({
        "key_style": "code", "agent": {"kind": "child", "group": "person"}, "distractors": 2, "units": {"distance": "cm", "speed": "m/s"},
        "basis": {name: ({"kind": "action", "criterion": None} if q["type"] == "choice" and out["labels"][name].get("kind") == "action"
                         else {"kind": "noul", "source": "event", "event": "contact"} if q["type"] == "noul"
                         else {"kind": "score", "source": "quantity", "name": "delay"} if q["type"] == "score"
                         else {"kind": "state", "state": "intent"} if out["labels"][name].get("kind") == "state"
                         else {"kind": "rule", "classes": {}})
                  for name, q in out["request"]["questions"].items()},
    })
    return out


def test_question_family_comes_from_the_engine_basis_when_it_is_there(fixtures):
    record = _v0a_style(fixtures[3])  # fx-004: action · noul · score · rule
    record["meta"]["basis"]["decision"]["criterion"] = "time_critical"
    families = {r["question"]: r["family"] for r in M.question_rows([record], [_oracle(record)])}
    assert families == {"decision": "action:time_critical", "conflict": "noul:contact", "risk_proceed": "score:delay", "zone": "rule"}
    plain = _v0a_style(fixtures[0])
    assert {r["question"]: r["family"] for r in M.question_rows([plain], [_oracle(plain)])}["person_next"] == "state:intent"
    assert next(r for r in M.question_rows([plain], [_oracle(plain)]) if r["question"] == "action")["family"] == "action:default"


def test_new_record_groupings_read_key_style_agent_distractors_and_units(fixtures):
    record = _v0a_style(fixtures[0])
    groups = M.question_rows([record], [_oracle(record)])[0]["groups"]
    assert (groups["key_style"], groups["agent"], groups["distractors"], groups["distance_unit"]) == ("code", "child", "2", "cm")
    bare = M.question_rows([fixtures[3]], [_oracle(fixtures[3])])[0]["groups"]
    assert (bare["key_style"], bare["agent"], bare["distractors"], bare["distance_unit"]) == ("unknown", "none", "unknown", "m")
    summary = M.summarize([record], [_oracle(record)], n_boot=10)
    assert {"key_style", "agent", "distractors", "distance_unit"} <= set(summary["groups"])


def test_a_recommendation_that_names_its_question_is_matched_to_that_question_only(fixtures):
    record = copy.deepcopy(fixtures[0])
    # 같은 키를 가진 두 번째 행동 질문 — 권고는 첫 질문만 겨눈다
    record["request"]["questions"]["backup_action"] = copy.deepcopy(record["request"]["questions"]["action"])
    record["labels"]["backup_action"] = copy.deepcopy(record["labels"]["action"])
    pred = _oracle(record)
    loose = M.summarize([record], [pred], n_boot=10)["recommendation"]["right"]["n"]
    record["meta"]["recommendation"]["question"] = "action"
    strict = M.summarize([record], [pred], n_boot=10)["recommendation"]["right"]["n"]
    assert (loose, strict) == (2, 1)


def test_pair_direction_follows_the_engine_roles(fixtures):
    preds = {r["id"]: _oracle(r) for r in fixtures}
    records = copy.deepcopy(fixtures)
    by_id = {r["id"]: r for r in records}
    # fx-001이 짝 정보를 들고 base, fx-002가 variant (엔진처럼 둘 다 짝을 가리킨다); 파일 순서상 fx-001이 먼저
    by_id["fx-001"]["meta"]["pair"] = {"of": "fx-002", "kind": "numeric", "role": "base"}
    by_id["fx-002"]["meta"]["pair"] = {"of": "fx-001", "kind": "numeric", "role": "variant"}
    preds["fx-001"]["questions"]["contact_if_continue"]["answer"] = {"p_yes": 0.60}
    preds["fx-002"]["questions"]["contact_if_continue"]["answer"] = {"p_yes": 0.50}
    numeric = M.summarize(records, list(preds.values()), n_boot=10)["pairs"]["numeric"]
    assert numeric["pairs"] == 1 and numeric["by_type"]["noul"]["direction_correct"] == 1
    assert M.pair_deltas(records, M.question_rows(records, list(preds.values())))["numeric"][0]["base"] == "fx-001"
