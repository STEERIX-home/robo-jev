"""`robo_jev.decisions.metrics` — 사후 확률 정답에 대한 지표 (Task v0c §2.3).

손으로 계산한 작은 사례와 맞춘다: soft 이진 로그 손실·KL·Brier, ECE(10구간, 정답 확률 대비), choice 교차 엔트로피,
정답 argmax(= 기대 비용 최소) 일치, 기대 비용 후회, 쌍의 방향 맞음, 단어 쌍의 흔들림, 틀린 권고를 따른 비율, 위치 편향,
여유 구간, 레코드 단위 bootstrap, 같은 레코드의 두 run 비교.
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
    assert m["regret"] == pytest.approx(1.0) and m["expected_regret"] == pytest.approx(2.9)
    assert m["confidence"] == pytest.approx(0.5) and m["target"] == pytest.approx(0.3)
    assert m["brier"] == pytest.approx(0.16 + 0.04 + 0.04)


def test_ties_in_the_minimum_expected_cost_accept_any_minimiser_with_zero_regret():
    label = {"type": "choice", "kind": "action", "probabilities": {"a": 0.45, "b": 0.45, "c": 0.1}, "expected_cost": {"a": 1.0, "b": 1.0, "c": 5.0}}
    m = M.choice_metrics({"a": 0.1, "b": 0.8, "c": 0.1}, label)
    assert m["best"] == ["a", "b"] and m["match"] is True and m["regret"] == 0.0


def test_without_expected_costs_the_best_option_is_the_label_argmax_and_there_is_no_regret():
    label = {"type": "choice", "kind": "state", "probabilities": {"x": 0.2, "y": 0.8}}
    m = M.choice_metrics({"x": 0.9, "y": 0.1}, label)
    assert m["best"] == ["y"] and m["match"] is False and m["regret"] is None and m["expected_regret"] is None


def test_state_choice_ece_is_soft_top_label_calibration():
    rows = [
        {"type": "choice", "kind": "state", "m": M.choice_metrics({"x": 0.8, "y": 0.2}, {"probabilities": {"x": 0.6, "y": 0.4}})},
        {"type": "choice", "kind": "state", "m": M.choice_metrics({"x": 0.3, "y": 0.7}, {"probabilities": {"x": 0.6, "y": 0.4}})},
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
    assert reference["choice/action"]["expected_regret_mean"] == pytest.approx(sum(regrets) / len(regrets))
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
    assert bias["model_argmax_positions"] == [4, 0, 0] and bias["label_best_positions"] == [1, 1, 2]
    assert bias["model_first_rate"] == 1.0 and bias["label_first_rate"] == 0.25
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


def test_comparing_a_run_with_itself_gives_zero_differences(fixtures):
    preds = [_uniform(r) for r in fixtures]
    cmp = M.compare_runs(fixtures, preds, preds, n_boot=100, seed=1)
    for key, row in cmp["differences"].items():
        assert row["value"] == pytest.approx(0.0, abs=1e-12) and row["low"] == pytest.approx(0.0, abs=1e-12) and row["high"] == pytest.approx(0.0, abs=1e-12), key
    assert cmp["records"] == len(fixtures)


def test_comparing_oracle_with_uniform_shows_the_oracle_ahead(fixtures):
    cmp = M.compare_runs(fixtures, [_oracle(r) for r in fixtures], [_uniform(r) for r in fixtures], n_boot=200, seed=1)
    assert cmp["differences"]["noul.log_loss"]["value"] < 0  # a − b: 오라클이 손실이 작다
    assert cmp["differences"]["choice/action.accuracy"]["value"] > 0


def test_code_mass_is_averaged_by_question_type(fixtures):
    preds = [_prediction(r, {n: q["answer"] for n, q in _oracle(r)["questions"].items()}, mass=0.8) for r in fixtures]
    summary = M.summarize(fixtures, preds, n_boot=10)
    assert summary["overall"]["noul"]["code_mass"] == pytest.approx(0.8) and summary["overall"]["_all"]["code_mass"] == pytest.approx(0.8)


# -- 선택지 순서 교란 (T6 꼴의 판정 변경률) ---------------------------------------------------------------------------


def test_permutation_sensitivity_measures_how_answers_move_when_only_the_option_order_changes(fixtures):
    record = fixtures[3]  # fx-004: choice(행동) · noul · score · choice(규칙)
    a = _oracle(record)
    b = copy.deepcopy(a)
    b["questions"]["decision"]["answer"] = {"probabilities": {"proceed": 0.7, "slow_down": 0.1, "stop_moderate": 0.1, "stop_hard": 0.1}}
    b["questions"]["conflict"]["answer"] = {"p_yes": 0.61}  # 0.41 → 0.61: 0.2 움직이고 0.5 경계를 넘는다
    c = copy.deepcopy(a)
    sens = M.permutation_sensitivity([[a], [b], [c]])
    # 쌍 셋(a-b, a-c, b-c): noul 이동 0.2, 0, 0.2 → 평균 0.4/3, 뒤집힘 2/3
    assert sens["noul"]["questions"] == 1 and sens["noul"]["pairs"] == 3
    assert sens["noul"]["shift_mean"] == pytest.approx(0.4 / 3) and sens["noul"]["flip_rate"] == pytest.approx(2 / 3)
    label = record["labels"]["decision"]["probabilities"]
    tv = 0.5 * sum(abs(label[k] - v) for k, v in {"proceed": 0.7, "slow_down": 0.1, "stop_moderate": 0.1, "stop_hard": 0.1}.items())
    assert sens["choice"]["questions"] == 2 and sens["choice"]["shift_mean"] == pytest.approx(2 * tv / 6)
    assert sens["choice"]["flip_rate"] == pytest.approx(2 / 6)  # 행동 질문의 두 쌍에서 argmax가 proceed로 바뀐다
    assert sens["score"]["shift_mean"] == 0.0 and sens["_all"]["questions"] == 4
    assert M.permutation_sensitivity([[a]]) == {"_all": {"questions": 0, "pairs": 0, "shift_mean": None, "flip_rate": None}}


def test_side_agreement_counts_the_same_side_of_one_half_and_skips_ambiguous_labels():
    rows = [{"type": "noul", "kind": None, "block": "noul", "m": M.binary_metrics(p, y)} for p, y in [(0.8, 0.9), (0.3, 0.7), (0.6, 0.55), (0.2, 0.1)]]
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
    three = _prediction(record, {"y": {"p_yes": 0.9}, "s": {"value": 0.3}, "c": {"probabilities": {"a": 0.6, "b": 0.3, "c": 0.1}}},
                        orders={"y": ["yes", "no"], "s": bins, "c": ["a", "b", "c"]})
    return record, [[one], [two], [three]]


def test_order_averaging_weighs_each_distinct_order_equally():
    record, runs = _orders_runs()
    (avg,) = M.average_over_orders(runs)
    q = avg["questions"]
    # noul: Yes 먼저 (0.7, 0.9) → 0.8, Yes 나중 0.3 → (0.8 + 0.3) / 2
    assert q["y"]["answer"]["p_yes"] == pytest.approx(0.55) and q["y"]["orders_seen"] == 2
    assert q["s"]["answer"]["value"] == pytest.approx(0.4)
    assert q["c"]["answer"]["probabilities"] == pytest.approx({"a": 0.4, "b": 0.3, "c": 0.3})
    assert M.summarize([record], [avg], n_boot=10)["overall"]["noul"]["n"] == 1


def test_paired_order_effects_compare_the_same_question_under_different_orders():
    _, runs = _orders_runs()
    effects = M.order_effects(runs)
    assert effects["noul"]["questions"] == 1 and effects["noul"]["mean_shift"] == pytest.approx(0.5) and effects["noul"]["positive_rate"] == 1.0
    assert effects["score"]["mean_shift"] == pytest.approx(-0.2)
    # choice: 키별 평균(a .4·b .3·c .3)에서 벗어난 정도를 자리별로 — 순서 (a,b,c)·(c,b,a)·(a,b,c)의 세 run
    dev = effects["choice"]["3"]["mean_deviation_by_position"]
    assert dev[0] > 0 > dev[2] and dev[1] == pytest.approx(0.0, abs=1e-12)


def test_a_choice_label_without_kind_but_with_expected_costs_is_read_as_an_action_choice(fixtures):
    record = copy.deepcopy(fixtures[0])
    del record["labels"]["action"]["kind"]
    del record["labels"]["person_next"]["kind"]
    rows = M.question_rows([record], [_oracle(record)])
    blocks = {r["question"]: r["block"] for r in rows}
    assert blocks["action"] == "choice/action" and blocks["person_next"] == "choice/other"
