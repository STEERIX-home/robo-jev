"""`scripts/decision_cell_strata.py` — 판정 칸을 "라벨이 지금 commitment인가"로 가르는 계산 (P2 리뷰 1 I1).

이 계산이 이 라운드의 판정을 바꾼다: 집계 여유는 70 %가 commitment 반복인 모집단에서 잰 값이고, 읽기가 필요한
층에서 값이 갈린다. 그러니 계산 자체가 시험으로 묶여 있어야 한다 — 특히 (1) 층이 틱을 하나도 잃거나 겹치지 않고,
(2) 여유의 구간이 전체 칸과 **같은 쌍 부트스트랩**이며, (3) 틱별 예측이 없는 보고서는 빈칸으로 남고 그렇게 적힌다.
"""

import functools
import importlib.util
import json
import sys

from helpers import REPO

SCRIPT = REPO / "scripts" / "decision_cell_strata.py"


@functools.lru_cache(maxsize=1)
def script():
    spec = importlib.util.spec_from_file_location("decision_cell_strata", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _ticks():
    """두 편 × 다섯 틱. 편 A는 commitment 넷 + 비-commitment 하나, 편 B는 그 반대."""
    rows = []
    for episode, pattern in (("ep-A", [True, True, True, True, False]), ("ep-B", [False, False, False, False, True])):
        for tick, is_commitment in enumerate(pattern):
            rows.append({
                "episode_id": episode, "tick": tick, "label": ["a" if is_commitment else "b"],
                "commitment": "a" if is_commitment else None,
                "is_commitment": is_commitment, "key": "grasp" if is_commitment else "observe",
                "rule": "expert-e0.4/keep_commitment" if is_commitment else "other",
                "mechanical": "a" if is_commitment else "b",
            })  # fmt: skip
    return rows


def _per_record(ticks, correct):
    """``correct``는 (편, 틱) → 맞았는가. 예측 id는 commitment가 있으면 그것을 되풀이한 것으로 둔다."""
    return [
        {"record_id": row["episode_id"], "tick": row["tick"], "question": "q_main",
         "predicted": row["commitment"] or "z", "correct": bool(correct[(row["episode_id"], row["tick"])])}
        for row in ticks
    ]  # fmt: skip


def _table(ticks, model, control, instruction=None):
    out = {
        "model": {"q_main": {"per_record": _per_record(ticks, model)}},
        "context_shuffle": {"q_main": {"per_record": _per_record(ticks, control)}},
    }
    if instruction is not None:
        out["instruction_shuffle"] = {"q_main": {"per_record": _per_record(ticks, instruction)}}
    return out


def test_the_two_strata_partition_the_cell_and_each_carries_its_own_paired_interval():
    module = script()
    ticks = _ticks()
    # 모델은 어디서나 맞고, 대조군은 commitment 층에서만 맞는다 — 층을 가르면 여유가 한쪽에 몰려야 한다
    model = {(row["episode_id"], row["tick"]): True for row in ticks}
    control = {(row["episode_id"], row["tick"]): row["is_commitment"] for row in ticks}
    strata = module.run_strata(_table(ticks, model, control), ticks)

    assert strata["available"] is True
    assert strata["whole_cell"]["n"] == 10 == strata["commitment"]["n"] + strata["non_commitment"]["n"]
    assert strata["commitment"]["n"] == 5 and strata["non_commitment"]["n"] == 5
    assert strata["commitment"]["episodes"] == {"ep-A": 4, "ep-B": 1}  # 층은 편마다 크기가 다르다
    assert strata["commitment"]["model"] == 1.0 and strata["commitment"]["state_shuffle"] == 1.0
    assert strata["commitment"]["state_shuffle_margin"] == 0.0 and strata["commitment"]["state_shuffle_margin_includes_zero"]
    assert strata["non_commitment"]["model"] == 1.0 and strata["non_commitment"]["state_shuffle"] == 0.0
    assert strata["non_commitment"]["state_shuffle_margin"] == 1.0
    assert strata["non_commitment"]["state_shuffle_margin_includes_zero"] is False
    # 전체 칸의 여유는 두 층의 가운데다 — 이것이 "집계가 읽기를 희석한다"는 말의 산술이다
    assert strata["whole_cell"]["state_shuffle_margin"] == 0.5

    # 구간은 `evaluate.episode_bootstrap` 그대로여야 한다 (쌍, 편이 표본 단위)
    from robo_jev.evaluate import episode_bootstrap

    rows = module.stratum_per_episode(
        _per_record(ticks, model), {(row["episode_id"], row["tick"]) for row in ticks if not row["is_commitment"]}
    )
    control_rows = module.stratum_per_episode(
        _per_record(ticks, control), {(row["episode_id"], row["tick"]) for row in ticks if not row["is_commitment"]}
    )
    assert rows == [{"episode_id": "ep-A", "n": 1, "graded": 1, "correct": 1}, {"episode_id": "ep-B", "n": 4, "graded": 4, "correct": 4}]
    assert strata["non_commitment"]["state_shuffle_margin_ci"] == episode_bootstrap(rows, control_rows)["margin_ci"]


def test_the_instruction_shuffle_column_is_stratified_the_same_way_and_a_missing_column_is_left_out():
    module = script()
    ticks = _ticks()
    model = {(row["episode_id"], row["tick"]): True for row in ticks}
    control = {(row["episode_id"], row["tick"]): row["is_commitment"] for row in ticks}
    strata = module.run_strata(_table(ticks, model, control, instruction=model), ticks)
    assert strata["non_commitment"]["instruction_shuffle"] == 1.0
    assert strata["non_commitment"]["instruction_shuffle_margin"] == 0.0
    assert strata["non_commitment"]["instruction_shuffle_margin_includes_zero"] is True
    assert "instruction_shuffle" not in module.run_strata(_table(ticks, model, control), ticks)["non_commitment"]


def test_a_report_without_per_tick_predictions_is_named_rather_than_silently_skipped(tmp_path):
    """리뷰 1 M10의 짝 — 없는 줄은 **빈칸으로 남고 그렇게 적혀야** 한다(조용히 건너뛰면 표가 완전해 보인다)."""
    module = script()
    ticks = _ticks()
    assert module.run_strata({"model": {"q_main": {}}}, ticks)["available"] is False
    assert "store_predictions" in module.run_strata({"model": {"q_main": {}}}, ticks)["reason"]

    model = {(row["episode_id"], row["tick"]): True for row in ticks}
    (tmp_path / "with.json").write_text(json.dumps({"evaluation": {"eval_set": {"sha256": "x"}, "splits": {module.SPLIT: _table(ticks, model, model)}}}), encoding="utf-8")
    (tmp_path / "without.json").write_text(json.dumps({"evaluation": {"splits": {module.SPLIT: {"model": {"q_main": {"accuracy": 0.5}}}}}}), encoding="utf-8")
    built = module.build(reports=tmp_path, runs={"has it": "with.json", "has none": "without.json", "never ran": "absent.json"})
    assert set(built["runs"]) == {"has it"} and set(built["missing"]) == {"has none", "never ran"}
    assert built["missing"]["never ran"]["reason"] == "report not produced"
    assert built["runs"]["has it"]["eval_set_sha256"] == "x"


def test_the_mechanism_block_counts_what_the_control_carries_through():
    module = script()
    ticks = _ticks()
    found = module.mechanism(ticks)
    assert found["ticks"] == 10 and found["label_is_the_commitment"] == 5 and found["share_of_all_ticks"] == 0.5
    assert found["ticks_with_a_commitment"] == 5 and found["share_of_ticks_that_have_one"] == 1.0
    assert found["ticks_without_a_commitment"] == 5 and found["label_differs_from_a_live_commitment"] == 0
    assert found["mechanical_policy"]["accuracy"] == 1.0  # 이 fixture에서는 아무것도 안 읽고 다 맞는다
    assert found["non_commitment_key_families"] == {"observe": 5}
    assert found["ticks_per_episode"] == {"ep-A": 5, "ep-B": 5}
    assert found["keep_commitment_rule"] == 5

    # 섞인 열이 commitment를 그대로 되풀이하는 비율 — 대조군이 무엇을 하고 있는지
    repeats = module.repeats_the_commitment(_per_record(ticks, {(r["episode_id"], r["tick"]): r["is_commitment"] for r in ticks}), ticks)
    assert repeats["ticks_with_a_commitment"] == 5 and repeats["predicted_that_commitment"] == 5 and repeats["share"] == 1.0
    assert repeats["correct_answers"] == 5 and repeats["share_of_correct_answers"] == 1.0


def test_the_real_decision_cell_is_seventy_percent_commitment_repetition():
    """데이터셋 쪽 기전 — 리뷰어가 CPU로 잰 수(595/844 = 70.5 %, 0.890)를 그대로 다시 낸다."""
    module = script()
    ticks = module.cell_ticks()
    found = module.mechanism(ticks)
    assert found["ticks"] == 844 and found["singleton_labels"] == 844
    assert found["ticks_with_a_commitment"] == 604 and found["label_is_the_commitment"] == 595
    assert round(found["share_of_all_ticks"], 4) == 0.7050 and round(found["share_of_ticks_that_have_one"], 4) == 0.9851
    assert found["ticks_without_a_commitment"] == 240 and found["label_differs_from_a_live_commitment"] == 9
    assert found["keep_commitment_rule"] == 565
    assert found["mechanical_policy"]["correct"] == 751 and round(found["mechanical_policy"]["accuracy"], 4) == 0.8898
    assert found["non_commitment_key_families"] == {"observe": 157, "hold": 83, "grasp": 9}
    assert sum(found["non_commitment_key_families"].values()) == 844 - 595 == 249
    # 한 편이 칸의 3분의 1이고 비-commitment 층의 3분의 2다 (M4 / N2)
    assert found["ticks_per_episode"]["ep-E1-000235"] == 300
    assert sorted(found["ticks_per_episode"].values()) == [67, 70, 72, 79, 80, 82, 94, 300]
    assert found["non_commitment_per_episode"]["ep-E1-000235"] == 159


# --------------------------------------------------------------------------
# P3 — 모집단 구성과 새 열들
# --------------------------------------------------------------------------


def test_the_population_block_names_the_largest_episode_and_what_the_stratum_is_made_of():
    """A1/N2 — "249틱"은 실제보다 균형 있게 읽힌다. 층의 크기가 아니라 **몇 편에서 왔는지**를 같이 적는다."""
    module = script()
    found = module.population_composition(_ticks())
    assert found["episodes"] == 2 and found["ticks"] == 10
    assert found["non_commitment_ticks"] == 5 and found["non_commitment_share"] == 0.5
    assert found["key_families"] == {"observe": 5}
    assert found["largest_episode"]["ticks"] == 5 and found["largest_episode"]["share_of_all_ticks"] == 0.5
    stratum = found["largest_episode_of_the_non_commitment_stratum"]
    assert stratum["episode_id"] == "ep-B" and stratum["ticks"] == 4 and stratum["share_of_the_stratum"] == 0.8
    assert found["episodes_with_no_non_commitment_tick"] == []
    assert found["per_episode"]["ep-A"] == {"ticks": 5, "non_commitment_ticks": 1, "non_commitment_share": 0.2, "key_families": {"observe": 1}}


def test_the_strata_table_carries_the_commitment_shuffle_margin_the_baselines_and_both_means():
    """B2·B3·A3의 열들이 층화 표에 그대로 들어온다 — 없는 열은 조용히 빠진다(옛 보고서도 다시 만들어진다)."""
    module = script()
    ticks = _ticks()
    model = {(row["episode_id"], row["tick"]): True for row in ticks}
    control = {(row["episode_id"], row["tick"]): row["is_commitment"] for row in ticks}
    table = _table(ticks, model, control)
    table["commitment_shuffle"] = {"q_main": {"per_record": _per_record(ticks, dict.fromkeys(control, False))}}
    table["mechanical_baseline"] = {"q_main": {"per_record": _per_record(ticks, control)}}
    strata = module.run_strata(table, ticks)

    block = strata["non_commitment"]
    assert block["commitment_shuffle"] == 0.0 and block["commitment_shuffle_margin"] == 1.0
    assert block["commitment_shuffle_margin_includes_zero"] is False
    assert block["mechanical_baseline"] == 0.0 and "mechanical_baseline_margin" not in block  # 기준선은 여유를 세우지 않는다
    assert block["model_episode_balanced"] == 1.0
    assert block["state_shuffle_margin_episode_balanced"] == 1.0
    assert isinstance(block["state_shuffle_margin_episode_balanced_ci"], list)

    without = module.run_strata(_table(ticks, model, control), ticks)
    assert "commitment_shuffle" not in without["non_commitment"] and "mechanical_baseline" not in without["non_commitment"]
    assert module.PRIMARY_STRATUM == "non_commitment"


def test_the_new_population_is_the_whole_ood_dev_split_and_no_episode_owns_the_stratum():
    """A1/A2 — 새 모집단의 실측. 이 수들이 P3의 판정을 받치므로 시험이 들고 있어야 한다.

    옛 칸(8편)과 새 칸(24편 전부)을 같은 함수로 재서 나란히 둔다: 한 편의 몫이 35.5 % → 11.9 %,
    읽기가 필요한 층에서 63.9 % → 30.3 %로 내려가고, **24편 모두가 그 층에 틱을 낸다**."""
    module = script()
    old = module.population_composition(module.cell_ticks())
    new = module.population_composition(module.cell_ticks(module.REPO / "configs" / "eval" / "p3-decision-cell.yaml"))

    assert old["episodes"] == 8 and old["ticks"] == 844 and old["non_commitment_ticks"] == 249
    assert round(old["largest_episode"]["share_of_all_ticks"], 4) == 0.3555
    assert round(old["largest_episode_of_the_non_commitment_stratum"]["share_of_the_stratum"], 4) == 0.6386

    assert new["episodes"] == 24 and new["ticks"] == 2530 and new["non_commitment_ticks"] == 525
    assert new["key_families"] == {"hold": 251, "observe": 210, "grasp": 61, "place": 3}
    assert new["largest_episode"]["episode_id"] == "ep-E1-000235" and new["largest_episode"]["ticks"] == 300
    assert round(new["largest_episode"]["share_of_all_ticks"], 4) == 0.1186
    assert round(new["largest_episode_of_the_non_commitment_stratum"]["share_of_the_stratum"], 4) == 0.3029
    assert new["episodes_with_no_non_commitment_tick"] == []
    assert min(entry["non_commitment_ticks"] for entry in new["per_episode"].values()) == 3
    assert sorted(new["per_episode"]) == module.split_episodes(module.DEFAULT_MANIFEST, "ood_dev")

    # 아무것도 읽지 않는 정책은 넓힌 모집단에서도 여전히 0.875다 — 희석은 줄었지만 사라지지 않았다
    found = module.mechanism(module.cell_ticks(module.REPO / "configs" / "eval" / "p3-decision-cell.yaml"))
    assert found["label_is_the_commitment"] == 2005 and round(found["share_of_all_ticks"], 4) == 0.7925
    assert found["mechanical_policy"]["correct"] == 2214 and round(found["mechanical_policy"]["accuracy"], 4) == 0.8751
