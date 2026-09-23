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


@functools.lru_cache(maxsize=4)
def real(suite=None):
    """실제 데이터의 판정 칸 — 편마다 한 번만 읽는다 (여러 시험이 같은 모집단을 본다). 돌려준 목록은 읽기 전용."""
    module = script()
    return tuple(module.cell_ticks(suite) if suite else module.cell_ticks())


P3_SUITE = REPO / "configs" / "eval" / "p3-decision-cell.yaml"


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


def test_the_primary_stratum_carries_a_leave_one_episode_out_refit():
    """N2의 물음 — **한 편이 판정을 만드는가**. 구간은 그 물음에 답하지 않으므로 따로 센다.

    fixture: 편 A는 모델만 맞고(여유가 거기 있다) 편 B는 두 열이 같다. A를 빼면 여유가 사라져야 하고, B를 빼면
    남아야 한다 — 그리고 "모든 제거에서 0을 제외했는가"가 한 줄로 적혀야 한다."""
    module = script()
    ticks = _ticks()
    others = {(row["episode_id"], row["tick"]) for row in ticks if not row["is_commitment"]}
    model = {key: True for key in {(r["episode_id"], r["tick"]) for r in ticks}}
    control = {key: (key[0] == "ep-B") for key in model}
    strata = module.run_strata(_table(ticks, model, control), ticks)
    loo = strata["primary_stratum_leave_one_episode_out"]

    assert loo["episodes"] == 2 and set(loo["drops"]) == {"ep-A", "ep-B"}
    assert loo["drops"]["ep-B"]["margin"] == 1.0        # 편 A만 남으면 모델이 대조군을 1.0으로 앞선다
    assert loo["drops"]["ep-A"]["margin"] == 0.0        # 편 B만 남으면 두 열이 같다
    assert loo["worst_drop"]["episode_id"] == "ep-A" and loo["worst_drop"]["margin"] == 0.0
    assert loo["every_drop_excludes_zero"] is False
    assert loo["drops"]["ep-B"]["graded"] == len({key for key in others if key[0] == "ep-A"})

    # 대조군 열이 없으면 이 블록도 없다 (있다고 주장하지 않는다)
    assert "primary_stratum_leave_one_episode_out" not in module.run_strata({"model": {"q_main": {"per_record": _per_record(ticks, model)}}}, ticks)


# --------------------------------------------------------------------------
# 리뷰 1 — 읽기 문장은 **이 파일의 수에서** 나온다, 기증자 회전, 갈래별 구간
# --------------------------------------------------------------------------


def test_the_reading_sentence_is_generated_from_this_cells_own_numbers():
    """C1 — 이 과제가 반박한 주장이 이 과제의 산출물 안에 있었다.

    `build()`는 `reading=`을 받았지만 `main()`이 넘기지 않아 `--runs p3`가 **P2의 문단을 그대로** 재생산했다
    ("595 of its 844 ticks", "the 4B T0 loses to its own control", "Both strata are 8 episodes"). 문장을 모집단의
    수에서 만들면 그 일이 구조적으로 불가능해진다: 새 모집단의 문장에는 옛 칸의 수가 들어갈 자리가 없다.
    이 시험은 옛 문단 위에서 **실패한다** — 그것이 이 시험의 일이다."""
    module = script()
    new = list(real(P3_SUITE))
    text = module.reading_text(
        module.population_composition(new), module.mechanism(new),
        module.donor_rotation(new, module.cell_records(P3_SUITE)),
    )
    assert "24 episodes / 2,530" in text and "525 ticks" in text and "2,005" in text
    assert "2,214/2,530 = 0.875" in text                      # 아무것도 읽지 않는 열이 문장 안에 있다
    assert "hold 251 · observe 210 · grasp 61 · place 3" in text
    assert "11.9 %" in text and "30.3 %" in text and "20.7 %" in text
    # 옛 모집단의 수는 하나도 들어 있지 않다 (여기 있으면 그것이 바로 C1이다)
    for stale in ("844", "595", "249", "8 episodes", "35.5 %", "63.9 %"):
        assert stale not in text
    # 판정 문장은 여기 적지 않는다 — 판정은 `runs[*]`의 구간이 한다
    for verdict in ("loses to its own control", "by a wide margin", "beats"):
        assert verdict not in text

    old = list(real())
    was = module.reading_text(
        module.population_composition(old), module.mechanism(old),
        module.donor_rotation(old, module.cell_records()),
    )
    assert "8 episodes / 844" in was and "249 ticks" in was and "751/844 = 0.890" in was and "30.1 %" in was
    assert "2,530" not in was and "525" not in was


def test_the_generated_reading_is_what_the_script_writes_and_a_flag_can_override_it(tmp_path):
    """`main()`이 그 문장을 실제로 싣는가 — C1의 원인은 계산이 아니라 `main()`이 넘기지 않은 것이었다."""
    module = script()
    out = tmp_path / "strata.json"
    assert module.main(["--suite", str(P3_SUITE), "--runs", "p3", "--reports", str(tmp_path), "--out", str(out)]) == 0
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["reading"].startswith("This cell is 24 episodes / 2,530")
    assert payload["donor_rotation"]["clamped_ticks"] == 0
    assert payload["donor_rotation"]["clamped_ticks_under_the_old_rule"] == 524

    assert module.main(["--suite", str(P3_SUITE), "--runs", "p3", "--reports", str(tmp_path),
                        "--reading", "quote nothing", "--out", str(out)]) == 0
    assert json.loads(out.read_text(encoding="utf-8"))["reading"] == "quote nothing"


def test_the_donor_rotation_is_measured_for_the_population_it_sits_in():
    """C1b/리뷰 1 I1 + Task R1 C2 — 대조군 값이 **어느 draw에서 나왔는지**가 파일 안에 있어야 한다.

    이제 회전은 **편 길이로 짝짓고** 남는 차이는 감아 돈다(`robo_jev.evaluate.context_shuffle_records`) — 그래서
    **고정된 틱은 0**이다. 옛 규칙(설정 순서 + 클램프)이었다면 몇 틱이 얼어붙었을지는 같은 파일에 남는다:
    8편 회전 254/844 = 30.1 %, 24편 회전 524/2,530 = 20.7 %. 두 수가 같이 있어야 "왜 P1~P3의 상태 섞기 값과
    나란히 놓을 수 없는가"가 파일 안의 수로 말해진다."""
    module = script()
    new = module.donor_rotation(list(real(P3_SUITE)), module.cell_records(P3_SUITE))
    assert new["ticks"] == 2530 and new["clamped_ticks"] == 0 and new["clamped_share"] == 0.0
    assert new["clamped_ticks_under_the_old_rule"] == 524
    assert round(new["clamped_share_under_the_old_rule"], 4) == 0.2071
    # 길이로 짝지으면 300틱짜리 둘이 서로를 받는다 — 감아 돌 틱도 없다.
    assert new["per_episode"]["ep-E1-000235"]["ticks"] == 300 and new["per_episode"]["ep-E1-000235"]["wrapped_ticks"] == 0
    assert all(entry["clamped_ticks"] == 0 for entry in new["per_episode"].values())

    old = module.donor_rotation(list(real()), module.cell_records())
    assert old["ticks"] == 844 and old["clamped_ticks"] == 0
    assert old["clamped_ticks_under_the_old_rule"] == 254 and round(old["clamped_share_under_the_old_rule"], 4) == 0.3009
    # P2의 +0.257을 만든 짝(옛 규칙): 300틱짜리가 82틱짜리를 받아 218틱이 얼어붙은 완료 상태와 섞였다 — 지금은
    # 길이 순이라 `ep-E1-000235`가 가장 긴 편이고 가장 짧은 편을 감아 돌아 읽는다.
    assert old["per_episode"]["ep-E1-000235"]["wrapped_ticks"] > 0


def test_each_key_family_carries_the_same_paired_interval_as_its_stratum():
    """리뷰 1 I4 — "여유가 전부 `grasp`에서 나온다"는 결론이 산출물에 없는 구간에 기대고 있었다.

    fixture: 편 A만 대조군이 틀리는 갈래 하나 + 두 열이 같은 갈래 하나. 갈래의 구간은 층과 **같은 쌍 부트스트랩**
    이어야 하고, 두 열이 같은 갈래는 0을 포함해야 한다."""
    module = script()
    ticks = _ticks()
    for row in ticks:  # 비-commitment 층을 두 갈래로 가른다
        if not row["is_commitment"]:
            row["key"] = "grasp" if row["episode_id"] == "ep-A" else "observe"
    model = {(row["episode_id"], row["tick"]): True for row in ticks}
    control = {(row["episode_id"], row["tick"]): (row["episode_id"] == "ep-B") for row in ticks}
    families = module.run_strata(_table(ticks, model, control), ticks)["primary_stratum_by_key_family"]

    from robo_jev.evaluate import episode_bootstrap

    assert set(families) == {"grasp", "observe"}
    assert families["grasp"]["n"] == 1 and families["grasp"]["state_shuffle_margin"] == 1.0
    assert families["grasp"]["state_shuffle_margin_includes_zero"] is False
    assert families["observe"]["state_shuffle_margin"] == 0.0 and families["observe"]["state_shuffle_margin_includes_zero"] is True
    wanted = {(row["episode_id"], row["tick"]) for row in ticks if row["key"] == "grasp" and not row["is_commitment"]}
    paired = episode_bootstrap(
        module.stratum_per_episode(_per_record(ticks, model), wanted),
        module.stratum_per_episode(_per_record(ticks, control), wanted),
    )
    assert families["grasp"]["state_shuffle_margin_ci"] == paired["margin_ci"]
    assert families["grasp"]["state_shuffle_margin_episode_balanced"] == paired["episode_balanced_margin"]


def test_rescoping_a_saved_report_touches_only_the_commitment_shuffle_scope(tmp_path):
    """리뷰 1 I3 — 이미 저장된 보고서에 범위를 적는 길은 **재실행이 아니다**.

    부트스트랩의 seed가 고정이라 저장된 블록은 CPU에서 비트 단위로 다시 나온다. 그러므로 이 경로는 범위와
    `q_main` 밖의 여유 말고는 **아무것도 바꿀 수 없어야** 하고, 달라지면 쓰지 않고 멈춰야 한다."""
    module = script()
    from robo_jev.evaluate import COMMITMENT_SHUFFLE_SCOPE, split_episode_bootstrap

    rows = [{"episode_id": name, "n": 10, "graded": 10, "correct": value} for name, value in (("a", 9), ("b", 8))]
    worse = [{"episode_id": name, "n": 10, "graded": 10, "correct": value} for name, value in (("a", 2), ("b", 1))]
    table = {
        "model": {qid: {"per_episode": rows} for qid in ("q_main", "q_done")},
        "context_shuffle": {qid: {"per_episode": worse} for qid in ("q_main", "q_done")},
        "commitment_shuffle": {qid: {"per_episode": worse} for qid in ("q_main", "q_done")},
        "commitment_shuffle_kind": "state_commitment",
    }
    stale = split_episode_bootstrap(table)
    stale["q_done"]["commitment_shuffle"] = {"control_accuracy": 0.15, "margin": 0.7, "margin_ci": [0.5, 0.9],
                                             "margin_half_width": 0.2, "margin_includes_zero": False}
    report = tmp_path / "reeval.json"
    report.write_text(json.dumps({"evaluation": {"splits": {module.SPLIT: {**table, "episode_bootstrap": stale}}}}), encoding="utf-8")

    assert module.main(["--rescope", str(report)]) == 0
    payload = json.loads(report.read_text(encoding="utf-8"))
    split = payload["evaluation"]["splits"][module.SPLIT]
    assert split["commitment_shuffle_scope"] == COMMITMENT_SHUFFLE_SCOPE
    assert split["episode_bootstrap"]["q_done"]["commitment_shuffle"]["margin"] is None          # 거짓 발견이 걷혔다
    assert split["episode_bootstrap"]["q_main"]["commitment_shuffle"]["margin_includes_zero"] is False  # 읽을 수 있는 칸은 그대로
    assert split["episode_bootstrap"]["q_main"]["state_shuffle"] == stale["q_main"]["state_shuffle"]
    assert payload["rescoped"]["by"].endswith("--rescope")

    # 저장된 다른 값이 다시 낸 것과 다르면 **쓰지 않고 멈춘다** — 이 경로가 측정을 바꿀 수는 없다
    split["episode_bootstrap"]["q_main"]["state_shuffle"]["margin"] = 0.123
    report.write_text(json.dumps(payload), encoding="utf-8")
    import pytest

    with pytest.raises(SystemExit):
        module.rescope(report)


def test_every_control_column_gets_its_paired_interval_in_the_key_family_table():
    """Task R2 C1 — **지시 섞기**의 갈래별 구간도 산출물에 있어야 한다.

    R1까지 갈래 표는 상태 섞기의 여유만 구간과 함께 실었다. R2의 한 줄짜리 답은 **지시 섞기**의 여유이고,
    "여유가 `grasp`에서 나오는가"는 같은 꼴로 그 열에도 물어야 한다 — 없으면 갈래별 지시 섞기 판정이 다시
    스크래치 스크립트로 내려간다(리뷰 1 I4가 상태 섞기에서 고친 바로 그 자리다)."""
    module = script()
    ticks = _ticks()
    for row in ticks:
        if not row["is_commitment"]:
            row["key"] = "grasp" if row["episode_id"] == "ep-A" else "observe"
    model = {(row["episode_id"], row["tick"]): True for row in ticks}
    control = {(row["episode_id"], row["tick"]): (row["episode_id"] == "ep-B") for row in ticks}
    instruction = {(row["episode_id"], row["tick"]): False for row in ticks}
    families = module.run_strata(_table(ticks, model, control, instruction=instruction), ticks)["primary_stratum_by_key_family"]

    from robo_jev.evaluate import episode_bootstrap

    for column in ("state_shuffle", "instruction_shuffle"):
        for family in ("grasp", "observe"):
            assert f"{column}_margin_ci" in families[family], (column, family)
            assert f"{column}_margin_includes_zero" in families[family]
    assert families["grasp"]["instruction_shuffle_margin"] == 1.0
    wanted = {(row["episode_id"], row["tick"]) for row in ticks if row["key"] == "observe" and not row["is_commitment"]}
    paired = episode_bootstrap(
        module.stratum_per_episode(_per_record(ticks, model), wanted),
        module.stratum_per_episode(_per_record(ticks, instruction), wanted),
    )
    assert families["observe"]["instruction_shuffle_margin_ci"] == paired["margin_ci"]
    assert families["observe"]["instruction_shuffle_margin_episode_balanced"] == paired["episode_balanced_margin"]


def test_the_leave_one_episode_out_check_runs_for_every_control_column():
    """Task R2 C1 — 한 줄짜리 답은 **지시 섞기**의 여유이므로 "한 편을 빼도 0을 제외하는가"도 그 열에 물어야 한다.

    옛 키(상태 섞기)는 자리를 지킨다 — 옛 보고서를 다시 만드는 경로가 그 이름을 읽는다."""
    module = script()
    ticks = _ticks()
    model = {(row["episode_id"], row["tick"]): True for row in ticks}
    control = {(row["episode_id"], row["tick"]): (row["episode_id"] == "ep-B") for row in ticks}
    instruction = {(row["episode_id"], row["tick"]): False for row in ticks}
    strata = module.run_strata(_table(ticks, model, control, instruction=instruction), ticks)

    by_control = strata["primary_stratum_leave_one_episode_out_by_control"]
    assert set(by_control) == {"state_shuffle", "instruction_shuffle"}
    assert by_control["state_shuffle"] == strata["primary_stratum_leave_one_episode_out"]
    # 모든 틱에서 대조군이 틀리는 지시 섞기 열은 어느 편을 빼도 여유가 1.0이다
    drops = by_control["instruction_shuffle"]["drops"]
    assert drops and all(row["margin"] == 1.0 for row in drops.values())
    assert by_control["instruction_shuffle"]["every_drop_excludes_zero"] is True


def test_the_r3a_run_sets_name_this_rounds_reports_and_keep_r2s_row_on_the_same_ruler():
    """Task R3a — seed 18·19와 466 step 행, 그리고 **R2의 233 step 행**이 한 표에 있어야 한다.

    네 줄을 한 자로 읽으려면 R2의 행이 그 표 안에 있어야 한다 — 같은 평가 집합, 같은 층, 같은 쌍 부트스트랩.
    둘째 칸(`dev`)의 짝은 같은 run들의 `r2-dev-`/`r3a-dev-` 보고서다.
    """
    module = script()
    assert set(module.RUN_SETS) >= {"r2", "r2dev", "r3a", "r3adev"}
    assert module.RUN_SETS["r3a"] is module.R3A_RUNS and module.RUN_SETS["r3adev"] is module.R3A_DEV_RUNS
    assert list(module.R3A_RUNS) == list(module.R3A_DEV_RUNS)
    assert len(module.R3A_RUNS) == 4
    # R2의 행은 R2가 쓴 바로 그 보고서다 (다시 평가하지 않는다)
    assert module.R3A_RUNS["2B T1 fp32 seed 17 (233 = 1 epoch)"] == "r2-reeval-2b-t1-fp32-233.json"
    assert module.R3A_DEV_RUNS["2B T1 fp32 seed 17 (233 = 1 epoch)"] == "r2-dev-2b-t1-fp32-233.json"
    # 이 판의 행들은 이 판의 보고서를 가리키고, 칸마다 접두사가 다르다
    for name, report in module.R3A_RUNS.items():
        if "seed 17 (233" in name:
            continue
        assert report.startswith("r3a-reeval-") and module.R3A_DEV_RUNS[name].startswith("r3a-dev-")
    assert "466" in module.R3A_RUNS["2B T1 fp32 seed 17 (466 = +1 epoch, rescheduled)"]
