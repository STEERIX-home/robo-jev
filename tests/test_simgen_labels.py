"""simgen 정답기 (Task v0a 지시서 §3.6, docs/12 §5 정정본, 리뷰 1 질문 1·6·m2).

행동 choice = 기대 비용의 소프트맥스(argmax = 기대 비용 최소, '최선이었던 빈도'가 아님), noul = 사건 빈도(또는 닫힌 꼴), score = [0, 1]
양의 평균, 상태 choice = 사후 확률, 규칙 적용 choice = 결정적(확률 1). 가치표의 두 기제(충돌 속도에 따른 닿음 비용, 급제동 비용)는
values-v0에서 꺼져 있고 values-v0.1에서 켜진다.
"""

import numpy as np
import pytest

from robo_jev.simgen import labels as lb
from robo_jev.simgen.catalog import VALUES_PATH, VALUES_V01_PATH, load_config
from robo_jev.simgen.rollout import Outcomes

COSTS = {"contact_person": 100.0, "contact_object": 15.0, "spill_hot": 25.0, "delay_per_s": 1.0}
VALUES = {"costs": COSTS, "criteria": {"avoid_contact": {"contact_person": 1000.0}}}
MODEL = lb.cost_model(VALUES, None, "mobile_base")


def outcomes_two_options():
    """'risky'는 10 세계 중 1번 사람과 닿고 지연 0초, 'safe'는 닿지 않고 지연 3초."""
    contact = np.zeros((2, 10), dtype=bool)
    contact[0, 0] = True
    out = Outcomes(("risky", "safe"))
    out.events = {"contact": contact}
    out.quantities = {"delay": np.array([np.zeros(10), np.full(10, 3.0)])}
    out.cost_items = {"contact": "contact_person"}
    return out


def test_expected_cost_is_the_mean_cost_over_worlds():
    ec = lb.expected_costs(outcomes_two_options(), MODEL)
    # risky: 100 × 0.1 + 0 = 10, safe: 0 + 3 × 1 = 3
    assert ec.tolist() == pytest.approx([10.0, 3.0])


def test_action_probabilities_are_a_softmax_of_expected_cost_differences():
    label = lb.action_label(outcomes_two_options(), {"a1": "risky", "a2": "safe"}, MODEL, tau=2.0)
    assert label["type"] == "choice" and label["kind"] == "action"
    assert label["expected_cost"] == {"a1": 10.0, "a2": 3.0}
    z = np.exp(-np.array([10.0 - 3.0, 0.0]) / 2.0)
    expected = z / z.sum()
    assert label["probabilities"]["a1"] == pytest.approx(expected[0], abs=1e-8)
    assert label["probabilities"]["a2"] == pytest.approx(expected[1], abs=1e-8)
    assert sum(label["probabilities"].values()) == pytest.approx(1.0, abs=1e-6)


def test_action_label_is_not_the_frequency_of_being_best():
    """'최선이었던 빈도'라면 risky가 90 %(9/10 세계에서 지연 0으로 이긴다)를 받는다 — 정정된 정의는 기대 비용 최소(safe)를 고른다."""
    label = lb.action_label(outcomes_two_options(), {"risky": "risky", "safe": "safe"}, MODEL, tau=2.0)
    best_frequency = {"risky": 0.9, "safe": 0.1}
    assert max(label["probabilities"], key=label["probabilities"].get) == "safe"
    assert label["probabilities"]["risky"] < best_frequency["risky"]


def test_action_argmax_is_always_the_minimum_expected_cost_on_random_outcomes():
    rng = np.random.default_rng(0)
    for _ in range(300):
        a, k = int(rng.integers(2, 9)), 32
        out = Outcomes(tuple(f"o{i}" for i in range(a)))
        out.events = {"contact": rng.random((a, k)) < rng.random((a, 1)), "spill": rng.random((a, k)) < 0.3}
        out.quantities = {"delay": rng.random((a, 1)) * 4 + np.zeros((a, k))}
        out.cost_items = {"contact": "contact_person", "spill": "spill_hot"}
        keys = {f"k{i}": f"o{i}" for i in range(a)}
        label = lb.action_label(out, keys, MODEL, tau=2.0)
        best_p = max(label["probabilities"], key=label["probabilities"].get)
        assert label["expected_cost"][best_p] == min(label["expected_cost"].values())


def test_criterion_override_changes_only_the_named_costs():
    values = {"costs": dict(COSTS), "criteria": {"avoid_spill": {"spill_hot": 1000}}}
    costs = lb.costs_for(values, "avoid_spill")
    assert costs["spill_hot"] == 1000 and costs["contact_person"] == 100.0
    assert lb.costs_for(values, None) == COSTS


def test_noul_is_the_event_frequency_under_the_option():
    out = outcomes_two_options()
    assert lb.noul_label(out, {"source": "event", "event": "contact", "option": "risky"}) == {"type": "noul", "p_yes": 0.1}
    world = Outcomes(("x",))
    world.world_events = {"passable": np.array([True, True, False, True])}
    assert lb.noul_label(world, {"source": "world", "event": "passable"})["p_yes"] == 0.75
    assert lb.noul_label(world, {"source": "world", "event": "passable", "negate": True})["p_yes"] == 0.25


def test_score_is_the_mean_of_a_unit_interval_quantity():
    out = Outcomes(("a", "b"))
    out.quantities = {"closeness": np.array([[0.0, 0.5, 1.0, 0.5], [0.2, 0.2, 0.2, 0.2]]), "delay": np.array([[0.0, 1.5, 3.0, 6.0], [0.0] * 4])}
    out.events = {"contact": np.array([[True, False, False, False], [False] * 4])}
    assert lb.score_label(out, {"source": "quantity", "name": "closeness", "option": "a"})["value"] == 0.5
    # 지연은 3초를 1로 잘라 [0, 1]로: (0 + 0.5 + 1 + 1) / 4
    assert lb.score_label(out, {"source": "quantity", "name": "delay", "option": "a", "transform": "per_3s"})["value"] == 0.625
    assert lb.score_label(out, {"source": "event", "name": "contact", "option": "a"})["value"] == 0.25


def test_state_choice_is_the_posterior_share_of_each_state():
    out = Outcomes(("x",))
    out.states = {"intent": np.array([0, 0, 1, 2, 0, 1, 0, 0], dtype=float)}
    label = lb.state_label(out, {"state": "intent", "options": {"stays": [0], "crosses": [1], "backs": [2]}})
    assert label == {"type": "choice", "kind": "state", "probabilities": {"stays": 0.625, "crosses": 0.25, "backs": 0.125}}


def test_rule_choice_puts_all_mass_on_the_answer():
    label = lb.rule_label({"answer": "near", "options": ["far", "near", "close"]})
    assert label == {"type": "choice", "kind": "rule", "probabilities": {"far": 0.0, "near": 1.0, "close": 0.0}}


def test_probabilities_are_rounded_but_still_sum_to_one():
    out = Outcomes(tuple(f"o{i}" for i in range(7)))
    out.events = {"contact": np.zeros((7, 3), dtype=bool)}
    out.quantities = {"delay": np.array([[x] * 3 for x in (0.1, 0.2, 0.3, 0.35, 0.4, 0.45, 0.5)])}
    out.cost_items = {"contact": "contact_person"}
    label = lb.action_label(out, {f"k{i}": f"o{i}" for i in range(7)}, MODEL, tau=2.0)
    assert abs(sum(label["probabilities"].values()) - 1.0) <= 1e-6
    assert all(round(v, 8) == v for v in label["probabilities"].values())


def test_tied_expected_costs_get_exactly_equal_probabilities():
    out = Outcomes(("a", "b", "c"))
    out.events = {"contact": np.zeros((3, 4), dtype=bool)}
    out.quantities = {"delay": np.array([[0.0] * 4, [0.0] * 4, [2.5] * 4])}
    out.cost_items = {"contact": "contact_person"}
    label = lb.action_label(out, {"a": "a", "b": "b", "c": "c"}, MODEL, tau=2.0)
    assert label["probabilities"]["a"] == label["probabilities"]["b"] > label["probabilities"]["c"]


# ----------------------------------------------------------------------------------------------------------------
# 가치표의 두 기제 (리뷰 1 질문 1·6) — values-v0에서는 꺼짐, values-v0.1에서 켜짐
# ----------------------------------------------------------------------------------------------------------------


def outcomes_with_impact(speeds, hard=(False, True)):
    """두 선택지, 세계 4개: 선택지마다 닿음과 충격 속도."""
    out = Outcomes(("firm", "hard"), body="mobile_base")
    out.events = {"contact": np.array([[True, True, False, False], [True, False, False, False]])}
    out.impact = {"contact": np.array(speeds, dtype=float)}
    out.quantities = {"delay": np.zeros((2, 4))}
    out.cost_items = {"contact": "contact_person"}
    out.hard_brake = np.array(hard)
    return out


def speed_values(enabled: bool, floor: float = 0.25, hard_cost: float = 2.0) -> dict:
    return {**VALUES, "contact_speed": {"enabled": enabled, "v_ref": {"mobile_base": 1.0}, "floor": floor}, "hard_brake_cost": {"mobile_base": hard_cost}}


def test_contact_cost_scales_with_the_square_of_impact_speed_with_a_floor():
    out = outcomes_with_impact([[1.5, 0.5, 0.0, 0.0], [0.2, 0.0, 0.0, 0.0]], hard=(False, False))
    ec = lb.expected_costs(out, lb.cost_model(speed_values(True), None, "mobile_base"))
    # firm: (100 × min(1, 1.5²) + 100 × 0.5²) / 4 = (100 + 25) / 4; hard: 100 × max(0.25, 0.2²) / 4 = 25 / 4
    assert ec.tolist() == pytest.approx([125.0 / 4.0, 25.0 / 4.0])


def test_speed_scaling_is_off_when_disabled_and_skips_items_a_criterion_overrides():
    out = outcomes_with_impact([[1.5, 0.5, 0.0, 0.0], [0.2, 0.0, 0.0, 0.0]], hard=(False, False))
    off = lb.expected_costs(out, lb.cost_model(speed_values(False), None, "mobile_base"))
    assert off.tolist() == pytest.approx([50.0, 25.0])
    # 기준 'avoid_contact'가 덮어쓴 닿음 비용(1000)은 속도와 무관하게 그대로 — 기준은 '닿지 말 것'이다
    crit = lb.expected_costs(out, lb.cost_model(speed_values(True), "avoid_contact", "mobile_base"))
    assert crit.tolist() == pytest.approx([500.0, 250.0])


def test_hard_braking_carries_its_own_cost_per_body_and_load():
    out = outcomes_with_impact([[0.0] * 4, [0.0] * 4])
    out.events["contact"][:] = False
    model = lb.cost_model(speed_values(False, hard_cost=2.0), None, "mobile_base")
    assert lb.expected_costs(out, model).tolist() == pytest.approx([0.0, 2.0])
    table = {**VALUES, "hard_brake_cost": {"forklift_amr": 3.0, "forklift_amr_loaded": 6.0}}
    assert lb.cost_model(table, None, "forklift_amr", loaded=False)["hard_brake"] == 3.0
    assert lb.cost_model(table, None, "forklift_amr", loaded=True)["hard_brake"] == 6.0


def test_values_v0_has_both_mechanisms_off_and_values_v01_has_them_on():
    v0 = load_config(values_path=VALUES_PATH).values
    v01 = load_config(values_path=VALUES_V01_PATH).values
    out = outcomes_with_impact([[0.3, 0.3, 0.0, 0.0], [0.3, 0.0, 0.0, 0.0]])
    for body in ("mobile_base", "humanoid", "forklift_amr", "arm"):
        out.body = body
        plain = {"costs": {**v0["costs"], "contact_person": 100.0}, "criteria": {}}
        assert lb.expected_costs(out, lb.cost_model(v0, None, body)).tolist() == pytest.approx(lb.expected_costs(out, lb.cost_model(plain, None, body)).tolist())
    assert v01["contact_speed"]["enabled"] and 0.0 < v01["contact_speed"]["floor"] < 1.0
    assert all(v01["hard_brake_cost"][b] > 0 for b in ("mobile_base", "humanoid", "forklift_amr", "forklift_amr_loaded", "arm"))
    assert v01["hard_brake_cost"]["humanoid"] > v01["hard_brake_cost"]["mobile_base"]
    assert v01["hard_brake_cost"]["forklift_amr_loaded"] > v01["hard_brake_cost"]["forklift_amr"]


# ----------------------------------------------------------------------------------------------------------------
# 닫힌 꼴 정답 (리뷰 1 m2)
# ----------------------------------------------------------------------------------------------------------------


def test_calibration_makes_the_batch_match_the_closed_forms():
    """수정 1 M5: 닫힌 꼴은 정답에 직접 쓰지 않고 표본의 무게를 그 값에 맞춘다 — 그 뒤 noul·상태 정답(같은 무게의 빈도)이 닫힌 꼴과 같고,
    행동의 기대 비용도 같은 무게를 쓴다(한 레코드 안에서 닫힌 꼴과 K개 빈도를 섞지 않는다)."""
    import itertools

    grid = np.array(list(itertools.product((0, 1), (0, 1, 2), (0, 1), (0, 1))), dtype=float)   # 세계 24개 = 모든 조합(맞출 수 있다)
    world = Outcomes(("risky", "safe"))
    world.world_events = {"stops_short": grid[:, 0] > 0.5}
    world.states = {"intent": grid[:, 1], "lid": grid[:, 2]}
    world.events = {"contact": np.array([grid[:, 3] > 0.5, np.zeros(24, dtype=bool)])}
    world.quantities = {"delay": np.array([np.zeros(24), np.full(24, 2.0)])}
    world.cost_items = {"contact": "contact_person"}
    exact = {"stops_short": 0.9, "intent": np.array([0.2, 0.5, 0.3]), "lid": 0.1, "contact|risky": 0.25, "unknown_key": 0.5}
    result = lb.calibrate(world, exact)
    assert result == {"matched": ["contact|risky", "intent", "lid", "stops_short"], "unmatched": [], "trimmed": {}}
    noul = {"kind": "noul", "source": "world", "event": "stops_short"}
    assert lb.noul_label(world, noul)["p_yes"] == 0.9
    assert lb.noul_label(world, {**noul, "negate": True})["p_yes"] == pytest.approx(0.1)
    assert lb.state_label(world, {"state": "intent", "options": {"a": [0.0], "b": [1.0, 2.0]}})["probabilities"] == {"a": 0.2, "b": 0.8}
    assert lb.state_label(world, {"state": "lid", "options": {"has_lid": [1.0], "open": [0.0]}})["probabilities"] == {"has_lid": 0.1, "open": 0.9}
    # 행동의 기대 비용 = 같은 무게: P(닿음 | risky) 0.25 × 비용(그 밖 0) — 닫힌 꼴의 확률이 그대로 들어간다
    model = {"costs": {"contact_person": 10.0, "delay_per_s": 1.0}, "speed_scaled": False, "v_ref": 1.0, "floor": 1.0, "overridden": frozenset(), "hard_brake": 0.0}
    assert lb.expected_costs(world, model)[0] == pytest.approx(2.5)
    assert lb.uses_exact({"kind": "noul", **noul}, result["matched"]) and not lb.uses_exact({"kind": "noul", "source": "event", "event": "contact", "option": "safe"}, result["matched"])


def test_a_class_with_no_sample_is_trimmed_and_recorded_and_the_label_stays_the_batch_frequency():
    """표본이 하나도 없는 범주(꼬리)는 0으로 두고 나머지를 다시 나눠 맞춘다 — 잘라 낸 질량을 적는다; 모든 범주가 비면 맞추지 않는다."""
    world = Outcomes(("x",))
    world.world_events = {"passable": np.array([False, False, False, False])}
    world.states = {"intent": np.array([0.0, 0.0, 1.0, 1.0])}
    result = lb.calibrate(world, {"passable": 0.003, "intent": np.array([0.5, 0.3, 0.2])})
    assert result == {"matched": ["intent", "passable"], "unmatched": [], "trimmed": {"intent": 0.2, "passable": 0.003}}
    assert lb.noul_label(world, {"source": "world", "event": "passable"})["p_yes"] == 0.0
    assert lb.state_label(world, {"state": "intent", "options": {"a": [0.0], "b": [1.0], "c": [2.0]}})["probabilities"] == {"a": 0.625, "b": 0.375, "c": 0.0}
    assert lb.calibrate(world, {"passable": 1.0}) == {"matched": [], "unmatched": ["passable"], "trimmed": {}}