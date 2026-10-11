"""simgen 정답기 (Task v0a 지시서 §3.6, docs/12 §5 정정본).

행동 choice = 기대 비용의 소프트맥스(argmax = 기대 비용 최소, '최선이었던 빈도'가 아님), noul = 사건 빈도, score = [0, 1] 양의 평균,
상태 choice = 사후 비율, 규칙 적용 choice = 결정적(확률 1).
"""

import numpy as np
import pytest

from robo_jev.simgen import labels as lb
from robo_jev.simgen.rollout import Outcomes

COSTS = {"contact_person": 100.0, "contact_object": 15.0, "spill_hot": 25.0, "delay_per_s": 1.0}


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
    ec = lb.expected_costs(outcomes_two_options(), COSTS)
    # risky: 100 × 0.1 + 0 = 10, safe: 0 + 3 × 1 = 3
    assert ec.tolist() == pytest.approx([10.0, 3.0])


def test_action_probabilities_are_a_softmax_of_expected_cost_differences():
    label = lb.action_label(outcomes_two_options(), {"a1": "risky", "a2": "safe"}, COSTS, tau=2.0)
    assert label["type"] == "choice" and label["kind"] == "action"
    assert label["expected_cost"] == {"a1": 10.0, "a2": 3.0}
    z = np.exp(-np.array([10.0 - 3.0, 0.0]) / 2.0)
    expected = z / z.sum()
    assert label["probabilities"]["a1"] == pytest.approx(expected[0], abs=1e-8)
    assert label["probabilities"]["a2"] == pytest.approx(expected[1], abs=1e-8)
    assert sum(label["probabilities"].values()) == pytest.approx(1.0, abs=1e-6)


def test_action_label_is_not_the_frequency_of_being_best():
    """'최선이었던 빈도'라면 risky가 90 %(9/10 세계에서 지연 0으로 이긴다)를 받는다 — 정정된 정의는 기대 비용 최소(safe)를 고른다."""
    label = lb.action_label(outcomes_two_options(), {"risky": "risky", "safe": "safe"}, COSTS, tau=2.0)
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
        label = lb.action_label(out, keys, COSTS, tau=2.0)
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
    label = lb.action_label(out, {f"k{i}": f"o{i}" for i in range(7)}, COSTS, tau=2.0)
    assert abs(sum(label["probabilities"].values()) - 1.0) <= 1e-6
    assert all(round(v, 8) == v for v in label["probabilities"].values())


def test_tied_expected_costs_get_exactly_equal_probabilities():
    out = Outcomes(("a", "b", "c"))
    out.events = {"contact": np.zeros((3, 4), dtype=bool)}
    out.quantities = {"delay": np.array([[0.0] * 4, [0.0] * 4, [2.5] * 4])}
    out.cost_items = {"contact": "contact_person"}
    label = lb.action_label(out, {"a": "a", "b": "b", "c": "c"}, COSTS, tau=2.0)
    assert label["probabilities"]["a"] == label["probabilities"]["b"] > label["probabilities"]["c"]
