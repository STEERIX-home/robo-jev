"""정답기: 사후 세계 K개의 굴림 결과에서 질문 종류별 정답 (docs/12 §5 정정본, Task v0a 지시서 §3.6).

* ``noul`` — K개 세계에서 그 사건이 일어난 비율.
* ``score`` — [0, 1] 양의 K개 평균(대개 어떤 사건의 확률, 또는 가까움·빈 높이 사용·3초로 자른 지연).
* 행동 ``choice`` — 선택지마다 기대 비용 E[c_a] = K개 세계의 비용 평균(사고 비용 합 + 진행 손실 초 × 시간 비용),
  p_a = softmax(−(E[c_a] − min_b E[c_b]) / τ). argmax는 언제나 기대 비용 최소다. 기대 비용을 정답에 같이 적는다.
  **'각 선택지가 최선이었던 빈도'는 쓰지 않는다**(docs/12 §5 — 가끔 큰 사고를 내는 선택지가 높게 나온다).
* 상태 ``choice`` — 이산 상태(의도, 채움 구간, 뚜껑 …)의 사후 비율 그대로.
* 규칙 적용 ``choice`` — 질문 글에 적힌 규칙을 글에 적힌 값에 적용한 결과(확률 1). 답은 질문 생성기가 정한다.

반올림: 기대 비용은 소수 4자리로 먼저 반올림하고 그 값으로 확률을 계산한 뒤 소수 8자리로 반올림한다(합 오차 ≤ 4e-8 — 계약은 ±1e-6).
반올림이 argmax를 바꾸지 않고(기대 비용 차가 1e-4 이상이면 확률 차가 반올림 단위보다 훨씬 크다), 기대 비용이 같으면 확률도 같다.
세계 모듈을 import하지 않는다.
"""

from __future__ import annotations

import numpy as np

from robo_jev.simgen.rollout import Outcomes

__all__ = [
    "action_label",
    "costs_for",
    "expected_costs",
    "label_for",
    "noul_label",
    "realized",
    "rule_label",
    "score_label",
    "softmax_probabilities",
    "state_label",
]


def costs_for(values: dict, criterion: str | None) -> dict[str, float]:
    """기본 가치표에 질문 기준의 덮어쓰기를 얹은 비용표."""
    costs = {k: float(v) for k, v in values["costs"].items()}
    if criterion:
        costs.update({k: float(v) for k, v in values["criteria"][criterion].items()})
    return costs


def world_costs(outcomes: Outcomes, costs: dict[str, float]) -> np.ndarray:
    """세계마다의 비용 (A, K)."""
    total = costs["delay_per_s"] * outcomes.quantities["delay"]
    for event, item in outcomes.cost_items.items():
        total = total + costs[item] * outcomes.events[event]
    return total


def expected_costs(outcomes: Outcomes, costs: dict[str, float]) -> np.ndarray:
    """선택지마다의 기대 비용 (A,)."""
    return world_costs(outcomes, costs).mean(axis=1)


def _rounded_distribution(probs: dict[str, float]) -> dict[str, float]:
    """소수 8자리 반올림. 합의 오차는 열쇠 8개라도 4e-8 이하라 따로 맞추지 않는다 — 맞추면 같은 기대 비용의 두 선택지가
    다른 확률을 받는다(같음이 깨진다). 반올림은 단조라 순서(약한 순서)는 그대로다."""
    return {k: round(float(v), 8) for k, v in probs.items()}


def softmax_probabilities(expected: dict[str, float], tau: float) -> dict[str, float]:
    """p = softmax(−(E − min E) / τ) (소수 8자리)."""
    keys = list(expected)
    ec = np.array([expected[k] for k in keys], dtype=float)
    z = np.exp(-(ec - ec.min()) / tau)
    p = z / z.sum()
    return _rounded_distribution(dict(zip(keys, p, strict=True)))


def action_label(outcomes: Outcomes, keymap: dict[str, str], costs: dict[str, float], tau: float) -> dict:
    """행동 choice 정답. keymap = 질문의 선택지 열쇠 → 정식 선택지."""
    ec_all = expected_costs(outcomes, costs)
    expected = {key: round(float(ec_all[outcomes.index(canon)]), 4) for key, canon in keymap.items()}
    return {
        "type": "choice",
        "kind": "action",
        "probabilities": softmax_probabilities(expected, tau),
        "expected_cost": expected,
    }


def _event(outcomes: Outcomes, recipe: dict) -> np.ndarray:
    if recipe["source"] == "world":
        values = outcomes.world_events[recipe["event"]]
    else:
        values = outcomes.events[recipe["event"]][outcomes.index(recipe["option"])]
    return ~values if recipe.get("negate") else values


def noul_label(outcomes: Outcomes, recipe: dict) -> dict:
    return {"type": "noul", "p_yes": round(float(np.mean(_event(outcomes, recipe))), 6)}


def _quantity(outcomes: Outcomes, recipe: dict) -> np.ndarray:
    i = outcomes.index(recipe["option"])
    if recipe["source"] == "event":
        values = outcomes.events[recipe["name"]][i].astype(float)
    else:
        values = outcomes.quantities[recipe["name"]][i]
    if recipe.get("transform") == "per_3s":
        values = np.clip(values / 3.0, 0.0, 1.0)
    return values


def score_label(outcomes: Outcomes, recipe: dict) -> dict:
    return {"type": "score", "value": round(float(np.clip(np.mean(_quantity(outcomes, recipe)), 0.0, 1.0)), 6)}


def _state_values(outcomes: Outcomes, recipe: dict) -> np.ndarray:
    if "state" in recipe:
        return outcomes.states[recipe["state"]]
    return outcomes.world_events[recipe["world_event"]].astype(float)


def state_label(outcomes: Outcomes, recipe: dict) -> dict:
    values = _state_values(outcomes, recipe)
    probs = {key: float(np.mean(np.isin(values, accepted))) for key, accepted in recipe["options"].items()}
    return {"type": "choice", "kind": "state", "probabilities": _rounded_distribution(probs)}


def rule_label(recipe: dict) -> dict:
    return {"type": "choice", "kind": "rule", "probabilities": {k: (1.0 if k == recipe["answer"] else 0.0) for k in recipe["options"]}}


def label_for(outcomes: Outcomes, recipe: dict, values_cfg: dict) -> dict:
    """질문 하나의 정답(조리법 = 질문 생성기의 내부 기록)."""
    kind = recipe["kind"]
    if kind == "action":
        return action_label(outcomes, recipe["options"], costs_for(values_cfg, recipe.get("criterion")), values_cfg["tau"])
    if kind == "noul":
        return noul_label(outcomes, recipe)
    if kind == "score":
        return score_label(outcomes, recipe)
    if kind == "state":
        return state_label(outcomes, recipe)
    if kind == "rule":
        return rule_label(recipe)
    raise ValueError(kind)


def realized(true_outcomes: Outcomes, recipe: dict, values_cfg: dict):
    """참 세계(K = 1)에서 실제로 일어난 것 — 평가용 기록."""
    kind = recipe["kind"]
    if kind == "action":
        costs = costs_for(values_cfg, recipe.get("criterion"))
        per_world = world_costs(true_outcomes, costs)[:, 0]
        realized_costs = {key: round(float(per_world[true_outcomes.index(canon)]), 4) for key, canon in recipe["options"].items()}
        best = min(realized_costs, key=realized_costs.get)
        return {"costs": realized_costs, "best": best}
    if kind == "noul":
        return bool(_event(true_outcomes, recipe)[0])
    if kind == "score":
        return round(float(_quantity(true_outcomes, recipe)[0]), 6)
    if kind == "state":
        value = _state_values(true_outcomes, recipe)[0]
        return next((key for key, accepted in recipe["options"].items() if value in accepted), None)
    if kind == "rule":
        return recipe["answer"]
    raise ValueError(kind)
