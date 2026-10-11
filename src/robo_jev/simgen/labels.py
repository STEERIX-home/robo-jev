"""정답기: 사후 세계 K개의 굴림 결과에서 질문 종류별 정답 (docs/12 §5 정정본, Task v0a 지시서 §3.6).

* ``noul`` — K개 세계에서 그 사건이 일어난 비율.
* ``score`` — [0, 1] 양의 K개 평균(대개 어떤 사건의 확률, 또는 가까움·빈 높이 사용·3초로 자른 지연).
* 행동 ``choice`` — 선택지마다 기대 비용 E[c_a] = K개 세계의 비용 평균(사고 비용 합 + 진행 손실 초 × 시간 비용 + 급제동 비용),
  p_a = softmax(−(E[c_a] − min_b E[c_b]) / τ). argmax는 언제나 기대 비용 최소다. 기대 비용을 정답에 같이 적는다.
  **'각 선택지가 최선이었던 빈도'는 쓰지 않는다**(docs/12 §5 — 가끔 큰 사고를 내는 선택지가 높게 나온다).
  가치표의 두 기제(리뷰 1 질문 1·6): 닿음·넘어섬 비용 × max(floor, min(1, (충격 속도 / v_ref)²)) — 질문 기준이 덮어쓴 항목에는 걸지
  않는다(기준은 '닿지 않기'를 속도와 무관하게 요구한다); 급제동 선택지에 몸별 비용(짐 실은 지게차는 따로). values-v0에서는 둘 다 꺼짐.
* 상태 ``choice`` — 이산 상태(의도, 채움 구간, 뚜껑 …)의 사후 확률(무게 있는 빈도).

**한 레코드의 정답은 모두 같은 무게의 같은 표본에서 나온다**(수정 1 M5 — 라운드 0은 noul·상태 정답만 닫힌 꼴을 쓰고 행동·다시 보기의
기대 비용은 K개 빈도를 써서, 쌓기에서 행동의 argmax가 2 % 바뀌었다). 닫힌 꼴(:func:`~robo_jev.simgen.posterior.exact_labels` — 의도·
뚜껑·채움 구간, 세계 사건 stops_short·passable·present …, 선택지 사건 넘어섬·쌓기의 넘어짐)은 정답에 직접 쓰지 않고 **표본의 무게를
그 값에 맞춘다**(:func:`calibrate` — 반복 비례 맞춤(raking): 사건·상태마다 범주의 무게 합이 닫힌 꼴과 같아질 때까지 범주별로 무게를
곱한다). 그래서 noul·상태 정답은 닫힌 꼴과 같고(맞출 수 있으면 1e-10 안), 행동·다시 보기의 기대 비용도 그 확률을 쓴다. 표본에 그 범주가
하나도 없어 맞출 수 없는 열쇠는 맞추지 않고 기록한다(``unmatched``) — 그때도 정답은 같은 표본의 빈도다.
* 규칙 적용 ``choice`` — 질문 글에 적힌 규칙을 글에 적힌 값에 적용한 결과(확률 1). 답은 질문 생성기가 정한다.

반올림: 기대 비용은 소수 4자리로 먼저 반올림하고 그 값으로 확률을 계산한 뒤 소수 8자리로 반올림한다(합 오차 ≤ 4e-8 — 계약은 ±1e-6).
반올림이 argmax를 바꾸지 않고(기대 비용 차가 1e-4 이상이면 확률 차가 반올림 단위보다 훨씬 크다), 기대 비용이 같으면 확률도 같다.
세계 모듈을 import하지 않는다.
"""

from __future__ import annotations

import numpy as np

from robo_jev.simgen.rollout import Outcomes

__all__ = [
    "calibrate",
    "calibration_targets",
    "uses_exact",
    "voi_decision",
    "voi_label",
    "wmean",
    "action_label",
    "cost_model",
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


def cost_model(values: dict, criterion: str | None, body: str, loaded: bool = False) -> dict:
    """비용표 + 두 기제의 설정(그 몸에 맞춘 값)."""
    costs = costs_for(values, criterion)
    speed = values.get("contact_speed") or {}
    hard = values.get("hard_brake_cost") or {}
    hard_key = f"{body}_loaded" if loaded and f"{body}_loaded" in hard else body
    return {
        "costs": costs,
        "speed_scaled": bool(speed.get("enabled", False)),
        "v_ref": float((speed.get("v_ref") or {}).get(body, 1.0)),
        "floor": float(speed.get("floor", 1.0)),
        "overridden": frozenset(values["criteria"][criterion]) if criterion else frozenset(),
        "hard_brake": float(hard.get(hard_key, 0.0)),
    }


def _model_for(outcomes: Outcomes, values: dict, criterion: str | None) -> dict:
    return cost_model(values, criterion, outcomes.body, outcomes.loaded)


def event_weights(outcomes: Outcomes, model: dict, event: str) -> np.ndarray:
    """사건 하나의 세계별 비용 비율 (A, K): 일어났나(0/1) × 충격 속도 비율(기제가 켜졌고 기준이 덮어쓰지 않은 항목만)."""
    weight = outcomes.events[event].astype(float)
    item = outcomes.cost_items[event]
    if model["speed_scaled"] and event in outcomes.impact and item not in model["overridden"]:
        ratio = np.minimum(1.0, (outcomes.impact[event] / model["v_ref"]) ** 2)
        weight = weight * np.maximum(model["floor"], ratio)
    return weight


def wmean(values: np.ndarray, outcomes: Outcomes) -> np.ndarray:
    """마지막 축(세계)의 사후 평균 — 무게가 있으면 무게 평균, 없으면 산술 평균(v0a와 같은 수)."""
    values = np.asarray(values, dtype=float)
    if outcomes.weights is None:
        return values.mean(axis=-1)
    return np.sum(values * outcomes.weights, axis=-1) / float(np.sum(outcomes.weights))


def _base_costs(outcomes: Outcomes, model: dict) -> np.ndarray:
    costs = model["costs"]
    total = costs["delay_per_s"] * outcomes.quantities["delay"]
    for event, item in outcomes.cost_items.items():
        total = total + costs[item] * event_weights(outcomes, model, event)
    if outcomes.hard_brake is not None and model["hard_brake"] > 0.0:
        total = total + model["hard_brake"] * outcomes.hard_brake[:, None].astype(float)
    return total


def voi_decision(outcomes: Outcomes, model: dict):
    """'다시 보기' 뒤의 결정 규칙 — 정답의 사후(참 세계면 ``policy``가 가리키는 사후)에서 정한다."""
    source = outcomes.policy if outcomes.policy is not None else outcomes
    base = [source.index(o) for o in source.voi.base]
    return source.voi.decide(_base_costs(source, model)[base], source.weights)


def world_costs(outcomes: Outcomes, model: dict) -> np.ndarray:
    """세계마다의 비용 (A, K). 'look_again'(다시 보기)이 있으면 그 줄은 보는 시간 + 본 뒤의 최선(voi 모듈)."""
    total = _base_costs(outcomes, model)
    if outcomes.voi is not None and outcomes.voi.option in outcomes.options:
        base = [outcomes.index(o) for o in outcomes.voi.base]
        decision = voi_decision(outcomes, model)
        total = total.copy()
        total[outcomes.index(outcomes.voi.option)] = model["costs"]["delay_per_s"] * outcomes.voi.time + outcomes.voi.after(total[base], decision)
    return total


def expected_costs(outcomes: Outcomes, model: dict) -> np.ndarray:
    """선택지마다의 기대 비용 (A,) — 사후 평균."""
    return wmean(world_costs(outcomes, model), outcomes)


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


def action_label(outcomes: Outcomes, keymap: dict[str, str], model: dict, tau: float) -> dict:
    """행동 choice 정답. keymap = 질문의 선택지 열쇠 → 정식 선택지."""
    ec_all = expected_costs(outcomes, model)
    expected = {key: round(float(ec_all[outcomes.index(canon)]), 4) for key, canon in keymap.items()}
    return {
        "type": "choice",
        "kind": "action",
        "probabilities": softmax_probabilities(expected, tau),
        "expected_cost": expected,
    }


def _event(outcomes: Outcomes, recipe: dict) -> np.ndarray:
    """사건(세계마다 0/1, 또는 그 세계에서 일어날 확률 — P4의 행동 결과)."""
    if recipe["source"] == "world":
        values = outcomes.world_events[recipe["event"]]
    else:
        values = outcomes.events[recipe["event"]][outcomes.index(recipe["option"])]
    values = np.asarray(values, dtype=float)
    return 1.0 - values if recipe.get("negate") else values


def _exact_key(recipe: dict) -> str:
    """닫힌 꼴 정답의 열쇠: 세계 사건은 이름, 선택지 사건은 '사건|선택지'."""
    return recipe["event"] if recipe["source"] == "world" else f"{recipe['event']}|{recipe['option']}"


def noul_label(outcomes: Outcomes, recipe: dict) -> dict:
    return {"type": "noul", "p_yes": round(float(np.clip(wmean(_event(outcomes, recipe), outcomes), 0.0, 1.0)), 6)}


def _quantity(outcomes: Outcomes, recipe: dict) -> np.ndarray:
    i = outcomes.index(recipe["option"])
    if recipe["source"] == "event":
        values = outcomes.events[recipe["name"]][i].astype(float)
    else:
        values = outcomes.quantities[recipe["name"]][i]
    if recipe.get("transform") == "per_3s":
        values = np.clip(values / 3.0, 0.0, 1.0)
    elif recipe.get("transform") == "per_30s":
        values = np.clip(values / 30.0, 0.0, 1.0)
    return values


def score_label(outcomes: Outcomes, recipe: dict) -> dict:
    return {"type": "score", "value": round(float(np.clip(wmean(_quantity(outcomes, recipe), outcomes), 0.0, 1.0)), 6)}


def _state_values(outcomes: Outcomes, recipe: dict) -> np.ndarray:
    if "state" in recipe:
        return outcomes.states[recipe["state"]]
    return outcomes.world_events[recipe["world_event"]].astype(float)


def uses_exact(recipe: dict, keys) -> bool:
    """이 질문의 정답이 닫힌 꼴과 같은가 — ``keys`` = 표본의 무게를 맞춘 닫힌 꼴의 열쇠들(:func:`calibrate`의 ``matched``)."""
    if not keys:
        return False
    if recipe["kind"] == "noul":
        return _exact_key(recipe) in keys
    if recipe["kind"] == "state":
        return "state" in recipe and recipe["state"] in keys
    return False


def state_label(outcomes: Outcomes, recipe: dict) -> dict:
    values = _state_values(outcomes, recipe)
    probs = {key: float(wmean(np.isin(values, accepted).astype(float), outcomes)) for key, accepted in recipe["options"].items()}
    return {"type": "choice", "kind": "state", "probabilities": _rounded_distribution(probs)}


# ----------------------------------------------------------------------------------------------------------------
# 닫힌 꼴에 표본의 무게를 맞추기 (수정 1 M5)
# ----------------------------------------------------------------------------------------------------------------


def calibration_targets(outcomes: Outcomes, exact: dict) -> list[tuple[str, np.ndarray, np.ndarray]]:
    """닫힌 꼴마다 (열쇠, 세계별 범주 번호 (K,), 범주별 목표 확률) — 표본에서 그 상태·사건을 읽을 수 있는 것만.

    상태(``outcomes.states`` — 의도·채움 구간·뚜껑·P4 상태·센서가 본 것·무게 쪽)는 그 값이 범주 번호이고 닫힌 꼴은 범주 분포(뚜껑은
    P(있음) 하나); 세계 사건(열쇠 = 이름)과 선택지 사건(열쇠 '사건|선택지')은 일어남(1)/아님(0)의 두 범주다."""
    out = []
    for key, value in exact.items():
        if key in outcomes.states:
            cls = np.asarray(outcomes.states[key], dtype=float)
            dist = np.asarray(value, dtype=float)
            if dist.ndim == 0:
                dist = np.array([1.0 - float(dist), float(dist)])
        else:
            if key in outcomes.world_events:
                values = np.asarray(outcomes.world_events[key], dtype=float)
            elif "|" in key and key.split("|")[0] in outcomes.events and key.split("|")[1] in outcomes.options:
                event, option = key.split("|")
                values = np.asarray(outcomes.events[event][outcomes.index(option)], dtype=float)
            else:
                continue
            if np.ndim(value) != 0 or not np.all((values == 0.0) | (values == 1.0)):
                continue  # 세계마다 확률인 사건(P4의 행동 결과)은 범주가 아니다 — 맞추지 않는다
            cls = values
            dist = np.array([1.0 - float(value), float(value)])
        out.append((key, np.rint(cls).astype(int), np.clip(dist, 0.0, 1.0)))
    return out


def _class_mass(cls: np.ndarray, n: int, w: np.ndarray) -> np.ndarray:
    """범주별 무게 합 (n + 1,) — 마지막 칸 = 닫힌 꼴의 범주 밖(목표 0)."""
    idx = np.where((cls >= 0) & (cls < n), cls, n)
    return np.bincount(idx, weights=w, minlength=n + 1)


def calibrate(outcomes: Outcomes, exact: dict, *, tol: float = 1e-12, sweeps: int = 2000) -> dict:
    """표본의 무게를 닫힌 꼴에 맞춘다(반복 비례 맞춤 — 범주마다 무게에 목표/지금을 곱하기를 모든 열쇠에 번갈아; 무게는 ``outcomes.weights``를
    바꾼다). 표본이 하나도 없는 범주(목표가 아주 작은 꼬리 — 채움 구간의 끝, 0.3 % 넘어섬 …)는 0으로 두고 나머지 목표를 다시 나눠 맞춘다
    — 그 범주의 목표 질량을 ``trimmed``에 적는다(정답과 닫힌 꼴의 차는 그 이하다). 돌려준다: {"matched": 맞춘 열쇠, "unmatched": 끝내
    수렴하지 않았거나 표본이 있는 범주에 목표가 없는 열쇠, "trimmed": 열쇠 → 잘라 낸 목표 질량}."""
    targets = calibration_targets(outcomes, exact)
    if not targets:
        return {"matched": [], "unmatched": [], "trimmed": {}}
    k = len(targets[0][1])
    w = np.full(k, 1.0 / k) if outcomes.weights is None else np.asarray(outcomes.weights, dtype=float) / float(np.sum(outcomes.weights))
    usable, unmatched, trimmed = [], [], {}
    for key, cls, dist in targets:
        mass = _class_mass(cls, len(dist), w)
        empty = mass[: len(dist)] <= 0.0
        cut = float(np.sum(dist[empty]))
        if cut >= 1.0 - 1e-12:
            unmatched.append(key)
            continue
        if cut > 0.0:
            trimmed[key] = round(cut, 6)
            dist = np.where(empty, 0.0, dist) / (1.0 - cut)
        usable.append((key, cls, dist))
    for _ in range(sweeps):
        worst = 0.0
        for _key, cls, dist in usable:
            n = len(dist)
            mass = _class_mass(cls, n, w)
            total = float(mass.sum())
            worst = max(worst, float(np.max(np.abs(mass[:n] / total - dist))) + float(mass[n] / total))
            factor = np.append(np.where(mass[:n] > 0.0, dist * total / np.where(mass[:n] > 0.0, mass[:n], 1.0), 0.0), 0.0)
            w = w * factor[np.where((cls >= 0) & (cls < n), cls, n)]
        w = w / float(np.sum(w))
        if worst < tol:
            break
    matched = []
    for key, cls, dist in usable:
        mass = _class_mass(cls, len(dist), w)
        (matched if float(np.max(np.abs(mass[: len(dist)] - dist))) < 1e-9 else unmatched).append(key)
    outcomes.weights = w
    return {"matched": sorted(matched), "unmatched": sorted(unmatched), "trimmed": {key: trimmed[key] for key in sorted(trimmed) if key in matched}}


def rule_label(recipe: dict) -> dict:
    return {"type": "choice", "kind": "rule", "probabilities": {k: (1.0 if k == recipe["answer"] else 0.0) for k in recipe["options"]}}


def voi_label(outcomes: Outcomes, recipe: dict, model: dict, tau: float) -> dict:
    """'다시 볼까' 정답: 지금 정하기 = 지금 가능한 행동(base)의 최소 기대 비용, 먼저 다시 보기 = 'look_again'의 기대 비용(보는 시간 +
    본 뒤의 최선) — 두 기대 비용의 소프트맥스(행동 choice와 같은 꼴, argmax = 기대 비용 최소)."""
    ec = expected_costs(outcomes, model)
    now = min(float(ec[outcomes.index(o)]) for o in recipe["base"])
    look = float(ec[outcomes.index(recipe["look"])])
    expected = {key: round(now if canon == "decide_now" else look, 4) for key, canon in recipe["options"].items()}
    return {"type": "choice", "kind": "action", "probabilities": softmax_probabilities(expected, tau), "expected_cost": expected}


def label_for(outcomes: Outcomes, recipe: dict, values_cfg: dict) -> dict:
    """질문 하나의 정답(조리법 = 질문 생성기의 내부 기록) — 모든 종류가 같은 무게의 같은 표본에서(닫힌 꼴은 :func:`calibrate`가 무게에)."""
    kind = recipe["kind"]
    if kind == "action":
        return action_label(outcomes, recipe["options"], _model_for(outcomes, values_cfg, recipe.get("criterion")), values_cfg["tau"])
    if kind == "voi":
        return voi_label(outcomes, recipe, _model_for(outcomes, values_cfg, None), values_cfg["tau"])
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
        per_world = world_costs(true_outcomes, _model_for(true_outcomes, values_cfg, recipe.get("criterion")))[:, 0]
        realized_costs = {key: round(float(per_world[true_outcomes.index(canon)]), 4) for key, canon in recipe["options"].items()}
        best = min(realized_costs, key=realized_costs.get)
        return {"costs": realized_costs, "best": best}
    if kind == "voi":
        # 지금 정하기 = 정답의 사후에서 기대 비용이 가장 낮은 지금 행동을 참 세계에서; 먼저 다시 보기 = 사후의 결정 규칙을 참 세계의 신호에
        model = _model_for(true_outcomes, values_cfg, None)
        posterior = true_outcomes.policy
        ec = expected_costs(posterior, model)
        now_option = min(recipe["base"], key=lambda o: float(ec[posterior.index(o)]))
        per_world = world_costs(true_outcomes, model)[:, 0]
        costs = {"decide_now": float(per_world[true_outcomes.index(now_option)]), "look_again": float(per_world[true_outcomes.index(recipe["look"])])}
        realized_costs = {key: round(costs[canon], 4) for key, canon in recipe["options"].items()}
        return {"costs": realized_costs, "best": min(realized_costs, key=realized_costs.get)}
    if kind == "noul":
        return bool(_event(true_outcomes, recipe)[0] >= 0.5)
    if kind == "score":
        return round(float(_quantity(true_outcomes, recipe)[0]), 6)
    if kind == "state":
        value = _state_values(true_outcomes, recipe)[0]
        return next((key for key, accepted in recipe["options"].items() if value in accepted), None)
    if kind == "rule":
        return recipe["answer"]
    raise ValueError(kind)
