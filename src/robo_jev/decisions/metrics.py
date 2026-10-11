"""모델 답 대 사후 확률 정답의 지표 (Task v0c §2.3; 기획서 docs/12 §5·§10).

정답은 확률이다(관측이 주어졌을 때 가능한 세계들에서 결과가 나온 빈도, 기획서 §5). 그래서 지표는 모두 **정답 확률과의
거리**로 잰다 — 참 세계의 0/1 결과(``truth.realized``)는 쓰지 않는다.

질문 하나의 지표 (행 = 기록 × 질문; 묶음 열쇠 = :func:`block_of`):

* ``noul``·``score`` — y = 정답(p_yes / value), p = 모델의 값. 로그 손실 = soft 이진 교차 엔트로피
  −[y ln p + (1−y) ln(1−p)] (p는 [EPS, 1−EPS]로 자른다), KL = 로그 손실 − H(y) (0이면 완벽; 정답이 확률이라 로그 손실의
  바닥은 0이 아니라 H(y)다), Brier = (p − y)², |p − y|. ECE는 묶음에서 10구간(p 기준 같은 폭)마다 |평균 p − 평균 y|의 가중합.
* ``choice`` — q = 정답 분포, p = 모델 분포(같은 키). 교차 엔트로피 −Σ q ln p, KL, Brier Σ(p−q)². **최선 집합** = 기대 비용이
  있으면 기대 비용 최소인 선택지들(행동 choice; 기획서 §5 — argmax 정답 = 기대 비용 최소), 없으면 정답 확률 최대인 선택지들
  (같은 값 1e-9 안은 모두). 일치 = 모델 argmax ∈ 최선 집합. **기대 비용 후회** = E[c(모델 argmax)] − min E[c]
  (`expected_cost`가 있을 때; 가장 중요한 지표), 기대 후회 = Σ p_a E[c_a] − min E[c] (모델 분포에서 뽑아 실행했을 때).
  상태 choice의 ECE는 soft top-label: 확신 = max p, 표적 = q[모델 argmax].
* 묶음 ``choice/rule``(적힌 규칙 적용)은 정확도가 주 지표다(정답이 결정적이라 KL = CE).

묶음(:func:`question_rows` 의 ``groups``): 장면, 현상(``P1``·``P5``·``P1+P5``), 분야, 몸, 감각 구성, 문체, 언어, 힌트 유무,
권고(없음·맞음·틀림), 여유 구간(:func:`margin_bands`), 질문 계열(:func:`question_family`), 쌍 종류, 정답 최선 선택지의 위치.

따로 보는 것: 권고(:func:`recommendation_summary` — 틀린 권고를 따른 비율), 위치 편향(:func:`position_bias`), 쌍
(:func:`pair_summary` — 숫자 쌍의 방향 맞음, 단어 쌍의 흔들림), 균등 기준선(:func:`uniform_reference`), 레코드 단위
bootstrap 구간(:func:`bootstrap`), 같은 레코드의 두 run 비교(:func:`compare_runs`, 짝지은 bootstrap).
"""

from __future__ import annotations

import itertools
import math
import statistics
from collections import Counter, defaultdict
from typing import Any, Iterable

import numpy as np

from robo_jev.decisions.records import pair_of

__all__ = [
    "BLOCKS",
    "ECE_BINS",
    "EPS",
    "FLIP_THRESHOLD",
    "GROUP_DIMENSIONS",
    "HEADLINES",
    "MOVE_MIN",
    "aggregate",
    "average_over_orders",
    "binary_metrics",
    "block_of",
    "bootstrap",
    "choice_metrics",
    "compare_runs",
    "ece",
    "group_tables",
    "margin_bands",
    "order_effects",
    "pair_summary",
    "permutation_sensitivity",
    "position_bias",
    "question_family",
    "question_rows",
    "recommendation_summary",
    "summarize",
    "uniform_reference",
]

EPS = 1e-12
ECE_BINS = 10
#: 쌍에서 정답이 "움직였다"고 보는 최소 변화 — noul·score는 |Δ|, choice는 전변동 거리 ½Σ|Δq|.
MOVE_MIN = 0.05
#: 모델이 "움직이지 않았다"고 보는 변화 (숫자 쌍의 still 비율).
STILL_MAX = 0.01
#: noul·score의 "판정 뒤집힘" 문턱 (단어 쌍).
FLIP_THRESHOLD = 0.5
#: 같은 쪽 일치(side agreement)에서 빼는 애매한 정답의 폭: |y − 0.5| < 이 값이면 세지 않는다.
SIDE_MARGIN = 0.1
_TIE = 1e-9
BLOCKS = ("noul", "score", "choice/action", "choice/state", "choice/rule", "choice/other")
GROUP_DIMENSIONS = (
    "scene", "phenomena", "domain", "body", "sensor_suite", "style", "language", "hint", "recommendation", "margin_band",
    "family", "pair", "label_position",
)
_BAND_NAMES = ("small", "medium", "large")
#: bootstrap·run 비교의 머리 지표 — (이름, 행 고르기, 값). 이름은 ``<묶음>.<지표>``.
HEADLINES: tuple[tuple[str, str, str], ...] = (
    ("noul.log_loss", "noul", "log_loss"),
    ("noul.kl", "noul", "kl"),
    ("noul.brier", "noul", "brier"),
    ("score.log_loss", "score", "log_loss"),
    ("score.kl", "score", "kl"),
    ("score.brier", "score", "brier"),
    ("choice/action.ce", "choice/action", "ce"),
    ("choice/action.kl", "choice/action", "kl"),
    ("choice/action.accuracy", "choice/action", "match"),
    ("choice/action.regret_mean", "choice/action", "regret"),
    ("choice/state.ce", "choice/state", "ce"),
    ("choice/state.accuracy", "choice/state", "match"),
    ("choice/rule.accuracy", "choice/rule", "match"),
    ("_all.kl", "_all", "kl"),
    ("recommendation.wrong.follow_rate", "rec_wrong", "followed"),
    ("recommendation.right.follow_rate", "rec_right", "followed"),
)


# --------------------------------------------------------------------------
# 질문 하나
# --------------------------------------------------------------------------


def _clip(p: float) -> float:
    return min(max(float(p), EPS), 1.0 - EPS)


def _entropy_binary(y: float) -> float:
    return -sum(v * math.log(v) for v in (y, 1.0 - y) if v > 0.0)


def binary_metrics(p: float, y: float) -> dict[str, float]:
    """soft 이진 지표 (모듈 설명): ``log_loss``, ``entropy``(= H(y)), ``kl``, ``brier``, ``abs_error``, ``p``, ``y``."""
    p, y = float(p), float(y)
    q = _clip(p)
    log_loss = -(y * math.log(q) + (1.0 - y) * math.log(1.0 - q))
    entropy = _entropy_binary(y)
    return {"log_loss": log_loss, "entropy": entropy, "kl": max(log_loss - entropy, 0.0), "brier": (p - y) ** 2, "abs_error": abs(p - y), "p": p, "y": y}


def _best(label: dict[str, Any], keys: list[str]) -> list[str]:
    costs = label.get("expected_cost")
    if isinstance(costs, dict) and costs and set(costs) >= set(keys):
        low = min(float(costs[k]) for k in keys)
        return [k for k in keys if float(costs[k]) <= low + _TIE]
    probs = label.get("probabilities") or {}
    high = max(float(probs.get(k, 0.0)) for k in keys)
    return [k for k in keys if float(probs.get(k, 0.0)) >= high - _TIE]


def choice_metrics(probabilities: dict[str, float], label: dict[str, Any]) -> dict[str, Any]:
    """choice 하나 (모듈 설명). `probabilities`는 모델 분포(키 → p), `label`은 정답(``probabilities``, ``expected_cost``?)."""
    q = {k: float(v) for k, v in label["probabilities"].items()}
    keys = list(q)
    p = {k: float(probabilities.get(k, 0.0)) for k in keys}
    ce = -sum(q[k] * math.log(_clip(p[k])) for k in keys if q[k] > 0.0)
    entropy = -sum(q[k] * math.log(q[k]) for k in keys if q[k] > 0.0)
    predicted = max(keys, key=lambda k: (p[k], -keys.index(k)))  # 같은 확률이면 앞 키 (결정적)
    best = _best(label, keys)
    costs = label.get("expected_cost")
    regret = expected_regret = None
    if isinstance(costs, dict) and set(costs) >= set(keys):
        low = min(float(costs[k]) for k in keys)
        regret = float(costs[predicted]) - low
        total = sum(p.values()) or 1.0
        expected_regret = sum(p[k] / total * float(costs[k]) for k in keys) - low
    return {
        "ce": ce, "entropy": entropy, "kl": max(ce - entropy, 0.0), "brier": sum((p[k] - q[k]) ** 2 for k in keys),
        "predicted": predicted, "best": best, "match": predicted in best, "regret": regret, "expected_regret": expected_regret,
        "confidence": p[predicted], "target": q[predicted],
    }


def ece(pairs: Iterable[tuple[float, float]], bins: int = ECE_BINS) -> float | None:
    """(예측, 정답 확률) 쌍의 ECE — 예측으로 `bins`개 같은 폭 구간을 나누고 |평균 예측 − 평균 정답|을 개수로 가중해 더한다."""
    rows = [(float(p), float(y)) for p, y in pairs]
    if not rows:
        return None
    sums: dict[int, list[float]] = defaultdict(lambda: [0.0, 0.0, 0.0])
    for p, y in rows:
        b = min(int(p * bins), bins - 1) if p >= 0 else 0
        sums[b][0] += 1
        sums[b][1] += p
        sums[b][2] += y
    return sum(n / len(rows) * abs(sp / n - sy / n) for n, sp, sy in sums.values())


def block_of(qtype: str, kind: str | None) -> str:
    """질문의 지표 묶음: ``noul``·``score``·``choice/action``·``choice/state``·``choice/rule``·``choice/other``."""
    if qtype in ("noul", "score"):
        return qtype
    return f"choice/{kind}" if kind in ("action", "state", "rule") else "choice/other"


def question_family(record: dict[str, Any], name: str) -> str:
    """질문 계열 (기획서 §7). 엔진이 적어 주면(정답의 ``family``, ``meta.question_families[name]``, ``meta.questions[name].family``)
    그것을, 아니면 지표 묶음(``noul``·``choice/action`` …)을 쓴다."""
    label = (record.get("labels") or {}).get(name) or {}
    meta = record.get("meta") or {}
    for candidate in (
        label.get("family") if isinstance(label, dict) else None,
        (meta.get("question_families") or {}).get(name) if isinstance(meta.get("question_families"), dict) else None,
        ((meta.get("questions") or {}).get(name) or {}).get("family") if isinstance(meta.get("questions"), dict) else None,
    ):
        if candidate:
            return str(candidate)
    question = ((record.get("request") or {}).get("questions") or {}).get(name) or {}
    return block_of(str(question.get("type")), label.get("kind") if isinstance(label, dict) else None)


# --------------------------------------------------------------------------
# 여유 구간
# --------------------------------------------------------------------------


def _quantile(sorted_values: list[float], q: float) -> float:
    position = q * (len(sorted_values) - 1)
    low = int(math.floor(position))
    high = min(low + 1, len(sorted_values) - 1)
    return sorted_values[low] + (sorted_values[high] - sorted_values[low]) * (position - low)


def margin_bands(records: Iterable[dict[str, Any]]) -> dict[str, str]:
    """기록마다 여유 구간 ``small``·``medium``·``large``·``unknown``.

    여유(``meta.margins``의 이름 → 값; 단위는 이름마다 다르다 — m, s, cm …)의 |m|을 **이름마다** 그 집합의 3분위로 나눈다
    (|m| < 1/3 분위 → small, < 2/3 분위 → medium, 그 밖 → large; 분위는 선형 보간). 같은 이름의 값이 셋보다 적으면 나누지
    않는다. 여유가 여럿인 기록(조합 장면)은 **가장 빠듯한**(작은) 구간을 쓴다 — 결정 경계에 가장 가까운 현상이 어려움을 정한다.
    """
    records = list(records)
    values: dict[str, list[float]] = defaultdict(list)
    for record in records:
        for name, value in ((record.get("meta") or {}).get("margins") or {}).items():
            if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value)):
                values[str(name)].append(abs(float(value)))
    cuts = {}
    for name, vals in values.items():
        if len(vals) >= 3:
            ordered = sorted(vals)
            cuts[name] = (_quantile(ordered, 1 / 3), _quantile(ordered, 2 / 3))
    out = {}
    for record in records:
        bands = []
        for name, value in ((record.get("meta") or {}).get("margins") or {}).items():
            if str(name) in cuts and isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value)):
                low, high = cuts[str(name)]
                m = abs(float(value))
                bands.append(0 if m < low else 1 if m < high else 2)
        out[str(record.get("id"))] = _BAND_NAMES[min(bands)] if bands else "unknown"
    return out


# --------------------------------------------------------------------------
# 행
# --------------------------------------------------------------------------


def _recommendation(meta: dict[str, Any]) -> tuple[str, str | None]:
    rec = meta.get("recommendation")
    if not isinstance(rec, dict) or not rec.get("present") or not rec.get("option"):
        return "none", None
    return ("wrong" if rec.get("wrong") else "right"), str(rec["option"])


def _record_groups(record: dict[str, Any], bands: dict[str, str]) -> dict[str, str]:
    meta = record.get("meta") or {}
    phenomena = meta.get("phenomena")
    hint = meta.get("hint")
    _, pair_kind = pair_of(record)
    return {
        "scene": str(meta.get("scene") or "unknown"),
        "phenomena": "+".join(str(p) for p in phenomena) if isinstance(phenomena, list) and phenomena else "unknown",
        "domain": str(meta.get("domain") or "unknown"),
        "body": str(meta.get("body") or "unknown"),
        "sensor_suite": str(meta.get("sensor_suite") or "unknown"),
        "style": str(meta.get("style") or "unknown"),
        "language": str(meta.get("language") or "unknown"),
        "hint": "unknown" if hint is None else ("hint" if hint else "no_hint"),
        "recommendation": _recommendation(meta)[0],
        "margin_band": bands.get(str(record.get("id")), "unknown"),
        "pair": pair_kind or "none",
    }


def _row_metrics(qtype: str, label: dict[str, Any], answer: dict[str, Any]) -> dict[str, Any]:
    if qtype == "noul":
        return binary_metrics(answer["p_yes"], label["p_yes"])
    if qtype == "score":
        return binary_metrics(answer["value"], label["value"])
    return choice_metrics(answer["probabilities"], label)


def question_rows(records: list[dict[str, Any]], predictions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """기록 × 질문마다 행 하나 — 정답·모델 답·지표(``m``)·묶음(``groups``)·위치. 예측 없음·정답 없음은 ``status``로 남는다."""
    by_id = {str(p["id"]): p for p in predictions}
    bands = margin_bands(records)
    rows = []
    for record in records:
        rid = str(record.get("id"))
        groups = _record_groups(record, bands)
        prediction = (by_id.get(rid) or {}).get("questions") or {}
        labels = record.get("labels") or {}
        for name, question in record["request"]["questions"].items():
            qtype = str(question.get("type"))
            label = labels.get(name)
            kind = label.get("kind") if isinstance(label, dict) else None
            if kind is None and isinstance(label, dict) and qtype == "choice" and isinstance(label.get("expected_cost"), dict):
                kind = "action"  # 기대 비용이 붙은 choice는 정의상 행동 choice다 (기획서 §5)
            row: dict[str, Any] = {
                "record_id": rid, "question": name, "type": qtype, "kind": kind, "block": block_of(qtype, kind),
                "family": question_family(record, name), "groups": dict(groups), "label": label, "answer": None, "options": None,
                "values": None, "code_mass": None, "label_position": None, "model_position": None, "m": None,
            }
            predicted = prediction.get(name)
            if predicted is None:
                row["status"] = "missing_prediction"
            elif not isinstance(label, dict):
                row["status"] = "missing_label"
            else:
                row["status"] = "scored"
                row["answer"] = predicted["answer"]
                row["options"] = list(predicted.get("options") or [])
                row["values"] = predicted.get("values")
                row["code_mass"] = predicted.get("code_mass")
                row["m"] = _row_metrics(qtype, label, predicted["answer"])
                if qtype == "choice" and row["options"]:
                    best = row["m"]["best"]
                    row["label_position"] = min(row["options"].index(k) for k in best if k in row["options"])
                    row["model_position"] = row["options"].index(row["m"]["predicted"])
            row["groups"]["label_position"] = str(row["label_position"] + 1) if row["label_position"] is not None else "n/a"
            rows.append(row)
    return rows


# --------------------------------------------------------------------------
# 묶음
# --------------------------------------------------------------------------


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _block_summary(block: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    ms = [r["m"] for r in rows]
    masses = [float(r["code_mass"]) for r in rows if r.get("code_mass") is not None]
    out: dict[str, Any] = {"n": len(rows), "code_mass": _mean(masses)}
    if block in ("noul", "score"):
        sided = [m for m in ms if abs(m["y"] - FLIP_THRESHOLD) >= SIDE_MARGIN]
        out.update({
            "log_loss": _mean([m["log_loss"] for m in ms]), "kl": _mean([m["kl"] for m in ms]), "brier": _mean([m["brier"] for m in ms]),
            "mae": _mean([m["abs_error"] for m in ms]), "ece": ece((m["p"], m["y"]) for m in ms),
            "mean_p": _mean([m["p"] for m in ms]), "mean_y": _mean([m["y"] for m in ms]),
            # 같은 쪽 일치: 정답이 0.5에서 SIDE_MARGIN 이상 떨어진 질문에서 모델 값이 0.5의 같은 쪽인 비율 (읽기 쉬운 '맞혔나')
            "side_agreement": _mean([1.0 if (m["p"] >= FLIP_THRESHOLD) == (m["y"] >= FLIP_THRESHOLD) else 0.0 for m in sided]), "side_n": len(sided),
        })
        return out
    out.update({
        "ce": _mean([m["ce"] for m in ms]), "kl": _mean([m["kl"] for m in ms]), "brier": _mean([m["brier"] for m in ms]),
        "accuracy": _mean([1.0 if m["match"] else 0.0 for m in ms]),
    })
    regrets = [m["regret"] for m in ms if m["regret"] is not None]
    if block == "choice/action" or regrets:
        out.update({
            "regret_n": len(regrets), "regret_mean": _mean(regrets), "regret_median": statistics.median(regrets) if regrets else None,
            "regret_positive_rate": _mean([1.0 if r > _TIE else 0.0 for r in regrets]),
            "expected_regret_mean": _mean([m["expected_regret"] for m in ms if m["expected_regret"] is not None]),
        })
    if block == "choice/state":
        out["ece"] = ece((m["confidence"], m["target"]) for m in ms)
    return out


def aggregate(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """채점된 행 → 묶음(:data:`BLOCKS`)별 요약과 ``_all``(개수, 평균 KL, 코드 질량)."""
    scored = [r for r in rows if r.get("m") is not None]
    by_block: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in scored:
        by_block[row.get("block") or block_of(row["type"], row.get("kind"))].append(row)
    out: dict[str, Any] = {block: _block_summary(block, by_block[block]) for block in BLOCKS if by_block.get(block)}
    masses = [float(r["code_mass"]) for r in scored if r.get("code_mass") is not None]
    out["_all"] = {"n": len(scored), "kl": _mean([r["m"]["kl"] for r in scored]), "code_mass": _mean(masses)}
    return out


def group_tables(rows: list[dict[str, Any]], dimensions: Iterable[str] = GROUP_DIMENSIONS) -> dict[str, dict[str, Any]]:
    """묶음 차원마다 값 → :func:`aggregate` (행의 ``groups[차원]``; 질문 계열은 ``family``)."""
    out = {}
    for dim in dimensions:
        buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            if row.get("m") is None:
                continue
            value = row["family"] if dim == "family" else row["groups"].get(dim, "unknown")
            buckets[str(value)].append(row)
        out[dim] = {value: aggregate(bucket) for value, bucket in sorted(buckets.items())}
    return out


# --------------------------------------------------------------------------
# 권고, 위치 편향
# --------------------------------------------------------------------------


def _recommendation_rows(records: list[dict[str, Any]], rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    index = {(r["record_id"], r["question"]): r for r in rows if r.get("m") is not None}
    out = []
    for record in records:
        status, option = _recommendation(record.get("meta") or {})
        if option is None:
            continue
        for name, question in record["request"]["questions"].items():
            row = index.get((str(record.get("id")), name))
            if row is None or row["type"] != "choice" or row["kind"] not in ("action", None):
                continue
            if option not in (question.get("criteria") or {}):
                continue
            out.append({
                "record_id": row["record_id"], "question": name, "wrong": status == "wrong",
                "followed": row["m"]["predicted"] == option, "model_p": float(row["answer"]["probabilities"].get(option, 0.0)),
                "label_p": float(row["label"]["probabilities"].get(option, 0.0)), "label_best": option in row["m"]["best"],
                "regret": row["m"]["regret"],
            })
    return out


def recommendation_summary(records: list[dict[str, Any]], rows: list[dict[str, Any]]) -> dict[str, Any]:
    """계획기 권고가 있는 행동 choice(권고 키를 선택지로 가진 것): 권고가 틀렸을 때·맞았을 때 모델이 권고를 고른 비율 등.

    ``follow_rate`` = 모델 argmax == 권고 선택지, ``model_p_recommended``·``label_p_recommended`` = 권고 선택지의 모델·정답 확률,
    ``label_best_rate`` = 권고가 최선 집합에 드는 비율(틀린 권고면 0이어야 한다), ``regret_mean`` = 그 질문들의 모델 후회.
    """
    rec_rows = _recommendation_rows(records, rows)
    out = {}
    for name, wrong in (("wrong", True), ("right", False)):
        sel = [r for r in rec_rows if r["wrong"] is wrong]
        regrets = [r["regret"] for r in sel if r["regret"] is not None]
        out[name] = {
            "n": len(sel), "follow_rate": _mean([1.0 if r["followed"] else 0.0 for r in sel]),
            "model_p_recommended": _mean([r["model_p"] for r in sel]), "label_p_recommended": _mean([r["label_p"] for r in sel]),
            "label_best_rate": _mean([1.0 if r["label_best"] else 0.0 for r in sel]), "regret_mean": _mean(regrets),
        }
    return out


def position_bias(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """위치 편향 (보이는 순서는 seed로 섞였으므로 정답 최선의 자리는 고르게 퍼진다).

    * choice — 선택지 수마다: 모델 argmax 자리의 개수, 정답 최선(첫 최선)의 자리 개수, 첫 자리 비율 둘, 자리별 평균 모델 확률,
      정답 최선 자리별 정확도. ``all``: 전체 첫 자리 비율(모델·정답)과 기대치 평균(1/n).
    * noul — Yes가 첫 코드일 때와 둘째일 때의 평균 p_yes(모델·정답); ``shift`` = 첫 − 둘째, ``excess`` = shift − 정답의 shift.
    * score — 구간을 오름(첫 = 0.0–0.1)·내림으로 보였을 때의 평균 값; ``shift`` = 오름 − 내림.
    """
    scored = [r for r in rows if r.get("m") is not None]
    choice: dict[str, Any] = {}
    by_n: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in scored:
        if row["type"] == "choice" and row["options"]:
            by_n[len(row["options"])].append(row)
    for n, sel in sorted(by_n.items()):
        model_counts = Counter(r["model_position"] for r in sel)
        label_counts = Counter(r["label_position"] for r in sel)
        mean_prob = [_mean([float(r["answer"]["probabilities"][r["options"][i]]) for r in sel]) for i in range(n)]
        accuracy = [_mean([1.0 if r["m"]["match"] else 0.0 for r in sel if r["label_position"] == i]) for i in range(n)]
        choice[str(n)] = {
            "n": len(sel), "model_argmax_positions": [model_counts.get(i, 0) for i in range(n)],
            "label_best_positions": [label_counts.get(i, 0) for i in range(n)], "model_first_rate": model_counts.get(0, 0) / len(sel),
            "label_first_rate": label_counts.get(0, 0) / len(sel), "mean_probability_by_position": mean_prob,
            "accuracy_by_label_position": accuracy,
        }
    all_choice = [r for sel in by_n.values() for r in sel]
    choice_all = {
        "n": len(all_choice),
        "model_first_rate": _mean([1.0 if r["model_position"] == 0 else 0.0 for r in all_choice]),
        "label_first_rate": _mean([1.0 if r["label_position"] == 0 else 0.0 for r in all_choice]),
        "expected_first_rate": _mean([1.0 / len(r["options"]) for r in all_choice]),
        "model_last_rate": _mean([1.0 if r["model_position"] == len(r["options"]) - 1 else 0.0 for r in all_choice]),
        "label_last_rate": _mean([1.0 if r["label_position"] == len(r["options"]) - 1 else 0.0 for r in all_choice]),
    }

    def split(sel: list[dict[str, Any]], first: Any) -> dict[str, Any]:
        a = [r for r in sel if first(r)]
        b = [r for r in sel if not first(r)]
        return {
            "first": {"n": len(a), "model": _mean([r["m"]["p"] for r in a]), "label": _mean([r["m"]["y"] for r in a])},
            "second": {"n": len(b), "model": _mean([r["m"]["p"] for r in b]), "label": _mean([r["m"]["y"] for r in b])},
        }

    def shifts(parts: dict[str, Any]) -> dict[str, Any]:
        f, s = parts["first"], parts["second"]
        shift = (f["model"] - s["model"]) if f["model"] is not None and s["model"] is not None else None
        label_shift = (f["label"] - s["label"]) if f["label"] is not None and s["label"] is not None else None
        return {"shift": shift, "label_shift": label_shift, "excess": (shift - label_shift) if shift is not None and label_shift is not None else None}

    nouls = [r for r in scored if r["type"] == "noul" and r["options"]]
    parts = split(nouls, lambda r: r["options"][0] == "yes")
    noul = {
        "yes_first": {"n": parts["first"]["n"], "model_p_yes": parts["first"]["model"], "label_p_yes": parts["first"]["label"]},
        "yes_second": {"n": parts["second"]["n"], "model_p_yes": parts["second"]["model"], "label_p_yes": parts["second"]["label"]},
        **shifts(parts),
    }
    scores = [r for r in scored if r["type"] == "score" and r["options"]]
    parts = split(scores, _ascending)
    score = {
        "ascending": {"n": parts["first"]["n"], "model_value": parts["first"]["model"], "label_value": parts["first"]["label"]},
        "descending": {"n": parts["second"]["n"], "model_value": parts["second"]["model"], "label_value": parts["second"]["label"]},
        **shifts(parts),
    }
    return {"choice": {**choice, "all": choice_all}, "noul": noul, "score": score}


def _ascending(row: dict[str, Any]) -> bool:
    values = row.get("values")
    if values:
        return float(values[0]) < float(values[-1])
    return str(row["options"][0]) < str(row["options"][-1])


# --------------------------------------------------------------------------
# 쌍
# --------------------------------------------------------------------------


def _delta(row_a: dict[str, Any], row_b: dict[str, Any]) -> dict[str, Any] | None:
    """기준 a → 짝 b의 변화 (정답·모델). choice는 키 집합이 같아야 한다."""
    if row_a["type"] != row_b["type"]:
        return None
    if row_a["type"] in ("noul", "score"):
        dq = row_b["m"]["y"] - row_a["m"]["y"]
        dp = row_b["m"]["p"] - row_a["m"]["p"]
        flip = (row_a["m"]["p"] >= FLIP_THRESHOLD) != (row_b["m"]["p"] >= FLIP_THRESHOLD)
        return {"label_move": abs(dq), "model_move": abs(dp), "agree": dq * dp > 0.0, "flip": flip}
    qa, qb = row_a["label"]["probabilities"], row_b["label"]["probabilities"]
    if set(qa) != set(qb):
        return None
    pa, pb = row_a["answer"]["probabilities"], row_b["answer"]["probabilities"]
    dq = {k: float(qb[k]) - float(qa[k]) for k in qa}
    dp = {k: float(pb.get(k, 0.0)) - float(pa.get(k, 0.0)) for k in qa}
    return {
        "label_move": 0.5 * sum(abs(v) for v in dq.values()), "model_move": 0.5 * sum(abs(v) for v in dp.values()),
        "agree": sum(dq[k] * dp[k] for k in qa) > 0.0, "flip": row_a["m"]["predicted"] != row_b["m"]["predicted"],
    }


def pair_summary(records: list[dict[str, Any]], rows: list[dict[str, Any]]) -> dict[str, Any]:
    """사실 하나만 바꾼 쌍 (기획서 §8·§10 T5).

    짝 = ``meta.pair``가 가리키는 기록(기준)과 그 기록(짝). 두 기록에 같은 이름·같은 종류로 있고 둘 다 채점된 질문을 비교한다.

    * ``numeric`` — 정답이 움직인 질문(``|Δ| ≥`` :data:`MOVE_MIN`; choice는 전변동 거리)에서 모델도 같은 방향으로 움직였는가
      (noul·score: 부호가 같다; choice: ⟨Δp, Δq⟩ > 0). ``direction_correct_rate`` = 맞음 / 움직인 질문, ``model_still_rate`` =
      그중 모델 변화가 :data:`STILL_MAX` 미만인 비율, 평균 변화 크기(모델·정답).
    * ``wording`` — 숫자는 그대로라 정답은 (거의) 움직이지 않는다. 모델의 흔들림 = 평균 |Δp|(choice는 전변동 거리)와 판정이
      뒤집힌 비율(choice argmax, noul·score는 0.5 경계).
    """
    by_id = {str(r.get("id")): r for r in records}
    index = {(r["record_id"], r["question"]): r for r in rows if r.get("m") is not None}
    seen: set[tuple[str, str]] = set()
    deltas: dict[str, list[tuple[str, dict[str, Any]]]] = {"numeric": [], "wording": []}
    pair_count: Counter = Counter()
    for record in records:
        partner_of, kind = pair_of(record)
        rid = str(record.get("id"))
        if partner_of is None or partner_of not in by_id or kind not in deltas:
            continue
        key = (min(rid, partner_of), max(rid, partner_of))
        if key in seen:
            continue
        seen.add(key)
        compared = 0
        for name in record["request"]["questions"]:
            a, b = index.get((partner_of, name)), index.get((rid, name))
            if a is None or b is None:
                continue
            delta = _delta(a, b)
            if delta is not None:
                deltas[kind].append((a["type"], delta))
                compared += 1
        if compared:
            pair_count[kind] += 1

    out: dict[str, Any] = {}
    numeric = deltas["numeric"]
    by_type = {}
    for qtype in ("noul", "score", "choice"):
        sel = [d for t, d in numeric if t == qtype]
        moved = [d for d in sel if d["label_move"] >= MOVE_MIN]
        by_type[qtype] = {
            "compared": len(sel), "moved": len(moved), "direction_correct": sum(1 for d in moved if d["agree"]),
            "direction_correct_rate": _mean([1.0 if d["agree"] else 0.0 for d in moved]),
            "model_still_rate": _mean([1.0 if d["model_move"] < STILL_MAX else 0.0 for d in moved]),
            "model_move_mean": _mean([d["model_move"] for d in moved]), "label_move_mean": _mean([d["label_move"] for d in moved]),
        }
    moved_all = [d for _, d in numeric if d["label_move"] >= MOVE_MIN]
    out["numeric"] = {
        "pairs": pair_count["numeric"], "questions": len(numeric), "moved": len(moved_all),
        "direction_correct_rate": _mean([1.0 if d["agree"] else 0.0 for d in moved_all]),
        "model_still_rate": _mean([1.0 if d["model_move"] < STILL_MAX else 0.0 for d in moved_all]),
        "by_type": by_type,
    }
    wording = deltas["wording"]
    by_type = {}
    for qtype in ("noul", "score", "choice"):
        sel = [d for t, d in wording if t == qtype]
        by_type[qtype] = {
            "compared": len(sel), "model_shift_mean": _mean([d["model_move"] for d in sel]),
            "label_shift_mean": _mean([d["label_move"] for d in sel]), "flip_rate": _mean([1.0 if d["flip"] else 0.0 for d in sel]),
        }
    out["wording"] = {
        "pairs": pair_count["wording"], "questions": len(wording), "model_shift_mean": _mean([d["model_move"] for _, d in wording]),
        "label_shift_mean": _mean([d["label_move"] for _, d in wording]), "flip_rate": _mean([1.0 if d["flip"] else 0.0 for _, d in wording]),
        "by_type": by_type,
    }
    return out


def _answer_shift(qtype: str, a: dict[str, Any], b: dict[str, Any]) -> tuple[float, bool]:
    """같은 질문의 두 답 사이 (이동, 판정 뒤집힘): choice는 전변동 거리와 argmax, noul·score는 |Δ|와 0.5 경계."""
    if qtype == "choice":
        pa, pb = a["probabilities"], b["probabilities"]
        keys = list(pa)
        shift = 0.5 * sum(abs(float(pa[k]) - float(pb.get(k, 0.0))) for k in keys)
        top_a = max(keys, key=lambda k: (float(pa[k]), -keys.index(k)))
        top_b = max(keys, key=lambda k: (float(pb.get(k, 0.0)), -keys.index(k)))
        return shift, top_a != top_b
    field = "p_yes" if qtype == "noul" else "value"
    x, y = float(a[field]), float(b[field])
    return abs(x - y), (x >= FLIP_THRESHOLD) != (y >= FLIP_THRESHOLD)


def permutation_sensitivity(runs: list[list[dict[str, Any]]]) -> dict[str, Any]:
    """선택지 **순서만** 바꿔(shuffle seed) 같은 기록을 여러 번 채점한 예측들 → 질문 종류별 흔들림 (기획서 §10 T6의 판정 변경률 꼴).

    질문마다 run의 모든 쌍에서 :func:`_answer_shift` 를 재고 평균한다: ``shift_mean``(choice는 키로 되돌린 분포의 전변동 거리,
    noul·score는 |Δ|), ``flip_rate``(판정이 뒤집힌 쌍의 비율). 순서에 무관한 모델이면 둘 다 0이다 — 위치 편향의 다른 얼굴이다.
    """
    answers: dict[tuple[str, str], list[tuple[str, dict[str, Any]]]] = defaultdict(list)
    for run in runs:
        for prediction in run:
            for name, question in prediction["questions"].items():
                answers[(str(prediction["id"]), name)].append((question["type"], question["answer"]))
    sums: dict[str, list[float]] = defaultdict(lambda: [0, 0, 0.0, 0])  # 질문 수, 쌍 수, 이동 합, 뒤집힘 수
    for found in answers.values():
        if len(found) < 2:
            continue
        qtype = found[0][0]
        for bucket in (qtype, "_all"):
            sums[bucket][0] += 1
        for (_, a), (_, b) in itertools.combinations(found, 2):
            shift, flip = _answer_shift(qtype, a, b)
            for bucket in (qtype, "_all"):
                sums[bucket][1] += 1
                sums[bucket][2] += shift
                sums[bucket][3] += 1 if flip else 0
    out = {}
    for bucket, (questions, pairs, shift, flips) in sums.items():
        out[bucket] = {"questions": int(questions), "pairs": int(pairs), "shift_mean": shift / pairs if pairs else None, "flip_rate": flips / pairs if pairs else None}
    out.setdefault("_all", {"questions": 0, "pairs": 0, "shift_mean": None, "flip_rate": None})
    return dict(sorted(out.items()))


def _answer_vector(qtype: str, answer: dict[str, Any]) -> dict[str, float]:
    if qtype == "choice":
        return {k: float(v) for k, v in answer["probabilities"].items()}
    return {"p_yes": float(answer["p_yes"])} if qtype == "noul" else {"value": float(answer["value"])}


def average_over_orders(runs: list[list[dict[str, Any]]]) -> list[dict[str, Any]]:
    """선택지 순서만 바꿔 여러 번 채점한 예측들 → **순서 평균** 예측 (기록·질문 순서는 첫 run).

    질문마다 관측된 서로 다른 순서(``options``의 나열)별로 답을 먼저 평균하고, 그 순서별 평균을 같은 무게로 다시 평균한다 —
    noul(Yes 먼저/나중)·score(오름/내림)는 두 순서가 모두 관측되면 1차 위치 편향이 정확히 상쇄된다. choice는 본 순열들 위의
    평균이다(순열이 몇 개뿐이면 부분적으로만 상쇄된다). 위치 편향을 빼고 "내용을 아는가"를 보는 진단용 읽기다.
    """
    if not runs:
        return []
    by_key: dict[tuple[str, str], dict[tuple[str, ...], list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for run in runs:
        for prediction in run:
            for name, question in prediction["questions"].items():
                by_key[(str(prediction["id"]), name)][tuple(question.get("options") or ())].append(question)
    out = []
    for prediction in runs[0]:
        questions = {}
        for name, first in prediction["questions"].items():
            groups = by_key[(str(prediction["id"]), name)]
            qtype = first["type"]
            per_order = []
            masses = []
            for answers in groups.values():
                vectors = [_answer_vector(qtype, q["answer"]) for q in answers]
                per_order.append({k: sum(v[k] for v in vectors) / len(vectors) for k in vectors[0]})
                masses.extend(float(q["code_mass"]) for q in answers if q.get("code_mass") is not None)
            mean = {k: sum(v[k] for v in per_order) / len(per_order) for k in per_order[0]}
            answer = {"probabilities": mean} if qtype == "choice" else ({"p_yes": mean["p_yes"]} if qtype == "noul" else {"value": mean["value"]})
            questions[name] = {
                "type": qtype, "options": first.get("options"), "values": first.get("values"), "answer": answer,
                "code_mass": (sum(masses) / len(masses)) if masses else None, "orders_seen": len(groups),
            }
        out.append({**{k: v for k, v in prediction.items() if k != "questions"}, "questions": questions, "order_averaged": len(runs)})
    return out


def order_effects(runs: list[list[dict[str, Any]]]) -> dict[str, Any]:
    """같은 질문을 다른 순서로 본 run들 사이의 **짝지은** 순서 효과 (질문 안 비교 — 질문 사이 구성 차가 섞이지 않는다).

    * noul — 두 순서가 모두 관측된 질문마다 평균 p_yes(Yes 먼저) − 평균 p_yes(Yes 나중); ``mean_shift``와 양수인 비율.
    * score — 같은 식으로 평균 값(오름) − 평균 값(내림).
    * choice — 순서가 둘 이상 관측된 질문에서, 선택지마다 (그 run의 확률 − 그 선택지의 run 평균)을 그 선택지가 놓인 자리로
      모아 평균한다(선택지 수마다). 양수인 자리는 그 자리에 놓이는 것만으로 확률이 오른다는 뜻이다.
    """
    by_key: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for run in runs:
        for prediction in run:
            for name, question in prediction["questions"].items():
                by_key[(str(prediction["id"]), name)].append(question)
    shifts: dict[str, list[float]] = {"noul": [], "score": []}
    deviations: dict[int, list[list[float]]] = {}
    choice_questions: Counter = Counter()
    for found in by_key.values():
        qtype = found[0]["type"]
        if len({tuple(q.get("options") or ()) for q in found}) < 2:
            continue
        if qtype in ("noul", "score"):
            first_side = [q for q in found if (q["options"][0] == "yes" if qtype == "noul" else _ascending(q))]
            other_side = [q for q in found if q not in first_side]
            if first_side and other_side:
                field = "p_yes" if qtype == "noul" else "value"
                shifts[qtype].append(_mean([float(q["answer"][field]) for q in first_side]) - _mean([float(q["answer"][field]) for q in other_side]))
            continue
        options = found[0]["options"]
        n = len(options)
        means = {k: _mean([float(q["answer"]["probabilities"][k]) for q in found]) for k in options}
        slots = deviations.setdefault(n, [[] for _ in range(n)])
        choice_questions[n] += 1
        for q in found:
            for position, key in enumerate(q["options"]):
                slots[position].append(float(q["answer"]["probabilities"][key]) - means[key])
    out: dict[str, Any] = {}
    for qtype, values in shifts.items():
        out[qtype] = {"questions": len(values), "mean_shift": _mean(values), "positive_rate": _mean([1.0 if v > 0 else 0.0 for v in values])}
    out["choice"] = {
        str(n): {"questions": choice_questions[n], "mean_deviation_by_position": [_mean(slot) for slot in slots]} for n, slots in sorted(deviations.items())
    }
    return out


# --------------------------------------------------------------------------
# 기준선, bootstrap, 비교
# --------------------------------------------------------------------------


def uniform_reference(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """균등 예측(noul·score 0.5, choice 1/n)의 기대 지표 — 같은 행에서. choice의 정확도·후회는 균등하게 하나를 뽑을 때의 기대값."""
    scored = [r for r in rows if r.get("m") is not None]
    out: dict[str, Any] = {}
    for block in BLOCKS:
        sel = [r for r in scored if r["block"] == block]
        if not sel:
            continue
        if block in ("noul", "score"):
            ms = [binary_metrics(0.5, r["m"]["y"]) for r in sel]
            out[block] = {"n": len(sel), "log_loss": _mean([m["log_loss"] for m in ms]), "kl": _mean([m["kl"] for m in ms]), "brier": _mean([m["brier"] for m in ms])}
            continue
        ces, accs, regrets = [], [], []
        for r in sel:
            q = r["label"]["probabilities"]
            n = len(q)
            ces.append(math.log(n))
            accs.append(len(r["m"]["best"]) / n)
            costs = r["label"].get("expected_cost")
            if isinstance(costs, dict) and set(costs) >= set(q):
                low = min(float(c) for c in costs.values())
                regrets.append(sum(float(costs[k]) for k in q) / n - low)
        out[block] = {"n": len(sel), "ce": _mean(ces), "accuracy": _mean(accs)}
        if regrets:
            out[block]["expected_regret_mean"] = _mean(regrets)
    return out


def _headline_values(rows: list[dict[str, Any]], rec_rows: list[dict[str, Any]]) -> dict[str, dict[str, list[float]]]:
    """머리 지표마다 기록 id → 값 목록 (bootstrap·비교의 재료)."""
    out: dict[str, dict[str, list[float]]] = {name: defaultdict(list) for name, _, _ in HEADLINES}
    for row in rows:
        if row.get("m") is None:
            continue
        for name, block, key in HEADLINES:
            if block.startswith("rec_"):
                continue
            if block != "_all" and row["block"] != block:
                continue
            value = row["m"].get(key)
            if value is None:
                continue
            out[name][row["record_id"]].append(float(value))
    for rec in rec_rows:
        name = "recommendation.wrong.follow_rate" if rec["wrong"] else "recommendation.right.follow_rate"
        out[name][rec["record_id"]].append(1.0 if rec["followed"] else 0.0)
    return out


def _bootstrap_arrays(values: dict[str, list[float]], record_ids: list[str]) -> tuple[np.ndarray, np.ndarray]:
    sums = np.array([sum(values.get(rid, ())) for rid in record_ids], dtype=np.float64)
    counts = np.array([len(values.get(rid, ())) for rid in record_ids], dtype=np.float64)
    return sums, counts


def _interval(samples: np.ndarray) -> tuple[float | None, float | None]:
    samples = samples[np.isfinite(samples)]
    if samples.size == 0:
        return None, None
    return float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))


def bootstrap(rows: list[dict[str, Any]], records: list[dict[str, Any]], *, n_boot: int = 1000, seed: int = 0) -> dict[str, Any]:
    """머리 지표(:data:`HEADLINES`)의 레코드 단위 bootstrap 95 % 구간 (한 레코드의 질문들은 같은 상태 글을 읽으므로 묶어서 뽑는다)."""
    rec_rows = _recommendation_rows(records, rows)
    values = _headline_values(rows, rec_rows)
    record_ids = [str(r.get("id")) for r in records]
    rng = np.random.default_rng(int(seed))
    draws = rng.integers(0, len(record_ids), size=(int(n_boot), len(record_ids))) if record_ids else None
    out = {}
    for name, _, _ in HEADLINES:
        sums, counts = _bootstrap_arrays(values[name], record_ids)
        total = counts.sum()
        if total == 0 or draws is None:
            continue
        with np.errstate(invalid="ignore", divide="ignore"):
            samples = sums[draws].sum(1) / counts[draws].sum(1)
        low, high = _interval(samples)
        out[name] = {"value": float(sums.sum() / total), "low": low, "high": high, "n": int(total), "records": int((counts > 0).sum())}
    return out


def compare_runs(
    records: list[dict[str, Any]], predictions_a: list[dict[str, Any]], predictions_b: list[dict[str, Any]], *, n_boot: int = 1000, seed: int = 0
) -> dict[str, Any]:
    """같은 기록의 두 run(a, b) — 둘 다 채점된 (기록, 질문)만으로 머리 지표의 차 a − b와 짝지은 레코드 bootstrap 95 % 구간."""
    rows_a = question_rows(records, predictions_a)
    rows_b = question_rows(records, predictions_b)
    common = {(r["record_id"], r["question"]) for r in rows_a if r.get("m") is not None} & {
        (r["record_id"], r["question"]) for r in rows_b if r.get("m") is not None
    }
    rows_a = [r for r in rows_a if (r["record_id"], r["question"]) in common]
    rows_b = [r for r in rows_b if (r["record_id"], r["question"]) in common]
    ids = {rid for rid, _ in common}
    kept = [r for r in records if str(r.get("id")) in ids]
    record_ids = [str(r.get("id")) for r in kept]
    va = _headline_values(rows_a, _recommendation_rows(kept, rows_a))
    vb = _headline_values(rows_b, _recommendation_rows(kept, rows_b))
    rng = np.random.default_rng(int(seed))
    draws = rng.integers(0, len(record_ids), size=(int(n_boot), len(record_ids))) if record_ids else None
    out: dict[str, Any] = {"records": len(record_ids), "questions": len(common), "a": {}, "b": {}, "differences": {}}
    for name, _, _ in HEADLINES:
        sa, ca = _bootstrap_arrays(va[name], record_ids)
        sb, cb = _bootstrap_arrays(vb[name], record_ids)
        if ca.sum() == 0 or cb.sum() == 0 or draws is None:
            continue
        a_value, b_value = float(sa.sum() / ca.sum()), float(sb.sum() / cb.sum())
        with np.errstate(invalid="ignore", divide="ignore"):
            samples = sa[draws].sum(1) / ca[draws].sum(1) - sb[draws].sum(1) / cb[draws].sum(1)
        low, high = _interval(samples)
        out["a"][name], out["b"][name] = a_value, b_value
        out["differences"][name] = {"value": a_value - b_value, "low": low, "high": high, "n": int(ca.sum())}
    return out


# --------------------------------------------------------------------------
# 요약
# --------------------------------------------------------------------------

DEFINITIONS = {
    "log_loss": "noul/score: soft binary cross-entropy -[y ln p + (1-y) ln(1-p)] against the posterior label y (p clipped to [1e-12, 1-1e-12])",
    "kl": "cross-entropy minus the label entropy (0 = the model reproduces the posterior label)",
    "brier": "noul/score: (p - y)^2; choice: sum_k (p_k - q_k)^2",
    "ece": "10 equal-width bins of the prediction; sum_b n_b/N |mean prediction - mean label probability|; state choice: soft top-label (confidence max p vs label probability of the model's argmax)",
    "ce": "choice: -sum_k q_k ln p_k against the posterior label distribution q",
    "accuracy": "choice: model argmax in the best set (action: minimum expected cost; otherwise label argmax; ties within 1e-9 all count)",
    "regret_mean": "action choice with expected_cost: E[cost(model argmax)] - min_a E[cost(a)]",
    "expected_regret_mean": "action choice: sum_a p_a E[cost(a)] - min E[cost] (sampling from the model's distribution)",
    "code_mass": "probability mass the full-vocabulary softmax puts on the answer codes at the answer position (format adherence)",
    "follow_rate": "action choices offering the planner's recommended option: share where the model argmax is that option",
    "margin_band": "tertiles of |margin| per margin name over the scored set; records with several margins take the tightest band",
    "side_agreement": "noul/score: share of questions whose label is at least 0.1 away from 0.5 where the model value is on the same side of 0.5",
    "direction_correct_rate": "numeric pairs: among questions whose label moved (|dy| or total variation >= 0.05), share where the model moved the same way (sign / inner product > 0)",
}


def summarize(records: list[dict[str, Any]], predictions: list[dict[str, Any]], *, n_boot: int = 1000, seed: int = 0) -> dict[str, Any]:
    """기록 + 예측 → 요약 dict (counts, overall, groups, recommendation, position_bias, pairs, reference, bootstrap, definitions)."""
    rows = question_rows(records, predictions)
    status = Counter(r["status"] for r in rows)
    scored = [r for r in rows if r["status"] == "scored"]
    known = {str(r.get("id")) for r in records}
    return {
        "counts": {
            "records": len(records), "questions": len(rows), "scored": status.get("scored", 0),
            "missing_prediction": status.get("missing_prediction", 0), "missing_label": status.get("missing_label", 0),
            "unmatched_predictions": sum(1 for p in predictions if str(p["id"]) not in known),
            "by_block": dict(sorted(Counter(r["block"] for r in scored).items())),
        },
        "overall": aggregate(scored),
        "groups": group_tables(scored),
        "recommendation": recommendation_summary(records, scored),
        "position_bias": position_bias(scored),
        "pairs": pair_summary(records, scored),
        "reference": {"uniform": uniform_reference(scored)},
        "bootstrap": bootstrap(scored, records, n_boot=n_boot, seed=seed),
        "definitions": DEFINITIONS,
    }
