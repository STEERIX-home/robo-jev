"""모델 답 대 사후 확률 정답의 지표 (Task v0c §2.3; 기획서 docs/12 §5·§10; fix round 1 — 리뷰 1 I1–I4와 통제자 결정).

정답은 확률이다(관측이 주어졌을 때 가능한 세계들에서 결과가 나온 빈도, 기획서 §5). 그래서 지표는 모두 **정답 확률과의
거리**로 잰다 — 참 세계의 0/1 결과(``truth.realized``)는 쓰지 않는다.

**배치와 두 읽기.** 예측은 질문마다 보이는 배치(listing) 여럿을 가질 수 있다(결정적 균형 배치 — :mod:`robo_jev.decisions.prompt`;
옛 seed run 여럿은 :func:`merge_runs` 로 같은 꼴이 된다). 두 읽기(:data:`READINGS`):

* ``single_order`` — **한 순서 읽기(배포 조건)**: 배치마다 행 하나, 무게 1/L(L = 그 질문의 서로 다른 배치 수). 지표는 배치별 지표의
  평균 = 한 순서로 물었을 때의 기대 성능(seed 하나의 운이 아니다).
* ``order_averaged`` — **순서 평균 읽기("무엇을 아는가")**: 질문마다 행 하나, 답은 서로 다른 배치의 답을 같은 무게로 평균한 것.
  균형 배치에서는 1차 위치 편향이 세 종류 모두 정확히 상쇄된다.

질문 하나의 지표 (행 = 기록 × 질문 [× 배치]; 묶음 열쇠 = :func:`block_of`):

* ``noul``·``score`` — y = 정답, p = 모델의 값. 로그 손실 −[y ln p + (1−y) ln(1−p)] (p는 [EPS, 1−EPS]로 자름), KL = 로그 손실 − H(y),
  Brier (p − y)², |p − y|. ECE는 10구간(p 기준 같은 폭)의 |평균 p − 평균 y| 가중합.
* ``choice`` — q = 정답 분포, p = 모델 분포. 교차 엔트로피, KL, Brier Σ(p−q)². **최선 집합** = 기대 비용 최소(행동) 또는 정답 확률
  최대(같은 값 1e-9 안은 모두). 일치 = 모델 argmax ∈ 최선 집합. **기대 비용 후회** = E[c(모델 argmax)] − min E[c]; **척도 없는 후회**
  ``regret_norm`` = 후회 ÷ 그 질문의 기대 비용 범위(max − min; 0이면 0) — 기준이 적힌 질문(``avoid_*``)의 비용은 위반이 ≈ 1,000이라
  기본 기준과 척도가 다르다(리뷰 1 I3); 표본 후회 ``sampled_regret`` = Σ p_a E[c_a] − min E[c](모델 분포에서 뽑아 실행했을 때).

**기준선과 짝지은 차** (리뷰 1 I1): 행마다 같은 질문에서의 기준선 값 ``u`` — noul·score는 "언제나 0.5", choice는 "무작위로 하나
고르기"(정확도 |최선|/n, 후회 = 평균 비용 − 최소 비용, CE = ln n). :func:`bootstrap` 과 :func:`group_signal` 은 모델 − 기준선을
**같은 레코드 뽑기**로 잰 짝지은 95 % 구간을 낸다(한 레코드의 질문은 같은 상태 글을 읽으므로 레코드 단위로 뽑는다).

묶음(:func:`question_rows` 의 ``groups``): 장면, 현상, 분야, 몸, 감각 구성, 문체, 언어, 힌트, 권고(없음·맞음·틀림), 행동 기준
(``criterion``: default·time_critical·avoid_* …), 어려움 축(:func:`margin_norm_band` — v0a의 ``meta.margin_norm``이 있을 때,
:func:`entropy_band` — 핵심 질문과 그 질문 자체의 정답 엔트로피), 옛 여유 3분위, 질문 계열, 쌍, 최선 선택지 자리, 키 문체, 상대, 관계없는
사실 수, 거리 단위.

따로 보는 것: 기준 계열별 후회(:func:`regret_by_criterion`), 묶음별 신호(:func:`group_signal` — 상관과 기준선 대비 짝지은 차),
권고(:func:`recommendation_summary`), 위치 편향(:func:`position_bias`), 순서 바꿈(:func:`permutation_sensitivity` — **서로 다른**
배치의 짝만, :func:`order_effects`), 쌍(:func:`pair_summary` — 순서 평균 답으로), 생각 뒤 답의 완료율 문턱(:func:`reasoning_stats`),
두 run 비교(:func:`compare_runs`).
"""

from __future__ import annotations

import itertools
import math
import statistics
from collections import Counter, defaultdict
from typing import Any, Callable, Iterable

import numpy as np

from robo_jev.decisions.records import pair_of

__all__ = [
    "BLOCKS",
    "COMPLETION_GATE",
    "DECISIVE_MARGIN",
    "ECE_BINS",
    "ENTROPY_CUTS",
    "EPS",
    "FLIP_THRESHOLD",
    "GROUP_DIMENSIONS",
    "HEADLINES",
    "MOVE_MIN",
    "NORM_CUTS",
    "READINGS",
    "SCORE_EDGE_RANGE",
    "SIGNAL_DIMENSIONS",
    "aggregate",
    "average_answers",
    "binary_metrics",
    "block_of",
    "bootstrap",
    "choice_metrics",
    "compare_runs",
    "criterion_of",
    "ece",
    "entropy_band",
    "group_signal",
    "group_tables",
    "key_entropy",
    "listings_of",
    "margin_bands",
    "margin_norm_band",
    "merge_runs",
    "normalized_entropy",
    "order_effects",
    "pair_deltas",
    "pair_summary",
    "permutation_sensitivity",
    "position_bias",
    "question_family",
    "question_rows",
    "reasoning_stats",
    "recommendation_summary",
    "regret_by_criterion",
    "score_label_range",
    "summarize",
    "summarize_all",
    "uniform_metrics",
    "uniform_reference",
]

EPS = 1e-12
ECE_BINS = 10
#: 쌍에서 정답이 "움직였다"고 보는 최소 변화 — noul·score는 |Δ|, choice는 전변동 거리 ½Σ|Δq|.
MOVE_MIN = 0.05
#: 모델이 "움직이지 않았다"고 보는 변화 (숫자 쌍의 still 비율).
STILL_MAX = 0.01
#: noul·score의 "판정 뒤집힘" 문턱 — 판정 = 값 ``>= 0.5`` (dev와 진단이 같은 규칙; Task v0d fix round 1).
FLIP_THRESHOLD = 0.5
#: "결정적인 바뀜"의 여유 (통제자 결정 6): 두 답이 모두 0.5에서 이만큼 이상 떨어진(choice는 1등이 2등을 2배 이상 앞선) 채 판정이 바뀐 것만.
DECISIVE_MARGIN = 0.1
#: v0c-p2의 score 읽기가 닿는 범위 — 정답이 이 밖인 score 질문을 따로 묶는다(``score_label_range``; 통제자 결정 1).
SCORE_EDGE_RANGE = (0.05, 0.95)
#: 같은 쪽 일치(side agreement)에서 빼는 애매한 정답의 폭: |y − 0.5| < 이 값이면 세지 않는다.
SIDE_MARGIN = 0.1
_TIE = 1e-9
BLOCKS = ("noul", "score", "choice/action", "choice/state", "choice/rule", "choice/other")
READINGS = ("single_order", "order_averaged")
#: 생각(설명) 뒤 답의 완료율 문턱 — 미만이면 그 run의 생각 대 바로 답 비교는 판정 불가 (통제자 결정, fix round 1).
COMPLETION_GATE = 0.9
#: |margin_norm|(= 여유 / σ_m, v0a) 구간: near < 0.5 ≤ mid < 1.5 ≤ far. v0a 문서: |값| < 1이 경계 가까이.
NORM_CUTS = (0.5, 1.5)
#: 정규화한 정답 엔트로피(비트, [0, 1]) 구간: low < 1/3 ≤ mid < 2/3 ≤ high (high = 정답이 0.5 가까이 = 경계).
ENTROPY_CUTS = (1 / 3, 2 / 3)
#: 후회의 쏠림을 보는 상위 질문 수 (리뷰 1 I3: 상위 10개가 2B 후회 합의 47.5 %).
TOP_K = 10
GROUP_DIMENSIONS = (
    "scene", "phenomena", "domain", "body", "sensor_suite", "style", "language", "hint", "recommendation", "criterion",
    "margin_norm_band", "key_entropy_band", "label_entropy_band", "margin_band", "family", "pair", "label_position",
    "key_style", "agent", "distractors", "distance_unit", "score_label_range",
)
#: 기준선 대비 짝지은 구간을 내는 묶음 (보고서의 신호·어려움 표).
SIGNAL_DIMENSIONS = (
    "scene", "phenomena", "margin_norm_band", "key_entropy_band", "label_entropy_band", "margin_band", "criterion", "recommendation",
    "hint", "style", "score_label_range",
)
_BAND_NAMES = ("small", "medium", "large")
#: bootstrap·run 비교의 머리 지표 — (이름, 행 고르기, 값, 기준 계열 거르개). 이름은 ``<묶음>.<지표>[계열]``. 기준선(``u``)에 같은 값이 있으면
#: 모델 − 기준선의 짝지은 구간도 낸다.
HEADLINES: tuple[tuple[str, str, str, str | None], ...] = (
    ("noul.log_loss", "noul", "log_loss", None),
    ("noul.kl", "noul", "kl", None),
    ("noul.brier", "noul", "brier", None),
    ("score.log_loss", "score", "log_loss", None),
    ("score.kl", "score", "kl", None),
    ("score.brier", "score", "brier", None),
    ("choice/action.ce", "choice/action", "ce", None),
    ("choice/action.kl", "choice/action", "kl", None),
    ("choice/action.accuracy", "choice/action", "match", None),
    ("choice/action.regret_mean", "choice/action", "regret", None),
    ("choice/action.regret_mean[default]", "choice/action", "regret", "default"),
    ("choice/action.regret_mean[stated]", "choice/action", "regret", "stated"),
    ("choice/action.regret_norm", "choice/action", "regret_norm", None),
    ("choice/action.regret_norm[default]", "choice/action", "regret_norm", "default"),
    ("choice/action.regret_norm[stated]", "choice/action", "regret_norm", "stated"),
    ("score.log_loss[label_outside]", "score", "log_loss", "label_outside"),
    ("score.log_loss[label_inside]", "score", "log_loss", "label_inside"),
    ("score.brier[label_outside]", "score", "brier", "label_outside"),
    ("score.brier[label_inside]", "score", "brier", "label_inside"),
    ("score.mae[label_outside]", "score", "abs_error", "label_outside"),
    ("score.mae[label_inside]", "score", "abs_error", "label_inside"),
    ("choice/state.ce", "choice/state", "ce", None),
    ("choice/state.accuracy", "choice/state", "match", None),
    ("choice/rule.accuracy", "choice/rule", "match", None),
    ("_all.kl", "_all", "kl", None),
    ("recommendation.wrong.follow_rate", "rec_wrong", "followed", None),
    ("recommendation.right.follow_rate", "rec_right", "followed", None),
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


def _costs(label: dict[str, Any], keys: list[str]) -> dict[str, float] | None:
    costs = label.get("expected_cost")
    if isinstance(costs, dict) and costs and set(costs) >= set(keys):
        return {k: float(costs[k]) for k in keys}
    return None


def _best(label: dict[str, Any], keys: list[str]) -> list[str]:
    costs = _costs(label, keys)
    if costs is not None:
        low = min(costs.values())
        return [k for k in keys if costs[k] <= low + _TIE]
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
    costs = _costs(label, keys)
    regret = sampled = norm = None
    if costs is not None:
        low, high = min(costs.values()), max(costs.values())
        regret = costs[predicted] - low
        total = sum(p.values()) or 1.0
        sampled = sum(p[k] / total * costs[k] for k in keys) - low
        norm = regret / (high - low) if high - low > 0 else 0.0
    return {
        "ce": ce, "entropy": entropy, "kl": max(ce - entropy, 0.0), "brier": sum((p[k] - q[k]) ** 2 for k in keys),
        "predicted": predicted, "best": best, "match": predicted in best, "regret": regret, "sampled_regret": sampled, "regret_norm": norm,
        "confidence": p[predicted], "target": q[predicted],
    }


def uniform_metrics(qtype: str, label: dict[str, Any], m: dict[str, Any]) -> dict[str, float]:
    """같은 질문에서의 기준선 값: noul·score = 언제나 0.5, choice = 무작위로 하나 고르기(분포 1/n; 정확도·후회는 그 기대값)."""
    if qtype in ("noul", "score"):
        u = binary_metrics(0.5, m["y"])
        return {"log_loss": u["log_loss"], "kl": u["kl"], "brier": u["brier"], "abs_error": u["abs_error"]}
    q = {k: float(v) for k, v in label["probabilities"].items()}
    keys = list(q)
    n = len(keys)
    out = {"ce": math.log(n), "kl": max(math.log(n) - m["entropy"], 0.0), "brier": sum((1.0 / n - q[k]) ** 2 for k in keys), "match": len(m["best"]) / n}
    costs = _costs(label, keys)
    if costs is not None:
        low, high = min(costs.values()), max(costs.values())
        mean = sum(costs.values()) / n
        out["regret"] = mean - low
        out["sampled_regret"] = mean - low
        out["regret_norm"] = (mean - low) / (high - low) if high - low > 0 else 0.0
    return out


def ece(pairs: Iterable[tuple[float, ...]], bins: int = ECE_BINS) -> float | None:
    """(예측, 정답 확률[, 무게]) → ECE: 예측으로 `bins`개 같은 폭 구간을 나누고 |평균 예측 − 평균 정답|을 무게로 가중해 더한다."""
    sums: dict[int, list[float]] = defaultdict(lambda: [0.0, 0.0, 0.0])
    total = 0.0
    for item in pairs:
        p, y = float(item[0]), float(item[1])
        w = float(item[2]) if len(item) > 2 else 1.0
        b = min(int(p * bins), bins - 1) if p >= 0 else 0
        sums[b][0] += w
        sums[b][1] += w * p
        sums[b][2] += w * y
        total += w
    if total <= 0:
        return None
    return sum(n / total * abs(sp / n - sy / n) for n, sp, sy in sums.values() if n > 0)


def block_of(qtype: str, kind: str | None) -> str:
    """질문의 지표 묶음: ``noul``·``score``·``choice/action``·``choice/state``·``choice/rule``·``choice/other``."""
    if qtype in ("noul", "score"):
        return qtype
    return f"choice/{kind}" if kind in ("action", "state", "rule") else "choice/other"


def _family_from_basis(basis: Any) -> str | None:
    """v0a의 ``meta.basis[이름]``(질문이 무엇에서 나왔나) → 계열 이름: ``action:<기준|default>``, ``noul:<사건>``,
    ``score:<양|사건>``, ``state:<상태>``, ``rule``."""
    if not isinstance(basis, dict) or not basis.get("kind"):
        return None
    kind = str(basis["kind"])
    if kind == "action":
        return f"action:{basis.get('criterion') or 'default'}"
    if kind == "rule":
        return "rule"
    tag = basis.get("event") or basis.get("name") or basis.get("state")
    return f"{kind}:{tag}" if tag else kind


def question_family(record: dict[str, Any], name: str) -> str:
    """질문 계열 (기획서 §7). 엔진이 적어 주면(정답의 ``family``, ``meta.question_families[name]``, ``meta.questions[name].family``,
    v0a의 ``meta.basis[name]`` — :func:`_family_from_basis`) 그것을, 아니면 지표 묶음(``noul``·``choice/action`` …)을 쓴다."""
    label = (record.get("labels") or {}).get(name) or {}
    meta = record.get("meta") or {}
    for candidate in (
        label.get("family") if isinstance(label, dict) else None,
        (meta.get("question_families") or {}).get(name) if isinstance(meta.get("question_families"), dict) else None,
        ((meta.get("questions") or {}).get(name) or {}).get("family") if isinstance(meta.get("questions"), dict) else None,
        _family_from_basis((meta.get("basis") or {}).get(name)) if isinstance(meta.get("basis"), dict) else None,
    ):
        if candidate:
            return str(candidate)
    question = ((record.get("request") or {}).get("questions") or {}).get(name) or {}
    return block_of(str(question.get("type")), label.get("kind") if isinstance(label, dict) else None)


def criterion_of(record: dict[str, Any], name: str) -> str:
    """행동 질문의 기준 계열: v0a의 ``meta.basis[name].criterion``(없으면 ``default``) 또는 ``meta.criteria[name]``; 둘 다 없으면 ``unknown``."""
    meta = record.get("meta") or {}
    basis = (meta.get("basis") or {}).get(name) if isinstance(meta.get("basis"), dict) else None
    if isinstance(basis, dict) and basis.get("kind") == "action":
        return str(basis.get("criterion") or "default")
    criteria = meta.get("criteria")
    if isinstance(criteria, dict) and name in criteria:
        return str(criteria[name] or "default")
    return "unknown"


def _criterion_class(criterion: str) -> str:
    """``default`` · ``stated``(기준이 적힌 질문 — time_critical·avoid_* …) · ``unknown``."""
    return criterion if criterion in ("default", "unknown") else "stated"


# --------------------------------------------------------------------------
# 어려움 축
# --------------------------------------------------------------------------


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def margin_norm_band(meta: dict[str, Any]) -> str:
    """v0a의 ``meta.margin_norm``(여유 이름 → 여유 / σ_m, 또는 수 하나)에서 가장 빠듯한 |z| → ``near``·``mid``·``far``; 없으면 ``absent``."""
    raw = meta.get("margin_norm")
    values = list(raw.values()) if isinstance(raw, dict) else [raw]
    finite = [abs(float(v)) for v in values if _finite(v)]
    if not finite:
        return "absent"
    z = min(finite)
    return "near" if z < NORM_CUTS[0] else "mid" if z < NORM_CUTS[1] else "far"


def normalized_entropy(qtype: str, label: dict[str, Any] | None) -> float | None:
    """정답 엔트로피를 [0, 1]로: noul·score는 이진 엔트로피(비트) — 정답이 0.5면 1, 0·1이면 0; choice는 H(q) / ln n."""
    if not isinstance(label, dict):
        return None
    if qtype in ("noul", "score"):
        y = label.get("p_yes" if qtype == "noul" else "value")
        return _entropy_binary(float(y)) / math.log(2) if _finite(y) else None
    probs = [float(v) for v in (label.get("probabilities") or {}).values()]
    if len(probs) < 2:
        return None
    return -sum(v * math.log(v) for v in probs if v > 0.0) / math.log(len(probs))


def entropy_band(h: float | None) -> str:
    """정규화한 엔트로피 → ``low``(정답이 한쪽으로 분명) · ``mid`` · ``high``(경계 — 정답이 0.5 가까이); 없으면 ``absent``."""
    if h is None:
        return "absent"
    return "low" if h < ENTROPY_CUTS[0] else "mid" if h < ENTROPY_CUTS[1] else "high"


def key_entropy(record: dict[str, Any]) -> float | None:
    """기록의 핵심 질문(v0a ``meta.key_question`` — 장면의 결정 여유를 묻는 noul) 정답의 정규화 엔트로피 — 기록 단위의 어려움."""
    name = (record.get("meta") or {}).get("key_question")
    question = (((record.get("request") or {}).get("questions") or {}).get(name)) if name else None
    if not isinstance(question, dict):
        return None
    return normalized_entropy(str(question.get("type")), (record.get("labels") or {}).get(name))


def _quantile(sorted_values: list[float], q: float) -> float:
    position = q * (len(sorted_values) - 1)
    low = int(math.floor(position))
    high = min(low + 1, len(sorted_values) - 1)
    return sorted_values[low] + (sorted_values[high] - sorted_values[low]) * (position - low)


def margin_bands(records: Iterable[dict[str, Any]]) -> dict[str, str]:
    """(옛 축) 기록마다 여유 3분위 구간 ``small``·``medium``·``large``·``unknown`` — |m|을 여유 이름마다 그 집합의 3분위로 나누고 가장
    빠듯한 구간을 쓴다. 단위가 이름마다 달라 장면 사이에 견줄 수 없다 — 어려움 축은 :func:`margin_norm_band` 와 정답 엔트로피다."""
    records = list(records)
    values: dict[str, list[float]] = defaultdict(list)
    for record in records:
        for name, value in ((record.get("meta") or {}).get("margins") or {}).items():
            if _finite(value):
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
            if str(name) in cuts and _finite(value):
                low, high = cuts[str(name)]
                m = abs(float(value))
                bands.append(0 if m < low else 1 if m < high else 2)
        out[str(record.get("id"))] = _BAND_NAMES[min(bands)] if bands else "unknown"
    return out


# --------------------------------------------------------------------------
# 배치(listing)
# --------------------------------------------------------------------------


def listings_of(question: dict[str, Any]) -> list[dict[str, Any]]:
    """예측의 질문 하나 → 그 배치들 (``listings``가 없으면 질문 자신이 배치 하나 — 옛 seed 하나짜리 예측)."""
    found = question.get("listings")
    return found if isinstance(found, list) and found else [question]


def average_answers(qtype: str, answers: list[dict[str, Any]]) -> dict[str, Any]:
    """배치들의 답을 같은 무게로 평균한 답 (choice는 키별 확률, noul은 p_yes, score는 value와 구간 질량)."""
    if not answers:
        raise ValueError("평균할 답이 없다")
    n = len(answers)
    if qtype == "choice":
        keys = list(answers[0]["probabilities"])
        return {"probabilities": {k: sum(float(a["probabilities"][k]) for a in answers) / n for k in keys}}
    if qtype == "noul":
        return {"p_yes": sum(float(a["p_yes"]) for a in answers) / n}
    out: dict[str, Any] = {"value": sum(float(a["value"]) for a in answers) / n}
    if all(isinstance(a.get("bins"), dict) for a in answers):
        out["bins"] = {b: sum(float(a["bins"][b]) for a in answers) / n for b in answers[0]["bins"]}
    return out


def merge_runs(runs: list[list[dict[str, Any]]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """같은 기록을 여러 번(옛 seed run들, 또는 배치를 가진 예측들) 채점한 것 → 질문마다 **서로 다른** 배치들을 모은 예측.

    같은 보이는 순서(``options``)는 한 번만 둔다(채점이 결정적이라 같은 배치는 같은 답이다 — 리뷰 1 I4); 건너뛴 수를 돌려준다.
    질문의 ``answer``는 배치 평균, ``code_mass``는 배치 평균. 기록·질문 순서는 첫 run을 따른다.
    """
    if not runs:
        return [], {"runs": 0, "duplicate_listings_skipped": {}}
    collected: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    seen: dict[tuple[str, str], set[tuple[str, ...]]] = defaultdict(set)
    skipped: Counter = Counter()
    for run in runs:
        for prediction in run:
            rid = str(prediction["id"])
            for name, question in prediction["questions"].items():
                for listing in listings_of(question):
                    order = tuple(listing.get("options") or ())
                    if order in seen[(rid, name)]:
                        skipped[question["type"]] += 1
                        continue
                    seen[(rid, name)].add(order)
                    collected[(rid, name)].append({k: v for k, v in listing.items() if k not in ("listings", "n_listings")})
    out = []
    for prediction in runs[0]:
        rid = str(prediction["id"])
        questions = {}
        for name, question in prediction["questions"].items():
            found = collected[(rid, name)]
            masses = [float(l["code_mass"]) for l in found if l.get("code_mass") is not None]
            questions[name] = {
                "type": question["type"], "n_listings": len(found), "answer": average_answers(question["type"], [l["answer"] for l in found]),
                "code_mass": (sum(masses) / len(masses)) if masses else None, "listings": found,
            }
        out.append({**{k: v for k, v in prediction.items() if k != "questions"}, "questions": questions})
    return out, {"runs": len(runs), "duplicate_listings_skipped": dict(sorted(skipped.items()))}


# --------------------------------------------------------------------------
# 행
# --------------------------------------------------------------------------


def _recommendation(meta: dict[str, Any]) -> tuple[str, str | None, str | None]:
    """(none|right|wrong, 권고 선택지 키, 권고가 겨눈 질문 이름 — v0a는 ``recommendation.question``에 적는다)."""
    rec = meta.get("recommendation")
    if not isinstance(rec, dict) or not rec.get("present") or not rec.get("option"):
        return "none", None, None
    return ("wrong" if rec.get("wrong") else "right"), str(rec["option"]), (str(rec["question"]) if rec.get("question") else None)


def _record_groups(record: dict[str, Any], bands: dict[str, str]) -> dict[str, str]:
    meta = record.get("meta") or {}
    phenomena = meta.get("phenomena")
    hint = meta.get("hint")
    _, pair_kind = pair_of(record)
    agent = meta.get("agent")
    distractors = meta.get("distractors")
    units = meta.get("units") if isinstance(meta.get("units"), dict) else {}
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
        "margin_norm_band": margin_norm_band(meta),
        "key_entropy_band": entropy_band(key_entropy(record)),
        "pair": pair_kind or "none",
        "key_style": str(meta.get("key_style") or "unknown"),
        "agent": str(agent.get("kind") or "unknown") if isinstance(agent, dict) else ("none" if agent is None else str(agent)),
        "distractors": str(distractors) if isinstance(distractors, int) and not isinstance(distractors, bool) else "unknown",
        "distance_unit": str(units.get("distance") or "unknown"),
    }


def score_label_range(qtype: str, label: dict[str, Any] | None) -> str:
    """score 질문의 정답이 v0c-p2의 읽기 범위 :data:`SCORE_EDGE_RANGE` 밖(``outside``)인지 안(``inside``)인지; 그 밖의 질문은 ``n/a``."""
    if qtype != "score" or not isinstance(label, dict) or not _finite(label.get("value")):
        return "n/a"
    value = float(label["value"])
    return "outside" if value < SCORE_EDGE_RANGE[0] or value > SCORE_EDGE_RANGE[1] else "inside"


def _row_metrics(qtype: str, label: dict[str, Any], answer: dict[str, Any]) -> dict[str, Any]:
    if qtype == "noul":
        return binary_metrics(answer["p_yes"], label["p_yes"])
    if qtype == "score":
        return binary_metrics(answer["value"], label["value"])
    return choice_metrics(answer["probabilities"], label)


def question_rows(records: list[dict[str, Any]], predictions: list[dict[str, Any]], *, reading: str = "single_order") -> list[dict[str, Any]]:
    """기록 × 질문 [× 배치]마다 행 — 정답·모델 답·지표(``m``)·기준선(``u``)·무게(``w``)·묶음(``groups``)·위치 (모듈 설명의 두 읽기).

    예측 없음·정답 없음은 ``status``로 남는다(무게 1의 행 하나).
    """
    if reading not in READINGS:
        raise ValueError(f"reading: {list(READINGS)} 중 하나 (받은 값: {reading!r})")
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
            block = block_of(qtype, kind)
            criterion = criterion_of(record, name) if block == "choice/action" else "n/a"
            qgroups = dict(groups)
            qgroups["criterion"] = criterion
            qgroups["label_entropy_band"] = entropy_band(normalized_entropy(qtype, label))
            qgroups["score_label_range"] = score_label_range(qtype, label)
            base: dict[str, Any] = {
                "record_id": rid, "question": name, "type": qtype, "kind": kind, "block": block, "family": question_family(record, name),
                "criterion": criterion, "label": label, "answer": None, "options": None, "values": None, "code_mass": None, "w": 1.0,
                "listing": None, "n_listings": 0, "label_position": None, "model_position": None, "m": None, "u": None,
            }
            predicted = prediction.get(name)
            if predicted is None or not isinstance(label, dict):
                row = {**base, "status": "missing_prediction" if predicted is None else "missing_label", "groups": {**qgroups, "label_position": "n/a"}}
                rows.append(row)
                continue
            listings = listings_of(predicted)
            if reading == "order_averaged":
                masses = [float(l["code_mass"]) for l in listings if l.get("code_mass") is not None]
                picks = [(None, average_answers(qtype, [l["answer"] for l in listings]), None, None, (sum(masses) / len(masses)) if masses else None, 1.0)]
            else:
                picks = [(i, l["answer"], l.get("options"), l.get("values"), l.get("code_mass"), 1.0 / len(listings)) for i, l in enumerate(listings)]
            for index, answer, options, values, mass, weight in picks:
                m = _row_metrics(qtype, label, answer)
                row = {
                    **base, "status": "scored", "answer": answer, "options": list(options) if options else None, "values": values,
                    "code_mass": mass, "w": weight, "listing": index, "n_listings": len(listings), "m": m, "u": uniform_metrics(qtype, label, m),
                }
                if qtype == "choice" and row["options"]:
                    row["label_position"] = min(row["options"].index(k) for k in m["best"] if k in row["options"])
                    row["model_position"] = row["options"].index(m["predicted"])
                row["groups"] = {**qgroups, "label_position": str(row["label_position"] + 1) if row["label_position"] is not None else "n/a"}
                rows.append(row)
    return rows


# --------------------------------------------------------------------------
# 묶음
# --------------------------------------------------------------------------


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _wmean(pairs: Iterable[tuple[Any, float]]) -> float | None:
    num = den = 0.0
    for value, weight in pairs:
        if value is None:
            continue
        num += float(weight) * float(value)
        den += float(weight)
    return num / den if den > 0 else None


def _wpearson(triples: Iterable[tuple[float, float, float]]) -> float | None:
    data = np.array([(float(p), float(y), float(w)) for p, y, w in triples], dtype=np.float64)
    if data.shape[0] < 3:
        return None
    p, y, w = data[:, 0], data[:, 1], data[:, 2]
    mp, my = np.average(p, weights=w), np.average(y, weights=w)
    vp, vy = np.average((p - mp) ** 2, weights=w), np.average((y - my) ** 2, weights=w)
    if vp <= 0 or vy <= 0:
        return None
    return float(np.average((p - mp) * (y - my), weights=w) / math.sqrt(vp * vy))


def _questions(rows: Iterable[dict[str, Any]]) -> int:
    return len({(r["record_id"], r["question"]) for r in rows})


def _value(source: dict[str, Any] | None, key: str) -> float | None:
    if source is None:
        return None
    value = source.get(key)
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    return float(value) if value is not None else None


def _regret_block(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """후회의 분포 (질문마다 배치 평균 후회 → 질문 위에서): 평균·중앙값·최대·상위 :data:`TOP_K` 비중·양수 비율·척도 없는 평균과 기준선."""
    sel = [r for r in rows if _value(r["m"], "regret") is not None]
    if not sel:
        return {"regret_n": 0}
    per_question: dict[tuple[str, str], list[float]] = defaultdict(lambda: [0.0, 0.0])
    for r in sel:
        acc = per_question[(r["record_id"], r["question"])]
        acc[0] += r["w"] * float(r["m"]["regret"])
        acc[1] += r["w"]
    expected = sorted((s / w for s, w in per_question.values()), reverse=True)
    total = sum(expected)
    regret_mean = _wmean((r["m"]["regret"], r["w"]) for r in sel)
    norm_mean = _wmean((r["m"]["regret_norm"], r["w"]) for r in sel)
    uniform_mean = _wmean((r["u"].get("regret"), r["w"]) for r in sel)
    uniform_norm = _wmean((r["u"].get("regret_norm"), r["w"]) for r in sel)
    return {
        "regret_n": len(expected), "regret_mean": regret_mean, "regret_median": statistics.median(expected), "regret_max": expected[0],
        "regret_top10_share": (sum(expected[:TOP_K]) / total) if total > 0 else None,
        "regret_positive_rate": _wmean((1.0 if r["m"]["regret"] > _TIE else 0.0, r["w"]) for r in sel),
        "regret_norm_mean": norm_mean, "sampled_regret_mean": _wmean((r["m"]["sampled_regret"], r["w"]) for r in sel),
        "uniform_regret_mean": uniform_mean, "uniform_regret_norm_mean": uniform_norm,
        "regret_vs_uniform": (regret_mean - uniform_mean) if regret_mean is not None and uniform_mean is not None else None,
        "regret_norm_vs_uniform": (norm_mean - uniform_norm) if norm_mean is not None and uniform_norm is not None else None,
    }


def _block_summary(block: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    def wm(key: str, source: str = "m") -> float | None:
        return _wmean((_value(r[source], key), r["w"]) for r in rows)

    out: dict[str, Any] = {"n": _questions(rows), "rows": len(rows), "code_mass": _wmean((r["code_mass"], r["w"]) for r in rows if r.get("code_mass") is not None)}
    if block in ("noul", "score"):
        sided = [r for r in rows if abs(r["m"]["y"] - FLIP_THRESHOLD) >= SIDE_MARGIN]
        out.update({
            "log_loss": wm("log_loss"), "kl": wm("kl"), "brier": wm("brier"), "mae": wm("abs_error"),
            "ece": ece((r["m"]["p"], r["m"]["y"], r["w"]) for r in rows), "mean_p": wm("p"), "mean_y": wm("y"),
            "pearson": _wpearson((r["m"]["p"], r["m"]["y"], r["w"]) for r in rows),
            # 같은 쪽 일치: 정답이 0.5에서 SIDE_MARGIN 이상 떨어진 질문에서 모델 값이 0.5의 같은 쪽인 비율 (읽기 쉬운 '맞혔나')
            "side_agreement": _wmean((1.0 if (r["m"]["p"] >= FLIP_THRESHOLD) == (r["m"]["y"] >= FLIP_THRESHOLD) else 0.0, r["w"]) for r in sided),
            "side_n": _questions(sided), "uniform": {"log_loss": wm("log_loss", "u"), "kl": wm("kl", "u"), "brier": wm("brier", "u")},
        })
        out["brier_vs_uniform"] = out["brier"] - out["uniform"]["brier"]
        out["kl_vs_uniform"] = out["kl"] - out["uniform"]["kl"]
        return out
    out.update({
        "ce": wm("ce"), "kl": wm("kl"), "brier": wm("brier"), "accuracy": wm("match"),
        "uniform": {"ce": wm("ce", "u"), "kl": wm("kl", "u"), "accuracy": wm("match", "u")},
    })
    out["accuracy_vs_uniform"] = out["accuracy"] - out["uniform"]["accuracy"]
    if block == "choice/action" or any(_value(r["m"], "regret") is not None for r in rows):
        out.update(_regret_block(rows))
    if block == "choice/state":
        out["ece"] = ece((r["m"]["confidence"], r["m"]["target"], r["w"]) for r in rows)
    return out


def aggregate(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """채점된 행 → 묶음(:data:`BLOCKS`)별 요약과 ``_all``(질문 수, 평균 KL과 그 기준선, 코드 질량)."""
    scored = [r for r in rows if r.get("m") is not None]
    by_block: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in scored:
        by_block[row.get("block") or block_of(row["type"], row.get("kind"))].append(row)
    out: dict[str, Any] = {block: _block_summary(block, by_block[block]) for block in BLOCKS if by_block.get(block)}
    out["_all"] = {
        "n": _questions(scored), "kl": _wmean((r["m"]["kl"], r["w"]) for r in scored), "uniform_kl": _wmean((r["u"]["kl"], r["w"]) for r in scored),
        "code_mass": _wmean((r["code_mass"], r["w"]) for r in scored if r.get("code_mass") is not None),
    }
    return out


def _group_value(row: dict[str, Any], dim: str) -> str:
    return str(row["family"] if dim == "family" else row["groups"].get(dim, "unknown"))


def group_tables(rows: list[dict[str, Any]], dimensions: Iterable[str] = GROUP_DIMENSIONS) -> dict[str, dict[str, Any]]:
    """묶음 차원마다 값 → :func:`aggregate` (행의 ``groups[차원]``; 질문 계열은 ``family``)."""
    out = {}
    for dim in dimensions:
        buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            if row.get("m") is not None:
                buckets[_group_value(row, dim)].append(row)
        out[dim] = {value: aggregate(bucket) for value, bucket in sorted(buckets.items())}
    return out


def regret_by_criterion(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """행동 질문의 후회를 기준 계열별로 (리뷰 1 I3): ``default``·``time_critical``·``avoid_contact``·… 각각과 ``stated``(기준이 적힌 것 전부),
    ``all``. 칸마다 :func:`_regret_block` (평균·중앙값·최대·상위 10 비중·양수 비율·척도 없는 평균·무작위 고르기 기준선)."""
    action = [r for r in rows if r.get("m") is not None and r["block"] == "choice/action" and _value(r["m"], "regret") is not None]
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in action:
        buckets[row["criterion"]].append(row)
        if _criterion_class(row["criterion"]) == "stated":
            buckets["stated"].append(row)
    out = {name: _regret_block(sel) for name, sel in sorted(buckets.items())}
    if action:
        out["all"] = _regret_block(action)
    return out


# --------------------------------------------------------------------------
# 짝지은 구간
# --------------------------------------------------------------------------


def _paired(entries: list[tuple[str, tuple[str, str], float, float, float | None]], *, n_boot: int, seed: int) -> dict[str, Any] | None:
    """(기록 id, 질문 열쇠, 무게, 모델 값, 기준선 값) 목록 → 모델 평균·기준선 평균과 레코드 bootstrap 95 % 구간(모델, 모델 − 기준선)."""
    if not entries:
        return None
    acc: dict[str, list[float]] = defaultdict(lambda: [0.0, 0.0, 0.0])
    has_base = all(e[4] is not None for e in entries)
    for rid, _, w, value, base in entries:
        a = acc[rid]
        a[0] += w * value
        a[1] += w * (base if base is not None else 0.0)
        a[2] += w
    ids = sorted(acc)
    sums = np.array([acc[i][0] for i in ids])
    bases = np.array([acc[i][1] for i in ids])
    weights = np.array([acc[i][2] for i in ids])
    total = weights.sum()
    if total <= 0:
        return None
    rng = np.random.default_rng(int(seed))
    draws = rng.integers(0, len(ids), size=(int(n_boot), len(ids)))
    with np.errstate(invalid="ignore", divide="ignore"):
        drawn_w = weights[draws].sum(1)
        value_samples = sums[draws].sum(1) / drawn_w
        diff_samples = (sums[draws].sum(1) - bases[draws].sum(1)) / drawn_w
    low, high = _interval(value_samples)
    out: dict[str, Any] = {
        "value": float(sums.sum() / total), "low": low, "high": high, "n": len({e[1] for e in entries}), "records": len(ids),
    }
    if has_base:
        d_low, d_high = _interval(diff_samples)
        out["uniform"] = float(bases.sum() / total)
        out["vs_uniform"] = {"value": float((sums.sum() - bases.sum()) / total), "low": d_low, "high": d_high}
    return out


def _interval(samples: np.ndarray) -> tuple[float | None, float | None]:
    samples = samples[np.isfinite(samples)]
    if samples.size == 0:
        return None, None
    return float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))


def _entries(rows: list[dict[str, Any]], key: str) -> list[tuple[str, tuple[str, str], float, float, float | None]]:
    out = []
    for r in rows:
        value = _value(r["m"], key)
        if value is None:
            continue
        out.append((r["record_id"], (r["record_id"], r["question"]), r["w"], value, _value(r["u"], key)))
    return out


def group_signal(rows: list[dict[str, Any]], dimensions: Iterable[str] = SIGNAL_DIMENSIONS, *, n_boot: int = 1000, seed: int = 0) -> dict[str, Any]:
    """묶음 값마다 기준선 대비 신호 (리뷰 1 I1·I2): noul·score는 (가중) 상관 r(p, y)와 Brier − Brier(0.5)의 짝지은 구간; 행동 choice는
    정확도 − 무작위 고르기와 척도 없는 후회 − 무작위 고르기; 상태·규칙 choice는 정확도 − 무작위 고르기. 구간은 그 묶음의 레코드만 뽑는다."""
    scored = [r for r in rows if r.get("m") is not None]
    out: dict[str, Any] = {}
    for dim in dimensions:
        buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in scored:
            buckets[_group_value(row, dim)].append(row)
        table = {}
        for value, sel in sorted(buckets.items()):
            entry: dict[str, Any] = {}
            for block in BLOCKS:
                part = [r for r in sel if r["block"] == block]
                if not part:
                    continue
                if block in ("noul", "score"):
                    entry[block] = {"n": _questions(part), "pearson": _wpearson((r["m"]["p"], r["m"]["y"], r["w"]) for r in part),
                                    "brier": _paired(_entries(part, "brier"), n_boot=n_boot, seed=seed)}
                else:
                    entry[block] = {"n": _questions(part), "accuracy": _paired(_entries(part, "match"), n_boot=n_boot, seed=seed)}
                    if block == "choice/action":
                        entry[block]["regret_norm"] = _paired(_entries(part, "regret_norm"), n_boot=n_boot, seed=seed)
                        entry[block]["regret"] = _paired(_entries(part, "regret"), n_boot=n_boot, seed=seed)
            table[value] = entry
        out[dim] = table
    return out


# --------------------------------------------------------------------------
# 권고, 위치 편향
# --------------------------------------------------------------------------


def _recommendation_rows(records: list[dict[str, Any]], rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    index: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        if r.get("m") is not None:
            index[(r["record_id"], r["question"])].append(r)
    out = []
    for record in records:
        status, option, target = _recommendation(record.get("meta") or {})
        if option is None:
            continue
        for name, question in record["request"]["questions"].items():
            if target is not None and target in record["request"]["questions"] and name != target:
                continue  # 권고가 겨눈 질문이 적혀 있으면 그 질문만 (같은 키를 쓰는 다른 행동 질문은 그 권고의 대상이 아니다)
            if option not in (question.get("criteria") or {}):
                continue
            for row in index.get((str(record.get("id")), name), []):
                if row["type"] != "choice" or row["kind"] not in ("action", None):
                    continue
                n = len(row["label"]["probabilities"])
                out.append({
                    "record_id": row["record_id"], "question": name, "w": row["w"], "wrong": status == "wrong",
                    "followed": row["m"]["predicted"] == option, "model_p": float(row["answer"]["probabilities"].get(option, 0.0)),
                    "label_p": float(row["label"]["probabilities"].get(option, 0.0)), "label_best": option in row["m"]["best"],
                    "regret": row["m"]["regret"], "uniform_follow": 1.0 / n,
                })
    return out


def recommendation_summary(records: list[dict[str, Any]], rows: list[dict[str, Any]]) -> dict[str, Any]:
    """계획기 권고가 겨눈 행동 질문(``meta.recommendation.question``이 있으면 그 질문, 없으면 권고 키를 선택지로 가진 행동 질문):
    권고가 틀렸을 때·맞았을 때 모델이 권고를 고른 비율 등. ``uniform_follow_rate`` = 무작위로 하나 고르면 권고를 고를 확률(1/n)."""
    rec_rows = _recommendation_rows(records, rows)
    out = {}
    for name, wrong in (("wrong", True), ("right", False)):
        sel = [r for r in rec_rows if r["wrong"] is wrong]
        out[name] = {
            "n": _questions(sel), "follow_rate": _wmean((1.0 if r["followed"] else 0.0, r["w"]) for r in sel),
            "uniform_follow_rate": _wmean((r["uniform_follow"], r["w"]) for r in sel),
            "model_p_recommended": _wmean((r["model_p"], r["w"]) for r in sel), "label_p_recommended": _wmean((r["label_p"], r["w"]) for r in sel),
            "label_best_rate": _wmean((1.0 if r["label_best"] else 0.0, r["w"]) for r in sel), "regret_mean": _wmean((r["regret"], r["w"]) for r in sel),
        }
    return out


def _ascending(row: dict[str, Any]) -> bool:
    values = row.get("values")
    if values:
        return float(values[0]) < float(values[-1])
    return str(row["options"][0]) < str(row["options"][-1])


def position_bias(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """위치 편향 — 한 순서 읽기의 행(보이는 배치가 있는 행)에서. 무게로 질문마다 같은 몫.

    * choice — 선택지 수마다: 모델 argmax 자리 비율, 정답 최선(첫 최선) 자리 비율, 자리별 평균 모델 확률, 정답 최선 자리별 정확도.
      ``all``: 첫·마지막 자리 비율(모델·정답)과 1/n의 평균.
    * noul — Yes가 첫 코드일 때와 둘째일 때의 평균 p_yes(모델·정답); ``shift`` = 첫 − 둘째, ``excess`` = shift − 정답의 shift.
    * score — 구간을 오름(첫 = 0.0–0.1)·내림으로 보였을 때의 평균 값; ``shift`` = 오름 − 내림.
    """
    scored = [r for r in rows if r.get("m") is not None and r.get("options")]
    choice: dict[str, Any] = {}
    by_n: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in scored:
        if row["type"] == "choice":
            by_n[len(row["options"])].append(row)
    for n, sel in sorted(by_n.items()):
        choice[str(n)] = {
            "n": _questions(sel),
            "model_argmax_position_rates": [_wmean((1.0 if r["model_position"] == i else 0.0, r["w"]) for r in sel) for i in range(n)],
            "label_best_position_rates": [_wmean((1.0 if r["label_position"] == i else 0.0, r["w"]) for r in sel) for i in range(n)],
            "mean_probability_by_position": [_wmean((float(r["answer"]["probabilities"][r["options"][i]]), r["w"]) for r in sel) for i in range(n)],
            "accuracy_by_label_position": [_wmean((1.0 if r["m"]["match"] else 0.0, r["w"]) for r in sel if r["label_position"] == i) for i in range(n)],
        }
    all_choice = [r for sel in by_n.values() for r in sel]
    choice_all = {
        "n": _questions(all_choice),
        "model_first_rate": _wmean((1.0 if r["model_position"] == 0 else 0.0, r["w"]) for r in all_choice),
        "label_first_rate": _wmean((1.0 if r["label_position"] == 0 else 0.0, r["w"]) for r in all_choice),
        "expected_first_rate": _wmean((1.0 / len(r["options"]), r["w"]) for r in all_choice),
        "model_last_rate": _wmean((1.0 if r["model_position"] == len(r["options"]) - 1 else 0.0, r["w"]) for r in all_choice),
        "label_last_rate": _wmean((1.0 if r["label_position"] == len(r["options"]) - 1 else 0.0, r["w"]) for r in all_choice),
    }

    def split(sel: list[dict[str, Any]], first: Callable[[dict[str, Any]], bool]) -> dict[str, Any]:
        a = [r for r in sel if first(r)]
        b = [r for r in sel if not first(r)]
        return {
            "first": {"n": len(a), "model": _wmean((r["m"]["p"], r["w"]) for r in a), "label": _wmean((r["m"]["y"], r["w"]) for r in a)},
            "second": {"n": len(b), "model": _wmean((r["m"]["p"], r["w"]) for r in b), "label": _wmean((r["m"]["y"], r["w"]) for r in b)},
        }

    def shifts(parts: dict[str, Any]) -> dict[str, Any]:
        f, s = parts["first"], parts["second"]
        shift = (f["model"] - s["model"]) if f["model"] is not None and s["model"] is not None else None
        label_shift = (f["label"] - s["label"]) if f["label"] is not None and s["label"] is not None else None
        return {"shift": shift, "label_shift": label_shift, "excess": (shift - label_shift) if shift is not None and label_shift is not None else None}

    parts = split([r for r in scored if r["type"] == "noul"], lambda r: r["options"][0] == "yes")
    noul = {
        "yes_first": {"n": parts["first"]["n"], "model_p_yes": parts["first"]["model"], "label_p_yes": parts["first"]["label"]},
        "yes_second": {"n": parts["second"]["n"], "model_p_yes": parts["second"]["model"], "label_p_yes": parts["second"]["label"]},
        **shifts(parts),
    }
    parts = split([r for r in scored if r["type"] == "score"], _ascending)
    score = {
        "ascending": {"n": parts["first"]["n"], "model_value": parts["first"]["model"], "label_value": parts["first"]["label"]},
        "descending": {"n": parts["second"]["n"], "model_value": parts["second"]["model"], "label_value": parts["second"]["label"]},
        **shifts(parts),
    }
    return {"choice": {**choice, "all": choice_all}, "noul": noul, "score": score}


# --------------------------------------------------------------------------
# 순서 바꿈 (배치 사이)
# --------------------------------------------------------------------------


def _decisive_choice(probabilities: dict[str, float]) -> bool:
    """choice 답이 결정적인가 — 1등 확률이 2등보다 2 × :data:`DECISIVE_MARGIN` 이상 크다(이진에서 |p − 0.5| ≥ 여유와 같은 뜻)."""
    values = sorted((float(v) for v in probabilities.values()), reverse=True)
    return len(values) < 2 or values[0] - values[1] >= 2 * DECISIVE_MARGIN


def _answer_shift(qtype: str, a: dict[str, Any], b: dict[str, Any]) -> tuple[float, bool, bool]:
    """같은 질문의 두 답 사이 (이동, 판정 뒤집힘, 결정적인 뒤집힘): choice는 전변동 거리와 argmax(결정적 = 두 답 모두 1등이 2등을
    0.2 이상 앞섬), noul·score는 |Δ|와 ``>= 0.5`` 판정(결정적 = 두 값 모두 0.5에서 :data:`DECISIVE_MARGIN` 이상 떨어짐)."""
    if qtype == "choice":
        pa, pb = a["probabilities"], b["probabilities"]
        keys = list(pa)
        shift = 0.5 * sum(abs(float(pa[k]) - float(pb.get(k, 0.0))) for k in keys)
        top_a = max(keys, key=lambda k: (float(pa[k]), -keys.index(k)))
        top_b = max(keys, key=lambda k: (float(pb.get(k, 0.0)), -keys.index(k)))
        flip = top_a != top_b
        return shift, flip, flip and _decisive_choice(pa) and _decisive_choice(pb)
    field = "p_yes" if qtype == "noul" else "value"
    x, y = float(a[field]), float(b[field])
    flip = (x >= FLIP_THRESHOLD) != (y >= FLIP_THRESHOLD)
    decisive = abs(x - FLIP_THRESHOLD) >= DECISIVE_MARGIN - _TIE and abs(y - FLIP_THRESHOLD) >= DECISIVE_MARGIN - _TIE
    return abs(x - y), flip, flip and decisive


def permutation_sensitivity(predictions: list[dict[str, Any]]) -> dict[str, Any]:
    """질문마다 **보이는 순서가 다른** 배치 짝 모두에서 :func:`_answer_shift` (기획서 §10 T6의 판정 변경률 꼴; 리뷰 1 I4).

    같은 순서의 짝은 세지 않고(채점이 결정적이라 늘 같은 답 — 세면 비율이 희석된다) ``identical_pairs_skipped``로 남긴다.
    ``shift_mean``(**주 지표** — choice는 키로 되돌린 분포의 전변동 거리, noul·score는 |Δ|), ``flip_rate``(판정이 뒤집힌 짝의 비율 —
    답이 0.5 가까이 있으면 작은 이동에도 뒤집혀 흔들린다), ``decisive_flip_rate``(두 답이 모두 경계에서 :data:`DECISIVE_MARGIN` 이상
    떨어진 채 뒤집힌 짝의 비율; 분모는 같은 모든 짝).
    """
    sums: dict[str, list[float]] = defaultdict(lambda: [0, 0, 0.0, 0, 0, 0])  # 질문, 짝, 이동 합, 뒤집힘, 건너뛴 같은 순서 짝, 결정적 뒤집힘
    for prediction in predictions:
        for question in prediction["questions"].values():
            qtype = question["type"]
            listings = listings_of(question)
            counted = False
            for a, b in itertools.combinations(listings, 2):
                if list(a.get("options") or ()) == list(b.get("options") or ()):
                    for bucket in (qtype, "_all"):
                        sums[bucket][4] += 1
                    continue
                shift, flip, decisive = _answer_shift(qtype, a["answer"], b["answer"])
                for bucket in (qtype, "_all"):
                    sums[bucket][1] += 1
                    sums[bucket][2] += shift
                    sums[bucket][3] += 1 if flip else 0
                    sums[bucket][5] += 1 if decisive else 0
                    if not counted:
                        sums[bucket][0] += 1
                counted = True
    out = {}
    for bucket, (questions, pairs, shift, flips, skipped, decisive) in sums.items():
        out[bucket] = {
            "questions": int(questions), "pairs": int(pairs), "shift_mean": shift / pairs if pairs else None, "flip_rate": flips / pairs if pairs else None,
            "decisive_flip_rate": decisive / pairs if pairs else None, "decisive_flips": int(decisive), "identical_pairs_skipped": int(skipped),
        }
    out.setdefault("_all", {"questions": 0, "pairs": 0, "shift_mean": None, "flip_rate": None, "decisive_flip_rate": None, "decisive_flips": 0,
                            "identical_pairs_skipped": 0})
    return dict(sorted(out.items()))


def order_effects(predictions: list[dict[str, Any]]) -> dict[str, Any]:
    """같은 질문을 다른 순서로 본 배치들 사이의 **짝지은** 순서 효과 (질문 안 비교).

    * noul — 두 순서가 모두 있는 질문마다 평균 p_yes(Yes 먼저) − 평균 p_yes(Yes 나중); ``mean_shift``와 양수인 비율.
    * score — 같은 식으로 평균 값(오름) − 평균 값(내림).
    * choice — 순서가 둘 이상인 질문에서, 선택지마다 (그 배치의 확률 − 그 선택지의 배치 평균)을 그 선택지가 놓인 자리로 모아 평균
      (선택지 수마다). 양수인 자리는 그 자리에 놓이는 것만으로 확률이 오른다는 뜻이다.
    """
    shifts: dict[str, list[float]] = {"noul": [], "score": []}
    deviations: dict[int, list[list[float]]] = {}
    choice_questions: Counter = Counter()
    for prediction in predictions:
        for question in prediction["questions"].values():
            qtype = question["type"]
            listings = listings_of(question)
            if len({tuple(l.get("options") or ()) for l in listings}) < 2:
                continue
            if qtype in ("noul", "score"):
                first = [l for l in listings if (l["options"][0] == "yes" if qtype == "noul" else _ascending(l))]
                other = [l for l in listings if l not in first]
                if first and other:
                    field = "p_yes" if qtype == "noul" else "value"
                    shifts[qtype].append(_mean([float(l["answer"][field]) for l in first]) - _mean([float(l["answer"][field]) for l in other]))
                continue
            options = listings[0]["options"]
            n = len(options)
            means = {k: _mean([float(l["answer"]["probabilities"][k]) for l in listings]) for k in options}
            slots = deviations.setdefault(n, [[] for _ in range(n)])
            choice_questions[n] += 1
            for l in listings:
                for position, key in enumerate(l["options"]):
                    slots[position].append(float(l["answer"]["probabilities"][key]) - means[key])
    out: dict[str, Any] = {}
    for qtype, values in shifts.items():
        out[qtype] = {"questions": len(values), "mean_shift": _mean(values), "positive_rate": _mean([1.0 if v > 0 else 0.0 for v in values])}
    out["choice"] = {
        str(n): {"questions": choice_questions[n], "mean_deviation_by_position": [_mean(slot) for slot in slots]} for n, slots in sorted(deviations.items())
    }
    return out


def listing_stats(predictions: list[dict[str, Any]], merge: dict[str, Any] | None = None) -> dict[str, Any]:
    """질문 종류마다 배치 수: 질문 수, 평균 배치 수, 순서가 둘 이상인 질문 비율 (+ 병합에서 건너뛴 같은 배치 수)."""
    per: dict[str, list[int]] = defaultdict(list)
    for prediction in predictions:
        for question in prediction["questions"].values():
            per[question["type"]].append(len({tuple(l.get("options") or ()) for l in listings_of(question)}))
    out: dict[str, Any] = {
        qtype: {"questions": len(counts), "mean_listings": _mean([float(c) for c in counts]), "multi_listing_rate": _mean([1.0 if c >= 2 else 0.0 for c in counts])}
        for qtype, counts in sorted(per.items())
    }
    if merge is not None:
        out["merge"] = merge
    return out


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
        _, flip, decisive = _answer_shift("noul", {"p_yes": row_a["m"]["p"]}, {"p_yes": row_b["m"]["p"]})
        return {"label_move": abs(dq), "model_move": abs(dp), "agree": dq * dp > 0.0, "flip": flip, "decisive_flip": decisive}
    qa, qb = row_a["label"]["probabilities"], row_b["label"]["probabilities"]
    if set(qa) != set(qb):
        return None
    pa, pb = row_a["answer"]["probabilities"], row_b["answer"]["probabilities"]
    dq = {k: float(qb[k]) - float(qa[k]) for k in qa}
    dp = {k: float(pb.get(k, 0.0)) - float(pa.get(k, 0.0)) for k in qa}
    flip = row_a["m"]["predicted"] != row_b["m"]["predicted"]
    return {
        "label_move": 0.5 * sum(abs(v) for v in dq.values()), "model_move": 0.5 * sum(abs(v) for v in dp.values()),
        "agree": sum(dq[k] * dp[k] for k in qa) > 0.0, "flip": flip, "decisive_flip": flip and _decisive_choice(pa) and _decisive_choice(pb),
    }


def pair_deltas(records: list[dict[str, Any]], rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """쌍마다(한 번씩) 같은 이름·같은 종류의 채점된 질문의 변화 base → variant (질문마다 행 하나인 순서 평균 읽기의 행을 준다).
    base·variant는 ``meta.pair.role``(v0a: base|variant)을 따르고, 역할이 없으면 ``pair.of``가 가리키는 쪽을 base로 둔다."""
    by_id = {str(r.get("id")): r for r in records}
    index = {(r["record_id"], r["question"]): r for r in rows if r.get("m") is not None}
    seen: set[tuple[str, str]] = set()
    out: dict[str, list[dict[str, Any]]] = {"numeric": [], "wording": []}
    for record in records:
        partner, kind = pair_of(record)
        rid = str(record.get("id"))
        if partner is None or partner not in by_id or kind not in out:
            continue
        key = (min(rid, partner), max(rid, partner))
        if key in seen:
            continue
        seen.add(key)
        role = ((record.get("meta") or {}).get("pair") or {}).get("role")
        base, variant = (rid, partner) if role == "base" else (partner, rid)
        for name in by_id[variant]["request"]["questions"]:
            a, b = index.get((base, name)), index.get((variant, name))
            if a is None or b is None:
                continue
            delta = _delta(a, b)
            if delta is not None:
                out[kind].append({"base": base, "variant": variant, "question": name, "type": a["type"], **delta})
    return out


def pair_summary(records: list[dict[str, Any]], rows: list[dict[str, Any]]) -> dict[str, Any]:
    """사실 하나만 바꾼 쌍 (기획서 §8·§10 T5) — 순서 평균 답으로 비교한다(쌍의 두 기록은 같은 배치들을 보므로 위치 효과가 빠진다).

    * ``numeric`` — 정답이 움직인 질문(``|Δ| ≥`` :data:`MOVE_MIN`; choice는 전변동 거리)에서 모델도 같은 방향으로 움직였는가
      (noul·score: 부호가 같다; choice: ⟨Δp, Δq⟩ > 0). ``direction_correct_rate``, ``model_still_rate``, 평균 변화 크기.
    * ``wording`` — 숫자는 그대로라 정답은 (거의) 움직이지 않는다. 모델의 흔들림 = 평균 |Δp|(choice는 전변동 거리)와 판정이 뒤집힌 비율.
    """
    found = pair_deltas(records, rows)
    deltas = {kind: [(d["type"], d) for d in entries] for kind, entries in found.items()}
    pair_count = Counter({kind: len({(d["base"], d["variant"]) for d in entries}) for kind, entries in found.items()})
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
            "decisive_flip_rate": _mean([1.0 if d["decisive_flip"] else 0.0 for d in sel]),
        }
    out["wording"] = {
        "pairs": pair_count["wording"], "questions": len(wording), "model_shift_mean": _mean([d["model_move"] for _, d in wording]),
        "label_shift_mean": _mean([d["label_move"] for _, d in wording]), "flip_rate": _mean([1.0 if d["flip"] else 0.0 for _, d in wording]),
        "decisive_flip_rate": _mean([1.0 if d["decisive_flip"] else 0.0 for _, d in wording]),
        "by_type": by_type,
    }
    return out


# --------------------------------------------------------------------------
# 기준선, bootstrap, 비교
# --------------------------------------------------------------------------


def uniform_reference(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """기준선(noul·score 언제나 0.5, choice 무작위로 하나 고르기)의 지표 — 같은 행·같은 무게에서."""
    scored = [r for r in rows if r.get("m") is not None]
    out: dict[str, Any] = {}
    for block in BLOCKS:
        sel = [r for r in scored if r["block"] == block]
        if not sel:
            continue

        def wu(key: str) -> float | None:
            return _wmean((_value(r["u"], key), r["w"]) for r in sel)

        if block in ("noul", "score"):
            out[block] = {"n": _questions(sel), "log_loss": wu("log_loss"), "kl": wu("kl"), "brier": wu("brier")}
            continue
        out[block] = {"n": _questions(sel), "ce": wu("ce"), "kl": wu("kl"), "accuracy": wu("match")}
        if any(_value(r["u"], "regret") is not None for r in sel):
            out[block]["regret_mean"] = wu("regret")
            out[block]["regret_norm_mean"] = wu("regret_norm")
    return out


def _headline_entries(rows: list[dict[str, Any]], rec_rows: list[dict[str, Any]]) -> dict[str, list[tuple[str, tuple[str, str], float, float, float | None]]]:
    """머리 지표마다 (기록 id, 질문 열쇠, 무게, 모델 값, 기준선 값) 목록 — bootstrap·비교의 재료."""
    out: dict[str, list[tuple[str, tuple[str, str], float, float, float | None]]] = {name: [] for name, *_ in HEADLINES}
    for row in rows:
        if row.get("m") is None:
            continue
        for name, block, key, criterion in HEADLINES:
            if block.startswith("rec_") or (block != "_all" and row["block"] != block):
                continue
            if criterion in ("label_outside", "label_inside"):
                if row["groups"].get("score_label_range") != criterion.removeprefix("label_"):
                    continue
            elif criterion is not None and _criterion_class(row["criterion"]) != criterion:
                continue
            value = _value(row["m"], key)
            if value is None:
                continue
            out[name].append((row["record_id"], (row["record_id"], row["question"]), row["w"], value, _value(row["u"], key)))
    for rec in rec_rows:
        name = "recommendation.wrong.follow_rate" if rec["wrong"] else "recommendation.right.follow_rate"
        out[name].append((rec["record_id"], (rec["record_id"], rec["question"]), rec["w"], 1.0 if rec["followed"] else 0.0, rec["uniform_follow"]))
    return out


def bootstrap(rows: list[dict[str, Any]], records: list[dict[str, Any]], *, n_boot: int = 1000, seed: int = 0) -> dict[str, Any]:
    """머리 지표(:data:`HEADLINES`)의 레코드 단위 bootstrap 95 % 구간과, 같은 뽑기로 잰 **모델 − 기준선**의 짝지은 구간(``vs_uniform``)."""
    entries = _headline_entries(rows, _recommendation_rows(records, rows))
    out = {}
    for name, *_ in HEADLINES:
        found = _paired(entries[name], n_boot=n_boot, seed=seed)
        if found is not None:
            out[name] = found
    return out


def reasoning_stats(predictions: list[dict[str, Any]], *, gate: float = COMPLETION_GATE) -> dict[str, Any] | None:
    """생각(설명) 뒤 답 run의 생성 통계와 **완료율 문턱**: 완료 = 모델이 상한 전에 스스로 끝냈다(답 표지와 코드 글자를 썼거나 끝 토큰).
    완료율이 `gate` 미만이면 ``gate.passed``가 거짓이고, 그 run의 생각 대 바로 답 비교는 판정 불가다."""
    entries = [l["reasoning"] for p in predictions for q in p["questions"].values() for l in listings_of(q) if l.get("reasoning")]
    if not entries:
        return None
    n = len(entries)

    def complete(e: dict[str, Any]) -> bool:
        if "complete" in e:
            return bool(e["complete"])
        return e.get("stop") in ("eos", "think_end", "answer")  # v0c-p1 기록(완료 칸 없음): 스스로 멈춘 것만

    rate = sum(1 for e in entries if complete(e)) / n
    return {
        "questions": n, "generated_tokens_mean": sum(e["generated_tokens"] for e in entries) / n,
        "cap_rate": sum(1 for e in entries if e["stop"] == "cap") / n, "self_answer_rate": sum(1 for e in entries if e.get("self_answer")) / n,
        "complete_rate": rate, "stops": dict(sorted(Counter(e["stop"] for e in entries).items())),
        "gate": {"threshold": gate, "passed": rate >= gate},
    }


def compare_runs(
    records: list[dict[str, Any]], predictions_a: list[dict[str, Any]], predictions_b: list[dict[str, Any]], *, n_boot: int = 1000, seed: int = 0
) -> dict[str, Any]:
    """같은 기록의 두 run(a, b) — 두 읽기 각각에서, 둘 다 채점된 (기록, 질문)만으로 머리 지표의 차 a − b와 짝지은 레코드 bootstrap
    95 % 구간. 어느 쪽이든 생각 뒤 답 run이 완료율 문턱에 못 미치면 ``gate.passed``가 거짓(판정 불가)이다."""
    ids_a = {str(p["id"]) for p in predictions_a}
    ids_b = {str(p["id"]) for p in predictions_b}
    kept = [r for r in records if str(r.get("id")) in ids_a & ids_b]
    out: dict[str, Any] = {"records": len(kept), "readings": {}}
    gates = {side: reasoning_stats(preds) for side, preds in (("a", predictions_a), ("b", predictions_b))}
    out["gate"] = {
        "passed": all(g is None or g["gate"]["passed"] for g in gates.values()),
        "a": gates["a"]["gate"] | {"complete_rate": gates["a"]["complete_rate"]} if gates["a"] else None,
        "b": gates["b"]["gate"] | {"complete_rate": gates["b"]["complete_rate"]} if gates["b"] else None,
    }
    for reading in READINGS:
        rows_a = question_rows(kept, predictions_a, reading=reading)
        rows_b = question_rows(kept, predictions_b, reading=reading)
        common = {(r["record_id"], r["question"]) for r in rows_a if r.get("m") is not None} & {
            (r["record_id"], r["question"]) for r in rows_b if r.get("m") is not None
        }
        rows_a = [r for r in rows_a if (r["record_id"], r["question"]) in common]
        rows_b = [r for r in rows_b if (r["record_id"], r["question"]) in common]
        ea = _headline_entries(rows_a, _recommendation_rows(kept, rows_a))
        eb = _headline_entries(rows_b, _recommendation_rows(kept, rows_b))
        result: dict[str, Any] = {"questions": len(common), "a": {}, "b": {}, "differences": {}}
        record_ids = sorted({rid for rid, _ in common})
        rng = np.random.default_rng(int(seed))
        draws = rng.integers(0, len(record_ids), size=(int(n_boot), len(record_ids))) if record_ids else None
        for name, *_ in HEADLINES:
            arrays = []
            for found in (ea[name], eb[name]):
                acc: dict[str, list[float]] = defaultdict(lambda: [0.0, 0.0])
                for rid, _, w, value, _ in found:
                    acc[rid][0] += w * value
                    acc[rid][1] += w
                arrays.append((np.array([acc[i][0] if i in acc else 0.0 for i in record_ids]), np.array([acc[i][1] if i in acc else 0.0 for i in record_ids])))
            (sa, ca), (sb, cb) = arrays
            if draws is None or ca.sum() == 0 or cb.sum() == 0:
                continue
            a_value, b_value = float(sa.sum() / ca.sum()), float(sb.sum() / cb.sum())
            with np.errstate(invalid="ignore", divide="ignore"):
                samples = sa[draws].sum(1) / ca[draws].sum(1) - sb[draws].sum(1) / cb[draws].sum(1)
            low, high = _interval(samples)
            result["a"][name], result["b"][name] = a_value, b_value
            result["differences"][name] = {"value": a_value - b_value, "low": low, "high": high, "n": len({e[1] for e in ea[name]})}
        out["readings"][reading] = result
    return out


# --------------------------------------------------------------------------
# 요약
# --------------------------------------------------------------------------

DEFINITIONS = {
    "readings": "single_order = one row per listing weighted 1/L (expected performance when asked in one order, the deployment condition); order_averaged = one row per question whose answer is the equal-weight average over its distinct listings (removes first-order position bias)",
    "log_loss": "noul/score: soft binary cross-entropy -[y ln p + (1-y) ln(1-p)] against the posterior label y (p clipped to [1e-12, 1-1e-12])",
    "kl": "cross-entropy minus the label entropy (0 = the model reproduces the posterior label)",
    "brier": "noul/score: (p - y)^2; choice: sum_k (p_k - q_k)^2",
    "vs_uniform": "paired record-bootstrap interval of model minus baseline on the same questions and the same draws: always 0.5 (noul, score) or a random pick (choice)",
    "ece": "10 equal-width bins of the prediction; sum_b w_b/W |mean prediction - mean label probability|; state choice: soft top-label",
    "ce": "choice: -sum_k q_k ln p_k against the posterior label distribution q",
    "accuracy": "choice: model argmax in the best set (action: minimum expected cost; otherwise label argmax; ties within 1e-9 all count)",
    "regret_mean": "action choice with expected_cost: E[cost(model argmax)] - min_a E[cost(a)]; reported per criterion family because stated-criterion costs (avoid_*) put violations near 1,000",
    "regret_norm": "scale-free regret: regret divided by the question's expected-cost range (max - min); random pick = (mean - min) / (max - min)",
    "regret_top10_share": "share of the total regret carried by the 10 questions with the largest (listing-averaged) regret",
    "sampled_regret": "action choice: sum_a p_a E[cost(a)] - min E[cost] (sampling the action from the model's distribution)",
    "code_mass": "probability mass the full-vocabulary softmax puts on the answer codes at the answer position (format adherence)",
    "follow_rate": "the recommendation's target question (meta.recommendation.question; otherwise action choices offering the recommended key): share where the model argmax is the recommended option",
    "margin_norm_band": "tightest |margin / sigma_m| of the record (v0a meta.margin_norm): near < 0.5 <= mid < 1.5 <= far; 'absent' when the engine did not export it",
    "entropy_bands": "normalized label entropy (binary entropy in bits for noul/score, H(q)/ln n for choice): low < 1/3 <= mid < 2/3 <= high (high = label near 0.5); key_entropy_band uses the record's key question, label_entropy_band the question itself",
    "margin_band": "legacy: tertiles of |margin| per margin name over the scored set (units differ per name); not a difficulty axis",
    "side_agreement": "noul/score: share of questions whose label is at least 0.1 away from 0.5 where the model value is on the same side of 0.5",
    "shift_mean": "primary order-sensitivity measure: mean |answer change| between listings of the same question whose option order differs (choice: total-variation distance of the key-mapped distributions; noul/score: |dp|)",
    "flip_rate": "pairs of listings of the same question whose option order differs (identical listings are not counted): share where the verdict changed (noul/score verdict = value >= 0.5; choice = argmax); unstable when answers sit near 0.5",
    "decisive_flip_rate": "share of the same pairs whose verdict changed while both answers were decisive: noul/score at least 0.1 away from 0.5, choice top-1 ahead of top-2 by at least 0.2",
    "score_label_range": "score questions whose label lies outside [0.05, 0.95] (the range the v0c-p2 bin reading cannot reach) vs inside; v0c-p3 reads all of [0, 1]",
    "direction_correct_rate": "numeric pairs: among questions whose label moved (|dy| or total variation >= 0.05), share where the model moved the same way (sign / inner product > 0)",
    "completion_gate": "explanation-first runs: a trace is complete when the model stopped on its own before the cap (it wrote the answer cue with a code letter, or an end token); below a 0.9 completion rate the run is marked inconclusive",
}


def summarize(
    records: list[dict[str, Any]], predictions: list[dict[str, Any]], *, reading: str = "single_order", n_boot: int = 1000, seed: int = 0
) -> dict[str, Any]:
    """기록 + 예측 → 한 읽기의 요약 (counts, overall, groups, signal, regret_by_criterion, recommendation, position_bias, pairs,
    reference, bootstrap, definitions). 위치 편향은 한 순서 읽기에서만, 쌍은 언제나 순서 평균 답으로 잰다."""
    rows = question_rows(records, predictions, reading=reading)
    status = Counter(r["status"] for r in rows if r["listing"] in (None, 0))
    scored = [r for r in rows if r["status"] == "scored"]
    known = {str(r.get("id")) for r in records}
    averaged_rows = scored if reading == "order_averaged" else [r for r in question_rows(records, predictions, reading="order_averaged") if r["status"] == "scored"]
    return {
        "reading": reading,
        "counts": {
            "records": len(records), "questions": sum(status.values()), "scored": status.get("scored", 0),
            "missing_prediction": status.get("missing_prediction", 0), "missing_label": status.get("missing_label", 0),
            "unmatched_predictions": sum(1 for p in predictions if str(p["id"]) not in known),
            "listing_rows": len(scored), "by_block": dict(sorted(Counter(r["block"] for r in scored if r["listing"] in (None, 0)).items())),
        },
        "overall": aggregate(scored),
        "groups": group_tables(scored),
        "signal": group_signal(scored, n_boot=n_boot, seed=seed),
        "regret_by_criterion": regret_by_criterion(scored),
        "recommendation": recommendation_summary(records, scored),
        "position_bias": position_bias(scored) if reading == "single_order" else None,
        "pairs": pair_summary(records, averaged_rows),
        "reference": {"uniform": uniform_reference(scored)},
        "bootstrap": bootstrap(scored, records, n_boot=n_boot, seed=seed),
        "definitions": DEFINITIONS,
    }


def summarize_all(
    records: list[dict[str, Any]], predictions: list[dict[str, Any]], *, n_boot: int = 1000, seed: int = 0, merge: dict[str, Any] | None = None
) -> dict[str, Any]:
    """두 읽기의 요약과 순서 분석(배치 수, 서로 다른 배치 사이의 판정 변경률, 짝지은 순서 효과), 생각 뒤 답의 완료율 문턱."""
    return {
        "readings": {reading: summarize(records, predictions, reading=reading, n_boot=n_boot, seed=seed) for reading in READINGS},
        "order": {
            "listings": listing_stats(predictions, merge), "permutation": permutation_sensitivity(predictions), "order_effects": order_effects(predictions),
        },
        "reasoning": reasoning_stats(predictions),
        "definitions": DEFINITIONS,
    }
