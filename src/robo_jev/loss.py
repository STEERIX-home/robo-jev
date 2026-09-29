"""판단 손실 — 질문별 손실을 상태 안에서 평균내고 상태들 사이에서 평균낸다 (docs/03 §4, docs/08 §7).

**출력 layout (Judge가 낼 것과 같다).** 상태(단일 요청) 또는 틱(스트림)마다 질문별 logits
하나씩이다. logits의 k번째 값은 그 질문의 **요청 순서 k번째 후보**의 점수이며, 어느 id인지는
같은 순서의 후보 id 목록이 말한다 — :func:`robo_jev.model.serialize.serialize_request`의
``candidate_mapping``(스트림은 틱의 ``candidate_mapping``)과 같은 순서다. boolean 질문의 후보는
``["true", "false"]``, ordinal은 수준 id 순이다.

.. code-block:: python

    outputs = {
        "logits":     [{question_id: Tensor[K]}, ...],        # 상태마다, 질문 → 후보 logits
        "candidates": [{question_id: [candidate_id, ...]}, ...],  # 같은 순서의 후보 id
    }
    labels = {"labels": [[label, ...], ...]}                   # 상태마다, 계약 형태의 라벨 목록

**라벨 종류별 손실.** ``p = softmax(logits)``.

* ``valid_set`` (허용 집합 A): ``-log Σ_A p_k``. ``unknown``(U, rollout하지 않은 후보)이 있으면
  부분 라벨 손실 ``-log(Σ_A p_k / (Σ_A p_k + Σ_I p_k))``, ``I``는 A에도 U에도 없는 후보(부적합·열등
  판정). U는 정규화에서 빠지므로 gradient도 받지 않는다.
* ``single``: ``-log p_answer``. boolean의 ``answer``(bool)는 ``true``/``false`` 후보로 옮긴다.
* ``distribution``: soft-label CE ``-Σ_k q_k log p_k``.
* ``event``: ``true`` 후보의 확률 ``p``에 대한 **시행당 평균** Bernoulli NLL
  ``-(s·log p + f·log(1-p)) / (s+f)``. 반복 수가 손실 크기를 키우지 않는다. ``s+f = 0``이면
  근거가 없으므로 mask다.

``valid_set`` 라벨에 붙은 ``event_results``(후보별 rollout 성공·실패)는 선택 분포의 손실에
섞지 않는다 — 두 라벨을 한 분포로 합치지 않는다(docs/03 §4). 별도의 사건 확률 질문이 있을
때만 ``event`` 라벨로 쓴다.

**mask와 정규화.** 라벨이 없는 질문, ``mask: false`` 라벨, 근거 없는 사건 라벨은 아무것도
기여하지 않고 gradient도 받지 않는다. 라벨의 ``weight``(기본 1)로
``L_state = Σ_i w_i L_i / Σ_i w_i``를 만들고, ``L_batch``는 유효 라벨이 하나라도 있는 상태들의
평균이다. 유효 라벨이 전혀 없으면 주어진 logits에 이어진 0(스칼라)이다 — `.backward()`가 되고
gradient는 전부 0이다(상태별 microbatch에서 한 상태가 통째로 mask인 경우, docs/03 §5).

**상수 사전분포 head의 손실 — 질문별 기준선 (Task R8 A1).** 입력을 보지 않고 늘 같은 분포 π를 내는 head가 같은
라벨에서 받을 손실이다. :func:`prior_label_loss` 는 위의 식을 **그대로** 쓰되 logits 자리에 ``log π_k``(이 상태의 후보 위로
다시 정규화한 π — 후보 밖의 질량은 빠진다)를 넣는다. 그래서 모델 손실과 같은 자다: 모델이 그 π를 내면 두 값이 같다.
π를 어떻게 정하는지(배치의 라벨 주변분포)는 학습 loop가 정한다(:func:`robo_jev.train.question_table`); 라벨 하나가 그 주변분포에
보태는 몫은 :func:`label_prior_share` 다 — single·boolean은 답 하나(1), valid_set은 허용 집합에 고르게(두 값 허용이면 반씩),
distribution은 그 분포, event는 ``true``에 성공 비율 ``s/(s+f)``·나머지 후보에 실패 비율을 고르게.
"""

from __future__ import annotations

import math
from typing import Any

import torch

from robo_jev.contracts import LABEL_KINDS

__all__ = ["judgment_loss", "label_loss", "label_prior_share", "prior_label_loss", "question_losses"]

_TRUE = "true"


def _index_of(candidates: list[str], candidate_id: Any, path: str) -> int:
    if not isinstance(candidate_id, str) or candidate_id not in candidates:
        raise ValueError(f"{path}: 존재하지 않는 후보 id: {candidate_id!r} (후보: {candidates})")
    return candidates.index(candidate_id)


def _indices(candidates: list[str], ids: Any, path: str) -> list[int]:
    if not isinstance(ids, list):
        raise ValueError(f"{path}: 후보 id 목록이어야 한다 (받은 값: {type(ids).__name__})")
    return [_index_of(candidates, candidate_id, f"{path}[{i}]") for i, candidate_id in enumerate(ids)]


def _logsumexp(values: torch.Tensor, indices: list[int]) -> torch.Tensor:
    return torch.logsumexp(values[torch.as_tensor(indices, dtype=torch.long)], dim=0)


def label_loss(
    logits: torch.Tensor, candidates: list[str], label: dict, path: str = "label"
) -> torch.Tensor | None:
    """질문 하나의 손실. 기여할 수 없는 라벨(mask=false, 근거 없는 사건)은 `None`."""
    if label.get("mask", True) is False:
        return None
    kind = label.get("kind")
    if kind not in LABEL_KINDS:
        raise ValueError(f"{path}.kind: {list(LABEL_KINDS)} 중 하나여야 한다 (받은 값: {kind!r})")
    if logits.dim() != 1 or logits.shape[0] != len(candidates):
        raise ValueError(
            f"{path}: logits 길이 {tuple(logits.shape)}와 candidates 길이 {len(candidates)}가 다르다"
        )
    # FP32 loss (docs/03 §5): 반정밀도 logits는 float32로 올리고, float64는 그대로 둔다(정확성 검사용).
    z = logits.to(torch.promote_types(logits.dtype, torch.float32))
    everything = list(range(len(candidates)))

    if kind == "valid_set":
        allowed = _indices(candidates, label.get("candidate_ids"), f"{path}.candidate_ids")
        if not allowed:
            raise ValueError(f"{path}.candidate_ids: 비어 있을 수 없다")
        unknown = set(_indices(candidates, label.get("unknown", []), f"{path}.unknown"))
        kept = [index for index in everything if index not in unknown or index in allowed]
        # 정규화를 A∪I 위에서 바로 계산해야 U가 그래프에 아예 들어가지 않는다 (gradient 0).
        return _logsumexp(z, kept) - _logsumexp(z, allowed)

    log_p = torch.log_softmax(z, dim=0)

    if kind == "single":
        answer = label.get("answer")
        if isinstance(answer, bool):
            answer = _TRUE if answer else "false"
        return -log_p[_index_of(candidates, answer, f"{path}.answer")]

    if kind == "distribution":
        probabilities = label.get("probabilities")
        if not isinstance(probabilities, dict) or not probabilities:
            raise ValueError(f"{path}.probabilities: 비어 있지 않은 분포여야 한다")
        total = torch.zeros((), dtype=log_p.dtype)
        for candidate_id, probability in probabilities.items():
            index = _index_of(candidates, candidate_id, f"{path}.probabilities.{candidate_id}")
            total = total - float(probability) * log_p[index]
        return total

    # event
    successes = int(label.get("successes", 0))
    failures = int(label.get("failures", 0))
    if successes < 0 or failures < 0:
        raise ValueError(f"{path}: successes·failures는 0 이상이어야 한다")
    if successes + failures == 0:
        return None
    true_index = _index_of(candidates, _TRUE, f"{path}.successes")
    others = [index for index in everything if index != true_index]
    if not others:
        raise ValueError(f"{path}: 사건 질문에는 true 외의 후보가 필요하다 (후보: {candidates})")
    log_true = log_p[true_index]
    log_false = _logsumexp(log_p, others)
    return -(successes * log_true + failures * log_false) / (successes + failures)


def _states(outputs: dict, labels: dict) -> list[tuple[dict, dict, list[dict]]]:
    logits = outputs.get("logits")
    candidates = outputs.get("candidates")
    per_state = labels.get("labels")
    if not isinstance(logits, list) or not isinstance(candidates, list) or not isinstance(per_state, list):
        raise ValueError("outputs.logits·outputs.candidates·labels.labels는 상태별 목록이어야 한다")
    if not len(logits) == len(candidates) == len(per_state):
        raise ValueError(
            f"states: 상태 수가 다르다 (logits {len(logits)}, candidates {len(candidates)}, "
            f"labels {len(per_state)})"
        )
    return list(zip(logits, candidates, per_state))


def question_losses(outputs: dict, labels: dict) -> list[dict[str, dict[str, Any]]]:
    """상태마다 ``{question_id: {"loss": Tensor, "weight": float, "kind": str}}``. 기여 없는 라벨은 뺀다."""
    result: list[dict[str, dict[str, Any]]] = []
    for state_index, (logits, candidates, state_labels) in enumerate(_states(outputs, labels)):
        entries: dict[str, dict[str, Any]] = {}
        for label_index, label in enumerate(state_labels):
            path = f"labels[{state_index}][{label_index}]"
            question_id = label.get("question_id")
            if question_id not in logits:
                raise ValueError(f"{path}.question_id: 출력에 없는 질문이다: {question_id!r}")
            if question_id not in candidates:
                raise ValueError(f"{path}.question_id: 후보 목록에 없는 질문이다: {question_id!r}")
            loss = label_loss(logits[question_id], list(candidates[question_id]), label, path)
            if loss is None:
                continue
            weight = float(label.get("weight", 1.0))
            if weight < 0:
                raise ValueError(f"{path}.weight: 0 이상이어야 한다 (받은 값: {weight})")
            if question_id in entries:
                raise ValueError(f"{path}.question_id: 한 상태에서 같은 질문의 라벨이 둘이다: {question_id!r}")
            entries[question_id] = {"loss": loss, "weight": weight, "kind": label["kind"]}
        result.append(entries)
    return result


def _graph_zero(outputs: dict) -> torch.Tensor:
    """주어진 logits에 이어진 0 — backward가 되고 gradient는 전부 0이다.

    상태별 microbatch(docs/03 §5)에서 한 상태의 라벨이 전부 mask·근거 없음이면 손실이 0인데,
    `torch.zeros(())`는 그래프가 없어 `.backward()`가 실패한다. logits가 하나도 없을 때만
    그래프 없는 0을 돌려준다.
    """
    zero: torch.Tensor | None = None
    for state in outputs["logits"]:
        for logits in state.values():
            term = 0.0 * logits.float().sum()
            zero = term if zero is None else zero + term
    return torch.zeros(()) if zero is None else zero


def judgment_loss(outputs: dict, labels: dict) -> torch.Tensor:
    """상태 평균 손실 (모듈 설명 참조). 유효 라벨이 없으면 logits에 이어진 0."""
    state_losses: list[torch.Tensor] = []
    for entries in question_losses(outputs, labels):
        total_weight = sum(entry["weight"] for entry in entries.values())
        if not entries or total_weight <= 0:
            continue
        weighted = sum(entry["weight"] * entry["loss"] for entry in entries.values())
        state_losses.append(weighted / total_weight)
    if not state_losses:
        return _graph_zero(outputs)
    return torch.stack(state_losses).mean()


# --------------------------------------------------------------------------
# 상수 사전분포 head — 질문별 기준선 (Task R8 A1; 모듈 설명 참조)
# --------------------------------------------------------------------------


def _answer_id(label: dict) -> Any:
    answer = label.get("answer")
    if isinstance(answer, bool):
        return _TRUE if answer else "false"
    return answer


def label_prior_share(candidates: list[str], label: dict, path: str = "label") -> dict[str, float] | None:
    """라벨 하나가 배치의 라벨 주변분포에 보태는 몫 ``{후보 id: 몫}``(합 1). 모델 손실에 기여하지 않는 라벨(mask=false, 근거 없는
    사건)은 `None` — :func:`label_loss` 가 `None`을 내는 조건과 같다. 후보에 없는 id는 `ValueError`(모델 손실과 같은 검사).

    * ``single``(boolean 포함): 답 하나에 1.
    * ``valid_set``: 허용 집합 A에 고르게 ``1/|A|`` — "두 값 허용"(A = {open, closed})이면 반씩. ``unknown``은 몫을 받지 않는다.
    * ``distribution``: 그 분포(합으로 다시 나눈다).
    * ``event``: ``true``에 ``s/(s+f)``, 나머지 후보에 ``f/(s+f)``를 고르게.
    """
    if label.get("mask", True) is False:
        return None
    kind = label.get("kind")
    if kind not in LABEL_KINDS:
        raise ValueError(f"{path}.kind: {list(LABEL_KINDS)} 중 하나여야 한다 (받은 값: {kind!r})")
    if kind == "single":
        answer = _answer_id(label)
        _index_of(candidates, answer, f"{path}.answer")
        return {str(answer): 1.0}
    if kind == "valid_set":
        allowed = label.get("candidate_ids")
        _indices(candidates, allowed, f"{path}.candidate_ids")
        if not allowed:
            raise ValueError(f"{path}.candidate_ids: 비어 있을 수 없다")
        unique = list(dict.fromkeys(str(cid) for cid in allowed))
        return {cid: 1.0 / len(unique) for cid in unique}
    if kind == "distribution":
        probabilities = label.get("probabilities")
        if not isinstance(probabilities, dict) or not probabilities:
            raise ValueError(f"{path}.probabilities: 비어 있지 않은 분포여야 한다")
        for cid in probabilities:
            _index_of(candidates, cid, f"{path}.probabilities.{cid}")
        total = sum(float(p) for p in probabilities.values())
        if total <= 0:
            raise ValueError(f"{path}.probabilities: 합이 0보다 커야 한다")
        return {str(cid): float(p) / total for cid, p in probabilities.items()}
    successes = int(label.get("successes", 0))
    failures = int(label.get("failures", 0))
    if successes < 0 or failures < 0:
        raise ValueError(f"{path}: successes·failures는 0 이상이어야 한다")
    if successes + failures == 0:
        return None
    _index_of(candidates, _TRUE, f"{path}.successes")
    others = [str(cid) for cid in candidates if cid != _TRUE]
    if not others:
        raise ValueError(f"{path}: 사건 질문에는 true 외의 후보가 필요하다 (후보: {candidates})")
    share = {_TRUE: successes / (successes + failures)}
    for cid in others:
        share[cid] = (failures / (successes + failures)) / len(others)
    return share


def _log_ratio(numerator: float, denominator: float) -> float:
    """``log(numerator / denominator)`` — 분자가 0이면 −inf (호출자가 계수 0인 항은 부르지 않는다)."""
    if numerator <= 0.0:
        return -math.inf
    return math.log(numerator / denominator)


def prior_label_loss(prior: dict[str, float], candidates: list[str], label: dict, path: str = "label") -> float | None:
    """늘 같은 분포 `prior`를 내는 **상수 head**가 이 라벨에서 받을 손실 — :func:`label_loss` 와 같은 식에 logits 대신
    ``log(prior_k / Σ_{j∈후보} prior_j)``를 넣은 값(모듈 설명). `prior`에서 이 상태의 후보가 아닌 id는 무시하고, 후보인데 `prior`에
    없는 id는 0이다. 기여하지 않는 라벨은 `None`. 계수가 0인 항(사건의 성공 0·실패 0, 분포의 0 확률)은 ``0·log 0 = 0``으로 읽는다.

    값은 Python float(float64)이다 — 학습 그래프에 붙지 않는다(기록용).
    """
    if label.get("mask", True) is False:
        return None
    kind = label.get("kind")
    if kind not in LABEL_KINDS:
        raise ValueError(f"{path}.kind: {list(LABEL_KINDS)} 중 하나여야 한다 (받은 값: {kind!r})")
    mass = {str(cid): max(0.0, float(prior.get(cid, 0.0))) for cid in candidates}
    total = sum(mass.values())
    if total <= 0.0:
        raise ValueError(f"{path}: 사전분포가 이 상태의 후보({candidates})에 질량을 두지 않는다")

    if kind == "valid_set":
        ids = label.get("candidate_ids")
        _indices(candidates, ids, f"{path}.candidate_ids")  # 후보에 없는 id는 모델 손실과 같이 거절
        allowed = [str(cid) for cid in ids]
        if not allowed:
            raise ValueError(f"{path}.candidate_ids: 비어 있을 수 없다")
        unknown_ids = label.get("unknown", [])
        _indices(candidates, unknown_ids, f"{path}.unknown")
        unknown = {str(cid) for cid in unknown_ids}
        kept = [cid for cid in candidates if cid not in unknown or cid in allowed]
        return -_log_ratio(sum(mass[cid] for cid in set(allowed)), sum(mass[cid] for cid in kept))
    if kind == "single":
        answer = _answer_id(label)
        _index_of(candidates, answer, f"{path}.answer")
        return -_log_ratio(mass[str(answer)], total)
    if kind == "distribution":
        probabilities = label.get("probabilities")
        if not isinstance(probabilities, dict) or not probabilities:
            raise ValueError(f"{path}.probabilities: 비어 있지 않은 분포여야 한다")
        value = 0.0
        for cid, probability in probabilities.items():
            _index_of(candidates, cid, f"{path}.probabilities.{cid}")
            if float(probability) != 0.0:
                value -= float(probability) * _log_ratio(mass[str(cid)], total)
        return value
    successes = int(label.get("successes", 0))
    failures = int(label.get("failures", 0))
    if successes < 0 or failures < 0:
        raise ValueError(f"{path}: successes·failures는 0 이상이어야 한다")
    if successes + failures == 0:
        return None
    _index_of(candidates, _TRUE, f"{path}.successes")
    true_mass = mass[_TRUE]
    value = 0.0
    if successes:
        value -= successes * _log_ratio(true_mass, total)
    if failures:
        value -= failures * _log_ratio(total - true_mass, total)
    return value / (successes + failures)
