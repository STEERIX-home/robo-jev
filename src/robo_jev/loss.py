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
"""

from __future__ import annotations

from typing import Any

import torch

from robo_jev.contracts import LABEL_KINDS

__all__ = ["judgment_loss", "label_loss", "question_losses"]

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
    z = logits.float()
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
