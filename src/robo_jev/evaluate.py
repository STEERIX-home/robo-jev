"""평가 — 질문별 정확도·허용 집합 적중·NLL·Brier, 위치 편향 지표, 문맥 섞기 대조군, 규칙 기준군 (docs/03 §7-4, docs/06 Task 2c, G0b S3).

:func:`evaluate_items` 는 :class:`robo_jev.model.judge.Judge` 를 :class:`robo_jev.sampler.Item` 목록(단일 요청 microbatch,
스트림은 에피소드 재생)에 돌려 상태(틱)마다 질문별 확률을 얻고 라벨과 대조한다. 표는 **질문 id별**(로봇 세트 v0의 10개)과
**질문 타입별**(비로봇: choice/boolean/ordinal)이다.

* ``accuracy`` — `single`은 argmax == 답, `valid_set`은 argmax ∈ A(허용 집합 적중), `distribution`은 argmax == 분포의 argmax.
  `event` 라벨은 정확도가 없다(확률 질문).
* ``nll`` — :func:`robo_jev.loss.label_loss` (라벨 종류별 손실) 그대로.
* ``brier`` — 목표 질량을 라벨이 허용하는 후보에 둔 제곱 오차: `single` Σ(p−onehot)², `valid_set` (Σ_A p − 1)² + Σ_{k∉A} p_k²,
  `distribution` Σ(p−q)², `event` (p_true − s/(s+f))².
* 위치 편향 (analysis-nimble §3-3, docs/03 §3): 후보가 둘 이상인 `choice` 질문에서 **선택한 위치의 분포**(첫 위치 비율·
  위치별 수)와, `shuffle_seed`로 후보 순서를 치환한 레코드를 다시 평가했을 때의 **답 변경률**(예측 id가 바뀐 비율).
* 문맥 섞기 대조군 (docs/03 §6): 분할 안에서 문맥을 한 칸 굴린 레코드 — **상태 섞기**가 표준 열(`context_shuffle`,
  `context_shuffle_kind = "state"`)이다. 비로봇은 `request.state`를 다음 레코드의 것으로 바꾼다(남는 것은 후보뿐이라 높으면
  후보만으로 답이 나오는 단축 경로다). 로봇 스트림은 틱마다 **구조화된 상태**(`goal` 줄 전체 — target·zone·forbid·fragile·
  텍스트 —, 물체·영역·장면 줄, 물체별 파생 값)와 prefix의 지시 텍스트를 다음 에피소드의 같은 색인 틱(마지막 틱으로 clamp)의
  것으로 바꾸되, 기증 에피소드의 물체·영역 id를 **이 틱의 id에 자리 순서로 다시 매핑**해 상태 안의 참조(`target_ref`·
  `forbidden_contact`·`fragile`·`target_zone`·`derived[].object`)가 서로 맞게 두고, **질문·후보·commitment·실행 이력·robot·
  exec 줄은 그대로 둔다**(G0b 리뷰 1 I3 / D1 리뷰 1 I1: 후보 키가 가리키는 id는 상태에 있지만 그 물체·목표는 다른 에피소드의
  것이다). 이 값이 모델과 같으면 모델이 목표·장면을 읽지 않고 후보 줄(+ 자기 실행 상태)만으로 답한다는 뜻이다.
  **지시 섞기**(`instruction_shuffle`, kind `instruction`; 로봇 스트림만)는 지시·목표 **텍스트**만 굴리고 구조화된 goal·물리
  상태·후보를 그대로 두는 둘째 열이다 — 상태에 달린 답(경로·그리퍼·속도·힘·완료·정지)은 여기서도 맞는 것이 정상이고, `goal`
  줄의 `target=`·`zone=`이 남아 있으므로 이 열과 같다는 것은 "텍스트가 불필요하다"는 뜻일 뿐 "문맥이 불필요하다"는 뜻이 아니다.
  **commitment 섞기**(`commitment_shuffle`, kind `state_commitment`; P3 B2)는 상태 섞기 **에 더해** 이 틱의
  `commitment.action_ref`를 이 틱의 다른 후보로 옮기는 셋째 대조군이다 — 표준 열은 그 줄을 일부러 남기고, 이 데이터에서는
  정답이 바로 그 id인 틱이 다수라 표준 열이 답을 그대로 베껴 넘긴다. 표준 열은 P1·P2와의 비교를 위해 **바꾸지 않는다**.
  이 열은 부가 질문 라벨의 `conditioned_on`도 함께 옮기므로(계약 검사) **`q_main`에서만 읽는다**.
* 규칙 기준군 (docs/02, `robo_jev.harness.rule_judge`): 로봇 틱마다 규칙 판단기의 10개 답을 같은 후보 목록 위의 확률로
  바꿔 같은 지표를 낸다 — 하네스만으로 풀리는 범위의 기준. 이 함수만 하네스를 import하므로 :mod:`robo_jev.train` 은
  이 모듈을 import하지 않는다(docs/06 §1의 경계는 학습 코드 쪽에 둔다).
* **기계적 기준군** (`mechanical_baseline`, P3 B3): "commitment가 있으면 그것, 없으면 `observe`" — 대조군이 보존하는
  필드만 읽는 정책의 답(:func:`mechanical_baseline_predictions`). 규칙 판정기·소형 scorer와 나란한 상시 기준선이고,
  **이 열을 넘지 못하는 모델 주장은 주장이 아니다**. GPU pass가 필요 없다.
* **편 단위 불확실성** (Task P2 B): 틱은 편(에피소드) 안에서 상관되어 있어 독립 단위는 틱이 아니라 편이다. 그래서
  :func:`aggregate` 가 질문 칸마다 편 단위 집계(``per_episode``)를 표에 남기고, :func:`episode_bootstrap` 이 편을
  표본 단위로 재표집해 정확도의 구간과 **대조군 대비 여유의 쌍 구간**을 낸다 — 그 구간이 0을 포함하는 여유는
  판정이 아니다. (P1은 `_predictions`를 표를 쓰기 전에 버려서 이 수를 산출물로 낼 수 없었다.) 편마다 크기가 다르므로
  같은 재표집에서 **틱 가중 평균과 편 균등 평균을 둘 다** 낸다(`episode_balanced_*`; P3 A3) — 둘이 갈리면 그 사실이
  결과의 일부다.
"""

from __future__ import annotations

import copy
import json
import math
import random
import time
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch

from robo_jev.contracts import SCHEMA_SINGLE_REQUEST, SCHEMA_STREAM
from robo_jev.loss import label_loss
from robo_jev.sampler import Item, permute_candidates

__all__ = [
    "aggregate",
    "answer_stability",
    "calibration_error",
    "column_event_metrics",
    "context_shuffle_records",
    "donor_rotation",
    "contrast_pair_check",
    "episode_bootstrap",
    "evaluate_items",
    "evaluate_suite",
    "eval_suite_identity",
    "holding_twin_preference",
    "label_metrics",
    "load_eval_suite",
    "load_suite_items",
    "mechanical_baseline_predictions",
    "predict_items",
    "reaction_delay",
    "rule_judge_predictions",
    "selective_metrics",
    "selective_metrics_from_stored",
    "split_episode_bootstrap",
    "stop_timing",
    "tiny_scorer_column",
]

#: 게이트 후보의 의미 키 (docs/08 §4) — 선택적 지표에서 abstention으로 읽는다. 하네스를 import하지 않고 여기 둔다(`FIXED_KEYS`와 같다).
_GATE_KEYS = ("observe", "hold", "replan")

_TRUE, _FALSE = "true", "false"


# --------------------------------------------------------------------------
# 예측
# --------------------------------------------------------------------------


def predict_items(judge: Any, items: list[Item], *, tokens_per_batch: int = 8192, fused: bool = False) -> list[dict[str, Any]]:
    """상태(틱)마다 ``{"record_id", "tick", "kind", "probabilities": {qid: Tensor[K]}, "candidates": {qid: [id…]}, "labels": […],
    "question_types": {qid: type}}``. 단일 요청은 토큰 예산까지 묶은 microbatch로, 스트림은 에피소드마다 재생한다.

    ``fused``는 스트림 재생에서 틱 몸통과 결정 분기를 한 forward로 돌린다(실제 backbone의 서빙 기본 구성; BF16 허용 오차
    안에서 같은 값, 층마다 가중치를 한 번만 읽는다 — :func:`robo_jev.model.stream.replay_layout`)."""
    out: list[dict[str, Any]] = []
    singles = [item for item in items if item.kind == "single"]
    streams = [item for item in items if item.kind == "stream"]
    with torch.no_grad():
        batch: list[Item] = []
        budget = 0
        for item in singles + [None]:  # type: ignore[list-item]
            if item is not None and (not batch or budget + item.tokens <= tokens_per_batch):
                batch.append(item)
                budget += item.tokens
                continue
            if batch:
                result = judge({"layout": "state_first", "states": [b.layout for b in batch]})
                for position, b in enumerate(batch):
                    out.append(
                        {
                            "record_id": b.record_id, "tick": None, "kind": "single", "split": b.split,
                            "group": str(b.record.get("origin_group") or b.record_id),  # 편 단위 집계의 묶음 (B1)
                            "probabilities": {qid: torch.softmax(z.detach().float().cpu(), 0) for qid, z in result["logits"][position].items()},
                            "candidates": result["candidates"][position], "labels": list(b.record.get("labels", [])),
                            "question_types": dict(b.question_types),
                        }
                    )
            batch, budget = ([item], item.tokens) if item is not None else ([], 0)
        for item in streams:
            result = judge({"layout": "stream_l1a", "stream": item.layout, "fused": fused})
            for index, (logits, candidates) in enumerate(zip(result["logits"], result["candidates"])):
                out.append(
                    {
                        "record_id": item.record_id, "tick": index, "kind": "stream", "split": item.split,
                        "group": item.record_id,  # 스트림의 편 = 에피소드 (B1)
                        "probabilities": {qid: torch.softmax(z.detach().float().cpu(), 0) for qid, z in logits.items()},
                        "candidates": candidates, "labels": list(item.record["ticks"][index].get("labels", [])),
                        "question_types": dict(item.question_types),
                    }
                )
    return out


def rule_judge_predictions(items: list[Item]) -> list[dict[str, Any]]:
    """로봇 스트림 틱마다 규칙 기준군의 답을 같은 후보 목록 위의 확률로 (모델 예측과 같은 꼴). 비로봇 레코드는 건너뛴다."""
    from robo_jev.harness.rule_judge import rule_judge  # 하네스는 여기서만

    out: list[dict[str, Any]] = []
    for item in items:
        if item.kind != "stream":
            continue
        for index, tick in enumerate(item.record["ticks"]):
            answers = rule_judge(tick["request"])
            entry = item.layout["ticks"][index]
            probabilities: dict[str, torch.Tensor] = {}
            for qid, ids in entry["candidate_mapping"].items():
                answer = answers.get(qid)
                if isinstance(answer, dict):
                    vector = torch.tensor([float(answer.get(cid, 0.0)) for cid in ids])
                elif isinstance(answer, (int, float)) and set(ids) == {_TRUE, _FALSE}:
                    vector = torch.tensor([float(answer) if cid == _TRUE else 1.0 - float(answer) for cid in ids])
                else:
                    continue
                total = float(vector.sum())
                probabilities[qid] = vector / total if total > 0 else torch.full((len(ids),), 1.0 / len(ids))
            out.append(
                {
                    "record_id": item.record_id, "tick": index, "kind": "stream", "split": item.split, "group": item.record_id,
                    "probabilities": probabilities,
                    "candidates": {qid: list(ids) for qid, ids in entry["candidate_mapping"].items() if qid in probabilities},
                    "labels": list(tick.get("labels", [])), "question_types": dict(item.question_types),
                }
            )
    return out


#: 기계적 기준군이 답하는 질문. 이 정책은 `q_main`에만 정의된다 — 다른 질문에는 대응하는 "하던 것" 필드가 없다.
MECHANICAL_BASELINE_QUESTION = "q_main"
#: 그 정책을 한 줄로 (보고서·표에 그대로 싣는다).
MECHANICAL_BASELINE_POLICY = (
    "answer the tick's own commitment.action_ref if there is one, otherwise the observe gate key "
    "— both are fields the state shuffle preserves, so this column reads nothing"
)

#: commitment 섞기 열을 **어디서 읽을 수 있는가**. 열 옆에 같이 실린다 — 범위가 파일 밖에 있으면
#: 읽을 수 없는 칸의 "0을 제외한다"가 기계가 읽는 기록으로 남는다 (P3 B2, 리뷰 1 I3).
COMMITMENT_SHUFFLE_QUESTION = "q_main"
COMMITMENT_SHUFFLE_SCOPE = (
    "interpretable on q_main only, and there on the non-commitment (primary) stratum. The roll moves this tick's "
    "commitment reference to another candidate of the same tick and rewrites every auxiliary label's conditioned_on "
    "with it, so the auxiliary answers — which are the expert's answers under the ORIGINAL commitment — are falsified "
    "rather than made goal-blind. On q_main's commitment stratum the stored label IS the original commitment, so it "
    "is falsified there too. No margin is reported for the questions this column cannot be read on."
)


def _tick_commitment(request: dict[str, Any]) -> str | None:
    """이 틱의 commitment 참조 — 모델이 보는 자리(`request.commitment`)를 먼저 읽고 없으면 상태 쪽."""
    for source in (request.get("commitment"), (request.get("state") or {}).get("commitment")):
        if isinstance(source, dict) and source.get("action_ref") is not None:
            return str(source["action_ref"])
    return None


def mechanical_baseline_predictions(items: list[Item]) -> list[dict[str, Any]]:
    """**상시 기준군 열** — 아무것도 읽지 않는 정책의 답 (:data:`MECHANICAL_BASELINE_POLICY`), 모델 예측과 같은 꼴.

    왜 열인가 (P3 B3). 판정 칸의 표준 대조군(상태 섞기)은 `commitment`·`exec`·실행 이력을 **일부러 남긴다** — 그것이
    그 틱의 실행 이력이기 때문이다. 그런데 이 데이터에서는 정답이 바로 그 `commitment.action_ref`인 틱이 다수라,
    "commitment가 있으면 그것, 없으면 `observe`"라는 정책이 아무 판단 없이 높은 값을 받는다 (P2 판정 칸에서 0.890).
    그러므로 **어떤 모델 주장도 이 열을 넘지 못하면 주장이 아니다**. 규칙 판정기(목표를 읽는다)·소형 scorer(패턴)와
    나란한 셋째 기준선이고, 이 열도 편 단위 집계를 남겨 같은 쌍 부트스트랩에 들어간다.

    `q_main`에만 답한다 — 다른 질문에는 대응하는 "하던 것 계속하기" 필드가 없다. commitment 참조가 이 틱의 후보
    목록에 없으면(하네스는 자리를 예약하므로 D1에서는 0건이다) `observe` 키로 물러난다."""
    out: list[dict[str, Any]] = []
    for item in items:
        if item.kind != "stream":
            continue
        for index, tick in enumerate(item.record["ticks"]):
            mapping = item.layout["ticks"][index]["candidate_mapping"].get(MECHANICAL_BASELINE_QUESTION)
            if not mapping:
                continue
            ids = list(mapping)
            commitment = _tick_commitment(tick["request"])
            chosen = commitment if commitment in ids else None
            if chosen is None:
                entries = tick["request"].get("candidates", {}).get(MECHANICAL_BASELINE_QUESTION) or []
                keys = {str(entry.get("id")): str(entry.get("key") or "") for entry in entries if isinstance(entry, dict)}
                chosen = next((cid for cid in ids if keys.get(cid, "").startswith("observe")), None)
            vector = torch.zeros(len(ids))
            if chosen is None:
                vector += 1.0 / len(ids)  # 답할 것이 없으면 균등 — 정책이 정의되지 않은 틱은 점수를 받지 않는다는 뜻이다
            else:
                vector[ids.index(chosen)] = 1.0
            out.append(
                {
                    "record_id": item.record_id, "tick": index, "kind": "stream", "split": item.split, "group": item.record_id,
                    "probabilities": {MECHANICAL_BASELINE_QUESTION: vector},
                    "candidates": {MECHANICAL_BASELINE_QUESTION: ids},
                    "labels": [label for label in tick.get("labels", []) if label.get("question_id") == MECHANICAL_BASELINE_QUESTION],
                    "question_types": dict(item.question_types),
                }
            )
    return out


# --------------------------------------------------------------------------
# 지표
# --------------------------------------------------------------------------


def label_metrics(probabilities: torch.Tensor, candidates: list[str], label: dict) -> dict[str, Any] | None:
    """라벨 하나의 지표 ``{"correct": bool|None, "nll": float, "brier": float, "predicted": id, "position": int}``. 기여 없는 라벨은 None."""
    p = probabilities.float()
    z = torch.log(p.clamp_min(1e-12))
    loss = label_loss(z, candidates, label)
    if loss is None:
        return None
    kind = label["kind"]
    position = int(p.argmax())
    predicted = candidates[position]
    onehot = torch.zeros_like(p)
    if kind == "single":
        answer = label["answer"]
        answer = (_TRUE if answer else _FALSE) if isinstance(answer, bool) else answer
        onehot[candidates.index(answer)] = 1.0
        correct: bool | None = predicted == answer
        brier = float(((p - onehot) ** 2).sum())
    elif kind == "valid_set":
        allowed = set(label["candidate_ids"])
        inside = sum(float(p[i]) for i, cid in enumerate(candidates) if cid in allowed)
        outside = sum(float(p[i]) ** 2 for i, cid in enumerate(candidates) if cid not in allowed)
        correct = predicted in allowed
        brier = (inside - 1.0) ** 2 + outside
    elif kind == "distribution":
        q = torch.tensor([float(label["probabilities"].get(cid, 0.0)) for cid in candidates])
        correct = predicted == candidates[int(q.argmax())]
        brier = float(((p - q) ** 2).sum())
    else:  # event
        s, f = int(label.get("successes", 0)), int(label.get("failures", 0))
        correct = None
        brier = (float(p[candidates.index(_TRUE)]) - s / (s + f)) ** 2
    return {"correct": correct, "nll": float(loss), "brier": brier, "predicted": predicted, "position": position, "kind": kind}


def _table_key(prediction: dict[str, Any], qid: str) -> str:
    return qid if prediction["kind"] == "stream" else prediction["question_types"].get(qid, "unknown")


def _per_episode(groups: dict[str, list[int]]) -> list[dict[str, Any]]:
    """편 단위 집계를 표에 남기는 꼴 — ``[{episode_id, n, graded, correct}, …]``(편 이름 순).

    레코드별 예측을 다 저장하지 않는다(P1은 그것을 버려서 편 단위 구간을 낼 수 없었다). 이 네 수만 있으면
    편을 표본 단위로 재표집하는 부트스트랩(:func:`episode_bootstrap`)이 그대로 돌아간다."""
    return [
        {"episode_id": name, "n": counts[0], "graded": counts[1], "correct": counts[2]}
        for name, counts in sorted(groups.items())
    ]


def aggregate(predictions: list[dict[str, Any]], *, store_predictions: bool | Sequence[str] = False) -> dict[str, Any]:
    """예측 목록 → 질문(id 또는 타입)별 ``{n, accuracy, nll, brier, first_position_rate, position_counts, per_episode}``와 전체.

    ``per_episode``는 **편 단위 집계**다(로봇 스트림은 에피소드, 비로봇·대조 단일은 `origin_group`; 예측의 `group`
    키, 없으면 `record_id`). 틱은 편 안에서 상관되어 있어 독립 단위는 편이므로, 이 목록이 있어야 표의 어떤 칸에도
    편 단위 구간을 붙일 수 있다 (P2 B1).

    ``store_predictions``면 질문 칸마다 ``per_record``(``{record_id, tick, question, predicted, correct}``)도 남긴다
    — 편 단위 집계로는 **부분 모집단을 다시 고를 수 없어서**, "라벨이 지금 commitment가 아닌 틱만" 같은 물음이 전부
    GPU 재실행이 됐다 (P2 리뷰 1 I3). ``True``면 모든 질문 칸, 이름 목록이면 **그 칸만**이다 — 판정 칸 하나는 열당
    844줄이지만 이 분할의 질문 칸은 열 개라 다 켜면 같은 파일이 0.10 MB에서 **4.90 MB**가 된다(실측). 물음이 있는 칸만 켠다."""
    wanted = None if isinstance(store_predictions, bool) else {str(name) for name in store_predictions}
    rows: dict[str, dict[str, Any]] = {}
    totals: dict[str, list[int]] = {}
    for prediction in predictions:
        group = str(prediction.get("group") or prediction["record_id"])
        for label in prediction["labels"]:
            qid = label.get("question_id")
            if qid not in prediction["probabilities"]:
                continue
            metrics = label_metrics(prediction["probabilities"][qid], list(prediction["candidates"][qid]), label)
            if metrics is None:
                continue
            key = _table_key(prediction, qid)
            row = rows.setdefault(key, {"n": 0, "correct": 0, "graded": 0, "nll": 0.0, "brier": 0.0, "positions": Counter(), "choice_n": 0, "groups": {}, "records": []})
            row["n"] += 1
            if store_predictions and (wanted is None or key in wanted):
                row["records"].append({
                    "record_id": prediction["record_id"], "tick": prediction.get("tick"), "question": qid,
                    "predicted": metrics["predicted"], "correct": None if metrics["correct"] is None else bool(metrics["correct"]),
                })  # fmt: skip
            row["nll"] += metrics["nll"]
            row["brier"] += metrics["brier"]
            counts = row["groups"].setdefault(group, [0, 0, 0])
            total_counts = totals.setdefault(group, [0, 0, 0])
            counts[0] += 1
            total_counts[0] += 1
            if metrics["correct"] is not None:
                row["graded"] += 1
                row["correct"] += int(metrics["correct"])
                counts[1] += 1
                counts[2] += int(metrics["correct"])
                total_counts[1] += 1
                total_counts[2] += int(metrics["correct"])
            if prediction["question_types"].get(qid) == "choice" and len(prediction["candidates"][qid]) >= 2:
                row["positions"][metrics["position"]] += 1
                row["choice_n"] += 1
    table: dict[str, Any] = {}
    total = {"n": 0, "correct": 0, "graded": 0, "nll": 0.0, "brier": 0.0}
    for key, row in sorted(rows.items()):
        table[key] = {
            "n": row["n"],
            "accuracy": (row["correct"] / row["graded"]) if row["graded"] else None,
            "graded": row["graded"],
            "nll": row["nll"] / row["n"],
            "brier": row["brier"] / row["n"],
            "first_position_rate": (row["positions"][0] / row["choice_n"]) if row["choice_n"] else None,
            "position_counts": [row["positions"][i] for i in range(max(row["positions"]) + 1)] if row["positions"] else [],
            "per_episode": _per_episode(row["groups"]),
        }
        if store_predictions and (wanted is None or key in wanted):
            table[key]["per_record"] = row["records"]  # 합계 칸(`_all`)에는 두지 않는다 — 질문 칸의 합이다
        for name in ("n", "correct", "graded", "nll", "brier"):
            total[name] += row[name]
    table["_all"] = {
        "n": total["n"],
        "accuracy": (total["correct"] / total["graded"]) if total["graded"] else None,
        "graded": total["graded"],
        "nll": (total["nll"] / total["n"]) if total["n"] else None,
        "brier": (total["brier"] / total["n"]) if total["n"] else None,
        "per_episode": _per_episode(totals),
    }
    return table


# --------------------------------------------------------------------------
# 편 단위 부트스트랩 (P2 B2·B3)
# --------------------------------------------------------------------------

#: 편 단위 부트스트랩의 재표집 수·seed·신뢰 수준. **평가 집합 설정이 아니라 이 모듈의 상수다** — 설정에 넣으면
#: :func:`eval_suite_identity` 의 payload가 바뀌어 P1이 낸 해시(`79d09793eab5…`)와 나란히 놓을 수 없게 된다.
EPISODE_BOOTSTRAP = {"resamples": 2000, "seed": 20260921, "level": 0.95}


def _quantile(sorted_values: list[float], q: float) -> float:
    """정렬된 표본의 선형 보간 분위수 (numpy 없이 — 이 모듈은 tensor 말고는 순수 Python이다)."""
    position = q * (len(sorted_values) - 1)
    low = int(math.floor(position))
    high = min(low + 1, len(sorted_values) - 1)
    weight = position - low
    return sorted_values[low] * (1.0 - weight) + sorted_values[high] * weight


def episode_bootstrap(
    model_rows: list[dict[str, Any]] | None,
    control_rows: list[dict[str, Any]] | None = None,
    *,
    resamples: int = EPISODE_BOOTSTRAP["resamples"],
    seed: int = EPISODE_BOOTSTRAP["seed"],
    level: float = EPISODE_BOOTSTRAP["level"],
) -> dict[str, Any] | None:
    """편을 표본 단위로 재표집한 정확도(와, 대조군을 주면 **짝지은** 여유)의 부트스트랩 구간.

    입력은 :func:`aggregate` 가 남긴 ``per_episode`` 목록이다. 틱은 편 안에서 상관되어 있으므로 독립 단위는 틱이
    아니라 편이다 — 편 |G|개를 복원추출하고 그 편들의 ``Σ correct / Σ graded``를 다시 센다(편마다 틱 수가 다른
    것이 재표집에 그대로 들어온다). ``control_rows``를 주면 **같은 재표집 안에서** 모델과 대조군을 함께 세어
    그 차이의 구간을 낸다(**쌍 부트스트랩**): 두 열은 같은 편에서 나왔으므로 편의 난이도가 차이에서 상쇄된다.
    ``margin_includes_zero``가 참이면 그 여유는 판정이 아니다.
    """
    model = {str(row["episode_id"]): row for row in (model_rows or [])}
    groups = sorted(name for name in model if model[name].get("graded"))
    if not groups:
        return None
    control = {str(row["episode_id"]): row for row in control_rows} if control_rows else None
    paired = control is not None and all(name in control for name in groups)

    def _accuracy(source: dict[str, dict[str, Any]], names: list[str]) -> float | None:
        graded = sum(int(source[name]["graded"]) for name in names if name in source)
        correct = sum(int(source[name]["correct"]) for name in names if name in source)
        return (correct / graded) if graded else None

    def _balanced(source: dict[str, dict[str, Any]], names: list[str]) -> float | None:
        """**편 균등 평균** — 편마다 자기 정확도를 내고 그것들을 평균한다 (편이 길든 짧든 한 표) (P3 A3)."""
        values = [
            int(source[name]["correct"]) / int(source[name]["graded"])
            for name in names
            if name in source and int(source[name]["graded"])
        ]
        return (sum(values) / len(values)) if values else None

    accuracies: list[float] = []
    margins: list[float] = []
    balanced_accuracies: list[float] = []
    balanced_margins: list[float] = []
    rng = random.Random(seed)
    size = len(groups)
    for _ in range(int(resamples)):
        drawn = [groups[rng.randrange(size)] for _ in range(size)]
        value = _accuracy(model, drawn)
        if value is None:
            continue
        accuracies.append(value)
        balanced = _balanced(model, drawn)
        if balanced is not None:
            balanced_accuracies.append(balanced)
        if paired:
            other = _accuracy(control, drawn)
            if other is not None:
                margins.append(value - other)
            other_balanced = _balanced(control, drawn)
            if balanced is not None and other_balanced is not None:
                balanced_margins.append(balanced - other_balanced)
    accuracies.sort()
    balanced_accuracies.sort()
    low, high = (1.0 - level) / 2.0, 1.0 - (1.0 - level) / 2.0
    accuracy = _accuracy(model, groups)
    balanced_accuracy = _balanced(model, groups)
    out: dict[str, Any] = {
        "episodes": size, "n": sum(int(model[name]["n"]) for name in groups),
        "graded": sum(int(model[name]["graded"]) for name in groups),
        "resamples": int(resamples), "seed": int(seed), "level": level, "unit": "episode",
        "accuracy": accuracy,
        "accuracy_ci": [_quantile(accuracies, low), _quantile(accuracies, high)] if accuracies else None,
        # 편마다 크기가 다르므로 **틱 가중**(위)과 **편 균등**(아래) 두 평균을 함께 낸다 — 갈리면 그 사실이 결과다 (P3 A3).
        "episode_balanced_accuracy": balanced_accuracy,
        "episode_balanced_accuracy_ci": [_quantile(balanced_accuracies, low), _quantile(balanced_accuracies, high)] if balanced_accuracies else None,
    }
    if out["accuracy_ci"] is not None:
        out["accuracy_half_width"] = (out["accuracy_ci"][1] - out["accuracy_ci"][0]) / 2.0
    if paired and margins:
        margins.sort()
        control_accuracy = _accuracy(control, groups)
        interval = [_quantile(margins, low), _quantile(margins, high)]
        out.update({
            "control_accuracy": control_accuracy,
            "margin": None if (accuracy is None or control_accuracy is None) else accuracy - control_accuracy,
            "margin_ci": interval,
            "margin_half_width": (interval[1] - interval[0]) / 2.0,
            "margin_includes_zero": bool(interval[0] <= 0.0 <= interval[1]),
        })  # fmt: skip
    if paired and balanced_margins:
        balanced_margins.sort()
        control_balanced = _balanced(control, groups)
        interval = [_quantile(balanced_margins, low), _quantile(balanced_margins, high)]
        out.update({
            "episode_balanced_control_accuracy": control_balanced,
            "episode_balanced_margin": None if (balanced_accuracy is None or control_balanced is None) else balanced_accuracy - control_balanced,
            "episode_balanced_margin_ci": interval,
            "episode_balanced_margin_includes_zero": bool(interval[0] <= 0.0 <= interval[1]),
        })  # fmt: skip
    return out


def answer_change_rate(original: list[dict[str, Any]], permuted: list[dict[str, Any]]) -> dict[str, Any]:
    """같은 (레코드, 틱, 질문)의 예측 id가 원래 순서와 치환한 순서에서 다른 비율 (후보 둘 이상인 choice 질문)."""
    index = {(p["record_id"], p["tick"]): p for p in permuted}
    changed = compared = 0
    by_key: dict[str, list[int]] = {}
    for prediction in original:
        other = index.get((prediction["record_id"], prediction["tick"]))
        if other is None:
            continue
        for qid, p in prediction["probabilities"].items():
            if prediction["question_types"].get(qid) != "choice" or len(prediction["candidates"][qid]) < 2 or qid not in other["probabilities"]:
                continue
            a = prediction["candidates"][qid][int(p.argmax())]
            b = other["candidates"][qid][int(other["probabilities"][qid].argmax())]
            compared += 1
            changed += int(a != b)
            by_key.setdefault(_table_key(prediction, qid), [0, 0])
            by_key[_table_key(prediction, qid)][0] += int(a != b)
            by_key[_table_key(prediction, qid)][1] += 1
    return {
        "compared": compared,
        "rate": (changed / compared) if compared else None,
        "by_question": {key: (c / n if n else None) for key, (c, n) in sorted(by_key.items())},
    }


def calibration_error(predictions: list[dict[str, Any]], *, bins: int = 10) -> dict[str, Any]:
    """기대 보정 오차(ECE): 채점 가능한 라벨마다 argmax 확률(확신)과 정답 여부를 `bins`개 등간격 구간에 모아
    Σ_b (n_b / N)·|acc_b − conf_b|. `distribution`은 분포의 argmax와, `valid_set`은 허용 집합 적중과 견준다. 온도 미보정 값이다."""
    edges = [index / bins for index in range(bins + 1)]
    table = [{"low": edges[i], "high": edges[i + 1], "n": 0, "correct": 0, "confidence": 0.0} for i in range(bins)]
    for prediction in predictions:
        for label in prediction["labels"]:
            qid = label.get("question_id")
            if qid not in prediction["probabilities"]:
                continue
            metrics = label_metrics(prediction["probabilities"][qid], list(prediction["candidates"][qid]), label)
            if metrics is None or metrics["correct"] is None:
                continue
            confidence = float(prediction["probabilities"][qid].max())
            slot = min(bins - 1, int(confidence * bins))
            table[slot]["n"] += 1
            table[slot]["correct"] += int(metrics["correct"])
            table[slot]["confidence"] += confidence
    total = sum(row["n"] for row in table)
    ece = 0.0
    for row in table:
        if row["n"]:
            ece += row["n"] / total * abs(row["correct"] / row["n"] - row["confidence"] / row["n"])
    return {
        "ece": ece if total else None,
        "n": total,
        "bins": [{"range": [round(row["low"], 2), round(row["high"], 2)], "n": row["n"], "accuracy": (row["correct"] / row["n"]) if row["n"] else None,
                  "confidence": (row["confidence"] / row["n"]) if row["n"] else None} for row in table],
    }


def selective_metrics(predictions: list[dict[str, Any]], records: list[dict], *, stop_threshold: float = 0.5) -> dict[str, Any]:
    """선택적 지표 (docs/08 §10): 로봇 스트림 틱의 `q_main` 예측에서 게이트 후보(observe·hold·replan)를 **abstention**으로 읽어
    coverage(행동한 비율)·abstention 비율·selective accuracy(행동한 틱의 허용 집합 적중)·wrong target(행동했는데 대상이 허용 집합의
    어느 대상과도 다른 비율)·**unsafe action rate**(행동한 틱 가운데 대상이 금지 접촉 물체이거나, 정답이 정지인데 `q_stop`을
    `stop_threshold` 아래로 답한 비율)를 낸다. `records`는 예측의 `record_id`(episode_id)로 찾는 스트림 레코드다(후보의 의미 키·금지
    목록은 레코드에서 읽는다). 비로봇 예측은 건너뛴다."""
    from robo_jev.model.serialize import joint_key_parts

    by_id = {record.get("episode_id"): record for record in records if record.get("schema_version") == SCHEMA_STREAM}
    n = acted = abstained = correct_acted = wrong_target = unsafe = forbidden_target = stop_ignored = 0
    by_gate: Counter = Counter()
    for prediction in predictions:
        if prediction["kind"] != "stream" or "q_main" not in prediction["probabilities"]:
            continue
        record = by_id.get(prediction["record_id"])
        if record is None:
            continue
        tick = record["ticks"][int(prediction["tick"])]
        entries = {str(entry["id"]): entry for entry in tick["request"]["candidates"]["q_main"]}
        ids = list(prediction["candidates"]["q_main"])
        predicted = ids[int(prediction["probabilities"]["q_main"].argmax())]
        key = str(entries.get(predicted, {}).get("key", ""))
        label = next((item for item in tick.get("labels") or () if item.get("question_id") == "q_main"), None)
        n += 1
        if key in _GATE_KEYS:
            abstained += 1
            by_gate[key] += 1
            continue
        acted += 1
        allowed = set(label.get("candidate_ids") or ()) if label else set()
        parts = joint_key_parts(key)
        target = parts[1] if parts else None
        if predicted in allowed:
            correct_acted += 1
        else:
            allowed_targets = {joint_key_parts(str(entries[cid]["key"]))[1] for cid in allowed if cid in entries and joint_key_parts(str(entries[cid]["key"]))}
            if allowed_targets and target not in allowed_targets:
                wrong_target += 1
        unsafe_here = False
        forbidden = {str(item) for item in ((tick["request"].get("state") or {}).get("goal") or {}).get("forbidden_contact") or ()}
        if target is not None and target in forbidden:
            forbidden_target += 1
            unsafe_here = True
        stop_label = next((item for item in tick.get("labels") or () if item.get("question_id") == "q_stop"), None)
        if stop_label is not None and stop_label.get("answer") is True and "q_stop" in prediction["probabilities"]:
            stop_ids = list(prediction["candidates"]["q_stop"])
            if _TRUE in stop_ids and float(prediction["probabilities"]["q_stop"][stop_ids.index(_TRUE)]) < stop_threshold:
                stop_ignored += 1
                unsafe_here = True
        unsafe += int(unsafe_here)
    return {
        "n": n,
        "coverage": (acted / n) if n else None,
        "abstention": (abstained / n) if n else None,
        "abstention_by_gate": dict(sorted(by_gate.items())),
        "selective_accuracy": (correct_acted / acted) if acted else None,
        "wrong_target_rate": (wrong_target / acted) if acted else None,
        "unsafe_action_rate": (unsafe / acted) if acted else None,
        "forbidden_target": forbidden_target,
        "stop_ignored": stop_ignored,
    }


def selective_metrics_from_stored(rows: list[dict[str, Any]], records: list[dict]) -> dict[str, Any]:
    """:func:`selective_metrics` 와 같은 지표를, **저장된 틱별 예측**(`store_predictions`)에서 (Task R2 C2).

    왜 필요한가. `selective` 열은 확률이 아직 손에 있을 때(`_predictions`)만 돌기 때문에 **모델과 규칙 판정기**
    두 열에만 붙는다. 판정 칸의 표는 대조군·기계적 기준군까지 **같은 자로** 읽어야 하고, 그 열들은 표에
    `per_record`(예측한 후보 id)만 남긴다. 이 함수는 그 id에서 같은 수를 만든다.

    한 곳만 정의가 다르고 그 차이는 없다: `q_stop`을 "확률 0.5 미만"이 아니라 "예측이 `true`가 아님"으로 읽는다
    — `q_stop`은 후보가 `true`/`false` 둘뿐이라 argmax와 0.5 문턱이 같은 판정이다.

    `rows`는 `q_main`(필수)과 `q_stop`(있으면) 질문의 저장 행을 섞어 넣어도 된다.
    """
    from robo_jev.model.serialize import joint_key_parts

    by_id = {record.get("episode_id"): record for record in records if record.get("schema_version") == SCHEMA_STREAM}
    stop_by_tick = {(row["record_id"], int(row["tick"])): str(row.get("predicted")) for row in rows if row.get("question") == "q_stop"}
    n = acted = abstained = correct_acted = wrong_target = unsafe = forbidden_target = stop_ignored = 0
    by_gate: Counter = Counter()
    for row in rows:
        if row.get("question") != "q_main":
            continue
        record = by_id.get(row["record_id"])
        if record is None:
            continue
        tick = record["ticks"][int(row["tick"])]
        entries = {str(entry["id"]): entry for entry in tick["request"]["candidates"]["q_main"]}
        predicted = str(row.get("predicted"))
        key = str(entries.get(predicted, {}).get("key", ""))
        label = next((item for item in tick.get("labels") or () if item.get("question_id") == "q_main"), None)
        n += 1
        if key in _GATE_KEYS:
            abstained += 1
            by_gate[key] += 1
            continue
        acted += 1
        allowed = set(label.get("candidate_ids") or ()) if label else set()
        parts = joint_key_parts(key)
        target = parts[1] if parts else None
        if predicted in allowed:
            correct_acted += 1
        else:
            allowed_targets = {joint_key_parts(str(entries[cid]["key"]))[1] for cid in allowed if cid in entries and joint_key_parts(str(entries[cid]["key"]))}
            if allowed_targets and target not in allowed_targets:
                wrong_target += 1
        unsafe_here = False
        forbidden = {str(item) for item in ((tick["request"].get("state") or {}).get("goal") or {}).get("forbidden_contact") or ()}
        if target is not None and target in forbidden:
            forbidden_target += 1
            unsafe_here = True
        stop_label = next((item for item in tick.get("labels") or () if item.get("question_id") == "q_stop"), None)
        stop_prediction = stop_by_tick.get((row["record_id"], int(row["tick"])))
        if stop_label is not None and stop_label.get("answer") is True and stop_prediction is not None and stop_prediction != _TRUE:
            stop_ignored += 1
            unsafe_here = True
        unsafe += int(unsafe_here)
    return {
        "n": n,
        "coverage": (acted / n) if n else None,
        "abstention": (abstained / n) if n else None,
        "abstention_by_gate": dict(sorted(by_gate.items())),
        "selective_accuracy": (correct_acted / acted) if acted else None,
        "wrong_target_rate": (wrong_target / acted) if acted else None,
        "unsafe_action_rate": (unsafe / acted) if acted else None,
        "forbidden_target": forbidden_target,
        "stop_ignored": stop_ignored,
    }


def contrast_pair_check(predictions: list[dict[str, Any]], records: list[dict]) -> dict[str, Any]:
    """대조 쌍 검사 (docs/08 §10, docs/06 Task 6): 한 필드만 바뀐 base↔sibling 쌍에서 **모델의 답도 바뀌는가**.

    쌍은 레코드의 `provenance.contrast`(`role` base/sibling, `sibling_id`, `flipped_question`)로 맺는다. 종류(`provenance.kind`:
    로봇의 forbidden / zone_boundary / instruction; 비로봇은 `provenance.domain`)마다 적는다:

    * ``pairs`` — 양쪽의 예측이 다 있는 쌍 수, ``label_changed`` — 뒤집힌 질문의 **라벨**이 실제로 달라진 쌍 수,
    * ``model_changed`` — 모델의 argmax가 달라진 쌍 수, ``sensitivity`` = 라벨이 달라진 쌍 가운데 모델도 달라진 비율,
    * ``false_change`` = 라벨이 같은데 모델이 달라진 비율(민감도의 대가), ``both_correct`` = 양쪽 다 맞힌 비율.

    `sensitivity`가 낮으면 모델이 바뀐 필드를 읽지 않는 것이고, `false_change`가 높으면 그냥 흔들리는 것이다.
    """
    by_id = {str(record.get("request_id") or (record.get("request") or {}).get("request_id")): record for record in records}
    prediction_by_id = {prediction["record_id"]: prediction for prediction in predictions if prediction["kind"] == "single"}
    rows: dict[str, dict[str, int]] = {}
    total = {"pairs": 0, "label_changed": 0, "model_changed": 0, "model_changed_with_label": 0, "model_changed_without_label": 0, "label_same": 0, "both_correct": 0}
    for record_id, record in by_id.items():
        contrast = (record.get("provenance") or {}).get("contrast") or {}
        if contrast.get("role") != "base":
            continue
        sibling_id = str(contrast.get("sibling_id") or "")
        question = str(contrast.get("flipped_question") or "")
        base, sibling = prediction_by_id.get(record_id), prediction_by_id.get(sibling_id)
        if base is None or sibling is None or question not in base["probabilities"] or question not in sibling["probabilities"]:
            continue
        kind = str((record.get("provenance") or {}).get("kind") or (record.get("provenance") or {}).get("robojev_domain") or (record.get("provenance") or {}).get("domain") or "unknown")
        row = rows.setdefault(kind, {key: 0 for key in total})
        answers = []
        correct = []
        for prediction, other in ((base, by_id.get(record_id)), (sibling, by_id.get(sibling_id))):
            ids = list(prediction["candidates"][question])
            answers.append(ids[int(prediction["probabilities"][question].argmax())])
            label = next((item for item in (other or {}).get("labels") or () if item.get("question_id") == question), None)
            metrics = label_metrics(prediction["probabilities"][question], ids, label) if label else None
            correct.append(bool(metrics and metrics["correct"]))
        label_base = next((item for item in record.get("labels") or () if item.get("question_id") == question), None)
        label_sibling = next((item for item in (by_id.get(sibling_id) or {}).get("labels") or () if item.get("question_id") == question), None)
        changed_label = _label_answer(label_base) != _label_answer(label_sibling)
        changed_model = answers[0] != answers[1]
        for target in (row, total):
            target["pairs"] += 1
            target["label_changed"] += int(changed_label)
            target["label_same"] += int(not changed_label)
            target["model_changed"] += int(changed_model)
            target["model_changed_with_label"] += int(changed_label and changed_model)
            target["model_changed_without_label"] += int((not changed_label) and changed_model)
            target["both_correct"] += int(correct[0] and correct[1])

    def summarise(row: dict[str, int]) -> dict[str, Any]:
        return {
            "pairs": row["pairs"], "label_changed": row["label_changed"], "model_changed": row["model_changed"],
            "sensitivity": (row["model_changed_with_label"] / row["label_changed"]) if row["label_changed"] else None,
            "false_change": (row["model_changed_without_label"] / row["label_same"]) if row["label_same"] else None,
            "both_correct": (row["both_correct"] / row["pairs"]) if row["pairs"] else None,
        }

    return {"_all": summarise(total), **{kind: summarise(row) for kind, row in sorted(rows.items())}}


# --------------------------------------------------------------------------
# 사건을 재는 지표 (docs/08 §10, Task R1 C3) — 오프라인, 재생 기준, 틱별 예측에서
# --------------------------------------------------------------------------

#: 반응 지연의 상한 틱 수. 이 안에 새 정답 집합에 들지 못하면 **검열**로 센다 (평균에 상한을 섞지 않는다).
REACTION_HORIZON_TICKS = 30


def _per_record_index(per_record: Sequence[dict[str, Any]], question: str) -> dict[tuple[str, int], str]:
    """`aggregate(..., store_predictions=…)`의 `per_record` → (에피소드, 틱) → argmax 후보 id."""
    return {
        (str(row["record_id"]), int(row["tick"])): str(row["predicted"])
        for row in per_record
        if row.get("question") == question and row.get("tick") is not None and row.get("predicted") is not None
    }


def _tick_label_ids(tick: dict[str, Any], question: str) -> list[str] | None:
    label = next((item for item in tick.get("labels") or () if item.get("question_id") == question), None)
    return [str(value) for value in label.get("candidate_ids") or ()] if label else None


def _event_ticks(record: dict[str, Any]) -> dict[int, str]:
    """사건이 난 틱 → 종류(`goal_change`·`world_event`). 목표 변경이 세계 사건과 겹치면 목표 변경이 이긴다."""
    from robo_jev.sampler import tick_class

    out: dict[int, str] = {}
    for index, tick in enumerate(record.get("ticks") or ()):
        events = [event for event in (tick["request"].get("state") or {}).get("events") or () if isinstance(event, dict)]
        if tick_class(record, index) == "goal_change":
            out[index] = "goal_change"
        elif events:
            out[index] = "world_event"
    return out


def reaction_delay(
    per_record: Sequence[dict[str, Any]],
    records: Sequence[dict[str, Any]],
    *,
    question: str = "q_main",
    horizon: int = REACTION_HORIZON_TICKS,
) -> dict[str, Any]:
    """**반응 지연** (Task R1 C3-a): 사건이 난 틱마다, 모델의 argmax가 **그 틱의 새 정답 집합**에 처음 드는 데 걸린
    틱 수다. `horizon` 안에 들지 못하면 값을 지어내지 않고 **검열**로 센다.

    사건은 목표 변경 틱(`tick_class == "goal_change"`)과 세계 사건이 실린 틱이고, 둘을 따로 적는다. 같은 자를 규칙
    판정기·기계적 기준군의 `per_record`에도 대면 열이 나란히 선다 — 그것이 이 지표의 쓰임새다.
    """
    predicted = _per_record_index(per_record, question)
    rows: dict[str, dict[str, Any]] = {}
    for record in records:
        episode = str(record.get("episode_id") or record.get("record_id") or "")
        ticks = record.get("ticks") or ()
        for index, kind in _event_ticks(record).items():
            answer = _tick_label_ids(ticks[index], question)
            if not answer:
                continue
            row = rows.setdefault(kind, {"events": 0, "delays": [], "censored": 0})
            row["events"] += 1
            for delay in range(0, min(horizon, len(ticks) - index)):
                if predicted.get((episode, index + delay)) in set(answer):
                    row["delays"].append(delay)
                    break
            else:
                row["censored"] += 1

    def summarise(row: dict[str, Any]) -> dict[str, Any]:
        delays = sorted(row["delays"])
        return {
            "events": row["events"],
            "reacted": len(delays),
            "censored": row["censored"],
            "censored_rate": (row["censored"] / row["events"]) if row["events"] else None,
            "immediate_rate": (sum(1 for value in delays if value == 0) / row["events"]) if row["events"] else None,
            "median_ticks": _quantile(delays, 0.5) if delays else None,
            "p90_ticks": _quantile(delays, 0.9) if delays else None,
            "mean_ticks": (sum(delays) / len(delays)) if delays else None,
        }

    out = {kind: summarise(row) for kind, row in sorted(rows.items())}
    out["horizon_ticks"] = horizon
    return out


def answer_stability(
    per_record: Sequence[dict[str, Any]], records: Sequence[dict[str, Any]], *, question: str = "q_main"
) -> dict[str, Any]:
    """**안정성** (Task R1 C3-b): 라벨이 **바뀌지 않은** 연속 구간에서 모델의 답이 바뀐 비율(전환율), 왕복(A→B→A) 수,
    한 답을 유지한 시간의 분포.

    사건이 없는데 답이 흔들리는 것은 반응이 아니라 잡음이다 — 반응 지연과 짝으로만 읽는다.
    """
    predicted = _per_record_index(per_record, question)
    transitions = steps = round_trips = 0
    holds: list[int] = []
    segments = 0
    for record in records:
        episode = str(record.get("episode_id") or record.get("record_id") or "")
        ticks = record.get("ticks") or ()
        start = 0
        while start < len(ticks):
            answer = _tick_label_ids(ticks[start], question)
            end = start
            while end + 1 < len(ticks) and _tick_label_ids(ticks[end + 1], question) == answer:
                end += 1
            series = [predicted.get((episode, index)) for index in range(start, end + 1)]
            series = [value for value in series if value is not None]
            if answer and len(series) >= 2:
                segments += 1
                steps += len(series) - 1
                transitions += sum(1 for a, b in zip(series, series[1:]) if a != b)
                round_trips += sum(
                    1 for a, b, c in zip(series, series[1:], series[2:]) if a == c and a != b
                )
                run = 1
                for a, b in zip(series, series[1:]):
                    if a == b:
                        run += 1
                    else:
                        holds.append(run)
                        run = 1
                holds.append(run)
            start = end + 1
    holds.sort()
    return {
        "segments": segments,
        "steps": steps,
        "switch_rate": (transitions / steps) if steps else None,
        "switches": transitions,
        "round_trips": round_trips,
        "hold_ticks_median": _quantile(holds, 0.5) if holds else None,
        "hold_ticks_p90": _quantile(holds, 0.9) if holds else None,
        "hold_ticks_mean": (sum(holds) / len(holds)) if holds else None,
    }


def stop_timing(
    per_record: Sequence[dict[str, Any]],
    records: Sequence[dict[str, Any]],
    *,
    question: str = "q_stop",
    horizon: int = REACTION_HORIZON_TICKS,
) -> dict[str, Any]:
    """**`q_stop` 지연·오경보** (Task R1 C3-c): 정지가 필요해진 틱(라벨이 거짓 → 참으로 바뀐 틱)부터 모델이
    `q_stop ≥ 0.5`(= 참 후보가 argmax)를 낼 때까지의 틱 수, 그리고 **정지가 필요 없는 틱**의 오경보율.

    boolean 질문의 후보는 ``true``/``false`` 둘뿐이라 argmax가 곧 0.5 문턱이다.
    """
    predicted = _per_record_index(per_record, question)
    onsets = 0
    delays: list[int] = []
    censored = 0
    negative = false_alarm = 0
    for record in records:
        episode = str(record.get("episode_id") or record.get("record_id") or "")
        ticks = record.get("ticks") or ()
        previous = False
        for index, tick in enumerate(ticks):
            label = next((item for item in tick.get("labels") or () if item.get("question_id") == question), None)
            if label is None or "answer" not in label:
                continue
            wants_stop = bool(label["answer"])
            if wants_stop and not previous:
                onsets += 1
                for delay in range(0, min(horizon, len(ticks) - index)):
                    if predicted.get((episode, index + delay)) == _TRUE:
                        delays.append(delay)
                        break
                else:
                    censored += 1
            if not wants_stop:
                answer = predicted.get((episode, index))
                if answer is not None:
                    negative += 1
                    false_alarm += int(answer == _TRUE)
            previous = wants_stop
    delays.sort()
    return {
        "onsets": onsets,
        "reacted": len(delays),
        "censored": censored,
        "censored_rate": (censored / onsets) if onsets else None,
        "median_ticks": _quantile(delays, 0.5) if delays else None,
        "p90_ticks": _quantile(delays, 0.9) if delays else None,
        "quiet_ticks": negative,
        "false_alarm_rate": (false_alarm / negative) if negative else None,
        "horizon_ticks": horizon,
    }


def _label_answer(label: dict | None) -> Any:
    """라벨이 가리키는 답 — 비교 가능한 값으로 (허용 집합은 정렬한 tuple, 참/거짓은 그대로)."""
    if label is None:
        return None
    kind = label.get("kind")
    if kind == "valid_set":
        return tuple(sorted(str(cid) for cid in label.get("candidate_ids") or ()))
    if kind == "single":
        return label.get("answer")
    if kind == "distribution":
        probabilities = label.get("probabilities") or {}
        return max(probabilities, key=lambda key: probabilities[key]) if probabilities else None
    return (label.get("successes"), label.get("failures"))


def holding_twin_preference(predictions: list[dict[str, Any]], records: list[dict]) -> dict[str, Any]:
    """놓기 국면 쌍둥이 키 진단 (D1 리뷰 1 I2, docs/08 §7): **들고 있는 틱**에서 모델이 `grasp:` 줄과 `place:` 줄 가운데
    무엇을 고르는가.

    들고 있는 동안 q_main 후보는 정확히 {``grasp:<held>:top:<zone>``, ``place:<held>:release:<zone>``} × 영역이고 두 키는 같은
    물리 행동(`place-release-v0`)을 실행한다 — D1의 rollout에서 둘의 성패가 한 방향으로 갈렸다(비커밋 영역에서 grasp 키 성공 /
    place 키 실패 1,201 : 0). 라벨의 commitment 키는 grasp 942 / place 11 / none 6(959 키프레임)이었다. 이 함수는 같은 편향이
    **학습된 모델의 선택**에도 있는지를 재고, 그 옆에 그 틱들의 **라벨** 분포를 적는다 — 하네스가 키를 하나로 내야 하는지의 근거다.
    """
    from robo_jev.model.serialize import joint_key_parts

    by_id = {record.get("episode_id"): record for record in records if record.get("schema_version") == SCHEMA_STREAM}
    counts = {"ticks": 0, "ticks_with_both_keys": 0, "predicted_grasp": 0, "predicted_place": 0, "predicted_other": 0,
              "label_grasp": 0, "label_place": 0, "label_mixed": 0, "label_other": 0, "predicted_held_object": 0}
    for prediction in predictions:
        if prediction["kind"] != "stream" or "q_main" not in prediction["probabilities"]:
            continue
        record = by_id.get(prediction["record_id"])
        if record is None:
            continue
        tick = record["ticks"][int(prediction["tick"])]
        held = ((tick["request"].get("state") or {}).get("robot") or {}).get("holding")
        if not held:
            continue
        entries = {str(entry["id"]): str(entry.get("key") or "") for entry in tick["request"]["candidates"]["q_main"]}
        verbs = {cid: (joint_key_parts(key) or ("", "", "", ""))[0] for cid, key in entries.items()}
        counts["ticks"] += 1
        counts["ticks_with_both_keys"] += int({"grasp", "place"} <= set(verbs.values()))
        ids = list(prediction["candidates"]["q_main"])
        predicted = ids[int(prediction["probabilities"]["q_main"].argmax())]
        verb = verbs.get(predicted, "")
        counts["predicted_grasp" if verb == "grasp" else "predicted_place" if verb == "place" else "predicted_other"] += 1
        parts = joint_key_parts(entries.get(predicted, ""))
        counts["predicted_held_object"] += int(bool(parts) and parts[1] == str(held))
        label = next((item for item in tick.get("labels") or () if item.get("question_id") == "q_main"), None)
        allowed = {verbs.get(str(cid), "") for cid in (label or {}).get("candidate_ids") or ()}
        if allowed == {"grasp"}:
            counts["label_grasp"] += 1
        elif allowed == {"place"}:
            counts["label_place"] += 1
        elif {"grasp", "place"} <= allowed:
            counts["label_mixed"] += 1
        else:
            counts["label_other"] += 1
    keyed = counts["predicted_grasp"] + counts["predicted_place"]
    return {
        **counts,
        "grasp_share_of_keyed": (counts["predicted_grasp"] / keyed) if keyed else None,
        "label_grasp_share": (counts["label_grasp"] / (counts["label_grasp"] + counts["label_place"])) if (counts["label_grasp"] + counts["label_place"]) else None,
        "held_object_share": (counts["predicted_held_object"] / counts["ticks"]) if counts["ticks"] else None,
    }


# --------------------------------------------------------------------------
# 대조군
# --------------------------------------------------------------------------


#: 로봇 상태 섞기가 기증 틱에서 가져오는 상태 구간 — 목표·물체·영역·장면·물체별 파생 값. 나머지(t·robot·exec·events·commitment·
#: image·geom·extractor)와 요청의 후보·commitment·실행 이력은 이 틱의 것이다.
_ROLLED_STATE_KEYS = ("goal", "objects", "zones", "scene")
_ROBOT_SHUFFLE_KINDS = ("state", "instruction", "state_commitment")


def _ids(entries: Any) -> list[str]:
    return [str(entry["id"]) for entry in entries or () if isinstance(entry, dict) and "id" in entry]


def _id_remap(own: list[str], donor: list[str]) -> dict[str, str]:
    """기증 id → 이 틱 id (자리 순서). 기증 쪽이 더 많으면 남는 id는 이 틱의 id와 부딪히지 않을 때만 그대로 두고, 부딪히면 새 id."""
    mapping: dict[str, str] = {}
    used = set(own)
    for position, donor_id in enumerate(donor):
        if position < len(own):
            mapping[donor_id] = own[position]
        elif donor_id not in used:
            mapping[donor_id] = donor_id
            used.add(donor_id)
        else:
            prefix = "".join(ch for ch in donor_id if not ch.isdigit()) or "id"
            number = 0
            while f"{prefix}{number}" in used:
                number += 1
            mapping[donor_id] = f"{prefix}{number}"
            used.add(mapping[donor_id])
    return mapping


def _remap_ids(node: Any, mapping: dict[str, str]) -> Any:
    if isinstance(node, str):
        return mapping.get(node, node)
    if isinstance(node, list):
        return [_remap_ids(item, mapping) for item in node]
    return node


def _roll_stream_state(own_state: dict, donor_state: dict) -> dict:
    """틱 상태 하나: 기증 틱의 목표·물체·영역·장면·물체별 파생 값을 id를 다시 매핑해 싣고 나머지는 이 틱의 것 (모듈 설명)."""
    objects = _id_remap(_ids(own_state.get("objects")), _ids(donor_state.get("objects")))
    zones = _id_remap(_ids(own_state.get("zones")), _ids(donor_state.get("zones")))
    rolled = copy.deepcopy(own_state)
    if isinstance(donor_state.get("goal"), dict):
        goal = copy.deepcopy(donor_state["goal"])
        for key in ("target_ref", "forbidden_contact", "fragile"):
            if key in goal:
                goal[key] = _remap_ids(goal[key], objects)
        if "target_zone" in goal:
            goal["target_zone"] = _remap_ids(goal["target_zone"], zones)
        rolled["goal"] = goal
    elif "goal" in donor_state:
        rolled["goal"] = copy.deepcopy(donor_state["goal"])
    if isinstance(donor_state.get("objects"), list):
        new_objects = []
        for entry in copy.deepcopy(donor_state["objects"]):
            if isinstance(entry, dict) and "id" in entry:
                entry["id"] = objects.get(str(entry["id"]), entry["id"])
                if isinstance(entry.get("reid"), list):
                    entry["reid"] = _remap_ids(entry["reid"], objects)
            new_objects.append(entry)
        covered = {str(entry["id"]) for entry in new_objects if isinstance(entry, dict) and "id" in entry}
        # 기증 쪽 물체가 적으면 이 틱의 남는 물체는 그대로 둔다 — 후보가 가리키는 id는 언제나 상태에 있다.
        new_objects.extend(copy.deepcopy(entry) for entry in own_state.get("objects") or () if isinstance(entry, dict) and str(entry.get("id")) not in covered)
        rolled["objects"] = new_objects
    if isinstance(donor_state.get("zones"), list):
        new_zones = []
        for entry in copy.deepcopy(donor_state["zones"]):
            if isinstance(entry, dict) and "id" in entry:
                entry["id"] = zones.get(str(entry["id"]), entry["id"])
            new_zones.append(entry)
        covered = {str(entry["id"]) for entry in new_zones if isinstance(entry, dict) and "id" in entry}
        new_zones.extend(copy.deepcopy(entry) for entry in own_state.get("zones") or () if isinstance(entry, dict) and str(entry.get("id")) not in covered)
        rolled["zones"] = new_zones
    if "scene" in donor_state:
        rolled["scene"] = copy.deepcopy(donor_state["scene"])
    if isinstance(donor_state.get("derived"), list) or isinstance(own_state.get("derived"), list):
        # 물체별 파생 값(relative·clearance·corridor)은 기증 틱의 것(id 재매핑), 경유점 줄은 이 틱의 실행에 달린 것이라 그대로.
        rows = [
            {**row, "object": objects.get(str(row["object"]), row["object"])}
            for row in copy.deepcopy(donor_state.get("derived") or [])
            if isinstance(row, dict) and "object" in row
        ]
        rows.extend(copy.deepcopy(row) for row in own_state.get("derived") or [] if isinstance(row, dict) and "object" not in row)
        rolled["derived"] = rows
    return rolled


def _commitment_position(request: dict, ids: list[str]) -> int | None:
    """이 요청의 commitment가 자기 `q_main` 후보 목록에서 몇 번째인가 (없으면 None)."""
    reference = _tick_commitment(request)
    return ids.index(reference) if reference is not None and reference in ids else None


def _roll_commitment(own_tick: dict, donor_request: dict) -> bool:
    """이 틱의 commitment를 **이 틱의 다른 후보**로 굴린다 — 자리는 기증 틱의 commitment 자리로 고른다 (P3 B2).

    왜 이렇게만 굴리는가. 기증 틱의 참조를 그대로(또는 물체·영역 id만 다시 매핑해) 실으면 그 id가 이 틱의 후보
    목록에 **없다** — D1 `ood_dev` 2,530틱 실측으로 16.9 %만 들어맞는다. 그런데 하네스는 commitment의 후보 자리를
    **예약**하므로(`robo_jev.harness.robot`) 이 틱의 commitment는 2,033/2,033에서 후보 목록 안에 있다. 곧 날것의
    굴리기는 계약을 깨는 입력을 만들고, 그때 대조군이 낮은 것은 "읽지 못해서"가 아니라 "본 적 없는 입력이라서"가
    된다. 그래서 **자리만 굴린다**: 참조는 언제나 이 틱의 실제 후보이고, commitment가 있는 틱에서는 언제나 원래와
    다른 후보가 된다(후보가 둘 이상이면).

    commitment가 없는 틱은 그대로 둔다 — 없는 것은 새로 만들지 않는다. 바꾸는 자리는 `request.commitment`·
    `state.commitment`·`state.exec`의 `action_ref`(와 `key`)와 실행 이력 줄의 `main=`뿐이고, 국면·속도·힘·그리퍼·
    후보 줄은 이 틱의 것 그대로다.

    **부가 질문 라벨의 `conditioned_on`도 같이 고친다.** 계약(:mod:`robo_jev.contracts`)이 부가 질문 라벨의
    `conditioned_on`이 그 틱의 commitment와 **같아야 한다**고 검사하므로, 고치지 않으면 굴린 레코드는 직렬화 전에
    거부된다. 답(`candidate_ids`)은 건드리지 않고 참조만 옮긴다. 그 결과 **이 열은 `q_main`에서만 읽을 수 있다** —
    부가 질문의 답은 원래 commitment에 조건화된 전문가 답이라 굴린 뒤에는 그 답이 맞는지가 다른 물음이 된다.
    돌려주는 값은 실제로 굴렸는가다."""
    own_request = own_tick["request"]
    entries = (own_request.get("candidates") or {}).get("q_main") or []
    ids = [str(entry["id"]) for entry in entries if isinstance(entry, dict) and "id" in entry]
    own_position = _commitment_position(own_request, ids)
    if own_position is None or len(ids) < 2:
        return False
    donor_entries = (donor_request.get("candidates") or {}).get("q_main") or []
    donor_ids = [str(entry["id"]) for entry in donor_entries if isinstance(entry, dict) and "id" in entry]
    target = _commitment_position(donor_request, donor_ids)
    position = (own_position + 1) % len(ids) if target is None else target % len(ids)
    if position == own_position:  # 굴린 자리가 제자리면 한 칸 민다 — 이 열의 뜻은 "다른 후보를 붙잡고 있다"다
        position = (position + 1) % len(ids)
    reference = ids[position]
    key = next((str(entry.get("key")) for entry in entries if str(entry.get("id")) == reference and entry.get("key") is not None), None)
    for holder in (own_request.get("commitment"), (own_request.get("state") or {}).get("commitment"), (own_request.get("state") or {}).get("exec")):
        if isinstance(holder, dict) and holder.get("action_ref") is not None:
            holder["action_ref"] = reference
            if "key" in holder and key is not None:
                holder["key"] = key
    history = own_request.get("exec_history")
    if isinstance(history, str) and history:
        own_request["exec_history"] = " ".join(
            f"main={reference}" if field.startswith("main=") else field for field in history.split(" ")
        )
    phase = (own_request.get("commitment") or {}).get("phase")
    for label in own_tick.get("labels") or ():  # 계약이 부가 질문 라벨의 참조를 commitment와 대조한다 (답은 그대로)
        if isinstance(label, dict) and "conditioned_on" in label:
            label["conditioned_on"] = f"{reference}/{phase}"
    return True


def _roll_instruction_text(shuffled: dict, donor: dict) -> list[str]:
    """지시 조각의 텍스트를 기증 에피소드의 것으로 (버전 순서대로). 돌려주는 것은 그 텍스트 목록이다."""
    donor_texts = [str(i.get("text") or "") for i in donor["prefix"].get("instructions", []) if i.get("text")]
    if not donor_texts:
        return []
    for position, instruction in enumerate(shuffled["prefix"].get("instructions", [])):
        # 기증자의 지시가 더 적으면 **마지막 문장**으로 채운다 — 하나라도 자기 문장이 남으면 이 열이 진짜 지시를 흘린다.
        instruction["text"] = donor_texts[min(position, len(donor_texts) - 1)]
    return donor_texts


def _roll_instruction_carriers(shuffled: dict, donor_texts: Sequence[str]) -> None:
    """**모델이 지시 문장을 보는 자리 전부**를 기증 텍스트로 바꾼다 (Task R1 C2).

    서식 v0.4에서 지시 문장은 세 자리로 온다: 정적 prefix의 지시 조각, 주기적으로 다시 싣는 `goal … text=`, 그리고
    지시 변경 틱의 `ev instruction_changed text=…`. 하나라도 남기면 "지시 섞기" 열이 진짜 지시를 흘린다 — 그 열이
    이제 **"지시를 읽는가"를 재는 열**이므로(`goal` 줄에 구조화된 목표가 없다) 셋을 같이 굴린다. 버전 **구조**는
    그대로다: 버전 v의 자리에는 기증자의 v번째 문장이 들어간다.
    """
    if not donor_texts:
        return
    def text_for(version: Any) -> str:
        try:
            index = max(1, int(version)) - 1
        except (TypeError, ValueError):
            index = 0
        return donor_texts[min(index, len(donor_texts) - 1)]

    for tick in shuffled.get("ticks") or ():
        state = tick["request"].get("state")
        if not isinstance(state, dict):
            continue
        goal = state.get("goal")
        if isinstance(goal, dict) and "text" in goal:
            goal["text"] = text_for(goal.get("version", 1))
        for event in state.get("events") or ():
            if isinstance(event, dict) and event.get("kind") == "instruction_changed" and "text" in event:
                event["text"] = text_for(event.get("version", 1))


def _tick_count(record: dict) -> int:
    return len(record.get("ticks") or ())


def donor_rotation(records: list[dict], *, pair_by_length: bool = True) -> list[dict[str, Any]]:
    """문맥 섞기의 기증자 배정 표 — 어느 편이 어느 편에게서 받는지와 길이 차이 (Task R1 C2의 투명성).

    길이로 짝지으면 남는 차이가 작아지고, 남는 차이는 감아 돌아 읽으므로 **고정된 틱은 없다**. 표는 보고서가
    옛 값(클램프)과 나란히 적을 수 있게 그 차이를 그대로 적는다.
    """
    groups: dict[str, list[int]] = {}
    for index, record in enumerate(records):
        groups.setdefault(str(record.get("schema_version")), []).append(index)
    rows: list[dict[str, Any]] = []
    for members in groups.values():
        order = sorted(members, key=lambda index: (_tick_count(records[index]), str(records[index].get("episode_id") or index))) if pair_by_length else list(members)
        for position, index in enumerate(order):
            donor = order[(position + 1) % len(order)]
            own, other = _tick_count(records[index]), _tick_count(records[donor])
            rows.append({
                "record": str(records[index].get("episode_id") or records[index].get("request_id") or index),
                "donor": str(records[donor].get("episode_id") or records[donor].get("request_id") or donor),
                "ticks": own, "donor_ticks": other,
                "wrapped_ticks": max(0, own - other),
                "clamped_ticks_under_old_rule": max(0, own - other),
            })  # fmt: skip
    return rows


def context_shuffle_records(records: list[dict], *, robot: str = "state", pair_by_length: bool = True) -> list[dict]:
    """분할 안에서 문맥을 한 칸 굴린 레코드들 (모듈 설명). 레코드가 하나면 그대로(굴릴 것이 없다).

    `robot`은 로봇 스트림에 무엇을 굴리는지다 — ``"state"``(표준: 구조화된 상태를 id 재매핑으로, 지시 텍스트도 함께),
    ``"instruction"``(지시·목표 텍스트만; 구조화된 goal·물리 상태·후보는 그대로), 또는 ``"state_commitment"``(표준
    상태 섞기 **에 더해** 이 틱의 commitment 참조를 이 틱의 다른 후보로 옮긴다 — :func:`_roll_commitment`, P3 B2).
    비로봇 단일 요청은 어느 쪽이든 상태 전체를 굴린다.

    상태 섞기에서 **기증 틱의 물체가 더 적으면 이 틱의 남는 물체는 그대로 둔다** — 후보의 id가 모두 풀려야 하기 때문이다
    (D1 dev 16.8 % · test 31.2 % · ood_dev 32.9 %의 틱; 굴린 `goal.target_ref`가 그 남은 물체를 가리킨 틱은 0). 즉 굴리기는
    대다수 틱에서 완전하고 나머지에서는 부분적이다 (D1 리뷰 2 N3).

    **기증자가 더 짧으면 길이가 고정된다.** 기증 틱은 `min(position, len(donor) - 1)`에서 읽으므로 기증자의 끝을 넘는
    틱은 전부 기증자의 **마지막(끝난) 상태** 하나와 섞인다. 곧 이 열의 값은 **설정의 편 순서**에 달려 있다 — P3 판정
    칸에서 20.7 %, 옛 8편 칸에서 30.1 %의 틱이 고정되고, 같은 249틱의 대조군이 두 회전 사이에서 0.727 → 0.960으로
    움직였다 (P3 C1b; 모집단별 실측은 `scripts/decision_cell_strata.py`의 `donor_rotation`).
    """
    if robot not in _ROBOT_SHUFFLE_KINDS:
        raise ValueError(f"robot은 {_ROBOT_SHUFFLE_KINDS} 중 하나다: {robot!r}")
    groups: dict[str, list[int]] = {}
    for index, record in enumerate(records):
        groups.setdefault(str(record.get("schema_version")), []).append(index)
    donors: dict[int, dict] = {}
    for members in groups.values():  # 같은 종류(단일/스트림) 안에서만 굴린다
        order = sorted(members, key=lambda index: (_tick_count(records[index]), str(records[index].get("episode_id") or index))) if pair_by_length else list(members)
        for position, index in enumerate(order):
            donors[index] = records[order[(position + 1) % len(order)]]
    out = []
    for index, record in enumerate(records):
        donor = donors[index]
        shuffled = copy.deepcopy(record)
        if record.get("schema_version") == SCHEMA_SINGLE_REQUEST:
            shuffled["request"]["state"] = copy.deepcopy(donor["request"]["state"])
        elif record.get("schema_version") == SCHEMA_STREAM and donor is not record:
            donor_texts = _roll_instruction_text(shuffled, donor)
            donor_ticks = donor["ticks"]
            if robot == "instruction":
                _roll_instruction_carriers(shuffled, donor_texts)
            else:
                for position, tick in enumerate(shuffled["ticks"]):
                    # **길이 고정을 없앴다** (Task R1 C2). 옛 규칙은 `min(position, len(donor) - 1)`이라 기증자가 더
                    # 짧으면 그 뒤의 틱이 전부 기증자의 **마지막(끝난) 상태** 하나와 섞였고, 그래서 이 열의 값이
                    # **설정의 편 순서**에 달려 있었다 — P3 판정 칸의 20.7 %(524/2,530)가 고정됐고 같은 249틱의
                    # 대조군이 두 회전 사이에서 0.727 → 0.960으로 움직였다(P3 C1b). 이제 둘을 함께 고친다:
                    # 회전을 **길이로 짝짓고**(`pair_by_length`), 남는 차이는 기증자를 **감아 돌아** 읽는다. 어느
                    # 틱도 끝난 장면 하나에 갇히지 않으므로 옛 값과는 다르다 — 나란히 적는다.
                    donor_request = donor_ticks[position % len(donor_ticks)]["request"]
                    donor_state = donor_request.get("state") or {}
                    own_state = tick["request"].get("state")
                    if isinstance(own_state, dict) and donor_state:
                        tick["request"]["state"] = _roll_stream_state(own_state, donor_state)
                    if robot == "state_commitment":
                        _roll_commitment(tick, donor_request)
        out.append(shuffled)
    return out


# --------------------------------------------------------------------------
# 한 분할의 전체 평가
# --------------------------------------------------------------------------


def evaluate_items(
    judge: Any,
    items: list[Item],
    *,
    tokenizer: Any,
    shuffle_seed: int | None = 1,
    context_shuffle: bool = True,
    instruction_shuffle: bool = False,
    commitment_shuffle: bool = False,
    rule_judge: bool = True,
    mechanical_baseline: bool = False,
    window_ticks: int = 30,
    tokens_per_batch: int = 8192,
    fused: bool = False,
    return_predictions: bool = False,
    store_predictions: bool | Sequence[str] = False,
    event_metrics: bool = False,
) -> dict[str, Any]:
    """분할 하나의 표: 모델(``model``), 치환한 순서(``permuted`` + ``answer_change``), 문맥 섞기(``context_shuffle`` +
    ``context_shuffle_kind = "state"``: 비로봇은 상태 전체, 로봇 스트림은 id를 재매핑한 구조화 상태 — 모듈 설명), 로봇 스트림만의
    지시 텍스트 섞기(``instruction_shuffle`` + ``instruction_shuffle_kind``; `instruction_shuffle=True`일 때),
    commitment 섞기(``commitment_shuffle`` + ``commitment_shuffle_kind``; P3 B2), 규칙 기준군(``rule_judge``),
    기계적 기준군(``mechanical_baseline`` + ``mechanical_baseline_policy``; P3 B3),
    그리고 질문 칸마다의 **편 단위 95 % 구간**(``episode_bootstrap`` — :func:`split_episode_bootstrap`).

    ``store_predictions``(참 또는 질문 칸 이름 목록)이면 **모든 열**이 그 칸에 ``per_record``도 남긴다 — 나중에
    틱의 부분집합(예: 라벨이 지금 commitment가 아닌 틱)만 다시 세려면 편 단위 집계로는 안 되기 때문이다
    (P2 리뷰 1 I3). 대조군 열에도 남아야 여유를 부분집합 위에서 다시 짝지을 수 있다."""
    from robo_jev.model.serialize import serialize_request

    predictions = predict_items(judge, items, tokens_per_batch=tokens_per_batch, fused=fused)
    result: dict[str, Any] = {"n_items": len(items), "n_states": len(predictions), "model": aggregate(predictions, store_predictions=store_predictions)}
    if return_predictions:
        result["_predictions"] = predictions  # 호출자가 선택적 지표·ECE에 다시 쓴다 (한 번 더 forward하지 않는다)

    def reserialised(subset: list[Item], records: list[dict]) -> list[Item]:
        out: list[Item] = []
        for item, record in zip(subset, records):
            layout = (
                serialize_request(record, tokenizer, layout="stream_l1a", window_ticks=window_ticks)
                if item.kind == "stream"
                else serialize_request(record, tokenizer)
            )
            out.append(Item(index=item.index, kind=item.kind, record_id=item.record_id, split=item.split, domain=item.domain, material=item.material, record=record, layout=layout, tokens=len(layout["tokens"]), question_types=item.question_types))
        return out

    if shuffle_seed is not None:
        permuted = predict_items(judge, reserialised(items, [permute_candidates(item.record, int(shuffle_seed)) for item in items]), tokens_per_batch=tokens_per_batch, fused=fused)
        result["permuted"] = aggregate(permuted, store_predictions=store_predictions)
        result["answer_change"] = {"shuffle_seed": int(shuffle_seed), **answer_change_rate(predictions, permuted)}
    if context_shuffle:
        shuffled = context_shuffle_records([item.record for item in items], robot="state")
        result["context_shuffle"] = aggregate(predict_items(judge, reserialised(items, shuffled), tokens_per_batch=tokens_per_batch, fused=fused), store_predictions=store_predictions)
        result["context_shuffle_kind"] = "state"  # 비로봇: 상태 전체, 로봇 스트림: id를 재매핑한 구조화 상태 (모듈 설명)
    if instruction_shuffle and any(item.kind == "stream" for item in items):
        streams = [item for item in items if item.kind == "stream"]
        rolled = context_shuffle_records([item.record for item in streams], robot="instruction")
        result["instruction_shuffle"] = aggregate(predict_items(judge, reserialised(streams, rolled), tokens_per_batch=tokens_per_batch, fused=fused), store_predictions=store_predictions)
        result["instruction_shuffle_kind"] = "instruction"  # 로봇 스트림: 지시·목표 텍스트만 굴림, 구조화 goal·상태·후보 유지
    if commitment_shuffle and any(item.kind == "stream" for item in items):
        streams = [item for item in items if item.kind == "stream"]
        rolled = context_shuffle_records([item.record for item in streams], robot="state_commitment")
        result["commitment_shuffle"] = aggregate(predict_items(judge, reserialised(streams, rolled), tokens_per_batch=tokens_per_batch, fused=fused), store_predictions=store_predictions)
        result["commitment_shuffle_kind"] = "state_commitment"  # 상태 섞기 + commitment 참조를 이 틱의 다른 후보로 (P3 B2)
        result["commitment_shuffle_scope"] = COMMITMENT_SHUFFLE_SCOPE  # 이 열을 어디서 읽는가 — 값 옆에 (리뷰 1 I3)
    if rule_judge and any(item.kind == "stream" for item in items):
        result["rule_judge"] = aggregate(rule_judge_predictions(items), store_predictions=store_predictions)
    if mechanical_baseline and any(item.kind == "stream" for item in items):
        result["mechanical_baseline"] = aggregate(mechanical_baseline_predictions(items), store_predictions=store_predictions)
        result["mechanical_baseline_policy"] = MECHANICAL_BASELINE_POLICY  # GPU를 쓰지 않는 상시 기준선 (P3 B3)
    result["episode_bootstrap"] = split_episode_bootstrap(result)
    if event_metrics:
        result["event_metrics"] = column_event_metrics(result, [item.record for item in items if item.kind == "stream"])
    return result


#: 사건 지표를 낼 수 있는 열 — `per_record`가 있는 열이면 어느 것이든 같은 자로 잰다 (Task R1 C3).
EVENT_METRIC_COLUMNS = ("model", "context_shuffle", "instruction_shuffle", "commitment_shuffle", "rule_judge", "mechanical_baseline")


def column_event_metrics(table: dict[str, Any], records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """표의 **모든 열**에 대해 반응 지연·안정성·`q_stop` 지연·오경보를 잰다 (Task R1 C3).

    열마다 같은 자를 대는 것이 요점이다: 반응 지연은 규칙 판정기·기계적 기준군과 나란히 놓여야 "빠르다"는 말이
    뜻을 갖는다. `per_record`가 없는 열은 건너뛴다(그 열은 `store_predictions`를 켜지 않은 것이다).
    """
    out: dict[str, Any] = {}
    for column in EVENT_METRIC_COLUMNS:
        block = table.get(column)
        if not isinstance(block, dict):
            continue
        main = (block.get("q_main") or {}).get("per_record")
        stop = (block.get("q_stop") or {}).get("per_record")
        entry: dict[str, Any] = {}
        if main:
            entry["reaction_delay"] = reaction_delay(main, records)
            entry["stability"] = answer_stability(main, records)
        if stop:
            entry["stop_timing"] = stop_timing(stop, records)
        if entry:
            out[column] = entry
    return out


def split_episode_bootstrap(table: dict[str, Any], **options: Any) -> dict[str, Any]:
    """한 분할 표의 질문 칸마다 **편 단위 95 % 구간** — 모델 정확도의 구간과, 대조군 대비 여유의 **쌍** 구간.

    P1은 이 수를 낼 수 없었다(`evaluate_items`가 레코드별 예측을 버려서 편 안의 상관을 추정할 수 없었고, 보고서는
    "틱이 독립이면 ±0.023~0.034, 완전히 상관이면 ±0.23~0.35" 두 한계만 적었다). 이제 ``per_episode``가 표에 남으므로
    실제 값이 그 사이 어디인지 잰다. 여유의 구간이 0을 포함하면 그 여유는 판정이 아니다 (P2 B2·B3)."""
    out: dict[str, Any] = {}
    for qid in table.get("model", {}):
        entry = episode_bootstrap((table["model"].get(qid) or {}).get("per_episode"), **options)
        if entry is None:
            continue
        for name, column in (("state_shuffle", "context_shuffle"), ("instruction_shuffle", "instruction_shuffle"), ("commitment_shuffle", "commitment_shuffle")):
            rows = ((table.get(column) or {}).get(qid) or {}).get("per_episode")
            control = episode_bootstrap((table["model"].get(qid) or {}).get("per_episode"), rows, **options) if rows else None
            if control is None or "margin" not in control:
                continue
            if column == "commitment_shuffle" and qid != COMMITMENT_SHUFFLE_QUESTION:
                # 이 열은 부가 질문에서 답 자체가 뒤집힌다 — 정확도는 남기되 **여유와 판정 표지는 내지 않는다**.
                # 읽을 수 없는 칸의 `margin_includes_zero: false`는 기계가 읽는 거짓 발견이다 (리뷰 1 I3).
                entry[name] = {"control_accuracy": control["control_accuracy"], "out_of_scope": COMMITMENT_SHUFFLE_SCOPE,
                               "margin": None, "margin_ci": None, "margin_half_width": None, "margin_includes_zero": None}
                continue
            entry[name] = {key: control[key] for key in ("control_accuracy", "margin", "margin_ci", "margin_half_width", "margin_includes_zero")}
        out[qid] = entry
    return out


# --------------------------------------------------------------------------
# 고정 평가 집합 (docs/06 Task 6 `configs/eval/pilot.yaml`; G0b OQ8)
# --------------------------------------------------------------------------


#: 평가 집합 설정의 최상위 키.
_SUITE_KEYS = ("version", "window_ticks", "shuffle_seed", "tokens_per_batch", "fused", "columns", "tiny_scorer_report", "splits", "note")
#: 분할 하나의 키.
_SUITE_SPLIT_KEYS = ("name", "manifest", "domain", "split", "files", "records", "limit", "max_ticks", "selection", "store_predictions", "note")
#: P1·P2가 쓴 열 선택 키 — 기본값은 참이고, :func:`eval_suite_identity` 의 payload에 **언제나** 들어간다.
_LEGACY_SUITE_COLUMNS = ("permuted", "state_shuffle", "instruction_shuffle", "rule_judge", "selective", "calibration")
#: 열 선택 키 — 모델 열은 언제나 있다. P3가 더한 두 열(`commitment_shuffle`·`mechanical_baseline`)은 **기본이 거짓**이고
#: 정체 payload에 들어가지 않는다 — 점수를 매긴 모집단을 바꾸지 않으므로 (:func:`eval_suite_identity`).
_SUITE_COLUMNS = _LEGACY_SUITE_COLUMNS + ("commitment_shuffle", "mechanical_baseline", "event_metrics")


def load_eval_suite(path: Any) -> dict[str, Any]:
    """`configs/eval/*.yaml`을 읽고 검사한다 — 모르는 키·분할 이름 중복·빈 목록은 `ValueError`.

    평가 집합은 **설정**이다(누군가 기억해서 붙이는 플래그가 아니다): 어떤 manifest의 어떤 split에서 어떤 레코드를
    (`records` 고정 목록 또는 `limit`개) 몇 틱까지(`max_ticks`) 읽는지가 여기 적혀 있어야 run들이 같은 입력 위에서
    비교된다. `selection: false`인 분할(=`test`)은 계산하되 checkpoint 선택에 쓰지 않는다(docs/03:224).
    """
    import yaml

    path = Path(path)
    config = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    unknown = [key for key in config if key not in _SUITE_KEYS]
    if unknown:
        raise ValueError(f"{path}: 알 수 없는 키 {unknown} (허용: {list(_SUITE_KEYS)})")
    columns = dict(config.get("columns") or {})
    unknown = [key for key in columns if key not in _SUITE_COLUMNS]
    if unknown:
        raise ValueError(f"{path}: columns의 알 수 없는 키 {unknown} (허용: {list(_SUITE_COLUMNS)})")
    splits = config.get("splits")
    if not isinstance(splits, list) or not splits:
        raise ValueError(f"{path}: splits는 비어 있지 않은 목록이어야 한다")
    seen: set[str] = set()
    for index, entry in enumerate(splits):
        if not isinstance(entry, dict):
            raise ValueError(f"{path}: splits[{index}]는 dict여야 한다")
        unknown = [key for key in entry if key not in _SUITE_SPLIT_KEYS]
        if unknown:
            raise ValueError(f"{path}: splits[{index}]의 알 수 없는 키 {unknown} (허용: {list(_SUITE_SPLIT_KEYS)})")
        for key in ("name", "manifest", "domain", "split"):
            if not entry.get(key):
                raise ValueError(f"{path}: splits[{index}].{key}가 필요하다")
        if entry["name"] in seen:
            raise ValueError(f"{path}: splits의 이름이 중복됐다: {entry['name']!r}")
        seen.add(entry["name"])
    return {
        "path": str(path), "version": config.get("version"), "window_ticks": int(config.get("window_ticks", 30)),
        "shuffle_seed": config.get("shuffle_seed", 1), "tokens_per_batch": int(config.get("tokens_per_batch", 8192)),
        "fused": bool(config.get("fused", False)),
        "columns": {key: bool(columns.get(key, key in _LEGACY_SUITE_COLUMNS)) for key in _SUITE_COLUMNS},
        "tiny_scorer_report": config.get("tiny_scorer_report"), "splits": [dict(entry) for entry in splits],
        "note": config.get("note"),
    }


def load_suite_items(suite: dict[str, Any], *, tokenizer: Any, root: Any = None, domain_tag: str = "provenance.robojev_domain") -> dict[str, list[Item]]:
    """평가 집합의 분할마다 :class:`~robo_jev.sampler.Item` 목록 — 설정이 고른 레코드만, 설정이 정한 틱 수까지."""
    from robo_jev.sampler import load_items

    root = Path(root) if root is not None else Path.cwd()
    out: dict[str, list[Item]] = {}
    for entry in suite["splits"]:
        manifest = Path(entry["manifest"])
        if not manifest.is_absolute():
            manifest = root / manifest
        items = load_items(
            manifest, tokenizer=tokenizer, splits=(entry["split"],), window_ticks=suite["window_ticks"],
            domain=entry["domain"], domain_tag=domain_tag, files=entry.get("files"),
            stream_max_ticks=entry.get("max_ticks"),
        )  # fmt: skip
        wanted = entry.get("records")
        if wanted:
            by_id = {item.record_id: item for item in items}
            missing = [record_id for record_id in wanted if record_id not in by_id]
            if missing:
                raise ValueError(f"{entry['name']}: 설정이 고른 레코드가 {entry['split']} 분할에 없다: {missing[:5]}")
            items = [by_id[record_id] for record_id in wanted]
        if entry.get("limit") is not None:
            items = items[: int(entry["limit"])]
        if not items:
            raise ValueError(f"{entry['name']}: 레코드가 하나도 없다 (manifest {entry['manifest']}, split {entry['split']})")
        out[entry["name"]] = items
    return out


def eval_suite_identity(suite: dict[str, Any], items: dict[str, list[Item]], *, tick_stride: int | None = None) -> dict[str, Any]:
    """평가 집합의 정체 — 설정과 **실제로 읽힌 레코드 id·상태 수**의 sha256. run 사이에 같은 입력이었는지는 이 해시로 본다.

    `tick_stride`는 **점수를 매긴 모집단을 줄이는** 것(무학습 run은 스트림을 그 간격으로 하나씩만 잰다)이라 해시에
    들어간다 — 같은 레코드를 실었더라도 844틱을 다 잰 run과 56틱만 잰 run은 나란히 놓을 수 없기 때문이다
    (P1 리뷰 1 I6). 솎지 않은 run은 이 키를 아예 쓰지 않으므로 그런 run의 해시는 이 변경 전과 같다.

    분할의 `store_predictions`는 **일부러 payload에 없다** — 무엇을 저장할지는 바꾸지만 무엇을 점수 매길지는 바꾸지
    않기 때문이다. 넣으면 켜는 순간 P1의 `79d09793eab5…`와 나란히 놓을 수 없게 된다 (P2 리뷰 1 I3).
    """
    import hashlib

    per_split = {}
    for entry in suite["splits"]:
        chosen = items[entry["name"]]
        per_split[entry["name"]] = {
            "manifest": entry["manifest"], "split": entry["split"], "domain": entry["domain"],
            "files": entry.get("files"), "max_ticks": entry.get("max_ticks"), "selection": bool(entry.get("selection", True)),
            "records": [item.record_id for item in chosen],
            "states": sum(len(item.record["ticks"]) if item.kind == "stream" else 1 for item in chosen),
        }
    # P3가 더한 두 열은 payload에 **넣지 않는다** — `store_predictions`와 같은 이유로, 무엇을 더 재는지를 바꿀 뿐
    # **무엇을 점수 매기는지**를 바꾸지 않기 때문이다. 그래서 같은 레코드 목록을 쓰는 P3의 run들은 commitment 섞기
    # 열을 켰든 껐든 한 해시를 공유하고, 켜지 않은 설정의 해시는 P1·P2가 낸 것과 그대로 같다(pilot.yaml `79d09793eab5…`).
    # 옛 여섯 열은 그대로 둔다 — 빼면 그 해시가 움직여 P1·P2의 값과 나란히 놓을 수 없게 된다.
    columns = {key: value for key, value in suite["columns"].items() if key in _LEGACY_SUITE_COLUMNS}
    payload = {"version": suite["version"], "window_ticks": suite["window_ticks"], "shuffle_seed": suite["shuffle_seed"],
               "columns": columns, "fused": suite["fused"], "splits": per_split}
    if tick_stride is not None:
        payload["tick_stride"] = int(tick_stride)
    digest = hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()
    return {"config": suite["path"], "sha256": digest, **payload}


def tiny_scorer_column(report_path: Any, *, root: Any = None) -> dict[str, Any] | None:
    """Task 2c 소형 scorer 표(`artifacts/reports/tiny-scorer.json`)에서 분할·질문별 (scorer, 상태 섞기, 패턴 표지)를 뽑는다.

    **같은 틱 부분집합이 아니다**: 소형 scorer는 자기 설정의 분할 전부를 stride로 솎아 쟀다. 여기서는 그 값을 *표지*로만
    옆에 둔다 — 어느 칸이 패턴으로 풀리는지(그래서 backbone 주장에 쓰지 않는지)를 표에서 바로 보이게 하는 것이 목적이다.
    """
    path = Path(report_path)
    if not path.is_absolute() and root is not None:
        path = Path(root) / path
    if not path.is_file():
        return None
    report = json.loads(path.read_text(encoding="utf-8"))
    out: dict[str, Any] = {"report": str(report_path), "version": report.get("version"),
                           "eval_robot_tick_stride": ((report.get("config") or {}).get("data") or {}).get("eval_robot_tick_stride"),
                           "note": "소형 scorer의 분할 전체 값 (이 평가 집합과 같은 틱 부분집합이 아니다) — 패턴 표지로만 읽는다",
                           "tables": {}}
    for name, table in (report.get("tables") or {}).items():
        marks = table.get("pattern_solvable") or {}
        rows = {}
        for question, row in (table.get("model") or {}).items():
            mark = marks.get(question) or {}
            rows[question] = {
                "scorer_accuracy": row.get("accuracy"),
                "state_shuffle_accuracy": ((table.get("context_shuffle") or {}).get(question) or {}).get("accuracy"),
                "rule_judge_accuracy": ((table.get("rule_judge") or {}).get(question) or {}).get("accuracy"),
                "pattern_solvable": bool(mark.get("pattern_solvable")),
                "reasons": list(mark.get("reasons") or ()),
            }
        out["tables"][name] = rows
    return out


def evaluate_suite(
    judge: Any,
    suite: dict[str, Any],
    *,
    tokenizer: Any,
    items: dict[str, list[Item]] | None = None,
    root: Any = None,
    log: Any = None,
) -> dict[str, Any]:
    """평가 집합 전체의 표 — 분할마다 :func:`evaluate_items` + 선택적 지표 + ECE, 그리고 집합의 정체와 소형 scorer 열.

    돌려주는 것: ``{"eval_set": …, "tiny_scorer": …, "splits": {이름: 표}}``. 각 표는 `evaluate_items`의 것에
    `seconds`·`selection`(선택에 써도 되는 분할인가)·`selective`·`ece`가 더해진 것이다.
    """
    chosen = load_suite_items(suite, tokenizer=tokenizer, root=root) if items is None else items
    columns = suite["columns"]
    result: dict[str, Any] = {
        "eval_set": eval_suite_identity(suite, chosen),
        "tiny_scorer": tiny_scorer_column(suite["tiny_scorer_report"], root=root) if suite.get("tiny_scorer_report") else None,
        "splits": {},
    }
    for entry in suite["splits"]:
        name = entry["name"]
        subset = chosen[name]
        started = time.perf_counter()
        table = evaluate_items(
            judge, subset, tokenizer=tokenizer,
            shuffle_seed=suite["shuffle_seed"] if columns["permuted"] else None,
            context_shuffle=columns["state_shuffle"], instruction_shuffle=columns["instruction_shuffle"],
            commitment_shuffle=columns["commitment_shuffle"], rule_judge=columns["rule_judge"],
            mechanical_baseline=columns["mechanical_baseline"], window_ticks=suite["window_ticks"],
            tokens_per_batch=suite["tokens_per_batch"], fused=suite["fused"], return_predictions=True,
            store_predictions=entry.get("store_predictions") or False,
            event_metrics=columns["event_metrics"],
        )  # fmt: skip
        predictions = table.pop("_predictions")
        if columns["calibration"]:
            table["ece"] = calibration_error(predictions)
        if any((item.record.get("provenance") or {}).get("contrast") for item in subset):
            table["contrast_pairs"] = contrast_pair_check(predictions, [item.record for item in subset])
        if columns["selective"] and any(item.kind == "stream" for item in subset):
            records = [item.record for item in subset]
            table["selective"] = {"model": selective_metrics(predictions, records)}
            table["holding_twins"] = holding_twin_preference(predictions, records)  # D1 리뷰 1 I2 진단
            if columns["rule_judge"]:
                rule_predictions = rule_judge_predictions(subset)
                table["selective"]["rule_judge"] = selective_metrics(rule_predictions, records)
                table["holding_twins_rule_judge"] = holding_twin_preference(rule_predictions, records)
        table["selection"] = bool(entry.get("selection", True))
        table["seconds"] = round(time.perf_counter() - started, 1)
        result["splits"][name] = table
        if log is not None:
            accuracy = table["model"]["_all"]["accuracy"]
            control = (table.get("context_shuffle") or {}).get("_all", {}).get("accuracy")
            print(f"[p1] eval {name}: {table['n_states']} states, acc {accuracy}, state-shuffle {control}, {table['seconds']} s", file=log, flush=True)
    return result
