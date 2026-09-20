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
* 문맥 섞기 대조군 (docs/03 §6): 분할 안에서 문맥을 한 칸 굴린 레코드 — 비로봇은 `request.state`를 다음 레코드의 것으로
  (**상태 섞기**: 남는 것은 후보뿐이라 높으면 후보만으로 답이 나오는 단축 경로다), 로봇 스트림은 지시 텍스트
  (`prefix.instructions[*].text`·`state.goal.text`)만 다음 에피소드의 것으로 바꾸고 **물리 상태·후보는 그대로 둔다**
  (**지시 섞기**: 상태에 달린 답(경로·그리퍼·속도·힘·완료·정지)은 대조군에서도 맞는 것이 정상이고, 이 값이 모델과 같으면
  모델이 지시·목표를 읽지 않는다는 뜻이다 — q_main이 그 잣대). 결과의 `context_shuffle_kind`가 어느 쪽인지 말한다.
  상태 블록을 굴리고 후보 id를 다시 매핑하는 로봇 상태 섞기 대조군은 D1 dev 검사로 미룬다(G0b 리뷰 1 I3).
* 규칙 기준군 (docs/02, `robo_jev.harness.rule_judge`): 로봇 틱마다 규칙 판단기의 10개 답을 같은 후보 목록 위의 확률로
  바꿔 같은 지표를 낸다 — 하네스만으로 풀리는 범위의 기준. 이 함수만 하네스를 import하므로 :mod:`robo_jev.train` 은
  이 모듈을 import하지 않는다(docs/06 §1의 경계는 학습 코드 쪽에 둔다).
"""

from __future__ import annotations

import copy
import math
from collections import Counter
from typing import Any

import torch

from robo_jev.contracts import SCHEMA_SINGLE_REQUEST, SCHEMA_STREAM
from robo_jev.loss import label_loss
from robo_jev.sampler import Item, permute_candidates

__all__ = [
    "aggregate",
    "calibration_error",
    "context_shuffle_records",
    "evaluate_items",
    "label_metrics",
    "predict_items",
    "rule_judge_predictions",
    "selective_metrics",
]

#: 게이트 후보의 의미 키 (docs/08 §4) — 선택적 지표에서 abstention으로 읽는다. 하네스를 import하지 않고 여기 둔다(`FIXED_KEYS`와 같다).
_GATE_KEYS = ("observe", "hold", "replan")

_TRUE, _FALSE = "true", "false"


# --------------------------------------------------------------------------
# 예측
# --------------------------------------------------------------------------


def predict_items(judge: Any, items: list[Item], *, tokens_per_batch: int = 8192) -> list[dict[str, Any]]:
    """상태(틱)마다 ``{"record_id", "tick", "kind", "probabilities": {qid: Tensor[K]}, "candidates": {qid: [id…]}, "labels": […],
    "question_types": {qid: type}}``. 단일 요청은 토큰 예산까지 묶은 microbatch로, 스트림은 에피소드마다 재생한다."""
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
                            "probabilities": {qid: torch.softmax(z.detach().float().cpu(), 0) for qid, z in result["logits"][position].items()},
                            "candidates": result["candidates"][position], "labels": list(b.record.get("labels", [])),
                            "question_types": dict(b.question_types),
                        }
                    )
            batch, budget = ([item], item.tokens) if item is not None else ([], 0)
        for item in streams:
            result = judge({"layout": "stream_l1a", "stream": item.layout})
            for index, (logits, candidates) in enumerate(zip(result["logits"], result["candidates"])):
                out.append(
                    {
                        "record_id": item.record_id, "tick": index, "kind": "stream", "split": item.split,
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
                    "record_id": item.record_id, "tick": index, "kind": "stream", "split": item.split, "probabilities": probabilities,
                    "candidates": {qid: list(ids) for qid, ids in entry["candidate_mapping"].items() if qid in probabilities},
                    "labels": list(tick.get("labels", [])), "question_types": dict(item.question_types),
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


def aggregate(predictions: list[dict[str, Any]]) -> dict[str, Any]:
    """예측 목록 → 질문(id 또는 타입)별 ``{n, accuracy, nll, brier, first_position_rate, position_counts}``와 전체."""
    rows: dict[str, dict[str, Any]] = {}
    for prediction in predictions:
        for label in prediction["labels"]:
            qid = label.get("question_id")
            if qid not in prediction["probabilities"]:
                continue
            metrics = label_metrics(prediction["probabilities"][qid], list(prediction["candidates"][qid]), label)
            if metrics is None:
                continue
            key = _table_key(prediction, qid)
            row = rows.setdefault(key, {"n": 0, "correct": 0, "graded": 0, "nll": 0.0, "brier": 0.0, "positions": Counter(), "choice_n": 0})
            row["n"] += 1
            row["nll"] += metrics["nll"]
            row["brier"] += metrics["brier"]
            if metrics["correct"] is not None:
                row["graded"] += 1
                row["correct"] += int(metrics["correct"])
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
        }
        for name in ("n", "correct", "graded", "nll", "brier"):
            total[name] += row[name]
    table["_all"] = {
        "n": total["n"],
        "accuracy": (total["correct"] / total["graded"]) if total["graded"] else None,
        "graded": total["graded"],
        "nll": (total["nll"] / total["n"]) if total["n"] else None,
        "brier": (total["brier"] / total["n"]) if total["n"] else None,
    }
    return table


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


# --------------------------------------------------------------------------
# 대조군
# --------------------------------------------------------------------------


def context_shuffle_records(records: list[dict]) -> list[dict]:
    """분할 안에서 문맥을 한 칸 굴린 레코드들 (모듈 설명). 레코드가 하나면 그대로(굴릴 것이 없다)."""
    groups: dict[str, list[int]] = {}
    for index, record in enumerate(records):
        groups.setdefault(str(record.get("schema_version")), []).append(index)
    donors: dict[int, dict] = {}
    for members in groups.values():  # 같은 종류(단일/스트림) 안에서만 굴린다
        for position, index in enumerate(members):
            donors[index] = records[members[(position + 1) % len(members)]]
    out = []
    for index, record in enumerate(records):
        donor = donors[index]
        shuffled = copy.deepcopy(record)
        if record.get("schema_version") == SCHEMA_SINGLE_REQUEST:
            shuffled["request"]["state"] = copy.deepcopy(donor["request"]["state"])
        elif record.get("schema_version") == SCHEMA_STREAM:
            donor_texts = [i.get("text") for i in donor["prefix"].get("instructions", [])]
            for position, instruction in enumerate(shuffled["prefix"].get("instructions", [])):
                if position < len(donor_texts) and donor_texts[position] is not None:
                    instruction["text"] = donor_texts[position]
            donor_goal = ((donor["ticks"][0]["request"].get("state") or {}).get("goal") or {}).get("text")
            for tick in shuffled["ticks"]:
                goal = (tick["request"].get("state") or {}).get("goal")
                if isinstance(goal, dict) and donor_goal is not None and "text" in goal:
                    goal["text"] = donor_goal
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
    rule_judge: bool = True,
    window_ticks: int = 30,
) -> dict[str, Any]:
    """분할 하나의 표: 모델(``model``), 치환한 순서(``permuted`` + ``answer_change``), 문맥 섞기(``context_shuffle`` +
    ``context_shuffle_kind``: 비로봇 ``state`` / 로봇 스트림 ``instruction`` — 모듈 설명), 규칙 기준군(``rule_judge``)."""
    from robo_jev.model.serialize import serialize_request

    predictions = predict_items(judge, items)
    result: dict[str, Any] = {"n_items": len(items), "n_states": len(predictions), "model": aggregate(predictions)}

    def reserialised(records: list[dict]) -> list[Item]:
        out: list[Item] = []
        for item, record in zip(items, records):
            layout = (
                serialize_request(record, tokenizer, layout="stream_l1a", window_ticks=window_ticks)
                if item.kind == "stream"
                else serialize_request(record, tokenizer)
            )
            out.append(Item(index=item.index, kind=item.kind, record_id=item.record_id, split=item.split, domain=item.domain, material=item.material, record=record, layout=layout, tokens=len(layout["tokens"]), question_types=item.question_types))
        return out

    if shuffle_seed is not None:
        permuted = predict_items(judge, reserialised([permute_candidates(item.record, int(shuffle_seed)) for item in items]))
        result["permuted"] = aggregate(permuted)
        result["answer_change"] = {"shuffle_seed": int(shuffle_seed), **answer_change_rate(predictions, permuted)}
    if context_shuffle:
        shuffled = context_shuffle_records([item.record for item in items])
        result["context_shuffle"] = aggregate(predict_items(judge, reserialised(shuffled)))
        kinds = {"instruction" if item.kind == "stream" else "state" for item in items}
        result["context_shuffle_kind"] = "+".join(sorted(kinds))  # state(비로봇: 상태 굴림) / instruction(로봇: 지시·목표 텍스트만 굴림)
    if rule_judge and any(item.kind == "stream" for item in items):
        result["rule_judge"] = aggregate(rule_judge_predictions(items))
    return result
