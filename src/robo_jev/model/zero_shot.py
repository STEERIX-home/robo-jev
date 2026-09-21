"""무학습 라벨 점수 읽기 — Nimble 방식 그대로 (docs/06 Task 2b 1단계 bullet 4, docs/03 §6 B1, analysis-nimble §3-5).

후보마다 한 글자 코드(A, B, …)를 주고 직렬화한 요청 뒤에 ``Requested question: <id>`` 와 코드가 붙은 후보 목록,
``Answer:`` 를 이어 붙인다. 마지막 위치의 hidden state를 **LM head의 코드 토큰 행에만** fp32로 투영해(전체
248,320행 head를 계산하지 않는다) 코드 위에서 softmax한다. 코드는 답 경계에서 **정확히 한 토큰**이어야 한다 —
:func:`verify_codes` 가 tokenizer로 ``"Answer:" + " " + 코드`` 의 마지막 토큰이 ``" " + 코드`` 한 토큰과 같은지(BPE 병합
검사) 확인한다. 후보 순서는 레코드마다 `shuffle_seed`로 치환해 코드와 순서의 상관을 끊는다. 학습은 없다.

단일 요청(비로봇)은 상태 텍스트(:func:`robo_jev.model.serialize.state_lines`) + 질문 문장 + 코드 후보 줄이다. 로봇 스트림
틱은 스트림 직렬화의 prefix + 윈도우(최근 30틱) 몸통 토큰을 그대로 문맥으로 쓰고 그 뒤에 질문·코드 후보를 붙인다 — 결정
표지 대신 코드 토큰을 읽는 것 말고는 모델이 보는 입력이 stream 경로와 같다. 정확도·NLL·Brier는 :mod:`robo_jev.evaluate`
의 지표와 같은 정의다.
"""

from __future__ import annotations

import copy
import hashlib
import random
import string
from typing import Any

import torch

from robo_jev.contracts import QUESTION_SET_V0, SCHEMA_SINGLE_REQUEST, SCHEMA_STREAM, model_input
from robo_jev.evaluate import aggregate, label_metrics
from robo_jev.model.serialize import candidate_line, serialize_request, state_lines, stream_candidate_line

__all__ = ["CODES", "build_single_prompts", "build_tick_prompts", "score_prompts", "verify_codes", "zero_shot_label_scores"]

CODES = tuple(string.ascii_uppercase)
_TRUE, _FALSE = "true", "false"


def verify_codes(tokenizer: Any, count: int) -> tuple[list[str], list[int]]:
    """앞 `count`개 코드가 답 경계에서 한 토큰인지 검사하고 (코드, 토큰 id)를 돌려준다. 아니면 `ValueError`."""
    if count > len(CODES):
        raise ValueError(f"후보가 {count}개 — 코드 {len(CODES)}개를 넘는다")
    codes, ids = [], []
    for code in CODES[:count]:
        alone = tokenizer.encode(" " + code, add_special_tokens=False).ids
        boundary = tokenizer.encode("Answer: " + code, add_special_tokens=False).ids
        if len(alone) != 1 or boundary[-1] != alone[0]:
            raise ValueError(f"코드 {code!r}가 답 경계에서 한 토큰이 아니다 (alone={alone}, boundary tail={boundary[-3:]})")
        codes.append(code)
        ids.append(int(alone[0]))
    return codes, ids


def _order(key: str, n: int, seed: int | None) -> list[int]:
    if seed is None:
        return list(range(n))
    rng = random.Random(int.from_bytes(hashlib.sha256(f"{key}|{seed}".encode("utf-8")).digest()[:8], "big"))
    order = list(range(n))
    rng.shuffle(order)
    return order


def _suffix(question_id: str, instruction: str, coded: list[tuple[str, str]]) -> str:
    lines = [f"\nRequested question: {question_id}", instruction, "Candidates:"]
    lines.extend(f" {code}. {line.rstrip()}" for code, line in coded)
    lines.append("Answer:")
    return "\n".join(lines)


def build_single_prompts(record: dict, tokenizer: Any, *, shuffle_seed: int | None) -> list[dict[str, Any]]:
    """단일 요청 레코드 → 질문마다 ``{"tokens", "question_id", "candidates"(코드 순서의 id), "codes", "code_ids", "label"}``."""
    projected = model_input(record)
    request = projected["request"]
    state_text = "[state]\n" + "\n".join(state_lines(request["state"])) + "\n"
    state_ids = tokenizer.encode(state_text, add_special_tokens=False).ids
    labels = {label.get("question_id"): label for label in record.get("labels", [])}
    out = []
    rid = str(request.get("request_id") or "")
    for question in request["questions"]:
        criteria = question["criteria"]
        order = _order(f"{rid}|{question['id']}", len(criteria), shuffle_seed)
        codes, code_ids = verify_codes(tokenizer, len(criteria))
        coded = [(codes[j], candidate_line(criteria[i])) for j, i in enumerate(order)]
        suffix_ids = tokenizer.encode(_suffix(question["id"], str(question.get("instructions", "")), coded), add_special_tokens=False).ids
        out.append(
            {
                "tokens": state_ids + suffix_ids, "question_id": question["id"], "type": question["type"],
                "candidates": [criteria[i]["id"] for i in order], "codes": codes, "code_ids": code_ids,
                "label": labels.get(question["id"]), "record_id": rid,
                "group": str(record.get("origin_group") or rid),  # 편 단위 집계의 묶음 (P2 B1)
            }
        )
    return out


def build_tick_prompts(record: dict, tick_index: int, tokenizer: Any, *, shuffle_seed: int | None, window_ticks: int = 30) -> list[dict[str, Any]]:
    """스트림 레코드의 틱 하나 → 질문마다 프롬프트 (prefix + 최근 `window_ticks`틱 몸통 토큰 + 질문·코드 후보)."""
    cut = copy.deepcopy(record)
    cut["ticks"] = cut["ticks"][: tick_index + 1]
    layout = serialize_request(cut, tokenizer, layout="stream_l1a", window_ticks=window_ticks)
    tokens = layout["tokens"]
    context = list(tokens[: layout["prefix_end"]])
    for tick in layout["ticks"][max(0, tick_index - window_ticks + 1) :]:
        context.extend(tokens[int(tick["start"]) : int(tick["body_end"])])
    entry = layout["ticks"][tick_index]
    request = record["ticks"][tick_index]["request"]
    labels = {label.get("question_id"): label for label in record["ticks"][tick_index].get("labels", [])}
    eid = str(record.get("episode_id") or "")
    geom_age = int(((request.get("state") or {}).get("age_ms") or {}).get("geom", 0) or 0)
    out = []
    for qid, ids in entry["candidate_mapping"].items():
        spec = QUESTION_SET_V0[qid]
        order = _order(f"{eid}|{tick_index}|{qid}", len(ids), shuffle_seed)
        codes, code_ids = verify_codes(tokenizer, len(ids))
        if spec["criteria"]:
            lines = [candidate_line(criterion) for criterion in spec["criteria"]]
        else:
            lines = [stream_candidate_line(candidate, geom_age_ms=geom_age, object_clearance={}) for candidate in request["candidates"][qid]]
        coded = [(codes[j], lines[i]) for j, i in enumerate(order)]
        suffix_ids = tokenizer.encode(_suffix(qid, str(spec["instructions"]), coded), add_special_tokens=False).ids
        out.append(
            {
                "tokens": context + suffix_ids, "question_id": qid, "type": spec["type"], "candidates": [ids[i] for i in order],
                "codes": codes, "code_ids": code_ids, "label": labels.get(qid), "record_id": eid, "tick": tick_index,
                "group": eid,  # 스트림의 편 = 에피소드 (P2 B1)
            }
        )
    return out


def score_prompts(backbone: Any, prompts: list[dict[str, Any]], *, batch: int = 8) -> list[dict[str, Any]]:
    """프롬프트 묶음 → 코드 위 확률 (마지막 유효 위치의 hidden × LM head의 코드 행, fp32). 오른쪽 padding 배치."""
    head = backbone.model.lm_head.weight
    device = backbone.device
    out = []
    with torch.no_grad():
        for start in range(0, len(prompts), batch):
            group = prompts[start : start + batch]
            length = max(len(p["tokens"]) for p in group)
            ids = torch.zeros(len(group), length, dtype=torch.long)
            for row, p in enumerate(group):
                ids[row, : len(p["tokens"])] = torch.tensor(p["tokens"], dtype=torch.long)
            positions = torch.arange(length)[None].expand(len(group), length)
            hidden = backbone(ids.to(device), positions.to(device))["hidden"]
            for row, p in enumerate(group):
                last = hidden[row, len(p["tokens"]) - 1].float()
                rows = head[torch.tensor(p["code_ids"], device=device)].float()
                logits = rows @ last
                out.append({**p, "probabilities": torch.softmax(logits, 0).cpu(), "logits": logits.cpu()})
    return out


def _as_prediction(scored: dict[str, Any], kind: str) -> dict[str, Any]:
    return {
        "record_id": scored["record_id"], "tick": scored.get("tick"), "kind": kind, "split": None,
        "group": scored.get("group") or scored["record_id"],
        "probabilities": {scored["question_id"]: scored["probabilities"]}, "candidates": {scored["question_id"]: list(scored["candidates"])},
        "labels": [scored["label"]] if scored["label"] is not None else [], "question_types": {scored["question_id"]: scored["type"]},
    }


def zero_shot_label_scores(
    model_id: str,
    records: list[dict],
    *,
    backbone: Any = None,
    tokenizer: Any,
    shuffle_seed: int | None = 1,
    tick_stride: int = 8,
    batch: int = 8,
    window_ticks: int = 30,
) -> dict[str, Any]:
    """docs/06 인터페이스: 학습 없이 후보 라벨 토큰 점수를 읽어 정확도·NLL·Brier(질문 id/타입별)와 위치 편향을 돌려준다.

    `records`는 단일 요청과 스트림이 섞여도 된다. 스트림은 `tick_stride`마다 한 틱(비용 때문에; 표에 적는다). `backbone`이
    없으면 `model_id`를 싣는다.
    """
    if backbone is None:
        from robo_jev.model.backbone_qwen import QwenBackbone

        backbone = QwenBackbone.load(model_id)
    predictions: list[dict[str, Any]] = []
    prompts_total = 0
    tokens_total = 0
    for record in records:
        schema = record.get("schema_version")
        if schema == SCHEMA_SINGLE_REQUEST:
            groups = [(build_single_prompts(record, tokenizer, shuffle_seed=shuffle_seed), "single")]
        elif schema == SCHEMA_STREAM:
            groups = [(build_tick_prompts(record, index, tokenizer, shuffle_seed=shuffle_seed, window_ticks=window_ticks), "stream") for index in range(0, len(record["ticks"]), max(1, tick_stride))]
        else:
            raise ValueError(f"zero_shot_label_scores: 알 수 없는 schema_version: {schema!r}")
        for prompts, kind in groups:
            prompts_total += len(prompts)
            tokens_total += sum(len(p["tokens"]) for p in prompts)
            for scored in score_prompts(backbone, prompts, batch=batch):
                predictions.append(_as_prediction(scored, kind))
    table = aggregate(predictions)
    return {
        "model_id": model_id, "recipe": "nimble: one-letter code per candidate, code rows of the LM head only, fp32 projection, per-record permutation",
        "shuffle_seed": shuffle_seed, "tick_stride": tick_stride, "prompts": prompts_total, "tokens": tokens_total, "table": table,
    }
