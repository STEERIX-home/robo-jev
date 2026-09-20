"""소형 scorer 기준군 — byte 임베딩 → 작은 Transformer 인코더 → 후보가 문맥에 attention → 공유 dot product → 후보 위 softmax
(docs/06 Task 2c, docs/03 §6; jevlike/cua-s1 계열 option-attention scorer, ≈1M 파라미터, CPU에서 처음부터 학습).

    uv run python -m robo_jev.baselines.tiny_scorer --config configs/baselines/tiny_scorer.yaml --out artifacts/reports/tiny-scorer.json

**무엇을 재는가.** 이 기준군이 높은 분할·질문은 의미 판단이 아니라 **패턴**(후보 줄·상태의 표면 규칙)으로 풀린다는 표지다 — 그
분할은 backbone의 성과 주장에 쓰지 않는다(docs/06 Task 2c). 표는 분할마다 (소형 scorer, 문맥 섞기 대조군, 후보 순서 치환의 답 변경률,
규칙 기준군(로봇 스트림), ECE, 선택적 지표)이며 :mod:`robo_jev.evaluate` 의 지표 함수를 그대로 쓴다.

**입력.** 문맥은 허용 필드만 투영한(:func:`robo_jev.contracts.model_input`) 레코드의 상태 텍스트다 — 비로봇 단일 요청은
:func:`robo_jev.model.serialize.state_lines`, 로봇 틱은 :func:`robo_jev.model.serialize.full_tick_sections`(그 틱을 첫 틱처럼 전부: 목표
텍스트·로봇·실행·commitment·이력을 앞에, 물체·영역·장면을 뒤에 두고 `max_context_bytes`에서 자른다). 질문 머리(`<id> <타입>: <문구>`)와
후보 줄(:func:`candidate_line` / :func:`stream_candidate_line`; boolean·ordinal은 계약의 고정 후보)이 후보 쪽 입력이고, 후보 id 순서는
직렬화의 `candidate_mapping`과 같다(요청 순서; 고정 질문은 계약 순서). 라벨·근거는 읽지 않는다(라벨은 손실·채점에만).

**모델.** 공유 byte 임베딩(256 + PAD) + 학습 위치 임베딩; 문맥 인코더(`context_layers`층); 후보·질문 머리 인코더 — 문맥의 마스크 평균
벡터를 후보 byte 임베딩에 더한 뒤(전역 문맥 주입: 초기화 직후 cross-attention만으로는 문맥이 점수의 1e-3 수준이라 한 epoch에 문맥을
읽지 않는 채로 남았다, batch-0 smoke) `candidate_layers`층 self-attention, 그 뒤 문맥 토큰에 대한 cross-attention 한 블록 → 마스크 평균
pooling; 점수 ``z_k = (U q)ᵀ(V c_k)/√d + b``(질문 머리 q, 후보 c_k; U·V·b는 모든 질문·타입이 공유) → 한 질문의 후보 위 softmax. 후보는
서로 독립으로 점수가 매겨지므로 **후보 순서에 구조적으로 불변**이다(치환 답 변경률은 0이 정상). 손실은 :func:`robo_jev.loss.label_loss`
(라벨 종류별; 마스크·unknown 처리 동일).

이 모듈은 generator·simulator를 import하지 않는다; 규칙 기준군 열만 :func:`robo_jev.evaluate.rule_judge_predictions`를 거쳐 하네스를 부른다.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import yaml
from torch import Tensor, nn

from robo_jev.contracts import QUESTION_SET_V0, SCHEMA_SINGLE_REQUEST, SCHEMA_STREAM, model_input
from robo_jev.evaluate import aggregate, answer_change_rate, calibration_error, context_shuffle_records, rule_judge_predictions, selective_metrics
from robo_jev.loss import label_loss
from robo_jev.model.serialize import candidate_line, full_tick_sections, state_lines, stream_candidate_line
from robo_jev.sampler import Item, manifest_files, permute_candidates

__all__ = [
    "Example",
    "QuestionExample",
    "SCORER_VERSION",
    "TinyScorer",
    "build_examples",
    "evaluate_split",
    "load_config",
    "load_records",
    "main",
    "predict",
    "train_tiny_scorer",
]

SCORER_VERSION = "tiny-scorer-v0.1"
PAD = 256
_VOCAB = 257
#: 로봇 틱 문맥의 구간 순서 — 부가 질문이 읽는 줄(목표·로봇·실행·commitment·이력)을 앞에 두어 byte 상한에 잘리지 않게 한다.
_CONTEXT_ORDER = ("t", "goal", "robot", "exec", "commitment", "exec_history", "events", "waypoints", "objects", "zones", "scene", "extra")


# --------------------------------------------------------------------------
# 예제 — 레코드 → (문맥 텍스트, 질문마다 머리·후보 줄·라벨)
# --------------------------------------------------------------------------


@dataclass
class QuestionExample:
    question_id: str
    question_type: str
    header: str
    candidate_ids: list[str]
    candidate_texts: list[str]
    label: dict | None


@dataclass
class Example:
    record_id: str
    tick: int | None
    kind: str  # single | stream
    split: str
    domain: str
    context: str
    questions: list[QuestionExample]
    labels: list[dict] = field(default_factory=list)

    @property
    def question_types(self) -> dict[str, str]:
        return {question.question_id: question.question_type for question in self.questions}

    @property
    def candidate_mapping(self) -> dict[str, list[str]]:
        return {question.question_id: list(question.candidate_ids) for question in self.questions}


def _truncate(text: str, limit: int) -> str:
    raw = text.encode("utf-8")
    if len(raw) <= limit:
        return text
    return raw[:limit].decode("utf-8", errors="ignore")


def _label_for(labels: list[dict], question_id: str) -> dict | None:
    return next((label for label in labels if label.get("question_id") == question_id), None)


def _single_examples(record: dict, *, domain: str, max_context: int, max_candidate: int) -> list[Example]:
    projected = model_input(record)
    request = projected["request"]
    context = _truncate("[state]\n" + "\n".join(state_lines(request["state"])) + "\n", max_context)
    labels = list(record.get("labels") or [])
    questions = []
    for spec in request["questions"]:
        questions.append(
            QuestionExample(
                question_id=str(spec["id"]),
                question_type=str(spec["type"]),
                header=_truncate(f"{spec['id']} {spec['type']}: {spec.get('instructions', '')}", max_candidate),
                candidate_ids=[str(criterion["id"]) for criterion in spec["criteria"]],
                candidate_texts=[_truncate(candidate_line(criterion).rstrip("\n"), max_candidate) for criterion in spec["criteria"]],
                label=_label_for(labels, str(spec["id"])),
            )
        )
    record_id = str(request.get("request_id") or record.get("origin_group") or "single")
    return [Example(record_id=record_id, tick=None, kind="single", split=str(record.get("split")), domain=domain, context=context, questions=questions, labels=labels)]


def _stream_examples(record: dict, *, domain: str, max_context: int, max_candidate: int, stride: int = 1) -> list[Example]:
    projected = model_input(record)
    first = projected["prefix"]["instructions"][0]
    examples = []
    for index, tick in enumerate(projected["ticks"]):
        if stride > 1 and index % stride:
            continue
        sections = full_tick_sections(tick)
        parts = [f"instruction v{first.get('version', 1)} {first.get('text', '')}\n"]
        parts.extend(sections[name] for name in _CONTEXT_ORDER if name in sections)
        context = _truncate("".join(parts), max_context)
        state = tick["request"].get("state") or {}
        geom_age = None
        if isinstance(state.get("t"), dict) and isinstance(state["t"].get("age_ms"), dict):
            geom_age = state["t"]["age_ms"].get("geom")
        labels = list(record["ticks"][index].get("labels") or [])
        candidates = tick["request"]["candidates"]
        questions = []
        for question_id, spec in QUESTION_SET_V0.items():
            if spec["criteria"]:
                ids = [str(criterion["id"]) for criterion in spec["criteria"]]
                texts = [_truncate(candidate_line(criterion).rstrip("\n"), max_candidate) for criterion in spec["criteria"]]
            elif question_id in candidates:
                ids = [str(entry["id"]) for entry in candidates[question_id]]
                texts = [_truncate(stream_candidate_line(entry, geom_age_ms=geom_age).rstrip("\n"), max_candidate) for entry in candidates[question_id]]
            else:
                continue
            questions.append(
                QuestionExample(
                    question_id=question_id, question_type=str(spec["type"]),
                    header=_truncate(f"{question_id} {spec['type']}: {spec.get('instructions', '')}", max_candidate),
                    candidate_ids=ids, candidate_texts=texts, label=_label_for(labels, question_id),
                )
            )
        examples.append(Example(record_id=str(record.get("episode_id")), tick=index, kind="stream", split=str(record.get("split")), domain=domain, context=context, questions=questions, labels=labels))
    return examples


def build_examples(records: list[dict], *, domain: str = "robot", max_context: int = 1024, max_candidate: int = 64, robot_tick_stride: int = 1) -> list[Example]:
    """레코드(judgment-v0·stream-v0) → 예제 목록. 스트림은 틱마다 하나(학습용 `robot_tick_stride`), 단일 요청은 하나."""
    out: list[Example] = []
    for record in records:
        schema = record.get("schema_version")
        if schema == SCHEMA_SINGLE_REQUEST:
            out.extend(_single_examples(record, domain=domain, max_context=max_context, max_candidate=max_candidate))
        elif schema == SCHEMA_STREAM:
            out.extend(_stream_examples(record, domain=domain, max_context=max_context, max_candidate=max_candidate, stride=robot_tick_stride))
        else:
            raise ValueError(f"모르는 schema_version: {schema!r}")
    return out


# --------------------------------------------------------------------------
# 모델
# --------------------------------------------------------------------------


def _encoder(d_model: int, heads: int, ff: int, layers: int, dropout: float) -> nn.TransformerEncoder:
    layer = nn.TransformerEncoderLayer(d_model, heads, ff, dropout=dropout, batch_first=True, norm_first=True, activation="gelu")
    return nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)


class TinyScorer(nn.Module):
    """byte 문맥 인코더 + 후보(문맥 attention) 인코더 + 공유 dot product (모듈 설명)."""

    def __init__(self, *, d_model: int = 128, heads: int = 4, ff: int = 256, context_layers: int = 3, candidate_layers: int = 1,
                 dropout: float = 0.1, max_context: int = 1024, max_candidate: int = 64) -> None:
        super().__init__()
        self.d_model = d_model
        self.max_context = max_context
        self.max_candidate = max_candidate
        self.embed = nn.Embedding(_VOCAB, d_model, padding_idx=PAD)
        self.pos_context = nn.Embedding(max_context, d_model)
        self.pos_candidate = nn.Embedding(max_candidate, d_model)
        self.context_encoder = _encoder(d_model, heads, ff, context_layers, dropout)
        self.candidate_encoder = _encoder(d_model, heads, ff, candidate_layers, dropout)
        self.context_pool_norm = nn.LayerNorm(d_model)
        self.context_inject = nn.Linear(d_model, d_model)
        self.cross_norm = nn.LayerNorm(d_model)
        self.cross = nn.MultiheadAttention(d_model, heads, dropout=dropout, batch_first=True)
        self.ff_norm = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(nn.Linear(d_model, ff), nn.GELU(), nn.Dropout(dropout), nn.Linear(ff, d_model))
        self.out_norm = nn.LayerNorm(d_model)
        self.query_proj = nn.Linear(d_model, d_model, bias=False)
        self.candidate_proj = nn.Linear(d_model, d_model, bias=False)
        self.bias = nn.Parameter(torch.zeros(()))
        self.dropout = nn.Dropout(dropout)

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def encode_context(self, context: Tensor, mask: Tensor) -> Tensor:
        positions = torch.arange(context.shape[1], device=context.device)
        x = self.dropout(self.embed(context) + self.pos_context(positions)[None])
        return self.context_encoder(x, src_key_padding_mask=mask)

    def encode_sequences(self, sequences: Tensor, mask: Tensor, context: Tensor, context_mask: Tensor, owner: Tensor) -> Tensor:
        """후보·질문 머리 byte 열(M×L) → 문맥(소유 상태의 것)에 attention한 뒤 마스크 평균 pooling한 벡터 (M×d)."""
        positions = torch.arange(sequences.shape[1], device=sequences.device)
        keep_context = (~context_mask).float().unsqueeze(-1)
        pooled = self.context_inject(self.context_pool_norm((context * keep_context).sum(1) / keep_context.sum(1).clamp_min(1.0)))
        x = self.dropout(self.embed(sequences) + self.pos_candidate(positions)[None] + pooled[owner][:, None, :])
        x = self.candidate_encoder(x, src_key_padding_mask=mask)
        # 모든 후보·질문 머리를 **한 번의** cross-attention으로 (memory = 소유 상태의 문맥, M×Lc×d gather ≈ 300 MB/step). 상태마다
        # 작은 호출로 나누는 쪽이 복사는 적지만 Python·커널 호출이 지배해 2배 느렸다(D1 실측 5.2 s/step vs 2.3 s/step, 4~16 thread).
        memory = context[owner]
        attended, _ = self.cross(self.cross_norm(x), memory, memory, key_padding_mask=context_mask[owner], need_weights=False)
        x = x + self.dropout(attended)
        x = x + self.dropout(self.ff(self.ff_norm(x)))
        x = self.out_norm(x)
        keep = (~mask).float().unsqueeze(-1)
        return (x * keep).sum(1) / keep.sum(1).clamp_min(1.0)

    def forward(self, batch: dict[str, Any]) -> list[dict[str, Tensor]]:
        """collate한 배치 → 상태마다 `{qid: logits[K]}`."""
        context = self.encode_context(batch["context"], batch["context_mask"])
        queries = self.query_proj(self.encode_sequences(batch["query"], batch["query_mask"], context, batch["context_mask"], batch["query_owner"]))
        candidates = self.candidate_proj(self.encode_sequences(batch["candidate"], batch["candidate_mask"], context, batch["context_mask"], batch["candidate_owner"]))
        scale = math.sqrt(self.d_model)
        out: list[dict[str, Tensor]] = [{} for _ in range(int(batch["context"].shape[0]))]
        for slot, (state_index, question_id, start, end) in enumerate(batch["questions"]):
            logits = (candidates[start:end] @ queries[slot]) / scale + self.bias
            out[state_index][question_id] = logits
        return out


def _bytes_tensor(texts: list[str], limit: int) -> tuple[Tensor, Tensor]:
    rows = [list(text.encode("utf-8"))[:limit] or [32] for text in texts]
    width = max(len(row) for row in rows) if rows else 1
    tokens = torch.full((len(rows), width), PAD, dtype=torch.long)
    for index, row in enumerate(rows):
        tokens[index, : len(row)] = torch.tensor(row, dtype=torch.long)
    return tokens, tokens.eq(PAD)


def collate(examples: list[Example], *, max_context: int, max_candidate: int) -> dict[str, Any]:
    """예제 묶음 → 텐서(문맥 B×Lc, 질문 머리 Q×Lq, 후보 M×Lk, 소유 색인, 질문 구간)."""
    context, context_mask = _bytes_tensor([example.context for example in examples], max_context)
    headers: list[str] = []
    header_owner: list[int] = []
    candidates: list[str] = []
    candidate_owner: list[int] = []
    questions: list[tuple[int, str, int, int]] = []
    for state_index, example in enumerate(examples):
        for question in example.questions:
            if not question.candidate_ids:
                continue
            start = len(candidates)
            candidates.extend(question.candidate_texts)
            candidate_owner.extend([state_index] * len(question.candidate_texts))
            headers.append(question.header)
            header_owner.append(state_index)
            questions.append((state_index, question.question_id, start, len(candidates)))
    query, query_mask = _bytes_tensor(headers, max_candidate)
    candidate, candidate_mask = _bytes_tensor(candidates, max_candidate)
    return {
        "context": context, "context_mask": context_mask,
        "query": query, "query_mask": query_mask, "query_owner": torch.tensor(header_owner, dtype=torch.long),
        "candidate": candidate, "candidate_mask": candidate_mask, "candidate_owner": torch.tensor(candidate_owner, dtype=torch.long),
        "questions": questions,
    }


# --------------------------------------------------------------------------
# 학습·예측
# --------------------------------------------------------------------------


def batch_loss(model: TinyScorer, examples: list[Example]) -> tuple[Tensor | None, int]:
    batch = collate(examples, max_context=model.max_context, max_candidate=model.max_candidate)
    outputs = model(batch)
    losses = []
    for state_index, example in enumerate(examples):
        for question in example.questions:
            if question.label is None or question.question_id not in outputs[state_index]:
                continue
            loss = label_loss(outputs[state_index][question.question_id], question.candidate_ids, question.label)
            if loss is not None:
                losses.append(loss)
    if not losses:
        return None, 0
    return torch.stack(losses).mean(), len(losses)


def predict(model: TinyScorer, examples: list[Example], *, batch_states: int = 32) -> list[dict[str, Any]]:
    """예제 → :mod:`robo_jev.evaluate` 의 예측 꼴 (`probabilities`·`candidates`·`labels`·`question_types`)."""
    model.eval()
    out: list[dict[str, Any]] = []
    with torch.no_grad():
        for start in range(0, len(examples), batch_states):
            chunk = examples[start : start + batch_states]
            outputs = model(collate(chunk, max_context=model.max_context, max_candidate=model.max_candidate))
            for state_index, example in enumerate(chunk):
                out.append(
                    {
                        "record_id": example.record_id, "tick": example.tick, "kind": example.kind, "split": example.split, "domain": example.domain,
                        "probabilities": {qid: torch.softmax(z.detach().float(), 0) for qid, z in outputs[state_index].items()},
                        "candidates": {qid: list(ids) for qid, ids in example.candidate_mapping.items() if qid in outputs[state_index]},
                        "labels": list(example.labels), "question_types": example.question_types,
                    }
                )
    return out


def _lr_at(step: int, total: int, *, lr: float, warmup_ratio: float) -> float:
    warmup = max(1, int(total * warmup_ratio))
    if step < warmup:
        return lr * (step + 1) / warmup
    progress = (step - warmup) / max(1, total - warmup)
    return lr * 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))


def train(model: TinyScorer, examples: list[Example], config: dict[str, Any], *, seed: int, log: Any = None) -> dict[str, Any]:
    """AdamW + 선형 warmup·cosine, 상태 묶음 `batch_states`, 예산 `max_wall_minutes`. 돌려주는 것은 학습 기록."""
    spec = config
    epochs = int(spec.get("epochs", 1))
    batch_states = int(spec.get("batch_states", 16))
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(spec.get("lr", 1e-3)), weight_decay=float(spec.get("weight_decay", 0.01)))
    rng = random.Random(seed)
    order = list(range(len(examples)))
    steps_per_epoch = max(1, math.ceil(len(order) / batch_states))
    total_steps = steps_per_epoch * epochs
    budget_s = float(spec.get("max_wall_minutes", 0) or 0) * 60.0
    started = time.perf_counter()
    history: list[dict[str, Any]] = []
    step = 0
    stopped = None
    for epoch in range(epochs):
        rng.shuffle(order)
        model.train()
        epoch_loss = epoch_labels = 0.0
        for start in range(0, len(order), batch_states):
            chunk = [examples[index] for index in order[start : start + batch_states]]
            lr = _lr_at(step, total_steps, lr=float(spec.get("lr", 1e-3)), warmup_ratio=float(spec.get("warmup_ratio", 0.05)))
            for group in optimizer.param_groups:
                group["lr"] = lr
            loss, labels = batch_loss(model, chunk)
            step += 1
            if loss is None:
                continue
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(spec.get("gradient_clip", 1.0)))
            optimizer.step()
            epoch_loss += float(loss.detach()) * labels
            epoch_labels += labels
            if log is not None and step % int(spec.get("log_every", 50)) == 0:
                print(f"  epoch {epoch + 1} step {step}/{total_steps} loss {float(loss.detach()):.4f} lr {lr:.2e} ({time.perf_counter() - started:.0f}s)", file=log, flush=True)
            if budget_s and time.perf_counter() - started > budget_s:
                stopped = {"epoch": epoch + 1, "step": step, "reason": "max_wall_minutes"}
                break
        history.append({"epoch": epoch + 1, "loss": (epoch_loss / epoch_labels) if epoch_labels else None, "labels": int(epoch_labels), "wall_s": round(time.perf_counter() - started, 1)})
        if log is not None:
            print(f"epoch {epoch + 1}: loss {history[-1]['loss']} over {int(epoch_labels)} labels ({history[-1]['wall_s']}s)", file=log, flush=True)
        if stopped:
            break
    return {"steps": step, "total_steps": total_steps, "epochs": history, "stopped": stopped, "wall_s": round(time.perf_counter() - started, 1)}


# --------------------------------------------------------------------------
# 표
# --------------------------------------------------------------------------


def rule_judge_predictions_for(records: list[dict], examples: list[Example]) -> list[dict[str, Any]]:
    """규칙 기준군 열(스트림만) — 예제가 있는 틱에 대해서만, 예제와 같은 후보 순서로. 평가 stride로 솎은 틱 집합이면 그 틱만 평가하고
    예측의 `tick`은 원래 틱 색인으로 되돌린다(선택적 지표가 레코드의 틱을 찾는다)."""
    by_id: dict[str, list[Example]] = {}
    for example in examples:
        if example.kind == "stream":
            by_id.setdefault(example.record_id, []).append(example)
    items: list[Item] = []
    original_index: dict[str, list[int]] = {}
    for index, record in enumerate(records):
        if record.get("schema_version") != SCHEMA_STREAM:
            continue
        ticks = sorted(by_id.get(str(record.get("episode_id")), []), key=lambda example: example.tick)
        if not ticks:
            continue
        subset = {**record, "ticks": [record["ticks"][example.tick] for example in ticks]}
        original_index[str(record.get("episode_id"))] = [int(example.tick) for example in ticks]
        layout = {"ticks": [{"candidate_mapping": example.candidate_mapping} for example in ticks]}
        items.append(Item(index=index, kind="stream", record_id=str(record.get("episode_id")), split=str(record.get("split")), domain="robot", material="existing",
                          record=subset, layout=layout, tokens=0, question_types=ticks[0].question_types))
    predictions = rule_judge_predictions(items)
    for prediction in predictions:
        prediction["tick"] = original_index[prediction["record_id"]][int(prediction["tick"])]
    return predictions


def _mark_pattern_solvable(model_table: dict[str, Any], shuffle_table: dict[str, Any] | None, rule: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """질문(id 또는 타입)마다 표지: 소형 scorer 정확도 ≥ `accuracy`, 문맥 섞기 정확도 ≥ `accuracy`, 또는 둘의 차이 ≤ `shuffle_gap`
    (소형 scorer ≥ `min_accuracy_for_gap`)."""
    threshold = float(rule.get("accuracy", 0.85))
    gap = float(rule.get("shuffle_gap", 0.05))
    floor = float(rule.get("min_accuracy_for_gap", 0.7))
    out: dict[str, dict[str, Any]] = {}
    for key, row in model_table.items():
        accuracy = row.get("accuracy")
        shuffled = (shuffle_table or {}).get(key, {}).get("accuracy") if shuffle_table else None
        reasons = []
        if accuracy is not None and accuracy >= threshold:
            reasons.append("scorer_high")
        if shuffled is not None and shuffled >= threshold:
            reasons.append("shuffle_high")
        if accuracy is not None and shuffled is not None and accuracy >= floor and accuracy - shuffled <= gap:
            reasons.append("context_irrelevant")
        out[key] = {"accuracy": accuracy, "shuffle_accuracy": shuffled, "pattern_solvable": bool(reasons), "reasons": reasons}
    return out


def evaluate_split(model: TinyScorer, records: list[dict], *, domain: str, config: dict[str, Any], data: dict[str, Any]) -> dict[str, Any]:
    """분할·분야 하나의 표: 소형 scorer, 치환 답 변경률, 문맥 섞기 대조군, 규칙 기준군(로봇 스트림), ECE, 선택적 지표, 표지."""
    max_context, max_candidate = int(data.get("max_context_bytes", 1024)), int(data.get("max_candidate_bytes", 64))
    stride = max(1, int(data.get("eval_robot_tick_stride", 1)))  # 평가 틱 솎기 (틱은 상관된 표본; 1이면 전부)
    build = lambda recs: build_examples(recs, domain=domain, max_context=max_context, max_candidate=max_candidate, robot_tick_stride=stride)  # noqa: E731
    examples = build(records)
    predictions = predict(model, examples)
    result: dict[str, Any] = {
        "n_records": len(records), "n_states": len(predictions), "kinds": dict(sorted(_count(example.kind for example in examples).items())),
        "eval_robot_tick_stride": stride,
        "model": aggregate(predictions), "ece": calibration_error(predictions, bins=int(config.get("ece_bins", 10))),
    }
    shuffle_seed = config.get("shuffle_seed", 1)
    if shuffle_seed is not None:
        permuted = predict(model, build([permute_candidates(record, int(shuffle_seed)) for record in records]))
        result["permuted"] = aggregate(permuted)
        result["answer_change"] = {"shuffle_seed": int(shuffle_seed), **answer_change_rate(predictions, permuted)}
    if config.get("context_shuffle", True):
        shuffled = predict(model, build(context_shuffle_records(records)))
        result["context_shuffle"] = aggregate(shuffled)
        result["context_shuffle_ece"] = calibration_error(shuffled, bins=int(config.get("ece_bins", 10)))
        kinds = {"instruction" if example.kind == "stream" else "state" for example in examples}
        result["context_shuffle_kind"] = "+".join(sorted(kinds))
    streams = [record for record in records if record.get("schema_version") == SCHEMA_STREAM]
    if streams:
        result["selective"] = selective_metrics(predictions, streams)
        if config.get("rule_judge", True):
            rule_predictions = rule_judge_predictions_for(records, examples)
            result["rule_judge"] = aggregate(rule_predictions)
            result["rule_judge_selective"] = selective_metrics(rule_predictions, streams)
    result["pattern_solvable"] = _mark_pattern_solvable(result["model"], result.get("context_shuffle"), config.get("pattern_solvable") or {})
    return result


def _count(values) -> dict[str, int]:
    out: dict[str, int] = {}
    for value in values:
        out[value] = out.get(value, 0) + 1
    return out


# --------------------------------------------------------------------------
# 데이터·설정·진입점
# --------------------------------------------------------------------------


def load_records(manifest_path: str | Path, *, splits: tuple[str, ...] | list[str], files: list[str] | None = None) -> list[dict]:
    """manifest의 파일들(`files` fnmatch 패턴으로 고른다)에서 `split`이 맞는 레코드만. sha256은 대조하지 않는다(적재기가 한다)."""
    manifest_file = Path(manifest_path)
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    entries = manifest_files(manifest, manifest_file)
    if files:
        from fnmatch import fnmatch

        entries = {name: entry for name, entry in entries.items() if any(fnmatch(name, pattern) for pattern in files)}
    out = []
    for name in entries:
        path = manifest_file.parent / name
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("split") in splits:
                out.append(record)
    return out


def load_config(path: str | Path) -> dict[str, Any]:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}


def build_model(config: dict[str, Any]) -> TinyScorer:
    spec = config.get("model") or {}
    data = config.get("data") or {}
    return TinyScorer(
        d_model=int(spec.get("d_model", 128)), heads=int(spec.get("heads", 4)), ff=int(spec.get("ff", 256)),
        context_layers=int(spec.get("context_layers", 3)), candidate_layers=int(spec.get("candidate_layers", 1)), dropout=float(spec.get("dropout", 0.1)),
        max_context=int(data.get("max_context_bytes", 1024)), max_candidate=int(data.get("max_candidate_bytes", 64)),
    )


def train_tiny_scorer(config: dict[str, Any], *, log: Any = None, checkpoint: Path | None = None) -> dict[str, Any]:
    """설정 → 학습(처음부터) → 분할·분야별 표 (docs/06 Task 2c의 `train_tiny_scorer(config) -> dict`)."""
    seed = int(config.get("seed", 17))
    torch.manual_seed(seed)
    if config.get("threads"):
        torch.set_num_threads(int(config["threads"]))
    data = config.get("data") or {}
    train_splits = tuple(data.get("train_splits") or ("train",))
    eval_splits = tuple(data.get("eval_splits") or ("dev",))
    max_context, max_candidate = int(data.get("max_context_bytes", 1024)), int(data.get("max_candidate_bytes", 64))
    started = time.perf_counter()

    train_examples: list[Example] = []
    sources: list[dict[str, Any]] = []
    eval_records: dict[tuple[str, str], list[dict]] = {}
    for entry in data.get("manifests") or ():
        domain = str(entry.get("domain") or "robot")
        train_records = load_records(entry["path"], splits=train_splits, files=entry.get("files"))
        examples = build_examples(train_records, domain=domain, max_context=max_context, max_candidate=max_candidate, robot_tick_stride=int(data.get("robot_tick_stride", 1)))
        train_examples.extend(examples)
        sources.append({"path": str(entry["path"]), "domain": domain, "train_records": len(train_records), "train_examples": len(examples)})
        for split in eval_splits:
            eval_records[(domain, split)] = load_records(entry["path"], splits=(split,), files=entry.get("files"))
    if not train_examples:
        raise ValueError("학습 예제가 없다 — data.manifests·train_splits를 확인한다")
    model = build_model(config)
    if log is not None:
        print(f"tiny scorer: {model.parameter_count():,} parameters, {len(train_examples)} train examples ({_count(e.kind for e in train_examples)})", file=log, flush=True)
    training = train(model, train_examples, config.get("train") or {}, seed=seed, log=log)
    if checkpoint is not None:
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"version": SCORER_VERSION, "config": config, "state_dict": model.state_dict()}, checkpoint)

    tables: dict[str, dict[str, Any]] = {}
    for (domain, split), records in sorted(eval_records.items()):
        if not records:
            continue
        # 로봇 manifest는 스트림(에피소드)과 틱 대조 쌍(단일)을 따로 표에 둔다 — 종류가 다르고 후자는 대조 쌍이다.
        groups = {"": records} if domain != "robot" else {
            "": [r for r in records if r.get("schema_version") == SCHEMA_STREAM],
            "_contrast": [r for r in records if r.get("schema_version") == SCHEMA_SINGLE_REQUEST],
        }
        for suffix, subset in groups.items():
            if not subset:
                continue
            name = f"{domain}{suffix}/{split}"
            if log is not None:
                print(f"evaluating {name}: {len(subset)} records", file=log, flush=True)
            tables[name] = evaluate_split(model, subset, domain=domain, config=config.get("eval") or {}, data=data)
    return {
        "version": SCORER_VERSION, "config": copy.deepcopy(config), "parameters": model.parameter_count(), "sources": sources,
        "train_examples": len(train_examples), "training": training, "tables": tables, "wall_s": round(time.perf_counter() - started, 1),
    }


def _print_tables(report: dict[str, Any], out: Any) -> None:
    for name, table in report["tables"].items():
        print(f"\n[{name}] states {table['n_states']} ece {table['ece']['ece']}", file=out)
        rows = sorted(set(table["model"]) | set((table.get("rule_judge") or {})) | set((table.get("context_shuffle") or {})))
        print(f"  {'question':<14}{'scorer':>8}{'shuffle':>9}{'rule':>8}{'nll':>8}{'change':>8}  pattern", file=out)
        for key in rows:
            model = table["model"].get(key, {})
            shuffle = (table.get("context_shuffle") or {}).get(key, {})
            rule = (table.get("rule_judge") or {}).get(key, {})
            change = ((table.get("answer_change") or {}).get("by_question") or {}).get(key)
            mark = table["pattern_solvable"].get(key, {})
            fmt = lambda value: f"{value:8.3f}" if isinstance(value, (int, float)) else f"{'-':>8}"  # noqa: E731
            print(f"  {key:<14}{fmt(model.get('accuracy'))}{fmt(shuffle.get('accuracy')):>9}{fmt(rule.get('accuracy'))}{fmt(model.get('nll'))}{fmt(change)}  {'yes ' + ','.join(mark.get('reasons', [])) if mark.get('pattern_solvable') else ''}", file=out)
        if table.get("selective"):
            s = table["selective"]
            print(f"  selective: coverage {s['coverage']} abstention {s['abstention']} selective_acc {s['selective_accuracy']} wrong_target {s['wrong_target_rate']} unsafe {s['unsafe_action_rate']}", file=out)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m robo_jev.baselines.tiny_scorer", description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=Path("configs/baselines/tiny_scorer.yaml"))
    parser.add_argument("--out", type=Path, default=Path("artifacts/reports/tiny-scorer.json"))
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--set", action="append", default=[], help="설정 덮어쓰기 KEY=YAML (점으로 중첩: train.epochs=1)")
    args = parser.parse_args(argv)
    config = load_config(args.config)
    for item in args.set:
        key, _, value = item.partition("=")
        node = config
        parts = key.split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = yaml.safe_load(value)
    report = train_tiny_scorer(config, log=sys.stdout, checkpoint=args.checkpoint)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(_jsonable(report), ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    _print_tables(report, sys.stdout)
    print(f"\n→ {args.out} ({report['parameters']:,} parameters, {report['wall_s']}s)")
    return 0


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Tensor):
        return value.tolist()
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
