"""판단 요청으로 Qwen을 학습한다 (Task v0d §1·§2) — 진단 도구(v0c)와 **같은 프롬프트·같은 코드 토큰·같은 읽기**로.

**목표.** 질문 하나 = 프롬프트 하나(:mod:`robo_jev.decisions.examples` — 진단의 :func:`~robo_jev.decisions.prompt.build_prompts` 로 만든다).
답 자리(``Answer:``의 마지막 토큰 — 다음 토큰을 예측하는 위치)의 hidden(최종 RMSNorm 뒤)을 LM head의 **그 질문의 코드 행에만** fp32로
투영해 코드 위 softmax를 얻고(진단의 :func:`~robo_jev.decisions.zero_shot.score_code_prompts` 와 같은 수학), 정답의 코드 분포
(:func:`~robo_jev.decisions.examples.code_target`)와의 **soft-label 교차 엔트로피**를 줄인다. 보고하는 값은 KL(= 교차 엔트로피 − 정답의
엔트로피, 0이 바닥)이고 gradient는 교차 엔트로피와 같다. 프롬프트의 다른 토큰에는 손실이 없다(검사가 hidden의 gradient가 답 자리 밖에서
0임을 본다). **보조 항** (Task v0d fix round 1, 통제자 결정 3): 같은 답 자리에서 −λ·log(코드 질량), λ = `code_mass_weight`(기본 0.2).
코드 질량 = 전체 어휘 softmax가 **보여 준 코드들**에 준 몫 = exp(logsumexp(코드 logits) − logsumexp(전체 logits)) — 진단의 ``code_mass``와 같은
수학(fp32 LM head). 주 항(코드 행 위 KL)만으로는 보여 주지 않은 글자(다른 질문의 코드 C–H 등)로 확률이 새어 나간다(시운전 1: noul 0.975 → 0.54).
step 손실 = Σ 예 (w_type · KL + λ · (−log 코드 질량)) / 예 수 — 종류별 가중치(`type_weights`, 기본 모두 1)는 KL 항에만 곱하는 배수이고 보조 항은
모든 예에 같다(형식의 문제라서).

**두 방식** (`mode`):

* ``lora`` (기본) — 저장소의 :func:`robo_jev.train.attach_lora` (peft 0.21 ``inject_adapter_in_model``, adapter는 fp32, 나머지 고정)로
  attention(full attention의 q·k·v·o, DeltaNet의 in_proj_qkv·in_proj_z·out_proj)과 MLP(gate·up·down) 투영에 r = 32, α = 64.
  LM head(= embedding, 묶인 가중치)는 고정이다. optimizer는 평범한 AdamW(fp32 파라미터 — master 사본이 필요 없다).
  ``lora_dtype: bfloat16``(선택)은 adapter를 bf16으로 계산하고 fp32 master 사본(:class:`robo_jev.train.MasterWeightAdamW`)으로 갱신한다 —
  fp32 adapter는 peft가 모듈마다 입력을 fp32로 바꾸고 넓은 출력(최대 [토큰, 6144])을 fp32로 곱하고 더하는 원소별 연산이 많아, 대역폭이 낮은
  GB10에서 step의 ≈ 30 %를 쓴다(보고서의 측정). 평가 적재(:func:`apply_checkpoint`)는 adapter를 언제나 fp32로 계산한다(같은 값, 더 정밀).
* ``full`` — text 모델 전체(묶인 embedding/LM head 포함)를 BF16으로 계산하고 저장소의 :class:`robo_jev.train.MasterWeightAdamW`
  (fp32 master 사본에서 갱신, bf16에는 반올림만)로 갱신한다 — 이것 없이 BF16에 직접 더하면 작은 갱신이 반올림에 사라진다(P2).
  gradient 누적은 bf16 ``.grad``에서 일어난다(저장소의 T1 학습과 같다; 보고서).

공통: warmup + cosine(:func:`robo_jev.train.lr_factor`), 층 단위 gradient checkpointing(:attr:`QwenBackbone.activation_checkpointing`),
**길이별 묶음 + gradient 누적**(step = `batch_examples`개의 예; 길이로 정렬해 `micro_batch_tokens` 안(예 수 × 가장 긴 길이, 오른쪽 padding)으로
나눠 차례로 backward — 손실을 step 전체의 예 수로 나누므로 누적 = 한 번에 묶은 것), gradient clip, atomic 체크포인트(:func:`robo_jev.checkpoint.save_checkpoint`
— 저장소의 저장 단위 형식 그대로), 재개(설정·데이터·모델·tokenizer·프롬프트의 정체가 같아야 한다), 결정성(데이터 계획은 (seed, epoch, id)의
해시 — 재개는 step 수만으로 같은 자리; LoRA 초기화는 seed; 확률 연산 없음).

**dev.** 학습 파일에서 레코드 열쇠 해시로 2 %를 뗀다(:func:`~robo_jev.decisions.examples.split_dev`). dev 손실은 dev 레코드의 **모든 질문 × 균형
배치 전부**(진단과 같은 프롬프트, shuffle seed 1)의 KL이다 — 질문마다 배치 평균, 종류마다 질문 평균, 머리 값은 종류 평균의 `type_weights` 가중
평균(보조 항은 넣지 않는다 — 코드 질량은 따로 적는다). 위치 쏠림도 같이 적는다: 평균 이동이 주 지표(noul 예-먼저 쏠림 = 예가 먼저일 때의 p_yes −
나중일 때, score 방향 쏠림 = 오름 − 내림), 판정 바뀜(판정 = 값 ``>= 0.5``, 진단과 같은 규칙)과 **결정적인 바뀜**(두 답이 모두 0.5에서 0.1 이상
떨어진 채 바뀐 것 — 통제자 결정 6). 코드 질량은 종류별 평균과 0.5 아래인 배치의 몫. dev 채점은 길이로 묶어 한 번에 읽는다(진단의 배치 1과
bf16 잡음만큼 다를 수 있다 — 추세용). `dev_max_records`를 넘는 dev 레코드는 학습에도 dev에도 쓰지 않고 manifest에 그 수를 적는다.

**고르기는 미리 정한다** (통제자 결정 4, `selection: final`): **평가하는 모델은 마지막 체크포인트**다. dev 최선은 ``metrics.json``의 ``best``에
기록으로만 남는다(`keep_best`면 ``checkpoint-best.pt``도 쓰지만 평가에 쓰지 않는다). 멈추기(`early_stop_patience`번 연속 나아지지 않으면
``stopped_early``)는 dev로만 한다 — 한 epoch 본 학습에서는 발산을 막는 장치일 뿐이다.

**일반 능력 보호 갈고리.** 다른 판단 데이터(나중에 일반 결정 묶음)를 섞을 자리만 있다: 설정의 ``mix``(지금은 비어 있어야 한다)와
:meth:`DecisionTrainer.mix_plan` (지금은 언제나 빈 목록). 섞는 방식은 그 데이터가 생길 때 정한다.

**운영 규칙**(GB10 통합 메모리): 진입점(``scripts/train_decisions.py``)이 첫 CUDA 할당 전에 울타리를 걸고 적재 전 여유를 확인한다
(:data:`MIN_FREE_TRAIN_BYTES`). 적재 뒤에는 **step의 메모리를 추정**해(적재된 것 + optimizer 상태 + micro-batch의 activation — 토큰당 값은 2B에서
잰 것, :data:`ACTIVATION_BYTES_PER_TOKEN`) 울타리의 90 %를 넘으면 시작하지 않는다(`allow_large_activations: true`로만 넘긴다) — gradient
checkpointing 없이 16 K 토큰 micro-batch를 쓴 측정이 울타리에 닿아 상자 전체의 여유가 1.5 GB까지 내려갔다(리뷰 1 Minor 6).
체크포인트를 쓰기 전마다 run 디렉터리 파일 시스템의 여유가 `min_free_disk_gb`(기본 60 GB) 이상인지 본다 — 모자라면 쓰지 않고
:class:`DiskSpaceError` (전체 미세조정의 저장 단위는 2B에서 ≈ 26 GB다).

**평가 적재** (:func:`apply_checkpoint`): 진단 도구(``scripts/diagnose_decisions.py --checkpoint``)가 기반 backbone을 실은 뒤 학습한 가중치를
얹는다 — LoRA는 같은 설정으로 adapter를 붙이고 adapter 값을, 전체는 모델 가중치를 싣는다. 재개용 저장 단위와 model-only(슬림) 파일을 둘 다
받는다. 기반 모델·tokenizer·프롬프트 판과 고정 문구(:func:`~robo_jev.decisions.prompt.fixed_text_digest`)가 학습 때와 다르면 거절한다.
프롬프트 판(`prompt_version`, 기본 v0c-p3)은 학습·dev·평가가 같은 것을 쓴다.
"""

from __future__ import annotations

import copy
import datetime as dt
import hashlib
import json
import math
import os
import random
import shutil
import statistics
import subprocess
import time
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import torch

from robo_jev.checkpoint import (
    CHECKPOINT_FORMAT,
    collect_rng_state,
    load_checkpoint,
    load_model_checkpoint,
    restore_rng_state,
    save_checkpoint,
)
from robo_jev.decisions.examples import (
    DEFAULT_DEV_FRACTION,
    DEFAULT_DEV_SALT,
    DEFAULT_QUESTIONS_PER_REQUEST,
    build_examples,
    dev_examples,
    epoch_plan,
    examples_per_epoch,
    split_dev,
)
from robo_jev.decisions.prompt import FORMATS, PROMPT_VERSION, PROMPT_VERSIONS, fixed_text_digest
from robo_jev.decisions.records import file_sha256, read_records, validate_record
from robo_jev.train import DEFAULT_LORA as JUDGE_LORA
from robo_jev.train import MasterWeightAdamW, attach_lora, clip_gradients, lr_factor

__all__ = [
    "ACTIVATION_BYTES_PER_TOKEN",
    "DEFAULTS",
    "DEFAULT_LORA",
    "DiskSpaceError",
    "MIN_FREE_TRAIN_BYTES",
    "MODES",
    "RESUME_FREE_KEYS",
    "TINY_MODEL",
    "TRAIN_KIND",
    "DecisionTrainer",
    "apply_checkpoint",
    "base_identity",
    "code_kl",
    "check_step_memory",
    "disk_free_bytes",
    "estimate_step_memory",
    "load_training_records",
    "pack_micro_batches",
    "records_digest",
    "resolve_config",
    "summarize_dev",
]

TRAIN_KIND = "decisions-train-v0"
MODES = ("lora", "full")
TYPES = ("choice", "noul", "score")
#: 검사 전용 소형 난수 Qwen (CPU) — 진단 도구의 ``--model tiny``와 같은 것(seed 4, tokenizer 어휘 크기).
TINY_MODEL = "tiny"
#: LoRA 기본값 (Task v0d §2: r = 32, attention·MLP 투영) — 대상 층은 저장소의 Judge LoRA와 같은 목록이다(full attention q·k·v·o,
#: DeltaNet in_proj_qkv·in_proj_z·out_proj, MLP gate·up·down). α = 2r (Judge 기본값의 비율).
DEFAULT_LORA = {"r": 32, "alpha": 64, "dropout": 0.0, "targets": list(JUDGE_LORA["targets"])}
#: 학습 적재 전에 있어야 하는 장치(통합) 메모리 여유 (Task v0d §6: ≥ 60 GiB). 근거: 2B 전체 미세조정의 정적 상태가 bf16 파라미터 3.8 GB +
#: bf16 gradient 3.8 GB + fp32 master 7.5 GB + fp32 Adam 상태 15.0 GB = 30.1 GB(28 GiB)이고, step 안에서만 드는 fp32 gradient 버퍼
#: 7.5 GB, CUDA context ≈ 1 GiB, gradient checkpointing을 켠 micro-batch(≤ 16 K 토큰)의 activation 몇 GiB, 적재 때 dtype 변환의 일시적 사본과
#: expandable segment의 조각이 더해진다 — 측정한 peak(보고서)에 다른 프로세스(데스크톱 포함)가 같은 통합 메모리를 쓰는 상자의 여유를 둔 값이다.
#: LoRA는 훨씬 덜 쓰지만(≈ 10 GiB) 같은 문턱을 둔다 — 학습 유닛은 하나의 규칙으로 띄운다.
MIN_FREE_TRAIN_BYTES = 60 * 2**30
DTYPES = {"float32": torch.float32, "bfloat16": torch.bfloat16}
#: micro-batch activation의 토큰(패딩 포함)당 바이트 — Qwen3.5-2B(hidden 2,048 × 24층)의 LoRA(fp32 adapter) 측정: gradient checkpointing을 켜면
#: 16 K 토큰에서 peak − 적재 = 5.7 GB(≈ 0.35 MB/토큰), 끄면 71.5 GB(≈ 4.4 MB/토큰). 다른 크기는 hidden × 층 수에 비례해 늘린다. 여기에
#: micro-batch마다 1 GB를 더한다(작은 micro-batch의 고정 몫).
ACTIVATION_BYTES_PER_TOKEN = {"checkpointed": 0.36e6, "not_checkpointed": 4.4e6}
_ACTIVATION_REFERENCE = 2048 * 24
#: 메모리 추정이 울타리의 이 몫을 넘으면 시작하지 않는다.
FENCE_SHARE = 0.9

DEFAULTS: dict[str, Any] = {
    "kind": TRAIN_KIND,
    "model_id": "Qwen/Qwen3.5-2B",
    "train_input": None,
    "out_dir": "artifacts/runs/decisions-v0",
    "run_name": None,
    "mode": "lora",
    "lora": DEFAULT_LORA,
    "lora_dtype": "float32",  # adapter 파라미터의 계산 dtype — bfloat16이면 fp32 master(MasterWeightAdamW)로 갱신한다 (모듈 설명)
    "format": "chat",
    "prompt_version": PROMPT_VERSION,  # 학습·dev·평가가 같은 판 (기본 v0c-p3; v0c-p2는 재현용)
    "code_mass_weight": 0.2,  # 보조 항 −λ·log(코드 질량)의 λ (통제자 결정 3; 0이면 끈다)
    "selection": "final",  # 평가하는 모델 — 마지막 체크포인트로 미리 정한다 (통제자 결정 4); dev 최선은 기록만
    "augment": True,
    "questions_per_request": DEFAULT_QUESTIONS_PER_REQUEST,
    "type_weights": {"choice": 1.0, "noul": 1.0, "score": 1.0},
    "dev_fraction": DEFAULT_DEV_FRACTION,
    "dev_salt": DEFAULT_DEV_SALT,
    "dev_every": 50,
    "dev_max_records": None,
    "dev_batch_tokens": None,  # null = micro_batch_tokens
    "seed": 0,
    "epochs": 1,
    "max_steps": None,  # null = epochs × epoch당 step 수 (풀어 둔 값이 여기에 들어간다)
    "batch_examples": 32,
    "micro_batch_tokens": 1024,  # GB10 실측 최적(LoRA; 보고서 §4) — 큰 micro-batch는 padding과 원소별 연산의 메모리 대역폭 때문에 오히려 느리다
    "lr": 2.0e-4,
    "weight_decay": 0.0,
    "betas": [0.9, 0.999],
    "eps": 1.0e-8,
    "warmup_ratio": 0.05,
    "gradient_clip": 1.0,
    "gradient_checkpointing": True,
    "checkpoint_every": 500,
    "save_checkpoints": True,
    "keep_best": False,
    "early_stop_patience": None,
    "min_free_disk_gb": 60,
    "metrics_every": 10,
    "max_wall_hours": None,
    "stop_after_steps": None,
    "mix": [],
    "device": "cuda",
    "dtype": "bfloat16",
    "tiny_seed": 4,
    "allow_large_activations": False,  # 메모리 추정이 울타리의 90 %를 넘어도 시작한다 (명시적 확인 — 보통은 거절)
}
#: 재개할 때 체크포인트의 설정과 달라도 되는 키 — 중단·예산 밖의 운영 값·경로뿐이다(입력 파일은 경로가 아니라 내용 digest로 대조한다).
RESUME_FREE_KEYS = (
    "train_input", "out_dir", "run_name", "checkpoint_every", "save_checkpoints", "metrics_every", "max_wall_hours", "stop_after_steps",
    "dev_batch_tokens", "device",
)  # fmt: skip


class DiskSpaceError(RuntimeError):
    """체크포인트를 쓰기 전 디스크 여유가 문턱보다 작다 — 쓰지 않았다."""


def _need(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def resolve_config(config: dict[str, Any]) -> dict[str, Any]:
    """설정 dict → 기본값을 채우고 검사한 설정(새 dict). 모르는 키나 범위 밖의 값은 ``ValueError``."""
    unknown = sorted(set(config) - set(DEFAULTS) - {"resume"})
    _need(not unknown, f"decisions 학습 설정: 모르는 키 {unknown}")
    out = copy.deepcopy(DEFAULTS)
    out.update(copy.deepcopy({key: value for key, value in config.items() if key != "resume"}))
    _need(out["kind"] == TRAIN_KIND, f"kind: {TRAIN_KIND!r}여야 한다 (받은 값: {out['kind']!r})")
    _need(out["mode"] in MODES, f"mode: {list(MODES)} 중 하나 (받은 값: {out['mode']!r})")
    _need(out["format"] in FORMATS, f"format: {list(FORMATS)} 중 하나 (받은 값: {out['format']!r})")
    _need(out["dtype"] in DTYPES, f"dtype: {list(DTYPES)} 중 하나 (받은 값: {out['dtype']!r})")
    _need(out["prompt_version"] in PROMPT_VERSIONS, f"prompt_version: {list(PROMPT_VERSIONS)} 중 하나 (받은 값: {out['prompt_version']!r})")
    _need(_is_number(out["code_mass_weight"]) and out["code_mass_weight"] >= 0, "code_mass_weight: 0 이상")
    out["code_mass_weight"] = float(out["code_mass_weight"])
    _need(out["selection"] == "final", "selection: 'final'만 — 평가하는 모델은 마지막 체크포인트로 미리 정했다 (dev 최선은 기록으로만 남는다)")
    lora = {**DEFAULT_LORA, **(out["lora"] or {})}
    _need(set(lora) == set(DEFAULT_LORA), f"lora: 키는 {sorted(DEFAULT_LORA)} (받은 값: {sorted(lora)})")
    _need(_is_int(lora["r"]) and lora["r"] >= 1 and _is_number(lora["alpha"]) and lora["alpha"] > 0, "lora: r ≥ 1 정수, alpha > 0")
    _need(_is_number(lora["dropout"]) and 0.0 <= lora["dropout"] < 1.0, "lora.dropout: [0, 1)")
    _need(isinstance(lora["targets"], list) and lora["targets"] and all(isinstance(t, str) for t in lora["targets"]), "lora.targets: 이름 목록")
    _need(out["lora_dtype"] in DTYPES, f"lora_dtype: {list(DTYPES)} 중 하나 (받은 값: {out['lora_dtype']!r})")
    out["lora"] = lora
    _need(_is_int(out["questions_per_request"]) and out["questions_per_request"] >= 1, "questions_per_request: 1 이상의 정수")
    weights = out["type_weights"]
    _need(isinstance(weights, dict) and set(weights) == set(TYPES), f"type_weights: 키는 정확히 {list(TYPES)}")
    _need(all(_is_number(w) and w >= 0 for w in weights.values()) and any(w > 0 for w in weights.values()), "type_weights: 0 이상, 하나는 양수")
    out["type_weights"] = {kind: float(weights[kind]) for kind in TYPES}
    _need(_is_number(out["dev_fraction"]) and 0.0 <= out["dev_fraction"] <= 0.5, "dev_fraction: [0, 0.5]")
    for key in ("batch_examples", "micro_batch_tokens", "epochs"):
        _need(_is_int(out[key]) and out[key] >= 1, f"{key}: 1 이상의 정수")
    for key in ("max_steps", "dev_max_records", "dev_batch_tokens", "early_stop_patience", "stop_after_steps"):
        _need(out[key] is None or (_is_int(out[key]) and out[key] >= 1), f"{key}: null 또는 1 이상의 정수")
    for key in ("dev_every", "checkpoint_every", "metrics_every", "seed", "tiny_seed"):
        _need(_is_int(out[key]) and out[key] >= 0, f"{key}: 0 이상의 정수")
    _need(_is_number(out["lr"]) and out["lr"] > 0, "lr: 양수")
    _need(_is_number(out["weight_decay"]) and out["weight_decay"] >= 0, "weight_decay: 0 이상")
    _need(_is_number(out["warmup_ratio"]) and 0.0 <= out["warmup_ratio"] < 1.0, "warmup_ratio: [0, 1)")
    _need(out["gradient_clip"] is None or (_is_number(out["gradient_clip"]) and out["gradient_clip"] > 0), "gradient_clip: null 또는 양수")
    _need(isinstance(out["betas"], (list, tuple)) and len(out["betas"]) == 2 and all(_is_number(b) and 0 <= b < 1 for b in out["betas"]), "betas: [b1, b2]")
    out["betas"] = [float(b) for b in out["betas"]]
    _need(_is_number(out["eps"]) and out["eps"] > 0, "eps: 양수")
    _need(_is_number(out["min_free_disk_gb"]) and out["min_free_disk_gb"] >= 0, "min_free_disk_gb: 0 이상")
    _need(out["max_wall_hours"] is None or (_is_number(out["max_wall_hours"]) and out["max_wall_hours"] > 0), "max_wall_hours: null 또는 양수")
    _need(
        isinstance(out["mix"], list) and not out["mix"],
        "mix: 일반 능력 보호용 데이터 섞기는 자리만 있다(Task v0d) — 지금은 빈 목록이어야 한다 (섞는 방식은 그 데이터가 생길 때 정한다)",
    )
    for key in ("augment", "gradient_checkpointing", "save_checkpoints", "keep_best", "allow_large_activations"):
        _need(isinstance(out[key], bool), f"{key}: true/false")
    return out


# --------------------------------------------------------------------------
# 데이터
# --------------------------------------------------------------------------


def load_training_records(path: str | Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """JSONL → (형식 검사를 통과하고 id가 처음 나온 기록, 검사 요약). 진단 도구와 같은 규칙(문제 있는 기록·겹친 id는 세고 건너뛴다)."""
    records = read_records(path)
    valid, problems, seen = [], {}, set()
    for record in records:
        found = validate_record(record)
        rid = str(record.get("id"))
        if rid in seen:
            found = found + [f"duplicate id {rid!r}"]
        if found:
            if len(problems) < 20:
                problems[rid] = found[:5]
            continue
        seen.add(rid)
        valid.append(record)
    return valid, {"read": len(records), "invalid": len(records) - len(valid), "problems": problems}


def records_digest(records: Sequence[dict[str, Any]]) -> str:
    """학습에 쓰는 기록 내용의 sha256 (정렬한 키의 JSON) — 재개가 같은 데이터인지 경로가 아니라 내용으로 대조한다."""
    digest = hashlib.sha256()
    for record in records:
        digest.update(json.dumps(record, sort_keys=True, ensure_ascii=False).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def pack_micro_batches(lengths: Sequence[int], max_tokens: int) -> list[list[int]]:
    """예의 길이들 → 길이로 정렬한 micro-batch(index 목록)들: 각 묶음의 (예 수 × 가장 긴 길이) ≤ `max_tokens` (한 예가 넘으면 혼자)."""
    order = sorted(range(len(lengths)), key=lambda i: (int(lengths[i]), i))
    groups: list[list[int]] = []
    current: list[int] = []
    for i in order:
        if current and int(lengths[i]) * (len(current) + 1) > int(max_tokens):
            groups.append(current)
            current = []
        current.append(i)
    if current:
        groups.append(current)
    return groups


# --------------------------------------------------------------------------
# 손실
# --------------------------------------------------------------------------


def code_kl(backbone: Any, examples: Sequence[dict[str, Any]], *, code_mass: bool = False, head32: torch.Tensor | None = None) -> dict[str, Any]:
    """예들(``tokens``·``code_ids``·``target``)을 오른쪽 padding으로 한 번에 forward → 답 자리의 코드 분포와 정답의 KL·교차 엔트로피.

    돌려주는 것: ``kl``·``ce``(``[B]`` fp32 tensor — gradient가 켜져 있으면 미분 가능), ``probs``(예마다 보이는 순서의 코드 확률 목록).
    코드 logits = 답 자리 hidden(fp32) · LM head 코드 행(fp32) — 진단의 읽기와 같은 수학이다. `code_mass`면 ``log_code_mass``(``[B]``,
    미분 가능 — 보조 항)와 ``code_mass``(목록)도: 전체 어휘 logits = hidden(fp32) · LM head(fp32) — `head32`(고정된 head의 fp32 사본; LoRA)를
    주면 그것을, 아니면 ``head.float()``(전체 미세조정: gradient가 묶인 head로 간다)를 쓴다.
    """
    device = backbone.device
    lengths = [len(e["tokens"]) for e in examples]
    width = max(lengths)
    ids = torch.zeros(len(examples), width, dtype=torch.long)
    for row, example in enumerate(examples):
        ids[row, : lengths[row]] = torch.tensor(example["tokens"], dtype=torch.long)
    positions = torch.arange(width)[None].expand(len(examples), width)
    hidden = backbone(ids.to(device), positions.to(device))["hidden"]
    answer = torch.tensor([n - 1 for n in lengths], device=hidden.device)
    states = hidden[torch.arange(len(examples), device=hidden.device), answer].float()  # [B, d] — 답 자리만
    n_codes = max(len(e["code_ids"]) for e in examples)
    index = torch.zeros(len(examples), n_codes, dtype=torch.long)
    mask = torch.zeros(len(examples), n_codes, dtype=torch.bool)
    target = torch.zeros(len(examples), n_codes, dtype=torch.float32)
    for row, example in enumerate(examples):
        k = len(example["code_ids"])
        index[row, :k] = torch.tensor(example["code_ids"], dtype=torch.long)
        mask[row, :k] = True
        target[row, :k] = torch.tensor(example["target"], dtype=torch.float32)
    head = backbone.model.lm_head.weight
    rows = head[index.to(head.device)].float()  # [B, K, d] — 코드 행만
    logits = torch.einsum("bkd,bd->bk", rows, states.to(rows.device))
    mask, target = mask.to(logits.device), target.to(logits.device)
    logits = logits.masked_fill(~mask, float("-inf"))
    logp = torch.where(mask, torch.log_softmax(logits, dim=-1), torch.zeros_like(logits))
    ce = -(target * logp).sum(dim=-1)
    entropy = -torch.special.xlogy(target, target).sum(dim=-1)
    probs = torch.softmax(logits.detach(), dim=-1).cpu()
    out = {
        "kl": ce - entropy,
        "ce": ce,
        "probs": [probs[row, : len(e["code_ids"])].tolist() for row, e in enumerate(examples)],
    }
    if code_mass:
        full = head32 if head32 is not None else head.float()
        lse_all = torch.logsumexp(states.to(full.device) @ full.T, dim=-1)  # [B] — 전체 어휘
        log_mass = torch.clamp(torch.logsumexp(logits, dim=-1) - lse_all.to(logits.device), max=0.0)
        out["log_code_mass"] = log_mass
        out["code_mass"] = log_mass.detach().exp().cpu().tolist()
    return out


# --------------------------------------------------------------------------
# 모델
# --------------------------------------------------------------------------


def _load_backbone(config: dict[str, Any], tokenizer: Any) -> Any:
    from robo_jev.model.backbone_qwen import QwenBackbone

    dtype = DTYPES[config["dtype"]]
    if config["model_id"] == TINY_MODEL:
        backbone = QwenBackbone.tiny(seed=int(config["tiny_seed"]), vocab_size=tokenizer.get_vocab_size())
        backbone.model.to(dtype)
        return backbone
    return QwenBackbone.load(config["model_id"], dtype=dtype, device=config["device"])


def base_identity(backbone: Any) -> dict[str, Any]:
    """기반 모델의 정체: id(소형은 seed가 든 이름), 가중치 digest(실제 모델의 manifest), dtype, 어휘 크기."""
    manifest = getattr(backbone, "manifest", None) or {}
    return {
        "model_id": str(backbone.model_id), "digest": manifest.get("digest"), "revision": manifest.get("revision"),
        "dtype": str(backbone.dtype).replace("torch.", ""), "vocab_size": int(backbone.config.vocab_size),
    }


def _trainable(backbone: Any, config: dict[str, Any]) -> list[tuple[str, torch.Tensor]]:
    """학습 대상을 정한다 — LoRA는 adapter만(fp32), 전체는 Qwen3_5ForCausalLM의 모든 파라미터(묶인 가중치는 한 번). (이름, 파라미터)."""
    if config["mode"] == "lora":
        attach_lora(backbone, config["lora"])
        if config["lora_dtype"] != "float32":  # bf16 adapter: 계산은 bf16, 갱신은 fp32 master (_optimizer가 MasterWeightAdamW를 고른다)
            for name, p in backbone.text.named_parameters():
                if "lora_" in name:
                    p.data = p.data.to(DTYPES[config["lora_dtype"]])
        return [(name, p) for name, p in backbone.text.named_parameters() if p.requires_grad]
    backbone.model.requires_grad_(True)
    return list(backbone.model.named_parameters())


def _optimizer(named: list[tuple[str, torch.Tensor]], config: dict[str, Any]) -> torch.optim.Optimizer:
    """AdamW — 2차원 이상에만 weight decay. fp32가 아닌 학습 대상이 있으면(전체 미세조정의 bf16) :class:`MasterWeightAdamW`."""
    kwargs = {"lr": float(config["lr"]), "betas": tuple(config["betas"]), "eps": float(config["eps"])}
    masters = {name: p.detach().clone().to(torch.float32) for name, p in named if p.dtype != torch.float32}
    pairs: list[tuple[str, torch.Tensor, torch.Tensor]] = []

    def swap(name: str, p: torch.Tensor) -> torch.Tensor:
        if name not in masters:
            return p
        pairs.append((name, p, masters[name]))
        return masters[name]

    decay = [swap(n, p) for n, p in named if p.dim() >= 2]
    rest = [swap(n, p) for n, p in named if p.dim() < 2]
    groups = []
    if decay:
        groups.append({"name": "decay", "params": decay, "weight_decay": float(config["weight_decay"])})
    if rest:
        groups.append({"name": "no_decay", "params": rest, "weight_decay": 0.0})
    if not masters:
        return torch.optim.AdamW(groups, **kwargs)
    return MasterWeightAdamW(groups, pairs=pairs, **kwargs)


def _model_state(backbone: Any, mode: str) -> dict[str, torch.Tensor]:
    """저장할 가중치 — LoRA는 adapter 값(text 모델 기준 이름), 전체는 Qwen3_5ForCausalLM의 state dict.

    복사하지 않고 살아 있는 tensor를 가리킨다(곧바로 ``torch.save``한다) — 묶인 가중치(embedding = LM head)가 한 저장소로 한 번만 쓰인다
    (2B에서 사본을 만들면 1 GB가 두 번 들어간다)."""
    if mode == "lora":
        return {name: p.detach() for name, p in backbone.text.named_parameters() if "lora_" in name}
    return {name: t.detach() for name, t in backbone.model.state_dict().items()}


def _load_model_state(backbone: Any, mode: str, saved: dict[str, torch.Tensor]) -> None:
    if mode == "lora":
        named = {name: p for name, p in backbone.text.named_parameters() if "lora_" in name}
        if set(named) != set(saved):
            missing, extra = sorted(set(named) - set(saved)), sorted(set(saved) - set(named))
            raise ValueError(f"LoRA adapter의 이름이 맞지 않는다 (없음 {missing[:3]}, 남음 {extra[:3]}) — 같은 LoRA 설정으로 붙였는가")
        with torch.no_grad():
            for name, p in named.items():
                p.copy_(saved[name].to(device=p.device, dtype=p.dtype))
        return
    result = backbone.model.load_state_dict({k: v for k, v in saved.items()}, strict=True)
    if result.missing_keys or result.unexpected_keys:  # pragma: no cover — strict=True가 이미 거절한다
        raise ValueError(f"전체 가중치의 키가 맞지 않는다: {result}")


def estimate_step_memory(backbone: Any, trainable: list[tuple[str, torch.Tensor]], config: dict[str, Any], *, allocated: int) -> dict[str, Any]:
    """학습 step의 장치 메모리 추정 (바이트): 지금 할당된 것(가중치·fp32 master·고정 head 사본) + optimizer 상태(gradient, AdamW 모멘트 둘,
    master가 있으면 step 안의 fp32 gradient 버퍼) + micro-batch activation(:data:`ACTIVATION_BYTES_PER_TOKEN` × hidden·층 비례 × 토큰 + 1 GB)
    + 전체 미세조정의 보조 항이 micro-batch마다 만드는 head의 fp32 사본과 그 gradient. 보수적으로 더한다(동시에 다 들지는 않는다)."""
    cfg = backbone.text.config
    scale = (int(cfg.hidden_size) * int(cfg.num_hidden_layers)) / _ACTIVATION_REFERENCE
    per_token = ACTIVATION_BYTES_PER_TOKEN["checkpointed" if config["gradient_checkpointing"] else "not_checkpointed"] * scale
    activations = 1e9 + per_token * int(config["micro_batch_tokens"])
    optimizer = 0.0
    for _, p in trainable:
        master = p.dtype != torch.float32
        optimizer += p.numel() * (p.element_size() + 8 + (4 if master else 0))
    head = backbone.model.lm_head.weight
    aux = 2 * head.numel() * 4 if config["mode"] == "full" and config["code_mass_weight"] > 0 else 0
    total = float(allocated) + optimizer + activations + aux
    return {"allocated": int(allocated), "optimizer": int(optimizer), "activations": int(activations), "aux_head": int(aux), "total": int(total),
            "per_token": per_token, "checkpointed": bool(config["gradient_checkpointing"])}


def check_step_memory(estimate: dict[str, Any], *, limit_bytes: float, config: dict[str, Any], log: Callable[[str], None] | None = None) -> dict[str, Any]:
    """추정(:func:`estimate_step_memory`)이 울타리(`limit_bytes`)의 :data:`FENCE_SHARE` 를 넘으면 ``ValueError`` — `allow_large_activations`면 경고만.
    돌려주는 것: 추정에 울타리·문턱을 더한 dict (manifest에 남는다)."""
    out = {**estimate, "fence_bytes": int(limit_bytes), "refuse_above": int(FENCE_SHARE * limit_bytes)}
    if out["total"] > FENCE_SHARE * limit_bytes:
        message = (
            f"step 메모리 추정 {out['total'] / 1e9:.1f} GB > 울타리 {limit_bytes / 1e9:.1f} GB의 {FENCE_SHARE:.0%} "
            f"(activation {out['activations'] / 1e9:.1f} GB — micro_batch_tokens {config['micro_batch_tokens']}, "
            f"gradient_checkpointing {config['gradient_checkpointing']}) — micro-batch를 줄이거나 checkpointing을 켠다"
        )
        if not config["allow_large_activations"]:
            raise ValueError(message + " (그래도 돌리려면 allow_large_activations: true)")
        if log:
            log("경고: " + message + " — allow_large_activations: true로 계속한다")
        out["override"] = True
    return out


def disk_free_bytes(path: str | Path) -> int:
    """`path`(없으면 가장 가까운 있는 상위 디렉터리)가 있는 파일 시스템의 여유 바이트."""
    target = Path(path)
    while not target.exists() and target != target.parent:
        target = target.parent
    return int(shutil.disk_usage(target).free)


def _git_sha() -> str | None:
    try:
        root = Path(__file__).resolve().parents[3]
        return subprocess.run(["git", "-C", str(root), "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True, timeout=10).stdout.strip() or None
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None


def summarize_dev(
    examples: Sequence[dict[str, Any]], *, kl: Sequence[float], probs: Sequence[Sequence[float]], mass: Sequence[float], type_weights: dict[str, float]
) -> dict[str, Any]:
    """dev 예(질문 × 균형 배치)마다의 KL·코드 확률·코드 질량 → dev 요약 (모듈 설명): 손실(종류 평균의 가중 평균), 종류별 KL·코드 질량,
    0.5 아래 코드 질량의 몫(배치 단위), noul·score의 평균 이동(주 지표)·판정(``>= 0.5``) 바뀜·결정적인 바뀜(두 값 모두 0.5에서
    :data:`robo_jev.decisions.metrics.DECISIVE_MARGIN` 이상)."""
    from robo_jev.decisions.metrics import DECISIVE_MARGIN, FLIP_THRESHOLD

    questions: dict[tuple[str, str], list[int]] = {}
    for i, example in enumerate(examples):
        questions.setdefault((example["record_id"], example["question"]), []).append(i)
    by_type: dict[str, list[float]] = {}
    mass_by_type: dict[str, list[float]] = {}
    shifts: dict[str, list[float]] = {"noul": [], "score": []}
    flips: dict[str, list[float]] = {"noul": [], "score": []}
    decisive: dict[str, list[float]] = {"noul": [], "score": []}

    def pair(kind: str, first: float, second: float) -> None:
        shifts[kind].append(first - second)
        flip = (first >= FLIP_THRESHOLD) != (second >= FLIP_THRESHOLD)
        flips[kind].append(float(flip))
        sure = min(abs(first - FLIP_THRESHOLD), abs(second - FLIP_THRESHOLD)) >= DECISIVE_MARGIN - 1e-9
        decisive[kind].append(float(flip and sure))

    for members in questions.values():
        kind = examples[members[0]]["type"]
        by_type.setdefault(kind, []).append(statistics.fmean(kl[i] for i in members))
        mass_by_type.setdefault(kind, []).append(statistics.fmean(mass[i] for i in members))
        if kind == "noul" and len(members) == 2:
            p_yes = {}
            for i in members:
                options = examples[i]["options"]
                p_yes[options[0]] = probs[i][options.index("yes")]
            if set(p_yes) == {"yes", "no"}:
                pair("noul", p_yes["yes"], p_yes["no"])
        if kind == "score" and len(members) == 2:
            values = {}
            for i in members:
                shown = examples[i]["values"]
                values["up" if shown[0] < shown[-1] else "down"] = sum(p * v for p, v in zip(probs[i], shown))
            if set(values) == {"up", "down"}:
                pair("score", values["up"], values["down"])
    kl_by_type = {kind: statistics.fmean(values) for kind, values in sorted(by_type.items())}
    weights = {kind: type_weights[kind] for kind in kl_by_type}
    denominator = sum(weights.values())

    def mean(values: list[float]) -> float | None:
        return statistics.fmean(values) if values else None

    return {
        "loss": (sum(weights[k] * v for k, v in kl_by_type.items()) / denominator) if denominator > 0 else None,
        "kl_by_type": kl_by_type, "questions_by_type": {kind: len(values) for kind, values in sorted(by_type.items())},
        "questions": len(questions), "prompts": len(examples),
        "code_mass": mean([v for values in mass_by_type.values() for v in values]),
        "code_mass_by_type": {kind: statistics.fmean(values) for kind, values in sorted(mass_by_type.items())},
        "code_mass_below_half": mean([1.0 if m < 0.5 else 0.0 for m in mass]),
        "noul_yes_first_shift": mean(shifts["noul"]), "noul_flip_rate": mean(flips["noul"]), "noul_decisive_flip_rate": mean(decisive["noul"]),
        "score_direction_shift": mean(shifts["score"]), "score_flip_rate": mean(flips["score"]), "score_decisive_flip_rate": mean(decisive["score"]),
    }


# --------------------------------------------------------------------------
# 학습기
# --------------------------------------------------------------------------


class DecisionTrainer:
    """판단 요청 학습 run 하나 (모듈 설명). ``run()``이 끝까지(또는 멈출 때까지) 돌린다."""

    def __init__(
        self,
        config: dict[str, Any],
        *,
        records: Sequence[dict[str, Any]] | None = None,
        tokenizer: Any = None,
        backbone: Any = None,
        resume: str | Path | None = None,
        log: Callable[[str], None] | None = print,
    ) -> None:
        self.config = resolve_config(config)
        self.log = log
        cfg = self.config
        validation = None
        if records is None:
            _need(cfg["train_input"] is not None, "train_input: 학습 JSONL 경로가 없다")
            records, validation = load_training_records(cfg["train_input"])
        records = list(records)
        _need(bool(records), "학습 기록이 하나도 없다")
        if tokenizer is None:
            from robo_jev.model.tokenizer import available_tokenizer, load_tokenizer

            tokenizer = load_tokenizer(available_tokenizer()[0])
        self.tokenizer = tokenizer
        self.train_records, dev_records = split_dev(records, fraction=cfg["dev_fraction"], salt=cfg["dev_salt"])
        _need(bool(self.train_records), "dev를 떼고 나니 학습 기록이 없다")
        self.dev_records = dev_records[: cfg["dev_max_records"]] if cfg["dev_max_records"] else dev_records
        self.dev_unused = len(dev_records) - len(self.dev_records)  # dev로 뗐지만 dev에도 학습에도 쓰지 않는 레코드 (리뷰 1 Minor 7 — manifest에 적는다)
        self.dev = dev_examples(self.dev_records, tokenizer, fmt=cfg["format"], version=cfg["prompt_version"])
        self.examples_per_epoch = examples_per_epoch(self.train_records, q=cfg["questions_per_request"])
        self.steps_per_epoch = math.ceil(self.examples_per_epoch / cfg["batch_examples"])
        if cfg["max_steps"] is None:
            cfg["max_steps"] = int(cfg["epochs"]) * self.steps_per_epoch
        self.max_steps = int(cfg["max_steps"])
        # 결정성: LoRA 초기화(peft의 kaiming)가 전역 torch generator에서 뽑는다
        random.seed(cfg["seed"])
        np.random.seed(cfg["seed"] % 2**32)
        torch.manual_seed(cfg["seed"])
        self.backbone = backbone if backbone is not None else _load_backbone(cfg, tokenizer)
        self.base = base_identity(self.backbone)
        self.trainable = _trainable(self.backbone, cfg)
        self.backbone.activation_checkpointing = bool(cfg["gradient_checkpointing"])
        self.optimizer = _optimizer(self.trainable, cfg)
        # 고정된 LM head의 fp32 사본 (LoRA: 코드 질량의 전체 어휘 logsumexp — 보조 항과 dev). 전체 미세조정은 head가 학습되므로 그때그때 만든다.
        head = self.backbone.model.lm_head.weight
        self._head32 = head.detach().float() if cfg["mode"] == "lora" else None
        self.memory_estimate = None
        if self.backbone.device.type == "cuda":
            estimate = estimate_step_memory(self.backbone, self.trainable, cfg, allocated=int(torch.cuda.memory_allocated()))
            fraction = float(torch.cuda.get_per_process_memory_fraction(self.backbone.device))
            limit = fraction * float(torch.cuda.get_device_properties(self.backbone.device).total_memory)
            self.memory_estimate = check_step_memory(estimate, limit_bytes=limit, config=cfg, log=self.log)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, lambda step: lr_factor(step, max_steps=self.max_steps, warmup_ratio=float(cfg["warmup_ratio"]))
        )
        from robo_jev.model.tokenizer import available_tokenizer, describe_tokenizer

        tokenizer_block = describe_tokenizer(available_tokenizer()[0])
        self.manifest = {
            "kind": TRAIN_KIND,
            "model": {
                "id": cfg["model_id"], "mode": cfg["mode"], "lora": cfg["lora"] if cfg["mode"] == "lora" else None,
                "lora_dtype": cfg["lora_dtype"] if cfg["mode"] == "lora" else None, "base": self.base,
                "trainable_parameters": int(sum(p.numel() for _, p in self.trainable)),
                "total_parameters": int(sum(p.numel() for p in self.backbone.model.parameters())),
                "state_scope": "text_model_lora" if cfg["mode"] == "lora" else "causal_lm_state_dict",
            },
            "tokenizer": {"id": tokenizer_block.get("id"), "sha256": tokenizer_block.get("sha256"), "revision": tokenizer_block.get("revision")},
            "prompt": {"version": cfg["prompt_version"], "digest": fixed_text_digest(cfg["prompt_version"]), "format": cfg["format"], "mode": "direct"},
            "objective": {"main": "soft-label KL over the shown code rows at the answer position", "code_mass_weight": cfg["code_mass_weight"],
                          "aux": "-log(code mass) at the answer position (full-vocabulary logsumexp, fp32 head)" if cfg["code_mass_weight"] > 0 else None},
            "selection": {"evaluated": "final checkpoint (pre-registered)", "best_on_dev": "recorded in metrics.json `best` only — not used for evaluation"},
            "memory_estimate": self.memory_estimate,
            "data": {
                "train_input": cfg["train_input"], "file_sha256": file_sha256(cfg["train_input"]) if cfg["train_input"] else None,
                "records_sha256": records_digest(records), "records": len(records), "validation": validation,
                "train_records": len(self.train_records), "dev_records": len(self.dev_records),
                "dev_ids": [str(r.get("id")) for r in self.dev_records], "dev_prompts": len(self.dev), "dev_unused_records": self.dev_unused,
                "examples_per_epoch": self.examples_per_epoch, "steps_per_epoch": self.steps_per_epoch, "mix": [],
            },
            "git": _git_sha(),
            "created": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        }
        self.run_id = cfg["run_name"] or f"decisions-v0-{cfg['mode']}-{dt.datetime.now():%Y%m%d-%H%M%S}"
        self.run_dir = Path(cfg["out_dir"]) / self.run_id
        self.step = 0
        self.history: list[dict[str, Any]] = []
        self.dev_history: list[dict[str, Any]] = []
        self.best: dict[str, Any] | None = None
        self.status = "running"
        self._plan_cache: tuple[int, list[tuple[int, str, int]]] | None = None
        self._started = time.perf_counter()
        if resume is not None:
            self._load(resume)

    # -- 데이터 --

    def mix_plan(self, epoch: int) -> list[tuple[int, str, int]]:
        """일반 능력 보호 갈고리 — 다른 판단 데이터를 섞을 자리(모듈 설명). 지금은 언제나 빈 목록이다.

        채울 때 할 일: 섞는 데이터의 기록(지금 계획의 index는 `train_records`를 가리킨다)과 그 몫을 정하고, epoch당 예 수
        (:attr:`examples_per_epoch` → `steps_per_epoch`·`max_steps`)에 섞는 예를 더하고, 재개 정체(manifest의 data)에 그 데이터의 digest를 넣는다."""
        return []

    def _plan(self, epoch: int) -> list[tuple[int, str, int]]:
        if self._plan_cache is None or self._plan_cache[0] != epoch:
            cfg = self.config
            plan = epoch_plan(self.train_records, seed=cfg["seed"], epoch=epoch, q=cfg["questions_per_request"], augment=cfg["augment"])
            self._plan_cache = (epoch, plan + self.mix_plan(epoch))
        return self._plan_cache[1]

    def step_examples(self, step: int) -> list[dict[str, Any]]:
        """step 번호 → 그 step의 학습 예들 (epoch 계획의 `batch_examples`개 조각; 토큰은 여기서 만든다)."""
        epoch, index = divmod(int(step), self.steps_per_epoch)
        size = self.config["batch_examples"]
        chunk = self._plan(epoch)[index * size : (index + 1) * size]
        out = []
        for record_index, name, order_seed in chunk:
            out.extend(build_examples(self.train_records[record_index], [name], self.tokenizer, order_seed=order_seed, fmt=self.config["format"],
                                      version=self.config["prompt_version"]))
            out[-1]["epoch"] = epoch
        return out

    # -- 학습 step --

    def step_loss(self, examples: list[dict[str, Any]], *, backward: bool) -> tuple[float, dict[str, Any]]:
        """한 step의 예들 → (손실 Σ (w·KL + λ·(−log 코드 질량)) / 예 수, 예별 값). `backward`면 micro-batch마다 backward해 gradient를 쌓는다.

        예별 값: ``kl``·``ce``·``code_mass``; 합: ``kl_loss``(Σ w·KL / n)·``aux_loss``(λ · Σ −log m / n) — ``kl_loss + aux_loss`` = 손실."""
        weights = [self.config["type_weights"][e["type"]] for e in examples]
        lam = float(self.config["code_mass_weight"])
        groups = pack_micro_batches([len(e["tokens"]) for e in examples], self.config["micro_batch_tokens"])
        kl: list[float] = [0.0] * len(examples)
        ce: list[float] = [0.0] * len(examples)
        mass: list[float] = [0.0] * len(examples)
        total = kl_total = aux_total = 0.0
        padded = 0
        for group in groups:
            batch = [examples[i] for i in group]
            with torch.set_grad_enabled(backward):
                out = code_kl(self.backbone, batch, code_mass=True, head32=self._head32)
                w = torch.tensor([weights[i] for i in group], dtype=torch.float32, device=out["kl"].device)
                kl_part = (w * out["kl"]).sum() / len(examples)
                aux_part = lam * (-out["log_code_mass"]).sum() / len(examples)
                loss = kl_part + aux_part if lam > 0 else kl_part
            if backward:
                loss.backward()
            total += float(loss.detach())
            kl_total += float(kl_part.detach())
            aux_total += float(aux_part.detach())
            for j, i in enumerate(group):
                kl[i], ce[i], mass[i] = float(out["kl"][j].detach()), float(out["ce"][j].detach()), float(out["code_mass"][j])
            padded += len(batch) * max(len(e["tokens"]) for e in batch)
        return total, {"kl": kl, "ce": ce, "code_mass": mass, "kl_loss": kl_total, "aux_loss": aux_total, "micro_batches": len(groups), "padded_tokens": padded}

    def train_step(self) -> dict[str, Any]:
        """step 하나: 예 만들기 → micro-batch마다 forward·backward → clip → optimizer → 일정. 지표를 history에 더하고 돌려준다."""
        started = time.perf_counter()
        examples = self.step_examples(self.step)
        lr = float(self.optimizer.param_groups[0]["lr"])
        loss, parts = self.step_loss(examples, backward=True)
        grad_norm = clip_gradients([p for _, p in self.trainable], self.config["gradient_clip"])
        self.optimizer.step()
        self.scheduler.step()
        self.optimizer.zero_grad(set_to_none=True)
        if self.backbone.device.type == "cuda":
            torch.cuda.synchronize()
        seconds = time.perf_counter() - started
        by_type: dict[str, list[float]] = {}
        for example, value in zip(examples, parts["kl"]):
            by_type.setdefault(example["type"], []).append(value)
        tokens = sum(len(e["tokens"]) for e in examples)
        metrics = {
            "step": self.step + 1, "epoch": examples[0]["epoch"] if examples else None, "loss": loss, "kl_loss": parts["kl_loss"],
            "aux_loss": parts["aux_loss"], "code_mass_mean": statistics.fmean(parts["code_mass"]) if examples else None,
            "kl_mean": statistics.fmean(parts["kl"]) if examples else None,
            "kl_by_type": {kind: statistics.fmean(values) for kind, values in sorted(by_type.items())},
            "examples_by_type": {kind: len(values) for kind, values in sorted(by_type.items())},
            "examples": len(examples), "clamped": sum(1 for e in examples if e.get("clamped")), "tokens": tokens,
            "padded_tokens": parts["padded_tokens"], "micro_batches": parts["micro_batches"], "grad_norm": grad_norm, "lr": lr,
            "seconds": seconds, "tokens_per_second": tokens / seconds if seconds > 0 else None,
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()) if self.backbone.device.type == "cuda" else None,
        }
        self.step += 1
        self.history.append(metrics)
        return metrics

    # -- dev --

    def evaluate_dev(self) -> dict[str, Any]:
        """dev 손실(모든 질문 × 균형 배치)·코드 질량·위치 쏠림 — 모듈 설명. gradient 없이 길이로 묶어 읽는다.

        요약은 :func:`summarize_dev`."""
        started = time.perf_counter()
        budget = self.config["dev_batch_tokens"] or self.config["micro_batch_tokens"]
        kl = [0.0] * len(self.dev)
        mass = [0.0] * len(self.dev)
        probs: list[list[float]] = [[] for _ in self.dev]
        with torch.no_grad():
            head32 = self._head32 if self._head32 is not None else self.backbone.model.lm_head.weight.detach().float()
            for group in pack_micro_batches([len(e["tokens"]) for e in self.dev], budget):
                out = code_kl(self.backbone, [self.dev[i] for i in group], code_mass=True, head32=head32)
                for j, i in enumerate(group):
                    kl[i], probs[i], mass[i] = float(out["kl"][j]), out["probs"][j], float(out["code_mass"][j])
            del head32
        return {**summarize_dev(self.dev, kl=kl, probs=probs, mass=mass, type_weights=self.config["type_weights"]), "seconds": time.perf_counter() - started}

    def _dev_round(self) -> bool:
        """dev를 재고 기록한다; 고르기(keep_best)·멈추기(patience)를 판정한다 → 멈춰야 하면 True."""
        result = dict(self.evaluate_dev())
        result["step"] = self.step
        self.dev_history.append(result)
        if self.log:
            self.log(json.dumps({"dev": {k: v for k, v in result.items() if k != "seconds"}}, ensure_ascii=False, default=float))
        loss = result.get("loss")
        if loss is None or self.step == 0:
            return False  # step 0(학습 전)은 기준값일 뿐 — 고르기의 후보가 아니다
        if self.best is None or loss < self.best["loss"]:
            path = self.run_dir / "checkpoint-best.pt"
            self.best = {"step": self.step, "loss": float(loss), "checkpoint": str(path) if self.config["keep_best"] else None,
                         "role": "record only — the evaluated model is the final checkpoint (selection: final, pre-registered)"}
            if self.config["keep_best"]:
                self.save(path)
            return False
        patience = self.config["early_stop_patience"]
        if patience is None:
            return False
        since = [d for d in self.dev_history if d["step"] > self.best["step"] and d.get("loss") is not None]
        return len(since) >= int(patience)

    # -- run --

    def run(self) -> dict[str, Any]:
        """`max_steps`까지(또는 멈출 때까지) 돌리고 체크포인트·metrics.json을 쓴다."""
        cfg = self.config
        dev_every = int(cfg["dev_every"])
        if self.backbone.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()  # step의 peak은 학습만의 것(적재의 peak은 실행기가 따로 적는다)
        if self.dev and dev_every and self.step == 0 and not self.dev_history:
            self._dev_round()  # 학습 전 기준값
        deadline = None if cfg["max_wall_hours"] is None else self._started + float(cfg["max_wall_hours"]) * 3600.0
        while self.step < self.max_steps:
            if deadline is not None and time.perf_counter() >= deadline:
                self.status = "interrupted"
                break
            metrics = self.train_step()
            if self.log:
                self.log(json.dumps({k: metrics[k] for k in ("step", "epoch", "loss", "kl_by_type", "grad_norm", "lr", "tokens", "seconds")}, ensure_ascii=False))
            stop = False
            if self.dev and dev_every and (self.step % dev_every == 0 or self.step == self.max_steps):
                stop = self._dev_round()
            if stop:
                self.status = "stopped_early"
                break
            if cfg["stop_after_steps"] is not None and self.step == int(cfg["stop_after_steps"]) and self.step < self.max_steps:
                self.status = "interrupted"
                break
            if cfg["save_checkpoints"] and cfg["checkpoint_every"] and self.step % int(cfg["checkpoint_every"]) == 0 and self.step < self.max_steps:
                self.save()
            if cfg["metrics_every"] and self.step % int(cfg["metrics_every"]) == 0:
                self.write_metrics()
        if self.step >= self.max_steps and self.status == "running":
            self.status = "completed"
        path = self.save() if cfg["save_checkpoints"] else None
        self.write_metrics()
        return {"run_id": self.run_id, "run_dir": str(self.run_dir), "checkpoint": str(path) if path else None, "step": self.step, "status": self.status}

    # -- 저장·재개 --

    def checkpoint_state(self) -> dict[str, Any]:
        return {
            "format": CHECKPOINT_FORMAT, "run_id": self.run_id, "step": self.step, "status": self.status,
            "model": _model_state(self.backbone, self.config["mode"]),
            "optimizer": self.optimizer.state_dict(), "scheduler": self.scheduler.state_dict(), "rng": collect_rng_state(),
            "sampler": {"kind": "decisions-epoch-plan-v0", "seed": self.config["seed"], "steps_per_epoch": self.steps_per_epoch,
                        "examples_per_epoch": self.examples_per_epoch, "next_step": self.step},
            "progress": None, "config": copy.deepcopy(self.config), "manifest": copy.deepcopy(self.manifest),
            "history": copy.deepcopy(self.history), "dev_history": copy.deepcopy(self.dev_history), "best": copy.deepcopy(self.best),
        }  # fmt: skip

    def save(self, path: str | Path | None = None) -> Path:
        """체크포인트를 atomic하게 쓴다 — 먼저 디스크 여유가 `min_free_disk_gb` 이상인지 본다(모자라면 쓰지 않고 :class:`DiskSpaceError`)."""
        target = Path(path) if path is not None else self.run_dir / "checkpoint.pt"
        floor = float(self.config["min_free_disk_gb"]) * 1e9
        free = disk_free_bytes(target.parent)
        if free < floor:
            raise DiskSpaceError(f"{target}: 디스크 여유 {free / 1e9:.1f} GB < 문턱 {floor / 1e9:.0f} GB — 체크포인트를 쓰지 않는다")
        save_checkpoint(target, self.checkpoint_state())
        return target

    def _load(self, path: str | Path) -> None:
        state = load_checkpoint(path)
        manifest = state.get("manifest") or {}
        _need(manifest.get("kind") == TRAIN_KIND, f"resume: {path}는 {TRAIN_KIND} 체크포인트가 아니다 (kind {manifest.get('kind')!r})")
        saved = {k: v for k, v in (state.get("config") or {}).items() if k not in RESUME_FREE_KEYS}
        current = {k: v for k, v in self.config.items() if k not in RESUME_FREE_KEYS}
        different = sorted(k for k in set(saved) | set(current) if saved.get(k) != current.get(k))
        _need(not different, f"resume: 체크포인트의 설정과 다르다: {different} (달라도 되는 것: {list(RESUME_FREE_KEYS)})")
        for block, keys in (("model", ("id", "mode", "lora", "base")), ("tokenizer", ("sha256",)), ("prompt", ("version", "digest", "format")),
                            ("data", ("records_sha256", "records", "train_records", "dev_records"))):  # fmt: skip
            for key in keys:
                a, b = (manifest.get(block) or {}).get(key), (self.manifest.get(block) or {}).get(key)
                _need(a == b, f"resume: {block}.{key}가 체크포인트와 다르다 ({str(a)[:40]} vs {str(b)[:40]}) — 같은 run이 아니다 (데이터·모델이 바뀌면 새 run)")
        self.run_id = str(state["run_id"])
        self.run_dir = Path(self.config["out_dir"]) / self.run_id
        _load_model_state(self.backbone, self.config["mode"], state["model"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.scheduler.load_state_dict(state["scheduler"])
        restore_rng_state(state["rng"])
        self.step = int(state["step"])
        self.history = list(state.get("history") or [])
        self.dev_history = list(state.get("dev_history") or [])
        self.best = state.get("best")
        self.status = "running"
        self.optimizer.zero_grad(set_to_none=True)
        self.manifest["resumed_from"] = {"path": str(path), "step": self.step, "created": manifest.get("created")}

    def summary(self) -> dict[str, Any]:
        steady = self.history[2:] if len(self.history) > 4 else self.history
        tokens = sum(h["tokens"] for h in steady)
        seconds = sum(h["seconds"] for h in steady)
        return {
            "last_loss": self.history[-1]["loss"] if self.history else None,
            "last_dev": self.dev_history[-1] if self.dev_history else None,
            "best": self.best,
            "tokens_per_second_steady": tokens / seconds if seconds > 0 else None,
            "steady_steps": len(steady),
            "seconds_per_step_steady": seconds / len(steady) if steady else None,
            "peak_allocated_bytes": max((h.get("peak_allocated_bytes") or 0 for h in self.history), default=None),
            "seconds_total": time.perf_counter() - self._started,
        }

    def write_metrics(self) -> Path:
        """``metrics.json``(run_id·status·step·config(풀어 둔 max_steps 포함)·manifest·steps·dev·summary)과 ``config.yaml``을 atomic하게 쓴다."""
        import yaml

        self.run_dir.mkdir(parents=True, exist_ok=True)
        (self.run_dir / "config.yaml").write_text(yaml.safe_dump(self.config, allow_unicode=True, sort_keys=False), encoding="utf-8")
        payload = {
            "run_id": self.run_id, "status": self.status, "step": self.step, "config": self.config, "manifest": self.manifest,
            "steps": self.history, "dev": self.dev_history, "summary": self.summary(),
        }
        path = self.run_dir / "metrics.json"
        temp = path.with_name(path.name + ".tmp")
        temp.write_text(json.dumps(payload, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
        os.replace(temp, path)
        return path


# --------------------------------------------------------------------------
# 평가 적재
# --------------------------------------------------------------------------


def apply_checkpoint(backbone: Any, path: str | Path, *, tokenizer_sha256: str | None, prompt_version: str = PROMPT_VERSION) -> dict[str, Any]:
    """학습한 가중치를 기반 backbone에 얹는다 (모듈 설명의 "평가 적재"). 돌려주는 것: 경로·run id·step·status·방식·형식·프롬프트 판.

    기반 모델의 정체(:func:`base_identity`), tokenizer 해시(`tokenizer_sha256`; None이면 대조하지 않는다), 프롬프트 판(평가가 쓸
    `prompt_version`)과 그 판의 고정 문구 digest가 학습 때와 다르면 ``ValueError`` — 학습한 모델은 같은 프롬프트를 기대한다.
    """
    state = load_model_checkpoint(path, mmap=True)  # mmap: 전체 미세조정의 재개용 파일(≈ 26 GB)에서 `model`만 페이지가 올라온다
    manifest = state.get("manifest") or {}
    _need(manifest.get("kind") == TRAIN_KIND, f"{path}: {TRAIN_KIND} 체크포인트가 아니다 (kind {manifest.get('kind')!r})")
    model = manifest.get("model") or {}
    current = base_identity(backbone)
    saved_base = model.get("base") or {}
    differences = [key for key in ("model_id", "digest", "dtype", "vocab_size") if saved_base.get(key) != current.get(key)]
    _need(not differences, f"{path}: 기반 모델(base)이 학습 때와 다르다: {[(k, saved_base.get(k), current.get(k)) for k in differences]}")
    saved_tokenizer = (manifest.get("tokenizer") or {}).get("sha256")
    _need(tokenizer_sha256 is None or saved_tokenizer == tokenizer_sha256, f"{path}: tokenizer 해시가 학습 때와 다르다 ({saved_tokenizer} vs {tokenizer_sha256})")
    prompt = manifest.get("prompt") or {}
    _need(prompt_version in PROMPT_VERSIONS, f"prompt_version: {list(PROMPT_VERSIONS)} 중 하나 (받은 값: {prompt_version!r})")
    _need(
        prompt.get("version") == prompt_version and prompt.get("digest") == fixed_text_digest(prompt_version),
        f"{path}: prompt 판·고정 문구가 학습 때와 다르다 (학습 {prompt.get('version')}/{str(prompt.get('digest'))[:12]} vs 평가 "
        f"{prompt_version}/{fixed_text_digest(prompt_version)[:12]}) — 같은 판으로 평가한다 (--prompt-version)",
    )
    mode = model.get("mode")
    _need(mode in MODES, f"{path}: 알 수 없는 학습 방식 {mode!r}")
    if mode == "lora":
        attach_lora(backbone, model["lora"])
    _load_model_state(backbone, mode, state["model"])
    backbone.model.requires_grad_(False)
    backbone.model.eval()
    return {
        "path": str(path), "run_id": state.get("run_id"), "step": state.get("step"), "status": state.get("status"), "mode": mode,
        "format": state.get("format"), "lora": model.get("lora"), "base": saved_base, "prompt_version": prompt.get("version"),
        "train_records_sha256": (manifest.get("data") or {}).get("records_sha256"),
    }
