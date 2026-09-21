"""학습 — 실제 가중치 업데이트, truncated BPTT, 혼합 sampler, atomic checkpoint와 정확한 재개 (docs/06 Task 5, docs/03 §4–§5, docs/08 §3.1·§8).

``train(config) -> dict`` 는 run id·checkpoint 경로·마지막 step·지표를 돌려준다. 설정은
``configs/train/tiny_cpu.yaml``(계획서 YAML의 키를 그대로 씀)이고, 이 파일은 단일 프로세스·CPU용이다:
분산 코드는 없지만 step의 구조는 model 둘레에 FSDP wrapper를 씌워도 loop를 바꾸지 않게 두었다
(model은 :func:`build_model` 한 곳에서 만들고 loop는 ``model(batch)``·``loss.backward()``·
:func:`clip_gradients`·``optimizer.step()`` 만 부른다).

**한 step (docs/03 §5, docs/04 §2 — 판정 e086c90).** step은 ``gradient_accumulation``개의 accumulation
단위로 되어 있고 **로봇 스트림 단위(에피소드 하나 = TBPTT 구간열)와 비로봇 단위(단일 요청을
``nonrobot_tokens_per_unit``까지 묶은 microbatch, P0 경로가 한 forward로 돈다)를 번갈아** 넣는다 — 한
step에 두 종류가 다 든다. step 손실은 ``robot_loss_share·L_robot + (1 − robot_loss_share)·L_nonrobot``
(시작 0.6/0.4)이다. ``L_robot``은 step의 로봇 상태(틱)들의 틱 종류 가중 평균 ``Σ_t w_t L_t / Σ_t w_t``,
``L_nonrobot``은 step의 비로봇 상태들의 평균이다(둘 다 유효 라벨이 있는 상태만). 분모(step의 로봇 틱
가중치 합, 비로봇 상태 수)는 라벨만으로 **단위를 돌리기 전에** 정하므로 accumulation이 이 식을 정확히
재현한다: 단위의 기여 = 그 분야의 비중 / 그 분야의 분모 × (틱이면 ``w_t L_t``, 상태면 ``L_s``). 한 분야가
step에 없으면(한쪽 데이터만) 있는 분야가 비중 1을 받는다.

**truncated BPTT (docs/08 §8, docs/03 §5).** 에피소드는 모의 시각 기준 ``stream_chunk_seconds`` 구간으로
나뉘고(:func:`episode_chunks`), 구간은 **같은 optimizer step 안에서** 차례로 forward·backward된다.
구간 경계에서는 DeltaNet recurrent·conv 상태와 윈도우 KV(prefix KV·정적 후보 hidden 포함)를
**detach**해 다음 구간에 넘긴다(:func:`detach_stream_state`) — 상태의 값은 추론의 증분 계산과 같고
gradient만 경계에서 끊긴다. 에피소드 손실은 ``L_episode = Σ_t w_t L_t / Σ_t w_t`` (유효 라벨이 있는
틱, ``w_t`` = 틱 종류의 가중치)이며, 구간의 손실은 그 부분합을 **에피소드 전체의 분모**로 정규화한
것이라 구간 손실의 합 = 에피소드 손실이다. 한 에피소드 = accumulation 단위 하나.

**sampler와 유효 loss 비중 (docs/04 §2).** :class:`robo_jev.sampler.MixedSampler` 가 두 종류를 번갈아
뽑고 기존 자료/오류 계열/새 의미 계열을 70/20/10으로 나누며, 틱 종류 가중치가 정상 유지 틱을
하향·이벤트/목표 변경 틱을 상향한다. step 지표에 loss·분야별 평균 loss·질문 타입별 loss·gradient
norm·토큰 수와 함께 **실현 토큰 비중**과 **유효 loss 비중** 둘을 적는다: ``loss_share`` = 그 축이 step
손실에서 실제로 받은 **계수 질량**(분야는 0.6/0.4 그대로 — 한쪽이 없으면 1.0; 묶음·틱 종류는 그
안의 배분), ``loss_contribution`` = 그 축의 기여 **값**이 step 손실 값에서 차지하는 몫.

**정밀도 (docs/03 §5).** 실제 backbone은 BF16으로 계산하고 readout은 fp32다. 거기에 더해, 학습 대상 가운데 fp32가
**아닌** 파라미터(= BF16 backbone을 통째로 학습하는 T1)는 optimizer가 **fp32 master 사본**을 들고 fp32로 갱신한 뒤
bf16으로 되쓴다(:class:`MasterWeightAdamW`, 설정 `fp32_master_weights`, 기본 켜짐). 이것이 없으면 |p| ≈ 0.03의
가중치에서 한 step의 1e-5가 bf16 눈금 2⁻¹³ = 1.22e-4에 반올림돼 **사라진다** — P1이 그 조건으로 돌았고, 40 step 뒤
임베딩을 뺀 표본의 23.32 %만 움직였으며 움직인 원소의 평균 |Δ|는 반올림 없는 기대치의 7~9 %였다. readout과 LoRA는
이미 fp32라 사본을 만들지 않는다.

**저장·재개 (docs/03 §5).** step 사이에서는 model/optimizer/scheduler/RNG/sampler 위치/config/manifest를,
step 도중(구간 경계)에서는 여기에 진행 위치(단위·구간 index), 누적 gradient, 이어 붙일 공통 상태를
더해 :func:`robo_jev.checkpoint.save_checkpoint` 로 atomic하게 쓴다(fp32 master 사본은 optimizer의 `state_dict`에
함께 들어간다 — bf16 파라미터에서 되살릴 수 없는 정밀도다). 재개는 그 위치의 다음 구간부터
이어가며, 같은 seed의 연속 실행과 FP32·CPU에서 비트 단위로 같아야 한다(tests/test_resume.py). 중단은
``stop_after``(결정적 검사용)나 ``max_wall_hours``(예산)로 구간 경계에서 일어난다.

**run의 정체 (리뷰 11 S1).** 재개는 두 가지를 대조한다. (1) 설정 — 중단·예산·경로·이름(:data:`RESUME_FREE_KEYS`)과
내용으로 대조하는 경로 키(:data:`RESUME_PATH_KEYS`: 모델 설정 파일, tokenizer; `dataset_manifests`의 경로)를 뺀
나머지는 문자 그대로 같아야 한다. (2) manifest의 **identity 블록**(:func:`manifest_identity`) — 데이터 manifest의
sha256과 파일별 sha256·분야 태그·레코드 수, tokenizer의 종류·파일 sha256·id·revision, 토큰 직렬화 버전, 질문
세트와 표지, layout, 실제로 만든 모델(종류·설정 파일 sha256·이름·어휘·seed·readout·rank·파라미터 수). 경로는
정체가 아니고 내용이 정체다: 같은 데이터의 사본을 다른 경로에 두고 재개해도 되지만, 레코드 하나가 바뀐 데이터
(manifest 해시를 맞춰도)에는 옛 run을 이어 붙이지 않는다 — 달라진 키를 모두 이름으로 적고 거절한다. sampler
위치도 자기 index가 가리키는 레코드의 출처(파일별 sha256)를 함께 대조한다(:mod:`robo_jev.sampler`).

이 모듈은 generator·simulator·하네스를 import하지 않는다 (docs/06 §1).
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import json
import math
import random
import subprocess
import sys
import time
from collections.abc import Iterator
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch import Tensor

from robo_jev.checkpoint import (
    CHECKPOINT_FORMAT,
    collect_rng_state,
    load_checkpoint,
    restore_rng_state,
    save_checkpoint,
    stream_state_from_dict,
    stream_state_to_dict,
)
from robo_jev.contracts import QUESTION_SET_V0
from robo_jev.loss import judgment_loss, question_losses
from robo_jev.model.backbone_qwen import DEFAULT_WINDOW_CAPACITY, QwenBackbone, candidate_ids
from robo_jev.model.contract_digest import contract_differences, contract_digest
from robo_jev.model.hybrid import DEFAULT_CONFIG, TinyHybrid
from robo_jev.model.judge import Judge
from robo_jev.model.serialize import TOKEN_SERIALIZER_VERSION
from robo_jev.model.stream import StreamState
from robo_jev.model.tokenizer import WhitespaceTokenizer, describe_tokenizer, load_tokenizer
from robo_jev.sampler import (
    DEFAULT_LAYOUTS,
    DEFAULT_MATERIAL_SHARES,
    DOMAINS,
    MATERIALS,
    TICK_CLASSES,
    Item,
    MixedSampler,
    Unit,
    load_manifests,
    manifest_files,
    sha256_of,
    tick_class,
    valid_label_ticks,
    valid_single,
)

__all__ = [
    "ChunkResult",
    "EpisodePlan",
    "MASTER_WEIGHTS_KEY",
    "MODEL_IDS",
    "MasterWeightAdamW",
    "RESUME_FREE_KEYS",
    "RESUME_PATH_KEYS",
    "Trainer",
    "build_model",
    "build_optimizer",
    "clip_gradients",
    "detach_stream_state",
    "episode_chunks",
    "fp32_master_weights",
    "identity_differences",
    "layout_prefix",
    "lr_factor",
    "main",
    "manifest_identity",
    "model_block",
    "parameter_groups",
    "plan_episode",
    "resolve_config",
    "resume_config",
    "run_single_unit",
    "run_stream_chunk",
    "tokenizer_block",
    "train",
    "load_readout_checkpoint",
    "trainable_state_dict",
]

# --------------------------------------------------------------------------
# 설정
# --------------------------------------------------------------------------

READOUTS = {"decision_pointer": "pointer", "candidate_branch": "candidate_branch"}
DTYPES = {"float32": torch.float32, "float64": torch.float64, "bfloat16": torch.bfloat16}
READOUT_DTYPES = {"float32": torch.float32, "bfloat16": torch.bfloat16}
TRAINABLE = ("readout_only", "text_backbone_and_readout", "lora_and_readout")
EXECUTION_BACKENDS = ("independent_paths",)  # P0. `shared_hybrid`(P1)는 state_first에 아직 없다 (4b 보고 §5)
OPTIMIZERS = ("adamw",)
DEVICES = ("cpu", "cuda")
#: 만들 수 있는 모델: 4b의 소형 계산 fixture와 `configs/model/candidates.yaml`의 실제 backbone(Qwen3.5 계열, G0b의
#: :class:`robo_jev.model.backbone_qwen.QwenBackbone` adapter). 다른 id는 설정 단계에서 거절한다(리뷰 11 S2).
FIXTURE_MODEL_ID = "tiny_hybrid"
MODEL_IDS = (FIXTURE_MODEL_ID, *candidate_ids())
#: 실제 backbone의 기본 readout rank (fixture는 설정 파일의 값).
DEFAULT_REAL_READOUT_RANK = 64
#: LoRA 기본값 (analysis-nimble §3-1: r=16, α=32, attention·MLP projection) — `lora:` 블록이 덮어쓴다.
DEFAULT_LORA = {"r": 16, "alpha": 32, "dropout": 0.0, "targets": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "in_proj_qkv", "in_proj_z", "out_proj"]}
DEFAULT_TICK_WEIGHTS = {"steady": 0.25, "event": 2.0, "goal_change": 2.0, "other": 1.0}
DEFAULT_SAMPLER = {
    "material_shares": dict(DEFAULT_MATERIAL_SHARES),
    "domain_tag": "provenance.domain",
    "material_tag": "provenance.material",
    "tick_weights": dict(DEFAULT_TICK_WEIGHTS),
    "steady_min_held_ticks": 3,
    "permute_candidates_seed": None,  # 후보 순서 치환 증강 (null = 끔; 정수면 레코드·seed로 정해진 순열로 직렬화)
}

#: 재개할 때 checkpoint의 설정과 달라도 되는 키 — 중단·예산·경로·이름뿐이다(run id는 checkpoint의 것을
#: 쓴다). 나머지는 run의 정체라 같아야 한다.
RESUME_FREE_KEYS = ("resume", "stop_after", "max_wall_hours", "checkpoint_every", "artifacts_dir", "run_id", "run_name")
#: 값이 경로·이름인 키 — 문자 그대로가 아니라 **가리키는 내용**(manifest의 identity 블록: 설정 파일 sha256, tokenizer
#: 파일 sha256·id·revision)으로 대조한다. `dataset_manifests`도 경로는 내용(manifest·파일 sha256)으로, 태그는 그대로.
RESUME_PATH_KEYS = ("model_config", "tokenizer")
#: 실제로 만든 backbone 클래스 → 그 종류 (manifest의 `model.kind`). :func:`build_model`이 새 종류를 만들면 여기도 더한다.
_BACKBONE_KINDS = {"TinyHybrid": "tiny_hybrid", "QwenBackbone": "qwen3_5"}
#: identity 블록에 들어가는 tokenizer·모델 블록의 키 (:func:`manifest_identity`).
_TOKENIZER_IDENTITY = ("kind", "sha256", "id", "revision")
_MODEL_IDENTITY = ("kind", "class", "config_sha256", "name", "vocab_size", "seed", "readout", "rank", "parameters", "revision", "digest", "trainable", "lora")

DEFAULTS: dict[str, Any] = {
    "run_name": "run",
    "run_id": None,
    "model_id": "tiny_hybrid",
    "model_config": str(DEFAULT_CONFIG),
    "model_vocab_size": None,
    "model_seed": None,
    "model_revision_manifest": None,
    "dataset_manifest": None,  # manifest 하나 (= dataset_manifests: [그것]). 정규화 뒤에는 null이다
    "dataset_manifests": None,  # 여러 manifest: 경로 또는 {path, domain, material} — 로봇 batch + 비로봇 데이터를 한 run에
    "splits": ["train"],
    "tokenizer": "whitespace",
    "dtype": "float32",
    "execution_backend": "independent_paths",
    "readout": "decision_pointer",
    "layout": dict(DEFAULT_LAYOUTS),
    "stream_chunk_seconds": 10,
    "stream_window_ticks": 30,
    "stream_max_ticks": None,
    "trainable": "text_backbone_and_readout",
    "freeze_vision_encoder": True,
    "optimizer": "adamw",
    # 학습 대상 가운데 fp32가 아닌 파라미터(= BF16 backbone을 통째로 학습하는 T1)의 **fp32 master 사본**을 optimizer가
    # 들고 fp32로 갱신한 뒤 bf16으로 되쓴다. readout(이미 fp32)·LoRA(attach_lora가 fp32로 올린다)에는 사본이 생기지
    # 않는다. false는 P1이 돌린 조건(bf16 tensor를 AdamW가 직접 갱신 — 갱신폭이 bf16 격자에 반올림돼 사라진다)이다.
    "fp32_master_weights": True,
    "backbone_lr": 1e-5,
    "readout_lr": 1e-4,
    "weight_decay": 0.01,
    "gradient_clip": 1.0,
    "warmup_ratio": 0.05,
    "microbatch_states_per_rank": 1,
    "gradient_accumulation": 4,
    "robot_loss_share": 0.6,
    "nonrobot_tokens_per_unit": 8192,
    "world_size": 1,
    "max_total_tokens": 8192,
    "max_state_tokens": 2048,
    "activation_checkpointing": False,  # 실제 backbone의 층 단위 activation checkpointing (LoRA·full 학습; tiny_hybrid에는 없다)
    "max_steps": None,
    "max_wall_hours": None,
    "seed": 17,
    "torch_threads": 1,
    "checkpoint_every": None,
    "artifacts_dir": "artifacts/runs",
    "stop_after": None,
    "resume": None,
    "sampler": dict(DEFAULT_SAMPLER),
    # 실제 backbone(G0b)용 — fixture는 기본값 그대로 둔다.
    "device": "cpu",
    "readout_rank": None,  # null = fixture는 설정 파일의 rank, 실제 backbone은 DEFAULT_REAL_READOUT_RANK
    "readout_dtype": "float32",  # readout(U·V·b)의 dtype — BF16 backbone에도 fp32 readout이 기본 (docs/03 §5 FP32 loss)
    "model_root": None,  # 가중치 보관 디렉터리 (null = artifacts/models)
    "window_capacity": None,  # 윈도우 KV 버퍼 용량(토큰; null = DEFAULT_WINDOW_CAPACITY)
    "lora": None,  # trainable: lora_and_readout일 때 {r, alpha, dropout, targets} (없는 키는 DEFAULT_LORA)
}


def _need(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _dataset_manifests(single: Any, many: Any) -> list[dict[str, Any]]:
    """`dataset_manifest`(경로 하나)와 `dataset_manifests`(경로 또는 `{path, domain, material}` 목록)를 하나의 정규화된
    목록으로. 둘 다 주면 이어 붙인다(단일 것이 먼저). 비어 있으면 오류."""
    entries: list[Any] = []
    if single is not None:
        _need(isinstance(single, str) and bool(single), f"dataset_manifest: 데이터 manifest 경로(문자열)여야 한다 (받은 값: {single!r})")
        entries.append(single)
    if many is not None:
        _need(isinstance(many, list), f"dataset_manifests: 목록이어야 한다 (받은 값: {many!r})")
        entries.extend(many)
    _need(bool(entries), "dataset_manifests: 데이터 manifest 경로가 하나 이상 필요하다 (dataset_manifest 또는 dataset_manifests)")
    out: list[dict[str, Any]] = []
    for position, entry in enumerate(entries):
        if isinstance(entry, str):
            entry = {"path": entry}
        _need(isinstance(entry, dict), f"dataset_manifests[{position}]: 경로 또는 {{path, domain, material}}여야 한다 (받은 값: {entry!r})")
        unknown = [key for key in entry if key not in ("path", "domain", "material", "files")]
        _need(not unknown, f"dataset_manifests[{position}]: 알 수 없는 키 {unknown} (허용: ['path', 'domain', 'material', 'files'])")
        files = entry.get("files")
        _need(files is None or (isinstance(files, list) and files and all(isinstance(f, str) for f in files)), f"dataset_manifests[{position}].files: manifest 파일 키의 fnmatch 패턴 목록이거나 null이어야 한다")
        path = entry.get("path")
        _need(isinstance(path, str) and bool(path), f"dataset_manifests[{position}].path: manifest 경로(문자열)가 필요하다")
        domain = entry.get("domain")
        _need(domain is None or domain in DOMAINS, f"dataset_manifests[{position}].domain: {list(DOMAINS)} 중 하나이거나 null이어야 한다 (받은 값: {domain!r})")
        material = entry.get("material")
        _need(material is None or material in MATERIALS, f"dataset_manifests[{position}].material: {list(MATERIALS)} 중 하나이거나 null이어야 한다 (받은 값: {material!r})")
        out.append({"path": path, "domain": domain, "material": material, "files": None if files is None else list(files)})
    return out


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def resolve_config(config: dict) -> dict:
    """기본값을 채우고 검사한 설정. 알 수 없는 키, CPU 경로가 구현하지 않은 값은 `ValueError`."""
    if not isinstance(config, dict):
        raise ValueError(f"config: dict여야 한다 (받은 값: {type(config).__name__})")
    unknown = [key for key in config if key not in DEFAULTS]
    _need(not unknown, f"config: 알 수 없는 키 {unknown} (허용: {list(DEFAULTS)})")
    out = copy.deepcopy(DEFAULTS)
    out.update(copy.deepcopy(config))
    sampler = dict(DEFAULT_SAMPLER)
    sampler.update(config.get("sampler") or {})
    unknown = [key for key in sampler if key not in DEFAULT_SAMPLER]
    _need(not unknown, f"sampler: 알 수 없는 키 {unknown} (허용: {list(DEFAULT_SAMPLER)})")
    out["sampler"] = sampler

    out["dataset_manifests"] = _dataset_manifests(out.pop("dataset_manifest"), out["dataset_manifests"])
    out["dataset_manifest"] = None  # 정규화한 목록이 run의 정체다 — `dataset_manifest: x`와 `dataset_manifests: [x]`는 같은 run
    _need(_is_int(out["max_steps"]) and out["max_steps"] >= 1, f"max_steps: 1 이상의 정수여야 한다 (받은 값: {out['max_steps']!r})")
    _need(
        out["model_id"] in MODEL_IDS,
        f"model_id: {list(MODEL_IDS)}만 만들 수 있다 — 소형 계산 fixture({FIXTURE_MODEL_ID!r})와 candidates.yaml의 실제 backbone"
        f"(adapter: robo_jev.model.backbone_qwen). 다른 id를 조용히 fixture로 바꾸지 않는다 (받은 값: {out['model_id']!r})",
    )
    real = out["model_id"] != FIXTURE_MODEL_ID
    _need(out["execution_backend"] in EXECUTION_BACKENDS, f"execution_backend: {list(EXECUTION_BACKENDS)}만 구현했다 — shared_hybrid(P1)는 state_first에 아직 없다 (받은 값: {out['execution_backend']!r})")
    _need(out["readout"] in READOUTS, f"readout: {list(READOUTS)} 중 하나여야 한다 (받은 값: {out['readout']!r})")
    _need(out["dtype"] in DTYPES, f"dtype: {list(DTYPES)} 중 하나여야 한다 (받은 값: {out['dtype']!r})")
    _need(real or out["dtype"] != "bfloat16", "dtype: bfloat16은 실제 backbone에서만 — fixture의 CPU 검증은 float32/float64다 (BF16 허용 오차는 G0b가 실측)")
    _need(out["trainable"] in TRAINABLE, f"trainable: {list(TRAINABLE)} 중 하나여야 한다 (받은 값: {out['trainable']!r})")
    _need(out["device"] in DEVICES, f"device: {list(DEVICES)} 중 하나여야 한다 (받은 값: {out['device']!r})")
    _need(out["readout_dtype"] in READOUT_DTYPES, f"readout_dtype: {list(READOUT_DTYPES)} 중 하나여야 한다 (받은 값: {out['readout_dtype']!r})")
    _need(out["readout_rank"] is None or (_is_int(out["readout_rank"]) and out["readout_rank"] >= 1), f"readout_rank: 1 이상의 정수이거나 null이어야 한다 (받은 값: {out['readout_rank']!r})")
    _need(out["window_capacity"] is None or (_is_int(out["window_capacity"]) and out["window_capacity"] >= 1), f"window_capacity: 1 이상의 정수이거나 null이어야 한다 (받은 값: {out['window_capacity']!r})")
    _need(out["model_root"] is None or isinstance(out["model_root"], str), "model_root: 경로(문자열)이거나 null이어야 한다")
    if out["trainable"] == "lora_and_readout":
        _need(real, "trainable: lora_and_readout은 실제 backbone에서만 (fixture에는 LoRA를 붙이지 않는다)")
        lora = dict(DEFAULT_LORA)
        lora.update(out["lora"] or {})
        unknown = [key for key in lora if key not in DEFAULT_LORA]
        _need(not unknown, f"lora: 알 수 없는 키 {unknown} (허용: {list(DEFAULT_LORA)})")
        _need(_is_int(lora["r"]) and lora["r"] >= 1, f"lora.r: 1 이상의 정수여야 한다 (받은 값: {lora['r']!r})")
        _need(_is_number(lora["alpha"]) and lora["alpha"] > 0, f"lora.alpha: 양수여야 한다 (받은 값: {lora['alpha']!r})")
        _need(_is_number(lora["dropout"]) and 0.0 <= lora["dropout"] < 1.0, f"lora.dropout: [0, 1) 안이어야 한다 (받은 값: {lora['dropout']!r})")
        _need(isinstance(lora["targets"], list) and lora["targets"] and all(isinstance(t, str) for t in lora["targets"]), "lora.targets: module 이름 목록이어야 한다")
        out["lora"] = lora
    else:
        _need(out["lora"] is None, "lora: trainable이 lora_and_readout일 때만 준다")
    _need(out["optimizer"] in OPTIMIZERS, f"optimizer: {list(OPTIMIZERS)}만 구현했다 (받은 값: {out['optimizer']!r})")
    _need(isinstance(out["fp32_master_weights"], bool), f"fp32_master_weights: true/false여야 한다 (받은 값: {out['fp32_master_weights']!r})")
    _need(isinstance(out["activation_checkpointing"], bool), "activation_checkpointing: true/false여야 한다")
    _need(out["activation_checkpointing"] is False or out["model_id"] != "tiny_hybrid", "activation_checkpointing: CPU fixture 경로에는 없다 — 실제 backbone(qwen3_5)의 층 단위 checkpointing만 있다 (tiny_hybrid에서는 false여야 한다)")
    _need(out["world_size"] == 1, f"world_size: 이 학습기는 단일 프로세스다 (1이어야 한다, 받은 값: {out['world_size']!r})")
    _need(isinstance(out["freeze_vision_encoder"], bool), "freeze_vision_encoder: true/false여야 한다")
    layout = out["layout"]
    _need(isinstance(layout, dict) and set(layout) == {"single", "stream"}, f"layout: {{single: state_first, stream: stream_l1a}} 꼴이어야 한다 (받은 값: {layout!r})")
    _need(layout["single"] == "state_first" and layout["stream"] == "stream_l1a", f"layout: 단일 요청은 state_first, 스트림은 stream_l1a만 있다 (받은 값: {layout!r})")
    _need(isinstance(out["splits"], list) and out["splits"], "splits: 비어 있지 않은 목록이어야 한다")
    for key in ("backbone_lr", "readout_lr", "weight_decay"):
        _need(_is_number(out[key]) and out[key] >= 0, f"{key}: 0 이상의 수여야 한다 (받은 값: {out[key]!r})")
    _need(out["gradient_clip"] is None or (_is_number(out["gradient_clip"]) and out["gradient_clip"] > 0), f"gradient_clip: 양수이거나 null이어야 한다 (받은 값: {out['gradient_clip']!r})")
    _need(_is_number(out["warmup_ratio"]) and 0.0 <= out["warmup_ratio"] <= 1.0, f"warmup_ratio: [0, 1] 안이어야 한다 (받은 값: {out['warmup_ratio']!r})")
    for key in ("gradient_accumulation", "stream_window_ticks", "torch_threads", "nonrobot_tokens_per_unit"):
        _need(_is_int(out[key]) and out[key] >= 1, f"{key}: 1 이상의 정수여야 한다 (받은 값: {out[key]!r})")
    _need(out["microbatch_states_per_rank"] == 1, "microbatch_states_per_rank: rank의 accumulation 단위는 에피소드 하나 또는 토큰 예산(nonrobot_tokens_per_unit)까지 묶은 단일 요청 microbatch 하나다 — 1만 지원한다")
    _need(_is_number(out["robot_loss_share"]) and 0.0 <= out["robot_loss_share"] <= 1.0, f"robot_loss_share: [0, 1] 안이어야 한다 (받은 값: {out['robot_loss_share']!r})")
    for key in ("max_total_tokens", "max_state_tokens", "stream_max_ticks", "model_vocab_size", "model_seed", "checkpoint_every"):
        _need(out[key] is None or (_is_int(out[key]) and out[key] >= 1), f"{key}: 1 이상의 정수이거나 null이어야 한다 (받은 값: {out[key]!r})")
    _need(out["stream_chunk_seconds"] is None or (_is_number(out["stream_chunk_seconds"]) and out["stream_chunk_seconds"] > 0), f"stream_chunk_seconds: 양수이거나 null(구간 없음)이어야 한다 (받은 값: {out['stream_chunk_seconds']!r})")
    _need(out["max_wall_hours"] is None or (_is_number(out["max_wall_hours"]) and out["max_wall_hours"] > 0), "max_wall_hours: 양수이거나 null이어야 한다")
    _need(_is_int(out["seed"]), f"seed: 정수여야 한다 (받은 값: {out['seed']!r})")
    stop = out["stop_after"]
    if stop is not None:
        _need(isinstance(stop, dict) and _is_int(stop.get("step")) and stop["step"] >= 0, "stop_after: {step: n} 또는 {step: n, unit: u, chunk: c}여야 한다")
        _need(set(stop) in ({"step"}, {"step", "unit", "chunk"}), "stop_after: step 하나거나 step·unit·chunk 셋이어야 한다")
        if "unit" in stop:
            _need(_is_int(stop["unit"]) and stop["unit"] >= 0 and _is_int(stop["chunk"]) and stop["chunk"] >= 0, "stop_after: unit·chunk는 0 이상의 정수여야 한다")
    _need(out["resume"] is None or isinstance(out["resume"], str), "resume: checkpoint 경로(문자열)이거나 null이어야 한다")
    if out["checkpoint_every"] is None:
        out["checkpoint_every"] = out["max_steps"]
    if out["run_id"] is None:
        out["run_id"] = f"{out['run_name']}-{time.strftime('%Y%m%d-%H%M%S')}"
    _need(isinstance(out["run_id"], str) and out["run_id"], "run_id: 비어 있지 않은 문자열이어야 한다")
    _need(_is_int(sampler["steady_min_held_ticks"]) and sampler["steady_min_held_ticks"] >= 1, "sampler.steady_min_held_ticks: 1 이상의 정수여야 한다")
    _need(sampler["permute_candidates_seed"] is None or _is_int(sampler["permute_candidates_seed"]), "sampler.permute_candidates_seed: 정수이거나 null이어야 한다")
    weights = sampler["tick_weights"]
    _need(isinstance(weights, dict) and set(weights) == set(TICK_CLASSES), f"sampler.tick_weights: {list(TICK_CLASSES)} 네 종류의 가중치가 필요하다 (받은 값: {weights!r})")
    return out


# --------------------------------------------------------------------------
# 모델·optimizer·일정
# --------------------------------------------------------------------------


def build_tokenizer(name: str) -> Any:
    if name == "whitespace":
        return WhitespaceTokenizer()
    return load_tokenizer(name)


def build_model(config: dict) -> Judge:
    """설정의 backbone(fixture 또는 실제 Qwen3.5 adapter) + readout. FSDP 등 wrapper는 이 함수의 결과에 씌운다.

    실제 backbone은 BF16으로 싣고(가중치는 `artifacts/models/<id>`, manifest 대조), readout은 `readout_dtype`(기본 fp32)이다.
    `trainable: lora_and_readout`이면 peft LoRA를 text 모델의 projection에 붙인다(LoRA·readout만 학습, 나머지는 고정).
    """
    if config["model_id"] == FIXTURE_MODEL_ID:
        judge = Judge.from_config(
            config["model_config"], seed=config["model_seed"], readout=READOUTS[config["readout"]],
            vocab_size=config["model_vocab_size"],
        )  # fmt: skip
        judge = judge.to(DTYPES[config["dtype"]])
        if config["readout_rank"] is not None and config["readout_rank"] != judge.rank:
            judge = Judge(judge.backbone, rank=config["readout_rank"], readout=judge.readout, seed=(judge.backbone.config.seed if config["model_seed"] is None else config["model_seed"]) + 1000)
        if config["trainable"] == "readout_only":
            judge.backbone.requires_grad_(False)  # T0: readout만 (docs/03 §5)
        return judge
    kv_mode = "dynamic" if config["trainable"] != "readout_only" else "static"
    backbone = QwenBackbone.load(
        config["model_id"], root=config["model_root"], dtype=DTYPES[config["dtype"]], device=config["device"], kv_mode=kv_mode,
        window_capacity=DEFAULT_WINDOW_CAPACITY if config["window_capacity"] is None else int(config["window_capacity"]),
    )  # fmt: skip
    if config["trainable"] == "text_backbone_and_readout":
        backbone.model.requires_grad_(True)
    elif config["trainable"] == "lora_and_readout":
        attach_lora(backbone, config["lora"])
    backbone.activation_checkpointing = bool(config["activation_checkpointing"])  # 층 단위 (gradient가 켜진 forward에서만 작동)
    rank = DEFAULT_REAL_READOUT_RANK if config["readout_rank"] is None else int(config["readout_rank"])
    seed = 1000 + (0 if config["model_seed"] is None else int(config["model_seed"]))
    return Judge(backbone, rank=rank, readout=READOUTS[config["readout"]], seed=seed, readout_dtype=READOUT_DTYPES[config["readout_dtype"]])


def attach_lora(backbone: QwenBackbone, lora: dict[str, Any]) -> list[str]:
    """peft LoRA를 text 모델의 `targets` projection에 붙인다 (LoRA 파라미터는 fp32, 학습 대상; 기본 가중치는 고정)."""
    from peft import LoraConfig
    from peft.mapping import inject_adapter_in_model

    config = LoraConfig(r=int(lora["r"]), lora_alpha=float(lora["alpha"]), lora_dropout=float(lora["dropout"]), target_modules=list(lora["targets"]), bias="none")
    inject_adapter_in_model(config, backbone.text)
    names: list[str] = []
    for name, parameter in backbone.text.named_parameters():
        if "lora_" in name:
            parameter.data = parameter.data.to(torch.float32)
            parameter.requires_grad_(True)
            names.append(name)
        else:
            parameter.requires_grad_(False)
    if not names:
        raise ValueError(f"lora.targets: {lora['targets']}에 맞는 module이 없다")
    return names


def load_trainable_state(model: Judge, saved: dict[str, Tensor]) -> None:
    """:func:`trainable_state_dict` 가 저장한 것을 싣는다 — 저장된 키는 전부 있어야 하고, 빠진 키는 고정된 backbone 가중치뿐이어야 한다."""
    result = model.load_state_dict(saved, strict=False)
    if result.unexpected_keys:
        raise ValueError(f"checkpoint: 모델에 없는 파라미터가 저장되어 있다: {sorted(result.unexpected_keys)[:5]}")
    trainable = {
        name for name, parameter in model.named_parameters(remove_duplicate=False)
        if parameter.requires_grad or not name.startswith("backbone.")
    }  # 묶인 가중치(lm_head↔embedding)는 이름이 둘이라 중복을 지우지 않고 본다
    missing = [name for name in result.missing_keys if name in trainable]
    if missing:
        raise ValueError(f"checkpoint: 학습 대상 파라미터가 저장되어 있지 않다: {missing[:5]}")


def load_readout_checkpoint(model: Judge, path: str | Path, *, tokenizer_sha256: str | None, trust_checkpoint_tokenizer: bool = False) -> dict[str, Any]:
    """서빙·평가용: checkpoint의 readout(과 LoRA)을 `model`에 싣는다 — 먼저 배포 계약 digest(직렬화·계약 소스, 하네스 버전,
    **tokenizer 파일 해시**)를 지금 체크아웃과 대조해 다르면 거절한다(다른 조각 이름을 말한다).

    `tokenizer_sha256`은 지금 체크아웃의 tokenizer 파일 해시(`tokenizer_block(name)["sha256"]`)다. `None`은
    `trust_checkpoint_tokenizer=True`와 함께일 때만 허용되며(checkpoint가 적은 해시로 digest를 만들어 코드·하네스 버전만 대조 —
    tokenizer가 없는 검사용), 그 밖에는 ValueError. 돌려주는 것은 checkpoint의 manifest.
    """
    state = load_checkpoint(path)
    manifest = state.get("manifest") if isinstance(state.get("manifest"), dict) else {}
    if tokenizer_sha256 is None:
        if not trust_checkpoint_tokenizer:
            raise ValueError(f"{path}: tokenizer_sha256이 없다 — 지금 체크아웃의 tokenizer 해시를 주거나 trust_checkpoint_tokenizer=True를 명시한다")
        tokenizer_sha256 = (manifest.get("contract") or {}).get("tokenizer_sha256") or (manifest.get("tokenizer") or {}).get("sha256") or "whitespace"
    current = contract_digest(tokenizer_sha256)
    differences = contract_differences(manifest.get("contract"), current)
    if differences:
        raise ValueError(f"{path}: 배포 계약 digest가 지금 체크아웃과 다르다 (다른 조각: {differences}) — 이 checkpoint를 싣지 않는다")
    saved_rank = (manifest.get("model") or {}).get("rank")
    if saved_rank is not None and int(saved_rank) != model.rank:
        raise ValueError(f"{path}: checkpoint의 readout rank {saved_rank}와 모델의 {model.rank}가 다르다")
    load_trainable_state(model, state["model"])
    return manifest


def trainable_state_dict(model: Judge) -> dict[str, Tensor]:
    """저장할 가중치: fixture는 전부, 실제 backbone은 readout(U·V·b)과 학습 대상(LoRA) 파라미터만 — 고정된 backbone은 저장하지 않는다."""
    if isinstance(model.backbone, TinyHybrid):
        return model.state_dict()
    trainable = {name for name, parameter in model.named_parameters() if parameter.requires_grad or not name.startswith("backbone.")}
    return {name: tensor for name, tensor in model.state_dict().items() if name in trainable}


def parameter_groups(model: Judge, config: dict) -> list[dict]:
    """backbone/readout에 각각의 lr, 2차원 이상의 tensor에만 weight decay (bias·norm·게이트는 0)."""
    groups: list[dict] = []

    def add(name: str, params: list[Tensor], lr: float) -> None:
        decay = [p for p in params if p.dim() >= 2]
        rest = [p for p in params if p.dim() < 2]
        if decay:
            groups.append({"name": f"{name}/decay", "params": decay, "lr": lr, "weight_decay": config["weight_decay"]})
        if rest:
            groups.append({"name": f"{name}/no_decay", "params": rest, "lr": lr, "weight_decay": 0.0})

    readout = [p for n, p in model.named_parameters() if not n.startswith("backbone.")]
    backbone = [p for n, p in model.named_parameters() if n.startswith("backbone.") and p.requires_grad]
    if config["trainable"] in ("text_backbone_and_readout", "lora_and_readout"):
        add("backbone", backbone, config["backbone_lr"])
    add("readout", readout, config["readout_lr"])
    return groups


#: fp32 master 사본이 optimizer의 `state_dict`에 들어가는 자리 (= checkpoint의 `optimizer` 블록 안).
MASTER_WEIGHTS_KEY = "fp32_master_weights"


def fp32_master_weights(model: Judge) -> dict[str, Tensor]:
    """학습 대상 가운데 **fp32가 아닌** 파라미터의 fp32 master 사본 ``{이름: 사본}``.

    readout(U·V·b)은 이미 fp32이고(`readout_dtype` 기본값) LoRA 파라미터도 :func:`attach_lora` 가 fp32로 올려
    두므로 둘 다 사본이 생기지 않는다 — 사본이 생기는 것은 BF16 backbone을 통째로 학습하는 T1뿐이다(중복 금지).
    """
    return {
        name: parameter.detach().clone().to(torch.float32)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and parameter.dtype != torch.float32
    }


class MasterWeightAdamW(torch.optim.AdamW):
    """bf16 학습 대상의 **fp32 master 사본**을 들고 fp32로 갱신한 뒤 bf16으로 되쓰는 AdamW (고전적 혼합 정밀도).

    `param_groups`에 든 것은 master 사본이고, 모델의 bf16 파라미터는 ``pairs``(이름, 모델 파라미터, master)로
    짝지어 둔다. 한 step은 셋이다: (1) 모델 파라미터의 gradient를 master의 fp32 gradient 버퍼에 옮기고,
    (2) fp32로 AdamW 한 step을 밟고, (3) master를 bf16 파라미터에 되쓴다. **반올림은 (3)에서 한 번만** 일어나고
    누적은 master가 하므로, ``lr``이 bf16 눈금의 절반보다 작아도 갱신이 사라지지 않는다 (P1 §C2·D의 병리).

    메모리 (2B, 학습 대상 1.88B). bf16 파라미터 3.76 GB + bf16 gradient 3.76 GB는 그대로이고, master 7.52 GB와
    fp32 Adam 상태 15.04 GB가 더해진다 = backward 동안 30.08 GB(28.0 GiB), bf16 직접 갱신의 15.04 GB(14.0 GiB)보다
    **+14.0 GiB**. fp32 gradient 버퍼(7.52 GB)는 :meth:`step` 안에서만 들고 step 끝에 놓는다 — 그때는 활성값이
    이미 풀려 있어 backward의 peak를 올리지 않는다. (`torch`는 파라미터와 다른 dtype의 ``.grad`` 대입을 거절하므로
    bf16 gradient를 그대로 넘길 수는 없다.)
    """

    def __init__(self, groups: list[dict], *, pairs: list[tuple[str, Tensor, Tensor]], **kwargs: Any) -> None:
        super().__init__(groups, **kwargs)
        self._pairs: list[tuple[str, Tensor, Tensor]] = list(pairs)

    @property
    def master_pairs(self) -> list[tuple[str, Tensor, Tensor]]:
        """(이름, 모델 파라미터, fp32 master) 짝 — 검사·측정용."""
        return list(self._pairs)

    def step(self, closure: Any = None) -> Any:  # type: ignore[override]
        for _, parameter, master in self._pairs:
            master.grad = None if parameter.grad is None else parameter.grad.detach().to(torch.float32)
        loss = super().step(closure)
        for _, parameter, master in self._pairs:
            parameter.data.copy_(master.data)  # 반올림은 여기 한 번 — 누적은 master가 한다
            master.grad = None  # fp32 gradient 버퍼는 step 밖에서 들고 있지 않는다
        return loss

    def zero_grad(self, set_to_none: bool = True) -> None:  # type: ignore[override]
        """master의 gradient뿐 아니라 **모델 파라미터의** gradient도 지운다 (모델 파라미터는 param_groups에 없다)."""
        super().zero_grad(set_to_none=set_to_none)
        for _, parameter, _ in self._pairs:
            if parameter.grad is None:
                continue
            if set_to_none:
                parameter.grad = None
            else:
                parameter.grad.zero_()

    def state_dict(self) -> dict[str, Any]:
        """AdamW의 상태 + master 사본(:data:`MASTER_WEIGHTS_KEY`) — master는 bf16 파라미터로 복원할 수 없는 정밀도다."""
        state = super().state_dict()
        state[MASTER_WEIGHTS_KEY] = {name: master for name, _, master in self._pairs}
        return state

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:  # type: ignore[override]
        saved = state_dict.get(MASTER_WEIGHTS_KEY)
        super().load_state_dict({key: value for key, value in state_dict.items() if key != MASTER_WEIGHTS_KEY})
        if not isinstance(saved, dict):
            raise ValueError(
                f"optimizer: fp32 master 사본({MASTER_WEIGHTS_KEY!r})이 checkpoint에 없다 — bf16으로 직접 갱신한 run은 "
                "fp32 master로 이어갈 수 없다 (master가 없으면 잃어버린 하위 비트를 되살릴 수 없다). 새 run으로 시작한다"
            )
        missing = [name for name, _, _ in self._pairs if name not in saved]
        if missing:
            raise ValueError(f"optimizer: fp32 master 사본이 없는 학습 대상이 있다: {missing[:5]}")
        for name, parameter, master in self._pairs:
            master.data.copy_(saved[name].to(device=master.device, dtype=master.dtype))
            parameter.data.copy_(master.data)  # bf16 사본은 master의 반올림이다 — 둘을 한 값에서 맞춘다


def build_optimizer(model: Judge, config: dict) -> torch.optim.AdamW:
    """설정의 optimizer. `fp32_master_weights`가 켜져 있고 학습 대상에 fp32가 아닌 파라미터가 있으면
    :class:`MasterWeightAdamW`, 아니면 평범한 ``torch.optim.AdamW``다 (T0·LoRA·fixture는 후자 — 사본이 없다)."""
    groups = parameter_groups(model, config)
    masters = fp32_master_weights(model) if config["fp32_master_weights"] else {}
    if not masters:
        return torch.optim.AdamW(groups, betas=(0.9, 0.999), eps=1e-8)
    by_id = {id(parameter): (name, masters[name]) for name, parameter in model.named_parameters() if name in masters}
    pairs: list[tuple[str, Tensor, Tensor]] = []
    for group in groups:
        swapped: list[Tensor] = []
        for parameter in group["params"]:
            found = by_id.get(id(parameter))
            if found is None:
                swapped.append(parameter)
                continue
            name, master = found
            swapped.append(master)
            pairs.append((name, parameter, master))
        group["params"] = swapped
    return MasterWeightAdamW(groups, pairs=pairs, betas=(0.9, 0.999), eps=1e-8)


def lr_factor(step: int, *, max_steps: int, warmup_ratio: float) -> float:
    """선형 warmup(``warmup_ratio × max_steps`` step) 뒤 cosine decay (docs/03 §5). `step` = 끝난 step 수."""
    warmup = int(round(warmup_ratio * max_steps))
    if step < warmup:
        return (step + 1) / warmup
    remaining = max(1, max_steps - warmup)
    progress = min(1.0, max(0.0, (step - warmup) / remaining))
    return 0.5 * (1.0 + math.cos(math.pi * progress))


def clip_gradients(parameters: list[Tensor], max_norm: float | None) -> float:
    """전체 gradient norm(clip 전)을 돌려주고 `max_norm`이 있으면 clip한다. FSDP는 자기 clip으로 바꾼다."""
    limit = float("inf") if max_norm is None else float(max_norm)
    return float(torch.nn.utils.clip_grad_norm_(parameters, limit))


# --------------------------------------------------------------------------
# truncated BPTT의 조각
# --------------------------------------------------------------------------

_PER_TOKEN_FIELDS = ("tokens", "kind", "state", "question", "candidate", "position", "tick", "segment")


def episode_chunks(record: dict, chunk_seconds: float | None) -> list[tuple[int, int]]:
    """에피소드를 첫 틱의 모의 시각 기준 `chunk_seconds` 구간으로 나눈 틱 범위 ``[(start, end), …]``."""
    ticks = record["ticks"]
    if chunk_seconds is None:
        return [(0, len(ticks))]
    if chunk_seconds <= 0:
        raise ValueError(f"stream_chunk_seconds: 양수이거나 None이어야 한다 (받은 값: {chunk_seconds})")
    chunk_ms = float(chunk_seconds) * 1000.0
    origin = int(ticks[0]["sim_ms"])
    chunks: list[tuple[int, int]] = []
    start, current = 0, 0
    for index, tick in enumerate(ticks):
        which = int((int(tick["sim_ms"]) - origin) // chunk_ms)
        if which != current:
            chunks.append((start, index))
            start, current = index, which
    chunks.append((start, len(ticks)))
    return chunks


def layout_prefix(layout: dict, end_tick: int) -> dict:
    """직렬화된 스트림 layout의 앞 `end_tick`틱(+prefix)만 남긴 view. 토큰·position은 그대로다."""
    ticks = layout["ticks"]
    if not 1 <= end_tick <= len(ticks):
        raise ValueError(f"end_tick: 1 이상 {len(ticks)} 이하여야 한다 (받은 값: {end_tick})")
    if end_tick == len(ticks):
        return layout
    cut = int(ticks[end_tick - 1]["end"])
    view = {key: value for key, value in layout.items() if key != "text"}
    for name in _PER_TOKEN_FIELDS:
        if name in layout:
            view[name] = layout[name][:cut]
    view["ticks"] = ticks[:end_tick]
    view["tick_boundaries"] = layout["tick_boundaries"][:end_tick]
    view["segments"] = [segment for segment in layout["segments"] if segment["end"] <= cut]
    view["instruction_positions"] = [p for p in layout["instruction_positions"] if p < cut]
    return view


def detach_stream_state(state: Any, *, requires_grad: bool = False) -> Any:
    """구간 경계에서 넘기는 공통 상태 — 모든 tensor를 detach한 새 상태 (값은 같고 gradient만 끊긴다).

    `requires_grad=True`면 detach한 tensor를 leaf로 만들어 다음 구간의 gradient가 경계에 얼마나
    닿는지 관찰할 수 있다(검사용). 상태 클래스(fixture의 :class:`StreamState`, 실제 backbone의
    :class:`robo_jev.model.backbone_qwen.QwenStreamState`)의 ``detach``에 맡긴다.
    """
    if state.is_branch:
        raise ValueError("branch 상태는 넘기지 않는다 — 다음 구간은 분기 이전 공통 상태에서 이어간다")
    return state.detach(requires_grad=requires_grad)


@dataclass
class EpisodePlan:
    """에피소드의 구간과 틱별 종류·가중치·유효 라벨, 손실의 정규화 분모 ``Σ_t w_t`` (유효 틱만)."""

    chunks: list[tuple[int, int]]
    classes: list[str]
    weights: list[float]
    valid: list[bool]
    normaliser: float


def plan_episode(
    record: dict,
    *,
    chunk_seconds: float | None,
    tick_weights: dict[str, float],
    steady_min_held_ticks: int = 3,
) -> EpisodePlan:
    classes = [tick_class(record, i, steady_min_held_ticks=steady_min_held_ticks) for i in range(len(record["ticks"]))]
    weights = [float(tick_weights[name]) for name in classes]
    valid = valid_label_ticks(record)
    normaliser = sum(w for w, ok in zip(weights, valid) if ok)
    return EpisodePlan(
        chunks=episode_chunks(record, chunk_seconds), classes=classes, weights=weights, valid=valid,
        normaliser=normaliser,
    )  # fmt: skip


@dataclass
class ChunkResult:
    """한 forward·backward 조각의 결과: 그래프에 이어진 손실 기여, 그 값, 스트림이면 구간 끝의 공통 상태."""

    loss: Tensor | None
    value: float
    state: StreamState | None
    outputs: dict | None
    stats: dict = field(default_factory=dict)


def _add_type_losses(by_type: dict[str, list[float]], entries: dict, question_types: dict[str, str]) -> None:
    for qid, entry in entries.items():
        kind = question_types.get(qid, "unknown")
        slot = by_type.setdefault(kind, [0.0, 0])
        slot[0] += float(entry["loss"].detach())
        slot[1] += 1


def _count_labels(states: list[dict]) -> tuple[int, int]:
    """상태(틱·단일 요청) 목록의 라벨 수와 그 가운데 낮은 신뢰도(`label_confidence: low`) 라벨 수.

    낮은 신뢰도 라벨은 퇴화 틱(목표 후보 없음·재시도 차단·실행기 사정)의 표지이며 생성기가 `weight`로
    내려 준다(I4 완화) — 손실이 그 weight를 쓰므로 step 지표에 따로 세어 그 비중을 읽을 수 있게 한다.
    """
    total = low = 0
    for state in states:
        for label in state.get("labels", []):
            total += 1
            if label.get("label_confidence") == "low":
                low += 1
    return total, low


def run_stream_chunk(
    judge: Judge,
    item: Item,
    chunk: tuple[int, int],
    *,
    carried: StreamState | None,
    plan: EpisodePlan,
    scale: float = 1.0,
) -> ChunkResult:
    """에피소드 구간 ``[start, end)``의 forward와 손실 기여 ``scale × Σ_t w_t L_t`` (유효 라벨이 있는 틱).

    ``scale``은 호출자가 정한다 — 학습 loop는 ``로봇 비중 / step의 로봇 틱 가중치 합``, 에피소드 하나의
    정규화된 손실은 ``1 / plan.normaliser``. `carried`는 앞 구간이 넘긴(detach된) 공통 상태다(첫 구간은
    `None` — prefix부터 읽는다). 돌려주는 ``state``는 이 구간 끝의 공통 상태로 그래프가 붙어 있다 —
    다음 구간에 넘기기 전에 호출자가 :func:`detach_stream_state` 한다.
    """
    start, end = chunk
    layout = item.layout
    outputs = judge({"layout": "stream_l1a", "stream": layout_prefix(layout, end), "state": carried, "start_tick": start})
    record = item.record
    total: Tensor | None = None
    loss_sum = 0.0  # Σ w_t L_t (scale 전)
    weight_sum = 0.0  # Σ w_t (유효 틱)
    value_by_class: dict[str, float] = {}
    weight_by_class: dict[str, float] = {}
    by_type: dict[str, list[float]] = {}
    valid_ticks = 0
    for offset, index in enumerate(range(start, end)):
        if not plan.valid[index] or plan.weights[index] <= 0:
            continue
        one = {"logits": [outputs["logits"][offset]], "candidates": [outputs["candidates"][offset]]}
        labels = {"labels": [record["ticks"][index].get("labels", [])]}
        tick_loss = judgment_loss(one, labels)
        weight = plan.weights[index]
        term = tick_loss * (weight * scale)
        total = term if total is None else total + term
        valid_ticks += 1
        loss_sum += weight * float(tick_loss.detach())
        weight_sum += weight
        name = plan.classes[index]
        value_by_class[name] = value_by_class.get(name, 0.0) + float(term.detach())
        weight_by_class[name] = weight_by_class.get(name, 0.0) + weight * scale
        _add_type_losses(by_type, question_losses(one, labels)[0], item.question_types)
    tokens = (int(layout["prefix_end"]) if start == 0 else 0) + sum(
        int(t["end"]) - int(t["start"]) for t in layout["ticks"][start:end]
    )
    labels_total, labels_low = _count_labels(record["ticks"][start:end])
    return ChunkResult(
        loss=total,
        value=0.0 if total is None else float(total.detach()),
        state=outputs["state"],
        outputs=outputs,
        stats={
            "tokens": tokens, "ticks": end - start, "valid_ticks": valid_ticks, "loss_sum": loss_sum,
            "weight_sum": weight_sum, "loss_by_class": value_by_class, "weight_by_class": weight_by_class,
            "loss_by_type": by_type, "labels_total": labels_total, "labels_low_confidence": labels_low,
        },  # fmt: skip
    )


def run_single_unit(judge: Judge, items: list[Item], *, scale: float = 1.0) -> ChunkResult:
    """단일 요청 microbatch(한 forward)의 손실 기여 ``scale × Σ_s L_s`` (유효 라벨이 있는 상태; 상태마다 기록)."""
    outputs = judge({"layout": "state_first", "states": [item.layout for item in items]})
    total: Tensor | None = None
    per_item: list[dict[str, Any]] = []
    by_type: dict[str, list[float]] = {}
    valid_states = 0
    for position, item in enumerate(items):
        if not valid_single(item.record):
            per_item.append({"index": item.index, "valid": False, "loss": 0.0, "contribution": 0.0})
            continue
        one = {"logits": [outputs["logits"][position]], "candidates": [outputs["candidates"][position]]}
        labels = {"labels": [item.record.get("labels", [])]}
        state_loss = judgment_loss(one, labels)
        term = state_loss * scale
        total = term if total is None else total + term
        valid_states += 1
        per_item.append(
            {"index": item.index, "valid": True, "loss": float(state_loss.detach()), "contribution": float(term.detach())}
        )
        _add_type_losses(by_type, question_losses(one, labels)[0], item.question_types)
    labels_total, labels_low = _count_labels([item.record for item in items])
    return ChunkResult(
        loss=total,
        value=0.0 if total is None else float(total.detach()),
        state=None,
        outputs=outputs,
        stats={
            "tokens": sum(item.tokens for item in items), "states": len(items), "valid_states": valid_states,
            "per_item": per_item, "loss_by_type": by_type, "labels_total": labels_total,
            "labels_low_confidence": labels_low,
        },  # fmt: skip
    )


# --------------------------------------------------------------------------
# manifest
# --------------------------------------------------------------------------


def git_revision() -> dict[str, Any] | None:
    root = Path(__file__).resolve().parents[2]
    try:
        sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=True, timeout=10).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=root, capture_output=True, text=True, check=True, timeout=10).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None
    return {"sha": sha, "dirty": bool(dirty)}


def tokenizer_block(name: str) -> dict[str, Any]:
    """manifest의 tokenizer 블록 — 설정의 이름과 실제 파일의 정체(:func:`robo_jev.model.tokenizer.describe_tokenizer`)."""
    if name == "whitespace":
        return {"name": "whitespace", "kind": "whitespace"}
    return {"name": name, **describe_tokenizer(name)}


def model_block(config: dict, model: Judge) -> dict[str, Any]:
    """manifest의 model 블록 — 요청한 id가 아니라 **실제로 만든 것**: 종류·클래스·설정 파일과 그 sha256·이름·어휘·seed·
    readout·rank·파라미터 수(전체·학습 대상)·dtype·장치 (리뷰 11 S2). 실제 backbone은 가중치 manifest의 revision·지문과
    학습 범위(`trainable`, LoRA 설정)도 정체다."""
    parameters = list(model.parameters())
    backbone = type(model.backbone).__name__
    real = isinstance(model.backbone, QwenBackbone)
    config_path = None if real else Path(config["model_config"])
    manifest = model.backbone.manifest if real else {}
    return {
        "kind": _BACKBONE_KINDS.get(backbone, backbone),
        "id": config["model_id"],
        "class": backbone,
        "config": None if config_path is None else str(config_path),
        "config_sha256": None if config_path is None else sha256_of(config_path),
        "name": model.backbone.config.name,
        "vocab_size": model.backbone.config.vocab_size,
        "seed": model.backbone.config.seed if config["model_seed"] is None else config["model_seed"],
        "readout": model.readout,
        "rank": model.rank,
        "readout_dtype": str(model.readout_dtype).removeprefix("torch."),
        "parameters": sum(p.numel() for p in parameters),
        "trainable_parameters": sum(p.numel() for p in parameters if p.requires_grad),
        "trainable": config["trainable"],
        "lora": copy.deepcopy(config["lora"]),
        "dtype": str(next(model.backbone.parameters()).dtype).removeprefix("torch."),
        "device": str(parameters[0].device),
        "revision": manifest.get("revision"),
        "digest": manifest.get("digest"),
        "revision_manifest": config["model_revision_manifest"],
    }


def manifest_identity(manifest: dict) -> dict[str, Any]:
    """manifest의 **identity 블록** — 재개 때 같아야 하는 내용(모듈 설명 "run의 정체"). 경로·git·torch·시각은 뺀다."""
    return {
        "datasets": [
            {key: copy.deepcopy(entry[key]) for key in ("sha256", "files", "domain", "material", "records")}
            for entry in manifest["dataset_manifests"]
        ],
        "splits": list(manifest["splits"]),
        "serializer_version": manifest["serializer_version"],
        "question_set": copy.deepcopy(manifest["question_set"]),
        "layouts": dict(manifest["layouts"]),
        "tokenizer": {key: manifest["tokenizer"][key] for key in _TOKENIZER_IDENTITY if key in manifest["tokenizer"]},
        "model": {key: manifest["model"].get(key) for key in _MODEL_IDENTITY},
        "contract_sha256": manifest.get("contract_sha256"),
    }


def _flatten(value: Any, prefix: str = "") -> dict[str, Any]:
    if isinstance(value, dict) and value:
        out: dict[str, Any] = {}
        for key, inner in value.items():
            out.update(_flatten(inner, f"{prefix}.{key}" if prefix else str(key)))
        return out
    if isinstance(value, list) and value:
        out = {}
        for position, inner in enumerate(value):
            out.update(_flatten(inner, f"{prefix}[{position}]"))
        return out
    return {prefix: value}


def _short(value: Any) -> str:
    if isinstance(value, str) and len(value) > 16:
        return value[:12] + "…"
    return repr(value)


def identity_differences(saved: dict, current: dict) -> list[str]:
    """두 identity 블록에서 다른 키를 모두 ``키 (저장 값, 지금 값)`` 꼴로 — 지금 블록의 순서, 저장에만 있는 키는 뒤에."""
    before, after = _flatten(saved), _flatten(current)
    out: list[str] = []
    for key in [*after, *(key for key in before if key not in after)]:
        if key in before and key in after and before[key] == after[key]:
            continue
        was = _short(before[key]) if key in before else "없음"
        now = _short(after[key]) if key in after else "없음"
        out.append(f"{key} (저장 {was}, 지금 {now})")
    return out


def resume_config(config: dict) -> dict[str, Any]:
    """재개 때 문자 그대로 같아야 하는 설정 — :data:`RESUME_FREE_KEYS` 와 내용으로 대조하는 :data:`RESUME_PATH_KEYS` 를
    뺀 것. `dataset_manifests`는 경로를 빼고 태그(domain·material)만 남긴다."""
    out = {key: value for key, value in config.items() if key not in RESUME_FREE_KEYS and key not in RESUME_PATH_KEYS}
    out["dataset_manifests"] = [
        {"domain": entry.get("domain"), "material": entry.get("material"), "files": entry.get("files")} for entry in config.get("dataset_manifests") or []
    ]
    return out


def build_manifest(config: dict, items: list[Item], model: Judge) -> dict[str, Any]:
    """checkpoint에 함께 적는 것: 데이터 manifest 참조(manifest마다 경로·sha256·파일 해시·분야 태그·레코드 수), 토큰
    직렬화·질문 세트 버전, tokenizer의 정체, 실제로 만든 모델, git SHA — 그리고 이것들 가운데 재개 때 같아야 하는
    내용만 모은 ``identity`` 블록(:func:`manifest_identity`)."""
    datasets = []
    for entry in config["dataset_manifests"]:
        manifest_path = Path(entry["path"])
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        own = [item for item in items if item.manifest == str(manifest_path)]
        datasets.append(
            {
                "path": str(manifest_path),
                "sha256": sha256_of(manifest_path),
                "builder_version": manifest.get("builder_version") or manifest.get("generator"),
                "files": {name: (item or {}).get("sha256") for name, item in manifest_files(manifest, manifest_path).items()},
                "domain": entry.get("domain"),
                "material": entry.get("material"),
                "records": {"single": sum(i.kind == "single" for i in own), "stream": sum(i.kind == "stream" for i in own)},
            }
        )
    tokenizer = tokenizer_block(config["tokenizer"])
    contract = contract_digest(tokenizer.get("sha256") or "whitespace")
    manifest = {
        "dataset_manifests": datasets,
        "splits": list(config["splits"]),
        "serializer_version": TOKEN_SERIALIZER_VERSION,  # 토큰 직렬화의 버전 (레코드의 versions.serializer와 다른 것)
        "question_set": {"id": "qs-v0", "markers": {qid: spec["marker"] for qid, spec in QUESTION_SET_V0.items()}},
        "layouts": dict(config["layout"]),
        "tokenizer": tokenizer,
        "model": model_block(config, model),
        # 배포 계약 digest (analysis-nimble §3-4): 직렬화·질문 세트 소스, 하네스 버전, tokenizer 해시 — 적재·서빙이 대조한다
        "contract_sha256": contract["sha256"],
        "contract": contract,
        "git": git_revision(),
        "torch": str(torch.__version__),  # TorchVersion 객체가 아니라 문자열 — weights_only 로 읽힌다
    }
    manifest["identity"] = manifest_identity(manifest)
    return manifest


def check_contract(saved_manifest: Any, current_manifest: dict[str, Any], *, where: str) -> None:
    """저장된 manifest의 계약 digest가 지금 체크아웃과 다르면 거절한다 (어느 조각이 다른지 이름으로)."""
    saved = saved_manifest.get("contract") if isinstance(saved_manifest, dict) else None
    differences = contract_differences(saved, current_manifest["contract"])
    if differences:
        raise ValueError(
            f"{where}: 배포 계약 digest가 지금 체크아웃과 다르다 (다른 조각: {differences}; 저장 "
            f"{str((saved or {}).get('sha256'))[:12]}…, 지금 {current_manifest['contract_sha256'][:12]}…) — 직렬화·질문 세트·하네스 버전·"
            "tokenizer가 같은 체크아웃에서만 이 checkpoint를 싣는다"
        )


# --------------------------------------------------------------------------
# Trainer
# --------------------------------------------------------------------------


def _new_accumulators() -> dict[str, Any]:
    return {
        "loss": 0.0,  # step 손실 값 = Σ 기여
        "loss_sum": {domain: 0.0 for domain in DOMAINS},  # Σ w·L (정규화 전) → 분야 평균 = loss_sum / 분모
        "weight": {domain: 0.0 for domain in DOMAINS},  # 적용된 계수 질량 Σ w × scale (= 유효 loss 비중)
        "weight_by_material": {material: 0.0 for material in MATERIALS},
        "weight_by_class": {name: 0.0 for name in TICK_CLASSES},
        "value_by_domain": {domain: 0.0 for domain in DOMAINS},  # 기여 값
        "value_by_material": {material: 0.0 for material in MATERIALS},
        "value_by_class": {name: 0.0 for name in TICK_CLASSES},
        "loss_by_type": {},
        "tokens": {domain: 0 for domain in DOMAINS},
        "items": {"single": 0, "stream": 0},
        "valid_states": {"single": 0, "stream": 0},  # 유효 라벨이 있는 단일 요청 상태 / 틱
        "labels_total": 0,  # step의 상태들에 실린 라벨 수
        "labels_low_confidence": 0,  # 그 가운데 낮은 신뢰도(퇴화 틱, weight로 내린) 라벨 수
        "chunks": 0,
        "seconds": 0.0,
    }


class Trainer:
    """설정 하나의 학습 상태 (모듈 설명 참조).

    * :meth:`accumulate` — 현재 step의 accumulation 단위들(재개했으면 남은 것)을 forward·backward한다.
      끝나면 `True`, 중단 지점(`stop_after`·`max_wall_hours`)이면 진행 위치를 남기고 `False`.
    * :meth:`apply` — clip → optimizer step → schedule step, step 지표를 돌려준다.
    * :meth:`run` — `max_steps`까지 돌리고 checkpoint·metrics를 쓴다.

    ``torch_threads``는 run의 정체(재개의 비트 동일은 같은 스레드 수에서만)지만 `torch.set_num_threads`는
    프로세스 전역이다. 그래서 Trainer는 그 값을 **자기 계산 안에서만** 건다(:meth:`_threads` — 구성·accumulate·
    apply·run·load)이고 나올 때 바깥 값을 되돌린다: context manager 없이 만들어도, 생성이 실패해도 프로세스의
    스레드 수는 바뀌지 않는다. `with Trainer(...)`·:meth:`close` 는 그대로 쓸 수 있다(잡고 있는 전역 상태가 없다).
    """

    def __init__(self, config: dict, *, resume: str | Path | None = None) -> None:
        self.config = resolve_config(config)
        with self._threads():
            self._build(resume)

    @contextlib.contextmanager
    def _threads(self) -> Iterator[None]:
        """설정의 `torch_threads`를 이 블록 안에서만 건다 — 나올 때 바깥 값으로 되돌린다 (중첩해도 된다)."""
        before = torch.get_num_threads()
        wanted = int(self.config["torch_threads"])
        if wanted != before:
            torch.set_num_threads(wanted)
        try:
            yield
        finally:
            if torch.get_num_threads() != before:
                torch.set_num_threads(before)

    def _build(self, resume: str | Path | None) -> None:
        self._started = time.perf_counter()
        seed = int(self.config["seed"])
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed % (2**32))

        sampler_config = self.config["sampler"]
        self.tokenizer = build_tokenizer(self.config["tokenizer"])
        self.items = load_manifests(
            self.config["dataset_manifests"], tokenizer=self.tokenizer, splits=tuple(self.config["splits"]),
            layouts=self.config["layout"], window_ticks=self.config["stream_window_ticks"],
            max_state_tokens=self.config["max_state_tokens"], max_total_tokens=self.config["max_total_tokens"],
            stream_max_ticks=self.config["stream_max_ticks"], domain_tag=sampler_config["domain_tag"],
            material_tag=sampler_config["material_tag"], permute_seed=sampler_config["permute_candidates_seed"],
        )  # fmt: skip
        if not self.items:
            raise ValueError(f"dataset_manifests: split {self.config['splits']}에 레코드가 없다")
        self.model = build_model(self.config)
        vocab = self.model.backbone.config.vocab_size
        largest = max(max(item.layout["tokens"]) for item in self.items)
        if largest >= vocab:
            raise ValueError(f"model_vocab_size: 토큰 id {largest}가 어휘 {vocab}를 넘는다 — tokenizer에 맞는 어휘를 써야 한다")
        self.optimizer = build_optimizer(self.model, self.config)
        max_steps, warmup = int(self.config["max_steps"]), float(self.config["warmup_ratio"])
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, lambda step: lr_factor(step, max_steps=max_steps, warmup_ratio=warmup)
        )
        self.sampler = MixedSampler(
            self.items, material_shares=sampler_config["material_shares"], seed=seed,
            nonrobot_tokens_per_unit=self.config["nonrobot_tokens_per_unit"],
        )  # fmt: skip
        if len(self.sampler.domains) == 2 and int(self.config["gradient_accumulation"]) < 2:
            raise ValueError(
                "gradient_accumulation: 로봇·비로봇 레코드가 둘 다 있으면 step마다 두 종류가 다 들어가야 하므로 2 이상이어야 한다 (docs/04 §2)"
            )
        self.manifest = build_manifest(self.config, self.items, self.model)
        self.run_id: str = self.config["run_id"]
        self.step = 0
        self.progress: dict[str, Any] | None = None
        self.history: list[dict[str, Any]] = []
        self.status = "running"
        self._plans: dict[int, EpisodePlan] = {}
        #: step마다 ``hook(trainer, metrics)``로 불린다 (:meth:`run`). 학습에는 영향이 없다 — 측정·로그용
        #: (RSS·GPU peak를 step 1과 step 10에서 재는 것이 docs/06 Task 5 선결 조건 1의 확인이다).
        self.step_hook: Any = None
        if resume is not None:
            self.load(resume)

    # -- 수명 --

    def close(self) -> None:
        """잡고 있는 프로세스 전역 상태가 없다 — 스레드 수는 계산 블록마다 되돌려진다. API 호환용."""

    def __enter__(self) -> Trainer:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    @property
    def run_dir(self) -> Path:
        return Path(self.config["artifacts_dir"]) / self.run_id

    @property
    def trainable_parameters(self) -> list[Tensor]:
        return [p for p in self.model.parameters() if p.requires_grad]

    # -- 단위 --

    def _plan(self, item: Item) -> EpisodePlan:
        if item.index not in self._plans:
            sampler_config = self.config["sampler"]
            self._plans[item.index] = plan_episode(
                item.record, chunk_seconds=self.config["stream_chunk_seconds"],
                tick_weights=sampler_config["tick_weights"],
                steady_min_held_ticks=sampler_config["steady_min_held_ticks"],
            )  # fmt: skip
        return self._plans[item.index]

    def _chunk_count(self, unit: Unit) -> int:
        if unit.kind == "stream":
            return len(self._plan(self.items[unit.items[0]]).chunks)
        return 1

    def _run(self, unit: Unit, chunk_index: int, carried: StreamState | None, scale: float) -> ChunkResult:
        if unit.kind == "stream":
            item = self.items[unit.items[0]]
            plan = self._plan(item)
            return run_stream_chunk(self.model, item, plan.chunks[chunk_index], carried=carried, plan=plan, scale=scale)
        return run_single_unit(self.model, [self.items[index] for index in unit.items], scale=scale)

    def _unit_weight(self, unit: Unit) -> float:
        """단위의 상태 가중치 합 — 라벨만으로: 에피소드는 유효 틱의 종류 가중치 합, 단일 요청은 유효 상태 수."""
        if unit.kind == "stream":
            return float(self._plan(self.items[unit.items[0]]).normaliser)
        return float(sum(1 for index in unit.items if valid_single(self.items[index].record)))

    def _plan_step(self, units: list[Unit]) -> dict[str, Any]:
        """분모·비중·scale을 단위를 돌리기 전에 정한다 (모듈 설명 참조)."""
        denominators = {domain: 0.0 for domain in DOMAINS}
        for unit in units:
            denominators[unit.domain] += self._unit_weight(unit)
        present = [domain for domain in DOMAINS if denominators[domain] > 0]
        robot = float(self.config["robot_loss_share"])
        configured = {"robot": robot, "non_robot": 1.0 - robot}
        if len(present) == 2:
            shares = dict(configured)
        else:
            shares = {domain: (1.0 if domain in present else 0.0) for domain in DOMAINS}
        scales = {
            domain: (shares[domain] / denominators[domain] if denominators[domain] > 0 else 0.0) for domain in DOMAINS
        }
        return {"denominators": denominators, "shares": shares, "scales": scales}

    def _record(self, unit: Unit, result: ChunkResult, acc: dict[str, Any], scale: float) -> None:
        stats = result.stats
        domain = unit.domain
        acc["loss"] += result.value
        acc["value_by_domain"][domain] += result.value
        acc["tokens"][domain] += int(stats["tokens"])
        acc["chunks"] += 1
        acc["labels_total"] = acc.get("labels_total", 0) + int(stats["labels_total"])
        acc["labels_low_confidence"] = acc.get("labels_low_confidence", 0) + int(stats["labels_low_confidence"])
        for kind, (total, count) in stats["loss_by_type"].items():
            slot = acc["loss_by_type"].setdefault(kind, [0.0, 0])
            slot[0] += total
            slot[1] += count
        if unit.kind == "stream":
            material = unit.materials[0]
            acc["loss_sum"][domain] += stats["loss_sum"]
            acc["weight"][domain] += stats["weight_sum"] * scale
            acc["weight_by_material"][material] += stats["weight_sum"] * scale
            acc["value_by_material"][material] += result.value
            acc["valid_states"]["stream"] += int(stats["valid_ticks"])
            for name, value in stats["loss_by_class"].items():
                acc["value_by_class"][name] += value
            for name, weight in stats["weight_by_class"].items():
                acc["weight_by_class"][name] += weight
        else:
            for entry, material in zip(stats["per_item"], unit.materials):
                if not entry["valid"]:
                    continue
                acc["loss_sum"][domain] += entry["loss"]
                acc["weight"][domain] += scale
                acc["weight_by_material"][material] += scale
                acc["value_by_material"][material] += entry["contribution"]
                acc["valid_states"]["single"] += 1

    def _should_stop(self, done: tuple[int, int]) -> bool:
        stop = self.config["stop_after"]
        if stop is not None and "unit" in stop and stop["step"] == self.step and (stop["unit"], stop["chunk"]) == done:
            self.status = "interrupted"
            return True
        budget = self.config["max_wall_hours"]
        if budget is not None and time.perf_counter() - self._started > float(budget) * 3600.0:
            self.status = "interrupted"
            return True
        return False

    # -- 공개 API --

    def accumulate(self) -> bool:
        """현재 step의 남은 단위를 forward·backward한다. 끝나면 True, 중단 지점이면 False."""
        with self._threads():
            return self._accumulate()

    def _accumulate(self) -> bool:
        if self.step >= int(self.config["max_steps"]):
            raise RuntimeError(f"max_steps {self.config['max_steps']}에 이미 도달했다")
        if self.progress is None:
            units = self.sampler.draw_step(int(self.config["gradient_accumulation"]))
            self.progress = {
                "units": units,
                **self._plan_step(units),
                "unit_index": 0,
                "chunk_index": 0,
                "carried": None,
                "acc": _new_accumulators(),
            }
        progress = self.progress
        while progress["unit_index"] < len(progress["units"]):
            unit = progress["units"][progress["unit_index"]]
            scale = float(progress["scales"][unit.domain])
            chunks = self._chunk_count(unit)
            started = time.perf_counter()
            done = (progress["unit_index"], progress["chunk_index"])
            result = self._run(unit, progress["chunk_index"], progress["carried"], scale)
            if result.loss is not None and result.loss.requires_grad:
                result.loss.backward()
            self._record(unit, result, progress["acc"], scale)
            if unit.kind == "stream" and progress["chunk_index"] == 0:
                progress["acc"]["items"]["stream"] += 1
            elif unit.kind == "single":
                progress["acc"]["items"]["single"] += len(unit.items)
            if progress["chunk_index"] + 1 >= chunks:  # 단위의 마지막 구간 → 다음 단위
                progress["unit_index"] += 1
                progress["chunk_index"] = 0
                progress["carried"] = None
            else:  # 다음 구간으로: 공통 상태를 detach해 넘긴다 (gradient는 여기서 끊긴다)
                progress["chunk_index"] += 1
                progress["carried"] = detach_stream_state(result.state)
            progress["acc"]["seconds"] += time.perf_counter() - started
            if self._should_stop(done):
                return False
        return True

    def apply(self) -> dict[str, Any]:
        """clip → optimizer step → schedule step. step 지표를 돌려주고 history에 더한다."""
        with self._threads():
            return self._apply()

    def _apply(self) -> dict[str, Any]:
        progress = self.progress
        if progress is None or progress["unit_index"] < len(progress["units"]):
            raise RuntimeError("apply: accumulate()가 끝나지 않았다")
        started = time.perf_counter()
        grad_norm = clip_gradients(self.trainable_parameters, self.config["gradient_clip"])
        self.optimizer.step()
        self.scheduler.step()
        self.optimizer.zero_grad(set_to_none=True)
        self.step += 1
        acc = progress["acc"]

        def share(values: dict[str, float]) -> dict[str, float]:
            denominator = sum(values.values())
            return {key: (value / denominator if denominator > 0 else 0.0) for key, value in values.items()}

        denominators = progress["denominators"]
        tokens_total = sum(acc["tokens"].values())
        lrs = {group["name"].split("/")[0]: float(group["lr"]) for group in self.optimizer.param_groups}
        metrics = {
            "step": self.step,
            "loss": acc["loss"],
            "loss_by_domain": {
                domain: (acc["loss_sum"][domain] / denominators[domain] if denominators[domain] > 0 else None)
                for domain in DOMAINS
            },
            "loss_by_type": {kind: value / count for kind, (value, count) in acc["loss_by_type"].items() if count},
            "grad_norm": grad_norm,
            "lr": {"backbone": lrs.get("backbone"), "readout": lrs.get("readout")},
            "tokens": {**acc["tokens"], "total": tokens_total},
            "token_share": {
                domain: (acc["tokens"][domain] / tokens_total if tokens_total else 0.0) for domain in DOMAINS
            },
            "items": dict(acc["items"]),
            "valid_states": dict(acc["valid_states"]),
            "labels_total": int(acc.get("labels_total", 0)),
            "labels_low_confidence": int(acc.get("labels_low_confidence", 0)),  # 퇴화 틱의 라벨 (I4 완화, weight로 내림)
            "chunks": acc["chunks"],
            "units": [
                {
                    "kind": unit.kind, "domain": unit.domain, "materials": list(unit.materials), "tokens": unit.tokens,
                    "records": [self.items[index].record_id for index in unit.items],
                }  # fmt: skip
                for unit in progress["units"]
            ],
            "shares": dict(progress["shares"]),
            "loss_share": {  # 실제로 적용된 계수 질량 (분야는 합 1 = 0.6/0.4, 묶음·틱 종류는 그 안의 배분)
                "domain": dict(acc["weight"]),
                "material": share(acc["weight_by_material"]),
                "tick_class": share(acc["weight_by_class"]) if acc["items"]["stream"] else {},
            },
            "loss_contribution": {  # 기여 값의 몫
                "domain": share(acc["value_by_domain"]),
                "material": share(acc["value_by_material"]),
                "tick_class": share(acc["value_by_class"]) if acc["items"]["stream"] else {},
            },
            "sampler": self.sampler.realized(),
            "seconds": acc["seconds"] + (time.perf_counter() - started),
        }
        self.history.append(metrics)
        self.progress = None
        return metrics

    def run_step(self) -> dict[str, Any] | None:
        if not self.accumulate():
            return None
        return self.apply()

    def run(self) -> dict[str, Any]:
        """`max_steps`까지(또는 중단 지점까지) 돌리고 checkpoint·metrics.json을 쓴다."""
        with self._threads():
            return self._run_until_done()

    def _run_until_done(self) -> dict[str, Any]:
        max_steps = int(self.config["max_steps"])
        every = int(self.config["checkpoint_every"])
        stop = self.config["stop_after"]
        while self.step < max_steps:
            if not self.accumulate():
                break
            metrics = self.apply()
            if self.step_hook is not None:
                self.step_hook(self, metrics)
            if stop is not None and "unit" not in stop and stop["step"] == self.step:
                self.status = "interrupted"
                break
            if self.step < max_steps and every and self.step % every == 0:
                self.save()
        if self.step >= max_steps and self.status != "interrupted":
            self.status = "completed"
        path = self.save()
        self.write_metrics()
        return {
            "run_id": self.run_id,
            "run_dir": str(self.run_dir),
            "checkpoint": str(path),
            "step": self.step,
            "status": self.status,
            "metrics": {"steps": list(self.history), "summary": self._summary()},
        }

    # -- 저장·재개 --

    def _summary(self) -> dict[str, Any]:
        return {
            "last_loss": self.history[-1]["loss"] if self.history else None,
            "seconds_total": time.perf_counter() - self._started,
            "sampler": self.sampler.realized(),
            "in_progress": None if self.progress is None else {
                "unit_index": self.progress["unit_index"], "chunk_index": self.progress["chunk_index"],
            },  # fmt: skip
        }

    def checkpoint_state(self) -> dict[str, Any]:
        progress = None
        if self.progress is not None:
            p = self.progress
            progress = {
                "units": [asdict(unit) for unit in p["units"]],
                "denominators": dict(p["denominators"]),
                "shares": dict(p["shares"]),
                "scales": dict(p["scales"]),
                "unit_index": p["unit_index"],
                "chunk_index": p["chunk_index"],
                "carried_state": None if p["carried"] is None else stream_state_to_dict(p["carried"]),
                "accumulators": copy.deepcopy(p["acc"]),
                "grads": {
                    name: param.grad.detach().clone()
                    for name, param in self.model.named_parameters() if param.grad is not None
                },
            }
        return {
            "format": CHECKPOINT_FORMAT,
            "run_id": self.run_id,
            "step": self.step,
            "status": self.status,
            "model": trainable_state_dict(self.model),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "rng": collect_rng_state(),
            "sampler": self.sampler.state_dict(),
            "progress": progress,
            "config": copy.deepcopy(self.config),
            "manifest": copy.deepcopy(self.manifest),
            "history": copy.deepcopy(self.history),
        }

    def save(self, path: str | Path | None = None) -> Path:
        target = Path(path) if path is not None else self.run_dir / "checkpoint.pt"
        save_checkpoint(target, self.checkpoint_state())
        return target

    def load(self, path: str | Path) -> None:
        """checkpoint에서 이어간다. run의 정체(설정, manifest의 identity 블록, sampler 위치의 레코드 출처)가 다르면 거절한다."""
        with self._threads():
            self._load(path)

    def _load(self, path: str | Path) -> None:
        state = load_checkpoint(path)
        check_contract(state.get("manifest"), self.manifest, where=f"resume: {path}")
        saved, current = resume_config(state["config"]), resume_config(self.config)
        differences = [key for key in current if saved.get(key) != current[key]]
        if differences:
            raise ValueError(
                f"resume: checkpoint의 설정과 다르다: {differences} — 중단·예산·경로·이름({list(RESUME_FREE_KEYS)})과 "
                f"내용으로 대조하는 경로({list(RESUME_PATH_KEYS)}, dataset_manifests[].path) 말고는 같아야 한다"
            )
        saved_identity = state["manifest"].get("identity") if isinstance(state["manifest"], dict) else None
        if saved_identity is None:
            raise ValueError(
                f"resume: {path}: checkpoint의 manifest에 identity 블록이 없다(run 정체 대조 이전 형식) — 이어갈 수 없다, 새 run으로 시작한다"
            )
        mismatches = identity_differences(saved_identity, self.manifest["identity"])
        if mismatches:
            raise ValueError(
                f"resume: checkpoint의 run 정체(데이터·tokenizer·직렬화·질문 세트·모델의 내용)와 다르다: {'; '.join(mismatches)} — "
                "경로가 아니라 내용이 같아야 한다. 데이터를 바꾸는 후속 학습은 새 run으로 시작한다"
            )
        self.run_id = str(state["run_id"])
        self.config["run_id"] = self.run_id  # 이어가는 run의 정체는 checkpoint의 것
        load_trainable_state(self.model, state["model"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.scheduler.load_state_dict(state["scheduler"])
        restore_rng_state(state["rng"])
        self.sampler.load_state_dict(state["sampler"])
        self.step = int(state["step"])
        self.history = list(state.get("history", []))
        self.status = "running"
        self.optimizer.zero_grad(set_to_none=True)
        progress = state["progress"]
        self.progress = None
        if progress is not None:
            units = [Unit(**unit) for unit in progress["units"]]
            carried = progress["carried_state"]
            named = dict(self.model.named_parameters())
            for name, grad in progress["grads"].items():
                if name not in named:
                    raise ValueError(f"resume: 저장된 gradient의 파라미터가 없다: {name}")
                named[name].grad = grad.clone().to(named[name].dtype)
            self.progress = {
                "units": units,
                "denominators": {domain: float(v) for domain, v in progress["denominators"].items()},
                "shares": {domain: float(v) for domain, v in progress["shares"].items()},
                "scales": {domain: float(v) for domain, v in progress["scales"].items()},
                "unit_index": int(progress["unit_index"]),
                "chunk_index": int(progress["chunk_index"]),
                "carried": None if carried is None else stream_state_from_dict(carried, self.model.backbone),
                "acc": copy.deepcopy(progress["accumulators"]),
            }

    def write_metrics(self) -> Path:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        (self.run_dir / "config.yaml").write_text(yaml.safe_dump(self.config, allow_unicode=True, sort_keys=False), encoding="utf-8")
        payload = {
            "run_id": self.run_id,
            "status": self.status,
            "step": self.step,
            "config": self.config,
            "manifest": self.manifest,
            "steps": self.history,
            "summary": self._summary(),
        }
        path = self.run_dir / "metrics.json"
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
        return path


# --------------------------------------------------------------------------
# 공개 API·CLI
# --------------------------------------------------------------------------


def train(config: dict) -> dict[str, Any]:
    """설정대로 학습한다. ``config["resume"]``가 있으면 그 checkpoint에서 이어간다.

    돌려주는 것: ``run_id``, ``run_dir``, ``checkpoint``(경로), ``step``(마지막 step), ``status``
    (``completed`` / ``interrupted``), ``metrics``(step별 지표와 요약).
    """
    with Trainer(config, resume=config.get("resume")) as trainer:
        return trainer.run()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m robo_jev.train", description="robojev 학습 (단일 프로세스, CPU 검증 경로)")
    parser.add_argument("--config", required=True, help="학습 설정 YAML (configs/train/tiny_cpu.yaml)")
    parser.add_argument("--resume", default=None, help="이어갈 checkpoint 경로")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="설정 값 덮어쓰기 (값은 YAML로 읽는다)")
    args = parser.parse_args(argv)
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8")) or {}
    for assignment in args.set:
        key, separator, value = assignment.partition("=")
        if not separator:
            parser.error(f"--set은 KEY=VALUE 꼴이어야 한다: {assignment!r}")
        config[key] = yaml.safe_load(value)
    if args.resume:
        config["resume"] = args.resume
    result = train(config)
    print(json.dumps(
        {
            "run_id": result["run_id"], "checkpoint": result["checkpoint"], "step": result["step"],
            "status": result["status"], "loss": result["metrics"]["summary"]["last_loss"],
        },
        ensure_ascii=False,
    ))  # fmt: skip
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
