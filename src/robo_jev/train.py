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

**저장·재개 (docs/03 §5).** step 사이에서는 model/optimizer/scheduler/RNG/sampler 위치/config/manifest를,
step 도중(구간 경계)에서는 여기에 진행 위치(단위·구간 index), 누적 gradient, 이어 붙일 공통 상태를
더해 :func:`robo_jev.checkpoint.save_checkpoint` 로 atomic하게 쓴다. 재개는 그 위치의 다음 구간부터
이어가며, 같은 seed의 연속 실행과 FP32·CPU에서 비트 단위로 같아야 한다(tests/test_resume.py). 중단은
``stop_after``(결정적 검사용)나 ``max_wall_hours``(예산)로 구간 경계에서 일어난다.

이 모듈은 generator·simulator·하네스를 import하지 않는다 (docs/06 §1).
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
import subprocess
import sys
import time
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
from robo_jev.model.hybrid import DEFAULT_CONFIG
from robo_jev.model.judge import Judge
from robo_jev.model.serialize import TOKEN_SERIALIZER_VERSION
from robo_jev.model.stream import StreamState
from robo_jev.model.tokenizer import WhitespaceTokenizer, load_tokenizer
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
    "RESUME_FREE_KEYS",
    "Trainer",
    "build_model",
    "clip_gradients",
    "detach_stream_state",
    "episode_chunks",
    "layout_prefix",
    "lr_factor",
    "main",
    "parameter_groups",
    "plan_episode",
    "resolve_config",
    "run_single_unit",
    "run_stream_chunk",
    "train",
]

# --------------------------------------------------------------------------
# 설정
# --------------------------------------------------------------------------

READOUTS = {"decision_pointer": "pointer", "candidate_branch": "candidate_branch"}
DTYPES = {"float32": torch.float32, "float64": torch.float64}
TRAINABLE = ("readout_only", "text_backbone_and_readout")
EXECUTION_BACKENDS = ("independent_paths",)  # P0. `shared_hybrid`(P1)는 state_first에 아직 없다 (4b 보고 §5)
OPTIMIZERS = ("adamw",)
DEFAULT_TICK_WEIGHTS = {"steady": 0.25, "event": 2.0, "goal_change": 2.0, "other": 1.0}
DEFAULT_SAMPLER = {
    "material_shares": dict(DEFAULT_MATERIAL_SHARES),
    "domain_tag": "provenance.domain",
    "material_tag": "provenance.material",
    "tick_weights": dict(DEFAULT_TICK_WEIGHTS),
    "steady_min_held_ticks": 3,
}

#: 재개할 때 checkpoint의 설정과 달라도 되는 키 — 중단·예산·경로·이름뿐이다(run id는 checkpoint의 것을
#: 쓴다). 나머지는 run의 정체라 같아야 한다.
RESUME_FREE_KEYS = ("resume", "stop_after", "max_wall_hours", "checkpoint_every", "artifacts_dir", "run_id", "run_name")

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
    "activation_checkpointing": False,
    "max_steps": None,
    "max_wall_hours": None,
    "seed": 17,
    "torch_threads": 1,
    "checkpoint_every": None,
    "artifacts_dir": "artifacts/runs",
    "stop_after": None,
    "resume": None,
    "sampler": dict(DEFAULT_SAMPLER),
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
        unknown = [key for key in entry if key not in ("path", "domain", "material")]
        _need(not unknown, f"dataset_manifests[{position}]: 알 수 없는 키 {unknown} (허용: ['path', 'domain', 'material'])")
        path = entry.get("path")
        _need(isinstance(path, str) and bool(path), f"dataset_manifests[{position}].path: manifest 경로(문자열)가 필요하다")
        domain = entry.get("domain")
        _need(domain is None or domain in DOMAINS, f"dataset_manifests[{position}].domain: {list(DOMAINS)} 중 하나이거나 null이어야 한다 (받은 값: {domain!r})")
        material = entry.get("material")
        _need(material is None or material in MATERIALS, f"dataset_manifests[{position}].material: {list(MATERIALS)} 중 하나이거나 null이어야 한다 (받은 값: {material!r})")
        out.append({"path": path, "domain": domain, "material": material})
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
    _need(out["execution_backend"] in EXECUTION_BACKENDS, f"execution_backend: {list(EXECUTION_BACKENDS)}만 구현했다 — shared_hybrid(P1)는 state_first에 아직 없다 (받은 값: {out['execution_backend']!r})")
    _need(out["readout"] in READOUTS, f"readout: {list(READOUTS)} 중 하나여야 한다 (받은 값: {out['readout']!r})")
    _need(out["dtype"] in DTYPES, f"dtype: {list(DTYPES)}만 CPU 검증 범위다 — BF16 허용 오차는 클라우드 단계 (받은 값: {out['dtype']!r})")
    _need(out["trainable"] in TRAINABLE, f"trainable: {list(TRAINABLE)} 중 하나여야 한다 (받은 값: {out['trainable']!r})")
    _need(out["optimizer"] in OPTIMIZERS, f"optimizer: {list(OPTIMIZERS)}만 구현했다 (받은 값: {out['optimizer']!r})")
    _need(out["activation_checkpointing"] is False, "activation_checkpointing: CPU fixture 경로에는 없다 — 클라우드 단계에서 붙인다 (false여야 한다)")
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
    """설정의 backbone fixture + readout. FSDP 등 wrapper는 이 함수의 결과에 씌운다."""
    judge = Judge.from_config(
        config["model_config"], seed=config["model_seed"], readout=READOUTS[config["readout"]],
        vocab_size=config["model_vocab_size"],
    )  # fmt: skip
    judge = judge.to(DTYPES[config["dtype"]])
    if config["trainable"] == "readout_only":
        judge.backbone.requires_grad_(False)  # T0: readout만 (docs/03 §5)
    return judge


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
    if config["trainable"] == "text_backbone_and_readout":
        add("backbone", backbone, config["backbone_lr"])
    add("readout", readout, config["readout_lr"])
    return groups


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


def detach_stream_state(state: StreamState, *, requires_grad: bool = False) -> StreamState:
    """구간 경계에서 넘기는 공통 상태 — 모든 tensor를 detach한 새 상태 (값은 같고 gradient만 끊긴다).

    `requires_grad=True`면 detach한 tensor를 leaf로 만들어 다음 구간의 gradient가 경계에 얼마나
    닿는지 관찰할 수 있다(검사용).
    """
    if state.is_branch:
        raise ValueError("branch 상태는 넘기지 않는다 — 다음 구간은 분기 이전 공통 상태에서 이어간다")

    def cut(tensor: Tensor | None) -> Tensor | None:
        if tensor is None:
            return None
        out = tensor.detach()
        if requires_grad and out.is_floating_point():
            out.requires_grad_(True)
        return out

    return StreamState(
        state.backbone,
        delta=[{key: cut(value) for key, value in layer.items()} for layer in state.delta],
        kv=[{key: cut(value) for key, value in layer.items()} for layer in state.kv],
        cache_ticks=state.cache_ticks.detach(),
        position=state.position,
        tick=state.tick,
        window_ticks=state.window_ticks,
        prefix_hidden=cut(state.prefix_hidden),
        hidden=cut(state.hidden),
        is_branch=False,
    )


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


def build_manifest(config: dict, items: list[Item], model: Judge) -> dict[str, Any]:
    """checkpoint에 함께 적는 것: 데이터 manifest 참조(manifest마다 경로·sha256·파일 해시·분야 태그·레코드 수), 토큰
    직렬화·질문 세트 버전, git SHA, 모델 fixture."""
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
    return {
        "dataset_manifests": datasets,
        "splits": list(config["splits"]),
        "serializer_version": TOKEN_SERIALIZER_VERSION,  # 토큰 직렬화의 버전 (레코드의 versions.serializer와 다른 것)
        "question_set": {"id": "qs-v0", "markers": {qid: spec["marker"] for qid, spec in QUESTION_SET_V0.items()}},
        "layouts": dict(config["layout"]),
        "tokenizer": config["tokenizer"],
        "model": {
            "id": config["model_id"],
            "config": str(config["model_config"]),
            "name": model.backbone.config.name,
            "vocab_size": model.backbone.config.vocab_size,
            "seed": model.backbone.config.seed if config["model_seed"] is None else config["model_seed"],
            "readout": model.readout,
            "rank": model.rank,
            "revision_manifest": config["model_revision_manifest"],
        },
        "git": git_revision(),
        "torch": str(torch.__version__),  # TorchVersion 객체가 아니라 문자열 — weights_only 로 읽힌다
    }


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
    """설정 하나의 학습 상태 (모듈 설명 참조). context manager로 쓰면 스레드 수를 되돌린다.

    * :meth:`accumulate` — 현재 step의 accumulation 단위들(재개했으면 남은 것)을 forward·backward한다.
      끝나면 `True`, 중단 지점(`stop_after`·`max_wall_hours`)이면 진행 위치를 남기고 `False`.
    * :meth:`apply` — clip → optimizer step → schedule step, step 지표를 돌려준다.
    * :meth:`run` — `max_steps`까지 돌리고 checkpoint·metrics를 쓴다.
    """

    def __init__(self, config: dict, *, resume: str | Path | None = None) -> None:
        self.config = resolve_config(config)
        self._threads_before = torch.get_num_threads()
        torch.set_num_threads(int(self.config["torch_threads"]))
        try:
            self._build(resume)
        except BaseException:
            self.close()
            raise

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
            material_tag=sampler_config["material_tag"],
        )  # fmt: skip
        if not self.items:
            raise ValueError(f"dataset_manifests: split {self.config['splits']}에 레코드가 없다")
        self.model = build_model(self.config)
        vocab = self.model.backbone.config.vocab_size
        largest = max(max(item.layout["tokens"]) for item in self.items)
        if largest >= vocab:
            raise ValueError(f"model_vocab_size: 토큰 id {largest}가 어휘 {vocab}를 넘는다 — tokenizer에 맞는 어휘를 써야 한다")
        self.optimizer = torch.optim.AdamW(parameter_groups(self.model, self.config), betas=(0.9, 0.999), eps=1e-8)
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
        if resume is not None:
            self.load(resume)

    # -- 수명 --

    def close(self) -> None:
        torch.set_num_threads(self._threads_before)

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
        max_steps = int(self.config["max_steps"])
        every = int(self.config["checkpoint_every"])
        stop = self.config["stop_after"]
        while self.step < max_steps:
            if not self.accumulate():
                break
            self.apply()
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
            "model": self.model.state_dict(),
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
        """checkpoint에서 이어간다. run의 정체(설정)가 다르면 거절한다."""
        state = load_checkpoint(path)
        saved = state["config"]
        differences = [
            key for key in self.config
            if key not in RESUME_FREE_KEYS and saved.get(key) != self.config[key]
        ]  # fmt: skip
        if differences:
            raise ValueError(
                f"resume: checkpoint의 설정과 다르다: {differences} — 중단·예산·경로({list(RESUME_FREE_KEYS)}) 말고는 같아야 한다"
            )
        self.run_id = str(state["run_id"])
        self.config["run_id"] = self.run_id  # 이어가는 run의 정체는 checkpoint의 것
        self.model.load_state_dict(state["model"])
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
