"""Task P1 stage C — docs/06 Task 5의 GPU 인수 검사를 실제 2B에서 하나씩 재고 `artifacts/reports/p1-acceptance.json`에 적는다.

세 가지다 (docs/06:315-320, :362).

* ``frozen`` (C1) — T0 한 step에서 **readout만 바뀌고 backbone은 한 tensor도 바뀌지 않는다**. 모든 backbone tensor의
  sha256을 step 전후로 대조하고(표본이 아니라 전부), `trainable_state_dict`가 readout만 저장하는지 본다.
* ``trains`` (C2) — LoRA와 **text backbone 전체(T1)** 짧은 run: text backbone 파라미터에 gradient가 0이 아니고,
  step 사이에 값이 실제로 움직이며(고정 표본의 Δ L2), readout도 움직인다. s/step·peak GiB·tokens/s를 함께 적는다.
* ``resume`` (C3, 긴 run의 관문) — "동일 seed의 20 step 연속 실행 대 10 step 저장 + **프로세스 재시작** + 10 step".
  재시작은 진짜 재시작이다(이 스크립트가 자기 자신을 `--phase`로 다시 띄운다). loss·sampler 위치·optimizer step 수·
  고정 parameter tensor를 대조하며, **허용 오차는 비교 전에 :data:`RESUME_TOLERANCE`로 고정**되어 있다.

모든 진입점은 :func:`robo_jev.gpu.limit_gpu_memory` 안에서 돈다. 실행:

    uv run python scripts/p1_acceptance.py --check frozen,trains,resume --out artifacts/reports/p1-acceptance.json
"""

from __future__ import annotations

import argparse
import copy
import datetime as dt
import hashlib
import json
import subprocess
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from robo_jev.gpu import DEFAULT_FRACTION, limit_gpu_memory, memory_report, require_free

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

CHECKS = ("frozen", "trains", "resume", "baseline")
DEFAULT_CONFIG = REPO / "configs" / "train" / "qwen35-2b-pilot.yaml"

#: **비교 전에 고정한** 재개 허용 오차 (C3). 정수(위치·step 수·뽑힌 단위)는 정확히 같아야 하고, 수치는 아래 안이어야 한다.
#: BF16 backbone + fp32 readout이고 재개는 **다른 프로세스**라 kernel 선택(Triton autotune)이 달라질 수 있다 — G0b 리뷰 1 I4에서
#: 같은 코드의 fp32 오차가 프로세스에 따라 0.0101 ↔ 0.0243으로 움직인 것이 그 증거다. 그래서 비트 동일을 요구하지 않고,
#: "재개가 다른 데이터·다른 optimizer 상태로 이어가는 것"을 잡는 크기로 잡는다(그 실패는 loss를 0.1 이상 움직인다).
RESUME_TOLERANCE = {
    "loss_abs": 0.02,        # step마다의 |Δloss|
    "loss_rel": 0.02,        # 또는 상대 2 %
    "param_max_abs": 0.01,   # 고정 표본 tensor의 최대 절대 차
    "param_rel_l2": 0.05,    # 그 tensor의 상대 L2 (G0b의 BF16 readout 상수와 같은 크기)
}
#: **학습 범위별** 사전 등록 허용 오차. 등록된 것은 `t0`뿐이다 — :data:`RESUME_TOLERANCE` 는 P1이 **T0** run에
#: 대고(그 경로는 이 상자에서 비트 결정적이었다) 비교 전에 고정한 값이기 때문이다. T1은 재시작이 전혀 없는 두
#: 프로세스의 loss가 이미 0.064 벌어진다(P2 보고서 A1b) — 그 run을 T0의 오차로 재고 `passed: false`라 적으면
#: "재개가 깨졌다"로 읽히지만 실제로 깬 것은 **오차가 그 범위에 등록돼 있지 않다**는 사실이다. 본 값에 맞춰
#: 오차를 고치는 것은 이 프로젝트가 금하는 수이므로, 등록될 때까지 그 범위의 판정은 `passed: None`이고 게이트는
#: 멈춘다(사전 등록 절차: 재시작 없는 같은 설정 두 run의 벌어짐을 먼저 기록하고, 그 위에 오차를 고정한다).
#: **T1 범위의 사전 등록 허용 오차** — **세 쌍**에서 (R2 A1, 2026-09-23). P3가 남긴 이월 항목("쌍을 최소 셋 재서
#: 그 최댓값 위에 다시 고정한다")을 그대로 닫은 값이다. :data:`RESUME_TOLERANCE_RULE` 은 셋째 쌍을 재기 **전에**
#: 고쳐 커밋했고(`19292e4`), 여기 적용만 했다.
#:
#: 세 쌍의 최악값 (`PRIOR_BASELINE_PAIRS["t1"]` 둘 + `artifacts/reports/r2-acceptance.json`의 `checks.baseline_t1`):
#:
#:   | 쌍 | 데이터 | step | loss \|Δ\| | loss 상대 | param 최대 절대 | param 상대 L2 |
#:   | P2 2026-09-21 | D1  | 3 | 0.064136 | 1.778 % | (snapshot 없음) | (없음) |
#:   | P3 2026-09-22 | D1  | 5 | 0.023574 | 0.826 % | 5.819e-4 | 6.31e-4 |
#:   | R2 2026-09-23 | R1  | 5 | **0.076560** | **3.933 %** | **6.079e-4** | **9.124e-4** |
#:
#: 곧 셋의 퍼짐은 **3.2배**(0.0236 ~ 0.0766)이고 **가장 큰 쌍이 가장 마지막에 나왔다** — 한 쌍으로 오차를 고정하는
#: 것이 왜 위험한지가 이 표다. 규칙대로 ×2 → 유효숫자 한 자리 올림 → T0보다 느슨하게만:
#: loss_abs 0.1531 → **0.2**, loss_rel 0.0787 → **0.08**, param 둘은 T0의 값이 더 커서 그대로(**0.01**·**0.05**).
#:
#: **한계는 그대로 적는다**: 셋 가운데 둘은 D1 데이터, 하나는 R1 데이터다 — 같은 *경로*(2B·T1·fp32 master·5초 구간)
#: 이지만 같은 *설정*은 아니다. 그리고 0.2는 느슨하다: 이 오차는 "재개가 다른 데이터·다른 optimizer 상태로
#: 이어가는 것"(loss를 0.1 이상 **일관되게** 움직인다)을 잡되, 그보다 작은 어긋남은 이 경로의 프로세스 간 잡음과
#: 구분하지 못한다. 정수 기준(sampler 위치·뽑힌 단위·step 수·빠진 tensor)이 그 구분의 대부분을 지고 있다.
RESUME_TOLERANCE_T1 = {"loss_abs": 0.2, "loss_rel": 0.08, "param_max_abs": 0.01, "param_rel_l2": 0.05}

RESUME_TOLERANCES: dict[str, dict[str, float]] = {"t0": RESUME_TOLERANCE, "t1": RESUME_TOLERANCE_T1}

#: 상대 L2의 **0에 가까운 분모 가드** (P2 A1b에서 `bias` 한 tensor가 절대 차이 4.85e-7인데 자기 norm이 ≈3e-6이라
#: 비 0.16으로 걸렸다). 자기 L2 norm이 이 값보다 작은 tensor는 **상대** 기준에서 빼고 절대 기준(`param_max_abs`)으로만
#: 본다 — 원소 수가 몇이든 norm이 1e-3 아래인 tensor는 사실상 0이고, 그 위의 비는 분모가 정하는 수다. |p| ≈ 0.03에서
#: bf16 눈금 한 칸이 1.22e-4이므로 1e-3은 그 여덟 칸이다. **값을 보기 전에 고정했다** (P3 D).
RELATIVE_L2_REFERENCE_FLOOR = 1e-3

#: 사전 등록 규칙 — **재시작 없는 두 run의 퍼짐**에서 그 범위의 허용 오차를 만드는 방법. 값을 보기 전에 고정한다
#: (P2가 남긴 이월 항목: T1은 재시작이 전혀 없는 두 프로세스의 loss가 이미 0.064 벌어진다).
#: 규칙: 기준마다 **두 run 사이 최악값 × :data:`TOLERANCE_SAFETY_FACTOR`**를 유효숫자 한 자리로 **올림**하고,
#: T0의 값보다 **느슨해지기만** 한다(T1 경로가 T0보다 조용할 리 없으므로 더 조이지 않는다).
TOLERANCE_SAFETY_FACTOR = 2.0
#: 그 범위의 오차를 고정하기 전에 있어야 하는 **최소 쌍 수** (R2 A1). 한 쌍은 이 경로의 잡음을 대표하지 못한다 —
#: P2의 쌍(0.0641)과 P3의 쌍(0.0236)이 **2.7배** 달랐고, 뒤엣것 하나로 고정한 오차는 이미 관측된 퍼짐보다 작았다.
MINIMUM_BASELINE_PAIRS = 3
RESUME_TOLERANCE_RULE = (
    "run the same config twice with no restart (two separate processes, same seed) and measure the run-to-run "
    "spread; do this at least MINIMUM_BASELINE_PAIRS times on that scope's path; then, for each criterion, take "
    "the worst value over every measured pair, multiply by TOLERANCE_SAFETY_FACTOR, round up to one significant "
    "figure, and never go below the t0 tolerance. A pair that could not measure a criterion does not contribute "
    "to it. Fixed in this file before the pair that completes the set was measured."
)

#: 이 범위의 경로에서 **이미 잰** 재시작 없는 쌍들 — 값은 저장된 보고서에서 읽었고, 어디서 왔는지가 함께 적혀
#: 있다(시험이 보고서와 대조한다). 새로 재는 쌍은 :func:`check_baseline` 이 여기에 이어 붙인다.
PRIOR_BASELINE_PAIRS: dict[str, list[dict[str, Any]]] = {
    "t1": [
        {
            "label": "P2 (2026-09-21) — `continuous`의 앞 3 step 대 `first`의 3 step",
            "unit": "two runs of the same config, same seed, **no restart** (two separate processes)",
            "source": "artifacts/reports/p2-acceptance.json",
            "config": "configs/train/qwen35-2b-pilot.yaml (D1 데이터)",
            "steps": 3,
            "worst_loss_abs": 0.06413567066192627,
            "worst_loss_rel": 0.017781185372826285,
            # snapshot(`first-t1.pt`)이 남아 있지 않다 — 이 쌍은 loss 기준에만 기여한다
            "worst_param_max_abs": None,
            "worst_param_relative_l2": None,
        },
        {
            "label": "P3 (2026-09-22) — `--check baseline` 두 프로세스, 5 step",
            "unit": "two runs of the same config, same seed, **no restart** (two separate processes)",
            "source": "artifacts/reports/p3-acceptance.json",
            "config": "configs/train/qwen35-2b-pilot.yaml (D1 데이터)",
            "steps": 5,
            "worst_loss_abs": 0.023573994636535645,
            "worst_loss_rel": 0.00825756566314512,
            "worst_param_max_abs": 0.000581890344619751,
            "worst_param_relative_l2": 0.0006311355571226999,
        },
        {
            "label": "R2 (2026-09-23) — `--check baseline` 두 프로세스, 5 step",
            "unit": "two runs of the same config, same seed, **no restart** (two separate processes)",
            "source": "artifacts/reports/r2-acceptance.json",
            "config": "configs/train/qwen35-2b-r2.yaml (R1 데이터, rollout 라벨판)",
            "steps": 5,
            "worst_loss_abs": 0.07655954360961914,
            "worst_loss_rel": 0.0393281228574283,
            "worst_param_max_abs": 0.0006078882142901421,
            "worst_param_relative_l2": 0.0009123694716359487,
        },
    ],
}

#: T1·LoRA에서 "움직였다"고 보는 최소 변화 — 이보다 작으면 업데이트가 빠진 것이다.
MOVE_EPSILON = 1e-9


def _repo_relative(path: str | Path) -> str:
    """저장소 안의 경로는 저장소 기준 상대 경로로 — worktree의 절대 경로를 보고서에 남기지 않는다.

    `--config configs/…`처럼 **이미 상대 경로**로 받은 값이 `Path.relative_to(REPO)`에서 떨어지던 자리를 함께
    막는다(R2 A1에서 첫 run이 여기서 죽었다)."""
    resolved = Path(path).resolve()
    try:
        return str(resolved.relative_to(REPO))
    except ValueError:
        return str(path)


def _runner() -> Any:
    import importlib.util

    spec = importlib.util.spec_from_file_location("adapt_readout", REPO / "scripts" / "adapt_readout.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["adapt_readout"] = module
    spec.loader.exec_module(module)
    return module


def _config(mode: str, *, steps: int, run_id: str, overrides: dict[str, Any] | None = None, config_path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    return _runner().load_train_config(config_path, mode=mode, steps=steps, run_id=run_id, overrides=overrides or {})


def _hash(tensor: Any) -> str:
    """tensor의 바이트 그대로의 sha256 — bf16은 numpy가 모르는 dtype이라 uint8로 다시 본다(값 변환 없음)."""
    import torch

    return hashlib.sha256(tensor.detach().to("cpu").contiguous().flatten().view(torch.uint8).numpy().tobytes()).hexdigest()


def _memory() -> dict[str, Any]:
    import torch

    return {"peak_allocated_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2), "peak_reserved_gib": round(torch.cuda.max_memory_reserved() / 2**30, 2)}


# --------------------------------------------------------------------------
# C1 — T0는 backbone을 얼리고 readout만 바꾼다
# --------------------------------------------------------------------------


def check_frozen(*, steps: int = 1, config_path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    import torch

    from robo_jev.train import Trainer, trainable_state_dict

    torch.cuda.reset_peak_memory_stats()
    config = _config("t0", steps=steps, run_id=f"p1-c1-frozen-{dt.datetime.now().strftime('%H%M%S')}", config_path=config_path)
    started = time.perf_counter()
    with Trainer(config) as trainer:
        model = trainer.model
        backbone_before = {name: _hash(parameter) for name, parameter in model.backbone.named_parameters()}
        readout_before = {name: parameter.detach().clone() for name, parameter in model.named_parameters() if not name.startswith("backbone.")}
        requires_grad = [name for name, parameter in model.backbone.named_parameters() if parameter.requires_grad]
        metrics = trainer.run_step()
        backbone_after = {name: _hash(parameter) for name, parameter in model.backbone.named_parameters()}
        changed = sorted(name for name in backbone_before if backbone_before[name] != backbone_after[name])
        readout_delta = {
            name: {
                "max_abs": float((parameter.detach() - readout_before[name]).abs().max()),
                "l2": float((parameter.detach() - readout_before[name]).norm()),
            }
            for name, parameter in model.named_parameters() if not name.startswith("backbone.")
        }
        saved = trainable_state_dict(model)
        out = {
            "check": "frozen", "model_id": config["model_id"], "steps": int(steps), "seconds": round(time.perf_counter() - started, 1),
            "backbone_tensors": len(backbone_before), "backbone_tensors_changed": len(changed), "changed_names": changed[:5],
            "backbone_requires_grad": len(requires_grad),
            "readout_tensors": len(readout_delta), "readout_delta": readout_delta,
            "readout_moved": all(entry["max_abs"] > MOVE_EPSILON for entry in readout_delta.values()),
            "saved_keys": sorted(saved), "saved_only_readout": all(not key.startswith("backbone.") for key in saved),
            "loss": metrics["loss"] if metrics else None, "grad_norm": metrics["grad_norm"] if metrics else None,
            "trainable_parameters": int(sum(p.numel() for p in model.parameters() if p.requires_grad)),
            "memory": _memory(),
        }
    out["passed"] = bool(out["backbone_tensors_changed"] == 0 and out["readout_moved"] and out["saved_only_readout"] and out["backbone_requires_grad"] == 0)
    return out


# --------------------------------------------------------------------------
# C2 — LoRA와 T1은 실제로 backbone을 학습한다
# --------------------------------------------------------------------------


def check_trains(mode: str, *, steps: int = 3, sample: int = 8, config_path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    """`mode`(lora | t1)로 `steps` step 돌리고 gradient·parameter 이동·비용을 잰다."""
    import gc

    import torch

    from robo_jev.train import Trainer

    gc.collect()  # 앞 모드(LoRA)의 모델·optimizer가 아직 allocator에 잡혀 있으면 다음 모드가 울타리를 넘는다
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.reset_accumulated_memory_stats()
    require_free(60 * 2**30, what=f"acceptance {mode}")
    config = _config(mode, steps=steps, run_id=f"p1-c2-{mode}-{dt.datetime.now().strftime('%H%M%S')}", config_path=config_path)
    started = time.perf_counter()
    with Trainer(config) as trainer:
        model = trainer.model
        trainable = [(name, parameter) for name, parameter in model.backbone.named_parameters() if parameter.requires_grad]
        # 고정 표본: 학습 대상 backbone tensor를 이름 정렬 순서로 균등 간격으로 뽑는다 (run 사이에 같은 표본)
        picked = [trainable[index] for index in range(0, len(trainable), max(1, len(trainable) // sample))][:sample]
        before = {name: parameter.detach().clone().float() for name, parameter in picked}
        readout_before = {name: parameter.detach().clone() for name, parameter in model.named_parameters() if not name.startswith("backbone.")}
        load_seconds = round(time.perf_counter() - started, 1)

        # gradient는 optimizer step 뒤 지워지므로(set_to_none) 첫 step의 accumulate와 apply **사이**에서 읽는다
        trainer.accumulate()
        nonzero = 0
        max_abs = 0.0
        with_grad = 0
        for name, parameter in trainable:
            if parameter.grad is None:
                continue
            with_grad += 1
            value = float(parameter.grad.detach().abs().max())
            max_abs = max(max_abs, value)
            nonzero += int(value > 0)
        readout_grad = [float(p.grad.abs().max()) for n, p in model.named_parameters() if not n.startswith("backbone.") and p.grad is not None]
        trainer.apply()
        while trainer.step < int(config["max_steps"]):
            if not trainer.accumulate():
                break
            trainer.apply()
        metrics = trainer.history
        seconds = [m["seconds"] for m in metrics]
        tokens = [m["tokens"]["total"] for m in metrics]
        def _delta(name: str, parameter: Any) -> dict[str, Any]:
            difference = parameter.detach().float() - before[name]
            reference = float(before[name].norm())
            # LoRA의 `lora_B`는 0으로 시작하므로 상대 L2의 기준이 0이다 — 그 칸은 None으로 둔다(나눗셈으로 만든 큰 수를 적지 않는다).
            return {"max_abs": float(difference.abs().max()), "l2": float(difference.norm()),
                    "relative_l2": (float(difference.norm()) / reference) if reference > 0 else None,
                    "reference_l2": reference}

        delta = {name: _delta(name, parameter) for name, parameter in picked}
        readout_delta = {
            name: float((parameter.detach() - readout_before[name]).abs().max())
            for name, parameter in model.named_parameters() if not name.startswith("backbone.")
        }
        out = {
            "check": "trains", "mode": mode, "model_id": config["model_id"], "steps": len(metrics),
            "trainable": config["trainable"], "stream_chunk_seconds": config["stream_chunk_seconds"],
            "activation_checkpointing": config["activation_checkpointing"],
            "trainable_backbone_tensors": len(trainable),
            "trainable_parameters": int(sum(p.numel() for p in model.parameters() if p.requires_grad)),
            "gradient": {"tensors_with_grad": with_grad, "tensors_nonzero": nonzero, "max_abs": max_abs, "readout_max_abs": max(readout_grad) if readout_grad else None},
            "sample_tensors": [name for name, _ in picked], "parameter_delta": delta,
            "backbone_moved": all(entry["max_abs"] > MOVE_EPSILON for entry in delta.values()),
            "readout_moved": all(value > MOVE_EPSILON for value in readout_delta.values()),
            "readout_delta_max_abs": readout_delta,
            "load_seconds": load_seconds,
            "step_seconds": {"mean": round(sum(seconds) / len(seconds), 2), "p50": round(sorted(seconds)[len(seconds) // 2], 2), "max": round(max(seconds), 2)},
            "tokens": {"total": int(sum(tokens)), "per_second": round(sum(tokens) / max(sum(seconds), 1e-9), 1)},
            "loss_first_last": [metrics[0]["loss"], metrics[-1]["loss"]],
            "memory": _memory(),
        }
    out["passed"] = bool(out["backbone_moved"] and out["readout_moved"] and out["gradient"]["tensors_nonzero"] > 0)
    gc.collect()
    torch.cuda.empty_cache()
    return out


# --------------------------------------------------------------------------
# C3 — 저장·재개 동등성 (긴 run의 관문)
# --------------------------------------------------------------------------


def _snapshot(trainer: Any) -> dict[str, Any]:
    """비교에 쓰는 것: 학습 대상 tensor 전부(이름 → CPU tensor), sampler 위치, optimizer step 수."""
    return {
        "parameters": {name: parameter.detach().to("cpu").clone() for name, parameter in trainer.model.named_parameters() if parameter.requires_grad},
        "sampler": trainer.sampler.state_dict(),
        "optimizer_steps": int(trainer.scheduler.last_epoch),
        "step": int(trainer.step),
        "losses": [m["loss"] for m in trainer.history],
        "units": [[record for unit in m["units"] for record in unit["records"]] for m in trainer.history],
    }


def _save_snapshot(snapshot: dict[str, Any], path: Path) -> None:
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(snapshot, path)


def run_resume_phase(phase: str, *, steps: int, run_dir: Path, config_path: Path = DEFAULT_CONFIG, mode: str = "t0") -> dict[str, Any]:
    """C3의 한 조각 — 자기 자신을 다시 띄워 돌린다(진짜 프로세스 재시작). `phase`는 continuous | first | second.

    `mode`는 t0 | lora | t1이다. P1의 게이트는 **T0만** 덮었다(readout tensor 셋뿐 — backbone optimizer 상태도,
    LoRA adapter도, fp32 master 사본도 지나지 않는다). 그래서 이 게이트는 학습 범위를 인자로 받는다 (P1 리뷰 1 M3)."""
    from robo_jev.train import Trainer

    half = steps // 2
    if phase == "continuous":
        config = _config(mode, steps=steps, run_id=f"p1-c3-continuous-{mode}", overrides={"artifacts_dir": str(run_dir)}, config_path=config_path)
        with Trainer(config) as trainer:
            trainer.run()
            _save_snapshot(_snapshot(trainer), run_dir / f"continuous-{mode}.pt")
            return {"phase": phase, "mode": mode, "step": trainer.step, "losses": [m["loss"] for m in trainer.history]}
    if phase == "first":
        config = _config(mode, steps=steps, run_id=f"p1-c3-split-{mode}", overrides={"artifacts_dir": str(run_dir), "stop_after": {"step": half}, "checkpoint_every": half}, config_path=config_path)
        with Trainer(config) as trainer:
            result = trainer.run()
            return {"phase": phase, "mode": mode, "step": trainer.step, "checkpoint": result["checkpoint"], "losses": [m["loss"] for m in trainer.history]}
    if phase == "second":
        config = _config(mode, steps=steps, run_id=f"p1-c3-split-{mode}", overrides={"artifacts_dir": str(run_dir)}, config_path=config_path)
        checkpoint = run_dir / f"p1-c3-split-{mode}" / "checkpoint.pt"
        with Trainer(config, resume=checkpoint) as trainer:
            trainer.run()
            _save_snapshot(_snapshot(trainer), run_dir / f"split-{mode}.pt")
            return {"phase": phase, "mode": mode, "step": trainer.step, "losses": [m["loss"] for m in trainer.history]}
    raise ValueError(f"phase: continuous | first | second (받은 값: {phase!r})")


def compare_resume(continuous: dict[str, Any], split: dict[str, Any], *, steps: int, mode: str = "t0") -> dict[str, Any]:
    """두 :func:`_snapshot` 을 그 **학습 범위의** 사전 등록 허용 오차로 견준다 — 게이트의 **판정 부분**만 떼어 둔
    것이라 GPU 없이도 시험할 수 있다 (P1 리뷰 1 M3: 이 판정에 시험이 하나도 없었다).

    `mode`의 오차가 :data:`RESUME_TOLERANCES` 에 없으면 ``passed``는 **`None`**(`verdict`
    ``"tolerance-unregistered"``)이고, 잰 값은 그대로 남기되 다른 범위의 오차로 합격·불합격을 선고하지 않는다.
    정수 기준(sampler 위치·뽑힌 단위·step 수·빠진 tensor)은 오차와 무관하므로 등록 여부와 상관없이 판정한다 —
    그것이 깨지면 ``passed``는 `False`다. 참고로 다른 범위의 오차가 무어라 했을지는 ``would_pass_under``에 적는다.
    """
    import torch

    loss_rows = []
    for index, (a, b) in enumerate(zip(continuous["losses"], split["losses"]), start=1):
        loss_rows.append({"step": index, "continuous": a, "split": b, "abs": abs(a - b), "rel": abs(a - b) / max(abs(a), 1e-9)})
    parameters = []
    for name, tensor in continuous["parameters"].items():
        other = split["parameters"].get(name)
        if other is None:
            parameters.append({"tensor": name, "missing": True})
            continue
        difference = (tensor.float() - other.float())
        reference = float(tensor.float().norm())
        parameters.append({
            "tensor": name, "max_abs": float(difference.abs().max()),
            "relative_l2": float(difference.norm()) / max(reference, 1e-12),
            "reference_l2": reference,
            # 자기 norm이 바닥 아래면 상대 기준에서 뺀다 — 그 비는 분모가 정하는 수다 (P3 D, RELATIVE_L2_REFERENCE_FLOOR)
            "relative_l2_judged": bool(reference >= RELATIVE_L2_REFERENCE_FLOOR),
            "bit_identical": bool(torch.equal(tensor, other)),
        })  # fmt: skip
    sampler_equal = continuous["sampler"] == split["sampler"]
    units_equal = continuous["units"] == split["units"]
    worst_loss = max((row["abs"] for row in loss_rows), default=0.0)
    worst_loss_rel = max((row["rel"] for row in loss_rows), default=0.0)
    worst_param = max((row.get("max_abs", 0.0) for row in parameters), default=0.0)
    judged = [row for row in parameters if row.get("relative_l2_judged")]
    worst_rel = max((row.get("relative_l2", 0.0) for row in judged), default=0.0)
    skipped = [row["tensor"] for row in parameters if "relative_l2" in row and not row.get("relative_l2_judged")]
    exact = bool(
        sampler_equal
        and units_equal
        and continuous["optimizer_steps"] == split["optimizer_steps"] == steps
        and continuous["step"] == split["step"] == steps
        and not any(row.get("missing") for row in parameters)
    )

    def _within(tolerance: dict[str, float]) -> bool:
        return bool(
            (worst_loss <= tolerance["loss_abs"] or worst_loss_rel <= tolerance["loss_rel"])
            and worst_param <= tolerance["param_max_abs"]
            and worst_rel <= tolerance["param_rel_l2"]
        )

    registered = RESUME_TOLERANCES.get(mode)
    passed: bool | None
    if not exact:
        passed, verdict = False, "fail"
    elif registered is None:
        passed, verdict = None, "tolerance-unregistered"
    else:
        passed = _within(registered)
        verdict = "pass" if passed else "fail"
    return {
        "check": "resume", "steps": int(steps), "scope": mode,
        "tolerance": dict(registered) if registered is not None else None,
        "tolerance_note": (
            f"fixed in scripts/p1_acceptance.py before the comparison (RESUME_TOLERANCES[{mode!r}])"
            if registered is not None
            else f"no tolerance is pre-registered for scope {mode!r} (RESUME_TOLERANCES); the gate stops instead of borrowing another scope's"
        ),
        "verdict": verdict, "exact_criteria_passed": exact,
        "would_pass_under": {name: (exact and _within(value)) for name, value in sorted(RESUME_TOLERANCES.items())},
        "losses": loss_rows, "worst_loss_abs": worst_loss, "worst_loss_rel": worst_loss_rel,
        "parameters": parameters, "worst_param_max_abs": worst_param, "worst_param_relative_l2": worst_rel,
        "relative_l2_reference_floor": RELATIVE_L2_REFERENCE_FLOOR,
        "relative_l2_not_judged": skipped,  # 자기 norm이 바닥 아래라 **절대 기준으로만** 본 tensor들
        "bit_identical_tensors": sum(1 for row in parameters if row.get("bit_identical")), "tensors": len(parameters),
        "sampler_position_equal": sampler_equal, "drawn_units_equal": units_equal,
        "optimizer_steps": {"continuous": continuous["optimizer_steps"], "split": split["optimizer_steps"]},
        "passed": passed,
    }


# --------------------------------------------------------------------------
# D — 재개 허용 오차의 **사전 등록**: 재시작 없는 두 run의 퍼짐을 먼저 잰다
# --------------------------------------------------------------------------


def _round_up_one_significant_figure(value: float) -> float:
    """0.0876 → 0.09, 0.0006 → 0.0006, 0.0 → 0.0. 규칙이 정한 반올림 (:data:`RESUME_TOLERANCE_RULE`)."""
    import math

    if value <= 0:
        return 0.0
    exponent = math.floor(math.log10(value))
    scale = 10.0 ** exponent
    return float(f"{math.ceil(value / scale) * scale:.10g}")


def measure_spread(first: dict[str, Any], second: dict[str, Any]) -> dict[str, Any]:
    """**판정 없이** 두 snapshot의 run 간 퍼짐만 잰다 — :func:`compare_resume` 과 같은 계산, 같은 분모 가드.

    사전 등록은 순서가 전부다: 이 수를 먼저 적고, 그 위에 오차를 고정하고, 그 다음에 판정한다. 그래서 이 함수는
    `passed`를 돌려주지 않는다."""
    measured = compare_resume(first, second, steps=int(second.get("step") or 0), mode="__spread__")
    return {
        "unit": "two runs of the same config, same seed, **no restart** (two separate processes)",
        "steps": measured["steps"],
        "losses": measured["losses"],
        "worst_loss_abs": measured["worst_loss_abs"], "worst_loss_rel": measured["worst_loss_rel"],
        "worst_param_max_abs": measured["worst_param_max_abs"], "worst_param_relative_l2": measured["worst_param_relative_l2"],
        "relative_l2_reference_floor": measured["relative_l2_reference_floor"],
        "relative_l2_not_judged": measured["relative_l2_not_judged"],
        "bit_identical_tensors": measured["bit_identical_tensors"], "tensors": measured["tensors"],
        "sampler_position_equal": measured["sampler_position_equal"], "drawn_units_equal": measured["drawn_units_equal"],
        "optimizer_steps": measured["optimizer_steps"],
        "parameters_worst": sorted(measured["parameters"], key=lambda row: -row.get("max_abs", 0.0))[:8],
    }


#: 기준 이름 → 퍼짐 기록의 자리.
_SPREAD_KEYS = (
    ("loss_abs", "worst_loss_abs"), ("loss_rel", "worst_loss_rel"),
    ("param_max_abs", "worst_param_max_abs"), ("param_rel_l2", "worst_param_relative_l2"),
)


def worst_of_pairs(spreads: Sequence[dict[str, Any]]) -> dict[str, float]:
    """잰 쌍 전부에서 **기준마다 최악값**을 모은다 (R2 A1의 규칙 개정).

    어떤 쌍이 그 기준을 재지 못했으면(`None`) 그 쌍은 그 기준에 기여하지 않는다 — 못 잰 것을 0으로 세면 오차가
    조여지고, 그것이 사전 등록이 막으려는 방향이다."""
    return {
        name: max([float(spread[name]) for spread in spreads if spread.get(name) is not None], default=0.0)
        for _, name in _SPREAD_KEYS
    }


def derive_tolerance(spread: dict[str, Any] | Sequence[dict[str, Any]], *, floor: dict[str, float] = RESUME_TOLERANCE) -> dict[str, float]:
    """퍼짐(쌍 하나 또는 쌍 목록) → 허용 오차, :data:`RESUME_TOLERANCE_RULE` 그대로.

    쌍 전부의 최악값 → 안전 계수 → 유효숫자 한 자리 올림 → T0보다 느슨하게만."""
    worst = worst_of_pairs([spread] if isinstance(spread, dict) else list(spread))
    return {
        key: max(float(floor[key]), _round_up_one_significant_figure(TOLERANCE_SAFETY_FACTOR * worst[name]))
        for key, name in _SPREAD_KEYS
    }


def check_baseline(mode: str, *, steps: int = 5, run_dir: Path, config_path: Path = DEFAULT_CONFIG, python: str | None = None) -> dict[str, Any]:
    """그 학습 범위의 **두 run 기준선** — 같은 설정·seed로 재시작 없이 두 번 돌려 퍼짐을 적는다 (P3 D).

    두 run은 진짜로 다른 프로세스다(이 스크립트를 `--phase continuous`로 두 번 띄운다). 그래야 kernel 선택·
    atomics 순서 같은 프로세스 간 차이가 그대로 들어온다 — 재개 게이트가 견뎌야 하는 잡음이 바로 그것이다."""
    import torch

    python = python or sys.executable
    started = time.perf_counter()
    phases = []
    snapshots = []
    for index in ("a", "b"):
        directory = run_dir / f"baseline-{mode}-{index}"
        phase_started = time.perf_counter()
        command = [python, str(REPO / "scripts" / "p1_acceptance.py"), "--phase", "continuous", "--steps", str(steps),
                   "--run-dir", str(directory), "--config", str(config_path), "--resume-modes", mode]
        completed = subprocess.run(command, cwd=REPO, capture_output=True, text=True, check=False)
        print(completed.stdout[-2000:], file=sys.stderr, flush=True)
        if completed.returncode != 0:
            return {"check": "baseline", "scope": mode, "steps": steps, "failed_run": index, "stderr": completed.stderr[-4000:], "phases": phases}
        phases.append({"run": index, "seconds": round(time.perf_counter() - phase_started, 1), "stdout_tail": completed.stdout.strip().splitlines()[-1:]})
        snapshots.append(torch.load(directory / f"continuous-{mode}.pt", map_location="cpu", weights_only=False))

    spread = measure_spread(snapshots[0], snapshots[1])
    prior = [dict(pair) for pair in PRIOR_BASELINE_PAIRS.get(mode, [])]
    measured = {**spread, "label": f"R2 ({dt.date.today().isoformat()}) — `--check baseline` 두 프로세스, {steps} step",
                "source": "this report (checks.baseline_%s.spread)" % mode,
                "config": _repo_relative(config_path)}  # fmt: skip
    pairs = [*prior, measured]
    return {
        "check": "baseline", "scope": mode, "rule": RESUME_TOLERANCE_RULE,
        "safety_factor": TOLERANCE_SAFETY_FACTOR,
        "spread": spread,
        # **이 쌍 하나가 아니라 잰 쌍 전부**가 오차를 만든다 (R2 A1). 앞 쌍의 출처는 값 옆에 적혀 있다.
        "pairs": [{key: pair.get(key) for key in ("label", "source", "config", "steps", "unit", *(name for _, name in _SPREAD_KEYS))} for pair in pairs],
        "pairs_measured": len(pairs), "pairs_required": MINIMUM_BASELINE_PAIRS,
        "enough_pairs": len(pairs) >= MINIMUM_BASELINE_PAIRS,
        "worst_over_pairs": worst_of_pairs(pairs),
        "derived_tolerance_this_pair_only": derive_tolerance(spread),
        "derived_tolerance": derive_tolerance(pairs),
        "registered_tolerance": dict(RESUME_TOLERANCES[mode]) if mode in RESUME_TOLERANCES else None,
        "seconds": round(time.perf_counter() - started, 1), "phases": phases,
    }


def check_resume(*, steps: int = 20, run_dir: Path, config_path: Path = DEFAULT_CONFIG, python: str | None = None, mode: str = "t0") -> dict[str, Any]:
    import torch

    python = python or sys.executable
    started = time.perf_counter()
    phases = []
    for phase in ("continuous", "first", "second"):
        phase_started = time.perf_counter()
        command = [python, str(REPO / "scripts" / "p1_acceptance.py"), "--phase", phase, "--steps", str(steps), "--run-dir", str(run_dir), "--config", str(config_path), "--resume-modes", mode]
        completed = subprocess.run(command, cwd=REPO, capture_output=True, text=True, check=False)
        print(completed.stdout[-2000:], file=sys.stderr, flush=True)
        if completed.returncode != 0:
            return {"check": "resume", "mode": mode, "scope": mode, "verdict": "fail", "passed": False, "failed_phase": phase, "stderr": completed.stderr[-4000:], "phases": phases}
        phases.append({"phase": phase, "seconds": round(time.perf_counter() - phase_started, 1), "stdout_tail": completed.stdout.strip().splitlines()[-1:]})

    continuous = torch.load(run_dir / f"continuous-{mode}.pt", map_location="cpu", weights_only=False)
    split = torch.load(run_dir / f"split-{mode}.pt", map_location="cpu", weights_only=False)
    return {**compare_resume(continuous, split, steps=steps, mode=mode), "mode": mode,
            "seconds": round(time.perf_counter() - started, 1), "phases": phases}  # fmt: skip


#: 학습 범위 → 보고서의 `checks` 자리 이름. `t0`만 옛 이름(`resume`)을 쓴다 — P1의 런처가 그 키를 본다.
def resume_gate_key(mode: str) -> str:
    return "resume" if mode == "t0" else f"resume_{mode}"


def resume_gate(report_path: Path | str, mode: str) -> dict[str, Any]:
    """**띄우려는 학습 범위의** 재개 결과를 읽어 긴 run을 시작해도 되는지 판정한다 (P2 리뷰 1 focus 5 / N4).

    P1의 유일한 자동 게이트(`artifacts/scratch/p1/run_stage_d1_gated.sh`)는 `checks.resume`, 곧 **T0의 자리**를
    읽었다. P2는 T1의 결과를 `checks.resume_t1`에 적었지만 그것을 읽는 것이 아무것도 없었으므로, 긴 T1 run은
    사실상 T0의 결과로 통과되고 있었다. 여기서는 자리를 범위로 고르고, **없거나·판정이 없거나·다른 범위의 결과가
    적혀 있으면 멈춘다**(통과가 기본값이 되지 않게).

    종료 코드: 0 통과 · 2 불합격 · 3 판정 없음(자리 없음·오차 미등록·범위 불일치).
    """
    path = Path(report_path)
    key = resume_gate_key(mode)
    if not path.is_file():
        return {"mode": mode, "key": key, "passed": None, "verdict": "missing-report", "exit_code": 3,
                "reason": f"{path}: 인수 검사 보고서가 없다 — {key}를 읽을 수 없다"}  # fmt: skip
    checks = (json.loads(path.read_text(encoding="utf-8")) or {}).get("checks") or {}
    check = checks.get(key)
    if not isinstance(check, dict):
        return {"mode": mode, "key": key, "passed": None, "verdict": "missing-check", "exit_code": 3,
                "reason": f"{path}: checks.{key}가 없다 — {mode} 범위의 재개는 아직 재지 않았다 (있는 자리: {sorted(checks)})"}  # fmt: skip
    # P1의 보고서에는 `mode`가 없다 — 그 스크립트는 `checks.resume` 자리에 **T0만** 적었으므로 그 조합은 t0로 읽는다.
    recorded = check.get("scope") or check.get("mode") or ("t0" if key == "resume" else None)
    if recorded != mode:
        return {"mode": mode, "key": key, "passed": None, "verdict": "scope-mismatch", "exit_code": 3,
                "reason": f"{path}: checks.{key}에 적힌 mode가 {recorded!r}이다 — {mode!r} 범위의 결과가 아니다"}  # fmt: skip
    passed = check.get("passed")
    if passed is None:
        return {"mode": mode, "key": key, "passed": None, "verdict": check.get("verdict") or "undecided", "exit_code": 3,
                "reason": f"{path}: checks.{key}에 판정이 없다 ({check.get('verdict')}) — {mode}의 허용 오차가 아직 사전 등록되지 않았다 "
                          f"(RESUME_TOLERANCES). 두 run 기준선을 먼저 재고 그 위에 오차를 고정한다"}  # fmt: skip
    if not passed:
        return {"mode": mode, "key": key, "passed": False, "verdict": check.get("verdict") or "fail", "exit_code": 2,
                "reason": f"{path}: checks.{key}.passed = false — 긴 {mode} run을 시작하지 않는다"}  # fmt: skip
    return {"mode": mode, "key": key, "passed": True, "verdict": check.get("verdict") or "pass", "exit_code": 0,
            "reason": f"{path}: checks.{key}.passed = true (steps {check.get('steps')})"}  # fmt: skip


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", default="frozen,trains,resume", help="frozen | trains | resume | baseline, 쉼표로")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--steps", type=int, default=20, help="resume: 연속 실행의 step 수 (절반에서 저장·재시작)")
    parser.add_argument("--train-steps", dest="train_steps", type=int, default=3, help="trains: LoRA·T1의 step 수")
    parser.add_argument("--trains-modes", dest="trains_modes", default="lora,t1")
    parser.add_argument("--resume-modes", dest="resume_modes", default="t0", help="resume: 어느 학습 범위에서 게이트를 돌릴지 (t0 | lora | t1, 쉼표로) — P1은 t0만 덮었다")
    parser.add_argument("--run-dir", dest="run_dir", default=str(REPO / "artifacts" / "runs" / "p1-acceptance"))
    parser.add_argument("--phase", default=None, help="내부용 — C3의 조각을 다시 띄울 때")
    parser.add_argument("--out", default=str(REPO / "artifacts" / "reports" / "p1-acceptance.json"))
    parser.add_argument("--gpu-memory-fraction", dest="gpu_memory_fraction", type=float, default=DEFAULT_FRACTION)
    parser.add_argument("--gate", default=None, help="긴 run을 띄우기 전의 관문 — 이 학습 범위(t0 | lora | t1)의 재개 결과만 읽고 종료 코드로 답한다 (GPU를 쓰지 않는다)")
    parser.add_argument("--gate-report", dest="gate_report", default=str(REPO / "artifacts" / "reports" / "p1-acceptance.json"))
    args = parser.parse_args(argv)
    if args.gate:
        gate = resume_gate(args.gate_report, args.gate)
        print(json.dumps(gate, ensure_ascii=False))
        return int(gate["exit_code"])
    guard = limit_gpu_memory(args.gpu_memory_fraction)
    print(f"[p1] gpu guard {guard} · memory {memory_report()}", file=sys.stderr, flush=True)
    run_dir = Path(args.run_dir)
    if args.phase:
        modes = [name.strip() for name in args.resume_modes.split(",") if name.strip()]
        if len(modes) != 1:
            parser.error(f"--phase와 함께 쓰는 --resume-modes는 하나여야 한다 (받은 값: {args.resume_modes!r})")
        result = run_resume_phase(args.phase, steps=args.steps, run_dir=run_dir, config_path=Path(args.config), mode=modes[0])
        print(json.dumps(result, ensure_ascii=False))
        return 0
    wanted = [name.strip() for name in args.check.split(",") if name.strip()]
    unknown = [name for name in wanted if name not in CHECKS]
    if unknown:
        parser.error(f"--check: 알 수 없는 검사 {unknown} (있는 것: {list(CHECKS)})")
    out: dict[str, Any] = {
        "task": "p1-acceptance", "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "config": _repo_relative(args.config),
        "gpu": {"guard": guard, "memory_at_start": memory_report()}, "checks": {},
    }
    existing = Path(args.out)
    if existing.is_file():
        out["checks"] = json.loads(existing.read_text(encoding="utf-8")).get("checks", {})
    for name in wanted:
        started = time.perf_counter()
        if name == "frozen":
            out["checks"]["frozen"] = check_frozen(config_path=Path(args.config))
        elif name == "trains":
            for mode in (m.strip() for m in args.trains_modes.split(",") if m.strip()):
                out["checks"][f"trains_{mode}"] = check_trains(mode, steps=args.train_steps, config_path=Path(args.config))
        elif name == "baseline":
            for mode in (m.strip() for m in args.resume_modes.split(",") if m.strip()):
                out["checks"][f"baseline_{mode}"] = check_baseline(mode, steps=args.steps, run_dir=run_dir, config_path=Path(args.config))
        else:
            for mode in (m.strip() for m in args.resume_modes.split(",") if m.strip()):
                # t0의 자리 이름은 그대로 둔다 — P1의 런처가 `checks.resume.passed`를 본다. 다른 범위는
                # `resume_<mode>`이고, 그 자리를 읽는 것이 :func:`resume_gate`다 (P2 리뷰 1 N4).
                key = resume_gate_key(mode)
                out["checks"][key] = check_resume(steps=args.steps, run_dir=run_dir, config_path=Path(args.config), mode=mode)
        print(f"[p1] {name}: {round(time.perf_counter() - started, 1)} s", file=sys.stderr, flush=True)
        target = Path(args.out)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps({**out, "gpu": {**out["gpu"], "memory_at_end": memory_report()}}, ensure_ascii=False, indent=1, default=str) + "\n", encoding="utf-8")
    print(json.dumps({name: check.get("passed") for name, check in out["checks"].items()}, ensure_ascii=False))
    # 기준선은 판정이 아니라 측정이다 — `passed`가 없다고 실패로 세지 않는다
    verdicts = [check.get("passed") for name, check in out["checks"].items() if not name.startswith("baseline_")]
    return 0 if all(verdicts) else 1


if __name__ == "__main__":
    sys.exit(main())
