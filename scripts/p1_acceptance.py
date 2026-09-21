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
from pathlib import Path
from typing import Any

from robo_jev.gpu import DEFAULT_FRACTION, limit_gpu_memory, memory_report, require_free

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

CHECKS = ("frozen", "trains", "resume")
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
#: T1·LoRA에서 "움직였다"고 보는 최소 변화 — 이보다 작으면 업데이트가 빠진 것이다.
MOVE_EPSILON = 1e-9


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


def compare_resume(continuous: dict[str, Any], split: dict[str, Any], *, steps: int) -> dict[str, Any]:
    """두 :func:`_snapshot` 을 :data:`RESUME_TOLERANCE` 로 견준다 — 게이트의 **판정 부분**만 떼어 둔 것이라
    GPU 없이도 시험할 수 있다 (P1 리뷰 1 M3: 이 판정에 시험이 하나도 없었다)."""
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
        parameters.append({
            "tensor": name, "max_abs": float(difference.abs().max()),
            "relative_l2": float(difference.norm() / tensor.float().norm().clamp_min(1e-12)),
            "bit_identical": bool(torch.equal(tensor, other)),
        })  # fmt: skip
    sampler_equal = continuous["sampler"] == split["sampler"]
    units_equal = continuous["units"] == split["units"]
    worst_loss = max((row["abs"] for row in loss_rows), default=0.0)
    worst_loss_rel = max((row["rel"] for row in loss_rows), default=0.0)
    worst_param = max((row.get("max_abs", 0.0) for row in parameters), default=0.0)
    worst_rel = max((row.get("relative_l2", 0.0) for row in parameters), default=0.0)
    passed = bool(
        sampler_equal
        and units_equal
        and continuous["optimizer_steps"] == split["optimizer_steps"] == steps
        and continuous["step"] == split["step"] == steps
        and not any(row.get("missing") for row in parameters)
        and (worst_loss <= RESUME_TOLERANCE["loss_abs"] or worst_loss_rel <= RESUME_TOLERANCE["loss_rel"])
        and worst_param <= RESUME_TOLERANCE["param_max_abs"]
        and worst_rel <= RESUME_TOLERANCE["param_rel_l2"]
    )
    return {
        "check": "resume", "steps": int(steps),
        "tolerance": dict(RESUME_TOLERANCE), "tolerance_note": "fixed in scripts/p1_acceptance.py before the comparison (RESUME_TOLERANCE)",
        "losses": loss_rows, "worst_loss_abs": worst_loss, "worst_loss_rel": worst_loss_rel,
        "parameters": parameters, "worst_param_max_abs": worst_param, "worst_param_relative_l2": worst_rel,
        "bit_identical_tensors": sum(1 for row in parameters if row.get("bit_identical")), "tensors": len(parameters),
        "sampler_position_equal": sampler_equal, "drawn_units_equal": units_equal,
        "optimizer_steps": {"continuous": continuous["optimizer_steps"], "split": split["optimizer_steps"]},
        "passed": passed,
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
            return {"check": "resume", "mode": mode, "passed": False, "failed_phase": phase, "stderr": completed.stderr[-4000:], "phases": phases}
        phases.append({"phase": phase, "seconds": round(time.perf_counter() - phase_started, 1), "stdout_tail": completed.stdout.strip().splitlines()[-1:]})

    continuous = torch.load(run_dir / f"continuous-{mode}.pt", map_location="cpu", weights_only=False)
    split = torch.load(run_dir / f"split-{mode}.pt", map_location="cpu", weights_only=False)
    return {**compare_resume(continuous, split, steps=steps), "mode": mode,
            "seconds": round(time.perf_counter() - started, 1), "phases": phases}  # fmt: skip


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", default="frozen,trains,resume", help="frozen | trains | resume, 쉼표로")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--steps", type=int, default=20, help="resume: 연속 실행의 step 수 (절반에서 저장·재시작)")
    parser.add_argument("--train-steps", dest="train_steps", type=int, default=3, help="trains: LoRA·T1의 step 수")
    parser.add_argument("--trains-modes", dest="trains_modes", default="lora,t1")
    parser.add_argument("--resume-modes", dest="resume_modes", default="t0", help="resume: 어느 학습 범위에서 게이트를 돌릴지 (t0 | lora | t1, 쉼표로) — P1은 t0만 덮었다")
    parser.add_argument("--run-dir", dest="run_dir", default=str(REPO / "artifacts" / "runs" / "p1-acceptance"))
    parser.add_argument("--phase", default=None, help="내부용 — C3의 조각을 다시 띄울 때")
    parser.add_argument("--out", default=str(REPO / "artifacts" / "reports" / "p1-acceptance.json"))
    parser.add_argument("--gpu-memory-fraction", dest="gpu_memory_fraction", type=float, default=DEFAULT_FRACTION)
    args = parser.parse_args(argv)
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
        "config": str(Path(args.config).relative_to(REPO)) if str(Path(args.config).resolve()).startswith(str(REPO)) else args.config,
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
        else:
            for mode in (m.strip() for m in args.resume_modes.split(",") if m.strip()):
                # t0의 자리 이름은 그대로 둔다 — stage D의 관문(`run_stage_d1_gated.sh`)이 `checks.resume.passed`를 본다
                key = "resume" if mode == "t0" else f"resume_{mode}"
                out["checks"][key] = check_resume(steps=args.steps, run_dir=run_dir, config_path=Path(args.config), mode=mode)
        print(f"[p1] {name}: {round(time.perf_counter() - started, 1)} s", file=sys.stderr, flush=True)
        target = Path(args.out)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps({**out, "gpu": {**out["gpu"], "memory_at_end": memory_report()}}, ensure_ascii=False, indent=1, default=str) + "\n", encoding="utf-8")
    print(json.dumps({name: check.get("passed") for name, check in out["checks"].items()}, ensure_ascii=False))
    return 0 if all(check.get("passed") for check in out["checks"].values()) else 1


if __name__ == "__main__":
    sys.exit(main())
