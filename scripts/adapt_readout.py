"""Task 2b G0b S3 — 실제 backbone의 readout-only 적응(T0), 짧은 LoRA, 무학습 라벨 점수, 학습 구간 메모리 (DGX Spark, BF16).

docs/06 Task 2b의 인터페이스 ``adapt_readout(model_id, records, steps) -> dict``: 레코드 묶음으로 readout(U·V·b)만 짧게
학습한 뒤 평가 분할의 품질(질문별 정확도·허용 집합 적중·NLL·Brier, 위치 편향, 문맥 섞기·규칙 기준군)과 학습 구간
메모리·step 시간을 돌려준다. 학습은 :mod:`robo_jev.train` 의 Trainer(혼합 sampler, TBPTT, 후보 순서 치환 증강)
그대로이고, 평가는 :mod:`robo_jev.evaluate`, 무학습 점수는 :mod:`robo_jev.model.zero_shot` 다.

모드(``--mode``):

* ``t0`` — readout-only (backbone 고정 BF16, readout fp32, 정적 윈도우 KV). 데이터 = D0(train) + batch-0 train + pilot train.
* ``lora`` — peft LoRA(r=16, α=32, attention·MLP projection, lr 5e-5) + readout, 같은 데이터, `--steps` step(≈1 epoch은 전체
  pilot train 단위 수 ≈ 50 step). KV는 그래프를 유지하는 dynamic 모드, 구간은 5초(활성 메모리).
* ``zero-shot`` — 학습 없이 Nimble 방식 코드 토큰 점수 (평가 분할만).
* ``chunk-memory`` — 1→2→5→10초 구간 forward+backward의 peak 메모리·시간 — readout-only, full(backbone 전체 gradient, 층 단위
  activation checkpointing), full_nockpt(checkpointing 없이, 2초까지 — 45K 토큰은 ≈240 GB라 통합 메모리를 넘긴다). 매 실행 전에
  여유 메모리를 보고 total의 20 % 미만이면 멈춘다(`stopped_at`).

평가 분할: batch-0 `dev`(5편)·`ood_dev`(1편) — 작다(브리프가 말한 대로 적는다) — 와 pilot `dev`·`ood_dev`, D0 `dev`.

실행: `uv run python scripts/adapt_readout.py --model Qwen/Qwen3.5-2B --mode t0 --steps 150 --out artifacts/reports/adapt-2b-t0.json`
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any

from robo_jev.gpu import DEFAULT_FRACTION, limit_gpu_memory, memory_report, require_free

REPO = Path(__file__).resolve().parents[1]
D0_MANIFEST = REPO / "tests" / "fixtures" / "d0_manifest.json"
BATCH0_MANIFEST = REPO / "artifacts" / "datasets" / "d1-robot" / "batch-0" / "manifest.json"
PILOT_MANIFEST = REPO / "artifacts" / "datasets" / "pilot" / "manifest.json"
MODES = ("t0", "lora", "zero-shot", "chunk-memory")
EVAL_SPLITS = (("batch0", "dev"), ("batch0", "ood_dev"), ("pilot", "dev"), ("pilot", "ood_dev"), ("d0", "dev"))
MANIFESTS = {"d0": D0_MANIFEST, "batch0": BATCH0_MANIFEST, "pilot": PILOT_MANIFEST}
DOMAINS = {"d0": None, "batch0": "robot", "pilot": "non_robot"}
#: batch-0 manifest에는 에피소드 스트림 말고 대조 단일 요청(`contrast/records.jsonl`)도 있다 — 브리프의 데이터 mix(batch-0 train 에피소드)만 읽는다.
FILES = {"d0": None, "batch0": ["episodes/*/streams.jsonl"], "pilot": None}
#: 로봇/비로봇 축의 태그 — pilot의 `provenance.domain`은 분야 이름이라 쓸 수 없다 (학습 설정 `sampler.domain_tag`와 같은 값)
DOMAIN_TAG = "provenance.robojev_domain"
#: LoRA 학습 전에 있어야 하는 장치(통합) 메모리 여유 — 4B 5초 full 구간 peak 58 GiB + 여유
LORA_MIN_FREE_BYTES = 60 * 2**30


def _tokenizer_id() -> str:
    from robo_jev.model.tokenizer import available_tokenizer

    found = available_tokenizer()
    if found is None:
        raise FileNotFoundError("실제 tokenizer가 없다 — `uv run python scripts/fetch_tokenizer.py`")
    return found[0]


def train_config(model_id: str, *, mode: str, steps: int, run_id: str, seed: int = 17, chunk_seconds: float | None = None, lr: float | None = None, stop_after: int | None = None) -> dict[str, Any]:
    """Trainer 설정 (T0 / LoRA)."""
    lora = mode == "lora"
    return {
        "run_name": f"g0b-{mode}", "run_id": run_id, "model_id": model_id, "dtype": "bfloat16", "device": "cuda",
        "trainable": "lora_and_readout" if lora else "readout_only", "readout_rank": 64, "readout_dtype": "float32",
        "lora": {"r": 16, "alpha": 32, "dropout": 0.0} if lora else None,
        # LoRA는 backbone 전체의 activation을 backward까지 들고 있어야 한다 — 층 단위 checkpointing 없이는 5초 구간(≈22K 토큰)도 GB10 메모리를 넘긴다
        "activation_checkpointing": lora,
        "dataset_manifests": [{"path": str(MANIFESTS[name]), "domain": DOMAINS[name], "files": FILES[name]} for name in ("d0", "batch0", "pilot")],
        "splits": ["train"], "tokenizer": _tokenizer_id(), "stream_chunk_seconds": chunk_seconds if chunk_seconds is not None else (5 if lora else 10),
        "stream_window_ticks": 30, "gradient_accumulation": 2, "robot_loss_share": 0.6, "nonrobot_tokens_per_unit": 8192,
        "backbone_lr": 5e-5, "readout_lr": lr if lr is not None else 1e-4, "weight_decay": 0.01, "gradient_clip": 1.0, "warmup_ratio": 0.05,
        "max_steps": int(steps), "checkpoint_every": int(steps), "seed": seed, "torch_threads": 8,
        # lr 탐침: 스케줄(warmup·cosine)은 `steps` 기준 그대로 두고 `stop_after` step 뒤 멈춘다
        "stop_after": {"step": int(stop_after)} if stop_after else None,
        "artifacts_dir": str(REPO / "artifacts" / "runs"),
        # pilot 레코드의 `provenance.domain`은 분야 이름(spatial·dom·…)이라 로봇/비로봇 축의 태그로 쓸 수 없다 — manifest의 domain 기본값을 쓴다
        "sampler": {"permute_candidates_seed": seed, "domain_tag": DOMAIN_TAG},
    }


def load_eval_items(tokenizer: Any, *, splits: tuple[tuple[str, str], ...] = EVAL_SPLITS) -> dict[str, list[Any]]:
    from robo_jev.sampler import load_items

    out: dict[str, list[Any]] = {}
    for name, split in splits:
        path = MANIFESTS[name]
        if not path.is_file():
            continue
        # pilot 레코드의 `provenance.domain`은 분야 이름(spatial·dom·…) — 학습 설정과 같은 태그(`provenance.robojev_domain`)로 읽는다
        items = load_items(path, tokenizer=tokenizer, splits=(split,), domain=DOMAINS[name], files=FILES[name], domain_tag=DOMAIN_TAG)
        if items:
            out[f"{name}/{split}"] = items
    return out


def evaluate_judge(judge: Any, tokenizer: Any, *, shuffle_seed: int = 1, splits: tuple[tuple[str, str], ...] = EVAL_SPLITS) -> dict[str, Any]:
    """평가 분할마다 :func:`robo_jev.evaluate.evaluate_items` 의 표."""
    from robo_jev.evaluate import evaluate_items

    tables: dict[str, Any] = {}
    for key, items in load_eval_items(tokenizer, splits=splits).items():
        started = time.perf_counter()
        tables[key] = evaluate_items(judge, items, tokenizer=tokenizer, shuffle_seed=shuffle_seed)
        tables[key]["seconds"] = round(time.perf_counter() - started, 1)
        print(f"[G0b] eval {key}: {tables[key]['n_states']} states, acc {tables[key]['model']['_all']['accuracy']}, "
              f"nll {tables[key]['model']['_all']['nll']:.3f}, change {tables[key]['answer_change']['rate']}, "
              f"shuffle-ctrl acc {tables[key]['context_shuffle']['_all']['accuracy']} ({tables[key]['seconds']} s)", file=sys.stderr, flush=True)
    return tables


def _memory() -> dict[str, Any]:
    import torch

    stats = torch.cuda.memory_stats()
    return {
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()), "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
        "num_alloc_retries": int(stats.get("num_alloc_retries", 0)), "num_ooms": int(stats.get("num_ooms", 0)),
    }


def run_training(model_id: str, *, mode: str, steps: int, seed: int = 17, eval_after: bool = True, lr: float | None = None, stop_after: int | None = None) -> dict[str, Any]:
    """T0 또는 LoRA 학습 + 평가. 돌려주는 것: 곡선, 메모리, step 시간, checkpoint, 평가 표."""
    import torch

    from robo_jev.train import Trainer

    torch.cuda.reset_peak_memory_stats()
    if mode == "lora":
        # 앞 프로세스가 아직 메모리를 놓지 않았으면 OOM 대신 분명한 오류로 (chunk-memory: 2B 5초 full 25 GiB, 4B 58 GiB)
        require_free(LORA_MIN_FREE_BYTES, what=f"LoRA {model_id}")
    run_id = f"g0b-{mode}-{model_id.split('/')[-1].lower()}-{dt.datetime.now().strftime('%Y%m%d-%H%M%S')}"
    config = train_config(model_id, mode=mode, steps=steps, run_id=run_id, seed=seed, lr=lr, stop_after=stop_after)
    started = time.perf_counter()
    with Trainer(config) as trainer:
        load_seconds = round(time.perf_counter() - started, 1)
        result = trainer.run()
        tokenizer = trainer.tokenizer
        curve = [
            {"step": m["step"], "loss": m["loss"], "loss_by_domain": m["loss_by_domain"], "loss_by_type": m["loss_by_type"], "grad_norm": m["grad_norm"], "lr": m["lr"], "tokens": m["tokens"]["total"], "seconds": m["seconds"]}
            for m in result["metrics"]["steps"]
        ]
        memory = _memory()
        seconds = [m["seconds"] for m in result["metrics"]["steps"]]
        out = {
            "model_id": model_id, "mode": mode, "steps": int(steps), "run_id": result["run_id"], "checkpoint": result["checkpoint"], "status": result["status"],
            "config": {key: value for key, value in trainer.config.items() if key != "dataset_manifests"},
            "manifest": {key: trainer.manifest[key] for key in ("model", "contract_sha256", "serializer_version", "tokenizer")},
            "items": {"total": len(trainer.items), "stream": sum(i.kind == "stream" for i in trainer.items), "single": sum(i.kind == "single" for i in trainer.items)},
            "load_seconds": load_seconds, "train_seconds": round(sum(seconds), 1),
            "step_seconds": {"mean": round(statistics.fmean(seconds), 2), "p50": round(statistics.median(seconds), 2), "max": round(max(seconds), 2)},
            "memory": memory, "curve": curve,
            "loss_first_last": [curve[0]["loss"], curve[-1]["loss"]] if curve else None,
        }
        if eval_after:
            out["evaluation"] = evaluate_judge(trainer.model, tokenizer)
    return out


def evaluate_checkpoint(model_id: str, *, mode: str, checkpoint: str, steps: int, seed: int = 17, lr: float | None = None) -> dict[str, Any]:
    """이미 끝난 run의 checkpoint(readout(+LoRA))를 같은 설정의 모델에 싣고 **평가만** 한다 — 학습 뒤 평가가 실패했을 때 다시 학습하지
    않으려고. 곡선·step 시간은 run 디렉터리의 `metrics.json`에서 읽는다; `memory`는 평가 프로세스의 peak다(학습 peak가 아니다)."""
    import torch

    from robo_jev.train import build_model, build_tokenizer, load_readout_checkpoint, resolve_config

    path = Path(checkpoint)
    run_dir = path.parent
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8")) if (run_dir / "metrics.json").is_file() else {}
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    config = resolve_config(train_config(model_id, mode=mode, steps=steps, run_id=run_dir.name, seed=seed, lr=lr))
    judge = build_model(config)
    tokenizer = build_tokenizer(config["tokenizer"])
    manifest = load_readout_checkpoint(judge, path)
    judge.eval()
    load_seconds = round(time.perf_counter() - started, 1)
    steps_done = metrics.get("steps") or []
    curve = [
        {"step": m["step"], "loss": m["loss"], "loss_by_domain": m["loss_by_domain"], "loss_by_type": m["loss_by_type"], "grad_norm": m["grad_norm"], "lr": m["lr"], "tokens": m["tokens"]["total"], "seconds": m["seconds"]}
        for m in steps_done
    ]
    seconds = [m["seconds"] for m in steps_done]
    out: dict[str, Any] = {
        "model_id": model_id, "mode": mode, "steps": int(metrics.get("step", steps)), "run_id": metrics.get("run_id", run_dir.name), "checkpoint": str(path), "status": metrics.get("status"),
        "evaluated_from_checkpoint": True,
        "config": {key: value for key, value in config.items() if key != "dataset_manifests"},
        "manifest": {key: manifest.get(key) for key in ("model", "contract_sha256", "serializer_version", "tokenizer")},
        "items": None, "load_seconds": load_seconds, "train_seconds": round(sum(seconds), 1) if seconds else None,
        "step_seconds": {"mean": round(statistics.fmean(seconds), 2), "p50": round(statistics.median(seconds), 2), "max": round(max(seconds), 2)} if seconds else None,
        "memory": {**_memory(), "note": "evaluation process only (the training peak is not in metrics.json)"}, "curve": curve,
        "loss_first_last": [curve[0]["loss"], curve[-1]["loss"]] if curve else None,
    }
    out["evaluation"] = evaluate_judge(judge, tokenizer)
    return out


def adapt_readout(model_id: str, records: list[dict], steps: int, *, eval_records: list[dict] | None = None, mode: str = "t0", seed: int = 17) -> dict[str, Any]:
    """docs/06 인터페이스 — 레코드 묶음(단일 요청·스트림)을 임시 manifest로 써서 readout만 `steps` step 학습하고 `eval_records`로 평가한다."""
    import tempfile

    from robo_jev.evaluate import evaluate_items
    from robo_jev.model.tokenizer import load_tokenizer, sha256_of_file
    from robo_jev.sampler import load_items
    from robo_jev.train import Trainer

    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        for name, rows in (("train.jsonl", records), ("eval.jsonl", eval_records or [])):
            (root / name).write_text("".join(json.dumps({**row, "split": "train" if name == "train.jsonl" else "dev"}, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
        files = {name: {"sha256": sha256_of_file(root / name), "records": len(rows)} for name, rows in (("train.jsonl", records), ("eval.jsonl", eval_records or [])) if rows}
        (root / "manifest.json").write_text(json.dumps({"version": "manifest-v0", "files": files}), encoding="utf-8")
        tokenizer_id = _tokenizer_id()
        config = train_config(model_id, mode=mode, steps=steps, run_id=f"adapt-{dt.datetime.now().strftime('%H%M%S')}", seed=seed)
        config["dataset_manifests"] = [{"path": str(root / "manifest.json")}]
        with Trainer(config) as trainer:
            result = trainer.run()
            out: dict[str, Any] = {"run_id": result["run_id"], "steps": result["step"], "curve": [(m["step"], m["loss"]) for m in result["metrics"]["steps"]], "memory": _memory()}
            if eval_records:
                items = load_items(root / "manifest.json", tokenizer=load_tokenizer(tokenizer_id), splits=("dev",))
                out["evaluation"] = evaluate_items(trainer.model, items, tokenizer=trainer.tokenizer)
    return out


def run_zero_shot(model_id: str, *, tick_stride: int = 8, shuffle_seed: int = 1, batch: int = 8) -> dict[str, Any]:
    import torch

    from robo_jev.model.backbone_qwen import QwenBackbone
    from robo_jev.model.tokenizer import load_tokenizer
    from robo_jev.model.zero_shot import zero_shot_label_scores

    tokenizer = load_tokenizer(_tokenizer_id())
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    backbone = QwenBackbone.load(model_id)
    out: dict[str, Any] = {"model_id": model_id, "mode": "zero-shot", "load_seconds": round(time.perf_counter() - started, 1), "splits": {}}
    for key, items in load_eval_items(tokenizer).items():
        started = time.perf_counter()
        records = [item.record for item in items]
        result = zero_shot_label_scores(model_id, records, backbone=backbone, tokenizer=tokenizer, shuffle_seed=shuffle_seed, tick_stride=tick_stride, batch=batch)
        result["seconds"] = round(time.perf_counter() - started, 1)
        out["splits"][key] = result
        print(f"[G0b] zero-shot {key}: {result['prompts']} prompts, acc {result['table']['_all']['accuracy']}, nll {result['table']['_all']['nll']:.3f} ({result['seconds']} s)", file=sys.stderr, flush=True)
    out["memory"] = _memory()
    return out


CHUNK_SECONDS_LADDER = (1.0, 2.0, 5.0, 10.0)
#: 구간을 늘리기 전에 남아 있어야 하는 장치(통합) 메모리의 몫 — 그보다 적으면 계획을 멈춘다(커널 OOM으로 세션이 죽는 대신).
CHUNK_MIN_FREE_SHARE = 0.2
#: 5초 → 10초 full 구간으로 가려면 5초 실행의 peak reserved가 남긴 여유가 이 몫 이상이어야 한다.
CHUNK_FULL_10S_HEADROOM = 0.4
#: checkpointing 없는 full은 이 길이까지만 (45K 토큰은 ≈240 GB — 2026-09-20의 커널 OOM 원인).
CHUNK_NOCKPT_MAX_SECONDS = 2.0


def run_chunk_memory(model_id: str, *, modes: tuple[str, ...] = ("readout", "full", "full_nockpt"), ladder: tuple[float, ...] = CHUNK_SECONDS_LADDER) -> dict[str, Any]:
    """batch-0에서 가장 긴 에피소드의 첫 구간 forward+backward — readout-only, full(backbone 전체 gradient, 층 단위 activation
    checkpointing), full_nockpt(checkpointing 없이)의 peak 메모리·시간.

    구간을 1 → 2 → 5 → 10초로 늘리며 매 실행 전에 `mem_get_info`의 여유를 보고(total의 20 % 미만이면 멈추고 `stopped_at`과
    이유를 적는다), `full_nockpt`는 2초까지만, 10초 `full`은 5초 실행의 peak reserved가 total의 40 % 이상을 남겼을 때만 간다.
    넘치는 할당은 :func:`robo_jev.gpu.limit_gpu_memory` 의 상한 덕에 프로세스 안의 `OutOfMemoryError`로 끝난다(`ok: False`).
    """
    import gc

    import torch

    from robo_jev.model.backbone_qwen import QwenBackbone
    from robo_jev.model.judge import Judge
    from robo_jev.model.tokenizer import load_tokenizer
    from robo_jev.sampler import load_items
    from robo_jev.train import plan_episode, run_stream_chunk

    tokenizer = load_tokenizer(_tokenizer_id())
    # 에피소드 스트림만 읽는다 — batch-0 manifest의 대조 단일 요청(`contrast/records.jsonl`)에는 `ticks`가 없다
    items = [i for i in load_items(BATCH0_MANIFEST, tokenizer=tokenizer, splits=("train", "dev", "calibration", "test", "ood_dev", "ood_test"), domain="robot", files=FILES["batch0"]) if i.kind == "stream"]
    item = max(items, key=lambda i: len(i.record["ticks"]))
    out: dict[str, Any] = {"model_id": model_id, "mode": "chunk-memory", "episode": item.record_id, "ticks_total": len(item.record["ticks"]), "ladder_seconds": list(ladder), "min_free_share": CHUNK_MIN_FREE_SHARE, "runs": [], "stopped_at": None}
    for mode in modes:
        if mode not in ("readout", "full", "full_nockpt"):
            raise ValueError(f"chunk-memory mode: readout | full | full_nockpt (받은 값: {mode!r})")
        previous: dict[str, Any] | None = None
        for seconds in ladder:
            if mode == "full_nockpt" and seconds > CHUNK_NOCKPT_MAX_SECONDS:
                break
            if previous is not None and not previous.get("ok"):
                out["runs"].append({"mode": mode, "chunk_seconds": seconds, "skipped": f"{previous['chunk_seconds']}s run failed ({previous.get('error', '')[:60]})"})
                break
            gc.collect()
            torch.cuda.empty_cache()
            report = memory_report()
            free_share = report["free_bytes"] / report["total_bytes"]
            if free_share < CHUNK_MIN_FREE_SHARE:
                out["stopped_at"] = {"mode": mode, "chunk_seconds": seconds, "reason": f"free {free_share:.0%} of total < {CHUNK_MIN_FREE_SHARE:.0%}", "memory": report}
                print(f"[G0b] chunk-memory {model_id}: stopped before {mode} {seconds}s — {out['stopped_at']['reason']}", file=sys.stderr, flush=True)
                return out
            if mode == "full" and seconds >= 10.0 and previous is not None:
                headroom = 1.0 - previous["peak_reserved_bytes"] / report["total_bytes"]
                if headroom < CHUNK_FULL_10S_HEADROOM:
                    out["runs"].append({"mode": mode, "chunk_seconds": seconds, "skipped": f"{previous['chunk_seconds']}s run left {headroom:.0%} headroom < {CHUNK_FULL_10S_HEADROOM:.0%}"})
                    print(f"[G0b] chunk-memory {model_id}: skip {mode} {seconds}s — headroom {headroom:.0%}", file=sys.stderr, flush=True)
                    break
            torch.cuda.reset_peak_memory_stats()
            backbone = QwenBackbone.load(model_id, kv_mode="static" if mode == "readout" else "dynamic")
            if mode != "readout":
                backbone.model.requires_grad_(True)
                backbone.activation_checkpointing = mode == "full"
            judge = Judge(backbone, rank=64, readout="pointer", seed=1000, readout_dtype=torch.float32)
            plan = plan_episode(item.record, chunk_seconds=seconds, tick_weights={"steady": 0.25, "event": 2.0, "goal_change": 2.0, "other": 1.0})
            chunk = plan.chunks[0]
            tokens = int(item.layout["prefix_end"]) + sum(int(t["end"]) - int(t["start"]) for t in item.layout["ticks"][chunk[0] : chunk[1]])
            entry: dict[str, Any] = {"mode": mode, "activation_checkpointing": bool(backbone.activation_checkpointing), "chunk_seconds": seconds, "ticks": chunk[1] - chunk[0], "tokens": tokens, "weights_bytes": int(sum(p.numel() * p.element_size() for p in backbone.model.parameters())), "free_before_bytes": report["free_bytes"]}
            result = None
            try:
                torch.cuda.synchronize()
                started = time.perf_counter()
                result = run_stream_chunk(judge, item, chunk, carried=None, plan=plan, scale=1.0 / max(plan.normaliser, 1e-9))
                torch.cuda.synchronize()
                forward = time.perf_counter() - started
                result.loss.backward()
                torch.cuda.synchronize()
                total = time.perf_counter() - started
                grads = sum(p.grad.numel() * p.grad.element_size() for p in judge.parameters() if p.grad is not None)
                entry.update({"ok": True, "loss": result.value, "forward_seconds": round(forward, 2), "forward_backward_seconds": round(total, 2), "gradient_bytes": int(grads)})
            except torch.OutOfMemoryError as exc:
                entry.update({"ok": False, "error": f"OutOfMemoryError: {str(exc)[:200]}"})
            entry.update(_memory())
            out["runs"].append(entry)
            print(f"[G0b] chunk-memory {model_id} {mode} {seconds}s: {entry}", file=sys.stderr, flush=True)
            previous = entry
            del result, judge, backbone
            gc.collect()
            torch.cuda.empty_cache()
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", required=True, help="candidates.yaml의 후보 id")
    parser.add_argument("--mode", default="t0", choices=MODES)
    parser.add_argument("--steps", type=int, default=150)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--lr", type=float, default=None, help="readout lr (기본 1e-4 — 1e-3은 2B에서 한 step 만에 손실이 2→8로 튀었다)")
    parser.add_argument("--stop-after", dest="stop_after", type=int, default=None, help="스케줄은 --steps 기준으로 두고 이 step 뒤 멈춘다 (lr 탐침)")
    parser.add_argument("--tick-stride", dest="tick_stride", type=int, default=8, help="zero-shot: 스트림에서 몇 틱마다 하나를 잴지")
    parser.add_argument("--chunk-modes", dest="chunk_modes", default="readout,full,full_nockpt", help="chunk-memory: readout | full(층 단위 checkpointing) | full_nockpt, 쉼표로 — 구간은 1→2→5→10초 사다리")
    parser.add_argument("--no-eval", dest="no_eval", action="store_true")
    parser.add_argument("--eval-checkpoint", dest="eval_checkpoint", default=None, help="t0/lora: 학습하지 않고 이 checkpoint를 실어 평가만 (run 디렉터리의 metrics.json에서 곡선을 읽는다)")
    parser.add_argument("--out", required=True)
    parser.add_argument("--gpu-memory-fraction", dest="gpu_memory_fraction", type=float, default=DEFAULT_FRACTION, help="프로세스가 쓸 장치(통합) 메모리 몫 (robo_jev.gpu; 첫 CUDA 할당 전에 건다)")
    args = parser.parse_args(argv)
    guard = limit_gpu_memory(args.gpu_memory_fraction)
    memory_start = memory_report()
    print(f"[G0b] gpu guard {guard} · memory at start {memory_start}", file=sys.stderr, flush=True)
    started = time.perf_counter()
    if args.mode in ("t0", "lora") and args.eval_checkpoint:
        result = evaluate_checkpoint(args.model, mode=args.mode, checkpoint=args.eval_checkpoint, steps=args.steps, seed=args.seed, lr=args.lr)
    elif args.mode in ("t0", "lora"):
        result = run_training(args.model, mode=args.mode, steps=args.steps, seed=args.seed, eval_after=not args.no_eval, lr=args.lr, stop_after=args.stop_after)
    elif args.mode == "zero-shot":
        result = run_zero_shot(args.model, tick_stride=args.tick_stride)
    else:
        result = run_chunk_memory(args.model, modes=tuple(m.strip() for m in args.chunk_modes.split(",") if m.strip()))
    result["wall_seconds"] = round(time.perf_counter() - started, 1)
    result["gpu"] = {"guard": guard, "memory_at_start": memory_start, "memory_at_end": memory_report()}
    print(f"[G0b] memory at end {result['gpu']['memory_at_end']}", file=sys.stderr, flush=True)
    result["generated_at"] = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    result["command"] = "uv run python scripts/adapt_readout.py " + " ".join(argv if argv is not None else sys.argv[1:])
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=1, default=str) + "\n", encoding="utf-8")
    print(f"→ {out} ({result['wall_seconds']} s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
