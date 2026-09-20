"""Task P1 — D1 규모의 파일럿 학습 실행기: readout-only(T0) · LoRA(진단) · **text backbone 전체(T1)** · 무학습 라벨 점수 ·
학습 구간 메모리 (DGX Spark, BF16). G0b의 `--mode t0|lora|zero-shot|chunk-memory`를 그대로 두고 D1과 `t1`을 더했다.

docs/06 Task 2b의 인터페이스 ``adapt_readout(model_id, records, steps) -> dict``와 docs/06 Task 5의 파일럿 학습이 같은
진입점이다. 학습은 :mod:`robo_jev.train` 의 Trainer(혼합 sampler, TBPTT, 후보 순서 치환 증강) 그대로이고, 평가는
:func:`robo_jev.evaluate.evaluate_suite`(고정 평가 집합 `configs/eval/pilot.yaml`), 무학습 점수는
:mod:`robo_jev.model.zero_shot` 이다.

**설정이 데이터를 정한다.** 학습 설정은 ``--config``(`configs/train/qwen35-2b-pilot.yaml`; `extends:`로 4B 형제를 잇고
`modes:`가 `--mode`별 차이를 담는다)이고, 평가 집합은 ``--eval-config``(`configs/eval/pilot.yaml`)다. ``--dataset``은
학습 manifest만 바꾼다: ``d1``(기본, 이 과제) 또는 ``batch0``(G0b가 쓴 D0 + batch-0 + 옛 pilot — 그 수를 재현할 때).

모드(``--mode``):

* ``t0`` — readout-only (backbone 고정 BF16, readout fp32, 정적 윈도우 KV, 10초 구간).
* ``lora`` — peft LoRA(r=16) + readout. **진단 조건이지 T1이 아니다** (docs/03 §5) — 5초 구간, 층 단위 activation checkpointing.
* ``t1`` — text backbone 전체 + readout (10초 구간 + checkpointing; G0b 사다리로 2B는 47.3 GiB).
* ``zero-shot`` — 학습 없이 Nimble 방식 코드 토큰 점수 (평가 집합만; 스트림은 ``--tick-stride`` 틱마다).
* ``chunk-memory`` — 1→2→5→10초 구간 forward+backward의 peak 메모리·시간.

실행 예:

    uv run python scripts/adapt_readout.py --config configs/train/qwen35-2b-pilot.yaml --mode t0 --steps 200 \
        --eval-config configs/eval/pilot.yaml --out artifacts/reports/p1-2b-t0.json
"""

from __future__ import annotations

import argparse
import copy
import datetime as dt
import json
import resource
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import yaml

from robo_jev.gpu import DEFAULT_FRACTION, limit_gpu_memory, memory_report, require_free

REPO = Path(__file__).resolve().parents[1]
D0_MANIFEST = REPO / "tests" / "fixtures" / "d0_manifest.json"
BATCH0_MANIFEST = REPO / "artifacts" / "datasets" / "d1-robot" / "batch-0" / "manifest.json"
PILOT_MANIFEST = REPO / "artifacts" / "datasets" / "pilot" / "manifest.json"
DEFAULT_TRAIN_CONFIG = REPO / "configs" / "train" / "qwen35-2b-pilot.yaml"
DEFAULT_EVAL_CONFIG = REPO / "configs" / "eval" / "pilot.yaml"
MODES = ("t0", "lora", "t1", "zero-shot", "chunk-memory")
#: 학습 데이터 선택 — `d1`(이 과제) / `batch0`(G0b 재현: D0 + batch-0 에피소드 + 옛 pilot 2,000건).
DATASETS = ("d1", "batch0")
BATCH0_MANIFESTS = [
    {"path": str(D0_MANIFEST), "domain": None, "files": None},
    {"path": str(BATCH0_MANIFEST), "domain": "robot", "files": ["episodes/*/streams.jsonl"]},
    {"path": str(PILOT_MANIFEST), "domain": "non_robot", "files": None},
]
#: 로봇/비로봇 축의 태그 — pilot의 `provenance.domain`은 분야 이름이라 쓸 수 없다 (학습 설정 `sampler.domain_tag`와 같은 값)
DOMAIN_TAG = "provenance.robojev_domain"
#: LoRA·T1 학습 전에 있어야 하는 장치(통합) 메모리 여유 — G0b 사다리의 4B 5초 full 구간 peak 58 GiB + 여유
TRAIN_MIN_FREE_BYTES = 60 * 2**30
#: RSS를 적는 step (docs/06 Task 5 선결 조건 1의 확인 — sampler 적재가 D1에서 몇 GiB인가).
RSS_STEPS = (1, 10)


def _tokenizer_id(name: str | None = None) -> str:
    """설정의 `tokenizer` 값을 실제 id로. ``auto``(기본)는 `artifacts/tokenizers`에 받아 둔 것을 쓴다.

    Qwen3.5 계열은 Qwen3.8-27B와 같은 tokenizer 파일을 쓰고(해시 일치 확인됨, 2026-09-19) G0b의 모든 실측도 그 파일로
    했다 — 배포 계약 digest에 tokenizer 파일 해시가 들어가므로 여기서 파일을 바꾸면 G0b·D1의 값과 비교할 수 없다.
    """
    if name and name not in ("auto", "whitespace"):
        return name
    if name == "whitespace":
        return name
    from robo_jev.model.tokenizer import available_tokenizer

    found = available_tokenizer()
    if found is None:
        raise FileNotFoundError("실제 tokenizer가 없다 — `uv run python scripts/fetch_tokenizer.py`")
    return found[0]


def load_train_config(
    path: str | Path,
    *,
    mode: str,
    steps: int | None = None,
    seed: int | None = None,
    dataset: str = "d1",
    run_id: str | None = None,
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """학습 설정 YAML → Trainer 설정.

    `extends:`(같은 디렉터리의 다른 설정)를 먼저 읽어 덮어쓰고, `modes:`에서 `mode`의 블록을 얹은 뒤 그 두 키를 버린다
    (`robo_jev.train.resolve_config`는 모르는 키를 거절한다). 상대 경로의 manifest·tokenizer는 저장소 기준으로 푼다.
    """
    path = Path(path)
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    modes = raw.pop("modes", None) or {}
    if "extends" in raw:
        config = {**load_train_config(path.parent / raw.pop("extends"), mode=mode, dataset=dataset), **raw}
    else:
        config = dict(raw)
    if modes:
        if mode not in modes:
            raise ValueError(f"{path}: modes에 {mode!r} 블록이 없다 (있는 것: {sorted(modes)})")
        config.update(copy.deepcopy(modes[mode]))
    if dataset == "batch0":
        config["dataset_manifests"] = copy.deepcopy(BATCH0_MANIFESTS)
    manifests = []
    for entry in config.get("dataset_manifests") or []:
        entry = dict(entry)
        manifest_path = Path(entry["path"])
        entry["path"] = str(manifest_path if manifest_path.is_absolute() else REPO / manifest_path)
        manifests.append(entry)
    config["dataset_manifests"] = manifests
    config["tokenizer"] = _tokenizer_id(config.get("tokenizer"))
    if config.get("artifacts_dir") and not Path(config["artifacts_dir"]).is_absolute():
        config["artifacts_dir"] = str(REPO / config["artifacts_dir"])
    if steps is not None:
        config["max_steps"] = int(steps)
        config["checkpoint_every"] = int(steps)
    if seed is not None:
        config["seed"] = int(seed)
        sampler = dict(config.get("sampler") or {})
        if sampler.get("permute_candidates_seed") is not None:
            sampler["permute_candidates_seed"] = int(seed)
        config["sampler"] = sampler
    if run_id is not None:
        config["run_id"] = run_id
    config.update(copy.deepcopy(overrides or {}))
    return config


def _memory() -> dict[str, Any]:
    import torch

    stats = torch.cuda.memory_stats()
    return {
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()), "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
        "num_alloc_retries": int(stats.get("num_alloc_retries", 0)), "num_ooms": int(stats.get("num_ooms", 0)),
    }


def _rss_gib() -> float:
    """이 프로세스의 최대 RSS (GiB) — D1 규모에서 sampler 적재가 얼마나 드는지의 실측 (docs/06 Task 5 선결 조건 1)."""
    return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024**2), 2)


def evaluate_judge(judge: Any, tokenizer: Any, *, eval_config: str | Path, log: Any = sys.stderr) -> dict[str, Any]:
    """고정 평가 집합(`configs/eval/pilot.yaml`)의 표 — 분할마다 표준 열 + 선택적 지표 + 대조 쌍 검사."""
    from robo_jev.evaluate import evaluate_suite, load_eval_suite

    suite = load_eval_suite(eval_config)
    return evaluate_suite(judge, suite, tokenizer=tokenizer, root=REPO, log=log)


def run_training(
    config: dict[str, Any],
    *,
    mode: str,
    eval_after: bool = True,
    eval_config: str | Path = DEFAULT_EVAL_CONFIG,
) -> dict[str, Any]:
    """T0 / LoRA / T1 학습 + 평가. 돌려주는 것: 곡선, 메모리(RSS 포함), step 시간, checkpoint, 평가 표."""
    import torch

    from robo_jev.train import Trainer

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.reset_accumulated_memory_stats()
    if mode in ("lora", "t1"):
        # 앞 프로세스가 아직 메모리를 놓지 않았으면 OOM 대신 분명한 오류로 (G0b 구간 메모리: 2B full 10초 47.3 GiB, 4B LoRA 5초 64.8 GiB)
        require_free(TRAIN_MIN_FREE_BYTES, what=f"{mode} {config['model_id']}")
    started = time.perf_counter()
    rss_before_load = _rss_gib()
    with Trainer(config) as trainer:
        load_seconds = round(time.perf_counter() - started, 1)
        rss_after_load = _rss_gib()
        per_step: list[dict[str, Any]] = []

        def hook(_: Any, metrics: dict[str, Any]) -> None:
            if metrics["step"] in RSS_STEPS:
                per_step.append({"step": metrics["step"], "rss_gib": _rss_gib(), "gpu_peak_allocated_bytes": int(torch.cuda.max_memory_allocated())})

        trainer.step_hook = hook
        result = trainer.run()
        tokenizer = trainer.tokenizer
        curve = [
            {"step": m["step"], "loss": m["loss"], "loss_by_domain": m["loss_by_domain"], "loss_by_type": m["loss_by_type"], "grad_norm": m["grad_norm"], "lr": m["lr"], "tokens": m["tokens"]["total"], "seconds": m["seconds"]}
            for m in result["metrics"]["steps"]
        ]
        memory = _memory()
        seconds = [m["seconds"] for m in result["metrics"]["steps"]]
        tokens = [m["tokens"]["total"] for m in result["metrics"]["steps"]]
        checkpoint = result["checkpoint"]
        if checkpoint and str(checkpoint).startswith(str(REPO)):
            checkpoint = str(Path(checkpoint).relative_to(REPO))  # 저장소 기준 상대 경로 (worktree 절대 경로를 남기지 않는다 — G0b 리뷰 1 M8)
        out = {
            "model_id": config["model_id"], "mode": mode, "steps": int(config["max_steps"]), "run_id": result["run_id"], "checkpoint": checkpoint, "status": result["status"],
            "config": {key: value for key, value in trainer.config.items() if key != "dataset_manifests"},
            "dataset_manifests": [{"path": str(Path(entry["path"]).relative_to(REPO)) if str(entry["path"]).startswith(str(REPO)) else entry["path"], "domain": entry.get("domain"), "files": entry.get("files")} for entry in trainer.config["dataset_manifests"]],
            "manifest": {key: trainer.manifest[key] for key in ("model", "contract_sha256", "serializer_version", "tokenizer")},
            "items": {"total": len(trainer.items), "stream": sum(i.kind == "stream" for i in trainer.items), "single": sum(i.kind == "single" for i in trainer.items)},
            "load_seconds": load_seconds, "train_seconds": round(sum(seconds), 1),
            "step_seconds": {"mean": round(statistics.fmean(seconds), 2), "p50": round(statistics.median(seconds), 2), "max": round(max(seconds), 2)},
            "tokens": {"total": int(sum(tokens)), "per_step_mean": round(statistics.fmean(tokens), 1), "per_second": round(sum(tokens) / max(sum(seconds), 1e-9), 1)},
            "memory": memory,
            # 선결 조건 1(sampler 적재 메모리)의 실측 — 적재 전후와 step 1·10의 RSS
            "rss_gib": {"before_load": rss_before_load, "after_load": rss_after_load, "per_step": per_step, "end": _rss_gib()},
            "curve": curve,
            "loss_first_last": [curve[0]["loss"], curve[-1]["loss"]] if curve else None,
        }
        if eval_after:
            trainer.model.eval()
            out["evaluation"] = evaluate_judge(trainer.model, tokenizer, eval_config=eval_config)
            out["rss_gib"]["after_eval"] = _rss_gib()
    return out


def evaluate_checkpoint(config: dict[str, Any], *, mode: str, checkpoint: str, eval_config: str | Path = DEFAULT_EVAL_CONFIG) -> dict[str, Any]:
    """이미 끝난 run의 checkpoint(readout(+LoRA))를 같은 설정의 모델에 싣고 **평가만** 한다 — 학습 뒤 평가가 실패했을 때 다시 학습하지
    않으려고. 곡선·step 시간은 run 디렉터리의 `metrics.json`에서 읽는다; `memory`는 평가 프로세스의 peak다(학습 peak가 아니다)."""
    import torch

    from robo_jev.train import build_model, build_tokenizer, load_readout_checkpoint, resolve_config, tokenizer_block

    path = Path(checkpoint)
    run_dir = path.parent
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8")) if (run_dir / "metrics.json").is_file() else {}
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    resolved = resolve_config({**config, "run_id": run_dir.name})
    judge = build_model(resolved)
    tokenizer = build_tokenizer(resolved["tokenizer"])
    # 배포 계약 digest의 네 조각(직렬화·계약 소스, 하네스 버전, tokenizer 파일 해시)을 지금 체크아웃 기준으로 대조한다
    manifest = load_readout_checkpoint(judge, path, tokenizer_sha256=tokenizer_block(resolved["tokenizer"])["sha256"])
    judge.eval()
    load_seconds = round(time.perf_counter() - started, 1)
    steps_done = metrics.get("steps") or []
    curve = [
        {"step": m["step"], "loss": m["loss"], "loss_by_domain": m["loss_by_domain"], "loss_by_type": m["loss_by_type"], "grad_norm": m["grad_norm"], "lr": m["lr"], "tokens": m["tokens"]["total"], "seconds": m["seconds"]}
        for m in steps_done
    ]
    seconds = [m["seconds"] for m in steps_done]
    out: dict[str, Any] = {
        "model_id": resolved["model_id"], "mode": mode, "steps": int(metrics.get("step", resolved["max_steps"])), "run_id": metrics.get("run_id", run_dir.name),
        "checkpoint": str(path), "status": metrics.get("status"), "evaluated_from_checkpoint": True,
        "config": {key: value for key, value in resolved.items() if key != "dataset_manifests"},
        "manifest": {key: manifest.get(key) for key in ("model", "contract_sha256", "serializer_version", "tokenizer")},
        "items": None, "load_seconds": load_seconds, "train_seconds": round(sum(seconds), 1) if seconds else None,
        "step_seconds": {"mean": round(statistics.fmean(seconds), 2), "p50": round(statistics.median(seconds), 2), "max": round(max(seconds), 2)} if seconds else None,
        "memory": {**_memory(), "note": "evaluation process only (the training peak is not in metrics.json)"}, "curve": curve,
        "loss_first_last": [curve[0]["loss"], curve[-1]["loss"]] if curve else None,
    }
    out["evaluation"] = evaluate_judge(judge, tokenizer, eval_config=eval_config)
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
        config = load_train_config(DEFAULT_TRAIN_CONFIG, mode=mode, steps=steps, seed=seed, run_id=f"adapt-{dt.datetime.now().strftime('%H%M%S')}")
        config["model_id"] = model_id
        config["dataset_manifests"] = [{"path": str(root / "manifest.json")}]
        with Trainer(config) as trainer:
            result = trainer.run()
            out: dict[str, Any] = {"run_id": result["run_id"], "steps": result["step"], "curve": [(m["step"], m["loss"]) for m in result["metrics"]["steps"]], "memory": _memory()}
            if eval_records:
                items = load_items(root / "manifest.json", tokenizer=load_tokenizer(config["tokenizer"]), splits=("dev",))
                out["evaluation"] = evaluate_items(trainer.model, items, tokenizer=trainer.tokenizer)
    return out


def run_zero_shot(config: dict[str, Any], *, eval_config: str | Path = DEFAULT_EVAL_CONFIG, tick_stride: int = 8, shuffle_seed: int = 1, batch: int = 8) -> dict[str, Any]:
    """무학습 라벨 점수 (Nimble 방식) — **같은 고정 평가 집합**의 레코드에, 스트림은 `tick_stride` 틱마다."""
    import torch

    from robo_jev.evaluate import eval_suite_identity, load_eval_suite, load_suite_items
    from robo_jev.model.backbone_qwen import QwenBackbone
    from robo_jev.model.tokenizer import load_tokenizer
    from robo_jev.model.zero_shot import zero_shot_label_scores

    model_id = config["model_id"]
    tokenizer = load_tokenizer(config["tokenizer"])
    suite = load_eval_suite(eval_config)
    items = load_suite_items(suite, tokenizer=tokenizer, root=REPO, domain_tag=DOMAIN_TAG)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    backbone = QwenBackbone.load(model_id, root=config.get("model_root"), dtype=torch.bfloat16, device="cuda", kv_mode="static")
    out: dict[str, Any] = {
        "model_id": model_id, "mode": "zero-shot", "load_seconds": round(time.perf_counter() - started, 1),
        "tick_stride": int(tick_stride), "evaluation": {"eval_set": eval_suite_identity(suite, items), "splits": {}},
    }
    for key, subset in items.items():
        started = time.perf_counter()
        records = [item.record for item in subset]
        result = zero_shot_label_scores(model_id, records, backbone=backbone, tokenizer=tokenizer, shuffle_seed=shuffle_seed, tick_stride=tick_stride, batch=batch)
        result["seconds"] = round(time.perf_counter() - started, 1)
        out["evaluation"]["splits"][key] = result
        print(f"[p1] zero-shot {key}: {result['prompts']} prompts, acc {result['table']['_all']['accuracy']}, nll {result['table']['_all']['nll']:.3f} ({result['seconds']} s)", file=sys.stderr, flush=True)
    out["memory"] = _memory()
    return out


CHUNK_SECONDS_LADDER = (1.0, 2.0, 5.0, 10.0)
#: 구간을 늘리기 전에 남아 있어야 하는 장치(통합) 메모리의 몫 — 그보다 적으면 계획을 멈춘다(커널 OOM으로 세션이 죽는 대신).
CHUNK_MIN_FREE_SHARE = 0.2
#: 5초 → 10초 full 구간으로 가려면 5초 실행의 peak reserved가 남긴 여유가 이 몫 이상이어야 한다.
CHUNK_FULL_10S_HEADROOM = 0.4
#: checkpointing 없는 full은 이 길이까지만 (45K 토큰은 ≈240 GB — 2026-09-20의 커널 OOM 원인).
CHUNK_NOCKPT_MAX_SECONDS = 2.0


def run_chunk_memory(config: dict[str, Any], *, modes: tuple[str, ...] = ("readout", "full", "full_nockpt"), ladder: tuple[float, ...] = CHUNK_SECONDS_LADDER) -> dict[str, Any]:
    """학습 데이터에서 가장 긴 에피소드의 첫 구간 forward+backward — readout-only, full(backbone 전체 gradient, 층 단위 activation
    checkpointing), full_nockpt(checkpointing 없이)의 peak 메모리·시간 (G0b §S3.4와 같은 사다리)."""
    import gc

    import torch

    from robo_jev.model.backbone_qwen import QwenBackbone
    from robo_jev.model.judge import Judge
    from robo_jev.model.tokenizer import load_tokenizer
    from robo_jev.sampler import load_items
    from robo_jev.train import plan_episode, run_stream_chunk

    model_id = config["model_id"]
    tokenizer = load_tokenizer(config["tokenizer"])
    robot = next(entry for entry in config["dataset_manifests"] if entry.get("domain") == "robot")
    items = [i for i in load_items(robot["path"], tokenizer=tokenizer, splits=("train",), domain="robot", domain_tag=DOMAIN_TAG, files=robot.get("files")) if i.kind == "stream"]
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
                print(f"[p1] chunk-memory {model_id}: stopped before {mode} {seconds}s — {out['stopped_at']['reason']}", file=sys.stderr, flush=True)
                return out
            if mode == "full" and seconds >= 10.0 and previous is not None:
                headroom = 1.0 - previous["peak_reserved_bytes"] / report["total_bytes"]
                if headroom < CHUNK_FULL_10S_HEADROOM:
                    out["runs"].append({"mode": mode, "chunk_seconds": seconds, "skipped": f"{previous['chunk_seconds']}s run left {headroom:.0%} headroom < {CHUNK_FULL_10S_HEADROOM:.0%}"})
                    print(f"[p1] chunk-memory {model_id}: skip {mode} {seconds}s — headroom {headroom:.0%}", file=sys.stderr, flush=True)
                    break
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.reset_accumulated_memory_stats()  # num_alloc_retries·num_ooms는 프로세스 누적이라 실행마다 0에서 (G0b 리뷰 1 M2)
            backbone = QwenBackbone.load(model_id, root=config.get("model_root"), kv_mode="static" if mode == "readout" else "dynamic")
            if mode != "readout":
                backbone.model.requires_grad_(True)
                backbone.activation_checkpointing = mode == "full"
            judge = Judge(backbone, rank=64, readout="pointer", seed=1000, readout_dtype=torch.float32)
            plan = plan_episode(item.record, chunk_seconds=seconds, tick_weights=config["sampler"]["tick_weights"])
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
            print(f"[p1] chunk-memory {model_id} {mode} {seconds}s: {entry}", file=sys.stderr, flush=True)
            previous = entry
            del result, judge, backbone
            gc.collect()
            torch.cuda.empty_cache()
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default=str(DEFAULT_TRAIN_CONFIG), help="학습 설정 YAML (extends·modes를 푼다)")
    parser.add_argument("--eval-config", dest="eval_config", default=str(DEFAULT_EVAL_CONFIG), help="고정 평가 집합 설정")
    parser.add_argument("--dataset", default="d1", choices=DATASETS, help="학습 데이터 — d1(기본) 또는 batch0(G0b 재현)")
    parser.add_argument("--model", default=None, help="설정의 model_id를 덮어쓴다 (candidates.yaml의 후보 id)")
    parser.add_argument("--mode", default="t0", choices=MODES)
    parser.add_argument("--steps", type=int, default=None, help="설정의 max_steps를 덮어쓴다")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--tick-stride", dest="tick_stride", type=int, default=8, help="zero-shot: 스트림에서 몇 틱마다 하나를 잴지")
    parser.add_argument("--chunk-modes", dest="chunk_modes", default="readout,full,full_nockpt", help="chunk-memory: readout | full(층 단위 checkpointing) | full_nockpt, 쉼표로 — 구간은 1→2→5→10초 사다리")
    parser.add_argument("--no-eval", dest="no_eval", action="store_true")
    parser.add_argument("--eval-checkpoint", dest="eval_checkpoint", default=None, help="t0/lora/t1: 학습하지 않고 이 checkpoint를 실어 평가만 (run 디렉터리의 metrics.json에서 곡선을 읽는다)")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="학습 설정 덮어쓰기 (값은 YAML로 읽는다)")
    parser.add_argument("--out", required=True)
    parser.add_argument("--gpu-memory-fraction", dest="gpu_memory_fraction", type=float, default=DEFAULT_FRACTION, help="프로세스가 쓸 장치(통합) 메모리 몫 (robo_jev.gpu; 첫 CUDA 할당 전에 건다)")
    args = parser.parse_args(argv)
    guard = limit_gpu_memory(args.gpu_memory_fraction)
    memory_start = memory_report()
    print(f"[p1] gpu guard {guard} · memory at start {memory_start}", file=sys.stderr, flush=True)

    overrides: dict[str, Any] = {}
    if args.model:
        overrides["model_id"] = args.model
    for assignment in args.set:
        key, separator, value = assignment.partition("=")
        if not separator:
            parser.error(f"--set은 KEY=VALUE 꼴이어야 한다: {assignment!r}")
        overrides[key] = yaml.safe_load(value)
    run_id = f"p1-{args.mode}-{(args.model or '').split('/')[-1].lower() or 'model'}-{dt.datetime.now().strftime('%Y%m%d-%H%M%S')}"
    config = load_train_config(
        args.config, mode=args.mode if args.mode in ("t0", "lora", "t1") else "t0", steps=args.steps, seed=args.seed,
        dataset=args.dataset, run_id=run_id, overrides=overrides,
    )  # fmt: skip
    started = time.perf_counter()
    if args.mode in ("t0", "lora", "t1") and args.eval_checkpoint:
        result = evaluate_checkpoint(config, mode=args.mode, checkpoint=args.eval_checkpoint, eval_config=args.eval_config)
    elif args.mode in ("t0", "lora", "t1"):
        result = run_training(config, mode=args.mode, eval_after=not args.no_eval, eval_config=args.eval_config)
    elif args.mode == "zero-shot":
        from robo_jev.train import resolve_config

        result = run_zero_shot(resolve_config(config), eval_config=args.eval_config, tick_stride=args.tick_stride)
    else:
        from robo_jev.train import resolve_config

        result = run_chunk_memory(resolve_config(config), modes=tuple(m.strip() for m in args.chunk_modes.split(",") if m.strip()))
    result["wall_seconds"] = round(time.perf_counter() - started, 1)
    result["dataset"] = args.dataset
    result["eval_config"] = str(Path(args.eval_config).relative_to(REPO)) if str(Path(args.eval_config).resolve()).startswith(str(REPO)) else str(args.eval_config)
    result["train_config"] = str(Path(args.config).relative_to(REPO)) if str(Path(args.config).resolve()).startswith(str(REPO)) else str(args.config)
    result["gpu"] = {"guard": guard, "memory_at_start": memory_start, "memory_at_end": memory_report()}
    print(f"[p1] memory at end {result['gpu']['memory_at_end']}", file=sys.stderr, flush=True)
    result["generated_at"] = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    result["command"] = "uv run python scripts/adapt_readout.py " + " ".join(argv if argv is not None else sys.argv[1:])
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=1, default=str) + "\n", encoding="utf-8")
    print(f"→ {out} ({result['wall_seconds']} s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
