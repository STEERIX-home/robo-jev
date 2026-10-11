#!/usr/bin/env python
"""Task v0c — 판단 요청 진단: 학습 없는 Qwen이 `simgen-request/0` 요청에 내는 확률을 읽고 사후 확률 정답과 비교한다.

    # 채점 (GPU — artifacts/scratch/v0c/gpu.sh로만): 2B·4B, 바로 답(direct) / 생각하고 답(reasoning)
    uv run python scripts/diagnose_decisions.py --model Qwen/Qwen3.5-2B --mode direct --input artifacts/simgen/v0a/sample-2000.jsonl \\
        --limit 300 --seed 0 --out artifacts/reports/v0c-preview-2b-direct.json --preview
    uv run python scripts/diagnose_decisions.py --model Qwen/Qwen3.5-2B --mode reasoning --input … --limit 100 --seed 0 --out … --preview
    # 저장한 예측에서 요약만 다시 (CPU)
    uv run python scripts/diagnose_decisions.py --rescore <run>.predictions.jsonl --input … --out <rescored>.json
    # 같은 기록의 두 run 비교 — 차 a − b, 짝지은 레코드 bootstrap (CPU)
    uv run python scripts/diagnose_decisions.py --compare <a>.predictions.jsonl <b>.predictions.jsonl --labels "2B direct" "2B reasoning" --input … --out <cmp>.json

산출물(`--out X.json`): ``X.json``(설정·검증·실행·요약), ``X.predictions.jsonl``(레코드마다 모델 답 — 정답은 넣지 않는다),
``X.md``(사람이 읽는 표). 형식이 틀린 기록(:func:`robo_jev.decisions.records.validate_record`)은 세고 건너뛴다. `--limit`은 쌍을
깨지 않는 무작위 표본(`--seed`)이고, 선택지 순서는 `--shuffle-seed`로 섞는다.

GPU 규칙(GB10 통합 메모리): 첫 CUDA 할당 전에 울타리(`--gpu-memory-fraction`, 기본 0.6), 적재 전에 여유 확인(2B ≥ 20 GiB,
4B ≥ 30 GiB — 근거는 :data:`robo_jev.decisions.zero_shot.MIN_FREE_BYTES`), 처음과 끝의 메모리 보고. ``--model tiny``는 검사 전용의
소형 난수 Qwen(CPU)이다.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from robo_jev.gpu import DEFAULT_FRACTION, limit_gpu_memory, memory_report, require_free  # noqa: E402

SCRIPT_VERSION = "v0c-diagnose-1"
TINY = "tiny"
#: 모르는 모델 id의 적재 전 여유 (보수적으로 4B의 값).
FALLBACK_MIN_FREE = 30 * 2**30


def _log(message: str) -> None:
    print(f"[v0c {dt.datetime.now().strftime('%H:%M:%S')}] {message}", file=sys.stderr, flush=True)


def _git_sha() -> str | None:
    try:
        return subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True).stdout.strip() or None
    except (OSError, subprocess.CalledProcessError):
        return None


def _paths(out: Path) -> dict[str, Path]:
    return {"summary": out, "predictions": out.with_suffix(".predictions.jsonl"), "markdown": out.with_suffix(".md")}


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=1, default=str) + "\n", encoding="utf-8")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    from robo_jev.decisions.records import read_records

    return read_records(path)


def _load_valid(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    from robo_jev.decisions.records import validate_record

    records = _read_jsonl(path)
    valid, problems = [], {}
    for record in records:
        found = validate_record(record)
        if found:
            if len(problems) < 20:
                problems[str(record.get("id"))] = found[:5]
            continue
        valid.append(record)
    return valid, {"read": len(records), "invalid": len(records) - len(valid), "problems": problems}


def _parser() -> argparse.ArgumentParser:
    from robo_jev.decisions.prompt import DEFAULT_FORMAT, FORMATS, MODES
    from robo_jev.decisions.zero_shot import DEFAULT_BATCH, DEFAULT_GEN_BATCH, REASONING_MAX_NEW_TOKENS

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--model", help="Qwen/Qwen3.5-2B | Qwen/Qwen3.5-4B (가중치는 artifacts/models/) | tiny(검사 전용)")
    action.add_argument("--rescore", type=Path, help="저장한 예측 JSONL에서 요약만 다시 계산 (CPU)")
    action.add_argument("--compare", type=Path, nargs=2, metavar=("A", "B"), help="두 예측 JSONL의 짝지은 비교 a − b (CPU)")
    parser.add_argument("--input", type=Path, required=True, help="simgen-request/0 JSONL")
    parser.add_argument("--out", type=Path, required=True, help="요약 JSON 경로 (옆에 .predictions.jsonl·.md)")
    parser.add_argument("--mode", choices=MODES, default="direct")
    parser.add_argument("--format", dest="fmt", choices=FORMATS, default=DEFAULT_FORMAT, help=f"프롬프트 형식 (기본 {DEFAULT_FORMAT})")
    parser.add_argument("--limit", type=int, default=None, help="쌍을 깨지 않는 무작위 표본 크기 (없으면 전부)")
    parser.add_argument("--seed", type=int, default=0, help="표본 고르기 seed")
    parser.add_argument("--shuffle-seed", type=int, default=1, help="선택지 순서 seed")
    parser.add_argument("--shuffle-seeds", type=int, nargs="+", default=None,
                        help="선택지 순서 seed 여럿 — 첫 seed가 주 run이고, 나머지로 순서만 바꾼 판정 변경률(permutation)을 잰다")
    parser.add_argument("--batch", type=int, default=DEFAULT_BATCH)
    parser.add_argument("--gen-batch", type=int, default=DEFAULT_GEN_BATCH)
    parser.add_argument("--max-new-tokens", type=int, default=REASONING_MAX_NEW_TOKENS)
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("--title", default=None)
    parser.add_argument("--preview", action="store_true", help="보고서 머리에 '봉인된 시험 세트가 아닌 생성 표본의 미리보기'를 단다")
    parser.add_argument("--labels", nargs=2, default=("a", "b"), metavar=("LABEL_A", "LABEL_B"), help="--compare의 열 이름")
    parser.add_argument("--gpu-memory-fraction", type=float, default=DEFAULT_FRACTION)
    return parser


def _summarize_with_reasoning(records: list[dict[str, Any]], predictions: list[dict[str, Any]], *, n_boot: int, seed: int) -> dict[str, Any]:
    from robo_jev.decisions.metrics import summarize
    from robo_jev.decisions.report import reasoning_stats

    summary = summarize(records, predictions, n_boot=n_boot, seed=seed)
    stats = reasoning_stats(predictions)
    if stats is not None:
        summary["reasoning"] = stats
    return summary


def _write_outputs(out: Path, payload: dict[str, Any], predictions: list[dict[str, Any]] | None, *, title: str, note: str | None) -> None:
    from robo_jev.decisions.report import render_markdown

    paths = _paths(out)
    _write_json(paths["summary"], payload)
    if predictions is not None:
        with paths["predictions"].open("w", encoding="utf-8") as handle:
            for prediction in predictions:
                handle.write(json.dumps(prediction, ensure_ascii=False) + "\n")
    paths["markdown"].write_text(render_markdown(payload["summary"], config=payload["config"], title=title, note=note), encoding="utf-8")


def _rescore(args: argparse.Namespace) -> int:
    from robo_jev.decisions.records import file_sha256
    from robo_jev.decisions.report import PREVIEW_NOTE

    records, validation = _load_valid(args.input)
    predictions = _read_jsonl(args.rescore)
    ids = {str(p["id"]) for p in predictions}
    records = [r for r in records if str(r["id"]) in ids]
    first = predictions[0] if predictions else {}
    config = {
        "model": first.get("model"), "mode": first.get("mode"), "format": first.get("format"), "prompt_version": first.get("prompt_version"),
        "shuffle_seed": first.get("shuffle_seed"), "records": len(records), "input": str(args.input), "input_sha256": file_sha256(args.input),
        "predictions": str(args.rescore), "rescored": True, "script_version": SCRIPT_VERSION, "git": _git_sha(),
    }
    summary = _summarize_with_reasoning(records, predictions, n_boot=args.n_boot, seed=args.seed)
    payload = {"config": config, "validation": validation, "run": None, "summary": summary}
    _write_outputs(args.out, payload, None, title=args.title or f"v0c diagnosis (rescored) — {config['model']} {config['mode']}", note=PREVIEW_NOTE if args.preview else None)
    _log(f"rescored {len(records)} records → {args.out}")
    return 0


def _compare(args: argparse.Namespace) -> int:
    from robo_jev.decisions.metrics import compare_runs
    from robo_jev.decisions.records import file_sha256
    from robo_jev.decisions.report import PREVIEW_NOTE, render_comparison

    records, validation = _load_valid(args.input)
    a, b = _read_jsonl(args.compare[0]), _read_jsonl(args.compare[1])
    comparison = compare_runs(records, a, b, n_boot=args.n_boot, seed=args.seed)
    payload = {
        "config": {"a": str(args.compare[0]), "b": str(args.compare[1]), "labels": list(args.labels), "input": str(args.input),
                   "input_sha256": file_sha256(args.input), "n_boot": args.n_boot, "seed": args.seed, "script_version": SCRIPT_VERSION, "git": _git_sha()},
        "validation": validation, "comparison": comparison,
    }
    _write_json(args.out, payload)
    title = args.title or f"v0c comparison — {args.labels[0]} vs {args.labels[1]}"
    args.out.with_suffix(".md").write_text(
        render_comparison(comparison, label_a=args.labels[0], label_b=args.labels[1], title=title, note=PREVIEW_NOTE if args.preview else None), encoding="utf-8"
    )
    _log(f"compared {comparison['records']} records → {args.out}")
    return 0


def _load_backbone(model_id: str, tokenizer: Any) -> Any:
    from robo_jev.model.backbone_qwen import QwenBackbone

    if model_id == TINY:
        return QwenBackbone.tiny(seed=4, vocab_size=tokenizer.get_vocab_size())
    return QwenBackbone.load(model_id)


def _run(args: argparse.Namespace) -> int:
    # 울타리는 첫 CUDA 할당 전에 (robo_jev.gpu) — 기록·tokenizer 읽기는 CUDA를 건드리지 않는다
    guard = limit_gpu_memory(args.gpu_memory_fraction)
    memory_start = memory_report()
    _log(f"gpu guard {guard} · memory at start {memory_start}")
    from robo_jev.decisions.prompt import PROMPT_VERSION, fixed_text_digest
    from robo_jev.decisions.records import file_sha256, select_records
    from robo_jev.decisions.report import PREVIEW_NOTE
    from robo_jev.decisions.zero_shot import MIN_FREE_BYTES, diagnose_records
    from robo_jev.model.tokenizer import available_tokenizer, describe_tokenizer, load_tokenizer

    records, validation = _load_valid(args.input)
    chosen = select_records(records, args.limit, seed=args.seed)
    _log(f"{validation['read']} records read, {validation['invalid']} invalid, {len(chosen)} chosen (limit {args.limit}, seed {args.seed})")
    tokenizer_id = available_tokenizer()[0]
    tokenizer = load_tokenizer(tokenizer_id)
    if args.model != TINY:
        require_free(MIN_FREE_BYTES.get(args.model, FALLBACK_MIN_FREE), what=f"load {args.model}")
    started = time.perf_counter()
    backbone = _load_backbone(args.model, tokenizer)
    load_seconds = time.perf_counter() - started
    _log(f"loaded {args.model} in {load_seconds:.1f} s · memory {memory_report()}")
    seeds = list(args.shuffle_seeds) if args.shuffle_seeds else [args.shuffle_seed]
    started = time.perf_counter()
    runs = []
    for shuffle_seed in seeds:
        runs.append(diagnose_records(
            chosen, backbone, tokenizer, model_id=args.model, mode=args.mode, fmt=args.fmt, seed=shuffle_seed, batch=args.batch,
            gen_batch=args.gen_batch, max_new_tokens=args.max_new_tokens, log=_log,
        ))
    predictions = runs[0]
    seconds = time.perf_counter() - started
    prompts = sum(len(p["questions"]) for one in runs for p in one)
    import torch

    from robo_jev.decisions.metrics import average_over_orders, order_effects, permutation_sensitivity, summarize

    peak = int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else None
    summary = _summarize_with_reasoning(chosen, predictions, n_boot=args.n_boot, seed=args.seed)
    if len(runs) > 1:
        # 같은 기록을 선택지 순서만 바꿔 다시 읽은 것: 판정 변경률, 질문 안의 짝지은 순서 효과, 순서 평균 읽기(위치 편향을 뺀 진단)
        summary["permutation"] = permutation_sensitivity(runs)
        summary["order_effects"] = order_effects(runs)
        averaged = average_over_orders(runs)
        averaged_summary = summarize(chosen, averaged, n_boot=args.n_boot, seed=args.seed)
        summary["order_averaged"] = {
            "orders": len(runs), "overall": averaged_summary["overall"], "recommendation": averaged_summary["recommendation"],
            "groups": {dim: averaged_summary["groups"][dim] for dim in ("phenomena", "scene", "recommendation", "margin_band")},
            "bootstrap": averaged_summary["bootstrap"],
        }
        extra = [(f".predictions.seed{shuffle_seed}.jsonl", run_predictions) for shuffle_seed, run_predictions in zip(seeds[1:], runs[1:])]
        for suffix, rows in extra + [(".predictions.order-averaged.jsonl", averaged)]:
            path = args.out.with_suffix(suffix)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("".join(json.dumps(p, ensure_ascii=False) + "\n" for p in rows), encoding="utf-8")
    config = {
        "model": args.model, "mode": args.mode, "format": args.fmt, "prompt_version": PROMPT_VERSION, "prompt_digest": fixed_text_digest(),
        "seed": args.seed, "shuffle_seed": seeds[0], "shuffle_seeds": seeds, "limit": args.limit, "records": len(chosen), "record_ids": [str(r["id"]) for r in chosen],
        "selection": f"pair-preserving random sample, seed {args.seed}, limit {args.limit}" if args.limit else "all valid records",
        "input": str(args.input), "input_sha256": file_sha256(args.input), "batch": args.batch, "gen_batch": args.gen_batch,
        "max_new_tokens": args.max_new_tokens if args.mode == "reasoning" else None, "n_boot": args.n_boot,
        "tokenizer": describe_tokenizer(tokenizer_id), "backbone_manifest": getattr(backbone, "manifest", None),
        "script_version": SCRIPT_VERSION, "git": _git_sha(), "created": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
    }
    run = {
        "seconds": round(seconds, 2), "load_seconds": round(load_seconds, 2), "prompts": prompts,
        "prompt_tokens": sum(q["prompt_tokens"] for one in runs for p in one for q in p["questions"].values()),
        "scored_tokens": sum(q["scored_tokens"] for one in runs for p in one for q in p["questions"].values()),
        "gpu": {"guard": guard, "memory_at_start": memory_start, "memory_at_end": memory_report(), "peak_allocated_bytes": peak},
    }
    payload = {"config": config, "validation": validation, "run": run, "summary": summary}
    title = args.title or f"v0c diagnosis — {args.model} {args.mode} ({args.fmt})"
    _write_outputs(args.out, payload, predictions, title=title, note=PREVIEW_NOTE if args.preview else None)
    _log(f"{len(predictions)} records, {prompts} prompts in {seconds:.1f} s → {args.out} · memory at end {run['gpu']['memory_at_end']}")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.rescore is not None:
        return _rescore(args)
    if args.compare is not None:
        return _compare(args)
    return _run(args)


if __name__ == "__main__":
    raise SystemExit(main())
