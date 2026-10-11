#!/usr/bin/env python
"""Task v0c — 판단 요청 진단: 학습 없는 Qwen이 `simgen-request/0` 요청에 내는 확률을 읽고 사후 확률 정답과 비교한다.

    # 채점 (GPU — artifacts/scratch/v0c/gpu.sh로만): 2B·4B, 바로 답(direct) / 설명 뒤 답(reasoning, v0c-p2)
    uv run python scripts/diagnose_decisions.py --model Qwen/Qwen3.5-2B --mode direct --input artifacts/simgen/v0a/sample-2000.jsonl \\
        --limit 300 --seed 0 --out artifacts/reports/v0c-preview-2b-direct.json --preview
    uv run python scripts/diagnose_decisions.py --model Qwen/Qwen3.5-2B --mode reasoning --input … --limit 100 --seed 0 --out … --preview
    # 저장한 예측에서 요약만 다시 (CPU) — 옛 seed run 여럿은 질문마다 서로 다른 배치로 합친다
    uv run python scripts/diagnose_decisions.py --rescore <run>.predictions.jsonl [<run>.predictions.seed2.jsonl …] --input … --out <rescored>.json
    # 같은 기록의 두 run 비교 — 두 읽기, 차 a − b, 짝지은 레코드 bootstrap (CPU)
    uv run python scripts/diagnose_decisions.py --compare <a>.predictions.jsonl <b>.predictions.jsonl --labels "2B direct" "2B reasoning" --input … --out <cmp>.json
    # 학습한 가중치(Task v0d — LoRA adapter 또는 전체 가중치; 재개용 저장 단위·model-only 파일 모두)를 같은 프롬프트·같은 배치·같은 두 읽기로
    uv run python scripts/diagnose_decisions.py --model Qwen/Qwen3.5-2B --checkpoint <run>/checkpoint.pt --input … --out <after>.json

배치(`--orders`): ``counterbalanced``(기본 — 질문마다 결정적 균형 배치 모두: choice는 순환 이동 n개, noul·score는 정순·역순)와
``seeded``(seed 배치 하나; ``--shuffle-seeds``를 여럿 주면 run을 여러 번 돌려 서로 다른 배치로 합친다 — 옛 방식). 요약은 두 읽기 —
한 순서(배포 조건, 배치별 지표의 평균)와 순서 평균(배치 평균 답의 지표) — 와 기준선(언제나 0.5 / 무작위 고르기) 대비 짝지은 구간을 낸다.

산출물(`--out X.json`): ``X.json``(설정·검증·실행·요약), ``X.predictions.jsonl``(레코드마다 모델 답 — 질문마다 배치들과 배치 평균;
정답은 넣지 않는다), ``X.md``(사람이 읽는 표). 형식이 틀린 기록과 겹친 id(:func:`robo_jev.decisions.records.validate_record`)는 세고
건너뛴다. `--limit`은 쌍을 깨지 않는 무작위 표본(`--seed`)이다.

`--checkpoint`(Task v0d): 기반 모델(`--model`)을 실은 뒤 :func:`robo_jev.decisions.train.apply_checkpoint` 로 학습한 가중치를 얹는다 —
기반 모델·tokenizer·프롬프트 고정 문구가 학습 때와 다르면 거절한다. 채점은 그 밖에 아무것도 바꾸지 않는다(같은 프롬프트·균형 배치·배치 1·
두 읽기); 예측의 ``model``은 ``<model>+<방식>:<run id>@<step>``이고 요약의 ``config.checkpoint``에 체크포인트의 정체가 남는다.
`--checkpoint`가 없으면 결과는 v0c의 실행기와 비트 단위로 같다(검사).

`--prompt-version`(Task v0d fix round 1): 프롬프트 판 — 기본 ``v0c-p3``(score 기준점 11개, [0, 1] 전체를 읽는다; choice·noul은 p2와 바이트
단위로 같다), ``v0c-p2``는 v0c 결과의 재현용. `--checkpoint`의 판은 학습 때의 판과 같아야 한다(다르면 거절).

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

SCRIPT_VERSION = "v0d-diagnose-4"
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
    """형식 검사를 통과하고 id가 처음 나온 기록만 (겹친 id는 뒤의 것을 문제로 적고 건너뛴다 — 예측이 조용히 섞이지 않게; 리뷰 1 Minor 7)."""
    from robo_jev.decisions.records import validate_record

    records = _read_jsonl(path)
    valid, problems, seen = [], {}, set()
    for record in records:
        found = validate_record(record)
        rid = str(record.get("id"))
        if rid in seen:
            found = found + [f"duplicate id {rid!r} (an earlier record has it)"]
        if found:
            if len(problems) < 20:
                problems.setdefault(rid, []).extend(found[:5])
            continue
        seen.add(rid)
        valid.append(record)
    return valid, {"read": len(records), "invalid": len(records) - len(valid), "problems": problems}


def _parser() -> argparse.ArgumentParser:
    from robo_jev.decisions.prompt import DEFAULT_FORMAT, DEFAULT_ORDERS, FORMATS, MODES, ORDERS, PROMPT_VERSION, PROMPT_VERSIONS
    from robo_jev.decisions.zero_shot import DEFAULT_BATCH, DEFAULT_GEN_BATCH, REASONING_MAX_NEW_TOKENS

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--model", help="Qwen/Qwen3.5-2B | Qwen/Qwen3.5-4B (가중치는 artifacts/models/) | tiny(검사 전용)")
    action.add_argument("--rescore", type=Path, nargs="+", help="저장한 예측 JSONL(여럿이면 서로 다른 배치로 합친다)에서 요약만 다시 계산 (CPU)")
    action.add_argument("--compare", type=Path, nargs=2, metavar=("A", "B"), help="두 예측 JSONL의 짝지은 비교 a − b (CPU)")
    parser.add_argument("--input", type=Path, required=True, help="simgen-request/0 JSONL")
    parser.add_argument("--out", type=Path, required=True, help="요약 JSON 경로 (옆에 .predictions.jsonl·.md)")
    parser.add_argument("--mode", choices=MODES, default="direct")
    parser.add_argument("--format", dest="fmt", choices=FORMATS, default=DEFAULT_FORMAT, help=f"프롬프트 형식 (기본 {DEFAULT_FORMAT})")
    parser.add_argument("--orders", choices=ORDERS, default=DEFAULT_ORDERS, help=f"보이는 순서의 배치 (기본 {DEFAULT_ORDERS})")
    parser.add_argument("--limit", type=int, default=None, help="쌍을 깨지 않는 무작위 표본 크기 (없으면 전부)")
    parser.add_argument("--seed", type=int, default=0, help="표본 고르기 seed")
    parser.add_argument("--shuffle-seed", type=int, default=1, help="선택지 순서 seed (균형 배치의 0번 배치 = 이 seed의 순서)")
    parser.add_argument("--shuffle-seeds", type=int, nargs="+", default=None,
                        help="(seeded 배치) seed 여럿으로 run을 여러 번 돌려 서로 다른 배치로 합친다 — 옛 방식")
    parser.add_argument("--batch", type=int, default=DEFAULT_BATCH)
    parser.add_argument("--gen-batch", type=int, default=DEFAULT_GEN_BATCH)
    parser.add_argument("--max-new-tokens", type=int, default=REASONING_MAX_NEW_TOKENS)
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("--title", default=None)
    parser.add_argument("--preview", action="store_true", help="보고서 머리에 '봉인된 시험 세트가 아닌 생성 표본의 미리보기'를 단다")
    parser.add_argument("--labels", nargs=2, default=("a", "b"), metavar=("LABEL_A", "LABEL_B"), help="--compare의 열 이름")
    parser.add_argument("--gpu-memory-fraction", type=float, default=DEFAULT_FRACTION)
    parser.add_argument("--prompt-version", choices=PROMPT_VERSIONS, default=PROMPT_VERSION,
                        help=f"프롬프트 판 (기본 {PROMPT_VERSION}; v0c-p2는 v0c 결과의 재현용)")
    parser.add_argument("--checkpoint", type=Path, default=None,
                        help="(--model과 함께) 학습한 가중치 — Task v0d의 체크포인트(LoRA adapter 또는 전체 가중치; 재개용·model-only 모두)")
    return parser


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
    from robo_jev.decisions.metrics import merge_runs, summarize_all
    from robo_jev.decisions.records import file_sha256
    from robo_jev.decisions.report import PREVIEW_NOTE

    records, validation = _load_valid(args.input)
    runs = [_read_jsonl(path) for path in args.rescore]
    predictions, merge = merge_runs(runs)
    ids = {str(p["id"]) for p in predictions}
    records = [r for r in records if str(r["id"]) in ids]
    first = predictions[0] if predictions else {}
    config = {
        "model": first.get("model"), "mode": first.get("mode"), "format": first.get("format"), "prompt_version": first.get("prompt_version"),
        "orders": first.get("orders") or ("seeded" if len(runs) == 1 else f"seeded × {len(runs)} runs, merged"),
        "shuffle_seeds": [run[0].get("shuffle_seed") for run in runs if run], "records": len(records), "input": str(args.input),
        "input_sha256": file_sha256(args.input), "predictions": [str(p) for p in args.rescore], "rescored": True,
        "script_version": SCRIPT_VERSION, "git": _git_sha(),
    }
    summary = summarize_all(records, predictions, n_boot=args.n_boot, seed=args.seed, merge=merge)
    payload = {"config": config, "validation": validation, "run": None, "summary": summary}
    _write_outputs(args.out, payload, predictions, title=args.title or f"v0c diagnosis (rescored) — {config['model']} {config['mode']}",
                   note=PREVIEW_NOTE if args.preview else None)
    _log(f"rescored {len(records)} records from {len(runs)} run file(s) → {args.out}")
    return 0


def _compare(args: argparse.Namespace) -> int:
    from robo_jev.decisions.metrics import compare_runs
    from robo_jev.decisions.records import file_sha256
    from robo_jev.decisions.report import PREVIEW_NOTE, render_comparison

    records, validation = _load_valid(args.input)
    a, b = _read_jsonl(args.compare[0]), _read_jsonl(args.compare[1])
    ids = {str(p["id"]) for p in a} | {str(p["id"]) for p in b}
    records = [r for r in records if str(r["id"]) in ids]  # 예측이 있는 기록만 (리뷰 1 Minor 8)
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
    _log(f"compared {comparison['records']} records (gate passed: {comparison['gate']['passed']}) → {args.out}")
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
    from robo_jev.decisions.metrics import merge_runs, summarize_all
    from robo_jev.decisions.prompt import fixed_text_digest
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
    model_label, checkpoint = args.model, None
    if args.checkpoint is not None:
        from robo_jev.decisions.train import apply_checkpoint

        checkpoint = apply_checkpoint(backbone, args.checkpoint, tokenizer_sha256=describe_tokenizer(tokenizer_id)["sha256"], prompt_version=args.prompt_version)
        checkpoint["sha256"] = file_sha256(args.checkpoint)
        model_label = f"{args.model}+{checkpoint['mode']}:{checkpoint['run_id']}@{checkpoint['step']}"
        _log(f"applied {checkpoint['mode']} checkpoint {args.checkpoint} (run {checkpoint['run_id']}, step {checkpoint['step']}) · memory {memory_report()}")
    seeds = list(args.shuffle_seeds) if args.shuffle_seeds else [args.shuffle_seed]
    if len(seeds) > 1 and args.orders != "seeded":
        raise SystemExit("--shuffle-seeds는 --orders seeded와 함께 쓴다 (균형 배치는 seed 하나로 모든 배치를 만든다)")
    started = time.perf_counter()
    runs = []
    for shuffle_seed in seeds:
        runs.append(diagnose_records(
            chosen, backbone, tokenizer, model_id=model_label, mode=args.mode, fmt=args.fmt, seed=shuffle_seed, orders=args.orders,
            batch=args.batch, gen_batch=args.gen_batch, max_new_tokens=args.max_new_tokens, log=_log, version=args.prompt_version,
        ))
    seconds = time.perf_counter() - started
    predictions, merge = merge_runs(runs) if len(runs) > 1 else (runs[0], None)
    prompts = sum(len(q["listings"]) for run in runs for p in run for q in p["questions"].values())
    import torch

    peak = int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else None
    summary = summarize_all(chosen, predictions, n_boot=args.n_boot, seed=args.seed, merge=merge)
    config = {
        "model": model_label, "mode": args.mode, "format": args.fmt, "orders": args.orders, "prompt_version": args.prompt_version,
        "prompt_digest": fixed_text_digest(args.prompt_version), "seed": args.seed, "shuffle_seed": seeds[0], "shuffle_seeds": seeds, "limit": args.limit,
        "records": len(chosen), "record_ids": [str(r["id"]) for r in chosen],
        "selection": f"pair-preserving random sample, seed {args.seed}, limit {args.limit}" if args.limit else "all valid records",
        "input": str(args.input), "input_sha256": file_sha256(args.input), "batch": args.batch, "gen_batch": args.gen_batch,
        "max_new_tokens": args.max_new_tokens if args.mode == "reasoning" else None, "n_boot": args.n_boot,
        "tokenizer": describe_tokenizer(tokenizer_id), "backbone_manifest": getattr(backbone, "manifest", None), "checkpoint": checkpoint,
        "script_version": SCRIPT_VERSION, "git": _git_sha(), "created": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
    }
    listings = [l for run in runs for p in run for q in p["questions"].values() for l in q["listings"]]
    run = {
        "seconds": round(seconds, 2), "load_seconds": round(load_seconds, 2), "prompts": prompts,
        "prompt_tokens": sum(l["prompt_tokens"] for l in listings), "scored_tokens": sum(l["scored_tokens"] for l in listings),
        "gpu": {"guard": guard, "memory_at_start": memory_start, "memory_at_end": memory_report(), "peak_allocated_bytes": peak},
    }
    payload = {"config": config, "validation": validation, "run": run, "summary": summary}
    title = args.title or f"v0c diagnosis — {model_label} {args.mode} ({args.fmt})"
    _write_outputs(args.out, payload, predictions, title=title, note=PREVIEW_NOTE if args.preview else None)
    _log(f"{len(predictions)} records, {prompts} prompts in {seconds:.1f} s → {args.out} · memory at end {run['gpu']['memory_at_end']}")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.checkpoint is not None and args.model is None:
        raise SystemExit("--checkpoint는 --model(기반 모델)과 함께 쓴다")
    if args.rescore is not None:
        return _rescore(args)
    if args.compare is not None:
        return _compare(args)
    return _run(args)


if __name__ == "__main__":
    raise SystemExit(main())
