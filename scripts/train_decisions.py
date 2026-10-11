#!/usr/bin/env python
"""Task v0d — 판단 요청으로 Qwen 학습: 진단 도구(v0c)와 같은 프롬프트·같은 코드 토큰으로 (모듈 :mod:`robo_jev.decisions.train`).

    # 학습 (GPU — artifacts/scratch/v0d/gpu.sh로만, 한 번에 하나)
    uv run python scripts/train_decisions.py train --config configs/train/decisions-v0-lora.yaml --set train_input=<train.jsonl> --set run_name=<이름>
    uv run python scripts/train_decisions.py train --config configs/train/decisions-v0-lora.yaml --set train_input=… --resume <run>/checkpoint.pt
    # 처리량 측정 = 체크포인트 없는 짧은 run
    uv run python scripts/train_decisions.py train --config configs/train/decisions-v0-full.yaml --set train_input=… --set max_steps=20 \\
        --set save_checkpoints=false --set dev_every=0 --set run_name=<이름>
    # 레코드 단위(쌍은 함께) 학습/평가 나눔 (CPU — 기록을 읽는 일은 systemd 유닛으로)
    uv run python scripts/train_decisions.py split --input <sample.jsonl> --eval-records 400 --seed 0 --out-dir <dir>
    # 프롬프트 길이 분포 (CPU): 질문마다 한 배치 · 학습 계획(epoch 0, 요청마다 q개)
    uv run python scripts/train_decisions.py lengths --input <sample.jsonl> --out <lengths.json>

``train``의 GPU 규칙(GB10 통합 메모리): 첫 CUDA 할당 전에 울타리(`--gpu-memory-fraction`, 기본 0.6), 적재 전에 여유 ≥ 60 GiB(근거는
:data:`robo_jev.decisions.train.MIN_FREE_TRAIN_BYTES`), 처음·적재 뒤·끝의 메모리 보고(run의 manifest·metrics.json에 남는다). 체크포인트를
쓰는 run은 시작 전에도 run 디렉터리 파일 시스템의 여유가 `min_free_disk_gb`(기본 60 GB) 이상인지 본다(쓰기마다 학습기가 다시 본다).
``--set model_id=tiny``는 검사 전용 소형 난수 Qwen(CPU)이다(``--set device=cpu --set dtype=float32``와 함께).
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import random
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from robo_jev.gpu import DEFAULT_FRACTION, limit_gpu_memory, memory_report, require_free  # noqa: E402

SCRIPT_VERSION = "v0d-train-2"


def _log(message: str) -> None:
    print(f"[v0d {dt.datetime.now().strftime('%H:%M:%S')}] {message}", file=sys.stderr, flush=True)


def _read_config(path: Path | None, assignments: list[str]) -> dict[str, Any]:
    import yaml

    config = (yaml.safe_load(path.read_text(encoding="utf-8")) or {}) if path else {}
    for assignment in assignments:
        key, separator, value = assignment.partition("=")
        if not separator:
            raise SystemExit(f"--set은 KEY=VALUE 꼴이어야 한다: {assignment!r}")
        config[key] = yaml.safe_load(value)
    return config


def cmd_train(args: argparse.Namespace) -> int:
    # 울타리는 첫 CUDA 할당 전에 (robo_jev.gpu) — 설정·기록 읽기는 CUDA를 건드리지 않는다
    guard = limit_gpu_memory(args.gpu_memory_fraction)
    memory_start = memory_report()
    _log(f"gpu guard {guard} · memory at start {memory_start}")
    from robo_jev.decisions.train import MIN_FREE_TRAIN_BYTES, TINY_MODEL, DecisionTrainer, disk_free_bytes, resolve_config

    config = _read_config(args.config, args.set)
    resolved = resolve_config(config)
    if resolved["save_checkpoints"]:
        out_dir = Path(resolved["out_dir"])
        free = disk_free_bytes(out_dir)
        floor = float(resolved["min_free_disk_gb"]) * 1e9
        if free < floor:
            raise SystemExit(f"train: {out_dir}의 디스크 여유 {free / 1e9:.1f} GB < {floor / 1e9:.0f} GB — 체크포인트를 쓸 run을 시작하지 않는다")
        _log(f"disk free at {out_dir}: {free / 1e9:.1f} GB (floor {floor / 1e9:.0f} GB)")
    if resolved["model_id"] != TINY_MODEL:
        require_free(MIN_FREE_TRAIN_BYTES, what=f"decisions training ({resolved['mode']}) of {resolved['model_id']}")
    trainer = DecisionTrainer(config, resume=args.resume, log=_log)
    trainer.manifest["run_env"] = {
        "script_version": SCRIPT_VERSION, "argv": sys.argv[1:], "gpu_guard": guard, "memory_at_start": memory_start,
        "memory_after_load": memory_report(),
    }
    _log(
        f"run {trainer.run_id}: {trainer.config['mode']} · {len(trainer.train_records)} train / {len(trainer.dev_records)} dev records · "
        f"{trainer.examples_per_epoch} examples/epoch · {trainer.steps_per_epoch} steps/epoch · max_steps {trainer.max_steps} · "
        f"trainable {trainer.manifest['model']['trainable_parameters']:,} · memory {trainer.manifest['run_env']['memory_after_load']}"
    )
    result = trainer.run()
    trainer.manifest["run_env"]["memory_at_end"] = memory_report()
    trainer.write_metrics()
    summary = trainer.summary()
    _log(f"{result['status']} at step {result['step']} · steady {summary['tokens_per_second_steady']} tokens/s · peak {summary['peak_allocated_bytes']}")
    print(json.dumps({**result, "summary": {k: v for k, v in summary.items() if k != "last_dev"}}, ensure_ascii=False, default=float))
    return 0


def cmd_split(args: argparse.Namespace) -> int:
    """레코드 단위 학습/평가 나눔 — 평가 쪽은 쌍을 깨지 않는 무작위 표본(:func:`select_records`), 나머지가 학습. 파일 순서를 지킨다."""
    from robo_jev.decisions.records import file_sha256, pair_of, select_records
    from robo_jev.decisions.train import load_training_records

    records, validation = load_training_records(args.input)
    evaluation = select_records(records, args.eval_records, seed=args.seed)
    chosen = {str(r["id"]) for r in evaluation}
    train = [r for r in records if str(r["id"]) not in chosen]
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    stem = args.input.stem
    paths = {"train": out / f"{stem}.train.jsonl", "eval": out / f"{stem}.eval.jsonl"}
    for name, rows in (("train", train), ("eval", evaluation)):
        with paths[name].open("w", encoding="utf-8") as handle:
            for record in rows:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    broken = sum(1 for r in train if pair_of(r)[0] in chosen) + sum(1 for r in evaluation if pair_of(r)[0] and pair_of(r)[0] not in chosen)
    manifest = {
        "script_version": SCRIPT_VERSION, "input": str(args.input), "input_sha256": file_sha256(args.input), "validation": validation,
        "seed": args.seed, "eval_records_requested": args.eval_records, "rule": "pair-preserving random sample (select_records) → eval; the rest → train",
        "train": {"path": str(paths["train"]), "records": len(train), "sha256": file_sha256(paths["train"])},
        "eval": {"path": str(paths["eval"]), "records": len(evaluation), "sha256": file_sha256(paths["eval"]), "ids": sorted(chosen)},
        "pairs_split_across": broken,
    }
    (out / f"{stem}.split.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    _log(f"{len(records)} valid records → train {len(train)} · eval {len(evaluation)} (pairs split across: {broken}) → {out}")
    return 0


def cmd_lengths(args: argparse.Namespace) -> int:
    """프롬프트 길이 분포 — (a) 질문마다 seed 배치 하나(진단의 0번 배치), 종류별; (b) 학습 계획 epoch 0(요청마다 q개, 증강)의 예."""
    from robo_jev.decisions.examples import build_examples, epoch_plan, length_summary
    from robo_jev.decisions.prompt import record_prompts
    from robo_jev.decisions.records import file_sha256
    from robo_jev.decisions.train import load_training_records
    from robo_jev.model.tokenizer import available_tokenizer, describe_tokenizer, load_tokenizer

    records, validation = load_training_records(args.input)
    tokenizer_id = available_tokenizer()[0]
    tokenizer = load_tokenizer(tokenizer_id)
    per_question: dict[str, list[int]] = {}
    for record in records:
        for prompt in record_prompts(record, tokenizer, seed=1, fmt=args.format, mode="direct", orders="seeded", version=args.prompt_version):
            per_question.setdefault(prompt["type"], []).append(len(prompt["tokens"]))
    plan = epoch_plan(records, seed=args.seed, epoch=0, q=args.questions_per_request, augment=True)
    planned: dict[str, list[int]] = {}
    for index, name, order_seed in plan:
        (example,) = build_examples(records[index], [name], tokenizer, order_seed=order_seed, fmt=args.format, version=args.prompt_version)
        planned.setdefault(example["type"], []).append(len(example["tokens"]))
    everything = [n for values in per_question.values() for n in values]
    sampled = [n for values in planned.values() for n in values]
    rng = random.Random(0)
    payload = {
        "script_version": SCRIPT_VERSION, "input": str(args.input), "input_sha256": file_sha256(args.input), "validation": validation,
        "tokenizer": describe_tokenizer(tokenizer_id), "format": args.format, "prompt_version": args.prompt_version, "records": len(records),
        "every_question": {"all": length_summary(everything), **{kind: length_summary(values) for kind, values in sorted(per_question.items())}},
        "training_plan_epoch0": {
            "questions_per_request": args.questions_per_request, "seed": args.seed, "examples": len(plan),
            "all": length_summary(sampled), **{kind: length_summary(values) for kind, values in sorted(planned.items())},
        },
        "example_lengths": rng.sample(sampled, min(20, len(sampled))),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    _log(f"{len(records)} records · every question {payload['every_question']['all']} · plan {payload['training_plan_epoch0']['all']} → {args.out}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    train = sub.add_parser("train", help="학습 (GPU 유닛 안에서)")
    train.add_argument("--config", type=Path, default=None, help="설정 YAML (configs/train/decisions-v0-lora.yaml · -full.yaml)")
    train.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="설정 값 덮어쓰기 (값은 YAML로 읽는다)")
    train.add_argument("--resume", type=Path, default=None, help="이어갈 체크포인트 (같은 설정·같은 데이터여야 한다)")
    train.add_argument("--gpu-memory-fraction", type=float, default=DEFAULT_FRACTION)
    train.set_defaults(func=cmd_train)
    split = sub.add_parser("split", help="레코드 단위(쌍은 함께) 학습/평가 나눔 (CPU)")
    split.add_argument("--input", type=Path, required=True)
    split.add_argument("--eval-records", type=int, required=True)
    split.add_argument("--seed", type=int, default=0)
    split.add_argument("--out-dir", type=Path, required=True)
    split.set_defaults(func=cmd_split)
    lengths = sub.add_parser("lengths", help="프롬프트 길이 분포 (CPU)")
    lengths.add_argument("--input", type=Path, required=True)
    lengths.add_argument("--out", type=Path, required=True)
    lengths.add_argument("--format", default="chat", choices=("chat", "plain"))
    lengths.add_argument("--questions-per-request", type=int, default=2)
    lengths.add_argument("--seed", type=int, default=0)
    lengths.add_argument("--prompt-version", default="v0c-p3", choices=("v0c-p2", "v0c-p3"))
    lengths.set_defaults(func=cmd_lengths)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
