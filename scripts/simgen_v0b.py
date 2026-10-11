"""Task v0b — 학습 세트(train-50k), 봉인된 시험 세트 T1~T7(+ manifest), 검수용 예시 30개를 만든다.

    # 학습 세트(가치표 values-v0, 프로세스 여럿 — 레코드 순서·바이트는 프로세스 수와 무관)
    uv run python scripts/simgen_v0b.py train --out artifacts/simgen/v0b/train-50k.jsonl --workers 12
    # 사용자가 가치표 v0.1을 고르면 같은 한 줄로 다시(같은 seed, 정답만 v0.1 비용표로):
    uv run python scripts/simgen_v0b.py train --out artifacts/simgen/v0b/train-50k.values-v0.1.jsonl --values configs/simgen/values-v0.1.yaml
    # 시험 세트 T1~T7과 manifest.json(sha256·seed 범위·빼 둔 칸의 정의·생성기 git)
    uv run python scripts/simgen_v0b.py tests --out-dir artifacts/simgen/v0b/tests --workers 7
    # 검수용 예시 30개
    uv run python scripts/simgen_v0b.py examples --out artifacts/simgen/v0b/examples-30

지시서 §9대로 systemd-run 유닛(MemoryMax, choom -n 1000, CUDA 숨김)으로 띄운다. GPU·모델을 쓰지 않는다.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from robo_jev.simgen import GENERATOR_VERSION, ID_PREFIX, SCHEMA  # noqa: E402
from robo_jev.simgen.catalog import TRAIN_GROUPS, VALUES_PATH, default_config  # noqa: E402
from robo_jev.simgen.generate import HeldOutError, generate_unit, git_short_sha, plan_units  # noqa: E402
from robo_jev.simgen.holdout import load_holdout, split_for  # noqa: E402
from robo_jev.simgen.record import dumps, validate_record  # noqa: E402
from robo_jev.simgen.stats import dataset_stats  # noqa: E402

TRAIN_N = 50_000
_SPARE = (500_000, 700_000, 900_000)     # 학습 단위를 만들 수 없을 때 바꿔 쓰는 seed(같은 단위는 늘 같은 대체 — 결정적)


def _train_chunk(task: tuple) -> tuple[str, int, list[str]]:
    """학습 단위 묶음 하나를 파일로(작업자 프로세스)."""
    values_path, units, part, git = task
    cfg = default_config(Path(values_path))
    split = split_for("train")
    replaced = []
    with open(part, "w", encoding="utf-8") as handle:
        for unit in units:
            for shift in (0, *_SPARE):
                try:
                    records = generate_unit(cfg, {**unit, "seed": unit["seed"] + shift}, split=split, git=git, id_prefix=ID_PREFIX)
                    break
                except HeldOutError:
                    replaced.append(f"{unit['seed']}+{shift}")
            else:
                raise HeldOutError(f"unit {unit} has no feasible seed")
            for record in records:
                handle.write(dumps(record) + "\n")
    return part, len(units), replaced


def cmd_train(args) -> int:
    holdout = load_holdout()
    git = git_short_sha(REPO)
    start_seed = holdout.seed_ranges["train"][0]
    units = plan_units(args.n, 0, groups=TRAIN_GROUPS, seed_start=start_seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.out.parent / f".{args.out.name}.parts"
    tmp.mkdir(exist_ok=True)
    size = (len(units) + args.workers * 4 - 1) // (args.workers * 4)
    tasks = [(str(args.values.resolve()), units[i : i + size], str(tmp / f"part-{i // size:04d}.jsonl"), git) for i in range(0, len(units), size)]
    start = time.perf_counter()
    with mp.get_context("fork").Pool(args.workers) as pool:
        results = pool.map(_train_chunk, tasks, chunksize=1)
    seconds = time.perf_counter() - start
    replaced = [r for _, _, rs in results for r in rs]
    with args.out.open("w", encoding="utf-8") as out:
        for part, _, _ in results:
            with open(part, encoding="utf-8") as handle:
                for line in handle:
                    out.write(line)
            Path(part).unlink()
    tmp.rmdir()
    records = [json.loads(line) for line in args.out.open(encoding="utf-8")]
    errors = {r["id"]: e for r in records if (e := validate_record(r))}
    cfg = default_config(args.values.resolve())
    stats = {
        "schema": SCHEMA,
        "generator": {"version": GENERATOR_VERSION, "git": git, **cfg.versions, "seed_start": start_seed, "n": args.n, "workers": args.workers,
                      "replaced_seeds": replaced},  # fmt: skip
        "output": {"jsonl": str(args.out), "bytes": args.out.stat().st_size},
        "validation": {"records_with_errors": len(errors), "examples": dict(list(errors.items())[:5])},
        **dataset_stats(records, cfg, seconds),
        "korean_share": round(sum(r["meta"]["language"] == "ko" for r in records) / len(records), 4),
        "group_shares": _shares(r["meta"]["group"] for r in records),
    }
    stats_path = args.out.with_suffix(".stats.json")
    stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"{len(records)} records in {seconds:.1f} s with {args.workers} workers, errors {len(errors)}, replaced seeds {len(replaced)} → {args.out}, {stats_path}")
    return 1 if errors else 0


def _shares(values) -> dict:
    from collections import Counter

    counts = Counter(values)
    total = sum(counts.values())
    return {k: round(v / total, 4) for k, v in sorted(counts.items())}


def _test_set(task: tuple) -> tuple:
    name, out_dir, git = task
    from robo_jev.simgen.testsets import T7_K, build_set, manifest_entry, write_set

    cfg = default_config()
    holdout = load_holdout()
    records, used = build_set(cfg, name, git=git, holdout=holdout)
    path = Path(out_dir) / f"{name}.jsonl"
    start = time.perf_counter()
    kept = []

    def keep(stream):
        for record in stream:
            kept.append(record)
            yield record

    written = write_set(path, keep(records))
    seconds = time.perf_counter() - start
    errors = sum(bool(validate_record(r)) for r in kept)
    k = T7_K if name == "T7" else cfg.catalog["posterior_k"]
    entry = manifest_entry(name, path, written, used, holdout, k)
    summary = {
        "records": len(kept), "seconds": round(seconds, 1), "validation_errors": errors,
        "domain_shares": _shares(r["meta"]["domain"] for r in kept), "group_shares": _shares(r["meta"]["group"] for r in kept),
        "style_shares": _shares(r["meta"]["style"] for r in kept), "korean_share": round(sum(r["meta"]["language"] == "ko" for r in kept) / len(kept), 4),
        "holdout_cells": _shares(h for r in kept for h in (r["meta"]["holdout"] or ["none"])),
        "key_noul_mean": round(sum(r["labels"][r["meta"]["key_question"]]["p_yes"] for r in kept) / len(kept), 4),
        "action_top_mean": round(sum(max(r["labels"][r["meta"]["recommendation"]["question"]]["probabilities"].values()) for r in kept) / len(kept), 4),
    }  # fmt: skip
    return name, entry, summary


def cmd_tests(args) -> int:
    from robo_jev.simgen.testsets import dump_manifest, manifest

    holdout = load_holdout()
    git = git_short_sha(REPO)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    names = args.sets or ["T1", "T2", "T3", "T4", "T5", "T6", "T7"]
    start = time.perf_counter()
    with mp.get_context("fork").Pool(min(args.workers, len(names))) as pool:
        results = pool.map(_test_set, [(n, str(args.out_dir), git) for n in names], chunksize=1)
    entries = {name: entry for name, entry, _ in results}
    summaries = {name: summary for name, _, summary in results}
    dump_manifest(args.out_dir / "manifest.json", manifest(entries, holdout, default_config(), git))
    (args.out_dir / "stats.json").write_text(json.dumps(summaries, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    bad = sum(s["validation_errors"] for s in summaries.values())
    print(f"sets {names} in {time.perf_counter() - start:.1f} s, validation errors {bad} → {args.out_dir}")
    return 1 if bad else 0


def cmd_examples(args) -> int:
    from robo_jev.simgen.examples import build_examples, dumps_jsonl, render_markdown

    cfg = default_config()
    records = build_examples(cfg, git=git_short_sha(REPO))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    Path(f"{args.out}.jsonl").write_text(dumps_jsonl(records), encoding="utf-8")
    Path(f"{args.out}.md").write_text(render_markdown(records, cfg), encoding="utf-8")
    errors = sum(bool(validate_record(r)) for r in records)
    print(f"{len(records)} examples, validation errors {errors} → {args.out}.jsonl, {args.out}.md")
    return 1 if errors else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    train = sub.add_parser("train")
    train.add_argument("--out", type=Path, required=True)
    train.add_argument("--n", type=int, default=TRAIN_N)
    train.add_argument("--workers", type=int, default=12)
    train.add_argument("--values", type=Path, default=VALUES_PATH, help="가치표(기본 values-v0; v0.1이면 configs/simgen/values-v0.1.yaml)")
    tests = sub.add_parser("tests")
    tests.add_argument("--out-dir", type=Path, required=True)
    tests.add_argument("--workers", type=int, default=7)
    tests.add_argument("--sets", nargs="*")
    examples = sub.add_parser("examples")
    examples.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    return {"train": cmd_train, "tests": cmd_tests, "examples": cmd_examples}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
