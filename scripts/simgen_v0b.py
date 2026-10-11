"""Task v0b — 학습 세트(train-50k), 봉인된 시험 세트 T1~T7(+ manifest), 검수용 예시 30개를 만든다.

    # 학습 세트(가치표 values-v0, 프로세스 여럿 — 레코드 순서·바이트는 프로세스 수와 무관)
    uv run python scripts/simgen_v0b.py train --out artifacts/simgen/v0b/train-50k.jsonl --workers 12
    # 사용자가 가치표 v0.1을 고르면 같은 한 줄로 다시(같은 seed, 정답만 v0.1 비용표로):
    uv run python scripts/simgen_v0b.py train --out artifacts/simgen/v0b/train-50k.values-v0.1.jsonl --values configs/simgen/values-v0.1.yaml
    # 시험 세트 T1~T7과 manifest.json(sha256·seed 범위·빼 둔 칸의 정의·생성기 git)
    uv run python scripts/simgen_v0b.py tests --out-dir artifacts/simgen/v0b/tests --workers 7
    # 검수용 예시 30개
    uv run python scripts/simgen_v0b.py examples --out artifacts/simgen/v0b/examples-30
    # Task v0.1a — 학습 세트 v01(같은 seed·같은 설정, 학습 문구 판 v01 + 사실 순서 섞기 0.5; 옆에 stats·manifest) — 시험 세트는 v0b 그대로
    uv run python scripts/simgen_v0b.py train --out artifacts/simgen/v01/train-50k.jsonl --phrasing v01 --workers 12
    # v0b 학습 세트와 레코드마다 맞대기(정답·참 세계·계산 과정·장부가 바이트 단위로 같고 요청 글만 다른가) + 비교 표
    uv run python scripts/simgen_v0b.py compare --v0b artifacts/simgen/v0b/train-50k.jsonl --v01 artifacts/simgen/v01/train-50k.jsonl \
        --out artifacts/simgen/v01/train-50k.compare.json

지시서 §9대로 systemd-run 유닛(MemoryMax, choom -n 1000, CUDA 숨김)으로 띄운다. GPU·모델을 쓰지 않는다.
"""

from __future__ import annotations

import argparse
import hashlib
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
    values_path, units, part, git, phrasing, fact_shuffle = task
    cfg = default_config(Path(values_path))
    split = split_for("train")
    replaced = []
    with open(part, "w", encoding="utf-8") as handle:
        for unit in units:
            for shift in (0, *_SPARE):
                try:
                    records = generate_unit(cfg, {**unit, "seed": unit["seed"] + shift}, split=split, git=git, id_prefix=ID_PREFIX, phrasing=phrasing,
                                            fact_shuffle=fact_shuffle)  # fmt: skip
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
    fact_shuffle = args.fact_shuffle if args.fact_shuffle is not None else (0.5 if args.phrasing == "v01" else 0.0)
    tasks = [(str(args.values.resolve()), units[i : i + size], str(tmp / f"part-{i // size:04d}.jsonl"), git, args.phrasing, fact_shuffle)
             for i in range(0, len(units), size)]  # fmt: skip
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
    if args.phrasing != "v0b" or fact_shuffle:
        from robo_jev.simgen.phrasing import fact_order_stats, wording_stats

        stats["phrasing"] = {"version": args.phrasing, "fact_shuffle": fact_shuffle}
        stats["fact_order"] = fact_order_stats(records)
        stats["wording"] = wording_stats(records)
    stats_path = args.out.with_suffix(".stats.json")
    stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    manifest_path = args.out.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(_train_manifest(args, git, start_seed, fact_shuffle, len(records), replaced), ensure_ascii=False, indent=2) + "\n",
                             encoding="utf-8")  # fmt: skip
    print(f"{len(records)} records in {seconds:.1f} s with {args.workers} workers, errors {len(errors)}, replaced seeds {len(replaced)} → {args.out}, {stats_path}, "
          f"{manifest_path}")  # fmt: skip
    return 1 if errors else 0


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _train_manifest(args, git: str | None, start_seed: int, fact_shuffle: float, n: int, replaced: list[str]) -> dict:
    """학습 세트의 manifest(파일 sha256, 생성기 git·버전, 문구 판, 사실 순서 섞기, seed 범위, 새 문구 표의 sha256)."""
    from robo_jev.simgen.phrasing import bank_digest

    cfg = default_config(args.values.resolve())
    return {
        "file": args.out.name, "sha256": _sha256(args.out), "records": n, "bytes": args.out.stat().st_size, "schema": SCHEMA,
        "generator": {"version": GENERATOR_VERSION, "git": git, **cfg.versions}, "phrasing": args.phrasing, "fact_shuffle": fact_shuffle,
        "split": "train", "seed_start": start_seed, "n": args.n, "replaced_seeds": replaced,
        "wording_banks_sha256": {"en": bank_digest("en"), "ko": bank_digest("ko")} if args.phrasing != "v0b" else None,
        "note": "Training data only. Sealed test sets T1-T7 stay the v0b files (artifacts/simgen/v0b/tests, v0b phrasing, fixed fact order).",
    }  # fmt: skip


def cmd_compare(args) -> int:
    """v0b 학습 세트와 v01 학습 세트를 레코드마다 맞댄다: 요청 글 말고는 모두 같은가(정답은 바이트 단위로), 상태 글은 같은 줄을 담나."""
    from collections import Counter

    from robo_jev.simgen.phrasing import without_text

    counts = Counter()
    bad: list[str] = []
    with args.v0b.open(encoding="utf-8") as a_handle, args.v01.open(encoding="utf-8") as b_handle:
        for line_a, line_b in zip(a_handle, b_handle, strict=True):
            a, b = json.loads(line_a), json.loads(line_b)
            counts["records"] += 1
            if json.dumps(a["labels"], ensure_ascii=False) != json.dumps(b["labels"], ensure_ascii=False):
                bad.append(f"labels differ: {a['id']}")
            if without_text(a) != without_text(b):
                bad.append(f"record differs beyond the request text: {a['id']}")
            state_a, state_b = a["request"]["state"], b["request"]["state"]
            if state_a != state_b:
                counts["state_differs"] += 1
                if not b["meta"].get("fact_order", {}).get("shuffled"):
                    bad.append(f"state differs without a fact shuffle: {a['id']}")
                if len(state_a) != len(state_b) or Counter(state_a) != Counter(state_b):
                    bad.append(f"shuffled state is not a reordering: {a['id']}")
            elif b["meta"].get("fact_order", {}).get("changed"):
                bad.append(f"fact_order.changed but the state is the same: {a['id']}")
            qa, qb = a["request"]["questions"], b["request"]["questions"]
            counts["questions"] += len(qa)
            counts["instructions_differ"] += sum(qa[n]["instructions"] != qb[n]["instructions"] for n in qa)
            for n in qa:
                ca, cb = qa[n].get("criteria"), qb[n].get("criteria")
                if isinstance(ca, dict):
                    counts["options"] += len(ca)
                    counts["option_texts_differ"] += sum(ca[k] != cb[k] for k in ca)
                elif ca != cb:
                    bad.append(f"score anchors differ: {a['id']} {n}")
    out = {"v0b": {"path": str(args.v0b), "sha256": _sha256(args.v0b)}, "v01": {"path": str(args.v01), "sha256": _sha256(args.v01)}, "counts": dict(counts),
           "violations": len(bad), "examples": bad[:20]}  # fmt: skip
    if args.v0b_stats and args.v01_stats:
        sa, sb = (json.loads(p.read_text(encoding="utf-8")) for p in (args.v0b_stats, args.v01_stats))
        same = {}
        for key in ("records", "shares", "coverage", "flags", "questions", "labels", "margins", "text_length", "truth_check", "korean_share", "group_shares"):
            same[key] = sa.get(key) == sb.get(key)
        out["stats_identical"] = same
    args.out.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: out[k] for k in ("counts", "violations")} | ({"stats_identical": out["stats_identical"]} if "stats_identical" in out else {}), ensure_ascii=False))
    return 1 if bad else 0


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
    train.add_argument("--phrasing", choices=("v0b", "v01"), default="v0b", help="학습 문구 판(기본 v0b; v01 = 학습 묶음을 넓힌 판 — Task v0.1a)")
    train.add_argument("--fact-shuffle", type=float, default=None, help="사실 순서 섞기 확률(기본: v01이면 0.5, v0b면 0)")
    compare = sub.add_parser("compare")
    compare.add_argument("--v0b", type=Path, required=True)
    compare.add_argument("--v01", type=Path, required=True)
    compare.add_argument("--v0b-stats", type=Path)
    compare.add_argument("--v01-stats", type=Path)
    compare.add_argument("--out", type=Path, required=True)
    tests = sub.add_parser("tests")
    tests.add_argument("--out-dir", type=Path, required=True)
    tests.add_argument("--workers", type=int, default=7)
    tests.add_argument("--sets", nargs="*")
    examples = sub.add_parser("examples")
    examples.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    return {"train": cmd_train, "tests": cmd_tests, "examples": cmd_examples, "compare": cmd_compare}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
