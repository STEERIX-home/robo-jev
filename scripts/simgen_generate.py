"""Task v0a — 시뮬레이션 판단 데이터 엔진으로 요청 N개를 만들고(JSONL) 통계(stats.json)를 쓴다.

    uv run python scripts/simgen_generate.py --n 2000 --seed 0 \
        --out artifacts/simgen/v0a/sample-2000.jsonl --stats artifacts/simgen/v0a/stats.json

2,000개 이상은 지시서 §7대로 systemd-run 유닛(MemoryMax, choom)으로 띄운다. GPU·모델을 쓰지 않는다. 같은 (seed, n, 커밋)이면 같은 바이트다.
통계에는 생성 속도(레코드/초 — 생성·검사·통계를 뺀 순수 생성), 검사기 오류 수, 커밋과 설정 버전이 함께 든다.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from robo_jev.simgen import GENERATOR_VERSION, SCHEMA  # noqa: E402
from robo_jev.simgen.catalog import default_config  # noqa: E402
from robo_jev.simgen.generate import generate_batch, git_short_sha  # noqa: E402
from robo_jev.simgen.record import dumps, validate_record  # noqa: E402
from robo_jev.simgen.stats import dataset_stats  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--n", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--k", type=int, default=None, help="사후 표본 수(기본: catalog의 posterior_k)")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--stats", type=Path, required=True)
    args = parser.parse_args(argv)

    cfg = default_config()
    git = git_short_sha(REPO)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    records = []
    start = time.perf_counter()
    for record in generate_batch(args.n, base_seed=args.seed, cfg=cfg, git=git, k=args.k):
        records.append(record)
    seconds = time.perf_counter() - start
    with args.out.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(dumps(record) + "\n")
    errors = {r["id"]: validate_record(r) for r in records}
    errors = {k: v for k, v in errors.items() if v}
    stats = {
        "schema": SCHEMA,
        "generator": {"version": GENERATOR_VERSION, "git": git, **cfg.versions, "base_seed": args.seed, "n": args.n, "posterior_k": args.k or cfg.catalog["posterior_k"]},
        "output": {"jsonl": str(args.out), "bytes": args.out.stat().st_size},
        "validation": {"records_with_errors": len(errors), "examples": dict(list(errors.items())[:5])},
        **dataset_stats(records, cfg, seconds),
    }
    args.stats.parent.mkdir(parents=True, exist_ok=True)
    args.stats.write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"{len(records)} records in {seconds:.1f} s ({len(records) / seconds:.1f}/s), validation errors: {len(errors)} → {args.out}, {args.stats}")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
