#!/usr/bin/env python
"""Task bench0 — bench-v0 그래프(SVG, 의존성 없음): 그림 1 산점도(T2–T4 physics skill × G2 정확도)와 그림 2 세트별 막대, 밝은·어두운 판.

    # 실제 결과 (scripts/bench.py score가 쓴 results.json) → docs/figures/ (커밋한다)
    uv run python scripts/plot_bench.py --results artifacts/bench/v0/results.json --out-dir docs/figures
    # 가짜 API 결과로 장치 시험 (워터마크; docs/figures에는 쓰지 않는다)
    uv run python scripts/plot_bench.py --results artifacts/bench/v0/results-fake.json --out-dir artifacts/bench/v0/figures-preview

README에 넣을 ``<picture>`` 조각(GitHub 테마에 맞는 판을 고른다)을 마지막에 찍는다. 그리는 규칙은 :mod:`robo_jev.bench.figures` 의 모듈 설명이다.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))


def main(argv: list[str] | None = None) -> int:
    from robo_jev.bench.figures import bars_svg, readme_snippet, scatter_svg

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--results", type=Path, required=True, help="scripts/bench.py score의 results.json")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--prefix", default="bench-v0")
    parser.add_argument("--top", type=int, default=None, help="그림 2에 넣을 API 모델 수 (T2–T4 physics skill 순; 기본: 전부)")
    parser.add_argument("--allow-fake", action="store_true", help="가짜 결과를 docs/ 아래에도 쓴다 (쓰지 않는다 — 시험 전용)")
    args = parser.parse_args(argv)
    results = json.loads(args.results.read_text(encoding="utf-8"))
    docs = (REPO / "docs").resolve()
    if results.get("fake") and not args.allow_fake and docs in [args.out_dir.resolve(), *args.out_dir.resolve().parents]:
        raise SystemExit("results hold FAKE API values — not writing them under docs/ (use another --out-dir)")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    names = []
    for kind, render in (("scatter", lambda t: scatter_svg(results, theme_name=t)), ("sets", lambda t: bars_svg(results, theme_name=t, top=args.top))):
        name = f"{args.prefix}-{kind}"
        names.append(name)
        for theme in ("light", "dark"):
            path = args.out_dir / f"{name}-{theme}.svg"
            path.write_text(render(theme), encoding="utf-8")
            print(f"wrote {path}", file=sys.stderr)
    folder = args.out_dir.resolve().relative_to(REPO.resolve()) if args.out_dir.resolve().is_relative_to(REPO.resolve()) else args.out_dir
    print(readme_snippet(names, folder=str(folder)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
