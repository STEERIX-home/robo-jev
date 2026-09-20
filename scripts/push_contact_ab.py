#!/usr/bin/env python
"""밀기 접촉 거리 A/B — 같은 키프레임·snapshot·후보·paired seed의 밀기 job을 하네스 설정만 다른 팔마다 돌린다
(D1-prep 리뷰 2 N3·N7; 논리는 `robo_jev.data.push_ab`).

    uv run python scripts/push_contact_ab.py --dataset artifacts/datasets/d1-robot/batch-0 --workers 8 \\
        --arm 'old_30mm={"candidates": {"push_contact_mm": 30, "push_reach_mm": null}}' \\
        --arm 'h0.6={"candidates": {"push_reach_mm": null}}' --arm 'h0.7={}' \\
        --out artifacts/reports/d1-robot-push-contact-ab.json

`--arm NAME=JSON`의 JSON은 하네스 설정에 얹는 중첩 override다(`null`은 키 삭제, `{}`는 지금 설정 그대로). 팔의 yaml은 `--scratch`
(기본 artifacts/scratch/d1/push-ab) 아래에 남고 결과 JSON이 명령줄·override·sha256을 적는다. 배치는 지금 체크아웃의 버전이어야
한다(재생의 `ConfigMismatch`).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from robo_jev.data.push_ab import DEFAULT_SCRATCH, run


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="scripts/push_contact_ab.py", description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", type=Path, default=Path("artifacts/datasets/d1-robot/batch-0"))
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--arm", action="append", required=True, help="NAME=JSON (하네스 설정 override; 반복)")
    parser.add_argument("--directions", default=None, help="쉼표로 나눈 밀기 방향 (기본: 전부)")
    parser.add_argument("--limit-keyframes", type=int, default=None, help="앞 N개 키프레임만 (smoke)")
    parser.add_argument("--out", type=Path, default=Path("artifacts/reports/d1-robot-push-contact-ab.json"))
    parser.add_argument("--scratch", type=Path, default=DEFAULT_SCRATCH)
    parser.add_argument("--events", default="configs/sim/events.yaml")
    parser.add_argument("--generator-config", default="configs/data/d1_robot.yaml")
    args = parser.parse_args(argv)
    arms = {}
    for item in args.arm:
        name, _, text = item.partition("=")
        arms[name.strip()] = json.loads(text or "{}")
    directions = tuple(part.strip() for part in args.directions.split(",")) if args.directions else None
    report = run(
        args.dataset, arms, workers=args.workers, directions=directions, limit_keyframes=args.limit_keyframes, out=args.out,
        scratch=args.scratch, events_path=args.events, generator_config=args.generator_config,
        command=[sys.argv[0] if argv is None else "scripts/push_contact_ab.py", *(sys.argv[1:] if argv is None else argv)], log=sys.stdout,
    )
    for name, arm in report["arms"].items():
        print(f"{name}: {arm['outcomes']} wall {arm['wall_s']}s")
        for direction, row in arm["by_direction"].items():
            print(f"  {direction}: {row}")
    print(f"→ {args.out}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
