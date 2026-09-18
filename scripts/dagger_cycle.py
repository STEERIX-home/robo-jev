"""DAgger식 재라벨링 사이클 (docs/04 §7): 정책이 실행하고 전문가가 `labels`에만 답한다.

    uv run python scripts/dagger_cycle.py --episodes 200 --out artifacts/datasets/d1-robot/dagger-0 [--policy rule_judge]

오늘의 정책 대역은 규칙 기준군이다. 학습 모델의 클라이언트는 `policy(request) -> results` 인터페이스로 붙는다.
"""

import sys

from robo_jev.data.dagger import main

if __name__ == "__main__":
    sys.exit(main())
