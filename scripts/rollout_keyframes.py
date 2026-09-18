"""키프레임 rollout을 돌려 라벨과 제작 비용을 적는다 (docs/04 §4, Task 3c-2).

    uv run python scripts/rollout_keyframes.py --dataset artifacts/datasets/d1-robot/batch-0 --limit 100 [--workers 8]

첫 100개는 비용 산정용이다. 128,000개(D1) 전체는 후보 공간이 확정된 뒤에 돌린다.
"""

import sys

from robo_jev.data.rollouts import main

if __name__ == "__main__":
    sys.exit(main())
