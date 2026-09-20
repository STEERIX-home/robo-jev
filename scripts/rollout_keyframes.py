"""키프레임 rollout을 돌려 라벨과 제작 비용을 적는다 (docs/04 §4, Task 3c-2).

    uv run python scripts/rollout_keyframes.py --dataset artifacts/datasets/d1-robot/batch-0 --limit 100 [--workers 8]

첫 100개는 비용 산정용이다. 128,000개(D1) 전체는 후보 공간이 확정된 뒤에 돌린다. `--limit 1000 --out <dir>`이 128k 전
관문 sweep이고, `--summarise <dir>`은 rollout 없이 그 출력의 요약(밀기 성공률을 접근 시간·시작 거리로 조건화한 표, censoring
사유, 키프레임 종류 혼합, 128k 산정)을 `sweep-summary.json`으로 쓴다.
"""

import sys

from robo_jev.data.rollouts import main

if __name__ == "__main__":
    sys.exit(main())
