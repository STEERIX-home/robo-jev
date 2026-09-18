#!/usr/bin/env python
"""D1 로봇 에피소드 배치를 만든다 (docs/08 §9) — `robo_jev.data.robot_episodes.main`의 얇은 겉옷.

    uv run python scripts/generate_episodes.py --config configs/data/d1_robot.yaml \\
        --count 40 --out artifacts/datasets/d1-robot/batch-0 [--resume]

검사가 아니라 스크립트다: 40편 배치는 몇 분이 걸리고 artifacts/(git 무시)에 쓴다. 2편 smoke는
tests/test_robot_episodes.py가 한다.
"""

import sys

from robo_jev.data.robot_episodes import main

if __name__ == "__main__":
    sys.exit(main())
