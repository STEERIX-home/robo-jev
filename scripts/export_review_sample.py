#!/usr/bin/env python
"""D1 검수 표본(≥ 500 질문, 층화, 이중 검수 100+)과 규약을 내보낸다 (docs/04 §6) — `robo_jev.data.review_sample.main`의 얇은 겉옷.

    uv run python scripts/export_review_sample.py --robot artifacts/datasets/d1-robot/d1-rollout-labels/manifest.json \\
        --single artifacts/datasets/d1/single/manifest.json --out artifacts/reports/d1-review-sample
"""

import sys

from robo_jev.data.review_sample import main

if __name__ == "__main__":
    sys.exit(main())
