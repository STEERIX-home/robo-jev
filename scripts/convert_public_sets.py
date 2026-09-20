#!/usr/bin/env python
"""BoolQ validation·MultiNLI dev_matched → `judgment-v0` state_first 레코드 (analysis-nimble §3-7; 학습 manifest에 넣지 않는다) —
`robo_jev.data.public_sets.main`의 얇은 겉옷.

    uv run --with pyarrow python scripts/convert_public_sets.py --out artifacts/datasets/public
"""

import sys

from robo_jev.data.public_sets import main

if __name__ == "__main__":
    sys.exit(main())
