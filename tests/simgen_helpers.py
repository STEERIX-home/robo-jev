"""simgen 시험 공용 도우미: 표본 레코드(한 프로세스에서 한 번만 만든다)와 손으로 고친 관측.

2,000개 표본은 지시서 §5의 '생성한 2,000개 전부' 검사(행동 choice argmax, 분야 균형, 여유 분포, 핵심 noul 평균)가 함께 쓴다.
"""

from __future__ import annotations

import functools
from dataclasses import replace

from robo_jev.simgen.catalog import default_config
from robo_jev.simgen.generate import build_moment, generate_batch

SAMPLE_N = 2000
SAMPLE_SEED = 0


@functools.lru_cache(maxsize=4)
def sample_records(n: int = SAMPLE_N, seed: int = SAMPLE_SEED) -> tuple[dict, ...]:
    return tuple(generate_batch(n, base_seed=seed, cfg=default_config()))


def moment(seed: int, domain: str, **forced):
    """강제한 틀의 판단 순간 하나(Moment)."""
    return build_moment(default_config(), seed, domain, forced)


def with_reading(reading, value: float):
    return replace(reading, value=value)
