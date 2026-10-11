"""세계와 관측이 함께 쓰는 중립 타입: 수치 보고(Reading), 로봇 자신의 사양(RobotSpec), 굴리기 입력(Batch), 난수 갈래.

* :class:`Reading` — 글에 쓰이는 수치 하나. ``value``는 이미 보고 해상도로 반올림한 값(글이 보여 주는 값 그대로)이고, 사후 표본은
  감각 잡음 σ와 반올림을 합친 ``sigma_eff``로 그 값 주변에서 뽑는다.
* :class:`RobotSpec` — 로봇 자신의 값(속도·감속 세 단계·반응 지연·반지름 …). 사양과 고유 감각이라 **관측된다** — 글에 적힌 값이
  곧 참값이다(그래서 세계와 관측이 같은 객체를 나눠 쓴다). 숨은 값은 하나도 없다.
* :class:`Batch` — 굴리기의 입력: 세계 K개(사후 표본) 또는 1개(참 세계)의 값 배열. 열쇠는 장면이 정한다.
* :func:`stream` — 레코드 seed에서 이름 붙은 난수 갈래(``SeedSequence([seed, crc32(이름) …])``). 갈래를 하나 더해도 다른 갈래의 수가
  바뀌지 않는다 — 같은 seed면 같은 레코드(바이트 단위).
"""

from __future__ import annotations

import math
import zlib
from dataclasses import dataclass, field

import numpy as np

__all__ = ["FILL_BINS", "Batch", "Reading", "RobotSpec", "decimals_for", "fill_bin_index", "fill_bins", "round_to", "stream"]


#: 채움을 말로 줄 때의 구간(하한, 상한, 문구). 관측 값은 구간 가운데, 해상도는 구간 폭이다. 상태 choice(채움 구간)의 선택지도 이 구간을 쓴다.
FILL_BINS = (
    (0.00, 0.30, "less than a third full"),
    (0.30, 0.42, "about a third full"),
    (0.42, 0.58, "about half full"),
    (0.58, 0.70, "about two-thirds full"),
    (0.70, 0.82, "about three-quarters full"),
    (0.82, 0.92, "almost full"),
    (0.92, 0.98, "nearly to the brim"),
    (0.98, 1.00, "full to the brim"),
)
_FILL_EDGES = np.array([hi for _, hi, _ in FILL_BINS[:-1]])


def fill_bins(fill) -> np.ndarray:
    """채움 비율 → FILL_BINS의 구간 번호(배열)."""
    return np.searchsorted(_FILL_EDGES, np.asarray(fill, dtype=float), side="right").astype(float)


def fill_bin_index(fill: float) -> int:
    return int(fill_bins(np.array([fill]))[0])


def stream(seed: int, *names: object) -> np.random.Generator:
    """레코드 seed의 이름 붙은 갈래."""
    seed = int(seed)
    keys = [seed & 0xFFFFFFFF, (seed >> 32) & 0xFFFFFFFF] + [zlib.crc32(str(n).encode("utf-8")) for n in names]
    return np.random.default_rng(np.random.SeedSequence(keys))


def decimals_for(resolution: float) -> int:
    """해상도를 정확히 보이는 데 필요한 소수 자릿수(0.05 → 2, 0.1 → 1, 1 → 0)."""
    if resolution >= 1:
        return 0
    return max(0, int(math.ceil(-math.log10(resolution) - 1e-9)))


def round_to(x: float, resolution: float) -> float:
    """해상도 격자로 반올림(부동소수 꼬리 없이)."""
    return round(round(float(x) / resolution) * resolution, decimals_for(resolution) + 1)


@dataclass(frozen=True)
class Reading:
    """글에 쓰이는 수치 하나: 보고된 값(반올림 끝), 감각 잡음 σ, 보고 해상도, 보고 바닥.

    ``floor``가 있으면 날 값이 바닥보다 작을 때 바닥으로 보고한다(속도 0.00 m/s·간격 0.00 m 같은 글이 나오지 않게, 리뷰 1 m7).
    바닥에 붙은 보고는 '그 이하'라는 검열된 관측이다 — 사후 표본이 그렇게 다룬다(:attr:`censored`)."""

    value: float
    sigma: float
    resolution: float
    floor: float | None = None

    @property
    def sigma_eff(self) -> float:
        """사후 표본의 σ: 감각 잡음과 반올림(균등, 분산 res²/12)을 합친 것."""
        return math.sqrt(self.sigma * self.sigma + self.resolution * self.resolution / 12.0)

    @property
    def censored(self) -> bool:
        """보고가 바닥에 붙었나(날 값 ≤ 바닥 + 해상도/2였다)."""
        return self.floor is not None and self.value <= self.floor + 1e-9


@dataclass(frozen=True)
class RobotSpec:
    """로봇 자신의 사양·상태(관측됨 = 참값). 속도·가속도 m/s·m/s², 시간 s, 길이 m."""

    body: str
    speed: float                    # 지금 속도(멈춘 변형이면 0)
    cruise: float                   # 순항 속도(멈춘 변형에서 출발 목표; 그 밖에는 speed와 같다)
    latency: float
    decel_gentle: float
    decel_moderate: float
    decel_hard: float
    accel: float
    radius: float
    can_sidestep: bool
    sidestep_offset: float = 0.0
    sidestep_duration: float = 1.0
    sidestep_speed_ratio: float = 0.6
    slow_ratio: float = 0.5
    creep_ratio: float = 0.3
    retract_speed: float = 0.0      # 팔만
    loaded: bool = False            # 지게차가 짐을 실었다(감속 세 단계에 이미 반영)
    passing_clearance: float = 0.2  # head-on에서 필요한 폭

    def decel(self, level: str) -> float:
        return {"gentle": self.decel_gentle, "moderate": self.decel_moderate, "hard": self.decel_hard}[level]


@dataclass(frozen=True)
class Batch:
    """세계 K개의 값 배열(열쇠 → (K,) 배열). ``weights``가 있으면 세계마다의 사후 무게(합 1) — 이산 사후를 정확히 나열할 때(P4 상태,
    P6 있음/없음; Task v0b). 없으면 K개가 같은 무게(표본)."""

    values: dict[str, np.ndarray] = field(default_factory=dict)
    weights: np.ndarray | None = None

    @property
    def size(self) -> int:
        return len(next(iter(self.values.values())))

    def __getitem__(self, key: str) -> np.ndarray:
        return self.values[key]

    def get(self, key: str, default=None):
        return self.values.get(key, default)
