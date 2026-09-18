"""origin group → split (docs/04 §5).

규칙은 두 단계다.

1. **의미 계열·분야 holdout을 먼저 통째로 제외한다.** 제외된 group은 ``ood``다.
2. 남은 group을 **stable hash**로 ``train/dev/calibration/test``에 배정한다.

해시는 sha256이다. 파이썬 내장 `hash()`는 문자열마다 프로세스 seed가 섞이므로
(``PYTHONHASHSEED``) 실행마다 split이 뒤집힌다 — 배포한 데이터의 계보가 깨진다.

분할은 표현 증강·후보 재배열보다 **먼저** 한다. 파생본(번역·질문 재표현·후보 재배열·
counterfactual sibling, 스트림의 틱·rollout·구간·재라벨링)은 부모의 origin group을
그대로 쓰므로 자동으로 같은 split을 받는다. 로봇 장면 계열도 에피소드를 만들기 전에
이 함수로 배정한다(:mod:`robo_jev.data` 참고).
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "DEFAULT_WEIGHTS",
    "HASH_RESOLUTION",
    "HOLDOUT_SPLIT",
    "SplitPolicy",
    "assign_split",
    "group_point",
]

#: holdout된 group이 받는 split.
HOLDOUT_SPLIT = "ood"

#: 기본 비중 (docs/04 §5 표). 순서가 곧 해시 구간의 순서이므로 바꾸면 계보가 깨진다.
DEFAULT_WEIGHTS: tuple[tuple[str, int], ...] = (
    ("train", 70),
    ("dev", 10),
    ("calibration", 10),
    ("test", 10),
)

#: 해시를 떨어뜨릴 구간 수. 비중이 정수 퍼센트가 아니어도 되게 넉넉히 잡는다.
HASH_RESOLUTION = 1_000_000


def group_point(origin_group: str) -> int:
    """group id를 ``[0, HASH_RESOLUTION)`` 안의 한 점으로 보낸다."""
    digest = hashlib.sha256(origin_group.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % HASH_RESOLUTION


@dataclass(frozen=True)
class SplitPolicy:
    """holdout 목록과 비중. 설정 파일(`configs/data/pilot.yaml`)에서 만든다."""

    weights: tuple[tuple[str, int], ...] = DEFAULT_WEIGHTS
    holdout_groups: frozenset[str] = field(default_factory=frozenset)
    holdout_prefixes: tuple[str, ...] = ()
    holdout_domains: frozenset[str] = field(default_factory=frozenset)

    def __post_init__(self) -> None:
        if not self.weights:
            raise ValueError("split weights가 비어 있다")
        allowed = {name for name, _ in DEFAULT_WEIGHTS}
        for name, weight in self.weights:
            if name not in allowed:
                raise ValueError(
                    f"해시 분할에 쓸 수 없는 split이다: {name!r} (가능: {sorted(allowed)}) "
                    f"— {HOLDOUT_SPLIT}는 holdout 목록으로만 만든다"
                )
            if not isinstance(weight, int) or isinstance(weight, bool) or weight <= 0:
                raise ValueError(f"split weights는 양의 정수여야 한다: {name}={weight!r}")

    @classmethod
    def from_config(cls, config: Mapping[str, Any] | None) -> SplitPolicy:
        """설정의 ``split`` 절을 정책으로 바꾼다. 없는 키는 기본값을 쓴다."""
        config = config or {}
        raw_weights = config.get("weights")
        weights = (
            tuple((str(name), weight) for name, weight in raw_weights.items())
            if raw_weights
            else DEFAULT_WEIGHTS
        )
        return cls(
            weights=weights,
            holdout_groups=frozenset(config.get("holdout_groups") or ()),
            holdout_prefixes=tuple(config.get("holdout_prefixes") or ()),
            holdout_domains=frozenset(config.get("holdout_domains") or ()),
        )

    def is_holdout(self, origin_group: str) -> bool:
        """의미 계열·분야 holdout인가. 해시 분할보다 **먼저** 본다."""
        domain = origin_group.split("/", 1)[0]
        return (
            origin_group in self.holdout_groups
            or domain in self.holdout_domains
            or any(origin_group.startswith(prefix) for prefix in self.holdout_prefixes)
        )

    def assign(self, origin_group: str) -> str:
        if self.is_holdout(origin_group):
            return HOLDOUT_SPLIT
        total = sum(weight for _, weight in self.weights)
        point = group_point(origin_group)
        edge = 0
        for name, weight in self.weights:
            edge += weight * HASH_RESOLUTION // total
            if point < edge:
                return name
        return self.weights[-1][0]  # 나눗셈 나머지 구간


#: 기본 정책 (holdout 없음).
DEFAULT_POLICY = SplitPolicy()


def assign_split(origin_group: str, policy: SplitPolicy | None = None) -> str:
    """origin group이 속할 split. 같은 group은 언제 어디서 불러도 같은 값을 받는다."""
    return (policy or DEFAULT_POLICY).assign(origin_group)
