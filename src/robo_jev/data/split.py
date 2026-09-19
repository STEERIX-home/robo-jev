"""origin group → split (docs/04 §5).

규칙은 두 단계다.

1. **holdout을 먼저 통째로 제외한다.** 의미 계열(group·prefix)·분야(domain), 그리고 계약 v0.3의 **템플릿 변형
   계열**(`holdout_templates` — 문구 템플릿 변형 id)과 **개념 계열**(`holdout_concepts` — 분야마다 개념 하나)이다.
   제외된 group은 OOD이며, OOD는 group 해시로 ``ood_dev``(개발용)와 ``ood_test``(봉인)로 반분한다.
2. 남은 group을 **stable hash**로 ``train/dev/calibration/test``에 배정한다.

해시는 sha256이다. 파이썬 내장 `hash()`는 문자열마다 프로세스 seed가 섞이므로
(``PYTHONHASHSEED``) 실행마다 split이 뒤집힌다 — 배포한 데이터의 계보가 깨진다.

분할은 표현 증강·후보 재배열보다 **먼저** 한다. 파생본(번역·질문 재표현·후보 재배열·
counterfactual sibling, 스트림의 틱·rollout·구간·재라벨링)은 부모의 origin group을
그대로 쓰므로 자동으로 같은 split을 받는다. 로봇 장면 계열도 에피소드를 만들기 전에
이 함수로 배정한다(:mod:`robo_jev.data` 참고). 템플릿·개념 태그(`tags`)는 생성기가 레코드의
provenance에 적는 것과 같은 문자열(``template:<id>``·``concept:<id>``)이며, 한 계열의 레코드 중 하나라도
holdout 태그를 가지면 계열 전체가 OOD다 — 그래야 한 group이 두 split에 걸치지 않는다.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "CONCEPT_TAG",
    "DEFAULT_WEIGHTS",
    "HASH_RESOLUTION",
    "OOD_SPLITS",
    "TEMPLATE_TAG",
    "SplitPolicy",
    "assign_split",
    "group_point",
    "ood_split",
]

#: holdout된 group이 받는 split: 개발용과 봉인용. group 해시로 반분한다 (docs/04 §5).
OOD_SPLITS = ("ood_dev", "ood_test")

#: 태그 접두 — provenance의 문구 템플릿 변형 id와 개념 id를 holdout 목록과 맞댈 때 쓴다.
TEMPLATE_TAG = "template:"
CONCEPT_TAG = "concept:"

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


def ood_split(origin_group: str) -> str:
    """holdout된 group의 OOD split — 다른 해시 바이트로 ``ood_dev``/``ood_test``를 반분한다 (기본 split의 점과 독립)."""
    digest = hashlib.sha256(origin_group.encode("utf-8")).digest()
    return OOD_SPLITS[int.from_bytes(digest[8:16], "big") % len(OOD_SPLITS)]


@dataclass(frozen=True)
class SplitPolicy:
    """holdout 목록과 비중. 설정 파일(`configs/data/pilot.yaml`·`d1_robot.yaml`)에서 만든다."""

    weights: tuple[tuple[str, int], ...] = DEFAULT_WEIGHTS
    holdout_groups: frozenset[str] = field(default_factory=frozenset)
    holdout_prefixes: tuple[str, ...] = ()
    holdout_domains: frozenset[str] = field(default_factory=frozenset)
    #: 문구 템플릿 변형 id (예: ``spatial.target.ko#2``, 로봇 ``v1#2``) — 이 변형을 쓴 계열은 통째로 OOD.
    holdout_templates: frozenset[str] = field(default_factory=frozenset)
    #: 개념 id (예: ``spatial:nearest``, ``robot:goal-zone:zoneF``) — 이 개념을 다루는 계열은 통째로 OOD.
    holdout_concepts: frozenset[str] = field(default_factory=frozenset)

    def __post_init__(self) -> None:
        if not self.weights:
            raise ValueError("split weights가 비어 있다")
        allowed = {name for name, _ in DEFAULT_WEIGHTS}
        for name, weight in self.weights:
            if name not in allowed:
                raise ValueError(
                    f"해시 분할에 쓸 수 없는 split이다: {name!r} (가능: {sorted(allowed)}) "
                    f"— {'/'.join(OOD_SPLITS)}는 holdout 목록으로만 만든다"
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
            holdout_groups=frozenset(str(item) for item in config.get("holdout_groups") or ()),
            holdout_prefixes=tuple(str(item) for item in config.get("holdout_prefixes") or ()),
            holdout_domains=frozenset(str(item) for item in config.get("holdout_domains") or ()),
            holdout_templates=frozenset(str(item) for item in config.get("holdout_templates") or ()),
            holdout_concepts=frozenset(str(item) for item in config.get("holdout_concepts") or ()),
        )

    def holdout_reasons(self, origin_group: str, tags: Iterable[str] = ()) -> list[str]:
        """이 group(과 그 레코드들의 태그)이 holdout인 이유 — 종류마다 ``<종류>:<값>``. 비어 있으면 holdout이 아니다."""
        reasons: list[str] = []
        domain = origin_group.split("/", 1)[0]
        if origin_group in self.holdout_groups:
            reasons.append(f"group:{origin_group}")
        if domain in self.holdout_domains:
            reasons.append(f"domain:{domain}")
        reasons.extend(f"prefix:{prefix}" for prefix in self.holdout_prefixes if origin_group.startswith(prefix))
        seen: set[str] = set()
        for tag in tags:
            if tag in seen:
                continue
            seen.add(tag)
            if tag.startswith(TEMPLATE_TAG) and tag[len(TEMPLATE_TAG):] in self.holdout_templates:
                reasons.append(tag)
            elif tag.startswith(CONCEPT_TAG) and tag[len(CONCEPT_TAG):] in self.holdout_concepts:
                reasons.append(tag)
        return reasons

    def is_holdout(self, origin_group: str, tags: Iterable[str] = ()) -> bool:
        """의미 계열·분야·템플릿·개념 holdout인가. 해시 분할보다 **먼저** 본다."""
        return bool(self.holdout_reasons(origin_group, tags))

    def assign(self, origin_group: str, tags: Iterable[str] = ()) -> str:
        if self.is_holdout(origin_group, tags):
            return ood_split(origin_group)
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


def assign_split(origin_group: str, policy: SplitPolicy | None = None, tags: Iterable[str] = ()) -> str:
    """origin group이 속할 split. 같은 group(과 태그)은 언제 어디서 불러도 같은 값을 받는다."""
    return (policy or DEFAULT_POLICY).assign(origin_group, tags)
