"""앞단(지각) — 3D 재구성 결과를 공통 구조화 상태로 옮긴다 (docs/08 §3.2).

robojev의 입력은 **하나의 공통 상태 스키마**이고 앞단은 교체 가능한 모듈이다.
:mod:`robo_jev.perception.pointworld`가 그 인터페이스(`extract`)를 고정하고, D1에서는
참값 어댑터(:class:`~robo_jev.perception.pointworld.GroundTruthAdapter`)가 같은 스키마를
채운다. E2의 시뮬 3D 카메라·재구성 결함 모델은 같은 인터페이스 뒤에 붙는다.
"""

from robo_jev.perception.pointworld import (
    EXTRACTOR_VERSION,
    GroundTruthAdapter,
    Reconstruction,
    SceneSummary,
    TrackedInstance,
    extract,
)

__all__ = [
    "EXTRACTOR_VERSION",
    "GroundTruthAdapter",
    "Reconstruction",
    "SceneSummary",
    "TrackedInstance",
    "extract",
]
