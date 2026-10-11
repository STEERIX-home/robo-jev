"""참 세계: 숨은 값과 참값 (docs/12 §2 ②·§5, Task v0a 지시서 §3.1).

참 세계 하나는 **관측을 만드는 데**(:mod:`~robo_jev.simgen.scenes`)와 **그 세계에서 실제로 일어난 결과**(레코드의 `truth`, 평가용)에만
쓴다. 정답(`labels`)은 관측에서 뽑은 사후 세계 K개에서 계산한다. 서술기·질문 생성기·사후 표본·정답기는 이 모듈을 import하지 않는다.

``values``는 굴리기 입력(:class:`~robo_jev.simgen.spec.Batch`)과 같은 열쇠를 갖는다 — :meth:`World.as_batch` 가 크기 1의 batch로
바꿔 사후 세계와 같은 굴리기를 돌린다. ``at_observation``은 관측 시점의 사건(그때의 의도, 단서, 옆 공간의 보임 상태)이다.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np

from robo_jev.simgen.spec import Batch, RobotSpec

__all__ = ["World"]


@dataclass(frozen=True)
class World:
    scene: str
    domain: str
    variant: str
    robot: RobotSpec
    agent_kind: str | None
    container: dict | None                    # 분야 카드의 용기 항목 {kind, content, phrases}
    values: dict[str, float]                  # 참값·숨은 값(굴리기 batch 열쇠)
    at_observation: dict[str, Any] = field(default_factory=dict)
    margins: dict[str, float] = field(default_factory=dict)
    geometry: dict[str, float] = field(default_factory=dict)   # 알려진 장면 기하(회전 반경·각도 등) — 관측에도 그대로 간다
    parts: dict = field(default_factory=dict)                  # 장면별 카드 항목(실은 물건·받침 표면·놓는 물건·센서 단서 …, Task v0b)

    def as_batch(self) -> Batch:
        return Batch({k: np.array([v]) for k, v in self.values.items()})

    def with_values(self, **changes: float) -> World:
        """값 몇 개만 바꾼 세계(사실 하나 바꾼 쌍, 누출 시험)."""
        return replace(self, values={**self.values, **changes})
