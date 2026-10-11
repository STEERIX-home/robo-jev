"""학습 문구 판 v01의 새 문구(영어) — Task v0.1a §1-1. 학습 묶음 ``train_v01`` = v0b ``train`` + 이 목록(:mod:`~robo_jev.simgen.phrasing`).

v0b의 ``train``·``heldout`` 목록(:mod:`~robo_jev.simgen.phrases`)은 그대로 둔다. 새 문구는 같은 열쇠의 v0b 학습 문구와 **뜻이 정확히 같다**:
가정 행동({action}), 시간 창, 세는 것(충돌을 세나 — v0b 수정 1 I2), 점수 기준점의 방향, 답 형식(묻는 꼴)을 바꾸지 않는다. 빼 둔 문구(T2)에
가까이 가지 않는다(정규화 뒤 같지 않고, 낱말 유사도 < 0.8, 빼 둔 문구에만 있는 3-gram 금지 — 시험). 자리 표시는 그 열쇠의 v0b 학습 문구가
쓰는 것만 쓴다.
"""

from __future__ import annotations

__all__ = ["NEW", "NEW_DESCRIPTIONS"]

#: 질문 지시문·점수 기준 열쇠 → 새 문구(열쇠 순서는 v0b 표와 같다)
NEW: dict[str, list[str]] = {}

#: 선택지 설명 표(갈래) → 정식 선택지 → 새 문구 (회전 표는 열쇠 하나: turn · turn_full)
NEW_DESCRIPTIONS: dict[str, dict[str, list[str]]] = {"main": {}, "arm": {}, "head_on": {}, "p6": {}, "p4": {}, "turn": {"turn": []}, "turn_full": {"turn_full": []}}
