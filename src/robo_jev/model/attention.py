"""기준 mask — full-attention 층의 query×key 허용 행렬 (docs/03 §3, docs/08 §3.1).

:func:`build_reference_mask`는 :func:`robo_jev.model.serialize.serialize_request`가 낸 layout
(토큰별 ``kind``·``state``·``question``·``candidate``·``position``, 스트림은 ``tick``)을 받아
``[n, n]`` bool tensor를 돌려준다. ``mask[q, k]``가 참이면 query 토큰 q가 key 토큰 k를 본다.
느리지만 규칙을 그대로 적은 기준이며, 최적화 kernel(FlexAttention 등)은 이 행렬과 대조한다.

공통 규칙: causal(``k <= q``)이고 다른 상태(``state``가 다른 토큰, 즉 묶음 안의 다른 레코드)는
보지 않는다.

``state_first`` (L0): key가 공유 상태(``question == -1``)이거나 query와 같은 질문이면 본다.
그래서 S는 S만, T_i는 S + T_i만 보고, T_i의 결정 위치는 자기 후보를 전부 본다.

``stream_l1a`` (L1-a): key의 종류로 정한다.

* ``decision`` key — 같은 분기(같은 틱·같은 질문)의 query만 본다. 1토큰 분기에서는 자기 자신뿐이다.
  다른 결정 분기도, 다음 틱의 토큰도 결정 위치를 보지 못한다(분기 상태는 버린다).
* ``prefix`` key — 언제나 본다. 에피소드 시작 시의 정적 prefix(시작 지시·질문 세트·정적 후보)만
  여기 속하며 윈도우 밖으로 내보내지 않는다. 도중에 덧붙는 ``[지시·제약 v2]``는 그 틱의 토큰이라
  아래 몸통 규칙으로 윈도우 밖으로 나간다 — 현재 지시는 매 틱 `goal`이 다시 싣는다(docs/08 §3.1).
* 그 밖의 key(틱 몸통: 도중 지시·상태·실행 이력·동적 후보) — ``tick[q] - tick[k] < window_ticks``일
  때 본다. 기본 윈도우는 최근 30틱(자기 틱 포함, docs/08 §3.1 "정적 prefix + 최근 3초")이다.

layout 종류는 ``layout["layout"]``이 있으면 그것, 없으면 ``tick`` 필드의 유무로 정한다.
"""

from __future__ import annotations

from typing import Any

import torch

from robo_jev.model.serialize import WINDOW_TICKS

__all__ = ["WINDOW_TICKS", "build_reference_mask"]

_PER_TOKEN = ("state", "question", "candidate", "kind", "position")


def _layout_kind(layout: dict) -> str:
    explicit = layout.get("layout")
    if explicit is not None:
        if explicit not in ("state_first", "stream_l1a"):
            raise ValueError(f"layout: 알 수 없는 배치다: {explicit!r}")
        return explicit
    return "stream_l1a" if "tick" in layout else "state_first"


def _check_lengths(layout: dict, fields: tuple[str, ...]) -> int:
    n = len(layout["kind"])
    for field in fields:
        if field not in layout:
            raise ValueError(f"{field}: layout에 토큰별 필드가 없다")
        if len(layout[field]) != n:
            raise ValueError(f"{field}: 길이가 kind와 다르다 ({len(layout[field])} != {n})")
    return n


def build_reference_mask(layout: dict[str, Any], *, window_ticks: int | None = None) -> torch.Tensor:
    """query×key 허용 행렬 ``[n, n]`` (bool). 규칙은 모듈 설명 참조."""
    kind_of = _layout_kind(layout)
    fields = _PER_TOKEN + (("tick",) if kind_of == "stream_l1a" else ())
    n = _check_lengths(layout, fields)

    state = torch.as_tensor(layout["state"], dtype=torch.long)
    question = torch.as_tensor(layout["question"], dtype=torch.long)
    index = torch.arange(n)
    allowed = (index[None, :] <= index[:, None]) & (state[:, None] == state[None, :])

    if kind_of == "state_first":
        shared_key = question[None, :] == -1
        same_question = question[:, None] == question[None, :]
        return allowed & (shared_key | same_question)

    window = layout.get("window_ticks", WINDOW_TICKS) if window_ticks is None else window_ticks
    if int(window) < 1:
        raise ValueError(f"window_ticks: 1 이상이어야 한다 (받은 값: {window})")
    tick = torch.as_tensor(layout["tick"], dtype=torch.long)
    kinds = layout["kind"]
    decision_key = torch.tensor([kind == "decision" for kind in kinds], dtype=torch.bool)[None, :]
    prefix_key = torch.tensor([kind == "prefix" for kind in kinds], dtype=torch.bool)[None, :]
    same_branch = (question[:, None] == question[None, :]) & (tick[:, None] == tick[None, :])
    in_window = (tick[:, None] - tick[None, :]) < int(window)
    body = ~decision_key & (prefix_key | in_window)
    return allowed & ((decision_key & same_branch) | body)
