"""스트림 상태 — recurrent/conv 상태와 윈도우 KV의 유지·분기·복원 (docs/08 §3.1, docs/03 §3 L1-a).

:class:`StreamState`는 에피소드 스트림 하나의 계산 상태다: DeltaNet 층별 ``{"recurrent", "conv"}``,
attention 층별 KV cache(``{"k", "v"}``, RoPE 적용 뒤의 k), cache 항목별 틱 표(``cache_ticks``, prefix는
-1), 다음 논리적 position, 현재 틱 번호. 값은 어느 경로에서도 in-place로 바꾸지 않는다 — 새 tensor를
만든다.

* ``from_tokens(prefix_tokens, tick_tokens)`` — 정적 prefix를 읽은 뒤 틱을 차례로 ``advance``한다.
  ``tick_tokens``는 한 틱(정수 목록) 또는 여러 틱(목록의 목록)이다.
* ``advance(tokens) -> StreamState`` — **분기 이전 공통 상태**에 다음 틱 토큰을 이어 붙인 새 상태를
  돌려준다(자신은 그대로). 윈도우(정적 prefix + 최근 ``window_ticks``틱, 자기 틱 포함) 밖의 KV를
  먼저 내보낸다. 도중 추가되는 지시 조각은 그 틱의 토큰이므로 특별 취급이 없다.
* ``fork(n) -> list[StreamState]`` — 결정 분기용 일시적 상태 n개. recurrent·conv는
  :func:`robo_jev.model.hybrid.fork_delta_state`로 복제(독립 버퍼, gradient 연결)하고 KV는 읽기 전용으로
  공유한다. 분기는 ``advance``할 수 없다 — 다음 틱은 부모(분기 이전 공통 상태)에서 이어간다.
* ``step(token) -> Tensor[d]`` — 토큰 하나를 이 상태에 이어 붙이고(분기의 결정 토큰) 그 hidden state를
  돌려준다. 분기의 갱신은 부모·형제에 닿지 않는다(자기 버퍼와 자기 KV 꼬리만 바뀐다).
* ``branch_step(tokens) -> Tensor[n, d]`` — 결정 표지 n개의 분기를 한꺼번에(``fork(n)`` + ``step`` n번과 같다).
  실제 backbone(:mod:`robo_jev.model.backbone_qwen`)은 이것을 한 배치 forward로 구현하며 :func:`replay_layout` 이
  이것을 부른다. ``detach()``·``to_dict()``/``from_dict()``는 truncated BPTT의 구간 경계와 checkpoint가 쓴다.
* ``clone()`` — 스냅샷(모든 tensor를 clone, 그래프 유지).

position 규칙은 직렬화(:mod:`robo_jev.model.serialize`)와 같다: prefix 0부터, 틱 몸통은 이어서, 결정
분기는 모두 몸통 끝 position, 다음 틱은 그 position에서 이어간다. RoPE는 이 position을 쓴다.

:func:`replay_layout`은 직렬화된 스트림 layout을 **증분**으로 재생한다(prefix → 틱마다 ``advance`` →
결정마다 ``fork``/``step``). :func:`forward_layout`은 같은 layout을 **처음부터** 한 번의 forward로
계산한다(:func:`robo_jev.model.attention.build_reference_mask` + 결정 토큰 transient). 둘의 hidden
state는 윈도우 안에서 FP32 잡음 안에서 같아야 하고, 윈도우 절단(``window_ticks``)은 정의된 근사라
절단 없는 계산과의 차이를 기록만 한다 (docs/08 §3.1). "hidden state"는 backbone의 최종 RMSNorm 뒤
출력이다.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
from torch import Tensor

from robo_jev.model.attention import build_reference_mask
from robo_jev.model.hybrid import TinyHybrid, causal_mask, default_backbone, fork_delta_state

__all__ = ["LAYOUT_WINDOW", "StreamState", "forward_layout", "replay_layout", "stream_state_class"]

#: `window_ticks` 인수의 기본값 — layout의 `window_ticks`(없으면 backbone 설정)를 쓴다.
LAYOUT_WINDOW = "layout"


def _as_token_list(tokens: Any, name: str) -> list[int]:
    if isinstance(tokens, Tensor):
        tokens = tokens.tolist()
    out: list[int] = []
    for token in tokens:
        if isinstance(token, bool) or not isinstance(token, int):
            raise ValueError(f"{name}: 정수 토큰 id 목록이어야 한다 (받은 항목: {token!r})")
        out.append(token)
    return out


def _as_ticks(tick_tokens: Any) -> list[list[int]]:
    """한 틱(정수 목록) 또는 여러 틱(목록의 목록) → 틱 목록."""
    if isinstance(tick_tokens, Tensor):
        tick_tokens = tick_tokens.tolist()
    items = list(tick_tokens)
    if not items:
        return []
    if all(isinstance(item, int) and not isinstance(item, bool) for item in items):
        return [items]
    if all(isinstance(item, (list, tuple, Tensor)) for item in items):
        return [_as_token_list(item, f"tick_tokens[{index}]") for index, item in enumerate(items)]
    raise ValueError("tick_tokens: 한 틱의 정수 목록이거나 틱마다 정수 목록인 목록이어야 한다")


class StreamState:
    """스트림 하나의 계산 상태 (모듈 설명 참조)."""

    __slots__ = (
        "backbone", "delta", "kv", "cache_ticks", "position", "tick", "window_ticks",
        "prefix_hidden", "hidden", "is_branch",
    )  # fmt: skip

    def __init__(
        self,
        backbone: TinyHybrid,
        *,
        delta: list[dict[str, Tensor]],
        kv: list[dict[str, Tensor]],
        cache_ticks: Tensor,
        position: int,
        tick: int,
        window_ticks: int,
        prefix_hidden: Tensor | None = None,
        hidden: Tensor | None = None,
        is_branch: bool = False,
    ) -> None:
        self.backbone = backbone
        self.delta = delta
        self.kv = kv
        self.cache_ticks = cache_ticks
        self.position = position
        self.tick = tick
        self.window_ticks = window_ticks
        self.prefix_hidden = prefix_hidden
        self.hidden = hidden
        self.is_branch = is_branch

    # -- 만들기 --

    @classmethod
    def initial(
        cls,
        backbone: TinyHybrid | None = None,
        *,
        window_ticks: int | None = None,
        initial: list[dict[str, Tensor]] | None = None,
    ) -> StreamState:
        """에피소드 시작 상태(빈 cache, 0 recurrent 상태 — 또는 주어진 초기 상태)."""
        backbone = default_backbone() if backbone is None else backbone
        window = backbone.config.attention.window_ticks if window_ticks is None else int(window_ticks)
        if window < 1:
            raise ValueError(f"window_ticks: 1 이상이어야 한다 (받은 값: {window})")
        delta = backbone.initial_state(1) if initial is None else list(initial)
        if len(delta) != len(backbone.delta_layers):
            raise ValueError(f"initial: DeltaNet 층 수 {len(backbone.delta_layers)}개여야 한다 (받은 수: {len(delta)})")
        dtype = backbone.embed.weight.dtype
        kv = [
            {
                "k": torch.zeros(1, 0, layer.heads, layer.head_dim, dtype=dtype),
                "v": torch.zeros(1, 0, layer.heads, layer.head_dim, dtype=dtype),
            }
            for layer in backbone.attention_layers
        ]
        return cls(
            backbone, delta=delta, kv=kv, cache_ticks=torch.zeros(0, dtype=torch.long),
            position=0, tick=-1, window_ticks=window,
        )  # fmt: skip

    @classmethod
    def from_tokens(
        cls,
        prefix_tokens: Sequence[int],
        tick_tokens: Any,
        *,
        backbone: TinyHybrid | None = None,
        window_ticks: int | None = None,
        initial: list[dict[str, Tensor]] | None = None,
    ) -> StreamState:
        """정적 prefix를 읽고 틱을 차례로 이어 붙인 상태."""
        state = cls.initial(backbone, window_ticks=window_ticks, initial=initial)
        prefix = _as_token_list(prefix_tokens, "prefix_tokens")
        if prefix:
            state = state.extend_prefix(prefix)
        for tokens in _as_ticks(tick_tokens):
            state = state.advance(tokens)
        return state

    # -- 읽기 --

    @property
    def recurrent(self) -> list[Tensor]:
        return [layer["recurrent"] for layer in self.delta]

    @property
    def conv(self) -> list[Tensor]:
        return [layer["conv"] for layer in self.delta]

    @property
    def cached_tokens(self) -> int:
        return int(self.cache_ticks.shape[0])

    def __repr__(self) -> str:
        kind = "branch" if self.is_branch else "base"
        return (
            f"StreamState({kind}, tick={self.tick}, position={self.position}, "
            f"cached={self.cached_tokens}, window={self.window_ticks})"
        )

    # -- 계산 --

    def _visible(self, tick: int, cache_ticks: Tensor) -> Tensor:
        """cache 항목별로 틱 `tick`의 query가 보는지: prefix는 언제나, 몸통은 윈도우 안일 때."""
        return (cache_ticks == -1) | ((tick - cache_ticks) < self.window_ticks)

    def _run(self, tokens: list[int], tick: int, kv: list[dict[str, Tensor]], cache_ticks: Tensor) -> dict:
        if not tokens:
            raise ValueError("tokens: 빈 토큰 목록은 이어 붙일 수 없다")
        length = len(tokens)
        cached = int(cache_ticks.shape[0])
        visible = self._visible(tick, cache_ticks)[None, :].expand(length, cached)
        mask = torch.cat([visible, causal_mask(length)], dim=1)
        ids = torch.tensor([tokens], dtype=torch.long)
        positions = torch.arange(self.position, self.position + length)[None]
        return self.backbone(ids, positions, state=self.delta, kv=kv, mask=mask)

    @staticmethod
    def _append(kv: list[dict[str, Tensor]], new: list[dict[str, Tensor]]) -> list[dict[str, Tensor]]:
        return [
            {"k": torch.cat([old["k"], fresh["k"]], dim=1), "v": torch.cat([old["v"], fresh["v"]], dim=1)}
            for old, fresh in zip(kv, new)
        ]

    def extend_prefix(self, tokens: Sequence[int]) -> StreamState:
        """정적 prefix 토큰을 이어 붙인 새 상태(틱 시작 전에만). cache 표는 -1이라 윈도우 밖으로 나가지 않는다."""
        if self.tick != -1 or self.is_branch:
            raise ValueError("prefix는 첫 틱 전의 공통 상태에만 붙일 수 있다")
        tokens = _as_token_list(tokens, "prefix_tokens")
        out = self._run(tokens, -1, self.kv, self.cache_ticks)
        hidden = out["hidden"][0]
        return StreamState(
            self.backbone,
            delta=out["state"],
            kv=self._append(self.kv, out["kv"]),
            cache_ticks=torch.cat([self.cache_ticks, torch.full((len(tokens),), -1, dtype=torch.long)]),
            position=self.position + len(tokens),
            tick=-1,
            window_ticks=self.window_ticks,
            prefix_hidden=hidden if self.prefix_hidden is None else torch.cat([self.prefix_hidden, hidden]),
            hidden=None,
        )

    def advance(self, tokens: Sequence[int]) -> StreamState:
        """분기 이전 공통 상태에 다음 틱을 이어 붙인 새 상태. 윈도우 밖 KV는 내보낸다."""
        if self.is_branch:
            raise ValueError(
                "분기 상태에서는 다음 틱으로 이어갈 수 없다 — 다음 틱은 분기 이전 공통 상태에서 이어간다 (docs/08 §3.1)"
            )
        tokens = _as_token_list(tokens, "tokens")
        tick = self.tick + 1
        keep = self._visible(tick, self.cache_ticks)
        kv = [{"k": layer["k"][:, keep], "v": layer["v"][:, keep]} for layer in self.kv]
        cache_ticks = self.cache_ticks[keep]
        out = self._run(tokens, tick, kv, cache_ticks)
        return StreamState(
            self.backbone,
            delta=out["state"],
            kv=self._append(kv, out["kv"]),
            cache_ticks=torch.cat([cache_ticks, torch.full((len(tokens),), tick, dtype=torch.long)]),
            position=self.position + len(tokens),
            tick=tick,
            window_ticks=self.window_ticks,
            prefix_hidden=self.prefix_hidden,
            hidden=out["hidden"][0],
        )

    def fork(self, n: int) -> list[StreamState]:
        """결정 분기용 일시적 상태 n개 (recurrent·conv 복제, KV 읽기 전용 공유)."""
        forked = [fork_delta_state(layer, n) for layer in self.delta]
        return [
            StreamState(
                self.backbone,
                delta=[layer[index] for layer in forked],
                kv=self.kv,
                cache_ticks=self.cache_ticks,
                position=self.position,
                tick=self.tick,
                window_ticks=self.window_ticks,
                prefix_hidden=self.prefix_hidden,
                hidden=None,
                is_branch=True,
            )
            for index in range(n)
        ]

    def step(self, token: int | Tensor) -> Tensor:
        """토큰 하나를 이 상태에 이어 붙이고 그 hidden state ``[d]``를 돌려준다."""
        if isinstance(token, Tensor):
            token = int(token.item())
        if isinstance(token, bool) or not isinstance(token, int):
            raise ValueError(f"token: 정수 토큰 id여야 한다 (받은 값: {token!r})")
        out = self._run([token], self.tick, self.kv, self.cache_ticks)
        self.delta = out["state"]
        self.kv = self._append(self.kv, out["kv"])
        self.cache_ticks = torch.cat([self.cache_ticks, torch.tensor([self.tick], dtype=torch.long)])
        self.position += 1
        self.hidden = out["hidden"][0]
        return out["hidden"][0, 0]

    def branch_step(self, tokens: Sequence[int]) -> Tensor:
        """결정 표지 n개를 각각 1토큰 분기로 → hidden ``[n, d]``. ``fork(n)`` 뒤 분기마다 ``step``과 같다.

        실제 backbone(:class:`robo_jev.model.backbone_qwen.QwenStreamState`)은 이것을 **한 배치 forward**로 구현하고,
        fixture는 분기를 차례로 돈다 — 둘 다 상태·KV를 바꾸지 않는다.
        """
        tokens = _as_token_list(tokens, "tokens")
        if not tokens:
            raise ValueError("tokens: 결정 토큰이 하나 이상 필요하다")
        return torch.stack([branch.step(token) for branch, token in zip(self.fork(len(tokens)), tokens)])

    def detach(self, *, requires_grad: bool = False) -> StreamState:
        """구간 경계에서 넘기는 공통 상태 — 모든 tensor를 detach한 새 상태 (값은 같고 gradient만 끊긴다).

        `requires_grad=True`면 detach한 tensor를 leaf로 만들어 다음 구간의 gradient가 경계에 얼마나 닿는지 관찰할 수
        있다(검사용). :func:`robo_jev.train.detach_stream_state` 가 부른다.
        """
        if self.is_branch:
            raise ValueError("branch 상태는 넘기지 않는다 — 다음 구간은 분기 이전 공통 상태에서 이어간다")

        def cut(tensor: Tensor | None) -> Tensor | None:
            if tensor is None:
                return None
            out = tensor.detach()
            if requires_grad and out.is_floating_point():
                out.requires_grad_(True)
            return out

        return StreamState(
            self.backbone,
            delta=[{key: cut(value) for key, value in layer.items()} for layer in self.delta],
            kv=[{key: cut(value) for key, value in layer.items()} for layer in self.kv],
            cache_ticks=self.cache_ticks.detach(),
            position=self.position,
            tick=self.tick,
            window_ticks=self.window_ticks,
            prefix_hidden=cut(self.prefix_hidden),
            hidden=cut(self.hidden),
            is_branch=False,
        )

    def to_dict(self) -> dict[str, Any]:
        """checkpoint용 — 분기 이전 공통 상태를 detach된 tensor dict로 (:mod:`robo_jev.checkpoint`)."""
        if self.is_branch:
            raise ValueError("branch 상태는 저장하지 않는다 — 구간 경계의 상태는 분기 이전 공통 상태다")
        return {
            "kind": "tiny",
            "delta": [{key: value.detach().clone() for key, value in layer.items()} for layer in self.delta],
            "kv": [{key: value.detach().clone() for key, value in layer.items()} for layer in self.kv],
            "cache_ticks": self.cache_ticks.detach().clone(),
            "position": int(self.position),
            "tick": int(self.tick),
            "window_ticks": int(self.window_ticks),
            "prefix_hidden": None if self.prefix_hidden is None else self.prefix_hidden.detach().clone(),
            "hidden": None if self.hidden is None else self.hidden.detach().clone(),
        }

    @classmethod
    def from_dict(cls, packed: dict[str, Any], backbone: TinyHybrid) -> StreamState:
        """:meth:`to_dict` 의 역 — 주어진 backbone에 붙인 공통 상태."""
        for key in ("delta", "kv", "cache_ticks", "position", "tick", "window_ticks"):
            if key not in packed:
                raise ValueError(f"carried_state.{key}: 없다")
        return cls(
            backbone,
            delta=[dict(layer) for layer in packed["delta"]],
            kv=[dict(layer) for layer in packed["kv"]],
            cache_ticks=packed["cache_ticks"],
            position=int(packed["position"]),
            tick=int(packed["tick"]),
            window_ticks=int(packed["window_ticks"]),
            prefix_hidden=packed.get("prefix_hidden"),
            hidden=packed.get("hidden"),
            is_branch=False,
        )

    def clone(self) -> StreamState:
        """스냅샷 — 모든 tensor를 clone한다(그래프 유지)."""
        return StreamState(
            self.backbone,
            delta=[{key: value.clone() for key, value in layer.items()} for layer in self.delta],
            kv=[{key: value.clone() for key, value in layer.items()} for layer in self.kv],
            cache_ticks=self.cache_ticks.clone(),
            position=self.position,
            tick=self.tick,
            window_ticks=self.window_ticks,
            prefix_hidden=None if self.prefix_hidden is None else self.prefix_hidden.clone(),
            hidden=None if self.hidden is None else self.hidden.clone(),
            is_branch=self.is_branch,
        )


# --------------------------------------------------------------------------
# 직렬화된 layout의 재생(증분)과 처음부터 계산
# --------------------------------------------------------------------------


def stream_state_class(backbone: Any) -> Any:
    """backbone의 스트림 상태 클래스 — fixture는 :class:`StreamState`, 실제 backbone은 자기 것(`stream_state_class`)."""
    return getattr(backbone, "stream_state_class", StreamState)


def _resolve_window(layout: dict, backbone: TinyHybrid, window_ticks: Any) -> int | None:
    if window_ticks == LAYOUT_WINDOW:
        return int(layout.get("window_ticks", backbone.config.attention.window_ticks))
    if window_ticks is None:
        return None
    return int(window_ticks)


def forward_layout(
    layout: dict,
    *,
    backbone: TinyHybrid | None = None,
    window_ticks: Any = LAYOUT_WINDOW,
    return_layers: bool = False,
) -> Any:
    """처음부터 한 번의 forward: 기준 mask + 결정 토큰 transient. ``[n, d]`` hidden state.

    ``window_ticks=None``은 절단 없음(윈도우 절단 차이를 재는 기준)이다. ``return_layers``면
    ``(hidden, [층별 출력 [n, d] …])``를 돌려준다(어느 층에서 차이가 나는지 볼 때).
    """
    backbone = default_backbone() if backbone is None else backbone
    window = _resolve_window(layout, backbone, window_ticks)
    n = len(layout["tokens"])
    if window is None:
        window = max(int(t) for t in layout["tick"]) + 2 if n else 1  # 어떤 틱도 밖으로 나가지 않는다
    own = getattr(backbone, "forward_layout", None)
    if own is not None:  # 실제 backbone은 자기 기준 계산(공식 forward + 물질화한 기준 mask)으로
        if return_layers:
            raise ValueError("return_layers: 실제 backbone의 기준 계산은 층별 출력을 내지 않는다")
        return own(layout, window_ticks=window)
    mask = build_reference_mask(layout, window_ticks=window)
    tokens = torch.tensor([layout["tokens"]], dtype=torch.long)
    positions = torch.tensor([layout["position"]], dtype=torch.long)
    transient = torch.tensor([kind == "decision" for kind in layout["kind"]], dtype=torch.bool)
    out = backbone(tokens, positions, mask=mask, transient=transient)
    if return_layers:
        return out["hidden"][0], [layer[0] for layer in out["layer_hidden"]]
    return out["hidden"][0]


def _device_of(backbone: Any) -> torch.device:
    return next(backbone.parameters()).device


def _check_position(layout: dict, index: int, expected: int) -> None:
    positions = layout.get("position")
    if positions is not None and int(positions[index]) != expected:
        raise ValueError(
            f"position[{index}]: 스트림 상태의 position {expected}와 layout의 {positions[index]}가 다르다 "
            "(결정 분기는 몸통 끝 position을 공유하고 다음 틱은 거기서 이어간다)"
        )


def replay_layout(
    layout: dict,
    *,
    backbone: TinyHybrid | None = None,
    window_ticks: Any = LAYOUT_WINDOW,
    initial: list[dict[str, Tensor]] | None = None,
    state: StreamState | None = None,
    start_tick: int = 0,
) -> dict[str, Any]:
    """직렬화된 스트림 layout을 증분으로 재생한다.

    prefix(``tokens[:prefix_end]``)를 읽고 틱마다 몸통을 ``advance``한 뒤 결정 토큰마다 ``fork``/``step``
    한다. 돌려주는 것: ``hidden [n, d]``(모든 토큰; 결정 토큰은 분기의 값), ``final``(마지막 틱의
    분기 이전 공통 상태), ``tick_states``(재생한 틱마다 그 공통 상태), ``branch_hidden``(틱마다 결정
    index → hidden), ``start_tick``. ``state``를 주면 prefix를 다시 읽지 않고 거기서 이어가며,
    ``start_tick``부터 재생한다(앞 틱은 그 상태가 이미 읽은 것으로 보고 hidden 행은 0이다) — 구간을
    이어 붙이는 학습(truncated BPTT)의 근거다.
    """
    backbone = default_backbone() if backbone is None else backbone
    window = _resolve_window(layout, backbone, window_ticks)
    if window is None:
        raise ValueError("window_ticks: 증분 재생은 윈도우(정수)가 필요하다 — 절단 없는 기준은 forward_layout")
    tokens = layout["tokens"]
    prefix_end = int(layout["prefix_end"])
    pieces: list[Tensor] = []
    if state is None:
        if start_tick:
            raise ValueError("start_tick: 앞 틱을 이미 읽은 state와 함께만 쓸 수 있다")
        state = stream_state_class(backbone).initial(backbone, window_ticks=window, initial=initial)
        if prefix_end:
            _check_position(layout, 0, state.position)
            state = state.extend_prefix(tokens[:prefix_end])
    elif state.is_branch:
        raise ValueError("state: 분기 상태에서는 재생을 이어갈 수 없다")
    elif state.tick != start_tick - 1:
        raise ValueError(f"start_tick: state는 틱 {state.tick}까지 읽었으니 {state.tick + 1}부터 이어가야 한다 (받은 값: {start_tick})")
    if prefix_end:
        if state.prefix_hidden is None or state.prefix_hidden.shape[0] < prefix_end:
            raise ValueError("state: 이 layout의 prefix를 이미 읽은 상태여야 한다")
        pieces.append(state.prefix_hidden[:prefix_end])
    cursor = prefix_end
    tick_states: list[StreamState] = []
    branch_hidden: list[dict[int, Tensor]] = []
    for tick in layout["ticks"]:
        start, body_end, end = int(tick["start"]), int(tick["body_end"]), int(tick["end"])
        if start != cursor:
            raise ValueError(f"ticks[{tick.get('index')}].start: {start} — 토큰 {cursor}부터 이어져야 한다")
        if int(tick["index"]) < start_tick:
            pieces.append(
                torch.zeros(
                    end - start, backbone.config.d_model,
                    dtype=state.prefix_hidden.dtype if state.prefix_hidden is not None else torch.float32, device=_device_of(backbone),
                )
            )  # fmt: skip
            cursor = end
            continue
        _check_position(layout, start, state.position)
        state = state.advance(tokens[start:body_end])
        tick_states.append(state)
        pieces.append(state.hidden)
        decisions = list(range(body_end, end))
        outputs: dict[int, Tensor] = {}
        if decisions:
            for index in decisions:
                _check_position(layout, index, state.position)
            branch_hidden_rows = state.branch_step([tokens[index] for index in decisions])  # 분기 n개 (실제 backbone은 한 배치)
            for index, row in zip(decisions, branch_hidden_rows):
                outputs[index] = row
            pieces.append(branch_hidden_rows)
        branch_hidden.append(outputs)
        cursor = end
    if cursor != len(tokens):
        raise ValueError(f"ticks: 토큰 {cursor}까지만 틱에 속한다 (전체 {len(tokens)})")
    hidden = torch.cat(pieces) if pieces else torch.zeros(0, backbone.config.d_model, device=_device_of(backbone))
    return {
        "hidden": hidden,
        "final": state,
        "tick_states": tick_states,
        "branch_hidden": branch_hidden,
        "start_tick": int(start_tick),
    }
