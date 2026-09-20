"""실제 backbone adapter — Qwen3.5 text 모델을 Judge의 backbone과 스트림 상태로 (docs/03 §3, docs/08 §3.1, Task 2b G0b S1).

:class:`QwenBackbone` 은 `transformers`의 ``Qwen3_5ForCausalLM``(가중치는 `artifacts/models/<id>`,
:func:`robo_jev.model.backbone.describe_backbone` 이 manifest와 대조한 것)을 감싸고, 소형 fixture
:class:`robo_jev.model.hybrid.TinyHybrid` 와 같은 자리에서 :class:`robo_jev.model.judge.Judge` 의 backbone이 된다.
lm_head는 쓰지 않는다 — readout이 읽는 것은 최종 RMSNorm 뒤의 hidden state다. `transformers`·`fla`·`causal_conv1d`는
**함수 안에서만** import한다(가중치가 없는 검사는 이 모듈을 import만 해도 된다).

세 계산 경로가 있고 셋 다 같은 가중치·같은 position 규칙(직렬화의 논리적 position)을 쓴다.

* ``forward(tokens, positions)`` — 오른쪽 padding한 배치의 plain causal forward(P0의 독립 경로 ``S+T_i``). 공식
  forward 그대로다(``position_ids``만 우리 것).
* ``reference_forward(tokens, positions, mask)`` — **명시적 기준 mask**(:func:`robo_jev.model.attention.build_reference_mask`
  의 ``[n, n]``)를 물질화해 공식 forward에 넣는다. 느리고 mask 메모리가 O(n²)라 검사·기준 계산 전용이다.
  :meth:`forward_layout` 이 이것으로 스트림 layout을 처음부터 계산한다(몸통 한 번 + 틱마다 결정 분기 배치).
* :class:`QwenStreamState` — 증분 스트림 경로(docs/08 §3.1). DeltaNet 층은 recurrent 상태(fp32)와 conv history를
  fla의 ``chunk_gated_delta_rule(initial_state=…, output_final_state=True)``·``causal_conv1d_fn``으로 명시적으로
  넘기고, full-attention 층의 KV는 **정적 prefix 버퍼 + 미리 할당한 윈도우 버퍼**(``torch.cat`` 성장 없음, 틱 단위
  퇴출은 ``advance`` 때 앞쪽 포인터를 옮기는 것뿐, 버퍼 끝에 닿을 때만 살아 있는 윈도우를 앞으로 당긴다)에 둔다.

**mask 없는 attention.** query가 보는 key는 ``prefix ‖ 윈도우 ‖ 자기 틱의 앞 토큰``이고 버퍼가 물리적으로 그 순서라,
attention을 세 조각 — prefix(전부 봄), 윈도우(전부 봄), 새 토큰끼리(causal) — 의 flash 호출로 나눠 각 조각의
logsumexp로 합친다(:func:`merge_attention_parts`; flash-decoding의 split-KV 합산과 같은 식). 어느 조각에도 mask
tensor가 없고 GQA는 query head를 토큰 축으로 접어(``[T·g, Hk, D]``) key를 복제하지 않는다. 결정 분기 10개는
**한 번의 배치 forward**다: DeltaNet 층은 공통 상태 ``S``를 복제하지 않고 ``Sᵀq``·``Sᵀk``만 읽어 분기별 갱신을
계산하고(transient 읽기 — fixture의 ``transient``와 같은 수학), attention 층은 공유 KV 위에 분기 query 10개를
한 flash 호출로 돌린 뒤 자기 토큰 하나를 logsumexp로 합친다. 분기의 어떤 값도 버퍼·상태에 쓰이지 않으므로
격리는 구조로 성립한다. 어느 sdpa backend가 돌았는지는 :attr:`QwenBackbone.attention_backend` 가 말한다.

윈도우 규칙·position·"다음 틱은 분기 이전 공통 상태에서"는 fixture의 :class:`robo_jev.model.stream.StreamState` 와
같다(검사가 실제 2B에서 증분 == 처음부터(기준 mask)를 BF16 허용 오차 안에서 확인한다). 학습(LoRA)용으로는
``kv_mode="dynamic"``(KV를 틱별 tensor 목록으로 두고 그래프를 유지)이 있다 — 정적 버퍼는 in-place 쓰기라
autograd와 함께 쓸 수 없다.
"""

from __future__ import annotations

import functools

import math
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from robo_jev.model.attention import build_reference_mask
from robo_jev.model.backbone import describe_backbone
from robo_jev.model.serialize import WINDOW_TICKS

__all__ = [
    "AttentionParts",
    "DEFAULT_WINDOW_CAPACITY",
    "KV_MODES",
    "QwenBackbone",
    "QwenConfigView",
    "QwenStreamState",
    "candidate_ids",
    "kernel_names",
    "merge_attention_parts",
    "torch_reference_kernels",
    "windowed_attention",
]

#: 윈도우 KV 버퍼의 기본 용량(토큰). 30틱 × ≤1K 토큰의 두 배 남짓 — 버퍼 끝에 닿아 앞으로 당기는 일이 윈도우 한 번
#: 분량마다 한 번꼴이 되게 한다. 2B(6층·2 KV head·256)에서 0.6 GB, 4B(8층·4 head)에서 1.6 GB.
DEFAULT_WINDOW_CAPACITY = 40_000
KV_MODES = ("static", "dynamic")
_L2NORM_EPS = 1e-6


def candidate_ids(config: str | Path | None = None) -> tuple[str, ...]:
    """`configs/model/candidates.yaml`의 후보 id (train.py의 `MODEL_IDS`가 fixture 옆에 더한다)."""
    import yaml

    from robo_jev.model.backbone import CANDIDATES_CONFIG, _PACKAGE_ROOT

    path = Path(config) if config is not None else _PACKAGE_ROOT / CANDIDATES_CONFIG
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return tuple(str(entry["id"]) for entry in data["candidates"])


# --------------------------------------------------------------------------
# mask 없는 attention: 조각 flash 호출 + logsumexp 합산
# --------------------------------------------------------------------------


@dataclass
class AttentionParts:
    """attention의 한 조각 결과 — 정규화된 출력 ``out [T, H, D]``와 그 logsumexp ``lse [T, H]``(fp32)."""

    out: Tensor
    lse: Tensor


def _flash_part(q: Tensor, k: Tensor, v: Tensor, *, causal: bool, scale: float) -> AttentionParts:
    """query ``[T, H, D]``가 key ``[N, Hk, D]``를 (causal이면 같은 길이의 자기 토큰끼리) 보는 한 조각.

    CUDA에서는 varlen flash(`aten::_flash_attention_forward`)로, 그 밖에서는 CPU flash로 돈다. 둘 다 mask tensor가
    없다. causal이 아니면 query head를 토큰 축으로 접어(``[T·g, Hk, D]``) GQA를 key 복제 없이 처리하고, causal이면
    (접으면 causal 정렬이 깨지므로) 새 토큰 T개의 key만 head 축으로 펼친다(작다).
    """
    T, H, D = q.shape
    N, Hk, _ = k.shape
    g = H // Hk
    if q.is_cuda and q.dtype not in (torch.float16, torch.bfloat16):
        return _math_part(q, k, v, causal=causal, scale=scale)  # fp32 CUDA는 검사용 — flash는 반정밀도뿐
    if q.is_cuda and torch.is_grad_enabled() and (q.requires_grad or k.requires_grad or v.requires_grad):
        return _flash_part_differentiable(q, k, v, causal=causal, scale=scale)
    if q.is_cuda:
        if causal:
            kk = k.repeat_interleave(g, dim=1) if g > 1 else k
            vv = v.repeat_interleave(g, dim=1) if g > 1 else v
            cu_q = torch.tensor([0, T], device=q.device, dtype=torch.int32)
            out, lse = torch.ops.aten._flash_attention_forward(
                q, kk, vv, cu_q, cu_q, T, N, 0.0, True, False, scale=scale
            )[:2]
            return AttentionParts(out, lse.transpose(0, 1))  # lse [H, T] → [T, H]
        # head h = kv·g + j (repeat_kv 규약) → query 행 (t, j)를 토큰 축에 접고 kv head만 남긴다
        folded = q.view(T, Hk, g, D).permute(0, 2, 1, 3).reshape(T * g, Hk, D) if g > 1 else q
        cu_q = torch.tensor([0, T * g], device=q.device, dtype=torch.int32)
        cu_k = torch.tensor([0, N], device=q.device, dtype=torch.int32)
        out, lse = torch.ops.aten._flash_attention_forward(
            folded, k, v, cu_q, cu_k, T * g, N, 0.0, False, False, scale=scale
        )[:2]
        if g > 1:
            out = out.view(T, g, Hk, D).permute(0, 2, 1, 3).reshape(T, H, D)
            lse = lse.view(Hk, T, g).permute(1, 0, 2).reshape(T, H)
        else:
            lse = lse.transpose(0, 1)
        return AttentionParts(out, lse)
    # CPU: `[B, H, T, D]` 꼴의 CPU flash (검사용 — fixture 크기에서만)
    kk = k.repeat_interleave(g, dim=1) if g > 1 else k
    vv = v.repeat_interleave(g, dim=1) if g > 1 else v
    out, lse = torch.ops.aten._scaled_dot_product_flash_attention_for_cpu(
        q.transpose(0, 1)[None], kk.transpose(0, 1)[None], vv.transpose(0, 1)[None], 0.0, causal, scale=scale
    )[:2]
    return AttentionParts(out[0].transpose(0, 1), lse[0].transpose(0, 1))


def _flash_part_differentiable(q: Tensor, k: Tensor, v: Tensor, *, causal: bool, scale: float) -> AttentionParts:
    """학습(LoRA)용 조각: autograd 공식이 있는 `aten::_scaled_dot_product_flash_attention`(``[B, H, N, D]``, lse 반환)으로.

    varlen op에는 backward가 없다. causal이 아니면 같은 접기(query head → 토큰 축)로 key 복제가 없고, causal이면 새 토큰의
    key만 head 축으로 펼친다. mask tensor는 없다.
    """
    T, H, D = q.shape
    N, Hk, _ = k.shape
    g = H // Hk
    if causal:
        kk = (k.repeat_interleave(g, dim=1) if g > 1 else k).transpose(0, 1)[None]
        vv = (v.repeat_interleave(g, dim=1) if g > 1 else v).transpose(0, 1)[None]
        out, lse = torch.ops.aten._scaled_dot_product_flash_attention(q.transpose(0, 1)[None], kk, vv, 0.0, True, False, scale=scale)[:2]
        return AttentionParts(out[0].transpose(0, 1), lse[0].transpose(0, 1))
    folded = q.view(T, Hk, g, D).permute(1, 0, 2, 3).reshape(1, Hk, T * g, D) if g > 1 else q.transpose(0, 1)[None]
    out, lse = torch.ops.aten._scaled_dot_product_flash_attention(folded, k.transpose(0, 1)[None], v.transpose(0, 1)[None], 0.0, False, False, scale=scale)[:2]
    if g > 1:
        out = out[0].view(Hk, T, g, D).permute(1, 0, 2, 3).reshape(T, H, D)
        lse = lse[0].view(Hk, T, g).permute(1, 0, 2).reshape(T, H)
    else:
        out, lse = out[0].transpose(0, 1), lse[0].transpose(0, 1)
    return AttentionParts(out, lse)


def _math_part(q: Tensor, k: Tensor, v: Tensor, *, causal: bool, scale: float) -> AttentionParts:
    """fp32 참조 조각 (검사용; causal이면 삼각 mask를 물질화한다 — 서빙 경로가 아니다)."""
    T, H, D = q.shape
    g = H // k.shape[1]
    kk = k.repeat_interleave(g, dim=1) if g > 1 else k
    vv = v.repeat_interleave(g, dim=1) if g > 1 else v
    scores = torch.einsum("thd,nhd->htn", q.float(), kk.float()) * scale  # [H, T, N]
    if causal:
        scores = scores.masked_fill(torch.ones(T, k.shape[0], dtype=torch.bool, device=q.device).triu(1)[None], float("-inf"))
    lse = torch.logsumexp(scores, dim=-1)  # [H, T]
    out = torch.einsum("htn,nhd->thd", torch.exp(scores - lse[..., None]), vv.float())
    return AttentionParts(out.to(q.dtype), lse.transpose(0, 1))


def _flash_part_static(
    q: Tensor, k: Tensor, v: Tensor, cu_k: Tensor, scale: float, *, seqused: Tensor | None = None, capacity: int | None = None
) -> AttentionParts:
    """graph 캡처용 조각: key 버퍼 **전체**를 넘기고 유효 길이는 장치 tensor(`cu_k`·`seqused`)가 말한다 — 호출 안에서 tensor를 만들지 않는다."""
    T, H, D = q.shape
    N, Hk, _ = k.shape
    g = H // Hk
    folded = q.view(T, Hk, g, D).permute(0, 2, 1, 3).reshape(T * g, Hk, D) if g > 1 else q
    cu_q = _CU_CACHE.get(q.device, T * g)
    max_k = N if capacity is None else capacity
    kwargs = {"scale": scale}
    if seqused is not None:
        kwargs["seqused_k"] = seqused
    out, lse = torch.ops.aten._flash_attention_forward(folded, k, v, cu_q, cu_k, T * g, max_k, 0.0, False, False, **kwargs)[:2]
    if g > 1:
        out = out.view(T, g, Hk, D).permute(0, 2, 1, 3).reshape(T, H, D)
        lse = lse.view(Hk, T, g).permute(1, 0, 2).reshape(T, H)
    else:
        lse = lse.transpose(0, 1)
    return AttentionParts(out, lse)


class _CuCache:
    """`[0, n]` int32 장치 tensor를 미리 만들어 둔다 (graph 캡처 중에는 H2D 복사를 만들 수 없다)."""

    def __init__(self) -> None:
        self._cache: dict[tuple[str, int], Tensor] = {}

    def get(self, device: torch.device, n: int) -> Tensor:
        key = (str(device), n)
        if key not in self._cache:
            self._cache[key] = torch.tensor([0, n], device=device, dtype=torch.int32)
        return self._cache[key]


_CU_CACHE = _CuCache()


@dataclass
class _StaticWindow:
    """graph 캡처용 윈도우 경계 — 장치 tensor라 재생 때 값만 바꾼다. 윈도우 버퍼 전체 위의 varlen 호출: 원소 0 = ``[0, start)``
    (지난 틱의 낡은 key — 결과는 버린다), 원소 1 = ``[start, end)``(살아 있는 윈도우)."""

    prefix_cu: Tensor  # [0, P]
    window_cu: Tensor  # [0, end] (seqused가 유효 길이)
    seqused: Tensor  # [end − start] … 아래 설명


@dataclass
class _BranchStatic:
    ids: Tensor
    position: Tensor
    delta: list[dict[str, Tensor]]
    windows: list[_StaticWindow]


def _kv_buffer_key(state: QwenStreamState) -> tuple[tuple[int, int, int, int, int], ...]:
    """graph 재캡처 판정용: 층마다 (prefix K·V 주소, 윈도우 K·V 주소, 윈도우 용량)."""
    return tuple(
        (store.k_prefix.data_ptr(), store.v_prefix.data_ptr(), store.k_win.data_ptr() if store.k_win is not None else 0, store.v_win.data_ptr() if store.v_win is not None else 0, store.capacity)
        for store in state._kv
    )


class _BranchGraph:
    """결정 분기 배치 forward의 CUDA graph (지렛대 graphs).

    캡처 조건: 분기 수 n, prefix 버퍼(주소·길이), 윈도우 버퍼 주소가 에피소드 동안 고정. 틱마다 바뀌는 것 — 결정 토큰 id,
    position, DeltaNet 상태(fla가 틱마다 새 tensor를 내므로 정적 버퍼로 복사), 윈도우의 살아 있는 구간 — 은 재생 전에 정적
    버퍼에 써 넣는다. 윈도우 조각은 버퍼 ``[0, end)`` 위의 varlen 호출에 ``seqused_k = end − start``를 주되 key 포인터를
    ``start``에서 시작하게 하려면 주소가 바뀌므로, 대신 살아 있는 구간을 ``[0, live)``로 두는 **틱마다의 당김**(compaction)을
    재생 직전에 한다 — 복사 한 번(윈도우 KV 크기)이 graph 재생과 함께 틱 비용에 든다(보고서에 따로 적는다).
    """

    def __init__(self, state: QwenStreamState, n: int) -> None:
        bb = state.backbone
        device = bb.device
        self.n = n
        #: 캡처한 버퍼들의 주소·용량 (층마다 prefix·윈도우 K/V) — 하나라도 바뀌면 재캡처 (allocator가 새 prefix에 옛 주소를 줄 수 있다)
        self.buffer_key = _kv_buffer_key(state)
        self.ids = torch.zeros(n, dtype=torch.long, device=device)
        self.position = torch.zeros(1, dtype=torch.long, device=device)
        self.delta = [{k: torch.empty_like(v) for k, v in layer.items()} for layer in state.delta]
        self.windows = [
            _StaticWindow(
                prefix_cu=torch.tensor([0, store.k_prefix.shape[0]], device=device, dtype=torch.int32),
                window_cu=torch.tensor([0, store.capacity], device=device, dtype=torch.int32),
                seqused=torch.tensor([max(store.end - store.start, 1)], device=device, dtype=torch.int32),
            )
            for store in state._kv
        ]
        self.static = _BranchStatic(self.ids, self.position, self.delta, self.windows)
        self._stage(state, list(range(n)))
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(2):  # 예열 (Triton autotune·allocator)
                state._branch_step_eager([0] * n, static=self.static)
        torch.cuda.current_stream().wait_stream(stream)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.out = state._branch_step_eager([0] * n, static=self.static)

    def _stage(self, state: QwenStreamState, tokens: list[int]) -> None:
        """재생 전: 입력·상태·윈도우 경계를 정적 버퍼에 쓰고 윈도우를 앞으로 당긴다."""
        self.ids.copy_(torch.tensor(tokens, dtype=torch.long), non_blocking=True)
        self.position.fill_(state.position)
        for target, source in zip(self.delta, state.delta):
            for key in target:
                target[key].copy_(source[key])
        for store, window in zip(state._kv, self.windows):
            live = store.end - store.start
            if store.start:
                store.k_win[:live].copy_(store.k_win[store.start : store.end].clone())
                store.v_win[:live].copy_(store.v_win[store.start : store.end].clone())
                store.start, store.end = 0, live
            window.seqused.fill_(max(live, 1))

    def replay(self, state: QwenStreamState, tokens: list[int]) -> Tensor:
        self._stage(state, tokens)
        self.graph.replay()
        return self.out.clone()


def merge_attention_parts(parts: list[AttentionParts]) -> Tensor:
    """조각들의 softmax 합산: ``out = Σ_i out_i · exp(lse_i − lse)``, ``lse = logsumexp_i lse_i`` (fp32에서)."""
    if len(parts) == 1:
        return parts[0].out
    lses = torch.stack([part.lse.float() for part in parts])  # [P, T, H]
    total = torch.logsumexp(lses, dim=0)
    weights = torch.exp(lses - total[None])  # [P, T, H]
    out = sum(part.out.float() * weight[..., None] for part, weight in zip(parts, weights))
    return out.to(parts[0].out.dtype)


def windowed_attention(
    q: Tensor, segments: list[tuple[Tensor, Tensor]], new_kv: tuple[Tensor, Tensor] | None, *, scale: float
) -> Tensor:
    """query ``[T, H, D]`` × (전부 보이는 key 조각들 + 자기 토큰끼리 causal) → ``[T, H, D]``. 비어 있는 조각은 건너뛴다."""
    parts: list[AttentionParts] = []
    for k, v in segments:
        if k.shape[0]:
            parts.append(_flash_part(q, k, v, causal=False, scale=scale))
    if new_kv is not None:
        parts.append(_flash_part(q, new_kv[0], new_kv[1], causal=True, scale=scale))
    if not parts:
        raise ValueError("attention: 볼 key가 하나도 없다")
    return merge_attention_parts(parts)


# --------------------------------------------------------------------------
# backbone
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class _AttentionView:
    window_ticks: int = WINDOW_TICKS


@dataclass(frozen=True)
class QwenConfigView:
    """fixture의 `HybridConfig`가 주던 것 가운데 Judge·학습 manifest가 읽는 값."""

    name: str
    vocab_size: int
    d_model: int
    seed: int | None = None
    attention: _AttentionView = _AttentionView()
    layers: tuple[str, ...] = ()


class QwenBackbone(nn.Module):
    """Qwen3.5 text 모델 adapter (모듈 설명 참조). ``load``로 만든다."""

    def __init__(
        self,
        model: Any,
        *,
        model_id: str,
        manifest: dict[str, Any] | None = None,
        kv_mode: str = "static",
        window_capacity: int = DEFAULT_WINDOW_CAPACITY,
    ) -> None:
        super().__init__()
        if kv_mode not in KV_MODES:
            raise ValueError(f"kv_mode: {list(KV_MODES)} 중 하나 (받은 값: {kv_mode!r})")
        self.model = model  # Qwen3_5ForCausalLM — lm_head는 쓰지 않는다
        self.model_id = model_id
        self.manifest = dict(manifest or {})
        self.kv_mode = kv_mode
        self.window_capacity = int(window_capacity)
        cfg = model.model.config
        self.layer_types = tuple(cfg.layer_types)
        self.config = QwenConfigView(name=model_id, vocab_size=int(cfg.vocab_size), d_model=int(cfg.hidden_size), layers=self.layer_types)
        self.delta_layer_ids = [i for i, kind in enumerate(self.layer_types) if kind == "linear_attention"]
        self.attention_layer_ids = [i for i, kind in enumerate(self.layer_types) if kind == "full_attention"]
        self.head_dim = int(cfg.head_dim)
        self.num_heads = int(cfg.num_attention_heads)
        self.num_kv_heads = int(cfg.num_key_value_heads)
        self.attention_backend: str | None = None
        self.stream_state_class = QwenStreamState
        #: 지렛대 (G0b S2.2): 결정 분기 배치 forward의 CUDA graph 재생, 층의 dense 부분 torch.compile.
        self.use_branch_graph = False
        self._branch_graph: _BranchGraph | None = None
        self.compiled = False
        #: 층 단위 activation checkpointing (gradient가 켜진 forward에서만; 틱 몸통의 층마다 입력 `[1, T, d]`만 남기고
        #: backward 때 그 층을 다시 계산한다). 10초 구간(≈45K 토큰)의 full/LoRA 학습은 이것 없이는 GB10의 통합 메모리를 넘긴다.
        self.activation_checkpointing = False

    # -- 만들기 --

    @classmethod
    def load(
        cls,
        model_id: str,
        *,
        root: str | Path | None = None,
        dtype: torch.dtype = torch.bfloat16,
        device: str = "cuda",
        kv_mode: str = "static",
        window_capacity: int = DEFAULT_WINDOW_CAPACITY,
        verify_full: bool = False,
    ) -> QwenBackbone:
        """`artifacts/models/<id>`의 가중치를 manifest와 대조한 뒤 싣는다 (`transformers`는 여기서만 import)."""
        from transformers import AutoModelForCausalLM
        from transformers.utils import logging as hf_logging

        described = describe_backbone(model_id, root, full=verify_full)
        hf_logging.disable_progress_bar()
        model = AutoModelForCausalLM.from_pretrained(described["path"], dtype=dtype)
        model.to(device)
        model.eval()
        model.requires_grad_(False)
        manifest = {key: described.get(key) for key in ("revision", "digest", "bytes_total", "verified", "fetched_at", "license")}
        return cls(model, model_id=model_id, manifest=manifest, kv_mode=kv_mode, window_capacity=window_capacity)

    @classmethod
    def tiny(cls, *, seed: int = 0, vocab_size: int = 512, layers: tuple[str, ...] | None = None, **overrides: Any) -> QwenBackbone:
        """검사용 소형 난수 Qwen3.5 (CPU, transformers의 torch 참조 경로). 가중치가 아니라 **구조**를 검사할 때 쓴다."""
        from transformers import Qwen3_5ForCausalLM
        from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig

        layers = layers or ("linear_attention", "linear_attention", "full_attention", "linear_attention", "full_attention")
        config = Qwen3_5TextConfig(
            vocab_size=vocab_size, hidden_size=64, intermediate_size=96, num_hidden_layers=len(layers), layer_types=list(layers),
            num_attention_heads=4, num_key_value_heads=2, head_dim=32, linear_num_key_heads=2, linear_num_value_heads=4,
            linear_key_head_dim=16, linear_value_head_dim=16, linear_conv_kernel_dim=4, max_position_embeddings=4096,
            rope_parameters={"rope_type": "default", "rope_theta": 10000.0, "partial_rotary_factor": 0.25, "mrope_section": [2, 1, 1], "mrope_interleaved": True},
            tie_word_embeddings=True, attn_implementation="sdpa", **overrides,
        )  # fmt: skip
        torch.manual_seed(seed)
        model = Qwen3_5ForCausalLM(config)
        model.eval()
        model.requires_grad_(False)
        return cls(model, model_id=f"tiny-qwen3_5-seed{seed}", kv_mode="static", window_capacity=256)

    # -- 읽기 --

    @property
    def text(self) -> Any:
        """Qwen3_5TextModel — `self.model.model`(같은 submodule을 두 이름으로 등록하지 않는다: state_dict 키가 겹친다)."""
        return self.model.model

    @property
    def grad_enabled(self) -> bool:
        return any(p.requires_grad for p in self.model.parameters())

    @property
    def device(self) -> torch.device:
        return self.text.embed_tokens.weight.device

    @property
    def dtype(self) -> torch.dtype:
        return self.text.embed_tokens.weight.dtype

    @property
    def delta_layers(self) -> list[Any]:
        return [self.text.layers[i].linear_attn for i in self.delta_layer_ids]

    @property
    def attention_layers(self) -> list[Any]:
        return [self.text.layers[i].self_attn for i in self.attention_layer_ids]

    def initial_state(self, batch: int = 1) -> list[dict[str, Tensor]]:
        """DeltaNet 층별 0 상태 — recurrent ``[B, Hv, dk, dv]`` fp32(런타임의 `mamba_ssm_dtype`), conv ``[B, C, K−1]``."""
        out = []
        for layer in self.delta_layers:
            out.append(
                {
                    "recurrent": torch.zeros(batch, layer.num_v_heads, layer.head_k_dim, layer.head_v_dim, dtype=torch.float32, device=self.device),
                    "conv": torch.zeros(batch, layer.conv_dim, layer.conv_kernel_size - 1, dtype=self.dtype, device=self.device),
                }
            )
        return out

    # -- 공식 forward 두 가지 --

    def forward(self, tokens: Tensor, positions: Tensor, *, mask: Tensor | None = None) -> dict[str, Any]:
        """오른쪽 padding한 배치의 plain causal forward → ``{"hidden": [B, T, d]}``. `mask`는 받지 않는다(causal뿐)."""
        if mask is not None:
            raise ValueError("QwenBackbone.forward: 명시적 mask는 reference_forward로 (여기는 causal 배치 경로뿐)")
        grad = self.grad_enabled and torch.is_grad_enabled()
        if grad and self.activation_checkpointing:
            # 공식 forward(P0 `state_first` 경로: 질문 경로 Q개 × S+T_i 행)도 층 단위 checkpointing — LoRA에서 8K 토큰 단위가 ≈60K 행이 되어
            # activation을 다 들고 있으면 ≈240 GB다. HF의 GradientCheckpointingLayer는 training 모드에서만 checkpoint하므로 forward 동안만
            # train()으로 둔다(Qwen3.5는 dropout 0 — 결과가 바뀌지 않는다).
            if not getattr(self.text, "gradient_checkpointing", False):
                self.text.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            was_training = self.text.training
            self.text.train()
            try:
                out = self.text(input_ids=tokens.to(self.device), position_ids=positions.to(self.device), use_cache=False)
            finally:
                self.text.train(was_training)
            return {"hidden": out.last_hidden_state}
        with torch.set_grad_enabled(grad):
            out = self.text(input_ids=tokens.to(self.device), position_ids=positions.to(self.device), use_cache=False)
        return {"hidden": out.last_hidden_state}

    def reference_forward(self, tokens: Tensor, positions: Tensor, mask: Tensor) -> Tensor:
        """명시적 기준 mask(``[T, T]`` 또는 ``[B, T, T]`` bool, True = 본다)를 물질화해 공식 forward에 넣는다 → ``[B, T, d]``."""
        if mask.dim() == 2:
            mask = mask[None]
        mask4 = mask[:, None].to(self.device)
        with torch.no_grad():
            out = self.text(
                input_ids=tokens.to(self.device), position_ids=positions.to(self.device),
                attention_mask={"full_attention": mask4, "linear_attention": None}, use_cache=False,
            )  # fmt: skip
        return out.last_hidden_state

    def forward_layout(
        self, layout: dict, *, window_ticks: int | None = None, decision_batch: int = 16, decision_ticks: list[int] | None = None
    ) -> Tensor:
        """스트림 layout을 **처음부터** — 몸통 토큰 전부를 기준 mask로 한 번, 틱마다 결정 분기를 ``몸통[:끝] + [d]`` 배치로.

        결정 토큰은 뒤 토큰이 없는 시퀀스 끝이라 transient 규칙이 그대로 성립한다(상태에 남을 다음 토큰이 없다).
        돌려주는 것은 layout 순서의 hidden ``[n, d]``. 증분 경로와 무관한 기준 계산이다. `decision_ticks`를 주면 그
        틱들의 결정만 계산한다(나머지 결정 행은 0 — 결정 배치가 틱마다 몸통 길이만큼 비싸서 검사가 범위를 제한한다).
        """
        n = len(layout["tokens"])
        window = int(layout.get("window_ticks", WINDOW_TICKS)) if window_ticks is None else int(window_ticks)
        mask = build_reference_mask(layout, window_ticks=window)
        tokens = torch.tensor(layout["tokens"], dtype=torch.long)
        positions = torch.tensor(layout["position"], dtype=torch.long)
        body = torch.tensor([kind != "decision" for kind in layout["kind"]], dtype=torch.bool)
        body_index = torch.nonzero(body)[:, 0]
        hidden = torch.zeros(n, self.config.d_model, dtype=self.dtype, device=self.device)
        body_hidden = self.reference_forward(tokens[body_index][None], positions[body_index][None], mask[body_index][:, body_index])
        hidden[body_index.to(self.device)] = body_hidden[0]
        for tick in layout["ticks"]:
            if decision_ticks is not None and int(tick["index"]) not in decision_ticks:
                continue
            body_end, end = int(tick["body_end"]), int(tick["end"])
            decisions = list(range(body_end, end))
            if not decisions:
                continue
            prefix_body = body_index[body_index < body_end]
            L = int(prefix_body.shape[0])
            for start in range(0, len(decisions), decision_batch):
                group = decisions[start : start + decision_batch]
                ids = torch.stack([torch.cat([tokens[prefix_body], tokens[d : d + 1]]) for d in group])
                pos = torch.stack([torch.cat([positions[prefix_body], positions[d : d + 1]]) for d in group])
                index = torch.cat([prefix_body, torch.tensor([group[0]])])
                sub = mask[index][:, index]  # 결정 행은 틱의 모든 결정에서 같다 (prefix + 윈도우 몸통 + 자기 자신)
                out = self.reference_forward(ids, pos, sub[None].expand(len(group), L + 1, L + 1))
                hidden[torch.tensor(group, device=self.device)] = out[:, -1]
        return hidden

    def kernel_names(self) -> dict[str, str | None]:
        return kernel_names(self.device)

    # -- 지렛대 --

    def release_branch_graph(self) -> None:
        """에피소드가 끝나면 캡처한 분기 graph를 버린다 (prefix 길이·버퍼 주소가 에피소드마다 다르다)."""
        self._branch_graph = None

    def compile_dense_parts(self) -> float | None:
        """층의 dense 부분(MLP, RMSNorm들, gated norm)을 `torch.compile(dynamic=True)`로 감싸고 예열 forward로 컴파일 시간을 잰다.

        fla·causal_conv1d·flash 호출은 그대로 둔다(Triton/CUDA 확장 kernel — compile 그래프 밖). 돌아오는 값은 예열에 든 초.
        """
        import time

        if self.compiled:
            return 0.0
        text = self.text
        for layer in text.layers:
            layer.mlp = torch.compile(layer.mlp, dynamic=True)
            layer.input_layernorm = torch.compile(layer.input_layernorm, dynamic=True)
            layer.post_attention_layernorm = torch.compile(layer.post_attention_layernorm, dynamic=True)
            if hasattr(layer, "linear_attn"):
                layer.linear_attn.norm = torch.compile(layer.linear_attn.norm, dynamic=True)
            else:
                layer.self_attn.q_norm = torch.compile(layer.self_attn.q_norm, dynamic=True)
                layer.self_attn.k_norm = torch.compile(layer.self_attn.k_norm, dynamic=True)
        text.norm = torch.compile(text.norm, dynamic=True)
        self.compiled = True
        started = time.perf_counter()
        state = QwenStreamState.initial(self, window_ticks=WINDOW_TICKS).extend_prefix(list(range(100, 164)))
        for length in (37, 130, 517):
            state = state.advance(list(range(200, 200 + length)))
            state.branch_step(list(range(65, 75)))
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        self.release_branch_graph()
        return round(time.perf_counter() - started, 1)

    # -- 층 조각 (스트림 경로가 쓴다) --

    def rotary(self, positions: Tensor) -> tuple[Tensor, Tensor]:
        """논리적 position ``[T]`` → (cos, sin) ``[1, T, rot]`` (공식 rotary 모듈, 텍스트 세 축 동일)."""
        pos = positions.to(self.device).view(1, 1, -1).expand(3, 1, -1)
        probe = self.text.embed_tokens.weight[:1]
        return self.text.rotary_emb(probe, pos)


# --------------------------------------------------------------------------
# 스트림 상태 — 정적 prefix + 미리 할당한 윈도우 KV, 배치 결정 분기
# --------------------------------------------------------------------------


class _WindowKV:
    """attention 층 하나의 KV 저장소 (모듈 설명 "mask 없는 attention").

    ``static``: prefix ``[P, Hk, D]`` 버퍼 + 윈도우 버퍼 ``[cap, Hk, D]``, 살아 있는 구간 ``[start, end)``. 틱 퇴출은
    ``start``를 옮기고, ``end + T > cap``일 때만 살아 있는 구간을 앞으로 당긴다(한 번의 복사). ``dynamic``: 틱별
    tensor 목록(그래프 유지) — 학습용.
    """

    __slots__ = ("mode", "k_prefix", "v_prefix", "k_win", "v_win", "start", "end", "chunks", "capacity", "heads", "dim", "dtype", "device")

    def __init__(self, *, mode: str, capacity: int, heads: int, dim: int, dtype: torch.dtype, device: torch.device) -> None:
        self.mode, self.capacity, self.heads, self.dim, self.dtype, self.device = mode, int(capacity), heads, dim, dtype, device
        self.k_prefix = torch.zeros(0, heads, dim, dtype=dtype, device=device)
        self.v_prefix = torch.zeros(0, heads, dim, dtype=dtype, device=device)
        self.k_win: Tensor | None = None
        self.v_win: Tensor | None = None
        self.start = self.end = 0
        self.chunks: list[tuple[Tensor, Tensor]] = []  # dynamic: 틱별 (k, v)

    def shallow(self) -> _WindowKV:
        """같은 tensor를 가리키는 새 저장소 객체 (dynamic 모드에서 이전 상태 객체의 목록을 보존한다)."""
        other = _WindowKV.__new__(_WindowKV)
        for name in self.__slots__:
            value = getattr(self, name)
            setattr(other, name, list(value) if isinstance(value, list) else value)
        return other

    def clone(self) -> _WindowKV:
        other = self.shallow()
        for name in ("k_prefix", "v_prefix", "k_win", "v_win"):
            value = getattr(self, name)
            if isinstance(value, Tensor):
                setattr(other, name, value.clone())
        other.chunks = [(k.clone(), v.clone()) for k, v in self.chunks]
        return other

    def detached(self, requires_grad: bool) -> _WindowKV:
        """그래프를 끊은 사본. 정적 버퍼에는 그래프가 없으므로 버퍼를 공유한다(복사 없음)."""
        other = self.shallow()

        def cut(t: Tensor) -> Tensor:
            t = t.detach()
            if requires_grad and t.is_floating_point():
                t.requires_grad_(True)
            return t

        other.k_prefix, other.v_prefix = cut(self.k_prefix), cut(self.v_prefix)
        other.chunks = [(cut(k), cut(v)) for k, v in self.chunks]
        return other

    @property
    def window_tokens(self) -> int:
        if self.mode == "dynamic":
            return sum(int(k.shape[0]) for k, _ in self.chunks)
        return self.end - self.start

    def set_prefix(self, k: Tensor, v: Tensor) -> None:
        self.k_prefix, self.v_prefix = k, v
        if self.mode == "static" and self.k_win is None:
            self.k_win = torch.empty(self.capacity, self.heads, self.dim, dtype=self.dtype, device=self.device)
            self.v_win = torch.empty(self.capacity, self.heads, self.dim, dtype=self.dtype, device=self.device)

    def segments(self) -> list[tuple[Tensor, Tensor]]:
        """전부 보이는 key 조각들: prefix, 윈도우 (dynamic은 틱별 조각)."""
        out = [(self.k_prefix, self.v_prefix)]
        if self.mode == "dynamic":
            if len(self.chunks) == 1:
                out.append(self.chunks[0])
            elif self.chunks:
                ks, vs = zip(*self.chunks)
                out.append((torch.cat(ks), torch.cat(vs)))
        elif self.k_win is not None and self.end > self.start:
            out.append((self.k_win[self.start : self.end], self.v_win[self.start : self.end]))
        return out

    def evict(self, tokens: int) -> None:
        """앞쪽(가장 오래된 틱)에서 `tokens`개를 윈도우 밖으로."""
        if tokens <= 0:
            return
        if self.mode == "dynamic":
            remaining = tokens
            while remaining > 0:
                k, v = self.chunks[0]
                if k.shape[0] <= remaining:
                    remaining -= int(k.shape[0])
                    self.chunks.pop(0)
                else:
                    self.chunks[0] = (k[remaining:], v[remaining:])
                    remaining = 0
            return
        self.start += tokens

    def append(self, k: Tensor, v: Tensor) -> None:
        T = int(k.shape[0])
        if self.mode == "dynamic":
            self.chunks.append((k, v))
            return
        if self.k_win is None:
            self.set_prefix(self.k_prefix, self.v_prefix)
        if self.end + T > self.capacity:
            live = self.end - self.start
            if live + T > self.capacity:
                raise RuntimeError(
                    f"윈도우 KV 버퍼 용량 {self.capacity} 토큰을 넘는다 (살아 있는 윈도우 {live} + 새 토큰 {T}) — window_capacity를 키운다"
                )
            self.k_win[:live].copy_(self.k_win[self.start : self.end].clone())
            self.v_win[:live].copy_(self.v_win[self.start : self.end].clone())
            self.start, self.end = 0, live
        self.k_win[self.end : self.end + T].copy_(k)
        self.v_win[self.end : self.end + T].copy_(v)
        self.end += T

    def gathered(self) -> dict[str, Tensor]:
        """검사용: 보이는 KV 전부를 ``{"k": [1, n, Hk, D], "v"}``로 모은다(복사)."""
        ks, vs = zip(*self.segments())
        return {"k": torch.cat(ks)[None], "v": torch.cat(vs)[None]}

    def buffer_bytes(self) -> int:
        total = (self.k_prefix.numel() + self.v_prefix.numel()) * self.k_prefix.element_size()
        if self.k_win is not None:
            total += (self.k_win.numel() + self.v_win.numel()) * self.k_win.element_size()
        for k, v in self.chunks:
            total += (k.numel() + v.numel()) * k.element_size()
        return int(total)


class QwenStreamState:
    """실제 backbone의 스트림 상태 — fixture `StreamState`와 같은 공개 API (모듈 설명 참조)."""

    def __init__(
        self,
        backbone: QwenBackbone,
        *,
        delta: list[dict[str, Tensor]],
        kv: list[_WindowKV],
        ticks: deque[tuple[int, int]],
        position: int,
        tick: int,
        window_ticks: int,
        prefix_hidden: Tensor | None = None,
        hidden: Tensor | None = None,
        is_branch: bool = False,
        prefix_len: int = 0,
    ) -> None:
        self.backbone = backbone
        self.delta = delta
        self._kv = kv
        self._ticks = ticks  # (틱, 토큰 수) — 윈도우 안의 틱, 오래된 것부터
        self.position = position
        self.tick = tick
        self.window_ticks = window_ticks
        self.prefix_hidden = prefix_hidden
        self.hidden = hidden
        self.is_branch = is_branch
        self.prefix_len = prefix_len

    # -- 만들기 --

    @classmethod
    def initial(
        cls, backbone: QwenBackbone, *, window_ticks: int | None = None, initial: list[dict[str, Tensor]] | None = None
    ) -> QwenStreamState:
        window = backbone.config.attention.window_ticks if window_ticks is None else int(window_ticks)
        if window < 1:
            raise ValueError(f"window_ticks: 1 이상이어야 한다 (받은 값: {window})")
        delta = backbone.initial_state(1) if initial is None else list(initial)
        if len(delta) != len(backbone.delta_layer_ids):
            raise ValueError(f"initial: DeltaNet 층 수 {len(backbone.delta_layer_ids)}개여야 한다 (받은 수: {len(delta)})")
        kv = [
            _WindowKV(
                mode=backbone.kv_mode, capacity=backbone.window_capacity, heads=backbone.num_kv_heads, dim=backbone.head_dim,
                dtype=backbone.dtype, device=backbone.device,
            )
            for _ in backbone.attention_layer_ids
        ]
        return cls(backbone, delta=delta, kv=kv, ticks=deque(), position=0, tick=-1, window_ticks=window)

    @classmethod
    def from_tokens(
        cls, prefix_tokens: Any, tick_tokens: Any, *, backbone: QwenBackbone, window_ticks: int | None = None,
        initial: list[dict[str, Tensor]] | None = None,
    ) -> QwenStreamState:  # fmt: skip
        state = cls.initial(backbone, window_ticks=window_ticks, initial=initial)
        prefix = list(prefix_tokens)
        if prefix:
            state = state.extend_prefix(prefix)
        ticks = list(tick_tokens)
        if ticks and all(isinstance(t, int) for t in ticks):
            ticks = [ticks]
        for tokens in ticks:
            state = state.advance(list(tokens))
        return state

    # -- 읽기 --

    @property
    def recurrent(self) -> list[Tensor]:
        return [layer["recurrent"] for layer in self.delta]

    @property
    def conv(self) -> list[Tensor]:
        return [layer["conv"] for layer in self.delta]

    @property
    def kv(self) -> list[dict[str, Tensor]]:
        """검사용 — 층별 보이는 KV를 모은 것(복사)."""
        return [layer.gathered() for layer in self._kv]

    @property
    def cache_ticks(self) -> Tensor:
        entries = [-1] * self.prefix_len
        for tick, count in self._ticks:
            entries.extend([tick] * count)
        return torch.tensor(entries, dtype=torch.long)

    @property
    def cached_tokens(self) -> int:
        return self.prefix_len + sum(count for _, count in self._ticks)

    @property
    def window_tokens(self) -> int:
        return sum(count for _, count in self._ticks)

    def kv_bytes(self) -> int:
        return sum(layer.buffer_bytes() for layer in self._kv)

    def __repr__(self) -> str:
        kind = "branch" if self.is_branch else "base"
        return f"QwenStreamState({kind}, tick={self.tick}, position={self.position}, cached={self.cached_tokens}, window={self.window_ticks})"

    # -- 계산 --

    def _grad(self) -> Any:
        return torch.set_grad_enabled(self.backbone.grad_enabled and torch.is_grad_enabled())

    def _run_body(
        self, tokens: list[int], *, write: bool, kv: list[_WindowKV] | None = None
    ) -> tuple[Tensor, list[dict[str, Tensor]], list[tuple[Tensor, Tensor]]]:
        """토큰 T개를 공통 상태에 이어 붙여 hidden ``[T, d]``, 새 DeltaNet 상태, 층별 새 (k, v)를 돌려준다. `write`면 KV를 저장소에 쓴다."""
        if not tokens:
            raise ValueError("tokens: 빈 토큰 목록은 이어 붙일 수 없다")
        kv = self._kv if kv is None else kv
        bb = self.backbone
        text = bb.text
        T = len(tokens)
        ids = torch.tensor(tokens, dtype=torch.long, device=bb.device)
        positions = torch.arange(self.position, self.position + T, device=bb.device)
        x = text.embed_tokens(ids)[None]  # [1, T, d]
        cos, sin = bb.rotary(positions)
        new_delta: list[dict[str, Tensor]] = []
        new_kv: list[tuple[Tensor, Tensor]] = []
        # 층 단위 activation checkpointing: gradient가 켜진 forward에서만 (추론 경로는 그대로)
        checkpointing = bb.activation_checkpointing and bb.grad_enabled and torch.is_grad_enabled()
        for index, layer in enumerate(text.layers):
            if bb.layer_types[index] == "linear_attention":
                state = self.delta[len(new_delta)]
                step = functools.partial(_delta_layer_step, layer, state["recurrent"], state["conv"])
                x, recurrent, conv_state = _maybe_checkpoint(step, x, enabled=checkpointing)
                new_delta.append({"recurrent": recurrent, "conv": conv_state})
            else:
                # 조각(prefix·윈도우 KV)은 지금 시점의 것을 넘긴다 — checkpoint의 재계산은 backward 때라 저장소가 그새 자랐을 수 있다
                step = functools.partial(_attention_layer_step, layer, cos, sin, kv[len(new_kv)].segments(), bb)
                x, k, v = _maybe_checkpoint(step, x, enabled=checkpointing)
                new_kv.append((k, v))
        hidden = text.norm(x)[0]
        if write:
            for store, (k, v) in zip(kv, new_kv):
                store.append(k, v)
        return hidden, new_delta, new_kv

    def extend_prefix(self, tokens: Any) -> QwenStreamState:
        """정적 prefix — 첫 틱 전의 공통 상태에만. KV는 prefix 버퍼로(윈도우 밖으로 나가지 않는다)."""
        if self.tick != -1 or self.is_branch or self.prefix_len:
            raise ValueError("prefix는 첫 틱 전의 공통 상태에 한 번만 붙일 수 있다")
        tokens = [int(t) for t in tokens]
        with self._grad():
            hidden, delta, new_kv = self._run_body(tokens, write=False)
        kv = [layer.shallow() if layer.mode == "dynamic" else layer for layer in self._kv]
        for store, (k, v) in zip(kv, new_kv):
            store.set_prefix(k, v)
        return QwenStreamState(
            self.backbone, delta=delta, kv=kv, ticks=deque(), position=self.position + len(tokens), tick=-1,
            window_ticks=self.window_ticks, prefix_hidden=hidden, hidden=None, prefix_len=len(tokens),
        )  # fmt: skip

    def advance(self, tokens: Any) -> QwenStreamState:
        """분기 이전 공통 상태에 다음 틱을 이어 붙인 새 상태. 윈도우 밖 틱의 KV를 먼저 내보낸다.

        정적 버퍼는 in-place로 이어 쓰므로 **이전 상태 객체는 이 호출 뒤에 더 쓰지 않는다**(fixture와 달리 값이
        분리되지 않는다 — 스냅샷이 필요하면 `clone()`을 먼저 한다). 분기(`branch_step`)는 어느 버퍼에도 쓰지 않는다.
        """
        if self.is_branch:
            raise ValueError("분기 상태에서는 다음 틱으로 이어갈 수 없다 — 다음 틱은 분기 이전 공통 상태에서 이어간다 (docs/08 §3.1)")
        tokens = [int(t) for t in tokens]
        tick = self.tick + 1
        ticks = deque(self._ticks)
        evicted = 0
        while ticks and tick - ticks[0][0] >= self.window_ticks:
            evicted += ticks.popleft()[1]
        kv = [layer.shallow() if layer.mode == "dynamic" else layer for layer in self._kv]
        for store in kv:
            store.evict(evicted)
        with self._grad():
            hidden, delta, _ = self._run_body(tokens, write=True, kv=kv)
        ticks.append((tick, len(tokens)))
        return QwenStreamState(
            self.backbone, delta=delta, kv=kv, ticks=ticks, position=self.position + len(tokens), tick=tick,
            window_ticks=self.window_ticks, prefix_hidden=self.prefix_hidden, hidden=hidden, prefix_len=self.prefix_len,
        )  # fmt: skip

    def branch_step(self, tokens: Any) -> Tensor:
        """결정 표지 n개를 공통 상태에서 갈라지는 **1토큰 분기 n개의 한 배치 forward**로 → hidden ``[n, d]``.

        분기 결과는 어디에도 남지 않는다(상태·버퍼 불변). fixture의 ``fork(n)`` + ``step`` n번과 같은 계산이다.
        """
        if self.is_branch:
            raise ValueError("분기에서 다시 분기하지 않는다")
        tokens = [int(t) for t in tokens]
        if not tokens:
            raise ValueError("tokens: 결정 토큰이 하나 이상 필요하다")
        bb = self.backbone
        if bb.use_branch_graph and bb.device.type == "cuda" and bb.kv_mode == "static" and not bb.grad_enabled and self.prefix_len:
            return self._branch_step_graphed(tokens)
        return self._branch_step_eager(tokens)

    def _branch_step_graphed(self, tokens: list[int]) -> Tensor:
        """지렛대 graphs: 같은 (분기 수, prefix 버퍼)의 graph를 에피소드마다 한 번 캡처하고 틱마다 재생한다."""
        bb = self.backbone
        graph = bb._branch_graph
        if graph is None or graph.n != len(tokens) or graph.buffer_key != _kv_buffer_key(self):
            graph = _BranchGraph(self, len(tokens))
            bb._branch_graph = graph
        return graph.replay(self, tokens)

    def _branch_step_eager(self, tokens: list[int], *, static: _BranchStatic | None = None) -> Tensor:
        bb = self.backbone
        text = bb.text
        n = len(tokens)
        if static is None:
            ids = torch.tensor(tokens, dtype=torch.long, device=bb.device)
            position = torch.full((1,), self.position, dtype=torch.long, device=bb.device)
            delta = self.delta
            windows = None
        else:  # graph 캡처/재생: 입력·상태·윈도우 경계가 정적 버퍼다
            ids, position, delta, windows = static.ids, static.position, static.delta, static.windows
        with self._grad():
            x = text.embed_tokens(ids)[:, None]  # [n, 1, d]
            cos, sin = bb.rotary(position)
            d_index = a_index = 0
            for index, layer in enumerate(text.layers):
                residual = x
                h = layer.input_layernorm(x)
                if bb.layer_types[index] == "linear_attention":
                    y = _delta_branches(layer.linear_attn, h, delta[d_index])
                    d_index += 1
                else:
                    y = _attention_branches(layer.self_attn, h, cos, sin, self._kv[a_index], bb, window=None if windows is None else windows[a_index])
                    a_index += 1
                x = residual + y
                x = x + layer.mlp(layer.post_attention_layernorm(x))
            return text.norm(x)[:, 0]

    def advance_with_branches(self, tokens: Any, decisions: Any) -> tuple[QwenStreamState, Tensor]:
        """틱 몸통 T개와 결정 표지 n개를 **한 forward**로 (지렛대 `fused`) → (새 공통 상태, 분기 hidden ``[n, d]``).

        `advance` 뒤 `branch_step`과 같은 계산이지만 층마다 projection·o_proj·MLP를 ``[T+n]`` 행 위에서 한 번 돌려 가중치를
        한 번만 읽는다(작은 모델의 forward 절편은 가중치 읽기다 — S2.5 귀속). DeltaNet은 몸통을 chunk kernel로 돌려 새 상태를
        만들고 분기는 그 상태의 transient 읽기, attention의 분기 query는 prefix·윈도우·몸통 전부와 자기 토큰을 본다. 분기는
        어디에도 남지 않는다.
        """
        if self.is_branch:
            raise ValueError("분기 상태에서는 다음 틱으로 이어갈 수 없다 — 다음 틱은 분기 이전 공통 상태에서 이어간다 (docs/08 §3.1)")
        tokens = [int(t) for t in tokens]
        decisions = [int(t) for t in decisions]
        if not tokens or not decisions:
            raise ValueError("tokens·decisions: 몸통 토큰과 결정 토큰이 하나 이상 필요하다")
        bb = self.backbone
        text = bb.text
        T, n = len(tokens), len(decisions)
        tick = self.tick + 1
        ticks = deque(self._ticks)
        evicted = 0
        while ticks and tick - ticks[0][0] >= self.window_ticks:
            evicted += ticks.popleft()[1]
        kv = [layer.shallow() if layer.mode == "dynamic" else layer for layer in self._kv]
        for store in kv:
            store.evict(evicted)
        with self._grad():
            ids = torch.tensor(tokens + decisions, dtype=torch.long, device=bb.device)
            positions = torch.cat([
                torch.arange(self.position, self.position + T, device=bb.device),
                torch.full((n,), self.position + T, dtype=torch.long, device=bb.device),
            ])  # fmt: skip
            x = text.embed_tokens(ids)[None]  # [1, T+n, d]
            cos, sin = bb.rotary(positions)
            new_delta: list[dict[str, Tensor]] = []
            new_kv: list[tuple[Tensor, Tensor]] = []
            for index, layer in enumerate(text.layers):
                residual = x
                h = layer.input_layernorm(x)
                if bb.layer_types[index] == "linear_attention":
                    raw, z, b, a = _delta_projections(layer.linear_attn, h)
                    y_body, state = _delta_body_from(layer.linear_attn, raw[:, :T], z[:, :T], b[:, :T], a[:, :T], self.delta[len(new_delta)])
                    branch_rows = lambda t: t[0, T:].unsqueeze(1)  # [n, 1, ·]  # noqa: E731
                    y_branch = _delta_branches_from(layer.linear_attn, branch_rows(raw), branch_rows(z), branch_rows(b), branch_rows(a), state)
                    new_delta.append(state)
                    y = torch.cat([y_body, y_branch.transpose(0, 1)], dim=1)
                else:
                    store = kv[len(new_kv)]
                    y, k, v = _attention_fused(layer.self_attn, h, cos, sin, store, bb, T)
                    new_kv.append((k, v))
                x = residual + y
                x = x + layer.mlp(layer.post_attention_layernorm(x))
            hidden = text.norm(x)[0]
        for store, (k, v) in zip(kv, new_kv):
            store.append(k, v)
        ticks.append((tick, T))
        state_out = QwenStreamState(
            self.backbone, delta=new_delta, kv=kv, ticks=ticks, position=self.position + T, tick=tick,
            window_ticks=self.window_ticks, prefix_hidden=self.prefix_hidden, hidden=hidden[:T], prefix_len=self.prefix_len,
        )  # fmt: skip
        return state_out, hidden[T:]

    def fork(self, n: int) -> list[_QwenBranch]:
        """fixture 호환 — 분기 n개. 각 분기의 ``step``은 그 토큰 하나의 배치 forward다(배치로 묶으려면 ``branch_step``)."""
        if not isinstance(n, int) or n < 1:
            raise ValueError(f"n: 1 이상의 정수여야 한다 (받은 값: {n!r})")
        return [_QwenBranch(self) for _ in range(n)]

    def clone(self) -> QwenStreamState:
        return QwenStreamState(
            self.backbone, delta=[{k: v.clone() for k, v in layer.items()} for layer in self.delta], kv=[layer.clone() for layer in self._kv],
            ticks=deque(self._ticks), position=self.position, tick=self.tick, window_ticks=self.window_ticks,
            prefix_hidden=None if self.prefix_hidden is None else self.prefix_hidden.clone(),
            hidden=None if self.hidden is None else self.hidden.clone(), is_branch=self.is_branch, prefix_len=self.prefix_len,
        )  # fmt: skip

    def detach(self, *, requires_grad: bool = False) -> QwenStreamState:
        """구간 경계에서 넘기는 공통 상태 — 모든 tensor를 detach한 새 상태 (값은 같고 gradient만 끊긴다)."""
        if self.is_branch:
            raise ValueError("branch 상태는 넘기지 않는다 — 다음 구간은 분기 이전 공통 상태에서 이어간다")

        def cut(t: Tensor | None) -> Tensor | None:
            if t is None:
                return None
            out = t.detach()
            if requires_grad and out.is_floating_point():
                out.requires_grad_(True)
            return out

        return QwenStreamState(
            self.backbone, delta=[{k: cut(v) for k, v in layer.items()} for layer in self.delta],
            kv=[layer.detached(requires_grad) for layer in self._kv], ticks=deque(self._ticks), position=self.position, tick=self.tick,
            window_ticks=self.window_ticks, prefix_hidden=cut(self.prefix_hidden), hidden=cut(self.hidden), prefix_len=self.prefix_len,
        )  # fmt: skip

    # -- checkpoint --

    def to_dict(self) -> dict[str, Any]:
        if self.is_branch:
            raise ValueError("branch 상태는 저장하지 않는다")
        return {
            "kind": "qwen",
            "delta": [{k: v.detach().clone() for k, v in layer.items()} for layer in self.delta],
            "kv": [{k: v.detach().clone()[0] for k, v in layer.gathered().items()} for layer in self._kv],
            "ticks": [[int(t), int(c)] for t, c in self._ticks],
            "prefix_len": int(self.prefix_len),
            "position": int(self.position),
            "tick": int(self.tick),
            "window_ticks": int(self.window_ticks),
            "prefix_hidden": None if self.prefix_hidden is None else self.prefix_hidden.detach().clone(),
            "hidden": None if self.hidden is None else self.hidden.detach().clone(),
        }

    @classmethod
    def from_dict(cls, packed: dict[str, Any], backbone: QwenBackbone) -> QwenStreamState:
        state = cls.initial(backbone, window_ticks=int(packed["window_ticks"]), initial=[dict(layer) for layer in packed["delta"]])
        P = int(packed["prefix_len"])
        ticks: deque[tuple[int, int]] = deque((int(t), int(c)) for t, c in packed["ticks"])
        for store, layer in zip(state._kv, packed["kv"]):
            k, v = layer["k"].to(backbone.device), layer["v"].to(backbone.device)
            store.set_prefix(k[:P], v[:P])
            cursor = P
            for _, count in ticks:
                store.append(k[cursor : cursor + count], v[cursor : cursor + count])
                cursor += count
        state._ticks = ticks
        state.prefix_len = P
        state.position, state.tick = int(packed["position"]), int(packed["tick"])
        state.prefix_hidden = None if packed.get("prefix_hidden") is None else packed["prefix_hidden"].to(backbone.device)
        state.hidden = None if packed.get("hidden") is None else packed["hidden"].to(backbone.device)
        return state


class _QwenBranch:
    """fixture 호환 분기 객체 — ``step(token)``이 부모의 ``branch_step([token])``이다. 부모·형제에 닿지 않는다."""

    __slots__ = ("parent", "position", "tick", "is_branch", "hidden")

    def __init__(self, parent: QwenStreamState) -> None:
        self.parent = parent
        self.position = parent.position
        self.tick = parent.tick
        self.is_branch = True
        self.hidden: Tensor | None = None

    def step(self, token: int | Tensor) -> Tensor:
        if isinstance(token, Tensor):
            token = int(token.item())
        out = self.parent.branch_step([token])[0]
        self.hidden = out[None]
        self.position += 1
        return out

    def advance(self, tokens: Any) -> QwenStreamState:
        raise ValueError("분기 상태에서는 다음 틱으로 이어갈 수 없다 — 다음 틱은 분기 이전 공통 상태에서 이어간다 (docs/08 §3.1)")


# --------------------------------------------------------------------------
# 층 계산 — 몸통(fla·causal_conv1d kernel)과 분기(공유 상태의 transient 읽기)
# --------------------------------------------------------------------------


def _kernels() -> dict[str, Any]:
    """CUDA kernel(fla·causal_conv1d)과 transformers의 torch 참조 — 호출 때 tensor의 장치로 고른다."""
    import transformers.models.qwen3_5.modeling_qwen3_5 as modeling

    out: dict[str, Any] = {"apply_rotary": modeling.apply_rotary_pos_emb, "chunk_torch": _torch_chunk(modeling), "chunk_cuda": None, "conv_cuda": None}
    try:
        from fla.ops.gated_delta_rule import chunk_gated_delta_rule

        out["chunk_cuda"] = chunk_gated_delta_rule
    except ImportError:  # pragma: no cover — fla 없는 환경은 torch 참조로 돈다
        pass
    try:
        from causal_conv1d import causal_conv1d_fn

        out["conv_cuda"] = causal_conv1d_fn
    except ImportError:  # pragma: no cover
        pass
    return out


def _torch_chunk(modeling: Any) -> Any:
    import inspect

    reference = inspect.getclosurevars(modeling.torch_chunk_gated_delta_rule).nonlocals.get("torch_function")
    return reference if reference is not None else modeling.torch_chunk_gated_delta_rule


_KERNELS: dict[str, Any] | None = None


def kernels() -> dict[str, Any]:
    global _KERNELS
    if _KERNELS is None:
        _KERNELS = _kernels()
    return _KERNELS


def kernel_names(device: torch.device | str = "cuda") -> dict[str, str | None]:
    """어느 구현이 도는지 (보고서용) — `device`의 tensor로 호출했을 때."""
    k = kernels()
    cuda = str(device).startswith("cuda")
    chunk = k["chunk_cuda"] if cuda and k["chunk_cuda"] is not None else k["chunk_torch"]
    conv = k["conv_cuda"] if cuda and k["conv_cuda"] is not None else None
    return {
        "chunk_gated_delta_rule": f"{getattr(chunk, '__module__', None)}.{getattr(chunk, '__name__', None)}",
        "causal_conv1d": "torch.nn.functional.conv1d" if conv is None else f"{conv.__module__}.{conv.__name__}",
        "attention": "aten::_flash_attention_forward (varlen, no mask)" if cuda else "aten::_scaled_dot_product_flash_attention_for_cpu (no mask)",
    }


_HUB_FUNCTIONS = ("torch_chunk_gated_delta_rule", "torch_recurrent_gated_delta_rule", "causal_conv1d_fn", "causal_conv1d_update")


class torch_reference_kernels:
    """transformers의 공식 forward를 **torch 참조 구현**으로 돌리는 context manager (CPU 검사용).

    `use_kernel_func_from_hub_with_fallback`는 import 시점에 CUDA가 있으면 fla·causal_conv1d를 고르고 CPU tensor에도
    그것을 부른다(실패). 소형 난수 모델을 CPU에서 공식 forward와 대조하려면 그 선택을 잠시 torch 참조로 바꾼다 —
    closure cell을 바꾸는 것이라 검사 밖에서는 쓰지 않는다.
    """

    def __enter__(self) -> None:
        import transformers.models.qwen3_5.modeling_qwen3_5 as modeling

        self._saved: list[tuple[Any, Any]] = []
        for name in _HUB_FUNCTIONS:
            fn = getattr(modeling, name)
            cells = dict(zip(fn.__code__.co_freevars, fn.__closure__ or ()))
            if "implementation" not in cells or "torch_function" not in cells:
                continue
            cell = cells["implementation"]
            self._saved.append((cell, cell.cell_contents))
            cell.cell_contents = cells["torch_function"].cell_contents

    def __exit__(self, *exc: object) -> None:
        for cell, value in self._saved:
            cell.cell_contents = value


def _maybe_checkpoint(step: Any, x: Tensor, *, enabled: bool) -> Any:
    """`enabled`면 층 하나를 non-reentrant activation checkpoint로 (입력 `x`만 남기고 backward에서 다시 계산), 아니면 그대로 부른다."""
    if not enabled:
        return step(x)
    from torch.utils.checkpoint import checkpoint

    return checkpoint(step, x, use_reentrant=False)


def _delta_layer_step(layer: Any, recurrent: Tensor, conv: Tensor, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """DeltaNet 층 하나: pre-norm → 몸통 → residual → MLP. ``x [1, T, d]`` → (x, 새 recurrent, 새 conv history)."""
    y, new_state = _delta_body(layer.linear_attn, layer.input_layernorm(x), {"recurrent": recurrent, "conv": conv})
    x = x + y
    x = x + layer.mlp(layer.post_attention_layernorm(x))
    return x, new_state["recurrent"], new_state["conv"]


def _attention_layer_step(
    layer: Any, cos: Tensor, sin: Tensor, segments: list[tuple[Tensor, Tensor]], bb: QwenBackbone, x: Tensor
) -> tuple[Tensor, Tensor, Tensor]:
    """full-attention 층 하나: pre-norm → 윈도우 attention(주어진 조각 위) → residual → MLP. ``x [1, T, d]`` → (x, k, v)."""
    y, k, v = _attention_body_on(layer.self_attn, layer.input_layernorm(x), cos, sin, segments, bb)
    x = x + y
    x = x + layer.mlp(layer.post_attention_layernorm(x))
    return x, k, v


def _conv_body(layer: Any, raw: Tensor, conv_state: Tensor) -> tuple[Tensor, Tensor]:
    """depthwise causal conv: ``raw [1, T, C]`` + history ``[1, C, K−1]`` → 활성화 뒤 ``[1, T, C]``, 새 history."""
    K = layer.conv_kernel_size
    x = raw.transpose(1, 2)  # [1, C, T]
    seq = torch.cat([conv_state.to(x.dtype), x], dim=2)  # [1, C, K−1+T]
    conv = kernels()["conv_cuda"]
    weight = layer.conv1d.weight.squeeze(1)
    if conv is not None and x.is_cuda:
        out = conv(seq, weight, layer.conv1d.bias, activation="silu")[:, :, K - 1 :]
    else:
        out = F.conv1d(seq.to(weight.dtype), layer.conv1d.weight, layer.conv1d.bias, groups=layer.conv_dim)
        out = F.silu(out).to(x.dtype)
    return out.transpose(1, 2), seq[:, :, -(K - 1) :].contiguous()


def _delta_projections(layer: Any, h: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """DeltaNet 층의 네 projection ``(raw qkv, z, b, a)`` — 몸통·분기를 합친 행 위에서 한 번에(가중치를 한 번 읽는다)."""
    return layer.in_proj_qkv(h), layer.in_proj_z(h), layer.in_proj_b(h), layer.in_proj_a(h)


def _delta_body(layer: Any, h: Tensor, state: dict[str, Tensor]) -> tuple[Tensor, dict[str, Tensor]]:
    """gated DeltaNet 몸통 — 공식 forward의 계산을 명시적 상태 입출력으로 (chunk kernel, `initial_state`/`output_final_state`)."""
    return _delta_body_from(layer, *_delta_projections(layer, h), state)


def _delta_body_from(layer: Any, raw: Tensor, z_raw: Tensor, b: Tensor, a: Tensor, state: dict[str, Tensor]) -> tuple[Tensor, dict[str, Tensor]]:
    B, T, _ = raw.shape
    mixed, conv_state = _conv_body(layer, raw, state["conv"])
    z = z_raw.reshape(B, T, -1, layer.head_v_dim)
    beta = b.sigmoid()
    g = -layer.A_log.float().exp() * F.softplus(a.float() + layer.dt_bias)
    query, key, value = torch.split(mixed, [layer.key_dim, layer.key_dim, layer.value_dim], dim=-1)
    query = query.reshape(B, T, -1, layer.head_k_dim)
    key = key.reshape(B, T, -1, layer.head_k_dim)
    value = value.reshape(B, T, -1, layer.head_v_dim)
    rep = layer.num_v_heads // layer.num_k_heads
    if rep > 1:
        query = query.repeat_interleave(rep, dim=2)
        key = key.repeat_interleave(rep, dim=2)
    chunk = kernels()["chunk_cuda"] if query.is_cuda and kernels()["chunk_cuda"] is not None else kernels()["chunk_torch"]
    out, recurrent = chunk(
        query, key, value, g=g, beta=beta, initial_state=state["recurrent"], output_final_state=True, use_qk_l2norm_in_kernel=True,
    )
    out = layer.norm(out.reshape(-1, layer.head_v_dim), z.reshape(-1, layer.head_v_dim)).reshape(B, T, -1)
    return layer.out_proj(out), {"recurrent": recurrent.to(torch.float32), "conv": conv_state}


def _l2norm(x: Tensor) -> Tensor:
    return x * torch.rsqrt((x * x).sum(dim=-1, keepdim=True) + _L2NORM_EPS)


def _delta_branches(layer: Any, h: Tensor, state: dict[str, Tensor]) -> Tensor:
    """분기 n개의 1토큰 DeltaNet — 공통 상태 S를 복제하지 않고 ``Sᵀq``·``Sᵀk``만 읽어 분기별 갱신·출력을 계산한다.

    ``S' = αS + β k (v − α Sᵀk)ᵀ``, ``o = S'ᵀ q = α (Sᵀq) + β (kᵀq)(v − α Sᵀk)`` — fp32에서 (kernel과 같은 정밀도).
    ``h [n, 1, d]`` → ``[n, 1, d]``.
    """
    return _delta_branches_from(layer, *_delta_projections(layer, h), state)


def _delta_branches_from(layer: Any, raw: Tensor, z_raw: Tensor, b: Tensor, a: Tensor, state: dict[str, Tensor]) -> Tensor:
    """분기 계산 본체 — projection ``raw [n, 1, C]``·``z_raw``·``b``·``a``(모두 ``[n, 1, ·]``)를 받는다."""
    n = raw.shape[0]
    K = layer.conv_kernel_size
    window = torch.cat([state["conv"].to(raw.dtype).expand(n, -1, -1), raw.transpose(1, 2)], dim=2)  # [n, C, K]
    weight = layer.conv1d.weight.squeeze(1)  # [C, K]
    mixed = (window.float() * weight.float()[None]).sum(-1)
    if layer.conv1d.bias is not None:
        mixed = mixed + layer.conv1d.bias.float()[None]
    mixed = F.silu(mixed).to(raw.dtype)  # [n, C] (kernel처럼 bf16으로 되돌린 뒤 정규화)
    query, key, value = torch.split(mixed, [layer.key_dim, layer.key_dim, layer.value_dim], dim=-1)
    query = query.reshape(n, -1, layer.head_k_dim).float()
    key = key.reshape(n, -1, layer.head_k_dim).float()
    value = value.reshape(n, -1, layer.head_v_dim).float()
    rep = layer.num_v_heads // layer.num_k_heads
    if rep > 1:
        query = query.repeat_interleave(rep, dim=1)
        key = key.repeat_interleave(rep, dim=1)
    query = _l2norm(query) * (layer.head_k_dim**-0.5)
    key = _l2norm(key)
    beta = b[:, 0].sigmoid().float()  # [n, Hv] — kernel처럼 bf16 sigmoid 뒤 fp32
    g = -layer.A_log.float().exp() * F.softplus(a[:, 0].float() + layer.dt_bias)
    alpha = g.exp()[..., None]  # [n, Hv, 1]
    S = state["recurrent"][0].float()  # [Hv, dk, dv]
    Sq = torch.einsum("hkv,nhk->nhv", S, query)
    Sk = torch.einsum("hkv,nhk->nhv", S, key)
    kq = (key * query).sum(-1, keepdim=True)  # [n, Hv, 1]
    out = alpha * Sq + beta[..., None] * kq * (value - alpha * Sk)
    z = z_raw.reshape(-1, layer.head_v_dim)
    out = layer.norm(out.to(raw.dtype).reshape(-1, layer.head_v_dim), z).reshape(n, 1, -1)
    return layer.out_proj(out)


def _project_qkv(layer: Any, h: Tensor, cos: Tensor, sin: Tensor, bb: QwenBackbone) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """``h [B, T, d]`` → q ``[B, T, H, D]``(RoPE·norm 뒤), gate ``[B, T, H·D]``, k ``[B, T, Hk, D]``, v."""
    apply_rotary = kernels()["apply_rotary"]
    B, T, _ = h.shape
    query, gate = torch.chunk(layer.q_proj(h).view(B, T, -1, layer.head_dim * 2), 2, dim=-1)
    gate = gate.reshape(B, T, -1)
    query = layer.q_norm(query.reshape(B, T, -1, layer.head_dim)).transpose(1, 2)  # [B, H, T, D]
    key = layer.k_norm(layer.k_proj(h).view(B, T, -1, layer.head_dim)).transpose(1, 2)
    value = layer.v_proj(h).view(B, T, -1, layer.head_dim)
    query, key = apply_rotary(query, key, cos, sin)
    return query.transpose(1, 2), gate, key.transpose(1, 2), value


def _attention_fused(layer: Any, h: Tensor, cos: Tensor, sin: Tensor, store: _WindowKV, bb: QwenBackbone, T: int) -> tuple[Tensor, Tensor, Tensor]:
    """몸통 T개 + 분기 n개의 attention을 한 projection으로: 몸통은 prefix·윈도우 + 자기 causal, 분기는 prefix·윈도우·몸통 전부 + 자기 토큰.
    ``h [1, T+n, d]`` → (``[1, T+n, d]``, 몸통 k, 몸통 v)."""
    query, gate, key, value = _project_qkv(layer, h, cos, sin, bb)
    q_all, k_all, v_all = query[0], key[0], value[0]  # [T+n, H, D], [T+n, Hk, D]
    q_body, k_body, v_body = q_all[:T], k_all[:T], v_all[:T]
    q_branch, k_branch, v_branch = q_all[T:], k_all[T:], v_all[T:]
    segments = [(k, v) for k, v in store.segments() if k.shape[0]]
    out_body = windowed_attention(q_body, segments, (k_body, v_body), scale=layer.scaling)
    parts = [_flash_part(q_branch, k, v, causal=False, scale=layer.scaling) for k, v in segments]
    parts.append(_flash_part(q_branch, k_body, v_body, causal=False, scale=layer.scaling))  # 몸통 전부가 보인다
    g = bb.num_heads // bb.num_kv_heads
    k_own = k_branch.repeat_interleave(g, dim=1) if g > 1 else k_branch
    v_own = v_branch.repeat_interleave(g, dim=1) if g > 1 else v_branch
    own_score = (q_branch.float() * k_own.float()).sum(-1) * layer.scaling  # [n, H]
    parts.append(AttentionParts(v_own, own_score))
    out_branch = merge_attention_parts(parts)
    out = torch.cat([out_body, out_branch]).reshape(1, q_all.shape[0], -1) * torch.sigmoid(gate)
    if bb.attention_backend is None:
        bb.attention_backend = kernel_names(q_all.device)["attention"]
    return layer.o_proj(out), k_body, v_body


def _attention_body(layer: Any, h: Tensor, cos: Tensor, sin: Tensor, store: _WindowKV, bb: QwenBackbone) -> tuple[Tensor, Tensor, Tensor]:
    """틱 몸통의 full attention: prefix·윈도우(전부) + 자기 틱(causal), mask 없이. ``h [1, T, d]`` → (``[1, T, d]``, k, v)."""
    return _attention_body_on(layer, h, cos, sin, store.segments(), bb)


def _attention_body_on(
    layer: Any, h: Tensor, cos: Tensor, sin: Tensor, segments: list[tuple[Tensor, Tensor]], bb: QwenBackbone
) -> tuple[Tensor, Tensor, Tensor]:
    """:func:`_attention_body` 의 본체 — 보이는 KV 조각을 인자로 받는다 (checkpoint 재계산이 같은 조각을 쓰도록)."""
    query, gate, key, value = _project_qkv(layer, h, cos, sin, bb)
    q, k, v = query[0], key[0], value[0]  # [T, H, D], [T, Hk, D]
    out = windowed_attention(q, segments, (k, v), scale=layer.scaling)
    out = out.reshape(1, q.shape[0], -1) * torch.sigmoid(gate)
    if bb.attention_backend is None:
        bb.attention_backend = kernel_names(q.device)["attention"]
    return layer.o_proj(out), k, v


def _attention_branches(
    layer: Any, h: Tensor, cos: Tensor, sin: Tensor, store: _WindowKV, bb: QwenBackbone, *, window: _StaticWindow | None = None
) -> Tensor:
    """분기 n개의 1토큰 attention: 공유 KV 위의 query n개를 한 flash 호출로, 자기 토큰은 logsumexp로 합친다. ``h [n, 1, d]``.

    `window`(graph 캡처용)가 있으면 윈도우 조각을 정적 버퍼 전체 위의 varlen 호출(경계는 장치 tensor)로 돈다.
    """
    n = h.shape[0]
    query, gate, key, value = _project_qkv(layer, h, cos.expand(n, -1, -1), sin.expand(n, -1, -1), bb)
    q = query[:, 0]  # [n, H, D] — 모두 같은 position; 분기 n개가 flash 호출의 query 토큰 n개다
    if window is None:
        parts = [_flash_part(q, k, v, causal=False, scale=layer.scaling) for k, v in store.segments() if k.shape[0]]
    else:
        parts = [_flash_part_static(q, store.k_prefix, store.v_prefix, window.prefix_cu, layer.scaling)]
        parts.append(_flash_part_static(q, store.k_win, store.v_win, window.window_cu, layer.scaling, seqused=window.seqused, capacity=store.capacity))
    g = bb.num_heads // bb.num_kv_heads
    k_own = key[:, 0].repeat_interleave(g, dim=1) if g > 1 else key[:, 0]  # [n, H, D]
    v_own = value[:, 0].repeat_interleave(g, dim=1) if g > 1 else value[:, 0]
    own_score = (q.float() * k_own.float()).sum(-1) * layer.scaling  # [n, H] — 자기 토큰 하나의 점수
    parts.append(AttentionParts(v_own, own_score))
    out = merge_attention_parts(parts)
    out = out.reshape(n, 1, -1) * torch.sigmoid(gate)
    return layer.o_proj(out)
