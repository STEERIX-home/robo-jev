"""Hybrid 상태 분기와 소형 hybrid 계산 fixture (docs/03 §3, docs/06 Task 4, docs/08 §3.1).

세 가지를 둔다.

* :func:`fork_delta_state` — 한 층의 DeltaNet 상태(``recurrent``·``conv``)를 분기 수만큼 **독립
  버퍼**로 복제하되 autograd 그래프는 유지한다. 분기의 in-place 갱신이 형제·부모에 닿지 않고,
  분기에서 온 gradient는 부모에 합쳐진다 (계획서의 두 최소 예시).
* 기준(reference) 층 — 순수 PyTorch, kernel 없음. :class:`GatedDeltaNetLayer`는 명시적 recurrent
  상태 ``[B, H, dk, dv]``와 짧은 causal conv의 history ``[B, C, kernel-1]``을 입출력하며,
  ``step``(토큰 하나)과 ``forward``(시퀀스)의 결과가 같다. ``transient`` 토큰(결정 위치 같은 일시적
  1토큰 분기)은 자기 갱신을 **읽되 상태에 남기지 않는다** — fork → step → 폐기와 같은 계산이다.
  :class:`WindowedAttentionLayer`는 RoPE + 명시적 query×key mask + KV cache다. mask는
  :func:`robo_jev.model.attention.build_reference_mask`(처음부터 계산) 또는 스트림 상태가 윈도우
  규칙으로 만든 것(증분 계산)이다.
* :class:`TinyHybrid` — ``configs/model/tiny_hybrid.yaml``의 소형 backbone(d_model 64, DeltaNet 2층 +
  full attention 1층, 어휘 = 실제 tokenizer 크기). **계산 fixture이지 연구 모델이 아니다**
  (docs/06 §2): mask·상태 분기·gradient의 수학적 정합성만 검사하며 성능 비교에 쓰지 않는다.
  가중치는 seed로 고정한 난수다.

DeltaNet 갱신(gated delta rule, 한 head, ``S ∈ R^{dk×dv}``)::

    S ← α_t S
    S ← S + β_t k_t (v_t − Sᵀ k_t)ᵀ
    o_t = Sᵀ q_t                      (q, k는 L2 정규화, q는 1/√dk 배)

``α_t = exp(−exp(A_log)·softplus(a_t + dt_bias)) ∈ (0, 1)``, ``β_t = sigmoid(b_t)``. 출력은 head별
gated RMSNorm(``norm(o)·silu(z)``) 뒤 projection이다. q·k·v projection 앞의 depthwise causal conv
(kernel 4, SiLU)가 공개 hybrid 구현과 같은 자리에 있다. 상태는 어느 경로에서도 in-place로 갱신하지
않는다 — 새 tensor를 만든다.

이 모듈은 generator·simulator·하네스를 import하지 않는다 (docs/06 §1).
"""

from __future__ import annotations

import functools
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
import yaml
from torch import Tensor, nn

__all__ = [
    "DEFAULT_CONFIG",
    "AttentionConfig",
    "DeltaNetConfig",
    "GatedDeltaNetLayer",
    "HybridConfig",
    "RMSNorm",
    "TinyHybrid",
    "WindowedAttentionLayer",
    "causal_mask",
    "default_backbone",
    "fork_delta_state",
]

#: 소형 fixture 설정 (저장소 뿌리 기준).
DEFAULT_CONFIG = Path(__file__).resolve().parents[3] / "configs" / "model" / "tiny_hybrid.yaml"


# --------------------------------------------------------------------------
# 상태 분기
# --------------------------------------------------------------------------


def _fork_one(node: Any) -> Any:
    if isinstance(node, Tensor):
        return node.clone()  # 새 버퍼, autograd로 원본에 이어진다
    if isinstance(node, dict):
        return {key: _fork_one(value) for key, value in node.items()}
    if isinstance(node, (list, tuple)):
        return type(node)(_fork_one(value) for value in node)
    return node


def fork_delta_state(state: dict, branches: int) -> list[dict]:
    """한 층의 ``recurrent``·``conv`` tensor를 ``branches``개의 독립 버퍼로 분기한다.

    각 분기는 원본의 ``clone`` — 저장 공간이 분리되어 한 분기의 in-place 갱신이 형제나 원본에
    닿지 않고, gradient는 분기들에서 원본으로 합쳐진다. 중첩 dict/list와 `None`도 그대로 따라간다.
    """
    if not isinstance(branches, int) or branches < 1:
        raise ValueError(f"branches: 1 이상의 정수여야 한다 (받은 값: {branches!r})")
    if not isinstance(state, dict):
        raise ValueError(f"state: dict여야 한다 (받은 값: {type(state).__name__})")
    return [_fork_one(state) for _ in range(branches)]


# --------------------------------------------------------------------------
# 공통 조각
# --------------------------------------------------------------------------


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        variance = x.float().pow(2).mean(dim=-1, keepdim=True)
        return (x * torch.rsqrt(variance + self.eps)) * self.weight


def causal_mask(length: int, cache: int = 0) -> Tensor:
    """``[length, cache + length]`` bool — cache는 전부 보고, 새 토큰끼리는 causal."""
    new = torch.tril(torch.ones(length, length, dtype=torch.bool))
    if cache == 0:
        return new
    return torch.cat([torch.ones(length, cache, dtype=torch.bool), new], dim=1)


# --------------------------------------------------------------------------
# 기준 gated DeltaNet 층
# --------------------------------------------------------------------------


class GatedDeltaNetLayer(nn.Module):
    """명시적 상태를 입출력하는 기준 gated DeltaNet (모듈 설명 참조).

    상태: ``{"recurrent": [B, H, dk, dv], "conv": [B, C, kernel-1]}``, ``C = 2·H·dk + H·dv``.
    """

    def __init__(
        self, d_model: int, heads: int, head_k: int, head_v: int, conv_kernel: int, eps: float = 1e-6
    ) -> None:
        super().__init__()
        if conv_kernel < 2:
            raise ValueError(f"conv_kernel: 2 이상이어야 한다 (받은 값: {conv_kernel})")
        self.heads, self.head_k, self.head_v, self.kernel = heads, head_k, head_v, conv_kernel
        self.channels = 2 * heads * head_k + heads * head_v
        self.qkv_proj = nn.Linear(d_model, self.channels, bias=False)
        self.conv_weight = nn.Parameter(torch.empty(self.channels, conv_kernel))
        self.conv_bias = nn.Parameter(torch.zeros(self.channels))
        self.beta_proj = nn.Linear(d_model, heads, bias=False)
        self.a_proj = nn.Linear(d_model, heads, bias=False)
        self.A_log = nn.Parameter(torch.zeros(heads))
        self.dt_bias = nn.Parameter(torch.zeros(heads))
        self.z_proj = nn.Linear(d_model, heads * head_v, bias=False)
        self.o_norm = RMSNorm(head_v, eps)
        self.o_proj = nn.Linear(heads * head_v, d_model, bias=False)
        self.reset_parameters()

    def reset_parameters(self, generator: torch.Generator | None = None) -> None:
        with torch.no_grad():
            bound = 1.0 / math.sqrt(self.kernel)
            self.conv_weight.uniform_(-bound, bound, generator=generator)
            self.conv_bias.zero_()
            # Mamba2/GDN 관례: A ~ U(1, 16), dt = softplus(dt_bias) ~ logU(1e-3, 1e-1)
            self.A_log.copy_(torch.log(torch.empty(self.heads).uniform_(1.0, 16.0, generator=generator)))
            dt = torch.exp(
                torch.empty(self.heads).uniform_(math.log(1e-3), math.log(1e-1), generator=generator)
            )
            self.dt_bias.copy_(dt + torch.log(-torch.expm1(-dt)))

    # -- 상태 --

    def initial_state(self, batch: int = 1, *, requires_grad: bool = False) -> dict[str, Tensor]:
        weight = self.qkv_proj.weight
        return {
            "recurrent": torch.zeros(
                batch, self.heads, self.head_k, self.head_v, dtype=weight.dtype, requires_grad=requires_grad
            ),
            "conv": torch.zeros(
                batch, self.channels, self.kernel - 1, dtype=weight.dtype, requires_grad=requires_grad
            ),
        }

    def _check_state(self, state: dict, batch: int) -> None:
        for key in ("recurrent", "conv"):
            if key not in state:
                raise ValueError(f"state.{key}: 없다")
        expect_r = (batch, self.heads, self.head_k, self.head_v)
        expect_c = (batch, self.channels, self.kernel - 1)
        if tuple(state["recurrent"].shape) != expect_r:
            raise ValueError(f"state.recurrent: {expect_r}이어야 한다 (받은 값: {tuple(state['recurrent'].shape)})")
        if tuple(state["conv"].shape) != expect_c:
            raise ValueError(f"state.conv: {expect_c}이어야 한다 (받은 값: {tuple(state['conv'].shape)})")

    # -- 조각 계산 --

    def _split(self, mixed: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """conv 뒤의 채널 ``[..., C]`` → q ``[..., H, dk]``, k, v ``[..., H, dv]`` (정규화 포함)."""
        H, dk, dv = self.heads, self.head_k, self.head_v
        q, k, v = torch.split(mixed, [H * dk, H * dk, H * dv], dim=-1)
        q = F.normalize(q.reshape(*q.shape[:-1], H, dk), dim=-1) * (dk**-0.5)
        k = F.normalize(k.reshape(*k.shape[:-1], H, dk), dim=-1)
        v = v.reshape(*v.shape[:-1], H, dv)
        return q, k, v

    def _gates(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """``[..., d]`` → α ``[..., H]``(decay), β ``[..., H]``(update), z ``[..., H, dv]``(output gate)."""
        alpha = torch.exp(-torch.exp(self.A_log) * F.softplus(self.a_proj(x) + self.dt_bias))
        beta = torch.sigmoid(self.beta_proj(x))
        z = self.z_proj(x).reshape(*x.shape[:-1], self.heads, self.head_v)
        return alpha, beta, z

    @staticmethod
    def _recur(S: Tensor, q: Tensor, k: Tensor, v: Tensor, alpha: Tensor, beta: Tensor) -> tuple[Tensor, Tensor]:
        """gated delta rule 한 스텝. ``S [B,H,dk,dv]``, q/k ``[B,H,dk]``, v ``[B,H,dv]``, α/β ``[B,H]``."""
        S = S * alpha[..., None, None]
        predicted = (S * k[..., None]).sum(dim=-2)  # Sᵀ k  → [B,H,dv]
        delta = (v - predicted) * beta[..., None]
        S = S + k[..., None] * delta[..., None, :]
        out = (S * q[..., None]).sum(dim=-2)  # Sᵀ q  → [B,H,dv]
        return out, S

    def _output(self, o: Tensor, z: Tensor) -> Tensor:
        gated = self.o_norm(o) * F.silu(z)
        return self.o_proj(gated.reshape(*gated.shape[:-2], self.heads * self.head_v))

    # -- 공개 API --

    def step(self, x: Tensor, state: dict) -> tuple[Tensor, dict]:
        """토큰 하나 ``x [B, d]`` → ``(y [B, d], 새 상태)``. 입력 상태는 건드리지 않는다."""
        if x.dim() != 2:
            raise ValueError(f"x: [B, d]여야 한다 (받은 모양: {tuple(x.shape)})")
        self._check_state(state, x.shape[0])
        raw = self.qkv_proj(x)  # [B, C]
        window = torch.cat([state["conv"], raw[:, :, None]], dim=2)  # [B, C, K]
        mixed = F.silu((window * self.conv_weight).sum(dim=-1) + self.conv_bias)
        q, k, v = self._split(mixed)
        alpha, beta, z = self._gates(x)
        out, S = self._recur(state["recurrent"], q, k, v, alpha, beta)
        return self._output(out, z), {"recurrent": S, "conv": window[:, :, 1:]}

    def forward(self, x: Tensor, state: dict, *, transient: Tensor | None = None) -> tuple[Tensor, dict]:
        """시퀀스 ``x [B, T, d]`` → ``(y [B, T, d], 최종 상태)``.

        ``transient [T]``(bool)가 참인 토큰은 일시적 분기다: 자기 갱신을 읽되 recurrent 상태와 conv
        history에 남기지 않으므로, 뒤 토큰은 그 토큰이 없었던 것처럼 계산된다.
        """
        if x.dim() != 3:
            raise ValueError(f"x: [B, T, d]여야 한다 (받은 모양: {tuple(x.shape)})")
        B, T, _ = x.shape
        self._check_state(state, B)
        K = self.kernel
        raw = self.qkv_proj(x).transpose(1, 2)  # [B, C, T]
        seq = torch.cat([state["conv"], raw], dim=2)  # [B, C, K-1+T]

        if transient is not None:
            transient = torch.as_tensor(transient, dtype=torch.bool)
            if tuple(transient.shape) != (T,):
                raise ValueError(f"transient: [T]={T}이어야 한다 (받은 모양: {tuple(transient.shape)})")
        if transient is None or not bool(transient.any()):
            conv = F.conv1d(seq, self.conv_weight[:, None, :], self.conv_bias, groups=self.channels)
            history = seq[:, :, -(K - 1) :]
        else:
            # 토큰 t의 창 = 마지막 K-1개의 **남는(non-transient)** 토큰 + 자기 자신
            recent = list(range(K - 1))
            windows = []
            for t in range(T):
                windows.append(recent[-(K - 1) :] + [K - 1 + t])
                if not bool(transient[t]):
                    recent.append(K - 1 + t)
            index = torch.tensor(windows, dtype=torch.long)  # [T, K]
            gathered = seq[:, :, index]  # [B, C, T, K]
            conv = (gathered * self.conv_weight[None, :, None, :]).sum(dim=-1) + self.conv_bias[None, :, None]
            history = seq[:, :, torch.tensor(recent[-(K - 1) :], dtype=torch.long)]
        mixed = F.silu(conv).transpose(1, 2)  # [B, T, C]
        q, k, v = self._split(mixed)
        alpha, beta, z = self._gates(x)

        S = state["recurrent"]
        outputs = []
        for t in range(T):
            out, S_next = self._recur(S, q[:, t], k[:, t], v[:, t], alpha[:, t], beta[:, t])
            outputs.append(out)
            if transient is None or not bool(transient[t]):
                S = S_next
        return self._output(torch.stack(outputs, dim=1), z), {"recurrent": S, "conv": history}


# --------------------------------------------------------------------------
# windowed attention 층
# --------------------------------------------------------------------------


def _rope(positions: Tensor, head_dim: int, theta: float) -> tuple[Tensor, Tensor]:
    half = head_dim // 2
    inv_freq = theta ** (-torch.arange(0, half, dtype=torch.float32) / half)
    angles = positions.to(torch.float32)[..., None] * inv_freq  # [B, T, half]
    emb = torch.cat([angles, angles], dim=-1)
    return emb.cos()[:, :, None, :], emb.sin()[:, :, None, :]  # [B, T, 1, hd]


def _rotate_half(x: Tensor) -> Tensor:
    half = x.shape[-1] // 2
    return torch.cat([-x[..., half:], x[..., :half]], dim=-1)


class WindowedAttentionLayer(nn.Module):
    """RoPE + 명시적 mask + KV cache의 full attention.

    ``forward(x [B,T,d], positions [B,T], mask, cache)`` → ``(y [B,T,d], k [B,T,H,hd], v)``.
    ``mask``는 ``[T, n+T]`` 또는 ``[B, T, n+T]`` bool(query × [cache | 새 토큰]), ``cache``는
    ``{"k": [B,n,H,hd], "v": [B,n,H,hd]}``(k는 RoPE 적용 뒤의 값)이다. 돌려주는 k·v는 새 토큰의 것만이며
    cache에 이어 붙이는 일은 호출자가 한다(공유 cache를 in-place로 바꾸지 않기 위해서다).
    """

    def __init__(self, d_model: int, heads: int, head_dim: int, rope_theta: float = 10000.0) -> None:
        super().__init__()
        if head_dim % 2:
            raise ValueError(f"head_dim: 짝수여야 한다 (받은 값: {head_dim})")
        self.heads, self.head_dim, self.theta = heads, head_dim, rope_theta
        self.q_proj = nn.Linear(d_model, heads * head_dim, bias=False)
        self.k_proj = nn.Linear(d_model, heads * head_dim, bias=False)
        self.v_proj = nn.Linear(d_model, heads * head_dim, bias=False)
        self.o_proj = nn.Linear(heads * head_dim, d_model, bias=False)

    def forward(
        self, x: Tensor, positions: Tensor, *, mask: Tensor | None = None, cache: dict | None = None
    ) -> tuple[Tensor, Tensor, Tensor]:
        B, T, _ = x.shape
        H, hd = self.heads, self.head_dim
        q = self.q_proj(x).reshape(B, T, H, hd)
        k = self.k_proj(x).reshape(B, T, H, hd)
        v = self.v_proj(x).reshape(B, T, H, hd)
        cos, sin = _rope(positions.reshape(B, T), hd, self.theta)
        q = q * cos + _rotate_half(q) * sin
        k = k * cos + _rotate_half(k) * sin

        n = 0
        if cache is not None and cache["k"].shape[1] > 0:
            n = cache["k"].shape[1]
            k_all = torch.cat([cache["k"], k], dim=1)
            v_all = torch.cat([cache["v"], v], dim=1)
        else:
            k_all, v_all = k, v
        if mask is None:
            mask = causal_mask(T, n)
        if mask.dim() == 2:
            mask = mask[None]
        if tuple(mask.shape[-2:]) != (T, n + T):
            raise ValueError(f"mask: [T, cache+T]=({T}, {n + T})이어야 한다 (받은 모양: {tuple(mask.shape)})")

        scores = torch.einsum("bqhd,bkhd->bhqk", q, k_all) / math.sqrt(hd)
        scores = scores.masked_fill(~mask[:, None, :, :], float("-inf"))
        weights = torch.softmax(scores, dim=-1)
        y = torch.einsum("bhqk,bkhd->bqhd", weights, v_all).reshape(B, T, H * hd)
        return self.o_proj(y), k, v


# --------------------------------------------------------------------------
# 소형 hybrid backbone
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class DeltaNetConfig:
    heads: int = 2
    head_k: int = 16
    head_v: int = 16
    conv_kernel: int = 4


@dataclass(frozen=True)
class AttentionConfig:
    heads: int = 2
    head_dim: int = 32
    rope_theta: float = 10000.0
    window_ticks: int = 30


@dataclass(frozen=True)
class HybridConfig:
    name: str = "tiny-hybrid-v0"
    seed: int = 0
    vocab_size: int = 248077
    d_model: int = 64
    layers: tuple[str, ...] = ("deltanet", "deltanet", "attention")
    mlp_hidden: int = 128
    norm_eps: float = 1e-6
    deltanet: DeltaNetConfig = DeltaNetConfig()
    attention: AttentionConfig = AttentionConfig()
    readout_rank: int = 16

    @classmethod
    def from_dict(cls, raw: dict) -> HybridConfig:
        layers = tuple(str(layer) for layer in raw.get("layers", cls.layers))
        unknown = [layer for layer in layers if layer not in ("deltanet", "attention")]
        if unknown:
            raise ValueError(f"layers: 알 수 없는 층 종류 {unknown} (허용: deltanet, attention)")
        return cls(
            name=str(raw.get("name", cls.name)),
            seed=int(raw.get("seed", cls.seed)),
            vocab_size=int(raw.get("vocab_size", cls.vocab_size)),
            d_model=int(raw.get("d_model", cls.d_model)),
            layers=layers,
            mlp_hidden=int(raw.get("mlp_hidden", cls.mlp_hidden)),
            norm_eps=float(raw.get("norm_eps", cls.norm_eps)),
            deltanet=DeltaNetConfig(**{k: int(v) for k, v in (raw.get("deltanet") or {}).items()}),
            attention=AttentionConfig(
                **{
                    k: (float(v) if k == "rope_theta" else int(v))
                    for k, v in (raw.get("attention") or {}).items()
                }
            ),
            readout_rank=int((raw.get("readout") or {}).get("rank", cls.readout_rank)),
        )

    @classmethod
    def load(cls, path: str | Path = DEFAULT_CONFIG) -> HybridConfig:
        return cls.from_dict(yaml.safe_load(Path(path).read_text(encoding="utf-8")))


class _Block(nn.Module):
    def __init__(self, config: HybridConfig, kind: str) -> None:
        super().__init__()
        self.kind = kind
        d = config.d_model
        self.norm1 = RMSNorm(d, config.norm_eps)
        if kind == "deltanet":
            c = config.deltanet
            self.mixer: nn.Module = GatedDeltaNetLayer(d, c.heads, c.head_k, c.head_v, c.conv_kernel, config.norm_eps)
        else:
            a = config.attention
            self.mixer = WindowedAttentionLayer(d, a.heads, a.head_dim, a.rope_theta)
        self.norm2 = RMSNorm(d, config.norm_eps)
        self.gate_proj = nn.Linear(d, config.mlp_hidden, bias=False)
        self.up_proj = nn.Linear(d, config.mlp_hidden, bias=False)
        self.down_proj = nn.Linear(config.mlp_hidden, d, bias=False)

    def mlp(self, x: Tensor) -> Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class TinyHybrid(nn.Module):
    """소형 hybrid backbone — 계산 fixture (모듈 설명 참조).

    ``forward(tokens [B,T], positions [B,T], *, state, kv, mask, transient)`` →
    ``{"hidden": [B,T,d], "state": [DeltaNet 층별 상태], "kv": [attention 층별 {"k","v"}(새 토큰만)],
    "layer_hidden": [층별 출력], "embedded": [B,T,d]}``. ``state``가 없으면 0 초기 상태, ``kv``가
    없으면 빈 cache, ``mask``가 없으면 cache 전부 + causal이다. 최종 RMSNorm 뒤의 ``hidden``이
    readout이 읽는 "hidden state"다.
    """

    def __init__(self, config: HybridConfig, *, seed: int | None = None) -> None:
        super().__init__()
        self.config = config
        self.embed = nn.Embedding(config.vocab_size, config.d_model)
        self.blocks = nn.ModuleList(_Block(config, kind) for kind in config.layers)
        self.norm = RMSNorm(config.d_model, config.norm_eps)
        self.reset_parameters(config.seed if seed is None else seed)

    @classmethod
    def from_config(
        cls, path: str | Path = DEFAULT_CONFIG, *, seed: int | None = None, vocab_size: int | None = None
    ) -> TinyHybrid:
        config = HybridConfig.load(path)
        if vocab_size is not None:
            config = HybridConfig(**{**config.__dict__, "vocab_size": int(vocab_size)})
        return cls(config, seed=seed)

    def reset_parameters(self, seed: int) -> None:
        """모든 가중치를 seed에서 결정적으로 초기화한다 (전역 RNG와 무관)."""
        generator = torch.Generator().manual_seed(int(seed))
        with torch.no_grad():
            for name, parameter in self.named_parameters():
                if name.endswith("norm.weight") or ".norm1." in name or ".norm2." in name or name.startswith("norm."):
                    parameter.fill_(1.0)
                elif parameter.dim() >= 2:
                    fan_in = parameter.shape[-1]
                    parameter.normal_(0.0, 1.0 / math.sqrt(fan_in), generator=generator)
                else:
                    parameter.zero_()
            for block in self.blocks:
                if block.kind == "deltanet":
                    block.mixer.reset_parameters(generator)

    @property
    def delta_layers(self) -> list[GatedDeltaNetLayer]:
        return [block.mixer for block in self.blocks if block.kind == "deltanet"]

    @property
    def attention_layers(self) -> list[WindowedAttentionLayer]:
        return [block.mixer for block in self.blocks if block.kind == "attention"]

    def initial_state(self, batch: int = 1, *, requires_grad: bool = False) -> list[dict[str, Tensor]]:
        """DeltaNet 층별 0 상태(에피소드 시작)."""
        return [layer.initial_state(batch, requires_grad=requires_grad) for layer in self.delta_layers]

    def forward(
        self,
        tokens: Tensor,
        positions: Tensor,
        *,
        state: list[dict] | None = None,
        kv: list[dict] | None = None,
        mask: Tensor | None = None,
        transient: Tensor | None = None,
    ) -> dict[str, Any]:
        if tokens.dim() != 2:
            raise ValueError(f"tokens: [B, T]여야 한다 (받은 모양: {tuple(tokens.shape)})")
        B, T = tokens.shape
        if tuple(positions.shape) != (B, T):
            raise ValueError(f"positions: tokens와 같은 [B, T]여야 한다 (받은 모양: {tuple(positions.shape)})")
        if state is None:
            state = self.initial_state(B)
        if len(state) != len(self.delta_layers):
            raise ValueError(f"state: DeltaNet 층 수 {len(self.delta_layers)}개여야 한다 (받은 수: {len(state)})")
        if kv is not None and len(kv) != len(self.attention_layers):
            raise ValueError(f"kv: attention 층 수 {len(self.attention_layers)}개여야 한다 (받은 수: {len(kv)})")

        embedded = self.embed(tokens)
        x = embedded
        new_state: list[dict] = []
        new_kv: list[dict] = []
        layer_hidden: list[Tensor] = []
        for block in self.blocks:
            h = block.norm1(x)
            if block.kind == "deltanet":
                y, layer_state = block.mixer(h, state[len(new_state)], transient=transient)
                new_state.append(layer_state)
            else:
                cache = kv[len(new_kv)] if kv is not None else None
                y, k, v = block.mixer(h, positions, mask=mask, cache=cache)
                new_kv.append({"k": k, "v": v})
            x = x + y
            x = x + block.mlp(block.norm2(x))
            layer_hidden.append(x)
        return {
            "hidden": self.norm(x),
            "state": new_state,
            "kv": new_kv,
            "layer_hidden": layer_hidden,
            "embedded": embedded,
        }


@functools.lru_cache(maxsize=1)
def default_backbone() -> TinyHybrid:
    """설정 파일 그대로의 소형 fixture(한 번만 만든다). gradient 검사는 자기 인스턴스를 만든다."""
    return TinyHybrid.from_config()
