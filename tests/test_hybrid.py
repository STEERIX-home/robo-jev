"""Hybrid 상태 분기와 소형 hybrid fixture 검사 (docs/06 Task 4, docs/03 §3).

`fork_delta_state`는 계획서의 최소 예시를 그대로 통과해야 한다. 그 뒤는 기준 DeltaNet 층(명시적
recurrent 상태 + 짧은 causal conv history)과 windowed attention 층, 그리고 둘을 묶은 소형
backbone의 정합성이다 — step-by-step과 통째 계산이 같고, 일시적(transient) 토큰이 상태를 남기지
않으며, gradient가 초기 상태와 conv history로 돌아간다.
"""

import copy

import pytest
import torch
import yaml
from helpers import SMALL_VOCAB

from robo_jev.model.hybrid import (
    DEFAULT_CONFIG,
    GatedDeltaNetLayer,
    HybridConfig,
    TinyHybrid,
    WindowedAttentionLayer,
    causal_mask,
    default_backbone,
    fork_delta_state,
)
from robo_jev.model.tokenizer import FETCH_SCRIPT, available_tokenizer, load_tokenizer

# --------------------------------------------------------------------------
# 계획서의 최소 예시 (docs/06 Task 4, 그대로)
# --------------------------------------------------------------------------


def test_fork_preserves_prefix_gradient():
    state = {
        "recurrent": torch.ones(2, requires_grad=True),
        "conv": torch.ones(3, requires_grad=True),
    }
    children = fork_delta_state(state, branches=2)
    loss = sum(child["recurrent"].sum() + child["conv"].sum() for child in children)
    loss.backward()
    for tensor in state.values():
        torch.testing.assert_close(tensor.grad, torch.full_like(tensor, 2))


def test_branch_updates_cannot_mutate_siblings():
    state = {"recurrent": torch.ones(2), "conv": torch.ones(3)}
    children = fork_delta_state(state, branches=2)
    children[0]["conv"].zero_()
    children[0]["recurrent"].zero_()
    for key in state:
        torch.testing.assert_close(children[1][key], state[key])
        assert torch.all(state[key] == 1)


# --------------------------------------------------------------------------
# fork_delta_state — 그 밖의 성질
# --------------------------------------------------------------------------


def test_fork_returns_independent_buffers_that_are_not_views():
    state = {"recurrent": torch.arange(4.0), "conv": torch.ones(2, 3)}
    children = fork_delta_state(state, branches=3)
    assert len(children) == 3
    for child in children:
        for key, tensor in state.items():
            assert child[key].data_ptr() != tensor.data_ptr()
            assert child[key].shape == tensor.shape
            assert torch.equal(child[key], tensor)
    for key in state:
        assert len({child[key].data_ptr() for child in children}) == 3


def test_fork_handles_nested_states_and_rejects_bad_branch_counts():
    nested = {"layer": {"recurrent": torch.ones(2, requires_grad=True), "conv": None}, "tag": 7}
    children = fork_delta_state(nested, branches=2)
    assert children[0]["layer"]["conv"] is None and children[0]["tag"] == 7
    (children[0]["layer"]["recurrent"].sum() + children[1]["layer"]["recurrent"].sum()).backward()
    torch.testing.assert_close(nested["layer"]["recurrent"].grad, torch.full((2,), 2.0))
    with pytest.raises(ValueError, match="branches"):
        fork_delta_state(nested, branches=0)


# --------------------------------------------------------------------------
# 기준 gated DeltaNet 층
# --------------------------------------------------------------------------


@pytest.fixture
def delta_layer() -> GatedDeltaNetLayer:
    torch.manual_seed(0)
    return GatedDeltaNetLayer(d_model=16, heads=2, head_k=8, head_v=8, conv_kernel=4)


def random_delta_state(layer: GatedDeltaNetLayer, batch: int, seed: int, *, requires_grad: bool = False) -> dict:
    generator = torch.Generator().manual_seed(seed)
    state = layer.initial_state(batch)
    return {
        key: (torch.randn(tensor.shape, generator=generator) * 0.5).requires_grad_(requires_grad)
        for key, tensor in state.items()
    }


def test_delta_step_by_step_matches_whole_sequence(delta_layer):
    generator = torch.Generator().manual_seed(1)
    x = torch.randn(2, 9, 16, generator=generator)
    state = random_delta_state(delta_layer, 2, seed=2)
    whole, final = delta_layer(x, state)

    running = state
    outputs = []
    for t in range(x.shape[1]):
        y, running = delta_layer.step(x[:, t], running)
        outputs.append(y)
    torch.testing.assert_close(torch.stack(outputs, dim=1), whole, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(running["recurrent"], final["recurrent"], rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(running["conv"], final["conv"], rtol=1e-5, atol=1e-6)
    assert final["recurrent"].shape == (2, 2, 8, 8) and final["conv"].shape == (2, 3 * 8 * 2, 3)


def test_delta_conv_stage_matches_grouped_conv1d_within_fp32_and_the_step_window_exactly(delta_layer):
    """conv 단계 = 창을 unfold로 펼쳐 곱하고 더한 것 (Task 5 수정 라운드 1, Minor 2).

    이전 경로(grouped ``F.conv1d`` — 여기 검사에만 남긴다)와는 합산 순서·FMA가 달라 FP32 잡음 안에서만
    같다(어느 정식화도 conv1d와 비트 단위로 같지 않음을 실측: unfold-sum·순차 합·einsum·matmul 모두
    ≈1e-6). 대신 ``step``의 한 토큰 창 계산·transient 경로의 gather 계산과는 **같은 op**라 비트 단위로
    같다 — 그 성질을 여기서 고정한다.
    """
    import torch.nn.functional as F

    layer = delta_layer
    C, K = layer.channels, layer.kernel
    generator = torch.Generator().manual_seed(7)
    for batch, length in ((1, 1), (3, 5), (2, 200)):
        seq = torch.randn(batch, C, K - 1 + length, generator=generator)
        old = F.conv1d(seq, layer.conv_weight[:, None, :], layer.conv_bias, groups=C)  # 이전 경로 (검사에만)
        new = layer._conv(seq)
        assert new.shape == old.shape == (batch, C, length)
        torch.testing.assert_close(new, old, rtol=1e-6, atol=1e-5)
        assert (new - old).abs().max() < 5e-6
        # step의 창 계산과 비트 단위로 같다
        for t in range(length):
            window = seq[:, :, t : t + K]
            by_step = (window * layer.conv_weight).sum(dim=-1) + layer.conv_bias
            assert torch.equal(new[:, :, t], by_step)
        # transient 경로(gather)와도 같다
        index = torch.tensor([[t + k for k in range(K)] for t in range(length)], dtype=torch.long)
        gathered = seq[:, :, index]
        by_gather = (gathered * layer.conv_weight[None, :, None, :]).sum(dim=-1) + layer.conv_bias[None, :, None]
        assert torch.equal(new, by_gather)
    # transient 표지가 전부 False인 forward는 표지 없는 forward와 비트 단위로 같다
    x = torch.randn(2, 9, 16, generator=generator)
    state = random_delta_state(layer, 2, seed=3)
    plain, plain_state = layer(x, state)
    flagged, flagged_state = layer(x, state, transient=torch.zeros(9, dtype=torch.bool))
    assert torch.equal(plain, flagged) and torch.equal(plain_state["conv"], flagged_state["conv"])


def test_delta_forward_never_writes_the_input_state_in_place(delta_layer):
    x = torch.randn(1, 5, 16)
    state = random_delta_state(delta_layer, 1, seed=3)
    before = {key: tensor.clone() for key, tensor in state.items()}
    _, after = delta_layer(x, state)
    for key in state:
        assert torch.equal(state[key], before[key])
        assert after[key].data_ptr() != state[key].data_ptr()


def test_delta_transient_tokens_read_but_do_not_commit(delta_layer):
    """transient 토큰은 자기 갱신을 읽되(분기 1스텝) 상태·conv history에 남기지 않는다."""
    generator = torch.Generator().manual_seed(4)
    x = torch.randn(1, 7, 16, generator=generator)
    state = random_delta_state(delta_layer, 1, seed=5)
    transient = torch.tensor([False, False, True, True, False, True, False])
    y, final = delta_layer(x, state, transient=transient)

    # 손으로: 몸통 토큰은 step, transient 토큰은 fork → step → 폐기
    running = state
    expected = []
    for t in range(x.shape[1]):
        if transient[t]:
            (branch,) = fork_delta_state(running, branches=1)
            out, _ = delta_layer.step(x[:, t], branch)
        else:
            out, running = delta_layer.step(x[:, t], running)
        expected.append(out)
    torch.testing.assert_close(y, torch.stack(expected, dim=1), rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(final["recurrent"], running["recurrent"], rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(final["conv"], running["conv"], rtol=1e-5, atol=1e-6)

    # transient 토큰을 아예 빼고 돌린 것과 몸통 출력·최종 상태가 같다
    body = ~transient
    y_body, final_body = delta_layer(x[:, body], state)
    torch.testing.assert_close(y[:, body], y_body, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(final["recurrent"], final_body["recurrent"], rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(final["conv"], final_body["conv"], rtol=1e-5, atol=1e-6)


def test_delta_gradient_reaches_initial_recurrent_state_and_conv_history(delta_layer):
    x = torch.randn(1, 6, 16)
    state = random_delta_state(delta_layer, 1, seed=6, requires_grad=True)
    y, final = delta_layer(x, state)
    y.pow(2).sum().backward()
    for key in ("recurrent", "conv"):
        assert state[key].grad is not None
        assert torch.isfinite(state[key].grad).all()
        assert state[key].grad.norm() > 0.1, key  # 실측 recurrent 1.86, conv 1.13 — 경로만이 아니라 크기


def test_delta_short_sequence_keeps_old_conv_history(delta_layer):
    """kernel-1보다 짧은 시퀀스 뒤의 history에는 옛 history가 남는다."""
    state = random_delta_state(delta_layer, 1, seed=7)
    x = torch.randn(1, 1, 16)
    _, final = delta_layer(x, state)
    torch.testing.assert_close(final["conv"][:, :, :2], state["conv"][:, :, 1:])


# --------------------------------------------------------------------------
# windowed attention 층 (KV cache + 명시적 mask + RoPE)
# --------------------------------------------------------------------------


@pytest.fixture
def attention_layer() -> WindowedAttentionLayer:
    torch.manual_seed(0)
    return WindowedAttentionLayer(d_model=16, heads=2, head_dim=8, rope_theta=10000.0)


def test_attention_incremental_cache_matches_full_causal(attention_layer):
    generator = torch.Generator().manual_seed(8)
    x = torch.randn(1, 8, 16, generator=generator)
    positions = torch.arange(8)[None]
    full, k_all, v_all = attention_layer(x, positions, mask=causal_mask(8))

    cache = None
    outputs = []
    for t in range(8):
        n = 0 if cache is None else cache["k"].shape[1]
        y, k, v = attention_layer(x[:, t : t + 1], positions[:, t : t + 1], mask=torch.ones(1, n + 1, dtype=torch.bool), cache=cache)
        cache = {"k": k, "v": v} if cache is None else {"k": torch.cat([cache["k"], k], 1), "v": torch.cat([cache["v"], v], 1)}
        outputs.append(y)
    torch.testing.assert_close(torch.cat(outputs, dim=1), full, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(cache["k"], k_all)
    torch.testing.assert_close(cache["v"], v_all)


def test_attention_mask_hides_keys_and_positions_are_relative(attention_layer):
    generator = torch.Generator().manual_seed(9)
    x = torch.randn(1, 6, 16, generator=generator)
    positions = torch.arange(6)[None]
    mask = causal_mask(6)
    mask[5, :3] = False  # 마지막 query가 앞 3개 key를 못 본다
    hidden, _, _ = attention_layer(x, positions, mask=mask)
    # 가려진 key의 값을 바꿔도 마지막 query의 출력은 그대로다
    x2 = x.clone()
    x2[:, :3] += 1.0
    hidden2, _, _ = attention_layer(x2, positions, mask=mask)
    torch.testing.assert_close(hidden[:, 5], hidden2[:, 5])
    assert not torch.allclose(hidden[:, 4], hidden2[:, 4])
    # RoPE: 모든 position을 같은 값만큼 옮기면 출력이 같다 (상대 위치만 본다)
    shifted, _, _ = attention_layer(x, positions + 1000, mask=mask)
    torch.testing.assert_close(shifted, hidden, rtol=1e-4, atol=1e-5)


# --------------------------------------------------------------------------
# 소형 hybrid backbone (configs/model/tiny_hybrid.yaml)
# --------------------------------------------------------------------------


def test_config_file_is_a_documented_computation_fixture():
    raw = yaml.safe_load(DEFAULT_CONFIG.read_text(encoding="utf-8"))
    config = HybridConfig.from_dict(raw)
    assert config.d_model == 64 and config.layers == ("deltanet", "deltanet", "attention")
    assert config.deltanet.conv_kernel == 4 and config.attention.heads == 2
    assert config.attention.window_ticks == 30 and config.readout_rank > 0
    assert "fixture" in DEFAULT_CONFIG.read_text(encoding="utf-8")


def test_vocab_matches_the_real_tokenizer():
    found = available_tokenizer()
    if found is None:
        pytest.skip(f"실제 tokenizer가 없다 — `uv run python {FETCH_SCRIPT}`로 받는다")
    tokenizer = load_tokenizer(found[1])
    assert default_backbone().config.vocab_size == tokenizer.get_vocab_size()


def test_seeded_init_is_deterministic_and_the_default_is_cached():
    a = TinyHybrid.from_config(vocab_size=SMALL_VOCAB)
    b = TinyHybrid.from_config(vocab_size=SMALL_VOCAB)
    for (name, pa), (_, pb) in zip(a.named_parameters(), b.named_parameters()):
        assert torch.equal(pa, pb), name
    c = TinyHybrid.from_config(seed=1, vocab_size=SMALL_VOCAB)
    assert not torch.equal(a.embed.weight, c.embed.weight)
    assert default_backbone() is default_backbone()
    # 캐시된 기본 fixture는 gradient를 받지 않는다 — 그것을 통해 backward하지 않는다(검사는 자기 인스턴스를 만든다).
    assert not any(parameter.requires_grad for parameter in default_backbone().parameters())
    assert all(parameter.requires_grad for parameter in a.parameters())


def test_truncated_vocab_fixture_is_the_config_fixture_on_the_ids_it_covers():
    """embedding은 자기 generator로 초기화하므로 작은 어휘의 fixture는 설정 fixture를 id < V로 제한한 것이다."""
    full = default_backbone()
    small = TinyHybrid.from_config(vocab_size=512)
    assert torch.equal(small.embed.weight, full.embed.weight[:512])
    for (name, a), (_, b) in zip(small.named_parameters(), full.named_parameters()):
        if name != "embed.weight":
            assert torch.equal(a, b), name
    generator = torch.Generator().manual_seed(12)
    tokens = torch.randint(0, 512, (1, 9), generator=generator)
    positions = torch.arange(9)[None]
    torch.testing.assert_close(small(tokens, positions)["hidden"], full(tokens, positions)["hidden"], rtol=0, atol=0)


def test_backbone_forward_shapes_and_state_layout():
    backbone = TinyHybrid.from_config(vocab_size=SMALL_VOCAB)
    tokens = torch.tensor([[5, 7, 11, 13, 17]])
    out = backbone(tokens, torch.arange(5)[None])
    assert out["hidden"].shape == (1, 5, 64)
    assert len(out["state"]) == 2 and len(out["kv"]) == 1
    assert out["state"][0]["recurrent"].shape == (1, 2, 16, 16)
    assert out["state"][0]["conv"].shape == (1, 3 * 2 * 16, 3)
    assert out["kv"][0]["k"].shape == (1, 5, 2, 32)
    assert len(out["layer_hidden"]) == 3
    initial = backbone.initial_state(batch=2, requires_grad=True)
    assert all(layer["recurrent"].requires_grad and layer["recurrent"].shape[0] == 2 for layer in initial)


#: backbone 전체(3층) 수준의 FP32 허용 오차. step-by-step과 통째 계산은 float64에서 1e-14 안에서 같고
#: (아래 검사), FP32 차이는 합산 순서(conv1d ↔ 내적, softmax 모양)의 잡음이다 — 실측 최대 1.1e-5.
FP32 = {"rtol": 2e-5, "atol": 5e-5}


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_backbone_step_by_step_equals_whole_sequence_with_cache(dtype):
    backbone = TinyHybrid.from_config(vocab_size=SMALL_VOCAB).to(dtype)
    tolerance = FP32 if dtype == torch.float32 else {"rtol": 1e-10, "atol": 1e-12}
    generator = torch.Generator().manual_seed(10)
    tokens = torch.randint(0, 500, (1, 12), generator=generator)
    positions = torch.arange(12)[None]
    whole = backbone(tokens, positions)

    state, kv = None, None
    hidden = []
    for t in range(12):
        out = backbone(tokens[:, t : t + 1], positions[:, t : t + 1], state=state, kv=kv)
        state = out["state"]
        kv = out["kv"] if kv is None else [
            {"k": torch.cat([old["k"], new["k"]], 1), "v": torch.cat([old["v"], new["v"]], 1)}
            for old, new in zip(kv, out["kv"])
        ]
        hidden.append(out["hidden"])
    torch.testing.assert_close(torch.cat(hidden, dim=1), whole["hidden"], **tolerance)
    for layer_step, layer_whole in zip(state, whole["state"]):
        torch.testing.assert_close(layer_step["recurrent"], layer_whole["recurrent"], **tolerance)
        torch.testing.assert_close(layer_step["conv"], layer_whole["conv"], **tolerance)


def test_backbone_batch_rows_are_independent_under_right_padding():
    """오른쪽 padding은 causal 계산에 영향을 주지 않는다 — 질문별 경로를 한 배치로 돌릴 근거."""
    backbone = TinyHybrid.from_config(vocab_size=SMALL_VOCAB)
    generator = torch.Generator().manual_seed(11)
    short = torch.randint(0, 500, (1, 6), generator=generator)
    long = torch.randint(0, 500, (1, 9), generator=generator)
    padded = torch.cat([short, torch.zeros(1, 3, dtype=torch.long)], dim=1)
    batch = torch.cat([padded, long], dim=0)
    positions = torch.arange(9)[None].expand(2, 9)
    out = backbone(batch, positions, mask=causal_mask(9))
    alone = backbone(short, torch.arange(6)[None])
    torch.testing.assert_close(out["hidden"][0, :6], alone["hidden"][0], **FP32)
    alone_long = backbone(long, torch.arange(9)[None])
    torch.testing.assert_close(out["hidden"][1], alone_long["hidden"][0], **FP32)
