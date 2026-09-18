"""스트림 상태 검사 — 분기·상태 전달·윈도우·증분/전체 정합성 (docs/08 §3.1, docs/06 Task 4).

계획서의 최소 예시(`test_decision_branches_do_not_leak_into_next_tick`)는 그대로 두고 fixture 토큰만
여기서 정한다. 그 뒤는 (a) 증분(`advance` 틱 단위 + 결정 분기 fork/step)과 처음부터 계산(한 번의
forward + 기준 mask + transient 결정)의 hidden state 일치, 윈도우 절단 차이의 기록, (b) 결정
분기의 순서·단독 실행 무관성, (c) gradient가 초기 recurrent 상태·conv history·prefix hidden으로
돌아가는지다.
"""

import random

import pytest
import torch

from robo_jev.model.attention import build_reference_mask
from robo_jev.model.hybrid import TinyHybrid, default_backbone
from robo_jev.model.stream import StreamState, forward_layout, replay_layout

# fixture 토큰 (계획서의 검사가 쓰는 이름 그대로). 값은 어휘 안의 아무 id다.
prefix_tokens = [11, 12, 13, 14, 15, 16]
tick_tokens = [21, 22, 23, 24, 25]
decision_token = 65  # 결정 표지 "A" (byte-level BPE에서 한 토큰)
next_tick_tokens = [31, 32, 33, 34]

#: backbone 전체 수준의 FP32 허용 오차 (tests/test_hybrid.py와 같은 근거).
FP32 = {"rtol": 2e-5, "atol": 5e-5}
EXACT64 = {"rtol": 1e-10, "atol": 1e-12}


# --------------------------------------------------------------------------
# 계획서의 최소 예시 (docs/06 Task 4, 그대로)
# --------------------------------------------------------------------------


def test_decision_branches_do_not_leak_into_next_tick():
    base = StreamState.from_tokens(prefix_tokens, tick_tokens)
    snapshot = base.clone()
    branches = base.fork(9)
    for branch in branches:
        branch.step(decision_token)          # 일시적 분기, 결과는 버림
    torch.testing.assert_close(base.recurrent, snapshot.recurrent)
    torch.testing.assert_close(base.kv, snapshot.kv)
    next_state = base.advance(next_tick_tokens)
    full = StreamState.from_tokens(prefix_tokens, tick_tokens + next_tick_tokens)
    torch.testing.assert_close(next_state.recurrent, full.recurrent)


# --------------------------------------------------------------------------
# 합성 스트림 layout (직렬화 결과와 같은 모양·position 규칙)
# --------------------------------------------------------------------------


def synthetic_stream(
    rng: random.Random, *, prefix: int, ticks: list[tuple[int, int]], window_ticks: int
) -> dict:
    """prefix 길이와 틱별 (몸통 길이, 결정 수)로 layout을 만든다. 결정은 몸통 끝 position의 1토큰 분기다."""
    layout = {
        "layout": "stream_l1a", "tokens": [], "kind": [], "state": [], "question": [], "candidate": [],
        "position": [], "tick": [], "window_ticks": window_ticks, "ticks": [],
    }

    def push(token, kind, question, tick, position):
        layout["tokens"].append(token)
        layout["kind"].append(kind)
        layout["state"].append(0)
        layout["question"].append(question)
        layout["candidate"].append(-1)
        layout["position"].append(position)
        layout["tick"].append(tick)

    cursor = 0
    for _ in range(prefix):
        push(rng.randint(100, 900), "prefix", -1, -1, cursor)
        cursor += 1
    layout["prefix_end"] = len(layout["tokens"])
    for index, (body, decisions) in enumerate(ticks):
        start = len(layout["tokens"])
        for _ in range(body):
            push(rng.randint(100, 900), rng.choice(["state", "exec", "candidate"]), -1, index, cursor)
            cursor += 1
        body_end = len(layout["tokens"])
        positions = {}
        for branch in range(decisions):
            positions[f"q{branch}"] = len(layout["tokens"])
            push(65 + branch, "decision", branch, index, cursor)  # 분기는 같은 position
        layout["ticks"].append(
            {"index": index, "start": start, "body_end": body_end, "end": len(layout["tokens"]),
             "decision_positions": positions}
        )
    return layout


@pytest.fixture
def small_stream() -> dict:
    return synthetic_stream(random.Random(0), prefix=7, ticks=[(5, 3), (6, 2), (4, 3)], window_ticks=30)


# --------------------------------------------------------------------------
# 상태 객체의 성질
# --------------------------------------------------------------------------


def test_from_tokens_accepts_one_tick_or_many_and_tracks_positions():
    one = StreamState.from_tokens(prefix_tokens, tick_tokens)
    many = StreamState.from_tokens(prefix_tokens, [tick_tokens, next_tick_tokens])
    assert one.tick == 0 and many.tick == 1
    assert one.position == len(prefix_tokens) + len(tick_tokens)
    assert many.position == one.position + len(next_tick_tokens)
    assert one.cached_tokens == one.position and many.cached_tokens == many.position
    assert one.prefix_hidden.shape == (len(prefix_tokens), 64)
    assert one.hidden.shape == (len(tick_tokens), 64)
    assert many.hidden.shape == (len(next_tick_tokens), 64)
    empty = StreamState.from_tokens(prefix_tokens, [])
    assert empty.tick == -1 and empty.hidden is None and empty.cached_tokens == len(prefix_tokens)


def test_advance_returns_a_new_state_and_leaves_the_base_untouched():
    base = StreamState.from_tokens(prefix_tokens, tick_tokens)
    snapshot = base.clone()
    nxt = base.advance(next_tick_tokens)
    assert nxt is not base and nxt.tick == base.tick + 1
    torch.testing.assert_close(base.recurrent, snapshot.recurrent)
    torch.testing.assert_close(base.conv, snapshot.conv)
    torch.testing.assert_close(base.kv, snapshot.kv)
    assert base.position == snapshot.position and base.cached_tokens == snapshot.cached_tokens
    assert nxt.cached_tokens == base.cached_tokens + len(next_tick_tokens)
    for layer_before, layer_after in zip(base.kv, nxt.kv):
        torch.testing.assert_close(layer_after["k"][:, : base.cached_tokens], layer_before["k"])


def test_fork_shares_kv_read_only_and_clones_recurrent_state():
    base = StreamState.from_tokens(prefix_tokens, tick_tokens)
    branches = base.fork(3)
    assert len(branches) == 3 and all(branch.is_branch for branch in branches)
    for branch in branches:
        for shared, own in zip(base.kv, branch.kv):
            assert shared["k"] is own["k"] and shared["v"] is own["v"]  # 읽기 전용 공유
        for layer_base, layer_branch in zip(base.delta, branch.delta):
            assert layer_branch["recurrent"].data_ptr() != layer_base["recurrent"].data_ptr()
            assert layer_branch["conv"].data_ptr() != layer_base["conv"].data_ptr()
        assert branch.position == base.position and branch.tick == base.tick
    hidden = branches[0].step(decision_token)
    assert hidden.shape == (64,)
    assert branches[0].position == base.position + 1 and branches[0].cached_tokens == base.cached_tokens + 1
    assert branches[1].cached_tokens == base.cached_tokens  # 형제는 그대로
    for shared, own in zip(base.kv, branches[0].kv):
        assert shared["k"].shape[1] == base.cached_tokens and own["k"] is not shared["k"]
    with pytest.raises(ValueError, match="분기"):
        branches[0].advance(next_tick_tokens)


def test_branch_outputs_do_not_depend_on_order_or_company():
    base = StreamState.from_tokens(prefix_tokens, tick_tokens)
    tokens = [65, 66, 67, 68]
    forward = [branch.step(token) for branch, token in zip(base.fork(4), tokens)]
    backward = [branch.step(token) for branch, token in zip(base.fork(4), reversed(tokens))]
    alone = [base.fork(1)[0].step(token) for token in tokens]
    for index in range(4):
        torch.testing.assert_close(forward[index], backward[3 - index])
        torch.testing.assert_close(forward[index], alone[index])


# --------------------------------------------------------------------------
# 증분 == 처음부터 (윈도우 안), 윈도우 절단 차이는 기록
# --------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_incremental_replay_matches_from_scratch_inside_the_window(small_stream, dtype):
    backbone = TinyHybrid.from_config().to(dtype)
    tolerance = FP32 if dtype == torch.float32 else EXACT64
    scratch = forward_layout(small_stream, backbone=backbone)
    replay = replay_layout(small_stream, backbone=backbone)
    assert scratch.shape == replay["hidden"].shape == (len(small_stream["tokens"]), 64)
    torch.testing.assert_close(replay["hidden"], scratch, **tolerance)
    # 마지막 틱의 공통 상태 = 몸통만 통째로 돌린 DeltaNet 상태 (결정 분기는 남지 않는다)
    body_only = [token for token, kind in zip(small_stream["tokens"], small_stream["kind"]) if kind != "decision"]
    tokens = torch.tensor([body_only])
    plain = backbone(tokens, torch.arange(len(body_only))[None])
    for layer_replay, layer_plain in zip(replay["final"].delta, plain["state"]):
        torch.testing.assert_close(layer_replay["recurrent"], layer_plain["recurrent"], **tolerance)
        torch.testing.assert_close(layer_replay["conv"], layer_plain["conv"], **tolerance)


def test_window_eviction_matches_the_reference_mask_and_records_the_truncation(capsys):
    rng = random.Random(1)
    window = 3
    layout = synthetic_stream(rng, prefix=6, ticks=[(4, 2)] * 8, window_ticks=window)
    backbone = default_backbone()
    scratch = forward_layout(layout, backbone=backbone)  # 기준 mask가 윈도우 규칙을 그대로 적용
    replay = replay_layout(layout, backbone=backbone)
    torch.testing.assert_close(replay["hidden"], scratch, **FP32)
    # cache에는 prefix + 최근 window 틱만 남는다
    final = replay["final"]
    assert final.cached_tokens == 6 + window * 4
    assert final.cache_ticks.tolist() == [-1] * 6 + sum(([t] * 4 for t in range(8 - window, 8)), [])
    # 윈도우 절단의 차이: 절단 없는 계산과의 거리 (정의된 근사, 기록만 한다 — docs/08 §3.1)
    untruncated = forward_layout(layout, backbone=backbone, window_ticks=None)
    inside = [i for i, tick in enumerate(layout["tick"]) if tick < window]
    beyond = [i for i, tick in enumerate(layout["tick"]) if tick >= window]
    torch.testing.assert_close(scratch[inside], untruncated[inside], **FP32)
    gap = (scratch[beyond] - untruncated[beyond]).norm(dim=-1)
    print(
        f"window truncation (window={window}, ticks=8, body 4 + 2 decisions): "
        f"max L2 {gap.max().item():.4g}, mean {gap.mean().item():.4g} over {len(beyond)} tokens "
        f"(hidden norm mean {untruncated[beyond].norm(dim=-1).mean().item():.4g})"
    )
    assert torch.isfinite(gap).all() and gap.max() > 0


def test_replay_rejects_layouts_whose_positions_break_the_stream_rule(small_stream):
    broken = dict(small_stream)
    broken["position"] = list(range(len(small_stream["tokens"])))  # 결정 분기가 position을 소비
    with pytest.raises(ValueError, match="position"):
        replay_layout(broken)


# --------------------------------------------------------------------------
# gradient: 초기 recurrent 상태·conv history·prefix hidden으로 돌아간다
# --------------------------------------------------------------------------


def test_branch_loss_gradient_reaches_initial_state_conv_history_and_prefix():
    """분기의 손실이 KV cache·recurrent 상태를 거쳐 초기 상태와 prefix 토큰까지 돌아간다.

    `prefix_hidden`(최종 norm 뒤 출력)은 분기가 읽는 것이 아니다 — 분기는 층 안의 KV와 recurrent
    상태로 prefix를 본다. 그래서 여기서는 prefix 토큰의 embedding 행에 gradient가 닿는지로 확인하고,
    정적 후보의 `prefix_hidden`을 readout이 읽는 경로는 Judge 검사가 본다.
    """
    backbone = TinyHybrid.from_config(seed=3)
    initial = backbone.initial_state(1, requires_grad=True)
    base = StreamState.from_tokens(prefix_tokens, tick_tokens, backbone=backbone, initial=initial)
    branches = base.fork(2)
    loss = sum(branch.step(token).pow(2).sum() for branch, token in zip(branches, [65, 66]))
    loss.backward()
    for layer in initial:
        for key in ("recurrent", "conv"):
            assert layer[key].grad is not None and torch.isfinite(layer[key].grad).all()
            assert layer[key].grad.abs().sum() > 0, key
    rows = backbone.embed.weight.grad
    assert rows is not None
    assert (rows[prefix_tokens].abs().sum(dim=-1) > 0).all()  # prefix 토큰 전부에 gradient
    assert (rows[tick_tokens].abs().sum(dim=-1) > 0).all()
    assert rows[[900, 901]].abs().sum() == 0  # 쓰이지 않은 토큰은 0


def test_mask_reference_and_replay_agree_on_a_layout_with_an_instruction_piece():
    """도중 지시 조각은 그 틱의 토큰이다 — 윈도우 밖으로 나가고, 증분 계산은 특별 취급하지 않는다."""
    rng = random.Random(2)
    layout = synthetic_stream(rng, prefix=5, ticks=[(3, 2), (4, 2), (3, 2), (3, 2)], window_ticks=2)
    # 두 번째 틱의 첫 두 토큰을 지시 조각으로 표시 (kind state, 그 틱) — 직렬화와 같은 모양
    second = layout["ticks"][1]
    layout["kind"][second["start"]] = "state"
    layout["kind"][second["start"] + 1] = "state"
    mask = build_reference_mask(layout)
    last_decision = layout["ticks"][3]["body_end"]
    assert not mask[last_decision, second["start"]]  # 2틱 뒤의 결정은 지시 조각을 못 본다
    assert mask[last_decision, 0]  # 정적 prefix는 본다
    torch.testing.assert_close(replay_layout(layout)["hidden"], forward_layout(layout), **FP32)
