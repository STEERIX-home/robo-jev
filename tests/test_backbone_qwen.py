"""실제 backbone adapter(`robo_jev.model.backbone_qwen`)의 검사 (Task 2b G0b S1).

CPU 검사는 **소형 난수 Qwen3.5**(transformers의 구조, torch 참조 kernel)로 bookkeeping을 본다 — 정적 윈도우 버퍼의 퇴출·
당김, 배치 결정 분기 == 순차 분기, 증분 == 처음부터(기준 mask), dynamic KV 모드의 gradient, checkpoint 왕복, Judge·
학습 loop 연결, 배포 계약 digest. GPU 검사는 **실제 Qwen3.5-2B**(가중치가 있을 때만; 없으면 skip)로 BF16 허용 오차를
실측해 고정한 상수와 대조한다 — 증분 == 처음부터, 윈도우 절단 뒤 cache 길이 불변, 분기 격리, native == stream(단일 요청).

BF16 허용 오차(2026-09-19 GB10 실측, D0 스트림 6틱·1,525토큰, `.superpowers/sdd/task-g0b-report.md` S1.3):
결정 위치·후보 경계 hidden의 토큰별 상대 L2 오차 최대 0.020(중앙값 0.015), 전 토큰 최대 0.47(outlier 차원의 bf16 반올림,
hidden 절대 최대 104). 공식 구현 자체의 kernel 사이 차이(mask 있음/없음, DynamicCache 틱별/한 번)가 같은 자릿수
(상대 L2 최대 0.27~0.40, 중앙값 0.014)이고 fp32에서는 증분 경로가 0.018(공식 캐시 경로 0.027) 안이라 bf16 잡음이다.
"""

import copy
import functools
import hashlib
import random
import sys

import pytest
import torch
import yaml
from helpers import D0, D0_STREAMS, REPO, SMALL_VOCAB, read_jsonl

from robo_jev.checkpoint import stream_state_from_dict, stream_state_to_dict
from robo_jev.model import backbone_qwen as adapter
from robo_jev.model.attention import build_reference_mask
from robo_jev.model.backbone import describe_backbone
from robo_jev.model.backbone_qwen import QwenBackbone, QwenStreamState, merge_attention_parts, torch_reference_kernels
from robo_jev.model.contract_digest import CONTRACT_PARTS, contract_differences, contract_digest, harness_version
from robo_jev.model.judge import Judge
from robo_jev.model.serialize import WINDOW_TICKS, serialize_request
from robo_jev.model.stream import replay_layout
from robo_jev.model.tokenizer import WhitespaceTokenizer, available_tokenizer, load_tokenizer
from test_stream import synthetic_stream

pytest.importorskip("transformers")

#: 실측으로 고정한 BF16 허용 오차 (모듈 설명): readout이 읽는 위치의 토큰별 상대 L2, 전 토큰 중앙값, 전 토큰 최대.
BF16_REL_READOUT = 0.04  # 실측 0.020의 2배
BF16_REL_MEDIAN = 0.03  # 실측 0.015
BF16_REL_MAX = 0.6  # 실측 0.47 (공식 구현의 kernel 사이 차이 0.27~0.40)
FP32_REL_MAX = 0.03  # 실측 0.018 (공식 DynamicCache 틱별 경로 0.027)
CPU_TOL = {"rtol": 1e-4, "atol": 1e-4}

REAL_2B = "Qwen/Qwen3.5-2B"


def real_weights_present() -> bool:
    try:
        describe_backbone(REAL_2B)
    except (FileNotFoundError, ValueError):
        return False
    return torch.cuda.is_available()


needs_real_2b = pytest.mark.skipif(not real_weights_present(), reason="Qwen3.5-2B 가중치(artifacts/models)와 CUDA가 있어야 한다")


# --------------------------------------------------------------------------
# 소형 난수 Qwen (CPU)
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def tiny() -> QwenBackbone:
    return QwenBackbone.tiny(seed=1, vocab_size=1024)


@pytest.fixture(autouse=True)
def _torch_kernels():
    """CPU 검사는 공식 forward도 torch 참조 kernel로 돈다 (CUDA가 있는 venv에서 fla·causal_conv1d가 CPU tensor를 받지 않게)."""
    with torch_reference_kernels():
        yield


def incremental(backbone: QwenBackbone, layout: dict, *, window_ticks: int | None = None) -> tuple[torch.Tensor, QwenStreamState]:
    window = layout.get("window_ticks", WINDOW_TICKS) if window_ticks is None else window_ticks
    state = QwenStreamState.initial(backbone, window_ticks=window).extend_prefix(layout["tokens"][: layout["prefix_end"]])
    pieces = [state.prefix_hidden]
    for tick in layout["ticks"]:
        state = state.advance(layout["tokens"][tick["start"] : tick["body_end"]])
        assert state.position == (layout["position"][tick["body_end"]] if tick["body_end"] < len(layout["position"]) else state.position)
        pieces.append(state.hidden)
        decisions = layout["tokens"][tick["body_end"] : tick["end"]]
        if decisions:
            pieces.append(state.branch_step(decisions))
            assert state.position == layout["position"][tick["body_end"]]  # 분기는 position을 쓰지 않는다
    return torch.cat(pieces), state


def relative_l2(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return ((a.float() - b.float()).norm(dim=-1) / b.float().norm(dim=-1)).cpu()


def readout_positions(layout: dict) -> list[int]:
    out = []
    for tick in layout["ticks"]:
        out.extend(int(p) for p in tick["decision_positions"].values())
        for boundaries in tick["candidate_boundaries"].values():
            out.extend(int(b) for b in boundaries)
    return sorted(set(out))


def test_tiny_incremental_matches_from_scratch_and_evicts_by_tick(tiny):
    layout = synthetic_stream(random.Random(0), prefix=7, ticks=[(5, 3), (6, 2), (4, 3), (5, 2), (3, 2)], window_ticks=2)
    reference = tiny.forward_layout(layout)
    hidden, state = incremental(tiny, layout)
    torch.testing.assert_close(hidden, reference, **CPU_TOL)
    # 윈도우(2틱) 밖의 틱은 cache에서 나갔다 — prefix는 남는다
    assert state.cache_ticks.tolist() == [-1] * 7 + [3] * 5 + [4] * 3
    assert state.cached_tokens == 15 and state.window_tokens == 8
    mask = build_reference_mask(layout)
    last = layout["ticks"][-1]["body_end"] - 1
    assert int(mask[last].sum()) == state.cached_tokens  # 기준 mask가 보이는 key 수 == 버퍼의 살아 있는 토큰 수
    # 절단 없는 계산과의 차이는 정의된 근사 — 0이 아니고 유한하다 (docs/08 §3.1)
    untruncated = tiny.forward_layout(layout, window_ticks=100)
    beyond = [i for i, t in enumerate(layout["tick"]) if t >= 2]
    gap = (reference[beyond].float() - untruncated[beyond].float()).norm(dim=-1)
    assert torch.isfinite(gap).all() and gap.max() > 0


def test_tiny_window_buffer_compacts_when_it_reaches_the_end_and_never_grows(tiny):
    """버퍼 용량이 작으면 살아 있는 윈도우를 앞으로 당긴다(한 번의 복사) — 결과는 같고 버퍼 크기는 고정이다."""
    layout = synthetic_stream(random.Random(3), prefix=4, ticks=[(6, 1)] * 12, window_ticks=3)
    reference = tiny.forward_layout(layout)
    small = QwenBackbone(tiny.model, model_id=tiny.model_id, kv_mode="static", window_capacity=20)  # 윈도우 18 + 여유 2
    hidden, state = incremental(small, layout)
    torch.testing.assert_close(hidden, reference, **CPU_TOL)
    store = state._kv[0]
    assert store.capacity == 20 and store.k_win.shape[0] == 20 and store.end - store.start == 18
    assert state.kv_bytes() == sum(s.buffer_bytes() for s in state._kv)
    tight = QwenBackbone(tiny.model, model_id=tiny.model_id, kv_mode="static", window_capacity=17)
    with pytest.raises(RuntimeError, match="window_capacity"):
        incremental(tight, layout)


def test_tiny_batched_branches_equal_sequential_and_leave_the_common_state_untouched(tiny):
    layout = synthetic_stream(random.Random(1), prefix=6, ticks=[(5, 4), (4, 4)], window_ticks=30)
    _, state = incremental(tiny, layout)
    before = state.clone()
    tokens = [65, 66, 67, 68]
    batched = state.branch_step(tokens)
    sequential = torch.stack([branch.step(token) for branch, token in zip(state.fork(4), tokens)])
    torch.testing.assert_close(batched, sequential, **CPU_TOL)
    subset = state.branch_step(tokens[1:3])
    torch.testing.assert_close(subset, batched[1:3], **CPU_TOL)  # 분기끼리 서로의 출력을 바꾸지 않는다
    for layer_before, layer_after in zip(before.delta, state.delta):
        assert torch.equal(layer_before["recurrent"], layer_after["recurrent"]) and torch.equal(layer_before["conv"], layer_after["conv"])
    for kv_before, kv_after in zip(before.kv, state.kv):
        assert torch.equal(kv_before["k"], kv_after["k"]) and torch.equal(kv_before["v"], kv_after["v"])
    assert state.position == before.position and state.cached_tokens == before.cached_tokens
    # 다음 틱은 분기 이전 공통 상태에서: 분기를 돌린 상태와 안 돌린 상태의 다음 틱이 비트 단위로 같다
    nxt = state.advance([31, 32, 33])
    plain = before.advance([31, 32, 33])
    for a, b in zip(nxt.delta, plain.delta):
        assert torch.equal(a["recurrent"], b["recurrent"])
    assert torch.equal(nxt.hidden, plain.hidden)
    with pytest.raises(ValueError, match="분기"):
        state.fork(1)[0].advance([1])


def test_tiny_dynamic_kv_mode_matches_static_and_carries_gradient_through_the_window(tiny):
    layout = synthetic_stream(random.Random(5), prefix=5, ticks=[(4, 2), (3, 2), (4, 2)], window_ticks=30)
    static_hidden, _ = incremental(tiny, layout)
    dynamic = QwenBackbone(tiny.model, model_id=tiny.model_id, kv_mode="dynamic")
    dynamic_hidden, state = incremental(dynamic, layout)
    torch.testing.assert_close(dynamic_hidden, static_hidden, **CPU_TOL)
    # gradient: 마지막 분기의 손실이 앞 틱의 KV·recurrent 상태를 거쳐 prefix 토큰의 embedding까지 닿는다
    trainable = QwenBackbone.tiny(seed=1, vocab_size=1024, layers=("linear_attention", "full_attention"))
    trainable.model.requires_grad_(True)
    trainable.kv_mode = "dynamic"
    with torch_reference_kernels():
        _, state = incremental(trainable, layout)
        loss = state.branch_step([65, 66]).pow(2).sum()
        loss.backward()
    rows = trainable.text.embed_tokens.weight.grad
    assert rows is not None and (rows[layout["tokens"][: layout["prefix_end"]]].norm(dim=-1) > 0).all()
    assert rows[[1000, 1001]].abs().sum() == 0


def test_tiny_state_round_trips_through_dict_and_detach(tiny):
    layout = synthetic_stream(random.Random(7), prefix=5, ticks=[(4, 2), (3, 2), (4, 2)], window_ticks=2)
    _, state = incremental(tiny, layout)
    packed = stream_state_to_dict(state)
    assert packed["kind"] == "qwen" and packed["prefix_len"] == 5 and packed["ticks"] == [[1, 3], [2, 4]]
    restored = stream_state_from_dict(packed, tiny)
    assert restored.cache_ticks.tolist() == state.cache_ticks.tolist() and restored.position == state.position
    torch.testing.assert_close(restored.advance([40, 41]).hidden, state.clone().advance([40, 41]).hidden, **CPU_TOL)
    detached = state.detach()
    assert not detached.is_branch and detached.tick == state.tick and detached.cached_tokens == state.cached_tokens
    with pytest.raises(ValueError, match="qwen"):
        stream_state_from_dict({**packed, "kind": "tiny"}, tiny)


def test_merge_attention_parts_reproduces_a_masked_softmax_on_cpu():
    torch.manual_seed(0)
    T, H, Hk, D, N = 5, 4, 2, 8, 9
    q = torch.randn(T, H, D)
    k_old, v_old = torch.randn(N, Hk, D), torch.randn(N, Hk, D)
    k_new, v_new = torch.randn(T, Hk, D), torch.randn(T, Hk, D)
    out = adapter.windowed_attention(q, [(k_old, v_old)], (k_new, v_new), scale=D**-0.5)
    k_all = torch.cat([k_old, k_new]).repeat_interleave(H // Hk, dim=1)
    v_all = torch.cat([v_old, v_new]).repeat_interleave(H // Hk, dim=1)
    mask = torch.ones(T, N + T, dtype=torch.bool).tril(diagonal=N)
    reference = torch.nn.functional.scaled_dot_product_attention(q.transpose(0, 1)[None], k_all.transpose(0, 1)[None], v_all.transpose(0, 1)[None], attn_mask=mask)[0].transpose(0, 1)
    torch.testing.assert_close(out, reference, rtol=1e-4, atol=1e-5)
    assert merge_attention_parts([adapter.AttentionParts(out, torch.zeros(T, H))]) is out


def test_judge_runs_both_layouts_on_the_adapter_and_from_scratch_agrees(tiny, streams, singles):
    tokenizer = WhitespaceTokenizer()
    record = copy.deepcopy(streams[0])
    record["ticks"] = record["ticks"][:2]
    layout = serialize_request(record, tokenizer, layout="stream_l1a")
    judge = Judge(tiny, rank=8, readout="pointer", seed=3)
    incremental_out = judge({"layout": "stream_l1a", "stream": layout})
    scratch = judge({"layout": "stream_l1a", "stream": layout, "from_scratch": True})
    for a, b in zip(incremental_out["logits"], scratch["logits"]):
        for qid in a:
            torch.testing.assert_close(a[qid], b[qid], rtol=1e-3, atol=1e-3)
    assert isinstance(incremental_out["state"], QwenStreamState)
    replay = replay_layout(layout, backbone=tiny)
    assert isinstance(replay["final"], QwenStreamState) and replay["hidden"].shape[0] == len(layout["tokens"])
    single = next(r for r in singles if len(r["request"]["questions"]) >= 2)
    out = judge({"layout": "state_first", "states": [serialize_request(single, tokenizer)]})
    assert set(out["logits"][0]) == set(single_q["id"] for single_q in single["request"]["questions"])
    with pytest.raises(ValueError, match="mask"):
        tiny(torch.zeros(1, 3, dtype=torch.long), torch.zeros(1, 3, dtype=torch.long), mask=torch.ones(3, 3, dtype=torch.bool))


# --------------------------------------------------------------------------
# 배포 계약 digest·학습 연결 (CPU)
# --------------------------------------------------------------------------


def test_contract_digest_covers_the_four_parts_and_moves_with_one_byte():
    base = contract_digest("00" * 32)
    assert set(base["parts"]) == set(CONTRACT_PARTS) and base["harness_version"] == harness_version()
    source = (REPO / "src" / "robo_jev" / "model" / "serialize.py").read_bytes()
    assert base["parts"]["serialize_py"] == hashlib.sha256(source).hexdigest()
    flipped = bytearray(source)
    flipped[len(flipped) // 2] ^= 0x01
    changed = contract_digest("00" * 32, serialize_source=bytes(flipped))
    assert changed["sha256"] != base["sha256"] and contract_differences(base, changed) == ["serialize_py"]
    assert contract_differences(base, contract_digest("11" * 32)) == ["tokenizer_sha256"]
    assert contract_differences(base, contract_digest("00" * 32, harness="h9.9")) == ["harness_version"]
    assert contract_differences(base, base) == [] and contract_differences(None, base) == ["missing"]


def test_train_manifest_carries_the_contract_digest_and_resume_refuses_a_different_checkout(tmp_path, monkeypatch):
    from robo_jev import train as training
    from test_train import tiny_config

    with training.Trainer(tiny_config(tmp_path, max_steps=1)) as trainer:
        manifest = trainer.manifest
        assert manifest["contract_sha256"] == manifest["contract"]["sha256"] == manifest["identity"]["contract_sha256"]
        assert manifest["contract"]["tokenizer_sha256"] == "whitespace"
        trainer.run_step()
        path = trainer.save(tmp_path / "ckpt.pt")
    original = training.contract_digest
    monkeypatch.setattr(training, "contract_digest", lambda sha, **kw: original(sha, harness="h9.9", **kw))
    with pytest.raises(ValueError, match="harness_version"):
        training.Trainer(tiny_config(tmp_path, max_steps=2), resume=path)


def test_readout_only_run_on_the_adapter_saves_only_the_readout_and_resumes(tmp_path, monkeypatch):
    from robo_jev import train as training
    from test_train import tiny_config

    fake = QwenBackbone.tiny(seed=2, vocab_size=SMALL_VOCAB)
    fake.manifest = {"revision": "deadbeef" * 5, "digest": "cafe" * 16}
    monkeypatch.setattr(training.QwenBackbone, "load", classmethod(lambda cls, model_id, **kwargs: fake))
    config = tiny_config(tmp_path, max_steps=2, trainable="readout_only", model_id=REAL_2B, readout_rank=4, stream_max_ticks=3)
    with training.Trainer(config) as trainer:
        assert isinstance(trainer.model.backbone, QwenBackbone) and trainer.model.rank == 4
        model = trainer.manifest["model"]
        assert model["kind"] == "qwen3_5" and model["class"] == "QwenBackbone" and model["revision"] == "deadbeef" * 5
        assert model["trainable_parameters"] == sum(p.numel() for n, p in trainer.model.named_parameters() if not n.startswith("backbone."))
        assert trainer.manifest["identity"]["model"]["digest"] == "cafe" * 16 and model["readout_dtype"] == "float32"
        trainer.run_step()
        state = trainer.checkpoint_state()
        assert set(state["model"]) == {"U.weight", "V.weight", "bias"}  # readout-only checkpoint: backbone은 저장하지 않는다
        path = trainer.save(tmp_path / "ckpt.pt")
        readout_after = trainer.model.U.weight.detach().clone()
    with training.Trainer(config, resume=path) as resumed:
        assert resumed.step == 1 and torch.equal(resumed.model.U.weight, readout_after)
        assert all(not p.requires_grad for n, p in resumed.model.named_parameters() if n.startswith("backbone."))


# --------------------------------------------------------------------------
# 실제 Qwen3.5-2B (GPU, 가중치가 있을 때만)
# --------------------------------------------------------------------------


@functools.lru_cache(maxsize=1)
def real_2b() -> QwenBackbone:
    return QwenBackbone.load(REAL_2B)


@functools.lru_cache(maxsize=1)
def real_tokenizer():
    return load_tokenizer(available_tokenizer()[0])


def d0_stream_layout(ticks: int, index: int = 0) -> dict:
    record = copy.deepcopy(read_jsonl(D0_STREAMS)[index])
    record["ticks"] = record["ticks"][:ticks]
    return serialize_request(record, real_tokenizer(), layout="stream_l1a")


@needs_real_2b
def test_real_2b_incremental_matches_the_reference_within_the_measured_bf16_tolerance(capsys):
    backbone = real_2b()
    layout = d0_stream_layout(6)
    reference = backbone.forward_layout(layout)
    hidden, state = incremental(backbone, layout)
    again, _ = incremental(backbone, layout)
    assert torch.equal(hidden, again)  # 증분 경로는 실행마다 비트 단위로 같다
    rel = relative_l2(hidden, reference)
    positions = readout_positions(layout)
    print(
        f"real 2B bf16 incremental vs reference: rel L2 at readout positions max {rel[positions].max():.4f}, "
        f"median all {rel.median():.4f}, max all {rel.max():.4f}, hidden absmax {reference.float().abs().max():.1f}, "
        f"attention backend {backbone.attention_backend}"
    )
    assert rel[positions].max() <= BF16_REL_READOUT and rel.median() <= BF16_REL_MEDIAN and rel.max() <= BF16_REL_MAX
    assert backbone.attention_backend is not None and "no mask" in backbone.attention_backend
    assert state.cached_tokens == len(layout["tokens"]) - sum(kind == "decision" for kind in layout["kind"])
    assert layout["decision_markers"] == {qid: layout["decision_markers"][qid] for qid in layout["question_ids"]}  # 표지 고정


@needs_real_2b
def test_real_2b_window_truncation_keeps_the_cache_constant_and_matches_the_reference_at_the_last_tick():
    backbone = real_2b()
    layout = d0_stream_layout(33)  # 윈도우 30 → 마지막 세 틱은 절단된 채 계산된다
    last = layout["ticks"][-1]
    state = QwenStreamState.initial(backbone).extend_prefix(layout["tokens"][: layout["prefix_end"]])
    sizes = []
    for tick in layout["ticks"]:
        state = state.advance(layout["tokens"][tick["start"] : tick["body_end"]])
        sizes.append(state.cached_tokens)
    mask = build_reference_mask(layout)
    visible = int(mask[last["body_end"] - 1].sum())
    assert state.cached_tokens == visible and len(state._ticks) == 30
    bodies = [int(t["body_end"]) - int(t["start"]) for t in layout["ticks"]]
    for index in range(29, 33):  # 윈도우가 찬 뒤 cache 길이 = prefix + 최근 30틱의 몸통 토큰 — 더 자라지 않는다
        assert sizes[index] == int(layout["prefix_end"]) + sum(bodies[index - 29 : index + 1])
    branches = state.branch_step(layout["tokens"][last["body_end"] : last["end"]])
    reference = backbone.forward_layout(layout, decision_ticks=[int(last["index"])])
    decisions = list(range(last["body_end"], last["end"]))
    rel = relative_l2(branches, reference[decisions])
    body_rel = relative_l2(state.hidden, reference[last["start"] : last["body_end"]])
    boundaries = [b - last["start"] for qid in last["candidate_boundaries"] for b in last["candidate_boundaries"][qid] if b >= last["start"]]
    print(f"real 2B window-truncated last tick: decisions rel max {rel.max():.4f}, body rel max {body_rel.max():.4f}, at boundaries {body_rel[boundaries].max():.4f}")
    assert rel.max() <= BF16_REL_READOUT and body_rel[boundaries].max() <= BF16_REL_READOUT


@needs_real_2b
def test_real_2b_branches_are_isolated_and_the_next_tick_continues_from_the_common_state():
    backbone = real_2b()
    layout = d0_stream_layout(3)
    _, state = incremental(backbone, layout)
    before = state.clone()
    tokens = layout["tokens"][layout["ticks"][-1]["body_end"] : layout["ticks"][-1]["end"]]
    batched = state.branch_step(tokens)
    for layer_before, layer_after in zip(before.delta, state.delta):
        assert torch.equal(layer_before["recurrent"], layer_after["recurrent"]) and torch.equal(layer_before["conv"], layer_after["conv"])
    for kv_before, kv_after in zip(before.kv, state.kv):
        assert torch.equal(kv_before["k"], kv_after["k"]) and torch.equal(kv_before["v"], kv_after["v"])
    subset = state.branch_step(tokens[2:5])
    assert relative_l2(subset, batched[2:5]).max() <= BF16_REL_READOUT  # 분기끼리 서로의 logits를 바꾸지 않는다
    tail = layout["tokens"][layout["ticks"][-1]["start"] : layout["ticks"][-1]["body_end"]]
    torch.testing.assert_close(state.advance(tail).hidden, before.advance(tail).hidden, rtol=0, atol=0)  # 다음 틱은 분기 이전 공통 상태에서


@needs_real_2b
def test_real_2b_native_and_stream_paths_agree_on_a_single_request():
    """native(공식 배치 forward, P0 경로)와 stream(prefix로 읽고 결정 표지 분기)이 같은 단일 요청에서 같은 hidden을 낸다."""
    backbone = real_2b()
    record = next(r for r in read_jsonl(D0) if len(r["request"]["questions"]) >= 2)
    layout = serialize_request(record, real_tokenizer())
    judge = Judge(backbone, rank=16, readout="pointer", seed=1, readout_dtype=torch.float32)
    native = judge({"layout": "state_first", "states": [layout]})["logits"][0]
    S = int(layout["state_end"])
    for branch, qid in enumerate(layout["question_ids"]):
        decision = int(layout["decision_positions"][qid])
        owned = [i for i, q in enumerate(layout["question"]) if q == branch]
        path = layout["tokens"][:S] + layout["tokens"][owned[0] : decision]
        state = QwenStreamState.initial(backbone).extend_prefix(path)  # 질문 경로 전체를 prefix처럼 읽고
        state.position = int(layout["position"][decision])
        hidden = state.branch_step([layout["tokens"][decision]])[0]  # 결정 표지를 1토큰 분기로
        boundaries = [S + (b - owned[0]) for b in layout["candidate_boundaries"][qid]]
        logits = judge.pointer_logits(hidden, state.prefix_hidden[boundaries])
        probs_native, probs_stream = torch.softmax(native[qid].float(), 0), torch.softmax(logits.float(), 0)
        assert (probs_native - probs_stream).abs().max() <= 0.05, (qid, probs_native, probs_stream)


@needs_real_2b
def test_real_2b_fp32_incremental_is_within_the_official_cache_path_noise():
    """fp32에서는 증분 경로가 공식 구현의 캐시 경로 잡음 안 — bf16 허용 오차가 버그가 아니라 반정밀도 잡음임을 보인다."""
    backbone = QwenBackbone.load(REAL_2B, dtype=torch.float32)
    layout = d0_stream_layout(4)
    reference = backbone.forward_layout(layout)
    hidden, _ = incremental(backbone, layout)
    rel = relative_l2(hidden, reference)
    print(f"real 2B fp32 incremental vs reference: rel L2 max {rel.max():.4f} median {rel.median():.5f}")
    assert rel.max() <= FP32_REL_MAX
    del backbone
    torch.cuda.empty_cache()
