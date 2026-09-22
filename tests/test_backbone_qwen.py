"""실제 backbone adapter(`robo_jev.model.backbone_qwen`)의 검사 (Task 2b G0b S1).

CPU 검사는 **소형 난수 Qwen3.5**(transformers의 구조, torch 참조 kernel)로 bookkeeping을 본다 — 정적 윈도우 버퍼의 퇴출·
당김, 배치 결정 분기 == 순차 분기, 증분 == 처음부터(기준 mask), dynamic KV 모드의 gradient, checkpoint 왕복, Judge·
학습 loop 연결, 배포 계약 digest. GPU 검사는 **실제 Qwen3.5-2B**(가중치가 있을 때만; 없으면 skip)로 BF16 허용 오차를
실측해 고정한 상수와 대조한다 — 증분 == 처음부터, 윈도우 절단 뒤 cache 길이 불변, 분기 격리, native == stream(단일 요청).

BF16 허용 오차(D0 스트림 6틱·1,525토큰, `.superpowers/sdd/task-g0b-report.md` S1.3·Fix round 1): GPU 검사의 기준은
**공식 forward + 공식 kernel**(fla `chunk_gated_delta_rule`, causal_conv1d, varlen flash — 배포가 도는 것과 같다; CPU 검사만
torch 참조 kernel)이며, 그 조건에서 2026-09-20에 5회 재측정한 값이 결정 위치·후보 경계 hidden의 토큰별 상대 L2 0.0264
(중앙값 0.0152), 전 토큰 최대 0.466(outlier 차원의 bf16 반올림, hidden 절대 최대 104), fp32 0.0101이다 — 상수는 그 2배.
공식 구현 자체의 kernel 사이 차이(mask 있음/없음, DynamicCache 틱별/한 번)가 같은 자릿수(상대 L2 최대 0.27~0.40,
중앙값 0.014)라 bf16 잡음이다. (2026-09-19의 첫 측정 0.020 / 0.015 / 0.47 / 0.018은 참조 kernel이 섞인 조건이었다.)
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
#: 2026-09-20 GB10, **이 테스트와 같은 조건**(공식 forward + 공식 kernel: fla chunk_gated_delta_rule·causal_conv1d·varlen flash)에서
#: 5회 반복 — 다섯 번 모두 같은 값(결정적): readout 0.0264 / 중앙값 0.0152 / 전체 최대 0.4659, 절단 마지막 틱 결정 0.0233·
#: 경계 0.0166, 분기 부분집합 vs 배치 0.0126, fp32 최대 0.0101. 상수 = 실측의 2배 (G0b 리뷰 1 I4).
BF16_REL_READOUT = 0.053  # 실측 0.0264의 2배
BF16_REL_MEDIAN = 0.031  # 실측 0.0152
BF16_REL_MAX = 0.93  # 실측 0.4659 (공식 구현의 kernel 사이 차이 0.27~0.40)
#: fp32는 프로세스에 따라 다르다 — 같은 조건의 단독 탐침 5회는 모두 0.0101, 전체 suite 안에서는 0.0243(리뷰 1의 측정도 0.0243).
#: **원인은 autotune tiling이 아니라 기준 쪽의 kernel이 바뀌는 것이다**(G0b 리뷰 1 I4에서 진단, 그 보고서 §Fix round 1):
#: suite 안에서는 transformers의 hub-kernels wrapper가 "`chunk_gated_delta_rule` is falling back to its reference PyTorch
#: implementation"을 찍는다 — 즉 **공식 forward(기준)가 torch chunk 구현으로 내려가고** adapter는 fla의 fp32 kernel을 그대로 쓰므로
#: 0.0243은 그 kernel 대 kernel의 fp32 차이다. 단독 탐침에서는 양쪽 다 fla라 0.0101이 된다. bf16은 반올림이 지배해 두 조건에서
#: 거의 같다. 상수는 관측 최대의 2배.
FP32_REL_MAX = 0.05  # 실측 0.0101(탐침) / 0.0243(suite)
CPU_TOL = {"rtol": 1e-4, "atol": 1e-4}

REAL_2B = "Qwen/Qwen3.5-2B"


def real_weights_present() -> bool:
    try:
        describe_backbone(REAL_2B)
    except (FileNotFoundError, ValueError):
        return False
    return torch.cuda.is_available()


def needs_real_2b(func):
    """실제 2B 가중치 + CUDA가 있어야 하는 검사 — 없으면 skip하고, **`real_2b` 표지를 붙인다**.

    `_torch_kernels` fixture가 이 표지로 그 검사를 고른다(검사 **이름**에 `real_2b`가 들어 있는지 보던 것을 바꾼 것 —
    G0b 리뷰 2 M-c): 이름을 바꿔도 kernel 선택이 조용히 뒤집히지 않는다.
    """
    marked = pytest.mark.real_2b(func)
    return pytest.mark.skipif(not real_weights_present(), reason="Qwen3.5-2B 가중치(artifacts/models)와 CUDA가 있어야 한다")(marked)


# --------------------------------------------------------------------------
# 소형 난수 Qwen (CPU)
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def tiny() -> QwenBackbone:
    return QwenBackbone.tiny(seed=1, vocab_size=1024)


@pytest.fixture(autouse=True, scope="module")
def _gpu_guard():
    """실제 2B 검사는 GB10의 통합 메모리 울타리 안에서 (robo_jev.gpu; 첫 CUDA 할당 전에 건다)."""
    if torch.cuda.is_available():
        from robo_jev.gpu import limit_gpu_memory

        limit_gpu_memory()


@pytest.fixture(autouse=True)
def _torch_kernels(request):
    """CPU 검사는 공식 forward도 torch 참조 kernel로 돈다 (CUDA가 있는 venv에서 fla·causal_conv1d가 CPU tensor를 받지 않게).
    **`real_2b` 표지가 붙은 검사**는 공식 kernel(fla `chunk_gated_delta_rule`·causal_conv1d·flash)이 기준이다 — 허용 오차 상수도
    그 조건에서 쟀다(모듈 설명; G0b 리뷰 1 I4). 이름이 아니라 표지로 고르는 이유는 검사 이름을 바꿨을 때 kernel 선택이
    조용히 뒤집히지 않게 하려는 것이다(G0b 리뷰 2 M-c)."""
    if request.node.get_closest_marker("real_2b") is not None:
        yield
        return
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


def test_the_r1_serializer_change_moves_the_contract_digest_so_old_checkpoints_are_refused(tmp_path, monkeypatch):
    """Task R1 A2: 풀어 놓은 목표를 모델 입력에서 뺀 것은 `serialize.py`의 변경이므로 digest가 바뀐다 — P1~P3의
    체크포인트(`ts0.5` 서식으로 학습된 것)는 적재에서 거절되고, 별도 가드가 필요 없다."""
    from robo_jev import train as training
    from test_train import tiny_config

    source = (REPO / "src" / "robo_jev" / "model" / "serialize.py").read_bytes()
    assert b'TOKEN_SERIALIZER_VERSION = "ts0.6"' in source
    old_source = source.replace(b'TOKEN_SERIALIZER_VERSION = "ts0.6"', b'TOKEN_SERIALIZER_VERSION = "ts0.5"')
    base = contract_digest("00" * 32)
    assert contract_differences(base, contract_digest("00" * 32, serialize_source=old_source)) == ["serialize_py"]

    # 그 차이가 실제로 적재를 막는다: 옛 서식의 digest를 단 체크포인트는 지금 체크아웃에서 재개되지 않는다.
    with training.Trainer(tiny_config(tmp_path, max_steps=1)) as trainer:
        trainer.run_step()
        path = trainer.save(tmp_path / "ts05.pt")
    original = training.contract_digest
    monkeypatch.setattr(training, "contract_digest", lambda sha, **kw: original(sha, serialize_source=old_source, **kw))
    with pytest.raises(ValueError, match="serialize_py"):
        training.Trainer(tiny_config(tmp_path, max_steps=2), resume=path)


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


def test_load_readout_checkpoint_checks_the_tokenizer_hash_and_requires_it(tmp_path, monkeypatch):
    """서빙·평가용 적재는 배포 계약 digest의 네 조각을 지금 체크아웃 기준으로 대조한다 — tokenizer 파일 해시가 다르면 그 조각 이름을
    들어 거절하고, 해시를 안 주면 `trust_checkpoint_tokenizer=True`를 명시할 때만 checkpoint의 해시로 대신한다 (리뷰 1 I2)."""
    from robo_jev import train as training
    from test_train import tiny_config

    fake = QwenBackbone.tiny(seed=2, vocab_size=SMALL_VOCAB)
    fake.manifest = {"revision": "deadbeef" * 5, "digest": "cafe" * 16}
    monkeypatch.setattr(training.QwenBackbone, "load", classmethod(lambda cls, model_id, **kwargs: fake))
    config = tiny_config(tmp_path, max_steps=1, trainable="readout_only", model_id=REAL_2B, readout_rank=4, stream_max_ticks=3)
    with training.Trainer(config) as trainer:
        trainer.run_step()
        path = trainer.save(tmp_path / "ckpt.pt")
        saved = trainer.model.U.weight.detach().clone()
        saved_hash = trainer.manifest["contract"]["tokenizer_sha256"]
    judge = Judge(fake, rank=4, readout="pointer", seed=99, readout_dtype=torch.float32)
    assert not torch.equal(judge.U.weight, saved)
    with pytest.raises(ValueError, match="tokenizer_sha256"):
        training.load_readout_checkpoint(judge, path, tokenizer_sha256="11" * 32)
    with pytest.raises(ValueError, match="tokenizer_sha256"):
        training.load_readout_checkpoint(judge, path, tokenizer_sha256=None)
    manifest = training.load_readout_checkpoint(judge, path, tokenizer_sha256=saved_hash)
    assert torch.equal(judge.U.weight, saved) and manifest["contract"]["tokenizer_sha256"] == saved_hash
    trusting = Judge(fake, rank=4, readout="pointer", seed=98, readout_dtype=torch.float32)
    training.load_readout_checkpoint(trusting, path, tokenizer_sha256=None, trust_checkpoint_tokenizer=True)
    assert torch.equal(trusting.U.weight, saved)


# --------------------------------------------------------------------------
# 실제 Qwen3.5-2B (GPU, 가중치가 있을 때만)
# --------------------------------------------------------------------------


@functools.lru_cache(maxsize=1)
def real_2b() -> QwenBackbone:
    return QwenBackbone.load(REAL_2B)


@functools.lru_cache(maxsize=1)
def real_2b_fp32() -> QwenBackbone:
    """fp32 2B — 의미 검사(native == stream)와 fp32 잡음 검사가 같이 쓴다 (bf16 잡음이 아닌 것을 보려면 fp32여야 한다)."""
    return QwenBackbone.load(REAL_2B, dtype=torch.float32)


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
    """native(공식 배치 forward, P0 경로)와 stream(prefix로 읽고 결정 표지 분기)이 같은 단일 요청에서 같은 hidden을 낸다 —
    **fp32**로 본다: bf16에서는 난수 readout의 확률 차이가 kernel 잡음만으로 0.07~0.11까지 벌어져(S1.3 탐침) 의미 검사가 되지 않는다."""
    backbone = real_2b_fp32()
    record = next(r for r in read_jsonl(D0) if len(r["request"]["questions"]) >= 2)
    layout = serialize_request(record, real_tokenizer())
    judge = Judge(backbone, rank=16, readout="pointer", seed=1, readout_dtype=torch.float32)
    native = judge({"layout": "state_first", "states": [layout]})["logits"][0]  # P0: 질문 경로마다 독립 forward
    S = int(layout["state_end"])
    worst = 0.0
    worst_rel = 0.0
    for branch, qid in enumerate(layout["question_ids"]):
        decision = int(layout["decision_positions"][qid])
        owned = [i for i, q in enumerate(layout["question"]) if q == branch]
        path = layout["tokens"][:S] + layout["tokens"][owned[0] : decision]
        # 기준 hidden: **이 질문 경로만**(상태 + 자기 경로 + 결정 표지)의 공식 forward — P0 의미. 레이아웃 전체를 한 causal 시퀀스로
        # 돌리면 뒤 질문 경로가 앞 경로를 보게 되어 첫 질문 말고는 기준이 아니다 (fix round 1에서 그렇게 잘못 대조해 0.87이 나왔다).
        with torch.no_grad():
            native_hidden = backbone(torch.tensor([path + [layout["tokens"][decision]]]), torch.tensor([layout["position"][:S] + layout["position"][owned[0] : decision + 1]]))["hidden"][0]
        state = QwenStreamState.initial(backbone).extend_prefix(path)  # 질문 경로 전체를 prefix처럼 읽고
        state.position = int(layout["position"][decision])
        hidden = state.branch_step([layout["tokens"][decision]])[0]  # 결정 표지를 1토큰 분기로
        boundaries = [S + (b - owned[0]) for b in layout["candidate_boundaries"][qid]]
        rel_decision = float(relative_l2(hidden[None], native_hidden[len(path)][None]).max())
        rel_boundaries = float(relative_l2(state.prefix_hidden[boundaries], native_hidden[boundaries]).max())
        worst_rel = max(worst_rel, rel_decision, rel_boundaries)
        assert rel_decision <= FP32_REL_MAX and rel_boundaries <= FP32_REL_MAX, (qid, rel_decision, rel_boundaries)
        logits = judge.pointer_logits(hidden, state.prefix_hidden[boundaries])
        probs_native, probs_stream = torch.softmax(native[qid].detach().float(), 0), torch.softmax(logits.detach().float(), 0)
        worst = max(worst, float((probs_native - probs_stream).abs().max()))
        assert (probs_native - probs_stream).abs().max() <= 0.05, (qid, probs_native, probs_stream)
    print(f"real 2B fp32 native vs stream: hidden rel L2 max {worst_rel:.4f} (decision + boundary positions), readout prob abs diff max {worst:.4f}")


@needs_real_2b
def test_real_2b_fp32_incremental_is_within_the_official_cache_path_noise():
    """fp32에서는 증분 경로가 공식 구현의 캐시 경로 잡음 안 — bf16 허용 오차가 버그가 아니라 반정밀도 잡음임을 보인다."""
    backbone = real_2b_fp32()
    layout = d0_stream_layout(4)
    reference = backbone.forward_layout(layout)
    hidden, _ = incremental(backbone, layout)
    rel = relative_l2(hidden, reference)
    print(f"real 2B fp32 incremental vs reference: rel L2 max {rel.max():.4f} median {rel.median():.5f}")
    assert rel.max() <= FP32_REL_MAX
    real_2b_fp32.cache_clear()
    torch.cuda.empty_cache()


def test_lora_run_on_the_adapter_trains_only_lora_and_readout_and_checkpoints_them(tmp_path, monkeypatch):
    """`trainable: lora_and_readout` — peft LoRA가 projection에 붙고(fp32, 학습 대상), 기본 가중치는 고정·불변, checkpoint는
    LoRA + readout만. KV는 그래프를 유지하는 dynamic 모드라 분기 손실의 gradient가 LoRA에 닿는다."""
    pytest.importorskip("peft")
    from robo_jev import train as training
    from test_train import tiny_config

    fake = QwenBackbone.tiny(seed=3, vocab_size=SMALL_VOCAB)
    base_before = fake.text.layers[0].mlp.gate_proj.weight.detach().clone()
    monkeypatch.setattr(training.QwenBackbone, "load", classmethod(lambda cls, model_id, **kwargs: (setattr(fake, "kv_mode", kwargs.get("kv_mode", "static")) or fake)))
    config = tiny_config(
        tmp_path, max_steps=1, trainable="lora_and_readout", model_id=REAL_2B, readout_rank=4, stream_max_ticks=3,
        lora={"r": 2, "alpha": 4, "targets": ["q_proj", "gate_proj", "in_proj_qkv"]},
    )
    with training.Trainer(config) as trainer:
        assert trainer.model.backbone.kv_mode == "dynamic"
        names = [n for n, p in trainer.model.named_parameters() if p.requires_grad]
        assert all(("lora_" in n) or not n.startswith("backbone.") for n in names) and any("lora_A" in n for n in names)
        assert trainer.manifest["model"]["trainable"] == "lora_and_readout" and trainer.manifest["model"]["lora"]["r"] == 2
        metrics = trainer.run_step()
        assert metrics is not None and metrics["grad_norm"] > 0
        lora_grads = [n for n, p in trainer.model.named_parameters() if "lora_" in n and p.grad is not None]
        state = trainer.checkpoint_state()["model"]
        assert {"U.weight", "V.weight", "bias"} <= set(state) and any("lora_A" in key for key in state) and not any("base_layer.weight" in key for key in state)
        layer0 = trainer.model.backbone.text.layers[0].mlp.gate_proj
        assert torch.equal(layer0.base_layer.weight, base_before) and layer0.lora_A["default"].weight.dtype == torch.float32
        assert trainer.optimizer.param_groups[0]["name"] == "backbone/decay" and lora_grads == []  # gradient는 step 뒤 지워진다 (set_to_none)


def test_tiny_fused_advance_with_branches_equals_advance_then_branch_step(tiny):
    layout = synthetic_stream(random.Random(9), prefix=6, ticks=[(5, 3), (4, 3), (6, 3)], window_ticks=2)
    reference = tiny.forward_layout(layout)
    state = QwenStreamState.initial(tiny, window_ticks=2).extend_prefix(layout["tokens"][: layout["prefix_end"]])
    pieces = [state.prefix_hidden]
    for tick in layout["ticks"]:
        body = layout["tokens"][tick["start"] : tick["body_end"]]
        decisions = layout["tokens"][tick["body_end"] : tick["end"]]
        separate = state.clone().advance(body)
        expected = separate.branch_step(decisions)
        state, branches = state.advance_with_branches(body, decisions)
        torch.testing.assert_close(branches, expected, **CPU_TOL)
        torch.testing.assert_close(state.hidden, separate.hidden, **CPU_TOL)
        for a, b in zip(state.delta, separate.delta):
            torch.testing.assert_close(a["recurrent"], b["recurrent"], **CPU_TOL)
        assert state.cache_ticks.tolist() == separate.cache_ticks.tolist() and state.position == separate.position
        pieces.extend([state.hidden, branches])
    torch.testing.assert_close(torch.cat(pieces), reference, **CPU_TOL)


@needs_real_2b
def test_real_2b_fused_forward_matches_the_separate_forwards_within_the_bf16_tolerance():
    backbone = real_2b()
    layout = d0_stream_layout(4)
    state = QwenStreamState.initial(backbone).extend_prefix(layout["tokens"][: layout["prefix_end"]])
    for tick in layout["ticks"]:
        body = layout["tokens"][tick["start"] : tick["body_end"]]
        decisions = layout["tokens"][tick["body_end"] : tick["end"]]
        separate = state.clone().advance(body)
        expected = separate.branch_step(decisions)
        state, branches = state.advance_with_branches(body, decisions)
        assert relative_l2(branches, expected).max() <= BF16_REL_READOUT and relative_l2(state.hidden, separate.hidden).max() <= BF16_REL_MAX
        boundaries = [b - tick["start"] for qid in tick["candidate_boundaries"] for b in tick["candidate_boundaries"][qid] if b >= tick["start"]]
        if boundaries:
            assert relative_l2(state.hidden[boundaries], separate.hidden[boundaries]).max() <= BF16_REL_READOUT


def test_tiny_activation_checkpointing_reproduces_hidden_and_gradients(tiny):
    """층 단위 activation checkpointing(gradient가 켜진 dynamic 모드)은 hidden과 gradient를 바꾸지 않는다 (추론 경로는 건드리지 않는다)."""
    layout = synthetic_stream(random.Random(7), prefix=5, ticks=[(4, 2), (3, 2), (4, 2)], window_ticks=30)
    grads = {}
    hiddens = {}
    for checkpointing in (False, True):
        trainable = QwenBackbone.tiny(seed=1, vocab_size=1024, layers=("linear_attention", "full_attention"))
        trainable.model.requires_grad_(True)
        trainable.kv_mode = "dynamic"
        trainable.activation_checkpointing = checkpointing
        with torch_reference_kernels():
            hidden, state = incremental(trainable, layout)
            loss = state.branch_step([65, 66]).pow(2).sum()
            loss.backward()
        hiddens[checkpointing] = hidden.detach()
        grads[checkpointing] = {name: p.grad.detach().clone() for name, p in trainable.model.named_parameters() if p.grad is not None}
    torch.testing.assert_close(hiddens[True], hiddens[False], **CPU_TOL)
    assert grads[True].keys() == grads[False].keys() and grads[True]
    for name, grad in grads[False].items():
        torch.testing.assert_close(grads[True][name], grad, rtol=1e-4, atol=1e-5, msg=name)
    # 추론(gradient 없음)에서는 checkpointing flag가 결과에 아무 영향이 없다
    tiny.activation_checkpointing = True
    try:
        with torch.no_grad():
            hidden_flagged, _ = incremental(tiny, layout)
    finally:
        tiny.activation_checkpointing = False
    with torch.no_grad():
        hidden_plain, _ = incremental(tiny, layout)
    torch.testing.assert_close(hidden_flagged, hidden_plain, rtol=0, atol=0)


def test_tiny_official_forward_with_activation_checkpointing_reproduces_hidden_and_gradients():
    """P0 `state_first` 경로(공식 batched forward)도 checkpointing flag가 켜지면 HF 층 단위 checkpoint로 돌고 결과·gradient가 같다."""
    tokens = torch.tensor([[3, 5, 7, 11, 13, 17, 19, 23], [2, 4, 6, 8, 10, 12, 14, 16]])
    positions = torch.arange(8)[None].expand(2, 8)
    grads = {}
    hiddens = {}
    for checkpointing in (False, True):
        trainable = QwenBackbone.tiny(seed=1, vocab_size=1024, layers=("linear_attention", "full_attention"))
        trainable.model.requires_grad_(True)
        trainable.activation_checkpointing = checkpointing
        with torch_reference_kernels():
            hidden = trainable(tokens, positions)["hidden"]
            hidden.pow(2).sum().backward()
        hiddens[checkpointing] = hidden.detach()
        grads[checkpointing] = {name: p.grad.detach().clone() for name, p in trainable.model.named_parameters() if p.grad is not None}
        assert not trainable.text.training  # forward 뒤에는 eval 모드로 돌아온다
    torch.testing.assert_close(hiddens[True], hiddens[False], **CPU_TOL)
    assert grads[True].keys() == grads[False].keys() and grads[True]
    for name, grad in grads[False].items():
        torch.testing.assert_close(grads[True][name], grad, rtol=1e-4, atol=1e-5, msg=name)
