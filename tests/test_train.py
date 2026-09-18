"""학습 검사 — T0/T1 단계, truncated BPTT, 혼합 sampler의 유효 loss 비중, 일정·clip·누적, CLI (docs/06 Task 5, docs/03 §5).

CPU의 소형 hybrid fixture(4b)로 학습 step의 논리만 검증한다. 실제 backbone·GPU는 같은 코드 경로로
클라우드 단계에서 돈다. 스트림 검사는 D0의 10초 에피소드를 5초 구간 둘로 나눈다(브리프).
"""

import copy
import json
import math
import subprocess
import sys

import pytest
import torch
import yaml
from helpers import D0_MANIFEST, REPO, SMALL_VOCAB

from robo_jev.loss import label_loss
from robo_jev.model.judge import Judge
from robo_jev.model.stream import StreamState, replay_layout
from robo_jev.model.tokenizer import WhitespaceTokenizer
from robo_jev.sampler import load_items, tick_weights, valid_label_ticks
from robo_jev.train import (
    Trainer,
    detach_stream_state,
    episode_chunks,
    layout_prefix,
    lr_factor,
    plan_episode,
    resolve_config,
    run_stream_chunk,
    train,
)

TICK_WEIGHTS = {"steady": 0.25, "event": 2.0, "goal_change": 2.0, "other": 1.0}
FP32 = {"rtol": 1e-5, "atol": 1e-6}


def tiny_config(tmp_path, **overrides) -> dict:
    """소형 fixture의 학습 설정 — 검사용으로 에피소드를 20틱(2초)으로 자르고 1초 구간(2구간)을 쓴다."""
    config = {
        "run_name": "test",
        "run_id": "run-under-test",
        "model_config": str(REPO / "configs" / "model" / "tiny_hybrid.yaml"),
        "model_vocab_size": SMALL_VOCAB,
        "dataset_manifest": str(D0_MANIFEST),
        "splits": ["train"],
        "tokenizer": "whitespace",
        "stream_chunk_seconds": 1,
        "stream_window_ticks": 30,
        "stream_max_ticks": 20,
        "trainable": "text_backbone_and_readout",
        "gradient_accumulation": 2,
        "microbatch_states_per_rank": 1,
        "max_steps": 2,
        "warmup_ratio": 0.5,
        "seed": 17,
        "torch_threads": 1,
        "artifacts_dir": str(tmp_path / "runs"),
        "sampler": {"tick_weights": TICK_WEIGHTS},
    }
    config.update(overrides)
    return config


def snapshot(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {name: p.detach().clone() for name, p in module.named_parameters()}


# --------------------------------------------------------------------------
# T0 / T1 (docs/06 Task 5 bullets 1–2)
# --------------------------------------------------------------------------


def test_readout_only_step_changes_the_readout_and_keeps_every_backbone_tensor_bit_identical(tmp_path):
    with Trainer(tiny_config(tmp_path, trainable="readout_only", max_steps=1)) as trainer:
        before = snapshot(trainer.model)
        assert trainer.accumulate()
        backbone = [name for name, _ in trainer.model.named_parameters() if name.startswith("backbone.")]
        assert backbone and all(
            p.grad is None for name, p in trainer.model.named_parameters() if name.startswith("backbone.")
        )
        readout = [name for name, _ in trainer.model.named_parameters() if not name.startswith("backbone.")]
        assert set(readout) == {"U.weight", "V.weight", "bias"}
        assert all(dict(trainer.model.named_parameters())[name].grad is not None for name in readout)
        metrics = trainer.apply()
        after = snapshot(trainer.model)
        for name in backbone:
            assert torch.equal(before[name], after[name]), name
        for name in readout:
            assert not torch.equal(before[name], after[name]), name
        assert metrics["step"] == 1 and math.isfinite(metrics["loss"]) and metrics["grad_norm"] > 0
        assert set(trainer.optimizer.param_groups[0]["params"]) <= set(
            dict(trainer.model.named_parameters())[name] for name in readout
        )


def test_full_training_step_gives_every_backbone_tensor_gradient_and_changes_it(tmp_path):
    with Trainer(tiny_config(tmp_path, max_steps=1)) as trainer:
        before = snapshot(trainer.model)
        assert trainer.accumulate()
        for name, p in trainer.model.named_parameters():
            assert p.grad is not None and torch.isfinite(p.grad).all(), name
            assert p.grad.abs().sum() > 0, name  # loss만 줄고 backbone 업데이트가 빠지는 오류를 막는다
        metrics = trainer.apply()
        after = snapshot(trainer.model)
        for name in before:
            assert not torch.equal(before[name], after[name]), name
        assert metrics["items"] == {"single": 1, "stream": 1}  # 첫 단위는 에피소드(토큰 비중 0 < 0.6), 그다음 단일 요청
        assert metrics["chunks"] == 3  # 20틱 에피소드 = 1초 구간 2개 + 단일 요청 1
        assert set(metrics["loss_by_type"]) <= {"choice", "boolean", "ordinal"}
        assert abs(sum(metrics["loss_share"]["domain"].values()) - 1.0) < 1e-9
        assert abs(sum(metrics["loss_share"]["tick_class"].values()) - 1.0) < 1e-9
        assert metrics["loss_share"]["material"]["existing"] == 1.0
        assert metrics["tokens"]["robot"] > metrics["tokens"]["non_robot"] > 0
        assert metrics["tokens"]["total"] == metrics["tokens"]["robot"] + metrics["tokens"]["non_robot"]
        assert metrics["lr"]["backbone"] == pytest.approx(1e-5 * lr_factor(1, max_steps=1, warmup_ratio=0.5))
        assert metrics["sampler"]["token_share"]["robot"] > 0.6


# --------------------------------------------------------------------------
# truncated BPTT (docs/06 Task 5 bullet 3, docs/08 §8 "학습 시퀀스")
# --------------------------------------------------------------------------


def test_episode_chunks_follow_the_simulated_time(streams):
    record = streams[0]
    assert episode_chunks(record, 5) == [(0, 50), (50, 100)]
    assert episode_chunks(record, 10) == [(0, 100)]
    assert episode_chunks(record, 100) == [(0, 100)]
    assert episode_chunks(record, None) == [(0, 100)]
    irregular = copy.deepcopy(record)
    irregular["ticks"] = irregular["ticks"][:6]
    for tick, ms in zip(irregular["ticks"], (300, 400, 1200, 1300, 2300, 2350)):
        tick["sim_ms"] = ms
    assert episode_chunks(irregular, 1) == [(0, 3), (3, 4), (4, 6)]  # 구간은 첫 틱의 모의 시각 기준
    with pytest.raises(ValueError, match="stream_chunk_seconds"):
        episode_chunks(record, 0)


@pytest.fixture(scope="module")
def single_thread():
    before = torch.get_num_threads()
    torch.set_num_threads(1)  # 기준 fixture의 grouped conv1d는 스레드 동기화 비용이 크다 (보고서)
    yield
    torch.set_num_threads(before)


@pytest.fixture(scope="module")
def episode(single_thread):
    """D0의 10초 train 에피소드(100틱) 하나와 5초 구간 둘의 계획(틱 종류·가중치·유효 라벨·정규화 분모)."""
    items = load_items(D0_MANIFEST, tokenizer=WhitespaceTokenizer(), splits=("train",))
    item = next(i for i in items if i.kind == "stream")
    plan = plan_episode(item.record, chunk_seconds=5, tick_weights=TICK_WEIGHTS)
    assert plan.chunks == [(0, 50), (50, 100)] and plan.normaliser > 0
    assert plan.weights == tick_weights(item.record, weights=TICK_WEIGHTS) and plan.valid == valid_label_ticks(item.record)
    judge = Judge.from_config(seed=5, vocab_size=SMALL_VOCAB)
    return {"item": item, "plan": plan, "judge": judge}


@pytest.fixture(scope="module")
def chunked(episode):
    """구간 0을 돌리고, 경계 상태를 detach(+ leaf로 gradient 관찰)해 구간 1을 이어 돌린 결과."""
    judge, item, plan = episode["judge"], episode["item"], episode["plan"]
    first = run_stream_chunk(judge, item, plan.chunks[0], carried=None, plan=plan)
    first.state.delta[0]["recurrent"].retain_grad()
    carried = detach_stream_state(first.state, requires_grad=True)
    second = run_stream_chunk(judge, item, plan.chunks[1], carried=carried, plan=plan)
    return {"first": first, "carried": carried, "second": second}


def test_chunk_boundary_state_equals_the_incremental_inference_state(episode, chunked):
    """(a) 구간 경계의 공통 상태 == 추론의 증분 계산(prefix → 틱마다 advance)이 같은 틱에서 가진 상태."""
    item, judge = episode["item"], episode["judge"]
    layout = item.layout
    boundary = chunked["first"].state
    assert boundary.tick == 49 and not boundary.is_branch
    prefix = layout["tokens"][: layout["prefix_end"]]
    bodies = [layout["tokens"][t["start"]: t["body_end"]] for t in layout["ticks"][:50]]
    inference = StreamState.from_tokens(prefix, bodies, backbone=judge.backbone, window_ticks=layout["window_ticks"])
    assert inference.tick == 49 and inference.position == boundary.position
    for a, b in zip(boundary.delta, inference.delta):
        assert torch.equal(a["recurrent"], b["recurrent"]) and torch.equal(a["conv"], b["conv"])
    for a, b in zip(boundary.kv, inference.kv):
        assert torch.equal(a["k"], b["k"]) and torch.equal(a["v"], b["v"])
    assert torch.equal(boundary.cache_ticks, inference.cache_ticks)
    assert torch.equal(boundary.prefix_hidden, inference.prefix_hidden)
    # replay_layout(구간 0의 layout)의 마지막 공통 상태와도 같다
    replay = replay_layout(layout_prefix(layout, 50), backbone=judge.backbone)["final"]
    assert torch.equal(replay.recurrent[0], boundary.recurrent[0]) and replay.tick == 49
    # 구간 1을 이어 붙인 뒤의 상태는 틱 99까지 읽은 공통 상태다 (값의 연속성은 (c)가 손실로 확인)
    assert chunked["second"].state.tick == 99 and chunked["second"].state.position > boundary.position


def test_branch_gradient_merges_into_the_carried_state_and_stops_at_the_detach_boundary(episode, chunked):
    """(b) 구간 2의 결정 손실 → 구간 2 토큰과 넘겨받은 공통 상태(leaf)에는 gradient가 있고, detach 너머(구간 1의 그래프)에는 없다."""
    item, judge = episode["item"], episode["judge"]
    layout, record = item.layout, item.record
    first, carried, second = chunked["first"], chunked["carried"], chunked["second"]
    judge.zero_grad(set_to_none=True)
    tick = 60
    offset = tick - 50
    labels = {l["question_id"]: l for l in record["ticks"][tick]["labels"]}
    loss = sum(  # 동적 후보의 q_main + 정적 후보(prefix)의 q_stop — 두 결정 분기의 손실
        label_loss(second.outputs["logits"][offset][qid], second.outputs["candidates"][offset][qid], labels[qid])
        for qid in ("q_main", "q_stop")
    )
    assert loss.requires_grad
    loss.backward()
    # 분기(결정 토큰)의 gradient가 공통 상태로 합쳐져 경계까지 온다
    for layer in carried.delta:
        assert layer["recurrent"].grad is not None and layer["recurrent"].grad.norm() > 0
        assert layer["conv"].grad is not None and layer["conv"].grad.norm() > 0
    for layer in carried.kv:
        rows = layer["k"].grad[0].norm(dim=(-1, -2))  # cache 항목별
        ticks = carried.cache_ticks
        assert (rows[ticks == -1] > 0).all()  # prefix KV는 언제나 보인다
        assert (rows[(ticks >= 0) & (tick - ticks >= 30)] == 0).all()  # 윈도우 밖의 틱은 gradient 0
        assert (rows[(ticks >= 0) & (tick - ticks < 30)] > 0).all()  # 윈도우 안의 틱은 gradient를 받는다
    # 정적 후보의 h_c는 prefix hidden — q_stop의 true/false 경계 행만 gradient를 받는다
    static_rows = layout["static_candidate_boundaries"]["q_stop"]
    prefix_rows = carried.prefix_hidden.grad.norm(dim=-1)
    assert (prefix_rows[static_rows] > 0).all() and prefix_rows.count_nonzero() == len(static_rows)
    # detach 너머: 구간 1의 그래프에는 아무것도 흐르지 않는다
    assert first.state.delta[0]["recurrent"].grad is None
    # 구간 2 안: 결정이 있는 틱의 토큰은 gradient를 받고, 구간 1에만 있는 토큰·결정 뒤 틱의 토큰은 0
    embed_grad = judge.backbone.embed.weight.grad
    tokens_of = lambda ticks: {layout["tokens"][i] for t in ticks for i in range(t["start"], t["end"])}  # noqa: E731
    prefix_tokens = set(layout["tokens"][: layout["prefix_end"]])
    only_first = tokens_of(layout["ticks"][:50]) - tokens_of(layout["ticks"][50:]) - prefix_tokens
    only_later = tokens_of(layout["ticks"][61:]) - tokens_of(layout["ticks"][50:61]) - prefix_tokens
    body = {layout["tokens"][i] for i in range(layout["ticks"][tick]["start"], layout["ticks"][tick]["body_end"])}
    decisions = layout["ticks"][tick]["decision_positions"]
    unrelated_markers = {layout["tokens"][index] for qid, index in decisions.items() if qid not in ("q_main", "q_stop")}
    assert only_first and only_later and body and len(unrelated_markers) == len(decisions) - 2
    assert (embed_grad[sorted(only_first)].abs().sum(dim=-1) == 0).all()
    assert (embed_grad[sorted(only_later)].abs().sum(dim=-1) == 0).all()
    assert (embed_grad[sorted(body)].norm(dim=-1) > 0).all()  # 틱 몸통 = 분기 이전 공통 상태의 토큰
    assert (embed_grad[sorted(unrelated_markers)].abs().sum(dim=-1) == 0).all()  # 다른 분기의 결정 토큰은 격리
    assert (embed_grad[[layout["tokens"][decisions["q_main"]], layout["tokens"][decisions["q_stop"]]]].norm(dim=-1) > 0).all()


def test_sum_of_chunk_losses_equals_the_whole_episode_loss(episode, chunked):
    """(c) 구간별 손실(에피소드 분모로 정규화)의 합 == 한 구간으로 돌린 에피소드 전체 손실 (FP32)."""
    judge, item = episode["judge"], episode["item"]
    whole = run_stream_chunk(judge, item, (0, 100), carried=None, plan=episode["plan"])
    parts = chunked["first"].value + chunked["second"].value
    assert chunked["first"].value > 0 and chunked["second"].value > 0
    torch.testing.assert_close(torch.tensor(parts), torch.tensor(whole.value), **FP32)
    assert whole.stats["ticks"] == 100 and chunked["first"].stats["ticks"] == chunked["second"].stats["ticks"] == 50
    # 정상 유지 틱은 하향, 이벤트·목표 변경 틱은 상향 가중된 기여가 기록된다
    by_class = whole.stats["loss_by_class"]
    assert set(by_class) <= {"steady", "event", "goal_change", "other"} and by_class["goal_change"] > 0
    assert abs(sum(by_class.values()) - whole.value) < 1e-6


def test_layout_prefix_keeps_the_leading_ticks_and_the_prefix(episode):
    layout = episode["item"].layout
    view = layout_prefix(layout, 50)
    assert len(view["ticks"]) == 50 and len(view["tokens"]) == layout["ticks"][49]["end"]
    assert view["prefix_end"] == layout["prefix_end"] and view["tokens"] == layout["tokens"][: len(view["tokens"])]
    assert all(len(view[field]) == len(view["tokens"]) for field in ("kind", "question", "candidate", "position", "tick"))
    assert layout_prefix(layout, 100) is layout
    with pytest.raises(ValueError, match="end_tick"):
        layout_prefix(layout, 0)


# --------------------------------------------------------------------------
# 일정·설정
# --------------------------------------------------------------------------


def test_lr_factor_warms_up_linearly_then_decays_by_cosine():
    factors = [lr_factor(s, max_steps=20, warmup_ratio=0.05) for s in range(20)]
    assert factors[0] == 1.0  # warmup 1 step: 첫 step부터 base lr
    factors = [lr_factor(s, max_steps=100, warmup_ratio=0.05) for s in range(100)]
    assert factors[:5] == pytest.approx([0.2, 0.4, 0.6, 0.8, 1.0])
    assert factors[5] == 1.0 and all(a > b for a, b in zip(factors[5:], factors[6:]))  # 그 뒤 cosine 단조 감소
    assert factors[-1] > 0 and factors[52] == pytest.approx(0.5 * (1 + math.cos(math.pi * 47 / 95)))
    assert lr_factor(0, max_steps=10, warmup_ratio=0.0) == 1.0


def test_config_rejects_what_the_cpu_path_does_not_implement(tmp_path):
    base = tiny_config(tmp_path)
    resolved = resolve_config(base)
    assert resolved["layout"] == {"single": "state_first", "stream": "stream_l1a"}
    assert resolved["checkpoint_every"] == resolved["max_steps"] == 2
    for key, value, message in (
        ("execution_backend", "shared_hybrid", "execution_backend"),
        ("readout", "lm_head", "readout"),
        ("activation_checkpointing", True, "activation_checkpointing"),
        ("world_size", 8, "world_size"),
        ("trainable", "vision_and_readout", "trainable"),
        ("optimizer", "sgd", "optimizer"),
        ("dtype", "bfloat16", "dtype"),
        ("gradient_accumulation", 0, "gradient_accumulation"),
        ("warmup_ratio", 1.5, "warmup_ratio"),
        ("layout", {"single": "stream_l1a"}, "layout"),
        ("unknown_key", 1, "unknown_key"),
    ):
        with pytest.raises(ValueError, match=message):
            resolve_config({**base, key: value})
    with pytest.raises(ValueError, match="dataset_manifest"):
        resolve_config({k: v for k, v in base.items() if k != "dataset_manifest"})


# --------------------------------------------------------------------------
# CLI·import 경계
# --------------------------------------------------------------------------


def test_cli_runs_the_shipped_config_for_three_steps_and_writes_checkpoint_and_metrics(tmp_path):
    """`python -m robo_jev.train --config configs/train/tiny_cpu.yaml` (max_steps 3; 검사 시간 때문에 에피소드는 20틱)."""
    shipped = REPO / "configs" / "train" / "tiny_cpu.yaml"
    loaded = yaml.safe_load(shipped.read_text(encoding="utf-8"))
    assert loaded["max_steps"] == 30 and loaded["stream_chunk_seconds"] == 5 and loaded["stream_window_ticks"] == 30
    assert loaded["trainable"] == "text_backbone_and_readout" and loaded["execution_backend"] == "independent_paths"
    result = subprocess.run(
        [
            sys.executable, "-m", "robo_jev.train", "--config", str(shipped),
            "--set", "max_steps=3", "--set", "run_id=cli-3", "--set", f"artifacts_dir={tmp_path / 'runs'}",
            "--set", "stream_max_ticks=20", "--set", "stream_chunk_seconds=1", "--set", "checkpoint_every=null",
        ],  # fmt: skip
        capture_output=True, text=True, cwd=REPO,
    )
    assert result.returncode == 0, result.stderr
    summary = json.loads(result.stdout.strip().splitlines()[-1])
    run_dir = tmp_path / "runs" / "cli-3"
    assert summary["run_id"] == "cli-3" and summary["step"] == 3 and summary["status"] == "completed"
    assert summary["checkpoint"] == str(run_dir / "checkpoint.pt") and math.isfinite(summary["loss"])
    assert (run_dir / "checkpoint.pt").is_file() and (run_dir / "metrics.json").is_file()
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    assert [m["step"] for m in metrics["steps"]] == [1, 2, 3] and metrics["status"] == "completed"
    assert metrics["steps"][0]["items"] == {"single": 3, "stream": 1} and metrics["steps"][0]["chunks"] == 5
    assert metrics["manifest"]["serializer_version"] == "s0.2" and metrics["manifest"]["question_set"]["id"] == "qs-v0"
    assert metrics["manifest"]["dataset_manifest"]["sha256"] and metrics["manifest"]["git"]["sha"]
    assert metrics["config"]["max_steps"] == 3 and metrics["config"]["checkpoint_every"] == 3
    assert (run_dir / "config.yaml").is_file()


def test_training_code_does_not_import_generator_simulator_or_harness():
    script = (
        "import sys, robo_jev.train, robo_jev.checkpoint, robo_jev.sampler;"
        "print(sorted(n for n in sys.modules if n.startswith('robo_jev.')))"
    )
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, check=True, cwd=REPO)
    loaded = result.stdout.strip()
    assert "robo_jev.train" in loaded
    for forbidden in ("robo_jev.sim", "robo_jev.harness", "robo_jev.data", "robo_jev.perception"):
        assert forbidden not in loaded, loaded


def test_train_function_returns_the_documented_result(tmp_path):
    threads_before = torch.get_num_threads()
    result = train(tiny_config(tmp_path, max_steps=1, run_id="fn-1"))
    assert result["run_id"] == "fn-1" and result["step"] == 1 and result["status"] == "completed"
    assert (tmp_path / "runs" / "fn-1" / "checkpoint.pt").is_file()
    assert result["checkpoint"].endswith("checkpoint.pt") and len(result["metrics"]["steps"]) == 1
    assert torch.get_num_threads() == threads_before  # 스레드 수는 train()이 끝나면 되돌린다
