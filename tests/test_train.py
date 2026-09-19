"""학습 검사 — T0/T1 단계, truncated BPTT, 혼합 sampler의 유효 loss 비중, 일정·clip·누적, CLI (docs/06 Task 5, docs/03 §5).

CPU의 소형 hybrid fixture(4b)로 학습 step의 논리만 검증한다. 실제 backbone·GPU는 같은 코드 경로로
클라우드 단계에서 돈다. 스트림 검사는 D0의 10초 에피소드를 5초 구간 둘로 나눈다(브리프).
"""

import copy
import hashlib
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
    run_single_unit,
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
        "nonrobot_tokens_per_unit": 512,
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
        # step = 로봇 단위(에피소드, 1초 구간 2개) + 비로봇 단위(512 토큰까지 묶은 단일 요청들, forward 1번)
        assert [u["kind"] for u in metrics["units"]] == ["stream", "single"]
        assert metrics["items"]["stream"] == 1 and metrics["items"]["single"] >= 3 and metrics["chunks"] == 3
        assert metrics["units"][1]["tokens"] <= 512 and metrics["valid_states"]["single"] == metrics["items"]["single"]
        assert set(metrics["loss_by_type"]) <= {"choice", "boolean", "ordinal"}
        # 유효 loss 비중 0.6/0.4: 적용된 계수 질량이 정확히 그 값이고 step 손실 = 0.6·L_robot + 0.4·L_nonrobot
        share = metrics["loss_share"]["domain"]
        assert abs(share["robot"] - 0.6) < 1e-6 and abs(share["non_robot"] - 0.4) < 1e-6
        assert metrics["shares"] == {"robot": 0.6, "non_robot": 0.4}
        by_domain = metrics["loss_by_domain"]
        assert metrics["loss"] == pytest.approx(0.6 * by_domain["robot"] + 0.4 * by_domain["non_robot"], rel=1e-6)
        assert abs(sum(metrics["loss_share"]["tick_class"].values()) - 1.0) < 1e-9
        assert metrics["loss_share"]["material"]["existing"] == pytest.approx(1.0)
        contribution = metrics["loss_contribution"]["domain"]
        assert abs(sum(contribution.values()) - 1.0) < 1e-9 and contribution["robot"] != pytest.approx(0.6, abs=1e-3)
        assert metrics["tokens"]["robot"] > metrics["tokens"]["non_robot"] > 0
        assert metrics["tokens"]["total"] == metrics["tokens"]["robot"] + metrics["tokens"]["non_robot"]
        assert metrics["token_share"]["robot"] == pytest.approx(metrics["tokens"]["robot"] / metrics["tokens"]["total"])
        assert metrics["lr"]["backbone"] == pytest.approx(1e-5 * lr_factor(1, max_steps=1, warmup_ratio=0.5))
        assert metrics["sampler"]["token_share"]["robot"] > 0.6  # 토큰 비중은 기록만 (에피소드가 크다)


def test_every_step_mixes_robot_and_nonrobot_units(tmp_path):
    """docs/04 §2 (판정 e086c90): optimizer step마다 로봇 스트림 단위와 비로봇 단위가 둘 다 든다."""
    with Trainer(tiny_config(tmp_path, max_steps=3, gradient_accumulation=3, stream_max_ticks=4, stream_chunk_seconds=0.2)) as trainer:
        for _ in range(3):
            metrics = trainer.run_step()
            kinds = [u["kind"] for u in metrics["units"]]
            assert kinds == ["stream", "single", "stream"]
            assert {"stream", "single"} <= set(kinds)
            assert metrics["items"]["stream"] == 2 and metrics["items"]["single"] >= 3
            assert abs(metrics["loss_share"]["domain"]["robot"] - 0.6) < 1e-6
    with pytest.raises(ValueError, match="gradient_accumulation"):
        Trainer(tiny_config(tmp_path, gradient_accumulation=1))


def test_packed_nonrobot_unit_loss_equals_the_mean_of_singles_run_one_by_one(tmp_path):
    """비로봇 단위(한 forward에 묶은 단일 요청들)의 손실 = 상태를 하나씩 돌린 손실의 평균 (FP32)."""
    with Trainer(tiny_config(tmp_path, nonrobot_tokens_per_unit=600)) as trainer:
        unit = trainer.sampler.draw("non_robot")
        items = [trainer.items[index] for index in unit.items]
        assert len(items) >= 4 and unit.tokens <= 600
        packed = run_single_unit(trainer.model, items, scale=1.0 / len(items))
        assert packed.stats["valid_states"] == len(items) and packed.loss.requires_grad
        alone = [run_single_unit(trainer.model, [item], scale=1.0).value for item in items]
        assert packed.value == pytest.approx(sum(alone) / len(alone), rel=1e-5, abs=1e-6)
        for entry, value in zip(packed.stats["per_item"], alone):
            assert entry["loss"] == pytest.approx(value, rel=1e-5, abs=1e-6)


def test_low_confidence_labels_are_counted_separately_per_step(tmp_path):
    """퇴화 틱의 낮은 신뢰도 라벨(`label_confidence: low`, `weight` 0.25 — I4 완화)은 step 지표에 따로 센다:
    `labels_low_confidence` = 그 step의 상태(틱·단일 요청)에 실린 낮은 신뢰도 라벨 수."""
    with Trainer(tiny_config(tmp_path, max_steps=2)) as trainer:
        metrics = trainer.run_step()
        assert metrics["labels_low_confidence"] == 0  # D0에는 낮은 신뢰도 라벨이 없다
        assert metrics["labels_total"] > 0

        marked = 0
        for item in trainer.items:
            states = item.record["ticks"] if item.kind == "stream" else [item.record]
            for state in states:
                for label in state.get("labels", []):
                    if label["question_id"] in ("q_main", "q_target") and marked % 2 == 0:
                        label["label_confidence"] = "low"
                        label["weight"] = 0.25
                    marked += 1
        assert trainer.accumulate()
        expected = 0
        for unit in trainer.progress["units"]:
            for index in unit.items:
                item = trainer.items[index]
                states = item.record["ticks"] if item.kind == "stream" else [item.record]
                expected += sum(
                    1 for state in states for label in state.get("labels", []) if label.get("label_confidence") == "low"
                )
        assert expected > 0
        metrics = trainer.apply()
        assert metrics["labels_low_confidence"] == expected
        assert metrics["labels_total"] >= expected


def one_sided_manifest(tmp_path, name: str) -> str:
    """D0 manifest에서 파일 하나만 남긴 manifest (같은 sha256)."""
    manifest = json.loads(D0_MANIFEST.read_text(encoding="utf-8"))
    manifest["files"] = {name: manifest["files"][name]}
    (tmp_path / name).write_bytes(D0_MANIFEST.with_name(name).read_bytes())
    path = tmp_path / "one_sided.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return str(path)


def robot_batch(tmp_path, *, max_ticks: int) -> str:
    """전문가로 만든 로봇 에피소드 2편(E0 seed 17·E1 seed 29, 앞 `max_ticks`틱)의 manifest 경로. split은 전부 train."""
    from robo_jev.data.robot_episodes import build_manifest, generate_episode, load_generator_config, write_episode
    from robo_jev.sim.expert import Expert

    config = {**load_generator_config(), "split": {"weights": {"train": 1}, "holdout_prefixes": []}}
    expert = Expert()
    out = tmp_path / "d1-robot"
    for profile, seed in (("E0", 17), ("E1", 29)):
        write_episode(generate_episode(profile, seed, policy=expert, expert=expert, config=config, max_ticks=max_ticks), out)
    manifest = build_manifest(out, config, batch_wall_s=1.0)
    assert manifest["episodes"] == 2 and manifest["splits"] == {"train": 2} and isinstance(manifest["files"], dict)
    return str(out / "manifest.json")


def test_a_generated_robot_batch_and_the_d0_singles_train_together_from_two_manifests(tmp_path):
    """`dataset_manifests: [로봇 batch manifest, D0 단일 요청 manifest]` — 생성기가 쓴 manifest(files dict)를 적재기가
    그대로 읽고, manifest마다 분야를 달 수 있으며, step마다 로봇 에피소드와 비로봇 묶음이 든다."""
    robot = robot_batch(tmp_path, max_ticks=6)
    singles = one_sided_manifest(tmp_path, "d0.jsonl")
    config = tiny_config(
        tmp_path, max_steps=2, stream_max_ticks=4, stream_chunk_seconds=0.2, run_id="two-manifests",
        dataset_manifests=[robot, {"path": singles, "domain": "non_robot"}],
    )
    del config["dataset_manifest"]
    resolved = resolve_config(config)
    assert resolved["dataset_manifest"] is None
    assert resolved["dataset_manifests"] == [
        {"path": robot, "domain": None, "material": None}, {"path": singles, "domain": "non_robot", "material": None},
    ]
    # `dataset_manifest: x`는 `dataset_manifests: [x]`와 같은 run이다 (재개의 정체 비교).
    assert resolve_config(tiny_config(tmp_path, dataset_manifest=singles))["dataset_manifests"] == resolve_config(
        {**tiny_config(tmp_path, dataset_manifests=[singles]), "dataset_manifest": None}
    )["dataset_manifests"]
    with pytest.raises(ValueError, match="dataset_manifests"):
        resolve_config({**config, "dataset_manifests": [{"path": robot, "domain": "space"}]})
    with pytest.raises(ValueError, match="dataset_manifests"):
        resolve_config({**config, "dataset_manifests": []})

    with Trainer(config) as trainer:
        robot_items = [item for item in trainer.items if item.kind == "stream"]
        single_items = [item for item in trainer.items if item.kind == "single"]
        assert [item.record_id for item in robot_items] == ["ep-E0-000017", "ep-E1-000029"]
        assert all(item.domain == "robot" and item.manifest == robot for item in robot_items)
        assert len(single_items) == 32 and all(item.domain == "non_robot" and item.manifest == singles for item in single_items)
        assert [item.index for item in trainer.items] == list(range(len(trainer.items)))
        assert all(len(item.record["ticks"]) == 4 for item in robot_items)
        result = trainer.run()
    assert result["status"] == "completed" and result["step"] == 2
    for metrics in result["metrics"]["steps"]:
        assert [u["kind"] for u in metrics["units"]] == ["stream", "single"]
        assert metrics["units"][0]["records"][0].startswith("ep-E") and metrics["chunks"] == 3
        assert abs(metrics["loss_share"]["domain"]["robot"] - 0.6) < 1e-6 and math.isfinite(metrics["loss"])
    manifest = trainer.manifest
    assert [entry["path"] for entry in manifest["dataset_manifests"]] == [robot, singles]
    assert manifest["dataset_manifests"][0]["records"] == {"single": 0, "stream": 2}
    assert manifest["dataset_manifests"][1]["records"] == {"single": 32, "stream": 0}
    assert manifest["dataset_manifests"][1]["domain"] == "non_robot"
    assert all(len(entry["sha256"]) == 64 and entry["files"] for entry in manifest["dataset_manifests"])
    assert (tmp_path / "runs" / "two-manifests" / "checkpoint.pt").is_file()


def test_one_sided_data_gives_the_present_kind_the_whole_share(tmp_path):
    singles_only = tiny_config(tmp_path, dataset_manifest=one_sided_manifest(tmp_path, "d0.jsonl"), gradient_accumulation=1, max_steps=1)
    with Trainer(singles_only) as trainer:
        metrics = trainer.run_step()
        assert [u["kind"] for u in metrics["units"]] == ["single"] and metrics["items"]["stream"] == 0
        assert metrics["shares"] == {"robot": 0.0, "non_robot": 1.0}
        assert abs(metrics["loss_share"]["domain"]["non_robot"] - 1.0) < 1e-6 and metrics["loss_share"]["domain"]["robot"] == 0.0
        assert metrics["loss"] == pytest.approx(metrics["loss_by_domain"]["non_robot"], rel=1e-6)
        assert metrics["loss_by_domain"]["robot"] is None and metrics["loss_share"]["tick_class"] == {}


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
    scale = 1.0 / plan.normaliser  # 에피소드 하나의 정규화된 손실 Σ w_t L_t / Σ w_t
    first = run_stream_chunk(judge, item, plan.chunks[0], carried=None, plan=plan, scale=scale)
    first.state.delta[0]["recurrent"].retain_grad()
    carried = detach_stream_state(first.state, requires_grad=True)
    second = run_stream_chunk(judge, item, plan.chunks[1], carried=carried, plan=plan, scale=scale)
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
    plan = episode["plan"]
    whole = run_stream_chunk(judge, item, (0, 100), carried=None, plan=plan, scale=1.0 / plan.normaliser)
    parts = chunked["first"].value + chunked["second"].value
    assert chunked["first"].value > 0 and chunked["second"].value > 0
    torch.testing.assert_close(torch.tensor(parts), torch.tensor(whole.value), **FP32)
    assert whole.stats["ticks"] == 100 and chunked["first"].stats["ticks"] == chunked["second"].stats["ticks"] == 50
    # 정상 유지 틱은 하향, 이벤트·목표 변경 틱은 상향 가중된 기여가 기록된다
    by_class = whole.stats["loss_by_class"]
    assert set(by_class) <= {"steady", "event", "goal_change", "other"} and by_class["goal_change"] > 0
    assert abs(sum(by_class.values()) - whole.value) < 1e-6
    assert whole.stats["weight_sum"] == pytest.approx(plan.normaliser) and whole.stats["loss_sum"] == pytest.approx(whole.value * plan.normaliser, rel=1e-6)
    assert sum(whole.stats["weight_by_class"].values()) == pytest.approx(1.0)  # 적용된 계수 질량 (scale = 1/Σw)


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
        ("microbatch_states_per_rank", 2, "microbatch_states_per_rank"),
        ("robot_loss_share", 1.5, "robot_loss_share"),
        ("nonrobot_tokens_per_unit", 0, "nonrobot_tokens_per_unit"),
        ("warmup_ratio", 1.5, "warmup_ratio"),
        ("layout", {"single": "stream_l1a"}, "layout"),
        ("unknown_key", 1, "unknown_key"),
    ):
        with pytest.raises(ValueError, match=message):
            resolve_config({**base, key: value})
    with pytest.raises(ValueError, match="dataset_manifest"):
        resolve_config({k: v for k, v in base.items() if k != "dataset_manifest"})


def test_model_id_other_than_the_fixture_is_rejected_and_the_manifest_records_what_was_built(tmp_path):
    """리뷰 11 S2: fixture 전용 경로는 `tiny_hybrid` 이외의 model_id를 설정 단계에서 거절한다(실모델 adapter는 아직
    없다 — 잘못된 설정이 성공처럼 보이면 안 된다). manifest의 `model` 블록은 요청한 id가 아니라 **실제로 만든 것**
    (종류·설정 파일 해시·파라미터 수·dtype·장치)을 적는다."""
    with pytest.raises(ValueError, match="model_id") as excinfo:
        resolve_config(tiny_config(tmp_path, model_id="Qwen/Qwen3.5-9B"))
    assert "adapter" in str(excinfo.value) and "tiny_hybrid" in str(excinfo.value)
    shipped = yaml.safe_load((REPO / "configs" / "train" / "tiny_cpu.yaml").read_text(encoding="utf-8"))
    assert shipped["model_id"] == "tiny_hybrid"
    with pytest.raises(ValueError, match="model_id"):  # 리뷰의 재현 그대로: 배포 설정에서 model_id만 바꾼다
        resolve_config({**shipped, "model_id": "Qwen/Qwen3.5-9B"})

    with Trainer(tiny_config(tmp_path, max_steps=1)) as trainer:
        model = trainer.manifest["model"]
        assert model["kind"] == "tiny_hybrid" and model["id"] == "tiny_hybrid" and model["class"] == "TinyHybrid"
        assert model["parameters"] == sum(p.numel() for p in trainer.model.parameters()) == 376_745
        assert model["trainable_parameters"] == sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
        assert model["dtype"] == "float32" and model["device"] == "cpu"
        assert model["config"] == str(REPO / "configs" / "model" / "tiny_hybrid.yaml")
        assert model["config_sha256"] == hashlib.sha256((REPO / "configs" / "model" / "tiny_hybrid.yaml").read_bytes()).hexdigest()
        assert model["name"] == "tiny-hybrid-v0" and model["vocab_size"] == SMALL_VOCAB and model["readout"] == "pointer"
        assert trainer.manifest["identity"]["model"]["parameters"] == 376_745
    with Trainer(tiny_config(tmp_path, max_steps=1, trainable="readout_only")) as frozen:
        assert frozen.manifest["model"]["parameters"] == 376_745
        assert frozen.manifest["model"]["trainable_parameters"] == sum(
            p.numel() for name, p in frozen.model.named_parameters() if not name.startswith("backbone.")
        )


# --------------------------------------------------------------------------
# CLI·import 경계
# --------------------------------------------------------------------------


def test_cli_runs_the_shipped_config_for_three_steps_and_writes_checkpoint_and_metrics(tmp_path):
    """`python -m robo_jev.train --config configs/train/tiny_cpu.yaml` (max_steps 3; 검사 시간 때문에 에피소드 20틱·묶음 512 토큰)."""
    shipped = REPO / "configs" / "train" / "tiny_cpu.yaml"
    loaded = yaml.safe_load(shipped.read_text(encoding="utf-8"))
    assert loaded["stream_chunk_seconds"] == 5 and loaded["stream_window_ticks"] == 30 and loaded["stream_max_ticks"] is None
    assert loaded["trainable"] == "text_backbone_and_readout" and loaded["execution_backend"] == "independent_paths"
    assert loaded["robot_loss_share"] == 0.6 and loaded["nonrobot_tokens_per_unit"] == 8192 and loaded["gradient_accumulation"] == 2
    result = subprocess.run(
        [
            sys.executable, "-m", "robo_jev.train", "--config", str(shipped),
            "--set", "max_steps=3", "--set", "run_id=cli-3", "--set", f"artifacts_dir={tmp_path / 'runs'}",
            "--set", "stream_max_ticks=20", "--set", "stream_chunk_seconds=1", "--set", "checkpoint_every=null",
            "--set", "nonrobot_tokens_per_unit=512",
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
    assert [u["kind"] for u in metrics["steps"][0]["units"]] == ["stream", "single"] and metrics["steps"][0]["chunks"] == 3
    assert all(abs(m["loss_share"]["domain"]["robot"] - 0.6) < 1e-6 for m in metrics["steps"])
    assert metrics["manifest"]["serializer_version"] == "ts0.3" and metrics["manifest"]["question_set"]["id"] == "qs-v0"
    assert len(metrics["manifest"]["dataset_manifests"]) == 1 and metrics["manifest"]["dataset_manifests"][0]["sha256"]
    assert metrics["manifest"]["git"]["sha"]
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


def test_trainer_applies_torch_threads_only_while_it_computes_and_never_leaks_it(tmp_path, monkeypatch):
    """`torch_threads`는 run의 정체(재개의 비트 동일 조건)지만 프로세스 전역 상태다. Trainer는 그 값을 자기 계산
    (구성·accumulate·apply·run) 안에서만 걸고 나올 때 되돌린다 — context manager 없이 만들어도, 생성이 실패해도 새지 않는다."""
    import robo_jev.train as train_module

    outside = torch.get_num_threads()
    inside = 1 if outside != 1 else 2
    seen: list[int] = []
    original = train_module.run_single_unit

    def recording(*args, **kwargs):
        seen.append(torch.get_num_threads())
        return original(*args, **kwargs)

    monkeypatch.setattr(train_module, "run_single_unit", recording)
    trainer = Trainer(tiny_config(tmp_path, max_steps=1, torch_threads=inside))  # `with` 없이
    assert torch.get_num_threads() == outside
    metrics = trainer.run_step()
    assert metrics["step"] == 1 and seen and set(seen) == {inside}
    assert torch.get_num_threads() == outside
    trainer.close()
    assert torch.get_num_threads() == outside
    with pytest.raises(ValueError, match="gradient_accumulation"):
        Trainer(tiny_config(tmp_path, gradient_accumulation=1, torch_threads=inside))
    assert torch.get_num_threads() == outside
    with Trainer(tiny_config(tmp_path, max_steps=1, torch_threads=inside)) as scoped:
        assert torch.get_num_threads() == outside  # 계산 밖에서는 바깥 값 그대로
        scoped.run()
    assert torch.get_num_threads() == outside and set(seen) == {inside}


def test_train_function_returns_the_documented_result(tmp_path):
    threads_before = torch.get_num_threads()
    result = train(tiny_config(tmp_path, max_steps=1, run_id="fn-1"))
    assert result["run_id"] == "fn-1" and result["step"] == 1 and result["status"] == "completed"
    assert (tmp_path / "runs" / "fn-1" / "checkpoint.pt").is_file()
    assert result["checkpoint"].endswith("checkpoint.pt") and len(result["metrics"]["steps"]) == 1
    assert torch.get_num_threads() == threads_before  # 스레드 수는 train()이 끝나면 되돌린다
