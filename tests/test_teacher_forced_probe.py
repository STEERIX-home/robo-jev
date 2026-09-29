"""`scripts/teacher_forced_probe.py` (Task R8 A3)의 순수 부분 — CPU, 소형 fixture.

진단은 같은 에피소드를 **학습 계산**(구간마다 `run_stream_chunk`, 동적 상태, 구간 경계 detach)과 **평가 계산**(`predict_items` 전체 재생)으로 돌려
두 값이 갈리는지 본다. 그 비교가 뜻을 가지려면 도구 자체가 두 계산에서 같은 모델에 같은 표를 내야 한다 — 소형 모델(float64)에서 그것을 고정한다.
"""

import functools
import importlib.util
import sys

import pytest
import torch
from helpers import D0_MANIFEST, REPO, SMALL_VOCAB

from robo_jev.evaluate import predict_items
from robo_jev.model.judge import Judge
from robo_jev.model.tokenizer import WhitespaceTokenizer
from robo_jev.sampler import load_items
from robo_jev.train import plan_episode

SCRIPT = REPO / "scripts" / "teacher_forced_probe.py"
TICK_WEIGHTS = {"steady": 0.25, "event": 2.0, "goal_change": 2.0, "other": 1.0}


@functools.lru_cache(maxsize=1)
def script():
    spec = importlib.util.spec_from_file_location("teacher_forced_probe", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_the_episodes_are_the_last_distinct_expert_episodes_the_run_drew():
    """학습 마지막 무렵 뽑힌 expert 편 — 끝 step부터 거슬러 처음 만나는 순서, 편마다 마지막으로 뽑힌 step; DAgger·비로봇 단위는 건너뛴다."""
    metrics = {"steps": [
        {"step": 1, "units": [{"kind": "stream", "materials": ["existing"], "records": ["ep-a"]}, {"kind": "single", "materials": ["existing"], "records": ["x"]}]},
        {"step": 2, "units": [{"kind": "stream", "materials": ["error_family"], "records": ["ep-dagger"]}]},
        {"step": 3, "units": [{"kind": "stream", "materials": ["existing"], "records": ["ep-b"]}]},
        {"step": 4, "units": [{"kind": "stream", "materials": ["existing"], "records": ["ep-a"]}]},
    ]}  # fmt: skip
    module = script()
    assert module.last_drawn_episodes(metrics, count=2) == [{"record_id": "ep-a", "last_step": 4}, {"record_id": "ep-b", "last_step": 3}]
    assert module.last_drawn_episodes(metrics, count=5) == [{"record_id": "ep-a", "last_step": 4}, {"record_id": "ep-b", "last_step": 3}]
    assert module.drawn_steps(metrics, ["ep-a", "ep-dagger", "ep-none"]) == {"ep-a": [1, 4], "ep-dagger": [2], "ep-none": []}


def test_paths_a_finished_run_wrote_under_a_removed_worktree_are_rehomed_into_this_checkout():
    module = script()
    old = "/home/u/robo-jev/.claude/worktrees/task-r7/artifacts/datasets/r1-robot/r1-rollout-labels-g2/manifest.json"
    assert module.rehome(old) == str(REPO / "artifacts/datasets/r1-robot/r1-rollout-labels-g2/manifest.json")
    assert module.rehome("/x/worktrees/task-r5/configs/model/tiny_hybrid.yaml") == str(REPO / "configs/model/tiny_hybrid.yaml")
    assert module.rehome("Qwen/Qwen3.8-27B") == "Qwen/Qwen3.8-27B"
    config = module.rehomed_config({"dataset_manifests": [{"path": old, "domain": "robot"}], "model_config": "/a/configs/m.yaml", "artifacts_dir": "/a/artifacts/runs", "resume": "x"})
    assert config["dataset_manifests"][0] == {"path": str(REPO / "artifacts/datasets/r1-robot/r1-rollout-labels-g2/manifest.json"), "domain": "robot"}
    assert config["artifacts_dir"] == str(REPO / "artifacts/runs") and config["resume"] is None


def test_the_training_and_evaluation_computations_give_the_same_question_table_and_probes_on_one_model():
    """같은 모델·같은 에피소드: 학습 계산(1초 구간으로 나눈 `run_stream_chunk`)과 평가 계산(`predict_items` 전체 재생)이 같은 질문별 가중
    손실·기준선·탐침을 낸다(float64). 그래서 실제 모델에서 두 계산이 갈리면 그것은 도구가 아니라 계산 경로의 차이다."""
    module = script()
    items = [item for item in load_items(D0_MANIFEST, tokenizer=WhitespaceTokenizer(), splits=("train",), stream_max_ticks=30) if item.kind == "stream"]
    items = {item.record_id: item for item in items}
    plans = {record_id: plan_episode(item.record, chunk_seconds=1, tick_weights=TICK_WEIGHTS) for record_id, item in items.items()}
    assert all(len(plan.chunks) >= 3 for plan in plans.values())
    judge = Judge.from_config(seed=5, vocab_size=SMALL_VOCAB).to(torch.float64)
    judge.eval()
    trained, trained_probes, rows = module._training_path(judge, items, plans)
    predictions = predict_items(judge, list(items.values()))
    evaluated, evaluated_probes = module.prediction_records(predictions, items, plans)
    a, b = module.summarize(trained, trained_probes), module.summarize(evaluated, evaluated_probes)
    assert set(a["per_episode"]) == set(b["per_episode"]) == set(items)
    for record_id, table in a["per_episode"].items():
        assert set(table) == set(b["per_episode"][record_id])
        for key, entry in table.items():
            other = b["per_episode"][record_id][key]
            assert entry["n"] == other["n"] and entry["mass"] == pytest.approx(other["mass"], rel=1e-12)
            assert entry["loss"] == pytest.approx(other["loss"], rel=1e-7), (record_id, key)
            assert entry["baseline"] == pytest.approx(other["baseline"], rel=1e-12), (record_id, key)
    assert a["probes"] == b["probes"] and a["probes"]["q_gripper"]
    agreement = module.argmax_agreement(rows, predictions, "q_gripper")
    assert agreement["ticks"] == sum(len(item.record["ticks"]) for item in items.values()) and agreement["rate"] == 1.0
    stats = a["monitor_statistic"]["q_main"]
    per = [table["q_main"] for table in a["per_episode"].values()]
    assert stats["episodes"] == len(per) and stats["loss_mean"] == pytest.approx(sum(e["loss"] for e in per) / len(per))
