"""재개 검사 — 계획서의 핵심 인수 검사 (docs/06 Task 5).

같은 seed의 **20 step 연속 실행** 대 **10 step + checkpoint + 새 Python 프로세스에서 10 step**을
비교한다: 마지막 loss, sampler 위치, optimizer step 수, 고정한 parameter tensor 집합 — FP32·CPU에서
**정확히 같아야** 한다(허용 오차 없음: 두 실행이 같은 op를 같은 순서·같은 스레드 수(1)로 계산하고
누적 gradient·상태는 tensor 그대로 저장·복원된다). 에피소드 **구간 도중**의 중단(구간 위치와 넘겨받은
공통 상태의 복원)도 같은 기준으로 본다.

step마다 로봇 스트림 단위(에피소드)와 비로봇 단위(토큰 예산까지 묶은 단일 요청들)가 둘 다 들므로
(docs/04 §2) 검사용 설정은 에피소드를 4틱(0.4초)으로 잘라 0.2초 구간 둘로 나눈다 — 20 step 모두가
구간 학습·재개를 지난다.
"""

import copy
import json
import subprocess
import sys

import pytest
import torch
import yaml
from helpers import D0_MANIFEST, REPO, SMALL_VOCAB

from robo_jev.checkpoint import load_checkpoint

#: 비교할 parameter tensor (readout, embedding, DeltaNet·attention 층, 최종 norm).
WATCHED = (
    "U.weight",
    "V.weight",
    "bias",
    "backbone.embed.weight",
    "backbone.blocks.0.mixer.qkv_proj.weight",
    "backbone.blocks.0.mixer.A_log",
    "backbone.blocks.1.mixer.o_proj.weight",
    "backbone.blocks.2.mixer.q_proj.weight",
    "backbone.blocks.2.down_proj.weight",
    "backbone.norm.weight",
)


def base_config(root) -> dict:
    return {
        "run_name": "resume",
        "model_config": str(REPO / "configs" / "model" / "tiny_hybrid.yaml"),
        "model_vocab_size": SMALL_VOCAB,
        "dataset_manifest": str(D0_MANIFEST),
        "splits": ["train"],
        "tokenizer": "whitespace",
        "stream_chunk_seconds": 0.2,
        "stream_window_ticks": 30,
        "stream_max_ticks": 4,
        "trainable": "text_backbone_and_readout",
        "gradient_accumulation": 2,
        "microbatch_states_per_rank": 1,
        "nonrobot_tokens_per_unit": 512,
        "max_steps": 20,
        "warmup_ratio": 0.1,
        "seed": 17,
        "torch_threads": 1,
        "artifacts_dir": str(root / "runs"),
    }


def run_train(root, name: str, config: dict, *, resume=None) -> dict:
    """새 Python 프로세스에서 학습을 돌리고 CLI 요약을 돌려준다."""
    path = root / f"{name}.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    command = [sys.executable, "-m", "robo_jev.train", "--config", str(path)]
    if resume is not None:
        command += ["--resume", str(resume)]
    result = subprocess.run(command, capture_output=True, text=True, cwd=REPO)
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout.strip().splitlines()[-1])


def strip_timing(steps: list[dict]) -> list[dict]:
    out = copy.deepcopy(steps)
    for step in out:
        step.pop("seconds", None)
    return out


def metrics_of(root, run_id: str) -> dict:
    return json.loads((root / "runs" / run_id / "metrics.json").read_text(encoding="utf-8"))


def optimizer_steps(state: dict) -> set[int]:
    return {int(entry["step"]) for entry in state["optimizer"]["state"].values()}


def assert_identical_runs(continuous: dict, resumed: dict, *, continuous_metrics: dict, resumed_metrics: dict) -> None:
    # 마지막 loss와 step별 지표 전체 (시간 제외)
    assert resumed_metrics["steps"][-1]["loss"] == continuous_metrics["steps"][-1]["loss"]
    assert strip_timing(resumed_metrics["steps"]) == strip_timing(continuous_metrics["steps"])
    # sampler 위치
    assert resumed["sampler"] == continuous["sampler"]
    assert resumed["sampler"]["drawn"] == 40
    # optimizer step 수
    assert optimizer_steps(resumed) == optimizer_steps(continuous) == {20}
    assert resumed["scheduler"]["last_epoch"] == continuous["scheduler"]["last_epoch"] == 20
    # 고정한 parameter tensor 집합 — 비트 단위
    for name in WATCHED:
        assert torch.equal(resumed["model"][name], continuous["model"][name]), name
    # 그리고 모든 parameter·optimizer 모멘트
    assert set(resumed["model"]) == set(continuous["model"])
    for name in continuous["model"]:
        assert torch.equal(resumed["model"][name], continuous["model"][name]), name
    for key, entry in continuous["optimizer"]["state"].items():
        for moment in ("exp_avg", "exp_avg_sq"):
            assert torch.equal(resumed["optimizer"]["state"][key][moment], entry[moment]), (key, moment)
    # RNG
    assert torch.equal(resumed["rng"]["torch"], continuous["rng"]["torch"])
    assert resumed["rng"]["python"] == continuous["rng"]["python"]
    assert resumed["progress"] is None and continuous["progress"] is None


@pytest.fixture(scope="module")
def root(tmp_path_factory):
    return tmp_path_factory.mktemp("resume")


@pytest.fixture(scope="module")
def continuous(root) -> dict:
    """같은 seed의 20 step 연속 실행."""
    config = {**base_config(root), "run_id": "continuous-20"}
    summary = run_train(root, "continuous", config)
    assert summary["status"] == "completed" and summary["step"] == 20
    state = load_checkpoint(summary["checkpoint"])
    metrics = metrics_of(root, "continuous-20")
    assert all(s["items"]["stream"] == 1 and s["items"]["single"] >= 3 for s in metrics["steps"])  # step마다 두 종류
    assert all([u["kind"] for u in s["units"]] == ["stream", "single"] for s in metrics["steps"])
    assert all(abs(s["loss_share"]["domain"]["robot"] - 0.6) < 1e-6 for s in metrics["steps"])
    return {"state": state, "metrics": metrics}


def test_ten_steps_plus_a_fresh_process_of_ten_equals_twenty_continuous_steps(root, continuous):
    config = {**base_config(root), "run_id": "split-20", "stop_after": {"step": 10}}
    first = run_train(root, "first-half", config)
    assert first["status"] == "interrupted" and first["step"] == 10
    halfway = load_checkpoint(first["checkpoint"])
    assert halfway["step"] == 10 and halfway["progress"] is None and optimizer_steps(halfway) == {10}
    assert halfway["sampler"]["drawn"] == 20
    # 새 프로세스가 checkpoint를 읽어 10 step 더 돈다 (중단 지점은 지운다; run id는 checkpoint의 것; 나머지 설정은 같다)
    second = run_train(root, "second-half", base_config(root), resume=first["checkpoint"])
    assert second["status"] == "completed" and second["step"] == 20 and second["run_id"] == "split-20"
    resumed = load_checkpoint(second["checkpoint"])
    resumed_metrics = metrics_of(root, "split-20")
    assert [s["step"] for s in resumed_metrics["steps"]] == list(range(1, 21))
    assert_identical_runs(continuous["state"], resumed, continuous_metrics=continuous["metrics"], resumed_metrics=resumed_metrics)
    # 재개 직후의 다음 batch·loss·업데이트가 연속 실행과 같다 (계획서의 G0 비교)
    assert strip_timing(resumed_metrics["steps"][10:11]) == strip_timing(continuous["metrics"]["steps"][10:11])
    assert resumed_metrics["steps"][10]["units"] == continuous["metrics"]["steps"][10]["units"]


def test_resume_in_the_middle_of_an_episodes_chunk_sequence(root, continuous):
    """구간 도중 중단: step 1의 첫 단위(에피소드)의 구간 0 뒤에서 멈추고, 새 프로세스가 구간 1부터 이어간다."""
    config = {**base_config(root), "run_id": "mid-episode", "stop_after": {"step": 0, "unit": 0, "chunk": 0}}
    first = run_train(root, "mid-first", config)
    assert first["status"] == "interrupted" and first["step"] == 0
    halted = load_checkpoint(first["checkpoint"])
    progress = halted["progress"]
    assert progress is not None and progress["unit_index"] == 0 and progress["chunk_index"] == 1
    assert progress["units"][0]["kind"] == "stream" and len(progress["units"]) == 2
    carried = progress["carried_state"]
    assert carried["tick"] == 1 and carried["position"] > 0 and carried["cache_ticks"].shape[0] == carried["kv"][0]["k"].shape[1]
    assert progress["shares"] == {"robot": 0.6, "non_robot": 0.4} and progress["denominators"]["non_robot"] >= 3
    assert not carried["delta"][0]["recurrent"].requires_grad
    assert progress["grads"] and all(torch.isfinite(g).all() for g in progress["grads"].values())
    assert progress["accumulators"]["chunks"] == 1 and progress["accumulators"]["items"] == {"single": 0, "stream": 1}
    assert halted["sampler"]["drawn"] == 2  # step의 단위는 이미 뽑혀 있다
    second = run_train(root, "mid-second", {**base_config(root), "run_id": "mid-episode"}, resume=first["checkpoint"])
    assert second["status"] == "completed" and second["step"] == 20
    resumed = load_checkpoint(second["checkpoint"])
    resumed_metrics = metrics_of(root, "mid-episode")
    assert_identical_runs(continuous["state"], resumed, continuous_metrics=continuous["metrics"], resumed_metrics=resumed_metrics)
    # step 1의 지표(구간 0은 checkpoint의 누적값, 구간 1과 단일 요청은 재개한 프로세스의 계산)가 연속 실행과 같다
    assert strip_timing(resumed_metrics["steps"][:1]) == strip_timing(continuous["metrics"]["steps"][:1])
    assert resumed_metrics["steps"][0]["chunks"] == 3


def test_resume_refuses_a_checkpoint_from_a_different_run_identity(root, continuous):
    config = {**base_config(root), "run_id": "other", "seed": 18}
    path = root / "other.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    checkpoint = root / "runs" / "continuous-20" / "checkpoint.pt"
    result = subprocess.run(
        [sys.executable, "-m", "robo_jev.train", "--config", str(path), "--resume", str(checkpoint)],
        capture_output=True, text=True, cwd=REPO,
    )
    assert result.returncode != 0 and "seed" in result.stderr and "resume" in result.stderr
