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
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import torch
import yaml
from helpers import D0_MANIFEST, REPO, SMALL_VOCAB, read_jsonl

from robo_jev.checkpoint import load_checkpoint, save_checkpoint
from robo_jev.train import Trainer

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


# --------------------------------------------------------------------------
# run의 정체 — 데이터·tokenizer·직렬화·모델의 내용 (리뷰 11 S1)
# --------------------------------------------------------------------------


def copy_d0(target: Path) -> Path:
    """D0 fixture 세 파일의 사본 — 원본은 건드리지 않는다. 사본 manifest의 경로를 돌려준다."""
    target.mkdir(parents=True, exist_ok=True)
    for name in ("d0.jsonl", "d0_streams.jsonl", "d0_manifest.json"):
        shutil.copy2(D0_MANIFEST.with_name(name), target / name)
    return target / "d0_manifest.json"


def small_config(root, manifest: Path, **overrides) -> dict:
    return {**base_config(root), "dataset_manifest": str(manifest), "max_steps": 2, **overrides}


def flip_first_train_answer(manifest: Path) -> tuple[str, str]:
    """첫 train 단일 요청의 정답을 c0→c1로 바꾸고 manifest의 파일 해시·크기도 맞춘다 (리뷰 11의 재현). (이전 해시, 새 해시)."""
    rows_path = manifest.with_name("d0.jsonl")
    rows = read_jsonl(rows_path)
    assert rows[0]["split"] == "train" and rows[0]["labels"][0]["candidate_ids"] == ["c0"]
    rows[0]["labels"][0]["candidate_ids"] = ["c1"]
    rows_path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    record = json.loads(manifest.read_text(encoding="utf-8"))
    before = record["files"]["d0.jsonl"]["sha256"]
    after = hashlib.sha256(rows_path.read_bytes()).hexdigest()
    assert after != before
    record["files"]["d0.jsonl"]["sha256"] = after
    record["files"]["d0.jsonl"]["bytes"] = rows_path.stat().st_size
    manifest.write_text(json.dumps(record, ensure_ascii=False), encoding="utf-8")
    return before, after


def parameters_of(trainer: Trainer) -> dict[str, torch.Tensor]:
    return {name: p.detach().clone() for name, p in trainer.model.named_parameters()}


def test_resume_refuses_the_run_when_the_data_content_changed_even_with_updated_manifest_hashes(tmp_path):
    """리뷰 11 S1의 재현: D0 사본으로 1 step 후 저장 → 첫 train 레코드의 정답을 바꾸고 사본 manifest의 파일 해시도
    갱신 → 같은 설정·같은 경로로 재개. 설정은 같지만 run의 정체(데이터 내용)가 다르므로 거절해야 하고, 오류는
    달라진 키(파일 해시·manifest 해시)를 모두 이름으로 적는다."""
    manifest = copy_d0(tmp_path / "data")
    config = small_config(tmp_path, manifest, run_id="identity")
    with Trainer(config) as first:
        first.run_step()
        checkpoint = first.save()
        saved_manifest_sha = first.manifest["dataset_manifests"][0]["sha256"]
    before, after = flip_first_train_answer(manifest)
    with pytest.raises(ValueError, match="resume") as excinfo:
        Trainer(config, resume=checkpoint)
    message = str(excinfo.value)
    assert "d0.jsonl" in message and before[:12] in message and after[:12] in message
    assert saved_manifest_sha[:12] in message  # manifest 자체의 해시도 달라졌다
    assert "d0_streams.jsonl" not in message  # 바뀌지 않은 파일은 이름에 오르지 않는다
    # 같은 데이터로는 여전히 재개된다 (거절이 checkpoint를 망가뜨리지 않았다)
    copy_d0(tmp_path / "data")
    with Trainer(config, resume=checkpoint) as resumed:
        assert resumed.step == 1


def test_resume_accepts_the_same_data_and_model_config_at_different_paths_and_stays_bit_exact(tmp_path):
    """경로는 재개 자유 항목이고 정체는 내용이다: 같은 D0 사본을 다른 디렉터리에 두고 재개해도 되며(모델 설정 파일도
    다른 경로의 같은 내용), 결과는 같은 경로의 연속 실행과 비트 단위로 같다."""
    a, b = copy_d0(tmp_path / "a"), copy_d0(tmp_path / "b")
    model_copy = tmp_path / "model" / "tiny_hybrid_copy.yaml"
    model_copy.parent.mkdir()
    shutil.copy2(REPO / "configs" / "model" / "tiny_hybrid.yaml", model_copy)

    with Trainer(small_config(tmp_path, a, run_id="continuous-2")) as continuous:
        continuous_metrics = [continuous.run_step() for _ in range(2)]
        continuous_parameters = parameters_of(continuous)
        continuous_sampler = continuous.sampler.state_dict()
    with Trainer(small_config(tmp_path, a, run_id="split-2")) as first:
        first.run_step()
        checkpoint = first.save()
    moved = small_config(tmp_path, b, run_id="ignored-run-id", model_config=str(model_copy))
    with Trainer(moved, resume=checkpoint) as resumed:
        assert resumed.run_id == "split-2" and resumed.step == 1
        assert resumed.manifest["dataset_manifests"][0]["path"] == str(b)
        assert resumed.manifest["model"]["config"] == str(model_copy)
        resumed_metrics = [*resumed.history, resumed.run_step()]
        assert strip_timing(resumed_metrics) == strip_timing(continuous_metrics)
        assert resumed.sampler.state_dict() == continuous_sampler
        for name, tensor in parameters_of(resumed).items():
            assert torch.equal(tensor, continuous_parameters[name]), name


def test_resume_names_every_differing_identity_key(tmp_path):
    """정체 블록의 어느 항목이 달라도 거절하고, 달라진 키를 **모두** 이름으로 적는다 (설정 불일치와 같은 형식)."""
    manifest = copy_d0(tmp_path / "data")
    config = small_config(tmp_path, manifest, run_id="identity-keys")
    with Trainer(config) as first:
        first.run_step()
        checkpoint = first.save()
        identity = first.manifest["identity"]
    assert set(identity) >= {"datasets", "splits", "serializer_version", "question_set", "layouts", "tokenizer", "model"}
    assert identity["datasets"][0]["files"]["d0.jsonl"] == json.loads(manifest.read_text(encoding="utf-8"))["files"]["d0.jsonl"]["sha256"]
    assert identity["tokenizer"] == {"kind": "whitespace"}
    assert identity["model"]["kind"] == "tiny_hybrid" and identity["model"]["parameters"] > 0
    assert "path" not in identity["datasets"][0] and "config" not in identity["model"]  # 경로는 정체가 아니다

    state = load_checkpoint(checkpoint)
    tampered = copy.deepcopy(state["manifest"])
    tampered["identity"]["serializer_version"] = "ts9.9"
    tampered["identity"]["question_set"]["markers"]["q_main"] = "<other>"
    tampered["identity"]["model"]["parameters"] += 1
    tampered["identity"]["tokenizer"] = {"kind": "tokenizers", "sha256": "0" * 64, "id": "Some/Model", "revision": "abc"}
    tampered["identity"]["datasets"][0]["files"]["d0_streams.jsonl"] = "f" * 64
    save_checkpoint(checkpoint, {**state, "manifest": tampered})
    with pytest.raises(ValueError, match="resume") as excinfo:
        Trainer(config, resume=checkpoint)
    message = str(excinfo.value)
    for key in (
        "serializer_version", "question_set.markers.q_main", "model.parameters", "tokenizer.kind", "tokenizer.sha256",
        "datasets[0].files.d0_streams.jsonl",
    ):
        assert key in message, key
    assert "datasets[0].files.d0.jsonl" not in message and "ts9.9" in message


# --------------------------------------------------------------------------
# 끝난 run에 더 잇기 — `max_steps`를 바꾸는 재개 (Task R3a C1)
# --------------------------------------------------------------------------


def test_resume_refuses_a_different_max_steps_unless_the_run_says_to_reschedule(root, continuous):
    """1 epoch을 끝낸 run에 1 epoch을 더 잇는다 — **말하고** 이어야 한다 (Task R3a C1).

    `max_steps`는 예산이 아니라 **일정**이다: warmup과 cosine이 그 값에서 나오므로, 바꾸고 이어 가는 것은
    같은 run의 연장이 아니라 **다른 일정 위의 연속 학습**이다. 그래서 기본은 거절이고, 설정이
    `resume_reschedule: true`로 그러겠다고 말할 때만 허용하며, 그 사실이 run 기록에 남는다.
    """
    ten = {**base_config(root), "run_id": "extend-10", "max_steps": 10}
    first = run_train(root, "extend-first", ten)
    assert first["status"] == "completed" and first["step"] == 10
    at_ten = root / "extend-at10.pt"
    shutil.copy2(first["checkpoint"], at_ten)

    twenty = {**base_config(root), "max_steps": 20}
    path = root / "extend-refuse.yaml"
    path.write_text(yaml.safe_dump(twenty), encoding="utf-8")
    refused = subprocess.run(
        [sys.executable, "-m", "robo_jev.train", "--config", str(path), "--resume", str(at_ten)],
        capture_output=True, text=True, cwd=REPO,
    )
    assert refused.returncode != 0 and "max_steps" in refused.stderr and "resume" in refused.stderr

    second = run_train(root, "extend-second", {**twenty, "resume_reschedule": True}, resume=at_ten)
    assert second["status"] == "completed" and second["step"] == 20 and second["run_id"] == "extend-10"
    metrics = metrics_of(root, "extend-10")
    assert [entry["step"] for entry in metrics["steps"]] == list(range(1, 21))
    # 무엇이 다시 잡혔는지가 run 기록에 있다 — 나중에 이 곡선을 읽는 사람이 한 run으로 오해하지 않게
    assert metrics["summary"]["rescheduled"] == {"max_steps": {"from": 10, "to": 20}}
    # 재개하지 않은 run에는 그 자리가 비어 있다
    assert metrics_of(root, "continuous-20")["summary"]["rescheduled"] is None


def test_rescheduling_puts_the_first_resumed_step_on_the_new_schedule_not_the_old_ones_last_value(root):
    """옛 일정의 마지막 lr은 **0**이다(cosine의 끝) — 그대로 두면 재개 첫 step이 아무것도 배우지 않는다.

    `LambdaLR.load_state_dict`는 lr을 다시 계산하지 않는다(람다는 상태에 담기지 않으므로 새 람다가 그대로
    남고, optimizer에는 저장된 옛 값이 실린다). 일정을 다시 잡았다면 **지금 자리의 새 값**을 건다.
    """
    from robo_jev.train import Trainer, lr_factor

    ten = {**base_config(root), "run_id": "lr-10", "max_steps": 10}
    first = run_train(root, "lr-first", ten)
    at_ten = root / "lr-at10.pt"
    shutil.copy2(first["checkpoint"], at_ten)
    assert load_checkpoint(at_ten)["optimizer"]["param_groups"][0]["lr"] == pytest.approx(0.0)  # 옛 일정의 끝

    twenty = {**base_config(root), "run_id": "lr-10", "max_steps": 20, "resume_reschedule": True}
    with Trainer(twenty, resume=at_ten) as trainer:
        expected = lr_factor(10, max_steps=20, warmup_ratio=0.1)
        assert expected > 0
        assert trainer.scheduler.last_epoch == 10
        for group in trainer.optimizer.param_groups:
            assert group["lr"] == pytest.approx(group["initial_lr"] * expected)
        assert trainer.rescheduled == {"max_steps": {"from": 10, "to": 20}}
