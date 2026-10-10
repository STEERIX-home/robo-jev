"""`scripts/slim_checkpoint.py` — 끝난 run의 체크포인트 슬림화 (Task R8 A2, 사용자 승인: 정확히 아홉 파일; Task R14 A1: 일곱 파일 더, 상시 규칙).

도구는 파일마다 optimizer만 뺀 model-only 파일을 같은 디렉터리의 임시 파일에 쓰고, (1) model tensor가 원본과 비트 단위로 같은지,
(2) optimizer 밖의 다른 키가 같은지, (3) 평가 경로가 그 파일을 받는지를 본 뒤 **셋 다 통과할 때만** 원본을 대체한다. 하나라도 떨어지면
원본은 바이트 하나 바뀌지 않고 임시 파일도 남지 않는다. 대체할 수 있는 경로는 사용자가 승인한 파일뿐이다(승인마다 날짜가 붙은 묶음). 여기서는 소형 fixture의
진짜 저장 단위와, 평가 경로 자리에 같은 `load_readout_checkpoint`를 소형 모델로 부르는 검사를 쓴다(실제 2B 적재는 그 도구의 몫).
"""

import functools
import hashlib
import importlib.util
import json
import sys

import pytest
import torch
from helpers import D0_MANIFEST, REPO, SMALL_VOCAB

from robo_jev.checkpoint import CHECKPOINT_FORMAT, MODEL_ONLY_FORMAT, load_checkpoint, load_model_checkpoint
from robo_jev.train import Trainer, build_model, load_readout_checkpoint, resolve_config

SCRIPT = REPO / "scripts" / "slim_checkpoint.py"


@functools.lru_cache(maxsize=1)
def script():
    spec = importlib.util.spec_from_file_location("slim_checkpoint", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def tiny_config(tmp_path, run_id: str) -> dict:
    return {
        "run_name": "slim", "run_id": run_id, "model_config": str(REPO / "configs" / "model" / "tiny_hybrid.yaml"),
        "model_vocab_size": SMALL_VOCAB, "dataset_manifest": str(D0_MANIFEST), "splits": ["train"], "tokenizer": "whitespace",
        "stream_chunk_seconds": 0.2, "stream_window_ticks": 30, "stream_max_ticks": 4, "trainable": "text_backbone_and_readout",
        "gradient_accumulation": 2, "nonrobot_tokens_per_unit": 512, "max_steps": 2, "warmup_ratio": 0.5, "seed": 17,
        "torch_threads": 1, "artifacts_dir": str(tmp_path / "runs"),
    }  # fmt: skip


def finished_run(tmp_path, run_id: str = "done"):
    config = tiny_config(tmp_path, run_id)
    with Trainer(config) as trainer:
        trainer.run()
    return tmp_path / "runs" / run_id / "checkpoint.pt", config


def tiny_evaluation(config: dict):
    """평가 경로 자리: 같은 설정의 소형 모델을 만들고 `load_readout_checkpoint`(계약 digest 대조 + model 적재)를 부른 뒤 실린 tensor를
    돌려준다 — 실제 도구의 `serving_load_check`와 같은 모양(소형 fixture에는 실제 tokenizer가 없어 checkpoint의 해시를 믿는다)."""

    def check(path, model):
        judge = build_model(resolve_config({**config, "run_id": "eval"}))
        manifest = load_readout_checkpoint(judge, path, tokenizer_sha256=None, trust_checkpoint_tokenizer=True)
        return script().loaded_state_check(judge.state_dict(), model, contract_sha256=manifest.get("contract_sha256"))

    return check


def sha256(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_a_file_is_replaced_by_its_model_only_twin_only_after_every_check_passes(tmp_path):
    path, config = finished_run(tmp_path)
    before = sha256(path)
    size_before = path.stat().st_size
    original = load_checkpoint(path)
    row = script().slim_one(path, evaluate=tiny_evaluation(config))
    assert row["replaced"] is True and row["error"] is None
    assert row["checks"]["model_bits"]["equal"] and row["checks"]["model_bits"]["tensors"] == len(original["model"])
    assert row["checks"]["other_keys"]["equal"] and row["checks"]["evaluation"]["params_equal"]
    assert row["sha256_before"] == before and row["bytes_before"] == size_before
    assert row["sha256_after"] == sha256(path) and row["bytes_after"] == path.stat().st_size < size_before
    assert row["freed_bytes"] == size_before - row["bytes_after"]
    slim = load_model_checkpoint(path)
    assert slim["format"] == MODEL_ONLY_FORMAT and "optimizer" not in slim
    assert slim["slimmed"]["source_sha256"] == before and slim["slimmed"]["source_bytes"] == size_before
    for name, tensor in original["model"].items():
        assert torch.equal(slim["model"][name], tensor), name
    assert not list(path.parent.glob("*.tmp"))
    # 다시 돌리면 이미 슬림한 파일은 건드리지 않는다
    again = script().slim_one(path, evaluate=tiny_evaluation(config))
    assert again["replaced"] is False and again["skipped"] == "already model-only" and sha256(path) == row["sha256_after"]


def test_a_failing_check_leaves_the_original_byte_for_byte_and_no_temporary_file(tmp_path, monkeypatch):
    path, config = finished_run(tmp_path)
    before = sha256(path)
    module = script()

    def broken(*args, **kwargs):
        raise ValueError("배포 계약 digest가 지금 체크아웃과 다르다 (시험)")

    failed = module.slim_one(path, evaluate=broken)
    assert failed["replaced"] is False and "digest" in failed["error"]
    assert sha256(path) == before and load_checkpoint(path)["format"] == CHECKPOINT_FORMAT
    assert not list(path.parent.glob("*.tmp"))

    real = module.model_only_state

    def perturbed(state, *, slimmed):
        out = real(state, slimmed=slimmed)
        name = sorted(out["model"])[0]
        out["model"] = {**out["model"], name: out["model"][name] + 1}
        return out

    monkeypatch.setattr(module, "model_only_state", perturbed)
    flipped = module.slim_one(path, evaluate=tiny_evaluation(config))
    assert flipped["replaced"] is False and not flipped["checks"]["model_bits"]["equal"]
    assert flipped["checks"]["model_bits"]["mismatches"][0]["reason"] == "values"
    assert sha256(path) == before and not list(path.parent.glob("*.tmp"))


def test_a_dry_run_verifies_but_never_replaces(tmp_path):
    path, config = finished_run(tmp_path)
    before = sha256(path)
    row = script().slim_one(path, evaluate=tiny_evaluation(config), dry_run=True)
    assert row["replaced"] is False and row["dry_run"] is True and row["checks"]["model_bits"]["equal"]
    assert row["checks"]["evaluation"]["params_equal"] and row["bytes_after"] is None
    assert sha256(path) == before and not list(path.parent.glob("*.tmp"))


#: Task R8 A2 — 사용자 승인(2026-09-29)의 아홉 파일.
R8_NINE = (
    "artifacts/runs/r2-t1-fp32-2b/checkpoint.pt", "artifacts/runs/r2-t1-fp32-2b/checkpoint-step40.pt", "artifacts/runs/r3a-t1-fp32-2b-s18/checkpoint.pt",
    "artifacts/runs/r3a-t1-fp32-2b-s19/checkpoint.pt", "artifacts/runs/r3a-t1-fp32-2b-s19/checkpoint-step40.pt",
    "artifacts/runs/r3a-t1-466/r3a-t1-fp32-2b-466/checkpoint.pt", "artifacts/runs/r5-t1-fp32-2b-s18/checkpoint.pt",
    "artifacts/runs/r6-t1-fp32-2b-s18/checkpoint.pt", "artifacts/runs/r7-t1-fp32-2b-s18/checkpoint.pt",
)  # fmt: skip
#: Task R14 A1 — 사용자 승인(2026-10-09, AskUserQuestion "일곱 개 슬림화 + 이후 자동")의 일곱 파일, 브리프의 목록 그대로.
R14_SEVEN = (
    "artifacts/runs/r8-t1-fp32-2b-s18/checkpoint.pt", "artifacts/runs/r8-t1-fp32-2b-s19/checkpoint.pt", "artifacts/runs/r9-t1-fp32-2b-s17/checkpoint.pt",
    "artifacts/runs/r9-t1-fp32-2b-s18c/r8-t1-fp32-2b-s18/checkpoint.pt", "artifacts/runs/r12-t1-fp32-2b-s17/checkpoint.pt",
    "artifacts/runs/r12-t1-fp32-2b-s18/checkpoint.pt", "artifacts/runs/r12-t1-fp32-2b-s19/checkpoint.pt",
)  # fmt: skip
#: 상시 규칙(2026-10-09, HANDOFF §5) — Task R14 자신의 세 체크포인트(모든 R14 평가 run이 끝난 뒤 승인 묶음에 더했다).
R14_OWN = ("artifacts/runs/r14-t1-fp32-2b-s18/checkpoint.pt", "artifacts/runs/r14-t1-fp32-2b-s17/checkpoint.pt",
           "artifacts/runs/r14-t1-fp32-2b-s19/checkpoint.pt")


def test_only_the_user_approved_checkpoints_can_be_slimmed_each_under_its_dated_approval(tmp_path):
    """대체할 수 있는 것은 사용자가 승인한 파일뿐이다 — R8의 아홉(2026-09-29), R14 A1의 일곱(2026-10-09), 그리고 상시 규칙(2026-10-09)으로 더한 R14 자신의
    셋(평가 run이 모두 끝난 뒤에 더했다 — 완주한 run만 대체한다); 승인마다 날짜·근거가 붙은 묶음이다. 목록 밖의 경로는 아무것도 읽기 전에 거절한다."""
    module = script()
    assert [(approval["date"], approval["standing_rule"], approval["paths"]) for approval in module.APPROVALS] == [
        ("2026-09-29", False, R8_NINE), ("2026-10-09", False, R14_SEVEN), ("2026-10-09", True, R14_OWN)]
    assert module.APPROVED == R8_NINE + R14_SEVEN + R14_OWN and len(set(module.APPROVED)) == 19
    assert module.approval_of("artifacts/runs/r14-t1-fp32-2b-s19/checkpoint.pt")["standing_rule"] is True  # 상시 규칙 — 완주한 run만 대체한다
    assert all(path.startswith("artifacts/runs/") and path.endswith(".pt") for path in module.APPROVED)
    assert module.approval_of("artifacts/runs/r12-t1-fp32-2b-s19/checkpoint.pt") == {
        "date": "2026-10-09", "basis": module.APPROVALS[1]["basis"], "standing_rule": False}
    assert module.approval_of("artifacts/runs/r7-t1-fp32-2b-s18/checkpoint.pt")["date"] == "2026-09-29"
    with pytest.raises(KeyError):
        module.approval_of("artifacts/runs/r6-t1-fp32-2b-s18/metrics.json")
    path, _ = finished_run(tmp_path)
    with pytest.raises(SystemExit):
        module.main(["run", "--files", str(path), "--out", str(tmp_path / "report.json")])
    assert not (tmp_path / "report.json").exists() and load_checkpoint(path)["format"] == CHECKPOINT_FORMAT


def test_a_standing_rule_checkpoint_is_slimmed_only_when_its_run_completed_and_every_row_names_its_approval(tmp_path, monkeypatch):
    """상시 규칙(2026-10-09)으로 더한 묶음은 run이 `completed`로 `max_steps`에 닿았을 때만 대체한다 — 끝나지 않은 run(재개할 수 있는 상태)은
    읽지도 않고 그대로 두고 이유를 적는다. 보고서의 행마다 승인 묶음(날짜·근거·상시 규칙 여부)이, 머리에 실행한 커밋과 작업 트리의 변경 파일이 실린다."""
    module = script()
    done_config = {**tiny_config(tmp_path, "done"), "artifacts_dir": str(tmp_path / "artifacts" / "runs")}
    with Trainer(done_config) as trainer:
        trainer.run()
    half_config = {**tiny_config(tmp_path, "half"), "artifacts_dir": str(tmp_path / "artifacts" / "runs"), "stop_after": {"step": 1}}
    with Trainer(half_config) as trainer:
        assert trainer.run()["status"] == "interrupted"
    done, half = tmp_path / "artifacts/runs/done/checkpoint.pt", tmp_path / "artifacts/runs/half/checkpoint.pt"
    half_before = sha256(half)
    approvals = ({"date": "2026-10-09", "basis": "standing rule (test)", "standing_rule": True,
                  "paths": ("artifacts/runs/done/checkpoint.pt", "artifacts/runs/half/checkpoint.pt")},)  # fmt: skip
    monkeypatch.setattr(module, "REPO", tmp_path)
    monkeypatch.setattr(module, "APPROVALS", approvals)
    monkeypatch.setattr(module, "APPROVED", approvals[0]["paths"])
    monkeypatch.setattr(module, "serving_load_check", tiny_evaluation(done_config))
    assert module.run_completed(done) == (True, "completed 2/2") and module.run_completed(half)[0] is False
    assert module.main(["run", "--out", str(tmp_path / "report.json")]) == 1  # 한 파일이 대체되지 않았다
    report = json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))
    rows = {row["path"]: row for row in report["files"]}
    assert rows["artifacts/runs/done/checkpoint.pt"]["replaced"] is True and load_model_checkpoint(done)["format"] == MODEL_ONLY_FORMAT
    assert rows["artifacts/runs/half/checkpoint.pt"]["replaced"] is False and "standing rule" in rows["artifacts/runs/half/checkpoint.pt"]["error"]
    assert "status interrupted" in rows["artifacts/runs/half/checkpoint.pt"]["error"]
    assert sha256(half) == half_before and load_checkpoint(half)["format"] == CHECKPOINT_FORMAT
    assert all(row["approval"] == {"date": "2026-10-09", "basis": "standing rule (test)", "standing_rule": True} for row in report["files"])
    assert report["failed"] == ["artifacts/runs/half/checkpoint.pt"] and report["replaced"] == 1
    assert report["approvals"] == [{**approvals[0], "paths": list(approvals[0]["paths"])}] and report["script"] == "slim-checkpoint-1.1"
    assert "git" in report and "git_dirty_files" in report  # 저장소 밖(tmp)이라 None이다 — 저장소 안에서는 커밋과 변경 파일 목록
    assert set(report["tool_files_sha256"]) == {"scripts/slim_checkpoint.py", "src/robo_jev/checkpoint.py", "src/robo_jev/train.py",
                                                "src/robo_jev/harness/model_policy.py"}  # 실행한 코드의 sha256 (tmp에는 없어 None)


def test_the_loaded_state_check_compares_what_the_evaluation_loader_put_into_the_model_bit_for_bit():
    module = script()
    model = {"U.weight": torch.ones(2, 2), "bias": torch.zeros(1)}
    assert module.loaded_state_check({**model, "other": torch.zeros(3)}, model, contract_sha256="x")["params_equal"] is True
    assert module.loaded_state_check({"U.weight": torch.ones(2, 2), "bias": torch.ones(1)}, model, contract_sha256="x")["mismatches"] == ["bias"]
    assert module.loaded_state_check({"U.weight": torch.ones(2, 2)}, model, contract_sha256="x")["mismatches"] == ["bias"]


def test_the_dirty_file_list_names_each_changed_tracked_file_by_its_full_path(tmp_path, monkeypatch):
    """보고서의 `git_dirty_files`는 바뀐 추적 파일의 **온전한** 저장소 기준 경로다 — `git status --porcelain`의 첫 줄 앞 공백(상태 열)을 지우면 첫
    경로의 첫 글자가 잘린다(R14 A1의 마른 실행에서 `cripts/…`로 나온 결함). 추적하지 않는 파일은 세지 않는다."""
    import subprocess

    module = script()
    for name in ("scripts/tool.py", "src/model.py"):
        (tmp_path / name).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / name).write_text("x = 1\n", encoding="utf-8")
    git = ["git", "-C", str(tmp_path), "-c", "user.name=test", "-c", "user.email=test@example.com"]
    subprocess.run([*git, "init", "-q"], check=True)
    subprocess.run([*git, "add", "."], check=True)
    subprocess.run([*git, "commit", "-q", "-m", "init"], check=True)
    monkeypatch.setattr(module, "REPO", tmp_path)
    assert module.dirty_files() == []
    for name in ("scripts/tool.py", "src/model.py"):
        (tmp_path / name).write_text("x = 2\n", encoding="utf-8")
    (tmp_path / "untracked.txt").write_text("u", encoding="utf-8")
    assert module.dirty_files() == ["scripts/tool.py", "src/model.py"]
