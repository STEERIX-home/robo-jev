"""`scripts/slim_checkpoint.py` — 끝난 run의 체크포인트 슬림화 (Task R8 A2, 사용자 승인: 정확히 아홉 파일).

도구는 파일마다 optimizer만 뺀 model-only 파일을 같은 디렉터리의 임시 파일에 쓰고, (1) model tensor가 원본과 비트 단위로 같은지,
(2) optimizer 밖의 다른 키가 같은지, (3) 평가 경로가 그 파일을 받는지를 본 뒤 **셋 다 통과할 때만** 원본을 대체한다. 하나라도 떨어지면
원본은 바이트 하나 바뀌지 않고 임시 파일도 남지 않는다. 대체할 수 있는 경로는 사용자가 승인한 아홉 개뿐이다. 여기서는 소형 fixture의
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


def test_only_the_nine_approved_checkpoints_can_be_slimmed(tmp_path):
    """사용자가 승인한 것은 정확히 아홉 파일이다 — 목록 밖의 경로는 아무것도 읽기 전에 거절한다."""
    module = script()
    assert len(module.APPROVED) == 9 and len(set(module.APPROVED)) == 9
    assert all(path.startswith("artifacts/runs/") and path.endswith(".pt") for path in module.APPROVED)
    assert "artifacts/runs/r7-t1-fp32-2b-s18/checkpoint.pt" in module.APPROVED and "artifacts/runs/r2-t1-fp32-2b/checkpoint-step40.pt" in module.APPROVED
    path, _ = finished_run(tmp_path)
    with pytest.raises(SystemExit):
        module.main(["run", "--files", str(path), "--out", str(tmp_path / "report.json")])
    assert not (tmp_path / "report.json").exists() and load_checkpoint(path)["format"] == CHECKPOINT_FORMAT


def test_the_loaded_state_check_compares_what_the_evaluation_loader_put_into_the_model_bit_for_bit():
    module = script()
    model = {"U.weight": torch.ones(2, 2), "bias": torch.zeros(1)}
    assert module.loaded_state_check({**model, "other": torch.zeros(3)}, model, contract_sha256="x")["params_equal"] is True
    assert module.loaded_state_check({"U.weight": torch.ones(2, 2), "bias": torch.ones(1)}, model, contract_sha256="x")["mismatches"] == ["bias"]
    assert module.loaded_state_check({"U.weight": torch.ones(2, 2)}, model, contract_sha256="x")["mismatches"] == ["bias"]
