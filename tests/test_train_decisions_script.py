"""`scripts/train_decisions.py` — 판단 요청 학습 실행기 (Task v0d).

소형 난수 Qwen(``--set model_id=tiny``, CPU)으로 실행기를 끝까지 돌린다: ``split``이 쌍을 깨지 않고 학습/평가 파일과 manifest를 쓰는지,
``lengths``가 질문마다·학습 계획의 길이 분포를 내는지, ``train``이 체크포인트·metrics.json(완주 표시·dev·실행 환경)을 남기고 그 체크포인트를
진단 도구가 ``--checkpoint``로 같은 프롬프트에 얹어 채점하는지(기반 모델의 예측과 달라진다). 실제 모델 학습은 GPU 유닛의 일이다.
"""

import functools
import importlib.util
import json
import sys

import pytest
from helpers import FIXTURES, REPO

from robo_jev.checkpoint import load_checkpoint
from robo_jev.decisions.records import pair_of, read_records

pytest.importorskip("transformers")
from robo_jev.model.backbone_qwen import torch_reference_kernels  # noqa: E402

TRAIN = REPO / "scripts" / "train_decisions.py"
DIAGNOSE = REPO / "scripts" / "diagnose_decisions.py"
REQUESTS = FIXTURES / "decisions" / "requests.jsonl"


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@functools.lru_cache(maxsize=1)
def train_script():
    return _load(TRAIN, "train_decisions")


@functools.lru_cache(maxsize=1)
def diagnose_script():
    return _load(DIAGNOSE, "diagnose_decisions_v0d")


TINY = ["--set", "model_id=tiny", "--set", "device=cpu", "--set", "dtype=float32", "--set", "min_free_disk_gb=0", "--set", "batch_examples=4",
        "--set", "lora={r: 4, alpha: 8, dropout: 0.0, targets: [q_proj, v_proj, in_proj_qkv, down_proj]}", "--set", "lr=0.01"]


def test_split_keeps_pairs_together_and_writes_a_manifest(tmp_path):
    assert train_script().main(["split", "--input", str(REQUESTS), "--eval-records", "3", "--seed", "0", "--out-dir", str(tmp_path)]) == 0
    manifest = json.loads((tmp_path / "requests.split.json").read_text(encoding="utf-8"))
    train, evaluation = read_records(tmp_path / "requests.train.jsonl"), read_records(tmp_path / "requests.eval.jsonl")
    ids_train, ids_eval = {r["id"] for r in train}, {r["id"] for r in evaluation}
    assert len(train) + len(evaluation) == 8 and not (ids_train & ids_eval) and len(evaluation) >= 3
    assert manifest["pairs_split_across"] == 0 and manifest["eval"]["records"] == len(evaluation) and sorted(ids_eval) == manifest["eval"]["ids"]
    group = {"fx-001", "fx-002", "fx-003"}
    assert group <= ids_train or group <= ids_eval
    assert all((pair_of(r)[0] in ids_eval) == (r["id"] in ids_eval) for r in train + evaluation if pair_of(r)[0])


def test_lengths_reports_every_question_and_the_training_plan(tmp_path):
    out = tmp_path / "lengths.json"
    assert train_script().main(["lengths", "--input", str(REQUESTS), "--out", str(out)]) == 0
    payload = json.loads(out.read_text(encoding="utf-8"))
    records = read_records(REQUESTS)
    assert payload["every_question"]["all"]["count"] == sum(len(r["request"]["questions"]) for r in records) == 33
    assert payload["training_plan_epoch0"]["examples"] == payload["training_plan_epoch0"]["all"]["count"] == sum(min(2, len(r["request"]["questions"])) for r in records)
    assert set(payload["every_question"]) == {"all", "choice", "noul", "score"} and payload["every_question"]["all"]["max"] >= payload["every_question"]["all"]["mean"]


def test_a_tiny_training_run_writes_its_records_and_the_diagnosis_reads_its_checkpoint(tmp_path):
    with torch_reference_kernels():
        assert train_script().main(["train", *TINY, "--set", f"train_input={REQUESTS}", "--set", f"out_dir={tmp_path / 'runs'}", "--set", "run_name=tiny",
                                    "--set", "dev_fraction=0.3", "--set", "dev_every=2", "--set", "epochs=2"]) == 0
    run = tmp_path / "runs" / "tiny"
    metrics = json.loads((run / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["status"] == "completed" and metrics["step"] == metrics["config"]["max_steps"] and metrics["dev"] and metrics["dev"][0]["step"] == 0
    assert metrics["manifest"]["run_env"]["script_version"] == "v0d-train-1" and metrics["manifest"]["data"]["file_sha256"]
    state = load_checkpoint(run / "checkpoint.pt")
    assert state["manifest"]["model"]["mode"] == "lora" and state["step"] == metrics["step"]
    base, after = tmp_path / "base.json", tmp_path / "after.json"
    with torch_reference_kernels():
        assert diagnose_script().main(["--model", "tiny", "--input", str(REQUESTS), "--out", str(base), "--n-boot", "10", "--batch", "8"]) == 0
        assert diagnose_script().main(["--model", "tiny", "--checkpoint", str(run / "checkpoint.pt"), "--input", str(REQUESTS), "--out", str(after),
                                       "--n-boot", "10", "--batch", "8"]) == 0
    summary = json.loads(after.read_text(encoding="utf-8"))
    assert summary["config"]["checkpoint"]["run_id"] == "tiny" and summary["config"]["checkpoint"]["mode"] == "lora"
    assert summary["config"]["model"] == f"tiny+lora:tiny@{metrics['step']}" and len(summary["config"]["checkpoint"]["sha256"]) == 64
    rows_base = [json.loads(line) for line in base.with_suffix(".predictions.jsonl").read_text(encoding="utf-8").splitlines()]
    rows_after = [json.loads(line) for line in after.with_suffix(".predictions.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [r["id"] for r in rows_base] == [r["id"] for r in rows_after]
    listings = lambda rows: [l["code_probs"] for r in rows for q in r["questions"].values() for l in q["listings"]]  # noqa: E731
    assert listings(rows_base) != listings(rows_after)
    options = lambda rows: [l["options"] for r in rows for q in r["questions"].values() for l in q["listings"]]  # noqa: E731
    assert options(rows_base) == options(rows_after)  # 같은 프롬프트·같은 균형 배치


def test_the_diagnosis_refuses_a_checkpoint_without_a_base_model(tmp_path):
    with pytest.raises(SystemExit):
        diagnose_script().main(["--rescore", str(tmp_path / "x.jsonl"), "--checkpoint", str(tmp_path / "c.pt"), "--input", str(REQUESTS), "--out", str(tmp_path / "o.json")])
