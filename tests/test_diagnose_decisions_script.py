"""`scripts/diagnose_decisions.py` — 진단 실행기 (Task v0c §2.4).

소형 난수 Qwen(``--model tiny``, 검사 전용 — CPU)으로 실행기를 끝까지 돌려 세 산출물(요약 JSON, 레코드별 예측 JSONL, markdown)이
생기고 서로 맞는지, 형식이 틀린 기록은 세고 건너뛰는지, ``--limit``·``--seed``가 쌍 단위 표본을 고르는지, CPU 전용
``--rescore``(저장한 예측에서 요약 다시 계산)와 ``--compare``(두 run의 짝지은 비교)를 본다. 실제 모델 run은 GPU 유닛의 일이다.
"""

import copy
import functools
import importlib.util
import json
import sys

import pytest
from helpers import FIXTURES, REPO

pytest.importorskip("transformers")
from robo_jev.model.backbone_qwen import torch_reference_kernels  # noqa: E402

SCRIPT = REPO / "scripts" / "diagnose_decisions.py"
REQUESTS = FIXTURES / "decisions" / "requests.jsonl"


@functools.lru_cache(maxsize=1)
def script():
    spec = importlib.util.spec_from_file_location("diagnose_decisions", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _run(argv: list[str]) -> int:
    with torch_reference_kernels():
        return script().main(argv)


def test_a_tiny_run_writes_the_summary_the_predictions_and_the_markdown(tmp_path):
    out = tmp_path / "run.json"
    assert _run(["--model", "tiny", "--mode", "direct", "--input", str(REQUESTS), "--out", str(out), "--n-boot", "20", "--format", "plain"]) == 0
    summary = json.loads(out.read_text(encoding="utf-8"))
    predictions = [json.loads(line) for line in out.with_suffix(".predictions.jsonl").read_text(encoding="utf-8").splitlines()]
    markdown = out.with_suffix(".md").read_text(encoding="utf-8")
    assert summary["config"]["model"] == "tiny" and summary["config"]["mode"] == "direct" and summary["config"]["records"] == 8
    assert summary["config"]["prompt_version"] == "v0c-p1" and len(summary["config"]["input_sha256"]) == 64
    assert summary["validation"] == {"read": 8, "invalid": 0, "problems": {}}
    assert [p["id"] for p in predictions] == [f"fx-00{i}" for i in range(1, 9)]
    assert summary["summary"]["counts"]["scored"] == sum(len(p["questions"]) for p in predictions)
    assert "## By question type" in markdown and "tiny" in markdown
    assert summary["run"]["prompts"] == summary["summary"]["counts"]["questions"] and summary["run"]["seconds"] >= 0


def test_invalid_records_are_counted_and_skipped(tmp_path):
    records = [json.loads(line) for line in REQUESTS.read_text(encoding="utf-8").splitlines()]
    broken = copy.deepcopy(records[0])
    broken["id"] = "broken"
    broken["request"]["questions"] = {}
    path = tmp_path / "in.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in records[:3] + [broken]) + "\n", encoding="utf-8")
    out = tmp_path / "run.json"
    assert _run(["--model", "tiny", "--input", str(path), "--out", str(out), "--n-boot", "10"]) == 0
    summary = json.loads(out.read_text(encoding="utf-8"))
    assert summary["validation"]["read"] == 4 and summary["validation"]["invalid"] == 1 and "broken" in summary["validation"]["problems"]
    assert summary["config"]["records"] == 3


def test_limit_and_seed_pick_a_pair_preserving_sample(tmp_path):
    out = tmp_path / "run.json"
    assert _run(["--model", "tiny", "--input", str(REQUESTS), "--out", str(out), "--limit", "4", "--seed", "3", "--n-boot", "10"]) == 0
    summary = json.loads(out.read_text(encoding="utf-8"))
    ids = summary["config"]["record_ids"]
    assert 4 <= len(ids) <= 6  # 쌍 묶음(fx-001·002·003)은 통째로 들어온다
    if any(i in ids for i in ("fx-001", "fx-002", "fx-003")):
        assert {"fx-001", "fx-002", "fx-003"} <= set(ids)


def test_rescore_and_compare_run_on_the_cpu_from_saved_predictions(tmp_path):
    a, b = tmp_path / "a.json", tmp_path / "b.json"
    assert _run(["--model", "tiny", "--input", str(REQUESTS), "--out", str(a), "--n-boot", "10"]) == 0
    assert _run(["--model", "tiny", "--input", str(REQUESTS), "--out", str(b), "--n-boot", "10", "--shuffle-seed", "2"]) == 0
    rescored = tmp_path / "rescored.json"
    assert script().main(["--rescore", str(a.with_suffix(".predictions.jsonl")), "--input", str(REQUESTS), "--out", str(rescored), "--n-boot", "10"]) == 0
    first = json.loads(a.read_text(encoding="utf-8"))["summary"]
    again = json.loads(rescored.read_text(encoding="utf-8"))["summary"]
    assert again["overall"] == first["overall"] and again["counts"] == first["counts"]
    cmp_out = tmp_path / "cmp.json"
    assert script().main(["--compare", str(a.with_suffix(".predictions.jsonl")), str(b.with_suffix(".predictions.jsonl")), "--input", str(REQUESTS),
                          "--labels", "seed1", "seed2", "--out", str(cmp_out), "--n-boot", "10"]) == 0
    cmp = json.loads(cmp_out.read_text(encoding="utf-8"))
    assert cmp["comparison"]["records"] == 8 and "noul.log_loss" in cmp["comparison"]["differences"]
    assert "seed1" in cmp_out.with_suffix(".md").read_text(encoding="utf-8")


def test_the_run_refuses_an_unknown_mode_or_format(tmp_path):
    with pytest.raises(SystemExit):
        script().main(["--model", "tiny", "--mode", "slow", "--input", str(REQUESTS), "--out", str(tmp_path / "x.json")])
    with pytest.raises(SystemExit):
        script().main(["--model", "tiny", "--format", "xml", "--input", str(REQUESTS), "--out", str(tmp_path / "x.json")])


def test_several_shuffle_seeds_add_an_option_order_sensitivity_section(tmp_path):
    out = tmp_path / "run.json"
    assert _run(["--model", "tiny", "--input", str(REQUESTS), "--out", str(out), "--n-boot", "10", "--shuffle-seeds", "1", "2", "3"]) == 0
    summary = json.loads(out.read_text(encoding="utf-8"))
    questions = summary["summary"]["counts"]["questions"]
    perm = summary["summary"]["permutation"]
    assert summary["config"]["shuffle_seeds"] == [1, 2, 3] and summary["config"]["shuffle_seed"] == 1
    assert perm["_all"]["questions"] == questions and perm["_all"]["pairs"] == 3 * questions
    for seed in (2, 3):
        extra = [json.loads(line) for line in (tmp_path / f"run.predictions.seed{seed}.jsonl").read_text(encoding="utf-8").splitlines()]
        assert len(extra) == 8 and all(p["shuffle_seed"] == seed for p in extra)
    first = [json.loads(line) for line in out.with_suffix(".predictions.jsonl").read_text(encoding="utf-8").splitlines()]
    assert all(p["shuffle_seed"] == 1 for p in first)
    markdown = out.with_suffix(".md").read_text(encoding="utf-8")
    assert "## Option-order sensitivity" in markdown and "### Order-averaged reading" in markdown and "### Paired order effects" in markdown
    averaged = summary["summary"]["order_averaged"]
    assert averaged["orders"] == 3 and averaged["overall"]["_all"]["n"] == questions
    assert set(summary["summary"]["order_effects"]) == {"noul", "score", "choice"}
    rows = [json.loads(line) for line in (tmp_path / "run.predictions.order-averaged.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 8 and all(p["order_averaged"] == 3 for p in rows)
