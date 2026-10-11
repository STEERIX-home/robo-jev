"""`scripts/bench.py` (Task bench0) — build → dry-run → run(가짜 서버) → score를 작은 정의로 끝까지 돈다. 유료 호출은 없다:
run은 숨은 ``--api-url``로 127.0.0.1의 가짜 서버에 보내고, 키 파일·장부·캐시는 tmp에 둔다."""

import functools
import importlib.util
import json
import os
import shutil
import sys

import pytest
import yaml
from helpers import FIXTURES, REPO
from test_bench_openrouter import KEY, FakeServer

SCRIPT = REPO / "scripts" / "bench.py"
REQUESTS = FIXTURES / "decisions" / "requests.jsonl"


@functools.lru_cache(maxsize=1)
def script():
    spec = importlib.util.spec_from_file_location("bench_script", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _setup(tmp_path):
    shutil.copy(REQUESTS, tmp_path / "T1.jsonl")
    shutil.copy(REQUESTS, tmp_path / "G2.jsonl")
    config = {
        "version": "bench-test", "seed": 1, "strata": ["language"],
        "sets": {"T1": {"source": "T1.jsonl", "unit": "record", "count": 3}, "G2": {"source": "G2.jsonl", "unit": "all"}},
        "pilot": {"from": "T1", "count": 2},
        "models": {"default": [{"id": "vendor/a", "why": "test"}, {"id": "vendor/b", "why": "test"}], "exclude_suffixes": [":free"]},
        "ours": [],
    }
    (tmp_path / "bench.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
    snapshot = {"data": [{"id": "vendor/a", "pricing": {"prompt": "0.000000042", "completion": "0"}, "context_length": 32768},
                         {"id": "vendor/b", "pricing": {"prompt": "0.0000001", "completion": "0"}, "context_length": 8192},
                         {"id": "vendor/b:free", "pricing": {"prompt": "0", "completion": "0"}, "context_length": 8192}]}
    (tmp_path / "models.json").write_text(json.dumps(snapshot), encoding="utf-8")
    key = tmp_path / "openrouter.env"
    key.write_text(f"OPENROUTER_API_KEY={KEY}\n", encoding="utf-8")
    os.chmod(key, 0o600)
    common = ["--config", str(tmp_path / "bench.yaml"), "--root", str(tmp_path)]
    assert script().main(common + ["build", "--out", str(tmp_path / "out")]) == 0
    return common


def _counts(tmp_path) -> dict:
    manifest = json.loads((tmp_path / "out" / "manifest.json").read_text(encoding="utf-8"))
    return {"T1": manifest["sets"]["T1"]["records"], "G2": manifest["sets"]["G2"]["records"], "pilot": manifest["pilot"]["records"]}


def test_dry_run_estimates_without_a_key_or_a_call(tmp_path, capsys):
    common = _setup(tmp_path)
    assert script().main(common + ["dry-run", "--out", str(tmp_path / "out"), "--snapshot", str(tmp_path / "models.json")]) == 0
    report = json.loads((tmp_path / "out" / "dry-run.json").read_text(encoding="utf-8"))
    counts = _counts(tmp_path)
    assert [row["model"] for row in report["models"]] == ["vendor/a", "vendor/b"]
    assert 3 <= counts["T1"] < 8 and report["sets"]["T1"]["requests"] == counts["T1"] and report["sets"]["G2"]["requests"] == 8
    row = report["models"][1]
    assert row["requests"] == counts["T1"] + 8 and row["cost_high"] == pytest.approx(row["tokens_high"] * 1e-7) and row["cost_low"] < row["cost_high"]
    assert report["total"]["all_high"] == pytest.approx(sum(r["cost_high"] + r["pilot_cost_high"] for r in report["models"]))
    assert "TOTAL" in capsys.readouterr().out


def test_run_writes_predictions_ledger_and_reuses_the_cache_then_score_reads_them(tmp_path):
    common = _setup(tmp_path)
    out = tmp_path / "out"
    paid = ["--out", str(out), "--snapshot", str(tmp_path / "models.json"), "--key-file", str(tmp_path / "openrouter.env"),
            "--ledger", str(tmp_path / "ledger.json"), "--cache", str(tmp_path / "cache"), "--calls-log", str(tmp_path / "calls.jsonl"),
            "--max-cost-usd", "0.05", "--max-cost-per-model-usd", "0.03", "--concurrency", "2"]
    with FakeServer() as server:
        server_url = server.url
        assert script().main(common + ["run", *paid, "--api-url", server_url]) == 0
        first = len(server.requests)
        assert script().main(common + ["run", *paid, "--api-url", server_url]) == 0  # 다시 돌리면 캐시 — 호출 0
        # 이 시험의 G2는 T1과 같은 fixture라 같은 (모델, id, 요청)이 겹친다 — 겹친 것은 캐시에서 온다: 모델마다 서로 다른 레코드 8개만 보낸다
        assert len(server.requests) == first == 2 * 8
    ledger = json.loads((tmp_path / "ledger.json").read_text(encoding="utf-8"))
    assert ledger["totals"]["calls"] == first and 0 < ledger["totals"]["spent_usd"] <= 0.05 and len(ledger["runs"]) == 2
    assert ledger["runs"][1]["spent_usd"] == 0.0
    for model in ("vendor__a", "vendor__b"):
        lines = (out / "api" / model / "T1.predictions.jsonl").read_text(encoding="utf-8").splitlines()
        assert len(lines) == _counts(tmp_path)["T1"] and json.loads(lines[0])["api"]["model"].endswith("-20261012")
        assert json.loads((out / "api" / model / "T1.run.json").read_text(encoding="utf-8"))["written"] == _counts(tmp_path)["T1"]
    texts = [p.read_text(encoding="utf-8") for p in tmp_path.rglob("*") if p.is_file() and p.name != "openrouter.env"]
    assert all(KEY not in text for text in texts)
    assert script().main(common + ["score", "--out", str(out), "--snapshot", str(tmp_path / "models.json"), "--n-boot", "20"]) == 0
    results = json.loads((out / "results.json").read_text(encoding="utf-8"))
    assert [m["model_id"] for m in results["models"]] == ["vendor/a", "vendor/b"]
    assert results["models"][1]["price_per_m_input"] == pytest.approx(0.1)
    assert results["models"][0]["sets"]["G2"]["readings"]["single_order"]["coverage"] == 1.0
    assert "| vendor/a |" in (out / "results.md").read_text(encoding="utf-8")
    detail = (out / "results-detail.md").read_text(encoding="utf-8")
    assert "Value-dependent: action accuracy" in detail and "General reading" in detail and detail.count("| vendor/b |") >= 10


def test_paid_commands_need_both_caps_and_refuse_a_spent_ledger(tmp_path):
    common = _setup(tmp_path)
    with pytest.raises(SystemExit):
        script().main(common + ["run", "--out", str(tmp_path / "out"), "--max-cost-usd", "1"])  # 모델별 상한이 없다
    (tmp_path / "ledger.json").write_text(json.dumps({"models": {"vendor/a": {"spent_usd": 2.0}}}), encoding="utf-8")
    with pytest.raises(SystemExit, match="already holds"):
        script().main(common + ["run", "--out", str(tmp_path / "out"), "--snapshot", str(tmp_path / "models.json"),
                                "--key-file", str(tmp_path / "openrouter.env"), "--ledger", str(tmp_path / "ledger.json"),
                                "--max-cost-usd", "1", "--max-cost-per-model-usd", "1", "--api-url", "http://127.0.0.1:9/never"])


def test_the_pilot_sends_score_questions_in_both_formats(tmp_path):
    common = _setup(tmp_path)
    out = tmp_path / "out"
    paid = ["--out", str(out), "--snapshot", str(tmp_path / "models.json"), "--key-file", str(tmp_path / "openrouter.env"),
            "--ledger", str(tmp_path / "ledger.json"), "--cache", str(tmp_path / "cache"), "--calls-log", str(tmp_path / "calls.jsonl"),
            "--max-cost-usd", "0.05", "--max-cost-per-model-usd", "0.03", "--models", "vendor/a"]
    with FakeServer() as server:
        assert script().main(common + ["pilot", *paid, "--api-url", server.url]) == 0
        sent = [r["body"] for r in server.requests]
    pilot = _counts(tmp_path)["pilot"]
    assert pilot >= 2 and len(sent) == 2 * pilot and all(q["type"] == "score" for body in sent for q in body["questions"].values())
    assert sorted(len(next(iter(body["questions"].values()))["criteria"]) for body in sent) == [2] * pilot + [11] * pilot
    report = json.loads((out / "pilot" / "report.json").read_text(encoding="utf-8"))
    assert set(report["models"]["vendor/a"]["variants"]) == {"score2", "score11"}
    assert (out / "pilot" / "report.md").read_text(encoding="utf-8").count("| vendor/a |") == 1
