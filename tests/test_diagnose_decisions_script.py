"""`scripts/diagnose_decisions.py` — 진단 실행기 (Task v0c §2.4).

소형 난수 Qwen(``--model tiny``, 검사 전용 — CPU)으로 실행기를 끝까지 돌려 세 산출물(요약 JSON, 레코드별 예측 JSONL, markdown)이
생기고 서로 맞는지, 형식이 틀린 기록·겹친 id는 세고 건너뛰는지, ``--limit``·``--seed``가 쌍 단위 표본을 고르는지, 균형 배치(기본)가
질문마다 배치를 모두 채점하고 두 읽기를 내는지, CPU 전용 ``--rescore``(저장한 예측 — 옛 seed run 여럿이면 서로 다른 배치로 합쳐 —
에서 요약 다시 계산)와 ``--compare``(두 run의 짝지은 비교, 두 읽기)를 본다. Task v0d: ``--checkpoint``가 없으면 예측이 손대지 않은
기반 모델을 직접 채점한 것과 비트 단위로 같다(``--checkpoint``가 있는 run은 tests/test_train_decisions_script.py). 실제 모델 run은 GPU 유닛의 일이다.
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
    """소형 모델로 실행기 돌리기 — 여기 시험은 배치 불변을 보지 않으므로 묶어서(배치 8) 빨리 돌린다."""
    with torch_reference_kernels():
        return script().main(argv + ([] if "--batch" in argv or "--model" not in argv else ["--batch", "8"]))


def test_a_tiny_run_writes_the_summary_the_predictions_and_the_markdown(tmp_path):
    out = tmp_path / "run.json"
    assert _run(["--model", "tiny", "--mode", "direct", "--input", str(REQUESTS), "--out", str(out), "--n-boot", "20", "--format", "plain"]) == 0
    summary = json.loads(out.read_text(encoding="utf-8"))
    predictions = [json.loads(line) for line in out.with_suffix(".predictions.jsonl").read_text(encoding="utf-8").splitlines()]
    markdown = out.with_suffix(".md").read_text(encoding="utf-8")
    assert summary["config"]["model"] == "tiny" and summary["config"]["mode"] == "direct" and summary["config"]["records"] == 8
    assert summary["config"]["prompt_version"] == "v0c-p3" and summary["config"]["orders"] == "counterbalanced" and len(summary["config"]["input_sha256"]) == 64
    assert summary["validation"] == {"read": 8, "invalid": 0, "problems": {}}
    assert [p["id"] for p in predictions] == [f"fx-00{i}" for i in range(1, 9)]
    readings = summary["summary"]["readings"]
    assert set(readings) == {"single_order", "order_averaged"}
    questions = sum(len(p["questions"]) for p in predictions)
    assert readings["single_order"]["counts"]["scored"] == readings["order_averaged"]["counts"]["scored"] == questions
    listings = sum(len(q["listings"]) for p in predictions for q in p["questions"].values())
    assert summary["run"]["prompts"] == listings == readings["single_order"]["counts"]["listing_rows"] and summary["run"]["seconds"] >= 0
    assert "## By question type (both readings)" in markdown and "tiny" in markdown


def test_invalid_records_and_duplicate_ids_are_counted_and_skipped(tmp_path):
    records = [json.loads(line) for line in REQUESTS.read_text(encoding="utf-8").splitlines()]
    broken = copy.deepcopy(records[0])
    broken["id"] = "broken"
    broken["request"]["questions"] = {}
    duplicate = copy.deepcopy(records[1])  # 리뷰 1 Minor 7: 같은 id가 두 번이면 뒤의 것을 건너뛴다 (예측이 섞이지 않게)
    path = tmp_path / "in.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in records[:3] + [broken, duplicate]) + "\n", encoding="utf-8")
    out = tmp_path / "run.json"
    assert _run(["--model", "tiny", "--input", str(path), "--out", str(out), "--n-boot", "10", "--orders", "seeded"]) == 0
    summary = json.loads(out.read_text(encoding="utf-8"))
    assert summary["validation"]["read"] == 5 and summary["validation"]["invalid"] == 2 and "broken" in summary["validation"]["problems"]
    assert any("duplicate id" in problem for problem in summary["validation"]["problems"][records[1]["id"]])
    assert summary["config"]["records"] == 3


def test_limit_and_seed_pick_a_pair_preserving_sample(tmp_path):
    out = tmp_path / "run.json"
    assert _run(["--model", "tiny", "--input", str(REQUESTS), "--out", str(out), "--limit", "4", "--seed", "3", "--n-boot", "10", "--orders", "seeded"]) == 0
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
    first = json.loads(a.read_text(encoding="utf-8"))["summary"]["readings"]
    again = json.loads(rescored.read_text(encoding="utf-8"))["summary"]["readings"]
    for reading in ("single_order", "order_averaged"):
        assert again[reading]["overall"] == first[reading]["overall"] and again[reading]["counts"] == first[reading]["counts"]
    cmp_out = tmp_path / "cmp.json"
    assert script().main(["--compare", str(a.with_suffix(".predictions.jsonl")), str(b.with_suffix(".predictions.jsonl")), "--input", str(REQUESTS),
                          "--labels", "seed1", "seed2", "--out", str(cmp_out), "--n-boot", "10"]) == 0
    cmp = json.loads(cmp_out.read_text(encoding="utf-8"))
    assert cmp["comparison"]["records"] == 8 and "noul.log_loss" in cmp["comparison"]["readings"]["order_averaged"]["differences"]
    assert "seed1" in cmp_out.with_suffix(".md").read_text(encoding="utf-8")


def test_rescoring_several_seeded_runs_merges_their_distinct_listings(tmp_path):
    """옛 미리보기(round 0)처럼 seed 배치 하나짜리 run 여럿 → 질문마다 서로 다른 배치로 합쳐 두 읽기를 낸다; 같은 배치는 한 번만."""
    one, two = tmp_path / "one.json", tmp_path / "two.json"
    assert _run(["--model", "tiny", "--input", str(REQUESTS), "--out", str(one), "--n-boot", "10", "--orders", "seeded", "--shuffle-seed", "1"]) == 0
    assert _run(["--model", "tiny", "--input", str(REQUESTS), "--out", str(two), "--n-boot", "10", "--orders", "seeded", "--shuffle-seed", "2"]) == 0
    merged = tmp_path / "merged.json"
    files = [str(one.with_suffix(".predictions.jsonl")), str(two.with_suffix(".predictions.jsonl")), str(one.with_suffix(".predictions.jsonl"))]
    assert script().main(["--rescore", *files, "--input", str(REQUESTS), "--out", str(merged), "--n-boot", "10"]) == 0
    summary = json.loads(merged.read_text(encoding="utf-8"))["summary"]
    skipped = summary["order"]["listings"]["merge"]["duplicate_listings_skipped"]
    assert sum(skipped.values()) >= 33  # 셋째 파일은 첫째와 같다 — 33문항 모두 건너뛴다(둘째와 겹친 배치도 더한다)
    rows = [json.loads(line) for line in merged.with_suffix(".predictions.jsonl").read_text(encoding="utf-8").splitlines()]
    assert all(1 <= q["n_listings"] <= 2 for p in rows for q in p["questions"].values())
    assert summary["order"]["permutation"]["_all"]["identical_pairs_skipped"] == 0


def test_the_run_refuses_an_unknown_mode_or_format(tmp_path):
    with pytest.raises(SystemExit):
        script().main(["--model", "tiny", "--mode", "slow", "--input", str(REQUESTS), "--out", str(tmp_path / "x.json")])
    with pytest.raises(SystemExit):
        script().main(["--model", "tiny", "--format", "xml", "--input", str(REQUESTS), "--out", str(tmp_path / "x.json")])
    with pytest.raises(SystemExit):
        script().main(["--model", "tiny", "--orders", "random", "--input", str(REQUESTS), "--out", str(tmp_path / "x.json")])


def test_the_counterbalanced_default_scores_every_listing_and_counts_only_distinct_pairs(tmp_path):
    out = tmp_path / "run.json"
    assert _run(["--model", "tiny", "--input", str(REQUESTS), "--out", str(out), "--n-boot", "10"]) == 0
    summary = json.loads(out.read_text(encoding="utf-8"))["summary"]
    listings = summary["order"]["listings"]
    assert listings["noul"]["mean_listings"] == 2.0 and listings["score"]["mean_listings"] == 2.0 and listings["noul"]["multi_listing_rate"] == 1.0
    perm = summary["order"]["permutation"]
    assert perm["noul"]["pairs"] == perm["noul"]["questions"] == 11 and perm["_all"]["identical_pairs_skipped"] == 0
    assert set(summary["order"]["order_effects"]) == {"noul", "score", "choice"}
    markdown = out.with_suffix(".md").read_text(encoding="utf-8")
    assert "### Verdict changes between listings" in markdown and "### Paired order effects" in markdown


def test_several_shuffle_seeds_need_seeded_orders_and_are_merged(tmp_path):
    out = tmp_path / "run.json"
    with pytest.raises(SystemExit):
        _run(["--model", "tiny", "--input", str(REQUESTS), "--out", str(out), "--n-boot", "10", "--shuffle-seeds", "1", "2"])
    assert _run(["--model", "tiny", "--input", str(REQUESTS), "--out", str(out), "--n-boot", "10", "--orders", "seeded", "--shuffle-seeds", "1", "2", "3"]) == 0
    summary = json.loads(out.read_text(encoding="utf-8"))
    assert summary["config"]["shuffle_seeds"] == [1, 2, 3] and summary["summary"]["order"]["listings"]["merge"]["runs"] == 3
    rows = [json.loads(line) for line in out.with_suffix(".predictions.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 8 and all(q["n_listings"] == len({tuple(l["options"]) for l in q["listings"]}) for p in rows for q in p["questions"].values())


def test_without_a_checkpoint_the_run_is_bit_identical_to_scoring_the_untouched_base_model(tmp_path):
    """Task v0d: ``--checkpoint``를 더한 뒤에도 그것이 없으면 실행기는 v0c와 같은 일만 한다 — 손대지 않은 기반 모델을
    :func:`robo_jev.decisions.zero_shot.diagnose_records` 로 직접 채점한 것과 예측이 비트 단위로 같다(JSON을 거친 float까지)."""
    from robo_jev.decisions.records import read_records
    from robo_jev.decisions.zero_shot import diagnose_records
    from robo_jev.model.backbone_qwen import QwenBackbone
    from robo_jev.model.tokenizer import available_tokenizer, load_tokenizer

    out = tmp_path / "run.json"
    assert _run(["--model", "tiny", "--input", str(REQUESTS), "--out", str(out), "--n-boot", "10", "--batch", "1"]) == 0
    summary = json.loads(out.read_text(encoding="utf-8"))
    assert summary["config"]["checkpoint"] is None and summary["config"]["model"] == "tiny"
    saved = [json.loads(line) for line in out.with_suffix(".predictions.jsonl").read_text(encoding="utf-8").splitlines()]
    tokenizer = load_tokenizer(available_tokenizer()[0])
    with torch_reference_kernels():
        direct = diagnose_records(read_records(REQUESTS), QwenBackbone.tiny(seed=4, vocab_size=tokenizer.get_vocab_size()), tokenizer, model_id="tiny",
                                  mode="direct", fmt="chat", seed=1, orders="counterbalanced", batch=1)
    assert saved == json.loads(json.dumps(direct))


def test_the_prompt_version_flag_reproduces_p2_and_p3_is_the_default(tmp_path):
    """Task v0d fix round 1: 기본 판은 v0c-p3; ``--prompt-version v0c-p2``는 v0c 결과의 재현용 — choice·noul의 예측은 두 판에서 같고(글이 바이트
    단위로 같다), score만 다르다(6구간 대 기준점 11개)."""
    p2, p3 = tmp_path / "p2.json", tmp_path / "p3.json"
    assert _run(["--model", "tiny", "--input", str(REQUESTS), "--out", str(p2), "--n-boot", "10", "--prompt-version", "v0c-p2", "--batch", "1"]) == 0
    assert _run(["--model", "tiny", "--input", str(REQUESTS), "--out", str(p3), "--n-boot", "10", "--batch", "1"]) == 0
    c2, c3 = (json.loads(path.read_text(encoding="utf-8"))["config"] for path in (p2, p3))
    assert (c2["prompt_version"], c3["prompt_version"]) == ("v0c-p2", "v0c-p3") and c2["prompt_digest"] != c3["prompt_digest"]
    rows2 = [json.loads(line) for line in p2.with_suffix(".predictions.jsonl").read_text(encoding="utf-8").splitlines()]
    rows3 = [json.loads(line) for line in p3.with_suffix(".predictions.jsonl").read_text(encoding="utf-8").splitlines()]
    for a, b in zip(rows2, rows3):
        assert (a["prompt_version"], b["prompt_version"]) == ("v0c-p2", "v0c-p3")
        for name, q in a["questions"].items():
            other = b["questions"][name]
            if q["type"] == "score":
                assert len(q["listings"][0]["codes"]) == 6 and len(other["listings"][0]["codes"]) == 11
            else:
                assert q["listings"] == other["listings"]
    with pytest.raises(SystemExit):
        script().main(["--model", "tiny", "--prompt-version", "v0c-p1", "--input", str(REQUESTS), "--out", str(tmp_path / "x.json")])

