"""일반 능력 시험 G1 — 옛 비로봇 단일 요청(`judgment-v0`) → 판단 요청 기록 변환, 두 run의 묶음별 짝지은 비교, 섞기용 학습 파일 (Task v0.1b).

변환 규칙(:mod:`robo_jev.decisions.single_probe`)을 작은 손 레코드로 하나씩 고정하고, 변환본을 진단 도구(소형 난수 Qwen, CPU)와 학습기의
데이터 적재기·예 빌더가 그대로 읽는지 본다. 실제 산출물(`artifacts/simgen/probes/`)이 있으면 manifest의 sha256과 봉인 G1 ↔ 섞기용의
origin group 겹침 0도 확인한다(없으면 건너뛴다).
"""

from __future__ import annotations

import copy
import functools
import importlib.util
import json
import sys
from pathlib import Path

import pytest
from helpers import FIXTURES, REPO

from robo_jev.decisions.metrics import block_of, question_rows
from robo_jev.decisions.records import file_sha256, read_records, validate_record
from robo_jev.decisions.single_probe import (
    G1_SPLITS,
    MIX_SPLITS,
    compare_probe,
    convert_record,
    convert_records,
    overlap,
    parse_state_text,
    partition,
    render_probe_comparison,
    state_text,
    trim_options,
)

SCRIPT = REPO / "scripts" / "convert_single_probe.py"
REQUESTS = FIXTURES / "decisions" / "requests.jsonl"
PROBES = REPO / "artifacts" / "simgen" / "probes"

STATE = {
    "observed_at_ms": 3700,
    "goal": {"text": "초록 물체를 오른쪽 영역으로 옮긴다", "color": "green", "zone": "zoneR"},
    "thresholds": {"stale_after_ms": 300, "distance_edges_mm": [250, 500, 750]},
    "glossary": {"근접역": "원점에서 250mm 이내", "note": "a: b — with a colon\nand a newline"},
    "objects": [{"id": "o1", "desc": "초록 물체 o1", "x_mm": -565, "visible": True, "age_ms": 40}, {"id": "o2", "desc": "빨간 물체 o2", "x_mm": -25, "visible": False}],
    "resources": [{"id": "r1", "desc": "개발 담당", "available_h": 6}],
    "empty": [],
    "flags": [1, 2.5, None, "x"],
}


def judgment(questions: list[dict], labels: list[dict], *, rid: str = "spatial-0001-0", split: str = "test", origin: str = "spatial/zone-color/0001",
             domain: str = "spatial", language: str = "ko", state: dict | None = None) -> dict:  # fmt: skip
    return {
        "schema_version": "judgment-v0", "origin_group": origin, "split": split,
        "request": {"request_id": rid, "state": copy.deepcopy(state or STATE), "questions": questions},
        "labels": labels,
        "provenance": {"generator": "gen-single-v0.2.0", "config_version": "pilot-v0.3", "domain": domain, "language": language, "template": "zone-color",
                       "seed": 17, "variants": ["near_miss"], "holdout": [], "concepts": ["spatial:nearest"], "contrast": None},
        "evidence": {}, "usage": {},
    }  # fmt: skip


def choice(qid: str, options: list[tuple[str, str, str | None]], text: str = "가장 가까운 초록 물체를 고르라.") -> dict:
    criteria = []
    for cid, description, ref in options:
        criteria.append({"id": cid, "description": description, **({"ref": ref} if ref else {})})
    return {"id": qid, "type": "choice", "instructions": text, "criteria": criteria}


def boolean(qid: str, text: str = "o1이 오른쪽 영역에 있는가.") -> dict:
    return {"id": qid, "type": "boolean", "instructions": text, "criteria": [{"id": "true", "description": "그렇다"}, {"id": "false", "description": "아니다"}]}


def ordinal(qid: str, order: list[int] | None = None) -> dict:
    levels = [{"id": str(v), "description": name, "value": float(v)} for v, name in enumerate(["근접역 안", "가까움", "보통", "멂"])]
    return {"id": qid, "type": "ordinal", "instructions": "가장 가까운 초록 물체의 거리 수준을 고르라.", "criteria": [levels[i] for i in (order or range(4))]}


def single(qid: str, answer, source: str = "spatial/nearest-observed-v0") -> dict:
    return {"question_id": qid, "kind": "single", "answer": answer, "source": source, "label_confidence": "high"}


def valid_set(qid: str, ids: list[str], source: str = "spatial/distance-level-v0") -> dict:
    return {"question_id": qid, "kind": "valid_set", "candidate_ids": ids, "source": source, "label_confidence": "medium"}


@functools.lru_cache(maxsize=1)
def script():
    spec = importlib.util.spec_from_file_location("convert_single_probe", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def mixed_record() -> dict:
    questions = [
        choice("q_nearest_0", [("c29", "초록 물체 o1", "o1"), ("c61", "빨간 물체 o2", "o2"), ("c79", "후보 중에 답이 없거나 정보가 부족함", None)]),
        choice("q_owner_1", [("c16", "개발 담당", "r1"), ("c31", "QA 담당", "r3")], text="빌드 (s2)를 수행할 자원을 고르라."),
        boolean("q_in_zone_2"),
        ordinal("q_distance_3", order=[2, 0, 3, 1]),
        boolean("q_masked_4"),
        choice("q_lonely_5", [("c11", "초록 물체 o1", "o1")]),
    ]
    labels = [single("q_nearest_0", "c29"), single("q_owner_1", "c16", source="workflow/step-owner-v0"), single("q_in_zone_2", True),
              valid_set("q_distance_3", ["1", "2"]), single("q_lonely_5", "c11")]  # fmt: skip
    return judgment(questions, labels)


# --------------------------------------------------------------------------
# 상태 글
# --------------------------------------------------------------------------


def test_the_state_text_keeps_every_field_and_reads_back_exactly():
    text = state_text(STATE)
    assert parse_state_text(text) == STATE
    lines = text.split("\n")
    assert lines[0] == "observed_at_ms: 3700" and "objects:" in lines and "empty: []" in lines
    assert lines[lines.index("objects:") + 1].startswith('- {"id": "o1", "desc": "초록 물체 o1"')  # 객체 목록은 항목마다 한 줄, 한글 그대로
    assert '"distance_edges_mm": [250, 500, 750]' in text and "근접역" in text  # thresholds·glossary도 글에 있다
    assert all("\n" not in line for line in lines) and len(lines) == 1 + 1 + 1 + 1 + 3 + 2 + 1 + 1


@pytest.mark.parametrize("key", ["a: b", "end:", "- item", "two\nlines", ""])
def test_keys_that_would_break_the_line_format_are_refused(key):
    with pytest.raises(ValueError):
        state_text({key: 1})


# --------------------------------------------------------------------------
# 레코드 변환
# --------------------------------------------------------------------------


def test_a_judgment_record_becomes_a_valid_decisions_request():
    (record,), counts = convert_record(mixed_record())
    assert validate_record(record) == []
    assert record["schema"] == "simgen-request/0" and record["id"] == "spatial-0001-0"
    questions, labels = record["request"]["questions"], record["labels"]
    assert list(questions) == ["q_nearest_0", "q_owner_1", "q_in_zone_2", "q_distance_3"]  # 마스크·1개짜리는 빠지고 순서는 그대로
    assert questions["q_nearest_0"] == {"type": "choice", "instructions": "가장 가까운 초록 물체를 고르라.",
                                        "criteria": {"c29": "초록 물체 o1", "c61": "빨간 물체 o2", "c79": "후보 중에 답이 없거나 정보가 부족함"}}  # fmt: skip
    assert questions["q_owner_1"]["criteria"] == {"c16": "개발 담당 (ref=r1)", "c31": "QA 담당 (ref=r3)"}  # 설명에 없는 ref는 붙인다
    assert questions["q_in_zone_2"] == {"type": "noul", "instructions": "o1이 오른쪽 영역에 있는가."}
    assert list(questions["q_distance_3"]["criteria"]) == ["0", "1", "2", "3"]  # ordinal은 수준 순서
    assert labels["q_nearest_0"] == {"type": "choice", "kind": "probe", "probabilities": {"c29": 1.0, "c61": 0.0, "c79": 0.0}}
    assert labels["q_in_zone_2"] == {"type": "noul", "p_yes": 1.0}
    assert labels["q_distance_3"]["probabilities"] == {"0": 0.0, "1": 0.5, "2": 0.5, "3": 0.0}  # 허용 집합에 고르게
    assert parse_state_text(record["request"]["state"]) == STATE
    meta = record["meta"]
    assert (meta["domain"], meta["language"], meta["split"], meta["origin_group"], meta["source_record"]) == (
        "spatial", "ko", "test", "spatial/zone-color/0001", "spatial-0001-0")
    assert meta["questions"]["q_distance_3"] == {"source_type": "ordinal", "source": "spatial/distance-level-v0", "family": "spatial/distance-level-v0",
                                                 "label_kind": "valid_set", "label_confidence": "medium", "options_trimmed": None}  # fmt: skip
    assert meta["questions"]["q_in_zone_2"]["source_type"] == "boolean" and "pair" not in meta
    assert counts["questions_in"] == 6 and counts["questions_out"] == 4 and counts["q:masked"] == 1 and counts["q:too_few_options"] == 1


def test_a_false_boolean_is_p_yes_zero_and_a_valid_set_choice_shares_the_label():
    questions = [boolean("q_b"), choice("q_c", [("c1", "o1", "o1"), ("c2", "o2", "o2"), ("c3", "none", None)])]
    (record,), _ = convert_record(judgment(questions, [single("q_b", False), valid_set("q_c", ["c1", "c3"])]))
    assert record["labels"]["q_b"]["p_yes"] == 0.0
    assert record["labels"]["q_c"]["probabilities"] == {"c1": 0.5, "c2": 0.0, "c3": 0.5}


def test_a_long_candidate_list_is_cut_to_eight_keeping_the_answer_and_the_none_option():
    options = [(f"c{i:02d}", f"요소 e{i}", f"e{i}") for i in range(1, 13)] + [("c99", "해당하는 후보가 없거나 주어진 정보로 결정할 수 없음", None)]
    question = choice("q_action_0", options, text="목표를 위해 지금 누를 요소를 고르라.")
    for i in range(50):  # 고정 seed 하나로는 "없음"이 우연히 살아남을 수 있다(리뷰 1 M1) — 레코드 id 50개 모두에서 정답·"없음"이 남아야 한다
        (each,), _ = convert_record(judgment([question], [single("q_action_0", "c07")], rid=f"dom-{i:04d}-1", domain="dom"))
        names = list(each["request"]["questions"]["q_action_0"]["criteria"])
        assert len(names) == 8 and "c07" in names and "c99" in names, (i, names)
    (record,), counts = convert_record(judgment([question], [single("q_action_0", "c07")], rid="dom-0003-1", domain="dom"))
    kept = list(record["request"]["questions"]["q_action_0"]["criteria"])
    assert len(kept) == 8 and "c07" in kept and "c99" in kept
    assert kept == [c for c, _, _ in options if c in kept]  # 원래 순서
    assert record["meta"]["questions"]["q_action_0"]["options_trimmed"] == [13, 8] and counts["trimmed"] == 1
    assert record["labels"]["q_action_0"]["probabilities"]["c07"] == 1.0 and validate_record(record) == []
    (again,), _ = convert_record(judgment([question], [single("q_action_0", "c07")], rid="dom-0003-1", domain="dom"))
    assert list(again["request"]["questions"]["q_action_0"]["criteria"]) == kept  # (레코드, 질문)으로 정한 seed — 결정적


def test_trimming_refuses_when_the_options_it_must_keep_do_not_fit():
    import random

    criteria = [{"id": f"c{i}", "description": "none"} for i in range(10)]  # ref 없는 후보 10개 — 남겨야 할 것이 8개를 넘는다
    with pytest.raises(ValueError):
        trim_options(criteria, ["c0"], limit=8, rng=random.Random(0))
    assert trim_options(criteria[:5], ["c0"], limit=8, rng=random.Random(0)) == criteria[:5]


def test_more_than_twelve_questions_split_evenly_over_records_with_the_same_state():
    questions = [boolean(f"q_b_{i}") for i in range(16)]
    records, counts = convert_record(judgment(questions, [single(f"q_b_{i}", i % 2 == 0) for i in range(16)]))
    assert [r["id"] for r in records] == ["spatial-0001-0.p1", "spatial-0001-0.p2"] and counts["parts"] == 2
    assert [len(r["request"]["questions"]) for r in records] == [8, 8]
    assert records[0]["request"]["state"] == records[1]["request"]["state"]
    assert [(r["meta"]["part"], r["meta"]["parts"]) for r in records] == [(1, 2), (2, 2)]
    assert all(validate_record(r) == [] for r in records)
    assert partition([str(i) for i in range(13)]) == [[str(i) for i in range(7)], [str(i) for i in range(7, 13)]]
    assert partition([str(i) for i in range(12)]) == [[str(i) for i in range(12)]] and partition([]) == []


def test_a_record_without_any_labelled_question_is_dropped_and_counted():
    records, counts = convert_record(judgment([boolean("q_b")], []))
    assert records == [] and counts["r:no_labeled_question"] == 1 and counts["q:masked"] == 1


def test_only_the_asked_splits_are_converted_and_the_summary_counts_them():
    a = mixed_record()
    b = judgment([boolean("q_b")], [single("q_b", True)], rid="rules-0002-0", split="train", origin="rules/x/0002", domain="rules", language="en")
    converted, stats = convert_records([a, b], splits=G1_SPLITS)
    assert [r["id"] for r in converted] == ["spatial-0001-0"] and stats["source_records"] == 1 and stats["questions"] == 4
    assert stats["questions_by"]["source_type"] == {"boolean": 1, "choice": 2, "ordinal": 1} and stats["questions_by"]["type"] == {"choice": 3, "noul": 1}
    mix, _ = convert_records([a, b], splits=MIX_SPLITS)
    assert [r["id"] for r in mix] == ["rules-0002-0"] and overlap(converted, mix) == {"origin_groups": [], "source_records": []}


def test_a_non_judgment_record_is_refused():
    with pytest.raises(ValueError, match="judgment-v0"):
        convert_record({**mixed_record(), "schema_version": "stream-v0"})


# --------------------------------------------------------------------------
# 진단 지표·학습기·진단 도구가 그대로 읽는다
# --------------------------------------------------------------------------


def test_probe_choices_get_their_own_metric_block_and_other_kinds_are_still_refused():
    assert block_of("choice", "probe") == "choice/other"
    (record,), _ = convert_record(mixed_record())
    predictions = [{"id": record["id"], "questions": {
        name: {"type": q["type"], "answer": ({"p_yes": 0.7} if q["type"] == "noul" else {"probabilities": {k: 1 / len(q["criteria"]) for k in q["criteria"]}})}
        for name, q in record["request"]["questions"].items()}}]  # fmt: skip
    rows = question_rows([record], predictions)
    assert {r["block"] for r in rows} == {"choice/other", "noul"} and all(r["status"] == "scored" for r in rows)
    broken = copy.deepcopy(record)
    broken["labels"]["q_nearest_0"]["kind"] = "nonsense"
    assert any("kind 'nonsense'" in p for p in validate_record(broken))


def test_the_trainer_loads_converted_records_and_builds_examples_whose_targets_follow_the_labels(tmp_path):
    from robo_jev.decisions.examples import build_examples, dev_examples
    from robo_jev.decisions.train import load_training_records
    from robo_jev.model.tokenizer import available_tokenizer, load_tokenizer

    records, _ = convert_record(mixed_record())
    path = tmp_path / "mix.jsonl"
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records), encoding="utf-8")
    loaded, check = load_training_records(path)
    assert check["invalid"] == 0 and [r["id"] for r in loaded] == [r["id"] for r in records]
    tokenizer = load_tokenizer(available_tokenizer()[0])
    record = loaded[0]
    for name in record["request"]["questions"]:
        (example,) = build_examples(record, [name], tokenizer, order_seed=7, fmt="chat")
        assert abs(sum(example["target"]) - 1.0) < 1e-9 and "Situation:\nobserved_at_ms: 3700" in example["text"]
        label = record["labels"][name]
        expected = [label["p_yes"] if k == "yes" else 1 - label["p_yes"] for k in example["options"]] if example["type"] == "noul" else [
            label["probabilities"][k] for k in example["options"]]
        assert example["target"] == pytest.approx(expected)
    dev = dev_examples(loaded, tokenizer, fmt="chat")
    assert len(dev) == 3 + 2 + 2 + 4  # choice 3·2·4개는 순환 배치마다, noul은 둘


def test_the_diagnosis_script_scores_a_converted_file_as_is(tmp_path):
    pytest.importorskip("transformers")
    from robo_jev.model.backbone_qwen import torch_reference_kernels

    spec = importlib.util.spec_from_file_location("diagnose_decisions_probe", REPO / "scripts" / "diagnose_decisions.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    records = convert_records([mixed_record()], splits=G1_SPLITS)[0]
    path = tmp_path / "G1.jsonl"
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records), encoding="utf-8")
    out = tmp_path / "run.json"
    with torch_reference_kernels():
        assert module.main(["--model", "tiny", "--input", str(path), "--out", str(out), "--n-boot", "10", "--batch", "8"]) == 0
    summary = json.loads(out.read_text(encoding="utf-8"))
    assert summary["validation"]["invalid"] == 0 and summary["config"]["records"] == 1
    counts = summary["summary"]["readings"]["single_order"]["counts"]
    assert counts["scored"] == 4 and counts["by_block"] == {"choice/other": 3, "noul": 1} and counts["listing_rows"] == 3 + 2 + 2 + 4


# --------------------------------------------------------------------------
# 두 run의 묶음별 짝지은 비교
# --------------------------------------------------------------------------


def _probe_records() -> list[dict]:
    out = []
    for i, (domain, language, origin) in enumerate((("spatial", "ko", "g/a"), ("rules", "en", "g/b"), ("rules", "en", "g/b"))):
        questions = [choice("q_c", [("c1", "o1", "o1"), ("c2", "o2", "o2"), ("c3", "none", None)]), boolean("q_b")]
        record = judgment(questions, [single("q_c", "c1"), single("q_b", True)], rid=f"r{i}", origin=origin, domain=domain, language=language)
        out.extend(convert_record(record)[0])
    return out


def _predictions(records: list[dict], *, right: bool, confidence: float = 0.9) -> list[dict]:
    """정답 쪽(`right`) 또는 오답 쪽에 `confidence`를 둔 예측 — 균형 배치(choice 순환 이동, noul 두 순서)를 흉내 낸다."""
    out = []
    for record in records:
        questions = {}
        for name, question in record["request"]["questions"].items():
            label = record["labels"][name]
            if question["type"] == "noul":
                p = confidence if right else 1 - confidence
                listings = [{"options": order, "answer": {"p_yes": p}, "code_mass": 0.8} for order in (["yes", "no"], ["no", "yes"])]
            else:
                keys = list(question["criteria"])
                best = max(keys, key=lambda k: label["probabilities"][k])
                target = best if right else next(k for k in keys if k != best)
                probs = {k: (confidence if k == target else (1 - confidence) / (len(keys) - 1)) for k in keys}
                listings = [{"options": keys[s:] + keys[:s], "answer": {"probabilities": probs}, "code_mass": 0.6} for s in range(len(keys))]
            questions[name] = {"type": question["type"], "listings": listings}
        out.append({"id": record["id"], "model": "m", "mode": "direct", "format": "chat", "prompt_version": "v0c-p3", "orders": "counterbalanced",
                    "shuffle_seed": 1, "questions": questions})  # fmt: skip
    return out


def test_the_paired_comparison_reads_every_group_and_bootstraps_over_origin_groups():
    records = _probe_records()
    good, bad = _predictions(records, right=True), _predictions(records, right=False)
    result = compare_probe(records, good, bad, n_boot=200)
    for reading in ("single_order", "order_averaged"):
        table = result["readings"][reading]["table"]
        overall = table["_all"]
        assert overall["questions"] == 6 and overall["clusters"] == 2  # 원본 레코드 셋, origin group 둘
        assert overall["accuracy"]["a"] == 1.0 and overall["accuracy"]["b"] == 0.0
        assert overall["accuracy"]["diff"] == overall["accuracy"]["low"] == overall["accuracy"]["high"] == 1.0 and overall["accuracy"]["excludes_zero"]
        assert overall["brier"]["diff"] < 0 and overall["code_mass"]["diff"] == 0.0
        assert set(table["domain"]) == {"spatial", "rules"} and table["domain"]["rules"]["questions"] == 4 and table["domain"]["rules"]["clusters"] == 1
        assert set(table["source_type"]) == {"choice", "boolean"} and table["language"]["en"]["questions"] == 4
        assert table["type"]["choice"]["chance_accuracy"] == pytest.approx(1 / 3)
    same = compare_probe(records, good, copy.deepcopy(good), n_boot=200)["readings"]["single_order"]["table"]["_all"]
    assert same["accuracy"]["diff"] == same["brier"]["diff"] == 0.0 and not same["accuracy"]["excludes_zero"]
    per_record = compare_probe(records, good, bad, n_boot=50, cluster="record")["readings"]["single_order"]["table"]["_all"]
    assert per_record["clusters"] == 3
    markdown = render_probe_comparison(result, label_a="v0", label_b="base", title="t")
    assert "| **all** | 6 / 2 |" in markdown and "domain = rules" in markdown and "+1.000 [1.000, 1.000] *" in markdown
    with pytest.raises(ValueError):
        compare_probe(records, good, bad, cluster="nope")


def test_the_compare_command_refuses_runs_scored_under_different_conditions(tmp_path):
    records = _probe_records()
    g1 = tmp_path / "G1.jsonl"
    g1.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records), encoding="utf-8")
    a, b = tmp_path / "a.predictions.jsonl", tmp_path / "b.predictions.jsonl"
    a.write_text("".join(json.dumps(p) + "\n" for p in _predictions(records, right=True)), encoding="utf-8")
    other = _predictions(records, right=False)
    for p in other:
        p["prompt_version"] = "v0c-p2"
    b.write_text("".join(json.dumps(p) + "\n" for p in other), encoding="utf-8")
    argv = ["compare", "--input", str(g1), "--a", str(a), "--b", str(b), "--labels", "v0", "base", "--out", str(tmp_path / "cmp.json"), "--n-boot", "50"]
    with pytest.raises(SystemExit, match="prompt_version"):
        script().main(argv)
    assert script().main([*argv, "--allow-mismatch"]) == 0
    payload = json.loads((tmp_path / "cmp.json").read_text(encoding="utf-8"))
    assert payload["config"]["mismatch"] == {"prompt_version": ["v0c-p3", "v0c-p2"]} and payload["comparison"]["records"] == 3
    assert "Scoring conditions differ" in (tmp_path / "cmp.md").read_text(encoding="utf-8")


# --------------------------------------------------------------------------
# 섞기용 학습 파일
# --------------------------------------------------------------------------


def _write(path: Path, records: list[dict]) -> Path:
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records), encoding="utf-8")
    return path


def test_the_mix_command_appends_the_probe_after_the_base_and_reports_the_example_share(tmp_path):
    train = judgment([boolean("q_b"), choice("q_c", [("c1", "o1", "o1"), ("c2", "none", None)])], [single("q_b", True), single("q_c", "c1")],
                     rid="rules-0009-0", split="train", origin="rules/x/0009", domain="rules")  # fmt: skip
    probe = _write(tmp_path / "G1-train-mix.jsonl", convert_records([train], splits=MIX_SPLITS)[0])
    sealed = _write(tmp_path / "G1.jsonl", convert_records([mixed_record()], splits=G1_SPLITS)[0])
    out = tmp_path / "mix.jsonl"
    argv = ["mix", "--base", str(REQUESTS), "--probe", str(probe), "--sealed", str(sealed), "--out", str(out),
            "--train-config", str(REPO / "configs" / "train" / "decisions-v0-lora-h100.yaml")]  # fmt: skip
    assert script().main(argv) == 0
    assert out.read_bytes() == REQUESTS.read_bytes() + probe.read_bytes()  # 학습 파일이 앞, 섞기용이 뒤 — 바이트 그대로
    manifest = json.loads((tmp_path / "mix.manifest.json").read_text(encoding="utf-8"))
    base = read_records(REQUESTS)
    assert manifest["out"]["sha256"] == file_sha256(out) and manifest["out"]["records"] == len(base) + 1
    assert manifest["base"]["sha256"] == file_sha256(REQUESTS) and manifest["probe"]["sha256"] == file_sha256(probe)
    examples = manifest["examples_per_epoch"]
    assert examples["probe"] == 2 and examples["probe_share"] == pytest.approx(2 / examples["total"])
    assert manifest["sealed_check"]["overlap"] == {"origin_groups": 0, "source_records": 0, "base_ids": 0}
    assert manifest["train_config"]["questions_per_request"] == 2 and manifest["train_config"]["dev_max_records"] == 300


def test_the_mix_command_refuses_sealed_splits_shared_lineage_and_clashing_ids(tmp_path):
    sealed_records = convert_records([mixed_record()], splits=G1_SPLITS)[0]
    sealed = _write(tmp_path / "G1.jsonl", sealed_records)
    out = tmp_path / "mix.jsonl"
    with pytest.raises(SystemExit):  # --sealed는 반드시 준다(리뷰 1 M4) — 빠뜨리면 argparse가 거절한다
        script().main(["mix", "--base", str(REQUESTS), "--probe", str(sealed), "--out", str(out)])
    with pytest.raises(SystemExit, match="only"):  # 봉인 분할의 레코드
        script().main(["mix", "--base", str(REQUESTS), "--probe", str(sealed), "--sealed", str(sealed), "--out", str(out)])
    same_group = judgment([boolean("q_b")], [single("q_b", True)], rid="spatial-0001-9", split="train", origin="spatial/zone-color/0001")
    probe = _write(tmp_path / "probe.jsonl", convert_records([same_group], splits=MIX_SPLITS)[0])
    with pytest.raises(SystemExit, match="shares origin groups"):
        script().main(["mix", "--base", str(REQUESTS), "--probe", str(probe), "--sealed", str(sealed), "--out", str(out)])
    clash = copy.deepcopy(read_records(probe))
    clash[0]["id"] = read_records(REQUESTS)[0]["id"]
    with pytest.raises(SystemExit, match="both files"):
        script().main(["mix", "--base", str(REQUESTS), "--probe", str(_write(tmp_path / "clash.jsonl", clash)), "--sealed", str(sealed), "--out", str(out)])
    assert not out.exists()


# --------------------------------------------------------------------------
# 실제 산출물 (있으면)
# --------------------------------------------------------------------------


@pytest.mark.skipif(not (PROBES / "G1.manifest.json").is_file(), reason="artifacts/simgen/probes/가 없다 (변환을 돌리지 않은 체크아웃)")
def test_the_shipped_probe_files_match_their_manifests_and_share_no_lineage():
    files = {}
    for name, splits in (("G1", G1_SPLITS), ("G1-train-mix", MIX_SPLITS)):
        manifest = json.loads((PROBES / f"{name}.manifest.json").read_text(encoding="utf-8"))
        path = PROBES / f"{name}.jsonl"
        assert manifest["file"]["sha256"] == file_sha256(path) and manifest["converter"]["git_dirty"] is False
        records = read_records(path)
        assert len(records) == manifest["file"]["records"] == manifest["counts"]["records"]
        assert {r["meta"]["split"] for r in records} <= set(splits) and all(validate_record(r) == [] for r in records)
        files[name] = records
    assert overlap(files["G1"], files["G1-train-mix"]) == {"origin_groups": [], "source_records": []}


# --------------------------------------------------------------------------
# G2 — 공개 세트에서 seed로 뽑은 독립 일반 시험 (통제자 결정 2026-10-12)
# --------------------------------------------------------------------------


def _public(name: str, n: int) -> list[dict]:
    """공개 세트 변환본(`public-sets-v0.1`)과 같은 꼴의 손 레코드 — boolq는 예/아니오 하나, multinli는 3지선다 하나."""
    out = []
    for i in range(n):
        if name == "boolq":
            state = {"passage": f"Passage {i} says the sky is blue.", "question": f"is the sky blue {i}"}
            question = {"id": "q_boolq", "type": "boolean", "instructions": "Given the passage, is the answer to the question yes?",
                        "criteria": [{"id": "true", "description": "yes"}, {"id": "false", "description": "no"}]}  # fmt: skip
            label = {"question_id": "q_boolq", "kind": "single", "answer": i % 2 == 0, "source": "boolq/validation", "label_confidence": "high"}
        else:
            state = {"premise": f"Premise {i}.", "hypothesis": f"Hypothesis {i}.", "genre": "fiction"}
            question = {"id": "q_nli", "type": "choice", "instructions": "What is the relation of the hypothesis to the premise?",
                        "criteria": [{"id": f"c_{c}", "description": c} for c in ("entailment", "neutral", "contradiction")]}  # fmt: skip
            label = {"question_id": "q_nli", "kind": "single", "answer": "c_neutral", "source": "multinli/dev_matched", "label_confidence": "high"}
        out.append({"schema_version": "judgment-v0", "origin_group": f"public/{name}/{i:05d}", "split": "dev",
                    "request": {"request_id": f"{name}-{i:05d}", "state": state, "questions": [question]}, "labels": [label],
                    "provenance": {"generator": "public-sets-v0.1", "domain": f"public:{name}", "public_source": name, "language": "en"}})  # fmt: skip
    return out


def test_the_g2_sample_is_seeded_keeps_file_order_and_maps_boolq_to_noul_and_multinli_to_three_options():
    from robo_jev.decisions.single_probe import sample_and_convert

    sources = {"boolq": _public("boolq", 40), "multinli": _public("multinli", 30)}
    converted, picked = sample_and_convert(sources, per_source=10, seed=0)
    again, _ = sample_and_convert(sources, per_source=10, seed=0)
    other, _ = sample_and_convert(sources, per_source=10, seed=1)
    assert converted == again and [r["id"] for r in converted] != [r["id"] for r in other]
    assert {name: spec["sampled"] for name, spec in picked.items()} == {"boolq": 10, "multinli": 10}
    for name in sources:
        ids = picked[name]["request_ids"]
        assert ids == sorted(ids) and len(set(ids)) == 10  # 원본 파일 순서
    types = {r["meta"]["domain"]: {(q["type"], len(q.get("criteria") or {})) for q in r["request"]["questions"].values()} for r in converted}
    assert types == {"public:boolq": {("noul", 0)}, "public:multinli": {("choice", 3)}}
    assert all(validate_record(r) == [] and r["meta"]["split"] == "dev" for r in converted)  # dev — mix 명령이 학습에 섞을 수 없다
    with pytest.raises(ValueError, match="뽑을 수 없다"):
        sample_and_convert(sources, per_source=31, seed=0)


def test_the_sample_command_records_the_seed_and_refuses_unexpected_question_types(tmp_path):
    paths = {name: _write(tmp_path / f"{name}.jsonl", _public(name, 12)) for name in ("boolq", "multinli")}
    argv = ["sample", "--source", f"boolq={paths['boolq']}", "--source", f"multinli={paths['multinli']}", "--per-source", "5", "--seed", "7",
            "--name", "G2", "--out-dir", str(tmp_path / "out")]  # fmt: skip
    assert script().main([*argv, "--expect-types", "boolq=noul", "multinli=choice:3"]) == 0
    manifest = json.loads((tmp_path / "out" / "G2.manifest.json").read_text(encoding="utf-8"))
    assert manifest["seed"] == 7 and manifest["role"] == "sealed" and manifest["file"]["records"] == 10
    assert manifest["file"]["sha256"] == file_sha256(tmp_path / "out" / "G2.jsonl")
    assert manifest["sources"]["boolq"]["sha256"] == file_sha256(paths["boolq"]) and manifest["sources"]["multinli"]["question_types"] == ["choice:3"]
    assert len(manifest["sources"]["boolq"]["request_ids"]) == 5 and manifest["counts"]["questions_by"]["type"] == {"choice": 5, "noul": 5}
    with pytest.raises(SystemExit, match="expected"):
        script().main([*argv, "--expect-types", "multinli=choice:4"])


def test_crossed_dimensions_split_the_comparison_by_two_groups_at_once():
    records = _probe_records()
    good, bad = _predictions(records, right=True), _predictions(records, right=False)
    table = compare_probe(records, good, bad, dimensions=("domain+source_type",), n_boot=50)["readings"]["single_order"]["table"]
    assert set(table["domain+source_type"]) == {"rules/boolean", "rules/choice", "spatial/boolean", "spatial/choice"}
    assert table["domain+source_type"]["rules/choice"]["questions"] == 2
    with pytest.raises(ValueError, match="모르는 묶음 차원"):
        compare_probe(records, good, bad, dimensions=("domain+nope",))


@pytest.mark.skipif(not (PROBES / "G2.manifest.json").is_file(), reason="artifacts/simgen/probes/G2가 없다 (뽑기를 돌리지 않은 체크아웃)")
def test_the_shipped_g2_matches_its_manifest_and_holds_five_hundred_of_each_public_set():
    manifest = json.loads((PROBES / "G2.manifest.json").read_text(encoding="utf-8"))
    assert manifest["file"]["sha256"] == file_sha256(PROBES / "G2.jsonl") and manifest["converter"]["git_dirty"] is False and manifest["seed"] == 0
    records = read_records(PROBES / "G2.jsonl")
    domains = {}
    for record in records:
        assert validate_record(record) == [] and record["meta"]["split"] == "dev"
        (question,) = record["request"]["questions"].values()
        domains.setdefault(record["meta"]["domain"], set()).add((question["type"], len(question.get("criteria") or {})))
    assert domains == {"public:boolq": {("noul", 0)}, "public:multinli": {("choice", 3)}}
    assert {name: spec["sampled"] for name, spec in manifest["sources"].items()} == {"boolq": 500, "multinli": 500}
