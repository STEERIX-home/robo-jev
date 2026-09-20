"""소형 scorer 기준군(Task 2c) — 예제 구성의 정보 경계·후보 순서, 모델 크기·모양, 작은 학습·표, 선택적 지표·ECE (CPU, D0 fixture)."""

import copy
import json

import pytest
import torch
from helpers import D0_MANIFEST, FIXTURES

from robo_jev.baselines.tiny_scorer import (
    Example, QuestionExample, TinyScorer, _mark_pattern_solvable, batch_loss, build_examples, build_model, collate, count_examples_by_kind, evaluate_checkpoint,
    evaluate_split, load_checkpoint, load_records, predict, train_tiny_scorer,
)
from robo_jev.contracts import QUESTION_SET_V0
from robo_jev.evaluate import calibration_error, selective_metrics
from robo_jev.model.serialize import full_tick_sections, serialize_request
from robo_jev.model.tokenizer import WhitespaceTokenizer

SMALL = {
    "seed": 3, "threads": 1,
    "data": {"manifests": [{"path": str(D0_MANIFEST), "domain": "robot"}], "train_splits": ["train"], "eval_splits": ["dev"], "robot_tick_stride": 10,
             "max_context_bytes": 384, "max_candidate_bytes": 48},
    "model": {"d_model": 32, "heads": 2, "ff": 64, "context_layers": 1, "candidate_layers": 1, "dropout": 0.0},
    "train": {"epochs": 1, "batch_states": 8, "lr": 0.002, "warmup_ratio": 0.1, "gradient_clip": 1.0, "max_wall_minutes": 5, "log_every": 1000},
    "eval": {"shuffle_seed": 1, "context_shuffle": True, "instruction_shuffle": True, "rule_judge": True, "ece_bins": 5,
             "pattern_solvable": {"accuracy": 0.85, "shuffle_gap": 0.05, "min_accuracy_for_gap": 0.7, "rule_judge_margin": 0.0}},
}


@pytest.fixture(scope="module")
def d0():
    singles = [json.loads(line) for line in (FIXTURES / "d0.jsonl").read_text(encoding="utf-8").splitlines()]
    streams = [json.loads(line) for line in (FIXTURES / "d0_streams.jsonl").read_text(encoding="utf-8").splitlines()]
    return singles, streams


def test_examples_follow_the_serializers_candidate_order_and_stay_inside_the_input_boundary(d0):
    """후보 id 순서는 직렬화의 `candidate_mapping`과 같고(규칙 기준군 열과 같은 후보 위의 확률), 문맥에는 라벨·근거·참값이 없다."""
    singles, streams = d0
    tokenizer = WhitespaceTokenizer()
    single = next(record for record in singles if any(q["type"] == "choice" for q in record["request"]["questions"]))
    example = build_examples([single], domain="non_robot")[0]
    layout = serialize_request(single, tokenizer)
    assert example.candidate_mapping == layout["candidate_mapping"] and example.kind == "single"
    assert example.context.startswith("[state]\n") and "true_state" not in example.context and "rule_trace" not in example.context
    for question in example.questions:
        assert len(question.candidate_texts) == len(question.candidate_ids)
        assert all(text.startswith(f"{cid}:") for cid, text in zip(question.candidate_ids, question.candidate_texts))
    stream = copy.deepcopy(streams[0])
    stream["ticks"] = stream["ticks"][:5]
    examples = build_examples([stream], domain="robot", max_context=2048)
    layout = serialize_request(stream, tokenizer, layout="stream_l1a")
    assert [example.tick for example in examples] == [0, 1, 2, 3, 4]
    for example, entry in zip(examples, layout["ticks"]):
        assert example.candidate_mapping == entry["candidate_mapping"]
        assert set(example.question_types) == set(entry["posed"])
        assert "labels" not in example.context and "model_output" not in example.context and "adopted" not in example.context
        assert example.context.startswith("instruction v1 ")
        assert "goal " in example.context and "robot " in example.context
    # 문맥 상한은 byte 단위이며 앞을 남긴다 (목표·로봇 줄이 물체보다 앞).
    short = build_examples([stream], domain="robot", max_context=200)[0]
    assert len(short.context.encode("utf-8")) <= 200 and short.context.startswith("instruction v1 ")
    # stride는 학습용 표본 솎기다.
    assert [example.tick for example in build_examples([stream], domain="robot", robot_tick_stride=2)] == [0, 2, 4]
    # 한 틱의 전체 렌더링: 첫 틱처럼 모든 물체의 소개·동적 줄, 목표 텍스트.
    sections = full_tick_sections(stream["ticks"][3])
    assert {"t", "goal", "objects", "robot"} <= set(sections)
    assert sections["objects"].count("\n") >= 2 * len(stream["ticks"][3]["request"]["state"]["objects"])


def test_the_scorer_has_about_a_million_parameters_and_scores_every_candidate_of_every_question(d0):
    singles, streams = d0
    default = TinyScorer()
    assert 700_000 <= default.parameter_count() <= 1_300_000
    model = build_model(SMALL)
    stream = copy.deepcopy(streams[0])
    stream["ticks"] = stream["ticks"][:2]
    examples = build_examples([stream], domain="robot", max_context=384, max_candidate=48) + build_examples(singles[:2], domain="non_robot", max_context=384, max_candidate=48)
    batch = collate(examples, max_context=384, max_candidate=48)
    outputs = model(batch)
    assert len(outputs) == len(examples)
    for example, logits in zip(examples, outputs):
        for question in example.questions:
            if question.candidate_ids:
                assert tuple(logits[question.question_id].shape) == (len(question.candidate_ids),)
    loss, labels = batch_loss(model, examples)
    assert loss is not None and torch.isfinite(loss) and labels > 0
    loss.backward()
    assert all(p.grad is not None for name, p in model.named_parameters() if "pos_context" not in name)
    predictions = predict(model, examples, batch_states=2)
    assert len(predictions) == len(examples)
    assert all(abs(float(p["probabilities"][qid].sum()) - 1.0) < 1e-5 for p in predictions for qid in p["probabilities"])
    assert predictions[0]["candidates"] == examples[0].candidate_mapping
    # 같은 seed·같은 입력이면 같은 점수 (결정성).
    torch.manual_seed(1)
    a = build_model(SMALL)
    torch.manual_seed(1)
    b = build_model(SMALL)
    a.eval(), b.eval()
    with torch.no_grad():
        assert torch.equal(a(batch)[0]["q_main"], b(batch)[0]["q_main"])


def test_train_tiny_scorer_returns_the_table_with_the_standard_columns(tmp_path):
    """분할·분야별 (소형 scorer, 문맥 섞기 대조군, 치환 답 변경률, 규칙 기준군, ECE, 선택적 지표, 패턴 표지) 표 — 인수 기준은 표가 있다는 것."""
    report = train_tiny_scorer(SMALL, checkpoint=tmp_path / "scorer.pt")
    assert report["parameters"] == build_model(SMALL).parameter_count() and (tmp_path / "scorer.pt").is_file()
    assert report["training"]["steps"] > 0 and report["training"]["epochs"][0]["loss"] is not None
    assert set(report["tables"]) == {"robot/dev", "robot_contrast/dev"}
    stream_table = report["tables"]["robot/dev"]
    assert stream_table["n_records"] == 1 and stream_table["n_states"] == 100 and stream_table["kinds"] == {"stream": 100}
    for column in ("model", "permuted", "answer_change", "context_shuffle", "context_shuffle_ece", "instruction_shuffle", "rule_judge", "ece", "selective", "rule_judge_selective", "pattern_solvable", "eval_robot_tick_stride"):
        assert column in stream_table, column
    assert stream_table["context_shuffle_kind"] == "state" and stream_table["instruction_shuffle_kind"] == "instruction"  # 상태 섞기가 표준 열, 지시 섞기는 둘째 열 (D1 리뷰 1 I1)
    assert stream_table["instruction_shuffle"]["q_main"]["n"] == stream_table["context_shuffle"]["q_main"]["n"] == stream_table["model"]["q_main"]["n"]
    # 채점 가능한 라벨이 있는 질문마다 행이 있다 (D0 dev 스트림의 q_retry는 전부 마스크라 행이 없다 — aggregate의 규칙).
    assert {"q_main", "q_done", "q_gripper", "q_path", "q_speed", "q_force", "q_stop"} <= set(stream_table["model"]) and "_all" in stream_table["model"]
    assert set(stream_table["model"]) - {"_all"} <= set(QUESTION_SET_V0)
    assert set(stream_table["rule_judge"]) >= {"q_main", "q_done"}
    for key, row in stream_table["model"].items():
        assert row["n"] > 0 and row["nll"] >= 0
    assert 0 <= stream_table["ece"]["ece"] <= 1 and stream_table["ece"]["n"] > 0
    assert stream_table["answer_change"]["compared"] > 0
    selective = stream_table["selective"]
    assert selective["n"] == 100 and abs(selective["coverage"] + selective["abstention"] - 1.0) < 1e-9
    marks = stream_table["pattern_solvable"]
    assert set(marks) == set(stream_table["model"])
    assert all(set(mark) == {"accuracy", "shuffle_accuracy", "instruction_shuffle_accuracy", "rule_judge_accuracy", "pattern_solvable", "reasons"} for mark in marks.values())
    assert marks["q_main"]["rule_judge_accuracy"] == stream_table["rule_judge"]["q_main"]["accuracy"] and marks["q_main"]["instruction_shuffle_accuracy"] == stream_table["instruction_shuffle"]["q_main"]["accuracy"]
    assert all(reason in ("scorer_high", "shuffle_high", "state_shuffle_irrelevant", "scorer_beats_rule_judge") for mark in marks.values() for reason in mark["reasons"])
    single_table = report["tables"]["robot_contrast/dev"]
    assert single_table["kinds"] == {"single": 8} and single_table["context_shuffle_kind"] == "state" and "rule_judge" not in single_table
    assert "instruction_shuffle" not in single_table and "eval_robot_tick_stride" not in single_table  # 단일 레코드만의 표에는 뜻이 없다 (리뷰 1 M8)
    assert {"choice", "boolean", "ordinal"} & set(single_table["model"])
    assert report["sources"][0]["train_examples"] == 32 + 2 * 10  # 단일 32 + 스트림 2편 × (100틱 / stride 10)
    assert report["sources"][0]["train_examples_by_kind"] == {"single": 32, "stream": 20} == report["train_examples_by_kind"]


def test_evaluate_checkpoint_reproduces_the_trained_tables_without_training_and_refuses_another_model_shape(tmp_path):
    """D1 리뷰 1 I1: 대조군 열을 바꾼 재평가는 저장 모델에서 학습 없이 — 같은 모델이면 같은 표, `training`은 학습 보고서에서 옮겨 적고, 모델 모양이
    다른 설정은 거절한다."""
    trained = train_tiny_scorer(SMALL, checkpoint=tmp_path / "scorer.pt")
    model, saved = load_checkpoint(tmp_path / "scorer.pt", config=SMALL)
    assert saved["model"] == SMALL["model"] and model.parameter_count() == trained["parameters"]
    again = evaluate_checkpoint(SMALL, tmp_path / "scorer.pt", training_report=trained)
    assert again["evaluation"]["eval_only"] is True and again["evaluation"]["checkpoint"] == str(tmp_path / "scorer.pt")
    assert again["training"] == trained["training"] and again["parameters"] == trained["parameters"]
    assert again["sources"] == trained["sources"] and again["train_examples_by_kind"] == trained["train_examples_by_kind"]
    assert set(again["tables"]) == set(trained["tables"])
    for name, table in trained["tables"].items():
        for column in ("model", "context_shuffle", "instruction_shuffle", "rule_judge", "pattern_solvable"):
            if column in table:
                assert again["tables"][name][column] == table[column], (name, column)
    assert evaluate_checkpoint(SMALL, tmp_path / "scorer.pt")["training"] is None
    with pytest.raises(ValueError, match="model"):
        load_checkpoint(tmp_path / "scorer.pt", config={**SMALL, "model": {**SMALL["model"], "d_model": 64}})
    with pytest.raises(ValueError, match="max_context_bytes"):
        load_checkpoint(tmp_path / "scorer.pt", config={**SMALL, "data": {**SMALL["data"], "max_context_bytes": 512}})


def test_pattern_marks_name_what_they_measure(d0):
    """`state_shuffle_irrelevant`는 상태 섞기 열에만, `scorer_beats_rule_judge`는 규칙 열이 있을 때만; 지시 섞기 값은 적기만 하고 표지를 정하지 않는다."""
    rule = {"accuracy": 0.85, "shuffle_gap": 0.05, "min_accuracy_for_gap": 0.7, "rule_judge_margin": 0.0}
    model = {"q_main": {"accuracy": 0.79}, "q_done": {"accuracy": 0.99}, "q_low": {"accuracy": 0.40}}
    state = {"q_main": {"accuracy": 0.60}, "q_done": {"accuracy": 0.98}, "q_low": {"accuracy": 0.39}}
    instruction = {"q_main": {"accuracy": 0.79}, "q_done": {"accuracy": 0.99}, "q_low": {"accuracy": 0.40}}
    judge = {"q_main": {"accuracy": 0.74}, "q_done": {"accuracy": 1.00}, "q_low": {"accuracy": 0.73}}
    marks = _mark_pattern_solvable(model, state, rule, instruction_table=instruction, rule_table=judge)
    assert marks["q_main"]["reasons"] == ["scorer_beats_rule_judge"] and marks["q_main"]["instruction_shuffle_accuracy"] == 0.79 and marks["q_main"]["rule_judge_accuracy"] == 0.74
    assert marks["q_done"]["reasons"] == ["scorer_high", "shuffle_high", "state_shuffle_irrelevant"]  # 규칙 1.00 > scorer 0.99 → 규칙 이유 없음
    assert marks["q_low"] == {"accuracy": 0.40, "shuffle_accuracy": 0.39, "instruction_shuffle_accuracy": 0.40, "rule_judge_accuracy": 0.73, "pattern_solvable": False, "reasons": []}
    without_rule = _mark_pattern_solvable(model, state, rule, instruction_table=instruction)
    assert without_rule["q_main"]["reasons"] == [] and without_rule["q_main"]["rule_judge_accuracy"] is None
    assert "context_irrelevant" not in json.dumps(marks)
    singles, streams = d0
    stream = copy.deepcopy(streams[0])
    stream["ticks"] = stream["ticks"][:7]
    assert count_examples_by_kind([stream] + singles[:3], robot_tick_stride=3) == {"stream": 3, "single": 3}
    assert count_examples_by_kind([stream], robot_tick_stride=1) == {"stream": 7}


def test_selective_metrics_read_gates_as_abstention_and_forbidden_targets_as_unsafe(d0):
    _, streams = d0
    record = copy.deepcopy(streams[0])
    record["ticks"] = record["ticks"][:3]
    ticks = record["ticks"]
    examples = build_examples([record], domain="robot")
    ids = examples[0].candidate_mapping["q_main"]
    entries = {e["id"]: e for e in ticks[0]["request"]["candidates"]["q_main"]}
    keys = {cid: str(entries[cid]["key"]) for cid in ids}
    gate = next(cid for cid in ids if keys[cid] in ("observe", "hold", "replan"))
    label = next(l for l in ticks[0]["labels"] if l["question_id"] == "q_main")
    allowed = label["candidate_ids"][0]
    forbidden = set(ticks[0]["request"]["state"]["goal"]["forbidden_contact"])
    bad = next(cid for cid in ids if keys[cid].split(":")[0] in ("grasp", "push") and keys[cid].split(":")[1] in forbidden)

    def prediction(choice: str, tick: int, stop_true: float = 1.0) -> dict:
        probs = torch.full((len(ids),), 0.01)
        probs[ids.index(choice)] = 1.0
        return {"record_id": record["episode_id"], "tick": tick, "kind": "stream", "split": "dev",
                "probabilities": {"q_main": probs / probs.sum(), "q_stop": torch.tensor([stop_true, 1.0 - stop_true])},
                "candidates": {"q_main": ids, "q_stop": ["true", "false"]}, "labels": ticks[tick]["labels"], "question_types": examples[0].question_types}

    result = selective_metrics([prediction(gate, 0), prediction(allowed, 1), prediction(bad, 2)], [record])
    assert result["n"] == 3 and result["abstention"] == pytest.approx(1 / 3) and result["coverage"] == pytest.approx(2 / 3)
    assert result["abstention_by_gate"] == {keys[gate]: 1}
    assert result["selective_accuracy"] == pytest.approx(0.5) and result["unsafe_action_rate"] == pytest.approx(0.5) and result["forbidden_target"] == 1
    assert result["wrong_target_rate"] == pytest.approx(0.5)
    # 정답이 정지인데 q_stop을 낮게 답하면 unsafe.
    stopped = copy.deepcopy(record)
    for l in stopped["ticks"][1]["labels"]:
        if l["question_id"] == "q_stop":
            l["answer"] = True
    result = selective_metrics([prediction(allowed, 1, stop_true=0.2)], [stopped])
    assert result["stop_ignored"] == 1 and result["unsafe_action_rate"] == 1.0
    assert selective_metrics([{**prediction(allowed, 1), "kind": "single"}], [record])["n"] == 0


def test_calibration_error_is_zero_for_confident_correct_answers_and_grows_with_overconfidence():
    labels = [{"question_id": "q", "kind": "single", "answer": "a"}]
    sure = {"record_id": "r", "tick": None, "kind": "single", "split": "dev", "probabilities": {"q": torch.tensor([0.999, 0.001])}, "candidates": {"q": ["a", "b"]}, "labels": labels, "question_types": {"q": "choice"}}
    wrong = {**sure, "probabilities": {"q": torch.tensor([0.001, 0.999])}}
    perfect = calibration_error([sure] * 4, bins=10)
    assert perfect["ece"] == pytest.approx(0.001, abs=1e-3) and perfect["n"] == 4 and len(perfect["bins"]) == 10
    over = calibration_error([sure, wrong, wrong, wrong], bins=10)
    assert over["ece"] == pytest.approx(0.75, abs=0.01)
    event = {**sure, "candidates": {"q": ["true", "false"]}, "labels": [{"question_id": "q", "kind": "event", "successes": 1, "failures": 1}]}
    assert calibration_error([event])["n"] == 0  # 확률 질문(event)은 채점하지 않는다
