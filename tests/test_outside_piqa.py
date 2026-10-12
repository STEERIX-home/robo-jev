"""T8a — PIQA validation → decision requests (Task T8 §1-1)."""

import json

import pytest

from robo_jev.decisions.prompt import check_displayed_lines, record_prompts
from robo_jev.decisions.records import validate_record
from robo_jev.model.tokenizer import available_tokenizer, load_tokenizer
from robo_jev.outside.piqa import INSTRUCTIONS, LICENSE, QUESTION, convert, piqa_record, read_source

ROWS = [
    {"goal": "How do I  ready a guinea pig cage?", "sol1": "Use paper bedding.", "sol2": "Use\njeans bedding."},
    {"goal": "To cool soup quickly,", "sol1": "put the pot in the oven.", "sol2": "set the pot in a sink of cold water."},
]


def test_record_shape_and_label():
    record = piqa_record(0, ROWS[0], 0)
    assert validate_record(record) == []
    assert record["id"] == "piqa-val-00000"
    assert record["request"]["state"] == "Goal: How do I ready a guinea pig cage?"  # whitespace collapsed, nothing added
    question = record["request"]["questions"][QUESTION]
    assert question == {"type": "choice", "instructions": INSTRUCTIONS, "criteria": {"sol1": "Use paper bedding.", "sol2": "Use jeans bedding."}}
    assert record["labels"][QUESTION] == {"type": "choice", "kind": "probe", "probabilities": {"sol1": 1.0, "sol2": 0.0}}
    second = piqa_record(1, ROWS[1], 1)
    assert second["labels"][QUESTION]["probabilities"] == {"sol1": 0.0, "sol2": 1.0}
    assert second["meta"]["domain"] == "public:piqa" and second["meta"]["set"] == "T8a" and second["meta"]["origin_group"] == "public/piqa/00001"


def test_model_input_has_no_label_hint():
    record = piqa_record(1, ROWS[1], 1)
    text = json.dumps(record["request"])
    for word in ("correct", "label", "answer", "robot"):
        assert word not in text.lower()


def test_read_source_and_convert(tmp_path):
    rows = tmp_path / "valid.jsonl"
    rows.write_text("".join(json.dumps(r) + "\n" for r in ROWS), encoding="utf-8")
    labels = tmp_path / "valid-labels.lst"
    labels.write_text("0\n1\n", encoding="utf-8")
    records, summary = convert(read_source(rows, labels))
    assert [r["id"] for r in records] == ["piqa-val-00000", "piqa-val-00001"]
    assert summary["label_sol1"] == 1 and summary["label_sol2"] == 1
    labels.write_text("0\n", encoding="utf-8")
    with pytest.raises(ValueError):
        read_source(rows, labels)
    labels.write_text("0\n2\n", encoding="utf-8")
    with pytest.raises(ValueError):
        read_source(rows, labels)


def test_empty_solution_is_refused():
    with pytest.raises(ValueError):
        piqa_record(0, {"goal": "x", "sol1": " ", "sol2": "y"}, 0)


def test_license_is_recorded():
    assert "unknown" in LICENSE["dataset_card"].lower()
    assert "AFL-3.0" in LICENSE["homepage"]


def test_prompts_build_with_the_diagnosis_prompt():
    tokenizer = load_tokenizer(available_tokenizer()[0])
    record = piqa_record(0, ROWS[0], 0)
    prompts = record_prompts(record, tokenizer, seed=1, fmt="chat", mode="direct", orders="counterbalanced")
    assert len(prompts) == 2  # two balanced listings of a two-option choice
    assert {tuple(p["options"]) for p in prompts} == {("sol1", "sol2"), ("sol2", "sol1")}
    for prompt in prompts:
        assert check_displayed_lines(prompt) == []
        assert "Goal: How do I ready a guinea pig cage?" in prompt["text"]
