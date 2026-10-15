"""The many-option opt-in (Task v0.7a) — `robo_jev.decisions.records.option_range_for` and its readers.

The choice range stays 2–8 for every record and every request (the serving path included); only a record whose ``meta.option_range`` is
exactly :data:`OPTION_RANGE_MARK` may carry 2–26 options; any other value under that key is a problem, never a silent default. The trainer
reads the opt-in from the data (its loader validates every record), and the prompt, the target and the loss handle 26 codes (A–Z, each a
single token at the answer boundary).
"""

from __future__ import annotations

import copy
import functools
import json
import math
import string

import pytest
import torch

from robo_jev.decisions import examples as E
from robo_jev.decisions import prompt as P
from robo_jev.decisions.records import OPTION_RANGE, OPTION_RANGE_MARK, WIDE_OPTION_RANGE, option_range_for, validate_record
from robo_jev.model.tokenizer import available_tokenizer, load_tokenizer
from robo_jev.model.zero_shot import verify_codes


@functools.lru_cache(maxsize=1)
def tokenizer():
    return load_tokenizer(available_tokenizer()[0])


def _record(n: int, *, mark: object = None, rid: str = "r1") -> dict:
    keys = [f"k{i}" for i in range(n)]
    record = {"schema": "simgen-request/0", "id": rid, "meta": {"set": "test"},
              "request": {"state": "A robot is about to move a cup.", "questions": {
                  "pick": {"type": "choice", "instructions": "Which option?", "criteria": {k: f"option number {i}" for i, k in enumerate(keys)}},
                  "spill": {"type": "noul", "instructions": "Will it spill?"}}},
              "labels": {"pick": {"type": "choice", "probabilities": {k: (1.0 if i == 0 else 0.0) for i, k in enumerate(keys)}},
                         "spill": {"type": "noul", "p_yes": 0.3}}}
    if mark is not None:
        record["meta"]["option_range"] = mark
    return record


def test_the_default_range_is_unchanged_and_the_wide_range_is_the_26_codes():
    assert OPTION_RANGE == (2, 8) and WIDE_OPTION_RANGE == (2, len(string.ascii_uppercase)) == (2, 26)
    assert OPTION_RANGE_MARK == {"version": "options-26-1", "max": 26}


@pytest.mark.parametrize("n", [2, 5, 8])
def test_up_to_eight_options_pass_with_or_without_the_mark(n):
    assert validate_record(_record(n)) == []
    assert validate_record(_record(n, mark=dict(OPTION_RANGE_MARK))) == []


@pytest.mark.parametrize("n", [9, 17, 26])
def test_more_than_eight_options_fail_by_default_and_pass_only_with_the_mark(n):
    assert validate_record(_record(n)) == [f"question 'pick': choice needs 2..8 options (has {n})"]
    assert validate_record(_record(n, mark=dict(OPTION_RANGE_MARK))) == []


def test_the_mark_never_lifts_past_26_or_below_2():
    assert validate_record(_record(27, mark=dict(OPTION_RANGE_MARK))) == ["question 'pick': choice needs 2..26 options (has 27)"]
    one = _record(2, mark=dict(OPTION_RANGE_MARK))
    one["request"]["questions"]["pick"]["criteria"] = {"k0": "only"}
    one["labels"]["pick"]["probabilities"] = {"k0": 1.0}
    assert validate_record(one) == ["question 'pick': choice needs 2..26 options (has 1)"]


@pytest.mark.parametrize("mark", [{"version": "options-26-1", "max": 30}, {"version": "options-26-2", "max": 26}, {"max": 26},
                                  {"version": "options-26-1", "max": 26, "extra": 1}, True, 26, "options-26-1", []])
def test_any_other_value_under_the_key_is_a_problem_and_keeps_the_default(mark):
    problems = validate_record(_record(12, mark=mark))
    assert any("meta.option_range" in p for p in problems) and "question 'pick': choice needs 2..8 options (has 12)" in problems
    assert option_range_for(_record(3, mark=mark))[0] == OPTION_RANGE and option_range_for(_record(3, mark=mark))[1]


def test_a_request_without_meta_the_serving_path_always_gets_two_to_eight():
    """robo_jev.infer validates a bare request ({schema, id, request}, no labels, no meta) — the opt-in cannot reach it."""
    record = _record(12, mark=dict(OPTION_RANGE_MARK))
    bare = {"schema": "simgen-request/0", "id": "request", "request": record["request"]}
    assert validate_record(bare, require_labels=False) == ["question 'pick': choice needs 2..8 options (has 12)"]
    assert option_range_for(bare) == (OPTION_RANGE, [])


def test_the_trainers_loader_reads_the_opt_in_from_the_data(tmp_path):
    from robo_jev.decisions.train import load_training_records

    path = tmp_path / "train.jsonl"
    rows = [_record(26, mark=dict(OPTION_RANGE_MARK), rid="wide"), _record(9, rid="unmarked"), _record(4, rid="plain")]
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    records, validation = load_training_records(path)
    assert [r["id"] for r in records] == ["wide", "plain"] and validation["invalid"] == 1 and "unmarked" in validation["problems"]


def test_the_26_codes_are_single_tokens_at_the_answer_boundary():
    codes, ids = verify_codes(tokenizer(), 26)
    assert codes == list(string.ascii_uppercase) and len(set(ids)) == 26
    with pytest.raises(ValueError):
        verify_codes(tokenizer(), 27)


def test_a_26_option_prompt_shows_every_code_and_its_target_gives_distractors_zero():
    record = _record(26, mark=dict(OPTION_RANGE_MARK))
    (example,) = E.build_examples(record, ["pick"], tokenizer(), order_seed=E.derive_seed(0, 0, "r1", "order"))
    assert example["codes"] == list(string.ascii_uppercase) and len(example["code_ids"]) == 26
    assert P.check_displayed_lines(example) == []
    assert sorted(example["options"]) == sorted(record["request"]["questions"]["pick"]["criteria"])
    assert example["target"][example["options"].index("k0")] == 1.0 and sum(example["target"]) == 1.0
    positions = {E.build_examples(record, ["pick"], tokenizer(), order_seed=s)[0]["options"].index("k0") for s in range(400)}
    assert len(positions) >= 24, "the trainer's order augmentation spreads the correct option over the letters"
    dev = E.dev_examples([record], tokenizer())
    assert sum(1 for d in dev if d["question"] == "pick") == 26, "counterbalanced dev listings: each option at each position once"


def test_the_trainers_loss_reads_26_code_rows():
    pytest.importorskip("transformers")
    from robo_jev.decisions import train as T
    from robo_jev.model.backbone_qwen import QwenBackbone, torch_reference_kernels

    record = _record(26, mark=dict(OPTION_RANGE_MARK))
    short = _record(3, rid="r2")
    examples = E.build_examples(record, ["pick"], tokenizer(), order_seed=7) + E.build_examples(short, ["pick"], tokenizer(), order_seed=7)
    with torch_reference_kernels():
        backbone = QwenBackbone.tiny(seed=4, vocab_size=tokenizer().get_vocab_size())
        out = T.code_kl(backbone, examples, code_mass=True)
    assert [len(p) for p in out["probs"]] == [26, 3]
    assert all(math.isfinite(float(v)) for v in out["kl"]) and all(abs(sum(p) - 1.0) < 1e-5 for p in out["probs"])
    assert all(0.0 < m <= 1.0 for m in out["code_mass"])
