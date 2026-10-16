"""The v0.8 many-option format (Task v0.8a) — the code pool, the 255-option opt-in and assigned codes, through records, prompt, target and loss.

* The pool (:data:`robo_jev.decisions.codes.CODE_POOL`) is the Hub runner's ``CODES[:255]``: A–Z, then the first 229 two-letter codes; every
  one is a single token at the answer boundary with the pinned tokenizer.
* ``meta.option_range`` = ``{"version": "options-255-1", "max": 255}`` allows 2–255 options; the default stays 2–8 and the 26-option mark
  keeps 2–26.
* ``meta.codes[question]`` shows a choice question's options with given pool codes in display order; the readout reads those codes' rows.
* A record without these metas gives byte-identical prompts and targets to the base (``88653e1``) — the fixture digest below was computed
  with the base code (``artifacts/scratch/v08a/fixture_digest.py`` on ``git archive 88653e1 src``) and equals the new code's.
"""

from __future__ import annotations

import copy
import functools
import hashlib
import json
import math
import string

import pytest
import torch
from helpers import FIXTURES

from robo_jev.decisions import codes as C
from robo_jev.decisions import examples as E
from robo_jev.decisions import prompt as P
from robo_jev.decisions.records import (
    OPTION_RANGE,
    OPTION_RANGE_MARK,
    OPTION_RANGE_MARK_255,
    POOL_OPTION_RANGE,
    WIDE_OPTION_RANGE,
    option_range_for,
    read_records,
    validate_record,
)
from robo_jev.model.tokenizer import available_tokenizer, load_tokenizer
from robo_jev.model.zero_shot import verify_codes

REQUESTS = FIXTURES / "decisions" / "requests.jsonl"
#: sha256 over every fixture prompt (plain/chat × direct/reasoning × seeded/counterbalanced × p2/p3) and every training / dev example of the
#: fixture requests, computed with the base code 88653e1 (1,200 prompts and examples) — the unchanged path is byte-identical.
BASE_FIXTURE_DIGEST = "be043c60feddca07598e709d52ba9dfb7193208a832deaa4ba5e0a9fcf9d46b8"
FIELDS = ("question", "type", "listing", "n_listings", "options", "option_lines", "values", "codes", "code_ids", "tokens", "prefix_tokens", "text",
          "prefix_chars", "format", "mode", "version")


@functools.lru_cache(maxsize=1)
def tokenizer():
    return load_tokenizer(available_tokenizer()[0])


def _runner():
    from robo_jev.hf import robo_jev_decide

    return robo_jev_decide


def _record(n: int, *, mark: object = None, codes: object = None, rid: str = "r1", correct: int = 0) -> dict:
    keys = [f"k{i:03d}" for i in range(n)]
    record = {"schema": "simgen-request/0", "id": rid, "meta": {"set": "test"},
              "request": {"state": "A robot is about to move a cup.", "questions": {
                  "pick": {"type": "choice", "instructions": "Which option?", "criteria": {k: f"option number {i}" for i, k in enumerate(keys)}},
                  "spill": {"type": "noul", "instructions": "Will it spill?"}}},
              "labels": {"pick": {"type": "choice", "probabilities": {k: (1.0 if i == correct else 0.0) for i, k in enumerate(keys)}},
                         "spill": {"type": "noul", "p_yes": 0.3}}}
    if mark is not None:
        record["meta"]["option_range"] = mark
    if codes is not None:
        record["meta"]["codes"] = codes
    return record


# -- the pool --------------------------------------------------------------------------------------------------------------------------


def test_the_pool_is_the_runners_first_255_codes_letters_first():
    runner = _runner()
    assert C.CODE_POOL == tuple(runner.CODES[:255]) and C.POOL_SIZE == 255 == len(set(C.CODE_POOL))
    assert C.LETTERS == tuple(string.ascii_uppercase) == C.CODE_POOL[:26]
    assert all(len(c) == 2 for c in C.CODE_POOL[26:]) and C.CODE_POOL[26] == "AA" and C.CODE_POOL[-1] == "MO"
    assert not set(C.CODE_POOL) & set(runner.CODE_STOPLIST)
    assert [C.pool_index(c) for c in C.CODE_POOL] == list(range(255))
    assert C.sequential_codes(26) == list(string.ascii_uppercase) and C.sequential_codes(255) == list(C.CODE_POOL)
    with pytest.raises(ValueError):
        C.sequential_codes(256)


def test_every_pool_code_is_one_token_at_the_answer_boundary_with_the_pinned_tokenizer():
    ids = C.verify_pool_codes(tokenizer(), C.CODE_POOL)
    assert len(ids) == 255 == len(set(ids))
    letters, letter_ids = verify_codes(tokenizer(), 26)
    assert ids[:26] == letter_ids and letters == list(C.LETTERS)
    assert ids == _runner()._verify_codes(tokenizer(), 255)[1], "the runner reads the same rows"
    for bad in (["AC"], ["IT"], ["a"], ["ZZZ"]):
        with pytest.raises(ValueError):
            C.verify_pool_codes(tokenizer(), bad)


def test_assigned_codes_are_checked_for_length_pool_and_repeats():
    assert C.codes_problems(["A", "GX", "MO"], 3) == []
    assert C.codes_problems(["A", "GX"], 3) == ["2 assigned codes for 3 options"]
    assert C.codes_problems(["A", "A", "B"], 3) == ["assigned codes repeat"]
    assert C.codes_problems(["A", "AC", "B"], 3) == ["codes outside the pool: ['AC']"]
    assert C.codes_problems("ABC", 3) and C.codes_problems(None, 3)


# -- the 255-option opt-in -------------------------------------------------------------------------------------------------------------


def test_the_255_mark_allows_two_to_255_and_the_other_ranges_are_unchanged():
    assert OPTION_RANGE == (2, 8) and WIDE_OPTION_RANGE == (2, 26) and POOL_OPTION_RANGE == (2, 255)
    assert OPTION_RANGE_MARK == {"version": "options-26-1", "max": 26} and OPTION_RANGE_MARK_255 == {"version": "options-255-1", "max": 255}
    for n in (2, 8, 9, 26, 27, 60, 61, 120, 255):
        assert validate_record(_record(n, mark=dict(OPTION_RANGE_MARK_255))) == [], n
        assert option_range_for(_record(n, mark=dict(OPTION_RANGE_MARK_255))) == (POOL_OPTION_RANGE, [])
    assert validate_record(_record(256, mark=dict(OPTION_RANGE_MARK_255))) == ["question 'pick': choice needs 2..255 options (has 256)"]


@pytest.mark.parametrize("n", [9, 27, 255])
def test_without_a_mark_the_default_stays_two_to_eight_and_the_26_mark_stays_26(n):
    assert validate_record(_record(n)) == [f"question 'pick': choice needs 2..8 options (has {n})"]
    expected = [] if n <= 26 else [f"question 'pick': choice needs 2..26 options (has {n})"]
    assert validate_record(_record(n, mark=dict(OPTION_RANGE_MARK))) == expected


@pytest.mark.parametrize("mark", [{"version": "options-255-1", "max": 256}, {"version": "options-255-2", "max": 255}, {"max": 255},
                                  {"version": "options-255-1", "max": 255, "extra": 1}, 255, "options-255-1"])
def test_any_other_255_like_value_is_a_problem_and_keeps_the_default(mark):
    problems = validate_record(_record(30, mark=mark))
    assert any("meta.option_range" in p for p in problems) and "question 'pick': choice needs 2..8 options (has 30)" in problems
    assert option_range_for(_record(3, mark=mark))[0] == OPTION_RANGE


def test_a_bare_request_still_gets_two_to_eight():
    bare = {"schema": "simgen-request/0", "id": "request", "request": _record(30, mark=dict(OPTION_RANGE_MARK_255))["request"]}
    assert validate_record(bare, require_labels=False) == ["question 'pick': choice needs 2..8 options (has 30)"]


def test_the_trainers_loader_keeps_255_marked_records(tmp_path):
    from robo_jev.decisions.train import load_training_records

    path = tmp_path / "train.jsonl"
    rows = [_record(255, mark=dict(OPTION_RANGE_MARK_255), rid="wide"), _record(30, mark=dict(OPTION_RANGE_MARK), rid="over26"), _record(4, rid="plain")]
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    records, validation = load_training_records(path)
    assert [r["id"] for r in records] == ["wide", "plain"] and validation["invalid"] == 1 and "over26" in validation["problems"]


# -- meta.codes ------------------------------------------------------------------------------------------------------------------------


def test_meta_codes_is_validated_against_the_questions():
    good = _record(3, codes={"pick": ["GX", "A", "LZ"]})
    assert validate_record(good) == []
    cases = {
        "wrong length": ({"pick": ["GX", "A"]}, "meta.codes 'pick': 2 assigned codes for 3 options"),
        "repeat": ({"pick": ["A", "A", "B"]}, "meta.codes 'pick': assigned codes repeat"),
        "outside": ({"pick": ["A", "B", "AC"]}, "meta.codes 'pick': codes outside the pool: ['AC']"),
        "noul": ({"spill": ["A", "B"]}, "meta.codes names 'spill', which is not a choice question of the request"),
        "unknown": ({"nope": ["A", "B"]}, "meta.codes names 'nope', which is not a choice question of the request"),
    }
    for name, (codes, problem) in cases.items():
        assert validate_record(_record(3, codes=codes)) == [problem], name
    assert validate_record(_record(3, codes=["A", "B", "C"])) and validate_record(_record(3, codes={}))


def test_assigned_codes_show_in_display_order_and_the_target_and_readout_follow_them():
    codes = ["GX", "A", "LZ", "MO", "Z"]
    record = _record(5, codes={"pick": codes}, correct=2)
    seen = set()
    for seed in range(60):
        (example,) = E.build_examples(record, ["pick"], tokenizer(), order_seed=seed)
        assert example["codes"] == codes and example["code_ids"] == C.verify_pool_codes(tokenizer(), codes)
        assert P.check_displayed_lines(example) == []
        position = example["options"].index("k002")
        assert example["target"][position] == 1.0 and sum(example["target"]) == 1.0
        assert f"\n{codes[position]}. k002: option number 2" in example["text"]
        seen.add(codes[position])
    assert seen == set(codes), "the order augmentation moves the correct option over every assigned code"
    (noul,) = E.build_examples(record, ["spill"], tokenizer(), order_seed=3)
    assert noul["codes"] == ["A", "B"], "yes/no keeps its two letters"
    dev = [d for d in E.dev_examples([record], tokenizer()) if d["question"] == "pick"]
    assert len(dev) == 5 and all(d["codes"] == codes for d in dev) and all(P.check_displayed_lines(d) == [] for d in dev)


def test_more_than_26_options_show_the_pool_codes_in_order():
    record = _record(40, mark=dict(OPTION_RANGE_MARK_255), correct=33)
    (example,) = E.build_examples(record, ["pick"], tokenizer(), order_seed=11)
    assert example["codes"] == list(C.CODE_POOL[:40]) and P.check_displayed_lines(example) == []
    assert example["code_ids"] == C.verify_pool_codes(tokenizer(), C.CODE_POOL[:40])
    assert example["target"][example["options"].index("k033")] == 1.0
    full = _record(255, mark=dict(OPTION_RANGE_MARK_255))
    (example,) = P.record_prompts(full, tokenizer(), seed=1, fmt="chat", orders="seeded")[:1]
    assert example["codes"] == list(C.CODE_POOL) and P.check_displayed_lines(example) == []
    assert example["text"].endswith("<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\nAnswer:")
    listings = [p for p in P.record_prompts(_record(30, mark=dict(OPTION_RANGE_MARK_255)), tokenizer(), seed=1, orders="counterbalanced") if p["question"] == "pick"]
    assert len(listings) == 30 and all(p["codes"] == list(C.CODE_POOL[:30]) for p in listings)


def test_assigned_codes_are_refused_for_a_question_they_do_not_fit():
    record = _record(3)
    request = {"state": record["request"]["state"], "questions": record["request"]["questions"]}
    with pytest.raises(ValueError, match="only for choice"):
        P.build_prompts(request, tokenizer(), key="k", seed=1, codes={"spill": ["A", "B"]})
    with pytest.raises(ValueError, match="assigned codes"):
        P.build_prompts(request, tokenizer(), key="k", seed=1, codes={"pick": ["A", "B"]})
    with pytest.raises(ValueError, match="does not ask"):
        P.build_prompts(request, tokenizer(), key="k", seed=1, codes={"other": ["A", "B"]})
    with pytest.raises(ValueError, match="not in the code pool"):
        P.build_prompts(request, tokenizer(), key="k", seed=1, codes={"pick": ["A", "B", "AC"]})


def test_the_line_check_reads_two_letter_codes_and_still_catches_a_desync():
    record = _record(30, mark=dict(OPTION_RANGE_MARK_255))
    (example,) = E.build_examples(record, ["pick"], tokenizer(), order_seed=5)
    broken = dict(example, codes=list(example["codes"]))
    broken["codes"][27] = "MO"
    assert P.check_displayed_lines(broken) == ["line 27: code 'AB' vs recorded 'MO'"]


def test_the_loss_reads_the_assigned_and_two_letter_code_rows():
    pytest.importorskip("transformers")
    from robo_jev.decisions import train as T
    from robo_jev.model.backbone_qwen import QwenBackbone, torch_reference_kernels

    wide = _record(40, mark=dict(OPTION_RANGE_MARK_255), correct=30)
    assigned = _record(4, codes={"pick": ["GX", "C", "MO", "AA"]}, rid="r2", correct=1)
    examples = E.build_examples(wide, ["pick"], tokenizer(), order_seed=3) + E.build_examples(assigned, ["pick"], tokenizer(), order_seed=3)
    with torch_reference_kernels():
        backbone = QwenBackbone.tiny(seed=4, vocab_size=tokenizer().get_vocab_size())
        out = T.code_kl(backbone, examples, code_mass=True)
        head = backbone.model.lm_head.weight
        hidden = backbone(torch.tensor([examples[1]["tokens"]]), torch.arange(len(examples[1]["tokens"]))[None])["hidden"][0, -1].float()
        expected = torch.softmax(head[torch.tensor(examples[1]["code_ids"])].float() @ hidden, 0).tolist()
    assert [len(p) for p in out["probs"]] == [40, 4]
    assert out["probs"][1] == pytest.approx(expected, abs=1e-5)
    assert all(math.isfinite(float(v)) for v in out["kl"]) and all(0.0 < m <= 1.0 for m in out["code_mass"])


# -- the unchanged path ------------------------------------------------------------------------------------------------------------------


def test_records_without_the_new_metas_give_the_base_codes_prompts_and_targets_byte_for_byte():
    tok = tokenizer()
    records = read_records(REQUESTS)
    digest = hashlib.sha256()
    count = 0
    for fmt in ("plain", "chat"):
        for mode in ("direct", "reasoning"):
            for orders in ("seeded", "counterbalanced"):
                for version in ("v0c-p2", "v0c-p3"):
                    for record in records:
                        for p in P.record_prompts(record, tok, seed=1, fmt=fmt, mode=mode, orders=orders, version=version):
                            digest.update(json.dumps({k: p[k] for k in FIELDS}, sort_keys=True).encode())
                            count += 1
    for record in records:
        for seed in (0, 7, 123):
            for e in E.build_examples(record, list(record["request"]["questions"]), tok, order_seed=seed):
                digest.update(json.dumps({k: e[k] for k in FIELDS + ("target", "clamped")}, sort_keys=True).encode())
                count += 1
    for e in E.dev_examples(records, tok):
        digest.update(json.dumps({k: e[k] for k in FIELDS + ("target", "clamped")}, sort_keys=True).encode())
        count += 1
    assert count == 1200 and digest.hexdigest() == BASE_FIXTURE_DIGEST


def test_a_record_with_codes_differs_from_its_plain_twin_only_in_the_code_letters():
    plain = _record(4, correct=3)
    coded = copy.deepcopy(plain)
    coded["meta"]["codes"] = {"pick": ["A", "B", "C", "D"]}
    a = E.build_examples(plain, ["pick", "spill"], tokenizer(), order_seed=9)
    b = E.build_examples(coded, ["pick", "spill"], tokenizer(), order_seed=9)
    assert [{k: x[k] for k in FIELDS + ("target",)} for x in a] == [{k: x[k] for k in FIELDS + ("target",)} for x in b]
