"""`robo_jev.infer.RoboJev` and the standalone runner `robo_jev_decide.py` (Task hf0 §2).

On the small random Qwen (CPU, the torch reference kernels) with the real tokenizer:

* ``decide()``/``score_request()`` give **byte-identical** answers to ``scripts/diagnose_decisions.py`` for the same request, model,
  key, seed and order — both the single order (``seeded``) and the balanced one (``counterbalanced``);
* the answers have the Decisions API shape;
* the standalone runner builds the same prompts token for token and gives the same answers, and imports nothing from this package;
* (Task rel07) both runners take choice questions with 2–26 options and refuse 27: the 26 answer letters are single tokens with the pinned
  tokenizer, a 9–26-option request gets the training and diagnosis prompts of a record that opts in to many options
  (``meta.option_range``), and the handwritten 21- and 26-option requests get byte-identical answers from both runners.
"""

import ast
import functools
import importlib.util
import json
import math
import sys

import pytest
from helpers import FIXTURES, REPO

pytest.importorskip("transformers")
from robo_jev.decisions.examples import build_examples  # noqa: E402
from robo_jev.decisions.prompt import build_prompts, record_prompts  # noqa: E402
from robo_jev.decisions.records import OPTION_RANGE_MARK, SCHEMA, read_records, shuffle_key, validate_record  # noqa: E402
from robo_jev.infer import CHOICE_OPTIONS, RoboJev, default_key, file_sha256, to_decision  # noqa: E402
from robo_jev.model.backbone_qwen import QwenBackbone, torch_reference_kernels  # noqa: E402
from robo_jev.model.tokenizer import available_tokenizer, load_tokenizer  # noqa: E402
from robo_jev.model.zero_shot import verify_codes  # noqa: E402

REQUESTS = FIXTURES / "decisions" / "requests.jsonl"
MANY_OPTIONS = FIXTURES / "decisions" / "many-options.jsonl"
DIAGNOSE = REPO / "scripts" / "diagnose_decisions.py"
STANDALONE = REPO / "src" / "robo_jev" / "hf" / "robo_jev_decide.py"
ORDERS = {"single": "seeded", "balanced": "counterbalanced"}
#: The tokenizer every robo-jev checkpoint was trained with (Qwen/Qwen3.8-27B tokenizer.json; the training manifests' sha256).
TRAINING_TOKENIZER_SHA256 = "0997f410c57a1f4e53b09e4be8f4a172d90edd9564368fb0847030937229b9f3"


@functools.lru_cache(maxsize=1)
def tokenizer():
    return load_tokenizer(available_tokenizer()[0])


@functools.lru_cache(maxsize=1)
def tiny_model():
    """The diagnosis's ``--model tiny`` (seed 4, the tokenizer's vocabulary) — the same weights every time."""
    return QwenBackbone.tiny(seed=4, vocab_size=tokenizer().get_vocab_size()).model


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@functools.lru_cache(maxsize=1)
def diagnose():
    return _load(DIAGNOSE, "diagnose_decisions_for_infer")


@functools.lru_cache(maxsize=1)
def standalone():
    return _load(STANDALONE, "robo_jev_decide_standalone")


def records():
    return read_records(REQUESTS)


@pytest.mark.parametrize("orders", sorted(ORDERS))
def test_decide_is_byte_identical_to_the_diagnosis(tmp_path, orders):
    out = tmp_path / "run.json"
    with torch_reference_kernels():
        assert diagnose().main(["--model", "tiny", "--input", str(REQUESTS), "--out", str(out), "--orders", ORDERS[orders], "--batch", "1",
                                "--n-boot", "10"]) == 0
        model = RoboJev(tiny_model(), tokenizer())
        reference = {p["id"]: p for p in map(json.loads, out.with_suffix(".predictions.jsonl").read_text(encoding="utf-8").splitlines())}
        for record in records():
            request = record["request"]
            scored = model.score_request(request["state"], request["questions"], orders=orders, key=shuffle_key(record), seed=1)
            answers = model.decide(request["state"], request["questions"], orders=orders, key=shuffle_key(record), seed=1)
            expected = reference[record["id"]]["questions"]
            assert list(scored) == list(expected) == list(answers)
            for name, question in expected.items():
                mine = scored[name]
                assert [l["options"] for l in mine["listings"]] == [l["options"] for l in question["listings"]]
                assert [l["code_probs"] for l in mine["listings"]] == [l["code_probs"] for l in question["listings"]]  # exact floats
                assert json.dumps(mine["answer"], sort_keys=True) == json.dumps(question["answer"], sort_keys=True)
                decision = to_decision(request["questions"][name], question["answer"])
                assert json.dumps(answers[name], sort_keys=True) == json.dumps(decision, sort_keys=True)


def test_decide_returns_the_decisions_api_shape():
    record = records()[0]
    request = record["request"]
    with torch_reference_kernels():
        model = RoboJev(tiny_model(), tokenizer())
        answers = model.decide(request["state"], request["questions"])
        balanced = model.decide(request["state"], request["questions"], orders="balanced")
    kinds = {name: q["type"] for name, q in request["questions"].items()}
    assert set(answers) == set(kinds) == set(balanced)
    for name, answer in answers.items():
        assert answer["type"] == kinds[name]
        if kinds[name] == "choice":
            keys = list(request["questions"][name]["criteria"])
            assert list(answer["probabilities"]) == keys and math.isclose(sum(answer["probabilities"].values()), 1.0, abs_tol=1e-6)
            assert answer["choice"] == max(keys, key=answer["probabilities"].get)
        elif kinds[name] == "noul":
            assert set(answer) == {"type", "noul"} and 0.0 <= answer["noul"] <= 1.0
        else:
            assert list(answer["probabilities"]) == [f"{i / 10:.1f}" for i in range(11)]
            assert math.isclose(sum(answer["probabilities"].values()), 1.0, abs_tol=1e-6) and 0.0 <= answer["score"] <= 1.0
            assert math.isclose(answer["score"], sum(float(k) * p for k, p in answer["probabilities"].items()), abs_tol=1e-9)


def test_the_default_key_is_a_hash_of_the_state_and_seed_none_keeps_the_request_order():
    record = records()[0]
    request = record["request"]
    assert default_key(request["state"]) == default_key(request["state"]) != default_key(request["state"] + " ")
    assert len(default_key(request["state"])) == 16
    with torch_reference_kernels():
        model = RoboJev(tiny_model(), tokenizer())
        implicit = model.decide(request["state"], request["questions"])
        explicit = model.decide(request["state"], request["questions"], key=default_key(request["state"]))
        assert implicit == explicit
        unshuffled = model.score_request(request["state"], request["questions"], seed=None)
    for name, question in request["questions"].items():
        shown = unshuffled[name]["listings"][0]["options"]
        if question["type"] == "choice":
            assert shown == list(question["criteria"])
        elif question["type"] == "noul":
            assert shown == ["yes", "no"]
        else:
            assert shown == [f"{i / 10:.1f}" for i in range(11)]


def test_decide_refuses_bad_requests():
    record = records()[0]
    request = record["request"]
    model = RoboJev(tiny_model(), tokenizer())
    with pytest.raises(ValueError, match="orders"):
        model.decide(request["state"], request["questions"], orders="counterbalanced")
    with pytest.raises(ValueError, match="not a valid decision request"):
        model.decide(request["state"], {f"q{i}": {"type": "noul", "instructions": "Is it?"} for i in range(13)})
    with pytest.raises(ValueError, match="not a valid decision request"):
        model.decide(request["state"], {"q": {"type": "rank", "instructions": "Rank them."}})
    with pytest.raises(ValueError, match="not a valid decision request"):
        model.decide("", {"q": {"type": "noul", "instructions": "Is it?"}})


@pytest.mark.parametrize("orders", sorted(ORDERS))
def test_the_standalone_runner_builds_the_same_prompts(orders):
    runner = standalone()
    for record in records():
        request = record["request"]
        for key, seed in ((shuffle_key(record), 1), (None, 1), ("k", 7), (None, None)):
            ours = build_prompts(request, tokenizer(), key=default_key(request["state"]) if key is None else key, seed=seed, fmt="chat",
                                 mode="direct", orders=ORDERS[orders])
            theirs = runner.build_prompts(request["state"], request["questions"], tokenizer(), key=key, seed=seed, orders=orders)
            assert [(p["question"], p["type"], p["listing"], p["options"], p["codes"], p["code_ids"], p["text"], p["tokens"]) for p in ours] == [
                (p["question"], p["type"], p["listing"], p["options"], p["codes"], p["code_ids"], p["text"], p["tokens"]) for p in theirs
            ]
            assert [p["values"] for p in ours] == [p["values"] for p in theirs]


@pytest.mark.parametrize("orders", sorted(ORDERS))
def test_the_standalone_runner_gives_the_same_answers(orders):
    runner = standalone()
    with torch_reference_kernels():
        ours = RoboJev(tiny_model(), tokenizer())
        theirs = runner.RoboJevRunner(tiny_model(), tokenizer())
        for record in records():
            request = record["request"]
            for key in (shuffle_key(record), None):
                a = ours.decide(request["state"], request["questions"], orders=orders, key=key)
                b = theirs.decide(request["state"], request["questions"], orders=orders, key=key)
                assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)


def test_the_standalone_runner_imports_nothing_from_this_package():
    tree = ast.parse(STANDALONE.read_text(encoding="utf-8"))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported |= {alias.name.split(".")[0] for alias in node.names}
        elif isinstance(node, ast.ImportFrom):
            imported.add((node.module or "").split(".")[0])
    assert "robo_jev" not in imported
    assert imported <= {"__future__", "argparse", "hashlib", "json", "os", "random", "re", "sys", "typing", "torch", "tokenizers", "transformers",
                        "peft", "huggingface_hub"}
    assert standalone().PROMPT_VERSION == "v0c-p3" and standalone().SYSTEM_TEXT == __import__("robo_jev.decisions.prompt").decisions.prompt.SYSTEM_TEXT


def test_the_standalone_runner_validates_requests():
    runner = standalone()
    with pytest.raises(ValueError, match="orders"):
        runner.build_prompts("A state.", {"q": {"type": "noul", "instructions": "Is it?"}}, tokenizer(), orders="counterbalanced")
    with pytest.raises(ValueError, match="1..12"):
        runner.build_prompts("A state.", {}, tokenizer())
    with pytest.raises(ValueError, match="1..12"):
        runner.build_prompts("A state.", {f"q{i}": {"type": "noul", "instructions": "Is it?"} for i in range(13)}, tokenizer())
    with pytest.raises(ValueError, match=r"2\.\.26 option keys to descriptions \(got 1\)"):
        runner.build_prompts("A state.", {"q": {"type": "choice", "instructions": "Pick.", "criteria": {"a": "only one"}}}, tokenizer())
    with pytest.raises(ValueError, match="max 16,000"):
        runner.build_prompts("x" * 16_001, {"q": {"type": "noul", "instructions": "Is it?"}}, tokenizer())
    assert runner.build_prompts("x" * 16_000, {"q": {"type": "noul", "instructions": "Is it?"}}, tokenizer())
    assert runner.LIMITS == {"state_chars": 16_000, "questions": (1, 12), "choice_options": (2, 26)} and CHOICE_OPTIONS == (2, 26)


# --------------------------------------------------------------------------
# Many options (Task rel07): 2..26 in both runners
# --------------------------------------------------------------------------


def many_option_record(n: int, *, mark: bool = True) -> dict:
    """A request with an `n`-option choice (keys spot_00…, in that order) and a yes/no question, as a labelled record that opts in to
    many options (``meta.option_range``) unless `mark` is false."""
    criteria = {f"spot_{i:02d}": f"Floor spot {i}: {'flat and clear' if i % 3 else 'on the ramp'}, {i + 1} m from the dock." for i in range(n)}
    request = {"state": "You are a warehouse robot carrying a tall carton on your forks.\n- Speed 0.6 m/s.\n- The staging area is clear.",
               "questions": {"place": {"type": "choice", "instructions": "Where should the robot put the carton down?", "criteria": criteria},
                             "tip": {"type": "noul", "instructions": "If it puts the carton on the ramp, will the carton tip over?"}}}
    probabilities = {key: (1.0 if key == "spot_01" else 0.0) for key in criteria}
    record = {"schema": SCHEMA, "id": f"many-{n}", "meta": {"option_range": dict(OPTION_RANGE_MARK)} if mark else {}, "request": request,
              "labels": {"place": {"type": "choice", "kind": "action", "probabilities": probabilities}, "tip": {"type": "noul", "p_yes": 0.2}}}
    return record


def _shown(prompts):
    return [(p["question"], p["type"], p["listing"], p["options"], p["codes"], p["code_ids"], p["text"], p["tokens"]) for p in prompts]


def test_the_26_answer_letters_are_single_tokens_with_the_pinned_tokenizer():
    assert file_sha256(available_tokenizer()[1]) == TRAINING_TOKENIZER_SHA256
    codes, ids = standalone()._verify_codes(tokenizer(), 26)
    assert codes == list("ABCDEFGHIJKLMNOPQRSTUVWXYZ") and len(set(ids)) == 26
    assert (codes, ids) == tuple(verify_codes(tokenizer(), 26))
    for code, token in zip(codes, ids):
        assert tokenizer().encode(" " + code, add_special_tokens=False).ids == [token]
        assert tokenizer().encode("Answer: " + code, add_special_tokens=False).ids[-1] == token
    with pytest.raises(ValueError, match="more than 26 codes"):
        standalone()._verify_codes(tokenizer(), 27)


@pytest.mark.parametrize("n", [9, 21, 26])
def test_many_option_prompts_equal_the_training_and_diagnosis_prompts(n):
    record = many_option_record(n)
    assert validate_record(record) == []
    assert any("choice needs 2..8 options" in problem for problem in validate_record(many_option_record(n, mark=False)))  # records: 2–8 by default
    runner = standalone()
    request = record["request"]
    for orders in sorted(ORDERS):
        diagnosis = record_prompts(record, tokenizer(), seed=1, fmt="chat", mode="direct", orders=ORDERS[orders])
        bundle = runner.build_prompts(request["state"], request["questions"], tokenizer(), key=shuffle_key(record), seed=1, orders=orders)
        assert _shown(diagnosis) == _shown(bundle)
        assert [p["values"] for p in diagnosis] == [p["values"] for p in bundle]
    balanced = [p for p in runner.build_prompts(request["state"], request["questions"], tokenizer(), key=shuffle_key(record), orders="balanced")
                if p["type"] == "choice"]
    assert len(balanced) == n and all(p["codes"] == [chr(ord("A") + i) for i in range(n)] for p in balanced)
    for position in range(n):  # balanced: every option in every position once
        assert sorted(p["options"][position] for p in balanced) == sorted(request["questions"]["place"]["criteria"])
    assert all(("\nZ. " in p["text"]) == (n == 26) for p in balanced)
    for order_seed in (0, 1, 7):  # the trainer's prompts (seeded order per example)
        training = build_examples(record, list(request["questions"]), tokenizer(), order_seed=order_seed, fmt="chat")
        bundle = runner.build_prompts(request["state"], request["questions"], tokenizer(), key=shuffle_key(record), seed=order_seed)
        assert [entry[:2] + entry[3:] for entry in _shown(training)] == [entry[:2] + entry[3:] for entry in _shown(bundle)]


@pytest.mark.parametrize("n", [9, 21, 26])
def test_both_runners_answer_up_to_26_options_identically(n):
    request = many_option_record(n)["request"]
    with torch_reference_kernels():
        ours = RoboJev(tiny_model(), tokenizer())
        theirs = standalone().RoboJevRunner(tiny_model(), tokenizer())
        for orders in sorted(ORDERS):
            a = ours.decide(request["state"], request["questions"], orders=orders)
            b = theirs.decide(request["state"], request["questions"], orders=orders)
            assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)
            assert list(a["place"]["probabilities"]) == list(request["questions"]["place"]["criteria"])
            assert math.isclose(sum(a["place"]["probabilities"].values()), 1.0, abs_tol=1e-6)
            assert a["place"]["choice"] == max(a["place"]["probabilities"], key=a["place"]["probabilities"].get)


def test_both_runners_refuse_27_options():
    request = many_option_record(27)["request"]
    with pytest.raises(ValueError, match=r"2\.\.26 option keys to descriptions \(got 27\)"):
        standalone().build_prompts(request["state"], request["questions"], tokenizer())
    with pytest.raises(ValueError, match=r"choice needs 2\.\.26 options \(has 27\)"):
        RoboJev(tiny_model(), tokenizer()).decide(request["state"], request["questions"])


def test_the_handwritten_21_and_26_option_requests_get_the_same_answers_from_both_runners():
    records = read_records(MANY_OPTIONS)
    assert [r["id"] for r in records] == ["fx-m21", "fx-m26"]
    assert [max(len(q.get("criteria") or []) for q in r["request"]["questions"].values()) for r in records] == [21, 26]
    assert all(validate_record(r, require_labels=False) == [] for r in records)  # they carry the many-option opt-in
    with torch_reference_kernels():
        ours = RoboJev(tiny_model(), tokenizer())
        theirs = standalone().RoboJevRunner(tiny_model(), tokenizer())
        for record in records:
            request = record["request"]
            for key in (None, shuffle_key(record)):
                a = ours.decide(request["state"], request["questions"], key=key)
                b = theirs.decide(request["state"], request["questions"], key=key)
                assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)
