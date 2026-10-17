"""The order-consistency loss (Task v0.9a, arm C) — the trainer's ``consistency`` option and the permuted copies of
:mod:`robo_jev.decisions.examples`.

* With the option off, the trainer's losses and gradients are byte-identical to today's (the base 61d36e4 ``step_loss``, reproduced here,
  and ``code_kl`` with and without its new ``logits`` output).
* The JS term is 0 for order-invariant (content-keyed) logits and positive for position-keyed ones.
* A copy maps back to the option contents for sequential and random (``meta.codes``) codes: the copy shows the same codes by position, the
  contents permuted; the shift copy is the diagnosis's counterbalanced listing of the same order, token for token.
* Gradients flow through both copies (each copy's logits get a gradient, and each copy changes the parameters' gradients).
* The loss holds the usual target terms on the originals only plus λ × the mean JS / the step's examples.
No model is loaded beyond the tiny random Qwen on the CPU.
"""

from __future__ import annotations

import copy
import functools
import math

import pytest
import torch
from helpers import FIXTURES

from robo_jev.decisions import examples as E
from robo_jev.decisions.records import read_records

REQUESTS = FIXTURES / "decisions" / "requests.jsonl"
MANY = FIXTURES / "decisions" / "many-options.jsonl"


@functools.lru_cache(maxsize=1)
def tokenizer():
    from robo_jev.model.tokenizer import available_tokenizer, load_tokenizer

    return load_tokenizer(available_tokenizer()[0])


def _labelled(records: list[dict]) -> list[dict]:
    """The many-option fixture's requests with labels (it has none): the second option of each choice question, yes 0.3, score 0.6."""
    out = []
    for record in records:
        record = copy.deepcopy(record)
        labels = {}
        for name, question in record["request"]["questions"].items():
            if question["type"] == "choice":
                keys = list(question["criteria"])
                labels[name] = {"type": "choice", "probabilities": {k: (1.0 if i == 1 else 0.0) for i, k in enumerate(keys)}}
            elif question["type"] == "noul":
                labels[name] = {"type": "noul", "p_yes": 0.3}
            else:
                labels[name] = {"type": "score", "value": 0.6}
        record["labels"] = labels
        out.append(record)
    return out


def _many() -> list[dict]:
    return _labelled(read_records(MANY))


def _with_codes(record: dict, name: str, codes: list[str]) -> dict:
    out = copy.deepcopy(record)
    out.setdefault("meta", {})["codes"] = {name: list(codes)}
    return out


# -- the copies ----------------------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("n", [2, 3, 4, 5, 8, 21, 26, 255])
def test_the_shift_moves_every_option_half_way_round_and_the_reverse_reverses(n):
    shift, reverse = E.copy_permutation("shift", n), E.copy_permutation("reverse", n)
    s = math.ceil(n / 2)
    assert shift == [(j + s) % n for j in range(n)] and sorted(shift) == list(range(n))
    assert reverse == list(range(n))[::-1]
    assert (shift == reverse) is (n == 2), "only N = 2 makes the two copies the same order"
    with pytest.raises(ValueError):
        E.copy_permutation("swap", n)


def test_selection_is_a_fixed_share_of_questions_independent_of_the_run_seed():
    picked = [E.consistency_selected({"id": f"r{i}"}, "q", share=0.3) for i in range(20000)]
    assert abs(sum(picked) / len(picked) - 0.3) < 0.015
    assert picked == [E.consistency_selected({"id": f"r{i}"}, "q", share=0.3) for i in range(20000)], "deterministic"
    other = [E.consistency_selected({"id": f"r{i}"}, "q", share=0.3, salt="another") for i in range(20000)]
    assert other != picked, "the salt picks"
    smaller = [E.consistency_selected({"id": f"r{i}"}, "q", share=0.15) for i in range(20000)]
    assert all(a or not b for a, b in zip(picked, smaller)), "a smaller share picks a subset"


@pytest.mark.parametrize("codes", [None, "random"])
def test_a_copy_shows_the_same_codes_by_position_and_maps_back_to_the_contents(codes):
    record = read_records(REQUESTS)[7]  # fx-008: an 8-option action question
    name = "action"
    if codes == "random":
        record = _with_codes(record, name, ["KD", "B", "LZ", "MO", "AA", "Z", "HB", "E"][: len(record["request"]["questions"][name]["criteria"])])
    (example,) = E.build_examples(record, [name], tokenizer(), order_seed=12345)
    assert example["codes"] == (["KD", "B", "LZ", "MO", "AA", "Z", "HB", "E"] if codes else list("ABCDEFGH"))
    copies = E.consistency_copies(record, example, tokenizer())
    assert [c["copy"] for c in copies] == ["shift", "reverse"]
    for c in copies:
        perm = c["permutation"]
        assert c["codes"] == example["codes"] and c["code_ids"] == example["code_ids"], "codes stay by position"
        assert c["options"] == [example["options"][i] for i in perm], "contents move"
        assert [c["options"][c["back"][i]] for i in range(len(perm))] == example["options"], "back reads the copy in the original's order"
        assert [c["target"][c["back"][i]] for i in range(len(perm))] == example["target"]
        assert c["tokens"][: example["prefix_tokens"]] == example["tokens"][: example["prefix_tokens"]], "same prefix tokens"
        assert c["text"] != example["text"] and len(c["tokens"]) == len(example["tokens"])
        for j, key in enumerate(c["options"]):  # the shown line j carries code j and the content at copy position j
            assert f"\n{c['codes'][j]}. {key}:" in c["text"]


def test_the_identity_copy_is_the_original_and_the_shift_copy_is_the_diagnosis_listing():
    from robo_jev.decisions.prompt import build_prompts
    from robo_jev.decisions.records import shuffle_key

    for record in read_records(REQUESTS) + _many():
        for name, question in record["request"]["questions"].items():
            if question["type"] != "choice":
                continue
            (example,) = E.build_examples(record, [name], tokenizer(), order_seed=77)
            n = len(example["options"])
            same = E.permuted_copy(record, example, list(range(n)), tokenizer())
            assert same["text"] == example["text"] and same["tokens"] == example["tokens"]
            listings = build_prompts({"state": record["request"]["state"], "questions": {name: question}}, tokenizer(), key=shuffle_key(record),
                                     seed=77, fmt="chat", mode="direct", orders="counterbalanced")
            shift = E.permuted_copy(record, example, E.copy_permutation("shift", n), tokenizer())
            listing = listings[math.ceil(n / 2) % n]
            assert shift["text"] == listing["text"] and shift["tokens"] == listing["tokens"] and shift["options"] == listing["options"]
            assert shift["code_ids"] == listing["code_ids"]


def test_n_two_gets_one_copy():
    record = read_records(REQUESTS)[3]  # fx-004 zone: 2 options
    (example,) = E.build_examples(record, ["zone"], tokenizer(), order_seed=5)
    copies = E.consistency_copies(record, example, tokenizer())
    assert len(copies) == 1 and copies[0]["options"] == example["options"][::-1]


# -- the JS term ---------------------------------------------------------------------------------------------------------------------------


def _content_logits(options: list[str]) -> torch.Tensor:
    """Logits that depend only on each option's content (an order-invariant 'model')."""
    return torch.tensor([float((sum(map(ord, key)) * 7919) % 101) / 10.0 for key in options])


@pytest.mark.parametrize("codes", [None, "random"])
def test_the_js_term_is_zero_for_order_invariant_logits_and_positive_for_position_bias(codes):
    from robo_jev.decisions.train import consistency_js

    record = _many()[1]  # fx-m26: 26 options
    name = "place"
    if codes == "random":
        record = _with_codes(record, name, list(__import__("robo_jev.decisions.codes", fromlist=["CODE_POOL"]).CODE_POOL[100:126]))
    (example,) = E.build_examples(record, [name], tokenizer(), order_seed=9)
    original = _content_logits(example["options"])
    for c in E.consistency_copies(record, example, tokenizer()):
        assert float(consistency_js(original, _content_logits(c["options"]), c["back"])) == pytest.approx(0.0, abs=1e-6)
        positional = original.clone()  # the same logits by position: a pure position bias
        assert float(consistency_js(original, positional, c["back"])) > 1e-3
    a, b = torch.tensor([2.0, 0.0, -1.0]), torch.tensor([-1.0, 0.0, 2.0])
    p, q = torch.softmax(a, -1), torch.softmax(b, -1)
    m = (p + q) / 2
    expected = 0.5 * (p * (p / m).log()).sum() + 0.5 * (q * (q / m).log()).sum()
    assert float(consistency_js(a, b, [0, 1, 2])) == pytest.approx(float(expected), rel=1e-6)
    assert float(consistency_js(a, b, [2, 1, 0])) == pytest.approx(0.0, abs=1e-7), "b read back in reverse is a"
    assert float(consistency_js(a, b, [0, 1, 2])) <= math.log(2.0)


# -- the trainer -------------------------------------------------------------------------------------------------------------------------


def _trainer(tmp_path, records, **overrides):
    pytest.importorskip("transformers")
    from robo_jev.decisions import train as T

    config = {"model_id": "tiny", "device": "cpu", "dtype": "float32", "out_dir": str(tmp_path / "runs"), "run_name": "t", "mode": "lora",
              "lora": {"r": 4, "alpha": 8, "dropout": 0.0, "targets": ["q_proj", "v_proj", "in_proj_qkv", "out_proj", "down_proj"]},
              "batch_examples": 8, "micro_batch_tokens": 4096, "lr": 1e-3, "warmup_ratio": 0.25, "epochs": 1, "dev_fraction": 0.0,
              "dev_every": 0, "checkpoint_every": 0, "seed": 3, "type_weights": {"choice": 1.5, "noul": 1.0, "score": 0.5}, "code_mass_weight": 0.2}
    config.update(overrides)
    return T.DecisionTrainer(config, records=records, tokenizer=tokenizer(), log=None)


def _base_step_loss(trainer, examples):
    """Today's ``step_loss`` (base 61d36e4, label prior off): type weights × KL + λ · −log code mass, per micro-batch, / n."""
    from robo_jev.decisions import train as T

    weights = [trainer.config["type_weights"][e["type"]] for e in examples]
    lam = float(trainer.config["code_mass_weight"])
    total = 0.0
    for group in T.pack_micro_batches([len(e["tokens"]) for e in examples], trainer.config["micro_batch_tokens"]):
        out = T.code_kl(trainer.backbone, [examples[i] for i in group], code_mass=True, head32=trainer._head32)
        w = torch.tensor([weights[i] for i in group], dtype=torch.float32, device=out["kl"].device)
        loss = (w * out["kl"]).sum() / len(examples) + lam * (-out["log_code_mass"]).sum() / len(examples)
        loss.backward()
        total += float(loss.detach())
    return total


def _grads(trainer):
    return {name: p.grad.detach().clone() for name, p in trainer.trainable if p.grad is not None}


ON = {"share": 1.0, "lambda": 0.5, "copies": ["shift", "reverse"], "divergence": "js"}


def test_with_the_option_off_losses_and_gradients_are_byte_identical_to_todays_trainer(tmp_path):
    from robo_jev.model.backbone_qwen import torch_reference_kernels

    records = read_records(REQUESTS) + _many()
    with torch_reference_kernels():
        off = _trainer(tmp_path / "off", records)
        assert off.config["consistency"] is None and "consistency" not in off.manifest["objective"]
        for step in range(off.steps_per_epoch):
            examples = off.step_examples(step)
            assert all("copies" not in e for e in examples)
            off.optimizer.zero_grad(set_to_none=True)
            loss, parts = off.step_loss(examples, backward=True)
            assert "consistency_loss" not in parts and "js" not in parts
            mine = _grads(off)
            ref = _trainer(tmp_path / f"ref{step}", records)
            base = _base_step_loss(ref, ref.step_examples(step))
            theirs = _grads(ref)
            assert loss == base
            assert set(mine) == set(theirs) and all(torch.equal(mine[k], theirs[k]) for k in mine)
        metrics = off.train_step()
    assert not {"consistency_loss", "copy_rows", "js_mean"} & set(metrics)


def test_code_kl_is_unchanged_by_its_logits_output(tmp_path):
    from robo_jev.decisions import train as T
    from robo_jev.model.backbone_qwen import torch_reference_kernels

    records = read_records(REQUESTS)
    with torch_reference_kernels():
        run = _trainer(tmp_path, records)
        examples = run.step_examples(0)
        a = T.code_kl(run.backbone, examples, code_mass=True, head32=run._head32)
        b = T.code_kl(run.backbone, examples, code_mass=True, head32=run._head32, logits=True)
    assert torch.equal(a["kl"], b["kl"]) and torch.equal(a["ce"], b["ce"]) and torch.equal(a["log_code_mass"], b["log_code_mass"])
    assert a["probs"] == b["probs"] and "logits" not in a and b["logits"].shape[0] == len(examples)


def test_with_the_option_on_the_loss_is_the_target_terms_on_the_originals_plus_lambda_times_the_mean_js(tmp_path):
    from robo_jev.decisions import train as T
    from robo_jev.model.backbone_qwen import torch_reference_kernels

    records = read_records(REQUESTS) + _many()
    with torch_reference_kernels():
        on = _trainer(tmp_path / "on", records, consistency=ON)
        summary = on.manifest["objective"]["consistency"]
        assert summary["share"] == 1.0 and summary["picked_examples"] == summary["choice_examples"] > 0
        off = _trainer(tmp_path / "off", records)
        seen = 0
        for step in range(on.steps_per_epoch):
            examples = on.step_examples(step)
            copies = [e for e in examples if e.get("copies")]
            assert all((e["type"] == "choice") == bool(e.get("copies")) for e in examples), "share 1.0: every choice example, nothing else"
            loss, parts = on.step_loss(examples, backward=False)
            _, plain = off.step_loss(off.step_examples(step), backward=False)
            assert parts["kl"] == pytest.approx(plain["kl"], rel=1e-5, abs=1e-6) and parts["code_mass"] == pytest.approx(plain["code_mass"], rel=1e-5)
            assert parts["kl_loss"] == pytest.approx(plain["kl_loss"], rel=1e-5) and parts["aux_loss"] == pytest.approx(plain["aux_loss"], rel=1e-5)
            expected = 0.5 * sum(v for v in parts["js"] if v is not None) / len(examples)
            assert parts["consistency_loss"] == pytest.approx(expected, rel=1e-6)
            assert loss == pytest.approx(parts["kl_loss"] + parts["aux_loss"] + parts["consistency_loss"], rel=1e-6)
            assert parts["copy_rows"] == sum(len(e["copies"]) for e in copies) and parts["consistency_examples"] == len(copies)
            assert all(v is not None and v >= 0.0 for e, v in zip(examples, parts["js"]) if e.get("copies"))
            seen += len(copies)
        assert seen == summary["picked_examples"]
        metrics = on.train_step()
    assert metrics["consistency_examples"] > 0 and metrics["consistency_loss"] > 0 and metrics["copy_rows"] >= metrics["consistency_examples"]


def test_gradients_flow_through_both_copies(tmp_path):
    from robo_jev.decisions import train as T
    from robo_jev.model.backbone_qwen import torch_reference_kernels

    record = read_records(REQUESTS)[7]
    with torch_reference_kernels():
        run = _trainer(tmp_path / "g", [record], consistency=ON)
        (example,) = E.build_examples(record, ["action"], tokenizer(), order_seed=3)
        copies = E.consistency_copies(record, example, tokenizer())
        out = T.code_kl(run.backbone, [example, *copies], code_mass=True, head32=run._head32, logits=True)
        k = len(example["code_ids"])
        js = torch.stack([T.consistency_js(out["logits"][0, :k], out["logits"][1 + c, :k], cp["back"]) for c, cp in enumerate(copies)]).mean()
        (grad,) = torch.autograd.grad(js, out["logits"], retain_graph=True)
        assert all(float(grad[row, :k].abs().sum()) > 0 for row in range(3)), "the original and both copies get a gradient"

        def grads_with(kinds):
            trainer = _trainer(tmp_path / "-".join(kinds or ["off"]), [record], consistency=None if kinds is None else {**ON, "copies": kinds})
            (ex,) = E.build_examples(record, ["action"], tokenizer(), order_seed=3)
            if kinds is not None:
                ex["copies"] = E.consistency_copies(record, ex, tokenizer(), kinds=kinds)
            trainer.optimizer.zero_grad(set_to_none=True)
            trainer.step_loss([ex], backward=True)
            return _grads(trainer)

        off, both, shift, reverse = grads_with(None), grads_with(["shift", "reverse"]), grads_with(["shift"]), grads_with(["reverse"])
    keys = sorted(off)
    differs = lambda a, b: any(not torch.allclose(a[n], b[n], rtol=0, atol=1e-12) for n in keys)  # noqa: E731
    assert differs(both, off) and differs(shift, off) and differs(reverse, off) and differs(shift, reverse)
    for name in keys:  # the JS term's gradient is the mean of the two copies' (each copy's term is λ · JS / 2 within `both`)
        assert torch.allclose(both[name] - off[name], ((shift[name] - off[name]) + (reverse[name] - off[name])) / 2, rtol=1e-4, atol=1e-7)


def test_pack_units_with_single_rows_is_the_old_packing_and_keeps_a_unit_together():
    from robo_jev.decisions import train as T

    lengths = [37, 12, 400, 12, 99, 3000, 5, 64, 64, 1023]
    assert T.pack_units(lengths, [1] * len(lengths), 1024) == T.pack_micro_batches(lengths, 1024)
    groups = T.pack_units(lengths, [3, 1, 2, 3, 1, 3, 2, 1, 1, 3], 1024)
    assert sorted(i for g in groups for i in g) == list(range(len(lengths)))
    for g in groups:
        rows = sum([3, 1, 2, 3, 1, 3, 2, 1, 1, 3][i] for i in g)
        assert len(g) == 1 or rows * max(lengths[i] for i in g) <= 1024


def test_the_option_is_validated_and_old_checkpoints_resume_as_off(tmp_path):
    from robo_jev.decisions import train as T
    from robo_jev.model.backbone_qwen import torch_reference_kernels

    assert T.resolve_config({"train_input": "x"})["consistency"] is None
    got = T.resolve_config({"train_input": "x", "consistency": {"share": 0.3}})["consistency"]
    assert got == {"share": 0.3, "lambda": 0.5, "copies": ["shift", "reverse"], "divergence": "js", "salt": E.CONSISTENCY_SALT}
    for bad in ({"share": 0}, {"share": 1.5}, {"lambda": 0}, {"copies": []}, {"copies": ["shift", "shift"]}, {"copies": ["swap"]},
                {"divergence": "kl"}, {"salt": " "}, {"weight": 1}, "on"):
        with pytest.raises(ValueError):
            T.resolve_config({"train_input": "x", "consistency": bad})
    records = read_records(REQUESTS)
    with torch_reference_kernels():
        run = _trainer(tmp_path, records, stop_after_steps=1, save_checkpoints=True)
        result = run.run()
        state = torch.load(result["checkpoint"], map_location="cpu", weights_only=False)
        state["config"].pop("consistency")
        torch.save(state, result["checkpoint"])
        again = T.DecisionTrainer(copy.deepcopy(run.config) | {"stop_after_steps": None}, records=records, tokenizer=tokenizer(), log=None,
                                  resume=result["checkpoint"])
        assert again.step == 1
        with pytest.raises(ValueError, match="consistency"):
            T.DecisionTrainer(copy.deepcopy(run.config) | {"stop_after_steps": None, "consistency": ON}, records=records, tokenizer=tokenizer(),
                              log=None, resume=result["checkpoint"])
