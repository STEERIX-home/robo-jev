"""H-prior loss weights (Task v0.8a, the FP arm) — `robo_jev.decisions.prior_weights` and the trainer's ``label_prior_weights`` option.

* Families are source × question name (trailing indices dropped); classes are the band (yes/no, score), the best level (``s1_band*``), the
  kind of the best action option (hold / continue / move) or the answer key.
* Within a family every class carries equal total weight, the mean weight is 1 and the weights are capped (synthetic data).
* The action-kind rule is checked on real option texts in English and Korean (copied from v0.7's training input) and on the records'
  structure (the engine's canonical names, the MuJoCo authors' plan).
* With the option off, the trainer's losses and gradients are byte-identical to the unweighted computation (the base trainer's
  ``step_loss``, reproduced here); with it on, every example's KL is multiplied by its weight and nothing else changes.
"""

from __future__ import annotations

import copy
import functools
import math
import statistics
from collections import Counter, defaultdict

import pytest
import torch
from helpers import FIXTURES

from robo_jev.decisions import prior_weights as W
from robo_jev.decisions.records import read_records

REQUESTS = FIXTURES / "decisions" / "requests.jsonl"

#: Real action options of v0.8's training input (v0.7's options), with the kind the engine's canonical name gives.
REAL_EN = [
    ("brake_firmly", "Come to a stop with firm, standard braking.", "hold"),
    ("a2", "Brake as hard as you can and stop as quickly as possible.", "hold"),
    ("ease_to_stop", "Ease to a stop using soft braking.", "hold"),
    ("stop_gentle", "Stop smoothly, braking at only 1.0 m/s².", "hold"),
    ("look_again", "Take another measurement first (about 1.0 s), then decide.", "hold"),
    ("a1", "Take a second look first; it takes about 1.0 s.", "hold"),
    ("call_someone", "Call someone over to check it (about 20.0 s).", "hold"),
    ("a2", "Stay where you are and let them pass.", "hold"),
    ("check_weight", "Check how the weight sits before placing (about 1.0 s).", "hold"),
    ("v1", "Make the call now without checking again.", "continue"),
    ("decide_now", "Act on the current information.", "continue"),
    ("keep_going", "Carry on as you are, without slowing.", "continue"),
    ("continue", "Do nothing more to it and continue with the task.", "continue"),
    ("a3", "Leave your speed unchanged (0.80 m/s) and continue.", "continue"),
    ("full_speed_turn", "Hold 1.20 m/s all the way through the corner.", "continue"),
    ("ramp_normal", "Drive the ramp as you normally would.", "continue"),
    ("place_now", "Put it down where you planned to.", "continue"),
    ("a1", "Cross now.", "continue"),
    ("a1", "Set off at your usual acceleration of 1.2 m/s².", "continue"),
    ("slow_down", "Cut your speed roughly in half but keep moving.", "move"),
    ("slow", "Slow to about half speed (0.40 m/s) without stopping.", "move"),
    ("sidestep_right", "Veer right and keep moving slowly.", "move"),
    ("a4", "Hug the left side of the aisle and continue at reduced speed.", "move"),
    ("press_down", "Push down on it to seat it (about 1.5 s).", "move"),
    ("retract", "Stop and pull the tool back, away from the hand.", "move"),
    ("secure_first", "Stop and secure the load first (about 2.0 s), then go.", "move"),
    ("a2", "Take the corner at 1.80 km/h.", "move"),
    ("redo_it", "Take it back out and start the move over (about 6.0 s).", "move"),
]
REAL_KO = [
    ("stop_moderate", "평소 하던 대로 단단하게 제동해서 완전히 멈춥니다.", "hold"),
    ("brake_hard", "가능한 한 짧은 거리에서 서도록 최대로 제동합니다(3.3 m/s²).", "hold"),
    ("stop_gentle", "약한 제동으로 서서히 멈춥니다.", "hold"),
    ("look_again", "먼저 한 번 더 잰 뒤(약 1.0 s) 정합니다.", "hold"),
    ("a2", "당분간 지금 위치를 지킵니다.", "hold"),
    ("check_weight", "먼저 잠깐 들어 무게 균형을 시험합니다(약 1.0 s).", "hold"),
    ("decide_now", "다시 확인하지 않고 지금 결정합니다.", "continue"),
    ("continue", "아무 조정 없이 현재 속도 그대로 진로를 따라갑니다.", "continue"),
    ("keep_going", "속도를 줄이지 않고 지금처럼 계속 갑니다.", "continue"),
    ("a1", "회전 때문에 감속하지 않습니다(1.2 m/s).", "continue"),
    ("ramp_normal", "평소처럼 경사로를 달립니다.", "continue"),
    ("continue", "놓인 그대로 받아들이고 계속합니다.", "continue"),
    ("slow", "멈추지 않고, 속도만 절반 정도(0.40 m/s)로 줄여서 이어 갑니다.", "move"),
    ("sidestep_left", "왼쪽으로 비켜서며 속도를 약간 낮춘 채 계속 움직입니다.", "move"),
    ("creep_forward", "속도를 크게 줄여 모퉁이 쪽으로 천천히 다가섭니다.", "move"),
    ("retract", "움직임을 멈추고 공구를 손과 먼 쪽으로 후퇴시킵니다.", "move"),
    ("press_down", "위에서 눌러 안착시킵니다(약 1.5 s).", "move"),
    ("a3", "속도를 현재의 절반가량인 0.40 m/s까지 낮춘 다음 계속 갑니다.", "move"),
]


@functools.lru_cache(maxsize=1)
def tokenizer():
    from robo_jev.model.tokenizer import available_tokenizer, load_tokenizer

    return load_tokenizer(available_tokenizer()[0])


def _q(record_id: str, questions: dict, labels: dict, meta: dict | None = None, generator: dict | None = None) -> dict:
    return {"schema": "simgen-request/0", "id": record_id, "generator": generator or {"version": "v0b"}, "meta": meta or {},
            "request": {"state": "A robot carries a tray.", "questions": questions}, "labels": labels}


# -- families, classes -------------------------------------------------------------------------------------------------------------------


def test_the_family_is_the_source_and_the_question_name_without_indices():
    assert W.base_name("s1_band_3") == "s1_band" and W.base_name("s1_cmp_1_2") == "s1_cmp" and W.base_name("action_2") == "action"
    assert W.base_name("spills_at_30pct") == "spills_at_30pct" and W.base_name("still_in_lane_2s") == "still_in_lane_2s"
    engine = _q("v0b-1", {}, {}, generator={"version": "v0b"})
    mujoco = _q("m-1", {}, {}, meta={"set": "v06-a2"})
    general = _q("g-1", {}, {}, meta={"source": {"set": "qasc", "row": 1}}, generator={"converter": "single-probe-v0"})
    d1 = _q("d-1", {}, {}, generator={"converter": "single-probe-v0", "source_generator": "gen-single-v0.2.0"})
    assert [W.source_of(r) for r in (engine, mujoco, general, d1)] == ["gen:v0b", "set:v06-a2", "general:qasc", "gen:gen-single-v0.2.0"]
    assert W.family_of(engine, "s1_band_4") == "gen:v0b|s1_band" and W.family_of(mujoco, "best_move") == "set:v06-a2|best_move"


def test_the_classes_follow_the_brief():
    from robo_jev.bench.system1 import band

    questions = {
        "touch": {"type": "noul", "instructions": "?"}, "risk": {"type": "score", "instructions": "?", "criteria": ["0", "1"]},
        "s1_band_2": {"type": "choice", "instructions": "?", "criteria": {"low": "a", "medium": "b", "high": "c"}},
        "action": {"type": "choice", "instructions": "?", "criteria": {"continue": "Keep going.", "stop_hard": "Brake hard.", "slow": "Slow down."}},
        "zone": {"type": "choice", "instructions": "?", "criteria": {"near": "a", "far": "b"}},
    }
    meta = {"options": {"action": {"continue": "continue", "stop_hard": "stop_hard", "slow": "slow"}}}
    for p_yes, value in ((0.0, 1.0), (1 / 3, 2 / 3), (0.3333, 0.6667), (0.5, 0.2)):
        labels = {"touch": {"type": "noul", "p_yes": p_yes}, "risk": {"type": "score", "value": value},
                  "s1_band_2": {"type": "choice", "probabilities": {"low": 0.1, "medium": 0.6, "high": 0.3}},
                  "action": {"type": "choice", "kind": "action", "probabilities": {"continue": 0.2, "stop_hard": 0.3, "slow": 0.5}},
                  "zone": {"type": "choice", "kind": "rule", "probabilities": {"near": 0.5, "far": 0.5}}}
        record = _q("v0b-1", questions, labels, meta=meta)
        assert W.label_class(record, "touch") == band(p_yes) and W.label_class(record, "risk") == band(value)
        assert W.label_class(record, "s1_band_2") == "level:medium"
        assert W.label_class(record, "action") == "action:move"
        assert W.label_class(record, "zone") == "key:far", "a tie goes to the smallest key"
    assert W.label_class(_q("v0b-1", questions, {**labels, "touch": {"p_yes": 1 / 3}}, meta=meta), "touch") == "middle"


# -- action kinds ----------------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("key, text, kind", REAL_EN + REAL_KO)
def test_the_text_rule_reads_real_option_texts_in_english_and_korean(key, text, kind):
    assert W.text_action_kind(key, text) == kind


def test_every_canonical_engine_action_has_a_kind_and_the_three_kinds_are_used():
    seen = {"continue", "stop_moderate", "stop_hard", "look_again", "stop_gentle", "slow", "reseat", "call_human", "sidestep_right", "sidestep_left",
            "press_down", "decide_now", "turn_80", "turn_60", "turn_100", "turn_30", "turn_45", "detour", "push_harder", "creep", "secure_first",
            "ramp_slow", "ramp_normal", "check_weight", "place_elsewhere", "place_now", "recentre", "retract", "stop_before", "wait", "go_now",
            "start_gentle", "start_normal", "start_quick"}  # every canonical name in v0.7's training input (34)
    assert seen == set(W.CANONICAL_KINDS) and set(W.CANONICAL_KINDS.values()) == set(W.ACTION_KINDS)
    assert {k for k, v in W.CANONICAL_KINDS.items() if v == "hold"} == {"stop_gentle", "stop_moderate", "stop_hard", "stop_before", "wait",
                                                                        "look_again", "call_human", "check_weight"}
    assert {k for k, v in W.CANONICAL_KINDS.items() if v == "continue"} == {"continue", "turn_100", "ramp_normal", "place_now", "start_normal",
                                                                            "go_now", "decide_now"}


def test_the_structure_decides_where_a_record_has_it():
    criteria = {"a1": "Keep going at 1.2 m/s.", "a2": "Brake hard.", "a3": "Slow down."}
    engine = _q("v0b-1", {"action": {"type": "choice", "instructions": "?", "criteria": criteria}},
                {"action": {"type": "choice", "kind": "action", "probabilities": {"a1": 0.1, "a2": 0.2, "a3": 0.7}}},
                meta={"options": {"action": {"a1": "continue", "a2": "stop_hard", "a3": "slow"}}})
    assert [W.action_kind(engine, "action", k) for k in criteria] == ["continue", "hold", "move"]
    engine["meta"]["options"]["action"]["a3"] = "teleport"
    with pytest.raises(ValueError, match="unknown canonical action"):
        W.action_kind(engine, "action", "a3")
    descent = {"stop_on_ramp": "ramp A at 0.46 m/s", "use_ramp_b": "ramp B at 0.46 m/s", "crawl_down": "ramp A at 0.23 m/s"}
    stop = {"brake_as_planned": "7.0 m/s2, lift as is", "soft_brake": "0.8 m/s2, lift as is", "drop_lift_first": "lift to 0.40 m, then 7.0 m/s2"}
    for options, plan in ((descent, {"stop_on_ramp": "planned", "use_ramp_b": "opt_b", "crawl_down": "opt_slow"}),
                          (stop, {"brake_as_planned": "planned", "soft_brake": "opt_soft_brake", "drop_lift_first": "opt_drop"})):
        record = _q("m-1", {"pick": {"type": "choice", "instructions": "?", "criteria": options}},
                    {"pick": {"type": "choice", "kind": "action", "probabilities": {k: 1 / 3 for k in options}}}, meta={"set": "v06-a1", "plan": {"options": plan}})
        kinds = [W.action_kind(record, "pick", k) for k in options]
        assert kinds == ["continue", "move", "move"], kinds
    # the two A1 texts the text rule alone reads otherwise (the plan option "stop_on_ramp", the alternative "lift as is")
    assert W.text_action_kind("stop_on_ramp", "ramp A at 0.46 m/s") == "hold" and W.text_action_kind("soft_brake", "0.8 m/s2, lift as is") == "continue"


# -- weights ---------------------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("counts", [{"low": 49, "middle": 16, "high": 35}, {"hold": 44, "continue": 14, "move": 42}, {"a": 1, "b": 1},
                                    {"o1": 10, "o2": 12, "o3": 9, "o4": 11}, {"only": 7}])
def test_uncapped_classes_carry_equal_total_weight_with_mean_one(counts):
    weights = W.class_weights(counts, cap=5.0)
    totals = {c: counts[c] * weights[c] for c in counts}
    n = sum(counts.values())
    assert sum(totals.values()) == pytest.approx(n)
    assert max(totals.values()) == pytest.approx(min(totals.values()))
    assert max(weights.values()) <= 5.0


def test_the_cap_binds_and_the_rest_keeps_the_mean_at_one():
    counts = {"a": 98, "b": 1, "c": 1}
    weights = W.class_weights(counts, cap=5.0)
    assert weights["b"] == weights["c"] == 5.0
    assert weights["a"] == pytest.approx(90 / 98)
    assert sum(counts[c] * weights[c] for c in counts) == pytest.approx(100)
    chain = W.class_weights({"a": 100, "b": 10, "c": 1}, cap=5.0)  # capping c raises λ, which pushes b over the cap in the next round
    assert chain["c"] == chain["b"] == 5.0 and chain["a"] == pytest.approx(0.56)
    assert sum(n * chain[c] for c, n in {"a": 100, "b": 10, "c": 1}.items()) == pytest.approx(111)
    loose = W.class_weights({"a": 100, "b": 30, "c": 1}, cap=5.0)  # b stays under the cap: a and b carry equal totals
    assert loose["c"] == 5.0 and loose["a"] * 100 == pytest.approx(loose["b"] * 30) and loose["b"] < 5.0
    with pytest.raises(ValueError):
        W.class_weights(counts, cap=0.5)
    assert W.class_weights(counts, cap=1.0) == {"a": 1.0, "b": 1.0, "c": 1.0}


def test_plan_weights_count_the_plans_own_examples_per_family():
    questions = {"touch": {"type": "noul", "instructions": "?"}, "zone": {"type": "choice", "instructions": "?", "criteria": {"x": "a", "y": "b"}}}
    records = []
    for i in range(12):
        p_yes = 0.9 if i < 9 else 0.1
        records.append(_q(f"v0b-{i}", questions, {"touch": {"type": "noul", "p_yes": p_yes},
                                                  "zone": {"type": "choice", "kind": "rule", "probabilities": {"x": 1.0, "y": 0.0}}}))
    records.append(_q("m-0", questions, {"touch": {"type": "noul", "p_yes": 0.5}, "zone": {"type": "choice", "kind": "rule", "probabilities": {"x": 0.0, "y": 1.0}}},
                      meta={"set": "mj"}))
    plan = [(i, "touch", 0) for i in range(13)] + [(i, "zone", 0) for i in range(13)] + [(0, "touch", 5)]  # record 0 twice
    weights, table = W.plan_weights(records, plan, cap=5.0)
    by = defaultdict(float)
    for (index, name, _), w in zip(plan, weights):
        by[(W.family_of(records[index], name), W.label_class(records[index], name))] += w
    assert by[("gen:v0b|touch", "high")] == pytest.approx(by[("gen:v0b|touch", "low")]) == pytest.approx(13 / 2)
    assert weights[-1] == weights[0], "a question picked twice counts twice and keeps its class weight"
    assert table["families"]["gen:v0b|zone"]["classes"] == {"key:x": {"n": 12, "share": 1.0, "weight": 1.0}}
    assert table["families"]["set:mj|touch"]["n"] == 1 and table["n_families"] == 4 and table["examples"] == len(plan)
    fam = table["families"]["gen:v0b|touch"]
    assert fam["mean"] == pytest.approx(1.0) and fam["ess"] < fam["n"] and fam["min"] < 1.0 < fam["max"]


# -- the trainer -----------------------------------------------------------------------------------------------------------------------


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
    """The unweighted trainer's ``step_loss`` as in the base 88653e1 (type weights × KL + λ · −log code mass, per micro-batch, / n)."""
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


def test_with_the_option_off_losses_and_gradients_are_byte_identical_to_the_unweighted_trainer(tmp_path):
    from robo_jev.model.backbone_qwen import torch_reference_kernels

    records = read_records(REQUESTS)
    with torch_reference_kernels():
        off = _trainer(tmp_path / "off", records)
        assert off.config["label_prior_weights"] is False and "label_prior_weights" not in off.manifest["objective"]
        examples = off.step_examples(0)
        assert all("prior_weight" not in e for e in examples)
        loss, _ = off.step_loss(examples, backward=True)
        mine = _grads(off)
        ref = _trainer(tmp_path / "ref", records)
        base = _base_step_loss(ref, ref.step_examples(0))
        theirs = _grads(ref)
    assert loss == base
    assert set(mine) == set(theirs) and all(torch.equal(mine[k], theirs[k]) for k in mine)


def _with_flipped_twins(records: list[dict]) -> list[dict]:
    """The fixture requests plus a twin of each whose yes/no and score labels are flipped (1 − p) — so that families hold several classes."""
    twins = []
    for record in records:
        twin = copy.deepcopy(record)
        twin["id"] = record["id"] + "-flip"
        for name, label in twin["labels"].items():
            if "p_yes" in label:
                label["p_yes"] = 1.0 - float(label["p_yes"])
            if "value" in label:
                label["value"] = 1.0 - float(label["value"])
        twins.append(twin)
    return list(records) + twins


def test_with_the_option_on_each_kl_is_multiplied_by_its_class_weight(tmp_path):
    from robo_jev.model.backbone_qwen import torch_reference_kernels

    records = _with_flipped_twins(read_records(REQUESTS))
    type_w = None
    seen = []
    with torch_reference_kernels():
        on = _trainer(tmp_path / "on", records, label_prior_weights=True, label_prior_cap=5.0)
        type_w = on.config["type_weights"]
        for step in range(on.steps_per_epoch):
            examples = on.step_examples(step)
            loss, parts = on.step_loss(examples, backward=False)
            expected = sum(type_w[e["type"]] * e["prior_weight"] * kl for e, kl in zip(examples, parts["kl"])) / len(examples)
            assert parts["kl_loss"] == pytest.approx(expected, rel=1e-6)
            assert parts["aux_loss"] == pytest.approx(0.2 * sum(-math.log(m) for m in parts["code_mass"]) / len(examples), rel=1e-6)
            seen += [e["prior_weight"] for e in examples]
        weights = [e["prior_weight"] for e in on.step_examples(0)]
        metrics = on.train_step()
    assert len(set(seen)) > 1 and len(seen) == on.examples_per_epoch
    summary = on.manifest["objective"]["label_prior_weights"]
    assert summary["version"] == W.PRIOR_WEIGHTS_VERSION and summary["cap"] == 5.0 and summary["examples"] == on.examples_per_epoch
    plan_weights, table = on._prior(0)
    by_family = defaultdict(list)
    for (index, name, _), w in zip(on._plan(0), plan_weights):
        by_family[W.family_of(on.train_records[index], name)].append(w)
    assert all(statistics.fmean(ws) == pytest.approx(1.0) for ws in by_family.values())
    assert metrics["prior_weight_mean"] == pytest.approx(statistics.fmean(weights))


def test_the_option_is_a_boolean_and_the_cap_at_least_one():
    from robo_jev.decisions import train as T

    config = T.resolve_config({"train_input": "x.jsonl"})
    assert config["label_prior_weights"] is False and config["label_prior_cap"] == 5.0
    assert T.resolve_config({"train_input": "x.jsonl", "label_prior_weights": True, "label_prior_cap": 3})["label_prior_cap"] == 3.0
    for bad in ({"label_prior_weights": "yes"}, {"label_prior_cap": 0.5}, {"label_prior_cap": "5"}):
        with pytest.raises(ValueError):
            T.resolve_config({"train_input": "x.jsonl", **bad})


def test_a_checkpoint_saved_before_the_option_existed_resumes_as_off(tmp_path):
    from robo_jev.model.backbone_qwen import torch_reference_kernels

    from robo_jev.decisions import train as T

    records = read_records(REQUESTS)
    with torch_reference_kernels():
        run = _trainer(tmp_path, records, stop_after_steps=1, save_checkpoints=True)
        result = run.run()
        state = torch.load(result["checkpoint"], map_location="cpu", weights_only=False)
        for key in ("label_prior_weights", "label_prior_cap"):
            state["config"].pop(key)
        torch.save(state, result["checkpoint"])
        again = T.DecisionTrainer(copy.deepcopy(run.config) | {"stop_after_steps": None}, records=records, tokenizer=tokenizer(), log=None,
                                  resume=result["checkpoint"])
        assert again.step == 1
        with pytest.raises(ValueError, match="label_prior_weights"):
            T.DecisionTrainer(copy.deepcopy(run.config) | {"stop_after_steps": None, "label_prior_weights": True}, records=records,
                              tokenizer=tokenizer(), log=None, resume=result["checkpoint"])
