"""Task B1 — the adapters from mjgen records to the System-1 metrics (robust keys, group clusters, direction on one-fact pairs,
comparison on the in-record comparison questions, band confusion, answer profiles) and the pre-declared rule of the cause check
(R1 / R2 / R3 and the positive control) applied by ``scripts/b1_verdict.py``."""

import importlib.util
import json
import sys

import pytest
from helpers import REPO

from robo_jev.bench import infamily, system1
from robo_jev.decisions.records import file_sha256
from robo_jev.mjgen import families as fam
from robo_jev.mjgen import generate as gen

FRESH = 202610131


def _script(name: str):
    spec = importlib.util.spec_from_file_location(f"{name}_b1test", REPO / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_the_constants_of_the_adapter_agree_with_the_generator():
    assert infamily.EVENT == {k: tuple(v) for k, v in fam.EVENT.items()}
    assert infamily.PLAN_KEY == gen.PLANNED_KEY


# -- adapters onto the System-1 metrics ---------------------------------------------------------------------------------


def synthetic(rid: str, group: int, *, family: str = "tilt", key: float = 0.2, severe: float = 0.1, score: float = 0.8,
              robust: tuple = (True, True, False), plan_event: float = 0.2, other_event: float = 0.6, pair: dict | None = None) -> dict:
    """A record shaped like an mjgen one, with exactly the fields the adapters read."""
    names = {"tilt": ("moves_at_plan", "topples_at_plan", "risk_at_other_pitch", "riskier_orientation", "lengthwise")}[family]
    k, s, c, cmp_name, option = names
    events = infamily.EVENT[family]
    side = option if other_event > plan_event + 0.1 else ("as_it_stands" if other_event < plan_event - 0.1 else "about_same")
    compare = {key_: (0.8 if key_ == side else 0.1) for key_ in ("as_it_stands", option, "about_same")}
    labels = {k: {"type": "noul", "p_yes": key}, s: {"type": "noul", "p_yes": severe}, c: {"type": "score", "value": score},
              cmp_name: {"type": "choice", "kind": "state", "probabilities": compare}}
    questions = {k: {"type": "noul", "instructions": "k?"}, s: {"type": "noul", "instructions": "s?"},
                 c: {"type": "score", "instructions": "c?", "criteria": ["0 = no", "1 = yes"]},
                 cmp_name: {"type": "choice", "instructions": "which?", "criteria": {"as_it_stands": "plan", option: "other", "about_same": "same"}}}
    robustness = {name: {"robust": flag, "bands": {"base": system1.band(labels[name].get("p_yes", labels[name].get("value"))), "alt_a": "middle", "alt_b": "high"},
                         "min": 0.0, "max": 1.0} for name, flag in zip((k, s, c), robust)}
    robustness[cmp_name] = {"robust": True}
    outcomes = {"planned": {"base": {events[0]: plan_event, events[1]: 0.0, "stay": 1 - plan_event}},
                "opt_along": {"base": {events[0]: other_event, events[1]: 0.0, "stay": 1 - other_event}}}
    return {"id": rid, "generator": {"family": family, "group": group, "member": "pair" if rid.endswith("p") else "base"},
            "meta": {"scene": family, "question_kind": {k: "key", s: "severe", c: "score_cf", cmp_name: "compare"}, "robustness": robustness,
                     "compare_option": option, "plan": {"options": {"upright": "planned", option: "opt_along", "crosswise": "opt_across"}},
                     "outcomes": outcomes, "pair": pair},
            "request": {"state": f"state {rid}", "questions": questions}, "labels": labels}


def answer(rid: str, scalars: dict, choice: list | None = None) -> dict:
    """A prediction: yes/no and score questions answered in two listings with the given values; the comparison in the given listings."""
    questions = {}
    for name, values in scalars.items():
        qtype = "score" if name.startswith("risk_") else "noul"
        field = "value" if qtype == "score" else "p_yes"
        questions[name] = {"type": qtype, "listings": [{"options": None if qtype == "score" else ["yes", "no"][:: 1 - 2 * i], "answer": {field: v}}
                                                       for i, v in enumerate(values)]}
    if choice is not None:
        questions["riskier_orientation"] = {"type": "choice", "listings": [{"options": list(p), "answer": {"probabilities": p}} for p in choice]}
    return {"id": rid, "questions": questions}


def test_robust_and_ambiguous_keys_split_the_yes_no_and_score_questions():
    records = [synthetic("b1-tilt-000001", 1, robust=(True, False, True)), synthetic("b1-tilt-000002", 2, robust=(False, False, False))]
    assert infamily.robust_keys(records) == {("b1-tilt-000001", "moves_at_plan"), ("b1-tilt-000001", "risk_at_other_pitch")}
    assert len(infamily.ambiguous_keys(records)) == 4
    predictions = [answer(r["id"], {"moves_at_plan": [0.1, 0.2], "topples_at_plan": [0.5, 0.5], "risk_at_other_pitch": [0.9, 0.7]}) for r in records]
    items = system1.question_items("B1", records, predictions)
    kept = infamily.keep(items["band_balanced"], infamily.robust_keys(records))
    assert sorted(k[-2:] for k in kept) == sorted(infamily.robust_keys(records))
    assert {k[0] for k in kept} == {"low", "high"}  # the stratum (label band) is kept in the key


def test_clusters_are_generator_groups_so_a_pair_stays_together():
    pair = {"of": "b1-tilt-000003p", "kind": "numeric", "role": "base"}
    base = synthetic("b1-tilt-000003", 3, key=0.2, pair=pair)
    partner = synthetic("b1-tilt-000003p", 3, key=0.7, pair={"of": "b1-tilt-000003", "kind": "numeric", "role": "variant"})
    records = [base, partner]
    predictions = [answer(base["id"], {"moves_at_plan": [0.3, 0.3]}), answer(partner["id"], {"moves_at_plan": [0.6, 0.6]})]
    items = infamily.regroup(system1.question_items("B1", records, predictions)["band_accuracy"], records)
    assert {cluster for cluster, _ in items.values()} == {"tilt:3"}
    direction = infamily.direction_items("B1", records, predictions)
    assert list(direction.values()) == [("tilt:3", 1.0)]  # key moves 0.2 → 0.7 and the answer moves up; severe/score do not move
    wrong = [answer(base["id"], {"moves_at_plan": [0.6, 0.6]}), answer(partner["id"], {"moves_at_plan": [0.6, 0.6]})]
    assert list(infamily.direction_items("B1", records, wrong).values()) == [("tilt:3", 0.0)]  # an unchanged answer misses


def test_the_comparison_adapter_orders_the_two_actions_like_their_event_shares():
    up = synthetic("b1-tilt-000004", 4, plan_event=0.2, other_event=0.6)          # the compared action is riskier (+0.4)
    down = synthetic("b1-tilt-000005", 5, plan_event=0.7, other_event=0.3)        # the plan is riskier (−0.4)
    flat = synthetic("b1-tilt-000006", 6, plan_event=0.3, other_event=0.4)        # differs by < 0.15: no item
    records = [up, down, flat]
    listings = [{"as_it_stands": 0.2, "lengthwise": 0.5, "about_same": 0.3},   # orders "compared riskier": right for up
                {"as_it_stands": 0.4, "lengthwise": 0.4, "about_same": 0.2},   # equal: a miss
                {"as_it_stands": 0.6, "lengthwise": 0.1, "about_same": 0.3}]   # orders "plan riskier": wrong for up
    predictions = [answer(r["id"], {}, listings) for r in records]
    items = infamily.comparison_items("B1", records, predictions)
    assert set(items) == {("B1", up["id"], "riskier_orientation"), ("B1", down["id"], "riskier_orientation")}
    assert items[("B1", up["id"], "riskier_orientation")] == ("tilt:4", pytest.approx(1 / 3))
    assert items[("B1", down["id"], "riskier_orientation")] == ("tilt:5", pytest.approx(1 / 3))
    oracle = infamily.comparison_items("B1", records, system1.oracle_predictions(records))
    assert all(value == 1.0 for _, value in oracle.values())  # the label orders itself (its argmax side is the riskier action)


def test_the_band_confusion_and_answer_profile_weigh_each_question_once():
    records = [synthetic("b1-tilt-000007", 7, key=0.2, severe=0.5, score=0.9, robust=(True, True, True))]
    predictions = [answer(records[0]["id"], {"moves_at_plan": [0.1, 0.5], "topples_at_plan": [0.5, 0.5], "risk_at_other_pitch": [0.9, 0.2]})]
    keys = infamily.robust_keys(records)
    confusion = infamily.band_confusion(records, predictions, keys)
    assert confusion["questions"] == pytest.approx(3.0)
    assert confusion["counts"]["low"] == {"low": 0.5, "middle": 0.5, "high": 0.0}
    assert confusion["rows"]["high"] == {"low": 0.5, "middle": 0.0, "high": 0.5}
    profile = infamily.answer_profile(records, predictions, keys)
    assert profile["questions"] == pytest.approx(3.0) and profile["hedge_share"] == pytest.approx(0.5)
    assert profile["mean"] == pytest.approx((0.1 + 0.5 + 0.5 + 0.5 + 0.9 + 0.2) / 6)
    # bands the label takes under some prior: its base band plus "middle" and "high" (the synthetic alternatives); only the score's low
    # answer (0.2 against {high, middle}) is outside them
    assert profile["in_prior_bands"] == pytest.approx(2.5 / 3)
    assert profile["in_prior_range"] == pytest.approx(1.0)


def test_items_split_by_family_follow_their_record():
    records = [synthetic("b1-tilt-000008", 8), synthetic("b1-tilt-000009", 9)]
    records[1]["meta"]["scene"] = "grip"
    predictions = [answer(r["id"], {"moves_at_plan": [0.1, 0.2]}) for r in records]
    items = system1.question_items("B1", records, predictions)
    split = infamily.by_family(items["band_balanced"], records)
    assert set(split) == {"tilt", "grip"} and all(len(v) == 1 for v in split.values())


# -- the pre-declared rule ----------------------------------------------------------------------------------------------


def test_the_rule_separates_r1_r2_and_r3_at_its_bounds():
    verdict = _script("b1_verdict")
    assert verdict.rule_outcome(0.11, 0.30, 0.55) == "R1"
    assert verdict.rule_outcome(0.11, 0.30, 0.549) == "R3"   # learned more than the base but not to 0.55
    assert verdict.rule_outcome(0.10, 0.30, 0.90) == "R3"    # the lower bound must exceed +0.10
    assert verdict.rule_outcome(-0.05, 0.099, 0.40) == "R2"
    assert verdict.rule_outcome(-0.05, 0.10, 0.40) == "R3"   # the upper bound must be below +0.10
    with pytest.raises(ValueError):
        verdict.rule_outcome(None, 0.2, 0.5)
    assert verdict.control_outcome(0.80, 0.80)["pass"] and not verdict.control_outcome(0.799, 0.95)["pass"]
    failed = verdict.decide({"low": 0.2, "high": 0.4}, {"value": 0.7}, {"value": 0.5}, {"value": 0.9})
    assert failed["outcome"] == "R1" and not failed["interpreted"] and not failed["b2_runs"] and "no interpretation" in failed["meaning"]
    passed = verdict.decide({"low": -0.1, "high": 0.05}, {"value": 0.4}, {"value": 0.9}, {"value": 0.9})
    assert passed["outcome"] == "R2" and passed["interpreted"] and not passed["b2_runs"]


def _write_predictions(path, predictions, sha, **config):
    path.write_text("".join(json.dumps(p) + "\n" for p in predictions), encoding="utf-8")
    conditions = {"mode": "direct", "format": "chat", "prompt_version": "v0c-p3", "orders": "counterbalanced", "shuffle_seed": 1}
    summary = path.with_name(path.name.replace(".predictions.jsonl", ".json"))
    summary.write_text(json.dumps({"config": {"input_sha256": sha, "model": "test", "records": len(predictions), **conditions, **config}}), encoding="utf-8")


def test_the_verdict_script_scores_three_models_and_applies_the_rule(tmp_path):
    verdict = _script("b1_verdict")
    records = []
    for g in range(30):
        key = (0.1, 0.5, 0.9)[g % 3]
        pair_base = {"of": f"b1-tilt-{g:06d}p", "kind": "numeric", "role": "base"}
        records.append(synthetic(f"b1-tilt-{g:06d}", g, key=key, plan_event=key, other_event=1 - key, pair=pair_base))
        records.append(synthetic(f"b1-tilt-{g:06d}p", g, key=1 - key, plan_event=1 - key, other_event=key,
                                 pair={"of": f"b1-tilt-{g:06d}", "kind": "numeric", "role": "variant"}))
    path = tmp_path / "b1.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
    sha = file_sha256(path)
    path.with_suffix(".manifest.json").write_text(json.dumps({"sha256": sha, "master_seed": FRESH, "overlap_with_train": {"verdict": {"passed": True}}}), encoding="utf-8")
    constant = [answer(r["id"], {"moves_at_plan": [0.5, 0.5], "topples_at_plan": [0.5, 0.5], "risk_at_other_pitch": [0.5, 0.5]},
                       [{"as_it_stands": 0.3, "lengthwise": 0.3, "about_same": 0.4}]) for r in records]
    files = {}
    for tag, preds in (("base", constant), ("v01mix", constant), ("v03", system1.oracle_predictions(records))):
        files[tag] = tmp_path / f"B1-{tag}-2b.predictions.jsonl"
        _write_predictions(files[tag], preds, sha)
    out = tmp_path / "verdict.json"
    args = ["--input", str(path), "--out", str(out), "--n-boot", "50"] + [x for tag, f in files.items() for x in (f"--{tag}", str(f))]
    assert verdict.main(args) == 0
    result = json.loads(out.read_text(encoding="utf-8"))
    r = result["readings"]
    assert r["models"]["v03"]["primary"]["value"] == pytest.approx(1.0) and r["models"]["base"]["primary"]["value"] == pytest.approx(1 / 3)
    assert r["models"]["v03"]["direction"]["value"] == 1.0 and r["models"]["v03"]["comparison"]["value"] == 1.0
    assert r["models"]["base"]["direction"]["value"] == 0.0 and r["models"]["base"]["comparison"]["value"] == 0.0
    assert result["verdict"]["outcome"] == "R1" and result["verdict"]["positive_control"]["pass"] and result["verdict"]["b2_runs"]
    assert r["oracle"]["primary"] == 1.0 and all(c["band"] == 1.0 for c in r["coverage"].values())
    assert r["paired"]["v03-base"]["primary"]["value"] == pytest.approx(2 / 3)
    assert out.with_suffix(".md").read_text(encoding="utf-8").count("R1") >= 1
    # a prediction file scored on another set, or under other conditions, is refused
    _write_predictions(files["base"], constant, "f" * 64)
    with pytest.raises(SystemExit):
        verdict.main(args)
    _write_predictions(files["base"], constant, sha, prompt_version="v0c-p2")
    with pytest.raises(SystemExit):
        verdict.main(args)
