"""Task T9 — the second outside test: physics, priors, candidate groups, records, text, references and System-1 scoring.

MuJoCo checks use few draws so the file runs in seconds; the full physics sanity report is ``scripts/t9.py sanity``."""

import ast
import json
import math
from collections import Counter
from pathlib import Path

import pytest
from helpers import REPO

from robo_jev.bench import system1 as S1
from robo_jev.decisions.records import validate_record
from robo_jev.t9 import build as B
from robo_jev.t9 import overlap as O
from robo_jev.t9 import physics as ph
from robo_jev.t9 import priors as P
from robo_jev.t9 import reference as R
from robo_jev.t9 import sanity
from robo_jev.t9 import text as T
from robo_jev.t9.families import FAMILIES

MASTER = 4242


def test_t9_code_imports_nothing_from_the_engine_mjgen_or_the_t8b_generator():
    """T9 is written by a fresh author: no simgen (engine phrase banks), no mjgen (v0.3b), no T8b moments (catalog, wording); only the
    low-level MuJoCo constants of robo_jev.outside.physics and the T8 Brier helper for scoring."""
    files = sorted((REPO / "src" / "robo_jev" / "t9").glob("*.py")) + [REPO / "scripts" / "t9.py"]
    banned = ("robo_jev.simgen", "robo_jev.mjgen", "robo_jev.outside.moments", "robo_jev.outside.reference", "robo_jev.e1")
    for path in files:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            names = [n.name for n in node.names] if isinstance(node, ast.Import) else ([node.module or ""] if isinstance(node, ast.ImportFrom) else [])
            for name in names:
                assert not name.startswith(banned), f"{path.name} imports {name}"


def test_alternative_priors_shift_every_word_range_by_a_third_of_its_width_and_clip_to_bounds():
    words = {"mu": (0.30, 0.60), "vacuum": (25000.0, 40000.0), "mu_roll": (0.0, 0.003)}
    low, high = P.prior_ranges(words, "low_reader"), P.prior_ranges(words, "high_reader")
    assert low["mu"] == pytest.approx((0.20, 0.50)) and high["mu"] == pytest.approx((0.40, 0.70))
    assert high["vacuum"] == pytest.approx((30000.0, 45000.0))
    assert low["mu_roll"] == pytest.approx((0.0, 0.002))  # clipped at the physical bound 0
    assert P.prior_ranges(words, "base") == {k: tuple(map(float, v)) for k, v in words.items()}
    units = P.unit_draws(["mu", "mu_roll", "vacuum"], 16, 7)
    base = P.draws_for(words, {}, "base", units, ["mu", "mu_roll", "vacuum"])
    shifted = P.draws_for(words, {}, "high_reader", units, ["mu", "mu_roll", "vacuum"])
    assert [round(b["mu"] + 0.1, 9) for b in base] == [round(s["mu"], 9) for s in shifted]  # same quantile worlds
    fixed = P.draws_for(words, {"err": (-0.01, 0.01)}, "low_reader", P.unit_draws(["err"], 8, 1), ["err"])
    assert all(-0.01 <= d["err"] <= 0.01 for d in fixed)  # stated tolerances are not re-read


def test_every_scene_is_deterministic_and_reports_an_outcome_of_its_family():
    import random

    for name, fam in FAMILIES.items():
        vis = fam.sample(random.Random(3))
        draw = P.draws_for(fam.words(vis), fam.fixed(vis), "base", P.unit_draws(fam.hidden_names(), 4, 11), fam.hidden_names())[0]
        scene = ph.make_scene(name, fam.scene(vis))
        first = scene.run(fam.action(vis), fam.physics_draw(vis, draw))
        again = ph.make_scene(name, fam.scene(vis)).run(fam.action(vis), fam.physics_draw(vis, draw))
        assert first == again and first["outcome"] in ph.OUTCOMES[name] and not first["unstable"]


def test_a_rolling_solid_cylinder_accelerates_at_two_thirds_g_sin_theta_and_a_thin_tube_at_one_half():
    for k, shape in ((0.5, "capsule"), (1.0, "capsule"), (0.4, "sphere")):
        row = sanity._roll_accel(k, shape)
        assert row["ratio"] == pytest.approx(1.0, abs=0.03), row


def test_the_board_tips_where_the_lever_rule_says():
    for row in sanity._overhang_tip():
        assert row["pass"], row


def test_the_conveyor_calculator_and_mujoco_agree_on_most_draws_of_a_tippable_item():
    fam = FAMILIES["conveyor"]
    vis = {"context": "packing", "item": "tins", "mass": 1.7, "belt": "roughtop", "event": "stop", "speed": 1.0, "rate": 7.0, "ramp": 0.05, "turned": "narrow"}
    draws = P.draws_for(fam.words(vis), fam.fixed(vis), "base", P.unit_draws(fam.hidden_names(), 8, 3), fam.hidden_names())
    calc = fam.calc(vis, draws)
    scene = ph.make_scene("conveyor", fam.scene(vis))
    mj = [scene.run(fam.action(vis), d)["outcome"] for d in draws]
    assert sum((c == "topple") == (m == "topple") for c, m in zip(calc, mj)) >= 6


@pytest.fixture(scope="module")
def labelled_group():
    group = B.refine(B.candidate("overhang", 3, MASTER, k=8))
    return group, B.label_group(group)


def test_a_candidate_group_is_reproducible_from_its_seed():
    assert B.candidate("swing", 2, MASTER, k=8) == B.candidate("swing", 2, MASTER, k=8)
    assert B.candidate("swing", 2, MASTER, k=8) != B.candidate("swing", 3, MASTER, k=8)


def test_records_validate_and_carry_labels_from_the_rollout_codes(labelled_group):
    group, labelled = labelled_group
    records = B.records_of(group, labelled, mujoco_version="test")
    assert len(records) == 2 and [r["id"][-1] for r in records] == ["a", "b"]
    fam = FAMILIES["overhang"]
    names = T.NAMES["overhang"]
    for record, moment in zip(records, ("a", "b")):
        assert validate_record(record) == []
        meta = record["meta"]
        assert meta["set"] == "T9" and meta["events"] == [names["key"], names["cf"], names["alt"]] and meta["law_group"] == "seen_law_new_scene"
        for prior in P.PRIORS:
            codes = labelled["labels"][moment]["plan"][prior]["codes"]
            share = sum(ph.OUTCOMES["overhang"][int(c)] in fam.key for c in codes) / len(codes)
            assert meta["labels_by_prior"][prior][names["key"]]["p_yes"] == pytest.approx(share)
        assert record["labels"] == meta["labels_by_prior"]["base"]
        delta = record["labels"][names["dir"]]["delta"]
        top = max(record["labels"][names["dir"]]["probabilities"].items(), key=lambda kv: kv[1])[0]
        assert top == ("likelier" if delta >= 0.1 else "less_likely" if delta <= -0.1 else "no_real_change")
        costs = record["labels"][names["act"]]["expected_cost"]
        plan, gentler, safe = T.ACT_OPTIONS["overhang"]
        assert costs[plan] == pytest.approx(record["labels"][names["key"]]["p_yes"] * meta["costs"]["fail"])
        assert costs[safe] == meta["costs"]["secure"]
        rob = meta["robustness"][names["key"]]
        assert rob["robust"] == (len(set(rob["bands"].values())) == 1)
    a, b = records
    assert a["meta"]["pair"]["of"] == b["id"] and b["meta"]["pair"]["of"] == a["id"] and a["meta"]["pair"]["kind"] == "numeric"
    assert a["meta"]["pair"]["role"] == "base" and b["meta"]["pair"]["role"] == "variant"


def test_the_text_states_the_numbers_but_never_a_hidden_quantity(labelled_group):
    group, labelled = labelled_group
    for record in B.records_of(group, labelled, mujoco_version="test"):
        text = (record["request"]["state"] + " " + json.dumps(record["request"]["questions"])).lower()
        for word in ("friction", "coefficient", "inertia", "centre of mass", "prior", "pascal", "kpa", "mujoco", "simulat"):
            assert word not in text, word


def test_the_calculator_and_constant_references_answer_every_question_in_the_prediction_format(labelled_group):
    from robo_jev.t9.score import _validity

    group, labelled = labelled_group
    records = B.records_of(group, labelled, mujoco_version="test")
    for predictions in ([R.calculator_prediction(r) for r in records], [R.calculator_prediction(r, means=True) for r in records],
                        *[[R.constant_prediction(r, kind) for r in records] for kind in R.CONSTANTS]):
        assert _validity(records, predictions)["valid"]


def _t9_record(rid: str, key: float, cf: float, alt: float, *, by_prior: dict | None = None, robust: bool = True, pair: dict | None = None) -> dict:
    labels = {"k": {"type": "noul", "p_yes": key}, "c": {"type": "score", "value": cf}, "a": {"type": "noul", "p_yes": alt},
              "o": {"type": "choice", "kind": "state", "probabilities": {"x": 0.7, "y": 0.3}},
              "d": {"type": "choice", "kind": "state", "probabilities": {"up": 1.0, "down": 0.0, "same": 0.0}}}
    questions = {"k": {"type": "noul", "instructions": "k?"}, "c": {"type": "score", "instructions": "c?", "criteria": ["0", "1"]},
                 "a": {"type": "noul", "instructions": "a?"}, "o": {"type": "choice", "instructions": "o?", "criteria": {"x": "x", "y": "y"}},
                 "d": {"type": "choice", "instructions": "d?", "criteria": {"up": "u", "down": "d", "same": "s"}}}
    priors = by_prior or {"base": {"k": key, "c": cf, "a": alt}, "alt1": {"k": key, "c": cf, "a": alt}}
    meta = {"set": "T9", "scene": "fam", "events": ["k", "c", "a"], "outcome_question": "o", "direction_question": "d",
            "labels_by_prior": {p: {"k": {"type": "noul", "p_yes": v["k"]}, "c": {"type": "score", "value": v["c"]}, "a": {"type": "noul", "p_yes": v["a"]}}
                                for p, v in priors.items()},
            "robustness": {q: {"robust": robust} for q in ("k", "c", "a")}}
    if pair:
        meta["pair"] = pair
    return {"schema": "simgen-request/0", "id": rid, "request": {"state": "s", "questions": questions}, "labels": labels, "meta": meta}


def test_system1_scores_declared_t9_events_with_t8b_definitions_and_the_robust_reading():
    records = [_t9_record("t9-fam-0000a", 0.2, 0.8, 0.05, by_prior={"base": {"k": 0.2, "c": 0.8, "a": 0.05}, "alt1": {"k": 0.3, "c": 0.35, "a": 0.0}})]
    oracle = S1.set_items("T9", records, S1.oracle_predictions(records))
    assert {k for k, _, _ in [(p[1][1], 0, 0) for p in S1.t8b_direction_pairs(records)]} == {"k"}
    assert len(S1.t8b_direction_pairs(records)) == 2 and len(S1.t8b_event_pairs(records)) == 3
    for metric in ("direction", "comparison", "outcome", "direction_choice", "band_balanced_robust", "severe_error_robust"):
        assert metric in oracle, metric
        value = S1.summarize(oracle[metric], stratified=metric in S1.STRATIFIED)["value"]
        assert value == (0.0 if metric.startswith("severe") else 1.0), metric
    # k → c moves +0.6 under base but only +0.05 under alt1: not robust; k → a moves −0.15 / −0.3: robust
    robust = S1.robust_pairs(records, S1.t8b_direction_pairs(records))
    assert [(p[1][1], p[2][1]) for p in robust] == [("k", "a")]
    assert len(oracle["direction"]) == 2 and len(oracle["direction_robust"]) == 1
    flat = [{"id": "t9-fam-0000a", "questions": {q: {"type": t, "answer": a, "listings": [{"options": None, "answer": a}]} for q, t, a in (
        ("k", "noul", {"p_yes": 0.5}), ("c", "score", {"value": 0.5}), ("a", "noul", {"p_yes": 0.5}),
        ("o", "choice", {"probabilities": {"x": 0.2, "y": 0.8}}), ("d", "choice", {"probabilities": {"up": 0.1, "down": 0.8, "same": 0.1}}))}}]
    items = S1.set_items("T9", records, flat)
    assert S1.summarize(items["direction"])["value"] == 0.0 and S1.summarize(items["outcome"])["value"] == 0.0
    assert S1.summarize(items["direction_choice"])["value"] == 0.0


def test_unrobust_questions_are_left_out_of_the_robust_band_reading():
    records = [_t9_record("t9-fam-0001a", 0.9, 0.1, 0.1, robust=False)]
    items = S1.set_items("T9", records, S1.oracle_predictions(records))
    assert "band_balanced_robust" not in items and "band_balanced" in items


def test_selection_keeps_equal_numbers_per_tenth_and_fills_short_tenths_from_the_nearest():
    groups = [{"family": "roll", "index": i, "key_a": (i % 10) / 10 + 0.05, "unstable": 0} for i in range(40)]
    groups += [{"family": "roll", "index": 100 + i, "key_a": 0.95, "unstable": 0} for i in range(10)]
    groups.append({"family": "roll", "index": 999, "key_a": 0.55, "unstable": 3})
    chosen, report = B.select(groups, 20)
    assert len(chosen) == 20 and report["roll"]["kept_by_tenth"] == [2] * 10 and all(g["index"] != 999 for g in chosen)
    sparse = [g for g in groups if g["key_a"] < 0.5 and g["index"] < 100]
    chosen, report = B.select(sparse + [{"family": "roll", "index": 200 + i, "key_a": 0.95, "unstable": 0} for i in range(30)], 30)
    assert len(chosen) == 30 and sum(report["roll"]["kept_by_tenth"]) == 30


def test_the_overlap_units_catch_a_shared_sentence_and_a_shared_five_gram(tmp_path):
    ours = [{"id": "t9-x-0000a", "request": {"state": "The lid of the crate stands open. I hold it with two fingers.",
                                              "questions": {"q": {"type": "noul", "instructions": "Will the crate tip over at 2.5 m/s?"}}}}]
    t9 = O.T9Units(ours)
    theirs = tmp_path / "theirs.jsonl"
    theirs.write_text(json.dumps({"id": "z", "request": {"state": "Something else. The lid of the crate stands open.", "questions": {}}}) + "\n"
                      + json.dumps({"id": "w", "request": {"state": "Will the crate tip over at 9 m/s, I wonder.", "questions": {}}}) + "\n", encoding="utf-8")
    report = O.compare_file(t9, "T8b", theirs, grams=True)
    assert report["shared"]["sentences"] == 1 and report["ngrams"]["5"]["shared"] >= 1
    assert not O.verdict([report])["passed"]
    clean = tmp_path / "clean.jsonl"
    clean.write_text(json.dumps({"id": "c", "request": {"state": "A wholly different sentence lives here.", "questions": {}}}) + "\n", encoding="utf-8")
    assert O.verdict([O.compare_file(t9, "mujoco-train", clean, grams=True)])["passed"]


def test_the_generator_writes_every_family_and_the_built_set_if_present_follows_the_conventions():
    path = REPO / "artifacts" / "t9" / "T9.jsonl"
    if not path.is_file():
        pytest.skip("T9 not built here")
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    families = Counter(r["meta"]["family"] for r in records)
    assert set(families) == set(FAMILIES) and all(validate_record(r) == [] for r in records)
    assert len({r["id"] for r in records}) == len(records)
    by_id = {r["id"]: r for r in records}
    assert all(by_id[r["meta"]["pair"]["of"]]["meta"]["pair"]["of"] == r["id"] for r in records)
