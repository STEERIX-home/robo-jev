"""Author A3 (v0.6): catalogs and priors, plans and pairs, wording, labels, MuJoCo physics, record assembly and scorer compatibility."""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from robo_jev.authors import a3
from robo_jev.authors.a3 import build, catalog as cat, labels as lab, sim, text
from robo_jev.authors.a3.scenes import SCENES
from robo_jev.decisions.records import NAME_RE, validate_record

SCENE_NAMES = ("indexdial", "mayostand", "holddown")


# --------------------------------------------------------------------------
# catalogs and priors
# --------------------------------------------------------------------------


def test_every_hidden_quantity_has_at_least_four_named_options():
    assert len(cat.DIAL_BASE) >= 4 and len(cat.DIAL_PLATE) >= 4 and len(cat.DIAL_BUILD) >= 4 and len(cat.DIAL_PLACEMENT) >= 4
    assert len(cat.STAND_SET) >= 4 and len(cat.STAND_PACKING) >= 4 and len(cat.STAND_MODEL) >= 4 and len(cat.STAND_METHOD) >= 4
    assert len(cat.CLAMP_NEST) >= 4 and len(cat.CLAMP_PAD) >= 4 and len(cat.CLAMP_CONDITION) >= 4
    for name, sc in SCENES.items():
        assert set(sc.hidden_names) <= set(cat.QUANTITY_KIND), name


def test_prior_rule():
    base = {"mu": ["u", 0.4, 0.6], "kh": ["u", 0.3, 0.5], "dr": ["n", 0.0, 0.01], "force_factor": ["u", 0.8, 1.0]}
    pr = cat.prior_ranges(base)
    assert pr["alt_a"]["mu"] == pytest.approx(["u", 0.28, 0.42]) and pr["alt_b"]["mu"] == pytest.approx(["u", 0.52, 0.78])
    assert pr["alt_a"]["kh"] == pytest.approx(["u", 0.25, 0.55]) and pr["alt_b"]["kh"] == pytest.approx(["u", 0.35, 0.45])
    assert pr["alt_a"]["dr"] == pytest.approx(["n", 0.0, 0.015]) and pr["alt_b"]["dr"] == pytest.approx(["n", 0.0, 0.0075])
    assert pr["alt_a"]["force_factor"] == pytest.approx(["u", 0.75, 1.05])
    assert pr["base"] == base


def test_draw_points_are_shared_and_clipped():
    u = cat.uniform_points(7, 32, ("a", "b"))
    assert u["a"].shape == (32,) and np.array_equal(u["a"], cat.uniform_points(7, 32, ("a", "b"))["a"])
    z = cat.draw_values(["n", 0.0, 1.0], np.array([0.0, 0.5, 1.0]))
    assert z[0] == pytest.approx(-2.5, abs=1e-3) and z[1] == pytest.approx(0.0) and z[2] == pytest.approx(2.5, abs=1e-3)
    assert cat.draw_values(["u", 1.0, 3.0], np.array([0.25]))[0] == pytest.approx(1.5)


# --------------------------------------------------------------------------
# plans and pairs
# --------------------------------------------------------------------------


def _diff_fields(sc, a, b):
    fields = set(a) | set(b)
    out = {f for f in fields if a.get(f) != b.get(f)}
    if "dims_mm" in out and a["dims_mm"][:2] == b["dims_mm"][:2]:
        out = (out - {"dims_mm"}) | {"part_height_mm"}
    return out


@pytest.mark.parametrize("scene", SCENE_NAMES)
def test_plans_are_deterministic_and_pairs_change_one_fact(scene):
    sc = SCENES[scene]
    for g in range(6):
        plan = build.plan_group(scene, g)
        assert json.dumps(plan, sort_keys=True) == json.dumps(build.plan_group(scene, g), sort_keys=True)
        changed = _diff_fields(sc, plan["base_state"], plan["pair_state"])
        assert changed == {plan["field"]}, (g, changed, plan["field"])
        if plan["design"] == "far_number":
            assert plan["field"] == sc.decisive and plan["factor"] is None
        cf = plan["cf"]["value"]
        for state in (plan["base_state"], plan["pair_state"]):
            assert abs(cf - state[sc.decisive]) >= sc.step - 1e-9
        alternatives = {a.key: a for a in sc.alternatives()}
        assert len(plan["options"]) == 2 and plan["compare_option"] in plan["options"]
        for key in plan["options"]:
            assert alternatives[key].valid(plan["base_state"]) and alternatives[key].valid(plan["pair_state"])


def test_design_shares_close_to_two_thirds():
    designs = [build.plan_group("holddown", g)["design"] for g in range(90)]
    assert 0.5 < designs.count("one_factor") / len(designs) < 0.8


# --------------------------------------------------------------------------
# wording
# --------------------------------------------------------------------------


@pytest.mark.parametrize("scene", SCENE_NAMES)
def test_pair_texts_differ_in_one_line_and_share_the_questions(scene):
    for g in range(4):
        plan = build.plan_group(scene, g)
        a = text.state_text(scene, plan["base_state"], plan["style"]).splitlines()
        b = text.state_text(scene, plan["pair_state"], plan["style"]).splitlines()
        assert len(a) == len(b) == 6 and a[0] == b[0]
        assert sum(x != y for x, y in zip(a, b)) == 1
        assert all(line[0].isdigit() for line in a[1:])  # numbered work-order fields


def test_question_names_and_option_keys_are_valid():
    for scene in SCENE_NAMES:
        names = text.QUESTION_NAMES[scene]
        assert set(names) == set(text.KINDS)
        for name in names.values():
            assert NAME_RE.match(name)
        sc = SCENES[scene]
        for alt in sc.alternatives():
            assert NAME_RE.match(alt.key)
        for o in sc.outcomes:
            assert NAME_RE.match(o)


# --------------------------------------------------------------------------
# labels
# --------------------------------------------------------------------------


def test_three_way_reading():
    for delta in (-0.8, -0.2, -0.1, 0.0, 0.05, 0.1, 0.3, 1.0):
        up, down, same = lab.three_way(delta)
        assert up + down + same == pytest.approx(1.0) and min(up, down, same) >= 0
    assert lab.three_way(0.4)[0] > 0.99 and lab.three_way(-0.4)[1] > 0.99 and lab.three_way(0.0)[2] > 0.8


def _freq(outcomes, shares):
    return {o: s for o, s in zip(outcomes, shares)}


def test_record_labels_semantics():
    sc = SCENES["holddown"]
    outcomes = sc.outcomes
    freq = {
        "planned": {p: _freq(outcomes, (0.5, 0.25, 0.25)) for p in cat.PRIORS},
        "cf": {p: _freq(outcomes, (1.0, 0.0, 0.0)) for p in cat.PRIORS},
        "opt_raise_clamp": {p: _freq(outcomes, (1.0, 0.0, 0.0)) for p in cat.PRIORS},
        "opt_soft_clamp": {p: _freq(outcomes, (0.0, 0.0, 1.0)) for p in cat.PRIORS},
    }
    names = text.QUESTION_NAMES["holddown"]
    by_prior = lab.record_labels(freq=freq, outcomes=outcomes, key_events=sc.key_events, severe=sc.severe, costs=sc.costs, names=names,
                                 options={"as_written": "planned", "raise_clamp": "opt_raise_clamp", "soft_clamp": "opt_soft_clamp"},
                                 compare_option="soft_clamp")
    base = by_prior["base"]
    assert base[names["key"]]["p_yes"] == 0.5 and base[names["severe"]]["p_yes"] == 0.25 and base[names["score_cf"]]["value"] == 0.0
    assert base[names["direction"]]["probabilities"]["lower"] > 0.99
    assert base[names["compare"]]["probabilities"]["soft_clamp"] > 0.99  # the compared change is the riskier one
    action = base[names["action"]]
    assert action["expected_cost"]["raise_clamp"] == 0.0 and action["expected_cost"]["soft_clamp"] == pytest.approx(2.2)
    assert max(action["probabilities"], key=action["probabilities"].get) == "raise_clamp"
    for label in base.values():
        if label["type"] == "choice":
            assert sum(label["probabilities"].values()) == pytest.approx(1.0, abs=1e-6)
    robust = lab.robustness(by_prior, {n: list(base[n]["probabilities"]) for n in base if base[n]["type"] == "choice"}, {})
    assert robust[names["key"]]["robust"] is True and robust[names["key"]]["bands"]["base"] == "middle"


# --------------------------------------------------------------------------
# MuJoCo physics (small)
# --------------------------------------------------------------------------


def test_dial_holds_slides_and_is_deterministic():
    low = sim.DialModel.build(0.16, 0.15, 0.05, 2.0)
    gentle = sim.index_profile(90, 2.5)
    assert sim.run_dial(low, gentle, radius=0.3, mu=0.5)["outcome"] == "stays_put"
    fast = sim.index_profile(90, 0.6)  # peak ≈ 0.86 g at 0.3 m
    first = sim.run_dial(low, fast, radius=0.3, mu=0.3)
    assert first["outcome"] == "slides_off_mark"
    again = sim.run_dial(sim.DialModel.build(0.16, 0.15, 0.05, 2.0), fast, radius=0.3, mu=0.3)
    assert json.dumps(first, sort_keys=True) == json.dumps(again, sort_keys=True)


def test_dial_tall_part_tips_on_high_friction():
    tall = sim.DialModel.build(0.09, 0.08, 0.30, 2.0, (0.0, 0.0, 0.03))
    coef = sim.dial_peak_accel(90, 1.0, 0.35)
    threshold = sim.GRAVITY * 0.04 / (0.6 * 0.30)
    calm = sim.index_profile(90, math.sqrt(coef / (0.6 * threshold)))
    hard = sim.index_profile(90, math.sqrt(coef / (2.5 * threshold)))
    assert sim.run_dial(tall, calm, radius=0.35, mu=1.4)["outcome"] == "stays_put"
    assert sim.run_dial(tall, hard, radius=0.35, mu=1.4)["outcome"] in ("rocks_back", "falls_over")


def test_stand_tips_exactly_when_the_centre_of_mass_passes_the_front_feet():
    reach, tray, top, box = 0.20, 0.58, 1.05, (0.30, 0.26, 0.10)
    ms, com, _ = sim.stand_inertial(reach, tray, top, 3.0)
    x_star = reach + ms * (reach - com[0]) / 6.0
    model = sim.StandModel.build(reach, tray, top, box, base_mass=3.0, load_mass=6.0, load_com_x=0.0)
    assert sim.run_stand(model, setdown_x=x_star - 0.015, release_h=0.0)["outcome"] == "stays_level"
    model = sim.StandModel.build(reach, tray, top, box, base_mass=3.0, load_mass=6.0, load_com_x=0.0)
    assert sim.run_stand(model, setdown_x=x_star + 0.015, release_h=0.0)["outcome"] == "tips_over"


def test_clamp_holds_below_capacity_and_breaks_free_above():
    model = sim.ClampModel.build((0.12, 0.08, 0.04), 1.5)
    capacity = sim.clamp_capacity(mu_nest=0.4, mu_pad=0.5, clamp_force=150.0, mass=1.5)
    assert sim.run_clamp(model, press_force=0.85 * capacity, press_time=0.3, clamp_force=150.0, mu_nest=0.4, mu_pad=0.5)["outcome"] == "stays_put"
    assert sim.run_clamp(model, press_force=1.5 * capacity, press_time=0.3, clamp_force=150.0, mu_nest=0.4, mu_pad=0.5)["outcome"] == "breaks_free"


def test_dial_slip_model_matches_simple_limits():
    r = np.array([0.3])
    assert sim.dial_slip(90, 3.0, r, np.array([0.5]))[0] < sim.DIAL_SHIFT_M
    assert sim.dial_slip(90, 0.5, r, np.array([0.2]))[0] > 0.05


# --------------------------------------------------------------------------
# assembly and scorer compatibility (fake outcome codes — no MuJoCo)
# --------------------------------------------------------------------------


def _fake_labelled(plan, seed=0):
    sc = SCENES[plan["scene"]]
    rng = np.random.default_rng(seed)
    states, outcomes = {}, {}
    for member in ("base", "pair"):
        for full in build.member_states(sc, plan, member).values():
            sid = build.state_id(full)
            states[sid] = full
            outcomes[sid] = {p: [int(x) for x in rng.integers(0, len(sc.outcomes), size=build.K)] for p in cat.PRIORS}
    return {"scene": plan["scene"], "group": plan["group"], "states": states, "outcomes": outcomes}


@pytest.mark.parametrize("scene", SCENE_NAMES)
def test_assembled_records_match_the_contract(scene):
    plan = build.plan_group(scene, 3)
    records = build.assemble_group(plan, _fake_labelled(plan), git="test")
    assert [r["id"] for r in records] == [f"v06a3-{scene}-000003", f"v06a3-{scene}-000003p"]
    for r in records:
        assert validate_record(r) == []
        meta = r["meta"]
        assert meta["set"] == "v06-a3" and meta["split"] == "train" and r["labels"] == meta["labels_by_prior"]["base"]
        assert set(meta["question_kind"].values()) == {"key", "severe", "score_cf", "direction", "compare", "action", "outcome"}
        assert meta["plan"]["options"][meta["compare_option"]] in meta["outcomes"]
        assert set(meta["robustness"]) == set(r["request"]["questions"])
        assert meta["pair"]["of"] in {x["id"] for x in records} and meta["pair"]["kind"] == "numeric"
        assert r["generator"]["k"] == 32 and r["generator"]["priors"] == list(cat.PRIORS)
    assert records[0]["request"]["questions"] == records[1]["request"]["questions"]


def test_infamily_readers_run_on_a3_records():
    from robo_jev.bench import infamily, system1

    a3.register_infamily()
    records = []
    for g in range(4):
        plan = build.plan_group("holddown", g)
        records += build.assemble_group(plan, _fake_labelled(plan, seed=g), git="test")
    oracle = system1.oracle_predictions(records)
    infamily.robust_keys(records)
    comparison = infamily.comparison_items("A3", records, oracle)
    direction = infamily.direction_items("A3", records, oracle)
    assert all(v == 1.0 for _, v in comparison.values()) and all(v == 1.0 for _, v in direction.values())
    assert set(a3.EVENT) == set(SCENE_NAMES) and all(set(a3.EVENT[s]) == set(SCENES[s].key_events) for s in SCENE_NAMES)


def test_request_length_budget():
    tok = pytest.importorskip("robo_jev.model.tokenizer")
    found = tok.available_tokenizer()
    if found is None:
        pytest.skip("tokenizer not fetched")
    tokenizer = tok.load_tokenizer(found[1])
    lengths = []
    for scene in SCENE_NAMES:
        for g in range(8):
            plan = build.plan_group(scene, g)
            record = build.assemble_group(plan, _fake_labelled(plan), git="test")[0]
            parts = [record["request"]["state"]]
            for q in record["request"]["questions"].values():
                parts.append(q["instructions"])
                crit = q.get("criteria")
                parts += [f"{k}: {v}" if v else k for k, v in crit.items()] if isinstance(crit, dict) else list(crit or [])
            lengths.append(len(tokenizer.encode("\n".join(parts), add_special_tokens=False).ids))
    assert max(lengths) <= 500 and float(np.median(lengths)) <= 360
