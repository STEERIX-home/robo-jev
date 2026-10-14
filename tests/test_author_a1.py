"""Author A1 (v0.6, warehouse and logistics, telemetry block) — generator checks.

The MuJoCo tests run a handful of rollouts (seconds). Run the suite inside a capped unit, as every MuJoCo job here:
``systemd-run --user --scope -p MemoryMax=2G -- choom -n 1000 -- python -m pytest tests/test_author_a1.py``.
"""

from __future__ import annotations

import math
import random

import numpy as np
import pytest

from robo_jev.authors.a1 import build, catalog, labels, sims
from robo_jev.authors.a1.scenes import DIR_KEYS, PLAN_KEY, PRIORS, SCENES, TIE_KEY, lhs, priors
from robo_jev.decisions.records import validate_record


# --------------------------------------------------------------------------
# Catalogs and priors
# --------------------------------------------------------------------------


def test_every_hidden_quantity_has_four_named_options():
    for table in (catalog.MAST_FLOORS, catalog.MAST_CONTENTS, catalog.MAST_SEATS, catalog.DOCK_LOADS, catalog.DOCK_SURFACES,
                  catalog.DOCK_GUIDANCE, catalog.RAMP_CONTENTS, catalog.RAMP_SURFACES):
        assert len(table) >= 4
    for entry in catalog.DOCK_LOADS.values():
        assert entry["cg_x"][0] < entry["cg_x"][1] and entry["cg_z"][0] < entry["cg_z"][1]


def test_alternative_priors_follow_the_common_rule():
    out = priors({"mu": ("friction", 0.4, 0.6), "m": ("range", 10.0, 20.0), "e": ("placement", 0.0, 0.02)})
    assert out["base"]["mu"] == ["u", 0.4, 0.6]
    assert out["alt_a"]["mu"][0] == "u" and out["alt_a"]["mu"][1:] == pytest.approx([0.28, 0.42])
    assert out["alt_b"]["mu"][1:] == pytest.approx([0.52, 0.78])
    assert out["alt_a"]["m"][1:] == pytest.approx([7.5, 22.5])
    assert out["alt_b"]["m"][1:] == pytest.approx([12.5, 17.5])
    assert out["alt_a"]["e"] == ["n", 0.0, pytest.approx(0.03)]
    assert out["alt_b"]["e"] == ["n", 0.0, pytest.approx(0.015)]


def test_latin_hypercube_has_one_point_per_stratum():
    u = lhs(32, 3, seed=7)
    assert u.shape == (32, 3)
    for j in range(3):
        assert sorted(np.floor(u[:, j] * 32).astype(int).tolist()) == list(range(32))


# --------------------------------------------------------------------------
# Label math
# --------------------------------------------------------------------------


def test_three_way_is_a_distribution_and_moves_with_the_change():
    k = build.K
    up = labels.three_way(np.array([1.0] * 16 + [0.0] * (k - 16)))
    flat = labels.three_way(np.zeros(k))
    down = labels.three_way(np.array([-1.0] * 16 + [0.0] * (k - 16)))
    for p in (up, flat, down):
        assert sum(p) == pytest.approx(1.0) and min(p) >= 0
    assert up[0] > 0.99 and down[1] > 0.99 and flat[2] > 0.99


def test_normalized_sums_to_one_within_the_loader_tolerance():
    rng = random.Random(0)
    for _ in range(200):
        probs = {k: rng.random() ** 3 for k in ("a", "b", "c")}
        out = labels.normalized(probs)
        assert abs(sum(out.values()) - 1.0) <= 1e-6 and all(0 <= v <= 1 for v in out.values())


def test_action_label_prefers_the_cheapest_option():
    probs, costs = labels.action_label({"x": {"ok": 1.0, "bad": 0.0}, "y": {"ok": 0.0, "bad": 1.0}}, {"ok": 0.0, "bad": 1.0}, {"x": 0.1, "y": 0.0})
    assert costs == {"x": 0.1, "y": 1.0} and probs["x"] > 0.99


def test_robustness_records_bands_and_reversals():
    r = labels.robust_scalar({"base": 0.1, "alt_a": 0.2, "alt_b": 0.3})
    assert r["robust"] and r["bands"]["base"] == "low"
    r = labels.robust_scalar({"base": 0.1, "alt_a": 0.5, "alt_b": 0.3})
    assert not r["robust"]
    c = labels.robust_choice({"base": {"higher": 0.8, "lower": 0.1, "unchanged": 0.1}, "alt_a": {"higher": 0.1, "lower": 0.8, "unchanged": 0.1},
                              "alt_b": {"higher": 0.8, "lower": 0.1, "unchanged": 0.1}}, opposite=("higher", "lower"))
    assert not c["robust"] and not c["no_reversal"]


# --------------------------------------------------------------------------
# Plans, pairs and records (no MuJoCo)
# --------------------------------------------------------------------------


@pytest.mark.parametrize("scene", sorted(SCENES))
def test_plan_is_deterministic_and_the_partner_changes_one_stated_fact(scene):
    sc = SCENES[scene]
    for index in range(12):
        a, b = build.plan_group(scene, index), build.plan_group(scene, index)
        assert a == b
        changed = [k for k in a["visible"] if a["visible"][k] != a["partner"][k]]
        assert changed == [a["pair"]["field"]]
        base_lines = sc.render(a["visible"], a["template"]).splitlines()
        partner_lines = sc.render(a["partner"], a["template"]).splitlines()
        assert len(base_lines) == len(partner_lines)
        assert sum(x != y for x, y in zip(base_lines, partner_lines)) == 1
        lo, hi = sc.bounds(a["visible"])
        assert lo - 1e-9 <= a["visible"][sc.decisive] <= hi + 1e-9 and lo - 1e-9 <= a["cf_value"] <= hi + 1e-9


def test_pair_kinds_are_about_two_thirds_one_factor():
    kinds = [build.plan_group("dock", i)["pair"]["kind"] for i in range(150)]
    share = kinds.count("one_factor") / len(kinds)
    assert 0.55 < share < 0.78


def _fake_cached(scene: str, index: int, seed: int = 0) -> dict:
    """A cached group with random outcomes in place of rollouts (record structure only)."""
    sc = SCENES[scene]
    plan = build.plan_group(scene, index)
    rng = random.Random(seed)
    members = {}
    for member, visible in (("base", plan["visible"]), ("pair", plan["partner"])):
        members[member] = {var: {p: [rng.choice(sc.outcomes) for _ in range(build.K)] for p in PRIORS}
                           for var in build.variants_of(sc, visible, plan["cf_value"])}
    return {"plan": plan, "members": members, "unstable": 0, "rollouts": 0, "mujoco": "test"}


@pytest.mark.parametrize("scene", sorted(SCENES))
def test_records_match_the_contract_and_pass_the_loader_checks(scene):
    sc = SCENES[scene]
    base, partner = build.records_of_group(_fake_cached(scene, 3), git="test")
    assert base["id"] == f"v06a1-{scene}-000003" and partner["id"] == f"v06a1-{scene}-000003p"
    for record in (base, partner):
        assert validate_record(record) == []
        meta = record["meta"]
        assert meta["set"] == "v06-a1" and meta["split"] == "train" and record["generator"]["family"] == scene
        assert set(meta["question_kind"].values()) == {"key", "severe", "score_cf", "direction", "compare", "action", "outcome"}
        assert record["labels"] == meta["labels_by_prior"]["base"]
        assert set(meta["labels_by_prior"]) == set(PRIORS) and set(meta["robustness"]) == set(record["request"]["questions"])
        assert meta["plan"]["options"][meta["compare_option"]] in meta["variants"]
        assert set(meta["outcomes"]) == set(meta["variants"])
        compare = record["request"]["questions"][sc.q["compare"]]["criteria"]
        assert set(compare) == {PLAN_KEY, sc.compare_option, TIE_KEY}
        assert set(record["request"]["questions"][sc.q["direction"]]["criteria"]) == set(DIR_KEYS)
        assert meta["action_options"] == list(sc.actions)
        assert set(record["labels"][sc.q["action"]]["expected_cost"]) == set(sc.actions)
    assert base["meta"]["pair"]["of"] == partner["id"] and partner["meta"]["pair"]["of"] == base["id"]
    assert base["meta"]["pair"]["kind"] == "numeric" and base["meta"]["pair"]["role"] == "base" and partner["meta"]["pair"]["role"] == "variant"


def test_flat_selection_keeps_pairs_and_flattens():
    rng = random.Random(1)
    pairs = [(rng.random() ** 2, rng.random() ** 2) for _ in range(200)]
    from robo_jev.authors.a1.assemble import BINS, select_flat

    kept = select_flat(pairs, 150)
    assert len(kept) == 150 == len(set(kept))
    before = np.histogram([x for p in pairs for x in p], bins=BINS, range=(0, 1))[0]
    after = np.histogram([x for i in kept for x in pairs[i]], bins=BINS, range=(0, 1))[0]
    assert after.std() / after.mean() < before.std() / before.mean()


# --------------------------------------------------------------------------
# MuJoCo (a few rollouts)
# --------------------------------------------------------------------------


def test_dock_tips_once_the_centre_of_mass_passes_the_edge():
    spec = {"step_h": 0.35, "pallet_dims": [1.2, 0.8, 1.4], "mass": 500.0}
    hidden = {"mu": 0.5, "cg_x": 0.08, "cg_z": 0.45, "place": 0.0}
    edge = 0.6 - 0.08
    assert sims.rollout("dock", spec, hidden, {"overhang": edge - 0.03})["outcome"] == "level"
    assert sims.rollout("dock", spec, hidden, {"overhang": edge + 0.03})["outcome"] != "level"


def test_ramp_holds_below_the_friction_limit_and_runs_away_above():
    spec = {"tug_mass": 330.0, "cart_tare": 70.0}
    hidden = {"mu": 0.4, "payload": 300.0}
    limit = 100 * 0.4 * 330.0 / (330.0 + 370.0)
    assert sims.rollout("ramp", spec, hidden, {"grade": 0.6 * limit, "speed": 0.05})["outcome"] == "held"
    assert sims.rollout("ramp", spec, hidden, {"grade": 1.25 * limit, "speed": 0.05})["outcome"] == "runaway"


def test_mast_skids_on_a_slippery_floor_and_lifts_on_a_grippy_one():
    unit = dict(catalog.MAST_UNITS["Kestrel MP-4"])
    spec = {**unit, "lift_h": 2.3, "tote_dims": [0.6, 0.4, 0.32]}
    slippery = sims.rollout("mast", spec, {"mu": 0.15, "tote_mass": 35.0, "tote_x": 0.0}, {"speed": 1.8, "decel": 5.0})
    grippy = sims.rollout("mast", spec, {"mu": 1.0, "tote_mass": 35.0, "tote_x": 0.0}, {"speed": 1.8, "decel": 5.0})
    gentle = sims.rollout("mast", spec, {"mu": 1.0, "tote_mass": 35.0, "tote_x": 0.0}, {"speed": 1.8, "decel": 0.8})
    assert slippery["outcome"] == "skid" and grippy["outcome"] == "lift" and gentle["outcome"] == "clean"


def test_rollouts_are_deterministic():
    sc = SCENES["mast"]
    plan = build.plan_group("mast", 5)
    v = plan["visible"]
    h = sc.hidden(v, "base", lhs(build.K, len(sc.hidden_names), plan["draws_seed"]))[0]
    a = sims.rollout("mast", sc.spec(v), h, sc.state(v))
    b = sims.rollout("mast", sc.spec(v), h, sc.state(v))
    assert a == b and not a["unstable"]
    assert math.isfinite(a["max_tilt"])
