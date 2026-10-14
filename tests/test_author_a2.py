"""Author A2 (v0.6) — catalogs and priors, planning, wording, record contract, scorers, prompt lengths; MuJoCo physics sanity
(analytic thresholds, label-level monotonicity, determinism) only inside a capped systemd unit (``INVOCATION_ID`` set), as the
brief requires for any MuJoCo run."""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from robo_jev.authors.a2 import EVENT, PLAN_KEY, registered_with_infamily
from robo_jev.authors.a2 import catalog as C
from robo_jev.authors.a2 import text as T
from robo_jev.authors.a2.generate import K, assemble_group, member_variants, plan_group, soft_three
from robo_jev.authors.a2.scenes import SCENES, hidden_values

REPO = Path(__file__).resolve().parents[1]
SCENE_NAMES = ("turntable", "tiltback", "wallpress")
IN_UNIT = bool(os.environ.get("INVOCATION_ID"))
mujoco_only = pytest.mark.skipif(not IN_UNIT, reason="MuJoCo runs only inside a capped systemd-run unit (brief §3)")


def _fake_labels(plan: dict, seed: int = 0) -> dict:
    """Synthetic outcome codes for every setup of a group (no MuJoCo): enough to exercise assembly, labels and scorers."""
    spec = SCENES[plan["scene"]]
    rng = np.random.default_rng(seed + plan["index"])
    setups = {}
    for member in ("base", "pair"):
        for facts in member_variants(plan, member).values():
            setups[spec.setup_key(facts)] = {"codes": {p: "".join(str(int(x)) for x in rng.integers(0, 3, K)) for p in C.PRIORS},
                                             "metric": {p: [0.0] * K for p in C.PRIORS}}
    return {"scene": plan["scene"], "index": plan["index"], "setups": setups, "unstable": 0, "cpu_s": 0.0}


def _records(scene: str, n: int) -> list[dict]:
    out = []
    for i in range(n):
        plan = plan_group(scene, i)
        out += assemble_group(plan, _fake_labels(plan), git="test", mujoco_version="3.9.0")
    return out


# --------------------------------------------------------------------------
# Catalogs and priors
# --------------------------------------------------------------------------


def test_prior_settings_follow_the_standard_alternative_rule():
    assert C.prior_range("friction", 0.4, 0.6, "alt_a") == pytest.approx((0.28, 0.42))
    assert C.prior_range("friction", 0.4, 0.6, "alt_b") == pytest.approx((0.52, 0.78))
    assert C.prior_range("spread", 0.4, 0.6, "alt_a") == pytest.approx((0.35, 0.65))  # 1.5 × wider about the same centre
    assert C.prior_range("spread", 0.4, 0.6, "alt_b") == pytest.approx((0.45, 0.55))  # half as wide
    assert C.prior_range("place", 0.0, 0.02, "alt_a")[1] == pytest.approx(0.03)
    assert C.prior_range("place", 0.0, 0.02, "alt_b")[1] == pytest.approx(0.015)
    assert C.prior_range("spread", 0.4, 0.6, "base") == (0.4, 0.6)


def test_every_hidden_quantity_has_at_least_four_named_options():
    assert len(C.TT_BASES) >= 4 and len(C.TT_TOPS) >= 4 and len(C.TT_CONDITIONS) >= 4
    assert all(len(rows) >= 4 for rows in C.TT_CONTENTS.values())
    assert len(C.TB_FLOORS) >= 4 and len(C.TB_CONTENTS) >= 4 and len(C.TB_BUILDS) >= 4
    assert len(C.WP_BACKS) >= 4 and len(C.WP_WALLS) >= 4 and len(C.WP_PADS) >= 4
    assert all(len(spec["builds"]) >= 4 for spec in C.WP_ITEMS.values())


def test_hidden_draws_are_shared_across_priors_and_variants():
    spec = SCENES["wallpress"]
    plan = plan_group("wallpress", 3)
    draws = spec.draws(np.random.default_rng(plan["draws_seed"]), K)
    a = hidden_values(spec, spec.ranges(plan["facts"]), draws, "base")
    b = hidden_values(spec, spec.ranges(plan["facts"]), draws, "alt_a")
    assert np.allclose(b["mu"], a["mu"] * 0.7)  # same draws, the friction range × 0.7
    assert np.allclose(b["dy"], a["dy"] * 1.5)  # same normals, σ × 1.5
    again = spec.draws(np.random.default_rng(plan["draws_seed"]), K)
    assert all(np.array_equal(draws[k], again[k]) for k in draws)


# --------------------------------------------------------------------------
# Planning
# --------------------------------------------------------------------------


@pytest.mark.parametrize("scene", SCENE_NAMES)
def test_plans_are_deterministic_and_pairs_change_exactly_one_stated_fact(scene):
    spec = SCENES[scene]
    plans = [plan_group(scene, i) for i in range(150)]
    assert plans[7] == plan_group(scene, 7)
    far = 0
    for plan in plans:
        base, partner = plan["facts"], plan["partner"]
        changed = [k for k in base if base[k] != partner.get(k)]
        assert changed == [plan["pair"]["field"]], (plan["index"], changed)
        assert spec.decisive_ok(base[spec.decisive]) and spec.decisive_ok(partner[spec.decisive])
        assert plan["cf"] not in (base[spec.decisive], partner[spec.decisive])
        assert plan["alt"] in spec.alternatives(base) and plan["alt"] in spec.alternatives(partner)
        far += plan["pair"]["closeness"] == "far"
        assert plan["pair"]["one_factor"] == (plan["pair"]["closeness"] != "far")
    assert 0.2 < far / len(plans) < 0.47  # about a third far-number pairs


@pytest.mark.parametrize("scene", SCENE_NAMES)
def test_static_aims_spread_over_zero_to_one(scene):
    aims = np.array([plan_group(scene, i)["aim_key"] for i in range(200)])
    hist = np.histogram(aims, bins=5, range=(0, 1))[0]
    assert hist.min() >= 20, hist  # roughly flat: every fifth of [0, 1] gets at least half its share


def test_soft_three_reads_paired_differences():
    up, down, same = soft_three(np.ones(32))
    assert up > 0.99 and down < 1e-6
    up, down, same = soft_three(np.zeros(32))
    assert same > 0.95 and abs(up - down) < 1e-12
    up, down, same = soft_three(np.array([-1.0] * 16 + [0.0] * 16))
    assert down > 0.99


# --------------------------------------------------------------------------
# Wording
# --------------------------------------------------------------------------


@pytest.mark.parametrize("scene", SCENE_NAMES)
def test_pair_texts_differ_only_where_the_fact_changed(scene):
    for i in range(60):
        plan = plan_group(scene, i)
        base, partner = assemble_group(plan, _fake_labels(plan), git="t", mujoco_version="3.9.0")
        a, b = base["request"]["state"], partner["request"]["state"]
        assert a != b
        sa, sb = a.replace("\n", " ").split(". "), b.replace("\n", " ").split(". ")
        assert len(sa) == len(sb)
        assert sum(x != y for x, y in zip(sa, sb)) <= 2, (plan["pair"], a, b)
        assert base["request"]["questions"].keys() == partner["request"]["questions"].keys()


@pytest.mark.parametrize("scene", SCENE_NAMES)
def test_briefings_are_second_person_metric_and_english(scene):
    for record in _records(scene, 30):
        state = record["request"]["state"]
        assert "you" in state.lower()
        assert not any(unit in state for unit in (" inch", " lb", " feet", " ft ", "°F"))
        assert state.isascii() or set(state) - set(chr(i) for i in range(128)) <= {"°", "×"}


def test_prompt_lengths_within_budget():
    from robo_jev.model.tokenizer import available_tokenizer, load_tokenizer

    found = available_tokenizer()
    if found is None:
        pytest.skip("no tokenizer file")
    tok = load_tokenizer(found[0])
    for scene in SCENE_NAMES:
        lengths = []
        for record in _records(scene, 60):
            q = record["request"]
            parts = [q["state"]]
            for question in q["questions"].values():
                crit = question.get("criteria")
                lines = crit if isinstance(crit, list) else [f"{k}: {v}" for k, v in (crit or {}).items()]
                parts.append(question["instructions"] + "\n" + "\n".join(lines))
            lengths.append(len(tok.encode("\n".join(parts), add_special_tokens=False).ids))
        assert np.median(lengths) <= 350 and max(lengths) <= 500, (scene, np.median(lengths), max(lengths))


# --------------------------------------------------------------------------
# Record contract, loader and scorers
# --------------------------------------------------------------------------


@pytest.mark.parametrize("scene", SCENE_NAMES)
def test_records_follow_the_contract(scene):
    from robo_jev.decisions.records import validate_record

    records = _records(scene, 20)
    ids = [r["id"] for r in records]
    assert len(set(ids)) == len(ids)
    for i, record in enumerate(records):
        assert validate_record(record) == []
        assert record["id"] == f"v06a2-{scene}-{i // 2:06d}" + ("p" if i % 2 else "")
        meta = record["meta"]
        assert meta["set"] == "v06-a2" and meta["split"] == "train" and meta["scene"] == scene
        kinds = meta["question_kind"]
        assert sorted(kinds.values()) == sorted(["key", "severe", "score_cf", "direction", "compare", "action", "outcome"])
        assert set(kinds) == set(record["request"]["questions"]) == set(record["labels"])
        assert meta["key_question"] == next(q for q, k in kinds.items() if k == "key")
        assert set(meta["labels_by_prior"]) == set(C.PRIORS) and record["labels"] == meta["labels_by_prior"]["base"]
        assert set(meta["robustness"]) == set(kinds)
        assert meta["compare_option"] in meta["action_options"] and meta["plan"]["options"][meta["compare_option"]] in meta["outcomes"]
        assert meta["key_events"] == list(EVENT[scene]) and meta["plan_key"] == PLAN_KEY[scene]
        partner = meta["pair"]["of"]
        assert partner == (record["id"] + "p" if i % 2 == 0 else record["id"][:-1])
        assert record["generator"]["k"] == K and record["generator"]["member"] == ("base" if i % 2 == 0 else "pair")
        for prior in C.PRIORS:
            for variant, freq in meta["outcomes"].items():
                assert abs(sum(freq[prior].values()) - 1) < 1e-9
        action = next(q for q, k in kinds.items() if k == "action")
        assert set(record["labels"][action]["expected_cost"]) == set(meta["action_options"])


def test_the_trainer_loader_reads_every_record(tmp_path):
    from robo_jev.decisions.train import load_training_records

    path = tmp_path / "a2.jsonl"
    records = [r for scene in SCENE_NAMES for r in _records(scene, 15)]
    path.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
    valid, summary = load_training_records(path)
    assert len(valid) == len(records) and summary["invalid"] == 0


def _synthetic_predictions(records: list[dict], seed: int = 0) -> list[dict]:
    rng = np.random.default_rng(seed)
    out = []
    for record in records:
        questions = {}
        for name, question in record["request"]["questions"].items():
            if question["type"] == "choice":
                keys = list(question["criteria"])
                p = rng.dirichlet(np.ones(len(keys)))
                answer = {"probabilities": dict(zip(keys, map(float, p)))}
                questions[name] = {"type": "choice", "n_listings": 1, "answer": answer, "listings": [{"options": keys, "values": None, "answer": answer}]}
            else:
                value = float(rng.random())
                answer = {"p_yes": value} if question["type"] == "noul" else {"value": value, "bins": None}
                questions[name] = {"type": question["type"], "n_listings": 1, "answer": answer, "listings": [{"options": None, "values": None, "answer": answer}]}
        out.append({"id": record["id"], "model": "synthetic", "questions": questions})
    return out


def test_infamily_readings_run_on_a_sample():
    from robo_jev.bench import infamily

    records = [r for scene in SCENE_NAMES for r in _records(scene, 25)]
    predictions = _synthetic_predictions(records)
    robust = infamily.robust_keys(records)
    assert robust and robust <= {(r["id"], q) for r in records for q in r["request"]["questions"]}
    direction = infamily.direction_items("a2", records, predictions)
    assert direction and all(cluster.split(":")[0] in SCENE_NAMES for cluster, _ in direction.values())
    with registered_with_infamily():
        comparison = infamily.comparison_items("a2", records, predictions)
    assert comparison and all(0.0 <= v <= 1.0 for _, v in comparison.values())
    assert "turntable" not in infamily.EVENT  # restored


def test_the_verdict_call_view_runs():
    spec = importlib.util.spec_from_file_location("a2_verdict_v05", REPO / "scripts" / "verdict_v05.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    records = [r for scene in SCENE_NAMES for r in _records(scene, 20)]
    items = module.set_calls("A2", records, _synthetic_predictions(records))
    metrics = module.call_metrics(items, n_boot=50, seed=0)
    assert metrics["n"] == len(items) > 0 and sum(metrics["classes"].values()) == metrics["n"]
    assert metrics["DC"]["value"] is not None
    other = module.set_calls("A2", records, _synthetic_predictions(records, seed=1))
    assert module.call_difference(other, items, n_boot=50)["n"] == len(items)


# --------------------------------------------------------------------------
# MuJoCo (capped unit only)
# --------------------------------------------------------------------------


@mujoco_only
def test_rollouts_are_deterministic():
    from robo_jev.authors.a2.sanity import determinism

    assert determinism() == {"turntable": True, "tiltback": True, "wallpress": True}


@mujoco_only
def test_each_law_flips_at_its_analytic_threshold():
    from robo_jev.authors.a2.sanity import threshold_checks

    rows = threshold_checks()
    assert {row["law"] for row in rows} == {"A", "B", "C"}
    assert all(row["ok"] for row in rows), [row for row in rows if not row["ok"]]


@mujoco_only
@pytest.mark.parametrize("scene", SCENE_NAMES)
def test_key_frequency_is_monotone_in_the_decisive_number(scene):
    from robo_jev.authors.a2.sanity import SWEEPS, sweep, violations

    field, factors, direction = SWEEPS[scene][0]
    for index in (0, 1):
        result = sweep(plan_group(scene, index), field, factors, k=12)
        assert violations(result["freqs"], direction) == 0, result


@mujoco_only
def test_only_wallpress_has_a_midphase_pair_and_only_it_refreshes_bounding_volumes():
    """Physics audit 2026-10-14: a run-time centre of mass leaves MuJoCo's stored bounding volumes behind, which matters only for body
    pairs that go through the midphase. Only wallpress has one (the world holds the floor and the wall), and only it refreshes."""
    import mujoco

    from robo_jev.authors.a2 import physics as PH

    scenes = {"turntable": PH.TurntableScene((0.065, 0.065, 0.24), 0.9), "tiltback": PH.TiltbackScene((0.32, 0.8, 1.8), 50.0),
              "wallpress": PH.WallpressScene((0.02, 0.5, 0.6), 5.0, 0.5)}
    pairs = {}
    for scene, obj in scenes.items():
        model = obj.scene.model
        pairs[scene] = [tuple(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b) or "world" for b in pair) for pair in obj.scene.midphase_pairs()]
    assert pairs == {"turntable": [], "tiltback": [], "wallpress": [("load", "world")]}
    assert {scene: obj.scene.refresh_bvh for scene, obj in scenes.items()} == {"turntable": False, "tiltback": False, "wallpress": True}
    assert SCENES["wallpress"].refresh_bvh is True and not hasattr(SCENES["turntable"], "refresh_bvh")
    # the refresh really acts: after a rollout (hidden centre of mass at the geometric centre) the load's stored bounding volumes sit
    # 1 mm (the compiled nominal inertial offset) further along each axis of its inertial frame than compiled; without it they stay
    from robo_jev.outside.physics import NOMINAL_IPOS

    for flag in (True, False):
        wp = PH.WallpressScene((0.02, 0.5, 0.6), 5.0, 0.5, refresh_bvh=flag)
        model, load = wp.scene.model, wp.scene.load
        start, count = int(model.body_bvhadr[load]), int(model.body_bvhnum[load])
        compiled = np.array(model.bvh_aabb[start : start + count])
        wp.run(force=60.0, reflex=2.0, mu=0.4, mass=4.0, pad_mu=0.6, seconds=0.05)
        moved = np.array(model.bvh_aabb[start : start + count]) - compiled
        assert np.allclose(moved[:, :3], NOMINAL_IPOS if flag else 0.0) and np.allclose(moved[:, 3:], 0.0)


@mujoco_only
@pytest.mark.parametrize("scene", ("turntable", "tiltback"))
def test_turntable_and_tiltback_rollouts_are_bit_identical_with_the_flag(scene):
    class Flagged(dict):  # every scene put in this cache re-expresses its bounding volumes after each centre-of-mass write
        def __setitem__(self, key, value):
            value.scene.refresh_bvh = True
            super().__setitem__(key, value)

    spec = SCENES[scene]
    plan = plan_group(scene, 11)
    draws = spec.draws(np.random.default_rng(plan["draws_seed"]), K)
    plain: dict = {}
    flagged = Flagged()
    for member in ("base", "pair"):
        for facts in member_variants(plan, member).values():
            vals = hidden_values(spec, spec.ranges(facts), draws, "alt_a")
            for k in range(0, K, 4):
                values = {n: float(vals[n][k]) for n in vals}
                assert spec.run(plain, facts, values) == spec.run(flagged, facts, values)
    assert flagged and all(obj.scene.refresh_bvh for obj in flagged.values())
    assert not any(obj.scene.refresh_bvh for obj in plain.values())


@mujoco_only
@pytest.mark.parametrize("scene", SCENE_NAMES)
def test_a_group_labels_end_to_end(scene):
    from robo_jev.authors.a2.generate import label_group
    from robo_jev.decisions.records import validate_record

    plan = plan_group(scene, 5)
    labelled = label_group(plan)
    assert labelled["unstable"] == 0
    assert labelled.get("refresh_bvh", False) == (scene == "wallpress")
    for entry in labelled["setups"].values():
        assert all(len(codes) == K for codes in entry["codes"].values())
    for record in assemble_group(plan, labelled, git="t", mujoco_version="3.9.0"):
        assert validate_record(record) == []
        assert record["generator"].get("refresh_bvh", False) == (scene == "wallpress")
    assert Counter(len(e["codes"]) for e in labelled["setups"].values()) == Counter({3: len(labelled["setups"])})
