"""Author A4 (v0.6 held-out test author): the sealed registration, the scenes' physics against their closed forms, the compiled centre
of mass, groups, labels, the record contract, the one-fact pairs' text, the in-family readings and the call view on synthetic answers,
and the written test set when it is present. MuJoCo runs here are single rollouts (fast); the labelled set's sanity report is
``artifacts/v06/a4/sanity.md``."""

import copy
import difflib
import importlib.util
import json
import math
import sys
from pathlib import Path
from unittest import mock

import pytest
from helpers import FIXTURES, REPO

from robo_jev.authors.a4 import build as B
from robo_jev.authors.a4 import design as D
from robo_jev.authors.a4 import label as L
from robo_jev.authors.a4 import radio as R
from robo_jev.authors.a4 import sims as S
from robo_jev.data.sealed import NEVER_SENT_SETS, SEALED_RECORD_FILES, SEALED_RECORD_PREFIXES, SEALED_RECORD_SETS, refuse_sealed_records, sealed_record_set
from robo_jev.decisions.records import read_records, validate_record

TEST_FILE = REPO / "artifacts" / "v06" / "a4" / "test.jsonl"


def _script(name: str):
    spec = importlib.util.spec_from_file_location(f"a4test_{Path(name).stem}", REPO / "scripts" / name)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _fake_sim(group: dict) -> dict:
    """Outcomes from the closed form instead of MuJoCo (the label and record logic do not depend on the simulator)."""
    scene = group["scene"]
    points = D.draw_points(group["draws_seed"], scene)
    draws = []
    for vis in group["records"]:
        specs = D.prior_specs(scene, vis)
        variants = D.resolve_variants(scene, vis, group["options"], group["cf_value"])
        per = {}
        for variant in variants:
            phys = D.variant_facts(scene, vis, variant, variants)
            per[variant] = {}
            for prior in D.PRIORS:
                key, severe = D.key_analytic(scene, phys, D.hidden_values(scene, specs[prior], points))
                outs = D.OUTCOMES[scene]
                per[variant][prior] = [outs[2] if s else (outs[1] if k else outs[0]) for k, s in zip(key, severe)]
        draws.append(per)
    return {"draws": draws, "rollouts": 0, "unstable": 0}


@pytest.fixture(scope="module")
def groups() -> list[dict]:
    out = []
    for scene in D.SCENES:
        for index in range(4):
            group = L.build_group(scene, index, 0)
            assert group is not None
            out.append(group)
    return out


@pytest.fixture(scope="module")
def records(groups) -> list[dict]:
    return [r for g in groups for r in B.group_records(g, _fake_sim(g), git="test")]


# --------------------------------------------------------------------------
# sealing
# --------------------------------------------------------------------------


def test_the_a4_test_set_is_registered_as_sealed_and_never_sent():
    assert "v06-a4" in SEALED_RECORD_SETS and SEALED_RECORD_PREFIXES["v06-a4"] == "v06a4-"
    assert SEALED_RECORD_FILES["v06-a4"] == "artifacts/v06/a4/test.jsonl" and "v06-a4" in NEVER_SENT_SETS
    assert B.SET_NAME == "v06-a4" and B.ID_PREFIX + "-" == SEALED_RECORD_PREFIXES["v06-a4"]


def test_a4_records_are_recognised_by_set_or_id_and_refused_by_the_trainer(records):
    from robo_jev.decisions.train import DecisionTrainer

    plain = read_records(FIXTURES / "decisions" / "requests.jsonl")
    assert sealed_record_set(plain[0]) is None
    assert all(sealed_record_set(r) == "v06-a4" for r in records)
    by_id = json.loads(json.dumps(plain[0]))
    by_id["id"] = "v06a4-bank-000123"
    by_set = json.loads(json.dumps(plain[0]))
    by_set["meta"] = {**(by_set.get("meta") or {}), "set": "v06-a4"}
    for marked in (by_id, by_set):
        assert sealed_record_set(marked) == "v06-a4"
        with pytest.raises(ValueError, match="sealed outside test"):
            refuse_sealed_records([plain[0], marked], where="training input")
        with pytest.raises(ValueError, match="sealed outside test"):
            DecisionTrainer({}, records=[*plain, marked], log=None)


def test_the_training_mix_assembly_refuses_an_a4_record():
    script = _script("convert_single_probe.py")
    plain = read_records(FIXTURES / "decisions" / "requests.jsonl")
    marked = dict(plain[1], id="v06a4-ledge-000007p")
    with pytest.raises(SystemExit, match="sealed outside test"):
        script.outside_sealed_check({"mujoco": [plain[0], marked]}, [])


def test_the_bench_refuses_to_send_a4_to_any_api():
    bench = _script("bench.py")
    plain = read_records(FIXTURES / "decisions" / "requests.jsonl")
    bench.refuse_never_sent({"T2": [plain[0]]})
    with pytest.raises(SystemExit, match="never sent"):
        bench.refuse_never_sent({"T2": [plain[0], dict(plain[1], id="v06a4-anchor-000001")]})


def test_the_script_writes_only_under_its_folder(tmp_path):
    script = _script("author_a4.py")
    script.refuse_write(REPO / "artifacts" / "v06" / "a4" / "x.json")
    for bad in (tmp_path / "x.json", REPO / "artifacts" / "v06" / "a1" / "x.json", REPO / "artifacts" / "t9" / "T9.jsonl"):
        with pytest.raises(SystemExit, match="only under"):
            script.refuse_write(bad)


# --------------------------------------------------------------------------
# physics
# --------------------------------------------------------------------------


@pytest.mark.parametrize("mu", [0.3, 0.6])
def test_bank_slides_at_atan_mu(mu):
    edge = math.degrees(math.atan(mu))
    rig = lambda: S.bank_scene((0.40, 0.60, 0.40), 20.0, (0.0, 0.0, 0.0))  # noqa: E731 — b/h = 1, so sliding comes first
    assert S.run_bank(rig(), slope_deg=edge - 1.5, mu=mu)["outcome"] == "stays_put"
    assert S.run_bank(rig(), slope_deg=edge + 1.5, mu=mu)["outcome"] == "slides_down"


def test_bank_tips_at_atan_b_over_h_when_friction_holds():
    # b = 0.15, h = 0.25 + 0.05 → b/h = 0.5 → 26.6°
    rig = lambda: S.bank_scene((0.30, 0.60, 0.50), 20.0, (0.0, 0.0, 0.05))  # noqa: E731
    assert S.run_bank(rig(), slope_deg=25.0, mu=0.95)["outcome"] == "stays_put"
    assert S.run_bank(rig(), slope_deg=28.5, mu=0.95)["outcome"] == "tips_over"


@pytest.mark.parametrize("cx", [-0.1, 0.1])
def test_ledge_nose_drops_once_the_centre_of_mass_is_past_the_edge(cx):
    edge = 0.5 - cx
    assert S.run_ledge(S.ledge_scene((1.0, 0.6, 0.5), 100.0, 0.3, (cx, 0.0, 0.05), edge - 0.03), mu=0.5)["outcome"] == "holds"
    assert S.run_ledge(S.ledge_scene((1.0, 0.6, 0.5), 100.0, 0.3, (cx, 0.0, 0.05), edge + 0.03), mu=0.5)["outcome"] != "holds"


def test_the_compiled_centre_of_mass_keeps_box_contacts():
    """A robot body with a large centre-of-mass offset rests on the upper level instead of sinking into it (the run-time body_ipos route
    lost these contacts — module docstring of sims)."""
    import mujoco

    rig = S.ledge_scene((0.8, 0.6, 0.4), 90.0, 0.3, (0.08, 0.0, 0.12), 0.1)
    rig.model.geom_friction[rig.surface, 0] = 0.5
    mujoco.mj_forward(rig.model, rig.data)
    z0 = float(rig.data.xpos[rig.load][2])
    mujoco.mj_step(rig.model, rig.data, nstep=300)
    assert rig.data.ncon > 0 and abs(float(rig.data.xpos[rig.load][2]) - z0) < 0.003
    assert abs(rig.com_offset()[2] - 0.12) < 1e-4


def test_anchor_slips_at_the_peak_tension_and_is_dragged_at_the_steady_tension():
    theta, m_r, mu, drag, accel = 10.0, 400.0, 0.55, 0.04, 0.8
    g = 9.81
    capacity = (mu * math.cos(math.radians(theta)) - math.sin(math.radians(theta))) * m_r * g
    per_kg = g * (math.sin(math.radians(theta)) + drag * math.cos(math.radians(theta)))
    peak, steady = capacity / (per_kg + accel), capacity / per_kg
    rig = S.anchor_scene((1.2, 0.8, 0.6), m_r)

    def run(mass):
        return S.run_anchor(rig, robot_slope_deg=theta, mu=mu, load_mass=mass, load_slope_deg=theta, drag=drag, start_accel=accel, line_speed=0.4)["outcome"]

    assert run(0.9 * peak) == "holds_fast"
    assert run(0.5 * (peak + steady) * 1.05) == "creeps"
    assert run(1.15 * steady) == "dragged"


def test_rollouts_are_deterministic():
    a = S.run_ledge(S.ledge_scene((0.9, 0.6, 0.45), 80.0, 0.35, (0.03, 0.0, 0.02), 0.5), mu=0.4)
    b = S.run_ledge(S.ledge_scene((0.9, 0.6, 0.45), 80.0, 0.35, (0.03, 0.0, 0.02), 0.5), mu=0.4)
    assert a == b


# --------------------------------------------------------------------------
# priors, groups, labels
# --------------------------------------------------------------------------


def test_the_alternative_priors_follow_the_rule():
    vis = D.sample_visible("bank", __import__("numpy").random.default_rng(3))
    specs = D.prior_specs("bank", vis)
    base, a, b = specs["base"], specs["alt_a"], specs["alt_b"]
    assert a["mu"][1:] == pytest.approx([0.7 * base["mu"][1], 0.7 * base["mu"][2]]) and b["mu"][1:] == pytest.approx([1.3 * base["mu"][1], 1.3 * base["mu"][2]])
    centre, half = (base["com_z"][1] + base["com_z"][2]) / 2, (base["com_z"][2] - base["com_z"][1]) / 2
    assert a["com_z"][1:] == pytest.approx([centre - 1.5 * half, centre + 1.5 * half]) and b["com_z"][1:] == pytest.approx([centre - 0.5 * half, centre + 0.5 * half])
    assert a["slope_dev"][2] == pytest.approx(1.5 * base["slope_dev"][2]) and b["slope_dev"][2] == pytest.approx(0.75 * base["slope_dev"][2])
    anchor = D.prior_specs("anchor", D.sample_visible("anchor", __import__("numpy").random.default_rng(4)))
    assert anchor["alt_a"]["drag"][1] == pytest.approx(0.7 * anchor["base"]["drag"][1])
    assert anchor["alt_b"]["mass_ratio"][2] - anchor["alt_b"]["mass_ratio"][1] == pytest.approx(0.5 * (anchor["base"]["mass_ratio"][2] - anchor["base"]["mass_ratio"][1]))


def test_groups_are_deterministic_and_pairs_change_one_stated_fact(groups):
    for group in groups:
        again = L.build_group(group["scene"], group["index"], 0)
        assert json.dumps(again, sort_keys=True, default=float) == json.dumps(group, sort_keys=True, default=float)
        a, b = group["records"]
        changed = [k for k in a if a[k] != b[k]]
        assert changed == [group["pair"]["field"]]
        decisive = D.SCENE_SPECS[group["scene"]].decisive
        assert (group["pair"]["factor"] is None) == (group["pair"]["field"] == decisive)


def test_the_partners_text_differs_only_where_the_fact_changes(groups):
    for group in groups:
        states = [R.render(group, m)[0] for m in (0, 1)]
        words = [s.split() for s in states]
        diff = [w for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(a=words[0], b=words[1]).get_opcodes() if tag != "equal"
                for w in words[0][i1:i2] + words[1][j1:j2]]
        assert diff, group["pair"]
        assert len(diff) <= 16, (group["pair"], diff)
        q0, q1 = (R.render(group, m)[1] for m in (0, 1))
        assert list(q0) == list(q1) and all(q0[n]["type"] == q1[n]["type"] for n in q0)


def test_records_pass_the_loader_and_carry_the_contract(records):
    for r in records:
        assert validate_record(r) == [], r["id"]
        meta = r["meta"]
        assert r["id"].startswith("v06a4-") and meta["set"] == "v06-a4" and meta["split"] == "test" and r["generator"]["k"] == 32
        assert set(meta["question_kind"].values()) == {"key", "severe", "score_cf", "direction", "compare", "action", "outcome"}
        assert r["labels"] == meta["labels_by_prior"]["base"] and set(meta["labels_by_prior"]) == set(D.PRIORS)
        assert set(meta["robustness"]) == set(r["request"]["questions"])
        assert meta["key_events"] == list(D.EVENT[meta["scene"]]) and meta["plan_option"] == "the_plan"
        compare = next(q for q, k in meta["question_kind"].items() if k == "compare")
        assert set(r["request"]["questions"][compare]["criteria"]) == {"the_plan", meta["compare_option"], "no_difference"}
        action = next(q for q, k in meta["question_kind"].items() if k == "action")
        assert set(r["labels"][action]["expected_cost"]) == set(meta["action_options"])
        partner = next(x for x in records if x["id"] == meta["pair"]["of"])
        assert partner["meta"]["pair"]["of"] == r["id"] and partner["generator"]["group"] == r["generator"]["group"]


def test_labels_follow_the_outcome_frequencies(groups):
    group = groups[0]
    sim = _fake_sim(group)
    names = R.QUESTION_NAMES[group["scene"]]
    lab = L.record_labels(group, sim, 0, names)
    planned = sim["draws"][0]["planned"]["base"]
    events = D.EVENT[group["scene"]]
    assert lab["labels_by_prior"]["base"][names["key"]]["p_yes"] == pytest.approx(sum(o in events for o in planned) / 32)
    up = L.paired_change([0.0] * 32, [1.0] * 32)
    same = L.paired_change([1.0] * 32, [1.0] * 32)
    assert up["up"] > 0.999 and same["same"] > 0.999 and L.paired_change([1.0] * 16 + [0.0] * 16, [0.0] * 32)["down"] > 0.999


def test_in_family_readings_and_the_call_view_run_on_synthetic_answers(records):
    from robo_jev.authors.a4.checks import _synthetic
    from robo_jev.bench import infamily

    oracle = _synthetic(records, noise=0.0, seed=0, listings=1)
    noisy = _synthetic(records, noise=0.3, seed=1, listings=2)
    with mock.patch.dict(infamily.EVENT, {s: tuple(v) for s, v in D.EVENT.items()}), mock.patch.dict(infamily.PLAN_KEY, dict(D.PLAN_KEY)):
        assert infamily.robust_keys(records)
        comparison = infamily.comparison_items("A4", records, oracle)
        assert comparison and all(v == 1.0 for _, v in comparison.values())
        assert all(0.0 <= v <= 1.0 for _, v in infamily.comparison_items("A4", records, noisy).values())
    direction = infamily.direction_items("A4", records, oracle)
    assert direction and all(v == 1.0 for _, v in direction.values())
    verdict = _script("verdict_v05.py")
    calls = verdict.set_calls("A4", records, noisy)
    assert len(calls) == sum(1 for r in records for q in r["request"]["questions"].values() if q["type"] in ("noul", "score"))
    metrics = verdict.call_metrics(calls, n_boot=50, seed=0)
    assert metrics["n"] == len(calls) and metrics["DC"]["value"] is not None
    assert verdict.call_difference(calls, verdict.set_calls("A4", records, oracle), n_boot=50, seed=0)["n"] == len(calls)


# --------------------------------------------------------------------------
# the written set (when present)
# --------------------------------------------------------------------------


@pytest.mark.skipif(not TEST_FILE.is_file(), reason="the A4 test set is not built here")
def test_the_written_test_set_is_sealed_and_well_formed():
    rows = read_records(TEST_FILE)
    assert 780 <= len(rows) <= 830
    assert all(sealed_record_set(r) == "v06-a4" and r["meta"]["split"] == "test" for r in rows)
    assert all(validate_record(r) == [] for r in rows)
    assert len({r["id"] for r in rows}) == len(rows)
    manifest = json.loads((TEST_FILE.parent / "manifest.json").read_text(encoding="utf-8"))
    from robo_jev.decisions.records import file_sha256

    assert manifest["sha256"] == file_sha256(TEST_FILE) and manifest["sealed"] is True and manifest["records"] == len(rows)
    with pytest.raises(ValueError, match="sealed outside test"):
        refuse_sealed_records(copy.deepcopy(rows[:2]), where="training input")
