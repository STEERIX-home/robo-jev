"""Task v0.4a — MuJoCo-labelled training scenes (three laws × new scenes): physics against closed forms, determinism, records and labels
under three priors, one-fact pairs, the internal scene holdout, text hygiene, zero overlap with T8b-style text, and loading through the
trainer."""

import ast
import copy
import dataclasses
import functools
import json
import math
import random
import re
import subprocess
import sys

import mujoco
import pytest
from helpers import REPO

from robo_jev.decisions.records import validate_record
from robo_jev.mjgen import families as fam
from robo_jev.mjgen import generate as gen
from robo_jev.mjgen import physics as ph
from robo_jev.mjgen.scenes import HOLDOUT_SCENES, LAWS, SCENES, TRAIN_SCENES
from robo_jev.mjgen.scenes import build as B
from robo_jev.mjgen.scenes import rigs
from robo_jev.mjgen.scenes import sanity as S

MASTER = 20261040
G = rigs.GRAVITY
PACKAGE = REPO / "src" / "robo_jev" / "mjgen" / "scenes"
NUMBER = re.compile(r"-?\d+(?:\.\d+)?")


@functools.lru_cache(maxsize=1)
def labelled() -> tuple:
    """One group with a pair per scene (a number pair or a factor pair), K = 4."""
    out = []
    for name in SCENES:
        for index in range(40):
            group = B.candidate(name, 810_000 + index, MASTER, k=4, pair_share=1.0)
            if group.pair is None:
                pair = B.factor_pair(dataclasses.replace(group, pair=None), random.Random(group.seed ^ 0xFAC7), k=4)
                if pair is None:
                    continue
                group = dataclasses.replace(group, pair=pair)
            out.append((group, B.label_group(group, k=4)))
            break
    return tuple(out)


@functools.lru_cache(maxsize=1)
def records() -> tuple:
    return tuple(r for group, lab in labelled() for r in B.group_records(group, lab, mujoco_version=mujoco.__version__))


# -- scenes and laws -------------------------------------------------------------------------------------------------------------


def test_nine_new_scenes_three_per_law_and_two_held_out_from_two_laws():
    by_law = {}
    for name, scene in SCENES.items():
        by_law.setdefault(scene.law, []).append(name)
    assert set(by_law) == set(LAWS) and all(len(v) == 3 for v in by_law.values())
    assert len(HOLDOUT_SCENES) == 2 and len({SCENES[s].law for s in HOLDOUT_SCENES}) == 2
    assert set(TRAIN_SCENES) == set(SCENES) - set(HOLDOUT_SCENES) and len(TRAIN_SCENES) == 7
    assert not set(SCENES) & {"tilt", "grip", "tower"}  # v0.3b's scenes are not regenerated


def test_held_out_scenes_share_no_catalog_name_robot_or_option_text_with_training_scenes():
    from robo_jev.mjgen.overlap import _catalog_names

    held = set().union(*(SCENES[s].catalog_names() for s in HOLDOUT_SCENES))
    train = set().union(*(SCENES[s].catalog_names() for s in TRAIN_SCENES)) | _catalog_names()[0]
    assert not held & train

    def option_texts(names):
        out = set()
        for r in records():
            if r["meta"]["scene"] in names:
                for q in r["request"]["questions"].values():
                    crit = q.get("criteria")
                    out |= set(crit.values()) if isinstance(crit, dict) else set(crit or [])
        return out

    assert not option_texts(HOLDOUT_SCENES) & option_texts(TRAIN_SCENES)


def test_excluded_scene_types_do_not_appear_in_catalogs():
    words = re.compile(r"\b(conveyor|belt|shelf|shelves|suction|vacuum|hook|strap|plank|rolling|knock|strike|bumper)\b")
    for name, scene in SCENES.items():
        for catalog_name in scene.catalog_names():
            assert not words.search(catalog_name), (name, catalog_name)
    for r in records():
        assert not words.search(r["request"]["state"].lower()), r["id"]


def test_the_scene_package_imports_neither_the_engine_nor_sealed_test_code():
    for path in PACKAGE.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            modules = [node.module or ""] if isinstance(node, ast.ImportFrom) else [a.name for a in node.names] if isinstance(node, ast.Import) else []
            names = [a.name for a in node.names] if isinstance(node, ast.ImportFrom) else []
            for module in modules:
                assert not module.startswith(("robo_jev.simgen", "robo_jev.t9", "robo_jev.outside.moments")), path
                if module == "robo_jev.outside":
                    assert set(names) <= {"physics"}, path


# -- physics ---------------------------------------------------------------------------------------------------------------------


def test_a_load_on_the_forks_slides_at_mu_g_and_tips_at_g_b_over_h_in_their_own_geometry():
    slide = S.analytic("forks_slide", mu=0.4, dims=(0.6, 0.9, 0.3))
    assert 0.97 <= slide["ratio"] <= 1.06
    tilted = S.analytic("forks_slide", mu=0.3, tilt=4.0, dims=(0.6, 0.9, 0.3))
    assert 0.97 <= tilted["ratio"] <= 1.06  # a mast tilted back resists a forward slide: g(μ + tan β)/(1 − μ tan β)
    turn = S.analytic("forks_turn_tip", dims=(0.8, 1.0, 1.2), spread=0.4, tine_w=0.12)
    assert 0.95 <= turn["ratio"] <= 1.05  # in a turn a wide load tips about the outer tine's edge, not its own


def test_a_bin_wall_right_ahead_keeps_a_tipping_item_upright():
    assert S.analytic("bin_wall_holds")["passed"]


def test_a_held_item_slips_at_two_mu_f_equal_m_times_its_load():
    lift = S.analytic("squeeze_lift", mass=6.0, mu=0.6, accel=2.0, dims=(0.35, 0.30, 0.30), pad=(0.06, 0.012, 0.07))
    brake = S.analytic("squeeze_brake", mass=80.0, mu=0.5, accel=3.0, dims=(0.8, 0.7, 1.2), pad=(0.35, 0.02, 0.4), hand=3000.0)
    assert 0.95 <= lift["ratio"] <= 1.05 and 0.95 <= brake["ratio"] <= 1.05


def test_an_item_falls_off_a_table_when_its_centre_of_mass_passes_the_edge():
    plain = S.analytic("table_edge", dims=(0.30, 0.20, 0.15))
    heavy_end = S.analytic("table_edge", dims=(0.40, 0.25, 0.10), cx=0.1)
    assert abs(plain["diff_m"]) < 0.008 and abs(heavy_end["diff_m"]) < 0.008


def test_a_base_tips_when_the_centre_of_mass_of_base_and_mover_passes_its_front_feet():
    for scene in ("reach", "cabinet"):
        visible, mass = SCENES[scene].sanity_cases()[0]
        out = S.analytic(f"{scene}_tip", visible=visible, mover_mass=mass)
        assert abs(out["diff_m"]) < 0.025, out


def test_a_walking_bob_needs_at_most_its_static_peak_load_to_hold():
    walk = S.analytic("squeeze_walk", mass=6.0, mu=0.6, rise=0.05, hz=1.8, dims=(0.35, 0.30, 0.30), pad=(0.06, 0.012, 0.07))
    assert 0.85 <= walk["ratio"] <= 1.02


@pytest.mark.parametrize("scene", ["forks", "plate", "table"])
def test_the_key_event_is_monotone_along_the_decisive_number(scene):
    result = S.monotone(scene, "decisive", 1, MASTER, 4)
    assert result["label_violations"] == 0


def test_rollouts_and_labels_are_deterministic_also_in_another_process():
    group = B.candidate("bin", 3, MASTER, k=2)
    first = S.label_hash(B.label_group(group, k=2))
    assert first == S.label_hash(B.label_group(group, k=2))
    code = ("from robo_jev.mjgen.scenes import build as B, sanity as S; "
            f"print(S.label_hash(B.label_group(B.candidate('bin', 3, {MASTER}, k=2), k=2)))")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True, env={"PYTHONPATH": str(REPO / "src"), "PATH": ""})
    assert out.stdout.strip() == first


def test_a_load_with_a_high_hidden_centre_of_mass_stays_on_two_fork_tines():
    """Regression: with both tines in one body, a load whose centre of mass was moved up at runtime fell through the tines or was
    launched off them (MuJoCo's midphase keeps a body's bounding box in its compile-time inertial frame); every support geom now has its
    own body."""
    tines = [("tine_l", (0.2, 0.155, -0.025), (0.54, 0.065, 0.025), True), ("tine_r", (0.2, -0.155, -0.025), (0.54, 0.065, 0.025), True)]
    rig = rigs.platform_rig("t", tines, tilt_deg=5.0, load_dims=(0.56, 0.73, 1.41), load_mass=55.0)
    for com in ((0.0, 0.0, 0.05), (0.0, 0.0, 0.24), (0.1, 0.1, 0.3)):
        rig.set_hidden(friction={"tine_l": 0.5, "tine_r": 0.5}, com={"load": com})
        mujoco.mj_forward(rig.model, rig.data)
        z0 = float(rig.data.xpos[rig.boxes["load"]["body"]][2])
        for _ in range(400):
            mujoco.mj_step(rig.model, rig.data)
        assert abs(float(rig.data.xpos[rig.boxes["load"]["body"]][2]) - z0) < 0.002 and rig.data.ncon > 0, com


def test_a_short_stop_holds_its_stated_rate_for_most_of_the_stop():
    """Regression: a stop shorter than its own ramp-up reached the stated deceleration only for an instant, and nothing moved."""
    for name in ("forks", "bin", "tray"):
        for index in range(20):
            v = B.candidate(name, 700 + index, MASTER, k=4, pair_share=0.0).visible
            assert v["ramp"] <= SCENES[name].duration(v) / 3 + 1e-9


# -- candidates, labels, records -------------------------------------------------------------------------------------------------


def test_candidates_are_deterministic_and_the_table_aim_is_exact():
    for name in SCENES:
        a, b = B.candidate(name, 11, MASTER, k=4), B.candidate(name, 11, MASTER, k=4)
        assert a.visible == b.visible and a.variants == b.variants and a.pair == b.pair
    hits = []
    for index in range(20):
        group = B.candidate("table", index, MASTER, k=32)
        target = min(1.0, max(0.0, group.plan["aim_key"]))
        hits.append(abs(group.notes["static_key"] - target) <= 0.1)
    assert sum(hits) >= 18


def test_records_are_valid_and_carry_labels_under_three_priors():
    v03_kinds = set(gen.QUESTION_NAMES["tilt"])
    for record in records():
        assert validate_record(record) == []
        meta = record["meta"]
        assert set(meta["labels_by_prior"]) == set(fam.PRIORS) and record["labels"] == meta["labels_by_prior"]["base"]
        assert set(meta["robustness"]) == set(record["request"]["questions"])
        kinds = set(meta["question_kind"].values())
        assert {"key", "direction"} <= kinds <= v03_kinds  # v0.3b's question kinds, nothing new
        assert meta["law"] == SCENES[meta["scene"]].law and record["generator"]["k"] == 4
        for q in record["request"]["questions"].values():
            if q["type"] == "score":
                assert len(q["criteria"]) == 2 and q["criteria"][0].startswith("0 = ") and q["criteria"][1].startswith("1 = ")


def test_labels_are_the_outcome_frequencies_of_the_variants():
    for group, lab in labelled():
        scene = SCENES[group.family]
        for record in B.group_records(group, lab, mujoco_version="test"):
            member = "pair" if record["id"].endswith("p") else "base"
            variants = lab["members"][member]["variants"]
            for prior in fam.PRIORS:
                labels = record["meta"]["labels_by_prior"][prior]
                planned = variants["planned"]["freq"][prior]
                assert labels[scene.question_names["key"]]["p_yes"] == pytest.approx(sum(planned[o] for o in scene.event))
                if scene.question_names["action"] in labels:
                    costs = labels[scene.question_names["action"]]["expected_cost"]
                    probs = labels[scene.question_names["action"]]["probabilities"]
                    assert max(probs, key=probs.get) in [k for k, c in costs.items() if c == min(costs.values())]


def test_paired_records_differ_in_exactly_one_stated_fact():
    by_id = {r["id"]: r for r in records()}
    pairs = [r for r in records() if r["meta"]["pair"] and r["meta"]["pair"]["role"] == "base"]
    assert len(pairs) == len(SCENES)
    for base in pairs:
        variant = by_id[base["meta"]["pair"]["of"]]
        a, b = base["request"]["state"], variant["request"]["state"]
        assert a != b
        changed = [(x, y) for x, y in zip(NUMBER.findall(a), NUMBER.findall(b)) if x != y]
        if base["meta"]["pair"]["fact"] == "number":
            assert NUMBER.split(a) == NUMBER.split(b) and len(changed) == 1, (base["id"], changed)
        else:
            assert NUMBER.findall(a) == NUMBER.findall(b), base["id"]  # a word fact: the numbers stay, one phrase changes
        assert set(base["request"]["questions"]) == set(variant["request"]["questions"])


def test_the_text_states_the_numbers_and_hides_the_hidden_values():
    for record in records():
        state = record["request"]["state"].lower()
        for word in ("friction", "coefficient", "centre of mass", "center of mass", "probability", "hidden", "prior"):
            assert word not in state, record["id"]
        assert len(record["request"]["state"]) < 1200


def test_factor_pairs_cover_word_and_number_factors_in_every_law():
    seen = set()
    for name in SCENES:
        for index in range(12):
            group = dataclasses.replace(B.candidate(name, 820_000 + index, MASTER, k=4, pair_share=0.0), pair=None)
            pair = B.factor_pair(group, random.Random(group.seed ^ 0xFAC7), k=4)
            if pair is None:
                continue
            assert pair["closeness"] == "factor" and pair["factor"] in SCENES[name].factors and pair["from"] != pair["value"]
            seen.add((SCENES[name].law, pair["fact"]))
    assert {(law, fact) for law in LAWS for fact in ("word", "number")} <= seen


def test_balanced_selection_and_stats_run_on_scene_records():
    kept, report = B.select_balanced(list(labelled()), {name: 1 for name in SCENES})
    assert len(kept) == len(SCENES) and all(v["kept"] == 1 for v in report.values())
    summary = B.stats(list(records()))
    assert summary["by_law"]["all"]["records"] == len(records()) and set(summary["by_scene"]) == set(SCENES)


# -- sealed tests stay outside ---------------------------------------------------------------------------------------------------


def test_no_shared_sentences_hashes_or_long_ngrams_with_t8b_style_text():
    from robo_jev.mjgen.overlap import overlap_report, t8b_texts_from_generator

    report = overlap_report(list(records()), t8b_texts_from_generator(per_family=6))
    assert report["verdict"]["no_shared_text_hashes"] and report["verdict"]["no_shared_sentences"], report["verdict"]
    assert report["verdict"]["no_shared_long_ngrams"], report["ngrams"]["5"]["examples"]


def test_the_assembly_refuses_held_out_scenes_in_a_training_file(tmp_path):
    import importlib.util

    spec = importlib.util.spec_from_file_location("mjgen_v04", REPO / "scripts" / "mjgen_v04.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    with pytest.raises(SystemExit):
        module.main(["assemble", "--scenes", "forks", "tray", "--root", str(tmp_path), "--out", str(tmp_path / "x.jsonl")])
    with pytest.raises(SystemExit):
        module.main(["assemble", "--scenes", "forks", "--root", str(tmp_path), "--out", str(tmp_path / "x.jsonl"), "--split", "test"])
    with pytest.raises(SystemExit):
        module._guard_out(REPO / "artifacts" / "simgen" / "v03" / "x.jsonl")


# -- the trainer reads the records -----------------------------------------------------------------------------------------------


def test_the_trainer_loads_the_records_and_prompts_stay_within_the_10_hz_budget(tmp_path):
    pytest.importorskip("transformers")
    from robo_jev.decisions import examples as E
    from robo_jev.decisions import train as T
    from robo_jev.model.tokenizer import available_tokenizer, load_tokenizer

    path = tmp_path / "v04a.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in records()), encoding="utf-8")
    loaded, check = T.load_training_records(path)
    assert check["invalid"] == 0 and len(loaded) == len(records())
    tokenizer = load_tokenizer(available_tokenizer()[0])
    lengths = []
    for record in loaded:
        for name in record["request"]["questions"]:
            (example,) = E.build_examples(record, [name], tokenizer, order_seed=E.FIXED_ORDER_SEED)
            assert abs(sum(example["target"]) - 1.0) < 1e-6
            lengths.append(len(example["tokens"]))
    assert max(lengths) <= 500
