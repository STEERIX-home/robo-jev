"""Task v0.3b — MuJoCo-labelled training families (tilt, grip, tower): physics sanity, records, one-fact pairs, labels under three
priors, determinism, zero overlap with the sealed outside test T8b, and loading through the trainer and the diagnosis script."""

import ast
import copy
import functools
import importlib.util
import json
import math
import re
import subprocess
import sys

import mujoco
import numpy as np
import pytest
from helpers import REPO

from robo_jev.decisions.records import validate_record
from robo_jev.mjgen import families as fam
from robo_jev.mjgen import generate as gen
from robo_jev.mjgen import physics as ph
from robo_jev.mjgen import sanity
from robo_jev.mjgen.overlap import ngrams, overlap_report, sentences, t8b_texts_from_generator

MASTER = 20261013
G = ph.GRAVITY
PACKAGE = REPO / "src" / "robo_jev" / "mjgen"


@functools.lru_cache(maxsize=1)
def labelled_groups() -> tuple:
    """Two groups per family with a pair each, K = 4 (fast)."""
    out = []
    for family in fam.FAMILIES:
        found = 0
        index = 0
        while found < 2:
            group = fam.candidate(family, 800_000 + index, MASTER, k=4, pair_share=1.0)
            index += 1
            if group.pair is None:
                continue
            out.append((group, fam.label_group(group, k=4)))
            found += 1
    return tuple(out)


@functools.lru_cache(maxsize=1)
def records() -> tuple:
    return tuple(r for group, labelled in labelled_groups() for r in gen.group_records(group, labelled, mujoco_version=mujoco.__version__))


# -- physics -------------------------------------------------------------------------------------------------------------


def test_the_hidden_centre_of_mass_and_mass_are_in_force():
    rig = ph.tilt_rig((0.30, 0.30, 0.50), 4.0, "along")
    rig.set_hidden(friction={"deck_geom": 0.5}, com={"load": (0.04, -0.01, 0.06)})
    mujoco.mj_forward(rig.model, rig.data)
    rot = np.asarray(rig.data.xmat[rig.boxes["load"]["body"]]).reshape(3, 3)
    assert np.allclose(rig.com_offset("load"), rot @ np.array([0.04, -0.01, 0.06]), atol=1e-12)
    grip = ph.grip_rig((0.10, 0.08, 0.16), 1.0)
    grip.set_hidden(friction={"pad_l": 0.5, "pad_r": 0.5}, com={"load": (0.0, 0.0, 0.0)}, mass={"load": 2.5})
    assert grip.model.body_mass[grip.boxes["load"]["body"]] == pytest.approx(2.5)


def test_non_box_loads_and_same_frame_loads_are_refused():
    xml = ph.tilt_xml((0.2, 0.2, 0.4), 2.0).replace('type="box" size="0.100000 0.100000 0.200000"', 'type="cylinder" size="0.100000 0.200000"')
    with pytest.raises(ValueError):
        ph.Rig(xml, boxes=("load",), dims={"load": (0.2, 0.2, 0.4)})
    xml = ph.tilt_xml((0.2, 0.2, 0.4), 2.0).replace(f'pos="{ph.NOMINAL_IPOS} {ph.NOMINAL_IPOS} {ph.NOMINAL_IPOS}"', 'pos="0 0 0"')
    with pytest.raises(ValueError):
        ph.Rig(xml, boxes=("load",), dims={"load": (0.2, 0.2, 0.4)})


def test_a_box_slides_at_tan_theta_equal_mu_and_tips_at_tan_theta_equal_b_over_h():
    assert 0.97 <= sanity.analytic_tilt_slide(0.4)["ratio"] <= 1.03
    assert 0.95 <= sanity.analytic_tilt_tip(0.20, 0.60, 0.5)["ratio"] <= 1.03


def test_a_tipping_box_is_read_as_a_tip_not_a_slide():
    """Regression: a tall box that starts to tip moves its centre ≈ (H/2)·sin φ by rotation alone; that is not a slide."""
    rig = ph.tilt_rig((0.20, 0.30, 0.60), 4.0)
    assert ph.run_tilt(rig, ph.pitch_profile(-24.0, 1.5), mu=1.5, com=(0, 0, 0))["outcome"] == "tip"
    assert ph.run_tilt(rig, ph.pitch_profile(-24.0, 1.5), mu=0.3, com=(0, 0, 0))["outcome"] == "slide"


def test_the_grip_slips_below_two_mu_f_equal_m_g_plus_a_and_does_not_creep():
    assert 0.97 <= sanity.analytic_grip("lift", 3.0)["ratio"] <= 1.03
    assert 0.97 <= sanity.analytic_grip("swing", 5.0)["ratio"] <= 1.03
    assert sanity.creep_grip()["passed"]


def test_a_top_box_falls_when_its_centre_of_mass_passes_the_edge_and_two_boxes_go_together_past_the_joint_centre():
    top = sanity.analytic_tower_top(0.40, 0.30)
    assert abs(top["found"] - top["expected"]) < 0.005
    pair = sanity.analytic_tower_pair(0.12)
    assert abs(pair["found"] - pair["expected"]) < 0.006


@pytest.mark.parametrize("sweep", list(sanity.SWEEPS))
def test_the_key_event_is_monotone_along_each_sweep(sweep):
    result = sanity.monotone(sweep, 1, MASTER, 4)
    assert result["label_violations"] == 0 and result["draw_violations"] == 0


def test_rollouts_and_labels_are_deterministic_also_in_another_process():
    group = fam.candidate("grip", 3, MASTER, k=2)
    first = sanity.label_hash(fam.label_group(group, k=2))
    assert first == sanity.label_hash(fam.label_group(group, k=2))
    code = ("from robo_jev.mjgen import families as f, sanity as s; "
            f"print(s.label_hash(f.label_group(f.candidate('grip', 3, {MASTER}, k=2), k=2)))")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True, env={"PYTHONPATH": str(REPO / "src"), "PATH": ""})
    assert out.stdout.strip() == first


# -- candidates and labels -----------------------------------------------------------------------------------------------


def test_candidates_are_deterministic_and_aimed_at_their_key_target():
    for family in fam.FAMILIES:
        a, b = fam.candidate(family, 11, MASTER), fam.candidate(family, 11, MASTER)
        assert a.visible == b.visible and a.variants == b.variants and a.pair == b.pair
    hits = []
    for index in range(30):
        group = fam.candidate("tilt", index, MASTER)
        target = min(1.0, max(0.0, group.plan["aim_key"]))
        hits.append(abs(group.notes["static_key"] - target) <= 0.15)
    assert sum(hits) >= 25  # the static key rate lands near the uniform target (inverse-CDF aiming)


def test_alternative_priors_shift_friction_and_widen_or_narrow_the_spreads():
    hidden = {"mu": ("mu", 0.4, 0.6), "cx": ("u", -0.1, 0.1, -0.4, 0.4), "eps1": ("n", 0.0, 0.01)}
    a, b = fam.prior_ranges(hidden, "alt_a"), fam.prior_ranges(hidden, "alt_b")
    assert a["mu"] == ("u", pytest.approx(0.28), pytest.approx(0.42)) and b["mu"] == ("u", pytest.approx(0.52), pytest.approx(0.78))
    assert a["cx"] == ("u", pytest.approx(-0.15), pytest.approx(0.15)) and b["cx"] == ("u", pytest.approx(-0.05), pytest.approx(0.05))
    assert a["eps1"][2] == pytest.approx(0.015) and b["eps1"][2] == pytest.approx(0.0075)


def test_records_are_valid_and_carry_labels_under_three_priors():
    for record in records():
        assert validate_record(record) == []
        meta = record["meta"]
        assert set(meta["labels_by_prior"]) == set(fam.PRIORS) and record["labels"] == meta["labels_by_prior"]["base"]
        for prior in fam.PRIORS:
            assert set(meta["labels_by_prior"][prior]) == set(record["request"]["questions"])
        assert set(meta["robustness"]) == set(record["request"]["questions"]) and all("robust" in r for r in meta["robustness"].values())
        assert set(meta["outcomes"]["planned"]) == set(fam.PRIORS)
        assert meta["key_question"] in record["request"]["questions"]
        kinds = set(meta["question_kind"].values())
        assert {"key", "direction"} <= kinds  # every record has the key yes/no and a counterfactual inside it
        for q in record["request"]["questions"].values():
            if q["type"] == "score":
                assert len(q["criteria"]) == 2 and q["criteria"][0].startswith("0 = ") and q["criteria"][1].startswith("1 = ")


def test_labels_are_the_outcome_frequencies_of_the_variants():
    for group, labelled in labelled_groups():
        for record in gen.group_records(group, labelled, mujoco_version="test"):
            member = "pair" if record["id"].endswith("p") else "base"
            variants = labelled["members"][member]["variants"]
            names = gen.QUESTION_NAMES[group.family]
            for prior in fam.PRIORS:
                labels = record["meta"]["labels_by_prior"][prior]
                planned = variants["planned"]["freq"][prior]
                assert labels[names["key"]]["p_yes"] == pytest.approx(sum(planned[o] for o in fam.EVENT[group.family]))
                assert record["meta"]["outcomes"]["planned"][prior] == planned
                if names["action"] in labels:
                    costs = labels[names["action"]]["expected_cost"]
                    probs = labels[names["action"]]["probabilities"]
                    assert max(probs, key=probs.get) in [k for k, c in costs.items() if c == min(costs.values())]


def test_direction_and_comparison_labels_follow_the_shared_draws():
    same = gen.contrast_probs([True, False] * 16, [True, False] * 16)
    assert same["same"] > 0.99
    up = gen.contrast_probs([False] * 32, [True] * 16 + [False] * 16)
    assert up["more"] > 0.99 and abs(sum(up.values()) - 1.0) < 1e-9
    down = gen.contrast_probs([True] * 32, [True] * 30 + [False] * 2)
    assert down["same"] > down["less"]  # a 6 % drop is "about the same" (band 0.1)


def test_paired_records_differ_in_exactly_one_stated_number():
    number = re.compile(r"-?\d+(?:\.\d+)?")
    by_id = {r["id"]: r for r in records()}
    pairs = [r for r in records() if r["meta"]["pair"] and r["meta"]["pair"]["role"] == "base"]
    assert len(pairs) == 6
    for base in pairs:
        variant = by_id[base["meta"]["pair"]["of"]]
        assert variant["meta"]["pair"]["of"] == base["id"] and variant["meta"]["pair"]["role"] == "variant"
        a, b = base["request"]["state"], variant["request"]["state"]
        assert number.split(a) == number.split(b)
        changed = [(x, y) for x, y in zip(number.findall(a), number.findall(b)) if x != y]
        assert len(changed) == 1, (base["id"], changed)
        assert set(base["request"]["questions"]) == set(variant["request"]["questions"])
        assert set(base["meta"]["pair"]["direction_by_prior"]) == set(fam.PRIORS)


def test_the_text_states_the_numbers_and_hides_the_hidden_values():
    for record in records():
        state = record["request"]["state"].lower()
        for word in ("friction", "coefficient", "centre of mass", "center of mass", "probability", "hidden", "prior"):
            assert word not in state
        visible = record["meta"]["visible"]
        if record["meta"]["scene"] == "tilt":
            assert f"{visible['angle']:.1f}°" in record["request"]["state"]
        elif record["meta"]["scene"] == "grip":
            assert f"{visible['squeeze']:.1f} N" in record["request"]["state"] and f"{visible['accel']:.1f} m/s²" in record["request"]["state"]
        assert len(record["request"]["state"]) < 2500


def test_balanced_selection_spreads_the_key_label_over_bins_and_drops_unstable_groups():
    def fake(index, rate, unstable=0):
        k = 10
        outcomes = ["tip"] * int(round(rate * k)) + ["stay"] * (k - int(round(rate * k)))
        group = fam.Group("tilt", index, index, {}, {}, {}, {}, "log", 0, [])
        labelled = {"flags": {"unstable": unstable}, "members": {"base": {"variants": {"planned": {"outcomes": {"base": outcomes}}}}}}
        return group, labelled

    items = [fake(i, 0.0) for i in range(8)] + [fake(10 + i, 1.0) for i in range(2)] + [fake(20, 0.5)] + [fake(30, 0.5, unstable=1)]
    kept, report = gen.select_balanced(items, {"tilt": 4}, bins=2)
    assert [g.index for g, _ in kept] == [0, 1, 10, 11]
    assert report["tilt"]["dropped_unstable"] == 1 and report["tilt"]["kept_by_bin"] == [2, 2]


def test_stats_report_label_histograms_pairs_and_robustness():
    summary = gen.stats(list(records()))
    pooled = summary["by_family"]["all"]
    assert sum(pooled["key_label_hist10"]) == len(records())
    assert pooled["pairs"]["pairs"] == 6 and pooled["pairs"]["share_of_records"] == 1.0
    assert 0 <= pooled["robust"]["robust"] <= pooled["robust"]["questions"] == sum(len(r["request"]["questions"]) for r in records())


# -- T8b stays an outside test ---------------------------------------------------------------------------------------------


def test_the_families_and_question_names_differ_from_t8b():
    from robo_jev.outside import moments as t8

    assert not set(fam.FAMILIES) & set(t8.FAMILIES)
    t8_questions = {q for r in t8b_texts_from_generator(per_family=2) for q in r["request"]["questions"]}
    assert not t8_questions & {q for names in gen.QUESTION_NAMES.values() for q in names.values()}


def test_no_shared_sentences_hashes_long_ngrams_or_catalog_names_with_t8b():
    report = overlap_report(list(records()), t8b_texts_from_generator(per_family=8))
    assert report["verdict"]["passed"], report["verdict"]
    assert report["ngrams"]["6"]["shared"] == 0 and report["ngrams"]["8"]["shared"] == 0
    assert report["catalog_names"]["shared"] == []


def test_the_overlap_check_catches_planted_t8b_wording():
    t8b = t8b_texts_from_generator(per_family=2)
    planted = copy.deepcopy(records()[0])
    planted["request"]["state"] += " " + sentences(t8b[0]["request"]["state"])[2]
    report = overlap_report([planted], t8b)
    assert not report["verdict"]["no_shared_sentences"] and not report["verdict"]["passed"]
    assert ngrams("A b c d e f", 6) == {("a", "b", "c", "d", "e", "f")}


def test_the_generator_imports_neither_simgen_nor_t8b_catalogs_or_text():
    """Only T8's physics primitives may be imported (and T8's moments module by the overlap check, which only compares text)."""
    for path in PACKAGE.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            modules = [node.module or ""] if isinstance(node, ast.ImportFrom) else [a.name for a in node.names] if isinstance(node, ast.Import) else []
            names = [a.name for a in node.names] if isinstance(node, ast.ImportFrom) else []
            for module in modules:
                assert not module.startswith("robo_jev.simgen"), path
                if module == "robo_jev.outside":
                    assert set(names) <= {"physics", "moments"}, path
                    if "moments" in names:
                        assert path.name == "overlap.py", path
                if module.startswith("robo_jev.outside."):
                    assert module in ("robo_jev.outside.physics",) or path.name == "overlap.py", path


# -- the trainer and the diagnosis tool read the records -------------------------------------------------------------------


def test_the_trainer_loads_the_records_and_builds_its_examples(tmp_path):
    pytest.importorskip("transformers")
    from robo_jev.decisions import examples as E
    from robo_jev.decisions import train as T
    from robo_jev.model.tokenizer import available_tokenizer, load_tokenizer

    path = tmp_path / "v03b.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in records()), encoding="utf-8")
    loaded, check = T.load_training_records(path)
    assert check["invalid"] == 0 and len(loaded) == len(records())
    tokenizer = load_tokenizer(available_tokenizer()[0])
    plan = E.epoch_plan(loaded, seed=3, epoch=0, q=2, augment=True)
    for index, name, order_seed in plan[:12]:
        (example,) = E.build_examples(loaded[index], [name], tokenizer, order_seed=order_seed)
        assert abs(sum(example["target"]) - 1.0) < 1e-6 and example["tokens"]


def test_one_tiny_training_step_runs_on_the_records(tmp_path):
    pytest.importorskip("transformers")
    from robo_jev.decisions import train as T
    from robo_jev.model.backbone_qwen import torch_reference_kernels
    from robo_jev.model.tokenizer import available_tokenizer, load_tokenizer

    config = {"model_id": "tiny", "device": "cpu", "dtype": "float32", "out_dir": str(tmp_path / "runs"), "run_name": "v03b", "mode": "lora",
              "lora": {"r": 4, "alpha": 8, "dropout": 0.0, "targets": ["q_proj", "v_proj", "in_proj_qkv", "out_proj", "down_proj"]},
              "batch_examples": 4, "micro_batch_tokens": 8192, "lr": 1e-3, "warmup_ratio": 0.0, "epochs": 1, "dev_fraction": 0.0, "dev_every": 0,
              "checkpoint_every": 0, "seed": 3}
    with torch_reference_kernels():
        trainer = T.DecisionTrainer(config, records=list(records()), tokenizer=load_tokenizer(available_tokenizer()[0]), log=None)
        metrics = trainer.train_step()
    assert math.isfinite(metrics["loss"]) and metrics["examples"] == 4


def test_the_diagnosis_script_reads_the_records(tmp_path):
    pytest.importorskip("transformers")
    from robo_jev.model.backbone_qwen import torch_reference_kernels

    spec = importlib.util.spec_from_file_location("diagnose_decisions_v03b", REPO / "scripts" / "diagnose_decisions.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    path = tmp_path / "v03b.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in records()[:4]), encoding="utf-8")
    out = tmp_path / "diag.json"
    with torch_reference_kernels():
        assert module.main(["--model", "tiny", "--input", str(path), "--out", str(out), "--n-boot", "10", "--batch", "8", "--format", "plain"]) == 0
    summary = json.loads(out.read_text(encoding="utf-8"))
    assert summary["validation"] == {"read": 4, "invalid": 0, "problems": {}} and summary["config"]["records"] == 4
