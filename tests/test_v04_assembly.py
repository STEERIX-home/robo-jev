"""v0.4 training input (Task v0.4b): the action repair as a recorded pick rule, the MuJoCo recomposition, and the assembly's checks.

* **The pick rule** (``meta.question_pick``, :func:`robo_jev.decisions.examples.record_pick`): records without it are picked exactly as
  before (the trainer's default); under it, a record's q examples are the pick over its primary questions — the same as the record
  without its derived questions — plus every secondary question the plain pick (v0.3's) chooses. Counted on synthetic engine-like
  records: action examples = the untransformed layout's, derived examples = v0.3's, exactly; and on a slice of the real files when present.
* **The mix** (``scripts/convert_single_probe.py mix``): ``--extra-examples`` holds an extra part at an example count (recorded second
  pass), ``--general-records`` takes an exact general record count, pick rules are counted with the train config's seed and refused
  outside the base part.
* **The overlap check** (``scripts/system1_v03.py overlap``): ``--sealed-extra`` (B1, the H-set) and ``--mujoco-set``.
* **The assembly** (``scripts/assemble_v04.py``): the seeded, stratified subsample of whole pairs; the H-set check; the pick layout
  changes nothing but its field; the stand-in stays in scratch.
"""

from __future__ import annotations

import collections
import copy
import functools
import importlib.util
import json
import random
import sys
from pathlib import Path

import pytest
from helpers import FIXTURES, REPO
from test_v02_mix import _general, _probe_and_sealed, _write

from robo_jev.decisions import examples as E
from robo_jev.decisions.records import file_sha256, read_records
from robo_jev.simgen import system1_questions as s1

REQUESTS = FIXTURES / "decisions" / "requests.jsonl"
TRAIN_CONFIG = REPO / "configs" / "train" / "decisions-v0-lora-h100.yaml"
V02_TRAIN = REPO / "artifacts" / "simgen" / "v02" / "train-50k.jsonl"
ENGINE_S1 = REPO / "artifacts" / "simgen" / "v03" / "engine-s1.jsonl"


@functools.lru_cache(maxsize=None)
def _load(name: str):
    spec = importlib.util.spec_from_file_location(f"v04_{Path(name).stem}", REPO / "scripts" / name)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _old_plan(records, *, seed, epoch, q, augment):
    """The epoch plan as the trainer built it before Task v0.4b (verbatim logic) — the default must not move."""
    plan = []
    for index, record in enumerate(records):
        rid = str(record.get("id") or "")
        names = E.select_questions(record["request"]["questions"], q=q, rng=random.Random(E.derive_seed(seed, epoch, rid, "questions")))
        order_seed = E.derive_seed(seed, epoch, rid, "order") if augment else E.FIXED_ORDER_SEED
        plan.extend((index, name, order_seed) for name in names)
    random.Random(E.derive_seed(seed, epoch, "shuffle")).shuffle(plan)
    return plan


def _engine_like(i: int, *, derived: int) -> dict:
    """An engine-shaped training record: 1–2 action choices, a state choice, yes/no and score questions, then `derived` derived choice
    questions (as system1_questions appends them: meta.system1.derived, meta.question_families)."""
    questions, labels, basis = {}, {}, {}
    for k in range(1 + i % 2):
        questions[f"act_{k}"] = {"type": "choice", "instructions": f"Which action {k}?", "criteria": {"a": "go", "b": "stop"}}
        labels[f"act_{k}"] = {"type": "choice", "kind": "action", "probabilities": {"a": 0.4, "b": 0.6}}
        basis[f"act_{k}"] = {"kind": "action"}
    questions["state"] = {"type": "choice", "instructions": "Which state?", "criteria": {"x": "x", "y": "y"}}
    labels["state"] = {"type": "choice", "kind": "state", "probabilities": {"x": 1.0, "y": 0.0}}
    basis["state"] = {"kind": "state"}
    for k in range(2):
        questions[f"yn_{k}"] = {"type": "noul", "instructions": f"Will it happen {k}?"}
        labels[f"yn_{k}"] = {"type": "noul", "p_yes": 0.3}
        basis[f"yn_{k}"] = {"kind": "noul", "source": "event"}
    questions["sc"] = {"type": "score", "instructions": "How likely?", "criteria": ["0", "1"]}
    labels["sc"] = {"type": "score", "value": 0.7}
    basis["sc"] = {"kind": "score", "source": "event"}
    original = len(questions)
    families, made = {}, {}
    for k in range(derived):
        name = f"s1_band_{original + k}"
        questions[name] = {"type": "choice", "instructions": "Pick one of three levels.", "criteria": {"low": "l", "medium": "m", "high": "h"}}
        labels[name] = {"type": "choice", "kind": "state", "probabilities": {"low": 1.0, "medium": 0.0, "high": 0.0}}
        families[name] = "s1:band" if k % 3 else "s1:comparison"
        made[name] = {"kind": "band", "of": ["yn_0"], "p": [0.3]}
    meta = {"split": "train", "basis": basis, "question_families": families,
            "system1": {"version": s1.VERSION, "original_questions": original, "derived": made, "dropped": {}}}
    return {"schema": "simgen-request/0", "id": f"v0b-train-{i:06d}", "request": {"state": f"moment {i}", "questions": questions},
            "labels": labels, "meta": meta}


def _untransformed(record: dict) -> dict:
    out = copy.deepcopy(record)
    for name in record["meta"]["system1"]["derived"]:
        del out["request"]["questions"][name], out["labels"][name]
    out["meta"]["question_families"], out["meta"]["system1"] = {}, {}
    return out


# --------------------------------------------------------------------------
# the pick rule
# --------------------------------------------------------------------------


def test_records_without_a_pick_rule_are_planned_exactly_as_before():
    fixtures = read_records(REQUESTS)
    engine = [_engine_like(i, derived=i % 5) for i in range(60)]
    for records in (fixtures, engine):
        for seed, epoch, augment in ((0, 0, True), (3, 1, True), (1, 0, False)):
            assert E.epoch_plan(records, seed=seed, epoch=epoch, q=2, augment=augment) == _old_plan(records, seed=seed, epoch=epoch, q=2, augment=augment)
            assert E.examples_per_epoch(records, q=2) == sum(min(2, len(r["request"]["questions"])) for r in records)
    assert not E.has_pick_rule(engine) and all(E.pick_rule(r) is None for r in engine)


def test_the_pick_rule_trains_the_primary_pick_plus_the_secondary_questions_the_plain_pick_chooses():
    for i in range(80):
        record = _engine_like(i, derived=1 + i % 6)
        ruled = s1.with_pick_rule(record)
        assert E.pick_rule(ruled) == list(record["meta"]["system1"]["derived"])
        for seed in (0, 1, 7):
            names = E.record_pick(ruled, q=2, seed=seed, epoch=0)
            plain = E.record_pick(record, q=2, seed=seed, epoch=0)  # v0.3's pick (no rule)
            primary = E.record_pick(dict(_untransformed(record), id=record["id"]), q=2, seed=seed, epoch=0)  # v0.2's pick
            secondary = set(record["meta"]["system1"]["derived"])
            assert names[:2] == primary and len(set(primary) & secondary) == 0
            assert names[2:] == [n for n in plain if n in secondary] and len(names) == len(set(names))
    with pytest.raises(ValueError):
        s1.with_pick_rule(s1.with_pick_rule(_engine_like(0, derived=2)))
    with pytest.raises(ValueError, match="no derived"):
        s1.with_pick_rule(_engine_like(0, derived=0))


def test_on_engine_like_records_the_rule_restores_the_original_pick_and_keeps_the_derived_exposure_exactly():
    records = [_engine_like(i, derived=1 + i % 7) for i in range(400)]
    ruled = [s1.with_pick_rule(r) for r in records]
    before = [_untransformed(r) for r in records]
    for seed in (0, 1):
        v02, v03, v04 = (s1.exposure(rs, q=2, seed=seed) for rs in (before, records, ruled))
        derived = {k: v for k, v in v04.items() if k.startswith("s1:")}
        assert derived == {k: v for k, v in v03.items() if k.startswith("s1:")} and sum(derived.values()) > 0
        assert {k: v for k, v in v04.items() if not k.startswith("s1:")} == v02, "the original questions get the untransformed pick back"
        assert v04["choice:action"] > v03["choice:action"] and sum(v04.values()) == sum(v02.values()) + sum(derived.values())
        assert E.examples_per_epoch(ruled, q=2, seed=seed) == sum(v04.values()) == len(E.epoch_plan(ruled, seed=seed, epoch=0, q=2, augment=True))
    with pytest.raises(ValueError, match="seed"):
        E.examples_per_epoch(ruled, q=2)


@pytest.mark.parametrize("rule, needle", [
    ({"version": "qpick-0", "secondary": []}, "must be"), ({"version": E.QUESTION_PICK_VERSION}, "must be"), ("all", "must be"),
    ({"version": E.QUESTION_PICK_VERSION, "secondary": ["nope"]}, "does not ask"),
    ({"version": E.QUESTION_PICK_VERSION, "secondary": ["sc", "sc"]}, "repeats"),
])  # fmt: skip
def test_a_malformed_pick_rule_is_refused(rule, needle):
    record = _engine_like(1, derived=2)
    record["meta"]["question_pick"] = rule
    with pytest.raises(ValueError, match=needle):
        E.record_pick(record, q=2, seed=0, epoch=0)


def test_on_a_slice_of_the_real_files_the_rule_gives_v02s_pick_and_v03s_derived_picks():
    if not (V02_TRAIN.is_file() and ENGINE_S1.is_file()):
        pytest.skip("artifacts/simgen/v02/train-50k.jsonl or v03/engine-s1.jsonl not present")

    def head(path: Path, n: int) -> list[dict]:
        out = []
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                out.append(json.loads(line))
                if len(out) == n:
                    break
        return out

    v02, v03 = head(V02_TRAIN, 1500), head(ENGINE_S1, 1500)
    assert [r["id"] for r in v02] == [r["id"] for r in v03]
    for seed in (0, 1):
        for a, b in zip(v02, v03):
            ruled = s1.with_pick_rule(b)
            names = E.record_pick(ruled, q=2, seed=seed, epoch=0)
            n_primary = min(2, len(a["request"]["questions"]))
            assert names[:n_primary] == E.record_pick(a, q=2, seed=seed, epoch=0)
            assert names[n_primary:] == [n for n in E.record_pick(b, q=2, seed=seed, epoch=0) if n in b["meta"]["system1"]["derived"]]


def test_the_trainer_counts_a_ruled_epoch_with_its_seed_records_the_rule_and_stays_within_one_epoch(tmp_path):
    from test_decisions_train import tiny_config, tokenizer

    from robo_jev.decisions import train as T

    records = [s1.with_pick_rule(_engine_like(i, derived=1 + i % 4)) for i in range(12)]
    trainer = T.DecisionTrainer(tiny_config(tmp_path, seed=5), records=records, tokenizer=tokenizer(), log=None)
    assert trainer.examples_per_epoch == E.examples_per_epoch(trainer.train_records, q=2, seed=5) == len(trainer._plan(0))
    pick = trainer.manifest["data"]["question_pick"]
    assert pick["records"] == len(trainer.train_records) and pick["version"] == E.QUESTION_PICK_VERSION
    assert pick["secondary_examples_epoch0"] == trainer.examples_per_epoch - 2 * len(trainer.train_records)
    assert all(len(trainer.step_examples(s)) > 0 for s in range(trainer.max_steps))
    with pytest.raises(ValueError, match="one epoch"):
        T.DecisionTrainer(tiny_config(tmp_path, seed=5, epochs=2), records=records, tokenizer=tokenizer(), log=None)
    plain = T.DecisionTrainer(tiny_config(tmp_path, seed=5, epochs=2), records=[_engine_like(i, derived=2) for i in range(12)], tokenizer=tokenizer(), log=None)
    assert plain.manifest["data"]["question_pick"] is None and plain.max_steps == 2 * plain.steps_per_epoch


# --------------------------------------------------------------------------
# the mix
# --------------------------------------------------------------------------


def _paired(tmp_path: Path, scenes: tuple[str, ...], pairs: int, *, set_name: str = "v03b-mujoco-train", name: str = "mujoco") -> Path:
    base = read_records(REQUESTS)
    out = []
    for scene in scenes:
        for i in range(pairs):
            a, b = f"{set_name[:4]}-{scene}-{i:06d}", f"{set_name[:4]}-{scene}-{i:06d}p"
            for rid, partner, role in ((a, b, "base"), (b, a, "variant")):
                record = copy.deepcopy(base[(i + len(out)) % len(base)])
                record["id"] = rid
                record["meta"] = {"set": set_name, "split": "train", "scene": scene, "language": "en", "pair": {"of": partner, "kind": "numeric", "role": role}}
                out.append(record)
    path = _write(tmp_path / f"{name}.jsonl", out)
    (tmp_path / f"{name}.manifest.json").write_text(json.dumps({"sha256": file_sha256(path)}), encoding="utf-8")
    return path


def _mix_args(tmp_path: Path, base: Path, out: Path, **extra) -> list[str]:
    probe, sealed = _probe_and_sealed(tmp_path)
    return ["mix", "--base", str(base), "--probe", str(probe), "--sealed", str(sealed), "--train-config", str(TRAIN_CONFIG), "--out", str(out)]


def test_an_extra_part_held_at_an_example_count_gets_a_recorded_scene_balanced_second_pass(tmp_path):
    script = _load("convert_single_probe.py")
    mujoco = _paired(tmp_path, ("tilt", "conveyorx", "trayx"), pairs=5)  # 30 records, 60 examples at q = 2
    out = tmp_path / "train.jsonl"
    common = [*_mix_args(tmp_path, REQUESTS, out), "--extra", f"mujoco={mujoco}"]
    assert script.main([*common, "--extra-examples", "mujoco=70"]) == 0
    manifest = json.loads(out.with_name("train.manifest.json").read_text(encoding="utf-8"))
    (extra,) = manifest["extra"]
    assert extra["target_examples"] == 70 and extra["target_reached"] is True and extra["target_share"] is None
    assert 70 <= manifest["examples_per_epoch"]["mujoco"] <= 70 + 4 and extra["second_pass"]["records"] % 2 == 0
    assert "target 70 train examples" in extra["second_pass"]["why"]
    copies = [r for r in read_records(out) if str(r["id"]).endswith("~r2")]
    assert [r["meta"]["scene"] for r in copies][:6] == ["tilt", "tilt", "conveyorx", "conveyorx", "trayx", "trayx"], "round-robin over the scenes"
    assert script.main([*common, "--extra-examples", "mujoco=10"]) == 0  # already reached: no second pass
    assert json.loads(out.with_name("train.manifest.json").read_text(encoding="utf-8"))["extra"][0]["second_pass"] is None
    with pytest.raises(SystemExit, match="exclude each other"):
        script.main([*common, "--extra-examples", "mujoco=70", "--extra-share", "mujoco=0.2"])
    with pytest.raises(SystemExit, match="even a whole second pass"):
        script.main([*common, "--extra-examples", "mujoco=500"])
    with pytest.raises(SystemExit, match="positive integer"):
        script.main([*common, "--extra-examples", "mujoco=x"])


def test_the_general_part_can_be_taken_at_an_exact_record_count_including_the_recorded_second_pass(tmp_path):
    script = _load("convert_single_probe.py")
    general = _general(tmp_path, n_rows=4)  # 20 records
    out = tmp_path / "train.jsonl"
    common = [*_mix_args(tmp_path, REQUESTS, out), "--general", str(general)]
    assert script.main([*common, "--general-records", "27", "--general-repeat-if-short"]) == 0
    manifest = json.loads(out.with_name("train.manifest.json").read_text(encoding="utf-8"))
    block = manifest["general"]
    assert block["records_used"] == 27 and block["records_target"] == 27 and block["target_share"] is None and block["second_pass"]["records"] == 7
    lines = out.read_bytes().splitlines(keepends=True)[-27:]
    pool = read_records(general)
    assert [json.loads(line)["id"] for line in lines] == [r["id"] for r in pool] + [f"{r['id']}~r2" for r in pool[:7]]
    assert lines[:20] == general.read_bytes().splitlines(keepends=True)
    with pytest.raises(SystemExit, match="exactly one of"):
        script.main([*common, "--general-records", "5", "--general-share", "0.2"])
    with pytest.raises(SystemExit, match="exactly one of"):
        script.main(common)
    with pytest.raises(SystemExit, match="needs --general-repeat-if-short"):
        script.main([*common, "--general-records", "27"])


def test_the_mix_counts_a_ruled_base_with_the_config_seed_and_refuses_rules_elsewhere(tmp_path):
    script = _load("convert_single_probe.py")
    ruled = [s1.with_pick_rule(_engine_like(i, derived=1 + i % 4)) for i in range(30)]
    base = _write(tmp_path / "engine.jsonl", ruled)
    out = tmp_path / "train.jsonl"
    assert script.main(_mix_args(tmp_path, base, out)) == 0
    manifest = json.loads(out.with_name("train.manifest.json").read_text(encoding="utf-8"))
    train, _ = E.split_dev(ruled, fraction=0.02, salt="v0d-dev")
    assert manifest["examples_per_epoch"]["base"] == E.examples_per_epoch(train, q=2, seed=0) > 2 * len(train)
    assert manifest["question_pick"]["base_records_with_rule"] == 30 and manifest["question_pick"]["seed"] == 0 and manifest["train_config"]["seed"] == 0
    extra = _write(tmp_path / "ruled-extra.jsonl", [dict(r, id=f"x-{r['id']}") for r in ruled])
    with pytest.raises(SystemExit, match="only accepted in the base part"):
        script.main([*_mix_args(tmp_path, REQUESTS, tmp_path / "o.jsonl"), "--extra", f"mujoco={extra}"])


# --------------------------------------------------------------------------
# the overlap check
# --------------------------------------------------------------------------


def test_the_overlap_check_reads_extra_sealed_files_and_extra_mujoco_sets(tmp_path, monkeypatch):
    script = _load("system1_v03.py")
    records = read_records(_paired(tmp_path, ("standin_s1",), pairs=4, set_name="v04x-scenes"))
    monkeypatch.setattr(script, "SEALED", {})
    hold = copy.deepcopy(records[0])
    hold["id"] = "hset-0001"
    hset = _write(tmp_path / "scene-holdout.jsonl", [hold])
    clean = _write(tmp_path / "clean.jsonl", records[1:])
    args = ["overlap", "--input", str(clean), "--out", str(tmp_path / "c.json"), "--sealed-extra", f"H-set={hset}"]
    assert script.main(args) == 2, "an unknown set is a hit (unknown part)"
    assert script.main([*args, "--mujoco-set", "v04x-scenes"]) == 0
    report = json.loads((tmp_path / "c.json").read_text(encoding="utf-8"))
    assert report["parts"]["mujoco"]["records"] == len(records) - 1 and report["sealed_extra"] == ["H-set"] and "H-set" in report["sealed"]
    dirty = _write(tmp_path / "dirty.jsonl", records)  # records[0] shares its whole request and state with the H-set record
    assert script.main(["overlap", "--input", str(dirty), "--out", str(tmp_path / "d.json"), "--sealed-extra", f"H-set={hset}",
                        "--mujoco-set", "v04x-scenes"]) == 2  # fmt: skip
    hits = " ".join(sum(json.loads((tmp_path / "d.json").read_text(encoding="utf-8"))["first_hits"].values(), []))
    assert "H-set: request" in hits and "H-set: state" in hits
    with pytest.raises(SystemExit, match="does not exist"):
        script.main([*args, "--sealed-extra", f"B1={tmp_path / 'absent.jsonl'}"])
    with pytest.raises(SystemExit, match="NAME=PATH"):
        script.main([*args, "--sealed-extra", "noequals"])
    with pytest.raises(SystemExit, match="NAME=PATH"):
        script.main([*args, "--sealed-extra", f"H-set={hset}"])  # a repeated name


# --------------------------------------------------------------------------
# the assembly
# --------------------------------------------------------------------------


def _family_records(family: str, groups: int, *, seed: int = 0) -> list[dict]:
    """v0.3b-shaped pair groups: base key labels spread over the tenths; a third of the pairs one-factor."""
    rng = random.Random(f"{family}|{seed}")
    out = []
    for g in range(groups):
        a, b = f"v03b-{family}-{g:06d}", f"v03b-{family}-{g:06d}p"
        key = (g % 10 + rng.random()) / 10
        for rid, partner, role, value in ((a, b, "base", key), (b, a, "variant", 1 - key)):
            out.append({"schema": "simgen-request/0", "id": rid,
                        "request": {"state": f"{family} moment {g} {role}", "questions": {"k": {"type": "noul", "instructions": "Will it fail?"},
                                                                                           "d": {"type": "choice", "instructions": "Which way?",
                                                                                                 "criteria": {"up": "u", "down": "d"}}}},
                        "labels": {"k": {"type": "noul", "p_yes": round(value, 4)}, "d": {"type": "choice", "kind": "state", "probabilities": {"up": 1.0, "down": 0.0}}},
                        "meta": {"set": "v03b-mujoco-train", "split": "train", "scene": family, "key_question": "k",
                                 "pair": {"of": partner, "kind": "numeric", "role": role, "factor": "wet" if g % 3 == 0 else None}}})
    return out


def test_the_subsample_keeps_whole_pairs_equal_per_key_tenth_seeded_and_deterministic():
    script = _load("assemble_v04.py")
    records = [r for family in ("tilt", "grip", "tower") for r in _family_records(family, 120)]
    chosen, report = script.subsample(records, per_family=40, seed=11)
    again, _ = script.subsample(list(records), per_family=40, seed=11)
    other, _ = script.subsample(records, per_family=40, seed=12)
    assert [r["id"] for r in chosen] == [r["id"] for r in again] and [r["id"] for r in chosen] != [r["id"] for r in other]
    ids = {r["id"] for r in chosen}
    assert all((r["id"] in ids) == (r["meta"]["pair"]["of"] in ids) for r in records), "whole pairs only"
    for family in ("tilt", "grip", "tower"):
        entry = report[family]
        assert entry["groups_kept"] == 20 and entry["records_kept"] == 40 and entry["base_key_tenths_kept"] == [2] * 10
    assert [r["id"] for r in chosen] == [r["id"] for r in records if r["id"] in ids], "file order"
    with pytest.raises(SystemExit, match="families"):
        script.subsample([*records, *_family_records("other", 2)], per_family=4, seed=1)


def test_the_mujoco_part_refuses_holdout_scenes_reused_v03_scenes_and_non_train_records():
    script = _load("assemble_v04.py")
    config = _load("assemble_v04.py")._config(TRAIN_CONFIG)
    v03 = [r for family in ("tilt", "grip", "tower") for r in _family_records(family, 40)]

    def scene(name, groups, split="train", prefix="v04"):
        rows = _family_records(name, groups, seed=3)
        for r in rows:
            r["id"] = r["id"].replace("v03b", prefix)
            r["meta"]["pair"]["of"] = r["meta"]["pair"]["of"].replace("v03b", prefix)
            r["meta"].update(set="v04-scenes", split=split)
            r["request"]["state"] += f" {prefix}"
        return rows

    scenes = scene("forks", 10) + scene("tray", 10)
    holdout = scene("rack", 5, split="test", prefix="v04h")
    records, report = script.build_mujoco(v03, scenes, holdout, per_family=20, seed=1, config=config)
    assert report["records_subsample"] == 60 and report["records_scenes"] == 40 and report["holdout"]["pass"] is True
    assert [r["id"] for r in records[:60]] == [r["id"] for r in script.subsample(v03, per_family=20, seed=1)[0]]
    with pytest.raises(SystemExit, match="H-set"):
        script.build_mujoco(v03, scenes, scene("forks", 3, split="test", prefix="v04h"), per_family=20, seed=1, config=config)
    leaked = copy.deepcopy(scenes[0])  # a holdout record with a training scene record's whole request and state
    leaked["id"], leaked["meta"]["split"], leaked["meta"]["scene"] = "v04h-rack-999999", "test", "rack"
    with pytest.raises(SystemExit, match="H-set"):
        script.build_mujoco(v03, scenes, [leaked, *holdout[1:]], per_family=20, seed=1, config=config)
    with pytest.raises(SystemExit, match="re-uses v0.3b"):
        script.build_mujoco(v03, scenes + scene("tilt", 2, prefix="v04t"), holdout, per_family=20, seed=1, config=config)
    with pytest.raises(SystemExit, match="only train-split"):
        script.build_mujoco(v03, scenes + scene("bin", 2, split="test", prefix="v04b"), holdout, per_family=20, seed=1, config=config)


def test_the_holdout_check_finds_shared_ids_requests_states_and_scenes():
    script = _load("assemble_v04.py")
    train = _family_records("forks", 4)
    hold = copy.deepcopy(_family_records("rack", 2))
    for r in hold:
        r["meta"]["split"] = "test"
    assert script.holdout_check(train, hold)["pass"] is True
    same_text = copy.deepcopy(train[0])
    same_text["id"], same_text["meta"]["scene"], same_text["meta"]["split"] = "h-1", "rack", "test"
    result = script.holdout_check(train, [same_text])
    assert result["pass"] is False and result["shared"]["requests"] == 1 and result["shared"]["states"] == 1
    assert script.holdout_check(train, [dict(copy.deepcopy(hold[0]), id=train[0]["id"])])["shared"]["ids"] == 1
    trained = copy.deepcopy(hold[0])
    trained["meta"]["split"] = "train"
    assert script.holdout_check(train, [trained])["pass"] is False


def test_the_pick_step_changes_nothing_but_its_field_and_refuses_sealed_outputs(tmp_path):
    from robo_jev.simgen.record import dumps

    script = _load("assemble_v04.py")
    records = [_engine_like(i, derived=1 + i % 3) for i in range(20)]
    source = tmp_path / "engine-s1.jsonl"
    source.write_text("".join(dumps(r) + "\n" for r in records), encoding="utf-8")
    out = tmp_path / "engine-s1-pick.jsonl"
    assert script.main(["pick", "--input", str(source), "--out", str(out)]) == 0
    got = read_records(out)
    for before, after in zip(records, got):
        rule = after["meta"].pop("question_pick")
        assert after == before and rule == {"version": E.QUESTION_PICK_VERSION, "secondary": list(before["meta"]["system1"]["derived"])}
    manifest = json.loads(out.with_name("engine-s1-pick.manifest.json").read_text(encoding="utf-8"))
    assert manifest["out"]["sha256"] == file_sha256(out) and manifest["checks"]["secondary_questions"] == sum(len(r["meta"]["system1"]["derived"]) for r in records)
    loose = tmp_path / "loose.jsonl"
    loose.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")  # not the engine's byte-stable form
    with pytest.raises(SystemExit, match="byte-stable"):
        script.main(["pick", "--input", str(loose), "--out", str(tmp_path / "x.jsonl")])
    for place in ("artifacts/simgen/v03/x.jsonl", "artifacts/t9/x.jsonl", "artifacts/b1/x.jsonl", "artifacts/t8/x.jsonl"):
        with pytest.raises(SystemExit, match="refusing to write"):
            script.main(["pick", "--input", str(source), "--out", str(REPO / place)])


def test_the_stand_in_is_written_only_under_scratch(tmp_path):
    script = _load("assemble_v04.py")
    with pytest.raises(SystemExit, match="only under artifacts/scratch"):
        script.main(["standin", "--out-dir", str(tmp_path / "standin")])


def test_the_exposure_report_checks_the_brief_against_v03s_counts():
    script = _load("assemble_v04.py")
    config = script._config(TRAIN_CONFIG)
    engine = [s1.with_pick_rule(_engine_like(i, derived=1 + i % 5)) for i in range(50)]
    mujoco = _family_records("tilt", 10)
    report = script.exposure_report(engine + mujoco, config=config, seeds=[0, 1], mujoco_sets={"v03b-mujoco-train"})
    for seed in ("0", "1"):
        entry = report["seeds"][seed]
        assert entry["examples_per_epoch"]["total"] == entry["examples_per_epoch"]["engine"] + entry["examples_per_epoch"]["mujoco"]
        assert entry["derived"] == entry["band"] + entry["comparison"] > 0 and set(entry["checks"]) >= {
            "action_ge_21297", "derived_within_10pct", "steps_le_1.25x", "mujoco_examples_within_5pct", "mujoco_share_0.20"}
        assert entry["checks"]["steps_le_1.25x"] is True and entry["checks"]["action_ge_21297"] is False  # 50 records: far below 21,297


def test_the_built_v04_input_when_present():
    root = REPO / "artifacts" / "simgen" / "v04"
    summary_path = root / "train-v04.assembly.json"
    if not summary_path.is_file():
        pytest.skip("artifacts/simgen/v04/train-v04.jsonl not assembled yet (scripts/assemble_v04.py all, after task v0.4a)")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["standin"] is False and summary["overlap"] == 0
    assert summary["files"]["train-v04.jsonl"]["sha256"] == file_sha256(root / "train-v04.jsonl")
    exposure = json.loads((root / "train-v04.exposure.json").read_text(encoding="utf-8"))
    assert all(v["identical"] for v in exposure["reference"]["parts_identical"].values()), "d1 and general byte-identical to v0.3's"
    seed0 = exposure["seeds"]["0"]
    assert seed0["action"] == 21297 and seed0["derived"] == 18551 and seed0["checks"]["mujoco_examples_within_5pct"] and seed0["checks"]["steps_le_1.25x"]
    overlap = json.loads((root / "train-v04.overlap.json").read_text(encoding="utf-8"))
    assert overlap["total_hits"] == 0 and {"B1", "H-set"} <= set(overlap["sealed"]) and not overlap["missing_sealed"]
