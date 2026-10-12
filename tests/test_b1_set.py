"""Task B1 — a fresh in-family set from the v0.3b generator: bookkeeping options (set name, split, id prefix) and the zero-overlap check
with the training records (seeds, scene parameters, ids, state texts)."""

import copy
import dataclasses
import functools
import importlib.util
import json
import sys

import mujoco
import pytest
from helpers import REPO

from robo_jev.decisions.records import file_sha256, validate_record
from robo_jev.mjgen import families as fam
from robo_jev.mjgen import generate as gen
from robo_jev.mjgen.overlap import NARRATIVE_FIELDS, scene_parameters, train_overlap_report

MASTER = 20261013
FRESH = 202610131
ALL_KINDS = ["key", "direction", "outcome", "action", "compare", "score_cf", "severe"]


def _script(name: str):
    spec = importlib.util.spec_from_file_location(f"{name}_b1test", REPO / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@functools.lru_cache(maxsize=None)
def labelled(master: int, start: int, pair_share: float = 1.0) -> tuple:
    """One group per family (K = 4, every question kind asked), labelled in MuJoCo."""
    out = []
    for family in fam.FAMILIES:
        index = start
        while True:
            group = fam.candidate(family, index, master, k=4, pair_share=pair_share)
            index += 1
            if pair_share and group.pair is None:
                continue
            group = dataclasses.replace(group, mix=list(ALL_KINDS))
            out.append((group, fam.label_group(group, k=4)))
            break
    return tuple(out)


def build(master: int, start: int, pair_share: float = 1.0, **options) -> list[dict]:
    return [r for group, lab in labelled(master, start, pair_share) for r in gen.group_records(group, lab, mujoco_version=mujoco.__version__, **options)]


# -- bookkeeping options ------------------------------------------------------------------------------------------------


def test_the_script_defaults_are_the_training_set_bookkeeping():
    module = _script("mjgen_v03")
    assert (module.SET_NAME, module.SPLIT, module.ID_PREFIX) == (gen.SET_NAME, gen.SPLIT, gen.ID_PREFIX)


def test_the_set_name_split_and_id_prefix_change_only_the_bookkeeping():
    default = build(FRESH, 700_000)
    b1 = build(FRESH, 700_000, set_name="b1-infamily", split="test", id_prefix="b1")
    assert [r["id"] for r in default] == [f"v03b-{r['meta']['scene']}-{r['generator']['group']:06d}" + ("p" if r["generator"]["member"] == "pair" else "")
                                          for r in default]
    ids = {r["id"] for r in b1}
    for old, new in zip(default, b1):
        assert new["id"] == "b1" + old["id"][len("v03b"):]
        assert new["meta"]["set"] == "b1-infamily" and new["meta"]["split"] == "test"
        assert old["meta"]["set"] == "v03b-mujoco-train" and old["meta"]["split"] == "train"
        assert new["meta"]["pair"]["of"] in ids and new["meta"]["pair"]["of"] == "b1" + old["meta"]["pair"]["of"][len("v03b"):]
        assert new["request"] == old["request"] and new["labels"] == old["labels"] and new["generator"] == old["generator"]
        strip = lambda m: {k: v for k, v in m.items() if k not in ("set", "split", "pair")}  # noqa: E731
        assert strip(new["meta"]) == strip(old["meta"])
        assert {k: v for k, v in new["meta"]["pair"].items() if k != "of"} == {k: v for k, v in old["meta"]["pair"].items() if k != "of"}
        assert validate_record(new) == []


def test_assemble_writes_the_bookkeeping_into_the_records_and_the_manifest(tmp_path):
    module = _script("mjgen_v03")
    candidates = tmp_path / "cand.jsonl"
    with candidates.open("w", encoding="utf-8") as handle:
        for group, lab in labelled(FRESH, 700_000):
            handle.write(json.dumps({"group": module.group_to_json(group), "labelled": json.loads(json.dumps(lab)), "seconds": 1.0}) + "\n")
    (tmp_path / "cand.runs.jsonl").write_text(json.dumps({"groups": 3, "seed": FRESH, "start": 700_000, "groups_per_family": 1, "k": 4,
                                                          "families": list(fam.FAMILIES), "git": "test"}) + "\n", encoding="utf-8")
    out = tmp_path / "b1.jsonl"
    assert module.main(["assemble", "--candidates", str(candidates), "--records-per-family", "2", "--out", str(out), "--set-name", "b1-infamily",
                        "--split", "test", "--id-prefix", "b1", "--what", "fresh in-family evaluation set"]) == 0
    rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert rows and all(r["id"].startswith("b1-") and r["meta"]["set"] == "b1-infamily" and r["meta"]["split"] == "test" for r in rows)
    manifest = json.loads(out.with_suffix(".manifest.json").read_text(encoding="utf-8"))
    assert (manifest["set"], manifest["split"], manifest["id_prefix"], manifest["what"]) == ("b1-infamily", "test", "b1", "fresh in-family evaluation set")
    assert manifest["master_seed"] == FRESH and manifest["label_runs"][0]["start"] == 700_000
    assert set(manifest["group_indices"]) == set(fam.FAMILIES)
    assert all(lo == hi and lo >= 700_000 for lo, hi in manifest["group_indices"].values())
    assert manifest["sha256"] == file_sha256(out)


# -- zero overlap with the training records -----------------------------------------------------------------------------


def test_a_fresh_seed_shares_nothing_with_the_training_records_and_planted_copies_are_caught():
    ours = build(FRESH, 700_000, id_prefix="b1")
    train = build(MASTER, 0, pair_share=0.0)
    report = train_overlap_report(ours, train)
    assert report["verdict"]["passed"], report["verdict"]
    assert report["shared"]["scene_parameters"]["shared"] == 0 and report["shared"]["state_texts"]["shared"] == 0
    # a training copy of one of ours (same id, seeds, scene and text) is caught by every probe
    planted = train_overlap_report(ours, train + [copy.deepcopy(ours[0])])
    assert not planted["verdict"]["passed"]
    assert all(planted["shared"][k]["shared"] == 1 for k in ("group_seeds", "scene_parameters", "stated_facts", "ids", "state_texts", "requests"))
    # the same physical scene told with another robot and domain is still the same scene
    twin = copy.deepcopy(ours[0])
    twin["id"] = "v03b-elsewhere-000001"
    twin["generator"]["seed"] += 1
    twin["meta"]["visible"]["robot"] = "another robot"
    twin["meta"]["visible"]["domain"] = "another domain"
    twin["request"]["state"] += " (retold)"
    caught = train_overlap_report(ours, train + [twin])
    assert caught["shared"]["scene_parameters"]["shared"] == 1 and caught["shared"]["stated_facts"]["shared"] == 0
    assert not caught["verdict"]["passed"]
    assert {"robot", "domain"} <= NARRATIVE_FIELDS and scene_parameters(twin) == scene_parameters(ours[0])


def test_the_train_overlap_command_writes_its_report_into_the_manifest(tmp_path):
    module = _script("mjgen_v03")
    ours, train = tmp_path / "b1.jsonl", tmp_path / "train.jsonl"
    ours.write_text("".join(json.dumps(r) + "\n" for r in build(FRESH, 700_000, id_prefix="b1")), encoding="utf-8")
    train.write_text("".join(json.dumps(r) + "\n" for r in build(MASTER, 0, pair_share=0.0)), encoding="utf-8")
    ours.with_suffix(".manifest.json").write_text(json.dumps({"sha256": file_sha256(ours)}), encoding="utf-8")
    out = tmp_path / "overlap.json"
    assert module.main(["train-overlap", "--input", str(ours), "--train", str(train), "--out", str(out)]) == 0
    manifest = json.loads(ours.with_suffix(".manifest.json").read_text(encoding="utf-8"))
    assert manifest["overlap_with_train"]["verdict"]["passed"] and manifest["overlap_with_train"]["train_sha256"] == file_sha256(train)
    assert set(manifest["overlap_with_train"]["shared"].values()) == {0}
    # a stale manifest (the set changed after assembly) is refused
    ours.with_suffix(".manifest.json").write_text(json.dumps({"sha256": "0" * 64}), encoding="utf-8")
    with pytest.raises(SystemExit):
        module.main(["train-overlap", "--input", str(ours), "--train", str(train), "--out", str(out)])
