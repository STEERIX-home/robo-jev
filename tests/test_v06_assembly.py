"""v0.6 training input (Task v0.6b): v0.4's input except the MuJoCo part — A0 (v0.3b subsample) + three training authors, held at v0.4's
36,648 MuJoCo examples per epoch — and the assembly's checks.

* **Authors' files** (:func:`author_problems`): trainer validation, split, ``meta.set`` / id prefix, labels by prior, and every yes/no and
  score label equal to the base prior's (labels as authored); the held-out author A4 never enters (:func:`held_out_marks`).
* **Holding the part** (:func:`second_pass`, :func:`trim`): the second pass is the mix's own (``convert_single_probe._extra_second_passes``)
  except that a scene is keyed by ``meta.set:meta.scene``; the trim drops whole groups of the largest scene.
* **Sealed checks**: the H-set and A4 share no id, request or state with training (:func:`sealed_check`); ``system1_v03.py overlap
  --sealed-outside`` also matches A4's text units.
* **Exposure** (:func:`compare_exposure`) and the **cross-author overlap** (:func:`text_index`, :func:`pairwise_overlap`).
* The built input when present (``artifacts/simgen/v06/``), and A0 ⊂ v0.4's A0 on the real v0.3b file when present.
"""

from __future__ import annotations

import copy
import functools
import importlib.util
import json
import random
import sys
from pathlib import Path

import pytest
from helpers import REPO
from test_v02_mix import _write

from robo_jev.decisions.records import read_records

TRAIN_CONFIG = REPO / "configs" / "train" / "decisions-v0-lora-h100.yaml"
V03_MUJOCO = REPO / "artifacts" / "simgen" / "v03" / "mujoco-train.jsonl"
V04_MUJOCO = REPO / "artifacts" / "simgen" / "v04" / "mujoco-v04.jsonl"
BUILT = REPO / "artifacts" / "simgen" / "v06"


@functools.lru_cache(maxsize=None)
def _load(name: str):
    spec = importlib.util.spec_from_file_location(f"v06t_{Path(name).stem}", REPO / "scripts" / name)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def A():
    return _load("assemble_v06.py")


def _config() -> dict:
    return A()._config(TRAIN_CONFIG)


def _records(prefix: str, set_name: str, scene: str, groups: int, *, split: str = "train", seed: int = 0, factor_every: int = 3) -> list[dict]:
    """v0.3b-contract-shaped pair groups (two records each): a key yes/no, a score and a direction choice; labels by prior with the label
    = the base prior's; key labels spread over the tenths."""
    rng = random.Random(f"{prefix}|{scene}|{seed}")
    out = []
    for g in range(groups):
        a, b = f"{prefix}-{scene}-{g:06d}", f"{prefix}-{scene}-{g:06d}p"
        key = (g % 10 + rng.random()) / 10
        for rid, partner, role, value in ((a, b, "base", key), (b, a, "variant", 1 - key)):
            labels = {"k": {"type": "noul", "p_yes": round(value, 4)}, "s": {"type": "score", "value": round(value / 2, 4)},
                      "d": {"type": "choice", "kind": "state", "probabilities": {"up": 1.0, "down": 0.0}}}
            by_prior = {p: copy.deepcopy(labels) for p in ("base", "alt_a", "alt_b")}
            by_prior["alt_a"]["k"]["p_yes"] = min(1.0, round(value + 0.2, 4))
            out.append({"schema": "simgen-request/0", "id": rid,
                        "request": {"state": f"{set_name} {scene} moment {g} {role} with a long enough line of words here",
                                    "questions": {"k": {"type": "noul", "instructions": "Will it fail?"},
                                                  "s": {"type": "score", "instructions": "How likely at the other value?", "criteria": ["0", "1"]},
                                                  "d": {"type": "choice", "instructions": "Which way?", "criteria": {"up": "u", "down": "d"}}}},
                        "labels": labels,
                        "meta": {"set": set_name, "split": split, "scene": scene, "key_question": "k", "labels_by_prior": by_prior,
                                 "robustness": {"k": {"robust": value < 0.2 or value > 0.85}, "s": {"robust": True}},
                                 "pair": {"of": partner, "kind": "numeric", "role": role, "factor": "wet" if g % factor_every == 0 else None}}})
    return out


def _author(k: str, groups: int = 6, scenes=("law_a", "law_b", "law_c"), **kw) -> list[dict]:
    return [r for scene in scenes for r in _records(f"v06{k}", f"v06-{k}", scene, groups, **kw)]


def _v03(groups: int = 40) -> list[dict]:
    return [r for family in ("tilt", "grip", "tower") for r in _records("v03b", "v03b-mujoco-train", family, groups)]


# --------------------------------------------------------------------------
# authors' files
# --------------------------------------------------------------------------


def test_a_well_formed_author_file_has_no_problems_and_each_contract_break_is_named():
    a = A()
    good = _author("a1")
    assert a.author_problems(good, "a1") == []
    cases = {"split": lambda r: r["meta"].update(split="test"), "meta.set": lambda r: r["meta"].update(set="v06-a2"),
             "does not start": lambda r: r.update(id="x-" + r["id"]),
             "base prior": lambda r: r["labels"]["k"].update(p_yes=0.5 if r["labels"]["k"]["p_yes"] != 0.5 else 0.4),
             "labels_by_prior": lambda r: r["meta"]["labels_by_prior"].pop("alt_b"),
             "validation": lambda r: r["request"].update(questions={})}
    for needle, mutate in cases.items():
        broken = copy.deepcopy(good)
        mutate(broken[3])
        assert any(needle in p for p in a.author_problems(broken, "a1")), needle
    twice = good + [copy.deepcopy(good[0])]
    assert any("appears 2 times" in p for p in a.author_problems(twice, "a1"))


def test_a_clipped_label_is_refused_labels_stay_as_authored():
    a = A()
    records = _author("a2")
    clipped = copy.deepcopy(records)
    for r in clipped:
        if not r["meta"]["robustness"]["k"]["robust"]:
            r["labels"]["k"]["p_yes"] = min(0.6, max(0.4, r["labels"]["k"]["p_yes"]))  # v0.5's clip: must not appear in v0.6
    assert a.author_problems(records, "a2") == [] and any("base prior" in p for p in a.author_problems(clipped, "a2"))


def test_records_of_the_held_out_author_are_found_by_set_or_prefix():
    a = A()
    a4 = _author("a4", split="test")
    assert len(a.held_out_marks(a4)) == len(a4)
    sneaky = copy.deepcopy(_author("a1")[:2])
    sneaky[0]["id"] = "v06a4-x-000001"
    sneaky[1]["meta"]["set"] = "v06-a4"
    assert a.held_out_marks(sneaky) == [r["id"] for r in sneaky] and a.held_out_marks(_author("a3")) == []


def test_the_composition_counts_scenes_pairs_one_factor_pairs_tenths_and_robust_share():
    c = A().composition(_author("a1", groups=10, factor_every=2))
    assert c["records"] == 60 and c["groups"] == 30 and c["pairs"] == 30 and c["unpaired"] == 0
    assert all(s["records"] == 20 and s["one_factor_pairs"] == 5 and sum(s["base_key_tenths"]) == 10 for s in c["scenes"].values())
    assert c["yes_no_score_questions"] == 120 and 0 < c["robust_share"] < 1 and sum(c["bands"].values()) == 120


# --------------------------------------------------------------------------
# holding the part at v0.4's examples
# --------------------------------------------------------------------------


def test_the_second_pass_is_the_mixs_own_when_scenes_are_keyed_alike(tmp_path):
    a, mix = A(), _load("convert_single_probe.py")
    config = _config()
    records = _author("a1", groups=8) + _author("a2", groups=8, scenes=("law_x", "law_y"))
    target = A()._train_examples(records, config) + 37
    mine, report = a.second_pass(records, target=target, config=config, key=lambda r: str(r["meta"]["scene"]))
    extra = {"name": "mujoco", "records": records, "path": tmp_path / "x"}
    mix._extra_second_passes([extra], {"mujoco": a._train_examples(records, config)}, {}, general_share=0.0, config=config, q=2,
                             counts={"mujoco": target})
    assert [r["id"] for r in mine] == [r["id"] for r in extra["copies"]] and mine == extra["copies"], "same copies, same order"
    assert report["examples"] >= target and report["examples"] - target <= a.HOLD_SLACK and report["records"] == len(mine)
    assert all(r["id"].endswith("~r2") and r["meta"]["repeat_of"] == r["id"][:-3] for r in mine)
    assert all(r["meta"]["pair"]["of"].endswith("~r2") for r in mine)


def test_the_second_pass_keys_scenes_by_set_so_two_authors_scene_names_do_not_merge():
    a = A()
    config = _config()
    records = _author("a1", groups=6, scenes=("shelf",)) + _author("a2", groups=6, scenes=("shelf",)) + _author("a3", groups=6, scenes=("ramp",))
    target = a._train_examples(records, config) + 30
    copies, report = a.second_pass(records, target=target, config=config)
    assert set(report["by_scene"]) == {"v06-a1:shelf", "v06-a2:shelf", "v06-a3:ramp"}
    first_three = [a.scene_key(r) for r in copies[::2][:3]]
    assert first_three == ["v06-a1:shelf", "v06-a2:shelf", "v06-a3:ramp"], "round-robin over the three scenes, not two"
    none, report = a.second_pass(records, target=10, config=config)
    assert none == [] and report["records"] == 0
    with pytest.raises(ValueError, match="even a whole second pass"):
        a.second_pass(records, target=10 * target, config=config)


def test_an_over_full_part_is_trimmed_by_whole_groups_of_the_largest_scene():
    a = A()
    config = _config()
    records = _author("a1", groups=20, scenes=("big",)) + _author("a2", groups=8, scenes=("small",))
    have = a._train_examples(records, config)
    kept, report = a.trim(records, target=have - 20, config=config)
    assert have - 20 - a.HOLD_SLACK <= report["examples"] <= have - 20 == report["target_examples"]
    assert report["examples"] == a._train_examples(kept, config)
    assert set(report["by_scene"]) == {"v06-a1:big"} and report["dropped_records"] % 2 == 0
    groups = {r["id"].rstrip("p") for r in kept}
    assert all((g + "p") in {r["id"] for r in kept} for g in groups), "whole pairs only"
    same, report = a.trim(records, target=have, config=config)
    assert same == records and report["dropped_records"] == 0


# --------------------------------------------------------------------------
# the MuJoCo part
# --------------------------------------------------------------------------


def test_the_mujoco_part_is_a0_then_the_authors_held_at_the_target_and_refuses_what_must_never_train():
    a = A()
    config = _config()
    v03 = _v03()
    authors = {k: _author(k) for k in ("a1", "a2", "a3")}
    hset = _records("v04h", "v04a-scene-holdout", "tray", 4, split="test")
    a4 = _author("a4", groups=3, split="test")
    first, copies, report = a.build_mujoco(v03, authors, sealed={"H-set": hset, "A4": a4}, config=config, per_family=20, seed=1, target=400)
    a0_ids = [r["id"] for r in _load("assemble_v04.py").subsample(v03, per_family=20, seed=1, families=a.A0_FAMILIES)[0]]
    assert [r["id"] for r in first[:60]] == a0_ids and [r["id"] for r in first[60:]] == [r["id"] for k in ("a1", "a2", "a3") for r in authors[k]]
    assert 400 <= report["held"]["examples"] <= 400 + a.HOLD_SLACK and len(copies) == report["held"]["records"] > 0 and report["sealed"]["A4"]["pass"] is True
    assert report["authors"]["a2"] == {"records_in_file": 36, "records_used": 36, "set": "v06-a2"}
    with pytest.raises(ValueError, match="held-out author"):
        a.build_mujoco(v03, {**authors, "a3": authors["a3"] + [dict(copy.deepcopy(a4[0]), id="v06a3-x-1")]}, sealed={}, config=config, per_family=20, seed=1)
    leaked = copy.deepcopy(authors["a1"][0])  # an A4 record that is a training record's whole request
    leaked["id"], leaked["meta"]["split"] = "v06a4-leak-000001", "test"
    with pytest.raises(ValueError, match="sealed set"):
        a.build_mujoco(v03, authors, sealed={"A4": [leaked]}, config=config, per_family=20, seed=1)
    broken = {**authors, "a1": [dict(r, meta={**r["meta"], "split": "test"}) for r in authors["a1"]]}
    with pytest.raises(ValueError, match="problems in the authors"):
        a.build_mujoco(v03, broken, sealed={}, config=config, per_family=20, seed=1)
    big = {k: _author(k, groups=40) for k in ("a1", "a2", "a3")}
    first, copies, report = a.build_mujoco(v03, big, sealed={}, config=config, per_family=20, seed=1, target=500)
    assert copies == [] and report["held"]["examples"] <= 500 and report["held"]["dropped_records"] > 0


def test_sealed_check_names_shared_ids_requests_and_states():
    a = A()
    train = _author("a1", groups=3)
    held = copy.deepcopy(_author("a4", groups=2, split="test"))
    assert a.sealed_check(train, held)["pass"] is True
    same = dict(copy.deepcopy(train[0]), id="v06a4-z-000001")
    same["meta"]["split"] = "test"
    result = a.sealed_check(train, [same])
    assert result["pass"] is False and result["shared"]["requests"] == 1 and result["shared"]["states"] == 1
    assert a.sealed_check(train, [dict(copy.deepcopy(held[0]), id=train[0]["id"])])["shared"]["ids"] == 1
    assert a.sealed_check(train, [dict(copy.deepcopy(held[0]), meta={**held[0]["meta"], "split": "train"})])["pass"] is False


def test_the_overlap_check_matches_a_sealed_outside_sets_text_units(tmp_path, monkeypatch):
    script = _load("system1_v03.py")
    monkeypatch.setattr(script, "SEALED", {})
    records = _author("a1", groups=3)
    for r in records:
        r["request"]["questions"]["d"]["criteria"] = {"up": f"the load moves further toward the edge {r['id']}", "down": "d"}
    held = copy.deepcopy(_author("a4", groups=1, split="test"))
    held[0]["request"]["questions"]["d"]["criteria"] = {"up": records[2]["request"]["questions"]["d"]["criteria"]["up"], "down": "d"}  # one option text
    train, a4 = _write(tmp_path / "train.jsonl", records), _write(tmp_path / "a4.jsonl", held)
    common = ["overlap", "--input", str(train), "--mujoco-set", "v06-a1"]
    assert script.main([*common, "--out", str(tmp_path / "x.json"), "--sealed-extra", f"A4={a4}"]) == 0, "whole texts only: no hit"
    assert script.main([*common, "--out", str(tmp_path / "y.json"), "--sealed-outside", f"A4={a4}"]) == 2, "a shared option text is a hit"
    report = json.loads((tmp_path / "y.json").read_text(encoding="utf-8"))
    assert report["sealed_outside"] == ["A4"] and report["parts"]["mujoco"]["A4:units"] == 1
    with pytest.raises(SystemExit, match="new name"):
        script.main([*common, "--out", str(tmp_path / "z.json"), "--sealed-outside", f"A4={a4}", "--sealed-extra", f"A4={a4}"])


# --------------------------------------------------------------------------
# exposure, cross-author overlap, writes
# --------------------------------------------------------------------------


def _seed_view(**changes) -> dict:
    view = {"examples_per_epoch": {"engine": 116601, "mujoco": 36648, "d1": 2724, "general": 45808, "total": 201781}, "steps": 6306,
            "action": 21297, "derived": 18551, "band": 15356, "comparison": 3195, "engine_by_category": {"choice:action": 21297}}
    for key, value in changes.items():
        if key in view["examples_per_epoch"]:
            view["examples_per_epoch"] = {**view["examples_per_epoch"], key: value}
        else:
            view[key] = value
    return view


def test_the_exposure_check_wants_v04s_engine_d1_general_and_the_mujoco_count_held():
    a = A()
    assert a.compare_exposure(_seed_view(), _seed_view())["pass"] is True
    assert a.compare_exposure(_seed_view(mujoco=36651), _seed_view())["pass"] is True, "within one pair group's slack"
    assert a.compare_exposure(_seed_view(mujoco=36652), _seed_view())["checks"]["mujoco_held"] is False
    for change in (dict(engine=116600), dict(action=21296), dict(engine_by_category={"choice:action": 1}), dict(general=45807)):
        assert a.compare_exposure(_seed_view(**change), _seed_view())["pass"] is False, change
    assert a.compare_exposure(_seed_view(mujoco=36650, total=201783, steps=6307), _seed_view())["difference"] == {"total": 2, "mujoco": 2, "steps": 1}


def test_the_layout_check_wants_v04s_file_with_only_the_mujoco_block_swapped(tmp_path):
    a = A()
    engine, m4, m6, tail = b'{"e": 1}\n{"e": 2}\n', b'{"m": "v04"}\n' * 3, b'{"m": "v06"}\n' * 2, b'{"d1": 1}\n{"g": 1}\n'
    files = {name: tmp_path / f"{name}.jsonl" for name in ("engine", "mujoco", "v04", "v06")}
    for name, data in (("engine", engine), ("mujoco", m6), ("v04", engine + m4 + tail), ("v06", engine + m6 + tail)):
        files[name].write_bytes(data)
    good = a.layout_check(files["v06"], files["engine"], files["mujoco"], files["v04"])
    assert good["pass"] is True and good["bytes"] == {"engine": len(engine), "mujoco_v06": len(m6), "tail": len(tail), "mujoco_v04_block": len(m4)}
    for name, data, broken in (("tail", engine + m6 + b'{"g": 2}\n{"g": 1}\n', "tail_is_v04s"), ("block", engine + m6[:-1] + b" \n" + tail, "mujoco_block"),
                               ("head", b'{"e": 9}\n{"e": 2}\n' + m6 + tail, "engine_prefix")):
        path = tmp_path / f"bad-{name}.jsonl"
        path.write_bytes(data)
        result = a.layout_check(path, files["engine"], files["mujoco"], files["v04"])
        assert result["pass"] is False and result["checks"][broken] is False, name


def test_the_cross_author_overlap_counts_shared_texts_and_5grams_without_text():
    a = A()
    one = _author("a1", groups=2, scenes=("s",))
    two = copy.deepcopy(one)
    for r in two:  # another author re-using A1's whole state as one sentence of its own
        r["id"] = r["id"].replace("v06a1", "v06a2")
        r["request"]["state"] = "Radio check, unit two. " + r["request"]["state"] + "."
    three = _author("a3", groups=2, scenes=("t",))
    for r in three:  # nothing in common with A1: own states, own instructions
        r["request"]["state"] = f"Work order {r['id']}: lift the crate onto the cart slowly"
        for q in r["request"]["questions"].values():
            q["instructions"] = "Give your own estimate for this order."
    idx = {name: a.text_index(rs) for name, rs in (("A1", one), ("A2", two), ("A3", three))}
    pairs = a.pairwise_overlap(idx)
    assert pairs["A1~A2"]["ids"] == 0 and pairs["A1~A2"]["states"] == 0 and pairs["A1~A2"]["requests"] == 0
    assert pairs["A1~A2"]["sentences"] >= len(one), "every A1 state is a sentence of A2's"
    assert pairs["A1~A3"]["sentences"] == 0 and pairs["A1~A3"]["units"] == 0
    assert 0 < pairs["A1~A2"]["grams_share_of_A1"] <= 1 and pairs["A1~A2"]["grams_jaccard"] <= min(pairs["A1~A2"]["grams_share_of_A1"], pairs["A1~A2"]["grams_share_of_A2"])
    assert pairs["A1~A3"]["grams_shared"] < pairs["A1~A2"]["grams_shared"]
    same = a.pairwise_overlap({"X": idx["A1"], "Y": idx["A1"]})["X~Y"]
    assert same["grams_jaccard"] == 1.0 and same["ids"] == len(one)
    assert all(isinstance(h, bytes) for h in idx["A1"]["grams"]), "hashes only"


def test_outputs_never_go_to_v04_v05_the_sealed_sets_or_the_authors_folders(tmp_path):
    a = A()
    for place in (REPO / "artifacts" / "simgen" / "v04" / "x.jsonl", REPO / "artifacts" / "simgen" / "v05" / "x", REPO / "artifacts" / "v06" / "a1" / "x",
                  REPO / "artifacts" / "reports" / "v05" / "x", REPO / "artifacts" / "t9" / "x", REPO / "artifacts" / "b1" / "x"):
        with pytest.raises(SystemExit, match="refusing"):
            a.refuse_write(place)
    a.refuse_write(REPO / "artifacts" / "simgen" / "v06" / "train-v06.jsonl")
    a.refuse_write(tmp_path / "x")


def test_the_stand_in_is_written_only_under_scratch_and_refused_for_a_final_build(tmp_path):
    a = A()
    with pytest.raises(SystemExit, match="only under artifacts/scratch"):
        a.main(["standin", "--out-dir", str(tmp_path / "standin")])
    with pytest.raises(SystemExit, match="scratch stand-in"):
        a.main(["all", "--author", f"a1={REPO / 'artifacts' / 'scratch' / 'v06b' / 'standin' / 'a1' / 'train.jsonl'}", "--author", "a2=x", "--author", "a3=y",
                "--out-dir", str(tmp_path / "final")])
    with pytest.raises(SystemExit, match="missing"):
        a.main(["all", "--author", "a1=x", "--out-dir", str(tmp_path / "final")])


# --------------------------------------------------------------------------
# real files, when present
# --------------------------------------------------------------------------


def test_a0_is_the_head_of_v04s_own_a0_on_the_real_v03b_file():
    if not (V03_MUJOCO.is_file() and V04_MUJOCO.is_file()):
        pytest.skip("v0.3b's MuJoCo file or v0.4's MuJoCo part not on this box")
    a = A()
    v03 = read_records(V03_MUJOCO)
    a0, selection = _load("assemble_v04.py").subsample(v03, per_family=a.A0_PER_FAMILY, seed=a.A0_SEED, families=a.A0_FAMILIES)
    v04_a0 = {r["id"] for r in read_records(V04_MUJOCO) if r["meta"]["set"] == a.A0_SET}
    assert {r["id"] for r in a0} <= v04_a0 and len(a0) in range(3745, 3751)
    assert all(e["groups_kept"] == 625 and e["base_key_tenths_kept"] == [63] * 5 + [62] * 5 for e in selection.values())


def test_the_built_v06_input_when_present():
    summary = BUILT / "train-v06.assembly.json"
    if not summary.is_file():
        pytest.skip("artifacts/simgen/v06 not assembled yet (scripts/assemble_v06.py all, after the three authors land)")
    built = json.loads(summary.read_text(encoding="utf-8"))
    assert built["pass"] is True and built["standin"] is False and built["overlap_hits"] == 0 and built["training_vs_A4_clean"] is True
    assert {p: v for p, v in built["parts_identical_to_v04"].items() if p != "mujoco"} == {"engine": True, "d1": True, "general": True}
    assert built["a0"]["within_v04_subsample"]["a0_not_in_v04"] == 0
    assert set(built["sealed_checked"]) >= {"T1", "T2", "T3", "T4", "T5", "T6", "T7", "G1", "G2", "T8a", "T8b", "T9", "B1", "H-set", "A4"}
