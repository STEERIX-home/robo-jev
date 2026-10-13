"""v0.5 training input (Task v0.5a) — ``scripts/assemble_v05.py``: v0.4's input with one label change.

* **The rule** (:func:`clarity_value`, :func:`question_plan`, :func:`transform_record`): a MuJoCo yes/no or score question whose band is
  not the same under the record's three priors gets clip(mean over the priors, 0.40, 0.60) in ``p_yes`` / ``value``; robust questions,
  choice questions and every other field stay. Robustness is read from ``meta.robustness`` and must agree with the bands re-derived from
  ``meta.labels_by_prior`` (built here with the generator's own :func:`robo_jev.mjgen.generate._robust`).
* **The file** (:func:`transform_file`): non-MuJoCo lines and untouched MuJoCo lines byte for byte, changed lines differ only in the
  changed numbers (same JSON style), order and ids kept, second-pass repeats transformed like their originals, band counts and digests
  reported; inconsistent inputs are refused and leave no output.
* **Real files** (when present): a slice of v0.4's MuJoCo part and scenes, and the built v0.5 input's reports (parts identical, exposure
  identical to v0.4's per seed, 0 sealed hits).
"""

from __future__ import annotations

import functools
import importlib.util
import json
import sys
from fractions import Fraction
from itertools import islice
from pathlib import Path

import pytest
from helpers import REPO

from robo_jev.decisions.records import file_sha256
from robo_jev.mjgen.generate import _robust

PRIORS = ("base", "alt_a", "alt_b")
V04 = REPO / "artifacts" / "simgen" / "v04"
V05 = REPO / "artifacts" / "simgen" / "v05"
TRAIN_CONFIG = REPO / "configs" / "train" / "decisions-v0-lora-h100.yaml"


@functools.lru_cache(maxsize=None)
def A():
    spec = importlib.util.spec_from_file_location("v05_assemble", REPO / "scripts" / "assemble_v05.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _mj(rid: str, scalars: dict[str, tuple[str, tuple[float, float, float]]], *, set_name: str = "v03b-mujoco-train",
        repeat_of: str | None = None, partner: str | None = None) -> dict:
    """A MuJoCo-shaped record: `scalars` = {question: (type noul|score, (base, alt_a, alt_b))} plus one choice question; labels = the base
    prior's; ``meta.robustness`` from the generator's own rule."""
    generator = {"v03b-mujoco-train": "v03b-mujoco-train-v1", "v04a-mujoco-scenes": "v04a-mujoco-scenes-v1"}[set_name]
    questions, by_prior = {}, {p: {} for p in PRIORS}
    for name, (qtype, values) in scalars.items():
        questions[name] = {"type": qtype, "instructions": f"Does {name} happen?"} | ({"criteria": ["0 never", "1 always"]} if qtype == "score" else {})
        key = "p_yes" if qtype == "noul" else "value"
        for prior, value in zip(PRIORS, values):
            by_prior[prior][name] = {"type": qtype, key: value}
    questions["which_way"] = {"type": "choice", "instructions": "Which way?", "criteria": {"more_likely": "m", "less_likely": "l", "about_same": "s"}}
    for prior, top in zip(PRIORS, ("more_likely", "about_same", "less_likely")):
        by_prior[prior]["which_way"] = {"type": "choice", "kind": "state", "probabilities": {k: float(k == top) for k in questions["which_way"]["criteria"]}}
    robustness = {name: _robust(questions[name], {p: by_prior[p][name] for p in PRIORS}) for name in questions}
    meta = {"set": set_name, "split": "train", "scene": "tilt", "labels_by_prior": by_prior, "robustness": robustness,
            "pair": {"of": partner, "kind": "numeric", "role": "base"} if partner else None}
    if repeat_of:
        meta["repeat_of"] = repeat_of
    return {"schema": "simgen-request/0", "id": rid, "generator": {"name": generator, "priors": list(PRIORS), "k": 32},
            "meta": meta, "request": {"state": f"moment {rid}", "questions": questions}, "labels": json.loads(json.dumps(by_prior["base"]))}


SCALARS = {"key": ("noul", (0.125, 1.0, 0.0)),         # low / high / low → prior-dependent, mean 0.375 → 0.40
           "severe": ("noul", (0.0, 0.03125, 0.0)),     # robust low → unchanged
           "cf": ("score", (0.5625, 1.0, 0.9375)),      # middle / high / high → prior-dependent, mean 0.833 → 0.60
           "tip": ("noul", (0.5, 0.4375, 0.53125)),     # robust middle → unchanged
           "slip": ("score", (0.3125, 0.375, 0.5))}     # low / middle / middle → mean 0.396 → 0.40


# --------------------------------------------------------------------------
# the rule
# --------------------------------------------------------------------------


@pytest.mark.parametrize("values, expected", [
    ((0.0, 0.0, 0.0), 0.40), ((1.0, 1.0, 1.0), 0.60), ((0.125, 1.0, 0.0), 0.40), ((0.5625, 1.0, 0.9375), 0.60),
    ((0.40, 0.40, 0.40), 0.40), ((0.60, 0.60, 0.60), 0.60), ((0.45, 0.45, 0.45), 0.45),
    ((0.3, 0.5, 0.6), float((Fraction(0.3) + Fraction(0.5) + Fraction(0.6)) / 3)),
    ((0.125, 0.9375, 0.40625), (0.125 + 0.9375 + 0.40625) / 3),
])  # fmt: skip
def test_the_rule_is_the_clipped_mean_over_the_three_priors(values, expected):
    value = A().clarity_value(values)
    assert value == expected and 0.40 <= value <= 0.60, "the exact mean, rounded once (three equal values give that value)"
    assert A().band(value) == "middle", "the clip band lies inside the middle band: every prior-dependent question becomes 'look again'"
    with pytest.raises(ValueError, match="three|3 priors"):
        A().clarity_value(values[:2])


def test_only_prior_dependent_yes_no_and_score_labels_change():
    record = _mj("v03b-tilt-000001", SCALARS)
    before = json.loads(json.dumps(record))
    new, plan = A().transform_record(record)
    assert record == before, "the input is not modified"
    by_name = {e["question"]: e for e in plan}
    assert set(by_name) == set(SCALARS), "choice questions are not in the plan"
    assert [n for n, e in by_name.items() if not e["robust"]] == ["key", "cf", "slip"]
    assert new["labels"]["key"] == {"type": "noul", "p_yes": 0.40} and new["labels"]["cf"] == {"type": "score", "value": 0.60}
    assert new["labels"]["slip"]["value"] == 0.40 and new["labels"]["severe"] == before["labels"]["severe"] and new["labels"]["tip"] == before["labels"]["tip"]
    assert new["labels"]["which_way"] == before["labels"]["which_way"]
    back = json.loads(json.dumps(new))
    for name in ("key", "cf", "slip"):
        back["labels"][name] = before["labels"][name]
    assert back == before, "nothing but the three label values moved (meta, labels_by_prior, robustness, request untouched)"


def test_a_record_without_prior_dependent_questions_is_left_alone():
    record = _mj("v04a-forks-000002", {"key": ("noul", (1.0, 1.0, 0.96875)), "cf": ("score", (0.0, 0.0625, 0.0))}, set_name="v04a-mujoco-scenes")
    new, plan = A().transform_record(record)
    assert new is None and all(e["robust"] and e["new"] is None for e in plan)
    already = _mj("v03b-tilt-000003", {"key": ("noul", (0.5, 0.25, 0.75))})  # prior-dependent, mean 0.5 = the label already
    new, plan = A().transform_record(already)
    assert new is None and plan[0]["robust"] is False and plan[0]["new"] == 0.5 == plan[0]["old"]


def test_inconsistent_mujoco_records_are_refused():
    record = _mj("v03b-tilt-000004", SCALARS)
    bad = json.loads(json.dumps(record))
    bad["meta"]["robustness"]["key"]["robust"] = True
    with pytest.raises(ValueError, match="robust"):
        A().question_plan(bad)
    off = json.loads(json.dumps(record))
    off["labels"]["key"]["p_yes"] = 0.2
    with pytest.raises(ValueError, match="base prior"):
        A().question_plan(off)
    two = json.loads(json.dumps(record))
    del two["meta"]["labels_by_prior"]["alt_b"]
    with pytest.raises(ValueError, match="labels_by_prior"):
        A().question_plan(two)
    gen = json.loads(json.dumps(record))
    gen["generator"]["name"] = "v04a-mujoco-scenes-v1"
    with pytest.raises(ValueError, match="generator"):
        A().mujoco_kind(gen)
    stray = json.loads(json.dumps(record))
    stray["meta"]["set"] = "T9"
    with pytest.raises(ValueError, match="outside its MuJoCo set"):
        A().mujoco_kind(stray)
    loose = json.loads(json.dumps(record))
    loose["meta"]["set"], loose["generator"] = "something-else", {"name": "x"}
    with pytest.raises(ValueError, match="labels by prior outside"):
        A().mujoco_kind(loose)


# --------------------------------------------------------------------------
# the file
# --------------------------------------------------------------------------


def _engine(i: int) -> dict:
    return {"schema": "simgen-request/0", "id": f"v0b-train-{i:06d}", "generator": {"name": "simgen-v0b", "seed": 1000000 + i},
            "meta": {"split": "train", "basis": {"yn": {"kind": "noul", "source": "event"}},
                     "question_pick": {"version": "qpick-1", "secondary": []}},
            "request": {"state": f"engine moment {i}", "questions": {"yn": {"type": "noul", "instructions": "Will it hit?"}}},
            "labels": {"yn": {"type": "noul", "p_yes": 0.125}}}


def _general(i: int) -> dict:
    return {"schema": "simgen-request/0", "id": f"gen-csqa-{i:06d}", "meta": {"split": "train", "source": {"set": "commonsense_qa", "id": str(i)}},
            "request": {"state": "Where do you keep a cup?", "questions": {"q": {"type": "choice", "instructions": "Pick one.", "criteria": {"a": "shelf", "b": "sea"}}}},
            "labels": {"q": {"type": "choice", "kind": "probe", "probabilities": {"a": 1.0, "b": 0.0}}}}


def _training_file(tmp_path: Path) -> tuple[Path, list[bytes]]:
    from robo_jev.simgen.record import dumps

    first = [_mj("v03b-tilt-000010", SCALARS, partner="v03b-tilt-000010p"),
             _mj("v03b-tilt-000010p", {**SCALARS, "key": ("noul", (0.375, 1.0, 0.0625))}, partner="v03b-tilt-000010"),
             _mj("v04a-forks-000000", {"key": ("noul", (1.0, 1.0, 1.0))}, set_name="v04a-mujoco-scenes"),
             _mj("v04a-forks-000001", {"key": ("noul", (0.0, 0.875, 0.0)), "cf": ("score", (0.96875, 0.21875, 1.0))}, set_name="v04a-mujoco-scenes")]
    repeats = []
    for record in first[:2]:
        copy = json.loads(json.dumps(record))
        copy["id"] = f"{record['id']}~r2"
        copy["meta"]["repeat_of"] = record["id"]
        copy["meta"]["pair"]["of"] = f"{record['meta']['pair']['of']}~r2"
        repeats.append(copy)
    lines = [(dumps(_engine(i)) + "\n").encode("utf-8") for i in range(3)]  # the engine's compact style
    lines += [(json.dumps(r, ensure_ascii=False) + "\n").encode("utf-8") for r in first + repeats]  # the generators' / mix's default style
    lines += [(json.dumps(_general(i), ensure_ascii=False) + "\n").encode("utf-8") for i in range(2)]
    path = tmp_path / "train-v04.jsonl"
    path.write_bytes(b"".join(lines))
    return path, lines


def test_the_file_transform_keeps_everything_but_the_rules_labels_byte_for_byte(tmp_path):
    source, lines = _training_file(tmp_path)
    out = tmp_path / "v05" / "train-v05.jsonl"
    report = A().transform_file(source, out)
    new = out.read_bytes().splitlines(keepends=True)
    assert len(new) == len(lines)
    old_ids = [json.loads(x)["id"] for x in lines]
    assert [json.loads(x)["id"] for x in new] == old_ids, "order and ids kept"
    changed = [i for i, (a, b) in enumerate(zip(lines, new)) if a != b]
    assert [old_ids[i] for i in changed] == ["v03b-tilt-000010", "v03b-tilt-000010p", "v04a-forks-000001", "v03b-tilt-000010~r2", "v03b-tilt-000010p~r2"]
    for i in changed:
        a, b = json.loads(lines[i]), json.loads(new[i])
        assert new[i] == (json.dumps(b, ensure_ascii=False) + "\n").encode("utf-8"), "same JSON style as the input line"
        for name, label in b["labels"].items():
            if label != a["labels"][name]:
                assert not a["meta"]["robustness"][name]["robust"] and label["type"] in ("noul", "score")
                b["labels"][name] = a["labels"][name]
        assert b == a
    assert report["parts"]["engine"]["identical"] and report["parts"]["general"]["identical"] and not report["parts"]["mujoco"]["identical"]
    m = report["mujoco"]
    assert m["records"] == 6 and m["repeats"] == 2 and m["records_changed"] == 5 and m["records_unchanged"] == 1
    # first pass: tilt-000010 key, cf, slip; -000010p key, cf, slip; forks-000001 key, cf → 8 prior-dependent; the two repeats: 6 more
    assert m["questions"]["prior_dependent"] == 14 and m["questions"]["labels_changed"] == 14
    assert m["bands_after"]["prior_dependent"] == {"low": 0, "middle": 14, "high": 0}
    assert m["bands_before"]["robust"] == m["bands_after"]["robust"]
    assert sum(sum(row.values()) for row in m["prior_dependent_transition"].values()) == 14
    assert report["digests"]["mujoco_untouched_lines"]["identical"] and report["digests"]["mujoco_untouched_lines"]["records"] == 1
    assert report["digests"]["mujoco_masked"]["identical"] and report["digests"]["id_order"]["identical"]
    repeat = json.loads(new[old_ids.index("v03b-tilt-000010~r2")])
    original = json.loads(new[old_ids.index("v03b-tilt-000010")])
    assert repeat["labels"] == original["labels"], "a repeat is transformed exactly like its original"


def test_a_repeat_that_disagrees_with_its_original_or_an_unknown_record_is_refused_and_leaves_nothing(tmp_path):
    source, lines = _training_file(tmp_path)
    records = [json.loads(x) for x in lines]
    bent = [json.loads(json.dumps(r)) for r in records]
    i = next(k for k, r in enumerate(bent) if r["id"] == "v03b-tilt-000010~r2")
    for prior in PRIORS:  # the repeat's priors now differ from its original's: its transformed label would too
        bent[i]["meta"]["labels_by_prior"][prior]["key"]["p_yes"] = {"base": 0.125, "alt_a": 0.875, "alt_b": 0.875}[prior]
    bent[i]["labels"]["key"]["p_yes"] = 0.125
    bent[i]["meta"]["robustness"]["key"] = _robust({"type": "noul"}, {p: bent[i]["meta"]["labels_by_prior"][p]["key"] for p in PRIORS})
    path = tmp_path / "bent.jsonl"
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in bent), encoding="utf-8")
    out = tmp_path / "out" / "train-v05.jsonl"
    with pytest.raises(ValueError, match="repeats"):
        A().transform_file(path, out)
    assert not out.exists() and not out.with_name(out.name + ".tmp").exists()
    odd = tmp_path / "odd.jsonl"
    odd.write_text(json.dumps({"schema": "simgen-request/0", "id": "x-1", "meta": {}, "request": {"state": "s", "questions": {}}, "labels": {}}) + "\n")
    with pytest.raises(ValueError, match="no known part"):
        A().transform_file(odd, out)


def test_line_styles_are_detected_and_an_unknown_style_is_refused():
    record = _mj("v03b-tilt-000020", SCALARS)
    assert A().line_style((json.dumps(record, ensure_ascii=False) + "\n").encode(), record) == "default"
    assert A().line_style((json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n").encode(), record) == "compact"
    with pytest.raises(ValueError, match="style"):
        A().line_style((json.dumps(record, indent=1) + "\n").encode(), record)


def test_outputs_never_go_to_v04_or_the_sealed_places(tmp_path):
    for place in (V04 / "x.jsonl", REPO / "artifacts" / "t9" / "x.jsonl", REPO / "artifacts" / "reports" / "v04" / "x.json",
                  REPO / "artifacts" / "simgen" / "v03" / "x.jsonl", REPO / "artifacts" / "b1" / "x.jsonl"):
        with pytest.raises(SystemExit, match="refusing"):
            A().refuse_write(place)
    A().refuse_write(tmp_path / "train-v05.jsonl")
    A().refuse_write(V05 / "train-v05.jsonl")


def test_the_dose_counts_trained_examples_whose_label_changed_on_the_transformed_file(tmp_path):
    source, lines = _training_file(tmp_path)
    out = tmp_path / "train-v05.jsonl"
    A().transform_file(source, out)
    sets = {"v03b-mujoco-train", "v04a-mujoco-scenes"}
    before, after = [json.loads(x) for x in lines], [json.loads(x) for x in out.read_bytes().splitlines()]
    plans = A().rule_plans(before, mujoco_sets=sets)
    config = A()._v04()._config(TRAIN_CONFIG)
    dose = A().dose(after, plans, config=config, seeds=[0, 1], mujoco_sets=sets)
    for seed in ("0", "1"):
        entry = dose[seed]
        assert entry["mujoco_examples"] == sum(v for k, v in entry.items() if k.startswith("examples:")) > 0
        assert 0 <= entry["label_changed"] <= entry.get("prior_dependent", 0) <= entry["mujoco_examples"]
    with pytest.raises(ValueError, match="rule's"):
        A().dose(before, plans, config=config, seeds=[0], mujoco_sets=sets)  # the untransformed file does not hold the rule's labels


# --------------------------------------------------------------------------
# real files
# --------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["mujoco-v04.jsonl", "mujoco-scenes.jsonl"])
def test_on_a_slice_of_v04s_mujoco_records_the_rule_is_consistent(name):
    path = V04 / name
    if not path.is_file():
        pytest.skip(f"{path} not present")
    with path.open("rb") as handle:
        lines = list(islice(handle, 400))
    moved = robust = 0
    for raw in lines:
        record = json.loads(raw)
        assert A().mujoco_kind(record) in ("v03b-mujoco-train", "v04a-mujoco-scenes")
        assert A().line_style(raw, record) == "default"
        new, plan = A().transform_record(record)
        for entry in plan:
            robust += entry["robust"]
            if not entry["robust"]:
                assert 0.40 <= entry["new"] <= 0.60 and entry["new"] == A().clarity_value(list(entry["priors"].values()))
                moved += entry["new"] != entry["old"]
        if new is not None:
            back = json.loads(json.dumps(new))
            for entry in plan:
                back["labels"][entry["question"]] = record["labels"][entry["question"]]
            assert back == record
    assert moved > 50 and robust > 50


def test_the_built_v05_input_when_present():
    summary_path = V05 / "train-v05.assembly.json"
    if not summary_path.is_file():
        pytest.skip("artifacts/simgen/v05/train-v05.jsonl not assembled yet (scripts/assemble_v05.py all)")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["pass"] is True and summary["overlap"] == 0 and summary["exposure_identical_to_v04"] is True
    assert summary["files"]["train-v05.jsonl"]["sha256"] == file_sha256(V05 / "train-v05.jsonl")
    assert summary["files"]["input"]["sha256"] == A().TRAIN_V04_SHA
    assert set(summary["sealed_checked"]) >= {"T1", "T2", "T3", "T4", "T5", "T6", "T7", "G1", "G2", "T8a", "T8b", "T9", "B1", "H-set"}
    changes = json.loads((V05 / "train-v05.changes.json").read_text(encoding="utf-8"))
    assert all(e["identical"] for p, e in changes["parts"].items() if p != "mujoco") and set(changes["parts"]) == {"engine", "mujoco", "d1", "general"}
    m = changes["mujoco"]
    assert m["bands_after"]["prior_dependent"]["low"] == 0 and m["bands_after"]["prior_dependent"]["high"] == 0
    assert m["bands_before"]["robust"] == m["bands_after"]["robust"] and m["records"] == 18696 and m["repeats"] == 3700
    exposure = json.loads((V05 / "train-v05.exposure.json").read_text(encoding="utf-8"))
    for seed, steps, action in (("0", 6306, 21297), ("1", 6308, 21272)):
        assert exposure["v05"]["seeds"][seed]["steps"] == steps and exposure["v05"]["seeds"][seed]["action"] == action
        assert all(exposure["identical"][seed]["vs_v04_same_code"].values()) and all(exposure["identical"][seed]["vs_v04_stored"].values())
