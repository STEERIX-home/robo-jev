"""System-1 derived questions for engine training records (Task v0.3c §1-1, :mod:`robo_jev.simgen.system1_questions`).

Hand-checked label rules (soft bands and comparisons), and on real training records generated here with the v0.2 path (phrasing v01b,
fact shuffle 0.5, training split): the originals stay byte-identical, every derived label follows the rule from the source label, the
pairs are exactly bench.system1's comparison pairs, the 12-question budget keeps the key question's band first, both validators and the
trainer's example builder accept the records, the transform is deterministic, adds no held-out (T2) wording, and refuses sealed records.
The built file ``artifacts/simgen/v03/engine-s1.jsonl`` is checked against its manifest when it exists.
"""

from __future__ import annotations

import copy
import functools
import importlib.util
import json
import sys

import pytest
from helpers import REPO

from robo_jev.bench.system1 import basis_event_pairs
from robo_jev.decisions.records import validate_record as trainer_validate
from robo_jev.simgen import system1_questions as s1
from robo_jev.simgen.record import dumps
from robo_jev.simgen.record import validate_record as engine_validate

V03 = REPO / "artifacts" / "simgen" / "v03"
V02 = REPO / "artifacts" / "simgen" / "v02"


@functools.lru_cache(maxsize=1)
def train_records() -> tuple[dict, ...]:
    """Real training records through the v0.2 generation path (60 units — some pairs, both languages)."""
    from robo_jev.simgen.catalog import TRAIN_GROUPS, default_config
    from robo_jev.simgen.generate import HeldOutError, generate_unit, plan_units
    from robo_jev.simgen.holdout import load_holdout, split_for

    cfg = default_config()
    split = split_for("train")
    out = []
    for unit in plan_units(60, 0, groups=TRAIN_GROUPS, seed_start=load_holdout().seed_ranges["train"][0]):
        try:
            out += generate_unit(cfg, unit, split=split, phrasing="v01b", fact_shuffle=0.5)
        except HeldOutError:
            continue
    assert any(r["meta"]["language"] == "ko" for r in out) and len(out) >= 50
    return tuple(out)


@functools.lru_cache(maxsize=1)
def transformed() -> tuple[dict, ...]:
    return tuple(s1.transform_record(r) for r in train_records())


def _p(record: dict, name: str) -> float:
    label = record["labels"][name]
    return float(label.get("p_yes", label.get("value")))


# --------------------------------------------------------------------------
# label rules
# --------------------------------------------------------------------------


@pytest.mark.parametrize("p, expected", [
    (0.0, {"low": 1.0, "medium": 0.0, "high": 0.0}), (0.28, {"low": 1.0, "medium": 0.0, "high": 0.0}),
    (1 / 3, {"low": 0.5, "medium": 0.5, "high": 0.0}), (1 / 3 + 0.025, {"low": 0.25, "medium": 0.75, "high": 0.0}),
    (0.39, {"low": 0.0, "medium": 1.0, "high": 0.0}), (0.5, {"low": 0.0, "medium": 1.0, "high": 0.0}),
    (2 / 3 - 0.025, {"low": 0.0, "medium": 0.75, "high": 0.25}), (2 / 3, {"low": 0.0, "medium": 0.5, "high": 0.5}),
    (0.72, {"low": 0.0, "medium": 0.0, "high": 1.0}), (1.0, {"low": 0.0, "medium": 0.0, "high": 1.0}),
])  # fmt: skip
def test_band_labels_are_one_hot_away_from_the_edges_and_a_linear_ramp_within_005(p, expected):
    label = s1.band_label(p)
    assert label == pytest.approx(expected, abs=1e-6) and abs(sum(label.values()) - 1.0) < 1e-9
    assert list(label) == list(s1.BAND_KEYS)


def test_band_labels_are_continuous_monotone_and_reject_non_probabilities():
    grid = [i / 2000 for i in range(2001)]
    labels = [s1.band_label(p) for p in grid]
    for a, b in zip(labels, labels[1:]):
        assert all(abs(a[k] - b[k]) <= 0.0101 for k in s1.BAND_KEYS)  # no jump (ramp slope 10 per unit, grid step 0.0005)
        assert b["high"] >= a["high"] - 1e-9 and b["low"] <= a["low"] + 1e-9
    hard = [p for p, label in zip(grid, labels) if max(label.values()) == 1.0]
    assert all(abs(p - 1 / 3) >= 0.05 - 1e-9 and abs(p - 2 / 3) >= 0.05 - 1e-9 for p in hard), "hard labels only outside ±0.05 of an edge"
    for bad in (-0.01, 1.01, float("nan")):
        with pytest.raises(ValueError):
            s1.band_label(bad)


@pytest.mark.parametrize("a, b, expected", [
    (0.2, 0.2, {"first": 0.0, "second": 0.0, "same": 1.0}), (0.2, 0.25, {"first": 0.0, "second": 0.0, "same": 1.0}),
    (0.2, 0.3, {"first": 0.0, "second": 0.5, "same": 0.5}), (0.3, 0.2, {"first": 0.5, "second": 0.0, "same": 0.5}),
    (0.2, 0.32, {"first": 0.0, "second": 0.7, "same": 0.3}), (0.2, 0.35, {"first": 0.0, "second": 1.0, "same": 0.0}),
    (0.9, 0.1, {"first": 1.0, "second": 0.0, "same": 0.0}),
])  # fmt: skip
def test_comparison_labels_say_about_the_same_below_01_with_a_soft_edge(a, b, expected):
    label = s1.compare_label(a, b)
    assert label == pytest.approx(expected, abs=1e-6) and abs(sum(label.values()) - 1.0) < 1e-9
    swapped = s1.compare_label(b, a)
    assert swapped["first"] == pytest.approx(label["second"]) and swapped["same"] == pytest.approx(label["same"])


# --------------------------------------------------------------------------
# real training records
# --------------------------------------------------------------------------


def test_the_originals_stay_byte_identical_and_derived_questions_are_appended_after_them():
    for source, out in zip(train_records(), transformed()):
        n = len(source["request"]["questions"])
        assert list(out["request"]["questions"])[:n] == list(source["request"]["questions"])
        assert all(out["request"]["questions"][k] == v and out["labels"][k] == source["labels"][k] for k, v in source["request"]["questions"].items())
        assert out["request"]["state"] == source["request"]["state"] and out["id"] == source["id"] and out["generator"] == source["generator"]
        assert {k: v for k, v in out["meta"].items() if k not in ("system1", "question_families")} == source["meta"]
        assert out["truth"] == source["truth"] and out["rationale"] == source["rationale"] and list(out) == list(source)
        assert out["meta"]["key_question"] == source["meta"]["key_question"]
        derived = list(out["request"]["questions"])[n:]
        assert derived == list(out["meta"]["system1"]["derived"]) == [k for k in out["labels"]][n:]
        assert all(out["meta"]["question_families"][k] in ("s1:band", "s1:comparison") for k in derived)


def test_every_derived_label_follows_the_rule_from_its_source_labels():
    bands = comparisons = 0
    for source, out in zip(train_records(), transformed()):
        for name, item in out["meta"]["system1"]["derived"].items():
            label = out["labels"][name]
            assert label["type"] == "choice" and label["kind"] == "state"
            assert item["p"] == [_p(source, q) for q in item["of"]]
            if item["kind"] == "band":
                bands += 1
                (q,) = item["of"]
                assert label["probabilities"] == s1.band_label(_p(source, q))
                assert list(out["request"]["questions"][name]["criteria"]) == list(s1.BAND_KEYS)
            else:
                comparisons += 1
                first, second = item["of"]
                assert label["probabilities"] == s1.compare_label(_p(source, first), _p(source, second))
                text = out["request"]["questions"][name]["instructions"]
                assert text.index(source["request"]["questions"][first]["instructions"].strip()) < text.index(
                    source["request"]["questions"][second]["instructions"].strip()), "the first quoted question is the label's first"
    assert bands > 100 and comparisons > 10


def test_band_questions_cover_the_event_questions_and_comparisons_are_bench_system1_pairs():
    for source, out in zip(train_records(), transformed()):
        events = s1.event_questions(source)
        basis = source["meta"]["basis"]
        assert all(source["request"]["questions"][q]["type"] == "noul" or basis[q]["source"] == "event" for q in events)
        assert not [q for q, question in source["request"]["questions"].items() if question["type"] == "score" and basis[q]["source"] == "quantity" and q in events]
        pairs = {frozenset((qa, qb)) for _, (_, qa, na), (_, qb, nb) in basis_event_pairs([source]) if na == nb}
        derived = out["meta"]["system1"]["derived"].values()
        made_bands = [item["of"][0] for item in derived if item["kind"] == "band"]
        made_pairs = {frozenset(item["of"]) for item in derived if item["kind"] == "comparison"}
        dropped = out["meta"]["system1"]["dropped"]
        assert len(made_bands) + dropped.get("band", 0) == len(events) and set(made_bands) <= set(events)
        assert made_pairs <= pairs and len(made_pairs) + dropped.get("comparison", 0) == len(pairs)


def test_the_budget_keeps_twelve_questions_and_the_key_questions_band_first():
    truncated = 0
    for source in train_records():
        out = s1.transform_record(source)
        assert len(out["request"]["questions"]) <= s1.MAX_QUESTIONS
        derived = list(out["meta"]["system1"]["derived"].values())
        key = source["meta"]["key_question"]
        if key in s1.event_questions(source) and len(source["request"]["questions"]) < s1.MAX_QUESTIONS:
            assert derived[0] == {"kind": "band", "of": [key], "p": [_p(source, key)]}
        kinds = [item["kind"] for item in derived[1:]]
        assert kinds == sorted(kinds, key=lambda k: 0 if k == "comparison" else 1), "comparisons before the other bands"
        truncated += bool(out["meta"]["system1"]["dropped"])
    # a record with 8 questions and many event questions must be trimmed: force one
    big = copy.deepcopy(max(train_records(), key=lambda r: len(s1.event_questions(r)) + len(s1.comparison_candidates(r))))
    for i in range(12 - len(big["request"]["questions"])):
        big["request"]["questions"][f"extra_{i}"] = {"type": "noul", "instructions": f"Is fact {i} true?"}
        big["labels"][f"extra_{i}"] = {"type": "noul", "p_yes": 0.5}
        big["meta"]["basis"][f"extra_{i}"] = {"kind": "noul", "source": "world", "event": f"extra_{i}"}
    out = s1.transform_record(big)
    assert len(out["request"]["questions"]) == 12 and not out["meta"]["system1"]["derived"] and sum(out["meta"]["system1"]["dropped"].values()) > 0
    capped = s1.transform_record(train_records()[0], max_derived=1)
    assert len(capped["meta"]["system1"]["derived"]) == 1 and capped["meta"]["system1"]["max_derived"] == 1


def test_both_validators_and_the_trainers_example_builder_accept_the_records(tmp_path):
    from robo_jev.decisions.examples import build_examples, dev_examples
    from robo_jev.decisions.train import load_training_records
    from robo_jev.model.tokenizer import available_tokenizer, load_tokenizer

    for out in transformed():
        assert engine_validate(out) == [] and trainer_validate(out) == []
    path = tmp_path / "engine-s1.jsonl"
    path.write_text("".join(dumps(r) + "\n" for r in transformed()), encoding="utf-8")
    loaded, check = load_training_records(path)
    assert check["invalid"] == 0 and len(loaded) == len(transformed())
    tokenizer = load_tokenizer(available_tokenizer()[0])
    record = next(r for r in loaded if any(v["kind"] == "comparison" for v in r["meta"]["system1"]["derived"].values()))
    names = list(record["meta"]["system1"]["derived"])
    for example in build_examples(record, names, tokenizer, order_seed=7, fmt="chat"):
        probs = record["labels"][example["question"]]["probabilities"]
        assert example["target"] == pytest.approx([probs[k] for k in example["options"]])
        assert example["text"].count("\nOptions:\n") == 1
    prompts = dev_examples([record], tokenizer)
    assert {p["question"] for p in prompts} == set(record["request"]["questions"])


def test_the_diagnosis_tool_scores_the_records_and_reports_the_derived_families(tmp_path):
    pytest.importorskip("transformers")
    from robo_jev.model.backbone_qwen import torch_reference_kernels

    spec = importlib.util.spec_from_file_location("diagnose_decisions_s1", REPO / "scripts" / "diagnose_decisions.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    picked = [r for r in transformed() if any(v["kind"] == "comparison" for v in r["meta"]["system1"]["derived"].values())][:2]
    path = tmp_path / "s1.jsonl"
    path.write_text("".join(dumps(r) + "\n" for r in picked), encoding="utf-8")
    out = tmp_path / "run.json"
    with torch_reference_kernels():
        assert module.main(["--model", "tiny", "--input", str(path), "--out", str(out), "--n-boot", "10", "--batch", "8"]) == 0
    summary = json.loads(out.read_text(encoding="utf-8"))
    reading = summary["summary"]["readings"]["single_order"]
    assert summary["validation"]["invalid"] == 0 and reading["counts"]["scored"] == sum(len(r["request"]["questions"]) for r in picked)
    assert {"s1:band", "s1:comparison"} <= set(reading["groups"]["family"])


def test_the_transform_is_deterministic_and_does_not_touch_its_input():
    record = train_records()[3]
    frozen = json.dumps(record, sort_keys=True)
    first, second = s1.transform_record(record), s1.transform_record(copy.deepcopy(record))
    assert dumps(first) == dumps(second) and json.dumps(record, sort_keys=True) == frozen
    coins = {s1._first_is(f"v0b-{i:06d}", "a", "b") for i in range(64)}
    assert coins == {"a", "b"}, "the first/second coin takes both sides"


def test_korean_records_get_korean_frames_and_no_derived_text_uses_t2_held_out_wording():
    from robo_jev.simgen.phrasing import language_wide_hits

    for lang, frames in s1.FRAMES.items():
        texts = [frames["band_noul"].replace("{q}", ""), frames["band_score"].replace("{q}", ""),
                 frames["compare"].replace("{a}", "").replace("{b}", ""), *frames["band_options"].values(), *frames["compare_options"].values()]
        for text in texts:
            assert language_wide_hits(lang, text) == ([], []), (lang, text)
    assert any(r["meta"]["language"] == "ko" for r in transformed())
    tails = {lang: (frames["band_noul"].split("{q}")[-1], frames["band_score"].split("{q}")[-1], frames["compare"].split("{b}")[-1])
             for lang, frames in s1.FRAMES.items()}
    for out in transformed():
        lang = out["meta"]["language"]
        for name in out["meta"]["system1"]["derived"]:
            assert out["request"]["questions"][name]["instructions"].endswith(tails[lang]), (lang, name)
    script = _script()
    for source, out in zip(train_records(), transformed()):
        assert script._new_held_out_hits(source, out) == []


def test_sealed_and_already_transformed_records_are_refused():
    record = copy.deepcopy(train_records()[0])
    for meta_split, gen_split in (("T1", "T1"), ("train", "T2"), ("T7", "train"), (None, "train")):
        bad = copy.deepcopy(record)
        bad["meta"]["split"] = meta_split
        bad["generator"]["split"]["name"] = gen_split
        with pytest.raises(s1.SealedRecordError):
            s1.transform_record(bad)
    with pytest.raises(ValueError, match="already transformed"):
        s1.transform_record(s1.transform_record(record))
    script = _script()
    for sealed in (REPO / "artifacts" / "simgen" / "v0b" / "tests" / "T1.jsonl", REPO / "artifacts" / "t8" / "T8b.jsonl",
                   REPO / "artifacts" / "simgen" / "probes" / "G2.jsonl"):
        with pytest.raises(SystemExit, match="sealed"):
            script.refuse_sealed_path(sealed)


def test_exposure_counts_two_examples_per_record_and_derived_ones_only_in_the_choice_slot():
    before = s1.exposure(train_records(), q=2)
    after = s1.exposure(transformed(), q=2)
    assert sum(before.values()) == sum(after.values()) == 2 * len(train_records())
    assert not any(k.startswith("s1:") for k in before) and any(k.startswith("s1:") for k in after)
    choice = lambda counts: sum(v for k, v in counts.items() if k.startswith(("choice:", "s1:")))  # noqa: E731
    assert abs(choice(after) - choice(before)) <= 0.15 * len(train_records()), "derived questions take the choice slot, not new slots"


@functools.lru_cache(maxsize=1)
def _script():
    spec = importlib.util.spec_from_file_location("system1_v03_script", REPO / "scripts" / "system1_v03.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_the_script_writes_the_file_and_its_manifest_and_rebuilds_byte_identically(tmp_path):
    source = tmp_path / "train.jsonl"
    source.write_text("".join(dumps(r) + "\n" for r in train_records()), encoding="utf-8")
    outs = []
    for name in ("a", "b"):
        out = tmp_path / name / "engine-s1.jsonl"
        assert _script().main(["transform", "--input", str(source), "--out", str(out)]) == 0
        outs.append(out)
    assert outs[0].read_bytes() == outs[1].read_bytes()
    manifest = json.loads(outs[0].with_name("engine-s1.manifest.json").read_text(encoding="utf-8"))
    assert manifest["checks"]["validation_errors"] == manifest["checks"]["originals_changed"] == manifest["checks"]["held_out_wording_new_hits"] == 0
    assert manifest["stats"]["records"] == len(train_records()) and manifest["stats"]["derived_total"] > 0
    assert sum(manifest["exposure"]["after"].values()) == sum(manifest["exposure"]["before"].values())
    bad = tmp_path / "bad.jsonl"
    record = copy.deepcopy(train_records()[0])
    record["meta"]["split"] = "T3"
    bad.write_text(dumps(record) + "\n", encoding="utf-8")
    with pytest.raises(s1.SealedRecordError):
        _script().main(["transform", "--input", str(bad), "--out", str(tmp_path / "c" / "x.jsonl")])
    assert not (tmp_path / "c" / "x.jsonl").exists()


def test_the_built_engine_s1_file_matches_its_manifest_and_its_source():
    path, manifest_path = V03 / "engine-s1.jsonl", V03 / "engine-s1.manifest.json"
    if not manifest_path.is_file():
        pytest.skip("artifacts/simgen/v03/engine-s1.jsonl not built yet (scripts/system1_v03.py transform)")
    from robo_jev.decisions.records import file_sha256

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["out"]["sha256"] == file_sha256(path) and manifest["out"]["records"] == 50_000
    assert manifest["input"]["sha256"] == json.loads((V02 / "train-50k.manifest.json").read_text(encoding="utf-8"))["sha256"]
    assert manifest["checks"]["validation_errors"] == manifest["checks"]["originals_changed"] == manifest["checks"]["held_out_wording_new_hits"] == 0
    with path.open(encoding="utf-8") as built, (V02 / "train-50k.jsonl").open(encoding="utf-8") as source:
        for _, a, b in zip(range(200), built, source):
            out, record = json.loads(a), json.loads(b)
            assert dumps(s1.transform_record(record)) == a.rstrip("\n") and out["id"] == record["id"]
    assert 0.05 < manifest["stats"]["labels"]["band"]["soft_share"] < 0.2
