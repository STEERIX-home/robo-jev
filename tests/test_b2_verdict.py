"""Task B2 — prior robustness of T8b's labels (reproduction check, bands under three priors, records with ``meta.robustness``) and the
pre-declared rule of the cause check (instrument check, P1 / P2 / P3, the two-band variant) applied by ``scripts/b2_verdict.py``."""

import importlib.util
import json
import sys

import pytest
from helpers import REPO

from robo_jev.bench import infamily, system1
from robo_jev.decisions.records import file_sha256
from robo_jev.outside import robustness as rb


def _verdict():
    spec = importlib.util.spec_from_file_location("b2_verdict_test", REPO / "scripts" / "b2_verdict.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def t8b_like(rid: str, family: str = "stack", *, planned: float = 0.2, out: float = 0.8, inner: float = 0.1) -> dict:
    """A record shaped like a T8b release moment (the fields the readings use)."""
    return {"id": rid, "meta": {"set": "T8b", "scene": family},
            "request": {"state": f"state {rid}", "questions": {"falls_off": {"type": "noul", "instructions": "?"},
                                                              "falls_if_further_out": {"type": "score", "instructions": "?", "criteria": ["0", "1"]},
                                                              "falls_if_further_in": {"type": "noul", "instructions": "?"}}},
            "labels": {"falls_off": {"type": "noul", "p_yes": planned}, "falls_if_further_out": {"type": "score", "value": out},
                       "falls_if_further_in": {"type": "noul", "p_yes": inner}}}


def row(record: dict, **values) -> dict:
    labels = json.loads(json.dumps(record["labels"]))
    for name, value in values.items():
        labels[name]["p_yes" if labels[name]["type"] == "noul" else "value"] = value
    return {"id": record["id"], "labels": labels}


def answer(rid: str, values: dict) -> dict:
    questions = {}
    for name, (a, b) in values.items():
        field = "value" if name == "falls_if_further_out" else "p_yes"
        questions[name] = {"type": "score" if field == "value" else "noul",
                           "listings": [{"options": None if field == "value" else o, "answer": {field: v}} for o, v in ((["yes", "no"], a), (["no", "yes"], b))]}
    return {"id": rid, "questions": questions}


def test_reproduction_counts_questions_identical_within_1e_9():
    records = [t8b_like("t8b-stack-0000"), t8b_like("t8b-stack-0001", planned=0.5)]
    rows = {"t8b-stack-0000": row(records[0], falls_off=0.2 + 1e-12), "t8b-stack-0001": row(records[1], falls_if_further_in=0.125)}
    found = rb.reproduction(records, rows)
    assert (found["identical"], found["questions"]) == (5, 6) and found["share"] == pytest.approx(5 / 6)
    assert found["mismatches"][0]["question"] == "falls_if_further_in"
    assert rb.reproduction(records, {})["identical"] == 0  # a missing row reproduces nothing


def test_a_question_is_robust_only_when_its_band_is_the_same_under_every_prior():
    record = t8b_like("t8b-stack-0000", planned=0.2, out=0.8, inner=0.5)
    found = rb.band_robustness([record], {"alt_a": {record["id"]: row(record, falls_off=0.30, falls_if_further_out=0.6)},
                                          "alt_b": {record["id"]: row(record, falls_off=0.05, falls_if_further_in=0.4)}})
    assert found[(record["id"], "falls_off")]["robust"]                       # low, low, low
    assert not found[(record["id"], "falls_if_further_out")]["robust"]         # high, middle, high
    assert found[(record["id"], "falls_if_further_in")]["bands"] == {"original": "middle", "alt_a": "middle", "alt_b": "middle"}
    assert found[(record["id"], "falls_if_further_out")]["min"] == 0.6 and found[(record["id"], "falls_if_further_out")]["max"] == 0.8
    with pytest.raises(ValueError):  # a missing relabelled label never shrinks the robust set silently
        rb.band_robustness([record], {"alt_a": {}})
    choice = {"id": "t8b-brake-0000", "meta": {"set": "T8b", "scene": "deck_brake"},
              "request": {"state": "s", "questions": {"what_happens": {"type": "choice", "instructions": "?", "criteria": {"stays": "", "slides": "", "topples": ""}}}},
              "labels": {"what_happens": {"type": "choice", "kind": "state", "probabilities": {"stays": 0.6, "slides": 0.3, "topples": 0.1}}}}
    alt = json.loads(json.dumps(choice))
    alt["labels"]["what_happens"]["probabilities"] = {"stays": 0.2, "slides": 0.7, "topples": 0.1}
    entry = rb.band_robustness([choice], {"alt_a": {choice["id"]: alt}})[(choice["id"], "what_happens")]
    assert entry["argmax"] == {"original": "stays", "alt_a": "slides"} and not entry["robust"]


def test_records_with_robustness_feed_the_in_family_adapters_without_touching_the_originals():
    record = t8b_like("t8b-stack-0000", planned=0.2, out=0.8, inner=0.5)
    robustness = rb.band_robustness([record], {"alt_a": {record["id"]: row(record, falls_if_further_out=0.6)}})
    copies = rb.with_robustness([record], robustness)
    assert "robustness" not in record["meta"]
    assert infamily.robust_keys(copies) == {(record["id"], "falls_off"), (record["id"], "falls_if_further_in")}
    assert infamily.ambiguous_keys(copies) == {(record["id"], "falls_if_further_out")}


def test_the_rule_separates_p1_p2_and_p3_and_tightens_the_level_threshold_for_two_bands():
    v = _verdict()
    assert v.rule_outcome(0.11, 0.4, 0.55, two_band=False) == "P1"
    assert v.rule_outcome(0.11, 0.4, 0.60, two_band=True) == "P3"    # two bands need 0.65
    assert v.rule_outcome(0.11, 0.4, 0.65, two_band=True) == "P1"
    assert v.rule_outcome(0.10, 0.4, 0.90, two_band=False) == "P3"   # the lower bound must exceed +0.10
    assert v.rule_outcome(-0.2, 0.099, 0.30, two_band=False) == "P2"
    assert v.rule_outcome(-0.2, 0.10, 0.30, two_band=False) == "P3"  # the upper bound must be below +0.10
    with pytest.raises(ValueError):
        v.rule_outcome(None, 0.1, 0.5, two_band=False)
    assert v.instrument_check(0.995, 100, 100)["pass"]
    assert not v.instrument_check(0.9949, 300, 300)["pass"] and not v.instrument_check(1.0, 99, 300)["pass"]
    failed = v.decide(v.instrument_check(0.99, 300, 300), {"low": 0.2, "high": 0.4}, {"value": 0.7}, two_band=False)
    assert failed["outcome"] == "P1" and not failed["interpreted"] and "no interpretation" in failed["meaning"]


def test_the_primary_drops_the_middle_stratum_only_in_the_two_band_variant():
    v = _verdict()
    items = {("low", "T8b", "a", "q"): ("a", 1.0), ("middle", "T8b", "b", "q"): ("b", 0.0), ("high", "T8b", "c", "q"): ("c", 0.0)}
    robust = {("a", "q"), ("b", "q"), ("c", "q")}
    assert set(v.primary_items(items, robust, two_band=False)) == set(items)
    assert {k[0] for k in v.primary_items(items, robust, two_band=True)} == {"low", "high"}
    assert system1.summarize(v.primary_items(items, robust, two_band=True), stratified=True)["value"] == 0.5


def _write_predictions(path, predictions, sha, **config):
    path.write_text("".join(json.dumps(p) + "\n" for p in predictions), encoding="utf-8")
    conditions = {"mode": "direct", "format": "chat", "prompt_version": "v0c-p3", "orders": "counterbalanced", "shuffle_seed": 1}
    path.with_name(path.name.replace(".predictions.jsonl", ".json")).write_text(
        json.dumps({"config": {"input_sha256": sha, "model": "test", "records": len(predictions), **conditions, **config}}), encoding="utf-8")


def test_the_verdict_script_applies_the_rule_end_to_end(tmp_path):
    v = _verdict()
    records = []
    for i in range(120):
        level = (0.1, 0.5, 0.9)[i % 3]
        records.append(t8b_like(f"t8b-stack-{i:04d}", "stack" if i % 4 else "push", planned=level, out=1 - level, inner=level))
    t8b = tmp_path / "T8b.jsonl"
    t8b.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
    sha = file_sha256(t8b)
    labels = tmp_path / "b2"
    labels.mkdir()
    files = {}
    for prior in v.PRIORS:
        rows = []
        for r in records:
            shift = {"original": 0.0, "alt_a": 0.1, "alt_b": -0.1}[prior]
            # the planned and further-out questions keep their band; the further-in one moves across a band edge under alt_a when middle
            changed = row(r, falls_if_further_in=min(1.0, r["labels"]["falls_if_further_in"]["p_yes"] + (0.2 if prior == "alt_a" else 0.0)))
            changed["labels"]["falls_off"]["p_yes"] = max(0.0, min(1.0, r["labels"]["falls_off"]["p_yes"] + shift / 10))
            if prior == "original":
                changed = row(r)
            rows.append({**changed, "prior": prior, "family": r["meta"]["scene"], "scene_matches_generator": True, "request_matches": True,
                         "draws_seed_matches": True, "unstable": 0, "off_deck": 0})
        path = labels / f"T8b-labels-{prior}.jsonl"
        path.write_text("".join(json.dumps(x) + "\n" for x in rows), encoding="utf-8")
        files[prior] = {"file": str(path), "sha256": file_sha256(path), "rows": len(rows), "scene_matches_generator": len(rows), "request_matches": len(rows),
                        "draws_seed_matches": len(rows), "moments_with_unstable_rollouts": 0, "moments_off_deck": 0}
    (labels / "T8b-labels.manifest.json").write_text(json.dumps({"input_sha256": sha, "files": files, "priors": {p: {} for p in v.PRIORS}}), encoding="utf-8")
    constant = [answer(r["id"], {q: (0.5, 0.5) for q in r["request"]["questions"]}) for r in records]
    oracle = system1.oracle_predictions(records)
    paths = {}
    for tag, preds in (("base", constant), ("v01mix", constant), ("v03", oracle)):
        paths[tag] = tmp_path / f"{tag}.predictions.jsonl"
        _write_predictions(paths[tag], preds, sha)
    out = tmp_path / "verdict.json"
    args = ["--input", str(t8b), "--labels", str(labels), "--out", str(out), "--n-boot", "50"] + [x for t, p in paths.items() for x in (f"--{t}", str(p))]
    assert v.main(args) == 0
    result = json.loads(out.read_text(encoding="utf-8"))
    ins = result["verdict"]["instrument"]
    assert ins["reproduced_share"] == 1.0 and ins["robust_low"] >= 100 and ins["robust_high"] >= 100 and ins["pass"]
    assert result["readings"]["models"]["v03"]["primary"]["value"] == pytest.approx(1.0)
    assert result["verdict"]["outcome"] == "P1" and result["verdict"]["interpreted"]
    assert out.with_suffix(".md").is_file()
    # the robust middle band holds 40 planned questions + 40 further-out ones (≥ 50): three bands
    assert not result["verdict"]["two_band"]
    sens = result["readings"]["sensitivity"]
    assert sens["other_band_choice"]["two_band"] and sens["other_band_choice"]["outcome_if_applied"] in ("P1", "P2", "P3")
    assert sens["without_flagged_moments"]["flagged_moments"] == [] and sens["without_flagged_moments"]["questions"] == result["readings"]["models"]["v03"]["primary"]["n"]
    # a prediction file scored on another T8b file is refused
    _write_predictions(paths["base"], constant, "0" * 64)
    with pytest.raises(SystemExit):
        v.main(args)
