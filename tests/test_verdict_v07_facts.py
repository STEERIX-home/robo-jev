"""The v0.7 secondary reading "facts in instructions" (task v0.7f §1.4) — ``scripts/verdict_v07_facts.py``.

The pure reading is pinned on hand-made readings, every threshold at its bound (v0.6's gap_F +0.05 for activity, Δgap_F −0.05 and its upper
bound < 0, v0.7's gap_F +0.03, v0.7's ΔCA_F −0.05) and each outcome (READS FACTS, NO HARM, NOT SHOWN — and "facts not used"). The readings are
pinned on synthetic M2g-short / M2g-F records and predictions giving each outcome; the helper's per-arm DC(short) equals the value
``scripts/verdict_v07.py`` computes on the same predictions; exclusions, per-template readings and seed spread; ``main`` end to end on
synthetic files (checked predictions in, the reading out). No model, no sealed file, no real prediction.
"""

from __future__ import annotations

import functools
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest
from helpers import REPO

from robo_jev.bench import gates_v07f as F


@functools.lru_cache(maxsize=None)
def H():
    spec = importlib.util.spec_from_file_location("v07_facts_reading_helper", REPO / "scripts" / "verdict_v07_facts.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _d(value=None, low=None, high=None) -> dict:
    return {"value": value, "low": low, "high": high}


def _entry(gap6=0.10, gap7=0.0, dgap=(-0.10, -0.15, -0.05), dca=0.0) -> dict:
    return {"gap_v06": _d(gap6), "gap_v07": _d(gap7), "dgap": _d(*dgap), "dca_v07": _d(dca)}


# --------------------------------------------------------------------------
# the pure reading
# --------------------------------------------------------------------------


@pytest.mark.parametrize("entry, status, outcome, key, expected", [
    (_entry(), "active", "READS FACTS", None, None),
    (_entry(gap6=0.05), "active", "READS FACTS", None, None),                                         # v0.6's gap_F exactly +0.05 is active (≥)
    (_entry(gap6=0.0499, dgap=(0.02, 0.0, 0.04)), "inactive", "NO HARM", None, None),                # just below: inactive, Δgap not read
    (_entry(dgap=(-0.05, -0.1, -0.01)), "active", "READS FACTS", "dgap_le_-0.05", True),               # Δgap exactly −0.05 counts (≤)
    (_entry(dgap=(-0.0499, -0.1, -0.01)), "active", "NOT SHOWN", "dgap_le_-0.05", False),
    (_entry(dgap=(-0.1, -0.2, 0.0)), "active", "NOT SHOWN", "dgap_high_lt_0", False),                 # an upper bound of exactly 0 does not (<)
    (_entry(dgap=(-0.1, -0.2, -1e-9)), "active", "READS FACTS", "dgap_high_lt_0", True),
    (_entry(gap7=0.03), "active", "READS FACTS", "gap_v07_le_+0.03", True),                            # v0.7's gap_F exactly +0.03 counts (≤)
    (_entry(gap7=0.0301), "active", "NOT SHOWN", "gap_v07_le_+0.03", False),
    (_entry(dca=-0.05), "active", "READS FACTS", "dca_v07_ge_-0.05", True),                            # v0.7's ΔCA_F exactly −0.05 counts (≥)
    (_entry(dca=-0.0501), "active", "NOT SHOWN", "dca_v07_ge_-0.05", False),
    (_entry(gap6=0.0, gap7=0.03, dgap=(0.03, 0.01, 0.05)), "inactive", "NO HARM", "gap_v07_le_+0.03", True),
    (_entry(gap6=0.0, gap7=0.0301, dgap=(0.03, 0.01, 0.05)), "inactive", "NOT SHOWN", "gap_v07_le_+0.03", False),  # v0.7 must not make a gap
    (_entry(gap6=0.0, dca=-0.05), "inactive", "NO HARM", "dca_v07_ge_-0.05", True),
    (_entry(gap6=0.0, dca=-0.0501), "inactive", "NOT SHOWN", "dca_v07_ge_-0.05", False),
    ({"gap_v07": _d(0.0), "dca_v07": _d(0.0), "dgap": _d(-0.1, -0.2, -0.05), "gap_v06": _d()}, "missing", "NOT SHOWN", None, None),
    (_entry(dca=None), "active", "NOT SHOWN", "dca_v07_ge_-0.05", False),                              # a missing ΔCA_F does not pass
])  # fmt: skip
def test_each_outcome_and_threshold_at_its_bound(entry, status, outcome, key, expected):
    result = H().facts_outcome(entry)
    assert result["status"] == status and result["outcome"] == outcome
    if key is not None:
        assert result["checks"][key] is expected
    if status == "inactive":
        assert set(result["checks"]) == {"gap_v07_le_+0.03", "dca_v07_ge_-0.05"}, "an inactive F is read on v0.7's gap_F and ΔCA_F only"
    assert result["consequence"] == H().CONSEQUENCE[outcome]


def test_a_ca_drop_marks_the_facts_as_not_used_even_with_a_small_gap():
    v = H()
    result = v.facts_outcome(_entry(gap7=0.0, dca=-0.2))
    assert result["outcome"] == "NOT SHOWN" and result["facts_not_used"] is True and "not read as robustness" in result["note"]
    assert v.facts_outcome(_entry(dca=-0.05))["facts_not_used"] is False
    assert v.CONSEQUENCE["READS FACTS"] == v.CONSEQUENCE["NO HARM"] == "the facts transform stays in the recipe"
    assert "own primary gate before more data" in v.CONSEQUENCE["NOT SHOWN"]
    assert (v.F_ACTIVE, v.F_DGAP, v.F_HIGH, v.F_GAP, v.F_DCA) == (0.05, -0.05, 0.0, 0.03, -0.05)


# --------------------------------------------------------------------------
# synthetic records and predictions
# --------------------------------------------------------------------------

STATE = ("SUPERVISOR: Rover Six, depot. Where are you leaving the box?\nROBOT: On the slope. Ground is stone flags at {g} degrees.\n"
         "SUPERVISOR: Contents?\nROBOT: Half full, {w} kilograms.")


def _records(pairs: int = 60) -> tuple[list[dict], list[dict]]:
    """Synthetic M2g-short records (one-fact pairs; the key question's class by g mod 3: clear-risky, clear-safe, unclear; the score question
    unclear) and their M2g-F records (task v0.7f's assignment over two strata)."""
    short = []
    for g in range(pairs):
        label = {0: 0.9, 1: 0.1, 2: 0.5}[g % 3]
        for member, weight in (("", 40), ("p", 55)):
            rid = f"v07m2g-v06a4-bank-{g:06d}{member}"
            partner = f"v07m2g-v06a4-bank-{g:06d}{'p' if not member else ''}"
            short.append({"schema": "simgen-request/0", "id": rid, "generator": {"family": "bank", "group": g},
                          "meta": {"set": "v07-m2g", "scene": "bank", "pair": {"of": partner},
                                   "source": {"set": "A4" if g % 2 else "Hset", "id": rid[7:]}, "gate": {"name": "M2g", "style": "ticket"},
                                   "robustness": {"key": {"robust": True}, "score": {"robust": False}}},
                          "request": {"state": STATE.format(g=10 + g % 20, w=weight),
                                      "questions": {"key": {"type": "noul", "instructions": "Will the box move?"},
                                                    "score": {"type": "score", "instructions": "Rate the chance it moves.", "criteria": ["0", "1"]}}},
                          "labels": {"key": {"type": "noul", "p_yes": label}, "score": {"type": "score", "value": 0.5}}})
    strata: dict[str, list[str]] = {}
    for r in short:
        key = min(r["id"], r["meta"]["pair"]["of"])
        strata.setdefault(r["meta"]["source"]["set"], [])
        if key not in strata[r["meta"]["source"]["set"]]:
            strata[r["meta"]["source"]["set"]].append(key)
    chosen = F.assign(strata, seed=0)
    facts = [F.facts_record(r, **chosen[min(r["id"], r["meta"]["pair"]["of"])]) for r in short]
    assert all(F.facts_problems(f, s) == [] for f, s in zip(facts, short))
    return short, facts


def _u(*parts) -> float:
    return int.from_bytes(hashlib.sha256("|".join(map(str, parts)).encode()).digest()[:8], "big") / 2**64


def _preds(records: list[dict], model: str, *, unclear: float, clear_safe: float = 0.6) -> list[dict]:
    """Unclear questions are called "safe" (0.1) iff their fixed draw u < `unclear`, clear-safe ones iff u < `clear_safe`, clear-risky ones
    are always "look again" (0.5) — the same u in both forms and for every model, so DC = 0.8 × (share of unclear safe calls) and CA = half
    the clear-safe share."""
    v5 = H().V7().V5()
    out = []
    for r in records:
        questions = {}
        for name, q in r["request"]["questions"].items():
            truth = v5.truth_class(r, name)
            share = {"unclear": unclear, "clear_safe": clear_safe, "clear_risky": -1.0}[truth]
            x = 0.1 if _u(r["id"], name) < share else 0.5
            answer = {"p_yes": x} if q["type"] == "noul" else {"value": x}
            questions[name] = {"type": q["type"], "n_listings": 1, "answer": answer, "listings": [{"options": ["yes", "no"], "answer": answer}]}
        out.append({"id": r["id"], "model": model, "questions": questions})
    return out


def _all_preds(spec: dict, short: list[dict], facts: list[dict]) -> dict:
    """``{model: {"short": {...}, "facts": {...}}}`` → predictions (``unclear`` and optional ``clear_safe`` shares per form)."""
    records = {"short": short, "facts": facts}
    return {m: {form: _preds(records[form], m, **shares) for form, shares in forms.items()} for m, forms in spec.items()}


def _spec(v07=(0.25, 0.25), v06=(0.25, 0.45), base=(0.40, 0.45), cs07=(0.6, 0.6)) -> dict:
    """Per arm the unclear safe share (short, F) — seed 1 a little above seed 0 in both forms; v0.7's clear-safe share (short, F)."""
    def arm(shares, cs=(0.6, 0.6)):
        return [{"short": {"unclear": shares[0] + d, "clear_safe": cs[0]}, "facts": {"unclear": shares[1] + d, "clear_safe": cs[1]}} for d in (0.0, 0.03)]
    s0, s1 = arm(v07, cs07)
    t0, t1 = arm(v06)
    return {"v07s0": s0, "v07s1": s1, "v06s0": t0, "v06s1": t1, "base": arm(base)[0]}


@pytest.mark.parametrize("spec, outcome, status, not_used", [
    (_spec(), "READS FACTS", "active", False),                                         # v0.6 gaps on F, v0.7 does not
    (_spec(v07=(0.25, 0.45)), "NOT SHOWN", "active", False),                           # the same gap for both arms
    (_spec(v06=(0.25, 0.27)), "NO HARM", "inactive", False),                           # no v0.6 gap, none made by v0.7
    (_spec(v06=(0.25, 0.27), v07=(0.25, 0.40)), "NOT SHOWN", "inactive", False),       # v0.7 makes a gap where v0.6 had none
    (_spec(cs07=(0.6, 0.15)), "NOT SHOWN", "active", True),                            # no gap, but v0.7's clear-safe calls collapse on F
])  # fmt: skip
def test_synthetic_predictions_give_each_outcome(spec, outcome, status, not_used):
    v = H()
    short, facts = _records()
    readings = v.facts_readings(short, facts, _all_preds(spec, short, facts), n_boot=300, seed=0)
    result = readings["outcome"]
    assert result["outcome"] == outcome and result["status"] == status and result["facts_not_used"] is not_used
    rule = readings["rule"]
    assert rule["n"] == 240 and rule["dgap"]["value"] == pytest.approx(rule["gap_v07"]["value"] - rule["gap_v06"]["value"])
    # seed means: the arm's gap_F is the mean of its seeds' on the same questions (the call distributions average)
    for arm, seeds in v.V7().ARMS.items():
        per_seed = [readings["facts_minus_short"][s]["DC"]["value"] for s in seeds]
        assert readings["facts_minus_short"][arm]["DC"]["value"] == pytest.approx(sum(per_seed) / 2)
    assert rule["gap_v07"]["value"] == pytest.approx(readings["facts_minus_short"]["v07"]["DC"]["value"])
    assert rule["dca_v07"]["value"] == pytest.approx(readings["facts_minus_short"]["v07"]["CA"]["value"])
    assert set(readings["by_template"]) == set(F.TEMPLATES) and set(readings["by_set"]) == {"A4", "Hset"}
    assert set(readings["seed_spread"]) == {"v07", "v06"} and "base" in readings["facts_minus_short"]


def test_the_helpers_dc_short_per_arm_equals_verdict_v07s_on_the_same_predictions():
    v = H()
    v7 = v.V7()
    short, facts = _records()
    preds = _all_preds(_spec(), short, facts)
    mine = v.facts_readings(short, facts, preds, n_boot=200, seed=0)
    theirs = v7.m2g_readings({"short": short}, {m: {"short": p["short"]} for m, p in preds.items()}, n_boot=200, seed=0)
    for model in ("v07", "v06", "v07s0", "v07s1", "v06s0", "v06s1", "base"):
        assert mine["levels"][model]["short"]["DC"]["value"] == theirs["levels"][model]["short"]["value"], model
        assert mine["levels"][model]["short"]["DC"]["low"] == theirs["levels"][model]["short"]["low"]
    # and the gap reading is verdict_v07's own (two sources, the arm's F vs short)
    calls = {(m, form): v7.m2g_calls({"short": short, "facts": facts}[form], preds[m][form]) for m in ("v07s0", "v07s1") for form in ("short", "facts")}
    arm = {form: v7.V6().arm_calls(calls[("v07s0", form)], calls[("v07s1", form)]) for form in ("short", "facts")}
    gap = v7.gap_reading(arm["facts"], arm["short"], n_boot=200, seed=0)
    assert mine["rule"]["gap_v07"]["value"] == pytest.approx(gap["value"], abs=1e-12)
    assert (mine["rule"]["gap_v07"]["low"], mine["rule"]["gap_v07"]["high"]) == pytest.approx((gap["low"], gap["high"]), abs=1e-12)


def test_four_sources_extends_verdict_v07s_dc_bootstrap_to_every_metric():
    v = H()
    v7 = v.V7()
    short, facts = _records()
    preds = _all_preds(_spec(), short, facts)
    calls = {(m, form): v7.m2g_calls({"short": short, "facts": facts}[form], preds[m][form]) for m in ("v07s0", "v06s0") for form in ("short", "facts")}
    sources = (calls[("v07s0", "facts")], calls[("v07s0", "short")], calls[("v06s0", "facts")], calls[("v06s0", "short")])
    mine = v.four_sources(*sources, n_boot=300, seed=0)
    theirs = v7.dgap_reading(*sources, n_boot=300, seed=0)
    for key, other in (("a", "gap_v07"), ("b", "gap_v06"), ("a_minus_b", "dgap")):
        for field in ("value", "low", "high"):
            assert mine["DC"][key][field] == pytest.approx(theirs[other][field], abs=1e-12), (key, field)
    assert set(mine) >= set(v.METRICS)


def test_exclusions_leave_the_moments_out_of_both_forms():
    v = H()
    short, facts = _records()
    preds = _all_preds(_spec(), short, facts)
    out = v.facts_readings(short, facts, preds, n_boot=50, seed=0, exclude={short[0]["id"], short[1]["id"]})
    assert out["records"] == len(short) - 2 and out["rule"]["n"] == 2 * (len(short) - 2) and out["excluded_ids"] == sorted({short[0]["id"], short[1]["id"]})
    with pytest.raises(ValueError, match="same records"):
        v.facts_readings(short, facts[::-1], preds, n_boot=10, seed=0)


# --------------------------------------------------------------------------
# files
# --------------------------------------------------------------------------


def test_the_helper_never_writes_into_earlier_or_sealed_folders_nor_onto_the_v07_verdict():
    v = H()
    for bad in ("reports/v06/x.json", "v07/gates/verdict-facts.json", "iface/x.json", "t9/x.json", "reports/v07/verdict.json", "reports/v07/verdict.md"):
        with pytest.raises(SystemExit, match="refusing"):
            v.refuse_write(REPO / "artifacts" / bad)
    v.refuse_write(REPO / "artifacts" / "reports" / "v07" / "verdict-facts.json")
    with pytest.raises(SystemExit, match="--s0-dir"):
        v.main(["--out", str(REPO / "artifacts" / "reports" / "v07" / "verdict-facts.json")])
    paths = v.prediction_paths(REPO / "artifacts" / "reports" / "v07" / "spark")
    assert paths["v07s0"]["facts"].name == "M2g-facts-v07s0-2b.predictions.jsonl" and paths["base"]["short"].name == "M2g-short-base-2b.predictions.jsonl"


def _write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    return path


def test_main_reads_checked_predictions_and_writes_the_reading(tmp_path, monkeypatch):
    from robo_jev.decisions.prompt import PROMPT_VERSION, fixed_text_digest
    from robo_jev.decisions.records import file_sha256

    v = H()
    v7 = v.V7()
    short, facts = _records()
    short_file = _write_jsonl(tmp_path / "gates" / "m2g" / "M2g-short.jsonl", short)
    facts_file = _write_jsonl(tmp_path / "gates" / "m2gf" / "M2g-facts.jsonl", facts)
    manifest = tmp_path / "gates" / "m2gf" / "manifest.json"
    manifest.write_text(json.dumps({"file": {"sha256": file_sha256(facts_file)}}), encoding="utf-8")
    overlap = tmp_path / "gates" / "overlap-v07.json"
    overlap.write_text(json.dumps({"train_v07": {"sha256": "f" * 64}, "exclude": {"M2g_ids": [short[0]["id"], short[1]["id"]]}}), encoding="utf-8")
    monkeypatch.setattr(v7, "M2G_DIR", short_file.parent)
    monkeypatch.setattr(v7, "OVERLAP_V07", overlap)
    monkeypatch.setattr(v, "FACTS_FILE", facts_file)
    monkeypatch.setattr(v, "FACTS_MANIFEST", manifest)
    checkpoints = {}
    for name in ("v07-s0", "v07-s1", "v06-s0", "v06-s1"):
        path = tmp_path / f"{name}-01-fetched" / "runs" / f"{name}-2b" / "checkpoint.pt"
        path.parent.mkdir(parents=True)
        path.write_bytes(name.encode())
        checkpoints[name] = path
    monkeypatch.setattr(v7, "V06_CKPT", {0: checkpoints["v06-s0"], 1: checkpoints["v06-s1"]})
    spark = tmp_path / "spark"
    preds = _all_preds(_spec(), short, facts)
    ckpt_of = {"v07s0": checkpoints["v07-s0"], "v07s1": checkpoints["v07-s1"], "v06s0": checkpoints["v06-s0"], "v06s1": checkpoints["v06-s1"], "base": None}
    for model, forms in preds.items():
        for form, rows in forms.items():
            stem = spark / f"M2g-{form}-{v7.SPARK_TAGS[model]}"
            _write_jsonl(stem.with_suffix(".predictions.jsonl"), rows)
            config = {**v7.M2G_CONDITIONS, "input_sha256": file_sha256(short_file if form == "short" else facts_file),
                      "prompt_digest": fixed_text_digest(PROMPT_VERSION), "checkpoint": {"sha256": file_sha256(ckpt_of[model])} if ckpt_of[model] else None}
            stem.with_suffix(".json").write_text(json.dumps({"config": config}), encoding="utf-8")
    out = tmp_path / "reports" / "verdict-facts.json"
    args = ["--s0-dir", str(tmp_path / "v07-s0-01-fetched" / "reports"), "--s1-dir", str(tmp_path / "v07-s1-01-fetched" / "reports"),
            "--spark-dir", str(spark), "--n-boot", "100", "--out", str(out)]
    assert v.main(args) == 0
    result = json.loads(out.read_text(encoding="utf-8"))
    assert result["outcome"] == "READS FACTS" and result["problems"] == [] and result["readings"]["records"] == len(short) - 2
    assert out.with_suffix(".md").read_text(encoding="utf-8").startswith("# v0.7 secondary reading — facts in instructions (M2g-F): READS FACTS")
    # a prediction scored on another file, or with another checkpoint, is refused
    stem = spark / f"M2g-facts-{v7.SPARK_TAGS['v07s0']}"
    config = json.loads(stem.with_suffix(".json").read_text(encoding="utf-8"))["config"]
    stem.with_suffix(".json").write_text(json.dumps({"config": {**config, "input_sha256": file_sha256(short_file)}}), encoding="utf-8")
    with pytest.raises(SystemExit, match="fail their checks"):
        v.main(args)
    stem.with_suffix(".predictions.jsonl").unlink()
    with pytest.raises(SystemExit, match="missing prediction files"):
        v.main(args)
