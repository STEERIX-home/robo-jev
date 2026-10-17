"""``scripts/assemble_v09.py`` (Task v0.9a) — the pure parts of arm CN's assembly, on synthetic records.

* The replaced slots are general records the trainer trains on (not dev), not expanded by v0.8, an equal quota per source, chosen by a
  salted id hash — deterministic; the count is the share of ALL records.
* The arm scan finds a planted arm-like text and counts only (no text in its output); the consistency rows follow the trainer's rule
  (N = 2: one copy); the time model reproduces v0.8 F's measured pod for v0.8 F's padded tokens.
* Writing is refused under earlier inputs, sealed sets, v0.9b's folder and the reports.
No model is loaded.
"""

from __future__ import annotations

import functools
import importlib.util
import json
import sys
from pathlib import Path

import pytest
from helpers import REPO

CONFIG = {"dev_fraction": 0.02, "dev_salt": "v0d-dev", "questions_per_request": 2}


@functools.lru_cache(maxsize=None)
def a09():
    spec = importlib.util.spec_from_file_location("test_v09_assemble", REPO / "scripts" / "assemble_v09.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _general(i: int, source: str, *, expanded: bool = False) -> dict:
    meta = {"source": {"set": source, "row": i}, "domain": f"public:{source}", "language": "en"}
    if expanded:
        meta["format"] = {"expanded": {"q_general": {"to": 30}}}
    return {"id": f"{source}-train-{i:05d}", "meta": meta}


def test_the_replaced_slots_are_eligible_general_records_with_equal_quotas():
    from robo_jev.decisions.examples import is_dev

    sources = ("commonsense_qa", "cosmos_qa", "hellaswag", "qasc", "social_i_qa")
    records = []
    for s in sources:
        for i in range(400):
            records.append(_general(i, s, expanded=(i % 7 == 0)))
    records += [{"id": f"v0b-x-{i}", "meta": {"basis": {}}} for i in range(500)]  # not general
    indexed = list(enumerate(records))
    plan = a09().plan_replacements(indexed, total_records=10000, config=CONFIG, share=0.07)
    assert plan["count"] == 700 and sum(plan["quota"].values()) == 700 and set(plan["quota"].values()) == {140}
    chosen = [records[i] for i in plan["slots"]]
    assert len(chosen) == 700
    for record in chosen:
        assert record["id"].split("-train-")[0] in sources
        assert not record["meta"].get("format"), "an expanded record is never replaced"
        assert not is_dev(record, fraction=0.02, salt="v0d-dev"), "a dev record is never replaced"
    assert plan == a09().plan_replacements(indexed, total_records=10000, config=CONFIG, share=0.07), "deterministic"
    assert all(v < 400 for v in plan["eligible"].values())
    with pytest.raises(ValueError):
        a09().plan_replacements(indexed, total_records=100000, config=CONFIG, share=0.07)


def test_the_arm_scan_finds_a_planted_text_and_writes_only_counts():
    arm = {"A": [{"request": {"state": "alpha beta gamma delta epsilon zeta eta theta iota kappa", "questions": {"q": {"type": "choice",
                                                                                                               "instructions": "pick one", "criteria": {"opt_one": "lambda mu nu"}}}}}]}
    def numeric(state: str) -> dict:
        return {"request": {"state": state, "questions": {"q": {"type": "choice", "instructions": "which", "criteria": {"x": "y", "z": "w"}}}},
                "meta": {"numeric": {"family": "step", "template": "step-prose"}}}
    clean = a09().arm_scan([numeric("nothing like it at all in here, only words of our own today")], arm)
    assert all(v["shared_windows"] == 0 for v in clean["sizes"].values())
    planted = a09().arm_scan([numeric("intro words then alpha beta gamma delta epsilon zeta eta theta and more")], arm)
    assert planted["sizes"]["8"]["shared_windows"] == 1 and planted["sizes"]["5"]["shared_windows"] == 4
    assert planted["sizes"]["8"]["records_with_shared_windows_by_template"] == {"step/step-prose": 1}
    text = json.dumps(planted)
    assert "alpha" not in text and "gamma" not in text, "counts only"


def test_consistency_rows_follow_the_trainers_rule():
    full = {**a09().CONSISTENCY, "share": 1.0}
    record = {"id": "r1", "request": {"questions": {"two": {"type": "choice", "criteria": {"a": "", "b": ""}},
                                                    "five": {"type": "choice", "criteria": {k: "" for k in "abcde"}},
                                                    "yn": {"type": "noul"}}}}
    assert a09().consistency_rows(record, "two", full) == 2
    assert a09().consistency_rows(record, "five", full) == 3
    assert a09().consistency_rows(record, "yn", full) == 1
    assert a09().consistency_rows(record, "five", None) == 1
    tiny = {**full, "share": 1e-12}
    assert a09().consistency_rows(record, "five", tiny) == 1


def test_the_time_model_reproduces_v08f_for_v08fs_padded_tokens():
    m = a09().V08F_MEASURED
    sim = {seed: {"by_share": {"off": {"padded_tokens": m[seed]["padded_tokens"]}}} for seed in m}
    out = a09().time_and_budget(sim, input_bytes=m["0"]["input_bytes"])
    for seed in m:
        row = out[seed]["by_share"]["off"]
        expected = (m[seed]["pod_seconds"] + 0.10 * m[seed]["train_step_seconds"]) / 3600
        assert row["pod_hours"] == pytest.approx(expected, rel=1e-9)
        assert row["padded_tokens_ratio_vs_v08f_measured"] == 1.0


@pytest.mark.parametrize("place", ["simgen/v08/x.jsonl", "simgen/v07/x.jsonl", "v09/n9g/x.jsonl", "reports/v08/x.json", "t9/x", "v08/gates/x"])
def test_writing_is_refused_in_earlier_inputs_sealed_sets_and_v09b(place):
    with pytest.raises(SystemExit):
        a09().refuse_write(REPO / "artifacts" / place)


def test_writing_is_allowed_in_the_v09_input_folder():
    a09().refuse_write(REPO / "artifacts" / "simgen" / "v09" / "train-v09n.jsonl")


@pytest.mark.skipif(not (REPO / "artifacts" / "simgen" / "v09" / "train-v09.assembly.json").is_file(), reason="built by scripts/assemble_v09.py all")
def test_the_built_assembly_passed():
    out = REPO / "artifacts" / "simgen" / "v09"
    summary = json.loads((out / "train-v09.assembly.json").read_text(encoding="utf-8"))
    assert summary["pass"] and summary["check"] and summary["armscan"] and summary["sealed"] and summary["exposure"]
    manifest = json.loads((out / "train-v09n.manifest.json").read_text(encoding="utf-8"))
    assert 0.06 <= manifest["share_of_records"] <= 0.08 and manifest["out"]["records"] == 116890
    assert summary["files"]["train-v09n.jsonl"]["sha256"] == manifest["sha256"]
