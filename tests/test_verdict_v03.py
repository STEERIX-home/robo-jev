"""The v0.3 rule helper (``scripts/verdict_v03.py``, Task v0.3c §1-5): the pre-declared rule applied mechanically.

The thresholds are pinned at their boundaries on hand-made readings. With the real files present: v0.1-mix's own predictions put in
the v0.3 slot reproduce the System-1 scorecard's published numbers (T8b balanced band −0.022 [−0.053, +0.011] vs the base, severe errors
0.278 vs 0.155, direction 0.565; G2 −0.061 vs the pod base) and fail O and G while passing H against itself; the labels as answers pass;
and a prediction file scored on another set file is refused.
"""

from __future__ import annotations

import functools
import importlib.util
import json
import sys
from pathlib import Path

import pytest
from helpers import REPO

from robo_jev.bench.system1 import oracle_predictions
from robo_jev.decisions.records import file_sha256, read_records


@functools.lru_cache(maxsize=1)
def script():
    spec = importlib.util.spec_from_file_location("verdict_v03_script", REPO / "scripts" / "verdict_v03.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


GOOD_O = {"band_low": 0.01, "severe": 0.10, "severe_base": 0.155, "direction": 0.75}
GOOD_H = {"band_low": -0.04, "severe": 0.02, "severe_mix": 0.02}
GOOD_G = {"diff": -0.01, "low": -0.03}


@pytest.mark.parametrize("part, change, passes", [
    ("O", {"band_low": 0.0}, False), ("O", {"band_low": 1e-9}, True), ("O", {"severe": 0.155}, True), ("O", {"severe": 0.1551}, False),
    ("O", {"direction": 0.70}, True), ("O", {"direction": 0.6999}, False), ("O", {"band_low": None}, False),
    ("H", {"band_low": -0.05}, False), ("H", {"band_low": -0.0499}, True), ("H", {"severe": 0.03}, True), ("H", {"severe": 0.0301}, False),
    ("G", {"diff": -0.02}, True), ("G", {"diff": -0.0201}, False), ("G", {"low": -0.045}, False), ("G", {"low": -0.0449}, True),
])  # fmt: skip
def test_the_rule_thresholds_hold_at_their_boundaries(part, change, passes):
    readings = {"O": dict(GOOD_O), "H": dict(GOOD_H), "G": dict(GOOD_G)}
    readings[part].update(change)
    result = script().rule_checks(readings["O"], readings["H"], readings["G"])
    assert result[part]["pass"] is passes
    assert result["verdict"] == ("PASSES" if passes else "DOES NOT PASS")
    assert all(result[p]["pass"] for p in "OHG" if p != part)


MIX = REPO / "artifacts" / "cloud" / "runs" / "v01-mix-01-fetched" / "reports"
T8 = REPO / "artifacts" / "reports" / "t8"
NEEDED = [MIX / f"after-v01mix-2b-{s}.predictions.jsonl" for s in ("T2", "T3", "T4", "G2")] + [
    T8 / "T8b-v01mix-2b.predictions.jsonl", T8 / "T8b-base-2b.predictions.jsonl", script().BASE_G2_POD, *script().SET_PATHS.values()]


def _need_files():
    missing = [str(p) for p in NEEDED if not Path(p).is_file()]
    if missing:
        pytest.skip(f"real prediction / set files absent: {missing[:2]}")


def _stand_in(tmp_path: Path) -> Path:
    """v0.1-mix's own predictions under the v0.3 names (symlinks, with their diagnosis summaries)."""
    directory = tmp_path / "reports"
    directory.mkdir()
    sources = {s: MIX / f"after-v01mix-2b-{s}" for s in ("T2", "T3", "T4", "G2")} | {"T8b": T8 / "T8b-v01mix-2b"}
    for s, stem in sources.items():
        for suffix in (".predictions.jsonl", ".json"):
            (directory / f"after-v03-2b-{s}{suffix}").symlink_to(Path(str(stem) + suffix))
    return directory


def test_v01_mix_in_the_v03_slot_reproduces_the_published_system1_numbers_and_fails_o_and_g(tmp_path):
    _need_files()
    out = tmp_path / "verdict.json"
    assert script().main(["--v03-dir", str(_stand_in(tmp_path)), "--out", str(out), "--n-boot", "2000"]) == 0
    result = json.loads(out.read_text(encoding="utf-8"))
    o, h, g = result["readings"]["O"], result["readings"]["H"], result["readings"]["G"]
    assert o["band_balanced"]["value"] == pytest.approx(-0.022, abs=0.001) and o["band_balanced"]["low"] == pytest.approx(-0.053, abs=0.002)
    assert o["severe"] == pytest.approx(0.278, abs=0.001) and o["severe_base"] == pytest.approx(0.155, abs=0.001)
    assert o["direction_value"] == pytest.approx(0.565, abs=0.001) and o["direction"]["n"] == 336
    assert g["diff"] == pytest.approx(-0.061, abs=0.001) and g["low"] == pytest.approx(-0.086, abs=0.002)
    assert h["band_balanced"]["value"] == 0.0 and h["severe"] == h["severe_mix"]
    assert result["rule"]["O"]["pass"] is False and result["rule"]["H"]["pass"] is True and result["rule"]["G"]["pass"] is False
    assert result["rule"]["verdict"] == "DOES NOT PASS" and all(src["same_set"] for src in result["sources"].values())
    assert all(src["coverage"] == 1.0 for src in result["sources"].values())
    assert out.with_suffix(".md").read_text(encoding="utf-8").startswith("# v0.3 pre-declared rule — DOES NOT PASS")


def _oracle_dir(tmp_path: Path) -> Path:
    directory = tmp_path / "oracle"
    directory.mkdir()
    conditions = {k: v for k, v in read_records(T8 / "T8b-base-2b.predictions.jsonl")[0].items() if k in script().SCORING_KEYS}
    for s in ("T2", "T3", "T4", "G2", "T8b"):
        path = script().SET_PATHS[s]
        predictions = [p | conditions for p in oracle_predictions(read_records(path))]
        (directory / f"after-v03-2b-{s}.predictions.jsonl").write_text("".join(json.dumps(p) + "\n" for p in predictions), encoding="utf-8")
        (directory / f"after-v03-2b-{s}.json").write_text(json.dumps({"config": {"input_sha256": file_sha256(path)}}), encoding="utf-8")
    return directory


def test_the_labels_as_answers_pass_every_part(tmp_path):
    _need_files()
    out = tmp_path / "oracle.json"
    assert script().main(["--v03-dir", str(_oracle_dir(tmp_path)), "--out", str(out), "--n-boot", "200"]) == 0
    result = json.loads(out.read_text(encoding="utf-8"))
    assert result["rule"]["verdict"] == "PASSES", result["rule"]
    assert result["readings"]["O"]["severe"] == 0.0 and result["readings"]["O"]["direction_value"] == 1.0


def test_a_prediction_file_scored_on_another_set_file_is_refused(tmp_path):
    _need_files()
    directory = _oracle_dir(tmp_path)
    (directory / "after-v03-2b-T8b.json").write_text(json.dumps({"config": {"input_sha256": "0" * 64}}), encoding="utf-8")
    with pytest.raises(SystemExit, match="scored on another file"):
        script().main(["--v03-dir", str(directory), "--out", str(tmp_path / "v.json"), "--n-boot", "50"])
    assert script().main(["--v03-dir", str(directory), "--out", str(tmp_path / "v.json"), "--n-boot", "50", "--allow-mismatch"]) == 0
    with pytest.raises(SystemExit, match="missing prediction files"):
        script().main(["--v03-dir", str(tmp_path / "nowhere"), "--out", str(tmp_path / "w.json")])
