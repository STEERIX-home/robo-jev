"""v0.3 training input (Task v0.3c §1-3): engine set (with System-1 questions) + MuJoCo set + d1 train mix + general reading.

``scripts/convert_single_probe.py mix --extra NAME=PATH`` places extra training parts after the base file (byte for byte), counts them in
the shares and the token estimate, and sizes the general prefix against all the other parts. When the general file cannot reach its
share alone, ``--general-repeat-if-short`` appends a recorded second pass (ids ``<id>~r2``) and ``--general-take-all-if-short`` takes the
whole file and records the share reached. ``scripts/system1_v03.py overlap`` checks every record against every sealed set. Small
hand-made files pin the rules and refusals; the built input is checked when it exists.
"""

from __future__ import annotations

import copy
import functools
import importlib.util
import json
import sys
from pathlib import Path

import pytest
from helpers import FIXTURES, REPO
from test_v02_mix import _general, _probe_and_sealed, _write

from robo_jev.decisions.examples import split_dev
from robo_jev.decisions.records import file_sha256, read_records
from robo_jev.decisions.train import load_training_records

REQUESTS = FIXTURES / "decisions" / "requests.jsonl"
TRAIN_CONFIG = REPO / "configs" / "train" / "decisions-v0-lora-h100.yaml"
V03 = REPO / "artifacts" / "simgen" / "v03"


@functools.lru_cache(maxsize=None)
def _load(name: str):
    spec = importlib.util.spec_from_file_location(f"v03_{Path(name).stem}", REPO / "scripts" / name)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _config() -> dict:
    import yaml

    from robo_jev.decisions.train import resolve_config

    return resolve_config(yaml.safe_load(TRAIN_CONFIG.read_text(encoding="utf-8")))


def _mujoco_like(tmp_path: Path, n: int = 40, *, split: str = "train", manifest: bool = True) -> Path:
    """Valid train-split records shaped like the MuJoCo set (meta.set v03b-mujoco-train), with a manifest recording the file's sha256."""
    base = read_records(REQUESTS)
    out = []
    for i in range(n):
        record = copy.deepcopy(base[i % len(base)])
        record["id"] = f"mj-{i:04d}"
        record["meta"] = {"set": "v03b-mujoco-train", "split": split, "language": "en"}
        out.append(record)
    path = _write(tmp_path / "mujoco-train.jsonl", out)
    if manifest:
        (tmp_path / "mujoco-train.manifest.json").write_text(json.dumps({"sha256": file_sha256(path)}), encoding="utf-8")
    return path


def _common(tmp_path: Path, probe: Path, sealed: Path, out: Path) -> list[str]:
    return ["mix", "--base", str(REQUESTS), "--probe", str(probe), "--sealed", str(sealed), "--train-config", str(TRAIN_CONFIG), "--out", str(out)]


def test_an_extra_part_goes_after_the_base_and_counts_in_the_shares_and_the_general_prefix(tmp_path):
    probe, sealed = _probe_and_sealed(tmp_path)
    general = _general(tmp_path)
    mujoco = _mujoco_like(tmp_path)
    out = tmp_path / "train-v03.jsonl"
    assert _load("convert_single_probe.py").main([*_common(tmp_path, probe, sealed, out), "--extra", f"mujoco={mujoco}", "--general", str(general),
                                                  "--general-share", "0.25"]) == 0  # fmt: skip
    manifest = json.loads((tmp_path / "train-v03.manifest.json").read_text(encoding="utf-8"))
    n = manifest["general"]["records_used"]
    lines = general.read_bytes().splitlines(keepends=True)
    assert out.read_bytes() == REQUESTS.read_bytes() + mujoco.read_bytes() + probe.read_bytes() + b"".join(lines[:n])
    epoch = manifest["examples_per_epoch"]
    assert epoch["total"] == epoch["base"] + epoch["mujoco"] + epoch["probe"] + epoch["general"] and epoch["general_share"] >= 0.25
    assert manifest["general"]["target_reached"] is True and manifest["general"]["second_pass"] is None
    config = _config()
    train_mj, dev_mj = split_dev(read_records(mujoco), fraction=config["dev_fraction"], salt=config["dev_salt"])
    (extra,) = manifest["extra"]
    assert extra["name"] == "mujoco" and extra["sha256"] == file_sha256(mujoco) and extra["train"] == len(train_mj) and extra["dev"] == len(dev_mj)
    assert extra["manifest"]["file_sha256_recorded"] == file_sha256(mujoco)
    assert manifest["shares"]["examples"]["mujoco"] == pytest.approx(epoch["mujoco"] / epoch["total"]) and epoch["mujoco_share"] == pytest.approx(
        epoch["mujoco"] / epoch["total"])
    assert sum(manifest["shares"]["examples"].values()) == pytest.approx(1.0)
    assert "'mujoco'" in manifest["order"] and manifest["out"]["records"] == len(read_records(out))
    loaded, check = load_training_records(out)
    assert check["invalid"] == 0
    base_dev = split_dev(read_records(REQUESTS), fraction=config["dev_fraction"], salt=config["dev_salt"])[1]
    assert [r["id"] for r in split_dev(loaded, fraction=config["dev_fraction"], salt=config["dev_salt"])[1]][: len(base_dev)] == [r["id"] for r in base_dev]


def test_without_extras_the_output_is_the_v02_layout(tmp_path):
    probe, sealed = _probe_and_sealed(tmp_path)
    out = tmp_path / "mix.jsonl"
    assert _load("convert_single_probe.py").main(_common(tmp_path, probe, sealed, out)) == 0
    assert out.read_bytes() == REQUESTS.read_bytes() + probe.read_bytes()
    manifest = json.loads((tmp_path / "mix.manifest.json").read_text(encoding="utf-8"))
    assert "extra" not in manifest and "shares" not in manifest


@pytest.mark.parametrize("case, needle", [
    ("split", "train-split records only"), ("clash", "appear in another part"), ("stale", "manifest records sha256"),
    ("reserved", "reserved or repeated"), ("spec", "NAME=PATH"), ("invalid", "invalid records"),
])  # fmt: skip
def test_extra_parts_are_refused_when_not_train_clashing_stale_or_badly_named(tmp_path, case, needle):
    probe, sealed = _probe_and_sealed(tmp_path)
    out = tmp_path / "x.jsonl"
    if case == "split":
        spec = f"mujoco={_mujoco_like(tmp_path, split='test')}"
    elif case == "clash":
        records = read_records(_mujoco_like(tmp_path))
        records[0]["id"] = read_records(REQUESTS)[0]["id"]
        spec = f"mujoco={_write(tmp_path / 'mujoco-train.jsonl', records)}"
    elif case == "stale":
        path = _mujoco_like(tmp_path)
        records = read_records(path)
        _write(path, records[:-1])
        spec = f"mujoco={path}"
    elif case == "reserved":
        spec = f"general={_mujoco_like(tmp_path)}"
    elif case == "spec":
        spec = str(_mujoco_like(tmp_path))
    else:
        records = read_records(_mujoco_like(tmp_path, manifest=False))
        records[1]["labels"] = {}
        spec = f"mujoco={_write(tmp_path / 'mujoco-train.jsonl', records)}"
    with pytest.raises(SystemExit, match=needle):
        _load("convert_single_probe.py").main([*_common(tmp_path, probe, sealed, out), "--extra", spec])
    assert not out.exists()


def test_a_short_general_file_is_refused_or_taken_whole_or_given_a_recorded_second_pass(tmp_path):
    probe, sealed = _probe_and_sealed(tmp_path)
    general = _general(tmp_path, n_rows=4)  # 20 records — far below 40 % of the other parts
    mujoco = _mujoco_like(tmp_path)
    common = [*_common(tmp_path, probe, sealed, tmp_path / "out.jsonl"), "--extra", f"mujoco={mujoco}", "--general", str(general), "--general-share", "0.3"]
    script = _load("convert_single_probe.py")
    with pytest.raises(SystemExit, match="too few records"):
        script.main(common)
    with pytest.raises(SystemExit, match="exclude each other"):
        script.main([*common, "--general-take-all-if-short", "--general-repeat-if-short"])
    assert script.main([*common, "--general-take-all-if-short"]) == 0
    whole = json.loads((tmp_path / "out.manifest.json").read_text(encoding="utf-8"))
    assert whole["general"]["records_used"] == 20 and whole["general"]["target_reached"] is False and whole["general"]["second_pass"] is None
    (tmp_path / "big").mkdir()
    general_big = _general(tmp_path / "big", n_rows=12)  # 60 records: one pass misses 45 %, a second pass reaches it
    repeat = [*_common(tmp_path, probe, sealed, tmp_path / "rep.jsonl"), "--extra", f"mujoco={mujoco}", "--general", str(general_big),
              "--general-share", "0.45"]  # fmt: skip
    with pytest.raises(SystemExit, match="too few records"):
        script.main(repeat)
    assert script.main([*repeat, "--general-repeat-if-short"]) == 0
    manifest = json.loads((tmp_path / "rep.manifest.json").read_text(encoding="utf-8"))
    loaded, check = load_training_records(tmp_path / "rep.jsonl")
    assert check["invalid"] == 0 and manifest["general"]["target_reached"] is True and manifest["examples_per_epoch"]["general_share"] >= 0.45
    repeated = [r for r in loaded if str(r["id"]).endswith("~r2")]
    assert repeated and manifest["general"]["second_pass"]["records"] == len(repeated) and manifest["general"]["records_used"] == 60 + len(repeated)
    originals = {r["id"]: r for r in read_records(general_big)}
    for r in repeated:
        source = originals[r["meta"]["repeat_of"]]
        assert r["request"] == source["request"] and r["labels"] == source["labels"] and r["id"] == f"{source['id']}~r2"
    assert [r["meta"]["repeat_of"] for r in repeated] == [r["id"] for r in read_records(general_big)][: len(repeated)], "the second pass is a prefix"


# --------------------------------------------------------------------------
# the unified sealed-overlap check
# --------------------------------------------------------------------------


def test_the_overlap_check_finds_a_sealed_state_request_id_or_outside_text_unit_and_passes_a_clean_file(tmp_path, monkeypatch):
    script = _load("system1_v03.py")
    records = read_records(_mujoco_like(tmp_path, n=8, manifest=False))  # a known part (records of an unknown part count as hits)
    sealed_t1 = _write(tmp_path / "T1.jsonl", [dict(records[0], id="v0b-T1-000000")])
    outside = copy.deepcopy(records[1])
    outside["id"] = "g2-0001"
    outside["request"] = {"state": "passage: The quick brown fox jumps over the lazy dog near the river bank today.",
                          "questions": {"q": {"type": "noul", "instructions": "Is it true?"}}}
    outside["labels"] = {"q": {"type": "noul", "p_yes": 1.0}}
    sealed_g2 = _write(tmp_path / "G2.jsonl", [outside])
    monkeypatch.setattr(script, "SEALED", {"T1": sealed_t1, "G2": sealed_g2})
    clean = _write(tmp_path / "clean.jsonl", records[2:])
    assert script.main(["overlap", "--input", str(clean), "--out", str(tmp_path / "clean.json"), "--allow-missing"]) == 0
    report = json.loads((tmp_path / "clean.json").read_text(encoding="utf-8"))
    assert report["total_hits"] == 0 and report["input"]["records"] == len(records) - 2 and report["parts"]["mujoco"]["records"] == 6
    leaked = copy.deepcopy(records[3])
    leaked["request"]["state"] = "Context line.\npassage: The quick brown fox jumps over the lazy dog near the river bank today."
    dirty = _write(tmp_path / "dirty.jsonl", [records[0], leaked, *records[4:]])
    assert script.main(["overlap", "--input", str(dirty), "--out", str(tmp_path / "dirty.json"), "--allow-missing"]) == 2
    report = json.loads((tmp_path / "dirty.json").read_text(encoding="utf-8"))
    hits = " ".join(sum(report["first_hits"].values(), []))
    assert "T1: state" in hits and "T1: request" in hits and "G2: 1 units" in hits
    with pytest.raises(SystemExit, match="missing"):
        monkeypatch.setattr(script, "SEALED", {"T1": sealed_t1, "T9": tmp_path / "absent.jsonl"})
        script.main(["overlap", "--input", str(clean), "--out", str(tmp_path / "x.json")])


# --------------------------------------------------------------------------
# the built input
# --------------------------------------------------------------------------


def test_the_built_v03_input_holds_engine_then_mujoco_then_d1_then_general_at_its_dose():
    path, manifest_path = V03 / "train-v03.jsonl", V03 / "train-v03.manifest.json"
    if not manifest_path.is_file():
        pytest.skip("artifacts/simgen/v03/train-v03.jsonl not built yet (scripts/convert_single_probe.py mix --extra mujoco=…)")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["out"]["sha256"] == file_sha256(path) and manifest["git"]["dirty"] is False
    engine = json.loads((V03 / "engine-s1.manifest.json").read_text(encoding="utf-8"))
    assert manifest["base"]["path"].endswith("artifacts/simgen/v03/engine-s1.jsonl") and manifest["base"]["sha256"] == engine["out"]["sha256"]
    (extra,) = manifest["extra"]
    assert extra["name"] == "mujoco" and extra["path"].endswith("artifacts/simgen/v03/mujoco-train.jsonl")
    assert extra["sha256"] == extra["manifest"]["file_sha256_recorded"] == file_sha256(V03 / "mujoco-train.jsonl")
    assert manifest["probe"]["path"].endswith("probes/G1-train-mix.jsonl") and manifest["sealed_check"]["overlap"] == {"origin_groups": 0, "source_records": 0, "base_ids": 0}
    general = json.loads((REPO / "artifacts" / "simgen" / "v02" / "general-train.manifest.json").read_text(encoding="utf-8"))
    assert manifest["general"]["sha256"] == general["file"]["sha256"] and manifest["general"]["target_reached"] is True
    epoch = manifest["examples_per_epoch"]
    assert 0.25 <= epoch["general_share"] < 0.251, epoch
    assert manifest["split"]["base_dev"] >= manifest["train_config"]["dev_max_records"], "the dev set must come from the engine file only"
    assert extra["dev_used"] == 0 and manifest["split"]["probe_dev_used"] == 0 and manifest["split"]["general_dev_used"] == 0
    with path.open("rb") as out, (V03 / "engine-s1.jsonl").open("rb") as base:
        assert out.read(1 << 20) == base.read(1 << 20)  # the engine file comes first, byte for byte
    overlap = json.loads((V03 / "train-v03.overlap.json").read_text(encoding="utf-8"))
    assert overlap["input"]["sha256"] == manifest["out"]["sha256"] and overlap["total_hits"] == 0 and not overlap["missing_sealed"]
    assert set(overlap["parts"]) == {"engine", "mujoco", "d1", "general"}
