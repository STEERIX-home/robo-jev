"""v0.2 training inputs (Task v0.2 §1-3): physics file + d1 train mix + a prefix of the general-reading file.

`scripts/convert_single_probe.py mix --general … --general-share …` appends, after the base and probe files (byte for byte), the
shortest prefix of the general file whose train examples — counted with the trainer's own dev split and q — reach the target share of
all train examples per epoch. Small hand-made files pin the rule and the refusals; when the built inputs
(``artifacts/simgen/v02/train-v02-{a,b}.jsonl``) exist, their manifests, byte layout, shares and the trainer's dev set are checked.
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

from robo_jev.data import general_sets as gs
from robo_jev.decisions.examples import examples_per_epoch, split_dev
from robo_jev.decisions.records import file_sha256, read_records
from robo_jev.decisions.single_probe import G1_SPLITS, MIX_SPLITS, convert_records

REQUESTS = FIXTURES / "decisions" / "requests.jsonl"
TRAIN_CONFIG = REPO / "configs" / "train" / "decisions-v0-lora-h100.yaml"
V02 = REPO / "artifacts" / "simgen" / "v02"
DOSES = {"a": 0.10, "b": 0.25}


@functools.lru_cache(maxsize=1)
def script():
    spec = importlib.util.spec_from_file_location("convert_single_probe_v02", REPO / "scripts" / "convert_single_probe.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write(path: Path, records: list[dict]) -> Path:
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records), encoding="utf-8")
    return path


def _probe_and_sealed(tmp_path: Path) -> tuple[Path, Path]:
    from test_single_probe import boolean, choice, judgment, mixed_record, single

    train = judgment([boolean("q_b"), choice("q_c", [("c1", "o1", "o1"), ("c2", "none", None)])], [single("q_b", True), single("q_c", "c1")],
                     rid="rules-0009-0", split="train", origin="rules/x/0009", domain="rules")  # fmt: skip
    probe = _write(tmp_path / "G1-train-mix.jsonl", convert_records([train], splits=MIX_SPLITS)[0])
    sealed = _write(tmp_path / "G1.jsonl", convert_records([mixed_record()], splits=G1_SPLITS)[0])
    return probe, sealed


def _general(tmp_path: Path, n_rows: int = 30) -> Path:
    """A small general file (synthetic rows of every source, built as the real one) with a manifest recording a zero overlap."""
    rows = {}
    for name in gs.SOURCES:
        rows[name] = []
        for i in range(n_rows):
            if name == "commonsense_qa":
                rows[name].append({"question": f"Question number {i} about daily things?", "choices": {"label": list("ABCDE"),
                                   "text": [f"answer {i} {k}" for k in "abcde"]}, "answerKey": "C"})  # fmt: skip
            elif name == "hellaswag":
                rows[name].append({"activity_label": "Cooking", "ctx": f"Step {i}: a man cuts bread and", "endings": [f"ending {i} {k}" for k in "abcd"], "label": "2"})
            elif name == "social_i_qa":
                rows[name].append({"context": f"Person {i} helped a friend move.", "question": "Why?", "answerA": f"kind {i}", "answerB": f"bored {i}",
                                   "answerC": f"paid {i}", "label": "1"})  # fmt: skip
            elif name == "qasc":
                rows[name].append({"question": f"What does thing {i} need?", "choices": {"label": list("ABCDEFGH"), "text": [f"item {i} {k}" for k in "abcdefgh"]},
                                   "answerKey": "D", "fact1": f"fact one {i}", "fact2": f"fact two {i}"})  # fmt: skip
            else:
                rows[name].append({"context": f"Story {i}: we went out.", "question": "What happened?", **{f"answer{k}": f"option {i} {k}" for k in range(4)}, "label": 3})
    records, report = gs.build_general(rows, per_source=n_rows, seed=1, sealed={})
    path = _write(tmp_path / "general-train.jsonl", records)
    manifest = {"file": {"sha256": file_sha256(path), "records": len(records)}, "licenses": {n: s["license"] for n, s in gs.SOURCES.items()},
                "overlap_check": {"overlap_after_build": {"G1": 0, "G2": 0}}}  # fmt: skip
    (tmp_path / "general-train.manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return path


def test_the_general_prefix_is_the_shortest_that_reaches_the_target_share(tmp_path):
    probe, sealed = _probe_and_sealed(tmp_path)
    general = _general(tmp_path)
    out = tmp_path / "mix.jsonl"
    argv = ["mix", "--base", str(REQUESTS), "--probe", str(probe), "--sealed", str(sealed), "--general", str(general), "--general-share", "0.4",
            "--train-config", str(TRAIN_CONFIG), "--out", str(out)]  # fmt: skip
    assert script().main(argv) == 0
    manifest = json.loads((tmp_path / "mix.manifest.json").read_text(encoding="utf-8"))
    n = manifest["general"]["records_used"]
    lines = general.read_bytes().splitlines(keepends=True)
    assert out.read_bytes() == REQUESTS.read_bytes() + probe.read_bytes() + b"".join(lines[:n])  # base, probe, then the general prefix
    epoch = manifest["examples_per_epoch"]
    assert epoch["general_share"] >= 0.4 and epoch["total"] == epoch["base"] + epoch["probe"] + epoch["general"]
    import yaml

    from robo_jev.decisions.train import resolve_config

    config = resolve_config(yaml.safe_load(TRAIN_CONFIG.read_text(encoding="utf-8")))
    shorter = read_records(general)[: n - 1]
    train_short, _ = split_dev(shorter, fraction=config["dev_fraction"], salt=config["dev_salt"])
    short = examples_per_epoch(train_short, q=2)
    from fractions import Fraction

    assert Fraction(short, epoch["base"] + epoch["probe"] + short) < Fraction(2, 5)  # one record fewer would miss the target
    assert manifest["general"]["sha256"] == file_sha256(general) and sum(manifest["general"]["records_by_source"].values()) == n
    assert manifest["out"]["records"] == len(read_records(REQUESTS)) + 1 + n
    # the trainer reads the whole file; its dev set is the base file's dev records in file order
    from robo_jev.decisions.train import load_training_records

    mixed, check = load_training_records(out)
    assert check["invalid"] == 0 and len(mixed) == manifest["out"]["records"]
    base_dev = split_dev(read_records(REQUESTS), fraction=config["dev_fraction"], salt=config["dev_salt"])[1]
    assert [r["id"] for r in split_dev(mixed, fraction=config["dev_fraction"], salt=config["dev_salt"])[1]][: len(base_dev)] == [r["id"] for r in base_dev]


def test_the_general_part_is_refused_without_a_matching_manifest_a_share_or_train_records(tmp_path):
    probe, sealed = _probe_and_sealed(tmp_path)
    general = _general(tmp_path)
    out = tmp_path / "mix.jsonl"
    common = ["mix", "--base", str(REQUESTS), "--probe", str(probe), "--sealed", str(sealed), "--train-config", str(TRAIN_CONFIG), "--out", str(out)]
    with pytest.raises(SystemExit, match="general-share"):
        script().main([*common, "--general", str(general)])
    with pytest.raises(SystemExit, match="too few records"):
        script().main([*common, "--general", str(general), "--general-share", "0.99"])
    records = read_records(general)
    wrong = copy.deepcopy(records)
    wrong[0]["meta"]["split"] = "test"
    bad = _write(tmp_path / "bad" / "general-train.jsonl", wrong) if (tmp_path / "bad").mkdir() is None else None
    with pytest.raises(SystemExit, match="train-split records"):
        script().main([*common, "--general", str(bad), "--general-share", "0.2"])
    stale = tmp_path / "stale"
    stale.mkdir()
    copy_path = _write(stale / "general-train.jsonl", records[:-1])
    (stale / "general-train.manifest.json").write_text((tmp_path / "general-train.manifest.json").read_text(encoding="utf-8"), encoding="utf-8")
    with pytest.raises(SystemExit, match="manifest"):
        script().main([*common, "--general", str(copy_path), "--general-share", "0.2"])
    assert not out.exists()


# --------------------------------------------------------------------------
# the built inputs
# --------------------------------------------------------------------------


@pytest.mark.parametrize("dose", sorted(DOSES))
def test_the_built_training_inputs_hold_physics_then_d1_then_the_general_prefix_at_their_dose(dose):
    path = V02 / f"train-v02-{dose}.jsonl"
    manifest_path = V02 / f"train-v02-{dose}.manifest.json"
    if not manifest_path.is_file():
        pytest.skip(f"{path} not built yet (scripts/convert_single_probe.py mix --general …)")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["out"]["sha256"] == file_sha256(path) and manifest["git"]["dirty"] is False
    assert manifest["base"]["path"].endswith("artifacts/simgen/v02/train-50k.jsonl") and manifest["probe"]["path"].endswith("probes/G1-train-mix.jsonl")
    assert manifest["base"]["sha256"] == json.loads((V02 / "train-50k.manifest.json").read_text(encoding="utf-8"))["sha256"]
    assert manifest["general"]["sha256"] == json.loads((V02 / "general-train.manifest.json").read_text(encoding="utf-8"))["file"]["sha256"]
    epoch = manifest["examples_per_epoch"]
    assert DOSES[dose] <= epoch["general_share"] < DOSES[dose] + 0.001, epoch
    assert manifest["split"]["base_dev"] >= manifest["train_config"]["dev_max_records"], "the dev set must come from the physics file only"
    assert manifest["split"]["probe_dev_used"] == 0 and manifest["split"]["general_dev_used"] == 0
    by_source = manifest["general"]["records_by_source"]
    assert set(by_source) == set(gs.SOURCES) and max(by_source.values()) - min(by_source.values()) <= 1
    with path.open("rb") as out, (V02 / "train-50k.jsonl").open("rb") as base:
        assert out.read(1 << 20) == base.read(1 << 20)  # the physics file comes first, byte for byte
