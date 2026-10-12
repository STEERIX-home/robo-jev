"""Task T9 — T9 is sealed: refused in any training input (trainer, mix assembly, unified overlap check), listed in every sealed index,
and never sent to any API (bench runs, the narration and E1 rewriters)."""

import importlib.util
import json
import sys
from pathlib import Path

import pytest
from helpers import FIXTURES, REPO
from test_v02_mix import _probe_and_sealed, _write

from robo_jev.data.sealed import NEVER_SENT_SETS, SEALED_RECORD_FILES, SEALED_RECORD_SETS, refuse_sealed_records, sealed_record_set
from robo_jev.decisions.records import read_records

REQUESTS = FIXTURES / "decisions" / "requests.jsonl"
TRAIN_CONFIG = REPO / "configs" / "train" / "decisions-v0-lora-h100.yaml"


def _script(name: str):
    spec = importlib.util.spec_from_file_location(f"t9sealed_{Path(name).stem}", REPO / "scripts" / name)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _t9_like(record: dict, *, mark: str = "meta") -> dict:
    out = json.loads(json.dumps(record))
    if mark == "meta":
        out["meta"] = {**(out.get("meta") or {}), "set": "T9"}
    else:
        out["id"] = "t9-roll-0001a"
    return out


def test_a_record_of_t9_is_recognised_by_its_set_or_its_id_and_refused():
    base = read_records(REQUESTS)[0]
    assert "T9" in SEALED_RECORD_SETS and "T9" in NEVER_SENT_SETS and SEALED_RECORD_FILES["T9"] == "artifacts/t9/T9.jsonl"
    assert sealed_record_set(base) is None
    assert sealed_record_set(_t9_like(base)) == "T9" and sealed_record_set(_t9_like(base, mark="id")) == "T9"
    refuse_sealed_records([base], where="x")
    with pytest.raises(ValueError, match="sealed outside test"):
        refuse_sealed_records([base, _t9_like(base)], where="training input")


def test_the_trainer_refuses_a_training_input_that_holds_a_t9_record():
    from robo_jev.decisions.train import DecisionTrainer

    records = read_records(REQUESTS)
    with pytest.raises(ValueError, match="sealed outside test"):
        DecisionTrainer({}, records=[*records, _t9_like(records[0], mark="id")], log=None)


def _extra_part(tmp_path: Path, name: str, records: list[dict]) -> Path:
    """Valid train-split records for an ``--extra`` part (fresh ids, as the MuJoCo set's)."""
    out = []
    for i, record in enumerate(records):
        record = json.loads(json.dumps(record))
        record["id"] = record["id"] if str(record["id"]).startswith("t9-") else f"x{name}-{i:04d}"
        record["meta"] = {**(record.get("meta") or {}), "split": "train", "language": "en"}
        out.append(record)
    return _write(tmp_path / f"{name}.jsonl", out)


def test_the_mix_assembly_refuses_a_t9_record_in_any_part_and_any_state_shared_with_a_sealed_file(tmp_path):
    probe, sealed = _probe_and_sealed(tmp_path)
    records = read_records(REQUESTS)
    script = _script("convert_single_probe.py")
    common = ["mix", "--base", str(REQUESTS), "--probe", str(probe), "--sealed", str(sealed), "--train-config", str(TRAIN_CONFIG)]
    poisoned = _extra_part(tmp_path, "marked", [records[0], _t9_like(records[1])])
    with pytest.raises(SystemExit, match="sealed outside test"):
        script.main([*common, "--extra", f"mujoco={poisoned}", "--out", str(tmp_path / "a.jsonl")])
    by_id = _extra_part(tmp_path, "byid", [records[0], _t9_like(records[1], mark="id")])
    with pytest.raises(SystemExit, match="sealed outside test"):
        script.main([*common, "--extra", f"mujoco={by_id}", "--out", str(tmp_path / "a2.jsonl")])
    # an unmarked copy of a sealed state (another id, no T9 meta) is caught through a further --sealed file
    sealed_t9 = _write(tmp_path / "T9.jsonl", [dict(records[1], id="t9-roll-0001a", meta={"set": "T9"})])
    twin = _extra_part(tmp_path, "twin", [records[0], records[1]])
    with pytest.raises(SystemExit, match="shares ids or whole states"):
        script.main(["mix", "--base", str(REQUESTS), "--probe", str(probe), "--sealed", str(sealed), str(sealed_t9), "--train-config",
                     str(TRAIN_CONFIG), "--extra", f"mujoco={twin}", "--out", str(tmp_path / "b.jsonl")])
    # a clean assembly records the outside check in its manifest (the registered T9 file is checked whenever it exists)
    out = tmp_path / "c.jsonl"
    assert script.main([*common, "--out", str(out)]) == 0
    manifest = json.loads((tmp_path / "c.manifest.json").read_text(encoding="utf-8"))
    outside = manifest["sealed_outside"]
    assert outside["files"].get("T9") is not None or "T9" in outside["absent"]
    if (REPO / "artifacts" / "t9" / "T9.jsonl").is_file():
        assert all(v == {"ids": 0, "states": 0} for v in outside["files"]["T9"]["shared"].values())


def test_the_unified_overlap_check_lists_t9_refuses_its_directory_and_flags_a_marked_record(tmp_path, monkeypatch):
    script = _script("system1_v03.py")
    assert script.SEALED["T9"] == REPO / "artifacts" / "t9" / "T9.jsonl"
    assert any(Path(p) == REPO / "artifacts" / "t9" for p in script.SEALED_LOCATIONS)
    with pytest.raises(SystemExit, match="sealed location"):
        script.refuse_sealed_path(REPO / "artifacts" / "t9" / "T9.jsonl")
    records = read_records(REQUESTS)
    known = [dict(r, id=f"mj-{i:04d}", meta={"set": "v03b-mujoco-train", "split": "train"}) for i, r in enumerate(records)]  # a known part
    sealed_t1 = _write(tmp_path / "T1.jsonl", [dict(records[0], id="v0b-T1-000000")])
    monkeypatch.setattr(script, "SEALED", {"T1": sealed_t1})
    clean = _write(tmp_path / "clean.jsonl", known[1:4])
    assert script.main(["overlap", "--input", str(clean), "--out", str(tmp_path / "clean.json")]) == 0
    marked = _write(tmp_path / "marked.jsonl", [*known[1:3], _t9_like(known[3], mark="id")])
    assert script.main(["overlap", "--input", str(marked), "--out", str(tmp_path / "marked.json")]) == 2
    report = json.loads((tmp_path / "marked.json").read_text(encoding="utf-8"))
    assert report["parts"]["mujoco"].get("T9:marked") == 1


def test_every_sealed_index_lists_t9():
    from robo_jev.simgen.narration import SealedIndex

    assert "T9" in SealedIndex.OUTSIDE
    assert _script("narrate_v03.py").SEALED["T9"] == REPO / "artifacts" / "t9" / "T9.jsonl"
    e1 = _script("e1.py")
    assert e1.SEALED_FILES["T9"] == REPO / "artifacts" / "t9" / "T9.jsonl" and "T9" in e1.OUTSIDE_SETS
    assert _script("convert_general_sets.py").SEALED["T9"] == "artifacts/t9/T9.jsonl"


def test_t9_is_never_sent_to_any_api():
    from robo_jev.e1.rewrite import SealedIndex, SealedTextError, assert_sendable

    record = read_records(REQUESTS)[0]
    index = SealedIndex(set(), set(), {})
    assert_sendable(record, [], index)
    for marked in (_t9_like(record), _t9_like(record, mark="id")):
        with pytest.raises(SealedTextError, match="T9"):
            assert_sendable(marked, [], index)
    bench = _script("bench.py")
    bench.refuse_never_sent({"T2": [record]})
    with pytest.raises(SystemExit, match="never sent"):
        bench.refuse_never_sent({"T2": [record, _t9_like(record)]})
    with pytest.raises(SystemExit, match="never sent"):
        bench.refuse_never_sent({"T9": [record]})


@pytest.mark.parametrize("job_path", sorted((REPO / "configs" / "cloud").glob("*.yaml")), ids=lambda p: p.stem)
def test_no_shipped_job_trains_on_t9(job_path):
    from test_cloud_jobs_v01 import _script as job_script
    from test_cloud_jobs_v01 import _steps

    from robo_jev.decisions.train import resolve_config

    for name, argv in _steps(job_path)[1]:
        if argv[1:2] != ["scripts/train_decisions.py"]:
            continue
        script = job_script("train_decisions.py")
        args = script.build_parser().parse_args(argv[2:])
        train_input = str(resolve_config(script._read_config(REPO / args.config, args.set))["train_input"])
        assert "artifacts/t9" not in train_input and "T9" not in Path(train_input).name, f"{job_path.name}:{name}"
