"""v0.3 job definition (Task v0.3c §1-4) — `decisions-v03-h100`.

Same pattern as the v0.1 / v0.2 job tests: the job trains once with the v0.1-mix recipe (only the training input and the run name
differ), diagnoses its own checkpoint under the Spark / v0.1 scoring conditions — first the sets the pre-declared v0.3 rule reads (T8b,
G2, T2–T4), then T1, T5, T7, T8a, G1, T6 — uploads T8 as regular inputs (final now), and writes the prediction names the rule helper
reads. No model or data is loaded.
"""

from __future__ import annotations

import functools
import importlib.util
import json
import sys
from pathlib import Path

import pytest
from helpers import REPO

from robo_jev.decisions.train import resolve_config
from robo_jev.launch.jobs import expand, load_job, plan_job

JOB = REPO / "configs" / "cloud" / "decisions-v03-h100.yaml"
V01_MIX = REPO / "configs" / "cloud" / "decisions-v01-mix-h100.yaml"
TRAIN_INPUT = "artifacts/simgen/v03/train-v03.jsonl"
RUN_NAME = "v03-2b"
TESTS = {f"T{i}": f"artifacts/simgen/v0b/tests/T{i}.jsonl" for i in range(1, 8)}
SETS = {**TESTS, "G1": "artifacts/simgen/probes/G1.jsonl", "G2": "artifacts/simgen/probes/G2.jsonl", "T8a": "artifacts/t8/T8a.jsonl",
        "T8b": "artifacts/t8/T8b.jsonl"}  # fmt: skip
ORDER = ["T8b", "G2", "T2", "T3", "T4", "T1", "T5", "T7", "T8a", "G1", "T6"]
CONTEXT = {"python": "python", "uv": "uv", "repo": "/r", "run_dir": "/r/run", "runs": "/r/run/runs", "reports": "/r/run/reports", "run_id": "x"}


@functools.lru_cache(maxsize=None)
def _script(name: str):
    spec = importlib.util.spec_from_file_location(f"v03_job_{Path(name).stem}", REPO / "scripts" / name)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _steps(path: Path) -> tuple[dict, list[tuple[str, list[str]]]]:
    spec = load_job(path)
    job = spec["job"]
    return spec, [(step["name"], expand(step["argv"], {**CONTEXT, "model": job["model"], "step": step["name"]})) for step in job["setup"] + job["steps"]]


def _train(path: Path) -> dict:
    (argv,) = [argv for _, argv in _steps(path)[1] if argv[1:2] == ["scripts/train_decisions.py"]]
    args = _script("train_decisions.py").build_parser().parse_args(argv[2:])
    return resolve_config(_script("train_decisions.py")._read_config(REPO / args.config, args.set))


def _diagnoses(path: Path) -> list[tuple[str, object]]:
    return [(name, _script("diagnose_decisions.py")._parser().parse_args(argv[2:])) for name, argv in _steps(path)[1] if argv[1:2] == ["scripts/diagnose_decisions.py"]]


def test_the_v03_job_trains_once_with_the_v01_mix_recipe():
    config = _train(JOB)
    reference = _train(V01_MIX)
    assert config["train_input"] == TRAIN_INPUT and config["run_name"] == RUN_NAME
    assert sorted(k for k in config if config[k] != reference[k]) == ["run_name", "train_input"], "only the training input and the name change"
    assert config["mix"] == [] and config["epochs"] == 1 and config["selection"] == "final" and config["questions_per_request"] == 2
    spec, _ = _steps(JOB)
    train = next(step for step in spec["job"]["steps"] if step["name"] == "train")
    reference_train = next(step for step in load_job(V01_MIX)["job"]["steps"] if step["name"] == "train")
    assert train["wall_set"] == "max_wall_hours" and train["max_minutes"] == reference_train["max_minutes"]
    assert spec["job"]["inputs"][0] == TRAIN_INPUT and spec["budget"]["hourly_usd"] == 3.99
    # 6.5 h (v0.2: 6 h): 5,424 training steps and the T8 readings leave < 30 min of slack under 6 h with every diagnosis
    assert spec["budget"]["max_wall_hours"] == 6.5 and spec["budget"]["phase"] == {"name": "cloud-phase-1", "cap_usd": 150}
    assert [s["name"] for s in spec["job"]["steps"]].count("train") == 1


def test_the_v03_job_diagnoses_its_own_checkpoint_rule_sets_first_with_t8_as_regular_inputs():
    diagnoses = _diagnoses(JOB)
    assert [str(args.input) for _, args in diagnoses] == [SETS[s] for s in ORDER], (
        "T8b (O), G2 (G), T2–T4 (H) first — then T1, T5, T7, T8a, G1, T6 — each once")
    for name, args in diagnoses:
        assert str(args.checkpoint) == f"/r/run/runs/{RUN_NAME}/checkpoint.pt", f"{name}: the v0.3 model's own checkpoint"
        assert not args.preview and args.model == "Qwen/Qwen3.5-2B"
        assert (args.prompt_version, args.fmt, args.orders, args.mode, args.shuffle_seed, args.batch, args.limit) == ("v0c-p3", "chat", "counterbalanced", "direct", 1, 1, None)
        assert str(args.out) == f"/r/run/reports/after-v03-2b-{name.removeprefix('after-')}.json"
    spec, _ = _steps(JOB)
    job = spec["job"]
    assert set(SETS.values()) <= set(job["inputs"]) and not job.get("optional_inputs"), "T8 is final: regular inputs, uploaded always"
    assert {"artifacts/t8/T8a.manifest.json", "artifacts/t8/T8b.manifest.json", "artifacts/simgen/v0b/tests/manifest.json"} <= set(job["inputs"])
    assert all(step.get("optional") for step in job["steps"] if step["name"].startswith("after-"))
    assert not any(step.get("requires") for step in job["steps"])
    assert not any("base" in step["name"] for step in job["steps"]), "no base step: the rule's T8b base is the T8 task's Spark run (noted by the helper)"


def test_the_job_writes_the_prediction_names_the_rule_helper_reads():
    helper = _script("verdict_v03.py")
    diagnosed = {Path(str(args.out)).name.replace(".json", ".predictions.jsonl") for _, args in _diagnoses(JOB)}
    parser_defaults = {"tag": "after-v03-2b"}
    for s in ("T8b", "G2", *helper.HOME):
        assert f"{parser_defaults['tag']}-{s}.predictions.jsonl" in diagnosed
    assert all(str(SETS[s]) == str(helper.SET_PATHS[s].relative_to(REPO)) for s in ("T2", "T3", "T4", "G2", "T8b", "T8a"))


def test_the_v03_job_does_not_train_on_a_sealed_set_and_its_input_was_checked_against_every_sealed_set():
    config = _train(JOB)
    assert config["train_input"] not in SETS.values() and "/tests/" not in config["train_input"] and "/t8/" not in config["train_input"]
    manifest_path = REPO / TRAIN_INPUT.replace(".jsonl", ".manifest.json")
    if not manifest_path.is_file():
        pytest.skip(f"{TRAIN_INPUT} not built yet")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["sealed_check"]["overlap"] == {"origin_groups": 0, "source_records": 0, "base_ids": 0}
    assert not any(manifest["general"]["manifest"]["overlap_after_build"].values())
    overlap = json.loads((REPO / TRAIN_INPUT.replace(".jsonl", ".overlap.json")).read_text(encoding="utf-8"))
    assert overlap["total_hits"] == 0 and overlap["input"]["sha256"] == manifest["out"]["sha256"]
    assert set(overlap["sealed"]) == {"T1", "T2", "T3", "T4", "T5", "T6", "T7", "G1", "G2", "T8a", "T8b"}


def test_the_v03_dry_run_plan_keeps_half_an_hour_of_slack_under_the_cap():
    spec = load_job(JOB)
    job = spec["job"]
    budget = {"hourly_usd": 3.99, "gpus": 1, "max_wall_hours": None, "max_gpu_hours": None, "max_usd": None}
    uploads = [{"path": p, "bytes": 1, "sha256": "0" * 64} for p in job["inputs"]]
    budget["max_wall_hours"] = spec["budget"]["max_wall_hours"]
    plan = plan_job({"run_id": "r", "budget": budget, "provider": spec["provider"], "estimate": spec["estimate"], "job": {**job, "uploads": uploads}})
    assert plan["time"]["total_minutes"] < spec["budget"]["max_wall_hours"] * 60 - 30, "half an hour of slack under the wall cap"
    train = next(step for step in job["steps"] if step["name"] == "train")
    assert 70 <= train["est_minutes"] <= 200 and "TRAIN_ASSUMPTION" not in spec["estimate"]["assumption"]
