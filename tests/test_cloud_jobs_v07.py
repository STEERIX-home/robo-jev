"""v0.7 job definitions (Task v0.7a) — `decisions-v07-s0-h100` and `decisions-v07-s1-h100`.

Each v0.7 job is the v0.6 job of the same seed with the v0.7 training input (the shape transform; no training-config change — the
many-option opt-in rides in the data): the training input, the run name and the names that follow from it differ, and so do the train
step's time (estimate and step cap, from the padded-token simulation of scripts/assemble_v07.py exposure), the wall cap (raised so the
pod order T2, T3, T4, G2, T1, T5, T7, then T8a, G1, T6 keeps ≥ 30 min of slack) and the estimate's text. The phase cap stays $350, a
run's cap stays ≤ $35, nothing scored on Spark is uploaded, and no sealed set is trained on. No model or data is loaded.
"""

from __future__ import annotations

import functools
import importlib.util
import json
import sys
from pathlib import Path

import pytest
import yaml
from helpers import REPO

from robo_jev.decisions.train import resolve_config
from robo_jev.launch.jobs import expand, load_job

JOBS = {0: REPO / "configs" / "cloud" / "decisions-v07-s0-h100.yaml", 1: REPO / "configs" / "cloud" / "decisions-v07-s1-h100.yaml"}
V06_JOBS = {0: REPO / "configs" / "cloud" / "decisions-v06-s0-h100.yaml", 1: REPO / "configs" / "cloud" / "decisions-v06-s1-h100.yaml"}
TRAIN_INPUT = "artifacts/simgen/v07/train-v07.jsonl"
RUN_NAME = {0: "v07-s0-2b", 1: "v07-s1-2b"}
TESTS = {f"T{i}": f"artifacts/simgen/v0b/tests/T{i}.jsonl" for i in range(1, 8)}
SETS = {**TESTS, "G1": "artifacts/simgen/probes/G1.jsonl", "G2": "artifacts/simgen/probes/G2.jsonl", "T8a": "artifacts/t8/T8a.jsonl"}
ORDER = ["T2", "T3", "T4", "G2", "T1", "T5", "T7", "T8a", "G1", "T6"]
SPARK_ONLY = ("artifacts/t9/", "artifacts/t8/T8b", "artifacts/b1/", "scene-holdout", "artifacts/v06/", "artifacts/iface/", "artifacts/v07/")
CONTEXT = {"python": "python", "uv": "uv", "repo": "/r", "run_dir": "/r/run", "runs": "/r/run/runs", "reports": "/r/run/reports", "run_id": "x"}
HOURLY, RUN_CAP = 3.99, 35.0


@functools.lru_cache(maxsize=None)
def _script(name: str):
    spec = importlib.util.spec_from_file_location(f"v07_job_{Path(name).stem}", REPO / "scripts" / name)
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


def _body(path: Path) -> dict:
    return yaml.safe_load("\n".join(line for line in path.read_text(encoding="utf-8").splitlines() if not line.startswith("#")))


@pytest.mark.parametrize("seed", [0, 1])
def test_each_v07_job_is_the_v06_job_of_its_seed_with_the_v07_input_and_more_train_time(seed):
    mine, want = _body(JOBS[seed]), yaml.safe_load(json.dumps(_body(V06_JOBS[seed])).replace("v06", "v07"))
    for block in (mine, want):
        train = next(s for s in block["job"]["steps"] if s["name"] == "train")
        train.pop("est_minutes"), train.pop("max_minutes")
        block["budget"].pop("max_wall_hours")
        block["estimate"].pop("assumption")
    assert mine == want, "only the input, the run name and its names, the train step's time, the wall cap and the estimate text differ"
    config, v06 = _train(JOBS[seed]), _train(V06_JOBS[seed])
    assert config["train_input"] == TRAIN_INPUT and config["run_name"] == RUN_NAME[seed] and config["seed"] == seed
    assert sorted(k for k in config if config[k] != v06[k]) == ["run_name", "train_input"], "the recipe is v0.6's"
    assert config["mix"] == [] and config["epochs"] == 1 and config["selection"] == "final" and config["questions_per_request"] == 2


@pytest.mark.parametrize("seed", [0, 1])
def test_the_train_step_and_wall_cap_fit_the_estimate_and_the_run_cap(seed):
    spec, _ = _steps(JOBS[seed])
    steps = {s["name"]: s for s in spec["job"]["steps"]}
    train = steps["train"]
    assert train["est_minutes"] > 135 and train["max_minutes"] >= 1.25 * train["est_minutes"]
    assert train["wall_set"] == "max_wall_hours"
    work = sum(s["est_minutes"] or 0 for s in spec["job"]["setup"] + spec["job"]["steps"])
    wall = spec["budget"]["max_wall_hours"]
    total = work + spec["estimate"]["boot_minutes"] + spec["estimate"]["fetch_minutes"]
    assert wall * 60 - total >= 30, "≥ 30 min of slack in the plan"
    assert wall * HOURLY <= RUN_CAP, "a run's cap within the brief's $35"
    assert spec["budget"]["hourly_usd"] == HOURLY and spec["budget"]["phase"] == {"name": "cloud-phase-1", "cap_usd": 350}


@pytest.mark.parametrize("seed", [0, 1])
def test_each_v07_job_diagnoses_its_own_checkpoint_with_the_rules_sets_first_and_t6_last(seed):
    diagnoses = _diagnoses(JOBS[seed])
    assert [str(args.input) for _, args in diagnoses] == [SETS[s] for s in ORDER]
    for name, args in diagnoses:
        assert str(args.checkpoint) == f"/r/run/runs/{RUN_NAME[seed]}/checkpoint.pt", f"{name}: the job's own checkpoint"
        assert not args.preview and args.model == "Qwen/Qwen3.5-2B"
        assert (args.prompt_version, args.fmt, args.orders, args.mode, args.shuffle_seed, args.batch, args.limit) == ("v0c-p3", "chat", "counterbalanced", "direct", 1, 1, None)
        assert str(args.out) == f"/r/run/reports/after-{RUN_NAME[seed]}-{name.removeprefix('after-')}.json"


@pytest.mark.parametrize("seed", [0, 1])
def test_nothing_scored_on_spark_goes_to_the_pod_and_no_sealed_set_is_trained(seed):
    spec, steps = _steps(JOBS[seed])
    everything = list(spec["job"]["inputs"]) + [part for _, argv in steps for part in argv]
    assert not [p for p in everything if any(marker in str(p) for marker in SPARK_ONLY)], "A4, T9, T8b, B1, the H-set, iface and the v0.7 gates stay here"
    config = _train(JOBS[seed])
    assert not any(marker in config["train_input"] for marker in ("/tests/", "/t8/", "/t9/", "/b1/", "holdout", "/probes/", "/v06/a", "/iface/", "/gates/"))


def test_the_two_jobs_differ_only_by_the_seed_and_its_names():
    a = json.dumps(_body(JOBS[0]))
    b = json.dumps(_body(JOBS[1])).replace("v07-s1", "v07-s0").replace('"--set", "seed=1", ', "")
    assert json.loads(a) == json.loads(b)
