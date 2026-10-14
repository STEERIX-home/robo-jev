"""v0.6 job definitions (Task v0.6b §1-2) — `decisions-v06-s0-h100` and `decisions-v06-s1-h100`.

Each v0.6 job is the v0.5 job of the same seed with the v0.6 training input: the training input, the run name and the names that follow from
it (job, checkpoint, reports) differ; besides, the pod order puts T6 last (T8a, G1, T6 — lesson 33: v0.5-s1's slower pod lost T6, G1 and
T8a to the 6.5 h cap) and the phase block carries the raised $350 cap. The recipe, the seed, the diagnosis conditions, the budget and the
estimate are v0.5's (= v0.4's). The jobs never upload the sets the rule scores on Spark (A4, T9, T8b, B1-infamily, the H-set), write the
prediction names the v0.6 verdict helper reads, and do not train on a sealed set. No model or data is loaded.
"""

from __future__ import annotations

import functools
import importlib.util
import json
import re
import sys
from pathlib import Path

import pytest
import yaml
from helpers import REPO

from robo_jev.decisions.train import resolve_config
from robo_jev.launch.jobs import expand, load_job, plan_job

JOBS = {0: REPO / "configs" / "cloud" / "decisions-v06-s0-h100.yaml", 1: REPO / "configs" / "cloud" / "decisions-v06-s1-h100.yaml"}
V05_JOBS = {0: REPO / "configs" / "cloud" / "decisions-v05-s0-h100.yaml", 1: REPO / "configs" / "cloud" / "decisions-v05-s1-h100.yaml"}
V04_JOBS = {0: REPO / "configs" / "cloud" / "decisions-v04-s0-h100.yaml", 1: REPO / "configs" / "cloud" / "decisions-v04-s1-h100.yaml"}
TRAIN_INPUT = "artifacts/simgen/v06/train-v06.jsonl"
RUN_NAME = {0: "v06-s0-2b", 1: "v06-s1-2b"}
TESTS = {f"T{i}": f"artifacts/simgen/v0b/tests/T{i}.jsonl" for i in range(1, 8)}
SETS = {**TESTS, "G1": "artifacts/simgen/probes/G1.jsonl", "G2": "artifacts/simgen/probes/G2.jsonl", "T8a": "artifacts/t8/T8a.jsonl"}
ORDER = ["T2", "T3", "T4", "G2", "T1", "T5", "T7", "T8a", "G1", "T6"]
SPARK_ONLY = ("artifacts/t9/", "artifacts/t8/T8b", "artifacts/b1/", "scene-holdout", "artifacts/v06/")
CONTEXT = {"python": "python", "uv": "uv", "repo": "/r", "run_dir": "/r/run", "runs": "/r/run/runs", "reports": "/r/run/reports", "run_id": "x"}


@functools.lru_cache(maxsize=None)
def _script(name: str):
    spec = importlib.util.spec_from_file_location(f"v06_job_{Path(name).stem}", REPO / "scripts" / name)
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
def test_each_v06_job_is_the_v05_job_of_its_seed_with_the_v06_input_t6_last_and_the_350_dollar_phase(seed):
    want = yaml.safe_load(json.dumps(_body(V05_JOBS[seed])).replace("v05", "v06"))
    steps = {s["name"]: s for s in want["job"]["steps"]}
    want["job"]["steps"] = [s for s in want["job"]["steps"] if not s["name"].startswith("after-")] + [steps[f"after-{name}"] for name in ORDER]
    want["budget"]["phase"]["cap_usd"] = 350
    assert _body(JOBS[seed]) == want, "only the input, the run name and its names, the pod order (T6 last) and the phase cap differ from v0.5's job"
    config, v05, v04 = _train(JOBS[seed]), _train(V05_JOBS[seed]), _train(V04_JOBS[seed])
    assert config["train_input"] == TRAIN_INPUT and config["run_name"] == RUN_NAME[seed] and config["seed"] == seed
    assert sorted(k for k in config if config[k] != v05[k]) == sorted(k for k in config if config[k] != v04[k]) == ["run_name", "train_input"]
    assert config["mix"] == [] and config["epochs"] == 1 and config["selection"] == "final" and config["questions_per_request"] == 2
    spec, _ = _steps(JOBS[seed])
    assert spec["job"]["inputs"][0] == TRAIN_INPUT and spec["budget"]["hourly_usd"] == 3.99 and spec["budget"]["max_wall_hours"] == 6.5
    assert spec["budget"]["phase"] == {"name": "cloud-phase-1", "cap_usd": 350}
    assert [s["name"] for s in spec["job"]["steps"]].count("train") == 1


@pytest.mark.parametrize("seed", [0, 1])
def test_each_v06_job_diagnoses_its_own_checkpoint_with_the_rules_sets_first_and_t6_last(seed):
    diagnoses = _diagnoses(JOBS[seed])
    assert [str(args.input) for _, args in diagnoses] == [SETS[s] for s in ORDER]
    for name, args in diagnoses:
        assert str(args.checkpoint) == f"/r/run/runs/{RUN_NAME[seed]}/checkpoint.pt", f"{name}: the job's own checkpoint"
        assert not args.preview and args.model == "Qwen/Qwen3.5-2B"
        assert (args.prompt_version, args.fmt, args.orders, args.mode, args.shuffle_seed, args.batch, args.limit) == ("v0c-p3", "chat", "counterbalanced", "direct", 1, 1, None)
        assert str(args.out) == f"/r/run/reports/after-{RUN_NAME[seed]}-{name.removeprefix('after-')}.json"
    rule_sets = {"T1", "T2", "T3", "T4", "T5", "T7", "G2"}
    names = [Path(str(args.input)).stem for _, args in diagnoses]
    assert set(names[: len(rule_sets)]) == rule_sets and names[-1] == "T6", "every set the rule reads on the pod before any secondary one"


@pytest.mark.parametrize("seed", [0, 1])
def test_a4_t9_t8b_b1_and_the_h_set_never_go_to_the_pod(seed):
    spec, steps = _steps(JOBS[seed])
    everything = list(spec["job"]["inputs"]) + [part for _, argv in steps for part in argv]
    assert not [p for p in everything if any(marker in str(p) for marker in SPARK_ONLY)], "scored on Spark for every model, never uploaded"


@pytest.mark.parametrize("seed", [0, 1])
def test_each_job_writes_the_prediction_names_the_verdict_helper_reads(seed):
    helper = _script("verdict_v06.py")
    diagnosed = {Path(str(args.out)).name.replace(".json", ".predictions.jsonl") for _, args in _diagnoses(JOBS[seed])}
    for s in helper.POD_SETS:
        assert helper.pod_prediction_name("v06", seed, s) in diagnosed
    assert all(str(SETS[s]) == str(helper.SET_PATHS[s].relative_to(REPO)) for s in helper.POD_SETS)


@pytest.mark.parametrize("seed", [0, 1])
def test_each_v06_job_does_not_train_on_a_sealed_set(seed):
    config = _train(JOBS[seed])
    assert not any(marker in config["train_input"] for marker in ("/tests/", "/t8/", "/t9/", "/b1/", "holdout", "/probes/", "/v06/a"))
    summary = REPO / "artifacts" / "simgen" / "v06" / "train-v06.assembly.json"
    if not summary.is_file():
        pytest.skip(f"{TRAIN_INPUT} not assembled yet (scripts/assemble_v06.py all, after the three authors land)")
    built = json.loads(summary.read_text(encoding="utf-8"))
    overlap = json.loads((REPO / "artifacts" / "simgen" / "v06" / "train-v06.overlap.json").read_text(encoding="utf-8"))
    assert built["pass"] is True and overlap["total_hits"] == 0
    assert set(overlap["sealed"]) >= {"T1", "T2", "T3", "T4", "T5", "T6", "T7", "G1", "G2", "T8a", "T8b", "T9", "B1", "H-set", "A4"}


@pytest.mark.parametrize("seed", [0, 1])
def test_each_v06_dry_run_plan_keeps_half_an_hour_of_slack_under_the_cap(seed):
    spec = load_job(JOBS[seed])
    job = spec["job"]
    budget = {"hourly_usd": 3.99, "gpus": 1, "max_wall_hours": spec["budget"]["max_wall_hours"], "max_gpu_hours": None, "max_usd": None}
    uploads = [{"path": p, "bytes": 1, "sha256": "0" * 64} for p in job["inputs"]]
    plan = plan_job({"run_id": "r", "budget": budget, "provider": spec["provider"], "estimate": spec["estimate"], "job": {**job, "uploads": uploads}})
    assert plan["time"]["total_minutes"] < spec["budget"]["max_wall_hours"] * 60 - 30, "half an hour of slack under the wall cap"


def test_the_header_names_the_one_command_and_the_spark_sets():
    for seed, path in JOBS.items():
        text = path.read_text(encoding="utf-8")
        assert "python scripts/assemble_v06.py all --out-dir artifacts/simgen/v06" in text and "scripts/verdict_v06.py" in text
        assert re.search(r"A4, T9, T8b, B1-infamily and the H-set are scored on Spark", text), seed
