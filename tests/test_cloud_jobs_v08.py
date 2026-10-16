"""v0.8 job definitions (Task v0.8a) — `decisions-v08f-s{0,1}-h100` (arm F, the FORMAT package) and `decisions-v08p-s{0,1}-h100` (arm FP, FORMAT
plus the H-prior loss weights).

Each v0.8 job is v0.7's job of the same seed with the v0.8 training input (the same file for both arms): the training input, the run name
and the names that follow from it differ, and so do the train step's time (estimate and step cap, from the padded-token simulation of
``scripts/assemble_v08.py exposure`` scaled from v0.7's measured pods), the wall cap, the phase cap, the uplink rate of the estimate (v0.7's
measured upload) and the estimate's text. Arm FP adds exactly ``label_prior_weights: true`` and ``label_prior_cap: 5.0``. The pod order T2,
T3, T4, G2, T1, T5, T7, then T8a, G1, T6 is v0.7's; a run's cap stays ≤ $45; nothing scored on Spark is uploaded; no sealed set is trained on.
No model or data is loaded.
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

ARMS = ("f", "p")
JOBS = {(arm, seed): REPO / "configs" / "cloud" / f"decisions-v08{arm}-s{seed}-h100.yaml" for arm in ARMS for seed in (0, 1)}
V07_JOBS = {0: REPO / "configs" / "cloud" / "decisions-v07-s0-h100.yaml", 1: REPO / "configs" / "cloud" / "decisions-v07-s1-h100.yaml"}
TRAIN_INPUT = "artifacts/simgen/v08/train-v08.jsonl"
TESTS = {f"T{i}": f"artifacts/simgen/v0b/tests/T{i}.jsonl" for i in range(1, 8)}
SETS = {**TESTS, "G1": "artifacts/simgen/probes/G1.jsonl", "G2": "artifacts/simgen/probes/G2.jsonl", "T8a": "artifacts/t8/T8a.jsonl"}
ORDER = ["T2", "T3", "T4", "G2", "T1", "T5", "T7", "T8a", "G1", "T6"]
SPARK_ONLY = ("artifacts/t9/", "artifacts/t8/T8b", "artifacts/b1/", "scene-holdout", "artifacts/v06/", "artifacts/iface/", "artifacts/v07/",
              "artifacts/v08/")
CONTEXT = {"python": "python", "uv": "uv", "repo": "/r", "run_dir": "/r/run", "runs": "/r/run/runs", "reports": "/r/run/reports", "run_id": "x"}
HOURLY, RUN_CAP = 3.99, 45.0
PHASE_CAP = 500


def run_name(arm: str, seed: int) -> str:
    return f"v08{arm}-s{seed}-2b"


@functools.lru_cache(maxsize=None)
def _script(name: str):
    spec = importlib.util.spec_from_file_location(f"v08_job_{Path(name).stem}", REPO / "scripts" / name)
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


def _strip_allowed(block: dict) -> dict:
    train = next(s for s in block["job"]["steps"] if s["name"] == "train")
    train.pop("est_minutes"), train.pop("max_minutes")
    block["budget"].pop("max_wall_hours"), block["budget"].pop("phase")
    block["estimate"].pop("assumption"), block["estimate"].pop("uplink_mbps")
    return block


@pytest.mark.parametrize("arm, seed", sorted(JOBS))
def test_each_v08_job_is_the_v07_job_of_its_seed_with_the_v08_input(arm, seed):
    mine = _strip_allowed(_body(JOBS[(arm, seed)]))
    text = json.dumps(_body(V07_JOBS[seed])).replace("train-v07", "train-v08").replace("simgen/v07/", "simgen/v08/").replace("v07-", f"v08{arm}-")
    want = _strip_allowed(json.loads(text))
    if arm == "p":
        train = next(s for s in want["job"]["steps"] if s["name"] == "train")
        cut = train["argv"].index("--gpu-memory-fraction")
        train["argv"][cut:cut] = ["--set", "label_prior_weights=true", "--set", "label_prior_cap=5.0"]
    assert mine == want, "only the input, the run name and its names, the train step's time, the wall and phase caps, the uplink rate and the estimate text differ"
    config, v07 = _train(JOBS[(arm, seed)]), _train(V07_JOBS[seed])
    assert config["train_input"] == TRAIN_INPUT and config["run_name"] == run_name(arm, seed) and config["seed"] == seed
    changed = sorted(k for k in config if config[k] != v07[k])
    assert changed == (["label_prior_weights", "run_name", "train_input"] if arm == "p" else ["run_name", "train_input"]), "the recipe is v0.7's"
    assert config["label_prior_weights"] is (arm == "p") and config["label_prior_cap"] == 5.0
    assert config["mix"] == [] and config["epochs"] == 1 and config["selection"] == "final" and config["questions_per_request"] == 2


@pytest.mark.parametrize("seed", [0, 1])
def test_the_two_arms_train_the_same_input_and_differ_only_by_the_weights(seed):
    f = json.dumps(_body(JOBS[("f", seed)]))
    p = json.dumps(_body(JOBS[("p", seed)])).replace("v08p-", "v08f-").replace('"--set", "label_prior_weights=true", "--set", "label_prior_cap=5.0", ', "")
    assert json.loads(f) == json.loads(p)
    assert _train(JOBS[("f", seed)])["train_input"] == _train(JOBS[("p", seed)])["train_input"] == TRAIN_INPUT


@pytest.mark.parametrize("arm, seed", sorted(JOBS))
def test_the_train_step_and_wall_cap_fit_the_estimate_and_the_run_cap(arm, seed):
    spec, _ = _steps(JOBS[(arm, seed)])
    steps = {s["name"]: s for s in spec["job"]["steps"]}
    train = steps["train"]
    assert train["est_minutes"] > 200 and train["max_minutes"] >= 1.5 * train["est_minutes"]
    assert train["wall_set"] == "max_wall_hours"
    work = sum(s["est_minutes"] or 0 for s in spec["job"]["setup"] + spec["job"]["steps"])
    wall = spec["budget"]["max_wall_hours"]
    upload = 30.0  # ≈ 1.02 GB at v0.7's measured ≈ 5 Mbit/s (the dry-run computes it from the files)
    total = work + upload + spec["estimate"]["boot_minutes"] + spec["estimate"]["fetch_minutes"]
    assert wall * 60 - total >= 60, "≥ 60 min of slack in the plan"
    assert wall * HOURLY <= RUN_CAP, "a run's cap within $45 (the brief)"
    assert spec["budget"]["hourly_usd"] == HOURLY and spec["budget"]["phase"] == {"name": "cloud-phase-1", "cap_usd": PHASE_CAP}
    assert spec["estimate"]["uplink_mbps"] == 5, "v0.7's measured upload: 950 MB in ≈ 26 min ≈ 5 Mbit/s"


@pytest.mark.parametrize("arm, seed", sorted(JOBS))
def test_each_v08_job_diagnoses_its_own_checkpoint_with_the_rules_sets_first_and_t6_last(arm, seed):
    diagnoses = _diagnoses(JOBS[(arm, seed)])
    assert [str(args.input) for _, args in diagnoses] == [SETS[s] for s in ORDER]
    for name, args in diagnoses:
        assert str(args.checkpoint) == f"/r/run/runs/{run_name(arm, seed)}/checkpoint.pt", f"{name}: the job's own checkpoint"
        assert not args.preview and args.model == "Qwen/Qwen3.5-2B"
        assert (args.prompt_version, args.fmt, args.orders, args.mode, args.shuffle_seed, args.batch, args.limit) == ("v0c-p3", "chat", "counterbalanced", "direct", 1, 1, None)
        assert str(args.out) == f"/r/run/reports/after-{run_name(arm, seed)}-{name.removeprefix('after-')}.json"


@pytest.mark.parametrize("arm, seed", sorted(JOBS))
def test_nothing_scored_on_spark_goes_to_the_pod_and_no_sealed_set_is_trained(arm, seed):
    spec, steps = _steps(JOBS[(arm, seed)])
    everything = list(spec["job"]["inputs"]) + [part for _, argv in steps for part in argv]
    assert not [p for p in everything if any(marker in str(p) for marker in SPARK_ONLY)], "A4, T9, T8b, B1, the H-set, iface and the v0.7 / v0.8 gates stay here"
    config = _train(JOBS[(arm, seed)])
    assert not any(marker in config["train_input"] for marker in ("/tests/", "/t8/", "/t9/", "/b1/", "holdout", "/probes/", "/v06/a", "/iface/", "/gates/"))


@pytest.mark.parametrize("arm", ARMS)
def test_the_two_seeds_differ_only_by_the_seed_and_its_names(arm):
    a = json.dumps(_body(JOBS[(arm, 0)]))
    b = json.dumps(_body(JOBS[(arm, 1)])).replace(f"v08{arm}-s1", f"v08{arm}-s0").replace('"--set", "seed=1", ', "")
    assert json.loads(a) == json.loads(b)
