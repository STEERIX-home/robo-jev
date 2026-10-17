"""v0.9 job definitions (Task v0.9a) — `decisions-v09c-s{0,1}-h100` (arm C: v0.8's input + the order-consistency loss) and
`decisions-v09cn-s{0,1}-h100` (arm CN: the measurement-literacy input + the same loss).

Each v0.9 job is v0.8 F's job of the same seed (`decisions-v08f-s{seed}-h100`) with: the run name and the names that follow from it, the
consistency option on the train step (``--set consistency={share: 0.3, lambda: 0.5, copies: [shift, reverse], divergence: js}``), the train
step's time (estimate and step cap, from ``scripts/assemble_v09.py exposure`` scaled from v0.8 F's measured pods), the wall cap, the phase cap
($650) and the estimate's text; arm CN also trains ``artifacts/simgen/v09/train-v09n.jsonl``. The pod order T2, T3, T4, G2, T1, T5, T7, then
T8a, G1, T6 is v0.8's; a run's cap stays ≤ $48; nothing scored on Spark is uploaded; no sealed set is trained on. No model or data is loaded.
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

ARMS = ("c", "cn")
JOBS = {(arm, seed): REPO / "configs" / "cloud" / f"decisions-v09{arm}-s{seed}-h100.yaml" for arm in ARMS for seed in (0, 1)}
V08F_JOBS = {seed: REPO / "configs" / "cloud" / f"decisions-v08f-s{seed}-h100.yaml" for seed in (0, 1)}
INPUTS = {"c": "artifacts/simgen/v08/train-v08.jsonl", "cn": "artifacts/simgen/v09/train-v09n.jsonl"}
CONSISTENCY = {"share": 0.3, "lambda": 0.5, "copies": ["shift", "reverse"], "divergence": "js", "salt": "v09-consistency-1"}
SET_ARG = "consistency={share: 0.3, lambda: 0.5, copies: [shift, reverse], divergence: js}"
TESTS = {f"T{i}": f"artifacts/simgen/v0b/tests/T{i}.jsonl" for i in range(1, 8)}
SETS = {**TESTS, "G1": "artifacts/simgen/probes/G1.jsonl", "G2": "artifacts/simgen/probes/G2.jsonl", "T8a": "artifacts/t8/T8a.jsonl"}
ORDER = ["T2", "T3", "T4", "G2", "T1", "T5", "T7", "T8a", "G1", "T6"]
SPARK_ONLY = ("artifacts/t9/", "artifacts/t8/T8b", "artifacts/b1/", "scene-holdout", "artifacts/v06/", "artifacts/iface/", "artifacts/v07/",
              "artifacts/v08/", "artifacts/v09/")
CONTEXT = {"python": "python", "uv": "uv", "repo": "/r", "run_dir": "/r/run", "runs": "/r/run/runs", "reports": "/r/run/reports", "run_id": "x"}
HOURLY, RUN_CAP = 3.99, 48.0
PHASE_CAP = 650


def run_name(arm: str, seed: int) -> str:
    return f"v09{arm}-s{seed}-2b"


@functools.lru_cache(maxsize=None)
def _script(name: str):
    spec = importlib.util.spec_from_file_location(f"v09_job_{Path(name).stem}", REPO / "scripts" / name)
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
    block["estimate"].pop("assumption")
    return block


@pytest.mark.parametrize("arm, seed", sorted(JOBS))
def test_each_v09_job_is_v08fs_job_of_its_seed_with_the_consistency_option(arm, seed):
    mine = _strip_allowed(_body(JOBS[(arm, seed)]))
    text = json.dumps(_body(V08F_JOBS[seed])).replace("v08f-", f"v09{arm}-")
    if arm == "cn":
        text = text.replace("artifacts/simgen/v08/train-v08.jsonl", INPUTS["cn"])
    want = _strip_allowed(json.loads(text))
    train = next(s for s in want["job"]["steps"] if s["name"] == "train")
    cut = train["argv"].index("--gpu-memory-fraction")
    train["argv"][cut:cut] = ["--set", SET_ARG]
    assert mine == want, "only the run names, the input (CN), the consistency option, the train step's time, the wall and phase caps and the estimate text differ"
    config, v08 = _train(JOBS[(arm, seed)]), _train(V08F_JOBS[seed])
    assert config["train_input"] == INPUTS[arm] and config["run_name"] == run_name(arm, seed) and config["seed"] == seed
    changed = sorted(k for k in config if config[k] != v08[k])
    assert changed == (["consistency", "run_name"] if arm == "c" else ["consistency", "run_name", "train_input"]), "the recipe is v0.8 F's"
    assert config["consistency"] == CONSISTENCY and config["label_prior_weights"] is False
    assert config["mix"] == [] and config["epochs"] == 1 and config["selection"] == "final" and config["questions_per_request"] == 2


@pytest.mark.parametrize("seed", [0, 1])
def test_the_two_arms_differ_only_by_the_input_and_the_names(seed):
    c = json.dumps(_body(JOBS[("c", seed)]))
    cn = json.dumps(_body(JOBS[("cn", seed)])).replace("v09cn-", "v09c-").replace(INPUTS["cn"], INPUTS["c"])
    a, b = json.loads(c), json.loads(cn)
    for block in (a, b):
        _strip_allowed(block)
    assert a == b
    assert _train(JOBS[("c", seed)])["consistency"] == _train(JOBS[("cn", seed)])["consistency"] == CONSISTENCY


@pytest.mark.parametrize("arm, seed", sorted(JOBS))
def test_the_train_step_and_wall_cap_fit_the_estimate_and_the_run_cap(arm, seed):
    spec, _ = _steps(JOBS[(arm, seed)])
    steps = {s["name"]: s for s in spec["job"]["steps"]}
    train = steps["train"]
    assert train["est_minutes"] > 225 and train["max_minutes"] >= 1.4 * train["est_minutes"]
    assert train["wall_set"] == "max_wall_hours"
    work = sum(s["est_minutes"] or 0 for s in spec["job"]["setup"] + spec["job"]["steps"])
    wall = spec["budget"]["max_wall_hours"]
    upload = 32.0  # ≈ 1.03 GB at v0.8's measured ≈ 5 Mbit/s (the dry-run computes it from the files)
    total = work + upload + spec["estimate"]["boot_minutes"] + spec["estimate"]["fetch_minutes"]
    assert wall * 60 - total >= 60, "≥ 60 min of slack in the plan"
    assert wall * HOURLY <= RUN_CAP, "a run's cap within $48 (the brief)"
    assert spec["budget"]["hourly_usd"] == HOURLY and spec["budget"]["phase"] == {"name": "cloud-phase-1", "cap_usd": PHASE_CAP}
    assert spec["estimate"]["uplink_mbps"] == 5


@pytest.mark.parametrize("arm, seed", sorted(JOBS))
def test_each_v09_job_diagnoses_its_own_checkpoint_in_v08s_order(arm, seed):
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
    assert not [p for p in everything if any(marker in str(p) for marker in SPARK_ONLY)], "A4, T9, T8b, B1, the H-set, iface and the gates stay here"
    assert INPUTS[arm] in spec["job"]["inputs"] and sum(1 for p in spec["job"]["inputs"] if "simgen/v0" in p and "train-" in p) == 1
    config = _train(JOBS[(arm, seed)])
    assert not any(marker in config["train_input"] for marker in ("/tests/", "/t8/", "/t9/", "/b1/", "holdout", "/probes/", "/v06/a", "/iface/", "/gates/"))


@pytest.mark.parametrize("arm", ARMS)
def test_the_two_seeds_differ_only_by_the_seed_and_its_names(arm):
    a = _body(JOBS[(arm, 0)])
    b = json.loads(json.dumps(_body(JOBS[(arm, 1)])).replace(f"v09{arm}-s1", f"v09{arm}-s0").replace('"--set", "seed=1", ', ""))
    for block in (a, b):
        _strip_allowed(block)
    assert a == b
