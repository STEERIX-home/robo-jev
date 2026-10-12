"""v0.2 job definitions (Task v0.2 §1-4) — `decisions-v02-a-h100` (general ≈ 10 %) and `decisions-v02-b-h100` (≈ 25 %).

Same pattern as the v0.1 job tests: each job trains once with the v0.1-mix recipe (only the training input and the run name differ),
diagnoses its own checkpoint on G1 → G2 → T1–T7 under the Spark/v0.1 scoring conditions, then the outside test T8 **only if its files
exist when the job is prepared** — `optional_inputs` + steps that `require` them, skipped cleanly by the runner and the dry-run when
absent. The mechanism itself (optional inputs, `requires`, skipped steps) is tested on a small local bundle. No model or data is loaded.
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
from robo_jev.launch.jobs import JobError, all_inputs, expand, load_job, missing_requirements, optional_inputs_status, plan_job, resolve_job, run_job

JOBS = {"a": REPO / "configs" / "cloud" / "decisions-v02-a-h100.yaml", "b": REPO / "configs" / "cloud" / "decisions-v02-b-h100.yaml"}
V01_MIX = REPO / "configs" / "cloud" / "decisions-v01-mix-h100.yaml"
TRAIN_INPUT = {d: f"artifacts/simgen/v02/train-v02-{d}.jsonl" for d in JOBS}
RUN_NAME = {d: f"v02-{d}-2b" for d in JOBS}
TESTS = [f"artifacts/simgen/v0b/tests/T{i}.jsonl" for i in range(1, 8)]
G1, G2 = "artifacts/simgen/probes/G1.jsonl", "artifacts/simgen/probes/G2.jsonl"
#: where the T8 task writes its sets (the brief said artifacts/simgen/t8/; the T8 task's scripts/t8.py writes artifacts/t8/)
T8 = {"after-T8a": "artifacts/t8/T8a.jsonl", "after-T8b": "artifacts/t8/T8b.jsonl"}
SEALED = (*TESTS, G1, G2, *T8.values())
CONTEXT = {"python": "python", "uv": "uv", "repo": "/r", "run_dir": "/r/run", "runs": "/r/run/runs", "reports": "/r/run/reports", "run_id": "x"}


@functools.lru_cache(maxsize=None)
def _script(name: str):
    spec = importlib.util.spec_from_file_location(f"v02_{Path(name).stem}", REPO / "scripts" / name)
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


@pytest.mark.parametrize("dose", sorted(JOBS))
def test_each_v02_job_trains_once_with_the_v01_mix_recipe(dose):
    config = _train(JOBS[dose])
    reference = _train(V01_MIX)
    assert config["train_input"] == TRAIN_INPUT[dose] and config["run_name"] == RUN_NAME[dose]
    assert sorted(k for k in config if config[k] != reference[k]) == ["run_name", "train_input"], "only the training input and the name change"
    assert config["mix"] == [] and config["epochs"] == 1 and config["selection"] == "final"
    spec, _ = _steps(JOBS[dose])
    train = next(step for step in spec["job"]["steps"] if step["name"] == "train")
    reference_train = next(step for step in load_job(V01_MIX)["job"]["steps"] if step["name"] == "train")
    assert train["wall_set"] == "max_wall_hours" and train["max_minutes"] == reference_train["max_minutes"]
    assert spec["job"]["inputs"][0] == TRAIN_INPUT[dose] and spec["budget"]["max_wall_hours"] == 6 and spec["budget"]["hourly_usd"] == 3.99


@pytest.mark.parametrize("dose", sorted(JOBS))
def test_each_v02_job_diagnoses_its_own_checkpoint_gating_sets_first_then_g1_t6_then_t8_only_if_present(dose):
    diagnoses = _diagnoses(JOBS[dose])
    gating = [G2, *TESTS[:5], TESTS[6]]  # the pre-declared v0.2 rule reads G2 (G), T2+T3+T4 (P) and T1, T3, T4, T5, T7 (C)
    assert [str(args.input) for _, args in diagnoses] == [*gating, G1, TESTS[5], *T8.values()], (
        "the gating sets first, then the secondary G1 and T6, then T8a, T8b — each once, so a slow pod only cuts secondary readings")
    checkpoint = f"/r/run/runs/{RUN_NAME[dose]}/checkpoint.pt"
    for name, args in diagnoses:
        assert str(args.checkpoint) == checkpoint, f"{name}: the v0.2 model's own checkpoint"
        assert not args.preview and args.model == "Qwen/Qwen3.5-2B"
        # the scoring conditions of the Spark baselines and of the v0.1 runs (the diagnosis tool's defaults)
        assert (args.prompt_version, args.fmt, args.orders, args.mode, args.shuffle_seed, args.batch, args.limit) == ("v0c-p3", "chat", "counterbalanced", "direct", 1, 1, None)
        assert str(args.out) == f"/r/run/reports/after-v02{dose}-2b-{name.removeprefix('after-')}.json"
    spec, _ = _steps(JOBS[dose])
    job = spec["job"]
    by_name = {step["name"]: step for step in job["steps"]}
    assert all(by_name[f"after-{s}"]["optional"] for s in ("G1", "G2", *(f"T{i}" for i in range(1, 8))))
    for name, path in T8.items():
        assert by_name[name]["optional"] is True and by_name[name]["requires"] == [path]
        assert path in job["optional_inputs"] and path not in job["inputs"], "T8 is uploaded only when it exists"
    assert [s["name"] for s in job["steps"]][-2:] == list(T8), "T8 runs last — it never delays the gating sets"
    assert set(SEALED) - set(T8.values()) <= set(job["inputs"]) and by_name["after-T6"]["max_minutes"] >= 80
    assert not any(step["requires"] for step in job["steps"] if step["name"] not in T8)


def test_the_two_v02_jobs_differ_only_in_the_training_input_the_names_and_the_training_estimate():
    def normalised(dose: str) -> str:
        spec = load_job(JOBS[dose])
        spec["estimate"] = {k: v for k, v in spec["estimate"].items() if k != "assumption"}
        for step in spec["job"]["steps"]:
            if step["name"] == "train":
                step["est_minutes"] = None
        text = json.dumps({k: v for k, v in spec.items() if k != "name"}, sort_keys=True, ensure_ascii=False)
        return text.replace(TRAIN_INPUT[dose], "<train>").replace(RUN_NAME[dose], "<run>").replace(f"after-v02{dose}-", "after-<tag>-")

    assert normalised("a") == normalised("b")
    estimates = {d: next(s["est_minutes"] for s in load_job(JOBS[d])["job"]["steps"] if s["name"] == "train") for d in JOBS}
    assert 66 <= estimates["a"] < estimates["b"] <= 120, estimates


@pytest.mark.parametrize("dose", sorted(JOBS))
def test_no_v02_job_trains_on_a_sealed_set_and_the_training_input_was_built_from_unsealed_material(dose):
    config = _train(JOBS[dose])
    assert config["train_input"] not in SEALED and "/tests/" not in config["train_input"] and "/t8/" not in config["train_input"]
    manifest_path = REPO / TRAIN_INPUT[dose].replace(".jsonl", ".manifest.json")
    if not manifest_path.is_file():
        pytest.skip(f"{TRAIN_INPUT[dose]} not built yet")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["sealed_check"]["overlap"] == {"origin_groups": 0, "source_records": 0, "base_ids": 0}
    assert not any(manifest["general"]["manifest"]["overlap_after_build"].values())
    assert manifest["probe"]["path"].endswith("probes/G1-train-mix.jsonl") and manifest["base"]["path"].endswith("v02/train-50k.jsonl")


# --------------------------------------------------------------------------
# the mechanism: optional inputs and steps that require them
# --------------------------------------------------------------------------


def _job(**job) -> dict:
    data = {"kind": "robojev-cloud-job-v0", "name": "t", "job": {"kind": "decisions", "model": "tiny", "inputs": ["in.txt"],
            "steps": [{"name": "s", "argv": ["true"]}], **job}, "budget": {"hourly_usd": 0.0, "max_wall_hours": 0.5}}  # fmt: skip
    return data


@pytest.mark.parametrize(
    "job, needle",
    [
        ({"optional_inputs": ["/abs/x"]}, "상대 경로"),
        ({"optional_inputs": ["../x"]}, "상대 경로"),
        ({"optional_inputs": ["in.txt"]}, "이미 inputs"),
        ({"optional_inputs": ["t8.jsonl"], "steps": [{"name": "s", "argv": ["true"], "requires": ["t8.jsonl"]}]}, "optional 단계에만"),
        ({"steps": [{"name": "s", "argv": ["true"], "optional": True, "requires": ["nowhere.jsonl"]}]}, "inputs나 optional_inputs"),
    ],
)
def test_optional_inputs_and_requires_are_checked_before_anything_is_sent(job, needle):
    with pytest.raises(JobError, match=needle):
        resolve_job(_job(**job))


def test_optional_inputs_are_uploaded_only_when_present_and_the_plan_skips_steps_that_need_absent_ones(tmp_path):
    repo = tmp_path / "repo"
    (repo / "artifacts" / "tokenizers" / "tok").mkdir(parents=True)
    (repo / "artifacts" / "tokenizers" / "manifest.json").write_text(json.dumps({"id": "tok"}), encoding="utf-8")
    (repo / "artifacts" / "tokenizers" / "tok" / "tokenizer.json").write_text("{}", encoding="utf-8")
    (repo / "in.txt").write_text("x", encoding="utf-8")
    (repo / "here.jsonl").write_text("{}\n", encoding="utf-8")
    steps = [{"name": "main", "argv": ["true"], "est_minutes": 5},
             {"name": "t8-here", "argv": ["true"], "optional": True, "requires": ["here.jsonl"], "est_minutes": 7},
             {"name": "t8-absent", "argv": ["true"], "optional": True, "requires": ["absent.jsonl"], "est_minutes": 11}]  # fmt: skip
    job = resolve_job(_job(optional_inputs=["here.jsonl", "absent.jsonl"], steps=steps))["job"]
    assert optional_inputs_status(job, repo) == {"here.jsonl": True, "absent.jsonl": False}
    uploads = all_inputs(job, repo)
    assert "here.jsonl" in uploads and "absent.jsonl" not in uploads and "in.txt" in uploads
    job["uploads"] = [{"path": p, "bytes": 1, "sha256": "0" * 64} for p in uploads]
    plan = plan_job({"run_id": "r", "job": job, "budget": {"hourly_usd": 1.0, "gpus": 0, "max_wall_hours": 1.0, "max_gpu_hours": None, "max_usd": None}})
    rows = {row["name"]: row for row in plan["steps"]}
    assert rows["t8-here"]["est_minutes"] == 7 and rows["t8-here"]["skipped_missing_inputs"] == []
    assert rows["t8-absent"]["est_minutes"] == 0.0 and rows["t8-absent"]["skipped_missing_inputs"] == ["absent.jsonl"]
    assert plan["time"]["work_minutes"] == 12
    assert missing_requirements(job["steps"][2], repo=repo) == ["absent.jsonl"] and missing_requirements(job["steps"][1], repo=repo) == []


def test_the_runner_skips_a_step_whose_required_input_is_absent_and_the_job_goes_on(tmp_path):
    run_dir = tmp_path / "run"
    (run_dir / "bundle").mkdir(parents=True)
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "here.jsonl").write_text("{}\n", encoding="utf-8")
    steps = [
        {"name": "t8-absent", "argv": ["{python}", "-c", "open('absent-ran','w')"], "optional": True, "requires": ["absent.jsonl"]},
        {"name": "t8-here", "argv": ["{python}", "-c", "open('here-ran','w')"], "optional": True, "requires": ["here.jsonl"]},
        {"name": "last", "argv": ["{python}", "-c", "open('last-ran','w')"]},
    ]
    data = _job(optional_inputs=["here.jsonl", "absent.jsonl"], steps=steps)
    data["job"]["inputs"] = []
    job = resolve_job(data)["job"]
    job["runtime"]["python"] = sys.executable
    job["uploads"] = []
    manifest = {"run_id": "t", "job": job, "budget": {"hourly_usd": 1.0, "gpus": 0, "max_wall_hours": 0.5, "max_gpu_hours": None, "max_usd": None}}
    (run_dir / "bundle" / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    assert run_job(run_dir / "bundle", repo=repo) == 0
    report = json.loads((run_dir / "job-report.json").read_text(encoding="utf-8"))
    rows = {r["name"]: r for r in report["steps"]}
    assert rows["t8-absent"]["status"] == "skipped" and rows["t8-absent"]["reason"] == "missing-inputs" and rows["t8-absent"]["missing"] == ["absent.jsonl"]
    assert rows["t8-here"]["status"] == "ok" and rows["last"]["status"] == "ok" and report["state"] == "completed"
    assert not (repo / "absent-ran").exists() and (repo / "here-ran").exists() and (repo / "last-ran").exists()


@pytest.mark.parametrize("dose", sorted(JOBS))
def test_the_v02_dry_run_plan_counts_t8_only_when_its_files_were_uploaded(dose):
    spec = load_job(JOBS[dose])
    job = spec["job"]
    budget = {"hourly_usd": 3.99, "gpus": 1, "max_wall_hours": 6, "max_gpu_hours": None, "max_usd": None}
    common = {"run_id": "r", "budget": budget, "provider": spec["provider"], "estimate": spec["estimate"]}
    base_uploads = [{"path": p, "bytes": 1, "sha256": "0" * 64} for p in job["inputs"]]
    without = plan_job({**common, "job": {**job, "uploads": base_uploads}})
    with_t8 = plan_job({**common, "job": {**job, "uploads": base_uploads + [{"path": p, "bytes": 1, "sha256": "0" * 64} for p in T8.values()]}})
    rows = {row["name"]: row for row in without["steps"]}
    assert all(rows[name]["skipped_missing_inputs"] == [path] and rows[name]["est_minutes"] == 0.0 for name, path in T8.items())
    t8_minutes = sum(step["est_minutes"] for step in job["steps"] if step["name"] in T8)
    assert with_t8["time"]["work_minutes"] == pytest.approx(without["time"]["work_minutes"] + t8_minutes)
    assert with_t8["time"]["total_minutes"] < 6 * 60 - 30, "even with T8 the plan keeps half an hour of slack under the 6 h cap"
    assert yaml.safe_load(JOBS[dose].read_text(encoding="utf-8"))["job"]["optional_inputs"][0] == T8["after-T8a"]
