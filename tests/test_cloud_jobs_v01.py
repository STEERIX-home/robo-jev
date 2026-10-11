"""v0.1 본 run의 job 정의 둘 (Task v0.1b) — `decisions-v01-main-h100`(v0.1a의 train-50k)·`decisions-v01-mix-h100`(그 뒤에 G1-train-mix ×1).

v0 job 정의의 시험(tests/test_cloud_jobs.py의 실어 보내는 job 검사 — 그것이 이 둘도 자기 argparse·resolve_config로 이미 읽는다)과 같은 방식으로
더 본다: 두 job이 v0의 학습 조리법을 그대로 쓰고 학습 입력·이름만 다른지, 학습한 체크포인트로 G1·G2와 T1~T7을 한 번씩 진단하는지(main job은 그 뒤에
같은 하드웨어 기준 — 학습 없는 2B와 v0을 G1·G2에 — 을 optional로 더 잰다, 42dc431; mix job은 자기 체크포인트만), 봉인 세트(T1~T7·G1·G2)가 어느 job의 학습 입력도 아닌지, 섞기용 학습 입력이
`convert_single_probe.py mix`로 봉인되지 않은 재료에서만 만들어지는지. v0.1a의 train-50k가 아직 없으면 가짜 파일로 한다. 모델·데이터는 싣지 않는다.
"""

from __future__ import annotations

import functools
import importlib.util
import json
import shutil
import sys
from pathlib import Path

import pytest
import yaml
from helpers import FIXTURES, REPO

from robo_jev.decisions.records import read_records
from robo_jev.decisions.single_probe import G1_SPLITS, MIX_SPLITS, convert_records, overlap
from robo_jev.decisions.train import load_training_records, resolve_config
from robo_jev.launch.jobs import expand, load_job

JOBS = {"main": REPO / "configs" / "cloud" / "decisions-v01-main-h100.yaml", "mix": REPO / "configs" / "cloud" / "decisions-v01-mix-h100.yaml"}
V0_MAIN = REPO / "configs" / "cloud" / "decisions-v0-main-h100.yaml"
TRAIN_INPUT = {"main": "artifacts/simgen/v01/train-50k.jsonl", "mix": "artifacts/simgen/v01-mix/train-50k-mix.jsonl"}
RUN_NAME = {"main": "v01-main-2b", "mix": "v01-mix-2b"}
TESTS = [f"artifacts/simgen/v0b/tests/T{i}.jsonl" for i in range(1, 8)]
G1 = "artifacts/simgen/probes/G1.jsonl"
G2 = "artifacts/simgen/probes/G2.jsonl"  # 공개 BoolQ·MultiNLI 표본 (통제자 결정 2026-10-12)
SEALED = (*TESTS, G1, G2)
CONTEXT = {"python": "python", "uv": "uv", "repo": "/r", "run_dir": "/r/run", "runs": "/r/run/runs", "reports": "/r/run/reports", "run_id": "x"}
PROBES = REPO / "artifacts" / "simgen" / "probes"
#: main job의 같은 하드웨어 기준(통제자 결정 2026-10-12): v0 adapter — 원래 자리 그대로 입력으로 올린다.
V0_ADAPTER = "artifacts/cloud/runs/v0-main-01-fetched/runs/v0-main-2b/checkpoint.pt"
V0_ADAPTER_SHA256 = "4859a972027fa4d9c24caaa92a372139a0e55ff850e9299377709511f120f091"
#: main job에만 있는 기준 단계 (이름, 체크포인트, 입력, 출력) — v0.1 모델의 진단이 모두 끝난 뒤.
BASELINES = [
    ("base-G1", None, G1, "/r/run/reports/base-2b-G1.json"),
    ("base-G2", None, G2, "/r/run/reports/base-2b-G2.json"),
    ("v0-G1", V0_ADAPTER, G1, "/r/run/reports/v0-2b-G1.json"),
    ("v0-G2", V0_ADAPTER, G2, "/r/run/reports/v0-2b-G2.json"),
]


@functools.lru_cache(maxsize=None)
def _script(name: str):
    spec = importlib.util.spec_from_file_location(f"v01_{Path(name).stem}", REPO / "scripts" / name)
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


@pytest.mark.parametrize("which", sorted(JOBS))
def test_each_v01_job_trains_once_with_the_v0_recipe(which):
    config = _train(JOBS[which])
    reference = _train(V0_MAIN)
    assert config["train_input"] == TRAIN_INPUT[which] and config["run_name"] == RUN_NAME[which]
    differ = sorted(k for k in config if config[k] != reference[k])
    assert differ == ["run_name", "train_input"], f"v0 조리법에서 학습 입력·이름 말고는 바꾸지 않는다: {differ}"
    assert config["mix"] == [], "섞기는 데이터 이어 붙이기로 한다 — 학습기의 mix 자리는 그대로 비어 있다"


@pytest.mark.parametrize("which", sorted(JOBS))
def test_each_v01_job_diagnoses_its_own_checkpoint_on_g1_g2_then_the_seven_sealed_sets(which):
    diagnoses = _diagnoses(JOBS[which])
    own = diagnoses[: 2 + len(TESTS)]
    assert [str(args.input) for _, args in own] == [G1, G2, *TESTS], "G1·G2가 먼저, 그다음 T1~T7 — 한 번씩"
    checkpoint = f"/r/run/runs/{RUN_NAME[which]}/checkpoint.pt"
    for name, args in own:
        assert str(args.checkpoint) == checkpoint, f"{name}: v0.1 모델 자신의 체크포인트"
    rest = [name for name, _ in diagnoses[len(own):]]
    assert rest == ([b[0] for b in BASELINES] if which == "main" else []), "같은 하드웨어 기준은 main에만, 자기 진단이 모두 끝난 뒤에"
    for name, args in diagnoses:
        assert not args.preview and args.model == "Qwen/Qwen3.5-2B"
        # Spark의 G1 기준·v0 run과 같은 채점 조건 (진단 도구의 기본값)
        assert (args.prompt_version, args.fmt, args.orders, args.mode, args.shuffle_seed, args.batch, args.limit) == ("v0c-p3", "chat", "counterbalanced", "direct", 1, 1, None)
    outs = [str(args.out) for _, args in diagnoses]
    assert len(set(outs)) == len(outs) and all(out.startswith("/r/run/reports/") for out in outs)
    spec, _ = _steps(JOBS[which])
    assert set(SEALED) <= set(spec["job"]["inputs"]) and spec["budget"]["max_wall_hours"] == 6
    t6 = next(step for step in spec["job"]["steps"] if step["name"] == "after-T6")
    assert t6["max_minutes"] >= 80, "v0의 T6 진단은 H100에서 53.8분 — 60분 상한은 빠듯했다"


def test_the_main_job_adds_optional_same_hardware_baselines_on_g1_and_g2_after_its_own_results():
    spec, steps = _steps(JOBS["main"])
    job = spec["job"]
    names = [step["name"] for step in job["steps"]]
    assert names[-len(BASELINES):] == [b[0] for b in BASELINES] and names.index("after-T7") < names.index("base-G1")
    by_name = {step["name"]: step for step in job["steps"]}
    parsed = dict(_diagnoses(JOBS["main"]))
    for name, checkpoint, probe, out in BASELINES:
        args = parsed[name]
        assert by_name[name]["optional"] is True, f"{name}: 기준 단계의 실패가 본 결과를 막지 않는다"
        assert (str(args.checkpoint) if args.checkpoint else None) == checkpoint and str(args.input) == probe and str(args.out) == out
        assert args.model == "Qwen/Qwen3.5-2B" == job["model"]
    assert V0_ADAPTER in job["inputs"], "v0 adapter는 job inputs로 올라가야 pod에 있다"
    assert V0_ADAPTER_SHA256 in JOBS["main"].read_text(encoding="utf-8"), "머리 주석에 adapter의 sha256을 적는다"
    adapter = REPO / V0_ADAPTER
    if adapter.is_file():  # 원래 파일 그대로인가 (고치거나 줄이지 않았다)
        from robo_jev.decisions.records import file_sha256

        assert file_sha256(adapter) == V0_ADAPTER_SHA256
    assert V0_ADAPTER not in load_job(JOBS["mix"])["job"]["inputs"], "mix job은 그대로다"


def test_the_two_v01_jobs_differ_only_in_the_training_input_and_the_names():
    def normalised(which: str) -> str:
        spec = load_job(JOBS[which])
        if which == "main":  # main에만 있는 같은 하드웨어 기준(단계 넷과 v0 adapter 입력)을 빼고 견준다
            spec["job"]["steps"] = [step for step in spec["job"]["steps"] if step["name"] not in {b[0] for b in BASELINES}]
            spec["job"]["inputs"] = [path for path in spec["job"]["inputs"] if path != V0_ADAPTER]
        spec["estimate"] = {k: v for k, v in spec["estimate"].items() if k != "assumption"}  # 가정 글은 main에 기준 단계의 몫이 더 적혀 있다
        text = json.dumps({k: v for k, v in spec.items() if k != "name"}, sort_keys=True, ensure_ascii=False)
        return text.replace(TRAIN_INPUT[which], "<train>").replace(RUN_NAME[which], "<run>").replace("after-v01mix-", "after-<tag>-").replace("after-v01-", "after-<tag>-")

    assert normalised("main") == normalised("mix")


@pytest.mark.parametrize("job_path", sorted((REPO / "configs" / "cloud").glob("*.yaml")), ids=lambda p: p.stem)
def test_no_shipped_job_trains_on_a_sealed_set(job_path):
    for name, argv in _steps(job_path)[1]:
        if argv[1:2] != ["scripts/train_decisions.py"]:
            continue
        args = _script("train_decisions.py").build_parser().parse_args(argv[2:])
        train_input = str(resolve_config(_script("train_decisions.py")._read_config(REPO / args.config, args.set))["train_input"])
        assert train_input not in SEALED and "/tests/" not in train_input and "probes/G1.jsonl" not in train_input and "probes/G2" not in train_input, (
            f"{job_path.name}:{name}")


def test_the_mix_training_input_is_built_only_from_unsealed_material(tmp_path):
    """v0.1a의 train-50k가 아직 없어도 된다 — 가짜 학습 파일(진단 fixture)로 `mix`를 돌려, 섞인 파일에 봉인 G1의 분할·계보·id가 없는지 본다."""
    fake_v01 = tmp_path / "train-50k.jsonl"
    shutil.copyfile(FIXTURES / "decisions" / "requests.jsonl", fake_v01)
    if (PROBES / "G1-train-mix.jsonl").is_file() and (PROBES / "G1.jsonl").is_file():
        probe, sealed = PROBES / "G1-train-mix.jsonl", PROBES / "G1.jsonl"
    else:  # 변환을 돌리지 않은 체크아웃 — 손 레코드로
        from test_single_probe import judgment, mixed_record, boolean, single

        sealed = tmp_path / "G1.jsonl"
        sealed.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in convert_records([mixed_record()], splits=G1_SPLITS)[0]), encoding="utf-8")
        train = judgment([boolean("q_b")], [single("q_b", True)], rid="rules-0009-0", split="train", origin="rules/x/0009", domain="rules")
        probe = tmp_path / "G1-train-mix.jsonl"
        probe.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in convert_records([train], splits=MIX_SPLITS)[0]), encoding="utf-8")
    out = tmp_path / "train-50k-mix.jsonl"
    assert _script("convert_single_probe.py").main(["mix", "--base", str(fake_v01), "--probe", str(probe), "--sealed", str(sealed),
                                                    "--train-config", "configs/train/decisions-v0-lora-h100.yaml", "--out", str(out)]) == 0  # fmt: skip
    mixed, check = load_training_records(out)
    assert check["invalid"] == 0 and len(mixed) == len(read_records(fake_v01)) + len(read_records(probe))
    probe_part = [r for r in mixed if (r.get("meta") or {}).get("source_record")]
    assert probe_part and {r["meta"]["split"] for r in probe_part} <= set(MIX_SPLITS)
    g1 = read_records(sealed)
    assert overlap(g1, probe_part) == {"origin_groups": [], "source_records": []}
    assert not {r["id"] for r in g1} & {r["id"] for r in mixed}
    manifest = json.loads(out.with_name("train-50k-mix.manifest.json").read_text(encoding="utf-8"))
    assert 0 < manifest["examples_per_epoch"]["probe_share"] < 1 and manifest["sealed_check"]["overlap"]["origin_groups"] == 0
    assert yaml.safe_load(JOBS["mix"].read_text(encoding="utf-8"))["job"]["inputs"][0] == TRAIN_INPUT["mix"]
