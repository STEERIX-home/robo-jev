"""`scripts/adapt_readout.py`의 설정 해석과 저장소의 파일럿 설정 — CPU에서, 가중치 없이 (Task P1 stage B).

여기서 보는 것은 (1) `extends`·`modes`가 무엇을 덮어쓰는지, (2) `--dataset`이 학습 manifest만 바꾸는지,
(3) 저장소의 `configs/train/qwen35-{2b,4b}-pilot.yaml`이 `robo_jev.train.resolve_config`를 통과하고 브리프가 정한
실측값(readout_lr 3e-4, backbone_lr 5e-5, 누적 2, 구간 10/5초 …)을 갖는지, (4) `configs/eval/*.yaml`이 읽히고
봉인 분할이 없는지다. 실제 학습·평가는 GPU의 몫이다.
"""

import functools
import importlib.util
import sys

import pytest
import yaml
from helpers import REPO

from robo_jev.evaluate import load_eval_suite
from robo_jev.train import resolve_config

SCRIPT = REPO / "scripts" / "adapt_readout.py"
TRAIN_2B = REPO / "configs" / "train" / "qwen35-2b-pilot.yaml"
TRAIN_4B = REPO / "configs" / "train" / "qwen35-4b-pilot.yaml"
EVAL_PILOT = REPO / "configs" / "eval" / "pilot.yaml"
EVAL_BATCH0 = REPO / "configs" / "eval" / "batch0-g0b.yaml"


@functools.lru_cache(maxsize=1)
def script():
    spec = importlib.util.spec_from_file_location("adapt_readout", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _resolved(path, mode, **kwargs):
    return resolve_config(script().load_train_config(path, mode=mode, run_id="test", **kwargs))


def test_modes_change_only_the_three_things_that_differ_between_t0_lora_and_t1():
    """모드는 학습 범위·구간·activation checkpointing(과 LoRA 블록)만 바꾼다 — lr·seed·데이터·혼합은 같아야 비교가 된다."""
    t0, lora, t1 = (_resolved(TRAIN_2B, mode) for mode in ("t0", "lora", "t1"))
    assert (t0["trainable"], t0["stream_chunk_seconds"], t0["activation_checkpointing"], t0["lora"]) == ("readout_only", 10, False, None)
    assert (lora["trainable"], lora["stream_chunk_seconds"], lora["activation_checkpointing"]) == ("lora_and_readout", 5, True)
    assert lora["lora"]["r"] == 16 and lora["lora"]["alpha"] == 32
    # T1의 구간은 10초 → 5초다 (P2 A2: fp32 master가 +14.02 GiB라 10초는 40 step에서 73 GiB 울타리를 넘는다)
    assert (t1["trainable"], t1["stream_chunk_seconds"], t1["activation_checkpointing"], t1["lora"]) == ("text_backbone_and_readout", 5, True, None)
    assert t0["fp32_master_weights"] is lora["fp32_master_weights"] is t1["fp32_master_weights"] is True
    shared = ("readout_lr", "weight_decay", "gradient_clip", "warmup_ratio", "gradient_accumulation",
              "robot_loss_share", "nonrobot_tokens_per_unit", "stream_window_ticks", "seed", "sampler", "dataset_manifests", "model_id")
    for key in shared:
        assert t0[key] == lora[key] == t1[key], key
    # backbone_lr만 T1에서 다르다 — 5e-5(LoRA)는 2B 전체 학습에서 3 step 만에 발산했고(P1 stage C2: 손실 2.63 → 28.93),
    # T1은 docs/03 §5의 계획값 1e-5를 쓴다. 설정 파일이 그 이유를 적는다.
    assert t0["backbone_lr"] == lora["backbone_lr"] == pytest.approx(5e-5) and t1["backbone_lr"] == pytest.approx(1e-5)
    assert "28.93" in TRAIN_2B.read_text(encoding="utf-8")


def test_the_pilot_config_carries_the_measured_values_not_the_planned_ones():
    """docs/03 §5의 계획값(backbone 1e-5 · readout 1e-4 · 누적 4)이 아니라 이 상자에서 잰 값이다 — 설정 주석이 그 차이를 적는다."""
    config = _resolved(TRAIN_2B, "t0")
    assert config["readout_lr"] == pytest.approx(3e-4) and config["backbone_lr"] == pytest.approx(5e-5)
    assert config["gradient_accumulation"] == 2 and config["weight_decay"] == 0.01 and config["gradient_clip"] == 1.0
    assert config["warmup_ratio"] == 0.05 and config["robot_loss_share"] == 0.6 and config["nonrobot_tokens_per_unit"] == 8192
    assert config["stream_window_ticks"] == 30 and config["seed"] == 17 and config["sampler"]["permute_candidates_seed"] == 17
    assert config["sampler"]["tick_weights"] == {"steady": 0.25, "event": 2.0, "goal_change": 2.0, "other": 1.0}
    assert config["dtype"] == "bfloat16" and config["readout_dtype"] == "float32" and config["readout_rank"] == 64
    text = TRAIN_2B.read_text(encoding="utf-8")
    assert "1e-5" in text and "docs/03 §5" in text  # 계획값과 다른 이유가 파일에 적혀 있다
    robot = next(entry for entry in config["dataset_manifests"] if entry["domain"] == "robot")
    assert robot["path"].endswith("d1-robot/d1-rollout-labels/manifest.json") and robot["files"] == ["episodes/*/streams.jsonl"]
    assert any(entry["domain"] == "non_robot" and entry["path"].endswith("d1/single/manifest.json") for entry in config["dataset_manifests"])
    assert config["splits"] == ["train"]


def test_the_4b_sibling_extends_the_2b_config_and_moves_exactly_two_things():
    """4B 파일이 덮어쓰는 것은 이제 `model_id`와 `run_name` **둘뿐**이다 (P2 A2).

    셋째였던 `modes.t1.stream_chunk_seconds: 5`는 2B가 fp32 master 때문에 5초로 내려오면서 없앴다 — 4B는 그 값을
    그대로 물려받는다. 세 모드 모두에서 **구간이 완전히 같다**.
    """
    for mode in ("t0", "lora", "t1"):
        two, four = _resolved(TRAIN_2B, mode), _resolved(TRAIN_4B, mode)
        assert four["model_id"] == "Qwen/Qwen3.5-4B" and two["model_id"] == "Qwen/Qwen3.5-2B"
        # `tokenizer`·`run_id`는 값이 같거나 실행 시각에서 나온다 — 실제로 갈리는 것은 model_id와 run_name뿐이다.
        assert {key for key in two if two[key] != four[key]} <= {"model_id", "run_name", "run_id"}, mode
        assert four["stream_chunk_seconds"] == two["stream_chunk_seconds"], mode
    assert _resolved(TRAIN_2B, "t1")["stream_chunk_seconds"] == _resolved(TRAIN_4B, "t1")["stream_chunk_seconds"] == 5
    # 4B의 T1이 이 상자에서 돌 수 없는 이유(fp32 master 바닥 94.0 GiB)가 파일에 적혀 있다
    assert "94.0 GiB" in TRAIN_4B.read_text(encoding="utf-8")


def test_dataset_switch_changes_only_the_training_manifests():
    d1, batch0 = _resolved(TRAIN_2B, "t0"), _resolved(TRAIN_2B, "t0", dataset="batch0")
    assert [entry["domain"] for entry in batch0["dataset_manifests"]] == [None, "robot", "non_robot"]
    assert any("batch-0" in entry["path"] for entry in batch0["dataset_manifests"])
    assert {key for key in d1 if d1[key] != batch0[key]} == {"dataset_manifests"}


def test_steps_seed_and_overrides_are_applied_and_unknown_keys_are_refused():
    config = _resolved(TRAIN_2B, "t0", steps=7, seed=5)
    assert config["max_steps"] == 7 and config["checkpoint_every"] == 7 and config["seed"] == 5
    assert config["sampler"]["permute_candidates_seed"] == 5  # 치환 증강 seed는 run seed를 따라간다
    assert _resolved(TRAIN_2B, "t0", overrides={"torch_threads": 4})["torch_threads"] == 4
    with pytest.raises(ValueError, match="알 수 없는 키"):
        _resolved(TRAIN_2B, "t0", overrides={"nonsense": 1})
    with pytest.raises(ValueError, match="modes에"):
        script().load_train_config(TRAIN_2B, mode="t9")


@pytest.mark.parametrize("path", [EVAL_PILOT, EVAL_BATCH0])
def test_the_eval_configs_load_and_never_touch_the_sealed_split(path):
    suite = load_eval_suite(path)
    assert suite["splits"] and suite["columns"]["state_shuffle"] and suite["columns"]["instruction_shuffle"]
    assert all(entry["split"] != "ood_test" for entry in suite["splits"])
    assert suite["fused"] is True and suite["shuffle_seed"] == 1


def test_the_pilot_eval_set_is_fixed_by_name_and_keeps_test_out_of_selection():
    suite = load_eval_suite(EVAL_PILOT)
    by_name = {entry["name"]: entry for entry in suite["splits"]}
    assert {"robot/dev", "robot/ood_dev", "robot/test", "robot_contrast/dev", "robot_contrast/ood_dev", "non_robot/dev", "non_robot/ood_dev"} == set(by_name)
    assert by_name["robot/test"]["selection"] is False and by_name["robot/dev"].get("selection", True) is True
    for name in ("robot/dev", "robot/ood_dev", "robot/test"):
        records = by_name[name]["records"]
        assert records and len(set(records)) == len(records) and all(record.startswith("ep-E") for record in records)
        assert "max_ticks" not in by_name[name]  # 로봇은 에피소드를 통째로 (지시 변경 틱이 중반에 있다)
    for name in ("robot_contrast/dev", "robot_contrast/ood_dev"):
        # 대조 쌍은 **이름으로** 고른다: 적재 순서 앞쪽은 E0의 forbidden·zone_boundary뿐이라 지시 대조 쌍(E1에만 있다)이
        # 들어오지 않는다. base와 sibling이 반드시 짝이고 세 종류가 모두 있어야 한다.
        records = by_name[name]["records"]
        assert len(records) % 2 == 0 and len(set(records)) == len(records)
        bases = [record for record in records if record.endswith("-base")]
        assert len(bases) * 2 == len(records)
        assert all(f"{base[:-len('-base')]}" in records for base in bases)
        kinds = {base.split("-")[-2] for base in bases}
        assert {"forbidden", "zone_boundary", "instruction"} == kinds, kinds
    assert yaml.safe_load(EVAL_PILOT.read_text(encoding="utf-8"))["tiny_scorer_report"].endswith("tiny-scorer.json")


def test_the_zero_shot_defaults_reproduce_the_runs_that_were_recorded():
    """무학습 run의 기본값이 실제로 돌린 값과 같아야 명령줄이 그 run을 재현한다 (P1 리뷰 1 M12).

    P1의 두 무학습 run은 **16틱마다**(G0b의 값) 쟀는데 `--tick-stride`의 기본값은 8이었고, 후보 순서 치환 seed는
    평가 집합의 `shuffle_seed`가 아니라 함수 안에 1로 박혀 있었다. 기본은 집합이 정하게 둔다.
    """
    import inspect

    module = script()
    assert module.ZERO_SHOT_TICK_STRIDE == 16
    assert module.build_parser().parse_args(["--out", "x.json"]).tick_stride == module.ZERO_SHOT_TICK_STRIDE
    parameters = inspect.signature(module.run_zero_shot).parameters
    assert parameters["tick_stride"].default == module.ZERO_SHOT_TICK_STRIDE
    assert parameters["shuffle_seed"].default is None  # None = 평가 집합의 shuffle_seed를 쓴다


def test_run_training_hands_the_resume_checkpoint_to_the_trainer(monkeypatch, tmp_path):
    """**아무 일도 하지 않는 설정 키는 없어야 한다** (Task R3a C1의 값비싼 교훈).

    `resolve_config`는 `resume`을 받아들이므로 설정이나 `--set resume=…`은 아무 불평 없이 통과했는데,
    `run_training`이 `Trainer(config)`를 `resume=` 없이 불러 그 키를 **조용히 버리고** 있었다. 그래서 "이어서
    233 step"이라고 적힌 run이 실제로는 처음부터 466 step을 돌았고, 4.3 GPU-h가 그 차이를 메우는 데 갔다.
    """
    import robo_jev.train as train_module

    module = script()
    seen: dict = {}

    class FakeTrainer:
        def __init__(self, config, *, resume=None):
            seen["config"], seen["resume"] = config, resume
            self.config, self.tokenizer, self.model = config, object(), None
            self.step_hook = None

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def run(self):
            return {"run_id": "r", "checkpoint": str(tmp_path / "c.pt"), "status": "completed",
                    "metrics": {"steps": [{"step": 1, "loss": 1.0, "loss_by_domain": {}, "loss_by_type": {},
                                           "grad_norm": 0.0, "lr": {}, "tokens": {"total": 4}, "seconds": 1.0}]}}

        manifest = {"model": {}, "contract_sha256": "x", "serializer_version": "v", "tokenizer": {}}
        items: list = []

    monkeypatch.setattr(train_module, "Trainer", FakeTrainer)
    monkeypatch.setattr(module, "_memory", lambda: {"peak_allocated_bytes": 0, "peak_reserved_bytes": 0,
                                                    "num_alloc_retries": 0, "num_ooms": 0})
    import torch

    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda *a, **k: None)
    monkeypatch.setattr(torch.cuda, "reset_accumulated_memory_stats", lambda *a, **k: None)
    config = {"model_id": "m", "max_steps": 466, "resume": str(tmp_path / "from.pt"),
              "dataset_manifests": [{"path": str(tmp_path / "m.json"), "domain": "robot"}]}

    module.run_training(dict(config), mode="t0", eval_after=False, log_stream=None)
    assert seen["resume"] == config["resume"]          # 넘어간다

    seen.clear()
    module.run_training({k: v for k, v in config.items() if k != "resume"}, mode="t0", eval_after=False, log_stream=None)
    assert seen["resume"] is None                       # 없으면 None — 새 run이다
