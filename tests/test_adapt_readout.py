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
    assert (t1["trainable"], t1["stream_chunk_seconds"], t1["activation_checkpointing"], t1["lora"]) == ("text_backbone_and_readout", 10, True, None)
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


def test_the_4b_sibling_extends_the_2b_config_and_only_moves_the_model_and_the_chunk():
    two, four = _resolved(TRAIN_2B, "lora"), _resolved(TRAIN_4B, "lora")
    assert four["model_id"] == "Qwen/Qwen3.5-4B" and two["model_id"] == "Qwen/Qwen3.5-2B"
    differences = {key for key in two if two[key] != four[key]}
    assert differences <= {"model_id", "tokenizer", "run_name", "run_id", "stream_chunk_seconds"}
    assert four["stream_chunk_seconds"] == 5 and two["stream_chunk_seconds"] == 5  # LoRA는 둘 다 5초
    assert _resolved(TRAIN_4B, "t1")["stream_chunk_seconds"] == 5  # 4B의 10초 full은 울타리를 넘는다 (G0b 사다리)
    assert _resolved(TRAIN_2B, "t1")["stream_chunk_seconds"] == 10


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
        assert by_name[name]["limit"] % 2 == 0  # 대조 쌍은 적재 순서로 붙어 있다 — 짝수여야 쌍이 잘리지 않는다
    assert yaml.safe_load(EVAL_PILOT.read_text(encoding="utf-8"))["tiny_scorer_report"].endswith("tiny-scorer.json")
