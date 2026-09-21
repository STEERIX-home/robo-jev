"""`scripts/p1_acceptance.py`의 세 인수 검사에 붙는 시험 — 작은 fixture에서, GPU 없이 (P1 리뷰 1 M3).

P1의 C1·C2·C3는 일회성 스크립트였고 `tests/`에 아무 것도 없었다(`grep -rl p1_acceptance tests/`가 비어 있었다):
누가 스크립트를 돌릴 때만 검사가 돌고, `Trainer.load`가 망가져도 CI는 초록이었다. 여기서 고정하는 것은 셋이다.

1. **T0는 backbone을 한 비트도 바꾸지 않는다** — 표본이 아니라 backbone tensor **전부**의 sha256을 step 전후로
   대조한다(스크립트의 :func:`_hash` 그대로).
2. **학습 대상은 실제로 움직인다** — gradient가 0이 아니고 고정 표본이 전부 움직인다(스크립트의 판정 기준
   :data:`MOVE_EPSILON` 그대로). fp32 master 아래에서 **기대치만큼** 움직이는지는 `tests/test_train.py`의 A3 짝이 본다.
3. **재개 동등성의 판정**(:func:`compare_resume`) — 통과해야 할 때 통과하고, 여섯 가지 어긋남마다 떨어진다.

실제 2B가 있어야만 할 수 있는 것(LoRA adapter, BF16 backbone의 fp32 master, 에피소드 중간 구간 위치)은 GPU 게이트의
몫이고, 그 게이트는 이제 `--resume-modes t0,lora,t1`로 학습 범위를 받는다 (P1은 T0만 덮었다).
"""

import copy
import functools
import importlib.util
import sys

import pytest
import torch
from helpers import D0_MANIFEST, REPO, SMALL_VOCAB

from robo_jev.train import Trainer

SCRIPT = REPO / "scripts" / "p1_acceptance.py"


@functools.lru_cache(maxsize=1)
def script():
    spec = importlib.util.spec_from_file_location("p1_acceptance", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def fixture_config(tmp_path, **overrides) -> dict:
    """스크립트가 실제 2B에서 쓰는 설정의 소형 대응 — 에피소드를 20틱으로 자르고 1초 구간 둘."""
    config = {
        "run_name": "acceptance",
        "run_id": "acceptance-under-test",
        "model_config": str(REPO / "configs" / "model" / "tiny_hybrid.yaml"),
        "model_vocab_size": SMALL_VOCAB,
        "dataset_manifest": str(D0_MANIFEST),
        "splits": ["train"],
        "tokenizer": "whitespace",
        "stream_chunk_seconds": 1,
        "stream_window_ticks": 30,
        "stream_max_ticks": 20,
        "trainable": "text_backbone_and_readout",
        "gradient_accumulation": 2,
        "nonrobot_tokens_per_unit": 512,
        "max_steps": 1,
        "warmup_ratio": 0.5,
        "seed": 17,
        "torch_threads": 1,
        "artifacts_dir": str(tmp_path / "runs"),
    }
    config.update(overrides)
    return config


# --------------------------------------------------------------------------
# C1 — T0는 backbone을 얼린다 (표본이 아니라 전부)
# --------------------------------------------------------------------------


def test_the_acceptance_hash_catches_a_single_grid_step_in_a_bf16_tensor():
    """`_hash`는 바이트를 본다 — 값이 한 눈금만 달라도 다른 해시여야 C1이 "0개 바뀌었다"를 말할 자격이 있다."""
    _hash = script()._hash
    base = torch.full((16,), 0.03, dtype=torch.bfloat16)
    moved = base.clone()
    moved[3] = torch.tensor(0.03 + 1.220703125e-4, dtype=torch.bfloat16)  # bf16 한 눈금 (2⁻¹³)
    assert _hash(base) == _hash(base.clone()) and _hash(base) != _hash(moved)
    assert _hash(base) != _hash(base.to(torch.float32))  # dtype이 다르면 바이트도 다르다


def test_t0_leaves_every_backbone_tensor_bit_identical_and_moves_every_readout_tensor(tmp_path):
    module = script()
    with Trainer(fixture_config(tmp_path, trainable="readout_only")) as trainer:
        model = trainer.model
        backbone = dict(model.backbone.named_parameters())
        assert len(backbone) > 10  # 표본이 아니라 전부를 본다
        before = {name: module._hash(parameter) for name, parameter in backbone.items()}
        readout_before = {name: parameter.detach().clone() for name, parameter in model.named_parameters() if not name.startswith("backbone.")}
        metrics = trainer.run_step()
        after = {name: module._hash(parameter) for name, parameter in model.backbone.named_parameters()}
        assert [name for name in before if before[name] != after[name]] == []
        assert [name for name, p in model.backbone.named_parameters() if p.requires_grad] == []
        moved = {name: float((p.detach() - readout_before[name]).abs().max()) for name, p in model.named_parameters() if not name.startswith("backbone.")}
        assert set(moved) == {"U.weight", "V.weight", "bias"}
        assert all(value > module.MOVE_EPSILON for value in moved.values()), moved
        assert metrics["grad_norm"] > 0


def test_t1_gives_every_trainable_tensor_a_nonzero_gradient_and_moves_it(tmp_path):
    """C2의 판정 기준 그대로 — gradient가 0이 아니고, 고정 표본이 **전부** 움직인다."""
    module = script()
    with Trainer(fixture_config(tmp_path)) as trainer:
        model = trainer.model
        trainable = [(name, p) for name, p in model.backbone.named_parameters() if p.requires_grad]
        picked = [trainable[index] for index in range(0, len(trainable), max(1, len(trainable) // 8))][:8]
        before = {name: p.detach().clone().float() for name, p in picked}
        assert trainer.accumulate()
        nonzero = sum(1 for _, p in trainable if p.grad is not None and float(p.grad.abs().max()) > 0)
        assert nonzero == len(trainable) > 0
        trainer.apply()
        delta = {name: float((p.detach().float() - before[name]).abs().max()) for name, p in picked}
        assert all(value > module.MOVE_EPSILON for value in delta.values()), delta


# --------------------------------------------------------------------------
# C3 — 재개 동등성의 **판정**
# --------------------------------------------------------------------------


def _snapshot_pair(steps: int = 20) -> tuple[dict, dict]:
    """같은 run을 두 번 찍은 것처럼 만든 두 snapshot (`_snapshot`이 남기는 자리 그대로)."""
    snapshot = {
        "parameters": {"U.weight": torch.arange(6, dtype=torch.float32).reshape(2, 3), "bias": torch.zeros(1)},
        "sampler": {"drawn": 20, "cursors": {"robot": 3}, "rng": [1, 2, 3]},
        "optimizer_steps": steps,
        "step": steps,
        "losses": [2.0 - index * 0.05 for index in range(steps)],
        "units": [[f"ep-{index}"] for index in range(steps)],
    }
    return copy.deepcopy(snapshot), copy.deepcopy(snapshot)


def test_the_resume_comparison_passes_on_two_identical_snapshots():
    module = script()
    continuous, split = _snapshot_pair()
    result = module.compare_resume(continuous, split, steps=20)
    assert result["passed"] and result["worst_loss_abs"] == 0.0 and result["worst_param_max_abs"] == 0.0
    assert result["bit_identical_tensors"] == result["tensors"] == 2
    assert result["sampler_position_equal"] and result["drawn_units_equal"]
    assert result["tolerance"] == module.RESUME_TOLERANCE  # 허용 오차는 비교 전에 고정되어 있다


@pytest.mark.parametrize(
    "what",
    ["loss", "parameter", "sampler", "units", "optimizer_steps", "missing_tensor", "short_run"],
)
def test_the_resume_comparison_fails_on_each_way_a_resumed_run_can_differ(what):
    module = script()
    continuous, split = _snapshot_pair()
    if what == "loss":
        split["losses"][7] += 0.5  # 허용 오차(0.02 절대 / 2 % 상대) 밖
    elif what == "parameter":
        split["parameters"]["U.weight"] = split["parameters"]["U.weight"] + 1.0
    elif what == "sampler":
        split["sampler"]["cursors"]["robot"] = 4
    elif what == "units":
        split["units"][11] = ["ep-other"]
    elif what == "optimizer_steps":
        split["optimizer_steps"] = 19
    elif what == "missing_tensor":
        del split["parameters"]["bias"]
    else:
        split["step"] = 19
    assert not module.compare_resume(continuous, split, steps=20)["passed"], what


def test_a_loss_difference_inside_the_tolerance_still_passes():
    """관문은 비트 동일을 요구하지 않는다 — 다른 프로세스가 다른 kernel을 고를 수 있기 때문이다 (스크립트의 docstring)."""
    module = script()
    continuous, split = _snapshot_pair()
    split["losses"][3] += module.RESUME_TOLERANCE["loss_abs"] / 2
    result = module.compare_resume(continuous, split, steps=20)
    assert result["passed"] and 0 < result["worst_loss_abs"] <= module.RESUME_TOLERANCE["loss_abs"]
    assert result["bit_identical_tensors"] == 2


# --------------------------------------------------------------------------
# 게이트가 무엇을 덮는가
# --------------------------------------------------------------------------


def test_the_resume_gate_takes_the_training_scope_instead_of_always_running_t0():
    """P1의 게이트는 `_config("t0", …)`가 박혀 있어 readout tensor 셋만 지났다 (리뷰 1 M3)."""
    import inspect

    module = script()
    for name in ("run_resume_phase", "check_resume"):
        signature = inspect.signature(getattr(module, name))
        assert signature.parameters["mode"].default == "t0", name  # 기본은 P1과 같고, lora·t1을 줄 수 있다
    source = inspect.getsource(module.run_resume_phase)
    assert '_config("t0"' not in source and "_config(mode," in source
    assert "--resume-modes" in inspect.getsource(module.main)


def test_the_trainer_refuses_to_resume_a_run_that_changed_the_master_weight_setting(tmp_path):
    """`fp32_master_weights`는 run의 정체다 — 켜고 끄면 같은 run을 이어갈 수 없다(갱신 규칙이 달라진다)."""
    config = fixture_config(tmp_path, max_steps=2, fp32_master_weights=True)
    with Trainer(config) as trainer:
        trainer.run_step()
        checkpoint = trainer.save()
    with pytest.raises(ValueError, match="fp32_master_weights"):
        Trainer({**config, "fp32_master_weights": False}, resume=checkpoint)
    resumed = Trainer(config, resume=checkpoint)  # 같은 설정이면 이어간다
    assert resumed.step == 1
    resumed.close()
