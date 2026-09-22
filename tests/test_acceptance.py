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


def test_the_master_weight_flag_is_run_identity_only_where_it_makes_master_copies(tmp_path):
    """`fp32_master_weights`는 **사본이 생기는 경로(T1)에서만** run의 정체다 (P2 리뷰 1 M5).

    fixture 모델은 학습 대상이 전부 fp32라 사본이 생기지 않고, `build_optimizer`는 플래그가 어느 쪽이든 평범한
    `AdamW`를 돌려준다 — 그런 run에서 이 키로 거절하면 **키 자체가 없는 P2 이전 T0·LoRA checkpoint가 통째로
    되살릴 수 없게 된다**(`None != True`). 그래서 여기서는 켜고 끈 채로도 이어가고, 이어간 뒤가 같아야 한다.
    T1 쪽(사본이 생기므로 거절이 맞는 쪽)의 판정은 `tests/test_train.py`의
    `test_the_master_weight_flag_is_run_identity_only_where_it_changes_the_optimizer`가 잡는다.
    """
    config = fixture_config(tmp_path, max_steps=2, fp32_master_weights=True)
    with Trainer(config) as trainer:
        trainer.run_step()
        checkpoint = trainer.save()
    flipped = Trainer({**config, "fp32_master_weights": False}, resume=checkpoint)
    assert flipped.step == 1
    flipped.close()
    resumed = Trainer(config, resume=checkpoint)
    assert resumed.step == 1
    resumed.close()
    with pytest.raises(ValueError, match="readout_lr"):  # 다른 키는 예외 없이 그대로 거절한다
        Trainer({**config, "readout_lr": 9e-9}, resume=checkpoint)


# --------------------------------------------------------------------------
# 게이트가 게이트인가 (P2 리뷰 1, focus 5 / N4)
# --------------------------------------------------------------------------


def test_a_scope_without_a_pre_registered_tolerance_is_not_judged_by_another_scopes_tolerance():
    """T1에는 아직 **사전 등록된 허용 오차가 없다** — 그러면 게이트가 그렇게 말해야 한다 (P2 리뷰 1 focus 5).

    `RESUME_TOLERANCE`는 P1이 **T0** run에 대고 고정한 값이고, 그 T0 경로는 이 상자에서 비트 결정적이었다. T1은
    재시작 없이 돌린 두 프로세스의 loss가 이미 0.064 벌어진다. 그 run을 T0의 오차로 재고 `passed: false`라 적으면
    "재개가 깨졌다"로 읽힌다 — 실제로 깬 것은 **오차가 그 범위에 없다**는 사실이다. 본 값 뒤에 오차를 맞추는 것은
    이 프로젝트가 금하는 수이므로(P2 보고서 A1b), 등록되지 않은 범위는 `passed: None`으로 **멈춘다**.
    """
    module = script()
    continuous, split = _snapshot_pair()
    assert module.RESUME_TOLERANCES["t0"] == module.RESUME_TOLERANCE
    assert "lora" not in module.RESUME_TOLERANCES  # LoRA 범위는 아직 두 run 기준선을 재지 않았다

    t0 = module.compare_resume(continuous, split, steps=20, mode="t0")
    assert t0["passed"] is True and t0["verdict"] == "pass" and t0["tolerance"] == module.RESUME_TOLERANCE

    lora = module.compare_resume(continuous, split, steps=20, mode="lora")
    assert lora["passed"] is None and lora["verdict"] == "tolerance-unregistered" and lora["tolerance"] is None
    assert lora["exact_criteria_passed"] is True  # 정수 기준(위치·뽑힌 단위·step 수·빠진 tensor)은 그대로 잰다
    assert lora["worst_loss_abs"] == 0.0 and lora["tensors"] == 2
    # 무엇을 쟀는지는 남는다 — 등록된 오차가 생기면 그대로 다시 판정할 수 있게
    assert lora["would_pass_under"]["t0"] is True
    split["losses"][7] += 0.5
    broken = module.compare_resume(continuous, split, steps=20, mode="lora")
    assert broken["passed"] is None and broken["would_pass_under"]["t0"] is False
    # 정수 기준이 깨지면 오차와 무관하게 떨어진다 — 등록 여부가 면허가 되지는 않는다
    split["sampler"]["cursors"]["robot"] = 4
    assert module.compare_resume(continuous, split, steps=20, mode="lora")["passed"] is False


def _gate_report(tmp_path, **checks):
    import json

    path = tmp_path / f"acceptance-{len(list(tmp_path.glob('acceptance-*.json')))}.json"
    path.write_text(json.dumps({"task": "p1-acceptance", "checks": checks}, ensure_ascii=False), encoding="utf-8")
    return path


def test_the_launcher_gate_reads_the_key_for_the_scope_it_is_about_to_launch(tmp_path):
    """`checks.resume_t1`을 **아무도 읽지 않았다** (P2 리뷰 1 focus 5 / N4).

    자동 게이트는 `artifacts/scratch/p1/run_stage_d1_gated.sh` 하나뿐이고 그것이 보는 키는 `checks.resume`,
    곧 **T0의 것**이다. 긴 T1 run이 T0의 결과로 통과되고 있었다는 뜻이다. 게이트는 띄우려는 범위의 키를 읽고,
    그 자리가 없거나 판정이 없으면 **멈춰야** 한다.
    """
    module = script()
    report = _gate_report(
        tmp_path,
        resume={"check": "resume", "mode": "t0", "passed": True},
        resume_t1={"check": "resume", "mode": "t1", "passed": None, "verdict": "tolerance-unregistered"},
        resume_lora={"check": "resume", "mode": "lora", "passed": False, "verdict": "fail"},
    )
    t0 = module.resume_gate(report, "t0")
    assert t0["key"] == "resume" and t0["passed"] is True and t0["exit_code"] == 0
    t1 = module.resume_gate(report, "t1")
    assert t1["key"] == "resume_t1" and t1["passed"] is None and t1["exit_code"] == 3 and "tolerance" in t1["reason"]
    lora = module.resume_gate(report, "lora")
    assert lora["key"] == "resume_lora" and lora["passed"] is False and lora["exit_code"] == 2
    # 자리가 아예 없으면 통과가 아니라 멈춤이다 (오늘의 T1이 정확히 이 경우였다)
    empty = module.resume_gate(_gate_report(tmp_path, resume={"check": "resume", "mode": "t0", "passed": True}), "t1")
    assert empty["passed"] is None and empty["exit_code"] == 3 and "resume_t1" in empty["reason"]
    # 다른 범위의 결과가 그 자리에 적혀 있으면 믿지 않는다
    crossed = module.resume_gate(_gate_report(tmp_path, resume_t1={"check": "resume", "mode": "t0", "passed": True}), "t1")
    assert crossed["passed"] is None and crossed["exit_code"] == 3 and "mode" in crossed["reason"]
    # P1의 보고서에는 `mode` 키가 없다 — `checks.resume`에는 T0만 적혔으므로 그 조합만 t0로 읽는다
    legacy = _gate_report(tmp_path, resume={"check": "resume", "passed": True, "steps": 20})
    assert module.resume_gate(legacy, "t0")["exit_code"] == 0
    assert module.resume_gate(_gate_report(tmp_path, resume_t1={"check": "resume", "passed": True}), "t1")["exit_code"] == 3

    # CLI는 GPU를 건드리지 않고 같은 판정을 종료 코드로 돌려준다 (런처가 부르는 자리)
    assert module.main(["--gate", "t0", "--gate-report", str(report)]) == 0
    assert module.main(["--gate", "t1", "--gate-report", str(report)]) == 3
    assert module.main(["--gate", "lora", "--gate-report", str(report)]) == 2


# --------------------------------------------------------------------------
# P3 D — 허용 오차의 사전 등록: 두 run 기준선, 그리고 0에 가까운 분모
# --------------------------------------------------------------------------


def test_the_relative_l2_criterion_skips_a_tensor_whose_own_norm_is_below_the_floor():
    """P2 A1b에서 `bias` 한 tensor가 **절대 차이 4.85e-7**인데 비 0.16으로 걸렸다 — 분모가 ≈3e-6이었기 때문이다.

    그런 tensor는 상대 기준에서 빼고 절대 기준으로만 본다. 바닥 위의 tensor는 그대로 판정한다 — 가드가 기준을
    통째로 꺼 버리면 안 된다."""
    module = script()
    continuous, split = _snapshot_pair()
    continuous["parameters"]["bias"] = torch.full((1,), 3e-6)
    split["parameters"]["bias"] = torch.full((1,), 3e-6 + 4.85e-7)
    result = module.compare_resume(continuous, split, steps=20)
    rows = {row["tensor"]: row for row in result["parameters"]}
    assert rows["bias"]["relative_l2"] > module.RESUME_TOLERANCE["param_rel_l2"]  # 비는 그대로 적는다
    assert rows["bias"]["relative_l2_judged"] is False and result["relative_l2_not_judged"] == ["bias"]
    assert result["worst_param_relative_l2"] == rows["U.weight"]["relative_l2"]
    assert result["relative_l2_reference_floor"] == module.RELATIVE_L2_REFERENCE_FLOOR
    assert result["passed"] is True  # 분모 때문에 떨어지지 않는다

    # 바닥 위의 tensor는 여전히 상대 기준으로 떨어진다
    over = copy.deepcopy(continuous)
    broken = copy.deepcopy(continuous)
    broken["parameters"]["U.weight"] = broken["parameters"]["U.weight"] * 1.5
    judged = module.compare_resume(over, broken, steps=20)
    assert judged["passed"] is False and judged["worst_param_relative_l2"] > module.RESUME_TOLERANCE["param_rel_l2"]


def test_the_tolerance_is_derived_from_the_two_run_spread_by_the_rule_fixed_beforehand():
    """D — 사전 등록의 산술. 규칙은 :data:`RESUME_TOLERANCE_RULE`이고 **값을 보기 전에** 파일에 있었다.

    본 값에 맞춰 오차를 고치는 것은 이 프로젝트가 금하는 수다. 그래서 규칙이 함수이고, 그 함수가 시험에 묶여 있다."""
    module = script()
    assert module.TOLERANCE_SAFETY_FACTOR == 2.0 and "round up to one significant figure" in module.RESUME_TOLERANCE_RULE
    assert module._round_up_one_significant_figure(0.0876) == pytest.approx(0.09)
    assert module._round_up_one_significant_figure(0.0006) == pytest.approx(0.0006)
    assert module._round_up_one_significant_figure(0.0) == 0.0

    spread = {"worst_loss_abs": 0.0876, "worst_loss_rel": 0.0459,
              "worst_param_max_abs": 5.7e-4, "worst_param_relative_l2": 9.34e-4}
    derived = module.derive_tolerance(spread)
    assert derived["loss_abs"] == pytest.approx(0.2)      # 2 × 0.0876 = 0.1752 → 0.2
    assert derived["loss_rel"] == pytest.approx(0.1)      # 2 × 0.0459 = 0.0918 → 0.1
    assert derived["param_max_abs"] == pytest.approx(module.RESUME_TOLERANCE["param_max_abs"])  # T0보다 조이지 않는다
    assert derived["param_rel_l2"] == pytest.approx(module.RESUME_TOLERANCE["param_rel_l2"])
    # 퍼짐이 T0의 값을 넘으면 그만큼 느슨해진다
    wider = module.derive_tolerance({**spread, "worst_param_max_abs": 0.04, "worst_param_relative_l2": 0.3})
    assert wider["param_max_abs"] == pytest.approx(0.08) and wider["param_rel_l2"] == pytest.approx(0.6)


def test_the_two_run_spread_is_measured_without_a_verdict():
    """기준선은 **측정**이지 판정이 아니다 — `passed`를 돌려주면 사전 등록의 순서가 무너진다."""
    module = script()
    first, second = _snapshot_pair(steps=5)
    second["losses"][2] += 0.06
    second["parameters"]["U.weight"] = second["parameters"]["U.weight"] + 1e-4
    spread = module.measure_spread(first, second)
    assert "passed" not in spread and "verdict" not in spread
    assert spread["worst_loss_abs"] == pytest.approx(0.06) and spread["steps"] == 5
    assert spread["worst_param_max_abs"] == pytest.approx(1e-4, rel=1e-2)
    assert spread["sampler_position_equal"] and spread["drawn_units_equal"]
    assert spread["relative_l2_reference_floor"] == module.RELATIVE_L2_REFERENCE_FLOOR
    assert len(spread["parameters_worst"]) <= 8
    assert "no restart" in spread["unit"]
    import inspect

    assert "--check" in inspect.getsource(module.main) and "baseline" in module.CHECKS
