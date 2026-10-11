"""`robo_jev.decisions.train` — 판단 요청으로 Qwen 학습 (Task v0d §1·§2·§5).

CPU 검사는 소형 난수 Qwen3.5(``QwenBackbone.tiny``, torch 참조 kernel)로 학습기를 본다: 손실이 **답 자리에만** 걸리는지(hidden의
gradient가 그 자리 밖에서 0), 손실이 진단 도구의 읽기(:func:`robo_jev.decisions.zero_shot.score_code_prompts` 의 코드 확률)와 같은
KL인지, 오른쪽 padding이 손실을 바꾸지 않는지, 길이별 묶음 + gradient 누적이 한 번에 묶은 것과 같은 갱신인지, LoRA와 전체 미세조정의
학습 대상 파라미터 수, BF16 전체 미세조정이 fp32 master(MasterWeightAdamW)로 도는지, warmup + cosine 일정, 체크포인트의 표준 키·
metrics.json의 완주 표시, **재개 후 결정성**(끊었다 이은 run = 한 번에 돈 run, 비트 단위), 다른 설정·데이터의 재개 거절, dev 손실의
보고, dev로만 하는 고르기·멈추기, 디스크 여유 문턱, 평가 적재(:func:`apply_checkpoint`), 비어 있는 일반 데이터 섞기 자리.

Task v0d fix round 1: 프롬프트 판 v0c-p3(기본), 보조 항 −λ·log(코드 질량)(진단의 code_mass와 같은 값, 답 자리에만), 누적 시험은
**gradient 자체**를 비교(리뷰 1 Minor 1), dev의 코드 질량·결정적인 바뀜·``>= 0.5`` 판정, 미리 정한 고르기(마지막 체크포인트), step 메모리
추정과 울타리 거절(리뷰 1 Minor 6), 남는 dev 레코드 수의 기록(Minor 7).

GPU 검사(`real_2b` — 실제 Qwen3.5-2B와 CUDA가 있을 때만, 전체 suite는 GPU 유닛 안에서)는 r = 32 LoRA의 학습 대상 수와 한 step을 본다.
"""

import copy
import functools
import json
from types import SimpleNamespace
import math
import statistics

import pytest
import torch
import yaml
from helpers import FIXTURES, REPO

from robo_jev.checkpoint import CHECKPOINT_FORMAT, REQUIRED_KEYS, load_checkpoint, model_only_state, save_checkpoint
from robo_jev.decisions import examples as E
from robo_jev.decisions import prompt as P
from robo_jev.decisions import train as T
from robo_jev.decisions import zero_shot as Z
from robo_jev.decisions.records import read_records
from robo_jev.model.backbone import describe_backbone
from robo_jev.model.tokenizer import available_tokenizer, load_tokenizer
from robo_jev.train import MasterWeightAdamW, lr_factor

pytest.importorskip("transformers")
from robo_jev.model.backbone_qwen import QwenBackbone, torch_reference_kernels  # noqa: E402

REQUESTS = FIXTURES / "decisions" / "requests.jsonl"
REAL_2B = "Qwen/Qwen3.5-2B"


def real_weights_present() -> bool:
    try:
        describe_backbone(REAL_2B)
    except (FileNotFoundError, ValueError):
        return False
    return torch.cuda.is_available()


def needs_real_2b(func):
    marked = pytest.mark.real_2b(func)
    return pytest.mark.skipif(not real_weights_present(), reason="Qwen3.5-2B 가중치(artifacts/models)와 CUDA가 있어야 한다")(marked)


@pytest.fixture(autouse=True)
def _torch_kernels(request):
    if request.node.get_closest_marker("real_2b") is not None:
        yield
        return
    with torch_reference_kernels():
        yield


@functools.lru_cache(maxsize=1)
def tokenizer():
    return load_tokenizer(available_tokenizer()[0])


@pytest.fixture(scope="module")
def fixtures() -> list[dict]:
    return read_records(REQUESTS)


def tiny_config(tmp_path, **overrides) -> dict:
    """소형 모델·CPU·작은 묶음의 학습 설정 (fixture 8개 → dev 0개가 되게 dev_fraction 0)."""
    config = {
        "model_id": "tiny", "device": "cpu", "dtype": "float32", "out_dir": str(tmp_path / "runs"), "run_name": "t",
        "mode": "lora", "lora": {"r": 4, "alpha": 8, "dropout": 0.0, "targets": ["q_proj", "v_proj", "in_proj_qkv", "out_proj", "down_proj"]},
        "batch_examples": 4, "micro_batch_tokens": 4096, "lr": 1e-3, "warmup_ratio": 0.25, "epochs": 1, "dev_fraction": 0.0,
        "dev_every": 0, "checkpoint_every": 0, "seed": 3,
    }
    config.update(overrides)
    return config


def make_trainer(tmp_path, records, **overrides) -> "T.DecisionTrainer":
    return T.DecisionTrainer(tiny_config(tmp_path, **overrides), records=records, tokenizer=tokenizer(), log=None)


def lora_state(trainer) -> dict[str, torch.Tensor]:
    return {name: p.detach().clone() for name, p in trainer.backbone.text.named_parameters() if "lora_" in name}


# -- 설정 -------------------------------------------------------------------------------------------------


def test_the_config_fills_defaults_and_refuses_unknown_keys_and_bad_values(tmp_path):
    config = T.resolve_config({"train_input": "x.jsonl"})
    assert config["mode"] == "lora" and config["lora"]["r"] == 32 and config["questions_per_request"] == 2 and config["format"] == "chat"
    assert config["type_weights"] == {"choice": 1.0, "noul": 1.0, "score": 1.0} and config["dev_fraction"] == 0.02 and config["mix"] == []
    assert config["prompt_version"] == "v0c-p3" and config["code_mass_weight"] == 0.2 and config["selection"] == "final"
    assert set(config["lora"]["targets"]) >= {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}
    for bad in (
        {"nope": 1}, {"mode": "qlora"}, {"questions_per_request": 0}, {"type_weights": {"choice": 1.0}},
        {"type_weights": {"choice": -1.0, "noul": 1.0, "score": 1.0}}, {"format": "xml"}, {"batch_examples": 0},
        {"mix": [{"path": "general.jsonl", "share": 0.1}]}, {"dev_fraction": 0.7}, {"warmup_ratio": 1.5},
        {"prompt_version": "v0c-p9"}, {"code_mass_weight": -0.1}, {"selection": "best_dev"}, {"allow_large_activations": "yes"},
    ):
        with pytest.raises(ValueError):
            T.resolve_config({"train_input": "x.jsonl", **bad})


def test_the_h100_main_run_config_is_bf16_lora_with_the_code_mass_term_for_one_epoch():
    """사용자 결정(2026-10-11 21:16): 본 학습은 H100 80 GB 한 장에서도 돌 수 있게 — LoRA, bf16 adapter + fp32 master, λ = 0.2, 한 epoch, 미리 정한
    고르기, p3; micro-batch 상한은 Spark보다 크게(덮어쓸 수 있는 첫 값)."""
    resolved = T.resolve_config(yaml.safe_load((REPO / "configs" / "train" / "decisions-v0-lora-h100.yaml").read_text(encoding="utf-8")))
    assert resolved["mode"] == "lora" and resolved["lora_dtype"] == "bfloat16" and resolved["lora"] == T.DEFAULT_LORA
    assert resolved["code_mass_weight"] == 0.2 and resolved["epochs"] == 1 and resolved["max_steps"] is None and resolved["selection"] == "final"
    assert resolved["prompt_version"] == "v0c-p3" and resolved["micro_batch_tokens"] > 1024 and resolved["gradient_checkpointing"] is True
    assert resolved["batch_examples"] == 32 and resolved["lr"] == 2e-4 and resolved["train_input"] is None


@pytest.mark.parametrize("name, mode", [("decisions-v0-lora.yaml", "lora"), ("decisions-v0-full.yaml", "full")])
def test_the_shipped_configs_resolve(name, mode):
    config = yaml.safe_load((REPO / "configs" / "train" / name).read_text(encoding="utf-8"))
    resolved = T.resolve_config(config)
    assert resolved["mode"] == mode and resolved["model_id"] == "Qwen/Qwen3.5-2B" and resolved["format"] == "chat" and resolved["augment"]
    assert resolved["questions_per_request"] == 2 and resolved["dev_fraction"] == 0.02 and resolved["min_free_disk_gb"] == 60 and resolved["mix"] == []
    assert resolved["prompt_version"] == "v0c-p3" and resolved["code_mass_weight"] == 0.2 and resolved["selection"] == "final" and resolved["epochs"] == 1
    assert resolved["keep_best"] is False and resolved["allow_large_activations"] is False
    if mode == "lora":
        assert resolved["lora"] == T.DEFAULT_LORA and resolved["lr"] == 2e-4 and resolved["micro_batch_tokens"] == 1024
        assert resolved["lora_dtype"] == "bfloat16"  # 본 학습 방식 (fix round 1, 통제자 결정 2)
    else:
        assert resolved["lr"] == 1e-5 and resolved["micro_batch_tokens"] == 2048


# -- 손실: 답 자리의 코드 행에서만, 진단의 읽기와 같은 KL ---------------------------------------------------------


def test_the_training_loss_is_the_kl_of_the_diagnosis_reading(tmp_path, fixtures):
    trainer = make_trainer(tmp_path, fixtures)
    examples = [e for r in fixtures[:3] for e in E.build_examples(r, list(r["request"]["questions"]), tokenizer(), order_seed=5, fmt="chat")]
    with torch.no_grad():
        out = T.code_kl(trainer.backbone, examples)
    reading = Z.score_code_prompts(trainer.backbone, examples, batch=1, code_mass=False)
    for i, (example, read) in enumerate(zip(examples, reading)):
        expected = sum(t * (math.log(t) - math.log(p)) for t, p in zip(example["target"], read["code_probs"]) if t > 0)
        assert float(out["kl"][i]) == pytest.approx(expected, abs=2e-5)
        assert out["probs"][i] == pytest.approx(read["code_probs"], abs=2e-6)
        assert float(out["ce"][i]) >= float(out["kl"][i]) - 1e-9


def test_the_loss_reaches_only_the_answer_position(tmp_path, fixtures, monkeypatch):
    trainer = make_trainer(tmp_path, fixtures)
    examples = [e for r in fixtures[:2] for e in E.build_examples(r, list(r["request"]["questions"])[:2], tokenizer(), order_seed=1, fmt="chat")]
    captured = {}
    forward = trainer.backbone.forward

    def spy(tokens, positions, **kwargs):
        out = forward(tokens, positions, **kwargs)
        out["hidden"].retain_grad()
        captured["hidden"] = out["hidden"]
        return out

    monkeypatch.setattr(trainer.backbone, "forward", spy)
    out = T.code_kl(trainer.backbone, examples, code_mass=True)  # 주 항 + 보조 항 둘 다
    (out["kl"].sum() - 0.2 * out["log_code_mass"].sum()).backward()
    grad = captured["hidden"].grad
    for row, example in enumerate(examples):
        answer = len(example["tokens"]) - 1
        assert grad[row, answer].abs().sum() > 0
        others = torch.cat([grad[row, :answer], grad[row, answer + 1 :]])
        assert torch.count_nonzero(others) == 0


def test_right_padding_does_not_change_an_examples_loss(tmp_path, fixtures):
    trainer = make_trainer(tmp_path, fixtures)
    examples = [e for r in fixtures[:4] for e in E.build_examples(r, list(r["request"]["questions"])[:1], tokenizer(), order_seed=2, fmt="chat")]
    examples.sort(key=lambda e: len(e["tokens"]))
    with torch.no_grad():
        together = T.code_kl(trainer.backbone, examples)["kl"]
        alone = torch.stack([T.code_kl(trainer.backbone, [e])["kl"][0] for e in examples])
    assert torch.allclose(together, alone, atol=1e-5)


def test_type_weights_multiply_each_types_kl_and_the_code_mass_term_is_added_for_every_example(tmp_path, fixtures):
    trainer = make_trainer(tmp_path, fixtures, type_weights={"choice": 2.0, "noul": 1.0, "score": 0.0}, code_mass_weight=0.3)
    examples = trainer.step_examples(0)
    loss, parts = trainer.step_loss(examples, backward=False)
    by_type = {}
    for example, kl in zip(examples, parts["kl"]):
        by_type.setdefault(example["type"], []).append(kl)
    expected_kl = (2.0 * sum(by_type.get("choice", [])) + sum(by_type.get("noul", []))) / len(examples)
    expected_aux = 0.3 * sum(-math.log(m) for m in parts["code_mass"]) / len(examples)  # 보조 항은 종류 가중치 없이 모든 예에
    assert parts["kl_loss"] == pytest.approx(expected_kl, rel=1e-5) and parts["aux_loss"] == pytest.approx(expected_aux, rel=1e-5)
    assert loss == pytest.approx(expected_kl + expected_aux, rel=1e-5)
    off = make_trainer(tmp_path / "off", fixtures, type_weights={"choice": 2.0, "noul": 1.0, "score": 0.0}, code_mass_weight=0.0)
    loss_off, parts_off = off.step_loss(examples, backward=False)
    assert loss_off == pytest.approx(expected_kl, rel=1e-5) and parts_off["aux_loss"] == 0.0


def test_the_code_mass_is_the_diagnosis_code_mass_and_the_term_moves_mass_onto_the_shown_codes(tmp_path, fixtures):
    """보조 항의 코드 질량 = 진단의 ``code_mass``(전체 어휘 softmax가 보여 준 코드에 준 몫, fp32 head) — 그리고 그 항이 있으면 몇 step 뒤 그 몫이
    오르고(λ = 1) 없으면 거의 그대로다(λ = 0)."""
    trainer = make_trainer(tmp_path, fixtures)
    examples = [e for r in fixtures[:2] for e in E.build_examples(r, list(r["request"]["questions"])[:3], tokenizer(), order_seed=4, fmt="chat")]
    with torch.no_grad():
        out = T.code_kl(trainer.backbone, examples, code_mass=True)
    reading = Z.score_code_prompts(trainer.backbone, examples, batch=1, code_mass=True)
    for mass, read in zip(out["code_mass"], reading):
        assert mass == pytest.approx(read["code_mass"], rel=1e-4)
    masses, before = {}, None
    for lam in (0.0, 1.0):  # 소형 난수 모델의 어휘는 24만 — 코드 질량을 빨리 움직이려고 전체 미세조정으로 (LoRA r 4는 몇 step에 거의 못 움직인다)
        run = make_trainer(tmp_path / f"lam{lam}", fixtures, mode="full", code_mass_weight=lam, lr=5e-3, warmup_ratio=0.0)
        batch = run.step_examples(0)
        if before is None:
            before = statistics.fmean(run.step_loss(batch, backward=False)[1]["code_mass"])
        for _ in range(6):
            run.optimizer.zero_grad(set_to_none=True)
            run.step_loss(batch, backward=True)
            run.optimizer.step()
        masses[lam] = statistics.fmean(run.step_loss(batch, backward=False)[1]["code_mass"])
    assert masses[1.0] > 5 * before and masses[1.0] > 5 * masses[0.0], (before, masses)


# -- 묶음과 누적 --------------------------------------------------------------------------------------------


def test_micro_batches_are_length_sorted_cover_the_step_once_and_respect_the_token_budget():
    lengths = [300, 120, 450, 451, 90, 600, 610, 200, 1200, 130]
    groups = T.pack_micro_batches(lengths, 1000)
    flat = [i for g in groups for i in g]
    assert sorted(flat) == list(range(len(lengths)))
    assert [lengths[i] for i in flat] == sorted(lengths)
    for group in groups:
        assert len(group) == 1 or max(lengths[i] for i in group) * len(group) <= 1000
    assert T.pack_micro_batches(lengths, 100000) == [sorted(range(len(lengths)), key=lambda i: (lengths[i], i))]


def test_gradient_accumulation_over_micro_batches_equals_one_batch(tmp_path, fixtures):
    """리뷰 1 Minor 1: optimizer step **전에 gradient 자체를** 비교한다 — AdamW의 첫 갱신은 거의 lr·sign(g)라 파라미터 비교는 크기만 틀린
    누적(예: micro-batch마다 자기 크기로 나누기)을 놓친다. 종류 가중치를 고르지 않게 하고 보조 항을 켠 채, 예 하나씩 쌓은 gradient = 한 번에
    묶은 gradient (모든 학습 대상 tensor, 가장 큰 gradient 대비 1e-6 안)."""
    kw = {"type_weights": {"choice": 2.0, "noul": 1.0, "score": 0.5}, "code_mass_weight": 0.3}
    one = make_trainer(tmp_path / "a", fixtures, micro_batch_tokens=100000, **kw)
    many = make_trainer(tmp_path / "b", fixtures, micro_batch_tokens=1, **kw)  # 예 하나씩
    examples = one.step_examples(0)
    assert [e["tokens"] for e in examples] == [e["tokens"] for e in many.step_examples(0)]
    loss_one, parts_one = one.step_loss(examples, backward=True)
    loss_many, parts_many = many.step_loss(examples, backward=True)
    assert parts_one["micro_batches"] == 1 and parts_many["micro_batches"] == len(examples) > 1
    assert loss_one == pytest.approx(loss_many, rel=1e-5)
    grads_one, grads_many = dict(one.trainable), dict(many.trainable)
    assert set(grads_one) == set(grads_many)
    largest = max(float(p.grad.abs().max()) for p in grads_one.values() if p.grad is not None)
    compared = 0
    for name, param in grads_one.items():
        other = grads_many[name]
        assert (param.grad is None) == (other.grad is None), name
        if param.grad is None:
            continue
        assert torch.allclose(param.grad, other.grad, rtol=1e-4, atol=1e-6 * largest), name
        compared += int(torch.count_nonzero(param.grad) > 0)
    assert compared > 0 and largest > 0
    norm_one = torch.sqrt(sum((p.grad.double() ** 2).sum() for p in grads_one.values() if p.grad is not None))
    norm_many = torch.sqrt(sum((p.grad.double() ** 2).sum() for p in grads_many.values() if p.grad is not None))
    assert float(norm_one) == pytest.approx(float(norm_many), rel=1e-5)


# -- 학습 방식: LoRA·전체 미세조정 -------------------------------------------------------------------------------


def test_lora_trains_only_the_adapter_with_the_expected_parameter_count(tmp_path, fixtures):
    trainer = make_trainer(tmp_path, fixtures)
    targets = set(tiny_config(tmp_path)["lora"]["targets"])
    expected = 0
    base = QwenBackbone.tiny(seed=4, vocab_size=tokenizer().get_vocab_size())
    for name, module in base.text.named_modules():
        if name.split(".")[-1] in targets and isinstance(module, torch.nn.Linear):
            expected += 4 * (module.in_features + module.out_features)
    trainable = [(n, p) for n, p in trainer.backbone.model.named_parameters() if p.requires_grad]
    assert trainable and all("lora_" in n for n, _ in trainable) and all(p.dtype == torch.float32 for _, p in trainable)
    assert sum(p.numel() for _, p in trainable) == expected == trainer.manifest["model"]["trainable_parameters"]
    assert not trainer.backbone.model.lm_head.weight.requires_grad
    assert type(trainer.optimizer) is torch.optim.AdamW


def test_full_fine_tuning_in_bf16_trains_every_parameter_through_fp32_masters(tmp_path, fixtures):
    trainer = make_trainer(tmp_path, fixtures, mode="full", dtype="bfloat16", lr=1e-3)
    unique = {id(p): p for p in trainer.backbone.model.parameters()}
    assert all(p.requires_grad for p in unique.values())
    assert trainer.manifest["model"]["trainable_parameters"] == sum(p.numel() for p in unique.values())
    assert isinstance(trainer.optimizer, MasterWeightAdamW)
    pairs = trainer.optimizer.master_pairs
    assert len(pairs) == len(unique) and all(m.dtype == torch.float32 and p.dtype == torch.bfloat16 for _, p, m in pairs)
    before = {name: p.detach().clone() for name, p, _ in pairs}
    trainer.train_step()
    changed = sum(not torch.equal(before[name], p) for name, p, _ in pairs)
    assert changed > len(pairs) // 2
    for _, p, m in pairs:  # bf16 파라미터 = master의 반올림
        assert torch.equal(p, m.to(torch.bfloat16))


def test_a_bf16_adapter_computes_in_bf16_and_updates_through_fp32_masters(tmp_path, fixtures):
    trainer = make_trainer(tmp_path, fixtures, lora_dtype="bfloat16", lr=5e-3)
    lora = [(n, p) for n, p in trainer.backbone.text.named_parameters() if "lora_" in n]
    assert lora and all(p.dtype == torch.bfloat16 and p.requires_grad for _, p in lora)
    assert isinstance(trainer.optimizer, MasterWeightAdamW) and len(trainer.optimizer.master_pairs) == len(lora)
    assert trainer.manifest["model"]["lora_dtype"] == "bfloat16"
    path = trainer.run()["checkpoint"]
    for _, p, m in trainer.optimizer.master_pairs:
        assert m.dtype == torch.float32 and torch.equal(p, m.to(torch.bfloat16))
    assert any(torch.count_nonzero(p) > 0 for n, p in lora if "lora_B" in n)
    fresh = QwenBackbone.tiny(seed=4, vocab_size=tokenizer().get_vocab_size())
    T.apply_checkpoint(fresh, path, tokenizer_sha256=trainer.manifest["tokenizer"]["sha256"])
    loaded = {n: p for n, p in fresh.text.named_parameters() if "lora_" in n}
    assert all(loaded[n].dtype == torch.float32 and torch.equal(loaded[n], p.float()) for n, p in lora)  # 평가는 fp32 adapter (같은 값)


def test_the_learning_rate_follows_warmup_then_cosine(tmp_path, fixtures):
    trainer = make_trainer(tmp_path, fixtures, epochs=2, lr=1e-3, warmup_ratio=0.25)
    seen = []
    while trainer.step < trainer.max_steps:
        trainer.train_step()
        seen.append(trainer.history[-1]["lr"])
    expected = [1e-3 * lr_factor(s, max_steps=trainer.max_steps, warmup_ratio=0.25) for s in range(trainer.max_steps)]
    assert seen == pytest.approx(expected, rel=1e-9)


def test_one_cpu_step_runs_and_a_few_steps_lower_the_loss_on_the_seen_examples(tmp_path, fixtures):
    trainer = make_trainer(tmp_path, fixtures, lr=5e-3, warmup_ratio=0.0, epochs=1)
    examples = trainer.step_examples(0)
    before, _ = trainer.step_loss(examples, backward=False)
    metrics = trainer.train_step()
    assert math.isfinite(metrics["loss"]) and metrics["examples"] == len(examples) and metrics["tokens"] == sum(len(e["tokens"]) for e in examples)
    assert set(metrics["kl_by_type"]) <= {"choice", "noul", "score"} and metrics["grad_norm"] > 0
    for _ in range(5):
        trainer.optimizer.zero_grad(set_to_none=True)
        trainer.step_loss(examples, backward=True)
        trainer.optimizer.step()
    after, _ = trainer.step_loss(examples, backward=False)
    assert after < before


# -- 체크포인트·재개 ---------------------------------------------------------------------------------------------


def test_a_finished_run_writes_a_standard_checkpoint_and_marks_completion(tmp_path, fixtures):
    trainer = make_trainer(tmp_path, fixtures)
    result = trainer.run()
    assert result["status"] == "completed" and result["step"] == trainer.max_steps == trainer.config["max_steps"]
    state = load_checkpoint(result["checkpoint"])
    assert state["format"] == CHECKPOINT_FORMAT and all(key in state for key in REQUIRED_KEYS)
    assert state["manifest"]["kind"] == T.TRAIN_KIND and state["manifest"]["model"]["mode"] == "lora"
    assert set(state["model"]) == set(lora_state(trainer))
    metrics = json.loads((trainer.run_dir / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["status"] == "completed" and metrics["step"] == metrics["config"]["max_steps"] == trainer.max_steps
    assert len(metrics["steps"]) == trainer.max_steps and metrics["summary"]["last_loss"] == metrics["steps"][-1]["loss"]


@pytest.mark.parametrize("mode", ["lora", "lora-bf16", "full"])
def test_resuming_from_a_checkpoint_is_bit_identical_to_an_uninterrupted_run(tmp_path, fixtures, mode):
    extra = {"mode": mode.split("-")[0], "epochs": 2, "lr": 2e-3}
    if mode == "full":
        extra["dtype"] = "bfloat16"
    if mode == "lora-bf16":
        extra["lora_dtype"] = "bfloat16"
    straight = make_trainer(tmp_path / "straight", fixtures, **extra)
    straight.run()
    first = make_trainer(tmp_path / "split", fixtures, stop_after_steps=3, **extra)
    interrupted = first.run()
    assert interrupted["status"] == "interrupted" and interrupted["step"] == 3
    resumed = T.DecisionTrainer(tiny_config(tmp_path / "split", **extra), records=fixtures, tokenizer=tokenizer(), log=None, resume=interrupted["checkpoint"])
    assert resumed.step == 3
    finished = resumed.run()
    assert finished["status"] == "completed" and finished["step"] == straight.max_steps
    assert [h["loss"] for h in resumed.history] == [h["loss"] for h in straight.history]
    a = {n: p for n, p in straight.backbone.model.named_parameters() if p.requires_grad}
    b = {n: p for n, p in resumed.backbone.model.named_parameters() if p.requires_grad}
    assert set(a) == set(b) and all(torch.equal(a[n], b[n]) for n in a)


def test_resume_refuses_a_different_config_or_different_data(tmp_path, fixtures):
    first = make_trainer(tmp_path, fixtures, stop_after_steps=1, epochs=2)
    checkpoint = first.run()["checkpoint"]
    with pytest.raises(ValueError, match="lr"):
        T.DecisionTrainer(tiny_config(tmp_path, epochs=2, lr=5e-4), records=fixtures, tokenizer=tokenizer(), log=None, resume=checkpoint)
    other = copy.deepcopy(fixtures)
    name = next(n for n, q in other[0]["request"]["questions"].items() if q["type"] == "noul")
    other[0]["labels"][name]["p_yes"] = 0.123
    with pytest.raises(ValueError, match="data"):
        T.DecisionTrainer(tiny_config(tmp_path, epochs=2), records=other, tokenizer=tokenizer(), log=None, resume=checkpoint)
    # 중단 표지·저장 간격 같은 것은 달라도 된다
    T.DecisionTrainer(tiny_config(tmp_path, epochs=2, checkpoint_every=5), records=fixtures, tokenizer=tokenizer(), log=None, resume=checkpoint)


def test_a_checkpoint_is_not_written_when_the_disk_is_below_the_floor(tmp_path, fixtures, monkeypatch):
    trainer = make_trainer(tmp_path, fixtures, min_free_disk_gb=60)
    monkeypatch.setattr(T, "disk_free_bytes", lambda path: 59 * 10**9)
    with pytest.raises(T.DiskSpaceError):
        trainer.save()
    assert not (trainer.run_dir / "checkpoint.pt").exists()
    monkeypatch.setattr(T, "disk_free_bytes", lambda path: 61 * 10**9)
    trainer.save()
    assert (trainer.run_dir / "checkpoint.pt").exists()


# -- dev: 손실 보고, 고르기·멈추기 ------------------------------------------------------------------------------------


def test_dev_loss_is_reported_by_type_with_the_position_bias_on_counterbalanced_listings(tmp_path, fixtures):
    trainer = make_trainer(tmp_path, fixtures, dev_fraction=0.4, dev_every=1, epochs=1)
    assert trainer.dev_records and trainer.train_records and len(trainer.dev_records) + len(trainer.train_records) == len(fixtures)
    dev_ids = {r["id"] for r in trainer.dev_records}
    assert all(trainer.train_records[i]["id"] not in dev_ids for i, _, _ in E.epoch_plan(trainer.train_records, seed=3, epoch=0, q=2, augment=True))
    trainer.run()
    assert [d["step"] for d in trainer.dev_history] == list(range(0, trainer.max_steps + 1))
    first = trainer.dev_history[0]
    assert set(first["kl_by_type"]) <= {"choice", "noul", "score"} and first["questions"] == sum(len(r["request"]["questions"]) for r in trainer.dev_records)
    assert math.isfinite(first["loss"]) and "noul_yes_first_shift" in first and "noul_flip_rate" in first and "score_direction_shift" in first
    assert first["prompts"] == len(trainer.dev)
    for key in ("noul_decisive_flip_rate", "score_flip_rate", "score_decisive_flip_rate", "code_mass", "code_mass_below_half"):
        assert key in first
    assert set(first["code_mass_by_type"]) == set(first["kl_by_type"]) and 0.0 < first["code_mass"] <= 1.0
    assert "code_mass_mean" in trainer.history[0] and trainer.history[0]["loss"] == pytest.approx(trainer.history[0]["kl_loss"] + trainer.history[0]["aux_loss"])


def _dev_example(rid: str, name: str, kind: str, options: list[str], values: list[float] | None = None) -> dict:
    return {"record_id": rid, "question": name, "type": kind, "options": options, "values": values}


def test_the_dev_summary_reads_verdicts_with_at_least_one_half_and_counts_decisive_flips_apart():
    """통제자 결정 6 / 리뷰 1 Minor 2: dev의 판정도 ``>= 0.5``(진단과 같다); 평균 이동이 주 지표, 결정적인 바뀜은 두 값 모두 0.5에서 0.1 이상."""
    examples, probs = [], []
    for rid, yes_first, no_first in (("a", 0.5, 0.49), ("b", 0.3, 0.7), ("c", 0.55, 0.45)):
        examples += [_dev_example(rid, "q", "noul", ["yes", "no"]), _dev_example(rid, "q", "noul", ["no", "yes"])]
        probs += [[yes_first, 1 - yes_first], [1 - no_first, no_first]]
    examples += [_dev_example("d", "s", "score", ["0.0", "1.0"], [0.0, 1.0]), _dev_example("d", "s", "score", ["1.0", "0.0"], [1.0, 0.0])]
    probs += [[0.7, 0.3], [0.4, 0.6]]  # 오름 0.3, 내림 0.4 — 같은 쪽
    n = len(examples)
    out = T.summarize_dev(examples, kl=[0.1] * n, probs=probs, mass=[0.9] * (n - 1) + [0.4], type_weights={"choice": 1.0, "noul": 1.0, "score": 1.0})
    assert out["noul_flip_rate"] == pytest.approx(3 / 3)  # a: 0.5(예) vs 0.49(아니오) — >= 0.5라서 바뀜
    assert out["noul_decisive_flip_rate"] == pytest.approx(1 / 3)  # b만 (0.3 vs 0.7)
    assert out["noul_yes_first_shift"] == pytest.approx(((0.5 - 0.49) + (0.3 - 0.7) + (0.55 - 0.45)) / 3)
    assert out["score_direction_shift"] == pytest.approx(0.3 - 0.4) and out["score_flip_rate"] == 0.0
    assert out["code_mass_below_half"] == pytest.approx(1 / n) and out["code_mass_by_type"]["score"] == pytest.approx(0.65)


def test_keep_best_and_early_stopping_follow_the_dev_loss_only(tmp_path, fixtures, monkeypatch):
    trainer = make_trainer(tmp_path, fixtures, epochs=3, dev_every=1, keep_best=True, early_stop_patience=2, dev_fraction=0.4)
    scripted = iter([5.0, 4.0, 3.0, 3.5, 3.2, 3.1, 3.0, 2.0])
    monkeypatch.setattr(trainer, "evaluate_dev", lambda: {"loss": next(scripted)})
    result = trainer.run()
    assert result["status"] == "stopped_early" and trainer.step == 4  # step 2의 3.0 뒤로 두 번(3.5·3.2) 나아지지 않음
    assert {k: v for k, v in trainer.best.items() if k != "role"} == {"step": 2, "loss": 3.0, "checkpoint": str(trainer.run_dir / "checkpoint-best.pt")}
    assert "record only" in trainer.best["role"]  # 미리 정한 고르기: 평가하는 모델은 마지막 체크포인트
    best = load_checkpoint(trainer.run_dir / "checkpoint-best.pt")
    assert best["step"] == 2


# -- 평가 적재 ------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("mode", ["lora", "full"])
def test_apply_checkpoint_loads_the_trained_weights_into_a_fresh_backbone(tmp_path, fixtures, mode):
    trainer = make_trainer(tmp_path, fixtures, mode=mode, lr=5e-3)
    path = trainer.run()["checkpoint"]
    examples = trainer.step_examples(0)
    with torch.no_grad():
        trained = T.code_kl(trainer.backbone, examples)["probs"]
    fresh = QwenBackbone.tiny(seed=4, vocab_size=tokenizer().get_vocab_size())
    with torch.no_grad():
        base = T.code_kl(fresh, examples)["probs"]
    info = T.apply_checkpoint(fresh, path, tokenizer_sha256=trainer.manifest["tokenizer"]["sha256"])
    assert info["mode"] == mode and info["step"] == trainer.step and info["run_id"] == trainer.run_id
    assert not any(p.requires_grad for p in fresh.model.parameters())
    with torch.no_grad():
        loaded = T.code_kl(fresh, examples)["probs"]
    assert loaded == trained and loaded != base
    # model-only(슬림) 파일도 받는다
    slim = tmp_path / "slim.pt"
    state = load_checkpoint(path)
    torch.save(model_only_state(state, slimmed={"source_sha256": "x", "at": "now"}), slim)
    again = QwenBackbone.tiny(seed=4, vocab_size=tokenizer().get_vocab_size())
    T.apply_checkpoint(again, slim, tokenizer_sha256=trainer.manifest["tokenizer"]["sha256"])
    with torch.no_grad():
        assert T.code_kl(again, examples)["probs"] == trained


def test_apply_checkpoint_refuses_another_base_model_tokenizer_or_prompt(tmp_path, fixtures):
    trainer = make_trainer(tmp_path, fixtures)
    path = trainer.run()["checkpoint"]
    assert trainer.manifest["prompt"]["version"] == "v0c-p3"
    with pytest.raises(ValueError, match="prompt"):  # p3로 학습한 것을 p2 프롬프트로 평가하지 않는다
        T.apply_checkpoint(QwenBackbone.tiny(seed=4, vocab_size=tokenizer().get_vocab_size()), path, tokenizer_sha256=None, prompt_version="v0c-p2")
    with pytest.raises(ValueError, match="base"):
        T.apply_checkpoint(QwenBackbone.tiny(seed=5, vocab_size=tokenizer().get_vocab_size()), path, tokenizer_sha256=trainer.manifest["tokenizer"]["sha256"])
    with pytest.raises(ValueError, match="tokenizer"):
        T.apply_checkpoint(QwenBackbone.tiny(seed=4, vocab_size=tokenizer().get_vocab_size()), path, tokenizer_sha256="0" * 64)
    state = load_checkpoint(path)
    state["manifest"]["prompt"]["digest"] = "f" * 64
    changed = tmp_path / "changed.pt"
    save_checkpoint(changed, state)
    with pytest.raises(ValueError, match="prompt"):
        T.apply_checkpoint(QwenBackbone.tiny(seed=4, vocab_size=tokenizer().get_vocab_size()), changed, tokenizer_sha256=trainer.manifest["tokenizer"]["sha256"])


# -- 일반 능력 보호 갈고리 ---------------------------------------------------------------------------------------------


def test_the_general_data_mixing_hook_is_an_empty_slot(tmp_path, fixtures):
    trainer = make_trainer(tmp_path, fixtures)
    assert trainer.mix_plan(0) == []
    assert trainer.manifest["data"]["mix"] == []


def test_the_evaluated_model_is_pre_registered_as_the_final_checkpoint_and_unused_dev_records_are_counted(tmp_path, fixtures):
    trainer = make_trainer(tmp_path, fixtures, dev_fraction=0.4, dev_max_records=1)
    assert trainer.config["selection"] == "final" and trainer.manifest["selection"]["evaluated"].startswith("final checkpoint")
    total_dev = len(E.split_dev(fixtures, fraction=0.4, salt=trainer.config["dev_salt"])[1])
    assert len(trainer.dev_records) == 1 and trainer.manifest["data"]["dev_unused_records"] == total_dev - 1 > 0
    assert trainer.manifest["objective"]["code_mass_weight"] == 0.2 and trainer.manifest["prompt"]["digest"] == P.fixed_text_digest("v0c-p3")


def test_the_step_memory_estimate_refuses_a_setting_past_the_fence_unless_overridden():
    """리뷰 1 Minor 6: gradient checkpointing 없이 16 K 토큰 micro-batch(2B)는 activation만 ≈ 73 GB — 울타리(≈ 78 GB)의 90 %를 넘어 시작하지 않는다;
    checkpointing을 켜거나 micro-batch를 줄이면 통과한다. 추정 = 할당된 것 + optimizer 상태 + activation (+ 전체 미세조정 보조 항의 head 사본)."""
    two_b = SimpleNamespace(text=SimpleNamespace(config=SimpleNamespace(hidden_size=2048, num_hidden_layers=24)),
                            model=SimpleNamespace(lm_head=SimpleNamespace(weight=torch.empty(248320, 1, device="meta"))))
    adapter = [("lora", torch.empty(31_260_672, dtype=torch.float32, device="meta"))]
    fence = 0.6 * 130.6e9
    base = {"mode": "lora", "code_mass_weight": 0.2, "allow_large_activations": False}
    risky = T.estimate_step_memory(two_b, adapter, {**base, "gradient_checkpointing": False, "micro_batch_tokens": 16384}, allocated=int(5.9e9))
    assert risky["activations"] == pytest.approx(1e9 + 4.4e6 * 16384) and risky["total"] > 0.9 * fence
    with pytest.raises(ValueError, match="allow_large_activations"):
        T.check_step_memory(risky, limit_bytes=fence, config={**base, "gradient_checkpointing": False, "micro_batch_tokens": 16384})
    warned = []
    allowed = T.check_step_memory(risky, limit_bytes=fence, config={**base, "gradient_checkpointing": False, "micro_batch_tokens": 16384,
                                                                    "allow_large_activations": True}, log=warned.append)
    assert allowed["override"] is True and warned and "경고" in warned[0]
    for setting in ({"gradient_checkpointing": True, "micro_batch_tokens": 16384}, {"gradient_checkpointing": False, "micro_batch_tokens": 2048}):
        fine = T.estimate_step_memory(two_b, adapter, {**base, **setting}, allocated=int(5.9e9))
        assert T.check_step_memory(fine, limit_bytes=fence, config={**base, **setting})["total"] < 0.9 * fence
    full = [("all", torch.empty(1_881_825_088, dtype=torch.bfloat16, device="meta"))]
    estimate = T.estimate_step_memory(two_b, full, {**base, "mode": "full", "gradient_checkpointing": True, "micro_batch_tokens": 2048}, allocated=int(11.3e9))
    assert estimate["optimizer"] == 1_881_825_088 * (2 + 8 + 4) and estimate["aux_head"] == 2 * 248320 * 4  # bf16 grad + Adam 둘 + fp32 버퍼


# -- 실제 2B (GPU 유닛 안에서만) ------------------------------------------------------------------------------------------


@functools.lru_cache(maxsize=1)
def real_trainer_inputs():
    from robo_jev.gpu import limit_gpu_memory, require_free

    limit_gpu_memory(0.6)
    require_free(Z.MIN_FREE_BYTES[REAL_2B], what="decisions train real_2b test load")
    return QwenBackbone.load(REAL_2B)


@needs_real_2b
def test_real_2b_lora_r32_counts_its_parameters_and_takes_one_step(tmp_path, fixtures):
    backbone = real_trainer_inputs()
    config = {"model_id": REAL_2B, "out_dir": str(tmp_path / "runs"), "run_name": "real", "batch_examples": 4, "micro_batch_tokens": 4096,
              "dev_fraction": 0.0, "dev_every": 0, "checkpoint_every": 0, "epochs": 1, "lr": 1e-4}
    trainer = T.DecisionTrainer(config, records=fixtures[:4], tokenizer=tokenizer(), log=None, backbone=backbone)
    expected = sum(32 * (m.in_features + m.out_features) for name, m in backbone.text.named_modules()
                   if name.split(".")[-1] in T.DEFAULT_LORA["targets"] and hasattr(m, "base_layer"))
    assert trainer.manifest["model"]["trainable_parameters"] == expected > 0
    examples = trainer.step_examples(0)
    with torch.no_grad():
        fresh = T.code_kl(trainer.backbone, examples)["probs"]  # B = 0 → 기반 모델의 읽기
    reading = Z.score_code_prompts(trainer.backbone, examples, batch=1, code_mass=False)
    for probs, read in zip(fresh, reading):
        assert probs == pytest.approx(read["code_probs"], abs=2e-2)
    metrics = trainer.train_step()
    assert math.isfinite(metrics["loss"]) and metrics["grad_norm"] > 0
    lora_b = [p for n, p in trainer.backbone.text.named_parameters() if "lora_B" in n]
    assert any(torch.count_nonzero(p) > 0 for p in lora_b)
