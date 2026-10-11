"""`robo_jev.decisions.train` — 판단 요청으로 Qwen 학습 (Task v0d §1·§2·§5).

CPU 검사는 소형 난수 Qwen3.5(``QwenBackbone.tiny``, torch 참조 kernel)로 학습기를 본다: 손실이 **답 자리에만** 걸리는지(hidden의
gradient가 그 자리 밖에서 0), 손실이 진단 도구의 읽기(:func:`robo_jev.decisions.zero_shot.score_code_prompts` 의 코드 확률)와 같은
KL인지, 오른쪽 padding이 손실을 바꾸지 않는지, 길이별 묶음 + gradient 누적이 한 번에 묶은 것과 같은 갱신인지, LoRA와 전체 미세조정의
학습 대상 파라미터 수, BF16 전체 미세조정이 fp32 master(MasterWeightAdamW)로 도는지, warmup + cosine 일정, 체크포인트의 표준 키·
metrics.json의 완주 표시, **재개 후 결정성**(끊었다 이은 run = 한 번에 돈 run, 비트 단위), 다른 설정·데이터의 재개 거절, dev 손실의
보고, dev로만 하는 고르기·멈추기, 디스크 여유 문턱, 평가 적재(:func:`apply_checkpoint`), 비어 있는 일반 데이터 섞기 자리.

GPU 검사(`real_2b` — 실제 Qwen3.5-2B와 CUDA가 있을 때만, 전체 suite는 GPU 유닛 안에서)는 r = 32 LoRA의 학습 대상 수와 한 step을 본다.
"""

import copy
import functools
import json
import math

import pytest
import torch
import yaml
from helpers import FIXTURES, REPO

from robo_jev.checkpoint import CHECKPOINT_FORMAT, REQUIRED_KEYS, load_checkpoint, model_only_state, save_checkpoint
from robo_jev.decisions import examples as E
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
    assert set(config["lora"]["targets"]) >= {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}
    for bad in (
        {"nope": 1}, {"mode": "qlora"}, {"questions_per_request": 0}, {"type_weights": {"choice": 1.0}},
        {"type_weights": {"choice": -1.0, "noul": 1.0, "score": 1.0}}, {"format": "xml"}, {"batch_examples": 0},
        {"mix": [{"path": "general.jsonl", "share": 0.1}]}, {"dev_fraction": 0.7}, {"warmup_ratio": 1.5},
    ):
        with pytest.raises(ValueError):
            T.resolve_config({"train_input": "x.jsonl", **bad})


@pytest.mark.parametrize("name, mode", [("decisions-v0-lora.yaml", "lora"), ("decisions-v0-full.yaml", "full")])
def test_the_shipped_configs_resolve(name, mode):
    config = yaml.safe_load((REPO / "configs" / "train" / name).read_text(encoding="utf-8"))
    resolved = T.resolve_config(config)
    assert resolved["mode"] == mode and resolved["model_id"] == "Qwen/Qwen3.5-2B" and resolved["format"] == "chat" and resolved["augment"]
    assert resolved["questions_per_request"] == 2 and resolved["dev_fraction"] == 0.02 and resolved["min_free_disk_gb"] == 60 and resolved["mix"] == []
    if mode == "lora":
        assert resolved["lora"] == T.DEFAULT_LORA and resolved["lr"] == 2e-4
    else:
        assert resolved["lr"] == 1e-5


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
    out = T.code_kl(trainer.backbone, examples)
    out["kl"].sum().backward()
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


def test_type_weights_multiply_each_types_contribution(tmp_path, fixtures):
    trainer = make_trainer(tmp_path, fixtures, type_weights={"choice": 2.0, "noul": 1.0, "score": 0.0})
    examples = trainer.step_examples(0)
    loss, parts = trainer.step_loss(examples, backward=False)
    by_type = {}
    for example, kl in zip(examples, parts["kl"]):
        by_type.setdefault(example["type"], []).append(kl)
    expected = (2.0 * sum(by_type.get("choice", [])) + sum(by_type.get("noul", []))) / len(examples)
    assert loss == pytest.approx(expected, rel=1e-5)


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
    one = make_trainer(tmp_path / "a", fixtures, micro_batch_tokens=100000)
    many = make_trainer(tmp_path / "b", fixtures, micro_batch_tokens=1)  # 예 하나씩
    one.train_step()
    many.train_step()
    assert one.history[-1]["micro_batches"] == 1 and many.history[-1]["micro_batches"] == one.history[-1]["examples"]
    assert one.history[-1]["loss"] == pytest.approx(many.history[-1]["loss"], rel=1e-5)
    a, b = lora_state(one), lora_state(many)
    assert set(a) == set(b)
    for name in a:
        assert torch.allclose(a[name], b[name], atol=1e-6, rtol=1e-4), name


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


def test_keep_best_and_early_stopping_follow_the_dev_loss_only(tmp_path, fixtures, monkeypatch):
    trainer = make_trainer(tmp_path, fixtures, epochs=3, dev_every=1, keep_best=True, early_stop_patience=2, dev_fraction=0.4)
    scripted = iter([5.0, 4.0, 3.0, 3.5, 3.2, 3.1, 3.0, 2.0])
    monkeypatch.setattr(trainer, "evaluate_dev", lambda: {"loss": next(scripted)})
    result = trainer.run()
    assert result["status"] == "stopped_early" and trainer.step == 4  # step 2의 3.0 뒤로 두 번(3.5·3.2) 나아지지 않음
    assert trainer.best == {"step": 2, "loss": 3.0, "checkpoint": str(trainer.run_dir / "checkpoint-best.pt")}
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
