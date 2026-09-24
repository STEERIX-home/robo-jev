"""checkpoint 검사 — atomic 저장, 왕복, 중단 시 이전 checkpoint 보존, RNG·스트림 상태 복원 (docs/06 Task 5).

저장 단위(docs/03 §5)는 model/optimizer/scheduler/RNG/sampler 위치/config/manifest다. 저장은 같은
디렉터리의 임시 파일에 쓰고 fsync한 뒤 rename하므로, 임시 파일을 쓰는 도중이나 rename 직전에
죽어도 이전 checkpoint는 그대로 읽힌다.
"""

import os
import random

import numpy as np
import pytest
import torch
from helpers import SMALL_VOCAB

from robo_jev import checkpoint as checkpoint_module
from robo_jev.checkpoint import (
    CHECKPOINT_FORMAT,
    REQUIRED_KEYS,
    collect_rng_state,
    load_checkpoint,
    restore_rng_state,
    save_checkpoint,
    stream_state_from_dict,
    stream_state_to_dict,
)
from robo_jev.model.hybrid import TinyHybrid
from robo_jev.model.stream import StreamState


def tiny_state(step: int = 3) -> dict:
    """작은 모델·optimizer·scheduler를 한 step 돌린 뒤의 저장 단위."""
    torch.manual_seed(step)
    model = torch.nn.Linear(4, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: 0.5 ** s)
    model(torch.randn(3, 4)).sum().backward()
    optimizer.step()
    scheduler.step()
    return {
        "format": CHECKPOINT_FORMAT,
        "run_id": "unit-test",
        "step": step,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "rng": collect_rng_state(),
        "sampler": {"drawn": 7, "cursors": {"non_robot/existing": {"cursor": 3, "epoch": 0}}},
        "progress": None,
        "config": {"seed": 17, "max_steps": 20, "layout": {"single": "state_first"}},
        "manifest": {"dataset_manifest": {"path": "tests/fixtures/d0_manifest.json"}, "git_sha": None},
    }


def assert_same_tree(a, b, path="state"):
    if isinstance(a, torch.Tensor):
        assert isinstance(b, torch.Tensor), path
        assert a.dtype == b.dtype and torch.equal(a, b), path
    elif isinstance(a, dict):
        assert isinstance(b, dict) and list(a) == list(b), path
        for key in a:
            assert_same_tree(a[key], b[key], f"{path}.{key}")
    elif isinstance(a, (list, tuple)):
        assert type(a) is type(b) and len(a) == len(b), path
        for index, (x, y) in enumerate(zip(a, b)):
            assert_same_tree(x, y, f"{path}[{index}]")
    else:
        assert a == b and type(a) is type(b), path


# --------------------------------------------------------------------------
# 왕복
# --------------------------------------------------------------------------


def test_checkpoint_round_trips_every_part_of_the_saving_unit(tmp_path):
    state = tiny_state()
    path = tmp_path / "run" / "checkpoint.pt"
    save_checkpoint(path, state)
    loaded = load_checkpoint(path)
    assert set(REQUIRED_KEYS) <= set(loaded)
    assert_same_tree(state, loaded)
    # optimizer의 step 수와 scheduler의 위치까지 그대로다
    assert loaded["optimizer"]["state"][0]["step"].item() == 1
    assert loaded["scheduler"]["last_epoch"] == 1
    assert not list(tmp_path.glob("run/*.tmp*"))  # 임시 파일은 남지 않는다


def test_checkpoint_rejects_incomplete_units_and_foreign_files(tmp_path):
    state = tiny_state()
    del state["optimizer"]
    with pytest.raises(ValueError, match="optimizer"):
        save_checkpoint(tmp_path / "c.pt", state)
    assert not (tmp_path / "c.pt").exists()
    torch.save({"weights": torch.zeros(2)}, tmp_path / "foreign.pt")
    with pytest.raises(ValueError, match="format"):
        load_checkpoint(tmp_path / "foreign.pt")
    with pytest.raises(FileNotFoundError):
        load_checkpoint(tmp_path / "missing.pt")


# --------------------------------------------------------------------------
# 중단 — 임시 파일 쓰기 도중, rename 직전
# --------------------------------------------------------------------------


def test_crash_between_temp_write_and_rename_leaves_the_previous_checkpoint(tmp_path, monkeypatch):
    path = tmp_path / "checkpoint.pt"
    first = tiny_state(step=1)
    save_checkpoint(path, first)

    def crash(src, dst):
        raise OSError("simulated crash before rename")

    monkeypatch.setattr(checkpoint_module.os, "replace", crash)
    with pytest.raises(OSError, match="simulated crash"):
        save_checkpoint(path, tiny_state(step=2))
    monkeypatch.undo()
    loaded = load_checkpoint(path)
    assert loaded["step"] == 1
    assert_same_tree(first, loaded)
    assert not list(tmp_path.glob("*.tmp*"))  # 실패한 임시 파일은 치운다


def test_crash_while_writing_the_temp_file_leaves_the_previous_checkpoint(tmp_path, monkeypatch):
    path = tmp_path / "checkpoint.pt"
    first = tiny_state(step=1)
    save_checkpoint(path, first)
    real_save = checkpoint_module.torch.save

    def partial_save(obj, f, *args, **kwargs):
        f.write(b"partial bytes")
        raise RuntimeError("simulated crash while writing")

    monkeypatch.setattr(checkpoint_module.torch, "save", partial_save)
    with pytest.raises(RuntimeError, match="simulated crash"):
        save_checkpoint(path, tiny_state(step=2))
    monkeypatch.setattr(checkpoint_module.torch, "save", real_save)
    assert_same_tree(first, load_checkpoint(path))
    assert not list(tmp_path.glob("*.tmp*"))


def test_save_replaces_the_file_atomically_in_place(tmp_path):
    path = tmp_path / "checkpoint.pt"
    save_checkpoint(path, tiny_state(step=1))
    inode = os.stat(path).st_ino
    save_checkpoint(path, tiny_state(step=2))
    assert load_checkpoint(path)["step"] == 2
    assert os.stat(path).st_ino != inode  # 덮어쓰기가 아니라 rename이다


# --------------------------------------------------------------------------
# RNG (torch CPU + Python + numpy)
# --------------------------------------------------------------------------


def test_rng_state_round_trip_reproduces_the_next_draws(tmp_path):
    torch.manual_seed(3)
    random.seed(4)
    np.random.seed(5)
    snapshot = collect_rng_state()
    expected = (torch.rand(3), random.random(), np.random.rand(2))
    # 다른 곳으로 옮긴 뒤 복원한다 — 저장 파일을 거쳐서
    torch.rand(10), random.random(), np.random.rand(5)
    state = tiny_state()
    state["rng"] = snapshot
    save_checkpoint(tmp_path / "c.pt", state)
    restore_rng_state(load_checkpoint(tmp_path / "c.pt")["rng"])
    got = (torch.rand(3), random.random(), np.random.rand(2))
    assert torch.equal(expected[0], got[0]) and expected[1] == got[1]
    assert np.array_equal(expected[2], got[2])


def test_rng_state_carries_the_cuda_generator_only_when_this_process_has_one(monkeypatch):
    """CUDA를 **켠 적이 있는** 프로세스에서만 `cuda` 키가 생긴다 (Task R3a A1).

    `torch.cuda.get_rng_state_all()`은 CUDA를 초기화한다 — 그래서 켜지 않은 프로세스(CPU 학습·이 시험
    대부분)에서는 부르지 않는다. 켠 적이 없으면 뽑은 것도 없으므로 저장할 상태가 아예 없다.
    """
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert "cuda" not in collect_rng_state()

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: False)
    assert "cuda" not in collect_rng_state()

    drawn = [torch.arange(8, dtype=torch.uint8), torch.arange(8, 16, dtype=torch.uint8)]
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_rng_state_all", lambda: drawn)
    got = collect_rng_state()["cuda"]
    assert [t.tolist() for t in got] == [t.tolist() for t in drawn]
    assert all(t.dtype == torch.uint8 for t in got)
    assert got[0] is not drawn[0]  # 복사본이다 — 뒤이은 뽑기가 저장된 상태를 바꾸지 않는다


def test_restore_puts_the_cuda_generator_back_and_names_a_box_that_cannot_take_it(monkeypatch):
    """복원은 저장된 장치 수가 맞을 때만 한다 — 안 맞으면 조용히 넘기지 않고 이름으로 거절한다 (Task R3a A1)."""
    state = {**collect_rng_state(), "cuda": [torch.arange(8, dtype=torch.uint8)]}

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(ValueError, match="rng.cuda"):
        restore_rng_state(state)

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    with pytest.raises(ValueError, match="장치 수"):
        restore_rng_state(state)

    put: list = []
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(torch.cuda, "set_rng_state_all", put.extend)
    restore_rng_state(state)
    assert [t.tolist() for t in put] == [list(range(8))]
    assert all(t.dtype == torch.uint8 for t in put)


def test_restore_still_reads_a_checkpoint_written_before_the_cuda_generator_was_saved():
    """R2까지의 checkpoint에는 `cuda` 키가 없다 — 그것들을 계속 이어갈 수 있어야 한다 (Task R3a A1·C1)."""
    old = {key: value for key, value in collect_rng_state().items() if key != "cuda"}
    assert "cuda" not in old
    restore_rng_state(old)  # 거절하지 않는다


def test_cuda_rng_round_trip_reproduces_the_next_draws_on_this_box():
    """CUDA가 있으면 **실제 generator로** 왕복을 본다; 없으면 CPU 대체 경로의 구조를 본다 (Task R3a A1).

    건너뛰지 않는다 — 상자가 무엇이든 이 시험은 무언가를 확인한다.
    """
    if not torch.cuda.is_available():
        assert "cuda" not in collect_rng_state()
        return
    torch.cuda.init()  # manual_seed_all은 게으르다 — 이 시험이 혼자 돌아도 generator가 실제로 있게 한다 (리뷰 1 I3)
    torch.cuda.manual_seed_all(11)
    snapshot = collect_rng_state()
    assert "cuda" in snapshot
    expected = torch.rand(4, device="cuda").cpu()
    torch.rand(64, device="cuda")  # generator를 앞으로 민다
    restore_rng_state(snapshot)
    assert torch.equal(expected, torch.rand(4, device="cuda").cpu())


# --------------------------------------------------------------------------
# 스트림 상태 (진행 중이던 에피소드의 구간 위치·상태)
# --------------------------------------------------------------------------


def test_stream_state_survives_the_checkpoint_detached(tmp_path):
    backbone = TinyHybrid.from_config(seed=3, vocab_size=SMALL_VOCAB)
    state = StreamState.from_tokens([11, 12, 13], [[21, 22, 23], [31, 32]], backbone=backbone, window_ticks=5)
    assert state.delta[0]["recurrent"].requires_grad  # 그래프가 붙어 있는 상태
    packed = stream_state_to_dict(state)
    unit = tiny_state()
    unit["progress"] = {"carried_state": packed}
    save_checkpoint(tmp_path / "c.pt", unit)
    restored = stream_state_from_dict(load_checkpoint(tmp_path / "c.pt")["progress"]["carried_state"], backbone)
    assert restored.tick == state.tick == 1 and restored.position == state.position
    assert restored.window_ticks == state.window_ticks and not restored.is_branch
    for a, b in zip(restored.delta, state.delta):
        for key in ("recurrent", "conv"):
            assert torch.equal(a[key], b[key]) and not a[key].requires_grad
    for a, b in zip(restored.kv, state.kv):
        assert torch.equal(a["k"], b["k"]) and torch.equal(a["v"], b["v"])
    assert torch.equal(restored.cache_ticks, state.cache_ticks)
    assert torch.equal(restored.prefix_hidden, state.prefix_hidden) and torch.equal(restored.hidden, state.hidden)
    # 복원한 상태로 다음 틱을 이어가면 끊기지 않은 계산과 같다
    torch.testing.assert_close(restored.advance([41, 42]).recurrent, state.advance([41, 42]).recurrent)
    with pytest.raises(ValueError, match="branch"):
        stream_state_to_dict(state.fork(1)[0])
