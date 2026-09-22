"""기준 mask 검사 — full-attention 층의 query×key 허용 행렬 (docs/03 §3, docs/08 §3.1).

`state_first`: 다른 질문을 볼 수 없고, 같은 질문 안에서는 결정 위치가 모든 후보를 본다.
`stream_l1a`: 정적 prefix는 모두가 보고, 윈도우(prefix + 최근 30틱) 밖의 키는 가려지며, 결정
분기는 서로를 보지 못하고 다음 틱은 어떤 결정 위치도 보지 못한다.
"""

import copy
import random

import pytest
import torch

from robo_jev.model.attention import WINDOW_TICKS, build_reference_mask
from robo_jev.model.serialize import serialize_request
from robo_jev.model.tokenizer import WhitespaceTokenizer

# --------------------------------------------------------------------------
# 계획서의 최소 예시 (docs/06 Task 4, 그대로)
# --------------------------------------------------------------------------


def test_questions_cannot_read_each_other():
    layout = {
        "state": [0] * 9,
        "question": [-1, -1, 1, 1, 1, 1, 2, 2, 2],
        "candidate": [-1, -1, -1, 1, 2, -1, -1, 1, -1],
        "kind": ["state", "state", "question", "candidate", "candidate", "decision",
                 "question", "candidate", "decision"],
        "position": [0, 1, 2, 3, 4, 5, 2, 3, 4],
    }
    expected = torch.tensor([
        [1,0,0,0,0,0,0,0,0], [1,1,0,0,0,0,0,0,0],
        [1,1,1,0,0,0,0,0,0], [1,1,1,1,0,0,0,0,0], [1,1,1,1,1,0,0,0,0], [1,1,1,1,1,1,0,0,0],
        [1,1,0,0,0,0,1,0,0], [1,1,0,0,0,0,1,1,0], [1,1,0,0,0,0,1,1,1],
    ], dtype=torch.bool)
    assert torch.equal(build_reference_mask(layout), expected)


# --------------------------------------------------------------------------
# 무작위 layout 생성기 (직렬화 결과와 같은 모양)
# --------------------------------------------------------------------------


def random_state_first(rng: random.Random, *, states: int = 1) -> dict:
    layout = {"state": [], "question": [], "candidate": [], "kind": [], "position": [],
              "candidate_boundaries": [], "decision_positions": []}

    def push(state, question, candidate, kind, position):
        layout["state"].append(state)
        layout["question"].append(question)
        layout["candidate"].append(candidate)
        layout["kind"].append(kind)
        layout["position"].append(position)

    for state in range(states):
        state_len = rng.randint(1, 5)
        for position in range(state_len):
            push(state, -1, -1, "state", position)
        for branch in range(rng.randint(1, 4)):
            position = state_len
            for _ in range(rng.randint(1, 3)):
                push(state, branch, -1, "question", position)
                position += 1
            boundaries = []
            for candidate in range(rng.randint(1, 4)):
                for _ in range(rng.randint(1, 3)):
                    push(state, branch, candidate, "candidate", position)
                    position += 1
                boundaries.append(len(layout["kind"]) - 1)
            push(state, branch, -1, "decision", position)
            layout["candidate_boundaries"].append((state, branch, boundaries))
            layout["decision_positions"].append((state, branch, len(layout["kind"]) - 1))
    return layout


def random_stream(rng: random.Random, *, ticks: int | None = None, questions: int = 4) -> dict:
    layout = {"layout": "stream_l1a", "state": [], "question": [], "candidate": [], "kind": [],
              "position": [], "tick": [], "decision_positions": [], "static_candidates": [],
              "dynamic_candidates": []}

    def push(question, candidate, kind, tick, position):
        layout["state"].append(0)
        layout["question"].append(question)
        layout["candidate"].append(candidate)
        layout["kind"].append(kind)
        layout["tick"].append(tick)
        layout["position"].append(position)

    position = 0
    for _ in range(rng.randint(2, 6)):
        candidate = rng.choice([-1, 0, 1])
        push(-1, candidate, "prefix", -1, position)
        if candidate >= 0:
            layout["static_candidates"].append(len(layout["kind"]) - 1)
        position += 1
    for tick in range(ticks if ticks is not None else rng.randint(1, 40)):
        if tick and rng.random() < 0.15:  # 지시 변경: 그 틱의 첫 토큰들 (prefix가 아니다, docs/08 §3.1)
            for _ in range(rng.randint(1, 2)):
                push(-1, -1, "state", tick, position)
                position += 1
        for _ in range(rng.randint(2, 6)):
            kind = rng.choice(["state", "exec", "candidate"])
            candidate = rng.randint(0, 3) if kind == "candidate" else -1
            push(-1, candidate, kind, tick, position)
            if kind == "candidate":
                layout["dynamic_candidates"].append((tick, len(layout["kind"]) - 1))
            position += 1
        for branch in range(questions):
            if rng.random() < 0.8:
                push(branch, -1, "decision", tick, position)  # 분기는 같은 position
                layout["decision_positions"].append((tick, branch, len(layout["kind"]) - 1))
    return layout


SEEDS = list(range(12))


# --------------------------------------------------------------------------
# state_first 성질
# --------------------------------------------------------------------------


@pytest.mark.parametrize("seed", SEEDS)
def test_state_first_rows_are_exactly_state_plus_own_question(seed):
    layout = random_state_first(random.Random(seed), states=2)
    mask = build_reference_mask(layout)
    n = len(layout["kind"])
    assert mask.shape == (n, n) and mask.dtype == torch.bool
    for q in range(n):
        for k in range(n):
            expected = (
                k <= q
                and layout["state"][k] == layout["state"][q]
                and layout["question"][k] in (-1, layout["question"][q])
            )
            assert bool(mask[q, k]) == expected, (q, k)


@pytest.mark.parametrize("seed", SEEDS)
def test_state_first_decision_sees_every_candidate_of_its_own_question_only(seed):
    layout = random_state_first(random.Random(seed))
    mask = build_reference_mask(layout)
    boundaries = {branch: b for _, branch, b in layout["candidate_boundaries"]}
    for _, branch, decision in layout["decision_positions"]:
        assert all(bool(mask[decision, index]) for index in boundaries[branch])
        for other, indices in boundaries.items():
            if other != branch:
                assert not any(bool(mask[decision, index]) for index in indices)


def test_state_first_is_causal_and_packed_states_are_isolated():
    layout = random_state_first(random.Random(99), states=3)
    mask = build_reference_mask(layout)
    assert not torch.triu(mask, diagonal=1).any()
    assert bool(mask.diagonal().all())
    state = torch.tensor(layout["state"])
    assert not (mask & (state[:, None] != state[None, :])).any()


# --------------------------------------------------------------------------
# stream_l1a 성질
# --------------------------------------------------------------------------


def expected_stream(layout: dict, window: int) -> torch.Tensor:
    """규칙을 그대로 적은 느린 기준."""
    n = len(layout["kind"])
    kind, question, tick, state = layout["kind"], layout["question"], layout["tick"], layout["state"]
    out = torch.zeros(n, n, dtype=torch.bool)
    for q in range(n):
        for k in range(q + 1):
            if state[k] != state[q]:
                continue
            if kind[k] == "decision":
                out[q, k] = question[k] == question[q] and tick[k] == tick[q]
            elif kind[k] == "prefix":
                out[q, k] = True
            else:
                out[q, k] = tick[q] - tick[k] < window
    return out


@pytest.mark.parametrize("seed", SEEDS)
def test_stream_mask_matches_the_written_rule(seed):
    layout = random_stream(random.Random(seed))
    assert torch.equal(build_reference_mask(layout), expected_stream(layout, WINDOW_TICKS))


@pytest.mark.parametrize("seed", SEEDS)
def test_stream_causality_and_prefix_visibility(seed):
    layout = random_stream(random.Random(seed))
    mask = build_reference_mask(layout)
    n = len(layout["kind"])
    assert not torch.triu(mask, diagonal=1).any()
    assert bool(mask.diagonal().all())
    prefix = [index for index in range(n) if layout["kind"][index] == "prefix"]
    for q in range(n):
        for k in prefix:
            if k <= q:
                assert bool(mask[q, k])


@pytest.mark.parametrize("seed", SEEDS)
def test_stream_decision_branches_are_invisible_to_everyone_else(seed):
    layout = random_stream(random.Random(seed))
    mask = build_reference_mask(layout)
    n = len(layout["kind"])
    decisions = [index for index in range(n) if layout["kind"][index] == "decision"]
    for k in decisions:
        column = mask[:, k].clone()
        column[k] = False
        assert not column.any(), k  # 다른 결정 분기도, 다음 틱도 결정 위치를 보지 못한다


@pytest.mark.parametrize("seed", SEEDS)
def test_stream_decision_sees_prefix_window_and_all_candidates_of_its_tick(seed):
    layout = random_stream(random.Random(seed))
    mask = build_reference_mask(layout)
    for tick, _, decision in layout["decision_positions"]:
        for index in layout["static_candidates"]:
            assert bool(mask[decision, index])
        for candidate_tick, index in layout["dynamic_candidates"]:
            if candidate_tick == tick:
                assert bool(mask[decision, index])
            else:
                assert bool(mask[decision, index]) == (0 <= tick - candidate_tick < WINDOW_TICKS)


def test_stream_window_truncation_is_exact_over_many_ticks():
    layout = random_stream(random.Random(7), ticks=45)
    mask = build_reference_mask(layout)
    n = len(layout["kind"])
    kind, tick = layout["kind"], layout["tick"]
    seen_truncation = False
    for q in range(n):
        for k in range(q + 1):
            if kind[k] in ("prefix", "decision"):
                continue
            expected = tick[q] - tick[k] < WINDOW_TICKS
            assert bool(mask[q, k]) == expected, (q, k)
            seen_truncation |= not expected
    assert seen_truncation


def test_window_size_can_be_overridden_and_defaults_to_the_layout_value():
    layout = random_stream(random.Random(3), ticks=12)
    assert torch.equal(build_reference_mask(layout, window_ticks=4), expected_stream(layout, 4))
    layout["window_ticks"] = 2
    assert torch.equal(build_reference_mask(layout), expected_stream(layout, 2))
    with pytest.raises(ValueError, match="window_ticks"):
        build_reference_mask(layout, window_ticks=0)


def test_layout_lengths_must_agree():
    layout = random_state_first(random.Random(1))
    layout["question"] = layout["question"][:-1]
    with pytest.raises(ValueError, match="question"):
        build_reference_mask(layout)


# --------------------------------------------------------------------------
# 직렬화 결과를 그대로 넣는다
# --------------------------------------------------------------------------


def test_serialized_single_request_feeds_the_mask(singles):
    record = next(r for r in singles if len(r["request"]["questions"]) == 3)
    out = serialize_request(copy.deepcopy(record), WhitespaceTokenizer())
    mask = build_reference_mask(out)
    n = len(out["tokens"])
    assert mask.shape == (n, n)
    for branch, question_id in enumerate(out["question_ids"]):
        decision = out["decision_positions"][question_id]
        row = mask[decision]
        assert all(bool(row[index]) for index in out["candidate_boundaries"][question_id])
        assert all(bool(row[index]) for index in range(out["state_end"]))
        for other in out["question_ids"]:
            if other != question_id:
                assert not any(bool(row[index]) for index in out["candidate_boundaries"][other])
                assert not bool(row[out["decision_positions"][other]])
        # 결정 위치가 보는 것은 S와 자기 질문 전부다.
        assert int(row.sum()) == out["state_end"] + sum(1 for value in out["question"] if value == branch)


def test_serialized_stream_feeds_the_mask(streams):
    record = copy.deepcopy(streams[0])
    record["ticks"] = record["ticks"][:4]
    out = serialize_request(record, WhitespaceTokenizer(), layout="stream_l1a")
    mask = build_reference_mask(out)
    assert torch.equal(mask, expected_stream(out, WINDOW_TICKS))
    last = out["ticks"][-1]
    for question_id, decision in last["decision_positions"].items():
        row = mask[decision]
        assert all(bool(row[index]) for index in last["candidate_boundaries"][question_id])
        for other, position in last["decision_positions"].items():
            assert bool(row[position]) == (other == question_id)
        assert all(bool(row[index]) for index in range(out["prefix_end"]))
    # 다음 틱의 첫 토큰은 앞 틱의 결정 위치를 보지 못한다.
    first, second = out["ticks"][0], out["ticks"][1]
    assert not any(bool(mask[second["start"], index]) for index in first["decision_positions"].values())
    assert all(bool(mask[second["start"], index]) for index in range(first["start"], first["body_end"]))


def test_a_mid_stream_instruction_change_leaves_the_window_like_any_tick_token(streams):
    """docs/08 §3.1: 도중 추가되는 `[지시·제약 v2]`는 그 틱의 토큰이라 윈도우 밖으로 나간다.

    현재 지시는 매 틱 `goal` 필드가 다시 실으므로 잊히지 않는다. 처음 prefix는 고정 크기다.
    """
    record = copy.deepcopy(streams[0])
    second = record["prefix"]["instructions"][1]
    change = next(
        index
        for index, tick in enumerate(record["ticks"])
        if int(tick["request"]["state"]["goal"]["version"]) >= second["version"]
    )
    # 변경 한 틱 전부터 31틱 뒤까지, 상태는 goal만 남겨 mask를 작게 한다.
    record["ticks"] = record["ticks"][change - 1 : change + 32]
    for tick in record["ticks"]:
        tick["request"]["state"] = {"goal": tick["request"]["state"]["goal"]}
        for field in ("labels", "model_output", "adopted", "ack"):
            tick.pop(field, None)
    tokenizer = WhitespaceTokenizer()
    out = serialize_request(record, tokenizer, layout="stream_l1a")
    mask = build_reference_mask(out)

    piece = next(s for s in out["segments"] if s["name"] == f"instruction:{second['version']}")
    assert piece["kind"] != "prefix" and piece["tick"] == 1
    assert out["ticks"][1]["start"] == piece["start"]
    v2 = range(piece["start"], piece["end"])

    def decision_of(tick_index: int) -> int:
        return out["ticks"][tick_index]["decision_positions"]["q_main"]

    def state_text_of(tick_index: int) -> str:
        state = [s for s in out["segments"] if s["tick"] == tick_index and s["name"].startswith("state:")]
        return "".join(tokenizer.decode(out["tokens"][s["start"] : s["end"]]) for s in state)

    inside = decision_of(1 + WINDOW_TICKS - 1)  # 29틱 뒤: 아직 윈도우 안
    outside = decision_of(1 + WINDOW_TICKS + 1)  # 31틱 뒤: 윈도우 밖
    assert all(bool(mask[inside, index]) for index in v2)
    assert not any(bool(mask[outside, index]) for index in v2)
    assert all(bool(mask[outside, index]) for index in range(out["prefix_end"]))  # 정적 prefix는 남는다
    # 그 틱의 goal 줄은 여전히 v2를 싣는다 — 서식 v0.4에서는 **버전만** 남으므로(Task R1 A1) 지시 문장은
    # `goal_text_period_ticks`마다의 재적재와 지시 조각에서만 온다.
    assert any(line == "goal v2" for line in state_text_of(1 + WINDOW_TICKS + 1).splitlines())
