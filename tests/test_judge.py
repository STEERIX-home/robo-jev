"""Judge 검사 — pointer readout, 두 실행 backend, 참고군 R, 손실 연결, gradient 비교 (docs/03 §3, docs/06 Task 4).

`state_first`(P0, 질문별 독립 causal 경로를 한 배치로)와 `stream_l1a`(틱 몸통 한 번 + 일시적
결정 분기)가 같은 readout으로 `judgment_loss`의 layout을 낸다. P0에서는 질문 단독/묶음의
logits·loss·gradient가 허용 오차 안에서 같아야 하고(경로가 독립), 스트림에서는 결정 분기끼리
서로의 출력을 바꾸지 않되 질문 세트 자체를 바꾸면 공통 상태가 바뀐다(L1-a, 기록만 한다).
"""

import copy
import math
import random
import subprocess
import sys

import pytest
import torch
from helpers import SMALL_VOCAB

from robo_jev.contracts import QUESTION_SET_V0
from robo_jev.loss import judgment_loss
from robo_jev.model import serialize as serialize_module
from robo_jev.model.hybrid import TinyHybrid
from robo_jev.model.judge import Judge, candidate_span, typed_outputs
from robo_jev.model.serialize import serialize_request
from robo_jev.model.stream import forward_layout
from robo_jev.model.tokenizer import WhitespaceTokenizer

FP32 = {"rtol": 2e-5, "atol": 5e-5}
EXACT64 = {"rtol": 1e-10, "atol": 1e-12}

#: 공백 tokenizer는 id를 처음 본 순서로 주므로, 비교하는 직렬화들은 **같은 인스턴스**를 써야 한다.
TOKENIZER = WhitespaceTokenizer()


@pytest.fixture(scope="module")
def judge() -> Judge:
    return Judge.from_config(seed=5, vocab_size=SMALL_VOCAB)


@pytest.fixture
def three(singles) -> dict:
    """세 타입이 다 있고 세 라벨이 모두 유효한 D0 레코드."""
    record = next(r for r in singles if len(r["request"]["questions"]) == 3 and len(r["labels"]) == 3)
    return copy.deepcopy(record)


@pytest.fixture
def stream(streams) -> dict:
    record = copy.deepcopy(streams[0])
    record["ticks"] = record["ticks"][:2]
    return record


def stream_batch(record: dict, **kwargs) -> dict:
    out = serialize_request(record, TOKENIZER, layout="stream_l1a", **kwargs)
    return {"layout": "stream_l1a", "stream": out}


def state_batch(*records: dict) -> dict:
    return {"layout": "state_first", "states": [serialize_request(record, TOKENIZER) for record in records]}


def tick_labels(record: dict) -> dict:
    return {"labels": [tick.get("labels", []) for tick in record["ticks"]]}


def parameter_gradients(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {name: p.grad.clone() for name, p in module.named_parameters() if p.grad is not None}


def readout_parameters(judge: Judge) -> set[str]:
    return {name for name, _ in judge.named_parameters() if not name.startswith("backbone.")}


# --------------------------------------------------------------------------
# readout과 타입 변환
# --------------------------------------------------------------------------


def test_pointer_readout_is_the_documented_bilinear_form(judge):
    d, r = judge.backbone.config.d_model, judge.rank
    generator = torch.Generator().manual_seed(0)
    h_d = torch.randn(d, generator=generator)
    h_c = torch.randn(5, d, generator=generator)
    z = judge.pointer_logits(h_d, h_c)
    expected = (judge.U(h_d)[None, :] * judge.V(h_c)).sum(-1) / math.sqrt(r) + judge.bias
    torch.testing.assert_close(z, expected)
    assert z.shape == (5,) and r == 16


def test_typed_outputs_follow_the_contract():
    logits = {
        "q_a": torch.tensor([2.0, 0.0, -1.0]),
        "q_b": torch.tensor([0.0, 0.0]),
        "q_c": torch.tensor([0.0, math.log(3.0), 0.0]),
    }
    candidates = {"q_a": ["x", "y", "z"], "q_b": ["true", "false"], "q_c": ["0", "1", "2"]}
    questions = {
        "q_a": {"type": "choice", "criteria": [{"id": "x"}, {"id": "y"}, {"id": "z"}]},
        "q_b": {"type": "boolean", "criteria": []},
        "q_c": {"type": "ordinal", "criteria": [{"id": "0", "value": 0.0}, {"id": "1", "value": 2.0}, {"id": "2", "value": 5.0}]},
    }
    typed = typed_outputs(logits, candidates, questions)
    assert typed["q_a"]["choice"] == "x" and abs(sum(typed["q_a"]["probabilities"].values()) - 1) < 1e-6
    assert typed["q_b"]["p_true"] == pytest.approx(0.5) and typed["q_b"]["probabilities"]["false"] == pytest.approx(0.5)
    p = typed["q_c"]["probabilities"]
    assert p["1"] == pytest.approx(0.6) and typed["q_c"]["expected_value"] == pytest.approx(0.6 * 2 + 0.2 * 5)
    assert typed["q_c"]["choice"] == "1"
    with pytest.raises(ValueError, match="type"):
        typed_outputs({"q": torch.zeros(2)}, {"q": ["a", "b"]}, {"q": {"type": "regression", "criteria": []}})
    with pytest.raises(ValueError, match="candidates"):
        typed_outputs({"q": torch.zeros(2)}, {"q": ["a", "b"]}, {"q": {"type": "boolean", "criteria": []}})


def test_candidate_span_is_the_candidate_line_ending_at_its_boundary(three):
    out = serialize_request(three, TOKENIZER)
    newline = TOKENIZER.encode("\n").ids[0]
    for qid, boundaries in out["candidate_boundaries"].items():
        for index, boundary in enumerate(boundaries):
            start, end = candidate_span(out, boundary)
            assert end == boundary + 1 and start < boundary
            assert all(out["candidate"][i] == index and out["kind"][i] == "candidate" for i in range(start, end))
            assert out["candidate"][start - 1] != index or out["kind"][start - 1] != "candidate"
            assert out["tokens"][boundary] == newline  # 경계 = 후보 줄의 마지막 토큰(줄바꿈)


# --------------------------------------------------------------------------
# state_first (P0): 손실 layout, 단독/묶음 일치
# --------------------------------------------------------------------------


def test_state_first_outputs_feed_the_loss(judge, three):
    batch = state_batch(three)
    outputs = judge(batch)
    layout = batch["states"][0]
    assert outputs["candidates"] == [layout["candidate_mapping"]]
    assert outputs["question_ids"] == [layout["question_ids"]]
    for qid, ids in layout["candidate_mapping"].items():
        assert outputs["logits"][0][qid].shape == (len(ids),)
    loss = judgment_loss(outputs, {"labels": [three["labels"]]})
    assert loss.shape == () and torch.isfinite(loss) and loss.requires_grad
    loss.backward()
    assert judge.U.weight.grad is not None and judge.backbone.embed.weight.grad is not None


@pytest.mark.parametrize("readout", ["pointer", "candidate_branch"])
def test_every_parameter_of_the_selected_readout_and_backbone_gets_gradient(three, readout):
    """선택한 readout만 만든다 — 학습(Task 5)이 find_unused_parameters 없이 돌 수 있어야 한다."""
    judge = Judge.from_config(seed=5, readout=readout, vocab_size=SMALL_VOCAB)
    expected = {"U.weight", "V.weight", "bias"} if readout == "pointer" else {"w.weight", "w.bias"}
    assert readout_parameters(judge) == expected
    loss = judgment_loss(judge(state_batch(three)), {"labels": [three["labels"]]})
    loss.backward()
    missing = [name for name, p in judge.named_parameters() if p.grad is None]
    assert missing == []


def single_question_records(record: dict) -> list[dict]:
    singles = []
    for question in record["request"]["questions"]:
        one = copy.deepcopy(record)
        one["request"]["questions"] = [copy.deepcopy(question)]
        one["labels"] = [copy.deepcopy(l) for l in record["labels"] if l["question_id"] == question["id"]]
        one["usage"] = {"questions_used": [question["id"]], "commands": []}
        singles.append(one)
    return singles


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_state_first_single_vs_bundled_logits_loss_and_gradient_match(three, dtype):
    """P0의 경로는 독립이다: 질문 하나만 물어도 logits가 같고, 묶음 손실의 gradient = 단독 gradient의 평균.

    L0의 결정 표지는 모든 질문이 같은 고정 토큰이라(docs/03 §3) 단독 요청의 T_i 토큰이 묶음 안의
    것과 그대로 같다 — 표지를 고정해 줄 필요가 없다.
    """
    tolerance = FP32 if dtype == torch.float32 else EXACT64
    judge = Judge.from_config(seed=7, vocab_size=SMALL_VOCAB).to(dtype)
    batch = state_batch(three)
    bundled = judge(batch)
    loss_bundled = judgment_loss(bundled, {"labels": [three["labels"]]})
    loss_bundled.backward()
    grads_bundled = parameter_gradients(judge)

    singles = single_question_records(three)
    single_losses = []
    grads_single: dict[str, torch.Tensor] = {}
    for record in singles:
        judge.zero_grad(set_to_none=True)
        out = judge(state_batch(record))
        qid = record["request"]["questions"][0]["id"]
        torch.testing.assert_close(out["logits"][0][qid], bundled["logits"][0][qid], **tolerance)
        loss = judgment_loss(out, {"labels": [record["labels"]]})
        loss.backward()
        single_losses.append(loss.detach())
        for name, grad in parameter_gradients(judge).items():
            grads_single[name] = grads_single.get(name, 0) + grad / len(singles)
    torch.testing.assert_close(loss_bundled.detach(), torch.stack(single_losses).mean(), **tolerance)
    assert set(grads_bundled) == set(grads_single)
    for name in grads_bundled:
        torch.testing.assert_close(grads_bundled[name], grads_single[name], **tolerance)


def test_state_first_logits_do_not_depend_on_other_questions_or_their_order(judge, three):
    """docs/03 §3: 질문 추가·삭제·재배열이 기존 질문의 결과를 수치 오차 이상 바꾸지 않는다 — 구조로 성립한다."""
    reference = judge(state_batch(three))["logits"][0]
    reordered = copy.deepcopy(three)
    reordered["request"]["questions"] = list(reversed(reordered["request"]["questions"]))
    out = judge(state_batch(reordered))["logits"][0]
    for qid in reference:
        torch.testing.assert_close(out[qid], reference[qid], **FP32)
    two = copy.deepcopy(three)
    two["request"]["questions"] = two["request"]["questions"][1:]  # q_target 삭제
    two["labels"] = [l for l in two["labels"] if l["question_id"] != "q_target"]
    two["usage"]["questions_used"] = [q["id"] for q in two["request"]["questions"]]
    out = judge(state_batch(two))["logits"][0]
    assert set(out) == {"q_done", "q_speed"}
    for qid in out:
        torch.testing.assert_close(out[qid], reference[qid], **FP32)
    added = copy.deepcopy(three)  # 질문 추가: 첫 질문의 복제를 새 id로 앞에 끼워 넣는다
    extra = copy.deepcopy(added["request"]["questions"][0])
    extra["id"] = "q_extra"
    added["request"]["questions"].insert(0, extra)
    added["usage"]["questions_used"] = [q["id"] for q in added["request"]["questions"]]
    out = judge(state_batch(added))["logits"][0]
    for qid in reference:
        torch.testing.assert_close(out[qid], reference[qid], **FP32)


def test_state_first_batches_several_states(judge, singles):
    records = [copy.deepcopy(r) for r in singles[:3]]
    outputs = judge(state_batch(*records))
    assert len(outputs["logits"]) == 3
    loss = judgment_loss(outputs, {"labels": [r["labels"] for r in records]})
    assert torch.isfinite(loss)
    alone = judge(state_batch(records[1]))["logits"][0]
    for qid, logits in alone.items():
        torch.testing.assert_close(outputs["logits"][1][qid], logits, **FP32)


# --------------------------------------------------------------------------
# stream_l1a: 손실 layout, 증분 == 처음부터, 분기 격리
# --------------------------------------------------------------------------


def test_stream_outputs_feed_the_loss_and_reach_static_candidates(stream):
    judge = Judge.from_config(seed=9, vocab_size=SMALL_VOCAB)
    batch = stream_batch(stream)
    outputs = judge(batch)
    ticks = batch["stream"]["ticks"]
    assert len(outputs["logits"]) == len(ticks) == 2
    for tick, logits, candidates in zip(ticks, outputs["logits"], outputs["candidates"]):
        assert set(logits) == set(tick["decision_positions"]) == set(candidates)
        for qid, ids in candidates.items():
            assert logits[qid].shape == (len(ids),) and ids == tick["candidate_mapping"][qid]
    state = outputs["state"]
    assert state.tick == 1 and not state.is_branch
    state.prefix_hidden.retain_grad()
    loss = judgment_loss(outputs, tick_labels(stream))
    assert loss.shape == () and torch.isfinite(loss) and loss.item() > 0
    loss.backward()
    static = batch["stream"]["static_candidate_boundaries"]
    rows = state.prefix_hidden.grad
    assert rows is not None
    labelled = {l["question_id"] for tick in stream["ticks"] for l in tick["labels"] if l["kind"] == "single"}
    assert labelled & set(static)
    for qid, boundaries in static.items():
        if qid in labelled:
            assert (rows[boundaries].norm(dim=-1) > 1e-3).all(), qid  # 정적 후보의 h_c는 prefix hidden (실측 ≥ 0.061)
    assert judge.backbone.embed.weight.grad is not None


@pytest.mark.parametrize("window_ticks", [30, 1])
def test_stream_incremental_matches_from_scratch(stream, window_ticks):
    judge = Judge.from_config(seed=9, vocab_size=SMALL_VOCAB)
    batch = stream_batch(stream, window_ticks=window_ticks)
    incremental = judge(batch)
    scratch = judge({**batch, "from_scratch": True})
    for a, b in zip(incremental["logits"], scratch["logits"]):
        assert set(a) == set(b)
        for qid in a:
            torch.testing.assert_close(a[qid], b[qid], **FP32)
    if window_ticks == 1:  # 절단이 실제로 일어난다: 절단 없는 계산과 두 번째 틱이 다르다
        untruncated = judge({**batch, "from_scratch": True, "window_ticks": None})
        gap = max((incremental["logits"][1][q] - untruncated["logits"][1][q]).abs().max().item() for q in incremental["logits"][1])
        same = max((incremental["logits"][0][q] - untruncated["logits"][0][q]).abs().max().item() for q in incremental["logits"][0])
        print(f"window truncation at the logits (window=1, tick 1): max |Δz| {gap:.4g}; tick 0 (inside window) {same:.2g}")
        assert gap > 1e-3 and same < 5e-5


def test_stream_state_can_be_continued_tick_by_tick(stream):
    """틱을 하나씩 넣어도(이전 상태에서 이어감) 한 번에 넣은 것과 같다 — TBPTT 구간 이어 붙이기의 근거."""
    judge = Judge.from_config(seed=9, vocab_size=SMALL_VOCAB)
    whole = judge(stream_batch(stream))
    first = copy.deepcopy(stream)
    first["ticks"] = first["ticks"][:1]
    out1 = judge(stream_batch(first))
    out2 = judge({**stream_batch(stream), "state": out1["state"], "start_tick": 1})
    assert len(out2["logits"]) == 1
    for qid in whole["logits"][1]:
        torch.testing.assert_close(out2["logits"][0][qid], whole["logits"][1][qid], **FP32)
    for qid in whole["logits"][0]:
        torch.testing.assert_close(out1["logits"][0][qid], whole["logits"][0][qid], **FP32)


def reorder_decisions(layout: dict, rng: random.Random) -> dict:
    """틱 안의 결정 토큰 순서를 섞는다 (분기는 position을 공유하므로 layout 규칙은 그대로다)."""
    out = copy.deepcopy(layout)
    per_token = ("tokens", "kind", "state", "question", "candidate", "position", "tick", "segment")
    for tick in out["ticks"]:
        indices = list(range(tick["body_end"], tick["end"]))
        shuffled = indices[:]
        rng.shuffle(shuffled)
        rows = {field: [layout[field][i] for i in shuffled] for field in per_token}
        for offset, index in enumerate(indices):
            for field in per_token:
                out[field][index] = rows[field][offset]
        relocated = {}
        for qid, old in tick["decision_positions"].items():
            relocated[qid] = indices[shuffled.index(old)]
        tick["decision_positions"] = relocated
    return out


def keep_only_decisions(layout: dict, keep: set[str]) -> dict:
    """다른 질문의 결정 토큰을 layout에서 뺀다 (index를 다시 센다)."""
    drop = {index for tick in layout["ticks"] for qid, index in tick["decision_positions"].items() if qid not in keep}
    remap = {}
    for index in range(len(layout["tokens"])):
        if index not in drop:
            remap[index] = len(remap)
    out = copy.deepcopy(layout)
    for field in ("tokens", "kind", "state", "question", "candidate", "position", "tick", "segment"):
        out[field] = [value for index, value in enumerate(layout[field]) if index not in drop]
    for tick in out["ticks"]:
        body = tick["body_end"] - tick["start"]  # 몸통 토큰은 빠지지 않는다
        tick["start"] = remap[tick["start"]]
        tick["body_end"] = tick["start"] + body
        tick["decision_positions"] = {qid: remap[index] for qid, index in tick["decision_positions"].items() if qid in keep}
        tick["posed"] = [qid for qid in tick["posed"] if qid in keep]
        tick["candidate_boundaries"] = {qid: [remap[i] for i in b] for qid, b in tick["candidate_boundaries"].items() if qid in keep}
        tick["candidate_mapping"] = {qid: ids for qid, ids in tick["candidate_mapping"].items() if qid in keep}
        tick["end"] = max([tick["body_end"]] + [index + 1 for index in tick["decision_positions"].values()])
    out.pop("segments", None)
    return out


def test_stream_decision_branches_do_not_change_each_other(stream):
    judge = Judge.from_config(seed=9, vocab_size=SMALL_VOCAB)
    batch = stream_batch(stream)
    reference = judge(batch)["logits"]
    shuffled = judge({"layout": "stream_l1a", "stream": reorder_decisions(batch["stream"], random.Random(3))})["logits"]
    alone = judge({"layout": "stream_l1a", "stream": keep_only_decisions(batch["stream"], {"q_main"})})["logits"]
    pair = judge({"layout": "stream_l1a", "stream": keep_only_decisions(batch["stream"], {"q_stop", "q_speed"})})["logits"]
    for t, tick_reference in enumerate(reference):
        for qid, logits in tick_reference.items():
            torch.testing.assert_close(shuffled[t][qid], logits)
        torch.testing.assert_close(alone[t]["q_main"], tick_reference["q_main"])
        assert set(alone[t]) == {"q_main"}
        for qid in ("q_stop", "q_speed"):
            torch.testing.assert_close(pair[t][qid], tick_reference[qid])


# --------------------------------------------------------------------------
# 섭동 지표 (docs/08 §3.1: L1-a에서는 0이 아닐 수 있다 — 기록만 한다)
# --------------------------------------------------------------------------


def state_distance(a, b) -> dict[str, object]:
    """다음 틱 공통 상태의 거리. KV는 토큰 수가 같을 때만 뺄 수 있다(문구가 바뀌면 길이가 다르다)."""
    distance: dict[str, object] = {
        "recurrent": round(max((x - y).norm().item() for x, y in zip(a.recurrent, b.recurrent)), 4),
        "conv": round(max((x - y).norm().item() for x, y in zip(a.conv, b.conv)), 4),
    }
    if a.cached_tokens == b.cached_tokens:
        distance["kv"] = round(max((x["k"] - y["k"]).norm().item() for x, y in zip(a.kv, b.kv)), 4)
    else:
        distance["kv"] = f"n/a (cached {a.cached_tokens} vs {b.cached_tokens} tokens)"
    return distance


def test_perturbation_of_one_question_is_measured_not_asserted(stream, monkeypatch, capsys):
    judge = Judge.from_config(seed=9, vocab_size=SMALL_VOCAB)
    reference = judge(stream_batch(stream))
    report = []

    # (1) 다른 질문의 문구를 바꾼다 (prefix의 질문 세트 텍스트)
    altered = copy.deepcopy(QUESTION_SET_V0)
    altered["q_done"]["instructions"] = "목표가 이미 충족되었는지 판정하라."
    monkeypatch.setitem(serialize_module.QUESTION_SETS, "qs-v0", altered)
    text_changed = judge(stream_batch(stream))
    monkeypatch.setitem(serialize_module.QUESTION_SETS, "qs-v0", QUESTION_SET_V0)
    # (2) q_main의 후보 하나의 설명을 바꾼다 (틱 0의 동적 후보)
    candidates_changed_record = copy.deepcopy(stream)
    candidates_changed_record["ticks"][0]["request"]["candidates"]["q_main"][0]["desc"] = "완전히 다른 설명의 후보"
    candidates_changed = judge(stream_batch(candidates_changed_record))

    for name, changed, focus in (("text of q_done", text_changed, "q_done"), ("candidate of q_main", candidates_changed, "q_main")):
        others = {
            qid: (changed["logits"][0][qid] - reference["logits"][0][qid]).abs().max().item()
            for qid in reference["logits"][0] if qid != focus
        }
        own = (changed["logits"][0][focus] - reference["logits"][0][focus]).abs().max().item()
        next_common = state_distance(changed["tick_states"][0], reference["tick_states"][0])
        next_tick = {
            qid: (changed["logits"][1][qid] - reference["logits"][1][qid]).abs().max().item()
            for qid in reference["logits"][1]
        }
        report.append((name, own, others, next_common, next_tick))
        print(
            f"perturbation [{name}] own |Δz| {own:.3g}; other branches max |Δz| {max(others.values()):.3g} "
            f"(mean {sum(others.values()) / len(others):.3g}); next-tick common state Δ {next_common}; "
            f"next tick logits max |Δz| {max(next_tick.values()):.3g}"
        )
    assert all(math.isfinite(v) for _, own, others, _, _ in report for v in [own, *others.values()])


# --------------------------------------------------------------------------
# 스트림의 질문 단독/묶음 (L1-a: 세트를 바꾸면 공통 상태가 바뀐다 — 어느 층이 원인인지 기록)
# --------------------------------------------------------------------------


def test_stream_single_question_vs_bundled_difference_is_reported(stream, monkeypatch, capsys):
    judge = Judge.from_config(seed=9, vocab_size=SMALL_VOCAB)
    bundled = judge(stream_batch(stream))
    # 질문 세트를 q_main 하나로 줄인 스트림 (prefix에 다른 질문 텍스트·정적 후보가 없다)
    monkeypatch.setitem(serialize_module.QUESTION_SETS, "qs-v0", {"q_main": copy.deepcopy(QUESTION_SET_V0["q_main"])})
    solo_record = copy.deepcopy(stream)
    for tick in solo_record["ticks"]:
        tick["labels"] = [l for l in tick.get("labels", []) if l["question_id"] == "q_main"]
    solo_batch = stream_batch(solo_record)
    solo = judge(solo_batch)
    monkeypatch.setitem(serialize_module.QUESTION_SETS, "qs-v0", QUESTION_SET_V0)
    assert set(solo["logits"][0]) == {"q_main"}

    gap = [(solo["logits"][t]["q_main"] - bundled["logits"][t]["q_main"]).abs().max().item() for t in range(2)]
    # 어느 층이 다른가: 틱 0의 q_main 결정 토큰에서 층별 hidden 차이 (입력 embedding은 같은 토큰이라 0)
    full_layout = stream_batch(stream)["stream"]
    _, solo_layers = forward_layout(solo_batch["stream"], backbone=judge.backbone, return_layers=True)
    _, full_layers = forward_layout(full_layout, backbone=judge.backbone, return_layers=True)
    solo_index = solo_batch["stream"]["ticks"][0]["decision_positions"]["q_main"]
    full_index = full_layout["ticks"][0]["decision_positions"]["q_main"]
    layer_gap = [(a[solo_index] - b[full_index]).norm().item() for a, b in zip(solo_layers, full_layers)]
    print(
        f"stream single-question vs bundled (q_main): max |Δz| tick0 {gap[0]:.3g}, tick1 {gap[1]:.3g}; "
        f"per-layer hidden Δ at the decision token {['%.3g' % g for g in layer_gap]} "
        "(layer 0 = DeltaNet whose recurrent state read the other questions' prefix text; "
        "layer 2 = attention whose prefix KV differs)"
    )
    assert layer_gap[0] > 0 and all(math.isfinite(g) for g in layer_gap)
    # 손실도 계산된다 (단독 라벨만)
    assert torch.isfinite(judgment_loss(solo, tick_labels(solo_record)))


# --------------------------------------------------------------------------
# gradient: 손실 → 초기 recurrent 상태·conv history·공유 prefix (docs/06 Task 4 bullet 7)
# --------------------------------------------------------------------------


@pytest.mark.parametrize("seed", [13, 9])  # 9 = 망각 분석(보고 §4.5)과 같은 fixture
def test_stream_loss_gradient_reaches_initial_state_conv_history_and_prefix(stream, seed):
    """손실 → 초기 recurrent 상태·conv history·prefix. 하한은 실측의 1/10 아래로 잡아 게이트를 지나
    살아남는 크기를 고정한다(실측: seed 13 recurrent 2.69/0.77, conv 0.72/0.041; seed 9 recurrent
    1.59/0.32, conv 0.114/0.014; prefix_hidden 행 최소 0.053; prefix embedding 행 최소 0.032)."""
    judge = Judge.from_config(seed=seed, vocab_size=SMALL_VOCAB)
    initial = judge.backbone.initial_state(1, requires_grad=True)
    batch = {**stream_batch(stream), "initial": initial}
    outputs = judge(batch)
    outputs["state"].prefix_hidden.retain_grad()
    loss = judgment_loss(outputs, tick_labels(stream))
    loss.backward()
    for layer in initial:
        for key, floor in (("recurrent", 0.03), ("conv", 1e-3)):
            grad = layer[key].grad
            assert grad is not None and torch.isfinite(grad).all()
            assert grad.norm() > floor, (key, grad.norm().item())
    prefix_rows = outputs["state"].prefix_hidden.grad.norm(dim=-1)
    static = batch["stream"]["static_candidate_boundaries"]
    labelled = {l["question_id"] for tick in stream["ticks"] for l in tick["labels"] if l["kind"] == "single"}
    read = [index for qid, rows in static.items() if qid in labelled for index in rows]
    assert (prefix_rows[read] > 1e-3).all()  # 정적 후보 경계의 hidden을 readout이 읽는다
    prefix_tokens = batch["stream"]["tokens"][: batch["stream"]["prefix_end"]]
    assert (judge.backbone.embed.weight.grad[prefix_tokens].norm(dim=-1) > 1e-3).all()  # KV·recurrent를 거쳐 prefix 토큰까지


def test_state_first_loss_gradient_reaches_the_shared_state_tokens(three):
    judge = Judge.from_config(seed=13, vocab_size=SMALL_VOCAB)
    batch = state_batch(three)
    loss = judgment_loss(judge(batch), {"labels": [three["labels"]]})
    loss.backward()
    layout = batch["states"][0]
    state_tokens = sorted(set(layout["tokens"][: layout["state_end"]]))
    rows = judge.backbone.embed.weight.grad[state_tokens].norm(dim=-1)
    assert (rows > 0.05).all()  # 실측 최소 0.86 (25개 S 토큰)


# --------------------------------------------------------------------------
# 참고군 R: 후보별 분기 readout — 돌아가고 모양이 맞는 것만 확인 (docs/06 Task 4)
# --------------------------------------------------------------------------


def test_candidate_branch_reference_group_runs_with_the_documented_shape(three, stream):
    judge = Judge.from_config(seed=11, readout="candidate_branch", vocab_size=SMALL_VOCAB)
    assert judge.readout == "candidate_branch"
    batch = state_batch(three)
    outputs = judge(batch)
    layout = batch["states"][0]
    for qid, ids in layout["candidate_mapping"].items():
        assert outputs["logits"][0][qid].shape == (len(ids),)
    loss = judgment_loss(outputs, {"labels": [three["labels"]]})
    assert torch.isfinite(loss)
    loss.backward()
    # R은 scalar readout w만 가진다 — pointer의 U·V·b는 만들지 않는다(학습에서 쓰이지 않는 파라미터가 없다)
    assert readout_parameters(judge) == {"w.weight", "w.bias"}
    assert judge.w.weight.grad is not None and judge.w.bias.grad is not None
    with pytest.raises(ValueError, match="readout"):
        judge.pointer_logits(torch.zeros(64), torch.zeros(2, 64))

    one_tick = copy.deepcopy(stream)
    one_tick["ticks"] = one_tick["ticks"][:1]
    stream_out = judge(stream_batch(one_tick))
    tick = stream_batch(one_tick)["stream"]["ticks"][0]
    for qid, ids in tick["candidate_mapping"].items():
        assert stream_out["logits"][0][qid].shape == (len(ids),)
    assert torch.isfinite(judgment_loss(stream_out, tick_labels(one_tick)))


# --------------------------------------------------------------------------
# 입력 오류와 import 경계
# --------------------------------------------------------------------------


def test_judge_rejects_unknown_layouts_and_mismatched_batches(judge, three):
    with pytest.raises(ValueError, match="layout"):
        judge({"layout": "stream_l1b", "states": []})
    with pytest.raises(ValueError, match="states"):
        judge({"layout": "state_first"})
    layout = serialize_request(three, TOKENIZER)
    with pytest.raises(ValueError, match="layout"):
        judge({"layout": "stream_l1a", "stream": layout})
    with pytest.raises(ValueError, match="readout"):
        Judge(TinyHybrid.from_config(vocab_size=SMALL_VOCAB), rank=4, readout="lm_head")


def test_model_code_does_not_import_generator_simulator_or_harness():
    """docs/06 §1: model code가 generator·simulator·하네스를 import하지 않는다 (새 인터프리터에서 확인)."""
    script = (
        "import sys, robo_jev.model.hybrid, robo_jev.model.stream, robo_jev.model.judge, robo_jev.loss;"
        "print(sorted(n for n in sys.modules if n.startswith('robo_jev.')))"
    )
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, check=True)
    loaded = result.stdout.strip()
    assert "robo_jev.model.judge" in loaded
    for forbidden in ("robo_jev.sim", "robo_jev.harness", "robo_jev.data", "robo_jev.perception"):
        assert forbidden not in loaded, loaded
