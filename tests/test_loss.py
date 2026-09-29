"""손실 검사 — docs/03 §4, docs/08 §7.

작은 logits에 손으로 계산한 값을 대조한다. valid-set CE, 단일 정답 CE, 분포 CE, 사건 Bernoulli
NLL, 결측 mask, 상태별 정규화, `unknown` 후보를 정규화에서 빼는 부분 라벨 손실.
"""

import copy
import math

import pytest
import torch

from robo_jev.loss import judgment_loss, label_loss, question_losses
from robo_jev.model.serialize import serialize_request
from robo_jev.model.tokenizer import WhitespaceTokenizer


def softmax(values: list[float]) -> list[float]:
    total = sum(math.exp(v) for v in values)
    return [math.exp(v) / total for v in values]


def state(logits: dict[str, list[float]], candidates: dict[str, list[str]] | None = None) -> dict:
    """상태 하나의 출력. 후보 id를 주지 않으면 c0, c1, …이다."""
    tensors = {qid: torch.tensor(values, dtype=torch.float32, requires_grad=True) for qid, values in logits.items()}
    ids = candidates or {qid: [f"c{i}" for i in range(len(values))] for qid, values in logits.items()}
    return {"logits": tensors, "candidates": ids}


def batch(*states: dict) -> dict:
    return {"logits": [s["logits"] for s in states], "candidates": [s["candidates"] for s in states]}


def labelled(*per_state: list[dict]) -> dict:
    return {"labels": list(per_state)}


# --------------------------------------------------------------------------
# 라벨 종류별 손실
# --------------------------------------------------------------------------


def test_valid_set_is_minus_log_of_the_set_probability():
    z = [1.0, 2.0, 3.0]
    p = softmax(z)
    out = state({"q": z})
    label = {"question_id": "q", "kind": "valid_set", "candidate_ids": ["c1", "c2"]}
    loss = judgment_loss(batch(out), labelled([label]))
    assert loss.item() == pytest.approx(-math.log(p[1] + p[2]), rel=1e-5, abs=1e-6)


def test_single_answer_is_cross_entropy():
    z = [0.5, -1.0, 2.0]
    p = softmax(z)
    out = state({"q": z})
    loss = judgment_loss(batch(out), labelled([{"question_id": "q", "kind": "single", "answer": "c2"}]))
    assert loss.item() == pytest.approx(-math.log(p[2]), rel=1e-5, abs=1e-6)


def test_boolean_single_answer_maps_to_the_true_false_candidates():
    z = [0.3, -0.7]
    p = softmax(z)
    out = state({"q": z}, {"q": ["true", "false"]})
    yes = judgment_loss(batch(out), labelled([{"question_id": "q", "kind": "single", "answer": True}]))
    no = judgment_loss(batch(out), labelled([{"question_id": "q", "kind": "single", "answer": False}]))
    assert yes.item() == pytest.approx(-math.log(p[0]), rel=1e-5, abs=1e-6)
    assert no.item() == pytest.approx(-math.log(p[1]), rel=1e-5, abs=1e-6)


def test_distribution_is_soft_label_cross_entropy():
    z = [1.0, 0.0, -1.0]
    p = softmax(z)
    q = {"c0": 0.5, "c1": 0.3, "c2": 0.2}
    out = state({"q": z})
    loss = judgment_loss(batch(out), labelled([{"question_id": "q", "kind": "distribution", "probabilities": q}]))
    expected = -sum(q[f"c{i}"] * math.log(p[i]) for i in range(3))
    assert loss.item() == pytest.approx(expected, rel=1e-5, abs=1e-6)


def test_event_is_the_per_trial_bernoulli_nll_of_the_true_candidate():
    z = [0.8, -0.2]
    p_true = softmax(z)[0]
    out = state({"q": z}, {"q": ["true", "false"]})
    label = {"question_id": "q", "kind": "event", "event_id": "e0", "successes": 7, "failures": 1, "censored": 1}
    loss = judgment_loss(batch(out), labelled([label]))
    expected = -(7 * math.log(p_true) + 1 * math.log(1 - p_true)) / 8
    assert loss.item() == pytest.approx(expected, rel=1e-5, abs=1e-6)


def test_event_without_trials_has_no_grounds_and_is_masked():
    out = state({"q": [0.1, 0.2]}, {"q": ["true", "false"]})
    label = {"question_id": "q", "kind": "event", "event_id": "e0", "successes": 0, "failures": 0, "censored": 3}
    assert label_loss(out["logits"]["q"], out["candidates"]["q"], label) is None
    assert judgment_loss(batch(out), labelled([label])).item() == 0.0


# --------------------------------------------------------------------------
# 부분 라벨 손실 (docs/08 §7)
# --------------------------------------------------------------------------


def test_unknown_candidates_leave_the_normalisation():
    z = [1.0, 2.0, 3.0, 4.0]
    e = [math.exp(v) for v in z]
    out = state({"q": z})
    label = {"question_id": "q", "kind": "valid_set", "candidate_ids": ["c0"], "unknown": ["c3"]}
    loss = judgment_loss(batch(out), labelled([label]))
    # A = {c0}, I = {c1, c2}, U = {c3}: -log(p0 / (p0 + p1 + p2))
    assert loss.item() == pytest.approx(-math.log(e[0] / (e[0] + e[1] + e[2])), rel=1e-5, abs=1e-6)


def test_unknown_case_equals_the_renormalised_valid_set_loss():
    z = [0.3, -1.2, 2.2, 0.7, -0.4]
    out = state({"q": z})
    partial = {"question_id": "q", "kind": "valid_set", "candidate_ids": ["c0", "c2"], "unknown": ["c1", "c4"]}
    restricted = state({"q": [z[0], z[2], z[3]]}, {"q": ["c0", "c2", "c3"]})
    full = {"question_id": "q", "kind": "valid_set", "candidate_ids": ["c0", "c2"]}
    assert judgment_loss(batch(out), labelled([partial])).item() == pytest.approx(
        judgment_loss(batch(restricted), labelled([full])).item(), rel=1e-5, abs=1e-6
    )
    # unknown이 없으면 부분 손실은 보통의 valid-set 손실이다.
    no_unknown = {**partial, "unknown": []}
    assert judgment_loss(batch(out), labelled([no_unknown])).item() == pytest.approx(
        judgment_loss(batch(out), labelled([full])).item(), rel=1e-5, abs=1e-6
    )


def test_unknown_candidates_receive_no_gradient():
    out = state({"q": [1.0, 2.0, 3.0, 4.0]})
    label = {"question_id": "q", "kind": "valid_set", "candidate_ids": ["c0"], "unknown": ["c3"]}
    judgment_loss(batch(out), labelled([label])).backward()
    grad = out["logits"]["q"].grad
    assert grad[3].item() == 0.0
    assert grad[0].item() < 0 and grad[1].item() > 0 and grad[2].item() > 0


def test_hold_not_in_a_labels_leave_the_admissible_candidates_out_of_the_normalisation():
    """hold∉A 규칙 (docs/08 §7): A = {hold}, unknown = semantic_admissible, 나머지(부적합 후보)가 I. 손실은
    -log(p_hold / (p_hold + Σ_I p))이고 적합 후보의 logits는 gradient를 받지 않는다 — 답은 언제나 A 안이다."""
    candidates = ["c1", "c2", "c3", "ch", "co"]
    label = {"question_id": "q_main", "kind": "valid_set", "candidate_ids": ["ch"],
             "semantic_admissible": ["c1", "c2"], "unknown": ["c1", "c2"], "label_confidence": "low", "weight": 0.25}
    logits = torch.tensor([2.0, 1.0, 0.5, 0.0, -1.0], requires_grad=True)
    loss = label_loss(logits, candidates, label)
    p = torch.softmax(logits, dim=0)
    expected = -torch.log(p[3] / (p[3] + p[2] + p[4]))
    assert torch.allclose(loss, expected)
    loss.backward()
    assert torch.all(logits.grad[:2] == 0) and logits.grad[3] != 0 and logits.grad[2] != 0
    assert set(label["candidate_ids"]) <= set(candidates) and not set(label["candidate_ids"]) & set(label["unknown"])
    # 적합 후보가 없는 퇴화 틱(goal_candidate_missing): unknown 없음 → hold 밖의 전부가 I.
    plain = {**label, "semantic_admissible": [], "unknown": []}
    assert torch.allclose(label_loss(logits.detach(), candidates, plain), -torch.log_softmax(logits.detach(), dim=0)[3])
    # 가중치는 question_losses가 그대로 옮긴다.
    outputs = {"logits": [{"q_main": logits.detach()}], "candidates": [{"q_main": candidates}]}
    assert question_losses(outputs, {"labels": [[label]]})[0]["q_main"]["weight"] == 0.25


def test_event_results_on_a_valid_set_label_are_not_a_loss_term():
    """rollout 결과는 선택 분포와 합치지 않는다 (docs/03 §4)."""
    z = [1.0, 2.0, 3.0]
    out = state({"q": z})
    plain = {"question_id": "q", "kind": "valid_set", "candidate_ids": ["c1"]}
    with_results = {**plain, "event_results": {"c1": {"s": 8, "f": 0}, "c2": {"s": 1, "f": 7}}}
    assert judgment_loss(batch(out), labelled([plain])).item() == pytest.approx(
        judgment_loss(batch(out), labelled([with_results])).item()
    )


# --------------------------------------------------------------------------
# mask와 정규화 (docs/03 §4)
# --------------------------------------------------------------------------


def test_questions_without_labels_contribute_nothing_and_get_no_gradient():
    out = state({"a": [1.0, 2.0], "b": [0.0, 0.0, 0.0]})
    label = {"question_id": "a", "kind": "single", "answer": "c1"}
    loss = judgment_loss(batch(out), labelled([label]))
    assert loss.item() == pytest.approx(-math.log(softmax([1.0, 2.0])[1]), rel=1e-5, abs=1e-6)
    loss.backward()
    assert out["logits"]["b"].grad is None
    assert out["logits"]["a"].grad is not None and out["logits"]["a"].grad.abs().sum() > 0


def test_mask_false_labels_are_excluded():
    out = state({"a": [1.0, 2.0], "b": [0.0, 1.0]})
    labels = [
        {"question_id": "a", "kind": "single", "answer": "c1"},
        {"question_id": "b", "kind": "single", "answer": "c0", "mask": False},
    ]
    loss = judgment_loss(batch(out), labelled(labels))
    assert loss.item() == pytest.approx(-math.log(softmax([1.0, 2.0])[1]), rel=1e-5, abs=1e-6)
    loss.backward()
    assert out["logits"]["b"].grad is None


def test_loss_is_a_mean_over_states_not_over_questions():
    first = state({"a": [1.0, 2.0], "b": [0.0, 1.0, 2.0]})
    second = state({"a": [2.0, -1.0]})
    labels_first = [
        {"question_id": "a", "kind": "single", "answer": "c0"},
        {"question_id": "b", "kind": "valid_set", "candidate_ids": ["c2"]},
    ]
    labels_second = [{"question_id": "a", "kind": "single", "answer": "c1"}]
    l1 = -math.log(softmax([1.0, 2.0])[0])
    l2 = -math.log(softmax([0.0, 1.0, 2.0])[2])
    l3 = -math.log(softmax([2.0, -1.0])[1])
    loss = judgment_loss(batch(first, second), labelled(labels_first, labels_second))
    assert loss.item() == pytest.approx(((l1 + l2) / 2 + l3) / 2, rel=1e-5, abs=1e-6)
    assert loss.item() != pytest.approx((l1 + l2 + l3) / 3, rel=1e-5, abs=1e-6)


def test_states_without_any_valid_label_do_not_dilute_the_mean():
    first = state({"a": [1.0, 2.0]})
    empty = state({"a": [5.0, 5.0]})
    labels_first = [{"question_id": "a", "kind": "single", "answer": "c0"}]
    loss = judgment_loss(batch(first, empty), labelled(labels_first, []))
    assert loss.item() == pytest.approx(-math.log(softmax([1.0, 2.0])[0]), rel=1e-5, abs=1e-6)
    nothing = judgment_loss(batch(empty), labelled([]))
    assert nothing.item() == 0.0 and nothing.shape == ()
    # 상태별 microbatch(docs/03 §5)에서 한 상태의 라벨이 전부 mask면 여기로 온다 — backward가
    # 되어야 하고 gradient는 0이어야 한다.
    nothing.backward()
    assert nothing.grad_fn is not None
    assert torch.equal(empty["logits"]["a"].grad, torch.zeros(2))


def test_all_masked_labels_still_give_a_differentiable_zero():
    out = state({"a": [1.0, 2.0], "b": [0.3, -0.3]}, {"a": ["c0", "c1"], "b": ["true", "false"]})
    labels = [
        {"question_id": "a", "kind": "single", "answer": "c0", "mask": False},
        {"question_id": "b", "kind": "event", "event_id": "e0", "successes": 0, "failures": 0, "censored": 2},
    ]
    loss = judgment_loss(batch(out), labelled(labels))
    assert loss.item() == 0.0 and loss.requires_grad
    loss.backward()
    for logits in out["logits"].values():
        assert torch.equal(logits.grad, torch.zeros_like(logits))


def test_no_logits_at_all_gives_a_plain_zero():
    loss = judgment_loss({"logits": [{}], "candidates": [{}]}, labelled([]))
    assert loss.item() == 0.0 and loss.shape == () and not loss.requires_grad


def test_label_weights_scale_inside_the_state():
    out = state({"a": [1.0, 2.0], "b": [0.0, 1.0]})
    labels = [
        {"question_id": "a", "kind": "single", "answer": "c1", "weight": 3.0},
        {"question_id": "b", "kind": "single", "answer": "c1"},
    ]
    la = -math.log(softmax([1.0, 2.0])[1])
    lb = -math.log(softmax([0.0, 1.0])[1])
    loss = judgment_loss(batch(out), labelled(labels))
    assert loss.item() == pytest.approx((3 * la + lb) / 4, rel=1e-5, abs=1e-6)


def test_question_losses_report_each_labelled_question():
    out = state({"a": [1.0, 2.0], "b": [0.0, 1.0, 2.0]})
    labels = [
        {"question_id": "a", "kind": "single", "answer": "c0"},
        {"question_id": "b", "kind": "valid_set", "candidate_ids": ["c2"], "mask": False},
    ]
    per_state = question_losses(batch(out), labelled(labels))
    assert len(per_state) == 1 and list(per_state[0]) == ["a"]
    assert per_state[0]["a"]["loss"].item() == pytest.approx(-math.log(softmax([1.0, 2.0])[0]), rel=1e-5, abs=1e-6)
    assert per_state[0]["a"]["weight"] == 1.0


# --------------------------------------------------------------------------
# 계약 위반은 오류다
# --------------------------------------------------------------------------


def test_unknown_candidate_or_question_is_an_error():
    out = state({"a": [1.0, 2.0]})
    with pytest.raises(ValueError, match="candidate_ids"):
        judgment_loss(batch(out), labelled([{"question_id": "a", "kind": "valid_set", "candidate_ids": ["zz"]}]))
    with pytest.raises(ValueError, match="question_id"):
        judgment_loss(batch(out), labelled([{"question_id": "nope", "kind": "single", "answer": "c0"}]))
    with pytest.raises(ValueError, match="kind"):
        judgment_loss(batch(out), labelled([{"question_id": "a", "kind": "mystery"}]))
    with pytest.raises(ValueError, match="states"):
        judgment_loss(batch(out), labelled([], []))


def test_logits_and_candidates_must_agree():
    out = {"logits": {"a": torch.zeros(3)}, "candidates": {"a": ["c0", "c1"]}}
    with pytest.raises(ValueError, match="candidates"):
        judgment_loss(batch(out), labelled([{"question_id": "a", "kind": "single", "answer": "c0"}]))


# --------------------------------------------------------------------------
# 4a ↔ 4b 인수: 실제 D0 레코드의 직렬화 결과(candidate_mapping)와 실제 라벨을 그대로 잇는다
# --------------------------------------------------------------------------


def judge_like_outputs(mapping: dict[str, list[str]], seed: int) -> dict[str, torch.Tensor]:
    """Judge가 낼 모양: 질문 → 요청 순서 후보 logits. 값은 무작위다."""
    generator = torch.Generator().manual_seed(seed)
    return {
        qid: torch.randn(len(ids), generator=generator).requires_grad_(True) for qid, ids in mapping.items()
    }


def contributing(labels: list[dict], mapping: dict[str, list[str]]) -> set[str]:
    """gradient가 0이 아닐 라벨의 질문 id.

    mask=false, 근거 없는 사건, 그리고 허용 집합 A ∪ unknown이 후보 전부인 valid_set(전환 틱의
    `q_gripper` 두 상태 허용처럼 손실이 항등적으로 0인 라벨)은 뺀다.
    """
    active: set[str] = set()
    for label in labels:
        if label.get("mask", True) is False:
            continue
        if label["kind"] == "event" and label["successes"] + label["failures"] == 0:
            continue
        if label["kind"] == "valid_set":
            allowed = set(label["candidate_ids"]) | set(label.get("unknown", []))
            if allowed >= set(mapping[label["question_id"]]):
                continue
        active.add(label["question_id"])
    return active


def test_serialized_d0_single_requests_feed_the_loss(singles):
    tokenizer = WhitespaceTokenizer()
    with_labels = 0
    for index, record in enumerate(singles):
        out = serialize_request(record, tokenizer)
        logits = judge_like_outputs(out["candidate_mapping"], seed=index)
        labels = record.get("labels", [])
        loss = judgment_loss(
            {"logits": [logits], "candidates": [out["candidate_mapping"]]}, {"labels": [labels]}
        )
        assert loss.shape == () and torch.isfinite(loss) and loss.requires_grad
        loss.backward()
        active = contributing(labels, out["candidate_mapping"])
        with_labels += bool(active)
        for qid, tensor in logits.items():
            if qid in active:
                assert tensor.grad is not None and torch.isfinite(tensor.grad).all()
                assert tensor.grad.abs().sum() > 0
            else:
                assert tensor.grad is None or torch.equal(tensor.grad, torch.zeros_like(tensor))
    assert with_labels > 0


def test_serialized_d0_stream_ticks_feed_the_loss(streams):
    record = copy.deepcopy(streams[0])
    out = serialize_request(record, WhitespaceTokenizer(), layout="stream_l1a")
    per_tick = [judge_like_outputs(tick["candidate_mapping"], seed=tick["index"]) for tick in out["ticks"]]
    loss = judgment_loss(
        {"logits": per_tick, "candidates": [tick["candidate_mapping"] for tick in out["ticks"]]},
        {"labels": [tick.get("labels", []) for tick in record["ticks"]]},
    )
    assert loss.shape == () and torch.isfinite(loss) and loss.item() > 0
    loss.backward()
    for tick, serialized, logits in zip(record["ticks"], out["ticks"], per_tick):
        active = contributing(tick.get("labels", []), serialized["candidate_mapping"])  # 라벨 있는 질문만 gradient
        for qid, tensor in logits.items():
            if qid in active:
                assert tensor.grad is not None and tensor.grad.abs().sum() > 0
            else:
                assert tensor.grad is None or torch.equal(tensor.grad, torch.zeros_like(tensor))

    first = out["ticks"][0]  # 한 틱만 떼어도 같은 경로다
    one = judgment_loss(
        {"logits": [judge_like_outputs(first["candidate_mapping"], seed=99)], "candidates": [first["candidate_mapping"]]},
        {"labels": [record["ticks"][0]["labels"]]},
    )
    assert one.shape == () and torch.isfinite(one) and one.requires_grad


# --------------------------------------------------------------------------
# 상수 사전분포 head의 손실 — 질문별 기준선 (Task R8 A1)
# --------------------------------------------------------------------------

from robo_jev.loss import label_prior_share, prior_label_loss  # noqa: E402

GRIPPER = ["open", "closed"]
BOOLEAN = ["true", "false"]


def test_a_labels_share_of_the_marginal_is_its_answer_its_allowed_set_its_distribution_or_its_success_fraction():
    """라벨 하나가 배치 주변분포에 보태는 몫(합 1): single·boolean은 답 하나, valid_set은 허용 집합에 고르게(두 값 허용이면 반씩),
    distribution은 그 분포, event는 true에 성공 비율·나머지 후보에 실패 비율. 기여하지 않는 라벨은 None."""
    assert label_prior_share(GRIPPER, {"question_id": "q_gripper", "kind": "valid_set", "candidate_ids": ["closed"]}) == {"closed": 1.0}
    assert label_prior_share(GRIPPER, {"question_id": "q_gripper", "kind": "valid_set", "candidate_ids": ["open", "closed"]}) == {"open": 0.5, "closed": 0.5}
    assert label_prior_share(BOOLEAN, {"question_id": "q_stop", "kind": "single", "answer": False}) == {"false": 1.0}
    assert label_prior_share(["0", "1", "2"], {"question_id": "q_force", "kind": "single", "answer": "2"}) == {"2": 1.0}
    assert label_prior_share(["a", "b", "c"], {"question_id": "q", "kind": "distribution", "probabilities": {"a": 0.25, "c": 0.75}}) == {"a": 0.25, "c": 0.75}
    event = label_prior_share(["true", "false"], {"question_id": "q", "kind": "event", "successes": 3, "failures": 1})
    assert event == pytest.approx({"true": 0.75, "false": 0.25})
    assert label_prior_share(BOOLEAN, {"question_id": "q", "kind": "event", "successes": 0, "failures": 0}) is None
    assert label_prior_share(GRIPPER, {"question_id": "q_gripper", "kind": "valid_set", "candidate_ids": ["open"], "mask": False}) is None
    with pytest.raises(ValueError, match="후보"):
        label_prior_share(GRIPPER, {"question_id": "q_gripper", "kind": "valid_set", "candidate_ids": ["ajar"]})


@pytest.mark.parametrize(
    "candidates,label",
    [
        (GRIPPER, {"question_id": "q_gripper", "kind": "valid_set", "candidate_ids": ["closed"]}),
        (GRIPPER, {"question_id": "q_gripper", "kind": "valid_set", "candidate_ids": ["open"]}),
        (BOOLEAN, {"question_id": "q_stop", "kind": "single", "answer": True}),
        (["0", "1", "2", "3"], {"question_id": "q_speed", "kind": "valid_set", "candidate_ids": ["1", "2"]}),
        (["0", "1", "2"], {"question_id": "q_force", "kind": "single", "answer": "0"}),
        (["a", "b", "c", "d"], {"question_id": "q", "kind": "valid_set", "candidate_ids": ["b"], "unknown": ["d"]}),
        (["a", "b", "c"], {"question_id": "q", "kind": "distribution", "probabilities": {"a": 0.2, "b": 0.8}}),
        (["true", "false"], {"question_id": "q", "kind": "event", "successes": 2, "failures": 5}),
    ],
)
def test_the_constant_prior_heads_loss_is_label_loss_on_the_log_prior(candidates, label):
    """기준선의 정의 = 같은 `label_loss` 식에 logits 대신 log π(이 상태의 후보 위로 다시 정규화한 사전분포)를 넣은 값 — 모델 손실과
    같은 자로 잰다. 후보 밖에 있는 사전분포 질량(다른 상태의 후보)은 정규화에서 빠진다."""
    prior = {cid: 0.1 + 0.2 * position for position, cid in enumerate(candidates)}
    prior["elsewhere"] = 5.0  # 이 상태의 후보가 아니다 — 무시돼야 한다
    total = sum(prior[cid] for cid in candidates)
    logits = torch.tensor([math.log(prior[cid] / total) for cid in candidates], dtype=torch.float64)
    expected = float(label_loss(logits, list(candidates), label))
    assert prior_label_loss(prior, list(candidates), label) == pytest.approx(expected, rel=1e-12, abs=1e-12)


def test_a_two_valued_gripper_label_costs_nothing_for_the_model_and_for_the_baseline():
    """두 값 허용(open·closed 둘 다 허용)은 후보 전체라 어떤 head에도 손실 0 — 모델과 기준선 모두 0이다(질량만 같이 진다)."""
    label = {"question_id": "q_gripper", "kind": "valid_set", "candidate_ids": ["open", "closed"]}
    assert prior_label_loss({"open": 0.9, "closed": 0.1}, GRIPPER, label) == pytest.approx(0.0, abs=1e-15)
    assert float(label_loss(torch.tensor([3.0, -2.0]), GRIPPER, label)) == pytest.approx(0.0, abs=1e-6)


def test_a_uniform_prior_is_the_no_information_baseline_of_a_dynamic_candidate_question():
    """후보가 틱마다 바뀌는 질문(q_main·q_path·비로봇 choice)의 기준선: 이 상태 후보 위의 균등 — single은 log K, 허용 집합 A는
    −log(|A| / |A ∪ I|)(unknown은 정규화에서 빠진다)."""
    candidates = ["c1", "c2", "c3", "c4", "c5"]
    uniform = {cid: 1.0 for cid in candidates}
    assert prior_label_loss(uniform, candidates, {"question_id": "q_main", "kind": "valid_set", "candidate_ids": ["c2"]}) == pytest.approx(math.log(5))
    assert prior_label_loss(uniform, candidates, {"question_id": "q_main", "kind": "valid_set", "candidate_ids": ["c2", "c3"], "unknown": ["c5"]}) == pytest.approx(-math.log(2 / 4))


def test_an_event_label_with_only_failures_does_not_turn_a_zero_prior_into_nan():
    """성공 0인 사건 라벨은 log π(true) 항의 계수가 0이다 — π(true) = 0이어도 0·log 0 = 0으로 읽는다(nan이 아니다)."""
    value = prior_label_loss({"true": 0.0, "false": 1.0}, ["true", "false"], {"question_id": "q", "kind": "event", "successes": 0, "failures": 3})
    assert value == 0.0


def test_a_label_that_contributes_nothing_to_the_model_loss_has_no_baseline_either():
    """mask=false·근거 없는 사건 라벨은 모델 손실에서 빠지므로(`label_loss`가 None) 기준선도 None이다 — 같은 라벨 집합 위에서 잰다."""
    prior = {"true": 0.5, "false": 0.5}
    assert prior_label_loss(prior, BOOLEAN, {"question_id": "q_stop", "kind": "single", "answer": True, "mask": False}) is None
    assert prior_label_loss(prior, BOOLEAN, {"question_id": "q", "kind": "event", "successes": 0, "failures": 0}) is None
