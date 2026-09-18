"""손실 검사 — docs/03 §4, docs/08 §7.

작은 logits에 손으로 계산한 값을 대조한다. valid-set CE, 단일 정답 CE, 분포 CE, 사건 Bernoulli
NLL, 결측 mask, 상태별 정규화, `unknown` 후보를 정규화에서 빼는 부분 라벨 손실.
"""

import math

import pytest
import torch

from robo_jev.loss import judgment_loss, label_loss, question_losses


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
