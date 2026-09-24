"""모델 정책 어댑터 검사 (Task R4 A1) — 루프 안의 모델이 오프라인 재생 평가와 **틱마다 같은 답**을 내는가.

소형 fixture backbone(CPU)으로 잰다: 같은 직렬화·같은 가중치면 `predict_items`(에피소드 재생)와 `ModelPolicy.act`(틱마다
증분)의 질문별 확률이 같아야 한다. 실제 2B의 같은 증명은 GPU 산출물(`artifacts/reports/r4-a3-*.json`)이 든다.
"""

import copy

import pytest
import torch
from helpers import D0_MANIFEST, SMALL_VOCAB

from robo_jev.contracts import QUESTION_SET_V0, validate_record
from robo_jev.evaluate import predict_items
from robo_jev.harness.model_policy import POLICY_VERSION, ModelPolicy, answers_from_typed, batched_pointer_logits
from robo_jev.model.judge import Judge
from robo_jev.model.serialize import serialize_request
from robo_jev.model.tokenizer import WhitespaceTokenizer
from robo_jev.sampler import load_items

TICKS = 16  # 지시 변경(틱 12)과 동적 주기(10)를 지난다


def _request_view(tick: dict) -> dict:
    """레코드의 틱에서 하네스가 그 틱에 냈던 요청만 — 출력·라벨은 정책이 보지 않는다."""
    return {key: copy.deepcopy(tick[key]) for key in ("t", "sim_ms", "observed_at_ms", "obs_age_ms", "request")}


@pytest.fixture(scope="module")
def setup():
    tokenizer = WhitespaceTokenizer()
    judge = Judge.from_config(seed=5, vocab_size=SMALL_VOCAB)
    items = load_items(D0_MANIFEST, tokenizer=tokenizer, splits=("dev",), stream_max_ticks=TICKS)
    item = next(entry for entry in items if entry.kind == "stream")
    predictions = predict_items(judge, [item])
    return {"tokenizer": tokenizer, "judge": judge, "item": item, "predictions": predictions}


def _run_policy(setup, *, instructions=True) -> tuple[ModelPolicy, list[dict]]:
    record = setup["item"].record
    policy = ModelPolicy(setup["judge"], setup["tokenizer"], question_set=record["prefix"]["question_set"])
    if instructions:
        policy.begin_episode(instructions=record["prefix"]["instructions"])
    answers = [policy.act(_request_view(tick), {"junk": True}, {"scene": "ignored"}) for tick in record["ticks"]]
    return policy, answers


def test_the_policy_answers_every_tick_like_the_offline_replay(setup):
    """확률이 같고(허용 오차 안) argmax가 같다 — 열 질문 모두, 틱 16개 모두."""
    _, answers = _run_policy(setup)
    predictions = setup["predictions"]
    assert len(answers) == len(predictions) == TICKS
    for tick_answers, prediction in zip(answers, predictions):
        assert set(tick_answers) == set(QUESTION_SET_V0)
        # D0 fixture에는 경로 후보가 없다 — 묻지 않은 동적 질문은 전문가처럼 빈 분포다
        assert all(tick_answers[qid] == {} for qid in QUESTION_SET_V0 if qid not in prediction["probabilities"])
        for qid, probabilities in prediction["probabilities"].items():
            ids = list(prediction["candidates"][qid])
            answer = tick_answers[qid]
            if QUESTION_SET_V0[qid]["type"] == "boolean":
                assert isinstance(answer, float)  # 전문가·규칙 판정기와 같은 꼴: p_true
                vector = torch.tensor([answer if cid == "true" else 1.0 - answer for cid in ids])
            else:
                assert list(answer) == ids  # 후보 순서 = 요청 순서
                vector = torch.tensor([float(answer[cid]) for cid in ids])
            assert torch.allclose(vector, probabilities, atol=1e-5), (prediction["tick"], qid)
            assert int(vector.argmax()) == int(probabilities.argmax()), (prediction["tick"], qid)


def test_the_policy_derives_a_mid_episode_instruction_from_the_goal_when_none_was_given(setup):
    """루프에서는 뒤에 올 지시를 모른다 — 목표 버전이 오르면 생성기 규칙으로 만든 지시가 레코드의 것과 같다."""
    record = setup["item"].record
    _, from_record = _run_policy(setup, instructions=True)
    policy, from_goal = _run_policy(setup, instructions=False)
    assert policy.serializer.instructions == record["prefix"]["instructions"]
    assert from_goal == from_record


def test_tick_zero_resets_the_episode_and_a_tick_number_that_does_not_grow_is_refused(setup):
    """틱 번호는 제어 스텝 수(실제 하네스에서 0, 5, 10, …)라 +1이 아니라 증가만 요구한다 — 계약과 같은 규칙."""
    record = setup["item"].record
    policy = ModelPolicy(setup["judge"], setup["tokenizer"], question_set=record["prefix"]["question_set"])
    first = policy.act(_request_view(record["ticks"][0]))
    policy.act(_request_view(record["ticks"][1]))
    stale = _request_view(record["ticks"][2])
    stale["t"] = 1
    with pytest.raises(ValueError, match="t"):
        policy.act(stale)
    again = policy.act(_request_view(record["ticks"][0]))  # t == 0은 새 에피소드다
    assert again == first and policy.tick == 0 and policy.episodes == 2
    policy.reset()
    with pytest.raises(ValueError, match="t"):
        policy.act(_request_view(record["ticks"][1]))  # 에피소드는 0에서 시작한다


def test_the_policy_reads_nothing_but_the_request(setup):
    """commitment·관측 인자는 정보 경계 밖이다 — 무엇을 넣어도 답이 같다."""
    record = setup["item"].record
    policy = ModelPolicy(setup["judge"], setup["tokenizer"], question_set=record["prefix"]["question_set"])
    a = policy.act(_request_view(record["ticks"][0]), None, None)
    policy.reset()
    b = policy.act(_request_view(record["ticks"][0]), {"action_ref": "c9", "phase": "grasp"}, {"objects": [], "instruction": {"text": "x"}})
    assert a == b


def test_timing_rows_are_recorded_per_tick(setup):
    policy, _ = _run_policy(setup)
    rows = policy.timing
    assert len(rows) == TICKS and [row["t"] for row in rows] == list(range(TICKS))
    for row, prediction in zip(rows, setup["predictions"]):
        assert {"t", "tokens_body", "tokens_decision", "cached_tokens", "serialize_ms", "model_ms", "readout_ms", "act_ms", "prefix_tokens"} <= set(row)
        assert row["act_ms"] >= row["model_ms"] >= 0.0 and row["tokens_decision"] == len(prediction["probabilities"])  # 결정 표지 = 그 틱에 물은 질문 수
    assert rows[0]["prefix_tokens"] == policy.serializer.prefix_end and all(row["prefix_tokens"] == 0 for row in rows[1:])
    assert policy.describe()["version"] == POLICY_VERSION and policy.describe()["fused"] is False  # fixture에는 fused 경로가 없다


def test_batched_readout_equals_the_judges_pointer_logits(setup):
    """열 결정을 한 matmul로 읽는 것이 질문마다 `pointer_logits`를 부르는 것과 같은 수다."""
    record = setup["item"].record
    tokenizer, judge = setup["tokenizer"], setup["judge"]
    layout = serialize_request(record, tokenizer, layout="stream_l1a")
    result = judge({"layout": "stream_l1a", "stream": layout})
    tick_state = result["tick_states"][3]
    entry = layout["ticks"][3]
    branch_hidden = tick_state.branch_step([layout["tokens"][index] for index in entry["decision_positions"].values()])
    batched = batched_pointer_logits(judge, tick_state, entry, branch_hidden, int(layout["prefix_end"]))
    for row, (qid, decision) in enumerate(entry["decision_positions"].items()):
        boundaries = [int(b) for b in entry["candidate_boundaries"][qid]]
        expected = judge.pointer_logits(result["hidden"][int(decision)], result["hidden"][boundaries])
        assert torch.allclose(batched[qid], expected, atol=1e-5), qid


def test_answers_from_typed_keeps_the_model_output_format():
    typed = {
        "q_main": {"type": "choice", "probabilities": {"c1": 0.7, "c2": 0.3}, "choice": "c1"},
        "q_done": {"type": "boolean", "probabilities": {"true": 0.2, "false": 0.8}, "choice": "false", "p_true": 0.2},
        "q_speed": {"type": "ordinal", "probabilities": {"0": 0.5, "1": 0.5}, "choice": "0", "expected_value": 0.05},
    }
    assert answers_from_typed(typed) == {"q_main": {"c1": 0.7, "c2": 0.3}, "q_done": 0.2, "q_speed": {"0": 0.5, "1": 0.5}}


def test_the_policy_runs_in_the_generator_loop_with_the_expert_as_the_reference():
    """루프 한 바퀴 (docs/06 Task 3 DAgger 씨앗): 정책은 모델, 참조·라벨은 전문가, `model_output`은 모델의 raw 답."""
    from robo_jev.data.robot_episodes import generate_episode, load_generator_config
    from robo_jev.sim.expert import Expert

    config = load_generator_config()
    judge = Judge.from_config(seed=5, vocab_size=1 << 15)  # 공백 tokenizer의 어휘가 실제 틱에서 자란다
    policy = ModelPolicy(judge, WhitespaceTokenizer())
    expert = Expert()
    record = generate_episode("E0", 17, policy=policy, expert=expert, config=config, max_ticks=6)
    validate_record(record)
    assert record["provenance"]["policy"] == {"name": "ModelPolicy", "version": POLICY_VERSION}
    assert len(record["ticks"]) == 6 and len(policy.timing) == 6
    assert policy.tick == record["ticks"][-1]["t"] == 25  # 틱 번호는 제어 스텝 수 (10 Hz 틱 = 50 Hz 5회)
    assert policy.serializer.instructions == record["prefix"]["instructions"]
    for tick in record["ticks"]:
        ids = [entry["id"] for entry in tick["request"]["candidates"]["q_main"]]
        assert list(tick["model_output"]["q_main"]) == ids
        assert all(label["source"] == "expert_v0" for label in tick["labels"])
        assert isinstance(tick["model_output"]["q_stop"], float)
