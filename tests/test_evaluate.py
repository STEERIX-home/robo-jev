"""평가 지표·대조군·치환 증강·무학습 점수 읽기의 검사 (G0b S3) — 소형 fixture와 소형 난수 Qwen으로 CPU에서."""

import copy

import pytest
import torch
from helpers import D0_MANIFEST, SMALL_VOCAB

from robo_jev.contracts import validate_record
from robo_jev.evaluate import aggregate, answer_change_rate, context_shuffle_records, evaluate_items, label_metrics, predict_items, rule_judge_predictions
from robo_jev.model.judge import Judge
from robo_jev.model.serialize import serialize_request
from robo_jev.model.tokenizer import WhitespaceTokenizer, available_tokenizer, load_tokenizer
from robo_jev.sampler import Item, load_items, permute_candidates


@pytest.fixture(scope="module")
def items():
    tokenizer = WhitespaceTokenizer()
    return load_items(D0_MANIFEST, tokenizer=tokenizer, splits=("dev",), stream_max_ticks=3), tokenizer


def test_permute_candidates_is_deterministic_per_seed_and_keeps_labels_and_contract_valid(singles, streams):
    single = next(r for r in singles if any(q["type"] == "choice" and len(q["criteria"]) >= 3 for q in r["request"]["questions"]))
    once, again, other = permute_candidates(single, 7), permute_candidates(single, 7), permute_candidates(single, 8)
    validate_record(once)
    assert once == again and once != single
    ids = lambda rec: [[c["id"] for c in q["criteria"]] for q in rec["request"]["questions"]]  # noqa: E731
    assert all(sorted(a) == sorted(b) for a, b in zip(ids(once), ids(single))) and any(a != b for a, b in zip(ids(once), ids(single)))
    assert any(a != b for a, b in zip(ids(once), ids(other)))
    assert [q["criteria"] for q in once["request"]["questions"] if q["type"] != "choice"] == [q["criteria"] for q in single["request"]["questions"] if q["type"] != "choice"]
    stream = copy.deepcopy(streams[0])
    stream["ticks"] = stream["ticks"][:3]
    permuted = permute_candidates(stream, 3)
    validate_record(permuted)
    original_ids = [[c["id"] for c in t["request"]["candidates"]["q_main"]] for t in stream["ticks"]]
    permuted_ids = [[c["id"] for c in t["request"]["candidates"]["q_main"]] for t in permuted["ticks"]]
    assert all(sorted(a) == sorted(b) for a, b in zip(original_ids, permuted_ids)) and any(a != b for a, b in zip(original_ids, permuted_ids))
    tokenizer = WhitespaceTokenizer()
    layout = serialize_request(permuted, tokenizer, layout="stream_l1a")
    assert layout["ticks"][0]["candidate_mapping"]["q_main"] == permuted_ids[0]
    with pytest.raises(ValueError, match="schema_version"):
        permute_candidates({"schema_version": "x"}, 1)


def test_label_metrics_follow_the_documented_definitions():
    p = torch.tensor([0.7, 0.2, 0.1])
    single = label_metrics(p, ["a", "b", "c"], {"kind": "single", "answer": "a"})
    assert single["correct"] and single["predicted"] == "a" and single["position"] == 0
    assert single["nll"] == pytest.approx(-torch.log(torch.tensor(0.7)).item(), abs=1e-6) and single["brier"] == pytest.approx(0.09 + 0.04 + 0.01)
    valid = label_metrics(p, ["a", "b", "c"], {"kind": "valid_set", "candidate_ids": ["b", "c"]})
    assert not valid["correct"] and valid["brier"] == pytest.approx((0.3 - 1.0) ** 2 + 0.49)
    dist = label_metrics(p, ["a", "b", "c"], {"kind": "distribution", "probabilities": {"a": 0.5, "b": 0.5}})
    assert dist["correct"] and dist["brier"] == pytest.approx(0.04 + 0.09 + 0.01)
    event = label_metrics(torch.tensor([0.25, 0.75]), ["true", "false"], {"kind": "event", "successes": 1, "failures": 3})
    assert event["correct"] is None and event["brier"] == pytest.approx(0.0)
    assert label_metrics(p, ["a", "b", "c"], {"kind": "single", "answer": "a", "mask": False}) is None
    boolean = label_metrics(torch.tensor([0.9, 0.1]), ["true", "false"], {"kind": "single", "answer": True})
    assert boolean["correct"] and boolean["predicted"] == "true"


def test_evaluate_items_reports_tables_controls_and_position_bias_on_the_fixture(items):
    loaded, tokenizer = items
    judge = Judge.from_config(seed=5, vocab_size=SMALL_VOCAB)
    result = evaluate_items(judge, loaded, tokenizer=tokenizer, shuffle_seed=2, instruction_shuffle=True)
    assert result["n_items"] == len(loaded) and result["n_states"] > len(loaded)
    model = result["model"]
    assert {"q_main", "q_done", "choice", "boolean", "_all"} <= set(model)
    assert 0.0 <= model["q_main"]["accuracy"] <= 1.0 and model["_all"]["nll"] > 0 and model["_all"]["brier"] >= 0
    assert model["q_main"]["first_position_rate"] is not None and sum(model["q_main"]["position_counts"]) == model["q_main"]["n"]
    assert model["q_done"]["first_position_rate"] is None  # boolean은 위치 편향을 세지 않는다
    assert result["answer_change"]["shuffle_seed"] == 2 and result["answer_change"]["compared"] > 0 and 0.0 <= result["answer_change"]["rate"] <= 1.0
    assert "q_main" in result["answer_change"]["by_question"] and "permuted" in result
    assert "context_shuffle" in result and 0.0 <= result["context_shuffle"]["_all"]["accuracy"] <= 1.0
    assert result["context_shuffle_kind"] == "state"  # 표준 열 = 상태 섞기: 비로봇은 상태 전체, 로봇 스트림은 id를 재매핑한 구조화 상태 (D1 리뷰 1 I1)
    assert result["instruction_shuffle_kind"] == "instruction" and "q_main" in result["instruction_shuffle"] and "choice" not in result["instruction_shuffle"]
    assert result["instruction_shuffle"]["q_main"]["n"] == model["q_main"]["n"]  # 지시 섞기 열은 로봇 스트림만, 같은 틱 수
    assert "instruction_shuffle" not in evaluate_items(judge, loaded, tokenizer=tokenizer, shuffle_seed=None, rule_judge=False)
    rule = result["rule_judge"]
    assert set(rule) - {"_all"} <= set(model) and rule["q_main"]["n"] == model["q_main"]["n"] and 0.0 <= rule["q_main"]["accuracy"] <= 1.0
    assert "choice" not in rule  # 규칙 기준군은 로봇 틱만


def test_context_shuffle_rolls_state_for_singles_and_text_or_remapped_state_for_streams(singles, streams):
    rolled = context_shuffle_records(singles[:3])
    assert rolled[0]["request"]["state"] == singles[1]["request"]["state"] and rolled[2]["request"]["state"] == singles[0]["request"]["state"]
    assert rolled[0]["labels"] == singles[0]["labels"] and rolled[0]["request"]["questions"] == singles[0]["request"]["questions"]
    for record in rolled:
        validate_record(record)
    episodes = [copy.deepcopy(r) for r in streams[:2]]
    for episode in episodes:
        episode["ticks"] = episode["ticks"][:2]
    # 지시 섞기: 텍스트만 굴리고 구조화 goal·물리 상태·후보는 그대로.
    swapped = context_shuffle_records(episodes, robot="instruction")
    assert swapped[0]["prefix"]["instructions"][0]["text"] == episodes[1]["prefix"]["instructions"][0]["text"]
    assert swapped[0]["ticks"][0]["request"]["candidates"] == episodes[0]["ticks"][0]["request"]["candidates"]
    own, rolled_state = episodes[0]["ticks"][0]["request"]["state"], swapped[0]["ticks"][0]["request"]["state"]
    assert rolled_state["objects"] == own["objects"] and {k: v for k, v in rolled_state["goal"].items() if k != "text"} == {k: v for k, v in own["goal"].items() if k != "text"}
    for record in swapped:
        validate_record(record)
    assert context_shuffle_records(singles[:1])[0] == singles[0] and context_shuffle_records(streams[:1])[0] == streams[0]
    with pytest.raises(ValueError, match="robot"):
        context_shuffle_records(episodes, robot="objects")


def test_the_robot_state_shuffle_remaps_donor_ids_onto_the_ticks_ids_and_keeps_the_ticks_own_execution(streams):
    """D1 리뷰 1 I1: 기증 에피소드의 goal·물체·영역·장면·파생 값이 이 틱의 id 공간으로 재매핑되어 들어오고(D0 fixture는 에피소드마다 물체 id가
    다르다: o7·o3·o4 / o7·o3·o5 / o4·o3·o7), 후보·commitment·실행 이력·robot·exec·t는 이 틱의 것이다."""
    rolled = context_shuffle_records(streams, robot="state")
    assert len(rolled) == len(streams) >= 3
    changed_objects = 0
    for position, (original, new) in enumerate(zip(streams, rolled)):
        donor = streams[(position + 1) % len(streams)]
        assert new["prefix"]["instructions"][0]["text"] == donor["prefix"]["instructions"][0]["text"]
        validate_record(new)
        for index, (tick, shuffled) in enumerate(zip(original["ticks"], new["ticks"])):
            own, state = tick["request"]["state"], shuffled["request"]["state"]
            donor_state = donor["ticks"][min(index, len(donor["ticks"]) - 1)]["request"]["state"]
            own_ids, donor_ids = [o["id"] for o in own["objects"]], [o["id"] for o in donor_state["objects"]]
            # id 공간은 이 틱의 것, 내용(설명·자세)은 기증 틱의 것 (자리 순서 재매핑).
            assert [o["id"] for o in state["objects"]] == own_ids
            assert [o["desc"] for o in state["objects"]][: len(donor_ids)] == [o["desc"] for o in donor_state["objects"]][: len(own_ids)]
            remap = dict(zip(donor_ids, own_ids))
            assert state["goal"]["text"] == donor_state["goal"]["text"]
            assert state["goal"].get("forbidden_contact") == [remap.get(i, i) for i in donor_state["goal"].get("forbidden_contact", [])]
            assert set(state["goal"].get("forbidden_contact", [])) <= set(own_ids)
            assert [z["id"] for z in state["zones"]] == [z["id"] for z in own["zones"]]
            assert state["scene"] == donor_state["scene"]
            # 이 틱의 것: 후보·commitment·실행 이력·robot·exec·t.
            assert shuffled["request"]["candidates"] == tick["request"]["candidates"] and shuffled["request"].get("commitment") == tick["request"].get("commitment")
            assert shuffled["request"].get("exec_history") == tick["request"].get("exec_history")
            for key in ("robot", "exec", "t", "events"):
                assert state.get(key) == own.get(key), key
            assert shuffled["labels"] == tick["labels"]
            changed_objects += int([o["desc"] for o in state["objects"]] != [o["desc"] for o in own["objects"]])
    assert changed_objects > 0


def test_the_robot_state_shuffle_moves_the_rule_judges_main_answer_where_the_text_shuffle_does_not():
    """구조화된 goal(D1 서식)에서 규칙 기준군은 텍스트를 읽지 않으므로 지시 섞기는 답을 바꾸지 못하고, 상태 섞기(목표·물체·영역이 기증
    에피소드의 것)는 바꾼다 — 대조군이 무엇을 재는지의 fixture 수준 증거 (D1 리뷰 1 I1)."""
    from robo_jev.data.robot_episodes import generate_episode, load_generator_config
    from robo_jev.sim.expert import Expert

    config, expert = load_generator_config(), Expert()
    episodes = [generate_episode("E0", seed, policy=expert, expert=expert, config=config, max_ticks=20) for seed in (5, 7)]  # 대상 o0→zoneL vs o1→zoneR
    goals = [episode["ticks"][0]["request"]["state"]["goal"] for episode in episodes]
    assert (goals[0]["target_ref"], goals[0]["target_zone"]) != (goals[1]["target_ref"], goals[1]["target_zone"]) and all("target_desc" in goal for goal in goals)
    tokenizer = WhitespaceTokenizer()

    def items(records):
        out = []
        for index, record in enumerate(records):
            layout = serialize_request(record, tokenizer, layout="stream_l1a")
            out.append(Item(index=index, kind="stream", record_id=record["episode_id"], split=record["split"], domain="robot", material="existing", record=record, layout=layout, tokens=len(layout["tokens"]), question_types={q: "choice" for q in layout["ticks"][0]["candidate_mapping"]}))
        return out

    def main_answers(records):
        return [(p["record_id"], p["tick"], p["candidates"]["q_main"][int(p["probabilities"]["q_main"].argmax())]) for p in rule_judge_predictions(items(records))]

    base = main_answers(episodes)
    text = main_answers(context_shuffle_records(episodes, robot="instruction"))
    state = main_answers(context_shuffle_records(episodes, robot="state"))
    assert len(base) == 40 and text == base
    moved = sum(1 for a, b in zip(base, state) if a != b)
    assert moved >= len(base) // 2, (moved, len(base))


def test_rule_judge_predictions_cover_every_posed_question_with_a_distribution(items):
    loaded, _ = items
    predictions = rule_judge_predictions([item for item in loaded if item.kind == "stream"])
    assert predictions and all(p["kind"] == "stream" for p in predictions)
    first = predictions[0]
    assert set(first["probabilities"]) == set(first["candidates"])
    for qid, p in first["probabilities"].items():
        assert p.shape[0] == len(first["candidates"][qid]) and p.sum().item() == pytest.approx(1.0, abs=1e-5)


def test_answer_change_rate_compares_the_same_state_and_question_only():
    a = [{"record_id": "r", "tick": 0, "kind": "single", "probabilities": {"q": torch.tensor([0.6, 0.4])}, "candidates": {"q": ["x", "y"]}, "question_types": {"q": "choice"}}]
    b = [{"record_id": "r", "tick": 0, "kind": "single", "probabilities": {"q": torch.tensor([0.6, 0.4])}, "candidates": {"q": ["y", "x"]}, "question_types": {"q": "choice"}}]
    assert answer_change_rate(a, b) == {"compared": 1, "rate": 1.0, "by_question": {"choice": 1.0}}
    assert answer_change_rate(a, a)["rate"] == 0.0 and answer_change_rate(a, [])["rate"] is None


def test_zero_shot_prompts_use_single_token_codes_and_score_on_the_tiny_qwen(singles, streams):
    pytest.importorskip("transformers")
    from robo_jev.model.backbone_qwen import QwenBackbone, torch_reference_kernels
    from robo_jev.model.zero_shot import build_single_prompts, build_tick_prompts, verify_codes, zero_shot_label_scores

    tokenizer = load_tokenizer(available_tokenizer()[0])
    codes, ids = verify_codes(tokenizer, 12)
    assert codes == list("ABCDEFGHIJKL") and len(set(ids)) == 12 and all(len(tokenizer.encode(" " + c, add_special_tokens=False).ids) == 1 for c in codes)
    single = next(r for r in singles if len(r["request"]["questions"]) == 3 and len(r["labels"]) == 3)
    prompts = build_single_prompts(single, tokenizer, shuffle_seed=1)
    assert len(prompts) == 3 and all(p["tokens"][-1] == tokenizer.encode("Answer:", add_special_tokens=False).ids[-1] for p in prompts)
    assert all(sorted(p["candidates"]) == sorted(c["id"] for c in q["criteria"]) for p, q in zip(prompts, single["request"]["questions"]))
    assert build_single_prompts(single, tokenizer, shuffle_seed=1) == prompts and build_single_prompts(single, tokenizer, shuffle_seed=None)[0]["candidates"] == [c["id"] for c in single["request"]["questions"][0]["criteria"]]
    stream = copy.deepcopy(streams[0])
    stream["ticks"] = stream["ticks"][:4]
    tick_prompts = build_tick_prompts(stream, 3, tokenizer, shuffle_seed=1, window_ticks=2)
    assert [p["question_id"] for p in tick_prompts] == list(serialize_request(stream, tokenizer, layout="stream_l1a")["ticks"][3]["candidate_mapping"])
    assert all(len(p["codes"]) == len(p["candidates"]) for p in tick_prompts)
    tiny = QwenBackbone.tiny(seed=4, vocab_size=tokenizer.get_vocab_size())
    with torch_reference_kernels():
        result = zero_shot_label_scores("tiny", [single, stream], backbone=tiny, tokenizer=tokenizer, shuffle_seed=1, tick_stride=2, batch=4)
    posed = serialize_request(stream, tokenizer, layout="stream_l1a")["ticks"]
    assert result["prompts"] == 3 + len(posed[0]["candidate_mapping"]) + len(posed[2]["candidate_mapping"])  # 틱 0·2 (stride 2)
    assert {"choice", "boolean", "q_main", "_all"} <= set(result["table"])
    assert 0.0 <= result["table"]["_all"]["accuracy"] <= 1.0 and result["table"]["_all"]["nll"] > 0
