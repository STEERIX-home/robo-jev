"""평가 지표·대조군·치환 증강·무학습 점수 읽기의 검사 (G0b S3) — 소형 fixture와 소형 난수 Qwen으로 CPU에서."""

import copy
import math

import pytest
import torch
from helpers import D0_MANIFEST, SMALL_VOCAB

from robo_jev.contracts import validate_record
from robo_jev.evaluate import aggregate, answer_change_rate, context_shuffle_records, episode_bootstrap, evaluate_items, label_metrics, predict_items, rule_judge_predictions
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
    # 예측은 **레코드마다 하나**로 합쳐 나온다 (R1 fix round 1): 무학습 채점은 질문 하나가 프롬프트 하나라
    # 예측도 질문마다 나지만, 대조 쌍 검사는 한 레코드의 여러 질문을 한 자리에서 본다.
    merged = result["predictions"]
    assert len(merged) == 1 + 2  # 단일 요청 하나 + 채점한 두 틱
    by_single = next(item for item in merged if item["kind"] == "single")
    assert set(by_single["probabilities"]) == {q["id"] for q in single["request"]["questions"]}
    assert set(by_single["candidates"]) == set(by_single["probabilities"]) and len(by_single["labels"]) == 3


# --------------------------------------------------------------------------
# 고정 평가 집합 (configs/eval/pilot.yaml; G0b OQ8)
# --------------------------------------------------------------------------


def _suite_file(tmp_path, **overrides) -> str:
    import yaml

    config = {
        "version": "test-suite-v0", "window_ticks": 30, "shuffle_seed": 2, "fused": False,
        "columns": {"permuted": True, "state_shuffle": True, "instruction_shuffle": True, "rule_judge": True, "selective": True, "calibration": True},
        "splits": [
            {"name": "d0/dev", "manifest": str(D0_MANIFEST), "domain": "robot", "split": "dev", "files": ["d0_streams.jsonl"], "max_ticks": 3, "selection": True},
            {"name": "d0/dev_singles", "manifest": str(D0_MANIFEST), "domain": "non_robot", "split": "dev", "files": ["d0.jsonl"], "limit": 3, "selection": False},
        ],
    }
    config.update(overrides)
    path = tmp_path / "suite.yaml"
    path.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return str(path)


def test_the_eval_suite_config_picks_fixed_records_and_its_identity_moves_with_them(tmp_path):
    """평가 집합은 설정이다 — 어떤 레코드를 몇 틱까지 읽는지가 파일에 있고, 실제로 읽힌 id·상태 수의 해시가 보고서에 들어간다."""
    from robo_jev.evaluate import eval_suite_identity, load_eval_suite, load_suite_items

    tokenizer = WhitespaceTokenizer()
    suite = load_eval_suite(_suite_file(tmp_path))
    items = load_suite_items(suite, tokenizer=tokenizer)
    assert set(items) == {"d0/dev", "d0/dev_singles"} and len(items["d0/dev_singles"]) == 3
    assert all(len(item.record["ticks"]) <= 3 for item in items["d0/dev"])
    identity = eval_suite_identity(suite, items)
    assert identity["splits"]["d0/dev_singles"]["selection"] is False and identity["splits"]["d0/dev"]["max_ticks"] == 3
    assert identity["splits"]["d0/dev"]["records"] == [item.record_id for item in items["d0/dev"]]
    assert len(identity["sha256"]) == 64

    smaller = load_eval_suite(_suite_file(tmp_path, splits=[
        {"name": "d0/dev", "manifest": str(D0_MANIFEST), "domain": "robot", "split": "dev", "files": ["d0_streams.jsonl"], "max_ticks": 2},
        {"name": "d0/dev_singles", "manifest": str(D0_MANIFEST), "domain": "non_robot", "split": "dev", "files": ["d0.jsonl"], "limit": 3, "selection": False},
    ]))
    assert eval_suite_identity(smaller, load_suite_items(smaller, tokenizer=tokenizer))["sha256"] != identity["sha256"]

    with pytest.raises(ValueError, match="알 수 없는 키"):
        load_eval_suite(_suite_file(tmp_path, kind="oops"))
    with pytest.raises(ValueError, match="이름이 중복"):
        load_eval_suite(_suite_file(tmp_path, splits=[
            {"name": "same", "manifest": str(D0_MANIFEST), "domain": "robot", "split": "dev"},
            {"name": "same", "manifest": str(D0_MANIFEST), "domain": "non_robot", "split": "dev"},
        ]))
    chosen = load_eval_suite(_suite_file(tmp_path, splits=[{"name": "d0/dev", "manifest": str(D0_MANIFEST), "domain": "robot", "split": "dev", "files": ["d0_streams.jsonl"], "records": ["없는-에피소드"]}]))
    with pytest.raises(ValueError, match="설정이 고른 레코드가"):
        load_suite_items(chosen, tokenizer=tokenizer)


def test_the_eval_set_identity_covers_the_tick_stride_that_subsets_what_is_scored(tmp_path):
    """해시는 **점수를 매긴 집합**을 덮어야 한다 (P1 리뷰 1 I6).

    무학습 run은 스트림을 `tick_stride`마다 하나씩만 잰다 — 같은 설정·같은 레코드라도 점수가 매겨진 모집단이 다르다.
    솎지 않은 run(기본)은 옛 해시를 그대로 유지하고(키가 없다), 솎은 run은 다른 해시를 받는다.
    """
    from robo_jev.evaluate import eval_suite_identity, load_eval_suite, load_suite_items

    tokenizer = WhitespaceTokenizer()
    suite = load_eval_suite(_suite_file(tmp_path))
    items = load_suite_items(suite, tokenizer=tokenizer)

    full = eval_suite_identity(suite, items)
    assert "tick_stride" not in full  # 솎지 않은 run은 이 키를 쓰지 않는다 — 기존 해시가 그대로다
    assert eval_suite_identity(suite, items, tick_stride=None)["sha256"] == full["sha256"]

    strided = eval_suite_identity(suite, items, tick_stride=16)
    assert strided["tick_stride"] == 16
    assert strided["sha256"] != full["sha256"]
    assert eval_suite_identity(suite, items, tick_stride=8)["sha256"] != strided["sha256"]


def test_evaluate_suite_runs_every_split_with_the_standard_columns_and_marks_what_is_for_selection(tmp_path):
    from robo_jev.evaluate import evaluate_suite, load_eval_suite

    tokenizer = WhitespaceTokenizer()
    suite = load_eval_suite(_suite_file(tmp_path))
    judge = Judge.from_config(seed=5, vocab_size=SMALL_VOCAB)
    result = evaluate_suite(judge, suite, tokenizer=tokenizer)
    assert set(result["splits"]) == {"d0/dev", "d0/dev_singles"} and result["tiny_scorer"] is None
    stream_table = result["splits"]["d0/dev"]
    assert stream_table["selection"] is True and result["splits"]["d0/dev_singles"]["selection"] is False
    assert stream_table["context_shuffle_kind"] == "state" and stream_table["instruction_shuffle_kind"] == "instruction"
    assert "rule_judge" in stream_table and stream_table["ece"]["n"] > 0
    assert set(stream_table["selective"]) == {"model", "rule_judge"} and 0.0 <= stream_table["selective"]["model"]["coverage"] <= 1.0
    assert "_predictions" not in stream_table and stream_table["seconds"] >= 0
    assert "selective" not in result["splits"]["d0/dev_singles"]  # 스트림이 없는 분할
    assert result["eval_set"]["sha256"] and result["eval_set"]["config"] == suite["path"]


def test_contrast_pair_check_separates_sensitivity_from_noise():
    """한 필드만 바뀐 쌍에서 라벨이 바뀐 쪽과 모델이 바뀐 쪽을 따로 센다 — 민감도와 헛흔들림은 다른 수다."""
    from robo_jev.evaluate import contrast_pair_check

    def record(request_id, role, sibling, answer, kind):
        return {"schema_version": "judgment-v0", "request": {"request_id": request_id},
                "labels": [{"question_id": "q_main", "kind": "valid_set", "candidate_ids": [answer]}],
                "provenance": {"kind": kind, "contrast": {"role": role, "sibling_id": sibling, "flipped_question": "q_main"}}}

    def prediction(request_id, best):
        probabilities = torch.tensor([0.8, 0.2]) if best == "a" else torch.tensor([0.2, 0.8])
        return {"record_id": request_id, "tick": None, "kind": "single", "probabilities": {"q_main": probabilities},
                "candidates": {"q_main": ["a", "b"]}, "labels": [], "question_types": {"q_main": "choice"}}

    records = [
        record("p1-base", "base", "p1-sib", "a", "instruction"), record("p1-sib", "sibling", "p1-base", "b", "instruction"),
        record("p2-base", "base", "p2-sib", "a", "instruction"), record("p2-sib", "sibling", "p2-base", "b", "instruction"),
        record("p3-base", "base", "p3-sib", "a", "forbidden"), record("p3-sib", "sibling", "p3-base", "a", "forbidden"),
    ]
    predictions = [prediction("p1-base", "a"), prediction("p1-sib", "b"),   # 라벨이 바뀌고 모델도 바뀐다
                   prediction("p2-base", "a"), prediction("p2-sib", "a"),   # 라벨은 바뀌었는데 모델은 그대로
                   prediction("p3-base", "a"), prediction("p3-sib", "b")]   # 라벨은 같은데 모델이 바뀐다
    result = contrast_pair_check(predictions, records)
    assert result["instruction"]["pairs"] == 2 and result["instruction"]["label_changed"] == 2
    assert result["instruction"]["sensitivity"] == pytest.approx(0.5) and result["instruction"]["false_change"] is None
    assert result["instruction"]["both_correct"] == pytest.approx(0.5)
    assert result["forbidden"]["label_changed"] == 0 and result["forbidden"]["false_change"] == pytest.approx(1.0)
    assert result["_all"]["pairs"] == 3 and result["_all"]["model_changed"] == 2


def test_holding_twin_preference_counts_which_key_the_model_picks_while_holding():
    """놓기 국면 쌍둥이 키(D1 리뷰 1 I2): 들고 있는 틱에서 grasp 줄과 place 줄 가운데 무엇을 고르는지와, 그 틱들의 라벨 분포."""
    from robo_jev.evaluate import holding_twin_preference

    def tick(label_ids):
        return {
            "t": 0, "sim_ms": 0,
            "request": {
                "state": {"robot": {"holding": "o7"}},
                "exec_history": "",
                "commitment": None,
                "candidates": {"q_main": [
                    {"id": "g", "key": "grasp:o7:top:zoneL"},
                    {"id": "p", "key": "place:o7:release:zoneL"},
                    {"id": "h", "key": "hold"},
                ]},
            },
            "labels": [{"question_id": "q_main", "kind": "valid_set", "candidate_ids": list(label_ids)}],
        }

    record = {"schema_version": "stream-v0", "episode_id": "ep-1", "prefix": {"instructions": []},
              "ticks": [tick(["g"]), tick(["p"]), tick(["g", "p"])]}
    free = {"t": 0, "sim_ms": 0, "request": {"state": {"robot": {"holding": None}}, "exec_history": "", "commitment": None,
                                             "candidates": {"q_main": [{"id": "g", "key": "grasp:o7:top:zoneL"}]}},
            "labels": []}
    record["ticks"].append(free)  # 들고 있지 않은 틱은 세지 않는다

    def prediction(index, best):
        probabilities = {"g": torch.tensor([0.8, 0.1, 0.1]), "p": torch.tensor([0.1, 0.8, 0.1]), "h": torch.tensor([0.1, 0.1, 0.8])}[best]
        return {"record_id": "ep-1", "tick": index, "kind": "stream", "probabilities": {"q_main": probabilities},
                "candidates": {"q_main": ["g", "p", "h"]}, "labels": [], "question_types": {"q_main": "choice"}}

    result = holding_twin_preference([prediction(0, "g"), prediction(1, "g"), prediction(2, "p"), prediction(3, "g")], [record])
    assert result["ticks"] == 3 and result["ticks_with_both_keys"] == 3
    assert (result["predicted_grasp"], result["predicted_place"], result["predicted_other"]) == (2, 1, 0)
    assert result["grasp_share_of_keyed"] == pytest.approx(2 / 3)
    assert (result["label_grasp"], result["label_place"], result["label_mixed"]) == (1, 1, 1)
    assert result["label_grasp_share"] == pytest.approx(0.5)
    assert result["predicted_held_object"] == 3 and result["held_object_share"] == pytest.approx(1.0)


# --------------------------------------------------------------------------
# 편 단위 집계와 부트스트랩 (Task P2 B1·B2·B3)
# --------------------------------------------------------------------------


def test_aggregate_leaves_per_episode_rows_that_add_up_to_the_table(items):
    loaded, tokenizer = items
    judge = Judge.from_config(seed=5, vocab_size=SMALL_VOCAB)
    predictions = predict_items(judge, loaded)
    # 편 = 스트림이면 에피소드 id, 단일이면 origin_group (없으면 record_id)
    for prediction in predictions:
        record = next(item.record for item in loaded if item.record_id == prediction["record_id"])
        expected = prediction["record_id"] if prediction["kind"] == "stream" else str(record.get("origin_group") or prediction["record_id"])
        assert prediction["group"] == expected
    table = aggregate(predictions)
    for key, row in table.items():
        rows = row["per_episode"]
        assert rows, key
        assert [entry["episode_id"] for entry in rows] == sorted(entry["episode_id"] for entry in rows)
        assert sum(entry["n"] for entry in rows) == row["n"], key
        assert sum(entry["graded"] for entry in rows) == row["graded"], key
        if row["accuracy"] is not None:
            assert sum(entry["correct"] for entry in rows) / row["graded"] == pytest.approx(row["accuracy"]), key
        assert {entry["episode_id"] for entry in rows} <= {p["group"] for p in predictions}


def test_the_episode_clustered_interval_is_much_wider_than_treating_ticks_as_independent():
    """편 안의 틱이 완전히 상관된 극단 — P1이 낸 두 한계 가운데 위쪽이 답이 되는 경우."""
    rows = [{"episode_id": f"ep-{index}", "n": 100, "graded": 100, "correct": 100 if index < 4 else 0} for index in range(8)]
    result = episode_bootstrap(rows)
    assert result["unit"] == "episode" and result["episodes"] == 8 and result["graded"] == 800
    assert result["accuracy"] == 0.5 and result["resamples"] == 2000
    naive = 1.96 * math.sqrt(0.25 / 800)  # 틱이 독립이라면 ±0.035
    assert result["accuracy_half_width"] > 5 * naive
    assert result["accuracy_ci"][0] < 0.25 and result["accuracy_ci"][1] > 0.75
    assert "margin" not in result  # 대조군이 없으면 여유 칸도 없다


def test_the_paired_margin_interval_says_whether_a_margin_is_a_finding():
    model = [{"episode_id": f"ep-{i}", "n": 50, "graded": 50, "correct": c} for i, c in enumerate([30, 25, 40, 20, 35, 28, 33, 22])]
    flat = episode_bootstrap(model, [dict(row) for row in model])
    assert flat["margin"] == 0.0 and flat["margin_includes_zero"] and flat["margin_ci"] == [0.0, 0.0]
    # 편마다 정확히 같은 크기로 대조군이 낮다 → 여유는 확실하고 구간의 폭은 0이다 (쌍 부트스트랩이 편 효과를 지운다)
    lifted = episode_bootstrap(model, [{**row, "correct": row["correct"] - 10} for row in model])
    assert lifted["margin"] == pytest.approx(0.2) and not lifted["margin_includes_zero"]
    assert lifted["margin_ci"][0] > 0.0 and lifted["margin_half_width"] < 1e-9
    # 편마다 부호가 엇갈리면 같은 크기의 점추정도 0을 포함한다
    deltas = [12, -10, 9, -8, 11, -9, 10, -12]
    noisy = episode_bootstrap(model, [{**row, "correct": row["correct"] - d} for row, d in zip(model, deltas)])
    assert noisy["margin"] == pytest.approx(sum(deltas) / 400) and noisy["margin"] != 0.0
    assert noisy["margin_includes_zero"] and noisy["margin_ci"][0] < 0.0 < noisy["margin_ci"][1]


def test_the_episode_bootstrap_is_a_deterministic_function_of_its_seed():
    rows = [{"episode_id": f"ep-{i}", "n": 40, "graded": 40, "correct": c} for i, c in enumerate([10, 30, 20, 25, 35, 15])]
    assert episode_bootstrap(rows) == episode_bootstrap(rows)
    assert episode_bootstrap(rows, seed=1)["accuracy_ci"] != episode_bootstrap(rows, seed=2)["accuracy_ci"]
    assert episode_bootstrap([]) is None and episode_bootstrap(None) is None
    assert episode_bootstrap([{"episode_id": "a", "n": 3, "graded": 0, "correct": 0}]) is None


def test_evaluate_items_puts_an_episode_interval_and_a_paired_control_margin_on_the_table(items):
    loaded, tokenizer = items
    judge = Judge.from_config(seed=5, vocab_size=SMALL_VOCAB)
    result = evaluate_items(judge, loaded, tokenizer=tokenizer, shuffle_seed=None, instruction_shuffle=True)
    intervals = result["episode_bootstrap"]
    assert intervals and set(intervals) <= set(result["model"]) and "q_main" in intervals and "_all" in intervals
    cell = intervals["q_main"]
    assert cell["unit"] == "episode" and cell["episodes"] >= 1
    assert cell["accuracy"] == pytest.approx(result["model"]["q_main"]["accuracy"])
    assert cell["accuracy_ci"][0] <= cell["accuracy"] <= cell["accuracy_ci"][1]
    margin = cell["state_shuffle"]
    assert margin["control_accuracy"] == pytest.approx(result["context_shuffle"]["q_main"]["accuracy"])
    assert margin["margin"] == pytest.approx(cell["accuracy"] - margin["control_accuracy"])
    assert margin["margin_ci"][0] <= margin["margin"] <= margin["margin_ci"][1]
    assert isinstance(margin["margin_includes_zero"], bool)
    # 지시 섞기는 로봇 스트림 열이라 비로봇 타입 칸에는 없다
    assert "instruction_shuffle" in cell and "instruction_shuffle" not in intervals["choice"]


def test_a_split_can_ask_for_its_per_tick_predictions_without_moving_the_eval_set_identity(tmp_path):
    """판정 칸처럼 **지정한 분할에만** 틱별 예측을 남긴다 (P2 리뷰 1 I3).

    P1·P2의 산출물은 편 단위 집계까지만 남겨서 "라벨이 지금 commitment가 아닌 틱만 골라 보면 얼마인가" 같은 질문이
    전부 GPU 재실행이었다. 이 열은 **점수를 매긴 모집단을 바꾸지 않으므로** 평가 집합의 해시에 들어가면 안 된다 —
    들어가면 P1의 `79d09793eab5…`와 나란히 놓을 수 없게 된다.
    """
    from robo_jev.evaluate import eval_suite_identity, evaluate_suite, load_eval_suite, load_suite_items

    tokenizer = WhitespaceTokenizer()
    plain = load_eval_suite(_suite_file(tmp_path))
    assert all(entry.get("store_predictions", False) is False for entry in plain["splits"])

    splits = [dict(entry) for entry in plain["splits"]]
    splits[0]["store_predictions"] = ["q_main"]  # 질문 칸 이름 목록 — 이 분할의 질문 칸 열 개를 다 켜면 4.8 MB다
    asked = load_eval_suite(_suite_file(tmp_path, splits=splits))
    assert asked["splits"][0]["store_predictions"] == ["q_main"]

    items = load_suite_items(asked, tokenizer=tokenizer)
    assert eval_suite_identity(asked, items)["sha256"] == eval_suite_identity(plain, load_suite_items(plain, tokenizer=tokenizer))["sha256"]

    judge = Judge.from_config(seed=5, vocab_size=SMALL_VOCAB)
    result = evaluate_suite(judge, asked, tokenizer=tokenizer, items=items)
    stream_table, singles_table = result["splits"]["d0/dev"], result["splits"]["d0/dev_singles"]
    assert all("per_record" not in row for row in singles_table["model"].values())  # 켜지 않은 분할은 그대로다

    for column in ("model", "permuted", "context_shuffle", "instruction_shuffle", "rule_judge"):
        table = stream_table[column]
        assert "per_record" not in table["_all"]  # 질문 칸마다 있으니 합계 칸에 또 두지 않는다
        assert [key for key, row in table.items() if "per_record" in row] == ["q_main"], column  # 고른 칸만
        row = table["q_main"]
        records = row["per_record"]
        assert len(records) == row["n"] == sum(entry["n"] for entry in row["per_episode"]), column
        assert sum(entry["correct"] is True for entry in records) == sum(entry["correct"] for entry in row["per_episode"]), column
        assert {name for entry in records for name in entry} == {"record_id", "tick", "question", "predicted", "correct"}
        assert all(entry["tick"] is not None and entry["question"] == "q_main" for entry in records), column

    # `true`면 모든 질문 칸에 남는다 — 무엇을 켤지는 설정이 고른다
    splits[0]["store_predictions"] = True
    everything = evaluate_suite(judge, load_eval_suite(_suite_file(tmp_path, splits=splits)), tokenizer=tokenizer, items=items)
    assert all("per_record" in row for key, row in everything["splits"]["d0/dev"]["model"].items() if key != "_all")


# --------------------------------------------------------------------------
# P3 — 대조군 하나 더, 상시 기준군 하나 더, 그리고 두 가지 평균
# --------------------------------------------------------------------------


def _commitment_streams(streams, *, hold_at=(1,)):
    """fixture 스트림에 **commitment를 심는다** — D0 fixture는 commitment가 비어 있어 이 층이 아예 없다.

    `hold_at` 틱은 `hold` 후보를 붙잡고 있고(라벨은 `c1`/`c2`이므로 **비-commitment 층**), 나머지 틱은 `c1`을
    붙잡고 있다(라벨 안에 있으므로 **commitment 층**). 곧 한 레코드가 두 층을 모두 낸다."""
    out = []
    for record in copy.deepcopy(list(streams)):
        for index, tick in enumerate(record["ticks"]):
            reference = "c7" if index in hold_at else "c1"
            key = next(c["key"] for c in tick["request"]["candidates"]["q_main"] if c["id"] == reference)
            tick["request"]["commitment"] = {"action_ref": reference, "key": key, "phase": "approach", "held_ticks": 3, "last_switch_tick": 0}
            tick["request"]["state"]["commitment"] = dict(tick["request"]["commitment"])
            tick["request"]["state"]["exec"] = {"seq": index, "action_ref": reference, "phase": "approach"}
            tick["request"]["exec_history"] = f"main={reference} phase=approach path=p0 speed=0 force=0 gripper=open stop=0 ack=ok"
            for label in tick["labels"]:  # 계약: 부가 질문 라벨의 참조는 그 틱의 commitment와 같아야 한다
                if "conditioned_on" in label:
                    label["conditioned_on"] = f"{reference}/approach"
        out.append(record)
    return out


def _stream_items(records, tokenizer, *, window_ticks=30):
    """스트림 레코드들을 그대로 :class:`~robo_jev.sampler.Item` 으로 — 샘플러를 거치지 않는 fixture용 어댑터.

    `_commitment_streams`가 심은 commitment를 그대로 둔 채 열 하나만 부르고 싶을 때 쓴다(`tokens`는 이 시험들이
    배치를 나누지 않으므로 1로 둔다)."""
    return [
        Item(index=index, kind="stream", record_id=record["episode_id"], split="dev", domain="robot", material="stream",
             record=record, layout=serialize_request(record, tokenizer, layout="stream_l1a", window_ticks=window_ticks),
             tokens=1, question_types={"q_main": "choice"})
        for index, record in enumerate(records)
    ]


def test_the_mechanical_baseline_answers_the_commitment_then_the_observe_gate_and_reads_nothing_else(streams):
    """B3 — "commitment가 있으면 그것, 없으면 `observe`"를 상시 열로 (P3 B3).

    이 열은 대조군이 **보존하는 필드만** 읽는다. 어떤 모델 주장도 이 열을 넘지 못하면 주장이 아니므로, 그 값이
    보고서에 늘 있어야 하고 계산이 시험으로 묶여 있어야 한다."""
    from robo_jev.evaluate import MECHANICAL_BASELINE_POLICY, aggregate, mechanical_baseline_predictions

    tokenizer = WhitespaceTokenizer()
    records = _commitment_streams(streams[:2], hold_at=(1,))
    predictions = mechanical_baseline_predictions(_stream_items(records, tokenizer))
    assert predictions and all(set(p["probabilities"]) == {"q_main"} for p in predictions)  # 정의된 질문에만 답한다
    by_tick = {(p["record_id"], p["tick"]): p for p in predictions}
    for record in records:
        for index in range(len(record["ticks"])):
            entry = by_tick[(record["episode_id"], index)]
            ids = entry["candidates"]["q_main"]
            chosen = ids[int(entry["probabilities"]["q_main"].argmax())]
            assert chosen == ("c7" if index == 1 else "c1")  # commitment 그대로
    table = aggregate(predictions)
    ticks = sum(len(record["ticks"]) for record in records)
    # 손으로 센 기대값: 그 틱의 commitment가 허용 집합 안이면 맞다 (정책이 읽는 것은 그 줄뿐이다)
    expected = sum(
        int(("c7" if index == 1 else "c1") in next(label["candidate_ids"] for label in tick["labels"] if label["question_id"] == "q_main"))
        for record in records
        for index, tick in enumerate(record["ticks"])
    )
    assert table["q_main"]["n"] == ticks
    assert table["q_main"]["accuracy"] == pytest.approx(expected / ticks) and 0 < expected < ticks
    assert "commitment" in MECHANICAL_BASELINE_POLICY and "observe" in MECHANICAL_BASELINE_POLICY

    # commitment를 지우면 `observe` 후보(c6)로 물러난다 — 그리고 라벨이 c1/c2이므로 전부 틀린다
    blind = copy.deepcopy(records)
    for record in blind:
        for tick in record["ticks"]:
            tick["request"]["commitment"] = None
            tick["request"]["state"].pop("commitment", None)
            tick["request"]["state"].pop("exec", None)
            tick["labels"] = [label for label in tick["labels"] if "conditioned_on" not in label]  # 계약: commitment 없으면 부가 라벨도 없다
    fallback = mechanical_baseline_predictions(_stream_items(blind, tokenizer))
    assert all(entry["candidates"]["q_main"][int(entry["probabilities"]["q_main"].argmax())] == "c6" for entry in fallback)
    blind_expected = sum(
        int("c6" in next(label["candidate_ids"] for label in tick["labels"] if label["question_id"] == "q_main"))
        for record in blind for tick in record["ticks"]
    )
    assert aggregate(fallback)["q_main"]["accuracy"] == pytest.approx(blind_expected / ticks)


def test_rolling_the_commitment_keeps_the_reference_inside_this_tick_s_own_candidate_list(streams):
    """B2 — commitment를 굴리되 **이 틱의 다른 후보로** 굴린다 (P3 B2).

    기증 틱의 참조를 그대로 실으면 그 id가 이 틱의 후보 목록에 없다(실측 16.9 %만 들어맞는다). 하네스는 commitment의
    후보 자리를 예약하므로 그런 입력은 계약 밖이고, 그때 대조군이 낮은 것은 읽지 못해서가 아니라 본 적 없는 입력이기
    때문이 된다. 그래서 자리만 굴린다 — 참조는 언제나 이 틱의 실제 후보이고, 언제나 원래와 다르다."""
    from robo_jev.evaluate import context_shuffle_records

    records = _commitment_streams(streams[:3], hold_at=(1,))
    rolled = context_shuffle_records(records, robot="state_commitment")
    assert len(rolled) == len(records)
    moved = 0
    for before, after in zip(records, rolled):
        for own, new in zip(before["ticks"], after["ticks"]):
            ids = [c["id"] for c in new["request"]["candidates"]["q_main"]]
            assert [c["id"] for c in own["request"]["candidates"]["q_main"]] == ids  # 후보 줄은 그대로다
            reference = new["request"]["commitment"]["action_ref"]
            assert reference in ids                                    # (i) 내부 정합성
            assert reference != own["request"]["commitment"]["action_ref"]  # 언제나 굴렸다
            assert new["request"]["state"]["commitment"]["action_ref"] == reference
            assert new["request"]["state"]["exec"]["action_ref"] == reference
            assert new["request"]["exec_history"].split(" ")[0] == f"main={reference}"   # 실행 이력도 같은 참조
            assert new["request"]["exec_history"].split(" ")[1:] == own["request"]["exec_history"].split(" ")[1:]
            assert new["request"]["commitment"]["key"] == next(c["key"] for c in new["request"]["candidates"]["q_main"] if c["id"] == reference)
            moved += 1
    assert moved == sum(len(record["ticks"]) for record in records)

    # 표준 상태 섞기가 하던 일은 그대로 한다 — 목표가 기증 편의 것으로 바뀐다
    plain = context_shuffle_records(records, robot="state")
    assert rolled[0]["ticks"][0]["request"]["state"]["goal"] == plain[0]["ticks"][0]["request"]["state"]["goal"]
    assert plain[0]["ticks"][0]["request"]["commitment"]["action_ref"] == records[0]["ticks"][0]["request"]["commitment"]["action_ref"]

    # commitment가 없는 틱은 만들어 주지 않는다 — 없는 것은 굴릴 것이 없다
    empty = copy.deepcopy(records)
    for record in empty:
        for tick in record["ticks"]:
            tick["request"]["commitment"] = {}
            tick["request"]["state"]["commitment"] = {}
    untouched = context_shuffle_records(empty, robot="state_commitment")
    assert all(tick["request"]["commitment"] == {} for record in untouched for tick in record["ticks"])

    with pytest.raises(ValueError, match="state_commitment"):
        context_shuffle_records(records, robot="commitment")

    # **굴린 레코드는 계약 안에 남는다** — 이 열의 논거가 "굴린 참조는 언제나 이 틱의 실제 후보"이므로, 그 말이
    # 참이면 굴린 레코드가 직렬화 검사를 그대로 통과해야 한다. 부가 라벨의 `conditioned_on`을 같이 옮기는 이유이기도
    # 하다(옮기지 않으면 여기서 막힌다). 리뷰 1 M5 — 보고서 B2의 주장을 시험이 들고 있게 한다.
    from robo_jev.contracts import validate_record

    for record in rolled:
        validate_record(record)
    for record in plain:
        validate_record(record)


def test_the_tick_weighted_and_the_episode_balanced_mean_are_both_reported_and_can_disagree():
    """A3 — 편마다 크기가 다르면 두 평균이 갈린다. 갈리는 것 자체가 결과의 일부라 **둘 다** 낸다 (P3 A3)."""
    from robo_jev.evaluate import episode_bootstrap

    model = [{"episode_id": "big", "n": 300, "graded": 300, "correct": 300},
             {"episode_id": "s1", "n": 20, "graded": 20, "correct": 0},
             {"episode_id": "s2", "n": 20, "graded": 20, "correct": 0}]
    control = [{"episode_id": "big", "n": 300, "graded": 300, "correct": 150},
               {"episode_id": "s1", "n": 20, "graded": 20, "correct": 10},
               {"episode_id": "s2", "n": 20, "graded": 20, "correct": 10}]
    out = episode_bootstrap(model, control)
    assert out["accuracy"] == pytest.approx(300 / 340)               # 틱 가중: 큰 편이 끈다
    assert out["episode_balanced_accuracy"] == pytest.approx(1 / 3)  # 편 균등: 한 편이 한 표
    assert out["margin"] == pytest.approx(300 / 340 - 170 / 340)
    assert out["episode_balanced_margin"] == pytest.approx(1 / 3 - 0.5)
    for key in ("accuracy_ci", "episode_balanced_accuracy_ci", "margin_ci", "episode_balanced_margin_ci"):
        low, high = out[key]
        assert low <= high
    assert out["margin"] > 0 > out["episode_balanced_margin"]  # **부호가 갈린다** — 그 사실이 결과의 일부다
    assert isinstance(out["episode_balanced_margin_includes_zero"], bool)

    # 편 크기가 같으면 두 평균이 같다
    same = [{"episode_id": name, "n": 10, "graded": 10, "correct": value} for name, value in (("a", 3), ("b", 7))]
    both = episode_bootstrap(same)
    assert both["accuracy"] == both["episode_balanced_accuracy"] == pytest.approx(0.5)


def test_the_two_new_columns_run_and_do_not_move_the_eval_set_identity(tmp_path):
    """P3의 두 열은 **무엇을 더 재는지**를 바꿀 뿐 무엇을 점수 매기는지를 바꾸지 않는다 — `store_predictions`와 같다.

    그래서 켜고 끈 두 설정의 해시가 같아야 하고(같지 않으면 P3의 run들끼리도 나란히 놓을 수 없다), 옛 설정의 해시는
    이 열들이 생기기 전과 같아야 한다(P1의 `79d09793eab5…`)."""
    from robo_jev.evaluate import eval_suite_identity, evaluate_suite, load_eval_suite, load_suite_items

    tokenizer = WhitespaceTokenizer()
    plain = load_eval_suite(_suite_file(tmp_path))
    assert plain["columns"]["commitment_shuffle"] is False and plain["columns"]["mechanical_baseline"] is False  # 기본은 꺼짐

    columns = {**plain["columns"], "commitment_shuffle": True, "mechanical_baseline": True}
    asked = load_eval_suite(_suite_file(tmp_path, columns=columns))
    items = load_suite_items(asked, tokenizer=tokenizer)
    assert eval_suite_identity(asked, items)["sha256"] == eval_suite_identity(plain, load_suite_items(plain, tokenizer=tokenizer))["sha256"]

    judge = Judge.from_config(seed=5, vocab_size=SMALL_VOCAB)
    table = evaluate_suite(judge, asked, tokenizer=tokenizer, items=items)["splits"]["d0/dev"]
    assert table["commitment_shuffle_kind"] == "state_commitment" and "mechanical_baseline_policy" in table
    assert table["commitment_shuffle"]["_all"]["n"] == table["context_shuffle"]["_all"]["n"]
    assert table["mechanical_baseline"]["q_main"]["n"] == table["model"]["q_main"]["n"]
    intervals = table["episode_bootstrap"]["q_main"]
    assert set(intervals["commitment_shuffle"]) == {"control_accuracy", "margin", "margin_ci", "margin_half_width", "margin_includes_zero"}
    assert "episode_balanced_accuracy" in intervals

    # 스트림이 없는 분할에는 두 열이 없다
    assert "commitment_shuffle" not in table.get("d0/dev_singles", {})


def test_the_commitment_shuffle_column_carries_its_scope_and_no_margin_where_it_cannot_be_read():
    """리뷰 1 I3 — 이 열은 `q_main`에서만 읽을 수 있는데 여유와 판정 표지가 **모든 질문 칸에** 나갔다.

    부가 질문의 답은 **원래** commitment에 조건화된 전문가 답이라, 참조를 옮기면 대조군이 목표를 못 읽어서 틀리는
    것이 아니라 **답 자체가 뒤집힌다**. 그런 칸의 `margin_includes_zero: false`는 기계가 읽는 거짓 발견이고, 그것이
    P2-I2가 일어난 방식이다. 그래서 범위를 값 옆에 적고 읽을 수 없는 칸에서는 여유를 내지 않는다."""
    from robo_jev.evaluate import COMMITMENT_SHUFFLE_SCOPE, split_episode_bootstrap

    rows = [{"episode_id": name, "n": 10, "graded": 10, "correct": value} for name, value in (("a", 9), ("b", 8))]
    worse = [{"episode_id": name, "n": 10, "graded": 10, "correct": value} for name, value in (("a", 2), ("b", 1))]
    table = {
        "model": {"q_main": {"per_episode": rows}, "q_done": {"per_episode": rows}, "_all": {"per_episode": rows}},
        "context_shuffle": {qid: {"per_episode": worse} for qid in ("q_main", "q_done", "_all")},
        "commitment_shuffle": {qid: {"per_episode": worse} for qid in ("q_main", "q_done", "_all")},
    }
    out = split_episode_bootstrap(table)

    main = out["q_main"]["commitment_shuffle"]
    assert main["margin"] > 0 and main["margin_includes_zero"] is False and "out_of_scope" not in main
    for qid in ("q_done", "_all"):
        entry = out[qid]["commitment_shuffle"]
        assert entry["control_accuracy"] == pytest.approx(0.15)     # 정확도는 남는다 — 재지 않은 것이 아니다
        assert entry["out_of_scope"] == COMMITMENT_SHUFFLE_SCOPE    # 왜 못 읽는지가 값 옆에 있다
        assert entry["margin"] is None and entry["margin_ci"] is None and entry["margin_includes_zero"] is None
        # 같은 칸의 **표준 대조군**은 그대로 판정한다 — 범위는 이 열만의 것이다
        assert out[qid]["state_shuffle"]["margin_includes_zero"] is False
    assert "q_main" in COMMITMENT_SHUFFLE_SCOPE and "falsified" in COMMITMENT_SHUFFLE_SCOPE


# --------------------------------------------------------------------------
# Task R1 Stage C — 사건을 재는 지표, 지시 대조군의 승격, 기증자 고정의 제거
# --------------------------------------------------------------------------


def _per_record(rows):
    """`aggregate(..., store_predictions=…)`의 `per_record` 꼴로 (record_id, tick, question, predicted)."""
    return [{"record_id": rid, "tick": tick, "question": "q_main", "predicted": pid, "correct": None} for rid, tick, pid in rows]


def _toy_record(labels, *, events=None, episode_id="ep-toy"):
    """라벨과 사건만 있는 최소 스트림 레코드 — 새 지표는 예측과 라벨·사건만 읽는다."""
    ticks = []
    for index, answer in enumerate(labels):
        state = {"goal": {"version": 1 + sum(1 for e in (events or {}) if e <= index and (events or {})[e] == "goal"), "text": "t"}}
        state["events"] = [{"kind": "object_moved", "sim_ms": index * 100}] if (events or {}).get(index) == "world" else []
        ticks.append({
            "t": index, "sim_ms": index * 100, "request": {"state": state, "candidates": {"q_main": []}},
            "labels": [{"question_id": "q_main", "kind": "valid_set", "candidate_ids": list(answer)}],
        })  # fmt: skip
    return {"schema_version": "stream-v0", "episode_id": episode_id, "prefix": {"instructions": [{"version": 1, "t_ms": 0, "text": "t"}]}, "ticks": ticks}


def test_reaction_delay_counts_the_ticks_to_the_new_answer_and_censors_what_never_arrives():
    """C3-a: 사건 틱마다 모델의 argmax가 **그 틱의 새 정답 집합**에 처음 드는 데 걸린 틱 수. 상한 안에 들지 못하면
    값을 지어내지 않고 검열로 센다 — 상한을 평균에 섞으면 "느리게 반응함"과 "반응하지 않음"이 같은 수가 된다."""
    from robo_jev.evaluate import reaction_delay

    record = _toy_record([["a"]] * 3 + [["b"]] * 5 + [["c"]] * 4, events={3: "goal", 8: "world"})
    quick = _per_record([("ep-toy", i, "a") for i in range(3)] + [("ep-toy", i, "b") for i in range(3, 8)] + [("ep-toy", i, "c") for i in range(8, 12)])
    out = reaction_delay(quick, [record])
    assert out["goal_change"]["events"] == 1 and out["goal_change"]["median_ticks"] == 0 and out["goal_change"]["censored"] == 0
    assert out["world_event"]["events"] == 1 and out["world_event"]["median_ticks"] == 0

    slow = _per_record([("ep-toy", i, "a") for i in range(3)] + [("ep-toy", i, "a") for i in range(3, 6)] + [("ep-toy", i, "b") for i in range(6, 8)] + [("ep-toy", i, "b") for i in range(8, 12)])
    out = reaction_delay(slow, [record])
    assert out["goal_change"]["median_ticks"] == 3  # 세 틱 늦게 새 답으로 갔다
    assert out["world_event"]["censored"] == 1 and out["world_event"]["censored_rate"] == 1.0
    assert out["horizon_ticks"] == 30


def test_answer_stability_reads_only_the_stretches_where_the_label_did_not_move():
    """C3-b: 사건이 없는데 답이 흔들리는 것은 반응이 아니라 잡음이다. 라벨이 그대로인 구간에서만 센다."""
    from robo_jev.evaluate import answer_stability

    record = _toy_record([["a"]] * 6)
    steady = answer_stability(_per_record([("ep-toy", i, "a") for i in range(6)]), [record])
    assert steady["switch_rate"] == 0.0 and steady["round_trips"] == 0 and steady["hold_ticks_median"] == 6

    flapping = answer_stability(_per_record([("ep-toy", i, "a" if i % 2 == 0 else "b") for i in range(6)]), [record])
    assert flapping["switch_rate"] == 1.0 and flapping["round_trips"] == 4 and flapping["hold_ticks_median"] == 1
    assert flapping["segments"] == 1 and flapping["steps"] == 5


def test_stop_timing_reports_the_delay_after_a_stop_and_the_false_alarms_when_there_is_none():
    """C3-c: 정지가 필요해진 틱부터 `q_stop ≥ 0.5`(참 후보가 argmax)까지의 틱 수와, 정지가 필요 없는 틱의 오경보율."""
    from robo_jev.evaluate import stop_timing

    record = _toy_record([["a"]] * 6)
    for index, tick in enumerate(record["ticks"]):
        tick["labels"].append({"question_id": "q_stop", "kind": "single", "answer": index in (2, 3)})
    rows = [{"record_id": "ep-toy", "tick": index, "question": "q_stop", "predicted": value, "correct": None}
            for index, value in enumerate(["false", "true", "false", "true", "false", "false"])]
    out = stop_timing(rows, [record])
    assert out["onsets"] == 1 and out["median_ticks"] == 1 and out["censored"] == 0
    assert out["quiet_ticks"] == 4 and out["false_alarm_rate"] == 0.25  # 틱 1의 참이 오경보


def test_the_instruction_shuffle_replaces_every_place_the_model_reads_the_instruction(streams):
    """C2: `goal` 줄에서 구조화된 목표가 빠졌으므로 **지시 섞기가 "지시를 읽는가"를 재는 열**이다. 그러려면 모델이
    지시 문장을 보는 **세 자리**(prefix 조각, 주기적 `goal … text=`, `ev instruction_changed text=`)가 전부 바뀌어야
    한다 — 하나라도 남으면 이 열이 진짜 지시를 흘린다."""
    episodes = [copy.deepcopy(record) for record in streams[:2]]
    # D0 fixture의 두 에피소드는 지시 문장이 같다 — 섞기가 실제로 바꾸는지 보려면 달라야 한다.
    for number, episode in enumerate(episodes):
        text = f"지시 {number}: 대상을 영역 {number}로 옮겨라"
        for instruction in episode["prefix"]["instructions"]:
            instruction["text"] = text
        for tick in episode["ticks"]:
            state = tick["request"]["state"]
            state["goal"]["text"] = text
            state["events"] = [{"kind": "instruction_changed", "sim_ms": int(tick.get("sim_ms", 0)), "version": int(state["goal"].get("version", 1)), "text": text}]
    rolled = context_shuffle_records(episodes, robot="instruction")
    own_text = episodes[0]["prefix"]["instructions"][0]["text"]
    donor_text = episodes[1]["prefix"]["instructions"][0]["text"]
    assert own_text != donor_text
    tokenizer = WhitespaceTokenizer()
    text = tokenizer.decode(serialize_request(rolled[0], tokenizer, layout="stream_l1a")["tokens"])
    assert donor_text in text and own_text not in text
    for record in rolled:
        validate_record(record)


def test_the_state_shuffle_no_longer_freezes_a_tick_on_a_finished_donor_scene(streams):
    """C2 (P3 C1b 이월): 옛 규칙은 기증자가 짧으면 그 뒤의 틱을 전부 기증자의 **마지막(끝난) 상태** 하나에 묶었고,
    그래서 이 열의 값이 설정의 편 순서에 달려 있었다. 이제 회전을 길이로 짝짓고 남는 차이는 감아 돈다."""
    from robo_jev.evaluate import donor_rotation

    episodes = [copy.deepcopy(record) for record in streams[:3]]
    for length, episode in zip((2, 5, 9), episodes):
        episode["ticks"] = (episode["ticks"] * 5)[:length]
    rows = donor_rotation(episodes)
    assert [row["ticks"] for row in rows] == [2, 5, 9]  # 길이 순으로 짝짓는다
    assert [row["donor_ticks"] for row in rows] == [5, 9, 2]
    rolled = context_shuffle_records(episodes, robot="state")
    longest = next(record for record in rolled if len(record["ticks"]) == 9)
    donor = next(record for record in episodes if len(record["ticks"]) == 2)
    rolled_goals = [tick["request"]["state"]["goal"] for tick in longest["ticks"]]
    donor_goals = [tick["request"]["state"]["goal"] for tick in donor["ticks"]]
    # 감아 돌았으므로 기증자의 두 틱이 번갈아 나온다 — 한 상태에 갇히지 않는다.
    assert rolled_goals[0]["text"] == donor_goals[0]["text"] and rolled_goals[1]["text"] == donor_goals[1]["text"]
    assert rolled_goals[2]["text"] == donor_goals[0]["text"]


def test_event_metrics_are_computed_for_every_column_that_has_per_tick_predictions():
    """C3: 같은 자를 **모든 열**에 댄다 — 규칙 판정기·기계적 기준군 옆에 서지 않으면 "빠르다"가 뜻이 없다."""
    from robo_jev.evaluate import column_event_metrics

    record = _toy_record([["a"]] * 4 + [["b"]] * 4, events={4: "goal"})
    for tick in record["ticks"]:
        tick["labels"].append({"question_id": "q_stop", "kind": "single", "answer": False})
    rows = _per_record([("ep-toy", i, "a" if i < 4 else "b") for i in range(8)])
    stop_rows = [{"record_id": "ep-toy", "tick": i, "question": "q_stop", "predicted": "false", "correct": None} for i in range(8)]
    table = {
        "model": {"q_main": {"per_record": rows}, "q_stop": {"per_record": stop_rows}},
        "mechanical_baseline": {"q_main": {"per_record": _per_record([("ep-toy", i, "a") for i in range(8)])}},
        "rule_judge": {"q_main": {}},  # per_record가 없는 열은 건너뛴다
    }
    out = column_event_metrics(table, [record])
    assert set(out) == {"model", "mechanical_baseline"}
    assert out["model"]["reaction_delay"]["goal_change"]["median_ticks"] == 0
    assert out["mechanical_baseline"]["reaction_delay"]["goal_change"]["censored"] == 1  # 늘 `a`라 새 답에 못 든다
    assert out["model"]["stability"]["switch_rate"] == 0.0
    assert out["model"]["stop_timing"]["false_alarm_rate"] == 0.0
    assert "stop_timing" not in out["mechanical_baseline"]
