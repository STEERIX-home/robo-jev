"""`robo_jev.decisions.prompt` — 판단 요청 → 질문마다 프롬프트 하나 (Task v0c §2.1).

검사하는 것: 코드가 답 경계에서 한 토큰이다; 선택지 섞기가 seed로 재현되고 확률이 원래 키로 돌아온다; noul(예/아니오 코드 둘,
순서도 섞음)·score(6구간 코드 → 구간 중앙값의 기대값) 변환이 맞다; 같은 상태 글이 질문마다 바이트(토큰) 단위로 같다;
프롬프트는 ``request``만으로 만든다(숨은 값 ``truth``·``meta``·``labels``·``rationale``는 들어가지 않는다); chat 형식은
Qwen3.5의 공식 chat template이 그리는 글과 같다; 고정 문구는 버전과 함께 고정돼 있다.

Fix round 1(리뷰 1 I5·통제자 결정): **글에 보이는 선택지 줄을 다시 읽어** 기록된 키·Yes/No·구간 중앙값과 맞는지 본다(키·구간이
어긋난 변이체는 떨어진다); 결정적 균형 배치(choice는 순환 이동으로 각 선택지가 각 자리에 한 번, noul·score는 정순·역순);
v0c-p2의 "짧게 보이는 설명 뒤 답"(생각 블록 없이)."""

import copy
import functools
import hashlib
import json
from collections import Counter, defaultdict

import pytest
import torch
from helpers import FIXTURES

from robo_jev.decisions import prompt as P
from robo_jev.decisions.records import read_records, shuffle_key
from robo_jev.model.tokenizer import available_tokenizer, load_tokenizer
from robo_jev.model.zero_shot import verify_codes

REQUESTS = FIXTURES / "decisions" / "requests.jsonl"


@functools.lru_cache(maxsize=1)
def tokenizer():
    return load_tokenizer(available_tokenizer()[0])


@pytest.fixture(scope="module")
def fixtures() -> list[dict]:
    return read_records(REQUESTS)


def _all_prompts(records, **kw):
    return [p for record in records for p in P.record_prompts(record, tokenizer(), **kw)]


# -- 고정 문구와 버전 ------------------------------------------------------------------------------------------------


def test_the_fixed_wording_is_pinned_to_its_version():
    """문구를 바꾸면 버전도 올려야 한다 — 고정 문구 전부의 해시가 버전과 함께 고정돼 있다."""
    assert P.PROMPT_VERSION == "v0c-p2"
    assert P.fixed_text_digest() == "3d075458260069584d91d5d400b9a3eb1ace1748294d9ab6ea65907424c4e6ff"


def test_score_bins_cover_zero_to_one_and_their_centers_are_the_midpoints():
    edges = [lo for lo, _ in P.SCORE_BINS] + [P.SCORE_BINS[-1][1]]
    assert edges == [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
    assert P.SCORE_CENTERS == pytest.approx((0.05, 0.2, 0.4, 0.6, 0.8, 0.95))


# -- 코드 ------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("fmt", P.FORMATS)
@pytest.mark.parametrize("mode", P.MODES)
def test_every_code_is_one_token_at_the_answer_boundary(fixtures, fmt, mode):
    prompts = _all_prompts(fixtures, seed=1, fmt=fmt, mode=mode)
    assert len(prompts) == sum(len(r["request"]["questions"]) for r in fixtures)
    for p in prompts:
        codes, ids = verify_codes(tokenizer(), len(p["options"]))
        assert p["codes"] == codes and p["code_ids"] == ids
        assert all(len(tokenizer().encode(" " + c, add_special_tokens=False).ids) == 1 for c in codes)
    eight = next(p for p in prompts if len(p["options"]) == 8)
    assert eight["codes"] == list("ABCDEFGH")


@pytest.mark.parametrize("fmt", P.FORMATS)
def test_a_direct_prompt_ends_at_the_answer_cue_so_the_next_token_is_the_code(fixtures, fmt):
    answer_ids = tokenizer().encode("Answer:", add_special_tokens=False).ids
    for p in _all_prompts(fixtures, seed=1, fmt=fmt, mode="direct"):
        assert p["tokens"][-len(answer_ids):] == answer_ids and p["text"].endswith("Answer:")
        for code, line in zip(p["codes"], p["option_lines"]):
            assert f"\n{code}. {line}\n" in p["text"] or f"\n{code}. {line}<|im_end|>" in p["text"]


# -- 섞기와 되돌리기 ---------------------------------------------------------------------------------------------------


def test_the_option_order_is_reproducible_from_the_seed_and_differs_across_seeds(fixtures):
    a = _all_prompts(fixtures, seed=1, fmt="plain", mode="direct")
    b = _all_prompts(fixtures, seed=1, fmt="plain", mode="direct")
    c = _all_prompts(fixtures, seed=2, fmt="plain", mode="direct")
    assert [p["options"] for p in a] == [p["options"] for p in b] and [p["tokens"] for p in a] == [p["tokens"] for p in b]
    assert [p["options"] for p in a] != [p["options"] for p in c]
    unshuffled = _all_prompts(fixtures, seed=None, fmt="plain", mode="direct")
    first = fixtures[0]["request"]["questions"]
    assert unshuffled[0]["options"] == list(first["action"]["criteria"])  # seed None → 요청의 순서 그대로
    for p in a:
        if p["type"] == "choice":
            assert sorted(p["options"]) == sorted(next(r for r in fixtures if r["id"] == p["record_id"])["request"]["questions"][p["question"]]["criteria"])


def test_the_shuffle_breaks_the_link_between_position_and_option_across_records():
    """같은 질문 이름·같은 키라도 레코드마다 순서가 다르다 (열쇠가 레코드 id)."""
    request = {"state": "s", "questions": {"q": {"type": "choice", "instructions": "?", "criteria": {k: k for k in "abcdef"}}}}
    orders = {tuple(P.build_prompts(request, tokenizer(), key=f"rec{i}", seed=1)[0]["options"]) for i in range(30)}
    assert len(orders) > 20


def test_choice_probabilities_return_to_the_original_keys(fixtures):
    for p in _all_prompts(fixtures, seed=3, fmt="plain", mode="direct"):
        if p["type"] != "choice":
            continue
        for j, key in enumerate(p["options"]):
            onehot = torch.zeros(len(p["options"]))
            onehot[j] = 1.0
            assert P.answer_from_codes(p, onehot)["probabilities"][key] == 1.0
        probs = torch.softmax(torch.arange(len(p["options"]), dtype=torch.float32), 0)
        mapped = P.answer_from_codes(p, probs)["probabilities"]
        assert set(mapped) == set(p["options"]) and sum(mapped.values()) == pytest.approx(1.0)
        assert [mapped[key] for key in p["options"]] == pytest.approx(probs.tolist())


def test_noul_uses_two_shuffled_codes_and_reads_p_yes_from_the_yes_code(fixtures):
    nouls = [p for p in _all_prompts(fixtures, seed=1, fmt="plain", mode="direct") if p["type"] == "noul"]
    assert all(sorted(p["options"]) == ["no", "yes"] and len(p["codes"]) == 2 for p in nouls)
    assert {tuple(p["options"]) for p in nouls} == {("yes", "no"), ("no", "yes")}  # 순서도 섞인다
    for p in nouls:
        yes = p["options"].index("yes")
        probs = torch.tensor([0.0, 0.0])
        probs[yes], probs[1 - yes] = 0.7, 0.3
        assert P.answer_from_codes(p, probs)["p_yes"] == pytest.approx(0.7)
        assert f"{p['codes'][yes]}. Yes" in p["text"] and f"{p['codes'][1 - yes]}. No" in p["text"]


def test_score_reads_the_expected_bin_center_whichever_direction_the_bins_are_listed(fixtures):
    scores = [p for p in _all_prompts(fixtures, seed=1, fmt="plain", mode="direct") if p["type"] == "score"]
    assert {tuple(p["values"]) == tuple(sorted(p["values"])) for p in scores} == {True, False}  # 오름·내림 둘 다 나온다
    for p in scores:
        assert sorted(p["values"]) == pytest.approx(list(P.SCORE_CENTERS)) and len(p["codes"]) == 6
        top = p["values"].index(max(p["values"]))
        onehot = torch.zeros(6)
        onehot[top] = 1.0
        assert P.answer_from_codes(p, onehot)["value"] == pytest.approx(0.95)
        mixed = torch.zeros(6)
        mixed[p["values"].index(0.05)], mixed[p["values"].index(0.6)] = 0.5, 0.5
        answer = P.answer_from_codes(p, mixed)
        assert answer["value"] == pytest.approx(0.325) and sum(answer["bins"].values()) == pytest.approx(1.0)
        assert answer["bins"]["0.0-0.1"] == pytest.approx(0.5)
        # 지시와 0·1 기준 글이 프롬프트에 그대로 있다
        record = next(r for r in fixtures if r["id"] == p["record_id"])
        for line in record["request"]["questions"][p["question"]]["criteria"]:
            assert line in p["text"]


def test_ascending_and_descending_listings_give_the_same_value_for_the_same_bin_mass():
    request = {"state": "s", "questions": {"risk": {"type": "score", "instructions": "How risky?", "criteria": ["0 = none", "1 = certain"]}}}
    seen = {}
    for i in range(40):
        p = P.build_prompts(request, tokenizer(), key=f"k{i}", seed=1)[0]
        mass = {0.05: 0.1, 0.2: 0.2, 0.4: 0.3, 0.6: 0.2, 0.8: 0.1, 0.95: 0.1}
        probs = torch.tensor([mass[v] for v in p["values"]])
        seen[p["values"][0] < p["values"][-1]] = P.answer_from_codes(p, probs)["value"]
    assert set(seen) == {True, False} and seen[True] == pytest.approx(seen[False]) == pytest.approx(0.46)


# -- 상태 글 ---------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("fmt", P.FORMATS)
@pytest.mark.parametrize("mode", P.MODES)
def test_the_state_text_is_byte_identical_across_the_questions_of_a_record(fixtures, fmt, mode):
    for record in fixtures:
        prompts = P.record_prompts(record, tokenizer(), seed=1, fmt=fmt, mode=mode)
        prefixes = {tuple(p["tokens"][: p["prefix_tokens"]]) for p in prompts}
        texts = {p["text"][: p["prefix_chars"]] for p in prompts}
        assert len(prefixes) == 1 and len(texts) == 1
        (prefix_text,) = texts
        assert record["request"]["state"].strip() in prefix_text
        assert tokenizer().decode(list(next(iter(prefixes))), skip_special_tokens=False) == prefix_text
        # 질문 글은 접두 뒤에만 있다
        assert all(record["request"]["questions"][p["question"]]["instructions"] not in prefix_text for p in prompts)


def test_direct_and_reasoning_share_the_prefix(fixtures):
    for fmt in P.FORMATS:
        d = P.record_prompts(fixtures[0], tokenizer(), seed=1, fmt=fmt, mode="direct")
        r = P.record_prompts(fixtures[0], tokenizer(), seed=1, fmt=fmt, mode="reasoning")
        assert [p["tokens"][: p["prefix_tokens"]] for p in d] == [p["tokens"][: p["prefix_tokens"]] for p in r]
        assert [p["options"] for p in d] == [p["options"] for p in r]


# -- request만 -------------------------------------------------------------------------------------------------------


def _poisoned(record: dict) -> dict:
    out = copy.deepcopy(record)
    out["truth"] = {"world": {"intent": "POISON_TRUTH_INTENT", "slosh_k": 9.87654}, "realized": {"x": "POISON_REALIZED"}}
    out["rationale"] = "POISON_RATIONALE"
    out["labels"] = {name: {"poison": "POISON_LABEL"} for name in out["labels"]}
    pair = out["meta"].get("pair")
    out["meta"] = {"scene": "POISON_SCENE", "recommendation": {"present": True, "option": "POISON_REC", "wrong": True}, "pair": pair}
    out["generator"] = {"version": "POISON_GENERATOR"}
    return out


@pytest.mark.parametrize("fmt", P.FORMATS)
@pytest.mark.parametrize("mode", P.MODES)
def test_prompts_are_built_from_the_request_alone(fixtures, fmt, mode):
    """숨은 값을 모두 바꿔도 프롬프트가 토큰 하나까지 같고, 바꾼 값은 어느 글에도 나오지 않는다."""
    for record in fixtures:
        clean = P.record_prompts(record, tokenizer(), seed=1, fmt=fmt, mode=mode)
        dirty = P.record_prompts(_poisoned(record), tokenizer(), seed=1, fmt=fmt, mode=mode)
        assert [p["tokens"] for p in clean] == [p["tokens"] for p in dirty]
        assert [p["text"] for p in clean] == [p["text"] for p in dirty]
        for p in dirty:
            assert "POISON" not in p["text"] and "9.87654" not in p["text"]
            assert "POISON" not in tokenizer().decode(p["tokens"], skip_special_tokens=False)
        bare = {"id": record["id"], "meta": {"pair": record["meta"].get("pair")}, "request": copy.deepcopy(record["request"])}
        assert [p["tokens"] for p in P.record_prompts(bare, tokenizer(), seed=1, fmt=fmt, mode=mode)] == [p["tokens"] for p in clean]


def test_build_prompts_reads_only_the_request_keys_it_needs(fixtures):
    """`build_prompts`는 요청 dict만 받는다 — 레코드를 넘기면 거절한다(숨은 값이 섞일 길을 막는다)."""
    with pytest.raises(ValueError):
        P.build_prompts(fixtures[0], tokenizer(), key="x", seed=1)
    with pytest.raises(ValueError):
        P.build_prompts({**fixtures[0]["request"], "labels": {}}, tokenizer(), key="x", seed=1)


def test_the_pair_members_see_their_options_in_the_same_order(fixtures):
    by_id = {r["id"]: r for r in fixtures}
    orders = {rid: [p["options"] for p in P.record_prompts(by_id[rid], tokenizer(), seed=5)] for rid in ("fx-001", "fx-002", "fx-003")}
    assert orders["fx-001"] == orders["fx-002"] == orders["fx-003"]
    assert shuffle_key(by_id["fx-002"]) == "fx-001"


# -- chat 형식 -------------------------------------------------------------------------------------------------------


def test_the_chat_rendering_matches_the_official_qwen35_template():
    """기대 글은 Qwen/Qwen3.5-2B(revision 15852e8)의 `chat_template.jinja`를 jinja2로 그린 것(system + user, add_generation_prompt,
    enable_thinking 기본 = 생각 없음 — 4B는 enable_thinking=False로 같은 글) 뒤에, direct면 우리 답 자리를 붙인 것이다. direct의
    글은 v0c-p1과 바이트 단위로 같다(이 기대 글이 p1 때와 같다); reasoning(p2)은 같은 생각 없는 렌더에서 생성을 시작한다."""
    request = {"state": "STATE LINE", "questions": {"q": {"type": "noul", "instructions": "Will it spill?"}}}
    direct = P.build_prompts(request, tokenizer(), key="k", seed=None, fmt="chat", mode="direct")[0]["text"]
    reasoning = P.build_prompts(request, tokenizer(), key="k", seed=None, fmt="chat", mode="reasoning")[0]["text"]
    official_direct = (
        "<|im_start|>system\n" + P.SYSTEM_TEXT + "<|im_end|>\n<|im_start|>user\nSituation:\nSTATE LINE\n\nQuestion: Will it spill?\n"
        "Options:\nA. Yes\nB. No<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    )
    official_reasoning = (  # v0c-p2: 생각 블록은 비우고(공식 기본 = 생각 없음) 보이는 설명을 쓰게 한다
        "<|im_start|>system\n" + P.SYSTEM_TEXT + "<|im_end|>\n<|im_start|>user\nSituation:\nSTATE LINE\n\nQuestion: Will it spill?\n"
        "Options:\nA. Yes\nB. No\n" + P.EXPLAIN_TEXT + "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    )
    assert direct == official_direct + "Answer:"
    assert reasoning == official_reasoning
    special = tokenizer().encode("<|im_start|>", add_special_tokens=False).ids
    ids = P.build_prompts(request, tokenizer(), key="k", seed=None, fmt="chat", mode="direct")[0]["tokens"]
    assert ids[: len(special)] == special and len(special) == 1


def test_the_plain_rendering_is_the_documented_layout():
    request = {"state": "STATE LINE", "questions": {"q": {"type": "choice", "instructions": "Do what?", "criteria": {"go": "Keep going.", "stop": "Stop now."}}}}
    direct = P.build_prompts(request, tokenizer(), key="k", seed=None, fmt="plain", mode="direct")[0]["text"]
    assert direct == P.SYSTEM_TEXT + "\n\nSituation:\nSTATE LINE\n\nQuestion: Do what?\nOptions:\nA. go: Keep going.\nB. stop: Stop now.\nAnswer:"
    reasoning = P.build_prompts(request, tokenizer(), key="k", seed=None, fmt="plain", mode="reasoning")[0]["text"]
    assert reasoning == P.SYSTEM_TEXT + "\n\nSituation:\nSTATE LINE\n\nQuestion: Do what?\nOptions:\nA. go: Keep going.\nB. stop: Stop now.\n" + P.EXPLAIN_TEXT + "\nExplanation:"
    assert P.reasoning_close("plain") == "\nAnswer:" and P.reasoning_close("chat") == "\nAnswer:"


def test_option_descriptions_are_kept_on_one_line():
    request = {"state": "s", "questions": {"q": {"type": "choice", "instructions": "Pick\n one.", "criteria": {"a": "two\nlines  here", "b": ""}}}}
    text = P.build_prompts(request, tokenizer(), key="k", seed=None)[0]["text"]
    assert "A. a: two lines here\nB. b\n" in text and "Question: Pick one.\n" in text


def test_a_record_with_an_unknown_question_type_is_refused():
    request = {"state": "s", "questions": {"q": {"type": "ranking", "instructions": "?"}}}
    with pytest.raises(ValueError):
        P.build_prompts(request, tokenizer(), key="k", seed=1)


def test_prompt_metadata_names_the_record_question_and_format(fixtures):
    p = P.record_prompts(fixtures[3], tokenizer(), seed=1, fmt="chat", mode="reasoning")[0]
    assert (p["record_id"], p["question"], p["format"], p["mode"], p["version"]) == ("fx-004", "decision", "chat", "reasoning", "v0c-p2")
    assert (p["listing"], p["n_listings"]) == (0, 1)
    assert json.loads(json.dumps({k: v for k, v in p.items() if k != "tokens"})) is not None
    assert hashlib.sha256(p["text"].encode()).hexdigest() == hashlib.sha256(tokenizer().decode(p["tokens"], skip_special_tokens=False).encode()).hexdigest()


# -- Fix round 1: 보이는 줄 대 기록된 키·값 (리뷰 1 I5) ----------------------------------------------------------------


def _mutants_fail(p: dict) -> None:
    """한 프롬프트의 기록을 어긋나게 바꾼 변이체가 모두 검사에 걸리는지 (검사의 이빨)."""
    if len(p["options"]) >= 2:
        swapped = copy.deepcopy(p)
        swapped["options"][0], swapped["options"][1] = swapped["options"][1], swapped["options"][0]
        assert P.check_displayed_lines(swapped), ("options swap passed", p["question"])
    if p["type"] == "score":
        reversed_values = copy.deepcopy(p)
        reversed_values["values"] = list(reversed(reversed_values["values"]))
        assert P.check_displayed_lines(reversed_values), ("values reversal passed", p["question"])
        renamed = copy.deepcopy(p)
        renamed["options"] = list(reversed(renamed["options"]))
        assert P.check_displayed_lines(renamed), ("bin-name reversal passed", p["question"])
    lines = p["text"].split("\n")
    coded = [i for i, line in enumerate(lines) if len(line) > 2 and line[1:3] == ". " and line[0] in p["codes"]]
    if len(coded) >= 2:  # 글 쪽에서 두 줄의 내용만 맞바꾼 변이체(코드 글자는 그대로)
        a, b = coded[-2], coded[-1]
        texts = copy.deepcopy(p)
        moved = lines[:]
        moved[a], moved[b] = lines[a][:3] + lines[b][3:], lines[b][:3] + lines[a][3:]
        texts["text"] = "\n".join(moved)
        if texts["text"] != p["text"]:
            assert P.check_displayed_lines(texts), ("text swap passed", p["question"])


@pytest.mark.parametrize("fmt", P.FORMATS)
@pytest.mark.parametrize("mode", P.MODES)
@pytest.mark.parametrize("orders", P.ORDERS)
def test_every_displayed_line_matches_the_recorded_key_yes_no_or_bin(fixtures, fmt, mode, orders):
    """글에서 ``X. …`` 줄을 다시 읽어: choice는 줄 j가 ``options[j] + ":"``로 시작하고, noul은 options[j]에 맞는 Yes/No이고,
    score는 중앙값이 values[j]이고 이름이 options[j]인 구간이다 — 세 종류, 두 형식, 두 방식, 두 배치, seed 넷."""
    checked = Counter()
    for seed in (1, 2, 3, 4):
        for p in _all_prompts(fixtures, seed=seed, fmt=fmt, mode=mode, orders=orders):
            assert P.check_displayed_lines(p) == [], (p["record_id"], p["question"], p["listing"])
            checked[p["type"]] += 1
    assert set(checked) == {"choice", "noul", "score"}


def test_desynced_keys_bins_or_lines_fail_the_line_check(fixtures):
    """리뷰 1 I5: 키·구간·줄을 어긋나게 한 변이체는 떨어진다 (옛 시험은 같은 목록을 정답으로 써서 이것을 못 잡았다)."""
    for fmt in P.FORMATS:
        for p in _all_prompts(fixtures, seed=3, fmt=fmt, orders="counterbalanced"):
            _mutants_fail(p)


# -- Fix round 1: 결정적 균형 배치 (통제자 결정) -----------------------------------------------------------------------


def test_counterbalanced_listings_put_every_option_once_in_every_position(fixtures):
    for record in fixtures:
        prompts = P.record_prompts(record, tokenizer(), seed=1, orders="counterbalanced")
        seeded = {p["question"]: p for p in P.record_prompts(record, tokenizer(), seed=1, orders="seeded")}
        by_question = defaultdict(list)
        for p in prompts:
            by_question[p["question"]].append(p)
        for name, listings in by_question.items():
            question = record["request"]["questions"][name]
            listings.sort(key=lambda p: p["listing"])
            assert [p["listing"] for p in listings] == list(range(len(listings))) and all(p["n_listings"] == len(listings) for p in listings)
            assert listings[0]["options"] == seeded[name]["options"]  # 0번 배치 = seed 하나로 섞은 옛 배치
            if question["type"] == "choice":
                n = len(question["criteria"])
                assert len(listings) == n
                for position in range(n):  # 라틴 방진: 각 자리에 각 선택지가 정확히 한 번
                    assert sorted(p["options"][position] for p in listings) == sorted(question["criteria"])
            elif question["type"] == "noul":
                assert {tuple(p["options"]) for p in listings} == {("yes", "no"), ("no", "yes")}
            else:
                assert len(listings) == 2 and listings[1]["values"] == list(reversed(listings[0]["values"]))
                assert {p["values"][0] < p["values"][-1] for p in listings} == {True, False}


def test_counterbalanced_listings_share_the_prefix_and_are_reproducible(fixtures):
    a = P.record_prompts(fixtures[0], tokenizer(), seed=1, orders="counterbalanced")
    b = P.record_prompts(fixtures[0], tokenizer(), seed=1, orders="counterbalanced")
    assert [p["tokens"] for p in a] == [p["tokens"] for p in b]
    assert len({tuple(p["tokens"][: p["prefix_tokens"]]) for p in a}) == 1
    with pytest.raises(ValueError):
        P.record_prompts(fixtures[0], tokenizer(), seed=1, orders="shuffled-twice")


def test_the_answer_cue_appears_once_in_a_direct_prompt_and_not_in_an_explanation_prompt(fixtures):
    for p in _all_prompts(fixtures, seed=1, fmt="chat", mode="direct"):
        assert p["text"][p["prefix_chars"]:].count("Answer:") == 1
    for p in _all_prompts(fixtures, seed=1, fmt="chat", mode="reasoning"):
        assert p["text"].endswith("<think>\n\n</think>\n\n") and P.EXPLAIN_TEXT in p["text"]
