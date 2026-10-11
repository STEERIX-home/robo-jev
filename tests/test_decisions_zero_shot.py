"""`robo_jev.decisions.zero_shot` — backbone에서 코드 확률 읽기 (Task v0c §2.2).

CPU 검사는 소형 난수 Qwen3.5(torch 참조 kernel)로 배관을 본다: 기존 Nimble 채점(`robo_jev.model.zero_shot.score_prompts`)과
같은 확률, 확률 합 1, 코드 질량, 배치 크기와 무관, 레코드마다 예측 하나(질문마다 균형 배치 모두), reasoning(설명 생성 → ``Answer:`` →
같은 코드 읽기), 왼쪽 padding 배치 생성 == 하나씩 생성. 설명을 자르는 규칙은 모델 없이 검사한다. 가짜 backbone(글에 보이는 줄을 읽어
고른 코드를 미는 것)으로 글 → 코드 → 키·Yes/No·구간의 대응을 끝까지 본다(리뷰 1 I5).

GPU 검사(`real_2b` 표지 — 실제 Qwen3.5-2B 가중치와 CUDA가 있을 때만; 전체 suite는 GPU 유닛 안에서 돈다)는 fixture 요청을
끝까지 채점해 확률 합이 1이고 같은 입력 두 번이 같은 값인지 본다. 적재 전에 울타리 0.6과 여유 20 GiB를 건다.
"""

import functools
from types import SimpleNamespace

import pytest
import torch
from helpers import FIXTURES

from robo_jev.decisions import prompt as P
from robo_jev.decisions import zero_shot as Z
from robo_jev.decisions.records import read_records
from robo_jev.model.backbone import describe_backbone
from robo_jev.model.tokenizer import available_tokenizer, load_tokenizer

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
    """실제 2B + CUDA가 있어야 하는 검사 — 없으면 skip, `real_2b` 표지 (tests/test_backbone_qwen.py와 같은 방식)."""
    marked = pytest.mark.real_2b(func)
    return pytest.mark.skipif(not real_weights_present(), reason="Qwen3.5-2B 가중치(artifacts/models)와 CUDA가 있어야 한다")(marked)


@pytest.fixture(autouse=True)
def _torch_kernels(request):
    """CPU 검사는 공식 forward를 torch 참조 kernel로 (CUDA가 있는 venv에서 fla가 CPU tensor를 받지 않게); `real_2b`는 공식 kernel."""
    if request.node.get_closest_marker("real_2b") is not None:
        yield
        return
    with torch_reference_kernels():
        yield


@functools.lru_cache(maxsize=1)
def tokenizer():
    return load_tokenizer(available_tokenizer()[0])


@functools.lru_cache(maxsize=1)
def tiny() -> QwenBackbone:
    return QwenBackbone.tiny(seed=4, vocab_size=tokenizer().get_vocab_size())


@pytest.fixture(scope="module")
def fixtures() -> list[dict]:
    return read_records(REQUESTS)


# -- 생각 자르기 (모델 없음) -------------------------------------------------------------------------------------------


def _ids(text: str) -> list[int]:
    return tokenizer().encode(text, add_special_tokens=False).ids


def _reasoning_prompt(fmt: str) -> dict:
    request = {"state": "s", "questions": {"q": {"type": "noul", "instructions": "Will it spill?"}}}
    return P.build_prompts(request, tokenizer(), key="k", seed=None, fmt=fmt, mode="reasoning")[0]


def test_finishing_an_explanation_stops_at_the_end_token_and_appends_the_answer_cue():
    prompt = _reasoning_prompt("plain")
    generated = _ids(" The cup is full, so braking hard spills it.") + [Z.special_id(tokenizer(), "<|im_end|>")] + _ids(" junk")
    done = Z.finish_reasoning(prompt, generated, tokenizer(), cap=150)
    assert done["stop"] == "eos" and done["reasoning"] == " The cup is full, so braking hard spills it."
    assert done["generated_tokens"] == len(_ids(" The cup is full, so braking hard spills it.")) + 1
    assert done["tokens"][: len(prompt["tokens"])] == prompt["tokens"]
    assert tokenizer().decode(done["tokens"], skip_special_tokens=False).endswith("Explanation: The cup is full, so braking hard spills it.\nAnswer:")
    assert done["self_answer"] is False and done["complete"] is True  # 끝 토큰으로 스스로 끝냈다


def test_a_self_written_answer_with_a_code_letter_is_cut_so_the_codes_are_read_at_our_answer_cue():
    prompt = _reasoning_prompt("plain")
    generated = _ids(" Hard braking tilts the surface past the rim.\nAnswer: A because")
    done = Z.finish_reasoning(prompt, generated, tokenizer(), cap=len(generated))
    assert done["stop"] == "cap" and done["self_answer"] is True and done["complete"] is True
    assert done["reasoning"] == " Hard braking tilts the surface past the rim."
    text = tokenizer().decode(done["tokens"], skip_special_tokens=False)
    assert text.endswith("rim.\nAnswer:") and "because" not in text
    bold = Z.finish_reasoning(prompt, _ids(" It spills. **Answer:** B"), tokenizer(), cap=150)
    assert bold["self_answer"] is True and bold["reasoning"] == " It spills."


def test_an_answer_cue_without_a_valid_code_letter_does_not_cut_the_explanation():
    """리뷰 1 Minor 6: 설명 속의 'answer:'(뒤에 그 질문의 코드 글자가 없는 것)는 자르지 않는다 — 상한에 닿으면 미완료다."""
    prompt = _reasoning_prompt("plain")  # noul: 코드 A·B
    for body in (" The answer: depends on the fill level, which", " Answer: Z is not an option here", " so the answer: a careful stop"):
        generated = _ids(body)
        done = Z.finish_reasoning(prompt, generated, tokenizer(), cap=len(generated))
        assert done["self_answer"] is False and done["complete"] is False and done["stop"] == "cap", body
        assert done["reasoning"] == body.rstrip()


def test_chat_explanations_stop_at_the_turn_end_and_reopen_the_answer_slot():
    prompt = _reasoning_prompt("chat")
    generated = _ids("The bowl is shallow, so it spills. Answer: B") + [Z.special_id(tokenizer(), "<|im_end|>")]
    done = Z.finish_reasoning(prompt, generated, tokenizer(), cap=150)
    assert done["stop"] == "eos" and done["self_answer"] is True and done["complete"] is True
    assert done["reasoning"] == "The bowl is shallow, so it spills."
    assert tokenizer().decode(done["tokens"], skip_special_tokens=False).endswith("<think>\n\n</think>\n\nThe bowl is shallow, so it spills.\nAnswer:")


def test_an_empty_generation_still_leaves_a_readable_answer_slot():
    for fmt in P.FORMATS:
        prompt = _reasoning_prompt(fmt)
        done = Z.finish_reasoning(prompt, [Z.special_id(tokenizer(), "<|endoftext|>")], tokenizer(), cap=150)
        assert done["reasoning"] == "" and done["stop"] == "eos"
        assert tokenizer().decode(done["tokens"], skip_special_tokens=False).endswith(P.reasoning_close(fmt))


# -- 소형 난수 Qwen (CPU) ------------------------------------------------------------------------------------------------


def test_code_probabilities_match_the_existing_nimble_scorer_on_the_tiny_qwen(fixtures):
    from robo_jev.model.zero_shot import score_prompts

    prompts = [p for r in fixtures[:3] for p in P.record_prompts(r, tokenizer(), seed=1)]
    ours = Z.score_code_prompts(tiny(), prompts, batch=4)
    theirs = score_prompts(tiny(), prompts, batch=4)
    assert len(ours) == len(theirs) == len(prompts)
    for a, b in zip(ours, theirs):
        assert torch.allclose(torch.tensor(a["code_probs"]), b["probabilities"].float(), atol=1e-5)


def test_probabilities_sum_to_one_and_code_mass_is_the_share_of_the_full_softmax(fixtures):
    prompts = [p for p in P.record_prompts(fixtures[0], tokenizer(), seed=1)]
    scored = Z.score_code_prompts(tiny(), prompts, batch=8)
    for p, s in zip(prompts, scored):
        assert sum(s["code_probs"]) == pytest.approx(1.0, abs=1e-6) and 0.0 < s["code_mass"] <= 1.0
        with torch.no_grad():
            hidden = tiny()(torch.tensor([p["tokens"]]), torch.arange(len(p["tokens"]))[None])["hidden"][0, -1].float()
            full = torch.softmax(tiny().model.lm_head.weight.float() @ hidden, 0)
        assert s["code_mass"] == pytest.approx(float(full[p["code_ids"]].sum()), rel=1e-4)
        assert s["code_probs"] == pytest.approx((full[p["code_ids"]] / full[p["code_ids"]].sum()).tolist(), abs=1e-5)


def test_scoring_does_not_depend_on_the_batch_size_or_the_order(fixtures):
    prompts = [p for r in fixtures[:2] for p in P.record_prompts(r, tokenizer(), seed=1)]
    one = Z.score_code_prompts(tiny(), prompts, batch=1)
    many = Z.score_code_prompts(tiny(), prompts, batch=5)
    backwards = Z.score_code_prompts(tiny(), prompts[::-1], batch=3)[::-1]
    for a, b, c in zip(one, many, backwards):
        assert a["code_probs"] == pytest.approx(b["code_probs"], abs=1e-5)
        assert a["code_probs"] == pytest.approx(c["code_probs"], abs=1e-5)


@pytest.mark.parametrize("fmt", P.FORMATS)
def test_diagnose_returns_one_prediction_per_record_with_every_listing_of_every_question(fixtures, fmt):
    from robo_jev.decisions.metrics import average_answers

    preds = Z.diagnose_records(fixtures, tiny(), tokenizer(), model_id="tiny", mode="direct", fmt=fmt, seed=1, batch=6)
    assert [p["id"] for p in preds] == [r["id"] for r in fixtures]
    for record, pred in zip(fixtures, preds):
        assert set(pred["questions"]) == set(record["request"]["questions"])
        assert (pred["model"], pred["mode"], pred["format"], pred["prompt_version"], pred["orders"], pred["shuffle_seed"]) == (
            "tiny", "direct", fmt, P.PROMPT_VERSION, "counterbalanced", 1)
        for name, q in pred["questions"].items():
            question = record["request"]["questions"][name]
            expected = len(question["criteria"]) if question["type"] == "choice" else 2
            assert q["type"] == question["type"] and q["n_listings"] == len(q["listings"]) == expected
            assert len({tuple(l["options"]) for l in q["listings"]}) == expected  # 서로 다른 배치
            for listing in q["listings"]:
                assert sum(listing["code_probs"]) == pytest.approx(1.0, abs=1e-6) and listing["prompt_tokens"] > 0 and "reasoning" not in listing
                if q["type"] == "choice":
                    assert set(listing["answer"]["probabilities"]) == set(question["criteria"])
                elif q["type"] == "noul":
                    assert 0.0 <= listing["answer"]["p_yes"] <= 1.0
                else:
                    assert 0.05 - 1e-9 <= listing["answer"]["value"] <= 0.95 + 1e-9
            assert q["answer"] == average_answers(q["type"], [l["answer"] for l in q["listings"]])


def test_seeded_orders_score_one_listing_per_question(fixtures):
    preds = Z.diagnose_records(fixtures[:2], tiny(), tokenizer(), model_id="tiny", orders="seeded", seed=1)
    assert all(q["n_listings"] == 1 and q["answer"] == q["listings"][0]["answer"] for p in preds for q in p["questions"].values())


def test_reasoning_mode_generates_then_reads_the_same_codes_after_the_answer_cue(fixtures):
    preds = Z.diagnose_records(
        fixtures[:2], tiny(), tokenizer(), model_id="tiny", mode="reasoning", fmt="plain", seed=1, orders="seeded", batch=4, gen_batch=3, max_new_tokens=5
    )
    for pred in preds:
        for q in pred["questions"].values():
            (listing,) = q["listings"]
            r = listing["reasoning"]
            assert 1 <= r["generated_tokens"] <= 5 and r["stop"] in ("eos", "cap", "think_end", "end") and isinstance(r["text"], str)
            assert isinstance(r["complete"], bool) and isinstance(r["self_answer"], bool)
            assert sum(listing["code_probs"]) == pytest.approx(1.0, abs=1e-6) and listing["scored_tokens"] > listing["prompt_tokens"]


def test_a_prompt_whose_lines_do_not_match_its_record_is_refused_before_scoring(fixtures, monkeypatch):
    real = P.record_prompts

    def desynced(*args, **kwargs):
        prompts = real(*args, **kwargs)
        for p in prompts:
            if p["type"] == "choice":
                p["options"] = list(reversed(p["options"]))
        return prompts

    monkeypatch.setattr(Z, "record_prompts", desynced)
    with pytest.raises(ValueError, match="어긋난"):
        Z.diagnose_records(fixtures[:1], tiny(), tokenizer(), model_id="tiny")


class _StubBackbone:
    """가짜 backbone: 프롬프트 글의 마지막 ``Options:`` 줄들을 읽어 ``target(type, 줄들)``이 고른 줄의 코드 글자 쪽으로 hidden을 민다.
    LM head는 각 대문자 코드(" A"…" Z")의 행에 단위 벡터 하나씩 — 그래서 확률은 고른 코드에 몰린다. 글 → 코드 → 키의 대응을 기록된
    ``options``와 따로 끝까지 본다(배치마다 순서가 달라도 같은 키·Yes·구간이 나와야 한다)."""

    def __init__(self, target):
        self.target = target
        vocab = tokenizer().get_vocab_size()
        weight = torch.zeros(vocab, 26)
        for i, letter in enumerate("ABCDEFGHIJKLMNOPQRSTUVWXYZ"):
            weight[tokenizer().encode(" " + letter, add_special_tokens=False).ids[0], i] = 1.0
        self.model = SimpleNamespace(lm_head=SimpleNamespace(weight=weight))
        self.device = torch.device("cpu")

    def __call__(self, ids, positions):
        hidden = torch.zeros(ids.shape[0], ids.shape[1], 26)
        for row in range(ids.shape[0]):
            text = tokenizer().decode(ids[row].tolist(), skip_special_tokens=False)
            block = text[text.rfind("\nOptions:\n") + len("\nOptions:\n"):]
            lines = []
            for raw in block.split("\n"):
                line = raw.split("<|im_end|>")[0]
                if len(line) < 3 or line[1:3] != ". ":
                    break
                lines.append(line)
            letter = self.target(lines)[0]
            hidden[row, :, "ABCDEFGHIJKLMNOPQRSTUVWXYZ".index(letter)] = 30.0
        return {"hidden": hidden}


def _favour(lines: list[str]) -> str:
    """고르는 규칙(위치와 무관): Yes 줄, "0.7 to 0.9" 줄(p2)·"0.8" 줄(p3), 그 밖(choice)이면 키가 사전순으로 가장 큰 줄."""
    for line in lines:
        if line[3:] in ("Yes", "0.7 to 0.9", "0.8"):
            return line
    return max(lines, key=lambda line: line[3:].split(":")[0])


@pytest.mark.parametrize("version", P.PROMPT_VERSIONS)
@pytest.mark.parametrize("fmt", P.FORMATS)
def test_the_code_printed_next_to_a_key_maps_back_to_that_key_in_every_listing(fixtures, fmt, version):
    preds = Z.diagnose_records(fixtures, _StubBackbone(_favour), tokenizer(), model_id="stub", fmt=fmt, seed=2, version=version)
    assert {p["prompt_version"] for p in preds} == {version}
    for record, pred in zip(fixtures, preds):
        for name, q in pred["questions"].items():
            question = record["request"]["questions"][name]
            for listing in q["listings"]:
                if q["type"] == "choice":
                    probs = listing["answer"]["probabilities"]
                    assert max(probs, key=probs.get) == max(question["criteria"]) and probs[max(question["criteria"])] > 0.99
                elif q["type"] == "noul":
                    assert listing["answer"]["p_yes"] > 0.99
                else:
                    assert listing["answer"]["value"] == pytest.approx(0.8, abs=1e-3)


def test_left_padded_batch_generation_matches_one_at_a_time_on_the_tiny_qwen(fixtures):
    prompts = [p for r in fixtures[:2] for p in P.record_prompts(r, tokenizer(), seed=1, mode="reasoning")][:5]
    batched = Z.generate_reasoning(tiny(), prompts, tokenizer(), max_new_tokens=6, batch=5)
    single = Z.generate_reasoning(tiny(), prompts, tokenizer(), max_new_tokens=6, batch=1)
    assert [g["generated"] for g in batched] == [g["generated"] for g in single]
    assert all(len(g["generated"]) <= 6 for g in batched)


def test_the_default_diagnosis_reads_each_prompt_alone_bit_for_bit(fixtures):
    """기본 채점은 프롬프트를 하나씩 읽는다 — 진단 결과가 프롬프트마다 혼자 채점한 값과 **비트 단위로** 같다. 같은 소형 모델에서도
    배치 5는 비트가 다르므로(padding·묶음이 계산을 바꾼다; 실제 2B bf16에서는 0.02까지·argmax 33개 중 3~4개, 보고서) 이 검사는 기본값이
    바뀌면 떨어진다(리뷰 1 Minor 1 — 옛 검사는 배치 1끼리 비교했다)."""
    prompts = [p for r in fixtures[:2] for p in P.record_prompts(r, tokenizer(), seed=1, orders="counterbalanced")]
    alone = [Z.score_code_prompts(tiny(), [p])[0]["code_probs"] for p in prompts]
    preds = Z.diagnose_records(fixtures[:2], tiny(), tokenizer(), model_id="tiny", seed=1)
    produced = [l["code_probs"] for pred in preds for q in pred["questions"].values() for l in q["listings"]]
    assert produced == alone
    batched = [s["code_probs"] for s in Z.score_code_prompts(tiny(), prompts, batch=5)]
    assert batched != alone  # 검사의 이빨: 배치로 읽으면 비트가 달라진다


def test_required_free_memory_is_declared_for_both_candidate_models():
    assert Z.MIN_FREE_BYTES["Qwen/Qwen3.5-2B"] == 20 * 2**30 and Z.MIN_FREE_BYTES["Qwen/Qwen3.5-4B"] == 30 * 2**30


# -- 실제 Qwen3.5-2B (GPU) ---------------------------------------------------------------------------------------------


@functools.lru_cache(maxsize=1)
def real_2b() -> QwenBackbone:
    from robo_jev.gpu import limit_gpu_memory, require_free

    limit_gpu_memory(0.6)
    require_free(Z.MIN_FREE_BYTES[REAL_2B], what="decisions real_2b test load")
    return QwenBackbone.load(REAL_2B)


@needs_real_2b
@pytest.mark.parametrize("fmt", P.FORMATS)
def test_real_2b_direct_scoring_of_the_fixture_requests_sums_to_one_and_is_deterministic(fixtures, fmt):
    first = Z.diagnose_records(fixtures, real_2b(), tokenizer(), model_id=REAL_2B, mode="direct", fmt=fmt, seed=1)
    second = Z.diagnose_records(fixtures, real_2b(), tokenizer(), model_id=REAL_2B, mode="direct", fmt=fmt, seed=1)
    assert [p["id"] for p in first] == [r["id"] for r in fixtures]
    for a, b in zip(first, second):
        assert set(a["questions"]) == set(b["questions"])
        for name, q in a["questions"].items():
            assert q["n_listings"] >= 2
            for la, lb in zip(q["listings"], b["questions"][name]["listings"]):
                assert sum(la["code_probs"]) == pytest.approx(1.0, abs=1e-5) and 0.0 < la["code_mass"] <= 1.0 + 1e-6
                assert la["code_probs"] == lb["code_probs"] and la["code_mass"] == lb["code_mass"] and la["answer"] == lb["answer"]
            assert q["answer"] == b["questions"][name]["answer"]


@needs_real_2b
def test_real_2b_explanation_first_generates_with_the_official_kernels_and_reads_a_distribution(fixtures):
    kwargs = dict(model_id=REAL_2B, mode="reasoning", fmt="chat", seed=1, orders="seeded", gen_batch=8)
    first = Z.diagnose_records(fixtures[:2], real_2b(), tokenizer(), **kwargs)
    second = Z.diagnose_records(fixtures[:2], real_2b(), tokenizer(), **kwargs)
    for a, b in zip(first, second):
        for name, q in a["questions"].items():
            (la,), (lb,) = q["listings"], b["questions"][name]["listings"]
            assert 1 <= la["reasoning"]["generated_tokens"] <= Z.REASONING_MAX_NEW_TOKENS and la["reasoning"]["text"].strip()
            assert isinstance(la["reasoning"]["complete"], bool) and sum(la["code_probs"]) == pytest.approx(1.0, abs=1e-5)
            assert la["reasoning"] == lb["reasoning"] and la["code_probs"] == lb["code_probs"]
