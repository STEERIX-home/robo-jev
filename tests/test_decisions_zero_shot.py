"""`robo_jev.decisions.zero_shot` — backbone에서 코드 확률 읽기 (Task v0c §2.2).

CPU 검사는 소형 난수 Qwen3.5(torch 참조 kernel)로 배관을 본다: 기존 Nimble 채점(`robo_jev.model.zero_shot.score_prompts`)과
같은 확률, 확률 합 1, 코드 질량, 배치 크기와 무관, 레코드마다 예측 하나, reasoning(생성 → ``Answer:`` → 같은 코드 읽기),
왼쪽 padding 배치 생성 == 하나씩 생성. 생각을 자르는 규칙은 모델 없이 검사한다.

GPU 검사(`real_2b` 표지 — 실제 Qwen3.5-2B 가중치와 CUDA가 있을 때만; 전체 suite는 GPU 유닛 안에서 돈다)는 fixture 요청을
끝까지 채점해 확률 합이 1이고 같은 입력 두 번이 같은 값인지 본다. 적재 전에 울타리 0.6과 여유 20 GiB를 건다.
"""

import functools

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


def test_finishing_a_reasoning_stops_at_the_end_token_and_appends_the_answer_cue():
    prompt = _reasoning_prompt("plain")
    generated = _ids(" The cup is full, so braking hard spills it.") + [Z.special_id(tokenizer(), "<|im_end|>")] + _ids(" junk")
    done = Z.finish_reasoning(prompt, generated, tokenizer(), cap=256)
    assert done["stop"] == "eos" and done["reasoning"] == " The cup is full, so braking hard spills it."
    assert done["generated_tokens"] == len(_ids(" The cup is full, so braking hard spills it.")) + 1
    assert done["tokens"][: len(prompt["tokens"])] == prompt["tokens"]
    assert tokenizer().decode(done["tokens"], skip_special_tokens=False).endswith("Reasoning: The cup is full, so braking hard spills it.\nAnswer:")
    assert done["self_answer"] is False


def test_a_self_written_answer_is_cut_so_the_codes_are_read_at_our_answer_cue():
    prompt = _reasoning_prompt("plain")
    generated = _ids(" Hard braking tilts the surface past the rim.\nAnswer: A because")
    done = Z.finish_reasoning(prompt, generated, tokenizer(), cap=len(generated))
    assert done["stop"] == "cap" and done["self_answer"] is True
    assert done["reasoning"] == " Hard braking tilts the surface past the rim."
    text = tokenizer().decode(done["tokens"], skip_special_tokens=False)
    assert text.endswith("rim.\nAnswer:") and "because" not in text


def test_chat_reasoning_stops_at_the_think_close_and_reopens_the_answer_slot():
    prompt = _reasoning_prompt("chat")
    generated = _ids("The bowl is shallow.") + [Z.special_id(tokenizer(), "</think>")] + _ids("\n\nB")
    done = Z.finish_reasoning(prompt, generated, tokenizer(), cap=256)
    assert done["stop"] == "think_end" and done["reasoning"] == "The bowl is shallow."
    assert tokenizer().decode(done["tokens"], skip_special_tokens=False).endswith("<think>\nThe bowl is shallow.\n</think>\n\nAnswer:")


def test_an_empty_generation_still_leaves_a_readable_answer_slot():
    for fmt in P.FORMATS:
        prompt = _reasoning_prompt(fmt)
        done = Z.finish_reasoning(prompt, [Z.special_id(tokenizer(), "<|endoftext|>")], tokenizer(), cap=256)
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
def test_diagnose_returns_one_prediction_per_record_with_every_question_answered(fixtures, fmt):
    preds = Z.diagnose_records(fixtures, tiny(), tokenizer(), model_id="tiny", mode="direct", fmt=fmt, seed=1, batch=6)
    assert [p["id"] for p in preds] == [r["id"] for r in fixtures]
    for record, pred in zip(fixtures, preds):
        assert set(pred["questions"]) == set(record["request"]["questions"])
        assert (pred["model"], pred["mode"], pred["format"], pred["prompt_version"], pred["shuffle_seed"]) == ("tiny", "direct", fmt, P.PROMPT_VERSION, 1)
        for name, q in pred["questions"].items():
            assert q["type"] == record["request"]["questions"][name]["type"] and sum(q["code_probs"]) == pytest.approx(1.0, abs=1e-6)
            if q["type"] == "choice":
                assert set(q["answer"]["probabilities"]) == set(record["request"]["questions"][name]["criteria"])
            elif q["type"] == "noul":
                assert 0.0 <= q["answer"]["p_yes"] <= 1.0
            else:
                assert 0.05 - 1e-9 <= q["answer"]["value"] <= 0.95 + 1e-9
            assert "reasoning" not in q and q["prompt_tokens"] > 0


def test_reasoning_mode_generates_then_reads_the_same_codes_after_the_answer_cue(fixtures):
    preds = Z.diagnose_records(fixtures[:2], tiny(), tokenizer(), model_id="tiny", mode="reasoning", fmt="plain", seed=1, batch=4, gen_batch=3, max_new_tokens=5)
    for pred in preds:
        for q in pred["questions"].values():
            r = q["reasoning"]
            assert 1 <= r["generated_tokens"] <= 5 and r["stop"] in ("eos", "cap", "think_end") and isinstance(r["text"], str)
            assert sum(q["code_probs"]) == pytest.approx(1.0, abs=1e-6) and q["scored_tokens"] > q["prompt_tokens"]


def test_left_padded_batch_generation_matches_one_at_a_time_on_the_tiny_qwen(fixtures):
    prompts = [p for r in fixtures[:2] for p in P.record_prompts(r, tokenizer(), seed=1, mode="reasoning")][:5]
    batched = Z.generate_reasoning(tiny(), prompts, tokenizer(), max_new_tokens=6, batch=5)
    single = Z.generate_reasoning(tiny(), prompts, tokenizer(), max_new_tokens=6, batch=1)
    assert [g["generated"] for g in batched] == [g["generated"] for g in single]
    assert all(len(g["generated"]) <= 6 for g in batched)


def test_the_default_reading_is_one_prompt_at_a_time_so_it_does_not_depend_on_the_other_prompts(fixtures):
    """기본 배치는 1이다 — padding이 없어 한 프롬프트의 확률이 같이 묶인 다른 프롬프트에 기대지 않는다(실제 2B bf16에서 배치 구성이
    코드 확률을 0.02까지 움직이고 argmax를 33개 중 3~4개 뒤집었다; 보고서)."""
    assert Z.DEFAULT_BATCH == 1
    prompts = [p for r in fixtures[:2] for p in P.record_prompts(r, tokenizer(), seed=1)]
    alone = [Z.score_code_prompts(tiny(), [p])[0]["code_probs"] for p in prompts]
    together = [s["code_probs"] for s in Z.score_code_prompts(tiny(), prompts)]
    assert alone == together


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
            assert sum(q["code_probs"]) == pytest.approx(1.0, abs=1e-5) and 0.0 < q["code_mass"] <= 1.0 + 1e-6
            assert q["code_probs"] == b["questions"][name]["code_probs"] and q["code_mass"] == b["questions"][name]["code_mass"]
            assert q["answer"] == b["questions"][name]["answer"]


@needs_real_2b
def test_real_2b_reasoning_generates_with_the_official_kernels_and_reads_a_distribution(fixtures):
    first = Z.diagnose_records(fixtures[:2], real_2b(), tokenizer(), model_id=REAL_2B, mode="reasoning", fmt="chat", seed=1, batch=8, gen_batch=8, max_new_tokens=24)
    second = Z.diagnose_records(fixtures[:2], real_2b(), tokenizer(), model_id=REAL_2B, mode="reasoning", fmt="chat", seed=1, batch=8, gen_batch=8, max_new_tokens=24)
    for a, b in zip(first, second):
        for name, q in a["questions"].items():
            assert 1 <= q["reasoning"]["generated_tokens"] <= 24 and q["reasoning"]["text"].strip()
            assert sum(q["code_probs"]) == pytest.approx(1.0, abs=1e-5)
            assert q["reasoning"]["text"] == b["questions"][name]["reasoning"]["text"] and q["code_probs"] == b["questions"][name]["code_probs"]
