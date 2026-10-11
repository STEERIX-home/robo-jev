"""판단 요청 → 질문마다 프롬프트 하나 (Task v0c §2.1; 기획서 docs/12 §11 "학습 없는 기준선은 같은 요청에서 LM head의 선택지 토큰 확률로 읽는다").

**Nimble 방식** (:mod:`robo_jev.model.zero_shot` 과 같은 규칙): 선택지마다 한 글자 코드(A, B, …)를 주고 프롬프트를
``Answer:``에서 끝낸다. 다음 토큰 자리의 LM head **코드 행만** 읽어 코드 위 확률을 얻는다. 코드는 답 경계에서 한 토큰이어야 하고
(:func:`robo_jev.model.zero_shot.verify_codes` 로 검사), 선택지 순서는 레코드마다 정해진 seed로 섞는다(위치와 정답의 상관을
끊고, 위치 편향을 잰다). 학습은 없다.

**한 질문 = 한 프롬프트.** 프롬프트 = 접두(고정 지시 + 상태 글) + 그 질문. 접두는 한 레코드의 모든 질문에서 **같은 문자열이고
따로 토큰화한 같은 토큰열**이다(질문 글과 경계에서 BPE가 섞이지 않는다). 모델 입력은 ``request``(``state``·``questions``)뿐이다 —
:func:`build_prompts` 는 요청 dict만 받고 다른 키가 있으면 거절한다. 레코드의 ``id``와 ``meta.pair``는 선택지 **순서**의 seed
열쇠로만 쓴다(:func:`robo_jev.decisions.records.shuffle_key` — 쌍의 두 레코드가 같은 순서를 본다).

질문 종류별 답 자리 (`options`는 보이는 순서의 키, 코드 A, B, …가 그 순서로 붙는다):

* ``choice`` — 선택지 줄 ``<코드>. <키>: <설명>``. 키를 보이는 이유: 상태 글의 계획기 권고가 선택지를 키로 부른다. 섞기는
  정렬한 키 목록에 seed 순열을 건다(요청 안의 순서와 무관 — 쌍의 두 레코드가 같은 순서를 본다). seed가 None이면 요청 순서.
* ``noul`` — 코드 둘 ``Yes``·``No``(순서도 섞는다) → ``p_yes`` = Yes 코드의 확률.
* ``score`` — [0, 1]을 6구간 코드로 묻는다: 0.0–0.1 · 0.1–0.3 · 0.3–0.5 · 0.5–0.7 · 0.7–0.9 · 0.9–1.0. 점수 = 구간 중앙값
  (0.05 · 0.2 · 0.4 · 0.6 · 0.8 · 0.95)의 기대값. 양 끝 구간을 좁게 둔 것은 점수가 대개 확률이고(기획서 §5) 0·1 가까운 값의
  구별이 중요해서다 — 그 대가로 읽을 수 있는 값이 [0.05, 0.95] 안이다(정답 0이면 바닥 로그 손실 −log 0.95). 구간은 순서가
  있는 척도라 마구 섞지 않고 **오름·내림 방향만** seed로 고른다(첫 자리 = 낮은 값이라는 상관을 끊는다). 질문의 criteria(0·1의
  뜻)를 ``Scale:`` 줄로 보인다.

**두 형식** (:data:`FORMATS`): ``plain``(맨 글)과 ``chat``(Qwen3.5의 chat template — system·user 차례, 생각 없는 답이면
``<think>\\n\\n</think>\\n\\n`` 뒤, 생각하는 답이면 ``<think>\\n`` 뒤; 공식 ``chat_template.jinja``(2B revision 15852e8)를 그린
글과 같음을 검사가 고정한다). 어느 쪽을 기본으로 쓸지는 실측으로 고른다(보고서 `.superpowers/sdd/task-v0c-report.md`).

**두 방식** (:data:`MODES`): ``direct``는 위 그대로 ``Answer:``에서 읽는다(System 1과 같은 조건). ``reasoning``은 질문 뒤에
"짧게 생각하고 답하라"(:data:`REASONING_TEXT`)를 붙여 생성하게 하고(plain은 ``Reasoning:`` 뒤, chat은 ``<think>`` 안),
생성 뒤에 :func:`reasoning_close` 를 붙여 같은 코드 확률을 읽는다(:mod:`robo_jev.decisions.zero_shot`).

고정 문구(지시·생각 지시·구간 이름 등)는 :data:`PROMPT_VERSION` 과 함께 고정한다 — 문구를 바꾸면 버전을 올린다(검사가
:func:`fixed_text_digest` 로 둘을 묶어 둔다).
"""

from __future__ import annotations

import hashlib
import json
import random
from typing import Any, Sequence

from robo_jev.decisions.records import shuffle_key
from robo_jev.model.zero_shot import verify_codes

__all__ = [
    "ANSWER_CUE",
    "DEFAULT_FORMAT",
    "FORMATS",
    "MODES",
    "NOUL_LINES",
    "PROMPT_VERSION",
    "REASONING_TEXT",
    "SCORE_BINS",
    "SCORE_CENTERS",
    "SCORE_TEXT",
    "SYSTEM_TEXT",
    "answer_from_codes",
    "build_prompts",
    "fixed_text_digest",
    "option_order",
    "reasoning_close",
    "record_prompts",
]

PROMPT_VERSION = "v0c-p1"
FORMATS = ("plain", "chat")
MODES = ("direct", "reasoning")
#: 진단의 기본 형식 — plain과 chat을 2B에서 실측해 고른다(보고서 `.superpowers/sdd/task-v0c-report.md`의 "Prompt format").
DEFAULT_FORMAT = "chat"

#: 프롬프트 머리의 고정 지시 — 무엇을 하는 모델인지(L2 판단, docs/12 §1).
SYSTEM_TEXT = (
    "You are the judgment module of a robot. You read what the robot senses right now and answer one question about "
    "the next few seconds. Judge from the physical situation: distances, speeds, braking, timing, and what the robot "
    "is carrying. Answer with the letter of exactly one option."
)
#: reasoning 방식에서 질문 뒤에 붙는 지시.
REASONING_TEXT = "Think briefly about the physics (a few sentences), then answer with one letter."
#: score 질문의 구간 앞 지시.
SCORE_TEXT = "Pick the range that contains your estimate."
ANSWER_CUE = "Answer:"
_PLAIN_REASONING_CUE = "Reasoning:"
_CHAT_SYSTEM = "<|im_start|>system\n"
_CHAT_USER = "<|im_start|>user\n"
_CHAT_END = "<|im_end|>\n"
_CHAT_ASSISTANT = "<|im_start|>assistant\n"
_THINK_EMPTY = "<think>\n\n</think>\n\n"
_THINK_OPEN = "<think>\n"
_THINK_CLOSE = "\n</think>\n\n"

#: score의 6구간과 중앙값 (모듈 설명).
SCORE_BINS: tuple[tuple[float, float], ...] = ((0.0, 0.1), (0.1, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 0.9), (0.9, 1.0))
SCORE_CENTERS: tuple[float, ...] = tuple(round((lo + hi) / 2, 6) for lo, hi in SCORE_BINS)
#: noul의 두 선택지 (내부 키, 보이는 글).
NOUL_LINES: tuple[tuple[str, str], ...] = (("yes", "Yes"), ("no", "No"))
_REQUEST_KEYS = frozenset({"state", "questions"})


def fixed_text_digest() -> str:
    """고정 문구 전부의 sha256 — 검사가 :data:`PROMPT_VERSION` 과 함께 고정한다."""
    fixed = {
        "version": PROMPT_VERSION, "system": SYSTEM_TEXT, "reasoning": REASONING_TEXT, "score": SCORE_TEXT, "answer": ANSWER_CUE,
        "plain_reasoning": _PLAIN_REASONING_CUE, "bins": [list(b) for b in SCORE_BINS], "noul": [list(n) for n in NOUL_LINES],
        "chat": [_CHAT_SYSTEM, _CHAT_USER, _CHAT_END, _CHAT_ASSISTANT, _THINK_EMPTY, _THINK_OPEN, _THINK_CLOSE],
        "layout": ["Situation:\n", "Question: ", "Scale:\n", "Options:\n", "{code}. {key}: {description}"],
    }
    return hashlib.sha256(json.dumps(fixed, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def _rng(key: str, question: str, seed: int, salt: str) -> random.Random:
    digest = hashlib.sha256(f"{key}|{question}|{seed}|{salt}".encode("utf-8")).digest()
    return random.Random(int.from_bytes(digest[:8], "big"))


def option_order(key: str, question: str, n: int, seed: int | None) -> list[int]:
    """(열쇠, 질문 이름, seed)로 정해지는 `range(n)`의 순열. seed가 None이면 그대로."""
    order = list(range(n))
    if seed is not None:
        _rng(key, question, int(seed), "order").shuffle(order)
    return order


def _one_line(text: Any) -> str:
    return " ".join(str(text).split())


def _bin_name(lo: float, hi: float) -> str:
    return f"{lo:.1f}-{hi:.1f}"


def _displayed(name: str, question: dict, *, key: str, seed: int | None) -> dict[str, Any]:
    """보이는 순서의 선택지: ``{"options": [키], "lines": [줄 글], "values": [score 값] | None}``."""
    kind = question.get("type")
    if kind == "choice":
        criteria = question.get("criteria")
        if not isinstance(criteria, dict) or len(criteria) < 2:
            raise ValueError(f"choice 질문 {name!r}: 선택지가 둘 이상인 criteria 객체가 필요하다")
        keys = list(criteria) if seed is None else sorted(criteria)
        keys = [keys[i] for i in option_order(key, name, len(keys), seed)]
        lines = [f"{k}: {_one_line(criteria[k])}" if _one_line(criteria[k]) else str(k) for k in keys]
        return {"options": keys, "lines": lines, "values": None}
    if kind == "noul":
        pairs = [NOUL_LINES[i] for i in option_order(key, name, len(NOUL_LINES), seed)]
        return {"options": [k for k, _ in pairs], "lines": [line for _, line in pairs], "values": None}
    if kind == "score":
        index = list(range(len(SCORE_BINS)))
        if seed is not None and _rng(key, name, int(seed), "direction").random() < 0.5:
            index.reverse()
        return {
            "options": [_bin_name(*SCORE_BINS[i]) for i in index],
            "lines": [f"{SCORE_BINS[i][0]:.1f} to {SCORE_BINS[i][1]:.1f}" for i in index],
            "values": [SCORE_CENTERS[i] for i in index],
        }
    raise ValueError(f"질문 {name!r}: 알 수 없는 type {kind!r} (choice·noul·score)")


def _prefix(state: str, fmt: str) -> str:
    body = "Situation:\n" + state.strip() + "\n\n"
    if fmt == "plain":
        return SYSTEM_TEXT + "\n\n" + body
    return _CHAT_SYSTEM + SYSTEM_TEXT + _CHAT_END + _CHAT_USER + body


def _question_block(question: dict, codes: Sequence[str], lines: Sequence[str]) -> str:
    out = ["Question: " + _one_line(question.get("instructions", ""))]
    if question.get("type") == "score":
        criteria = [_one_line(item) for item in (question.get("criteria") or []) if _one_line(item)]
        if criteria:
            out.append("Scale:")
            out.extend(f"- {item}" for item in criteria)
        out.append(SCORE_TEXT)
    out.append("Options:")
    out.extend(f"{code}. {line}" for code, line in zip(codes, lines))
    return "\n".join(out)


def _suffix(block: str, fmt: str, mode: str) -> str:
    if fmt == "plain":
        if mode == "direct":
            return block + "\n" + ANSWER_CUE
        return block + "\n" + REASONING_TEXT + "\n" + _PLAIN_REASONING_CUE
    if mode == "direct":
        return block + _CHAT_END + _CHAT_ASSISTANT + _THINK_EMPTY + ANSWER_CUE
    return block + "\n" + REASONING_TEXT + _CHAT_END + _CHAT_ASSISTANT + _THINK_OPEN


def reasoning_close(fmt: str) -> str:
    """생성한 생각 뒤에 붙여 답 자리를 여는 글 — plain ``\\nAnswer:``, chat ``\\n</think>\\n\\nAnswer:``."""
    if fmt not in FORMATS:
        raise ValueError(f"format: {list(FORMATS)} 중 하나 (받은 값: {fmt!r})")
    return ("\n" if fmt == "plain" else _THINK_CLOSE) + ANSWER_CUE


def build_prompts(
    request: dict[str, Any], tokenizer: Any, *, key: str, seed: int | None, fmt: str = "plain", mode: str = "direct"
) -> list[dict[str, Any]]:
    """요청(``state``·``questions``만 있는 dict) → 질문마다 프롬프트 dict (요청의 질문 순서).

    각 dict: ``question``, ``type``, ``options``(보이는 순서의 키), ``option_lines``, ``values``(score 중앙값; 아니면 None),
    ``codes``, ``code_ids``, ``tokens``(접두 토큰 + 질문 토큰), ``prefix_tokens``, ``text``, ``prefix_chars``, ``format``,
    ``mode``, ``version``. `key`·`seed`는 선택지 순서만 정한다.
    """
    if fmt not in FORMATS:
        raise ValueError(f"format: {list(FORMATS)} 중 하나 (받은 값: {fmt!r})")
    if mode not in MODES:
        raise ValueError(f"mode: {list(MODES)} 중 하나 (받은 값: {mode!r})")
    if not isinstance(request, dict) or set(request) - _REQUEST_KEYS or not _REQUEST_KEYS <= set(request):
        extra = sorted(set(request) - _REQUEST_KEYS) if isinstance(request, dict) else type(request).__name__
        raise ValueError(f"build_prompts는 요청(state·questions)만 받는다 — 다른 키: {extra}")
    prefix = _prefix(str(request["state"]), fmt)
    prefix_ids = list(tokenizer.encode(prefix, add_special_tokens=False).ids)
    out = []
    for name, question in request["questions"].items():
        shown = _displayed(str(name), question, key=key, seed=seed)
        codes, code_ids = verify_codes(tokenizer, len(shown["options"]))
        suffix = _suffix(_question_block(question, codes, shown["lines"]), fmt, mode)
        suffix_ids = list(tokenizer.encode(suffix, add_special_tokens=False).ids)
        out.append(
            {
                "question": str(name), "type": question["type"], "options": shown["options"], "option_lines": shown["lines"],
                "values": shown["values"], "codes": codes, "code_ids": code_ids, "tokens": prefix_ids + suffix_ids,
                "prefix_tokens": len(prefix_ids), "text": prefix + suffix, "prefix_chars": len(prefix), "format": fmt, "mode": mode,
                "version": PROMPT_VERSION,
            }
        )
    return out


def record_prompts(record: dict[str, Any], tokenizer: Any, *, seed: int | None, fmt: str = "plain", mode: str = "direct") -> list[dict[str, Any]]:
    """기록 → 프롬프트들. 모델 입력은 ``record["request"]``뿐이고, id·``meta.pair``는 순서의 열쇠(:func:`shuffle_key`)로만 쓴다."""
    request = record["request"]
    prompts = build_prompts({"state": request["state"], "questions": request["questions"]}, tokenizer, key=shuffle_key(record), seed=seed, fmt=fmt, mode=mode)
    for p in prompts:
        p["record_id"] = str(record.get("id") or "")
    return prompts


def answer_from_codes(prompt: dict[str, Any], probabilities: Any) -> dict[str, Any]:
    """코드 위 확률(보이는 순서) → 질문의 답: choice ``{"probabilities": {키: p}}``, noul ``{"p_yes"}``, score ``{"value", "bins"}``."""
    probs = [float(p) for p in (probabilities.tolist() if hasattr(probabilities, "tolist") else probabilities)]
    if len(probs) != len(prompt["options"]):
        raise ValueError(f"확률 {len(probs)}개 ≠ 선택지 {len(prompt['options'])}개 ({prompt.get('question')})")
    if prompt["type"] == "choice":
        return {"probabilities": dict(zip(prompt["options"], probs))}
    if prompt["type"] == "noul":
        return {"p_yes": probs[prompt["options"].index("yes")]}
    value = sum(p * v for p, v in zip(probs, prompt["values"]))
    bins = {name: p for name, p in sorted(zip(prompt["options"], probs))}
    return {"value": value, "bins": bins}
