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
* ``score`` — 판(:data:`PROMPT_VERSIONS`)마다 다르다. **v0c-p3(기본, Task v0d fix round 1)**: [0, 1]을 **기준점 11개**
  0.0 · 0.1 · … · 1.0(코드 A–K)으로 묻고 점수 = 기준점의 기대값이다 — [0, 1] 전체에 닿는다(학습 목표는 이웃한 두 기준점 사이의
  선형 보간, :mod:`robo_jev.decisions.examples`). **v0c-p2(재현용)**: 6구간 0.0–0.1 · 0.1–0.3 · 0.3–0.5 · 0.5–0.7 · 0.7–0.9 ·
  0.9–1.0, 점수 = 구간 중앙값(0.05 · 0.2 · 0.4 · 0.6 · 0.8 · 0.95)의 기대값 — 읽을 수 있는 값이 [0.05, 0.95] 안이라(v0a.1 점수
  정답의 32 %가 그 밖) p3로 바꿨다. 어느 판이든 척도는 순서가 있어 마구 섞지 않고 **오름·내림 방향만** seed로 고른다(첫 자리 = 낮은
  값이라는 상관을 끊는다; 같은 열쇠·질문·seed면 두 판의 방향이 같다). 질문의 criteria(0·1의 뜻)를 ``Scale:`` 줄로 보인다.
  **choice·noul의 글은 두 판에서 바이트 단위로 같다** — 바뀐 것은 score 질문의 지시 한 줄과 선택지 줄뿐이다.

**두 형식** (:data:`FORMATS`): ``plain``(맨 글)과 ``chat``(Qwen3.5의 chat template — system·user 차례, 생각 없는 답이면
``<think>\\n\\n</think>\\n\\n`` 뒤, 생각하는 답이면 ``<think>\\n`` 뒤; 공식 ``chat_template.jinja``(2B revision 15852e8)를 그린
글과 같음을 검사가 고정한다). 어느 쪽을 기본으로 쓸지는 실측으로 고른다(보고서 `.superpowers/sdd/task-v0c-report.md`).

**두 방식** (:data:`MODES`): ``direct``는 위 그대로 ``Answer:``에서 읽는다(System 1과 같은 조건). ``reasoning``은 v0c-p2부터
**짧게 보이는 설명 뒤 답**이다: 질문 뒤에 :data:`EXPLAIN_TEXT` (세 문장 안에서 결정을 가르는 양을 계산하고 ``Answer:``와 글자 하나를
쓰라)를 붙이고, chat은 생각 블록을 비운 채(공식 기본 = 생각 없음) 답 차례에서, plain은 ``Explanation:`` 뒤에서 생성한다. 생성 뒤에
:func:`reasoning_close` (``\nAnswer:``)를 붙여 같은 코드 확률을 읽는다(:mod:`robo_jev.decisions.zero_shot`). v0c-p1의 원래 생각
(``<think>`` 안의 긴 생각)은 2B가 1,024토큰 안에 거의 끝내지 못해(리뷰 1, 보고서 §7.7) p2에서 바꿨다. direct 글은 p1과 바이트 단위로 같다.

**배치** (:data:`ORDERS`): ``seeded``는 질문마다 보이는 순서 하나(위의 seed 섞기), ``counterbalanced``(진단의 기본)는 결정적 균형
배치다 — choice는 seed로 섞은 순서의 **순환 이동 n개**(각 선택지가 각 자리에 정확히 한 번), noul은 Yes 먼저·나중 둘, score는
오름·내림 둘. 0번 배치는 언제나 ``seeded``의 순서와 같다. 배치마다 프롬프트 하나다(``listing`` 번호, ``n_listings``).

:func:`check_displayed_lines` 는 글에 보이는 ``X. …`` 줄을 다시 읽어 기록된 ``options``·``values``와 맞는지 본다(리뷰 1 I5 —
섞기와 되돌리기가 같은 목록을 쓰므로 그것만으로는 어긋남이 안 보인다). 진단은 채점 전에 모든 프롬프트를 이것으로 검사한다.

**Codes past Z and assigned codes (Task v0.8a).** The codes come from the pool :data:`robo_jev.decisions.codes.CODE_POOL` (A–Z, then
two-letter codes AA, AB, …). A question with up to 26 options and no assigned codes shows A, B, … exactly as before (the same
:func:`~robo_jev.model.zero_shot.verify_codes` path — byte-identical prompts). A choice question with more than 26 options (only a record
that opts in to 255 options has one) shows the pool codes in order. ``codes`` (:func:`build_prompts`; a record's ``meta.codes`` through
:func:`record_prompts` and the trainer's examples) gives a choice question **assigned codes**: one pool code per shown position, in display
order — position j of every listing shows ``codes[j]`` — and the readout reads those codes' rows. Each code is checked to be one token at the
answer boundary (:func:`~robo_jev.decisions.codes.verify_pool_codes`). The fixed text and the prompt version are unchanged.

고정 문구(지시·설명 지시·구간·기준점 이름 등)는 판마다 고정한다 — 문구를 바꾸면 판을 올린다(검사가 판마다
:func:`fixed_text_digest` 를 고정해 둔다). :data:`PROMPT_VERSION` 은 지금의 기본 판(진단·학습·평가 — p3)이고, 모든 함수가
``version``으로 다른 판(p2 재현)을 받는다. 프롬프트 dict의 ``version``이 그 프롬프트를 만든 판이다.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
from typing import Any, Mapping, Sequence

from robo_jev.decisions.codes import LETTERS, assigned_codes, sequential_codes, verify_pool_codes
from robo_jev.decisions.records import shuffle_key
from robo_jev.model.zero_shot import verify_codes

__all__ = [
    "ANSWER_CUE",
    "DEFAULT_FORMAT",
    "DEFAULT_ORDERS",
    "EXPLAIN_TEXT",
    "FORMATS",
    "MODES",
    "NOUL_LINES",
    "ORDERS",
    "PROMPT_VERSION",
    "PROMPT_VERSIONS",
    "SCORE_ANCHORS",
    "SCORE_ANCHOR_TEXT",
    "SCORE_BINS",
    "SCORE_CENTERS",
    "SCORE_TEXT",
    "SYSTEM_TEXT",
    "answer_from_codes",
    "build_prompts",
    "check_displayed_lines",
    "fixed_text_digest",
    "option_order",
    "reasoning_close",
    "record_prompts",
    "score_layout",
    "score_values",
]

#: 판들 — v0c-p2(재현용: 점수 6구간), v0c-p3(점수 기준점 11개; choice·noul은 p2와 바이트 단위로 같다).
PROMPT_VERSIONS = ("v0c-p2", "v0c-p3")
#: 지금의 기본 판 — 진단·학습·평가 모두 (Task v0d fix round 1, 통제자 결정 1).
PROMPT_VERSION = "v0c-p3"
FORMATS = ("plain", "chat")
MODES = ("direct", "reasoning")
ORDERS = ("counterbalanced", "seeded")
#: 진단의 기본 배치 (통제자 결정, fix round 1): 결정적 균형 배치.
DEFAULT_ORDERS = "counterbalanced"
#: 진단의 기본 형식 — plain과 chat을 2B에서 실측해 고른다(보고서 `.superpowers/sdd/task-v0c-report.md`의 "Prompt format").
DEFAULT_FORMAT = "chat"

#: 프롬프트 머리의 고정 지시 — 무엇을 하는 모델인지(L2 판단, docs/12 §1).
SYSTEM_TEXT = (
    "You are the judgment module of a robot. You read what the robot senses right now and answer one question about "
    "the next few seconds. Judge from the physical situation: distances, speeds, braking, timing, and what the robot "
    "is carrying. Answer with the letter of exactly one option."
)
#: reasoning 방식(v0c-p2: 짧게 보이는 설명 뒤 답)에서 질문 뒤에 붙는 지시.
EXPLAIN_TEXT = (
    "In at most three short sentences, work out the key quantity that decides this (a distance, time, speed or "
    'acceleration). Then write "Answer:" followed by one letter.'
)
#: score 질문의 구간 앞 지시 (v0c-p2).
SCORE_TEXT = "Pick the range that contains your estimate."
#: score 질문의 기준점 앞 지시 (v0c-p3).
SCORE_ANCHOR_TEXT = "Pick the value closest to your estimate."
ANSWER_CUE = "Answer:"
_PLAIN_EXPLAIN_CUE = "Explanation:"
_CHAT_SYSTEM = "<|im_start|>system\n"
_CHAT_USER = "<|im_start|>user\n"
_CHAT_END = "<|im_end|>\n"
_CHAT_ASSISTANT = "<|im_start|>assistant\n"
_THINK_EMPTY = "<think>\n\n</think>\n\n"
_OPTIONS_HEAD = "\nOptions:\n"
_OPTION_LINE = re.compile(r"^([A-Z]{1,2})\. (.*)$")  # one or two capital letters (codes past Z, Task v0.8a)
_BIN_LINE = re.compile(r"^(\d+\.\d) to (\d+\.\d)$")
_ANCHOR_LINE = re.compile(r"^(\d\.\d)$")

#: v0c-p2의 score 6구간과 중앙값 (모듈 설명).
SCORE_BINS: tuple[tuple[float, float], ...] = ((0.0, 0.1), (0.1, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 0.9), (0.9, 1.0))
SCORE_CENTERS: tuple[float, ...] = tuple(round((lo + hi) / 2, 6) for lo, hi in SCORE_BINS)
#: v0c-p3의 score 기준점 11개 0.0, 0.1, …, 1.0 (코드 A–K).
SCORE_ANCHORS: tuple[float, ...] = tuple(round(i / 10, 1) for i in range(11))
#: noul의 두 선택지 (내부 키, 보이는 글).
NOUL_LINES: tuple[tuple[str, str], ...] = (("yes", "Yes"), ("no", "No"))
_REQUEST_KEYS = frozenset({"state", "questions"})


def _check_version(version: str) -> str:
    if version not in PROMPT_VERSIONS:
        raise ValueError(f"prompt version: {list(PROMPT_VERSIONS)} 중 하나 (받은 값: {version!r})")
    return version


def score_layout(version: str = PROMPT_VERSION) -> dict[str, list[Any]]:
    """판의 score 선택지 (오름차순): ``{"names": 내부 키, "lines": 보이는 글, "values": 읽는 값}`` — p2는 6구간(이름 ``lo-hi``, 글
    ``lo to hi``, 값 = 중앙값), p3는 기준점 11개(이름·글 ``0.3``, 값 = 기준점)."""
    if _check_version(version) == "v0c-p2":
        return {
            "names": [_bin_name(lo, hi) for lo, hi in SCORE_BINS], "lines": [f"{lo:.1f} to {hi:.1f}" for lo, hi in SCORE_BINS],
            "values": [float(c) for c in SCORE_CENTERS],
        }
    return {"names": [f"{v:.1f}" for v in SCORE_ANCHORS], "lines": [f"{v:.1f}" for v in SCORE_ANCHORS], "values": [float(v) for v in SCORE_ANCHORS]}


def score_values(version: str = PROMPT_VERSION) -> tuple[float, ...]:
    """판의 score 읽기 값 (오름차순) — p2 구간 중앙값, p3 기준점. 읽을 수 있는 범위 = [첫 값, 끝 값]."""
    return tuple(score_layout(version)["values"])


def fixed_text_digest(version: str = PROMPT_VERSION) -> str:
    """판의 고정 문구 전부의 sha256 — 검사가 판마다 고정한다 (p2의 값은 v0c 때와 같다)."""
    fixed: dict[str, Any] = {
        "version": _check_version(version), "system": SYSTEM_TEXT, "explain": EXPLAIN_TEXT, "score": SCORE_TEXT, "answer": ANSWER_CUE,
        "plain_explain": _PLAIN_EXPLAIN_CUE, "bins": [list(b) for b in SCORE_BINS], "noul": [list(n) for n in NOUL_LINES],
        "chat": [_CHAT_SYSTEM, _CHAT_USER, _CHAT_END, _CHAT_ASSISTANT, _THINK_EMPTY],
        "layout": ["Situation:\n", "Question: ", "Scale:\n", "Options:\n", "{code}. {key}: {description}"],
        "orders": ["seeded: one seeded permutation", "counterbalanced: n cyclic shifts (choice), both directions (noul, score)"],
    }
    if version == "v0c-p3":
        del fixed["bins"]
        fixed["score"] = SCORE_ANCHOR_TEXT
        fixed["anchors"] = list(SCORE_ANCHORS)
        fixed["layout"] = fixed["layout"] + ["{code}. {anchor:.1f}"]
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


def _displayed(name: str, question: dict, *, key: str, seed: int | None, version: str = PROMPT_VERSION) -> dict[str, Any]:
    """보이는 순서의 선택지: ``{"options": [키], "lines": [줄 글], "values": [score 값] | None}`` (score는 판의 :func:`score_layout`)."""
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
        layout = score_layout(version)
        index = list(range(len(layout["values"])))
        if seed is not None and _rng(key, name, int(seed), "direction").random() < 0.5:
            index.reverse()
        return {
            "options": [layout["names"][i] for i in index],
            "lines": [layout["lines"][i] for i in index],
            "values": [layout["values"][i] for i in index],
        }
    raise ValueError(f"질문 {name!r}: 알 수 없는 type {kind!r} (choice·noul·score)")


def _listings(name: str, question: dict, *, key: str, seed: int | None, orders: str, version: str = PROMPT_VERSION) -> list[dict[str, Any]]:
    """질문 하나의 보이는 배치들 (모듈 설명의 "배치"). 0번은 언제나 :func:`_displayed` 의 seed 배치다."""
    base = _displayed(name, question, key=key, seed=seed, version=version)
    if orders == "seeded":
        return [base]
    if orders != "counterbalanced":
        raise ValueError(f"orders: {list(ORDERS)} 중 하나 (받은 값: {orders!r})")
    if question.get("type") == "choice":
        n = len(base["options"])
        return [{field: (base[field][shift:] + base[field][:shift]) if base[field] is not None else None for field in ("options", "lines", "values")} for shift in range(n)]
    # noul(둘)·score(순서 있는 척도): 정순과 역순
    return [base, {field: list(reversed(base[field])) if base[field] is not None else None for field in ("options", "lines", "values")}]


def _prefix(state: str, fmt: str) -> str:
    body = "Situation:\n" + state.strip() + "\n\n"
    if fmt == "plain":
        return SYSTEM_TEXT + "\n\n" + body
    return _CHAT_SYSTEM + SYSTEM_TEXT + _CHAT_END + _CHAT_USER + body


def _question_block(question: dict, codes: Sequence[str], lines: Sequence[str], version: str = PROMPT_VERSION) -> str:
    out = ["Question: " + _one_line(question.get("instructions", ""))]
    if question.get("type") == "score":
        criteria = [_one_line(item) for item in (question.get("criteria") or []) if _one_line(item)]
        if criteria:
            out.append("Scale:")
            out.extend(f"- {item}" for item in criteria)
        out.append(SCORE_TEXT if version == "v0c-p2" else SCORE_ANCHOR_TEXT)
    out.append("Options:")
    out.extend(f"{code}. {line}" for code, line in zip(codes, lines))
    return "\n".join(out)


def _suffix(block: str, fmt: str, mode: str) -> str:
    if fmt == "plain":
        if mode == "direct":
            return block + "\n" + ANSWER_CUE
        return block + "\n" + EXPLAIN_TEXT + "\n" + _PLAIN_EXPLAIN_CUE
    if mode == "direct":
        return block + _CHAT_END + _CHAT_ASSISTANT + _THINK_EMPTY + ANSWER_CUE
    return block + "\n" + EXPLAIN_TEXT + _CHAT_END + _CHAT_ASSISTANT + _THINK_EMPTY


def reasoning_close(fmt: str) -> str:
    """생성한 설명 뒤에 붙여 답 자리를 여는 글 — 두 형식 모두 ``\\nAnswer:`` (v0c-p2; 생각 블록이 없다)."""
    if fmt not in FORMATS:
        raise ValueError(f"format: {list(FORMATS)} 중 하나 (받은 값: {fmt!r})")
    return "\n" + ANSWER_CUE


def _question_codes(name: str, question: dict, count: int, tokenizer: Any, assigned: Sequence[str] | None) -> tuple[list[str], list[int]]:
    """The codes a question shows at its `count` positions and their token ids (module docstring, "Codes past Z and assigned codes"):
    the assigned codes, else A, B, … (≤ 26: the letters' path, unchanged), else the pool codes in order."""
    if assigned is not None:
        if question.get("type") != "choice":
            raise ValueError(f"question {name!r}: assigned codes are only for choice questions (type {question.get('type')!r})")
        assigned = [str(c) for c in assigned]
        if len(assigned) != count or len(set(assigned)) != count:
            raise ValueError(f"question {name!r}: {len(assigned)} assigned codes (distinct: {len(set(assigned))}) for {count} options")
        return assigned, verify_pool_codes(tokenizer, assigned)
    if count <= len(LETTERS):
        return verify_codes(tokenizer, count)
    codes = sequential_codes(count)
    return codes, verify_pool_codes(tokenizer, codes)


def build_prompts(
    request: dict[str, Any], tokenizer: Any, *, key: str, seed: int | None, fmt: str = "plain", mode: str = "direct", orders: str = "seeded",
    version: str = PROMPT_VERSION, codes: Mapping[str, Sequence[str]] | None = None,
) -> list[dict[str, Any]]:
    """요청(``state``·``questions``만 있는 dict) → 질문 × 배치마다 프롬프트 dict (요청의 질문 순서, 질문 안에서는 배치 순서).

    각 dict: ``question``, ``type``, ``listing``·``n_listings``(배치 번호·수), ``options``(보이는 순서의 키), ``option_lines``,
    ``values``(score 중앙값; 아니면 None), ``codes``, ``code_ids``, ``tokens``(접두 토큰 + 질문 토큰), ``prefix_tokens``, ``text``,
    ``prefix_chars``, ``format``, ``mode``, ``version``(만든 판). `key`·`seed`·`orders`는 선택지 순서만 정한다. `codes` (Task v0.8a): question
    name → assigned codes in display order (module docstring); every name must be a choice question of the request.
    """
    _check_version(version)
    if fmt not in FORMATS:
        raise ValueError(f"format: {list(FORMATS)} 중 하나 (받은 값: {fmt!r})")
    if mode not in MODES:
        raise ValueError(f"mode: {list(MODES)} 중 하나 (받은 값: {mode!r})")
    if not isinstance(request, dict) or set(request) - _REQUEST_KEYS or not _REQUEST_KEYS <= set(request):
        extra = sorted(set(request) - _REQUEST_KEYS) if isinstance(request, dict) else type(request).__name__
        raise ValueError(f"build_prompts는 요청(state·questions)만 받는다 — 다른 키: {extra}")
    codes = dict(codes or {})
    unknown = sorted(set(map(str, codes)) - set(map(str, request["questions"])))
    if unknown:
        raise ValueError(f"assigned codes for questions the request does not ask: {unknown}")
    prefix = _prefix(str(request["state"]), fmt)
    prefix_ids = list(tokenizer.encode(prefix, add_special_tokens=False).ids)
    if orders not in ORDERS:
        raise ValueError(f"orders: {list(ORDERS)} 중 하나 (받은 값: {orders!r})")
    out = []
    for name, question in request["questions"].items():
        listings = _listings(str(name), question, key=key, seed=seed, orders=orders, version=version)
        shown_codes, shown_ids = _question_codes(str(name), question, len(listings[0]["options"]), tokenizer, codes.get(str(name)))
        for index, shown in enumerate(listings):
            codes_here, code_ids = list(shown_codes), list(shown_ids)
            suffix = _suffix(_question_block(question, codes_here, shown["lines"], version), fmt, mode)
            suffix_ids = list(tokenizer.encode(suffix, add_special_tokens=False).ids)
            out.append(
                {
                    "question": str(name), "type": question["type"], "listing": index, "n_listings": len(listings),
                    "options": list(shown["options"]), "option_lines": list(shown["lines"]),
                    "values": list(shown["values"]) if shown["values"] is not None else None, "codes": codes_here, "code_ids": code_ids,
                    "tokens": prefix_ids + suffix_ids, "prefix_tokens": len(prefix_ids), "text": prefix + suffix, "prefix_chars": len(prefix),
                    "format": fmt, "mode": mode, "version": version,
                }
            )
    return out


def record_prompts(
    record: dict[str, Any], tokenizer: Any, *, seed: int | None, fmt: str = "plain", mode: str = "direct", orders: str = "seeded",
    version: str = PROMPT_VERSION,
) -> list[dict[str, Any]]:
    """기록 → 프롬프트들. 모델 입력은 ``record["request"]``뿐이고, id·``meta.pair``는 순서의 열쇠(:func:`shuffle_key`)로만 쓴다. A record's
    ``meta.codes`` (Task v0.8a) gives its choice questions assigned codes (module docstring); without it the prompts are unchanged."""
    request = record["request"]
    prompts = build_prompts(
        {"state": request["state"], "questions": request["questions"]}, tokenizer, key=shuffle_key(record), seed=seed, fmt=fmt, mode=mode, orders=orders,
        version=version, codes=assigned_codes(record) or None,
    )
    for p in prompts:
        p["record_id"] = str(record.get("id") or "")
    return prompts


def answer_from_codes(prompt: dict[str, Any], probabilities: Any) -> dict[str, Any]:
    """코드 위 확률(보이는 순서) → 질문의 답: choice ``{"probabilities": {키: p}}``, noul ``{"p_yes"}``, score ``{"value", "bins"}``
    (``value`` = 보이는 값(p2 구간 중앙값, p3 기준점)의 기대값; ``bins`` = 이름 → 확률 — p3에서는 기준점마다)."""
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


def check_displayed_lines(prompt: dict[str, Any]) -> list[str]:
    """글에 보이는 선택지 줄(``X. …``)을 다시 읽어 기록된 ``codes``·``options``·``values``와 맞는지 본다 → 문제 문장 목록(빈 목록 = 통과).

    줄 j는 코드 ``codes[j]``로 시작하고, choice면 ``options[j] + ":"``(설명이 없으면 키 그대로), noul이면 ``options[j]``에 맞는
    ``Yes``/``No``, score면 프롬프트의 판에 따라 ``options[j]``라는 이름의 구간(p2, ``lo to hi``; 그 중앙값이 ``values[j]``) 또는
    기준점(p3, ``0.3``; 그 값이 ``values[j]``)이어야 한다. 글은 기록과 따로
    다시 읽으므로, ``options``·줄·``values``가 어긋나면(섞기와 되돌리기가 같은 목록을 써서 다른 검사로는 안 보이는 일) 여기서 떨어진다.
    """
    suffix = prompt["text"][int(prompt.get("prefix_chars", 0)) :]
    head = suffix.rfind(_OPTIONS_HEAD)
    if head < 0:
        return ["no 'Options:' block in the question text"]
    parsed: list[tuple[str, str]] = []
    for raw in suffix[head + len(_OPTIONS_HEAD) :].split("\n"):
        line, ended = raw.split("<|im_end|>")[0], "<|im_end|>" in raw
        found = _OPTION_LINE.match(line)
        if not found:
            break
        parsed.append((found.group(1), found.group(2)))
        if ended:
            break
    options, codes, values = list(prompt["options"]), list(prompt["codes"]), prompt.get("values")
    version = str(prompt.get("version") or PROMPT_VERSION)
    problems = []
    if len(parsed) != len(options) or len(codes) != len(options):
        return [f"{len(parsed)} option lines in the text vs {len(options)} recorded options / {len(codes)} codes"]
    for j, ((code, rest), key) in enumerate(zip(parsed, options)):
        if code != codes[j]:
            problems.append(f"line {j}: code {code!r} vs recorded {codes[j]!r}")
        if prompt["type"] == "choice":
            if not (rest == key or rest.startswith(f"{key}: ")):
                problems.append(f"line {j}: {rest[:40]!r} does not show the recorded key {key!r}")
        elif prompt["type"] == "noul":
            expected = dict(NOUL_LINES).get(key)
            if expected is None or rest != expected:
                problems.append(f"line {j}: {rest!r} does not read as the recorded {key!r}")
        elif version == "v0c-p2":
            found = _BIN_LINE.match(rest)
            if not found:
                problems.append(f"line {j}: {rest!r} is not a score range")
                continue
            lo, hi = float(found.group(1)), float(found.group(2))
            if _bin_name(lo, hi) != key:
                problems.append(f"line {j}: range {rest!r} vs recorded bin {key!r}")
            if values is None or abs((lo + hi) / 2 - float(values[j])) > 1e-9:
                problems.append(f"line {j}: range {rest!r} midpoint vs recorded value {None if values is None else values[j]!r}")
        else:
            found = _ANCHOR_LINE.match(rest)
            if not found:
                problems.append(f"line {j}: {rest!r} is not a score anchor")
                continue
            anchor = float(found.group(1))
            if f"{anchor:.1f}" != key:
                problems.append(f"line {j}: anchor {rest!r} vs recorded anchor {key!r}")
            if values is None or abs(anchor - float(values[j])) > 1e-9:
                problems.append(f"line {j}: anchor {rest!r} vs recorded value {None if values is None else values[j]!r}")
    return problems
