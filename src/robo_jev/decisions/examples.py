"""판단 요청 → 학습 예 (Task v0d §1 — 학습 형식은 진단 도구의 프롬프트와 바이트 단위로 같다).

**같은 함수로 만든다.** 학습 예의 프롬프트는 :func:`robo_jev.decisions.prompt.build_prompts` 가 만든다 — 진단 도구
(:func:`robo_jev.decisions.prompt.record_prompts`)가 부르는 그 함수에, 같은 열쇠(:func:`robo_jev.decisions.records.shuffle_key`)와
같은 형식(chat 기본)·같은 방식(direct, ``Answer:``에서 끝남)으로 넘긴다. 학습 쪽에 따로 쓴 프롬프트 글은 없다. 고른 질문만 담은
요청(상태 글은 같은 문자열)을 넘기므로 접두 토큰과 질문 토큰이 진단의 것과 같다(질문의 선택지 순서는 열쇠·질문 이름·seed로만
정해진다). 모델 입력은 ``request``뿐이다 — ``labels``는 목표 분포를 만들 때만 읽는다.

**순서 증강.** 학습 예마다 순서 seed(:func:`epoch_plan` — run seed·epoch·레코드 id에서 해시로 뽑은 정수)를 넘겨 choice 선택지
순서(정렬한 키의 seed 순열), 예/아니오 순서, 점수 구간 방향을 무작위로 정한다. 목표 분포는 **보이는 순서로** 만든다(정답이 선택지와
함께 옮겨진다). 증강을 끄면(`augment=False`) 순서 seed는 :data:`FIXED_ORDER_SEED` (= 진단의 shuffle seed 1, 균형 배치의 0번 배치)다.

**Assigned codes (Task v0.8a).** A record's ``meta.codes`` (question name → pool codes in display order,
:mod:`robo_jev.decisions.codes`) reaches the prompt of each chosen question, so position j shows ``codes[j]`` whatever option the order seed
puts there; the target is built in display order as always. A record without it is trained exactly as before.

**정답 → 코드 분포** (:func:`code_target`, 보이는 순서):

* choice — 정답 확률 그대로(키로 옮긴다; 합 1 ± 1e-6을 정확히 1로 나눠 맞춘다).
* noul — ``[p_yes, 1 − p_yes]``를 예/아니오가 보이는 순서로.
* score — 프롬프트 판의 읽기 값(:func:`robo_jev.decisions.prompt.score_values`) 가운데 정답을 사이에 둔 **이웃한 두 값에 선형
  보간**: 기대값 Σ p·값 = 정답이다(:func:`score_target`). **v0c-p3(기본)**은 기준점 11개 0.0 · 0.1 · … · 1.0이라 [0, 1]의 모든 정답이
  그대로 닿는다(잘림 없음). **v0c-p2(재현용)**은 6구간 중앙값(0.05 … 0.95)이라 읽을 수 있는 값이 [0.05, 0.95](:data:`SCORE_READABLE`)
  뿐이고, 그 밖의 정답은 끝 구간에 몫 전부를 두고 ``clamped``로 적는다(v0a.1 표본의 score 정답 3,058개 중 989개, 32 % — p3로 바꾼 이유).

**질문 뽑기** (:func:`select_questions`): 요청마다 epoch당 최대 `q`개(기본 2)를 **종류별로 고르게** — 요청에 있는 종류를 무작위
순서로 돌며 종류마다 아직 안 뽑힌 질문 하나씩. 종류가 둘 이상이면 q = 2개는 서로 다른 종류다(세 종류가 다 있으면 각 종류가 2/3의
확률로 뽑힌다). 비용 조절이다(요청 하나의 질문 3~8개를 다 쓰지 않는다); epoch마다 다시 뽑으므로 여러 epoch에 걸쳐 다른 질문을 본다.

**Recorded per-record pick (Task v0.4b, opt-in by data).** A record may carry ``meta.question_pick = {"version": "qpick-1",
"secondary": [names]}`` (:func:`record_pick`). Its q examples are then picked from the *other* (primary) questions exactly as above —
the same seeded stream, so a record whose primary questions are those of an untransformed record gets that record's pick — and each
secondary question is trained **in addition** in the epochs where the plain pick over all its questions (same stream) chooses it. The
secondary questions keep the exposure they had when they shared the record's slots, and stop displacing the primary ones. Records
without the field are picked exactly as before (the trainer's default); the epoch length then depends on (seed, epoch)
(:func:`examples_per_epoch` with the run seed).

**Order-consistency copies (Task v0.9a, the trainer's opt-in ``consistency``).** A choice question picked by a fixed seed
(:func:`consistency_selected` — a hash of a fixed salt, the record id and the question name, below the share; the run seed does not enter,
so both seeds pick the same questions) is also shown as **permuted copies** of its training example (:func:`consistency_copies`): one cyclic
shift of the shown order by ⌈N/2⌉ and the reverse (:func:`copy_permutation`; for N = 2 the two are the same order, which is kept once). A
copy shows the same option lines in the permuted order with **the record's own code assignment by position** — position j shows the same
code as in the original (A, B, … or ``meta.codes``), so only the contents move — and the same prefix tokens; its text is what the
diagnosis builds for that order (a test checks it against the counterbalanced listings). ``permutation[j]`` is the original position shown at
copy position j and ``back`` its inverse: the copy's code distribution read at ``back`` is its distribution over the original's positions,
that is over the option contents (:func:`robo_jev.decisions.train.consistency_js`).

**dev 나눔** (:func:`split_dev`): 학습 파일에서 레코드 열쇠(:func:`shuffle_key` — 쌍이면 두 id 중 작은 것, 아니면 자기 id)의
salt 붙인 sha256이 `fraction`(기본 2 %) 아래인 레코드를 dev로 뗀다. 쌍(사실 하나만 바꾼 두 레코드)은 함께 간다 — 짝 하나로 학습하고
다른 하나로 dev 손실을 재면 거의 같은 글을 재는 것이 된다. 파일 순서와 무관하다. 고르기·멈추기는 dev로만 한다.

**epoch 계획** (:func:`epoch_plan`): (레코드 index, 질문 이름, 순서 seed)의 목록을 run seed·epoch로 정해진 순서로 섞은 것.
모든 무작위가 (seed, epoch, 레코드 id)의 해시에서 나오므로 계획은 그 셋의 순수 함수다 — 재개는 step 수만 알면 같은 자리에서 잇는다.
프롬프트 토큰은 step마다 그 step의 예만 만든다(:func:`build_examples`; 5만 요청 × 2질문을 미리 다 토큰화해 들고 있지 않는다).
"""

from __future__ import annotations

import hashlib
import math
import random
import statistics
from typing import Any, Sequence

from robo_jev.decisions.codes import assigned_codes
from robo_jev.decisions.prompt import FORMATS, PROMPT_VERSION, PROMPT_VERSIONS, build_prompts, check_displayed_lines, record_prompts, score_values
from robo_jev.decisions.prompt import _question_block, _suffix
from robo_jev.decisions.records import shuffle_key

__all__ = [
    "CONSISTENCY_COPIES",
    "CONSISTENCY_SALT",
    "DEFAULT_DEV_FRACTION",
    "DEFAULT_DEV_SALT",
    "DEFAULT_QUESTIONS_PER_REQUEST",
    "FIXED_ORDER_SEED",
    "QUESTION_PICK_VERSION",
    "SCORE_READABLE",
    "build_examples",
    "code_target",
    "consistency_copies",
    "consistency_selected",
    "copy_permutation",
    "derive_seed",
    "permuted_copy",
    "dev_examples",
    "epoch_plan",
    "examples_per_epoch",
    "has_pick_rule",
    "is_dev",
    "length_summary",
    "pick_rule",
    "record_pick",
    "score_target",
    "select_questions",
    "split_dev",
]

DEFAULT_QUESTIONS_PER_REQUEST = 2
DEFAULT_DEV_FRACTION = 0.02
DEFAULT_DEV_SALT = "v0d-dev"
#: 증강을 끈 학습과 dev 손실의 순서 seed — 진단 도구의 기본 shuffle seed(균형 배치의 0번 배치)와 같다.
FIXED_ORDER_SEED = 1
#: 판마다 score 읽기(읽기 값의 기대값)가 닿을 수 있는 범위 — p2 (0.05, 0.95), p3 (0.0, 1.0).
SCORE_READABLE = {version: (score_values(version)[0], score_values(version)[-1]) for version in PROMPT_VERSIONS}
_TYPES = ("choice", "noul", "score")
#: The recorded per-record pick rule (``meta.question_pick``, module docstring) — the only version the trainer knows.
QUESTION_PICK_VERSION = "qpick-1"
#: The permuted copies of the order-consistency option (Task v0.9a, module docstring): a cyclic shift by ⌈N/2⌉ and the reverse.
CONSISTENCY_COPIES = ("shift", "reverse")
#: The fixed salt that picks the consistency questions (independent of the run seed).
CONSISTENCY_SALT = "v09-consistency-1"


def derive_seed(*parts: Any) -> int:
    """부분들의 sha256 → 63비트 정수 (run seed·epoch·레코드 id 등에서 결정적인 seed를 뽑는다)."""
    digest = hashlib.sha256("|".join(str(part) for part in parts).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") & (2**63 - 1)


def score_target(value: float, *, version: str = PROMPT_VERSION) -> tuple[list[float], bool]:
    """score 정답 → 판의 읽기 값 위 분포(오름차순)와 ``clamped``(정답이 읽을 수 있는 범위 밖이었는지 — p3에서는 언제나 거짓). 모듈 설명."""
    v = float(value)
    if not (math.isfinite(v) and 0.0 <= v <= 1.0):
        raise ValueError(f"score 정답은 [0, 1] 안의 수여야 한다 (받은 값: {value!r})")
    centres = [float(c) for c in score_values(version)]
    probs = [0.0] * len(centres)
    if v <= centres[0]:
        probs[0] = 1.0
        return probs, v < centres[0]
    if v >= centres[-1]:
        probs[-1] = 1.0
        return probs, v > centres[-1]
    for i in range(len(centres) - 1):
        lo, hi = centres[i], centres[i + 1]
        if lo <= v <= hi:
            if v == lo:
                probs[i] = 1.0
            elif v == hi:
                probs[i + 1] = 1.0
            else:
                upper = (v - lo) / (hi - lo)
                probs[i], probs[i + 1] = 1.0 - upper, upper
            return probs, False
    raise AssertionError("unreachable")  # pragma: no cover


def code_target(prompt: dict[str, Any], label: dict[str, Any]) -> tuple[list[float], bool]:
    """프롬프트(보이는 순서의 ``options``·``values``)와 그 질문의 정답 → (보이는 순서의 코드 분포, clamped)."""
    kind = prompt["type"]
    if kind == "choice":
        probabilities = label["probabilities"]
        missing = [key for key in prompt["options"] if key not in probabilities]
        if missing:
            raise ValueError(f"{prompt.get('question')}: 정답에 없는 선택지 {missing}")
        raw = [float(probabilities[key]) for key in prompt["options"]]
        total = sum(raw)
        if not total > 0:
            raise ValueError(f"{prompt.get('question')}: 정답 확률의 합이 0이다")
        return [p / total for p in raw], False
    if kind == "noul":
        p_yes = float(label["p_yes"])
        return [p_yes if key == "yes" else 1.0 - p_yes for key in prompt["options"]], False
    if kind == "score":
        version = str(prompt.get("version") or PROMPT_VERSION)
        ascending, clamped = score_target(label["value"], version=version)
        index = {float(c): i for i, c in enumerate(score_values(version))}
        return [ascending[index[float(v)]] for v in prompt["values"]], clamped
    raise ValueError(f"{prompt.get('question')}: 알 수 없는 type {kind!r}")


def select_questions(questions: dict[str, dict[str, Any]], *, q: int, rng: random.Random) -> list[str]:
    """요청의 질문들 → 최대 `q`개의 이름(서로 다름), 종류별로 고르게 (모듈 설명). 결정적이다(같은 `rng` 상태면 같은 결과)."""
    if int(q) < 1:
        raise ValueError(f"questions_per_request: 1 이상이어야 한다 (받은 값: {q!r})")
    by_type: dict[str, list[str]] = {}
    for name, question in questions.items():
        by_type.setdefault(str(question["type"]), []).append(str(name))
    for names in by_type.values():
        rng.shuffle(names)
    types = sorted(by_type)
    rng.shuffle(types)
    chosen: list[str] = []
    while len(chosen) < int(q) and any(by_type.values()):
        for kind in types:
            if len(chosen) >= int(q):
                break
            if by_type[kind]:
                chosen.append(by_type[kind].pop())
    return chosen


def is_dev(record: dict[str, Any], *, fraction: float, salt: str) -> bool:
    """레코드 열쇠(쌍이면 작은 id)의 salt 붙인 해시가 `fraction` 아래면 dev."""
    digest = hashlib.sha256(f"{salt}|{shuffle_key(record)}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / 2.0**64 < float(fraction)


def split_dev(records: Sequence[dict[str, Any]], *, fraction: float = DEFAULT_DEV_FRACTION, salt: str = DEFAULT_DEV_SALT) -> tuple[list[dict], list[dict]]:
    """(학습, dev) — 파일 순서를 지킨다."""
    train, dev = [], []
    for record in records:
        (dev if is_dev(record, fraction=fraction, salt=salt) else train).append(record)
    return train, dev


def pick_rule(record: dict[str, Any]) -> list[str] | None:
    """The record's secondary questions under a recorded pick rule (``meta.question_pick``, module docstring), or None without one.
    A malformed rule — another version, no list of names, or a name the request does not ask — raises ``ValueError``."""
    rule = (record.get("meta") or {}).get("question_pick")
    if rule is None:
        return None
    rid = record.get("id")
    if not isinstance(rule, dict) or rule.get("version") != QUESTION_PICK_VERSION or not isinstance(rule.get("secondary"), list):
        raise ValueError(f"{rid}: meta.question_pick must be {{'version': {QUESTION_PICK_VERSION!r}, 'secondary': [names]}} (got {rule!r})")
    names = [str(name) for name in rule["secondary"]]
    unknown = [name for name in names if name not in record["request"]["questions"]]
    if unknown or len(set(names)) != len(names):
        raise ValueError(f"{rid}: meta.question_pick names questions the request does not ask, or repeats one: {unknown or names}")
    return names


def has_pick_rule(records: Sequence[dict[str, Any]]) -> bool:
    """Whether any record carries a recorded pick rule (``meta.question_pick``)."""
    return any((record.get("meta") or {}).get("question_pick") is not None for record in records)


def record_pick(record: dict[str, Any], *, q: int, seed: int, epoch: int) -> list[str]:
    """The questions a record trains in one epoch: :func:`select_questions` on its questions with the stream (seed, epoch, id) — or,
    under a recorded pick rule (module docstring), that pick over the primary questions followed by the secondary questions the plain
    pick over all questions (same stream) chooses."""
    questions = record["request"]["questions"]
    stream = derive_seed(seed, epoch, str(record.get("id") or ""), "questions")
    secondary = pick_rule(record)
    if secondary is None:
        return select_questions(questions, q=q, rng=random.Random(stream))
    extra = set(secondary)
    primary = {name: question for name, question in questions.items() if name not in extra}
    names = select_questions(primary, q=q, rng=random.Random(stream)) if primary else []
    plain = select_questions(questions, q=q, rng=random.Random(stream))
    return names + [name for name in plain if name in extra]


def examples_per_epoch(records: Sequence[dict[str, Any]], *, q: int, seed: int | None = None, epoch: int = 0) -> int:
    """epoch 하나의 학습 예 수 = Σ min(q, 요청의 질문 수) — epoch마다 같다. Under a recorded pick rule (module docstring) the count is that
    of the (seed, epoch) plan; the run `seed` must then be given."""
    if not has_pick_rule(records):
        return sum(min(int(q), len(record["request"]["questions"])) for record in records)
    if seed is None:
        raise ValueError("records with a recorded pick rule (meta.question_pick): the epoch length depends on the run seed — pass seed=")
    return sum(len(record_pick(record, q=q, seed=seed, epoch=epoch)) for record in records)


def epoch_plan(records: Sequence[dict[str, Any]], *, seed: int, epoch: int, q: int, augment: bool) -> list[tuple[int, str, int]]:
    """epoch 하나의 (레코드 index, 질문 이름, 순서 seed) 목록 — (seed, epoch)로 정해진 순서로 섞었다 (모듈 설명)."""
    plan: list[tuple[int, str, int]] = []
    for index, record in enumerate(records):
        rid = str(record.get("id") or "")
        names = record_pick(record, q=q, seed=seed, epoch=epoch)
        order_seed = derive_seed(seed, epoch, rid, "order") if augment else FIXED_ORDER_SEED
        plan.extend((index, name, order_seed) for name in names)
    random.Random(derive_seed(seed, epoch, "shuffle")).shuffle(plan)
    return plan


def _with_target(prompt: dict[str, Any], record: dict[str, Any]) -> dict[str, Any]:
    label = (record.get("labels") or {}).get(prompt["question"])
    if not isinstance(label, dict):
        raise ValueError(f"{record.get('id')}: 질문 {prompt['question']!r}의 정답이 없다")
    target, clamped = code_target(prompt, label)
    problems = check_displayed_lines(prompt)
    if problems:
        raise ValueError(f"{record.get('id')}/{prompt['question']}: 보이는 선택지 줄이 기록과 어긋난다 — {problems[0]}")
    prompt["target"] = target
    prompt["clamped"] = clamped
    return prompt


def build_examples(
    record: dict[str, Any], names: Sequence[str], tokenizer: Any, *, order_seed: int, fmt: str = "chat", mode: str = "direct",
    version: str = PROMPT_VERSION,
) -> list[dict[str, Any]]:
    """레코드와 고른 질문 이름들 → 학습 예들(질문 이름 순서). 프롬프트는 진단과 같은 함수·같은 열쇠로 만든다(모듈 설명).

    각 예 = :func:`build_prompts` 의 프롬프트 dict(``tokens``·``code_ids``·``options``·``values``·``text`` …) + ``record_id``,
    ``order_seed``, ``target``(보이는 순서의 코드 분포), ``clamped``.
    """
    if mode != "direct":
        raise ValueError(f"학습은 direct 방식(``Answer:``에서 끝나는 프롬프트)만 한다 (받은 값: {mode!r})")
    if fmt not in FORMATS:
        raise ValueError(f"format: {list(FORMATS)} 중 하나 (받은 값: {fmt!r})")
    request = record["request"]
    questions = request["questions"]
    unknown = [name for name in names if name not in questions]
    if unknown:
        raise ValueError(f"{record.get('id')}: 요청에 없는 질문 {unknown}")
    sub = {"state": request["state"], "questions": {name: questions[name] for name in names}}
    assigned = {name: codes for name, codes in assigned_codes(record).items() if name in sub["questions"]}  # Task v0.8a: meta.codes
    prompts = build_prompts(sub, tokenizer, key=shuffle_key(record), seed=int(order_seed), fmt=fmt, mode="direct", orders="seeded", version=version,
                            codes=assigned or None)
    out = []
    for prompt in prompts:
        prompt["record_id"] = str(record.get("id") or "")
        prompt["order_seed"] = int(order_seed)
        out.append(_with_target(prompt, record))
    return out


def consistency_selected(record: dict[str, Any], name: str, *, share: float, salt: str = CONSISTENCY_SALT) -> bool:
    """Whether a question of a record gets permuted copies (module docstring): a hash of the fixed `salt`, the record id and the question
    name below `share`. The caller checks that the question is a choice question."""
    return derive_seed(str(salt), str(record.get("id") or ""), str(name)) / float(2**63) < float(share)


def copy_permutation(kind: str, n: int) -> list[int]:
    """The permuted copy `kind` of an `n`-option listing: ``permutation[j]`` = the original position shown at copy position j — ``shift``
    moves every option ⌈n/2⌉ places (``old[s:] + old[:s]``), ``reverse`` reverses the listing."""
    n = int(n)
    if n < 2:
        raise ValueError(f"a permuted copy needs at least 2 options (got {n})")
    if kind == "shift":
        s = math.ceil(n / 2)
        return [(j + s) % n for j in range(n)]
    if kind == "reverse":
        return list(range(n - 1, -1, -1))
    raise ValueError(f"copy kind {kind!r} is not one of {list(CONSISTENCY_COPIES)}")


def permuted_copy(record: dict[str, Any], example: dict[str, Any], permutation: Sequence[int], tokenizer: Any) -> dict[str, Any]:
    """A choice example shown with its options in `permutation` (module docstring): same prefix tokens, the same codes by position, the
    option lines permuted. Carries ``permutation``, ``back`` (its inverse: the copy position of each original position) and the target
    in the copy's order (not trained — only the original carries the target loss). The shown lines are checked against the record."""
    if example.get("type") != "choice":
        raise ValueError(f"{example.get('record_id')}/{example.get('question')}: permuted copies are for choice questions")
    n = len(example["options"])
    permutation = [int(i) for i in permutation]
    if sorted(permutation) != list(range(n)):
        raise ValueError(f"{permutation} is not a permutation of {n} positions")
    question = record["request"]["questions"][example["question"]]
    lines = [example["option_lines"][i] for i in permutation]
    suffix = _suffix(_question_block(question, list(example["codes"]), lines, example["version"]), example["format"], example["mode"])
    prefix = example["text"][: int(example["prefix_chars"])]
    back = [0] * n
    for j, i in enumerate(permutation):
        back[i] = j
    copy = {key: example[key] for key in ("question", "type", "codes", "code_ids", "prefix_tokens", "prefix_chars", "format", "mode", "version",
                                          "record_id", "order_seed")}
    copy.update({"listing": None, "n_listings": None, "options": [example["options"][i] for i in permutation], "option_lines": lines,
                 "values": None, "codes": list(example["codes"]), "code_ids": list(example["code_ids"]),
                 "tokens": list(example["tokens"][: int(example["prefix_tokens"])]) + list(tokenizer.encode(suffix, add_special_tokens=False).ids),
                 "text": prefix + suffix, "target": [example["target"][i] for i in permutation], "clamped": example.get("clamped", False),
                 "permutation": permutation, "back": back})
    problems = check_displayed_lines(copy)
    if problems:
        raise ValueError(f"{example.get('record_id')}/{example.get('question')}: a permuted copy's lines do not read back — {problems[0]}")
    return copy


def consistency_copies(record: dict[str, Any], example: dict[str, Any], tokenizer: Any, *, kinds: Sequence[str] = CONSISTENCY_COPIES) -> list[dict[str, Any]]:
    """The distinct permuted copies of a choice example (module docstring), in the order of `kinds`; a copy whose order repeats an earlier
    one's (N = 2: the shift is the reverse) is kept once. Each copy carries its ``copy`` kind."""
    out: list[dict[str, Any]] = []
    seen: set[tuple[int, ...]] = set()
    for kind in kinds:
        permutation = copy_permutation(kind, len(example["options"]))
        if tuple(permutation) in seen:
            continue
        seen.add(tuple(permutation))
        made = permuted_copy(record, example, permutation, tokenizer)
        made["copy"] = kind
        out.append(made)
    return out


def dev_examples(
    records: Sequence[dict[str, Any]], tokenizer: Any, *, fmt: str = "chat", seed: int = FIXED_ORDER_SEED, version: str = PROMPT_VERSION
) -> list[dict[str, Any]]:
    """dev 손실용: 레코드마다 **모든 질문 × 균형 배치 모두**(진단과 같은 :func:`record_prompts`, shuffle seed `seed`) + 목표 분포.
    매번 같은 프롬프트라 dev 손실의 변화가 순서 표집의 잡음이 아니다."""
    out = []
    for record in records:
        for prompt in record_prompts(record, tokenizer, seed=seed, fmt=fmt, mode="direct", orders="counterbalanced", version=version):
            out.append(_with_target(prompt, record))
    return out


def length_summary(lengths: Sequence[int]) -> dict[str, Any]:
    """토큰 길이 분포: count·total·mean·min·p10·p50·p90·p99·max."""
    values = sorted(int(v) for v in lengths)
    if not values:
        return {"count": 0, "total": 0, "mean": None, "min": None, "p10": None, "p50": None, "p90": None, "p99": None, "max": None}

    def quantile(q: float) -> float:
        position = q * (len(values) - 1)
        low = math.floor(position)
        high = min(low + 1, len(values) - 1)
        return float(values[low] + (values[high] - values[low]) * (position - low))

    return {
        "count": len(values), "total": int(sum(values)), "mean": float(statistics.fmean(values)), "min": values[0],
        "p10": quantile(0.10), "p50": quantile(0.50), "p90": quantile(0.90), "p99": quantile(0.99), "max": values[-1],
    }
