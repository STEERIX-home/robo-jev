"""판단 요청 기록(`simgen-request/0`) 읽기·형식 검사·표본 고르기 (Task v0c).

진단 도구가 v0a 엔진의 **코드가 아니라 JSONL 형식에만** 기대는 자리다(형식 정의: v0a 지시서 §2). 한 줄 = 한 판단 순간 =
한 요청이고, 모델 입력은 ``request``(상태 글 + 질문들)뿐이다 — ``meta``·``labels``·``truth``·``rationale``는 채점과 묶음에만
쓴다(:mod:`robo_jev.decisions.prompt` 가 ``request``만 받는다).

* :func:`validate_record` — Decisions 제약(상태 ≤ 16,000자, 질문 1~12개, 이름 ``^[A-Za-z0-9_]{1,40}$``, choice 선택지 2~8개,
  score의 criteria는 문자열 목록)과 정답의 모양(choice 확률이 키를 정확히 덮고 합 1 ± 1e-6, noul·score는 [0, 1])을 본다.
  문제 문장의 목록을 돌려준다(빈 목록 = 통과). 진단 스크립트는 문제 있는 기록을 세고 건너뛴다.
* :func:`select_records` — 무작위 표본(seed 고정)을 **쌍 단위로** 뽑는다. 사실 하나만 바꾼 쌍(``meta.pair``)은 짝이 함께
  들어와야 쌍 분석이 된다(레코드 단위로 300/2,000을 뽑으면 짝이 같이 뽑히는 쌍이 몇 개 안 된다). 묶음을 섞은 순서대로
  개수가 `limit`에 닿을 때까지 통째로 담으므로 마지막 묶음이 쌍이면 `limit`보다 하나 많을 수 있고(쌍을 깨지 않는다),
  같은 seed의 작은 표본은 큰 표본의 부분집합이다(2B reasoning 100 ⊂ direct 300).
* :func:`shuffle_key` — 선택지 순서를 섞는 seed의 열쇠. 쌍의 두 레코드는 같은 열쇠(짝 id 중 작은 것)를 써서 **같은 순서**로
  보인다 — 단어만 바꾼 쌍의 흔들림에 순서 효과(위치 편향)가 섞이지 않게. ``meta.pair``는 순서만 정하고 글에는 들어가지 않는다.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import re
from pathlib import Path
from typing import Any, Iterable

__all__ = [
    "CHOICE_KINDS",
    "LABEL_SUM_TOL",
    "MAX_QUESTIONS",
    "MAX_STATE_CHARS",
    "NAME_RE",
    "OPTION_RANGE",
    "QUESTION_TYPES",
    "SCHEMA",
    "file_sha256",
    "pair_groups",
    "pair_of",
    "read_records",
    "select_records",
    "shuffle_key",
    "validate_record",
]

SCHEMA = "simgen-request/0"
NAME_RE = re.compile(r"^[A-Za-z0-9_]{1,40}$")
MAX_STATE_CHARS = 16_000
MAX_QUESTIONS = 12
OPTION_RANGE = (2, 8)
QUESTION_TYPES = ("choice", "noul", "score")
LABEL_SUM_TOL = 1e-6
#: choice 정답의 kind — 없음, v0a 엔진의 action·state·rule, 일반 능력 시험 G1의 probe(Task v0.1b — 지표에서 ``choice/other``로 따로 묶인다).
CHOICE_KINDS = (None, "action", "state", "rule", "probe")


def read_records(path: str | Path) -> list[dict[str, Any]]:
    """JSONL → 기록 목록 (빈 줄은 건너뛴다)."""
    out = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                out.append(json.loads(line))
    return out


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _is_unit(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value)) and 0.0 <= float(value) <= 1.0


def _validate_question(name: str, question: Any) -> list[str]:
    if not isinstance(question, dict):
        return [f"question {name!r}: not an object"]
    problems = []
    kind = question.get("type")
    if kind not in QUESTION_TYPES:
        return [f"question {name!r}: type {kind!r} not in {list(QUESTION_TYPES)}"]
    if not isinstance(question.get("instructions"), str) or not question["instructions"].strip():
        problems.append(f"question {name!r}: instructions must be a non-empty string")
    if kind == "choice":
        criteria = question.get("criteria")
        if not isinstance(criteria, dict):
            return problems + [f"question {name!r}: choice criteria must be an object of option key -> description"]
        low, high = OPTION_RANGE
        if not low <= len(criteria) <= high:
            problems.append(f"question {name!r}: choice needs {low}..{high} options (has {len(criteria)})")
        for key, description in criteria.items():
            if not NAME_RE.match(str(key)):
                problems.append(f"question {name!r}: option key {key!r} does not match {NAME_RE.pattern}")
            if not isinstance(description, str):
                problems.append(f"question {name!r}: option {key!r} description must be a string")
    elif kind == "score":
        criteria = question.get("criteria", [])
        if not isinstance(criteria, list) or not all(isinstance(item, str) for item in criteria):
            problems.append(f"question {name!r}: score criteria must be a list of strings")
    return problems


def _validate_label(name: str, question: dict, label: Any) -> list[str]:
    if not isinstance(label, dict):
        return [f"label {name!r}: missing or not an object"]
    kind = question.get("type")
    if label.get("type") not in (None, kind):
        return [f"label {name!r}: type {label.get('type')!r} differs from the question type {kind!r}"]
    if kind == "noul":
        return [] if _is_unit(label.get("p_yes")) else [f"label {name!r}: p_yes must be in [0, 1] (got {label.get('p_yes')!r})"]
    if kind == "score":
        return [] if _is_unit(label.get("value")) else [f"label {name!r}: value must be in [0, 1] (got {label.get('value')!r})"]
    problems = []
    keys = set(question.get("criteria") or {})
    probabilities = label.get("probabilities")
    if not isinstance(probabilities, dict):
        return [f"label {name!r}: choice probabilities must be an object"]
    if set(probabilities) != keys:
        problems.append(f"label {name!r}: probability keys {sorted(probabilities)} differ from option keys {sorted(keys)}")
    if not all(_is_unit(p) for p in probabilities.values()):
        problems.append(f"label {name!r}: every probability must be in [0, 1]")
    elif abs(sum(float(p) for p in probabilities.values()) - 1.0) > LABEL_SUM_TOL:
        problems.append(f"label {name!r}: probabilities sum to {sum(float(p) for p in probabilities.values())!r}, not 1")
    costs = label.get("expected_cost")
    if costs is not None:
        if not isinstance(costs, dict) or set(costs) != keys:
            problems.append(f"label {name!r}: expected_cost keys must equal the option keys")
        elif not all(isinstance(c, (int, float)) and not isinstance(c, bool) and math.isfinite(float(c)) for c in costs.values()):
            problems.append(f"label {name!r}: expected_cost values must be finite numbers")
    if label.get("kind") not in CHOICE_KINDS:
        problems.append(f"label {name!r}: kind {label.get('kind')!r} not in {[k for k in CHOICE_KINDS if k]}")
    return problems


def validate_record(record: dict[str, Any], *, require_labels: bool = True) -> list[str]:
    """형식 문제의 목록 (빈 목록 = 통과). `require_labels=False`면 정답 없이 요청만 본다."""
    problems: list[str] = []
    if record.get("schema") != SCHEMA:
        problems.append(f"schema {record.get('schema')!r} is not {SCHEMA!r}")
    if not isinstance(record.get("id"), str) or not record["id"]:
        problems.append("id must be a non-empty string")
    request = record.get("request")
    if not isinstance(request, dict):
        return problems + ["request must be an object"]
    state = request.get("state")
    if not isinstance(state, str) or not state.strip():
        problems.append("request.state must be a non-empty string")
    elif len(state) > MAX_STATE_CHARS:
        problems.append(f"request.state has {len(state)} characters (max {MAX_STATE_CHARS})")
    questions = request.get("questions")
    if not isinstance(questions, dict):
        return problems + ["request.questions must be an object of name -> question"]
    if not 1 <= len(questions) <= MAX_QUESTIONS:
        problems.append(f"request.questions needs 1..{MAX_QUESTIONS} questions (has {len(questions)})")
    labels = record.get("labels") if isinstance(record.get("labels"), dict) else {}
    for name, question in questions.items():
        if not NAME_RE.match(str(name)):
            problems.append(f"question name {name!r} does not match {NAME_RE.pattern}")
        own = _validate_question(str(name), question)
        problems.extend(own)
        if require_labels and not own:
            problems.extend(_validate_label(str(name), question, labels.get(name)))
    return problems


def pair_of(record: dict[str, Any]) -> tuple[str | None, str | None]:
    """(짝 id, 쌍 종류 numeric|wording) — 쌍이 아니면 (None, None)."""
    pair = (record.get("meta") or {}).get("pair")
    if not isinstance(pair, dict) or not pair.get("of"):
        return None, None
    return str(pair["of"]), (str(pair["kind"]) if pair.get("kind") else None)


def shuffle_key(record: dict[str, Any]) -> str:
    """선택지 순서를 섞는 열쇠: 쌍이면 두 id 중 작은 것(쌍이 같은 순서를 본다), 아니면 자기 id."""
    own = str(record.get("id") or "")
    partner, _ = pair_of(record)
    return min(own, partner) if partner else own


def pair_groups(records: Iterable[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    """``meta.pair``로 이어진 기록끼리 묶는다 (짝이 집합에 없으면 혼자). 묶음 안은 파일 순서, 묶음은 첫 기록의 파일 순서."""
    records = list(records)
    index = {str(r.get("id")): i for i, r in enumerate(records)}
    parent = list(range(len(records)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i, record in enumerate(records):
        partner, _ = pair_of(record)
        if partner is not None and partner in index:
            a, b = find(i), find(index[partner])
            if a != b:
                parent[max(a, b)] = min(a, b)
    groups: dict[int, list[int]] = {}
    for i in range(len(records)):
        groups.setdefault(find(i), []).append(i)
    return [[records[i] for i in members] for _, members in sorted(groups.items(), key=lambda kv: kv[1][0])]


def select_records(records: list[dict[str, Any]], limit: int | None, *, seed: int) -> list[dict[str, Any]]:
    """쌍 단위 무작위 표본 (모듈 설명). `limit`이 None이거나 전체 이상이면 전부를 파일 순서로. 결과는 파일 순서."""
    if limit is None or limit >= len(records):
        return list(records)
    groups = pair_groups(records)
    order = list(range(len(groups)))
    random.Random(int(seed)).shuffle(order)
    chosen: set[int] = set()
    count = 0
    for g in order:
        if count >= limit:
            break
        chosen.add(g)
        count += len(groups[g])
    picked = {id(record) for g in chosen for record in groups[g]}
    return [record for record in records if id(record) in picked]
