"""레코드 형식 `simgen-request/0`: 검사와 직렬화 (Task v0a 지시서 §2, 형식 문서는 이 패키지의 README.md).

한 줄 = 한 판단 순간 = 한 요청. **모델 입력은 `request`뿐**이다 — `meta`·`labels`·`truth`·`rationale`은 입력이 아니다.
Decisions 형식 제약(벤치마크·결정 모델들과 같은 꼴): `state` ≤ 16,000자; 질문 1~12개; 이름 ``^[A-Za-z0-9_]{1,40}$``;
`choice`는 선택지 2~8개(열쇠도 같은 정규식); `score`의 `criteria`는 문자열 목록. 정답: `choice` 확률은 모든 열쇠를 덮고 합 1(±1e-6),
`noul`·`score`는 [0, 1]; 행동 choice는 `expected_cost`가 같은 열쇠를 덮는다.
"""

from __future__ import annotations

import json
import re

from robo_jev.simgen import SCHEMA

__all__ = ["NAME_RE", "dumps", "validate_record"]

NAME_RE = re.compile(r"^[A-Za-z0-9_]{1,40}$")
MAX_STATE_CHARS = 16_000
TOP_KEYS = ("schema", "id", "generator", "meta", "request", "labels", "truth", "rationale")


def dumps(record: dict) -> str:
    """JSONL 한 줄(열쇠 순서 = 넣은 순서; 같은 레코드는 같은 바이트)."""
    return json.dumps(record, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def _check_question(name: str, q: dict, errors: list[str]) -> None:
    where = f"question {name}"
    if not NAME_RE.match(name):
        errors.append(f"{where}: bad name")
    qtype = q.get("type")
    if qtype not in ("choice", "noul", "score"):
        errors.append(f"{where}: bad type {qtype}")
    if not isinstance(q.get("instructions"), str) or not q["instructions"].strip():
        errors.append(f"{where}: empty instructions")
    criteria = q.get("criteria")
    if qtype == "choice":
        if not isinstance(criteria, dict) or not 2 <= len(criteria) <= 8:
            errors.append(f"{where}: choice needs 2–8 options")
        else:
            for key, text in criteria.items():
                if not NAME_RE.match(key) or not isinstance(text, str) or not text:
                    errors.append(f"{where}: bad option {key!r}")
    elif qtype == "score":
        if criteria is not None and (not isinstance(criteria, list) or not all(isinstance(t, str) for t in criteria)):
            errors.append(f"{where}: score criteria must be a list of strings")
    elif qtype == "noul" and criteria is not None:
        errors.append(f"{where}: noul takes no criteria")


def _check_label(name: str, q: dict, label: dict | None, errors: list[str]) -> None:
    where = f"label {name}"
    if label is None:
        errors.append(f"{where}: missing")
        return
    if label.get("type") != q.get("type"):
        errors.append(f"{where}: type {label.get('type')} != question type {q.get('type')}")
        return
    if label["type"] == "choice":
        probs = label.get("probabilities", {})
        if set(probs) != set(q.get("criteria", {})):
            errors.append(f"{where}: probability keys do not match options")
        if any(not 0.0 <= p <= 1.0 for p in probs.values()) or abs(sum(probs.values()) - 1.0) > 1e-6:
            errors.append(f"{where}: probabilities must be in [0, 1] and sum to 1")
        if label.get("kind") not in ("action", "state", "rule"):
            errors.append(f"{where}: bad kind {label.get('kind')}")
        if label.get("kind") == "action":
            ec = label.get("expected_cost", {})
            if set(ec) != set(probs):
                errors.append(f"{where}: expected_cost keys do not match options")
            elif probs:
                best = max(probs, key=probs.get)
                if ec[best] > min(ec.values()) + 1e-9:
                    errors.append(f"{where}: argmax is not the minimum expected cost")
    elif label["type"] == "noul":
        if not 0.0 <= label.get("p_yes", -1.0) <= 1.0:
            errors.append(f"{where}: p_yes must be in [0, 1]")
    elif label["type"] == "score":
        if not 0.0 <= label.get("value", -1.0) <= 1.0:
            errors.append(f"{where}: value must be in [0, 1]")


def validate_record(record: dict) -> list[str]:
    """형식 오류 목록(비면 통과)."""
    errors: list[str] = []
    if tuple(record) != TOP_KEYS:
        errors.append(f"top-level keys must be {TOP_KEYS}, got {tuple(record)}")
    if record.get("schema") != SCHEMA:
        errors.append(f"schema must be {SCHEMA}")
    request = record.get("request", {})
    if set(request) != {"state", "questions"}:
        errors.append("request must hold exactly state and questions")
    state = request.get("state")
    if not isinstance(state, str) or not state.strip() or len(state) > MAX_STATE_CHARS:
        errors.append("state must be a non-empty string of at most 16,000 characters")
    questions = request.get("questions", {})
    if not isinstance(questions, dict) or not 1 <= len(questions) <= 12:
        errors.append("questions must be a mapping of 1–12 entries")
        questions = {}
    for name, q in questions.items():
        _check_question(name, q, errors)
    labels = record.get("labels", {})
    if set(labels) != set(questions):
        errors.append("labels must cover exactly the questions")
    for name, q in questions.items():
        _check_label(name, q, labels.get(name), errors)
    return errors
