"""옛 줄기의 비로봇 단일 요청(`judgment-v0`) → 판단 요청 기록(`simgen-request/0`) — 일반 능력 시험 G1 (Task v0.1b; 기획서 docs/12 §10·§11·§15).

**왜.** 기획서는 학습이 기반 LLM의 일반 추론(뜻 읽기)을 해치지 않았는지 보는 시험을 요구한다(§10 "일반 판단 능력 퇴행 확인",
§11 "일반 능력 보호", §15). 재료는 옛 줄기의 비로봇 단일 요청(`artifacts/datasets/d1/single/` — spatial·dom·workflow·rules,
ko 70 %·en 30 %, 정답은 규칙 코드)이다. 새 줄기는 이 자료로 학습한 적이 없다. 이 모듈은 그것을 진단 도구
(``scripts/diagnose_decisions.py``)와 판단 학습기(:mod:`robo_jev.decisions.train`)가 **그대로** 읽는 꼴로 바꾼다.

**변환 규칙** (레코드 하나 → 판단 요청 하나 이상):

* **상태 글** (:func:`state_text`) — 구조 상태를 빠짐없이 글로: 최상위 키마다 한 줄 ``키: <JSON>``, 값이 객체의 목록이면 ``키:``
  다음 줄부터 항목마다 ``- <JSON 한 줄>``. 모든 값은 한 줄 JSON(``ensure_ascii=False``)이라 정보가 빠지지 않는다 —
  :func:`parse_state_text` 가 원래 dict로 되돌리고 검사가 모든 레코드에서 그것을 확인한다(glossary·thresholds 포함). 들여쓴 JSON보다
  토큰이 ≈ 25 % 적다(G1 265 레코드: 평균 735 대 984 토큰). 상태 글이 :data:`MAX_STATE_CHARS` (16,000자)를 넘는 레코드는 뺀다(세어 둔다).
* **질문** — ``choice`` → ``choice``(criteria 맵 = 선택지 id → 설명), ``boolean`` → ``noul``(예/아니오가 선택지라 criteria는 두지 않는다),
  ``ordinal`` → ``choice``(수준 값 순서대로). 선택지 설명에 원래 후보의 ``ref``(상태 안 항목 id)가 들어 있지 않으면 ``설명 (ref=<id>)``로
  붙인다 — 옛 직렬화가 후보 줄에 ``ref=``를 실었다(:func:`robo_jev.model.serialize.candidate_line`); workflow의 "개발 담당"(ref r1)처럼
  설명만으로는 상태의 어느 항목인지 덜 분명한 후보가 있다. 질문 이름은 원래 id 그대로(모두 ``^[A-Za-z0-9_]{1,40}$``).
* **정답이 없는 질문은 뺀다** — 원본은 규칙 코드가 답을 정할 수 없는 질문(관측 없음, "해당 없음" 후보가 없는 무정답 등)을 라벨 없이
  둔다(loss mask, 2,000 레코드에 1,961문항). 채점할 수 없으므로 요청에서 빼고 ``masked``로 센다. 프롬프트는 질문마다 하나(상태 + 그 질문)라
  다른 질문을 빼도 남은 질문의 프롬프트는 바뀌지 않는다.
* **선택지 수** — Decisions 제약은 choice 2~8개다. 1개짜리 choice는 뺀다(``too_few_options``). 9~14개(원본의 dom 후보 목록 변형)는
  **8개로 줄인다**(:func:`trim_options`): 정답(허용 집합 전부)과 ``ref`` 없는 후보("해당 없음·정보 부족")는 남기고, 나머지 오답 후보에서
  (레코드 id, 질문 id)로 정한 seed로 골라 8개를 채운다(원래 순서 유지). 정답은 상태와 지시에서 정해지므로 오답 후보를 빼도 그대로다.
  줄인 질문은 ``meta.questions[q].options_trimmed = [원래 수, 8]``에 적는다.
* **질문 수** — 한 요청은 1~12개다. 남은 질문이 12개를 넘으면 같은 상태로 레코드를 고르게 나눈다(16 → 8 + 8; id ``<원본 id>.p<k>``).
  라벨 있는 질문이 하나도 없는 레코드는 뺀다(``no_labeled_question``).
* **정답** — ``single``: choice는 one-hot, noul은 ``p_yes`` ∈ {0, 1}. ``valid_set``(복수 허용 — 경계 허용 오차 등): 허용 집합에 고르게
  나눈다(진단의 정확도는 argmax가 최선 집합 = 허용 집합 안이면 맞음). choice 정답의 ``kind``는 :data:`LABEL_KIND` (``probe``) —
  :func:`robo_jev.decisions.metrics.block_of` 가 ``choice/other``로 보내 T1~T7의 묶음(action·state·rule)과 섞이지 않는다.
* **meta** — ``domain``, ``language``, ``origin_group``, ``split``, ``source_record``(원본 request id), ``part``·``parts``, ``template``,
  ``variants``, ``holdout``, ``concepts``, ``contrast``(원본의 대조 sibling 정보 — ``pair``로 옮기지 않는다: 쌍 분석의 대상이 아니다),
  ``questions[q]`` = {``source_type``(원래 종류), ``source``(규칙 id), ``family``(= 규칙 id — 진단의 질문 계열 묶음), ``label_kind``,
  ``label_confidence``, ``options_trimmed``}. 모델 입력은 ``request``뿐이다(진단·학습 모두 — :mod:`robo_jev.decisions.prompt`).

**분할** (:data:`G1_SPLITS`, :data:`MIX_SPLITS`): 봉인 G1 = 원본의 ``test`` + ``ood_test``(학습에 쓰지 않는다); 섞기용 = ``train``만
(dev·calibration·ood_dev는 쓰지 않는다). 원본 분할은 origin group 단위라 두 파일의 origin group이 겹치지 않는다(검사 :func:`overlap`).
"""

from __future__ import annotations

import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from typing import Any, Iterable, Sequence

import numpy as np

from robo_jev.decisions.records import MAX_QUESTIONS, MAX_STATE_CHARS, NAME_RE, OPTION_RANGE, SCHEMA

__all__ = [
    "CLUSTERS",
    "CONVERTER_VERSION",
    "G1_SPLITS",
    "LABEL_KIND",
    "MIX_SPLITS",
    "PROBE_DIMENSIONS",
    "PROBE_METRICS",
    "SOURCE_SCHEMA",
    "TYPE_MAP",
    "compare_probe",
    "convert_record",
    "convert_records",
    "overlap",
    "parse_state_text",
    "partition",
    "probe_values",
    "render_probe_comparison",
    "sample_and_convert",
    "state_text",
    "summarize",
    "trim_options",
]

SOURCE_SCHEMA = "judgment-v0"
CONVERTER_VERSION = "single-probe-v0"
#: choice 정답의 kind — 진단 지표에서 ``choice/other``로 따로 묶인다.
LABEL_KIND = "probe"
TYPE_MAP = {"choice": "choice", "boolean": "noul", "ordinal": "choice"}
#: 봉인 G1의 원본 분할 (학습에 쓰지 않는다).
G1_SPLITS = ("test", "ood_test")
#: 섞기용의 원본 분할.
MIX_SPLITS = ("train",)
MAX_OPTIONS = OPTION_RANGE[1]


# --------------------------------------------------------------------------
# 상태 글
# --------------------------------------------------------------------------


def _line(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False)


def state_text(state: dict[str, Any]) -> str:
    """구조 상태 → 글: 최상위 키마다 ``키: <JSON 한 줄>``; 값이 (비지 않은) 객체 목록이면 ``키:`` 뒤에 항목마다 ``- <JSON 한 줄>``.

    :func:`parse_state_text` 가 정확히 되돌린다(정보가 빠지지 않는다)."""
    if not isinstance(state, dict) or not state:
        raise ValueError("state: 비지 않은 객체여야 한다")
    lines: list[str] = []
    for key, value in state.items():
        name = str(key)
        if not name or "\n" in name or ": " in name or name.endswith(":") or name.startswith("- "):
            raise ValueError(f"state 키 {name!r}는 한 줄 글의 키로 쓸 수 없다")
        if isinstance(value, list) and value and all(isinstance(item, dict) for item in value):
            lines.append(f"{name}:")
            lines.extend(f"- {_line(item)}" for item in value)
        else:
            lines.append(f"{name}: {_line(value)}")
    return "\n".join(lines)


def parse_state_text(text: str) -> dict[str, Any]:
    """:func:`state_text` 의 역 — 글 → 원래 dict (검사용: 정보가 빠지지 않았음을 보인다)."""
    out: dict[str, Any] = {}
    current: str | None = None
    for line in text.split("\n"):
        if line.startswith("- "):
            if current is None:
                raise ValueError(f"목록 항목 앞에 키가 없다: {line[:40]!r}")
            out[current].append(json.loads(line[2:]))
            continue
        if line.endswith(":") and ": " not in line:
            current = line[:-1]
            out[current] = []
            continue
        key, separator, value = line.partition(": ")
        if not separator:
            raise ValueError(f"읽을 수 없는 줄: {line[:40]!r}")
        out[key] = json.loads(value)
        current = None
    return out


# --------------------------------------------------------------------------
# 질문·정답
# --------------------------------------------------------------------------


def _seeded(*parts: Any) -> random.Random:
    digest = hashlib.sha256("|".join(str(p) for p in parts).encode("utf-8")).digest()
    return random.Random(int.from_bytes(digest[:8], "big"))


def _correct_ids(label: dict[str, Any]) -> list[str]:
    """정답 후보 id 목록 (single은 하나, valid_set은 허용 집합 — 원래 순서, 겹침 없이)."""
    if label.get("kind") == "single":
        return [str(label["answer"])]
    if label.get("kind") == "valid_set":
        return list(dict.fromkeys(str(c) for c in label.get("candidate_ids") or []))
    raise ValueError(f"알 수 없는 정답 kind {label.get('kind')!r}")


def trim_options(criteria: Sequence[dict[str, Any]], keep: Iterable[str], *, limit: int, rng: random.Random) -> list[dict[str, Any]]:
    """후보 목록을 `limit`개로: `keep`(정답)과 ``ref`` 없는 후보("해당 없음")는 남기고, 나머지에서 `rng`로 골라 채운다. 원래 순서를 지킨다.

    남겨야 할 후보가 `limit`보다 많으면 ``ValueError``."""
    if len(criteria) <= limit:
        return list(criteria)
    must = set(keep) | {str(c["id"]) for c in criteria if "ref" not in c}
    if len(must) > limit:
        raise ValueError(f"남겨야 할 후보 {len(must)}개 > {limit}")
    rest = [str(c["id"]) for c in criteria if str(c["id"]) not in must]
    chosen = must | set(rng.sample(rest, limit - len(must)))
    return [c for c in criteria if str(c["id"]) in chosen]


def _description(criterion: dict[str, Any]) -> str:
    text = " ".join(str(criterion.get("description") or "").split())
    ref = criterion.get("ref")
    if ref is not None and str(ref) not in text:
        text = f"{text} (ref={ref})" if text else f"ref={ref}"
    return text


def _convert_question(record_id: str, question: dict[str, Any], label: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """원본 질문 하나와 그 정답 → (판단 질문, 판단 정답, 질문 meta). 변환할 수 없으면 ``_Skip``."""
    source_type = str(question.get("type"))
    if source_type not in TYPE_MAP:
        raise _Skip("unsupported_type")
    qmeta: dict[str, Any] = {
        "source_type": source_type, "source": label.get("source"), "family": label.get("source"), "label_kind": label.get("kind"),
        "label_confidence": label.get("label_confidence"), "options_trimmed": None,
    }  # fmt: skip
    instructions = " ".join(str(question.get("instructions") or "").split())
    if not instructions:
        raise _Skip("no_instructions")
    if source_type == "boolean":
        if label.get("kind") != "single" or not isinstance(label.get("answer"), bool):
            raise _Skip("unsupported_label")
        return {"type": "noul", "instructions": instructions}, {"type": "noul", "p_yes": 1.0 if label["answer"] else 0.0}, qmeta
    criteria = list(question.get("criteria") or [])
    if source_type == "ordinal":
        criteria = sorted(criteria, key=lambda c: float(c.get("value", 0.0)))  # 수준 값 순서 (원본도 이미 그 순서다)
    if len(criteria) < OPTION_RANGE[0]:
        raise _Skip("too_few_options")
    correct = _correct_ids(label)
    ids = [str(c["id"]) for c in criteria]
    if len(set(ids)) != len(ids) or not all(NAME_RE.match(i) for i in ids):
        raise _Skip("bad_option_ids")
    if not correct or any(c not in ids for c in correct):
        raise _Skip("label_not_in_options")
    if len(criteria) > MAX_OPTIONS:
        try:
            kept = trim_options(criteria, correct, limit=MAX_OPTIONS, rng=_seeded(record_id, question["id"], "trim"))
        except ValueError:
            raise _Skip("too_many_required_options") from None
        qmeta["options_trimmed"] = [len(criteria), len(kept)]
        criteria = kept
    options = {str(c["id"]): _description(c) for c in criteria}
    share = 1.0 / len(correct)
    probabilities = {key: (share if key in correct else 0.0) for key in options}
    return (
        {"type": "choice", "instructions": instructions, "criteria": options},
        {"type": "choice", "kind": LABEL_KIND, "probabilities": probabilities},
        qmeta,
    )


class _Skip(Exception):
    """이 질문(또는 레코드)은 변환하지 않는다 — 이유는 args[0]."""


def partition(names: Sequence[str], *, limit: int = MAX_QUESTIONS) -> list[list[str]]:
    """질문 이름들을 원래 순서대로 ``ceil(n / limit)``개의 고른 묶음으로 (16 → 8 + 8, 13 → 7 + 6)."""
    n = len(names)
    if n == 0:
        return []
    parts = math.ceil(n / limit)
    base, extra = divmod(n, parts)
    out, start = [], 0
    for k in range(parts):
        size = base + (1 if k < extra else 0)
        out.append(list(names[start : start + size]))
        start += size
    return out


# --------------------------------------------------------------------------
# 레코드
# --------------------------------------------------------------------------


def convert_record(record: dict[str, Any]) -> tuple[list[dict[str, Any]], Counter]:
    """`judgment-v0` 레코드 하나 → (판단 요청 기록들, 센 것). 센 것: ``questions_in``·``questions_out``·질문을 뺀 이유(``q:<이유>``)·
    레코드를 뺀 이유(``r:<이유>``)·``trimmed``·``parts``. 모듈 설명의 규칙."""
    counts: Counter = Counter()
    if record.get("schema_version") != SOURCE_SCHEMA:
        raise ValueError(f"schema_version {record.get('schema_version')!r} is not {SOURCE_SCHEMA!r}")
    request = record["request"]
    provenance = record.get("provenance") or {}
    rid = str(request["request_id"])
    labels = {str(label["question_id"]): label for label in record.get("labels") or []}
    questions: dict[str, dict[str, Any]] = {}
    answers: dict[str, dict[str, Any]] = {}
    qmeta: dict[str, dict[str, Any]] = {}
    for question in request.get("questions") or []:
        name = str(question["id"])
        counts["questions_in"] += 1
        label = labels.get(name)
        if label is None:
            counts["q:masked"] += 1
            continue
        if not NAME_RE.match(name):
            counts["q:bad_name"] += 1
            continue
        try:
            q, a, m = _convert_question(rid, question, label)
        except _Skip as skip:
            counts[f"q:{skip.args[0]}"] += 1
            continue
        questions[name], answers[name], qmeta[name] = q, a, m
        counts["trimmed"] += 1 if m["options_trimmed"] else 0
    if not questions:
        counts["r:no_labeled_question"] += 1
        return [], counts
    text = state_text(request["state"])
    if len(text) > MAX_STATE_CHARS:
        counts["r:state_too_long"] += 1
        return [], counts
    groups = partition(list(questions))
    out = []
    for k, names in enumerate(groups):
        out.append({
            "schema": SCHEMA,
            "id": rid if len(groups) == 1 else f"{rid}.p{k + 1}",
            "generator": {"converter": CONVERTER_VERSION, "source_schema": SOURCE_SCHEMA, "source_generator": provenance.get("generator"),
                          "source_config": provenance.get("config_version"), "source_seed": provenance.get("seed")},
            "meta": {
                "domain": provenance.get("domain"), "language": provenance.get("language"), "origin_group": record.get("origin_group"),
                "split": record.get("split"), "source_record": rid, "part": k + 1, "parts": len(groups), "template": provenance.get("template"),
                "variants": provenance.get("variants"), "holdout": provenance.get("holdout"), "concepts": provenance.get("concepts"),
                "contrast": provenance.get("contrast"), "questions": {name: qmeta[name] for name in names},
            },
            "request": {"state": text, "questions": {name: questions[name] for name in names}},
            "labels": {name: answers[name] for name in names},
        })  # fmt: skip
    counts["questions_out"] += len(questions)
    counts["parts"] += len(groups) if len(groups) > 1 else 0
    return out, counts


def convert_records(records: Iterable[dict[str, Any]], *, splits: Sequence[str]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """원본 레코드들 중 `splits`에 든 것만 변환 → (판단 요청 기록들 — 원본 파일 순서, 요약 수)."""
    wanted = set(splits)
    out: list[dict[str, Any]] = []
    counts: Counter = Counter()
    source_records = 0
    for record in records:
        if record.get("split") not in wanted:
            continue
        source_records += 1
        converted, found = convert_record(record)
        counts.update(found)
        out.extend(converted)
    return out, {"source_records": source_records, **summarize(out), "conversion": dict(sorted(counts.items()))}


def sample_and_convert(sources: dict[str, Sequence[dict[str, Any]]], *, per_source: int, seed: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """원본 묶음마다 seed로 `per_source`개를 무작위로 뽑아(묶음 이름과 seed에서 정한 순열 — 원본 파일 순서로 되돌린다) 변환한다 — 일반 능력 시험
    G2(공개 BoolQ dev·MultiNLI dev_matched, 통제자 결정 2026-10-12). 분할은 거르지 않는다(공개 세트는 dev 하나다). 돌려주는 것: (변환본 —
    묶음 순서, 묶음 안은 원본 순서), 묶음마다 {원본 수, 뽑은 수, 뽑은 request id, 변환 센 것}."""
    out: list[dict[str, Any]] = []
    picked: dict[str, Any] = {}
    for name, records in sources.items():
        records = list(records)
        if per_source > len(records):
            raise ValueError(f"{name}: {per_source}개를 뽑을 수 없다 (원본 {len(records)}개)")
        index = sorted(_seeded(seed, name, "sample").sample(range(len(records)), per_source))
        counts: Counter = Counter()
        for i in index:
            converted, found = convert_record(records[i])
            counts.update(found)
            out.extend(converted)
        picked[name] = {"records_total": len(records), "sampled": len(index), "request_ids": [str(records[i]["request"]["request_id"]) for i in index],
                        "conversion": dict(sorted(counts.items()))}  # fmt: skip
    return out, picked


def summarize(converted: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """변환본의 구성: 레코드·질문 수, 원본 레코드·origin group 수, 분할·분야·언어·종류(원래 종류와 판단 종류)·정답 kind·신뢰도별 질문 수,
    분할·분야·언어별 레코드 수."""
    by: dict[str, Counter] = {k: Counter() for k in ("split", "domain", "language", "source_type", "type", "label_kind", "label_confidence")}
    records_by: dict[str, Counter] = {k: Counter() for k in ("split", "domain", "language")}
    cross: Counter = Counter()
    trimmed = 0
    for record in converted:
        meta = record["meta"]
        for key in records_by:
            records_by[key][str(meta.get(key))] += 1
        for name, question in record["request"]["questions"].items():
            qm = meta["questions"][name]
            by["split"][str(meta.get("split"))] += 1
            by["domain"][str(meta.get("domain"))] += 1
            by["language"][str(meta.get("language"))] += 1
            by["source_type"][str(qm.get("source_type"))] += 1
            by["type"][str(question["type"])] += 1
            by["label_kind"][str(qm.get("label_kind"))] += 1
            by["label_confidence"][str(qm.get("label_confidence"))] += 1
            cross[f"{meta.get('domain')}/{qm.get('source_type')}"] += 1
            trimmed += 1 if qm.get("options_trimmed") else 0
    return {
        "records": len(converted), "questions": sum(by["type"].values()),
        "source_records_kept": len({r["meta"]["source_record"] for r in converted}),
        "origin_groups": len({r["meta"]["origin_group"] for r in converted}), "trimmed_questions": trimmed,
        "questions_by": {key: dict(sorted(counter.items())) for key, counter in by.items()},
        "questions_by_domain_and_type": dict(sorted(cross.items())),
        "records_by": {key: dict(sorted(counter.items())) for key, counter in records_by.items()},
    }


def overlap(a: Iterable[dict[str, Any]], b: Iterable[dict[str, Any]]) -> dict[str, list[str]]:
    """두 변환본이 겹치는 origin group·원본 레코드 (봉인 G1과 섞기용은 둘 다 비어야 한다)."""
    a, b = list(a), list(b)
    groups = {r["meta"]["origin_group"] for r in a} & {r["meta"]["origin_group"] for r in b}
    sources = {r["meta"]["source_record"] for r in a} & {r["meta"]["source_record"] for r in b}
    return {"origin_groups": sorted(groups), "source_records": sorted(sources)}


# --------------------------------------------------------------------------
# 두 run의 묶음별 짝지은 비교 (§1-3 — v0 − 학습 없는 2B)
# --------------------------------------------------------------------------

#: 묶음 차원 — meta의 값(``source_type``은 원래 질문 종류 choice·boolean·ordinal, ``type``은 판단 종류 choice·noul).
PROBE_DIMENSIONS = ("domain", "language", "source_type", "split", "label_kind", "label_confidence", "type")
#: 질문마다 재는 값: 정확도(choice = argmax가 최선 집합 안; noul = 0.5의 같은 쪽 — 정답이 0/1이라 맞힘과 같다), Brier(진단의 정의 그대로 —
#: choice Σ_k (p_k − q_k)², noul (p − y)²), 코드 질량(답 자리의 전체 어휘 softmax가 코드에 준 몫 — 형식 준수).
PROBE_METRICS = ("accuracy", "brier", "code_mass")
#: 뽑기 단위: origin group(번역·재표현·대조 sibling을 한 단위로 — 기본), 원본 레코드, 변환 레코드.
CLUSTERS = ("origin_group", "source_record", "record")


def _question_meta(record: dict[str, Any], name: str) -> dict[str, Any]:
    return ((record.get("meta") or {}).get("questions") or {}).get(name) or {}


def probe_values(records: Sequence[dict[str, Any]], predictions: Sequence[dict[str, Any]], *, reading: str) -> dict[tuple[str, str], dict[str, Any]]:
    """(레코드 id, 질문) → 그 질문의 값과 묶음: ``accuracy``·``brier``·``code_mass``(한 순서 읽기는 배치들의 1/L 가중 평균 = 한 순서로
    물었을 때의 기대값, 순서 평균 읽기는 배치 평균 답의 값), ``chance``(무작위로 하나 고르기의 정확도 — choice |최선|/n, noul 0.5),
    ``groups``(:data:`PROBE_DIMENSIONS`), ``clusters``(:data:`CLUSTERS`). 행은 :func:`robo_jev.decisions.metrics.question_rows` 의 것이다."""
    from robo_jev.decisions.metrics import FLIP_THRESHOLD, question_rows

    by_id = {str(r.get("id")): r for r in records}
    sums: dict[tuple[str, str], dict[str, list[float]]] = defaultdict(lambda: defaultdict(lambda: [0.0, 0.0]))
    for row in question_rows(list(records), list(predictions), reading=reading):
        if row.get("m") is None:
            continue
        m, w = row["m"], float(row["w"])
        correct = m["match"] if row["type"] == "choice" else (m["p"] >= FLIP_THRESHOLD) == (m["y"] >= FLIP_THRESHOLD)
        chance = row["u"]["match"] if row["type"] == "choice" else 0.5
        acc = sums[(row["record_id"], row["question"])]
        for key, value in (("accuracy", 1.0 if correct else 0.0), ("brier", m["brier"]), ("code_mass", row.get("code_mass")), ("chance", chance)):
            if value is not None:
                acc[key][0] += w * float(value)
                acc[key][1] += w
    out = {}
    for (rid, name), acc in sums.items():
        record = by_id[rid]
        meta = record.get("meta") or {}
        qmeta = _question_meta(record, name)
        values = {key: (total / weight if weight > 0 else None) for key, (total, weight) in acc.items()}
        out[(rid, name)] = {
            **{key: values.get(key) for key in (*PROBE_METRICS, "chance")},
            "groups": {
                "domain": str(meta.get("domain")), "language": str(meta.get("language")), "split": str(meta.get("split")),
                "source_type": str(qmeta.get("source_type")), "label_kind": str(qmeta.get("label_kind")),
                "label_confidence": str(qmeta.get("label_confidence")), "type": str(record["request"]["questions"][name]["type"]),
            },
            "clusters": {"origin_group": str(meta.get("origin_group") or rid), "source_record": str(meta.get("source_record") or rid), "record": rid},
        }  # fmt: skip
    return out


def _paired_cell(keys: list[tuple[str, str]], a: dict, b: dict, *, cluster: str, n_boot: int, rng_seed: int) -> dict[str, Any]:
    """질문 열쇠들 위에서 지표마다 a·b의 평균과 a − b의 짝지은 cluster bootstrap 95 % 구간 (두 run이 같은 질문·같은 뽑기)."""
    clusters = sorted({a[k]["clusters"][cluster] for k in keys})
    index = {c: i for i, c in enumerate(clusters)}
    cell: dict[str, Any] = {"questions": len(keys), "clusters": len(clusters)}
    chance = [a[k]["chance"] for k in keys if a[k].get("chance") is not None]
    cell["chance_accuracy"] = float(np.mean(chance)) if chance else None
    rng = np.random.default_rng(int(rng_seed))
    draws = rng.integers(0, len(clusters), size=(int(n_boot), len(clusters))) if clusters else None
    for metric in PROBE_METRICS:
        sa, sb, n = np.zeros(len(clusters)), np.zeros(len(clusters)), np.zeros(len(clusters))
        for k in keys:
            va, vb = a[k].get(metric), b[k].get(metric)
            if va is None or vb is None:
                continue
            i = index[a[k]["clusters"][cluster]]
            sa[i] += va
            sb[i] += vb
            n[i] += 1
        if n.sum() == 0 or draws is None:
            cell[metric] = None
            continue
        value_a, value_b = float(sa.sum() / n.sum()), float(sb.sum() / n.sum())
        with np.errstate(invalid="ignore", divide="ignore"):
            samples = (sa[draws].sum(1) - sb[draws].sum(1)) / n[draws].sum(1)
        samples = samples[np.isfinite(samples)]
        low, high = (float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))) if samples.size else (None, None)
        cell[metric] = {
            "a": value_a, "b": value_b, "diff": value_a - value_b, "low": low, "high": high, "n": int(n.sum()),
            "excludes_zero": bool(low is not None and (low > 0 or high < 0)),
        }
    return cell


def compare_probe(
    records: Sequence[dict[str, Any]], predictions_a: Sequence[dict[str, Any]], predictions_b: Sequence[dict[str, Any]], *,
    dimensions: Sequence[str] = PROBE_DIMENSIONS, cluster: str = "origin_group", n_boot: int = 2000, seed: int = 0,
) -> dict[str, Any]:
    """같은 기록의 두 run(a, b) — 두 읽기 각각에서 둘 다 채점된 질문만으로, 전체(``_all``)와 묶음 값마다 정확도·Brier·코드 질량의
    a·b 평균과 a − b의 짝지은 bootstrap 95 % 구간. 차원을 ``+``로 이으면 교차 묶음이다(예: ``split+source_type`` → ``ood_test/choice``). 뽑기 단위는 `cluster`(기본 origin group — 번역·재표현·대조 sibling이 같은 계보라
    레코드보다 보수적이다); 칸마다 같은 seed로 뽑고, 한 칸의 세 지표는 같은 뽑기를 쓴다."""
    from robo_jev.decisions.metrics import READINGS

    if cluster not in CLUSTERS:
        raise ValueError(f"cluster: {list(CLUSTERS)} 중 하나 (받은 값: {cluster!r})")
    unknown = [d for d in dimensions if not d or any(part not in PROBE_DIMENSIONS for part in d.split("+"))]
    if unknown:
        raise ValueError(f"모르는 묶음 차원 {unknown} (쓸 수 있는 것: {list(PROBE_DIMENSIONS)})")
    ids = {str(p["id"]) for p in predictions_a} & {str(p["id"]) for p in predictions_b}
    kept = [r for r in records if str(r.get("id")) in ids]
    out: dict[str, Any] = {"records": len(kept), "cluster": cluster, "n_boot": int(n_boot), "seed": int(seed), "dimensions": list(dimensions), "readings": {}}
    for reading in READINGS:
        a = probe_values(kept, predictions_a, reading=reading)
        b = probe_values(kept, predictions_b, reading=reading)
        common = sorted(set(a) & set(b))
        table: dict[str, Any] = {"_all": _paired_cell(common, a, b, cluster=cluster, n_boot=n_boot, rng_seed=seed)}
        for dim in dimensions:
            key = {k: "/".join(a[k]["groups"][part] for part in dim.split("+")) for k in common}
            table[dim] = {v: _paired_cell([k for k in common if key[k] == v], a, b, cluster=cluster, n_boot=n_boot, rng_seed=seed)
                          for v in sorted(set(key.values()))}  # fmt: skip
        out["readings"][reading] = {"questions": len(common), "only_a": len(set(a) - set(b)), "only_b": len(set(b) - set(a)), "table": table}
    return out


def _fmt(value: float | None, digits: int = 3) -> str:
    return "—" if value is None else f"{value:.{digits}f}"


def _diff(cell: dict[str, Any] | None, digits: int = 3) -> str:
    if not cell:
        return "—"
    mark = " *" if cell["excludes_zero"] else ""
    return f"{cell['diff']:+.{digits}f} [{_fmt(cell['low'], digits)}, {_fmt(cell['high'], digits)}]{mark}"


def render_probe_comparison(comparison: dict[str, Any], *, label_a: str, label_b: str, title: str, note: str | None = None) -> str:
    """:func:`compare_probe` → markdown: 읽기마다 한 표(전체, 묶음 차원마다 값), 열 = 질문/cluster 수, 정확도(a, b, a − b [95 %]),
    Brier(a, b, a − b), 코드 질량(a, b, a − b). ``*`` = 구간이 0을 빼는 칸."""
    reading_names = {"single_order": "single order (deployment condition)", "order_averaged": "order-averaged (what it knows)"}
    lines = [f"# {title}", ""]
    if note:
        lines += [f"> {note}", ""]
    lines += [
        f"a = **{label_a}**, b = **{label_b}**. Paired differences a − b on the same questions with a {comparison['cluster']}-cluster bootstrap "
        f"95 % interval ({comparison['n_boot']} draws, seed {comparison['seed']} in every cell; the three metrics of a cell share its draws); "
        f"`*` = the interval excludes 0. "
        f"Accuracy: choice = argmax in the label's best set (valid sets count every allowed option), noul = same side of 0.5 as the 0/1 label; "
        f"chance = a uniform random pick. Brier as in the diagnosis tool: choice Σ_k (p_k − q_k)², noul (p − y)² (lower is better; the `_all` row "
        f"mixes the two definitions). Code mass = full-vocabulary probability on the answer codes at the answer position (format adherence).",
        f"Records compared: {comparison['records']}.", "",
    ]
    for reading, block in comparison["readings"].items():
        lines += [f"## {reading_names.get(reading, reading)} — {block['questions']} questions", ""]
        lines.append(f"| group | questions / clusters | chance acc | acc {label_a} | acc {label_b} | Δ acc [95 %] | Brier {label_a} | Brier {label_b} | Δ Brier [95 %] "
                     f"| mass {label_a} | mass {label_b} | Δ mass [95 %] |")
        lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|")

        def row(name: str, cell: dict[str, Any]) -> str:
            acc, brier, mass = cell.get("accuracy"), cell.get("brier"), cell.get("code_mass")
            return (f"| {name} | {cell['questions']} / {cell['clusters']} | {_fmt(cell.get('chance_accuracy'))} | {_fmt(acc and acc['a'])} | "
                    f"{_fmt(acc and acc['b'])} | {_diff(acc)} | {_fmt(brier and brier['a'])} | {_fmt(brier and brier['b'])} | {_diff(brier)} | "
                    f"{_fmt(mass and mass['a'])} | {_fmt(mass and mass['b'])} | {_diff(mass)} |")

        lines.append(row("**all**", block["table"]["_all"]))
        for dim in comparison["dimensions"]:
            for value, cell in block["table"][dim].items():
                lines.append(row(f"{dim} = {value}", cell))
        lines.append("")
    return "\n".join(lines) + "\n"
