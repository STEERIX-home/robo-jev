"""데이터셋 자동 QA (docs/04 §6).

    python -m robo_jev.data.validate --dataset artifacts/datasets/d1 --report artifacts/reports/d1-qa.json

:func:`validate_dataset`은 레코드 목록을 받아 **보고서 dict**를 낸다. 처음 위반에서 멈추지
않고 전부 모은다 — 한 번 돌려 고칠 목록을 얻는 것이 목적이다.

보는 것:

* 계약 위반 (:func:`robo_jev.contracts.validate_record`) — schema, 없는 정답 id, 확률 합·범위,
  ordinal 순서, 실행 이력과 라벨의 혼동, `action_ref`·국면 참조까지 여기서 걸린다.
* 정보 경계 — :func:`robo_jev.contracts.model_input`이 낸 입력에 비입력 필드 이름이 없는가.
* 후보 참조 — 후보의 `ref`가 상태에 실제로 있는 요소를 가리키는가.
* 라벨 출처 — 규칙·전문가 이름이 붙어 있는가.
* 계보 충돌 — 한 origin group이 두 split에 걸쳐 있는가, 파생본의 부모가 다른 group인가,
  표현·순서만 바꿨다는 파생본의 사실이 부모와 다른가.
* 중복 계보 — 표현을 걷어낸 **정규화된 사실**이 다른 group과 같은가.
* 정답 위치 편향, 분포 라벨의 합.

집계는 요청 수·질문 수·에피소드 수·틱 수·split별 원본 group 수를 모두 낸다.
위반 건수가 0이어야 배포할 데이터 버전으로 동결할 수 있다 (CLI는 그때만 0을 돌려준다).

위반은 `{"index", "path", "message"}`로 적는다. `index`가 `-1`이면 한 레코드가 아니라
데이터셋 전체 수준의 위반이다 (split 충돌, 정답 위치 편향).

스트림 레코드는 계약 검사·집계만 한다. 후보의 `ref`는 단일 요청에서만 상태 요소를
가리키기로 했으므로(스트림의 경유점 같은 하네스 참조는 상태 밖에 있다) 참조 검사는
`judgment-v0`에만 건다.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

from robo_jev.contracts import (
    FORBIDDEN_REQUEST_KEYS,
    PROB_SUM_TOL,
    QUESTION_SET_V0,
    SCHEMA_SINGLE_REQUEST,
    SCHEMA_STREAM,
    model_input,
    validate_record,
)

__all__ = ["REPORT_VERSION", "main", "validate_dataset"]

REPORT_VERSION = "qa-v0"

#: 정규화에서 걷어낼 **표현** 키. 남는 것이 사실이다.
WORDING_KEYS = (
    "text",
    "desc",
    "description",
    "name",
    "label",
    "aria_label",
    "glossary",
    "note",
    "title",
)

#: 표현·순서만 바꾼 파생본. 부모와 **정규화된 사실**이 같아야 한다.
FACT_PRESERVING_DERIVATIONS = ("paraphrase", "reorder")

#: 정답 위치 편향의 허용 초과분과 최소 표본 수.
#:
#: 표본이 적으면 균등해도 한 자리가 우연히 튄다 — 후보 4개·표본 35이면 한 자리의
#: 표준편차가 0.073이라 0.15 초과가 흔하다. 200 표본에서는 표준편차가 0.031이므로
#: 0.15 초과가 표준편차 4배가 넘는 사건이 된다. 그 아래 층은 집계만 하고 판정하지 않는다.
POSITION_BIAS_TOLERANCE = 0.15
POSITION_BIAS_MIN_SAMPLES = 200


def _error(index: int, path: str, message: str) -> dict:
    return {"index": index, "path": path, "message": message}


def _split_path(error: ValueError) -> tuple[str, str]:
    """계약 오류 문자열 `"path: message"`를 경로와 설명으로 나눈다."""
    text = str(error)
    path, separator, message = text.partition(": ")
    return (path, message) if separator else ("record", text)


def _collect_ids(node: Any, into: set[str]) -> None:
    if isinstance(node, dict):
        value = node.get("id")
        if isinstance(value, str):
            into.add(value)
        for child in node.values():
            _collect_ids(child, into)
    elif isinstance(node, list):
        for child in node:
            _collect_ids(child, into)


def _strip_wording(node: Any) -> Any:
    if isinstance(node, dict):
        return {
            key: _strip_wording(value)
            for key, value in node.items()
            if key not in WORDING_KEYS
        }
    if isinstance(node, list):
        return [_strip_wording(value) for value in node]
    return node


def _fact_key(state: Any) -> str | None:
    """표현을 걷어낸 사실의 정규 문자열. 번역·재표현본은 같은 값을 낸다."""
    if not isinstance(state, dict):
        return None
    return json.dumps(
        _strip_wording(state), ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )


def _scan_keys(node: Any, path: str, forbidden: Iterable[str], found: list[tuple[str, str]]) -> None:
    if isinstance(node, dict):
        for key, value in node.items():
            child = f"{path}.{key}"
            if key in forbidden:
                found.append((child, key))
            _scan_keys(value, child, forbidden, found)
    elif isinstance(node, list):
        for index, value in enumerate(node):
            _scan_keys(value, f"{path}[{index}]", forbidden, found)


def _posed_questions(tick: Any) -> int:
    """그 틱이 실제로 던진 질문 수.

    정적 후보를 가진 질문은 언제나 던지고, 후보가 틱마다 바뀌는 질문(`q_main`·`q_path`)은
    그 틱의 후보 목록에 있을 때만 던진다 — 계약의 후보 해석과 같은 규칙이다. 틱 수 × 10으로
    세면 후보가 없어 던지지 않은 질문까지 세게 된다.
    """
    request = tick.get("request") if isinstance(tick, dict) else None
    candidates = request.get("candidates") if isinstance(request, dict) else None
    if not isinstance(candidates, dict):
        return 0
    return sum(
        1
        for question_id, spec in QUESTION_SET_V0.items()
        if spec["criteria"] or question_id in candidates
    )


def validate_dataset(records: Sequence[dict]) -> dict:
    """데이터셋 QA 보고서. 위반은 모두 모으고 집계를 함께 낸다."""
    errors: list[dict] = []
    invalid = 0
    states = episodes = ticks = questions = labels = 0
    masked_questions = 0
    distribution_labels = 0
    labels_without_source = 0
    domains: Counter = Counter()
    question_types: Counter = Counter()
    label_kinds: Counter = Counter()
    split_records: Counter = Counter()
    variants: Counter = Counter()
    group_splits: dict[str, set[str]] = defaultdict(set)
    group_of_request: dict[str, str] = {}
    facts: dict[str, list[tuple[int, str]]] = defaultdict(list)
    positions: dict[int, Counter] = defaultdict(Counter)
    derived: list[dict] = []  # 파생본의 계보 (index, group, parent, derivation, facts)
    facts_of_request: dict[str, str | None] = {}

    for index, record in enumerate(records):
        try:
            validate_record(record)
        except ValueError as error:
            invalid += 1
            path, message = _split_path(error)
            errors.append(_error(index, path, message))

        # 정보 경계는 계약을 어긴 레코드에서도 본다 (serializer의 이중 방어).
        try:
            served = model_input(record)
        except ValueError as error:
            path, message = _split_path(error)
            errors.append(_error(index, f"model_input.{path}", f"모델 입력을 만들 수 없다: {message}"))
        else:
            found: list[tuple[str, str]] = []
            _scan_keys(served, "model_input", FORBIDDEN_REQUEST_KEYS, found)
            for leaked_path, key in found:
                errors.append(
                    _error(index, leaked_path, f"모델 입력에 비입력 필드 이름이 들어 있다: {key!r}")
                )

        if not isinstance(record, dict):
            continue
        schema_version = record.get("schema_version")
        group = record.get("origin_group")
        split = record.get("split")
        if isinstance(group, str) and isinstance(split, str):
            group_splits[group].add(split)
        if isinstance(split, str):
            split_records[split] += 1

        provenance = record.get("provenance") if isinstance(record.get("provenance"), dict) else {}
        for tag in provenance.get("variants") or ():
            variants[tag] += 1

        if schema_version == SCHEMA_STREAM:
            episodes += 1
            domains[provenance.get("domain") or "robot"] += 1
            tick_list = record.get("ticks")
            if isinstance(tick_list, list):
                ticks += len(tick_list)
                for tick in tick_list:
                    if not isinstance(tick, dict):
                        continue
                    questions += _posed_questions(tick)
                    for label in tick.get("labels") or ():
                        if not isinstance(label, dict):
                            continue
                        labels += 1
                        label_kinds[label.get("kind")] += 1
                        if not (label.get("source") or label.get("rule")):
                            labels_without_source += 1
            continue

        if schema_version != SCHEMA_SINGLE_REQUEST:
            continue  # 모르는 schema다. 계약 위반으로 이미 적혔으므로 집계에는 넣지 않는다.

        states += 1
        domains[provenance.get("domain") or "unknown"] += 1
        request = record.get("request")
        if not isinstance(request, dict):
            continue
        request_id = request.get("request_id")
        fact_key = _fact_key(request.get("state"))
        if isinstance(request_id, str) and isinstance(group, str):
            group_of_request[request_id] = group
            facts_of_request[request_id] = fact_key
        parent = provenance.get("derived_from")
        if isinstance(parent, str) and isinstance(group, str):
            derived.append(
                {
                    "index": index,
                    "group": group,
                    "parent": parent,
                    "derivation": provenance.get("derivation"),
                    "facts": fact_key,
                }
            )

        if fact_key is not None and isinstance(group, str):
            facts[fact_key].append((index, group))

        state_ids: set[str] = set()
        _collect_ids(request.get("state"), state_ids)
        record_questions = request.get("questions")
        if not isinstance(record_questions, list):
            continue

        by_id: dict[str, dict] = {}
        for position, question in enumerate(record_questions):
            if not isinstance(question, dict):
                continue
            questions += 1
            question_types[question.get("type")] += 1
            if isinstance(question.get("id"), str):
                by_id[question["id"]] = question
            criteria = question.get("criteria")
            if not isinstance(criteria, list):
                continue
            for criterion_index, criterion in enumerate(criteria):
                if not isinstance(criterion, dict):
                    continue
                ref = criterion.get("ref")
                if isinstance(ref, str) and ref not in state_ids:
                    errors.append(
                        _error(
                            index,
                            f"request.questions[{position}].criteria[{criterion_index}].ref",
                            f"상태에 없는 요소를 가리킨다: {ref!r}",
                        )
                    )

        record_labels = record.get("labels") or []
        labelled = set()
        for label_index, label in enumerate(record_labels):
            if not isinstance(label, dict):
                continue
            labels += 1
            label_kinds[label.get("kind")] += 1
            question_id = label.get("question_id")
            if isinstance(question_id, str):
                labelled.add(question_id)
            if not (label.get("source") or label.get("rule")):
                labels_without_source += 1
                errors.append(
                    _error(
                        index,
                        f"labels[{label_index}].source",
                        "라벨에 출처(규칙·전문가 이름)가 없다",
                    )
                )
            if label.get("kind") == "distribution":
                distribution_labels += 1
                probabilities = label.get("probabilities")
                if isinstance(probabilities, dict):
                    total = sum(
                        value for value in probabilities.values() if isinstance(value, (int, float))
                    )
                    if abs(total - 1.0) > PROB_SUM_TOL:
                        errors.append(
                            _error(
                                index,
                                f"labels[{label_index}].probabilities",
                                f"확률 합이 1이 아니다: {total!r}",
                            )
                        )
            question = by_id.get(question_id) if isinstance(question_id, str) else None
            if question is None or question.get("type") != "choice":
                continue
            chosen = label.get("candidate_ids") or (
                [label["answer"]] if isinstance(label.get("answer"), str) else []
            )
            candidates = [
                criterion.get("id")
                for criterion in question.get("criteria") or []
                if isinstance(criterion, dict)
            ]
            if len(chosen) == 1 and len(candidates) >= 3 and chosen[0] in candidates:
                positions[len(candidates)][candidates.index(chosen[0])] += 1
        masked_questions += sum(
            1 for question in by_id if question not in labelled
        )

    # 계보 --------------------------------------------------------------
    for group, splits in sorted(group_splits.items()):
        if len(splits) > 1:
            errors.append(
                _error(
                    -1,
                    "split",
                    f"한 origin group이 여러 split에 걸쳐 있다: {group!r} → {sorted(splits)}",
                )
            )
    for entry in derived:
        index, group, parent = entry["index"], entry["group"], entry["parent"]
        parent_group = group_of_request.get(parent)
        if parent_group is None:
            errors.append(
                _error(index, "provenance.derived_from", f"부모 레코드가 없다: {parent!r}")
            )
            continue
        if parent_group != group:
            errors.append(
                _error(
                    index,
                    "provenance.derived_from",
                    f"파생본의 부모가 다른 origin group에 있다: {parent_group!r} != {group!r}",
                )
            )
        preserves_facts = entry["derivation"] in FACT_PRESERVING_DERIVATIONS
        if preserves_facts and entry["facts"] != facts_of_request.get(parent):
            # 표현·순서만 바꾼 파생본은 사실이 같아야 한다. 사실이 다르면 정답도 달라야
            # 하므로 그 이름으로 부를 수 없다 (docs/04 §3).
            errors.append(
                _error(
                    index,
                    "request.state",
                    f"{entry['derivation']} 파생본인데 부모({parent!r})와 정규화된 사실이 다르다",
                )
            )

    # 중복 계보 ----------------------------------------------------------
    cross_group = 0
    duplicate_records = 0
    for entries in facts.values():
        if len(entries) < 2:
            continue
        duplicate_records += len(entries)
        groups = {group for _, group in entries}
        if len(groups) < 2:
            continue  # 같은 계열 안의 번역·재배열본은 정상이다
        cross_group += 1
        first_index, first_group = entries[0]
        for index, group in entries[1:]:
            if group == first_group:
                continue
            errors.append(
                _error(
                    index,
                    "request.state",
                    f"다른 origin group과 정규화된 사실이 같다: "
                    f"{first_group!r}의 {first_index}번 레코드와 중복이다",
                )
            )

    # 정답 위치 편향 ------------------------------------------------------
    by_candidate_count: dict[str, dict] = {}
    max_excess = 0.0
    for size, counter in sorted(positions.items()):
        total = sum(counter.values())
        shares = [counter[position] / total for position in range(size)]
        excess = max(shares) - 1 / size
        by_candidate_count[str(size)] = {
            "questions": total,
            "counts": [counter[position] for position in range(size)],
            "shares": shares,
            "uniform": 1 / size,
            "max_excess": excess,
        }
        if total >= POSITION_BIAS_MIN_SAMPLES:
            max_excess = max(max_excess, excess)
            if excess > POSITION_BIAS_TOLERANCE:
                errors.append(
                    _error(
                        -1,
                        f"answer_position[{size}]",
                        f"정답 위치가 한쪽으로 쏠렸다: 최대 {max(shares):.3f} "
                        f"(균등 {1 / size:.3f}, 표본 {total})",
                    )
                )

    split_groups: Counter = Counter()
    for group, splits in group_splits.items():
        for split in splits:
            split_groups[split] += 1

    return {
        "version": REPORT_VERSION,
        "invalid_records": invalid,
        "records": len(records),
        "states": states,
        "episodes": episodes,
        "ticks": ticks,
        "questions": questions,
        "labels": labels,
        "masked_questions": masked_questions,
        "distribution_labels": distribution_labels,
        "labels_without_source": labels_without_source,
        "origin_groups": len(group_splits),
        "split_groups": dict(sorted(split_groups.items())),
        "split_records": dict(sorted(split_records.items())),
        "domains": dict(sorted(domains.items())),
        "question_types": dict(sorted(question_types.items())),
        "label_kinds": dict(sorted(label_kinds.items())),
        "variants": dict(sorted(variants.items())),
        "duplicate_content": {
            "cross_group": cross_group,
            "records_with_shared_facts": duplicate_records,
        },
        "answer_position": {
            "questions": sum(sum(counter.values()) for counter in positions.values()),
            "tolerance": POSITION_BIAS_TOLERANCE,
            "max_excess": max_excess,
            "by_candidate_count": by_candidate_count,
        },
        "errors": errors,
    }


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def _read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def load_dataset(dataset: Path) -> tuple[list[dict], list[Path]]:
    """`records.jsonl`·`streams.jsonl`을 아래 단계까지 찾아 모은다."""
    paths = sorted(dataset.rglob("records.jsonl")) + sorted(dataset.rglob("streams.jsonl"))
    records: list[dict] = []
    for path in paths:
        records.extend(_read_jsonl(path))
    return records, paths


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m robo_jev.data.validate",
        description="데이터셋 QA 보고서를 만든다 (docs/04 §6).",
    )
    parser.add_argument("--dataset", type=Path, required=True, help="records.jsonl이 있는 디렉터리")
    parser.add_argument("--report", type=Path, required=True, help="보고서 JSON을 쓸 경로")
    args = parser.parse_args(argv)

    if not args.dataset.is_dir():
        print(f"데이터셋 디렉터리가 없다: {args.dataset}", file=sys.stderr)
        return 2
    records, paths = load_dataset(args.dataset)
    if not paths:
        print(f"{args.dataset} 아래에 records.jsonl·streams.jsonl이 없다", file=sys.stderr)
        return 2

    report = validate_dataset(records)
    report["dataset"] = str(args.dataset)
    report["files"] = [str(path.relative_to(args.dataset)) for path in paths]

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )

    print(
        f"{report['states']}상태 · {report['episodes']}에피소드 · {report['ticks']}틱 · "
        f"{report['questions']}질문 · {report['origin_groups']}계열"
    )
    print(f"  split별 계열 수: {report['split_groups']}")
    print(f"  계약 위반 {report['invalid_records']}건 · QA 위반 {len(report['errors'])}건")
    for error in report["errors"][:10]:
        print(f"  - [{error['index']}] {error['path']}: {error['message']}")
    if len(report["errors"]) > 10:
        print(f"  … 그리고 {len(report['errors']) - 10}건 더 (보고서 참고)")
    print(f"  → {args.report}")
    return 0 if report["invalid_records"] == 0 and not report["errors"] else 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
