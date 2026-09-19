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
* holdout 봉인 (docs/04 §5, 계약 v0.3) — `holdouts`(설정의 `split` 절; CLI는 데이터셋의 `manifest.json`에서 읽는다)를
  주면 계열마다 provenance의 문구 템플릿 변형(`phrasing`)·개념(`concepts`)·prefix·group·분야를 holdout 목록과 맞대,
  holdout 계열이 train/dev/calibration/test에 있으면(누출) 위반이고, ood_dev/ood_test인데 holdout 이유가 없어도 위반이다.
  종류별(템플릿·개념·prefix·group·분야) 계열·레코드 수와 split별 수를 `holdouts`에 적는다.
* 대조 쌍 (docs/04 §3·§6) — `provenance.derivation == "contrast"`인 sibling마다 부모가 같은 계열·split에 있고, 표현을 걷어낸
  상태가 부모와 **정확히 한 자리**만 다르며(초점 사실), 뒤집혔다는 질문의 라벨이 실제로 부모와 다르고, 초점 사실을 지운
  장면(분야의 `forget`)에서 그 질문을 다시 그리면 라벨이 마스크·"해당 없음"이 되는지(삭제 검사)를 다시 돌린다. 어느 하나라도
  어기면 위반이다 — 생성기는 삭제 검사에 실패한 쌍을 만들지 않으므로(빠진 이유를 기본 레코드에 적는다) 여기 위반은 생성기와
  규칙 코드가 어긋났다는 뜻이다. 쌍 수·split·분야별 수·삭제 결과·빠진 이유를 `contrast`에 적는다.

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
from robo_jev.data.split import CONCEPT_TAG, OOD_SPLITS, TEMPLATE_TAG, SplitPolicy

__all__ = ["REPORT_VERSION", "leaf_diff", "main", "validate_dataset"]

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


def leaf_diff(left: Any, right: Any, path: str = "state") -> list[str]:
    """두 (표현을 걷어낸) 상태가 다른 잎의 경로. 대조 sibling은 부모와 정확히 한 자리만 달라야 한다 (docs/04 §3)."""
    if isinstance(left, dict) and isinstance(right, dict):
        found: list[str] = []
        for key in sorted(set(left) | set(right)):
            if key not in left or key not in right:
                found.append(f"{path}.{key}")
            else:
                found.extend(leaf_diff(left[key], right[key], f"{path}.{key}"))
        return found
    if isinstance(left, list) and isinstance(right, list):
        if len(left) != len(right):
            return [path]
        found = []
        for index, (a, b) in enumerate(zip(left, right)):
            found.extend(leaf_diff(a, b, f"{path}[{index}]"))
        return found
    return [] if left == right else [path]


def _contrast_report(records: Sequence[dict], errors: list[dict]) -> dict:
    """대조 쌍의 검사와 집계 (모듈 설명의 "대조 쌍")."""
    from robo_jev.data.domains import DOMAINS, QuestionSpec
    from robo_jev.data.generate import contrast_counts, deletion_outcome, semantic_answers

    by_request: dict[str, tuple[int, dict]] = {}
    for index, record in enumerate(records):
        request = record.get("request") if isinstance(record, dict) else None
        if isinstance(request, dict) and isinstance(request.get("request_id"), str):
            by_request[request["request_id"]] = (index, record)

    checked = flip_failures = field_failures = deletion_failures = unpaired = 0
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            continue
        provenance = record.get("provenance") if isinstance(record.get("provenance"), dict) else {}
        contrast = provenance.get("contrast") if isinstance(provenance.get("contrast"), dict) else None
        if contrast is None:
            continue
        if contrast.get("role") == "base":
            sibling_id = contrast.get("sibling_id")
            if sibling_id is not None and sibling_id not in by_request:
                unpaired += 1  # count 절단 등으로 sibling이 데이터셋에 없다 — 위반은 아니다
            continue
        if provenance.get("derivation") != "contrast":
            continue
        checked += 1
        parent_id = provenance.get("derived_from")
        if contrast.get("sibling_id") != parent_id:
            errors.append(_error(index, "provenance.contrast.sibling_id", f"부모와 다르다: {contrast.get('sibling_id')!r} != {parent_id!r}"))
        parent_entry = by_request.get(parent_id) if isinstance(parent_id, str) else None
        if parent_entry is None:
            errors.append(_error(index, "provenance.contrast", f"대조 쌍의 부모가 없다: {parent_id!r}"))
            continue
        _, parent = parent_entry
        if parent.get("split") != record.get("split") or parent.get("origin_group") != record.get("origin_group"):
            errors.append(_error(index, "split", "대조 sibling이 부모와 다른 계열·split에 있다"))

        # 1. 정확히 한 자리만 다른가 (로봇 틱은 초점 사실과 그 파생값 — 대상의 relative_mm·같은 사실의 두 표현 — 만).
        diff = leaf_diff(_strip_wording(parent["request"]["state"]), _strip_wording(record["request"]["state"]))
        robot = provenance.get("domain") == "robot"
        if robot:
            from robo_jev.data.robot_contrast import allowed_diff_paths

            allowed = allowed_diff_paths(str(provenance.get("kind")), str(contrast.get("focus_field")))
            bad = [path for path in diff if not path.startswith(allowed)]
            if not diff or bad:
                field_failures += 1
                errors.append(_error(index, "request.state", f"로봇 대조 sibling은 초점 사실({contrast.get('focus_field')})만 달라야 한다 (다른 곳: {bad[:4] or diff[:4]})"))
        elif len(diff) != 1:
            field_failures += 1
            errors.append(_error(index, "request.state", f"대조 sibling은 부모와 한 자리만 달라야 한다 (다른 곳 {len(diff)}: {diff[:4]})"))

        # 2. 뒤집혔다는 질문의 라벨이 실제로 다른가.
        question_id = contrast.get("flipped_question")
        if robot:
            from robo_jev.data.robot_contrast import flipped_answer

            before, after = flipped_answer(parent, str(question_id)), flipped_answer(record, str(question_id))
            same = before is None or after is None or before == after
        else:
            base_answers = semantic_answers(parent)
            answers = semantic_answers(record)
            same = question_id not in base_answers or question_id not in answers or base_answers[question_id] == answers[question_id]
        if same:
            flip_failures += 1
            errors.append(_error(index, "provenance.contrast.flipped_question", f"{question_id!r}의 라벨이 부모와 같거나 마스크다"))

        # 3. 삭제 검사: 초점 사실을 지우면 라벨이 마스크·"해당 없음"인가 (분야 규칙 코드로 다시 돌린다).
        evidence = record.get("evidence") if isinstance(record.get("evidence"), dict) else {}
        recorded = (contrast.get("deletion") or {}).get("outcome")
        if robot:
            from robo_jev.data.robot_contrast import deletion_outcome as robot_deletion
            from robo_jev.sim.expert import Expert

            tick_request = (evidence.get("contrast") or {}).get("tick_request") if isinstance(evidence.get("contrast"), dict) else None
            if not isinstance(tick_request, dict):
                deletion_failures += 1
                errors.append(_error(index, "evidence.contrast", "삭제 analogue를 다시 돌릴 근거(틱의 후보·이력·commitment)가 없다"))
                continue
            outcome = robot_deletion(record["request"]["state"], tick_request, str(provenance.get("kind")), Expert())
            if outcome is None:
                deletion_failures += 1
                errors.append(_error(index, "provenance.contrast.deletion", f"초점 사실 {contrast.get('focus_field')!r}을 지워도 전문가가 게이트로 가지 않는다"))
            elif outcome != recorded:
                errors.append(_error(index, "provenance.contrast.deletion", f"기록된 삭제 결과와 다르다: {recorded!r} != {outcome!r}"))
            continue
        domain = DOMAINS.get(str(provenance.get("domain")))
        spec_data = (evidence.get("contrast") or {}).get("spec") if isinstance(evidence.get("contrast"), dict) else None
        scene = evidence.get("scene")
        if domain is None or not isinstance(spec_data, dict) or not isinstance(scene, dict):
            deletion_failures += 1
            errors.append(_error(index, "evidence.contrast", "삭제 검사를 다시 돌릴 근거(분야·질문 명세·장면)가 없다"))
            continue
        try:
            deleted = domain.forget(scene, str(contrast.get("focus_field")))
            spec = QuestionSpec(str(spec_data["id"]), str(spec_data["type"]), str(spec_data["kind"]), dict(spec_data.get("params") or {}))
            outcome = deletion_outcome(domain, deleted, spec, str(provenance.get("language", "ko")))
        except (KeyError, ValueError, StopIteration) as error:
            outcome = None
            errors.append(_error(index, "evidence.contrast", f"삭제 검사를 돌릴 수 없다: {error}"))
        if outcome is None:
            deletion_failures += 1
            errors.append(_error(index, "provenance.contrast.deletion", f"초점 사실 {contrast.get('focus_field')!r}을 지워도 {question_id!r}의 라벨이 남는다"))
        elif outcome != recorded:
            errors.append(_error(index, "provenance.contrast.deletion", f"기록된 삭제 결과와 다르다: {recorded!r} != {outcome!r}"))

    return {
        **contrast_counts([record for record in records if isinstance(record, dict) and isinstance(record.get("provenance"), dict)]),
        "checked": checked,
        "unpaired_bases": unpaired,
        "one_field_failures": field_failures,
        "flip_failures": flip_failures,
        "deletion_failures": deletion_failures,
    }


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


def _record_tags(provenance: dict) -> set[str]:
    """레코드의 holdout 태그: 문구 템플릿 변형(`phrasing`)과 개념(`concepts`) — 생성기가 provenance에 적은 것."""
    tags: set[str] = set()
    for item in provenance.get("phrasing") or ():
        if isinstance(item, str):
            tags.add(TEMPLATE_TAG + item)
    for item in provenance.get("concepts") or ():
        if isinstance(item, str):
            tags.add(CONCEPT_TAG + item)
    return tags


def _holdout_report(
    policy: SplitPolicy | None,
    group_splits: dict[str, set[str]],
    group_tags: dict[str, set[str]],
    group_records: Counter,
    errors: list[dict],
) -> dict:
    """계열마다 holdout 이유를 계산해 누출(train/dev/calibration/test에 있는 holdout 계열)과 이유 없는 OOD를 위반으로 적고,
    종류별 계열·레코드 수를 센다."""
    by_reason: dict[str, dict] = {}
    groups: Counter = Counter()
    records: Counter = Counter()
    leaked = 0
    for group in sorted(group_splits):
        splits = group_splits[group]
        reasons = policy.holdout_reasons(group, sorted(group_tags.get(group, ()))) if policy is not None else []
        in_ood = {split for split in splits if split in OOD_SPLITS}
        outside = {split for split in splits if split not in OOD_SPLITS}
        if reasons:
            for split in splits:
                groups[split] += 1
                records[split] += group_records[(group, split)]
            for reason in reasons:
                entry = by_reason.setdefault(reason, {"groups": 0, "records": 0, "splits": Counter()})
                entry["groups"] += 1
                for split in splits:
                    entry["records"] += group_records[(group, split)]
                    entry["splits"][split] += 1
            if outside:
                leaked += 1
                errors.append(
                    _error(-1, "split", f"holdout 계열이 학습 쪽 split에 있다(누출): {group!r} → {sorted(outside)} (이유: {reasons})")
                )
        elif policy is not None and in_ood:
            errors.append(_error(-1, "split", f"ood인데 holdout 이유가 없다: {group!r} → {sorted(in_ood)}"))
    return {
        "policy": None if policy is None else {
            "holdout_groups": sorted(policy.holdout_groups),
            "holdout_prefixes": list(policy.holdout_prefixes),
            "holdout_domains": sorted(policy.holdout_domains),
            "holdout_templates": sorted(policy.holdout_templates),
            "holdout_concepts": sorted(policy.holdout_concepts),
        },
        "groups": dict(sorted(groups.items())),
        "records": dict(sorted(records.items())),
        "leaked_groups": leaked,
        "by_reason": {
            reason: {"groups": entry["groups"], "records": entry["records"], "splits": dict(sorted(entry["splits"].items()))}
            for reason, entry in sorted(by_reason.items())
        },
    }


def validate_dataset(records: Sequence[dict], *, holdouts: dict | None = None) -> dict:
    """데이터셋 QA 보고서. 위반은 모두 모으고 집계를 함께 낸다. `holdouts`는 설정의 `split` 절(봉인 목록)이다."""
    errors: list[dict] = []
    policy = SplitPolicy.from_config(holdouts) if holdouts is not None else None
    group_tags: dict[str, set[str]] = defaultdict(set)
    group_records: Counter = Counter()
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
            group_records[(group, split)] += 1
        if isinstance(split, str):
            split_records[split] += 1

        provenance = record.get("provenance") if isinstance(record.get("provenance"), dict) else {}
        for tag in provenance.get("variants") or ():
            variants[tag] += 1
        if isinstance(group, str):
            group_tags[group] |= _record_tags(provenance)

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

    holdout_report = _holdout_report(policy, group_splits, group_tags, group_records, errors)
    contrast_report = _contrast_report(records, errors)

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
        "holdouts": holdout_report,
        "contrast": contrast_report,
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


def load_holdouts(path: Path | None) -> dict | None:
    """봉인 목록: manifest.json(`config.split`)이나 설정 YAML/JSON(`split` 절 또는 절 자체). 파일이 없으면 None."""
    if path is None or not path.is_file():
        return None
    if path.suffix in (".yaml", ".yml"):
        import yaml

        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    else:
        data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        return None
    if isinstance(data.get("config"), dict) and isinstance(data["config"].get("split"), dict):
        return dict(data["config"]["split"])
    if isinstance(data.get("split"), dict):
        return dict(data["split"])
    if any(key.startswith("holdout_") for key in data):
        return dict(data)
    return None


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
    parser.add_argument(
        "--holdouts", type=Path, default=None,
        help="봉인 목록(설정 YAML/JSON의 split 절 또는 manifest.json). 없으면 데이터셋의 manifest.json에서 읽는다",
    )
    args = parser.parse_args(argv)

    if not args.dataset.is_dir():
        print(f"데이터셋 디렉터리가 없다: {args.dataset}", file=sys.stderr)
        return 2
    records, paths = load_dataset(args.dataset)
    if not paths:
        print(f"{args.dataset} 아래에 records.jsonl·streams.jsonl이 없다", file=sys.stderr)
        return 2

    holdouts = load_holdouts(args.holdouts or args.dataset / "manifest.json")
    report = validate_dataset(records, holdouts=holdouts)
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
    holdouts_report = report["holdouts"]
    if holdouts_report["policy"] is None:
        print("  holdout: 봉인 목록 없음 (manifest.json이 없거나 split 절이 없다) — 누출 검사를 하지 않았다")
    else:
        print(f"  holdout 계열: {holdouts_report['groups']} · 레코드 {holdouts_report['records']} · 누출 {holdouts_report['leaked_groups']}건")
        for reason, entry in holdouts_report["by_reason"].items():
            print(f"    {reason}: 계열 {entry['groups']} · 레코드 {entry['records']} · {entry['splits']}")
    contrast = report["contrast"]
    print(
        f"  대조 쌍 {contrast['pairs']} (기본 레코드 {contrast['base_records']}의 {contrast['pair_share_of_bases']:.0%}) · split {contrast['by_split']} · "
        f"삭제 검사 실패 {contrast['deletion_failures']} · 한 자리 위반 {contrast['one_field_failures']} · 뒤집힘 위반 {contrast['flip_failures']} · 빠짐 {contrast['missing']}"
    )
    print(f"  계약 위반 {report['invalid_records']}건 · QA 위반 {len(report['errors'])}건")
    for error in report["errors"][:10]:
        print(f"  - [{error['index']}] {error['path']}: {error['message']}")
    if len(report["errors"]) > 10:
        print(f"  … 그리고 {len(report['errors']) - 10}건 더 (보고서 참고)")
    print(f"  → {args.report}")
    return 0 if report["invalid_records"] == 0 and not report["errors"] else 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
