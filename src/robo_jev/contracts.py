"""robojev 입력·라벨 계약 v0.

레코드는 두 종류다.

* ``judgment-v0`` — 단일 요청. 공통 상태 + 질문 묶음 + 질문별 라벨 (docs/04 §1).
* ``stream-v0``   — 에피소드 스트림. prefix + 틱 목록 (docs/08 §8).

이 모듈은 두 가지만 한다.

* :func:`validate_record` — 레코드가 계약을 지키는지 본다. 어기면 필드 경로가 앞에
  붙은 ``ValueError``를 낸다 (예: ``ticks[12].labels[3].conditioned_on: …``).
* :func:`model_input`     — 모델이 볼 수 있는 부분만 허용 목록으로 추려 **새 dict**로
  돌려준다. 라벨·근거·미래 결과·채택 결과는 어떤 경로로도 통과하지 못한다.

`model_input`은 검증된 레코드를 받는다고 가정한다. 검증과 직렬화를 분리해야
"입력 경계"를 라벨 유무와 무관하게 확인할 수 있기 때문이다(라벨을 지운 레코드와
원본의 입력이 같아야 한다).
"""

from __future__ import annotations

import copy
from typing import Any, NoReturn

__all__ = [
    "AUX_QUESTIONS",
    "LABEL_CONFIDENCE_LEVELS",
    "LABEL_KINDS",
    "NON_INPUT_FIELDS",
    "PHASES",
    "PROB_SUM_TOL",
    "QUESTION_SET_V0",
    "QUESTION_TYPES",
    "SCHEMA_SINGLE_REQUEST",
    "SCHEMA_STREAM",
    "SPLITS",
    "model_input",
    "validate_record",
]

SCHEMA_SINGLE_REQUEST = "judgment-v0"
SCHEMA_STREAM = "stream-v0"

QUESTION_TYPES = ("choice", "boolean", "ordinal")
LABEL_KINDS = ("valid_set", "single", "distribution", "event")
LABEL_CONFIDENCE_LEVELS = ("high", "medium", "low")
SPLITS = ("train", "dev", "calibration", "test", "ood")

#: 주 결정의 국면 (docs/08 §3.2 `commitment`).
PHASES = ("approach", "grasp", "lift", "transport", "place", "push", "none")

#: 분포 라벨의 합 허용 오차.
PROB_SUM_TOL = 1e-6

#: 모델 입력에 절대 들어가지 않는 필드 이름 (docs/04 §1 표).
NON_INPUT_FIELDS = (
    "labels",
    "provenance",
    "evidence",
    "split",
    "usage",
    "model_output",
    "adopted",
    "ack",
    "origin_group",
    "versions",
)

#: 부가 질문. 라벨은 그 틱의 commitment에 조건화된다 (docs/08 §7).
AUX_QUESTIONS = ("q_gripper", "q_path", "q_speed", "q_force")

# 모델 입력 허용 목록.
_SINGLE_REQUEST_FIELDS = ("request_id", "state", "questions")
_STREAM_REQUEST_FIELDS = ("state", "exec_history", "commitment", "candidates")
_QUESTION_FIELDS = ("id", "type", "instructions", "criteria")
_CRITERION_FIELDS = ("id", "description", "ref", "value")
_TICK_FIELDS = ("t", "sim_ms", "observed_at_ms", "obs_age_ms")
_PREFIX_FIELDS = ("instructions", "question_set")

_BOOLEAN_CRITERIA = [{"id": "true", "description": "예"}, {"id": "false", "description": "아니오"}]


def _question(question_type: str, instructions: str, criteria: list[dict]) -> dict[str, Any]:
    return {"type": question_type, "instructions": instructions, "criteria": criteria}


#: 질문 세트 v0 (docs/08 §4). 후보가 빈 질문은 틱마다 후보가 바뀌는 동적 질문이고,
#: 나머지는 하네스 버전에 고정된 정적 후보를 쓴다.
QUESTION_SET_V0: dict[str, dict[str, Any]] = {
    "q_main": _question("choice", "지금 실행할 행동을 고르라.", []),
    "q_done": _question("boolean", "현재 목표를 이미 만족했는가.", _BOOLEAN_CRITERIA),
    "q_instr": _question("boolean", "실행에 필요한 지시가 충분한가.", _BOOLEAN_CRITERIA),
    "q_observe": _question("boolean", "관측을 더 얻어야 하는가.", _BOOLEAN_CRITERIA),
    "q_retry": _question("boolean", "직전 실패와 같은 방식의 재시도가 적절한가.", _BOOLEAN_CRITERIA),
    "q_stop": _question("boolean", "지금 즉시 멈춰야 하는가.", _BOOLEAN_CRITERIA),
    "q_gripper": _question(
        "choice",
        "commitment의 행동·국면 기준으로 원하는 그리퍼 상태를 고르라.",
        [{"id": "open", "description": "열림"}, {"id": "closed", "description": "닫힘"}],
    ),
    "q_path": _question("choice", "commitment의 행동·국면 기준으로 경로를 고르라.", []),
    "q_speed": _question(
        "ordinal",
        "commitment의 행동·국면 기준으로 속도 수준을 고르라.",
        [
            {"id": "0", "description": "정지 (0 m/s)", "value": 0.0},
            {"id": "1", "description": "저속 (0.1 m/s)", "value": 0.1},
            {"id": "2", "description": "중속 (0.25 m/s)", "value": 0.25},
            {"id": "3", "description": "고속 (0.5 m/s)", "value": 0.5},
        ],
    ),
    "q_force": _question(
        "ordinal",
        "commitment의 행동·국면 기준으로 접촉·힘 수준을 고르라.",
        [
            {"id": "0", "description": "회피", "value": 0.0},
            {"id": "1", "description": "가벼운 접촉", "value": 1.0},
            {"id": "2", "description": "밀기", "value": 2.0},
        ],
    ),
}

#: 후보가 틱마다 바뀌는 질문.
_DYNAMIC_QUESTIONS = tuple(qid for qid, spec in QUESTION_SET_V0.items() if not spec["criteria"])


# --------------------------------------------------------------------------
# 경로가 붙은 오류
# --------------------------------------------------------------------------


def _fail(path: str, message: str) -> NoReturn:
    raise ValueError(f"{path}: {message}")


def _need_dict(node: Any, path: str) -> dict:
    if not isinstance(node, dict):
        _fail(path, f"객체여야 한다 (받은 값: {type(node).__name__})")
    return node


def _need_list(node: Any, path: str, *, allow_empty: bool = True) -> list:
    if not isinstance(node, list):
        _fail(path, f"배열이어야 한다 (받은 값: {type(node).__name__})")
    if not allow_empty and not node:
        _fail(path, "비어 있을 수 없다")
    return node


def _need_str(node: Any, path: str, *, allow_empty: bool = False) -> str:
    if not isinstance(node, str):
        _fail(path, f"문자열이어야 한다 (받은 값: {type(node).__name__})")
    if not allow_empty and not node:
        _fail(path, "비어 있을 수 없다")
    return node


def _need_bool(node: Any, path: str) -> bool:
    if not isinstance(node, bool):
        _fail(path, f"true/false여야 한다 (받은 값: {type(node).__name__})")
    return node


def _need_int(node: Any, path: str, *, minimum: int | None = None) -> int:
    if isinstance(node, bool) or not isinstance(node, int):
        _fail(path, f"정수여야 한다 (받은 값: {type(node).__name__})")
    if minimum is not None and node < minimum:
        _fail(path, f"{minimum} 이상이어야 한다 (받은 값: {node})")
    return node


def _need_number(node: Any, path: str) -> float:
    if isinstance(node, bool) or not isinstance(node, (int, float)):
        _fail(path, f"수치여야 한다 (받은 값: {type(node).__name__})")
    return float(node)


def _need_one_of(node: Any, path: str, allowed: tuple[str, ...]) -> str:
    if node not in allowed:
        _fail(path, f"{list(allowed)} 중 하나여야 한다 (받은 값: {node!r})")
    return node  # type: ignore[return-value]


# --------------------------------------------------------------------------
# 입력 영역 오염 검사
# --------------------------------------------------------------------------


def _is_label_shaped(node: Any) -> bool:
    """라벨 구조인가 — 라벨은 항상 `question_id`와 `kind`를 함께 가진다."""
    return isinstance(node, dict) and "question_id" in node and "kind" in node


def _reject_label_leak(node: Any, path: str) -> None:
    """입력 영역 안에 라벨 구조가 섞여 있으면 거절한다.

    실행 이력·채택 결과 자리에 라벨을 그대로 복사해 넣는 실수("실행 이력과 라벨의
    혼동")를 잡는다.
    """
    if _is_label_shaped(node):
        _fail(path, "라벨 구조(question_id+kind)가 모델 입력 영역에 들어 있다")
    if isinstance(node, dict):
        for key, value in node.items():
            _reject_label_leak(value, f"{path}.{key}")
    elif isinstance(node, list):
        for index, value in enumerate(node):
            _reject_label_leak(value, f"{path}[{index}]")


def _check_request_fields(request: dict, path: str, allowed: tuple[str, ...]) -> None:
    for key in request:
        if key not in allowed:
            _fail(f"{path}.{key}", f"request의 허용 필드가 아니다 (허용: {list(allowed)})")
    for key in allowed:
        if key not in request:
            _fail(f"{path}.{key}", "request에 필요한 필드가 없다")
    _reject_label_leak(request, path)


# --------------------------------------------------------------------------
# 질문과 후보
# --------------------------------------------------------------------------

# 질문 하나를 라벨 검사에 쓸 수 있게 줄인 형태: (타입, 후보 id 목록, 후보 경로)
_ResolvedQuestion = tuple[str, list[str] | None, str]


def _validate_criteria(
    criteria: Any,
    path: str,
    question_type: str,
    *,
    require_description: bool,
) -> list[str]:
    _need_list(criteria, path, allow_empty=False)
    ids: list[str] = []
    previous_value: float | None = None
    for index, criterion in enumerate(criteria):
        item_path = f"{path}[{index}]"
        _need_dict(criterion, item_path)
        criterion_id = _need_str(criterion.get("id"), f"{item_path}.id")
        if criterion_id in ids:
            _fail(f"{item_path}.id", f"후보 id가 중복됐다: {criterion_id!r}")
        ids.append(criterion_id)

        if require_description:
            _need_str(criterion.get("description"), f"{item_path}.description")
        if "ref" in criterion:
            _need_str(criterion["ref"], f"{item_path}.ref")

        if question_type == "ordinal":
            if "value" not in criterion:
                _fail(f"{item_path}.value", "ordinal 질문은 모든 수준에 수치 value가 있어야 한다")
            value = _need_number(criterion["value"], f"{item_path}.value")
            if previous_value is not None and value <= previous_value:
                _fail(
                    f"{item_path}.value",
                    f"ordinal 후보는 value 오름차순이어야 한다: {value} <= {previous_value}",
                )
            previous_value = value
        elif "value" in criterion:
            _need_number(criterion["value"], f"{item_path}.value")
    return ids


def _validate_question(question: Any, path: str) -> tuple[str, _ResolvedQuestion]:
    _need_dict(question, path)
    question_id = _need_str(question.get("id"), f"{path}.id")
    question_type = _need_one_of(question.get("type"), f"{path}.type", QUESTION_TYPES)
    _need_str(question.get("instructions"), f"{path}.instructions")
    criteria_path = f"{path}.criteria"
    ids = _validate_criteria(
        question.get("criteria"), criteria_path, question_type, require_description=True
    )
    return question_id, (question_type, ids, criteria_path)


# --------------------------------------------------------------------------
# 라벨
# --------------------------------------------------------------------------


def _candidate_ids(question: _ResolvedQuestion, label_path: str, question_id: str) -> list[str]:
    _, ids, criteria_path = question
    if ids is None:
        _fail(criteria_path, f"{question_id} 라벨이 있는데 그 틱의 후보 목록이 없다")
    return ids


def _need_candidate(value: Any, path: str, candidate_ids: list[str]) -> str:
    _need_str(value, path)
    if value not in candidate_ids:
        _fail(path, f"존재하지 않는 후보 id: {value!r} (후보: {candidate_ids})")
    return value


def _validate_id_set(
    values: Any, path: str, candidate_ids: list[str], *, allow_empty: bool
) -> list[str]:
    _need_list(values, path, allow_empty=allow_empty)
    seen: list[str] = []
    for index, value in enumerate(values):
        item_path = f"{path}[{index}]"
        _need_candidate(value, item_path, candidate_ids)
        if value in seen:
            _fail(item_path, f"중복된 후보 id: {value!r}")
        seen.append(value)
    return seen


def _validate_event_counts(label: dict, path: str) -> None:
    for field in ("successes", "failures", "censored"):
        if field not in label:
            _fail(f"{path}.{field}", "event 라벨에는 successes·failures·censored가 모두 필요하다")
        _need_int(label[field], f"{path}.{field}", minimum=0)
    if "event_id" not in label and "action_ref" not in label:
        _fail(f"{path}.event_id", "event 라벨은 event_id 또는 action_ref로 사건을 지정해야 한다")
    if "event_id" in label:
        _need_str(label["event_id"], f"{path}.event_id")


def _validate_event_results(results: Any, path: str, candidate_ids: list[str]) -> None:
    _need_dict(results, path)
    for candidate_id, result in results.items():
        item_path = f"{path}.{candidate_id}"
        if candidate_id not in candidate_ids:
            _fail(item_path, f"존재하지 않는 후보 id: {candidate_id!r} (후보: {candidate_ids})")
        _need_dict(result, item_path)
        for field in ("s", "f"):
            if field not in result:
                _fail(f"{item_path}.{field}", "rollout 결과에는 s(성공)·f(실패)가 필요하다")
            _need_int(result[field], f"{item_path}.{field}", minimum=0)
        if "censored" in result:
            _need_int(result["censored"], f"{item_path}.censored", minimum=0)
        if "seeds" in result:
            seeds = _need_list(result["seeds"], f"{item_path}.seeds")
            for seed_index, seed in enumerate(seeds):
                _need_int(seed, f"{item_path}.seeds[{seed_index}]")


def _validate_label(
    label: Any,
    path: str,
    questions: dict[str, _ResolvedQuestion],
    *,
    stream: bool,
    commitment: dict | None = None,
    main_candidate_ids: list[str] | None = None,
) -> None:
    _need_dict(label, path)
    question_id = _need_str(label.get("question_id"), f"{path}.question_id")
    if question_id not in questions:
        _fail(f"{path}.question_id", f"레코드에 없는 질문을 참조한다: {question_id!r}")
    question = questions[question_id]
    question_type = question[0]
    kind = _need_one_of(label.get("kind"), f"{path}.kind", LABEL_KINDS)

    if "mask" in label:
        _need_bool(label["mask"], f"{path}.mask")
    for field in ("source", "rule"):
        if field in label:
            _need_str(label[field], f"{path}.{field}")
    if "label_confidence" in label:
        _need_one_of(label["label_confidence"], f"{path}.label_confidence", LABEL_CONFIDENCE_LEVELS)

    if kind == "valid_set":
        ids = _candidate_ids(question, path, question_id)
        _validate_id_set(label.get("candidate_ids"), f"{path}.candidate_ids", ids, allow_empty=False)
    elif kind == "single":
        if "answer" not in label:
            _fail(f"{path}.answer", "single 라벨에는 answer가 필요하다")
        if question_type == "boolean":
            _need_bool(label["answer"], f"{path}.answer")
        else:
            ids = _candidate_ids(question, path, question_id)
            _need_candidate(label["answer"], f"{path}.answer", ids)
    elif kind == "distribution":
        ids = _candidate_ids(question, path, question_id)
        probabilities = _need_dict(label.get("probabilities"), f"{path}.probabilities")
        if not probabilities:
            _fail(f"{path}.probabilities", "분포가 비어 있다")
        total = 0.0
        for candidate_id, probability in probabilities.items():
            item_path = f"{path}.probabilities.{candidate_id}"
            if candidate_id not in ids:
                _fail(item_path, f"존재하지 않는 후보 id: {candidate_id!r} (후보: {ids})")
            value = _need_number(probability, item_path)
            if not 0.0 <= value <= 1.0:
                _fail(item_path, f"확률은 [0, 1] 안이어야 한다 (받은 값: {value})")
            total += value
        if abs(total - 1.0) > PROB_SUM_TOL:
            _fail(f"{path}.probabilities", f"확률 합이 1이 아니다: {total!r}")
    else:  # event
        _validate_event_counts(label, path)

    if not stream:
        return

    # 스트림 전용 필드 -------------------------------------------------------
    if "action_ref" in label:
        _need_candidate(label["action_ref"], f"{path}.action_ref", main_candidate_ids or [])

    # 허용 집합은 이 지점에서 이미 검사된 리스트다.
    valid_set: set[str] = set(label["candidate_ids"]) if kind == "valid_set" else set()
    for field in ("semantic_admissible", "unknown"):
        if field in label:
            ids = _candidate_ids(question, path, question_id)
            members = _validate_id_set(label[field], f"{path}.{field}", ids, allow_empty=True)
            if field == "unknown":
                overlap = sorted(set(members) & valid_set)
                if overlap:
                    _fail(f"{path}.unknown", f"허용 집합과 겹친다: {overlap}")
    if "event_results" in label:
        ids = _candidate_ids(question, path, question_id)
        _validate_event_results(label["event_results"], f"{path}.event_results", ids)

    if question_id in AUX_QUESTIONS:
        if commitment is None:
            _fail(
                f"{path}.conditioned_on",
                "commitment가 없는 틱에서는 부가 질문 라벨을 둘 수 없다 (마스킹해야 한다)",
            )
        expected = f"{commitment['action_ref']}/{commitment['phase']}"
        if "conditioned_on" not in label:
            _fail(f"{path}.conditioned_on", f"부가 질문 라벨에는 conditioned_on이 필요하다 (기대: {expected!r})")
        given = _need_str(label["conditioned_on"], f"{path}.conditioned_on")
        if given != expected:
            _fail(f"{path}.conditioned_on", f"틱의 commitment와 다르다: {given!r} != {expected!r}")
    elif "conditioned_on" in label:
        _need_str(label["conditioned_on"], f"{path}.conditioned_on")


def _validate_labels(
    labels: Any,
    path: str,
    questions: dict[str, _ResolvedQuestion],
    *,
    stream: bool,
    commitment: dict | None = None,
    main_candidate_ids: list[str] | None = None,
) -> None:
    """라벨 목록을 본다. 라벨이 없는 질문은 loss mask이므로 허용한다."""
    _need_list(labels, path)
    for index, label in enumerate(labels):
        _validate_label(
            label,
            f"{path}[{index}]",
            questions,
            stream=stream,
            commitment=commitment,
            main_candidate_ids=main_candidate_ids,
        )


# --------------------------------------------------------------------------
# 단일 요청 레코드
# --------------------------------------------------------------------------


def _validate_common_envelope(record: dict) -> None:
    if "split" in record:
        _need_one_of(record["split"], "split", SPLITS)
    if "origin_group" in record:
        _need_str(record["origin_group"], "origin_group")
    for field in ("provenance", "evidence", "usage", "versions"):
        if field in record:
            _need_dict(record[field], field)


def _validate_single_request(record: dict) -> None:
    _validate_common_envelope(record)
    request = _need_dict(record.get("request"), "request")
    _check_request_fields(request, "request", _SINGLE_REQUEST_FIELDS)
    _need_str(request["request_id"], "request.request_id")
    _need_dict(request["state"], "request.state")

    questions: dict[str, _ResolvedQuestion] = {}
    raw_questions = _need_list(request["questions"], "request.questions", allow_empty=False)
    for index, question in enumerate(raw_questions):
        question_id, resolved = _validate_question(question, f"request.questions[{index}]")
        if question_id in questions:
            _fail(f"request.questions[{index}].id", f"질문 id가 중복됐다: {question_id!r}")
        questions[question_id] = resolved

    _validate_labels(record.get("labels", []), "labels", questions, stream=False)


# --------------------------------------------------------------------------
# 스트림 레코드
# --------------------------------------------------------------------------


def _validate_prefix(prefix: Any) -> None:
    _need_dict(prefix, "prefix")
    _need_str(prefix.get("question_set"), "prefix.question_set")
    instructions = _need_list(prefix.get("instructions"), "prefix.instructions", allow_empty=False)
    previous_version = None
    previous_t_ms = None
    for index, instruction in enumerate(instructions):
        path = f"prefix.instructions[{index}]"
        _need_dict(instruction, path)
        version = _need_int(instruction.get("version"), f"{path}.version", minimum=1)
        t_ms = _need_int(instruction.get("t_ms"), f"{path}.t_ms", minimum=0)
        _need_str(instruction.get("text"), f"{path}.text")
        if previous_version is not None and version <= previous_version:
            _fail(f"{path}.version", f"지시 버전이 증가하지 않는다: {version} <= {previous_version}")
        if previous_t_ms is not None and t_ms < previous_t_ms:
            _fail(f"{path}.t_ms", f"지시 시각이 뒤로 간다: {t_ms} < {previous_t_ms}")
        previous_version, previous_t_ms = version, t_ms


def _validate_tick_candidates(
    candidates: Any, path: str
) -> tuple[dict[str, _ResolvedQuestion], list[str]]:
    """틱의 후보 목록과 질문 세트 v0를 합쳐 질문별 후보를 정한다."""
    _need_dict(candidates, path)
    for question_id in candidates:
        if question_id not in QUESTION_SET_V0:
            _fail(f"{path}.{question_id}", f"질문 세트 v0에 없는 질문이다: {question_id!r}")

    if "q_main" not in candidates:
        _fail(f"{path}.q_main", "틱마다 q_main의 후보 목록이 필요하다")

    resolved: dict[str, _ResolvedQuestion] = {}
    main_ids = _validate_criteria(
        candidates["q_main"], f"{path}.q_main", "choice", require_description=False
    )
    resolved["q_main"] = ("choice", main_ids, f"{path}.q_main")

    for question_id, spec in QUESTION_SET_V0.items():
        if question_id == "q_main":
            continue
        criteria_path = f"{path}.{question_id}"
        if question_id in candidates:
            ids = _validate_criteria(
                candidates[question_id], criteria_path, spec["type"], require_description=False
            )
        elif question_id in _DYNAMIC_QUESTIONS:
            ids = None  # 라벨이 이 질문을 참조할 때만 오류
        else:
            ids = [criterion["id"] for criterion in spec["criteria"]]
        resolved[question_id] = (spec["type"], ids, criteria_path)

    # 후보가 참조하는 action_ref는 그 틱의 q_main 후보여야 한다.
    for question_id, entries in candidates.items():
        for index, entry in enumerate(entries):
            if "action_ref" in entry:
                _need_candidate(
                    entry["action_ref"], f"{path}.{question_id}[{index}].action_ref", main_ids
                )
    return resolved, main_ids


def _validate_commitment(commitment: Any, path: str, main_ids: list[str]) -> dict | None:
    if commitment is None:
        return None
    _need_dict(commitment, path)
    _need_candidate(commitment.get("action_ref"), f"{path}.action_ref", main_ids)
    _need_one_of(commitment.get("phase"), f"{path}.phase", PHASES)
    if "key" in commitment:
        _need_str(commitment["key"], f"{path}.key")
    if "held_ticks" in commitment:
        _need_int(commitment["held_ticks"], f"{path}.held_ticks", minimum=0)
    return commitment


def _validate_adopted(adopted: Any, path: str, resolved: dict[str, _ResolvedQuestion]) -> None:
    _reject_label_leak(adopted, path)
    _need_dict(adopted, path)
    main_ids = resolved["q_main"][1] or []
    if adopted.get("main") is not None:
        _need_candidate(adopted["main"], f"{path}.main", main_ids)
    if "switch" in adopted:
        _need_bool(adopted["switch"], f"{path}.switch")
    if "stop" in adopted:
        _need_bool(adopted["stop"], f"{path}.stop")
    path_ids = resolved["q_path"][1]
    if adopted.get("path") is not None and path_ids is not None:
        _need_candidate(adopted["path"], f"{path}.path", path_ids)


def _validate_tick(tick: Any, path: str, previous_t: int | None) -> int:
    _need_dict(tick, path)
    tick_number = _need_int(tick.get("t"), f"{path}.t", minimum=0)
    if previous_t is not None and tick_number <= previous_t:
        _fail(f"{path}.t", f"틱 번호가 증가하지 않는다: {tick_number} <= {previous_t}")
    _need_int(tick.get("sim_ms"), f"{path}.sim_ms", minimum=0)
    _need_int(tick.get("observed_at_ms"), f"{path}.observed_at_ms", minimum=0)

    obs_age = tick.get("obs_age_ms")
    if isinstance(obs_age, dict):  # docs/08 §3.2의 소스별 관측 나이
        for source, age in obs_age.items():
            _need_int(age, f"{path}.obs_age_ms.{source}", minimum=0)
    else:
        _need_int(obs_age, f"{path}.obs_age_ms", minimum=0)

    request = _need_dict(tick.get("request"), f"{path}.request")
    _check_request_fields(request, f"{path}.request", _STREAM_REQUEST_FIELDS)
    _need_dict(request["state"], f"{path}.request.state")
    _need_str(request["exec_history"], f"{path}.request.exec_history")

    resolved, main_ids = _validate_tick_candidates(request["candidates"], f"{path}.request.candidates")
    commitment = _validate_commitment(request["commitment"], f"{path}.request.commitment", main_ids)

    if "adopted" in tick and tick["adopted"] is not None:
        _validate_adopted(tick["adopted"], f"{path}.adopted", resolved)
    for field in ("model_output", "ack"):
        if tick.get(field) is not None:
            _need_dict(tick[field], f"{path}.{field}")
            _reject_label_leak(tick[field], f"{path}.{field}")

    _validate_labels(
        tick.get("labels", []),
        f"{path}.labels",
        resolved,
        stream=True,
        commitment=commitment,
        main_candidate_ids=main_ids,
    )
    return tick_number


def _validate_stream(record: dict) -> None:
    _validate_common_envelope(record)
    if "episode_id" in record:
        _need_str(record["episode_id"], "episode_id")
    _validate_prefix(record.get("prefix"))

    ticks = _need_list(record.get("ticks"), "ticks", allow_empty=False)
    previous_t: int | None = None
    for index, tick in enumerate(ticks):
        previous_t = _validate_tick(tick, f"ticks[{index}]", previous_t)


# --------------------------------------------------------------------------
# 공개 API
# --------------------------------------------------------------------------


def validate_record(record: dict) -> None:
    """레코드가 입력·라벨 계약을 지키는지 본다. 어기면 `ValueError`."""
    _need_dict(record, "record")
    schema_version = _need_one_of(
        record.get("schema_version"), "schema_version", (SCHEMA_SINGLE_REQUEST, SCHEMA_STREAM)
    )
    if schema_version == SCHEMA_SINGLE_REQUEST:
        _validate_single_request(record)
    else:
        _validate_stream(record)


def _pick(node: dict, fields: tuple[str, ...]) -> dict:
    """허용 목록에 있는 필드만 깊은 복사로 옮긴다."""
    return {field: copy.deepcopy(node[field]) for field in fields if field in node}


def _single_request_input(record: dict) -> dict:
    request = record.get("request", {})
    questions = []
    for question in request.get("questions", []):
        allowed = _pick(question, _QUESTION_FIELDS)
        allowed["criteria"] = [
            _pick(criterion, _CRITERION_FIELDS) for criterion in question.get("criteria", [])
        ]
        questions.append(allowed)
    allowed_request = _pick(request, ("request_id", "state"))
    allowed_request["questions"] = questions
    return {"schema_version": record["schema_version"], "request": allowed_request}


def _stream_input(record: dict) -> dict:
    ticks = []
    for tick in record.get("ticks", []):
        allowed_tick = _pick(tick, _TICK_FIELDS)
        allowed_tick["request"] = _pick(tick.get("request", {}), _STREAM_REQUEST_FIELDS)
        ticks.append(allowed_tick)
    return {
        "schema_version": record["schema_version"],
        "prefix": _pick(record.get("prefix", {}), _PREFIX_FIELDS),
        "ticks": ticks,
    }


def model_input(record: dict) -> dict:
    """모델이 볼 수 있는 부분만 허용 목록으로 추려 새 dict로 돌려준다.

    원본은 건드리지 않는다. 라벨·근거·분할·모델 출력·채택 결과·ACK는 어떤 경로로도
    통과하지 못한다. 검증된 레코드를 받는다고 가정한다.
    """
    _need_dict(record, "record")
    schema_version = _need_one_of(
        record.get("schema_version"), "schema_version", (SCHEMA_SINGLE_REQUEST, SCHEMA_STREAM)
    )
    if schema_version == SCHEMA_SINGLE_REQUEST:
        return _single_request_input(record)
    return _stream_input(record)
