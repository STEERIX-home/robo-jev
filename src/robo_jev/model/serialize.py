"""직렬화 v0 — 알려진 입력을 토큰·구간·분기·position으로 (docs/08 §3.1~3.2, docs/03 §3).

입력은 레코드(`judgment-v0` 또는 `stream-v0`)다. 먼저 :func:`robo_jev.contracts.validate_record`로
입력 영역의 비입력 키·라벨 구조를 거절하고, :func:`robo_jev.contracts.model_input`이 추린 투영만
직렬화한다. 라벨·근거·모델 출력·채택 결과는 어떤 경로로도 토큰이 되지 않는다.

두 배치를 만든다.

* ``state_first`` (L0, 단일 요청) — 공통 상태 S 뒤에 질문 T_i가 병렬 분기로 놓인다.
  ``T_i = [질문 머리][후보 1 … c_i1][후보 2 … c_i2] … [결정 위치 d_i]``. 각 질문의 position은
  ``len(S)``에서 다시 센다 (docs/03 §3 "분기별 정보 접근과 위치 규칙").
* ``stream_l1a`` (L1-a, 에피소드 스트림) — 정적 prefix(시작 지시, 질문 세트 v0 텍스트와 정적
  후보) 뒤에 틱이 이어진다. 틱 = ``[틱 머리·상태][commitment][실행 이력][동적 후보]`` 뒤 결정
  위치들. 결정 위치는 틱 끝 공통 상태에서 갈라지는 **1토큰 분기**라 모두 같은 position을 갖고,
  다음 틱은 분기 이전 position에서 이어진다 (docs/08 §3.1). prefix는 에피소드 시작 시의 것으로
  고정이며, 지시가 바뀌면 그 버전을 처음 실은 틱의 **첫 토큰들**로 덧붙인다(리셋 없음) — 그
  틱의 토큰이라 다른 틱 토큰과 같이 윈도우 밖으로 나가고, 현재 지시는 매 틱 `goal`이 다시
  싣는다. 틱 머리는 ``[tick t]``뿐이며, 상태에 `t` 구간이 없는 레코드에서만 틱 겉봉투의 시각
  값을 머리에 싣는다(하네스 상태는 `t`에 같은 값을 이미 실으므로 중복을 만들지 않는다).

서식 규칙 (docs/08 §3.2): 한 줄에 한 항목, 고정 필드 순서, 위치는 mm 정수, 자세는 quaternion
소수 2자리, 시간은 ms 정수. 스칼라 묶음(`t`·`goal`·`scene`·`robot`·`exec`)은 ``이름 k=v k=v`` 한
줄, 항목 목록(`objects`·`zones`·`events`·`derived`)은 항목마다 한 줄, 빈 목록은 ``-``, 없음은
``none``이다. 필드 순서는 :data:`FIELD_ORDER`(스키마 순서)이고 거기 없는 키는 그 뒤에 이름순이다.
줄은 조각(chunk) 단위로 따로 토큰화해 이어 붙인다 — 스트림에서 틱마다 새 토큰만 붙이는 것과
같은 계산이며, 줄 경계에서는 통째 토큰화와 같다(검사가 실제 tokenizer로 확인).

**결정 위치의 예약 토큰.** 결정 위치는 대문자 한 글자 **한 토큰**이다(:data:`DECISION_MARKERS`).
backbone이 모르는 특수 토큰을 만들지 않는다 — byte-level BPE 어휘에서 ASCII 한 글자는 언제나 한
토큰이고, 검사가 실제 tokenizer로 확인한다. 어느 글자인지는 **요청 안의 질문 순서에 의존하지 않는다**
(docs/03 §3, docs/08 §3.1):

* ``state_first``: 모든 질문이 같은 고정 표지 :data:`STATE_FIRST_MARKER`를 쓴다. 질문의 정체는
  ``T_i`` 문맥이 주므로 질문 추가·삭제·재배열이 다른 질문의 토큰을 바꾸지 않는다(구조로 성립).
* ``stream_l1a``: 분기가 1토큰이라 표지가 질문의 정체를 져야 한다. 질문 세트 버전이 질문 id마다
  표지를 고정하고(``QUESTION_SET_V0[qid]["marker"]``), 직렬화는 그 map을 읽어(dict 순서가 아니라)
  정적 prefix에 ``markers q_main=A q_done=B …`` 한 줄로 선언한다. 틱마다 묻는 부분집합·순서, 세트에
  끼워 넣은 새 질문과 무관하게 id의 표지가 유지된다. map은 id마다 다른 예약 토큰이어야 한다(검사).

질문 머리에도 같은 글자가 붙어(``A q_main choice: …``) 결정 토큰이 어느 질문의 것인지 문맥에서
읽힌다. 후보 경계는 후보 줄의 **마지막 토큰**(줄바꿈)이다.

돌려주는 dict의 토큰별 필드(``kind``·``state``·``question``·``candidate``·``position``, 스트림은
``tick``)는 :func:`robo_jev.model.attention.build_reference_mask`가 그대로 받는다.

* ``kind`` — ``state``·``question``·``candidate``·``decision``, 스트림은 ``prefix``·``exec``가 더 있다.
  스트림의 처음 prefix는 정적 후보 줄까지 전부 ``prefix``다(mask에서 윈도우 밖으로 나가지 않는
  유일한 구간). 그 후보의 경계는 ``static_candidate_boundaries``와 토큰별 ``candidate``가
  가리킨다. 도중 지시 조각은 ``state``다.
* ``question`` — 논리적 분기 id(0부터, ``question_ids``의 색인). 공유 토큰은 -1. ``state_first``는
  T_i 전체가, ``stream_l1a``는 결정 토큰만 분기다.
* ``candidate`` — 그 질문의 후보 목록(``candidate_mapping``) 안 색인(0부터). 후보 줄 밖은 -1.
* ``position`` — 논리적 position. ``state`` — 상태(레코드) id, 한 레코드면 전부 0.
"""

from __future__ import annotations

import string
from typing import Any

from robo_jev.contracts import (
    QUESTION_SET_V0,
    QUESTION_SET_V0_EN,
    SCHEMA_SINGLE_REQUEST,
    SCHEMA_STREAM,
    model_input,
    validate_record,
)

__all__ = [
    "DECISION_MARKERS",
    "STATE_FIRST_MARKER",
    "FIELD_ORDER",
    "LAYOUTS",
    "POSITION_UNIT",
    "QUATERNION_DECIMALS",
    "QUESTION_SETS",
    "SECTION_ORDER",
    "TIME_UNIT",
    "TOKEN_SERIALIZER_VERSION",
    "WINDOW_TICKS",
    "candidate_line",
    "serialize_request",
    "state_lines",
]

LAYOUTS = ("state_first", "stream_l1a")

#: **토큰 직렬화**의 버전 — 이 모듈이 정하는 텍스트 서식·구간·표지·position 규칙의 버전이다. 결과 layout의
#: ``serializer``와 학습 manifest의 ``serializer_version``에 적힌다. 토큰 형식이 바뀌면(표지 선언 줄, L0 표지,
#: 필드 순서 …) 여기서 올린다.
#:
#: 레코드의 ``versions.serializer``(`s0.2`, configs/sim/tidy_clutter.yaml의 `version`)와는 **다른 것**이다 — 그것은
#: 환경·하네스가 상태를 레코드로 적는 **레코드 직렬화**(상태 스키마·mm/ms 정수·quaternion 자릿수)의 버전이고
#: :mod:`robo_jev.data.episode` 가 적는다. 4b가 토큰 형식을 바꿨을 때 두 형식이 `s0.2` 한 문자열을 나눠 가졌던
#: 일을 되풀이하지 않도록 이름부터 가른다(`ts…` 대 `s…`).
TOKEN_SERIALIZER_VERSION = "ts0.3"
POSITION_UNIT = "mm"
QUATERNION_DECIMALS = 2
TIME_UNIT = "ms"

#: Full-attention 윈도우: 정적 prefix + 최근 30틱 (docs/08 §3.1).
WINDOW_TICKS = 30

#: 결정 위치에 쓸 수 있는 예약 토큰(대문자 한 글자 = 한 토큰). 스트림의 질문 세트 map은 이 안에서 고른다.
DECISION_MARKERS = tuple(string.ascii_uppercase)

#: 상태 선행(L0)의 고정 표지 — 모든 질문이 같은 토큰을 쓴다 (docs/03 §3).
STATE_FIRST_MARKER = "A"

#: 스트림 prefix에 펼치는 질문 세트. id → 질문 정의 (docs/08 §4). 하네스 설정 `question_set_id`가 언어마다 내는 id가
#: 전부 여기 있어야 한다(검사가 대조한다). 영어판은 문구만 다르고 id·타입·후보·표지는 같다.
QUESTION_SETS: dict[str, dict[str, dict[str, Any]]] = {"qs-v0": QUESTION_SET_V0, "qs-v0-en": QUESTION_SET_V0_EN}

#: 상태 스키마 v0의 구간 순서 (docs/08 §3.2 표). 없는 키는 뒤에 이름순.
SECTION_ORDER = (
    "t", "goal", "objects", "scene", "zones", "robot", "exec", "events", "derived",
    "commitment", "image", "geom",
)  # fmt: skip

#: 항목 안 필드의 고정 순서. 없는 키는 뒤에 이름순.
FIELD_ORDER = (
    # 식별·설명
    "id", "desc", "description", "class", "kind", "key", "ref", "action_ref", "object", "waypoint",
    # 지시·시각
    "text", "version", "t_ms", "tick", "sim_ms", "observed_at_ms", "age_ms", "obs_age_ms", "seq",
    "candidate_set_version", "target_ref", "target_zone", "priority", "forbidden_contact", "fragile",
    # 자세·기하
    "pose_mm", "pos_mm", "quat", "ee_pose_mm", "ee_quat", "precision_mm", "pose_sigma_mm", "obb_mm",
    "top_mm", "graspable_faces", "surface_conf", "visible", "visible_ratio", "last_seen_ms", "reid",
    "attributes",
    # 장면·영역·로봇
    "work_surface_mm", "free_width_mm", "corridor_mm", "clearance_mm", "bounds_mm", "gripper_mm",
    "holding", "contact_n", "speed_mm_s",
    # 실행·commitment
    "phase", "path", "speed_level", "force_level", "gripper", "stop", "progress", "applied",
    "reject", "gripper_wait", "events", "executor", "held_ticks", "last_switch_tick",
    # 파생·후보
    "relative_mm", "nearest_clearance_mm", "around", "extra_mm", "derived", "value",
)  # fmt: skip

_SINGULAR = {"objects": "object", "zones": "zone", "events": "event"}
_FIELD_RANK = {name: index for index, name in enumerate(FIELD_ORDER)}
_SECTION_RANK = {name: index for index, name in enumerate(SECTION_ORDER)}


# --------------------------------------------------------------------------
# 값 서식 (docs/08 §3.2)
# --------------------------------------------------------------------------


def _integer_unit(key: str) -> bool:
    """위치(mm)·시간(ms) 필드 — 정수로 적는다."""
    return key.endswith(f"_{POSITION_UNIT}") or key.endswith(f"_{TIME_UNIT}") or key in (
        POSITION_UNIT,
        TIME_UNIT,
    )


def _scalar(key: str, value: Any) -> str:
    if value is None:
        return "none"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if _integer_unit(key):
            return str(int(round(value)))
        if "quat" in key:
            return f"{value:.{QUATERNION_DECIMALS}f}"
        text = f"{value:.2f}".rstrip("0").rstrip(".")
        return text if text not in ("", "-0") else "0"
    text = str(value).replace("\n", " ").strip()
    return text if text else "-"


def _ordered(node: dict, rank: dict[str, int]) -> list[tuple[str, Any]]:
    known = sorted((key for key in node if key in rank), key=rank.__getitem__)
    unknown = sorted(key for key in node if key not in rank)
    return [(key, node[key]) for key in known + unknown]


def _value(key: str, value: Any) -> str:
    """한 줄 안에 들어가는 값. 목록은 쉼표, 안의 항목은 세미콜론, 중첩 dict는 ``k:v``."""
    if isinstance(value, dict):
        return ",".join(f"{k}:{_value(k, v)}" for k, v in _ordered(value, _FIELD_RANK)) or "-"
    if isinstance(value, (list, tuple)):
        if not value:
            return "-"
        if all(isinstance(item, dict) for item in value):
            return ";".join(_value(key, item) for item in value)
        return ",".join(_value(key, item) for item in value)
    return _scalar(key, value)


def _pairs(item: dict) -> str:
    return " ".join(f"{key}={_value(key, value)}" for key, value in _ordered(item, _FIELD_RANK)) or "-"


def state_lines(state: dict) -> list[str]:
    """상태 dict → 줄 목록. 한 줄에 한 항목, 구간·필드 순서 고정."""
    lines: list[str] = []
    for key, value in _ordered(state, _SECTION_RANK):
        if isinstance(value, list) and value and all(isinstance(item, dict) for item in value):
            name = _SINGULAR.get(key, key)
            lines.extend(f"{name} {_pairs(item)}" for item in value)
        elif isinstance(value, dict):
            lines.append(f"{key} {_pairs(value)}" if value else f"{key}=-")
        else:
            lines.append(f"{key}={_value(key, value)}")
    return lines


def candidate_line(entry: dict) -> str:
    """후보 한 줄: ``<id>: <설명> k=v …``. 줄바꿈으로 끝나며 그 마지막 토큰이 후보 경계다.

    `action_ref`가 id와 같으면(결합 후보) 적지 않는다 — 같은 값을 두 번 읽을 이유가 없다.
    """
    rest = dict(entry)
    identifier = _scalar("id", rest.pop("id"))
    desc = rest.pop("desc", None)
    if desc is None:
        desc = rest.pop("description", None)
    else:
        rest.pop("description", None)
    if rest.get("action_ref") == entry.get("id"):
        del rest["action_ref"]
    parts = [f"{identifier}:"]
    if desc not in (None, ""):
        parts.append(_scalar("desc", desc))
    parts.extend(f"{key}={_value(key, value)}" for key, value in _ordered(rest, _FIELD_RANK))
    return " ".join(parts) + "\n"


def _question_header(marker: str, question_id: str, spec: dict) -> str:
    return f"{marker} {question_id} {spec['type']}: {_scalar('instructions', spec['instructions'])}\n"


# --------------------------------------------------------------------------
# 조각 → 토큰·layout
# --------------------------------------------------------------------------


#: `_Chunk.restart`의 값: 이 조각부터 position을 첫 구간(S)의 끝에서 다시 센다.
_RESTART_AT_STATE_END = "state_end"


class _Chunk:
    __slots__ = ("text", "kind", "name", "question", "candidate", "owner", "tick", "restart")

    def __init__(
        self,
        text: str,
        kind: str,
        name: str,
        *,
        question: int = -1,
        candidate: int = -1,
        owner: int = -1,
        tick: int = -1,
        restart: str | None = None,
    ) -> None:
        self.text = text
        self.kind = kind
        self.name = name
        self.question = question
        self.candidate = candidate
        self.owner = owner
        self.tick = tick
        self.restart = restart


def _encode_all(tokenizer: Any, texts: list[str]) -> list[list[int]]:
    if hasattr(tokenizer, "encode_batch"):
        return [list(item.ids) for item in tokenizer.encode_batch(texts, add_special_tokens=False)]
    return [list(tokenizer.encode(text, add_special_tokens=False).ids) for text in texts]


def _assemble(chunks: list[_Chunk], tokenizer: Any, *, stream: bool) -> dict[str, Any]:
    """조각을 토큰화해 이어 붙이고 토큰별 필드·구간 표·position을 만든다.

    position 규칙: 조각은 이어서 센다. ``restart``가 있는 조각은 첫 구간(S)의 끝에서 다시 센다
    (state_first의 질문 시작 = len(S)). 스트림의 결정 조각은 틱 몸통 끝 position에서 갈라지는
    분기라 서로 같은 position을 갖고, 다음 조각은 분기 이전 position에서 이어간다.
    """
    encoded = _encode_all(tokenizer, [chunk.text for chunk in chunks])
    tokens: list[int] = []
    kind: list[str] = []
    question: list[int] = []
    candidate: list[int] = []
    position: list[int] = []
    tick: list[int] = []
    segment: list[int] = []
    segments: list[dict[str, Any]] = []

    cursor = 0  # 다음 조각의 position
    for index, (chunk, ids) in enumerate(zip(chunks, encoded)):
        if chunk.restart == _RESTART_AT_STATE_END:
            cursor = len(encoded[0])
        start = len(tokens)
        tokens.extend(ids)
        kind.extend([chunk.kind] * len(ids))
        question.extend([chunk.question] * len(ids))
        candidate.extend([chunk.candidate] * len(ids))
        tick.extend([chunk.tick] * len(ids))
        segment.extend([index] * len(ids))
        position.extend(range(cursor, cursor + len(ids)))
        segments.append(
            {
                "id": index,
                "kind": chunk.kind,
                "name": chunk.name,
                "owner": chunk.owner,
                "tick": chunk.tick,
                "start": start,
                "end": start + len(ids),
            }
        )
        branch = stream and chunk.kind == "decision"
        if not branch:
            cursor += len(ids)

    out: dict[str, Any] = {
        "tokens": tokens,
        "text": "".join(chunk.text for chunk in chunks),
        "kind": kind,
        "state": [0] * len(tokens),
        "question": question,
        "candidate": candidate,
        "position": position,
        "segment": segment,
        "segments": segments,
    }
    if stream:
        out["tick"] = tick
    return out


def _last_index(segment: dict[str, Any]) -> int:
    return segment["end"] - 1


# --------------------------------------------------------------------------
# state_first (L0)
# --------------------------------------------------------------------------


def _serialize_state_first(
    projected: dict, tokenizer: Any, *, max_state_tokens: int | None, max_total_tokens: int | None
) -> dict[str, Any]:
    request = projected["request"]
    questions = request["questions"]
    marker = STATE_FIRST_MARKER  # 모든 질문이 같은 고정 표지 — 질문 수 상한은 프로파일(계약)이 정한다

    chunks = [_Chunk("[state]\n" + "\n".join(state_lines(request["state"])) + "\n", "state", "state")]
    for branch, spec in enumerate(questions):
        question_id = spec["id"]
        chunks.append(
            _Chunk(
                _question_header(marker, question_id, spec),
                "question",
                f"question:{question_id}",
                question=branch,
                owner=branch,
                restart=_RESTART_AT_STATE_END,  # 질문의 position은 len(S)부터
            )
        )
        for index, criterion in enumerate(spec["criteria"]):
            chunks.append(
                _Chunk(
                    candidate_line(criterion),
                    "candidate",
                    f"candidate:{question_id}:{index}",
                    question=branch,
                    candidate=index,
                    owner=branch,
                )
            )
        chunks.append(
            _Chunk(marker, "decision", f"decision:{question_id}", question=branch, owner=branch)
        )

    out = _assemble(chunks, tokenizer, stream=False)
    state_end = out["segments"][0]["end"]

    if max_state_tokens is not None and state_end > max_state_tokens:
        raise ValueError(f"request.state: 공통 상태가 {max_state_tokens} 토큰을 넘는다 ({state_end})")
    if max_total_tokens is not None and len(out["tokens"]) > max_total_tokens:
        raise ValueError(
            f"request: 요청 전체가 {max_total_tokens} 토큰을 넘는다 ({len(out['tokens'])})"
        )

    boundaries: dict[str, list[int]] = {spec["id"]: [] for spec in questions}
    decisions: dict[str, int] = {}
    for segment in out["segments"]:
        if segment["kind"] == "candidate":
            boundaries[questions[segment["owner"]]["id"]].append(_last_index(segment))
        elif segment["kind"] == "decision":
            decisions[questions[segment["owner"]]["id"]] = _last_index(segment)

    out.update(
        {
            "layout": "state_first",
            "serializer": TOKEN_SERIALIZER_VERSION,
            "request_id": request.get("request_id"),
            "question_ids": [spec["id"] for spec in questions],
            "state_end": state_end,
            "candidate_mapping": {
                spec["id"]: [criterion["id"] for criterion in spec["criteria"]] for spec in questions
            },
            "candidate_boundaries": boundaries,
            "decision_positions": decisions,
            "decision_markers": {spec["id"]: marker for spec in questions},
        }
    )
    return out


# --------------------------------------------------------------------------
# stream_l1a (L1-a)
# --------------------------------------------------------------------------


def _instruction_line(instruction: dict) -> str:
    return f"instruction {_pairs(instruction)}\n"


def _declared_markers(question_set_id: str, question_set: dict[str, dict[str, Any]]) -> dict[str, str]:
    """질문 세트가 id마다 선언한 표지 map. 빠짐·중복·예약 밖은 오류다 (dict 순서는 쓰지 않는다)."""
    markers: dict[str, str] = {}
    for question_id, spec in question_set.items():
        marker = spec.get("marker")
        path = f"prefix.question_set[{question_set_id}].{question_id}.marker"
        if not isinstance(marker, str) or marker not in DECISION_MARKERS:
            raise ValueError(
                f"{path}: 예약 토큰 {DECISION_MARKERS[0]}~{DECISION_MARKERS[-1]} 중 하나여야 한다 (받은 값: {marker!r})"
            )
        if marker in markers.values():
            taken = next(q for q, m in markers.items() if m == marker)
            raise ValueError(f"{path}: 표지 {marker!r}는 {taken!r}가 이미 쓴다 — id마다 달라야 한다")
        markers[question_id] = marker
    return markers


def _markers_line(markers: dict[str, str]) -> str:
    """정적 prefix의 표지 선언 한 줄: ``markers q_main=A q_done=B …``."""
    return "markers " + " ".join(f"{question_id}={marker}" for question_id, marker in markers.items()) + "\n"


_ENVELOPE_FIELDS = ("sim_ms", "observed_at_ms", "obs_age_ms")


def _tick_header(tick: dict) -> str:
    """틱 머리 ``[tick t]``.

    하네스가 만든 상태는 같은 시각 값을 `t` 구간에 싣는다 — 머리에 또 적으면 직렬화가 스스로
    중복을 만드는 것이다. `t` 구간이 없는 레코드(D0 fixture처럼 손으로 만든 것)에서만 틱
    겉봉투의 값을 머리에 실어 정보를 잃지 않는다.
    """
    header = f"[tick {_scalar('t', tick['t'])}]"
    if not isinstance(tick["request"].get("state", {}).get("t"), dict):
        envelope = {key: tick[key] for key in _ENVELOPE_FIELDS if key in tick}
        if envelope:
            header += " " + _pairs(envelope)
    return header + "\n"


def _instruction_slots(
    instructions: list[dict], ticks: list[dict]
) -> tuple[dict[int, list[dict]], list[int]]:
    """지시 버전 2 이상이 들어갈 틱 색인. 그 버전을 처음 실은 틱 앞이다.

    `state.goal.version`이 있으면 그것으로, 없으면 `t_ms <= sim_ms`로 정한다. 어느 틱도 싣지
    않는 지시(잘린 레코드의 꼬리)는 뒤에 오는 틱이 없어 어떤 결정에도 닿지 않으므로 넣지 않고
    버전만 따로 돌려준다.
    """
    slots: dict[int, list[dict]] = {}
    unplaced: list[int] = []
    for instruction in instructions[1:]:
        version = int(instruction["version"])
        target: int | None = None
        for index, tick in enumerate(ticks):
            goal = (tick["request"].get("state") or {}).get("goal")
            if isinstance(goal, dict) and goal.get("version") is not None:
                hit = int(goal["version"]) >= version
            else:
                hit = int(tick.get("sim_ms", 0)) >= int(instruction.get("t_ms", 0))
            if hit:
                target = index
                break
        if target is None:
            unplaced.append(version)
        else:
            slots.setdefault(target, []).append(instruction)
    return slots, unplaced


def _serialize_stream(projected: dict, tokenizer: Any, *, window_ticks: int) -> dict[str, Any]:
    prefix = projected["prefix"]
    ticks = projected["ticks"]
    question_set_id = str(prefix.get("question_set"))
    if question_set_id not in QUESTION_SETS:
        raise ValueError(
            f"prefix.question_set: 직렬화할 수 없는 질문 세트다: {question_set_id!r} "
            f"(아는 것: {list(QUESTION_SETS)})"
        )
    question_set = QUESTION_SETS[question_set_id]
    question_ids = list(question_set)
    markers = _declared_markers(question_set_id, question_set)
    branch_of = {question_id: branch for branch, question_id in enumerate(question_ids)}

    instructions = prefix["instructions"]
    chunks: list[_Chunk] = [
        _Chunk(_instruction_line(instructions[0]), "prefix", f"instruction:{instructions[0]['version']}")
    ]
    chunks.append(_Chunk(f"[questions {question_set_id}]\n", "prefix", "question_set"))
    chunks.append(_Chunk(_markers_line(markers), "prefix", "markers"))
    for question_id, spec in question_set.items():
        branch = branch_of[question_id]
        chunks.append(
            _Chunk(
                _question_header(markers[question_id], question_id, spec),
                "prefix",
                f"question:{question_id}",
                owner=branch,
            )
        )
        for index, criterion in enumerate(spec["criteria"]):
            chunks.append(
                _Chunk(
                    candidate_line(criterion),
                    "prefix",
                    f"candidate:{question_id}:{index}",
                    candidate=index,
                    owner=branch,
                )
            )
    prefix_chunks = len(chunks)

    slots, unplaced = _instruction_slots(instructions, ticks)
    for index, tick in enumerate(ticks):
        # 도중 추가되는 지시는 **그 틱의 토큰**이다 — prefix가 아니라 다른 틱 토큰과 같이 윈도우
        # 밖으로 나간다. 현재 지시는 매 틱 `goal`이 다시 실으므로 잊히지 않는다 (docs/08 §3.1).
        for instruction in slots.get(index, ()):
            chunks.append(
                _Chunk(
                    _instruction_line(instruction),
                    "state",
                    f"instruction:{instruction['version']}",
                    tick=index,
                )
            )
        request = tick["request"]
        chunks.append(
            _Chunk(
                _tick_header(tick)
                + "\n".join(state_lines(request["state"]))
                + "\n",
                "state",
                "state",
                tick=index,
            )
        )
        commitment = request.get("commitment")
        chunks.append(
            _Chunk(
                f"commitment {_pairs(commitment)}\n" if commitment else "commitment=none\n",
                "state",
                "commitment",
                tick=index,
            )
        )
        chunks.append(
            _Chunk(
                f"exec_history {_scalar('exec_history', request['exec_history'])}\n",
                "exec",
                "exec_history",
                tick=index,
            )
        )
        candidates = request["candidates"]
        for question_id in question_ids:
            if question_id not in candidates:
                continue
            branch = branch_of[question_id]
            chunks.append(
                _Chunk(
                    f"[candidates {question_id}]\n",
                    "question",
                    f"candidates:{question_id}",
                    owner=branch,
                    tick=index,
                )
            )
            for position, entry in enumerate(candidates[question_id]):
                chunks.append(
                    _Chunk(
                        candidate_line(entry),
                        "candidate",
                        f"candidate:{question_id}:{position}",
                        candidate=position,
                        owner=branch,
                        tick=index,
                    )
                )
        for question_id, spec in question_set.items():
            if not spec["criteria"] and question_id not in candidates:
                continue  # 후보 없는 동적 질문은 이 틱에 묻지 않는다
            branch = branch_of[question_id]
            chunks.append(
                _Chunk(
                    markers[question_id],
                    "decision",
                    f"decision:{question_id}",
                    question=branch,
                    owner=branch,
                    tick=index,
                )
            )
    out = _assemble(chunks, tokenizer, stream=True)
    segments = out["segments"]
    prefix_end = segments[prefix_chunks]["start"] if prefix_chunks < len(segments) else len(out["tokens"])

    static_boundaries: dict[str, list[int]] = {}
    static_mapping: dict[str, list[str]] = {}
    for segment in segments[:prefix_chunks]:
        if segment["name"].startswith("candidate:"):
            question_id = question_ids[segment["owner"]]
            static_boundaries.setdefault(question_id, []).append(_last_index(segment))
    for question_id, spec in question_set.items():
        if spec["criteria"]:
            static_mapping[question_id] = [criterion["id"] for criterion in spec["criteria"]]

    tick_entries: list[dict[str, Any]] = []
    instruction_positions: list[int] = [
        segment["start"] for segment in segments if segment["name"].startswith("instruction:")
    ]
    for index, tick in enumerate(ticks):
        own = [segment for segment in segments if segment["tick"] == index]
        body = [segment for segment in own if segment["kind"] != "decision"]
        decisions = [segment for segment in own if segment["kind"] == "decision"]
        candidates = tick["request"]["candidates"]
        posed = [qid for qid, spec in question_set.items() if spec["criteria"] or qid in candidates]
        boundaries: dict[str, list[int]] = {}
        mapping: dict[str, list[str]] = {}
        for question_id in posed:
            if question_id in candidates:
                boundaries[question_id] = [
                    _last_index(segment)
                    for segment in body
                    if segment["name"].startswith(f"candidate:{question_id}:")
                ]
                mapping[question_id] = [entry["id"] for entry in candidates[question_id]]
            else:
                boundaries[question_id] = list(static_boundaries[question_id])
                mapping[question_id] = list(static_mapping[question_id])
        tick_entries.append(
            {
                "index": index,
                "t": tick["t"],
                "start": body[0]["start"],
                "body_end": body[-1]["end"],
                "end": decisions[-1]["end"] if decisions else body[-1]["end"],
                "posed": posed,
                "decision_positions": {
                    question_ids[segment["owner"]]: _last_index(segment) for segment in decisions
                },
                "candidate_boundaries": boundaries,
                "candidate_mapping": mapping,
            }
        )

    out.update(
        {
            "layout": "stream_l1a",
            "serializer": TOKEN_SERIALIZER_VERSION,
            "question_set": question_set_id,
            "question_ids": question_ids,
            "window_ticks": window_ticks,
            "prefix_end": prefix_end,
            "instruction_positions": instruction_positions,
            "unplaced_instructions": unplaced,
            "static_candidate_boundaries": static_boundaries,
            "static_candidate_mapping": static_mapping,
            "decision_markers": dict(markers),
            "ticks": tick_entries,
            "tick_boundaries": [[entry["start"], entry["end"]] for entry in tick_entries],
        }
    )
    return out


# --------------------------------------------------------------------------
# 공개 API
# --------------------------------------------------------------------------


def serialize_request(
    request: dict,
    tokenizer: Any,
    layout: str = "state_first",
    *,
    max_state_tokens: int | None = 2048,
    max_total_tokens: int | None = 8192,
    window_ticks: int = WINDOW_TICKS,
) -> dict[str, Any]:
    """레코드 → 토큰·구간·분기·position (모듈 설명 참조).

    `request`는 `judgment-v0`(``state_first``) 또는 `stream-v0`(``stream_l1a``) 레코드다. 계약
    검사를 먼저 돌리므로 입력 영역의 비입력 키·라벨 구조는 경로가 붙은 `ValueError`로 거절되고,
    입력 영역 밖의 라벨·근거는 투영에서 빠져 결과에 영향을 주지 않는다. `max_*`는 L0 프로파일의
    상한(docs/06 Global Constraints)이며 넘으면 자르지 않고 오류다.
    """
    if layout not in LAYOUTS:
        raise ValueError(f"layout: {list(LAYOUTS)} 중 하나여야 한다 (받은 값: {layout!r})")
    validate_record(request)
    projected = model_input(request)  # 허용 필드만 깊은 복사한 새 dict — 원본은 그대로다
    schema_version = projected["schema_version"]
    expected = SCHEMA_SINGLE_REQUEST if layout == "state_first" else SCHEMA_STREAM
    if schema_version != expected:
        raise ValueError(
            f"layout: {layout!r}는 {expected!r} 레코드용이다 (받은 레코드: {schema_version!r})"
        )
    if layout == "state_first":
        return _serialize_state_first(
            projected, tokenizer, max_state_tokens=max_state_tokens, max_total_tokens=max_total_tokens
        )
    if window_ticks < 1:
        raise ValueError(f"window_ticks: 1 이상이어야 한다 (받은 값: {window_ticks})")
    return _serialize_stream(projected, tokenizer, window_ticks=window_ticks)
