"""직렬화 v0 — 알려진 입력을 토큰·구간·분기·position으로 (docs/08 §3.1~3.2, docs/03 §3).

입력은 레코드(`judgment-v0` 또는 `stream-v0`)다. 먼저 :func:`robo_jev.contracts.validate_record`로
입력 영역의 비입력 키·라벨 구조를 거절하고, :func:`robo_jev.contracts.model_input`이 추린 투영만
직렬화한다. 라벨·근거·모델 출력·채택 결과는 어떤 경로로도 토큰이 되지 않는다.

두 배치를 만든다.

* ``state_first`` (L0, 단일 요청) — 공통 상태 S 뒤에 질문 T_i가 병렬 분기로 놓인다.
  ``T_i = [질문 머리][후보 1 … c_i1][후보 2 … c_i2] … [결정 위치 d_i]``. 각 질문의 position은
  ``len(S)``에서 다시 센다 (docs/03 §3 "분기별 정보 접근과 위치 규칙").
* ``stream_l1a`` (L1-a, 에피소드 스트림, **서식 v0.3** — docs/08 §3.1·§3.2) — 정적 prefix(시작 지시,
  질문 세트 v0 텍스트와 정적 후보, 첫 틱의 영역·장면 요약) 뒤에 틱이 이어진다. 틱 =
  ``[t 머리][goal][물체 소개·동적 줄(변화분)][영역·장면(바뀐 틱만)][robot][exec][사건][경유점]
  [commitment][실행 이력][동적 후보]`` 뒤 결정 위치들. 결정 위치는 틱 끝 공통 상태에서 갈라지는
  **1토큰 분기**라 모두 같은 position을 갖고, 다음 틱은 분기 이전 position에서 이어진다. prefix는
  에피소드 시작 시의 것으로 고정이며, 지시가 바뀌면 그 버전을 처음 실은 틱의 **첫 토큰들**로
  덧붙인다(리셋 없음) — 그 틱의 토큰이라 다른 틱 토큰과 같이 윈도우 밖으로 나가고, 현재 지시는
  매 틱 `goal` 줄이 압축 참조(`goal v2 target=o7 zone=zoneL forbid=o3`)로 다시 싣고 텍스트는
  `goal_text_period_ticks`마다 다시 싣는다.

**변화분 틱 (docs/08 §3.1, 계약 v0.3).** 물체는 두 줄로 나뉜다. **소개 줄** ``obj <id> <desc> obb= top= faces=
attr=``(정적 필드)은 에피소드 시작·처음 관측·정적 필드 변경 시·`object_intro_period_ticks`(30, 윈도우 길이)마다,
**동적 줄** ``<id> p= q= s= conf= vis= seen= rel= clr= corr=``은 변화(자세 변화 > max(정밀도 `s`, `object_pose_delta_mm`),
자세 회전, 가시 비율 변화 ≥ `object_visibility_delta`, 관측 정지의 시작·끝, 재식별 사건, 들고 있는 물체) 또는
`object_dynamic_period_ticks`(10)마다 싣는다. 바뀌지 않은 물체는 틱에서 빠진다. `seen`(마지막 관측 시각)은 관측이
**끊긴** 물체에만 있다. 영역·장면 요약은 prefix에 있고 바뀐 틱에만 다시 싣는다. 나머지 구간(t·goal·robot·exec·
commitment·실행 이력·후보·결정)은 매 틱이다 — 후보 블록의 변화분은 계약 밖이다(memo 1b). 변화분은 **입력의
정의**이지 계산 근사가 아니다: 틱 k의 텍스트는 틱 0..k의 레코드만으로 정해지므로 앞 n틱만 직렬화한 결과는 전체
직렬화의 접두이고(검사), 증분 계산과 처음부터 계산이 같다. 문턱·주기는 :data:`DELTA_RULES`이며
`configs/harness/robot.yaml`의 `serialization:` 블록이 같은 값을 적는다(검사가 대조; 레코드의 `config_digest`에 든다).

서식 규칙 (docs/08 §3.2): 한 줄에 한 항목, 고정 필드 순서, 위치는 mm 정수, 자세는 quaternion 소수 2자리, 시간은
ms 정수. 스트림은 짧은 필드 이름과 기본값 생략(:data:`STREAM_FIELDS`)을 쓰고, 상태 선행(L0) 배치는 스키마 이름
그대로다: 스칼라 묶음은 ``이름 k=v k=v`` 한 줄, 항목 목록은 항목마다 한 줄, 빈 목록은 ``-``, 없음은 ``none``,
필드 순서는 :data:`FIELD_ORDER`(스키마 순서)이고 거기 없는 키는 그 뒤에 이름순이다. 줄은 조각(chunk) 단위로 따로
토큰화해 이어 붙인다 — 스트림에서 틱마다 새 토큰만 붙이는 것과 같은 계산이며, 줄 경계에서는 통째 토큰화와
같다(검사가 실제 tokenizer로 확인).

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

import math
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
    "DELTA_RULES",
    "STATE_FIRST_MARKER",
    "FIELD_ORDER",
    "LAYOUTS",
    "POSITION_UNIT",
    "QUATERNION_DECIMALS",
    "QUESTION_SETS",
    "SECTION_ORDER",
    "STREAM_FIELDS",
    "STREAM_FORMAT",
    "TIME_UNIT",
    "TOKEN_SERIALIZER_VERSION",
    "WINDOW_TICKS",
    "candidate_line",
    "serialize_request",
    "state_lines",
    "stream_candidate_line",
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
TOKEN_SERIALIZER_VERSION = "ts0.4"
#: 스트림 서식의 계약 버전 (docs/08 §3.2 표). ts0.4 = 서식 v0.3 (짧은 이름, 물체 소개/동적 분리, 변화분 틱).
STREAM_FORMAT = "v0.3"
POSITION_UNIT = "mm"
QUATERNION_DECIMALS = 2
TIME_UNIT = "ms"

#: Full-attention 윈도우: 정적 prefix + 최근 30틱 (docs/08 §3.1).
WINDOW_TICKS = 30

#: 변화분 틱의 문턱·주기 (docs/08 §3.1). `configs/harness/robot.yaml`의 `serialization:` 블록과 같아야 한다(검사).
#: `object_pose_delta_mm`은 물체의 정밀도 추정(`s`)에 더해지는 바닥값이다 — 문턱 = max(s, 이 값).
DELTA_RULES: dict[str, Any] = {
    "object_intro_period_ticks": 30,
    "object_dynamic_period_ticks": 10,
    "goal_text_period_ticks": 10,
    "object_pose_delta_mm": 0,
    "object_visibility_delta": 0.25,
}

#: 스트림 서식 v0.3의 짧은 필드 이름 (docs/08 §3.2 표 — 문서의 표와 같아야 한다).
STREAM_FIELDS: dict[str, dict[str, str]] = {
    "t": {"tick": "t", "age_ms.geom": "g", "age_ms.proprio": "p", "seq": "seq"},
    "goal": {"version": "v", "target_ref": "target", "target_desc": "desc", "target_zone": "zone", "forbidden_contact": "forbid", "fragile": "fragile", "priority": "prio", "text": "text"},
    "obj": {"desc": "", "obb_mm": "obb", "top_mm": "top", "graspable_faces": "faces", "attributes": "attr"},
    "object": {"pose_mm": "p", "yaw_deg": "yaw", "quat": "q", "precision_mm": "s", "pose_sigma_mm": "s", "surface_conf": "conf", "visible_ratio": "vis", "last_seen_ms": "seen", "clearance_mm": "clr", "nearest_clearance_mm": "clr", "corridor_mm": "corr", "reid": "reid"},
    "zone": {"desc": "", "bounds_mm": "b"},
    "scene": {"work_surface_mm": "surf", "free_width_mm": "free", "corridor_mm": "corr", "clearance_mm": "clr"},
    "robot": {"ee_pose_mm": "ee", "ee_quat": "eq", "gripper_mm": "grip", "holding": "hold", "contact_n": "f", "speed_mm_s": "v"},
    "exec": {"seq": "seq", "executor": "ex", "action_ref": "a", "phase": "ph", "path": "path", "speed_mm_s": "v", "speed_level": "lvl", "force_level": "f", "gripper": "g", "stop": "stop", "stale": "stale", "hold_after_stale": "hold_stale", "progress": "prog", "applied": "applied", "reject": "rej", "gripper_wait": "wait", "events": "ev"},
    "ev": {"object": "o", "displacement_mm": "d"},
    "wp": {"pos_mm": "p", "around": "around", "extra_mm": "extra"},
    "commitment": {"action_ref": "a", "phase": "ph", "held_ticks": "held", "last_switch_tick": "sw"},
    "hist": {"main": "main", "phase": "ph", "path": "path", "speed": "v", "force": "f", "gripper": "g", "stop": "stop", "gate": "gate", "ack": "ack", "fails": "fails"},
}

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
# stream_l1a (L1-a) — 서식 v0.3
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


# -- 짧은 줄 서식 (docs/08 §3.2 표) ----------------------------------------------

#: 값이 이것이면 줄에서 뺀다 (기본값 생략).
_OMIT = (None, False, 0, 0.0, "", "none", [], ())


def _short(prefix: str, item: dict, table: dict[str, str], *, keep_zero: tuple[str, ...] = ()) -> str:
    """짧은 이름의 ``k=v`` 묶음. 표에 없는 키는 스키마 이름 그대로 뒤에 붙고, 기본값(None·false·0·빈 목록)은 뺀다.

    참 boolean은 이름만 적는다(``stop``). 표의 순서가 곧 줄의 순서다.
    """
    parts: list[str] = [prefix] if prefix else []
    ordered = [key for key in table if key in item] + sorted(key for key in item if key not in table)
    for key in ordered:
        value = item[key]
        if key not in keep_zero and (value in _OMIT or (isinstance(value, (list, tuple)) and not value)):
            continue
        name = table.get(key, key)
        if isinstance(value, bool):
            parts.append(name)
        elif name == "":
            parts.append(_scalar(key, value))
        else:
            parts.append(f"{name}={_value(key, value)}")
    return " ".join(parts)


def _tick_header_line(tick: dict) -> str:
    """틱 머리 ``t <tick> age g<ms> p<ms> seq <n>`` (docs/08 §3.2). 모의 시각·관측 시각·후보 집합 해시는 겉봉투에만 있다.

    상태에 `t` 구간이 없는 레코드(D0 fixture)는 틱 겉봉투의 나이로 채운다.
    """
    state = tick["request"].get("state") or {}
    section = state.get("t") if isinstance(state.get("t"), dict) else {}
    number = section.get("tick", tick.get("t"))
    ages = section.get("age_ms", tick.get("obs_age_ms"))
    parts = [f"t {_scalar('tick', number)}"]
    if isinstance(ages, dict):
        parts.append("age " + " ".join(f"{STREAM_FIELDS['t'].get('age_ms.' + key, key)}{_scalar(key, value)}" for key, value in ages.items()))
    elif ages is not None:
        parts.append(f"age {_scalar('age_ms', ages)}")
    if section.get("seq") is not None:
        parts.append(f"seq {_scalar('seq', section['seq'])}")
    return " ".join(parts) + "\n"


def _goal_line(goal: Any, *, with_text: bool) -> str:
    """``goal v<n> target=<id> [desc=<설명>] zone=<id> forbid=<ids> fragile=<ids> [text=…]`` — 압축 참조. 문자열 goal(D0)은 그대로.

    `desc`(지시가 부르는 대상의 설명)는 대상이 아직 추적되지 않아 `target=-`일 때만 싣는다 — 그때 모델이 대상을 알 유일한 단서다.
    """
    if not isinstance(goal, dict):
        return f"goal {_scalar('goal', goal)}\n" if goal not in (None, "") else "goal -\n"
    item = {key: value for key, value in goal.items() if key not in ("t_ms",)}
    if not with_text:
        item.pop("text", None)
    version = item.pop("version", None)
    target = item.pop("target_ref", None)
    if target is not None:
        item.pop("target_desc", None)
    body = _short("", item, STREAM_FIELDS["goal"])
    head = "goal" + (f" v{_scalar('version', version)}" if version is not None else "") + f" target={_value('target_ref', target)}"
    return head + (f" {body}" if body else "") + "\n"


_INTRO_KEYS = ("desc", "obb_mm", "top_mm", "graspable_faces", "attributes")
#: 동적 줄의 항상 싣는 필드(자세·정밀도·여유·통로)와 바뀔 때만 싣는 필드(방향·표면 신뢰도·가시 비율·재식별).
_DYNAMIC_ALWAYS = ("pose_mm", "precision_mm", "pose_sigma_mm")
_DYNAMIC_OPTIONAL = ("quat", "surface_conf", "visible_ratio", "reid")


def _intro_line(entry: dict) -> str:
    """물체 소개 줄 ``obj <id> <desc> obb=<x,y,z> top=<mm> faces=<a,b> attr=<…>`` (정적 필드)."""
    item = {key: entry[key] for key in _INTRO_KEYS if key in entry}
    return _short(f"obj {_scalar('id', entry['id'])}", item, STREAM_FIELDS["obj"]) + "\n"


def _orientation(quat: Any) -> tuple[str, Any] | None:
    """자세 회전의 짧은 표기: 작업면 위 물체는 z축 회전뿐이므로 ``yaw=<도>``(정수, 회전 없음은 0), 기울어진 자세만
    ``q=<quaternion>``. 돌려주는 것은 (키, 값) — 키는 `yaw_deg` 또는 `quat`; quaternion이 아니면 None."""
    if not isinstance(quat, (list, tuple)) or len(quat) != 4:
        return None
    x, y, z, w = (float(value) for value in quat)
    if abs(x) < 0.005 and abs(y) < 0.005:
        yaw = int(round(math.degrees(2.0 * math.atan2(z, w))))
        return ("yaw_deg", (yaw + 180) % 360 - 180)
    return ("quat", list(quat))


def _dynamic_line(entry: dict, derived: dict, *, stale: bool, optional: tuple[str, ...], full: bool) -> str:
    """물체 동적 줄 ``<id> p= [yaw=|q=] s= [conf=] [vis=] [seen=] clr= corr= [reid=]``.

    자세·정밀도·여유·통로는 줄마다 싣고, `optional`에 든 방향·표면 신뢰도·가시 비율·재식별은 바뀐 것(또는 소개 틱의
    전체 줄)만 싣는다. 기본값 `yaw=0`(회전 없음)·`vis=1`(다 보임)은 **전체 줄(`full`)에서만** 뺀다 — 변화분 줄은 바뀐
    필드를 값이 기본값이라도 적어 "기본값으로 돌아왔다"(가시 비율 1→0.5→1, 요 0→90→0)가 "그대로다"와 같은 줄이 되지
    않게 한다. `seen`(마지막 관측 시각)은 관측이 끊긴 물체에만. 말단 기준 상대 벡터(`relative_mm`)는 싣지 않는다 —
    `robot ee`와 `p`로 정해진다.
    """
    item: dict[str, Any] = {key: entry[key] for key in _DYNAMIC_ALWAYS if key in entry}
    if "quat" in optional:
        orientation = _orientation(entry.get("quat"))
        if orientation is not None and not (full and orientation == ("yaw_deg", 0)):
            item[orientation[0]] = orientation[1]
    if "surface_conf" in optional and "surface_conf" in entry:
        item["surface_conf"] = entry["surface_conf"]
    if "visible_ratio" in optional and "visible_ratio" in entry and not (full and float(entry["visible_ratio"]) >= 1.0):
        item["visible_ratio"] = entry["visible_ratio"]
    if stale and "last_seen_ms" in entry:
        item["last_seen_ms"] = entry["last_seen_ms"]
    for key in ("clearance_mm", "nearest_clearance_mm", "corridor_mm"):
        if key in derived:
            item[key] = derived[key]
    if "reid" in optional and entry.get("reid"):
        item["reid"] = entry["reid"]
    return _short(_scalar("id", entry["id"]), item, STREAM_FIELDS["object"], keep_zero=("yaw_deg", "visible_ratio", "clearance_mm", "nearest_clearance_mm", "corridor_mm", "precision_mm", "pose_sigma_mm")) + "\n"


def _zone_line(zone: dict) -> str:
    item = {key: value for key, value in zone.items() if key != "id"}
    return _short(f"zone {_scalar('id', zone.get('id'))}", item, STREAM_FIELDS["zone"], keep_zero=("bounds_mm",)) + "\n"


def _scene_line(scene: dict) -> str:
    return _short("scene", scene, STREAM_FIELDS["scene"], keep_zero=tuple(scene)) + "\n"


def _robot_line(robot: dict, *, with_quat: bool) -> str:
    """``robot ee=<x,y,z> [eq=<quat>] grip=<mm> [hold=<id>] [f=<N>] [v=<mm/s>]`` — 말단 자세는 바뀔 때와 소개 틱에만."""
    item = dict(robot)
    if not with_quat:
        item.pop("ee_quat", None)
    return _short("robot", item, STREAM_FIELDS["robot"], keep_zero=("ee_pose_mm", "gripper_mm")) + "\n"


def _exec_line(execution: dict) -> str:
    return _short("exec", execution, STREAM_FIELDS["exec"]) + "\n"


def _event_line(event: dict) -> str:
    item = {key: value for key, value in event.items() if key not in ("kind", "sim_ms")}
    return _short(f"ev {_scalar('kind', event.get('kind'))}", item, STREAM_FIELDS["ev"], keep_zero=tuple(item)) + "\n"


def _waypoint_line(waypoint: dict) -> str:
    item = {key: value for key, value in waypoint.items() if key not in ("waypoint", "kind")}
    head = f"wp {_scalar('waypoint', waypoint.get('waypoint'))}"
    if waypoint.get("kind") is not None:
        head += f" {_scalar('kind', waypoint['kind'])}"
    return _short(head, item, STREAM_FIELDS["wp"], keep_zero=tuple(item)) + "\n"


def _commitment_line(commitment: Any) -> str:
    """``commitment a=<ref> ph=<phase> held=<n> sw=<tick>`` — 키는 후보 줄이 나르므로 적지 않는다."""
    if not isinstance(commitment, dict) or not commitment:
        return "commitment none\n"
    item = {key: value for key, value in commitment.items() if key != "key"}
    return _short("commitment", item, STREAM_FIELDS["commitment"], keep_zero=("held_ticks",)) + "\n"


def _history_line(text: Any) -> str:
    """실행 이력 ``hist main=<ref> ph=<phase> path=<id> v=<lvl> f=<lvl> g=<open|closed> [stop] [gate=…] ack=<…> [fails=<n>]``."""
    fields = _history_fields(text)
    if not fields:
        return "hist none\n"
    item: dict[str, Any] = {}
    for key, value in fields.items():
        if key == "stop":
            item[key] = value not in ("0", "false", "")
        elif key in ("speed", "force", "fails"):
            try:
                item[key] = int(value)
            except ValueError:
                item[key] = value
        else:
            item[key] = value
    return _short("hist", item, STREAM_FIELDS["hist"], keep_zero=("speed", "force")) + "\n"


def _history_fields(text: Any) -> dict[str, str]:
    """``main=c1 phase=approach …`` → 필드. :func:`robo_jev.harness.robot.parse_exec_history`와 같은 규칙(모델 코드가 하네스를
    import하지 않도록 여기 다시 적는다; 검사가 두 파서를 대조한다)."""
    if not isinstance(text, str) or text in ("", "none"):
        return {}
    fields: dict[str, str] = {}
    for token in text.split():
        if "=" in token:
            key, value = token.split("=", 1)
            fields[key] = value
    return fields


def stream_candidate_line(entry: dict, *, geom_age_ms: Any = None, object_clearance: dict[str, Any] | None = None) -> str:
    """스트림의 후보 한 줄 (서식 v0.3): 결합 후보 ``<id>: <기능> <대상> <접근>→<목적지> d= [clr=] [path=blocked] [g=]``,
    밀기는 ``<접근>``만(목적지 없음), 고정 후보·경로 후보는 ``<id>: <키>`` / ``<id>: <종류> [<경유점>]``. 자연어 설명은
    없다 — 키가 설명이다. 기본값·중복은 뺀다: ``path=ok``는 적지 않고 막힌 경로만 ``path=blocked``; ``clr``(대상의 최근접
    여유)는 그 틱의 대상 물체 동적 줄 `clr`(`object_clearance[대상 id]`)와 같으면 중복이라 뺀다(하네스가 같은 값에서
    채운다); 대상 기하의 나이 ``g``는 틱의 기하 나이(`t … age g<ms>`)와 다를 때만(관측이 끊긴 대상) 적는다. 옛 서식의
    항목(`desc`·`derived`)은 버리고, 그 밖의 필드는 ``k=v``로 뒤에 붙는다.
    """
    rest = dict(entry)
    identifier = _scalar("id", rest.pop("id"))
    rest.pop("desc", None)
    rest.pop("description", None)
    rest.pop("derived", None)
    if geom_age_ms is not None and "g" in rest and int(rest["g"]) == int(geom_age_ms):
        rest.pop("g")
    if rest.get("path") == "ok":
        rest.pop("path")
    parts_of_key = joint_key_parts(rest.get("key"))
    if object_clearance is not None and parts_of_key is not None and "clr" in rest:
        target_clearance = object_clearance.get(parts_of_key[1])
        if target_clearance is not None and int(rest["clr"]) == int(target_clearance):
            rest.pop("clr")
    if rest.get("action_ref") == entry.get("id") or "kind" in rest:
        rest.pop("action_ref", None)
    words: list[str] = []
    key = rest.pop("key", None)
    if key is not None:
        if parts_of_key is not None:
            function, target, approach, destination = parts_of_key
            arrow = approach if function == "push" and destination == "none" else f"{approach}→{destination}"
            words.append(f"{function} {target} {arrow}")
            words.extend(str(key).split(":")[4:])
        else:
            words.append(str(key))
    kind = rest.pop("kind", None)
    if kind is not None:
        words.append(_scalar("kind", kind))
        ref = rest.pop("ref", None)
        if ref is not None:
            words.append(_scalar("ref", ref))
    tail = " ".join(f"{key}={_value(key, value)}" for key, value in rest.items())
    return f"{identifier}: " + " ".join(words + ([tail] if tail else [])) + "\n"


def joint_key_parts(key: Any) -> tuple[str, str, str, str] | None:
    """결합 키 ``기능:대상:접근:목적지``의 앞 네 조각 (:func:`robo_jev.harness.robot.joint_key_parts`와 같은 규칙 — 모델 코드가
    하네스를 import하지 않도록 여기 다시 적는다; 검사가 둘을 대조한다). 결합 키가 아니면 None."""
    parts = str(key or "").split(":")
    if len(parts) < 4 or parts[0] not in ("grasp", "place", "push"):
        return None
    return parts[0], parts[1], parts[2], parts[3]


# -- 변화분 상태 --------------------------------------------------------------------


def _quat_text(value: Any) -> str | None:
    return _value("quat", value) if isinstance(value, (list, tuple)) and value else None


class _DeltaState:
    """틱 사이에 마지막으로 실은 것을 기억한다 — 변화분 판정의 근거. 틱 0..k의 레코드만 읽는다."""

    def __init__(self, rules: dict[str, Any]) -> None:
        self.rules = rules
        self.intro: dict[str, str] = {}
        self.pose: dict[str, list[float]] = {}
        self.quat: dict[str, str | None] = {}
        self.conf: dict[str, Any] = {}
        self.visible: dict[str, float] = {}
        self.stale: dict[str, bool] = {}
        self.zones: str | None = None
        self.scene: str | None = None
        self.robot_quat: str | None = None

    def object_lines(self, state: dict, index: int) -> tuple[list[str], list[str]]:
        """(소개 줄들, 동적 줄들) — 이 틱에 실을 것만."""
        period_intro = int(self.rules["object_intro_period_ticks"])
        period_dynamic = int(self.rules["object_dynamic_period_ticks"])
        floor_mm = float(self.rules["object_pose_delta_mm"])
        vis_delta = float(self.rules["object_visibility_delta"])
        refresh_intro = period_intro > 0 and index % period_intro == 0
        refresh_dynamic = period_dynamic > 0 and index % period_dynamic == 0
        derived_by_object = {
            str(item.get("object")): item for item in state.get("derived") or () if isinstance(item, dict) and "object" in item
        }
        geom_age = None
        section = state.get("t")
        if isinstance(section, dict) and isinstance(section.get("age_ms"), dict):
            geom_age = section["age_ms"].get("geom")
        holding = (state.get("robot") or {}).get("holding")

        intros: list[str] = []
        dynamics: list[str] = []
        present = {str(entry["id"]) for entry in state.get("objects") or () if isinstance(entry, dict) and "id" in entry}
        for object_id in [known for known in self.intro if known not in present]:
            # 추적 목록에서 빠진 물체는 한 번 ``<id> gone``으로 알리고 잊는다 — 다시 나타나면 처음 관측처럼 전체 줄이다.
            dynamics.append(f"{_scalar('id', object_id)} gone\n")
            for table in (self.intro, self.pose, self.quat, self.conf, self.visible, self.stale):
                table.pop(object_id, None)
        for entry in state.get("objects") or ():
            if not isinstance(entry, dict) or "id" not in entry:
                continue
            object_id = str(entry["id"])
            intro = _intro_line(entry)
            if refresh_intro or self.intro.get(object_id) != intro:
                intros.append(intro)
                self.intro[object_id] = intro

            pose = [float(value) for value in entry.get("pose_mm") or ()]
            orientation = _orientation(entry.get("quat"))
            quat = None if orientation is None else _value(orientation[0], orientation[1])
            conf = entry.get("surface_conf")
            visible = float(entry.get("visible_ratio", 1.0) or 0.0)
            age = entry.get("age_ms")
            stale = bool(geom_age is not None and age is not None and float(age) > float(geom_age))
            sigma = float(entry.get("precision_mm", entry.get("pose_sigma_mm", 0.0)) or 0.0)
            threshold = max(sigma, floor_mm)
            previous = self.pose.get(object_id)
            first = previous is None
            moved = first or (len(previous) == len(pose) and pose and _dist(previous, pose) > threshold)
            rotated = quat != self.quat.get(object_id)
            conf_change = conf != self.conf.get(object_id)
            seen_change = stale != self.stale.get(object_id, False)
            visibility_change = abs(visible - self.visible.get(object_id, visible if first else 0.0)) >= vis_delta
            reid = bool(entry.get("reid"))
            held = holding is not None and str(holding) == object_id
            if refresh_dynamic or first or moved or rotated or seen_change or visibility_change or reid or held:
                # 소개 틱·처음에는 전체 줄, 그 밖에는 바뀐 선택 필드만 (윈도우 길이의 소개 주기가 전체 줄을 보장한다).
                full = first or refresh_intro
                optional = tuple(
                    key for key, changed in (("quat", rotated), ("surface_conf", conf_change), ("visible_ratio", visibility_change), ("reid", reid))
                    if full or changed
                )
                dynamics.append(_dynamic_line(entry, derived_by_object.get(object_id, {}), stale=stale, optional=optional, full=full))
                self.pose[object_id] = pose
                self.quat[object_id] = quat
                self.conf[object_id] = conf
                self.visible[object_id] = visible
                self.stale[object_id] = stale
        return intros, dynamics

    def zone_lines(self, state: dict) -> list[str]:
        zones = state.get("zones")
        if not isinstance(zones, list):
            return []
        lines = [_zone_line(zone) for zone in zones if isinstance(zone, dict)]
        text = "".join(lines)
        if text == self.zones:
            return []
        self.zones = text
        return lines

    def scene_lines(self, state: dict) -> list[str]:
        scene = state.get("scene")
        if not isinstance(scene, dict) or not scene:
            return []
        line = _scene_line(scene)
        if line == self.scene:
            return []
        self.scene = line
        return [line]

    def robot_line(self, state: dict, index: int) -> str | None:
        robot = state.get("robot")
        if not isinstance(robot, dict):
            return None
        quat = _quat_text(robot.get("ee_quat"))
        period = int(self.rules["object_intro_period_ticks"])
        with_quat = quat != self.robot_quat or (period > 0 and index % period == 0)
        self.robot_quat = quat
        return _robot_line(robot, with_quat=with_quat)


def _dist(a: list[float], b: list[float]) -> float:
    return sum((x - y) ** 2 for x, y in zip(a, b)) ** 0.5


_STATE_ORDER = ("t", "goal", "objects", "scene", "zones", "robot", "exec", "events", "derived", "commitment", "image", "geom", "extractor")
#: 모델 텍스트에 싣지 않는 상태 키. `image`·`geom`은 soft token 슬롯이라 비어 있어야 하고(값이 있으면 이 직렬화가 그것을 조용히
#: 버리게 되므로 거절한다), `extractor`는 앞단 버전 문자열(겉봉투 — `versions`와 함께 검증용)이라 문자열만 받는다.
_ENVELOPE_KEYS = ("image", "geom", "extractor")


def _check_envelope(state: dict, index: int) -> None:
    for key in ("image", "geom"):
        if state.get(key):
            raise ValueError(f"틱 {index}: state.{key}가 비어 있지 않다 — 서식 v0.3은 soft token 슬롯을 싣지 않으므로 조용히 버리지 않고 거절한다")
    extractor = state.get("extractor")
    if extractor not in (None, "") and not isinstance(extractor, str):
        raise ValueError(f"틱 {index}: state.extractor는 앞단 버전 문자열(겉봉투)이어야 한다 (받은 값: {type(extractor).__name__})")


def _extra_state_lines(state: dict) -> list[str]:
    """스키마 밖의 상태 키(다른 도구의 틱)는 옛 서식 그대로 뒤에 붙는다 — 정보를 잃지 않는다."""
    extra = {key: value for key, value in state.items() if key not in _STATE_ORDER}
    return state_lines(extra) if extra else []


def _serialize_stream(
    projected: dict, tokenizer: Any, *, window_ticks: int, rules: dict[str, Any]
) -> dict[str, Any]:
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
    delta = _DeltaState(rules)

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
    # 영역·장면 요약은 정적 prefix다 (첫 틱의 것; 바뀐 틱에만 다시 싣는다).
    if ticks:
        first_state = ticks[0]["request"].get("state") or {}
        zone_lines = delta.zone_lines(first_state)
        if zone_lines:
            chunks.append(_Chunk("".join(zone_lines), "prefix", "zones"))
        scene_lines = delta.scene_lines(first_state)
        if scene_lines:
            chunks.append(_Chunk("".join(scene_lines), "prefix", "scene"))
    prefix_chunks = len(chunks)

    slots, unplaced = _instruction_slots(instructions, ticks)
    goal_period = int(rules["goal_text_period_ticks"])
    for index, tick in enumerate(ticks):
        # 도중 추가되는 지시는 **그 틱의 토큰**이다 — prefix가 아니라 다른 틱 토큰과 같이 윈도우
        # 밖으로 나간다. 현재 지시는 매 틱 `goal`이 압축 참조로 다시 싣는다 (docs/08 §3.1).
        carries_instruction = bool(slots.get(index))
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
        state = request.get("state") or {}
        _check_envelope(state, index)
        geom_age = None
        if isinstance(state.get("t"), dict) and isinstance(state["t"].get("age_ms"), dict):
            geom_age = state["t"]["age_ms"].get("geom")
        object_clearance = {
            str(item["object"]): item.get("clearance_mm")
            for item in state.get("derived") or ()
            if isinstance(item, dict) and "object" in item and item.get("clearance_mm") is not None
        }

        def add(name: str, text: str, kind: str = "state") -> None:
            if text:
                chunks.append(_Chunk(text, kind, name, tick=index))

        add("state:t", _tick_header_line(tick))
        with_text = goal_period > 0 and index > 0 and index % goal_period == 0 and not carries_instruction
        add("state:goal", _goal_line(state.get("goal"), with_text=with_text))
        intros, dynamics = delta.object_lines(state, index)
        add("state:objects_intro", "".join(intros))
        add("state:objects_dynamic", "".join(dynamics))
        add("state:zones", "".join(delta.zone_lines(state)))
        add("state:scene", "".join(delta.scene_lines(state)))
        add("state:robot", delta.robot_line(state, index) or "")
        if isinstance(state.get("exec"), dict):
            add("state:exec", _exec_line(state["exec"]))
        add("state:events", "".join(_event_line(event) for event in state.get("events") or () if isinstance(event, dict)))
        add(
            "state:waypoints",
            "".join(_waypoint_line(item) for item in state.get("derived") or () if isinstance(item, dict) and "waypoint" in item),
        )
        add("state:extra", "".join(line + "\n" for line in _extra_state_lines(state)))
        add("commitment", _commitment_line(request.get("commitment")))
        add("exec_history", _history_line(request.get("exec_history")), kind="exec")
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
                        stream_candidate_line(entry, geom_age_ms=geom_age, object_clearance=object_clearance),
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
            "format": STREAM_FORMAT,
            "delta_rules": dict(rules),
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
    delta_rules: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """레코드 → 토큰·구간·분기·position (모듈 설명 참조).

    `request`는 `judgment-v0`(``state_first``) 또는 `stream-v0`(``stream_l1a``) 레코드다. 계약
    검사를 먼저 돌리므로 입력 영역의 비입력 키·라벨 구조는 경로가 붙은 `ValueError`로 거절되고,
    입력 영역 밖의 라벨·근거는 투영에서 빠져 결과에 영향을 주지 않는다. `max_*`는 L0 프로파일의
    상한(docs/06 Global Constraints)이며 넘으면 자르지 않고 오류다. `delta_rules`는 스트림의 변화분
    문턱·주기(:data:`DELTA_RULES`의 키를 덮어쓴다; 기본은 계약값).
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
    rules = dict(DELTA_RULES)
    for key, value in (delta_rules or {}).items():
        if key not in DELTA_RULES:
            raise ValueError(f"delta_rules.{key}: 모르는 규칙이다 (아는 것: {list(DELTA_RULES)})")
        rules[key] = value
    return _serialize_stream(projected, tokenizer, window_ticks=window_ticks, rules=rules)
