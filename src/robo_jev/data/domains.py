"""비로봇 네 분야의 규칙 기반 생성기 (docs/04 §2 표의 아래 세 줄 + 색·위치·영역 제약).

* ``spatial``  — 색·위치·영역 제약. 기하 규칙으로 대상·명제·수준을 낸다.
* ``dom``      — 합성 DOM/도구 상태. 트리의 유효 가시성·활성 여부로 낸다.
* ``workflow`` — 업무 흐름·자원 제약. 선행 조건과 임계 경로 solver로 낸다.
* ``rules``    — 명시 규칙의 우선순위·예외. 규칙 해석기로 낸다.

분야마다 셋을 나눠 둔다.

``make_scene``
    **사실만** 만든다. 난수는 여기서만 장면을 흔든다.
``pool``
    장면에서 질문 명세를 낸다. 난수를 쓰지 않는다 — 같은 장면은 같은 질문 목록을 준다.
``render``
    명세를 **표현**으로 바꾼다. 난수와 언어는 문장·설명만 고른다.

이렇게 나누는 이유는 하나다: **라벨은 (사실, 질문 명세)의 순수 함수**다. 표현 변형본은
같은 장면·같은 명세를 다른 문장으로 다시 그린 것이므로 정답이 그대로이고, 사실을 바꾸면
규칙이 다시 돌아 정답이 따라 바뀐다 (docs/04 §3). 후보의 순서와 id를 섞는 것은 이 모듈이
아니라 :mod:`robo_jev.data.generate`가 마지막에 한다 — 정답 위치 편향은 한 곳에서만 막는다.

기하·규칙 문제에는 **실제 물리 성공 라벨(`event`)을 붙이지 않는다**. 여기서 나오는 라벨
종류는 `valid_set`과 `single`뿐이다.
"""

from __future__ import annotations

import random
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "DOMAINS",
    "NONE_ID",
    "VARIANT_TAGS",
    "QuestionSpec",
    "Rendered",
]

#: "해당 없음·정보 부족" 후보의 id. 어떤 상태 요소도 가리키지 않으므로 `ref`가 없다.
NONE_ID = "c_na"

#: docs/04 §3의 난이도 변형. 생성기는 쓴 변형을 provenance에 적어 QA가 집계한다.
VARIANT_TAGS = (
    "near_miss",  # 한 조건만 위반한 후보
    "multiple_valid",  # 허용 답이 둘 이상
    "none_candidate",  # "해당 없음·정보 부족"이 정답
    "missing_info",  # 판단에 필요한 사실이 관측되지 않음
    "stale_observation",  # 오래된 관측
    "no_correct_candidate",  # 후보에서 정답을 뺀 변형
    "unfamiliar_term",  # 상태에 정의가 주어진 낯선 용어
    "long_candidates",  # 후보 설명이 긴 변형
    "boundary_level",  # 경계 구간이라 인접 수준을 함께 허용
    "paraphrase",  # 표현 변형본 (generate가 붙인다)
    "reorder",  # 후보 순서·id 재배열본 (generate가 붙인다)
)


@dataclass(frozen=True)
class QuestionSpec:
    """질문 하나의 **의미**. 표현은 들어 있지 않다."""

    id: str
    type: str  # choice | boolean | ordinal
    kind: str  # 분야 안의 규칙 이름
    params: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Rendered:
    """명세 하나를 그린 결과."""

    question: dict
    label: dict | None  # None이면 근거가 없어 마스킹한다
    trace: dict  # evidence에 들어갈 규칙 실행 근거
    variants: tuple[str, ...] = ()
    mask_reason: str | None = None


# --------------------------------------------------------------------------
# 공용 도우미
# --------------------------------------------------------------------------


#: 렌더링 중에 쓴 문구 템플릿 변형 id (`<표>.<언어>#<번호>`). :func:`take_phrasing`이 레코드마다 비운다 — 생성기가
#: provenance에 적고 `split.holdout_templates`와 맞댄다 (docs/04 §5 템플릿 변형 계열).
_PHRASING: list[str] = []
#: 봉인된 문구 변형 id와 그 추첨 몫(%). :func:`begin_phrasing` 이 생성 설정(`split.holdout_templates`·`sealed_phrasing_share`)에서
#: 넣는다 — 봉인 변형이 든 표에서는 그 변형을 이 몫만큼만 뽑아(나머지는 다른 변형이 고르게) 템플릿 계열이 OOD의 몫을 넘기지
#: 않게 한다. 봉인 변형이 없는 표는 고르게 뽑고 난수 소비도 그대로다.
_SEALED: frozenset[str] = frozenset()
_SEALED_SHARE: float | None = None


def _named(prefix: str, tables: dict[str, dict[str, tuple[str, ...]]]) -> dict[str, dict[str, Any]]:
    """문구 표에 id(`<분야>.<종류>`)를 붙인다 — `_say`가 어느 표의 몇 번째 변형을 썼는지 적을 수 있게."""
    for key, table in tables.items():
        table["_id"] = f"{prefix}.{key}"
    return tables


def _draw_variant(rng: random.Random, table: str, count: int) -> int:
    """표 `<id>.<언어>`의 변형 번호 하나. 봉인 변형이 든 표는 봉인 변형에 `_SEALED_SHARE` %를, 나머지 변형에 그 나머지를 고르게 준다;
    아니면 고르게(`randrange`, 도입 전과 같은 난수 소비)."""
    sealed = [index for index in range(count) if f"{table}#{index}" in _SEALED]
    if not sealed or _SEALED_SHARE is None or len(sealed) == count:
        return rng.randrange(count)
    share = _SEALED_SHARE / 100.0
    weights = [share / len(sealed) if index in sealed else (1.0 - share) / (count - len(sealed)) for index in range(count)]
    point = rng.random()
    edge = 0.0
    for index, weight in enumerate(weights):
        edge += weight
        if point < edge:
            return index
    return count - 1


def _say(rng: random.Random, texts: dict[str, Any], language: str, **values) -> str:
    options = texts[language]
    table = f"{texts.get('_id', 'text')}.{language}"
    index = _draw_variant(rng, table, len(options))
    _PHRASING.append(f"{table}#{index}")
    return options[index].format(**values)


def begin_phrasing(*, sealed: Iterable[str] = (), share: float | None = None) -> None:
    """레코드 하나의 렌더링을 시작한다 — 변형 기록을 비우고, 봉인 변형 목록(`split.holdout_templates`)과 그 추첨 몫(%,
    `sealed_phrasing_share`; None이면 고르게)을 건다."""
    global _SEALED, _SEALED_SHARE
    _PHRASING.clear()
    _SEALED = frozenset(str(item) for item in sealed)
    if share is not None and not 0.0 < float(share) < 100.0:
        raise ValueError(f"sealed_phrasing_share는 0과 100 사이의 퍼센트여야 한다 (받은 값: {share})")
    _SEALED_SHARE = None if share is None else float(share)


def take_phrasing() -> list[str]:
    """지금까지 쓴 문구 템플릿 변형 id(정렬·중복 제거)를 돌려주고 비운다."""
    found = sorted(set(_PHRASING))
    _PHRASING.clear()
    return found


def phrasing_vocabulary() -> dict[str, list[str]]:
    """분야별 문구 템플릿 변형 id 전부 (holdout 목록을 고를 때와 보고서에 쓴다)."""
    out: dict[str, list[str]] = {}
    for table in (_NONE_TEXT, *_SPATIAL_TEXT.values(), *_DOM_TEXT.values(), *_WORKFLOW_TEXT.values(), *_RULES_TEXT.values()):
        identifier = str(table["_id"])
        domain = identifier.split(".", 1)[0]
        for language in ("ko", "en"):
            out.setdefault(domain, []).extend(f"{identifier}.{language}#{index}" for index in range(len(table[language])))
    for language in ("ko", "en"):
        out.setdefault("yesno", []).extend(f"yesno.{language}#{index}" for index in range(len(_YES[language])))
    return {domain: sorted(ids) for domain, ids in out.items()}


def _cid(entity_id: str) -> str:
    """상태 요소 id → 후보 id. generate가 마지막에 무작위 id로 다시 붙인다."""
    return f"c_{entity_id}"


def _label_for(question_id: str, ids: list[str], rule: str, confidence: str) -> dict:
    """정답이 하나면 `single`, 허용 집합이면 `valid_set`."""
    if len(ids) == 1:
        return {
            "question_id": question_id,
            "kind": "single",
            "answer": ids[0],
            "source": rule,
            "label_confidence": confidence,
        }
    return {
        "question_id": question_id,
        "kind": "valid_set",
        "candidate_ids": list(ids),
        "source": rule,
        "label_confidence": confidence,
    }


def _boolean_criteria(rng: random.Random, language: str) -> list[dict]:
    """예/아니오 후보. 두 설명은 짝이 맞아야 하므로 같은 자리에서 고른다."""
    index = _draw_variant(rng, f"yesno.{language}", len(_YES[language]))
    _PHRASING.append(f"yesno.{language}#{index}")
    return [
        {"id": "true", "description": _YES[language][index]},
        {"id": "false", "description": _NO[language][index]},
    ]


_YES = {"ko": ("그렇다", "예", "맞다"), "en": ("Yes", "It does", "True")}
_NO = {"ko": ("아니다", "아니오", "그렇지 않다"), "en": ("No", "It does not", "False")}

_NONE_TEXT = {
    "_id": "none",
    "ko": (
        "해당하는 후보가 없거나 주어진 정보로 결정할 수 없음",
        "후보 중에 답이 없거나 정보가 부족함",
    ),
    "en": (
        "No candidate applies, or the given information is not enough",
        "None of the above, or not decidable from what is given",
    ),
}


def _choice_question(
    spec: QuestionSpec,
    instructions: str,
    entities: list[tuple[str, str]],
    *,
    answer: list[str],
    none_text: str | None,
    rule: str,
    trace: dict,
    confidence: str = "high",
    variants: tuple[str, ...] = (),
) -> Rendered:
    """선택 질문 하나를 만든다.

    `spec.params["drop_answer"]`가 참이면 정답 후보를 빼고, 그때는 "해당 없음" 후보가
    정답이 되거나(있을 때) 근거가 없어 마스킹한다 (docs/04 §3).
    """
    drop = bool(spec.params.get("drop_answer"))
    kept = [item for item in entities if not (drop and item[0] in answer)]
    criteria = [
        {"id": _cid(entity_id), "description": description, "ref": entity_id}
        for entity_id, description in kept
    ]
    if none_text is not None:
        criteria.append({"id": NONE_ID, "description": none_text})

    question = {
        "id": spec.id,
        "type": "choice",
        "instructions": instructions,
        "criteria": criteria,
    }
    kept_ids = {entity_id for entity_id, _ in kept}
    present = [_cid(entity_id) for entity_id in answer if entity_id in kept_ids]
    tags = list(variants)
    if drop and len(present) < len(answer):
        tags.append("no_correct_candidate")

    if present:
        if len(present) > 1:
            tags.append("multiple_valid")
        label = _label_for(spec.id, present, rule, confidence)
        trace = {**trace, "answer": present}
        return Rendered(question, label, trace, tuple(dict.fromkeys(tags)))

    if none_text is not None:
        tags.append("none_candidate")
        label = _label_for(spec.id, [NONE_ID], rule, confidence)
        trace = {**trace, "answer": [NONE_ID]}
        return Rendered(question, label, trace, tuple(dict.fromkeys(tags)))

    reason = "후보에 정답이 없고 '해당 없음' 후보도 없다"
    trace = {**trace, "answer": None, "masked": reason}
    return Rendered(question, None, trace, tuple(dict.fromkeys(tags)), mask_reason=reason)


def _boolean_rendered(
    spec: QuestionSpec,
    instructions: str,
    criteria: list[dict],
    answer: bool | None,
    *,
    rule: str,
    trace: dict,
    confidence: str = "high",
    variants: tuple[str, ...] = (),
    mask_reason: str = "관측되지 않은 사실에 답이 걸려 있다",
) -> Rendered:
    question = {
        "id": spec.id,
        "type": "boolean",
        "instructions": instructions,
        "criteria": criteria,
    }
    if answer is None:
        return Rendered(
            question,
            None,
            {**trace, "answer": None, "masked": mask_reason},
            variants,
            mask_reason=mask_reason,
        )
    label = {
        "question_id": spec.id,
        "kind": "single",
        "answer": answer,
        "source": rule,
        "label_confidence": confidence,
    }
    return Rendered(question, label, {**trace, "answer": answer}, variants)


def _ordinal_rendered(
    spec: QuestionSpec,
    instructions: str,
    descriptions: list[str],
    levels: list[str] | None,
    *,
    rule: str,
    trace: dict,
    confidence: str = "high",
    variants: tuple[str, ...] = (),
    boundary: bool = True,
    mask_reason: str = "수준을 계산할 사실이 관측되지 않았다",
) -> Rendered:
    criteria = [
        {"id": str(index), "description": description, "value": float(index)}
        for index, description in enumerate(descriptions)
    ]
    question = {
        "id": spec.id,
        "type": "ordinal",
        "instructions": instructions,
        "criteria": criteria,
    }
    if levels is None:
        return Rendered(
            question,
            None,
            {**trace, "answer": None, "masked": mask_reason},
            variants,
            mask_reason=mask_reason,
        )
    tags = list(variants)
    if len(levels) > 1:
        # 인접 수준을 함께 허용한 경계 사례와, 답이 여럿인 사례를 구분해 표시한다.
        tags += ["multiple_valid"] + (["boundary_level"] if boundary else [])
    label = _label_for(spec.id, levels, rule, confidence)
    return Rendered(question, label, {**trace, "answer": levels}, tuple(dict.fromkeys(tags)))


def _bucket(value: float, edges: tuple[float, ...], margin: float) -> list[str]:
    """값을 수준으로 나눈다. 구간은 반열린 `[edge_i, edge_{i+1})`이다.

    `margin`은 **상태에 적어 둔 허용 오차**일 때만 쓴다. 그때는 경계에서 오차 안에 든 값이
    두 수준 중 어느 쪽인지 상태로 가릴 수 없으므로 인접 수준을 함께 허용한다. 오차가 0이면
    (물체 개수처럼 정확히 세는 값) 경계에 정확히 걸린 값도 **위쪽 한 수준**에만 속한다 —
    `<=`로 비교하면 `1 == 1`인 개수가 "주변이 비어 있음"까지 정답이 되어 버린다.
    """
    level = 0
    for edge in edges:
        if value >= edge:
            level += 1
    levels = {level}
    if margin > 0:
        for index, edge in enumerate(edges):
            if abs(value - edge) <= margin:
                levels |= {index, index + 1}
    return [str(item) for item in sorted(levels)]


def _long(rng: random.Random) -> bool:
    """후보 설명을 길게 쓸 것인가 (후보 길이 변형)."""
    return rng.random() < 0.35


# ==========================================================================
# spatial — 색·위치·영역 제약
# ==========================================================================

_COLORS = ("red", "blue", "green", "yellow")
_COLOR_TEXT = {
    "ko": {"red": "빨간", "blue": "파란", "green": "초록", "yellow": "노란"},
    "en": {"red": "red", "blue": "blue", "green": "green", "yellow": "yellow"},
}
_ZONES = (("zoneL", -600, -100), ("zoneC", -100, 100), ("zoneR", 100, 600))
#: 지시의 목표 영역 비중 (docs/04 §5). 가운데 영역(zoneC)은 봉인 개념(`spatial:goal-zone:zoneC`)이라 드물게 둔다 — 그 계열은
#: 전부 OOD로 가므로 개념 태그가 아니라 **생성 비중**으로 OOD의 몫을 맞춘다. 관측된 대상의 영역을 목표로 삼는 가지에서는
#: 대상을 이 비중으로 고르고, 무작위 목표 가지에서는 영역을 이 비중으로 고른다.
_GOAL_ZONE_WEIGHTS = {"zoneL": 45, "zoneC": 10, "zoneR": 45}
_ZONE_TEXT = {
    "ko": {"zoneL": "왼쪽 영역", "zoneC": "가운데 영역", "zoneR": "오른쪽 영역"},
    "en": {"zoneL": "left zone", "zoneC": "centre zone", "zoneR": "right zone"},
}
_DISTANCE_LEVELS = {
    "ko": ["근접역 안", "가까움", "보통", "멂"],
    "en": ["inside the near band", "close", "medium", "far"],
}
_CROWDING_LEVELS = {
    "ko": ["주변이 비어 있음", "여유 있음", "붐빔", "매우 붐빔"],
    "en": ["clear around it", "some room", "crowded", "very crowded"],
}
#: 혼잡 수준의 경계 (관측된 이웃 수). 정확히 세는 값이라 허용 오차가 없다.
_CROWDING_EDGES = (1.0, 2.0, 3.0)

_SPATIAL_TEXT = _named("spatial", {
    "target": {
        "ko": (
            "{color} 물체 중 {zone} 안에 있는 것을 고르라.",
            "{zone}에 놓인 {color} 물체를 고르라.",
            "지금 관측으로 {zone} 안의 {color} 물체를 하나 고르라.",
        ),
        "en": (
            "Pick the {color} object that is inside the {zone}.",
            "Which {color} object is located in the {zone}?",
            "From the current observation, choose the {color} object in the {zone}.",
        ),
    },
    "nearest": {
        "ko": (
            "원점에서 가장 가까운 {color} 물체를 고르라.",
            "{color} 물체 중 원점과의 거리가 가장 짧은 것을 고르라.",
        ),
        "en": (
            "Pick the {color} object nearest to the origin.",
            "Which {color} object has the shortest distance to the origin?",
        ),
    },
    "in_zone": {
        "ko": ("{entity}는 {zone} 안에 있는가.", "{entity}의 위치가 {zone}에 들어가는가."),
        "en": (
            "Is {entity} inside the {zone}?",
            "Does the position of {entity} fall in the {zone}?",
        ),
    },
    # 아래 셋은 **관측된 것만** 두고 묻는다. 가려지거나 오래된 물체가 있어도 답이 상태에서
    # 나오려면 질문이 그 범위를 말해야 한다 — 모든 표현에 "관측"/"observed"가 들어간다.
    "goal_met": {
        "ko": (
            "지금 관측으로 {zone} 안의 {color} 물체를 확인할 수 있는가.",
            "현재 관측만으로 {zone}의 {color} 물체 조건이 충족됐다고 말할 수 있는가.",
            "관측된 물체만 보면 {zone} 안에 {color} 물체가 있는가.",
        ),
        "en": (
            "Does the current observation show a {color} object inside the {zone}?",
            "Can the {color}-object condition in the {zone} be called satisfied from the observation alone?",
            "Among the observed objects, is there a {color} object in the {zone}?",
        ),
    },
    "distance": {
        "ko": (
            "관측된 {color} 물체 중 가장 가까운 것까지의 거리 수준을 고르라. 경계와 허용 오차는 상태의 thresholds에 있다.",
            "지금 관측된 {color} 물체 가운데 원점에 가장 가까운 것의 거리 수준을 고르라 (근접역 정의는 상태에 있다).",
        ),
        "en": (
            "Choose the distance level to the nearest observed {color} object; the cut points and tolerance are in the state thresholds.",
            "Pick the distance level of the closest observed {color} object (see the near band definition in the state).",
        ),
    },
    "crowding": {
        "ko": (
            "관측된 물체만 세어 {entity} 주변의 혼잡 수준을 고르라.",
            "{entity} 둘레에 관측된 물체가 몇이나 되는지로 혼잡 수준을 고르라.",
        ),
        "en": (
            "Counting only the observed objects, choose how crowded the area around {entity} is.",
            "Pick the congestion level around {entity} from the observed objects alone.",
        ),
    },
})


def _zone_of_x(x: int | None) -> str | None:
    if x is None:
        return None
    for zone_id, x_min, x_max in _ZONES:
        if x_min <= x < x_max:
            return zone_id
    return None


def _spatial_observable(scene: dict, obj: dict) -> bool:
    return bool(obj["visible"]) and obj["age_ms"] <= scene["stale_after_ms"]


def _spatial_distance(obj: dict) -> float | None:
    if obj["x"] is None or obj["y"] is None:
        return None
    return (obj["x"] ** 2 + obj["y"] ** 2) ** 0.5


def _spatial_object(scene: dict, object_id: str) -> dict:
    return next(obj for obj in scene["objects"] if obj["id"] == object_id)


def _spatial_describe(scene: dict, obj: dict, language: str, long: bool) -> str:
    color = _COLOR_TEXT[language].get(obj["color"], obj["color"])
    if language == "ko":
        head = f"{color} 물체 {obj['id']}"
        if not _spatial_observable(scene, obj):
            return f"{head} (관측이 없거나 오래됨)"
        return f"{head} (x={obj['x']}mm, y={obj['y']}mm)" if long else head
    head = f"{color} object {obj['id']}"
    if not _spatial_observable(scene, obj):
        return f"{head} (not observed, or the observation is stale)"
    return f"{head} (x={obj['x']}mm, y={obj['y']}mm)" if long else head


class SpatialDomain:
    """색·위치·영역 제약. 정답은 관측된 자세에 대한 기하 규칙이다."""

    name = "spatial"
    templates = ("zone-color", "zone-color-stale", "zone-color-occluded")

    def make_scene(self, rng: random.Random, template: str) -> dict:
        count = rng.randint(3, 6)
        objects = [
            {
                "id": f"o{index + 1}",
                "color": rng.choice(_COLORS),
                "x": rng.randrange(-580, 585, 5),
                "y": rng.randrange(-300, 305, 5),
                "visible": True,
                "age_ms": rng.choice((40, 60, 80, 120)),
            }
            for index in range(count)
        ]
        if template == "zone-color-stale":
            for obj in rng.sample(objects, k=min(2, count)):
                obj["age_ms"] = rng.choice((420, 900, 1500))
        if template == "zone-color-occluded":
            for obj in rng.sample(objects, k=1):
                obj["visible"] = False
                obj["x"] = None
                obj["y"] = None
        # 목표는 대개 실제로 관측된 물체를 가리킨다. 늘 무작위 (색, 영역) 조합을 쓰면
        # "해당 없음"만 정답인 문제가 대부분이 되어 라벨이 한쪽으로 쏠린다.
        reachable = [obj for obj in objects if obj["visible"] and obj["age_ms"] <= 300]
        if reachable and rng.random() < 0.7:
            target = rng.choices(reachable, weights=[_GOAL_ZONE_WEIGHTS[_zone_of_x(obj["x"])] for obj in reachable], k=1)[0]
            goal = {"color": target["color"], "zone": _zone_of_x(target["x"])}
        else:
            zone_ids = [zone[0] for zone in _ZONES]
            goal = {"color": rng.choice(_COLORS), "zone": rng.choices(zone_ids, weights=[_GOAL_ZONE_WEIGHTS[z] for z in zone_ids], k=1)[0]}
        return {
            "domain": self.name,
            "template": template,
            "observed_at_ms": rng.randrange(600, 9950, 50),
            "stale_after_ms": 300,
            "near_zone_mm": rng.choice((200, 250, 300)),
            "crowd_radius_mm": rng.choice((180, 220, 260)),
            # 관측 자세의 오차. 거리 수준의 경계에서 이 안에 들면 인접 수준도 허용한다.
            "distance_tolerance_mm": rng.choice((15, 20, 25)),
            "objects": objects,
            "zones": [
                {"id": zone_id, "x_min": x_min, "x_max": x_max}
                for zone_id, x_min, x_max in _ZONES
            ],
            "goal": goal,
            "drop_answer_at": rng.randrange(0, 8),
        }

    def state(self, scene: dict, rng: random.Random, language: str) -> dict:
        long = _long(rng)
        zone_text = _ZONE_TEXT[language]
        near = scene["near_zone_mm"]
        term = (
            f"원점에서 {near}mm 이내"
            if language == "ko"
            else f"within {near}mm of the origin"
        )
        goal_color = _COLOR_TEXT[language][scene["goal"]["color"]]
        goal_zone = zone_text[scene["goal"]["zone"]]
        goal_text = (
            f"{goal_color} 물체를 {goal_zone}으로 옮긴다"
            if language == "ko"
            else f"Move the {goal_color} object into the {goal_zone}"
        )
        return {
            "observed_at_ms": scene["observed_at_ms"],
            "goal": {
                "text": goal_text,
                "color": scene["goal"]["color"],
                "zone": scene["goal"]["zone"],
            },
            # 수준의 경계와 허용 오차는 상태에 적는다. 생성기 상수를 몰라도 답이 나와야 한다.
            "thresholds": {
                "stale_after_ms": scene["stale_after_ms"],
                "near_zone_mm": near,
                "distance_edges_mm": [near, near * 2, near * 3],
                "distance_tolerance_mm": scene["distance_tolerance_mm"],
                "crowd_radius_mm": scene["crowd_radius_mm"],
                "crowding_edges": list(_CROWDING_EDGES),
            },
            "glossary": {
                "근접역" if language == "ko" else "near band": term,
                "오래된 관측"
                if language == "ko"
                else "stale observation": (
                    f"age_ms가 {scene['stale_after_ms']}보다 큰 관측"
                    if language == "ko"
                    else f"an observation whose age_ms exceeds {scene['stale_after_ms']}"
                ),
                "거리 수준"
                if language == "ko"
                else "distance level": (
                    f"thresholds.distance_edges_mm를 경계로 0~3. 경계에서 "
                    f"{scene['distance_tolerance_mm']}mm 안이면 두 수준을 모두 허용한다."
                    if language == "ko"
                    else (
                        "0-3 split at thresholds.distance_edges_mm; within "
                        f"{scene['distance_tolerance_mm']}mm of a cut point both levels are accepted."
                    )
                ),
                "혼잡 수준"
                if language == "ko"
                else "congestion level": (
                    f"crowd_radius_mm 안의 관측된 다른 물체 수를 "
                    f"{list(_CROWDING_EDGES)} 경계로 나눈 0~3 (정확히 세므로 오차 없음)"
                    if language == "ko"
                    else (
                        "the number of other observed objects within crowd_radius_mm, cut at "
                        f"{list(_CROWDING_EDGES)} into 0-3 (an exact count, so no tolerance)"
                    )
                ),
            },
            "objects": [
                {
                    "id": obj["id"],
                    "desc": _spatial_describe(scene, obj, language, long),
                    "color": obj["color"],
                    "x_mm": obj["x"],
                    "y_mm": obj["y"],
                    "visible": obj["visible"],
                    "age_ms": obj["age_ms"],
                }
                for obj in scene["objects"]
            ],
            "zones": [
                {
                    "id": zone["id"],
                    "desc": zone_text[zone["id"]],
                    "x_min_mm": zone["x_min"],
                    "x_max_mm": zone["x_max"],
                }
                for zone in scene["zones"]
            ],
        }

    def pool(self, scene: dict) -> list[QuestionSpec]:
        ids = [obj["id"] for obj in scene["objects"]]
        zones = [zone["id"] for zone in scene["zones"]]
        goal = scene["goal"]
        # 관측된 물체가 만드는 (색, 영역) 쌍을 먼저 쓰고, 남는 자리는 조합 전체로 채운다.
        # 그래야 "해당 없음"만 정답인 문제로 쏠리지 않는다.
        pairs = [(goal["color"], goal["zone"])]
        pairs += [
            (obj["color"], _zone_of_x(obj["x"]))
            for obj in scene["objects"]
            if _spatial_observable(scene, obj) and _zone_of_x(obj["x"]) is not None
        ]
        pairs = list(dict.fromkeys(pairs))
        # 아무 물체도 만족하지 않는 조합은 몇 개만 섞는다. 전부 넣으면 "해당 없음"만
        # 정답인 문제가 대다수가 되어 규칙을 배울 신호가 사라진다.
        unmatched = [
            (color, zone) for color in _COLORS for zone in zones if (color, zone) not in pairs
        ]
        pairs = pairs[:6] + unmatched[:2]
        specs: list[QuestionSpec] = []
        for index, (color, zone) in enumerate(pairs):
            matching, _, _ = self._matches(scene, color, zone)
            # 후보 개수를 흔들되 정답은 남긴다 — 정답을 빼는 것은 `drop_answer`만 한다.
            droppable = [object_id for object_id in ids if object_id not in matching]
            subset = (
                [object_id for object_id in ids if object_id != droppable[-1]]
                if index % 3 == 0 and droppable and len(ids) > 2
                else ids
            )
            specs.append(
                QuestionSpec(
                    f"q_target_{index}",
                    "choice",
                    "target",
                    {
                        "color": color,
                        "zone": zone,
                        "candidates": list(subset),
                        "none": index % 4 != 3,
                        "drop_answer": index == scene["drop_answer_at"],
                    },
                )
            )
        # 장면에 있는 색을 먼저 묻고, 없는 색은 하나만 남긴다 ("해당 없음" 사례).
        present = list(
            dict.fromkeys(
                obj["color"] for obj in scene["objects"] if _spatial_observable(scene, obj)
            )
        )
        absent = [color for color in _COLORS if color not in present]
        asked_colors = present + absent[:1]
        for index, color in enumerate(asked_colors):
            specs.append(
                QuestionSpec(
                    f"q_nearest_{index}",
                    "choice",
                    "nearest",
                    {"color": color, "candidates": list(ids), "none": True},
                )
            )
        for index, object_id in enumerate(ids):
            zone = zones[index % len(zones)]
            specs.append(
                QuestionSpec(
                    f"q_in_zone_{index}", "boolean", "in_zone", {"object": object_id, "zone": zone}
                )
            )
        for index, (color, zone) in enumerate(pairs[:4]):
            specs.append(
                QuestionSpec(f"q_goal_met_{index}", "boolean", "goal_met", {"color": color, "zone": zone})
            )
        for index, color in enumerate(asked_colors):
            specs.append(QuestionSpec(f"q_distance_{index}", "ordinal", "distance", {"color": color}))
        for index, object_id in enumerate(ids):
            specs.append(
                QuestionSpec(f"q_crowding_{index}", "ordinal", "crowding", {"object": object_id})
            )
        return specs

    def render(
        self, scene: dict, spec: QuestionSpec, rng: random.Random, language: str
    ) -> Rendered:
        long = _long(rng)
        tags = ("long_candidates",) if long else ()
        if scene["template"] == "zone-color-stale":
            tags += ("stale_observation",)
        if scene["template"] == "zone-color-occluded":
            tags += ("missing_info",)
        zone_text = _ZONE_TEXT[language]
        color_text = _COLOR_TEXT[language]

        if spec.kind in ("target", "nearest"):
            entities = [
                (object_id, _spatial_describe(scene, _spatial_object(scene, object_id), language, long))
                for object_id in spec.params["candidates"]
            ]
            none_text = _say(rng, _NONE_TEXT, language) if spec.params["none"] else None
            if spec.kind == "target":
                answer, near_miss, unknown = self._matches(
                    scene, spec.params["color"], spec.params["zone"]
                )
                instructions = _say(
                    rng,
                    _SPATIAL_TEXT["target"],
                    language,
                    color=color_text[spec.params["color"]],
                    zone=zone_text[spec.params["zone"]],
                )
                rule = "spatial/target-in-zone-v0"
                trace = {
                    "question_id": spec.id,
                    "rule": rule,
                    "color": spec.params["color"],
                    "zone": spec.params["zone"],
                    "near_miss": near_miss,
                    "unobservable": unknown,
                }
            else:
                answer, near_miss, unknown = self._nearest(scene, spec.params["color"])
                instructions = _say(
                    rng,
                    _SPATIAL_TEXT["nearest"],
                    language,
                    color=color_text[spec.params["color"]],
                )
                rule = "spatial/nearest-observed-v0"
                if unknown:
                    # "가장 가까운"은 모든 같은 색 물체를 견줘야 정해진다. 하나라도 관측되지
                    # 않으면 그것이 더 가까울 수 있으므로 관측된 물체를 정답이라 할 수 없다
                    # — "해당 없음·정보 부족"이 유일하게 도출되는 답이다 (docs/04 §3).
                    answer = []
                trace = {
                    "question_id": spec.id,
                    "rule": rule,
                    "color": spec.params["color"],
                    "near_miss": near_miss,
                    "unobservable": unknown,
                    "undecidable": bool(unknown),
                }
            if near_miss:
                tags += ("near_miss",)
            if unknown:
                tags += ("missing_info",)
            return _choice_question(
                spec,
                instructions,
                entities,
                answer=answer,
                none_text=none_text,
                rule=rule,
                trace=trace,
                # nearest는 정보가 모자라면 그 사실 자체가 정답이라 근거가 약하지 않다.
                confidence="high" if spec.kind == "nearest" or not unknown else "medium",
                variants=tags,
            )

        if spec.kind == "in_zone":
            obj = _spatial_object(scene, spec.params["object"])
            zone = spec.params["zone"]
            observable = _spatial_observable(scene, obj)
            answer = (_zone_of_x(obj["x"]) == zone) if observable else None
            instructions = _say(
                rng,
                _SPATIAL_TEXT["in_zone"],
                language,
                entity=_spatial_describe(scene, obj, language, False),
                zone=zone_text[zone],
            )
            rule = "spatial/zone-membership-v0"
            return _boolean_rendered(
                spec,
                instructions,
                _boolean_criteria(rng, language),
                answer,
                rule=rule,
                trace={
                    "question_id": spec.id,
                    "rule": rule,
                    "object": obj["id"],
                    "zone": zone,
                    "observable": observable,
                },
                variants=tags if observable else tags + ("missing_info",),
                mask_reason="대상의 자세가 관측되지 않았거나 오래됐다",
            )

        if spec.kind == "goal_met":
            answer, near_miss, unknown = self._matches(
                scene, spec.params["color"], spec.params["zone"]
            )
            instructions = _say(
                rng,
                _SPATIAL_TEXT["goal_met"],
                language,
                color=color_text[spec.params["color"]],
                zone=zone_text[spec.params["zone"]],
            )
            rule = "spatial/goal-satisfied-v0"
            # 질문이 "지금 관측으로 확인되는가"를 묻는다. 가려진 물체가 있어도 답은
            # 관측된 물체만으로 정해지므로 근거가 약해지지 않는다 (in_zone처럼 특정 물체의
            # 자세를 묻는 질문만 관측이 없을 때 마스킹한다).
            return _boolean_rendered(
                spec,
                instructions,
                _boolean_criteria(rng, language),
                bool(answer),
                rule=rule,
                trace={
                    "question_id": spec.id,
                    "rule": rule,
                    "scope": "observed",
                    "matched": answer,
                    "unobservable": unknown,
                },
                variants=tags + (("missing_info",) if unknown else ()),
            )

        if spec.kind == "distance":
            distances = [
                _spatial_distance(obj)
                for obj in scene["objects"]
                if obj["color"] == spec.params["color"] and _spatial_observable(scene, obj)
            ]
            nearest = min((value for value in distances if value is not None), default=None)
            near = scene["near_zone_mm"]
            edges = (float(near), float(near) * 2, float(near) * 3)
            tolerance = float(scene["distance_tolerance_mm"])
            levels = None if nearest is None else _bucket(nearest, edges, margin=tolerance)
            instructions = _say(
                rng, _SPATIAL_TEXT["distance"], language, color=color_text[spec.params["color"]]
            )
            rule = "spatial/distance-level-v0"
            return _ordinal_rendered(
                spec,
                instructions,
                list(_DISTANCE_LEVELS[language]),
                levels,
                rule=rule,
                trace={
                    "question_id": spec.id,
                    "rule": rule,
                    "nearest_mm": None if nearest is None else round(nearest, 1),
                    "edges_mm": list(edges),
                    "tolerance_mm": tolerance,
                },
                variants=tags + ("unfamiliar_term",),
                mask_reason="그 색의 관측된 물체가 없다",
            )

        obj = _spatial_object(scene, spec.params["object"])
        radius = scene["crowd_radius_mm"]
        if not _spatial_observable(scene, obj):
            neighbours = None
        else:
            neighbours = sum(
                1
                for other in scene["objects"]
                if other["id"] != obj["id"]
                and _spatial_observable(scene, other)
                and abs(other["x"] - obj["x"]) <= radius
                and abs(other["y"] - obj["y"]) <= radius
            )
        # 이웃 수는 정확히 세는 값이다 — 경계에 딱 걸려도 수준은 하나다.
        levels = (
            None if neighbours is None else _bucket(float(neighbours), _CROWDING_EDGES, margin=0.0)
        )
        instructions = _say(
            rng,
            _SPATIAL_TEXT["crowding"],
            language,
            entity=_spatial_describe(scene, obj, language, False),
        )
        rule = "spatial/crowding-level-v0"
        return _ordinal_rendered(
            spec,
            instructions,
            list(_CROWDING_LEVELS[language]),
            levels,
            rule=rule,
            trace={
                "question_id": spec.id,
                "rule": rule,
                "object": obj["id"],
                "neighbours": neighbours,
                "radius_mm": radius,
            },
            variants=tags,
            mask_reason="대상이 관측되지 않아 주변을 셀 수 없다",
        )

    def _matches(self, scene: dict, color: str, zone: str) -> tuple[list[str], list[str], list[str]]:
        """(정답, 한 조건만 위반한 후보, 판단 불가 후보)."""
        answer: list[str] = []
        near_miss: list[str] = []
        unknown: list[str] = []
        for obj in scene["objects"]:
            color_ok = obj["color"] == color
            if not _spatial_observable(scene, obj):
                if color_ok:
                    unknown.append(obj["id"])
                continue
            zone_ok = _zone_of_x(obj["x"]) == zone
            if color_ok and zone_ok:
                answer.append(obj["id"])
            elif color_ok or zone_ok:
                near_miss.append(obj["id"])
        return answer, near_miss, unknown

    def _nearest(self, scene: dict, color: str) -> tuple[list[str], list[str], list[str]]:
        best: list[str] = []
        best_distance: float | None = None
        near_miss: list[str] = []
        unknown: list[str] = []
        for obj in scene["objects"]:
            if not _spatial_observable(scene, obj):
                if obj["color"] == color:
                    unknown.append(obj["id"])
                continue
            if obj["color"] != color:
                near_miss.append(obj["id"])
                continue
            distance = _spatial_distance(obj)
            assert distance is not None
            if best_distance is None or distance < best_distance - 1e-9:
                best, best_distance = [obj["id"]], distance
            elif abs(distance - best_distance) <= 1e-9:
                best.append(obj["id"])
        return best, near_miss, unknown


# ==========================================================================
# dom — 합성 DOM/도구 상태
# ==========================================================================

_DOM_NAMES = {
    "checkout": {"ko": "결제", "en": "Checkout"},
    "card": {"ko": "카드 번호", "en": "Card number"},
    "coupon": {"ko": "쿠폰 코드", "en": "Coupon code"},
    "submit": {"ko": "결제 제출", "en": "Submit payment"},
    "details": {"ko": "상세 정보", "en": "Details"},
    "expand": {"ko": "상세 정보 펼치기", "en": "Expand details"},
    "help": {"ko": "도움말", "en": "Help"},
    "cancel": {"ko": "취소", "en": "Cancel"},
    "terms": {"ko": "약관 동의", "en": "Accept terms"},
    "saved": {"ko": "저장된 카드로 결제", "en": "Pay with saved card"},
}
_DOM_ROLE_TEXT = {
    "ko": {
        "section": "구역",
        "button": "버튼",
        "link": "링크",
        "input": "입력란",
        "checkbox": "체크박스",
    },
    "en": {
        "section": "section",
        "button": "button",
        "link": "link",
        "input": "input",
        "checkbox": "checkbox",
    },
}
#: 비활성 사유. 상태에는 토큰으로 두고 뜻은 용어 정의에 적는다 (언어와 무관한 사실).
_DOM_REASONS = {
    "no_permission": {"ko": "권한이 없다", "en": "no permission"},
    "card_expired": {"ko": "카드가 만료됐다", "en": "the card has expired"},
}
_DOM_PROGRESS = {
    "ko": ["시작 전", "입력 중", "거의 다 됨", "제출 가능"],
    "en": ["not started", "filling in", "almost ready", "ready to submit"],
}
#: 진행 수준 = (만족한 조건 수 ÷ 전체 조건 수) × scale을 내림한 값. 정확히 세므로 오차 없음.
_DOM_PROGRESS_SCALE = 3
_DOM_PROGRESS_EDGES = (1.0, 2.0, 3.0)
_DOM_TEXT = _named("dom", {
    "action": {
        "ko": (
            "목표를 진행하려면 지금 어떤 요소를 조작해야 하는가.",
            "지금 시점에 조작할 요소를 하나 고르라.",
            "다음 조작 대상이 될 요소를 고르라.",
        ),
        "en": (
            "Which element should be acted on now to move the goal forward?",
            "Pick the element to interact with at this point.",
            "Choose the next element to act on.",
        ),
    },
    "blocker": {
        "ko": ("목표 달성을 막고 있는 요소를 고르라.", "지금 목표를 막는 요소는 무엇인가."),
        "en": (
            "Pick the element that blocks the goal right now.",
            "Which element is currently blocking the goal?",
        ),
    },
    "reachable": {
        "ko": (
            "지금 화면 상태에서 목표를 끝까지 달성할 수 있는가.",
            "이 상태에서 목표를 완료할 방법이 있는가.",
        ),
        "en": (
            "Can the goal be completed from the current page state?",
            "Is there a way to finish the goal in this state?",
        ),
    },
    "needs_input": {
        "ko": ("지금 입력이 더 필요한가.", "비어 있는 필수 입력란이 남아 있는가."),
        "en": ("Is more input needed right now?", "Is any required field still empty?"),
    },
    "actionable": {
        "ko": ("{entity}를 지금 조작할 수 있는가.", "{entity}가 지금 활성 요소인가."),
        "en": ("Can {entity} be acted on right now?", "Is {entity} an active element right now?"),
    },
    "progress": {
        "ko": ("목표 진행 수준을 고르라.", "지금까지의 진행 수준을 고르라."),
        "en": ("Choose the progress level towards the goal.", "Pick how far the goal has come."),
    },
})


def _dom_element(scene: dict, element_id: str) -> dict:
    return next(element for element in scene["elements"] if element["id"] == element_id)


def _dom_visible(scene: dict, element: dict) -> bool | None:
    """유효 가시성: 자신과 모든 조상이 보여야 한다. 관측 불가면 None."""
    current: dict | None = element
    while current is not None:
        if current.get("unknown"):
            return None
        if not current["visible"]:
            return False
        parent = current.get("parent")
        current = None if parent is None else _dom_element(scene, parent)
    return True


def _dom_actionable(scene: dict, element: dict) -> bool | None:
    visible = _dom_visible(scene, element)
    if visible is None or element.get("unknown"):
        return None
    return visible and element["enabled"]


def _dom_pending_inputs(scene: dict) -> tuple[list[str], bool]:
    """비어 있는 필수 입력란과, 관측되지 않아 알 수 없는 것이 있었는지."""
    pending: list[str] = []
    unknown = False
    for element in scene["elements"]:
        if element["role"] not in ("input", "checkbox") or not element.get("required"):
            continue
        actionable = _dom_actionable(scene, element)
        if actionable is None:
            unknown = True
        elif actionable and not element.get("value"):
            pending.append(element["id"])
    return pending, unknown


def _dom_reveal_control(scene: dict, target_id: str) -> str | None:
    for element in scene["elements"]:
        if element.get("reveals") == target_id and _dom_actionable(scene, element):
            return element["id"]
    return None


def _dom_describe(scene: dict, element: dict, language: str, long: bool) -> str:
    """후보 설명. 긴 형태는 트리를 걸어 얻은 **파생값**을 덧붙인다.

    `effective_visible`은 상태의 `visible`(그 요소 자체의 속성)과 다르다 — 상위 요소가
    접혀 있으면 자신은 visible이어도 유효 가시성은 거짓이다. 이름을 나눠 두 값이
    모순처럼 보이지 않게 한다.
    """
    name = _DOM_NAMES[element["name_key"]][language]
    role = _DOM_ROLE_TEXT[language][element["role"]]
    head = f"{name} {role} ({element['id']})"
    if not long:
        return head
    dash = "—" if language == "ko" else "-"
    if element.get("unknown"):
        unreadable = "상태를 읽지 못했다" if language == "ko" else "state could not be read"
        return f"{head} {dash} {unreadable}"
    return (
        f"{head} {dash} effective_visible={_dom_visible(scene, element)}, "
        f"enabled={element['enabled']}"
    )


class DomDomain:
    """합성 DOM/도구 상태. 정답은 트리의 유효 가시성·활성 여부 규칙이다."""

    name = "dom"
    templates = ("checkout-form", "checkout-collapsed", "checkout-unreadable")
    #: 장면 종류의 비중 (docs/04 §5). 접힌 구역(`checkout-collapsed`)은 봉인 개념 `dom:reveal`의 근원이라 드물게 둔다 — 개념
    #: 태그가 아니라 생성 비중으로 OOD의 몫(계열의 ≈8~10 %)을 맞춘다.
    template_weights = (45, 10, 45)

    def make_scene(self, rng: random.Random, template: str) -> dict:
        elements = [
            {
                "id": "e1",
                "role": "section",
                "parent": None,
                "name_key": "checkout",
                "visible": True,
                "enabled": True,
            },
            {
                "id": "e2",
                "role": "input",
                "parent": "e1",
                "name_key": "card",
                "visible": True,
                "enabled": True,
                "required": True,
                "value": rng.choice(("", "4111-****", "5500-1234", "6011-9876")),
            },
            {
                "id": "e3",
                "role": "input",
                "parent": "e1",
                "name_key": "coupon",
                "visible": True,
                "enabled": rng.random() < 0.8,
                "required": False,
                "value": rng.choice(("", "", "SAVE10")),
            },
            {
                "id": "e4",
                "role": "checkbox",
                "parent": "e1",
                "name_key": "terms",
                "visible": True,
                "enabled": True,
                "required": True,
                "value": rng.choice(("", "on")),
            },
            {
                "id": "e5",
                "role": "section",
                "parent": "e1",
                "name_key": "details",
                "visible": True,
                "enabled": True,
            },
            {
                # 목표 요소. 가끔 영구히 비활성이라 목표 자체를 이룰 수 없다.
                "id": "e6",
                "role": "button",
                "parent": "e5",
                "name_key": "submit",
                "visible": True,
                **(
                    {"enabled": False, "disabled_reason": rng.choice(tuple(_DOM_REASONS))}
                    if rng.random() < 0.25
                    else {"enabled": True, "disabled_reason": None}
                ),
            },
            {
                # 이름이 비슷하지만 조건이 다른 후보 (한 조건만 위반).
                "id": "e7",
                "role": "button",
                "parent": rng.choice(("e1", "e5")),
                "name_key": "saved",
                "visible": True,
                "enabled": False,
                "disabled_reason": rng.choice(tuple(_DOM_REASONS)),
            },
            {
                "id": "e8",
                "role": "button",
                "parent": "e1",
                "name_key": "expand",
                "visible": rng.random() < 0.85,
                "enabled": True,
                "reveals": "e5",
            },
        ]
        # 잡음 요소. 화면마다 개수·이름·역할·상태가 다르다 — 서로 다른 계열이 같은
        # 사실로 수렴하면 QA가 계보 중복으로 잡으므로 장면 공간을 넓게 둔다.
        for index in range(rng.randint(2, 5)):
            elements.append(
                {
                    "id": f"e{len(elements) + 1}",
                    "role": rng.choice(("button", "link", "checkbox")),
                    "parent": rng.choice(("e1", "e5")),
                    "name_key": rng.choice(("help", "cancel", "saved", "coupon", "details")),
                    "visible": rng.random() < 0.8,
                    "enabled": rng.random() < 0.6,
                }
            )
        if template == "checkout-collapsed":
            _dom_set(elements, "e5", visible=False)
        if template == "checkout-unreadable":
            _dom_set(elements, rng.choice(("e4", "e6", "e7")), unknown=True)
        return {
            "domain": self.name,
            "template": template,
            "observed_at_ms": rng.randrange(600, 9950, 50),
            "elements": elements,
            "goal": {"kind": "submit", "element": "e6"},
            "drop_answer_at": rng.randrange(0, 6),
        }

    def state(self, scene: dict, rng: random.Random, language: str) -> dict:
        long = _long(rng)
        goal_name = _DOM_NAMES[_dom_element(scene, scene["goal"]["element"])["name_key"]][language]
        goal_text = (
            f"{goal_name} 버튼까지 진행해 결제를 제출한다"
            if language == "ko"
            else f"Work through the page and press {goal_name}"
        )
        return {
            "observed_at_ms": scene["observed_at_ms"],
            "goal": {
                "text": goal_text,
                "kind": scene["goal"]["kind"],
                "element": scene["goal"]["element"],
            },
            "thresholds": {
                "progress_scale": _DOM_PROGRESS_SCALE,
                "progress_edges": list(_DOM_PROGRESS_EDGES),
            },
            "glossary": {
                "활성 요소"
                if language == "ko"
                else "active element": (
                    "자신과 모든 상위 요소가 visible이고 enabled가 참인 요소"
                    if language == "ko"
                    else "an element that is enabled and visible together with all of its ancestors"
                ),
                "진행 수준"
                if language == "ko"
                else "progress level": (
                    "(만족한 조건 수 ÷ 전체 조건 수) × progress_scale을 내림한 0~3. 조건은 필수 "
                    "입력란 각각과 '목표 요소가 활성인가'다 (정확히 세므로 오차 없음)."
                    if language == "ko"
                    else (
                        "floor of (satisfied conditions / all conditions) x progress_scale, 0-3. The "
                        "conditions are each required field plus 'is the goal element active' "
                        "(an exact count, so no tolerance)."
                    )
                ),
                **{token: text[language] for token, text in _DOM_REASONS.items()},
            },
            "elements": [
                {
                    "id": element["id"],
                    "desc": _dom_describe(scene, element, language, long),
                    "role": element["role"],
                    "parent": element["parent"],
                    "name": _DOM_NAMES[element["name_key"]][language],
                    **(
                        {"observed": False}
                        if element.get("unknown")
                        else {
                            "observed": True,
                            "visible": element["visible"],
                            "enabled": element["enabled"],
                        }
                    ),
                    **({"required": True} if element.get("required") else {}),
                    **({"value": element["value"]} if "value" in element else {}),
                    **(
                        {"disabled_reason": element["disabled_reason"]}
                        if element.get("disabled_reason")
                        else {}
                    ),
                    **({"reveals": element["reveals"]} if element.get("reveals") else {}),
                }
                for element in scene["elements"]
            ],
        }

    def pool(self, scene: dict) -> list[QuestionSpec]:
        ids = [element["id"] for element in scene["elements"]]
        leaves = [
            element["id"] for element in scene["elements"] if element["role"] != "section"
        ]
        specs: list[QuestionSpec] = []
        # 같은 규칙을 후보 개수·"해당 없음" 후보 유무만 바꿔 여러 번 낸다 (docs/04 §3).
        for index in range(6):
            subset = leaves if index % 2 else leaves[: max(3, len(leaves) - 2)]
            specs.append(
                QuestionSpec(
                    f"q_action_{index}",
                    "choice",
                    "action",
                    {
                        "candidates": list(subset),
                        "none": index % 3 != 2,
                        "drop_answer": index == scene["drop_answer_at"],
                    },
                )
            )
        for index in range(4):
            specs.append(
                QuestionSpec(
                    f"q_blocker_{index}",
                    "choice",
                    "blocker",
                    {"candidates": list(ids if index % 2 else leaves), "none": True},
                )
            )
        for index in range(3):
            specs.append(QuestionSpec(f"q_reachable_{index}", "boolean", "reachable", {}))
            specs.append(QuestionSpec(f"q_needs_input_{index}", "boolean", "needs_input", {}))
        for index, element_id in enumerate(leaves):
            specs.append(
                QuestionSpec(
                    f"q_actionable_{index}", "boolean", "actionable", {"element": element_id}
                )
            )
        for index in range(4):
            specs.append(QuestionSpec(f"q_progress_{index}", "ordinal", "progress", {}))
        return specs

    def render(
        self, scene: dict, spec: QuestionSpec, rng: random.Random, language: str
    ) -> Rendered:
        long = _long(rng)
        tags = ("long_candidates",) if long else ()
        tags += ("unfamiliar_term",)  # 상태의 용어 정의("활성 요소")를 읽어야 답이 나온다
        goal_id = scene["goal"]["element"]
        goal = _dom_element(scene, goal_id)
        pending, pending_unknown = _dom_pending_inputs(scene)
        goal_actionable = _dom_actionable(scene, goal)
        reveal = _dom_reveal_control(scene, goal["parent"]) if goal["parent"] else None
        if goal_actionable is None or pending_unknown:
            tags += ("missing_info",)

        if spec.kind == "action":
            answer, reason, unknown = self._next_action(scene, pending, pending_unknown)
            entities = [
                (element_id, _dom_describe(scene, _dom_element(scene, element_id), language, long))
                for element_id in spec.params["candidates"]
            ]
            near_miss = [
                element["id"]
                for element in scene["elements"]
                if element["id"] not in answer
                and element["role"] == "button"
                and _dom_actionable(scene, element) is not True
            ]
            rule = "dom/next-action-v0"
            return _choice_question(
                spec,
                _say(rng, _DOM_TEXT["action"], language),
                entities,
                answer=answer,
                none_text=_say(rng, _NONE_TEXT, language) if spec.params["none"] else None,
                rule=rule,
                trace={
                    "question_id": spec.id,
                    "rule": rule,
                    "reason": reason,
                    "pending_inputs": pending,
                    "near_miss": near_miss,
                },
                confidence="medium" if unknown else "high",
                variants=tags + (("near_miss",) if near_miss else ()),
            )

        if spec.kind == "blocker":
            blockers, unknown = self._blockers(scene, pending, pending_unknown)
            entities = [
                (element_id, _dom_describe(scene, _dom_element(scene, element_id), language, long))
                for element_id in spec.params["candidates"]
            ]
            rule = "dom/blocker-v0"
            return _choice_question(
                spec,
                _say(rng, _DOM_TEXT["blocker"], language),
                entities,
                answer=blockers,
                none_text=_say(rng, _NONE_TEXT, language),
                rule=rule,
                trace={"question_id": spec.id, "rule": rule, "blockers": blockers},
                confidence="medium" if unknown else "high",
                variants=tags,
            )

        if spec.kind == "reachable":
            if goal_actionable is None or pending_unknown:
                answer = None
            else:
                answer = bool(
                    not goal.get("disabled_reason")
                    and (goal_actionable or reveal is not None)
                )
            rule = "dom/goal-reachable-v0"
            return _boolean_rendered(
                spec,
                _say(rng, _DOM_TEXT["reachable"], language),
                _boolean_criteria(rng, language),
                answer,
                rule=rule,
                trace={
                    "question_id": spec.id,
                    "rule": rule,
                    "goal": goal_id,
                    "goal_actionable": goal_actionable,
                    "reveal_control": reveal,
                },
                variants=tags,
                mask_reason="목표 요소나 필수 입력란의 상태를 읽지 못했다",
            )

        if spec.kind == "needs_input":
            answer = None if pending_unknown and not pending else bool(pending)
            rule = "dom/needs-input-v0"
            return _boolean_rendered(
                spec,
                _say(rng, _DOM_TEXT["needs_input"], language),
                _boolean_criteria(rng, language),
                answer,
                rule=rule,
                trace={"question_id": spec.id, "rule": rule, "pending_inputs": pending},
                variants=tags,
                mask_reason="필수 입력란의 상태를 읽지 못했다",
            )

        if spec.kind == "actionable":
            element = _dom_element(scene, spec.params["element"])
            answer = _dom_actionable(scene, element)
            rule = "dom/actionable-v0"
            return _boolean_rendered(
                spec,
                _say(
                    rng,
                    _DOM_TEXT["actionable"],
                    language,
                    entity=_dom_describe(scene, element, language, False),
                ),
                _boolean_criteria(rng, language),
                answer,
                rule=rule,
                trace={
                    "question_id": spec.id,
                    "rule": rule,
                    "element": element["id"],
                    "visible": _dom_visible(scene, element),
                },
                variants=tags,
                mask_reason="그 요소의 상태를 읽지 못했다",
            )

        required = [
            element
            for element in scene["elements"]
            if element.get("required") and _dom_actionable(scene, element) is not None
        ]
        if goal_actionable is None or pending_unknown:
            levels = None
            done = total = None
        else:
            filled = sum(1 for element in required if element.get("value"))
            total = len(required) + 1
            done = filled + (1 if goal_actionable else 0)
            # 만족한 조건 수 ÷ 전체 조건 수 × 3을 내림한 값. 정확히 세는 값이라 오차가 없다.
            levels = _bucket(done / total * _DOM_PROGRESS_SCALE, _DOM_PROGRESS_EDGES, margin=0.0)
        rule = "dom/progress-level-v0"
        return _ordinal_rendered(
            spec,
            _say(rng, _DOM_TEXT["progress"], language),
            list(_DOM_PROGRESS[language]),
            levels,
            rule=rule,
            trace={
                "question_id": spec.id,
                "rule": rule,
                "required": [element["id"] for element in required],
                "goal_actionable": goal_actionable,
                "satisfied": done,
                "total": total,
            },
            variants=tags,
            mask_reason="진행 수준을 셀 요소의 상태를 읽지 못했다",
        )

    def _next_action(
        self, scene: dict, pending: list[str], pending_unknown: bool
    ) -> tuple[list[str], str, bool]:
        goal = _dom_element(scene, scene["goal"]["element"])
        if goal.get("disabled_reason") and not goal.get("unknown"):
            # 목표 요소가 영구히 비활성이면 입력란을 채워도 소용이 없다.
            return [], f"목표 요소가 비활성이다: {goal['disabled_reason']}", False
        if pending:
            return sorted(pending), "필수 입력란이 비어 있다", pending_unknown
        actionable = _dom_actionable(scene, goal)
        if actionable is None or pending_unknown:
            return [], "목표 요소나 입력란의 상태를 읽지 못했다", True
        if actionable:
            return [goal["id"]], "목표 요소를 바로 누를 수 있다", False
        reveal = _dom_reveal_control(scene, goal["parent"]) if goal["parent"] else None
        if reveal is not None:
            return [reveal], "목표 요소를 드러내는 조작이 먼저 필요하다", False
        return [], "조작할 수 있는 요소가 없다", False

    def _blockers(
        self, scene: dict, pending: list[str], pending_unknown: bool
    ) -> tuple[list[str], bool]:
        goal = _dom_element(scene, scene["goal"]["element"])
        actionable = _dom_actionable(scene, goal)
        if actionable is None or pending_unknown:
            return [], True
        if goal.get("disabled_reason"):
            return [goal["id"]], False
        if pending:
            return sorted(pending), False
        if actionable:
            return [], False
        if _dom_visible(scene, goal) is False:
            hidden = goal if not goal["visible"] else _dom_element(scene, goal["parent"])
            return [hidden["id"]], False
        return [goal["id"]], False


def _dom_set(elements: list[dict], element_id: str, **changes) -> None:
    for element in elements:
        if element["id"] == element_id:
            element.update(changes)
            return
    raise KeyError(element_id)  # pragma: no cover - 각본 오류


# ==========================================================================
# workflow — 업무 흐름·자원 제약
# ==========================================================================

_WORKFLOW_NAMES = {
    "spec": {"ko": "명세 확정", "en": "Freeze the spec"},
    "build": {"ko": "빌드", "en": "Build"},
    "review": {"ko": "코드 검토", "en": "Code review"},
    "qa": {"ko": "QA", "en": "QA"},
    "docs": {"ko": "문서화", "en": "Documentation"},
    "stage": {"ko": "스테이징 배포", "en": "Deploy to staging"},
    "release": {"ko": "릴리스", "en": "Release"},
}
_RESOURCE_NAMES = {
    "r1": {"ko": "개발 담당", "en": "Engineer"},
    "r2": {"ko": "검토 담당", "en": "Reviewer"},
    "r3": {"ko": "QA 담당", "en": "QA owner"},
}
_PRIORITY_LEVELS = {
    "ko": ["여유 있음", "계획대로", "서둘러야 함", "즉시"],
    "en": ["slack left", "on plan", "hurry", "immediate"],
}
_LOAD_LEVELS = {
    "ko": ["한가함", "보통", "빠듯함", "초과"],
    "en": ["idle", "normal", "tight", "over capacity"],
}
#: 시급도의 경계 (마감까지 남은 여유 시간). 여유가 24시간 이하면 1, 8시간 이하면 2, 0 이하면 3.
_URGENCY_EDGES_H = (24, 8, 0)
#: 부하의 경계 (남은 시간 ÷ 남은 용량).
_LOAD_EDGES = (0.5, 1.0, 1.5)
_WORKFLOW_TEXT = _named("workflow", {
    "next": {
        "ko": ("지금 착수할 수 있는 단계를 고르라.", "선행 조건과 자원을 모두 만족하는 단계를 고르라."),
        "en": (
            "Pick a step that can be started right now.",
            "Choose the step whose prerequisites and resource are both satisfied.",
        ),
    },
    "owner": {
        "ko": ("{entity}를 수행할 자원을 고르라.", "{entity}의 담당 자원은 무엇인가."),
        "en": ("Pick the resource that must run {entity}.", "Who owns {entity}?"),
    },
    "precondition": {
        "ko": ("{entity}의 선행 조건이 모두 충족되었는가.", "{entity}를 막는 선행 단계가 남아 있지 않은가."),
        "en": (
            "Are all prerequisites of {entity} satisfied?",
            "Is nothing left blocking {entity} among its prerequisites?",
        ),
    },
    "blocked": {
        "ko": ("자원이 모자라 막힌 단계가 남아 있는가.", "남은 단계 중 자원 제약으로 막힌 것이 있는가."),
        "en": (
            "Is any remaining step blocked by its resource?",
            "Does a resource constraint block one of the remaining steps?",
        ),
    },
    "priority": {
        "ko": ("{entity}의 시급도 수준을 고르라.", "마감까지의 여유로 {entity}의 시급도를 고르라."),
        "en": (
            "Choose the urgency level of {entity}.",
            "Pick the urgency of {entity} given the slack to the deadline.",
        ),
    },
    "load": {
        "ko": ("{entity}의 부하 수준을 고르라.", "{entity}에 남은 일이 용량 대비 어느 수준인가."),
        "en": (
            "Choose the load level of {entity}.",
            "How much work is left on {entity} relative to its capacity?",
        ),
    },
})

_WORKFLOW_CHAIN = (
    ("s1", "spec", [], "r1"),
    ("s2", "build", ["s1"], "r1"),
    ("s3", "review", ["s2"], "r2"),
    ("s4", "qa", ["s2"], "r3"),
    ("s5", "docs", ["s1"], "r2"),
    ("s6", "stage", ["s3", "s4"], "r1"),
    ("s7", "release", ["s6", "s5"], "r3"),
)


def _workflow_step(scene: dict, step_id: str) -> dict:
    return next(step for step in scene["steps"] if step["id"] == step_id)


def _workflow_resource(scene: dict, resource_id: str) -> dict | None:
    for resource in scene["resources"]:
        if resource["id"] == resource_id:
            return resource
    return None


def _workflow_resource_ok(scene: dict, step: dict) -> bool:
    resource = _workflow_resource(scene, step["resource"])
    return (
        resource is not None
        and resource["status"] == "available"
        and resource["available_h"] >= step["hours"]
    )


def _workflow_ready(scene: dict, step: dict) -> bool | None:
    if step["done"] is None:
        return None
    if step["done"]:
        return False
    for required in step["requires"]:
        state = _workflow_step(scene, required)["done"]
        if state is None:
            return None
        if not state:
            return False
    return _workflow_resource_ok(scene, step)


def _workflow_critical_path(scene: dict, step_id: str) -> int:
    """이 단계부터 끝까지 남은 최장 경로(시간). DAG이므로 재귀로 충분하다."""
    step = _workflow_step(scene, step_id)
    successors = [
        other for other in scene["steps"] if step_id in other["requires"] and other["done"] is not True
    ]
    best = max((_workflow_critical_path(scene, other["id"]) for other in successors), default=0)
    return step["hours"] + best


def _workflow_describe(scene: dict, step: dict, language: str, long: bool) -> str:
    name = _WORKFLOW_NAMES[step["name_key"]][language]
    head = f"{name} ({step['id']})"
    if not long:
        return head
    if language == "ko":
        return f"{head} — {step['hours']}시간, 선행 {step['requires'] or '없음'}"
    return f"{head} - {step['hours']}h, after {step['requires'] or 'nothing'}"


class WorkflowDomain:
    """업무 흐름·자원 제약. 정답은 상태 기계와 임계 경로 solver가 낸다."""

    name = "workflow"
    templates = ("release-train", "release-train-scarce", "release-train-unreported")
    #: 장면 종류의 비중 (docs/04 §5). 자원이 모자란 장면(`release-train-scarce`; 그 60 %에서 자원 하나가 오프라인)은 봉인 개념
    #: `workflow:resource_offline`의 근원이라 드물게 둔다 — 개념 태그가 아니라 생성 비중으로 OOD의 몫(계열의 ≈8~10 %)을 맞춘다.
    template_weights = (42.5, 15, 42.5)

    def make_scene(self, rng: random.Random, template: str) -> dict:
        done_upto = rng.randint(0, 3)
        steps = [
            {
                "id": step_id,
                "name_key": name_key,
                "requires": list(requires),
                "resource": resource,
                "hours": rng.choice((2, 4, 6, 8)),
                "done": index < done_upto,
            }
            for index, (step_id, name_key, requires, resource) in enumerate(_WORKFLOW_CHAIN)
        ]
        # 용량은 대개 넉넉하다. 늘 모자라면 "막힌 단계가 있는가"의 답이 언제나 예가 된다.
        resources = [
            {"id": resource_id, "available_h": rng.choice((6, 10, 14, 18)), "status": "available"}
            for resource_id in ("r1", "r2", "r3")
        ]
        if template == "release-train-scarce" and rng.random() < 0.6:
            rng.choice(resources)["status"] = "offline"
        if template == "release-train-unreported":
            # 진행 보고가 없는 단계: done을 알 수 없다 (필수 정보 결측).
            candidates = [step for step in steps if not step["done"]]
            rng.choice(candidates)["done"] = None
            if rng.random() < 0.4:
                # 담당 자원이 명단에 없다 — 담당을 물으면 "정보 부족"이 답이다.
                resources.pop(rng.randrange(len(resources)))
        return {
            "domain": self.name,
            "template": template,
            "observed_at_ms": rng.randrange(600, 9950, 50),
            "deadline_h": rng.choice((12, 20, 28, 40)),
            # 소요 시간은 추정치다. 경계에서 이 안에 들면 인접 수준도 허용한다.
            "schedule_tolerance_h": rng.choice((1, 2)),
            "load_tolerance": 0.05,
            "steps": steps,
            "resources": resources,
            "drop_answer_at": rng.randrange(0, 6),
        }

    def state(self, scene: dict, rng: random.Random, language: str) -> dict:
        long = _long(rng)
        goal_text = (
            f"마감 {scene['deadline_h']}시간 안에 릴리스까지 끝낸다"
            if language == "ko"
            else f"Finish the release within {scene['deadline_h']} hours"
        )
        return {
            "observed_at_ms": scene["observed_at_ms"],
            "goal": {"text": goal_text, "deadline_h": scene["deadline_h"]},
            "thresholds": {
                "urgency_edges_h": list(_URGENCY_EDGES_H),
                "schedule_tolerance_h": scene["schedule_tolerance_h"],
                "load_edges": list(_LOAD_EDGES),
                "load_tolerance": scene["load_tolerance"],
            },
            "glossary": {
                "착수 가능"
                if language == "ko"
                else "startable": (
                    "선행 단계가 모두 끝났고 담당 자원이 가용하며 남은 용량이 소요 시간 이상인 단계"
                    if language == "ko"
                    else "a step whose prerequisites are done and whose resource is available with enough capacity"
                ),
                "시급도"
                if language == "ko"
                else "urgency level": (
                    "여유 = 마감 − 그 단계부터 끝까지의 최장 경로 시간. urgency_edges_h를 경계로 "
                    f"0~3이고, 소요 시간이 ±{scene['schedule_tolerance_h']}시간의 추정치라 경계에서 "
                    "그 안에 들면 두 수준을 모두 허용한다."
                    if language == "ko"
                    else (
                        "slack = deadline - the longest remaining path from that step; cut at "
                        f"urgency_edges_h into 0-3. Hours are estimates within "
                        f"+/-{scene['schedule_tolerance_h']}h, so near a cut point both levels are accepted."
                    )
                ),
                "부하 수준"
                if language == "ko"
                else "load level": (
                    "남은 단계의 소요 시간 합 ÷ 남은 용량을 load_edges로 나눈 0~3. 경계에서 "
                    f"{scene['load_tolerance']} 안이면 두 수준을 모두 허용한다."
                    if language == "ko"
                    else (
                        "the remaining hours on the resource divided by its capacity, cut at load_edges "
                        f"into 0-3; within {scene['load_tolerance']} of a cut point both levels are accepted."
                    )
                ),
            },
            "steps": [
                {
                    "id": step["id"],
                    "desc": _workflow_describe(scene, step, language, long),
                    "name": _WORKFLOW_NAMES[step["name_key"]][language],
                    "requires": list(step["requires"]),
                    "resource": step["resource"],
                    "hours": step["hours"],
                    **({"done": step["done"]} if step["done"] is not None else {"reported": False}),
                }
                for step in scene["steps"]
            ],
            "resources": [
                {
                    "id": resource["id"],
                    "desc": _RESOURCE_NAMES[resource["id"]][language],
                    "available_h": resource["available_h"],
                    "status": resource["status"],
                }
                for resource in scene["resources"]
            ],
        }

    def pool(self, scene: dict) -> list[QuestionSpec]:
        step_ids = [step["id"] for step in scene["steps"]]
        resource_ids = [resource["id"] for resource in scene["resources"]]
        specs: list[QuestionSpec] = []
        for index in range(4):
            subset = step_ids if index % 2 else step_ids[: max(3, len(step_ids) - 2)]
            specs.append(
                QuestionSpec(
                    f"q_next_{index}",
                    "choice",
                    "next",
                    {
                        "candidates": list(subset),
                        "none": index % 3 != 2,
                        "drop_answer": index == scene["drop_answer_at"],
                    },
                )
            )
        for index, step_id in enumerate(step_ids):
            specs.append(
                QuestionSpec(
                    f"q_owner_{index}",
                    "choice",
                    "owner",
                    {"step": step_id, "candidates": list(resource_ids), "none": True},
                )
            )
            specs.append(
                QuestionSpec(
                    f"q_precondition_{index}", "boolean", "precondition", {"step": step_id}
                )
            )
            specs.append(QuestionSpec(f"q_priority_{index}", "ordinal", "priority", {"step": step_id}))
        for index in range(3):
            specs.append(QuestionSpec(f"q_blocked_{index}", "boolean", "blocked", {}))
        for index, resource_id in enumerate(resource_ids):
            specs.append(QuestionSpec(f"q_load_{index}", "ordinal", "load", {"resource": resource_id}))
        return specs

    def render(
        self, scene: dict, spec: QuestionSpec, rng: random.Random, language: str
    ) -> Rendered:
        long = _long(rng)
        tags = ("long_candidates",) if long else ()
        tags += ("unfamiliar_term",)  # "착수 가능"의 정의가 상태에 있다
        unreported = [step["id"] for step in scene["steps"] if step["done"] is None]
        if unreported:
            tags += ("missing_info",)

        if spec.kind == "next":
            ready = [
                step["id"] for step in scene["steps"] if _workflow_ready(scene, step) is True
            ]
            undetermined = [
                step["id"] for step in scene["steps"] if _workflow_ready(scene, step) is None
            ]
            near_miss = [
                step["id"]
                for step in scene["steps"]
                if step["id"] not in ready
                and step["done"] is False
                and (
                    not _workflow_resource_ok(scene, step)
                    or sum(
                        1
                        for required in step["requires"]
                        if _workflow_step(scene, required)["done"] is not True
                    )
                    == 1
                )
            ]
            entities = [
                (step_id, _workflow_describe(scene, _workflow_step(scene, step_id), language, long))
                for step_id in spec.params["candidates"]
            ]
            rule = "workflow/ready-step-v0"
            return _choice_question(
                spec,
                _say(rng, _WORKFLOW_TEXT["next"], language),
                entities,
                answer=ready if not (undetermined and not ready) else [],
                none_text=_say(rng, _NONE_TEXT, language) if spec.params["none"] else None,
                rule=rule,
                trace={
                    "question_id": spec.id,
                    "rule": rule,
                    "ready": ready,
                    "undetermined": undetermined,
                    "near_miss": near_miss,
                },
                confidence="medium" if undetermined else "high",
                variants=tags + (("near_miss",) if near_miss else ()),
            )

        if spec.kind == "owner":
            step = _workflow_step(scene, spec.params["step"])
            resource = _workflow_resource(scene, step["resource"])
            entities = [
                (resource_id, _RESOURCE_NAMES[resource_id][language])
                for resource_id in spec.params["candidates"]
            ]
            rule = "workflow/step-owner-v0"
            return _choice_question(
                spec,
                _say(
                    rng,
                    _WORKFLOW_TEXT["owner"],
                    language,
                    entity=_workflow_describe(scene, step, language, False),
                ),
                entities,
                answer=[resource["id"]] if resource is not None else [],
                none_text=_say(rng, _NONE_TEXT, language),
                rule=rule,
                trace={
                    "question_id": spec.id,
                    "rule": rule,
                    "step": step["id"],
                    "resource": step["resource"],
                },
                variants=tags,
            )

        if spec.kind == "precondition":
            step = _workflow_step(scene, spec.params["step"])
            states = [_workflow_step(scene, required)["done"] for required in step["requires"]]
            answer = None if any(state is None for state in states) else all(states)
            rule = "workflow/preconditions-v0"
            return _boolean_rendered(
                spec,
                _say(
                    rng,
                    _WORKFLOW_TEXT["precondition"],
                    language,
                    entity=_workflow_describe(scene, step, language, False),
                ),
                _boolean_criteria(rng, language),
                answer,
                rule=rule,
                trace={
                    "question_id": spec.id,
                    "rule": rule,
                    "step": step["id"],
                    "requires": step["requires"],
                },
                variants=tags,
                mask_reason="선행 단계의 진행 보고가 없다",
            )

        if spec.kind == "blocked":
            blocked = [
                step["id"]
                for step in scene["steps"]
                if step["done"] is False and not _workflow_resource_ok(scene, step)
            ]
            answer = True if blocked else (None if unreported else False)
            rule = "workflow/resource-blocked-v0"
            return _boolean_rendered(
                spec,
                _say(rng, _WORKFLOW_TEXT["blocked"], language),
                _boolean_criteria(rng, language),
                answer,
                rule=rule,
                trace={"question_id": spec.id, "rule": rule, "blocked": blocked},
                variants=tags,
                mask_reason="진행 보고가 없는 단계가 있어 남은 단계를 확정할 수 없다",
            )

        if spec.kind == "priority":
            step = _workflow_step(scene, spec.params["step"])
            if step["done"] is None:
                levels = None
                slack = None
            else:
                slack = scene["deadline_h"] - _workflow_critical_path(scene, step["id"])
                # 여유가 적을수록 높은 수준이므로 부호를 뒤집어 구간을 나눈다.
                levels = _bucket(
                    float(-slack),
                    tuple(float(-edge) for edge in _URGENCY_EDGES_H),
                    margin=float(scene["schedule_tolerance_h"]),
                )
            rule = "workflow/urgency-level-v0"
            return _ordinal_rendered(
                spec,
                _say(
                    rng,
                    _WORKFLOW_TEXT["priority"],
                    language,
                    entity=_workflow_describe(scene, step, language, False),
                ),
                list(_PRIORITY_LEVELS[language]),
                levels,
                rule=rule,
                trace={
                    "question_id": spec.id,
                    "rule": rule,
                    "step": step["id"],
                    "slack_h": slack,
                    "deadline_h": scene["deadline_h"],
                },
                variants=tags,
                mask_reason="그 단계의 진행 보고가 없어 남은 경로를 셀 수 없다",
            )

        resource = _workflow_resource(scene, spec.params["resource"])
        assert resource is not None
        remaining = [
            step
            for step in scene["steps"]
            if step["resource"] == resource["id"] and step["done"] is not True
        ]
        if any(step["done"] is None for step in remaining):
            levels = None
            load = None
        else:
            load = sum(step["hours"] for step in remaining) / max(resource["available_h"], 1)
            levels = _bucket(load, _LOAD_EDGES, margin=float(scene["load_tolerance"]))
        rule = "workflow/resource-load-v0"
        return _ordinal_rendered(
            spec,
            _say(
                rng,
                _WORKFLOW_TEXT["load"],
                language,
                entity=_RESOURCE_NAMES[resource["id"]][language],
            ),
            list(_LOAD_LEVELS[language]),
            levels,
            rule=rule,
            trace={
                "question_id": spec.id,
                "rule": rule,
                "resource": resource["id"],
                "load": None if load is None else round(load, 3),
            },
            variants=tags,
            mask_reason="그 자원에 남은 일을 확정할 수 없다",
        )


# ==========================================================================
# rules — 명시 규칙의 우선순위와 예외
# ==========================================================================

_RULE_ATTRS = {
    "zone": {"ko": "구역", "en": "zone"},
    "window": {"ko": "시간대", "en": "time window"},
    "load": {"ko": "적재량", "en": "load"},
    "approval": {"ko": "승인", "en": "approval"},
}
_RULE_VALUES = {
    "night": {"ko": "야간", "en": "night"},
    "day": {"ko": "주간", "en": "day"},
    "restricted": {"ko": "통제 구역", "en": "restricted"},
    "open": {"ko": "일반 구역", "en": "open"},
    "heavy": {"ko": "과적", "en": "heavy"},
    "light": {"ko": "정상", "en": "light"},
    "granted": {"ko": "있음", "en": "granted"},
    "absent": {"ko": "없음", "en": "absent"},
}
_RULE_ACTIONS = {
    "a_stop": {"ko": "즉시 중지", "en": "Stop immediately"},
    "a_slow": {"ko": "감속 진입", "en": "Enter slowly"},
    "a_escort": {"ko": "안내 요청", "en": "Request an escort"},
    "a_proceed": {"ko": "그대로 진행", "en": "Proceed as planned"},
}
#: 규칙 조치의 추첨 비중 (docs/04 §5). 안내 요청(`a_escort`)뿐인 지배 조치는 봉인 개념 `rules:escort-policy`의 근원이라 드물게
#: 둔다 — 개념 태그가 아니라 생성 비중으로 OOD의 몫(계열의 ≈8~10 %)을 맞춘다(고르게 뽑으면 seed에 따라 계열의 10~17 %).
_RULE_ACTION_WEIGHTS = {"a_stop": 30, "a_slow": 30, "a_escort": 24, "a_proceed": 30}
_SEVERITY_LEVELS = {
    "ko": ["기록만", "주의", "경고", "중대"],
    "en": ["log only", "caution", "warning", "critical"],
}
_RULES_TEXT = _named("rules", {
    "governing": {
        "ko": (
            "이 상황을 지배하는 규칙을 고르라. 우선순위 숫자가 작을수록 앞선다.",
            "적용할 규칙을 고르라 (우선순위 1이 가장 앞선다).",
        ),
        "en": (
            "Pick the rule that governs this situation; a smaller priority number wins.",
            "Choose the rule to apply (priority 1 is the strongest).",
        ),
    },
    "action": {
        "ko": ("지배 규칙이 지시하는 조치를 고르라.", "규칙에 따라 취할 조치를 고르라."),
        "en": (
            "Pick the action required by the governing rule.",
            "Choose the action the rules call for.",
        ),
    },
    "conflict": {
        "ko": (
            "같은 우선순위의 두 규칙이 서로 다른 조치를 지시하는가.",
            "우선순위가 같은데 조치가 충돌하는 규칙 쌍이 있는가.",
        ),
        "en": (
            "Do two rules of equal priority demand different actions?",
            "Is there a pair of equal-priority rules with conflicting actions?",
        ),
    },
    "enough": {
        "ko": (
            "주어진 상황 정보만으로 지배 규칙을 확정할 수 있는가.",
            "규칙을 고르기에 상황 정보가 충분한가.",
        ),
        "en": (
            "Is the given situation enough to settle which rule governs?",
            "Is the situation information sufficient to choose a rule?",
        ),
    },
    "applies": {
        "ko": ("규칙 {entity}가 이 상황에 적용되는가.", "{entity}의 조건이 이 상황에 들어맞는가."),
        "en": ("Does rule {entity} apply to this situation?", "Do the conditions of {entity} hold?"),
    },
    "severity": {
        "ko": ("지배 규칙이 정한 심각도 수준을 고르라.", "이 상황의 심각도 수준을 고르라."),
        "en": (
            "Choose the severity level set by the governing rule.",
            "Pick the severity level of this situation.",
        ),
    },
})


def _rules_rule(scene: dict, rule_id: str) -> dict:
    return next(rule for rule in scene["rules"] if rule["id"] == rule_id)


def _rules_match(scene: dict, rule: dict) -> bool | None:
    """규칙이 상황에 걸리는가. 상황에 없는 속성을 보면 `None`(판단 불가)."""
    situation = scene["situation"]
    for key, value in rule["when"].items():
        if key not in situation:
            return None
        if situation[key] != value:
            return False
    unless = rule.get("unless")
    if unless:
        undetermined = False
        hit = True
        for key, value in unless.items():
            if key not in situation:
                undetermined = True
            elif situation[key] != value:
                hit = False
        if hit and undetermined:
            return None
        if hit:
            return False
    return True


def _rules_outcome(scene: dict) -> dict:
    """지배 규칙·충돌·정보 충분 여부를 한 번에 낸다."""
    matched = []
    undetermined = []
    for rule in scene["rules"]:
        state = _rules_match(scene, rule)
        if state is True:
            matched.append(rule)
        elif state is None:
            undetermined.append(rule)
    best = min((rule["priority"] for rule in matched), default=None)
    governing = [rule["id"] for rule in matched if rule["priority"] == best]
    risky = [
        rule["id"]
        for rule in undetermined
        if best is None or rule["priority"] <= best
    ]
    conflict = any(
        one["priority"] == other["priority"] and one["action"] != other["action"]
        for index, one in enumerate(matched)
        for other in matched[index + 1 :]
    )
    actions = sorted({rule["action"] for rule in matched if rule["priority"] == best})
    return {
        "matched": [rule["id"] for rule in matched],
        "governing": governing,
        "undetermined": [rule["id"] for rule in undetermined],
        "risky": risky,
        "conflict": conflict,
        # 지배 규칙끼리 조치가 갈리면 명시 규칙만으로는 조치를 정할 수 없다.
        "governing_conflict": len(actions) > 1,
        "actions": actions,
        "enough": not risky,
    }


def _rules_describe(scene: dict, rule: dict, language: str, long: bool) -> str:
    conditions = ", ".join(
        f"{_RULE_ATTRS[key][language]}={_RULE_VALUES[value][language]}"
        for key, value in rule["when"].items()
    )
    head = f"{rule['id']} (우선순위 {rule['priority']})" if language == "ko" else (
        f"{rule['id']} (priority {rule['priority']})"
    )
    if not long:
        return head
    action = _RULE_ACTIONS[rule["action"]][language]
    if language == "ko":
        return f"{head} — {conditions}이면 {action}"
    return f"{head} - if {conditions} then {action}"


class RulesDomain:
    """명시 규칙의 우선순위·예외·정보 부족. 정답은 규칙 해석기가 낸다."""

    name = "rules"
    templates = ("access-policy", "access-policy-tie", "access-policy-partial")

    def make_scene(self, rng: random.Random, template: str) -> dict:
        attributes = {
            "zone": ("restricted", "open"),
            "window": ("night", "day"),
            "load": ("heavy", "light"),
            "approval": ("granted", "absent"),
        }
        situation = {key: rng.choice(values) for key, values in attributes.items()}
        keys = list(attributes)
        rules = []
        for index in range(rng.randint(4, 5)):
            key = keys[index % len(keys)]
            second = keys[(index + 1) % len(keys)]
            when = {key: rng.choice(attributes[key])}
            if index % 2:
                when[second] = rng.choice(attributes[second])
            rule = {
                "id": f"R{index + 1}",
                "priority": rng.randint(1, 3),
                "when": when,
                "action": rng.choices(list(_RULE_ACTIONS), weights=[_RULE_ACTION_WEIGHTS[a] for a in _RULE_ACTIONS], k=1)[0],
                "severity": rng.randint(0, 3),
            }
            if index % 3 == 2:
                # 예외: 조건에 걸려도 이 값이면 빠진다 (한 조건만 다른 near-miss의 원천).
                rule["unless"] = {"approval": "granted"}
            rules.append(rule)

        if template == "access-policy-tie":
            # 같은 우선순위에 다른 조치를 붙여 모순을 만든다.
            twin = dict(rules[0])
            twin["id"] = f"R{len(rules) + 1}"
            twin["action"] = next(
                action for action in _RULE_ACTIONS if action != rules[0]["action"]
            )
            twin["when"] = dict(rules[0]["when"])
            twin.pop("unless", None)
            rules.append(twin)
            for key, value in rules[0]["when"].items():
                situation[key] = value
        if template == "access-policy-partial":
            # 상황에서 한 속성을 지운다: 그 속성을 보는 규칙은 판단 불가가 된다.
            del situation[rng.choice(keys)]

        return {
            "domain": self.name,
            "template": template,
            "observed_at_ms": rng.randrange(600, 9950, 50),
            "rules": rules,
            "situation": situation,
            "drop_answer_at": rng.randrange(0, 6),
        }

    def state(self, scene: dict, rng: random.Random, language: str) -> dict:
        long = _long(rng)
        goal_text = (
            "명시된 규칙만으로 이 상황의 조치를 정한다"
            if language == "ko"
            else "Decide the action for this situation from the written rules alone"
        )
        return {
            "observed_at_ms": scene["observed_at_ms"],
            "goal": {"text": goal_text},
            # 속성·값은 언어와 무관한 토큰으로 두고 뜻은 용어 정의에 적는다. 그래야 번역본과
            # 원본의 **사실**이 같아지고 QA의 계보 검사가 두 언어를 가로질러 작동한다.
            "glossary": {
                "지배 규칙"
                if language == "ko"
                else "governing rule": (
                    "상황에 걸리는 규칙 중 우선순위 숫자가 가장 작은 규칙. 예외(unless)에 걸리면 빠진다."
                    if language == "ko"
                    else "the matching rule with the smallest priority number; an `unless` clause removes it"
                ),
                "심각도 수준"
                if language == "ko"
                else "severity level": (
                    "지배 규칙의 `severity` 값 그대로다. 지배 규칙이 여럿이면 각자의 값을 모두 허용한다."
                    if language == "ko"
                    else (
                        "the `severity` field of the governing rule itself; when several rules govern, "
                        "each of their values is accepted"
                    )
                ),
                **{key: text[language] for key, text in _RULE_ATTRS.items()},
                **{value: text[language] for value, text in _RULE_VALUES.items()},
            },
            "rules": [
                {
                    "id": rule["id"],
                    "desc": _rules_describe(scene, rule, language, long),
                    "priority": rule["priority"],
                    "when": dict(rule["when"]),
                    **({"unless": dict(rule["unless"])} if rule.get("unless") else {}),
                    "action": rule["action"],
                    "severity": rule["severity"],
                }
                for rule in scene["rules"]
            ],
            "actions": [
                {"id": action_id, "desc": text[language]}
                for action_id, text in _RULE_ACTIONS.items()
            ],
            "situation": dict(scene["situation"]),
        }

    def pool(self, scene: dict) -> list[QuestionSpec]:
        rule_ids = [rule["id"] for rule in scene["rules"]]
        action_ids = list(_RULE_ACTIONS)
        specs: list[QuestionSpec] = []
        for index in range(6):
            subset = rule_ids if index % 2 else rule_ids[: max(2, len(rule_ids) - 1)]
            specs.append(
                QuestionSpec(
                    f"q_governing_{index}",
                    "choice",
                    "governing",
                    {
                        "candidates": list(subset),
                        "none": index % 3 != 2,
                        "drop_answer": index == scene["drop_answer_at"],
                    },
                )
            )
        for index in range(4):
            specs.append(
                QuestionSpec(
                    f"q_action_{index}",
                    "choice",
                    "action",
                    {"candidates": list(action_ids), "none": index % 2 == 0},
                )
            )
        for index in range(3):
            specs.append(QuestionSpec(f"q_conflict_{index}", "boolean", "conflict", {}))
            specs.append(QuestionSpec(f"q_enough_{index}", "boolean", "enough", {}))
        for index, rule_id in enumerate(rule_ids):
            specs.append(QuestionSpec(f"q_applies_{index}", "boolean", "applies", {"rule": rule_id}))
        for index in range(4):
            specs.append(QuestionSpec(f"q_severity_{index}", "ordinal", "severity", {}))
        return specs

    def render(
        self, scene: dict, spec: QuestionSpec, rng: random.Random, language: str
    ) -> Rendered:
        long = _long(rng)
        tags = ("long_candidates",) if long else ()
        tags += ("unfamiliar_term",)  # "지배 규칙"의 정의가 상태에 있다
        outcome = _rules_outcome(scene)
        if outcome["undetermined"]:
            tags += ("missing_info",)
        near_miss = [
            rule["id"]
            for rule in scene["rules"]
            if rule["id"] not in outcome["governing"] and _rules_match(scene, rule) is not True
        ]

        if spec.kind == "governing":
            entities = [
                (rule_id, _rules_describe(scene, _rules_rule(scene, rule_id), language, long))
                for rule_id in spec.params["candidates"]
            ]
            rule_id = "rules/priority-with-exceptions-v0"
            return _choice_question(
                spec,
                _say(rng, _RULES_TEXT["governing"], language),
                entities,
                answer=[] if outcome["risky"] else outcome["governing"],
                none_text=_say(rng, _NONE_TEXT, language) if spec.params["none"] else None,
                rule=rule_id,
                trace={"question_id": spec.id, "rule": rule_id, **outcome, "near_miss": near_miss},
                confidence="medium" if outcome["risky"] else "high",
                variants=tags + (("near_miss",) if near_miss else ()),
            )

        if spec.kind == "action":
            # 정보가 모자라거나 지배 규칙끼리 조치가 갈리면 명시 규칙만으로 조치를 정할 수
            # 없다 — "해당 없음·정보 부족"이 답이고, 그 후보가 없으면 마스킹한다.
            undecidable = outcome["risky"] or outcome["governing_conflict"]
            entities = [
                (action_id, _RULE_ACTIONS[action_id][language])
                for action_id in spec.params["candidates"]
            ]
            rule_id = "rules/required-action-v0"
            return _choice_question(
                spec,
                _say(rng, _RULES_TEXT["action"], language),
                entities,
                answer=[] if undecidable else outcome["actions"],
                none_text=_say(rng, _NONE_TEXT, language) if spec.params["none"] else None,
                rule=rule_id,
                trace={"question_id": spec.id, "rule": rule_id, **outcome},
                confidence="medium" if undecidable else "high",
                variants=tags,
            )

        if spec.kind == "conflict":
            rule_id = "rules/conflict-v0"
            return _boolean_rendered(
                spec,
                _say(rng, _RULES_TEXT["conflict"], language),
                _boolean_criteria(rng, language),
                True if outcome["conflict"] else (None if outcome["undetermined"] else False),
                rule=rule_id,
                trace={"question_id": spec.id, "rule": rule_id, **outcome},
                variants=tags,
                mask_reason="상황 정보가 모자라 걸리는 규칙을 확정할 수 없다",
            )

        if spec.kind == "enough":
            rule_id = "rules/sufficient-information-v0"
            return _boolean_rendered(
                spec,
                _say(rng, _RULES_TEXT["enough"], language),
                _boolean_criteria(rng, language),
                outcome["enough"],
                rule=rule_id,
                trace={"question_id": spec.id, "rule": rule_id, **outcome},
                variants=tags,
            )

        if spec.kind == "applies":
            rule = _rules_rule(scene, spec.params["rule"])
            rule_id = "rules/rule-applies-v0"
            return _boolean_rendered(
                spec,
                _say(
                    rng,
                    _RULES_TEXT["applies"],
                    language,
                    entity=_rules_describe(scene, rule, language, False),
                ),
                _boolean_criteria(rng, language),
                _rules_match(scene, rule),
                rule=rule_id,
                trace={
                    "question_id": spec.id,
                    "rule": rule_id,
                    "target": rule["id"],
                    "when": rule["when"],
                    "unless": rule.get("unless"),
                },
                variants=tags,
                mask_reason="규칙이 보는 속성이 상황에 없다",
            )

        severities = sorted(
            {str(_rules_rule(scene, rule_id)["severity"]) for rule_id in outcome["governing"]}
        )
        rule_id = "rules/severity-level-v0"
        return _ordinal_rendered(
            spec,
            _say(rng, _RULES_TEXT["severity"], language),
            list(_SEVERITY_LEVELS[language]),
            None if (outcome["risky"] or not severities) else severities,
            rule=rule_id,
            trace={"question_id": spec.id, "rule": rule_id, **outcome, "severities": severities},
            # 지배 규칙이 여럿이면 각자의 심각도를 모두 허용하되 근거는 약하다.
            confidence="medium" if len(severities) > 1 else "high",
            variants=tags,
            boundary=False,
            mask_reason="지배 규칙을 확정할 수 없어 심각도를 낼 수 없다",
        )


#: 분야 이름 → 생성기.
#: 분야별 개념 어휘 (docs/04 §5 개념 계열). 질문의 규칙 종류(`<분야>:<종류>`)와 분야별 특수 개념. `split.holdout_concepts`는
#: 이 어휘에서 고른다 — 그 개념을 다루는 질문이 하나라도 있는 계열은 통째로 OOD다.
CONCEPT_VOCABULARY: dict[str, list[str]] = {
    "spatial": [
        "spatial:target", "spatial:nearest", "spatial:in_zone", "spatial:goal_met", "spatial:distance", "spatial:crowding",
        # 지시의 목표 영역이 이 영역인 계열 (target·goal_met·in_zone 질문 중 목표 영역을 다루는 것)
        "spatial:goal-zone:zoneL", "spatial:goal-zone:zoneC", "spatial:goal-zone:zoneR",
    ],
    "dom": [
        "dom:action", "dom:blocker", "dom:reachable", "dom:needs_input", "dom:actionable", "dom:progress",
        # 숨김 해제 컨트롤: 목표 요소가 접힌 구역 안에 있어 드러내는 조작(`reveals`)이 먼저 필요한 계열
        "dom:reveal",
    ],
    "workflow": [
        "workflow:next", "workflow:owner", "workflow:precondition", "workflow:blocked", "workflow:priority", "workflow:load",
        # 자원 차단: 담당 자원이 오프라인인 계열의 착수·선행·담당·차단 질문 (용량 부족만으로는 아니다 — 흔해서 계열의 반이 걸린다)
        "workflow:resource_offline",
    ],
    "rules": [
        "rules:governing", "rules:action", "rules:conflict", "rules:enough", "rules:applies", "rules:severity",
        # 안내 요청 정책: 지배 규칙의 조치가 "안내 요청"(a_escort)뿐인 상황의 지배·조치·심각도 질문 (정책 유형 하나)
        "rules:escort-policy",
    ],
}


def concepts_for(domain: str, scene: dict, spec: QuestionSpec) -> tuple[str, ...]:
    """질문 하나가 다루는 개념 id (:data:`CONCEPT_VOCABULARY`). 규칙 종류는 언제나, 특수 개념은 그 장면·질문이 실제로 다룰 때."""
    found = [f"{domain}:{spec.kind}"]
    if domain == "spatial":
        zone = spec.params.get("zone")
        if zone is not None and zone == scene["goal"]["zone"] and spec.kind in ("target", "goal_met", "in_zone"):
            found.append(f"spatial:goal-zone:{zone}")
    elif domain == "dom" and spec.kind in ("action", "reachable", "blocker"):
        goal = _dom_element(scene, scene["goal"]["element"])
        reveal = _dom_reveal_control(scene, goal["parent"]) if goal.get("parent") else None
        if reveal is not None and _dom_visible(scene, goal) is False:
            found.append("dom:reveal")
    elif domain == "workflow" and spec.kind in ("blocked", "next", "precondition", "owner"):
        if any(resource["status"] != "available" for resource in scene["resources"]):
            found.append("workflow:resource_offline")
    elif domain == "rules" and spec.kind in ("governing", "action", "severity"):
        if _rules_outcome(scene)["actions"] == ["a_escort"]:
            found.append("rules:escort-policy")
    return tuple(dict.fromkeys(found))


DOMAINS: dict[str, Any] = {
    domain.name: domain
    for domain in (SpatialDomain(), DomDomain(), WorkflowDomain(), RulesDomain())
}
