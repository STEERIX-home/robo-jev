"""문구 묶음(한국어): 질문 지시문·기준 문구·점수 기준·선택지의 꼴과 설명, 상태·규칙 글, 힌트 (Task v0b 지시서 §3).

문구 자체는 :mod:`~robo_jev.simgen.phrases_ko_data` (영어 :mod:`~robo_jev.simgen.phrases` 와 같은 열쇠·같은 묶음 크기 — 학습 묶음과 T2 전용
빼 둔 바꿔 쓰기 묶음), 이 모듈은 관측에서 자리 표시 값을 채운다. 관측만 받는다(세계 모듈을 import하지 않는다).

**자리 표시(한국어).** ``{이름}``·``{이름:조사}``(:func:`~robo_jev.simgen.korean.kfmt` — 조사는 채운 말의 받침에 맞춘다). 영어의 ``{action}``
(명령형)은 한국어에서 두 꼴이다: ``{act_if}``(조건 — '단단히 제동해 완전히 서면'), ``{act_noun}``(명사 — '… 서기'). 영어의 소유격·대명사
({poss}, {obj})는 쓰지 않는다. ``{entered}``는 끼어든 상대의 술어(사람은 걸어 들어왔다, 차량은 끼어들었다 …), ``{look}``/``{look_then}``
은 다시 보기의 명사·연결 꼴.
"""

from __future__ import annotations

from robo_jev.simgen import phrases_ko_data as data
from robo_jev.simgen import textfmt
from robo_jev.simgen.korean import has_batchim, kfmt
from robo_jev.simgen.observation import Observation
from robo_jev.simgen.phrases import _numbers, speed_text

__all__ = [
    "ANCHORS",
    "HINTS_KO",
    "KO",
    "MOTION_ONLY",
    "RULE_TEXT",
    "STATE_TEXT",
    "delay_reference",
    "ko_values",
    "look_forms",
    "look_phrase",
    "option_descriptions",
    "option_forms",
    "option_noun",
]

KO = data.KO
ANCHORS = data.ANCHORS_KO
#: 충돌이 있을 수 있는 순간에 움직임만 세는 물건 질문에 덧붙이는 정의(영어 phrases.MOTION_ONLY와 같은 뜻 — 수정 1 I2)
MOTION_ONLY = " (충돌은 빼고, 움직임 자체만으로 생기는 일만 셉니다.)"
STATE_TEXT = data.STATE_TEXT_KO
RULE_TEXT = data.RULE_TEXT_KO
HINTS_KO = data.HINTS_KO


# ----------------------------------------------------------------------------------------------------------------
# 선택지의 꼴(명사·조건)과 설명
# ----------------------------------------------------------------------------------------------------------------


def _short_table(canon: str, obs: Observation) -> dict:
    if obs.scene.startswith("p4") and canon in data.SHORT_P4_KO:     # 놓기·끼우기 확인의 '계속'은 그대로 두고 넘어가기(팔이어도)
        return data.SHORT_P4_KO[canon]
    if obs.body == "arm" and canon in data.SHORT_ARM_KO:
        return data.SHORT_ARM_KO[canon]
    if obs.scene == "p1_head_on" and canon in data.SHORT_HEAD_ON_KO:
        return data.SHORT_HEAD_ON_KO[canon]
    if obs.scene.startswith(("p6", "p1p6")) and canon in data.SHORT_P6_KO:
        return data.SHORT_P6_KO[canon]
    return data.SHORT_KO[canon]


def option_forms(canon: str, obs: Observation) -> dict[str, str]:
    """선택지의 질문용 꼴: {"act_if": 조건 마디(…면), "act_noun": 명사 꼴(…기)}."""
    if canon.startswith("turn_"):
        ratio = int(canon.split("_")[1]) / 100.0
        speed = speed_text(obs, ratio * obs.robot.speed)
        table = data.TURN_FULL_KO if ratio >= 1.0 else data.TURN_KO
        return {"act_if": kfmt(table["if"], speed=speed), "act_noun": kfmt(table["noun"], speed=speed)}
    table = _short_table(canon, obs)
    return {"act_if": table["if"], "act_noun": table["noun"]}


def option_noun(canon: str, obs: Observation) -> str:
    """계획표의 '가능한 동작'·권고 문장에 쓰는 명사 꼴."""
    return option_forms(canon, obs)["act_noun"]


def option_descriptions(canon: str, obs: Observation, pool: str = "train") -> list[str]:
    """선택지 설명의 바꿔 쓰기(학습 묶음은 셋 이상) — 영어 phrases.option_descriptions와 같은 고름. ``pool = "train_v01"``이면 v01 학습
    묶음(v0b 학습 묶음 + 새 문구, 같은 표)."""
    if pool == "train_v01":
        from robo_jev.simgen.phrasing import description_table, wide

        table, key = description_table(canon, obs.scene, obs.body)
        templates = wide("ko", table, key)
        if canon.startswith("turn_"):
            speed = speed_text(obs, int(canon.split("_")[1]) / 100.0 * obs.robot.speed)
            return [kfmt(t, speed=speed) for t in templates]
        numbers = _numbers(obs)
        return [kfmt(t, **numbers) for t in templates]
    if canon.startswith("turn_"):
        ratio = int(canon.split("_")[1]) / 100.0
        speed = speed_text(obs, ratio * obs.robot.speed)
        bank = data.DESCRIPTIONS_TURN_FULL_KO if ratio >= 1.0 else data.DESCRIPTIONS_TURN_KO
        return [kfmt(t, speed=speed) for t in bank[pool]]
    numbers = _numbers(obs)
    if obs.scene.startswith("p4") and canon in data.DESCRIPTIONS_P4_KO:
        bank = data.DESCRIPTIONS_P4_KO[canon]
    elif obs.body == "arm" and canon in data.DESCRIPTIONS_ARM_KO:
        bank = data.DESCRIPTIONS_ARM_KO[canon]
    elif obs.scene == "p1_head_on" and canon in data.DESCRIPTIONS_HEAD_ON_KO:
        bank = data.DESCRIPTIONS_HEAD_ON_KO[canon]
    elif obs.scene.startswith(("p6", "p1p6")) and canon in data.DESCRIPTIONS_P6_KO:
        bank = data.DESCRIPTIONS_P6_KO[canon]
    else:
        bank = data.DESCRIPTIONS_KO[canon]
    return [kfmt(t, **numbers) for t in bank[pool]]


# ----------------------------------------------------------------------------------------------------------------
# 다시 보기·지연 기준
# ----------------------------------------------------------------------------------------------------------------


def look_forms(obs: Observation) -> dict[str, str]:
    """'다시 볼까'의 행동: {"look": 명사 꼴(…기), "look_then": 연결 꼴(…고)}."""
    if obs.place is not None:
        return {"look": "다시 재 보기", "look_then": "다시 재 보고"}
    if obs.stack is not None:
        return {"look": "무게가 어떻게 실렸는지 확인하기", "look_then": "무게가 어떻게 실렸는지 확인하고"}
    if obs.corner is not None:
        return {"look": "모퉁이에서 멈춰 살펴보기", "look_then": "모퉁이에서 멈춰 살펴보고"}
    return {"look": "잠깐 멈춰 다시 보기", "look_then": "잠깐 멈춰 다시 보고"}


def look_phrase(obs: Observation) -> str:
    return look_forms(obs)["look"]


def delay_reference(obs: Observation) -> str:
    """지연 score의 기준(명사구)."""
    if obs.variant == "stopped":
        return "지금 바로 가는 것"
    if obs.load is not None and obs.env.maneuver == "start":
        return "빠르게 출발하는 것"
    return "지금 속도를 유지하는 것"


# ----------------------------------------------------------------------------------------------------------------
# 자리 표시 값
# ----------------------------------------------------------------------------------------------------------------


def _entered(obs: Observation) -> str:
    """끼어든 상대의 술어 — 몸·상대 갈래에 맞게(사람은 걸어 들어오고, 차량은 끼어들고, 공은 굴러 들어온다)."""
    if obs.body == "arm":
        return "막 들어왔습니다"
    group = obs.agent.group if obs.agent is not None else None
    return {"ball": "굴러 들어왔습니다", "vehicle": "막 끼어들었습니다", "robot": "막 들어왔습니다"}.get(group, "막 걸어 들어왔습니다")


def _or(word: str) -> str:
    """'A나/이나' — 받침에 맞춘 조사."""
    return word + ("이나" if has_batchim(word) else "나")


def ko_values(obs: Observation) -> dict:
    """한국어 문구의 자리 표시 값(상황 옷 문구는 관측에 이미 한국어로 들어 있다)."""
    a = obs.agent
    arm = obs.body == "arm"
    ref = a.descriptor.ref if a is not None else "장애물"
    values = {
        "ref": ref,
        "cref": (obs.container.ref or "용기") if obs.container is not None else "용기",
        "obstacle": obs.env.obstacle or "장애물",
        "lane": "공구 경로" if arm else "당신의 주행로",
        "entered": _entered(obs),
    }
    values["hazard"] = values["obstacle"] if obs.env.obstacle is not None else (f"{_or(ref)} 옆의 무언가" if a is not None else "무언가")
    if obs.load is not None:
        values.update(lref=obs.load.ref, deck=obs.load.deck_phrase)
    if obs.stack is not None:
        values.update(sref=obs.stack.ref, spot=obs.stack.spot)
    if obs.place is not None:
        values.update(item=obs.place.item, target=obs.place.target)
    if obs.env.ramp is not None:
        values["ramp"] = obs.env.ramp
    if obs.stale is not None:
        values.update(ref=obs.stale.ref, spot=obs.stale.spot)
    if obs.corner is not None:
        values["corner"] = obs.corner.phrase
    if obs.conflict is not None:
        values["dist"] = textfmt.distance(obs.conflict.distance.value, obs.conflict.distance.resolution, obs.fmt.distance_unit)
    if obs.env.occluder is not None:
        values["occluder"] = obs.env.occluder
    return values
