"""교란 시험 세트 T6의 변형 (Task v0b 지시서 §4 T6) — 관측 쪽 변환(세계 모듈을 import하지 않는다).

기준 레코드 하나마다 12가지: Microsoft-Decision-1식 8가지(상태 바꿔 쓰기, 지시 바꿔 쓰기, 선택지 설명 바꿔 쓰기, 순서 뒤집기, 섞기, 열쇠
바꾸기, 서식 잡음, 한 가지 더 정하기) + 물리 4가지(단위, 숫자 표기, 좌우 거울상, 사실 순서). 정답은 같아야 한다 — 좌우 거울상은 대응으로
(왼쪽으로 비켜서기 ↔ 오른쪽으로, 왼쪽 옆 공간 ↔ 오른쪽).

**거울상.** 관측의 옆 방향 양을 모두 뒤집는다(끼어듦의 옆 위치 보고·움직이는 쪽, 가로지름이 오는 쪽, 옆 공간 두 쪽, 회전 방향, 모퉁이 쪽,
센서 방위, 쌓기의 계획 어긋남과 쏠림 단서, 안착의 기울기·옆 힘). 사후 표본도 같은 난수로 거울상을 잡는다(:func:`mirror_batch` — 공통 난수:
정답이 대응으로 **정확히** 같고, 다르면 굴리기가 좌우 대칭이 아니라는 뜻이다). 마주 옴(p1_head_on)은 거울상 불변이 아니다 — 마주 오는
사람이 자기 오른쪽(로봇의 왼쪽)으로 비키는 사전이 더 크다(속성 목록 intent_prior) — 그래서 T6 기준에서 뺀다.
"""

from __future__ import annotations

import re
from dataclasses import replace

import numpy as np

from robo_jev.simgen.observation import Formatting, Observation
from robo_jev.simgen.spec import Batch

__all__ = [
    "MIRROR_SCENES_EXCLUDED",
    "VARIANTS",
    "alt_formatting",
    "format_noise",
    "mirror_batch",
    "mirror_canon",
    "mirror_observation",
    "mirror_recipe",
    "reorder_request",
    "strip_trailing_zeros",
]

VARIANTS = (
    "state_paraphrase", "instruction_paraphrase", "option_paraphrase", "order_reversal", "shuffle", "key_change", "format_noise",
    "one_more_decision", "units", "number_notation", "mirror", "fact_order",
)  # fmt: skip
#: 거울상이 정답을 바꾸는 장면(좌우가 다른 사전) — T6 기준에서 뺀다
MIRROR_SCENES_EXCLUDED = ("p1_head_on",)
_STEP_IN = ("p1_step_in", "p1p5_step_in_carrying", "p1p6_emerging_step_in", "p1p3_step_in_carrying_tall")
_LR = {"left": "right", "right": "left", "from_left": "from_right", "from_right": "from_left"}
_CANON = {"sidestep_left": "sidestep_right", "sidestep_right": "sidestep_left"}
_EVENTS = {"side_free_left": "side_free_right", "side_free_right": "side_free_left"}


def _neg(reading):
    return None if reading is None else replace(reading, value=-reading.value if reading.value != 0.0 else 0.0)


def mirror_observation(obs: Observation) -> Observation:
    """좌우 거울상의 관측(같은 상황을 왼쪽·오른쪽을 바꿔 본 것)."""
    if obs.scene in MIRROR_SCENES_EXCLUDED:
        raise ValueError(f"{obs.scene} is not mirror-symmetric (handedness prior)")
    agent = obs.agent
    if agent is not None:
        if obs.scene in _STEP_IN:
            agent = replace(agent, track=tuple(_neg(r) for r in agent.track), heading=_LR[agent.heading])
        elif obs.scene == "p1_crossing":
            agent = replace(agent, heading=_LR[agent.heading])
    env = obs.env
    if env.sides is not None:
        env = replace(env, sides=(env.sides[1], env.sides[0]))
    if env.turn_direction is not None:
        env = replace(env, turn_direction=_LR[env.turn_direction])
    out = replace(obs, agent=agent, env=env)
    if obs.corner is not None:
        out = replace(out, corner=replace(obs.corner, side=_LR[obs.corner.side]))
    if obs.conflict is not None and obs.conflict.bearing in _LR:
        out = replace(out, conflict=replace(obs.conflict, bearing=_LR[obs.conflict.bearing]))
    if obs.stack is not None:
        out = replace(out, stack=replace(obs.stack, offset=-obs.stack.offset, cue=None if obs.stack.cue is None else _LR[obs.stack.cue]))
    if obs.place is not None:
        readings = dict(obs.place.readings)
        for name in ("tilt_deg", "lateral_n"):
            if name in readings:
                readings[name] = _neg(readings[name])
        out = replace(out, place=replace(obs.place, readings=readings))
    return out


def mirror_batch(obs: Observation, batch: Batch) -> Batch:
    """사후 세계(기준 관측에서 뽑은 것)의 거울상 — 거울상 관측의 사후를 같은 난수로 뽑은 것과 같다(옆 방향 숨은 값의 부호, 좌우 칸 바꾸기)."""
    v = {key: np.array(value, dtype=float, copy=True) for key, value in batch.values.items()}
    if obs.scene in _STEP_IN:
        v["lat0"], v["dir"] = -v["lat0"], -v["dir"]
        v["side_left"], v["side_right"] = v["side_right"], v["side_left"]
    elif obs.scene == "p1_crossing":
        v["from_sign"] = -v["from_sign"]
    elif obs.scene == "p3_place_stack":
        v["c"], v["eps"] = -v["c"], -v["eps"]
    elif obs.place is not None:
        for key in ("state", "state_then"):
            s = v[key].astype(int)
            v[key] = np.where(s == 2, 3, np.where(s == 3, 2, s)).astype(float)   # 셋째 상태의 +/− 하위 상태
        for key in ("x_tilt_deg", "x_lateral_n"):
            if key in v:
                v[key] = -v[key]
    return Batch(v, weights=batch.weights)


def mirror_canon(option: str) -> str:
    return _CANON.get(option, option)


def mirror_recipe(recipe: dict) -> dict:
    """정답 조리법의 거울상(선택지·사건 이름) — 같은 것을 묻는 질문의 정체를 맞추려고."""
    out = dict(recipe)
    if isinstance(recipe.get("options"), dict) and recipe["kind"] in ("action",):
        out["options"] = {k: mirror_canon(v) for k, v in recipe["options"].items()}
    if "option" in recipe and recipe["option"] is not None:
        out["option"] = mirror_canon(recipe["option"])
    if "event" in recipe:
        out["event"] = _EVENTS.get(recipe["event"], recipe["event"])
    if "base" in recipe:
        out["base"] = sorted(mirror_canon(o) for o in recipe["base"])
    return out


# ----------------------------------------------------------------------------------------------------------------
# 글의 변형
# ----------------------------------------------------------------------------------------------------------------


def alt_formatting(obs: Observation) -> Formatting:
    """다른 단위 표기(정확히 바뀌는 것만): 거리 m ↔ cm(팔은 cm ↔ mm), 속도 m/s ↔ km/h(팔은 m/s ↔ mm/s), 시간 s ↔ ms."""
    f = obs.fmt
    if obs.body == "arm":
        distance = {"cm": "mm", "mm": "cm", "m": "cm"}[f.distance_unit]
        speed = {"m/s": "mm/s", "mm/s": "m/s"}[f.speed_unit]
    else:
        distance = {"m": "cm", "cm": "m"}[f.distance_unit]
        speed = {"m/s": "km/h", "km/h": "m/s"}[f.speed_unit]
    return Formatting(distance, speed, {"s": "ms", "ms": "s"}[f.time_unit])


_DECIMAL = re.compile(r"(?<![\d.])(\d+)\.(\d*?)0+(?![\d])")


def strip_trailing_zeros(text: str) -> str:
    """숫자 표기 바꾸기: 소수의 뒤 0을 뺀다('1.20 m' → '1.2 m', '2.0 s' → '2 s') — 값은 같다."""
    return _DECIMAL.sub(lambda m: m.group(1) + ("." + m.group(2) if m.group(2) else ""), text)


def format_noise(text: str, rng: np.random.Generator) -> str:
    """서식 잡음: 글머리표 모양, 들여쓰기, 문장 사이 공백, 빈 줄 — 낱말·수는 그대로."""
    bullet = ("* ", "• ", "– ")[int(rng.integers(3))]
    lines = []
    for line in text.split("\n"):
        if line.startswith("- "):
            line = bullet + line[2:]
        elif line.startswith("  - "):
            line = "    " + bullet + line[4:]
        elif line.startswith("  "):
            line = "\t" + line[2:]
        if rng.random() < 0.3:
            line = line.replace(". ", ".  ")
        lines.append(line.rstrip() + (" " if rng.random() < 0.2 else ""))
        if rng.random() < 0.15:
            lines.append("")
    return "\n".join(lines)


def reorder_request(request: dict, rng: np.random.Generator | None) -> dict:
    """질문 순서와 choice 선택지의 순서를 뒤집거나(rng = None) 섞는다 — 열쇠·글은 그대로(정답 불변)."""

    def order(n: int) -> list[int]:
        return list(range(n))[::-1] if rng is None else [int(i) for i in rng.permutation(n)]

    questions = request["questions"]
    names = list(questions)
    out = {}
    for i in order(len(names)):
        q = dict(questions[names[i]])
        if isinstance(q.get("criteria"), dict):
            keys = list(q["criteria"])
            q["criteria"] = {keys[j]: q["criteria"][keys[j]] for j in order(len(keys))}
        out[names[i]] = q
    return {**request, "questions": out}
