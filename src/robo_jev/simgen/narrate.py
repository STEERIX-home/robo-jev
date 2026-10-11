"""관측 서술기 3종(영어): ``facts``(글머리 목록), ``prose``(문단), ``planner``(계획표) (docs/12 §6, Task v0a 지시서 §3.8).

**관측만 받는다** — 세계 모듈을 import하지 않는다. 같은 관측이면 숨은 값이 무엇이든 같은 글이 나온다(시험이 바이트 단위로 확인).

* 첫 1~2문장은 상황 옷: 누가·어디서·무슨 일(분야 카드).
* 수치는 관측의 값 그대로(:mod:`~robo_jev.simgen.textfmt`), 단위는 관측의 표기 선택.
* 거리는 면 사이 간격이다: 로봇 앞면 → 상대(장애물)의 가까운 면, 로봇 진로 가장자리 → 상대의 가까운 면. 마주 옴(head-on)의 위치
  보고는 **그 시각의** 둘 사이 간격이다(리뷰 1 C1 — 글이 정본).
* 바닥에 붙은 보고(``Reading.censored``)는 "… or less"로 쓴다(검열된 관측 — 사후가 그렇게 다룬다).
* 계획기 권고(:class:`RenderPlan` ``recommendation``)와 계산된 힌트(``hint``)는 생성기가 정해 넘긴다 — 서술기는 쓰기만 한다.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from robo_jev.simgen import textfmt
from robo_jev.simgen.observation import Observation
from robo_jev.simgen.phrases import cap, option_short, speed_text
from robo_jev.simgen.spec import FILL_BINS, fill_bin_index

__all__ = ["RenderPlan", "compute_hint", "render_state"]


@dataclass(frozen=True)
class RenderPlan:
    style: str
    actions: tuple[str, ...]              # 계획표의 '가능한 동작'(정식 선택지)
    recommendation: str | None = None     # 계획기 권고(정식 선택지)
    hint: str | None = None               # 계산된 힌트 문장


# ----------------------------------------------------------------------------------------------------------------
# 수치 조각
# ----------------------------------------------------------------------------------------------------------------


def _or_less(reading, text: str) -> str:
    return f"{text} or less" if reading.censored else text


def _dist(obs: Observation, reading) -> str:
    return _or_less(reading, textfmt.distance(reading.value, reading.resolution, obs.fmt.distance_unit))


def _len(obs: Observation, value: float, resolution: float = 0.01) -> str:
    unit = obs.fmt.distance_unit
    return textfmt.distance(value, resolution, unit)


def _time(obs: Observation, seconds: float, resolution: float = 0.05) -> str:
    return textfmt.time(seconds, resolution, obs.fmt.time_unit)


def _ago(obs: Observation, age: float) -> str:
    return "just now" if age < 0.025 else f"{_time(obs, age)} ago"


def _speed(obs: Observation, reading) -> str:
    return _or_less(reading, textfmt.speed(reading.value, reading.resolution, obs.fmt.speed_unit))


def _side_of(obs: Observation, reading, axis: str) -> str:
    """옆 위치 보고(+ = 왼쪽) → '0.12 m left of your path's centreline'."""
    if abs(reading.value) < reading.resolution / 2.0:
        return f"right on {axis}"
    side = "left" if reading.value > 0 else "right"
    return f"{textfmt.distance(abs(reading.value), reading.resolution, obs.fmt.distance_unit)} {side} of {axis}"


def _fill_text(obs: Observation) -> str:
    c = obs.container
    if c.fill_mode == "phrase":
        return FILL_BINS[fill_bin_index(c.fill.value)][2]
    return f"filled to about {textfmt.percent(c.fill.value)}"


def _lid_text(obs: Observation) -> str:
    return {"open": "it has no lid", "lid": "it has a lid on", "unknown": "you cannot see whether it has a lid"}[obs.container.lid]


def _poss(obs: Observation) -> str:
    """상대의 소유격: 사람(카트를 미는 사람 포함)은 their, 그 밖은 its."""
    return "their" if obs.agent is not None and obs.agent.group == "person" else "its"


def _carry_where(obs: Observation) -> str:
    """용기를 실은 곳 — 몸 이름 항목이 정한 말(견인차는 '끄는 짐칸 위' 등), 없으면 몸의 기본값(skins carry_default)."""
    return obs.skin.carry_where


# ----------------------------------------------------------------------------------------------------------------
# 힌트(관측에서 계산한 명목값 — 참값이 아니다)
# ----------------------------------------------------------------------------------------------------------------


#: 힌트 문장 틀(영어) — 자리 표시는 :func:`hint_parts` 가 채운 값(단위가 붙은 글).
HINTS_EN = {
    "stop": "Estimated stopping distance {how}, reaction included: {stop}.",
    "stop_rise": "Estimated stopping distance {how}, reaction included: {stop}. Nominal liquid rise under {when}: {rise} against about {free} of headroom.",
    "crossing": "At {poss} current pace {ref} reaches the edge of your path in about {t_arr}; you would clear the crossing line in about {t_clear}.",
    "head_on": "Free width if you both keep to the walls: {free}.",
    "turn": "Sideways acceleration in this turn at your current speed: {a_lat}.",
    "tip_start": "If the load's centre of mass sat at half its height, it would start to tip backward at an acceleration of about {a_tip}.",
    "ramp": "On this slope, {what} pushes the load like a level-floor acceleration of {a_eq}.",
    "stack": "If the weight of {sref} were centred, its centre of mass would sit {inside} inside the edge of the support.",
    "p4": "Computed from the readings: the measured {what} is {times} the tolerance.",
    "stale": "At your current speed you reach that spot in about {t_reach}; your stopping distance with maximum braking, reaction included, is {stop}.",
    "hard_stop": "Stopping distance with maximum braking at your current speed, reaction included: {stop}.",
}


def hint_parts(obs: Observation, nominal_agent_radius: float | None = None, *, k_ref: float = 1.3, g: float = 9.81, lang: str = "en") -> tuple[str, dict] | None:
    """계획기가 계산해 둔 듯한 수치(관측값과 명목 매개변수로 — 참값이 아니다): (틀 열쇠, 채울 값). 장면에 맞는 것이 없으면 None.
    ``k_ref``·``g``는 속성 목록의 값을 생성기가 넘긴다(리뷰 1 m9). ``lang``은 값 안의 말(영어/한국어)만 정한다."""
    r = obs.robot
    ko = lang == "ko"
    arm = obs.body == "arm"
    env = obs.env
    firm = obs.scene in ("p1_step_in", "p1p5_step_in_carrying", "p5_braking", "p1p6_emerging_step_in", "p1p3_step_in_carrying_tall")
    firm = firm or (obs.load is not None and env.maneuver == "brake")
    if firm:
        stop = r.speed * r.latency + r.speed**2 / (2.0 * r.decel_moderate)
        how = ("보통 정지 감속으로" if arm else "단단한 제동으로") if ko else ("at the normal pause deceleration" if arm else "with firm braking")
        values = {"how": how, "stop": _len(obs, stop, 0.005 if arm else 0.01)}
        if obs.scene == "p1p5_step_in_carrying" and obs.container is not None and obs.container.fill_mode == "percent":
            c = obs.container
            rise = k_ref * (c.diameter / 2.0) * r.decel_moderate / g
            free = max(c.height * (1.0 - c.fill.value), 0.0)
            when = ("보통 정지" if arm else "단단한 제동") if ko else ("a normal pause" if arm else "firm braking")
            return "stop_rise", {**values, "when": when, "rise": _len(obs, rise, 0.001), "free": _len(obs, free, 0.001)}
        return "stop", values
    if obs.scene == "p1_crossing":
        a = obs.agent
        speed = a.speed.value if a.speed is not None else max((a.track[0].value - a.track[1].value) / (a.track_ages[0] - a.track_ages[1]), 0.05)
        now = a.track[-1].value - speed * a.age
        t_arr = max(now, 0.0) / max(speed, 0.05)
        clear = a.ahead.value + 2.0 * r.radius + (nominal_agent_radius or 0.25)
        if obs.variant == "stopped":
            d_ramp = r.cruise**2 / (2.0 * r.accel)
            t_clear = r.latency + (np.sqrt(2.0 * clear / r.accel) if clear <= d_ramp else r.cruise / r.accel + (clear - d_ramp) / r.cruise)
        else:
            t_clear = clear / r.speed
        return "crossing", {"poss": _poss(obs), "ref": obs.agent.descriptor.ref, "t_arr": _time(obs, t_arr, 0.1), "t_clear": _time(obs, float(t_clear), 0.1)}
    if obs.scene == "p1_head_on":
        free = obs.env.corridor_width.value - 2.0 * r.radius - obs.agent.size.value
        return "head_on", {"free": _len(obs, max(free, 0.0))}
    if env.turn_radius is not None and obs.scene != "p1_crossing":
        return "turn", {"a_lat": textfmt.accel(r.speed**2 / env.turn_radius)}
    if obs.load is not None and env.maneuver == "start":
        ld = obs.load
        return "tip_start", {"a_tip": textfmt.accel(g * ld.base.value / max(ld.height.value, 1e-6))}
    if obs.scene == "p3_ramp":
        phi = math.radians(env.ramp_angle_deg)
        down = env.ramp_direction == "down"
        a = r.decel_moderate if down else r.accel
        what = (("경사로를 내려가다 보통으로 제동하면" if down else "경사로를 오르며 보통으로 출발하면") if ko
                else ("braking firmly on the way down" if down else "pulling away normally on the way up"))  # fmt: skip
        return "ramp", {"what": what, "a_eq": textfmt.accel((g * math.sin(phi) + a) / math.cos(phi))}
    if obs.stack is not None:
        st = obs.stack
        return "stack", {"sref": st.ref, "inside": _len(obs, max(st.length_low / 2.0 - abs(st.offset), 0.0), 0.005)}
    if obs.place is not None:
        name, reading = obs.place.primary()
        tol = obs.place.tolerance[name]
        what = {"height_mm": ("높이 차이", "height difference"), "depth_mm": ("남은 삽입 깊이", "remaining insertion depth")}[name][0 if ko else 1]
        times = f"{textfmt.number(abs(reading.value) / tol, 1)}" + ("배" if ko else "×")
        return "p4", {"what": what, "times": times}
    if obs.stale is not None:
        s = obs.stale
        stop = ph_stop(r.speed, r.latency, r.decel_hard)
        return "stale", {"t_reach": _time(obs, s.ahead.value / max(r.speed, 0.05), 0.1), "stop": _len(obs, stop)}
    if obs.corner is not None or obs.conflict is not None:
        return "hard_stop", {"stop": _len(obs, ph_stop(r.speed, r.latency, r.decel_hard))}
    return None


def ph_stop(v: float, latency: float, decel: float) -> float:
    """반응 + 제동 멈춤 거리."""
    return v * latency + v * v / (2.0 * decel)


def compute_hint(obs: Observation, nominal_agent_radius: float | None = None, *, k_ref: float = 1.3, g: float = 9.81, lang: str = "en") -> str | None:
    """힌트 문장(언어별 틀 + :func:`hint_parts`). 장면에 맞는 것이 없으면 None."""
    found = hint_parts(obs, nominal_agent_radius, k_ref=k_ref, g=g, lang=lang)
    if found is None:
        return None
    key, values = found
    if lang == "ko":
        from robo_jev.simgen.korean import kfmt
        from robo_jev.simgen.phrases_ko import HINTS_KO

        return kfmt(HINTS_KO[key], **values)
    return HINTS_EN[key].format(**values)


# ----------------------------------------------------------------------------------------------------------------
# 사실 조각: 각각 (글머리, 문장) 두 표현과 계획표 칸
# ----------------------------------------------------------------------------------------------------------------


@dataclass
class _Fact:
    bullet: str
    sentence: str
    section: str
    line: str


def _identity(obs: Observation, u: np.ndarray) -> str:
    skin = obs.skin
    return f"You are {skin.body_name}, {skin.task}."


def _motion_fact(obs: Observation, u: np.ndarray) -> _Fact:
    r = obs.robot
    place = obs.env.place
    special = _motion_fact_v0b(obs, u)
    if special is not None:
        return special
    if obs.variant == "stopped":
        cruise = speed_text(obs, r.cruise)
        sentence = [
            f"You are stopped {place}, waiting to cross; once you go you accelerate at {textfmt.accel(r.accel)} to {cruise}.",
            f"You are standing still {place}. Moving off, you accelerate at {textfmt.accel(r.accel)} up to {cruise}.",
        ][int(u[0] * 2)]
        return _Fact(f"Stopped {place}; start-up acceleration {textfmt.accel(r.accel)}, cruising speed {cruise}.", sentence, "ROBOT", f"state: stopped {place}; accel {textfmt.accel(r.accel)} → cruise {cruise}")
    v = speed_text(obs, r.speed)
    if obs.body == "arm":
        sentence = [f"The tool is moving along its path over {place} at {v}.", f"Your end effector is travelling at {v} above {place}."][int(u[0] * 2)]
        return _Fact(f"Tool speed: {v}, over {place}.", sentence, "ROBOT", f"tool speed: {v} over {place}")
    if obs.scene == "p1_head_on":
        sentence = [f"You are driving down {place} at {v}.", f"You are moving along {place} at {v}."][int(u[0] * 2)]
        return _Fact(f"Moving down {place} at {v}.", sentence, "ROBOT", f"speed: {v} in {place}")
    if obs.scene == "p5_turning":
        sentence = [f"You are travelling straight at {v}.", f"You are moving at {v} toward {place}."][int(u[0] * 2)]
        return _Fact(f"Current speed: {v}.", sentence, "ROBOT", f"speed: {v}")
    if obs.scene == "p1_crossing":
        sentence = [f"You are moving at {v}, approaching {place}.", f"You are travelling at {v} and coming up to {place}."][int(u[0] * 2)]
        return _Fact(f"Moving at {v}, approaching {place}.", sentence, "ROBOT", f"speed: {v} (approaching {place})")
    sentence = [f"You are moving {place} at {v}.", f"You are travelling at {v} {place}."][int(u[0] * 2)]
    return _Fact(f"Moving {place} at {v}.", sentence, "ROBOT", f"speed: {v} ({place})")


def _motion_fact_v0b(obs: Observation, u: np.ndarray) -> _Fact | None:
    """새 장면(Task v0b)의 로봇 움직임 문장 — 해당 없으면 None."""
    r, env = obs.robot, obs.env
    place = env.place
    if env.maneuver == "start":
        cruise = speed_text(obs, r.cruise)
        text = f"You are standing still {place} and about to set off, accelerating up to {cruise}."
        return _Fact(f"Standing still {place}; about to set off toward {cruise}.", text, "ROBOT", f"state: stopped {place}; next: pull away to {cruise}")
    if obs.scene == "p3_ramp":
        if env.ramp_direction == "down":
            v = speed_text(obs, r.speed)
            text = f"You are heading for {env.ramp} at {v}; at the bottom you have to stop at {env.ramp_end}."
            return _Fact(f"Heading for {env.ramp} at {v}; must stop at {env.ramp_end} at the bottom.", text, "ROBOT", f"speed: {v}; next: down {env.ramp}, stop at {env.ramp_end}")
        cruise = speed_text(obs, r.cruise)
        text = f"You are stopped at the foot of {env.ramp}; next you drive up it, accelerating from standstill to {cruise}."
        return _Fact(f"Stopped at the foot of {env.ramp}; next: drive up it to {cruise}.", text, "ROBOT", f"state: stopped at the foot of {env.ramp}; next: up to {cruise}")
    if obs.stack is not None:
        st = obs.stack
        text = f"You are holding {st.phrase} over {st.spot}, about to set it down."
        return _Fact(f"Holding {st.phrase} over {st.spot}.", text, "ROBOT", f"holding: {st.phrase} over {st.spot}")
    if obs.place is not None:
        p = obs.place
        verb = "set down" if p.family == "place" else "pushed"
        prep = "in" if p.family == "place" else "into"
        text = f"You have just {verb} {p.item} {prep} {p.target}."
        return _Fact(f"Just {verb} {p.item} {prep} {p.target}.", text, "ROBOT", f"last action: {verb} {p.item} {prep} {p.target}")
    return None


def _capability_fact(obs: Observation, u: np.ndarray) -> _Fact:
    r = obs.robot
    lat = _time(obs, r.latency, 0.01)
    if obs.env.maneuver == "start" or (obs.scene == "p3_ramp" and obs.env.ramp_direction == "up"):
        accels = obs.env.start_accels or {}
        if accels:
            g, n, q = (textfmt.accel(accels[k]) for k in ("start_gentle", "start_normal", "start_quick"))
            text = f"Pulling away you can accelerate gently ({g}), normally ({n}) or quickly ({q}); reaction latency {lat}."
            return _Fact(f"Acceleration: gentle {g}, normal {n}, quick {q}; reaction latency {lat}.", text, "ROBOT", f"accel: gentle {g} | normal {n} | quick {q}; latency {lat}")
        a = textfmt.accel(r.accel)
        text = f"You normally pull away at {a}; reaction latency {lat}."
        return _Fact(f"Pull-away acceleration {a}; reaction latency {lat}.", text, "ROBOT", f"accel: {a}; latency {lat}")
    if obs.stack is not None or obs.place is not None:
        return None
    g, m, h = textfmt.accel(r.decel_gentle), textfmt.accel(r.decel_moderate), textfmt.accel(r.decel_hard)
    if obs.body == "arm":
        size = _len(obs, r.radius)
        bullet = f"Deceleration: soft {g}, normal {m}, protective stop {h}; reaction latency {lat}; tool clearance radius {size}."
        sentence = f"You can decelerate softly at {g}, normally at {m}, or with a protective stop at {h}; your reaction latency is {lat} and the tool needs a clearance radius of {size}."
        return _Fact(bullet, sentence, "ROBOT", f"decel: soft {g} | normal {m} | protective {h}; latency {lat}; clearance radius {size}")
    size = _len(obs, r.radius)
    load = " (reduced because you are carrying a load)" if r.loaded else ""
    bullet = f"Braking: gentle {g}, firm {m}, maximum {h}{load}; reaction latency {lat}; footprint radius {size}."
    sentence = [
        f"Your brakes give {g} gentle, {m} firm or {h} at maximum{load}, after a reaction latency of {lat}; your footprint radius is {size}.",
        f"You can brake gently ({g}), firmly ({m}) or at maximum ({h}){load}. Reaction latency is {lat}, and your footprint radius is {size}.",
    ][int(u[1] * 2)]
    return _Fact(bullet, sentence, "ROBOT", f"braking: gentle {g} | firm {m} | max {h}{load}; latency {lat}; footprint radius {size}")


def _container_fact(obs: Observation, u: np.ndarray) -> _Fact:
    c = obs.container
    fill, lid, where = _fill_text(obs), _lid_text(obs), _carry_where(obs)
    bullet = f"Carrying {c.phrase} {where}: {fill}; {lid}."
    sentence = [
        f"{cap(c.phrase)} is {where}, {fill}; {lid}.",
        f"You are carrying {c.phrase} {where}. It is {fill}, and {lid}.",
    ][int(u[2] * 2)]
    lid_short = {"open": "none", "lid": "on", "unknown": "cannot tell (not visible)"}[c.lid]
    fill_short = fill if c.fill_mode == "phrase" else f"~{textfmt.percent(c.fill.value)}"
    return _Fact(bullet, sentence, "ROBOT", f"payload: {c.phrase} {where}; fill {fill_short}; lid {lid_short}")


def _agent_facts(obs: Observation, u: np.ndarray) -> list[_Fact]:
    a = obs.agent
    ref, np_ = a.descriptor.ref, a.descriptor.np
    facts: list[_Fact] = []
    two_fix = a.speed is None
    if obs.scene in ("p1_step_in", "p1p5_step_in_carrying", "p1p6_emerging_step_in", "p1p3_step_in_carrying_tall"):
        verb = {"ball": "is rolling across your lane", "vehicle": "is pulling into your lane", "robot": "is moving into your lane"}.get(a.group, "is stepping into your lane")
        if obs.body == "arm":
            verb = "is reaching into the tool's path"
        gap = _dist(obs, a.ahead)
        axis = "the tool path" if obs.body == "arm" else "your centreline"
        heading = f"toward your {a.heading}"
        if two_fix:
            p0, p1 = _side_of(obs, a.track[0], axis), _side_of(obs, a.track[1], axis)
            motion = f"{p0} {_ago(obs, a.track_ages[0])}, {p1} {_ago(obs, a.track_ages[1])} (moving {heading})"
            motion_sent = f"{cap(ref)} was {p0} {_ago(obs, a.track_ages[0])} and {p1} {_ago(obs, a.track_ages[1])}, heading {heading}."
        else:
            pos = _side_of(obs, a.track[0], axis)
            motion = f"{pos}, moving {heading} at {_speed(obs, a.speed)} (last update {_ago(obs, a.age)})"
            motion_sent = f"As of {_ago(obs, a.age)}, {ref} was {pos} and moving {heading} at {_speed(obs, a.speed)}."
        facts.append(_Fact(f"{cap(np_)} {verb}, {gap} ahead of you.", f"{cap(np_)} {verb}, {gap} ahead of you.", "PERCEIVED", f"{np_}: {gap} ahead"))
        facts.append(_Fact(f"{cap(ref)}: {motion}.", motion_sent, "PERCEIVED", f"  {motion}"))
    elif obs.scene == "p1_crossing":
        side = "left" if a.heading == "from_left" else "right"
        line = _dist(obs, a.ahead)
        verb = "is rolling toward" if a.group == "ball" else "is approaching"
        facts.append(_Fact(
            f"{cap(np_)} {verb} your path from the {side}; {_poss(obs)} line crosses your path {line} ahead of your front.",
            f"{cap(np_)} {verb} your path from the {side}, on a line that crosses your path {line} ahead of your front.",
            "PERCEIVED", f"{np_}: approaching from the {side}; crossing line {line} ahead",
        ))  # fmt: skip
        if two_fix:
            d0, d1 = _dist(obs, a.track[0]), _dist(obs, a.track[1])
            motion = f"{d0} from your path {_ago(obs, a.track_ages[0])}, {d1} from it {_ago(obs, a.track_ages[1])}"
            sent = f"{cap(_ago(obs, a.track_ages[0]))} {ref} was {d0} from the edge of your path; {_ago(obs, a.track_ages[1])} it was {d1}."
        else:
            d1 = _dist(obs, a.track[0])
            motion = f"{d1} from your path, closing at {_speed(obs, a.speed)} (last update {_ago(obs, a.age)})"
            sent = f"As of {_ago(obs, a.age)}, {ref} was {d1} from the edge of your path, moving toward it at {_speed(obs, a.speed)}."
        facts.append(_Fact(f"{cap(ref)}: {motion}.", sent, "PERCEIVED", f"  {motion}"))
    else:  # p1_head_on
        size = _dist(obs, a.size)
        lateral = _side_of(obs, a.lateral, "the aisle centre")
        facts.append(_Fact(
            f"{cap(np_)} is coming toward you down the aisle, taking up about {size} of its width, {lateral}.",
            f"{cap(np_)} is coming toward you down the aisle; {ref} takes up about {size} of its width and is {lateral}.",
            "PERCEIVED", f"{np_} (width {size}): oncoming, {lateral}",
        ))  # fmt: skip
        if two_fix:
            d0, d1 = _dist(obs, a.track[0]), _dist(obs, a.track[1])
            motion = f"{d0} ahead {_ago(obs, a.track_ages[0])}, {d1} ahead {_ago(obs, a.track_ages[1])}"
            sent = f"The gap between you was {d0} {_ago(obs, a.track_ages[0])} and {d1} {_ago(obs, a.track_ages[1])}."
        else:
            d1 = _dist(obs, a.track[0])
            motion = f"{d1} ahead, moving toward you at {_speed(obs, a.speed)} (last update {_ago(obs, a.age)})"
            sent = f"As of {_ago(obs, a.age)} the gap between you was {d1}, with {ref} moving toward you at {_speed(obs, a.speed)}."
        facts.append(_Fact(f"{cap(ref)}: {motion}.", sent, "PERCEIVED", f"  {motion}"))
    if a.cue_phrase:
        facts.append(_Fact(a.cue_phrase, a.cue_phrase, "PERCEIVED", f"  cue: {a.cue_phrase}"))
    return facts


def _sides_fact(obs: Observation, u: np.ndarray) -> _Fact | None:
    sides = obs.env.sides
    if sides is None:
        return None
    parts = []
    for name, s in zip(("left", "right"), sides, strict=True):
        if s.status == "clear":
            parts.append(f"the {name} side is clear")
        elif s.status == "blocked":
            parts.append(f"{s.phrase} is right beside you on the {name}")
        else:
            parts.append(f"{s.phrase} hides the space on your {name}")
    text = f"{cap(parts[0])}; {parts[1]}."
    line = "; ".join(f"{n}: {s.status}" + (f" ({s.phrase})" if s.phrase else "") for n, s in zip(("left", "right"), sides, strict=True))
    return _Fact(text, text, "PERCEIVED", f"sides — {line}")


def _corridor_fact(obs: Observation, u: np.ndarray) -> _Fact | None:
    if obs.env.corridor_width is None:
        return None
    w = _dist(obs, obs.env.corridor_width)
    y = obs.env.robot_lateral
    where = "in the middle of it" if abs(y) < 0.005 else f"{_len(obs, abs(y))} {'left' if y > 0 else 'right'} of its centre"
    text = f"The aisle is {w} wide wall to wall; you are {where}."
    return _Fact(text, text, "PERCEIVED", f"aisle width: {w}; your offset: {where}")


def _obstacle_fact(obs: Observation, u: np.ndarray) -> _Fact | None:
    if obs.env.obstacle is None:
        return None
    d = _dist(obs, obs.env.obstacle_distance)
    o = obs.env.obstacle
    if obs.body == "arm":
        text = f"{cap(o)} is {d} ahead of the tool along its path; the move has to end before it."
    else:
        text = [f"{cap(o)} is {d} in front of you; you have to stop before reaching it.", f"You need to come to a stop before {o}, which is {d} ahead."][int(u[3] * 2)]
    return _Fact(text, text, "PERCEIVED", f"stop target: {o}, {d} ahead")


def _turn_fact(obs: Observation, u: np.ndarray) -> _Fact | None:
    env = obs.env
    if env.turn_radius is None:
        return None
    radius = _len(obs, env.turn_radius, 0.01 if obs.body == "arm" else 0.05)
    approach = _dist(obs, env.turn_approach)
    angle = int(env.turn_angle_deg)
    side = f" to the {env.turn_direction}" if env.turn_direction else ""
    if obs.body == "arm":
        text = f"In {approach} the path swings through a {angle}° arc{side} of radius {radius} toward {env.place}."
    else:
        text = f"In {approach} you reach {env.place}: a {angle}° turn{side} with a radius of {radius}."
    return _Fact(text, text, "PERCEIVED", f"turn ahead: {angle}°{side} at radius {radius}, starts in {approach}")


def _mm(value: float, decimals: int) -> str:
    return f"{textfmt.number(value, decimals)} mm"


def _v0b_facts(obs: Observation, u: np.ndarray) -> list[_Fact]:
    """새 장면(Task v0b)의 사실: 키 큰 물건·경사로·쌓기·놓기 측정·오래된 추적·모퉁이·센서 불일치·가림 뒤에서 나옴."""
    env = obs.env
    out: list[_Fact] = []
    if obs.load is not None:
        ld = obs.load
        h, b = _dist(obs, ld.height), _dist(obs, ld.base)
        ef, eb, es = _dist(obs, ld.edge_front), _dist(obs, ld.edge_back), _dist(obs, ld.edge_side)
        where = obs.skin.carry_where
        text = (f"{cap(ld.phrase)} stands on {ld.deck_phrase} {where}. It is {h} tall with a base {b} wide, and sits {ef} from the front edge, "
                f"{eb} from the back edge and {es} from the side edge.")  # fmt: skip
        out.append(_Fact(f"Load: {ld.phrase} on {ld.deck_phrase} {where} — {h} tall, base {b}; edges: front {ef}, back {eb}, side {es}.", text, "ROBOT",
                         f"load: {ld.phrase} on {ld.deck_phrase}; height {h}, base {b}; to edges front {ef} | back {eb} | side {es}"))  # fmt: skip
    if env.secure_s is not None and obs.load is not None and (env.maneuver == "start" or obs.scene == "p3_ramp"):
        t = _time(obs, env.secure_s, 0.5)
        text = f"Securing the load first (strapping or holding it) would take about {t}."
        out.append(_Fact(text, text, "ROBOT", f"securing the load: about {t}"))
    if obs.scene == "p3_ramp":
        length = _len(obs, env.ramp_length, 0.5)
        text = f"{cap(env.ramp)} slopes at {textfmt.number(env.ramp_angle_deg, 1)}° over {length}."
        out.append(_Fact(text, text, "PERCEIVED", f"ramp: {textfmt.number(env.ramp_angle_deg, 1)}° over {length}"))
        t = _time(obs, env.detour_s, 1.0)
        text = f"{cap(env.detour)} would avoid the ramp but adds about {t}."
        out.append(_Fact(text, text, "PERCEIVED", f"detour: {env.detour}, about {t} longer"))
    if obs.stack is not None:
        st = obs.stack
        side = "left" if st.offset > 0 else "right"
        lo, top = _len(obs, st.length_low), _len(obs, st.length_top)
        off, sig = _len(obs, abs(st.offset), 0.005), _len(obs, st.sigma_place, 0.001)
        text = (f"The top of {st.spot} is {lo} long; {st.ref} is {top} long. You plan to set its centre {off} to the {side} of the centre of what is below, "
                f"and your placement is accurate to about {sig} (one standard deviation).")  # fmt: skip
        out.append(_Fact(f"Support below {lo} long; {st.ref} {top} long; planned centre {off} to the {side}; placement accuracy ±{sig}.", text, "PERCEIVED",
                         f"stack: support {lo}, item {top}; planned offset {off} {side}; accuracy ±{sig}"))  # fmt: skip
        if st.cue is not None:
            text = f"When you lifted it, {st.ref} felt heavier toward its {st.cue} end."
            out.append(_Fact(text, text, "PERCEIVED", f"lift feel: heavier toward the {st.cue} end"))
    if obs.place is not None:
        out.extend(_place_facts(obs))
    if obs.stale is not None:
        s = obs.stale
        ahead = _dist(obs, s.ahead)
        text = (f"{cap(s.descriptor.np)} was standing at {s.spot} — now {ahead} ahead of you — when you last saw {s.pronoun} {_time(obs, s.age, 0.1)} ago; "
                f"since then {s.occluder} has blocked your view of that spot.")  # fmt: skip
        out.append(_Fact(f"{cap(s.descriptor.np)}: last seen {_time(obs, s.age, 0.1)} ago at {s.spot}, now {ahead} ahead; view blocked by {s.occluder}.", text,
                         "PERCEIVED", f"{s.descriptor.np}: last seen {_time(obs, s.age, 0.1)} ago, {ahead} ahead; hidden by {s.occluder}"))  # fmt: skip
        sight = _len(obs, s.sight, 0.05)
        text = f"You will see that spot again only when you are about {sight} from it."
        out.append(_Fact(text, text, "PERCEIVED", f"view reopens at about {sight} from the spot"))
    if obs.corner is not None:
        c = obs.corner
        text = (f"{cap(c.phrase)} is {_dist(obs, c.ahead)} ahead on your {c.side}, and you cannot see past it. Anyone stepping out from behind it would come "
                f"into your path from {_dist(obs, c.lateral)} away (corner edge to your path).")  # fmt: skip
        out.append(_Fact(text, text, "PERCEIVED", f"blind corner: {c.phrase}, {_dist(obs, c.ahead)} ahead on the {c.side}; edge {_dist(obs, c.lateral)} from your path"))
    if obs.conflict is not None:
        c = obs.conflict
        where = {"ahead": "straight ahead", "left": "slightly to your left", "right": "slightly to your right"}[c.bearing]
        dist = _dist(obs, c.distance)
        if c.lidar:
            text = f"Your lidar reports an object about {dist} away, {where}; your camera sees nothing there."
            line = f"lidar: object at about {dist} ({where}); camera: nothing"
        else:
            text = f"Your camera reports an object about {dist} away, {where}; your lidar gets no return there."
            line = f"camera: object at about {dist} ({where}); lidar: nothing"
        out.append(_Fact(text, text, "PERCEIVED", line))
        text = f"{cap(c.phrase)} is in that direction."
        out.append(_Fact(text, text, "PERCEIVED", f"in that direction: {c.phrase}"))
        if env.detour_s is not None:
            t = _time(obs, env.detour_s, 1.0)
            text = f"Going around that spot would add about {t}."
            out.append(_Fact(text, text, "PERCEIVED", f"going around: about {t} longer"))
    if env.occluder is not None and obs.agent is not None:
        text = f"{cap(obs.agent.descriptor.ref)} has just come out from behind {env.occluder}, which hides the space behind {_poss_obj(obs)}."
        out.append(_Fact(text, text, "PERCEIVED", f"  came out from behind {env.occluder} (space behind hidden)"))
    if env.look_s is not None:
        t = _time(obs, env.look_s, 0.1)
        text = f"Pausing to look again would take about {t}."
        out.append(_Fact(text, text, "ROBOT", f"re-check: about {t}"))
    return out


def _poss_obj(obs: Observation) -> str:
    return "them" if obs.agent is not None and obs.agent.group == "person" else "it"


_TILT_SIDE = {True: "toward the left", False: "toward the right"}


def _place_facts(obs: Observation) -> list[_Fact]:
    p = obs.place
    out: list[_Fact] = []
    readings = p.readings
    parts, lines = [], []
    if "height_mm" in readings:
        r = readings["height_mm"]
        parts.append(f"height {_mm(r.value, 2 if r.resolution < 0.1 else 1)} above nominal (gauge accuracy ±{_mm(r.sigma, 2)})")
    if "depth_mm" in readings:
        r = readings["depth_mm"]
        parts.append(f"insertion depth {_mm(r.value, 2 if r.resolution < 0.1 else 1)} short of fully home (accuracy ±{_mm(r.sigma, 2)})")
    if "tilt_deg" in readings:
        r = readings["tilt_deg"]
        side = "" if abs(r.value) < 0.05 else " " + _TILT_SIDE[r.value > 0]
        parts.append(f"tilt {textfmt.number(abs(r.value), 1)}°{side} (±{textfmt.number(r.sigma, 2)}°)")
    if "force_n" in readings:
        r = readings["force_n"]
        what = "contact force at release" if p.family == "place" else "peak insertion force"
        parts.append(f"{what} {textfmt.number(r.value, 1)} N (±{textfmt.number(r.sigma, 2)} N)")
    if "lateral_n" in readings:
        r = readings["lateral_n"]
        side = "" if abs(r.value) < 0.05 else (" to the left" if r.value > 0 else " to the right")
        parts.append(f"side force {textfmt.number(abs(r.value), 1)} N{side} (±{textfmt.number(r.sigma, 2)} N)")
    text = "Measured: " + "; ".join(parts) + "."
    out.append(_Fact(text, text, "PERCEIVED", "measured: " + "; ".join(parts)))
    tol = p.tolerance
    if p.family == "place":
        text = f"Spec: properly seated means within ±{_mm(tol['height_mm'], 1)} of nominal height and under {textfmt.number(tol['tilt_deg'], 1)}° of tilt."
    else:
        text = f"Spec: fully home means within {_mm(tol['depth_mm'], 1)} of full depth."
    out.append(_Fact(text, text, "PERCEIVED", text[:1].lower() + text[1:]))
    if p.ages:
        age = max(p.ages.values())
        names = " and ".join({"height_mm": "height", "tilt_deg": "tilt", "depth_mm": "depth"}[n] for n in p.ages)
        text = f"The camera readings ({names}) are {_time(obs, age, 0.1)} old because {p.stale_reason}; the force reading is current."
        out.append(_Fact(text, text, "PERCEIVED", f"camera readings {_time(obs, age, 0.1)} old ({p.stale_reason}); force current"))
    return out


# ----------------------------------------------------------------------------------------------------------------
# 문체
# ----------------------------------------------------------------------------------------------------------------


def _all_facts(obs: Observation, u: np.ndarray) -> list[_Fact]:
    facts = [f for f in (_motion_fact(obs, u), _capability_fact(obs, u)) if f is not None]
    if obs.container is not None:
        facts.append(_container_fact(obs, u))
    for builder in (_corridor_fact, _turn_fact, _obstacle_fact):
        fact = builder(obs, u)
        if fact is not None:
            facts.append(fact)
    if obs.agent is not None:
        facts.extend(_agent_facts(obs, u))
    facts.extend(_v0b_facts(obs, u))
    side = _sides_fact(obs, u)
    if side is not None:
        facts.append(side)
    return facts


def _recommendation_sentence(obs: Observation, canon: str, u: float) -> str:
    action = option_short(canon, obs)
    return [f"Your motion planner suggests that you {action}.", f"The planner's recommendation: {action}.", f"Planner advice: {action}."][int(u * 3)]


def _render_facts(obs: Observation, plan: RenderPlan, u: np.ndarray, rng: np.random.Generator) -> str:
    lines = [f"{obs.skin.setting} {_identity(obs, u)}"]
    facts = _all_facts(obs, u)
    body = [f.bullet for f in facts]
    extra = list(obs.skin.distractors)
    if plan.hint:
        extra.append(plan.hint)
    order = rng.permutation(len(extra))
    body += [extra[int(i)] for i in order]
    if plan.recommendation:
        body.append(_recommendation_sentence(obs, plan.recommendation, u[4]))
    lines += [f"- {b}" for b in body]
    return "\n".join(lines)


def _render_prose(obs: Observation, plan: RenderPlan, u: np.ndarray, rng: np.random.Generator) -> str:
    facts = _all_facts(obs, u)
    robot = [f.sentence for f in facts if f.section == "ROBOT"]
    world = [f.sentence for f in facts if f.section == "PERCEIVED"]
    first = " ".join([obs.skin.setting, _identity(obs, u)] + robot)
    second = " ".join(world)
    tail = list(obs.skin.distractors)
    if plan.hint:
        tail.append(plan.hint)
    order = rng.permutation(len(tail))
    tail = [tail[int(i)] for i in order]
    if plan.recommendation:
        tail.append(_recommendation_sentence(obs, plan.recommendation, u[4]))
    paragraphs = [first, second] + ([" ".join(tail)] if tail else [])
    return "\n\n".join(p for p in paragraphs if p)


def _render_planner(obs: Observation, plan: RenderPlan, u: np.ndarray, rng: np.random.Generator) -> str:
    facts = _all_facts(obs, u)
    out = [f"CONTEXT: {obs.skin.setting}", f"TASK: you are {obs.skin.body_name}, {obs.skin.task}"]
    out.append("ROBOT STATE")
    out += [f"  {f.line}" for f in facts if f.section == "ROBOT"]
    out.append("PERCEIVED")
    out += [f"  {f.line}" for f in facts if f.section == "PERCEIVED"]
    notes = list(obs.skin.distractors) + ([plan.hint] if plan.hint else [])
    if notes:
        order = rng.permutation(len(notes))
        out.append("NOTES")
        out += [f"  - {notes[int(i)]}" for i in order]
    if plan.recommendation:
        out.append(f"PLANNER RECOMMENDATION: {option_short(plan.recommendation, obs)}")
    out.append("AVAILABLE ACTIONS")
    out += [f"  - {option_short(a, obs)}" for a in plan.actions]
    return "\n".join(out)


def render_state(obs: Observation, plan: RenderPlan, rng: np.random.Generator) -> str:
    """상태 글 하나."""
    u = rng.random(8)
    if plan.style == "facts":
        return _render_facts(obs, plan, u, rng)
    if plan.style == "prose":
        return _render_prose(obs, plan, u, rng)
    if plan.style == "planner":
        return _render_planner(obs, plan, u, rng)
    raise ValueError(plan.style)
