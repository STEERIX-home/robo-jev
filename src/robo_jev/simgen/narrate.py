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


def compute_hint(obs: Observation, nominal_agent_radius: float | None = None, *, k_ref: float = 1.3, g: float = 9.81) -> str | None:
    """계획기가 계산해 둔 듯한 수치 하나(관측값과 명목 매개변수로). 장면에 맞는 것이 없으면 None. ``k_ref``·``g``는 속성 목록의
    값을 생성기가 넘긴다(리뷰 1 m9)."""
    r = obs.robot
    if obs.scene in ("p1_step_in", "p1p5_step_in_carrying", "p5_braking"):
        stop = r.speed * r.latency + r.speed**2 / (2.0 * r.decel_moderate)
        arm = obs.body == "arm"
        how = "at the normal pause deceleration" if arm else "with firm braking"
        text = f"Estimated stopping distance {how}, reaction included: {_len(obs, stop, 0.005 if arm else 0.01)}."
        if obs.scene == "p1p5_step_in_carrying" and obs.container is not None and obs.container.fill_mode == "percent":
            c = obs.container
            rise = k_ref * (c.diameter / 2.0) * r.decel_moderate / g
            free = max(c.height * (1.0 - c.fill.value), 0.0)
            when = "a normal pause" if arm else "firm braking"
            text += f" Nominal liquid rise under {when}: {_len(obs, rise, 0.001)} against about {_len(obs, free, 0.001)} of headroom."
        return text
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
        return (
            f"At {_poss(obs)} current pace {obs.agent.descriptor.ref} reaches the edge of your path in about {_time(obs, t_arr, 0.1)}; "
            f"you would clear the crossing line in about {_time(obs, float(t_clear), 0.1)}."
        )
    if obs.scene == "p1_head_on":
        free = obs.env.corridor_width.value - 2.0 * r.radius - obs.agent.size.value
        return f"Free width if you both keep to the walls: {_len(obs, max(free, 0.0))}."
    if obs.scene == "p5_turning":
        a_lat = r.speed**2 / obs.env.turn_radius
        return f"Sideways acceleration in this turn at your current speed: {textfmt.accel(a_lat)}."
    return None


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


def _capability_fact(obs: Observation, u: np.ndarray) -> _Fact:
    r = obs.robot
    lat = _time(obs, r.latency, 0.01)
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
    if obs.scene in ("p1_step_in", "p1p5_step_in_carrying"):
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
    if obs.body == "arm":
        text = f"In {approach} the path swings through a {angle}° arc of radius {radius} toward {env.place}."
    else:
        text = f"In {approach} you reach {env.place}: a {angle}° turn with a radius of {radius}."
    return _Fact(text, text, "PERCEIVED", f"turn ahead: {angle}° at radius {radius}, starts in {approach}")


# ----------------------------------------------------------------------------------------------------------------
# 문체
# ----------------------------------------------------------------------------------------------------------------


def _all_facts(obs: Observation, u: np.ndarray) -> list[_Fact]:
    facts = [_motion_fact(obs, u), _capability_fact(obs, u)]
    if obs.container is not None:
        facts.append(_container_fact(obs, u))
    for builder in (_corridor_fact, _turn_fact, _obstacle_fact):
        fact = builder(obs, u)
        if fact is not None:
            facts.append(fact)
    if obs.agent is not None:
        facts.extend(_agent_facts(obs, u))
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
