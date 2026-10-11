"""관측 서술기(한국어): 학습 문체 셋 ``facts``·``prose``·``planner``와 빼 둔 문체 ``log``(T2 전용) (Task v0b 지시서 §3).

영어 서술기(:mod:`~robo_jev.simgen.narrate`)와 **같은 사실을 같은 순서·같은 수로** 쓴다 — 사실 조각마다 영어 판을 그대로 옮긴 한국어 판이
있다. 단위는 영어와 같은 규칙(기호 m·m/s·s …, 관측의 표기 선택)이고, 조사는 :func:`~robo_jev.simgen.korean.kfmt` 가 채운 말의 받침에
맞춘다. 관측만 받는다(세계 모듈을 import하지 않는다) — 상황 옷 문구는 관측에 이미 한국어로 들어 있다(scenes._skin_ko).
"""

from __future__ import annotations

import numpy as np

from robo_jev.simgen import textfmt
from robo_jev.simgen.korean import has_batchim, kfmt
from robo_jev.simgen.narrate import HEDGED_CUES, RenderPlan, _Fact, _len, _time, ordered, render_log
from robo_jev.simgen.observation import Observation
from robo_jev.simgen.phrases import speed_text
from robo_jev.simgen.spec import fill_bin_index

__all__ = ["FILL_KO", "render_state_ko"]

#: 채움 구간(FILL_BINS 순서)의 한국어 — (서술 술어, 짧은 꼴)
FILL_KO = (
    ("3분의 1이 안 되게 차 있습니다", "3분의 1 미만"),
    ("3분의 1쯤 차 있습니다", "약 3분의 1"),
    ("절반쯤 차 있습니다", "약 절반"),
    ("3분의 2쯤 차 있습니다", "약 3분의 2"),
    ("4분의 3쯤 차 있습니다", "약 4분의 3"),
    ("거의 가득 차 있습니다", "거의 가득"),
    ("가장자리 가까이까지 차 있습니다", "가장자리 가까이"),
    ("가장자리까지 가득 차 있습니다", "가장자리까지 가득"),
)
_SIDE = {"left": "왼쪽", "right": "오른쪽"}


def _k(template: str, **values) -> str:
    return kfmt(template, **values)


def _ago(obs: Observation, age: float) -> str:
    return "방금" if age < 0.025 else f"{_time(obs, age)} 전"


def _at(obs: Observation, age: float) -> str:
    """'…에'(시각): 방금 / 1.20 s 전에."""
    return "방금" if age < 0.025 else f"{_time(obs, age)} 전에"


def _or_less(reading, text: str) -> str:
    return f"{text} 이하" if reading.censored else text


def _d(obs: Observation, reading) -> str:
    """거리 보고(바닥에 붙은 보고는 '이하')."""
    return _or_less(reading, textfmt.distance(reading.value, reading.resolution, obs.fmt.distance_unit))


def _sp(obs: Observation, reading) -> str:
    return _or_less(reading, textfmt.speed(reading.value, reading.resolution, obs.fmt.speed_unit))


def _side_of(obs: Observation, reading, axis: str) -> str:
    """옆 위치 보고(+ = 왼쪽) → '당신의 중심선에서 왼쪽으로 0.12 m' / '당신의 중심선 바로 위'."""
    if abs(reading.value) < reading.resolution / 2.0:
        return f"{axis} 바로 위"
    side = "왼쪽" if reading.value > 0 else "오른쪽"
    return f"{axis}에서 {side}으로 {textfmt.distance(abs(reading.value), reading.resolution, obs.fmt.distance_unit)}"


def _where_noun(where: str) -> str:
    """싣는 곳('끌고 가는 화물 플랫폼 위에') → 명사구('끌고 가는 화물 플랫폼 위')."""
    return where.removesuffix("에")


# ----------------------------------------------------------------------------------------------------------------
# 사실 조각(영어 narrate의 조각과 하나씩 대응)
# ----------------------------------------------------------------------------------------------------------------


def _identity(obs: Observation) -> str:
    return f"당신은 {obs.skin.body_name}입니다. 지금 {obs.skin.task} 중입니다."


def _motion_fact(obs: Observation, u: np.ndarray) -> _Fact:
    r = obs.robot
    place = obs.env.place
    special = _motion_fact_v0b(obs, u)
    if special is not None:
        return special
    if obs.variant == "stopped":
        cruise = speed_text(obs, r.cruise)
        a = textfmt.accel(r.accel)
        sentence = [
            _k("{place}에 멈춰 서서 건너기를 기다리고 있습니다. 출발하면 {a:으로} 가속해 {cruise}까지 올라갑니다.", place=place, a=a, cruise=cruise),
            _k("지금 {place}에 서 있습니다. 출발하면 {a:으로} 가속해 {cruise}까지 냅니다.", place=place, a=a, cruise=cruise),
        ][int(u[0] * 2)]
        return _Fact(f"{place}에 멈춰 있음; 출발 가속 {a}, 순항 속도 {cruise}.", sentence, "ROBOT", f"상태: {place}에 멈춤; 가속 {a} → 순항 {cruise}")
    v = speed_text(obs, r.speed)
    if obs.body == "arm":
        sentence = [_k("공구가 {place} 위에서 경로를 따라 {v:으로} 움직이고 있습니다.", place=place, v=v),
                    _k("말단 장치가 {place} 위를 {v:으로} 지나가고 있습니다.", place=place, v=v)][int(u[0] * 2)]  # fmt: skip
        return _Fact(f"공구 속도: {v}, {place} 위.", sentence, "ROBOT", f"공구 속도: {v} ({place} 위)")
    if obs.scene == "p1_head_on":
        sentence = [_k("{place:을} 따라 {v:으로} 달리고 있습니다.", place=place, v=v), _k("{place}에서 {v:으로} 움직이고 있습니다.", place=place, v=v)][int(u[0] * 2)]
        return _Fact(f"{place}에서 {v}로 이동 중.", sentence, "ROBOT", f"속도: {v} ({place})")
    if obs.scene == "p5_turning":
        sentence = [_k("지금 {v:으로} 곧게 달리고 있습니다.", v=v), _k("{place} 쪽으로 {v:으로} 가고 있습니다.", place=place, v=v)][int(u[0] * 2)]
        return _Fact(f"지금 속도: {v}.", sentence, "ROBOT", f"속도: {v}")
    if obs.scene == "p1_crossing":
        sentence = [_k("{v:으로} 움직이며 {place}에 다가가고 있습니다.", v=v, place=place), _k("{v:으로} 달려 {place} 가까이 왔습니다.", v=v, place=place)][int(u[0] * 2)]
        return _Fact(f"{v}로 이동, {place}에 다가가는 중.", sentence, "ROBOT", f"속도: {v} ({place}에 접근)")
    sentence = [_k("{place}에서 {v:으로} 움직이고 있습니다.", place=place, v=v), _k("{v:으로} {place}에서 이동하고 있습니다.", place=place, v=v)][int(u[0] * 2)]
    return _Fact(f"{place}에서 {v}로 이동 중.", sentence, "ROBOT", f"속도: {v} ({place})")


def _motion_fact_v0b(obs: Observation, u: np.ndarray) -> _Fact | None:
    r, env = obs.robot, obs.env
    place = env.place
    if env.maneuver == "start":
        cruise = speed_text(obs, r.cruise)
        text = f"{place}에 멈춰 서 있고, 곧 출발해 {cruise}까지 가속합니다."
        return _Fact(f"{place}에 멈춰 있음; 곧 {cruise}까지 출발.", text, "ROBOT", f"상태: {place}에 멈춤; 다음: {cruise}까지 출발")
    if obs.scene == "p3_ramp":
        if env.ramp_direction == "down":
            v = speed_text(obs, r.speed)
            text = _k("{v:으로} {ramp}에 다가가고 있습니다. 다 내려가면 {ramp_end}에서 서야 합니다.", v=v, ramp=env.ramp, ramp_end=env.ramp_end)
            return _Fact(f"{v}로 {env.ramp}에 접근; 내려가서 {env.ramp_end}에서 멈춰야 함.", text, "ROBOT",
                         f"속도: {v}; 다음: {env.ramp} 내려감, {env.ramp_end}에서 정지")  # fmt: skip
        cruise = speed_text(obs, r.cruise)
        text = f"{env.ramp} 아래에 멈춰 있습니다. 이제 선 자리에서 {cruise}까지 가속하며 올라갑니다."
        return _Fact(f"{env.ramp} 아래에 멈춰 있음; 다음: {cruise}까지 올라감.", text, "ROBOT", f"상태: {env.ramp} 아래에 멈춤; 다음: {cruise}까지 올라감")
    if obs.stack is not None:
        st = obs.stack
        text = _k("{phrase:을} {spot} 위로 들고 있으며, 곧 내려놓으려 합니다.", phrase=st.phrase, spot=st.spot)
        return _Fact(f"{st.spot} 위로 {st.phrase} 들고 있음.", text, "ROBOT", f"들고 있음: {st.phrase} ({st.spot} 위)")
    if obs.place is not None:
        p = obs.place
        verb = "내려놓았습니다" if p.family == "place" else "밀어 넣었습니다"
        short = "내려놓음" if p.family == "place" else "밀어 넣음"
        text = _k("방금 {item:을} {target}에 {verb}.", item=p.item, target=p.target, verb=verb)
        return _Fact(_k("방금 {item:을} {target}에 {short}.", item=p.item, target=p.target, short=short), text, "ROBOT", f"직전 동작: {p.item} → {p.target} ({short})")
    return None


def _capability_fact(obs: Observation, u: np.ndarray) -> _Fact | None:
    r = obs.robot
    lat = _time(obs, r.latency, 0.01)
    if obs.env.maneuver == "start" or (obs.scene == "p3_ramp" and obs.env.ramp_direction == "up"):
        accels = obs.env.start_accels or {}
        if accels:
            g, n, q = (textfmt.accel(accels[k]) for k in ("start_gentle", "start_normal", "start_quick"))
            text = f"출발할 때 부드럽게({g}), 보통으로({n}), 빠르게({q}) 가속할 수 있습니다. 반응 지연은 {lat}입니다."
            return _Fact(f"출발 가속: 부드럽게 {g}, 보통 {n}, 빠르게 {q}; 반응 지연 {lat}.", text, "ROBOT", f"가속: 부드럽게 {g} | 보통 {n} | 빠르게 {q}; 지연 {lat}")
        a = textfmt.accel(r.accel)
        text = f"보통 {a}로 출발합니다. 반응 지연은 {lat}입니다."
        return _Fact(f"출발 가속 {a}; 반응 지연 {lat}.", text, "ROBOT", f"가속: {a}; 지연 {lat}")
    if obs.stack is not None or obs.place is not None:
        return None
    g, m, h = textfmt.accel(r.decel_gentle), textfmt.accel(r.decel_moderate), textfmt.accel(r.decel_hard)
    if obs.body == "arm":
        size = _len(obs, r.radius)
        bullet = f"감속: 부드럽게 {g}, 보통 {m}, 보호 정지 {h}; 반응 지연 {lat}; 공구 여유 반경 {size}."
        sentence = f"부드럽게 {g}, 보통 {m}, 보호 정지로 {h}까지 감속할 수 있습니다. 반응 지연은 {lat}이며 공구에는 {size}의 여유 반경이 필요합니다."
        return _Fact(bullet, sentence, "ROBOT", f"감속: 부드럽게 {g} | 보통 {m} | 보호 정지 {h}; 지연 {lat}; 여유 반경 {size}")
    size = _len(obs, r.radius)
    load = " (짐을 싣고 있어 낮춘 값)" if r.loaded else ""
    bullet = f"제동: 부드럽게 {g}, 단단히 {m}, 최대 {h}{load}; 반응 지연 {lat}; 차체 반경 {size}."
    sentence = [
        f"제동은 부드럽게 {g}, 단단히 {m}, 최대 {h}입니다{load}. 반응 지연은 {lat}, 차체 반경은 {size}입니다.",
        f"부드럽게({g}), 단단히({m}), 최대로({h}) 제동할 수 있습니다{load}. 반응 지연은 {lat}이고 차체 반경은 {size}입니다.",
    ][int(u[1] * 2)]
    return _Fact(bullet, sentence, "ROBOT", f"제동: 부드럽게 {g} | 단단히 {m} | 최대 {h}{load}; 지연 {lat}; 차체 반경 {size}")


def _fill_text(obs: Observation) -> tuple[str, str]:
    """(서술 술어, 짧은 꼴)."""
    c = obs.container
    if c.fill_mode == "phrase":
        return FILL_KO[fill_bin_index(c.fill.value)]
    pct = textfmt.percent(c.fill.value)
    return f"약 {pct}까지 차 있습니다", f"~{pct}"


def _lid_text(obs: Observation) -> tuple[str, str]:
    return {"open": ("뚜껑이 없습니다", "없음"), "lid": ("뚜껑이 덮여 있습니다", "있음"), "unknown": ("뚜껑이 있는지 보이지 않습니다", "알 수 없음(보이지 않음)")}[obs.container.lid]


def _container_fact(obs: Observation, u: np.ndarray) -> _Fact:
    c = obs.container
    (fill, fill_short), (lid, lid_short) = _fill_text(obs), _lid_text(obs)
    where = obs.skin.carry_where
    bullet = f"싣고 있는 것: {c.phrase}({_where_noun(where)}) — {fill_short}; 뚜껑 {lid_short}."
    sentence = [
        _k("{where} {phrase:이} 있습니다. {fill}. {lid}.", where=where, phrase=c.phrase, fill=fill, lid=lid),
        _k("{phrase:을} {where} 싣고 가고 있습니다. {fill}. 그리고 {lid}.", where=where, phrase=c.phrase, fill=fill, lid=lid),
    ][int(u[2] * 2)]
    return _Fact(bullet, sentence, "ROBOT", f"짐: {c.phrase} ({_where_noun(where)}); 채움 {fill_short}; 뚜껑 {lid_short}")


def _agent_facts(obs: Observation, u: np.ndarray) -> list[_Fact]:
    a = obs.agent
    ref, np_ = a.descriptor.ref, a.descriptor.np
    facts: list[_Fact] = []
    two_fix = a.speed is None
    if obs.scene in ("p1_step_in", "p1p5_step_in_carrying", "p1p6_emerging_step_in", "p1p3_step_in_carrying_tall"):
        verb, short = {"ball": ("당신의 주행로를 가로질러 굴러가고 있습니다", "주행로를 가로질러 굴러가는 중"),
                       "vehicle": ("당신의 주행로로 끼어들고 있습니다", "주행로로 끼어드는 중"),
                       "robot": ("당신의 주행로로 들어오고 있습니다", "주행로로 들어오는 중")}.get(a.group, ("당신의 주행로로 걸어 들어오고 있습니다", "주행로로 걸어 들어오는 중"))  # fmt: skip
        if obs.body == "arm":
            verb, short = "공구 경로 안으로 들어오고 있습니다", "공구 경로로 들어오는 중"
        gap = _d(obs, a.ahead)
        axis = "공구 경로" if obs.body == "arm" else "당신의 중심선"
        heading = _SIDE[a.heading]
        if two_fix:
            p0, p1 = _side_of(obs, a.track[0], axis), _side_of(obs, a.track[1], axis)
            motion = f"{_ago(obs, a.track_ages[0])} {p0}, {_ago(obs, a.track_ages[1])} {p1} ({heading}으로 이동)"
            sent = _k("{ref:은} {t0} {p0} 지점에, {t1} {p1} 지점에 있었고 당신의 {heading}으로 움직이고 있습니다.",
                      ref=ref, t0=_at(obs, a.track_ages[0]), p0=p0, t1=_at(obs, a.track_ages[1]), p1=p1, heading=heading)  # fmt: skip
            events = ((a.track_ages[0], f"{ref}: {p0}"), (a.track_ages[1], f"{ref}: {p1}, {heading}으로 이동"))
        else:
            pos = _side_of(obs, a.track[0], axis)
            speed = _sp(obs, a.speed)
            motion = f"{pos}, {heading}으로 {speed} (마지막 갱신 {_ago(obs, a.age)})"
            sent = _k("{t} {ref:은} {pos} 지점에 있었고, 당신의 {heading}으로 {speed:으로} 움직이고 있었습니다.",
                      t=_at(obs, a.age), ref=ref, pos=pos, heading=heading, speed=speed)  # fmt: skip
            events = ((a.age, f"{ref}: {pos}, {heading}으로 {speed}"),)
        first = _k("{np:이} {gap} 앞에서 {verb}.", np=np_, gap=gap, verb=verb)
        facts.append(_Fact(first, first, "PERCEIVED", f"{np_}: {gap} 앞", ((None, f"{np_}: {short}, 지금 간격 {gap}"),)))
        facts.append(_Fact(f"{ref}: {motion}.", sent, "PERCEIVED", f"  {motion}", events))
    elif obs.scene == "p1_crossing":
        side = _SIDE["left" if a.heading == "from_left" else "right"]
        line = _d(obs, a.ahead)
        verb = "굴러오고 있습니다" if a.group == "ball" else "다가오고 있습니다"
        bullet = _k("{np:이} {side}에서 당신의 진로로 {verb}. 그 이동선은 당신 앞면에서 {line} 앞에서 진로와 만납니다.", np=np_, side=side, verb=verb, line=line)
        sentence = _k("{np:이} {side}에서 당신의 진로 쪽으로 {verb}. 그 이동선이 당신 앞면에서 {line} 앞에서 진로를 가로지릅니다.", np=np_, side=side, verb=verb, line=line)
        facts.append(_Fact(bullet, sentence, "PERCEIVED", f"{np_}: {side}에서 접근; 가로지르는 선 {line} 앞"))
        if two_fix:
            d0, d1 = _d(obs, a.track[0]), _d(obs, a.track[1])
            motion = f"{_ago(obs, a.track_ages[0])} 진로에서 {d0}, {_ago(obs, a.track_ages[1])} {d1}"
            sent = _k("{ref:은} {t0} 당신 진로 가장자리에서 {d0}, {t1} {d1} 떨어져 있었습니다.",
                      ref=ref, t0=_at(obs, a.track_ages[0]), d0=d0, t1=_at(obs, a.track_ages[1]), d1=d1)  # fmt: skip
            events = ((a.track_ages[0], f"{ref}: 진로 가장자리에서 {d0}"), (a.track_ages[1], f"{ref}: 진로 가장자리에서 {d1}"))
        else:
            d1 = _d(obs, a.track[0])
            speed = _sp(obs, a.speed)
            motion = f"진로에서 {d1}, {speed}로 다가옴 (마지막 갱신 {_ago(obs, a.age)})"
            sent = _k("{t} {ref:은} 당신 진로 가장자리에서 {d1} 떨어져 있었고, {speed:으로} 다가오고 있었습니다.", t=_at(obs, a.age), ref=ref, d1=d1, speed=speed)
            events = ((a.age, f"{ref}: 진로 가장자리에서 {d1}, {speed}로 다가옴"),)
        facts.append(_Fact(f"{ref}: {motion}.", sent, "PERCEIVED", f"  {motion}", events))
    else:  # p1_head_on
        size = _d(obs, a.size)
        lateral = _side_of(obs, a.lateral, "통로 중심")
        bullet = _k("{np:이} 통로를 따라 당신 쪽으로 오고 있습니다. 통로 폭 가운데 약 {size}를 차지하며 {lateral} 지점에 있습니다.", np=np_, size=size, lateral=lateral)
        sentence = _k("{np:이} 통로를 따라 당신 쪽으로 오고 있습니다. {ref:은} 통로 폭에서 약 {size}를 차지하고 {lateral} 지점에 있습니다.",
                      np=np_, ref=ref, size=size, lateral=lateral)  # fmt: skip
        facts.append(_Fact(bullet, sentence, "PERCEIVED", f"{np_} (폭 {size}): 마주 오는 중, {lateral}"))
        if two_fix:
            d0, d1 = _d(obs, a.track[0]), _d(obs, a.track[1])
            motion = f"{_ago(obs, a.track_ages[0])} {d0} 앞, {_ago(obs, a.track_ages[1])} {d1} 앞"
            sent = _k("둘 사이 간격은 {t0} {d0}, {t1} {d1:으로} 보고되었습니다.", t0=_at(obs, a.track_ages[0]), d0=d0, t1=_at(obs, a.track_ages[1]), d1=d1)
            events = ((a.track_ages[0], f"{ref}까지 간격: {d0}"), (a.track_ages[1], f"{ref}까지 간격: {d1}"))
        else:
            d1 = _d(obs, a.track[0])
            speed = _sp(obs, a.speed)
            motion = f"{d1} 앞, {speed}로 다가옴 (마지막 갱신 {_ago(obs, a.age)})"
            sent = _k("{t} 둘 사이 간격은 {d1:으로} 보고되었고, {ref:은} {speed:으로} 당신 쪽으로 오고 있었습니다.", t=_at(obs, a.age), d1=d1, ref=ref, speed=speed)
            events = ((a.age, f"{ref}까지 간격: {d1}; {ref}: {speed}로 다가옴"),)
        facts.append(_Fact(f"{ref}: {motion}.", sent, "PERCEIVED", f"  {motion}", events))
    if a.cue_phrase:
        facts.append(_Fact(a.cue_phrase, a.cue_phrase, "PERCEIVED", f"  단서: {a.cue_phrase}"))
    return facts


def _sides_fact(obs: Observation, u: np.ndarray) -> _Fact | None:
    sides = obs.env.sides
    if sides is None:
        return None
    parts, lines = [], []
    for name, s in zip(("left", "right"), sides, strict=True):
        side = _SIDE[name]
        if s.status == "clear":
            parts.append(f"{side}은 비어 있습니다.")
            lines.append(f"{side}: 비어 있음")
        elif s.status == "blocked":
            parts.append(_k("{side} 바로 옆에 {phrase:이} 있습니다.", side=side, phrase=s.phrase))
            lines.append(f"{side}: 막힘 ({s.phrase})")
        else:
            parts.append(_k("{phrase:이} {side} 공간을 가리고 있습니다.", phrase=s.phrase, side=side))
            lines.append(f"{side}: 가려짐 ({s.phrase})")
    text = " ".join(parts)
    return _Fact(text, text, "PERCEIVED", "옆 공간 — " + "; ".join(lines))


def _corridor_fact(obs: Observation, u: np.ndarray) -> _Fact | None:
    if obs.env.corridor_width is None:
        return None
    w = _d(obs, obs.env.corridor_width)
    y = obs.env.robot_lateral
    where = "통로 한가운데에" if abs(y) < 0.005 else f"통로 중심에서 {'왼쪽' if y > 0 else '오른쪽'}으로 {_len(obs, abs(y))} 떨어진 곳에"
    text = f"통로 폭은 벽에서 벽까지 {w}입니다. 당신은 {where} 있습니다."
    return _Fact(text, text, "PERCEIVED", f"통로 폭: {w}; 당신의 위치: {where.removesuffix('에')}")


def _obstacle_fact(obs: Observation, u: np.ndarray) -> _Fact | None:
    if obs.env.obstacle is None:
        return None
    d = _d(obs, obs.env.obstacle_distance)
    o = obs.env.obstacle
    if obs.body == "arm":
        text = _k("공구 경로를 따라 {d} 앞에 {o:이} 있습니다. 동작은 그 앞에서 끝나야 합니다.", d=d, o=o)
    else:
        text = [_k("{o:이} {d} 앞에 있습니다. 거기 닿기 전에 서야 합니다.", o=o, d=d), _k("{d} 앞의 {o} 앞에서 멈춰야 합니다.", d=d, o=o)][int(u[3] * 2)]
    return _Fact(text, text, "PERCEIVED", f"멈출 곳: {o}, {d} 앞")


def _turn_fact(obs: Observation, u: np.ndarray) -> _Fact | None:
    env = obs.env
    if env.turn_radius is None:
        return None
    radius = _len(obs, env.turn_radius, 0.01 if obs.body == "arm" else 0.05)
    approach = _d(obs, env.turn_approach)
    angle = int(env.turn_angle_deg)
    side = f"{_SIDE[env.turn_direction]} " if env.turn_direction else ""
    if obs.body == "arm":
        text = f"{approach} 뒤에 경로가 반경 {radius}의 {angle}° {side}호를 그리며 {env.place} 쪽으로 돕니다."
    else:
        text = f"{approach} 뒤 {env.place}에 이릅니다. 반경 {radius}의 {angle}° {side}회전입니다."
    return _Fact(text, text, "PERCEIVED", f"앞의 회전: {angle}° {side}반경 {radius}, {approach} 뒤 시작")


def _mm(value: float, decimals: int) -> str:
    return f"{textfmt.number(value, decimals)} mm"


def _v0b_facts(obs: Observation, u: np.ndarray) -> list[_Fact]:
    env = obs.env
    out: list[_Fact] = []
    if obs.load is not None:
        ld = obs.load
        h, b = _d(obs, ld.height), _d(obs, ld.base)
        ef, eb, es = _d(obs, ld.edge_front), _d(obs, ld.edge_back), _d(obs, ld.edge_side)
        where = obs.skin.carry_where
        text = _k("{where} 있는 {deck} 위에 {phrase:이} 서 있습니다. 높이는 {h}, 바닥 폭은 {b}이며, 앞 가장자리에서 {ef}, 뒤 가장자리에서 {eb}, "
                  "옆 가장자리에서 {es} 떨어져 있습니다.", where=where, deck=ld.deck_phrase, phrase=ld.phrase, h=h, b=b, ef=ef, eb=eb, es=es)  # fmt: skip
        out.append(_Fact(f"짐: {ld.phrase}({_where_noun(where)}의 {ld.deck_phrase} 위) — 높이 {h}, 바닥 폭 {b}; 가장자리까지 앞 {ef}, 뒤 {eb}, 옆 {es}.", text, "ROBOT",
                         f"짐: {ld.phrase} ({ld.deck_phrase} 위); 높이 {h}, 바닥 폭 {b}; 가장자리까지 앞 {ef} | 뒤 {eb} | 옆 {es}"))  # fmt: skip
    if env.secure_s is not None and obs.load is not None and (env.maneuver == "start" or obs.scene == "p3_ramp"):
        t = _time(obs, env.secure_s, 0.5)
        text = _k("짐을 먼저 고정하려면(끈으로 묶거나 붙잡기) 약 {t:이} 걸립니다.", t=t)
        out.append(_Fact(text, text, "ROBOT", f"짐 고정: 약 {t}"))
    if obs.scene == "p3_ramp":
        length = _len(obs, env.ramp_length, 0.5)
        angle = textfmt.number(env.ramp_angle_deg, 1)
        text = _k("{ramp:은} {length}에 걸쳐 {angle}° 기울어 있습니다.", ramp=env.ramp, length=length, angle=angle)
        out.append(_Fact(text, text, "PERCEIVED", f"경사로: {angle}°, 길이 {length}"))
        t = _time(obs, env.detour_s, 1.0)
        text = _k("{detour:으로} 가면 경사로를 피할 수 있지만 약 {t:이} 더 걸립니다.", detour=env.detour, t=t)
        out.append(_Fact(text, text, "PERCEIVED", f"돌아가는 길: {env.detour}, 약 {t} 더 걸림"))
    if obs.stack is not None:
        st = obs.stack
        side = "왼쪽" if st.offset > 0 else "오른쪽"
        lo, top = _len(obs, st.length_low), _len(obs, st.length_top)
        off, sig = _len(obs, abs(st.offset), 0.005), _len(obs, st.sigma_place, 0.001)
        text = (f"{st.spot}의 길이는 {lo}, {st.ref}의 길이는 {top}입니다. {st.ref}의 중심을 아래 물건의 중심에서 {side}으로 {off} 떨어진 곳에 놓을 "
                f"계획이며, 놓기 정확도는 약 {sig}(표준편차 하나)입니다.")  # fmt: skip
        out.append(_Fact(f"아래 받침 {lo}; {st.ref} {top}; 계획한 중심 {side}으로 {off}; 놓기 정확도 ±{sig}.", text, "PERCEIVED",
                         f"쌓기: 받침 {lo}, 물건 {top}; 계획 어긋남 {side} {off}; 정확도 ±{sig}"))  # fmt: skip
        if st.cue is not None:
            text = _k("들어 올렸을 때 {ref:은} {cue} 끝 쪽이 더 무겁게 느껴졌습니다.", ref=st.ref, cue=_SIDE[st.cue])
            out.append(_Fact(text, text, "PERCEIVED", f"들 때 느낌: {_SIDE[st.cue]} 끝이 더 무거움"))
    if obs.place is not None:
        out.extend(_place_facts(obs))
    if obs.stale is not None:
        s = obs.stale
        ahead = _d(obs, s.ahead)
        age = _time(obs, s.age, 0.1)
        text = _k("{age} 전 마지막으로 보았을 때 {np:은} {spot}에 서 있었습니다(그 자리는 지금 {ahead} 앞). 그 뒤로 {occluder:이} 그 자리를 가려 "
                  "보이지 않습니다.", age=age, np=s.descriptor.np, spot=s.spot, ahead=ahead, occluder=s.occluder)  # fmt: skip
        out.append(_Fact(_k("{np}: {age} 전 {spot}에서 마지막으로 봄, 지금 {ahead} 앞; {occluder:이} 가림.", np=s.descriptor.np, age=age, spot=s.spot, ahead=ahead,
                            occluder=s.occluder), text, "PERCEIVED", f"{s.descriptor.np}: {age} 전 마지막으로 봄, {ahead} 앞; {s.occluder} 뒤로 가려짐",
                         ((s.age, f"{s.descriptor.np}: {s.spot}에 서 있음; 이후 {s.occluder} 뒤로 가려짐"), (None, f"{s.spot}: 지금 {ahead} 앞, 여전히 가려짐"))))  # fmt: skip
        sight = _len(obs, s.sight, 0.05)
        text = f"그 자리까지 약 {sight} 남았을 때에야 다시 보입니다."
        out.append(_Fact(text, text, "PERCEIVED", f"그 자리가 다시 보이는 거리: 약 {sight}"))
    if obs.corner is not None:
        c = obs.corner
        d, lat = _d(obs, c.ahead), _d(obs, c.lateral)
        side = _SIDE[c.side]
        text = _k("{d} 앞 {side}에 {phrase:이} 있어 그 너머가 보이지 않습니다. 그 뒤에서 누가 나오면 {lat} 떨어진 곳(가림 가장자리에서 당신의 진로까지)에서 "
                  "당신의 진로로 들어오게 됩니다.", d=d, side=side, phrase=c.phrase, lat=lat)  # fmt: skip
        out.append(_Fact(text, text, "PERCEIVED", f"보이지 않는 모퉁이: {c.phrase}, {side} {d} 앞; 가장자리에서 진로까지 {lat}"))
    if obs.conflict is not None:
        c = obs.conflict
        where = {"ahead": "정면", "left": "약간 왼쪽", "right": "약간 오른쪽"}[c.bearing]
        dist = _d(obs, c.distance)
        if c.lidar:
            text = f"라이다는 약 {dist} 앞 {where}에 물체가 있다고 보고하지만, 카메라에는 그곳에 아무것도 보이지 않습니다."
            line = f"라이다: 약 {dist} 앞에 물체 ({where}); 카메라: 없음"
        else:
            text = f"카메라는 약 {dist} 앞 {where}에 물체가 있다고 보고하지만, 라이다는 그곳에서 반사를 받지 못합니다."
            line = f"카메라: 약 {dist} 앞에 물체 ({where}); 라이다: 없음"
        out.append(_Fact(text, text, "PERCEIVED", line))
        if c.cue in HEDGED_CUES:
            text = _k("그 방향에 {phrase:이} 있을 수 있습니다.", phrase=c.phrase)      # 물체 자체일 수 있는 단서는 단정하지 않는다(수정 1 M2)
            out.append(_Fact(text, text, "PERCEIVED", f"그 방향에 있을 수 있음: {c.phrase}"))
        else:
            text = _k("그 방향에 {phrase:이} 있습니다.", phrase=c.phrase)
            out.append(_Fact(text, text, "PERCEIVED", f"그 방향: {c.phrase}"))
        if env.detour_s is not None:
            t = _time(obs, env.detour_s, 1.0)
            text = _k("그 자리를 돌아가면 약 {t:이} 더 걸립니다.", t=t)
            out.append(_Fact(text, text, "PERCEIVED", f"돌아가기: 약 {t} 더 걸림"))
    if env.occluder is not None and obs.agent is not None:
        text = _k("{ref:은} 방금 {occluder} 뒤에서 나왔고, {occluder:은} 그 뒤 공간을 가리고 있습니다.", ref=obs.agent.descriptor.ref, occluder=env.occluder)
        out.append(_Fact(text, text, "PERCEIVED", f"  {env.occluder} 뒤에서 나옴 (뒤 공간은 가려짐)"))
    if env.look_s is not None:
        t = _time(obs, env.look_s, 0.1)
        text = _k("멈춰서 다시 살피는 데 약 {t:이} 걸립니다.", t=t)
        out.append(_Fact(text, text, "ROBOT", f"다시 살피기: 약 {t}"))
    return out


_TILT_SIDE = {True: " 왼쪽으로", False: " 오른쪽으로"}


def _place_facts(obs: Observation) -> list[_Fact]:
    p = obs.place
    out: list[_Fact] = []
    readings = p.readings
    parts = []
    if "height_mm" in readings:
        r = readings["height_mm"]
        where = "높음" if r.value >= 0 else "낮음"
        parts.append(f"높이가 기준보다 {_mm(abs(r.value), 2 if r.resolution < 0.1 else 1)} {where}(게이지 정확도 ±{_mm(r.sigma, 2)})")
    if "depth_mm" in readings:
        r = readings["depth_mm"]
        where = "모자람" if r.value >= 0 else "더 들어감"
        parts.append(f"삽입 깊이가 끝까지 들어간 위치보다 {_mm(abs(r.value), 2 if r.resolution < 0.1 else 1)} {where}(정확도 ±{_mm(r.sigma, 2)})")
    if "tilt_deg" in readings:
        r = readings["tilt_deg"]
        side = "" if abs(r.value) < 0.05 else _TILT_SIDE[r.value > 0]
        parts.append(f"기울기 {textfmt.number(abs(r.value), 1)}°{side}(±{textfmt.number(r.sigma, 2)}°)")
    if "force_n" in readings:
        r = readings["force_n"]
        what = "놓을 때의 접촉력" if p.family == "place" else "최대 삽입력"
        parts.append(f"{what} {textfmt.number(r.value, 1)} N(±{textfmt.number(r.sigma, 2)} N)")
    if "lateral_n" in readings:
        r = readings["lateral_n"]
        side = "" if abs(r.value) < 0.05 else (" 왼쪽으로" if r.value > 0 else " 오른쪽으로")
        parts.append(f"옆 힘 {textfmt.number(abs(r.value), 1)} N{side}(±{textfmt.number(r.sigma, 2)} N)")
    text = "측정값: " + "; ".join(parts) + "."
    out.append(_Fact(text, text, "PERCEIVED", "측정값: " + "; ".join(parts)))
    tol = p.tolerance
    if p.family == "place":
        text = f"규격: 제대로 앉았다는 것은 기준 높이에서 ±{_mm(tol['height_mm'], 1)} 안이고 기울기가 {textfmt.number(tol['tilt_deg'], 1)}° 미만이라는 뜻입니다."
    else:
        text = f"규격: 끝까지 들어갔다는 것은 끝 깊이에서 {_mm(tol['depth_mm'], 1)} 안이라는 뜻입니다."
    out.append(_Fact(text, text, "PERCEIVED", text.removesuffix(".")))
    if p.ages:
        age = max(p.ages.values())
        names = "·".join({"height_mm": "높이", "tilt_deg": "기울기", "depth_mm": "깊이"}[n] for n in p.ages)
        t = _time(obs, age, 0.1)
        text = f"카메라 측정값({names})은 {t} 전에 잰 것입니다. {p.stale_reason} 힘 측정값은 지금 값입니다."
        out.append(_Fact(text, text, "PERCEIVED", f"카메라 측정값 {t} 전 것 ({p.stale_reason.removesuffix('.')}); 힘은 지금 값",
                         ((age, f"마지막 카메라 측정({names})"), (None, f"카메라 시야 잃음: {p.stale_reason.removesuffix('.')}; 힘 측정은 지금 값"))))  # fmt: skip
    return out


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


# ----------------------------------------------------------------------------------------------------------------
# 문체
# ----------------------------------------------------------------------------------------------------------------


def _noun(canon: str, obs: Observation) -> str:
    from robo_jev.simgen.phrases_ko import option_noun

    return option_noun(canon, obs)


def _recommendation_sentence(obs: Observation, canon: str, u: float) -> str:
    action = _noun(canon, obs)
    return [_k("모션 계획기는 {action:을} 권합니다.", action=action), f"계획기의 권고: {action}.", f"계획기 조언: {action}."][int(u * 3)]


def _render_facts(obs: Observation, plan: RenderPlan, u: np.ndarray, rng: np.random.Generator) -> str:
    lines = [f"{obs.skin.setting} {_identity(obs)}"]
    body = [f.bullet for f in ordered(_all_facts(obs, u), plan)]
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
    facts = ordered(_all_facts(obs, u), plan)
    robot = [f.sentence for f in facts if f.section == "ROBOT"]
    world = [f.sentence for f in facts if f.section == "PERCEIVED"]
    first = " ".join([obs.skin.setting, _identity(obs)] + robot)
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
    facts = ordered(_all_facts(obs, u), plan)
    out = [f"상황: {obs.skin.setting}", f"과제: {obs.skin.body_name} — 지금 {obs.skin.task} 중"]
    out.append("로봇 상태")
    out += [f"  {f.line}" for f in facts if f.section == "ROBOT"]
    out.append("인식")
    out += [f"  {f.line}" for f in facts if f.section == "PERCEIVED"]
    notes = list(obs.skin.distractors) + ([plan.hint] if plan.hint else [])
    if notes:
        order = rng.permutation(len(notes))
        out.append("메모")
        out += [f"  - {notes[int(i)]}" for i in order]
    if plan.recommendation:
        out.append(f"계획기 권고: {_noun(plan.recommendation, obs)}")
    out.append("가능한 동작")
    out += [f"  - {_noun(a, obs)}" for a in plan.actions]
    return "\n".join(out)


LOG_LABELS_KO = {"header": "사건 기록 — {setting}", "unit": "기체: {body} — 과제: {task} 중", "now": "지금", "robot": "로봇", "sensors": "센서",
                 "note": "메모", "calc": "계산", "planner": "계획기", "recommend": "권고: {action}", "actions": "가능한 동작: "}  # fmt: skip


def _render_log(obs: Observation, plan: RenderPlan, u: np.ndarray, rng: np.random.Generator) -> str:
    rec = LOG_LABELS_KO["recommend"].format(action=_noun(plan.recommendation, obs)) if plan.recommendation else None
    return render_log(obs, plan, rng, ordered(_all_facts(obs, u), plan), LOG_LABELS_KO, lambda a: _noun(a, obs), rec)


def render_state_ko(obs: Observation, plan: RenderPlan, rng: np.random.Generator) -> str:
    """한국어 상태 글 하나(영어 서술기와 같은 난수 소비 순서)."""
    u = rng.random(8)
    if plan.style == "facts":
        return _render_facts(obs, plan, u, rng)
    if plan.style == "prose":
        return _render_prose(obs, plan, u, rng)
    if plan.style == "planner":
        return _render_planner(obs, plan, u, rng)
    if plan.style == "log":
        return _render_log(obs, plan, u, rng)
    raise ValueError(plan.style)

