"""질문 생성기: 세 종류(choice·noul·score)와 흔들기 (docs/12 §7, Task v0a 지시서 §3.9).

**관측만 받는다** — 세계 모듈을 import하지 않는다. 질문마다 모델이 받는 부분(이름·종류·지시문·criteria)과 정답을 계산하는 조리법
(``recipe``, 모델 입력이 아니다)을 함께 만든다. 정답은 :mod:`~robo_jev.simgen.labels` 가 사후 굴림 결과와 조리법으로 계산한다.

요청당 3~8개: 행동 choice 1개(25 %는 하나 더), 핵심 noul 1개(장면의 결정 여유를 묻는 것 — 통계의 '핵심 noul'), noul 0~2개 더,
score 1~2개, 상태 choice(50 %), 규칙 적용 choice(35 %). 행동 choice의 약 20 %는 지시문에 기준을 적고 그 기준의 비용 덮어쓰기를
조리법에 넣는다. 선택지 열쇠 문체(단어 / 짧은 코드 / 동사형)는 레코드마다, 선택지 순서와 질문 순서는 섞는다.

뚜껑이 보이는 용기(약 5 %)에서는 쏟아짐이 늘 0이라 쏟아짐 질문 대신 '액면이 뚜껑까지 차오르나'(사건 rim_reached)를 묻고, avoid_spill
기준을 쓰지 않는다(리뷰 1 질문 7). 지연 score는 그 지연의 정의(장면마다)를 지시문에 적는다(리뷰 1 m3): P1은 '앞으로 3초 동안 잃는
진행', 감속(P5)은 '목표 앞에 일찍 서서 남은 거리를 다시 다가가는 시간', 회전은 '회전을 마치는 시각이 늦어지는 만큼'.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from robo_jev.simgen import textfmt
from robo_jev.simgen.catalog import INTENTS, SimgenConfig
from robo_jev.simgen.observation import Observation
from robo_jev.simgen.phrases import (
    ACTION_INSTRUCTIONS,
    ARM_VERB_KEYS,
    CRITERION_INSTRUCTIONS,
    VERB_KEYS,
    cap,
    container_ref,
    option_descriptions,
    option_short,
)
from robo_jev.simgen.spec import Reading

__all__ = ["KEY_STYLES", "Question", "QuestionSet", "write_questions"]

KEY_STYLES = ("word", "code", "verb")


@dataclass
class Question:
    name: str
    qtype: str                       # choice / noul / score
    instructions: str
    criteria: dict | list | None
    recipe: dict                     # 정답 조리법(모델 입력 아님)

    def request_entry(self) -> dict:
        entry: dict = {"type": self.qtype, "instructions": self.instructions}
        if self.criteria is not None:
            entry["criteria"] = self.criteria
        return entry


@dataclass
class QuestionSet:
    questions: list[Question]
    key_style: str
    primary_action: str              # 계획기 권고가 겨누는 행동 질문의 이름
    key_question: str                # 핵심 noul의 이름


_OPT_NAME = {
    "continue": "keep_speed", "slow": "slow", "stop_gentle": "gentle_stop", "stop_moderate": "firm_stop", "stop_hard": "hard_stop",
    "sidestep_left": "swerve_left", "sidestep_right": "swerve_right", "retract": "retract", "go_now": "go_now", "creep": "creep",
    "wait": "wait", "turn_100": "full_speed", "turn_80": "at_80pct", "turn_60": "at_60pct", "turn_45": "at_45pct", "turn_30": "at_30pct",
}  # fmt: skip


def _entered(agent, arm: bool) -> str:
    """끼어듦의 동사 — 몸·상대 갈래에 맞게(사람은 걸어 들어오고, 차량은 끼어들고, 공은 굴러 들어온다)."""
    if arm:
        return "has just reached into"
    group = agent.group if agent is not None else None
    return {"ball": "has rolled into", "vehicle": "has just pulled into", "robot": "has just moved into"}.get(group, "has just stepped into")


class _Ctx:
    """질문 문구에 채울 말과 난수."""

    def __init__(self, cfg: SimgenConfig, obs: Observation, rng: np.random.Generator):
        self.cfg, self.obs, self.rng = cfg, obs, rng
        a = obs.agent
        self.ref = a.descriptor.ref if a is not None else "the obstacle"
        self.cref = container_ref(obs.container.phrase) if obs.container is not None else "the container"
        self.obstacle = obs.env.obstacle or "the obstacle"
        self.names: set[str] = set()

    def fmt(self, text: str, **extra) -> str:
        ref_bare = self.ref[4:] if self.ref.startswith("the ") else self.ref
        agent = self.obs.agent
        arm = self.obs.body == "arm"
        values = {
            "ref": self.ref, "Ref": cap(self.ref), "cref": self.cref, "obstacle": self.obstacle, "ref_bare": ref_bare,
            "lane": "the tool's path" if arm else "your lane",
            "poss": "their" if agent is not None and agent.group == "person" else "its",
            "entered": _entered(agent, arm),
            **extra,
        }  # fmt: skip
        return text.format(**values)

    def pick(self, items):
        return items[int(self.rng.integers(len(items)))]

    def name(self, *candidates: str) -> str:
        base = self.pick(list(candidates))
        name, i = base, 2
        while name in self.names:
            name, i = f"{base}_{i}", i + 1
        self.names.add(name)
        return name


# ----------------------------------------------------------------------------------------------------------------
# 선택지 열쇠
# ----------------------------------------------------------------------------------------------------------------


def _option_keys(canons: list[str], style: str, prefix: str, obs: Observation) -> list[str]:
    if style == "code":
        return [f"{prefix}{i + 1}" for i in range(len(canons))]
    if style == "verb":
        table = {**VERB_KEYS, **(ARM_VERB_KEYS if obs.body == "arm" else {})}
        return [table[c] for c in canons]
    return list(canons)


def _choice(c: _Ctx, canons: list[str], descriptions: list[str], style: str, prefix: str) -> tuple[dict, dict]:
    """(criteria 열쇠 → 설명, 열쇠 → 정식 값) — 순서를 섞는다."""
    order = c.rng.permutation(len(canons))
    canons = [canons[int(i)] for i in order]
    descriptions = [descriptions[int(i)] for i in order]
    keys = _option_keys(canons, style, prefix, c.obs) if prefix == "a" or prefix == "b" else _state_keys(canons, style, prefix)
    return dict(zip(keys, descriptions, strict=True)), dict(zip(keys, canons, strict=True))


def _state_keys(canons: list[str], style: str, prefix: str) -> list[str]:
    if style == "code":
        return [f"{prefix}{i + 1}" for i in range(len(canons))]
    return list(canons)


# ----------------------------------------------------------------------------------------------------------------
# 행동 choice
# ----------------------------------------------------------------------------------------------------------------


def _lid_visible(obs: Observation) -> bool:
    return obs.container is not None and obs.container.lid == "lid"


def _criteria_for_scene(obs: Observation) -> list[str]:
    scene = obs.scene
    if scene == "p1p5_step_in_carrying":
        out = ["avoid_contact", "avoid_spill", "time_critical"]
    elif scene.startswith("p5"):
        out = ["avoid_spill", "time_critical"]
    else:
        out = ["avoid_contact", "time_critical"]
    return [c for c in out if not (c == "avoid_spill" and _lid_visible(obs))]


def _action_question(c: _Ctx, options: tuple[str, ...], style: str, prefix: str, criterion: str | None) -> Question:
    obs = c.obs
    canons = list(options)
    if len(canons) > 2 and c.rng.random() < 0.35:
        size = int(c.rng.integers(max(2, min(3, len(canons) - 1)), len(canons)))
        chosen = sorted(c.rng.choice(len(canons), size=size, replace=False))
        canons = [canons[int(i)] for i in chosen]
    descriptions = [c.pick(option_descriptions(canon, obs)) for canon in canons]
    criteria, keymap = _choice(c, canons, descriptions, style, prefix)
    if criterion is not None:
        instructions = c.fmt(c.pick(CRITERION_INSTRUCTIONS[criterion]))
        name = c.name("action_under_rule", "constrained_action", "action_with_rule")
    else:
        scene_key = "p1_crossing_stopped" if obs.scene == "p1_crossing" and obs.variant == "stopped" else obs.scene
        bank = ACTION_INSTRUCTIONS["generic"] + ACTION_INSTRUCTIONS[scene_key]
        instructions = c.fmt(c.pick(bank))
        name = c.name("next_action", "what_to_do", "action", "decision", "response")
    recipe = {"kind": "action", "options": keymap, "criterion": criterion}
    return Question(name, "choice", instructions, criteria, recipe)


# ----------------------------------------------------------------------------------------------------------------
# noul
# ----------------------------------------------------------------------------------------------------------------


def _noul(c: _Ctx, name: str, templates: list[str], recipe: dict, **extra) -> Question:
    return Question(name, "noul", c.fmt(c.pick(templates), **extra), None, {"kind": "noul", **recipe})


def _key_noul(c: _Ctx) -> Question:
    obs = c.obs
    if obs.scene in ("p1_step_in", "p1p5_step_in_carrying"):
        if obs.body == "arm":
            templates = [
                "If you pause at the normal deceleration now, does the tool stop before reaching {ref}?",
                "Is the tool's normal pause distance, reaction included, shorter than its gap to {ref}?",
                "Pausing at the standard rate from now, does the tool come to rest short of {ref}?",
            ]
        else:
            templates = [
                "If you brake firmly now, will you come to rest before reaching {ref}'s current position?",
                "Is your firm-braking stopping distance, reaction included, shorter than the gap to {ref}?",
                "Braking firmly from now on, do you stop short of where {ref} is now?",
            ]
        return _noul(c, c.name("stops_in_time", "firm_stop_short", "can_stop_short"), templates, {"source": "world", "event": "stops_short"})
    if obs.scene == "p1_crossing":
        if obs.variant == "stopped":
            templates = [
                "If you start now and {ref} keeps {poss} current pace, will you be across {ref}'s line at least 0.3 s before {ref} reaches your path?",
                "Going right away, do you clear the crossing with at least 0.3 s to spare, assuming {ref} does not change pace?",
                "If you move off now, would you be past the crossing line at least 0.3 s ahead of {ref} (both at constant pace)?",
            ]
        else:
            templates = [
                "If you keep your speed and {ref} keeps {poss} current pace, will you be clear of {ref}'s line at least 0.3 s before {ref} reaches your path?",
                "Keeping your speed, do you get across {ref}'s line with at least 0.3 s to spare, assuming {ref} does not change pace?",
                "At constant speeds on both sides, would you clear the crossing at least 0.3 s ahead of {ref}?",
            ]
        return _noul(c, c.name("clear_before_them", "crossing_in_time", "clears_first"), templates, {"source": "world", "event": "clears_first"})
    if obs.scene == "p1_head_on":
        clearance = textfmt.distance(obs.robot.passing_clearance, 0.01, "cm" if obs.fmt.distance_unit == "cm" else obs.fmt.distance_unit)
        templates = [
            "If you and {ref} both keep to opposite walls, is the aisle wide enough to pass with at least {clear} of spare width in total?",
            "Could you and {ref} pass each other here with {clear} of combined side clearance, hugging opposite walls?",
            "With both of you against opposite walls, is there at least {clear} of spare width left over?",
        ]
        return _noul(c, c.name("room_to_pass", "aisle_wide_enough", "can_pass"), templates, {"source": "world", "event": "passable"}, clear=clearance)
    if obs.scene == "p5_braking":
        return _spill_noul(c, "stop_moderate")
    return _spill_noul(c, "turn_100")


def _spill_noul(c: _Ctx, option: str) -> Question:
    action = option_short(option, c.obs)
    if _lid_visible(c.obs):
        # 뚜껑이 보이면 쏟아짐은 늘 0 — 대신 액면이 뚜껑(림)까지 차오르나를 묻는다.
        templates = [
            "If you {action}, will the liquid in {cref} surge up against the lid?",
            "Does the surface in {cref} climb all the way to the lid if you {action}?",
            "Suppose you {action}. Does the liquid in {cref} reach the lid?",
        ]
        name = c.name(f"reaches_lid_if_{_OPT_NAME[option]}", f"lid_wet_{_OPT_NAME[option]}")
        return _noul(c, name, templates, {"source": "event", "event": "rim_reached", "option": option}, action=action)
    templates = [
        "If you {action}, will anything spill from {cref}?",
        "Does liquid slop over the rim of {cref} if you {action}?",
        "Suppose you {action}. Does {cref} spill?",
    ]
    name = c.name(f"spill_if_{_OPT_NAME[option]}", f"spills_{_OPT_NAME[option]}")
    return _noul(c, name, templates, {"source": "event", "event": "spill", "option": option}, action=action)


def _contact_noul(c: _Ctx, option: str) -> Question:
    action = option_short(option, c.obs)
    templates = [
        "If you {action}, will you touch {ref} within the next 3 seconds?",
        "Suppose you {action}. Do you make contact with {ref} in the next 3 s?",
        "Would you end up touching {ref} if you {action}?",
    ]
    name = c.name(f"contact_if_{_OPT_NAME[option]}", f"touch_{_OPT_NAME[option]}")
    return _noul(c, name, templates, {"source": "event", "event": "contact", "option": option}, action=action)


def _extra_nouls(c: _Ctx, options: tuple[str, ...]) -> list:
    """장면별 추가 noul 후보(만들 함수의 목록)."""
    obs = c.obs
    cands = []
    if obs.agent is not None:
        for option in options:
            if option not in ("go_now", "creep", "wait") or obs.variant == "stopped":
                cands.append(lambda o=option: _contact_noul(c, o))
    if obs.scene in ("p1_step_in", "p1p5_step_in_carrying"):
        if obs.agent.group == "ball":
            cands.append(lambda: _noul(c, c.name("ball_stops_in_lane", "ball_stays_in_path"), [
                "Will {ref} come to rest inside your lane?", "Does {ref} stop rolling before it leaves your lane?", "Is {ref} going to stop in your path?",
            ], {"source": "world", "event": "ball_stops_in_lane"}))  # fmt: skip
        else:
            cands.append(lambda: _noul(c, c.name("still_in_lane_2s", "in_path_in_2s"), [
                "Will {ref} still be in {lane} 2 seconds from now?", "Two seconds from now, is {ref} still inside {lane}?",
                "Is {ref} going to be in your way 2 s from now?",
            ], {"source": "world", "event": "in_lane_2s"}))  # fmt: skip
        if obs.env.sides is not None:
            for side, s in zip(("left", "right"), obs.env.sides, strict=True):
                if s.status == "occluded":
                    cands.append(lambda side=side: _noul(c, c.name(f"{side}_side_free", f"room_{side}"), [
                        "Is the space you cannot see on your {side} actually free for a swerve?", "If you swerve {side}, is there room (nothing out of sight there)?",
                        "Is the area you cannot see on your {side} empty?",
                    ], {"source": "world", "event": f"side_free_{side}"}, side=side))  # fmt: skip
    if obs.scene == "p1_crossing" and obs.agent.group == "ball":
        cands.append(lambda: _noul(c, c.name("ball_reaches_path", "ball_gets_to_path"), [
            "Will {ref} actually reach your path before it stops rolling?", "Does {ref} roll far enough to enter your path?", "Will {ref} make it into your path?",
        ], {"source": "world", "event": "ball_reaches_path"}))  # fmt: skip
    if obs.scene == "p1_head_on" and "stop_moderate" in options:
        cands.append(lambda: _noul(c, c.name("stuck_if_stop", "blocked_if_stop"), [
            "If you brake firmly to a stop now, will you and {ref} end up stuck facing each other?",
            "Stopping now, do the two of you end up blocking each other at a standstill?",
            "If you stop here, are you and {ref} left facing each other with no one able to pass?",
        ], {"source": "event", "event": "blocked", "option": "stop_moderate"}))  # fmt: skip
    if obs.container is not None:
        for option in options:
            if option not in ("continue",):
                cands.append(lambda o=option: _spill_noul(c, o))
    if obs.scene == "p5_braking":
        for option in ("stop_gentle", "stop_moderate"):
            action = option_short(option, obs)
            cands.append(lambda o=option, a=action: _noul(c, c.name(f"hit_if_{_OPT_NAME[o]}", f"overrun_{_OPT_NAME[o]}"), [
                "If you {action}, will you hit {obstacle} before you stop?", "Suppose you {action}. Do you reach {obstacle} before coming to rest?",
                "Would you run into {obstacle} if you {action}?",
            ], {"source": "event", "event": "overrun", "option": o}, action=a))  # fmt: skip
    return cands


# ----------------------------------------------------------------------------------------------------------------
# score
# ----------------------------------------------------------------------------------------------------------------


def _delay_templates(obs: Observation) -> list[str]:
    """지연 score의 지시문 — 그 장면의 지연 정의를 그대로 말한다(리뷰 1 m3)."""
    if obs.scene == "p5_braking":
        return [
            "If you {action}, how much time do you lose because you stop short of the spot right in front of {obstacle} and have to creep up the rest of the way?",
            "Rate the time lost if you {action}: stopping early means closing the remaining gap to {obstacle} again at a gentle pace.",
            "If you {action}, how long does it take to make up the distance you leave before {obstacle}, compared with stopping right at it?",
        ]
    if obs.scene == "p5_turning":
        return [
            "How much later do you come out of the turn if you {action} than if you took it without slowing?",
            "Rate the delay if you {action}, counting the slowdown, the slower arc and getting back up to speed, against not slowing at all.",
            "If you {action}, how much time does the turn cost you compared with taking it at your current speed?",
        ]
    reference = "going now" if obs.variant == "stopped" else "keeping your current speed"
    return [
        f"Over the next 3 seconds, how much progress do you lose if you {{action}}, compared with {reference}?",
        f"Rate the delay if you {{action}}: the progress you give up during the next 3 s relative to {reference}.",
        f"If you {{action}}, how far behind {reference} are you after 3 seconds, measured in seconds of travel?",
    ]


def _score_candidates(c: _Ctx, options: tuple[str, ...]) -> list:
    obs = c.obs
    cands = []
    near = "20 cm" if obs.body == "arm" else "1 m"
    for option in options:
        action = option_short(option, obs)
        short = _OPT_NAME[option]
        if obs.agent is not None and (option not in ("go_now", "creep", "wait") or obs.variant == "stopped"):
            cands.append(lambda o=option, a=action, s=short: Question(
                c.name(f"contact_risk_{s}", f"risk_{s}"), "score",
                c.fmt(c.pick(["How likely is contact with {ref} if you {action}?", "Rate the risk of touching {ref} if you {action}.",
                              "What is the chance that you touch {ref} if you {action}?"]), action=a),
                ["0 = no chance of contact", "1 = contact is certain"], {"kind": "score", "source": "event", "name": "contact", "option": o}))  # fmt: skip
            cands.append(lambda o=option, a=action, s=short: Question(
                c.name(f"closeness_{s}", f"how_close_{s}"), "score",
                c.fmt(c.pick(["How close will you come to {ref} if you {action}?", "Rate how near you pass {ref} if you {action}.",
                              "If you {action}, how tight is your closest approach to {ref}?"]), action=a),
                [f"0 = you stay at least {near} clear of {c.ref}", f"1 = you touch {c.ref}"], {"kind": "score", "source": "quantity", "name": "closeness", "option": o}))  # fmt: skip
        if obs.container is not None:
            cands.append(lambda o=option, a=action, s=short: Question(
                c.name(f"rim_reach_{s}", f"headroom_used_{s}"), "score",
                c.fmt(c.pick(["How much of the free space above the liquid in {cref} gets used up if you {action}?",
                              "Rate how close the liquid in {cref} comes to the rim if you {action}.",
                              "If you {action}, how near does the surface in {cref} climb to the rim?"]), action=a),
                ["0 = the surface barely moves", "1 = it reaches the rim (or spills)"], {"kind": "score", "source": "quantity", "name": "headroom", "option": o}))  # fmt: skip
            if _lid_visible(obs):
                cands.append(lambda o=option, a=action, s=short: Question(
                    c.name(f"lid_reach_risk_{s}", f"reaches_lid_chance_{s}"), "score",
                    c.fmt(c.pick(["How likely is the liquid in {cref} to surge up to the lid if you {action}?",
                                  "Rate the chance that the surface in {cref} reaches the lid if you {action}.",
                                  "If you {action}, what is the chance that the liquid in {cref} touches the lid?"]), action=a),
                    ["0 = it certainly stays below the lid", "1 = it certainly reaches the lid"],
                    {"kind": "score", "source": "event", "name": "rim_reached", "option": o}))  # fmt: skip
            else:
                cands.append(lambda o=option, a=action, s=short: Question(
                    c.name(f"spill_risk_{s}", f"spill_chance_{s}"), "score",
                    c.fmt(c.pick(["How likely is a spill from {cref} if you {action}?", "Rate the chance that {cref} spills if you {action}.",
                                  "What is the risk of spilling from {cref} if you {action}?"]), action=a),
                    ["0 = no chance of a spill", "1 = a spill is certain"], {"kind": "score", "source": "event", "name": "spill", "option": o}))  # fmt: skip
        if option not in ("continue", "go_now", "turn_100"):
            templates = _delay_templates(obs)
            cands.append(lambda o=option, a=action, s=short, t=templates: Question(
                c.name(f"delay_{s}", f"time_cost_{s}"), "score", c.fmt(c.pick(t), action=a),
                ["0 = no time lost", "1 = 3 seconds or more"], {"kind": "score", "source": "quantity", "name": "delay", "option": o, "transform": "per_3s"}))  # fmt: skip
        if obs.scene == "p5_braking" and option != "continue":
            cands.append(lambda o=option, a=action, s=short: Question(
                c.name(f"overrun_risk_{s}", f"hit_risk_{s}"), "score",
                c.fmt(c.pick(["How likely are you to hit {obstacle} if you {action}?", "Rate the risk of reaching {obstacle} before you stop if you {action}.",
                              "What is the chance of running into {obstacle} if you {action}?"]), action=a),
                ["0 = you certainly stop short", "1 = you certainly hit it"], {"kind": "score", "source": "event", "name": "overrun", "option": o}))  # fmt: skip
    return cands


# ----------------------------------------------------------------------------------------------------------------
# 상태 choice
# ----------------------------------------------------------------------------------------------------------------

_INTENT_TEXT = {
    "step_in": {
        "stay": ["stop and stay in your lane", "stay where they are, inside your path"],
        "cross": ["keep going across your lane", "carry on across and out of your path"],
        "back": ["step back the way they came", "retreat out of your path on the side they came from"],
    },
    "crossing": {
        "cross": ["keep coming at the same pace and cross your path", "carry on across at their current pace"],
        "yield": ["stop and wait before your path", "hold back and let you go first"],
        "hurry": ["speed up to get across first", "hurry across ahead of you"],
    },
    "head_on": {
        "keep": ["keep coming down the middle", "hold their line in the centre"],
        "aside_left": ["move over to your left", "step aside toward your left"],
        "aside_right": ["move over to your right", "step aside toward your right"],
        "stop": ["stop and wait", "come to a halt and let you decide"],
    },
}
_HAND_TEXT = {
    "stay": ["stay in the tool path", "remain where it is, in the path"],
    "cross": ["sweep on across the path", "continue across and out of the path"],
    "back": ["pull back out of the path", "withdraw the way it came"],
    "yield": ["pause before reaching the path", "hold back short of the path"],
    "hurry": ["dart across quickly", "move across faster"],
}
_STATE_KEYS = {
    "stay": "stays", "cross": "crosses", "back": "backs_off", "yield": "yields", "hurry": "hurries",
    "keep": "keeps_centre", "aside_left": "moves_left", "aside_right": "moves_right", "stop": "stops",
}  # fmt: skip


def _state_candidates(c: _Ctx, style: str) -> list:
    obs = c.obs
    cands = []
    family = {"p1_step_in": "step_in", "p1p5_step_in_carrying": "step_in", "p1_crossing": "crossing", "p1_head_on": "head_on"}.get(obs.scene)
    if obs.agent is not None and obs.agent.group != "ball":

        def intent_q():
            intents = list(INTENTS[family])
            texts = [c.pick((_HAND_TEXT if obs.agent.group == "hand" else _INTENT_TEXT[family])[i]) for i in intents]
            criteria, keymap = _choice(c, [_STATE_KEYS[i] for i in intents], [cap(t) for t in texts], style, "s")
            options = {key: [float(intents.index(_inverse_state(val)))] for key, val in keymap.items()}
            instructions = c.fmt(c.pick(["What will {ref} most likely do in the next few seconds?", "Which best describes what {ref} is about to do?",
                                         "Predict what {ref} will do next."]))  # fmt: skip
            return Question(c.name("their_next_move", "likely_move", "what_they_do"), "choice", instructions, criteria,
                            {"kind": "state", "state": "intent", "options": options})  # fmt: skip

        cands.append(intent_q)
    if obs.agent is not None and obs.agent.group == "ball" and obs.scene in ("p1_step_in", "p1p5_step_in_carrying", "p1_crossing"):

        def ball_q():
            if obs.scene == "p1_crossing":
                canon = {"reaches_path": True, "stops_short": False}
                texts = {"reaches_path": "It rolls on into your path", "stops_short": "It stops before reaching your path"}
                event = "ball_reaches_path"
            else:
                canon = {"stops_in_lane": True, "rolls_clear": False}
                texts = {"stops_in_lane": "It comes to rest inside your lane", "rolls_clear": "It rolls on out of your lane"}
                event = "ball_stops_in_lane"
            criteria, keymap = _choice(c, list(canon), [texts[k] for k in canon], style, "s")
            options = {key: [1.0 if canon[val] else 0.0] for key, val in keymap.items()}
            return Question(c.name("ball_outcome", "where_ball_ends"), "choice", c.fmt("Where does {ref} end up?"), criteria,
                            {"kind": "state", "world_event": event, "options": options})  # fmt: skip

        cands.append(ball_q)
    if obs.container is not None:

        def fill_q():
            groups = {"under_70": ([0, 1, 2, 3], "Less than 70 % full"), "from_70_to_82": ([4], "Between 70 % and 82 % full"),
                      "from_82_to_92": ([5], "Between 82 % and 92 % full"), "over_92": ([6, 7], "More than 92 % full")}  # fmt: skip
            criteria, keymap = _choice(c, list(groups), [groups[g][1] for g in groups], style, "s")
            options = {key: [float(b) for b in groups[val][0]] for key, val in keymap.items()}
            instructions = c.fmt(c.pick(["How full is {cref} most likely?", "Which fill range best matches {cref}?", "Estimate the fill level of {cref}."]))
            return Question(c.name("fill_level", "how_full"), "choice", instructions, criteria, {"kind": "state", "state": "fill_bin", "options": options})

        cands.append(fill_q)
        if obs.container.lid == "unknown":

            def lid_q():
                criteria, keymap = _choice(c, ["has_lid", "open"], ["Yes, it has a lid on", "No, it is open"], style, "s")
                options = {key: [1.0 if val == "has_lid" else 0.0] for key, val in keymap.items()}
                instructions = c.fmt(c.pick(["Does {cref} have a lid on?", "Is {cref} covered?", "Is there a lid on {cref}?"]))
                return Question(c.name("lid_on", "has_lid"), "choice", instructions, criteria, {"kind": "state", "state": "lid", "options": options})

            cands.append(lid_q)
    return cands


def _inverse_state(key: str) -> str:
    return next(k for k, v in _STATE_KEYS.items() if v == key)


# ----------------------------------------------------------------------------------------------------------------
# 규칙 적용 choice (글에 보인 수에 질문에 적힌 규칙을 적용 — 결정적)
# ----------------------------------------------------------------------------------------------------------------


def _nice(value: float, step: float, up: bool) -> float:
    k = math.ceil(value / step - 1e-9) if up else math.floor(value / step + 1e-9)
    return round(k * step, 6)


def _threshold_text(value_si: float, kind: str, unit: str) -> tuple[str, float]:
    """문턱의 글(단위 포함)과 그 글이 보이는 값(SI) — 답은 보이는 문턱으로 정한다(km/h 표기는 반올림된다)."""
    if kind == "percent":
        return textfmt.percent(value_si), round(value_si, 6)
    resolution = 0.001 if unit == "mm" else 0.01
    text = {"distance": textfmt.distance, "speed": textfmt.speed, "time": textfmt.time}[kind](value_si, resolution, unit)
    return text, textfmt.shown_si(Reading(value_si, 0.0, resolution), unit)


def _rule_two(c: _Ctx, shown: float, step: float, margin: float) -> tuple[float, bool]:
    """두 갈래 규칙의 문턱과 답(보인 값이 문턱 이상인가). 보인 값과 문턱 사이는 margin 이상 띄운다."""
    above = bool(c.rng.random() < 0.5)
    gap = margin + float(c.rng.random()) * 3 * step
    threshold = _nice(shown - gap, step, up=False) if above else _nice(shown + gap, step, up=True)
    if threshold <= 0:
        threshold, above = _nice(shown + gap, step, up=True), False
    return threshold, above


def _rule_candidates(c: _Ctx, style: str) -> list:
    obs = c.obs
    cands = []
    dist_unit = obs.fmt.distance_unit
    arm = obs.body == "arm"
    alt_units = ("cm", "mm") if arm else ("m", "cm")

    def unit_for_rule() -> str:
        return dist_unit if c.rng.random() < 0.7 else c.pick([u for u in alt_units if u != dist_unit] or [dist_unit])

    def distance_reading() -> tuple[Reading, str] | None:
        if obs.scene in ("p1_step_in", "p1p5_step_in_carrying"):
            return obs.agent.ahead, f"the gap to {c.ref}"
        if obs.scene == "p1_crossing":
            return obs.agent.track[-1], f"{c.ref}'s last reported distance from your path"
        if obs.scene == "p1_head_on":
            return obs.agent.track[-1], f"the last reported gap to {c.ref}"
        if obs.scene == "p5_braking":
            return obs.env.obstacle_distance, f"the distance to {c.obstacle}"
        return None

    found = distance_reading()
    if found is not None:
        reading, what = found

        def distance_rule():
            shown = textfmt.shown_si(reading, dist_unit)
            step = 0.01 if arm else 0.1
            margin = max(reading.resolution, step / 2)
            cls = c.pick(["far", "near", "close"])
            if cls == "far":
                hi = _nice(shown - margin - float(c.rng.random()) * 2 * step, step, up=False)
                lo = _nice(hi - step * (1 + int(c.rng.integers(3))), step, up=False)
            elif cls == "near":
                lo = _nice(shown - margin - float(c.rng.random()) * 2 * step, step, up=False)
                hi = _nice(shown + margin + float(c.rng.random()) * 2 * step, step, up=True)
            else:
                lo = _nice(shown + margin + float(c.rng.random()) * 2 * step, step, up=True)
                hi = _nice(lo + step * (1 + int(c.rng.integers(3))), step, up=True)
            if lo <= 0:
                return None
            unit = unit_for_rule()
            (lo_t, lo), (hi_t, hi) = _threshold_text(lo, "distance", unit), _threshold_text(hi, "distance", unit)
            answer = "far" if shown >= hi else ("near" if shown >= lo else "close")
            descriptions = {"far": f"at least {hi_t}", "near": f"at least {lo_t} but less than {hi_t}", "close": f"less than {lo_t}"}
            instructions = f"Site rule: classify {what} as 'far' if it is at least {hi_t}, 'near' if it is at least {lo_t} but less than {hi_t}, and 'close' otherwise."
            return _rule_question(c, ["distance_class", "gap_rule", "range_category"], instructions, ["far", "near", "close"], descriptions, answer, style)

        cands.append(distance_rule)

    if obs.agent is not None:

        def age_rule():
            age = obs.agent.age
            threshold, above = _rule_two(c, age, 0.1, 0.05)
            t_text, threshold = _threshold_text(threshold, "time", obs.fmt.time_unit)
            answer = "stale" if age > threshold else "fresh"
            instructions = c.fmt(f"Treat a track as 'stale' if its last update is older than {t_text}, otherwise 'fresh'. Which applies to {{ref}}?")
            return _rule_question(c, ["track_age_rule", "staleness"], instructions, ["stale", "fresh"],
                                  {"stale": f"last update older than {t_text}", "fresh": f"last update {t_text} old or newer"}, answer, style)  # fmt: skip

        cands.append(age_rule)

    if obs.robot.speed > 0:

        def speed_rule():
            resolution = 0.01 if arm else 0.05
            shown = textfmt.shown_si(Reading(obs.robot.speed, 0.0, resolution), obs.fmt.speed_unit)
            step = 0.05 if arm else 0.1
            threshold, above = _rule_two(c, shown, step, resolution)
            t_text, threshold = _threshold_text(threshold, "speed", obs.fmt.speed_unit)
            answer = "over_limit" if shown > threshold else "within_limit"
            instructions = f"The speed limit in this zone is {t_text}. Under that limit, is your current speed 'over_limit' or 'within_limit'?"
            return _rule_question(c, ["speed_rule", "zone_limit"], instructions, ["over_limit", "within_limit"],
                                  {"over_limit": f"faster than {t_text}", "within_limit": f"{t_text} or slower"}, answer, style)  # fmt: skip

        cands.append(speed_rule)

    if obs.container is not None and obs.container.fill_mode == "percent":

        def fill_rule():
            shown = obs.container.fill.value
            threshold, above = _rule_two(c, shown, 0.05, 0.05)
            t_text, threshold = _threshold_text(min(threshold, 0.95), "percent", "%")
            answer = "reduced" if shown > threshold else "normal"
            instructions = c.fmt(f"House rule: anything filled above {t_text} must be carried at reduced speed. Under this rule, is {{cref}} a 'reduced' or a 'normal' case?")
            return _rule_question(c, ["fill_rule", "carry_rule"], instructions, ["reduced", "normal"],
                                  {"reduced": f"filled above {t_text}", "normal": f"filled to {t_text} or less"}, answer, style)  # fmt: skip

        cands.append(fill_rule)

    if obs.env.corridor_width is not None:

        def width_rule():
            shown = textfmt.shown_si(obs.env.corridor_width, dist_unit)
            threshold, above = _rule_two(c, shown, 0.1, obs.env.corridor_width.resolution)
            t_text, threshold = _threshold_text(threshold, "distance", unit_for_rule())
            answer = "single_file" if shown < threshold else "two_way"
            instructions = f"Aisle rule: an aisle narrower than {t_text} is 'single_file'; otherwise it is 'two_way'. Which is this aisle?"
            return _rule_question(c, ["aisle_rule", "aisle_class"], instructions, ["single_file", "two_way"],
                                  {"single_file": f"narrower than {t_text}", "two_way": f"{t_text} wide or more"}, answer, style)  # fmt: skip

        cands.append(width_rule)

    if obs.env.turn_radius is not None:

        def turn_rule():
            radius = obs.env.turn_radius
            step = 0.05 if arm else 0.1
            threshold, above = _rule_two(c, radius, step, step / 2)
            t_text, threshold = _threshold_text(threshold, "distance", unit_for_rule())
            answer = "sharp" if radius < threshold else "gentle"
            instructions = f"Route rule: a turn with a radius under {t_text} counts as 'sharp', otherwise 'gentle'. How is the coming turn classed?"
            return _rule_question(c, ["turn_rule", "turn_class"], instructions, ["sharp", "gentle"],
                                  {"sharp": f"radius under {t_text}", "gentle": f"radius {t_text} or more"}, answer, style)  # fmt: skip

        cands.append(turn_rule)
    return cands


def _rule_question(c: _Ctx, names: list[str], instructions: str, classes: list[str], descriptions: dict, answer: str, style: str) -> Question:
    criteria, keymap = _choice(c, classes, [descriptions[k] for k in classes], style, "r")
    answer_key = next(k for k, v in keymap.items() if v == answer)
    return Question(c.name(*names), "choice", instructions, criteria, {"kind": "rule", "answer": answer_key, "options": list(criteria), "classes": keymap})


# ----------------------------------------------------------------------------------------------------------------


def write_questions(cfg: SimgenConfig, obs: Observation, options: tuple[str, ...], rng: np.random.Generator) -> QuestionSet:
    """요청 하나의 질문들(3~8개)."""
    c = _Ctx(cfg, obs, rng)
    style = KEY_STYLES[int(rng.integers(len(KEY_STYLES)))]
    share = cfg.values["criterion_share"]
    criteria = _criteria_for_scene(obs)

    def criterion() -> str | None:
        return c.pick(criteria) if rng.random() < share else None

    primary = _action_question(c, options, style, "a", criterion())
    questions = [primary]
    if rng.random() < 0.25:
        questions.append(_action_question(c, options, style, "b", criterion()))
    key = _key_noul(c)
    questions.append(key)
    extra = _extra_nouls(c, options)
    for i in rng.permutation(len(extra))[: int(rng.integers(0, 3))]:
        questions.append(extra[int(i)]())
    scores = _score_candidates(c, options)
    for i in rng.permutation(len(scores))[: int(rng.integers(1, 3))]:
        questions.append(scores[int(i)]())
    states = _state_candidates(c, style)
    if states and rng.random() < 0.5:
        questions.append(states[int(rng.integers(len(states)))]())
    rules = _rule_candidates(c, style)
    if rules and rng.random() < 0.35:
        rule = rules[int(rng.integers(len(rules)))]()
        if rule is not None:
            questions.append(rule)
    questions = questions[:8]
    order = list(rng.permutation(len(questions)))
    if rng.random() < 0.5:
        order.remove(0)
        order.insert(0, 0)
    questions = [questions[int(i)] for i in order]
    return QuestionSet(questions, style, primary.name, key.name)
