"""질문 생성기: 세 종류(choice·noul·score)와 흔들기 (docs/12 §7, Task v0a 지시서 §3.9, v0b §1~§4).

**관측만 받는다** — 세계 모듈을 import하지 않는다. 질문마다 모델이 받는 부분(이름·종류·지시문·criteria)과 정답을 계산하는 조리법
(``recipe``, 모델 입력이 아니다)을 함께 만든다. 정답은 :mod:`~robo_jev.simgen.labels` 가 사후 굴림 결과와 조리법으로 계산한다.

요청당 3~8개: 행동 choice 1개(25 %는 하나 더), 핵심 noul 1개(장면의 결정 여유를 묻는 것), noul 0~2개 더, score 1~2개, 상태 choice(50 %),
'다시 볼까' choice(다시 보기가 있는 장면에서 40 %), 규칙 적용 choice(35 %). 행동 choice의 약 20 %는 지시문에 기준을 적는다.

**같은 것을 두 번 묻지 않는다**(리뷰 1 재검증 m-c): 질문마다 묻는 것의 정체(:func:`identity`)가 있고 후보 묶음은 이미 물은 정체를 뺀 뒤에 고른다.

**문구가 정답과 같은 것을 묻는다**(v0c 진단의 반대 상관 점검, Task v0b): 쏟아짐·뚜껑까지 차오름·빈 높이 사용은 충돌이 있을 수 있는 순간이면
(:func:`impact_possible`) 그 원인을 문구에 적고(``*_impact`` 묶음), 닿음·가까움은 앞으로 3초를 적는다(가까움은 지금 간격도 센다).

**언어·묶음·흔들기.** ``Surface``의 ``lang``(en/ko)과 ``pool``(train/heldout — 빼 둔 바꿔 쓰기 묶음은 시험 세트 T2 전용)으로 문구 묶음을
고른다. ``shift``(지시문)·``desc_shift``(선택지 설명)는 고른 문구 번호를 그만큼 옮긴다 — 난수 소비가 같아 질문의 뜻(이름·조리법·선택지·순서)은
그대로이고 문구만 바뀐다(교란 시험 T6의 지시·설명 바꿔 쓰기). ``key_style``을 주면 선택지 열쇠 문체를 바꾼다(T6 키 바꾸기).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from robo_jev.simgen import textfmt
from robo_jev.simgen.catalog import INTENTS, SimgenConfig
from robo_jev.simgen.observation import Observation
from robo_jev.simgen.phrases import ANCHORS, ARM_VERB_KEYS, VERB_KEYS, bank, cap, container_ref, option_descriptions, option_short
from robo_jev.simgen.spec import Reading

__all__ = ["KEY_STYLES", "LONG_OPTIONS", "Question", "QuestionSet", "Surface", "VOI_SCENES", "candidates", "identity", "impact_possible", "write_questions"]

KEY_STYLES = ("word", "code", "verb")
#: 다시 보기(정보의 가치) 선택지가 있는 장면
VOI_SCENES = ("p3_place_stack", "p4_place_check", "p4_insert", "p4p6_stale_check", "p6_stale_track", "p6_blind_corner", "p6_sensor_conflict")
#: 지연이 수 초~수십 초인 선택지 — 지연 score는 30초 눈금(per_30s)
LONG_OPTIONS = ("secure_first", "detour", "recentre", "place_elsewhere", "check_weight", "press_down", "push_harder", "reseat", "call_human", "ramp_slow")


@dataclass(frozen=True)
class Surface:
    """문구 고르기의 손잡이(질문의 뜻은 바꾸지 않는다)."""

    lang: str = "en"
    pool: str = "train"
    shift: int = 0          # 지시문 묶음에서 고른 번호를 이만큼 옮긴다
    desc_shift: int = 0     # 선택지 설명 묶음에서
    key_style: str | None = None


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


def identity(recipe: dict) -> tuple:
    """질문이 묻는 것의 정체(정답 조리법에서): 같은 정체의 두 질문은 같은 정답을 갖는 같은 질문이다(m-c).

    행동·다시 볼까 = (기준, 선택지 집합); 사건 확률(noul, 사건 확률 score, 세계 사건 상태 choice) = (갈래, 사건, 선택지, 부정);
    양 score = (양, 선택지, 변환); 상태 = 상태 이름; 규칙 = 갈래 집합."""
    kind = recipe["kind"]
    if kind in ("action", "voi"):
        return (kind, recipe.get("criterion"), tuple(sorted(recipe["options"].values())))
    if kind == "noul":
        return ("event", recipe["source"], recipe["event"], recipe.get("option"), bool(recipe.get("negate")))
    if kind == "score":
        if recipe["source"] == "event":
            return ("event", "event", recipe["name"], recipe["option"], False)
        return ("quantity", recipe["name"], recipe["option"], recipe.get("transform"))
    if kind == "state":
        if "world_event" in recipe:
            return ("event", "world", recipe["world_event"], None, False)
        return ("state", recipe["state"])
    if kind == "rule":
        return ("rule", tuple(sorted(recipe["classes"].values())))
    raise ValueError(kind)


@dataclass
class _Cand:
    """후보 질문 하나: 정체(만들기 전에 안다)와 만드는 함수."""

    ident: tuple
    build: object


def _ev(source: str, event: str, option: str | None = None) -> tuple:
    return ("event", source, event, option, False)


_OPT_NAME = {
    "continue": "keep_speed", "slow": "slow", "stop_gentle": "gentle_stop", "stop_moderate": "firm_stop", "stop_hard": "hard_stop",
    "sidestep_left": "swerve_left", "sidestep_right": "swerve_right", "retract": "retract", "go_now": "go_now", "creep": "creep",
    "wait": "wait", "turn_100": "full_speed", "turn_80": "at_80pct", "turn_60": "at_60pct", "turn_45": "at_45pct", "turn_30": "at_30pct",
    "start_gentle": "gentle_start", "start_normal": "normal_start", "start_quick": "quick_start", "secure_first": "secure_first",
    "ramp_normal": "ramp_normal", "ramp_slow": "ramp_slow", "detour": "detour", "place_now": "place_now", "recentre": "recentre",
    "place_elsewhere": "elsewhere", "check_weight": "check_weight", "press_down": "press", "push_harder": "push", "reseat": "redo",
    "call_human": "call", "look_again": "look_again",
}  # fmt: skip

_STEP_IN_SCENES = ("p1_step_in", "p1p5_step_in_carrying", "p1p6_emerging_step_in", "p1p3_step_in_carrying_tall")


def impact_possible(obs: Observation) -> bool:
    """이 순간에 충돌(사람·옆 장애물·멈춰야 할 장애물)로 용기의 액체가 쏟아질 수 있나 — 그러면 쏟아짐 문구가 그 원인을 말한다."""
    if obs.container is None:
        return False
    if obs.scene in ("p1p5_step_in_carrying", "p5_braking"):
        return True
    return obs.scene == "p3p5_carrying_mixed" and obs.env.maneuver == "brake"


def _entered(agent, arm: bool) -> str:
    """끼어듦의 동사 — 몸·상대 갈래에 맞게(사람은 걸어 들어오고, 차량은 끼어들고, 공은 굴러 들어온다)."""
    if arm:
        return "has just reached into"
    group = agent.group if agent is not None else None
    return {"ball": "has rolled into", "vehicle": "has just pulled into", "robot": "has just moved into"}.get(group, "has just stepped into")


class _Ctx:
    """질문 문구에 채울 말과 난수."""

    def __init__(self, cfg: SimgenConfig, obs: Observation, rng: np.random.Generator, surface: Surface):
        self.cfg, self.obs, self.rng, self.surface = cfg, obs, rng, surface
        self.lang, self.pool = surface.lang, surface.pool
        self.names: set[str] = set()
        if self.lang == "ko":
            from robo_jev.simgen.phrases_ko import ko_values

            self.values = ko_values(obs)
        else:
            self.values = _en_values(obs)

    def text(self, key: str, **extra) -> str:
        """묶음 ``key``에서 문구 하나를 골라 채운다(``shift``만큼 옮긴 번호)."""
        templates = bank(key, self.lang, self.pool)
        index = (int(self.rng.integers(len(templates))) + self.surface.shift) % len(templates)
        return self.fill(templates[index], **extra)

    def fill(self, template: str, **extra) -> str:
        values = {**self.values, **extra}
        if self.lang == "ko":
            from robo_jev.simgen.korean import kfmt

            return kfmt(template, **values)
        return template.format(**values)

    def action(self, canon: str) -> dict:
        """{action} 자리(영어: 명령형; 한국어: 조건형·명사형)."""
        if self.lang == "ko":
            from robo_jev.simgen.phrases_ko import option_forms

            return option_forms(canon, self.obs)
        return {"action": option_short(canon, self.obs)}

    def describe(self, canon: str) -> str:
        if self.lang == "ko":
            from robo_jev.simgen.phrases_ko import option_descriptions as ko_descriptions

            variants = ko_descriptions(canon, self.obs, self.pool)
        else:
            variants = option_descriptions(canon, self.obs, self.pool)
        index = (int(self.rng.integers(len(variants))) + self.surface.desc_shift) % len(variants)
        return variants[index]

    def anchors(self, key: str, **extra) -> list[str]:
        if self.lang == "ko":
            from robo_jev.simgen.phrases_ko import ANCHORS as KO_ANCHORS

            return [self.fill(t, **extra) for t in KO_ANCHORS[key]]
        return [self.fill(t, **extra) for t in ANCHORS[key]]

    def pick(self, items):
        return items[int(self.rng.integers(len(items)))]

    def name(self, *candidates: str) -> str:
        base = self.pick(list(candidates))
        name, i = base, 2
        while name in self.names:
            name, i = f"{base}_{i}", i + 1
        self.names.add(name)
        return name


def _en_values(obs: Observation) -> dict:
    """영어 문구의 자리 표시 값."""
    a = obs.agent
    arm = obs.body == "arm"
    ref = a.descriptor.ref if a is not None else "the obstacle"
    values = {
        "ref": ref, "Ref": cap(ref), "ref_bare": ref[4:] if ref.startswith("the ") else ref,
        "cref": container_ref(obs.container.phrase) if obs.container is not None else "the container",
        "obstacle": obs.env.obstacle or "the obstacle",
        "lane": "the tool's path" if arm else "your lane",
        "poss": "their" if a is not None and a.group == "person" else "its",
        "entered": _entered(a, arm),
    }  # fmt: skip
    values["hazard"] = values["obstacle"] if obs.env.obstacle is not None else (f"{ref} or anything beside you" if a is not None else "anything")
    if obs.load is not None:
        values.update(lref=obs.load.ref, Lref=cap(obs.load.ref), deck=obs.load.deck_phrase)
    if obs.stack is not None:
        values.update(sref=obs.stack.ref, Sref=cap(obs.stack.ref), spot=obs.stack.spot)
    if obs.place is not None:
        values.update(item=obs.place.item, Item=cap(obs.place.item), target=obs.place.target)
    if obs.env.ramp is not None:
        values["ramp"] = obs.env.ramp
    if obs.stale is not None:
        values.update(ref=obs.stale.ref, Ref=cap(obs.stale.ref), spot=obs.stale.spot, obj=obs.stale.pronoun)
    if obs.corner is not None:
        values.update(corner=obs.corner.phrase, Corner=cap(obs.corner.phrase))
    if obs.conflict is not None:
        values["dist"] = textfmt.distance(obs.conflict.distance.value, obs.conflict.distance.resolution, obs.fmt.distance_unit)
    if obs.env.occluder is not None:
        values["occluder"] = obs.env.occluder
    return values


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
    keys = _option_keys(canons, style, prefix, c.obs) if prefix in ("a", "b") else _state_keys(canons, style, prefix)
    return dict(zip(keys, descriptions, strict=True)), dict(zip(keys, canons, strict=True))


def _state_keys(canons: list[str], style: str, prefix: str) -> list[str]:
    if style == "code":
        return [f"{prefix}{i + 1}" for i in range(len(canons))]
    return list(canons)


# ----------------------------------------------------------------------------------------------------------------
# 행동 choice와 '다시 볼까'
# ----------------------------------------------------------------------------------------------------------------


def _lid_visible(obs: Observation) -> bool:
    return obs.container is not None and obs.container.lid == "lid"


def _has_contact(obs: Observation) -> bool:
    return obs.agent is not None or obs.stale is not None or obs.corner is not None or obs.conflict is not None


def _criteria_for_scene(obs: Observation) -> list[str]:
    """그 순간에 뜻이 있는 기준(비용 덮어쓰기) — v0a 장면은 v0a와 같다."""
    out: list[str] = []
    if _has_contact(obs):
        out.append("avoid_contact")
    if obs.container is not None and not _lid_visible(obs):
        out.append("avoid_spill")
    if obs.load is not None or obs.stack is not None:
        out.append("avoid_fall")
    if obs.place is not None:
        out.append("no_defect")
    out.append("time_critical")
    return out


def _act_key(obs: Observation) -> str:
    scene = obs.scene
    if scene == "p1_crossing" and obs.variant == "stopped":
        return "act.p1_crossing_stopped"
    if scene == "p3_carrying_tall":
        return {"brake": "act.p3_brake", "turn": "act.p3_turn", "start": "act.p3_start"}[obs.env.maneuver]
    return {"p1p3_step_in_carrying_tall": "act.p1_step_in", "p3p5_carrying_mixed": "act.p3p5", "p4p6_stale_check": "act.p4_place_check",
            "p1p6_emerging_step_in": "act.p1p6"}.get(scene, f"act.{scene}")  # fmt: skip


def _action_question(c: _Ctx, options: tuple[str, ...], style: str, prefix: str, criterion: str | None) -> Question:
    canons = list(options)
    if len(canons) > 2 and c.rng.random() < 0.35:
        size = int(c.rng.integers(max(2, min(3, len(canons) - 1)), len(canons)))
        chosen = sorted(c.rng.choice(len(canons), size=size, replace=False))
        canons = [canons[int(i)] for i in chosen]
    descriptions = [c.describe(canon) for canon in canons]
    criteria, keymap = _choice(c, canons, descriptions, style, prefix)
    if criterion is not None:
        instructions = c.text(f"crit.{criterion}")
        name = c.name("action_under_rule", "constrained_action", "action_with_rule")
    else:
        key = "act.generic" if c.rng.random() < 0.4 else _act_key(c.obs)
        instructions = c.text(key)
        name = c.name("next_action", "what_to_do", "action", "decision", "response")
    return Question(name, "choice", instructions, criteria, {"kind": "action", "options": keymap, "criterion": criterion})


def _look_phrase(c: _Ctx) -> str:
    if c.lang == "ko":
        from robo_jev.simgen.phrases_ko import look_phrase

        return look_phrase(c.obs)
    obs = c.obs
    if obs.place is not None:
        return "measure it again"
    if obs.stack is not None:
        return "check how the weight sits"
    if obs.corner is not None:
        return "stop at the corner and look"
    return "pause and look again"


def look_time(obs: Observation) -> float:
    """다시 보기에 드는 시간(글에 적힌 값)."""
    if obs.place is not None:
        return obs.place.action_times["look_again"]
    if obs.stack is not None:
        return obs.stack.check_s
    return obs.env.look_s


def _voi_question(c: _Ctx, options: tuple[str, ...], style: str) -> Question:
    """'다시 볼까' — 지금 정하기(지금 가능한 행동 가운데 최선) 대 먼저 다시 보기(보는 시간 + 본 뒤의 최선)."""
    look = _look_phrase(c)
    t = textfmt.time(look_time(c.obs), 0.1, c.obs.fmt.time_unit)
    descriptions = [c.text("voi.decide_now"), c.text("voi.look_first", look=look, t=t)]
    criteria, keymap = _choice(c, ["decide_now", "look_first"], descriptions, style, "v")
    instructions = c.text("voi", look=look)
    base = sorted(o for o in options if o != "look_again")
    recipe = {"kind": "voi", "options": {k: ("look_again" if v == "look_first" else "decide_now") for k, v in keymap.items()}, "criterion": None, "base": base}
    return Question(c.name("look_first_or_decide", "check_again_first", "worth_a_second_look"), "choice", instructions, criteria, recipe)


# ----------------------------------------------------------------------------------------------------------------
# noul
# ----------------------------------------------------------------------------------------------------------------


def _noul(c: _Ctx, name: str, key: str, recipe: dict, **extra) -> Question:
    return Question(name, "noul", c.text(key, **extra), None, {"kind": "noul", **recipe})


def _spill_noul(c: _Ctx, option: str) -> Question:
    impact = impact_possible(c.obs)
    if _lid_visible(c.obs):
        name = c.name(f"reaches_lid_if_{_OPT_NAME[option]}", f"lid_wet_{_OPT_NAME[option]}")
        return _noul(c, name, "noul.rim_impact" if impact else "noul.rim_slosh", {"source": "event", "event": "rim_reached", "option": option}, **c.action(option))
    name = c.name(f"spill_if_{_OPT_NAME[option]}", f"spills_{_OPT_NAME[option]}")
    return _noul(c, name, "noul.spill_impact" if impact else "noul.spill_slosh", {"source": "event", "event": "spill", "option": option}, **c.action(option))


def p3_reference_option(obs: Observation) -> str:
    """P3의 핵심 noul이 묻는 기준 움직임(결정 여유의 기준 가속과 같은 것)."""
    if obs.scene == "p3_ramp":
        return "ramp_normal"
    return {"brake": "stop_moderate", "turn": "turn_100", "start": "start_normal"}[obs.env.maneuver]


def _key_noul(c: _Ctx) -> Question:
    obs = c.obs
    scene = obs.scene
    if scene in _STEP_IN_SCENES:
        key = "key.stops_short_arm" if obs.body == "arm" else "key.stops_short"
        return _noul(c, c.name("stops_in_time", "firm_stop_short", "can_stop_short"), key, {"source": "world", "event": "stops_short"})
    if scene == "p1_crossing":
        key = "key.clears_first_stopped" if obs.variant == "stopped" else "key.clears_first"
        return _noul(c, c.name("clear_before_them", "crossing_in_time", "clears_first"), key, {"source": "world", "event": "clears_first"})
    if scene == "p1_head_on":
        clearance = textfmt.distance(obs.robot.passing_clearance, 0.01, "cm" if obs.fmt.distance_unit == "cm" else obs.fmt.distance_unit)
        return _noul(c, c.name("room_to_pass", "aisle_wide_enough", "can_pass"), "key.passable", {"source": "world", "event": "passable"}, clear=clearance)
    if scene == "p5_braking":
        return _spill_noul(c, "stop_moderate")
    if scene == "p5_turning":
        return _spill_noul(c, "turn_100")
    if scene in ("p3_carrying_tall", "p3p5_carrying_mixed", "p3_ramp"):
        option = p3_reference_option(obs)
        return _noul(c, c.name(f"falls_if_{_OPT_NAME[option]}", f"topples_{_OPT_NAME[option]}"), "noul.fall", {"source": "event", "event": "fall", "option": option}, **c.action(option))
    if scene == "p3_place_stack":
        return _noul(c, c.name("topples_if_placed", "falls_off_stack"), "noul.topple_stack", {"source": "event", "event": "topple", "option": "place_now"}, **c.action("place_now"))
    if scene in ("p4_place_check", "p4p6_stale_check"):
        return _noul(c, c.name("seated_now", "properly_seated", "in_place"), "noul.seated", {"source": "world", "event": "ok_now"})
    if scene == "p4_insert":
        return _noul(c, c.name("fully_inserted", "inserted_now", "fully_home"), "noul.inserted", {"source": "world", "event": "ok_now"})
    if scene == "p6_stale_track":
        return _noul(c, c.name("still_there", "still_in_the_way", "has_stayed"), "noul.still_there", {"source": "world", "event": "present"})
    if scene == "p6_blind_corner":
        return _noul(c, c.name("hit_if_keep_speed", "contact_if_keep_speed"), "noul.contact_emerge", {"source": "event", "event": "contact", "option": "continue"}, **c.action("continue"))
    if scene == "p6_sensor_conflict":
        return _noul(c, c.name("really_there", "obstacle_is_real", "something_there"), "noul.present", {"source": "world", "event": "present"})
    raise ValueError(scene)


def _contact_key(obs: Observation) -> str:
    return {"p6_stale_track": "contact_stale", "p6_blind_corner": "contact_emerge", "p6_sensor_conflict": "contact_obstacle", "p1p6_emerging_step_in": "contact_any"}.get(obs.scene, "contact")


def _contact_options(obs: Observation, options: tuple[str, ...]) -> list[str]:
    """닿음을 물을 선택지(다시 보기는 빼고, 움직이는 교차에서는 멈춘 변형의 선택지를 빼고)."""
    out = []
    for option in options:
        if option == "look_again":
            continue
        if obs.scene == "p1_crossing" and obs.variant != "stopped" and option in ("go_now", "creep", "wait"):
            continue
        out.append(option)
    return out


def _extra_nouls(c: _Ctx, options: tuple[str, ...]) -> list:
    """장면별 추가 noul 후보."""
    obs = c.obs
    scene = obs.scene
    cands = []
    if _has_contact(obs):
        key = f"noul.{_contact_key(obs)}"
        for option in _contact_options(obs, options):
            cands.append(_Cand(_ev("event", "contact", option), lambda o=option: _noul(
                c, c.name(f"contact_if_{_OPT_NAME[o]}", f"touch_{_OPT_NAME[o]}"), key, {"source": "event", "event": "contact", "option": o}, **c.action(o))))
    if scene in _STEP_IN_SCENES:
        if obs.agent.group == "ball":
            cands.append(_Cand(_ev("world", "ball_stops_in_lane"), lambda: _noul(
                c, c.name("ball_stops_in_lane", "ball_stays_in_path"), "noul.ball_stops_in_lane", {"source": "world", "event": "ball_stops_in_lane"})))
        else:
            cands.append(_Cand(_ev("world", "in_lane_2s"), lambda: _noul(
                c, c.name("still_in_lane_2s", "in_path_in_2s"), "noul.in_lane_2s", {"source": "world", "event": "in_lane_2s"})))
        if obs.env.sides is not None:
            for side, s in zip(("left", "right"), obs.env.sides, strict=True):
                if s.status == "occluded":
                    cands.append(_Cand(_ev("world", f"side_free_{side}"), lambda side=side: _noul(
                        c, c.name(f"{side}_side_free", f"room_{side}"), "noul.side_free", {"source": "world", "event": f"side_free_{side}"}, **_side_values(c, side))))
        if scene == "p1p6_emerging_step_in":
            cands.append(_Cand(_ev("world", "follower"), lambda: _noul(c, c.name("someone_following", "second_person"), "noul.follower", {"source": "world", "event": "follower"})))
    if scene == "p1_crossing" and obs.agent.group == "ball":
        cands.append(_Cand(_ev("world", "ball_reaches_path"), lambda: _noul(
            c, c.name("ball_reaches_path", "ball_gets_to_path"), "noul.ball_reaches_path", {"source": "world", "event": "ball_reaches_path"})))
    if scene == "p1_head_on" and "stop_moderate" in options:
        cands.append(_Cand(_ev("event", "blocked", "stop_moderate"), lambda: _noul(
            c, c.name("stuck_if_stop", "blocked_if_stop"), "noul.blocked", {"source": "event", "event": "blocked", "option": "stop_moderate"})))
    if obs.container is not None:
        event = "rim_reached" if _lid_visible(obs) else "spill"
        for option in options:
            if option in ("continue", "secure_first", "detour"):
                continue
            cands.append(_Cand(_ev("event", event, option), lambda o=option: _spill_noul(c, o)))
    if scene == "p5_braking" or (obs.load is not None and obs.env.maneuver == "brake"):
        for option in ("stop_gentle", "stop_moderate"):
            if option in options:
                cands.append(_Cand(_ev("event", "overrun", option), lambda o=option: _noul(
                    c, c.name(f"hit_if_{_OPT_NAME[o]}", f"overrun_{_OPT_NAME[o]}"), "noul.overrun", {"source": "event", "event": "overrun", "option": o}, **c.action(o))))
    if obs.load is not None:
        for option in options:
            if option in ("secure_first", "detour"):
                continue
            for event, key, names in (("fall", "noul.fall", ("falls_if", "topples")), ("tip", "noul.tip", ("tips_if", "tips_over")), ("slide", "noul.slide", ("slides_if", "slips"))):
                cands.append(_Cand(_ev("event", event, option), lambda o=option, e=event, k=key, n=names: _noul(
                    c, c.name(f"{n[0]}_{_OPT_NAME[o]}", f"{n[1]}_{_OPT_NAME[o]}"), k, {"source": "event", "event": e, "option": o}, **c.action(o))))
    if obs.stack is not None and "recentre" in options:
        cands.append(_Cand(_ev("event", "topple", "recentre"), lambda: _noul(
            c, c.name("topples_if_recentred", "falls_off_if_recentred"), "noul.topple_stack", {"source": "event", "event": "topple", "option": "recentre"}, **c.action("recentre"))))
    if obs.place is not None:
        for option in options:
            if option == "look_again":
                continue
            cands.append(_Cand(_ev("event", "ok_after", option), lambda o=option: _noul(
                c, c.name(f"ok_if_{_OPT_NAME[o]}", f"in_place_if_{_OPT_NAME[o]}"), "noul.ok_after", {"source": "event", "event": "ok_after", "option": o}, **c.action(o))))
            if option in ("press_down", "push_harder"):
                cands.append(_Cand(_ev("event", "damage", option), lambda o=option: _noul(
                    c, c.name(f"breaks_if_{_OPT_NAME[o]}", f"damage_if_{_OPT_NAME[o]}"), "noul.damage", {"source": "event", "event": "damage", "option": o}, **c.action(o))))
    if scene == "p6_blind_corner":
        cands.append(_Cand(_ev("world", "emerge_3s"), lambda: _noul(c, c.name("someone_comes_out", "anyone_emerging"), "noul.emerge", {"source": "world", "event": "emerge_3s"})))
    return cands


def _side_values(c: _Ctx, side: str) -> dict:
    if c.lang == "ko":
        return {"side": {"left": "왼쪽", "right": "오른쪽"}[side]}
    return {"side": side}


# ----------------------------------------------------------------------------------------------------------------
# score
# ----------------------------------------------------------------------------------------------------------------


def _delay_key(obs: Observation, option: str) -> str:
    if option in LONG_OPTIONS or obs.scene in ("p3_ramp", "p3_place_stack") or obs.place is not None:
        return "score.delay.time"
    if obs.scene == "p5_braking" or (obs.load is not None and obs.env.maneuver == "brake"):
        return "score.delay.p5_braking"
    if obs.scene == "p5_turning" or (obs.load is not None and obs.env.maneuver == "turn"):
        return "score.delay.p5_turning"
    return "score.delay.p1"


def _delay_reference(c: _Ctx) -> str:
    obs = c.obs
    if c.lang == "ko":
        from robo_jev.simgen.phrases_ko import delay_reference

        return delay_reference(obs)
    if obs.variant == "stopped":
        return "going now"
    if obs.load is not None and obs.env.maneuver == "start":
        return "pulling away quickly"
    return "keeping your current speed"


def _score(c: _Ctx, names: tuple[str, ...], key: str, anchors: list[str], recipe: dict, **extra) -> Question:
    return Question(c.name(*names), "score", c.text(key, **extra), anchors, {"kind": "score", **recipe})


_NO_DELAY = ("continue", "go_now", "turn_100", "place_now", "ramp_normal", "start_quick", "look_again")


def _score_candidates(c: _Ctx, options: tuple[str, ...]) -> list:
    obs = c.obs
    scene = obs.scene
    cands = []
    near = textfmt.distance(0.2 if obs.body == "arm" else 1.0, 0.01, "cm" if obs.body == "arm" else "m")
    impact = impact_possible(obs)
    contact_options = set(_contact_options(obs, options)) if _has_contact(obs) else set()
    score_key = {"contact": "score.contact", "contact_any": "score.contact", "contact_stale": "score.contact_stale",
                 "contact_emerge": "score.contact_emerge", "contact_obstacle": "score.contact_obstacle"}[_contact_key(obs)]  # fmt: skip
    for option in options:
        if option == "look_again":
            continue
        act = c.action(option)
        short = _OPT_NAME[option]
        if option in contact_options:
            cands.append(_Cand(_ev("event", "contact", option), lambda o=option, a=act, s=short: _score(
                c, (f"contact_risk_{s}", f"risk_{s}"), score_key, c.anchors("contact"), {"source": "event", "name": "contact", "option": o}, **a)))
            if obs.agent is not None:
                cands.append(_Cand(("quantity", "closeness", option, None), lambda o=option, a=act, s=short: _score(
                    c, (f"closeness_{s}", f"how_close_{s}"), "score.closeness", c.anchors("closeness", near=near), {"source": "quantity", "name": "closeness", "option": o}, **a)))
        if obs.container is not None and option not in ("secure_first", "detour"):
            hkey = "headroom_impact" if impact else "headroom_slosh"
            cands.append(_Cand(("quantity", "headroom", option, None), lambda o=option, a=act, s=short, k=hkey: _score(
                c, (f"rim_reach_{s}", f"headroom_used_{s}"), f"score.{k}", c.anchors(k), {"source": "quantity", "name": "headroom", "option": o}, **a)))
            if _lid_visible(obs):
                rkey = "score.rim_impact" if impact else "score.rim_slosh"
                cands.append(_Cand(_ev("event", "rim_reached", option), lambda o=option, a=act, s=short, k=rkey: _score(
                    c, (f"lid_reach_risk_{s}", f"reaches_lid_chance_{s}"), k, c.anchors("rim"), {"source": "event", "name": "rim_reached", "option": o}, **a)))
            else:
                skey = "score.spill_impact" if impact else "score.spill_slosh"
                cands.append(_Cand(_ev("event", "spill", option), lambda o=option, a=act, s=short, k=skey: _score(
                    c, (f"spill_risk_{s}", f"spill_chance_{s}"), k, c.anchors("spill"), {"source": "event", "name": "spill", "option": o}, **a)))
        if option not in _NO_DELAY:
            dkey = _delay_key(obs, option)
            transform = "per_30s" if dkey == "score.delay.time" else "per_3s"
            anchors = c.anchors("delay_long", long=textfmt.time(30.0, 1.0, obs.fmt.time_unit)) if transform == "per_30s" else c.anchors("delay")
            cands.append(_Cand(("quantity", "delay", option, transform), lambda o=option, a=act, s=short, k=dkey, t=transform, an=anchors: _score(
                c, (f"delay_{s}", f"time_cost_{s}"), k, an, {"source": "quantity", "name": "delay", "option": o, "transform": t}, reference=_delay_reference(c), **a)))
        if (scene == "p5_braking" or (obs.load is not None and obs.env.maneuver == "brake")) and option != "continue":
            cands.append(_Cand(_ev("event", "overrun", option), lambda o=option, a=act, s=short: _score(
                c, (f"overrun_risk_{s}", f"hit_risk_{s}"), "score.overrun", c.anchors("overrun"), {"source": "event", "name": "overrun", "option": o}, **a)))
        if obs.load is not None and option not in ("secure_first", "detour"):
            cands.append(_Cand(_ev("event", "fall", option), lambda o=option, a=act, s=short: _score(
                c, (f"fall_risk_{s}", f"topple_risk_{s}"), "score.fall", c.anchors("fall"), {"source": "event", "name": "fall", "option": o}, **a)))
            cands.append(_Cand(("quantity", "slide_frac", option, None), lambda o=option, a=act, s=short: _score(
                c, (f"slide_{s}", f"slip_toward_edge_{s}"), "score.slide", c.anchors("slide"), {"source": "quantity", "name": "slide_frac", "option": o}, **a)))
        if obs.stack is not None and option in ("place_now", "recentre"):
            cands.append(_Cand(_ev("event", "topple", option), lambda o=option, a=act, s=short: _score(
                c, (f"topple_risk_{s}", f"falls_off_risk_{s}"), "score.topple_stack", c.anchors("topple"), {"source": "event", "name": "topple", "option": o}, **a)))
        if obs.place is not None:
            cands.append(_Cand(("quantity", "bad_after", option, None), lambda o=option, a=act, s=short: _score(
                c, (f"bad_result_{s}", f"left_wrong_{s}"), "score.unseated", c.anchors("unseated"), {"source": "quantity", "name": "bad_after", "option": o}, **a)))
            if option in ("press_down", "push_harder"):
                cands.append(_Cand(_ev("event", "damage", option), lambda o=option, a=act, s=short: _score(
                    c, (f"damage_risk_{s}", f"break_risk_{s}"), "score.damage", c.anchors("damage"), {"source": "event", "name": "damage", "option": o}, **a)))
    return cands


# ----------------------------------------------------------------------------------------------------------------
# 상태 choice
# ----------------------------------------------------------------------------------------------------------------

_STATE_KEYS = {
    "stay": "stays", "cross": "crosses", "back": "backs_off", "yield": "yields", "hurry": "hurries",
    "keep": "keeps_centre", "aside_left": "moves_left", "aside_right": "moves_right", "stop": "stops",
}  # fmt: skip
_INTENT_FAMILY = {"p1_step_in": "step_in", "p1p5_step_in_carrying": "step_in", "p1_crossing": "crossing", "p1_head_on": "head_on",
                  "p1p6_emerging_step_in": "step_in", "p1p3_step_in_carrying_tall": "step_in"}  # fmt: skip


def _state_candidates(c: _Ctx, style: str) -> list:
    from robo_jev.simgen.state_text import state_texts

    obs = c.obs
    texts = state_texts(c.lang)
    cands = []
    family = _INTENT_FAMILY.get(obs.scene)
    if obs.agent is not None and obs.agent.group != "ball" and obs.agent.cue is not None and family is not None:

        def intent_q():
            intents = list(INTENTS[family])
            table = texts["hand"] if obs.agent.group == "hand" else texts["intent"][family]
            criteria, keymap = _choice(c, [_STATE_KEYS[i] for i in intents], [cap(c.pick(table[i])) for i in intents], style, "s")
            options = {key: [float(intents.index(_inverse_state(val)))] for key, val in keymap.items()}
            return Question(c.name("their_next_move", "likely_move", "what_they_do"), "choice", c.text("state.intent"), criteria,
                            {"kind": "state", "state": "intent", "options": options})  # fmt: skip

        cands.append(_Cand(("state", "intent"), intent_q))
    if obs.agent is not None and obs.agent.group == "ball" and obs.scene in ("p1_step_in", "p1p5_step_in_carrying", "p1_crossing"):
        event = "ball_reaches_path" if obs.scene == "p1_crossing" else "ball_stops_in_lane"
        canon = {"reaches_path": True, "stops_short": False} if obs.scene == "p1_crossing" else {"stops_in_lane": True, "rolls_clear": False}

        def ball_q():
            criteria, keymap = _choice(c, list(canon), [texts["ball"][k] for k in canon], style, "s")
            options = {key: [1.0 if canon[val] else 0.0] for key, val in keymap.items()}
            return Question(c.name("ball_outcome", "where_ball_ends"), "choice", c.text("state.ball"), criteria, {"kind": "state", "world_event": event, "options": options})

        cands.append(_Cand(_ev("world", event), ball_q))
    if obs.container is not None:

        def fill_q():
            groups = {"under_70": [0, 1, 2, 3], "from_70_to_82": [4], "from_82_to_92": [5], "over_92": [6, 7]}
            criteria, keymap = _choice(c, list(groups), [texts["fill"][g] for g in groups], style, "s")
            options = {key: [float(b) for b in groups[val]] for key, val in keymap.items()}
            return Question(c.name("fill_level", "how_full"), "choice", c.text("state.fill"), criteria, {"kind": "state", "state": "fill_bin", "options": options})

        cands.append(_Cand(("state", "fill_bin"), fill_q))
        if obs.container.lid == "unknown":

            def lid_q():
                criteria, keymap = _choice(c, ["has_lid", "open"], [texts["lid"]["has_lid"], texts["lid"]["open"]], style, "s")
                options = {key: [1.0 if val == "has_lid" else 0.0] for key, val in keymap.items()}
                return Question(c.name("lid_on", "has_lid"), "choice", c.text("state.lid"), criteria, {"kind": "state", "state": "lid", "options": options})

            cands.append(_Cand(("state", "lid"), lid_q))
    if obs.place is not None:
        family_ = obs.place.family

        def p4_q():
            states = list(texts["p4"][family_])
            criteria, keymap = _choice(c, states, [c.fill(texts["p4"][family_][s]) for s in states], style, "s")
            options = {key: [float(states.index(val))] for key, val in keymap.items()}
            key = "state.p4_place" if family_ == "place" else "state.p4_insert"
            return Question(c.name("its_state", "how_it_sits", "condition_now"), "choice", c.text(key), criteria, {"kind": "state", "state": "p4_state", "options": options})

        cands.append(_Cand(("state", "p4_state"), p4_q))
    if obs.conflict is not None:
        lidar_says = obs.conflict.lidar

        def sensor_q():
            canon = {"lidar_right": 1.0 if lidar_says else 0.0, "camera_right": 0.0 if lidar_says else 1.0}
            criteria, keymap = _choice(c, list(canon), [texts["sensor"][k][int(lidar_says)] for k in canon], style, "s")
            options = {key: [canon[val]] for key, val in keymap.items()}
            return Question(c.name("which_sensor", "believe_which"), "choice", c.text("state.sensor"), criteria, {"kind": "state", "state": "present", "options": options})

        cands.append(_Cand(("state", "present"), sensor_q))
    if obs.stack is not None and obs.stack.offset != 0.0:

        def weight_q():
            canon = {"toward_overhang": 1.0, "away_from_overhang": 0.0}
            criteria, keymap = _choice(c, list(canon), [c.fill(texts["weight"][k]) for k in canon], style, "s")
            options = {key: [canon[val]] for key, val in keymap.items()}
            return Question(c.name("heavy_side", "weight_side"), "choice", c.text("state.weight_side"), criteria, {"kind": "state", "state": "weight_toward", "options": options})

        cands.append(_Cand(("state", "weight_toward"), weight_q))
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
    if kind == "plain":
        return textfmt.number(value_si, 1), round(value_si, 6)
    if kind == "angle":
        return f"{textfmt.number(value_si, 1)}°", round(value_si, 6)
    if kind == "mm":
        return f"{textfmt.number(value_si, 1)} mm", round(value_si, 6)
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


def _distance_reading(c: _Ctx, what: dict) -> tuple[Reading, str] | None:
    obs = c.obs
    if obs.scene in _STEP_IN_SCENES:
        return obs.agent.ahead, c.fill(what["gap"])
    if obs.scene == "p1_crossing":
        return obs.agent.track[-1], c.fill(what["crossing"])
    if obs.scene == "p1_head_on":
        return obs.agent.track[-1], c.fill(what["head_on"])
    if obs.env.obstacle_distance is not None:
        return obs.env.obstacle_distance, c.fill(what["obstacle"])
    if obs.conflict is not None:
        return obs.conflict.distance, c.fill(what["conflict"])
    if obs.stale is not None:
        return obs.stale.ahead, c.fill(what["stale"])
    if obs.corner is not None:
        return obs.corner.ahead, c.fill(what["corner"])
    return None


def _two_way_rule(c: _Ctx, style: str, r: dict, names: list[str], classes: tuple[str, str], value: float, step: float, margin: float, kind: str, unit: str, above_class: str, strict_above: bool = True):
    """두 갈래 규칙 질문 하나: 보인 값 ``value``가 문턱보다 크면(또는 작으면) ``above_class``."""
    threshold, _ = _rule_two(c, value, step, margin)
    t_text, threshold = _threshold_text(threshold, kind, unit)
    above = value > threshold if strict_above else value >= threshold
    other = classes[1] if above_class == classes[0] else classes[0]
    answer = above_class if above else other
    return _rule_question(c, names, c.fill(r["instructions"], t=t_text), list(classes), {k: c.fill(r[k], t=t_text) for k in classes}, answer, style)


def _rule_candidates(c: _Ctx, style: str) -> list:
    from robo_jev.simgen.rule_text import rule_texts

    obs = c.obs
    rt = rule_texts(c.lang)
    cands = []
    dist_unit = obs.fmt.distance_unit
    arm = obs.body == "arm"
    alt_units = ("cm", "mm") if arm else ("m", "cm")

    def unit_for_rule() -> str:
        return dist_unit if c.rng.random() < 0.7 else c.pick([u for u in alt_units if u != dist_unit] or [dist_unit])

    found = _distance_reading(c, rt["what"])
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
            r = rt["distance"]
            descriptions = {k: c.fill(r[k], lo=lo_t, hi=hi_t) for k in ("far", "near", "close")}
            instructions = c.fill(r["instructions"], what=what, lo=lo_t, hi=hi_t)
            return _rule_question(c, ["distance_class", "gap_rule", "range_category"], instructions, ["far", "near", "close"], descriptions, answer, style)

        cands.append(distance_rule)
    age = obs.agent.age if obs.agent is not None else (obs.stale.age if obs.stale is not None else None)
    if age is not None:
        cands.append(lambda: _two_way_rule(c, style, rt["age"], ["track_age_rule", "staleness"], ("stale", "fresh"), age, 0.1, 0.05, "time", obs.fmt.time_unit, "stale"))
    if obs.robot.speed > 0:
        resolution = 0.01 if arm else 0.05
        shown_speed = textfmt.shown_si(Reading(obs.robot.speed, 0.0, resolution), obs.fmt.speed_unit)
        cands.append(lambda: _two_way_rule(c, style, rt["speed"], ["speed_rule", "zone_limit"], ("over_limit", "within_limit"), shown_speed, 0.05 if arm else 0.1, resolution,
                                           "speed", obs.fmt.speed_unit, "over_limit"))  # fmt: skip
    if obs.container is not None and obs.container.fill_mode == "percent":

        def fill_rule():
            shown = obs.container.fill.value
            threshold, _ = _rule_two(c, shown, 0.05, 0.05)
            t_text, threshold = _threshold_text(min(threshold, 0.95), "percent", "%")
            answer = "reduced" if shown > threshold else "normal"
            r = rt["fill"]
            return _rule_question(c, ["fill_rule", "carry_rule"], c.fill(r["instructions"], t=t_text), ["reduced", "normal"],
                                  {k: c.fill(r[k], t=t_text) for k in ("reduced", "normal")}, answer, style)  # fmt: skip

        cands.append(fill_rule)
    if obs.env.corridor_width is not None:

        def width_rule():
            shown = textfmt.shown_si(obs.env.corridor_width, dist_unit)
            threshold, _ = _rule_two(c, shown, 0.1, obs.env.corridor_width.resolution)
            t_text, threshold = _threshold_text(threshold, "distance", unit_for_rule())
            answer = "single_file" if shown < threshold else "two_way"
            r = rt["width"]
            return _rule_question(c, ["aisle_rule", "aisle_class"], c.fill(r["instructions"], t=t_text), ["single_file", "two_way"],
                                  {k: c.fill(r[k], t=t_text) for k in ("single_file", "two_way")}, answer, style)  # fmt: skip

        cands.append(width_rule)
    if obs.env.turn_radius is not None:

        def turn_rule():
            radius = obs.env.turn_radius
            step = 0.05 if arm else 0.1
            threshold, _ = _rule_two(c, radius, step, step / 2)
            t_text, threshold = _threshold_text(threshold, "distance", unit_for_rule())
            answer = "sharp" if radius < threshold else "gentle"
            r = rt["turn"]
            return _rule_question(c, ["turn_rule", "turn_class"], c.fill(r["instructions"], t=t_text), ["sharp", "gentle"],
                                  {k: c.fill(r[k], t=t_text) for k in ("sharp", "gentle")}, answer, style)  # fmt: skip

        cands.append(turn_rule)
    if obs.load is not None:
        aspect = obs.load.aspect_shown()
        cands.append(lambda: _two_way_rule(c, style, rt["aspect"], ["load_rule", "tall_load_rule"], ("secure", "as_is"), aspect, 0.5, 0.2, "plain", "", "secure"))
    if obs.env.ramp_angle_deg is not None:
        angle = obs.env.ramp_angle_deg
        cands.append(lambda: _two_way_rule(c, style, rt["ramp"], ["ramp_rule", "slope_class"], ("steep", "normal"), angle, 0.5, 0.5, "angle", "", "steep"))
    if obs.place is not None:
        name, reading = obs.place.primary()
        step = 0.1 if obs.place.cls == "fine" else (0.5 if obs.place.cls == "medium" else 1.0)
        r = dict(rt["p4"])
        r["instructions"] = r[f"instructions_{name}"]
        cands.append(lambda: _two_way_rule(c, style, r, ["reading_rule", "tolerance_rule"], ("recheck", "accept"), abs(reading.value), step, step / 2, "mm", "", "recheck"))
    return cands


def _rule_question(c: _Ctx, names: list[str], instructions: str, classes: list[str], descriptions: dict, answer: str, style: str) -> Question:
    criteria, keymap = _choice(c, classes, [descriptions[k] for k in classes], style, "r")
    answer_key = next(k for k, v in keymap.items() if v == answer)
    return Question(c.name(*names), "choice", instructions, criteria, {"kind": "rule", "answer": answer_key, "options": list(criteria), "classes": keymap})


# ----------------------------------------------------------------------------------------------------------------


def candidates(cfg: SimgenConfig, obs: Observation, options: tuple[str, ...], rng: np.random.Generator, surface: Surface | None = None, style: str = "word") -> list:
    """추가 noul·score·상태 후보 전부(정체와 만드는 함수) — '한 가지 더 정하기' 교란(T6)이 쓰지 않은 것 하나를 고른다."""
    c = _Ctx(cfg, obs, rng, surface or Surface())
    return _extra_nouls(c, options) + _score_candidates(c, options) + _state_candidates(c, style)


def write_questions(cfg: SimgenConfig, obs: Observation, options: tuple[str, ...], rng: np.random.Generator, surface: Surface | None = None) -> QuestionSet:
    """요청 하나의 질문들(3~8개)."""
    surface = surface or Surface()
    c = _Ctx(cfg, obs, rng, surface)
    style = KEY_STYLES[int(rng.integers(len(KEY_STYLES)))]
    style = surface.key_style or style
    share = cfg.values["criterion_share"]
    criteria = _criteria_for_scene(obs)

    def criterion() -> str | None:
        return c.pick(criteria) if rng.random() < share else None

    primary = _action_question(c, options, style, "a", criterion())
    questions = [primary]
    if rng.random() < 0.25:
        second = _action_question(c, options, style, "b", criterion())
        if identity(second.recipe) != identity(primary.recipe):  # 같은 기준·같은 선택지면 같은 질문이다(m-c)
            questions.append(second)
    key = _key_noul(c)
    questions.append(key)
    used = {identity(q.recipe) for q in questions}

    def fresh(cands: list) -> list:
        return [cand for cand in cands if cand.ident not in used]

    def take(cand: _Cand) -> None:
        q = cand.build()
        assert identity(q.recipe) == cand.ident, (cand.ident, q.recipe)
        questions.append(q)
        used.add(cand.ident)

    extra = fresh(_extra_nouls(c, options))
    for i in rng.permutation(len(extra))[: int(rng.integers(0, 3))]:
        take(extra[int(i)])
    scores = fresh(_score_candidates(c, options))
    for i in rng.permutation(len(scores))[: int(rng.integers(1, 3))]:
        take(scores[int(i)])
    states = fresh(_state_candidates(c, style))
    if states and rng.random() < 0.5:
        take(states[int(rng.integers(len(states)))])
    if obs.scene in VOI_SCENES and "look_again" in options and rng.random() < 0.4:
        voi = _voi_question(c, options, style)
        if identity(voi.recipe) not in used:
            questions.append(voi)
            used.add(identity(voi.recipe))
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
