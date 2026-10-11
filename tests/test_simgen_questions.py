"""질문 생성기 (Task v0a 지시서 §3.9, docs/12 §7): 종류와 수, 바꿔 쓰기 묶음, 규칙 적용 정답이 글에 보인 수와 맞는지."""

import re

import pytest
from simgen_helpers import moment

from robo_jev.simgen import phrases, textfmt
from robo_jev.simgen.catalog import default_config
from robo_jev.simgen.labels import rule_label
from robo_jev.simgen.questions import write_questions
from robo_jev.simgen.rollout import available_options
from robo_jev.simgen.spec import Reading, stream

ALL_OPTIONS = ("continue", "slow", "stop_gentle", "stop_moderate", "stop_hard", "sidestep_left", "sidestep_right", "retract", "go_now", "creep", "wait",
               "turn_100", "turn_80", "turn_60", "turn_45", "turn_30")  # fmt: skip

MOMENTS = [
    (31, "cafe", {"scene": "p1p5_step_in_carrying", "body": "mobile_base"}),
    (32, "factory", {"scene": "p1_step_in", "body": "arm"}),
    (33, "warehouse", {"scene": "p1_crossing", "body": "forklift_amr", "variant": "stopped"}),
    (34, "transit", {"scene": "p1_head_on", "body": "humanoid"}),
    (35, "lab", {"scene": "p5_braking", "body": "arm"}),
    (36, "greenhouse", {"scene": "p5_turning", "body": "mobile_base"}),
]


@pytest.mark.parametrize(("seed", "domain", "forced"), MOMENTS)
def test_option_descriptions_have_at_least_three_paraphrases(seed, domain, forced):
    obs = moment(seed, domain, **forced).obs
    for option in available_options(obs):
        variants = phrases.option_descriptions(option, obs)
        assert len(set(variants)) >= 3, option
        assert phrases.option_short(option, obs)


def test_instruction_banks_have_three_or_more_variants_per_family_and_a_heldout_pool():
    """학습 묶음은 셋 이상, 빼 둔 묶음(T2 전용)은 하나 이상이고 학습 묶음과 겹치지 않는다."""
    for key, entry in phrases.EN.items():
        assert len(set(entry["train"])) >= 3 or key.startswith("voi."), key
        assert entry["heldout"] and not set(entry["heldout"]) & set(entry["train"]), key


@pytest.mark.parametrize(("seed", "domain", "forced"), MOMENTS)
def test_every_question_set_has_the_required_kinds_and_size(seed, domain, forced):
    cfg = default_config()
    obs = moment(seed, domain, **forced).obs
    for i in range(40):
        qset = write_questions(cfg, obs, available_options(obs), stream(i, "questions"))
        kinds = [q.recipe["kind"] for q in qset.questions]
        assert 3 <= len(kinds) <= 8
        assert "action" in kinds and "noul" in kinds and "score" in kinds
        assert qset.key_question in {q.name for q in qset.questions}
        assert len({q.name for q in qset.questions}) == len(kinds)


def test_option_keys_follow_the_record_key_style():
    cfg = default_config()
    obs = moment(31, "cafe", scene="p1p5_step_in_carrying", body="mobile_base").obs
    seen = set()
    for i in range(30):
        qset = write_questions(cfg, obs, available_options(obs), stream(i, "questions"))
        action = next(q for q in qset.questions if q.recipe["kind"] == "action")
        keys = list(action.criteria)
        if qset.key_style == "code":
            assert all(re.fullmatch(r"[ab]\d", k) for k in keys)
        elif qset.key_style == "word":
            assert set(keys) <= set(ALL_OPTIONS)
        else:
            assert set(keys) <= set(phrases.VERB_KEYS.values()) | set(phrases.ARM_VERB_KEYS.values())
        seen.add(qset.key_style)
    assert seen == {"word", "code", "verb"}


def test_option_order_is_shuffled_across_records():
    cfg = default_config()
    obs = moment(31, "cafe", scene="p1p5_step_in_carrying", body="mobile_base").obs
    orders = set()
    for i in range(30):
        qset = write_questions(cfg, obs, available_options(obs), stream(i, "questions"))
        action = next(q for q in qset.questions if q.recipe["kind"] == "action")
        orders.add(tuple(action.recipe["options"].values()))
    assert len(orders) > 5


_QUANTITY = re.compile(r"(\d+(?:\.\d+)?)\s*(m/s|km/h|mm/s|mm|cm|m|ms|s|%)(?![\w/])")
_SCALE = {"m": 1.0, "cm": 0.01, "mm": 0.001, "m/s": 1.0, "km/h": 1 / 3.6, "mm/s": 0.001, "s": 1.0, "ms": 0.001, "%": 0.01}


def _quantities(text: str) -> list[float]:
    """글에서 단위 붙은 수를 차례로 읽어 SI로 바꾼다('250 ml'의 m은 단위로 읽지 않는다)."""
    return [float(v) * _SCALE[u] for v, u in _QUANTITY.findall(text)]


def _expected_class(obs, text: str) -> str:
    """규칙 문장(문턱은 글에서 다시 읽음)을 글에 보인 값에 적용한 갈래."""
    numbers = _quantities(text)
    a = obs.agent
    if "'far'" in text:
        if obs.scene in ("p1_step_in", "p1p5_step_in_carrying"):
            reading = a.ahead
        elif obs.scene == "p5_braking":
            reading = obs.env.obstacle_distance
        else:
            reading = a.track[-1]
        shown, hi, lo = textfmt.shown_si(reading, obs.fmt.distance_unit), numbers[0], numbers[1]
        return "far" if shown >= hi else ("near" if shown >= lo else "close")
    if "'stale'" in text:
        return "stale" if a.age > numbers[0] else "fresh"
    if "speed limit" in text:
        shown = textfmt.shown_si(Reading(obs.robot.speed, 0.0, 0.01 if obs.body == "arm" else 0.05), obs.fmt.speed_unit)
        return "over_limit" if shown > numbers[0] else "within_limit"
    if "filled above" in text:
        return "reduced" if obs.container.fill.value > numbers[0] else "normal"
    if "'single_file'" in text:
        return "single_file" if textfmt.shown_si(obs.env.corridor_width, obs.fmt.distance_unit) < numbers[0] else "two_way"
    if "'sharp'" in text:
        return "sharp" if obs.env.turn_radius < numbers[0] else "gentle"
    raise AssertionError(f"unknown rule: {text}")


@pytest.mark.parametrize(("seed", "domain", "forced"), MOMENTS)
def test_rule_answers_follow_the_numbers_as_shown(seed, domain, forced):
    """규칙 문장의 문턱을 글에서 다시 읽고 글에 보인 값에 적용하면 정답 갈래와 같다(확률 1)."""
    cfg = default_config()
    obs = moment(seed, domain, **forced).obs
    kinds = set()
    for i in range(300):
        for q in write_questions(cfg, obs, available_options(obs), stream(i, "questions")).questions:
            if q.recipe["kind"] != "rule":
                continue
            label = rule_label(q.recipe)
            assert label["probabilities"][q.recipe["answer"]] == 1.0 and sum(label["probabilities"].values()) == 1.0
            expected = _expected_class(obs, q.instructions)
            assert q.recipe["classes"][q.recipe["answer"]] == expected, (q.instructions, q.criteria)
            kinds.add(expected)
    assert len(kinds) >= 2
