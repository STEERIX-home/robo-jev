"""관측 서술기 3종 (Task v0a 지시서 §3.8): 문체의 꼴, 보고된 수치가 글에 그대로 있는지, 권고·힌트·방해 문구, 길이."""

from dataclasses import replace

import numpy as np
import pytest
from simgen_helpers import moment, sample_records

from robo_jev.simgen import textfmt
from robo_jev.simgen.catalog import default_config
from robo_jev.simgen.narrate import RenderPlan, compute_hint, render_state
from robo_jev.simgen.phrases import option_short
from robo_jev.simgen.rollout import available_options
from robo_jev.simgen.skins import entry_text
from robo_jev.simgen.spec import Reading, stream

HINT_OPENINGS = ("Estimated stopping distance", "At their current pace", "At its current pace", "Free width", "Sideways acceleration",
                 "If the load's centre of mass sat at half its height", "On this slope, ", "If the weight of ", "Computed from the readings:",
                 "At your current speed you reach that spot", "Stopping distance with maximum braking",
                 # 한국어 힌트(phrases_ko_data.HINTS_KO의 고정된 첫머리)
                 "추정 정지 거리는", "현재 속도 그대로라면", "둘 다 각자 벽 쪽으로 붙을 때 남는 폭:", "이 회전의 횡가속도(지금 속도 기준):",
                 "짐의 무게중심이 높이의 절반에", "이 기울기에서는", "무게가 한가운데 있다고 치면", "측정값으로 계산하면", "현재 속도로 그 자리까지 약",
                 "최대 제동 시 정지 거리")  # fmt: skip


@pytest.fixture(scope="module")
def step_in_obs():
    return moment(41, "retail", scene="p1p5_step_in_carrying", body="humanoid", agent_kind="adult").obs


def _render(obs, style, recommendation=None, hint=None, seed=0):
    return render_state(obs, RenderPlan(style, available_options(obs), recommendation, hint), stream(seed, "render"))


def test_facts_style_is_an_opening_line_then_bullets(step_in_obs):
    lines = _render(step_in_obs, "facts").split("\n")
    assert not lines[0].startswith("- ") and lines[0].startswith(step_in_obs.skin.setting)
    assert len(lines) >= 5 and all(line.startswith("- ") for line in lines[1:])


def test_prose_style_is_paragraphs_without_bullets(step_in_obs):
    text = _render(step_in_obs, "prose")
    assert "\n- " not in text and not text.startswith("- ")
    assert text.count("\n\n") >= 1


def test_planner_style_has_sections_recommendation_and_every_available_action(step_in_obs):
    options = available_options(step_in_obs)
    text = _render(step_in_obs, "planner", recommendation="stop_moderate")
    for section in ("CONTEXT:", "TASK:", "ROBOT STATE", "PERCEIVED", "PLANNER RECOMMENDATION:", "AVAILABLE ACTIONS"):
        assert section in text
    tail = text.split("AVAILABLE ACTIONS\n", 1)[1].split("\n")
    assert [line.strip("- ").strip() for line in tail] == [option_short(o, step_in_obs) for o in options]
    assert f"PLANNER RECOMMENDATION: {option_short('stop_moderate', step_in_obs)}" in text


@pytest.mark.parametrize("style", ["facts", "prose", "planner"])
def test_reported_numbers_appear_exactly_as_formatted(step_in_obs, style):
    obs = step_in_obs
    text = _render(obs, style)
    a = obs.agent
    assert textfmt.distance(a.ahead.value, a.ahead.resolution, obs.fmt.distance_unit) in text
    assert textfmt.accel(obs.robot.decel_moderate) in text
    if a.speed is not None:
        assert textfmt.speed(a.speed.value, a.speed.resolution, obs.fmt.speed_unit) in text
    assert obs.container.phrase in text


def test_two_fix_mode_gives_two_positions_and_no_speed(step_in_obs):
    a = step_in_obs.agent
    two = replace(a, speed=None, track=(Reading(-0.62, 0.04, 0.01), Reading(-0.31, 0.04, 0.01)), track_ages=(1.0, 0.4))
    obs = replace(step_in_obs, agent=two, fmt=replace(step_in_obs.fmt, distance_unit="m", time_unit="s"))
    for style in ("facts", "prose", "planner"):
        text = _render(obs, style)
        assert "0.62 m right" in text and "0.31 m right" in text
        assert "1.00 s ago" in text and "0.40 s ago" in text
        assert f"toward your {a.heading} at" not in text  # 속도 보고("… toward your left at 0.9 m/s")가 없다
        assert f"toward your {a.heading}" in text  # 움직이는 쪽은 준다


def test_units_follow_the_observation_formatting(step_in_obs):
    obs = replace(step_in_obs, fmt=replace(step_in_obs.fmt, distance_unit="cm"))
    a = obs.agent
    text = _render(obs, "facts")
    assert textfmt.distance(a.ahead.value, a.ahead.resolution, "cm") in text and " cm" in text


def test_same_observation_and_seed_render_identically(step_in_obs):
    for style in ("facts", "prose", "planner"):
        assert _render(step_in_obs, style, "slow", "x") == _render(step_in_obs, style, "slow", "x")


@pytest.mark.parametrize(
    ("seed", "domain", "forced"),
    [
        (42, "warehouse", {"scene": "p1_crossing", "body": "mobile_base"}), (43, "hospital", {"scene": "p1_head_on"}),
        (44, "home", {"scene": "p5_turning", "body": "humanoid"}), (45, "cafe", {"scene": "p1_step_in"}),
        (46, "lab", {"scene": "p5_braking", "body": "arm"}), (47, "transit", {"scene": "p1p5_step_in_carrying"}),
    ],
)
def test_every_scene_has_a_computed_hint(seed, domain, forced):
    obs = moment(seed, domain, **forced).obs
    hint = compute_hint(obs, 0.25)
    assert hint and hint.startswith(HINT_OPENINGS)


def test_sample_texts_carry_recommendation_hint_and_distractors_as_flagged():
    cfg = default_config()
    for r in sample_records()[:400]:
        state, meta = r["request"]["state"], r["meta"]
        rec = meta["recommendation"]
        has_rec = any(m in state for m in ("PLANNER RECOMMENDATION:", "planner suggests", "planner's recommendation", "Planner advice",
                                            "계획기 권고:", "모션 계획기는", "계획기의 권고:", "계획기 조언:"))  # fmt: skip
        assert has_rec == rec["present"], r["id"]
        assert any(opening in state for opening in HINT_OPENINGS) == meta["hint"], r["id"]
        card = cfg.domain(meta["domain"])
        texts = [entry_text(d) for d in card["distractors"]]
        if meta["language"] == "ko":
            from robo_jev.simgen.korean import ko_skins

            texts = [ko_skins().t(t) for t in texts]
        assert sum(t in state for t in texts) == meta["distractors"], r["id"]


def test_state_lengths_are_mostly_400_to_1500_characters():
    """영어 글은 대부분 400~1,500자; 한국어는 같은 사실을 더 짧게 쓴다(한 글자가 더 많은 뜻을 담는다) — 대부분 150~1,300자."""
    for lang, (lo, hi, floor) in {"en": (400, 1500, 250), "ko": (150, 1300, 120)}.items():
        lengths = np.array([len(r["request"]["state"]) for r in sample_records() if r["meta"]["language"] == lang])
        assert len(lengths) > 200
        assert np.mean((lengths >= lo) & (lengths <= hi)) >= 0.9, lang
        assert lengths.min() >= floor and lengths.max() <= 2500, (lang, lengths.min(), lengths.max())
