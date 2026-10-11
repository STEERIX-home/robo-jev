"""simgen 갈라 굴리기 — 손 계산 사례 (docs/12 §2 ④, Task v0a 지시서 §3.3·§3.4, 리뷰 1 I1·I2·I3·질문 4).

굴리기는 정답과 참 결과가 함께 쓰므로 보정 시험(SBC)으로는 그 잘못이 보이지 않는다 — 그래서 손으로 푼 사례로 본다:

* 마주 옴: 줄이 겹치는 반응하는 상대는 멈춘 로봇 앞에서 선다(리뷰의 변이체 M4 '감속 안 함'이 여기서 떨어진다). 반응하지 않는 상대는 닿는다.
* 가로지름: 반응하는 상대는 자기 줄을 막은 로봇 앞에서 늦게라도 선다(리뷰 1 질문 4). 반응 확률은 단서에 묶인다(정신이 팔렸으면 낮다).
* 회전: 접근 구간 전체에 걸친 필요한 감속(편안함 하한 ~ 최대 감속) — 느린 회전이 가장 큰 가속을 줄인다(리뷰 1 I2). 지연은 손 계산과 같다.
* 감속(P5): 목표(장애물 앞, 몸별 허용치)보다 일찍 서면 남은 거리를 gentle로 다시 다가가는 시간 2·√(d/a) — 급제동은 공짜가 아니다(리뷰 1 I3).
  넘어서면 장애물에 닿는 속도를 잰다.
* 끼어듦: 처음 닿는 순간의 상대 속도(충돌 속도 비용 기제의 입력). 뚜껑은 쏟아짐을 막지만 '림까지 차오름'은 막지 않는다.
"""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pytest
from simgen_helpers import moment

from robo_jev.simgen import physics as ph
from robo_jev.simgen.catalog import default_config, reaction_probability
from robo_jev.simgen.posterior import sample_posterior
from robo_jev.simgen.rollout import rollout, turn_profile
from robo_jev.simgen.spec import Batch, Reading


@pytest.fixture(scope="module")
def cfg():
    return default_config()


def _batch(rows: list[dict]) -> Batch:
    keys = rows[0].keys()
    return Batch({k: np.array([row[k] for row in rows], dtype=float) for k in keys})


def _robot(obs, **changes):
    base = {"speed": 1.0, "cruise": 1.0, "latency": 0.2, "decel_gentle": 1.0, "decel_moderate": 1.75, "decel_hard": 3.0, "radius": 0.3}
    return replace(obs.robot, **{**base, **changes})


# ----------------------------------------------------------------------------------------------------------------
# 마주 옴 — 늦은 감속
# ----------------------------------------------------------------------------------------------------------------


def _head_on_case(cfg):
    obs = moment(13, "hospital", scene="p1_head_on", body="mobile_base", agent_kind="adult").obs
    env = replace(obs.env, robot_lateral=0.0, corridor_width=Reading(1.2, 0.02, 0.01))
    obs = replace(obs, robot=_robot(obs), env=env)
    row = {"dist": 4.0, "lat_a": 0.0, "width": 1.2, "va": 1.2, "ra": 0.25, "intent": 0.0, "aside_speed": 0.4, "keep_standoff": 0.5,
           "agent_decel": 2.0, "reactive": 1.0, "roll_decel": 0.0}  # fmt: skip
    return obs, row


def test_a_reactive_oncoming_agent_stops_short_of_a_stopped_robot(cfg):
    """같은 줄로 1.2 m/s로 오는 사람(의도 keep): 로봇이 최대로 멈추면(0.37 m에서 섬) 사람은 간격이 0.5 + 1.2²/(2·2) = 0.86 m일 때
    감속을 시작해 약 0.5 m 앞에서 선다 — 닿지 않고 서로 막힌다. 반응하지 않는 상대(대조)는 닿는다."""
    obs, row = _head_on_case(cfg)
    out = rollout(cfg, obs, _batch([row, {**row, "reactive": 0.0}]), ("stop_hard", "continue"))
    stop = out.index("stop_hard")
    assert out.events["contact"][stop].tolist() == [False, True]
    assert out.events["blocked"][stop].tolist() == [True, False]
    assert out.events["contact"][out.index("continue")].tolist() == [True, True]
    # 마지막 간격 ≈ keep_standoff (멈춘 로봇 앞)
    assert out.quantities["closeness"][stop][0] < 1.0


def test_an_oncoming_agent_moving_aside_lets_a_stopped_robot_be(cfg):
    obs, row = _head_on_case(cfg)
    aside = {**row, "intent": 1.0, "width": 2.0}  # 로봇의 왼쪽 벽으로 비킨다
    out = rollout(cfg, replace(obs, env=replace(obs.env, corridor_width=Reading(2.0, 0.02, 0.01))), _batch([aside]), ("stop_moderate",))
    assert not out.events["contact"][0, 0] and not out.events["blocked"][0, 0]


# ----------------------------------------------------------------------------------------------------------------
# 가로지름 — 반응하는 상대 (리뷰 1 질문 4)
# ----------------------------------------------------------------------------------------------------------------


def test_a_crossing_agent_that_reacts_brakes_for_a_robot_on_its_line(cfg):
    """로봇(1.0 m/s, 반지름 0.3)이 1.0 m 앞 교차선을 지나는 동안 사람(반지름 0.3, 진로 중심선에서 2.0 m, 1.2 m/s)이 다가온다.
    반응하지 않으면 t ≈ 1.25 s에 닿는다. 반응하면 로봇이 줄 위에 있을 때 남은 거리가 0.4 + 1.2²/(2·2) = 0.76 m가 되는 순간
    (t ≈ 0.53 s) 감속해 진로 앞에서 선다."""
    obs = moment(12, "warehouse", scene="p1_crossing", body="mobile_base", variant="moving", agent_kind="adult").obs
    obs = replace(obs, robot=_robot(obs))
    row = {"ahead": 1.0, "dist": 2.0, "from_sign": 1.0, "va": 1.2, "ra": 0.3, "intent": 0.0, "yield_standoff": 0.5, "agent_decel": 2.0,
           "hurry_factor": 1.5, "hurry_accel": 1.5, "roll_decel": 0.0, "reacts": 1.0, "react_standoff": 0.4, "distracted": 0.0, "static_u": 0.5}  # fmt: skip
    out = rollout(cfg, obs, _batch([row, {**row, "reacts": 0.0}]), ("continue",))
    assert out.events["contact"][0].tolist() == [False, True]
    assert out.impact["contact"][0][1] > 0.5  # 닿을 때의 상대 속도


def test_people_walk_around_a_robot_standing_still_unless_distracted_and_close(cfg):
    """수정 1 I3: '반응하지 않음'(reacts = 0)은 움직이는 로봇에 대한 것이다. 로봇이 최대로 서서(멈춤 거리 0.2 + 1/6 = 0.37 m) 교차선(0.5 m
    앞) 위에 서 있으면, 다가오는 사람(1.2 m/s)은 처음 마주칠 때의 주의·거리로 정한 확률로만 부딪힌다: 그때 남은 거리가 자기 반응 + 멈춤
    거리(1.2 × 0.5 + 1.2²/(2·2) = 0.96 m)보다 짧으면 '가깝다'(주의 0.05 / 흐트러짐 0.50), 아니면 0.01 / 0.05.
    로봇은 t ≈ 0.53 s부터 멈춘 물체다: 2.0 m에서 오던 사람은 그때 남은 거리 ≈ 0.76 m(가깝다), 4.0 m에서 오던 사람은 ≈ 2.76 m(멀다)."""
    obs = moment(12, "warehouse", scene="p1_crossing", body="mobile_base", variant="moving", agent_kind="adult").obs
    obs = replace(obs, robot=_robot(obs))
    base = {"ahead": 0.5, "dist": 2.0, "from_sign": 1.0, "va": 1.2, "ra": 0.3, "intent": 0.0, "yield_standoff": 0.5, "agent_decel": 2.0,
            "hurry_factor": 1.5, "hurry_accel": 1.5, "roll_decel": 0.0, "reacts": 0.0, "react_standoff": 0.4, "distracted": 0.0, "static_u": 0.5}  # fmt: skip
    rows = [
        base,                                                      # 주의, 가깝다: 0.5 ≥ 0.05 → 피한다
        {**base, "distracted": 1.0, "static_u": 0.3},              # 흐트러짐, 가깝다: 0.3 < 0.50 → 부딪힌다
        {**base, "distracted": 1.0, "static_u": 0.7},              # 흐트러짐, 가깝다: 0.7 ≥ 0.50 → 피한다
        {**base, "distracted": 1.0, "static_u": 0.3, "dist": 4.0},  # 흐트러짐, 멀다: 0.3 ≥ 0.05 → 피한다
        {**base, "static_u": 0.03},                                 # 주의, 가깝다: 0.03 < 0.05 → 부딪힌다
    ]
    out = rollout(cfg, obs, _batch(rows), ("stop_hard", "continue"))
    assert out.events["contact"][out.index("stop_hard")].tolist() == [False, True, False, False, True]
    speeds = out.aux["contact_robot_speed"][out.index("stop_hard")]
    assert np.all(speeds[[1, 4]] < 0.05)                           # 서 있는 로봇에 사람이 걸어와 닿는다
    # 움직이는 로봇(그대로 가기)은 반응하지 않는 사람에게 단서·거리와 무관하게 닿는다(대조)
    keep = rollout(cfg, obs, _batch([{**base, "ahead": 1.0}, {**base, "ahead": 1.0, "distracted": 1.0}]), ("continue",))
    assert keep.events["contact"][0].tolist() == [True, True]


def test_the_static_failure_probability_reads_the_catalog(cfg):
    """멈춘 로봇을 피하지 못할 확률은 속성 목록(static_avoidance)의 값이고 단서에 묶인다: 흐트러진 단서면 높고, 공은 언제나 피하지 못한다."""
    from robo_jev.simgen.catalog import distraction_probability
    from robo_jev.simgen.rollout import static_encounter

    spec = cfg.catalog["static_avoidance"]
    assert spec["fail"]["attentive"] < spec["fail"]["distracted"] < spec["fail_close"]["distracted"]
    assert spec["fail_close"]["attentive"] < spec["fail_close"]["distracted"]
    assert distraction_probability(cfg, "adult", "distracted") > distraction_probability(cfg, "adult", "steady") > distraction_probability(cfg, "adult", "glance")
    u = np.linspace(0.0, 1.0, 20001)
    k = len(u)
    for distracted in (False, True):
        for remaining, key in ((0.2, "fail_close"), (5.0, "fail")):
            decided, fail = static_encounter(cfg, np.ones(k, dtype=bool), np.full(k, remaining), np.full(k, 1.2), np.full(k, 2.0), np.full(k, distracted), u,
                                             np.zeros(k, dtype=bool), np.zeros(k, dtype=bool), np.zeros(k, dtype=bool))  # fmt: skip
            assert decided.all()
            assert fail.mean() == pytest.approx(spec[key]["distracted" if distracted else "attentive"], abs=1e-3)
    _, fail = static_encounter(cfg, np.ones(k, dtype=bool), np.full(k, 5.0), np.full(k, 1.2), np.full(k, 2.0), np.zeros(k, dtype=bool), u,
                               np.zeros(k, dtype=bool), np.zeros(k, dtype=bool), np.ones(k, dtype=bool))  # fmt: skip
    assert fail.all()


def test_reaction_probability_is_tied_to_the_cue(cfg):
    spec = cfg.catalog["crossing_reaction"]
    assert reaction_probability(cfg, "adult", "distracted") < reaction_probability(cfg, "adult", "steady") < reaction_probability(cfg, "adult", "glance")
    assert reaction_probability(cfg, "adult", "distracted") == pytest.approx(spec["base"]["person"] * spec["cue_factor"]["distracted"])
    assert reaction_probability(cfg, "ball", None) == 0.0
    assert all(reaction_probability(cfg, k, c) <= spec["max"] for k in ("adult", "forklift", "other_robot", "hand") for c in spec["cue_factor"])


def test_the_posterior_reaction_share_follows_the_cue(cfg):
    obs = moment(12, "warehouse", scene="p1_crossing", body="mobile_base", variant="moving", agent_kind="adult").obs
    for cue in ("distracted", "glance"):
        o = replace(obs, agent=replace(obs.agent, cue=cue))
        reacts = sample_posterior(cfg, o, 4000, np.random.default_rng(3))["reacts"]
        assert reacts.mean() == pytest.approx(reaction_probability(cfg, "adult", cue), abs=0.03)


# ----------------------------------------------------------------------------------------------------------------
# 회전 (리뷰 1 I2)
# ----------------------------------------------------------------------------------------------------------------


def _turn_obs(approach: float = 2.0):
    obs = moment(15, "home", scene="p5_turning", body="mobile_base").obs
    env = replace(obs.env, turn_radius=1.0, turn_angle_deg=90.0, turn_approach=Reading(approach, 0.0, 0.05))
    return replace(obs, robot=_robot(obs, speed=1.2, cruise=1.2, accel=0.8), env=env)


def test_turn_slowdown_brakes_at_the_rate_the_approach_needs(cfg):
    """v0 = 1.2 m/s, 접근 2.0 m, 반경 1.0 m, 90°: 60 %(0.72 m/s)로 돌려면 (1.44 − 0.5184)/(2·2) = 0.2304 m/s²로 줄인다 —
    가장 큰 가속은 회전의 0.518 m/s²(전속 1.44 m/s²의 1/3). 지연 = 2.083 + 2.182 + 0.6 − (2 + 1.571 + 0.576)/1.2 = 1.409 s."""
    obs = _turn_obs()
    delay, peak, hard = turn_profile(cfg, obs.robot, "turn_60", 1.0, 90.0, 2.0)
    assert peak == pytest.approx(0.5184, abs=1e-4) and not hard
    assert delay == pytest.approx(2.0833 + 2.1817 + 0.6 - (2.0 + math.pi / 2 + 0.576) / 1.2, abs=2e-3)
    full = turn_profile(cfg, obs.robot, "turn_100", 1.0, 90.0, 2.0)
    assert full == (0.0, pytest.approx(1.44), False)
    for option in ("turn_80", "turn_60", "turn_45", "turn_30"):
        assert turn_profile(cfg, obs.robot, option, 1.0, 90.0, 2.0)[1] < full[1], option


def test_turn_slowdown_has_a_comfort_floor_and_a_hard_ceiling(cfg):
    obs = _turn_obs()
    comfort = cfg.body("mobile_base")["comfort_decel"]
    # 아주 긴 접근: 필요한 감속이 하한보다 작으면 하한으로(늦게 시작해 회전 입구에서 끝낸다)
    v_entry, a_long = ph.turn_slowdown(1.2, 0.96, 10.0, comfort, 3.0)
    assert (v_entry, a_long) == (0.96, comfort)
    # 아주 짧은 접근: 최대 감속으로도 못 줄이면 더 빠르게 들어가고 급제동이다
    delay, peak, hard = turn_profile(cfg, obs.robot, "turn_30", 1.0, 90.0, 0.2)
    assert hard and peak == pytest.approx(3.0)
    assert ph.turn_slowdown(1.2, 0.36, 0.2, comfort, 3.0)[0] == pytest.approx(math.sqrt(1.44 - 2 * 3.0 * 0.2))


def test_slower_turns_lower_the_spill_risk_through_the_rollout(cfg):
    obs = _turn_obs()
    row = {"fill": 0.9, "k": 1.3, "tilt": 0.0, "lid": 0.0}
    obs = replace(obs, container=replace(obs.container, diameter=0.2, height=0.1))
    out = rollout(cfg, obs, _batch([row]))
    headroom = {o: out.quantities["headroom"][out.index(o)][0] for o in out.options}
    assert headroom["turn_60"] < headroom["turn_100"] and headroom["turn_45"] < headroom["turn_100"]
    assert out.hard_brake.tolist() == [False] * len(out.options)


# ----------------------------------------------------------------------------------------------------------------
# 감속 지연 (리뷰 1 I3)
# ----------------------------------------------------------------------------------------------------------------


def _braking(cfg, body: str, surface: float):
    obs = moment(14, "lab", scene="p5_braking", body=body).obs
    robot = _robot(obs)
    row = {"obstacle": surface + robot.radius, "fill": 0.5, "k": 1.0, "tilt": 0.0, "lid": 0.0}
    return rollout(cfg, replace(obs, robot=robot), _batch([row]))


def test_stopping_early_costs_the_time_to_creep_up_to_the_target(cfg):
    """1.0 m/s, 반응 0.2 s, 감속 1.0/1.75/3.0, 장애물 면까지 0.55 m, 이동 로봇 허용치 0.05 m:
    gentle 0.70 m → 넘어섬(닿는 속도 √(1 − 2·1·0.35) = 0.548); moderate 0.486 m → 남는 0.014 m를 다시 다가감 2·√(0.0143/1.0) = 0.239 s;
    hard 0.367 m → 0.133 m, 0.730 s. 급제동의 시간 비용이 단단한 제동보다 크다."""
    out = _braking(cfg, "mobile_base", 0.55)
    delay = {o: out.quantities["delay"][out.index(o)][0] for o in out.options}
    over = {o: bool(out.events["overrun"][out.index(o)][0]) for o in out.options}
    assert over == {"continue": True, "stop_gentle": True, "stop_moderate": False, "stop_hard": False}
    assert delay["stop_moderate"] == pytest.approx(2 * math.sqrt((0.55 - 0.05 - (0.2 + 1 / 3.5)) / 1.0), abs=1e-4)
    assert delay["stop_hard"] == pytest.approx(2 * math.sqrt((0.55 - 0.05 - (0.2 + 1 / 6)) / 1.0), abs=1e-4)
    assert delay["stop_hard"] > delay["stop_moderate"] > 0.0
    impact = out.impact["overrun"]
    assert impact[out.index("stop_gentle")][0] == pytest.approx(math.sqrt(1.0 - 2 * 1.0 * (0.55 - 0.2)), abs=1e-6)
    assert impact[out.index("continue")][0] == pytest.approx(1.0)
    assert out.hard_brake.tolist() == [o == "stop_hard" for o in out.options]


def test_the_stop_tolerance_is_body_scaled(cfg):
    """같은 기하에서 팔의 허용치(0.01 m)는 이동 로봇(0.05 m)보다 작아 같은 조기 정지가 더 큰 지연이 된다; 허용치 안에서 서면 지연 0."""
    arm = _braking(cfg, "arm", 0.55)
    base = _braking(cfg, "mobile_base", 0.55)
    i = arm.index("stop_hard")
    assert arm.quantities["delay"][i][0] > base.quantities["delay"][base.index("stop_hard")][0]
    snug = _braking(cfg, "mobile_base", 0.2 + 1 / 3.5 + 0.03)  # moderate가 허용치 안에서 선다
    assert snug.quantities["delay"][snug.index("stop_moderate")][0] == 0.0


# ----------------------------------------------------------------------------------------------------------------
# 끼어듦의 충돌 속도, 뚜껑
# ----------------------------------------------------------------------------------------------------------------


def test_step_in_contact_reports_the_relative_speed_at_first_contact(cfg):
    obs = moment(16, "sidewalk", scene="p1_step_in", body="mobile_base", agent_kind="adult").obs
    obs = replace(obs, robot=_robot(obs))
    row = {"gap": 1.5, "lat0": 0.0, "dir": 1.0, "u0": 0.0, "ra": 0.25, "intent": 0.0, "stop_decel": 2.0, "stay_s": 6.0, "cross_speed": 1.0,
           "cross_accel": 1.0, "back_decel": 2.0, "back_speed": 0.5, "roll_decel": 0.0, "side_left": 0.0, "side_right": 0.0}  # fmt: skip
    out = rollout(cfg, obs, _batch([row]), ("continue", "stop_moderate", "slow"))
    assert out.events["contact"][:, 0].tolist() == [True, False, True]
    assert out.impact["contact"][0][0] == pytest.approx(1.0, abs=1e-6)  # 등속 1.0 m/s로 서 있는 사람에게
    v_slow = out.impact["contact"][2][0]
    assert 0.5 - 1e-6 <= v_slow < 1.0  # 느려지는 중에 닿는다


def test_a_lid_stops_the_spill_but_not_the_liquid_reaching_the_rim(cfg):
    obs = moment(14, "lab", scene="p5_braking", body="mobile_base").obs
    robot = _robot(obs)
    row = {"obstacle": 3.0, "fill": 0.995, "k": 1.3, "tilt": 0.0, "lid": 1.0}
    out = rollout(cfg, replace(obs, robot=robot), _batch([row, {**row, "lid": 0.0}]))
    i = out.index("stop_hard")
    assert out.events["rim_reached"][i].tolist() == [True, True]
    assert out.events["spill"][i].tolist() == [False, True]
