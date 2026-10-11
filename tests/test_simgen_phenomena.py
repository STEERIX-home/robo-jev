"""새 현상 P3·P4·P6과 '다시 볼까'(정보의 가치)의 물리·사후·닫힌 꼴·단조성 (Task v0b 지시서 §1·§8).

* P3: 쏠림 비(경사 포함), 넘어짐·미끄러짐 판정, 미끄럼 거리(직선·회전) 손 계산; 감속·출발·회전·경사로 갈래에서 더 센 움직임의 넘어짐이
  약한 움직임의 넘어짐을 포함한다(세계마다); 쌓기의 닫힌 꼴(넘어짐·무게 쏠림)이 큰 표본과 맞다.
* P4: 정확한 베이즈(상태 사전 × 측정 우도)를 손으로 계산한 것과 같다; 오래된 카메라의 바뀜 확률 q̄ = 1 − E_λ[e^{−λa}]; 행동 결과.
* P6: 아직 있음 E_λ[e^{−λa}], 출현 1 − E_λ[e^{−3λ}], 센서 불일치의 베이즈, 모퉁이 '보이는 거리 안에 서나'의 닫힌 꼴이 표본과 맞다.
* 다시 볼까: 이산 신호의 닫힌 꼴이 손 계산(전수)과 같고, 연속 신호의 준난수 적분이 큰 표본과 맞고, 정보는 해가 되지 않는다(보는 시간을 빼면
  본 뒤의 기대 비용 ≤ 지금 최선).
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from simgen_helpers import moment

from robo_jev.simgen import p3, p4, p6
from robo_jev.simgen import physics as ph
from robo_jev.simgen.catalog import default_config
from robo_jev.simgen.labels import wmean
from robo_jev.simgen.posterior import exact_labels, sample_posterior
from robo_jev.simgen.rollout import available_options, rollout
from robo_jev.simgen.spec import Batch, stream
from robo_jev.simgen.voi import DiscreteSignal, GaussianSignal

G = ph.G


@pytest.fixture(scope="module")
def cfg():
    return default_config()


# ----------------------------------------------------------------------------------------------------------------
# P3 물리
# ----------------------------------------------------------------------------------------------------------------


def test_incline_ratio_is_sine_plus_accel_over_g_divided_by_cosine():
    assert float(ph.incline_ratio(0.0, 0.0)) == 0.0
    assert float(ph.incline_ratio(0.2 * G, 0.0)) == pytest.approx(0.2)
    phi = math.radians(6.0)
    assert float(ph.incline_ratio(0.0, phi)) == pytest.approx(math.tan(phi))
    assert float(ph.incline_ratio(1.5, phi)) == pytest.approx((math.sin(phi) + 1.5 / G) / math.cos(phi))


def test_tip_and_slide_hand_cases():
    # 넘어짐: min(r, μ)·h > b — 마찰이 버티면 r까지 기울인다
    tip, slide = ph.tip_slide(0.3, 0.5, 0.05, 0.2)
    assert bool(tip) and not bool(slide)
    # 마찰이 먼저 진다(μ < r): 쏠림이 μ에서 멎어 넘어지지 않고(0.2·0.2 = 0.04 < 0.05) 미끄러진다
    tip, slide = ph.tip_slide(0.3, 0.2, 0.05, 0.2)
    assert not bool(tip) and bool(slide)
    tip, slide = ph.tip_slide(0.1, 0.5, 0.05, 0.2)
    assert not bool(tip) and not bool(slide)


def test_slide_distance_hand_cases():
    a_obj = 0.2 * G
    assert ph.slide_distance(1.0, 3.0, 0.2) == pytest.approx(0.5 * (1.0 / a_obj - 1.0 / 3.0))
    assert ph.slide_distance(1.0, 1.5, 0.2) == 0.0                      # 받침 가속 ≤ 마찰 가속: 따라간다
    assert ph.slide_distance(1.0, 2.0, 0.05, math.radians(10.0)) == math.inf   # tan φ > μ: 경사에서 멈추지 못한다
    excess = 3.0 - 0.2 * G
    assert ph.turn_slide_distance(3.0, 0.2, 1.0) == pytest.approx(0.5 * excess + excess**2 / (2 * 0.2 * G))
    assert ph.turn_slide_distance(1.5, 0.2, 1.0) == 0.0


def _p3_out(cfg, seed, domain, **forced):
    m = moment(seed, domain, **forced)
    obs = m.obs
    out = rollout(cfg, obs, sample_posterior(cfg, obs, 512, stream(seed, "posterior")), available_options(obs))
    return obs, out


@pytest.mark.parametrize(
    ("seed", "domain", "forced", "order"),
    [
        (31, "cafe", {"scene": "p3_carrying_tall", "maneuver": "brake", "body": "mobile_base"}, ("stop_gentle", "stop_moderate", "stop_hard")),
        (32, "warehouse", {"scene": "p3_carrying_tall", "maneuver": "start", "body": "mobile_base"}, ("start_gentle", "start_normal", "start_quick")),
        (33, "home", {"scene": "p3_carrying_tall", "maneuver": "turn", "body": "mobile_base"}, ("turn_30", "turn_45", "turn_60", "turn_80", "turn_100")),
        (34, "warehouse", {"scene": "p3_ramp", "body": "forklift_amr"}, ("ramp_slow", "ramp_normal")),
    ],
)
def test_a_stronger_motion_tips_every_load_that_a_weaker_one_tips(cfg, seed, domain, forced, order):
    """세계마다: 쏠림 비가 큰 움직임의 넘어짐 ⊇ 작은 움직임의 넘어짐(같은 사후 세계) — 그래서 넘어짐 확률도 단조."""
    obs, out = _p3_out(cfg, seed, domain, **forced)
    tips = [out.events["tip"][out.index(o)] for o in order if o in out.options]
    assert len(tips) >= 2
    for weaker, stronger in zip(tips, tips[1:], strict=False):
        assert np.all(~weaker | stronger)
    if "secure_first" in out.options:
        assert not out.events["fall"][out.index("secure_first")].any()


def test_motion_only_load_events_never_count_the_collision(cfg):
    """수정 1 I2: 넘어짐·미끄러짐·움직임으로 떨어짐(fall_motion)은 받침의 움직임만의 결과다 — 장애물을 아주 가까이 옮겨 모든 멈춤이 넘어서
    부딪히게 해도 그대로이고, 충돌까지 세는 떨어짐(fall)만 늘어난다(fall = fall_motion ∪ 넘어섬)."""
    obs, out = _p3_out(cfg, 35, "cafe", scene="p3_carrying_tall", maneuver="brake", body="mobile_base")
    batch = sample_posterior(cfg, obs, 512, stream(35, "posterior"))
    near = Batch({**batch.values, "obstacle": np.full(batch.size, obs.robot.radius + 0.02)})
    out_near = rollout(cfg, obs, near, available_options(obs))
    for option in ("stop_gentle", "stop_moderate", "stop_hard"):
        i = out.index(option)
        assert out_near.events["overrun"][i].all()
        for name in ("tip", "slide", "fall_motion"):
            assert np.array_equal(out_near.events[name][i], out.events[name][i]), (option, name)
        assert np.array_equal(out_near.quantities["slide_frac"][i], out.quantities["slide_frac"][i])
        assert out_near.events["fall"][i].all()
        assert np.array_equal(out.events["fall"][i], out.events["fall_motion"][i] | out.events["overrun"][i])


def test_the_braking_key_asks_about_the_load_under_the_hardest_stop_not_the_overrun(cfg):
    """수정 1 I2: P3·P3+P5 감속의 핵심 질문 = 최대로 제동하면 움직임만으로 넘어지거나 미끄러져 떨어지나(결정 여유의 기준 움직임과 같다) —
    그 정답은 멈출 목표까지의 거리(넘어섬 여유)와 무관하다."""
    from robo_jev.simgen.generate import assemble, build_moment
    from robo_jev.simgen.p3_world import reference_ratio

    for seed, scene in ((41, "p3_carrying_tall"), (42, "p3p5_carrying_mixed")):
        m = build_moment(cfg, seed, "cafe", {"scene": scene, "maneuver": "brake", "body": "mobile_base"})
        record = assemble(cfg, m, 0)
        basis = record["meta"]["basis"][record["meta"]["key_question"]]
        assert (basis["event"], basis["option"]) == ("fall_motion", "stop_hard")
        world = m.world
        assert reference_ratio(world.robot, scene, "brake", world.geometry, world.parts) == pytest.approx(world.robot.decel_hard / 9.81, rel=1e-3)
        obs = m.obs
        batch = sample_posterior(cfg, obs, 256, stream(seed, "posterior"))
        options = available_options(obs)
        far = rollout(cfg, obs, Batch({**batch.values, "obstacle": batch["obstacle"] + 5.0}), options)
        near = rollout(cfg, obs, Batch({**batch.values, "obstacle": np.full(batch.size, obs.robot.radius + 0.02)}), options)
        i = far.index("stop_hard")
        assert np.array_equal(far.events["fall_motion"][i], near.events["fall_motion"][i])


def test_fall_contains_tip_and_slides_off_the_edge(cfg):
    obs, out = _p3_out(cfg, 35, "cafe", scene="p3_carrying_tall", maneuver="brake", body="mobile_base")
    fall, tip, slide = out.events["fall"], out.events["tip"], out.events["slide"]
    assert np.all(~tip | fall)
    off = out.quantities["slide_frac"] >= 1.0
    assert np.all(~(slide & off) | fall)


@pytest.mark.parametrize(("seed", "domain", "body"), [(41, "warehouse", "humanoid"), (42, "factory", "arm"), (43, "retail", "forklift_amr")])
def test_stack_closed_forms_match_a_large_sample(cfg, seed, domain, body):
    m = moment(seed, domain, scene="p3_place_stack", body=body)
    obs = m.obs
    exact = p3.exact(cfg, obs)
    out = rollout(cfg, obs, sample_posterior(cfg, obs, 40_000, stream(seed, "big")), ("place_now", "recentre", "place_elsewhere"))
    assert float(out.events["topple"][out.index("place_now")].mean()) == pytest.approx(exact["topple|place_now"], abs=0.012)
    assert float(out.events["topple"][out.index("recentre")].mean()) == pytest.approx(exact["topple|recentre"], abs=0.012)
    if "weight_toward" in exact:
        assert float(out.states["weight_toward"].mean()) == pytest.approx(exact["weight_toward"][1], abs=0.012)


@pytest.mark.parametrize(("seed", "domain", "body"), [(44, "warehouse", "humanoid"), (45, "factory", "arm"), (46, "retail", "forklift_amr")])
def test_a_stack_record_prices_its_actions_with_the_closed_form_topple_probability(cfg, seed, domain, body):
    """수정 1 M5 (리뷰: noul은 닫힌 꼴, 행동의 기대 비용은 K개 빈도 — 쌓기 행동의 argmax가 2 % 바뀜): 레코드의 모든 정답은 같은 무게의 같은
    표본에서 나오고, 그 무게에서 넘어짐 빈도가 닫힌 꼴과 같다 — 그래서 행동의 기대 비용도 닫힌 꼴의 확률을 쓴다."""
    from robo_jev.simgen.generate import assemble, build_moment
    from robo_jev.simgen.labels import calibrate, cost_model, expected_costs

    m = build_moment(cfg, seed, domain, {"scene": "p3_place_stack", "body": body})
    record = assemble(cfg, m, 0)
    obs = m.obs
    out = rollout(cfg, obs, sample_posterior(cfg, obs, cfg.catalog["posterior_k"], stream(m.seed, "posterior")), available_options(obs))
    exact = exact_labels(cfg, obs)
    assert calibrate(out, exact) == record["meta"]["calibration"]
    for option in ("place_now", "recentre"):
        slack = record["meta"]["calibration"]["trimmed"].get(f"topple|{option}", 0.0) + 1e-9
        assert abs(float(wmean(out.events["topple"][out.index(option)], out)) - exact[f"topple|{option}"]) <= slack
    for name, criterion in record["meta"]["criteria"].items():
        ec = expected_costs(out, cost_model(cfg.values, criterion, out.body, out.loaded))
        for key, canon in record["meta"]["options"][name].items():
            assert record["labels"][name]["expected_cost"][key] == round(float(ec[out.index(canon)]), 4), (name, key)


# ----------------------------------------------------------------------------------------------------------------
# P4
# ----------------------------------------------------------------------------------------------------------------


def _norm_pdf(x, mean, sd):
    return math.exp(-0.5 * ((x - mean) / sd) ** 2) / (sd * math.sqrt(2 * math.pi))


@pytest.mark.parametrize(("seed", "domain", "scene"), [(51, "lab", "p4_place_check"), (52, "factory", "p4_insert"), (53, "hospital", "p4_place_check")])
def test_p4_posterior_is_the_hand_computed_bayes_rule(cfg, seed, domain, scene):
    obs = moment(seed, domain, scene=scene).obs
    place = obs.place
    # 사전·상태 모형을 코드가 아니라 속성 목록에서 바로 읽는다(셋째 상태는 ± 둘로 반씩, 부호 있는 측정은 −쪽 평균의 부호를 뒤집는다)
    cls = cfg.catalog["p4"][place.family]["classes"][place.cls]
    names = p4.STATES[place.family]
    prior = [cls["prior"][names[0]], cls["prior"][names[1]], cls["prior"][names[2]] / 2, cls["prior"][names[2]] / 2]
    post = np.zeros(4)
    for s in range(4):
        like = prior[s] / sum(prior)
        for name, r in place.readings.items():
            mu, sd = cls[name][names[min(s, 2)]]
            if s == 3 and name == p4.SIGNED[place.family]:
                mu = -mu
            like *= _norm_pdf(r.value, mu, math.sqrt(sd**2 + r.sigma_eff**2))
        post[s] = like
    post /= post.sum()
    exact = p4.exact(cfg, obs)
    assert exact["ok_now"] == pytest.approx(post[0], abs=1e-9)
    assert exact["p4_state"] == pytest.approx([post[0], post[1], post[2] + post[3]], abs=1e-9)


def test_stale_camera_change_probability_is_the_mean_survival_over_the_hazard_range(cfg):
    lo, hi = cfg.catalog["p4"]["stale"]["hazard_per_s"]
    for age in (0.5, 1.7, 3.9):
        lam = np.linspace(lo, hi, 20001)
        assert p4.stale_q(cfg, age) == pytest.approx(1.0 - float(np.mean(np.exp(-lam * age))), abs=2e-5)
    assert p4.stale_q(cfg, 0.0) == 0.0


def test_p4_action_outcomes_follow_the_catalog(cfg):
    sub = np.array([0, 1, 2, 3])
    keep = p4.action_outcomes(cfg, "place", "continue", sub)
    assert keep["ok"].tolist() == [1.0, 0.0, 0.0, 0.0]
    human = p4.action_outcomes(cfg, "place", "call_human", sub)
    assert human["ok"].tolist() == [1.0] * 4 and human["human"].tolist() == [1.0] * 4
    for option in ("press_down", "reseat"):
        res = p4.action_outcomes(cfg, "place", option, sub)
        assert np.allclose(res["ok"] + res["damage"] + res["unseated"], 1.0)


# ----------------------------------------------------------------------------------------------------------------
# P6
# ----------------------------------------------------------------------------------------------------------------


def test_stay_and_emerge_probabilities_are_means_over_the_rate_range(cfg):
    lo, hi = cfg.catalog["p6"]["stale"]["leave_rate"]["adult"]
    lam = np.linspace(lo, hi, 20001)
    for age in (0.8, 2.5, 5.5):
        assert p6.stay_probability(cfg, "adult", age) == pytest.approx(float(np.mean(np.exp(-lam * age))), abs=2e-5)
    lo, hi = cfg.domain("sidewalk")["emerge_rate"]
    lam = np.linspace(lo, hi, 20001)
    assert p6.emerge_probability(cfg, "sidewalk", 3.0) == pytest.approx(1.0 - float(np.mean(np.exp(-3.0 * lam))), abs=2e-5)


def test_sensor_conflict_posterior_is_bayes_over_the_cue_table(cfg):
    table = cfg.catalog["p6"]["conflict"]["cues"]["glass"]
    prior = cfg.catalog["p6"]["conflict"]["prior_present"]
    with_obj = prior * table["lidar"][0] * (1 - table["camera"][0])
    without = (1 - prior) * table["lidar"][1] * (1 - table["camera"][1])
    assert p6.conflict_posterior(cfg, "glass", True, False) == pytest.approx(with_obj / (with_obj + without))
    with_obj = prior * (1 - table["lidar"][0]) * table["camera"][0]
    without = (1 - prior) * (1 - table["lidar"][1]) * table["camera"][1]
    assert p6.conflict_posterior(cfg, "glass", False, True) == pytest.approx(with_obj / (with_obj + without))


@pytest.mark.parametrize(("seed", "domain", "scene", "event"), [
    (61, "hospital", "p6_stale_track", "present"), (62, "sidewalk", "p6_blind_corner", "stops_before_corner"),
    (63, "warehouse", "p6_sensor_conflict", "present"), (64, "retail", "p1p6_emerging_step_in", "follower"),
    (65, "sidewalk", "p6_blind_corner", "emerge_3s"),
])  # fmt: skip
def test_p6_closed_forms_match_the_weighted_posterior(cfg, seed, domain, scene, event):
    obs = moment(seed, domain, scene=scene).obs
    exact = exact_labels(cfg, obs)
    out = rollout(cfg, obs, sample_posterior(cfg, obs, 20_000, stream(seed, "big")), available_options(obs))
    assert event in exact
    assert float(wmean(out.world_events[event], out)) == pytest.approx(exact[event], abs=0.012)


def _corner_case(cfg, ahead: float):
    """사각 모퉁이 손 계산: 로봇 1.0 m/s(반응 0.2 s, 최대 3.0 m/s²), 반지름 0.3; 사람(반지름 0.3, 1.4 m/s)이 가림 끝에서 1.0 m 떨어져 걷다가
    t_e에 보인다 — 반응하지 않고(reacts 0), 주의가 흐트러졌고, 멈춘 로봇 앞 실패의 균등 난수 static_u."""
    from dataclasses import replace

    from robo_jev.simgen.spec import Reading

    obs = moment(62, "sidewalk", scene="p6_blind_corner").obs
    robot = replace(obs.robot, speed=1.0, cruise=1.0, latency=0.2, decel_gentle=1.0, decel_moderate=1.75, decel_hard=3.0, radius=0.3)
    obs = replace(obs, robot=robot, corner=replace(obs.corner, ahead=Reading(ahead, 0.0, 0.01), lateral=Reading(1.0, 0.0, 0.01)))
    names = list(cfg.domain(obs.domain)["agents"]["crossing"])
    person = next(i for i, n in enumerate(names) if cfg.agent(n)["group"] == "person")
    row = {"lam": 0.2, "t_e": 1.0, "u": 1.4, "ra": 0.3, "agent_decel": 2.0, "reacts": 0.0, "react_standoff": 0.4, "ahead": ahead, "lateral": 1.0,
           "kind": float(person), "distracted": 1.0, "static_u": 0.0, "emerges": 1.0}  # fmt: skip
    return obs, row


def test_stopping_before_the_corner_never_meets_someone_who_steps_out(cfg):
    """수정 1 I3: '모퉁이 앞에서 멈추기'는 보고된 모퉁이까지 거리(반응 거리·여유 5 cm를 뺀 만큼) 안에 선다 — 앞면이 사람이 나오는 길에
    들어가지 않으므로, 나온 사람이 반응하지 않고 주의가 흐트러져 멈춘 로봇에도 걸어 들어오는 세계(static_u = 0)에서도 닿지 않는다.
    대조: 같은 세계에서 모퉁이 안에 서는 최대 제동(멈춤 거리 0.37 m > 모퉁이까지 0.25 m)은 닿는다."""
    from dataclasses import replace as replace_robot

    from robo_jev.simgen.p6 import stop_before_decel

    obs, row = _corner_case(cfg, 2.0)
    rows = [row, {**row, "t_e": 0.2}, {**row, "t_e": 2.5}, {**row, "distracted": 0.0, "static_u": 0.9}]
    out = rollout(cfg, obs, Batch({k: np.array([r[k] for r in rows], dtype=float) for k in row}), ("stop_before", "continue"))
    assert not out.events["contact"][out.index("stop_before")].any()
    # 필요한 감속 1.0²/(2·(2.0 − 0.2 − 0.05)) = 0.29 m/s² < 부드러운 감속 → 부드럽게(더 일찍 선다); 1.2 m에서는 1/(2·0.95) = 0.53 → 그대로
    assert stop_before_decel(obs.robot, 2.0) == pytest.approx(obs.robot.decel_gentle)
    assert stop_before_decel(replace_robot(obs.robot, decel_gentle=0.3), 1.2) == pytest.approx(1.0 / (2.0 * 0.95))
    obs2, row2 = _corner_case(cfg, 0.25)
    out2 = rollout(cfg, obs2, Batch({k: np.array([v], dtype=float) for k, v in row2.items()}), ("stop_before", "stop_hard"))
    assert stop_before_decel(obs2.robot, 0.25) == pytest.approx(obs2.robot.decel_hard)   # 최대로도 모자라면 최대로(모퉁이 안에 선다)
    assert out2.events["contact"][out2.index("stop_hard")].all()


def test_someone_stepping_out_walks_around_a_robot_standing_in_the_walkway_unless_distracted_and_close(cfg):
    """수정 1 I3: 로봇이 최대로 서서(0.37 m) 모퉁이 0.25 m 안쪽의 길에 서 있을 때, 1 s 뒤 나온 사람은 남은 거리 1.6 − 0.6 = 1.0 m에서 그
    로봇을 처음 본다 — 반응 + 멈춤 거리 1.4 × 0.5 + 1.4²/4 = 1.19 m보다 짧아 '가깝다': 주의면 0.05, 흐트러지면 0.50의 확률로만 부딪힌다
    (25 %의 '반응하지 않음'은 움직이는 로봇에만)."""
    obs, row = _corner_case(cfg, 0.25)
    rows = [{**row, "distracted": 0.0, "static_u": 0.5}, {**row, "distracted": 1.0, "static_u": 0.3}, {**row, "distracted": 1.0, "static_u": 0.7},
            {**row, "distracted": 0.0, "static_u": 0.03}]  # fmt: skip
    out = rollout(cfg, obs, Batch({k: np.array([r[k] for r in rows], dtype=float) for k in row}), ("stop_hard",))
    assert out.events["contact"][0].tolist() == [False, True, False, True]
    assert np.all(out.aux["contact_robot_speed"][0][[1, 3]] < 0.05)


def test_a_stale_spot_cannot_be_hit_once_its_occupant_has_left_or_was_never_there(cfg):
    obs = moment(66, "warehouse", scene="p6_stale_track").obs
    out = rollout(cfg, obs, sample_posterior(cfg, obs, 2000, stream(66, "p")), available_options(obs))
    gone = out.world_events["present"] < 0.5
    assert not out.events["contact"][:, gone].any()


# ----------------------------------------------------------------------------------------------------------------
# 다시 볼까(정보의 가치)
# ----------------------------------------------------------------------------------------------------------------


def test_discrete_signal_matches_the_brute_force_decision_and_never_hurts():
    rng = np.random.default_rng(3)
    for _ in range(50):
        costs = rng.uniform(0, 10, (3, 6))
        weights = rng.dirichlet(np.ones(6))
        lik = rng.dirichlet(np.ones(2), size=6).T            # (S = 2, K = 6), 열 합 1
        sig = DiscreteSignal(("a", "b", "c"), 2.0, lik)
        decision = sig.decide(costs, weights)
        for s in range(2):
            brute = int(np.argmin([np.sum(weights * lik[s] * costs[b]) for b in range(3)]))
            assert decision[s] == brute
        after = sig.after(costs, decision)
        expected_after = sum(weights[k] * sum(lik[s, k] * costs[decision[s], k] for s in range(2)) for k in range(6))
        assert float(np.sum(weights * after)) == pytest.approx(expected_after)
        assert float(np.sum(weights * after)) <= float(np.min(costs @ weights)) + 1e-9    # 정보는 해가 되지 않는다


def test_gaussian_signal_quadrature_matches_monte_carlo():
    rng = np.random.default_rng(4)
    means = rng.normal(0, 1, (5, 1))
    sds = rng.uniform(0.3, 1.0, (5, 1))
    costs = rng.uniform(0, 10, (3, 5))
    weights = rng.dirichlet(np.ones(5))
    sig = GaussianSignal(("a", "b", "c"), 1.0, means, sds)
    rule = sig.decide(costs, weights)
    after = sig.after(costs, rule)
    for k in range(5):
        draws = means[k] + sds[k] * rng.standard_normal((200_000, 1))
        mc = float(np.mean(costs[rule(draws), k]))
        assert after[k] == pytest.approx(mc, abs=0.04)
    assert float(np.sum(weights * after)) <= float(np.min(costs @ weights)) + 1e-9


@pytest.mark.parametrize(("seed", "domain", "scene"), [(71, "lab", "p4_place_check"), (72, "sidewalk", "p6_blind_corner"), (73, "warehouse", "p3_place_stack")])
def test_look_again_label_is_look_time_plus_the_best_after_looking(cfg, seed, domain, scene):
    from robo_jev.simgen.labels import cost_model, expected_costs, world_costs

    obs = moment(seed, domain, scene=scene).obs
    out = rollout(cfg, obs, sample_posterior(cfg, obs, 256, stream(seed, "posterior")), available_options(obs))
    assert out.voi is not None
    model = cost_model(cfg.values, None, out.body, out.loaded)
    look = out.voi.option
    ec = expected_costs(out, model)
    base_best = min(float(ec[out.index(o)]) for o in out.voi.base)
    # 다시 보기 = 보는 시간의 지연 비용 + 본 뒤의 기대 비용; 본 뒤의 기대 비용은 지금 최선보다 크지 않다
    look_cost = float(ec[out.index(look)])
    delay_cost = model["costs"]["delay_per_s"] * out.voi.time
    assert look_cost - delay_cost <= base_best + 1e-6
    per_world = world_costs(out, model)
    assert float(wmean(per_world[out.index(look)], out)) == pytest.approx(look_cost, abs=1e-6)


def _log_binorm(x1, x2, mu, sd2, se2):
    """log N2((x1, x2); (mu, mu), [[sd2 + se2, sd2], [sd2, sd2 + se2]]) — 같은 참값 x를 공유하는 두 보고."""
    a = sd2 + se2
    det = a * a - sd2 * sd2
    d1, d2 = x1 - mu, x2 - mu
    return -0.5 * (a * d1 * d1 - 2 * sd2 * d1 * d2 + a * d2 * d2) / det - 0.5 * np.log((2 * math.pi) ** 2 * det)


@pytest.mark.parametrize(("seed", "domain", "scene"), [(71, "lab", "p4_place_check"), (74, "factory", "p4_insert"), (75, "warehouse", "p4_place_check")])
def test_p4_look_again_cost_matches_an_independent_simulation_of_the_remeasurement(cfg, seed, domain, scene):
    """수정 1 M3 (리뷰의 MR4 — 다시 재기의 예측 분포에서 사후 불확실성을 빼도(sds = σ_e) 떨어지는 시험이 없었다): 다시 보기의 기대 비용을
    독립 모의로 — 세계 ~ 사후 무게, 참 x | 상태·지금 보고 ~ 조건부 정규, 새 보고 m' = x + 잡음, 본 뒤의 결정은 (m, m')의 결합 우도(같은 x를
    공유하는 이변량 정규) × 사전으로(코드의 예측 분포 N(사후 평균, 사후 분산 + σ_e²)를 쓰지 않는다). 코드의 값과 4 SE + 0.01 안에서 같다."""
    from robo_jev.simgen.labels import cost_model, expected_costs, world_costs

    obs = moment(seed, domain, scene=scene).obs
    assert not obs.place.ages                     # 오래된 카메라가 아닌 순간(그 경우는 바뀜 세계가 섞인다 — 같은 식, 여기서는 단순한 쪽)
    options = available_options(obs)
    batch = sample_posterior(cfg, obs, 256, stream(seed, "posterior"))
    out = rollout(cfg, obs, batch, options)
    model = cost_model(cfg.values, None, out.body, out.loaded)
    code = float(expected_costs(out, model)[out.index("look_again")])
    smod = p4._model_of(cfg, obs)
    dims = [d for d in smod.dims if d in obs.place.readings]
    w = np.asarray(out.weights, dtype=float) / float(np.sum(out.weights))
    state = batch["state"].astype(int)
    costs = world_costs(out, model)[[out.index(o) for o in out.voi.base]]       # (B, 세계)
    rng = np.random.default_rng(seed)
    n = 40_000
    idx = rng.choice(len(w), size=n, p=w)
    loglik = np.tile(np.log(smod.prior[state])[None, :], (n, 1))
    for d in dims:
        j = smod.dims.index(d)
        m, se2 = obs.place.readings[d].value, obs.place.readings[d].sigma_eff ** 2
        mu, sd2 = smod.mean[state, j], smod.sd[state, j] ** 2
        post_var = 1.0 / (1.0 / sd2[idx] + 1.0 / se2)
        x = post_var * (mu[idx] / sd2[idx] + m / se2) + np.sqrt(post_var) * rng.standard_normal(n)
        m2 = x + np.sqrt(se2) * rng.standard_normal(n)
        loglik += _log_binorm(m, m2[:, None], mu[None, :], sd2[None, :], se2)
    post = np.exp(loglik - loglik.max(axis=1, keepdims=True))
    post /= post.sum(axis=1, keepdims=True)
    choice = np.argmin(post @ costs.T, axis=1)
    realized = costs[choice, idx]
    mc = float(realized.mean()) + model["costs"]["delay_per_s"] * out.voi.time
    se_mc = float(realized.std()) / math.sqrt(n)
    assert abs(code - mc) < 4.0 * se_mc + 0.01, (code, mc, se_mc)


def test_a_stale_track_look_signal_follows_the_catalog_and_never_hurts(cfg):
    """수정 1 M3 (리뷰의 MR6 — 다시 봤을 때 '보인다'의 확률을 있음·없음에서 같게 만들어도 떨어지는 시험이 없었다): 오래된 추적의 다시 보기
    신호는 아직 그 자리에 있으면 detect, 없으면 false_alarm(속성 목록)의 확률로 '보인다'이고(세계마다 대조), 본 뒤의 기대 비용은 지금 최선보다
    크지 않다; 두 경우의 보인다 확률이 같으면(정보 없음) 본 뒤의 기대 비용이 지금 최선과 정확히 같다(대조). (이 장면에서 보는 것이 결정을
    바꾸는 경우는 드물다 — 다시 보기의 값이 대개 0이다: 리뷰 M1, 보고서의 미룬 항목.)"""
    from dataclasses import replace

    from robo_jev.simgen.labels import cost_model, expected_costs, voi_decision, world_costs

    look = cfg.catalog["p6"]["stale"]["look"]
    assert look["detect"] - look["false_alarm"] > 0.5
    gains = []
    for seed, domain in ((81, "hospital"), (86, "lab"), (87, "warehouse"), (88, "retail")):
        obs = moment(seed, domain, scene="p6_stale_track").obs
        batch = sample_posterior(cfg, obs, 512, stream(seed, "p"))
        out = rollout(cfg, obs, batch, available_options(obs))
        still = (batch["present"] > 0.5) & (batch["t_leave"] > obs.env.look_s)
        seen = out.voi.likelihood[0]
        assert np.allclose(seen[still], look["detect"]) and np.allclose(seen[~still], look["false_alarm"])
        model = cost_model(cfg.values, None, out.body, out.loaded)
        base = [out.index(o) for o in out.voi.base]
        now = float(min(expected_costs(out, model)[base]))
        after = float(wmean(world_costs(out, model)[out.index("look_again")], out)) - model["costs"]["delay_per_s"] * out.voi.time
        gains.append(now - after)
        flat = replace(out.voi, likelihood=np.vstack([np.full(batch.size, 0.5), np.full(batch.size, 0.5)]))
        out_flat = replace(out, voi=flat)
        decision = voi_decision(out_flat, model)
        after_flat = float(wmean(flat.after(world_costs(out_flat, model)[base], decision), out_flat))
        assert after_flat == pytest.approx(now, abs=1e-9)
    assert all(g >= -1e-9 for g in gains), gains


# ----------------------------------------------------------------------------------------------------------------
# 단조성(새 현상)
# ----------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(("seed", "domain", "scene"), [(81, "hospital", "p6_stale_track"), (82, "warehouse", "p6_sensor_conflict"), (86, "lab", "p6_stale_track")])
def test_slowing_down_never_raises_the_contact_probability(cfg, seed, domain, scene):
    """앞에 서 있을지 모르는 것(오래된 추적·센서 불일치)에는: 단단히 서기 ≤ 절반으로 줄이기 ≤ 그대로 가기(닿을 확률). 모퉁이는 일부러 뺀다 —
    교차하는 줄 위에서 멈추면 가로지르는 사람이 멈춘 로봇에 닿을 수 있어(빨리 지나가는 편이 덜 닿는 순간이 있다) 단조가 아니다."""
    obs = moment(seed, domain, scene=scene).obs
    out = rollout(cfg, obs, sample_posterior(cfg, obs, 2000, stream(seed, "p")), available_options(obs))
    p = {o: float(wmean(out.events["contact"][out.index(o)], out)) for o in ("continue", "slow", "stop_moderate")}
    assert p["stop_moderate"] <= p["slow"] + 1e-12 <= p["continue"] + 2e-12, p


@pytest.mark.parametrize(("seed", "domain", "scene"), [(84, "lab", "p4_place_check"), (85, "factory", "p4_insert")])
def test_a_primary_reading_further_from_nominal_lowers_the_seated_probability(cfg, seed, domain, scene):
    """주 측정(높이·깊이)을 기준(0)에서 멀리 옮길수록 '제대로 앉음'의 사후 확률이 줄어든다(다른 보고는 그대로)."""
    from dataclasses import replace

    obs = moment(seed, domain, scene=scene).obs
    name, reading = obs.place.primary()
    tol = obs.place.tolerance[name]
    values = [0.0, 0.5 * tol, tol, 2 * tol, 4 * tol]
    probs = []
    for v in values:
        readings = {**obs.place.readings, name: replace(reading, value=v)}
        probs.append(p4.exact(cfg, replace(obs, place=replace(obs.place, readings=readings)))["ok_now"])
    assert all(a >= b - 1e-12 for a, b in zip(probs, probs[1:], strict=False)), probs
    assert probs[0] > probs[-1] + 0.2
