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
from robo_jev.simgen.spec import stream
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
