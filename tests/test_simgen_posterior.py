"""simgen 사후 표본 (docs/12 §5, Task v0a 지시서 §3.5, 리뷰 1 C1·I1·I4).

손 계산 사례와 보정 시험으로 사후가 맞는지 본다. 리뷰의 변이체(추적 외삽 부호 뒤집기 M1, 의도 사후를 사전으로 M2, 채움을 구간 가운데의
가우스로 M3)가 여기서 떨어진다:

* 의도: 사전 × 단서 가능도의 베이즈 갱신 + 나이 혼합 — 손 계산(EX1)과, 생성 과정에서 뽑은 참 의도에 대한 보정과 날카로움(사전보다 낫다).
* 추적 외삽: 끼어듦(옆 위치), 교차(진로까지 거리), 마주 옴(그 시각의 둘 사이 간격 — C1, EX6 재계산 3.36 − 1.80 × 1.15 = 1.29 m)과
  위치 두 개(마주 옴에서 차이 / 시간 = 다가오는 속도).
* 마주 옴의 글과 정답이 같은 뜻인지: 세 문체로 쓴 글에서 수를 읽어 다시 계산한 지금 간격 = 사후의 지금 간격.
* 채움: 속성 목록의 채움 범위 위 평평한 사전 × 정확한 보고 모형 — 거절 표본과 같은 분포(검열된 맨 위·맨 아래 구간 포함).
* 뚜껑·옆 장애물·검열된 보고·닫힌 꼴 정답.
"""

from __future__ import annotations

import math
import re
from dataclasses import replace

import numpy as np
import pytest
from simgen_helpers import moment

from robo_jev.simgen import scenes
from robo_jev.simgen.catalog import INTENTS, default_config
from robo_jev.simgen.narrate import RenderPlan, render_state
from robo_jev.simgen.posterior import exact_labels, intent_posterior, lid_probability, sample_posterior
from robo_jev.simgen.rationale import head_on_current_gap
from robo_jev.simgen.rollout import available_options
from robo_jev.simgen.spec import FILL_BINS, Reading, fill_bin_index, fill_bins, round_to, stream

K = 4000


@pytest.fixture(scope="module")
def cfg():
    return default_config()


def _rng(seed: int = 0) -> np.random.Generator:
    return np.random.default_rng(seed)


# ----------------------------------------------------------------------------------------------------------------
# 의도
# ----------------------------------------------------------------------------------------------------------------


def test_intent_posterior_matches_the_hand_computed_bayes_update_with_age_mixing(cfg):
    """EX1: 어른·끼어듦 사전 .50/.40/.10 × 단서 'steady' 가능도 .15/.50/.10 → .263/.702/.035; 나이 0.15 s, 위험률 0.15/s →
    q = 1 − e^(−0.0225) = 0.0222 → .268/.695/.036 (리뷰 1의 손 계산)."""
    post = intent_posterior(cfg, "adult", "step_in", "steady", 0.15)
    assert post == pytest.approx([0.2685, 0.6951, 0.0364], abs=5e-4)
    fresh = intent_posterior(cfg, "adult", "step_in", "steady", 0.0)
    assert fresh == pytest.approx(np.array([0.075, 0.2, 0.01]) / 0.285, abs=1e-9)
    stale = intent_posterior(cfg, "adult", "step_in", "steady", 60.0)  # 오래되면 사전으로
    assert stale == pytest.approx([0.5, 0.4, 0.1], abs=1e-3)


def _intent_draws(cfg, n: int, seed: int = 7):
    """생성기와 같은 과정(관측 시점 의도 ~ 사전, 단서 ~ P(단서 | 의도), 관측 뒤 바뀜)으로 (사후, 사전, 참 의도)."""
    rng = _rng(seed)
    rows = []
    kinds = [k for k, spec in cfg.catalog["agents"].items() if "intent_prior" in spec]
    for _ in range(n):
        kind = kinds[int(rng.integers(len(kinds)))]
        families = list(cfg.agent(kind)["intent_prior"])
        family = families[int(rng.integers(len(families)))]
        age = round_to(float(rng.uniform(0.0, 1.5)), 0.05)
        _, cue, now = scenes._intent(cfg, kind, family, age, rng)
        prior = np.array([cfg.agent(kind)["intent_prior"][family][i] for i in INTENTS[family]])
        rows.append((intent_posterior(cfg, kind, family, cue, age), prior / prior.sum(), now))
    return rows


def test_intent_posterior_is_calibrated_and_sharper_than_the_prior(cfg):
    """참 의도에 대한 보정(확률 구간별 실제 비율)과 정보: 사후의 평균 로그 점수가 사전보다 분명히 높다(사전으로 바꾼 변이체는 같다)."""
    rows = _intent_draws(cfg, 12000)
    probs = np.array([p[i] for p, _, _ in rows for i in range(len(p))])
    hits = np.array([float(i == now) for p, _, now in rows for i in range(len(p))])
    for lo, hi in ((0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0)):
        mask = (probs >= lo) & (probs < hi)
        if mask.sum() < 200:
            continue
        p, y = probs[mask], hits[mask]
        z = (y.sum() - p.sum()) / math.sqrt(np.sum(p * (1 - p)))
        assert abs(z) < 4.0, (lo, hi, p.mean(), y.mean(), z)
    log_post = np.mean([math.log(p[now]) for p, _, now in rows])
    log_prior = np.mean([math.log(q[now]) for _, q, now in rows])
    assert log_post > log_prior + 0.08, (log_post, log_prior)  # 이 표에서 약 +0.13; 사전으로 바꾸면 0


def test_sampled_intents_follow_the_posterior(cfg):
    obs = moment(41, "retail", scene="p1p5_step_in_carrying", body="humanoid", agent_kind="adult").obs
    batch = sample_posterior(cfg, obs, K, _rng(1))
    expected = intent_posterior(cfg, "adult", "step_in", obs.agent.cue, obs.agent.age)
    shares = np.array([np.mean(batch["intent"] == i) for i in range(3)])
    assert shares == pytest.approx(expected, abs=0.025)


# ----------------------------------------------------------------------------------------------------------------
# 추적 외삽 (보고 나이 동안의 움직임)
# ----------------------------------------------------------------------------------------------------------------


def _step_in_obs(cfg, track, ages, speed, heading):
    base = moment(41, "retail", scene="p1_step_in", body="mobile_base", agent_kind="adult").obs
    return replace(base, agent=replace(base.agent, track=track, track_ages=ages, speed=speed, heading=heading))


def test_step_in_stale_fix_is_carried_forward_along_the_reported_heading(cfg):
    """옆 위치 +0.50 m(왼쪽)를 1.0 s 전에 봤고 왼쪽으로 0.60 m/s → 지금 약 +1.10 m(부호를 뒤집은 변이체는 −0.10 m)."""
    obs = _step_in_obs(cfg, (Reading(0.50, 0.01, 0.01),), (1.0,), Reading(0.60, 0.01, 0.05), "left")
    lat = sample_posterior(cfg, obs, K, _rng(2))["lat0"]
    assert lat.mean() == pytest.approx(1.10, abs=0.02)
    right = _step_in_obs(cfg, (Reading(-0.50, 0.01, 0.01),), (1.0,), Reading(0.60, 0.01, 0.05), "right")
    assert sample_posterior(cfg, right, K, _rng(2))["lat0"].mean() == pytest.approx(-1.10, abs=0.02)


def test_step_in_two_fixes_give_the_lateral_speed_and_the_current_position(cfg):
    """+0.20 m 1.5 s 전, +0.50 m 1.0 s 전 → 옆 속도 0.6 m/s, 지금 +1.10 m."""
    obs = _step_in_obs(cfg, (Reading(0.20, 0.005, 0.01), Reading(0.50, 0.005, 0.01)), (1.5, 1.0), None, "left")
    batch = sample_posterior(cfg, obs, K, _rng(3))
    assert batch["u0"].mean() == pytest.approx(0.60, abs=0.03)
    assert batch["lat0"].mean() == pytest.approx(1.10, abs=0.03)


def test_crossing_stale_fix_is_carried_forward_toward_the_path(cfg):
    """진로 가장자리까지 2.00 m를 0.5 s 전에 봤고 1.2 m/s로 다가옴 → 지금 1.40 m(면 간격)."""
    base = moment(12, "warehouse", scene="p1_crossing", body="forklift_amr", agent_kind="adult").obs
    obs = replace(base, agent=replace(base.agent, track=(Reading(2.00, 0.02, 0.01),), track_ages=(0.5,), speed=Reading(1.20, 0.02, 0.05)))
    batch = sample_posterior(cfg, obs, K, _rng(4))
    surface = batch["dist"] - obs.robot.radius - batch["ra"]
    assert surface.mean() == pytest.approx(1.40, abs=0.02)
    assert batch["va"].mean() == pytest.approx(1.20, abs=0.02)


def _head_on_obs(cfg, track, ages, speed, robot_speed=0.75):
    base = moment(13, "hospital", scene="p1_head_on", body="mobile_base", agent_kind="adult").obs
    robot = replace(base.robot, speed=robot_speed, cruise=robot_speed)
    return replace(base, robot=robot, agent=replace(base.agent, track=track, track_ages=ages, speed=speed))


def test_head_on_gap_report_is_the_gap_at_report_time(cfg):
    """리뷰 1 C1·EX6: '1.15 s 전 둘 사이 간격 3.36 m', 상대 1.05 m/s, 로봇 0.75 m/s → 지금 간격 3.36 − 1.80 × 1.15 = 1.29 m
    (옛 엔진은 상대 속도만 빼서 2.15 m였다; 외삽 부호를 뒤집은 변이체는 5.43 m)."""
    obs = _head_on_obs(cfg, (Reading(3.36, 0.03, 0.01),), (1.15,), Reading(1.05, 0.02, 0.05))
    batch = sample_posterior(cfg, obs, K, _rng(5))
    surface = batch["dist"] - obs.robot.radius - batch["ra"]
    assert surface.mean() == pytest.approx(3.36 - 1.80 * 1.15, abs=0.02)
    assert batch["va"].mean() == pytest.approx(1.05, abs=0.02)


def test_head_on_two_fixes_measure_the_closing_speed(cfg):
    """간격 4.00 m (1.65 s 전) → 3.10 m (1.15 s 전): 다가오는 속도 1.8 m/s = 상대 1.05 + 로봇 0.75, 지금 3.10 − 1.8 × 1.15 = 1.03 m."""
    obs = _head_on_obs(cfg, (Reading(4.00, 0.01, 0.01), Reading(3.10, 0.01, 0.01)), (1.65, 1.15), None)
    batch = sample_posterior(cfg, obs, K, _rng(6))
    assert batch["va"].mean() == pytest.approx(1.05, abs=0.05)
    surface = batch["dist"] - obs.robot.radius - batch["ra"]
    assert surface.mean() == pytest.approx(3.10 - 1.8 * 1.15, abs=0.05)


_GAP = re.compile(r"(?:gap between you was|ahead[^\n]*?\bwas)\s+([\d.]+)\s*(m|cm|mm)\b|([\d.]+)\s*(m|cm|mm) ahead, moving toward you")
_UNIT = {"m": 1.0, "cm": 0.01, "mm": 0.001, "m/s": 1.0, "km/h": 1 / 3.6, "mm/s": 0.001, "s": 1.0, "ms": 0.001}


def _first(pattern: str, text: str) -> float:
    found = re.search(pattern, text)
    assert found, (pattern, text)
    return float(found.group(1)) * _UNIT[found.group(2)]


@pytest.mark.parametrize("style", ["facts", "prose", "planner"])
def test_head_on_text_and_labels_mean_the_same_current_gap(cfg, style):
    """글에서 수를 읽어(간격, 나이, 상대 속도, 로봇 속도) 다시 계산한 지금 간격 = 사후의 지금 간격(리뷰 1 C1의 글-정답 일치)."""
    obs = _head_on_obs(cfg, (Reading(3.36, 0.03, 0.01),), (1.15,), Reading(1.05, 0.02, 0.05))
    obs = replace(obs, fmt=replace(obs.fmt, distance_unit="m", speed_unit="m/s", time_unit="s"))
    text = render_state(obs, RenderPlan(style, available_options(obs)), stream(0, "render"))
    gap = _first(r"([\d.]+) (m) ahead, moving toward you" if style != "prose" else r"the gap between you was ([\d.]+) (m)", text)
    agent_speed = _first(r"moving toward you at ([\d.]+) (m/s)", text)
    age = _first(r"(?:last update|As of) ([\d.]+) (s)", text)
    robot_speed = _first(r"(?:at|speed:) ([\d.]+) (m/s)", text)
    from_text = gap - (agent_speed + robot_speed) * age
    batch = sample_posterior(cfg, obs, K, _rng(7))
    assert (batch["dist"] - obs.robot.radius - batch["ra"]).mean() == pytest.approx(from_text, abs=0.02)
    assert head_on_current_gap(obs)[0] == pytest.approx(from_text, abs=1e-9)


# ----------------------------------------------------------------------------------------------------------------
# 채움 — 속성 목록의 범위 사전 × 정확한 보고 모형
# ----------------------------------------------------------------------------------------------------------------


_CENTRES = np.array([round((lo + hi) / 2.0, 4) for lo, hi, _ in FILL_BINS])


def _reports(raw: np.ndarray, mode: str) -> np.ndarray:
    """날 추정값 → 보고(생성기의 감각 모형과 같은 규칙: 말 구간의 가운데, 또는 5 % 반올림)."""
    if mode == "phrase":
        return _CENTRES[fill_bins(raw).astype(int)]
    return np.round(np.round(raw / 0.05) * 0.05, 3)


def _rejection(cfg, kind: str, mode: str, value: float, n: int, seed: int = 11) -> np.ndarray:
    """f ~ U(속성 목록 범위), 날 추정 = clip(f + σz, 0, 1), 보고 = 말 구간 가운데 / 5 % 반올림 — 보고가 같은 것만 받는다."""
    spec = cfg.container(kind)
    sigma = cfg.catalog["fill_sigma"]["transparent" if spec["transparent"] else "opaque"]
    rng = _rng(seed)
    f = rng.uniform(spec["fill"][0], spec["fill"][1], n)
    raw = np.clip(f + sigma * rng.standard_normal(n), 0.0, 1.0)
    return f[np.isclose(_reports(raw, mode), value)]


FILL_CASES = [
    ("mug", "phrase", 0.99),            # 'full to the brim' — 맨 위 구간(검열)
    ("mug", "phrase", 0.15),            # 'less than a third' — 범위 하한(0.40) 아래의 보고
    ("mug", "percent", 1.00),           # 100 % — 맨 위(검열)
    ("beaker_250", "phrase", 0.95),     # 투명, 'nearly to the brim'
    ("beaker_250", "percent", 0.85),
    ("bucket", "phrase", 0.5),          # 'about half full'
    ("takeaway_cup", "percent", 0.55),  # 범위 하한(0.50) 근처
]


def _container_obs(cfg, kind: str, mode: str, value: float):
    base = moment(14, "lab", scene="p5_braking", body="mobile_base").obs
    spec = cfg.container(kind)
    sigma = cfg.catalog["fill_sigma"]["transparent" if spec["transparent"] else "opaque"]
    if mode == "phrase":
        lo, hi, _ = FILL_BINS[fill_bin_index(value)]
        reading = Reading(round((lo + hi) / 2.0, 4), sigma, round(hi - lo, 4))
    else:
        reading = Reading(value, sigma, 0.05)
    container = replace(base.container, kind=kind, diameter=spec["diameter"], height=spec["height"], fill=reading, fill_mode=mode)
    return replace(base, container=container)


@pytest.mark.parametrize(("kind", "mode", "value"), FILL_CASES)
def test_fill_posterior_matches_rejection_sampling_under_the_catalog_range_prior(cfg, kind, mode, value):
    """사후 채움 = 속성 목록 범위의 평평한 사전에서 보고 모형을 거꾸로 — 거절 표본과 평균·분위수가 같다(구간 가운데의 가우스로 바꾼
    변이체는 맨 위·맨 아래 구간에서 크게 어긋난다)."""
    exact = _rejection(cfg, kind, mode, value, 400_000)
    assert len(exact) > 2000
    fill = sample_posterior(cfg, _container_obs(cfg, kind, mode, value), 20_000, _rng(12))["fill"]
    lo, hi = cfg.container(kind)["fill"]
    assert fill.min() >= lo - 1e-9 and fill.max() <= hi + 1e-9
    assert fill.mean() == pytest.approx(exact.mean(), abs=0.006)
    assert np.quantile(fill, [0.1, 0.5, 0.9]) == pytest.approx(np.quantile(exact, [0.1, 0.5, 0.9]), abs=0.012)


def test_exact_fill_bins_match_the_sampled_fill(cfg):
    obs = _container_obs(cfg, "mug", "phrase", 0.87)
    exact = exact_labels(cfg, obs)["fill_bin"]
    fill = sample_posterior(cfg, obs, 40_000, _rng(13))["fill"]
    bins = fill_bins(fill)
    sampled = np.array([np.mean(bins == b) for b in range(len(FILL_BINS))])
    assert exact == pytest.approx(sampled, abs=0.02)
    assert exact.sum() == pytest.approx(1.0)


# ----------------------------------------------------------------------------------------------------------------
# 뚜껑·옆 장애물·검열된 보고·닫힌 꼴
# ----------------------------------------------------------------------------------------------------------------


def test_lid_probability_follows_the_view_model(cfg):
    obs = _container_obs(cfg, "deli_tub", "percent", 0.8)
    prior = cfg.container("deli_tub")["lid_prior"]
    views = cfg.catalog["lid_view"]
    assert lid_probability(cfg, replace(obs, container=replace(obs.container, lid="open"))) == 0.0
    assert lid_probability(cfg, replace(obs, container=replace(obs.container, lid="lid"))) == 1.0
    p = lid_probability(cfg, replace(obs, container=replace(obs.container, lid="unknown")))
    with_lid = prior * views["given_lid"]["unknown"]
    assert p == pytest.approx(with_lid / (with_lid + (1 - prior) * views["given_open"]["unknown"]))


def test_lid_posterior_is_calibrated_against_the_generators_lids(cfg):
    rng = _rng(21)
    ctx = scenes.Context("cafe", "mobile_base", "p5_braking", "moving", None, "lidar_camera", "facts", 0)
    obs = _container_obs(cfg, "takeaway_cup", "percent", 0.8)
    rows = []
    for kind in ("takeaway_cup", "deli_tub", "specimen_cup", "paint_can", "mug"):
        for _ in range(3000):
            hidden, at = scenes._container_hidden(cfg, ctx, {"kind": kind}, rng)
            if at["lid_view"] != "unknown":
                assert (hidden["lid"] > 0.5) == (at["lid_view"] == "lid")
                continue
            p = lid_probability(cfg, replace(obs, container=replace(obs.container, kind=kind, lid="unknown")))
            rows.append((p, hidden["lid"]))
    p = np.array([a for a, _ in rows])
    y = np.array([b for _, b in rows])
    for value in np.unique(p):
        mask = p == value
        z = (y[mask].sum() - p[mask].sum()) / math.sqrt(np.sum(p[mask] * (1 - p[mask])) + 1e-12)
        assert abs(z) < 4.0, (value, y[mask].mean())


def test_side_obstacle_posterior_uses_the_view(cfg):
    from robo_jev.simgen.observation import SideObs

    base = moment(16, "sidewalk", scene="p1_step_in", body="mobile_base").obs
    prior = cfg.domain("sidewalk")["side_obstacle_prior"]
    env = replace(base.env, sides=(SideObs("occluded", "a parked van"), SideObs("blocked", "a lamp post")))
    batch = sample_posterior(cfg, replace(base, env=env), K, _rng(22))
    assert batch["side_left"].mean() == pytest.approx(prior, abs=0.03)
    assert batch["side_right"].min() == 1.0
    env = replace(base.env, sides=(SideObs("clear"), SideObs("clear")))
    batch = sample_posterior(cfg, replace(base, env=env), K, _rng(22))
    assert batch["side_left"].max() == 0.0 and batch["side_right"].max() == 0.0


def test_a_reading_at_its_floor_is_treated_as_censored(cfg):
    """속도 보고가 바닥(0.05 m/s)에 붙으면 '그 이하' — 사후는 바닥 칸(0.075) 아래에 몰리고 음수가 없다."""
    floor = Reading(0.05, 0.10, 0.05, 0.05)
    assert floor.censored and not Reading(0.10, 0.10, 0.05, 0.05).censored
    base = moment(12, "warehouse", scene="p1_crossing", body="forklift_amr", agent_kind="adult").obs
    obs = replace(base, agent=replace(base.agent, speed=floor, track=(Reading(2.0, 0.02, 0.01),), track_ages=(0.2,)))
    va = sample_posterior(cfg, obs, K, _rng(23))["va"]
    assert va.min() >= 0.05 - 1e-9
    assert va.mean() < 0.075 + 0.10  # 대부분 바닥 칸 아래(σ = 0.10의 꼬리만 위로)
    assert np.mean(va < 0.075 + 2 * 0.10) > 0.95


def test_exact_world_events_match_large_sample_frequencies(cfg):
    from robo_jev.simgen.rollout import rollout

    for seed, domain, forced in ((16, "sidewalk", {"scene": "p1_step_in", "body": "mobile_base"}), (13, "hospital", {"scene": "p1_head_on"})):
        obs = moment(seed, domain, **forced).obs
        exact = exact_labels(cfg, obs)
        out = rollout(cfg, obs, sample_posterior(cfg, obs, 20_000, _rng(24)))
        event = "stops_short" if obs.scene == "p1_step_in" else "passable"
        assert exact[event] == pytest.approx(out.world_events[event].mean(), abs=0.012), (obs.scene, exact[event])
    obs = moment(14, "lab", scene="p5_braking", body="mobile_base").obs
    exact = exact_labels(cfg, obs)
    out = rollout(cfg, obs, sample_posterior(cfg, obs, 20_000, _rng(25)))
    for option in ("stop_gentle", "stop_moderate", "stop_hard"):
        assert exact[f"overrun|{option}"] == pytest.approx(out.events["overrun"][out.index(option)].mean(), abs=0.012), option
