"""simgen 물리 핵심을 손 계산 사례와 대조한다 (Task v0a 지시서 §5 '물리', docs/12 §12).

멈춤 거리, 가감속 구간의 위치, 가로지름의 도착·비움 시각, 정적 액면 근사의 쏟아짐 판정, 회전의 옆 가속, 최소 거리.
"""

import math

import numpy as np
import pytest

from robo_jev.simgen import physics as ph

G = 9.81


def test_stopping_distance_is_reaction_plus_braking_distance():
    # 1.2 m/s, 반응 0.2 s, 감속 1.5 m/s² → 0.24 + 1.44 / 3 = 0.72 m
    assert ph.stopping_distance(1.2, 0.2, 1.5) == pytest.approx(0.72)
    # 반응 0이면 v²/2a 만 남는다
    assert ph.stopping_distance(2.0, 0.0, 2.0) == pytest.approx(1.0)


def test_stop_profile_rests_at_the_stopping_distance():
    t = np.arange(0.0, 5.0 + 1e-9, 0.05)
    x = ph.ramp_position(t, 1.2, 0.2, 1.5, 0.0)
    v = ph.ramp_velocity(t, 1.2, 0.2, 1.5, 0.0)
    assert x[-1] == pytest.approx(0.72)
    # 반응 시간 동안은 등속: t = 0.2에서 x = 0.24
    assert x[4] == pytest.approx(0.24)
    # 0.8 s 뒤(= 반응 0.2 + 감속 0.6)의 속도는 1.2 − 1.5·0.6 = 0.3
    assert v[16] == pytest.approx(0.3)
    assert v[-1] == 0.0


def test_slow_profile_settles_at_the_target_speed():
    t = np.arange(0.0, 3.0 + 1e-9, 0.05)
    v = ph.ramp_velocity(t, 1.0, 0.1, 1.0, 0.5)
    x = ph.ramp_position(t, 1.0, 0.1, 1.0, 0.5)
    assert v[-1] == pytest.approx(0.5)
    # 0.1 s 등속 + 0.5 s 감속(평균 0.75 m/s) + 2.4 s × 0.5 → 0.1 + 0.375 + 1.2
    assert x[-1] == pytest.approx(1.675)


def test_go_profile_from_rest_accelerates_to_cruise():
    t = np.array([0.0, 0.3, 1.3, 3.3])
    x = ph.ramp_position(t, 0.0, 0.3, 1.0, 1.0)
    # 0.3 s 기다림, 1 s 동안 0→1 m/s(0.5 m), 그 뒤 2 s 등속(2 m)
    assert x == pytest.approx([0.0, 0.0, 0.5, 2.5])


def test_crossing_arrival_and_clear_times():
    # 상대 중심이 진로 중심선에서 3.0 m, 반지름 합 0.5, 1.25 m/s → 가장자리 도착 (3.0 − 0.5) / 1.25 = 2.0 s
    assert ph.arrival_time(3.0, 0.5, 1.25) == pytest.approx(2.0)
    # 교차점 1.5 m 앞, 반지름 합 0.5, 1.0 m/s → 비움 (1.5 + 0.5) / 1.0 = 2.0 s
    assert ph.clear_time(1.5, 0.5, 1.0) == pytest.approx(2.0)


def test_clear_time_from_rest_uses_the_acceleration_phase():
    # 멈춘 로봇: 반응 0.2 s, 가속 1.0 m/s²로 순항 1.0 m/s, 비울 거리 1.5 m
    # 가속 1 s 동안 0.5 m, 나머지 1.0 m는 1 s → 0.2 + 1 + 1 = 2.2 s
    assert ph.clear_time_from_rest(1.5, latency=0.2, accel=1.0, cruise=1.0) == pytest.approx(2.2)
    # 순항에 닿기 전에 비우면 √(2d/a): d = 0.32 → 0.8 s (+ 반응 0.2)
    assert ph.clear_time_from_rest(0.32, latency=0.2, accel=1.0, cruise=1.0) == pytest.approx(1.0)


def test_liquid_rise_is_the_tilted_static_surface():
    # 머그 D = 0.08 m, 감속 2.0 m/s², k = 1, 평지: 상승 = 0.04 · 2/9.81
    assert ph.liquid_rise(0.08, 2.0, 1.0, 0.0) == pytest.approx(0.04 * 2.0 / G)
    # 바닥 기울기 2°는 액면 기울기에 더해진다
    tilt = math.radians(2.0)
    assert ph.liquid_rise(0.08, 2.0, 1.0, tilt) == pytest.approx(0.04 * math.tan(math.atan(2.0 / G) + tilt))
    # 출렁임 계수 k는 상승을 그대로 키운다
    assert ph.liquid_rise(0.08, 2.0, 1.5, 0.0) == pytest.approx(1.5 * 0.04 * 2.0 / G)


def test_spill_threshold_hand_cases():
    # 림 0.10 m, 채움 0.85 → 액면 0.085 + 0.0082 = 0.0932 < 0.10 → 안 쏟아짐
    assert not ph.spills(0.85, 0.10, 0.08, 2.0, 1.0, 0.0, lid=False)
    # 채움 0.95 → 0.095 + 0.0082 = 0.1032 > 0.10 → 쏟아짐
    assert ph.spills(0.95, 0.10, 0.08, 2.0, 1.0, 0.0, lid=False)
    # 뚜껑이 있으면 쏟아지지 않는다
    assert not ph.spills(0.95, 0.10, 0.08, 2.0, 1.0, 0.0, lid=True)


def test_spill_is_vectorised_over_worlds():
    fill = np.array([0.80, 0.90, 0.99])
    lid = np.array([False, False, True])
    out = ph.spills(fill, 0.10, 0.08, 3.0, 1.3, 0.0, lid=lid)
    rise = 1.3 * 0.04 * 3.0 / G  # 0.0159
    assert out.tolist() == [bool(0.080 + rise > 0.10), bool(0.090 + rise > 0.10), False]


def test_headroom_used_is_rise_over_free_height_clipped():
    # 채움 0.75, 림 0.10 → 빈 높이 0.025; 상승 0.0125 → 0.5
    assert ph.headroom_used(0.75, 0.10, 0.0125) == pytest.approx(0.5)
    assert ph.headroom_used(0.75, 0.10, 0.05) == 1.0
    assert ph.headroom_used(1.0, 0.10, 0.0) == 1.0


def test_turn_lateral_acceleration_is_v_squared_over_r():
    assert ph.lateral_accel(1.0, 0.5) == pytest.approx(2.0)
    assert ph.lateral_accel(np.array([0.5, 1.0]), 2.0) == pytest.approx([0.125, 0.5])


def test_smoothstep_lateral_peak_acceleration():
    # y = Δ(3u² − 2u³), u = t/T → 최대 |y''| = 6Δ/T²
    assert ph.smoothstep_peak_accel(0.5, 1.0) == pytest.approx(3.0)
    t = np.linspace(0.0, 1.0, 2001)
    y = ph.smoothstep(t, 0.0, 1.0, 0.5)
    acc = np.gradient(np.gradient(y, t), t)
    assert np.max(np.abs(acc[5:-5])) == pytest.approx(3.0, rel=0.02)
    assert y[0] == 0.0 and y[-1] == pytest.approx(0.5)


def test_segment_closest_approach_does_not_miss_a_pass_between_samples():
    """리뷰 1 m5: 표본 사이를 지나가는 가까움을 놓치지 않는다 — 두 표본 사이를 직선으로 보고 가장 가까운 점을 찾는다."""
    # 상대가 (0.5, −1) → (0.5, +1)로 한 칸에 지나간다(로봇은 원점): 표본에서는 거리 1.12, 실제 최소 0.5
    dx, dy = np.array([0.5, 0.5]), np.array([-1.0, 1.0])
    assert ph.min_distance(0.0, 0.0, dx, dy) == pytest.approx(0.5)


def test_first_contact_reports_the_relative_speed_at_impact():
    t = np.arange(0.0, 2.0 + 1e-9, 0.05)
    xr, yr = 1.0 * t, np.zeros_like(t)            # 로봇 1 m/s
    xa, ya = 1.5 - 0.5 * t, np.zeros_like(t)      # 마주 오는 상대 0.5 m/s
    hit, speed, d_min = ph.first_contact(xr, yr, xa, ya, 0.6, 0.05)
    assert bool(hit) and float(speed) == pytest.approx(1.5) and float(d_min) < 0.6
    hit, speed, _ = ph.first_contact(xr, yr, xa, ya + 2.0, 0.6, 0.05)
    assert not bool(hit) and float(speed) == 0.0


def test_min_distance_finds_the_closest_approach():
    t = np.linspace(0.0, 2.0, 41)
    xr, yr = 1.0 * t, 0.0 * t
    xa, ya = np.full_like(t, 1.5), 0.6 - 0.0 * t
    # 로봇이 x = 1.5를 지날 때 거리 0.6
    assert ph.min_distance(xr, yr, xa, ya) == pytest.approx(0.6)


def test_turn_traverse_delay_matches_a_hand_case():
    # v0 1.0, 회전 반경 1.0, 90° (호 1.571 m), 회전 시작까지 2.0 m, 0.5 m/s로 돈다; 감속 0.5, 재가속 0.5 m/s²
    delay = ph.turn_delay(v0=1.0, v=0.5, approach=2.0, radius=1.0, angle_rad=math.pi / 2, a_long=0.5, accel=0.5)
    s1 = (1.0 - 0.25) / (2 * 0.5)  # 감속 거리 0.75
    t1 = 0.5 / 0.5  # 감속 1 s
    arc = math.pi / 2
    s3 = 0.75  # 재가속 거리
    t3 = 1.0
    expected = ((2.0 - s1) / 1.0 + t1 + arc / 0.5 + t3) - (2.0 + arc + s3) / 1.0
    assert delay == pytest.approx(expected)
    # 같은 속도로 돌면 지연 0
    assert ph.turn_delay(v0=1.0, v=1.0, approach=2.0, radius=1.0, angle_rad=math.pi / 2, a_long=0.5, accel=0.5) == 0.0


def test_turn_slowdown_brakes_at_the_needed_rate_over_the_approach():
    """리뷰 1 I2: 회전 전 감속은 접근 구간 전체에 걸친 필요한 감속 (v0² − v²)/(2·접근)이다 — 편안함 하한과 최대 감속 사이로 자른다."""
    # 1.2 → 0.6 m/s, 접근 2.0 m: 필요한 감속 (1.44 − 0.36) / 4 = 0.27
    v_entry, a_long = ph.turn_slowdown(1.2, 0.6, 2.0, comfort=0.2, hard=3.0)
    assert (v_entry, a_long) == (0.6, pytest.approx(0.27))
    # 필요한 감속이 하한보다 작으면 하한으로(일찍 끝내고 v로 간다)
    assert ph.turn_slowdown(1.0, 0.95, 3.0, comfort=0.2, hard=3.0) == (0.95, 0.2)
    # 최대 감속으로도 모자라면 더 빠르게 들어간다: v_entry² = v0² − 2·hard·접근
    v_entry, a_long = ph.turn_slowdown(2.0, 0.5, 0.2, comfort=0.2, hard=3.0)
    assert a_long == 3.0 and v_entry == pytest.approx(math.sqrt(4.0 - 1.2))
    # 줄이지 않으면 감속 없음
    assert ph.turn_slowdown(1.0, 1.0, 2.0, comfort=0.2, hard=3.0) == (1.0, 0.0)
