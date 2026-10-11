"""simgen 물리 핵심: 운동학(P1)과 정적 액면 근사(P5) — numpy 벡터화 (docs/12 §3·§5, Task v0a 지시서 §3.3·§3.4).

모든 함수는 스칼라나 서로 broadcast되는 배열을 받는다. 시간 축은 마지막 축이다. 이 모듈은 세계·관측 타입을 모른다 — 수식만 둔다.

* 가감속 한 번(`ramp_*`): t_start 전에는 v0 등속, 그 뒤 크기 rate로 v_target까지 가감속, 그 뒤 등속. 멈춤·느려짐·출발·
  물러남(두 번 이어 붙임)이 모두 이 꼴이다.
* 쏟아짐(정적 근사): 액면이 수평에서 atan(a/g) + φ 기울고, 용기 가장자리에서 k·(D/2)·tan(그 각)만큼 오른다.
  f·H + 상승 > H 이면 쏟아진다. k ∈ [1.0, 1.6]은 출렁임을 거칠게 덮는 숨은 계수, φ는 바닥 기울기(라디안). 뚜껑이 있으면 쏟아지지 않는다.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "G",
    "arrival_time",
    "clear_time",
    "clear_time_from_rest",
    "headroom_used",
    "lateral_accel",
    "liquid_rise",
    "min_distance",
    "ramp_position",
    "ramp_velocity",
    "smoothstep",
    "smoothstep_peak_accel",
    "spills",
    "stopping_distance",
    "turn_delay",
]

G = 9.81


def stopping_distance(v0, latency, decel):
    """반응 거리 + 제동 거리: v0·t_r + v0² / (2·a)."""
    return v0 * latency + v0 * v0 / (2.0 * decel)


def _ramp_parts(t, v0, t_start, rate, v_target):
    t = np.asarray(t, dtype=float)
    v0 = np.asarray(v0, dtype=float)
    v_target = np.asarray(v_target, dtype=float)
    rate = np.asarray(rate, dtype=float)
    tau = np.clip(t - t_start, 0.0, None)
    dv = v_target - v0
    t_ramp = np.abs(dv) / rate
    tau1 = np.minimum(tau, t_ramp)
    return t, v0, v_target, rate, tau, np.sign(dv), tau1


def ramp_position(t, v0, t_start, rate, v_target):
    """가감속 한 번의 위치(t = 0에서 0). rate > 0은 가속도의 크기."""
    t, v0, v_target, rate, tau, sign, tau1 = _ramp_parts(t, v0, t_start, rate, v_target)
    pre = v0 * np.minimum(t, t_start)
    return pre + v0 * tau1 + 0.5 * sign * rate * tau1 * tau1 + v_target * (tau - tau1)


def ramp_velocity(t, v0, t_start, rate, v_target):
    """가감속 한 번의 속도."""
    t, v0, v_target, rate, tau, sign, tau1 = _ramp_parts(t, v0, t_start, rate, v_target)
    return v0 + sign * rate * tau1


def smoothstep(t, t_start, duration, offset):
    """0에서 offset까지 매끄럽게 옮기는 옆 이동 y(t) = Δ(3u² − 2u³), u = (t − t_start) / T ∈ [0, 1]."""
    u = np.clip((np.asarray(t, dtype=float) - t_start) / duration, 0.0, 1.0)
    return offset * (3.0 * u * u - 2.0 * u * u * u)


def smoothstep_peak_accel(offset, duration):
    """smoothstep의 최대 옆 가속 6Δ/T²."""
    return 6.0 * abs(offset) / (duration * duration)


def arrival_time(lateral_distance, radius_sum, speed):
    """진로 중심선에서 lateral_distance 떨어진 상대가 진로 가장자리(반지름 합)에 닿는 시각."""
    return (lateral_distance - radius_sum) / speed


def clear_time(conflict_ahead, radius_sum, speed):
    """앞 conflict_ahead의 교차점을 로봇이 다 지나가는(중심이 교차점 + 반지름 합을 넘는) 시각 — 등속."""
    return (conflict_ahead + radius_sum) / speed


def clear_time_from_rest(distance, latency, accel, cruise):
    """멈춘 로봇이 반응 뒤 accel로 cruise까지 가속하며 distance를 가는 데 걸리는 시각."""
    distance = np.asarray(distance, dtype=float)
    d_ramp = cruise * cruise / (2.0 * accel)
    t_ramp_only = np.sqrt(2.0 * np.clip(distance, 0.0, None) / accel)
    t_with_cruise = cruise / accel + (distance - d_ramp) / cruise
    out = latency + np.where(distance <= d_ramp, t_ramp_only, t_with_cruise)
    return float(out) if out.ndim == 0 else out


def liquid_rise(diameter, accel, k, tilt_rad, g=G):
    """정적 액면 근사의 가장자리 상승 k·(D/2)·|tan(atan(a/g) + φ)|."""
    angle = np.arctan(np.asarray(accel, dtype=float) / g) + tilt_rad
    out = k * (diameter / 2.0) * np.abs(np.tan(angle))
    return float(out) if np.ndim(out) == 0 else out


def spills(fill, height, diameter, accel, k, tilt_rad, lid, g=G):
    """f·H + 상승 > H 이고 뚜껑이 없으면 쏟아진다."""
    level = np.asarray(fill, dtype=float) * height + liquid_rise(diameter, accel, k, tilt_rad, g)
    out = (level > height) & ~np.asarray(lid, dtype=bool)
    return bool(out) if np.ndim(out) == 0 else out


def headroom_used(fill, height, rise):
    """출렁임이 쓴 빈 높이의 비율 min(1, 상승 / (H − f·H)) — 1이면 림에 닿거나 넘친다."""
    free = height * (1.0 - np.asarray(fill, dtype=float))
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(free > 1e-12, np.asarray(rise, dtype=float) / np.where(free > 1e-12, free, 1.0), 1.0)
    out = np.clip(ratio, 0.0, 1.0)
    return float(out) if out.ndim == 0 else out


def lateral_accel(speed, radius):
    """회전의 옆 가속 v²/r."""
    return np.asarray(speed, dtype=float) ** 2 / radius


def min_distance(xr, yr, xa, ya):
    """두 궤적의 중심 사이 최소 거리(마지막 축 = 시간)."""
    return np.min(np.hypot(np.asarray(xa) - np.asarray(xr), np.asarray(ya) - np.asarray(yr)), axis=-1)


def turn_delay(v0, v, approach, radius, angle_rad, decel, accel):
    """회전을 속도 v로 돌 때 v0 그대로 도는 것보다 늦는 시간(s)과 실제로 쓴 진행 방향 감속(m/s²).

    회전 시작까지 approach 동안 decel로 v까지 줄이되(모자라면 필요한 만큼 세게), 호 r·θ를 v로 돌고, 다시 accel로 v0까지 올린다.
    비교는 같은 거리(접근 + 호 + 재가속 거리)를 v0로 가는 시간이다.
    """
    if v >= v0:
        return 0.0, 0.0
    a_long = max(decel, (v0 * v0 - v * v) / (2.0 * approach))
    s1 = (v0 * v0 - v * v) / (2.0 * a_long)
    t1 = (v0 - v) / a_long
    arc = radius * angle_rad
    s3 = (v0 * v0 - v * v) / (2.0 * accel)
    t3 = (v0 - v) / accel
    taken = (approach - s1) / v0 + t1 + arc / v + t3
    reference = (approach + arc + s3) / v0
    return float(taken - reference), float(a_long)
