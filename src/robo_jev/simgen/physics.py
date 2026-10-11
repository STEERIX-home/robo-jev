"""simgen 물리 핵심: 운동학(P1)과 정적 액면 근사(P5) — numpy 벡터화 (docs/12 §3·§5, Task v0a 지시서 §3.3·§3.4).

모든 함수는 스칼라나 서로 broadcast되는 배열을 받는다. 시간 축은 마지막 축이다. 이 모듈은 세계·관측 타입을 모른다 — 수식만 둔다.

* 가감속 한 번(`ramp_*`): t_start 전에는 v0 등속, 그 뒤 크기 rate로 v_target까지 가감속, 그 뒤 등속. 멈춤·느려짐·출발·
  물러남(두 번 이어 붙임)이 모두 이 꼴이다.
* 쏟아짐(정적 근사): 액면이 수평에서 atan(a/g) + φ 기울고, 용기 가장자리에서 k·(D/2)·tan(그 각)만큼 오른다.
  f·H + 상승 > H 이면 쏟아진다. k ∈ [1.0, 1.6]은 출렁임을 거칠게 덮는 숨은 계수, φ는 바닥 기울기(라디안). 뚜껑이 있으면 쏟아지지 않는다.
* 받침 위 물건(P3, 준정적 — Task v0b): 쏠림 비 r = (받침이 물건에 요구하는 수평 힘) / (수직 힘) — 평지면 a/g, 기울기 φ에서 a로 서면
  (sin φ + a/g)/cos φ. 마찰이 낼 수 있는 비는 μ라서 실제로 전해지는 비는 min(r, μ): **넘어짐** min(r, μ)·h > b_eff(무게중심 높이 h,
  쏠리는 쪽 가장자리까지 수평 거리 b_eff), 그렇지 않고 r > μ면 **미끄러짐**. 미끄러진 거리: 받침 속도가 Δv만큼 a로 바뀌는 동안 물건은
  마찰로만(a_obj = g(μcos φ − sin φ)) 따라가 Δv²/2·(1/a_obj − 1/a)만큼 밀린다(a_obj ≤ 0이면 멈추지 않고 미끄러진다 — 무한).
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "G",
    "incline_ratio",
    "slide_distance",
    "tip_slide",
    "turn_slide_distance",
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
    "turn_slowdown",
    "first_contact",
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


def _segment_distances(dx: np.ndarray, dy: np.ndarray) -> np.ndarray:
    """상대 위치 표본 사이를 직선으로 이은 각 구간의 최소 거리 (…, T − 1)."""
    ddx, ddy = np.diff(dx, axis=-1), np.diff(dy, axis=-1)
    x0, y0 = dx[..., :-1], dy[..., :-1]
    length2 = ddx * ddx + ddy * ddy
    with np.errstate(divide="ignore", invalid="ignore"):
        s = np.where(length2 > 0.0, -(x0 * ddx + y0 * ddy) / np.where(length2 > 0.0, length2, 1.0), 0.0)
    s = np.clip(s, 0.0, 1.0)
    return np.hypot(x0 + s * ddx, y0 + s * ddy)


def min_distance(xr, yr, xa, ya):
    """두 궤적의 중심 사이 최소 거리(마지막 축 = 시간). 표본 사이는 직선으로 보고 구간마다 가장 가까운 점을 찾는다 — 0.05 s 간격이
    팔·손처럼 짧은 거리에서 지나가는 가까움을 놓치지 않게(리뷰 1 m5)."""
    dx = np.asarray(xa, dtype=float) - np.asarray(xr, dtype=float)
    dy = np.asarray(ya, dtype=float) - np.asarray(yr, dtype=float)
    dx, dy = np.broadcast_arrays(dx, dy)
    if dx.shape[-1] < 2:
        return np.hypot(dx, dy)[..., 0]
    return np.min(_segment_distances(dx, dy), axis=-1)


def first_contact(xr, yr, xa, ya, radius_sum, dt: float, with_step: bool = False):
    """(닿았나, 처음 닿는 구간의 상대 속도 — 안 닿았으면 0, 최소 거리[, 처음 닿는 구간 번호 — 안 닿았으면 −1]). 상대 속도는 그 구간의 평균이다."""
    dx = np.asarray(xa, dtype=float) - np.asarray(xr, dtype=float)
    dy = np.asarray(ya, dtype=float) - np.asarray(yr, dtype=float)
    dx, dy = np.broadcast_arrays(dx, dy)
    seg = _segment_distances(dx, dy)
    touching = seg < np.asarray(radius_sum, dtype=float)[..., None]
    hit = touching.any(axis=-1)
    first = np.argmax(touching, axis=-1)
    rel = np.hypot(np.diff(dx, axis=-1), np.diff(dy, axis=-1)) / dt
    speed = np.take_along_axis(rel, first[..., None], axis=-1)[..., 0]
    if with_step:
        return hit, np.where(hit, speed, 0.0), seg.min(axis=-1), np.where(hit, first, -1)
    return hit, np.where(hit, speed, 0.0), seg.min(axis=-1)


def turn_slowdown(v0: float, v: float, approach: float, comfort: float, hard: float) -> tuple[float, float]:
    """회전 전 감속 계획: (들어가는 속도, 쓰는 감속). 접근 구간 전체에 걸친 필요한 감속 (v0² − v²)/(2·접근)으로 줄이되 편안함 하한
    `comfort`보다 작으면 하한으로(일찍 끝내고 v로 간다), 최대 감속 `hard`로도 모자라면 hard로 줄여 더 빠르게 들어간다(리뷰 1 I2)."""
    if v >= v0:
        return float(v0), 0.0
    needed = (v0 * v0 - v * v) / (2.0 * approach)
    if needed > hard:
        return float(np.sqrt(max(0.0, v0 * v0 - 2.0 * hard * approach))), float(hard)
    return float(v), float(max(needed, comfort))


def turn_delay(v0, v, approach, radius, angle_rad, a_long, accel):
    """회전을 속도 v로 돌 때 v0 그대로 도는 것보다 늦는 시간(s). 회전 시작까지 approach 안에서 감속 a_long으로 v까지 줄이고(먼저 v0로
    가다가), 호 r·θ를 v로 돌고, 다시 accel로 v0까지 올린다. 비교는 같은 거리(접근 + 호 + 재가속 거리)를 v0로 가는 시간이다."""
    if v >= v0:
        return 0.0
    s1 = (v0 * v0 - v * v) / (2.0 * a_long)
    t1 = (v0 - v) / a_long
    arc = radius * angle_rad
    s3 = (v0 * v0 - v * v) / (2.0 * accel)
    t3 = (v0 - v) / accel
    taken = (approach - s1) / v0 + t1 + arc / v + t3
    reference = (approach + arc + s3) / v0
    return float(taken - reference)


# ----------------------------------------------------------------------------------------------------------------
# P3 받침 위 물건 (준정적)
# ----------------------------------------------------------------------------------------------------------------


def incline_ratio(accel, incline_rad, g=G):
    """기울기 φ에서 받침이 내리막 쪽으로 쏠리게 a로 속도를 바꿀 때의 쏠림 비 (sin φ + a/g)/cos φ (평지면 a/g)."""
    return (np.sin(incline_rad) + np.asarray(accel, dtype=float) / g) / np.cos(incline_rad)


def tip_slide(ratio, mu, b_eff, h):
    """(넘어짐, 미끄러짐): 넘어짐 = min(r, μ)·h > b_eff; 미끄러짐 = r > μ 이고 넘어지지 않음. 모두 broadcast."""
    ratio, mu = np.asarray(ratio, dtype=float), np.asarray(mu, dtype=float)
    tip = np.minimum(ratio, mu) * np.asarray(h, dtype=float) > np.asarray(b_eff, dtype=float)
    slide = (ratio > mu) & ~tip
    return tip, slide


def slide_distance(dv, accel, mu, incline_rad=0.0, g=G):
    """받침 속도가 dv만큼 accel로 바뀌는 동안 마찰로만 따라가는 물건의 상대 미끄럼 거리 dv²/2·(1/a_obj − 1/a),
    a_obj = g(μcos φ − sin φ). 미끄러지지 않으면(a ≤ a_obj) 0, a_obj ≤ 0이면 inf."""
    dv = np.asarray(dv, dtype=float)
    accel = np.asarray(accel, dtype=float)
    a_obj = g * (np.asarray(mu, dtype=float) * np.cos(incline_rad) - np.sin(incline_rad))
    with np.errstate(divide="ignore", invalid="ignore"):
        d = 0.5 * dv * dv * (1.0 / np.where(a_obj > 0, a_obj, 1.0) - 1.0 / np.where(accel > 0, accel, np.inf))
    out = np.where(a_obj <= 0, np.inf, np.where(accel > a_obj, np.maximum(d, 0.0), 0.0))
    return float(out) if np.ndim(out) == 0 else out


def turn_slide_distance(a_lat, mu, duration, g=G):
    """회전 호를 도는 duration 동안 옆으로 (a_lat − μg)로 밀리고, 회전이 끝난 뒤 남은 옆 속도가 마찰로 멎을 때까지의 미끄럼:
    ½(a − μg)T² + ((a − μg)T)²/(2μg). 미끄러지지 않으면 0."""
    a_lat, mu = np.asarray(a_lat, dtype=float), np.asarray(mu, dtype=float)
    excess = np.maximum(a_lat - mu * g, 0.0)
    out = 0.5 * excess * duration * duration + (excess * duration) ** 2 / (2.0 * np.maximum(mu * g, 1e-9))
    return float(out) if np.ndim(out) == 0 else out
