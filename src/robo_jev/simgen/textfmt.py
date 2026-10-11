"""수치 표기: 단위·자릿수 흔들기 (Task v0a 지시서 §3.8 '단위·숫자 표기 흔들기').

관측(:class:`~robo_jev.simgen.observation.Formatting`)이 단위를 정하고, 자릿수는 보고 해상도가 정한다 — 서술기는 값을 더 반올림하지
않는다(m ↔ cm ↔ mm는 정확히 바뀐다; km/h만 소수 한 자리로 반올림된다). 규칙 적용 질문은 :func:`shown_si` 로 **글에 보인 수**를 SI로
되돌려 규칙에 넣는다 — 정답이 글과 어긋나지 않는다.
"""

from __future__ import annotations

from robo_jev.simgen.spec import Reading, decimals_for

__all__ = ["accel", "distance", "number", "percent", "shown_si", "speed", "time"]

_SCALE = {"m": 1.0, "cm": 100.0, "mm": 1000.0, "m/s": 1.0, "km/h": 3.6, "mm/s": 1000.0, "cm/s": 100.0, "s": 1.0, "ms": 1000.0}


def number(value: float, decimals: int) -> str:
    text = f"{value:.{decimals}f}"
    if text.startswith("-") and float(text) == 0.0:
        text = text[1:]
    return text


def _in_unit(value: float, resolution: float, unit: str) -> tuple[str, float]:
    """해상도 격자로 반올림한 값을 그 단위로 쓴다(보고 값은 이미 격자 위라 그대로; 힌트처럼 계산한 값은 여기서 격자에 맞춘다)."""
    scale = _SCALE[unit]
    if resolution > 0:
        value = round(value / resolution) * resolution
    decimals = decimals_for(resolution * scale) if resolution > 0 else 2
    if unit == "km/h":
        decimals = 1
    text = number(value * scale, decimals)
    return text, float(text) / scale


def distance(value: float, resolution: float, unit: str) -> str:
    text, _ = _in_unit(value, resolution, unit)
    return f"{text} {unit}"


def speed(value: float, resolution: float, unit: str) -> str:
    text, _ = _in_unit(value, resolution, unit)
    return f"{text} {unit}"


def time(value: float, resolution: float, unit: str) -> str:
    text, _ = _in_unit(value, resolution, unit)
    return f"{text} {unit}"


def accel(value: float) -> str:
    return f"{number(value, 1)} m/s²"


def percent(fraction: float) -> str:
    return f"{number(fraction * 100.0, 0)} %"


def shown_si(reading: Reading, unit: str) -> float:
    """글에 보인 수를 SI로 되돌린 값."""
    _, si = _in_unit(reading.value, reading.resolution, unit)
    return si
