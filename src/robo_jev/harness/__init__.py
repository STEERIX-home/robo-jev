"""하네스 — 요청 구성과 결과 조합 (docs/02 §2~§4, docs/08 §3~§6).

* :mod:`robo_jev.harness.robot`      — 로봇 스트림 요청(10질문·결합 후보·경유점·실행
  이력)과 조합 규칙 v0(유효성·정지·게이팅·결정 유지·부가 답·명령 생성).
* :mod:`robo_jev.harness.rule_judge` — 모델 자리에 들어가는 규칙 기반 판단기 기준군.

이름을 늦게 푼다: 두 모듈은 설정 파일을 읽으므로 패키지 import만으로 파일을 건드리지
않게 한다.
"""

from typing import Any

__all__ = ["RobotHarness", "build_request", "compose", "rule_judge"]

_SOURCES = {
    "RobotHarness": "robo_jev.harness.robot",
    "build_request": "robo_jev.harness.robot",
    "compose": "robo_jev.harness.robot",
    "rule_judge": "robo_jev.harness.rule_judge",
}


def __getattr__(name: str) -> Any:
    module_name = _SOURCES.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    return getattr(importlib.import_module(module_name), name)


def __dir__() -> list[str]:
    return sorted(__all__)
