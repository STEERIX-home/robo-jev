"""시뮬레이션·컨트롤러 (Task 3a).

* :mod:`robo_jev.sim.controller` — docs/08 §6 컨트롤러 계약 (순수 논리).
* :mod:`robo_jev.sim.scene` — seed에서 만드는 E0/E1 장면과 일정 (순수 값).
* :mod:`robo_jev.sim.tidy_clutter` — 그 장면을 세우는 robosuite 환경.
* :mod:`robo_jev.sim.environment` — reset·step·snapshot·restore.

이름을 **늦게** 푼다. `Controller`·`ScenePlan`은 robosuite가 없어도 쓸 수 있어야 하는데,
패키지가 import될 때 `Environment`를 끌어오면 그 둘만 쓰는 쪽도 MuJoCo를 통째로 적재하게 된다.
"""

from typing import Any

__all__ = ["EXECUTORS", "Controller", "Environment", "ScenePlan", "build_plan"]

_SOURCES = {
    "EXECUTORS": "robo_jev.sim.controller",
    "Controller": "robo_jev.sim.controller",
    "ScenePlan": "robo_jev.sim.scene",
    "build_plan": "robo_jev.sim.scene",
    "Environment": "robo_jev.sim.environment",
}


def __getattr__(name: str) -> Any:
    module_name = _SOURCES.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    return getattr(importlib.import_module(module_name), name)


def __dir__() -> list[str]:
    return sorted(__all__)
