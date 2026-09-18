"""장면·일정 생성 검사 — 물리 없이 (docs/02 §1, docs/05 §2).

`robo_jev.sim.scene`은 순수 값이다. 이 파일이 robosuite를 import하지 않고 도는 것 자체가
그 경계의 검사다: 일정이 맞는지 보려고 MuJoCo를 띄울 필요가 없어야 한다.
"""

import copy
import itertools
import subprocess
import sys

import pytest
import yaml
from helpers import SIM_CONFIG

from robo_jev.sim.scene import ScenePlan, build_plan, merge_profile

CONFIG = yaml.safe_load(SIM_CONFIG.read_text(encoding="utf-8"))
GRID_MS = CONFIG["episode"]["schedule_grid_ms"]


@pytest.mark.parametrize(
    "module", ["robo_jev.sim.scene", "robo_jev.sim.controller", "robo_jev.sim"]
)
def test_pure_modules_do_not_pull_in_the_simulator(module):
    """계약·일정 검사가 시뮬레이터 적재 비용을 물지 않아야 한다.

    같은 과정 안에서는 다른 검사 파일이 이미 robosuite를 들여놨을 수 있으므로
    새 인터프리터에서 본다.
    """
    probe = (
        f"import importlib, sys; importlib.import_module({module!r});"
        " print(int('robosuite' in sys.modules), int('mujoco' in sys.modules))"
    )
    result = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, check=True
    )
    assert result.stdout.split() == ["0", "0"], f"{module}가 시뮬레이터를 끌어왔다"


def test_merge_profile_overlays_deeply():
    merged = merge_profile(CONFIG, "E0")
    assert merged["objects"]["count_min"] == CONFIG["profiles"]["E0"]["objects"]["count_min"]
    # 덮어쓰지 않은 형제 키는 그대로 남는다.
    assert merged["objects"]["palette"] == CONFIG["objects"]["palette"]
    assert merged["instruction"]["enabled"] is False
    assert merged["instruction"]["v1_template"] == CONFIG["instruction"]["v1_template"]
    assert "profiles" not in merged
    assert CONFIG["objects"]["count_min"] != merged["objects"]["count_min"], "원본이 바뀌었다"


def test_merge_profile_rejects_an_unknown_profile():
    with pytest.raises(KeyError):
        merge_profile(CONFIG, "E9")


def test_build_plan_is_a_pure_function_of_the_seed():
    first = build_plan(CONFIG, 7, "E1")
    second = build_plan(CONFIG, 7, "E1")
    assert first.to_json() == second.to_json()
    assert build_plan(CONFIG, 8, "E1").to_json() != first.to_json()


def test_plan_round_trips_through_json():
    plan = build_plan(CONFIG, 7, "E1")
    assert ScenePlan.from_json(plan.to_json()) == plan


def test_schedules_sit_on_the_fixed_grid_not_the_control_period():
    """격자는 제어 주기와 무관하다 — 주기를 바꿔도 같은 seed의 일정은 그대로여야 한다."""
    plan = build_plan(CONFIG, 12, "E1")
    for item in plan.disturbances:
        assert item.sim_ms % GRID_MS == 0
    for step in plan.instructions[1:]:
        assert step.sim_ms % GRID_MS == 0

    slower = copy.deepcopy(CONFIG)
    slower["simulator"]["control_hz"] = 25
    assert build_plan(slower, 12, "E1").to_json() == plan.to_json()


def test_disturbances_keep_the_minimum_gap_and_the_window():
    low, high = CONFIG["disturbance"]["window_ms"]
    gap = CONFIG["disturbance"]["min_gap_ms"]
    for seed in range(12):
        times = [item.sim_ms for item in build_plan(CONFIG, seed, "E1").disturbances]
        assert times == sorted(times)
        assert all(low - GRID_MS <= at <= high for at in times)
        assert all(b - a >= gap for a, b in itertools.pairwise(times))


def test_impossible_disturbance_schedule_raises_instead_of_under_generating():
    """조용히 적게 만들면 "외란 N개" 장면을 믿은 쪽이 틀린 수를 쓴다."""
    impossible = copy.deepcopy(CONFIG)
    impossible["disturbance"]["count"] = [3, 3]
    impossible["disturbance"]["window_ms"] = [1200, 1400]
    impossible["disturbance"]["min_gap_ms"] = 5000

    with pytest.raises(RuntimeError, match="외란"):
        build_plan(impossible, 3, "E1")


def test_impossible_attribute_counts_raise_instead_of_under_assigning():
    impossible = copy.deepcopy(CONFIG)
    impossible["objects"]["count_min"] = 3
    impossible["objects"]["count_max"] = 3
    impossible["objects"]["fragile_count"] = [2, 2]
    impossible["objects"]["forbidden_count"] = [2, 2]

    with pytest.raises(RuntimeError, match="붙일 물체가 없다"):
        build_plan(impossible, 3, "E1")


def test_attributes_never_overlap():
    for seed in range(12):
        for obj in build_plan(CONFIG, seed, "E1").objects:
            assert len(obj.attributes) <= 1


def test_e0_has_no_schedule():
    plan = build_plan(CONFIG, 3, "E0")
    assert plan.disturbances == ()
    assert [step.version for step in plan.instructions] == [1]
