"""장면·일정 생성 검사 — 물리 없이 (docs/02 §1, docs/05 §2).

`robo_jev.sim.scene`은 순수 값이다. 이 파일이 robosuite를 import하지 않고 도는 것 자체가
그 경계의 검사다: 일정이 맞는지 보려고 MuJoCo를 띄울 필요가 없어야 한다.
"""

import copy
import itertools
import math
import subprocess
import sys

import pytest
import yaml
from helpers import HARNESS_CONFIG, SIM_CONFIG

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
    assert merged["instruction"]["v1_templates"] == CONFIG["instruction"]["v1_templates"]
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


# --------------------------------------------------------------------------
# 금지 접촉 물체의 간격 (3b 리뷰 (a), Task 3c-1)
# --------------------------------------------------------------------------

PLANNER = yaml.safe_load(HARNESS_CONFIG.read_text(encoding="utf-8"))["planner"]


def forbidden_limit_mm(obj) -> float:
    """하네스가 금지 물체를 보는 장애물 반지름: 외접 반지름 + margin + forbidden_margin (robot.py `_first_blocker`)."""
    return math.dist((0.0, 0.0, 0.0), obj.half_size_mm) + PLANNER["margin_mm"] + PLANNER["forbidden_margin_mm"]


def xy_distance(a, b) -> float:
    return math.dist(a.pos_mm[:2], b.pos_mm[:2])


@pytest.mark.parametrize("profile", ["E0", "E1"])
def test_a_forbidden_object_keeps_the_planner_clearance_from_every_neighbour(profile):
    """`min_separation_mm: 85`는 하네스의 기본 한계(외접 반지름 + 25 ≈ 79)에 5mm 남짓만 남긴다.
    금지 물체는 `forbidden_margin_mm`만큼 더 큰 장애물이므로 그 안에 이웃이 놓이면 채택 → hold +
    forbidden_segment → forbidden_blocked가 반복된다. 생성기가 그 쌍의 간격을 보장한다."""
    base = CONFIG["objects"]["min_separation_mm"]
    for seed in range(40):
        plan = build_plan(CONFIG, seed, profile)
        for a in plan.objects:
            for b in plan.objects:
                if a.id >= b.id:
                    continue
                required = float(base)
                for member in (a, b):
                    if "forbidden" in member.attributes:
                        required = max(required, forbidden_limit_mm(member))
                assert xy_distance(a, b) >= required - 1e-9, (seed, profile, a.id, b.id, xy_distance(a, b), required)


def test_the_forbidden_separation_is_computed_for_the_largest_shapes():
    """가장 큰 원통(반지름 26·반높이 40)이 금지 물체여도 이웃은 외접 반지름 + 85mm 밖이다."""
    big = copy.deepcopy(CONFIG)
    big["objects"]["shapes"] = ["cylinder"]
    radius, half_height = big["objects"]["cylinder_radius_mm"][1], big["objects"]["cylinder_half_height_mm"][1]
    big["objects"]["cylinder_radius_mm"] = [radius, radius]
    big["objects"]["cylinder_half_height_mm"] = [half_height, half_height]
    # 가장 큰 원통 10개에 금지 물체 2개면 320×560mm 안에 139mm 구멍 둘을 늘 팔 수는 없다(그때는
    # 생성기가 멈춘다 — 아래 검사). 여기서는 간격 계산을 보므로 물체 수를 줄인다.
    big["objects"]["count_max"] = 8
    expected = math.dist((0.0, 0.0, 0.0), (radius, radius, half_height)) + PLANNER["margin_mm"] + PLANNER["forbidden_margin_mm"]
    assert expected > CONFIG["objects"]["min_separation_mm"] + 50  # 기본 간격으로는 턱없이 모자라다
    seen = 0
    for seed in range(20):
        plan = build_plan(big, seed, "E1")
        for a in plan.objects:
            if "forbidden" not in a.attributes:
                continue
            for b in plan.objects:
                if b.id != a.id:
                    seen += 1
                    assert xy_distance(a, b) >= expected - 1e-9
    assert seen > 0


def test_an_impossible_forbidden_separation_raises_instead_of_narrowing_the_gap(tmp_path):
    """자리를 못 찾으면 조용히 좁은 간격을 두지 않는다 — 설정의 모순이다. 배치를 다시 뽑는 것(`_LAYOUT_ATTEMPTS`)도 모순을
    풀지 못하면 그 횟수를 적어 멈춘다. 모순은 하네스 여유로 만든다(금지 물체의 장애물 반지름이 생성 범위보다 크다)."""
    harness = yaml.safe_load(HARNESS_CONFIG.read_text(encoding="utf-8"))
    harness["planner"]["forbidden_margin_mm"] = 2000
    path = tmp_path / "harness.yaml"
    path.write_text(yaml.safe_dump(harness), encoding="utf-8")
    impossible = copy.deepcopy(CONFIG)
    impossible["objects"]["forbidden_separation"] = {"harness_config": str(path)}
    with pytest.raises(RuntimeError, match="금지 물체") as error:
        for seed in range(10):
            build_plan(impossible, seed, "E1")
    assert "다시 뽑아도" in str(error.value)


def test_a_layout_that_cannot_separate_the_forbidden_object_is_redrawn_instead_of_raising():
    """리뷰 1 M9: E1 seed 243은 금지 물체를 400번 안에 떼어 놓을 자리가 없는 배치였다 — 그 배치를 버리고 같은 스트림으로
    다음 배치를 뽑는다. 처음 배치가 성립한 seed는 난수를 더 쓰지 않으므로 그대로다(계보 보존)."""
    plan = build_plan(CONFIG, 243, "E1")
    for a in plan.objects:
        if "forbidden" not in a.attributes:
            continue
        for b in plan.objects:
            if a.id != b.id:
                assert xy_distance(a, b) >= forbidden_limit_mm(a) - 1e-9
    assert len(plan.objects) >= CONFIG["objects"]["count_min"] and plan.instructions


def test_scenes_without_a_violation_are_unchanged_by_the_forbidden_rule():
    """규칙은 위반한 금지 물체만 다시 놓는다 — 위반이 없는 seed의 장면은 그대로다 (재현·계보 보존)."""
    loose = copy.deepcopy(CONFIG)
    loose["objects"].pop("forbidden_separation")
    changed = unchanged = 0
    for seed in range(30):
        before = build_plan(loose, seed, "E1")
        after = build_plan(CONFIG, seed, "E1")
        violated = any(
            "forbidden" in a.attributes and xy_distance(a, b) < forbidden_limit_mm(a)
            for a in before.objects for b in before.objects if a.id != b.id
        )
        if violated:
            changed += 1
            assert after.to_json() != before.to_json()
        else:
            unchanged += 1
            assert after.to_json() == before.to_json()
    assert changed > 0 and unchanged > 0


# --------------------------------------------------------------------------
# 구조화된 목표 (docs/08 §3.2 `goal`, Task 3c-1)
# --------------------------------------------------------------------------


def test_instructions_carry_the_structured_goal_next_to_the_text():
    """지시는 텍스트만이 아니다 — 대상 id·목적지 id를 같이 들고 다닌다.

    텍스트를 다시 파싱해 대상을 찾는 것은 근사다(3b `named_target`). 장면 계획은 지시를 만들 때
    어느 물체·어느 영역인지 이미 알고 있으므로 그것을 그대로 적는다.
    """
    labels = dict(CONFIG["objects"]["shape_labels"])
    for seed in range(1, 13):
        plan = build_plan(CONFIG, seed, "E1")
        objects = {obj.id: obj for obj in plan.objects}
        zones = {zone.id: zone for zone in plan.zones}
        first, second = plan.instructions
        for step in (first, second):
            assert step.target in objects and step.zone in zones
            assert objects[step.target].attributes == ()  # 지시의 대상은 평범한 물체다
            assert zones[step.zone].desc in step.text
        assert objects[first.target].describe(labels) in first.text
        if first.template == "v1#0":
            assert first.text.startswith(objects[first.target].describe(labels))
        assert second.target != first.target
        assert objects[second.target].describe(labels) in second.text
        assert second.zone == first.zone
        # v1은 취약 물체를 "건드리지 마라"로 부른다. 그 id도 구조화되어 있다.
        fragile = [obj.id for obj in plan.objects if "fragile" in obj.attributes]
        assert first.protected == (fragile[0],) if fragile else first.protected == ()


def test_the_structured_goal_round_trips_through_json():
    plan = build_plan(CONFIG, 9, "E1")
    restored = ScenePlan.from_json(plan.to_json())
    assert [(s.target, s.zone, s.protected) for s in restored.instructions] == [
        (s.target, s.zone, s.protected) for s in plan.instructions
    ]


def test_a_plan_without_structured_fields_still_loads():
    """이전 snapshot(구조화 필드가 없는 지시)도 읽힌다 — 필드는 선택이다."""
    data = build_plan(CONFIG, 9, "E1").to_json()
    for step in data["instructions"]:
        for key in ("target", "zone", "protected"):
            step.pop(key)
    plan = ScenePlan.from_json(data)
    assert plan.instructions[0].target is None and plan.instructions[0].protected == ()
