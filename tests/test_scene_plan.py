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
    # 창·간격은 **프로파일 병합 뒤**의 값이다 — s0.3부터 기본값은 E2의 것이고 E1이 D1의 값으로 덮는다.
    spec = merge_profile(CONFIG, "E1")["disturbance"]
    low, high = spec["window_ms"]
    gap = spec["min_gap_ms"]
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

    # E2는 기본값을 그대로 쓰는 프로파일이라 이 모순이 그대로 간다 (E1은 자기 창을 덮어쓴다).
    with pytest.raises(RuntimeError, match="외란"):
        build_plan(impossible, 3, "E2")


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
        if first.template == "v1#a":
            assert first.text.startswith(objects[first.target].describe(labels))
        assert second.target != first.target
        assert objects[second.target].describe(labels) in second.text
        assert second.zone == first.zone
        # v1은 취약 물체를 "건드리지 마라"로 부른다. 그 id도 구조화되어 있다.
        fragile = [obj.id for obj in plan.objects if "fragile" in obj.attributes]
        # v1은 취약 물체를 **전부** 부르므로(s0.3) 구조화된 보호 목록도 전부다.
        assert first.protected == tuple(fragile)


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


# --------------------------------------------------------------------------
# s0.3 — 사건이 잦은 에피소드 (Task R1 Stage B)
# --------------------------------------------------------------------------


def test_no_two_objects_in_a_scene_share_a_description():
    """지시 문장이 대상·제약을 `<색> <모양>`으로만 부르므로(서식 v0.4) 같은 설명이 둘이면 장면이 풀리지 않는다.
    색이 팔레트의 순열이라 보통 유일하고, 설정이 팔레트보다 많은 물체를 부르면 생성기가 **거절**한다."""
    labels = dict(CONFIG["objects"]["shape_labels"])
    for profile in ("E0", "E1", "E2"):
        for seed in range(20):
            described = [obj.describe(labels) for obj in build_plan(CONFIG, seed, profile).objects]
            assert len(set(described)) == len(described), (profile, seed, described)

    crowded = copy.deepcopy(CONFIG)
    crowded["objects"]["palette"] = crowded["objects"]["palette"][:2]
    crowded["objects"]["count_min"] = crowded["objects"]["count_max"] = 6
    crowded["profiles"]["E2"].pop("objects", None)
    with pytest.raises(ValueError, match="같은 설명"):
        build_plan(crowded, 3, "E2")


def test_e2_changes_the_instruction_several_times_inside_the_window():
    """지시 변경 여러 번 (s0.3, Task R1 B2-ii): 횟수는 `instruction.changes` 안, 시각은 창 안이고 간격을 지키며,
    대상은 매번 다른 **평범한** 물체다. 제약은 덧붙는다 — 변경 지시는 제약을 다시 말하지 않지만 보호 목록은 같다."""
    spec = merge_profile(CONFIG, "E2")["instruction"]
    low, high = spec["change_window_ms"]
    counts = set()
    for seed in range(30):
        plan = build_plan(CONFIG, seed, "E2")
        changes = [step for step in plan.instructions if step.version > 1]
        counts.add(len(changes))
        assert len(changes) <= spec["changes"][1]
        assert [step.version for step in plan.instructions] == list(range(1, len(plan.instructions) + 1))
        times = [step.sim_ms for step in changes]
        assert times == sorted(times)
        assert all(low - GRID_MS <= at <= high for at in times)
        assert all(b - a >= spec["min_gap_ms"] for a, b in itertools.pairwise(times))
        targets = [step.target for step in plan.instructions]
        assert len(set(targets)) == len(targets)  # 같은 대상을 다시 부르지 않는다
        plain = {obj.id for obj in plan.objects if not obj.attributes}
        assert all(step.target in plain for step in plan.instructions)
        assert all(step.protected == plan.instructions[0].protected for step in plan.instructions)
    assert len(counts) >= 3 and max(counts) == spec["changes"][1], counts  # 여러 횟수가 나오고 상한에 닿는다


def test_e2_schedules_more_disturbances_and_inside_the_episode():
    """외란이 **실제로 일어나게** (s0.3, Task R1 B2-iii): 창이 에피소드가 끝나기 전이라야 적용된다. D1의
    `[1200, 20000]`은 평균 9.2 s로 끝나는 에피소드에서 절반이 예정만 되고 끝났다."""
    spec = merge_profile(CONFIG, "E2")["disturbance"]
    low, high = spec["window_ms"]
    assert high <= merge_profile(CONFIG, "E2")["episode"]["max_ms"]
    counts = []
    for seed in range(20):
        times = [item.sim_ms for item in build_plan(CONFIG, seed, "E2").disturbances]
        counts.append(len(times))
        assert spec["count"][0] <= len(times) <= spec["count"][1]
        assert all(low - GRID_MS <= at <= high for at in times)
        assert all(b - a >= spec["min_gap_ms"] for a, b in itertools.pairwise(times))
    e1 = [len(build_plan(CONFIG, seed, "E1").disturbances) for seed in range(20)]
    assert sum(counts) > 2 * sum(e1)  # E2가 E1보다 확실히 잦다


def test_the_e1_profile_reproduces_the_d1_schedule_so_its_episodes_can_be_replayed():
    """E0·E1의 장면·지시·외란 일정은 s0.2(D1)와 **같다**: 변경 횟수·영역 변경은 주 난수를 쓰지 않고(seed 해시),
    변경이 한 번이면 난수 소비가 도입 전과 같다. 그래서 `ep-E1-000235`·`ep-E1-000244`를 재생할 수 있다."""
    for profile in ("E0", "E1"):
        for seed in (100, 235, 244, 295):
            plan = build_plan(CONFIG, seed, profile)
            changes = [step for step in plan.instructions if step.version > 1]
            assert len(changes) <= 1
            spec = merge_profile(CONFIG, profile)["disturbance"]
            assert spec["count"][1] <= 3 and len(plan.disturbances) <= 3


def test_the_instruction_lists_every_constraint_object_by_name():
    """서식 v0.4에서 물체 소개 줄의 `attr=`와 구조화된 목표가 모델 입력 밖이므로, 문장이 부르지 않은 제약은 모델이
    알 길이 없다 — v1 문장은 취약 물체와 금지 물체를 **전부** 부른다 (s0.3, Task R1 B1)."""
    labels = dict(CONFIG["objects"]["shape_labels"])
    for seed in range(15):
        plan = build_plan(CONFIG, seed, "E2")
        text = plan.instructions[0].text
        for obj in plan.objects:
            if obj.attributes:
                assert obj.describe(labels) in text, (seed, obj.id, obj.attributes, text)
        assert "건드리지 마라" in text
        # 변경 지시는 제약을 다시 말하지 않는다 — 모델이 v1의 제약을 기억해야 한다.
        for step in plan.instructions[1:]:
            assert "건드리지 마라" not in step.text
