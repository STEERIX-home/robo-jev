"""추출 인터페이스 검사 — `extract(recon, robot, now_ms)` (docs/08 §3.2).

앞단은 교체 가능한 모듈이고 모두 **같은 구조화 상태 스키마**를 채운다. 여기서는 그
스키마를 고정하고, D1의 참값 어댑터가 그 스키마만 채우는지(가려진 물체의 참값이
상태로 새지 않는지) 본다. E2의 시뮬 3D 카메라는 같은 인터페이스에 뒤에 붙는다.
"""

import copy

import pytest
import yaml
from helpers import HARNESS_CONFIG

from robo_jev.perception.pointworld import (
    EXTRACTOR_VERSION,
    GroundTruthAdapter,
    Reconstruction,
    SceneSummary,
    TrackedInstance,
    extract,
)

CONFIG = yaml.safe_load(HARNESS_CONFIG.read_text(encoding="utf-8"))
PERCEPTION = CONFIG["perception"]

#: docs/08 §3.2 표의 필드. 앞단이 무엇이든 이 키가 모두 있어야 한다.
STATE_FIELDS = (
    "t",
    "goal",
    "objects",
    "scene",
    "zones",
    "robot",
    "exec",
    "events",
    "derived",
    "commitment",
    "image",
    "geom",
)

OBJECT_FIELDS = (
    "id",
    "desc",
    "class",
    "pose_mm",
    "quat",
    "precision_mm",
    "pose_source",
    "obb_mm",
    "top_mm",
    "graspable_faces",
    "surface_conf",
    "visible_ratio",
    "last_seen_ms",
    "age_ms",
    "reid",
    "attributes",
)


def instance(track_id: str, pose_mm=(300, 0, -80), **over) -> TrackedInstance:
    fields = {
        "track_id": track_id,
        "desc": f"물체 {track_id}",
        "cls": "box",
        "pose_mm": tuple(pose_mm),
        "quat": (0.0, 0.0, 0.0, 1.0),
        "precision_mm": 3,
        "obb_mm": (60, 60, 64),
        "top_mm": pose_mm[2] + 32,
        "graspable_faces": ("top", "side"),
        "surface_conf": 0.92,
        "visible_ratio": 1.0,
        "points": 480,
        "last_seen_ms": 0,
        "observed_now": True,
        "attributes": (),
    }
    fields.update(over)
    return TrackedInstance(**fields)


def reconstruction(*instances: TrackedInstance, **over) -> Reconstruction:
    fields = {
        "instances": instances or (instance("o0"),),
        "scene": SceneSummary(
            work_surface_mm=-112, free_width_mm=420, corridor_mm=180, clearance_mm=40
        ),
        "zones": ({"id": "zoneL", "desc": "왼쪽 정리 영역", "bounds_mm": [-120, 150, 180, 330]},),
        "goal": {
            "text": "빨간 상자를 왼쪽 정리 영역으로 옮겨라",
            "version": 1,
            "target_zone": "zoneL",
            "forbidden_contact": [],
        },
        "events": (),
        "geom_ms": 0,
        "tick": 0,
        "sim_ms": 0,
        "source": "ground-truth",
    }
    fields.update(over)
    return Reconstruction(**fields)


def robot_block(**over) -> dict:
    block = {
        "ee_pose_mm": [0, 0, 200],
        "ee_quat": [0.0, 0.0, 0.0, 1.0],
        "gripper_mm": 80,
        "holding": None,
        "contact_n": 0.0,
        "speed_mm_s": 0,
        "observed_at_ms": 0,
    }
    block.update(over)
    return block


# --------------------------------------------------------------------------
# 공통 스키마
# --------------------------------------------------------------------------


def test_extract_fills_every_field_of_the_common_schema():
    state = extract(reconstruction(), robot_block(), now_ms=100)
    assert tuple(state) == STATE_FIELDS + ("extractor",)
    assert tuple(state["objects"][0]) == OBJECT_FIELDS


def test_image_and_geom_slots_are_reserved_but_empty():
    """영상·기하 soft token 슬롯은 예약만 한다 (docs/08 §3.2)."""
    state = extract(reconstruction(), robot_block(), now_ms=0)
    assert state["image"] == []
    assert state["geom"] == []


def test_source_wise_ages_are_separate():
    """기하는 10Hz보다 느리게 갱신될 수 있다 — 소스별 나이를 따로 낸다."""
    state = extract(reconstruction(geom_ms=800), robot_block(observed_at_ms=980), now_ms=1000)
    assert state["t"]["age_ms"] == {"geom": 200, "proprio": 20}


def test_extractor_version_is_fixed_and_recorded():
    state = extract(reconstruction(), robot_block(), now_ms=0)
    assert state["extractor"] == EXTRACTOR_VERSION


def test_commitment_and_exec_pass_through_the_robot_block():
    commitment = {"action_ref": "c1", "key": "grasp:o0:top:zoneL:slow", "phase": "approach"}
    state = extract(
        reconstruction(),
        robot_block(commitment=commitment, exec={"seq": 3, "action_ref": "c1"}),
        now_ms=0,
    )
    assert state["commitment"] == commitment
    assert state["exec"]["seq"] == 3


def test_derived_values_are_computed_from_observed_poses():
    """물체별 파생 값: 말단 기준 상대 벡터, 최근접 여유, 통로 폭 (docs/08 §3.2)."""
    recon = reconstruction(instance("o0", pose_mm=(300, 0, -80)), instance("o1", pose_mm=(300, 120, -80)))
    state = extract(recon, robot_block(ee_pose_mm=[0, 0, 200]), now_ms=0)
    first = state["derived"][0]
    assert first["object"] == "o0"
    assert first["relative_mm"] == [300, 0, -280]
    # 두 물체의 외접 반지름 합만큼을 중심 거리에서 뺀 값이다.
    assert 0 < first["clearance_mm"] < 120
    assert first["corridor_mm"] >= 0


# --------------------------------------------------------------------------
# 가림 — 참값은 상태에 들어가지 않는다
# --------------------------------------------------------------------------


def test_occluded_instance_keeps_the_last_seen_pose_and_its_age():
    recon = reconstruction(
        instance("o0", pose_mm=(300, 0, -80), observed_now=False, last_seen_ms=200, visible_ratio=0.1)
    )
    state = extract(recon, robot_block(), now_ms=1000)
    obj = state["objects"][0]
    assert obj["pose_mm"] == [300, 0, -80]  # 마지막으로 **관측된** 자세
    assert obj["last_seen_ms"] == 200
    assert obj["age_ms"] == 800


def observation(**over) -> dict:
    """시뮬레이터 관측 (`Environment._observation`)의 최소 형태."""
    base = {
        "tick": 0,
        "sim_time_ms": 0,
        "instruction": {"version": 1, "t_ms": 0, "text": "빨간 상자를 왼쪽 정리 영역으로 옮겨라"},
        "objects": [
            {
                "id": "o0",
                "class": "box",
                "shape": "box",
                "colour": "red",
                "pos_mm": [300, 0, -80],
                "quat": [0.0, 0.0, 0.0, 1.0],
                "obb_mm": [60, 60, 64],
                "visible": True,
                "visible_ratio": 1.0,
                "attributes": [],
                "last_seen_ms": 300,
            },
            {
                "id": "o1",
                "class": "cylinder",
                "shape": "cylinder",
                "colour": "blue",
                "pos_mm": [120, 200, -80],
                "quat": [0.0, 0.0, 0.0, 1.0],
                "obb_mm": [52, 52, 64],
                "visible": True,
                "visible_ratio": 1.0,
                "attributes": ["fragile"],
                "last_seen_ms": 0,
            },
        ],
        "zones": [{"id": "zoneL", "desc": "왼쪽 정리 영역", "bounds_mm": [-120, 150, 180, 330]}],
        "robot": {
            "ee_pos_mm": [0, 0, 200],
            "ee_quat": [0.0, 0.0, 0.0, 1.0],
            "gripper_mm": 80,
            "holding": None,
            "contact_force_n": 0.0,
            "speed_mm_s": 0,
        },
        "events": [],
        "exec": {"seq": 2, "executor": "HOLD", "action_ref": None, "phase": None},
        "ack": None,
    }
    base.update(over)
    return base


def occluded(source: dict, pos_mm=(999, -999, -80)) -> dict:
    """같은 관측에서 `o1`만 가리고 그 사이 참값이 움직인 사본."""
    hidden = copy.deepcopy(source)
    hidden["objects"][1].update(visible=False, visible_ratio=0.1, pos_mm=list(pos_mm))
    return hidden


def test_ground_truth_adapter_fills_the_same_schema():
    adapter = GroundTruthAdapter(PERCEPTION)
    state = extract(adapter.reconstruct(observation()), adapter.robot(observation()), now_ms=0)
    assert tuple(state) == STATE_FIELDS + ("extractor",)
    assert tuple(state["objects"][0]) == OBJECT_FIELDS


def test_adapter_precision_comes_from_config_not_from_truth():
    adapter = GroundTruthAdapter(PERCEPTION)
    adapter.reconstruct(observation())
    hidden = occluded(observation())
    hidden["sim_time_ms"] = PERCEPTION["geom_period_ms"]
    visible, gone = adapter.reconstruct(hidden).instances
    assert visible.precision_mm == PERCEPTION["precision_mm"]["visible"]
    assert gone.precision_mm == PERCEPTION["precision_mm"]["occluded"]


def test_hidden_truth_of_an_occluded_object_never_reaches_the_state():
    """가려진 물체가 실제로 움직여도 상태는 그대로다 (docs/08 §3.2 정보 경계)."""
    adapter = GroundTruthAdapter(PERCEPTION)
    first = observation()
    before = extract(adapter.reconstruct(first), adapter.robot(first), now_ms=0)

    moved = occluded(first)  # 가려진 동안 참값이 바뀐다
    moved["tick"], moved["sim_time_ms"] = 4, 400
    after = extract(adapter.reconstruct(moved), adapter.robot(moved), now_ms=400)

    assert after["objects"][1]["pose_mm"] == before["objects"][1]["pose_mm"] == [120, 200, -80]
    assert after["objects"][1]["last_seen_ms"] == 0
    assert after["objects"][1]["age_ms"] == 400


def test_hidden_truth_is_available_only_as_evidence():
    adapter = GroundTruthAdapter(PERCEPTION)
    adapter.reconstruct(observation())
    moved = occluded(observation())
    moved["sim_time_ms"] = 400
    state = extract(adapter.reconstruct(moved), adapter.robot(moved), now_ms=400)

    assert adapter.evidence()["occluded_true_poses"]["o1"] == [999, -999, -80]
    # 상태에는 어떤 경로로도 없다.
    assert [999, -999, -80] not in [obj["pose_mm"] for obj in state["objects"]]


def test_an_object_that_was_never_observed_is_not_in_the_state():
    """본 적 없는 물체는 앞단이 모른다 — 참값으로 채우지 않는다."""
    adapter = GroundTruthAdapter(PERCEPTION)
    start = occluded(observation(), pos_mm=(120, 200, -80))
    state = extract(adapter.reconstruct(start), adapter.robot(start), now_ms=0)

    assert [obj["id"] for obj in state["objects"]] == ["o0"]
    assert adapter.evidence()["occluded_true_poses"] == {"o1": [120, 200, -80]}


def test_hidden_attributes_never_reach_the_state():
    """가려진 물체의 속성·크기·설명이 참값에서 바뀌어도 상태는 마지막 관측을 말한다.

    자세만이 아니라 앞단이 관측으로만 알 수 있는 모든 필드가 경계 안에 있어야 한다
    (docs/08 §3.2 "가려진 물체의 참값은 넣지 않는다").
    """
    plain, tampered = GroundTruthAdapter(PERCEPTION), GroundTruthAdapter(PERCEPTION)
    first = observation()
    plain.reconstruct(first)
    tampered.reconstruct(first)

    hidden = occluded(observation(), pos_mm=(120, 200, -80))
    hidden["sim_time_ms"] = 400
    changed = copy.deepcopy(hidden)
    changed["objects"][1].update(
        attributes=["forbidden"], obb_mm=[90, 90, 90], colour="black", desc="검은 원통", shape="box"
    )

    expected = extract(plain.reconstruct(hidden), plain.robot(hidden), now_ms=400)
    actual = extract(tampered.reconstruct(changed), tampered.robot(changed), now_ms=400)
    assert actual == expected
    assert actual["objects"][1]["attributes"] == ["fragile"]
    assert actual["goal"]["forbidden_contact"] == []


def test_goal_references_only_tracked_objects():
    """구조화된 목표는 앞단이 추적하는 물체만 가리킨다 (본 적 없는 금지·취약 물체는 텍스트로만 남는다)."""
    adapter = GroundTruthAdapter(PERCEPTION)
    never_seen = observation()
    never_seen["objects"][1].update(
        visible=False, visible_ratio=0.0, attributes=["forbidden"], desc="파란 원통"
    )
    never_seen["objects"][0].update(visible=False, visible_ratio=0.0, desc="빨간 상자")
    never_seen["instruction"]["text"] = "빨간 상자를 왼쪽 정리 영역으로 옮기고 파란 원통은 건드리지 마라"
    goal = adapter.reconstruct(never_seen).goal

    assert goal["target_ref"] is None  # 아직 보지 못한 대상은 참조할 수 없다
    assert goal["forbidden_contact"] == [] and goal["fragile"] == []
    assert goal["target_zone"] == "zoneL"  # 영역은 작업 공간의 표시라 언제나 안다
    assert "파란 원통" in goal["text"]  # 제약은 지시 텍스트가 계속 나른다

    seen = copy.deepcopy(never_seen)
    seen["sim_time_ms"] = PERCEPTION["geom_period_ms"]
    for entry in seen["objects"]:
        entry.update(visible=True, visible_ratio=1.0)
    goal = adapter.reconstruct(seen).goal
    assert goal["target_ref"] == "o0" and goal["forbidden_contact"] == ["o1"]


def structured_goal(**over) -> dict:
    """`Environment._structured_goal`이 관측에 싣는 형태."""
    goal = {
        "target_ref": "o0",
        "target_desc": "빨간 상자",
        "zone_ref": "zoneL",
        "forbidden_refs": ["o1"],
        "fragile_refs": [],
        "version": 1,
        "text": "빨간 상자를 왼쪽 정리 영역으로 옮기고 파란 원통은 건드리지 마라",
    }
    goal.update(over)
    return goal


def test_a_structured_goal_passes_through_restricted_to_tracked_instances():
    """관측의 구조화된 목표(3c-1)는 텍스트 파싱 없이 상태의 `goal`이 된다 — 단, 참조는 **추적
    중인 물체**로 제한한다(본 적 없는 id는 `target_desc`·텍스트로만 남고, 보이면 채워진다)."""
    adapter = GroundTruthAdapter(PERCEPTION)
    hidden = observation(goal=structured_goal())
    hidden["instruction"]["text"] = hidden["goal"]["text"]
    hidden["objects"][0].update(visible=False, visible_ratio=0.0, desc="빨간 상자")
    hidden["objects"][1].update(visible=False, visible_ratio=0.0, attributes=["forbidden"], desc="파란 원통")
    goal = adapter.reconstruct(hidden).goal

    assert goal["target_ref"] is None  # 아직 보지 못했다
    assert goal["target_desc"] == "빨간 상자"  # 무엇을 찾는지는 안다
    assert goal["target_zone"] == "zoneL"
    assert goal["forbidden_contact"] == [] and goal["fragile"] == []
    assert goal["text"] == hidden["goal"]["text"] and goal["version"] == 1

    seen = copy.deepcopy(hidden)
    seen["sim_time_ms"] = PERCEPTION["geom_period_ms"]
    for entry in seen["objects"]:
        entry.update(visible=True, visible_ratio=1.0)
    goal = adapter.reconstruct(seen).goal
    assert goal["target_ref"] == "o0" and goal["forbidden_contact"] == ["o1"]
    assert goal["target_desc"] == "빨간 상자" and goal["target_zone"] == "zoneL"


def test_the_structured_goal_wins_over_the_text_approximation():
    """텍스트가 마지막으로 부르는 물체와 구조화된 대상이 다르면 구조화된 쪽이다."""
    adapter = GroundTruthAdapter(PERCEPTION)
    scene = observation(goal=structured_goal(target_ref="o0", target_desc="빨간 상자", forbidden_refs=[]))
    scene["objects"][0]["desc"] = "빨간 상자"
    scene["objects"][1].update(desc="파란 원통", attributes=[])
    scene["instruction"]["text"] = "빨간 상자 말고 파란 원통 옆에 있는 것을 왼쪽 정리 영역으로 옮겨라"
    scene["goal"]["text"] = scene["instruction"]["text"]
    goal = adapter.reconstruct(scene).goal
    assert goal["target_ref"] == "o0" and goal["target_desc"] == "빨간 상자"


def test_without_a_structured_goal_the_state_goal_has_no_target_desc():
    """텍스트 근사 경로(D0 fixture 등)는 `target_desc`를 만들지 않는다 — 판단기가 두 경로를 가른다."""
    adapter = GroundTruthAdapter(PERCEPTION)
    goal = adapter.reconstruct(observation()).goal
    assert "target_desc" not in goal


# --------------------------------------------------------------------------
# 사건은 관측된 변위에서만 만든다 (docs/08 §3.2 `events[]`)
# --------------------------------------------------------------------------


def disturbed(source: dict, object_id: str = "o1") -> dict:
    copied = copy.deepcopy(source)
    copied["events"] = [{"kind": "disturbance_applied", "object": object_id, "sim_ms": copied["sim_time_ms"]}]
    return copied


def test_a_hidden_disturbance_changes_nothing_the_model_sees():
    """docs/10 I5: 가려진 물체의 시뮬레이터 외란은 자세·사건·이동·파생 값 어디에도 새지 않는다."""
    plain, hidden_world = GroundTruthAdapter(PERCEPTION), GroundTruthAdapter(PERCEPTION)
    first = observation()
    plain.reconstruct(first)
    hidden_world.reconstruct(first)

    quiet = occluded(observation(), pos_mm=(120, 200, -80))
    quiet["tick"], quiet["sim_time_ms"] = 1, 100
    moved = disturbed(occluded(observation(), pos_mm=(650, 220, -80)))
    moved["tick"], moved["sim_time_ms"] = 1, 100

    expected = extract(plain.reconstruct(quiet), plain.robot(quiet), now_ms=100)
    actual = extract(hidden_world.reconstruct(moved), hidden_world.robot(moved), now_ms=100)

    assert actual["objects"] == expected["objects"]
    assert actual["events"] == expected["events"] == []
    assert actual["derived"] == expected["derived"]
    assert hidden_world.moving("o1") is False
    assert actual == expected
    # 시뮬레이터의 사건은 근거로만 남는다.
    assert hidden_world.evidence()["simulator_events"] == moved["events"]


def test_observed_displacement_becomes_an_object_moved_event():
    adapter = GroundTruthAdapter(PERCEPTION)
    adapter.reconstruct(observation())
    period = PERCEPTION["geom_period_ms"]

    still = observation(sim_time_ms=period)
    still["objects"][1]["pos_mm"] = [121, 201, -80]  # 정밀도 안의 흔들림
    state = extract(adapter.reconstruct(still), adapter.robot(still), now_ms=period)
    assert state["events"] == []
    assert adapter.moving("o1") is False

    shifted = observation(sim_time_ms=2 * period)
    shifted["objects"][1]["pos_mm"] = [160, 230, -80]
    state = extract(adapter.reconstruct(shifted), adapter.robot(shifted), now_ms=2 * period)
    assert [event["kind"] for event in state["events"]] == ["object_moved"]
    event = state["events"][0]
    assert event["object"] == "o1" and event["displacement_mm"] >= PERCEPTION["moved_threshold_mm"]
    assert adapter.moving("o1") is True

    # 사건은 관측된 틱에만 한 번이다. 이동 대상 표시는 창 안에서만 유지된다.
    later = observation(sim_time_ms=3 * period)
    later["objects"][1]["pos_mm"] = [160, 230, -80]
    state = extract(adapter.reconstruct(later), adapter.robot(later), now_ms=3 * period)
    assert state["events"] == []
    assert adapter.moving("o1") is True
    much_later = observation(sim_time_ms=3 * period + PERCEPTION["moving_window_ms"] + period)
    much_later["objects"][1]["pos_mm"] = [160, 230, -80]
    adapter.reconstruct(much_later)
    assert adapter.moving("o1") is False


def test_an_object_that_moved_while_hidden_reports_the_move_when_reobserved():
    adapter = GroundTruthAdapter(PERCEPTION)
    adapter.reconstruct(observation())
    period = PERCEPTION["geom_period_ms"]

    hidden = disturbed(occluded(observation(), pos_mm=(650, 220, -80)))
    hidden["sim_time_ms"] = period
    state = extract(adapter.reconstruct(hidden), adapter.robot(hidden), now_ms=period)
    assert state["events"] == []

    back = observation(sim_time_ms=2 * period)
    back["objects"][1].update(pos_mm=[650, 220, -80], last_seen_ms=2 * period)
    state = extract(adapter.reconstruct(back), adapter.robot(back), now_ms=2 * period)
    moves = [event for event in state["events"] if event["kind"] == "object_moved"]
    assert len(moves) == 1 and moves[0]["object"] == "o1"
    assert state["objects"][1]["pose_mm"] == [650, 220, -80]
    assert state["objects"][1]["reid"] == [f"reacquired:{2 * period}"]

    again = observation(sim_time_ms=3 * period)
    again["objects"][1].update(pos_mm=[650, 220, -80], last_seen_ms=3 * period)
    assert extract(adapter.reconstruct(again), adapter.robot(again), now_ms=3 * period)["events"] == []


def test_the_moved_threshold_is_at_least_the_pose_precision():
    broken = copy.deepcopy(PERCEPTION)
    broken["moved_threshold_mm"] = PERCEPTION["precision_mm"]["visible"] - 1
    with pytest.raises(ValueError, match="moved_threshold_mm"):
        GroundTruthAdapter(broken)


def test_visibility_on_a_stale_geometry_tick_does_not_hide_the_reacquisition():
    """기하가 갱신되지 않은 틱의 가시성은 추적 상태를 바꾸지 않는다 — 다음 갱신 틱에 재식별이 난다."""
    adapter = GroundTruthAdapter(PERCEPTION)
    period = PERCEPTION["geom_period_ms"]
    adapter.reconstruct(observation())
    hidden = occluded(observation(), pos_mm=(120, 200, -80))
    hidden["sim_time_ms"] = period
    adapter.reconstruct(hidden)

    glimpse = observation(sim_time_ms=period + period // 2)  # 기하 갱신 사이의 틱
    assert adapter.reconstruct(glimpse).instances[1].reid == ()

    fresh = observation(sim_time_ms=2 * period)
    fresh["objects"][1]["last_seen_ms"] = 2 * period
    assert adapter.reconstruct(fresh).instances[1].reid == (f"reacquired:{2 * period}",)


def test_graspable_faces_come_from_the_last_fresh_observation():
    """팔이 대상을 가리는 파지 국면에서 파지면이 사라지면 후보가 사라진다 — 가시 비율은 정보다."""
    adapter = GroundTruthAdapter(PERCEPTION)
    adapter.reconstruct(observation())
    covered = observation(sim_time_ms=PERCEPTION["geom_period_ms"])
    covered["objects"][0].update(visible=False, visible_ratio=0.0)
    instance = adapter.reconstruct(covered).instances[0]
    assert "top" in instance.graspable_faces
    assert instance.visible_ratio == 0.0


# --------------------------------------------------------------------------
# 파지 중인 물체 — 자세는 말단에서 (docs/08 §3.2 `objects[]`)
# --------------------------------------------------------------------------


def test_the_work_surface_survives_when_the_only_known_object_is_held():
    """놓인 물체가 없으면 마지막으로 안 작업면을 쓴다 — 들고 있는 물체가 작업면을 말하지는 않는다."""
    adapter = GroundTruthAdapter(PERCEPTION)
    alone = observation()
    alone["objects"] = alone["objects"][:1]
    alone["robot"].update(holding="o0", ee_pos_mm=[300, 0, 0])
    first = adapter.reconstruct(alone)
    assert first.scene.work_surface_mm == -80 - 32  # 첫 관측에서는 아직 놓여 있던 자세다

    lifted = copy.deepcopy(alone)
    lifted["sim_time_ms"] = PERCEPTION["geom_period_ms"]
    lifted["robot"]["ee_pos_mm"] = [300, 0, 100]
    lifted["objects"][0]["pos_mm"] = [300, 0, 20]
    assert adapter.reconstruct(lifted).scene.work_surface_mm == -80 - 32


def test_a_held_object_takes_its_pose_from_the_end_effector():
    adapter = GroundTruthAdapter(PERCEPTION)
    adapter.reconstruct(observation())
    period = PERCEPTION["geom_period_ms"]

    grasped = observation(sim_time_ms=period)
    grasped["robot"].update(ee_pos_mm=[300, 0, -70], holding="o0")
    grasped["objects"][0].update(visible=False, visible_ratio=0.0)
    adapter.reconstruct(grasped)

    lifted = observation(sim_time_ms=period + 100)
    lifted["robot"].update(ee_pos_mm=[300, 0, 50], holding="o0")
    lifted["objects"][0].update(visible=False, visible_ratio=0.0, pos_mm=[999, 999, 999])
    state = extract(adapter.reconstruct(lifted), adapter.robot(lifted), now_ms=period + 100)
    held = state["objects"][0]
    assert held["pose_mm"][:2] == [300, 0]
    assert held["pose_mm"][2] == 50 - (-70 - (-80))  # 파지 시점의 말단 기준 오프셋을 유지한다
    assert held["age_ms"] == 0 and held["last_seen_ms"] == period + 100
    assert held["pose_source"] == "proprio"
    assert state["events"] == []  # 들고 움직인 것은 외란이 아니다
    assert adapter.moving("o0") is False
    assert state["objects"][1]["pose_source"] == "geom"


def test_adapter_remembers_the_pose_it_last_saw():
    """다시 보이면 새 자세를 쓰고, 다시 가려지면 그 자세를 유지한다."""
    adapter = GroundTruthAdapter(PERCEPTION)
    adapter.reconstruct(observation())

    seen = observation()
    seen["objects"][1].update(pos_mm=[140, 210, -80], last_seen_ms=400)
    seen["sim_time_ms"] = 400
    adapter.reconstruct(seen)
    assert adapter.reconstruct(seen).instances[1].reid == ()

    hidden = occluded(seen, pos_mm=(500, 500, -80))
    hidden["sim_time_ms"] = 500
    state = extract(adapter.reconstruct(hidden), adapter.robot(hidden), now_ms=500)
    assert state["objects"][1]["pose_mm"] == [140, 210, -80]
    assert state["objects"][1]["last_seen_ms"] == 400


def test_reacquiring_a_track_is_reported_as_a_reid_event():
    adapter = GroundTruthAdapter(PERCEPTION)
    adapter.reconstruct(observation())
    hidden = occluded(observation())
    hidden["sim_time_ms"] = 400
    adapter.reconstruct(hidden)

    back = observation()
    back["sim_time_ms"] = 800
    back["objects"][1]["last_seen_ms"] = 800
    assert adapter.reconstruct(back).instances[1].reid == ("reacquired:800",)


def test_geometry_updates_slower_than_proprioception():
    """3D 재구성은 설정된 주기로만 갱신된다 (docs/08 §3.2 기하 5Hz 시작값)."""
    adapter = GroundTruthAdapter(PERCEPTION)
    period = PERCEPTION["geom_period_ms"]
    first = adapter.reconstruct(observation(sim_time_ms=0))
    assert first.geom_ms == 0

    early = adapter.reconstruct(observation(sim_time_ms=period // 2))
    assert early.geom_ms == 0  # 아직 새 기하가 오지 않았다

    late = adapter.reconstruct(observation(sim_time_ms=period))
    assert late.geom_ms == period


def test_reconstruction_is_a_value_and_does_not_hold_the_observation():
    adapter = GroundTruthAdapter(PERCEPTION)
    source = observation()
    recon = adapter.reconstruct(source)
    source["objects"][0]["pos_mm"] = [0, 0, 0]
    assert recon.instances[0].pose_mm == (300, 0, -80)


def test_an_unknown_source_age_is_refused():
    with pytest.raises(ValueError, match="관측 시각"):
        extract(reconstruction(geom_ms=500), robot_block(observed_at_ms=500), now_ms=100)
