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
