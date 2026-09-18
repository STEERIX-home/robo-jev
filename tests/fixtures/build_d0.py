#!/usr/bin/env python3
"""D0 fixture 빌더.

    uv run python tests/fixtures/build_d0.py

세 파일을 만든다.

* ``d0.jsonl``          — 단일 요청 64건 (`judgment-v0`). 0번 줄은 choice + valid_set으로 고정.
* ``d0_streams.jsonl``  — 에피소드 스트림 4건 (`stream-v0`), 10Hz 약 100틱씩.
* ``d0_manifest.json``  — 두 파일의 sha256, 건수, 빌더 버전, 검수자 목록.

D0는 계약 검사를 위한 **손으로 읽을 수 있는** 작은 표본이다. 물리 시뮬레이터는 쓰지
않는다. 스트림은 아래의 결정적인 장난감 시나리오(직선 보간으로 움직이는 말단, mm
정수 자세, 각본대로 답하는 "전문가")로 만든다. 표준 라이브러리만 쓰고, 같은 씨앗에서
언제나 같은 바이트를 낸다. 말단은 한 틱에 축마다 ``STEP_MM``을 넘게 움직이지 않는다
(파지 안착까지 포함해서).

**명령과 관측을 가른다.** 각본의 ``Segment.gripper``는 *명령*이고 ``exec_history``와
``q_gripper`` 라벨·``provenance.marks.gripper_transition_ticks``가 그것을 쓴다.
``state.robot.gripper_mm``은 *관측*이라 :func:`can_grasp`로 판정한 실제 파지 여부에서
나온다 (docs/08 §3.2 "실제 관측 또는 선언한 추정치만"). 그래서 파지 명령 틱과 관측
그리퍼가 닫히는 틱은 몇 틱 어긋난다 — 말단이 물체에 닿아야 닫힌다.

사람 검수는 아직 끝나지 않았다. manifest의 ``reviewed_by``가 비어 있으면 검수 전이다.
"""

from __future__ import annotations

import copy
import hashlib
import json
import random
from dataclasses import dataclass, field
from pathlib import Path

BUILDER_VERSION = "d0-builder-v0.1.0"
SEED = 20260918
FIXTURE_DIR = Path(__file__).resolve().parent

#: 질문 세트 v0 (docs/08 §4). `robo_jev.contracts.QUESTION_SET_V0`와 같아야 하며,
#: fixture를 검사 대상 코드에서 떼어 두려고 여기 다시 적는다. 두 목록이 같은지는
#: tests/test_d0_fixture.py가 확인한다.
QUESTION_IDS_V0 = (
    "q_main",
    "q_done",
    "q_instr",
    "q_observe",
    "q_retry",
    "q_stop",
    "q_gripper",
    "q_path",
    "q_speed",
    "q_force",
)

#: 장면 계열마다 split을 미리 배정한다 (docs/08 §8: 생성 전 배정, 계열 단위).
SPLIT_CYCLE = ("train", "train", "train", "train", "dev", "calibration", "test", "ood")

_KO_COLOR = {"red": "빨간", "blue": "파란", "green": "초록", "yellow": "노란"}
_EN_COLOR = {"red": "red", "blue": "blue", "green": "green", "yellow": "yellow"}

UNKNOWN_CANDIDATE_ID = "c_na"
UNKNOWN_CANDIDATE_KO = "제공된 정보로 대상을 결정할 수 없음"


# ==========================================================================
# 단일 요청 64건
# ==========================================================================


def _observed_object(oid: str, color: str, x: int, y: int, *, visible: bool = True) -> dict:
    return {
        "id": oid,
        "desc": f"{_KO_COLOR[color]} 물체 {oid}",
        "color": color,
        "pose_mm": [x, y, 742],
        "pose_sigma_mm": 3,
        "visible": visible,
        "visible_ratio": 1.0 if visible else 0.0,
    }


def _record(
    *,
    index: int,
    group: str,
    state: dict,
    questions: list[dict],
    labels: list[dict],
    evidence: dict,
    rules: str,
) -> dict:
    return {
        "schema_version": "judgment-v0",
        "origin_group": group,
        "split": SPLIT_CYCLE[index % len(SPLIT_CYCLE)],
        "request": {
            "request_id": f"d0-{index:03d}",
            "state": state,
            "questions": questions,
        },
        "labels": labels,
        "provenance": {
            "origin_group": group,
            "generator": BUILDER_VERSION,
            "seed": SEED,
            "rules": rules,
            "expert": "scripted-e0",
        },
        "evidence": evidence,
        "usage": {"questions_used": [q["id"] for q in questions], "commands": []},
    }


def _target_object_cases(rng: random.Random) -> list[dict]:
    """A 계열 (12건): choice + valid_set.

    유일 정답 7건, 복수 정답 3건, "정보 부족" 후보가 정답 2건.
    """
    scenes = [
        ("red", [("o1", "red"), ("o2", "blue")]),
        ("blue", [("o3", "blue"), ("o4", "green")]),
        ("green", [("o5", "green"), ("o6", "red"), ("o7", "blue")]),
        ("yellow", [("o8", "yellow"), ("o9", "red")]),
        ("red", [("o1", "red"), ("o2", "red")]),
        ("blue", [("o3", "blue"), ("o4", "blue"), ("o5", "green")]),
        ("green", [("o6", "green"), ("o7", "green")]),
        ("red", [("o2", "blue"), ("o4", "green")]),
        ("yellow", [("o1", "red"), ("o3", "blue"), ("o5", "green")]),
        ("red", [("o1", "red"), ("o2", "blue"), ("o4", "green"), ("o6", "yellow")]),
        ("blue", [("o2", "blue"), ("o5", "green"), ("o8", "yellow")]),
        ("green", [("o5", "green"), ("o1", "red")]),
    ]
    records = []
    for case_index, (goal_color, objects) in enumerate(scenes):
        observed = [
            _observed_object(oid, color, rng.randrange(-400, 400, 5), rng.randrange(-300, 300, 5))
            for oid, color in objects
        ]
        criteria = [
            {"id": f"c{position}", "description": f"관측된 {obj['desc']}", "ref": obj["id"]}
            for position, obj in enumerate(observed)
        ]
        criteria.append({"id": UNKNOWN_CANDIDATE_ID, "description": UNKNOWN_CANDIDATE_KO})
        matching = [
            criterion["id"]
            for criterion, obj in zip(criteria, observed)
            if obj["color"] == goal_color
        ]
        answer = matching or [UNKNOWN_CANDIDATE_ID]

        evidence = {
            "rule_trace": f"goal-object-rule-v0: goal_color={goal_color}, matched={answer}",
            # 가려진 참값은 evidence에만 둔다. 이 값만 바꿔도 모델 입력은 변하지 않아야 한다.
            "true_state": {obj["id"]: [obj["pose_mm"][0] + 2, obj["pose_mm"][1] - 3, 741] for obj in observed},
        }
        if not matching:
            evidence["occluded_true_poses"] = {"o0": [rng.randrange(-400, 400, 5), 250, 741]}

        records.append(
            _record(
                index=case_index,
                group=f"scene-family-a{case_index:02d}",
                state={
                    "goal": f"{_KO_COLOR[goal_color]} 물체를 왼쪽 영역으로 옮긴다",
                    "observed_at_ms": 1200 + 10 * case_index,
                    "objects": observed,
                    "zones": [{"id": "zoneL", "desc": "왼쪽 영역", "bounds_mm": [-500, -200, -100, 200]}],
                },
                questions=[
                    {
                        "id": "q_target",
                        "type": "choice",
                        "instructions": "현재 목표에서 옮겨야 하는 물체를 고르라.",
                        "criteria": criteria,
                    }
                ],
                labels=[
                    {
                        "question_id": "q_target",
                        "kind": "valid_set",
                        "candidate_ids": answer,
                        "source": "goal-object-rule-v0",
                        "mask": True,
                    }
                ],
                evidence=evidence,
                rules="goal-object-rule-v0",
            )
        )
    return records


def _goal_done_cases(rng: random.Random, start: int) -> list[dict]:
    """B 계열 (8건): boolean + single. 목표 달성 여부."""
    records = []
    for case_index in range(8):
        done = case_index % 2 == 0
        x = -300 if done else 260
        obj = _observed_object("o7", "red", x, rng.randrange(-100, 100, 5))
        records.append(
            _record(
                index=start + case_index,
                group=f"scene-family-b{case_index:02d}",
                state={
                    "goal": "빨간 컵이 왼쪽 영역 안에 있어야 한다",
                    "observed_at_ms": 800 + 10 * case_index,
                    "objects": [obj],
                    "zones": [{"id": "zoneL", "desc": "왼쪽 영역", "bounds_mm": [-500, -200, -100, 200]}],
                },
                questions=[
                    {
                        "id": "q_done",
                        "type": "boolean",
                        "instructions": "현재 목표를 이미 만족했는가.",
                        "criteria": [
                            {"id": "true", "description": "목표 조건을 만족한다"},
                            {"id": "false", "description": "아직 만족하지 않는다"},
                        ],
                    }
                ],
                labels=[
                    {
                        "question_id": "q_done",
                        "kind": "single",
                        "answer": done,
                        "source": "goal-evaluator-v0",
                    }
                ],
                evidence={"rule_trace": f"goal-evaluator-v0: x={x}mm, zoneL=[-500,-100]"},
                rules="goal-evaluator-v0",
            )
        )
    return records


def _instruction_sufficiency_cases(start: int) -> list[dict]:
    """C 계열 (6건): boolean + single, 영어 지시문."""
    scenes = [
        ("Move the red cup to the left zone.", True, [("o7", "red"), ("o2", "blue")]),
        ("Move it over there.", False, [("o7", "red"), ("o2", "blue")]),
        ("Push the green box until it touches the wall.", True, [("o5", "green")]),
        ("Tidy up.", False, [("o5", "green"), ("o1", "red")]),
        ("Place the yellow block into the left zone without touching the glass.", True, [("o8", "yellow")]),
        ("Handle the fragile one carefully.", False, [("o8", "yellow"), ("o3", "blue")]),
    ]
    records = []
    for case_index, (instruction, sufficient, objects) in enumerate(scenes):
        observed = [
            {
                "id": oid,
                "desc": f"{_EN_COLOR[color]} object {oid}",
                "color": color,
                "pose_mm": [120 + 40 * position, -60 + 30 * position, 742],
                "pose_sigma_mm": 3,
                "visible": True,
                "visible_ratio": 1.0,
            }
            for position, (oid, color) in enumerate(objects)
        ]
        records.append(
            _record(
                index=start + case_index,
                group=f"scene-family-c{case_index:02d}",
                state={
                    "goal": instruction,
                    "observed_at_ms": 600 + 10 * case_index,
                    "objects": observed,
                    "zones": [{"id": "zoneL", "desc": "left zone", "bounds_mm": [-500, -200, -100, 200]}],
                },
                questions=[
                    {
                        "id": "q_instr",
                        "type": "boolean",
                        "instructions": "Is the instruction complete enough to execute right now?",
                        "criteria": [
                            {"id": "true", "description": "Target, goal and constraints are all determined"},
                            {"id": "false", "description": "Something needed for execution is missing"},
                        ],
                    }
                ],
                labels=[
                    {
                        "question_id": "q_instr",
                        "kind": "single",
                        "answer": sufficient,
                        "source": "instruction-completeness-rule-v0",
                    }
                ],
                evidence={"rule_trace": f"instruction-completeness-rule-v0: sufficient={sufficient}"},
                rules="instruction-completeness-rule-v0",
            )
        )
    return records


def _speed_level_criteria() -> list[dict]:
    return [
        {"id": "0", "description": "정지 (0 m/s)", "value": 0.0},
        {"id": "1", "description": "저속 (0.1 m/s)", "value": 0.1},
        {"id": "2", "description": "중속 (0.25 m/s)", "value": 0.25},
        {"id": "3", "description": "고속 (0.5 m/s)", "value": 0.5},
    ]


def _speed_level_cases(start: int) -> list[dict]:
    """D 계열 (8건): ordinal + valid_set. 경계 구간은 인접 수준을 함께 허용한다."""
    scenes = [
        (420, False, ["3"]),
        (300, False, ["2", "3"]),
        (180, False, ["2"]),
        (120, True, ["1"]),
        (90, True, ["0", "1"]),
        (250, True, ["1", "2"]),
        (60, True, ["0"]),
        (500, False, ["3"]),
    ]
    records = []
    for case_index, (clearance_mm, fragile_near, answer) in enumerate(scenes):
        records.append(
            _record(
                index=start + case_index,
                group=f"scene-family-d{case_index:02d}",
                state={
                    "goal": "빨간 컵을 왼쪽 영역으로 옮기는 중이다",
                    "observed_at_ms": 2000 + 10 * case_index,
                    "scene": {"nearest_clearance_mm": clearance_mm, "corridor_width_mm": clearance_mm + 80},
                    "objects": [
                        {
                            "id": "o3",
                            "desc": "유리잔",
                            "pose_mm": [clearance_mm, 40, 742],
                            "pose_sigma_mm": 4,
                            "visible": True,
                            "visible_ratio": 0.9,
                            "attributes": ["fragile"] if fragile_near else [],
                        }
                    ],
                    "robot": {"ee_pose_mm": [0, 0, 880], "gripper_mm": 20, "holding": "o7"},
                },
                questions=[
                    {
                        "id": "q_speed",
                        "type": "ordinal",
                        "instructions": "지금 명령할 속도 수준을 고르라.",
                        "criteria": _speed_level_criteria(),
                    }
                ],
                labels=[
                    {
                        "question_id": "q_speed",
                        "kind": "valid_set",
                        "candidate_ids": answer,
                        "source": "expert-speed-profile-v0",
                    }
                ],
                evidence={
                    "rule_trace": (
                        f"expert-speed-profile-v0: clearance={clearance_mm}mm, fragile_near={fragile_near}"
                    )
                },
                rules="expert-speed-profile-v0",
            )
        )
    return records


def _force_level_cases(start: int) -> list[dict]:
    """E 계열 (6건): ordinal + single."""
    scenes = [
        ("approach", "0"),
        ("grasp", "1"),
        ("push", "2"),
        ("transport", "0"),
        ("place", "1"),
        ("push", "2"),
    ]
    records = []
    for case_index, (phase, answer) in enumerate(scenes):
        records.append(
            _record(
                index=start + case_index,
                group=f"scene-family-e{case_index:02d}",
                state={
                    "goal": "상자를 왼쪽 영역으로 밀거나 옮긴다",
                    "observed_at_ms": 3000 + 10 * case_index,
                    "phase": phase,
                    "robot": {"ee_pose_mm": [100, 0, 820], "contact_n": 0.0, "gripper_mm": 60},
                },
                questions=[
                    {
                        "id": "q_force",
                        "type": "ordinal",
                        "instructions": f"국면 {phase}에서 목표 접촉·힘 수준을 고르라.",
                        "criteria": [
                            {"id": "0", "description": "회피", "value": 0.0},
                            {"id": "1", "description": "가벼운 접촉", "value": 1.0},
                            {"id": "2", "description": "밀기", "value": 2.0},
                        ],
                    }
                ],
                labels=[
                    {
                        "question_id": "q_force",
                        "kind": "single",
                        "answer": answer,
                        "source": "expert-contact-mode-v0",
                    }
                ],
                evidence={"rule_trace": f"expert-contact-mode-v0: phase={phase}"},
                rules="expert-contact-mode-v0",
            )
        )
    return records


def _grasp_event_cases(rng: random.Random, start: int) -> list[dict]:
    """F 계열 (6건): 물리 성공 확률을 묻는 Bernoulli 질문 + event 라벨."""
    records = []
    for case_index in range(6):
        successes = rng.randint(1, 8)
        failures = 8 - successes
        censored = 1 if case_index % 3 == 0 else 0
        face = ("top", "side")[case_index % 2]
        records.append(
            _record(
                index=start + case_index,
                group=f"scene-family-f{case_index:02d}",
                state={
                    "goal": "빨간 컵을 집는다",
                    "observed_at_ms": 4000 + 10 * case_index,
                    "objects": [
                        {
                            "id": "o7",
                            "desc": "빨간 컵",
                            "pose_mm": [200 + 10 * case_index, -30, 742],
                            "pose_sigma_mm": 5,
                            "visible": True,
                            "visible_ratio": 0.75,
                            "graspable_faces": ["top", "side"],
                        }
                    ],
                },
                questions=[
                    {
                        "id": "q_grasp_success",
                        "type": "boolean",
                        "instructions": f"o7을 {face} 면으로 집으면 이번 시도가 성공하는가.",
                        "criteria": [
                            {"id": "true", "description": "성공한다"},
                            {"id": "false", "description": "실패한다"},
                        ],
                    }
                ],
                labels=[
                    {
                        "question_id": "q_grasp_success",
                        "kind": "event",
                        "event_id": f"grasp-o7-{face}-{case_index:02d}",
                        "action_ref": f"grasp:o7:{face}:zoneL:slow",
                        "successes": successes,
                        "failures": failures,
                        "censored": censored,
                        "source": "rollout-v0",
                    }
                ],
                evidence={
                    "rollouts": {
                        "seeds": [1] * successes + [0] * failures,
                        "horizon_ms": 5000,
                        "censor_reason": "시간 초과" if censored else None,
                    }
                },
                rules="rollout-v0",
            )
        )
    return records


def _ambiguous_zone_cases(rng: random.Random, start: int) -> list[dict]:
    """G 계열 (6건): choice + distribution."""
    weight_table = [
        [5, 3, 2],
        [7, 2, 1],
        [4, 4, 2],
        [6, 3, 1],
        [3, 3, 4],
        [8, 1, 1],
    ]
    records = []
    for case_index, weights in enumerate(weight_table):
        zones = [
            {"id": "z1", "description": "왼쪽 위 구획"},
            {"id": "z2", "description": "왼쪽 아래 구획"},
            {"id": "z3", "description": "통로 옆 구획"},
        ]
        total = sum(weights)
        probabilities = {zone["id"]: round(weight / total, 3) for zone, weight in zip(zones, weights)}
        # 반올림 오차는 가장 큰 항에 흡수시켜 합을 1로 맞춘다.
        largest = max(probabilities, key=lambda key: probabilities[key])
        probabilities[largest] = round(probabilities[largest] + (1.0 - sum(probabilities.values())), 6)
        records.append(
            _record(
                index=start + case_index,
                group=f"scene-family-g{case_index:02d}",
                state={
                    "goal": "컵을 왼쪽 영역 어딘가에 놓는다",
                    "observed_at_ms": 5000 + 10 * case_index,
                    "objects": [_observed_object("o7", "red", rng.randrange(-200, 200, 5), 0)],
                    "zones": [{"id": zone["id"], "desc": zone["description"]} for zone in zones],
                },
                questions=[
                    {
                        "id": "q_place_zone",
                        "type": "choice",
                        "instructions": "전문가가 놓을 구획의 분포를 고르라.",
                        "criteria": zones,
                    }
                ],
                labels=[
                    {
                        "question_id": "q_place_zone",
                        "kind": "distribution",
                        "probabilities": probabilities,
                        "source": "expert-demo-frequency-v0",
                    }
                ],
                evidence={"rule_trace": f"expert-demo-frequency-v0: counts={weights}"},
                rules="expert-demo-frequency-v0",
            )
        )
    return records


def _partial_label_cases(rng: random.Random, start: int) -> list[dict]:
    """H 계열 (6건): 라벨 결측(loss mask). 세 질문 중 하나는 라벨이 아예 없다."""
    records = []
    for case_index in range(6):
        objects = [
            _observed_object("o7", "red", rng.randrange(-300, 300, 5), 0),
            _observed_object("o3", "blue", rng.randrange(-300, 300, 5), 120),
        ]
        criteria = [
            {"id": "c0", "description": f"관측된 {objects[0]['desc']}", "ref": "o7"},
            {"id": "c1", "description": f"관측된 {objects[1]['desc']}", "ref": "o3"},
            {"id": UNKNOWN_CANDIDATE_ID, "description": UNKNOWN_CANDIDATE_KO},
        ]
        records.append(
            _record(
                index=start + case_index,
                group=f"scene-family-h{case_index:02d}",
                state={
                    "goal": "빨간 물체를 왼쪽 영역으로 옮긴다",
                    "observed_at_ms": 6000 + 10 * case_index,
                    "objects": objects,
                },
                questions=[
                    {
                        "id": "q_target",
                        "type": "choice",
                        "instructions": "현재 목표에서 옮겨야 하는 물체를 고르라.",
                        "criteria": criteria,
                    },
                    {
                        "id": "q_done",
                        "type": "boolean",
                        "instructions": "현재 목표를 이미 만족했는가.",
                        "criteria": [
                            {"id": "true", "description": "만족한다"},
                            {"id": "false", "description": "아직 아니다"},
                        ],
                    },
                    {
                        "id": "q_speed",
                        "type": "ordinal",
                        "instructions": "지금 명령할 속도 수준을 고르라.",
                        "criteria": _speed_level_criteria(),
                    },
                ],
                # q_speed 라벨은 근거가 없어 아예 넣지 않는다 (loss mask).
                # q_done 라벨은 mask=false로 두어 손실에서 빠진다.
                labels=[
                    {
                        "question_id": "q_target",
                        "kind": "valid_set",
                        "candidate_ids": ["c0"],
                        "source": "goal-object-rule-v0",
                        "mask": True,
                    },
                    {
                        "question_id": "q_done",
                        "kind": "single",
                        "answer": False,
                        "source": "goal-evaluator-v0",
                        "mask": False,
                    },
                ],
                evidence={"rule_trace": "goal-object-rule-v0; q_speed는 근거 없음 → 마스크"},
                rules="goal-object-rule-v0",
            )
        )
    return records


def _multi_question_cases(rng: random.Random, start: int) -> list[dict]:
    """I 계열 (6건): 한 레코드에 세 타입을 모두 담는다."""
    records = []
    for case_index in range(6):
        two_valid = case_index % 2 == 0
        objects = [_observed_object("o7", "red", rng.randrange(-300, 300, 5), -40)]
        if two_valid:
            objects.append(_observed_object("o9", "red", rng.randrange(-300, 300, 5), 90))
        else:
            objects.append(_observed_object("o2", "blue", rng.randrange(-300, 300, 5), 90))
        criteria = [
            {"id": f"c{position}", "description": f"관측된 {obj['desc']}", "ref": obj["id"]}
            for position, obj in enumerate(objects)
        ]
        criteria.append({"id": UNKNOWN_CANDIDATE_ID, "description": UNKNOWN_CANDIDATE_KO})
        target_answer = ["c0", "c1"] if two_valid else ["c0"]
        records.append(
            _record(
                index=start + case_index,
                group=f"scene-family-i{case_index:02d}",
                state={
                    "goal": "빨간 물체를 왼쪽 영역으로 옮긴다",
                    "observed_at_ms": 7000 + 10 * case_index,
                    "objects": objects,
                    "robot": {"ee_pose_mm": [0, 0, 900], "gripper_mm": 80, "holding": None},
                },
                questions=[
                    {
                        "id": "q_target",
                        "type": "choice",
                        "instructions": "현재 목표에서 옮겨야 하는 물체를 고르라.",
                        "criteria": criteria,
                    },
                    {
                        "id": "q_done",
                        "type": "boolean",
                        "instructions": "현재 목표를 이미 만족했는가.",
                        "criteria": [
                            {"id": "true", "description": "만족한다"},
                            {"id": "false", "description": "아직 아니다"},
                        ],
                    },
                    {
                        "id": "q_speed",
                        "type": "ordinal",
                        "instructions": "지금 명령할 속도 수준을 고르라.",
                        "criteria": _speed_level_criteria(),
                    },
                ],
                labels=[
                    {
                        "question_id": "q_target",
                        "kind": "valid_set",
                        "candidate_ids": target_answer,
                        "source": "goal-object-rule-v0",
                        "mask": True,
                    },
                    {"question_id": "q_done", "kind": "single", "answer": False, "source": "goal-evaluator-v0"},
                    {
                        "question_id": "q_speed",
                        "kind": "valid_set",
                        "candidate_ids": ["1", "2"],
                        "source": "expert-speed-profile-v0",
                    },
                ],
                evidence={"rule_trace": f"복수 정답={two_valid}"},
                rules="goal-object-rule-v0+goal-evaluator-v0+expert-speed-profile-v0",
            )
        )
    return records


def single_request_records() -> list[dict]:
    """단일 요청 64건. 0번 줄은 choice + valid_set이다."""
    rng = random.Random(SEED)
    records: list[dict] = []
    records += _target_object_cases(rng)
    records += _goal_done_cases(rng, len(records))
    records += _instruction_sufficiency_cases(len(records))
    records += _speed_level_cases(len(records))
    records += _force_level_cases(len(records))
    records += _grasp_event_cases(rng, len(records))
    records += _ambiguous_zone_cases(rng, len(records))
    records += _partial_label_cases(rng, len(records))
    records += _multi_question_cases(rng, len(records))
    return records


# ==========================================================================
# 에피소드 스트림 4건
# ==========================================================================


#: 가려진 물체의 관측 가시 비율. 참값은 `evidence.occluded_true_poses`에만 둔다.
OCCLUDED_VISIBLE_RATIO = 0.35


@dataclass(frozen=True)
class SceneObject:
    id: str
    desc: str
    pose: tuple[int, int, int]
    graspable_faces: tuple[str, ...] = ()
    pushable: bool = False
    fragile: bool = False
    #: 가려진 물체만 갖는다. 관측 자세와 다른 **참값**이며 모델 입력에는 절대 넣지 않는다.
    occluded_true_pose: tuple[int, int, int] | None = None

    @property
    def visible_ratio(self) -> float:
        """가려진 물체만 부분 관측이다. 두 필드가 따로 놀지 않게 여기서 파생한다."""
        return OCCLUDED_VISIBLE_RATIO if self.occluded_true_pose is not None else 1.0


@dataclass(frozen=True)
class Segment:
    """틱 구간. `action_key`가 None이면 그 구간에는 commitment가 없다."""

    start: int
    phase: str
    gripper: str
    action_key: str | None


@dataclass(frozen=True)
class EpisodeScript:
    episode_id: str
    origin_group: str
    split: str
    instructions: list[dict]
    objects: list[SceneObject]
    target: str
    zone: str
    n_ticks: int
    segments: list[Segment]
    keyframes: tuple[int, ...]
    stop_ticks: tuple[int, ...] = ()
    retry_from: int | None = None
    instr_false_until: int = 0
    stale_ticks: tuple[int, ...] = ()
    contrast_at: int | None = None
    observe_adopt_ticks: tuple[int, ...] = ()
    failure_ticks: tuple[int, ...] = ()
    notes: list[str] = field(default_factory=list)


ZONE_CENTER = {"zoneL": (-320, 0), "zoneR": (330, 0)}
STEP_MM = 25
SURFACE_Z_MM = 742  # 작업면 위의 물체 높이. 말단 목표는 물체 자세가 아니라 이 값을 기준으로 잡는다.
GEOM_PERIOD_MS = 100  # 3D 재구성이 한 틱 뒤처져 들어온다 (docs/08 §3.2)
STALE_OBSERVE_MS = 300  # 기하 나이가 이보다 크면 관측 게이트를 켠다

GRIPPER_OPEN_MM = 80
GRIPPER_CLOSED_MM = 20
#: 들고 있는 물체가 말단 아래 매달리는 거리. 파지 시 말단 높이 = 물체 z + 이 값.
CARRY_OFFSET_MM = 20
#: 물체 높이 (`obb_mm[2]`). 파지 판정의 높이 상한을 물체 윗면에서 잰다.
OBJECT_HEIGHT_MM = 95
#: 파지 판정 여유. 말단이 물체 윗면 + 이 값보다 위에 있으면 아직 잡을 수 없다.
GRASP_CLEARANCE_MM = 30
#: 물체를 들고 있을 수 있는 국면.
CARRY_PHASES = ("grasp", "lift", "transport", "place")


class CandidateIds:
    """의미 키 → 후보 id. 에피소드 안에서 같은 키는 항상 같은 id를 받는다."""

    def __init__(self) -> None:
        self._by_key: dict[str, str] = {}

    def id_for(self, key: str) -> str:
        if key not in self._by_key:
            self._by_key[key] = f"c{len(self._by_key) + 1}"
        return self._by_key[key]


def _step_toward(current: list[int], goal: tuple[int, int, int]) -> list[int]:
    moved = []
    for axis_now, axis_goal in zip(current, goal):
        delta = axis_goal - axis_now
        if abs(delta) <= STEP_MM:
            moved.append(axis_goal)
        else:
            moved.append(axis_now + (STEP_MM if delta > 0 else -STEP_MM))
    return moved


#: 밀기 국면에서 말단이 대상 뒤에 서는 접촉 여유 (mm).
PUSH_CONTACT_MM = 60
#: `+y` 밀기의 목표 y (통로 옆으로 치운다).
PUSH_ASIDE_Y_MM = 200


def _push_contact_offset(push_axis: str | None) -> tuple[int, int]:
    if push_axis == "+x":
        return (PUSH_CONTACT_MM, 0)
    if push_axis == "+y":
        return (0, PUSH_CONTACT_MM)
    return (0, 0)


def _ee_goal(
    phase: str,
    target_pose: list[int],
    zone: str,
    contact_offset: tuple[int, int] = (0, 0),
    push_axis: str | None = None,
) -> tuple[int, int, int]:
    """국면별 말단 목표점. 높이는 작업면 기준이라 들고 있는 물체 자세에 끌려가지 않는다.

    `contact_offset`은 밀기 대상처럼 말단이 물체 뒤에 서야 하는 경우의 접촉 여유다.
    """
    zone_x, zone_y = ZONE_CENTER[zone]
    if phase == "approach":
        return (
            target_pose[0] - contact_offset[0],
            target_pose[1] - contact_offset[1],
            SURFACE_Z_MM + 140,
        )
    if phase == "grasp":
        return (target_pose[0], target_pose[1], SURFACE_Z_MM + 20)
    if phase == "lift":
        return (target_pose[0], target_pose[1], SURFACE_Z_MM + 200)
    if phase == "transport":
        return (zone_x, zone_y, SURFACE_Z_MM + 200)
    if phase == "place":
        return (zone_x, zone_y, SURFACE_Z_MM + 60)
    if phase == "push":
        # 목표점은 고정값이어야 한다. 밀려서 움직이는 대상 자세를 목표로 삼으면 끝없이 밀게 된다.
        goal_x, goal_y = (zone_x, PUSH_ASIDE_Y_MM) if push_axis == "+y" else (zone_x, zone_y)
        return (goal_x - contact_offset[0], goal_y - contact_offset[1], SURFACE_Z_MM + 20)
    return (target_pose[0], target_pose[1], SURFACE_Z_MM + 200)  # none: 그대로 둔다


def _distance_mm(a: list[int] | tuple[int, ...], b: list[int] | tuple[int, ...]) -> int:
    return int(round(sum((int(p) - int(q)) ** 2 for p, q in zip(a, b)) ** 0.5))


def grasp_pose(pose: list[int]) -> tuple[int, int, int]:
    """물체를 잡은 순간의 말단 자세. 물체는 여기서 `CARRY_OFFSET_MM`만큼 아래에 있다."""
    return (pose[0], pose[1], pose[2] + CARRY_OFFSET_MM)


def can_grasp(ee: list[int], pose: list[int]) -> bool:
    """말단이 물체에 닿았는가 — 해제 쪽 검사와 대칭인 파지 쪽 검사.

    말단이 파지 자세에서 한 걸음 안에 들어왔고(세 축 모두) 물체 윗면 가까이 내려왔을
    때만 파지가 성립한다. 이 검사가 없으면 그리퍼가 닫히는 순간 멀리 있는 물체가
    말단으로 순간이동한다. 관측 그리퍼 열림(`gripper_mm`)도 이 판정에서 나온다.
    """
    within_one_step = all(abs(a - b) <= STEP_MM for a, b in zip(ee, grasp_pose(pose)))
    below_object_top = ee[2] <= pose[2] + OBJECT_HEIGHT_MM + GRASP_CLEARANCE_MM
    return within_one_step and below_object_top


def _segment_at(script: EpisodeScript, tick: int) -> Segment:
    chosen = script.segments[0]
    for segment in script.segments:
        if segment.start <= tick:
            chosen = segment
    return chosen


def _main_candidates(
    ids: CandidateIds,
    script: EpisodeScript,
    poses: dict[str, list[int]],
    ee: list[int],
    holding: str | None,
) -> list[dict]:
    """틱의 결합 행동 후보 (기능 × 대상 × 접근 × 목적지 × 프로파일) + observe/hold/replan."""
    candidates: list[dict] = []
    for obj in script.objects:
        pose = poses[obj.id]
        distance = _distance_mm(ee, pose)
        clearance = 18 if obj.fragile else 30 + (distance % 25)
        for face in obj.graspable_faces:
            if holding is not None and holding != obj.id:
                continue  # 다른 물체를 들고 있으면 새 파지 후보를 만들지 않는다
            key = f"grasp:{obj.id}:{face}:{script.zone}:slow"
            progress = " (진행 중)" if holding == obj.id else ""
            candidates.append(
                {
                    "id": ids.id_for(key),
                    "action_ref": ids.id_for(key),
                    "key": key,
                    "desc": f"{obj.desc}을 {face} 면으로 잡아 {script.zone}로 옮긴다{progress}",
                    "derived": f"reach ok, clr {clearance}mm, d {distance}mm",
                }
            )
        if holding == obj.id:
            key = f"place:{obj.id}:{script.zone}:slow"
            candidates.append(
                {
                    "id": ids.id_for(key),
                    "action_ref": ids.id_for(key),
                    "key": key,
                    "desc": f"들고 있는 {obj.desc}을 {script.zone}에 놓는다",
                    "derived": f"reach ok, clr 40mm, d {_distance_mm(ee, (*ZONE_CENTER[script.zone], pose[2]))}mm",
                }
            )
        if obj.pushable and holding is None:
            for direction in ("+x", "+y"):
                key = f"push:{obj.id}:{direction}:none:slow"
                candidates.append(
                    {
                        "id": ids.id_for(key),
                        "action_ref": ids.id_for(key),
                        "key": key,
                        "desc": f"{obj.desc}을 {direction} 방향으로 민다",
                        "derived": f"reach ok, clr {clearance}mm, d {distance}mm",
                    }
                )
    for key, desc in (
        ("observe", "관측을 더 얻는다"),
        ("hold", "현 상태를 유지한다"),
        ("replan", "재계획한다"),
    ):
        candidates.append(
            {"id": ids.id_for(key), "action_ref": ids.id_for(key), "key": key, "desc": desc, "derived": "-"}
        )
    return candidates


def _path_candidates(action_ref: str) -> list[dict]:
    return [
        {"id": "p0", "kind": "direct", "action_ref": action_ref, "desc": "대상으로 직접"},
        {"id": "p1", "kind": "via", "ref": "w1", "action_ref": action_ref, "desc": "경유점 w1 경유"},
        {"id": "p2", "kind": "retreat", "action_ref": action_ref, "desc": "후퇴"},
        {"id": "p3", "kind": "hold", "action_ref": action_ref, "desc": "정지 유지"},
    ]


def _speed_answer(phase: str, near_fragile: bool) -> str:
    base = {"approach": "2", "grasp": "1", "lift": "1", "transport": "2", "place": "1", "push": "1", "none": "0"}[phase]
    if near_fragile and base != "0":
        base = str(max(1, int(base) - 1))
    return base


def _force_answer(phase: str) -> str:
    return {"approach": "0", "grasp": "1", "lift": "0", "transport": "0", "place": "1", "push": "2", "none": "0"}[phase]


def _rollout_result(rng: random.Random, *, good: bool) -> dict:
    successes = rng.randint(6, 8) if good else rng.randint(1, 4)
    seeds = [1] * successes + [0] * (8 - successes)
    result = {"s": successes, "f": 8 - successes, "seeds": seeds}
    if not good and successes <= 2:
        result["censored"] = 1
    return result


def _main_label(
    rng: random.Random,
    *,
    candidates: list[dict],
    commitment_ref: str | None,
    target: str,
    keyframe: bool,
    done: bool,
) -> dict:
    """주 결정 라벨.

    의미 적합 후보는 목표 물체를 다루는 결합 후보다. commitment가 있고 그것이 적합하면
    그 후보만 정답이다(docs/08 §7의 commitment 규칙). 아니면 적합 집합 전체를 허용한다.
    """
    admissible = [c["id"] for c in candidates if f":{target}:" in c["key"]]
    rule_only = [c["id"] for c in candidates if c["key"] in ("observe", "hold", "replan")]
    hold_id = next(c["id"] for c in candidates if c["key"] == "hold")
    if done:
        # 목표를 이미 만족했으면 새 행동이 아니라 유지가 정답이다.
        return {
            "question_id": "q_main",
            "kind": "valid_set",
            "candidate_ids": [hold_id],
            "rule": "goal-satisfied-v1",
            "label_confidence": "high",
        }
    if commitment_ref in admissible:
        valid = [commitment_ref]
        rule = "admissible-then-performance-v1+commitment-v1"
    elif commitment_ref in rule_only:
        valid = [commitment_ref]
        rule = "gate-rule-v1"
    else:
        valid = admissible or rule_only[:1]
        rule = "admissible-then-performance-v1"

    label = {"question_id": "q_main", "kind": "valid_set", "candidate_ids": valid, "rule": rule}
    if not keyframe:
        label["label_confidence"] = "medium"
        return label

    unknown = [
        c["id"]
        for c in candidates
        if c["id"] not in admissible and c["id"] not in rule_only and c["id"] not in valid
    ]
    label["semantic_admissible"] = admissible
    if unknown:
        label["unknown"] = unknown
    label["event_results"] = {
        candidate_id: _rollout_result(rng, good=candidate_id in valid) for candidate_id in admissible
    }
    label["label_confidence"] = "high"
    return label


@dataclass(frozen=True)
class TickFacts:
    """한 틱에서 전문가와 하네스가 보는 사실. 라벨과 채택 결과를 이 값들로만 정한다."""

    t: int
    phase: str
    gripper: str
    segment_start: int
    commitment: dict | None
    candidates: list[dict]
    near_fragile: bool
    stop: bool
    done: bool
    observe: bool
    keyframe: bool
    in_gripper_window: bool


def _path_answer(phase: str, near_fragile: bool) -> list[str]:
    """경로 허용 집합. 취약 물체 근처에서는 비용이 비슷한 경유점 대안도 허용한다."""
    if phase == "none":
        return ["p3"]
    return ["p0", "p1"] if near_fragile else ["p0"]


def _tick_labels(rng: random.Random, script: EpisodeScript, facts: TickFacts) -> list[dict]:
    """틱 하나의 전문가 라벨. 근거가 없는 질문은 아예 넣지 않는다(loss mask)."""
    labels = [
        _main_label(
            rng,
            candidates=facts.candidates,
            commitment_ref=facts.commitment["action_ref"] if facts.commitment else None,
            target=script.target,
            keyframe=facts.keyframe,
            done=facts.done,
        ),
        {"question_id": "q_done", "kind": "single", "answer": facts.done},
        {"question_id": "q_instr", "kind": "single", "answer": facts.t >= script.instr_false_until},
        {"question_id": "q_observe", "kind": "single", "answer": facts.observe},
    ]
    if script.retry_from is not None and facts.t >= script.retry_from:
        # 실패 이력이 있을 때만 근거가 생긴다. 그 전 틱은 라벨 없음(loss mask).
        labels.append(
            {"question_id": "q_retry", "kind": "single", "answer": facts.t - script.retry_from > 6}
        )
    labels.append({"question_id": "q_stop", "kind": "single", "answer": facts.stop})

    if facts.commitment is None:
        # commitment가 없는 틱에서는 부가 질문을 마스킹한다 (docs/08 §4).
        return labels

    conditioned_on = f"{facts.commitment['action_ref']}/{facts.commitment['phase']}"
    if facts.in_gripper_window:
        # 전환 틱 ±1은 두 상태를 모두 허용한다 (docs/08 §7).
        labels.append(
            {
                "question_id": "q_gripper",
                "kind": "valid_set",
                "candidate_ids": ["open", "closed"],
                "conditioned_on": conditioned_on,
            }
        )
    else:
        labels.append(
            {
                "question_id": "q_gripper",
                "kind": "single",
                "answer": facts.gripper,
                "conditioned_on": conditioned_on,
            }
        )
    labels.append(
        {
            "question_id": "q_path",
            "kind": "valid_set",
            "candidate_ids": _path_answer(facts.phase, facts.near_fragile),
            "conditioned_on": conditioned_on,
        }
    )
    speed = _speed_answer(facts.phase, facts.near_fragile)
    at_phase_boundary = facts.t - facts.segment_start <= 1 and speed != "0"
    labels.append(
        {
            "question_id": "q_speed",
            "kind": "valid_set",
            # 국면이 바뀐 직후는 인접 수준도 허용한다.
            "candidate_ids": (
                sorted({speed, str(int(speed) - 1)}) if at_phase_boundary else [speed]
            ),
            "conditioned_on": conditioned_on,
        }
    )
    labels.append(
        {
            "question_id": "q_force",
            "kind": "single",
            "answer": _force_answer(facts.phase),
            "conditioned_on": conditioned_on,
        }
    )
    return labels


def _adopted_answer(
    ids: CandidateIds, script: EpisodeScript, facts: TickFacts, *, switched: bool
) -> dict:
    """하네스가 실제로 채택해 실행한 답 (docs/08 §5의 조합 규칙 결과)."""
    gate_tick = facts.t in script.observe_adopt_ticks or facts.t == script.contrast_at
    if gate_tick:
        # 관측 게이트가 발동하면 다른 답을 적용하지 않고 관측·유지로 전환한다.
        gate_key = "observe" if facts.t in script.observe_adopt_ticks else "hold"
        return {
            "main": ids.id_for(gate_key),
            "switch": True,
            "path": "p3",
            "speed": 0,
            "force": 0,
            "gripper": facts.gripper,
            "stop": facts.stop,
            "phase": "none",
        }
    if facts.commitment is None:
        return {
            "main": ids.id_for("hold"),
            "switch": False,
            "path": "p3",
            "speed": 0,
            "force": 0,
            "gripper": facts.gripper,
            "stop": facts.stop,
            "phase": "none",
        }
    return {
        "main": facts.commitment["action_ref"],
        "switch": switched,
        "path": _path_answer(facts.phase, facts.near_fragile)[0],
        "speed": int(_speed_answer(facts.phase, facts.near_fragile)),
        "force": int(_force_answer(facts.phase)),
        "gripper": facts.gripper,
        "stop": facts.stop,
        "phase": facts.phase,
    }


def _exec_history(adopted: dict | None) -> str:
    if adopted is None:
        return "none"
    return (
        f"main={adopted['main']} phase={adopted['phase']} path={adopted['path']} "
        f"speed={adopted['speed']} force={adopted['force']} gripper={adopted['gripper']} "
        f"stop={int(adopted['stop'])} ack=ok"
    )


def _instruction_at(script: EpisodeScript, tick: int) -> dict:
    chosen = script.instructions[0]
    for instruction in script.instructions:
        if instruction["t_ms"] <= tick * 100:
            chosen = instruction
    return chosen


def _build_episode(script: EpisodeScript) -> dict:
    rng = random.Random(f"{SEED}:{script.episode_id}")
    ids = CandidateIds()
    poses = {obj.id: list(obj.pose) for obj in script.objects}
    ee = [0, 0, SURFACE_Z_MM + 260]
    holding: str | None = None
    previous_adopted: dict | None = None
    previous_action_key: str | None = None
    last_geom_ms = 0  # 3D 재구성이 마지막으로 갱신된 모의 시각

    gripper_switch_ticks = {
        segment.start
        for previous, segment in zip(script.segments, script.segments[1:])
        if segment.gripper != previous.gripper
    }
    gripper_windows = {t for switch in gripper_switch_ticks for t in (switch - 1, switch, switch + 1)}
    switch_ticks: list[int] = []
    ticks: list[dict] = []
    carried_ticks = 0
    previous_tick_ee = list(ee)

    for t in range(script.n_ticks):
        segment = _segment_at(script, t)
        phase = segment.phase
        contrast_second = script.contrast_at is not None and t == script.contrast_at + 1

        # 기하 관측은 한 틱 뒤처져 들어오고, 정체 틱에는 아예 갱신되지 않는다.
        if t not in script.stale_ticks:
            last_geom_ms = t * 100 - GEOM_PERIOD_MS
        geom_age_ms = t * 100 - last_geom_ms

        if contrast_second:
            # 대조 쌍의 두 번째 틱: 직전 틱에 hold를 채택했으므로 팔이 움직이지 않았고
            # 기하 관측도 갱신되지 않았다. 따라서 관측 상태는 직전 틱과 완전히 같고,
            # 다른 것은 commitment뿐이다 (게이팅으로 해제되어 hold가 되었다).
            previous_tick = ticks[-1]
            state = copy.deepcopy(previous_tick["request"]["state"])
            candidates = copy.deepcopy(previous_tick["request"]["candidates"]["q_main"])
            phase = "none"
            commitment = {
                "action_ref": ids.id_for("hold"),
                "key": "hold",
                "phase": "none",
                "held_ticks": 1,
            }
        else:
            # 말단을 국면별 목표점 쪽으로 한 걸음 옮긴다.
            push_axis = None
            if segment.action_key is not None and segment.action_key.startswith("push:"):
                push_axis = segment.action_key.split(":")[2]
            contact_offset = _push_contact_offset(push_axis)

            # 밀기는 두 단계다. 접촉점에 서 있지 않으면 먼저 접촉점으로 가고, 서 있으면 민다.
            # 밀기 방향이 바뀌면 접촉점이 반대편으로 옮겨가므로 그동안 대상은 멈춰 있다.
            target_pose = poses[script.target]
            contact_point = (
                target_pose[0] - contact_offset[0],
                target_pose[1] - contact_offset[1],
                SURFACE_Z_MM + 20,
            )
            at_contact = phase == "push" and (
                abs(ee[0] - contact_point[0]) <= STEP_MM and abs(ee[1] - contact_point[1]) <= STEP_MM
            )

            previous_ee = list(ee)
            if phase != "none" and t not in script.stop_ticks:
                if phase == "push" and not at_contact:
                    goal = contact_point
                else:
                    goal = _ee_goal(phase, target_pose, script.zone, contact_offset, push_axis)
                ee = _step_toward(ee, goal)

            # 파지·해제는 이 틱의 상태를 만들기 전에 판정한다. 그래야 상태가 스스로
            # 모순되지 않는다(열린 그리퍼로 물체를 들고 있을 수 없다).
            if holding is None:
                # 파지 판정은 **이동 전** 자세로 한다. 그러면 이 틱의 총 이동이
                # (한 걸음 안인) 파지 자세까지로 끝나 축마다 STEP_MM을 넘지 않는다.
                if (
                    segment.gripper == "closed"
                    and phase in CARRY_PHASES
                    and can_grasp(previous_ee, poses[script.target])
                ):
                    holding = script.target
                    # 말단이 파지 자세에 정확히 안착한다. 물체는 움직이지 않는다.
                    ee = list(grasp_pose(poses[holding]))
            elif segment.gripper == "open":
                if phase == "place":
                    # 각본이 어긋나면(놓기 높이·목표 영역에 도달하기 전에 열면) 여기서 멈춘다.
                    zone_x, zone_y = ZONE_CENTER[script.zone]
                    assert ee[2] <= SURFACE_Z_MM + 70, (script.episode_id, t, "놓기 높이 미달", ee)
                    assert abs(ee[0] - zone_x) <= 150 and abs(ee[1] - zone_y) <= 150, (
                        script.episode_id,
                        t,
                        "목표 영역 밖에서 놓음",
                        ee,
                    )
                poses[holding] = [ee[0], ee[1], SURFACE_Z_MM]
                holding = None

            if holding is not None:
                # 들고 있는 물체는 말단을 따라간다.
                poses[holding] = [ee[0], ee[1], ee[2] - CARRY_OFFSET_MM]
            elif at_contact:
                poses[script.target] = [
                    target_pose[0] + (ee[0] - previous_ee[0]),
                    target_pose[1] + (ee[1] - previous_ee[1]),
                    target_pose[2],
                ]
            candidates = _main_candidates(ids, script, poses, ee, holding)
            commitment = None
            if segment.action_key is not None:
                commitment = {
                    "action_ref": ids.id_for(segment.action_key),
                    "key": segment.action_key,
                    "phase": phase,
                    "held_ticks": t - segment.start,
                }
            state = None  # 아래에서 만든다

        gripper_holds = holding is not None or (
            segment.gripper == "closed" and can_grasp(ee, poses[script.target])
        )
        if state is None:
            instruction = _instruction_at(script, t)
            state = {
                "goal": {
                    "text": instruction["text"],
                    "version": instruction["version"],
                    "target_zone": script.zone,
                    "forbidden_contact": [obj.id for obj in script.objects if obj.fragile],
                },
                "objects": [
                    {
                        "id": obj.id,
                        "desc": obj.desc,
                        "pose_mm": list(poses[obj.id]),
                        "pose_sigma_mm": 3 + (2 if obj.fragile else 0),
                        "obb_mm": [70, 70, OBJECT_HEIGHT_MM],
                        "top_mm": poses[obj.id][2] + OBJECT_HEIGHT_MM,
                        "graspable_faces": list(obj.graspable_faces),
                        "surface_conf": 0.92,
                        "visible_ratio": obj.visible_ratio,
                        "last_seen_ms": last_geom_ms,
                        "attributes": ["fragile"] if obj.fragile else [],
                    }
                    for obj in script.objects
                ],
                "scene": {"free_width_mm": 420, "work_surface_mm": 740},
                "zones": [
                    {
                        "id": script.zone,
                        "desc": "목표 영역",
                        "bounds_mm": [ZONE_CENTER[script.zone][0] - 150, -150, ZONE_CENTER[script.zone][0] + 150, 150],
                    }
                ],
                "robot": {
                    "ee_pose_mm": list(ee),
                    # 관측값이다: 명령이 아니라 실제로 물체를 물었을 때만 닫힌다
                    # (docs/08 §3.2 "실제 관측 또는 선언한 추정치만").
                    # 명령한 그리퍼 상태는 exec_history와 q_gripper 라벨에 있다.
                    "gripper_mm": GRIPPER_CLOSED_MM if gripper_holds else GRIPPER_OPEN_MM,
                    "holding": holding,
                    "contact_n": 1.5 if phase in ("grasp", "place", "push") else 0.0,
                    "speed_mm_s": 0 if phase == "none" else 120,
                },
                "events": (
                    [{"kind": "slip", "object": script.target, "t_ms": t * 100}] if t in script.failure_ticks else []
                ),
                "derived": [
                    {
                        "object": obj.id,
                        "relative_mm": [poses[obj.id][axis] - ee[axis] for axis in range(3)],
                        "nearest_clearance_mm": 18 if obj.fragile else 45,
                    }
                    for obj in script.objects
                ],
            }

        # 상태가 스스로 모순되면 각본이 어긋난 것이다. 빌드를 여기서 멈춘다.
        robot_state = state["robot"]
        emitted_ee = robot_state["ee_pose_mm"]
        assert all(abs(now - was) <= STEP_MM for was, now in zip(previous_tick_ee, emitted_ee)), (
            script.episode_id,
            t,
            "말단이 한 틱에 한 걸음보다 많이 움직였다",
            previous_tick_ee,
            emitted_ee,
        )
        previous_tick_ee = list(emitted_ee)
        assert robot_state["holding"] is None or robot_state["gripper_mm"] == GRIPPER_CLOSED_MM, (
            script.episode_id,
            t,
            "열린 그리퍼로 물체를 들고 있다",
            robot_state,
        )
        if robot_state["holding"] is not None:
            carried_ticks += 1
            carried = next(
                obj for obj in state["objects"] if obj["id"] == robot_state["holding"]
            )
            assert carried["pose_mm"] == [ee[0], ee[1], ee[2] - CARRY_OFFSET_MM], (
                script.episode_id,
                t,
                "들고 있는 물체가 말단을 따라오지 않는다",
                carried["pose_mm"],
                ee,
            )

        near_fragile = any(
            obj.fragile and _distance_mm(ee, poses[obj.id]) < 260 for obj in script.objects
        )
        stop = t in script.stop_ticks
        # 마지막 구간에 들어오면 목표를 달성한 것이다 (그 전 무-commitment 구간은 에피소드 시작).
        done = t >= script.segments[-1].start
        observe = geom_age_ms >= STALE_OBSERVE_MS or t in script.observe_adopt_ticks
        keyframe = t in script.keyframes

        facts = TickFacts(
            t=t,
            phase=phase,
            gripper=segment.gripper,
            segment_start=segment.start,
            commitment=commitment,
            candidates=candidates,
            near_fragile=near_fragile,
            stop=stop,
            done=done,
            observe=observe,
            keyframe=keyframe,
            in_gripper_window=t in gripper_windows,
        )
        labels = _tick_labels(rng, script, facts)
        switched = segment.action_key != previous_action_key and segment.action_key is not None
        adopted = _adopted_answer(ids, script, facts, switched=switched)
        if adopted["switch"]:
            switch_ticks.append(t)

        top = candidates[: min(3, len(candidates))]
        model_output = {
            "q_main": {
                candidate["id"]: round(value, 3)
                for candidate, value in zip(top, (0.71, 0.22, 0.07)[: len(top)])
            },
            "q_gripper": {"open": 0.96, "closed": 0.04}
            if segment.gripper == "open"
            else {"open": 0.05, "closed": 0.95},
            "q_stop": 0.92 if stop else 0.01,
        }

        tick_candidates: dict[str, list[dict]] = {"q_main": candidates}
        if commitment is not None:
            tick_candidates["q_path"] = _path_candidates(commitment["action_ref"])

        ticks.append(
            {
                "t": t,
                "sim_ms": t * 100,
                "observed_at_ms": max(0, t * 100 - 20),
                "obs_age_ms": {"geom": geom_age_ms, "proprio": 20},
                "request": {
                    "state": state,
                    "exec_history": _exec_history(previous_adopted),
                    "commitment": commitment,
                    "candidates": tick_candidates,
                },
                "model_output": model_output,
                "adopted": adopted,
                "ack": {
                    "seq": t,
                    "applied": not stop,
                    "gripper_event": f"ev-{script.episode_id}-{t}" if t in gripper_switch_ticks else None,
                },
                "labels": labels,
            }
        )

        previous_adopted = adopted
        # 대조 쌍의 두 번째 틱은 hold로 실행됐으므로, 다음 틱에 원래 행동으로 돌아가는 것은 전환이다.
        previous_action_key = "hold" if contrast_second else segment.action_key

    # 파지 검사가 너무 빡빡해 각본이 한 번도 물체를 잡지 못하면(조용한 무동작) 여기서 멈춘다.
    wants_grasp = any(
        segment.gripper == "closed" and segment.phase in CARRY_PHASES for segment in script.segments
    )
    assert wants_grasp == (carried_ticks > 0), (
        script.episode_id,
        "그리퍼를 닫는 각본인데 한 번도 파지하지 못했다" if wants_grasp else "각본에 없는 파지가 생겼다",
    )
    assert holding is None, (script.episode_id, "에피소드가 물체를 든 채로 끝났다", holding)

    # 가려진 참값은 그 장면에 실제로 있는 물체에 대해서만 적는다.
    evidence = {
        "expert_log": f"{script.episode_id}: 각본 전문가 (build_d0.py)",
        "rollouts": "키프레임 틱의 event_results에 요약",
    }
    occluded_true_poses = {
        obj.id: list(obj.occluded_true_pose)
        for obj in script.objects
        if obj.occluded_true_pose is not None
    }
    if occluded_true_poses:
        evidence["occluded_true_poses"] = occluded_true_poses

    return {
        "schema_version": "stream-v0",
        "episode_id": script.episode_id,
        "origin_group": script.origin_group,
        "split": script.split,
        "versions": {
            "harness": "h0.0-d0",
            "controller": "c0.0-d0",
            "expert": "scripted-e0",
            "rules": "d0-rule-v0",
            "serializer": "s0.0-d0",
        },
        "prefix": {"instructions": script.instructions, "question_set": "qs-v0"},
        "ticks": ticks,
        "provenance": {
            "generator": BUILDER_VERSION,
            "seed": SEED,
            "marks": {
                "target": script.target,
                "zone": script.zone,
                "keyframes": list(script.keyframes),
                "switch_ticks": switch_ticks,
                "gripper_transition_ticks": sorted(gripper_switch_ticks),
                "stop_ticks": list(script.stop_ticks),
                "contrast_pair": (
                    [script.contrast_at, script.contrast_at + 1] if script.contrast_at is not None else []
                ),
                "instruction_versions": [instruction["version"] for instruction in script.instructions],
                "notes": script.notes,
            },
        },
        "evidence": evidence,
    }


def _pick_place_segments(
    *, switch_at: int, grasp_at: int, place_at: int, release_at: int, end_at: int, keys: tuple[str, str]
) -> list[Segment]:
    """접근 → (전환) → 파지 → 들기 → 이동 → 놓기 → 해제 → 완료.

    `release_at`은 그리퍼를 여는 틱이다. 그 전에 말단이 놓기 자세에 도달해야 하며,
    도달하지 못하면 `_build_episode`의 검사에서 걸린다.
    """
    first, second = keys
    return [
        Segment(0, "none", "open", None),
        Segment(5, "approach", "open", first),
        Segment(switch_at, "approach", "open", second),
        Segment(grasp_at, "grasp", "closed", second),
        Segment(grasp_at + 7, "lift", "closed", second),
        Segment(grasp_at + 17, "transport", "closed", second),
        Segment(place_at, "place", "closed", second),
        Segment(release_at, "place", "open", second),
        Segment(end_at, "none", "open", None),
    ]


def episode_scripts() -> list[EpisodeScript]:
    cup = SceneObject("o7", "빨간 컵", (310, -40, 742), graspable_faces=("top", "side"))
    glass = SceneObject("o3", "유리잔", (150, 120, 742), graspable_faces=("side",), fragile=True)
    box = SceneObject("o4", "나무 상자", (-120, 210, 742), pushable=True)
    hidden = SceneObject(
        "o5",
        "가려진 통",
        (-210, 200, 742),
        graspable_faces=("top",),
        occluded_true_pose=(-210, 205, 741),
    )

    return [
        EpisodeScript(
            episode_id="ep-d0-001",
            origin_group="scene-family-s01",
            split="train",
            instructions=[
                {"version": 1, "t_ms": 0, "text": "빨간 컵을 왼쪽 영역으로 옮겨라"},
                {"version": 2, "t_ms": 5500, "text": "빨간 컵을 왼쪽 영역으로 옮기고 유리잔은 건드리지 마라"},
            ],
            objects=[cup, glass, box],
            target="o7",
            zone="zoneL",
            n_ticks=100,
            segments=_pick_place_segments(
                switch_at=20,
                grasp_at=31,
                place_at=78,
                release_at=86,
                end_at=90,
                keys=("grasp:o7:side:zoneL:slow", "grasp:o7:top:zoneL:slow"),
            ),
            keyframes=(5, 20, 35, 50, 70),
            notes=["지시 변경 v2 (t=55)", "접근 방식 전환 (t=20)"],
        ),
        EpisodeScript(
            episode_id="ep-d0-002",
            origin_group="scene-family-s02",
            split="train",
            instructions=[{"version": 1, "t_ms": 0, "text": "빨간 컵을 왼쪽 영역으로 옮겨라"}],
            objects=[cup, glass, box],
            target="o7",
            zone="zoneL",
            n_ticks=100,
            segments=[
                Segment(0, "none", "open", None),
                Segment(4, "approach", "open", "grasp:o7:top:zoneL:slow"),
                Segment(24, "grasp", "closed", "grasp:o7:top:zoneL:slow"),
                Segment(40, "approach", "open", "grasp:o7:top:zoneL:slow"),  # 미끄러짐 → 다시 접근
                Segment(44, "approach", "open", "grasp:o7:side:zoneL:slow"),  # 다른 접근으로 전환
                Segment(52, "grasp", "closed", "grasp:o7:side:zoneL:slow"),
                Segment(62, "lift", "closed", "grasp:o7:side:zoneL:slow"),
                Segment(69, "transport", "closed", "grasp:o7:side:zoneL:slow"),
                Segment(82, "place", "closed", "grasp:o7:side:zoneL:slow"),
                Segment(92, "place", "open", "grasp:o7:side:zoneL:slow"),
                Segment(96, "none", "open", None),
            ],
            keyframes=(5, 25, 44, 62, 78),
            stop_ticks=(56, 57, 58),
            retry_from=40,
            failure_ticks=(40,),
            notes=[
                "파지 미끄러짐 (t=40) → 같은 방식 재시도 부적절 → 다른 접근으로 전환 (t=44)",
                "사람 접근으로 즉시 정지 (t=56~58)",
                "q_retry 라벨은 실패 이력이 생긴 t=40부터만 붙는다 (그 전은 loss mask)",
            ],
        ),
        EpisodeScript(
            episode_id="ep-d0-003",
            origin_group="scene-family-s03",
            split="dev",
            instructions=[
                {"version": 1, "t_ms": 0, "text": "저것을 저쪽으로 옮겨라"},
                {"version": 2, "t_ms": 1200, "text": "빨간 컵을 왼쪽 영역으로 옮겨라"},
            ],
            objects=[cup, glass, hidden],
            target="o7",
            zone="zoneL",
            n_ticks=100,
            segments=_pick_place_segments(
                switch_at=22,
                grasp_at=33,
                place_at=80,
                release_at=88,
                end_at=92,
                keys=("grasp:o7:side:zoneL:slow", "grasp:o7:top:zoneL:slow"),
            ),
            keyframes=(12, 22, 40, 62, 75),
            instr_false_until=12,
            stale_ticks=(48, 49, 50),
            contrast_at=49,
            notes=[
                "지시가 불충분한 시작 구간 (t<12). t=12에 지시 v2가 들어온다",
                "기하 관측 정체 (t=48~50) → t=49에 관측 게이트가 켜져 hold를 채택",
                "commitment 대조 쌍 (t=49, 50): 관측은 같고 commitment만 다르다",
            ],
        ),
        EpisodeScript(
            episode_id="ep-d0-004",
            origin_group="scene-family-s04",
            split="test",
            instructions=[{"version": 1, "t_ms": 0, "text": "나무 상자를 밀어 통로를 비워라"}],
            objects=[box, glass, cup],
            target="o4",
            zone="zoneR",
            n_ticks=100,
            segments=[
                Segment(0, "none", "open", None),
                Segment(6, "approach", "open", "push:o4:+x:none:slow"),
                Segment(28, "push", "open", "push:o4:+x:none:slow"),
                Segment(54, "push", "open", "push:o4:+y:none:slow"),
                Segment(88, "none", "open", None),
            ],
            keyframes=(6, 28, 45, 54, 70),
            stale_ticks=(35, 36),
            observe_adopt_ticks=(36,),
            notes=[
                "밀기 방향 전환 (t=54)",
                "기하 관측 정체 (t=35~36) → 하네스가 observe 후보를 채택 (t=36)",
            ],
        ),
    ]


def stream_records() -> list[dict]:
    return [_build_episode(script) for script in episode_scripts()]


# ==========================================================================
# 쓰기와 manifest
# ==========================================================================


def _dump_jsonl(records: list[dict]) -> str:
    return "".join(
        json.dumps(record, ensure_ascii=False, sort_keys=False, separators=(",", ":")) + "\n"
        for record in records
    )


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _single_request_coverage(records: list[dict]) -> dict:
    types: dict[str, int] = {}
    kinds: dict[str, int] = {}
    questions = 0
    labels = 0
    unlabeled = 0
    multi_answer = 0
    not_applicable_answer = 0
    for record in records:
        labeled = {label["question_id"] for label in record["labels"]}
        for question in record["request"]["questions"]:
            questions += 1
            types[question["type"]] = types.get(question["type"], 0) + 1
            if question["id"] not in labeled:
                unlabeled += 1
        for label in record["labels"]:
            labels += 1
            kinds[label["kind"]] = kinds.get(label["kind"], 0) + 1
            ids = label.get("candidate_ids", [])
            if len(ids) >= 2:
                multi_answer += 1
            if ids == [UNKNOWN_CANDIDATE_ID]:
                not_applicable_answer += 1
    return {
        "questions": questions,
        "labels": labels,
        "question_types": types,
        "label_kinds": kinds,
        "questions_without_label": unlabeled,
        "labels_with_multiple_answers": multi_answer,
        "labels_answering_not_applicable": not_applicable_answer,
    }


def _stream_coverage(records: list[dict]) -> dict:
    ticks = 0
    labels = 0
    kinds: dict[str, int] = {}
    keyframe_labels = 0
    aux_labels = 0
    switch_ticks = 0
    for record in records:
        ticks += len(record["ticks"])
        for tick in record["ticks"]:
            if tick["adopted"].get("switch"):
                switch_ticks += 1
            for label in tick["labels"]:
                labels += 1
                kinds[label["kind"]] = kinds.get(label["kind"], 0) + 1
                if "event_results" in label:
                    keyframe_labels += 1
                if "conditioned_on" in label:
                    aux_labels += 1
    return {
        "ticks": ticks,
        "questions": ticks * len(QUESTION_IDS_V0),
        "labels": labels,
        "label_kinds": kinds,
        "keyframe_labels_with_event_results": keyframe_labels,
        "aux_labels_conditioned_on_commitment": aux_labels,
        "ticks_with_adopted_switch": switch_ticks,
    }


def build_all(out_dir: Path) -> dict:
    """세 파일을 `out_dir`에 쓰고 manifest를 돌려준다."""
    out_dir.mkdir(parents=True, exist_ok=True)
    singles = single_request_records()
    streams = stream_records()
    single_text = _dump_jsonl(singles)
    stream_text = _dump_jsonl(streams)

    manifest = {
        "builder_version": BUILDER_VERSION,
        "builder": "tests/fixtures/build_d0.py",
        "seed": SEED,
        "question_set": "qs-v0",
        "questions_per_tick": len(QUESTION_IDS_V0),
        "files": {
            "d0.jsonl": {
                "sha256": _sha256(single_text),
                "bytes": len(single_text.encode("utf-8")),
                "records": len(singles),
                **_single_request_coverage(singles),
            },
            "d0_streams.jsonl": {
                "sha256": _sha256(stream_text),
                "bytes": len(stream_text.encode("utf-8")),
                "records": len(streams),
                **_stream_coverage(streams),
            },
        },
        "review_note": (
            "사람 검수 전이다. 검수자는 두 파일을 읽고 이름·날짜를 reviewed_by에 추가한다. "
            "각 에피소드의 provenance.marks에 키프레임·전환·정지·대조 쌍 틱이 적혀 있다."
        ),
        "reviewed_by": [],
    }

    (out_dir / "d0.jsonl").write_text(single_text, encoding="utf-8", newline="\n")
    (out_dir / "d0_streams.jsonl").write_text(stream_text, encoding="utf-8", newline="\n")
    (out_dir / "d0_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return manifest


def main() -> None:
    manifest = build_all(FIXTURE_DIR)
    for name, info in manifest["files"].items():
        print(f"{name}: {info['records']} records, sha256 {info['sha256'][:16]}…")


if __name__ == "__main__":
    main()
