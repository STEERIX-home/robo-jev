#!/usr/bin/env python
"""손으로 쓴 판단 요청 fixture (`simgen-request/0`, Task v0a 지시서 §2) — 진단 도구(Task v0c)의 개발·검사용.

v0a 엔진 코드에 의존하지 않는다: 형식만 따른 **손으로 쓴** 8개다(상태 글·질문·정답·숨은 값 모두 직접 썼다).
숫자는 그럴듯하게 맞췄지만 엔진의 물리로 계산한 값이 아니다 — 도구가 형식을 읽고 채점하는지 보는 용도이고,
모델의 실력을 재는 데 쓰지 않는다. 행동 choice의 확률만 기획서 §5의 정의대로 기대 비용에서 계산한다:
p_a = softmax(−(E[c_a] − min_b E[c_b]) / τ), τ = 2.0 (v0a 지시서 §3.6의 기본 가치표 v0).

    uv run python tests/fixtures/decisions/build_requests.py   # → tests/fixtures/decisions/requests.jsonl

담은 것: 장면 6종(p1_step_in·p1_crossing·p1_head_on·p5_braking·p5_turning·p1p5_step_in_carrying), 문체 3종,
분야 7개, 몸 4종, 권고(맞음·틀림·없음), 힌트, 숫자 쌍(fx-002 ↔ fx-001)과 단어 쌍(fx-003 ↔ fx-001), 선택지 8개
질문(코드 A–H), 키 문체 3종(단어·짧은 코드·동사형), 규칙 적용 choice, 상태 choice.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

OUT = Path(__file__).with_name("requests.jsonl")
TAU = 2.0


def action_label(costs: dict[str, float]) -> dict:
    """기대 비용 → 행동 choice 정답 (argmax는 언제나 기대 비용 최소)."""
    low = min(costs.values())
    weights = {key: math.exp(-(cost - low) / TAU) for key, cost in costs.items()}
    total = sum(weights.values())
    probabilities = {key: round(w / total, 9) for key, w in weights.items()}
    # 반올림 뒤 합이 정확히 1이 되게 최소 비용 선택지에 나머지를 얹는다
    best = min(costs, key=costs.get)
    probabilities[best] = round(probabilities[best] + 1.0 - sum(probabilities.values()), 9)
    return {"type": "choice", "kind": "action", "probabilities": probabilities, "expected_cost": costs}


def state_label(probabilities: dict[str, float]) -> dict:
    return {"type": "choice", "kind": "state", "probabilities": probabilities}


def rule_label(keys: list[str], answer: str) -> dict:
    return {"type": "choice", "kind": "rule", "probabilities": {key: (1.0 if key == answer else 0.0) for key in keys}}


def noul(p: float) -> dict:
    return {"type": "noul", "p_yes": p}


def score(v: float) -> dict:
    return {"type": "score", "value": v}


def meta(scene, phenomena, domain, body, style, *, sensor="lidar_camera", hint=False, recommendation=None, margins=None, pair=None) -> dict:
    return {
        "scene": scene, "phenomena": phenomena, "domain": domain, "body": body, "sensor_suite": sensor, "style": style,
        "language": "en", "units": {"distance": "m", "time": "s"}, "hint": hint,
        "recommendation": recommendation or {"present": False},
        "margins": margins or {}, "pair": pair,
    }


def generator(seed: int) -> dict:
    return {"version": "fixture-v0c", "catalog": "handwritten", "values": "values-v0", "seed": seed, "git": None}


# -- 카페: 쟁반의 라테 + 통로에 끼어든 손님 (p1p5 조합; 쌍 둘의 기준) ------------------------------------

CAFE_ACTION = {
    "continue": "Keep going at the current speed (1.1 m/s).",
    "slow_half": "Slow to about half speed (0.55 m/s) and keep going.",
    "stop_gentle": "Stop with gentle braking (about 1.0 m/s^2).",
    "stop_moderate": "Stop with moderate braking (about 1.8 m/s^2).",
    "stop_hard": "Stop with hard braking (about 3.0 m/s^2).",
}
CAFE_NEXT = {
    "stays": "Stays where they are in the aisle.",
    "crosses": "Keeps walking and clears the aisle.",
    "steps_back": "Steps back out of the aisle.",
}


def cafe_state(*, gap: str, place: str = "café", person: str = "customer", drink: str = "iced latte", table: str = "table 7") -> str:
    return (
        f"You are a delivery robot (wheeled base) in a busy {place}, carrying a tray to {table}.\n"
        f"- Your speed: 1.1 m/s, driving straight down the aisle between tables.\n"
        f"- On the tray: one tall glass of {drink}, open top, about three-quarters full (camera estimate).\n"
        f"- An adult {person} stepped into the aisle {gap} m ahead of you, 0.4 s ago (lidar).\n"
        f"- The {person} is facing the counter and has not looked at you.\n"
        f"- Your braking: gentle 1.0 m/s^2, moderate 1.8 m/s^2, hard 3.0 m/s^2; reaction time 0.2 s; body radius 0.32 m.\n"
        f"- Two chairs are pushed out at the next table (not in your path). Music is playing.\n"
        f"- Planner recommendation: stop_moderate."
    )


def cafe_questions(person: str = "customer", drink: str = "latte") -> dict:
    return {
        "action": {"type": "choice", "instructions": "What should the robot do now?", "criteria": dict(CAFE_ACTION)},
        "contact_if_continue": {"type": "noul", "instructions": f"If the robot keeps its current speed, will it touch the {person} within the next 3 seconds?"},
        "spill_if_hard": {"type": "noul", "instructions": f"If the robot brakes hard (3.0 m/s^2) now, will the {drink} spill?"},
        "risk_gentle": {
            "type": "score", "instructions": f"How likely is it that the robot touches the {person} if it stops with gentle braking?",
            "criteria": [f"0 = it certainly will not touch the {person}", f"1 = it certainly will touch the {person}"],
        },
        "person_next": {"type": "choice", "instructions": f"What will the {person} most likely do in the next two seconds?", "criteria": dict(CAFE_NEXT)},
    }


def records() -> list[dict]:
    out = []
    # fx-001 — 기준 장면
    out.append({
        "schema": "simgen-request/0", "id": "fx-001", "generator": generator(1),
        "meta": meta("p1p5_step_in_carrying", ["P1", "P5"], "cafe", "mobile_base", "facts",
                     recommendation={"present": True, "option": "stop_moderate", "wrong": False},
                     margins={"stop_margin_m": 0.47, "spill_margin_cm": 2.1}),
        "request": {"state": cafe_state(gap="1.6"), "questions": cafe_questions()},
        "labels": {
            "action": action_label({"continue": 82.4, "slow_half": 38.0, "stop_gentle": 8.9, "stop_moderate": 3.4, "stop_hard": 4.6}),
            "contact_if_continue": noul(0.86),
            "spill_if_hard": noul(0.18),
            "risk_gentle": score(0.07),
            "person_next": state_label({"stays": 0.52, "crosses": 0.39, "steps_back": 0.09}),
        },
        "truth": {"world": {"intent": "stays", "true_gap_m": 1.58, "true_fill": 0.71, "slosh_k": 1.31},
                  "realized": {"contact_if_continue": True, "spill_if_hard": False, "person_next": "stays"}},
        "rationale": "Moderate braking stops in 0.22 + 0.34 = 0.56 m against 1.03 m of free gap; gentle needs 0.83 m and touches only if the customer drifts forward; hard braking risks the three-quarters-full glass, so moderate has the lowest expected cost.",
    })
    # fx-002 — fx-001의 숫자 쌍: 틈 1.6 m → 2.4 m (그 사실 하나만)
    out.append({
        "schema": "simgen-request/0", "id": "fx-002", "generator": generator(2),
        "meta": meta("p1p5_step_in_carrying", ["P1", "P5"], "cafe", "mobile_base", "facts",
                     recommendation={"present": True, "option": "stop_moderate", "wrong": False},
                     margins={"stop_margin_m": 1.27, "spill_margin_cm": 2.1}, pair={"of": "fx-001", "kind": "numeric"}),
        "request": {"state": cafe_state(gap="2.4"), "questions": cafe_questions()},
        "labels": {
            "action": action_label({"continue": 64.0, "slow_half": 21.5, "stop_gentle": 2.7, "stop_moderate": 3.1, "stop_hard": 4.5}),
            "contact_if_continue": noul(0.71),
            "spill_if_hard": noul(0.18),
            "risk_gentle": score(0.01),
            "person_next": state_label({"stays": 0.52, "crosses": 0.39, "steps_back": 0.09}),
        },
        "truth": {"world": {"intent": "crosses", "true_gap_m": 2.43, "true_fill": 0.78, "slosh_k": 1.12},
                  "realized": {"contact_if_continue": False, "spill_if_hard": False, "person_next": "crosses"}},
        "rationale": "With 1.83 m of free gap even gentle braking (0.83 m) stops well short, and it is the gentlest on the glass, so it overtakes moderate braking by a small margin.",
    })
    # fx-003 — fx-001의 단어 쌍: 숫자는 그대로, 상황 옷 단어만 (café→restaurant, customer→diner, latte→iced tea)
    out.append({
        "schema": "simgen-request/0", "id": "fx-003", "generator": generator(3),
        "meta": meta("p1p5_step_in_carrying", ["P1", "P5"], "cafe", "mobile_base", "facts",
                     recommendation={"present": True, "option": "stop_moderate", "wrong": False},
                     margins={"stop_margin_m": 0.47, "spill_margin_cm": 2.1}, pair={"of": "fx-001", "kind": "wording"}),
        "request": {"state": cafe_state(gap="1.6", place="restaurant", person="diner", drink="iced tea", table="booth 4"),
                    "questions": cafe_questions(person="diner", drink="iced tea")},
        "labels": {
            "action": action_label({"continue": 82.4, "slow_half": 38.0, "stop_gentle": 8.9, "stop_moderate": 3.4, "stop_hard": 4.6}),
            "contact_if_continue": noul(0.86),
            "spill_if_hard": noul(0.18),
            "risk_gentle": score(0.07),
            "person_next": state_label({"stays": 0.52, "crosses": 0.39, "steps_back": 0.09}),
        },
        "truth": {"world": {"intent": "steps_back", "true_gap_m": 1.61, "true_fill": 0.74, "slosh_k": 1.44},
                  "realized": {"contact_if_continue": False, "spill_if_hard": False, "person_next": "steps_back"}},
        "rationale": "Same numbers as fx-001 with restaurant wording; the posterior and the costs are unchanged.",
    })
    # fx-004 — 창고 교차로: 사람이 모는 지게차가 오른쪽에서 (산문, 힌트, 규칙 적용 choice)
    out.append({
        "schema": "simgen-request/0", "id": "fx-004", "generator": generator(4),
        "meta": meta("p1_crossing", ["P1"], "warehouse", "forklift_amr", "prose", hint=True, margins={"time_margin_s": 0.4}),
        "request": {
            "state": (
                "You are an autonomous forklift carrying an empty pallet along the main aisle of a distribution warehouse at 1.5 m/s. "
                "The aisle crosses a side lane 6.0 m ahead of your forks; the crossing zone is 3.0 m wide. A manually driven forklift "
                "is approaching the crossing from your right along the side lane. Your lidar puts it 4.6 m from the edge of the crossing, "
                "and its speed, estimated from the last two scans 0.5 s apart, is about 2.2 m/s. Its driver is looking down at a "
                "handheld scanner. Estimated time for the other forklift to reach the crossing: 2.1 s. Your braking with an empty pallet "
                "is moderate 1.6 m/s^2 or hard 3.0 m/s^2, reaction time 0.3 s. A stack of shrink-wrapped boxes on your left blocks part of "
                "your view of the side lane beyond 8 m. The overhead lights in this bay flicker occasionally."
            ),
            "questions": {
                "decision": {"type": "choice", "instructions": "Choose the safest option that still keeps the pallet moving if possible.", "criteria": {
                    "proceed": "Keep 1.5 m/s and drive through the crossing.",
                    "slow_down": "Slow to 0.8 m/s while approaching the crossing.",
                    "stop_moderate": "Stop before the crossing with moderate braking.",
                    "stop_hard": "Stop before the crossing with hard braking.",
                }},
                "conflict": {"type": "noul", "instructions": "If you keep 1.5 m/s, will the other forklift enter the crossing zone while you are still inside it?"},
                "risk_proceed": {"type": "score", "instructions": "How likely is a collision with the other forklift if you proceed at 1.5 m/s?",
                                 "criteria": ["0 = no chance of a collision", "1 = a collision is certain"]},
                "zone": {"type": "choice", "instructions": "Rule: if the other forklift is less than 5.0 m from the crossing, answer near; if it is 5.0 m or more away, answer far.", "criteria": {
                    "near": "The other forklift is less than 5.0 m from the crossing.",
                    "far": "The other forklift is 5.0 m or more from the crossing.",
                }},
            },
        },
        "labels": {
            "decision": action_label({"proceed": 31.0, "slow_down": 9.8, "stop_moderate": 5.2, "stop_hard": 6.9}),
            "conflict": noul(0.41),
            "risk_proceed": score(0.24),
            "zone": rule_label(["near", "far"], "near"),
        },
        "truth": {"world": {"true_speed_mps": 2.45, "driver_brakes": False},
                  "realized": {"conflict": True, "risk_proceed": 1.0}},
        "rationale": "You clear the 3 m zone from 6 m ahead at 1.5 m/s in 6.0 s while the forklift arrives in about 2.1 s; whether it brakes is unknown, so proceeding carries a 0.41 chance of sharing the zone; stopping moderately costs about 5 s of delay and nothing else.",
    })
    # fx-005 — 병원 복도 마주침: 보행 보조기 환자 (계획표 문체, 틀린 권고)
    out.append({
        "schema": "simgen-request/0", "id": "fx-005", "generator": generator(5),
        "meta": meta("p1_head_on", ["P1"], "hospital", "humanoid", "planner", sensor="camera_only",
                     recommendation={"present": True, "option": "continue", "wrong": True}, margins={"side_margin_m": -0.18}),
        "request": {
            "state": (
                "ROBOT STATE\n"
                "  body: humanoid porter, carrying a closed linen bag; speed 0.9 m/s; shoulder width 0.55 m\n"
                "  braking: gentle 0.6 m/s^2, moderate 1.2 m/s^2; reaction 0.25 s\n"
                "PERCEIVED (camera, 0.3 s old)\n"
                "  corridor width 1.6 m; handrails both sides\n"
                "  elderly patient with a walker, 4.2 m ahead, walking toward you at about 0.5 m/s, centered in the corridor\n"
                "  patient walker width about 0.6 m\n"
                "  a door alcove on your right 2.0 m ahead (0.5 m deep)\n"
                "PLANNER RECOMMENDATION\n"
                "  continue (path cost lowest)\n"
                "AVAILABLE ACTIONS\n"
                "  continue, sidestep_right, stop_gentle, stop_moderate\n"
                "DECLARED RULES\n"
                "  keep at least 0.3 m from patients"
            ),
            "questions": {
                "action": {"type": "choice", "instructions": "What should the robot do now?", "criteria": {
                    "continue": "Keep walking at 0.9 m/s along the current line.",
                    "sidestep_right": "Step into the door alcove on the right and let the patient pass.",
                    "stop_gentle": "Stop with gentle braking where you are.",
                    "stop_moderate": "Stop with moderate braking where you are.",
                }},
                "patient_yields": {"type": "noul", "instructions": "Will the patient move aside to make room for you?"},
                "squeeze_risk": {"type": "score", "instructions": "If you continue, how likely is it that you pass closer than 0.3 m to the patient?",
                                 "criteria": ["0 = you will certainly keep 0.3 m", "1 = you will certainly come closer than 0.3 m"]},
            },
        },
        "labels": {
            "action": action_label({"continue": 74.0, "sidestep_right": 2.2, "stop_gentle": 4.9, "stop_moderate": 4.1}),
            "patient_yields": noul(0.12),
            "squeeze_risk": score(0.81),
        },
        "truth": {"world": {"patient_will_yield": False, "true_speed_mps": 0.42}, "realized": {"patient_yields": False}},
        "rationale": "The free width beside a centered 0.6 m walker is 0.5 m per side, less than your 0.55 m plus the 0.3 m rule; the patient rarely yields, so continuing almost surely breaks the clearance rule; the alcove lets the patient pass with little delay.",
    })
    # fx-006 — 실험대: 팔이 비커를 옮기다 손이 들어옴 (짧은 코드 키, 상태 choice)
    out.append({
        "schema": "simgen-request/0", "id": "fx-006", "generator": generator(6),
        "meta": meta("p5_braking", ["P5"], "lab", "arm", "facts", sensor="wrist_camera", margins={"spill_margin_cm": -0.4}),
        "request": {
            "state": (
                "A fixed robot arm is moving a 250 ml glass beaker across the lab bench.\n"
                "- Gripper speed: 0.5 m/s toward the balance, 0.35 m away.\n"
                "- Beaker: open, inner diameter 7.0 cm, rim height 9.5 cm; liquid level looks about three-quarters full (wrist camera, glare on the glass).\n"
                "- Contents label: dilute acid.\n"
                "- A colleague's hand just entered the work area 0.22 m ahead of the gripper's path.\n"
                "- Arm deceleration options: 1.5, 4 or 8 m/s^2; reaction 0.05 s.\n"
                "- The fume hood fan is running."
            ),
            "questions": {
                "how_stop": {"type": "choice", "instructions": "How should the arm stop?", "criteria": {
                    "a1": "Decelerate at 1.5 m/s^2.",
                    "b2": "Decelerate at 4 m/s^2.",
                    "c3": "Decelerate at 8 m/s^2.",
                    "d4": "Do not stop; keep moving to the balance.",
                }},
                "spill_at_8": {"type": "noul", "instructions": "If the arm decelerates at 8 m/s^2, will liquid spill over the rim?"},
                "fill_band": {"type": "choice", "instructions": "How full is the beaker really?", "criteria": {
                    "half": "About half full or less.",
                    "three_quarters": "About three-quarters full.",
                    "nearly_full": "Nearly full.",
                }},
                "spill_risk_4": {"type": "score", "instructions": "How likely is a spill if the arm decelerates at 4 m/s^2?",
                                 "criteria": ["0 = certainly no spill", "1 = certainly a spill"]},
            },
        },
        "labels": {
            "how_stop": action_label({"a1": 31.5, "b2": 6.3, "c3": 17.8, "d4": 104.0}),
            "spill_at_8": noul(0.43),
            "fill_band": state_label({"half": 0.14, "three_quarters": 0.61, "nearly_full": 0.25}),
            "spill_risk_4": score(0.11),
        },
        "truth": {"world": {"true_fill": 0.81, "slosh_k": 1.5, "hand_moves": "stays"}, "realized": {"spill_at_8": True}},
        "rationale": "At 1.5 m/s^2 the gripper needs 0.11 m to stop and may reach the hand if it moves in; 4 m/s^2 stops in 0.06 m with a small spill chance; 8 m/s^2 tilts the surface enough to spill a near-full beaker.",
    })
    # fx-007 — 가정 부엌: 뜨거운 수프 그릇을 들고 모퉁이 돌기 (동사형 키, 산문)
    out.append({
        "schema": "simgen-request/0", "id": "fx-007", "generator": generator(7),
        "meta": meta("p5_turning", ["P5"], "home", "mobile_base", "prose", margins={"spill_margin_cm": 0.6}),
        "request": {
            "state": (
                "You are a home assistant robot on wheels carrying an open bowl of hot soup from the kitchen to the dining table. "
                "You are rolling at 0.8 m/s and have to turn left around the corner of the kitchen island; the tightest turn radius "
                "of your path is about 0.6 m. The bowl is a wide soup bowl, 16 cm across and 6 cm deep, and it looks a little more "
                "than half full. The floor is level tile. A cat is asleep on a chair across the room. Nobody is in the hallway."
            ),
            "questions": {
                "turn_speed": {"type": "choice", "instructions": "Choose the fastest option that does not spill the soup.", "criteria": {
                    "keep_speed": "Take the turn at 0.8 m/s.",
                    "slow_to_half": "Slow to 0.5 m/s for the turn.",
                    "slow_to_crawl": "Slow to 0.3 m/s for the turn.",
                    "stop_and_rotate": "Stop, rotate in place, then continue.",
                }},
                "spill_keep_speed": {"type": "noul", "instructions": "If you take the turn at 0.8 m/s, will soup spill?"},
                "spill_risk_half": {"type": "score", "instructions": "How likely is a spill at 0.5 m/s through the turn?",
                                    "criteria": ["0 = certainly no spill", "1 = certainly a spill"]},
            },
        },
        "labels": {
            "turn_speed": action_label({"keep_speed": 27.0, "slow_to_half": 3.1, "slow_to_crawl": 2.4, "stop_and_rotate": 4.0}),
            "spill_keep_speed": noul(0.55),
            "spill_risk_half": score(0.08),
        },
        "truth": {"world": {"true_fill": 0.58, "slosh_k": 1.2}, "realized": {"spill_keep_speed": True}},
        "rationale": "At 0.8 m/s the lateral acceleration is 1.07 m/s^2, tilting the surface by 6.2 degrees, enough to reach the rim of a shallow bowl in about half of the plausible fills; 0.3 m/s is nearly free and costs under a second.",
    })
    # fx-008 — 보도: 배달 로봇 앞에 아이가 뛰어듦 (선택지 8개, 계획표 문체, 힌트, 규칙 적용)
    out.append({
        "schema": "simgen-request/0", "id": "fx-008", "generator": generator(8),
        "meta": meta("p1_step_in", ["P1"], "sidewalk", "mobile_base", "planner", hint=True,
                     recommendation={"present": True, "option": "stop_moderate", "wrong": False}, margins={"stop_margin_m": 0.9}),
        "request": {
            "state": (
                "ROBOT STATE\n"
                "  sidewalk delivery robot, speed 1.4 m/s, radius 0.35 m, cargo lid closed\n"
                "  braking: gentle 1.0 m/s^2, moderate 1.8 m/s^2, hard 3.0 m/s^2; reaction 0.2 s\n"
                "PERCEIVED (lidar + camera, 0.2 s old)\n"
                "  a child (about 6 years old) ran out from behind a parked van and stopped 3.0 m ahead in your lane\n"
                "  an adult is calling the child from the other side of the van\n"
                "  sidewalk free space on your left: 1.2 m; on your right: the van\n"
                "  estimated stopping distance with moderate braking: 0.82 m\n"
                "PLANNER RECOMMENDATION\n"
                "  stop_moderate\n"
                "DECLARED RULES\n"
                "  children within 4 m: speed limit 0.5 m/s"
            ),
            "questions": {
                "action": {"type": "choice", "instructions": "What should the robot do now?", "criteria": {
                    "continue": "Keep 1.4 m/s.",
                    "slow_75": "Slow to 1.05 m/s.",
                    "slow_50": "Slow to 0.7 m/s.",
                    "slow_25": "Slow to 0.35 m/s.",
                    "stop_gentle": "Stop with gentle braking.",
                    "stop_moderate": "Stop with moderate braking.",
                    "stop_hard": "Stop with hard braking.",
                    "sidestep_left": "Swerve left into the free space and keep going.",
                }},
                "child_in_lane_2s": {"type": "noul", "instructions": "Will the child still be in your lane two seconds from now?"},
                "contact_risk_continue": {"type": "score", "instructions": "If you keep 1.4 m/s, how likely is contact with the child?",
                                          "criteria": ["0 = no contact possible", "1 = contact is certain"]},
                "speed_rule": {"type": "choice", "instructions": "Rule: when a child is within 4 m, the speed limit is 0.5 m/s. Is your current speed above or within the limit?", "criteria": {
                    "above": "Above the limit.",
                    "within": "Within the limit.",
                }},
            },
        },
        "labels": {
            "action": action_label({"continue": 98.0, "slow_75": 61.0, "slow_50": 30.0, "slow_25": 9.0, "stop_gentle": 5.5,
                                    "stop_moderate": 3.0, "stop_hard": 3.6, "sidestep_left": 44.0}),
            "child_in_lane_2s": noul(0.58),
            "contact_risk_continue": score(0.67),
            "speed_rule": rule_label(["above", "within"], "above"),
        },
        "truth": {"world": {"intent": "runs_back", "true_gap_m": 2.95}, "realized": {"child_in_lane_2s": False}},
        "rationale": "Moderate braking needs 0.28 + 0.54 = 0.82 m of the 2.3 m free gap; a child may dart again, so slowing alone is not enough, and swerving toward the hidden side of the van is risky.",
    })
    return out


def main() -> None:
    lines = [json.dumps(record, ensure_ascii=False, sort_keys=False) for record in records()]
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"{len(lines)} records → {OUT}")


if __name__ == "__main__":
    main()
