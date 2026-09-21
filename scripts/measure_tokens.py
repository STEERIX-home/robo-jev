"""B2 — 실제 tokenizer로 틱당 토큰을 잰다 (docs/08 §3.4; 계약 v0.3의 예산과 대조).

네 가지를 잰다.

(a) D0 스트림 4 에피소드(`tests/fixtures/d0_streams.jsonl`): prefix 토큰, 틱당 새 토큰
    p50/p95/max, 그 틱의 K에서의 후보 블록, 결정 위치 부담. (옛 서식의 손으로 만든 fixture라 물체 줄이
    v0.3의 소개/동적 분리를 온전히 타지 않는다 — 연속성 참고값.)
(b) D0 단일 요청 64건(`tests/fixtures/d0.jsonl`): 전체 토큰 p50/p95/max, 질문별 T_i 크기.
(c) 합성 스트림 에피소드: `RobotHarness.build_request`로 물체 6·10개 × K 상한(설정의 12; 상한을 푼 상계 셀 하나) ×
    지시 변경 유무의 장면을 100틱(10초 학습 구간) 돌려 prefix·첫 틱·통상 틱·갱신 틱(10틱마다)·소개 틱(30틱마다)·
    지시 변경 틱과 **구간별** 토큰을 잰다. 말단이 대상 쪽으로 움직이고 물체 하나가 외란으로 옮겨지며 하나가 잠깐
    가려지는 장면이라 통상 틱에도 작은 변화가 있다.
(d) `--episodes DIR`: 실제 에피소드 배치(`episodes/*/streams.jsonl`)의 틱당 토큰 분포와 100틱 구간 합.

**장면 빌더의 결합 규칙.** 아래 `obj`/`observation`은 `tests/test_harness.py`의 같은 이름
빌더를 **복제**한 것이다(`tests/`는 패키지가 아니고, 그 파일은 다른 브랜치에서 고쳐지고 있어
import로 묶지 않는다). 두 곳이 어긋나면 이 스크립트의 장면이 하네스 검사가 쓰는 관측과 달라져
측정이 다른 것을 재게 되므로, `tests/test_measure_tokens.py`가 두 빌더의 출력이 같은지(영역
목록만 이 스크립트의 `ZONES`로 다름) 검사로 대조한다. `tests/test_harness.py`의 빌더를 바꾸면
여기도 같이 바꾼다. `scripts/measure_candidates.py`는 이 장면·에피소드 빌더를 그대로 쓴다.

결과는 `artifacts/reports/tokens-b2.json`(git 제외)에 쓰고 표로 찍는다. 이 스크립트는
하네스를 import한다 — 모델 코드는 하지 않는다 (docs/06 §1).

실행: `uv run python scripts/measure_tokens.py [--tokenizer <id|path>] [--out …] [--episodes DIR] [--ticks 100]`
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any

from robo_jev.data.episode import append_tick, new_episode
from robo_jev.harness.robot import RobotHarness, candidate_id, load_harness_config
from robo_jev.model.serialize import STREAM_FORMAT, TOKEN_SERIALIZER_VERSION, serialize_request
from robo_jev.model.tokenizer import (
    FETCH_SCRIPT,
    MANIFEST_NAME,
    available_tokenizer,
    load_tokenizer,
    tokenizer_root,
)

REPO = Path(__file__).resolve().parents[1]
D0 = REPO / "tests" / "fixtures" / "d0.jsonl"
D0_STREAMS = REPO / "tests" / "fixtures" / "d0_streams.jsonl"
DEFAULT_OUT = REPO / "artifacts" / "reports" / "tokens-b2.json"

#: 계약 v0.3의 예산 (HANDOFF 결정 1, decisions-1-2-v03 §0): 10물체·K=12 장면에서 틱당 p50 ≤ 500, p95 ≤ 800, 첫 틱 ≤ 1,200,
#: 10초 학습 구간(100틱) ≤ 60K. 여기서는 대조만 하고 문서는 고치지 않는다.
BUDGET = {"tick_p50": 500, "tick_p95": 800, "first_tick": 1200, "chunk_100_ticks": 60_000}

#: 합성 셀 (물체 수, K 상한 — None은 상한 없음). K=32는 계약 v0.3에 없다; `(10, None)`은 10물체에서 상한을 푼 상계 참고값이다.
SYNTHETIC_CELLS = ((6, 12), (10, 12), (10, None))

#: 상한을 푼 셀은 **프로파일 밖**이다 — 10물체에서 후보가 42개까지 나온다. 계약 검사의 프로파일 상한(Q≤16·K≤32,
#: `contracts.PROFILE_LIMITS`)을 그 셀에만 넓혀 준다: 상계 참고값을 재는 것이 목적이고, 계약이 넓어진 것이 아니다.
UNCAPPED_LIMITS = {"max_questions": 16, "max_candidates": 64}

#: 예산을 판정하는 셀 (10물체, K=12).
BUDGET_CELL = (10, 12)

#: 구간 이름 (직렬화 조각 이름 → 표의 열). 앞 열한 개가 상태 구간이다.
STATE_SECTIONS = ("t", "goal", "objects_intro", "objects_dynamic", "zones", "scene", "robot", "exec", "events", "waypoints", "extra")
SECTIONS = STATE_SECTIONS + ("commitment", "exec_history", "q_main", "q_path", "decisions", "instruction_change")


# --------------------------------------------------------------------------
# 통계
# --------------------------------------------------------------------------


def percentile(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    rank = max(0, min(len(ordered) - 1, round(q * (len(ordered) - 1))))
    return float(ordered[rank])


def summary(values: list[float]) -> dict[str, float]:
    if not values:
        return {"n": 0}
    return {
        "n": len(values),
        "mean": round(statistics.fmean(values), 1),
        "p50": percentile(values, 0.5),
        "p95": percentile(values, 0.95),
        "max": float(max(values)),
        "min": float(min(values)),
    }


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def count(tokenizer: Any, text: str) -> int:
    return len(tokenizer.encode(text, add_special_tokens=False).ids)


# --------------------------------------------------------------------------
# 구간별 집계
# --------------------------------------------------------------------------


def segment_tokens(segment: dict) -> int:
    return int(segment["end"] - segment["start"])


def section_of(name: str) -> str:
    """직렬화 조각 이름 → 구간 열 이름."""
    if name.startswith("state:"):
        return name.split(":", 1)[1]
    if name.startswith("candidate:") or name.startswith("candidates:"):
        return name.split(":")[1]
    if name.startswith("decision:"):
        return "decisions"
    if name.startswith("instruction:"):
        return "instruction_change"
    return name


def tick_breakdown(out: dict, tick: dict) -> dict[str, Any]:
    """틱 하나의 새 토큰을 구간별로 나눈다."""
    own = [s for s in out["segments"] if s["tick"] == tick["index"]]
    sections: dict[str, int] = {}
    candidate_lines: dict[str, list[int]] = {}
    for segment in own:
        name = segment["name"]
        tokens = segment_tokens(segment)
        section = section_of(name)
        sections[section] = sections.get(section, 0) + tokens
        if name.startswith("candidate:"):
            candidate_lines.setdefault(section, []).append(tokens)
    new_tokens = tick["end"] - tick["start"]  # 도중 지시 조각은 그 틱의 토큰이라 이미 들어 있다
    change = sections.get("instruction_change", 0)
    candidate_blocks = {qid: sections[qid] for qid in tick["candidate_mapping"] if qid in sections}
    state = sum(tokens for section, tokens in sections.items() if section in STATE_SECTIONS)
    return {
        "t": tick["t"],
        "index": tick["index"],
        "new_tokens": new_tokens,
        "new_tokens_without_instruction_change": new_tokens - change,
        "state": state,
        "commitment": sections.get("commitment", 0),
        "exec_history": sections.get("exec_history", 0),
        "instruction_change": change,
        "decisions": sections.get("decisions", 0),
        "posed": len(tick["posed"]),
        "k": {qid: len(ids) for qid, ids in tick["candidate_mapping"].items() if qid in candidate_blocks},
        "candidate_block": candidate_blocks,
        "tokens_per_candidate": {qid: round(statistics.fmean(lines), 1) for qid, lines in candidate_lines.items()},
        "sections": {section: sections[section] for section in SECTIONS if sections.get(section)},
    }


def section_means(ticks: list[dict]) -> dict[str, float]:
    """틱 묶음의 구간별 평균 토큰 (그 틱에 없는 구간은 0으로 센다)."""
    if not ticks:
        return {}
    return {
        section: round(statistics.fmean(tick["sections"].get(section, 0) for tick in ticks), 1)
        for section in SECTIONS
        if any(tick["sections"].get(section, 0) for tick in ticks)
    }


# --------------------------------------------------------------------------
# (a) D0 스트림
# --------------------------------------------------------------------------


def measure_streams(tokenizer: Any, records: list[dict]) -> dict[str, Any]:
    episodes = []
    all_ticks: list[dict] = []
    for record in records:
        out = serialize_request(record, tokenizer, layout="stream_l1a")
        ticks = [tick_breakdown(out, tick) for tick in out["ticks"]]
        all_ticks.extend(ticks)
        objects = [len(tick["request"]["state"].get("objects") or []) for tick in record["ticks"]]
        episodes.append(
            {
                "episode_id": record.get("episode_id"),
                "ticks": len(ticks),
                "objects": sorted(set(objects)),
                "prefix_tokens": out["prefix_end"],
                "instruction_changes": len(out["instruction_positions"]) - 1,
                "instruction_change_tokens": sum(t["instruction_change"] for t in ticks),
                "new_tokens_per_tick": summary([t["new_tokens"] for t in ticks]),
                "state_per_tick": summary([t["state"] for t in ticks]),
                "q_main_block_per_tick": summary([t["candidate_block"].get("q_main", 0) for t in ticks]),
                "decisions_per_tick": summary([t["decisions"] for t in ticks]),
                "posed_per_tick": summary([t["posed"] for t in ticks]),
                "first_tick_sections": ticks[0]["sections"],
                "sections_mean": section_means(ticks),
            }
        )
    by_k: dict[int, list[int]] = {}
    per_candidate: list[float] = []
    for tick in all_ticks:
        k = tick["k"].get("q_main")
        if k is not None:
            by_k.setdefault(k, []).append(tick["candidate_block"]["q_main"])
            per_candidate.append(tick["tokens_per_candidate"]["q_main"])
    return {
        "episodes": episodes,
        "all_ticks": {
            "new_tokens": summary([t["new_tokens"] for t in all_ticks]),
            "state": summary([t["state"] for t in all_ticks]),
            "q_main_block": summary([t["candidate_block"].get("q_main", 0) for t in all_ticks]),
            "q_path_block": summary([t["candidate_block"]["q_path"] for t in all_ticks if "q_path" in t["candidate_block"]]),
            "decisions": summary([t["decisions"] for t in all_ticks]),
            "tokens_per_q_main_candidate": summary(per_candidate),
        },
        "q_main_block_by_k": {
            str(k): {"n": len(values), "mean": round(statistics.fmean(values), 1), "per_candidate": round(statistics.fmean(values) / k, 1)}
            for k, values in sorted(by_k.items())
        },
    }


# --------------------------------------------------------------------------
# (b) D0 단일 요청
# --------------------------------------------------------------------------


def measure_singles(tokenizer: Any, records: list[dict]) -> dict[str, Any]:
    totals: list[int] = []
    states: list[int] = []
    per_question: list[int] = []
    by_type: dict[str, list[int]] = {}
    per_candidate: list[float] = []
    for record in records:
        out = serialize_request(record, tokenizer)
        totals.append(len(out["tokens"]))
        states.append(out["state_end"])
        types = {q["id"]: q["type"] for q in record["request"]["questions"]}
        sizes: dict[str, int] = {}
        lines: dict[str, list[int]] = {}
        for segment in out["segments"]:
            if segment["owner"] < 0:
                continue
            question_id = out["question_ids"][segment["owner"]]
            sizes[question_id] = sizes.get(question_id, 0) + segment_tokens(segment)
            if segment["name"].startswith("candidate:"):
                lines.setdefault(question_id, []).append(segment_tokens(segment))
        for question_id, size in sizes.items():
            per_question.append(size)
            by_type.setdefault(types[question_id], []).append(size)
            per_candidate.append(statistics.fmean(lines[question_id]))
    return {
        "requests": len(records),
        "total_tokens": summary(totals),
        "state_tokens": summary(states),
        "question_tokens": summary(per_question),
        "question_tokens_by_type": {kind: summary(values) for kind, values in sorted(by_type.items())},
        "tokens_per_candidate": summary(per_candidate),
    }


# --------------------------------------------------------------------------
# (c) 합성 스트림 에피소드 — 장면 빌더는 tests/test_harness.py의 obj/observation 복제 (모듈 설명의 결합 규칙)
# --------------------------------------------------------------------------

COLOURS = ("red", "blue", "green", "yellow", "purple", "orange", "cyan", "pink", "brown", "grey")

#: 이 스크립트만의 차이: 영역 둘 (sim 설정 `zones.count_min: 2`). 나머지는 test_harness와 같다.
ZONES = [
    {"id": "zoneL", "desc": "왼쪽 정리 영역", "bounds_mm": [-120, 150, 180, 330]},
    {"id": "zoneR", "desc": "오른쪽 정리 영역", "bounds_mm": [-120, -330, 180, -150]},
]


def obj(object_id: str, pos_mm, **over) -> dict:
    entry = {
        "id": object_id,
        "class": "box",
        "shape": "box",
        "colour": "red",
        "pos_mm": list(pos_mm),
        "quat": [0.0, 0.0, 0.0, 1.0],
        "obb_mm": [60, 60, 64],
        "visible": True,
        "visible_ratio": 1.0,
        "attributes": [],
        "last_seen_ms": None,
        "desc": None,
    }
    entry.update(over)
    return entry


def observation(objects: list[dict], **over) -> dict:
    base = {
        "tick": 0,
        "sim_time_ms": 0,
        "instruction": {"version": 1, "t_ms": 0, "text": "red 상자를 왼쪽 정리 영역으로 옮겨라"},
        "objects": objects,
        "zones": copy.deepcopy(ZONES),
        "robot": {
            "ee_pos_mm": [0, 0, 200],
            "ee_quat": [0.0, 0.0, 0.0, 1.0],
            "gripper_mm": 80,
            "holding": None,
            "contact_force_n": 0.0,
            "speed_mm_s": 0,
        },
        "events": [],
        "exec": {"seq": 0, "executor": "HOLD", "action_ref": None, "phase": None},
        "ack": None,
    }
    base.update(over)
    for entry in base["objects"]:
        if entry.get("last_seen_ms") is None:
            entry["last_seen_ms"] = base["sim_time_ms"] if entry["visible"] else 0
        if entry.get("desc") is None:
            entry["desc"] = f"{entry['colour']} 상자"
    return base


def scene(n_objects: int) -> list[dict]:
    """`test_candidate_cap_and_inclusion_rate`의 무리 배치. 취약·금지 속성 하나씩."""
    objects = [
        obj(f"o{index}", (140 + 30 * index, -300 + 60 * index, -80), colour=COLOURS[index % len(COLOURS)])
        for index in range(n_objects)
    ]
    objects[1]["attributes"] = ["fragile"]
    return objects


def harness_with_cap(k_cap: int | None) -> RobotHarness:
    """K 상한을 바꾼 하네스. `None`이면 상한을 풀어(실행 가능한 후보 전부) 상한 없는 상계를 잰다."""
    config = copy.deepcopy(load_harness_config())
    config["candidates"]["max"] = int(k_cap) if k_cap is not None else 10_000
    return RobotHarness(config)


INSTRUCTION_V2 = "blue 상자를 오른쪽 정리 영역으로 옮기고 green 상자는 건드리지 마라"

#: 합성 에피소드의 작은 변화 일정: 외란(물체 o3가 +x 30mm), 가림(o1이 3틱), 말단 이동(틱마다 10mm, 대상 접근점까지).
DISTURBANCE_TICK = 15
OCCLUSION_TICKS = (20, 21, 22)
EE_STEP_MM = 10


def synthetic_observation(n_objects: int, index: int, *, instruction_change: bool, change_tick: int | None) -> dict:
    """틱 `index`의 관측: 정적 무리 배치에 작은 변화 일정을 얹는다 (모듈 설명 (c))."""
    objects = scene(n_objects)
    if index >= DISTURBANCE_TICK and n_objects > 3:
        objects[3]["pos_mm"][0] += 30
    if index in OCCLUSION_TICKS:
        objects[1].update(visible=False, visible_ratio=0.2)
    target = objects[0]["pos_mm"]
    approach = [target[0], target[1], target[2] + 32 + 60]
    start = [0, 0, 200]
    full = math.dist(start, approach)
    travelled = min(EE_STEP_MM * index, full)
    fraction = travelled / full if full else 0.0
    ee = [round(s + (a - s) * fraction) for s, a in zip(start, approach)]
    over: dict[str, Any] = {"tick": index, "sim_time_ms": 100 * index}
    if instruction_change and change_tick is not None and index >= change_tick:
        over["instruction"] = {"version": 2, "t_ms": 100 * change_tick, "text": INSTRUCTION_V2}
    out = observation(objects, **over)
    out["robot"]["ee_pos_mm"] = ee
    out["robot"]["speed_mm_s"] = EE_STEP_MM * 10 if 0 < travelled < full else 0
    for entry in out["objects"]:
        if not entry["visible"]:
            entry["last_seen_ms"] = 100 * (min(OCCLUSION_TICKS) - 1)
    return out


def synthetic_episode(
    n_objects: int, k_cap: int | None, *, instruction_change: bool, ticks: int, change_tick: int | None = None
) -> dict[str, Any]:
    """`ticks`틱의 합성 에피소드 — 같은 장면 빌더, 지시의 대상×영역 파지에 commitment가 틱마다 이어진다.

    `instruction_change`면 `change_tick`(기본: 마지막 틱 앞)부터 지시 v2가 실리고 그 틱의 첫 토큰이 지시 조각이 된다.
    """
    if ticks < 1:
        raise ValueError("ticks는 1 이상")
    if instruction_change:
        if change_tick is None:
            change_tick = max(1, ticks - 1)
        if not 1 <= change_tick < ticks:
            raise ValueError(f"change_tick은 1 이상 ticks({ticks}) 미만이어야 한다 (받은 값: {change_tick})")
    hrn = harness_with_cap(k_cap)
    first_obs = synthetic_observation(n_objects, 0, instruction_change=False, change_tick=None)
    first = hrn.build_request(first_obs, None, None)
    grasp_key = "grasp:o0:top:zoneL"
    grasp = next((c for c in first["request"]["candidates"]["q_main"] if c["key"] == grasp_key), None)
    exec_history = {
        "adopted": {"main": candidate_id("hold"), "phase": "none", "path": "p0", "speed": 0, "force": 0, "gripper": "open", "stop": False},
        "ack": {"applied": True},
    }
    cap = "none" if k_cap is None else k_cap
    record = new_episode(
        f"ep-synth-{n_objects}-{cap}-{'chg' if instruction_change else 'same'}",
        "scene-family-b2",
        instructions=[dict(first_obs["instruction"])],
        question_set=hrn.question_set_id(),
    )
    append_tick(record, first)
    for index in range(1, ticks):
        observation_ = synthetic_observation(n_objects, index, instruction_change=instruction_change, change_tick=change_tick)
        commitment = (
            {"action_ref": grasp["id"], "key": grasp_key, "phase": "approach", "held_ticks": index, "last_switch_tick": 0}
            if grasp
            else None
        )
        append_tick(record, hrn.build_request(observation_, exec_history, commitment))
    return record


def synthetic_record(n_objects: int, k_cap: int | None, instruction_change: bool) -> dict:
    """옛 B2의 2틱 레코드 (지시 변경은 틱 1) — 검사·호환용."""
    return synthetic_episode(n_objects, k_cap, instruction_change=instruction_change, ticks=2, change_tick=1 if instruction_change else None)


def episode_profile(out: dict, ticks: list[dict], *, change_tick: int | None, rules: dict[str, Any]) -> dict[str, Any]:
    """틱 종류별 토큰: 첫 틱, 통상 틱, 갱신 틱(동적 주기·지시 텍스트 주기), 소개 틱(소개 주기), 지시 변경 틱, 100틱 구간 합."""
    period_dynamic = int(rules["object_dynamic_period_ticks"])
    period_intro = int(rules["object_intro_period_ticks"])
    period_text = int(rules["goal_text_period_ticks"])

    def kind(tick: dict) -> str:
        index = tick["index"]
        if index == 0:
            return "first"
        if change_tick is not None and index == change_tick:
            return "instruction_change"
        if period_intro and index % period_intro == 0:
            return "intro"
        if (period_dynamic and index % period_dynamic == 0) or (period_text and index % period_text == 0):
            return "refresh"
        return "typical"

    groups: dict[str, list[dict]] = {}
    for tick in ticks:
        groups.setdefault(kind(tick), []).append(tick)
    profile = {
        name: {"ticks": len(group), "new_tokens": summary([t["new_tokens"] for t in group]), "sections": section_means(group)}
        for name, group in groups.items()
    }
    return {
        "prefix_tokens": out["prefix_end"],
        "ticks": len(ticks),
        "k_actual": ticks[0]["k"].get("q_main"),
        "all_ticks": summary([t["new_tokens"] for t in ticks]),
        "by_kind": profile,
        "chunk_100_ticks": sum(t["new_tokens"] for t in ticks[:100]) if len(ticks) >= 100 else None,
        "first_tick_sections": ticks[0]["sections"],
    }


def measure_synthetic(tokenizer: Any, *, ticks: int = 100) -> list[dict[str, Any]]:
    cells = []
    for n_objects, k_cap in SYNTHETIC_CELLS:
        for change in (False, True):
            change_tick = max(1, ticks // 2) if change else None
            record = synthetic_episode(n_objects, k_cap, instruction_change=change, ticks=ticks, change_tick=change_tick)
            out = serialize_request(record, tokenizer, layout="stream_l1a", limits=None if k_cap is not None else UNCAPPED_LIMITS)
            breakdown = [tick_breakdown(out, tick) for tick in out["ticks"]]
            cells.append(
                {
                    "objects": n_objects,
                    "k_cap": k_cap if k_cap is not None else "none",
                    "instruction_change": change,
                    "change_tick": change_tick,
                    "tick0": breakdown[0],
                    "tick1": breakdown[1] if len(breakdown) > 1 else None,
                    **episode_profile(out, breakdown, change_tick=change_tick, rules=out["delta_rules"]),
                }
            )
    return cells


# --------------------------------------------------------------------------
# (d) 실제 에피소드 배치
# --------------------------------------------------------------------------


def measure_episodes(tokenizer: Any, records: list[dict]) -> dict[str, Any]:
    """실제 배치의 틱당 토큰 분포·구간별 평균·100틱 구간 합 (에피소드별과 전체)."""
    episodes = []
    all_ticks: list[dict] = []
    chunks: list[int] = []
    for record in records:
        out = serialize_request(record, tokenizer, layout="stream_l1a")
        ticks = [tick_breakdown(out, tick) for tick in out["ticks"]]
        all_ticks.extend(ticks)
        for start in range(0, len(ticks), 100):
            block = ticks[start : start + 100]
            if len(block) == 100:
                chunks.append(sum(t["new_tokens"] for t in block))
        objects = [len(tick["request"]["state"].get("objects") or []) for tick in record["ticks"]]
        episodes.append(
            {
                "episode_id": record.get("episode_id"),
                "profile": (record.get("provenance") or {}).get("profile"),
                "ticks": len(ticks),
                "objects": max(objects) if objects else 0,
                "prefix_tokens": out["prefix_end"],
                "instruction_changes": len(out["instruction_positions"]) - 1,
                "new_tokens_per_tick": summary([t["new_tokens"] for t in ticks]),
                "first_tick": ticks[0]["new_tokens"],
                "sections_mean": section_means(ticks),
                "k": summary([t["k"].get("q_main", 0) for t in ticks]),
            }
        )
    return {
        "episodes": len(episodes),
        "ticks": len(all_ticks),
        "new_tokens": summary([t["new_tokens"] for t in all_ticks]),
        "first_tick": summary([e["first_tick"] for e in episodes]),
        "prefix_tokens": summary([e["prefix_tokens"] for e in episodes]),
        "chunk_100_ticks": summary(chunks),
        "sections_mean": section_means(all_ticks),
        "k": summary([t["k"].get("q_main", 0) for t in all_ticks]),
        "per_episode": episodes,
    }


# --------------------------------------------------------------------------
# 판정과 표
# --------------------------------------------------------------------------


def _check(measured: Any, budget: float) -> dict[str, Any]:
    return {"measured": measured, "budget": budget, "holds": measured is not None and measured <= budget}


def verdicts(report: dict[str, Any]) -> dict[str, Any]:
    """계약 v0.3 예산과의 대조 — 10물체·K=12 셀(지시 변경 유무 둘 다)에서, 실제 배치가 있으면 그것도."""
    cells = [c for c in report["synthetic"] if (c["objects"], c["k_cap"]) == BUDGET_CELL]
    checks: dict[str, dict[str, Any]] = {}
    for cell in cells:
        name = "instruction_change" if cell["instruction_change"] else "same_instruction"
        typical = cell["by_kind"].get("typical", {}).get("new_tokens", {})
        checks[name] = {
            "tick_p50": _check(cell["all_ticks"]["p50"], BUDGET["tick_p50"]),
            "tick_p95": _check(cell["all_ticks"]["p95"], BUDGET["tick_p95"]),
            "first_tick": _check(cell["tick0"]["new_tokens"], BUDGET["first_tick"]),
            "chunk_100_ticks": _check(cell["chunk_100_ticks"], BUDGET["chunk_100_ticks"]),
            "typical_tick_p50": _check(typical.get("p50"), BUDGET["tick_p50"]),
        }
    episodes = report.get("episodes")
    if episodes and episodes.get("ticks"):
        checks["real_episodes"] = {
            "tick_p50": _check(episodes["new_tokens"]["p50"], BUDGET["tick_p50"]),
            "tick_p95": _check(episodes["new_tokens"]["p95"], BUDGET["tick_p95"]),
            "first_tick_max": _check(episodes["first_tick"].get("max"), BUDGET["first_tick"]),
            "chunk_100_ticks_max": _check(episodes["chunk_100_ticks"].get("max"), BUDGET["chunk_100_ticks"]) if episodes["chunk_100_ticks"].get("n") else {"measured": None, "budget": BUDGET["chunk_100_ticks"], "holds": True},
        }
    return {
        "budget": dict(BUDGET),
        "cell": {"objects": BUDGET_CELL[0], "k_cap": BUDGET_CELL[1]},
        "checks": checks,
        "holds": all(check["holds"] for group in checks.values() for check in group.values()),
    }


def fmt(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.0f}" if value == int(value) else f"{value:.1f}"
    return str(value)


def print_table(report: dict[str, Any]) -> None:
    tok = report["tokenizer"]
    print(f"\n[B2] tokenizer = {tok['id']} @ {tok.get('revision')}  serializer = {report['serializer']}  format = {report['format']}\n")

    print("(a) D0 스트림 에피소드 (옛 서식의 손 fixture — 참고)")
    print(f"{'episode':<12}{'ticks':>6}{'objs':>6}{'prefix':>8}{'tick p50':>10}{'tick p95':>10}{'tick max':>10}{'state p50':>11}{'q_main p50':>12}{'decisions':>11}")
    s = report["streams"]
    for e in s["episodes"]:
        nt = e["new_tokens_per_tick"]
        print(
            f"{e['episode_id']:<12}{e['ticks']:>6}{'/'.join(map(str, e['objects'])):>6}{e['prefix_tokens']:>8}"
            f"{fmt(nt['p50']):>10}{fmt(nt['p95']):>10}{fmt(nt['max']):>10}{fmt(e['state_per_tick']['p50']):>11}"
            f"{fmt(e['q_main_block_per_tick']['p50']):>12}{fmt(e['decisions_per_tick']['p50']):>11}"
        )
    a = s["all_ticks"]
    print(f"  all ticks: new tokens p50={fmt(a['new_tokens']['p50'])} p95={fmt(a['new_tokens']['p95'])} max={fmt(a['new_tokens']['max'])}; "
          f"state p50={fmt(a['state']['p50'])}; tokens/q_main candidate mean={fmt(a['tokens_per_q_main_candidate']['mean'])}")

    print("\n(b) D0 단일 요청 64건")
    b = report["singles"]
    print(f"  total p50={fmt(b['total_tokens']['p50'])} p95={fmt(b['total_tokens']['p95'])} max={fmt(b['total_tokens']['max'])}; "
          f"state p50={fmt(b['state_tokens']['p50'])} max={fmt(b['state_tokens']['max'])}")

    print(f"\n(c) 합성 스트림 에피소드 ({report['synthetic'][0]['ticks']}틱; 첫 틱은 prefix를 포함하지 않는다)")
    print(f"{'objs':>5}{'Kcap':>6}{'K':>4}{'instr':>7}{'prefix':>8}{'first':>7}{'typ p50':>9}{'typ p95':>9}{'refresh':>9}{'intro':>7}{'change':>8}{'all p50':>9}{'all p95':>9}{'chunk100':>10}")
    for c in report["synthetic"]:
        kinds = c["by_kind"]
        typical = kinds.get("typical", {}).get("new_tokens", {})

        def mean_of(name: str) -> str:
            return fmt(kinds.get(name, {}).get("new_tokens", {}).get("mean"))

        print(
            f"{c['objects']:>5}{str(c['k_cap']):>6}{c['k_actual']:>4}{'yes' if c['instruction_change'] else 'no':>7}{c['prefix_tokens']:>8}"
            f"{c['tick0']['new_tokens']:>7}{fmt(typical.get('p50')):>9}{fmt(typical.get('p95')):>9}{mean_of('refresh'):>9}{mean_of('intro'):>7}"
            f"{mean_of('instruction_change'):>8}{fmt(c['all_ticks']['p50']):>9}{fmt(c['all_ticks']['p95']):>9}{fmt(c['chunk_100_ticks']):>10}"
        )
    cell = next((c for c in report["synthetic"] if (c["objects"], c["k_cap"]) == BUDGET_CELL and not c["instruction_change"]), None)
    if cell is not None:
        print("\n  구간별 평균 토큰 (10물체·K=12, 지시 변경 없음):")
        for name in ("first", "typical", "refresh", "intro"):
            sections = cell["by_kind"].get(name, {}).get("sections", {})
            print(f"    {name:<9}" + " ".join(f"{k}={fmt(v)}" for k, v in sections.items()))

    if report.get("episodes"):
        e = report["episodes"]
        print(f"\n(d) 실제 에피소드 {e['episodes']}편 · {e['ticks']}틱: tick p50={fmt(e['new_tokens']['p50'])} p95={fmt(e['new_tokens']['p95'])} "
              f"max={fmt(e['new_tokens']['max'])}; first tick p50={fmt(e['first_tick'].get('p50'))} max={fmt(e['first_tick'].get('max'))}; "
              f"prefix p50={fmt(e['prefix_tokens'].get('p50'))}; chunk100 p50={fmt(e['chunk_100_ticks'].get('p50'))} max={fmt(e['chunk_100_ticks'].get('max'))}; K mean={fmt(e['k'].get('mean'))}")
        print("    sections mean: " + " ".join(f"{k}={fmt(v)}" for k, v in e["sections_mean"].items()))

    print("\n판정 (계약 v0.3 예산, 10물체·K=12)")
    for name, checks in report["verdicts"]["checks"].items():
        print(f"  {name}:")
        for check, value in checks.items():
            print(f"    {check:<20} measured={fmt(value['measured'])}  budget={value['budget']}  holds={value['holds']}")
    print(f"  holds = {report['verdicts']['holds']}")


# --------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tokenizer", help="tokenizer id 또는 경로 (기본: artifacts/tokenizers의 manifest)")
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--ticks", type=int, default=100, help="합성 에피소드의 틱 수 (10초 구간 = 100)")
    parser.add_argument("--episodes", type=Path, default=None, help="실제 에피소드 배치 디렉터리 (episodes/*/streams.jsonl)")
    args = parser.parse_args(argv)

    if args.tokenizer:
        identifier = args.tokenizer
        tokenizer = load_tokenizer(identifier)
        meta: dict[str, Any] = {"id": identifier}
    else:
        found = available_tokenizer()
        if found is None:
            print(f"실제 tokenizer가 없다 — `uv run python {FETCH_SCRIPT}`로 받는다", file=sys.stderr)
            return 1
        identifier, _ = found
        tokenizer = load_tokenizer(identifier)
        manifest = tokenizer_root() / MANIFEST_NAME
        meta = json.loads(manifest.read_text(encoding="utf-8")) if manifest.is_file() else {"id": identifier}
        meta = {key: meta.get(key) for key in ("id", "revision", "files")}

    report: dict[str, Any] = {
        "tokenizer": meta,
        "serializer": TOKEN_SERIALIZER_VERSION,
        "format": STREAM_FORMAT,
        "budget": dict(BUDGET),
        "streams": measure_streams(tokenizer, read_jsonl(D0_STREAMS)),
        "singles": measure_singles(tokenizer, read_jsonl(D0)),
        "synthetic": measure_synthetic(tokenizer, ticks=args.ticks),
    }
    if args.episodes is not None:
        records = []
        for path in sorted(args.episodes.glob("episodes/*/streams.jsonl")):
            records.extend(read_jsonl(path))
        report["episodes"] = measure_episodes(tokenizer, records)
        report["episodes"]["dataset"] = str(args.episodes)
    report["verdicts"] = verdicts(report)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print_table(report)
    print(f"\n→ {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
