"""B2 — 실제 tokenizer로 틱당 토큰을 잰다 (docs/08 §3.4의 추정과 대조).

세 가지를 잰다.

(a) D0 스트림 4 에피소드(`tests/fixtures/d0_streams.jsonl`): prefix 토큰, 틱당 새 토큰
    p50/p95/max, 그 틱의 K에서의 후보 블록, 결정 위치 부담.
(b) D0 단일 요청 64건(`tests/fixtures/d0.jsonl`): 전체 토큰 p50/p95/max, 질문별 T_i 크기.
(c) 합성 스트림 요청: `RobotHarness.build_request`로 물체 6·10개 × K=12·32 × 지시 변경
    유무의 장면을 만들어 prefix·틱·후보 블록·결정 위치를 잰다.

**장면 빌더의 결합 규칙.** 아래 `obj`/`observation`은 `tests/test_harness.py`의 같은 이름
빌더를 **복제**한 것이다(`tests/`는 패키지가 아니고, 그 파일은 다른 브랜치에서 고쳐지고 있어
import로 묶지 않는다). 두 곳이 어긋나면 이 스크립트의 장면이 하네스 검사가 쓰는 관측과 달라져
측정이 다른 것을 재게 되므로, `tests/test_measure_tokens.py`가 두 빌더의 출력이 같은지(영역
목록만 이 스크립트의 `ZONES`로 다름) 검사로 대조한다. `tests/test_harness.py`의 빌더를 바꾸면
여기도 같이 바꾼다.

결과는 `artifacts/reports/tokens-b2.json`(git 제외)에 쓰고 표로 찍는다. 이 스크립트는
하네스를 import한다 — 모델 코드는 하지 않는다 (docs/06 §1).

실행: `uv run python scripts/measure_tokens.py [--tokenizer <id|path>] [--out …]`
"""

from __future__ import annotations

import argparse
import copy
import json
import statistics
import sys
from pathlib import Path
from typing import Any

from robo_jev.data.episode import append_tick, new_episode
from robo_jev.harness.robot import RobotHarness, candidate_id, load_harness_config
from robo_jev.model.serialize import TOKEN_SERIALIZER_VERSION, serialize_request, state_lines
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

#: docs/08 §3.4의 추정치. 여기서는 대조만 하고 문서는 고치지 않는다.
ESTIMATES = {
    "prefix": (600, 1000),
    "per_tick": (450, 700),
    "candidate_block_k32": 480,
    "tokens_per_object": 40,
    "tokens_per_candidate": 15,
    "decision_positions": 10,
}


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
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def count(tokenizer: Any, text: str) -> int:
    return len(tokenizer.encode(text, add_special_tokens=False).ids)


# --------------------------------------------------------------------------
# 구간별 집계
# --------------------------------------------------------------------------


def segment_tokens(segment: dict) -> int:
    return int(segment["end"] - segment["start"])


def tick_breakdown(out: dict, tick: dict) -> dict[str, Any]:
    """틱 하나의 새 토큰을 구간 종류별로 나눈다."""
    own = [s for s in out["segments"] if s["tick"] == tick["index"]]
    by_name: dict[str, int] = {}
    candidate_blocks: dict[str, int] = {}
    candidate_lines: dict[str, list[int]] = {}
    for segment in own:
        name = segment["name"]
        tokens = segment_tokens(segment)
        if name.startswith("candidate:"):
            _, question_id, _ = name.split(":")
            candidate_blocks[question_id] = candidate_blocks.get(question_id, 0) + tokens
            candidate_lines.setdefault(question_id, []).append(tokens)
        elif name.startswith("candidates:"):
            question_id = name.split(":")[1]
            candidate_blocks[question_id] = candidate_blocks.get(question_id, 0) + tokens
        elif name.startswith("decision:"):
            by_name["decisions"] = by_name.get("decisions", 0) + tokens
        elif name.startswith("instruction:"):
            by_name["instruction_change"] = by_name.get("instruction_change", 0) + tokens
        else:
            by_name[name] = by_name.get(name, 0) + tokens
    new_tokens = tick["end"] - tick["start"]  # 도중 지시 조각은 그 틱의 토큰이라 이미 들어 있다
    change = by_name.get("instruction_change", 0)
    return {
        "t": tick["t"],
        "new_tokens": new_tokens,
        "new_tokens_without_instruction_change": new_tokens - change,
        "state": by_name.get("state", 0),
        "commitment": by_name.get("commitment", 0),
        "exec_history": by_name.get("exec_history", 0),
        "instruction_change": change,
        "decisions": by_name.get("decisions", 0),
        "posed": len(tick["posed"]),
        "k": {qid: len(ids) for qid, ids in tick["candidate_mapping"].items() if qid in candidate_blocks},
        "candidate_block": candidate_blocks,
        "tokens_per_candidate": {
            qid: round(statistics.fmean(lines), 1) for qid, lines in candidate_lines.items()
        },
    }


def state_line_breakdown(tokenizer: Any, state: dict) -> dict[str, Any]:
    """상태 줄을 종류별로 토큰화한다 — 물체 한 줄이 몇 토큰인지 보기 위해."""
    groups: dict[str, list[int]] = {}
    for line in state_lines(state):
        head = line.split(" ", 1)[0].split("=", 1)[0]
        groups.setdefault(head, []).append(count(tokenizer, line + "\n"))
    return {
        head: {"lines": len(values), "tokens": sum(values), "per_line": round(statistics.fmean(values), 1)}
        for head, values in groups.items()
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
                "first_tick_state_lines": state_line_breakdown(tokenizer, record["ticks"][0]["request"]["state"]),
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
# (c) 합성 스트림 요청 — 장면 빌더는 tests/test_harness.py의 obj/observation 복제 (모듈 설명의 결합 규칙)
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


def harness_with_cap(k_cap: int) -> RobotHarness:
    config = copy.deepcopy(load_harness_config())
    config["candidates"]["max"] = int(k_cap)
    return RobotHarness(config)


def synthetic_record(n_objects: int, k_cap: int, instruction_change: bool) -> dict:
    hrn = harness_with_cap(k_cap)
    first = hrn.build_request(observation(scene(n_objects)), None, None)
    hold = candidate_id("hold")
    grasp_key = f"grasp:o0:top:zoneL:slow"
    grasp = next((c for c in first["request"]["candidates"]["q_main"] if c["key"] == grasp_key), None)
    commitment = (
        {"action_ref": grasp["id"], "key": grasp_key, "phase": "approach", "held_ticks": 1, "last_switch_tick": 0}
        if grasp
        else None
    )
    exec_history = {
        "adopted": {"main": hold, "phase": "none", "path": "p0", "speed": 0, "force": 0, "gripper": "open", "stop": False},
        "ack": {"applied": True},
    }
    second_obs = observation(scene(n_objects), tick=1, sim_time_ms=100)
    if instruction_change:
        second_obs["instruction"] = {
            "version": 2,
            "t_ms": 100,
            "text": "blue 상자를 오른쪽 정리 영역으로 옮기고 green 상자는 건드리지 마라",
        }
    second = hrn.build_request(second_obs, exec_history, commitment)

    record = new_episode(
        "ep-b2",
        "scene-family-b2",
        instructions=[dict(observation(scene(n_objects))["instruction"])],
        question_set=hrn.question_set_id(),
    )
    append_tick(record, first)
    append_tick(record, second)
    return record


def measure_synthetic(tokenizer: Any) -> list[dict[str, Any]]:
    cells = []
    for n_objects in (6, 10):
        for k_cap in (12, 32):
            for change in (False, True):
                record = synthetic_record(n_objects, k_cap, change)
                out = serialize_request(record, tokenizer, layout="stream_l1a")
                ticks = [tick_breakdown(out, tick) for tick in out["ticks"]]
                first_state = record["ticks"][0]["request"]["state"]
                cells.append(
                    {
                        "objects": n_objects,
                        "k_cap": k_cap,
                        "instruction_change": change,
                        "k_actual": ticks[0]["k"].get("q_main"),
                        "prefix_tokens": out["prefix_end"],
                        "tick0": ticks[0],
                        "tick1": ticks[1],
                        "state_lines_tick0": state_line_breakdown(tokenizer, first_state),
                    }
                )
    return cells


# --------------------------------------------------------------------------
# 판정과 표
# --------------------------------------------------------------------------


#: 점 추정치와의 대조 허용 폭 (양쪽 ±15%).
TOLERANCE = 0.15


def within(value: float, bounds: tuple[float, float]) -> bool:
    return bounds[0] <= value <= bounds[1]


def near(value: float, estimate: float) -> bool:
    return abs(value - estimate) <= TOLERANCE * estimate


def verdicts(report: dict[str, Any]) -> dict[str, Any]:
    streams = report["streams"]
    synthetic = report["synthetic"]
    prefix_values = [e["prefix_tokens"] for e in streams["episodes"]] + [c["prefix_tokens"] for c in synthetic]
    tick_p50 = streams["all_ticks"]["new_tokens"]["p50"]
    tick_p95 = streams["all_ticks"]["new_tokens"]["p95"]
    synth_ticks = [c["tick0"]["new_tokens"] for c in synthetic]
    k32 = [c["tick0"]["candidate_block"]["q_main"] for c in synthetic if c["k_actual"] == 32]
    per_object = [
        c["state_lines_tick0"]["object"]["per_line"] for c in synthetic if "object" in c["state_lines_tick0"]
    ]
    return {
        "prefix": {
            "estimate": ESTIMATES["prefix"],
            "measured": {"min": min(prefix_values), "max": max(prefix_values)},
            "holds": all(within(v, ESTIMATES["prefix"]) for v in prefix_values),
            "note": "prefix는 지시 1개 + 질문 세트 v0 텍스트 + 정적 후보다. 추정 하한보다 작다.",
        },
        "per_tick": {
            "estimate": ESTIMATES["per_tick"],
            "measured": {
                "d0_streams_p50": tick_p50,
                "d0_streams_p95": tick_p95,
                "synthetic_min": min(synth_ticks),
                "synthetic_max": max(synth_ticks),
            },
            "holds": all(within(v, ESTIMATES["per_tick"]) for v in synth_ticks) and within(tick_p95, ESTIMATES["per_tick"]),
        },
        "candidate_block_k32": {
            "estimate": ESTIMATES["candidate_block_k32"],
            "measured": {"min": min(k32) if k32 else None, "max": max(k32) if k32 else None},
            "holds": bool(k32) and all(near(v, ESTIMATES["candidate_block_k32"]) for v in k32),
        },
        "tokens_per_object": {
            "estimate": ESTIMATES["tokens_per_object"],
            "measured": {"min": min(per_object), "max": max(per_object)},
            "holds": all(near(v, ESTIMATES["tokens_per_object"]) for v in per_object),
        },
        "decision_positions": {
            "estimate": ESTIMATES["decision_positions"],
            "measured": {"synthetic": [c["tick0"]["decisions"] for c in synthetic]},
            "holds": all(c["tick0"]["decisions"] == c["tick0"]["posed"] for c in synthetic),
            "note": "결정 위치는 묻는 질문당 1토큰이다. 후보 없는 동적 질문은 그 틱에 묻지 않는다.",
        },
    }


def fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.0f}" if value == int(value) else f"{value:.1f}"
    return str(value)


def print_table(report: dict[str, Any]) -> None:
    tok = report["tokenizer"]
    print(f"\n[B2] tokenizer = {tok['id']} @ {tok.get('revision')}  serializer = {report['serializer']}\n")

    print("(a) D0 스트림 에피소드")
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
    print(f"  all 400 ticks: new tokens p50={fmt(a['new_tokens']['p50'])} p95={fmt(a['new_tokens']['p95'])} max={fmt(a['new_tokens']['max'])}; "
          f"state p50={fmt(a['state']['p50'])}; tokens/q_main candidate mean={fmt(a['tokens_per_q_main_candidate']['mean'])}")
    print("  q_main block by K: " + ", ".join(f"K={k}: {v['mean']} ({v['per_candidate']}/cand, n={v['n']})" for k, v in s["q_main_block_by_k"].items()))

    print("\n(b) D0 단일 요청 64건")
    b = report["singles"]
    print(f"  total p50={fmt(b['total_tokens']['p50'])} p95={fmt(b['total_tokens']['p95'])} max={fmt(b['total_tokens']['max'])}; "
          f"state p50={fmt(b['state_tokens']['p50'])} max={fmt(b['state_tokens']['max'])}")
    print(f"  T_i p50={fmt(b['question_tokens']['p50'])} p95={fmt(b['question_tokens']['p95'])} max={fmt(b['question_tokens']['max'])}; "
          + "; ".join(f"{kind} p50={fmt(v['p50'])} max={fmt(v['max'])}" for kind, v in b["question_tokens_by_type"].items()))

    print("\n(c) 합성 스트림 요청 (하네스 build_request; tick1은 지시 변경 조각을 포함한 새 토큰, 'of which instr'는 그 조각)")
    print(f"{'objs':>5}{'Kcap':>6}{'K':>4}{'instr':>7}{'prefix':>8}{'tick0':>7}{'tick1':>7}{'of which':>10}{'state':>7}{'q_main':>8}{'q_path':>8}{'dec':>5}{'tok/obj':>9}{'tok/cand':>10}")
    for c in report["synthetic"]:
        t0, t1 = c["tick0"], c["tick1"]
        print(
            f"{c['objects']:>5}{c['k_cap']:>6}{c['k_actual']:>4}{'yes' if c['instruction_change'] else 'no':>7}{c['prefix_tokens']:>8}"
            f"{t0['new_tokens']:>7}{t1['new_tokens']:>7}{t1['instruction_change']:>10}{t0['state']:>7}"
            f"{t0['candidate_block'].get('q_main', 0):>8}{t0['candidate_block'].get('q_path', 0):>8}{t0['decisions']:>5}"
            f"{fmt(c['state_lines_tick0'].get('object', {}).get('per_line', 0)):>9}{fmt(t0['tokens_per_candidate'].get('q_main', 0)):>10}"
        )

    print("\n판정 (docs/08 §3.4 추정과 대조)")
    for name, v in report["verdicts"].items():
        print(f"  {name:<22} estimate={v['estimate']}  measured={v['measured']}  holds={v['holds']}")


# --------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tokenizer", help="tokenizer id 또는 경로 (기본: artifacts/tokenizers의 manifest)")
    parser.add_argument("--out", default=str(DEFAULT_OUT))
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

    report = {
        "tokenizer": meta,
        "serializer": TOKEN_SERIALIZER_VERSION,
        "estimates": {key: list(value) if isinstance(value, tuple) else value for key, value in ESTIMATES.items()},
        "streams": measure_streams(tokenizer, read_jsonl(D0_STREAMS)),
        "singles": measure_singles(tokenizer, read_jsonl(D0)),
        "synthetic": measure_synthetic(tokenizer),
    }
    report["verdicts"] = verdicts(report)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print_table(report)
    print(f"\n→ {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
