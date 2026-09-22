#!/usr/bin/env python
"""사건 밀도 표 — 두 데이터 버전을 같은 자로 잰다 (Task R1 B4의 인수 기준).

세 가지 질문에 답한다.

1. **사건이 얼마나 자주 나는가** — `tick_class` 분포(`goal_change`·`event`·`steady`·`other`; `event`는 세계 사건이
   있는 `event_world`와 전환뿐인 `event_switch`로 나눈다), 세계 사건 종류별 수, 지시 변경 수와 모델이 실제로 보는
   `instruction_changed` 줄 수(틱 안의 사건을 합치기 전에는 뒤가 훨씬 작았다).
2. **읽어야 답이 나오는 틱이 얼마나 되는가** — 정답이 그 틱의 `commitment.action_ref`와 **다른** 틱(P3의 주 층)의
   비율과 키 갈래, 그리고 완료 뒤 꼬리와 정체를 뺀 수. 학습 신호로는 `tick_weights`로 잰 **가중 손실 몫**이 는다.
3. **죽은 구간이 남아 있는가** — 완료 뒤 꼬리 틱 수, `max_ms`로 끝난 편, 정체 감시 발동 수.

    uv run python scripts/event_density.py --dataset artifacts/datasets/r1-robot/r1 \\
        [--dataset artifacts/datasets/d1-robot/d1] [--out artifacts/reports/r1-density.json]

읽기 전용이다 — 데이터셋의 `episodes/*/stream.jsonl`만 읽는다.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from robo_jev.sampler import TICK_CLASSES, tick_class, tick_weights

REPO = Path(__file__).resolve().parents[1]

#: 학습 설정과 같은 틱 가중치 (configs/train/qwen35-2b-pilot.yaml `tick_weights`). 가중 손실 몫을 이것으로 센다.
TICK_WEIGHTS = {"steady": 0.25, "event": 2.0, "goal_change": 2.0, "other": 1.0}

#: 죽은 구간의 기준 — 완료 꼬리가 아닌 `hold` 경로가 이만큼 이어지면 정체로 센다. 하네스의 정체 감시(`m_hold` 15)가
#: 그보다 짧게 끊으므로, 이 수를 넘는 구간이 남았다는 것은 **감시가 일하지 않았다**는 뜻이다.
DEAD_RUN_TICKS = 20

#: B4의 목표 (브리프). 못 미치면 값과 모자란 손잡이를 적는다 — 목표를 맞추려고 빈도를 지어내지 않는다.
TARGETS = {
    "goal_change_share": 0.015,
    "world_event_share": 0.08,
    "reading_share": 0.08,
    "reading_weight_share": 0.30,
    "max_ms_episodes": 0,
    "stalled_episodes": 0,
}


def episodes(dataset: Path) -> list[dict[str, Any]]:
    """`episodes/*/stream.jsonl`의 레코드 (파일 하나에 한 편)."""
    out: list[dict[str, Any]] = []
    for path in sorted((dataset / "episodes").glob("*/*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                out.append(json.loads(line))
    return out


def _labels(tick: dict[str, Any], question_id: str = "q_main") -> list[str] | None:
    label = next((item for item in tick.get("labels") or () if item["question_id"] == question_id), None)
    return list(label.get("candidate_ids") or ()) if label else None


def _done_tail_start(record: dict[str, Any]) -> int:
    """완료 꼬리가 시작하는 틱 색인 (없으면 틱 수). `done` 게이트가 끝까지 이어진 구간의 첫 틱이다."""
    ticks = record["ticks"]
    index = len(ticks)
    while index > 0 and str((ticks[index - 1].get("usage") or {}).get("gate")) == "done":
        index -= 1
    return index


def measure(records: list[dict[str, Any]], name: str) -> dict[str, Any]:
    classes: Counter[str] = Counter()
    world_kinds: Counter[str] = Counter()
    reading_families: Counter[str] = Counter()
    stall_kinds: Counter[str] = Counter()
    ticks = reading = tail_ticks = 0
    instruction_changes = instruction_lines = 0
    weight_total = weight_reading = 0.0
    max_ms_episodes = stalled_episodes = 0
    per_episode_reading: list[float] = []

    for record in records:
        entries = record["ticks"]
        ticks += len(entries)
        weights = tick_weights(record, weights=TICK_WEIGHTS)
        tail_start = _done_tail_start(record)
        tail_ticks += len(entries) - tail_start
        outcome = (record.get("provenance") or {}).get("outcome") or {}
        if str(outcome.get("terminated")) == "max_ms":
            max_ms_episodes += 1
        last_ms = int(entries[-1].get("sim_ms", 0)) if entries else 0
        instruction_changes += sum(
            1 for step in (record.get("prefix") or {}).get("instructions", [])[1:] if int(step.get("t_ms", 0)) <= last_ms
        )
        episode_reading = 0
        dead_run = longest_dead = 0
        for index, tick in enumerate(entries):
            kind = tick_class(record, index)
            state = tick["request"].get("state") or {}
            events = [event for event in state.get("events") or () if isinstance(event, dict)]
            if kind == "event":
                classes["event_world" if events else "event_switch"] += 1
            classes[kind] += 1
            for event in events:
                world_kinds[str(event.get("kind"))] += 1
                if event.get("kind") == "instruction_changed":
                    instruction_lines += 1
            for record_kind, count in ((tick.get("usage") or {}).get("records") or {}).items():
                if record_kind in ("observe_stalled", "hold_stalled", "place_stalled"):
                    stall_kinds[record_kind] += int(count)
            # **정체한 편** = 감시가 끊지 못한 죽은 구간이 남은 편이다. 감시가 걸린 것 자체는 감시가 **일한** 것이므로
            # 따로 센다(`stall_records`) — 그것까지 정체로 세면 고친 것을 고장으로 읽게 된다.
            adopted = tick.get("adopted") or {}
            if str((tick.get("usage") or {}).get("gate")) != "done" and str(adopted.get("path_kind")) == "hold":
                dead_run += 1
                longest_dead = max(longest_dead, dead_run)
            else:
                dead_run = 0
            weight_total += weights[index]
            commitment = tick["request"].get("commitment")
            reference = commitment.get("action_ref") if isinstance(commitment, dict) else None
            labels = _labels(tick)
            if labels is None:
                continue
            if list(labels) != [reference]:
                reading += 1
                episode_reading += 1
                keys = {entry["id"]: entry.get("key") for entry in tick["request"]["candidates"].get("q_main", [])}
                family = str(keys.get(labels[0], "?")).split(":")[0] if labels else "none"
                reading_families[family] += 1
                if index >= tail_start:
                    reading_families["(of which done-tail)"] += 1
                else:
                    # 가중 손실 몫은 **꼬리를 뺀** 읽기 틱으로 센다 (브리프 B4: "꼬리·정체 제외").
                    weight_reading += weights[index]
        stalled_episodes += int(longest_dead > DEAD_RUN_TICKS)
        per_episode_reading.append(episode_reading / max(1, len(entries)))

    reading_outside_tail = reading - reading_families["(of which done-tail)"]
    return {
        "name": name,
        "episodes": len(records),
        "ticks": ticks,
        "tick_class": {key: classes[key] for key in list(TICK_CLASSES) + ["event_world", "event_switch"]},
        "tick_class_share": {
            key: round(classes[key] / max(1, ticks), 4) for key in list(TICK_CLASSES) + ["event_world", "event_switch"]
        },
        "world_event_kinds": dict(world_kinds.most_common()),
        "world_event_ticks": classes["event_world"],
        "world_event_share": round(classes["event_world"] / max(1, ticks), 4),
        "goal_change_share": round(classes["goal_change"] / max(1, ticks), 4),
        "instruction_changes": instruction_changes,
        "instruction_changed_lines": instruction_lines,
        "reading_ticks": reading,
        "reading_share": round(reading / max(1, ticks), 4),
        "reading_outside_tail": reading_outside_tail,
        "reading_share_outside_tail": round(reading_outside_tail / max(1, ticks), 4),
        "reading_by_key_family": dict(reading_families.most_common()),
        "reading_weight_share": round(weight_reading / max(1e-9, weight_total), 4),
        "goal_change_weight_share": round(
            (TICK_WEIGHTS["goal_change"] * classes["goal_change"]) / max(1e-9, weight_total), 4
        ),
        "done_tail_ticks": tail_ticks,
        "done_tail_share": round(tail_ticks / max(1, ticks), 4),
        "max_ms_episodes": max_ms_episodes,
        "stalled_episodes": stalled_episodes,
        "stall_records": dict(stall_kinds.most_common()),
        "mean_ticks_per_episode": round(ticks / max(1, len(records)), 1),
    }


def verdict(row: dict[str, Any]) -> dict[str, Any]:
    """목표 달성 여부 — 못 미치면 값과 함께 그대로 적는다."""
    return {
        "goal_change_share": {"measured": row["goal_change_share"], "target": TARGETS["goal_change_share"], "met": row["goal_change_share"] >= TARGETS["goal_change_share"]},
        "world_event_share": {"measured": row["world_event_share"], "target": TARGETS["world_event_share"], "met": row["world_event_share"] >= TARGETS["world_event_share"]},
        "reading_share_outside_tail": {"measured": row["reading_share_outside_tail"], "target": TARGETS["reading_share"], "met": row["reading_share_outside_tail"] >= TARGETS["reading_share"]},
        "reading_weight_share": {"measured": row["reading_weight_share"], "target": TARGETS["reading_weight_share"], "met": row["reading_weight_share"] >= TARGETS["reading_weight_share"]},
        "max_ms_episodes": {"measured": row["max_ms_episodes"], "target": TARGETS["max_ms_episodes"], "met": row["max_ms_episodes"] <= TARGETS["max_ms_episodes"]},
        "stalled_episodes": {"measured": row["stalled_episodes"], "target": TARGETS["stalled_episodes"], "met": row["stalled_episodes"] <= TARGETS["stalled_episodes"]},
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", action="append", required=True, type=Path, help="데이터셋 디렉터리 (여러 번)")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    rows = [measure(episodes(path), path.name) for path in args.dataset]
    report = {"datasets": rows, "tick_weights": TICK_WEIGHTS, "targets": TARGETS, "verdict": verdict(rows[-1])}
    header = ["metric"] + [row["name"] for row in rows]
    fields = [
        "episodes", "ticks", "mean_ticks_per_episode", "goal_change_share", "world_event_share",
        "instruction_changes", "instruction_changed_lines", "reading_ticks", "reading_share",
        "reading_share_outside_tail", "reading_weight_share", "done_tail_ticks", "done_tail_share",
        "max_ms_episodes", "stalled_episodes",
    ]  # fmt: skip
    width = max(len(name) for name in fields + header)
    print(" | ".join(name.ljust(width) for name in header))
    for field in fields:
        print(" | ".join([field.ljust(width)] + [str(row[field]).ljust(width) for row in rows]))
    for row in rows:
        print(f"\n[{row['name']}] tick_class {row['tick_class']}")
        print(f"[{row['name']}] world events {row['world_event_kinds']}")
        print(f"[{row['name']}] reading families {row['reading_by_key_family']}")
        print(f"[{row['name']}] stall records {row['stall_records']}")
    print("\n판정 (마지막 데이터셋):")
    for name, entry in report["verdict"].items():
        print(f"  {name:<28} measured={entry['measured']} target={entry['target']} met={entry['met']}")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n→ {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
