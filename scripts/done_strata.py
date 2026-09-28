#!/usr/bin/env python
"""`q_done` 거짓 done의 기제(A0)와 틱 층의 수(A1) — Task R6 Stage A (docs/08 §7 `q_done`, :mod:`robo_jev.data.done_strata`).

    # A0: R5 폐루프 기록에서 거짓 done 편의 사실 + 놓은 뒤 ≤ 6틱의 모델 `q_done` + g2 train의 놓기 창 세 수
    uv run python scripts/done_strata.py mechanism --out artifacts/reports/r6-a0-false-done.json
    # A1: 데이터셋마다 층별 틱 수 (봉인 분할은 편 수만) + 모델 답이 있는 기록이면 층별 모델 `q_done` 참 비율
    uv run python scripts/done_strata.py count --dataset g2=artifacts/datasets/r1-robot/r1-rollout-labels-g2 \
        --dataset dagger-0=artifacts/datasets/r5-dagger/dagger-0 --out artifacts/reports/r6-a1-done-strata.json
    # B1(수정 라운드 1): 루프의 거짓 done 상태(이전 대상을 이전 영역에 막 놓은 틱)가 판정 칸과 루프 기록에 몇 틱 있고 모델이 거기서 무엇을 답하는가
    uv run python scripts/done_strata.py loop-state --out artifacts/reports/r6-b1-loop-state.json

GPU를 쓰지 않는다. 데이터셋·기록은 읽기만 한다.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from robo_jev.data.done_strata import (  # noqa: E402
    DONE_GATE_THRESHOLD, DONE_STRATA, RELEASE_WINDOW_TICKS, SEALED_SPLITS, done_strata_by_split, false_done_facts, model_q_done,
    model_rates_by_stratum, old_target_release_ticks, old_target_releases, release_window_counts,
)

DATASETS = REPO / "artifacts" / "datasets"
#: R5의 모델 주행 기록 (조건 → 디렉터리). `ood_dev`는 **읽기만** 한다 — 기제의 셈이지 학습 재료가 아니다.
R5_CONDITIONS = {
    "dev_new": DATASETS / "r5-closed-loop" / "r5" / "dev_new",
    "ood_dev": DATASETS / "r5-closed-loop" / "r5" / "ood_dev",
    "dev (seen)": DATASETS / "r5-closed-loop" / "r5" / "dev",
}
G2 = DATASETS / "r1-robot" / "r1-rollout-labels-g2"


def _git() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def load_open_records(directory: Path) -> tuple[list[dict[str, Any]], dict[str, dict[str, int]]]:
    """디렉터리의 **봉인되지 않은** 레코드와 봉인 분할의 편 수. 봉인 분할(`ood_test`)의 파일은 **읽지 않는다** — manifest의 `files` 항목이
    가진 `split`으로 먼저 거르고(편 수만 센다), 남은 파일만 연다. manifest가 없거나 어떤 에피소드 항목에 `split`이 없으면 파일을 열지 않고는
    분할을 알 수 없으므로 **거절한다**(읽고 버리지 않는다 — 리뷰 1 I-1). Task R7 A3부터 규칙의 정본은 :func:`robo_jev.data.sealed.read_open_episodes`다."""
    from robo_jev.data.sealed import read_open_episodes

    return read_open_episodes(directory)


def _records(directory: Path) -> list[dict[str, Any]]:
    return load_open_records(directory)[0]


def build_mechanism(conditions: dict[str, Path] | None = None, *, g2: Path = G2) -> dict[str, Any]:
    """A0 — 거짓 done 편의 사실(조건별·합계), 층별 모델 `q_done` 참 비율, g2 train의 놓기 창 세 수."""
    conditions = conditions or R5_CONDITIONS
    out: dict[str, Any] = {"task": "r6-a0-false-done", "generated_at": _now(), "git": _git(), "window_ticks": RELEASE_WINDOW_TICKS,
                           "conditions": {}, "g2_train": None}
    everything: list[dict[str, Any]] = []
    for name, directory in conditions.items():
        records = _records(directory)
        everything.extend(records)
        out["conditions"][name] = {
            "records_dir": str(directory), "episodes": len(records),
            "false_done": false_done_facts(records),
            "model_rates_by_stratum": model_rates_by_stratum(records),
            "release_window": release_window_counts(records),
        }
    facts = false_done_facts(everything)
    out["all_conditions"] = {"episodes": len(everything), "false_done": facts["false_done"], "summary": facts["summary"]}
    if Path(g2).is_dir():
        train = [record for record in _records(g2) if record.get("split") == "train"]
        out["g2_train"] = {"dataset": str(g2), "episodes": len(train), "release_window": release_window_counts(train)}
    return out


def build_counts(datasets: dict[str, Path]) -> dict[str, Any]:
    """A1 — 데이터셋마다 분할별 층 수(봉인 분할은 편 수만)와, 모델 답이 있는 레코드면 층별 모델 `q_done` 참 비율."""
    out: dict[str, Any] = {"task": "r6-a1-done-strata", "generated_at": _now(), "git": _git(), "strata": list(DONE_STRATA),
                           "window_ticks": RELEASE_WINDOW_TICKS, "datasets": {}}
    for name, directory in datasets.items():
        open_records, sealed = load_open_records(Path(directory))
        block = {"dataset": str(directory), **done_strata_by_split(open_records), "sealed": sealed}
        if any((tick.get("model_output") or {}).get("q_done") is not None or (tick.get("usage") or {}).get("model_q_done") is not None
               for record in open_records[:5] for tick in record["ticks"]):
            block["model_rates_by_stratum"] = model_rates_by_stratum(open_records)
        out["datasets"][name] = block
    return out


#: B1의 판정 칸과 그 칸에 대한 두 run의 저장된 예측 (`q_done` 틱별 예측이 있는 보고서).
CELL_SUITE = REPO / "configs" / "eval" / "r6-decision-cell.yaml"
CELL_REPORTS = {"R5": REPO / "artifacts" / "reports" / "r6-reeval-2b-t1-fp32-r5.json", "R6": REPO / "artifacts" / "reports" / "r6-reeval-2b-t1-fp32-r6.json"}
#: 모델이 운전한 폐루프 기록 (이름 → 디렉터리). R5의 셋은 B1의 문장이 인용한 것이고, 새 seed의 R5·R6 줄은 같은 자로 나란히 둔다.
LOOPS = {
    "R5 × ood_dev (R4 26)": DATASETS / "r5-closed-loop" / "r5" / "ood_dev",
    "R5 × dev_new (R5 100)": DATASETS / "r5-closed-loop" / "r5" / "dev_new",
    "R5 × dev (R4 100, seen)": DATASETS / "r5-closed-loop" / "r5" / "dev",
    "R5 × ood_dev_new (R6 74)": DATASETS / "r6-closed-loop" / "r5" / "ood_dev_new",
    "R5 × dev_new2 (R6 100)": DATASETS / "r6-closed-loop" / "r5" / "dev_new2",
    "R6 × ood_dev (R4 26)": DATASETS / "r6-closed-loop" / "r6" / "ood_dev",
    "R6 × ood_dev_new (R6 74)": DATASETS / "r6-closed-loop" / "r6" / "ood_dev_new",
    "R6 × dev_new2 (R6 100)": DATASETS / "r6-closed-loop" / "r6" / "dev_new2",
}
#: 놓은 틱과 그 뒤 두 틱 — "놓은 뒤 3틱 안에 `q_done` ≥ 0.5가 들었는가"의 창.
FOLLOW_TICKS = 3


def _median(values: list[int]) -> float | None:
    ordered = sorted(values)
    if not ordered:
        return None
    middle = len(ordered) // 2
    return float(ordered[middle]) if len(ordered) % 2 else (ordered[middle - 1] + ordered[middle]) / 2.0


def _release_block(records: list[dict[str, Any]], *, answers: Any = None) -> dict[str, Any]:
    """이전 대상을 이전 영역에 놓은 사건: 수, 지금 지시가 시작된 뒤 틱의 중앙값·범위, (모델 답이 있으면) 놓은 틱과 그 뒤 두 틱 안에 `q_done` ≥ 0.5."""
    events = [(record, event) for record in records for event in old_target_releases(record)]
    since = [event["ticks_since_instruction_change"] for _, event in events]
    block: dict[str, Any] = {"releases": len(events), "episodes": len({str(record.get("episode_id")) for record, _ in events}),
                             "ticks_since_instruction_change": {"median": _median(since), "min": min(since) if since else None, "max": max(since) if since else None}}
    if answers is not None:
        followed = 0
        for record, event in events:
            window = range(event["tick"], min(event["tick"] + FOLLOW_TICKS, len(record["ticks"])))
            followed += int(any((answers(record, index) or 0.0) >= DONE_GATE_THRESHOLD for index in window))
        block["followed_by_q_done_within_3_ticks"] = followed
    return block


def build_loop_state(*, suite: Path = CELL_SUITE, reports: dict[str, Path] | None = None, loops: dict[str, Path] | None = None) -> dict[str, Any]:
    """B1 (수정 라운드 1, 리뷰 1 U-5) — 루프의 거짓 done 상태(:func:`robo_jev.data.done_strata.old_target_release_ticks`)가 **전문가가 운전한 판정 칸**과
    **모델이 운전한 폐루프 기록**에 몇 틱 있고, 거기서 모델의 `q_done`이 무엇을 답했는가. 칸은 저장된 틱별 예측(모델·세 섞기 대조군)으로, 루프는 기록의
    모델 raw 답으로 센다."""
    from robo_jev.evaluate import load_eval_suite

    reports = reports if reports is not None else CELL_REPORTS
    loops = loops if loops is not None else LOOPS
    loaded = load_eval_suite(suite)
    entry = next(item for item in loaded["splits"] if item["name"] == "robot/ood_dev")
    base = (REPO / entry["manifest"]).parent
    from robo_jev.data.sealed import read_episode_ids

    records = read_episode_ids(base, list(entry["records"]))  # manifest로 먼저 — 봉인 편은 열지 않고 거절 (R7 A3)
    state = {(str(record["episode_id"]), index) for record in records for index, flag in enumerate(old_target_release_ticks(record)) if flag}
    cell: dict[str, Any] = {"suite": str(suite.relative_to(REPO)), "split": entry["name"], "episodes": len(records), "ticks": len(state),
                            "episodes_with_the_state": len({episode for episode, _ in state}), "releases": _release_block(records), "answers_true": {}}
    for name, path in reports.items():
        if not Path(path).is_file():
            cell["answers_true"][name] = None
            continue
        table = json.loads(Path(path).read_text(encoding="utf-8"))["evaluation"]["splits"][entry["name"]]
        cell["answers_true"][name] = {
            column: sum(1 for row in ((table.get(column) or {}).get("q_done") or {}).get("per_record") or () if (str(row["record_id"]), int(row["tick"])) in state and row["predicted"] == "true")
            for column in ("model", "context_shuffle", "instruction_shuffle", "commitment_shuffle")
        }
    out: dict[str, Any] = {"task": "r6-b1-loop-state", "generated_at": _now(), "git": _git(), "window_ticks": RELEASE_WINDOW_TICKS,
                           "definition": "ticks within 0-6 of releasing an object that is not the current target, is the target of an earlier instruction, and lies inside that instruction's zone (not held) — the state the 25 R5 false dones ended in",
                           "cell": cell, "loops": {}}
    for name, directory in loops.items():
        if not Path(directory).is_dir():
            out["loops"][name] = None
            continue
        loop_records, _ = load_open_records(Path(directory))
        flagged = [(record, index) for record in loop_records for index, flag in enumerate(old_target_release_ticks(record)) if flag]
        high = sum(1 for record, index in flagged if (model_q_done(record["ticks"][index]) or 0.0) >= DONE_GATE_THRESHOLD)
        out["loops"][name] = {"records_dir": str(Path(directory)), "episodes": len(loop_records), "ticks": len(flagged),
                              "episodes_with_the_state": len({str(record.get("episode_id")) for record, _ in flagged}), "model_q_done_over_gate": high,
                              "releases": _release_block(loop_records, answers=lambda record, index: model_q_done(record["ticks"][index]))}
    return out


def print_counts(payload: dict[str, Any], file: Any = None) -> None:
    file = file or sys.stdout
    print("| dataset | split | episodes | labelled | " + " | ".join(DONE_STRATA) + " |", file=file)
    print("| --- | --- | ---: | ---: | " + " | ".join("---:" for _ in DONE_STRATA) + " |", file=file)
    for name, block in payload["datasets"].items():
        rows = [("all non-sealed", block)] + [(split, entry) for split, entry in block["by_split"].items()]
        for split, entry in rows:
            print(f"| {name} | {split} | {entry['episodes']} | {entry['labelled_ticks']:,} | "
                  + " | ".join(f"{entry['strata'][s]:,} ({entry['episodes_with'][s]})" for s in DONE_STRATA) + " |", file=file)
        rates = block.get("model_rates_by_stratum")
        if rates:
            print(f"| {name} | model `q_done` ≥ 0.5 | | | " + " | ".join(
                f"{rates[s]['model_true']}/{rates[s]['answered']}" + (f" = {rates[s]['rate']:.3f}" if rates[s]["rate"] is not None else "") for s in DONE_STRATA) + " |", file=file)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    mechanism = sub.add_parser("mechanism", help="A0: R5 기록의 거짓 done 기제")
    mechanism.add_argument("--out", required=True)
    count = sub.add_parser("count", help="A1: 데이터셋별 층 수")
    count.add_argument("--dataset", action="append", required=True, metavar="NAME=DIR")
    count.add_argument("--out", required=True)
    loop_state = sub.add_parser("loop-state", help="B1: 루프의 거짓 done 상태 — 판정 칸 대 모델 주행 기록")
    loop_state.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    if args.command == "mechanism":
        payload = build_mechanism()
        summary = payload["all_conditions"]
        print(f"false dones {summary['false_done']} of {summary['episodes']} · {json.dumps(summary['summary'], ensure_ascii=False)}")
        for name, block in payload["conditions"].items():
            rate = block["model_rates_by_stratum"]["post_release_other"]
            print(f"{name}: false dones {block['false_done']['false_done']} · post_release_other model q_done ≥ 0.5 {rate['model_true']}/{rate['answered']} · window {block['release_window']}")
        if payload["g2_train"]:
            print(f"g2 train release window: {payload['g2_train']['release_window']}")
    elif args.command == "loop-state":
        payload = build_loop_state()
        cell = payload["cell"]
        print(f"cell: {cell['ticks']} ticks / {cell['episodes_with_the_state']} episodes · releases {cell['releases']} · answers true {cell['answers_true']}")
        for name, block in payload["loops"].items():
            if block:
                print(f"{name}: {block['model_q_done_over_gate']}/{block['ticks']} ticks q_done ≥ 0.5 · releases {block['releases']}")
    else:
        datasets = {pair.split("=", 1)[0]: Path(pair.split("=", 1)[1]) for pair in args.dataset}
        payload = build_counts(datasets)
        print_counts(payload)
    target = Path(args.out)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=1, default=str) + "\n", encoding="utf-8")
    print(f"→ {target}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
