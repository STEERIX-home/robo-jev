#!/usr/bin/env python
"""`q_done` 거짓 done의 기제(A0)와 틱 층의 수(A1) — Task R6 Stage A (docs/08 §7 `q_done`, :mod:`robo_jev.data.done_strata`).

    # A0: R5 폐루프 기록에서 거짓 done 편의 사실 + 놓은 뒤 ≤ 6틱의 모델 `q_done` + g2 train의 놓기 창 세 수
    uv run python scripts/done_strata.py mechanism --out artifacts/reports/r6-a0-false-done.json
    # A1: 데이터셋마다 층별 틱 수 (봉인 분할은 편 수만) + 모델 답이 있는 기록이면 층별 모델 `q_done` 참 비율
    uv run python scripts/done_strata.py count --dataset g2=artifacts/datasets/r1-robot/r1-rollout-labels-g2 \
        --dataset dagger-0=artifacts/datasets/r5-dagger/dagger-0 --out artifacts/reports/r6-a1-done-strata.json

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
    DONE_STRATA, RELEASE_WINDOW_TICKS, SEALED_SPLITS, done_strata_by_split, false_done_facts, model_rates_by_stratum,
    release_window_counts,
)
from robo_jev.data.robot_episodes import read_episodes  # noqa: E402

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
    가진 `split`으로 먼저 거르고(편 수만 센다), 남은 파일만 연다. manifest가 없으면(시험의 작은 디렉터리) 전부 읽고 봉인 분할을 버린다."""
    directory = Path(directory)
    manifest_path = directory / "manifest.json"
    sealed: dict[str, dict[str, int]] = {}
    if manifest_path.is_file():
        entries = (json.loads(manifest_path.read_text(encoding="utf-8")).get("files") or {})
        wanted: list[Path] = []
        for name, entry in entries.items():
            if not str(name).startswith("episodes/") or not isinstance(entry, dict):
                continue
            split = str(entry.get("split"))
            if split in SEALED_SPLITS:
                sealed.setdefault(split, {"episodes": 0})["episodes"] += 1
                continue
            wanted.append(directory / name)
        out: list[dict[str, Any]] = []
        for path in sorted(wanted):
            out.extend(json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
        return out, sealed
    found = [record for _, record in read_episodes(directory)]
    for record in found:
        if str(record.get("split")) in SEALED_SPLITS:
            sealed.setdefault(str(record["split"]), {"episodes": 0})["episodes"] += 1
    return [record for record in found if str(record.get("split")) not in SEALED_SPLITS], sealed


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
