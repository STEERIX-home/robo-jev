#!/usr/bin/env python
"""Task v0.1b — 일반 능력 시험 G1: 옛 줄기의 비로봇 단일 요청(`judgment-v0`)을 판단 요청 기록으로 바꾸고, 두 run을 묶음별로 비교하고, 섞기용 학습 파일을 만든다.

    # 변환 (CPU) — 봉인 G1(test + ood_test)과 섞기용(train)을 manifest와 함께
    uv run python scripts/convert_single_probe.py convert --input artifacts/datasets/d1/single/records.jsonl --out-dir artifacts/simgen/probes

규칙은 :mod:`robo_jev.decisions.single_probe` 의 모듈 설명이다. 모든 변환본은 진단 도구·학습기와 같은 형식 검사
(:func:`robo_jev.decisions.records.validate_record`)를 통과해야 하고, 상태 글은 원래 dict로 되돌아가야 한다(둘 중 하나라도 어긋나면 쓰지 않는다).
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

SCRIPT_VERSION = "v01b-probe-1"
#: 변환본이 기대는 코드 — manifest의 ``git_dirty``는 이 파일들이 HEAD와 다른지다.
CODE_PATHS = ("src/robo_jev/decisions/single_probe.py", "src/robo_jev/decisions/records.py", "scripts/convert_single_probe.py")


def _log(message: str) -> None:
    print(f"[v01b {dt.datetime.now().strftime('%H:%M:%S')}] {message}", file=sys.stderr, flush=True)


def _git() -> dict[str, Any]:
    def run(*args: str) -> str | None:
        try:
            return subprocess.run(["git", "-C", str(REPO), *args], capture_output=True, text=True, check=True).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return None

    status = run("status", "--porcelain", "--", *CODE_PATHS)
    return {"commit": run("rev-parse", "--short", "HEAD"), "dirty": bool(status) if status is not None else None}


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    with temp.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    temp.replace(path)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(payload, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    temp.replace(path)


def _check(converted: list[dict[str, Any]], sources: dict[str, dict[str, Any]]) -> list[str]:
    """형식 검사(진단·학습과 같은 :func:`validate_record`) + 상태 글의 왕복 + id 겹침."""
    from robo_jev.decisions.records import validate_record
    from robo_jev.decisions.single_probe import parse_state_text

    problems = []
    seen: set[str] = set()
    for record in converted:
        rid = record["id"]
        if rid in seen:
            problems.append(f"{rid}: duplicate id")
        seen.add(rid)
        problems.extend(f"{rid}: {p}" for p in validate_record(record))
        source = sources[record["meta"]["source_record"]]
        if parse_state_text(record["request"]["state"]) != source["request"]["state"]:
            problems.append(f"{rid}: state text does not round-trip to the source state")
    return problems


def cmd_convert(args: argparse.Namespace) -> int:
    from robo_jev.decisions.records import file_sha256, read_records
    from robo_jev.decisions.single_probe import CONVERTER_VERSION, G1_SPLITS, MIX_SPLITS, convert_records, overlap

    source = read_records(args.input)
    by_id = {str(r["request"]["request_id"]): r for r in source}
    source_manifest = args.input.with_name("manifest.json")
    git = _git()
    outputs = {}
    for name, splits, role in (("G1", G1_SPLITS, "sealed"), ("G1-train-mix", MIX_SPLITS, "train-mix")):
        converted, stats = convert_records(source, splits=splits)
        problems = _check(converted, by_id)
        if problems:
            raise SystemExit(f"{name}: {len(problems)} problems — first: {problems[:5]}")
        outputs[name] = (converted, stats, splits, role)
    crossing = overlap(outputs["G1"][0], outputs["G1-train-mix"][0])
    if crossing["origin_groups"] or crossing["source_records"]:
        raise SystemExit(f"G1 and G1-train-mix share origin groups / records: {crossing}")
    for name, (converted, stats, splits, role) in outputs.items():
        path = args.out_dir / f"{name}.jsonl"
        _write_jsonl(path, converted)
        manifest = {
            "kind": "decisions-probe-v0", "name": name, "role": role,
            "use": ("SEALED general-ability probe — evaluation only, never training or selection" if role == "sealed"
                    else "training mix candidate (train split only; the controller decides whether v0.1 mixes it)"),
            "file": {"path": str(path), "sha256": file_sha256(path), "bytes": path.stat().st_size, "records": len(converted)},
            "source": {"path": str(args.input), "sha256": file_sha256(args.input), "splits": list(splits),
                       "manifest": str(source_manifest) if source_manifest.is_file() else None,
                       "manifest_sha256": file_sha256(source_manifest) if source_manifest.is_file() else None,
                       "generator": (by_id[next(iter(by_id))].get("provenance") or {}).get("generator") if by_id else None},
            "converter": {"version": CONVERTER_VERSION, "script_version": SCRIPT_VERSION, "git": git["commit"], "git_dirty": git["dirty"]},
            "counts": stats,
            "overlap_with_" + ("G1-train-mix" if name == "G1" else "G1"): {"origin_groups": 0, "source_records": 0},
            "created": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        }  # fmt: skip
        _write_json(args.out_dir / f"{name}.manifest.json", manifest)
        _log(f"{name}: {stats['source_records']} source records → {len(converted)} records · {stats['questions']} questions → {path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    convert = sub.add_parser("convert", help="judgment-v0 → G1(test + ood_test, 봉인) · G1-train-mix(train) + manifest (CPU)")
    convert.add_argument("--input", type=Path, required=True, help="judgment-v0 records.jsonl (옆의 manifest.json도 sha256을 적는다)")
    convert.add_argument("--out-dir", type=Path, required=True)
    convert.set_defaults(func=cmd_convert)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
