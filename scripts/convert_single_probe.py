#!/usr/bin/env python
"""Task v0.1b — 일반 능력 시험 G1: 옛 줄기의 비로봇 단일 요청(`judgment-v0`)을 판단 요청 기록으로 바꾸고, 두 run을 묶음별로 비교하고, 섞기용 학습 파일을 만든다.

    # 변환 (CPU) — 봉인 G1(test + ood_test)과 섞기용(train)을 manifest와 함께
    uv run python scripts/convert_single_probe.py convert --input artifacts/datasets/d1/single/records.jsonl --out-dir artifacts/simgen/probes
    # 두 run의 묶음별 짝지은 비교 (CPU) — 진단 도구(scripts/diagnose_decisions.py)가 G1에서 낸 예측 둘: a − b
    uv run python scripts/convert_single_probe.py compare --input artifacts/simgen/probes/G1.jsonl \
        --a <v0>.predictions.jsonl --b <base>.predictions.jsonl --labels v0 base --out <cmp>.json
    # 섞기용 학습 파일 (CPU) — 학습 파일 뒤에 G1-train-mix를 그대로 이어 붙이고 섞인 몫(예 기준)을 manifest에 적는다
    uv run python scripts/convert_single_probe.py mix --base artifacts/simgen/v01/train-50k.jsonl --probe artifacts/simgen/probes/G1-train-mix.jsonl \
        --sealed artifacts/simgen/probes/G1.jsonl --train-config configs/train/decisions-v0-lora-h100.yaml --out <mix>.jsonl

규칙은 :mod:`robo_jev.decisions.single_probe` 의 모듈 설명이다. 모든 변환본은 진단 도구·학습기와 같은 형식 검사
(:func:`robo_jev.decisions.records.validate_record`)를 통과해야 하고, 상태 글은 원래 dict로 되돌아가야 한다(둘 중 하나라도 어긋나면 쓰지 않는다).

``compare``는 두 예측이 같은 프롬프트 판·형식·배치·shuffle seed로 채점됐는지 먼저 본다(다르면 거절 — ``--allow-mismatch``로만 넘긴다).
같은 질문 위에서 전체와 분야·언어·원래 종류·분할(test / ood_test)·정답 kind·신뢰도·판단 종류별로 정확도·Brier·코드 질량의 a − b와
origin group 단위 bootstrap 95 % 구간을 낸다(:func:`robo_jev.decisions.single_probe.compare_probe`).

``mix``는 학습기의 ``mix`` 설정(지금은 빈 목록이어야 하는 자리)을 건드리지 않는다 — 데이터를 이어 붙인다. 학습 파일을 **앞에**, 섞기용을
**뒤에** 둔다: 학습기의 dev는 파일 순서의 앞 ``dev_max_records``개라 섞지 않은 run과 같은 dev 레코드를 쓴다(dev 손실을 나란히 본다).
섞기용 레코드가 G1 분할(test·ood_test)이거나 봉인 G1과 origin group을 나누거나 id가 학습 파일과 겹치면 쓰지 않는다.
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


#: 두 예측이 같아야 하는 채점 조건 (예측 줄마다 진단 도구가 적는다).
SCORING_KEYS = ("mode", "format", "prompt_version", "orders", "shuffle_seed")


def _conditions(predictions: list[dict[str, Any]]) -> dict[str, Any]:
    found: dict[str, set] = {key: {json.dumps(p.get(key)) for p in predictions} for key in (*SCORING_KEYS, "model")}
    return {key: (json.loads(next(iter(values))) if len(values) == 1 else sorted(values)) for key, values in found.items()}


def cmd_compare(args: argparse.Namespace) -> int:
    from robo_jev.decisions.records import file_sha256, read_records, validate_record
    from robo_jev.decisions.single_probe import compare_probe, render_probe_comparison

    records = read_records(args.input)
    invalid = [r.get("id") for r in records if validate_record(r)]
    if invalid:
        raise SystemExit(f"{args.input}: {len(invalid)} records fail validate_record (first {invalid[:3]}) — not a converted probe file")
    a, b = read_records(args.a), read_records(args.b)
    cond_a, cond_b = _conditions(a), _conditions(b)
    mismatch = {key: (cond_a[key], cond_b[key]) for key in SCORING_KEYS if cond_a[key] != cond_b[key]}
    if mismatch and not args.allow_mismatch:
        raise SystemExit(f"the two runs were not scored the same way: {mismatch} (use --allow-mismatch to compare anyway)")
    ids = {str(p["id"]) for p in a} | {str(p["id"]) for p in b}
    records = [r for r in records if str(r["id"]) in ids]
    comparison = compare_probe(records, a, b, cluster=args.cluster, n_boot=args.n_boot, seed=args.seed)
    payload = {
        "config": {"input": str(args.input), "input_sha256": file_sha256(args.input), "a": str(args.a), "b": str(args.b),
                   "a_sha256": file_sha256(args.a), "b_sha256": file_sha256(args.b), "labels": list(args.labels), "conditions": {"a": cond_a, "b": cond_b},
                   "mismatch": mismatch, "script_version": SCRIPT_VERSION, "git": _git()["commit"]},
        "comparison": comparison,
    }  # fmt: skip
    _write_json(args.out, payload)
    note = f"Scoring conditions differ: {mismatch}" if mismatch else (
        f"Both runs: prompt {cond_a['prompt_version']}, {cond_a['format']} format, {cond_a['orders']} listings, shuffle seed {cond_a['shuffle_seed']}, "
        f"{cond_a['mode']} reading. a = {cond_a['model']}; b = {cond_b['model']}.")
    title = args.title or f"G1 general-ability probe — {args.labels[0]} vs {args.labels[1]}"
    args.out.with_suffix(".md").write_text(render_probe_comparison(comparison, label_a=args.labels[0], label_b=args.labels[1], title=title, note=note),
                                           encoding="utf-8")
    _log(f"compared {comparison['records']} records ({comparison['readings']['single_order']['questions']} questions) → {args.out}")
    return 0


def _examples(records: list[dict[str, Any]], q: int) -> int:
    from robo_jev.decisions.examples import examples_per_epoch

    return examples_per_epoch(records, q=q)


def cmd_mix(args: argparse.Namespace) -> int:
    from robo_jev.decisions.examples import split_dev
    from robo_jev.decisions.records import file_sha256
    from robo_jev.decisions.single_probe import G1_SPLITS, MIX_SPLITS, overlap
    from robo_jev.decisions.train import load_training_records, resolve_config

    base, base_check = load_training_records(args.base)
    probe, probe_check = load_training_records(args.probe)
    if probe_check["invalid"] or not probe:
        raise SystemExit(f"{args.probe}: {probe_check['invalid']} invalid records — not a converted mix file")
    wrong = sorted({str((r.get("meta") or {}).get("split")) for r in probe} - set(MIX_SPLITS))
    if wrong:
        raise SystemExit(f"{args.probe}: records from splits {wrong} — only {list(MIX_SPLITS)} may be mixed into training (G1 = {list(G1_SPLITS)})")
    clash = {str(r["id"]) for r in base} & {str(r["id"]) for r in probe}
    if clash:
        raise SystemExit(f"{len(clash)} record ids appear in both files (first {sorted(clash)[:3]})")
    sealed = None
    if args.sealed is not None:
        g1, _ = load_training_records(args.sealed)
        crossing = overlap(g1, probe)
        if crossing["origin_groups"] or crossing["source_records"]:
            raise SystemExit(f"{args.probe} shares origin groups / records with the sealed {args.sealed}: {crossing}")
        if {str(r["id"]) for r in g1} & {str(r["id"]) for r in base}:
            raise SystemExit(f"{args.base} contains sealed G1 record ids")
        sealed = {"path": str(args.sealed), "sha256": file_sha256(args.sealed), "overlap": {"origin_groups": 0, "source_records": 0, "base_ids": 0}}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temp = args.out.with_name(args.out.name + ".tmp")
    with temp.open("wb") as out:
        for source in (args.base, args.probe):
            data = source.read_bytes()
            out.write(data)
            if data and not data.endswith(b"\n"):
                out.write(b"\n")
    temp.replace(args.out)
    config = resolve_config(_read_yaml(args.train_config)) if args.train_config else resolve_config({})
    q = int(config["questions_per_request"])
    train_base, dev_base = split_dev(base, fraction=config["dev_fraction"], salt=config["dev_salt"])
    train_probe, dev_probe = split_dev(probe, fraction=config["dev_fraction"], salt=config["dev_salt"])
    cap = config["dev_max_records"]
    dev_used_probe = max(0, min(len(dev_probe), (cap - len(dev_base)) if cap else len(dev_probe)))
    examples = {"base": _examples(train_base, q), "probe": _examples(train_probe, q)}
    total = examples["base"] + examples["probe"]
    batch = int(config["batch_examples"])
    manifest = {
        "kind": "decisions-train-mix-v0", "script_version": SCRIPT_VERSION, "git": _git(),
        "out": {"path": str(args.out), "sha256": file_sha256(args.out), "bytes": args.out.stat().st_size, "records": len(base) + len(probe)},
        "order": "base file first, then the probe file (byte-for-byte) — the trainer's dev set is the first dev_max_records dev records in file order, "
                 "so a mixed run keeps the unmixed run's dev records",
        "base": {"path": str(args.base), "sha256": file_sha256(args.base), "records": len(base), "validation": base_check},
        "probe": {"path": str(args.probe), "sha256": file_sha256(args.probe), "records": len(probe), "validation": probe_check},
        "sealed_check": sealed,
        "train_config": {"path": str(args.train_config) if args.train_config else None, "questions_per_request": q, "dev_fraction": config["dev_fraction"],
                         "dev_salt": config["dev_salt"], "dev_max_records": cap, "batch_examples": batch},
        "split": {"base_train": len(train_base), "base_dev": len(dev_base), "probe_train": len(train_probe), "probe_dev": len(dev_probe),
                  "probe_dev_used": dev_used_probe, "probe_dev_unused": len(dev_probe) - dev_used_probe},
        "examples_per_epoch": {**examples, "total": total, "probe_share": examples["probe"] / total if total else None},
        "records_share": len(train_probe) / (len(train_base) + len(train_probe)) if (train_base or train_probe) else None,
        "steps_per_epoch": {"base_only": -(-examples["base"] // batch), "mixed": -(-total // batch)},
        "created": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
    }  # fmt: skip
    _write_json(args.out.with_name(args.out.stem + ".manifest.json"), manifest)
    _log(f"mix: {len(base)} + {len(probe)} records → {args.out} · probe share {manifest['examples_per_epoch']['probe_share']:.4f} of examples/epoch")
    return 0


def _read_yaml(path: Path) -> dict[str, Any]:
    import yaml

    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    convert = sub.add_parser("convert", help="judgment-v0 → G1(test + ood_test, 봉인) · G1-train-mix(train) + manifest (CPU)")
    convert.add_argument("--input", type=Path, required=True, help="judgment-v0 records.jsonl (옆의 manifest.json도 sha256을 적는다)")
    convert.add_argument("--out-dir", type=Path, required=True)
    convert.set_defaults(func=cmd_convert)
    compare = sub.add_parser("compare", help="두 run(같은 G1)의 묶음별 짝지은 비교 a − b (CPU)")
    compare.add_argument("--input", type=Path, required=True, help="G1.jsonl (변환본)")
    compare.add_argument("--a", type=Path, required=True, help="run a의 .predictions.jsonl (scripts/diagnose_decisions.py)")
    compare.add_argument("--b", type=Path, required=True, help="run b의 .predictions.jsonl")
    compare.add_argument("--labels", nargs=2, default=("a", "b"), metavar=("LABEL_A", "LABEL_B"))
    compare.add_argument("--out", type=Path, required=True, help="비교 JSON (옆에 .md)")
    compare.add_argument("--cluster", default="origin_group", choices=("origin_group", "source_record", "record"))
    compare.add_argument("--n-boot", type=int, default=2000)
    compare.add_argument("--seed", type=int, default=0)
    compare.add_argument("--title", default=None)
    compare.add_argument("--allow-mismatch", action="store_true", help="두 예측의 프롬프트 판·형식·배치·seed가 달라도 비교한다 (보고서에 적힌다)")
    compare.set_defaults(func=cmd_compare)
    mix = sub.add_parser("mix", help="학습 파일 + G1-train-mix를 이어 붙인 학습 입력과 manifest (CPU)")
    mix.add_argument("--base", type=Path, required=True, help="학습 파일 (예: artifacts/simgen/v01/train-50k.jsonl)")
    mix.add_argument("--probe", type=Path, required=True, help="G1-train-mix.jsonl")
    mix.add_argument("--sealed", type=Path, default=None, help="봉인 G1.jsonl — 주면 섞기용과 origin group·원본 레코드가 겹치지 않는지 본다")
    mix.add_argument("--train-config", type=Path, default=None, help="학습 설정 YAML — 주면 그 dev 나눔·q로 epoch당 예 수와 섞인 몫을 정확히 센다")
    mix.add_argument("--out", type=Path, required=True)
    mix.set_defaults(func=cmd_mix)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
