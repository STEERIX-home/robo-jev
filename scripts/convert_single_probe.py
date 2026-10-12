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
    # Task v0.3c — the same with extra training parts after the base file (``--extra NAME=PATH``, e.g. the MuJoCo set), counted in the
    # shares and the token estimate; the general prefix is sized against base + extras + probe
    uv run python scripts/convert_single_probe.py mix --base artifacts/simgen/v03/engine-s1.jsonl --extra mujoco=artifacts/simgen/v03/mujoco-train.jsonl \
        --probe artifacts/simgen/probes/G1-train-mix.jsonl --sealed artifacts/simgen/probes/G1.jsonl --general artifacts/simgen/v02/general-train.jsonl \
        --general-share 0.25 --train-config configs/train/decisions-v0-lora-h100.yaml --token-sample 3000 --out artifacts/simgen/v03/train-v03.jsonl
    # Task v0.2 — the same, plus the shortest prefix of the general-reading file that makes ~10 % (v02-a) or ~25 % (v02-b) of the
    # train examples per epoch (counted with the trainer's own dev split and q), and a sampled token estimate
    uv run python scripts/convert_single_probe.py mix --base artifacts/simgen/v02/train-50k.jsonl --probe artifacts/simgen/probes/G1-train-mix.jsonl \
        --sealed artifacts/simgen/probes/G1.jsonl --general artifacts/simgen/v02/general-train.jsonl --general-share 0.10 \
        --train-config configs/train/decisions-v0-lora-h100.yaml --token-sample 2000 --out artifacts/simgen/v02/train-v02-a.jsonl

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

SCRIPT_VERSION = "v03-mix-1"
#: 변환본이 기대는 코드 — manifest의 ``git_dirty``는 이 파일들이 HEAD와 다른지다.
CODE_PATHS = ("src/robo_jev/decisions/single_probe.py", "src/robo_jev/decisions/records.py", "scripts/convert_single_probe.py",
              "src/robo_jev/data/general_sets.py")


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


def cmd_sample(args: argparse.Namespace) -> int:
    """공개 세트(judgment-v0)에서 묶음마다 seed로 N개를 뽑아 변환한 봉인 시험 (G2) + manifest."""
    from robo_jev.decisions.records import file_sha256, read_records
    from robo_jev.decisions.single_probe import CONVERTER_VERSION, sample_and_convert, summarize

    paths: dict[str, Path] = {}
    for item in args.source:
        name, separator, path = item.partition("=")
        if not separator or not name or name in paths:
            raise SystemExit(f"--source는 이름=경로 꼴이고 이름이 겹치지 않아야 한다: {item!r}")
        paths[name] = Path(path)
    sources = {name: read_records(path) for name, path in paths.items()}
    by_id = {str(r["request"]["request_id"]): r for records in sources.values() for r in records}
    converted, picked = sample_and_convert(sources, per_source=args.per_source, seed=args.seed)
    problems = _check(converted, by_id)
    if problems:
        raise SystemExit(f"{args.name}: {len(problems)} problems — first: {problems[:5]}")
    expected = {name: set(spec.split(",")) for name, spec in (e.partition("=")[::2] for e in args.expect_types)} if args.expect_types else {}
    by_source_type: dict[str, set[str]] = {}
    for name, spec in picked.items():
        ids = set(spec["request_ids"])
        found = set()
        for record in converted:
            if record["meta"]["source_record"] in ids:
                for question in record["request"]["questions"].values():
                    found.add(f"{question['type']}:{len(question.get('criteria') or {})}" if question["type"] == "choice" else question["type"])
        by_source_type[name] = found
        if name in expected and found != expected[name]:
            raise SystemExit(f"{name}: converted question types {sorted(found)} differ from the expected {sorted(expected[name])}")
    path = args.out_dir / f"{args.name}.jsonl"
    _write_jsonl(path, converted)
    git = _git()
    manifest = {
        "kind": "decisions-probe-v0", "name": args.name, "role": "sealed",
        "use": "SEALED general-ability probe (public sets) — evaluation only, never training or selection",
        "file": {"path": str(path), "sha256": file_sha256(path), "bytes": path.stat().st_size, "records": len(converted)},
        "seed": args.seed, "per_source": args.per_source,
        "sampling": "per source: sha256(seed|name|sample)-seeded random.sample of record indices, restored to source file order",
        "sources": {name: {"path": str(paths[name]), "sha256": file_sha256(paths[name]),
                           "manifest": str(paths[name].with_name("manifest.json")) if paths[name].with_name("manifest.json").is_file() else None,
                           "manifest_sha256": file_sha256(paths[name].with_name("manifest.json")) if paths[name].with_name("manifest.json").is_file() else None,
                           "question_types": sorted(by_source_type[name]), **picked[name]} for name in paths},
        "converter": {"version": CONVERTER_VERSION, "script_version": SCRIPT_VERSION, "git": git["commit"], "git_dirty": git["dirty"]},
        "counts": summarize(converted),
        "created": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
    }  # fmt: skip
    _write_json(args.out_dir / f"{args.name}.manifest.json", manifest)
    drawn = " · ".join(f"{n} {spec['sampled']}/{spec['records_total']}" for n, spec in picked.items())
    _log(f"{args.name}: {drawn} (seed {args.seed}) → {len(converted)} records → {path}")
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
    from robo_jev.decisions.single_probe import PROBE_DIMENSIONS

    comparison = compare_probe(records, a, b, dimensions=args.dimensions or PROBE_DIMENSIONS, cluster=args.cluster, n_boot=args.n_boot, seed=args.seed)
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


def _general_prefix(general: list[dict[str, Any]], *, target: float, fixed: int, config: dict[str, Any], q: int, allow_short: bool = False) -> int:
    """The shortest prefix length of the general file whose **train** examples (after the trainer's own dev split) reach `target` of
    all train examples per epoch, `fixed` being the examples of the other parts (Task v0.2 §1-3). The general file interleaves its
    sources, so a prefix stays balanced. If the whole file cannot reach the target: refuse, or with `allow_short` take the whole file
    (Task v0.3c — the manifest then records the share reached)."""
    from fractions import Fraction

    from robo_jev.decisions.examples import is_dev

    if not 0.0 < target < 1.0:
        raise SystemExit(f"--general-share must be in (0, 1) (got {target})")
    share = Fraction(str(target))  # exact: general / (fixed + general) >= share  <=>  general · (1 − share) >= share · fixed
    examples = 0
    for index, record in enumerate(general):
        if not is_dev(record, fraction=config["dev_fraction"], salt=config["dev_salt"]):
            examples += min(q, len(record["request"]["questions"]))
        if examples * (1 - share) >= share * fixed:
            return index + 1
    if allow_short:
        return len(general)
    raise SystemExit(f"the general file has too few records for a share of {target} (reached {examples} of {float(share / (1 - share) * fixed):.0f} examples)")


def _reaches(general: int, others: int, target: float) -> bool:
    """general / (others + general) >= target, exactly (the prefix rule's comparison)."""
    from fractions import Fraction

    share = Fraction(str(target))
    return general * (1 - share) >= share * others


def _repeated(record: dict[str, Any]) -> dict[str, Any]:
    """A second-pass copy of a general record (``--general-repeat-if-short``): new id ``<id>~r2`` (the trainer refuses repeated ids;
    the new id also gives it its own dev-split draw and option-order seed), ``meta.repeat_of`` = the original id; same request and label."""
    import copy

    out = copy.deepcopy(record)
    out["id"] = f"{record['id']}~r2"
    out["meta"]["repeat_of"] = str(record["id"])
    return out


def _shares_arg(specs: list[str], names: list[str]) -> dict[str, float]:
    """``--extra-share NAME=SHARE`` → name → target share of train examples per epoch."""
    out: dict[str, float] = {}
    for spec in specs:
        name, sep, raw = str(spec).partition("=")
        if not sep or name not in names:
            raise SystemExit(f"--extra-share takes NAME=SHARE for a given --extra NAME (got {spec!r}; extras {names})")
        share = float(raw)
        if not 0.0 < share < 1.0:
            raise SystemExit(f"--extra-share {name}: the share must be in (0, 1) (got {raw})")
        out[name] = share
    return out


def _family(record: dict[str, Any]) -> str:
    meta = record.get("meta") or {}
    return str(meta.get("scene") or meta.get("group") or "all")


def _repeated_group(group: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Second-pass copies of one pair group (Task v0.3c): ids ``<id>~r2``, ``meta.repeat_of`` = the original id, and a pair's ``of``
    pointing at its partner's copy — the copies stay a pair (same dev-split key) and are distinct records to the trainer."""
    import copy

    names = {str(r["id"]) for r in group}
    out = []
    for record in group:
        twin = copy.deepcopy(record)
        twin["id"] = f"{record['id']}~r2"
        twin["meta"]["repeat_of"] = str(record["id"])
        pair = twin["meta"].get("pair")
        if isinstance(pair, dict) and str(pair.get("of")) in names:
            pair["of"] = f"{pair['of']}~r2"
        out.append(twin)
    return out


def _extra_second_passes(extras: list[dict[str, Any]], examples: dict[str, int], shares: dict[str, float], *, general_share: float,
                         config: dict[str, Any], q: int) -> None:
    """``--extra-share NAME=SHARE`` (Task v0.3c): when an extra part's train examples fall short of SHARE of all train examples per
    epoch — the general part being sized afterwards to its own share g — append a recorded second pass of whole pair groups, taken
    round-robin over the part's families (``meta.scene``) in file order, until examples ≥ SHARE · others / (1 − g − SHARE) (+1 example
    for the general prefix's rounding). A copy gets a new id, so the trainer picks its own questions for it (different ones from the
    first pass, by the id-hashed pick). Sets ``copies`` / ``copy_train`` / ``copy_dev`` / ``second_pass`` on each extra."""
    from fractions import Fraction

    from robo_jev.decisions.examples import split_dev
    from robo_jev.decisions.records import pair_groups

    g = Fraction(str(general_share))
    for extra in extras:
        extra.update(copies=[], copy_train=[], copy_dev=[], second_pass=None, target_share=shares.get(extra["name"]),
                     first_pass_examples=examples[extra["name"]])
        if extra["target_share"] is None:
            continue
        s_ = Fraction(str(extra["target_share"]))
        if g + s_ >= 1:
            raise SystemExit(f"--extra-share {extra['name']} {extra['target_share']} + general share {general_share} must stay below 1")
        others = sum(v for k, v in examples.items() if k != extra["name"])
        need = s_ * others / (1 - g - s_) + 1
        if examples[extra["name"]] >= need:
            continue
        queues: dict[str, list[list[dict[str, Any]]]] = {}
        for group in pair_groups(extra["records"]):
            queues.setdefault(_family(group[0]), []).append(group)
        order = []
        while any(queues.values()):
            for family in list(queues):
                if queues[family]:
                    order.append(queues[family].pop(0))
        have, by_family, groups = examples[extra["name"]], {}, 0
        for group in order:
            if have >= need:
                break
            copies = _repeated_group(group)
            train, dev = split_dev(copies, fraction=config["dev_fraction"], salt=config["dev_salt"])
            have += sum(min(q, len(r["request"]["questions"])) for r in train)
            extra["copies"] += copies
            extra["copy_train"] += train
            extra["copy_dev"] += dev
            by_family[_family(group[0])] = by_family.get(_family(group[0]), 0) + len(copies)
            groups += 1
        if have < need:
            raise SystemExit(f"{extra['path']}: even a whole second pass cannot reach --extra-share {extra['target_share']}")
        examples[extra["name"]] = have
        extra["second_pass"] = {"records": len(extra["copies"]), "pair_groups": groups, "records_by_family": dict(sorted(by_family.items())),
                                "train_examples": have - extra["first_pass_examples"],
                                "ids": "<id>~r2 (meta.repeat_of = the original id; a pair's meta.pair.of points at its partner's copy)",
                                "order": "whole pair groups, round-robin over the families (meta.scene) in file order",
                                "why": f"the part's first pass gives {extra['first_pass_examples']} train examples, short of {extra['target_share']} "
                                       f"of all train examples next to the general share {general_share}"}  # fmt: skip


def _load_extras(specs: list[str], *, taken: set[str]) -> list[dict[str, Any]]:
    """``--extra NAME=PATH`` parts (Task v0.3c): every record must pass the trainer's validation and be a train-split record; ids must not
    clash with the other parts; a sibling ``<stem>.manifest.json`` that records a sha256 must match the file."""
    from robo_jev.decisions.records import file_sha256
    from robo_jev.decisions.train import load_training_records

    out: list[dict[str, Any]] = []
    for spec in specs:
        name, sep, raw = str(spec).partition("=")
        if not sep or not name or not raw:
            raise SystemExit(f"--extra takes NAME=PATH (got {spec!r})")
        if name in ("base", "probe", "general", "total") or any(name == extra["name"] for extra in out):
            raise SystemExit(f"--extra name {name!r} is reserved or repeated")
        path = Path(raw)
        records, check = load_training_records(path)
        if check["invalid"] or not records:
            raise SystemExit(f"{path}: {check['invalid']} invalid records of {check['read']} — every extra record must pass the trainer's validation")
        wrong = sorted({str((r.get("meta") or {}).get("split")) for r in records} - {"train"})
        if wrong:
            raise SystemExit(f"{path}: records from splits {wrong} — an extra part holds train-split records only")
        ids = {str(r["id"]) for r in records}
        clash = ids & taken
        if clash:
            raise SystemExit(f"{len(clash)} record ids of {path} appear in another part (first {sorted(clash)[:3]})")
        taken |= ids
        manifest_path = path.with_name(path.stem + ".manifest.json")
        recorded_manifest = None
        if manifest_path.is_file():
            recorded = json.loads(manifest_path.read_text(encoding="utf-8"))
            sha = recorded.get("sha256") or (recorded.get("file") or {}).get("sha256") or (recorded.get("out") or {}).get("sha256")
            if sha is not None and sha != file_sha256(path):
                raise SystemExit(f"{path}: its manifest records sha256 {str(sha)[:12]}… but the file differs — rebuild or regenerate the manifest")
            recorded_manifest = {"path": str(manifest_path), "sha256": file_sha256(manifest_path), "file_sha256_recorded": sha}
        out.append({"name": name, "path": path, "records": records, "validation": check, "manifest": recorded_manifest})
    return out


def _token_estimate(parts: dict[str, list[dict[str, Any]]], examples: dict[str, int], *, sample: int, config: dict[str, Any]) -> dict[str, Any]:
    """Prompt tokens per example (the training prompt — chat format, the run's prompt version, one seeded listing per question), from a
    seeded sample of `sample` train records per part, and the token share of each part = examples × mean tokens. An estimate."""
    import random

    from robo_jev.decisions.prompt import build_prompts
    from robo_jev.decisions.records import shuffle_key
    from robo_jev.model.tokenizer import available_tokenizer, load_tokenizer

    tokenizer = load_tokenizer(available_tokenizer()[0])
    out: dict[str, Any] = {"sample_records_per_part": sample, "sample_seed": "random.Random('tokens|<part>')", "format": config["format"],
                           "prompt_version": config["prompt_version"]}
    totals = {}
    for name, records in parts.items():
        chosen = random.Random(f"tokens|{name}").sample(records, min(sample, len(records))) if records else []
        lengths = []
        for record in chosen:
            for prompt in build_prompts(record["request"], tokenizer, key=shuffle_key(record), seed=1, fmt=config["format"], mode="direct",
                                        orders="seeded", version=config["prompt_version"]):  # fmt: skip
                lengths.append(len(prompt["tokens"]))
        mean = sum(lengths) / len(lengths) if lengths else 0.0
        totals[name] = mean * examples[name]
        out[name] = {"mean_tokens_per_example": round(mean, 1), "est_tokens_per_epoch": round(totals[name])}
    total = sum(totals.values())
    for name in parts:
        out[name]["est_token_share"] = round(totals[name] / total, 4) if total else None
    return out


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
    extras = _load_extras(args.extra or [], taken={str(r["id"]) for r in base} | {str(r["id"]) for r in probe})
    sealed = None
    if args.sealed is not None:
        g1, _ = load_training_records(args.sealed)
        crossing = overlap(g1, probe)
        if crossing["origin_groups"] or crossing["source_records"]:
            raise SystemExit(f"{args.probe} shares origin groups / records with the sealed {args.sealed}: {crossing}")
        if {str(r["id"]) for r in g1} & {str(r["id"]) for r in base}:
            raise SystemExit(f"{args.base} contains sealed G1 record ids")
        for extra in extras:
            if {str(r["id"]) for r in g1} & {str(r["id"]) for r in extra["records"]}:
                raise SystemExit(f"{extra['path']} contains sealed G1 record ids")
        sealed = {"path": str(args.sealed), "sha256": file_sha256(args.sealed), "overlap": {"origin_groups": 0, "source_records": 0, "base_ids": 0}}
    config = resolve_config(_read_yaml(args.train_config)) if args.train_config else resolve_config({})
    q = int(config["questions_per_request"])
    train_base, dev_base = split_dev(base, fraction=config["dev_fraction"], salt=config["dev_salt"])
    train_probe, dev_probe = split_dev(probe, fraction=config["dev_fraction"], salt=config["dev_salt"])
    cap = config["dev_max_records"]
    examples = {"base": _examples(train_base, q), "probe": _examples(train_probe, q)}
    for extra in extras:
        extra["train"], extra["dev"] = split_dev(extra["records"], fraction=config["dev_fraction"], salt=config["dev_salt"])
        examples[extra["name"]] = _examples(extra["train"], q)
    _extra_second_passes(extras, examples, _shares_arg(args.extra_share or [], [e["name"] for e in extras]),
                         general_share=float(args.general_share) if args.general is not None and args.general_share is not None else 0.0,
                         config=config, q=q)
    general_block, general_lines, train_general, dev_general = None, [], [], []
    if args.general is not None:
        from robo_jev.data.general_sets import SOURCES

        if args.general_share is None:
            raise SystemExit("--general needs --general-share (target share of train examples per epoch, e.g. 0.10)")
        general, general_check = load_training_records(args.general)
        if general_check["invalid"] or not general:
            raise SystemExit(f"{args.general}: {general_check['invalid']} invalid records — not a general training file")
        bad = [r["id"] for r in general if (r.get("meta") or {}).get("split") != "train" or ((r.get("meta") or {}).get("source") or {}).get("set") not in SOURCES]
        if bad:
            raise SystemExit(f"{args.general}: {len(bad)} records are not train-split records of a known general source (first {bad[:3]})")
        manifest_path = args.general.with_name(args.general.stem + ".manifest.json")
        general_manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else None
        if general_manifest is None or general_manifest["file"]["sha256"] != file_sha256(args.general):
            raise SystemExit(f"{args.general}: its manifest is missing or does not match the file (sha256) — rebuild with scripts/convert_general_sets.py")
        if any((general_manifest.get("overlap_check") or {}).get("overlap_after_build", {"missing": 1}).values()):
            raise SystemExit(f"{args.general}: the manifest does not record a zero overlap with the sealed sets")
        if args.general_take_all_if_short and args.general_repeat_if_short:
            raise SystemExit("--general-take-all-if-short and --general-repeat-if-short exclude each other")
        pool = general + ([_repeated(r) for r in general] if args.general_repeat_if_short else [])
        n = _general_prefix(pool, target=float(args.general_share), fixed=sum(examples.values()), config=config, q=q,
                            allow_short=bool(args.general_take_all_if_short))
        used = pool[:n]
        clash = {str(r["id"]) for r in used} & ({str(r["id"]) for r in base} | {str(r["id"]) for r in probe}
                                                | {str(r["id"]) for extra in extras for r in extra["records"] + extra["copies"]})
        if clash:
            raise SystemExit(f"{len(clash)} general record ids clash with the other files (first {sorted(clash)[:3]})")
        with args.general.open("rb") as handle:
            general_lines = [line for line in handle if line.strip()][:n]
        general_lines += [(json.dumps(r, ensure_ascii=False) + "\n").encode("utf-8") for r in used[len(general):]]
        train_general, dev_general = split_dev(used, fraction=config["dev_fraction"], salt=config["dev_salt"])
        examples["general"] = _examples(train_general, q)
        from collections import Counter

        general_block = {
            "path": str(args.general), "sha256": file_sha256(args.general), "records_in_file": len(general), "records_used": n,
            "prefix": (f"first {n} of {len(general)} records (the file interleaves its sources)" if n <= len(general) else
                       f"all {len(general)} records, then a second pass over the first {n - len(general)} (the file interleaves its sources)"),
            "target_share": float(args.general_share),
            "target_reached": _reaches(examples["general"], sum(v for k, v in examples.items() if k != "general"), float(args.general_share)),
            "second_pass": {"records": max(0, n - len(general)), "ids": "<id>~r2 (meta.repeat_of = the original id)",
                            "why": "the whole file could not reach the target share; a prefix of the file is used a second time"}
                           if n > len(general) else None,
            "manifest": {"path": str(manifest_path), "sha256": file_sha256(manifest_path), "licenses": general_manifest.get("licenses"),
                         "overlap_after_build": general_manifest["overlap_check"]["overlap_after_build"]},
            "records_by_source": dict(sorted(Counter(r["meta"]["source"]["set"] for r in used).items())),
            "train_examples_by_source": dict(sorted(Counter(r["meta"]["source"]["set"] for r in train_general).items())),
        }  # fmt: skip
    dev_used = {}
    room = (cap - len(dev_base)) if cap else None
    for name, dev in ([(extra["name"], extra["dev"] + extra["copy_dev"]) for extra in extras] + [("probe", dev_probe), ("general", dev_general)]):
        used_now = len(dev) if room is None else max(0, min(len(dev), room))
        dev_used[name] = used_now
        if room is not None:
            room -= used_now
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temp = args.out.with_name(args.out.name + ".tmp")
    with temp.open("wb") as out:
        for source in (args.base, *extras, args.probe):
            data = (source["path"] if isinstance(source, dict) else source).read_bytes()
            out.write(data)
            if data and not data.endswith(b"\n"):
                out.write(b"\n")
            for record in source.get("copies", []) if isinstance(source, dict) else []:
                out.write((json.dumps(record, ensure_ascii=False) + "\n").encode("utf-8"))
        for line in general_lines:
            out.write(line if line.endswith(b"\n") else line + b"\n")
    temp.replace(args.out)
    total = sum(examples.values())
    batch = int(config["batch_examples"])
    epoch = {**examples, "total": total, "probe_share": examples["probe"] / total if total else None}
    for extra in extras:
        epoch[f"{extra['name']}_share"] = examples[extra["name"]] / total if total else None
    if general_block is not None:
        epoch["general_share"] = examples["general"] / total if total else None
    manifest = {
        "kind": "decisions-train-mix-v0", "script_version": SCRIPT_VERSION, "git": _git(),
        "out": {"path": str(args.out), "sha256": file_sha256(args.out), "bytes": args.out.stat().st_size,
                "records": len(base) + sum(len(extra["records"]) + len(extra["copies"]) for extra in extras) + len(probe) + len(general_lines)},
        "order": "base file first, " + "".join(f"then the extra part {extra['name']!r}" + (" and its recorded second pass" if extra["copies"] else "") + ", "
                                               for extra in extras)
                 + "then the probe file (byte-for-byte)" + (", then a prefix of the general file (whole lines)" if general_block else "")
                 + " — the trainer's dev set is the first dev_max_records dev records in file order, so a mixed run keeps the unmixed run's dev records",
        "base": {"path": str(args.base), "sha256": file_sha256(args.base), "records": len(base), "validation": base_check},
        "probe": {"path": str(args.probe), "sha256": file_sha256(args.probe), "records": len(probe), "validation": probe_check},
        "sealed_check": sealed,
        "train_config": {"path": str(args.train_config) if args.train_config else None, "questions_per_request": q, "dev_fraction": config["dev_fraction"],
                         "dev_salt": config["dev_salt"], "dev_max_records": cap, "batch_examples": batch},
        "split": {"base_train": len(train_base), "base_dev": len(dev_base), "probe_train": len(train_probe), "probe_dev": len(dev_probe),
                  "probe_dev_used": dev_used["probe"], "probe_dev_unused": len(dev_probe) - dev_used["probe"]},
        "examples_per_epoch": epoch,
        "records_share": len(train_probe) / (len(train_base) + len(train_probe)) if (train_base or train_probe) else None,
        "steps_per_epoch": {"base_only": -(-examples["base"] // batch), "mixed": -(-total // batch)},
        "created": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
    }  # fmt: skip
    if extras:
        manifest["extra"] = [{k: extra[k] for k in ("name", "manifest", "validation")} | {"path": str(extra["path"]), "sha256": file_sha256(extra["path"]),
                              "records": len(extra["records"]), "train": len(extra["train"]), "dev": len(extra["dev"]),
                              "dev_used": dev_used[extra["name"]], "first_pass_examples": extra["first_pass_examples"],
                              "target_share": extra["target_share"],
                              "target_reached": (examples[extra["name"]] / total >= extra["target_share"]) if extra["target_share"] is not None and total else None,
                              "second_pass": extra["second_pass"]} for extra in extras]  # fmt: skip
        manifest["shares"] = {"examples": {name: value / total for name, value in examples.items()} if total else None}
    if general_block is not None:
        manifest["general"] = general_block
        manifest["split"].update(general_train=len(train_general), general_dev=len(dev_general), general_dev_used=dev_used["general"],
                                 general_dev_unused=len(dev_general) - dev_used["general"])  # fmt: skip
        manifest["steps_per_epoch"]["base_and_probe"] = -(-(examples["base"] + examples["probe"]) // batch)
    if args.token_sample:
        parts = {"base": train_base, **{extra["name"]: extra["train"] + extra["copy_train"] for extra in extras}, "probe": train_probe,
                 **({"general": train_general} if general_block is not None else {})}
        manifest["tokens"] = _token_estimate(parts, examples, sample=int(args.token_sample), config=config)
        if "shares" in manifest:
            manifest["shares"]["tokens_est"] = {name: manifest["tokens"][name]["est_token_share"] for name in parts}
    _write_json(args.out.with_name(args.out.stem + ".manifest.json"), manifest)
    shares = f"probe share {epoch['probe_share']:.4f}" + (f", general share {epoch['general_share']:.4f}" if general_block else "")
    _log(f"mix: {len(base)} + {len(probe)} + {len(general_lines)} records → {args.out} · {shares} of examples/epoch")
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
    compare.add_argument("--dimensions", nargs="+", default=None, metavar="DIM",
                         help="묶음 차원 (기본: 전부 — domain language source_type split label_kind label_confidence type); +로 이으면 교차 (split+source_type)")
    compare.set_defaults(func=cmd_compare)
    sample = sub.add_parser("sample", help="공개 세트(judgment-v0)에서 묶음마다 seed로 N개를 뽑아 변환한 봉인 시험 G2 + manifest (CPU)")
    sample.add_argument("--source", action="append", required=True, metavar="NAME=PATH", help="묶음 이름=judgment-v0 records.jsonl (여러 번)")
    sample.add_argument("--per-source", type=int, default=500)
    sample.add_argument("--seed", type=int, required=True, help="뽑기 seed (manifest에 적힌다)")
    sample.add_argument("--name", default="G2")
    sample.add_argument("--expect-types", nargs="*", default=None, metavar="NAME=TYPES",
                        help="묶음마다 기대하는 판단 질문 종류 (예: boolq=noul multinli=choice:3) — 다르면 쓰지 않는다")
    sample.add_argument("--out-dir", type=Path, required=True)
    sample.set_defaults(func=cmd_sample)
    mix = sub.add_parser("mix", help="학습 파일 + G1-train-mix를 이어 붙인 학습 입력과 manifest (CPU)")
    mix.add_argument("--base", type=Path, required=True, help="학습 파일 (예: artifacts/simgen/v01/train-50k.jsonl)")
    mix.add_argument("--probe", type=Path, required=True, help="G1-train-mix.jsonl")
    mix.add_argument("--sealed", type=Path, required=True, help="봉인 G1.jsonl — 섞기용과 origin group·원본 레코드가 겹치지 않는지 본다 (반드시 준다: 빠뜨리면 계보 검사가 조용히 빠졌다 — 리뷰 1 M4)")
    mix.add_argument("--train-config", type=Path, default=None, help="학습 설정 YAML — 주면 그 dev 나눔·q로 epoch당 예 수와 섞인 몫을 정확히 센다")
    mix.add_argument("--general", type=Path, default=None,
                     help="general-reading training file (scripts/convert_general_sets.py, Task v0.2) — a prefix of it goes after the probe file")
    mix.add_argument("--general-share", type=float, default=None,
                     help="target share of train examples per epoch for the general data (e.g. 0.10, 0.25) — the shortest prefix that reaches it")
    mix.add_argument("--extra", action="append", default=[], metavar="NAME=PATH",
                     help="an extra training part placed after the base file (Task v0.3c, e.g. mujoco=artifacts/simgen/v03/mujoco-train.jsonl); repeatable")
    mix.add_argument("--extra-share", action="append", default=[], metavar="NAME=SHARE",
                     help="target share of train examples per epoch for an --extra part (e.g. mujoco=0.20); if its first pass falls short, a "
                          "recorded second pass of whole pair groups, round-robin over its families, is appended (ids <id>~r2)")
    mix.add_argument("--general-take-all-if-short", action="store_true",
                     help="if the whole general file cannot reach --general-share, use all of it (the manifest records the share reached)")
    mix.add_argument("--general-repeat-if-short", action="store_true",
                     help="if the whole general file cannot reach --general-share, append a second pass over the shortest prefix that does "
                          "(ids <id>~r2; the manifest records it)")
    mix.add_argument("--token-sample", type=int, default=0,
                     help="estimate prompt tokens per example from this many train records per part (0 = off; needs the tokenizer)")
    mix.add_argument("--out", type=Path, required=True)
    mix.set_defaults(func=cmd_mix)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
