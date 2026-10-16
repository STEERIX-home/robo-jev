#!/usr/bin/env python
"""Task v0.9a — arm CN's training input ``train-v09n.jsonl`` and the v0.9 exposure of both arms (PRE-DECLARED v0.9 VERDICT RULE).

    # the whole assembly, one command (CPU unit, ≤ 48 GB): build → check → arm scan → sealed checks → exposure, tokens, time and cost
    python scripts/assemble_v09.py all --out-dir artifacts/simgen/v09
    # the steps one by one
    python scripts/assemble_v09.py build --out artifacts/simgen/v09/train-v09n.jsonl
    python scripts/assemble_v09.py check --input artifacts/simgen/v09/train-v09n.jsonl
    python scripts/assemble_v09.py armscan --input artifacts/simgen/v09/train-v09n.jsonl --out artifacts/simgen/v09/train-v09n.armscan.json
    python scripts/assemble_v09.py overlap --input artifacts/simgen/v09/train-v09n.jsonl --out-dir artifacts/simgen/v09
    python scripts/assemble_v09.py exposure --input artifacts/simgen/v09/train-v09n.jsonl --out artifacts/simgen/v09/train-v09.exposure.json

**Arm C** trains v0.8's input ``train-v08.jsonl`` (sha256 bffd7494…) unchanged, with the order-consistency loss (the trainer's
``consistency`` option). **Arm CN** trains ``train-v09n.jsonl``: ``train-v08.jsonl`` with general-mix records **replaced one for one** by
measurement-literacy records (:mod:`robo_jev.simgen.numeric_v09`), with the same option. The total count stays the same and the order is kept:
each numeric record takes the slot of the record it replaces (in file order). The replaced records are general records the trainer trains on
(not held out as dev, so the dev set stays v0.8's) whose question v0.8 did not expand to many options (so the many-option exposure stays),
an equal number per source (:data:`SHARE` of all records), ranked by a salted hash of their id. A numeric record's id is never a dev id of
the trainer's split, so the dev split of CN is v0.8's. Every other line is train-v08's, byte for byte (``check`` proves it).

``armscan`` scans every numeric record's word shingles (8, 6 and 5 words) against the arm requests A16, A16's y copy and the arm sweep A16s
and reads **only counts** (with two sensitivity controls). ``overlap`` runs the sealed checks on the whole input: ``system1_v03.py overlap``
(T1–T7, G1, G2, T8a, T8b, T9, B1, the H-set, A4, the iface M1 / M2 sets, M2g, M1g, M2g-F, and B8, M8g, R14, A16, A16y, S16, A16s as sets written
outside our generators), v0.7b's / v0.7f's gate scans, v0.8b's gate re-check (M8g, B8, R14, A16, S16, A16s) and the numeric records' text
units and 8-word shingles against every sealed set — hashes and counts only. ``exposure`` measures, per arm and seed, the trainer's view, the
tokens with the consistency copies packed as the trainer packs them, and the expected time and cost scaled from v0.8 F's **measured** pods.
"""

from __future__ import annotations

import argparse
import datetime as dt
import functools
import hashlib
import importlib.util
import json
import math
import random
import statistics
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

SCRIPT_VERSION = "v09a-assemble-1"
CODE_PATHS = ("scripts/assemble_v09.py", "src/robo_jev/simgen/numeric_v09.py", "src/robo_jev/decisions/examples.py", "src/robo_jev/decisions/train.py",
              "src/robo_jev/decisions/records.py", "src/robo_jev/decisions/prompt.py", "scripts/assemble_v08.py", "scripts/system1_v03.py",
              "scripts/assemble_v04.py")
ARTIFACTS = REPO / "artifacts"
TRAIN_V08 = ARTIFACTS / "simgen" / "v08" / "train-v08.jsonl"
TRAIN_V08_SHA = "bffd7494149596fd044616b8df85d76b948a7fce0ee7b706325cf28c2e45a673"
OUT_DIR = ARTIFACTS / "simgen" / "v09"
TRAIN_CONFIG = REPO / "configs" / "train" / "decisions-v0-lora-h100.yaml"
#: The numeric records' share of ALL records (the brief: 6–8 %).
SHARE = 0.07
REPLACE_SALT = "v09-replace-1"
#: The order-consistency option of the four v0.9 jobs (PRE-DECLARED v0.9 VERDICT RULE) and the floor the share may be lowered to.
CONSISTENCY = {"share": 0.3, "lambda": 0.5, "copies": ["shift", "reverse"], "divergence": "js"}
SHARE_FLOOR = 0.15
#: v0.8 F's measured pods (artifacts/cloud/runs/v08f-s{0,1}-01*.json, the fetched runs' metrics.json and job-report.json, and the ledger's
#: spent estimate): the training steps' seconds and padded tokens (dev passes excluded), the dev seconds, the whole train step, the pod's
#: lifetime and the diagnoses.
V08F_MEASURED = {
    "0": {"step_seconds": 11641.518, "tokens": 126397541, "padded_tokens": 189475860, "dev_seconds": 645.115, "train_step_seconds": 12429.831,
          "pod_seconds": 26.784636 / 3.99 * 3600, "diagnoses_seconds": 10672.289, "input_bytes": 910853933},
    "1": {"step_seconds": 11649.965, "tokens": 126419845, "padded_tokens": 189575957, "dev_seconds": 653.588, "train_step_seconds": 12452.731,
          "pod_seconds": 26.084919 / 3.99 * 3600, "diagnoses_seconds": 10348.120, "input_bytes": 910853933},
}
#: v0.8's measured upload rate (≈ 950 MB in ≈ 26 min ≈ 5 Mbit/s): seconds per byte.
UPLOAD_SECONDS_PER_BYTE = 1547.0 / 949798591
HOURLY_USD = 3.99
RUN_LIMIT_USD = 48.0
#: Places never written to (the brief): earlier inputs and reports, the sealed sets, v0.9b's folder.
NEVER_WRITE = tuple(ARTIFACTS / p for p in ("simgen/v02", "simgen/v03", "simgen/v04", "simgen/v05", "simgen/v06", "simgen/v07", "simgen/v08",
                                            "simgen/v0b/tests", "simgen/probes", "t8", "t9", "b1", "v06", "v07", "v08", "v09", "iface", "reports"))
#: The sealed sets every numeric record is checked against (relative to the repository).
SEALED_SETS = {
    **{f"T{i}": f"artifacts/simgen/v0b/tests/T{i}.jsonl" for i in range(1, 8)}, "G1": "artifacts/simgen/probes/G1.jsonl",
    "G2": "artifacts/simgen/probes/G2.jsonl", "T8a": "artifacts/t8/T8a.jsonl", "T8b": "artifacts/t8/T8b.jsonl", "T9": "artifacts/t9/T9.jsonl",
    "B1": "artifacts/b1/b1-infamily.jsonl", "H-set": "artifacts/simgen/v04/scene-holdout.jsonl", "A4": "artifacts/v06/a4/test.jsonl",
    "M1": "artifacts/iface/m1/m1-set.jsonl", "M2-short": "artifacts/iface/m2/M2-short.jsonl", "M2-w2k": "artifacts/iface/m2/M2-w2k.jsonl",
    "M2-w4k": "artifacts/iface/m2/M2-w4k.jsonl", "M2-w8k": "artifacts/iface/m2/M2-w8k.jsonl",
    "M2g": "artifacts/v07/gates/m2g/M2g-all-conditions.jsonl", "M1g": "artifacts/v07/gates/m1g/M1g.jsonl", "M2g-F": "artifacts/v07/gates/m2gf/M2g-facts.jsonl",
    "B8": "artifacts/v08/gates/b8/B8.jsonl", "M8g": "artifacts/v08/gates/m8g/M8g.jsonl", "R14": "artifacts/v08/r14/R14-all.jsonl",
    "A16": "artifacts/v08/r16/A16.jsonl", "A16y": "artifacts/v08/r16/A16-y-ablation.jsonl", "S16": "artifacts/v08/r16/S16.jsonl",
    "A16s": "artifacts/v08/a16s/A16s.jsonl",
}
ARM_SETS = ("A16", "A16y", "A16s")
V08_OUTSIDE = ("B8", "M8g", "R14", "A16", "A16y", "S16", "A16s")
MUJOCO_SETS = ("v03b-mujoco-train", "v06-a1", "v06-a2", "v06-a3")


def _log(message: str) -> None:
    print(f"[v09a {dt.datetime.now().strftime('%H:%M:%S')}] {message}", file=sys.stderr, flush=True)


@functools.lru_cache(maxsize=None)
def _script(name: str):
    spec = importlib.util.spec_from_file_location(f"v09a_{Path(name).stem}", REPO / "scripts" / name)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _git() -> dict[str, Any]:
    def run(*args: str) -> str | None:
        try:
            return subprocess.run(["git", "-C", str(REPO), *args], capture_output=True, text=True, check=True).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return None

    status = run("status", "--porcelain", "--", *CODE_PATHS)
    return {"commit": run("rev-parse", "--short", "HEAD"), "dirty": bool(status) if status is not None else None}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    refuse_write(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(payload, ensure_ascii=False, indent=1, default=str) + "\n", encoding="utf-8")
    temp.replace(path)


def _raw_lines(path: Path) -> Iterator[bytes]:
    with Path(path).open("rb") as handle:
        for raw in handle:
            if raw.strip():
                yield raw if raw.endswith(b"\n") else raw + b"\n"


def _now() -> str:
    return dt.datetime.now().astimezone().isoformat(timespec="seconds")


def refuse_write(path: Path) -> None:
    resolved = Path(path).resolve()
    for place in NEVER_WRITE:
        place = place.resolve()
        if resolved == place or place in resolved.parents:
            raise SystemExit(f"{path}: refusing to write into {place}")


def _config(path: Path) -> dict[str, Any]:
    return _script("assemble_v04.py")._config(path)


def part_of(record: Mapping[str, Any]) -> str:
    return _script("assemble_v04.py")._part(dict(record), set(MUJOCO_SETS))


def _rank(rid: str) -> str:
    return hashlib.sha256(f"{REPLACE_SALT}|{rid}".encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------
# build
# --------------------------------------------------------------------------


def eligible(record: Mapping[str, Any], config: Mapping[str, Any]) -> bool:
    """A general record the trainer trains on (not dev) whose question v0.8 did not expand (module docstring)."""
    from robo_jev.decisions.examples import is_dev

    if part_of(record) != "general":
        return False
    if is_dev(dict(record), fraction=config["dev_fraction"], salt=config["dev_salt"]):
        return False
    return not (((record.get("meta") or {}).get("format") or {}).get("expanded"))


def plan_replacements(records: Sequence[tuple[int, Mapping[str, Any]]], *, total_records: int, config: Mapping[str, Any],
                      share: float = SHARE) -> dict[str, Any]:
    """Which slots get numeric records: round(share × total) slots, an equal number per general source (the remainder to the first sources
    in name order), within a source the eligible records with the lowest salted id hash. `records` = (line index, record) of the input."""
    by_source: dict[str, list[tuple[str, int, str]]] = defaultdict(list)
    for index, record in records:
        if eligible(record, config):
            source = str(record["meta"]["source"]["set"])
            by_source[source].append((_rank(str(record["id"])), index, str(record["id"])))
    count = round(share * total_records)
    sources = sorted(by_source)
    quota = {s: count // len(sources) + (1 if i < count % len(sources) else 0) for i, s in enumerate(sources)}
    chosen: dict[int, dict[str, str]] = {}
    for source in sources:
        rows = sorted(by_source[source])
        if len(rows) < quota[source]:
            raise ValueError(f"{source}: {len(rows)} eligible records < quota {quota[source]}")
        for _, index, rid in rows[: quota[source]]:
            chosen[index] = {"id": rid, "source": source}
    return {"count": count, "quota": quota, "eligible": {s: len(v) for s, v in by_source.items()}, "slots": dict(sorted(chosen.items()))}


def cmd_build(args: argparse.Namespace) -> int:
    from robo_jev.decisions.examples import is_dev
    from robo_jev.decisions.records import validate_record
    from robo_jev.simgen import numeric_v09 as N

    refuse_write(args.out)
    sha = _sha256(args.input)
    if args.input.resolve() == TRAIN_V08.resolve() and sha != TRAIN_V08_SHA:
        raise SystemExit(f"{args.input}: sha256 {sha[:12]}… is not v0.8's input")
    config = _config(args.train_config)
    _log(f"pass 1: {args.input} ({sha[:12]}…)")
    rows, ids, total = [], set(), 0
    for index, raw in enumerate(_raw_lines(args.input)):
        record = json.loads(raw)
        ids.add(str(record["id"]))
        total += 1
        if part_of(record) == "general":
            rows.append((index, {"id": record["id"], "meta": record["meta"]}))
    plan = plan_replacements(rows, total_records=total, config=config, share=args.share)
    del rows
    _log(f"replace {plan['count']} of {total} records ({plan['count'] / total:.4f}); quota {plan['quota']}; eligible {plan['eligible']}")

    def id_ok(rid: str) -> bool:
        return rid not in ids and not is_dev({"id": rid}, fraction=config["dev_fraction"], salt=config["dev_salt"])

    numeric = N.generate(plan["count"], id_ok=id_ok)
    _log("sealed index (ids, states, text units, 8-word windows of every sealed set) — a numeric record that shares any is drawn again")
    index = SealedIndex.build()
    redrawn: Counter = Counter()
    redrawn_sets: Counter = Counter()
    spare = defaultdict(lambda: N.SPARE_START)
    for position, record in enumerate(numeric):
        while True:
            hits = index.hits(record)
            if not hits:
                break
            family = record["meta"]["numeric"]["family"]
            redrawn[family] += 1
            redrawn_sets.update(hits)
            rid = f"{N.ID_PREFIX}{family}-{spare[family]:06d}"
            while not id_ok(rid):
                spare[family] += 1
                rid = f"{N.ID_PREFIX}{family}-{spare[family]:06d}"
            record = N.make_record(family, spare[family], record_id=rid)
            spare[family] += 1
        numeric[position] = record
    _log(f"redrawn for a sealed hit: {dict(redrawn)} (sets {dict(redrawn_sets)})")
    bad = [(r["id"], p) for r in numeric for p in validate_record(r)]
    if bad:
        raise SystemExit(f"numeric records fail validation: {bad[:5]}")
    slots = list(plan["slots"])
    assignment = dict(zip(slots, numeric))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temp = args.out.with_name(args.out.name + ".tmp")
    replaced_ids: list[str] = []
    try:
        with temp.open("wb") as handle:
            for index, raw in enumerate(_raw_lines(args.input)):
                record = assignment.get(index)
                if record is None:
                    handle.write(raw)
                    continue
                old = json.loads(raw)
                if str(old["id"]) != plan["slots"][index]["id"]:
                    raise ValueError(f"line {index}: the input changed between the passes")
                replaced_ids.append(str(old["id"]))
                handle.write((json.dumps(record, ensure_ascii=False) + "\n").encode("utf-8"))
        temp.replace(args.out)
    finally:
        if temp.exists():
            temp.unlink()
    out_sha = _sha256(args.out)
    families = Counter(r["meta"]["numeric"]["family"] for r in numeric)
    templates = Counter(f"{r['meta']['numeric']['family']}/{r['meta']['numeric']['template']}" for r in numeric)
    domains = Counter(f"{r['meta']['numeric']['family']}/{r['meta']['numeric']['domain']}" for r in numeric)
    domain_totals = Counter(r["meta"]["numeric"]["domain"] for r in numeric)
    styles = {key: Counter(str(r["meta"]["numeric"]["style"][key]) for r in numeric) for key in ("plus", "minus", "sign_space", "unit_space", "thousands", "fixed")}
    questions = Counter(len(r["request"]["questions"]) for r in numeric)
    options = Counter(len(q["criteria"]) for r in numeric for q in r["request"]["questions"].values())
    n_questions = sum(questions[k] * k for k in questions)
    codes = sum(len((r["meta"].get("codes") or {})) for r in numeric)
    wide = sum(1 for r in numeric if r["meta"].get("option_range"))
    replaced_sources = Counter(plan["slots"][i]["source"] for i in slots)
    rule = ("PRE-DECLARED v0.9 VERDICT RULE (progress.md, 2026-10-17 ~07:56 KST): CN = C + measurement-literacy records replacing an equal number of "
            "general-mix records (same total; target 6–8 % of records); five families, seven domains, mixed number formats, dense near thresholds, "
            "option words in descriptions; robo-jev-web's arm template not used")
    manifest = {"kind": "v09n-train", "script_version": SCRIPT_VERSION, "git": _git(), "sha256": out_sha, "rule": rule,
                "out": {"path": str(args.out), "sha256": out_sha, "bytes": args.out.stat().st_size, "records": total},
                "input": {"path": str(args.input), "sha256": sha}, "numeric_records": len(numeric), "share_of_records": len(numeric) / total,
                "generator": N.VERSION, "replaced_by_source": dict(sorted(replaced_sources.items())), "quota": plan["quota"],
                "eligible_by_source": plan["eligible"], "families": dict(sorted(families.items())),
                "redrawn_for_sealed_hits": {"by_family": dict(sorted(redrawn.items())), "by_set": dict(sorted(redrawn_sets.items()))}, "created": _now()}
    changes = {**manifest, "kind": "v09n-train-changes", "templates": dict(sorted(templates.items())), "domains_by_family": dict(sorted(domains.items())),
               "domains": dict(sorted(domain_totals.items())), "number_formats": {k: dict(v) for k, v in styles.items()},
               "questions_per_record": dict(sorted(questions.items())), "numeric_questions": n_questions,
               "options_per_question": dict(sorted(options.items())), "random_code_questions": codes, "random_code_share": codes / n_questions,
               "records_over_8_options": wide, "replaced_ids_sha256": hashlib.sha256("\n".join(replaced_ids).encode()).hexdigest(),
               "replacement_rule": "general records the trainer trains on (not dev), not expanded by v0.8, an equal quota per source, lowest "
                                   f"sha256('{REPLACE_SALT}|id') first; each numeric record takes a replaced slot in file order"}
    stem = args.out.stem
    _write_json(args.out.with_name(f"{stem}.manifest.json"), manifest)
    _write_json(args.out.with_name(f"{stem}.changes.json"), changes)
    rnd = random.Random(f"{SCRIPT_VERSION}|audit")
    audit = [{"id": r["id"], "family": r["meta"]["numeric"]["family"], "template": r["meta"]["numeric"]["template"], "state": r["request"]["state"],
              "questions": r["request"]["questions"], "answers": {n: [k for k, p in r["labels"][n]["probabilities"].items() if p == 1.0][0]
                                                                  for n in r["labels"]}} for r in rnd.sample(numeric, min(60, len(numeric)))]
    _write_json(args.out.with_name(f"{stem}.audit.json"), {"kind": "v09n-audit-sample", "what": "60 numeric records drawn at random, as written",
                                                           "records": audit})
    _log(f"build: {len(numeric)} numeric records ({len(numeric) / total:.4f}) → {args.out} {out_sha[:16]}…; families {dict(families)}; "
         f"replaced {dict(replaced_sources)}")
    return 0


# --------------------------------------------------------------------------
# check
# --------------------------------------------------------------------------


def cmd_check(args: argparse.Namespace) -> int:
    """Lockstep with train-v08: every line is train-v08's byte for byte, or a numeric record in an eligible general slot; counts, order, ids,
    validation, one-hot labels; the trainer's loader keeps every record; the dev split is v0.8's."""
    from robo_jev.decisions.examples import is_dev, split_dev
    from robo_jev.decisions.records import validate_record
    from robo_jev.decisions.train import load_training_records
    from robo_jev.simgen import numeric_v09 as N

    config = _config(args.train_config)
    tally: Counter = Counter()
    failures: list[str] = []
    ids: set[str] = set()
    replaced_sources: Counter = Counter()
    for index, (old_raw, new_raw) in enumerate(zip(_raw_lines(args.reference), _raw_lines(args.input), strict=True)):
        new = json.loads(new_raw)
        rid = str(new["id"])
        if rid in ids:
            failures.append(f"duplicate id {rid}")
        ids.add(rid)
        if old_raw == new_raw:
            tally["identical"] += 1
            continue
        old = json.loads(old_raw)
        if not eligible(old, config):
            failures.append(f"line {index}: {old['id']} changed but is not an eligible general record")
        if (new.get("meta") or {}).get("set") != N.SET_NAME or not rid.startswith(N.ID_PREFIX):
            failures.append(f"line {index}: {rid} is not a numeric record")
        if is_dev(new, fraction=config["dev_fraction"], salt=config["dev_salt"]):
            failures.append(f"line {index}: {rid} would be a dev record")
        problems = validate_record(new)
        if problems:
            failures.append(f"{rid}: {problems[:2]}")
        for name, label in new["labels"].items():
            probabilities = list(label["probabilities"].values())
            if sorted(set(probabilities)) != [0.0, 1.0] or probabilities.count(1.0) != 1:
                failures.append(f"{rid}/{name}: not one-hot")
        tally["replaced"] += 1
        replaced_sources[old["meta"]["source"]["set"]] += 1
        if len(failures) > 20:
            break
    v08, _ = load_training_records(args.reference)
    v09, validation = load_training_records(args.input)
    dev08 = [str(r["id"]) for r in split_dev(v08, fraction=config["dev_fraction"], salt=config["dev_salt"])[1]]
    dev09 = [str(r["id"]) for r in split_dev(v09, fraction=config["dev_fraction"], salt=config["dev_salt"])[1]]
    out = {"kind": "v09n-check", "script_version": SCRIPT_VERSION, "git": _git(), "input": {"path": str(args.input), "sha256": _sha256(args.input)},
           "reference": {"path": str(args.reference), "sha256": _sha256(args.reference)}, "lines": dict(tally),
           "replaced_by_source": dict(sorted(replaced_sources.items())), "failures": failures[:20], "loader": validation,
           "dev_identical_to_v08": dev08 == dev09, "dev_records": len(dev09), "records": len(v09),
           "pass": not failures and validation["invalid"] == 0 and dev08 == dev09 and len(v08) == len(v09), "created": _now()}
    _write_json(args.out, out)
    _log(f"check: {dict(tally)} · failures {len(failures)} · loader invalid {validation['invalid']} · dev identical {dev08 == dev09} · pass {out['pass']}")
    return 0 if out["pass"] else 2


# --------------------------------------------------------------------------
# armscan and the numeric records' sealed scan (counts only)
# --------------------------------------------------------------------------


def record_texts(record: Mapping[str, Any]) -> list[str]:
    """Every text a record shows: the state, each instruction, each option key and description."""
    request = record.get("request") or {}
    out = [str(request.get("state") or "")]
    for question in (request.get("questions") or {}).values():
        out.append(str(question.get("instructions") or ""))
        criteria = question.get("criteria")
        if isinstance(criteria, dict):
            out += [str(k) for k in criteria] + [str(v) for v in criteria.values()]
        elif isinstance(criteria, list):
            out += [str(v) for v in criteria]
    return out


def windows(text: str, size: int) -> set[str]:
    """The `size`-word windows of the normalised text — full windows only (a shorter text has none)."""
    from robo_jev.data.general_sets import normalize

    words = normalize(text).split()
    return {" ".join(words[i:i + size]) for i in range(len(words) - size + 1)}


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _read(name: str) -> list[dict[str, Any]]:
    from robo_jev.decisions.records import read_records

    return read_records(REPO / SEALED_SETS[name])


def numeric_records(path: Path) -> Iterator[dict[str, Any]]:
    from robo_jev.simgen import numeric_v09 as N

    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if N.SET_NAME in line:
                record = json.loads(line)
                if (record.get("meta") or {}).get("set") == N.SET_NAME:
                    yield record


def arm_scan(records: Sequence[Mapping[str, Any]], sets: Mapping[str, Sequence[Mapping[str, Any]]], *, sizes: Sequence[int] = (8, 6, 5)) -> dict[str, Any]:
    """Counts of the numeric records' word windows (and whole short texts) that occur in the arm sets — hashes in memory, counts out."""
    from robo_jev.data.general_sets import normalize

    out: dict[str, Any] = {"records": len(records), "sizes": {}}
    for size in sizes:
        theirs: dict[str, set[str]] = {name: {_hash(w) for r in rows for t in record_texts(r) for w in windows(t, size)} for name, rows in sets.items()}
        union = set().union(*theirs.values())
        hit_records = Counter()
        by_template: Counter = Counter()
        shared: set[str] = set()
        mine: set[str] = set()
        for r in records:
            grams = {_hash(w) for t in record_texts(r) for w in windows(t, size)}
            mine |= grams
            common = grams & union
            if common:
                shared |= common
                for name, hashes in theirs.items():
                    if grams & hashes:
                        hit_records[name] += 1
                by_template[f"{r['meta']['numeric']['family']}/{r['meta']['numeric']['template']}"] += 1
        out["sizes"][str(size)] = {"sealed_windows": {n: len(h) for n, h in theirs.items()}, "numeric_windows": len(mine),
                                   "shared_windows": len(shared), "records_with_shared_windows": dict(hit_records),
                                   "records_with_shared_windows_by_template": dict(sorted(by_template.items()))}
    short_theirs = {name: {_hash(normalize(t)) for r in rows for t in record_texts(r) if 0 < len(normalize(t).split()) < 5} for name, rows in sets.items()}
    short_mine = Counter()
    for r in records:
        for t in record_texts(r):
            if 0 < len(normalize(t).split()) < 5:
                h = _hash(normalize(t))
                for name, hashes in short_theirs.items():
                    if h in hashes:
                        short_mine[name] += 1
    out["short_texts"] = {"what": "texts of 1–4 words (option keys, short descriptions) equal to one of the set's short texts after normalising — "
                                  "counted, not printed; generic words such as 'hold' can match", "matches": dict(short_mine),
                          "sealed_short_texts": {n: len(h) for n, h in short_theirs.items()}}
    return out


def cmd_armscan(args: argparse.Namespace) -> int:
    """The A16 / A16y / A16s scan (module docstring) with two sensitivity controls: A16 against A16s (A16s is derived from A16's requests,
    so they must share windows) and a planted record carrying one A16 state (must be found). Counts only."""
    records = list(numeric_records(args.input))
    sets = {name: _read(name) for name in ARM_SETS}
    result = arm_scan(records, sets)
    control_derived = arm_scan([{"request": r["request"], "meta": {"numeric": {"family": "control", "template": "A16"}}} for r in sets["A16"]],
                               {"A16s": sets["A16s"]}, sizes=(8,))
    planted = {"request": {"state": sets["A16"][0]["request"]["state"], "questions": {}}, "meta": {"numeric": {"family": "control", "template": "planted"}}}
    control_planted = arm_scan([planted], {"A16": sets["A16"]}, sizes=(8, 5))
    passing = all(v["shared_windows"] == 0 for k, v in result["sizes"].items() if k in ("8", "6"))
    out = {"kind": "v09n-armscan", "script_version": SCRIPT_VERSION, "git": _git(), "input": {"path": str(args.input), "sha256": _sha256(args.input)},
           "sets": {name: {"path": SEALED_SETS[name], "sha256": _sha256(REPO / SEALED_SETS[name]), "records": len(rows)} for name, rows in sets.items()},
           "what": "the numeric records' normalised word windows (8, 6 and 5 words; every text: state, instructions, option keys and descriptions) "
                   "against the arm requests A16, A16's y copy and the arm sweep A16s — hashed in memory, only counts written; nothing of the arm "
                   "sets is printed. Rule (fixed before the scan): 0 shared 8- and 6-word windows; 5-word and short-text matches are reported",
           "scan": result, "controls": {"A16_vs_A16s_8_word_shared": control_derived["sizes"]["8"]["shared_windows"],
                                         "planted_A16_state_found_8": control_planted["sizes"]["8"]["shared_windows"] > 0,
                                         "planted_A16_state_found_5": control_planted["sizes"]["5"]["shared_windows"] > 0},
           "pass": passing and control_derived["sizes"]["8"]["shared_windows"] > 0 and control_planted["sizes"]["8"]["shared_windows"] > 0,
           "created": _now()}
    _write_json(args.out, out)
    _log(f"armscan: {len(records)} numeric records; shared windows " + ", ".join(f"{k}w {v['shared_windows']}" for k, v in result["sizes"].items())
         + f"; short-text matches {result['short_texts']['matches']}; controls {out['controls']}; pass {out['pass']}")
    return 0 if out["pass"] else 2


class SealedIndex:
    """Hashes of every sealed set's (:data:`SEALED_SETS`) ids, whole normalised states, text units (:func:`record_units`) and full 8-word
    windows — kept in memory; :meth:`hits` names the sets a record shares any of them with (nothing of a set is ever written)."""

    def __init__(self) -> None:
        self.ids: dict[str, set[str]] = {}
        self.states: dict[str, set[str]] = {}
        self.units: dict[str, set[str]] = {}
        self.windows: dict[str, set[str]] = {}

    @classmethod
    def build(cls) -> "SealedIndex":
        from robo_jev.data.general_sets import normalize, record_units

        index = cls()
        for name in SEALED_SETS:
            rows = _read(name)
            index.ids[name] = {str(r.get("id")) for r in rows}
            index.states[name] = {_hash(normalize((r.get("request") or {}).get("state") or "")) for r in rows}
            index.units[name] = {_hash(u) for r in rows for u in record_units(r)}
            index.windows[name] = {_hash(w) for r in rows for t in record_texts(r) for w in windows(t, 8)}
        return index

    def hits(self, record: Mapping[str, Any]) -> list[str]:
        from robo_jev.data.general_sets import normalize, record_units

        rid = str(record.get("id"))
        state = _hash(normalize(record["request"]["state"]))
        units = {_hash(u) for u in record_units(record)}
        grams = {_hash(w) for t in record_texts(record) for w in windows(t, 8)}
        return [name for name in SEALED_SETS if rid in self.ids[name] or state in self.states[name] or units & self.units[name]
                or grams & self.windows[name]]


def sealed_scan(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """The numeric records against every sealed set (:data:`SEALED_SETS`): ids, whole states, text units (:func:`record_units`) and full
    8-word windows — hashes in memory, counts out."""
    from robo_jev.data.general_sets import normalize, record_units

    mine_units = [{_hash(u) for u in record_units(r)} for r in records]
    mine_grams = [{_hash(w) for t in record_texts(r) for w in windows(t, 8)} for r in records]
    mine_states = [_hash(normalize(r["request"]["state"])) for r in records]
    mine_ids = {str(r["id"]) for r in records}
    out = {}
    for name in SEALED_SETS:
        rows = _read(name)
        ids = {str(r.get("id")) for r in rows}
        states = {_hash(normalize((r.get("request") or {}).get("state") or "")) for r in rows}
        units = {_hash(u) for r in rows for u in record_units(r)}
        grams = {_hash(w) for r in rows for t in record_texts(r) for w in windows(t, 8)}
        out[name] = {"records": len(rows), "ids": len(mine_ids & ids), "states": sum(1 for s in mine_states if s in states),
                     "unit_records": sum(1 for u in mine_units if u & units), "window_records": sum(1 for g in mine_grams if g & grams)}
    return out


# --------------------------------------------------------------------------
# overlap — the sealed checks of the whole input
# --------------------------------------------------------------------------


def v08_gates_recheck(train: Path) -> dict[str, Any]:
    """v0.8b's re-check of its gates (``scripts/v08_gates_build.py recheck-v08``) on `train`: M8g's MMLU items (units, 8-word shingles) and
    physics items (whole states, state / correct-option shingles), B8 (whole states; state-line shingles reported), R14 (whole states), A16 /
    S16 / A16s (text units; question / option shingles reported). Counts only."""
    from robo_jev.bench import gates_v08 as G
    from robo_jev.bench.iface import shingles
    from robo_jev.data.general_sets import normalize, record_units
    from robo_jev.decisions.records import read_records

    gb, v7b = _script("v08_gates_build.py"), _script("v07_gates_build.py")
    items = read_records(G.M8G_DIR / "m8g-items.jsonl")
    general = [{"item": i["item"], "state": i["state"], "stem": i["stem"], "correct": i["correct"], "own": list(i["own"]) + [str(d) for d in i["drawn"]]}
               for i in items if i["source"] == G.GENERAL_SOURCE]
    physics = [{"item": i["item"], "state": i["state"], "correct": tuple(i["correct"])} for i in items if i["source"] != G.GENERAL_SOURCE]
    units, grams = gb.merged(v7b.candidates(general, {}), gb.physics_candidates(physics))
    for r in _read("B8"):
        units.setdefault(v7b._hash(normalize(r["request"]["state"])), set()).add(f"b8:{r['id']}")
        for line in r["request"]["state"].splitlines():
            if len(line.split()) >= 8:
                for gram in shingles(line):
                    grams.setdefault(gram, set()).add(f"b8shingle:{r['id']}")
    for r in _read("R14"):
        units.setdefault(v7b._hash(normalize(r["request"]["state"])), set()).add(f"r14:{r['id']}")
    for kind, name in (("a16", "A16"), ("s16", "S16"), ("a16s", "A16s"), ("a16y", "A16y")):
        for r in _read(name):
            for unit in record_units(r):
                units.setdefault(v7b._hash(unit), set()).add(f"{kind}:{r['id']}")
            for question in r["request"]["questions"].values():
                for text in [str(question.get("instructions") or ""), *[str(v) for v in (question.get("criteria") or {}).values()]]:
                    if len(text.split()) >= 8:
                        for gram in shingles(text):
                            grams.setdefault(gram, set()).add(f"{kind}shingle:{r['id']}")
    found = v7b.scan([train], units, grams, set())
    hits = found["hits"]
    groups = Counter(k.split(":")[0] for k in hits)
    return {"hits_by_kind": dict(sorted(groups.items())), "total_hits": sum(groups.values()), "files": found["files"],
            "rule": "v0.8b's recheck-v08 on this input: an M8g MMLU item with any unit or shingle hit, an M8g physics item with a whole-state or "
                    "state / correct-option shingle hit, a B8 or R14 request whose whole state is in the input, an A16 / S16 / A16s request with a "
                    "text unit in the input — each would be left out of its reading; shingle hits of B8 state lines and A16 / S16 / A16s question "
                    "texts reported"}


def text_subset(new_path: Path, old_path: Path) -> dict[str, Any]:
    """Every distinct text of the non-numeric records of the new input is a text of the old one (hashes only)."""
    a08 = _script("assemble_v08.py")
    old: set[str] = set()
    for raw in _raw_lines(old_path):
        old.update(_hash(t) for t in a08._texts(json.loads(raw)))
    missing = 0
    for raw in _raw_lines(new_path):
        record = json.loads(raw)
        if part_of(record) == "numeric":
            continue
        missing += sum(1 for t in a08._texts(record) if _hash(t) not in old)
    return {"texts_of_kept_records_not_in_reference": missing, "subset": missing == 0}


def cmd_overlap(args: argparse.Namespace) -> int:
    a08 = _script("assemble_v08.py")
    out_dir = args.out_dir
    refuse_write(out_dir)
    missing = [name for name, rel in SEALED_SETS.items() if not (REPO / rel).is_file()]
    if missing:
        raise SystemExit(f"sealed files missing: {missing} — every sealed set is checked")
    stem = args.input.stem
    extra = {"B1": SEALED_SETS["B1"], "H-set": SEALED_SETS["H-set"], **{n: SEALED_SETS[n] for n in ("M1", "M2-short", "M2-w2k", "M2-w4k", "M2-w8k",
                                                                                                       "M2g", "M1g", "M2g-F")}}
    outside = {"A4": SEALED_SETS["A4"], **{n: SEALED_SETS[n] for n in V08_OUTSIDE}}
    argv = [sys.executable, "scripts/system1_v03.py", "overlap", "--input", str(args.input), "--out", str(out_dir / f"{stem}.overlap.json")]
    argv += sum((["--sealed-extra", f"{n}={REPO / p}"] for n, p in extra.items()), [])
    argv += sum((["--sealed-outside", f"{n}={REPO / p}"] for n, p in outside.items()), [])
    argv += sum((["--mujoco-set", s] for s in MUJOCO_SETS), [])
    _log("system1_v03 overlap (the whole input)")
    rc = subprocess.run(argv, cwd=REPO).returncode
    overlap = json.loads((out_dir / f"{stem}.overlap.json").read_text(encoding="utf-8"))
    reference_overlap = None
    if overlap.get("total_hits"):  # the same check on train-v08: hits of the kept records are v0.8's (arm C trains them too)
        _log("system1_v03 overlap on the reference (the same sets)")
        ref_argv = list(argv)
        ref_argv[ref_argv.index("--input") + 1] = str(args.reference)
        ref_argv[ref_argv.index("--out") + 1] = str(out_dir / "train-v08.reference-overlap.json")
        subprocess.run(ref_argv, cwd=REPO)
        reference_overlap = json.loads((out_dir / "train-v08.reference-overlap.json").read_text(encoding="utf-8"))
    _log("v0.7b's M1g / M2g scan and v0.7f's M2g-F scan")
    scans = a08.gate_scans(args.input)
    _log("v0.8b's gate re-check (M8g, B8, R14, A16, S16, A16s, A16y) on this input and on train-v08")
    v08 = v08_gates_recheck(args.input)
    v08_reference = v08_gates_recheck(args.reference)
    _log("the numeric records against every sealed set (ids, states, text units, 8-word windows)")
    records = list(numeric_records(args.input))
    per_set = sealed_scan(records)
    _log("text subset: every text of the kept records is a text of train-v08")
    subset = text_subset(args.input, args.reference)
    numeric_hits = sum(v["ids"] + v["states"] + v["unit_records"] + v["window_records"] for v in per_set.values())
    controls_ok = all(scans["m2g_facts"]["controls_found"].values())
    numeric_part = (overlap.get("parts") or {}).get("numeric") or {}
    system1_same_as_v08 = not overlap.get("total_hits") or (reference_overlap is not None and reference_overlap.get("total_hits") == overlap.get("total_hits")
                                                           and not {k for k in numeric_part if k != "records"})
    exclusions = {k: v for k, v in v08["hits_by_kind"].items() if not k.endswith("shingle")}
    report = {"kind": "v09n-sealed", "script_version": SCRIPT_VERSION, "git": _git(), "input": {"path": str(args.input), "sha256": _sha256(args.input)},
              "system1_overlap": {"rc": rc, "total_hits": overlap.get("total_hits"), "sealed": sorted(overlap.get("sealed") or {}),
                                  "numeric_part": numeric_part, "reference_total_hits": (reference_overlap or {}).get("total_hits"),
                                  "hits_all_in_kept_records_and_equal_to_v08": system1_same_as_v08},
              "gate_scans_v07": scans, "gates_v08_recheck": v08, "gates_v08_recheck_on_train_v08": v08_reference,
              "numeric_records_vs_sealed_sets": per_set, "numeric_hits": numeric_hits, "text_subset": subset,
              "what": "hashes and counts only — no text of a sealed set is read out. Pass: the numeric records hit no sealed set (ids, states, text "
                      "units, 8-word windows); every whole-input hit is one the same check finds in train-v08 (arm C's input); v0.8b's exclusions 0",
              "pass": system1_same_as_v08 and scans["m1g_m2g"]["hits"] == 0 and scans["m2g_facts"]["pieces_with_exact_hits"] == 0
                      and scans["m2g_facts"]["shingles_found"] == 0 and controls_ok and not exclusions and v08["hits_by_kind"] == v08_reference["hits_by_kind"]
                      and numeric_hits == 0 and subset["subset"],
              "created": _now()}
    _write_json(out_dir / f"{stem}.sealed.json", report)
    _log(f"overlap: system1_v03 {overlap.get('total_hits')} (rc {rc}) · v0.7 gates {scans['m1g_m2g']['hits']} / {scans['m2g_facts']['pieces_with_exact_hits']} / "
         f"{scans['m2g_facts']['shingles_found']} · v0.8 gates {v08['total_hits']} {v08['hits_by_kind']} · numeric vs sealed {numeric_hits} · subset "
         f"{subset['subset']} · pass {report['pass']}")
    return 0 if report["pass"] else 2


# --------------------------------------------------------------------------
# exposure — the trainer's view, tokens with the consistency copies, time and cost
# --------------------------------------------------------------------------


def consistency_rows(record: Mapping[str, Any], name: str, consistency: Mapping[str, Any]) -> int:
    """1 + the copies the trainer adds for this plan entry (0 when the question is not a picked choice question)."""
    from robo_jev.decisions.examples import consistency_selected, copy_permutation

    question = record["request"]["questions"][name]
    if consistency is None or question.get("type") != "choice":
        return 1
    if not consistency_selected(dict(record), name, share=consistency["share"], salt=consistency.get("salt", "v09-consistency-1")):
        return 1
    n = len(question["criteria"])
    return 1 + len({tuple(copy_permutation(kind, n)) for kind in consistency["copies"]})


def token_simulation(records: Sequence[Mapping[str, Any]], *, config: Mapping[str, Any], seeds: Sequence[int], tokenizer: Any,
                     shares: Sequence[float | None], exact_sample: int = 300) -> dict[str, Any]:
    """Per seed and consistency share (None = off): the epoch-0 steps exactly as the trainer forms them — every example's prompt length
    (the record's prefix tokens + its question block's, :mod:`scripts.assemble_v08`'s method), its copies (same length — checked on a
    sample of exact copies), each step's micro-batches by the trainer's :func:`robo_jev.decisions.train.pack_units`, the padded tokens."""
    from robo_jev.decisions.codes import assigned_codes
    from robo_jev.decisions.examples import build_examples, consistency_copies, epoch_plan, split_dev
    from robo_jev.decisions.prompt import _prefix as _prompt_prefix
    from robo_jev.decisions.prompt import build_prompts
    from robo_jev.decisions.records import shuffle_key
    from robo_jev.decisions.train import pack_units

    q = int(config["questions_per_request"])
    fmt, version = config["format"], config["prompt_version"]
    batch, micro = int(config["batch_examples"]), int(config["micro_batch_tokens"])
    train, _ = split_dev(list(records), fraction=config["dev_fraction"], salt=config["dev_salt"])
    lengths: dict[tuple[int, str], int] = {}
    prefixes: dict[int, int] = {}

    def length(index: int, name: str) -> int:
        cached = lengths.get((index, name))
        if cached is None:
            record = train[index]
            if index not in prefixes:
                prefixes[index] = len(tokenizer.encode(_prompt_prefix(record["request"]["state"], fmt), add_special_tokens=False).ids)
            sub = {"state": "x", "questions": {name: record["request"]["questions"][name]}}
            codes = assigned_codes(record).get(name)
            (prompt,) = build_prompts(sub, tokenizer, key=shuffle_key(record), seed=1, fmt=fmt, mode="direct", orders="seeded", version=version,
                                      codes={name: codes} if codes else None)
            cached = lengths[(index, name)] = prefixes[index] + len(prompt["tokens"]) - prompt["prefix_tokens"]
        return cached

    out: dict[str, Any] = {}
    for seed in seeds:
        plan = epoch_plan(train, seed=int(seed), epoch=0, q=q, augment=True)
        per_share: dict[str, Any] = {}
        for share in shares:
            consistency = None if share is None else {**CONSISTENCY, "share": float(share)}
            tokens = padded = steps = micro_batches = copy_rows = copy_tokens = picked = numeric_examples = numeric_tokens = 0
            longest_unit = 0
            for start in range(0, len(plan), batch):
                chunk = plan[start:start + batch]
                ls = [length(i, n) for i, n, _ in chunk]
                rows = [consistency_rows(train[i], n, consistency) for i, n, _ in chunk]
                groups = pack_units(ls, rows, micro)
                tokens += sum(ls)
                padded += sum(sum(rows[i] for i in g) * max(ls[i] for i in g) for g in groups)
                micro_batches += len(groups)
                copy_rows += sum(r - 1 for r in rows)
                copy_tokens += sum((r - 1) * l for r, l in zip(rows, ls))
                picked += sum(1 for r in rows if r > 1)
                longest_unit = max(longest_unit, max(r * l for r, l in zip(rows, ls)))
                for (i, n, _), l in zip(chunk, ls):
                    if part_of(train[i]) == "numeric":
                        numeric_examples += 1
                        numeric_tokens += l
                steps += 1
            per_share["off" if share is None else str(share)] = {
                "steps": steps, "examples": len(plan), "tokens": tokens, "padded_tokens": padded, "micro_batches": micro_batches,
                "copy_rows": copy_rows, "copy_tokens": copy_tokens, "picked_examples": picked, "longest_unit_tokens": longest_unit,
                "numeric_examples": numeric_examples, "numeric_tokens": numeric_tokens}
        rnd = random.Random(f"{SCRIPT_VERSION}|exact|{seed}")
        picked_entries = [e for e in plan if consistency_rows(train[e[0]], e[1], CONSISTENCY) > 1]
        sample = rnd.sample(plan, min(exact_sample, len(plan))) + rnd.sample(picked_entries, min(exact_sample, len(picked_entries)))
        mismatches = copy_mismatches = copies_built = 0
        for i, n, order_seed in sample:
            (example,) = build_examples(train[i], [n], tokenizer, order_seed=order_seed, fmt=fmt, version=version)
            mismatches += int(len(example["tokens"]) != length(i, n))
            if consistency_rows(train[i], n, CONSISTENCY) > 1:
                for c in consistency_copies(train[i], example, tokenizer, kinds=CONSISTENCY["copies"]):
                    copies_built += 1
                    copy_mismatches += int(len(c["tokens"]) != len(example["tokens"]))
        out[str(seed)] = {"by_share": per_share, "exact_check": {"sample": len(sample), "length_mismatches": mismatches, "copies_built": copies_built,
                                                                 "copy_length_mismatches": copy_mismatches}}
    return out


def time_and_budget(sim: Mapping[str, Any], *, input_bytes: int) -> dict[str, Any]:
    """Per seed and share: v0.8 F's measured seconds per padded token × this simulation's padded tokens (+ v0.8 F's measured dev and loading
    seconds — the same dev records — and a 10 % margin) → the train step; the pod = v0.8 F's measured pod − its train step + this train step,
    ± the upload of this input's extra bytes → hours and $ at :data:`HOURLY_USD`."""
    out: dict[str, Any] = {}
    for seed, m in V08F_MEASURED.items():
        if seed not in sim:
            continue
        rate = m["step_seconds"] / m["padded_tokens"]
        other = m["train_step_seconds"] - m["step_seconds"] - m["dev_seconds"]
        rows = {}
        for share, s in sim[seed]["by_share"].items():
            steps_seconds = rate * s["padded_tokens"]
            train_seconds = steps_seconds + m["dev_seconds"] + other
            with_margin = train_seconds * 1.10
            upload_delta = (int(input_bytes) - m["input_bytes"]) * UPLOAD_SECONDS_PER_BYTE
            pod_seconds = m["pod_seconds"] - m["train_step_seconds"] + with_margin + upload_delta
            rows[share] = {"padded_tokens_ratio_vs_v08f_measured": s["padded_tokens"] / m["padded_tokens"], "step_minutes": steps_seconds / 60,
                           "train_minutes": train_seconds / 60, "train_minutes_with_margin": with_margin / 60, "pod_hours": pod_seconds / 3600,
                           "pod_usd": pod_seconds / 3600 * HOURLY_USD, "within_48_usd": pod_seconds / 3600 * HOURLY_USD <= RUN_LIMIT_USD}
        out[seed] = {"seconds_per_padded_token_v08f": rate, "v08f_pod_hours": m["pod_seconds"] / 3600, "v08f_pod_usd": m["pod_seconds"] / 3600 * HOURLY_USD,
                     "by_share": rows}
    return out


def cmd_exposure(args: argparse.Namespace) -> int:
    from robo_jev.decisions.examples import record_pick, split_dev
    from robo_jev.decisions.train import load_training_records
    from robo_jev.model.tokenizer import available_tokenizer, load_tokenizer

    refuse_write(args.out)
    config = _config(args.train_config)
    tokenizer = load_tokenizer(available_tokenizer()[0])
    shares = [None, CONSISTENCY["share"], SHARE_FLOOR]
    arms = {"C": args.reference, "CN": args.input}
    result: dict[str, Any] = {}
    for arm, path in arms.items():
        _log(f"exposure {arm}: {path}")
        records, validation = load_training_records(path)
        if validation["invalid"]:
            raise SystemExit(f"{path}: {validation['invalid']} records fail the trainer's validation")
        train, dev = split_dev(records, fraction=config["dev_fraction"], salt=config["dev_salt"])
        view = {}
        for seed in args.seeds:
            by_part: Counter = Counter()
            for r in train:
                by_part[part_of(r)] += len(record_pick(r, q=int(config["questions_per_request"]), seed=int(seed), epoch=0))
            view[str(seed)] = {"examples_by_part": dict(sorted(by_part.items())), "examples": sum(by_part.values()),
                               "steps": math.ceil(sum(by_part.values()) / int(config["batch_examples"]))}
        sim = token_simulation(records, config=config, seeds=args.seeds, tokenizer=tokenizer, shares=shares)
        result[arm] = {"input": {"path": str(path), "sha256": _sha256(path), "bytes": path.stat().st_size, "records": len(records)},
                       "train_records": len(train), "dev_records": len(dev), "view": view, "tokens": sim,
                       "time_and_budget": time_and_budget(sim, input_bytes=path.stat().st_size)}
        del records, train, dev
    v08f_ok = {seed: result["C"]["tokens"][seed]["by_share"]["off"]["padded_tokens"] == V08F_MEASURED[seed]["padded_tokens"] for seed in map(str, args.seeds)}
    worst = max(row["pod_usd"] for arm in result.values() for seed in arm["time_and_budget"].values() for share, row in seed["by_share"].items()
                if share == str(CONSISTENCY["share"]))
    out = {"kind": "v09-exposure", "script_version": SCRIPT_VERSION, "git": _git(), "train_config": str(args.train_config), "consistency": CONSISTENCY,
           "arms": result, "c_off_equals_v08f_measured_padded_tokens": v08f_ok, "worst_expected_usd_at_share_0.3": worst,
           "share_decision": ("keep 0.3: every run's expected cost is within $48" if worst <= RUN_LIMIT_USD else
                              "lower the share (not below 0.15) before training"),
           "pass": all(v08f_ok.values()) and not any(s["exact_check"]["length_mismatches"] or s["exact_check"]["copy_length_mismatches"]
                                                     for arm in result.values() for s in arm["tokens"].values()),
           "created": _now()}
    _write_json(args.out, out)
    _log(f"exposure: C off = v0.8 F measured {v08f_ok}; worst expected ${worst:.2f} at share 0.3; pass {out['pass']}")
    return 0 if out["pass"] else 2


# --------------------------------------------------------------------------
# all
# --------------------------------------------------------------------------


def _run(argv: list[str]) -> int:
    _log("$ " + " ".join(argv))
    return subprocess.run([sys.executable, *argv], cwd=REPO).returncode


def cmd_all(args: argparse.Namespace) -> int:
    out = args.out_dir
    refuse_write(out)
    out.mkdir(parents=True, exist_ok=True)
    train = out / "train-v09n.jsonl"
    me = str(Path(__file__).resolve().relative_to(REPO))
    if _run([me, "build", "--input", str(args.input), "--out", str(train), "--train-config", str(args.train_config)]) != 0:
        raise SystemExit("build failed")
    rc = {"check": _run([me, "check", "--input", str(train), "--reference", str(args.input), "--out", str(out / "train-v09n.check.json")]),
          "armscan": _run([me, "armscan", "--input", str(train), "--out", str(out / "train-v09n.armscan.json")]),
          "overlap": _run([me, "overlap", "--input", str(train), "--reference", str(args.input), "--out-dir", str(out)]),
          "exposure": _run([me, "exposure", "--input", str(train), "--reference", str(args.input), "--out", str(out / "train-v09.exposure.json")])}
    load = lambda name: json.loads((out / name).read_text(encoding="utf-8")) if (out / name).is_file() else {}  # noqa: E731
    manifest, check, arm, sealed, exposure = (load(n) for n in ("train-v09n.manifest.json", "train-v09n.check.json", "train-v09n.armscan.json",
                                                                "train-v09n.sealed.json", "train-v09.exposure.json"))
    summary = {"kind": "v09-assembly", "script_version": SCRIPT_VERSION, "git": _git(), "returncodes": rc,
               "files": {"train-v09n.jsonl": {"path": str(train), "sha256": _sha256(train)}, "train-v08.jsonl": {"path": str(args.input), "sha256": _sha256(args.input)}},
               "numeric_records": manifest.get("numeric_records"), "share_of_records": manifest.get("share_of_records"),
               "replaced_by_source": manifest.get("replaced_by_source"), "check": check.get("pass"), "armscan": arm.get("pass"), "sealed": sealed.get("pass"),
               "exposure": exposure.get("pass"), "worst_expected_usd_at_share_0.3": exposure.get("worst_expected_usd_at_share_0.3"),
               "pass": all(v == 0 for v in rc.values()), "created": _now()}
    _write_json(out / "train-v09.assembly.json", summary)
    _log(f"done: {train} sha256 {summary['files']['train-v09n.jsonl']['sha256'][:16]}… · pass {summary['pass']} {rc}")
    return 0 if summary["pass"] else 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    b = sub.add_parser("build", help="train-v08 → train-v09n (+ manifest, change report, audit sample)")
    b.add_argument("--input", type=Path, default=TRAIN_V08)
    b.add_argument("--out", type=Path, required=True)
    b.add_argument("--share", type=float, default=SHARE)
    b.add_argument("--train-config", type=Path, default=TRAIN_CONFIG)
    b.set_defaults(func=cmd_build)
    c = sub.add_parser("check", help="lockstep proof against train-v08, validation, labels, the trainer's loader and dev split")
    c.add_argument("--input", type=Path, required=True)
    c.add_argument("--reference", type=Path, default=TRAIN_V08)
    c.add_argument("--train-config", type=Path, default=TRAIN_CONFIG)
    c.add_argument("--out", type=Path, required=True)
    c.set_defaults(func=cmd_check)
    a = sub.add_parser("armscan", help="the numeric records' word windows against A16, A16y and A16s (counts only)")
    a.add_argument("--input", type=Path, required=True)
    a.add_argument("--out", type=Path, required=True)
    a.set_defaults(func=cmd_armscan)
    o = sub.add_parser("overlap", help="sealed checks of the whole input and of the numeric records (hashes and counts)")
    o.add_argument("--input", type=Path, required=True)
    o.add_argument("--reference", type=Path, default=TRAIN_V08)
    o.add_argument("--out-dir", type=Path, required=True)
    o.set_defaults(func=cmd_overlap)
    e = sub.add_parser("exposure", help="the trainer's view, tokens with the consistency copies, time and cost for arms C and CN")
    e.add_argument("--input", type=Path, required=True)
    e.add_argument("--reference", type=Path, default=TRAIN_V08)
    e.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    e.add_argument("--train-config", type=Path, default=TRAIN_CONFIG)
    e.add_argument("--out", type=Path, required=True)
    e.set_defaults(func=cmd_exposure)
    every = sub.add_parser("all", help="build → check → armscan → overlap → exposure")
    every.add_argument("--input", type=Path, default=TRAIN_V08)
    every.add_argument("--train-config", type=Path, default=TRAIN_CONFIG)
    every.add_argument("--out-dir", type=Path, default=OUT_DIR)
    every.set_defaults(func=cmd_all)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
