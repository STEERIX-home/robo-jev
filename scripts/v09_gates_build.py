#!/usr/bin/env python
"""Task v0.9b — build the sealed numeric-rule gate N9g, re-check the gates against CN's training input, and the Spark chain's gate (CPU).

    artifacts/scratch/v09b/cpu.sh n9g scripts/v09_gates_build.py build-n9g          # → artifacts/v09/n9g/ (N9g.jsonl, manifest, overlap, audit)
    artifacts/scratch/v09b/cpu.sh freeze scripts/v09_gates_build.py freeze-n9g      # after the ambiguity audit: frozen-before-scoring.json, 0444
    artifacts/scratch/v09b/cpu.sh recheck scripts/v09_gates_build.py recheck-v09    # once artifacts/simgen/v09/train-v09n.jsonl exists
    scripts/v09_gates_build.py check --c-runs <C s0> <C s1> --cn-runs <CN s0> <CN s1>    # the Spark chain's gate (exit 2 on any problem)

Design: :mod:`robo_jev.bench.n9g_v09`. Built by task v0.9b, which never opens task v0.9a's worktree, generator or ``artifacts/simgen/v09/``:
``train-v09n.jsonl`` is read only by ``recheck-v09`` and ``check``, in code, and only counts and hashes are written.

**build-n9g** — the records (:func:`robo_jev.bench.n9g_v09.n9g_records`), the independent self-check (``scripts/v09_n9g_selfcheck.py``,
must pass), the overlap scan against every training input that exists before v0.9 (:data:`TRAINING_INPUTS`: v0.8b's list and ``train-v08.jsonl``,
C's input) — PRE-DECLARED here: an N9g record with a text unit (whole state, a state line or an option text of ≥ 5 words, the question text of
≥ 5 words) in a training input is dropped before sealing; 8-word shingle hits are reported per template — prompt tokens through the runner's
builder (both readings), the ambiguity-audit sample sheet, ``N9g.jsonl`` and ``n9g.manifest.json``. Not frozen yet.

**freeze-n9g** — after the audit: the self-check again, then ``frozen-before-scoring.json`` (the sha256 of ``N9g.jsonl``); files 0444. The
scorer (``scripts/v08_bench_score.py --set N9g``), the chain's check and the verdict refuse a changed file.

**recheck-v09** — ``train-v09n.jsonl`` (CN's input) against N9g and the v0.8 gates the v0.9 rule reads; writes ``artifacts/v09/overlap-v09.json``.
PRE-DECLARED here (2026-10-17, before any v0.9 data exists): an N9g record is left out of every reading if any of its text units (above) is
in train-v09n, its 8-word shingle hits are reported (per record and per template; a template with hits in > 20 % of its records is flagged,
not excluded); the v0.8 gates keep recheck-v08's rules (an M8g MMLU item with any unit or shingle hit, an M8g physics item with a whole-state
or state / correct-option shingle hit, a B8 or R14 request whose whole state is in the input, an A16 / S16 / A16s request with any text unit
in it; their shingle hits reported). The verdict leaves out the union of overlap-v08.json's (C and F) and overlap-v09.json's (CN) exclusions.

**check** — the chain's gate (refuses on any problem): N9g unchanged since frozen and its self-check passing; the v0.8 gates unchanged and
the C runs trained on the input recheck-v08 read (``scripts/v08_gates_build.py``'s gate check, which also confirms main's many-option path);
the CN runs trained on the train-v09n that recheck-v09 read (``metrics.json`` → ``manifest.data.file_sha256``); N9g's build-time scan covered
C's input; the F runs are v0.8 F's; and the reused v0.8 F / base files stay comparable — the runner file and the package's prompt digest are
the ones those files were scored with.
"""

from __future__ import annotations

import argparse
import datetime as dt
import functools
import importlib.util
import json
import os
import stat
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

SCRIPT_VERSION = "v09-gates-build-1"
ARTIFACTS = REPO / "artifacts"
#: Every training input before v0.9 (task v0.8b's list) and C's own input, train-v08.jsonl.
TRAINING_INPUTS = (
    "simgen/v0b/train-50k.jsonl", "simgen/v01/train-50k.jsonl", "simgen/v01-mix/train-50k-mix.jsonl", "simgen/probes/G1-train-mix.jsonl",
    "simgen/v02/train-50k.jsonl", "simgen/v02/general-train.jsonl", "simgen/v02/train-v02-a.jsonl", "simgen/v02/train-v02-b.jsonl",
    "simgen/v03/train-v03.jsonl", "simgen/v04/train-v04.jsonl", "simgen/v05/train-v05.jsonl", "simgen/v06/train-v06.jsonl",
    "simgen/v07/train-v07.jsonl", "simgen/v08/train-v08.jsonl",
)
TRAIN_V08 = "simgen/v08/train-v08.jsonl"
#: CN's training input (task v0.9a's) — read only by recheck-v09 and check, in code (counts and hashes).
TRAIN_V09N = "simgen/v09/train-v09n.jsonl"
#: The v0.8 F runs (the v0.9 rule's reference arm, already scored).
F_RUNS = ("artifacts/cloud/runs/v08f-s0-01-fetched/runs/v08f-s0-2b", "artifacts/cloud/runs/v08f-s1-01-fetched/runs/v08f-s1-2b")
V08_SPARK = ARTIFACTS / "reports" / "v08" / "spark"
#: The reused v0.8 bench score files whose runner must be today's (F's and the base's A16s).
REUSED_BENCH = tuple(f"{s}-{m}-2b.json" for s in ("B8", "A16", "A16y", "A16s", "R14", "S16", "Jev") for m in ("v08Fs0", "v08Fs1")) + ("A16s-base-2b.json",)
REUSED_M8G = ("M8g-v08Fs0-2b.json", "M8g-v08Fs1-2b.json", "M8g-seed2-v08Fs0-2b.json", "M8g-seed2-v08Fs1-2b.json")
#: GPU seconds per prompt token on the Spark at batch 1 (v0.8's chain measured 0.16-0.18 ms LoRA, 0.09 ms base; planning rates as v0.8b).
SECONDS_PER_TOKEN = {"lora": 0.20e-3, "base": 0.12e-3}
NEVER_WRITE = (ARTIFACTS / "simgen", ARTIFACTS / "t8", ARTIFACTS / "t9", ARTIFACTS / "b1", ARTIFACTS / "v06", ARTIFACTS / "v07", ARTIFACTS / "v08",
               ARTIFACTS / "iface", ARTIFACTS / "reports")
SHINGLE_TEMPLATE_FLAG = 0.20
AUDIT_PER_FAMILY = 12


def _log(message: str) -> None:
    print(f"[v09-gates {dt.datetime.now().strftime('%H:%M:%S')}] {message}", file=sys.stderr, flush=True)


@functools.lru_cache(maxsize=None)
def _script(name: str):
    spec = importlib.util.spec_from_file_location(f"v09b_{Path(name).stem}", REPO / "scripts" / name)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def SELFCHECK():
    """scripts/v09_n9g_selfcheck.py — the independent relabel."""
    return _script("v09_n9g_selfcheck.py")


def V8B():
    """scripts/v08_gates_build.py — the v0.8 gates' candidates and the v0.8 chain's gate check."""
    return _script("v08_gates_build.py")


def _git() -> dict[str, Any]:
    try:
        sha = subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "-C", str(REPO), "status", "--porcelain", "--untracked-files=no"], capture_output=True, text=True, check=True).stdout.strip())
        return {"commit": sha, "dirty": dirty}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}


def refuse_write(path: Path) -> Path:
    resolved = Path(path).resolve()
    for place in NEVER_WRITE:
        place = place.resolve()
        if resolved == place or place in resolved.parents:
            raise SystemExit(f"{path}: never written by this task ({place})")
    return Path(path)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> str:
    from robo_jev.bench.n9g_v09 import sha256_file

    refuse_write(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.chmod(stat.S_IRUSR | stat.S_IWUSR)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return sha256_file(path)


def _write_json(path: Path, payload: Any) -> None:
    refuse_write(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.chmod(stat.S_IRUSR | stat.S_IWUSR)
    path.write_text(json.dumps(payload, indent=1, ensure_ascii=False, default=str) + "\n", encoding="utf-8")


def _seal(paths: list[Path]) -> None:
    for path in paths:
        path.chmod(stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)


def _rel(path: Path) -> str:
    """A path as written into manifests: relative to the checkout (``artifacts/…``) — also from a worktree whose artifacts/ is a symlink."""
    path = Path(path)
    for base in (REPO, REPO.resolve(), ARTIFACTS.resolve().parent):
        for candidate in (path, path.resolve()):
            try:
                return str(candidate.relative_to(base))
            except ValueError:
                continue
    return str(path)


def _now() -> str:
    return dt.datetime.now().astimezone().isoformat(timespec="seconds")


def hash_and_forget(path: Path) -> str:
    """sha256 of a (large) file, then its clean page cache dropped (GB10: page cache lowers MemFree, which the GPU guard reads)."""
    from robo_jev.bench.n9g_v09 import sha256_file

    digest = sha256_file(path)
    try:
        fd = os.open(path, os.O_RDONLY)
        try:
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        finally:
            os.close(fd)
    except OSError:
        pass
    return digest


# --------------------------------------------------------------------------
# Overlap: N9g's units (incl. the question text) and shingles; a streaming scan
# --------------------------------------------------------------------------


def _uhash(unit: str) -> str:
    import hashlib

    return hashlib.sha256(unit.encode("utf-8")).hexdigest()


def n9g_units(record: dict[str, Any]) -> set[str]:
    """The pre-declared text units of an N9g record: :func:`robo_jev.data.general_sets.record_units` (whole state, state lines and option
    texts of ≥ 5 words) and the question text (≥ 5 words)."""
    from robo_jev.data.general_sets import normalize, record_units

    units = set(record_units(record))
    for question in record["request"]["questions"].values():
        text = normalize(question.get("instructions") or "")
        if len(text.split()) >= 5:
            units.add(text)
    return units


def n9g_texts(record: dict[str, Any]) -> list[str]:
    out = [record["request"]["state"]]
    for question in record["request"]["questions"].values():
        out.append(str(question.get("instructions") or ""))
        out += [str(v) for v in (question.get("criteria") or {}).values()]
    return out


def n9g_candidates(records: list[dict[str, Any]]) -> tuple[dict[str, set[str]], dict[str, set[str]]]:
    """(unit hash → N9g ids) and (8-word shingle → N9g ids) over every line of every text with ≥ 8 words."""
    from robo_jev.bench.iface import shingles

    units: dict[str, set[str]] = defaultdict(set)
    grams: dict[str, set[str]] = defaultdict(set)
    for record in records:
        rid = str(record["id"])
        for unit in n9g_units(record):
            units[_uhash(unit)].add(rid)
        for text in n9g_texts(record):
            for line in text.split("\n"):
                if len(line.split()) >= 8:
                    for gram in shingles(line):
                        grams[gram].add(rid)
    return dict(units), dict(grams)


def scan(files: list[Path], *, n9g_units_map: dict[str, set[str]], n9g_grams: dict[str, set[str]], gate_units: dict[str, set[str]] | None = None,
         gate_grams: dict[str, set[str]] | None = None, forget: bool = False) -> dict[str, Any]:
    """Stream every training record: its :func:`record_units` and its question texts against N9g's units, its record units against the v0.8
    gates' (``gate_units``, recheck-v08's rule), and every text's 8-word shingles against both shingle maps. Only hit owners and counts
    are kept (``unit:`` / ``shingle:`` / ``instructions:`` + the file name)."""
    from robo_jev.bench.iface import shingles
    from robo_jev.data.general_sets import normalize, record_units

    gate_units, gate_grams = gate_units or {}, gate_grams or {}
    hits: dict[str, set[str]] = defaultdict(set)
    per_file: dict[str, Any] = {}
    seen: set[int] = set()
    for path in files:
        started, records, unit_hits, gram_hits = time.perf_counter(), 0, 0, 0
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                record = json.loads(line)
                records += 1
                for unit in record_units(record):
                    h = _uhash(unit)
                    for owner in n9g_units_map.get(h, ()):
                        hits[owner].add(f"unit:{path.name}")
                        unit_hits += 1
                    for owner in gate_units.get(h, ()):
                        hits[owner].add(f"unit:{path.name}")
                        unit_hits += 1
                questions = ((record.get("request") or {}).get("questions") or {}).values()
                texts = [str((record.get("request") or {}).get("state") or "")]
                for question in questions:
                    instructions = str(question.get("instructions") or "")
                    norm = normalize(instructions)
                    if len(norm.split()) >= 5:
                        for owner in n9g_units_map.get(_uhash(norm), ()):
                            hits[owner].add(f"instructions:{path.name}")
                            unit_hits += 1
                    texts.append(instructions)
                    criteria = question.get("criteria")
                    if isinstance(criteria, dict):
                        texts += [str(v) for v in criteria.values()]
                    elif isinstance(criteria, list):
                        texts += [str(v) for v in criteria]
                for text in texts:
                    h = hash(text)
                    if h in seen:
                        continue
                    seen.add(h)
                    for gram in shingles(text):
                        for owner in n9g_grams.get(gram, ()):
                            hits[owner].add(f"shingle:{path.name}")
                            gram_hits += 1
                        for owner in gate_grams.get(gram, ()):
                            hits[owner].add(f"shingle:{path.name}")
                            gram_hits += 1
        if forget:
            try:
                fd = os.open(path, os.O_RDONLY)
                try:
                    os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
                finally:
                    os.close(fd)
            except OSError:
                pass
        per_file[_rel(path)] = {"records": records, "unit_hits": unit_hits, "shingle_hits": gram_hits, "seconds": round(time.perf_counter() - started, 1)}
        _log(f"scan {path.name}: {records} records, unit hits {unit_hits}, shingle hits {gram_hits} ({time.perf_counter() - started:.1f} s)")
    return {"hits": {k: sorted(v) for k, v in hits.items()}, "files": per_file}


def template_shingle_share(records: list[dict[str, Any]], hits: dict[str, list[str]]) -> dict[str, Any]:
    """Per template: the share of its records with an 8-word shingle hit (flagged above :data:`SHINGLE_TEMPLATE_FLAG`)."""
    by_template: dict[str, list[str]] = defaultdict(list)
    for r in records:
        by_template[str(r["meta"]["template"])].append(str(r["id"]))
    out = {}
    for template, ids in sorted(by_template.items()):
        hit = sum(1 for i in ids if any(h.startswith("shingle:") for h in hits.get(i, [])))
        if hit:
            out[template] = {"records": len(ids), "with_shingle_hits": hit, "share": round(hit / len(ids), 4), "flagged": hit / len(ids) > SHINGLE_TEMPLATE_FLAG}
    return out


# --------------------------------------------------------------------------
# Tokens and the audit sheet
# --------------------------------------------------------------------------


@functools.lru_cache(maxsize=1)
def _tokenizer():
    from robo_jev.model.tokenizer import available_tokenizer, load_tokenizer

    return load_tokenizer(available_tokenizer()[0])


def prompt_tokens(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Prompt tokens per reading (single order, seeds 1 and 2: the same lengths) through the runner's builder, per family and in total."""
    from robo_jev.bench.n9g_v09 import FAMILIES
    from robo_jev.hf.robo_jev_decide import build_prompts

    tok = _tokenizer()
    per_family: dict[str, list[int]] = defaultdict(list)
    for r in records:
        for p in build_prompts(r["request"]["state"], r["request"]["questions"], tok, seed=1):
            per_family[r["meta"]["family"]].append(len(p["tokens"]))
    out = {f: {"prompts": len(v), "tokens": sum(v), "mean": round(sum(v) / len(v), 1), "max": max(v)} for f, v in per_family.items() if v}
    total = sum(v["tokens"] for v in out.values())
    return {"per_family": {f: out[f] for f in FAMILIES if f in out}, "per_reading": total, "two_readings": 2 * total,
            "gpu_seconds_two_readings": {kind: round(2 * total * rate) for kind, rate in SECONDS_PER_TOKEN.items()}}


def audit_sheet(records: list[dict[str, Any]], out: Path, *, per_family: int, seed: int) -> dict[str, Any]:
    """A stratified sample (per family: templates and formats spread) as a sheet for the ambiguity audit — each item with the right option
    marked and every other option listed: is any other option also right?"""
    from robo_jev.bench import n9g_v09 as N

    picked: list[dict[str, Any]] = []
    for family in N.FAMILIES:
        rows = [r for r in records if r["meta"]["family"] == family]
        r = N.rng("v09-n9g-audit", family, seed)
        by_template: dict[str, list] = defaultdict(list)
        for row in rows:
            by_template[row["meta"]["template"]].append(row)
        templates = sorted(by_template)
        r.shuffle(templates)
        chosen = []
        while len(chosen) < per_family and any(by_template[t] for t in templates):
            for t in templates:
                if by_template[t] and len(chosen) < per_family:
                    chosen.append(by_template[t].pop(r.randrange(len(by_template[t]))))
        picked += chosen
    lines = ["# N9g ambiguity audit sample", "", "For each item: does the stated rule give exactly one right option? Is any other option also right "
             "(or equally defensible)? Is the reading's sign / unit / format clear? Mark AMBIGUOUS, WRONG KEY, or OK.", ""]
    for i, rec in enumerate(picked, 1):
        q = rec["request"]["questions"]["q"]
        m = rec["meta"]
        lines += [f"## {i}. {rec['id']} — {m['template']} (format {m['format']}, near {m['near']}, wording {m['wording']})", "", "```",
                  rec["request"]["state"], "```", "", "Question: " + q["instructions"], ""]
        lines += [f"- {'**RIGHT** ' if k == m['correct'] else ''}`{k}`: {v}" for k, v in q["criteria"].items()]
        lines.append("")
    refuse_write(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"file": _rel(out), "items": [r["id"] for r in picked], "per_family": per_family}


# --------------------------------------------------------------------------
# build-n9g, freeze-n9g
# --------------------------------------------------------------------------


def cmd_build_n9g(args: argparse.Namespace) -> int:
    from robo_jev.bench import n9g_v09 as N

    started = time.perf_counter()
    out = refuse_write(args.out_dir)
    if (out / "frozen-before-scoring.json").is_file() and not args.force_before_scoring:
        raise SystemExit(f"{out}/frozen-before-scoring.json exists — N9g is frozen; rebuild only with --force-before-scoring before any scoring")
    records = N.n9g_records(args.seed)
    check = SELFCHECK().check(records)
    if not check["ok"]:
        raise SystemExit(f"the self-check fails on {check['records_with_problems']} records: {list(check['problems'].items())[:3]}")
    _log(f"{len(records)} records; the independent self-check relabels {check['relabel_matches']} of them identically")
    files = [ARTIFACTS / f for f in TRAINING_INPUTS]
    missing = [_rel(f) for f in files if not f.is_file()]
    if missing and not args.skip_overlap:
        raise SystemExit(f"training inputs missing: {missing}")
    units, grams = n9g_candidates(records)
    found = {"hits": {}, "files": {}, "skipped": True} if args.skip_overlap else scan(files, n9g_units_map=units, n9g_grams=grams, forget=True)
    hits = found["hits"]
    dropped = sorted(k for k, v in hits.items() if any(h.startswith(("unit:", "instructions:")) for h in v))
    kept = [r for r in records if str(r["id"]) not in set(dropped)]
    shingle_records = sorted(k for k, v in hits.items() if any(h.startswith("shingle:") for h in v))
    _log(f"overlap with earlier inputs: {len(dropped)} records dropped (unit hits), {len(shingle_records)} with shingle hits (reported)")
    path = out / "N9g.jsonl"
    sha = _write_jsonl(path, kept)
    tokens = prompt_tokens(kept)
    audit = audit_sheet(kept, out / "audit" / "n9g-audit-sample.md", per_family=AUDIT_PER_FAMILY, seed=args.seed)
    train_shas = {} if args.skip_overlap else {f: hash_and_forget(ARTIFACTS / f) for f in TRAINING_INPUTS}
    manifest = {
        "kind": "v09-n9g", "script_version": SCRIPT_VERSION, "git": _git(), "seed": args.seed, "set": N.N9G_SET, "prefix": N.N9G_PREFIX,
        "file": {"path": _rel(path), "sha256": sha, "records": len(kept), "questions": len(kept)},
        "design": {"families": list(N.FAMILIES), "seen_domains": list(N.SEEN_DOMAINS), "unseen_domains": list(N.UNSEEN_DOMAINS),
                   "formats": list(N.FORMATS), "wordings": list(N.WORDINGS), "question": N.N9G_QUESTION, "option_range": N.OPTION_RANGE,
                   "readings": list(N.READINGS), "scoring": "scripts/v08_bench_score.py --set N9g --serve-adapter --readings single-s1 single-s2 "
                   "(the bundle's runner, serve's conditions: default key, single order; single-s1 is the rule's reading)",
                   "accuracy": "the top option (ties to the earlier key of the record) is meta.correct; cluster = the record",
                   "near": "rel = within 5 % of a bound, abs = within 0.05 of it, never on it (threshold, interval, conversion); selection: rel = a "
                           "candidate breaks a bound by <= 5 %; comparison: close = values within ~4 % of each other, trap = mixed decimals (9.9 / 9.11)",
                   "independence": "written by task v0.9b without opening task v0.9a's worktree, generator or artifacts/simgen/v09/"},
        "counts": N.design_counts(kept), "prompt_tokens": tokens, "self_check": {k: v for k, v in check.items() if k != "problems"},
        "overlap": {"rule": "PRE-DECLARED (task v0.9b): an N9g record with a text unit (whole state, a state line or an option text of >= 5 "
                            "words, the question text of >= 5 words) in a training input is dropped before sealing; 8-word shingle hits are "
                            "reported per template", "training_inputs": train_shas, "files": found.get("files"), "skipped": bool(found.get("skipped")),
                    "dropped": dropped, "shingle_hit_records": len(shingle_records),
                    "template_shingle_share": template_shingle_share(records, hits)},
        "audit": audit, "sealed": "for training and never sent (robo_jev.data.sealed: v09-n9g / v09n9g-)", "created": _now(),
    }
    _write_json(out / "n9g.manifest.json", manifest)
    _write_json(out / "overlap.json", {"kind": "v09-n9g-overlap-build", "script_version": SCRIPT_VERSION, "hits": hits, "files": found.get("files"),
                                       "training_inputs": train_shas, "created": _now()})
    _log(f"done in {time.perf_counter() - started:.0f} s · N9g {len(kept)} records sha256 {sha[:12]} · tokens per reading {tokens['per_reading']:,}")
    return 0


def cmd_freeze_n9g(args: argparse.Namespace) -> int:
    from robo_jev.bench import n9g_v09 as N
    from robo_jev.decisions.records import read_records

    out = refuse_write(args.out_dir)
    frozen_path = out / "frozen-before-scoring.json"
    if frozen_path.is_file() and not args.force_before_scoring:
        raise SystemExit(f"{frozen_path} exists — already frozen")
    path = out / "N9g.jsonl"
    manifest = json.loads((out / "n9g.manifest.json").read_text(encoding="utf-8"))
    sha = N.sha256_file(path)
    if sha != manifest["file"]["sha256"]:
        raise SystemExit("N9g.jsonl is not the file its manifest records — rebuild")
    check = SELFCHECK().check(read_records(path))
    if not check["ok"]:
        raise SystemExit(f"the self-check fails: {list(check['problems'].items())[:3]}")
    _write_json(frozen_path, {"files": {"N9g.jsonl": sha}, "frozen": _now(), "self_check": {k: v for k, v in check.items() if k != "problems"},
                              "note": "written after the ambiguity audit and before any scoring: the scorer, the chain's check and the verdict "
                                      "refuse if the file changed"})
    _seal([path, frozen_path, out / "n9g.manifest.json"])
    _log(f"N9g frozen: sha256 {sha}")
    return 0


# --------------------------------------------------------------------------
# recheck-v09
# --------------------------------------------------------------------------


def v08_gate_candidates() -> tuple[dict[str, set[str]], dict[str, set[str]], dict[str, int]]:
    """The v0.8 gates' candidates exactly as recheck-v08 builds them (M8g items, B8 / R14 whole states and B8 state-line shingles, A16 / S16 /
    A16s record units and question / option shingles) — owners prefixed as there."""
    from robo_jev.bench import a16s_v08, r14_v08, r16_v08
    from robo_jev.bench import gates_v08 as G
    from robo_jev.bench.iface import shingles
    from robo_jev.data.general_sets import normalize, record_units
    from robo_jev.decisions.records import read_records

    v8 = V8B()
    items = read_records(G.M8G_DIR / "m8g-items.jsonl")
    general = [{"item": i["item"], "state": i["state"], "stem": i["stem"], "correct": i["correct"], "own": list(i["own"]) + [str(d) for d in i["drawn"]]}
               for i in items if i["source"] == G.GENERAL_SOURCE]
    physics = [{"item": i["item"], "state": i["state"], "correct": tuple(i["correct"])} for i in items if i["source"] != G.GENERAL_SOURCE]
    units, grams = v8.merged(v8.V7B().candidates(general, {}), v8.physics_candidates(physics))
    units = {k: set(v) for k, v in units.items()}
    grams = {k: set(v) for k, v in grams.items()}
    counts = {"m8g_items": len(items)}
    b8 = read_records(G.B8_FILE)
    counts["b8"] = len(b8)
    for r in b8:
        units.setdefault(v8.V7B()._hash(normalize(r["request"]["state"])), set()).add(f"b8:{r['id']}")
        for line in r["request"]["state"].splitlines():
            if len(line.split()) >= 8:
                for gram in shingles(line):
                    grams.setdefault(gram, set()).add(f"b8shingle:{r['id']}")
    r14 = read_records(r14_v08.R14_ALL)
    counts["r14"] = len(r14)
    for r in r14:
        units.setdefault(v8.V7B()._hash(normalize(r["request"]["state"])), set()).add(f"r14:{r['id']}")
    sets = {"a16": read_records(r16_v08.R16_DIR / "A16.jsonl"), "s16": read_records(r16_v08.R16_DIR / "S16.jsonl"),
            "a16s": read_records(a16s_v08.A16S_DIR / "A16s.jsonl")}
    for kind, rows in sets.items():
        counts[kind] = len(rows)
        for r in rows:
            for unit in record_units(r):
                units.setdefault(v8.V7B()._hash(unit), set()).add(f"{kind}:{r['id']}")
            for question in r["request"]["questions"].values():
                for text in [str(question.get("instructions") or ""), *[str(v) for v in (question.get("criteria") or {}).values()]]:
                    if len(text.split()) >= 8:
                        for gram in shingles(text):
                            grams.setdefault(gram, set()).add(f"{kind}shingle:{r['id']}")
    return units, grams, counts


def cmd_recheck(args: argparse.Namespace) -> int:
    """train-v09n.jsonl against N9g and the v0.8 gates (module docstring) → overlap-v09.json (counts and hashes only)."""
    from robo_jev.bench import n9g_v09 as N
    from robo_jev.decisions.records import read_records

    train = Path(args.train_v09n)
    if not train.is_file():
        raise SystemExit(f"{train}: not there yet — run this once v0.9's CN training input exists")
    n9g_dir = Path(args.n9g_dir)
    problems = N.check_frozen(n9g_dir)
    if problems:
        raise SystemExit(f"N9g is not frozen as built: {problems}")
    records = read_records(n9g_dir / "N9g.jsonl")
    n_units, n_grams = n9g_candidates(records)
    g_units, g_grams, counts = v08_gate_candidates()
    sha = hash_and_forget(train)
    found = scan([train], n9g_units_map=n_units, n9g_grams=n_grams, gate_units=g_units, gate_grams=g_grams, forget=True)
    hits = found["hits"]
    n9g_ids = {str(r["id"]) for r in records}
    excluded_n9g = sorted(k for k in hits if k in n9g_ids and any(h.startswith(("unit:", "instructions:")) for h in hits[k]))
    n9g_shingle = sorted(k for k in hits if k in n9g_ids and any(h.startswith("shingle:") for h in hits[k]))
    gate_hits = {k: v for k, v in hits.items() if k not in n9g_ids}
    exclude_items = sorted(k for k in gate_hits if k.startswith(("mmlu:", "phys:")))
    exclude_b8 = sorted(k[len("b8:"):] for k in gate_hits if k.startswith("b8:") and any(h.startswith("unit:") for h in gate_hits[k]))
    shares = template_shingle_share(records, hits)
    payload = {
        "kind": "v09-overlap-v09", "script_version": SCRIPT_VERSION, "git": _git(),
        "train_v09n": {"path": _rel(train), "sha256": sha, "records": found["files"][_rel(train)]["records"]}, "files": found["files"],
        "excluded_n9g": excluded_n9g, "n9g_records_checked": len(records), "n9g_shingle_hits": n9g_shingle, "n9g_template_shingle_share": shares,
        "n9g_templates_flagged": sorted(t for t, v in shares.items() if v["flagged"]),
        "excluded_items": exclude_items, "excluded_b8": exclude_b8,
        "excluded_r14": sorted(k[len("r14:"):] for k in gate_hits if k.startswith("r14:")),
        "excluded_a16": sorted(k[len("a16:"):] for k in gate_hits if k.startswith("a16:")),
        "excluded_s16": sorted(k[len("s16:"):] for k in gate_hits if k.startswith("s16:")),
        "excluded_a16s": sorted(k[len("a16s:"):] for k in gate_hits if k.startswith("a16s:")),
        "b8_shingle_hits": sorted(k[len("b8shingle:"):] for k in gate_hits if k.startswith("b8shingle:")),
        "a16_shingle_hits": sorted(k[len("a16shingle:"):] for k in gate_hits if k.startswith("a16shingle:")),
        "s16_shingle_hits": sorted(k[len("s16shingle:"):] for k in gate_hits if k.startswith("s16shingle:")),
        "a16s_shingle_hits": sorted(k[len("a16sshingle:"):] for k in gate_hits if k.startswith("a16sshingle:")),
        "gate_records_checked": counts,
        "hit_counts": {"n9g_unit": len(excluded_n9g), "n9g_shingle": len(n9g_shingle), "gates": len(gate_hits)},
        "rule": "PRE-DECLARED (task v0.9b, 2026-10-17, before any v0.9 data): an N9g record with any text unit (whole state, a state line or an "
                "option text of >= 5 words, the question text of >= 5 words) in train-v09n is left out of every reading; N9g 8-word shingle hits "
                "are reported per record and per template (> 20 % of a template's records flagged, not excluded). The v0.8 gates keep "
                "recheck-v08's rules: an M8g MMLU item with any unit or shingle hit, an M8g physics item with a whole-state or state / "
                "correct-option shingle hit, a B8 or R14 request whose whole state is in the input, an A16 / S16 / A16s request with any text "
                "unit in it are left out; their shingle hits reported. Only counts, ids and hashes are written.",
        "created": _now()}
    _write_json(Path(args.out), payload)
    _log(f"train-v09n {sha[:12]}: exclude {len(excluded_n9g)} N9g records ({len(n9g_shingle)} with shingle hits), {len(exclude_items)} M8g items, "
         f"{len(exclude_b8)} B8, {len(payload['excluded_r14'])} R14, {len(payload['excluded_a16'])} A16, {len(payload['excluded_s16'])} S16, "
         f"{len(payload['excluded_a16s'])} A16s; flagged templates {payload['n9g_templates_flagged']}")
    return 0


# --------------------------------------------------------------------------
# check — the Spark chain's gate
# --------------------------------------------------------------------------


def _trained_on(run: Path) -> str | None:
    metrics = Path(run) / "metrics.json"
    if not metrics.is_file():
        return None
    return ((json.loads(metrics.read_text(encoding="utf-8")).get("manifest") or {}).get("data") or {}).get("file_sha256")


def gate_check(c_runs: list[Path], cn_runs: list[Path], f_runs: list[Path], *, overlap_v09: Path, v08_spark: Path = V08_SPARK) -> tuple[list[str], dict[str, Any]]:
    """The v0.9 chain's gate (module docstring): problems (empty = open) and what was read."""
    from robo_jev.bench import n9g_v09 as N
    from robo_jev.decisions.prompt import PROMPT_VERSION, fixed_text_digest
    from robo_jev.decisions.records import read_records

    problems: list[str] = []
    info: dict[str, Any] = {}
    # N9g: frozen, its manifest, the self-check
    frozen_problems = N.check_frozen(N.N9G_DIR)
    problems += [f"N9g: {p}" for p in frozen_problems]
    manifest_path = N.N9G_DIR / "n9g.manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else {}
    if not manifest:
        problems.append("N9g: n9g.manifest.json is missing")
    if not frozen_problems and manifest:
        frozen = json.loads(N.N9G_FROZEN.read_text(encoding="utf-8"))
        if frozen["files"]["N9g.jsonl"] != manifest["file"]["sha256"]:
            problems.append("N9g: the frozen sha256 is not the manifest's")
        check = SELFCHECK().check(read_records(N.N9G_FILE))
        info["n9g_self_check"] = {k: v for k, v in check.items() if k != "problems"}
        if not check["ok"]:
            problems.append(f"N9g: the self-check fails ({check['records_with_problems']} records)")
    # the v0.8 gates and the C runs (their input = train-v08, which recheck-v08 read); main's many-option path
    v8_problems, v8_info = V8B().gate_check([Path(r) for r in c_runs], V8B().ARTIFACTS / "v08" / "gates") if c_runs else (["no C runs given"], {})
    problems += [f"v0.8 gates / C runs: {p}" for p in v8_problems]
    info["v08_check"] = v8_info
    v08_sha = None
    overlap_v08 = ARTIFACTS / "v08" / "gates" / "overlap-v08.json"
    if overlap_v08.is_file():
        v08_sha = json.loads(overlap_v08.read_text(encoding="utf-8"))["train_v08"]["sha256"]
    build_sha = ((manifest.get("overlap") or {}).get("training_inputs") or {}).get(TRAIN_V08)
    if v08_sha is None or build_sha != v08_sha:
        problems.append(f"N9g's build-time scan did not read C's input (train-v08 {str(v08_sha)[:12]}; the scan read {str(build_sha)[:12]})")
    # the F runs: v0.8 F, trained on train-v08, and their checkpoints still the files v0.8 scored (else the reused files would not match)
    for seed, run in enumerate(f_runs):
        got = _trained_on(Path(run))
        info.setdefault("f_trained_on", {})[str(run)] = got
        if got is None or got != v08_sha:
            problems.append(f"{run}: not a v0.8 F run on train-v08 (metrics say {str(got)[:12]})")
        scored = Path(v08_spark) / f"B8-v08Fs{seed}-2b.json"
        checkpoint = Path(run) / "checkpoint.pt"
        if scored.is_file() and checkpoint.is_file():
            want = (json.loads(scored.read_text(encoding="utf-8")).get("checkpoint") or {}).get("sha256")
            if hash_and_forget(checkpoint) != want:
                problems.append(f"{checkpoint}: not the checkpoint v0.8 scored ({str(want)[:12]}; slimmed or replaced?) — F's reused files would not match")
    # the CN runs: trained on the train-v09n recheck-v09 read
    if not Path(overlap_v09).is_file():
        problems.append(f"{_rel(Path(overlap_v09))} is missing — run recheck-v09 on CN's training input")
    else:
        recheck = json.loads(Path(overlap_v09).read_text(encoding="utf-8"))
        sha = (recheck.get("train_v09n") or {}).get("sha256")
        info["train_v09n"] = sha
        on_disk = ARTIFACTS / TRAIN_V09N
        if on_disk.is_file() and hash_and_forget(on_disk) != sha:
            problems.append(f"{TRAIN_V09N} changed since recheck-v09 ({str(sha)[:12]}) — run it again")
        if not cn_runs:
            problems.append("no CN runs given")
        for run in cn_runs:
            got = _trained_on(Path(run))
            info.setdefault("cn_trained_on", {})[str(run)] = got
            if got != sha:
                problems.append(f"{run}: trained on {str(got)[:12]}, recheck-v09 read {str(sha)[:12]}")
        for run in c_runs:
            if _trained_on(Path(run)) == sha and sha != v08_sha:
                problems.append(f"{run}: a C run trained on CN's input")
    # the reused v0.8 files stay comparable: today's runner file and prompt digest
    from robo_jev.bench.n9g_v09 import sha256_file

    bundle_sha = sha256_file(REPO / "src" / "robo_jev" / "hf" / "robo_jev_decide.py")
    info["bundle_sha256"] = bundle_sha
    for name in REUSED_BENCH:
        path = Path(v08_spark) / name
        if not path.is_file():
            problems.append(f"reused v0.8 file missing: {_rel(path)}")
            continue
        got = json.loads(path.read_text(encoding="utf-8")).get("bundle_sha256")
        if got != bundle_sha:
            problems.append(f"{name} was scored with runner {str(got)[:12]}, today's is {bundle_sha[:12]} — reused files would not be comparable")
    digest = fixed_text_digest(PROMPT_VERSION)
    for name in REUSED_M8G:
        path = Path(v08_spark) / name
        if not path.is_file():
            problems.append(f"reused v0.8 file missing: {_rel(path)}")
            continue
        got = (json.loads(path.read_text(encoding="utf-8")).get("config") or {}).get("prompt_digest")
        if got != digest:
            problems.append(f"{name} was scored with prompt digest {str(got)[:12]}, today's is {digest[:12]}")
    return problems, info


def cmd_check(args: argparse.Namespace) -> int:
    problems, info = gate_check([Path(r) for r in args.c_runs], [Path(r) for r in args.cn_runs], [REPO / r for r in args.f_runs] if not args.f_runs_abs else [Path(r) for r in args.f_runs],
                                overlap_v09=Path(args.overlap_v09))
    print(json.dumps({"problems": problems, **info}, indent=1, default=str))
    return 2 if problems else 0


def main(argv: list[str] | None = None) -> int:
    from robo_jev.bench import n9g_v09 as N

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    b = sub.add_parser("build-n9g")
    b.add_argument("--out-dir", type=Path, default=N.N9G_DIR)
    b.add_argument("--seed", type=int, default=N.SEED)
    b.add_argument("--skip-overlap", action="store_true", help="tests only")
    b.add_argument("--force-before-scoring", action="store_true")
    f = sub.add_parser("freeze-n9g")
    f.add_argument("--out-dir", type=Path, default=N.N9G_DIR)
    f.add_argument("--force-before-scoring", action="store_true")
    r = sub.add_parser("recheck-v09")
    r.add_argument("--train-v09n", type=Path, default=ARTIFACTS / TRAIN_V09N)
    r.add_argument("--out", type=Path, default=N.OVERLAP_V09)
    r.add_argument("--n9g-dir", type=Path, default=N.N9G_DIR, help="tests only")
    c = sub.add_parser("check")
    c.add_argument("--c-runs", nargs="*", default=[], help="the fetched C run directories (each with metrics.json)")
    c.add_argument("--cn-runs", nargs="*", default=[], help="the fetched CN run directories")
    c.add_argument("--f-runs", nargs="*", default=list(F_RUNS))
    c.add_argument("--f-runs-abs", action="store_true", help="--f-runs are absolute paths (tests)")
    c.add_argument("--overlap-v09", type=Path, default=N.OVERLAP_V09)
    args = parser.parse_args(argv)
    return {"build-n9g": cmd_build_n9g, "freeze-n9g": cmd_freeze_n9g, "recheck-v09": cmd_recheck, "check": cmd_check}[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
