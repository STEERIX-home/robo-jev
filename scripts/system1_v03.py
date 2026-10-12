#!/usr/bin/env python
"""Task v0.3c — System-1 derived questions for the engine training set, and the unified sealed-overlap check of a training input.

    # derived questions (CPU, a few minutes): v02 train-50k → engine-s1 + manifest (statistics, per-epoch exposure, checks)
    python scripts/system1_v03.py transform --input artifacts/simgen/v02/train-50k.jsonl --out artifacts/simgen/v03/engine-s1.jsonl \
        --train-config configs/train/decisions-v0-lora-h100.yaml
    # every record of a training input against every sealed set (T1–T7, G1, G2, T8a, T8b): ids, whole requests, whole states, and the
    # text units of the sets written outside our engine; plus the d1 lineage check against G1 and the engine seed ranges
    python scripts/system1_v03.py overlap --input artifacts/simgen/v03/train-v03.jsonl --out artifacts/simgen/v03/train-v03.overlap.json

The rules of the transform are :mod:`robo_jev.simgen.system1_questions`. ``transform`` refuses an input inside a sealed location and any
record that is not a training-split engine record; it writes byte-stable JSONL (``simgen.record.dumps``) and checks, for every record,
that the original questions, labels and fields are unchanged, that both validators (the engine's and the trainer's) pass, and that the
derived wording adds no word or 3-gram reserved for T2's held-out wordings. ``overlap`` reuses the existing checks — the narration
task's sealed index (:class:`robo_jev.simgen.narration.SealedIndex`: whole-state and whole-request hashes of every sealed set, text
units of G1, G2, T8a and T8b), the d1 mix lineage check (:func:`robo_jev.decisions.single_probe.overlap`) and the engine's seed ranges
(``configs/simgen/holdout-v0b.yaml``) — and exits non-zero on any hit.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterator

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

SCRIPT_VERSION = "v03c-1"
CODE_PATHS = ("src/robo_jev/simgen/system1_questions.py", "scripts/system1_v03.py", "src/robo_jev/bench/system1.py")
TESTS = REPO / "artifacts" / "simgen" / "v0b" / "tests"
PROBES = REPO / "artifacts" / "simgen" / "probes"
T8 = REPO / "artifacts" / "t8"
#: Every sealed set (the v0.3 rule's readings and the secondary sets): name → path.
SEALED = {**{f"T{i}": TESTS / f"T{i}.jsonl" for i in range(1, 8)}, "G1": PROBES / "G1.jsonl", "G2": PROBES / "G2.jsonl",
          "T8a": T8 / "T8a.jsonl", "T8b": T8 / "T8b.jsonl"}  # fmt: skip
#: Directories and files a training input must never come from.
SEALED_LOCATIONS = (TESTS, T8, PROBES / "G1.jsonl", PROBES / "G2.jsonl")


def _log(message: str) -> None:
    print(f"[v03c {dt.datetime.now().strftime('%H:%M:%S')}] {message}", file=sys.stderr, flush=True)


def _git() -> dict[str, Any]:
    def run(*args: str) -> str | None:
        try:
            return subprocess.run(["git", "-C", str(REPO), *args], capture_output=True, text=True, check=True).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return None

    status = run("status", "--porcelain", "--", *CODE_PATHS)
    return {"commit": run("rev-parse", "--short", "HEAD"), "dirty": bool(status) if status is not None else None}


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(payload, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    temp.replace(path)


def _lines(path: Path) -> Iterator[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _read_yaml(path: Path) -> dict[str, Any]:
    import yaml

    return yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}


def refuse_sealed_path(path: Path) -> None:
    """A training input or output must not live in a sealed location (T1–T7 directory, T8 directory, G1, G2)."""
    resolved = Path(path).resolve()
    for location in SEALED_LOCATIONS:
        location = location.resolve()
        if resolved == location or location in resolved.parents:
            raise SystemExit(f"{path}: refusing a sealed location ({location})")


# --------------------------------------------------------------------------
# transform
# --------------------------------------------------------------------------


def _unchanged(source: dict[str, Any], out: dict[str, Any]) -> list[str]:
    """Problems if the transform changed anything but appending derived questions/labels and the two meta fields."""
    problems = []
    names = list(source["request"]["questions"])
    if list(out["request"]["questions"])[: len(names)] != names or list(out["labels"])[: len(names)] != names:
        problems.append("original question order changed")
    if any(out["request"]["questions"][n] != source["request"]["questions"][n] or out["labels"][n] != source["labels"][n] for n in names):
        problems.append("an original question or label changed")
    if out["request"]["state"] != source["request"]["state"]:
        problems.append("state changed")
    for key in ("schema", "id", "generator", "truth", "rationale"):
        if out.get(key) != source.get(key):
            problems.append(f"{key} changed")
    meta = {k: v for k, v in out["meta"].items() if k not in ("system1", "question_families")}
    if meta != {k: v for k, v in source["meta"].items() if k != "question_families"} or list(out) != list(source):
        problems.append("meta or the key order changed")
    return problems


def _new_held_out_hits(source: dict[str, Any], out: dict[str, Any]) -> list[str]:
    """Held-out-only words / 3-grams (T2 wording bans) in a derived instruction or option that its source instructions did not already
    contain — the original instructions fill catalog names into their slots, so their own hits are not the transform's."""
    from robo_jev.simgen.phrasing import language_wide_hits

    lang = out["meta"].get("language") or "en"
    found = []
    for name, item in out["meta"]["system1"]["derived"].items():
        question = out["request"]["questions"][name]
        words, grams = language_wide_hits(lang, question["instructions"])
        known_words, known_grams = set(), set()
        for source_name in item["of"]:
            w, g = language_wide_hits(lang, source["request"]["questions"][source_name]["instructions"])
            known_words |= set(w)
            known_grams |= set(g)
        found += [f"{name}: word {w!r}" for w in set(words) - known_words] + [f"{name}: 3-gram {g!r}" for g in set(grams) - known_grams]
        for text in question["criteria"].values():
            w, g = language_wide_hits(lang, text)
            found += [f"{name}: option {x!r}" for x in w + g]
    return found


def cmd_transform(args: argparse.Namespace) -> int:
    from robo_jev.decisions.examples import is_dev
    from robo_jev.decisions.records import file_sha256
    from robo_jev.decisions.records import validate_record as trainer_validate
    from robo_jev.decisions.train import resolve_config
    from robo_jev.simgen import system1_questions as s1
    from robo_jev.simgen.record import dumps
    from robo_jev.simgen.record import validate_record as engine_validate

    refuse_sealed_path(args.input)
    refuse_sealed_path(args.out)
    config = resolve_config(_read_yaml(args.train_config)) if args.train_config else resolve_config({})
    q = int(config["questions_per_request"])
    problems: dict[str, list[str]] = {}
    counts: Counter = Counter()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temp = args.out.with_name(args.out.name + ".tmp")
    start = dt.datetime.now()
    with temp.open("w", encoding="utf-8") as handle:
        for index, record in enumerate(_lines(args.input)):
            if args.limit is not None and index >= args.limit:
                break
            out = s1.transform_record(record, max_derived=args.max_derived)
            found = engine_validate(out) + trainer_validate(out)
            changed = _unchanged(record, out)
            wording = _new_held_out_hits(record, out)
            counts["records"] += 1
            counts["validation_errors"] += bool(found)
            counts["originals_changed"] += bool(changed)
            counts["held_out_wording_hits"] += len(wording)
            if (found or changed or wording) and len(problems) < 20:
                problems[str(record["id"])] = (found + changed + wording)[:5]
            handle.write(dumps(out) + "\n")
    if counts["validation_errors"] or counts["originals_changed"] or counts["held_out_wording_hits"]:
        temp.unlink()
        raise SystemExit(f"transform checks failed: {dict(counts)} — first problems {json.dumps(problems, ensure_ascii=False)[:2000]}")
    temp.replace(args.out)
    seconds = (dt.datetime.now() - start).total_seconds()
    _log(f"transformed {counts['records']} records in {seconds:.0f} s → {args.out}; statistics …")

    def train_only(path: Path) -> Iterator[dict[str, Any]]:
        for i, record in enumerate(_lines(path)):
            if args.limit is not None and i >= args.limit:
                break
            if not is_dev(record, fraction=config["dev_fraction"], salt=config["dev_salt"]):
                yield record

    stats = s1.summarize(_lines(args.out))
    before = s1.exposure(train_only(args.input), q=q, seed=int(config["seed"]))
    after = s1.exposure(train_only(args.out), q=q, seed=int(config["seed"]))
    input_manifest = args.input.with_name(args.input.stem + ".manifest.json")
    manifest = {
        "kind": "system1-engine-v0", "version": s1.VERSION, "script_version": SCRIPT_VERSION, "git": _git(),
        "input": {"path": str(args.input), "sha256": file_sha256(args.input), "records": counts["records"],
                  "manifest": {"path": str(input_manifest), "sha256": file_sha256(input_manifest)} if input_manifest.is_file() else None},
        "out": {"path": str(args.out), "sha256": file_sha256(args.out), "bytes": args.out.stat().st_size, "records": counts["records"]},
        "rule": {"band_edges": [1 / 3, 2 / 3], "smooth": s1.SMOOTH, "same_below": s1.SAME_BELOW, "max_questions": s1.MAX_QUESTIONS,
                 "max_derived": args.max_derived,
                 "priority": "key-question band, then comparisons (request order), then the other bands (request order)",
                 "event_questions": "every yes/no question and every score whose meta.basis source is not 'quantity' (bench.system1 events)",
                 "pairs": "bench.system1.basis_event_pairs, same polarity only; first/second by a seeded coin per record and pair",
                 "label_kind": "state", "families": "meta.question_families: s1:band / s1:comparison",
                 "frames_sha256": hashlib.sha256(json.dumps(s1.FRAMES, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()},
        "stats": stats,
        "exposure": {
            "what": "examples of one training epoch (epoch 0, the trainer's own question pick: select_questions with its seed stream) by "
                    "question category, train-split records only — derived questions share the choice slot with the original choices",
            "train_config": str(args.train_config) if args.train_config else None, "q": q, "seed": int(config["seed"]),
            "dev_fraction": config["dev_fraction"], "dev_salt": config["dev_salt"], "before": before, "after": after,
        },
        "checks": {"validation_errors": counts["validation_errors"], "originals_changed": counts["originals_changed"],
                   "held_out_wording_new_hits": counts["held_out_wording_hits"],
                   "split_guard": "every record has meta.split = generator.split.name = 'train' (else SealedRecordError)"},
        "seconds": round(seconds, 1),
        "created": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
    }  # fmt: skip
    _write_json(args.out.with_name(args.out.stem + ".manifest.json"), manifest)
    _log(f"derived {stats['derived']} ({stats['derived_per_record_mean']:.2f} per record), truncated {stats['records_truncated_by_budget']} records "
         f"→ {args.out.with_name(args.out.stem + '.manifest.json')}")
    return 0


# --------------------------------------------------------------------------
# overlap
# --------------------------------------------------------------------------


def part_of(record: dict[str, Any]) -> str:
    """Which part of a v0.3 training input a record belongs to: engine · mujoco · d1 · general (else ``other``)."""
    from robo_jev.data.general_sets import SOURCES
    from robo_jev.decisions.single_probe import MIX_SPLITS

    meta = record.get("meta") or {}
    if meta.get("set") == "v03b-mujoco-train":
        return "mujoco"
    if isinstance(meta.get("source"), dict) and meta["source"].get("set") in SOURCES:
        return "general"
    if str(record.get("id", "")).startswith("v0b-") and isinstance(meta.get("basis"), dict):
        return "engine"
    if meta.get("split") in MIX_SPLITS and meta.get("origin_group"):
        return "d1"
    return "other"


def cmd_overlap(args: argparse.Namespace) -> int:
    from robo_jev.data.general_sets import record_units
    from robo_jev.decisions.records import file_sha256, read_records
    from robo_jev.decisions.single_probe import overlap
    from robo_jev.simgen.holdout import load_holdout, request_hash
    from robo_jev.simgen.narration import SealedIndex, state_sha256

    sealed = {name: path for name, path in SEALED.items() if path.is_file()}
    missing = sorted(set(SEALED) - set(sealed))
    if missing and not args.allow_missing:
        raise SystemExit(f"sealed sets missing: {missing} (--allow-missing to check against the rest)")
    index = SealedIndex.build(sealed)
    sealed_ids = {name: {str(r["id"]) for r in _lines(path)} for name, path in sealed.items()}
    outside_units = {name: units for name, units in index.units.items() if units}
    unit_hash = lambda unit: hashlib.sha256(unit.encode("utf-8")).hexdigest()[:16]  # noqa: E731 — SealedIndex's 16-hex unit hashes
    seed_ranges = load_holdout().seed_ranges
    train_low, train_high = seed_ranges["train"]
    report: dict[str, Any] = {"parts": {}, "hits": defaultdict(list)}
    per_part: dict[str, Counter] = defaultdict(Counter)
    d1_records: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    duplicates = 0
    for record in _lines(args.input):
        part = part_of(record)
        tally = per_part[part]
        tally["records"] += 1
        rid = str(record["id"])
        duplicates += rid in seen_ids
        seen_ids.add(rid)
        rh, sh = request_hash(record), state_sha256(record["request"]["state"])
        units = {unit_hash(u) for u in record_units(record)}
        for name in sealed:
            found = []
            if rid in sealed_ids[name]:
                found.append("id")
            if rh in index.requests[name]:
                found.append("request")
            if sh in index.states[name]:
                found.append("state")
            if name in outside_units and units & outside_units[name]:
                found.append(f"{len(units & outside_units[name])} units")
            for what in found:
                if part == "d1" and name == "G1" and what.endswith("units"):
                    tally["info:G1_template_units"] += 1  # same d1 generator templates by construction; the rule is the lineage check
                    continue
                tally[f"{name}:{what.split()[-1]}"] += 1
                if len(report["hits"][part]) < 20:
                    report["hits"][part].append(f"{rid} ↔ {name}: {what}")
        if part == "engine":
            seed = int(record["generator"]["seed"])
            if not train_low <= seed <= train_high:
                tally["seed_outside_train_range"] += 1
            if (record.get("meta") or {}).get("split") != "train":
                tally["not_train_split"] += 1
        elif part == "d1":
            d1_records.append(record)
        elif part == "other":
            tally["unknown_part"] += 1
    lineage = None
    if d1_records and "G1" in sealed:
        crossing = overlap(read_records(sealed["G1"]), d1_records)
        lineage = {"origin_groups": len(crossing["origin_groups"]), "source_records": len(crossing["source_records"])}
    hits = sum(v for tally in per_part.values() for k, v in tally.items() if k != "records" and not k.startswith("info:"))
    hits += duplicates + (sum(lineage.values()) if lineage else 0)
    out = {
        "kind": "v03-sealed-overlap", "script_version": SCRIPT_VERSION, "git": _git(),
        "input": {"path": str(args.input), "sha256": file_sha256(args.input), "records": sum(t["records"] for t in per_part.values())},
        "sealed": {name: {"path": str(path), "sha256": index.files[name]["sha256"], "records": len(sealed_ids[name]),
                          "units": len(index.units.get(name) or ())} for name, path in sealed.items()},
        "missing_sealed": missing,
        "checks": {"ids": "record id equal to a sealed record id", "request": "whole-request sha256 (holdout.request_hash)",
                   "state": "whole-state sha256 (normalised, narration.state_sha256)",
                   "units": "text units (normalised state, state lines / list items and option texts with >= 5 words) shared with a set "
                            "written outside our engine (G1, G2, T8a, T8b; T1–T7 share the engine's templates, so they are matched on whole "
                            "texts only — the narration task's rule)",
                   "seed_range": f"engine records' generator seeds inside the training range [{train_low}, {train_high}] (holdout-v0b)",
                   "d1_lineage": "d1 mix vs G1: shared origin groups / source records (single_probe.overlap — the mix command's check). "
                                 "The d1 mix and G1 come from the same templated d1 generator, so their text units overlap by construction "
                                 "(counted as info:G1_template_units, not as hits); G1 is partly in-family for the mix since v0.1 and is "
                                 "read only as a secondary set — G2 is the independent general check"},
        "parts": {part: dict(sorted(tally.items())) for part, tally in sorted(per_part.items())},
        "duplicate_ids": duplicates, "d1_lineage": lineage, "total_hits": hits, "first_hits": dict(report["hits"]),
        "created": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
    }  # fmt: skip
    _write_json(args.out, out)
    _log(f"overlap: {out['input']['records']} records, {hits} hits → {args.out}")
    return 0 if hits == 0 else 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    transform = sub.add_parser("transform", help="engine training records → + System-1 derived questions (+ manifest)")
    transform.add_argument("--input", type=Path, required=True)
    transform.add_argument("--out", type=Path, required=True)
    transform.add_argument("--train-config", type=Path, default=REPO / "configs" / "train" / "decisions-v0-lora-h100.yaml",
                           help="the training config whose dev split and q the exposure statistics use")
    transform.add_argument("--limit", type=int, default=None, help="only the first N records (smoke runs)")
    transform.add_argument("--max-derived", type=int, default=None,
                           help="optional cap on derived questions per record (default: none — only the 12-question budget)")
    transform.set_defaults(func=cmd_transform)
    check = sub.add_parser("overlap", help="every record of a training input against every sealed set → JSON (exit 2 on any hit)")
    check.add_argument("--input", type=Path, required=True)
    check.add_argument("--out", type=Path, required=True)
    check.add_argument("--allow-missing", action="store_true", help="check against the sealed sets that exist (else refuse)")
    check.set_defaults(func=cmd_overlap)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
