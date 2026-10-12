#!/usr/bin/env python
"""Task v0.2 §1-1 — build the general-reading training mix ``artifacts/simgen/v02/general-train.jsonl`` and its manifest.

    uv run --with "pyarrow<20" python scripts/convert_general_sets.py --out artifacts/simgen/v02/general-train.jsonl

(pyarrow 20+ wants NumPy 2; the lock pins NumPy 1.26.)

Downloads the train parquet of each source in :data:`robo_jev.data.general_sets.SOURCES` at its pinned revision with
``huggingface_hub`` (cache under ``--cache``), converts the rows (rules: :mod:`robo_jev.data.general_sets`), drops every row that
shares a text unit with a sealed set (G1, G2, T1–T7, every ``*.jsonl`` under ``--t8-dir`` — default ``artifacts/simgen/t8`` and
``artifacts/t8`` — when present, and the PIQA validation
strings, T8a's source, downloaded only for this check), samples ``--per-source`` rows per source with ``--seed`` and interleaves them.

The manifest records, per source: dataset, revision, file sha256, licence and where the card states it, counts (rows, converted,
dropped by reason, dropped for overlap per sealed set, sampled, gold-key counts); the excluded candidates and why; the sealed sets
checked (path, sha256, unit count) and the overlap after the build (must be 0); the seed, converter versions and git state.
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

SCRIPT_VERSION = "v02-general-1"
#: The code the output depends on — ``git_dirty`` in the manifest says whether these differ from HEAD.
CODE_PATHS = ("src/robo_jev/data/general_sets.py", "src/robo_jev/decisions/single_probe.py", "src/robo_jev/decisions/records.py",
              "scripts/convert_general_sets.py")  # fmt: skip
#: Sealed sets every general record is checked against (relative to the repo).
SEALED = {
    "G1": "artifacts/simgen/probes/G1.jsonl",
    "G2": "artifacts/simgen/probes/G2.jsonl",
    **{f"T{i}": f"artifacts/simgen/v0b/tests/T{i}.jsonl" for i in range(1, 8)},
}
#: PIQA validation (T8a's source) — read only for the overlap check, never converted or trained on.
PIQA = {"repo": "ybisk/piqa", "revision": "142c51238b3ca2bc61e9a075913871b8b600e8e1", "file": "plain_text/validation/0000.parquet",
        "use": "overlap check only (T8a source); never converted or trained on"}  # fmt: skip


def _log(message: str) -> None:
    print(f"[v02-general {dt.datetime.now().strftime('%H:%M:%S')}] {message}", file=sys.stderr, flush=True)


def _git() -> dict[str, Any]:
    def run(*args: str) -> str | None:
        try:
            return subprocess.run(["git", "-C", str(REPO), *args], capture_output=True, text=True, check=True).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return None

    status = run("status", "--porcelain", "--", *CODE_PATHS)
    return {"commit": run("rev-parse", "--short", "HEAD"), "dirty": bool(status) if status is not None else None}


def _download(repo: str, revision: str, filename: str, cache: Path) -> Path:
    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(repo, filename, repo_type="dataset", revision=revision, cache_dir=str(cache)))


def _rows(path: Path) -> list[dict[str, Any]]:
    import pyarrow.parquet as pq  # `uv run --with pyarrow`

    return pq.read_table(path).to_pylist()


def main(argv: list[str] | None = None) -> int:
    from robo_jev.data.general_sets import (
        EXCLUDED, GENERAL_VERSION, INSTRUCTIONS, MIN_UNIT_WORDS, SOURCES, build_general, raw_units, record_units, sealed_unit_sets, unit_digest,
    )  # fmt: skip
    from robo_jev.decisions.records import file_sha256, validate_record
    from robo_jev.decisions.single_probe import CONVERTER_VERSION, summarize

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=REPO / "artifacts" / "simgen" / "v02" / "general-train.jsonl")
    parser.add_argument("--cache", type=Path, default=REPO / "artifacts" / "datasets" / "public-train" / "hf-cache")
    parser.add_argument("--per-source", type=int, default=7000)
    parser.add_argument("--seed", type=int, default=20261012)
    parser.add_argument("--t8-dir", type=Path, nargs="+", default=[REPO / "artifacts" / "simgen" / "t8", REPO / "artifacts" / "t8"],
                        help="directories whose *.jsonl are the outside test T8 (the brief's artifacts/simgen/t8 and the T8 task's artifacts/t8)")
    parser.add_argument("--sources", default=",".join(SOURCES))
    args = parser.parse_args(argv)

    names = [n.strip() for n in args.sources.split(",") if n.strip()]
    unknown = [n for n in names if n not in SOURCES]
    if unknown:
        raise SystemExit(f"unknown sources {unknown} (known: {sorted(SOURCES)})")
    rows, files = {}, {}
    for name in names:
        spec = SOURCES[name]
        path = _download(spec["repo"], spec["revision"], spec["file"], args.cache)
        rows[name] = _rows(path)
        files[name] = {"path": str(path.relative_to(REPO)) if path.is_relative_to(REPO) else str(path), "sha256": file_sha256(path), "bytes": path.stat().st_size}
        _log(f"{name}: {len(rows[name])} rows from {spec['repo']}@{spec['revision'][:10]} {spec['file']}")

    sealed_paths = {name: REPO / rel for name, rel in SEALED.items()}
    missing = [str(p) for p in sealed_paths.values() if not p.is_file()]
    if missing:
        raise SystemExit(f"sealed sets missing — the overlap check needs them: {missing}")
    t8_files = sorted(path for directory in args.t8_dir if directory.is_dir() for path in directory.glob("*.jsonl"))
    for path in t8_files:
        sealed_paths[f"T8:{path.parent.name}/{path.name}"] = path
    piqa_path = _download(PIQA["repo"], PIQA["revision"], PIQA["file"], args.cache)
    piqa_rows = _rows(piqa_path)
    piqa_units = raw_units(text for row in piqa_rows for text in (row.get("goal"), row.get("sol1"), row.get("sol2")) if text)
    sealed = sealed_unit_sets(sealed_paths, extra={"PIQA-validation": piqa_units})
    _log(f"sealed units: " + ", ".join(f"{k} {len(v)}" for k, v in sealed.items()) + (f" (no T8 files under {[str(d) for d in args.t8_dir]})" if not t8_files else ""))

    records, report = build_general(rows, per_source=args.per_source, seed=args.seed, sealed=sealed, log=_log)
    problems = [(r["id"], p) for r in records for p in validate_record(r)]
    if problems:
        raise SystemExit(f"{len(problems)} invalid records — first {problems[:3]}")
    ids = [r["id"] for r in records]
    if len(set(ids)) != len(ids):
        raise SystemExit("duplicate record ids")
    # the overlap after the build — recomputed on the written records, must be 0 for every sealed set
    general_hashes = set()
    import hashlib

    for record in records:
        general_hashes |= {hashlib.sha256(u.encode("utf-8")).hexdigest() for u in record_units(record)}
    after = {name: len(general_hashes & units) for name, units in sealed.items()}
    if any(after.values()):
        raise SystemExit(f"overlap with sealed sets after the build: {after}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    temp = args.out.with_name(args.out.name + ".tmp")
    with temp.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    temp.replace(args.out)
    units_all = set().union(*(record_units(r) for r in records))
    git = _git()
    manifest = {
        "kind": "decisions-general-train-v0", "name": "general-train", "role": "training mix (general reading) — train splits only",
        "file": {"path": str(args.out.relative_to(REPO)) if args.out.is_relative_to(REPO) else str(args.out), "sha256": file_sha256(args.out),
                 "bytes": args.out.stat().st_size, "records": len(records)},
        "seed": args.seed, "per_source": args.per_source,
        "order": "round-robin over the sources, each round in a seeded source order — any prefix is balanced across sources (the doses take prefixes)",
        "sources": {name: {**{k: v for k, v in SOURCES[name].items()}, "downloaded": files[name], **report[name],
                           "instructions": list(INSTRUCTIONS[name])} for name in names},
        "licenses": {name: SOURCES[name]["license"] for name in names},
        "attribution": "CC-BY-4.0 sources (Social IQa, QASC, CosmosQA) require attribution if a model trained on them is released: cite "
                       + "; ".join(SOURCES[n]["citation"] for n in names) + ".",
        "excluded": EXCLUDED,
        "overlap_check": {
            "units": f"normalised whole state + state line values / list items and options with >= {MIN_UNIT_WORDS} words (instructions excluded)",
            "sealed": {name: {"path": str(path.relative_to(REPO)) if Path(path).is_relative_to(REPO) else str(path), "sha256": file_sha256(path),
                              "units": len(sealed[name])} for name, path in sealed_paths.items()},
            "piqa_validation": {**PIQA, "rows": len(piqa_rows), "units": len(piqa_units), "sha256": file_sha256(piqa_path)},
            "t8_files_present": [p.name for p in t8_files],
            "overlap_after_build": after,
            "general_units": len(units_all), "general_units_digest": unit_digest(units_all),
        },
        "converter": {"general": GENERAL_VERSION, "single_probe": CONVERTER_VERSION, "script_version": SCRIPT_VERSION, "git": git["commit"],
                      "git_dirty": git["dirty"]},
        "counts": summarize(records),
        "created": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
    }  # fmt: skip
    manifest_path = args.out.with_name(args.out.stem + ".manifest.json")
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    _log(f"{len(records)} records → {args.out} (sha256 {manifest['file']['sha256'][:12]}…), manifest {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
