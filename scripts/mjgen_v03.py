#!/usr/bin/env python
"""Task v0.3b — MuJoCo-labelled training families (a second physics source; no training here).

    # cost per rollout and per record on this machine (one process)
    python scripts/mjgen_v03.py measure --groups 4
    # label candidate groups (CPU; run in a capped systemd unit) — appends to the candidates file, resumable
    python scripts/mjgen_v03.py label --groups-per-family 4000 --workers 18
    # balanced selection → artifacts/simgen/v03/mujoco-train.jsonl + manifest + stats
    python scripts/mjgen_v03.py assemble --records-per-family 5000
    # physics sanity (monotonicity, analytic thresholds, determinism) and the T8b overlap check
    python scripts/mjgen_v03.py sanity --workers 8
    python scripts/mjgen_v03.py overlap

Records are ``simgen-request/0`` (the trainer and ``scripts/diagnose_decisions.py`` read them). T8 stays sealed: nothing here
reads T8 files except the overlap check, which only compares text.
"""

from __future__ import annotations

import argparse
import collections
import datetime as dt
import json
import math
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

SCRIPT_VERSION = "v03b-1"
OUT = REPO / "artifacts" / "simgen" / "v03"
MASTER_SEED = 20261013


def _log(message: str) -> None:
    print(f"[v03b {dt.datetime.now().strftime('%H:%M:%S')}] {message}", file=sys.stderr, flush=True)


def _git_sha() -> str | None:
    try:
        return subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True).stdout.strip() or None
    except (OSError, subprocess.CalledProcessError):
        return None


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(payload, ensure_ascii=False, indent=1, default=str) + "\n", encoding="utf-8")
    temp.replace(path)


def group_to_json(group: Any) -> dict[str, Any]:
    from dataclasses import asdict

    return json.loads(json.dumps(asdict(group), default=list))


def group_from_json(payload: dict[str, Any]) -> Any:
    from robo_jev.mjgen.families import Group

    return Group(**payload)


def _label_job(job: tuple[str, int, int, int, float]) -> dict[str, Any]:
    from robo_jev.mjgen.families import candidate, label_group

    family, index, master, k, pair_share = job
    started = time.perf_counter()
    group = candidate(family, index, master, k=k, pair_share=pair_share)
    labelled = label_group(group, k=k)
    return {"group": group_to_json(group), "labelled": json.loads(json.dumps(labelled)), "seconds": round(time.perf_counter() - started, 3)}


def cmd_measure(args: argparse.Namespace) -> int:
    from robo_jev.mjgen.families import FAMILIES
    from robo_jev.mjgen.generate import group_records

    report = {}
    for family in FAMILIES:
        started = time.perf_counter()
        rollouts = records = 0
        for index in range(args.groups):
            row = _label_job((family, 900_000 + index, args.seed, args.k, args.pair_share))
            rollouts += row["labelled"]["rollouts"]
            records += len(group_records(group_from_json(row["group"]), row["labelled"], mujoco_version="measure"))
        seconds = time.perf_counter() - started
        report[family] = {"groups": args.groups, "records": records, "rollouts": rollouts, "seconds": round(seconds, 2),
                          "ms_per_rollout": round(1000 * seconds / max(rollouts, 1), 2), "s_per_record": round(seconds / max(records, 1), 3)}
        _log(f"{family}: {report[family]}")
    _write_json(Path(args.out), {"kind": "v03b-measure", "k": args.k, "families": report, "created": dt.datetime.now().astimezone().isoformat(timespec="seconds")})
    return 0


def cmd_label(args: argparse.Namespace) -> int:
    import multiprocessing

    from robo_jev.mjgen.families import FAMILIES

    out = Path(args.candidates)
    out.parent.mkdir(parents=True, exist_ok=True)
    done: set[tuple[str, int]] = set()
    if out.is_file():
        with out.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    row = json.loads(line)
                    done.add((row["group"]["family"], int(row["group"]["index"])))
    families = args.families or list(FAMILIES)
    jobs = [(family, index, args.seed, args.k, args.pair_share) for index in range(args.start, args.start + args.groups_per_family) for family in families
            if (family, index) not in done]
    _log(f"labelling {len(jobs)} groups ({len(done)} already in {out}) with K={args.k} per prior on {args.workers} workers")
    started = time.perf_counter()
    cpu = 0.0
    count = 0
    with out.open("a", encoding="utf-8") as handle, multiprocessing.get_context("fork").Pool(args.workers) as pool:
        for row in pool.imap_unordered(_label_job, jobs, chunksize=1):
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
            cpu += row["seconds"]
            count += 1
            if count % 200 == 0:
                wall = time.perf_counter() - started
                _log(f"{count}/{len(jobs)} groups · wall {wall:.0f} s · worker cpu {cpu:.0f} s · ETA {wall / count * (len(jobs) - count):.0f} s")
    wall = time.perf_counter() - started
    _log(f"labelled {count} groups in {wall:.0f} s wall, {cpu:.0f} s worker time")
    log = out.with_suffix(".runs.jsonl")
    with log.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"groups": count, "wall_s": round(wall, 1), "worker_s": round(cpu, 1), "workers": args.workers, "k": args.k,
                                 "seed": args.seed, "families": families, "start": args.start, "groups_per_family": args.groups_per_family,
                                 "git": _git_sha(), "finished": dt.datetime.now().astimezone().isoformat(timespec="seconds")}) + "\n")
    return 0


def cmd_assemble(args: argparse.Namespace) -> int:
    import mujoco

    from robo_jev.bench.define import ids_sha256
    from robo_jev.decisions.records import file_sha256
    from robo_jev.mjgen import families as fam
    from robo_jev.mjgen import physics as ph
    from robo_jev.mjgen.generate import cap_still_pairs, group_records, select_balanced, stats

    rows = []
    with Path(args.candidates).open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    items = [(group_from_json(r["group"]), r["labelled"]) for r in rows]
    items, pair_cap = cap_still_pairs(items, share=args.still_pair_share)
    worker_seconds = {f: sum(r["seconds"] for r in rows if r["group"]["family"] == f) for f in fam.FAMILIES}
    targets = {}
    for family in fam.FAMILIES:
        pool = [(g, lab) for g, lab in items if g.family == family]
        per_group = sum(len(lab["members"]) for _, lab in pool) / max(1, len(pool))
        targets[family] = int(math.ceil(args.records_per_family / per_group)) if pool else 0
    kept, selection = select_balanced(items, targets)
    git = _git_sha()
    records = []
    for group, labelled in kept:
        records.extend(group_records(group, labelled, mujoco_version=mujoco.__version__, git=git))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    temp = out.with_name(out.name + ".tmp")
    with temp.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    temp.replace(out)
    summary = stats(records)
    _write_json(out.with_suffix(".stats.json"), summary)
    examples = {}
    for family in fam.FAMILIES:
        first = next((g for g, _ in kept if g.family == family), None)
        if first is None:
            continue
        v = first.visible
        if family == "tilt":
            examples[family] = ph.tilt_xml(v["dims"], v["mass"], v["orientation"])
        elif family == "grip":
            examples[family] = ph.grip_xml(v["dims"], 1.0)
        else:
            examples[family] = ph.tower_xml([(b["dims"], b["mass"]) for b in v["boxes"]])
    runs = []
    runs_path = Path(args.candidates).with_suffix(".runs.jsonl")
    if runs_path.is_file():
        runs = [json.loads(line) for line in runs_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    q_types = collections.Counter(q["type"] for r in records for q in r["request"]["questions"].values())
    manifest = {
        "set": "v03b-mujoco-train", "what": "MuJoCo-labelled training families (tilt, grip, tower); labels = outcome frequencies over K hidden draws "
        "under three prior settings; one-fact pairs; balanced key labels", "split": "train", "sealed": False,
        "generator": fam.GENERATOR, "script_version": SCRIPT_VERSION, "git": git, "mujoco": mujoco.__version__, "k_per_prior": fam.DEFAULT_K if not rows else rows[0]["labelled"]["k"],
        "priors": fam.PRIOR_SETTINGS, "master_seed": runs[-1]["seed"] if runs else None,
        "group_seed_rule": "sha256('v03b|master|family|index')[:8]; hidden-draw points: scrambled Latin hypercube seeded by group seed ^ 0xD3A5",
        "catalogs": {
            "tilt": {"surfaces": fam.TILT_SURFACES, "loads": fam.TILT_LOADS, "fills": fam.TILT_FILLS, "robots": fam.TILT_ROBOTS},
            "grip": {"pads": fam.GRIP_PADS, "surfaces": fam.GRIP_SURFACES, "items": fam.GRIP_ITEMS, "robots": fam.GRIP_ROBOTS, "moves": fam.GRIP_MOVES},
            "tower": {"boxes": fam.TOWER_BOXES, "contents": fam.TOWER_CONTENTS, "mu": fam.TOWER_MU, "robots": fam.TOWER_ROBOTS},
        },
        "physics": {"timestep": ph.TIMESTEP, "integrator": "implicitfast", "cone": "elliptic", "impratio": 10, "solref": "0.004 1", "condim": 3,
                    "tilt": {"slide_m": ph.TILT_SLIDE_M, "tip_deg": ph.TILT_TIP_DEG, "hold_s": ph.TILT_HOLD, "slide_stop_m": ph.TILT_SLIDE_STOP},
                    "grip": {"slip_m": ph.GRIP_SLIP_M, "pad_half": ph.PAD_HALF, "pre_s": ph.GRIP_PRE, "post_s": ph.GRIP_POST},
                    "tower": {"fall_deg": ph.TOWER_FALL_DEG, "seconds": ph.TOWER_SECONDS}},
        "xml_examples": examples, "candidates": len(rows), "candidates_file": str(Path(args.candidates).relative_to(REPO)) if Path(args.candidates).is_relative_to(REPO) else str(args.candidates),
        "selection": selection, "targets_groups": targets, "pair_cap": {"still_pair_share": args.still_pair_share, "by_family": pair_cap}, "records": len(records), "questions": {"total": sum(q_types.values()), **dict(sorted(q_types.items()))},
        "file": str(out.relative_to(REPO)) if out.is_relative_to(REPO) else str(out), "sha256": file_sha256(out), "ids_sha256": ids_sha256(r["id"] for r in records),
        "stats_file": str(out.with_suffix(".stats.json").name), "cpu": {"worker_seconds_by_family": worker_seconds, "worker_seconds": sum(worker_seconds.values()), "runs": runs},
        "created": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
    }
    _write_json(out.with_suffix(".manifest.json"), manifest)
    _log(f"{len(records)} records ({manifest['questions']}) → {out} · sha256 {manifest['sha256'][:12]} · selection " + json.dumps({f: s['kept'] for f, s in selection.items()}))
    return 0


def _sanity_job(job: tuple[str, dict[str, Any]]) -> Any:
    from robo_jev.mjgen import sanity

    kind, payload = job
    return getattr(sanity, kind)(**payload)


def cmd_sanity(args: argparse.Namespace) -> int:
    import multiprocessing

    import mujoco

    from robo_jev.mjgen import sanity

    jobs = sanity.jobs(master=args.seed, k=args.k, moments=args.moments)
    _log(f"sanity: {len(jobs)} jobs on {args.workers} workers")
    with multiprocessing.get_context("fork").Pool(args.workers) as pool:
        results = pool.map(_sanity_job, jobs, chunksize=1)
    report = sanity.summarize(jobs, results)
    report.update({"mujoco": mujoco.__version__, "k": args.k, "master_seed": args.seed, "script_version": SCRIPT_VERSION, "git": _git_sha(),
                   "created": dt.datetime.now().astimezone().isoformat(timespec="seconds")})
    _write_json(Path(args.out), report)
    _log(f"sanity → {args.out}: " + json.dumps(report["verdict"]))
    return 0 if report["verdict"]["passed"] else 2


def cmd_overlap(args: argparse.Namespace) -> int:
    from robo_jev.decisions.records import read_records
    from robo_jev.mjgen.overlap import overlap_report

    report = overlap_report(read_records(args.input), read_records(args.t8b))
    report.update({"input": str(args.input), "t8b": str(args.t8b), "git": _git_sha(), "created": dt.datetime.now().astimezone().isoformat(timespec="seconds")})
    _write_json(Path(args.out), report)
    _log(f"overlap → {args.out}: " + json.dumps(report["verdict"]))
    return 0 if report["verdict"]["passed"] else 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    measure = sub.add_parser("measure", help="rollout cost per family (one process)")
    measure.add_argument("--groups", type=int, default=4)
    measure.add_argument("--k", type=int, default=32)
    measure.add_argument("--seed", type=int, default=MASTER_SEED)
    measure.add_argument("--pair-share", type=float, default=0.5)
    measure.add_argument("--out", type=Path, default=OUT / "mujoco-measure.json")
    measure.set_defaults(func=cmd_measure)
    label = sub.add_parser("label", help="label candidate groups in MuJoCo (CPU unit); appends, resumable")
    label.add_argument("--groups-per-family", type=int, required=True)
    label.add_argument("--start", type=int, default=0)
    label.add_argument("--families", nargs="+", default=None)
    label.add_argument("--k", type=int, default=32, help="hidden draws per prior setting (three settings)")
    label.add_argument("--seed", type=int, default=MASTER_SEED)
    label.add_argument("--pair-share", type=float, default=0.5)
    label.add_argument("--workers", type=int, default=8)
    label.add_argument("--candidates", type=Path, default=OUT / "mujoco-candidates.jsonl")
    label.set_defaults(func=cmd_label)
    assemble = sub.add_parser("assemble", help="balanced selection → records + manifest + stats")
    assemble.add_argument("--candidates", type=Path, default=OUT / "mujoco-candidates.jsonl")
    assemble.add_argument("--records-per-family", type=int, default=5000)
    assemble.add_argument("--still-pair-share", type=float, default=0.2, help="max share of near/free (designed-still) pairs kept")
    assemble.add_argument("--out", type=Path, default=OUT / "mujoco-train.jsonl")
    assemble.set_defaults(func=cmd_assemble)
    check = sub.add_parser("sanity", help="physics sanity report")
    check.add_argument("--k", type=int, default=16)
    check.add_argument("--moments", type=int, default=12)
    check.add_argument("--seed", type=int, default=MASTER_SEED)
    check.add_argument("--workers", type=int, default=8)
    check.add_argument("--out", type=Path, default=OUT / "mujoco-sanity.json")
    check.set_defaults(func=cmd_sanity)
    over = sub.add_parser("overlap", help="zero-overlap check against T8b (text only)")
    over.add_argument("--input", type=Path, default=OUT / "mujoco-train.jsonl")
    over.add_argument("--t8b", type=Path, default=REPO / "artifacts" / "t8" / "T8b.jsonl")
    over.add_argument("--out", type=Path, default=OUT / "mujoco-overlap.json")
    over.set_defaults(func=cmd_overlap)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
