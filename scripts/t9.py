#!/usr/bin/env python
"""Task T9 — the second outside physics test: MuJoCo-labelled robot moments written by a fresh author, sealed (no training).

    # candidates + MuJoCo labels (CPU unit, ≤ 10 workers; resumable: groups already in the file are skipped)
    python scripts/t9.py label --groups 90 --workers 10 --out artifacts/t9/T9.candidates.jsonl
    # selection → records + manifest (CPU, seconds)
    python scripts/t9.py build --candidates artifacts/t9/T9.candidates.jsonl --per-family 50 --out artifacts/t9/T9.jsonl
    # physics sanity (monotonicity, analytic checks, determinism) → artifacts/t9/T9-sanity.json
    python scripts/t9.py sanity --workers 10 --out artifacts/t9/T9-sanity.json
    # overlap with every training input and T8b → artifacts/t9/T9-overlap.json (exit 2 on a failed verdict)
    python scripts/t9.py overlap --input artifacts/t9/T9.jsonl --out artifacts/t9/T9-overlap.json
    # label statistics → artifacts/t9/T9-stats.json (+ .md)
    python scripts/t9.py stats --input artifacts/t9/T9.jsonl --out artifacts/t9/T9-stats.json
    # references: the textbook calculator (over the base draws, and at the prior means) and constant answers → predictions
    python scripts/t9.py reference --input artifacts/t9/T9.jsonl --out-dir artifacts/reports/t9
    # System-1 metrics (+ robust reading) of predictions → artifacts/t9/instrument.json (+ .md); aggregate only for models
    python scripts/t9.py score --input artifacts/t9/T9.jsonl --pred base=artifacts/reports/t9/T9-base-2b.predictions.jsonl … --out artifacts/t9/instrument.json

Rules (brief §2): no training; no API of any kind — T9 is never sent anywhere; MuJoCo work runs in capped CPU units; the untrained 2B is
scored only through ``scripts/diagnose_decisions.py`` in the one-GPU-unit wrapper; trained models are not scored on T9.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import multiprocessing as mp
import os
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

SCRIPT_VERSION = "t9-1"
MASTER_SEED = 20261013
OUT = REPO / "artifacts" / "t9"


def _log(message: str) -> None:
    print(f"[t9 {dt.datetime.now().strftime('%H:%M:%S')}] {message}", file=sys.stderr, flush=True)


def _git() -> dict[str, Any]:
    def run(*args: str) -> str | None:
        try:
            return subprocess.run(["git", "-C", str(REPO), *args], capture_output=True, text=True, check=True).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return None

    status = run("status", "--porcelain", "--", "src/robo_jev/t9", "scripts/t9.py")
    return {"commit": run("rev-parse", "--short", "HEAD"), "dirty": bool(status) if status is not None else None}


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(payload, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    temp.replace(path)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    out = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                out.append(json.loads(line))
    return out


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _limit_threads() -> None:
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(name, "1")


# --------------------------------------------------------------------------
# label
# --------------------------------------------------------------------------


def _label_job(job: tuple[str, int, int, int]) -> dict[str, Any]:
    from robo_jev.t9 import build as B
    from robo_jev.t9.families import FAMILIES

    family_name, index, master, k = job
    start = time.time()
    try:
        group = B.candidate(family_name, index, master, k=k)
    except RuntimeError as error:
        return {"family": family_name, "index": index, "failed": str(error)}
    group = B.refine(group)
    labelled = B.label_group(group)
    fam = FAMILIES[family_name]
    key = {m: sum(labelled["labels"][m]["plan"]["base"]["freq"][o] for o in fam.key) for m in ("a", "b")}
    return {**group, "labelled": labelled["labels"], "unstable": labelled["unstable"], "cpu_s": labelled["cpu_s"], "key_a": key["a"],
            "key_b": key["b"], "wall_s": round(time.time() - start, 2)}


def cmd_label(args: argparse.Namespace) -> int:
    _limit_threads()
    from robo_jev.t9.families import FAMILIES

    if args.workers > 10:
        raise SystemExit("at most 10 workers (brief §2)")
    families = args.families or list(FAMILIES)
    done: set[tuple[str, int]] = set()
    if args.out.is_file():
        for row in _read_jsonl(args.out):
            done.add((row["family"], int(row["index"])))
    jobs = [(f, i, args.master, args.k) for f in families for i in range(args.start, args.start + args.groups) if (f, i) not in done]
    _log(f"label: {len(jobs)} groups to do ({len(done)} already in {args.out}); workers {args.workers}; K = {args.k}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    start = time.time()
    with args.out.open("a", encoding="utf-8") as out, mp.get_context("fork").Pool(args.workers, maxtasksperchild=20) as pool:
        for n, row in enumerate(pool.imap_unordered(_label_job, jobs), 1):
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            out.flush()
            if n % 10 == 0 or n == len(jobs):
                _log(f"{n}/{len(jobs)} groups · {time.time() - start:.0f} s")
    return 0


# --------------------------------------------------------------------------
# build
# --------------------------------------------------------------------------


def cmd_build(args: argparse.Namespace) -> int:
    import mujoco

    from robo_jev.decisions.records import validate_record
    from robo_jev.t9 import build as B
    from robo_jev.t9 import physics as ph
    from robo_jev.t9 import priors as P
    from robo_jev.t9.families import FAMILIES

    rows = [r for r in _read_jsonl(args.candidates) if not r.get("failed")]
    failed = [r for r in _read_jsonl(args.candidates) if r.get("failed")]
    by_key = {(r["family"], int(r["index"])): r for r in rows}
    rows = list(by_key.values())
    chosen, report = B.select(rows, args.per_family)
    git = _git()
    records = []
    for group in sorted(chosen, key=lambda g: (list(FAMILIES).index(g["family"]), g["index"])):
        records += B.records_of(group, {"labels": group["labelled"]}, mujoco_version=mujoco.__version__, commit=git["commit"])
    problems = {r["id"]: validate_record(r) for r in records}
    problems = {k: v for k, v in problems.items() if v}
    if problems:
        raise SystemExit(f"{len(problems)} records fail the format check: {list(problems.items())[:3]}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    # one compiled MJCF per family (the first record's plan) for the manifest
    mjcf = {}
    for fam_name, fam in FAMILIES.items():
        first = next((r for r in records if r["meta"]["family"] == fam_name), None)
        if first is not None:
            mjcf[fam_name] = ph.make_scene(fam_name, fam.scene(first["meta"]["visible"])).xml_text
    catalogs = _catalogs()
    ids = sorted(r["id"] for r in records)
    composition = Counter(r["meta"]["family"] for r in records)
    questions = Counter((r["meta"]["family"], q["type"]) for r in records for q in r["request"]["questions"].values())
    manifest = {
        "kind": "t9-outside-test", "generator": B.GENERATOR, "script_version": SCRIPT_VERSION, "git": git, "mujoco": mujoco.__version__,
        "use": "SEALED outside physical-judgment test (gate for the next trained version) — never trained on, never sent to any API; until "
               "that gate: instrument sanity only (label statistics, physics sanity, calculator reference, untrained base in aggregate)",
        "master_seed": args.master, "k_per_prior": int(rows[0]["k"]) if rows else None, "priors": list(P.PRIORS),
        "prior_rule": {"base": "the catalog ranges behind the words", "low_reader": "every word range shifted down by a third of its width",
                       "high_reader": "every word range shifted up by a third of its width",
                       "clip": "each to the quantity's physical bounds", "bounds": P.BOUNDS,
                       "fixed": "stated tolerances (placement accuracy, 'about' masses) are read the same by every prior"},
        "families": {name: {"law": fam.law, "group": fam.group, "key_outcomes": list(fam.key), "outcomes": list(ph.OUTCOMES[name]),
                            "hidden": list(fam.hidden_names()), "decisive": fam.decisive, "alternative_action": fam.alt_field}
                     for name, fam in FAMILIES.items()},
        "catalogs": catalogs, "mjcf": mjcf,
        "physics": {"timestep": ph.TIMESTEP, "integrator": "implicitfast", "cone": "elliptic", "impratio": 10, "solref": ph.SOLREF,
                    "strike_speed": ph.STRIKE_SPEED, "topple_deg": ph.TOPPLE_DEG, "lip_rolling_resistance": ph.LIP_ROLL},
        "labels": {"yes/no and score": "share of the K base-prior draws with the key event", "direction": "one-hot: Δ ≥ +0.10 likelier, ≤ −0.10 less likely, else no_real_change",
                   "action": "probability 1 on the least expected cost (share × stated failure cost + added seconds; ties split)",
                   "outcome": "the plan's outcome shares"},
        "selection": {"rule": "per family: equal numbers per tenth of the base record's key label, short tenths filled from the nearest",
                      "per_family": args.per_family, "report": report, "candidates": len(rows), "failed_candidates": len(failed),
                      "unstable_groups": sum(1 for r in rows if r.get("unstable"))},
        "composition": {"records": len(records), "groups": len(chosen), "by_family": dict(composition),
                        "questions": {f"{f}:{t}": n for (f, t), n in sorted(questions.items())}},
        "cpu": {"labelled_groups": len(rows), "cpu_seconds_labelling": round(sum(float(r.get("cpu_s") or 0) for r in rows), 1),
                "wall_seconds_per_group_mean": round(sum(float(r.get("wall_s") or 0) for r in rows) / max(len(rows), 1), 2)},
        "file": {"path": str(args.out), "sha256": _sha256(args.out), "ids_sha256": hashlib.sha256("\n".join(ids).encode()).hexdigest()},
        "candidates": {"path": str(args.candidates), "sha256": _sha256(args.candidates)},
        "created": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
    }
    _write_json(args.out.with_name(args.out.stem + ".manifest.json"), manifest)
    _log(f"build: {len(records)} records ({dict(composition)}) → {args.out}")
    return 0


def _catalogs() -> dict[str, Any]:
    from robo_jev.t9 import families as F

    def plain(table: dict[str, Any]) -> dict[str, Any]:
        return {k: {kk: (list(vv) if isinstance(vv, tuple) else vv) for kk, vv in v.items()} for k, v in table.items()}

    return {"roll": {"items": plain(F.ROLL_ITEMS), "surfaces": plain(F.ROLL_SURFACES)},
            "swing": {"loads": plain(F.SWING_LOADS), "straps": plain(F.SWING_STRAPS), "obstacles": list(F.SWING_OBSTACLES)},
            "knock": {"objects": plain(F.KNOCK_OBJECTS), "surfaces": plain(F.KNOCK_SURFACES)},
            "conveyor": {"items": plain(F.CONVEYOR_ITEMS), "belts": plain(F.CONVEYOR_BELTS)},
            "overhang": {"boards": plain(F.OVERHANG_BOARDS), "items": plain(F.OVERHANG_ITEMS)},
            "suction": {"panels": plain(F.SUCTION_PANELS), "faces": plain(F.SUCTION_FACES), "pumps": plain(F.SUCTION_PUMPS)}}


# --------------------------------------------------------------------------
# sanity, overlap, stats, reference, score
# --------------------------------------------------------------------------


def cmd_sanity(args: argparse.Namespace) -> int:
    _limit_threads()
    from robo_jev.t9 import sanity

    if args.workers > 10:
        raise SystemExit("at most 10 workers (brief §2)")
    report = sanity.run_all(workers=args.workers, k=args.k, moments=args.moments, master=args.master)
    _write_json(args.out, report)
    _log(f"sanity: verdict {report['verdict']} → {args.out}")
    return 0 if report["verdict"]["passed"] else 2


TRAINING_INPUTS = {
    "engine-train-50k": REPO / "artifacts" / "simgen" / "v02" / "train-50k.jsonl",
    "engine-s1": REPO / "artifacts" / "simgen" / "v03" / "engine-s1.jsonl",
    "mujoco-train": REPO / "artifacts" / "simgen" / "v03" / "mujoco-train.jsonl",
    "narrated-train": REPO / "artifacts" / "simgen" / "v03" / "narrated-train.jsonl",
    "general-train": REPO / "artifacts" / "simgen" / "v02" / "general-train.jsonl",
    "G1-train-mix": REPO / "artifacts" / "simgen" / "probes" / "G1-train-mix.jsonl",
    "train-v03 (assembled)": REPO / "artifacts" / "simgen" / "v03" / "train-v03.jsonl",
}
T8B = REPO / "artifacts" / "t8" / "T8b.jsonl"


def cmd_overlap(args: argparse.Namespace) -> int:
    from robo_jev.t9 import overlap as O

    records = _read_jsonl(args.input)
    t9 = O.T9Units(records)
    names = O.catalog_names()
    reports = []
    for name, path in {"T8b": T8B, **TRAINING_INPUTS}.items():
        if not path.is_file():
            raise SystemExit(f"missing {name}: {path}")
        start = time.time()
        reports.append(O.compare_file(t9, name, path, grams=name in O.NGRAM_WRITTEN_LIKE_T9, names=names if name in O.NGRAM_WRITTEN_LIKE_T9 else ()))
        _log(f"overlap {name}: {reports[-1]['records']} records, shared {reports[-1]['shared']} ({time.time() - start:.0f} s)")
    out = {"kind": "t9-overlap", "script_version": SCRIPT_VERSION, "git": _git(), "input": {"path": str(args.input), "sha256": _sha256(args.input),
                                                                                           "records": len(records)},
           "t9_units": {"ids": len(t9.ids), "states": len(t9.states), "instruction_or_option_texts": len(t9.pieces), "sentences": len(t9.sentences),
                        **{f"{n}-grams": len(g) for n, g in t9.grams.items()}},
           "catalog_names": names, "reports": reports, "verdict": O.verdict(reports),
           "created": dt.datetime.now().astimezone().isoformat(timespec="seconds")}
    _write_json(args.out, out)
    _log(f"overlap verdict: {out['verdict']['passed']} → {args.out}")
    return 0 if out["verdict"]["passed"] else 2


def cmd_stats(args: argparse.Namespace) -> int:
    from robo_jev.t9 import stats

    records = _read_jsonl(args.input)
    report = stats.label_statistics(records)
    _write_json(args.out, report)
    args.out.with_suffix(".md").write_text(stats.render_markdown(report), encoding="utf-8")
    _log(f"stats → {args.out}")
    return 0


def cmd_reference(args: argparse.Namespace) -> int:
    from robo_jev.t9 import reference as R

    records = _read_jsonl(args.input)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    outputs = {"calculator": [R.calculator_prediction(r) for r in records], "calculator-means": [R.calculator_prediction(r, means=True) for r in records]}
    for name in R.CONSTANTS:
        outputs[f"const-{name}"] = [R.constant_prediction(r, name) for r in records]
    for name, rows in outputs.items():
        path = args.out_dir / f"T9-{name}.predictions.jsonl"
        with path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        _log(f"reference {name} → {path}")
    return 0


def cmd_score(args: argparse.Namespace) -> int:
    from robo_jev.t9 import score as S

    records = _read_jsonl(args.input)
    models = []
    for item in args.pred:
        tag, _, path = item.partition("=")
        models.append({"tag": tag, "path": Path(path)})
    report = S.instrument_report(records, models, n_boot=args.n_boot, seed=args.seed, per_family_tags=set(args.per_family_tags or []))
    report["input"] = {"path": str(args.input), "sha256": _sha256(args.input)}
    _write_json(args.out, report)
    args.out.with_suffix(".md").write_text(S.render_markdown(report), encoding="utf-8")
    _log(f"score → {args.out}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    lab = sub.add_parser("label", help="candidate groups + MuJoCo labels under three priors (CPU, resumable)")
    lab.add_argument("--families", nargs="*", default=None)
    lab.add_argument("--groups", type=int, default=90, help="candidate groups per family")
    lab.add_argument("--start", type=int, default=0)
    lab.add_argument("--master", type=int, default=MASTER_SEED)
    lab.add_argument("--k", type=int, default=40, help="hidden draws per prior")
    lab.add_argument("--workers", type=int, default=10)
    lab.add_argument("--out", type=Path, default=OUT / "T9.candidates.jsonl")
    lab.set_defaults(func=cmd_label)
    bld = sub.add_parser("build", help="balanced selection → T9.jsonl + manifest")
    bld.add_argument("--candidates", type=Path, default=OUT / "T9.candidates.jsonl")
    bld.add_argument("--per-family", type=int, default=50, help="groups (= pairs) per family")
    bld.add_argument("--master", type=int, default=MASTER_SEED)
    bld.add_argument("--out", type=Path, default=OUT / "T9.jsonl")
    bld.set_defaults(func=cmd_build)
    san = sub.add_parser("sanity", help="physics sanity → JSON")
    san.add_argument("--workers", type=int, default=10)
    san.add_argument("--k", type=int, default=16)
    san.add_argument("--moments", type=int, default=8)
    san.add_argument("--master", type=int, default=MASTER_SEED + 1)
    san.add_argument("--out", type=Path, default=OUT / "T9-sanity.json")
    san.set_defaults(func=cmd_sanity)
    ovl = sub.add_parser("overlap", help="overlap with every training input and T8b → JSON (exit 2 on a failed verdict)")
    ovl.add_argument("--input", type=Path, default=OUT / "T9.jsonl")
    ovl.add_argument("--out", type=Path, default=OUT / "T9-overlap.json")
    ovl.set_defaults(func=cmd_overlap)
    sts = sub.add_parser("stats", help="label statistics → JSON + markdown")
    sts.add_argument("--input", type=Path, default=OUT / "T9.jsonl")
    sts.add_argument("--out", type=Path, default=OUT / "T9-stats.json")
    sts.set_defaults(func=cmd_stats)
    ref = sub.add_parser("reference", help="calculator and constant-answer predictions")
    ref.add_argument("--input", type=Path, default=OUT / "T9.jsonl")
    ref.add_argument("--out-dir", type=Path, default=REPO / "artifacts" / "reports" / "t9")
    ref.set_defaults(func=cmd_reference)
    sco = sub.add_parser("score", help="System-1 metrics (+ robust reading) of predictions")
    sco.add_argument("--input", type=Path, default=OUT / "T9.jsonl")
    sco.add_argument("--pred", nargs="+", required=True, metavar="TAG=PATH")
    sco.add_argument("--per-family-tags", nargs="*", default=None, help="tags whose per-family values are reported (references only)")
    sco.add_argument("--n-boot", type=int, default=2000)
    sco.add_argument("--seed", type=int, default=0)
    sco.add_argument("--out", type=Path, default=OUT / "instrument.json")
    sco.set_defaults(func=cmd_score)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
