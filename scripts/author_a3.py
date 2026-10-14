#!/usr/bin/env python
"""Author A3 (v0.6 author diversity): an independent MuJoCo-labelled generator — laboratory, hospital and light manufacturing, written as
numbered work orders. One scene per law: ``indexdial`` (A, tip or slide under acceleration), ``mayostand`` (B, centre of mass past the
support edge), ``holddown`` (C, friction-limited holding). See :mod:`robo_jev.authors.a3`.

    python scripts/author_a3.py plan --scene indexdial --groups 625
    python scripts/author_a3.py label --scene indexdial --workers 4      # MuJoCo: run inside a capped systemd-run unit
    python scripts/author_a3.py assemble                                   # → artifacts/v06/a3/train.jsonl, manifest.json, stats.json
    python scripts/author_a3.py sanity --workers 4                          # MuJoCo → sanity.json / sanity.md
    python scripts/author_a3.py overlap                                     # → overlap.json (counts and A3's own shared fragments only)
    python scripts/author_a3.py validate                                    # loader, in-family readers, v0.5 call view → validation.json
    python scripts/author_a3.py calibrate --scene indexdial                 # MuJoCo: re-measure the aim factors

MuJoCo runs only in capped units (at most 4 workers), e.g.::

    systemd-run --user --unit a3-label-indexdial -p MemoryMax=6G -p MemorySwapMax=0 -p CPUQuota=400% \\
        --setenv=PYTHONPATH=$PWD/src --working-directory=$PWD -- choom -n 1000 -- .venv/bin/python scripts/author_a3.py label ...

Labelling is incremental: one line per finished group in ``labels/<scene>.jsonl``; a restart skips finished groups.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from robo_jev.authors.a3 import build  # noqa: E402
from robo_jev.authors.a3.scenes import SCENES  # noqa: E402

OUT = REPO / "artifacts" / "v06" / "a3"
SCENE_ORDER = ("indexdial", "mayostand", "holddown")


def _git() -> str | None:
    try:
        return subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _guard(path: Path) -> Path:
    resolved = path.resolve()
    if OUT.resolve() not in (resolved, *resolved.parents):
        raise SystemExit(f"refusing to write outside {OUT}: {path}")
    return path


def cmd_plan(args: argparse.Namespace) -> None:
    t0 = time.perf_counter()
    if args.workers > 4:
        raise SystemExit("at most 4 workers")
    plans = build.plan_scene(args.scene, args.groups, workers=args.workers)
    n = build.write_jsonl(_guard(OUT / "plan" / f"{args.scene}.jsonl"), plans)
    designs = {d: sum(1 for p in plans if p["design"] == d) for d in ("one_factor", "far_number")}
    print(json.dumps({"scene": args.scene, "groups": n, "designs": designs, "seconds": round(time.perf_counter() - t0, 1)}))


def cmd_label(args: argparse.Namespace) -> None:
    if args.workers > 4:
        raise SystemExit("at most 4 workers (four authors label at the same time)")
    plans = build.read_jsonl(OUT / "plan" / f"{args.scene}.jsonl")
    if args.limit:
        plans = plans[: args.limit]
    result = build.label_scene(plans, _guard(OUT / "labels" / f"{args.scene}.jsonl"), workers=args.workers)
    print(json.dumps({"scene": args.scene, **result}))


def cmd_assemble(args: argparse.Namespace) -> None:
    from robo_jev.authors.a3 import report

    git = args.git or _git()
    records = []
    for scene in args.scenes:
        plans = {int(p["group"]): p for p in build.read_jsonl(OUT / "plan" / f"{scene}.jsonl")}
        labelled = {int(r["group"]): r for r in build.read_jsonl(OUT / "labels" / f"{scene}.jsonl")}
        missing = sorted(set(plans) - set(labelled))
        if missing and not args.allow_missing:
            raise SystemExit(f"{scene}: {len(missing)} groups not labelled yet (first {missing[:5]})")
        for g in sorted(set(plans) & set(labelled)):
            records += build.assemble_group(plans[g], labelled[g], git=git)
    path = _guard(OUT / args.out)
    build.write_jsonl(path, records)
    stats = report.composition(records)
    (OUT / "stats.json").write_text(json.dumps(stats, indent=1) + "\n", encoding="utf-8")
    manifest = report.manifest(records, path, git=git, scenes=args.scenes, labels_dir=OUT / "labels")
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({"records": len(records), "out": str(path.relative_to(REPO)), "sha256": manifest["sha256"]}))


def cmd_calibrate(args: argparse.Namespace) -> None:
    from robo_jev.authors.a3 import calibrate

    result = calibrate.calibrate_scene(args.scene, n=args.n, seed=args.seed)
    print(json.dumps({"scene": args.scene, **result}, indent=1))


def cmd_sanity(args: argparse.Namespace) -> None:
    from robo_jev.authors.a3 import checks

    result = checks.physics_sanity(workers=args.workers, quick=args.quick)
    (_guard(OUT / "sanity.json")).write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
    (_guard(OUT / "sanity.md")).write_text(checks.sanity_markdown(result), encoding="utf-8")
    print(json.dumps(result["summary"], indent=1))


def cmd_overlap(args: argparse.Namespace) -> None:
    from robo_jev.authors.a3 import checks

    result = checks.overlap_report(OUT / args.data, REPO)
    (_guard(OUT / "overlap.json")).write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(result["summary"], indent=1))


def cmd_validate(args: argparse.Namespace) -> None:
    from robo_jev.authors.a3 import checks

    result = checks.validate(OUT / args.data, REPO, sample=args.sample)
    (_guard(OUT / "validation.json")).write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(result["summary"], indent=1))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("plan")
    p.add_argument("--scene", choices=sorted(SCENES), required=True)
    p.add_argument("--groups", type=int, default=625)
    p.add_argument("--workers", type=int, default=1)
    p.set_defaults(func=cmd_plan)
    p = sub.add_parser("label")
    p.add_argument("--scene", choices=sorted(SCENES), required=True)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--limit", type=int, default=0)
    p.set_defaults(func=cmd_label)
    p = sub.add_parser("assemble")
    p.add_argument("--scenes", nargs="+", default=list(SCENE_ORDER))
    p.add_argument("--out", default="train.jsonl")
    p.add_argument("--git", default=None)
    p.add_argument("--allow-missing", action="store_true")
    p.set_defaults(func=cmd_assemble)
    p = sub.add_parser("calibrate")
    p.add_argument("--scene", choices=sorted(SCENES), required=True)
    p.add_argument("--n", type=int, default=40)
    p.add_argument("--seed", type=int, default=11)
    p.set_defaults(func=cmd_calibrate)
    p = sub.add_parser("sanity")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--quick", action="store_true")
    p.set_defaults(func=cmd_sanity)
    p = sub.add_parser("overlap")
    p.add_argument("--data", default="train.jsonl")
    p.set_defaults(func=cmd_overlap)
    p = sub.add_parser("validate")
    p.add_argument("--data", default="train.jsonl")
    p.add_argument("--sample", type=int, default=400)
    p.set_defaults(func=cmd_validate)
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
