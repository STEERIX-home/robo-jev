#!/usr/bin/env python
"""Author A1 (v0.6 author diversity) — warehouse and logistics MuJoCo generator: label, assemble, check.

    # MuJoCo only inside a capped unit, at most 4 workers (one per unit):
    systemd-run --user --unit=a1-label-w0 -p MemoryMax=2G --setenv=PYTHONPATH=<worktree>/src --working-directory=<worktree> \\
        choom -n 1000 -- <venv>/bin/python scripts/author_a1.py label --groups 720 --worker 0 --workers 4
    python scripts/author_a1.py assemble      # cached rollouts → artifacts/v06/a1/train.jsonl, manifest, stats
    python scripts/author_a1.py check         # loader, scorers, call view, prompt lengths → artifacts/v06/a1/validation.json
    python scripts/author_a1.py overlap       # ids, hashes, sentences, 5-grams vs the protected sets (counts only)
    python scripts/author_a1.py sanity        # (MuJoCo, capped unit) monotone sweeps, analytic thresholds, determinism

Everything is written under ``artifacts/v06/a1/`` only. See :mod:`robo_jev.authors.a1` for the scenes and labels.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

OUT = REPO / "artifacts" / "v06" / "a1"
LABELS = OUT / "labels"
SCENE_ORDER = ("mast", "dock", "ramp")


def _log(message: str) -> None:
    print(f"[a1 {time.strftime('%H:%M:%S')}] {message}", flush=True)


def cmd_label(args: argparse.Namespace) -> int:
    from robo_jev.authors.a1 import build

    tasks = [(scene, i) for i in range(int(args.start), int(args.groups)) for scene in (args.scenes or SCENE_ORDER)]
    mine = tasks[int(args.worker) :: int(args.workers)]
    _log(f"worker {args.worker}/{args.workers}: {len(mine)} groups")
    t0, done, rollouts = time.time(), 0, 0
    for scene, index in mine:
        path = build.group_path(LABELS, scene, index)
        if path.exists():
            continue
        result = build.label_group(LABELS, scene, index)
        done += 1
        rollouts += int(result["rollouts"])
        if done % 10 == 0:
            rate = (time.time() - t0) / done
            _log(f"{done} new groups, {rollouts} rollouts, {rate:.2f} s/group, last {scene}:{index}")
    _log(f"done: {done} new groups, {rollouts} rollouts in {time.time() - t0:.0f} s")
    return 0


def _git() -> str | None:
    import subprocess

    try:
        return subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True).stdout.strip() or None
    except (OSError, subprocess.CalledProcessError):
        return None


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")


def cmd_assemble(args: argparse.Namespace) -> int:
    import mujoco

    from robo_jev.authors.a1 import assemble, build

    git = args.git or _git()
    records, summary = assemble.build_records(LABELS, per_scene=int(args.per_scene), git=git)
    digest = assemble.write_jsonl(OUT / "train.jsonl", records)
    stats = assemble.stats(records)
    _write_json(OUT / "stats.json", stats)
    manifest = {
        "set": build.SET, "split": "train", "author": build.AUTHOR, "domain": "warehouse and logistics", "style": "telemetry block",
        "generator": build.GENERATOR, "git": git, "mujoco": mujoco.__version__, "master_seed": build.MASTER_SEED, "k": build.K,
        "priors": ["base", "alt_a", "alt_b"], "label": "base prior frequencies (no clipping)",
        "files": {"train.jsonl": {"records": len(records), "sha256": digest}},
        "scenes": summary, "id_pattern": "v06a1-<scene>-<group:06d>[p]", "families": {"mast": "mastpick", "dock": "dockedge", "ramp": "tugramp"},
        "command": ("scripts/author_a1.py label --groups 720 (4 capped workers, all scenes); scripts/author_a1.py label --groups 900 --start 720 "
                    "--scenes mast ramp (3 capped workers: more candidates for the flat-key selection); scripts/author_a1.py assemble --per-scene 625"),
    }
    _write_json(OUT / "manifest.json", manifest)
    _log(f"wrote {len(records)} records ({digest[:12]}) and the manifest / stats")
    for scene, entry in summary.items():
        _log(f"  {scene}: {entry}")
    return 0


def _tokenizer():
    from robo_jev.model.tokenizer import available_tokenizer, load_tokenizer

    found = available_tokenizer()
    if not found:
        raise SystemExit("no tokenizer under artifacts/tokenizers")
    return load_tokenizer(found[0]), found[0]


def cmd_check(args: argparse.Namespace) -> int:
    from robo_jev.authors.a1 import checks
    from robo_jev.decisions.records import read_records

    path = OUT / "train.jsonl"
    records = read_records(path)
    tokenizer, tok_id = _tokenizer()
    result = {"file": str(path.relative_to(REPO)), "loader": checks.loader_check(path),
              "trainer_examples": checks.trainer_examples_check(records, tokenizer),
              "scorers": checks.scorer_check(records), "prompt_tokens": {"tokenizer": tok_id, **checks.prompt_lengths(records, tokenizer)}}
    _write_json(OUT / "validation.json", result)
    _log(json.dumps(result, indent=1)[:4000])
    return 0


def cmd_overlap(args: argparse.Namespace) -> int:
    from robo_jev.authors.a1 import checks
    from robo_jev.decisions.records import read_records

    records = read_records(OUT / "train.jsonl")
    # a positive control: 200 of A1's own records passed as if they were a protected set (every count must be > 0)
    result = checks.overlap(records, sets={**checks.PROTECTED, "control_a1_sample": records[:200]})
    _write_json(OUT / "overlap.json", result)
    for name, entry in result["sets"].items():
        _log(f"{name}: " + ", ".join(f"{k}={v}" for k, v in entry.items() if not isinstance(v, list)))
    return 0


def cmd_sanity(args: argparse.Namespace) -> int:
    from robo_jev.authors.a1 import sanity

    t0 = time.time()
    result = {"monotone": sanity.monotone_sweeps(n_plans=int(args.plans)), "analytic": sanity.analytic_thresholds(),
              "determinism": sanity.determinism(), "cached_rerun": sanity.cached_rerun(LABELS)}
    result["seconds"] = round(time.time() - t0, 1)
    _write_json(OUT / "sanity.json", result)
    for scene, sweeps in result["monotone"].items():
        _log(f"{scene}: " + "; ".join(f"{k}: {v['sweeps']} sweeps, {v['label_violations']} label violations, {v['draw_flips']} draw flips" for k, v in sweeps.items()))
    for name, rows in result["analytic"].items():
        _log(f"{name}: {rows}")
    _log(f"determinism {result['determinism']}, cached re-run {result['cached_rerun']}, {result['seconds']} s")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("label", help="run the MuJoCo rollouts of a slice of the groups (resumable)")
    p.add_argument("--groups", type=int, default=720, help="groups per scene (indices 0 .. groups-1)")
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--worker", type=int, default=0)
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--scenes", nargs="*", choices=SCENE_ORDER)
    p.set_defaults(func=cmd_label)
    p = sub.add_parser("assemble", help="cached rollouts -> train.jsonl, manifest.json, stats.json")
    p.add_argument("--per-scene", type=int, default=625)
    p.add_argument("--git", default=None)
    p.set_defaults(func=cmd_assemble)
    p = sub.add_parser("check", help="loader, trainer examples, scorers, call view, prompt lengths -> validation.json")
    p.set_defaults(func=cmd_check)
    p = sub.add_parser("sanity", help="(MuJoCo, capped unit) monotone sweeps, analytic thresholds, determinism -> sanity.json")
    p.add_argument("--plans", type=int, default=6)
    p.set_defaults(func=cmd_sanity)
    p = sub.add_parser("overlap", help="ids, hashes, sentences, 5-grams vs the protected sets -> overlap.json (counts only)")
    p.set_defaults(func=cmd_overlap)
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
