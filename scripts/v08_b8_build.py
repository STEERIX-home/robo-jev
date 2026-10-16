#!/usr/bin/env python
"""Task v0.8b — gate B8 (fresh bench-shaped requests, the v0.8 rule's P2) and the bench part of the Jev comparison set.

    # 1. robo-jev-web's own builders under Node (read-only clone; capped unit) → candidate rows
    artifacts/scratch/v08b/run_node.sh b8-build build_requests.mts raw-requests.jsonl 20261016
    # 2. split, convert and write B8.jsonl (0444) + artifacts/v08/jev/jev-bench.jsonl + b8.manifest.json (independence proofs)
    python scripts/v08_b8_build.py convert --raw artifacts/scratch/v08b/b8/raw-requests.jsonl --seed 20261016
    # 3. tag every option and freeze B8 BEFORE any scoring (tags.json + frozen-before-scoring.json, all 0444)
    python scripts/v08_b8_build.py tag-freeze
    # the Spark gate and the verdict: B8 unchanged since it was frozen (exit 1 otherwise)
    python scripts/v08_b8_build.py check

See :mod:`robo_jev.bench.b8_v08` for the design. ``convert`` and ``tag-freeze`` refuse to replace a frozen B8 unless
``--force-before-scoring`` is given — allowed only while no model has been scored on it.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from robo_jev.bench import b8_v08 as B  # noqa: E402
from robo_jev.bench.gates_v08 import B8_DIR, OPTION_RANGE_VERSION, OPTION_RANGE_VERSION_CONFIRMED  # noqa: E402


def _convert(args: argparse.Namespace) -> int:
    rows = B.read_jsonl(args.raw)
    seeds = {r["generator"]["seed"] for r in rows}
    if seeds != {args.seed}:
        raise SystemExit(f"the rows were generated with seed(s) {sorted(seeds)}, not {args.seed}")
    commits = {r["builder"]["commit"] for r in rows}
    if len(commits) != 1:
        raise SystemExit(f"the rows come from several robo-jev-web commits: {sorted(commits)}")
    parts = B.split_rows(rows, seed=args.seed)
    b8, jev, counters = [], [], {"b8": Counter(), "jev": Counter()}
    for part, out in (("b8", b8), ("jev", jev)):
        for row in parts[part]:
            counters[part][row["env"]] += 1
            out.append(B.to_record(row, part=part, index=counters[part][row["env"]]))
    out_dir = Path(args.out_dir)
    b8_path = out_dir / B.B8_FILE_NAME
    frozen = out_dir / B.FROZEN_NAME
    text = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in b8)
    if frozen.exists() and b8_path.exists() and b8_path.read_text(encoding="utf-8") != text and not args.force_before_scoring:
        raise SystemExit(f"{b8_path} is frozen and would change — pass --force-before-scoring (only while no model has been scored on it)")
    b8_sha = B.write_jsonl(b8_path, b8, readonly=True)
    jev_sha = B.write_jsonl(Path(args.jev), jev)
    rel07 = B.read_jsonl(B.REL07_REQUESTS) if B.REL07_REQUESTS.is_file() else []
    extra = {"rel071_tubes": B.read_jsonl(B.REL071_TUBES)} if B.REL071_TUBES.is_file() else {}
    manifest = {
        "kind": "v08-b8-manifest", "rule": "progress.md: PRE-DECLARED v0.8 VERDICT RULE — P2 on gate B8",
        "created": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "builder": {"repo": "STEERIX-home/robo-jev-web", "commit": commits.pop(), "script": "artifacts/scratch/v08b/b8/build_requests.mts",
                    "node": sorted({r["generator"].get("node") for r in rows}), "seed": args.seed,
                    "raw": str(args.raw), "raw_sha256": B.sha256_file(args.raw), "candidates": len(rows),
                    "candidates_by_env": dict(Counter(r["env"] for r in rows))},
        "split": {"rule": "per environment (café also per stratum: requests carrying the request question, then the rest) each template's "
                          "candidates shuffled with rng(v08b-b8-split, seed, env/stratum, template) and dealt round-robin over the templates; "
                          "the first fill B8, the next the Jev bench part, the rest are spares (never written)",
                  "targets": {env: {"b8": b, "jev": j} for env, (b, j) in B.SPLIT.items()}, "cafe_request_stratum": dict(zip(("b8", "jev"), B.CAFE_REQUEST_SPLIT)),
                  "spare": dict(Counter(r["env"] for r in parts["spare"]))},
        "option_range": {"version": OPTION_RANGE_VERSION, "confirmed": OPTION_RANGE_VERSION_CONFIRMED},
        "files": {"b8": {"path": str(b8_path.relative_to(REPO)) if b8_path.is_relative_to(REPO) else str(b8_path), "sha256": b8_sha, "records": len(b8)},
                  "jev_bench": {"path": str(Path(args.jev)), "sha256": jev_sha, "records": len(jev)}},
        "b8": B.describe(b8), "jev_bench": B.describe(jev),
        "independence": B.overlap_report(b8, jev, rel07, extra),
        "rel07_excluded": {"requests": len(rel07), "file": str(B.REL07_REQUESTS), "sha256": B.sha256_file(B.REL07_REQUESTS) if rel07 else None},
        "rules_added": [{"id": rid, "class": cls, "pattern": pat, "borderline": b} for rid, cls, pat, b in B.V08_RULES],
    }
    (out_dir / B.MANIFEST_NAME).write_text(json.dumps(manifest, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"B8 {len(b8)} records {dict(counters['b8'])} sha256 {b8_sha}; Jev bench {len(jev)} {dict(counters['jev'])} sha256 {jev_sha}; "
          f"spare {manifest['split']['spare']}; identical states {manifest['independence']['identical_states']}")
    return 0


def _tag_freeze(args: argparse.Namespace) -> int:
    directory = Path(args.dir)
    if (directory / B.FROZEN_NAME).exists() and not args.force_before_scoring:
        problems = B.check_frozen(directory)
        raise SystemExit(f"B8 is already frozen ({'unchanged' if not problems else '; '.join(problems)}) — pass --force-before-scoring "
                         f"to re-freeze (only while no model has been scored on it)")
    record = B.freeze(directory)
    tags = json.loads((directory / B.TAGS_NAME).read_text(encoding="utf-8"))
    manifest_path = directory / B.MANIFEST_NAME
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        rule_hits = Counter(o["rule"] for r in tags["requests"] for q in r["questions"].values() for o in q["options"])
        planner_tags = Counter((r["env"], q["planner_tag"]) for r in tags["requests"] for q in r["questions"].values() if q["planner"] is not None)
        manifest["frozen"] = {**record, "tag_counts": tags["counts"], "rule_hits": dict(rule_hits.most_common()),
                              "borderline_options": sum(o["borderline"] for r in tags["requests"] for q in r["questions"].values() for o in q["options"]),
                              "planner_tags": {f"{env}/{tag}": n for (env, tag), n in sorted(planner_tags.items())}}
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(record, indent=1))
    print(f"tag counts {tags['counts']}")
    return 0


def _check(args: argparse.Namespace) -> int:
    problems = B.check_frozen(Path(args.dir))
    if problems:
        print("B8 check FAILED:\n  " + "\n  ".join(problems))
        return 1
    frozen = json.loads((Path(args.dir) / B.FROZEN_NAME).read_text(encoding="utf-8"))
    print(f"B8 unchanged since frozen {frozen['frozen']} (B8 {frozen['b8_sha256'][:12]}…, tags {frozen['tags_sha256'][:12]}…)")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    convert = sub.add_parser("convert", help="split the builder's rows, write B8.jsonl, jev-bench.jsonl and b8.manifest.json")
    convert.add_argument("--raw", type=Path, required=True)
    convert.add_argument("--seed", type=int, required=True)
    convert.add_argument("--out-dir", type=Path, default=B8_DIR)
    convert.add_argument("--jev", type=Path, default=B.JEV_BENCH_FILE)
    convert.add_argument("--force-before-scoring", action="store_true")
    freeze = sub.add_parser("tag-freeze", help="tags.json + frozen-before-scoring.json, before any scoring")
    freeze.add_argument("--dir", type=Path, default=B8_DIR)
    freeze.add_argument("--force-before-scoring", action="store_true")
    check = sub.add_parser("check", help="B8 unchanged since it was frozen")
    check.add_argument("--dir", type=Path, default=B8_DIR)
    args = parser.parse_args(argv)
    return {"convert": _convert, "tag-freeze": _tag_freeze, "check": _check}[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
