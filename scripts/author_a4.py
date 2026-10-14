#!/usr/bin/env python
"""Author A4 (v0.6 held-out test author) — label, assemble and check the sealed field-robot test set.

    PY=/home/rodex_dgx/Desktop/projects/robo-jev/.venv/bin/python
    systemd-run --user --wait --collect -p MemoryMax=6G -E PYTHONPATH=$PWD/src choom -n 1000 -- $PY scripts/author_a4.py label --groups 160 --workers 4
    $PY scripts/author_a4.py assemble --per-scene 135          # → artifacts/v06/a4/test.jsonl (+ manifest, stats)
    $PY scripts/author_a4.py retext                            # re-render the request text only (labels, ids, order, meta unchanged)
    $PY scripts/author_a4.py validate                          # loader, in-family readings, the v0.5 call view
    $PY scripts/author_a4.py tokens                            # prompt lengths (Qwen3.5-2B tokenizer, one example per question)
    systemd-run ... $PY scripts/author_a4.py sanity --workers 4  # monotonicity sweeps, analytic thresholds, determinism
    systemd-run ... $PY scripts/author_a4.py overlap           # against T1–T7, G1, G2, T8a, T8b, T9, B1, the H-set, train-v04

``label`` is incremental and resumable: one JSON per group under ``artifacts/v06/a4/work/<scene>/``; existing files are skipped. Every
output goes under ``artifacts/v06/a4/``. Nothing here trains, uses a GPU or calls any API; the set is sealed (never trained on, never
sent anywhere — :data:`robo_jev.data.sealed.SEALED_RECORD_SETS`).
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

OUT = REPO / "artifacts" / "v06" / "a4"
WORK = OUT / "work"
TEST = OUT / "test.jsonl"
TOKENIZER = Path("/home/rodex_dgx/Desktop/projects/robo-jev/artifacts/scratch/v0c/hf/Qwen3.5-2B/tokenizer.json")


def _log(message: str) -> None:
    print(f"[author-a4 {dt.datetime.now().strftime('%H:%M:%S')}] {message}", file=sys.stderr, flush=True)


def _git() -> str | None:
    try:
        return subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True).stdout.strip() or None
    except (OSError, subprocess.CalledProcessError):
        return None


def refuse_write(path: Path) -> None:
    """Outputs go under artifacts/v06/a4/ only."""
    resolved = Path(path).resolve()
    if OUT.resolve() not in resolved.parents and resolved != OUT.resolve():
        raise SystemExit(f"{path}: author A4 writes only under {OUT}")


def _write_json(path: Path, data: Any) -> None:
    refuse_write(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=1, ensure_ascii=False, default=float) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


# --------------------------------------------------------------------------
# label
# --------------------------------------------------------------------------


def _label_one(task: tuple[str, int, int, str]) -> dict[str, Any]:
    from robo_jev.authors.a4 import label as L

    scene, index, seed, out = task
    start = time.process_time()
    group = L.build_group(scene, index, seed)
    if group is None:
        result = {"scene": scene, "index": index, "seed": seed, "group": None}
    else:
        sim = L.simulate_group(group)
        result = {"scene": scene, "index": index, "seed": seed, "group": group, "sim": sim}
    result["cpu_seconds"] = time.process_time() - start
    _write_json(Path(out), result)
    return {"scene": scene, "index": index, "cpu": result["cpu_seconds"], "unstable": (result.get("sim") or {}).get("unstable")}


def cmd_label(args: argparse.Namespace) -> int:
    from multiprocessing import get_context

    from robo_jev.authors.a4.design import SCENES

    scenes = SCENES if args.scene == "all" else (args.scene,)
    tasks = []
    for scene in scenes:
        for index in range(args.start, args.start + args.groups):
            path = WORK / scene / f"{index:06d}.json"
            if not path.is_file():
                tasks.append((scene, index, args.seed, str(path)))
    _log(f"{len(tasks)} groups to label ({args.workers} workers); already done are skipped")
    t0, done, cpu = time.time(), 0, 0.0
    with get_context("fork").Pool(args.workers) as pool:
        for info in pool.imap_unordered(_label_one, tasks):
            done += 1
            cpu += info["cpu"]
            if done % 20 == 0 or done == len(tasks):
                _log(f"{done}/{len(tasks)} groups, wall {time.time() - t0:.0f} s, cpu {cpu:.0f} s (last {info['scene']}:{info['index']} unstable {info['unstable']})")
    return 0


# --------------------------------------------------------------------------
# assemble
# --------------------------------------------------------------------------


def _work_files() -> list[dict[str, Any]]:
    out = []
    for path in sorted(WORK.glob("*/*.json")):
        out.append(json.loads(path.read_text(encoding="utf-8")))
    return out


def cmd_assemble(args: argparse.Namespace) -> int:
    import numpy as np

    from robo_jev.authors.a4 import build as B
    from robo_jev.authors.a4 import design as D
    from robo_jev.authors.a4 import label as L
    from robo_jev.decisions.records import validate_record

    refuse_write(args.out)
    work = [w for w in _work_files() if w.get("group")]
    candidates = []
    for w in work:
        scene = w["scene"]
        draws = w["sim"]["draws"][0]["planned"]["base"]
        key = float(np.mean([o in D.EVENT[scene] for o in draws]))
        candidates.append({"scene": scene, "index": w["index"], "key": key, "unstable": bool(w["sim"]["unstable"]), "work": w})
    picked = B.select_groups(candidates, args.per_scene, seed=args.seed)
    git = _git()
    records = []
    for c in picked:
        records += B.group_records(c["work"]["group"], c["work"]["sim"], git=git)
    problems = [(r["id"], p) for r in records for p in validate_record(r)]
    if problems:
        raise SystemExit(f"{len(problems)} format problems, first {problems[:3]}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as handle:
        for r in records:
            handle.write(json.dumps(r, ensure_ascii=False, separators=(",", ":")) + "\n")
    stats = composition(records)
    stats["candidates"] = {s: {"labelled": sum(1 for c in candidates if c["scene"] == s), "unstable": sum(1 for c in candidates if c["scene"] == s and c["unstable"]),
                               "failed_to_build": sum(1 for w in _work_files() if w["scene"] == s and not w.get("group"))} for s in D.SCENES}
    stats["cpu_seconds_labelling"] = sum(w.get("cpu_seconds", 0.0) for w in work)
    stats["rollouts"] = sum(int(w["sim"]["rollouts"]) for w in work)
    manifest = {"kind": "v06-a4-test", "set": B.SET_NAME, "split": B.SPLIT, "sealed": True, "never_trained": True, "never_sent_to_any_api": True,
                "file": str(args.out.relative_to(REPO)), "sha256": _sha256(args.out), "records": len(records), "groups": len(picked),
                "per_scene": {s: sum(1 for r in records if r["meta"]["scene"] == s) for s in D.SCENES}, "id_prefix": B.ID_PREFIX + "-",
                "generator": B.GENERATOR, "git": git, "seed": args.seed, "k": D.K, "priors": list(D.PRIORS), "created": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
                "selection": f"{args.per_scene} groups per scene from the labelled candidates, binned by the base record's MuJoCo key label (tenths), "
                             "bins filled round-robin in a seeded order; groups with an unstable rollout never picked",
                "key_events": {s: list(D.EVENT[s]) for s in D.SCENES}, "severe_events": {s: list(D.SEVERE[s]) for s in D.SCENES},
                "plan_key": dict(D.PLAN_KEY), "question_names": __import__("robo_jev.authors.a4.radio", fromlist=["QUESTION_NAMES"]).QUESTION_NAMES,
                "label_rules": L.__doc__.split("**Labels**")[1].strip() if "**Labels**" in (L.__doc__ or "") else None}
    _write_json(OUT / "manifest.json", manifest)
    _write_json(OUT / "stats.json", stats)
    _log(f"wrote {len(records)} records ({manifest['per_scene']}) to {args.out} sha256 {manifest['sha256'][:12]}")
    print(json.dumps({k: stats[k] for k in ("records", "per_scene")}, indent=1))
    return 0


def composition(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Records, questions, band profile, robust share by question kind, pairs (factor share, moved), key histogram per scene."""
    from collections import Counter

    from robo_jev.authors.a4 import label as L

    out: dict[str, Any] = {"records": len(records), "per_scene": {}}
    for scene in sorted({r["meta"]["scene"] for r in records}):
        rs = [r for r in records if r["meta"]["scene"] == scene]
        kinds = rs[0]["meta"]["question_kind"]
        entry: dict[str, Any] = {"records": len(rs), "groups": len({r["generator"]["group"] for r in rs}), "questions": sum(len(r["request"]["questions"]) for r in rs)}
        bands: dict[str, Counter] = {}
        robust: dict[str, list[bool]] = {}
        for r in rs:
            for name, kind in r["meta"]["question_kind"].items():
                lab = r["labels"][name]
                robust.setdefault(kind, []).append(bool(r["meta"]["robustness"][name]["robust"]))
                if lab["type"] in ("noul", "score"):
                    bands.setdefault(kind, Counter())[L.band(float(lab.get("p_yes", lab.get("value"))))] += 1
        entry["bands"] = {k: {b: v[b] for b in ("low", "middle", "high")} for k, v in bands.items()}
        entry["robust_share"] = {k: round(sum(v) / len(v), 3) for k, v in robust.items()}
        key = kinds and next(q for q, k in kinds.items() if k == "key")
        entry["key_histogram_tenths"] = [sum(1 for r in rs if min(9, int(float(r["labels"][key]["p_yes"]) * 10)) == i) for i in range(10)]
        entry["key_strictly_inside_0_1"] = round(sum(1 for r in rs if 0 < float(r["labels"][key]["p_yes"]) < 1) / len(rs), 3)
        base = [r for r in rs if r["meta"]["pair"]["role"] == "base"]
        entry["pairs"] = {"pairs": len(base), "factor_pairs": sum(1 for r in base if r["meta"]["pair"]["factor"]),
                          "far_number_pairs": sum(1 for r in base if not r["meta"]["pair"]["factor"]),
                          "factor_share": round(sum(1 for r in base if r["meta"]["pair"]["factor"]) / max(1, len(base)), 3),
                          "factor_fields": dict(Counter(r["meta"]["pair"]["field"] for r in base if r["meta"]["pair"]["factor"])),
                          "key_moved_ge_0.1": sum(1 for r in base if abs(r["meta"]["pair"]["key_delta"]) >= 0.1 - 1e-9),
                          "no_reversal": sum(1 for r in base if r["meta"]["pair"]["no_reversal"]), "direction_same": sum(1 for r in base if r["meta"]["pair"]["direction_same"])}
        entry["cf_kind"] = dict(Counter(r["meta"]["plan"]["cf_kind"] for r in base))
        entry["compare_option"] = dict(Counter(r["meta"]["compare_option"] for r in rs))
        entry["domains"] = dict(Counter(r["meta"]["domain"] for r in base))
        entry["argmax"] = {k: dict(Counter(max(r["labels"][q]["probabilities"], key=r["labels"][q]["probabilities"].get) for r in rs))
                           for q, k in kinds.items() if rs[0]["request"]["questions"][q]["type"] == "choice"}
        out["per_scene"][scene] = entry
    return out


# --------------------------------------------------------------------------
# retext
# --------------------------------------------------------------------------


def _line(record: dict[str, Any]) -> str:
    return json.dumps(record, ensure_ascii=False, separators=(",", ":"))


def _strings_of(node: Any, path: str = "") -> dict[str, str]:
    """Every string leaf of a JSON value by its path."""
    if isinstance(node, str):
        return {path: node}
    out: dict[str, str] = {}
    if isinstance(node, dict):
        for k, v in node.items():
            out.update(_strings_of(v, f"{path}/{k}"))
    elif isinstance(node, list):
        for i, v in enumerate(node):
            out.update(_strings_of(v, f"{path}/{i}"))
    return out


def retext_records(records: list[dict[str, Any]], groups: dict[tuple[str, int], dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Each record with its ``request`` re-rendered from its stored group by the current :mod:`robo_jev.authors.a4.radio` — nothing else
    is touched. Returns the new records and the changes: {(question path, old text, new text): records}. Refuses a change of the
    request's structure (question names, types, option keys)."""
    from collections import Counter

    from robo_jev.authors.a4 import radio as R

    out, changes = [], Counter()
    for record in records:
        gen = record["generator"]
        group = groups[(str(gen["family"]), int(gen["group"]))]
        member = 0 if gen["member"] == "base" else 1
        state, questions = R.render(group, member)
        new = json.loads(json.dumps(record))
        new["request"] = {"state": state, "questions": questions}
        old_strings, new_strings = _strings_of(record["request"]), _strings_of(new["request"])
        if set(old_strings) != set(new_strings) or [q["type"] for q in record["request"]["questions"].values()] != [q["type"] for q in questions.values()]:
            raise SystemExit(f"{record['id']}: the re-rendered request changes its structure — refusing")
        for path, text in old_strings.items():
            if new_strings[path] != text:
                changes[(path, text, new_strings[path])] += 1
        out.append(new)
    return out, {"changes": [{"path": path, "from": a, "to": b, "records": n} for (path, a, b), n in sorted(changes.items())]}


def cmd_retext(args: argparse.Namespace) -> int:
    from robo_jev.decisions.records import validate_record

    refuse_write(args.input)
    raw = args.input.read_text(encoding="utf-8")
    records = [json.loads(line) for line in raw.splitlines() if line.strip()]
    if "".join(_line(r) + "\n" for r in records) != raw:
        raise SystemExit(f"{args.input}: does not re-serialise byte for byte — refusing to rewrite it")
    groups = {}
    for w in _work_files():
        if w.get("group"):
            groups[(w["scene"], int(w["index"]))] = w["group"]
    new, report = retext_records(records, groups)
    allowed = set(args.allow or [])
    unexpected = [c for c in report["changes"] if c["path"].split("/")[-1] not in allowed]
    if unexpected or not report["changes"]:
        raise SystemExit(f"text changes outside the allowed option keys {sorted(allowed)}: {len(unexpected)} (or nothing changed) — refusing")
    for a, b in zip(records, new):
        if {k: v for k, v in a.items() if k != "request"} != {k: v for k, v in b.items() if k != "request"} or validate_record(b):
            raise SystemExit(f"{a['id']}: something besides the request text changed, or the record no longer validates — refusing")
    old_sha = _sha256(args.input)
    tmp = args.input.with_suffix(".jsonl.tmp")
    tmp.write_text("".join(_line(r) + "\n" for r in new), encoding="utf-8")
    os.replace(tmp, args.input)
    sha = _sha256(args.input)
    git = _git()
    manifest = json.loads((OUT / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("sha256") != old_sha:
        raise SystemExit("the manifest does not describe the file that was rewritten")
    revision = {"git": git, "created": dt.datetime.now().astimezone().isoformat(timespec="seconds"), "previous_sha256": old_sha, "sha256": sha,
                "what": "request text re-rendered from the stored groups; labels, ids, order, meta and generator unchanged",
                "changes": [{"question_option": "/".join(c["path"].split("/")[-3:]), "from": c["from"], "to": c["to"], "records": c["records"]}
                            for c in report["changes"]]}
    manifest["sha256"] = sha
    manifest["text_revisions"] = [*(manifest.get("text_revisions") or []), revision]
    _write_json(OUT / "manifest.json", manifest)
    stats = json.loads((OUT / "stats.json").read_text(encoding="utf-8"))
    stats["file_sha256"] = sha
    stats["text_revisions"] = manifest["text_revisions"]
    _write_json(OUT / "stats.json", stats)
    _log(f"rewrote the request text of {len(new)} records: {sum(c['records'] for c in report['changes'])} option texts changed; sha256 {old_sha[:12]} -> {sha[:12]}")
    print(json.dumps(revision, indent=1, ensure_ascii=False))
    return 0


# --------------------------------------------------------------------------
# tokens
# --------------------------------------------------------------------------


def cmd_tokens(args: argparse.Namespace) -> int:
    from tokenizers import Tokenizer

    from robo_jev.decisions.examples import length_summary
    from robo_jev.decisions.prompt import build_prompts
    from robo_jev.decisions.records import read_records, shuffle_key

    tok = Tokenizer.from_file(str(args.tokenizer))
    records = read_records(args.input)
    per_scene: dict[str, list[int]] = {}
    state_lens: list[int] = []
    for r in records:
        prompts = build_prompts(r["request"], tok, key=shuffle_key(r), seed=1, fmt="chat", mode="direct", orders="seeded")
        per_scene.setdefault(r["meta"]["scene"], []).extend(len(p["tokens"]) for p in prompts)
        state_lens.append(len(tok.encode(r["request"]["state"], add_special_tokens=False).ids))
    every = [x for v in per_scene.values() for x in v]
    report = {"tokenizer": str(args.tokenizer), "tokenizer_sha256": _sha256(Path(args.tokenizer)), "counting": "one example per question as the trainer "
              "builds it (decisions.prompt.build_prompts, chat format, direct, seeded order): system text + state + that question",
              "all": length_summary(every), "per_scene": {s: length_summary(v) for s, v in per_scene.items()}, "state_only": length_summary(state_lens),
              "limits": {"median_le": 350, "max_le": 500}, "pass": length_summary(every)["p50"] <= 350 and length_summary(every)["max"] <= 500}
    _write_json(OUT / "tokens.json", report)
    print(json.dumps({"all": report["all"], "per_scene": {s: {k: v[k] for k in ("p50", "p90", "max")} for s, v in report["per_scene"].items()},
                      "pass": report["pass"]}, indent=1))
    return 0 if report["pass"] else 1


# --------------------------------------------------------------------------
# validate / sanity / overlap — robo_jev.authors.a4.checks
# --------------------------------------------------------------------------


def cmd_validate(args: argparse.Namespace) -> int:
    from robo_jev.authors.a4 import checks

    report = checks.validate(args.input, REPO)
    _write_json(OUT / "validate.json", report)
    print(json.dumps(report["summary"], indent=1))
    return 0 if report["summary"]["pass"] else 1


def cmd_sanity(args: argparse.Namespace) -> int:
    from robo_jev.authors.a4 import checks

    report = checks.sanity(workers=args.workers, input_path=args.input)
    _write_json(OUT / "sanity.json", report)
    (OUT / "sanity.md").write_text(checks.render_sanity(report), encoding="utf-8")
    print(json.dumps(report["summary"], indent=1))
    return 0 if report["summary"]["pass"] else 1


def cmd_overlap(args: argparse.Namespace) -> int:
    from robo_jev.authors.a4 import checks

    report = checks.overlap(args.input, REPO)
    _write_json(OUT / "overlap.json", report)
    print(json.dumps(report["summary"], indent=1))
    return 0 if report["summary"]["pass"] else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("label", help="build and MuJoCo-label groups (resumable)")
    p.add_argument("--scene", default="all", choices=("all", "bank", "ledge", "anchor"))
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--groups", type=int, default=160, help="groups per scene")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--workers", type=int, default=4)
    p.set_defaults(func=cmd_label)
    p = sub.add_parser("assemble", help="select groups and write test.jsonl + manifest + stats")
    p.add_argument("--per-scene", type=int, default=135)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, default=TEST)
    p.set_defaults(func=cmd_assemble)
    p = sub.add_parser("retext", help="re-render the request text of test.jsonl from the stored groups (no relabelling)")
    p.add_argument("--input", type=Path, default=TEST)
    p.add_argument("--allow", nargs="+", default=["the_plan"], help="option keys whose text may change (anything else refuses)")
    p.set_defaults(func=cmd_retext)
    p = sub.add_parser("tokens", help="prompt lengths")
    p.add_argument("--input", type=Path, default=TEST)
    p.add_argument("--tokenizer", type=Path, default=TOKENIZER)
    p.set_defaults(func=cmd_tokens)
    p = sub.add_parser("validate", help="loader + in-family readings + call view")
    p.add_argument("--input", type=Path, default=TEST)
    p.set_defaults(func=cmd_validate)
    p = sub.add_parser("sanity", help="monotonicity sweeps, analytic thresholds, determinism")
    p.add_argument("--input", type=Path, default=TEST)
    p.add_argument("--workers", type=int, default=4)
    p.set_defaults(func=cmd_sanity)
    p = sub.add_parser("overlap", help="overlap against the sealed and training sets")
    p.add_argument("--input", type=Path, default=TEST)
    p.set_defaults(func=cmd_overlap)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
