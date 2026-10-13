#!/usr/bin/env python
"""Task v0.4a — MuJoCo-labelled training scenes: three laws × many scenes, for the v0.4 level-supervision test (no training here).

    # label candidate groups of some scenes (CPU; run in a capped systemd unit) — appends per scene, resumable
    python scripts/mjgen_v04.py label --scenes forks bin --groups 940 --workers 16
    # balanced selection + one-factor partners for the kept groups without a far number pair (CPU unit; resumable)
    python scripts/mjgen_v04.py factor --scenes forks bin --records-per-scene 1500 --workers 16
    # records + manifest + stats (the training file; the H-set has its own candidates, selection and file)
    python scripts/mjgen_v04.py assemble --scenes forks bin ... --out artifacts/simgen/v04/mujoco-scenes.jsonl
    python scripts/mjgen_v04.py label --scenes tray palms --groups 270 --seed 20261041 --start 500000 --root artifacts/simgen/v04/holdout
    python scripts/mjgen_v04.py factor --scenes tray palms --records-per-scene 400 --root artifacts/simgen/v04/holdout
    python scripts/mjgen_v04.py assemble --scenes tray palms --root artifacts/simgen/v04/holdout --out artifacts/simgen/v04/scene-holdout.jsonl \\
        --set-name v04a-scene-holdout --split test --id-prefix v04h
    # physics sanity, one-factor sweeps, prompt lengths, overlap and sealing checks
    python scripts/mjgen_v04.py sanity --workers 16
    python scripts/mjgen_v04.py sweep --workers 16
    python scripts/mjgen_v04.py tokens --input artifacts/simgen/v04/mujoco-scenes.jsonl
    python scripts/mjgen_v04.py overlap

Records are ``simgen-request/0``. Nothing here reads the text of the sealed sets except the overlap command, which compares hashes,
sentences and n-grams in code and prints counts only.
"""

from __future__ import annotations

import argparse
import collections
import dataclasses
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

SCRIPT_VERSION = "v04a-1"
OUT = REPO / "artifacts" / "simgen" / "v04"
MASTER_SEED = 20261040
HOLDOUT_SEED = 20261041
TRAIN_SET, TRAIN_PREFIX = "v04a-mujoco-scenes", "v04a"
HOLDOUT_SET, HOLDOUT_PREFIX = "v04a-scene-holdout", "v04h"


def _log(message: str) -> None:
    print(f"[v04a {dt.datetime.now().strftime('%H:%M:%S')}] {message}", file=sys.stderr, flush=True)


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


def _read_rows(path: Path) -> list[dict[str, Any]]:
    out = []
    if path.is_file():
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    try:
                        out.append(json.loads(line))
                    except json.JSONDecodeError:  # a line cut by a stopped unit
                        pass
    return out


def _guard_out(path: Path) -> None:
    """Never write into v0.3's, T8's, T9's or B1's artifacts."""
    resolved = Path(path).resolve()
    for forbidden in ("simgen/v03", "artifacts/t8", "artifacts/t9", "artifacts/b1"):
        if forbidden in str(resolved):
            raise SystemExit(f"refusing to write {resolved}: {forbidden} is read-only for this task")


def group_to_json(group: Any) -> dict[str, Any]:
    return json.loads(json.dumps(dataclasses.asdict(group), default=list))


def group_from_json(payload: dict[str, Any]) -> Any:
    from robo_jev.mjgen.families import Group

    return Group(**payload)


def _candidates_path(root: Path, scene: str) -> Path:
    return root / "candidates" / f"{scene}.jsonl"


def _partners_path(root: Path, scene: str) -> Path:
    return root / "partners" / f"{scene}.jsonl"


# --------------------------------------------------------------------------
# label
# --------------------------------------------------------------------------


def _label_job(job: tuple[str, int, int, int, float]) -> dict[str, Any]:
    from robo_jev.mjgen.scenes.build import candidate, label_group

    scene, index, master, k, pair_share = job
    started = time.perf_counter()
    group = candidate(scene, index, master, k=k, pair_share=pair_share)
    labelled = label_group(group, k=k)
    return {"group": group_to_json(group), "labelled": json.loads(json.dumps(labelled)), "seconds": round(time.perf_counter() - started, 3)}


def cmd_label(args: argparse.Namespace) -> int:
    import multiprocessing

    root = Path(args.root)
    _guard_out(root)
    jobs = []
    done_total = 0
    for scene in args.scenes:
        path = _candidates_path(root, scene)
        path.parent.mkdir(parents=True, exist_ok=True)
        done = {int(r["group"]["index"]) for r in _read_rows(path)}
        done_total += len(done)
        jobs += [(scene, index, args.seed, args.k, args.pair_share) for index in range(args.start, args.start + args.groups) if index not in done]
    # interleave scenes so a stopped unit leaves every scene partly done
    jobs.sort(key=lambda j: (j[1], j[0]))
    _log(f"labelling {len(jobs)} groups of {args.scenes} ({done_total} already done) with K={args.k} per prior on {args.workers} workers")
    handles = {scene: _candidates_path(root, scene).open("a", encoding="utf-8") for scene in args.scenes}
    started, cpu, count = time.perf_counter(), 0.0, 0
    try:
        with multiprocessing.get_context("fork").Pool(args.workers) as pool:
            for row in pool.imap_unordered(_label_job, jobs, chunksize=1):
                handle = handles[row["group"]["family"]]
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                handle.flush()
                cpu += row["seconds"]
                count += 1
                if count % 100 == 0:
                    wall = time.perf_counter() - started
                    _log(f"{count}/{len(jobs)} groups · wall {wall:.0f} s · worker cpu {cpu:.0f} s · ETA {wall / count * (len(jobs) - count):.0f} s")
    finally:
        for handle in handles.values():
            handle.close()
    wall = time.perf_counter() - started
    _log(f"labelled {count} groups in {wall:.0f} s wall, {cpu:.0f} s worker time")
    with (root / "label-runs.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"groups": count, "wall_s": round(wall, 1), "worker_s": round(cpu, 1), "workers": args.workers, "k": args.k,
                                 "seed": args.seed, "scenes": args.scenes, "start": args.start, "groups_per_scene": args.groups,
                                 "pair_share": args.pair_share, "git": _git_sha(),
                                 "finished": dt.datetime.now().astimezone().isoformat(timespec="seconds")}) + "\n")
    return 0


# --------------------------------------------------------------------------
# factor — balanced selection + one-factor partners
# --------------------------------------------------------------------------


def _factor_job(job: tuple[dict[str, Any], int]) -> dict[str, Any]:
    import random

    from robo_jev.mjgen.scenes.build import factor_pair, label_group

    payload, k = job
    started = time.perf_counter()
    group = dataclasses.replace(group_from_json(payload), pair=None)
    pair = factor_pair(group, random.Random(group.seed ^ 0xFAC7), k=k)
    row: dict[str, Any] = {"key": [group.family, group.index], "pair": pair}
    if pair is not None:
        labelled = label_group(dataclasses.replace(group, pair=pair), k=k, only=("pair",))
        row.update({"member": json.loads(json.dumps(labelled["members"]["pair"])), "rollouts": labelled["rollouts"], "unstable": labelled["flags"]["unstable"]})
    row["seconds"] = round(time.perf_counter() - started, 3)
    return row


def _load_candidates(root: Path, scenes: list[str]) -> list[tuple[Any, dict[str, Any], float]]:
    out = []
    for scene in scenes:
        for row in _read_rows(_candidates_path(root, scene)):
            out.append((group_from_json(row["group"]), row["labelled"], float(row["seconds"])))
    return out


def cmd_factor(args: argparse.Namespace) -> int:
    import multiprocessing

    from robo_jev.mjgen.scenes.build import select_balanced

    root = Path(args.root)
    _guard_out(root)
    items = _load_candidates(root, args.scenes)
    targets = {scene: int(math.ceil(args.records_per_scene / 2)) for scene in args.scenes}
    kept, selection = select_balanced([(g, lab) for g, lab, _ in items], targets)
    sel_path = root / "selection.json"
    previous = json.loads(sel_path.read_text(encoding="utf-8")) if sel_path.is_file() else {"targets_groups": {}, "selection": {}, "kept": []}
    keep_other = [key for key in previous.get("kept", []) if key[0] not in args.scenes]
    payload = {"targets_groups": {**{k: v for k, v in previous.get("targets_groups", {}).items() if k not in args.scenes}, **targets},
               "selection": {**{k: v for k, v in previous.get("selection", {}).items() if k not in args.scenes}, **selection},
               "kept": keep_other + [[g.family, g.index] for g, _ in kept], "git": _git_sha(),
               "updated": dt.datetime.now().astimezone().isoformat(timespec="seconds")}
    _write_json(sel_path, payload)
    jobs = []
    for scene in args.scenes:
        path = _partners_path(root, scene)
        path.parent.mkdir(parents=True, exist_ok=True)
        done = {tuple(r["key"]) for r in _read_rows(path)}
        jobs += [(group_to_json(g), args.k) for g, _ in kept
                 if g.family == scene and not (g.pair is not None and g.pair["closeness"] == "far") and (g.family, g.index) not in done]
    _log(f"kept {len(kept)} groups ({ {s: v['kept'] for s, v in selection.items()} }); labelling {len(jobs)} one-factor partners on {args.workers} workers")
    handles = {scene: _partners_path(root, scene).open("a", encoding="utf-8") for scene in args.scenes}
    started, cpu, count = time.perf_counter(), 0.0, 0
    try:
        with multiprocessing.get_context("fork").Pool(args.workers) as pool:
            for row in pool.imap_unordered(_factor_job, jobs, chunksize=1):
                handles[row["key"][0]].write(json.dumps(row, ensure_ascii=False) + "\n")
                handles[row["key"][0]].flush()
                cpu += row["seconds"]
                count += 1
                if count % 100 == 0:
                    wall = time.perf_counter() - started
                    _log(f"{count}/{len(jobs)} partners · wall {wall:.0f} s · worker cpu {cpu:.0f} s · ETA {wall / count * (len(jobs) - count):.0f} s")
    finally:
        for handle in handles.values():
            handle.close()
    wall = time.perf_counter() - started
    _log(f"labelled {count} partners in {wall:.0f} s wall, {cpu:.0f} s worker time")
    with (root / "factor-runs.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"partners": count, "wall_s": round(wall, 1), "worker_s": round(cpu, 1), "workers": args.workers, "k": args.k,
                                 "scenes": args.scenes, "git": _git_sha(), "finished": dt.datetime.now().astimezone().isoformat(timespec="seconds")}) + "\n")
    return 0


# --------------------------------------------------------------------------
# assemble
# --------------------------------------------------------------------------


def cmd_assemble(args: argparse.Namespace) -> int:
    import mujoco

    from robo_jev.bench.define import ids_sha256
    from robo_jev.decisions.records import file_sha256
    from robo_jev.mjgen import families as fam
    from robo_jev.mjgen.scenes import HOLDOUT_SCENES, LAWS, SCENES
    from robo_jev.mjgen.scenes.build import GENERATOR, group_records, stats

    root = Path(args.root)
    out = Path(args.out)
    _guard_out(out)
    is_holdout = args.split != "train"
    if not is_holdout and set(args.scenes) & set(HOLDOUT_SCENES):
        raise SystemExit(f"the held-out scenes {sorted(set(args.scenes) & set(HOLDOUT_SCENES))} never go into a training file")
    if is_holdout and set(args.scenes) - set(HOLDOUT_SCENES):
        raise SystemExit(f"the holdout file holds only {HOLDOUT_SCENES}")
    chosen = json.loads((root / "selection.json").read_text(encoding="utf-8"))
    keys = {tuple(k) for k in chosen["kept"] if k[0] in args.scenes}
    items = _load_candidates(root, args.scenes)
    partners = {}
    for scene in args.scenes:
        partners.update({tuple(r["key"]): r for r in _read_rows(_partners_path(root, scene))})
    kept, counts = [], collections.Counter()
    worker_seconds: dict[str, float] = collections.defaultdict(float)
    rollouts: dict[str, int] = collections.defaultdict(int)
    for group, labelled, seconds in items:
        worker_seconds[group.family] += seconds
        rollouts[group.family] += int(labelled["rollouts"])
        key = (group.family, group.index)
        if key not in keys:
            continue
        row = partners.get(key)
        if group.pair is not None and group.pair["closeness"] == "far":
            counts[f"{group.family}:far_number_pair"] += 1
        elif row is not None and row["pair"] is not None and not row["unstable"]:
            group = dataclasses.replace(group, pair=row["pair"])
            labelled = {**labelled, "members": {"base": labelled["members"]["base"], "pair": row["member"]}, "rollouts": labelled["rollouts"] + row["rollouts"]}
            counts[f"{group.family}:factor_pair"] += 1
        else:
            group = dataclasses.replace(group, pair=None)
            labelled = {**labelled, "members": {"base": labelled["members"]["base"]}}
            counts[f"{group.family}:unpaired"] += 1
        kept.append((group, labelled))
    for row in partners.values():
        worker_seconds[row["key"][0]] += float(row["seconds"])
        rollouts[row["key"][0]] += int(row.get("rollouts", 0))
    kept.sort(key=lambda gl: (args.scenes.index(gl[0].family), gl[0].index))
    git = _git_sha()
    records = []
    for group, labelled in kept:
        records.extend(group_records(group, labelled, mujoco_version=mujoco.__version__, git=git, set_name=args.set_name, split=args.split,
                                     id_prefix=args.id_prefix))
    out.parent.mkdir(parents=True, exist_ok=True)
    temp = out.with_name(out.name + ".tmp")
    with temp.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    temp.replace(out)
    summary = stats(records)
    _write_json(out.with_suffix(".stats.json"), summary)
    examples = {}
    for scene in args.scenes:
        first = next((g for g, _ in kept if g.family == scene), None)
        if first is not None:
            examples[scene] = SCENES[scene].xml_example(first.visible)
    runs = [json.loads(line) for line in (root / "label-runs.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()] if (root / "label-runs.jsonl").is_file() else []
    q_types = collections.Counter(q["type"] for r in records for q in r["request"]["questions"].values())
    manifest = {
        "set": args.set_name, "split": args.split, "id_prefix": args.id_prefix, "sealed": is_holdout,
        "what": ("internal scene holdout (H-set): two new scenes of two laws, never in a training file" if is_holdout
                 else "MuJoCo-labelled training scenes: three laws × new scenes (the v0.3b scenes tilt, tower, grip are added at assembly)"),
        "generator": GENERATOR, "script_version": SCRIPT_VERSION, "git": git, "mujoco": mujoco.__version__, "k_per_prior": fam.DEFAULT_K,
        "priors": fam.PRIOR_SETTINGS, "laws": LAWS, "scenes": {s: {"law": SCENES[s].law, "outcomes": SCENES[s].outcomes, "event": SCENES[s].event,
                                                                  "severe": SCENES[s].severe, "question_names": SCENES[s].question_names,
                                                                  "factors": {f: {k: v for k, v in e.items() if k != "span"} for f, e in SCENES[s].factors.items()},
                                                                  "catalogs": SCENES[s].catalogs()} for s in args.scenes},
        "holdout_scenes": list(HOLDOUT_SCENES), "holdout_rule": "chosen before any labelling: one tip-or-slide and one friction-holding scene whose "
                                                                "robots appear in no training scene; never written to a training file",
        "label_runs": [r for r in runs if set(r.get("scenes", [])) & set(args.scenes)],
        "group_seed_rule": "sha256('v04a|master|scene|index')[:8]; hidden-draw points: scrambled Latin hypercube seeded by group seed ^ 0xD3A5",
        "selection": {s: v for s, v in chosen["selection"].items() if s in args.scenes}, "pairs": dict(sorted(counts.items())),
        "xml_examples": examples, "records": len(records), "questions": {"total": sum(q_types.values()), **dict(sorted(q_types.items()))},
        "file": str(out.relative_to(REPO)) if out.is_relative_to(REPO) else str(out), "sha256": file_sha256(out),
        "ids_sha256": ids_sha256(r["id"] for r in records), "stats_file": out.with_suffix(".stats.json").name,
        "cpu": {"worker_seconds_by_scene": dict(worker_seconds), "worker_seconds": sum(worker_seconds.values()), "rollouts_by_scene": dict(rollouts),
                "rollouts": sum(rollouts.values())},
        "created": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
    }
    _write_json(out.with_suffix(".manifest.json"), manifest)
    _log(f"{len(records)} records ({manifest['questions']}) → {out} · sha256 {manifest['sha256'][:12]} · pairs {dict(counts)}")
    return 0


# --------------------------------------------------------------------------
# sanity and sweeps
# --------------------------------------------------------------------------


def _sanity_job(job: tuple[str, dict[str, Any]]) -> Any:
    from robo_jev.mjgen.scenes import sanity

    kind, payload = job
    return getattr(sanity, kind)(**payload)


def cmd_sanity(args: argparse.Namespace) -> int:
    import multiprocessing

    import mujoco

    from robo_jev.mjgen.scenes import sanity

    jobs = sanity.jobs(scenes=args.scenes, master=args.seed, k=args.k, moments=args.moments)
    _log(f"sanity: {len(jobs)} jobs on {args.workers} workers")
    with multiprocessing.get_context("fork").Pool(args.workers) as pool:
        results = pool.map(_sanity_job, jobs, chunksize=1)
    report = sanity.summarize(jobs, results)
    report.update({"mujoco": mujoco.__version__, "k": args.k, "master_seed": args.seed, "script_version": SCRIPT_VERSION, "git": _git_sha(),
                   "created": dt.datetime.now().astimezone().isoformat(timespec="seconds")})
    _guard_out(Path(args.out))
    _write_json(Path(args.out), report)
    _log(f"sanity → {args.out}: " + json.dumps(report["verdict"]))
    return 0 if report["verdict"]["passed"] else 2


def _sweep_job(job: tuple[str, int, int, int]) -> dict[str, Any]:
    from robo_jev.mjgen.scenes.sanity import factor_sweep

    return factor_sweep(*job)


def cmd_sweep(args: argparse.Namespace) -> int:
    import multiprocessing

    from robo_jev.mjgen.scenes.sanity import summarize_sweeps

    jobs = [(scene, 950_000 + i, args.seed, args.k) for scene in args.scenes for i in range(args.count)]
    with multiprocessing.get_context("fork").Pool(args.workers) as pool:
        results = pool.map(_sweep_job, jobs, chunksize=1)
    report = {"kind": "v04a-factor-sweep", "summary": summarize_sweeps(results), "scenes": results, "k": args.k, "seed": args.seed, "git": _git_sha(),
              "created": dt.datetime.now().astimezone().isoformat(timespec="seconds")}
    _guard_out(Path(args.out))
    _write_json(Path(args.out), report)
    _log(f"sweep → {args.out}: " + json.dumps({s: {f: round(v["range_mean"], 3) for f, v in d.items()} for s, d in report["summary"].items()}))
    return 0


# --------------------------------------------------------------------------
# prompt lengths
# --------------------------------------------------------------------------


def cmd_tokens(args: argparse.Namespace) -> int:
    """Prompt tokens per (record, question) example as the trainer builds them (Qwen3.5 tokenizer, chat format, direct mode)."""
    from robo_jev.decisions import examples as E
    from robo_jev.decisions.records import read_records
    from robo_jev.model.tokenizer import available_tokenizer, load_tokenizer

    tokenizer = load_tokenizer(available_tokenizer()[0])
    records = read_records(args.input)
    lengths: dict[str, list[int]] = collections.defaultdict(list)
    for record in records:
        for name in record["request"]["questions"]:
            (example,) = E.build_examples(record, [name], tokenizer, order_seed=E.FIXED_ORDER_SEED)
            lengths[record["meta"]["scene"]].append(len(example["tokens"]))
            lengths["all"].append(len(example["tokens"]))
    report = {"kind": "v04a-prompt-tokens", "input": str(args.input), "tokenizer": str(available_tokenizer()[0]),
              "how": "len(build_examples(record, [question], tokenizer, order_seed=FIXED_ORDER_SEED)[0]['tokens']) — one example per question, "
                     "chat format, direct mode (the trainer's prompts)",
              "by_scene": {k: {**E.length_summary(v), "p95": sorted(v)[int(0.95 * (len(v) - 1))]} for k, v in sorted(lengths.items())},
              "budget": {"median_max": 350, "max_max": 500}}
    allv = lengths["all"]
    report["verdict"] = {"median_ok": report["by_scene"]["all"]["p50"] <= 350, "max_ok": max(allv) <= 500}
    report["verdict"]["passed"] = all(report["verdict"].values())
    out = Path(args.out) if args.out else Path(args.input).with_suffix(".tokens.json")
    _guard_out(out)
    _write_json(out, report)
    _log(f"tokens → {out}: " + json.dumps({k: (v["p50"], v["p95"], v["max"]) for k, v in report["by_scene"].items()}) + " " + json.dumps(report["verdict"]))
    return 0 if report["verdict"]["passed"] else 2


# --------------------------------------------------------------------------
# overlap and sealing
# --------------------------------------------------------------------------

TESTS = REPO / "artifacts" / "simgen" / "v0b" / "tests"
PROBES = REPO / "artifacts" / "simgen" / "probes"
#: every sealed or held-out evaluation set the new records must not touch (the v0.3 rule's readings, the outside tests, B1)
SEALED_SETS = {**{f"T{i}": TESTS / f"T{i}.jsonl" for i in range(1, 8)}, "G1": PROBES / "G1.jsonl", "G2": PROBES / "G2.jsonl",
               "T8a": REPO / "artifacts" / "t8" / "T8a.jsonl", "T8b": REPO / "artifacts" / "t8" / "T8b.jsonl", "T9": REPO / "artifacts" / "t9" / "T9.jsonl",
               "B1": REPO / "artifacts" / "b1" / "b1-infamily.jsonl"}  # fmt: skip
#: sets whose word n-grams are judged (written outside our templates, in the same physical domain)
NGRAM_SETS = ("T8b", "T9")
V03_TRAIN = REPO / "artifacts" / "simgen" / "v03" / "mujoco-train.jsonl"


def _units(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    from robo_jev.t9 import overlap as O

    return [O.units_of(r) for r in records]


def _against(mine: list[dict[str, Any]], theirs: Any, grams: dict[int, set] | None) -> dict[str, Any]:
    """Counts of our units found in another set's units (T9Units-shaped): ids, whole requests, whole states, instruction / option texts,
    exact and number-normalised sentences, and (when given) word n-grams. Counts only — no shared text is kept."""
    sentences = set().union(*(u["sentences"] for u in mine)) if mine else set()
    templates = set().union(*(u["templates"] for u in mine)) if mine else set()
    out = {"ids": sum(u["id"] in theirs.ids for u in mine), "requests": sum(u["request"] in theirs.requests for u in mine),
           "states": sum(u["state"] in theirs.states for u in mine), "instruction_or_option_texts": len(set().union(*(u["pieces"] for u in mine)) & theirs.pieces),
           "sentences": len(sentences & theirs.sentences), "sentences_number_normalised": len(templates & theirs.templates)}
    if grams is not None:
        out["ngrams"] = {str(n): len(grams[n] & theirs.grams[n]) for n in theirs.sizes}
    return out


def _where(mine_records: list[dict[str, Any]], bad_units: set[str], kind: str) -> list[str]:
    """Our record ids and question names whose sentences / pieces are in `bad_units` (to fix our templates; our text only)."""
    from robo_jev.t9 import overlap as O

    out = []
    for r in mine_records:
        u = O.units_of(r)
        if u[kind] & bad_units:
            out.append(r["id"])
        if len(out) >= 10:
            break
    return out


def cmd_overlap(args: argparse.Namespace) -> int:
    from robo_jev.data.general_sets import record_units
    from robo_jev.data.sealed import sealed_record_set
    from robo_jev.decisions.records import file_sha256, read_records
    from robo_jev.mjgen.overlap import _catalog_names as v03_catalog_names
    from robo_jev.mjgen.scenes import HOLDOUT_SCENES, SCENES
    from robo_jev.simgen.holdout import request_hash
    from robo_jev.simgen.narration import SealedIndex, state_sha256
    from robo_jev.t9 import overlap as O

    ours = {name: Path(path) for name, path in (("train", args.train), ("holdout", args.holdout)) if path is not None and Path(path).is_file()}
    if not ours:
        raise SystemExit("nothing to check")
    records = {name: read_records(path) for name, path in ours.items()}
    units = {name: _units(rows) for name, rows in records.items()}
    grams: dict[str, dict[int, set]] = {}
    for name, rows in units.items():
        grams[name] = {n: set() for n in (4, 5, 6)}
        for u in rows:
            for text in u["texts"]:
                for n in (4, 5, 6):
                    grams[name][n] |= O.ngrams(text, n)
    report: dict[str, Any] = {"kind": "v04a-overlap", "script_version": SCRIPT_VERSION, "git": _git_sha(),
                              "inputs": {name: {"path": str(path), "sha256": file_sha256(path), "records": len(records[name])} for name, path in ours.items()},
                              "against_sealed": {}, "checks": {}}
    missing = [name for name, path in SEALED_SETS.items() if not path.is_file()]
    if missing:
        raise SystemExit(f"sealed sets missing: {missing}")
    for set_name, path in SEALED_SETS.items():
        theirs = O.T9Units(read_records(path), sizes=(4, 5, 6) if set_name in NGRAM_SETS else ())
        entry = {"records": len(theirs.ids), "sha256": file_sha256(path)}
        for name in ours:
            entry[name] = _against(units[name], theirs, grams[name] if set_name in NGRAM_SETS else None)
            hits = entry[name]
            total = sum(v for k, v in hits.items() if k != "ngrams") + (sum(v for n, v in hits.get("ngrams", {}).items() if int(n) >= 5))
            if total:
                bad_sent = set().union(*(u["sentences"] for u in units[name])) & theirs.sentences
                bad_tmpl = set().union(*(u["templates"] for u in units[name])) & theirs.templates
                bad_piece = set().union(*(u["pieces"] for u in units[name])) & theirs.pieces
                entry[f"{name}_our_records_hit"] = {"sentences": _where(records[name], bad_sent, "sentences"),
                                                   "sentences_number_normalised": _where(records[name], bad_tmpl, "templates"),
                                                   "instruction_or_option_texts": _where(records[name], bad_piece, "pieces")}
                if set_name in NGRAM_SETS:
                    shared5 = grams[name][5] & theirs.grams[5]
                    entry[f"{name}_our_texts_with_shared_5grams"] = [
                        r["id"] for r in records[name] if any(O.ngrams(t, 5) & shared5 for t in O.units_of(r)["texts"])][:10]
        report["against_sealed"][set_name] = entry
        _log(f"overlap vs {set_name}: " + json.dumps({n: entry[n] for n in ours}))
    # the unified v0.3 check's text units (normalised state, state lines and option texts of ≥ 5 words) of the sets written outside the engine
    index = SealedIndex.build({k: v for k, v in SEALED_SETS.items() if k != "B1"})
    unified: dict[str, Any] = {}
    for name, rows in records.items():
        tally: collections.Counter = collections.Counter()
        for r in rows:
            short = {h[:16] for h in (__import__("hashlib").sha256(u.encode("utf-8")).hexdigest() for u in record_units(r))}
            rh, sh = request_hash(r), state_sha256(r["request"]["state"])
            for set_name in index.requests:
                tally[f"{set_name}:request"] += rh in index.requests[set_name]
                tally[f"{set_name}:state"] += sh in index.states[set_name]
                if index.units.get(set_name):
                    tally[f"{set_name}:units"] += len(short & index.units[set_name])
            marked = sealed_record_set(r)
            tally["marked_as_another_sealed_set"] += marked is not None and not (name == "holdout" and marked == HOLDOUT_SET)
        unified[name] = {k: v for k, v in sorted(tally.items()) if v}
    report["unified_v03_check"] = {"what": "SealedIndex (scripts/system1_v03.py overlap): whole-request and whole-state sha256 of every sealed set, "
                                           "text units of G1, G2, T8a, T8b, T9; robo_jev.data.sealed marks", "hits": unified}
    # catalog names: ours vs T9's and T8b's (exact names, and their names found as phrases inside ours)
    mine = set().union(*(SCENES[s].catalog_names() for s in SCENES))
    t9_names = {n.lower() for n in O.catalog_names()}
    t8b_names = {n.lower() for n in v03_catalog_names()[1]}
    import re as _re

    def phrase_hits(their: set[str], *, multi: bool) -> list[str]:
        """Our names that contain one of theirs as a phrase (their names of ≥ 2 words, or — for information — of one word)."""
        chosen = {t for t in their if (len(t.split()) >= 2) == multi}
        return sorted(m for m in mine if m not in their and any(_re.search(r"\b" + _re.escape(t) + r"\b", m) for t in chosen))

    report["catalog_names"] = {"ours": len(mine), "t9": len(t9_names), "t8b": len(t8b_names), "shared_with_t9": len(mine & t9_names),
                               "shared_with_t8b": len(mine & t8b_names), "our_names_containing_a_t9_name": phrase_hits(t9_names, multi=True),
                               "our_names_containing_a_t8b_name": phrase_hits(t8b_names, multi=True),
                               "information_our_names_containing_a_one_word_t9_name": phrase_hits(t9_names, multi=False),
                               "information_our_names_containing_a_one_word_t8b_name": phrase_hits(t8b_names, multi=False),
                               "rule": "exact names disjoint, and no multi-word T9 or T8b name inside ours (v0.3b's rule was exact names only); "
                                       "one-word hits (generic words such as a container type) are reported, not judged"}
    # the H-set against every training record (the new scenes and v0.3b's three families, which v0.4 trains on)
    if "holdout" in ours:
        train_rows = (records.get("train") or []) + read_records(V03_TRAIN)
        theirs = O.T9Units(train_rows, sizes=())
        hold = _against(units["holdout"], theirs, None)
        train_catalogs = set().union(*(SCENES[s].catalog_names() for s in SCENES if s not in HOLDOUT_SCENES)) | v03_catalog_names()[0]
        hold_catalogs = set().union(*(SCENES[s].catalog_names() for s in HOLDOUT_SCENES))
        hold["scenes_shared"] = sorted({r["meta"]["scene"] for r in records["holdout"]} & {r["meta"]["scene"] for r in train_rows})
        hold["group_seeds_shared"] = len({r["generator"]["seed"] for r in records["holdout"]} & {r["generator"]["seed"] for r in train_rows})
        hold["catalog_names_shared"] = sorted(hold_catalogs & train_catalogs)
        report["holdout_vs_train"] = {"train_records": len(train_rows), "train_files": [str(ours.get("train")), str(V03_TRAIN)], **hold}
        _log("holdout vs train: " + json.dumps(hold))
    checks = report["checks"]
    for set_name, entry in report["against_sealed"].items():
        for name in ours:
            hits = entry[name]
            checks[f"{name} vs {set_name}: ids, hashes, texts"] = hits["ids"] == 0 and hits["requests"] == 0 and hits["states"] == 0 and hits["instruction_or_option_texts"] == 0
            checks[f"{name} vs {set_name}: sentences"] = hits["sentences"] == 0 and hits["sentences_number_normalised"] == 0
            if set_name in NGRAM_SETS:
                checks[f"{name} vs {set_name}: no shared 5- or 6-grams"] = all(v == 0 for n, v in hits["ngrams"].items() if int(n) >= 5)
    for name, hits in unified.items():
        checks[f"{name}: unified v0.3 check (hashes, text units, sealed marks)"] = not hits
    checks["catalog names disjoint from T9 and T8b"] = not (report["catalog_names"]["shared_with_t9"] or report["catalog_names"]["shared_with_t8b"]
                                                            or report["catalog_names"]["our_names_containing_a_t9_name"]
                                                            or report["catalog_names"]["our_names_containing_a_t8b_name"])
    if "holdout_vs_train" in report:
        h = report["holdout_vs_train"]
        checks["holdout vs train: nothing shared"] = (not any(h[k] for k in ("ids", "requests", "states", "instruction_or_option_texts", "sentences",
                                                                            "sentences_number_normalised", "group_seeds_shared"))
                                                      and not h["scenes_shared"] and not h["catalog_names_shared"])
    report["verdict"] = {"passed": all(checks.values()), "failed": [k for k, v in checks.items() if not v]}
    report["created"] = dt.datetime.now().astimezone().isoformat(timespec="seconds")
    _guard_out(Path(args.out))
    _write_json(Path(args.out), report)
    _log(f"overlap → {args.out}: passed={report['verdict']['passed']} failed={report['verdict']['failed']}")
    return 0 if report["verdict"]["passed"] else 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    label = sub.add_parser("label", help="label candidate groups in MuJoCo (CPU unit); appends per scene, resumable")
    label.add_argument("--scenes", nargs="+", required=True)
    label.add_argument("--groups", type=int, required=True, help="candidate groups per scene")
    label.add_argument("--start", type=int, default=0)
    label.add_argument("--k", type=int, default=32)
    label.add_argument("--seed", type=int, default=MASTER_SEED)
    label.add_argument("--pair-share", type=float, default=0.33)
    label.add_argument("--workers", type=int, default=16)
    label.add_argument("--root", type=Path, default=OUT / "train")
    label.set_defaults(func=cmd_label)
    factor = sub.add_parser("factor", help="balanced selection + one-factor partners (CPU unit; resumable)")
    factor.add_argument("--scenes", nargs="+", required=True)
    factor.add_argument("--records-per-scene", type=int, default=1500)
    factor.add_argument("--k", type=int, default=32)
    factor.add_argument("--workers", type=int, default=16)
    factor.add_argument("--root", type=Path, default=OUT / "train")
    factor.set_defaults(func=cmd_factor)
    assemble = sub.add_parser("assemble", help="selected groups (+ partners) → records + manifest + stats")
    assemble.add_argument("--scenes", nargs="+", required=True)
    assemble.add_argument("--root", type=Path, default=OUT / "train")
    assemble.add_argument("--out", type=Path, default=OUT / "mujoco-scenes.jsonl")
    assemble.add_argument("--set-name", default=TRAIN_SET)
    assemble.add_argument("--split", default="train")
    assemble.add_argument("--id-prefix", default=TRAIN_PREFIX)
    assemble.set_defaults(func=cmd_assemble)
    check = sub.add_parser("sanity", help="physics sanity report")
    check.add_argument("--scenes", nargs="+", default=None)
    check.add_argument("--k", type=int, default=16)
    check.add_argument("--moments", type=int, default=8)
    check.add_argument("--seed", type=int, default=MASTER_SEED)
    check.add_argument("--workers", type=int, default=16)
    check.add_argument("--out", type=Path, default=OUT / "mujoco-scenes-sanity.json")
    check.set_defaults(func=cmd_sanity)
    sweep = sub.add_parser("sweep", help="one-factor sweeps: label sensitivity to each physical factor")
    sweep.add_argument("--scenes", nargs="+", required=True)
    sweep.add_argument("--count", type=int, default=16, help="candidate scenes per scene type")
    sweep.add_argument("--k", type=int, default=16)
    sweep.add_argument("--seed", type=int, default=MASTER_SEED)
    sweep.add_argument("--workers", type=int, default=16)
    sweep.add_argument("--out", type=Path, default=OUT / "mujoco-scenes-factor-sweep.json")
    sweep.set_defaults(func=cmd_sweep)
    tokens = sub.add_parser("tokens", help="prompt lengths (Qwen3.5 tokenizer, the trainer's examples)")
    tokens.add_argument("--input", type=Path, required=True)
    tokens.add_argument("--out", type=Path, default=None)
    tokens.set_defaults(func=cmd_tokens)
    over = sub.add_parser("overlap", help="zero-overlap and sealing checks (counts only; sealed texts are read in code, never printed)")
    over.add_argument("--train", type=Path, default=OUT / "mujoco-scenes.jsonl")
    over.add_argument("--holdout", type=Path, default=OUT / "scene-holdout.jsonl")
    over.add_argument("--out", type=Path, default=OUT / "mujoco-scenes-overlap.json")
    over.set_defaults(func=cmd_overlap)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
