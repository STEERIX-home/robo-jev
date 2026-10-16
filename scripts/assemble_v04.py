#!/usr/bin/env python
"""Task v0.4b — the v0.4 training input: v0.3's input with exactly two data changes (the action repair and the MuJoCo recomposition).

    # the whole assembly, one command, once task v0.4a's scenes exist (CPU unit, ~15 min, ≤ 16 GB):
    python scripts/assemble_v04.py all --scenes artifacts/simgen/v04/mujoco-scenes.jsonl --holdout artifacts/simgen/v04/scene-holdout.jsonl \
        --out-dir artifacts/simgen/v04
    # before v0.4a lands: a stand-in for its two files (v0.3b records not taken by the subsample, renamed into 7 + 2 stand-in scenes),
    # written only under artifacts/scratch/v04b/, and the same pipeline on it
    python scripts/assemble_v04.py standin --out-dir artifacts/scratch/v04b/standin
    python scripts/assemble_v04.py all --scenes artifacts/scratch/v04b/standin/mujoco-scenes.jsonl \
        --holdout artifacts/scratch/v04b/standin/scene-holdout.jsonl --out-dir artifacts/scratch/v04b/assembled
    # the steps one by one: pick (engine layout), mujoco (recomposition), then convert_single_probe.py mix, system1_v03.py overlap, exposure

**(ii) Action repair** (``pick``). v0.3's derived System-1 questions are choice questions inside the engine records, so they shared the
trainer's choice slot (two questions of different types per record and epoch) with the original choice questions: per epoch the choice
slot stayed ≈ 32.7 k examples, and the action share of it fell from 0.649 to 0.259 (action 21,297 → 8,446). ``pick`` copies
``artifacts/simgen/v03/engine-s1.jsonl`` unchanged except one field per record, ``meta.question_pick`` = the derived questions as
*secondary* questions (:func:`robo_jev.simgen.system1_questions.with_pick_rule`): the trainer then picks each record's two examples from
its original questions exactly as for v02 train-50k (v0.1-mix / v0.2-b's pick) and trains every derived question **in addition** in the
epochs where v0.3's pick chooses it (:func:`robo_jev.decisions.examples.record_pick`). Opt-in by data — the trainer config and its
defaults are unchanged.

**(i) MuJoCo recomposition** (``mujoco``). v0.3b's ``mujoco-train.jsonl`` is subsampled to ``--per-family`` records (default 1,500) per
family (tilt, grip, tower): whole pair groups, stratified by the tenth of the base record's key label (equal groups per tenth — the flat
key-label profile v0.3b selected), seeded (``--seed``, recorded). v0.4a's training scenes follow, all of them. The H-set (v0.4a's scene
holdout) is checked to share no scene, id, request or state with them. The mix then holds the part at v0.3's MuJoCo examples per epoch
(``--mujoco-examples``, default 36,648) with a recorded second pass of whole pair groups round-robin over the scenes if needed.

**Everything else is v0.3's**: the d1 mix (byte for byte), the general part (exactly v0.3's 46,771 records — the file and its recorded
second pass of 11,771, byte-identical lines; ``exposure`` checks it against ``train-v03.jsonl``), the train config. The sealed check
covers T1–T7, G1, G2, T8a, T8b, T9 (registered), B1-infamily and the H-set.

**Totals.** Holding the MuJoCo and general parts at v0.3's *counts* while the engine part gains its original pick back (+18.6 k
examples) leaves their *shares* at ≈ 18.2 % and ≈ 22.7 % (v0.3: 20 % and 25 %): no layout can restore the action examples, keep the
derived exposure and the other engine exposures, and keep both the MuJoCo share and count (the report gives the arithmetic).
``--mujoco-share 0.2`` instead holds the share (≈ 41.3 k MuJoCo examples, +12.6 %) — recorded, not the default.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import random
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterator

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

SCRIPT_VERSION = "v04b-assemble-1"
CODE_PATHS = ("scripts/assemble_v04.py", "src/robo_jev/decisions/examples.py", "src/robo_jev/simgen/system1_questions.py",
              "scripts/convert_single_probe.py", "scripts/system1_v03.py")
V03 = REPO / "artifacts" / "simgen" / "v03"
ENGINE_S1 = V03 / "engine-s1.jsonl"
MUJOCO_V03 = V03 / "mujoco-train.jsonl"
TRAIN_V03 = V03 / "train-v03.jsonl"
PROBE = REPO / "artifacts" / "simgen" / "probes" / "G1-train-mix.jsonl"
G1 = REPO / "artifacts" / "simgen" / "probes" / "G1.jsonl"
GENERAL = REPO / "artifacts" / "simgen" / "v02" / "general-train.jsonl"
B1 = REPO / "artifacts" / "b1" / "b1-infamily.jsonl"
#: The mix's ``--sealed`` list (G1 first: the d1 lineage check; the rest: no shared id or whole state — T9 is registered and always
#: checked): T1–T7, G2, T8a, T8b, T9, B1-infamily, and the H-set (added per run).
SEALED_FOR_MIX = (G1, *(REPO / "artifacts" / "simgen" / "v0b" / "tests" / f"T{i}.jsonl" for i in range(1, 8)), REPO / "artifacts" / "simgen" / "probes" / "G2.jsonl",
                  REPO / "artifacts" / "t8" / "T8a.jsonl", REPO / "artifacts" / "t8" / "T8b.jsonl", REPO / "artifacts" / "t9" / "T9.jsonl", B1)
TRAIN_CONFIG = REPO / "configs" / "train" / "decisions-v0-lora-h100.yaml"
#: v0.3's input, the reference of every count (train-v03.manifest.json; engine-s1.manifest.json exposure, seed 0).
V03_COUNTS = {"mujoco_examples": 36648, "general_records": 46771, "general_examples": 45808, "probe_examples": 2724, "engine_examples": 98050,
              "steps": 5726, "action": 21297, "derived": 18551, "band": 15356, "comparison": 3195, "total_examples": 183230}
#: The brief's tolerances: action ≥ v0.1-mix's 21,297 per epoch; derived exposure ±10 %; steps ≤ 1.25 × 5,726; MuJoCo examples ±5 %.
TOLERANCE = {"derived": 0.10, "steps": 1.25, "mujoco": 0.05}
SUBSAMPLE_SEED = 2026101304
SUBSAMPLE_FAMILIES = ("tilt", "grip", "tower")
#: Places never written to.
NEVER_WRITE = (V03, REPO / "artifacts" / "t8", REPO / "artifacts" / "t9", REPO / "artifacts" / "b1", REPO / "artifacts" / "simgen" / "v0b" / "tests",
               REPO / "artifacts" / "simgen" / "probes")
STANDIN_SET, STANDIN_HOLDOUT_SET = "v04-standin-mujoco-scenes", "v04-standin-scene-holdout"


def _log(message: str) -> None:
    print(f"[v04b {dt.datetime.now().strftime('%H:%M:%S')}] {message}", file=sys.stderr, flush=True)


def _git() -> dict[str, Any]:
    def run(*args: str) -> str | None:
        try:
            return subprocess.run(["git", "-C", str(REPO), *args], capture_output=True, text=True, check=True).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return None

    status = run("status", "--porcelain", "--", *CODE_PATHS)
    return {"commit": run("rev-parse", "--short", "HEAD"), "dirty": bool(status) if status is not None else None}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(payload, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    temp.replace(path)


def _raw_lines(path: Path) -> Iterator[bytes]:
    with Path(path).open("rb") as handle:
        for raw in handle:
            if raw.strip():
                yield raw if raw.endswith(b"\n") else raw + b"\n"


def refuse_write(path: Path) -> None:
    """Outputs never go to v0.3's files, the sealed sets or B1."""
    resolved = Path(path).resolve()
    for place in NEVER_WRITE:
        place = place.resolve()
        if resolved == place or place in resolved.parents:
            raise SystemExit(f"{path}: refusing to write into {place}")


def _recorded_sha(path: Path) -> str | None:
    manifest = Path(path).with_name(Path(path).stem + ".manifest.json")
    if not manifest.is_file():
        return None
    data = json.loads(manifest.read_text(encoding="utf-8"))
    return data.get("sha256") or (data.get("file") or {}).get("sha256") or (data.get("out") or {}).get("sha256")


def _check_recorded(path: Path, *, required: bool) -> str:
    sha = _sha256(path)
    recorded = _recorded_sha(path)
    if recorded is None and required:
        raise SystemExit(f"{path}: no manifest recording its sha256 next to it")
    if recorded is not None and recorded != sha:
        raise SystemExit(f"{path}: its manifest records sha256 {recorded[:12]}… but the file is {sha[:12]}…")
    return sha


def _config(path: Path) -> dict[str, Any]:
    import yaml

    from robo_jev.decisions.train import resolve_config

    return resolve_config(yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {})


# --------------------------------------------------------------------------
# pick — the action repair as a data layout
# --------------------------------------------------------------------------


def cmd_pick(args: argparse.Namespace) -> int:
    from robo_jev.decisions.records import validate_record
    from robo_jev.simgen import system1_questions as s1
    from robo_jev.simgen.record import dumps

    refuse_write(args.out)
    sha = _check_recorded(args.input, required=args.input.resolve() == ENGINE_S1.resolve())
    counts: Counter = Counter()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temp = args.out.with_name(args.out.name + ".tmp")
    with temp.open("w", encoding="utf-8") as handle:
        for raw in _raw_lines(args.input):
            record = json.loads(raw)
            if dumps(record).encode("utf-8") + b"\n" != raw:
                raise SystemExit(f"{record.get('id')}: the input line is not in the engine's byte-stable form — refusing to re-serialise it")
            out = s1.with_pick_rule(record)
            back = json.loads(json.dumps(out))
            rule = back["meta"].pop("question_pick")
            if back != record or rule["secondary"] != list(record["meta"]["system1"]["derived"]) or list(out["meta"])[-1] != "question_pick":
                raise SystemExit(f"{record.get('id')}: the pick layout changed more than meta.question_pick")
            if validate_record(out):
                raise SystemExit(f"{record.get('id')}: fails the trainer's validation: {validate_record(out)[:3]}")
            counts["records"] += 1
            counts["secondary_questions"] += len(rule["secondary"])
            handle.write(dumps(out) + "\n")
    temp.replace(args.out)
    from robo_jev.decisions.examples import QUESTION_PICK_VERSION

    manifest = {"kind": "v04-engine-pick", "script_version": SCRIPT_VERSION, "git": _git(),
                "input": {"path": str(args.input), "sha256": sha, "records": counts["records"]},
                "out": {"path": str(args.out), "sha256": _sha256(args.out), "bytes": args.out.stat().st_size, "records": counts["records"]},
                "sha256": _sha256(args.out),
                "rule": {"field": "meta.question_pick", "version": QUESTION_PICK_VERSION, "secondary": "meta.system1.derived (every derived question)",
                         "trainer": "robo_jev.decisions.examples.record_pick — q examples from the original questions with the plain pick's stream "
                                    "(= v02 train-50k's pick); each derived question in addition when the plain pick over all questions (= v0.3's pick) "
                                    "chooses it"},
                "checks": {"unchanged_but_the_field": counts["records"], "byte_stable_input": True, "trainer_validation": "pass",
                           "secondary_questions": counts["secondary_questions"]},
                "created": dt.datetime.now().astimezone().isoformat(timespec="seconds")}  # fmt: skip
    _write_json(args.out.with_name(args.out.stem + ".manifest.json"), manifest)
    _log(f"pick: {counts['records']} records, {counts['secondary_questions']} secondary questions → {args.out}")
    return 0


# --------------------------------------------------------------------------
# mujoco — subsample v0.3b, add v0.4a's training scenes
# --------------------------------------------------------------------------


def _key_label(record: dict[str, Any]) -> float:
    meta = record.get("meta") or {}
    label = (record.get("labels") or {}).get(meta.get("key_question")) or {}
    value = label.get("p_yes", label.get("value"))
    if value is None:
        raise SystemExit(f"{record.get('id')}: no key label (meta.key_question {meta.get('key_question')!r})")
    return float(value)


def _base_of(group: list[dict[str, Any]]) -> dict[str, Any]:
    for record in group:
        if ((record.get("meta") or {}).get("pair") or {}).get("role") == "base":
            return record
    return group[0]


def _tenth(value: float) -> int:
    return min(9, int(value * 10))


def subsample(records: list[dict[str, Any]], *, per_family: int, seed: int, families: tuple[str, ...] = SUBSAMPLE_FAMILIES
              ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Whole pair groups of each family, equal numbers per tenth of the base record's key label (short tenths filled from the nearest
    tenths, lower first), seeded per (seed, family, tenth); the chosen records in file order, and the selection report."""
    from robo_jev.decisions.records import pair_groups

    groups_by_family: dict[str, list[list[dict[str, Any]]]] = defaultdict(list)
    for group in pair_groups(records):
        families_in = {str((r.get("meta") or {}).get("scene")) for r in group}
        if len(families_in) != 1:
            raise SystemExit(f"a pair group spans families {sorted(families_in)}: {[r['id'] for r in group]}")
        groups_by_family[families_in.pop()].append(group)
    unknown = sorted(set(groups_by_family) - set(families))
    if unknown:
        raise SystemExit(f"records of families {unknown} — the subsample takes only {list(families)}")
    chosen_ids: set[str] = set()
    report: dict[str, Any] = {}
    for family in families:
        groups = groups_by_family.get(family) or []
        target = max(1, round(per_family / 2))  # pairs → per_family records
        bins: dict[int, list[list[dict[str, Any]]]] = defaultdict(list)
        for group in groups:
            bins[_tenth(_key_label(_base_of(group)))].append(group)
        quota = {b: target // 10 + (1 if b < target % 10 else 0) for b in range(10)}
        taken: list[list[dict[str, Any]]] = []
        left_over: dict[int, list[list[dict[str, Any]]]] = {}
        for b in range(10):
            pool = list(bins[b])
            random.Random(f"{SCRIPT_VERSION}|{seed}|{family}|{b}").shuffle(pool)
            taken += pool[: quota[b]]
            left_over[b] = pool[quota[b]:]
        short = target - len(taken)
        if short > 0:
            spare = []
            for b in range(10):
                gap = min((abs(b - c) for c in range(10) if len(bins[c]) < quota[c]), default=0)
                spare += [(gap, b, i, g) for i, g in enumerate(left_over[b])]
            spare.sort(key=lambda t: (t[0], t[1], t[2]))
            taken += [g for *_, g in spare[:short]]
        for group in taken:
            chosen_ids |= {str(r["id"]) for r in group}
        kept = [g for g in groups if str(g[0]["id"]) in chosen_ids]
        report[family] = {
            "groups_available": len(groups), "records_available": sum(len(g) for g in groups), "target_groups": target,
            "groups_kept": len(kept), "records_kept": sum(len(g) for g in kept),
            "base_key_tenths_available": [len(bins[b]) for b in range(10)],
            "base_key_tenths_kept": [sum(1 for g in kept if _tenth(_key_label(_base_of(g))) == b) for b in range(10)],
            "one_factor_pairs_available": sum(1 for g in groups if ((_base_of(g).get("meta") or {}).get("pair") or {}).get("factor")),
            "one_factor_pairs_kept": sum(1 for g in kept if ((_base_of(g).get("meta") or {}).get("pair") or {}).get("factor")),
            "all_key_label_tenths_kept": [sum(1 for g in kept for r in g if _tenth(_key_label(r)) == b) for b in range(10)],
        }
    chosen = [r for r in records if str(r["id"]) in chosen_ids]
    return chosen, report


def _index(records: list[dict[str, Any]]) -> dict[str, set[str]]:
    from robo_jev.simgen.holdout import request_hash
    from robo_jev.simgen.narration import state_sha256

    return {"ids": {str(r["id"]) for r in records}, "requests": {request_hash(r) for r in records},
            "states": {state_sha256(r["request"]["state"]) for r in records}}


def holdout_check(train: list[dict[str, Any]], holdout: list[dict[str, Any]]) -> dict[str, Any]:
    """The H-set never trained: no shared scene (``meta.scene``), id, whole request or whole state with the MuJoCo training records."""
    train_scenes = Counter(str((r.get("meta") or {}).get("scene")) for r in train)
    hold_scenes = Counter(str((r.get("meta") or {}).get("scene")) for r in holdout)
    a, b = _index(train), _index(holdout)
    shared = {"scenes": sorted(set(train_scenes) & set(hold_scenes)), **{k: len(a[k] & b[k]) for k in ("ids", "requests", "states")}}
    splits = sorted({str((r.get("meta") or {}).get("split")) for r in holdout})
    out = {"holdout_records": len(holdout), "holdout_scenes": dict(sorted(hold_scenes.items())), "holdout_splits": splits, "shared": shared,
           "pass": not shared["scenes"] and not any(shared[k] for k in ("ids", "requests", "states")) and "train" not in splits}
    return out


def build_mujoco(v03: list[dict[str, Any]], scenes: list[dict[str, Any]], holdout: list[dict[str, Any]], *, per_family: int, seed: int,
                 config: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """(the v0.4 MuJoCo part's records — the subsample, then every scene record —, report). Refuses what must never train."""
    from robo_jev.decisions.examples import examples_per_epoch, split_dev
    from robo_jev.decisions.records import validate_record

    sub, selection = subsample(v03, per_family=per_family, seed=seed)
    problems = [f"{r.get('id')}: {validate_record(r)[:2]}" for r in scenes if validate_record(r)]
    if problems:
        raise SystemExit(f"{len(problems)} scene records fail the trainer's validation (first {problems[:3]})")
    splits = sorted({str((r.get("meta") or {}).get("split")) for r in scenes})
    if splits != ["train"]:
        raise SystemExit(f"the scene file holds splits {splits} — only train-split records are trained")
    ids = Counter(str(r["id"]) for r in sub + scenes)
    if any(n > 1 for n in ids.values()):
        raise SystemExit(f"duplicate ids between the subsample and the scenes: {[i for i, n in ids.items() if n > 1][:5]}")
    old_scenes = {str((r.get("meta") or {}).get("scene")) for r in sub}
    reused = sorted(old_scenes & {str((r.get("meta") or {}).get("scene")) for r in scenes})
    if reused:
        raise SystemExit(f"the scene file re-uses v0.3b's scenes {reused} — v0.3b's scenes enter only through the subsample")
    held = holdout_check(sub + scenes, holdout)
    if not held["pass"]:
        raise SystemExit(f"the H-set would be trained or shares text with training records: {held}")
    records = sub + scenes
    train, dev = split_dev(records, fraction=config["dev_fraction"], salt=config["dev_salt"])
    by_scene = Counter(str((r.get("meta") or {}).get("scene")) for r in records)
    by_set = Counter(str((r.get("meta") or {}).get("set")) for r in records)
    law = Counter(str((r.get("meta") or {}).get("law") or (r.get("meta") or {}).get("scene")) for r in scenes)
    report = {"subsample": {"per_family": per_family, "seed": seed, "families": selection,
                            "rule": "whole pair groups; equal groups per tenth of the base record's key label (base prior); "
                                    "random.Random('<script version>|<seed>|<family>|<tenth>') per stratum; short tenths filled from the nearest"},
              "records": len(records), "records_subsample": len(sub), "records_scenes": len(scenes), "by_scene": dict(sorted(by_scene.items())),
              "by_set": dict(sorted(by_set.items())), "scene_law": dict(sorted(law.items())), "train": len(train), "dev": len(dev),
              "first_pass_examples": examples_per_epoch(train, q=config["questions_per_request"]), "holdout": held}
    return records, report


def _read_records(path: Path) -> list[dict[str, Any]]:
    return [json.loads(raw) for raw in _raw_lines(path)]


def cmd_mujoco(args: argparse.Namespace) -> int:
    refuse_write(args.out)
    v03_sha = _check_recorded(args.v03, required=True)
    scenes_sha = _check_recorded(args.scenes, required=False)
    holdout_sha = _check_recorded(args.holdout, required=False)
    config = _config(args.train_config)
    v03, scenes, holdout = _read_records(args.v03), _read_records(args.scenes), _read_records(args.holdout)
    records, report = build_mujoco(v03, scenes, holdout, per_family=args.per_family, seed=args.seed, config=config)
    lines = {str(json.loads(raw)["id"]): raw for path in (args.v03, args.scenes) for raw in _raw_lines(path)}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temp = args.out.with_name(args.out.name + ".tmp")
    with temp.open("wb") as handle:
        for record in records:  # the original lines, byte for byte
            handle.write(lines[str(record["id"])])
    temp.replace(args.out)
    sha = _sha256(args.out)
    manifest = {"kind": "v04-mujoco-part", "script_version": SCRIPT_VERSION, "git": _git(), "sha256": sha,
                "out": {"path": str(args.out), "sha256": sha, "bytes": args.out.stat().st_size, "records": len(records)},
                "sources": {"v03b": {"path": str(args.v03), "sha256": v03_sha}, "scenes": {"path": str(args.scenes), "sha256": scenes_sha},
                            "holdout": {"path": str(args.holdout), "sha256": holdout_sha}},
                **report, "created": dt.datetime.now().astimezone().isoformat(timespec="seconds")}
    _write_json(args.out.with_name(args.out.stem + ".manifest.json"), manifest)
    _log(f"mujoco: subsample {report['records_subsample']} + scenes {report['records_scenes']} = {len(records)} records "
         f"({report['first_pass_examples']} first-pass examples) → {args.out}")
    return 0


# --------------------------------------------------------------------------
# standin — v0.4a's two files, before they exist (scratch only)
# --------------------------------------------------------------------------


def _renamed(group: list[dict[str, Any]], *, scene: str, set_name: str, split: str, prefix: str) -> list[dict[str, Any]]:
    import copy

    names = {str(r["id"]): f"{prefix}-{scene}-{r['id']}" for r in group}
    out = []
    for record in group:
        twin = copy.deepcopy(record)
        twin["id"] = names[str(record["id"])]
        twin["meta"].update(set=set_name, split=split, scene=scene, group=scene, standin_of=str(record["id"]))
        pair = twin["meta"].get("pair")
        if isinstance(pair, dict) and str(pair.get("of")) in names:
            pair["of"] = names[str(pair["of"])]
        out.append(twin)
    return out


def cmd_standin(args: argparse.Namespace) -> int:
    """A stand-in for v0.4a's ``mujoco-scenes.jsonl`` (7 scenes) and ``scene-holdout.jsonl`` (2 scenes, split test): the v0.3b groups the
    subsample does not take, renamed (ids ``v04sb-…`` / ``v04sh-…``, ``meta.set``, ``meta.scene``, ``meta.split``) — only to build and
    test the pipeline; it is refused for a final input (``all`` outside the scratch directory)."""
    from robo_jev.decisions.records import pair_groups

    scratch = (REPO / "artifacts" / "scratch").resolve()
    if scratch not in args.out_dir.resolve().parents:
        raise SystemExit(f"{args.out_dir}: the stand-in is written only under artifacts/scratch/")
    v03 = _read_records(args.v03)
    taken, _ = subsample(v03, per_family=args.per_family, seed=args.seed)
    taken_ids = {str(r["id"]) for r in taken}
    rest = [g for g in pair_groups(v03) if str(g[0]["id"]) not in taken_ids]
    random.Random(f"{SCRIPT_VERSION}|standin|{args.seed}").shuffle(rest)
    holdout, scenes = [], []
    for i, group in enumerate(rest[: 2 * args.holdout_groups]):
        holdout += _renamed(group, scene=f"standin_h{i % 2 + 1}", set_name=STANDIN_HOLDOUT_SET, split="test", prefix="v04sh")
    for i, group in enumerate(rest[2 * args.holdout_groups:]):
        scenes += _renamed(group, scene=f"standin_s{i % 7 + 1}", set_name=STANDIN_SET, split="train", prefix="v04sb")
    order = {str(r["id"]): i for i, r in enumerate(v03)}
    for name, rows in (("mujoco-scenes", scenes), ("scene-holdout", holdout)):
        rows.sort(key=lambda r: (r["meta"]["scene"], order[r["meta"]["standin_of"]]))
        path = args.out_dir / f"{name}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            for record in rows:
                handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
        _write_json(path.with_name(f"{name}.manifest.json"), {"kind": "v04b-standin", "what": "STAND-IN for task v0.4a's file — scratch only",
                                                              "sha256": _sha256(path), "records": len(rows), "source": str(args.v03),
                                                              "scenes": dict(Counter(r["meta"]["scene"] for r in rows))})
        _log(f"standin {name}: {len(rows)} records → {path}")
    return 0


# --------------------------------------------------------------------------
# exposure — the trainer's own view of a built input, seed by seed
# --------------------------------------------------------------------------


def _part(record: dict[str, Any], mujoco_sets: set[str]) -> str:
    from robo_jev.data.general_sets import SOURCES
    from robo_jev.decisions.single_probe import MIX_SPLITS

    meta = record.get("meta") or {}
    if meta.get("set") == "v09-numeric":  # Task v0.9a: the measurement-literacy records
        return "numeric"
    if meta.get("set") in mujoco_sets:
        return "mujoco"
    if isinstance(meta.get("source"), dict) and meta["source"].get("set") in SOURCES:
        return "general"
    if str(record.get("id", "")).startswith("v0b-") and isinstance(meta.get("basis"), dict):
        return "engine"
    if meta.get("split") in MIX_SPLITS and meta.get("origin_group"):
        return "d1"
    return "other"


def exposure_report(records: list[dict[str, Any]], *, config: dict[str, Any], seeds: list[int], mujoco_sets: set[str]) -> dict[str, Any]:
    """Per seed, the trainer's epoch-0 view (its loader's records, dev split and dev cap, :func:`record_pick`): examples per part,
    engine examples by question category, steps — and the checks of the brief against v0.3's counts."""
    from robo_jev.decisions.examples import record_pick, split_dev
    from robo_jev.simgen.system1_questions import exposure

    q = int(config["questions_per_request"])
    train, dev = split_dev(records, fraction=config["dev_fraction"], salt=config["dev_salt"])
    dev_used = dev[: config["dev_max_records"]] if config["dev_max_records"] else dev
    parts = defaultdict(list)
    for record in train:
        parts[_part(record, mujoco_sets)].append(record)
    out: dict[str, Any] = {"records": len(records), "train_records": len(train), "dev_records": len(dev), "dev_used": len(dev_used),
                           "dev_parts": dict(Counter(_part(r, mujoco_sets) for r in dev_used)), "train_by_part": {k: len(v) for k, v in parts.items()},
                           "seeds": {}}
    for seed in seeds:
        examples = {name: sum(len(record_pick(r, q=q, seed=seed, epoch=0)) for r in rows) for name, rows in parts.items()}
        total = sum(examples.values())
        engine = exposure(parts.get("engine", []), q=q, seed=seed)
        derived = engine.get("s1:band", 0) + engine.get("s1:comparison", 0)
        steps = math.ceil(total / int(config["batch_examples"]))
        mujoco = examples.get("mujoco", 0)
        checks = {
            "action_ge_21297": engine.get("choice:action", 0) >= V03_COUNTS["action"],
            "derived_within_10pct": abs(derived - V03_COUNTS["derived"]) <= TOLERANCE["derived"] * V03_COUNTS["derived"],
            "band_within_10pct": abs(engine.get("s1:band", 0) - V03_COUNTS["band"]) <= TOLERANCE["derived"] * V03_COUNTS["band"],
            "comparison_within_10pct": abs(engine.get("s1:comparison", 0) - V03_COUNTS["comparison"]) <= TOLERANCE["derived"] * V03_COUNTS["comparison"],
            "steps_le_1.25x": steps <= TOLERANCE["steps"] * V03_COUNTS["steps"],
            "mujoco_examples_within_5pct": abs(mujoco - V03_COUNTS["mujoco_examples"]) <= TOLERANCE["mujoco"] * V03_COUNTS["mujoco_examples"],
            "mujoco_share_0.20": round(mujoco / total, 4) >= 0.2 if total else False,
        }
        out["seeds"][str(seed)] = {"examples_per_epoch": {**examples, "total": total}, "shares": {k: v / total for k, v in examples.items()},
                                   "steps": steps, "engine_by_category": engine, "action": engine.get("choice:action", 0), "derived": derived,
                                   "band": engine.get("s1:band", 0), "comparison": engine.get("s1:comparison", 0),
                                   "derived_vs_v03": derived / V03_COUNTS["derived"] - 1, "steps_vs_v03": steps / V03_COUNTS["steps"], "checks": checks}
    return out


def _part_digests(path: Path, mujoco_sets: set[str]) -> dict[str, Any]:
    """sha256 of each part's lines (file order) and its record count."""
    digests: dict[str, Any] = defaultdict(hashlib.sha256)
    counts: Counter = Counter()
    for raw in _raw_lines(path):
        part = _part(json.loads(raw), mujoco_sets)
        digests[part].update(raw)
        counts[part] += 1
    return {part: {"records": counts[part], "sha256": digests[part].hexdigest()} for part in counts}


def cmd_exposure(args: argparse.Namespace) -> int:
    from robo_jev.decisions.train import load_training_records

    config = _config(args.train_config)
    records, validation = load_training_records(args.input)
    if validation["invalid"]:
        raise SystemExit(f"{args.input}: {validation['invalid']} records fail the trainer's validation")
    mujoco_sets = {str((r.get("meta") or {}).get("set")) for r in records if "mujoco" in str((r.get("meta") or {}).get("set")) or
                   str((r.get("meta") or {}).get("set")) in set(args.mujoco_set or [])}
    report = exposure_report(records, config=config, seeds=args.seeds, mujoco_sets=mujoco_sets)
    report["mujoco_sets"] = sorted(mujoco_sets)
    report["input"] = {"path": str(args.input), "sha256": _sha256(args.input), "validation": validation}
    unchanged = {}
    if args.reference is not None and args.reference.is_file():  # v0.3's input: the d1 and general parts must be byte-identical
        mine, theirs = _part_digests(args.input, mujoco_sets), _part_digests(args.reference, {"v03b-mujoco-train"})
        unchanged = {part: {"this": mine.get(part), "v03": theirs.get(part), "identical": mine.get(part) == theirs.get(part)} for part in ("d1", "general")}
        report["reference"] = {"path": str(args.reference), "sha256": _sha256(args.reference), "parts_identical": unchanged}
    report["kind"], report["script_version"], report["git"] = "v04-exposure", SCRIPT_VERSION, _git()
    report["v03_counts"], report["tolerance"] = V03_COUNTS, TOLERANCE
    report["created"] = dt.datetime.now().astimezone().isoformat(timespec="seconds")
    _write_json(args.out, report)
    for seed, entry in report["seeds"].items():
        _log(f"seed {seed}: {entry['examples_per_epoch']} · steps {entry['steps']} · action {entry['action']} · derived {entry['derived']} "
             f"(band {entry['band']}, comparison {entry['comparison']}) · checks {entry['checks']}")
    if unchanged:
        _log(f"d1 / general identical to v0.3: {[(p, v['identical']) for p, v in unchanged.items()]}")
    return 0


# --------------------------------------------------------------------------
# all
# --------------------------------------------------------------------------


def share_to_examples(engine: Path, share: float, *, config: dict[str, Any]) -> int:
    """``--mujoco-share``: the MuJoCo examples per epoch that make `share` of the input at the train config's seed, next to the engine part
    (counted with its pick rule), the d1 mix and v0.3's general part (both fixed) — the mix then holds the part at that count."""
    from robo_jev.decisions.examples import examples_per_epoch, split_dev

    if not 0.0 < share < 1.0:
        raise SystemExit(f"--mujoco-share must be in (0, 1) (got {share})")
    q, seed = int(config["questions_per_request"]), int(config["seed"])
    others = examples_per_epoch(split_dev(_read_records(engine), fraction=config["dev_fraction"], salt=config["dev_salt"])[0], q=q, seed=seed)
    others += V03_COUNTS["probe_examples"] + V03_COUNTS["general_examples"]
    return math.ceil(share * others / (1.0 - share))


def _run(argv: list[str]) -> None:
    _log("$ " + " ".join(argv))
    subprocess.run([sys.executable, *argv], check=True, cwd=REPO)


def cmd_all(args: argparse.Namespace) -> int:
    out = args.out_dir
    refuse_write(out)
    standin = any(STANDIN_SET in raw.decode("utf-8", "replace") for _, raw in zip(range(5), _raw_lines(args.scenes)))
    scratch = (REPO / "artifacts" / "scratch").resolve()
    if standin and scratch not in out.resolve().parents:
        raise SystemExit("the scenes file is the scratch stand-in — a final input is built only from task v0.4a's files")
    out.mkdir(parents=True, exist_ok=True)
    engine, mujoco, train = out / "engine-s1-pick.jsonl", out / "mujoco-v04.jsonl", out / "train-v04.jsonl"
    me = str(Path(__file__).resolve().relative_to(REPO))
    _run([me, "pick", "--input", str(args.engine), "--out", str(engine)])
    _run([me, "mujoco", "--v03", str(args.v03), "--scenes", str(args.scenes), "--holdout", str(args.holdout), "--per-family", str(args.per_family),
          "--seed", str(args.seed), "--train-config", str(args.train_config), "--out", str(mujoco)])
    scene_sets = sorted({str(json.loads(raw).get("meta", {}).get("set")) for raw in _raw_lines(args.scenes)})
    missing = [str(p) for p in SEALED_FOR_MIX if not p.is_file()]
    if missing:
        raise SystemExit(f"sealed files missing: {missing} — every sealed set is checked")
    mix = ["scripts/convert_single_probe.py", "mix", "--base", str(engine), "--extra", f"mujoco={mujoco}", "--probe", str(PROBE),
           "--sealed", *map(str, SEALED_FOR_MIX), str(args.holdout), "--general", str(GENERAL), "--general-records", str(V03_COUNTS["general_records"]), "--general-repeat-if-short",
           "--train-config", str(args.train_config), "--token-sample", str(args.token_sample), "--out", str(train)]
    target = args.mujoco_examples if args.mujoco_share is None else share_to_examples(engine, args.mujoco_share, config=_config(args.train_config))
    mix += ["--extra-examples", f"mujoco={target}"]
    _run(mix)
    overlap = ["scripts/system1_v03.py", "overlap", "--input", str(train), "--out", str(out / "train-v04.overlap.json"),
               "--sealed-extra", f"B1={B1}", "--sealed-extra", f"H-set={args.holdout}"]
    for name in scene_sets:
        overlap += ["--mujoco-set", name]
    _run(overlap)
    _run([me, "exposure", "--input", str(train), "--seeds", "0", "1", "--train-config", str(args.train_config), "--reference", str(TRAIN_V03),
          "--out", str(out / "train-v04.exposure.json"), *sum((["--mujoco-set", s] for s in scene_sets), [])])
    summary = {"kind": "v04-assembly", "script_version": SCRIPT_VERSION, "git": _git(), "standin": standin,
               "files": {p.name: {"path": str(p), "sha256": _sha256(p)} for p in (engine, mujoco, train)},
               "overlap": json.loads((out / "train-v04.overlap.json").read_text(encoding="utf-8"))["total_hits"],
               "exposure": {s: e["checks"] for s, e in json.loads((out / "train-v04.exposure.json").read_text(encoding="utf-8"))["seeds"].items()},
               "created": dt.datetime.now().astimezone().isoformat(timespec="seconds")}
    _write_json(out / "train-v04.assembly.json", summary)
    _log(f"done: {train} sha256 {summary['files'][train.name]['sha256'][:16]}… · overlap hits {summary['overlap']} · checks {summary['exposure']}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    pick = sub.add_parser("pick", help="engine-s1 → engine-s1-pick (+ manifest): the derived questions as secondary picks")
    pick.add_argument("--input", type=Path, default=ENGINE_S1)
    pick.add_argument("--out", type=Path, required=True)
    pick.set_defaults(func=cmd_pick)
    mj = sub.add_parser("mujoco", help="v0.3b subsample + v0.4a's training scenes → the v0.4 MuJoCo part (+ manifest)")
    mj.add_argument("--v03", type=Path, default=MUJOCO_V03)
    mj.add_argument("--scenes", type=Path, required=True)
    mj.add_argument("--holdout", type=Path, required=True)
    mj.add_argument("--per-family", type=int, default=1500)
    mj.add_argument("--seed", type=int, default=SUBSAMPLE_SEED)
    mj.add_argument("--train-config", type=Path, default=TRAIN_CONFIG)
    mj.add_argument("--out", type=Path, required=True)
    mj.set_defaults(func=cmd_mujoco)
    st = sub.add_parser("standin", help="scratch stand-in for v0.4a's scenes and holdout (v0.3b records the subsample leaves)")
    st.add_argument("--v03", type=Path, default=MUJOCO_V03)
    st.add_argument("--per-family", type=int, default=1500)
    st.add_argument("--seed", type=int, default=SUBSAMPLE_SEED)
    st.add_argument("--holdout-groups", type=int, default=200, help="pair groups per stand-in holdout scene")
    st.add_argument("--out-dir", type=Path, required=True)
    st.set_defaults(func=cmd_standin)
    ex = sub.add_parser("exposure", help="the trainer's epoch-0 view of a built input, per seed, with the brief's checks")
    ex.add_argument("--input", type=Path, required=True)
    ex.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    ex.add_argument("--train-config", type=Path, default=TRAIN_CONFIG)
    ex.add_argument("--reference", type=Path, default=None, help="v0.3's input: its d1 and general parts must be byte-identical")
    ex.add_argument("--mujoco-set", action="append", default=[])
    ex.add_argument("--out", type=Path, required=True)
    ex.set_defaults(func=cmd_exposure)
    every = sub.add_parser("all", help="pick → mujoco → mix → overlap → exposure (one command)")
    every.add_argument("--engine", type=Path, default=ENGINE_S1)
    every.add_argument("--v03", type=Path, default=MUJOCO_V03)
    every.add_argument("--scenes", type=Path, required=True)
    every.add_argument("--holdout", type=Path, required=True)
    every.add_argument("--per-family", type=int, default=1500)
    every.add_argument("--seed", type=int, default=SUBSAMPLE_SEED)
    every.add_argument("--mujoco-examples", type=int, default=V03_COUNTS["mujoco_examples"])
    every.add_argument("--mujoco-share", type=float, default=None,
                       help="hold the MuJoCo share instead of v0.3's count (not the default): converted into an example count at the train config's seed")
    every.add_argument("--token-sample", type=int, default=3000)
    every.add_argument("--train-config", type=Path, default=TRAIN_CONFIG)
    every.add_argument("--out-dir", type=Path, required=True)
    every.set_defaults(func=cmd_all)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
