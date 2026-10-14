#!/usr/bin/env python
"""Task v0.6b — the v0.6 training input: v0.4's input except the MuJoCo part, which becomes 4 authors × 3 laws × 1 scene (H-author).

    # the whole assembly, one command, once the three training authors' files exist (CPU unit, ≈ 15 min, ≤ 32 GB):
    python scripts/assemble_v06.py all --out-dir artifacts/simgen/v06
    # before the authors land: stand-ins for their four files (v0.4a scene records and v0.3b records the A0 subsample leaves, relabelled
    # as fake authors), written only under artifacts/scratch/, and the same pipeline on them
    python scripts/assemble_v06.py standin --out-dir artifacts/scratch/v06b/standin
    python scripts/assemble_v06.py all --author a1=artifacts/scratch/v06b/standin/a1/train.jsonl --author a2=… --author a3=… \
        --a4 artifacts/scratch/v06b/standin/a4/test.jsonl --out-dir artifacts/scratch/v06b/assembled
    # the steps one by one: mujoco, then convert_single_probe.py mix, system1_v03.py overlap, exposure, authors

**The MuJoCo part** (``mujoco``; PRE-DECLARED v0.6 VERDICT RULE, progress.md):

* **A0** — v0.3b's ``tilt``, ``tower`` and ``grip`` (``artifacts/simgen/v03/mujoco-train.jsonl``) subsampled to ``--per-family`` records
  (default 1,250) each with v0.4's subsample rule (:func:`assemble_v04.subsample`: whole pair groups, equal groups per tenth of the base
  record's key label — the flat key profile —, ``random.Random('v04b-assemble-1|<seed>|<family>|<tenth>')``) and **v0.4's seed**
  2026101304 (recorded): every tenth's pool is shuffled exactly as for v0.4, so A0 is the head of v0.4's own A0 (a subset — checked).
* **A1–A3** — each author's ``artifacts/v06/a<k>/train.jsonl`` in full (lines byte for byte), after the checks below; the counts are
  reported (the plan: ≈ 3,750 records each). ``--max-per-author`` would subsample an author over it (not the default).
* **MuJoCo examples per epoch held at v0.4's 36,648** (the trainer's own count: its dev split, q = 2): if the first pass is short, a
  recorded second pass of whole pair groups — the mix's own rule (copies ``<id>~r2``, ``meta.repeat_of``, round-robin over the scenes in file
  order, stop at the first group that reaches the target) except that a scene is keyed by ``meta.set`` + ``meta.scene`` (two authors may
  name a scene alike); if over, whole pair groups are dropped from the end of the largest scene until the part is at or below it.
* **Labels as authored** (base prior, no clipping): every yes/no and score label must equal ``meta.labels_by_prior.base`` (checked; nothing
  is transformed). Authors' records must be valid for the trainer, split ``train``, ``meta.set`` ``v06-a<k>`` and ids ``v06a<k>-…``; no
  record of the held-out author A4 (``v06-a4`` / ``v06a4-``) may enter.

**Everything else is v0.4's, byte for byte**: the engine part (``artifacts/simgen/v04/engine-s1-pick.jsonl``, with ``meta.question_pick``),
the d1 mix and the general part (exactly v0.4's 46,771 records incl. its second pass) — the mix (``convert_single_probe.py mix``) writes them
from the same files; ``exposure`` proves it with part digests against ``train-v04.jsonl`` and compares the trainer's exposure per seed
(action, derived, every engine category, steps) with v0.4's stored report.

**Sealed checks** (0 hits required): T1–T7, G1, G2, T8a, T8b, T9 (registered), B1-infamily and the H-set (ids, whole requests, whole states;
text units for the sets written outside our generators) and the held-out author's test set A4 — ids, whole requests, whole states **and text
units** (``system1_v03.py overlap --sealed-outside``), plus the mix's own sealed checks (A4 is passed to it too).

**Cross-author overlap** (``authors``): pairwise over A0 (the subsample), A1, A2, A3 and A4 — shared ids, whole states, whole requests,
sentences (≥ 5 words) and text units, and the overlap of their word 5-grams; and each author's composition. Counts and hashes only — no
author's text is ever printed or written.
"""

from __future__ import annotations

import argparse
import copy
import datetime as dt
import functools
import hashlib
import importlib.util
import json
import random
import re
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Mapping, Sequence

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

SCRIPT_VERSION = "v06b-assemble-1"
CODE_PATHS = ("scripts/assemble_v06.py", "scripts/assemble_v04.py", "scripts/convert_single_probe.py", "scripts/system1_v03.py",
              "src/robo_jev/decisions/examples.py")
ARTIFACTS = REPO / "artifacts"
V03 = ARTIFACTS / "simgen" / "v03"
V04 = ARTIFACTS / "simgen" / "v04"
MUJOCO_V03 = V03 / "mujoco-train.jsonl"
ENGINE_PICK = V04 / "engine-s1-pick.jsonl"
TRAIN_V04 = V04 / "train-v04.jsonl"
EXPOSURE_V04 = V04 / "train-v04.exposure.json"
HSET = V04 / "scene-holdout.jsonl"
#: v0.4's files the v0.6 input is built from or compared with (progress.md: FINAL v0.4 INPUT; their manifests).
PINNED = {"engine-s1-pick": "12e434178e46af233633d9b18849b4d22a0932302b1e2b88880f5e41cbc4a76b",
          "train-v04": "89bca7153924bdced8347a0de1489dc9cd59b975bf4da5c36f7c80d99bf6da4d",
          "mujoco-train-v03b": "4c7c4c3a7de02380f0995fed5f68ef0cfa5d0a83dff6022be8c75ca7a01bbdf1"}
PROBE = ARTIFACTS / "simgen" / "probes" / "G1-train-mix.jsonl"
G1 = ARTIFACTS / "simgen" / "probes" / "G1.jsonl"
GENERAL = ARTIFACTS / "simgen" / "v02" / "general-train.jsonl"
B1 = ARTIFACTS / "b1" / "b1-infamily.jsonl"
V06 = ARTIFACTS / "v06"
OUT_DIR = ARTIFACTS / "simgen" / "v06"
TRAIN_CONFIG = REPO / "configs" / "train" / "decisions-v0-lora-h100.yaml"
#: The mix's ``--sealed`` list (G1 first: the d1 lineage check) — T1–T7, G2, T8a, T8b, T9, B1-infamily; the H-set and A4 are added per run.
SEALED_FOR_MIX = (G1, *(ARTIFACTS / "simgen" / "v0b" / "tests" / f"T{i}.jsonl" for i in range(1, 8)), ARTIFACTS / "simgen" / "probes" / "G2.jsonl",
                  ARTIFACTS / "t8" / "T8a.jsonl", ARTIFACTS / "t8" / "T8b.jsonl", ARTIFACTS / "t9" / "T9.jsonl", B1)
#: The training authors, the held-out author, and A0 (v0.3b's generator).
AUTHORS = ("a1", "a2", "a3")
HELD_OUT = "a4"
A0_SET = "v03b-mujoco-train"
A0_FAMILIES = ("tilt", "grip", "tower")
A0_PER_FAMILY = 1250
#: v0.4's subsample seed: v0.6's A0 is the head of each tenth's v0.4 draw, i.e. a subset of v0.4's A0.
A0_SEED = 2026101304
#: v0.4's MuJoCo examples per epoch and general records (train-v04.manifest.json; held).
MUJOCO_EXAMPLES = 36648
GENERAL_RECORDS = 46771
#: The plan's sizes (reported against, never enforced): records per training author and per scene.
PLAN_RECORDS = {"author": 3750, "scene": 1250, "a4": 800}
#: Places never written to (the brief: simgen/* but v06, t8/t9/b1, the authors' folders, v0.4 / v0.5 reports, sealed tests and probes).
NEVER_WRITE = (V03, V04, ARTIFACTS / "simgen" / "v05", ARTIFACTS / "simgen" / "v0b" / "tests", ARTIFACTS / "simgen" / "probes",
               ARTIFACTS / "simgen" / "v02", ARTIFACTS / "t8", ARTIFACTS / "t9", ARTIFACTS / "b1", V06, ARTIFACTS / "reports" / "v04",
               ARTIFACTS / "reports" / "v05")
SCRATCH = ARTIFACTS / "scratch"
STANDIN_MARK = "standin_of"
SCALAR_TYPES = ("noul", "score")
PRIORS = ("base", "alt_a", "alt_b")
#: Second-pass / trim tolerance: whole pair groups move the count in steps of ≤ 2 × q examples.
HOLD_SLACK = 3


def author_set(author: str) -> str:
    return f"v06-{author}"


def author_prefix(author: str) -> str:
    return f"v06{author}-"


def _log(message: str) -> None:
    print(f"[v06b {dt.datetime.now().strftime('%H:%M:%S')}] {message}", file=sys.stderr, flush=True)


@functools.lru_cache(maxsize=None)
def _script(name: str):
    """A sibling script as a module (scripts/assemble_v04.py, scripts/convert_single_probe.py)."""
    spec = importlib.util.spec_from_file_location(f"v06b_{Path(name).stem}", REPO / "scripts" / name)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _v04():
    return _script("assemble_v04.py")


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


def _read_records(path: Path) -> list[dict[str, Any]]:
    return [json.loads(raw) for raw in _raw_lines(path)]


def refuse_write(path: Path) -> None:
    """Outputs never go to v0.3's, v0.4's or v0.5's files, the sealed sets, B1, the authors' folders or the v0.4 / v0.5 reports."""
    resolved = Path(path).resolve()
    for place in NEVER_WRITE:
        place = place.resolve()
        if resolved == place or place in resolved.parents:
            raise SystemExit(f"{path}: refusing to write into {place}")


def _in_scratch(path: Path) -> bool:
    return SCRATCH.resolve() in Path(path).resolve().parents


def _meta(record: Mapping[str, Any]) -> Mapping[str, Any]:
    return record.get("meta") or {}


def scene_key(record: Mapping[str, Any]) -> str:
    """A scene of the MuJoCo part: ``<meta.set>:<meta.scene>`` (two authors may give a scene the same name)."""
    meta = _meta(record)
    return f"{meta.get('set')}:{meta.get('scene') or meta.get('group') or 'all'}"


def _scalar(label: Any) -> float | None:
    if not isinstance(label, Mapping):
        return None
    for key in ("p_yes", "value"):
        if label.get(key) is not None:
            return float(label[key])
    return None


def _config(path: Path) -> dict[str, Any]:
    return _v04()._config(path)


# --------------------------------------------------------------------------
# Author files: checks and composition (pure)
# --------------------------------------------------------------------------


def author_problems(records: Sequence[Mapping[str, Any]], author: str, *, split: str = "train") -> list[str]:
    """Every way an author's file breaks the contract the assembly relies on (empty = fine): trainer validation, split, ``meta.set``
    ``v06-<author>``, id prefix ``v06<author>-``, unique ids, labels by prior under base / alt_a / alt_b, and every yes/no and score label
    equal to the base prior's (labels as authored — nothing here transforms them)."""
    from robo_jev.decisions.records import validate_record

    problems: list[str] = []
    want_set, want_prefix = author_set(author), author_prefix(author)
    seen: Counter = Counter()
    for record in records:
        rid = str(record.get("id"))
        seen[rid] += 1
        meta = _meta(record)
        found = validate_record(dict(record))
        if found:
            problems.append(f"{rid}: fails the trainer's validation: {found[:2]}")
        if meta.get("split") != split:
            problems.append(f"{rid}: split {meta.get('split')!r}, want {split!r}")
        if meta.get("set") != want_set:
            problems.append(f"{rid}: meta.set {meta.get('set')!r}, want {want_set!r}")
        if not rid.startswith(want_prefix):
            problems.append(f"{rid}: id does not start with {want_prefix!r}")
        by_prior = meta.get("labels_by_prior")
        if not isinstance(by_prior, Mapping) or sorted(by_prior) != sorted(PRIORS):
            problems.append(f"{rid}: meta.labels_by_prior must hold {list(PRIORS)} (got {sorted(by_prior) if isinstance(by_prior, Mapping) else by_prior!r})")
            continue
        for name, question in (record.get("request") or {}).get("questions", {}).items():
            if question.get("type") not in SCALAR_TYPES:
                continue
            label, base = _scalar((record.get("labels") or {}).get(name)), _scalar((by_prior.get("base") or {}).get(name))
            if label is None or base is None or label != base:
                problems.append(f"{rid}/{name}: label {label!r} is not the base prior's {base!r} (labels must be as authored, base prior)")
    problems += [f"{rid}: id appears {n} times" for rid, n in seen.items() if n > 1]
    return problems


def held_out_marks(records: Iterable[Mapping[str, Any]]) -> list[str]:
    """Ids of records that belong to the held-out author A4 (``meta.set`` ``v06-a4`` or id prefix ``v06a4-``, or a registered sealed set)."""
    from robo_jev.data.sealed import sealed_record_set

    out = []
    for record in records:
        rid = str(record.get("id"))
        if _meta(record).get("set") == author_set(HELD_OUT) or rid.startswith(author_prefix(HELD_OUT)) or sealed_record_set(dict(record)):
            out.append(rid)
    return out


def _tenth(value: float) -> int:
    return min(9, int(float(value) * 10))


def composition(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Records, scenes, laws, pair groups (one-factor share), the base record's key-label tenths, the yes/no + score label bands and the
    robust share — counts only."""
    from robo_jev.bench.system1 import band
    from robo_jev.decisions.records import pair_groups

    groups = pair_groups(list(records))
    by_scene: dict[str, Counter] = defaultdict(Counter)
    for group in groups:
        base = next((r for r in group if (_meta(r).get("pair") or {}).get("role") == "base"), group[0])
        pair = _meta(base).get("pair") or {}
        tally = by_scene[str(_meta(base).get("scene"))]
        tally["groups"] += 1
        tally["records"] += len(group)
        tally["pairs"] += int(len(group) == 2)
        tally["one_factor_pairs"] += int(len(group) == 2 and bool(pair.get("factor")))
        key = _meta(base).get("key_question")
        value = _scalar((base.get("labels") or {}).get(key)) if key else None
        if value is not None:
            tally[f"key_tenth_{_tenth(value)}"] += 1
    bands: Counter = Counter()
    robust: Counter = Counter()
    kinds: Counter = Counter()
    laws: Counter = Counter()
    for record in records:
        meta = _meta(record)
        laws[str(meta.get("law"))] += 1
        kinds.update(str(k) for k in (meta.get("question_kind") or {}).values())
        robustness = meta.get("robustness") or {}
        for name, question in record["request"]["questions"].items():
            if question.get("type") not in SCALAR_TYPES:
                continue
            value = _scalar((record.get("labels") or {}).get(name))
            if value is None:
                continue
            bands[band(value)] += 1
            entry = robustness.get(name)
            robust["robust" if isinstance(entry, Mapping) and entry.get("robust") is True else "prior_dependent" if isinstance(entry, Mapping) else "no_entry"] += 1
    scalar = sum(bands.values())
    scenes = {}
    for scene, tally in sorted(by_scene.items()):
        tenths = [tally.get(f"key_tenth_{b}", 0) for b in range(10)]
        scenes[scene] = {"records": tally["records"], "groups": tally["groups"], "pairs": tally["pairs"], "one_factor_pairs": tally["one_factor_pairs"],
                         "one_factor_share": tally["one_factor_pairs"] / tally["pairs"] if tally["pairs"] else None, "base_key_tenths": tenths}
    return {"records": len(records), "groups": len(groups), "pairs": sum(1 for g in groups if len(g) == 2), "unpaired": sum(1 for g in groups if len(g) == 1),
            "scenes": scenes, "laws": dict(sorted(laws.items())), "question_kinds": dict(sorted(kinds.items())),
            "yes_no_score_questions": scalar, "bands": {b: bands.get(b, 0) for b in ("low", "middle", "high")},
            "band_shares": {b: bands.get(b, 0) / scalar for b in ("low", "middle", "high")} if scalar else None,
            "robustness": dict(robust), "robust_share": robust.get("robust", 0) / scalar if scalar else None,
            "questions_per_record": sum(len(r["request"]["questions"]) for r in records) / len(records) if records else None}


# --------------------------------------------------------------------------
# Holding the part at v0.4's examples (pure)
# --------------------------------------------------------------------------


def _train_examples(records: Sequence[Mapping[str, Any]], config: Mapping[str, Any]) -> int:
    from robo_jev.decisions.examples import examples_per_epoch, split_dev

    train, _ = split_dev(list(records), fraction=config["dev_fraction"], salt=config["dev_salt"])
    return examples_per_epoch(train, q=int(config["questions_per_request"]))


def second_pass(records: Sequence[Mapping[str, Any]], *, target: int, config: Mapping[str, Any],
                key: Callable[[Mapping[str, Any]], str] = scene_key) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """The mix's recorded second pass (``convert_single_probe._extra_second_passes`` with ``--extra-examples``): copies of whole pair groups
    (``_repeated_group``: ``<id>~r2``, ``meta.repeat_of``), taken round-robin over the scenes (`key`; the mix keys by ``meta.scene``) in
    file order, until the part's train examples reach `target`. Returns (copies, report); no copies when the first pass reaches it."""
    from robo_jev.decisions.examples import split_dev
    from robo_jev.decisions.records import pair_groups

    q = int(config["questions_per_request"])
    have = _train_examples(records, config)
    report: dict[str, Any] = {"first_pass_examples": have, "target_examples": int(target), "records": 0, "pair_groups": 0, "by_scene": {}}
    if have >= target:
        report["train_examples"] = 0
        return [], report
    queues: dict[str, list[list[dict[str, Any]]]] = {}
    for group in pair_groups(list(records)):
        queues.setdefault(key(group[0]), []).append(group)
    order = []
    while any(queues.values()):
        for scene in list(queues):
            if queues[scene]:
                order.append(queues[scene].pop(0))
    repeated = _script("convert_single_probe.py")._repeated_group
    copies: list[dict[str, Any]] = []
    by_scene: Counter = Counter()
    groups = 0
    for group in order:
        if have >= target:
            break
        twins = repeated(group)
        train, _ = split_dev(twins, fraction=config["dev_fraction"], salt=config["dev_salt"])
        have += sum(min(q, len(r["request"]["questions"])) for r in train)
        copies += twins
        by_scene[key(group[0])] += len(twins)
        groups += 1
    if have < target:
        raise ValueError(f"even a whole second pass cannot reach {target} examples (reached {have})")
    report.update(records=len(copies), pair_groups=groups, by_scene=dict(sorted(by_scene.items())), train_examples=have - report["first_pass_examples"],
                  examples=have, rule="whole pair groups, round-robin over the scenes (meta.set:meta.scene) in file order; copies <id>~r2 with "
                                      "meta.repeat_of (a pair's meta.pair.of points at its partner's copy) — the mix's own second pass")
    return copies, report


def trim(records: Sequence[Mapping[str, Any]], *, target: int, config: Mapping[str, Any],
         key: Callable[[Mapping[str, Any]], str] = scene_key) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """The other direction: drop whole pair groups — always the last group (file order) of the scene with the most groups left (ties: the
    later scene) — until the part's train examples are at or below `target`. Returns (kept records in file order, report)."""
    from robo_jev.decisions.records import pair_groups

    have = _train_examples(records, config)
    report: dict[str, Any] = {"first_pass_examples": have, "target_examples": int(target), "dropped_records": 0, "dropped_groups": 0, "by_scene": {}}
    if have <= target:
        report["examples"] = have
        return list(records), report
    queues: dict[str, list[list[dict[str, Any]]]] = {}
    order: list[str] = []
    for group in pair_groups(list(records)):
        scene = key(group[0])
        if scene not in queues:
            order.append(scene)
        queues.setdefault(scene, []).append(group)
    dropped: set[str] = set()
    by_scene: Counter = Counter()
    while have > target:
        scene = max(order, key=lambda s: (len(queues[s]), order.index(s)))
        if not queues[scene]:
            break
        group = queues[scene].pop()
        have -= _train_examples(group, config)
        dropped |= {str(r["id"]) for r in group}
        by_scene[scene] += len(group)
        report["dropped_groups"] += 1
    kept = [r for r in records if str(r["id"]) not in dropped]
    report.update(dropped_records=len(dropped), by_scene=dict(sorted(by_scene.items())), examples=have,
                  rule="drop the last pair group (file order) of the scene with the most groups left (ties: the later scene) until ≤ target")
    return kept, report


# --------------------------------------------------------------------------
# mujoco — A0 + A1–A3, held at v0.4's examples
# --------------------------------------------------------------------------


def _author_specs(specs: Sequence[str]) -> dict[str, Path]:
    out: dict[str, Path] = {}
    for spec in specs:
        name, sep, raw = str(spec).partition("=")
        if not sep or name not in AUTHORS or name in out:
            raise SystemExit(f"--author takes a1=PATH, a2=PATH, a3=PATH, each once (got {spec!r})")
        out[name] = Path(raw)
    missing = [a for a in AUTHORS if a not in out]
    if missing:
        raise SystemExit(f"--author: missing {missing} — v0.6 is A0 + all three training authors")
    return out


def _default_authors() -> list[str]:
    return [f"{a}={V06 / a / 'train.jsonl'}" for a in AUTHORS]


def _index(records: Sequence[Mapping[str, Any]]) -> dict[str, set[str]]:
    from robo_jev.simgen.holdout import request_hash
    from robo_jev.simgen.narration import state_sha256

    return {"ids": {str(r["id"]) for r in records}, "requests": {request_hash(dict(r)) for r in records},
            "states": {state_sha256(r["request"]["state"]) for r in records}}


def sealed_check(train: Sequence[Mapping[str, Any]], sealed: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """No shared id, whole request or whole state between the MuJoCo training records and a sealed set (the H-set, A4); scenes by
    ``meta.set:meta.scene`` (and the bare scene names, reported)."""
    a, b = _index(train), _index(sealed)
    shared = {k: len(a[k] & b[k]) for k in ("ids", "requests", "states")}
    keys = {"scene_keys": sorted({scene_key(r) for r in train} & {scene_key(r) for r in sealed})}
    names = sorted({str(_meta(r).get("scene")) for r in train} & {str(_meta(r).get("scene")) for r in sealed})
    splits = sorted({str(_meta(r).get("split")) for r in sealed})
    return {"records": len(sealed), "splits": splits, "shared": {**shared, **keys}, "same_scene_names_info": names,
            "pass": not any(shared.values()) and not keys["scene_keys"] and "train" not in splits}


def build_mujoco(v03: Sequence[Mapping[str, Any]], authors: Mapping[str, Sequence[Mapping[str, Any]]], *, sealed: Mapping[str, Sequence[Mapping[str, Any]]],
                 config: Mapping[str, Any], per_family: int = A0_PER_FAMILY, seed: int = A0_SEED, target: int = MUJOCO_EXAMPLES,
                 max_per_author: int | None = None) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """(first-pass records — A0's subsample, then each author's records —, second-pass copies, report). Refuses what must never train."""
    from robo_jev.decisions.records import pair_groups

    a0, selection = _v04().subsample(list(v03), per_family=per_family, seed=seed, families=A0_FAMILIES)
    marked = held_out_marks([r for records in authors.values() for r in records]) + held_out_marks(a0)
    if marked:
        raise ValueError(f"{len(marked)} record(s) of the held-out author A4 in the training part (first {marked[:3]}) — A4 is never trained")
    problems: list[str] = []
    for author, records in authors.items():
        problems += author_problems(records, author)
    if problems:
        raise ValueError(f"{len(problems)} problems in the authors' files (first {problems[:5]})")
    used: dict[str, list[dict[str, Any]]] = {}
    capped: dict[str, Any] = {}
    for author, records in authors.items():
        records = list(records)
        if max_per_author is not None and len(records) > max_per_author:  # not the default: whole groups, the author's own file order
            groups, kept = pair_groups(records), []
            rng = random.Random(f"{SCRIPT_VERSION}|cap|{author}|{seed}")
            order = list(range(len(groups)))
            rng.shuffle(order)
            keep_idx, count = set(), 0
            for i in order:
                if count + len(groups[i]) > max_per_author:
                    continue
                keep_idx.add(i)
                count += len(groups[i])
            kept_ids = {str(r["id"]) for i in keep_idx for r in groups[i]}
            kept = [r for r in records if str(r["id"]) in kept_ids]
            capped[author] = {"records_in_file": len(records), "records_kept": len(kept)}
            records = kept
        used[author] = records
    first = [dict(r) for r in a0] + [dict(r) for author in AUTHORS for r in used.get(author, [])]
    ids = Counter(str(r["id"]) for r in first)
    duplicates = [i for i, n in ids.items() if n > 1]
    if duplicates:
        raise ValueError(f"duplicate ids across A0 and the authors: {duplicates[:5]}")
    checks = {name: sealed_check(first, records) for name, records in sealed.items()}
    failing = {name: c for name, c in checks.items() if not c["pass"]}
    if failing:
        raise ValueError(f"the training part shares text or scenes with a sealed set, or the sealed set holds train records: {failing}")
    have = _train_examples(first, config)
    if have > target:
        first, held = trim(first, target=target, config=config)
        copies: list[dict[str, Any]] = []
    else:
        copies, held = second_pass(first, target=target, config=config)
    report = {
        "a0": {"source_set": A0_SET, "families": list(A0_FAMILIES), "per_family": per_family, "seed": seed, "records": len(a0), "selection": selection,
               "rule": "assemble_v04.subsample: whole pair groups; equal groups per tenth of the base record's key label (base prior); "
                       "random.Random('v04b-assemble-1|<seed>|<family>|<tenth>') per stratum — v0.4's seed, so A0 is the head of v0.4's draw"},
        "authors": {author: {"records_in_file": len(authors[author]), "records_used": len(used[author]), "set": author_set(author)} for author in AUTHORS},
        "capped": capped or None,
        "first_pass": {"records": len(first), "examples": have, "by_set": dict(Counter(str(_meta(r).get("set")) for r in first)),
                       "by_scene": dict(sorted(Counter(scene_key(r) for r in first).items()))},
        "held": held, "sealed": checks,
        "composition": {"a0": composition(a0), **{author: composition(used[author]) for author in AUTHORS}},
    }
    return first, copies, report


def cmd_mujoco(args: argparse.Namespace) -> int:
    refuse_write(args.out)
    v03_sha = _v04()._check_recorded(args.v03, required=True)
    specs = _author_specs(args.author)
    standin = False
    authors, sources = {}, {}
    for author, path in specs.items():
        records = _read_records(path)
        standin |= any(STANDIN_MARK in _meta(r) for r in records)
        authors[author] = records
        sources[author] = {"path": str(path), "sha256": _sha256(path), "records": len(records), "in_scratch": _in_scratch(path)}
    if (standin or any(s["in_scratch"] for s in sources.values())) and not _in_scratch(args.out):
        raise SystemExit("an author file is a scratch stand-in — a final MuJoCo part is built only from the authors' own files")
    sealed = {"H-set": _read_records(args.hset)}
    if args.a4 is not None:
        sealed["A4"] = _read_records(args.a4)
    elif not _in_scratch(args.out):
        raise SystemExit("--a4: the held-out author's test set is checked for every final build")
    config = _config(args.train_config)
    try:
        first, copies, report = build_mujoco(_read_records(args.v03), authors, sealed=sealed, config=config, per_family=args.per_family, seed=args.seed,
                                             target=args.examples, max_per_author=args.max_per_author)
    except ValueError as error:
        raise SystemExit(f"refusing: {error}") from error
    lines = {}
    for path in (args.v03, *specs.values()):
        for raw in _raw_lines(path):
            lines[str(json.loads(raw)["id"])] = raw
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temp = args.out.with_name(args.out.name + ".tmp")
    with temp.open("wb") as handle:
        for record in first:  # the original lines, byte for byte
            handle.write(lines[str(record["id"])])
        for record in copies:  # the mix's own serialisation of second-pass copies
            handle.write((json.dumps(record, ensure_ascii=False) + "\n").encode("utf-8"))
    temp.replace(args.out)
    sha = _sha256(args.out)
    a0_ids = {str(r["id"]) for r in first if _meta(r).get("set") == A0_SET}
    v04_part = V04 / "mujoco-v04.jsonl"
    if v04_part.is_file():  # A0 is the head of v0.4's own draw (same seed and rule)
        v04_a0 = {str(json.loads(raw)["id"]) for raw in _raw_lines(v04_part) if b'"v03b-mujoco-train"' in raw}
        report["a0"]["within_v04_subsample"] = {"v04_a0_records": len(v04_a0), "a0_records": len(a0_ids), "a0_not_in_v04": len(a0_ids - v04_a0)}
    manifest = {"kind": "v06-mujoco-part", "script_version": SCRIPT_VERSION, "git": _git(), "sha256": sha, "standin": standin,
                "out": {"path": str(args.out), "sha256": sha, "bytes": args.out.stat().st_size, "records": len(first) + len(copies),
                        "first_pass_records": len(first), "second_pass_records": len(copies)},
                "sources": {"v03b": {"path": str(args.v03), "sha256": v03_sha}, **sources,
                            **{name: {"path": str(path), "sha256": _sha256(path)} for name, path in (("H-set", args.hset), ("A4", args.a4)) if path is not None}},
                "target_examples": args.examples, "train_config": str(args.train_config), **report,
                "created": dt.datetime.now().astimezone().isoformat(timespec="seconds")}
    _write_json(args.out.with_name(args.out.stem + ".manifest.json"), manifest)
    held = report["held"]
    _log(f"mujoco: A0 {report['a0']['records']} + " + " + ".join(f"{a} {report['authors'][a]['records_used']}" for a in AUTHORS)
         + f" = {len(first)} first pass ({held['first_pass_examples']} examples) + {len(copies)} second pass → {held.get('examples', held['first_pass_examples'])} "
         f"examples (target {args.examples}) → {args.out} {sha[:16]}…")
    return 0


# --------------------------------------------------------------------------
# standin — the four authors' files before they exist (scratch only)
# --------------------------------------------------------------------------

#: Stand-in allocation: author → [(source, scene in the source, new scene name, first group, last group)] — whole pair groups in file order.
STANDIN_PLAN = {
    "a1": [("v04a", "forks", "forks", 0, 625), ("v04a", "table", "table", 0, 625), ("v04a", "plate", "plate", 0, 625)],
    "a2": [("v04a", "bin", "bin", 0, 625), ("v04a", "reach", "reach", 0, 625), ("v04a", "clamp", "clamp", 0, 625)],
    # A3 re-uses the scene names tilt and grip of A0 on purpose: the assembly keys scenes by meta.set:meta.scene
    "a3": [("v04a", "cabinet", "cabinet", 0, 625), ("v03b", "tilt", "tilt", 0, 625), ("v03b", "grip", "grip", 0, 625)],
    "a4": [("v04a", "forks", "forks", 625, 760), ("v04a", "table", "table", 625, 760), ("v04a", "plate", "plate", 625, 760)],
}
#: The stand-in A4's text is marked so that it shares no text unit with the stand-in A1 (same v0.4a templates) — a real A4 is another author.
STANDIN_A4_MARK = " (radio check, over)"


def _standin_record(record: Mapping[str, Any], *, author: str, scene: str, names: Mapping[str, str]) -> dict[str, Any]:
    twin = copy.deepcopy(dict(record))
    old = str(record["id"])
    twin["id"] = names[old]
    meta = twin["meta"]
    meta.update(set=author_set(author), split="test" if author == HELD_OUT else "train", scene=scene, group=scene, **{STANDIN_MARK: old})
    pair = meta.get("pair")
    if isinstance(pair, dict) and str(pair.get("of")) in names:
        pair["of"] = names[str(pair["of"])]
    if author == HELD_OUT:
        request = twin["request"]
        request["state"] = "\n".join(line + STANDIN_A4_MARK if line.strip() else line for line in str(request["state"]).split("\n"))
        for question in request["questions"].values():
            if isinstance(question.get("criteria"), dict):
                question["criteria"] = {k: str(v) + STANDIN_A4_MARK for k, v in question["criteria"].items()}
    return twin


def cmd_standin(args: argparse.Namespace) -> int:
    """Stand-ins for ``artifacts/v06/a{1,2,3}/train.jsonl`` and ``a4/test.jsonl`` (:data:`STANDIN_PLAN`): v0.4a's training scenes and the
    v0.3b groups the A0 subsample leaves, relabelled (ids ``v06a<k>-…``, ``meta.set`` ``v06-a<k>``, split, scene; ``meta.standin_of``) — only to
    build and test the pipeline; refused for a final input."""
    from robo_jev.decisions.records import pair_groups

    if not _in_scratch(args.out_dir):
        raise SystemExit(f"{args.out_dir}: the stand-in is written only under artifacts/scratch/")
    v03 = _read_records(args.v03)
    a0, _ = _v04().subsample(v03, per_family=args.per_family, seed=args.seed, families=A0_FAMILIES)
    taken = {str(r["id"]) for r in a0}
    pools: dict[tuple[str, str], list[list[dict[str, Any]]]] = defaultdict(list)
    for group in pair_groups([r for r in v03 if str(r["id"]) not in taken]):
        pools[("v03b", str(group[0]["meta"]["scene"]))].append(group)
    for group in pair_groups(_read_records(args.scenes)):
        pools[("v04a", str(group[0]["meta"]["scene"]))].append(group)
    summary = {}
    for author, plan in STANDIN_PLAN.items():
        rows: list[dict[str, Any]] = []
        for source, scene, new_scene, start, stop in plan:
            groups = pools[(source, scene)][start:stop]
            names = {str(r["id"]): author_prefix(author) + str(r["id"]).split("-", 1)[1] for g in groups for r in g}
            rows += [_standin_record(r, author=author, scene=new_scene, names=names) for g in groups for r in g]
        name = "test" if author == HELD_OUT else "train"
        path = args.out_dir / author / f"{name}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            for record in rows:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        summary[author] = {"path": str(path), "records": len(rows), "scenes": dict(Counter(r["meta"]["scene"] for r in rows)), "sha256": _sha256(path)}
        _write_json(path.with_name(f"{name}.manifest.json"), {"kind": "v06b-standin", "what": f"STAND-IN for author {author}'s file — scratch only",
                                                             "sha256": summary[author]["sha256"], "records": len(rows), "plan": plan})
        _log(f"standin {author}: {len(rows)} records {summary[author]['scenes']} → {path}")
    _write_json(args.out_dir / "standin.json", {"kind": "v06b-standin", "plan": STANDIN_PLAN, "a4_mark": STANDIN_A4_MARK, "authors": summary,
                                                "a0": {"per_family": args.per_family, "seed": args.seed}})
    return 0


# --------------------------------------------------------------------------
# exposure — the trainer's view of v0.6 against v0.4, per seed
# --------------------------------------------------------------------------

EXPOSURE_FIELDS = ("action", "derived", "band", "comparison", "engine_by_category")


def compare_exposure(mine: Mapping[str, Any], stored: Mapping[str, Any], *, target: int = MUJOCO_EXAMPLES) -> dict[str, Any]:
    """One seed's trainer view of v0.6 against v0.4's stored report: the engine, d1 and general examples and every engine field identical;
    the MuJoCo examples at the target (within :data:`HOLD_SLACK`); total and steps reported with their differences."""
    mx, sx = mine["examples_per_epoch"], stored["examples_per_epoch"]
    same = {f"examples:{part}": mx.get(part) == sx.get(part) for part in ("engine", "d1", "general")}
    same.update({field: mine[field] == stored[field] for field in EXPOSURE_FIELDS})
    checks = {**same, "mujoco_held": abs(int(mx.get("mujoco", 0)) - int(target)) <= HOLD_SLACK}
    return {"checks": checks, "pass": all(checks.values()),
            "v06": {"examples": mx, "steps": mine["steps"], "action": mine["action"], "derived": mine["derived"]},
            "v04": {"examples": sx, "steps": stored["steps"], "action": stored["action"], "derived": stored["derived"]},
            "difference": {"total": mx["total"] - sx["total"], "mujoco": mx.get("mujoco", 0) - sx.get("mujoco", 0), "steps": mine["steps"] - stored["steps"]}}


def author_exposure(records: Sequence[Mapping[str, Any]], *, config: Mapping[str, Any], seeds: Sequence[int], mujoco_sets: set[str]) -> dict[str, Any]:
    """Per seed, the trainer's epoch-0 MuJoCo examples by author (``meta.set``, first pass vs second-pass copies) and by scene."""
    from robo_jev.decisions.examples import record_pick, split_dev

    v04 = _v04()
    q = int(config["questions_per_request"])
    mujoco = [r for r in records if v04._part(r, mujoco_sets) == "mujoco"]
    train, _ = split_dev(mujoco, fraction=config["dev_fraction"], salt=config["dev_salt"])
    out = {}
    for seed in seeds:
        by_set, by_scene, kinds = Counter(), Counter(), Counter()
        for record in train:
            names = record_pick(record, q=q, seed=int(seed), epoch=0)
            tag = str(_meta(record).get("set")) + (":second_pass" if _meta(record).get("repeat_of") else "")
            by_set[tag] += len(names)
            by_scene[scene_key(record)] += len(names)
            for name in names:
                kinds[str((_meta(record).get("question_kind") or {}).get(name) or record["request"]["questions"][name]["type"])] += 1
        total = sum(by_set.values())
        authors = Counter()
        for tag, n in by_set.items():
            authors[tag.split(":")[0]] += n
        out[str(seed)] = {"mujoco_examples": total, "by_author": dict(sorted(authors.items())), "by_author_and_pass": dict(sorted(by_set.items())),
                          "author_shares": {k: v / total for k, v in sorted(authors.items())} if total else None,
                          "by_scene": dict(sorted(by_scene.items())), "by_question_kind": dict(sorted(kinds.items()))}
    return out


def cmd_exposure(args: argparse.Namespace) -> int:
    from robo_jev.decisions.train import load_training_records

    v04 = _v04()
    config = _config(args.train_config)
    records, validation = load_training_records(args.input)
    if validation["invalid"]:
        raise SystemExit(f"{args.input}: {validation['invalid']} records fail the trainer's validation")
    sets = {A0_SET, *(author_set(a) for a in AUTHORS), *args.mujoco_set}
    report = v04.exposure_report(records, config=config, seeds=args.seeds, mujoco_sets=sets)
    report["by_author"] = author_exposure(records, config=config, seeds=args.seeds, mujoco_sets=sets)
    del records
    stored = json.loads(args.stored.read_text(encoding="utf-8"))
    report["vs_v04"] = {str(s): compare_exposure(report["seeds"][str(s)], stored["seeds"][str(s)], target=args.target) for s in args.seeds}
    split_same = {k: report[k] == stored[k] for k in ("dev_used", "dev_parts")}
    mine = v04._part_digests(args.input, sets)
    theirs = v04._part_digests(args.reference, {A0_SET, "v04a-mujoco-scenes"})
    parts = {part: {"v06": mine.get(part), "v04": theirs.get(part), "identical": mine.get(part) == theirs.get(part)} for part in ("engine", "d1", "general")}
    parts["mujoco"] = {"v06": mine.get("mujoco"), "v04": theirs.get("mujoco"), "identical": mine.get("mujoco") == theirs.get("mujoco")}
    out = {"kind": "v06-exposure", "script_version": SCRIPT_VERSION, "git": _git(), "train_config": str(args.train_config), "mujoco_sets": sorted(sets),
           "input": {"path": str(args.input), "sha256": _sha256(args.input), "validation": validation},
           "reference": {"path": str(args.reference), "sha256": _sha256(args.reference), "stored_exposure": str(args.stored)},
           "parts": parts, "split_vs_v04": split_same, **report, "target_examples": args.target,
           "pass": all(e["pass"] for e in report["vs_v04"].values()) and all(parts[p]["identical"] for p in ("engine", "d1", "general"))
                   and all(split_same.values()) and not any(k == "other" for k in report.get("train_by_part", {})),
           "created": dt.datetime.now().astimezone().isoformat(timespec="seconds")}  # fmt: skip
    _write_json(args.out, out)
    for seed, entry in out["vs_v04"].items():
        _log(f"seed {seed}: v0.6 {entry['v06']} · v0.4 {entry['v04']} · Δ {entry['difference']} · checks pass {entry['pass']}")
    _log(f"parts identical to v0.4: {[(p, e['identical']) for p, e in parts.items()]} · pass {out['pass']}")
    return 0 if out["pass"] else 2


# --------------------------------------------------------------------------
# authors — cross-author overlap (counts only)
# --------------------------------------------------------------------------

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?;:])\s+|\n+")
#: What makes a training author touch the sealed A4 (as for the sets written outside our generators: G1, G2, T8a, T8b, T9).
SEALED_KINDS = ("ids", "states", "requests", "units")


def _texts(record: Mapping[str, Any]) -> list[str]:
    request = record.get("request") or {}
    texts = [str(request.get("state") or "")]
    for question in (request.get("questions") or {}).values():
        texts.append(str(question.get("instructions") or ""))
        criteria = question.get("criteria")
        if isinstance(criteria, Mapping):
            texts += [str(v) for v in criteria.values()]
        elif isinstance(criteria, (list, tuple)):
            texts += [str(v) for v in criteria]
    return texts


def _h(text: str) -> bytes:
    return hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest()


def text_index(records: Iterable[Mapping[str, Any]], *, min_words: int = 5, n: int = 5) -> dict[str, set]:
    """Hashes of a part's ids, whole states, whole requests, sentences (state, instructions and option texts split at . ! ? ; : and line
    breaks, normalised, ≥ `min_words` words), text units (:func:`robo_jev.data.general_sets.record_units`) and word `n`-grams (within each
    normalised text)."""
    from robo_jev.data.general_sets import normalize, record_units
    from robo_jev.simgen.holdout import request_hash
    from robo_jev.simgen.narration import state_sha256

    out: dict[str, set] = {"ids": set(), "states": set(), "requests": set(), "sentences": set(), "units": set(), "grams": set()}
    for record in records:
        out["ids"].add(str(record["id"]))
        out["states"].add(state_sha256(record["request"]["state"]))
        out["requests"].add(request_hash(dict(record)))
        out["units"] |= {_h(u) for u in record_units(dict(record))}
        for text in _texts(record):
            for piece in _SENTENCE_SPLIT.split(text):
                words = normalize(piece).split()
                if len(words) >= min_words:
                    out["sentences"].add(_h(" ".join(words)))
            words = normalize(text).split()
            out["grams"] |= {_h(" ".join(words[i:i + n])) for i in range(len(words) - n + 1)}
    return out


def pairwise_overlap(indexes: Mapping[str, Mapping[str, set]]) -> dict[str, Any]:
    """Every pair of parts: shared ids, states, requests, sentences and units, and the overlap of their 5-gram sets (shared count, the share
    of each side's 5-grams found in the other, Jaccard)."""
    names = list(indexes)
    out: dict[str, Any] = {}
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            x, y = indexes[a], indexes[b]
            shared = len(x["grams"] & y["grams"])
            union = len(x["grams"] | y["grams"])
            out[f"{a}~{b}"] = {**{k: len(x[k] & y[k]) for k in ("ids", "states", "requests", "sentences", "units")},
                               "grams_shared": shared, f"grams_share_of_{a}": shared / len(x["grams"]) if x["grams"] else None,
                               f"grams_share_of_{b}": shared / len(y["grams"]) if y["grams"] else None,
                               "grams_jaccard": shared / union if union else None}
    return out


def cmd_authors(args: argparse.Namespace) -> int:
    mujoco = [json.loads(raw) for raw in _raw_lines(args.mujoco)]
    parts = {"A0": [r for r in mujoco if _meta(r).get("set") == A0_SET and not _meta(r).get("repeat_of")]}
    specs = _author_specs(args.author)
    for author in AUTHORS:
        parts[author.upper()] = _read_records(specs[author])
    if args.a4 is not None:
        parts["A4"] = _read_records(args.a4)
    indexes = {name: text_index(records) for name, records in parts.items()}
    sizes = {name: {k: len(v) for k, v in index.items()} for name, index in indexes.items()}
    pairs = pairwise_overlap(indexes)
    held = {f"{a}~A4": {k: pairs[f"{a}~A4"][k] for k in SEALED_KINDS + ("sentences",)} for a in ("A0", "A1", "A2", "A3")} if "A4" in parts else {}
    out = {"kind": "v06-authors", "script_version": SCRIPT_VERSION, "git": _git(), "parts": {name: len(records) for name, records in parts.items()},
           "sources": {"mujoco": str(args.mujoco), **{a: str(p) for a, p in specs.items()}, "A4": str(args.a4) if args.a4 else None},
           "what": "counts only; sentences = state / instructions / option texts split at . ! ? ; : and line breaks, normalised, >= 5 words; units = "
                   "general_sets.record_units; grams = word 5-grams within each normalised text (blake2b-64 hashes)",
           "sizes": sizes, "pairwise": pairs, "training_vs_A4": held,
           "training_vs_A4_clean": all(not any(v[k] for k in SEALED_KINDS) for v in held.values()) if held else None,
           "training_vs_A4_clean_means": "no shared id, whole state, whole request or text unit (the sealed-outside rule); shared sentences and "
                                         "5-grams are reported, not refused (independent authors can coincide on a short sentence)",
           "composition": {name: composition(records) for name, records in parts.items()},
           "created": dt.datetime.now().astimezone().isoformat(timespec="seconds")}
    _write_json(args.out, out)
    args.out.with_suffix(".md").write_text(render_authors(out), encoding="utf-8")
    _log(f"authors: {out['parts']} · training vs A4 clean {out['training_vs_A4_clean']} → {args.out}")
    return 0


def render_authors(report: Mapping[str, Any]) -> str:
    lines = ["# v0.6 — cross-author overlap and composition (counts only)", "", report["what"] + ".", "",
             "| pair | ids | states | requests | sentences | units | 5-grams shared | share of first | share of second | Jaccard |", "|---|---|---|---|---|---|---|---|---|---|"]
    for pair, entry in report["pairwise"].items():
        a, b = pair.split("~")
        lines.append(f"| {a} ~ {b} | {entry['ids']} | {entry['states']} | {entry['requests']} | {entry['sentences']} | {entry['units']} | "
                     f"{entry['grams_shared']:,} | {entry[f'grams_share_of_{a}'] or 0:.3f} | {entry[f'grams_share_of_{b}'] or 0:.3f} | {entry['grams_jaccard'] or 0:.3f} |")
    lines += ["", "| part | records | groups | pairs | one-factor share | scenes | yes/no + score | low / middle / high | robust share |", "|---|---|---|---|---|---|---|---|---|"]
    for name, c in report["composition"].items():
        one = sum(s["one_factor_pairs"] for s in c["scenes"].values()) / c["pairs"] if c["pairs"] else 0
        shares = c["band_shares"] or {}
        lines.append(f"| {name} | {c['records']:,} | {c['groups']:,} | {c['pairs']:,} | {one:.2f} | " + ", ".join(f"{s} {v['records']}" for s, v in c["scenes"].items())
                     + f" | {c['yes_no_score_questions']:,} | {shares.get('low', 0):.2f} / {shares.get('middle', 0):.2f} / {shares.get('high', 0):.2f} | "
                     f"{c['robust_share'] or 0:.2f} |")
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------
# all
# --------------------------------------------------------------------------


def _run(argv: list[str]) -> int:
    _log("$ " + " ".join(argv))
    return subprocess.run([sys.executable, *argv], cwd=REPO).returncode


def cmd_all(args: argparse.Namespace) -> int:
    out = args.out_dir
    refuse_write(out)
    specs = _author_specs(args.author)
    if not _in_scratch(out):
        for path in (*specs.values(), args.a4):
            if _in_scratch(path):
                raise SystemExit(f"{path}: a scratch stand-in — a final input is built only from the authors' own files")
    for name, path, pinned in (("engine-s1-pick", ENGINE_PICK, PINNED["engine-s1-pick"]), ("train-v04", TRAIN_V04, PINNED["train-v04"])):
        if _sha256(path) != pinned:
            raise SystemExit(f"{path}: sha256 is not v0.4's {name} {pinned[:12]}…")
    missing = [str(p) for p in (*SEALED_FOR_MIX, HSET, args.a4) if not Path(p).is_file()]
    if missing:
        raise SystemExit(f"sealed files missing: {missing} — every sealed set is checked")
    out.mkdir(parents=True, exist_ok=True)
    mujoco, train = out / "mujoco-v06.jsonl", out / "train-v06.jsonl"
    me = str(Path(__file__).resolve().relative_to(REPO))
    authors = [f"--author={a}={specs[a]}" for a in AUTHORS]
    rc = {"mujoco": _run([me, "mujoco", "--v03", str(args.v03), *authors, "--a4", str(args.a4), "--hset", str(HSET), "--per-family", str(args.per_family),
                          "--seed", str(args.seed), "--examples", str(args.examples), "--train-config", str(args.train_config), "--out", str(mujoco)]
                         + (["--max-per-author", str(args.max_per_author)] if args.max_per_author else []))}
    if rc["mujoco"] != 0:
        raise SystemExit("mujoco failed")
    rc["mix"] = _run(["scripts/convert_single_probe.py", "mix", "--base", str(ENGINE_PICK), "--extra", f"mujoco={mujoco}", "--probe", str(PROBE),
                      "--sealed", *map(str, SEALED_FOR_MIX), str(HSET), str(args.a4), "--general", str(GENERAL), "--general-records", str(GENERAL_RECORDS),
                      "--general-repeat-if-short", "--train-config", str(args.train_config), "--token-sample", str(args.token_sample), "--out", str(train)])
    if rc["mix"] != 0:
        raise SystemExit("mix failed")
    sets = ["--mujoco-set", A0_SET, *sum((["--mujoco-set", author_set(a)] for a in AUTHORS), [])]
    rc["overlap"] = _run(["scripts/system1_v03.py", "overlap", "--input", str(train), "--out", str(out / "train-v06.overlap.json"), "--sealed-extra", f"B1={B1}",
                          "--sealed-extra", f"H-set={HSET}", "--sealed-outside", f"A4={args.a4}", *sets])
    rc["exposure"] = _run([me, "exposure", "--input", str(train), "--seeds", "0", "1", "--train-config", str(args.train_config), "--target", str(args.examples),
                           "--out", str(out / "train-v06.exposure.json")])
    rc["authors"] = _run([me, "authors", "--mujoco", str(mujoco), *authors, "--a4", str(args.a4), "--out", str(out / "train-v06.authors.json")])
    load = lambda name: json.loads((out / name).read_text(encoding="utf-8")) if (out / name).is_file() else {}  # noqa: E731
    part, mix, overlap, exposure, cross = (load(n) for n in ("mujoco-v06.manifest.json", "train-v06.manifest.json", "train-v06.overlap.json",
                                                             "train-v06.exposure.json", "train-v06.authors.json"))
    summary = {"kind": "v06-assembly", "script_version": SCRIPT_VERSION, "git": _git(), "standin": bool(part.get("standin")), "returncodes": rc,
               "files": {p.name: {"path": str(p), "sha256": _sha256(p)} for p in (mujoco, train)},
               "authors": part.get("authors"), "a0": {k: part.get("a0", {}).get(k) for k in ("records", "per_family", "seed", "within_v04_subsample")},
               "held": part.get("held"), "examples_per_epoch_mix": mix.get("examples_per_epoch"), "tokens_mix": mix.get("tokens"),
               "overlap_hits": overlap.get("total_hits"), "sealed_checked": sorted(overlap.get("sealed") or {}),
               "parts_identical_to_v04": {p: e.get("identical") for p, e in (exposure.get("parts") or {}).items()},
               "exposure_vs_v04": {s: {"pass": e["pass"], "difference": e["difference"]} for s, e in (exposure.get("vs_v04") or {}).items()},
               "training_vs_A4_clean": cross.get("training_vs_A4_clean"),
               "pass": all(v == 0 for v in rc.values()) and overlap.get("total_hits") == 0 and bool(exposure.get("pass")) and cross.get("training_vs_A4_clean") is True,
               "created": dt.datetime.now().astimezone().isoformat(timespec="seconds")}  # fmt: skip
    _write_json(out / "train-v06.assembly.json", summary)
    _log(f"done: {train} sha256 {summary['files'][train.name]['sha256'][:16]}… · overlap hits {summary['overlap_hits']} · exposure "
         f"{summary['exposure_vs_v04']} · A4 clean {summary['training_vs_A4_clean']} · pass {summary['pass']}")
    return 0 if summary["pass"] else 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    def authors_args(p: argparse.ArgumentParser) -> None:
        p.add_argument("--author", action="append", default=None, metavar="aK=PATH", help="a training author's file (default: artifacts/v06/a{1,2,3}/train.jsonl)")

    mj = sub.add_parser("mujoco", help="A0 (v0.3b subsample) + A1–A3 → the v0.6 MuJoCo part, held at v0.4's examples (+ manifest)")
    mj.add_argument("--v03", type=Path, default=MUJOCO_V03)
    authors_args(mj)
    mj.add_argument("--a4", type=Path, default=V06 / HELD_OUT / "test.jsonl", help="the held-out author's test set (sealed; checked, never trained)")
    mj.add_argument("--hset", type=Path, default=HSET)
    mj.add_argument("--per-family", type=int, default=A0_PER_FAMILY)
    mj.add_argument("--seed", type=int, default=A0_SEED)
    mj.add_argument("--examples", type=int, default=MUJOCO_EXAMPLES)
    mj.add_argument("--max-per-author", type=int, default=None, help="not the default: subsample an author's whole groups to at most N records")
    mj.add_argument("--train-config", type=Path, default=TRAIN_CONFIG)
    mj.add_argument("--out", type=Path, required=True)
    mj.set_defaults(func=cmd_mujoco)
    st = sub.add_parser("standin", help="scratch stand-ins for the four authors' files")
    st.add_argument("--v03", type=Path, default=MUJOCO_V03)
    st.add_argument("--scenes", type=Path, default=V04 / "mujoco-scenes.jsonl")
    st.add_argument("--per-family", type=int, default=A0_PER_FAMILY)
    st.add_argument("--seed", type=int, default=A0_SEED)
    st.add_argument("--out-dir", type=Path, required=True)
    st.set_defaults(func=cmd_standin)
    ex = sub.add_parser("exposure", help="the trainer's epoch-0 view of v0.6 per seed, against v0.4's stored report and part digests")
    ex.add_argument("--input", type=Path, required=True)
    ex.add_argument("--reference", type=Path, default=TRAIN_V04)
    ex.add_argument("--stored", type=Path, default=EXPOSURE_V04)
    ex.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    ex.add_argument("--target", type=int, default=MUJOCO_EXAMPLES)
    ex.add_argument("--mujoco-set", action="append", default=[])
    ex.add_argument("--train-config", type=Path, default=TRAIN_CONFIG)
    ex.add_argument("--out", type=Path, required=True)
    ex.set_defaults(func=cmd_exposure)
    au = sub.add_parser("authors", help="cross-author overlap (A0–A4, counts only) and composition")
    au.add_argument("--mujoco", type=Path, required=True, help="the built MuJoCo part (its A0 records are read from it)")
    authors_args(au)
    au.add_argument("--a4", type=Path, default=V06 / HELD_OUT / "test.jsonl")
    au.add_argument("--out", type=Path, required=True)
    au.set_defaults(func=cmd_authors)
    every = sub.add_parser("all", help="mujoco → mix → overlap → exposure → authors (one command)")
    every.add_argument("--v03", type=Path, default=MUJOCO_V03)
    authors_args(every)
    every.add_argument("--a4", type=Path, default=V06 / HELD_OUT / "test.jsonl")
    every.add_argument("--per-family", type=int, default=A0_PER_FAMILY)
    every.add_argument("--seed", type=int, default=A0_SEED)
    every.add_argument("--examples", type=int, default=MUJOCO_EXAMPLES)
    every.add_argument("--max-per-author", type=int, default=None)
    every.add_argument("--token-sample", type=int, default=3000)
    every.add_argument("--train-config", type=Path, default=TRAIN_CONFIG)
    every.add_argument("--out-dir", type=Path, default=OUT_DIR)
    every.set_defaults(func=cmd_all)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if hasattr(args, "author") and args.author is None:
        args.author = _default_authors()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
