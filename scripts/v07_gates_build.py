#!/usr/bin/env python
"""Task v0.7b — build the sealed v0.7 shape gates M2g and M1g (CPU), and re-check them against v0.7's training input once it exists.

    artifacts/scratch/v07b/cpu.sh build scripts/v07_gates_build.py build                 # → artifacts/v07/gates/
    artifacts/scratch/v07b/cpu.sh recheck scripts/v07_gates_build.py recheck-v07         # once artifacts/simgen/v07/train-v07.jsonl exists

Design: :mod:`robo_jev.bench.gates_v07` (module docstring). Inputs: A4 (``artifacts/v06/a4/test.jsonl`` — sealed; its second use, as the
v0.7 rule pre-declares), the H-set (``artifacts/simgen/v04/scene-holdout.jsonl``), B1-infamily (``artifacts/b1/b1-infamily.jsonl``), the MMLU
test rows (``scripts/v07_fetch_mc.py``), task iface's M1 items and M2 manifest (only to leave out what iface used). Nothing under
``artifacts/simgen/v07/`` is opened except ``train-v07.jsonl`` by ``recheck-v07`` (in code; only counts and hashes are written).

**build**

1. *M2g sample* (whole one-fact pairs, seed :data:`SEED`): per set a target number of pairs spread evenly over its scenes (a seeded order
   within each scene; a scene with too few gives all it has), never a pair task iface's M2 wrapped; B1's one unpaired record is left out.
   Styles: :func:`robo_jev.bench.gates_v07.assign_styles` per set and scene.
2. *M2g records*: the six conditions (:data:`robo_jev.bench.gates_v07.CONDITIONS`); every sheet passes
   :func:`~robo_jev.bench.gates_v07.sheet_problems`; both records of a pair share their filler; the three 4k tone versions are identical
   but for the tone sentences; a longer sheet holds every line of the shorter one; ids, questions, labels and meta are the same in all six
   files and the questions / labels equal the original's; ``validate_record`` passes.
3. *Overlap with every training input* (:data:`TRAINING_INPUTS` and ``train-v07.jsonl`` when it exists): the project's text units
   (:func:`robo_jev.data.general_sets.record_units`) of every training record and the 8-word shingles of every training text (state,
   instructions, options) against — MMLU candidates: their units (state = the stem, options ≥ 5 words) and the shingles of the stem and every
   option; physics and M2g moments: the whole normalised state (their template lines are the generators' — shared with training by design for
   the H-set and B1); the M2g filler: its 8-word shingles (reported — a style novelty count). An MMLU candidate with any hit is dropped; a
   physics or M2g moment with a state hit stops the build.
4. *M1g*: the MMLU targets (seeded sample), the physics items (A4 and H-set, one per pair, every eligible pair), distractors, the listings
   at N = 8, 21, 26; every record passes ``m1_problems`` with the cap lifted (and, with N ≤ 8, the standard cap).
5. *Files* (read-only after writing): ``m2g/M2g-<condition>.jsonl``, ``m2g/M2g-all-conditions.jsonl`` (the six conditions together — the
   registered sealed file), ``m1g/M1g.jsonl`` (registered), ``m1g/m1g-items.jsonl``, manifests, ``overlap.json``, an ambiguity-audit sample
   (``audit/``) and ``gates.manifest.json``.

**recheck-v07** — the same scan against ``train-v07.jsonl`` alone; writes ``overlap-v07.json`` (its sha256, hits per gate item, the M1g items
and M2g moments to leave out of the readings, the share of M2g filler shingles found in it). The Spark chain refuses to start without it.

**check** — the Spark chain's gate: the sealed files are unchanged (sha256 as recorded) and ``overlap-v07.json`` was computed on the input
the given v0.7 runs trained on (their ``metrics.json``).

**build-facts** (task v0.7f) — gate M2g-F (:mod:`robo_jev.bench.gates_v07f`) from the sealed ``m2g/M2g-short.jsonl`` (read only) into
``m2gf/``: every record checked (verbatim, ids, questions and labels, ``validate_record``), the pair's added text byte-identical, templates and
status lines balanced per set × scene, the seeded option order of every question equal to M2g-short's (the diagnosis prompt path, CPU),
prompt tokens F vs short, and every fixed template / status piece of 2+ words and every 8-word shingle touching added text against every
training input (``TRAINING_INPUTS`` and ``train-v07.jsonl``; counts only — no training text is kept, printed or written; the page cache of each
input is dropped as it is read). No existing gate file is written (sha256 of each recorded before and after).

**check-facts** (task v0.7f) — the M2g-F chain's gate: the sealed facts file has the sha256 its manifest recorded and was built from the sealed
M2g-short; every record passes its checks again (pure, no tokenizer: the seeded option order through the prompt code's order function); the
overlap scan of the manifest found nothing and read the ``train-v07.jsonl`` on disk and the input the given v0.7 runs trained on. ``check``
is unchanged.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import stat
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

SCRIPT_VERSION = "v07-gates-build-1"
ARTIFACTS = REPO / "artifacts"
OUT = ARTIFACTS / "v07" / "gates"
SETS = {"A4": ARTIFACTS / "v06" / "a4" / "test.jsonl", "Hset": ARTIFACTS / "simgen" / "v04" / "scene-holdout.jsonl",
        "B1": ARTIFACTS / "b1" / "b1-infamily.jsonl"}
MMLU = OUT / "m1g" / "source" / "mmlu-test.jsonl"
IFACE_M1_ITEMS = ARTIFACTS / "iface" / "m1" / "m1-items.jsonl"
IFACE_M2_MANIFEST = ARTIFACTS / "iface" / "m2" / "m2-set.manifest.json"
#: Every training input of the project's trained versions (task iface's list) — assembled files contain their parts.
TRAINING_INPUTS = (
    "simgen/v0b/train-50k.jsonl", "simgen/v01/train-50k.jsonl", "simgen/v01-mix/train-50k-mix.jsonl", "simgen/probes/G1-train-mix.jsonl",
    "simgen/v02/train-50k.jsonl", "simgen/v02/general-train.jsonl", "simgen/v02/train-v02-a.jsonl", "simgen/v02/train-v02-b.jsonl",
    "simgen/v03/train-v03.jsonl", "simgen/v04/train-v04.jsonl", "simgen/v05/train-v05.jsonl", "simgen/v06/train-v06.jsonl",
)
TRAIN_V07 = "simgen/v07/train-v07.jsonl"
SEED = 20261015
#: Never written by this task.
NEVER_WRITE = (ARTIFACTS / "t8", ARTIFACTS / "t9", ARTIFACTS / "b1", ARTIFACTS / "v06", ARTIFACTS / "iface", ARTIFACTS / "simgen")
#: The physics sets of M1g (items from A4 and the H-set; the other one supplies the rest of the distractors).
PHYSICS = {"a4": "A4", "hset": "Hset"}


def _log(message: str) -> None:
    print(f"[v07-gates {dt.datetime.now().strftime('%H:%M:%S')}] {message}", file=sys.stderr, flush=True)


def _git() -> dict[str, Any]:
    try:
        sha = subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "-C", str(REPO), "status", "--porcelain", "--untracked-files=no"], capture_output=True, text=True, check=True).stdout.strip())
        return {"commit": sha, "dirty": dirty}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}


def refuse_write(path: Path) -> Path:
    resolved = Path(path).resolve()
    for place in NEVER_WRITE:
        place = place.resolve()
        if resolved == place or place in resolved.parents:
            raise SystemExit(f"{path}: never written by this task ({place})")
    return path


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> str:
    from robo_jev.decisions.records import file_sha256

    refuse_write(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.chmod(stat.S_IRUSR | stat.S_IWUSR)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return file_sha256(path)


def _write_json(path: Path, payload: Any) -> None:
    refuse_write(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.chmod(stat.S_IRUSR | stat.S_IWUSR)
    path.write_text(json.dumps(payload, indent=1, ensure_ascii=False, default=str) + "\n", encoding="utf-8")


def _seal(paths: list[Path]) -> None:
    for path in paths:
        path.chmod(stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)


def _hash(unit: str) -> str:
    return hashlib.sha256(unit.encode("utf-8")).hexdigest()


def _rel(path: Path) -> str:
    """A path as written into the manifests: relative to the repository root (``artifacts/…``), so it holds in any checkout."""
    try:
        return str(Path(path).relative_to(REPO))
    except ValueError:
        return str(path)


# --------------------------------------------------------------------------
# M2g
# --------------------------------------------------------------------------


def m2g_sample(records: list[dict[str, Any]], pairs: int, *, exclude: set[str], seed: int, set_name: str) -> tuple[list[list[dict[str, Any]]], dict[str, Any]]:
    """`pairs` whole one-fact pairs spread evenly over the scenes (seeded order within each scene; a scene short of its quota gives all
    it has and the others fill in); pairs with an excluded id and unpaired records are left out."""
    from robo_jev.bench.gates_v07 import _rng
    from robo_jev.decisions.records import pair_groups

    by_scene: dict[str, list] = defaultdict(list)
    left_out = Counter()
    for group in pair_groups(records):
        if len(group) != 2:
            left_out["unpaired"] += 1
            continue
        if any(str(r["id"]) in exclude for r in group):
            left_out["used_by_iface_m2"] += 1
            continue
        by_scene[str((group[0].get("meta") or {}).get("scene"))].append(group)
    scenes = sorted(by_scene)
    for scene in scenes:
        by_scene[scene].sort(key=lambda g: str(g[0]["id"]))
        _rng("v07-m2g-sample", set_name, scene, seed).shuffle(by_scene[scene])
    quota = {s: pairs // len(scenes) + (1 if i < pairs % len(scenes) else 0) for i, s in enumerate(scenes)}
    chosen = []
    for s in scenes:
        chosen += by_scene[s][: quota[s]]
    rest = [g for s in scenes for g in by_scene[s][quota[s]:]]
    _rng("v07-m2g-sample-rest", set_name, seed).shuffle(rest)
    chosen += rest[: max(0, pairs - len(chosen))]
    if len(chosen) < pairs:
        raise SystemExit(f"{set_name}: only {len(chosen)} pairs available for {pairs}")
    order = {str(r["id"]): i for i, r in enumerate(records)}
    chosen.sort(key=lambda g: min(order[str(r["id"])] for r in g))
    info = {"available_pairs": {s: len(v) for s, v in by_scene.items()}, "left_out": dict(left_out),
            "chosen_pairs": dict(Counter(str((g[0].get("meta") or {}).get("scene")) for g in chosen))}
    return chosen, info


def build_m2g(sets: dict[str, list[dict[str, Any]]], args: argparse.Namespace) -> dict[str, Any]:
    from robo_jev.bench import gates_v07 as G
    from robo_jev.decisions.records import shuffle_key, validate_record

    iface_ids: set[str] = set()
    if IFACE_M2_MANIFEST.is_file():
        manifest = json.loads(IFACE_M2_MANIFEST.read_text(encoding="utf-8"))
        iface_ids = {str(i) for ids in (manifest.get("ids") or {}).values() for i in ids}
    targets = {"A4": args.a4_pairs, "Hset": args.hset_pairs, "B1": args.b1_pairs}
    groups, selection = {}, {}
    for name, records in sets.items():
        groups[name], selection[name] = m2g_sample(records, targets[name], exclude=iface_ids, seed=args.seed, set_name=name)
    strata = defaultdict(list)
    for name, chosen in groups.items():
        for group in chosen:
            strata[f"{name}:{(group[0].get('meta') or {}).get('scene')}"].append(G.M2G_PREFIX + shuffle_key(group[0]))
    style_of = G.assign_styles(strata, seed=args.seed)
    files: dict[str, list[dict[str, Any]]] = {cond: [] for cond in G.CONDITIONS}
    problems: dict[str, list[str]] = {}
    lengths: dict[str, list[int]] = defaultdict(list)
    filler_lines: set[str] = set()
    style_table: dict[str, Counter] = defaultdict(Counter)
    for name, chosen in groups.items():
        for group in chosen:
            gated = [G.gate_record(r, source=name, style="?") for r in group]
            key = shuffle_key(gated[0])
            if key != shuffle_key(gated[1]) or key != G.M2G_PREFIX + shuffle_key(group[0]):
                raise SystemExit(f"{group[0]['id']}: the pair's order key is not shared after the prefix")
            style = style_of[key]
            for g in gated:
                g["meta"]["gate"]["style"] = style
            reference = max(len(str(r["request"]["state"]).strip()) for r in group)
            fillers: dict[str, set[str]] = defaultdict(set)
            for original, g in zip(group, gated):
                style_table[style].update(_truth_classes(g))
                found = validate_record(g)
                if g["request"]["questions"] != {n: original["request"]["questions"][n] for n in G.call_view_questions(original)}:
                    found.append("questions differ from the original's call-view questions")
                if g["labels"] != {n: original["labels"][n] for n in G.call_view_questions(original)}:
                    found.append("labels differ from the original's")
                files["short"].append(g)
                sheets = {}
                for cond in G.CONDITIONS[1:]:
                    length, _, tone = cond[1:].partition("-")
                    tone = tone or "neutral"
                    w = G.wrap_record(g, length=length, tone=tone, reference_length=reference)
                    sheet = w["request"]["state"]
                    found += [f"{cond}: {p}" for p in G.sheet_problems(sheet, g["request"]["state"], style=style, key=key, tone=tone)]
                    found += [f"{cond}: {p}" for p in validate_record(w)]
                    if {k: v for k, v in w.items() if k != "request"} != {k: v for k, v in g.items() if k != "request"} \
                            or w["request"]["questions"] != g["request"]["questions"]:
                        found.append(f"{cond}: id, meta, questions or labels changed")
                    files[cond].append(w)
                    sheets[cond] = sheet
                    lengths[f"{style}:{cond}"].append(len(sheet))
                    filler = sheet.replace(g["request"]["state"].strip(), "<STATE>")
                    fillers[cond].add(filler)
                    filler_lines.update(line for line in filler.splitlines() if line.strip() and line.strip() != "<STATE>")
                skeletons = {G.tone_skeleton(sheets[c], key, t) for c, t in (("w4k", "neutral"), ("w4k-reassuring", "reassuring"), ("w4k-alarming", "alarming"))}
                if len(skeletons) != 1:
                    found.append("the tone versions differ outside the tone sentences")
                for short, long in (("w2k", "w4k"), ("w4k", "w8k")):  # a longer sheet extends the shorter one (sentence by sentence)
                    missing = [unit for unit in G.sentences(sheets[short]) if unit not in sheets[long]]
                    if missing:
                        found.append(f"{short} → {long}: {len(missing)} sentences of the shorter sheet are not in the longer one ({missing[:2]})")
                if found:
                    problems[g["id"]] = found
            for cond, variants in fillers.items():
                if len(variants) != 1:
                    problems[f"pair {key} {cond}"] = ["the two records of the pair do not share their filler"]
    if problems:
        raise SystemExit(f"M2g: {len(problems)} records fail their checks; first {list(problems.items())[:2]}")
    ids = [r["id"] for r in files["short"]]
    if len(set(ids)) != len(ids) or any([r["id"] for r in files[c]] != ids for c in G.CONDITIONS):
        raise SystemExit("M2g: ids are not unique or differ between the condition files")
    return {"files": files, "selection": selection, "style_of": style_of, "lengths": lengths, "filler_lines": filler_lines,
            "style_truth_classes": {s: dict(c) for s, c in style_table.items()}, "targets": targets, "iface_m2_ids_left_out": len(iface_ids)}


def _truth_classes(record: dict[str, Any]) -> list[str]:
    from robo_jev.bench.system1 import _scalar, band

    out = []
    for name in record["request"]["questions"]:
        y = _scalar((record.get("labels") or {}).get(name))
        robust = (((record.get("meta") or {}).get("robustness") or {}).get(name) or {}).get("robust") is True
        level = band(y) if y is not None else None
        out.append("clear_safe" if robust and level == "low" else "clear_risky" if robust and level == "high" else "unclear")
    return out


# --------------------------------------------------------------------------
# Overlap scan
# --------------------------------------------------------------------------


def _texts(record: dict[str, Any]) -> list[str]:
    request = record.get("request") or {}
    out = [str(request.get("state") or "")]
    for question in (request.get("questions") or {}).values():
        out.append(str(question.get("instructions") or ""))
        criteria = question.get("criteria")
        if isinstance(criteria, dict):
            out += [str(v) for v in criteria.values()]
        elif isinstance(criteria, list):
            out += [str(v) for v in criteria]
    return out


def scan(files: list[Path], units: dict[str, set[str]], grams: dict[str, set[str]], filler_grams: set[str]) -> dict[str, Any]:
    """Every training record's text units against the candidates' unit hashes, every training text's 8-word shingles against the
    candidates' shingles and the M2g filler's (texts seen before are skipped). Hits per candidate (with the file) and per file."""
    from robo_jev.bench.iface import shingles
    from robo_jev.data.general_sets import record_units
    from robo_jev.decisions.records import file_sha256

    hits: dict[str, set[str]] = defaultdict(set)
    filler_found: set[str] = set()
    per_file: dict[str, Any] = {}
    seen: set[int] = set()
    for path in files:
        started, records, unit_hits, gram_hits, filler_hits = time.perf_counter(), 0, 0, 0, 0
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                record = json.loads(line)
                records += 1
                for unit in record_units(record):
                    owners = units.get(_hash(unit))
                    if owners:
                        unit_hits += 1
                        for owner in owners:
                            hits[owner].add(f"unit:{path.name}")
                for text in _texts(record):
                    h = hash(text)
                    if h in seen:
                        continue
                    seen.add(h)
                    for gram in shingles(text):
                        owners = grams.get(gram)
                        if owners:
                            gram_hits += 1
                            for owner in owners:
                                hits[owner].add(f"shingle:{path.name}")
                        if gram in filler_grams:
                            filler_hits += 1
                            filler_found.add(gram)
        try:
            rel = str(path.relative_to(ARTIFACTS))
        except ValueError:
            rel = str(path)
        per_file[rel] = {"records": records, "unit_hits": unit_hits, "shingle_hits": gram_hits, "filler_shingle_hits": filler_hits,
                         "sha256": file_sha256(path), "seconds": round(time.perf_counter() - started, 1)}
        _log(f"scan {path.name}: {records} records, unit hits {unit_hits}, shingle hits {gram_hits}, filler shingle hits {filler_hits} "
             f"({per_file[rel]['seconds']} s)")
    return {"hits": {k: sorted(v) for k, v in hits.items()}, "files": per_file, "filler_shingles_found": sorted(filler_found),
            "filler_shingles": len(filler_grams)}


def candidates(mmlu: list[dict[str, Any]], states: dict[str, str]) -> tuple[dict[str, set[str]], dict[str, set[str]]]:
    """(unit hash → candidates) and (8-word shingle → MMLU candidates): MMLU items by their units and the shingles of their stem and
    options; physics and M2g moments by their whole normalised state."""
    from robo_jev.bench.iface import shingles
    from robo_jev.data.general_sets import normalize, text_units

    units: dict[str, set[str]] = defaultdict(set)
    grams: dict[str, set[str]] = defaultdict(set)
    for item in mmlu:
        for unit in text_units(item["state"], [item["correct"], *item["own"]]):
            units[_hash(unit)].add(item["item"])
        for text in [item["stem"], item["correct"], *item["own"]]:
            if len(text.split()) >= 5:
                for gram in shingles(text):
                    grams[gram].add(item["item"])
    for name, state in states.items():
        units[_hash(normalize(state))].add(name)
    return dict(units), dict(grams)


def filler_shingles(lines: set[str]) -> set[str]:
    from robo_jev.bench.iface import shingles

    out: set[str] = set()
    for line in lines:
        if len(line.split()) >= 8:
            out |= shingles(line)
    return out


# --------------------------------------------------------------------------
# M1g
# --------------------------------------------------------------------------


def build_m1g(sets: dict[str, list[dict[str, Any]]], mmlu_clean: list[dict[str, Any]], physics_clean: dict[str, list[dict[str, Any]]],
              args: argparse.Namespace) -> dict[str, Any]:
    from robo_jev.bench import gates_v07 as G
    from robo_jev.bench import iface
    from robo_jev.bench.gates_v07 import _rng
    from robo_jev.decisions.records import OPTION_RANGE

    pool = sorted((i for i in mmlu_clean if i["target"]), key=lambda x: x["item"])
    _rng("v07-m1g-mmlu", args.seed).shuffle(pool)
    if len(pool) < args.mmlu_items:
        raise SystemExit(f"MMLU: {len(pool)} clean targets < {args.mmlu_items}")
    chosen_general = pool[: args.mmlu_items]
    chosen_physics = []
    for source, items in physics_clean.items():
        groups = len({i["group"] for i in items})
        count = groups if args.physics_per_set is None else min(groups, args.physics_per_set)
        chosen_physics += iface.select_physics(items, count, seed=args.seed)
    items = chosen_general + chosen_physics
    _rng("v07-m1g-rank", args.seed).shuffle(items)
    index = iface.general_index(mmlu_clean)
    pools = {source: iface.physics_pool(sets[name]) for source, name in PHYSICS.items()}
    lines, records = [], []
    for rank, item in enumerate(items):
        if item["source"] == G.GENERAL_SOURCE:
            drawn = iface.draw_general(item, index, count=max(G.M1G_LEVELS) - 4, seed=args.seed)
        else:
            other = "hset" if item["source"] == "a4" else "a4"
            drawn = G.draw_physics(item, pools[item["source"]], pools[other], count=max(G.M1G_LEVELS) - 3, seed=args.seed)
        listings = G.expand_item(item, drawn, rank=rank, index=rank, seed=args.seed)
        records += listings
        lines.append({"item": item["item"], "source": item["source"], "rank": rank, "state": item["state"], "instructions": item["instructions"],
                      "correct": item["correct"], "own": item["own"], "drawn": drawn,
                      **({"record": item["record"], "question": item["question"], "scene": item["scene"], "group": item["group"]} if "record" in item
                         else {"stem": item["stem"], "subject": item["subject"]})})
    problems = {r["id"]: p for r in records if (p := iface.m1_problems(r, lift_cap=True))}
    if problems:
        raise SystemExit(f"M1g records fail validation with the cap lifted: {list(problems.items())[:3]}")
    capped = Counter((r["meta"]["n"] <= OPTION_RANGE[1], not iface.m1_problems(r, lift_cap=False)) for r in records)
    if capped.get((True, False)) or capped.get((False, True)):
        raise SystemExit(f"the standard cap does not behave as expected on the M1g records: {dict(capped)}")
    by_item = defaultdict(list)
    for r in records:
        by_item[r["meta"]["item"]].append(r)
    for item, rs in by_item.items():  # the same item and answer at every N, the set nested across N
        answers = {r["request"]["questions"][iface.QUESTION]["criteria"][r["meta"]["correct"]] for r in rs}
        option_sets = {r["meta"]["n"]: set(r["request"]["questions"][iface.QUESTION]["criteria"].values()) for r in rs}
        if len(answers) != 1 or not option_sets[8] <= option_sets[21] <= option_sets[26]:
            raise SystemExit(f"{item}: the answer differs across N or the option sets are not nested")
    return {"records": records, "items": lines, "capped": dict((str(k), v) for k, v in capped.items()),
            "composition": {"items_by_source": dict(Counter(i["source"] for i in items)),
                            "physics_by_scene": dict(Counter(f"{i['source']}:{i['scene']}" for i in items if "scene" in i)),
                            "physics_answers": dict(Counter(i["correct"][0] for i in items if "scene" in i)),
                            "mmlu_subjects": dict(Counter(i["subject"] for i in items if i["source"] == G.GENERAL_SOURCE).most_common()),
                            "records_by_n": dict(Counter(str(r["meta"]["n"]) for r in records)),
                            "correct_letter_by_n": {str(n): dict(sorted(Counter(r["meta"]["letter"] for r in records if r["meta"]["n"] == n).items()))
                                                    for n in G.M1G_LEVELS}}}


def audit_sample(m1g: dict[str, Any], out: Path, *, general: int, physics: int, seed: int) -> dict[str, Any]:
    """A seeded sample of items at N = 26 written as a sheet for the ambiguity audit (question, the right option, the other 25)."""
    from robo_jev.bench.gates_v07 import GENERAL_SOURCE, _rng
    from robo_jev.bench.iface import QUESTION

    by_item = {}
    for r in m1g["records"]:
        if r["meta"]["n"] == 26 and r["meta"]["listing"] == 0:
            by_item[r["meta"]["item"]] = r
    names = sorted(by_item)
    rng = _rng("v07-m1g-audit", seed)
    gen = [n for n in names if by_item[n]["meta"]["source"] == GENERAL_SOURCE]
    phy = [n for n in names if by_item[n]["meta"]["source"] != GENERAL_SOURCE]
    picked = rng.sample(gen, min(general, len(gen))) + rng.sample(phy, min(physics, len(phy)))
    lines = ["# M1g ambiguity audit sample (N = 26, listing 0)", "",
             "For each item: is any of the 25 other options also a right (or equally good) answer? Mark it in audit-findings.", ""]
    for i, name in enumerate(picked, 1):
        r = by_item[name]
        q = r["request"]["questions"][QUESTION]
        right = q["criteria"][r["meta"]["correct"]]
        lines += [f"## {i}. {name} ({r['meta'].get('subject') or r['meta'].get('scene')})", "", "State: " + r["request"]["state"].replace("\n", " "), "",
                  "Question: " + q["instructions"], "", f"**Right:** {right}", ""]
        lines += [f"- {text}" for key, text in q["criteria"].items() if key != r["meta"]["correct"]]
        lines.append("")
    refuse_write(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"file": _rel(out), "items": picked}


# --------------------------------------------------------------------------
# Tokens
# --------------------------------------------------------------------------


def prompt_tokens(records: list[dict[str, Any]], *, tokenizer: Any, orders: str = "seeded", seed: int | None = 1) -> dict[str, Any]:
    from robo_jev.decisions.prompt import record_prompts

    lengths = sorted(len(p["tokens"]) for r in records for p in record_prompts(r, tokenizer, seed=seed, fmt="chat", mode="direct", orders=orders))
    if not lengths:
        return {"prompts": 0}
    return {"prompts": len(lengths), "tokens": sum(lengths), "mean": round(sum(lengths) / len(lengths), 1), "max": lengths[-1]}


#: GPU seconds per prompt token on the Spark (GB10) at batch 1: the v0.7b smoke test measured 0.209 ms (M1g, 403 tokens per prompt) and
#: 0.203 ms (M2g 8k, 3,405 tokens per prompt) for a LoRA model, warm-up included (task iface's long runs: ≈ 0.185 ms); the base runs at ≈ 0.6
#: of a LoRA model (task iface: 43.5 vs 73 ms per M1 prompt).
SECONDS_PER_TOKEN = {"lora": 0.20e-3, "base": 0.12e-3}


# --------------------------------------------------------------------------
# Commands
# --------------------------------------------------------------------------


def cmd_build(args: argparse.Namespace) -> int:
    from robo_jev.bench import gates_v07 as G
    from robo_jev.bench import iface
    from robo_jev.data.general_sets import normalize
    from robo_jev.decisions.records import file_sha256, read_records
    from robo_jev.model.tokenizer import available_tokenizer, describe_tokenizer, load_tokenizer

    started = time.perf_counter()
    out = refuse_write(args.out)
    sets = {name: read_records(path) for name, path in SETS.items()}
    sha = {name: file_sha256(path) for name, path in SETS.items()}
    _log("sets read: " + ", ".join(f"{n} {len(r)}" for n, r in sets.items()))
    m2g = build_m2g(sets, args)
    _log(f"M2g: {len(m2g['files']['short'])} records × {len(G.CONDITIONS)} conditions, every sheet checked")
    rows = read_records(MMLU)  # line by line on "\n" only (the texts hold other Unicode line separators)
    mmlu_all, mmlu_dropped = G.mmlu_items(rows)
    used_m1 = read_records(IFACE_M1_ITEMS) if IFACE_M1_ITEMS.is_file() else []
    used_groups = {u["group"] for u in used_m1 if u.get("source") == "hset"}
    physics_all, physics_counts = {}, {}
    for source, name in PHYSICS.items():
        items, counts = iface.physics_items(sets[name], source=source)
        if source == "hset":
            counts["pair_groups_used_by_iface_m1"] = len({i["group"] for i in items if i["group"] in used_groups})
            items = [i for i in items if i["group"] not in used_groups]
        physics_all[source], physics_counts[source] = items, counts
    states = {f"m2g:{r['id']}": r["request"]["state"] for r in m2g["files"]["short"]}
    states.update({i["item"]: i["state"] for items in physics_all.values() for i in items})
    units, grams = candidates(mmlu_all, states)
    fgrams = filler_shingles(m2g["filler_lines"])
    files = [ARTIFACTS / f for f in TRAINING_INPUTS]
    missing = [str(f) for f in files if not f.is_file()]
    if missing:
        raise SystemExit(f"training inputs missing: {missing}")
    v07 = ARTIFACTS / TRAIN_V07
    if v07.is_file():
        files.append(v07)
    overlap = {"hits": {}, "files": {}, "filler_shingles_found": [], "filler_shingles": len(fgrams), "skipped": True} if args.skip_overlap \
        else scan(files, units, grams, fgrams)
    hits = overlap["hits"]
    bad_states = sorted(k for k in hits if k.startswith("m2g:") or ":" in k and not k.startswith("mmlu:"))
    if bad_states:
        raise SystemExit(f"{len(bad_states)} M2g / physics moments share a whole state with a training input (first {bad_states[:3]})")
    mmlu_clean = [i for i in mmlu_all if i["item"] not in hits]
    _log(f"MMLU {len(mmlu_all)} candidates ({sum(i['target'] for i in mmlu_all)} targets), {len(mmlu_all) - len(mmlu_clean)} with an overlap hit")
    m1g = build_m1g(sets, mmlu_clean, physics_all, args)
    _log(f"M1g: {len(m1g['items'])} items, {len(m1g['records'])} listings")
    # files
    m2_dir, m1_dir = out / "m2g", out / "m1g"
    written = []
    m2_files = {}
    for cond in G.CONDITIONS:
        path = m2_dir / f"M2g-{cond}.jsonl"
        m2_files[cond] = {"path": _rel(path), "sha256": _write_jsonl(path, m2g["files"][cond]), "records": len(m2g["files"][cond])}
        written.append(path)
    registry = m2_dir / "M2g-all-conditions.jsonl"
    m2_registry_sha = _write_jsonl(registry, [r for cond in G.CONDITIONS for r in m2g["files"][cond]])
    written.append(registry)
    m1_path = m1_dir / "M1g.jsonl"
    m1_sha = _write_jsonl(m1_path, m1g["records"])
    items_sha = _write_jsonl(m1_dir / "m1g-items.jsonl", m1g["items"])
    written += [m1_path, m1_dir / "m1g-items.jsonl"]
    for style in G.STYLES:  # one example per style and condition
        example = next(r for r in m2g["files"]["short"] if r["meta"]["gate"]["style"] == style)
        for cond in G.CONDITIONS[1:]:
            w = next(x for x in m2g["files"][cond] if x["id"] == example["id"])
            path = m2_dir / "examples" / f"{style}-{cond}-{example['id']}.txt"
            refuse_write(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(w["request"]["state"] + "\n", encoding="utf-8")
    tokenizer_id = available_tokenizer()[0]
    tokenizer = load_tokenizer(tokenizer_id)
    m2_tokens = {cond: prompt_tokens(m2g["files"][cond], tokenizer=tokenizer) for cond in G.CONDITIONS}
    m1_tokens = {str(n): prompt_tokens([r for r in m1g["records"] if r["meta"]["n"] == n], tokenizer=tokenizer, seed=None) for n in G.M1G_LEVELS}
    gpu = {"seconds_per_token": SECONDS_PER_TOKEN,
           "M2g_per_model_s": {kind: round(sum(t["tokens"] for t in m2_tokens.values()) * rate) for kind, rate in SECONDS_PER_TOKEN.items()},
           "M1g_per_model_s": {kind: round(sum(t["tokens"] for t in m1_tokens.values()) * rate) for kind, rate in SECONDS_PER_TOKEN.items()}}
    audit = audit_sample(m1g, out / "audit" / "m1g-audit-sample.md", general=args.audit_general, physics=args.audit_physics, seed=args.seed)
    lengths = {k: {"min": min(v), "median": sorted(v)[len(v) // 2], "max": max(v)} for k, v in sorted(m2g["lengths"].items())}
    m2_manifest = {
        "kind": "v07-m2g", "script_version": SCRIPT_VERSION, "git": _git(), "seed": args.seed, "set": G.M2G_SET, "prefix": G.M2G_PREFIX,
        "sources": {name: {"file": _rel(SETS[name]), "sha256": sha[name], "records": len(sets[name])} for name in SETS},
        "sample_rule": "per set the target number of whole one-fact pairs spread evenly over its scenes (seeded order within each scene, seed "
                       f"{args.seed}; a scene short of its quota gives all it has and the others fill in); pairs task iface's M2 wrapped and unpaired "
                       "records left out", "targets_pairs": m2g["targets"], "selection": m2g["selection"],
        "iface_m2_ids_left_out": m2g["iface_m2_ids_left_out"], "conditions": list(G.CONDITIONS), "files": m2_files,
        "registry_file": {"path": _rel(registry), "sha256": m2_registry_sha, "note": "the six condition files together (ids repeat across conditions)"},
        "records": len(m2g["files"]["short"]), "questions": sum(len(r["request"]["questions"]) for r in m2g["files"]["short"]),
        "by_set": dict(Counter(r["meta"]["source"]["set"] for r in m2g["files"]["short"])),
        "styles": dict(Counter(r["meta"]["gate"]["style"] for r in m2g["files"]["short"])),
        "styles_by_set": {s: dict(Counter(r["meta"]["gate"]["style"] for r in m2g["files"]["short"] if r["meta"]["source"]["set"] == s)) for s in SETS},
        "truth_classes_by_style": m2g["style_truth_classes"], "lengths_chars": lengths, "prompt_tokens": m2_tokens,
        "checks": ["every sheet: the original state exactly once as the whole report section; every original sentence verbatim, in order, inside it "
                   "and nowhere else; no fact word in the filler; no tone word outside the tone sentences; the tone sentences are the pair's",
                   "both records of every pair share their filler (they differ in their labels)", "the three 4k tone versions are identical except for "
                   "the three tone sentences", "every sentence (line part) of the 2k sheet is in the 4k sheet and every one of the 4k sheet in the 8k sheet",
                   "ids, meta, questions and labels identical in all six files; questions and labels equal to the original's call-view questions",
                   "validate_record passes on every record"],
        "fact_words": sorted(G.FACT_WORDS), "tone_words": sorted(G.TONE_WORDS), "tone_banks": G.TONE_BANKS,
        "overlap": {"moments_with_a_state_hit": 0, "filler_shingles": overlap["filler_shingles"], "filler_shingles_found_in_training": len(overlap["filler_shingles_found"]),
                    "filler_shingles_found_examples": overlap["filler_shingles_found"][:20], "skipped": bool(overlap.get("skipped"))},
        "created": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
    }
    _write_json(m2_dir / "m2g.manifest.json", m2_manifest)
    fetch = json.loads((MMLU.parent / "mmlu-test.fetch.json").read_text(encoding="utf-8"))
    m1_manifest = {
        "kind": "v07-m1g", "script_version": SCRIPT_VERSION, "git": _git(), "seed": args.seed, "set": G.M1G_SET, "prefix": G.M1G_PREFIX,
        "file": {"path": _rel(m1_path), "sha256": m1_sha, "records": len(m1g["records"])},
        "items_file": {"path": _rel(m1_dir / "m1g-items.jsonl"), "sha256": items_sha, "items": len(m1g["items"])},
        "design": {"levels": list(G.M1G_LEVELS), "listings_per_level": {str(k): v for k, v in G.M1G_LISTINGS.items()}, "buckets": [list(b) for b in iface.BUCKETS],
                   "late_min_n": iface.LATE_MIN_N, "instruction": iface.ARC_INSTRUCTION, "physics_min_top": iface.PHYSICS_MIN_TOP,
                   "orders": "controlled positions; the prompt code shows the request order (seed None) — no reshuffle",
                   "cap": {"lifted_in": "scripts/iface_m1_score.py --lift-option-cap", "check": m1g["capped"]}},
        "source": {"MMLU": {"repo": fetch.get("repo"), "revision": fetch.get("revision"), "file": fetch.get("file"), "licence": fetch.get("license"),
                            "licence_evidence": {"card_metadata": fetch.get("card", {}).get("license_metadata"), "card_text": fetch.get("card", {}).get("license_text")},
                            "citation": fetch.get("citation"), "split": "test (config all)", "rows": len(rows), "candidates": len(mmlu_all),
                            "targets": sum(i["target"] for i in mmlu_all), "dropped_by_filter": mmlu_dropped,
                            "dropped_by_overlap": len(mmlu_all) - len(mmlu_clean),
                            "why": "used in no training input and not in M1; not ARC, not OpenBookQA, not a general-train source; MIT licence on the card"},
                   **{source: {"file": _rel(SETS[name]), "sha256": sha[name], "direction_items": physics_counts[source],
                               "eligible_after_exclusions": len(physics_all[source]),
                               "pair_groups": len({i['group'] for i in physics_all[source]})} for source, name in PHYSICS.items()}},
        "composition": m1g["composition"], "prompt_tokens_by_n": m1_tokens, "audit": audit,
        "overlap": {"method": "text units (record_units of every training record vs the candidates' units) and 8-word shingles of every training "
                              "text vs the MMLU candidates' stem and options; physics: whole states", "dropped": len(mmlu_all) - len(mmlu_clean),
                    "chosen_items_with_hits": [i["item"] for i in m1g["items"] if i["item"] in hits]},
        "created": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
    }
    if m1_manifest["overlap"]["chosen_items_with_hits"]:
        raise SystemExit("a chosen M1g item overlaps a training input")
    _write_json(m1_dir / "m1g.manifest.json", m1_manifest)
    _write_json(out / "overlap.json", {"kind": "v07-gates-overlap", "script_version": SCRIPT_VERSION, "files": overlap["files"],
                                       "train_v07": {"path": _rel(v07), "present": v07.is_file()},
                                       "hits": hits, "filler_shingles": overlap["filler_shingles"],
                                       "filler_shingles_found": overlap["filler_shingles_found"], "skipped": bool(overlap.get("skipped")),
                                       "created": dt.datetime.now().astimezone().isoformat(timespec="seconds")})
    top = {"kind": "v07-gates", "script_version": SCRIPT_VERSION, "git": _git(), "seed": args.seed,
           "v07_models_or_predictions_at_build": sorted(str(p) for p in (ARTIFACTS / "reports" / "v07").rglob("*.predictions.jsonl"))
           if (ARTIFACTS / "reports" / "v07").is_dir() else [],
           "train_v07_present_at_build": v07.is_file(), "train_v07_sha256": file_sha256(v07) if v07.is_file() else None,
           "train_v07_use": "read in code only, by the overlap scan (counts and hashes written; no text printed or stored); the design — styles, "
                            "banks, sizes, seeds — does not depend on it",
           "M2g": {"manifest": _rel(m2_dir / "m2g.manifest.json"), "files": m2_files, "registry": _rel(registry)},
           "M1g": {"manifest": _rel(m1_dir / "m1g.manifest.json"), "file": _rel(m1_path), "sha256": m1_sha},
           "gpu_estimate": gpu, "tokenizer": describe_tokenizer(tokenizer_id), "seconds": round(time.perf_counter() - started, 1),
           "created": dt.datetime.now().astimezone().isoformat(timespec="seconds")}
    _write_json(out / "gates.manifest.json", top)
    _seal(written)
    _log(f"done in {top['seconds']} s · GPU estimate per LoRA model: M2g {gpu['M2g_per_model_s']['lora']} s, M1g {gpu['M1g_per_model_s']['lora']} s")
    return 0


def cmd_recheck(args: argparse.Namespace) -> int:
    """The scan against train-v07.jsonl alone (module docstring); writes overlap-v07.json."""
    from robo_jev.bench import gates_v07 as G
    from robo_jev.decisions.records import file_sha256, read_records

    v07 = args.train_v07
    if not v07.is_file():
        raise SystemExit(f"{v07}: not there yet — run this once v0.7's training input exists")
    out = refuse_write(args.out_dir)
    m1_items = read_records(out / "m1g" / "m1g-items.jsonl")
    m2_short = read_records(out / "m2g" / "M2g-short.jsonl")
    m2_all = read_records(out / "m2g" / "M2g-all-conditions.jsonl")
    general = [{"item": i["item"], "state": i["state"], "stem": i["stem"], "correct": i["correct"], "own": list(i["own"]) + [str(d) for d in i["drawn"]]}
               for i in m1_items if i["source"] == G.GENERAL_SOURCE]
    states = {f"m2g:{r['id']}": r["request"]["state"] for r in m2_short}
    states.update({i["item"]: i["state"] for i in m1_items if i["source"] != G.GENERAL_SOURCE})
    units, grams = candidates(general, states)
    lines = {line for r in m2_all for line in r["request"]["state"].splitlines() if line.strip()}
    short_lines = {line for r in m2_short for line in r["request"]["state"].splitlines() if line.strip()}
    fgrams = filler_shingles(lines - short_lines)
    found = scan([v07], units, grams, fgrams)
    hits = found["hits"]
    payload = {"kind": "v07-gates-overlap-v07", "script_version": SCRIPT_VERSION, "git": _git(), "train_v07": {"path": _rel(v07), "sha256": file_sha256(v07)},
               "files": found["files"], "exclude": {"M1g_items": sorted(k for k in hits if any(k == i["item"] for i in m1_items)),
                                                    "M2g_ids": sorted(k[len("m2g:"):] for k in hits if k.startswith("m2g:"))},
               "hits": hits, "filler_shingles": found["filler_shingles"], "filler_shingles_found": len(found["filler_shingles_found"]),
               "filler_share_found": (len(found["filler_shingles_found"]) / found["filler_shingles"]) if found["filler_shingles"] else None,
               "rule": "pre-declared (task v0.7b, before any v0.7 data): an M1g item with any unit or shingle hit and an M2g moment with a whole-state "
                       "hit in train-v07 are left out of every reading; the filler share is reported (style novelty)",
               "created": dt.datetime.now().astimezone().isoformat(timespec="seconds")}
    _write_json(out / "overlap-v07.json", payload)
    _log(f"train-v07: exclude {len(payload['exclude']['M1g_items'])} M1g items, {len(payload['exclude']['M2g_ids'])} M2g moments; filler shingles "
         f"found {payload['filler_shingles_found']} of {payload['filler_shingles']}")
    return 0


def cmd_check(args: argparse.Namespace) -> int:
    """Before the gate is opened (the Spark chain): every sealed gate file has the sha256 its manifest recorded; ``overlap-v07.json`` exists
    and was computed on the training input each given v0.7 run trained on (``metrics.json`` → ``manifest.data.file_sha256``) and on the
    ``train-v07.jsonl`` on disk. Prints the problems and returns 2 if there is any."""
    from robo_jev.decisions.records import file_sha256

    out = args.out_dir
    problems = []
    m2 = json.loads((out / "m2g" / "m2g.manifest.json").read_text(encoding="utf-8"))
    m1 = json.loads((out / "m1g" / "m1g.manifest.json").read_text(encoding="utf-8"))
    files = [(e["path"], e["sha256"]) for e in m2["files"].values()] + [(m2["registry_file"]["path"], m2["registry_file"]["sha256"]),
                                                                         (m1["file"]["path"], m1["file"]["sha256"]), (m1["items_file"]["path"], m1["items_file"]["sha256"])]
    for rel, want in files:
        path = REPO / rel
        if not path.is_file() or file_sha256(path) != want:
            problems.append(f"{rel}: missing or not the sealed file (sha256 {want[:12]})")
    recheck = out / "overlap-v07.json"
    trained: dict[str, str | None] = {}
    if not recheck.is_file():
        problems.append(f"{_rel(recheck)} is missing — run recheck-v07 on v0.7's training input")
    else:
        sha = json.loads(recheck.read_text(encoding="utf-8"))["train_v07"]["sha256"]
        on_disk = ARTIFACTS / TRAIN_V07
        if on_disk.is_file() and file_sha256(on_disk) != sha:
            problems.append(f"{TRAIN_V07} changed since the recheck ({sha[:12]}) — run recheck-v07 again")
        for run in args.runs:
            metrics = Path(run) / "metrics.json"
            got = None
            if metrics.is_file():
                got = ((json.loads(metrics.read_text(encoding="utf-8")).get("manifest") or {}).get("data") or {}).get("file_sha256")
            trained[str(run)] = got
            if got != sha:
                problems.append(f"{run}: trained on {str(got)[:12]}, the recheck read {sha[:12]} — run recheck-v07 on the input the run used")
    print(json.dumps({"problems": problems, "trained_inputs": trained}, indent=1))
    return 2 if problems else 0


# --------------------------------------------------------------------------
# Task v0.7f — M2g-F: the M2g moments with the facts moved into each question's instructions
# --------------------------------------------------------------------------

FACTS_SCRIPT_VERSION = "v07f-facts-build-1"
#: The sealed registry name of the facts file (:data:`robo_jev.data.sealed.SEALED_RECORD_FILES`).
FACTS_REGISTRY = "v07-m2g-facts"
#: The M2g-F scoring (``artifacts/scratch/v07f/score_facts.sh``): M2g's command — the standard diagnosis path with seeded orders, shuffle
#: seed 1, chat, direct, v0c-p3, batch 1 — for these five models.
FACTS_SCORING = {"script": "scripts/diagnose_decisions.py", "orders": "seeded", "shuffle_seed": 1, "format": "chat", "mode": "direct",
                 "prompt_version": "v0c-p3", "batch": 1}
FACTS_MODELS = ("v07s0-2b", "v07s1-2b", "v06s0-2b", "v06s1-2b", "base-2b")
#: A model load on the Spark (task v0.7b's smoke test: 21 s).
LOAD_SECONDS = 21.0
#: The prompt fields that fix what the model is shown as options and in which order (the order proof compares them).
ORDER_FIELDS = ("question", "type", "listing", "n_listings", "options", "option_lines", "values", "codes", "code_ids")
#: Common phrases the scan must find in the training inputs (a sensitivity control; reported, never deciding).
CONTROL_PIECES = ("of the", "the robot")


def _drop_cache(handle: Any) -> None:
    """Drop the page cache of an open file (the Spark chain's next GPU unit needs free unified memory)."""
    try:
        os.posix_fadvise(handle.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
    except (AttributeError, OSError):
        pass


def sha256_dropped(path: Path, *, drop_every: int = 256 << 20) -> str:
    """sha256 of a file, dropping its page cache as it is read."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        since = 0
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
            since += len(block)
            if since >= drop_every:
                _drop_cache(handle)
                since = 0
        _drop_cache(handle)
    return digest.hexdigest()


def facts_scan(files: list[Path], pieces: list[str], grams: set[str], *, chunk_chars: int = 48_000_000, drop_every: int = 256 << 20) -> dict[str, Any]:
    """The training inputs against M2g-F's added text — counts only (no training text is kept beyond the current chunk, printed or written).

    * `pieces`: every training record's state and question instructions (each distinct text once, whitespace runs collapsed to one space)
      — per piece the number of occurrences, exact (case-sensitive, with its punctuation) and case-folded.
    * `grams`: the 8-word shingles (:func:`robo_jev.bench.iface.shingles`) of every distinct training text — state, instructions, options
      (:func:`_texts`, as task v0.7b's scan) — per gram the number of texts holding it.

    Each file is read once (its sha256 computed on the way) and its page cache dropped as it is read."""
    from robo_jev.bench.iface import shingles

    exact = {p: 0 for p in pieces}
    folded = {p: 0 for p in pieces}
    lowered = [(p, p.lower()) for p in pieces]
    gram_hits: Counter = Counter()
    seen_pieces: set[int] = set()
    seen_grams: set[int] = set()
    buffer: list[str] = []
    size = 0

    def flush() -> None:
        nonlocal buffer, size
        if buffer:
            chunk = "\x00".join(buffer)
            low = chunk.lower()
            for piece, folded_piece in lowered:
                exact[piece] += chunk.count(piece)
                folded[piece] += low.count(folded_piece)
        buffer, size = [], 0

    per_file: dict[str, Any] = {}
    for path in files:
        started, records = time.perf_counter(), 0
        digest = hashlib.sha256()
        with Path(path).open("rb") as handle:
            since = 0
            for raw in handle:
                digest.update(raw)
                since += len(raw)
                if since >= drop_every:
                    _drop_cache(handle)
                    since = 0
                if not raw.strip():
                    continue
                record = json.loads(raw)
                records += 1
                request = record.get("request") or {}
                own = [str(request.get("state") or "")] + [str((q or {}).get("instructions") or "") for q in (request.get("questions") or {}).values()]
                for text in own:
                    h = hash(text)
                    if h in seen_pieces:
                        continue
                    seen_pieces.add(h)
                    flat = " ".join(text.split())
                    buffer.append(flat)
                    size += len(flat)
                    if size >= chunk_chars:
                        flush()
                if grams:
                    for text in _texts(record):
                        h = hash(text)
                        if h in seen_grams:
                            continue
                        seen_grams.add(h)
                        for gram in grams.intersection(shingles(text)):
                            gram_hits[gram] += 1
            _drop_cache(handle)
        flush()
        try:
            rel = str(Path(path).relative_to(ARTIFACTS))
        except ValueError:
            rel = str(path)
        per_file[rel] = {"records": records, "sha256": digest.hexdigest(), "seconds": round(time.perf_counter() - started, 1)}
        _log(f"facts scan {Path(path).name}: {records} records ({per_file[rel]['seconds']} s); pieces with an exact hit so far "
             f"{sum(1 for v in exact.values() if v)}, shingle hits so far {sum(gram_hits.values())}")
    return {"files": per_file, "pieces": {p: {"exact": exact[p], "folded": folded[p]} for p in pieces}, "grams": len(grams),
            "gram_hits": dict(gram_hits), "distinct_texts": {"pieces": len(seen_pieces), "shingles": len(seen_grams)}}


def _gate_file_hashes(out: Path) -> dict[str, str]:
    """sha256 of every existing gate file under `out` except ``m2gf/`` (recorded before and after the facts build)."""
    found = {}
    for path in sorted(Path(out).rglob("*")):
        rel = path.relative_to(out)
        if path.is_file() and rel.parts[0] != "m2gf":
            found[str(rel)] = sha256_dropped(path)
    return found


def _stats(values: list[int]) -> dict[str, Any]:
    values = sorted(values)
    return {"n": len(values), "min": values[0], "median": values[len(values) // 2], "mean": round(sum(values) / len(values), 1), "max": values[-1]} if values else {"n": 0}


def build_facts(short: list[dict[str, Any]], *, seed: int) -> dict[str, Any]:
    """M2g-F from M2g-short (module docstring of :mod:`robo_jev.bench.gates_v07f`): the records, every check, the assignment and its balance,
    the truth-class mix per template and status line, the lengths. Refuses (SystemExit) on any problem."""
    from robo_jev.bench import gates_v07f as F
    from robo_jev.bench.gates_v07 import sentences
    from robo_jev.decisions.records import pair_groups, shuffle_key

    found = F.text_problems()
    if found:
        raise SystemExit(f"the gate's own text fails its checks: {found}")
    groups = pair_groups(short)
    if any(len(g) != 2 for g in groups) or len({str(r["id"]) for r in short}) != len(short):
        raise SystemExit("M2g-short: every record must be one of a whole one-fact pair, ids unique")
    strata: dict[str, list[str]] = defaultdict(list)
    for group in groups:
        key = shuffle_key(group[0])
        if shuffle_key(group[1]) != key:
            raise SystemExit(f"{group[0]['id']}: the pair does not share its order key")
        meta = group[0].get("meta") or {}
        strata[f"{(meta.get('source') or {}).get('set')}:{meta.get('scene')}"].append(key)
    assignment = F.assign(strata, seed=seed)
    problems: dict[str, list[str]] = {}
    records = []
    for record in short:
        chosen = assignment[shuffle_key(record)]
        f = F.facts_record(record, template=chosen["template"], status=chosen["status"])
        bad = F.facts_problems(f, record)
        if bad:
            problems[str(record["id"])] = bad
        records.append(f)
    by_id = {str(r["id"]): r for r in records}
    identical = 0
    for a, b in groups:
        fa, fb = by_id[str(a["id"])], by_id[str(b["id"])]
        if {k: fa["meta"]["gate"][k] for k in F.ADDED_GATE_KEYS} != {k: fb["meta"]["gate"][k] for k in F.ADDED_GATE_KEYS}:
            problems.setdefault(f"pair {shuffle_key(a)}", []).append("the pair's template or status line differ")
        elif F.added_text(fa, a) != F.added_text(fb, b):
            problems.setdefault(f"pair {shuffle_key(a)}", []).append("the pair's added text differs")
        else:
            identical += 1
    balance = {}
    for stratum, keys in sorted(strata.items()):
        templates = Counter(assignment[k]["template"] for k in keys)
        status = Counter(assignment[k]["status"] for k in keys)
        cells = Counter(f"{assignment[k]['template']}|{assignment[k]['status']}" for k in keys)
        balance[stratum] = {"pairs": len(keys), "templates": {t: templates.get(t, 0) for t in F.TEMPLATES}, "status": {s: status.get(s, 0) for s in F.STATUS_FORMS},
                            "cells": {f"{t}|{s}": cells.get(f"{t}|{s}", 0) for t in F.TEMPLATES for s in F.STATUS_FORMS}}
        for name in ("templates", "status", "cells"):
            counts = balance[stratum][name]
            if max(counts.values()) - min(counts.values()) > 1:
                problems.setdefault(f"stratum {stratum}", []).append(f"{name} not balanced: {counts}")
    if problems:
        raise SystemExit(f"M2g-F: {len(problems)} records / pairs / strata fail their checks; first {list(problems.items())[:3]}")
    truth: dict[str, Counter] = defaultdict(Counter)
    for r in records:
        classes = _truth_classes(r)
        truth[f"template:{r['meta']['gate']['template']}"].update(classes)
        truth[f"status:{r['meta']['gate']['status']}"].update(classes)
        truth["all"].update(classes)
    lengths = {"state_chars": _stats([len(r["request"]["state"]) for r in records]),
               "facts_chars_short_state": _stats([len(" ".join(sentences(r["request"]["state"]))) for r in short]),
               "instructions_chars": {t: _stats([len(q["instructions"]) for r in records if r["meta"]["gate"]["template"] == t
                                                for q in r["request"]["questions"].values()]) for t in F.TEMPLATES},
               "short_instructions_chars": _stats([len(q["instructions"]) for r in short for q in r["request"]["questions"].values()])}
    return {"records": records, "assignment": assignment, "balance": balance, "pairs_identical": identical, "pairs": len(groups),
            "crossing": {t: dict(Counter(r["meta"]["gate"]["status"] for r in records if r["meta"]["gate"]["template"] == t)) for t in F.TEMPLATES},
            "truth_classes": {k: dict(v) for k, v in sorted(truth.items())},
            "counts": {"records": len(records), "questions": sum(len(r["request"]["questions"]) for r in records),
                       "templates": dict(Counter(r["meta"]["gate"]["template"] for r in records)),
                       "status": dict(Counter(r["meta"]["gate"]["status"] for r in records))},
            "lengths": lengths}


def order_proof(short: list[dict[str, Any]], facts: list[dict[str, Any]], tokenizer: Any) -> dict[str, Any]:
    """The seeded option order through the diagnosis prompt path (:func:`robo_jev.decisions.prompt.record_prompts` with M2g's settings —
    ``scripts/diagnose_decisions.py --orders seeded``, shuffle seed 1, chat, direct, v0c-p3): every prompt of every M2g-F record shows the
    same options, lines, values and codes in the same order as M2g-short's; and what the model reads — the question line holds every
    original sentence in order and the instruction, the situation holds none of them. Also the prompt tokens of both forms."""
    from robo_jev.bench.gates_v07 import sentences
    from robo_jev.decisions.prompt import PROMPT_VERSION, record_prompts

    by_id = {str(r["id"]): r for r in short}
    mismatched, shown, prompts = [], [], 0
    tokens: dict[str, list[int]] = {"short": [], "facts": []}
    template_tokens: dict[str, list[int]] = defaultdict(list)
    for f in facts:
        s = by_id[str(f["id"])]
        settings = {"seed": FACTS_SCORING["shuffle_seed"], "fmt": FACTS_SCORING["format"], "mode": FACTS_SCORING["mode"],
                    "orders": FACTS_SCORING["orders"], "version": PROMPT_VERSION}
        ps, pf = record_prompts(s, tokenizer, **settings), record_prompts(f, tokenizer, **settings)
        if [[json.dumps(p[k]) for k in ORDER_FIELDS] for p in ps] != [[json.dumps(p[k]) for k in ORDER_FIELDS] for p in pf]:
            mismatched.append(str(f["id"]))
        prompts += len(pf)
        tokens["short"] += [len(p["tokens"]) for p in ps]
        tokens["facts"] += [len(p["tokens"]) for p in pf]
        template_tokens[f["meta"]["gate"]["template"]] += [len(p["tokens"]) for p in pf]
        units = sentences(s["request"]["state"])
        for p in pf:
            situation, question = p["text"][: p["prefix_chars"]], p["text"][p["prefix_chars"]:]
            position = 0
            for unit in units:
                hit = question.find(unit, position)
                if hit < 0:
                    shown.append(f"{f['id']} {p['question']}: a sentence is not shown in order")
                    break
                position = hit + len(unit)
            if f["request"]["questions"][p["question"]]["instructions"] not in question:
                shown.append(f"{f['id']} {p['question']}: the instructions are not shown as stored")
            if any(len(u.split()) >= 3 and u in situation for u in units):
                shown.append(f"{f['id']} {p['question']}: an original sentence is shown in the situation")
    return {"records": len(facts), "prompts": prompts, "order_mismatches": mismatched, "shown_problems": shown,
            "tokens": {k: _stats(v) | {"total": sum(v)} for k, v in tokens.items()},
            "tokens_by_template": {k: _stats(v) | {"total": sum(v)} for k, v in sorted(template_tokens.items())}}


def facts_gpu_estimate(total_tokens: int) -> dict[str, Any]:
    """Expected Spark time of M2g-F: task v0.7b's measured seconds per prompt token (LoRA 0.20 ms, base 0.12 ms) plus one model load per
    unit; the chain scores four LoRA models and the base."""
    lora = total_tokens * SECONDS_PER_TOKEN["lora"] + LOAD_SECONDS
    base = total_tokens * SECONDS_PER_TOKEN["base"] + LOAD_SECONDS
    return {"seconds_per_token": SECONDS_PER_TOKEN, "load_seconds": LOAD_SECONDS, "prompt_tokens_per_model": total_tokens,
            "per_lora_model_s": round(lora), "base_s": round(base), "five_models_s": round(4 * lora + base),
            "five_models_min": round((4 * lora + base) / 60, 1)}


def _short_source(out: Path) -> tuple[Path, str]:
    """The sealed M2g-short file and its sha256 as task v0.7b's manifest recorded it."""
    m2 = json.loads((out / "m2g" / "m2g.manifest.json").read_text(encoding="utf-8"))
    return REPO / m2["files"]["short"]["path"], m2["files"]["short"]["sha256"]


def cmd_build_facts(args: argparse.Namespace) -> int:
    """Build gate M2g-F into ``<out-dir>/m2gf/`` (module docstring)."""
    from robo_jev.bench import gates_v07f as F
    from robo_jev.decisions.records import read_records

    started = time.perf_counter()
    out = Path(args.out_dir)
    target = refuse_write(out / "m2gf")
    before = _gate_file_hashes(out)
    short_path, short_sha = _short_source(out)
    if sha256_dropped(short_path) != short_sha:
        raise SystemExit(f"{short_path}: not the sealed M2g-short (sha256 {short_sha[:12]} in m2g.manifest.json)")
    short = read_records(short_path)
    built = build_facts(short, seed=args.seed)
    records = built["records"]
    _log(f"M2g-F: {len(records)} records, {built['counts']['questions']} questions, every record checked; pairs identical outside their labels "
         f"{built['pairs_identical']} of {built['pairs']}")
    tokens: dict[str, Any] = {"skipped": True}
    tokenizer_info = None
    if not args.skip_tokens:
        from robo_jev.model.tokenizer import available_tokenizer, describe_tokenizer, load_tokenizer

        tokenizer_id = available_tokenizer()[0]
        tokenizer_info = describe_tokenizer(tokenizer_id)
        tokens = order_proof(short, records, load_tokenizer(tokenizer_id))
        if tokens["order_mismatches"] or tokens["shown_problems"]:
            raise SystemExit(f"order or display proof fails: {tokens['order_mismatches'][:3]} {tokens['shown_problems'][:3]}")
        _log(f"order proof: {tokens['prompts']} prompts of {tokens['records']} records — the seeded option order equals M2g-short's everywhere; "
             f"mean prompt tokens short {tokens['tokens']['short']['mean']} vs facts {tokens['tokens']['facts']['mean']}")
    pieces = [p for p in F.fixed_pieces() if p["words"] >= 2]
    grams: set[str] = set()
    by_id = {str(r["id"]): r for r in short}
    for r in records:
        grams |= F.touching_shingles(r, by_id[str(r["id"])])
    # sensitivity controls (reported, never deciding): two common phrases, and the 8-word shingles of M2g-short's own instructions (the
    # H-set's and B1's question wording is their generators', shared with training by design) — the scan must find these
    from robo_jev.bench.iface import shingles
    from robo_jev.data.general_sets import normalize

    control_pieces = [p for p in CONTROL_PIECES if p not in {q["text"] for q in pieces}]
    control_grams = {g for r in short for q in r["request"]["questions"].values() for g in shingles(str(q["instructions"]))
                     if len(normalize(str(q["instructions"])).split()) >= 8} - grams
    files = [ARTIFACTS / f for f in TRAINING_INPUTS] + [ARTIFACTS / TRAIN_V07]
    missing = [str(f) for f in files if not f.is_file()]
    if args.skip_overlap:
        scan_result: dict[str, Any] = {"skipped": True, "files": {}, "pieces": {p["text"]: {"exact": None, "folded": None} for p in pieces},
                                       "grams": len(grams), "gram_hits": {}}
        controls: dict[str, Any] = {"skipped": True}
    else:
        if missing:
            raise SystemExit(f"training inputs missing: {missing}")
        found = facts_scan(files, [p["text"] for p in pieces] + control_pieces, grams | control_grams)
        scan_result = {**found, "pieces": {p["text"]: found["pieces"][p["text"]] for p in pieces},
                       "gram_hits": {g: n for g, n in found["gram_hits"].items() if g in grams}, "grams": len(grams)}
        controls = {"pieces": {p: found["pieces"][p] for p in control_pieces},
                    "instruction_shingles": len(control_grams), "instruction_shingles_found": sum(1 for g in found["gram_hits"] if g in control_grams),
                    "note": "the scan's sensitivity: common phrases and M2g-short's own question wording (shared with training generators for the H-set "
                            "and B1 by design) are found; reported, never deciding"}
        hits = {p: v for p, v in scan_result["pieces"].items() if v["exact"]}
        if hits or scan_result["gram_hits"]:
            raise SystemExit(f"added text occurs in a training input: {len(hits)} pieces ({sorted(hits)}), {len(scan_result['gram_hits'])} shingles")
        if not controls["instruction_shingles_found"] or not all(v["exact"] for v in controls["pieces"].values()):
            raise SystemExit(f"the scan's sensitivity controls were not found — the scan is not reading the training text: {controls}")
    v07_rel = str(Path(TRAIN_V07))
    overlap = {"method": "every fixed run of the templates and status lines with 2+ words (between slots, with its punctuation) against every "
                         "training record's state and question instructions (whitespace collapsed; exact and case-folded occurrence counts); "
                         "every 8-word shingle of an M2g-F text (state, each instruction) holding a word of added text against the 8-word "
                         "shingles of every training text (state, instructions, options); counts only",
               "skipped": bool(scan_result.get("skipped")), "inputs": scan_result["files"],
               "train_v07": {"path": f"artifacts/{v07_rel}", "sha256": (scan_result["files"].get(v07_rel) or {}).get("sha256")},
               "pieces": [{**p, **scan_result["pieces"][p["text"]]} for p in pieces],
               "pieces_with_exact_hits": sum(1 for v in scan_result["pieces"].values() if v["exact"]),
               "pieces_with_folded_hits": sum(1 for v in scan_result["pieces"].values() if v["folded"]),
               "shingles": len(grams), "shingles_found": len(scan_result["gram_hits"]),
               "distinct_texts": scan_result.get("distinct_texts"), "sensitivity_controls": controls}
    # files
    target.mkdir(parents=True, exist_ok=True)
    path = target / F.FILE_NAME
    sha = _write_jsonl(path, records)
    examples = []
    for template in F.TEMPLATES:
        example = next(r for r in records if r["meta"]["gate"]["template"] == template)
        name = next(iter(example["request"]["questions"]))
        text = (f"[{example['id']} · template {template} · status {example['meta']['gate']['status']} · question {name}]\n\nstate:\n"
                f"{example['request']['state']}\n\ninstructions:\n{example['request']['questions'][name]['instructions']}\n\nM2g-short state:\n"
                f"{by_id[str(example['id'])]['request']['state']}\n")
        example_path = refuse_write(target / "examples" / f"{template}-{example['id']}.txt")
        example_path.parent.mkdir(parents=True, exist_ok=True)
        example_path.write_text(text, encoding="utf-8")
        examples.append(_rel(example_path))
    _seal([path])
    after = _gate_file_hashes(out)
    if after != before:
        changed = sorted(k for k in set(before) | set(after) if before.get(k) != after.get(k))
        raise SystemExit(f"existing gate files changed during the build: {changed}")
    gpu = facts_gpu_estimate(tokens["tokens"]["facts"]["total"]) if not tokens.get("skipped") else None
    manifest = {
        "kind": "v07-m2g-facts", "script_version": FACTS_SCRIPT_VERSION, "git": _git(), "seed": args.seed, "set": F.M2G_SET, "prefix": F.M2G_PREFIX,
        "condition": F.CONDITION, "registry": {"name": FACTS_REGISTRY, "path": _rel(path)},
        "rule": "PRE-DECLARED v0.7 SECONDARY READING - FACTS IN INSTRUCTIONS, gate M2g-F (progress.md, 2026-10-15 ~19:57 KST)",
        "source": {"path": _rel(short_path), "sha256": short_sha, "records": len(short), "manifest": _rel(out / "m2g" / "m2g.manifest.json")},
        "file": {"path": _rel(path), "sha256": sha, "records": len(records), "questions": built["counts"]["questions"], "mode": "0444"},
        "design": {"status_forms": F.STATUS_FORMS, "templates": F.TEMPLATES, "slots": list(F.SLOTS),
                   "facts": "the original state's sentences (robo_jev.bench.gates_v07.sentences), verbatim and in order, joined by one space",
                   "split": "split = the first half of the facts: the sentences of the first ceil(L/2) lines of a state of L >= 2 lines, else "
                            "ceil(n/2) of its n sentences (the same cut for both records of a pair)",
                   "assignment": "template and status line per pair order key: within each set x scene stratum, the keys in a seeded order take "
                                 "the 9 (template, status) cells of a Latin order (every block of 3 cells holds each template and each status "
                                 "line once; templates and status lines in seeded orders) in turn from a seeded block - templates, status lines "
                                 "and cells each balanced to +-1 per stratum",
                   "added_gate_keys": list(F.ADDED_GATE_KEYS)},
        "counts": built["counts"], "crossing_template_x_status": built["crossing"], "balance_by_stratum": built["balance"],
        "truth_classes": built["truth_classes"], "lengths_chars": built["lengths"],
        "proofs": {"records_checked": len(records), "records_failing": 0, "pairs_added_text_identical": f"{built['pairs_identical']} of {built['pairs']}",
                   "order": {k: tokens.get(k) for k in ("records", "prompts")} | {"mismatches": len(tokens.get("order_mismatches") or []),
                                                                                "display_problems": len(tokens.get("shown_problems") or []),
                                                                                "settings": FACTS_SCORING} if not tokens.get("skipped") else "skipped",
                   "checks": ["facts_problems on every record: the state is the pair's status line and holds no original sentence of 3+ words, no fact or "
                              "tone word; every original sentence verbatim and in order in every question's instructions; the original instruction "
                              "exactly once; the instructions equal the template's rendering and are one single-spaced line; questions (names, order, "
                              "types, criteria), labels and every other field equal M2g-short's; validate_record passes",
                              "both records of every pair: the same template and status line, and byte-identical added text (original sentences and "
                              "instructions cut out)", "templates, status lines and their 9 cells balanced within every set x scene stratum "
                              "(max - min <= 1)",
                              "order: every prompt's options, lines, values and codes equal M2g-short's (record_prompts, M2g's settings)",
                              "existing gate files unchanged (sha256 before and after)"]},
        "prompt_tokens": tokens.get("tokens") if not tokens.get("skipped") else "skipped",
        "prompt_tokens_by_template": tokens.get("tokens_by_template") if not tokens.get("skipped") else "skipped",
        "tokenizer": tokenizer_info, "gpu_estimate": gpu, "scoring": {**FACTS_SCORING, "models": list(FACTS_MODELS)},
        "overlap": overlap, "existing_gate_files_sha256": {"before": before, "after": after, "unchanged": after == before},
        "examples": examples, "seconds": round(time.perf_counter() - started, 1),
        "created": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
    }
    _write_json(target / "manifest.json", manifest)
    _log(f"done in {manifest['seconds']} s → {_rel(path)} (sha256 {sha[:12]}); overlap: {overlap['pieces_with_exact_hits']} pieces with an exact hit, "
         f"{overlap['shingles_found']} of {overlap['shingles']} shingles found" + (f"; GPU ≈ {gpu['five_models_min']} min for five models" if gpu else ""))
    return 0


def check_facts(out: Path, runs: list[Path] | list[str]) -> list[str]:
    """Every problem of the sealed M2g-F gate (``check-facts``, module docstring); empty = the gate may be opened."""
    from robo_jev.bench import gates_v07f as F
    from robo_jev.decisions.prompt import PROMPT_VERSION, _listings
    from robo_jev.decisions.records import pair_groups, read_records, shuffle_key

    problems: list[str] = []
    manifest_path = Path(out) / "m2gf" / "manifest.json"
    if not manifest_path.is_file():
        return [f"{_rel(manifest_path)} is missing — run build-facts"]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    facts_path, short_path = REPO / manifest["file"]["path"], REPO / manifest["source"]["path"]
    if not facts_path.is_file() or sha256_dropped(facts_path) != manifest["file"]["sha256"]:
        return [f"{manifest['file']['path']}: missing or not the sealed file (sha256 {manifest['file']['sha256'][:12]})"]
    if stat.S_IMODE(facts_path.stat().st_mode) & 0o222:
        problems.append(f"{manifest['file']['path']} is writable (sealed files are 0444)")
    m2 = json.loads((Path(out) / "m2g" / "m2g.manifest.json").read_text(encoding="utf-8"))
    if m2["files"]["short"]["sha256"] != manifest["source"]["sha256"] or not short_path.is_file() or sha256_dropped(short_path) != manifest["source"]["sha256"]:
        problems.append("M2g-F was not built from the sealed M2g-short (sha256 differs from m2g.manifest.json or the file on disk)")
        return problems
    facts, short = read_records(facts_path), read_records(short_path)
    if [str(r["id"]) for r in facts] != [str(r["id"]) for r in short]:
        problems.append("the ids (or their order) differ from M2g-short's")
        return problems
    failing = {str(f["id"]): bad for f, s in zip(facts, short) if (bad := F.facts_problems(f, s))}
    if failing:
        problems.append(f"{len(failing)} records fail their checks (first {list(failing.items())[:2]})")
    by_id = {str(r["id"]): r for r in facts}
    for a, b in pair_groups(short):
        fa, fb = by_id[str(a["id"])], by_id[str(b["id"])]
        try:
            same = F.added_text(fa, a) == F.added_text(fb, b) and fa["meta"]["gate"] == fb["meta"]["gate"]
        except (KeyError, ValueError) as error:
            same = False
            problems.append(f"pair {shuffle_key(a)}: {error}")
        if not same:
            problems.append(f"pair {shuffle_key(a)}: the two records differ outside their labels in the added text")
    orders = 0
    for f, s in zip(facts, short):
        for name, question in s["request"]["questions"].items():
            settings = {"seed": FACTS_SCORING["shuffle_seed"], "orders": FACTS_SCORING["orders"], "version": PROMPT_VERSION}
            mine = _listings(name, f["request"]["questions"][name], key=shuffle_key(f), **settings)
            if mine != _listings(name, question, key=shuffle_key(s), **settings):
                problems.append(f"{f['id']} {name}: the seeded option order differs from M2g-short's")
            orders += 1
    overlap = manifest.get("overlap") or {}
    if overlap.get("skipped") or overlap.get("pieces_with_exact_hits") != 0 or overlap.get("shingles_found") != 0:
        problems.append("the manifest's overlap scan was skipped or found added text in a training input")
    scanned = (overlap.get("train_v07") or {}).get("sha256")
    on_disk = ARTIFACTS / TRAIN_V07
    if on_disk.is_file() and sha256_dropped(on_disk) != scanned:
        problems.append(f"{TRAIN_V07} changed since the facts overlap scan ({str(scanned)[:12]}) — rebuild")
    for run in runs:
        metrics = Path(run) / "metrics.json"
        got = ((json.loads(metrics.read_text(encoding="utf-8")).get("manifest") or {}).get("data") or {}).get("file_sha256") if metrics.is_file() else None
        if got != scanned:
            problems.append(f"{run}: trained on {str(got)[:12]}, the facts overlap scan read {str(scanned)[:12]}")
    _log(f"check-facts: {len(facts)} records, {orders} question orders checked, {len(problems)} problems")
    return problems


def cmd_check_facts(args: argparse.Namespace) -> int:
    """Before the M2g-F gate is opened (``artifacts/scratch/v07f/score_facts.sh``): :func:`check_facts`. Prints the problems; 2 if any."""
    problems = check_facts(args.out_dir, args.runs)
    print(json.dumps({"problems": problems}, indent=1))
    return 2 if problems else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    b = sub.add_parser("build")
    b.add_argument("--out", type=Path, default=OUT)
    b.add_argument("--seed", type=int, default=SEED)
    b.add_argument("--a4-pairs", type=int, default=100)
    b.add_argument("--hset-pairs", type=int, default=65)
    b.add_argument("--b1-pairs", type=int, default=65)
    b.add_argument("--mmlu-items", type=int, default=760)
    b.add_argument("--physics-per-set", type=int, default=None, help="cap per physics set (default: every eligible pair)")
    b.add_argument("--audit-general", type=int, default=40)
    b.add_argument("--audit-physics", type=int, default=10)
    b.add_argument("--skip-overlap", action="store_true", help="tests only: skip the training-input scan")
    r = sub.add_parser("recheck-v07")
    r.add_argument("--out-dir", type=Path, default=OUT)
    r.add_argument("--train-v07", type=Path, default=ARTIFACTS / TRAIN_V07)
    c = sub.add_parser("check")
    c.add_argument("--out-dir", type=Path, default=OUT)
    c.add_argument("--runs", nargs="*", default=[], help="v0.7 run directories (…-fetched/runs/v07-s<seed>-2b) whose training input must match")
    bf = sub.add_parser("build-facts", help="task v0.7f: gate M2g-F from the sealed M2g-short")
    bf.add_argument("--out-dir", type=Path, default=OUT)
    bf.add_argument("--seed", type=int, default=SEED)
    bf.add_argument("--skip-overlap", action="store_true", help="tests only: skip the training-input scan")
    bf.add_argument("--skip-tokens", action="store_true", help="tests only: no tokenizer (no order proof through the prompt path, no token counts)")
    cf = sub.add_parser("check-facts", help="task v0.7f: the M2g-F chain's gate")
    cf.add_argument("--out-dir", type=Path, default=OUT)
    cf.add_argument("--runs", nargs="*", default=[], help="v0.7 run directories (…-fetched/runs/v07-s<seed>-2b) whose training input must match")
    args = parser.parse_args(argv)
    return {"build": cmd_build, "recheck-v07": cmd_recheck, "check": cmd_check, "build-facts": cmd_build_facts,
            "check-facts": cmd_check_facts}[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
