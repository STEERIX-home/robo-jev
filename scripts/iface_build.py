#!/usr/bin/env python
"""Task iface — build the M1 (many options) and M2 (bench-shaped long states) sets of the pre-declared interface measurements (CPU).

    artifacts/scratch/iface/cpu.sh build scripts/iface_build.py            # defaults: artifacts/iface/{m1,m2}/

Inputs: the ARC test rows (``scripts/iface_fetch_arc.py``), B1-infamily (``artifacts/b1/b1-infamily.jsonl``) and the H-set
(``artifacts/simgen/v04/scene-holdout.jsonl``). The sealed gates are never read: any input under ``artifacts/t9`` or ``artifacts/v06/a4`` is
refused.

1. **Candidates** — :func:`robo_jev.bench.iface.arc_items` (ARC; drop counts recorded) and :func:`robo_jev.bench.iface.physics_items`
   (direction questions with a single clear best option) on B1 and the H-set.
2. **Overlap with every training input** (:data:`TRAINING_INPUTS` — the v0, v0.1, v0.1-mix, v0.2, v0.3, v0.4, v0.5 and v0.6 inputs, the
   general mix and the probe mix): the project's text units (:func:`robo_jev.data.general_sets.record_units` — the whole normalised state,
   state line values and options with ≥ 5 words) of every training record against the candidates' units — ARC: the state, the stem and every
   option with ≥ 5 words; physics and M2 records (generated): the whole normalised state, i.e. an exact duplicate (their state lines and
   option texts are the generator's templates — "packing: its contents fill it uniformly", "more likely than with the plan" — shared with
   training by design). ARC stems and long options are also checked by 8-word shingles against the general mix (the only question-answering
   text in training). A candidate with any hit is dropped (hits per candidate and file in the manifest).
3. **M1** — 450 ARC-Easy + 450 ARC-Challenge items (seeded, among those whose answer has ≥ 3 words), 150 B1 + 150 H-set physics items (one per one-fact pair, spread over the
   scenes); distractors drawn (:func:`~robo_jev.bench.iface.draw_general`, :func:`~robo_jev.bench.iface.draw_physics`); every item expanded to
   N ∈ {4, 8, 12, 16, 21, 26} at controlled positions — 3 listings at N = 8, 21, 26, 1 at N = 4, 12, 16 (:func:`~robo_jev.bench.iface.expand_item`). Every record passes
   :func:`~robo_jev.bench.iface.m1_problems` with the cap lifted, and the N ≤ 8 records also with the standard cap. Writes
   ``m1/m1-set.jsonl`` (one record per listing), ``m1/m1-items.jsonl`` (one line per item: options in draw order) and ``m1/m1-set.manifest.json``.
4. **M2** — a pair-preserving seeded sample of each set's records (``--m2-fraction`` per family / scene, whole one-fact pairs;
   :func:`robo_jev.decisions.records.select_records`), each wrapped at 2k, 4k and 8k characters (:func:`~robo_jev.bench.iface.wrap_record`;
   :func:`~robo_jev.bench.iface.sheet_problems` must be empty; :func:`robo_jev.decisions.records.validate_record` passes unchanged — ≤ 8
   options, state ≤ 16,000 characters). Writes ``m2/M2-short.jsonl`` (the sampled records unchanged — the short condition is scored
   fresh), ``m2/M2-<length>.jsonl`` (``length`` ∈ w2k, w4k, w8k; B1 then the H-set in each file), one example sheet per set and length
   (``m2/examples/``), and ``m2/m2-set.manifest.json`` (selection, ids, lengths, prompt tokens per length under the M2 orders from the real
   tokenizer, overlap). M2 is scored with ``--orders seeded`` (:data:`M2_ORDERS`).
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

SCRIPT_VERSION = "iface-build-1"
ARTIFACTS = REPO / "artifacts"
OUT = ARTIFACTS / "iface"
ARC = OUT / "m1" / "source" / "arc-test.jsonl"
ARC_FETCH = OUT / "m1" / "source" / "arc-test.fetch.json"
B1 = ARTIFACTS / "b1" / "b1-infamily.jsonl"
HSET = ARTIFACTS / "simgen" / "v04" / "scene-holdout.jsonl"
#: Every training input of the project's trained versions (assembled files contain their parts: engine, MuJoCo, authors, d1, general).
TRAINING_INPUTS = (
    "simgen/v0b/train-50k.jsonl", "simgen/v01/train-50k.jsonl", "simgen/v01-mix/train-50k-mix.jsonl", "simgen/probes/G1-train-mix.jsonl",
    "simgen/v02/train-50k.jsonl", "simgen/v02/general-train.jsonl", "simgen/v02/train-v02-a.jsonl", "simgen/v02/train-v02-b.jsonl",
    "simgen/v03/train-v03.jsonl", "simgen/v04/train-v04.jsonl", "simgen/v05/train-v05.jsonl", "simgen/v06/train-v06.jsonl",
)
GENERAL_MIX = "simgen/v02/general-train.jsonl"
#: Sealed gates — never opened.
SEALED = (ARTIFACTS / "t9", ARTIFACTS / "v06" / "a4")
SETS = {"B1": B1, "Hset": HSET}
#: M2 scoring orders (scripts/diagnose_decisions.py --orders): one listing per question, the seeded order — listing 0 of the counterbalanced
#: runs. Chosen for GPU time (measured on Spark: 0.14 s per prompt at 2k, 0.63 s at 8k for a LoRA model) before any M2 data existed.
M2_ORDERS = "seeded"


def _log(message: str) -> None:
    print(f"[iface-build {dt.datetime.now().strftime('%H:%M:%S')}] {message}", file=sys.stderr, flush=True)


def _git() -> dict[str, Any]:
    try:
        sha = subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "-C", str(REPO), "status", "--porcelain", "--untracked-files=no"], capture_output=True, text=True, check=True).stdout.strip())
        return {"commit": sha, "dirty": dirty}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}


def refuse_sealed(path: Path) -> Path:
    resolved = Path(path).resolve()
    for place in SEALED:
        place = place.resolve()
        if resolved == place or place in resolved.parents:
            raise SystemExit(f"{path}: a sealed gate ({place}) — never opened by this task")
    return path


def _read(path: Path) -> list[dict[str, Any]]:
    from robo_jev.decisions.records import read_records

    return read_records(refuse_sealed(path))


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    from robo_jev.decisions.records import file_sha256

    return file_sha256(path)


def _hash(unit: str) -> str:
    return hashlib.sha256(unit.encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------
# Overlap
# --------------------------------------------------------------------------


def candidate_units(arc: list[dict[str, Any]], states: dict[str, str]) -> tuple[dict[str, set[str]], dict[str, set[str]]]:
    """(unit hash → candidates) and (8-word shingle → ARC candidates): ARC items by their state and options, physics / M2 records by state."""
    from robo_jev.bench.iface import shingles
    from robo_jev.data.general_sets import normalize, text_units

    units: dict[str, set[str]] = defaultdict(set)
    grams: dict[str, set[str]] = defaultdict(set)
    for item in arc:
        for unit in text_units(item["state"], [item["correct"], *item["own"]]):
            units[_hash(unit)].add(item["item"])
        for text in [item["stem"]] + [t for t in [item["correct"], *item["own"]] if len(t.split()) >= 8]:
            for gram in shingles(text):
                grams[gram].add(item["item"])
    for name, state in states.items():
        # generated records: the whole normalised state (their state *lines* are generator templates shared with training by design —
        # "packing: its contents fill it uniformly" — and say nothing about whether the record itself was trained on)
        units[_hash(normalize(state))].add(name)
    return dict(units), dict(grams)


def scan_training(units: dict[str, set[str]], grams: dict[str, set[str]], files: list[Path], general: Path) -> dict[str, Any]:
    """Every training record's units against the candidates' (and the general mix's text shingles against the ARC shingles) → hits per
    candidate and per file."""
    from robo_jev.bench.iface import shingles
    from robo_jev.data.general_sets import record_units

    hits: dict[str, set[str]] = defaultdict(set)
    per_file: dict[str, Any] = {}
    for path in files:
        started, records, found = time.perf_counter(), 0, 0
        shingle_check = path.resolve() == general.resolve()
        with refuse_sealed(path).open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                record = json.loads(line)
                records += 1
                for unit in record_units(record):
                    owners = units.get(_hash(unit))
                    if owners:
                        found += 1
                        for owner in owners:
                            hits[owner].add(f"unit:{path.name}")
                if shingle_check:
                    request = record.get("request") or {}
                    texts = [str(request.get("state") or "")]
                    for question in (request.get("questions") or {}).values():
                        if isinstance(question.get("criteria"), dict):
                            texts += [str(v) for v in question["criteria"].values()]
                    for text in texts:
                        for gram in shingles(text):
                            owners = grams.get(gram)
                            if owners:
                                found += 1
                                for owner in owners:
                                    hits[owner].add(f"shingle:{path.name}")
        from robo_jev.decisions.records import file_sha256

        per_file[str(path.relative_to(ARTIFACTS))] = {"records": records, "unit_or_shingle_hits": found, "sha256": file_sha256(path),
                                                      "seconds": round(time.perf_counter() - started, 1), "shingles": shingle_check}
        _log(f"overlap scan {path.name}: {records} records, {found} hits, {per_file[str(path.relative_to(ARTIFACTS))]['seconds']} s")
    return {"hits": {k: sorted(v) for k, v in hits.items()}, "files": per_file}


# --------------------------------------------------------------------------
# Token estimates (real tokenizer, the prompt code)
# --------------------------------------------------------------------------


def prompt_tokens(records: list[dict[str, Any]], *, seed: int | None, orders: str, tokenizer: Any) -> dict[str, Any]:
    from robo_jev.decisions.prompt import record_prompts

    prompts = [p for r in records for p in record_prompts(r, tokenizer, seed=seed, fmt="chat", mode="direct", orders=orders)]
    lengths = sorted(len(p["tokens"]) for p in prompts)
    if not lengths:
        return {"prompts": 0}
    return {"prompts": len(lengths), "tokens": sum(lengths), "mean": round(sum(lengths) / len(lengths), 1), "median": lengths[len(lengths) // 2],
            "max": lengths[-1]}


# --------------------------------------------------------------------------
# Build
# --------------------------------------------------------------------------


def build(args: argparse.Namespace) -> int:
    from robo_jev.bench import iface
    from robo_jev.decisions.records import OPTION_RANGE, file_sha256, select_records, shuffle_key, validate_record
    from robo_jev.model.tokenizer import available_tokenizer, describe_tokenizer, load_tokenizer

    started = time.perf_counter()
    arc_rows = _read(args.arc)
    arc_all, arc_dropped = iface.arc_items(arc_rows)
    sets = {name: _read(path) for name, path in ((n, getattr(args, n.lower())) for n in SETS)}
    physics_all = {}
    physics_counts = {}
    for name, records in sets.items():
        physics_all[name], physics_counts[name] = iface.physics_items(records, source=name.lower())
    # M2 sample (pair-preserving, per family / scene)
    m2_ids: dict[str, list[str]] = {}
    m2_selection: dict[str, Any] = {}
    for name, records in sets.items():
        chosen = []
        by_scene: dict[str, list] = defaultdict(list)
        for record in records:
            by_scene[str((record.get("meta") or {}).get("scene"))].append(record)
        for scene, members in sorted(by_scene.items()):
            picked = select_records(members, round(args.m2_fraction * len(members)), seed=args.seed)
            chosen += picked
            m2_selection.setdefault(name, {})[scene] = {"records": len(members), "chosen": len(picked)}
        keep = {str(r["id"]) for r in chosen}
        m2_ids[name] = [str(r["id"]) for r in records if str(r["id"]) in keep]
    # overlap
    states = {}
    for name, items in physics_all.items():
        for item in items:
            states[item["item"]] = item["state"]
    for name, records in sets.items():
        by_id = {str(r["id"]): r for r in records}
        for rid in m2_ids[name]:
            states[f"m2:{name}:{rid}"] = by_id[rid]["request"]["state"]
    units, grams = candidate_units(arc_all, states)
    files = [ARTIFACTS / f for f in TRAINING_INPUTS]
    missing = [str(f) for f in files if not f.is_file()]
    if missing:
        raise SystemExit(f"training inputs missing: {missing}")
    if args.skip_overlap:
        overlap = {"hits": {}, "files": {}, "skipped": True}
    else:
        overlap = scan_training(units, grams, files, ARTIFACTS / GENERAL_MIX)
    hits = overlap["hits"]
    arc_clean = [i for i in arc_all if i["item"] not in hits]
    physics_clean = {name: [i for i in items if i["item"] not in hits] for name, items in physics_all.items()}
    m2_hits = {name: [rid for rid in ids if f"m2:{name}:{rid}" in hits] for name, ids in m2_ids.items()}
    if any(m2_hits.values()):
        raise SystemExit(f"M2 records overlap a training input: {m2_hits} — the rule needs never-trained records")
    _log(f"ARC {len(arc_all)} candidates, {len(arc_all) - len(arc_clean)} overlap; physics " +
         ", ".join(f"{n} {len(physics_all[n])} → {len(physics_clean[n])}" for n in physics_all))
    # M1 selection
    from robo_jev.bench.iface import _rng

    chosen_arc = []
    for subset, count in (("ARC-Easy", args.arc_easy), ("ARC-Challenge", args.arc_challenge)):
        pool = sorted((i for i in arc_clean if i["source"] == subset and iface.general_target(i)), key=lambda x: x["item"])
        _rng("iface-m1-arc", subset, args.seed).shuffle(pool)
        if len(pool) < count:
            raise SystemExit(f"{subset}: {len(pool)} clean items < {count}")
        chosen_arc += pool[:count]
    chosen_physics = []
    for name, count in (("B1", args.physics_b1), ("Hset", args.physics_hset)):
        chosen_physics += iface.select_physics(physics_clean[name], count, seed=args.seed)
    items = chosen_arc + chosen_physics
    _rng("iface-m1-rank", args.seed).shuffle(items)
    index = iface.general_index(arc_clean)
    pools = {name: iface.physics_pool(records) for name, records in sets.items()}
    item_lines, m1_records = [], []
    for rank, item in enumerate(items):
        if item["source"] in iface.GENERAL_SOURCES:
            drawn = iface.draw_general(item, index, count=max(iface.N_LEVELS) - 4, seed=args.seed)
        else:
            own_set = "B1" if item["source"] == "b1" else "Hset"
            other_set = "Hset" if own_set == "B1" else "B1"
            drawn = iface.draw_physics(item, pools[own_set], pools[other_set], count=max(iface.N_LEVELS) - 3, seed=args.seed)
        listings = iface.expand_item(item, drawn, rank=rank, index=rank, seed=args.seed)
        m1_records += listings
        item_lines.append({"item": item["item"], "source": item["source"], "rank": rank, "state": item["state"], "instructions": item["instructions"],
                           "correct": item["correct"], "own": item["own"], "drawn": drawn,
                           **({"record": item["record"], "question": item["question"], "scene": item["scene"], "group": item["group"]}
                              if "record" in item else {"stem": item["stem"]})})
    problems = {r["id"]: p for r in m1_records if (p := iface.m1_problems(r, lift_cap=True))}
    if problems:
        raise SystemExit(f"M1 records fail validation with the cap lifted: {list(problems.items())[:3]}")
    capped = Counter((r["meta"]["n"] <= OPTION_RANGE[1], not iface.m1_problems(r, lift_cap=False)) for r in m1_records)
    if capped.get((True, False)) or capped.get((False, True)):
        raise SystemExit(f"the standard cap does not behave as expected on the M1 records: {dict(capped)}")
    tokenizer_id = available_tokenizer()[0]
    tokenizer = load_tokenizer(tokenizer_id)
    m1_dir = args.out / "m1"
    m1_sha = _write_jsonl(m1_dir / "m1-set.jsonl", m1_records)
    items_sha = _write_jsonl(m1_dir / "m1-items.jsonl", item_lines)
    by_n = {n: [r for r in m1_records if r["meta"]["n"] == n] for n in iface.N_LEVELS}
    m1_tokens = {str(n): prompt_tokens(rs, seed=None, orders="seeded", tokenizer=tokenizer) for n, rs in by_n.items()}
    positions = {str(n): dict(sorted(Counter(r["meta"]["letter"] for r in rs).items())) for n, rs in by_n.items()}
    arc_fetch = json.loads(args.arc.with_name("arc-test.fetch.json").read_text(encoding="utf-8")) if args.arc.with_name("arc-test.fetch.json").is_file() else None
    manifest_m1 = {
        "kind": "iface-m1-set", "script_version": SCRIPT_VERSION, "git": _git(), "seed": args.seed, "rule": iface.RULE,
        "file": {"path": str(m1_dir / "m1-set.jsonl"), "sha256": m1_sha, "records": len(m1_records)},
        "items_file": {"path": str(m1_dir / "m1-items.jsonl"), "sha256": items_sha, "items": len(items)},
        "design": {"n_levels": list(iface.N_LEVELS), "listings_per_n": {str(k): v for k, v in iface.LISTINGS_BY_N.items()}, "buckets": [list(b) for b in iface.BUCKETS],
                   "late_min_n": iface.LATE_MIN_N, "arc_instruction": iface.ARC_INSTRUCTION, "physics_min_top": iface.PHYSICS_MIN_TOP,
                   "orders": "controlled positions; the prompt code shows the request order (seed None) — no reshuffle",
                   "cap": {"standard": list(OPTION_RANGE), "lifted_in_the_m1_script": list(iface.LIFTED_RANGE), "check": dict((str(k), v) for k, v in capped.items())}},
        "sources": {"ARC": {"rows": str(args.arc), "sha256": file_sha256(args.arc), "fetch": arc_fetch, "candidates": len(arc_all),
                            "dropped_by_filter": arc_dropped, "dropped_by_overlap": len(arc_all) - len(arc_clean),
                            "clean_by_subset": dict(Counter(i["source"] for i in arc_clean)),
                            "targets_by_subset": dict(Counter(i["source"] for i in arc_clean if iface.general_target(i))),
                            "min_answer_words": iface.MIN_ANSWER_WORDS, "min_distractor_words": iface.MIN_DISTRACTOR_WORDS,
                            "drawing_pool_options": sum(len(e["wrong"]) for e in index)},
                    **{name: {"file": str(getattr(args, name.lower())), "sha256": file_sha256(getattr(args, name.lower())), "direction_items": physics_counts[name],
                              "dropped_by_overlap": len(physics_all[name]) - len(physics_clean[name]),
                              "pair_groups_clean": len({i['group'] for i in physics_clean[name]})} for name in SETS}},
        "composition": {"items_by_source": dict(Counter(i["source"] for i in items)),
                        "physics_by_scene": dict(Counter(f"{i['source']}:{i['scene']}" for i in items if "scene" in i)),
                        "physics_answers": dict(Counter(i["correct"][0] for i in items if "scene" in i)),
                        "records_by_n": {str(n): len(rs) for n, rs in by_n.items()}, "correct_letter_by_n": positions},
        "prompt_tokens_by_n": m1_tokens, "prompts": len(m1_records), "prompt_tokens": sum(v.get("tokens", 0) for v in m1_tokens.values()),
        "overlap": {"units": "robo_jev.data.general_sets.record_units of every training record vs the candidates' text units (ARC: state + options; "
                             "physics and M2: state); ARC stems and options with >= 8 words also as 8-word shingles vs the general mix's texts",
                    "files": overlap["files"], "skipped": bool(overlap.get("skipped")),
                    "arc_hits": {k: v for k, v in hits.items() if k.startswith("arc:")},
                    "physics_hits": {k: v for k, v in hits.items() if k.startswith(("b1:", "hset:"))},
                    "chosen_items_with_hits": [i["item"] for i in items if i["item"] in hits]},
        "licence": {"ARC": "CC-BY-SA-4.0 (dataset card metadata); evaluation only, attributed (Clark et al. 2018); not committed or published"},
        "tokenizer": describe_tokenizer(tokenizer_id), "created": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
    }
    if manifest_m1["overlap"]["chosen_items_with_hits"]:
        raise SystemExit("a chosen M1 item overlaps a training input")
    (m1_dir / "m1-set.manifest.json").write_text(json.dumps(manifest_m1, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    _log(f"M1: {len(items)} items, {len(m1_records)} listings, {manifest_m1['prompt_tokens']} prompt tokens → {m1_dir}")
    # M2
    m2_dir = args.out / "m2"
    manifest_m2: dict[str, Any] = {"kind": "iface-m2-set", "script_version": SCRIPT_VERSION, "git": _git(), "seed": args.seed, "rule": iface.RULE,
                                   "fraction": args.m2_fraction, "selection": m2_selection, "lengths": iface.LENGTHS, "files": {}, "ids": m2_ids,
                                   "orders": M2_ORDERS,
                                   "short_reference": {name: {"file": str(getattr(args, name.lower())), "sha256": file_sha256(getattr(args, name.lower()))}
                                                       for name in SETS},
                                   "layout": ["ROLE", "ROBOT STATUS", "CAMERA TRACKS", "CURRENT SITUATION (the original state, verbatim)",
                                              "EARLIER OBSERVATIONS", "LOCAL PLAN (neutral recommendation)"],
                                   "overlap": {"m2_records_with_hits": 0, "skipped": bool(overlap.get("skipped"))}}
    combined: dict[str, list[dict[str, Any]]] = {"short": [], **{f"w{length}": [] for length in iface.LENGTHS}}
    for name, records in sets.items():
        by_id = {str(r["id"]): r for r in records}
        chosen = [by_id[rid] for rid in m2_ids[name]]
        combined["short"] += chosen
        manifest_m2["files"][name] = {"short": {"records": len(chosen), "tokens": prompt_tokens(chosen, seed=1, orders=M2_ORDERS, tokenizer=tokenizer)}}
        reference = iface.pair_reference_lengths(chosen)  # a one-fact pair gets exactly the same filler
        for length in iface.LENGTHS:
            wrapped = [iface.wrap_record(r, length=length, reference_length=reference[str(r["id"])]) for r in chosen]
            bad = {}
            for original, w in zip(chosen, wrapped):
                found = iface.sheet_problems(w["request"]["state"], original["request"]["state"]) + validate_record(w)
                if w["request"]["questions"] != original["request"]["questions"] or w["labels"] != original["labels"] or w["meta"] != original["meta"] or w["id"] != original["id"]:
                    found.append("questions, labels, meta or id changed")
                if found:
                    bad[w["id"]] = found
            fillers: dict[str, set[str]] = defaultdict(set)
            for original, w in zip(chosen, wrapped):
                fillers[shuffle_key(original)].add(w["request"]["state"].replace(original["request"]["state"].strip(), "<STATE>"))
            bad.update({f"pair {k}": ["the pair's fillers differ"] for k, v in fillers.items() if len(v) != 1})
            if bad:
                raise SystemExit(f"M2 {name} {length}: {len(bad)} sheets fail their checks, first {list(bad.items())[:2]}")
            combined[f"w{length}"] += wrapped
            chars = sorted(len(w["request"]["state"]) for w in wrapped)
            manifest_m2["files"][name][f"w{length}"] = {"records": len(wrapped), "chars": {"min": chars[0], "median": chars[len(chars) // 2], "max": chars[-1]},
                                                        "tokens": prompt_tokens(wrapped, seed=1, orders=M2_ORDERS, tokenizer=tokenizer)}
            example = m2_dir / "examples" / f"{name}-w{length}-{wrapped[0]['id']}.txt"
            example.parent.mkdir(parents=True, exist_ok=True)
            example.write_text(wrapped[0]["request"]["state"] + "\n", encoding="utf-8")
            _log(f"M2 {name} w{length}: {len(wrapped)} records, chars {chars[0]}–{chars[-1]}, {manifest_m2['files'][name][f'w{length}']['tokens']}")
    for cond, rows in combined.items():  # one file per condition (B1 then the H-set; ids are distinct) — one scoring unit per model and condition
        path = m2_dir / f"M2-{cond}.jsonl"
        ids = [str(r["id"]) for r in rows]
        if len(set(ids)) != len(ids):
            raise SystemExit(f"M2 {cond}: duplicate ids across the sets")
        manifest_m2["files"][f"M2-{cond}"] = {"path": str(path), "sha256": _write_jsonl(path, rows), "records": len(rows)}
    manifest_m2["created"] = dt.datetime.now().astimezone().isoformat(timespec="seconds")
    manifest_m2["seconds"] = round(time.perf_counter() - started, 1)
    (m2_dir / "m2-set.manifest.json").write_text(json.dumps(manifest_m2, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    _log(f"done in {manifest_m2['seconds']} s")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--arc", type=Path, default=ARC)
    parser.add_argument("--b1", type=Path, default=B1)
    parser.add_argument("--hset", type=Path, default=HSET)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--seed", type=int, default=20261014)
    parser.add_argument("--arc-easy", type=int, default=450)
    parser.add_argument("--arc-challenge", type=int, default=450)
    parser.add_argument("--physics-b1", type=int, default=150)
    parser.add_argument("--physics-hset", type=int, default=150)
    parser.add_argument("--m2-fraction", type=float, default=0.25, help="share of each family's / scene's records in the M2 sample (whole pairs)")
    parser.add_argument("--skip-overlap", action="store_true", help="tests only: skip the training-input scan")
    args = parser.parse_args(argv)
    for path in (args.arc, args.b1, args.hset, args.out):
        refuse_sealed(path)
    return build(args)


if __name__ == "__main__":
    raise SystemExit(main())
