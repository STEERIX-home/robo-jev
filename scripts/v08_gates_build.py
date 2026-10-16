#!/usr/bin/env python
"""Task v0.8b — build gate M8g (many options, sealed), the many-option part of the Jev comparison set and the storage-bin probe cases (CPU);
re-check the gates against v0.8's training input once it exists; the Spark chain's gate check.

    artifacts/scratch/v08b/cpu.sh m8g scripts/v08_gates_build.py build-m8g          # → artifacts/v08/gates/m8g/, artifacts/v08/jev/jev-many.jsonl
    artifacts/scratch/v08b/cpu.sh probes scripts/v08_gates_build.py build-probes    # → artifacts/v08/probes/bins.jsonl
    artifacts/scratch/v08b/cpu.sh jev scripts/v08_gates_build.py merge-jev          # jev-bench.jsonl + jev-many.jsonl → jev-set.jsonl
    artifacts/scratch/v08b/cpu.sh recheck scripts/v08_gates_build.py recheck-v08    # once artifacts/simgen/v08/train-v08.jsonl exists
    scripts/v08_gates_build.py check --runs <fetched run dirs…>                      # the Spark chain's gate (exit 2 on any problem)

Design: :mod:`robo_jev.bench.gates_v08` (M8g) and :mod:`robo_jev.bench.b8_v08` (B8). Inputs: the MMLU test rows task v0.7b fetched
(``artifacts/v07/gates/m1g/source/mmlu-test.jsonl``, MIT), M1g's item list (only to leave its items out) and every training input before v0.8
(:data:`TRAINING_INPUTS`, read in code; only counts and hashes are written). Nothing of task v0.8a is opened: ``train-v08.jsonl`` is read only
by ``recheck-v08``, in code, and only counts and hashes are written.

**build-m8g**

1. *MMLU candidates*: M1g's filter (:func:`robo_jev.bench.gates_v07.mmlu_items`), minus M1g's items (never measured or drawn here), minus any
   candidate sharing a text unit or an 8-word shingle with a training input (the v0.7 scan, :func:`scan`, on :data:`TRAINING_INPUTS` and
   ``train-v07.jsonl``). The rest is split by a seeded shuffle into the M8g side (3/4) and the Jev side (1/4); each side draws its
   distractors only from its own items, so nothing of M8g is ever sent to the API.
2. *Physics items*: :func:`robo_jev.bench.gates_v08.physics_item`, :data:`PHYSICS_PER_FAMILY` per family for M8g (indices 0…) and a separate
   index range for the Jev side; their states and options are checked against the training inputs too (whole state, 8-word shingles).
3. *Listings*: :func:`robo_jev.bench.gates_v08.expand_m8g` at N = 6, 18, 40, 120, 255, two listings each; every record passes
   :func:`robo_jev.bench.gates_v08.gate_problems`; the answer is the same at every N and the option sets are nested.
4. *Files* (read-only after writing): ``m8g/M8g.jsonl`` (registered), ``m8g/m8g-items.jsonl``, ``m8g/m8g.manifest.json`` (design, licence,
   counts, positions, prompt tokens per N through the runner's prompt builder, GPU estimate), ``overlap.json``, the audit samples
   ``audit/m8g-audit-n40.md`` and ``audit/m8g-audit-n255.md``; ``artifacts/v08/jev/jev-many.jsonl`` (not sealed: one listing per item, 15 items
   per N).

**recheck-v08** — the same scan against ``train-v08.jsonl`` alone; writes ``overlap-v08.json``: the M8g items (any unit or shingle hit for
MMLU; a whole-state or state / correct-option shingle hit for physics) and the B8 requests (whole-state hit) to leave out of every reading,
plus B8's shingle hits (reported). Pre-declared here, before any v0.8 data exists.

**check** — the Spark chain's gate: the sealed files are unchanged (M8g: the manifest's sha256; B8: ``frozen-before-scoring.json``),
``overlap-v08.json`` exists and was computed on the training input each given run trained on (``metrics.json`` → ``manifest.data.file_sha256``),
the opt-in is the one main accepts (:func:`robo_jev.decisions.records.option_range_for` gives 2–255 for the gate records without a problem)
and the package's answer codes for 255 options are the runner's ``CODES[:255]``.
"""

from __future__ import annotations

import argparse
import datetime as dt
import functools
import importlib.util
import json
import stat
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

SCRIPT_VERSION = "v08-gates-build-1"
ARTIFACTS = REPO / "artifacts"
MMLU = ARTIFACTS / "v07" / "gates" / "m1g" / "source" / "mmlu-test.jsonl"
MMLU_FETCH = MMLU.parent / "mmlu-test.fetch.json"
M1G_ITEMS = ARTIFACTS / "v07" / "gates" / "m1g" / "m1g-items.jsonl"
#: Every training input before v0.8 (task v0.7b's list and v0.7's input) — assembled files contain their parts.
TRAINING_INPUTS = (
    "simgen/v0b/train-50k.jsonl", "simgen/v01/train-50k.jsonl", "simgen/v01-mix/train-50k-mix.jsonl", "simgen/probes/G1-train-mix.jsonl",
    "simgen/v02/train-50k.jsonl", "simgen/v02/general-train.jsonl", "simgen/v02/train-v02-a.jsonl", "simgen/v02/train-v02-b.jsonl",
    "simgen/v03/train-v03.jsonl", "simgen/v04/train-v04.jsonl", "simgen/v05/train-v05.jsonl", "simgen/v06/train-v06.jsonl",
    "simgen/v07/train-v07.jsonl",
)
TRAIN_V08 = "simgen/v08/train-v08.jsonl"
SEED = 20261016
#: M8g size: MMLU targets and physics items per family; the Jev side: per N level, MMLU and physics items (one listing each).
M8G_MMLU, PHYSICS_PER_FAMILY = 160, 10
JEV_MMLU_PER_N, JEV_PHYS_PER_N = 12, 3
#: The Jev side's physics items use indices from here (disjoint scenarios and candidate numbers from M8g's).
JEV_PHYSICS_OFFSET = 500
#: GPU seconds per prompt token on the Spark at batch 1 (task v0.7b's smoke test: 0.20 ms LoRA, 0.12 ms base).
SECONDS_PER_TOKEN = {"lora": 0.20e-3, "base": 0.12e-3}
NEVER_WRITE = (ARTIFACTS / "simgen", ARTIFACTS / "t8", ARTIFACTS / "t9", ARTIFACTS / "b1", ARTIFACTS / "v06", ARTIFACTS / "v07", ARTIFACTS / "iface",
               ARTIFACTS / "reports")


def _log(message: str) -> None:
    print(f"[v08-gates {dt.datetime.now().strftime('%H:%M:%S')}] {message}", file=sys.stderr, flush=True)


@functools.lru_cache(maxsize=None)
def V7B():
    """scripts/v07_gates_build.py — the overlap scan (text units and 8-word shingles) and the file helpers."""
    spec = importlib.util.spec_from_file_location("v08_v07_gates_build", REPO / "scripts" / "v07_gates_build.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


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
    return Path(path)


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


def _rel(path: Path) -> str:
    try:
        return str(Path(path).relative_to(REPO))
    except ValueError:
        return str(path)


def _now() -> str:
    return dt.datetime.now().astimezone().isoformat(timespec="seconds")


# --------------------------------------------------------------------------
# Overlap candidates
# --------------------------------------------------------------------------


def physics_texts(item: dict[str, Any]) -> list[str]:
    """The texts of a physics item checked against training inputs: its state and its correct option (the distractors share the option
    template)."""
    return [item["state"], item["correct"][1]]


def physics_candidates(items: list[dict[str, Any]]) -> tuple[dict[str, set[str]], dict[str, set[str]]]:
    """(unit hash → items) — the whole normalised state — and (8-word shingle → items) of the state and the correct option."""
    from robo_jev.bench.iface import shingles
    from robo_jev.data.general_sets import normalize

    units: dict[str, set[str]] = {}
    grams: dict[str, set[str]] = {}
    for item in items:
        units.setdefault(V7B()._hash(normalize(item["state"])), set()).add(item["item"])
        for text in physics_texts(item):
            for line in text.splitlines():
                if len(line.split()) >= 8:
                    for gram in shingles(line):
                        grams.setdefault(gram, set()).add(item["item"])
    return units, grams


def merged(*pairs: tuple[dict[str, set[str]], dict[str, set[str]]]) -> tuple[dict[str, set[str]], dict[str, set[str]]]:
    units: dict[str, set[str]] = {}
    grams: dict[str, set[str]] = {}
    for u, g in pairs:
        for k, v in u.items():
            units.setdefault(k, set()).update(v)
        for k, v in g.items():
            grams.setdefault(k, set()).update(v)
    return units, grams


# --------------------------------------------------------------------------
# Tokens (the runner's prompt builder — the package's stops at 26 options before task v0.8a)
# --------------------------------------------------------------------------


@functools.lru_cache(maxsize=1)
def _tokenizer():
    from robo_jev.model.tokenizer import available_tokenizer, load_tokenizer

    return load_tokenizer(available_tokenizer()[0])


def prompt_tokens(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Prompt tokens of each record's questions in the record's order (seed None) through the runner's builder (A–Z, then two-letter codes —
    byte-identical to the package's prompt up to 26 options, task rel071)."""
    from robo_jev.hf.robo_jev_decide import build_prompts

    tok = _tokenizer()
    lengths = sorted(len(p["tokens"]) for r in records for p in build_prompts(r["request"]["state"], r["request"]["questions"], tok, seed=None))
    if not lengths:
        return {"prompts": 0, "tokens": 0}
    return {"prompts": len(lengths), "tokens": sum(lengths), "mean": round(sum(lengths) / len(lengths), 1), "max": lengths[-1]}


def gpu_seconds(tokens: int) -> dict[str, int]:
    return {kind: round(tokens * rate) for kind, rate in SECONDS_PER_TOKEN.items()}


# --------------------------------------------------------------------------
# build-m8g
# --------------------------------------------------------------------------


def _item_line(item: dict[str, Any], drawn: list[Any], rank: int) -> dict[str, Any]:
    from robo_jev.bench import gates_v08 as G

    base = {"item": item["item"], "source": item["source"], "rank": rank, "state": item["state"], "instructions": item["instructions"],
            "correct": item["correct"]}
    if item["source"] == G.GENERAL_SOURCE:
        return {**base, "stem": item["stem"], "subject": item["subject"], "own": item["own"], "drawn": drawn}
    return {**base, "family": item["family"], "constraints": item["constraints"], "drawn": drawn, "broken": item["broken"][: len(drawn)]}


def _check_records(records: list[dict[str, Any]]) -> None:
    from robo_jev.bench import gates_v08 as G

    problems = {r["id"]: p for r in records if (p := G.gate_problems(r))}
    if problems:
        raise SystemExit(f"{len(problems)} records fail their checks: {list(problems.items())[:3]}")
    ids = [r["id"] for r in records]
    if len(set(ids)) != len(ids):
        raise SystemExit("duplicate record ids")
    by_item: dict[str, list[dict[str, Any]]] = {}
    for r in records:
        by_item.setdefault(r["meta"]["item"], []).append(r)
    for item, rows in by_item.items():  # the same item and answer at every N; the option sets nested across N
        q = G.M8G_QUESTION
        answers = {r["request"]["questions"][q]["criteria"][r["meta"]["correct"]] for r in rows}
        sets = {}
        for r in rows:
            sets.setdefault(r["meta"]["n"], set()).update(r["request"]["questions"][q]["criteria"].values())
        levels = sorted(sets)
        if len(answers) != 1 or any(not sets[a] <= sets[b] for a, b in zip(levels, levels[1:])):
            raise SystemExit(f"{item}: the answer differs across N or the option sets are not nested")
        for r in rows:
            shown = list(r["request"]["questions"][q]["criteria"])
            if shown.index(r["meta"]["correct"]) != r["meta"]["position"] or G.code_of(r["meta"]["position"]) != r["meta"]["code"]:
                raise SystemExit(f"{r['id']}: position / code do not match the shown order")


def audit_sheet(records: list[dict[str, Any]], out: Path, *, n: int, general: int, physics: int, seed: int) -> dict[str, Any]:
    """A seeded sample of items at N = `n` (listing 0) as a sheet for the ambiguity audit: question, the right option, the others."""
    from robo_jev.bench import gates_v08 as G

    by_item = {r["meta"]["item"]: r for r in records if r["meta"]["n"] == n and r["meta"]["listing"] == 0}
    names = sorted(by_item)
    r = G.rng("v08-m8g-audit", n, seed)
    gen = [x for x in names if by_item[x]["meta"]["source"] == G.GENERAL_SOURCE]
    phy = [x for x in names if by_item[x]["meta"]["source"] != G.GENERAL_SOURCE]
    picked = r.sample(gen, min(general, len(gen))) + r.sample(phy, min(physics, len(phy)))
    lines = [f"# M8g ambiguity audit sample (N = {n}, listing 0)", "",
             "For each item: is any other option also a right (or equally good) answer? Physics items are checked by construction too.", ""]
    for i, name in enumerate(picked, 1):
        rec = by_item[name]
        q = rec["request"]["questions"][G.M8G_QUESTION]
        right = q["criteria"][rec["meta"]["correct"]]
        lines += [f"## {i}. {name} ({rec['meta'].get('subject') or rec['meta'].get('family')})", "", "State: " + rec["request"]["state"].replace("\n", " "),
                  "", "Question: " + q["instructions"], "", f"**Right:** {right}", ""]
        lines += [f"- {text}" for key, text in q["criteria"].items() if key != rec["meta"]["correct"]]
        lines.append("")
    refuse_write(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"file": _rel(out), "n": n, "items": picked}


def physics_violations(item: dict[str, Any]) -> list[str]:
    """Construction check of a physics item: re-derive, from the option texts and the state, that the correct candidate meets every
    constraint and every distractor breaks one — numbers parsed from the texts (an independent check of the generator's bookkeeping)."""
    import re

    def nums(text: str) -> list[float]:
        return [float(x) for x in re.findall(r"(\d+(?:\.\d+)?)", text)]

    state = item["state"]
    fam = item["family"]
    out = []
    candidates = [item["correct"]] + [tuple(c) for c in item["pool"]]
    sv = nums(state)
    for idx, (key, text) in enumerate(candidates):
        body = text.split(": ", 1)[1]
        v = nums(body)
        if fam == "bins":
            m, w, reach = sv[0], sv[1], sv[5]
            ok = v[0] >= m and v[1] > w and v[2] <= reach and "empty with its door open" in body
        elif fam == "shelves":
            m, d, lift = sv[0], sv[1], sv[2]
            ok = v[0] >= m and v[1] >= d and v[2] <= lift and "level and dry" in body
        elif fam == "spots":
            m, cw, smax = sv[0], sv[1], sv[2]
            ok = v[0] <= smax and v[1] > cw and body.split(", ")[2].startswith("dry") and body.rstrip(".").endswith(", free")
        else:
            w, step = sv[0], sv[1]
            th = 0.0 if "flush threshold" in body else v[1]
            ok = v[0] > w and th < step and body.rstrip(".").endswith(", open")
        if (idx == 0) != ok:
            out.append(f"{item['item']} {key}: {'correct candidate fails' if idx == 0 else 'distractor meets every constraint'}: {text}")
    return out


def cmd_build_m8g(args: argparse.Namespace) -> int:
    from robo_jev.bench import gates_v07 as G7
    from robo_jev.bench import gates_v08 as G
    from robo_jev.bench import iface
    from robo_jev.decisions.records import file_sha256, read_records
    from robo_jev.model.tokenizer import available_tokenizer, describe_tokenizer

    started = time.perf_counter()
    out = refuse_write(args.out)
    rows = read_records(MMLU)
    mmlu_all, dropped = G7.mmlu_items(rows)
    m1g = {str(i["item"]) for i in read_records(M1G_ITEMS)}
    in_m1g = [i for i in mmlu_all if i["item"] in m1g]
    referring = [i["item"] for i in mmlu_all if G.refers_to_options(i)]
    mmlu_free = [i for i in mmlu_all if i["item"] not in m1g and not G.refers_to_options(i)]
    _log(f"MMLU: {len(rows)} rows, {len(mmlu_all)} candidates after M1g's filter, {len(in_m1g)} in M1g left out, "
         f"{len(referring)} with an option referring to other options left out, {len(mmlu_free)} free")
    phys_m8g = [G.physics_item(f, k, seed=args.seed) for f in G.PHYSICS_FAMILIES for k in range(PHYSICS_PER_FAMILY)]
    phys_jev = [G.physics_item(f, JEV_PHYSICS_OFFSET + k, seed=args.seed, numbers=(1000, 10000)) for f in G.PHYSICS_FAMILIES for k in range(len(G.M8G_LEVELS))]
    bad = [p for item in phys_m8g + phys_jev for p in physics_violations(item)]
    if bad:
        raise SystemExit(f"{len(bad)} physics construction problems: {bad[:3]}")
    units, grams = merged(V7B().candidates(mmlu_free, {}), physics_candidates(phys_m8g + phys_jev))
    files = [ARTIFACTS / f for f in TRAINING_INPUTS]
    missing = [_rel(f) for f in files if not f.is_file()]
    if missing:
        raise SystemExit(f"training inputs missing: {missing}")
    overlap = {"hits": {}, "files": {}, "skipped": True} if args.skip_overlap else V7B().scan(files, units, grams, set())
    hits = overlap["hits"]
    phys_hit = sorted(k for k in hits if k.startswith("phys:"))
    if phys_hit:
        raise SystemExit(f"{len(phys_hit)} physics items share text with a training input: {phys_hit[:3]}")
    clean = [i for i in mmlu_free if i["item"] not in hits]
    _log(f"MMLU: {len(mmlu_free) - len(clean)} free candidates with an overlap hit dropped; {len(clean)} clean")
    # the two sides (seeded), each with its own drawing pool
    pool = sorted(clean, key=lambda x: x["item"])
    G.rng("v08-m8g-sides", args.seed).shuffle(pool)
    cut = (3 * len(pool)) // 4
    sides = {"m8g": pool[:cut], "jev": pool[cut:]}
    index = {side: iface.general_index(items) for side, items in sides.items()}
    targets = {side: [i for i in items if i["target"] and i["item"] not in G.KNOWN_KEY_ERRORS] for side, items in sides.items()}
    if len(targets["m8g"]) < args.mmlu_items or len(targets["jev"]) < JEV_MMLU_PER_N * len(G.M8G_LEVELS):
        raise SystemExit(f"not enough clean targets: {[(s, len(t)) for s, t in targets.items()]}")
    chosen = targets["m8g"][: args.mmlu_items] + phys_m8g
    G.rng("v08-m8g-rank", args.seed).shuffle(chosen)
    records, lines = [], []
    for rank, item in enumerate(chosen):
        if item["source"] == G.GENERAL_SOURCE:
            drawn = G.draw_general_many(item, index["m8g"], count=max(G.M8G_LEVELS) - 4, seed=args.seed)
        else:
            drawn = [tuple(c) for c in item["pool"][: max(G.M8G_LEVELS) - 1]]
        records += G.expand_m8g(item, drawn, rank=rank, index=rank, seed=args.seed)
        lines.append(_item_line(item, drawn, rank))
    _check_records(records)
    _log(f"M8g: {len(chosen)} items, {len(records)} listings, every record checked")
    # the Jev side: one listing per item, 15 items per N (12 MMLU + 3 physics), positions alternating single / two-letter at N ≥ 40. MMLU
    # repeats some option texts across questions, so the Jev side also skips every text M8g shows (normalised) — nothing of M8g is sent.
    from robo_jev.data.general_sets import normalize

    m8g_norm = {normalize(t) for r in records for t in r["request"]["questions"][G.M8G_QUESTION]["criteria"].values()}
    m8g_norm |= {normalize(i["stem"]) for i in chosen if "stem" in i}
    index["jev"] = [{**e, "wrong": [(t, tokens) for t, tokens in e["wrong"] if normalize(t) not in m8g_norm]} for e in index["jev"]]
    jev_free = [i for i in targets["jev"] if not {normalize(t) for t in [i["stem"], i["correct"], *i["own"]]} & m8g_norm]
    jev_mmlu = jev_free[: JEV_MMLU_PER_N * len(G.M8G_LEVELS)]
    jev_records, jev_lines = [], []
    for li, n in enumerate(G.M8G_LEVELS):
        group = jev_mmlu[li * JEV_MMLU_PER_N:(li + 1) * JEV_MMLU_PER_N] + [p for p in phys_jev if int(p["item"].split("-")[-1]) - JEV_PHYSICS_OFFSET == li][:JEV_PHYS_PER_N]
        G.rng("v08-jev-many-order", n, args.seed).shuffle(group)
        for j, item in enumerate(group):
            rank = len(jev_lines)
            if item["source"] == G.GENERAL_SOURCE:
                drawn = G.draw_general_many(item, index["jev"], count=max(n - 4, 0), seed=args.seed) if n > 4 else []
            else:
                drawn = [tuple(c) for c in item["pool"][: n - 1]]
            recs = G.expand_m8g(item, drawn, rank=rank, index=rank, seed=args.seed, levels=(n,), listings=1, prefix="v08jev-m-", set_name="v08-jev")
            for rec in recs:
                rec["meta"]["part"] = "many"
                rec["meta"]["target"] = [G.M8G_QUESTION]
            jev_records += recs
            jev_lines.append(_item_line(item, drawn, rank))
    jev_problems = {r["id"]: p for r in jev_records if (p := G.gate_problems(r))}
    if jev_problems:
        raise SystemExit(f"Jev many-option records fail their checks: {list(jev_problems.items())[:3]}")
    m8g_items = {i["item"] for i in chosen}
    jev_items = {r["meta"]["item"] for r in jev_records}
    m8g_texts = {normalize(t) for r in records for t in r["request"]["questions"][G.M8G_QUESTION]["criteria"].values()} | {normalize(r["request"]["state"]) for r in records}
    jev_texts = {normalize(t) for r in jev_records for t in r["request"]["questions"][G.M8G_QUESTION]["criteria"].values()} | {normalize(r["request"]["state"]) for r in jev_records}
    shared = m8g_texts & jev_texts
    if m8g_items & jev_items or shared:
        raise SystemExit(f"the Jev side shares items ({len(m8g_items & jev_items)}) or texts ({len(shared)}) with M8g: {sorted(shared)[:3]}")
    # files
    m_dir = out / "m8g"
    m_path = m_dir / "M8g.jsonl"
    m_sha = _write_jsonl(m_path, records)
    items_sha = _write_jsonl(m_dir / "m8g-items.jsonl", lines)
    jev_path = refuse_write(args.jev_out)
    jev_sha = _write_jsonl(jev_path, jev_records)
    _write_jsonl(jev_path.with_name("jev-many-items.jsonl"), jev_lines)
    # tokens and time
    tokens = {}
    for n in G.M8G_LEVELS:
        for listing in range(G.M8G_LISTINGS):
            tokens[f"{n}/{listing}"] = prompt_tokens([r for r in records if r["meta"]["n"] == n and r["meta"]["listing"] == listing])
    by_n = {str(n): {"tokens": sum(tokens[f"{n}/{l}"]["tokens"] for l in range(G.M8G_LISTINGS)),
                     "prompts": sum(tokens[f"{n}/{l}"]["prompts"] for l in range(G.M8G_LISTINGS)),
                     "mean": round(sum(tokens[f"{n}/{l}"]["tokens"] for l in range(G.M8G_LISTINGS)) / max(1, sum(tokens[f"{n}/{l}"]["prompts"] for l in range(G.M8G_LISTINGS))), 1),
                     "max": max(tokens[f"{n}/{l}"]["max"] for l in range(G.M8G_LISTINGS))} for n in G.M8G_LEVELS}
    by_source = {src: prompt_tokens([r for r in records if r["meta"]["source"] == src and r["meta"]["n"] == 255]) for src in (G.GENERAL_SOURCE, G.PHYSICS_SOURCE)}
    total = sum(v["tokens"] for v in by_n.values())
    listing0 = sum(tokens[f"{n}/0"]["tokens"] for n in G.M8G_LEVELS)
    jev_tokens = {str(n): prompt_tokens([r for r in jev_records if r["meta"]["n"] == n]) for n in G.M8G_LEVELS}
    audits = [audit_sheet(records, out / "audit" / f"m8g-audit-n{n}.md", n=n, general=g, physics=p, seed=args.seed)
              for n, g, p in ((40, 30, 8), (255, 4, 2))]
    fetch = json.loads(MMLU_FETCH.read_text(encoding="utf-8"))
    positions = {str(n): {str(l): dict(sorted(Counter(r["meta"]["code"] for r in records if r["meta"]["n"] == n and r["meta"]["listing"] == l).items()))
                          for l in range(G.M8G_LISTINGS)} for n in G.M8G_LEVELS}
    manifest = {
        "kind": "v08-m8g", "script_version": SCRIPT_VERSION, "git": _git(), "seed": args.seed, "set": G.M8G_SET, "prefix": G.M8G_PREFIX,
        "file": {"path": _rel(m_path), "sha256": m_sha, "records": len(records)},
        "items_file": {"path": _rel(m_dir / "m8g-items.jsonl"), "sha256": items_sha, "items": len(lines)},
        "option_range": G.option_range_mark(), "option_range_version_confirmed": G.OPTION_RANGE_VERSION_CONFIRMED,
        "codes": {"source": "robo_jev.hf.robo_jev_decide.CODES[:255] (the runner's, in order: A-Z, then two-letter codes)",
                  "first": list(G.CODES_255[:3]), "27th": G.CODES_255[26], "last": G.CODES_255[-1]},
        "design": {"levels": list(G.M8G_LEVELS), "listings": G.M8G_LISTINGS, "question": G.M8G_QUESTION,
                   "positions": "N <= 26: listing 0 in the first half, listing 1 in the second; N >= 40: listing 0 on a single-letter code (A-Z), "
                                "listing 1 on a two-letter code; spread over the band with the item's rank (gates_v08.m8g_position)",
                   "orders": "the record's own order (scored with seed None): codes assigned in order, A-Z first, as at inference",
                   "general": "MMLU: state 'question: \"<stem>\"', instruction '" + iface.ARC_INSTRUCTION + "', keys o1..oN (seeded), the three own wrong "
                              f"options then drawn texts (the first {G.STRICT_DRAWS} by task iface's strict rule, the rest topic-distant from the item; one per "
                              "source question)",
                   "physics": "fresh items written for this gate: four families (bins, shelves, spots, doors) of same-family candidates; exactly one "
                              "meets every constraint the state gives (with a margin), every other breaks exactly one (cycled); checked by "
                              "construction and re-derived from the texts (physics_violations)"},
        "composition": {"items": dict(Counter(i["source"] for i in chosen)), "physics_families": dict(Counter(i["family"] for i in chosen if "family" in i)),
                        "mmlu_subjects": dict(Counter(i["subject"] for i in chosen if "subject" in i).most_common()),
                        "records_by_n": dict(Counter(str(r["meta"]["n"]) for r in records)),
                        "two_letter_by_n": {str(n): sum(r["meta"]["two_letter"] for r in records if r["meta"]["n"] == n) for n in G.M8G_LEVELS},
                        "distinct_codes_by_n_listing": {k: {l: len(v) for l, v in d.items()} for k, d in positions.items()}},
        "positions": positions,
        "source": {"MMLU": {"repo": fetch.get("repo"), "revision": fetch.get("revision"), "file": fetch.get("file"), "licence": fetch.get("license"),
                            "licence_evidence": {"card_metadata": (fetch.get("card") or {}).get("license_metadata"),
                                                 "card_text": (fetch.get("card") or {}).get("license_text")},
                            "citation": fetch.get("citation"), "rows": len(rows), "candidates": len(mmlu_all), "dropped_by_filter": dropped,
                            "left_out_in_m1g": len(in_m1g), "left_out_referring_to_options": len(referring), "referring_items": referring,
                            "never_measured_known_key_errors": list(G.KNOWN_KEY_ERRORS),
                            "dropped_by_overlap": len(mmlu_free) - len(clean), "clean": len(clean),
                            "sides": {s: {"items": len(v), "targets": len(targets[s])} for s, v in sides.items()},
                            "why": "MMLU test (MIT): used in no training input (checked: text units and 8-word shingles), not M1 (ARC), and not M1g's items"},
                   "PHYS": {"licence": "written for this gate (task v0.8b); own work", "families": list(G.PHYSICS_FAMILIES),
                            "per_family": PHYSICS_PER_FAMILY}},
        "prompt_tokens": {"by_n": by_n, "by_n_listing": tokens, "at_255_by_source": by_source, "total": total, "listing0_total": listing0},
        "gpu_estimate_s": {"all_listings_per_model": gpu_seconds(total), "listing0_only_per_model": gpu_seconds(listing0)},
        "audit": audits,
        "overlap": {"method": "the v0.7 scan (scripts/v07_gates_build.py scan): text units of every training record and 8-word shingles of every "
                              "training text vs the MMLU candidates' units and stem / option shingles; physics: whole state and the shingles of the "
                              "state and the correct option", "training_inputs": list(TRAINING_INPUTS), "skipped": bool(overlap.get("skipped")),
                    "mmlu_dropped": len(mmlu_free) - len(clean), "physics_hits": 0},
        "created": _now(),
    }
    _write_json(m_dir / "m8g.manifest.json", manifest)
    _write_json(out / "overlap.json", {"kind": "v08-gates-overlap", "script_version": SCRIPT_VERSION, "files": overlap.get("files"), "hits": hits,
                                       "skipped": bool(overlap.get("skipped")), "created": _now()})
    jev_manifest = {"kind": "v08-jev-many", "script_version": SCRIPT_VERSION, "git": _git(), "seed": args.seed, "file": _rel(jev_path),
                    "sha256": jev_sha, "records": len(jev_records), "by_n": dict(Counter(str(r["meta"]["n"]) for r in jev_records)),
                    "by_source": dict(Counter(r["meta"]["source"] for r in jev_records)), "prompt_tokens": jev_tokens,
                    "independence": {"items_shared_with_m8g": 0, "texts_shared_with_m8g": 0, "items_in_m1g": 0,
                                     "rule": "the MMLU pool split by a seeded shuffle into the M8g side and the Jev side; each side draws only from its "
                                             "own items; Jev physics items use other indices"},
                    "not_sealed": "sent to OpenRouter (Jev) with provider.data_collection = deny", "created": _now()}
    _write_json(jev_path.with_name("jev-many.manifest.json"), jev_manifest)
    _seal([m_path, m_dir / "m8g-items.jsonl"])
    _log(f"done in {time.perf_counter() - started:.0f} s · M8g {len(records)} records sha256 {m_sha[:12]} · tokens {total:,} "
         f"(per LoRA model {gpu_seconds(total)['lora']} s) · Jev many {len(jev_records)} records")
    _log("tokens by N: " + json.dumps(by_n))
    return 0


# --------------------------------------------------------------------------
# build-probes — the 20 storage-bin cases of task rel071 (artifacts/scratch/rel071/gpu_probe*.py)
# --------------------------------------------------------------------------


def bins(n: int, seed: int) -> tuple[dict, str]:
    """task rel071's storage-bin choice (artifacts/scratch/rel071/gpu_check.py ``bins``, copied verbatim): `n` bins, exactly one of which
    can take the box."""
    import random

    rng = random.Random(seed)
    right = rng.randrange(n)
    reasons = ["full to the top", "rated for 5 kg only", "blocked by a pallet in front of it", "above the arm's reach at 2.4 m",
               "holding open chemical containers", "marked out of service", "its door jammed shut", "reserved for frozen goods"]
    criteria = {}
    for i in range(n):
        if i == right:
            criteria[f"bin_{i:03d}"] = f"Bin {i}: empty, rated for 25 kg, at waist height, door open."
        else:
            criteria[f"bin_{i:03d}"] = f"Bin {i}: {rng.choice(reasons)}."
    state = ("You are a mobile manipulator in a parts store, holding a closed 12 kg box of steel fittings in both grippers.\n"
             "- The box is 40 x 30 x 25 cm and must be stored now.\n- The aisle is clear; nobody is near you.\n"
             f"- The store has {n} bins along this aisle; their states are listed with the options.")
    question = {"type": "choice", "instructions": "Which bin should the robot put the box into?", "criteria": criteria}
    return {"state": state, "questions": {"bin": question}}, f"bin_{right:03d}"


#: gpu_probe.py: the right bin moved to these positions (request order, seed None); then the base requests in seeded single orders 1..8
#: (gpu_probe.py) and in balanced order with seeds 1..4 (gpu_probe2.py).
PROBE_POSITIONS = {46: (0, 5, 12, 25, 26, 27, 35, 45), 100: (0, 5, 12, 25, 26, 27, 35, 60, 80, 99)}
PROBE_SINGLE_SEEDS = tuple(range(1, 9))
PROBE_BALANCED_SEEDS = tuple(range(1, 5))


def probe_rows() -> list[dict[str, Any]]:
    out = []
    for n, seed in ((46, 46), (100, 100)):
        request, right = bins(n, seed)
        keys = [k for k in request["questions"]["bin"]["criteria"] if k != right]
        for position in PROBE_POSITIONS[n]:
            order = keys[:position] + [right] + keys[position:]
            moved = {**request, "questions": {"bin": {**request["questions"]["bin"], "criteria": {k: request["questions"]["bin"]["criteria"][k] for k in order}}}}
            out.append({"id": f"bins-{n}@{position}", "meta": {"set": "v08-probe", "n": n, "right": right, "position": position, "kind": "position",
                                                               "target": ["bin"], "readings": [{"name": "order-none", "orders": "single", "seed": None}]},
                        "request": moved})
        out.append({"id": f"bins-{n}", "meta": {"set": "v08-probe", "n": n, "right": right, "position": None, "kind": "base", "target": ["bin"],
                                                "readings": [{"name": f"single-s{s}", "orders": "single", "seed": s} for s in PROBE_SINGLE_SEEDS]
                                                + [{"name": f"balanced-s{s}", "orders": "balanced", "seed": s} for s in PROBE_BALANCED_SEEDS]},
                    "request": request})
    return out


def cmd_build_probes(args: argparse.Namespace) -> int:
    rows = probe_rows()
    path = refuse_write(args.out)
    sha = _write_jsonl(path, rows)
    tokens = prompt_tokens([{"request": r["request"]} for r in rows])
    _write_json(path.with_suffix(".manifest.json"), {
        "kind": "v08-probes", "script_version": SCRIPT_VERSION, "file": _rel(path), "sha256": sha, "cases": len(rows),
        "source": "task rel071's storage-bin probe (artifacts/scratch/rel071/gpu_probe.py, gpu_probe2.py; bins() copied verbatim): 18 position "
                  "cases (the right bin moved, request order) + the 2 base requests (seeded single orders 1-8, balanced seeds 1-4)",
        "prompt_tokens_one_listing": tokens, "reported_only": True, "created": _now()})
    _log(f"{len(rows)} probe cases → {path} ({sha[:12]})")
    return 0


# --------------------------------------------------------------------------
# merge-jev
# --------------------------------------------------------------------------


def cmd_merge_jev(args: argparse.Namespace) -> int:
    """jev-bench.jsonl (task v0.8b's B8 builder, fresh states distinct from B8) + jev-many.jsonl → jev-set.jsonl, with the independence
    proofs: no Jev request shares a state with B8 or rel07's check, no Jev item is an M8g or M1g item, and no record is sealed."""
    from robo_jev.bench import gates_v08 as G
    from robo_jev.data.sealed import sealed_record_set
    from robo_jev.decisions.records import read_records

    bench = read_records(args.bench)
    many = read_records(args.many)
    rows = bench + many
    ids = [r["id"] for r in rows]
    if len(set(ids)) != len(ids):
        raise SystemExit("duplicate ids in the Jev set")
    sealed = [r["id"] for r in rows if sealed_record_set(r)]
    if sealed:
        raise SystemExit(f"sealed records in the Jev set: {sealed[:3]}")
    b8 = read_records(G.B8_FILE) if G.B8_FILE.is_file() else []
    rel07 = read_records(ARTIFACTS / "scratch" / "rel07" / "bench-check" / "requests.jsonl")
    states = {r["request"]["state"] for r in rows}
    shared_b8 = states & {r["request"]["state"] for r in b8}
    shared_rel07 = states & {r["request"]["state"] for r in rel07}
    m8g_items = {str(i["item"]) for i in read_records(G.M8G_DIR / "m8g-items.jsonl")}
    m1g_items = {str(i["item"]) for i in read_records(M1G_ITEMS)}
    many_items = {r["meta"]["item"] for r in many}
    problems = {"states_shared_with_B8": len(shared_b8), "states_shared_with_rel07": len(shared_rel07),
                "items_shared_with_M8g": len(many_items & m8g_items), "items_shared_with_M1g": len(many_items & m1g_items)}
    if any(problems.values()):
        raise SystemExit(f"the Jev set is not independent: {problems}")
    path = refuse_write(args.out)
    sha = _write_jsonl(path, rows)
    _write_json(path.with_suffix(".manifest.json"), {
        "kind": "v08-jev-set", "script_version": SCRIPT_VERSION, "file": _rel(path), "sha256": sha, "records": len(rows),
        "parts": dict(Counter(r["meta"]["part"] for r in rows)), "bench_by_env": dict(Counter(r["meta"].get("env") for r in bench)),
        "many_by_n": dict(Counter(str(r["meta"]["n"]) for r in many)), "independence": problems,
        "b8_file_present": bool(b8), "not_sealed": True, "created": _now()})
    _log(f"Jev set: {len(rows)} records ({len(bench)} bench, {len(many)} many) → {path} ({sha[:12]}); independence {problems}")
    return 0


# --------------------------------------------------------------------------
# build-r14 — robo-jev-web's collected bench requests (reported; serve's conditions)
# --------------------------------------------------------------------------


def cmd_build_r14(args: argparse.Namespace) -> int:
    """R14 (:mod:`robo_jev.bench.r14_v08`): every deduplicated request as a sealed record (``R14-all.jsonl``), serve's sample
    (``R14-serve-sample.jsonl``, the pipeline check) and our evenly spaced sample (``R14-v08-sample.jsonl``), the tags and planner
    recommendations of both samples' model questions (``tags.json``), and ``frozen-before-scoring.json`` (sha256 of every one of them),
    written before any scoring; refuses to overwrite frozen files unless ``--force-before-scoring``."""
    import hashlib

    from robo_jev.bench import r14_v08 as R

    out = refuse_write(args.out_dir)
    frozen_path = out / "frozen-before-scoring.json"
    if frozen_path.is_file() and not args.force_before_scoring:
        raise SystemExit(f"{frozen_path} exists — R14 is frozen; pass --force-before-scoring only while no model has been scored on it")
    zip_path = R.R14_SOURCE.parent / f"{R.R14_SOURCE.name}.zip"
    zip_sha = hashlib.sha256(zip_path.read_bytes()).hexdigest() if zip_path.is_file() else None
    if zip_sha is not None and not zip_sha.startswith(R.R14_ZIP_SHA256_PREFIX):
        raise SystemExit(f"{zip_path}: sha256 {zip_sha[:16]} is not the relayed {R.R14_ZIP_SHA256_PREFIX}")
    all_records, serve_ids, v08_ids, counts = [], [], [], {}
    for env in R.R14_ENVS:
        raw = R.read_env(env)
        deduped = R.dedup(raw)
        first_line = {}
        for number, line in enumerate(raw, 1):
            first_line.setdefault(json.dumps(line["request"], sort_keys=True), number)
        records = [R.to_record(env, i, line, source_line=first_line[json.dumps(line["request"], sort_keys=True)]) for i, line in enumerate(deduped)]
        all_records += records
        serve_ids += [records[i]["id"] for i in R.serve_sample(records)]
        v08_ids += [records[i]["id"] for i in R.even_sample(records, args.per_env)]
        counts[env] = {"raw": len(raw), "dedup": len(deduped), "serve_sample": len(R.serve_sample(records)),
                       "v08_sample": len(R.even_sample(records, args.per_env)), "statuses": dict(Counter(str(l.get("status")) for l in deduped))}
    by_id = {r["id"]: r for r in all_records}
    files = {"R14-all.jsonl": all_records, "R14-serve-sample.jsonl": [by_id[i] for i in serve_ids], "R14-v08-sample.jsonl": [by_id[i] for i in v08_ids]}
    hashes = {name: _write_jsonl(out / name, rows) for name, rows in files.items()}
    scored = [by_id[i] for i in dict.fromkeys(serve_ids + v08_ids)]
    tags = R.tags_payload(scored, files={k: v for k, v in hashes.items()}, created=_now())
    _write_json(out / "tags.json", tags)
    hashes["tags.json"] = hashlib.sha256((out / "tags.json").read_bytes()).hexdigest()
    tokens = {}
    from robo_jev.hf.robo_jev_decide import build_prompts

    tok = _tokenizer()
    for name in ("R14-serve-sample.jsonl", "R14-v08-sample.jsonl"):
        per_env: dict[str, int] = {}
        for record in files[name]:
            asked = R.model_questions(record["request"]["questions"])["asked"]
            if not asked:
                continue
            n = sum(len(p["tokens"]) for p in build_prompts(record["request"]["state"], asked, tok))
            per_env[record["meta"]["env"]] = per_env.get(record["meta"]["env"], 0) + n * len(R.ORDERS)
        tokens[name] = {"by_env": per_env, "total": sum(per_env.values())}
    questions = Counter(f"{e['env']}/{q}" for e in tags["requests"] for q in e["questions"])
    planners = Counter(f"{e['env']}/{q}:{v['planner']}" for e in tags["requests"] for q, v in e["questions"].items() if v["planner"])
    manifest = {"kind": "v08-r14", "script_version": SCRIPT_VERSION, "git": _git(), "set": R.R14_SET, "prefix": R.R14_PREFIX,
                "source": {"dir": _rel(R.R14_SOURCE), "zip_sha256": zip_sha, "captured": "2026-10-14, robo-jev-web codex/robojev-direct e5a3b3a, "
                           "headless Chromium (the folder's README)"},
                "serve": {"repo": "STEERIX-home/robo-jev-serve", "commit": R.SERVE_COMMIT,
                          "replicated": ["dedup on json.dumps(request, sort_keys=True) (HANDOFF.md appendix B)", "trivial.split: 1-option choices "
                                         "answered with p = 1, never sent to the model", "scores.to_choice: a score with >= 2 criteria asked as a choice "
                                         "with keys '0'..'n-1' and the criteria sentences", "engine.decide -> the bundle's build_prompts / decide with their "
                                         "defaults (single order, seed 1, key = sha256 of the state as received)", "orders: 0, 1, 2 trailing spaces on "
                                         "the state (appendix B; the prompt uses state.strip())", "serve's sample: lines[::max(1, len // 20)][:20]"]},
                "counts": counts, "files": hashes, "per_env": args.per_env, "sample_rule": "our sample: floor(i * n / K), i < K, K = min(per_env, n)",
                "questions_scored": dict(sorted(questions.items())), "planner_recommendations": dict(sorted(planners.items())),
                "tag_counts": tags["counts"], "prompt_tokens_three_orders": tokens,
                "gpu_estimate_s_per_model": {name: gpu_seconds(t["total"]) for name, t in tokens.items()},
                "sealed": "for training (never trained on); NOT never-sent - the bench has already sent these requests", "created": _now()}
    _write_json(out / "r14.manifest.json", manifest)
    _write_json(frozen_path, {"files": hashes, "frozen": _now(), "note": "written before any scoring: the R14 scorer and the verdict refuse if "
                                                                          "any of these files changed"})
    _seal([out / name for name in (*files, "tags.json", "frozen-before-scoring.json")])
    _log(f"R14: {json.dumps(counts)} · tokens (3 orders) {json.dumps({k: v['total'] for k, v in tokens.items()})} · tags {tags['counts']}")
    return 0


# --------------------------------------------------------------------------
# recheck-v08 and check
# --------------------------------------------------------------------------


def cmd_recheck(args: argparse.Namespace) -> int:
    """The scan against train-v08.jsonl alone (module docstring) → overlap-v08.json."""
    from robo_jev.bench import gates_v08 as G
    from robo_jev.bench.iface import shingles
    from robo_jev.data.general_sets import normalize
    from robo_jev.decisions.records import file_sha256, read_records

    train = args.train_v08
    if not train.is_file():
        raise SystemExit(f"{train}: not there yet — run this once v0.8's training input exists")
    items = read_records(G.M8G_DIR / "m8g-items.jsonl")
    general = [{"item": i["item"], "state": i["state"], "stem": i["stem"], "correct": i["correct"], "own": list(i["own"]) + [str(d) for d in i["drawn"]]}
               for i in items if i["source"] == G.GENERAL_SOURCE]
    physics = [{"item": i["item"], "state": i["state"], "correct": tuple(i["correct"])} for i in items if i["source"] != G.GENERAL_SOURCE]
    units, grams = merged(V7B().candidates(general, {}), physics_candidates(physics))
    b8 = read_records(G.B8_FILE) if G.B8_FILE.is_file() else []
    for r in b8:  # B8: whole states (exclusions) and 8-word shingles of every state line (reported)
        units.setdefault(V7B()._hash(normalize(r["request"]["state"])), set()).add(f"b8:{r['id']}")
        for line in r["request"]["state"].splitlines():
            if len(line.split()) >= 8:
                for gram in shingles(line):
                    grams.setdefault(gram, set()).add(f"b8shingle:{r['id']}")
    from robo_jev.bench import r14_v08

    r14 = read_records(r14_v08.R14_ALL) if r14_v08.R14_ALL.is_file() else []
    for r in r14:  # R14: whole states (exclusions; sealed for training)
        units.setdefault(V7B()._hash(normalize(r["request"]["state"])), set()).add(f"r14:{r['id']}")
    found = V7B().scan([train], units, grams, set())
    hits = found["hits"]
    exclude_items = sorted(k for k in hits if k.startswith(("mmlu:", "phys:")) and (not k.startswith("phys:") or hits[k]))
    exclude_b8 = sorted(k[len("b8:"):] for k in hits if k.startswith("b8:") and any(h.startswith("unit:") for h in hits[k]))
    payload = {"kind": "v08-gates-overlap-v08", "script_version": SCRIPT_VERSION, "git": _git(),
               "train_v08": {"path": _rel(train), "sha256": file_sha256(train)}, "files": found["files"],
               "excluded_items": exclude_items, "excluded_b8": exclude_b8,
               "excluded_r14": sorted(k[len("r14:"):] for k in hits if k.startswith("r14:")), "r14_records_checked": len(r14),
               "b8_shingle_hits": sorted(k[len("b8shingle:"):] for k in hits if k.startswith("b8shingle:")),
               "hits": {k: v for k, v in hits.items() if not k.startswith("b8shingle:")},
               "rule": "pre-declared (task v0.8b, before any v0.8 data): an M8g MMLU item with any unit or shingle hit, an M8g physics item with a "
                       "whole-state or state / correct-option shingle hit, and a B8 or R14 request whose whole state is in train-v08 are left out "
                       "of every reading; B8 state-line shingle hits are reported",
               "created": _now()}
    _write_json(args.out_dir / "overlap-v08.json", payload)
    _log(f"train-v08: exclude {len(exclude_items)} M8g items, {len(exclude_b8)} B8 requests, {len(payload['excluded_r14'])} R14 requests; "
         f"B8 shingle hits {len(payload['b8_shingle_hits'])}")
    return 0


def interface_problems(record: dict[str, Any], info: dict[str, Any]) -> list[str]:
    """The many-option scoring path the chain uses (the package's prompt builder, as :func:`robo_jev.decisions.zero_shot.diagnose_records`
    calls it, record order) on one 255-option gate record: it must build the prompt, show the runner's ``CODES[:255]`` in order, and pass
    :func:`robo_jev.decisions.prompt.check_displayed_lines`. Whether the prompt is token-identical to the runner's is recorded in `info`."""
    from robo_jev.bench import gates_v08 as G
    from robo_jev.decisions.prompt import check_displayed_lines, record_prompts
    from robo_jev.hf.robo_jev_decide import build_prompts as runner_prompts

    try:
        prompts = record_prompts(record, _tokenizer(), seed=None, fmt="chat", mode="direct", orders="seeded")
    except (ValueError, KeyError) as error:
        return [f"the package cannot build a {len(G.CODES_255)}-option prompt yet ({error}) — task v0.8a's change must be on main"]
    out = []
    prompt = prompts[0]
    if tuple(prompt["codes"]) != G.CODES_255[: len(prompt["options"])]:
        out.append("the package's codes differ from the runner's CODES[:255] in order")
    out += check_displayed_lines(prompt)
    runner = runner_prompts(record["request"]["state"], record["request"]["questions"], _tokenizer(), seed=None)[0]
    info["package_prompt_identical_to_runner"] = runner["tokens"] == prompt["tokens"]
    info["package_codes_last"] = list(prompt["codes"][-3:])
    return out


def gate_check(runs: list[Path], out_dir: Path) -> tuple[list[str], dict[str, Any]]:
    """The chain's gate (module docstring): problems (empty = open) and what was read."""
    from robo_jev.bench import b8_v08
    from robo_jev.bench import gates_v08 as G
    from robo_jev.decisions.records import file_sha256, option_range_for, read_records

    problems: list[str] = []
    info: dict[str, Any] = {}
    manifest_path = out_dir / "m8g" / "m8g.manifest.json"
    if not manifest_path.is_file():
        return [f"{_rel(manifest_path)} is missing"], info
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for entry in (manifest["file"], manifest["items_file"]):
        path = REPO / entry["path"]
        if not path.is_file() or file_sha256(path) != entry["sha256"]:
            problems.append(f"{entry['path']}: missing or not the sealed file (sha256 {entry['sha256'][:12]})")
    problems += [f"B8: {p}" for p in b8_v08.check_frozen(out_dir / "b8")]
    from robo_jev.bench import r14_v08

    problems += [f"R14: {p}" for p in r14_v08.check_frozen(r14_v08.R14_DIR)]
    if not G.OPTION_RANGE_VERSION_CONFIRMED:
        problems.append(f"the opt-in's version name {G.OPTION_RANGE_VERSION!r} is a placeholder — set task v0.8a's name in gates_v08 and rebuild")
    if manifest.get("option_range") != G.option_range_mark():
        problems.append(f"M8g was built with {manifest.get('option_range')}, the code says {G.option_range_mark()} — rebuild")
    m8g_path = REPO / manifest["file"]["path"]
    if m8g_path.is_file():
        first = next(r for r in read_records(m8g_path) if r["meta"]["n"] == max(G.M8G_LEVELS))
        allowed, why = option_range_for(first)
        info["option_range_for"] = {"range": list(allowed), "problems": why}
        if tuple(allowed) != (2, G.OPTION_RANGE_MAX) or why:
            problems.append(f"main does not accept the gate's opt-in: option_range_for → {allowed}, {why} (task v0.8a's change must be on main)")
        problems += [f"interface: {p}" for p in interface_problems(first, info)]
    else:
        problems.append(f"{_rel(m8g_path)} is missing")
    recheck = out_dir / "overlap-v08.json"
    trained: dict[str, str | None] = {}
    if not recheck.is_file():
        problems.append(f"{_rel(recheck)} is missing — run recheck-v08 on v0.8's training input")
    else:
        sha = json.loads(recheck.read_text(encoding="utf-8"))["train_v08"]["sha256"]
        on_disk = ARTIFACTS / TRAIN_V08
        if on_disk.is_file() and file_sha256(on_disk) != sha:
            problems.append(f"{TRAIN_V08} changed since the recheck ({sha[:12]}) — run recheck-v08 again")
        if not runs:
            problems.append("no runs given — the check reads the input each v0.8 run trained on")
        for run in runs:
            metrics = Path(run) / "metrics.json"
            got = None
            if metrics.is_file():
                got = ((json.loads(metrics.read_text(encoding="utf-8")).get("manifest") or {}).get("data") or {}).get("file_sha256")
            trained[str(run)] = got
            if got != sha:
                problems.append(f"{run}: trained on {str(got)[:12]}, the recheck read {sha[:12]} — run recheck-v08 on the input the run used")
    info["trained_inputs"] = trained
    return problems, info


def cmd_check(args: argparse.Namespace) -> int:
    problems, info = gate_check([Path(r) for r in args.runs], args.out_dir)
    print(json.dumps({"problems": problems, **info}, indent=1, default=str))
    return 2 if problems else 0


def main(argv: list[str] | None = None) -> int:
    from robo_jev.bench import gates_v08 as G

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    b = sub.add_parser("build-m8g")
    b.add_argument("--out", type=Path, default=G.GATES)
    b.add_argument("--jev-out", type=Path, default=G.JEV_DIR / "jev-many.jsonl")
    b.add_argument("--seed", type=int, default=SEED)
    b.add_argument("--mmlu-items", type=int, default=M8G_MMLU)
    b.add_argument("--skip-overlap", action="store_true", help="tests only")
    p = sub.add_parser("build-probes")
    p.add_argument("--out", type=Path, default=G.PROBES_DIR / "bins.jsonl")
    j = sub.add_parser("merge-jev")
    j.add_argument("--bench", type=Path, default=G.JEV_DIR / "jev-bench.jsonl")
    j.add_argument("--many", type=Path, default=G.JEV_DIR / "jev-many.jsonl")
    j.add_argument("--out", type=Path, default=G.JEV_DIR / "jev-set.jsonl")
    r = sub.add_parser("recheck-v08")
    r.add_argument("--train-v08", type=Path, default=ARTIFACTS / TRAIN_V08)
    r.add_argument("--out-dir", type=Path, default=G.GATES)
    r14 = sub.add_parser("build-r14", help="R14: robo-jev-web's collected bench requests (reported; serve's conditions)")
    r14.add_argument("--out-dir", type=Path, default=G.ARTIFACTS / "v08" / "r14")
    r14.add_argument("--per-env", type=int, default=40)
    r14.add_argument("--force-before-scoring", action="store_true")
    c = sub.add_parser("check")
    c.add_argument("--runs", nargs="*", default=[], help="the fetched v0.8 run directories (each with metrics.json)")
    c.add_argument("--out-dir", type=Path, default=G.GATES)
    args = parser.parse_args(argv)
    return {"build-m8g": cmd_build_m8g, "build-probes": cmd_build_probes, "merge-jev": cmd_merge_jev, "recheck-v08": cmd_recheck,
            "build-r14": cmd_build_r14, "check": cmd_check}[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
