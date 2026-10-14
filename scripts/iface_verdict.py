#!/usr/bin/env python
"""Task iface — apply the PRE-DECLARED INTERFACE MEASUREMENTS M1 / M2 (progress.md, written 2026-10-14 ~21:05 KST, before any M1/M2 data
or result existed) mechanically (CPU).

    artifacts/scratch/iface/cpu.sh verdict scripts/iface_verdict.py          # → artifacts/iface/verdict.{json,md}

**Models** (the rule's six): the untrained Qwen3.5-2B (``base``) and five trained LoRA checkpoints — v0.1-mix, v0.4-s0, v0.4-s1, v0.6-s0,
v0.6-s1 (:data:`MODELS`). Every prediction file is checked before it is read: the scoring conditions of its summary (prompt v0c-p3, chat,
direct, batch 1; M2 also seeded orders with shuffle seed 1 on the standard diagnosis path), the sha256 of the set file it scored, the sha256 of
the checkpoint applied (against the file on disk) and the prompt digest.

**M1** (:mod:`robo_jev.bench.iface`): per model the accuracy and the probability on the correct option at each N, the bucket accuracies and
the late-letter penalty at N ≥ 17, item-cluster bootstrap 95 % (2,000 draws, seed 0), paired with the base on the same draws. Reading per
trained model, then over them: FINE-TUNE BIAS if any trained model's Δpenalty (model − base) ≥ 0.10 with its interval excluding 0;
GENERALISES if every trained model has |acc(N ≥ 17) − acc(N = 8)| ≤ 0.05 and |Δpenalty| ≤ 0.05; otherwise MIXED
(:func:`robo_jev.bench.iface.m1_outcome`). Breakdowns by source (ARC-Easy, ARC-Challenge, B1, H-set) are reported, not deciding.

**M2**: the sampled B1 and H-set records, short state vs the wrapped state at 2k, 4k and 8k characters, both scored by this task under the
same conditions (``--orders seeded``: one listing per question — listing 0 of the counterbalanced runs; the fresh short run is checked against
listing 0 of the existing Spark predictions on the full sets: same option orders, largest probability difference, same argmax). The two conditions must show every question in the same option orders (checked
listing by listing). Metrics on the pooled B1 + H-set items, cluster = the record's generator group (a one-fact pair's two records together,
as task B1): direction (one-fact pairs, label move ≥ 0.10; :func:`robo_jev.bench.infamily.direction_items`), comparison (the in-record
comparison question, key-event shares ≥ 0.15 apart; the per-scene mapping of ``scripts/verdict_v06.py`` — infamily's own on B1's families,
inferred from the labels on the H-set's), call-view DC (``scripts/verdict_v05.py``: P(call safe | truth clear-risky or unclear)), balanced
band (stratified by the label's band) and choice accuracy (every choice question: the argmax is in the label's best set). Δ = wrapped − short,
paired cluster bootstrap 95 % (2,000 draws, seed 0). Reading over the trained models (:func:`robo_jev.bench.iface.m2_outcome`): LONG-STATE GAP
if any trained model at any length has direction or comparison Δ ≤ −0.05, or DC Δ ≥ +0.05, with the interval excluding 0; NO GAP if every
gate Δ (direction, comparison, DC) of every trained model at every length is within ±0.03; otherwise MIXED. The base is context.

Writes ``verdict.json`` and ``verdict.md``. Never reads T9 or A4.
"""

from __future__ import annotations

import argparse
import datetime as dt
import functools
import importlib.util
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

SCRIPT_VERSION = "iface-verdict-1"
ARTIFACTS = REPO / "artifacts"
IFACE = ARTIFACTS / "iface"
CKPT = ARTIFACTS / "cloud" / "runs"
#: (tag, label, checkpoint relative to artifacts/cloud/runs or None) — report order; the first is the untrained base.
MODELS = (
    ("base", "untrained Qwen3.5-2B", None),
    ("v01mix", "v0.1-mix", "v01-mix-01-fetched/runs/v01-mix-2b/checkpoint.pt"),
    ("v04s0", "v0.4 seed 0", "v04-s0-01-fetched/runs/v04-s0-2b/checkpoint.pt"),
    ("v04s1", "v0.4 seed 1", "v04-s1-01-fetched/runs/v04-s1-2b/checkpoint.pt"),
    ("v06s0", "v0.6 seed 0", "v06-s0-01-fetched/runs/v06-s0-2b/checkpoint.pt"),
    ("v06s1", "v0.6 seed 1", "v06-s1-01-fetched/runs/v06-s1-2b/checkpoint.pt"),
)
BASE = "base"
TRAINED = tuple(tag for tag, _, _ in MODELS if tag != BASE)
LABELS = {tag: label for tag, label, _ in MODELS}
SETS = {"B1": ARTIFACTS / "b1" / "b1-infamily.jsonl", "Hset": ARTIFACTS / "simgen" / "v04" / "scene-holdout.jsonl"}
#: The existing Spark predictions on the full B1 / H-set (counterbalanced orders, shuffle seed 1): their listing 0 is the seeded listing,
#: so the fresh short-state run must reproduce it — a reproducibility check (v0.1-mix has none on the H-set).
EXISTING = {
    ("base", "B1"): ARTIFACTS / "reports" / "b1" / "B1-base-2b", ("base", "Hset"): ARTIFACTS / "reports" / "v04" / "spark" / "Hset-base-2b",
    ("v01mix", "B1"): ARTIFACTS / "reports" / "b1" / "B1-v01mix-2b",
    ("v04s0", "B1"): ARTIFACTS / "reports" / "v04" / "spark" / "B1-v04s0-2b", ("v04s0", "Hset"): ARTIFACTS / "reports" / "v04" / "spark" / "Hset-v04s0-2b",
    ("v04s1", "B1"): ARTIFACTS / "reports" / "v04" / "spark" / "B1-v04s1-2b", ("v04s1", "Hset"): ARTIFACTS / "reports" / "v04" / "spark" / "Hset-v04s1-2b",
    ("v06s0", "B1"): ARTIFACTS / "reports" / "v06" / "spark" / "B1-v06s0-2b", ("v06s0", "Hset"): ARTIFACTS / "reports" / "v06" / "spark" / "Hset-v06s0-2b",
    ("v06s1", "B1"): ARTIFACTS / "reports" / "v06" / "spark" / "B1-v06s1-2b", ("v06s1", "Hset"): ARTIFACTS / "reports" / "v06" / "spark" / "Hset-v06s1-2b",
}
#: The diagnosis conditions every M2 file must carry (short and wrapped — both scored by this task), and M1's. M2 orders: seeded (one
#: listing per question, listing 0 of the counterbalanced runs), chosen for GPU time before any M2 data existed.
M2_CONDITIONS = {"mode": "direct", "format": "chat", "prompt_version": "v0c-p3", "orders": "seeded", "shuffle_seed": 1, "batch": 1}
EXISTING_CONDITIONS = {"mode": "direct", "format": "chat", "prompt_version": "v0c-p3", "orders": "counterbalanced", "shuffle_seed": 1, "batch": 1}
M1_CONDITIONS = {"mode": "direct", "format": "chat", "prompt_version": "v0c-p3", "batch": 1, "lift_option_cap": True, "limit_items": None}
LENGTH_NAMES = ("2k", "4k", "8k")
M2_METRICS = ("direction", "comparison", "DC", "band_balanced", "choice_accuracy")
SEALED = (ARTIFACTS / "t9", ARTIFACTS / "v06" / "a4")


def _log(message: str) -> None:
    print(f"[iface-verdict {dt.datetime.now().strftime('%H:%M:%S')}] {message}", file=sys.stderr, flush=True)


def _git() -> str | None:
    try:
        return subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True).stdout.strip() or None
    except (OSError, subprocess.CalledProcessError):
        return None


def _refuse_sealed(path: Path) -> Path:
    resolved = Path(path).resolve()
    for place in SEALED:
        place = place.resolve()
        if resolved == place or place in resolved.parents:
            raise SystemExit(f"{path}: a sealed gate ({place}) — never opened by this task")
    return path


@functools.lru_cache(maxsize=None)
def _script(name: str):
    spec = importlib.util.spec_from_file_location(f"iface_{Path(name).stem}", REPO / "scripts" / name)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def V5():
    """scripts/verdict_v05.py — the call view (truth classes, 1/L call weights, call items, paired call differences)."""
    return _script("verdict_v05.py")


def V6():
    """scripts/verdict_v06.py — the per-scene comparison mapping and items."""
    return _script("verdict_v06.py")


def _read(path: Path) -> list[dict[str, Any]]:
    from robo_jev.decisions.records import read_records

    return read_records(_refuse_sealed(path))


@functools.lru_cache(maxsize=None)
def _sha(path: str) -> str:
    from robo_jev.decisions.records import file_sha256

    return file_sha256(path)


def checkpoint_sha(tag: str) -> str | None:
    rel = dict((t, c) for t, _, c in MODELS)[tag]
    return None if rel is None else _sha(str(CKPT / rel))


def check_config(stem: Path, *, conditions: Mapping[str, Any], input_path: Path, tag: str) -> dict[str, Any]:
    """The summary next to a prediction file: conditions, scored file (sha256), checkpoint (sha256 vs the file on disk; none for the base),
    prompt digest. Returns the identity and the problems found."""
    from robo_jev.decisions.prompt import PROMPT_VERSION, fixed_text_digest

    summary = stem.with_suffix(".json") if stem.suffix != ".json" else stem
    config = json.loads(summary.read_text(encoding="utf-8")).get("config") or {}
    problems = [f"{k} = {config.get(k)!r}, want {v!r}" for k, v in conditions.items() if config.get(k) != v]
    if config.get("input_sha256") != _sha(str(input_path)):
        problems.append(f"scored {str(config.get('input_sha256'))[:12]}, not {input_path.name} ({_sha(str(input_path))[:12]})")
    if config.get("prompt_digest") != fixed_text_digest(PROMPT_VERSION):
        problems.append("prompt digest differs")
    want = checkpoint_sha(tag)
    got = (config.get("checkpoint") or {}).get("sha256")
    if want != got:
        problems.append(f"checkpoint sha256 {str(got)[:12]} vs {str(want)[:12]} on disk")
    return {"summary": str(summary), "model": config.get("model"), "checkpoint_sha256": got, "input_sha256": config.get("input_sha256"),
            "git": config.get("git"), "created": config.get("created"), "tokenizer_sha256": (config.get("tokenizer") or {}).get("sha256"),
            "problems": problems}


# --------------------------------------------------------------------------
# M1
# --------------------------------------------------------------------------


def m1(args: argparse.Namespace) -> dict[str, Any]:
    from robo_jev.bench import iface

    records = _read(args.m1_set)
    rows, sources = {}, {}
    for tag, _, _ in MODELS:
        stem = args.m1_dir / f"M1-{tag}-2b"
        sources[tag] = check_config(stem.with_suffix(".json"), conditions=M1_CONDITIONS, input_path=args.m1_set, tag=tag)
        preds = _read(stem.with_suffix(".predictions.jsonl"))
        rows[tag] = iface.m1_rows(records, preds)
        sources[tag]["listings"] = len(rows[tag])
    problems = {t: s["problems"] for t, s in sources.items() if s["problems"]}
    if problems and not args.allow_mismatch:
        raise SystemExit(f"M1 prediction files fail their checks: {problems}")
    by_source = defaultdict(set)
    for record in records:
        by_source[record["meta"]["source"]].add(record["meta"]["item"])
    subsets = {**{f"source:{name}": members for name, members in sorted(by_source.items())},
               "general (ARC)": set().union(*(m for n, m in by_source.items() if n in iface.GENERAL_SOURCES)),
               "physics (B1 + H-set)": set().union(*(m for n, m in by_source.items() if n not in iface.GENERAL_SOURCES))}
    readings = iface.m1_readings(rows, base=BASE, n_boot=args.n_boot, seed=args.seed, subsets=subsets)
    outcome = iface.m1_outcome(readings, TRAINED)
    # descriptive: accuracy by exact position at N = 26 and N = 21 (pooled items)
    by_position: dict[str, dict[str, dict[str, float]]] = {}
    for tag, model_rows in rows.items():
        table: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
        for row in model_rows:
            if row["n"] >= iface.LATE_MIN_N:
                table[str(row["n"])][chr(ord("A") + row["position"])].append(row["correct"])
        by_position[tag] = {n: {letter: sum(v) / len(v) for letter, v in sorted(letters.items())} for n, letters in sorted(table.items())}
    manifest = json.loads(args.m1_set.with_name("m1-set.manifest.json").read_text(encoding="utf-8"))
    return {"readings": readings, "outcome": outcome, "sources": sources, "accuracy_by_position": by_position,
            "set": {"file": str(args.m1_set), "sha256": _sha(str(args.m1_set)), "records": len(records), "composition": manifest.get("composition"),
                    "overlap_files": manifest.get("overlap", {}).get("files"), "sources": manifest.get("sources"), "design": manifest.get("design")}}


# --------------------------------------------------------------------------
# M2
# --------------------------------------------------------------------------


def cluster_map(set_name: str, records: Sequence[Mapping[str, Any]]) -> dict[str, str]:
    """Record id → its generator group, prefixed with the set (a one-fact pair's two records share it)."""
    from robo_jev.bench.infamily import group_of

    return {str(r["id"]): f"{set_name}:{group_of(r)}" for r in records}


def _recluster(items: Mapping[tuple, tuple[str, float]], clusters: Mapping[str, str], rid_of) -> dict[tuple, tuple[str, float]]:
    return {key: (clusters[rid_of(key, cluster)], value) for key, (cluster, value) in items.items()}


def choice_items(set_name: str, records: Sequence[Mapping[str, Any]], predictions: Sequence[Mapping[str, Any]], clusters: Mapping[str, str]) -> dict:
    """Every labelled choice question: the weight of the listings (each 1/L) whose argmax is in the label's best set
    (:func:`robo_jev.decisions.metrics.choice_metrics` ``match``)."""
    from robo_jev.bench import system1
    from robo_jev.decisions.metrics import choice_metrics

    predicted = {str(p["id"]): p for p in predictions}
    out = {}
    for record in records:
        rid = str(record["id"])
        answers = (predicted.get(rid) or {}).get("questions") or {}
        for name, question in record["request"]["questions"].items():
            label = (record.get("labels") or {}).get(name)
            if question.get("type") != "choice" or not isinstance(label, Mapping) or name not in answers:
                continue
            value = sum(w for a, w in system1._answers(answers[name], "single_order") if choice_metrics(a["probabilities"], dict(label))["match"])
            out[(set_name, rid, name)] = (clusters[rid], float(value))
    return out


def m2_items(set_name: str, records: Sequence[Mapping[str, Any]], predictions: Sequence[Mapping[str, Any]], mapping: Mapping[str, Any]) -> dict[str, Any]:
    """The five M2 metrics' items of one set and condition (single order), clustered by generator group."""
    from robo_jev.bench import infamily, system1

    clusters = cluster_map(set_name, records)
    direction = infamily.direction_items(set_name, records, predictions)  # clusters = generator group (unprefixed)
    direction = {k: (f"{set_name}:{c}", v) for k, (c, v) in direction.items()}
    comparison = V6().comparison_items(set_name, records, predictions, mapping, clusters=clusters)
    band = system1.question_items(set_name, records, predictions).get("band_balanced") or {}
    band = {k: (clusters[k[2]], v) for k, (c, v) in band.items()}  # key = (label band, set, record id, question)
    calls = V5().call_items(set_name, records, predictions, clusters=clusters)
    return {"direction": direction, "comparison": comparison, "band_balanced": band, "choice_accuracy": choice_items(set_name, records, predictions, clusters),
            "calls": calls}


def _merge(parts: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = defaultdict(dict)
    for part in parts:
        for metric, items in part.items():
            out[metric].update(items)
    return dict(out)


def m2_delta(wrapped: Mapping[str, Any], short: Mapping[str, Any], *, n_boot: int, seed: int) -> dict[str, Any]:
    """Δ = wrapped − short for the five metrics (paired cluster bootstrap on the items both have)."""
    from robo_jev.bench import system1

    out: dict[str, Any] = {}
    for metric in ("direction", "comparison", "choice_accuracy"):
        out[metric] = system1.paired_difference(wrapped[metric], short[metric], n_boot=n_boot, seed=seed)
    out["band_balanced"] = system1.paired_difference(wrapped["band_balanced"], short["band_balanced"], n_boot=n_boot, seed=seed, stratified=True)
    calls = V5().call_difference(wrapped["calls"], short["calls"], n_boot=n_boot, seed=seed)
    out["DC"] = calls.get("DC")
    out["calls"] = {m: calls.get(m) for m in ("LU", "LC", "GAP", "CA", "safe|clear_risky", "safe|unclear")}
    return out


def pairing_problems(short: Sequence[Mapping[str, Any]], wrapped: Sequence[Mapping[str, Any]], ids: set[str]) -> list[str]:
    """Every question of every sampled record shown in the same option orders in both conditions (listing by listing)."""
    a = {str(p["id"]): p for p in short if str(p["id"]) in ids}
    b = {str(p["id"]): p for p in wrapped if str(p["id"]) in ids}
    problems = [f"missing in short: {sorted(ids - set(a))[:3]}"] if ids - set(a) else []
    problems += [f"missing in wrapped: {sorted(ids - set(b))[:3]}"] if ids - set(b) else []
    for rid in sorted(set(a) & set(b)):
        qa, qb = a[rid]["questions"], b[rid]["questions"]
        if set(qa) != set(qb):
            problems.append(f"{rid}: questions differ")
            continue
        for name in qa:
            la = [l.get("options") for l in qa[name]["listings"]]
            lb = [l.get("options") for l in qb[name]["listings"]]
            if la != lb:
                problems.append(f"{rid}/{name}: option orders differ")
        if len(problems) > 20:
            break
    return problems


def reproduction(existing: Sequence[Mapping[str, Any]], fresh: Sequence[Mapping[str, Any]], ids: set[str]) -> dict[str, Any]:
    """The fresh short-state run (seeded) against listing 0 of an existing counterbalanced run on the same records: same option orders,
    the largest code-probability difference, and the share of questions with the same argmax."""
    a = {str(p["id"]): p for p in existing if str(p["id"]) in ids}
    b = {str(p["id"]): p for p in fresh if str(p["id"]) in ids}
    diffs, same, orders, questions = [], 0, 0, 0
    for rid in sorted(set(a) & set(b)):
        for name, question in b[rid]["questions"].items():
            old = (a[rid]["questions"].get(name) or {}).get("listings") or []
            new = question["listings"][0]
            if not old:
                continue
            questions += 1
            if old[0]["options"] != new["options"]:
                orders += 1
                continue
            diffs.append(max(abs(float(x) - float(y)) for x, y in zip(old[0]["code_probs"], new["code_probs"])))
            same += int(max(range(len(new["code_probs"])), key=lambda i: new["code_probs"][i]) == max(range(len(old[0]["code_probs"])), key=lambda i: old[0]["code_probs"][i]))
    return {"questions": questions, "order_mismatches": orders, "max_abs_prob_diff": max(diffs) if diffs else None,
            "mean_abs_prob_diff": (sum(diffs) / len(diffs)) if diffs else None, "same_argmax_share": (same / len(diffs)) if diffs else None}


def m2(args: argparse.Namespace) -> dict[str, Any]:
    from robo_jev.bench import iface

    manifest = json.loads((args.m2_dir / "m2-set.manifest.json").read_text(encoding="utf-8"))
    sources: dict[str, Any] = {}
    problems: list[str] = []
    data: dict[str, Any] = {}
    mappings = {}
    repro: dict[str, Any] = {}
    files = {cond: args.m2_dir / f"M2-{cond}.jsonl" for cond in ("short", *(f"w{length}" for length in LENGTH_NAMES))}
    rows = {cond: {str(r["id"]): r for r in _read(path)} for cond, path in files.items()}
    for set_name, path in SETS.items():
        full = _read(path)
        ids = set(manifest["ids"][set_name])
        records = [r for r in full if str(r["id"]) in ids]
        if len(records) != len(ids):
            raise SystemExit(f"M2 {set_name}: {len(ids) - len(records)} sampled ids are not in {path}")
        for r in records:  # the scored files: the short one is the original record, the wrapped ones differ only in the state
            if rows["short"].get(str(r["id"])) != r:
                raise SystemExit(f"M2 short: record {r['id']} differs from {path}")
            for length in LENGTH_NAMES:
                w = rows[f"w{length}"].get(str(r["id"]))
                if w is None or w["request"]["questions"] != r["request"]["questions"] or w.get("labels") != r.get("labels") or w.get("meta") != r.get("meta"):
                    raise SystemExit(f"M2 w{length}: record {r['id']} differs from the original in questions, labels or meta")
        mappings[set_name] = V6().comparison_map(records)
        data[set_name] = {"records": records, "ids": ids, "path": path}
    items: dict[str, dict[str, Any]] = {}
    for tag, _, _ in MODELS:
        items[tag] = {}
        per_condition: dict[str, list] = defaultdict(list)
        preds = {}
        for cond, path in files.items():
            stem = args.m2_dir / "pred" / f"M2-{cond}-{tag}-2b"
            src = check_config(stem.with_suffix(".json"), conditions=M2_CONDITIONS, input_path=path, tag=tag)
            sources[f"{tag} {cond}"] = src
            problems += [f"{tag} {cond}: {p}" for p in src["problems"]]
            preds[cond] = _read(Path(str(stem) + ".predictions.jsonl"))
        for set_name, entry in data.items():
            short = [p for p in preds["short"] if str(p["id"]) in entry["ids"]]
            if (tag, set_name) in EXISTING:
                old_stem = EXISTING[(tag, set_name)]
                old_src = check_config(old_stem.with_suffix(".json"), conditions=EXISTING_CONDITIONS, input_path=entry["path"], tag=tag)
                repro[f"{tag} {set_name}"] = {**reproduction(_read(Path(str(old_stem) + ".predictions.jsonl")), short, entry["ids"]),
                                              "existing": str(old_stem), "existing_problems": old_src["problems"]}
            per_condition["short"].append(m2_items(set_name, entry["records"], short, mappings[set_name]))
            for length in LENGTH_NAMES:
                wrapped = [p for p in preds[f"w{length}"] if str(p["id"]) in entry["ids"]]
                problems += [f"{tag} {set_name} w{length} pairing: {p}" for p in pairing_problems(short, wrapped, entry["ids"])]
                per_condition[length].append(m2_items(set_name, entry["records"], wrapped, mappings[set_name]))
        items[tag] = {cond: {"pooled": _merge(parts), "by_set": dict(zip(data, parts))} for cond, parts in per_condition.items()}
    if problems and not args.allow_mismatch:
        raise SystemExit("M2 prediction files fail their checks:\n  " + "\n  ".join(problems[:30]))
    deltas: dict[str, Any] = {}
    levels: dict[str, Any] = {}
    by_set: dict[str, Any] = {}
    for tag in items:
        deltas[tag], by_set[tag] = {}, {}
        for length in LENGTH_NAMES:
            deltas[tag][length] = m2_delta(items[tag][length]["pooled"], items[tag]["short"]["pooled"], n_boot=args.n_boot, seed=args.seed)
            by_set[tag][length] = {s: {m: d for m, d in m2_delta(items[tag][length]["by_set"][s], items[tag]["short"]["by_set"][s], n_boot=args.n_boot,
                                                                  seed=args.seed).items() if m in M2_METRICS}
                                   for s in data}
        levels[tag] = {cond: _levels(items[tag][cond]["pooled"], n_boot=args.n_boot, seed=args.seed) for cond in ("short", *LENGTH_NAMES)}
    outcome = iface.m2_outcome({tag: {length: {m: deltas[tag][length][m] for m in ("direction", "comparison", "DC")} for length in LENGTH_NAMES}
                                for tag in TRAINED})
    context = iface.m2_model_reading({length: {m: deltas[BASE][length][m] for m in ("direction", "comparison", "DC")} for length in LENGTH_NAMES})
    return {"deltas": deltas, "levels": levels, "by_set": by_set, "outcome": outcome, "base_context": context, "sources": sources, "problems": problems,
            "reproduction": repro,
            "comparison_mapping": mappings, "set": {"manifest": str(args.m2_dir / "m2-set.manifest.json"), "fraction": manifest.get("fraction"),
                                                    "records": {s: len(d["ids"]) for s, d in data.items()}, "files": manifest.get("files")}}


def _levels(pooled: Mapping[str, Any], *, n_boot: int, seed: int) -> dict[str, Any]:
    from robo_jev.bench import system1

    out = {m: system1.summarize(pooled[m], n_boot=n_boot, seed=seed, stratified=(m == "band_balanced")) for m in ("direction", "comparison", "band_balanced", "choice_accuracy")}
    out["DC"] = V5().call_metrics(pooled["calls"], n_boot=n_boot, seed=seed).get("DC")
    return out


# --------------------------------------------------------------------------
# Markdown
# --------------------------------------------------------------------------


def _f(x: Any, digits: int = 3, signed: bool = False) -> str:
    if x is None:
        return "—"
    return f"{x:+.{digits}f}" if signed else f"{x:.{digits}f}"


def _ci(d: Mapping[str, Any] | None, signed: bool = False) -> str:
    if not d or d.get("value") is None:
        return "—"
    return f"{_f(d['value'], signed=signed)} [{_f(d.get('low'), signed=signed)}, {_f(d.get('high'), signed=signed)}]"


def render(result: Mapping[str, Any]) -> str:
    from robo_jev.bench import iface

    lines = ["# Interface measurements M1 / M2 — pre-declared readings", "",
             f"Rule written {iface.RULE['written']}. Applied mechanically by `scripts/iface_verdict.py` ({SCRIPT_VERSION}, git {result['git']}); "
             f"{result['n_boot']} bootstrap draws, seed {result['seed']}.", ""]
    if result.get("m1"):
        m = result["m1"]
        o = m["outcome"]
        r = m["readings"]
        lines += [f"## M1 — many options: **{o['outcome']}** → {o['next']}", "",
                  f"{r['items']} items (every listing scored for every model), item-cluster bootstrap. Accuracy = argmax over the shown codes is the "
                  "correct option; p = probability on the correct option; buckets and the late-letter penalty at N ≥ 17 (N = 21 and 26).", "",
                  "| model | " + " | ".join(f"acc N={n}" for n in iface.N_LEVELS) + " | acc N≥17 | p N=8 | p N≥17 |", "|---|" + "---|" * (len(iface.N_LEVELS) + 3)]
        for tag, label, _ in MODELS:
            x = r["models"][tag]
            lines.append(f"| {label} | " + " | ".join(_ci(x.get(f"acc@{n}")) if n in (8,) else _f(x.get(f"acc@{n}", {}).get("value")) for n in iface.N_LEVELS)
                         + f" | {_ci(x.get('acc@late'))} | {_f(x.get('p@8', {}).get('value'))} | {_f(x.get('p@late', {}).get('value'))} |")
        lines += ["", "| model | acc A–H | acc I–P | acc Q–Z | late-letter penalty | Δpenalty vs base | acc N≥17 − acc N=8 | reading |", "|---|---|---|---|---|---|---|---|"]
        for tag, label, _ in MODELS:
            x = r["models"][tag]
            p = r["paired_vs_base"].get(tag, {})
            reading = o["per_model"].get(tag, {}).get("outcome", "(base)")
            lines.append(f"| {label} | {_ci(x.get('bucket:A-H'))} | {_ci(x.get('bucket:I-P'))} | {_ci(x.get('bucket:Q-Z'))} | {_ci(x.get('penalty'), True)} | "
                         f"{_ci(p.get('penalty'), True)} | {_ci(x.get('late_minus_8'), True)} | {reading} |")
        lines += ["", "Per-model checks (FINE-TUNE BIAS: Δpenalty ≥ 0.10 and interval excludes 0; GENERALISES: |acc N≥17 − acc N=8| ≤ 0.05 and |Δpenalty| ≤ 0.05):", ""]
        for tag, entry in o["per_model"].items():
            lines.append(f"- {LABELS[tag]}: {entry['outcome']} — " + ", ".join(f"{k} {v}" for k, v in entry["checks"].items()))
        lines += ["", f"Aggregation: {o['aggregation']}.", "", "### M1 by source (reported, not deciding)", "",
                  "| subset | items | model | acc N=4 | acc N=8 | acc N≥17 | A–H | I–P | Q–Z | penalty | Δpenalty vs base |", "|---|---|---|---|---|---|---|---|---|---|---|"]
        for name, entry in r["subsets"].items():
            for i, (tag, label, _) in enumerate(MODELS):
                x = entry["models"][tag]
                head = f"{name} | {entry['items']}" if i == 0 else " | "
                lines.append(f"| {head} | {label} | {_f(x['acc@4']['value'])} | {_f(x['acc@8']['value'])} | {_f(x['acc@late']['value'])} | {_f(x['bucket:A-H']['value'])} | "
                             f"{_f(x['bucket:I-P']['value'])} | {_f(x['bucket:Q-Z']['value'])} | {_ci(x['penalty'], True)} | {_ci(x.get('penalty_minus_base'), True)} |")
        lines += ["", "Code mass (the shown codes' share of the full-vocabulary probability) by N:", "", "| model | " + " | ".join(f"N={n}" for n in iface.N_LEVELS) + " |",
                  "|---|" + "---|" * len(iface.N_LEVELS)]
        for tag, label, _ in MODELS:
            x = r["models"][tag]
            lines.append(f"| {label} | " + " | ".join(_f(x.get(f"mass@{n}", {}).get("value")) for n in iface.N_LEVELS) + " |")
        lines += ["", "Accuracy by the correct option's letter at N = 26 (pooled items; each letter ≈ the same number of listings):", "",
                  "| model | " + " | ".join(chr(65 + i) for i in range(26)) + " |", "|---|" + "---|" * 26]
        for tag, label, _ in MODELS:
            row = m["accuracy_by_position"][tag].get("26", {})
            lines.append(f"| {label} | " + " | ".join(_f(row.get(chr(65 + i)), 2) for i in range(26)) + " |")
        lines.append("")
    if result.get("m2"):
        m = result["m2"]
        o = m["outcome"]
        lines += [f"## M2 — bench-shaped long states: **{o['outcome']}** → {o['next']}", "",
                  "Δ = wrapped − short on the sampled B1 + H-set records (paired cluster bootstrap 95 %, cluster = generator group). Gate: direction / "
                  "comparison Δ ≤ −0.05 or DC Δ ≥ +0.05 with the interval excluding 0 (harm); NO GAP needs every gate Δ within ±0.03.", "",
                  "| model | length | Δ direction | Δ comparison | Δ DC | Δ balanced band | Δ choice accuracy | harm |", "|---|---|---|---|---|---|---|---|"]
        for tag, label, _ in MODELS:
            reading = o["per_model"].get(tag) or m["base_context"]
            for length in LENGTH_NAMES:
                d = m["deltas"][tag][length]
                harm = [t.split(":")[1] for t in reading["triggers"] if t.startswith(length + ":")]
                lines.append(f"| {label}{' (context)' if tag == BASE else ''} | {length} | {_ci(d['direction'], True)} | {_ci(d['comparison'], True)} | {_ci(d['DC'], True)} | "
                             f"{_ci(d['band_balanced'], True)} | {_ci(d['choice_accuracy'], True)} | {', '.join(harm) or '—'} |")
        lines += ["", "Levels (short → 2k → 4k → 8k):", "", "| model | metric | short | 2k | 4k | 8k |", "|---|---|---|---|---|---|"]
        for tag, label, _ in MODELS:
            for metric in M2_METRICS:
                cells = [_f((m["levels"][tag][c].get(metric) or {}).get("value")) for c in ("short", *LENGTH_NAMES)]
                lines.append(f"| {label} | {metric} | " + " | ".join(cells) + " |")
        lines += ["", "Call view parts behind DC (Δ wrapped − short; reported, not deciding): P(safe | clear-risky), P(safe | unclear), and the "
                  "look-again and clear-call readings of the v0.5 rule:", "",
                  "| model | length | Δ P(safe given clear-risky) | Δ P(safe given unclear) | Δ LU | Δ LC | Δ CA |", "|---|---|---|---|---|---|---|"]
        for tag, label, _ in MODELS:
            for length in LENGTH_NAMES:
                c = m["deltas"][tag][length].get("calls") or {}
                lines.append(f"| {label} | {length} | {_ci(c.get('safe|clear_risky'), True)} | {_ci(c.get('safe|unclear'), True)} | {_ci(c.get('LU'), True)} | "
                             f"{_ci(c.get('LC'), True)} | {_ci(c.get('CA'), True)} |")
        lines += ["", f"Aggregation: {o['aggregation']}.", "", "Per model: " + "; ".join(
            f"{LABELS[t]} triggers {r['triggers'] or 'none'}, all within ±0.03: {r['all_within_0.03']}" for t, r in o["per_model"].items()) + ".", "",
                  "### M2 by set (reported, not deciding)", "", "| model | length | set | Δ direction | Δ comparison | Δ DC | Δ balanced band | Δ choice |", "|---|---|---|---|---|---|---|---|"]
        for tag, label, _ in MODELS:
            for length in LENGTH_NAMES:
                for s, d in m["by_set"][tag][length].items():
                    lines.append(f"| {label} | {length} | {s} | {_ci(d['direction'], True)} | {_ci(d['comparison'], True)} | {_ci(d['DC'], True)} | {_ci(d['band_balanced'], True)} | "
                                 f"{_ci(d['choice_accuracy'], True)} |")
        lines += ["", "Reproduction of the existing Spark predictions (fresh short run vs their listing 0):", "",
                  "| model / set | questions | order mismatches | max |Δp| | mean |Δp| | same argmax |", "|---|---|---|---|---|---|"]
        for key, r in m.get("reproduction", {}).items():
            lines.append(f"| {key} | {r['questions']} | {r['order_mismatches']} | {_f(r['max_abs_prob_diff'], 4)} | {_f(r['mean_abs_prob_diff'], 5)} | {_f(r['same_argmax_share'])} |")
        lines.append("")
    lines += ["## Notes", ""] + [f"- {n}" for n in result["notes"]] + ["", "## Rule (verbatim parts)", ""] + [f"- **{k}** — {v}" for k, v in iface.RULE.items()]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--m1-set", type=Path, default=IFACE / "m1" / "m1-set.jsonl")
    parser.add_argument("--m1-dir", type=Path, default=IFACE / "m1" / "pred")
    parser.add_argument("--m2-dir", type=Path, default=IFACE / "m2")
    parser.add_argument("--only", choices=("m1", "m2"), default=None, help="read only one measurement (the other is left out of the output)")
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--allow-mismatch", action="store_true", help="read even if a check fails (the problems are written into the verdict)")
    parser.add_argument("--out", type=Path, default=IFACE / "verdict.json")
    args = parser.parse_args(argv)
    for path in (args.m1_set, args.m1_dir, args.m2_dir, args.out):
        _refuse_sealed(path)
    result: dict[str, Any] = {"kind": "iface-verdict", "script_version": SCRIPT_VERSION, "git": _git(), "n_boot": args.n_boot, "seed": args.seed,
                              "created": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
                              "models": [{"tag": t, "label": l, "checkpoint": c, "trained": t != BASE} for t, l, c in MODELS]}
    from robo_jev.bench import iface

    result["rule"] = iface.RULE
    if args.only in (None, "m1"):
        result["m1"] = m1(args)
        _log(f"M1: {result['m1']['outcome']['outcome']}")
    if args.only in (None, "m2"):
        result["m2"] = m2(args)
        _log(f"M2: {result['m2']['outcome']['outcome']}")
    result["notes"] = [
        "Aggregation over the five trained models: M1 FINE-TUNE BIAS if any trained model shows it, GENERALISES only if all do; M2 LONG-STATE GAP "
        "if any trained model triggers at any length, NO GAP only if every gate delta of every trained model is within +-0.03 (the rule's 'a trained "
        "model' / 'the trained models' read this way; every per-model reading is listed).",
        "'Within 0.05' / 'within +-0.03' are read on the point values; 'interval excluding 0' on the 95 % percentile interval.",
        "M2 short and wrapped states were both scored by this task with --orders seeded (one listing per question = listing 0 of the "
        "counterbalanced runs; chosen for GPU time before any M2 data: 0.14 s per prompt at 2k and 0.63 s at 8k for a LoRA model on Spark); the "
        "fresh short run is compared with listing 0 of the existing Spark predictions (reproduction table).",
        "M2 'choice accuracy' = every labelled choice question of the sampled records (direction, outcome, comparison, action kinds): the share of "
        "listings whose argmax is in the label's best set (choice_metrics 'match'); 'balanced band' over every yes/no and score question.",
    ]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=1, default=str) + "\n", encoding="utf-8")
    args.out.with_suffix(".md").write_text(render(result), encoding="utf-8")
    _log(f"→ {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
