"""Author A1 — checks that need no MuJoCo: the trainer's loader and format checks, the scorers on synthetic predictions, the
v0.5 call view, prompt lengths, and the overlap with the protected sets (read in code only; only counts and fragments of A1's
own text are ever reported).
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import random
import re
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

REPO = Path(__file__).resolve().parents[4]
#: The comparison scorer keys its event and plan-option tables by family; A1's families are registered from the records' own
#: ``meta.key_events`` / ``meta.plan_option`` (in this process only — :mod:`robo_jev.bench.infamily` is not edited).
A1_FAMILIES = ("mastpick", "dockedge", "tugramp")


# --------------------------------------------------------------------------
# Loader and format
# --------------------------------------------------------------------------


def loader_check(path: Path) -> dict[str, Any]:
    from robo_jev.decisions.records import pair_groups, read_records, select_records, validate_record

    records = read_records(path)
    problems = Counter()
    for r in records:
        for p in validate_record(r):
            problems[p.split(":")[0]] += 1
    groups = pair_groups(records)
    sample = select_records(records, 300, seed=0)
    ids = [r["id"] for r in records]
    return {"records": len(records), "format_problems": sum(problems.values()), "problem_kinds": dict(problems),
            "pair_groups": len(groups), "group_sizes": dict(Counter(len(g) for g in groups)), "unique_ids": len(set(ids)) == len(ids),
            "select_records_300": len(sample)}


def trainer_examples_check(records: Sequence[Mapping[str, Any]], tokenizer: Any, n: int = 60) -> dict[str, Any]:
    """The trainer's example builder (same prompt function, chat format) on a sample: every question builds, targets sum to 1."""
    from robo_jev.decisions.examples import build_examples

    rng = random.Random(0)
    built = bad = 0
    for record in rng.sample(list(records), min(n, len(records))):
        examples = build_examples(dict(record), list(record["request"]["questions"]), tokenizer, order_seed=rng.randrange(1 << 30))
        for e in examples:
            built += 1
            if abs(sum(e["target"]) - 1.0) > 1e-6:
                bad += 1
    return {"examples": built, "bad_targets": bad}


# --------------------------------------------------------------------------
# Scorers on synthetic predictions
# --------------------------------------------------------------------------


def synthetic_predictions(records: Sequence[Mapping[str, Any]], *, noise: float, seed: int) -> list[dict[str, Any]]:
    """Predictions in the evaluator's format: two listings per question, the label plus noise."""
    from robo_jev.bench.system1 import oracle_predictions

    rng = random.Random(seed)
    out = oracle_predictions(records)
    for pred in out:
        for q in pred["questions"].values():
            listings = []
            for _ in range(2):
                ans = json.loads(json.dumps(q["answer"]))
                if "p_yes" in ans:
                    ans["p_yes"] = min(1.0, max(0.0, ans["p_yes"] + rng.gauss(0, noise)))
                elif "value" in ans:
                    ans["value"] = min(1.0, max(0.0, ans["value"] + rng.gauss(0, noise)))
                else:
                    probs = {k: max(1e-6, v + abs(rng.gauss(0, noise))) for k, v in ans["probabilities"].items()}
                    total = sum(probs.values())
                    ans["probabilities"] = {k: v / total for k, v in probs.items()}
                listings.append({"options": q["listings"][0]["options"], "values": None, "answer": ans})
            q["listings"], q["n_listings"] = listings, 2
    return out


def register_families(records: Sequence[Mapping[str, Any]]) -> None:
    from robo_jev.bench import infamily

    for r in records:
        meta = r["meta"]
        infamily.EVENT.setdefault(meta["scene"], tuple(meta["key_events"]))
        infamily.PLAN_KEY.setdefault(meta["scene"], meta["plan_option"])


def _verdict_v05():
    name = "a1_verdict_v05"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, REPO / "scripts" / "verdict_v05.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def scorer_check(records: Sequence[Mapping[str, Any]], *, n: int = 400) -> dict[str, Any]:
    """infamily (robust keys, direction, comparison), system1 on the oracle, and the v0.5 call view on a pair-preserving sample."""
    from robo_jev.bench import infamily, system1
    from robo_jev.decisions.records import select_records

    sample = select_records([dict(r) for r in records], n, seed=1)
    register_families(sample)
    out: dict[str, Any] = {"sample_records": len(sample)}
    robust = infamily.robust_keys(sample)
    out["robust_keys"] = len(robust)
    out["ambiguous_keys"] = len(infamily.ambiguous_keys(sample))
    oracle = system1.oracle_predictions(sample)
    noisy = synthetic_predictions(sample, noise=0.15, seed=3)
    for tag, preds in (("oracle", oracle), ("noisy", noisy)):
        direction = infamily.direction_items("A1", sample, preds)
        comparison = infamily.comparison_items("A1", sample, preds)
        out[f"direction_items_{tag}"] = {"n": len(direction), "mean": round(system1.summarize(direction)["value"] or 0.0, 4)}
        out[f"comparison_items_{tag}"] = {"n": len(comparison), "mean": round(system1.summarize(comparison)["value"] or 0.0, 4)}
    items = system1.set_items("A1", sample, oracle)
    out["system1_oracle"] = {m: round(system1.summarize(v, stratified=m in system1.STRATIFIED)["value"], 4) for m, v in items.items()
                             if m in ("band_accuracy", "band_balanced", "severe_error", "direction", "safe_choice", "catastrophic_choice", "band_balanced_robust")}
    v5 = _verdict_v05()
    clusters = v5.V4().pair_clusters(sample)
    calls_oracle = v5.call_items("A1", sample, oracle, clusters=clusters)
    calls_noisy = v5.call_items("A1", sample, noisy, clusters=clusters)
    m_oracle = v5.call_metrics(calls_oracle, n_boot=200, seed=0)
    m_noisy = v5.call_metrics(calls_noisy, n_boot=200, seed=0)
    diff = v5.call_difference(calls_noisy, calls_oracle, n_boot=200, seed=0)
    out["call_view"] = {"items": len(calls_oracle), "classes": m_oracle.get("classes"),
                        "oracle": {k: m_oracle[k]["value"] for k in ("DC", "LU", "LC", "GAP", "CA")},
                        "noisy": {k: m_noisy[k]["value"] for k in ("DC", "LU", "LC", "GAP", "CA")},
                        "noisy_minus_oracle_DC": diff["DC"]["value"]}
    return out


# --------------------------------------------------------------------------
# Prompt lengths
# --------------------------------------------------------------------------


def prompt_lengths(records: Sequence[Mapping[str, Any]], tokenizer: Any) -> dict[str, Any]:
    """Tokens of every question prompt (chat format, the trainer's default: fixed instruction + state + one question), of the whole
    request as the trainer would format it (the same prefix + every question block), of the raw request text and of the state."""
    from robo_jev.decisions.prompt import record_prompts

    per_prompt, per_request, state_only, raw = [], [], [], []
    for record in records:
        prompts = record_prompts(dict(record), tokenizer, seed=1, fmt="chat", orders="seeded")
        prefix = prompts[0]["prefix_tokens"]
        per_prompt.extend(len(p["tokens"]) for p in prompts)
        per_request.append(prefix + sum(len(p["tokens"]) - prefix for p in prompts))
        state_only.append(len(tokenizer.encode(record["request"]["state"], add_special_tokens=False).ids))
        parts = [record["request"]["state"]]
        for q in record["request"]["questions"].values():
            parts.append(q["instructions"])
            crit = q.get("criteria")
            if isinstance(crit, Mapping):
                parts.extend(f"{k}: {v}" for k, v in crit.items())
            elif isinstance(crit, list):
                parts.extend(crit)
        raw.append(len(tokenizer.encode("\n".join(parts), add_special_tokens=False).ids))

    def summary(values: list[int]) -> dict[str, Any]:
        values = sorted(values)
        return {"n": len(values), "median": statistics.median(values), "p95": values[int(0.95 * (len(values) - 1))], "max": values[-1], "min": values[0]}

    return {"question_prompt": summary(per_prompt), "state_plus_all_question_blocks": summary(per_request),
            "state_plus_question_text_raw": summary(raw), "state_only": summary(state_only),
            "definitions": {"question_prompt": "one trainer prompt: chat template, fixed system text, state, one question with its option lines",
                            "state_plus_all_question_blocks": "the shared prefix plus every question block as the trainer formats it",
                            "state_plus_question_text_raw": "state, every instruction and every option / scale text, newline-joined (no system text, no chat tokens, no letter codes)",
                            "state_only": "the state block"}}


# --------------------------------------------------------------------------
# Overlap (protected sets read in code only)
# --------------------------------------------------------------------------

PROTECTED = {
    **{f"T{i}": REPO / "artifacts" / "simgen" / "v0b" / "tests" / f"T{i}.jsonl" for i in range(1, 8)},
    "G1": REPO / "artifacts" / "simgen" / "probes" / "G1.jsonl", "G2": REPO / "artifacts" / "simgen" / "probes" / "G2.jsonl",
    "T8a": REPO / "artifacts" / "t8" / "T8a.jsonl", "T8b": REPO / "artifacts" / "t8" / "T8b.jsonl", "T9": REPO / "artifacts" / "t9" / "T9.jsonl",
    "B1": REPO / "artifacts" / "b1" / "b1-infamily.jsonl", "Hset": REPO / "artifacts" / "simgen" / "v04" / "scene-holdout.jsonl",
    "train-v04": REPO / "artifacts" / "simgen" / "v04" / "train-v04.jsonl",
}
NGRAM_SETS = ("T8b", "T9", "Hset")
_WORD = re.compile(r"[a-z0-9]+(?:[.'][a-z0-9]+)*")
_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")


def _h(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _norm(text: str) -> str:
    return " ".join(str(text).lower().split())


def text_units(record: Mapping[str, Any]) -> list[str]:
    """The record's natural-language strings: the state, and every question's instructions and option / scale descriptions."""
    request = record.get("request") or {}
    units = [str(request.get("state") or "")]
    for q in (request.get("questions") or {}).values():
        units.append(str(q.get("instructions") or ""))
        crit = q.get("criteria")
        if isinstance(crit, Mapping):
            units.extend(str(v) for v in crit.values())
        elif isinstance(crit, list):
            units.extend(str(v) for v in crit)
    return [u for u in units if u.strip()]


def sentences(units: Iterable[str]) -> set[str]:
    out = set()
    for unit in units:
        for part in _SPLIT.split(unit):
            norm = _norm(part)
            if norm:
                out.add(norm)
    return out


def ngrams(units: Iterable[str], n: int = 5) -> set[tuple[str, ...]]:
    out = set()
    for unit in units:
        words = _WORD.findall(unit.lower())
        out.update(tuple(words[i : i + n]) for i in range(len(words) - n + 1))
    return out


def fingerprints(record: Mapping[str, Any]) -> dict[str, Any]:
    request = record.get("request") or {}
    instr = [str(q.get("instructions") or "") for q in (request.get("questions") or {}).values()]
    return {"id": str(record.get("id")), "request": _h(json.dumps(request, sort_keys=True, ensure_ascii=False)),
            "state": _h(str(request.get("state") or "")), "instructions": {_h(i) for i in instr if i}}


def _stream(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def overlap(records: Sequence[Mapping[str, Any]], *, sets: Mapping[str, Path] = PROTECTED) -> dict[str, Any]:
    """Counts of shared ids, request / state / instruction hashes and sentences with each protected set, and shared word 5-grams
    with T8b, T9 and the H-set. A1's own shared fragments are listed (at most 20 per set); the other sets' text is never kept or
    printed."""
    mine = [fingerprints(r) for r in records]
    my_ids = {m["id"] for m in mine}
    my_req = {m["request"] for m in mine}
    my_state = {m["state"] for m in mine}
    my_instr = set().union(*(m["instructions"] for m in mine))
    my_units = [u for r in records for u in text_units(r)]
    my_sent = sentences(my_units)
    my_grams = ngrams(my_units)
    out: dict[str, Any] = {"mine": {"records": len(records), "sentences": len(my_sent), "5grams": len(my_grams)}, "sets": {}}
    for name, path in sets.items():
        if not Path(path).exists():
            out["sets"][name] = {"missing": True}
            continue
        ids = req = state = instr = 0
        shared_sent: set[str] = set()
        shared_grams: set[tuple[str, ...]] = set()
        n = 0
        for record in _stream(Path(path)):
            n += 1
            fp = fingerprints(record)
            ids += fp["id"] in my_ids
            req += fp["request"] in my_req
            state += fp["state"] in my_state
            instr += len(fp["instructions"] & my_instr)
            units = text_units(record)
            shared_sent |= sentences(units) & my_sent
            if name in NGRAM_SETS:
                shared_grams |= ngrams(units) & my_grams
        entry = {"records": n, "shared_ids": ids, "shared_request_hashes": req, "shared_state_hashes": state,
                 "shared_instruction_hashes": instr, "shared_sentences": len(shared_sent), "shared_sentences_mine": sorted(shared_sent)[:20]}
        if name in NGRAM_SETS:
            entry["shared_5grams"] = len(shared_grams)
            entry["shared_5grams_mine"] = [" ".join(g) for g in sorted(shared_grams)[:20]]
        out["sets"][name] = entry
    return out
