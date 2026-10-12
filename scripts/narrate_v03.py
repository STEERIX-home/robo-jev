#!/usr/bin/env python
"""Task v0.3a — varied narration of training moments (LLM rewrite with fact preservation). No training, no GPU.

    # 1. sealed-set index (hashes only; reads T1–T7, G1, G2, T8a, T8b locally — nothing is sent)
    python scripts/narrate_v03.py index
    # 2. plan: stratified order over (scene, language), two voices per moment, sources checked against the index
    python scripts/narrate_v03.py plan
    # 3. (paid) model bake-off on the first plan moments, then the pilot (plan ranks 0–199), then the scale run
    python scripts/narrate_v03.py run --model M --start 0 --count 12 --tag bakeoff --max-total-usd 14.7
    python scripts/narrate_v03.py run --model M --start 0 --count 200 --tag pilot --max-total-usd 14.7
    python scripts/narrate_v03.py run --model M --start 200 --count 49800 --tag scale --max-total-usd 14.7
    # 4. checks over cached replies (free) → per-voice pass rates and examples
    python scripts/narrate_v03.py report --model M --start 0 --count 200 --out artifacts/simgen/v03/pilot-report.json
    # 5. (paid) back-check: a second LLM compares original and rewrite facts on a sample of passed rewrites
    python scripts/narrate_v03.py judge --model M --judge-model J --start 0 --count 200 --sample 400 --max-total-usd 14.7
    # 6. output records + manifest (free), and the request-log audit against the sealed hashes
    python scripts/narrate_v03.py assemble --model M --out artifacts/simgen/v03/narrated-train.jsonl
    python scripts/narrate_v03.py audit

The cap is cumulative over every paid command (one ledger, artifacts/simgen/v03/ledger.json); the hard cap of the task is $15,
the default stop is lower. The key is read from ~/.config/robo-jev/openrouter.env only and never printed.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import random
import sys
import threading
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

OUT = REPO / "artifacts" / "simgen" / "v03"
SOURCE = REPO / "artifacts" / "simgen" / "v02" / "train-50k.jsonl"
SOURCE_MANIFEST = REPO / "artifacts" / "simgen" / "v02" / "train-50k.manifest.json"
SEALED = {
    **{f"T{i}": REPO / "artifacts" / "simgen" / "v0b" / "tests" / f"T{i}.jsonl" for i in range(1, 8)},
    "G1": REPO / "artifacts" / "simgen" / "probes" / "G1.jsonl",
    "G2": REPO / "artifacts" / "simgen" / "probes" / "G2.jsonl",
    "T8a": REPO / "artifacts" / "t8" / "T8a.jsonl",
    "T8b": REPO / "artifacts" / "t8" / "T8b.jsonl",
}
INDEX = OUT / "sealed-index.json"
PLAN = OUT / "plan.jsonl"
LEDGER = OUT / "ledger.json"
CACHE = OUT / "cache"
CALLS = OUT / "requests.log.jsonl"
HARD_CAP_USD = 15.0
MODELS_URL = "https://openrouter.ai/api/v1/models"


def _log(message: str) -> None:
    print(f"[narrate {dt.datetime.now().strftime('%H:%M:%S')}] {message}", file=sys.stderr, flush=True)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(payload, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    temp.replace(path)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _index():
    from robo_jev.simgen.narration import SealedIndex

    if not INDEX.is_file():
        raise SystemExit(f"{INDEX} missing — run `index` first")
    return SealedIndex.from_json(json.loads(INDEX.read_text(encoding="utf-8")))


def _plan() -> list[dict[str, Any]]:
    if not PLAN.is_file():
        raise SystemExit(f"{PLAN} missing — run `plan` first")
    return [json.loads(line) for line in PLAN.read_text(encoding="utf-8").splitlines() if line.strip()]


def _sources(ids: set[str]) -> dict[str, dict[str, Any]]:
    out = {}
    with SOURCE.open(encoding="utf-8") as handle:
        for line in handle:
            # cheap id peek before parsing the whole line
            rid = line[line.find('"id":"') + 6 : line.find('"', line.find('"id":"') + 6)] if '"id":"' in line[:200] else None
            if rid is not None and rid not in ids:
                continue
            record = json.loads(line)
            if record["id"] in ids:
                out[record["id"]] = record
    return out


def _price(model: str) -> dict[str, float]:
    from robo_jev.launch.providers.runpod import urllib_transport

    status, _, body = urllib_transport("GET", MODELS_URL, {"User-Agent": "robo-jev-narrate/0.1", "Accept": "application/json"}, None, 60)
    if status != 200:
        raise SystemExit(f"model list GET returned {status}")
    entries = {m["id"]: m for m in json.loads(body.decode("utf-8"))["data"]}
    if model not in entries:
        raise SystemExit(f"{model}: not in the OpenRouter model list")
    pricing = entries[model].get("pricing") or {}
    return {"prompt": float(pricing.get("prompt") or 0), "completion": float(pricing.get("completion") or 0)}


# --------------------------------------------------------------------------
# index, plan
# --------------------------------------------------------------------------


def cmd_index(args: argparse.Namespace) -> int:
    from robo_jev.simgen.narration import SealedIndex

    index = SealedIndex.build(SEALED)
    _write_json(INDEX, {**index.to_json(), "created": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
                        "note": "hashes only: whole-state and whole-request sha256 of every sealed record; text-unit sha256 "
                                "prefixes (16 hex) of the sets written outside our engine (G1, G2, T8a, T8b)"})
    for name in SEALED:
        _log(f"{name}: {len(index.states[name])} states, {len(index.requests[name])} requests, {len(index.units[name])} units")
    return 0


def cmd_plan(args: argparse.Namespace) -> int:
    from robo_jev.simgen.narration import build_plan

    index = _index()
    rows, refused = [], []
    with SOURCE.open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            hits = index.hits(source=record, texts=[record["request"]["state"]])
            if hits:
                refused.append({"id": record["id"], "hits": hits})
                continue
            meta = record["meta"]
            rows.append({"id": record["id"], "scene": meta["scene"], "group": meta["group"], "language": meta["language"]})
    plan = build_plan(rows, seed=args.seed)
    PLAN.parent.mkdir(parents=True, exist_ok=True)
    with PLAN.open("w", encoding="utf-8") as handle:
        for entry in plan:
            handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
    summary = {"source": str(SOURCE.relative_to(REPO)), "source_sha256": _sha256_file(SOURCE), "moments": len(plan), "refused": refused,
               "seed": args.seed, "plan_sha256": _sha256_file(PLAN),
               "first_200": {"scene": Counter(e["scene"] for e in plan[:200]), "language": Counter(e["language"] for e in plan[:200]),
                             "voices": Counter(v for e in plan[:200] for v in e["voices"])}}
    _write_json(OUT / "plan.summary.json", summary)
    _log(f"plan: {len(plan)} moments ({len(refused)} sources refused by the sealed index) → {PLAN}")
    return 0


# --------------------------------------------------------------------------
# paid runs
# --------------------------------------------------------------------------


def _select(plan: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    entries = plan[args.start : args.start + args.count]
    if getattr(args, "only_language", None):
        entries = [e for e in entries if e["language"] == args.only_language]
    return entries


def _jobs(model: str, entries: list[dict[str, Any]], price: dict[str, float], *, voices: set[str] | None = None) -> list[Any]:
    from robo_jev.simgen.narration_run import make_job

    sources = _sources({e["source_id"] for e in entries})
    jobs = []
    for entry in entries:
        for voice in entry["voices"]:
            if voices and voice not in voices:
                continue
            jobs.append(make_job(entry, sources[entry["source_id"]], voice, model, price))
    return jobs


def _ledger(max_total: float):
    from robo_jev.bench.openrouter import Ledger

    if max_total > HARD_CAP_USD:
        raise SystemExit(f"--max-total-usd {max_total} is above the task's hard cap ${HARD_CAP_USD}")
    return Ledger(LEDGER, max_total_usd=max_total, max_model_usd=max_total)


def cmd_run(args: argparse.Namespace) -> int:
    from robo_jev.bench.openrouter import DecisionsClient, ResponseCache, load_api_key
    from robo_jev.simgen.narration import CHAT_URL, PROMPT_VERSION
    from robo_jev.simgen.narration_run import run_jobs

    plan = _plan()
    entries = _select(plan, args)
    price = _price(args.model)
    jobs = _jobs(args.model, entries, price)
    ledger = _ledger(args.max_total_usd)
    _log(f"{args.tag}: {len(entries)} moments, {len(jobs)} jobs, model {args.model} (price in {price['prompt'] * 1e6:.3f}/M, out "
         f"{price['completion'] * 1e6:.3f}/M), prompt {PROMPT_VERSION}; ledger so far ${ledger.spent():.4f}, stop at ${args.max_total_usd}")
    client = DecisionsClient(load_api_key(), url=CHAT_URL, timeout=args.timeout) if not args.cache_only else None
    started = dt.datetime.now().astimezone().isoformat(timespec="seconds")
    out = run_jobs(args.model, jobs, client=client, cache=ResponseCache(CACHE), ledger=ledger, price=price, index=_index(),
                   calls_log=CALLS, concurrency=args.concurrency, log=_log, stop=threading.Event())
    ledger.runs.append({"tag": args.tag, "model": args.model, "start": args.start, "count": args.count, "jobs": len(jobs),
                        "sent": out["stats"]["sent"], "cached": out["stats"]["cached"], "cost_usd": out["stats"]["cost_usd"],
                        "errors": len(out["errors"]), "budget": out["stats"]["budget"], "prompt_version": PROMPT_VERSION,
                        "started": started, "finished": dt.datetime.now().astimezone().isoformat(timespec="seconds")})
    ledger.save()
    if out["errors"]:
        for job_id, error in list(out["errors"].items())[:5]:
            _log(f"error {job_id}: {error[:200]}")
    return 0 if not out["stats"]["fatal"] else 2


def cmd_pipeline(args: argparse.Namespace) -> int:
    """Rewrite → checks → back-check per moment-voice, one pool, one ledger (both models count against the cap)."""
    from robo_jev.bench.openrouter import DecisionsClient, ResponseCache, load_api_key
    from robo_jev.simgen.narration import CHAT_URL, PROMPT_VERSION
    from robo_jev.simgen.narration_run import run_pipeline

    plan = _plan()
    entries = _select(plan, args)
    prices = {args.model: _price(args.model), args.judge_model: _price(args.judge_model)}
    jobs = _jobs(args.model, entries, prices[args.model])
    ledger = _ledger(args.max_total_usd)
    _log(f"{args.tag}: {len(entries)} moments, {len(jobs)} jobs; rewrite {args.model}, back-check {args.judge_model}; prompt "
         f"{PROMPT_VERSION}; ledger so far ${ledger.spent():.4f}, stop at ${args.max_total_usd}")
    client = DecisionsClient(load_api_key(), url=CHAT_URL, timeout=args.timeout) if not args.cache_only else None
    started = dt.datetime.now().astimezone().isoformat(timespec="seconds")
    out = run_pipeline(args.model, args.judge_model, jobs, client=client, cache=ResponseCache(CACHE), ledger=ledger, prices=prices,
                       index=_index(), calls_log=CALLS, concurrency=args.concurrency, log=_log, stop=threading.Event())
    stats = out["stats"]
    ledger.runs.append({"tag": args.tag, "rewrite_model": args.model, "judge_model": args.judge_model, "start": args.start,
                        "count": args.count, "jobs": len(jobs), "counts": stats["counts"],
                        "rewrite_cost_usd": stats["rewrite"]["cost_usd"], "judge_cost_usd": stats["judge"]["cost_usd"],
                        "errors": len(stats["errors"]), "budget": stats["budget"], "prompt_version": PROMPT_VERSION,
                        "started": started, "finished": dt.datetime.now().astimezone().isoformat(timespec="seconds")})
    ledger.save()
    for job_id, error in list(stats["errors"].items())[:5]:
        _log(f"error {job_id}: {error[:200]}")
    if stats["fatal"]:
        return 2
    return 4 if stats["budget"] else 0


# --------------------------------------------------------------------------
# checks over cached replies
# --------------------------------------------------------------------------


def _evaluate_range(model: str, entries: list[dict[str, Any]], judge_model: str | None = None) -> tuple[list[Any], dict[str, Any]]:
    """Cached replies of a plan range → (jobs, {"results": [(job, result, cache entry)]}); with a judge model, each passed result
    carries ``result["verdict"]`` (the cached back-check, or None when it was not run)."""
    from robo_jev.bench.openrouter import ResponseCache
    from robo_jev.simgen.narration_run import evaluate, judge_id, judge_job, parse_judge

    price = {"prompt": 0.0, "completion": 0.0}
    jobs = _jobs(model, entries, price)
    cache = ResponseCache(CACHE)
    results = []
    for job in jobs:
        hit = cache.get(model, job.job_id, job.sha)
        if hit is None:
            continue
        result = evaluate(job, hit["response"])
        if judge_model and result["passed"]:
            jjob = judge_job(judge_id(job.job_id), job.source["request"]["state"], result["state"], judge_model, price, job.voice.language,
                             entry=job.entry, source=job.source)
            jhit = cache.get(judge_model, jjob.job_id, jjob.sha)
            result["verdict"] = parse_judge(jhit["response"]) if jhit else None
            result["judge_cost_usd"] = float(jhit.get("cost_usd") or 0) if jhit else 0.0
        results.append((job, result, hit))
    return jobs, {"results": results}


def _rates(results: list[tuple[Any, dict[str, Any], dict[str, Any]]]) -> dict[str, Any]:
    from robo_jev.simgen.narration_checks import CHECK_NAMES

    per_voice: dict[str, Any] = {}
    by_voice = defaultdict(list)
    for job, result, _ in results:
        by_voice[job.voice.key].append(result)
    for voice, items in sorted(by_voice.items()):
        fails = Counter()
        for r in items:
            for name, problems in r["checks"].items():
                if problems:
                    fails[name] += 1
        q_total = sum(len(r["question_status"]) for r in items if r["passed"] and not r["question_status"].get(next(iter(r["question_status"]), ""), "").startswith("original (voice"))
        q_revoiced = sum(1 for r in items if r["passed"] for s in r["question_status"].values() if s == "revoiced")
        judged = [r for r in items if r.get("verdict") is not None]
        accepted = sum(1 for r in judged if r["verdict"].get("ok"))
        per_voice[voice] = {"replies": len(items), "passed": sum(r["passed"] for r in items),
                            "pass_rate": round(sum(r["passed"] for r in items) / len(items), 3) if items else None,
                            "fails": {k: fails[k] for k in (*CHECK_NAMES, "parse", "truncated") if fails[k]},
                            "questions_revoiced": f"{q_revoiced}/{q_total}" if q_total else None,
                            "back_check": {"judged": len(judged), "accepted": accepted,
                                           "major": sum(1 for r in judged if r["verdict"].get("major")),
                                           "unreadable": sum(1 for r in judged if r["verdict"].get("ok") is None)} if judged else None,
                            "final": sum(1 for r in items if r["passed"] and (r.get("verdict") or {}).get("ok")) if judged else None}
    total = len(results)
    passed = sum(r["passed"] for _, r, _ in results)
    all_fails = Counter()
    for _, r, _ in results:
        for name, problems in r["checks"].items():
            if problems:
                all_fails[name] += 1
    judged = [r for _, r, _ in results if r.get("verdict") is not None]
    back = {"judged": len(judged), "accepted": sum(1 for r in judged if r["verdict"].get("ok")),
            "with_major": sum(1 for r in judged if r["verdict"].get("major")),
            "unreadable": sum(1 for r in judged if r["verdict"].get("ok") is None)} if judged else None
    final = sum(1 for _, r, _ in results if r["passed"] and (r.get("verdict") or {}).get("ok")) if judged else None
    return {"replies": total, "passed": passed, "pass_rate": round(passed / total, 3) if total else None, "fails": dict(all_fails),
            "back_check": back, "final": final, "final_rate": round(final / total, 3) if judged and total else None, "per_voice": per_voice}


def cmd_report(args: argparse.Namespace) -> int:
    plan = _plan()
    entries = _select(plan, args)
    jobs, data = _evaluate_range(args.model, entries, args.judge_model)
    results = data["results"]
    rates = _rates(results)
    rng = random.Random(args.seed)
    examples = {"passed": [], "failed": []}
    passed = [x for x in results if x[1]["passed"]]
    failed = [x for x in results if not x[1]["passed"]]
    for kind, pool in (("passed", passed), ("failed", failed)):
        for job, result, hit in rng.sample(pool, min(args.examples, len(pool))):
            examples[kind].append({"job": job.job_id, "voice": job.voice.key, "language": job.voice.language, "scene": job.entry["scene"],
                                   "original": job.source["request"]["state"], "rewrite": result["state"],
                                   "checks": {k: v for k, v in result["checks"].items() if v}, "question_status": result["question_status"],
                                   "questions": result["questions"] if kind == "passed" else None, "verdict": result.get("verdict")})
    usage = Counter()
    for _, _, hit in results:
        for key in ("input_tokens", "output_tokens", "reasoning_tokens"):
            usage[key] += int((hit.get("usage") or {}).get(key) or 0)
        usage["cost_usd_x1e6"] += int(round(float(hit.get("cost_usd") or 0) * 1e6))
    report = {"model": args.model, "start": args.start, "count": args.count, "jobs": len(jobs), **rates,
              "usage": {**{k: v for k, v in usage.items() if k != "cost_usd_x1e6"}, "cost_usd": usage["cost_usd_x1e6"] / 1e6},
              "examples": examples}
    _write_json(args.out, report)
    _log(f"{args.model}: {rates['passed']}/{rates['replies']} passed checks ({rates['pass_rate']}); fails {rates['fails']}; "
         f"back-check {rates['back_check']}; final {rates['final']} ({rates['final_rate']}); cost ${report['usage']['cost_usd']:.4f} → {args.out}")
    for voice, row in rates["per_voice"].items():
        _log(f"  {voice:18s} {row['passed']:4d}/{row['replies']:<4d} {row['pass_rate']}  final {row['final']}  {row['fails']}  "
             f"q {row['questions_revoiced']}  bc {row['back_check']}")
    return 0


# --------------------------------------------------------------------------
# back-check
# --------------------------------------------------------------------------


def cmd_judge(args: argparse.Namespace) -> int:
    from robo_jev.bench.openrouter import DecisionsClient, ResponseCache, load_api_key
    from robo_jev.simgen.narration import CHAT_URL
    from robo_jev.simgen.narration_run import judge_job, parse_judge, run_jobs

    plan = _plan()
    entries = plan[args.start : args.start + args.count]
    rng = random.Random(args.seed)
    if args.sample_moments:  # evaluate only a random subset of the range (cheaper on a long range)
        entries = rng.sample(entries, min(args.sample_moments, len(entries)))
    _, data = _evaluate_range(args.model, entries, args.accepted_by)
    if args.accepted_by:  # audit the final records: rewrites that passed the checks and the first back-check
        pool = [x for x in data["results"] if x[1]["passed"] and (x[1].get("verdict") or {}).get("ok")]
    else:
        pool = [x for x in data["results"] if x[1]["passed"] or args.include_failed]
    sample = rng.sample(pool, min(args.sample, len(pool)))
    price = _price(args.judge_model)
    jobs = [judge_job(f"judge~{job.job_id}", job.source["request"]["state"], result["state"], args.judge_model, price, job.voice.language,
                      entry=job.entry, source=job.source) for job, result, _ in sample]
    ledger = _ledger(args.max_total_usd)
    client = DecisionsClient(load_api_key(), url=CHAT_URL, timeout=args.timeout) if not args.cache_only else None
    out = run_jobs(args.judge_model, jobs, client=client, cache=ResponseCache(CACHE), ledger=ledger, price=price, index=_index(),
                   calls_log=CALLS, concurrency=args.concurrency, log=_log)
    ledger.runs.append({"tag": f"judge-{args.tag}", "model": args.judge_model, "jobs": len(jobs), "sent": out["stats"]["sent"],
                        "cost_usd": out["stats"]["cost_usd"], "finished": dt.datetime.now().astimezone().isoformat(timespec="seconds")})
    ledger.save()
    verdicts = []
    per_voice = defaultdict(Counter)
    for (job, result, _), judge in zip(sample, jobs):
        response = out["responses"].get(judge.job_id)
        if response is None:
            continue
        verdict = parse_judge(response)
        verdicts.append({"job": job.job_id, "voice": job.voice.key, "passed_checks": result["passed"], **verdict})
        per_voice[job.voice.key]["judged"] += 1
        per_voice[job.voice.key]["judge_ok" if verdict.get("ok") else ("judge_flag" if verdict.get("ok") is False else "judge_error")] += 1
    report = {"rewrite_model": args.model, "judge_model": args.judge_model, "sample": len(sample), "judged": len(verdicts),
              "judge_ok": sum(1 for v in verdicts if v.get("ok")), "per_voice": {k: dict(v) for k, v in sorted(per_voice.items())},
              "verdicts": verdicts}
    _write_json(args.out, report)
    _log(f"judge {args.judge_model}: {report['judge_ok']}/{report['judged']} with no fact difference → {args.out}")
    return 0


# --------------------------------------------------------------------------
# assemble, audit
# --------------------------------------------------------------------------


def cmd_assemble(args: argparse.Namespace) -> int:
    from robo_jev.bench.openrouter import ResponseCache
    from robo_jev.simgen.narration import NARRATION_VERSION, PROMPT_VERSION, assemble_record
    from robo_jev.simgen.narration_run import evaluate, judge_id, judge_job, parse_judge
    from robo_jev.simgen.record import dumps, validate_record

    plan = _plan()
    cache = ResponseCache(CACHE)
    price = {"prompt": 0.0, "completion": 0.0}
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    temp = out_path.with_name(out_path.name + ".tmp")
    counts = {"voice": Counter(), "scene": Counter(), "language": Counter(), "group": Counter()}
    moments, replies, records, invalid = set(), 0, 0, 0
    results_all = []
    q_status = Counter()
    ratios: dict[str, list[float]] = defaultdict(list)
    chunk = 2000
    with temp.open("w", encoding="utf-8") as handle:
        for begin in range(0, len(plan), chunk):
            entries = plan[begin : begin + chunk]
            jobs = _jobs(args.model, entries, price)
            cached = [(job, cache.get(args.model, job.job_id, job.sha)) for job in jobs]
            cached = [(job, hit) for job, hit in cached if hit is not None]
            if not cached and begin > 0:
                break
            for job, hit in cached:
                replies += 1
                result = evaluate(job, hit["response"])
                if args.judge_model and result["passed"]:
                    jjob = judge_job(judge_id(job.job_id), job.source["request"]["state"], result["state"], args.judge_model, price,
                                     job.voice.language, entry=job.entry, source=job.source)
                    jhit = cache.get(args.judge_model, jjob.job_id, jjob.sha)
                    result["verdict"] = parse_judge(jhit["response"]) if jhit else None
                results_all.append((job, result, hit))
                if not result["passed"]:
                    continue
                if args.judge_model and not (result.get("verdict") or {}).get("ok"):
                    continue
                checks_summary = {k: "pass" for k in result["checks"]}
                narration = {"version": NARRATION_VERSION, "voice": job.voice.key, "person": job.voice.person, "family": job.voice.family,
                             "language": job.voice.language, "order_hint": int(job.entry["order"][job.voice.key]), "model": args.model,
                             "resolved_model": hit.get("resolved_model"), "provider": hit.get("provider"), "prompt_version": PROMPT_VERSION,
                             "source_id": job.source["id"], "source_state_sha256": hashlib.sha256(job.source["request"]["state"].encode("utf-8")).hexdigest(),
                             "plan_rank": job.entry["rank"], "checks": checks_summary, "questions": result["question_status"],
                             "back_check": ({"model": args.judge_model, "minor": result["verdict"]["minor"],
                                             "differences": result["verdict"]["differences"]} if args.judge_model else None),
                             "length_ratio": round(len(result["state"]) / max(1, len(job.source["request"]["state"])), 3)}
                record = assemble_record(job.source, voice=job.voice, state=result["state"], questions=result["questions"],
                                         question_status=result["question_status"], narration=narration)
                errors = validate_record(record)
                if errors:
                    invalid += 1
                    continue
                handle.write(dumps(record) + "\n")
                records += 1
                moments.add(job.source["id"])
                counts["voice"][job.voice.key] += 1
                counts["scene"][job.entry["scene"]] += 1
                counts["language"][job.voice.language] += 1
                counts["group"][job.entry["group"]] += 1
                ratios[job.voice.key].append(narration["length_ratio"])
                for status in result["question_status"].values():
                    q_status[status.split(" ")[0]] += 1
    temp.replace(out_path)
    rates = _rates(results_all)
    ledger = json.loads(LEDGER.read_text(encoding="utf-8")) if LEDGER.is_file() else {}
    audit = _audit()
    manifest = {
        "file": out_path.name, "sha256": _sha256_file(out_path), "bytes": out_path.stat().st_size, "records": records,
        "moments": len(moments), "replies_checked": replies, "invalid_format": invalid, "schema": "simgen-request/0",
        "narration_version": NARRATION_VERSION, "prompt_version": PROMPT_VERSION, "model": args.model,
        "source": {"path": str(SOURCE.relative_to(REPO)), "sha256": _sha256_file(SOURCE),
                   "manifest": json.loads(SOURCE_MANIFEST.read_text(encoding="utf-8")) if SOURCE_MANIFEST.is_file() else None},
        "plan": {"path": str(PLAN.relative_to(REPO)), "sha256": _sha256_file(PLAN), "moments": len(plan)},
        "counts": {k: dict(sorted(v.items())) for k, v in counts.items()},
        "questions": dict(q_status),
        "length_ratio": {voice: _quantiles(values) for voice, values in sorted(ratios.items())}
                        | {"all": _quantiles([v for values in ratios.values() for v in values])},
        "check_pass_rates": rates,
        "spend": {"ledger": str(LEDGER.relative_to(REPO)), "totals": ledger.get("totals"), "models": ledger.get("models"),
                  "hard_cap_usd": HARD_CAP_USD},
        "sealed_audit": audit,
        "note": "Training data only. labels, truth, rationale, generator and every source meta field are the v02 source record's, "
                "unchanged; request.state (and, for first/third-person voices, re-voiced question texts) are new; meta.narration "
                "records voice, model, checks and source id. meta.style / meta.fact_order describe the source text, not the narration.",
        "created": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
    }
    _write_json(out_path.with_name(out_path.stem + ".manifest.json"), manifest)
    _log(f"{records} records from {len(moments)} moments ({replies} replies checked, pass {rates['pass_rate']}) → {out_path}")
    return 0


def _quantiles(values: list[float]) -> dict[str, float] | None:
    if not values:
        return None
    ordered = sorted(values)
    pick = lambda q: ordered[min(len(ordered) - 1, int(q * len(ordered)))]  # noqa: E731
    return {"n": len(ordered), "p10": pick(0.1), "median": pick(0.5), "p90": pick(0.9), "mean": round(sum(ordered) / len(ordered), 3)}


def _audit() -> dict[str, Any]:
    from robo_jev.simgen.narration import audit_request_log

    ids = set()
    with SOURCE.open(encoding="utf-8") as handle:
        for line in handle:
            start = line.find('"id":"') + 6
            ids.add(line[start : line.find('"', start)])
    result = audit_request_log(CALLS, _index(), source_ids=ids)
    result["log"] = str(CALLS.relative_to(REPO))
    result["log_sha256"] = _sha256_file(CALLS) if CALLS.is_file() else None
    result["sealed_files"] = _index().files
    return result


def cmd_audit(args: argparse.Namespace) -> int:
    result = _audit()
    _log(f"audit: {result['requests']} logged requests, {len(result['violations'])} violations")
    for v in result["violations"][:20]:
        _log(f"  {v}")
    _write_json(OUT / "sealed-audit.json", result)
    return 0 if not result["violations"] else 3


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("index", help="sealed-set hash index")
    plan = sub.add_parser("plan", help="stratified narration plan")
    plan.add_argument("--seed", type=int, default=0)
    for name in ("run", "judge", "pipeline"):
        command = sub.add_parser(name)
        command.add_argument("--model", required=True)
        command.add_argument("--start", type=int, default=0)
        command.add_argument("--count", type=int, required=True)
        command.add_argument("--tag", default=name)
        command.add_argument("--max-total-usd", type=float, required=True)
        command.add_argument("--concurrency", type=int, default=16)
        command.add_argument("--timeout", type=float, default=60.0)
        command.add_argument("--cache-only", action="store_true")
        command.add_argument("--only-language", choices=("en", "ko"))
        if name == "pipeline":
            command.add_argument("--judge-model", required=True)
        if name == "judge":
            command.add_argument("--judge-model", required=True)
            command.add_argument("--sample", type=int, default=400)
            command.add_argument("--seed", type=int, default=0)
            command.add_argument("--include-failed", action="store_true")
            command.add_argument("--accepted-by", help="sample only rewrites this back-check model accepted (audit of the final records)")
            command.add_argument("--sample-moments", type=int, help="evaluate a random subset of this many plan moments")
            command.add_argument("--out", type=Path, default=OUT / "judge-report.json")
    report = sub.add_parser("report")
    report.add_argument("--model", required=True)
    report.add_argument("--start", type=int, default=0)
    report.add_argument("--count", type=int, required=True)
    report.add_argument("--examples", type=int, default=12)
    report.add_argument("--seed", type=int, default=0)
    report.add_argument("--only-language", choices=("en", "ko"))
    report.add_argument("--judge-model")
    report.add_argument("--out", type=Path, required=True)
    assemble = sub.add_parser("assemble")
    assemble.add_argument("--model", required=True)
    assemble.add_argument("--out", type=Path, default=OUT / "narrated-train.jsonl")
    assemble.add_argument("--judge-model")
    sub.add_parser("audit")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return {"index": cmd_index, "plan": cmd_plan, "run": cmd_run, "pipeline": cmd_pipeline, "report": cmd_report, "judge": cmd_judge,
            "assemble": cmd_assemble, "audit": cmd_audit}[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
