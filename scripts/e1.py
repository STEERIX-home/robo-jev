#!/usr/bin/env python
"""Task E1 — the pre-declared 2×2 isolation experiment (description format H1 vs physical factors H2).

    # A: fresh engine moments (P3 tipping/falling scenes, train split, v0b phrasing, seeds from 3,000,000) → artifacts/e1/A.jsonl
    python scripts/e1.py make-a --n-per-scene 100
    # B: the same moments rewritten by an LLM into first-person robot narration (OpenRouter, cap $3, cached) → artifacts/e1/B.jsonl
    python scripts/e1.py rewrite-b --model <openrouter model> --max-cost-usd 3.00 [--limit N]
    # C and D: T8b moments rendered in the engine's prose style (deterministic, no LLM) and T8b itself, same questions
    python scripts/e1.py make-cd
    # analysis: situation reading r, paired bootstrap, the pre-declared rule (after the GPU diagnoses of A–D for both models)
    python scripts/e1.py analyze

Run CPU work in a capped systemd unit; GPU scoring goes through scripts/diagnose_decisions.py (artifacts/scratch/e1/gpu.sh).
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

OUT = REPO / "artifacts" / "e1"
T8B = REPO / "artifacts" / "t8" / "T8b.jsonl"


def _log(message: str) -> None:
    print(f"[e1 {dt.datetime.now().strftime('%H:%M:%S')}] {message}", file=sys.stderr, flush=True)


def _git() -> str | None:
    try:
        return subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True).stdout.strip() or None
    except (OSError, subprocess.CalledProcessError):
        return None


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_jsonl(path: Path, records) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")


def cmd_make_a(args) -> int:
    from robo_jev.decisions.records import validate_record
    from robo_jev.simgen.catalog import default_config
    from robo_jev.e1.engine import generate_a, plan_a

    cfg = default_config()
    units = plan_a(cfg, args.n_per_scene)
    records, report = generate_a(cfg, units, git=_git(), log=_log)
    problems = {r["id"]: p for r in records if (p := validate_record(r))}
    path = args.out_dir / "A.jsonl"
    _write_jsonl(path, records)
    report.update({"file": str(path), "sha256": _sha256(path), "validation_problems": problems, "git": _git(), "created": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
                   "questions": sum(len(r["request"]["questions"]) for r in records)})
    _write_json(args.out_dir / "A.manifest.json", report)
    _log(f"A: {len(records)} records, {report['questions']} questions, {len(problems)} invalid → {path}")
    return 1 if problems else 0


def cmd_make_cd(args) -> int:
    from robo_jev.decisions.records import validate_record
    from robo_jev.e1.render import check_c, render_c, restrict_d

    t8b = _read_jsonl(args.t8b)
    c_records, d_records, problems = [], [], {}
    for record in t8b:
        c = render_c(record)
        found = check_c(c, record) + validate_record(c)
        if found:
            problems[record["id"]] = found
        c_records.append(c)
        d_records.append(restrict_d(record))
    again = [render_c(record) for record in t8b]
    deterministic = json.dumps(again, ensure_ascii=False) == json.dumps(c_records, ensure_ascii=False)
    paths = {"C": args.out_dir / "C.jsonl", "D": args.out_dir / "D.jsonl"}
    _write_jsonl(paths["C"], c_records)
    _write_jsonl(paths["D"], d_records)
    manifest = {"source": str(args.t8b), "source_sha256": _sha256(args.t8b), "records": len(t8b), "deterministic": deterministic,
                "check_problems": problems, "questions": {k: sum(len(r["request"]["questions"]) for r in v) for k, v in (("C", c_records), ("D", d_records))},
                "files": {k: {"path": str(p), "sha256": _sha256(p)} for k, p in paths.items()}, "git": _git(),
                "created": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
                "note": "C and D are T8b-derived and sealed: scored locally only, never sent to any API."}
    _write_json(args.out_dir / "CD.manifest.json", manifest)
    _log(f"C/D: {len(t8b)} moments, {manifest['questions']} questions, {len(problems)} with problems, deterministic {deterministic}")
    return 1 if problems or not deterministic else 0


CHAT_URL = "https://openrouter.ai/api/v1/chat/completions"
ALL_MODELS_URL = "https://openrouter.ai/api/v1/models"
SEALED_FILES = {**{f"T{i}": REPO / "artifacts" / "simgen" / "v0b" / "tests" / f"T{i}.jsonl" for i in range(1, 8)},
                "G1": REPO / "artifacts" / "simgen" / "probes" / "G1.jsonl", "G2": REPO / "artifacts" / "simgen" / "probes" / "G2.jsonl",
                "T8a": REPO / "artifacts" / "t8" / "T8a.jsonl", "T8b": REPO / "artifacts" / "t8" / "T8b.jsonl",
                "T9": REPO / "artifacts" / "t9" / "T9.jsonl"}
OUTSIDE_SETS = ("G1", "G2", "T8a", "T8b", "T9")


def cmd_rewrite_b(args) -> int:
    import threading
    import time
    from collections import Counter
    from concurrent.futures import ThreadPoolExecutor

    from robo_jev.bench.openrouter import BudgetExceeded, DecisionsClient, DecisionsError, Ledger, ResponseCache, fetch_models, load_api_key, price_of, \
        request_sha256
    from robo_jev.decisions.records import validate_record
    from robo_jev.e1.rewrite import PROMPT_VERSION, SealedIndex, assert_sendable, b_record, build_messages, check_b, object_phrases, parse_reply
    from robo_jev.simgen.catalog import default_config

    a_records = _read_jsonl(args.out_dir / "A.jsonl")
    if args.limit:
        a_records = a_records[: args.limit]
    cfg = default_config()
    index = SealedIndex.build(SEALED_FILES, OUTSIDE_SETS)
    _log(f"sealed index: {len(index.states)} states, {len(index.sentences)} outside sentences from {len(index.files)} files")
    catalog = fetch_models(url=ALL_MODELS_URL)
    entry = next((m for m in catalog["data"] if m.get("id") == args.model), None)
    if entry is None:
        raise SystemExit(f"{args.model} is not in the OpenRouter model list")
    price = price_of(entry)
    root = args.out_dir / "openrouter"
    cache = ResponseCache(root / "cache")
    ledger = Ledger(root / "ledger.json", max_total_usd=args.max_cost_usd, max_model_usd=args.max_cost_usd)
    client = None if args.cache_only else DecisionsClient(load_api_key(), url=CHAT_URL, timeout=180.0)
    calls_log = root / "calls.jsonl"
    lock = threading.Lock()
    results: dict[str, dict] = {}
    stats = Counter()
    stop = threading.Event()

    def note(entry: dict) -> None:
        with lock:
            calls_log.parent.mkdir(parents=True, exist_ok=True)
            with calls_log.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def one(a: dict) -> None:
        rid = a["id"]
        messages = build_messages(a)
        assert_sendable(a, [a["request"]["state"], *(q["instructions"] for q in a["request"]["questions"].values()), messages[0]["content"]], index)
        payload = {"model": args.model, "messages": messages, "max_tokens": args.max_tokens, "provider": {"data_collection": "deny"},
                   "usage": {"include": True}, "response_format": {"type": "json_object"}}
        sha = request_sha256(payload)
        hit = cache.get(args.model, rid, sha)
        if hit is not None:
            response = hit["response"]
            with lock:
                stats["cached"] += 1
        else:
            if client is None or stop.is_set():
                with lock:
                    results[rid] = {"status": "not_sent"}
                return
            chars = sum(len(m["content"]) for m in messages)
            estimate = (chars / 3.5) * price["prompt"] + min(args.max_tokens, 2.5 * len(a["request"]["state"]) / 3.5 + 400) * price["completion"]
            try:
                reserved = ledger.reserve(args.model, estimate)
            except BudgetExceeded as error:
                stop.set()
                with lock:
                    results[rid] = {"status": "budget", "error": str(error)}
                return
            try:
                response, info = client.post(payload)
            except DecisionsError as error:
                if error.uncertain:
                    ledger.settle(args.model, reserved, estimate, cost=None, usage=None, price=price, uncertain=True)
                else:
                    ledger.release(args.model, reserved)
                ledger.save()
                note({"time": time.time(), "record_id": rid, "request_sha256": sha, "error": str(error), "status": error.status})
                with lock:
                    results[rid] = {"status": "api_error", "error": str(error)}
                    if error.fatal:
                        stop.set()
                return
            raw_usage = response.get("usage") if isinstance(response.get("usage"), dict) else {}
            usage = {"input_tokens": raw_usage.get("prompt_tokens"), "output_tokens": raw_usage.get("completion_tokens"), "cost": raw_usage.get("cost")}
            cost = raw_usage.get("cost") if isinstance(raw_usage.get("cost"), (int, float)) else None
            actual = ledger.settle(args.model, reserved, estimate, cost=cost, usage=usage, price=price)
            if info["uncertain_attempts"]:
                ledger.add_uncertain(args.model, reserved * int(info["uncertain_attempts"]))
            ledger.save()
            meta = {"resolved_model": response.get("model"), "provider": response.get("provider"), "generation_id": response.get("id"), "usage": usage,
                    "cost_usd": actual, "attempts": info["attempts"], "seconds": info["seconds"], "prompt_version": PROMPT_VERSION}
            if isinstance(response.get("choices"), list) and response["choices"]:
                cache.put(args.model, rid, sha, response, meta=meta)
            note({"time": time.time(), "record_id": rid, "request_sha256": sha, **meta})
            with lock:
                stats["sent"] += 1
                stats["cost_usd"] += actual
        try:
            content = response["choices"][0]["message"]["content"]
            reply = parse_reply(content)
        except (KeyError, IndexError, TypeError, ValueError) as error:
            with lock:
                results[rid] = {"status": "failed", "reasons": {"parse": [str(error)[:300]]}, "request_sha256": sha}
            return
        objects = object_phrases(cfg, a)
        problems = check_b(a, reply, objects)
        record = b_record(a, reply, model=args.model, objects=objects, request_sha256=sha, resolved=response.get("model"))
        problems_v = validate_record(record)
        if problems_v:
            problems.setdefault("record", []).extend(problems_v)
        with lock:
            results[rid] = {"status": "failed" if problems else "passed", "reasons": problems, "record": record, "request_sha256": sha}

    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        list(pool.map(one, a_records))
    seconds = time.perf_counter() - started
    order = [a["id"] for a in a_records]
    passed = [results[rid]["record"] for rid in order if results.get(rid, {}).get("status") == "passed"]
    failed = [{"id": rid, "status": results[rid]["status"], "reasons": results[rid].get("reasons"),
               "reply": (results[rid].get("record") or {}).get("request")} for rid in order if results.get(rid, {}).get("status") != "passed"]
    by_reason = Counter(k for f in failed for k in (f["reasons"] or {"<" + f["status"] + ">": 1}))
    ledger.runs.append({"tag": args.tag, "model": args.model, "records": len(a_records), "sent": stats["sent"], "cached": stats["cached"],
                        "cost_usd": stats["cost_usd"], "prompt_version": PROMPT_VERSION, "finished": dt.datetime.now().astimezone().isoformat(timespec="seconds")})
    ledger.save()
    if args.limit:
        _log(f"pilot ({args.limit}): {len(passed)} passed, by reason {dict(by_reason)}; spent this run ${stats['cost_usd']:.4f}, ledger total ${ledger.spent():.4f}")
        _write_jsonl(args.out_dir / "B.pilot.jsonl", passed)
        _write_jsonl(args.out_dir / "B.pilot.failures.jsonl", failed)
        return 0
    path = args.out_dir / "B.jsonl"
    _write_jsonl(path, passed)
    _write_jsonl(args.out_dir / "B.failures.jsonl", failed)
    manifest = {"model": args.model, "price_per_token": price, "prompt_version": PROMPT_VERSION, "records_a": len(a_records), "passed": len(passed),
                "failed": len(failed), "failure_rate": round(len(failed) / max(1, len(a_records)), 4), "failures_by_check": dict(by_reason),
                "failures_by_status": dict(Counter(f["status"] for f in failed)), "sent": stats["sent"], "cached": stats["cached"],
                "spent_this_run_usd": stats["cost_usd"], "ledger_total_usd": ledger.spent(), "cap_usd": args.max_cost_usd, "seconds": round(seconds, 1),
                "sealed_index": {"files": index.files, "states": len(index.states), "outside_sentences": len(index.sentences)},
                "file": str(path), "sha256": _sha256(path), "a_sha256": _sha256(args.out_dir / "A.jsonl"), "git": _git(),
                "created": dt.datetime.now().astimezone().isoformat(timespec="seconds")}
    _write_json(args.out_dir / "B.manifest.json", manifest)
    _log(f"B: {len(passed)}/{len(a_records)} passed ({manifest['failure_rate']:.1%} dropped: {dict(by_reason)}); this run ${stats['cost_usd']:.4f}, "
         f"ledger ${ledger.spent():.4f} of ${args.max_cost_usd:.2f}")
    return 0


MODELS = {"v01mix-2b": "v0.1-mix (2B + LoRA v01-mix-2b@3150) — primary", "base-2b": "Qwen3.5-2B untrained — reference"}
BOOT_SEED = 20261012


def _moment_ab(record: dict) -> str:
    return record["meta"]["e1"]["source_id"] if record["id"].startswith("e1b-") else record["id"]


def _moment_cd(record: dict) -> str:
    return record["meta"]["e1"]["source_id"]


def _system1(records: list[dict], predictions: list[dict], key, keep: set[str] | None, reading: str) -> dict:
    """band_balanced / severe_error items of system1.py, with record ids replaced by the moment key (so two conditions pair)."""
    from robo_jev.bench.system1 import question_items

    ids = {}
    recs = []
    for r in records:
        k = key(r)
        if keep is not None and k not in keep:
            continue
        ids[r["id"]] = k
        recs.append({**r, "id": k})
    preds = [{**p, "id": ids[p["id"]]} for p in predictions if p["id"] in ids]
    items = question_items("E1", recs, preds, reading=reading)
    return {"band_balanced": items.get("band_balanced", {}), "severe_error": items.get("severe_error", {})}


def cmd_analyze(args) -> int:
    from robo_jev.bench.system1 import paired_difference, summarize
    from robo_jev.e1.analysis import apply_rule, build_items, paired_bootstrap

    data = {name: _read_jsonl(args.out_dir / f"{name}.jsonl") for name in ("A", "B", "C", "D")}
    b_moments = {_moment_ab(r) for r in data["B"]}
    out: dict = {"design": "progress.md — PRE-DECLARED ISOLATION EXPERIMENT E1 (2026-10-12 ~19:05 KST)", "n_boot": args.n_boot, "seed": BOOT_SEED,
                 "files": {name: {"path": str(args.out_dir / f"{name}.jsonl"), "sha256": _sha256(args.out_dir / f"{name}.jsonl"), "records": len(recs)}
                           for name, recs in data.items()},
                 "pairs": {"AB_moments": len(b_moments), "A_moments": len(data["A"]), "CD_moments": len(data["C"])}, "models": {}}
    for model, label in MODELS.items():
        preds = {name: _read_jsonl(args.out_dir / "pred" / f"{name}-{model}.predictions.jsonl") for name in data}
        items = {"A": build_items(data["A"], preds["A"], moment_key=_moment_ab, keep=b_moments),
                 "B": build_items(data["B"], preds["B"], moment_key=_moment_ab),
                 "C": build_items(data["C"], preds["C"], moment_key=_moment_cd),
                 "D": build_items(data["D"], preds["D"], moment_key=_moment_cd)}
        ab = paired_bootstrap(items["A"], items["B"], n_boot=args.n_boot, seed=BOOT_SEED)
        cd = paired_bootstrap(items["D"], items["C"], n_boot=args.n_boot, seed=BOOT_SEED)
        rule = apply_rule(ab["delta"], cd["delta"], cd["y"]["r"])
        secondary = {}
        for reading in ("single_order", "order_averaged"):
            s1 = {"A": _system1(data["A"], preds["A"], _moment_ab, b_moments, reading), "B": _system1(data["B"], preds["B"], _moment_ab, None, reading),
                  "C": _system1(data["C"], preds["C"], _moment_cd, None, reading), "D": _system1(data["D"], preds["D"], _moment_cd, None, reading)}
            block = {}
            for metric in ("band_balanced", "severe_error"):
                strat = metric == "band_balanced"
                block[metric] = {name: summarize(s1[name][metric], n_boot=args.n_boot, seed=BOOT_SEED, stratified=strat) for name in s1}
                block[metric]["B-A"] = paired_difference(s1["B"][metric], s1["A"][metric], n_boot=args.n_boot, seed=BOOT_SEED, stratified=strat)
                block[metric]["C-D"] = paired_difference(s1["C"][metric], s1["D"][metric], n_boot=args.n_boot, seed=BOOT_SEED, stratified=strat)
            secondary[reading] = block
        out["models"][model] = {"label": label, "r_A": ab["x"], "r_B": ab["y"], "delta_format_engine": ab["delta"], "r_D": cd["x"], "r_C": cd["y"],
                                "delta_format_t8b": cd["delta"], "groups_AB": ab["groups"], "groups_CD": cd["groups"], "rule": rule, "secondary": secondary}
        _log(f"{model}: r_A {ab['x']['r']:+.3f} r_B {ab['y']['r']:+.3f} Δe {ab['delta']['value']:+.3f} [{ab['delta']['low']:+.3f}, {ab['delta']['high']:+.3f}] · "
             f"r_D {cd['x']['r']:+.3f} r_C {cd['y']['r']:+.3f} Δt {cd['delta']['value']:+.3f} [{cd['delta']['low']:+.3f}, {cd['delta']['high']:+.3f}] → {rule['verdict']}")
    out["verdict"] = out["models"]["v01mix-2b"]["rule"]["verdict"]
    out["created"] = dt.datetime.now().astimezone().isoformat(timespec="seconds")
    out["git"] = _git()
    _write_json(args.out_dir / "results.json", out)
    (args.out_dir / "results.md").write_text(_render_results(out), encoding="utf-8")
    _log(f"verdict (v0.1-mix): {out['verdict']} → {args.out_dir / 'results.json'}, results.md")
    return 0


def _f(x, signed: bool = True) -> str:
    if x is None:
        return "—"
    return f"{x:+.3f}" if signed else f"{x:.3f}"


def _ci(entry) -> str:
    return f"[{_f(entry.get('low'))}, {_f(entry.get('high'))}]"


def _render_results(out: dict) -> str:
    lines = ["# Task E1 — results (pre-declared 2×2)", "",
             f"Moments: A/B {out['pairs']['AB_moments']} paired (of {out['pairs']['A_moments']} A), C/D {out['pairs']['CD_moments']}. "
             f"Paired bootstrap over moments, {out['n_boot']} draws, seed {out['seed']}.", "",
             "## Primary — situation reading (within-group Pearson r, pooled over groups with n ≥ 20, n-weighted)", "",
             "| model | r_A | r_B | Δformat_engine = r_B − r_A | r_D | r_C | Δformat_T8b = r_C − r_D | verdict |", "|---|---|---|---|---|---|---|---|"]
    for model, m in out["models"].items():
        lines.append(f"| {m['label']} | {_f(m['r_A']['r'])} {_ci(m['r_A'])} | {_f(m['r_B']['r'])} {_ci(m['r_B'])} | **{_f(m['delta_format_engine']['value'])}** "
                     f"{_ci(m['delta_format_engine'])} | {_f(m['r_D']['r'])} {_ci(m['r_D'])} | **{_f(m['r_C']['r'])}** {_ci(m['r_C'])} | "
                     f"**{_f(m['delta_format_t8b']['value'])}** {_ci(m['delta_format_t8b'])} | {m['rule']['verdict']} |")
    lines += ["", "Rule conditions (v0.1-mix is the primary model; the untrained base is a reference):", ""]
    for model, m in out["models"].items():
        lines.append(f"- {m['label']}: " + "; ".join(f"{name}: " + ", ".join(f"{k} {'✓' if v else '✗'}" for k, v in parts.items())
                                                       for name, parts in m["rule"]["conditions"].items()))
    for pair, (x, y) in (("A/B", ("r_A", "r_B")), ("C/D", ("r_D", "r_C"))):
        lines += ["", f"### Groups {pair}", "", "| model | group | n | " + f"{x} | {y} |", "|---|---|---|---|---|"]
        for model, m in out["models"].items():
            for g, row in m[x]["groups"].items():
                lines.append(f"| {model} | {g} | {row['n']} | {_f(row['r'])} | {_f(m[y]['groups'][g]['r'])} |")
    lines += ["", "## Secondary — system1.py (single-order reading; order-averaged in results.json)", "",
              "| model | metric | A | B | B − A | D | C | C − D |", "|---|---|---|---|---|---|---|---|"]
    for model, m in out["models"].items():
        s = m["secondary"]["single_order"]
        for metric in ("band_balanced", "severe_error"):
            b = s[metric]
            lines.append(f"| {model} | {metric} | {_f(b['A']['value'], False)} | {_f(b['B']['value'], False)} | {_f(b['B-A']['value'])} {_ci(b['B-A'])} | "
                         f"{_f(b['D']['value'], False)} | {_f(b['C']['value'], False)} | {_f(b['C-D']['value'])} {_ci(b['C-D'])} |")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    make_a = sub.add_parser("make-a")
    make_a.add_argument("--n-per-scene", type=int, default=100)
    make_a.add_argument("--out-dir", type=Path, default=OUT)
    make_cd = sub.add_parser("make-cd")
    make_cd.add_argument("--t8b", type=Path, default=T8B)
    make_cd.add_argument("--out-dir", type=Path, default=OUT)
    rewrite = sub.add_parser("rewrite-b")
    rewrite.add_argument("--model", required=True)
    rewrite.add_argument("--max-cost-usd", type=float, required=True, help="cap on the E1 ledger total (cumulative)")
    rewrite.add_argument("--max-tokens", type=int, default=3000)
    rewrite.add_argument("--concurrency", type=int, default=8)
    rewrite.add_argument("--limit", type=int, default=None, help="pilot: the first N A records only (writes B.pilot*.jsonl)")
    rewrite.add_argument("--cache-only", action="store_true", help="never send; read cached replies only")
    rewrite.add_argument("--tag", default="b")
    rewrite.add_argument("--out-dir", type=Path, default=OUT)
    analyze = sub.add_parser("analyze")
    analyze.add_argument("--n-boot", type=int, default=2000)
    analyze.add_argument("--out-dir", type=Path, default=OUT)
    args = parser.parse_args(argv)
    return {"make-a": cmd_make_a, "make-cd": cmd_make_cd, "rewrite-b": cmd_rewrite_b, "analyze": cmd_analyze}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
