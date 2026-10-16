#!/usr/bin/env python
"""Task v0.8b — send the Jev comparison set to ``~typesafe/jev-latest`` (OpenRouter Decisions API) and store its answers (CPU unit, network).

    artifacts/scratch/v08b/cpu.sh jev-dry scripts/v08_jev.py --dry-run                 # the estimate only (no key read, no spend)
    artifacts/scratch/v08b/cpu.sh jev-pilot scripts/v08_jev.py --limit 2               # a pilot
    artifacts/scratch/v08b/cpu.sh jev-run scripts/v08_jev.py                           # the rest (cached requests cost nothing again)

The set (``artifacts/v08/jev/jev-set.jsonl``, NOT sealed — built for this; disjoint from B8, M8g, M1g and the rel07 check) goes out through
:mod:`robo_jev.bench.openrouter` — retries, a disk cache, a money ledger that refuses a request that could pass the cap **before** it is sent,
the key read only from ``~/.config/robo-jev/openrouter.env`` and never printed. Every request carries ``provider.data_collection = "deny"``.
Only each record's target choice questions are sent (``meta.target``; the agreement reading is on them; the API answers questions
independently). The script refuses to send any record of a sealed or never-sent set (:func:`robo_jev.data.sealed.sealed_record_set`).
Writes ``jev-answers.predictions.jsonl`` (the diagnosis prediction format, :func:`robo_jev.bench.openrouter.to_prediction`) and
``jev-answers.json`` (spend, resolved model, providers, errors) next to the set.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

SCRIPT_VERSION = "v08-jev-1"
MODEL = "~typesafe/jev-latest"
PROVIDER = {"data_collection": "deny"}
CAP_USD = 10.0
#: A conservative per-question overhead for the reservation only (scripts/bench.py fits 98 / 78 tokens from the documented example).
OVERHEAD = {"per_question_low": 100.0, "per_question_high": 400.0}


def _log(message: str) -> None:
    print(f"[v08-jev {dt.datetime.now().strftime('%H:%M:%S')}] {message}", file=sys.stderr, flush=True)


def refuse_sealed(records: list[dict[str, Any]]) -> None:
    from robo_jev.data.sealed import NEVER_SENT_SETS, sealed_record_set

    bad = [(r.get("id"), s) for r in records if (s := sealed_record_set(r))]
    never = [(r.get("id"), (r.get("meta") or {}).get("set")) for r in records if (r.get("meta") or {}).get("set") in NEVER_SENT_SETS]
    if bad or never:
        raise SystemExit(f"refusing: records of sealed / never-sent sets would be sent: {(bad + never)[:5]}")


def jobs_for(records: list[dict[str, Any]], price: dict[str, float], tokenizer: Any) -> list[Any]:
    from robo_jev.bench.openrouter import Job, api_request, estimate_request, request_sha256

    out = []
    for record in records:
        target = list((record.get("meta") or {}).get("target") or [])
        payload = api_request(record, MODEL, levels=2, only=target or None, provider=PROVIDER)
        estimate = estimate_request(payload, tokenizer=tokenizer, overhead=OVERHEAD)
        out.append(Job(record=record, payload=payload, sha=request_sha256(payload), estimate_usd=estimate["high"] * price["prompt"], levels=2))
    return out


def main(argv: list[str] | None = None) -> int:
    from robo_jev.bench.gates_v08 import JEV_DIR

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input", type=Path, default=JEV_DIR / "jev-set.jsonl")
    parser.add_argument("--out-dir", type=Path, default=JEV_DIR)
    parser.add_argument("--limit", type=int, default=None, help="only the first K records (pilot)")
    parser.add_argument("--dry-run", action="store_true", help="estimate only: no key is read and nothing is sent")
    parser.add_argument("--concurrency", type=int, default=4)
    args = parser.parse_args(argv)
    from robo_jev.bench.openrouter import DecisionsClient, Ledger, ResponseCache, fetch_models, load_api_key, price_of, run_requests
    from robo_jev.decisions.records import file_sha256, read_records
    from robo_jev.model.tokenizer import available_tokenizer, load_tokenizer

    records = read_records(args.input)
    refuse_sealed(records)
    chosen = records[: args.limit] if args.limit else records
    catalog = fetch_models()
    entry = next((m for m in catalog.get("data") or [] if m.get("id") == MODEL), None)
    if entry is None:
        raise SystemExit(f"{MODEL} is not in the public Decisions model list")
    price = price_of(entry)
    jobs = jobs_for(chosen, price, load_tokenizer(available_tokenizer()[0]))
    estimate = sum(j.estimate_usd for j in jobs)
    _log(f"{len(jobs)} requests to {MODEL} (prompt price {price['prompt'] * 1e6:.4f} $/M); high estimate ${estimate:.4f}; cap ${CAP_USD}")
    if args.dry_run:
        return 0
    ledger = Ledger(args.out_dir / "ledger.json", max_total_usd=CAP_USD, max_model_usd=CAP_USD)
    if ledger.spent() + estimate * 2.0 > CAP_USD:
        raise SystemExit(f"spent ${ledger.spent():.4f} + this run's reservation could pass the ${CAP_USD} cap — not sending")
    client = DecisionsClient(load_api_key())
    result = run_requests(MODEL, jobs, client=client, cache=ResponseCache(args.out_dir / "cache"), ledger=ledger, price=price,
                          concurrency=args.concurrency, calls_log=args.out_dir / "calls.jsonl", log=_log)
    ledger.save()
    predictions = result["predictions"]
    path = args.out_dir / "jev-answers.predictions.jsonl"
    with path.open("w", encoding="utf-8") as handle:  # every answer so far, in the set's order (cached ones included)
        for record in records:
            if str(record["id"]) in predictions:
                handle.write(json.dumps(predictions[str(record["id"])], ensure_ascii=False) + "\n")
    resolved = sorted({str((p.get("api") or {}).get("model")) for p in predictions.values()})
    providers = sorted({str((p.get("api") or {}).get("provider")) for p in predictions.values()})
    summary = {"kind": "v08-jev-answers", "script_version": SCRIPT_VERSION, "model": MODEL, "resolved_models": resolved, "providers": providers,
               "provider_request": PROVIDER, "input": str(args.input), "input_sha256": file_sha256(args.input), "records": len(records),
               "sent_this_run": len(chosen), "answered": len(predictions), "stats": result["stats"], "errors": result["errors"],
               "skipped": result["skipped"], "problems": result["problems"], "ledger_total_usd": ledger.spent(), "cap_usd": CAP_USD,
               "questions_sent": "each record's target choice questions (meta.target)", "predictions": str(path),
               "created": dt.datetime.now().astimezone().isoformat(timespec="seconds")}
    (args.out_dir / "jev-answers.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    _log(f"answered {len(predictions)}/{len(chosen)}; spent so far ${ledger.spent():.5f}; resolved {resolved}; providers {providers}; "
         f"errors {len(result['errors'])}")
    return 0 if not result["errors"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
