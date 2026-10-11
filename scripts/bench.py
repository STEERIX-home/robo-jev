#!/usr/bin/env python
"""Task bench0 — 우리 판단 모델과 OpenRouter 결정 모델의 비교 벤치마크 bench-v0 (정의 configs/bench/bench-v0.yaml).

    # 정의대로 레코드를 다시 뽑아 artifacts/bench/v0/에 쓴다 (원본 sha256·레코드 수·질문 수·id sha256을 정의의 expect와 대조)
    uv run python scripts/bench.py build
    # 결정 모델 목록 (공개 GET, 키 없음) → 스냅숏 artifacts/bench/v0/models-<시각>.json
    uv run python scripts/bench.py models
    # 견적 (돈을 쓰지 않는다): 모델별 요청 수·입력 토큰(낮은·높은 끝)·비용, 시운전 몫, 전체
    uv run python scripts/bench.py dry-run [--snapshot artifacts/scratch/openrouter-decisions-models-2026-10-12.json]
    # (유료 — 승인 뒤 통제자만) score 형식 시운전: 벤치 밖 T1 100개, 단계 둘 그대로 대 11단계, score 질문만
    uv run python scripts/bench.py pilot --max-cost-usd 5 --max-cost-per-model-usd 0.5
    # (유료 — 승인 뒤 통제자만) 본 실행: 세트 × 모델, 캐시에 있으면 다시 보내지 않는다
    uv run python scripts/bench.py run --max-cost-usd 5 --max-cost-per-model-usd 1.5
    # 점수 (CPU): 우리 모델의 이미 있는 예측에서 벤치 부분집합을 골라(artifacts/bench/v0/ours/) API 결과와 같은 지표로 → results.json·.md
    uv run python scripts/bench.py score
    # 그래프·점수 장치 시험용 가짜 API 예측 (실제 결과로 쓰지 않는다) → 점수 → 미리보기 그래프
    uv run python scripts/bench.py fake-api && uv run python scripts/bench.py score --api-dir artifacts/bench/v0/fake-api --results artifacts/bench/v0/results-fake.json

정의의 규칙은 :mod:`robo_jev.bench.define`, 요청 변환·재시도·캐시·장부·상한은 :mod:`robo_jev.bench.openrouter` 의 모듈 설명이다.
상한(``--max-cost-usd``·``--max-cost-per-model-usd``)은 기본값이 없고 반드시 준다 — 누적 장부(artifacts/bench/openrouter-ledger.json)의
합에 걸린다(시운전에 쓴 돈도 센다). 키는 ~/.config/robo-jev/openrouter.env에서만 읽고 찍지 않는다.
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

SCRIPT_VERSION = "bench0-1"
DEFAULT_CONFIG = REPO / "configs" / "bench" / "bench-v0.yaml"
DEFAULT_OUT = REPO / "artifacts" / "bench" / "v0"
DEFAULT_LEDGER = REPO / "artifacts" / "bench" / "openrouter-ledger.json"
DEFAULT_CACHE = REPO / "artifacts" / "bench" / "cache" / "openrouter"
DEFAULT_CALLS = REPO / "artifacts" / "bench" / "openrouter-calls.jsonl"
#: 질문마다 공급자 프롬프트가 덧붙는 몫(Qwen 토큰) — 문서의 예(typesafe/jev-1.13, 질문 셋, 입력 476토큰)에 맞춘 값: 상태를 한 번 세면
#: 질문마다 +98, 질문마다 세면 +78. 통제자의 무료 모델 표본(inception, 질문 셋, 96토큰)은 덧붙는 몫이 거의 없다 — 공급자마다 다르다.
#: 그래서 견적은 두 끝(낮은 = 상태 한 번 + 98/질문, 높은 = 상태 질문마다 + 78/질문)을 내고 상한은 높은 끝으로 잡는다.
OVERHEAD = {"per_question_low": 98.0, "per_question_high": 78.0}


def _log(message: str) -> None:
    print(f"[bench {dt.datetime.now().strftime('%H:%M:%S')}] {message}", file=sys.stderr, flush=True)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(payload, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    temp.replace(path)


def cmd_build(args: argparse.Namespace) -> int:
    from robo_jev.bench.define import build, load_config

    config = load_config(args.config)
    manifest = build(config, root=args.root, out_dir=args.out, check=not args.no_check)
    manifest["config"] = str(args.config)
    manifest["script_version"] = SCRIPT_VERSION
    manifest["created"] = dt.datetime.now().astimezone().isoformat(timespec="seconds")
    _write_json(args.out / "manifest.json", manifest)
    for name, entry in manifest["sets"].items():
        _log(f"{name}: {entry['records']} records · {entry['questions']['total']} questions · ids {entry['ids_sha256'][:12]}")
    if manifest.get("pilot"):
        pilot = manifest["pilot"]
        _log(f"pilot ({pilot['from']}, not in the bench): {pilot['records']} records · {pilot['questions']['total']} questions")
    _log(f"→ {args.out}")
    return 0


def _load_bench(config: dict[str, Any], out: Path, sets: list[str] | None = None) -> dict[str, list[dict[str, Any]]]:
    """벤치 파일(build가 쓴 것)을 읽고 id sha256을 정의의 expect와 대조한다 — 다르면 거절(다른 레코드로 돈을 쓰지 않게)."""
    from robo_jev.bench.define import ids_sha256
    from robo_jev.decisions.records import read_records

    out_sets = {}
    for name in sets or list(config["sets"]):
        if name not in config["sets"]:
            raise SystemExit(f"모르는 세트 {name!r} (정의: {list(config['sets'])})")
        path = out / f"{name}.jsonl"
        if not path.is_file():
            raise SystemExit(f"{path}: 없다 — 먼저 `scripts/bench.py build`")
        records = read_records(path)
        expect = (config["sets"][name].get("expect") or {}).get("ids_sha256")
        if expect and ids_sha256(str(r["id"]) for r in records) != expect:
            raise SystemExit(f"{path}: id 목록이 정의와 다르다 — `scripts/bench.py build`로 다시 만든다")
        out_sets[name] = records
    return out_sets


def _load_pilot(config: dict[str, Any], out: Path) -> list[dict[str, Any]]:
    from robo_jev.bench.define import ids_sha256
    from robo_jev.decisions.records import read_records

    pilot = config.get("pilot") or {}
    path = out / f"pilot-{pilot.get('from', 'T1')}.jsonl"
    if not path.is_file():
        raise SystemExit(f"{path}: 없다 — 먼저 `scripts/bench.py build`")
    records = read_records(path)
    expect = (pilot.get("expect") or {}).get("ids_sha256")
    if expect and ids_sha256(str(r["id"]) for r in records) != expect:
        raise SystemExit(f"{path}: id 목록이 정의와 다르다")
    return records


def _catalog(args: argparse.Namespace) -> tuple[dict[str, Any], str]:
    """모델 목록 — `--snapshot`이 있으면 그 파일, 없으면 공개 GET(키 없음)으로 받아 스냅숏을 남긴다."""
    from robo_jev.bench.openrouter import fetch_models

    if getattr(args, "snapshot", None):
        return json.loads(Path(args.snapshot).read_text(encoding="utf-8")), str(args.snapshot)
    catalog = fetch_models()
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = args.out / f"models-{stamp}.json"
    _write_json(path, catalog)
    return catalog, str(path)


def _models(args: argparse.Namespace, config: dict[str, Any], catalog: dict[str, Any]) -> list[dict[str, Any]]:
    from robo_jev.bench.openrouter import select_models

    suffixes = tuple((config.get("models") or {}).get("exclude_suffixes") or (":free",))
    if getattr(args, "all_listed", False):
        return select_models(catalog, None, exclude_suffixes=suffixes)
    wanted = args.models or [m["id"] for m in (config.get("models") or {}).get("default") or []]
    return select_models(catalog, wanted, exclude_suffixes=suffixes)


def _tokenizer() -> tuple[Any | None, str]:
    try:
        from robo_jev.model.tokenizer import available_tokenizer, load_tokenizer

        found = available_tokenizer()
        if found:
            return load_tokenizer(found[0]), f"Qwen tokenizer ({found[0]})"
    except Exception as error:  # noqa: BLE001 — 견적은 글자 / 4로 물러난다
        _log(f"tokenizer unavailable ({type(error).__name__}); using chars/4")
    return None, "chars/4"


def _score_only(record: dict[str, Any]) -> list[str]:
    return [name for name, q in record["request"]["questions"].items() if q.get("type") == "score"]


def _estimates(records: list[dict[str, Any]], tokenizer: Any, *, levels: int = 11, score_only: bool = False) -> list[dict[str, Any]]:
    from robo_jev.bench.openrouter import api_request, estimate_request

    out = []
    for record in records:
        only = _score_only(record) if score_only else None
        if only is not None and not only:
            continue
        payload = api_request(record, "dry-run", levels=levels, only=only)
        out.append(estimate_request(payload, tokenizer=tokenizer, overhead=OVERHEAD))
    return out


def cmd_models(args: argparse.Namespace) -> int:
    from robo_jev.bench.define import load_config
    from robo_jev.bench.openrouter import price_of

    config = load_config(args.config)
    catalog, source = _catalog(args)
    default = {m["id"] for m in (config.get("models") or {}).get("default") or []}
    for entry in catalog.get("data") or []:
        price = price_of(entry)
        mark = "*" if entry["id"] in default else " "
        print(f"{mark} {entry['id']:45s} ${price['prompt'] * 1e6:7.3f}/M input  context {entry.get('context_length')}")
    _log(f"{len(catalog.get('data') or [])} decision models listed (* = bench-v0 default) — snapshot {source}")
    return 0


def cmd_dry_run(args: argparse.Namespace) -> int:
    """견적만 — 키를 읽지 않고 돈을 쓰지 않는다."""
    from robo_jev.bench.define import load_config
    from robo_jev.bench.openrouter import price_of

    config = load_config(args.config)
    bench = _load_bench(config, args.out, args.sets)
    pilot = _load_pilot(config, args.out)
    catalog, source = _catalog(args)
    models = _models(args, config, catalog)
    tokenizer, counter = _tokenizer()
    per_set = {name: _estimates(records, tokenizer) for name, records in bench.items()}
    pilot_est = _estimates(pilot, tokenizer, levels=11, score_only=True) + _estimates(pilot, tokenizer, levels=2, score_only=True)
    requests = sum(len(v) for v in per_set.values())
    questions = sum(e["questions"] for v in per_set.values() for e in v)
    low = sum(e["low"] for v in per_set.values() for e in v)
    high = sum(e["high"] for v in per_set.values() for e in v)
    p_low, p_high = sum(e["low"] for e in pilot_est), sum(e["high"] for e in pilot_est)
    biggest = max(e["high"] for v in per_set.values() for e in v)
    rows, total = [], {"bench_low": 0.0, "bench_high": 0.0, "pilot_low": 0.0, "pilot_high": 0.0}
    for entry in models:
        price = price_of(entry)
        context = int(entry.get("context_length") or 0)
        row = {
            "model": entry["id"], "price_per_m_input": price["prompt"] * 1e6, "context_length": context,
            "requests": requests, "questions": questions, "tokens_low": low, "tokens_high": high,
            "cost_low": low * price["prompt"], "cost_high": high * price["prompt"],
            "pilot_requests": len(pilot_est), "pilot_cost_low": p_low * price["prompt"], "pilot_cost_high": p_high * price["prompt"],
            "largest_request_tokens_high": biggest, "fits_context": (context == 0) or biggest < context,
        }
        rows.append(row)
        for key in total:
            total[key] += row[{"bench_low": "cost_low", "bench_high": "cost_high", "pilot_low": "pilot_cost_low", "pilot_high": "pilot_cost_high"}[key]]
    report = {
        "kind": "bench-v0-dry-run", "created": dt.datetime.now().astimezone().isoformat(timespec="seconds"), "script_version": SCRIPT_VERSION,
        "catalog": source, "token_counter": counter, "overhead": OVERHEAD, "score_levels": 11,
        "sets": {name: {"requests": len(v), "questions": sum(e["questions"] for e in v), "tokens_low": sum(e["low"] for e in v),
                        "tokens_high": sum(e["high"] for e in v)} for name, v in per_set.items()},
        "pilot": {"records": len(pilot), "requests_per_model": len(pilot_est), "tokens_low": p_low, "tokens_high": p_high},
        "models": rows, "total": {**total, "all_low": total["bench_low"] + total["pilot_low"], "all_high": total["bench_high"] + total["pilot_high"]},
        "note": "estimate only — no request was sent; output tokens are free for every listed decision model (completion price 0)",
    }
    _write_json(args.out / "dry-run.json", report)
    print(f"bench-v0 dry-run — {requests} requests · {questions} questions per model; tokens {counter}; catalog {source}")
    print(f"{'model':42s} {'$/M in':>7s} {'requests':>8s} {'tokens (low–high)':>22s} {'bench $ (low–high)':>20s} {'pilot $':>16s} {'ctx ok':>6s}")
    for row in rows:
        print(f"{row['model']:42s} {row['price_per_m_input']:7.3f} {row['requests']:8d} {row['tokens_low']:>10,d}–{row['tokens_high']:<11,d} "
              f"{row['cost_low']:>9.4f}–{row['cost_high']:<9.4f} {row['pilot_cost_low']:>7.4f}–{row['pilot_cost_high']:<7.4f} {str(row['fits_context']):>6s}")
    t = report["total"]
    print(f"{'TOTAL':42s} {'':7s} {requests * len(rows):8d} {'':22s} {t['bench_low']:>9.4f}–{t['bench_high']:<9.4f} {t['pilot_low']:>7.4f}–{t['pilot_high']:<7.4f}")
    print(f"all (bench + pilot): ${t['all_low']:.4f} – ${t['all_high']:.4f} → {args.out / 'dry-run.json'}")
    return 0


def _write_predictions(path: Path, records: list[dict[str, Any]], predictions: dict[str, Any]) -> int:
    """예측을 벤치 레코드 순서로 쓴다 (답이 있는 레코드만) — 진단 도구의 .predictions.jsonl 꼴."""
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    temp = path.with_name(path.name + ".tmp")
    with temp.open("w", encoding="utf-8") as handle:
        for record in records:
            prediction = predictions.get(str(record["id"]))
            if prediction:
                handle.write(json.dumps(prediction, ensure_ascii=False) + "\n")
                count += 1
    temp.replace(path)
    return count


def _paid_setup(args: argparse.Namespace, config: dict[str, Any]) -> tuple[Any, Any, Any, list[dict[str, Any]], str, Any]:
    """유료 경로의 준비: 모델 목록(공개 GET + 스냅숏), 키(파일에서만), 클라이언트, 누적 장부와 상한, 캐시."""
    from robo_jev.bench.openrouter import DecisionsClient, Ledger, ResponseCache, load_api_key

    catalog, source = _catalog(args)
    models = _models(args, config, catalog)
    ledger = Ledger(args.ledger, max_total_usd=args.max_cost_usd, max_model_usd=args.max_cost_per_model_usd)
    if ledger.spent() >= args.max_cost_usd:
        raise SystemExit(f"the ledger already holds ${ledger.spent():.4f} ≥ the cap ${args.max_cost_usd:.4f} — nothing sent")
    key = load_api_key(args.key_file)
    client = DecisionsClient(key, url=args.api_url, timeout=args.timeout, max_attempts=args.attempts)
    _log(f"models {[m['id'] for m in models]} (catalog {source}); ledger so far ${ledger.spent():.4f}; caps total ${args.max_cost_usd} · "
         f"per model ${args.max_cost_per_model_usd}")
    return client, ledger, ResponseCache(args.cache), models, source, catalog


def _jobs(records: list[dict[str, Any]], model: str, price: dict[str, float], tokenizer: Any, *, levels: int,
          provider: dict[str, Any] | None, score_only: bool = False) -> list[Any]:
    from robo_jev.bench.openrouter import Job, api_request, estimate_request, request_sha256

    jobs = []
    for record in records:
        only = _score_only(record) if score_only else None
        if only is not None and not only:
            continue
        payload = api_request(record, model, levels=levels, only=only, provider=provider)
        estimate = estimate_request(payload, tokenizer=tokenizer, overhead=OVERHEAD)
        jobs.append(Job(record=record, payload=payload, sha=request_sha256(payload), estimate_usd=estimate["high"] * price["prompt"], levels=levels))
    return jobs


def _run_entry(args: argparse.Namespace, kind: str, models: list[dict[str, Any]], source: str) -> dict[str, Any]:
    return {"kind": kind, "started": dt.datetime.now().astimezone().isoformat(timespec="seconds"), "script_version": SCRIPT_VERSION,
            "models": [m["id"] for m in models], "catalog": source, "caps": {"total_usd": args.max_cost_usd, "per_model_usd": args.max_cost_per_model_usd},
            "concurrency": args.concurrency, "data_collection": args.data_collection}


def cmd_run(args: argparse.Namespace) -> int:
    """(유료) 본 실행 — 세트 × 모델. 캐시에 있으면 보내지 않는다. 상한에 닿으면 멈추고, 다시 돌리면 이어서 한다."""
    import threading

    from robo_jev.bench.define import load_config
    from robo_jev.bench.openrouter import price_of, run_requests, safe_name

    config = load_config(args.config)
    bench = _load_bench(config, args.out, args.sets)
    client, ledger, cache, models, source, _ = _paid_setup(args, config)
    tokenizer, _ = _tokenizer()
    provider = {"data_collection": args.data_collection} if args.data_collection else None
    entry = _run_entry(args, "run", models, source)
    entry["sets"] = list(bench)
    spent_before = ledger.spent()
    stop = threading.Event()
    summary: dict[str, Any] = {}
    for set_name, records in bench.items():
        for model_entry in models:
            model, price = model_entry["id"], price_of(model_entry)
            jobs = _jobs(records, model, price, tokenizer, levels=int((config.get("request") or {}).get("score_levels", 11)), provider=provider)
            out = run_requests(model, jobs, client=client, cache=cache, ledger=ledger, price=price, concurrency=args.concurrency,
                               calls_log=args.calls_log, log=lambda m, s=set_name: _log(f"{s} · {m}"), stop=stop)
            target = args.out / "api" / safe_name(model) / f"{set_name}.predictions.jsonl"
            written = _write_predictions(target, records, out["predictions"])
            info = {"model": model, "set": set_name, "records": len(records), "written": written, "stats": out["stats"],
                    "errors": out["errors"], "skipped": out["skipped"], "problems": out["problems"],
                    "resolved_models": sorted({p["api"]["model"] for p in out["predictions"].values() if p["api"].get("model")}),
                    "providers": sorted({p["api"]["provider"] for p in out["predictions"].values() if p["api"].get("provider")})}
            _write_json(target.with_name(f"{set_name}.run.json"), info)
            summary.setdefault(model, {})[set_name] = {"written": written, "of": len(records), "errors": len(out["errors"]), "skipped": len(out["skipped"])}
            if stop.is_set():
                break
        if stop.is_set():
            _log("stopped (total cap or a fatal error) — rerun with the same command to continue from the cache")
            break
    entry.update({"finished": dt.datetime.now().astimezone().isoformat(timespec="seconds"), "spent_usd": ledger.spent() - spent_before,
                  "stopped": stop.is_set(), "summary": summary})
    ledger.runs.append(entry)
    ledger.save()
    _log(f"run spent ${ledger.spent() - spent_before:.5f} (ledger ${ledger.spent():.5f}) → {args.out / 'api'}")
    return 0 if not stop.is_set() else 3


def cmd_pilot(args: argparse.Namespace) -> int:
    """(유료) score 형식 시운전 — 벤치 밖 T1 100개, 단계 둘 그대로 대 11단계, score 질문만 보낸다. 보고서는 같은 지표로."""
    import threading

    from robo_jev.bench.define import load_config
    from robo_jev.bench.openrouter import estimate_request, price_of, run_requests, safe_name
    from robo_jev.bench.score import pilot_report, render_pilot

    config = load_config(args.config)
    records = _load_pilot(config, args.out)
    bench = _load_bench(config, args.out)
    client, ledger, cache, models, source, _ = _paid_setup(args, config)
    tokenizer, counter = _tokenizer()
    provider = {"data_collection": args.data_collection} if args.data_collection else None
    entry = _run_entry(args, "pilot", models, source)
    spent_before = ledger.spent()
    stop = threading.Event()
    bench_tokens = {"low": sum(e["low"] for v in bench.values() for e in _estimates(v, tokenizer)),
                    "high": sum(e["high"] for v in bench.values() for e in _estimates(v, tokenizer))}
    reports = {}
    for model_entry in models:
        model, price = model_entry["id"], price_of(model_entry)
        runs = {}
        for levels in (2, 11):
            jobs = _jobs(records, model, price, tokenizer, levels=levels, provider=provider, score_only=True)
            out = run_requests(model, jobs, client=client, cache=cache, ledger=ledger, price=price, concurrency=args.concurrency,
                               calls_log=args.calls_log, log=lambda m, n=levels: _log(f"pilot score{n} · {m}"), stop=stop)
            target = args.out / "pilot" / safe_name(model) / f"score{levels}.predictions.jsonl"
            _write_predictions(target, records, out["predictions"])
            estimates = {str(j.record["id"]): estimate_request(j.payload, tokenizer=tokenizer, overhead=OVERHEAD) for j in jobs}
            runs[levels] = {"predictions": [out["predictions"][str(r["id"])] for r in records if str(r["id"]) in out["predictions"]],
                            "estimates": estimates, "errors": out["errors"], "skipped": out["skipped"]}
            if stop.is_set():
                break
        if 2 in runs and 11 in runs:
            reports[model] = pilot_report(records, runs[11]["predictions"], runs[2]["predictions"], estimates={2: runs[2]["estimates"], 11: runs[11]["estimates"]},
                                          bench_tokens=bench_tokens, price=price)
            reports[model]["errors"] = {"score2": len(runs[2]["errors"]), "score11": len(runs[11]["errors"])}
        if stop.is_set():
            break
    payload = {"kind": "bench-v0-pilot", "created": dt.datetime.now().astimezone().isoformat(timespec="seconds"), "records": len(records),
               "token_counter": counter, "overhead": OVERHEAD, "models": reports, "stopped": stop.is_set()}
    _write_json(args.out / "pilot" / "report.json", payload)
    (args.out / "pilot" / "report.md").write_text(render_pilot(payload), encoding="utf-8")
    entry.update({"finished": dt.datetime.now().astimezone().isoformat(timespec="seconds"), "spent_usd": ledger.spent() - spent_before, "stopped": stop.is_set()})
    ledger.runs.append(entry)
    ledger.save()
    _log(f"pilot spent ${ledger.spent() - spent_before:.5f} (ledger ${ledger.spent():.5f}) → {args.out / 'pilot' / 'report.md'}")
    return 0 if not stop.is_set() else 3


def _latest_snapshot(out: Path) -> Path | None:
    found = sorted(out.glob("models-*.json"))
    if found:
        return found[-1]
    fallback = REPO / "artifacts" / "scratch" / "openrouter-decisions-models-2026-10-12.json"
    return fallback if fallback.is_file() else None


def cmd_score(args: argparse.Namespace) -> int:
    """우리 모델(이미 있는 예측의 벤치 부분집합)과 API 모델(run이 쓴 예측)을 같은 지표로 → results.json + results.md(README 표)."""
    from robo_jev.bench.define import load_config
    from robo_jev.bench.openrouter import price_of
    from robo_jev.bench.score import describe_source, pooled_brier, readme_table, resolve_ours, set_metrics, subset_predictions
    from robo_jev.decisions.records import read_records

    config = load_config(args.config)
    bench = _load_bench(config, args.out, args.sets)
    snapshot = args.snapshot or _latest_snapshot(args.out)
    prices = {e["id"]: price_of(e)["prompt"] * 1e6 for e in (json.loads(Path(snapshot).read_text(encoding="utf-8")).get("data") or [])} if snapshot else {}
    models: list[dict[str, Any]] = []
    for entry in config.get("ours") or []:
        model = {"tag": entry["tag"], "label": entry["label"], "role": entry.get("role", "ours"), "model_id": None, "price_per_m_input": None,
                 "sources": {}, "sets": {}}
        for set_name, records in bench.items():
            source = resolve_ours(entry, set_name, root=args.root)
            if source is None:
                continue
            predictions = subset_predictions(source, [r["id"] for r in records])
            target = args.out / "ours" / entry["tag"] / f"{set_name}.predictions.jsonl"
            _write_predictions(target, records, {str(p["id"]): p for p in predictions})
            model["sources"][set_name] = {**describe_source(source), "records": len(predictions), "subset": str(target)}
            model["sets"][set_name] = set_metrics(set_name, records, predictions, n_boot=args.n_boot)
            _log(f"{entry['tag']} · {set_name}: {len(predictions)}/{len(records)} records from {source}")
        if model["sets"]:
            models.append(model)
        else:
            _log(f"{entry['tag']}: no predictions yet — skipped (rerun `score` when they arrive)")
    api_dir = args.api_dir or (args.out / "api")
    for directory in sorted(p for p in api_dir.glob("*") if p.is_dir()) if api_dir.is_dir() else []:
        model: dict[str, Any] = {"tag": directory.name, "label": None, "role": "api", "model_id": None, "sources": {}, "sets": {}}
        for set_name, records in bench.items():
            path = directory / f"{set_name}.predictions.jsonl"
            if not path.is_file():
                continue
            predictions = read_records(path)
            model["model_id"] = model["model_id"] or predictions[0]["model"] if predictions else model["model_id"]
            model["sources"][set_name] = {**describe_source(path), "records": len(predictions),
                                          "resolved_models": sorted({(p.get("api") or {}).get("model") for p in predictions if (p.get("api") or {}).get("model")})}
            model["sets"][set_name] = set_metrics(set_name, records, predictions, n_boot=args.n_boot)
            _log(f"api {directory.name} · {set_name}: {len(predictions)}/{len(records)} records")
        if model["sets"]:
            model_id = str(model["model_id"])
            model["fake"] = model_id.startswith("FAKE ")
            plain = model_id.removeprefix("FAKE ")
            model["label"] = model_id
            model["price_per_m_input"] = prices.get(plain)
            models.append(model)
    for model in models:
        model["pooled"] = {"unseen": pooled_brier(model["sets"]), "unseen_noul": pooled_brier(model["sets"], blocks=("noul",)),
                           "unseen_score": pooled_brier(model["sets"], blocks=("score",)),
                           "all_physics": pooled_brier(model["sets"], sets=[s for s in ("T1", "T2", "T3", "T4", "T5", "T6", "T7") if s in model["sets"]])}
    date = dt.date.today().isoformat()
    results = {"kind": "bench-v0-results", "created": dt.datetime.now().astimezone().isoformat(timespec="seconds"), "script_version": SCRIPT_VERSION,
               "bench": config["version"], "date": date, "catalog": str(snapshot) if snapshot else None, "n_boot": args.n_boot,
               "fake": any(m.get("fake") for m in models), "models": models}
    path = args.results or (args.out / "results.json")
    _write_json(path, results)
    path.with_suffix(".md").write_text(readme_table(results, date=date), encoding="utf-8")
    _log(f"{len(models)} models scored → {path} (+ .md)")
    return 0


def cmd_fake_api(args: argparse.Namespace) -> int:
    """그래프·점수 장치 시험용 **가짜** API 예측 — 정의의 기본 모델마다 다른 세기의 잡음 답. 실제 결과로 쓰지 않는다."""
    from robo_jev.bench.define import load_config
    from robo_jev.bench.openrouter import safe_name
    from robo_jev.bench.score import fake_api_predictions

    config = load_config(args.config)
    bench = _load_bench(config, args.out)
    models = [m["id"] for m in (config.get("models") or {}).get("default") or []]
    for index, model in enumerate(models):
        strength = 0.15 + 0.12 * index
        for set_name, records in bench.items():
            predictions = fake_api_predictions(records, model, strength=strength, seed=index, coverage=0.98)
            _write_predictions(args.fake_dir / safe_name(model) / f"{set_name}.predictions.jsonl", records, {p["id"]: p for p in predictions})
    _log(f"FAKE predictions for {len(models)} models → {args.fake_dir} (pipeline test only)")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--root", type=Path, default=REPO, help="정의의 상대 경로가 기대는 뿌리 (기본: 저장소)")
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build", help="정의대로 레코드를 다시 뽑아 쓴다 (expect와 대조)")
    build.add_argument("--out", type=Path, default=DEFAULT_OUT)
    build.add_argument("--no-check", action="store_true", help="expect와 대조하지 않는다 (정의를 처음 만들 때만)")
    build.set_defaults(func=cmd_build)
    for name, func, text in (("models", cmd_models, "결정 모델 목록 (공개 GET) → 스냅숏"), ("dry-run", cmd_dry_run, "견적 (돈을 쓰지 않는다)")):
        command = sub.add_parser(name, help=text)
        command.add_argument("--out", type=Path, default=DEFAULT_OUT)
        command.add_argument("--snapshot", type=Path, default=None, help="모델 목록 스냅숏 파일 (없으면 공개 GET으로 받아 남긴다)")
        command.add_argument("--models", nargs="+", default=None, help="모델 id들 (기본: 정의의 models.default)")
        command.add_argument("--all-listed", action="store_true", help="목록의 모든 모델 (:free 제외)")
        command.add_argument("--sets", nargs="+", default=None, help="세트들 (기본: 전부)")
        command.set_defaults(func=func)
    from robo_jev.bench.openrouter import API_URL, DEFAULT_ATTEMPTS, DEFAULT_CONCURRENCY, DEFAULT_KEY_FILE, DEFAULT_TIMEOUT

    for name, func, text in (("run", cmd_run, "(유료) 본 실행"), ("pilot", cmd_pilot, "(유료) score 형식 시운전")):
        command = sub.add_parser(name, help=text)
        command.add_argument("--max-cost-usd", type=float, required=True, help="전체 상한 (누적 장부 기준, 기본값 없음)")
        command.add_argument("--max-cost-per-model-usd", type=float, required=True, help="모델별 상한 (누적 장부 기준, 기본값 없음)")
        command.add_argument("--out", type=Path, default=DEFAULT_OUT)
        command.add_argument("--snapshot", type=Path, default=None, help="모델 목록 스냅숏 (없으면 공개 GET으로 받아 남긴다)")
        command.add_argument("--models", nargs="+", default=None, help="모델 id들 (기본: 정의의 models.default)")
        command.add_argument("--all-listed", action="store_true", help="목록의 모든 모델 (:free 제외)")
        command.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY)
        command.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT, help="요청 하나의 시간 초과 (초)")
        command.add_argument("--attempts", type=int, default=DEFAULT_ATTEMPTS, help="요청 하나의 최대 시도 수")
        command.add_argument("--data-collection", choices=("allow", "deny"), default=None,
                             help="provider.data_collection (기본: 보내지 않음 = OpenRouter 기본 allow). deny면 봉인 세트를 저장·학습하는 공급자를 피한다")
        command.add_argument("--key-file", type=Path, default=DEFAULT_KEY_FILE)
        command.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)
        command.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
        command.add_argument("--calls-log", type=Path, default=DEFAULT_CALLS)
        command.add_argument("--api-url", default=API_URL, help=argparse.SUPPRESS)  # 검사(가짜 서버) 전용
        if name == "run":
            command.add_argument("--sets", nargs="+", default=None, help="세트들 (기본: 전부)")
        command.set_defaults(func=func)
    score = sub.add_parser("score", help="점수표 (우리 모델 부분집합 + API 결과) → results.json·.md")
    score.add_argument("--out", type=Path, default=DEFAULT_OUT)
    score.add_argument("--api-dir", type=Path, default=None, help="API 예측 디렉터리 (기본: <out>/api)")
    score.add_argument("--results", type=Path, default=None, help="결과 JSON (기본: <out>/results.json; 옆에 .md)")
    score.add_argument("--snapshot", type=Path, default=None, help="가격을 읽을 모델 목록 스냅숏 (기본: <out>/models-*.json의 마지막)")
    score.add_argument("--sets", nargs="+", default=None)
    score.add_argument("--n-boot", type=int, default=1000)
    score.set_defaults(func=cmd_score)
    fake = sub.add_parser("fake-api", help="그래프·점수 장치 시험용 가짜 API 예측")
    fake.add_argument("--out", type=Path, default=DEFAULT_OUT)
    fake.add_argument("--fake-dir", type=Path, default=DEFAULT_OUT / "fake-api")
    fake.set_defaults(func=cmd_fake_api)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
