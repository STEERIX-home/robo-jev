"""폐루프 평가 (Task R4, docs/06 Task 6) — 학습된 모델을 시뮬레이터 루프에 꽂고 10 Hz로 돌린다.

    # A3·A4: 녹화된 dev 에피소드에서 오프라인 재생 평가와 루프 정책의 답이 틱마다 같은지, 그리고 틱 지연
    uv run python scripts/closed_loop.py verify --checkpoint artifacts/runs/r3a-t1-fp32-2b-s18/checkpoint.pt \\
        --episodes ep-E0-400101,ep-E1-410101,ep-E2-420105 --stored-report artifacts/reports/r3a-dev-2b-t1-fp32-s18.json \\
        --out artifacts/reports/r4-a3-s18.json
    # B2: 장면(seed) 목록 — dev 계열 100 seed, ood_dev 계열 26 seed (새 seed, 같은 split 규칙)
    uv run python scripts/closed_loop.py seeds --out artifacts/reports/r4-seeds.json
    # B1: 정책 하나의 run (CPU: rule·mechanical·expert / GPU: model)
    uv run python scripts/closed_loop.py run --policy rule --condition dev,ood_dev --out artifacts/datasets/r4-closed-loop/rule
    uv run python scripts/closed_loop.py run --policy model --checkpoint … --label s18 --condition dev,ood_dev --out …
    # B3·C: 정책·조건별 지표와 seed로 짝지은 구간
    uv run python scripts/closed_loop.py report --runs artifacts/reports/r4-run-*.json --out artifacts/reports/r4-closed-loop.json

GPU 규칙: 울타리 0.6(`robo_jev.gpu`), 적재 전 여유 검사, 한 번에 하나(`systemd-run --user --unit=r4-<step> … choom -n 1000`).
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from robo_jev.gpu import DEFAULT_FRACTION, limit_gpu_memory, memory_report, require_free  # noqa: E402

DEFAULT_MANIFEST = REPO / "artifacts/datasets/r1-robot/r1/manifest.json"
DEFAULT_SEEDS_CONFIG = REPO / "configs/eval/r4-closed-loop.yaml"
#: A3 검증에 쓰는 녹화된 dev 에피소드 — 프로파일마다 하나(E2에는 지시 변경이 여럿이다). 전부 `dev` 분할(둘째 칸)이고 봉인은 없다.
DEFAULT_VERIFY_EPISODES = ("ep-E0-400101", "ep-E1-410101", "ep-E2-420105")
#: 서빙 모델 적재 전에 있어야 하는 장치(통합) 메모리 여유 — checkpoint(24.5 GiB)를 CPU에 통째로 읽은 뒤 backbone bf16 ≈ 4.2 GiB.
LOAD_MIN_FREE_BYTES = 36 * 2**30
SCRIPT_VERSION = "r4-closed-loop-1.0"


def _git_commit() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def _write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=1, default=str) + "\n", encoding="utf-8")
    print(f"→ {path}", flush=True)


def _log(message: str) -> None:
    print(f"[r4 {time.strftime('%H:%M:%S')}] {message}", file=sys.stderr, flush=True)


# --------------------------------------------------------------------------
# verify — A3 (같은 답 증명) + A4 (틱 지연, 재생 기준)
# --------------------------------------------------------------------------


def _read_episode(manifest_path: Path, episode_id: str) -> dict[str, Any]:
    path = manifest_path.parent / "episodes" / episode_id / "streams.jsonl"
    if not path.is_file():
        raise FileNotFoundError(f"에피소드가 없다: {path}")
    return json.loads(path.read_text(encoding="utf-8").splitlines()[0])


def _request_view(tick: dict[str, Any]) -> dict[str, Any]:
    return {key: tick[key] for key in ("t", "sim_ms", "observed_at_ms", "obs_age_ms", "request")}


def _offline_answers(judge: Any, layout: dict[str, Any], *, fused: bool) -> list[dict[str, Any]]:
    """오프라인 재생 평가(`predict_items`와 같은 호출) → 틱마다 ``{qid: {"ids", "probabilities", "argmax"}}``."""
    import torch

    with torch.no_grad():
        result = judge({"layout": "stream_l1a", "stream": layout, "fused": fused})
    out: list[dict[str, Any]] = []
    for logits, candidates in zip(result["logits"], result["candidates"]):
        entry: dict[str, Any] = {}
        for qid, z in logits.items():
            p = torch.softmax(z.detach().float().cpu(), 0)
            ids = list(candidates[qid])
            entry[qid] = {"ids": ids, "probabilities": [float(v) for v in p], "argmax": ids[int(p.argmax())]}
        out.append(entry)
    return out


def _policy_answers(policy: Any, record: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """루프 정책이 같은 틱 입력을 먹었을 때 — 틱마다 같은 꼴 + 지연 행."""
    from robo_jev.contracts import QUESTION_SET_V0

    policy.begin_episode(instructions=record["prefix"]["instructions"])
    out: list[dict[str, Any]] = []
    for tick in record["ticks"]:
        answers = policy.act(_request_view(tick))
        entry: dict[str, Any] = {}
        for qid, answer in answers.items():
            if QUESTION_SET_V0[qid]["type"] == "boolean":
                ids = ["true", "false"]
                probabilities = [float(answer), 1.0 - float(answer)]
            else:
                if not answer:
                    continue
                ids = list(answer)
                probabilities = [float(answer[cid]) for cid in ids]
            entry[qid] = {"ids": ids, "probabilities": probabilities, "argmax": ids[max(range(len(ids)), key=lambda i: probabilities[i])]}
        out.append(entry)
    return out, list(policy.timing)


def _compare(reference: list[dict[str, Any]], candidate: list[dict[str, Any]], *, name: str) -> dict[str, Any]:
    """틱·질문마다 argmax 일치와 확률 차이. 다른 틱은 전부 적는다."""
    agree: dict[str, int] = {}
    total: dict[str, int] = {}
    max_diff: dict[str, float] = {}
    disagreements: list[dict[str, Any]] = []
    for index, (a, b) in enumerate(zip(reference, candidate)):
        for qid, ra in a.items():
            rb = b.get(qid)
            if rb is None:
                continue
            total[qid] = total.get(qid, 0) + 1
            if ra["ids"] != rb["ids"]:
                raise ValueError(f"{name} 틱 {index} {qid}: 후보 순서가 다르다 — 같은 직렬화가 아니다")
            diff = max(abs(x - y) for x, y in zip(ra["probabilities"], rb["probabilities"]))
            max_diff[qid] = max(max_diff.get(qid, 0.0), diff)
            if ra["argmax"] == rb["argmax"]:
                agree[qid] = agree.get(qid, 0) + 1
            else:
                disagreements.append({"tick": index, "question": qid, "reference": ra["argmax"], "candidate": rb["argmax"],
                                      "reference_p": max(ra["probabilities"]), "candidate_p": max(rb["probabilities"]), "max_abs_diff": diff})
    per_question = {qid: {"n": total[qid], "agree": agree.get(qid, 0), "rate": agree.get(qid, 0) / total[qid], "max_abs_prob_diff": max_diff[qid]} for qid in sorted(total)}
    n = sum(total.values())
    return {"name": name, "ticks": len(reference), "questions_compared": n, "agree": sum(agree.values()),
            "agreement_rate": (sum(agree.values()) / n) if n else None, "max_abs_prob_diff": max(max_diff.values()) if max_diff else None,
            "per_question": per_question, "disagreements": disagreements}


def _stored_rows(report_path: Path, split_name: str, episode_id: str) -> dict[str, dict[int, str]]:
    """R3a 평가 산출물의 `per_record`(q_main·q_stop) → {질문: {틱 index: 예측 id}} — 다른 프로세스·동적 KV의 답."""
    report = json.loads(report_path.read_text(encoding="utf-8"))
    table = report["evaluation"]["splits"][split_name]["model"]
    out: dict[str, dict[int, str]] = {}
    for qid in ("q_main", "q_stop"):
        rows = (table.get(qid) or {}).get("per_record") or []
        out[qid] = {int(row["tick"]): str(row["predicted"]) for row in rows if row.get("record_id") == episode_id and row.get("tick") is not None}
    return out


def _compare_stored(stored: dict[str, dict[int, str]], candidate: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for qid, rows in stored.items():
        n = agree = 0
        differing: list[dict[str, Any]] = []
        for index, entry in enumerate(candidate):
            if index not in rows or qid not in entry:
                continue
            n += 1
            if rows[index] == entry[qid]["argmax"]:
                agree += 1
            else:
                differing.append({"tick": index, "stored": rows[index], "policy": entry[qid]["argmax"], "policy_p": max(entry[qid]["probabilities"])})
        out[qid] = {"n": n, "agree": agree, "rate": (agree / n) if n else None, "disagreements": differing}
    return out


def _latency_summary(rows: list[dict[str, Any]], *, key: str) -> dict[str, Any]:
    values = sorted(float(row[key]) for row in rows)
    if not values:
        return {"n": 0}

    def q(fraction: float) -> float:
        position = fraction * (len(values) - 1)
        low = int(position)
        high = min(low + 1, len(values) - 1)
        return values[low] + (values[high] - values[low]) * (position - low)

    return {
        "n": len(values), "mean": statistics.fmean(values), "p50": q(0.5), "p95": q(0.95), "p99": q(0.99), "max": values[-1],
        "over_80ms_rate": sum(v > 80.0 for v in values) / len(values), "over_100ms_rate": sum(v > 100.0 for v in values) / len(values),
    }


def latency_block(timing: list[dict[str, Any]]) -> dict[str, Any]:
    """틱 지연 요약 — 첫 틱(prefix 포함)은 따로, 나머지는 같이 (docs/03 §7-6: p95 ≤ 80 ms, 100 ms 초과율 ≤ 5 %)."""
    first = [row for row in timing if row.get("prefix_tokens")]
    rest = [row for row in timing if not row.get("prefix_tokens")]
    return {
        "ticks": len(timing),
        "model_ms": _latency_summary(rest, key="model_ms"), "act_ms": _latency_summary(rest, key="act_ms"),
        "serialize_ms": _latency_summary(rest, key="serialize_ms"), "readout_ms": _latency_summary(rest, key="readout_ms"),
        "first_tick_with_prefix": {"model_ms": _latency_summary(first, key="model_ms"), "act_ms": _latency_summary(first, key="act_ms")},
        "tokens_body": _latency_summary(rest, key="tokens_body") if rest else {"n": 0},
        "gate": {"rule": "upper p95 ≤ 80 ms and >100 ms rate ≤ 5 % (docs/03 §7-6), read on model_ms of the non-prefix ticks",
                 "passes": bool(rest) and _latency_summary(rest, key="model_ms")["p95"] <= 80.0 and _latency_summary(rest, key="model_ms")["over_100ms_rate"] <= 0.05},
    }


def cmd_verify(args: argparse.Namespace) -> int:
    import torch

    from robo_jev.harness.model_policy import ModelPolicy, load_serving_judge
    from robo_jev.model.incremental import IncrementalStreamSerializer, project_stream_tick
    from robo_jev.model.serialize import serialize_request

    guard = limit_gpu_memory(args.gpu_memory_fraction)
    torch.set_num_threads(int(args.threads))
    memory_start = memory_report()
    require_free(LOAD_MIN_FREE_BYTES, what="A3 serving load")
    _log(f"gpu guard {guard} · memory at start {memory_start}")
    manifest_path = Path(args.manifest)
    episodes = [_read_episode(manifest_path, episode_id) for episode_id in args.episodes.split(",") if episode_id]
    bundle = load_serving_judge(args.checkpoint, model_id=args.model, compile_dense=False)
    judge, tokenizer = bundle["judge"], bundle["tokenizer"]
    _log(f"loaded {args.checkpoint} in {bundle['load_seconds']} s · contract {bundle['manifest'].get('contract_sha256')}")
    result: dict[str, Any] = {
        "script": SCRIPT_VERSION, "generated_at": _now(), "git": _git_commit(), "checkpoint": str(args.checkpoint), "model_id": args.model,
        "checkpoint_manifest": {key: bundle["manifest"].get(key) for key in ("model", "contract_sha256", "serializer_version", "tokenizer")},
        "tokenizer": bundle["tokenizer_name"], "load_seconds": bundle["load_seconds"], "manifest": str(manifest_path),
        "policy": None, "episodes": [], "levers": {},
    }
    for lever in ("fused", "all"):
        if lever == "all":
            started = time.perf_counter()
            judge.backbone.compile_dense_parts()
            result["levers"]["all"] = {"compile_seconds": round(time.perf_counter() - started, 1)}
            _log(f"compiled dense parts in {result['levers']['all']['compile_seconds']} s")
        policy = ModelPolicy(judge, tokenizer, fused=True)
        if result["policy"] is None:
            result["policy"] = policy.describe()
        timing_all: list[dict[str, Any]] = []
        for record in episodes:
            episode_id = record["episode_id"]
            entry = next((item for item in result["episodes"] if item["episode_id"] == episode_id), None)
            if entry is None:
                entry = {"episode_id": episode_id, "profile": record["provenance"]["profile"], "seed": record["provenance"]["seed"], "split": record["split"],
                         "ticks": len(record["ticks"]), "goal_changes": sum(1 for a, b in zip(record["ticks"], record["ticks"][1:])
                                                                           if int((b["request"]["state"].get("goal") or {}).get("version", 1)) > int((a["request"]["state"].get("goal") or {}).get("version", 1))),
                         "by_lever": {}}
                result["episodes"].append(entry)
                layout = serialize_request(record, tokenizer, layout="stream_l1a", window_ticks=policy.window_ticks)
                entry["tokens"] = len(layout["tokens"])
                # 증분 직렬화 = 전체 직렬화를 틱 경계에서 자른 것 (실제 tokenizer·실제 에피소드에서 다시 한 번)
                from robo_jev.contracts import model_input

                projected = model_input(record)
                incremental = IncrementalStreamSerializer(tokenizer, question_set=projected["prefix"]["question_set"], instructions=projected["prefix"]["instructions"],
                                                          first_state=projected["ticks"][0]["request"]["state"], window_ticks=policy.window_ticks)
                token_mismatch = 0
                for tick in record["ticks"]:
                    piece = incremental.tick(project_stream_tick(tick))
                    expected = layout["ticks"][piece["index"]]
                    if piece["tokens"] != layout["tokens"][expected["start"]:expected["end"]] or piece["candidate_boundaries"] != expected["candidate_boundaries"]:
                        token_mismatch += 1
                entry["incremental_tokens_identical"] = token_mismatch == 0 and incremental.prefix_tokens == layout["tokens"][: layout["prefix_end"]] and incremental.cursor == len(layout["tokens"])
                entry["_layout"] = layout
                started = time.perf_counter()
                entry["_offline"] = _offline_answers(judge, layout, fused=True)
                entry["offline_seconds"] = round(time.perf_counter() - started, 2)
            started = time.perf_counter()
            answers, timing = _policy_answers(policy, record)
            block = {
                "policy_seconds": round(time.perf_counter() - started, 2),
                "vs_offline_replay": _compare(entry["_offline"], answers, name=f"{episode_id}/{lever}"),
                "latency": latency_block(timing),
            }
            if args.stored_report:
                block["vs_stored_r3a_per_record"] = _compare_stored(_stored_rows(Path(args.stored_report), args.stored_split, episode_id), answers)
            entry["by_lever"][lever] = block
            timing_all.extend(timing)
            _log(f"{episode_id} [{lever}] agreement {block['vs_offline_replay']['agreement_rate']:.4f} over {block['vs_offline_replay']['questions_compared']} answers · "
                 f"model p95 {block['latency']['model_ms'].get('p95', float('nan')):.1f} ms")
        result["levers"].setdefault(lever, {})["latency_all_episodes"] = latency_block(timing_all)
    for entry in result["episodes"]:
        entry.pop("_layout", None)
        entry.pop("_offline", None)
    result["summary"] = {
        lever: {
            "agreement_rate": (sum(e["by_lever"][lever]["vs_offline_replay"]["agree"] for e in result["episodes"]) / max(1, sum(e["by_lever"][lever]["vs_offline_replay"]["questions_compared"] for e in result["episodes"]))),
            "questions_compared": sum(e["by_lever"][lever]["vs_offline_replay"]["questions_compared"] for e in result["episodes"]),
            "disagreements": sum(len(e["by_lever"][lever]["vs_offline_replay"]["disagreements"]) for e in result["episodes"]),
            "max_abs_prob_diff": max(e["by_lever"][lever]["vs_offline_replay"]["max_abs_prob_diff"] or 0.0 for e in result["episodes"]),
            "stored_q_main_rate": (sum(e["by_lever"][lever].get("vs_stored_r3a_per_record", {}).get("q_main", {}).get("agree", 0) for e in result["episodes"]) /
                                   max(1, sum(e["by_lever"][lever].get("vs_stored_r3a_per_record", {}).get("q_main", {}).get("n", 0) for e in result["episodes"]))) if args.stored_report else None,
            "stored_q_stop_rate": (sum(e["by_lever"][lever].get("vs_stored_r3a_per_record", {}).get("q_stop", {}).get("agree", 0) for e in result["episodes"]) /
                                   max(1, sum(e["by_lever"][lever].get("vs_stored_r3a_per_record", {}).get("q_stop", {}).get("n", 0) for e in result["episodes"]))) if args.stored_report else None,
            "latency": result["levers"][lever]["latency_all_episodes"],
        }
        for lever in ("fused", "all")
    }
    result["incremental_tokens_identical"] = all(e["incremental_tokens_identical"] for e in result["episodes"])
    result["gpu"] = {"guard": guard, "memory_at_start": memory_start, "memory_at_end": memory_report(), "peak_allocated_bytes": int(torch.cuda.max_memory_allocated())}
    _write(Path(args.out), result)
    for lever, block in result["summary"].items():
        print(f"{lever}: agreement {block['agreement_rate']:.4f} ({block['disagreements']} of {block['questions_compared']} differ, max |Δp| {block['max_abs_prob_diff']:.4f}) · "
              f"model ms p50/p95/p99 {block['latency']['model_ms']['p50']:.1f}/{block['latency']['model_ms']['p95']:.1f}/{block['latency']['model_ms']['p99']:.1f} · "
              f">100 ms {block['latency']['model_ms']['over_100ms_rate']:.4f} · stored q_main {block['stored_q_main_rate']}")
    return 0


# --------------------------------------------------------------------------
# seeds / run / report — Stage B·C (robo_jev.closed_loop)
# --------------------------------------------------------------------------


def cmd_seeds(args: argparse.Namespace) -> int:
    from robo_jev.closed_loop import load_closed_loop_config, select_conditions

    config = load_closed_loop_config(args.config)
    conditions = select_conditions(config, manifest=Path(args.manifest))
    payload = {"script": SCRIPT_VERSION, "generated_at": _now(), "git": _git_commit(), "config": str(args.config), **conditions}
    _write(Path(args.out), payload)
    for name, block in conditions["conditions"].items():
        print(f"{name}: {len(block['seeds'])} seeds · profiles {block['by_profile']} · scanned {block['scanned']} · families in r1 {block['families_in_r1']}/{len(block['families'])}")
    return 0


def cmd_overlap(args: argparse.Namespace) -> int:
    """D1 (Task R6): 조건의 장면 계열이 학습 재료(train 분할)의 계열과 겹치는 수 — "본 적 없는 seed·대부분 학습한 계열"의 숫자."""
    from robo_jev.closed_loop import family_overlap

    seeds = json.loads(Path(args.seeds).read_text(encoding="utf-8"))
    materials = {pair.split("=", 1)[0]: pair.split("=", 1)[1] for pair in args.material}
    payload = {"script": SCRIPT_VERSION, "generated_at": _now(), "git": _git_commit(), "seeds": args.seeds, "materials": materials,
               "conditions": family_overlap(seeds, materials)}
    _write(Path(args.out), payload)
    for name, block in payload["conditions"].items():
        any_ = block["any_material"]
        print(f"{name}: {block['seeds']} seeds · origin groups {block['origin_groups']} (shared with training material {any_['origin_groups']}, seeds {any_['seeds']}) · "
              f"families {block['families']} (shared {any_['families']}, seeds {any_['seeds_by_family']})")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    from robo_jev.closed_loop import build_policy, load_closed_loop_config, run_condition, select_conditions

    guard = None
    if args.policy == "model":
        import torch

        guard = limit_gpu_memory(args.gpu_memory_fraction)
        torch.set_num_threads(int(args.threads))
        require_free(LOAD_MIN_FREE_BYTES, what="closed-loop serving load")
        _log(f"gpu guard {guard} · memory at start {memory_report()}")
    config = load_closed_loop_config(args.config)
    seeds = json.loads(Path(args.seeds).read_text(encoding="utf-8")) if args.seeds else select_conditions(config, manifest=Path(args.manifest))
    bundle = build_policy(args.policy, generator=config["generator"], checkpoint=args.checkpoint, model_id=args.model, compile_dense=not args.no_compile)
    label = args.label or args.policy
    out_root = Path(args.out)
    runs: dict[str, Any] = {}
    for condition in [name.strip() for name in args.condition.split(",") if name.strip()]:
        block = seeds["conditions"][condition]
        schedule = [(entry["profile"], int(entry["seed"])) for entry in block["seeds"]]
        if args.limit:
            schedule = schedule[: int(args.limit)]
        _log(f"run {label} × {condition}: {len(schedule)} episodes")
        runs[condition] = run_condition(bundle, schedule, config=config, out=out_root / condition, condition=condition, label=label, log=sys.stderr, max_ticks=args.max_ticks, id_tag=args.id_tag)
        _log(f"{label} × {condition}: done {runs[condition]['summary']['done']}/{runs[condition]['summary']['episodes']} · wall {runs[condition]['summary']['wall_seconds']:.1f} s")
    payload = {
        "script": SCRIPT_VERSION, "generated_at": _now(), "git": _git_commit(), "policy": bundle["describe"], "label": label,
        "checkpoint": args.checkpoint, "seeds_config": str(args.config), "seeds_file": args.seeds, "id_tag": args.id_tag, "conditions": runs,
    }
    if args.policy == "model":
        payload["gpu"] = {"guard": guard, "memory_at_end": memory_report()}
    _write(Path(args.report), payload)
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    from robo_jev.closed_loop import closed_loop_report

    paths = [Path(item) for pattern in args.runs for item in sorted(map(str, REPO.glob(pattern) if not Path(pattern).is_absolute() else Path("/").glob(pattern.lstrip("/"))))]
    if not paths:
        paths = [Path(item) for item in args.runs]
    merge = {pair.split("=", 1)[0]: [part for part in pair.split("=", 1)[1].split(",") if part] for pair in (args.merge or [])}
    only = [name for name in args.only.split(",") if name] if args.only else None
    seed_pairs = [tuple(pair.split(":", 1)) for pair in (args.seed_pairs or [])]
    report = closed_loop_report(paths, offline=[Path(item) for item in (args.offline or [])], merge=merge or None, only=only, seed_pairs=seed_pairs or None)
    report.update({"script": SCRIPT_VERSION, "generated_at": _now(), "git": _git_commit()})
    _write(Path(args.out), report)
    from robo_jev.closed_loop import print_report

    print_report(report)
    return 0


def print_verdict(verdict: dict[str, Any], file: Any = None) -> None:
    """등록 규칙 적용 결과의 표 (보고서에 옮겨 적는다)."""
    file = file or sys.stdout

    def row(name: str, entry: dict[str, Any]) -> str:
        if entry.get("stopped"):
            return f"| {name} | {entry['pair']} | {entry['metric']} | — (stopped: {entry['stopped']}) | {entry['holds_if']} | no |"
        low, high = entry["ci"]
        return f"| {name} | {entry['pair']} | {entry['metric']} | {entry['value']:+.3f} [{low:+.3f}, {high:+.3f}] | {entry['holds_if']} | {'yes' if entry['holds'] else 'no'} |"

    selection = verdict.get("selection")
    if selection:
        print(f"\n#### seed selection on {selection['condition']} (by {selection['metric']}, ties {selection['ties']})\n", file=file)
        print("| seed group | seed | eligible | episodes | strict | done | false done |", file=file)
        print("| --- | ---: | --- | ---: | ---: | ---: | ---: |", file=file)
        for candidate in selection["candidates"]:
            if not candidate["eligible"]:
                print(f"| {candidate['label']} | {candidate['seed']} | no ({candidate['reason']}) | — | — | — | — |", file=file)
                continue
            print(f"| {candidate['label']} | {candidate['seed']} | yes | {candidate['episodes']} | {candidate['strict']} | {candidate['done']} | {candidate['false_done']} |", file=file)
        print(f"- selected: **{selection['selected']}** (ranking {selection['ranking']}; decided by {selection['decided_by']})", file=file)
    blocks = [(verdict["primary"]["condition"] + " (primary)", verdict["primary"])]
    blocks += [(f"{name} (secondary)", block) for name, block in verdict["secondary"].items() if block.get("conditions")]
    for title, block in blocks:
        print(f"\n#### {title}\n", file=file)
        print("| condition | pair | metric | difference [95 %] | holds if | holds? |", file=file)
        print("| --- | --- | --- | --- | --- | --- |", file=file)
        for name, entry in block["conditions"].items():
            print(row(name, entry), file=file)
        for name, entry in block["robustness"].items():
            if entry.get("available") is False:
                continue
            alt = entry["alternative_seeds"]
            print(f"- robustness ({name}, {entry['pair']} {entry['metric']}): discordant {entry['discordant']['a_only']} vs {entry['discordant']['b_only']}, "
                  f"exact McNemar p {entry['mcnemar_exact_p']:.3f}; RNG seeds {alt.get('seeds')}: lower ≤ 0 in {alt.get('lower_bound_at_or_below_zero')} of {alt.get('count')}, "
                  f"upper ≥ 0 in {alt.get('upper_bound_at_or_above_zero')} of {alt.get('count')}", file=file)
        cause = block.get("cause")
        if cause:
            for name, entry in cause["metrics"].items():
                low, high = entry["ci"]
                print(f"- cause ({cause['pair']}) {name}: {entry['value']:+.3f} [{low:+.3f}, {high:+.3f}] → {entry['status']} — {entry['call']}", file=file)
            print(f"- cause call: **{cause['call']}** — {cause['text']}" + (f"; new regressions against r7: {cause['new_regressions']}" if cause["new_regressions"] else ""), file=file)
        for name, group in (block.get("groups") or {}).items():
            state = f"stopped ({group['stopped']})" if group.get("stopped") else ("passes all" if group["passed"] else f"fails {group['failed']}")
            print(f"- seed group {name}: {state}", file=file)
        for entry in block.get("stability") or ():
            if not entry.get("available"):
                print(f"- stability {entry['name']} ({entry['pair']}): not available", file=file)
                continue
            low, high = entry["ci"]
            print(f"- stability {entry['name']} ({entry['pair']} {entry['metric']}): {entry['value']:+.3f} [{low:+.3f}, {high:+.3f}]" + (" (0 inside)" if entry["includes_zero"] else ""), file=file)
        print(f"- cloud: **{'recommend' if block['cloud'] else 'closed'}**" + (f" (failed: {block['failed']})" if block["failed"] else ""), file=file)
    if verdict["verdict"].get("call"):
        print(f"\n- registered call: **{verdict['verdict']['call']}** — {verdict['verdict']['call_text']}", file=file)


def monitor_verdicts(registration: dict[str, Any], runs_root: Path) -> tuple[dict[str, str], dict[str, Any]]:
    """R8: 등록의 `groups`마다 run 디렉터리의 `metrics.json` 요약에서 head 적합 감시 판정을 읽는다 → (폐루프가 없는 묶음 → 이유, 묶음 → 판정 블록).

    감시가 `fits`가 아니면 그 seed는 멈춘 것이고, run이 없거나 감시 step에 닿지 않았거나 끝나지 않았으면(status가 `completed`가 아니면) 그 이유로
    폐루프가 없는 것이다. 학습이 적용한 감시(질문·step·창·비율)가 등록값과 다르면 판정하지 않는다(ValueError) — 규칙을 조용히 바꾸지 않는다."""
    monitor = registration.get("monitor")
    if not monitor:
        return {}, {}
    registered = (str(monitor["question"]), int(monitor["step"]), [int(monitor["step"]) - int(monitor["window"]) + 1, int(monitor["step"])], float(monitor["ratio"]))
    stopped: dict[str, str] = {}
    blocks: dict[str, Any] = {}
    for group, entry in registration["groups"].items():
        path = Path(runs_root) / entry["run"] / "metrics.json"
        if not path.is_file():
            stopped[group], blocks[group] = f"no training run ({entry['run']}/metrics.json is missing)", None
            continue
        metrics = json.loads(path.read_text(encoding="utf-8"))
        result = (metrics.get("summary") or {}).get("head_fit_monitor")
        if result is None:
            stopped[group], blocks[group] = f"the run did not reach the monitor step {registered[1]} (status {metrics.get('status')}, step {metrics.get('step')})", None
            continue
        applied = (result.get("question"), result.get("step"), list(result.get("window") or []), result.get("stop_ratio"))
        if applied != registered:
            raise ValueError(f"{group}: 학습이 적용한 감시 {applied}가 등록값 {registered}와 다르다 — 판정하지 않는다")
        blocks[group] = {**result, "status": metrics.get("status"), "step_reached": metrics.get("step")}
        if result.get("verdict") != "fits":
            stopped[group] = (f"head-fit monitor: {result.get('verdict')} ({result.get('question')} loss mean {result.get('loss_mean')} ≥ "
                              f"{result.get('stop_ratio')} × baseline mean {result.get('baseline_mean')})")  # fmt: skip
        elif metrics.get("status") != "completed":
            stopped[group] = f"the run did not complete (status {metrics.get('status')}, step {metrics.get('step')})"
    return stopped, blocks


def run_readings(registration: dict[str, Any], runs_root: Path) -> tuple[dict[str, str], dict[str, Any]]:
    """R9: seed를 고르는 등록이면 `groups`마다 run의 `metrics.json`을 읽는다 → (고를 수 없는 묶음 → 이유, 묶음 → 기록(:func:`robo_jev.closed_loop.run_reading`)).

    run이 없거나, status가 `completed`가 아니거나, step이 그 run의 `max_steps`에 닿지 않았으면 그 seed는 고를 수 없다(이유를 적는다). 등록이 감시를
    **기록만** 한다고 한 묶음(`run_readings.log_only`)의 학습이 감시를 적용했다면 판정하지 않는다(ValueError) — 규칙을 조용히 바꾸지 않는다."""
    from robo_jev.closed_loop import run_reading

    if not registration.get("selection"):
        return {}, {}
    readings = registration.get("run_readings") or {}
    log_only = set(readings.get("log_only") or ())
    unavailable: dict[str, str] = {}
    blocks: dict[str, Any] = {}
    for group, entry in registration["groups"].items():
        path = Path(runs_root) / entry["run"] / "metrics.json"
        if not path.is_file():
            unavailable[group], blocks[group] = f"no training run ({entry['run']}/metrics.json is missing)", None
            continue
        block = run_reading(json.loads(path.read_text(encoding="utf-8")), readings)
        blocks[group] = {"run": entry["run"], **block}
        if group in log_only and block["applied_monitor"] is not None:
            raise ValueError(f"{group}: 등록은 감시를 기록만 한다고 했는데 학습이 감시 {block['applied_monitor']}를 적용했다 — 판정하지 않는다")
        if block["status"] != "completed" or block["max_steps"] is None or int(block["step"] or 0) < int(block["max_steps"]):
            unavailable[group] = f"the run did not complete (status {block['status']}, step {block['step']} of max_steps {block['max_steps']})"
    return unavailable, blocks


def cmd_verdict(args: argparse.Namespace) -> int:
    """E1 (Task R7·R8·R9): 사전 등록 규칙(`configs/eval/r7-registration.yaml`·`r8-registration.yaml`·`r9-registration.yaml`)을 폐루프 보고서에 적힌 그대로
    적용한다. R8 등록이면 seed 묶음마다 run의 head 적합 감시 판정을 먼저 읽고(:func:`monitor_verdicts`), R9 등록(seed 고르기)이면 run마다 완주 여부와
    판정 옆의 기록을 읽는다(:func:`run_readings`)."""
    from robo_jev.closed_loop import apply_registration, load_registration

    report = json.loads(Path(args.report).read_text(encoding="utf-8"))
    registration = load_registration(args.registration)
    stopped, monitors = monitor_verdicts(registration, Path(args.runs_root))
    unavailable, readings = run_readings(registration, Path(args.runs_root))
    verdict = apply_registration(report, registration, stopped={**stopped, **unavailable})
    verdict.update({"script": SCRIPT_VERSION, "generated_at": _now(), "git": _git_commit(), "report": str(args.report), "monitor": monitors})
    if readings:
        verdict["runs"] = readings
    _write(Path(args.out), verdict)
    print_verdict(verdict)
    return 0


def cmd_transitions(args: argparse.Namespace) -> int:
    from robo_jev.closed_loop import offline_gripper_transitions
    from robo_jev.data.sealed import read_open_episodes

    report = json.loads(Path(args.report).read_text(encoding="utf-8"))
    out: dict[str, Any] = {"script": SCRIPT_VERSION, "generated_at": _now(), "git": _git_commit(), "report": str(args.report), "splits": {}}
    for pair in args.split:
        name, directory = pair.split("=", 1)
        records = read_open_episodes(Path(directory))[0]  # manifest로 먼저 — 봉인 편은 열지 않는다 (R7 A3)
        out["splits"][name] = {"records_dir": directory, "episodes": len(records), **offline_gripper_transitions(report, records, split_name=name)}
        block = out["splits"][name]
        print(f"{name}: q_gripper whole {block['whole_question_accuracy']:.4f} · initiate {block['initiate']['accuracy']} ({block['initiate']['correct']}/{block['initiate']['n']}) · "
              f"window predicted closed {block['window']['predicted_closed']}/{block['window']['n']} · "
              f"settled {block['settled']['accuracy']} ({block['settled']['correct']}/{block['settled']['n']}) · open {block['open']['accuracy']} ({block['open']['correct']}/{block['open']['n']}) · "
              f"episodes with initiate ticks {block['episodes_with_initiate_ticks']}, all wrong {block['episodes_where_every_initiate_tick_is_wrong']}")
    _write(Path(args.out), out)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    verify = sub.add_parser("verify", help="A3: 녹화된 dev 에피소드에서 오프라인 재생 평가 대 루프 정책의 틱별 일치 (+ A4 재생 지연)")
    verify.add_argument("--checkpoint", required=True)
    verify.add_argument("--model", default="Qwen/Qwen3.5-2B")
    verify.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    verify.add_argument("--episodes", default=",".join(DEFAULT_VERIFY_EPISODES), help="쉼표로 나눈 에피소드 id (dev 분할)")
    verify.add_argument("--stored-report", dest="stored_report", default=None, help="R3a 평가 산출물(per_record가 있는 것) — 다른 프로세스의 답과도 대조")
    verify.add_argument("--stored-split", dest="stored_split", default="robot/dev")
    verify.add_argument("--threads", type=int, default=8)
    verify.add_argument("--gpu-memory-fraction", dest="gpu_memory_fraction", type=float, default=DEFAULT_FRACTION)
    verify.add_argument("--out", required=True)
    verify.set_defaults(func=cmd_verify)

    seeds = sub.add_parser("seeds", help="B2: dev 100 · ood_dev 26 seed 목록")
    seeds.add_argument("--config", default=str(DEFAULT_SEEDS_CONFIG))
    seeds.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    seeds.add_argument("--out", required=True)
    seeds.set_defaults(func=cmd_seeds)

    overlap = sub.add_parser("overlap", help="R6 D1: 조건의 장면 계열과 학습 재료(train 분할)의 계열 겹침")
    overlap.add_argument("--seeds", required=True)
    overlap.add_argument("--material", action="append", required=True, metavar="NAME=MANIFEST")
    overlap.add_argument("--out", required=True)
    overlap.set_defaults(func=cmd_overlap)

    run = sub.add_parser("run", help="B1: 정책 하나를 조건의 seed 전부에 돌린다")
    run.add_argument("--policy", required=True, choices=["model", "rule", "mechanical", "expert"])
    run.add_argument("--checkpoint", default=None)
    run.add_argument("--model", default="Qwen/Qwen3.5-2B")
    run.add_argument("--label", default=None, help="보고서·디렉터리의 이름표 (예: s18, 466)")
    run.add_argument("--id-tag", dest="id_tag", default="r4", help="에피소드 id 꼬리의 라운드 표지 `-<tag>-<label>` (R4 기록은 r4; R5의 run은 r5)")
    run.add_argument("--condition", default="dev,ood_dev")
    run.add_argument("--config", default=str(DEFAULT_SEEDS_CONFIG))
    run.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    run.add_argument("--seeds", default=None, help="`seeds` 명령의 산출물 (없으면 다시 고른다 — 결정적이다)")
    run.add_argument("--out", required=True, help="에피소드를 쓸 디렉터리 (조건마다 하위 디렉터리)")
    run.add_argument("--report", required=True, help="run 요약 JSON (`report` 명령의 입력)")
    run.add_argument("--limit", type=int, default=None, help="조건마다 앞 N편만 (검사용)")
    run.add_argument("--max-ticks", dest="max_ticks", type=int, default=None)
    run.add_argument("--no-compile", dest="no_compile", action="store_true", help="모델: dense 부분 compile을 끈다 (지렛대 fused만)")
    run.add_argument("--threads", type=int, default=8)
    run.add_argument("--gpu-memory-fraction", dest="gpu_memory_fraction", type=float, default=DEFAULT_FRACTION)
    run.set_defaults(func=cmd_run)

    report = sub.add_parser("report", help="B3·C: 정책·조건별 지표와 seed로 짝지은 구간")
    report.add_argument("--runs", nargs="+", required=True, help="run 요약 JSON (glob 가능)")
    report.add_argument("--offline", nargs="*", default=None, help="같은 checkpoint의 오프라인 판정 칸 산출물 (나란히 적는다)")
    report.add_argument("--merge", action="append", default=None, metavar="NAME=COND,COND",
                        help="조건 여럿을 한 표로 합친다 (R6: ood_dev100=ood_dev,ood_dev_new); 같은 seed가 둘에 있으면 거절")
    report.add_argument("--only", default=None, help="보고할 조건 (쉼표로; 합친 이름도 된다)")
    report.add_argument("--seed-pairs", dest="seed_pairs", nargs="*", default=None, metavar="A:B",
                        help="이 쌍마다 seed 단위 짝지은 지표(그리퍼 연속·중복·q_stop·안전·오행동·실패 원인)와 엄격 성공·거짓 done의 견고성(다른 부트스트랩 seed 1~200)")
    report.add_argument("--out", required=True)
    report.set_defaults(func=cmd_report)

    verdict = sub.add_parser("verdict", help="E1 (R7·R8·R9): 사전 등록 규칙을 폐루프 보고서에 적힌 그대로 적용한다 (클라우드 조건·원인 판정·seed 감시·검증 집합의 seed 고르기)")
    verdict.add_argument("--report", required=True, help="`report`의 산출물 (주 집합과 seed 단위 쌍이 든 것)")
    verdict.add_argument("--registration", default=str(REPO / "configs/eval/r7-registration.yaml"))
    verdict.add_argument("--runs-root", dest="runs_root", default=str(REPO), help="등록의 groups[].run이 가리키는 run 디렉터리의 뿌리 (R8·R9; 기본: 저장소)")
    verdict.add_argument("--out", required=True)
    verdict.set_defaults(func=cmd_verdict)

    transitions = sub.add_parser("transitions", help="C: 오프라인 재생의 q_gripper 예측을 전환 틱(initiate)·정착 틱(settled)·open 틱으로 나눠 채점")
    transitions.add_argument("--report", required=True, help="adapt_readout --eval-checkpoint 산출물 (q_gripper per_record가 있는 것)")
    transitions.add_argument("--split", nargs="+", required=True, metavar="NAME=DIR", help="평가 집합의 분할 이름 = 그 레코드 디렉터리")
    transitions.add_argument("--out", required=True)
    transitions.set_defaults(func=cmd_transitions)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
