"""Task R8 A3 — 학습된 checkpoint를 **자기 학습 에피소드**에 교사 강제로 돌려 부가 head가 학습 데이터에 맞았는지 본다 (GPU ≈ 10분).

    artifacts/scratch/r8/gpu.sh a3 scripts/teacher_forced_probe.py --run r7=artifacts/runs/r7-t1-fp32-2b-s18 \\
        --run r5=artifacts/runs/r5-t1-fp32-2b-s18 --episodes-from r7 --count 10 --out artifacts/reports/r8-a3-teacher-forced.json

**무엇을 묻는가 (R7 리뷰 1 Item 1).** r7의 그리퍼 head는 판정 칸에서 settled 0.043·initiate 0.279였다. 둘 중 무엇인가 —
(a) **학습 데이터에도 맞지 않았다**(최적화 실패), (b) 학습 데이터에는 맞는데 평가에서 틀린다(학습/서빙 차이). 그래서 같은 에피소드를 두 계산으로 돌린다:

* **T (학습 계산)** — 학습 loop와 같은 호출: 에피소드를 `stream_chunk_seconds` 구간으로 나눠 :func:`robo_jev.train.run_stream_chunk`
  (동적 KV, 구간 경계 detach)를 no_grad로. 학습 설정의 후보 치환 seed로 적재한 **학습 때와 같은 입력**이다.
* **E (평가 계산)** — C2 판정 칸과 같은 호출 :func:`robo_jev.evaluate.predict_items` (에피소드 전체 재생, `fused`). 같은 입력(치환 판)과
  평가 칸처럼 치환하지 않은 입력(`E_unpermuted`) 둘.

셋 모두에서 `q_gripper`의 층(initiate·settled·open·window) argmax 정확도와 `q_stop`의 참/거짓 틱별 발화, 그리고 A1과 **같은 정의**의
질문별 가중 손실 대 기준선(:func:`robo_jev.train.question_table` — 틱 가중치 = 학습의 틱 종류 가중치, 에피소드 하나 = 학습의 step 하나의 로봇
묶음이므로 기준선은 에피소드의 라벨 주변분포)을 낸다. 에피소드는 `--episodes-from` run이 학습 **마지막 무렵** 뽑은 expert(g2) 편
`--count`개다(`metrics.json`의 `steps[].units`를 끝에서부터; material `existing`). 에피소드 id는 manifest의 train 항목에서만 고른다 —
다른 분할의 파일은 열지 않는다. 대조로 다른 run(예: r5)을 같은 에피소드에 돌린다 — 이 진단이 맞은 head를 알아보는지의 확인이다.

GPU 규칙: 울타리 0.6(`limit_gpu_memory`, 첫 CUDA 할당 전)과 `require_free`, 한 번에 하나(`gpu.sh`).
"""

from __future__ import annotations

import argparse
import datetime as dt
import gc
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from robo_jev.gpu import DEFAULT_FRACTION, limit_gpu_memory, memory_report, require_free  # noqa: E402

SCRIPT_VERSION = "r8-teacher-forced-1.0"
#: 적재 전에 있어야 하는 장치(통합) 메모리 여유 — 2B bf16 4.2 GiB + no_grad 구간 forward. 넉넉히.
LOAD_MIN_FREE_BYTES = 16 * 2**30
#: 표에 앞세우는 질문 (브리프: q_gripper · q_stop · q_main).
FOCUS = ("q_gripper", "q_stop", "q_main")


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def _log(message: str) -> None:
    print(f"[a3 {time.strftime('%H:%M:%S')}] {message}", file=sys.stderr, flush=True)


# --------------------------------------------------------------------------
# 순수 함수 (시험이 CPU에서 고정한다)
# --------------------------------------------------------------------------


def rehome(path: str) -> str:
    """run이 적은 절대 경로(지금은 지워진 worktree 아래일 수 있다)를 이 체크아웃으로 옮긴다 — `artifacts/`·`configs/` 뒤를 그대로 잇는다
    (`artifacts`는 모든 worktree에서 같은 디렉터리의 심볼릭 링크다). 그 밖의 경로는 그대로."""
    text = str(path)
    for anchor in ("/artifacts/", "/configs/"):
        if anchor in text:
            return str(REPO / (anchor.strip("/") + "/" + text.split(anchor, 1)[1]))
    return text


def rehomed_config(config: dict[str, Any]) -> dict[str, Any]:
    """run의 저장된 설정에서 경로 키만 이 체크아웃으로 옮긴 사본 (:func:`rehome`)."""
    out = dict(config)
    out["dataset_manifests"] = [{**entry, "path": rehome(entry["path"])} for entry in config.get("dataset_manifests") or []]
    for key in ("model_config", "artifacts_dir"):
        if isinstance(out.get(key), str):
            out[key] = rehome(out[key])
    out["resume"] = None
    return out


def last_drawn_episodes(metrics: dict[str, Any], *, count: int, material: str = "existing") -> list[dict[str, Any]]:
    """run의 `metrics.json`에서 학습 **마지막 무렵** 뽑힌 스트림 에피소드(`material`) `count`편 — 끝 step부터 거슬러 처음 만나는 순서로,
    편마다 마지막으로 뽑힌 step을 함께. 모자라면 있는 만큼."""
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for step in reversed(metrics.get("steps") or []):
        for unit in step.get("units") or ():
            if unit.get("kind") != "stream" or (unit.get("materials") or [None])[0] != material:
                continue
            for record_id in unit.get("records") or ():
                if record_id not in seen:
                    seen.add(record_id)
                    out.append({"record_id": str(record_id), "last_step": int(step["step"])})
        if len(out) >= count:
            break
    return out[:count]


def drawn_steps(metrics: dict[str, Any], record_ids: list[str]) -> dict[str, list[int]]:
    """편마다 그 run이 그 편을 뽑은 step들 (대조 run이 그 편을 학습에서 봤는지 적는다)."""
    wanted = set(record_ids)
    out: dict[str, list[int]] = {record_id: [] for record_id in record_ids}
    for step in metrics.get("steps") or []:
        for unit in step.get("units") or ():
            for record_id in unit.get("records") or ():
                if record_id in wanted:
                    out[record_id].append(int(step["step"]))
    return out


def prediction_records(predictions: list[dict[str, Any]], items: dict[str, Any], plans: dict[str, Any]) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    """평가 계산(:func:`robo_jev.evaluate.predict_items`)의 틱별 확률 → 에피소드마다 A1의 질문별 기록과 탐침 수.

    학습과 같은 틱(유효 라벨·가중치 > 0)만, 계수 = 틱 가중치 / 에피소드 분모(``Σ w_t``) — 학습 run의 로봇 몫(0.6)만 빠진 같은 비율이다.
    손실은 확률의 log를 logits 자리에 넣은 `label_loss`(softmax는 상수 이동에 불변이라 같은 값)다."""
    import torch

    from robo_jev.loss import question_losses
    from robo_jev.train import _probe_tick, question_records

    by_episode: dict[str, list[dict[str, Any]]] = {}
    probes: dict[str, Any] = {}
    for prediction in predictions:
        if prediction.get("kind") != "stream":
            continue
        record_id = str(prediction["record_id"])
        item, plan = items[record_id], plans[record_id]
        index = int(prediction["tick"])
        if not plan.valid[index] or plan.weights[index] <= 0:
            continue
        logits = {qid: torch.log(p.double()) for qid, p in prediction["probabilities"].items()}
        candidates = {qid: list(ids) for qid, ids in prediction["candidates"].items()}
        tick = item.record["ticks"][index]
        labels = tick.get("labels", [])
        entries = question_losses({"logits": [logits], "candidates": [candidates]}, {"labels": [labels]})[0]
        values = {qid: float(entry["loss"]) for qid, entry in entries.items()}
        by_episode.setdefault(record_id, []).extend(question_records(
            entries, values, labels, candidates, coefficient=plan.weights[index] / plan.normaliser, domain=item.domain,
            question_types=item.question_types,
        ))  # fmt: skip
        _probe_tick(probes, tick, logits, candidates)
    return by_episode, probes


def summarize(by_episode: dict[str, list[dict[str, Any]]], probes: dict[str, Any]) -> dict[str, Any]:
    """에피소드별 질문 표(:func:`robo_jev.train.question_table` — 기준선은 그 에피소드의 라벨 주변분포 = 학습 step 하나의 기준선)와
    감시가 읽는 통계(에피소드들의 가중 손실 평균 대 기준선 평균), 모든 에피소드를 합친 표, 탐침 요약."""
    from robo_jev.train import probe_summary, question_table

    tables = {record_id: question_table(rows) for record_id, rows in by_episode.items()}
    keys = sorted({key for table in tables.values() for key in table})
    means: dict[str, Any] = {}
    for key in keys:
        pairs = [(table[key]["loss"], table[key]["baseline"]) for table in tables.values() if key in table and table[key]["loss"] is not None and table[key]["baseline"] is not None]
        if not pairs:
            continue
        loss = sum(value for value, _ in pairs) / len(pairs)
        baseline = sum(value for _, value in pairs) / len(pairs)
        means[key] = {"episodes": len(pairs), "loss_mean": loss, "baseline_mean": baseline, "ratio_of_means": (loss / baseline) if baseline > 0 else None}
    pooled = question_table([row for rows in by_episode.values() for row in rows])
    return {"per_episode": tables, "monitor_statistic": means, "pooled": pooled, "probes": probe_summary(probes)}


def argmax_agreement(a: list[dict[str, Any]], b: list[dict[str, Any]], question: str) -> dict[str, Any]:
    """두 계산의 틱별 argmax가 같은 몫 (`question`만; 두 쪽에 다 있는 틱)."""
    def index(rows: list[dict[str, Any]]) -> dict[tuple[str, int], str]:
        out = {}
        for row in rows:
            if question in (row.get("probabilities") or {}):
                ids = list(row["candidates"][question])
                out[(str(row["record_id"]), int(row["tick"]))] = ids[int(row["probabilities"][question].argmax())]
        return out

    first, second = index(a), index(b)
    keys = sorted(set(first) & set(second))
    same = sum(1 for key in keys if first[key] == second[key])
    return {"question": question, "ticks": len(keys), "agree": same, "rate": (same / len(keys)) if keys else None}


# --------------------------------------------------------------------------
# GPU 실행
# --------------------------------------------------------------------------


def _training_path(judge: Any, items: dict[str, Any], plans: dict[str, Any]) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any], list[dict[str, Any]]]:
    """학습 계산: 구간마다 `run_stream_chunk`(no_grad) — 학습 loop의 호출 그대로(계수 = 틱 가중치 / 분모). 틱별 확률도 돌려준다(argmax 대조)."""
    import torch

    from robo_jev.train import _merge_probes, detach_stream_state, run_stream_chunk

    by_episode: dict[str, list[dict[str, Any]]] = {}
    probes: dict[str, Any] = {}
    rows: list[dict[str, Any]] = []
    with torch.no_grad():
        for record_id, item in items.items():
            plan = plans[record_id]
            carried = None
            for chunk in plan.chunks:
                result = run_stream_chunk(judge, item, chunk, carried=carried, plan=plan, scale=1.0 / plan.normaliser)
                by_episode.setdefault(record_id, []).extend(result.stats["questions"])
                _merge_probes(probes, result.stats["probes"])
                for offset, index in enumerate(range(chunk[0], chunk[1])):
                    rows.append({"record_id": record_id, "tick": index, "candidates": result.outputs["candidates"][offset],
                                 "probabilities": {qid: torch.softmax(z.detach().float().cpu(), 0) for qid, z in result.outputs["logits"][offset].items()}})  # fmt: skip
                carried = detach_stream_state(result.state)
                del result
    return by_episode, probes, rows


def run_one(label: str, run_dir: Path, record_ids: list[str], *, fused: bool) -> dict[str, Any]:
    import torch

    from robo_jev.evaluate import predict_items
    from robo_jev.sampler import load_items
    from robo_jev.train import build_model, build_tokenizer, load_readout_checkpoint, plan_episode, resolve_config, tokenizer_block

    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    config = resolve_config(rehomed_config(metrics["config"]))
    robot = next(entry for entry in config["dataset_manifests"] if entry.get("domain") == "robot" and entry.get("material") in (None, "existing"))
    manifest_path = Path(robot["path"])
    if not manifest_path.is_absolute():
        manifest_path = REPO / manifest_path
    patterns = [f"episodes/{record_id}/streams.jsonl" for record_id in record_ids]
    tokenizer = build_tokenizer(config["tokenizer"])
    sampler = config["sampler"]
    common = {"tokenizer": tokenizer, "splits": ("train",), "layouts": config["layout"], "window_ticks": config["stream_window_ticks"],
              "domain": "robot", "domain_tag": sampler["domain_tag"], "material_tag": sampler["material_tag"]}  # fmt: skip
    started = time.perf_counter()
    # manifest의 train 항목만 연다 (plan_manifest_files — 다른 분할의 파일은 해시도 파싱도 하지 않는다)
    permuted = {item.record_id: item for item in load_items(manifest_path, files=patterns, permute_seed=sampler["permute_candidates_seed"], **common)}
    plain = {item.record_id: item for item in load_items(manifest_path, files=patterns, permute_seed=None, **common)}
    missing = sorted(set(record_ids) - set(permuted))
    if missing:
        raise ValueError(f"{label}: train 분할에서 찾지 못한 편 {missing}")
    plans = {record_id: plan_episode(item.record, chunk_seconds=config["stream_chunk_seconds"], tick_weights=sampler["tick_weights"],
                                     steady_min_held_ticks=sampler["steady_min_held_ticks"]) for record_id, item in permuted.items()}  # fmt: skip
    judge = build_model(config)
    checkpoint = run_dir / "checkpoint.pt"
    manifest = load_readout_checkpoint(judge, checkpoint, tokenizer_sha256=tokenizer_block(config["tokenizer"])["sha256"])
    judge.eval()
    judge.requires_grad_(False)
    load_seconds = round(time.perf_counter() - started, 1)
    _log(f"{label}: loaded {checkpoint} ({load_seconds} s) · contract {str(manifest.get('contract_sha256'))[:12]} · {len(permuted)} episodes")
    out: dict[str, Any] = {"run_dir": str(run_dir.relative_to(REPO)) if run_dir.is_relative_to(REPO) else str(run_dir), "checkpoint_format": None,
                           "contract_sha256": manifest.get("contract_sha256"), "config_seed": config["seed"],
                           "permute_candidates_seed": sampler["permute_candidates_seed"], "stream_chunk_seconds": config["stream_chunk_seconds"],
                           "load_seconds": load_seconds, "paths": {}}  # fmt: skip
    from robo_jev.checkpoint import load_model_checkpoint

    out["checkpoint_format"] = load_model_checkpoint(checkpoint, mmap=True)["format"]
    started = time.perf_counter()
    by_episode, probes, rows_t = _training_path(judge, permuted, plans)
    out["paths"]["T_training"] = {**summarize(by_episode, probes), "seconds": round(time.perf_counter() - started, 1)}
    _log(f"{label}: T done in {out['paths']['T_training']['seconds']} s")
    for name, items in (("E_evaluation", permuted), ("E_unpermuted", plain)):
        started = time.perf_counter()
        predictions = predict_items(judge, list(items.values()), fused=fused)
        by_episode, probes = prediction_records(predictions, items, plans)
        out["paths"][name] = {**summarize(by_episode, probes), "seconds": round(time.perf_counter() - started, 1)}
        if name == "E_evaluation":
            out["agreement_T_vs_E"] = {question: argmax_agreement(rows_t, predictions, question) for question in FOCUS}
        _log(f"{label}: {name} done in {out['paths'][name]['seconds']} s")
        del predictions
    out["training_drawn_steps"] = drawn_steps(metrics, record_ids)
    out["gpu_peak_allocated_bytes"] = int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else None
    del judge
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", action="append", required=True, metavar="LABEL=RUN_DIR", help="checkpoint.pt와 metrics.json이 든 run 디렉터리")
    parser.add_argument("--episodes-from", dest="episodes_from", required=True, help="에피소드를 고를 run의 이름표")
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--no-fused", dest="fused", action="store_false", help="평가 계산에서 fused를 끈다 (기본: 판정 칸 설정처럼 켠다)")
    parser.add_argument("--gpu-memory-fraction", dest="gpu_memory_fraction", type=float, default=DEFAULT_FRACTION)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    guard = limit_gpu_memory(args.gpu_memory_fraction)  # 첫 CUDA 할당 전에
    free = require_free(LOAD_MIN_FREE_BYTES, what="A3 teacher-forced probe")
    _log(f"gpu guard {guard} · memory at start {free}")
    runs = {pair.split("=", 1)[0]: (REPO / pair.split("=", 1)[1]) for pair in args.run}
    if args.episodes_from not in runs:
        parser.error(f"--episodes-from {args.episodes_from!r}가 --run에 없다")
    source = json.loads((runs[args.episodes_from] / "metrics.json").read_text(encoding="utf-8"))
    chosen = last_drawn_episodes(source, count=args.count)
    record_ids = [entry["record_id"] for entry in chosen]
    _log(f"episodes (last drawn by {args.episodes_from}): {chosen}")
    try:
        git = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        git = None
    report: dict[str, Any] = {"script": SCRIPT_VERSION, "generated_at": _now(), "git": git, "gpu": {"guard": guard, "memory_at_start": free},
                              "episodes_from": args.episodes_from, "episodes": chosen, "fused": bool(args.fused), "runs": {}}  # fmt: skip
    out = Path(args.out)
    for label, run_dir in runs.items():
        report["runs"][label] = run_one(label, run_dir, record_ids, fused=bool(args.fused))
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, ensure_ascii=False, indent=1, default=str) + "\n", encoding="utf-8")
    report["gpu"]["memory_at_end"] = memory_report()
    out.write_text(json.dumps(report, ensure_ascii=False, indent=1, default=str) + "\n", encoding="utf-8")
    for label, block in report["runs"].items():
        for path, summary in block["paths"].items():
            grip = summary["probes"].get("q_gripper", {})
            stats = summary["monitor_statistic"]
            print(f"{label} {path}: q_gripper settled {grip.get('settled', {}).get('accuracy')} · initiate {grip.get('initiate', {}).get('accuracy')} · "
                  f"open {grip.get('open', {}).get('accuracy')} · " + " · ".join(
                      f"{key} {stats[key]['loss_mean']:.4f}/{stats[key]['baseline_mean']:.4f}" for key in FOCUS if key in stats))  # fmt: skip
    print(f"→ {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
