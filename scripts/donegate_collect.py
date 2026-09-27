#!/usr/bin/env python
"""done 게이트 수집 (Task R6 A3) — 거짓 done **뒤의** 상태를 방문하는 DAgger 수집. **평가가 아니다** (:mod:`robo_jev.data.done_gate`).

    # 장면: train 계열의 새 seed, E1·E2 각 100 (base = dagger 사이클 1의 시작 600100; 다른 분할은 세기만, 봉인은 나열도 하지 않는다)
    uv run python scripts/donegate_collect.py seeds --out artifacts/reports/r6-donegate-seeds.json
    # 수집 (GPU): R5 체크포인트의 모델 답 + expert의 q_done → 원 기록 (그 뒤 gripper_labels dagger --collection expert_done_gate로 재라벨)
    uv run python scripts/donegate_collect.py run --checkpoint artifacts/runs/r5-t1-fp32-2b-s18/checkpoint.pt \\
        --seeds artifacts/reports/r6-donegate-seeds.json --out artifacts/scratch/r6/donegate-raw/train --report artifacts/reports/r6-donegate-run.json

GPU 규칙: 울타리 0.6(`robo_jev.gpu`), 적재 전 여유 검사, 한 번에 하나(`artifacts/scratch/r6/gpu.sh`, `choom -n 1000`).
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from robo_jev.gpu import DEFAULT_FRACTION, limit_gpu_memory, memory_report, require_free  # noqa: E402

GENERATOR_CONFIG = "configs/data/r1_robot.yaml"
#: 서빙 모델 적재 전에 있어야 하는 장치(통합) 메모리 여유 — `scripts/closed_loop.py`와 같은 값(checkpoint를 CPU에 읽은 뒤 bf16 ≈ 4.2 GiB).
LOAD_MIN_FREE_BYTES = 36 * 2**30
#: 사이클 1의 수집 몫 — 지시가 바뀌는 프로파일만(거짓 done은 R5 기록에서 전부 E1·E2였다).
QUOTA = {"E1": 100, "E2": 100}
SCRIPT_VERSION = "r6-donegate-1.0"


def _git() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def _log(message: str) -> None:
    print(f"[r6 {time.strftime('%H:%M:%S')}] {message}", file=sys.stderr, flush=True)


def _write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=1, default=str) + "\n", encoding="utf-8")
    print(f"→ {path}", flush=True)


def cmd_seeds(args: argparse.Namespace) -> int:
    import yaml

    from robo_jev.closed_loop import select_seeds
    from robo_jev.data.done_gate import collection_seed_base
    from robo_jev.data.robot_episodes import config_paths, load_generator_config, seed_schedule
    from robo_jev.sim.controller import resolve_config_path

    generator = load_generator_config(args.config)
    sim = yaml.safe_load(resolve_config_path(config_paths(generator)["sim_config"]).read_text(encoding="utf-8"))
    base = collection_seed_base(generator, cycle=args.cycle)
    quota = dict(QUOTA)
    block = select_seeds(generator, sim, split="train", count=sum(quota.values()), base=base, per_profile_max=args.per_profile_max, quota=quota)
    r1 = {(profile, seed) for profile, seed in seed_schedule(generator, 400)}
    collisions = [item for item in block["seeds"] if (item["profile"], item["seed"]) in r1]
    if collisions:
        raise SystemExit(f"r1 seed와 겹친다: {collisions[:3]}")
    payload = {"script": SCRIPT_VERSION, "generated_at": _now(), "git": _git(), "config": args.config, "cycle": args.cycle, "seeds_base": base,
               "conditions": {"train_donegate": {**block, "families": sorted({item["origin_group"] for item in block["seeds"]})}}}
    _write(Path(args.out), payload)
    print(f"train_donegate: {len(block['seeds'])} seeds · profiles {block['by_profile']} · scanned {block['scanned']} · other splits {block['skipped_other_splits']}")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    import torch

    from robo_jev.closed_loop import build_policy
    from robo_jev.data.done_gate import DoneGatePolicy, collect_done_gate
    from robo_jev.data.robot_episodes import load_generator_config
    from robo_jev.sim.expert import Expert

    guard = limit_gpu_memory(args.gpu_memory_fraction)
    torch.set_num_threads(int(args.threads))
    require_free(LOAD_MIN_FREE_BYTES, what="done-gate collection serving load")
    _log(f"gpu guard {guard} · memory at start {memory_report()}")
    generator = load_generator_config(args.config)
    seeds = json.loads(Path(args.seeds).read_text(encoding="utf-8"))
    block = seeds["conditions"]["train_donegate"]
    schedule = [(entry["profile"], int(entry["seed"])) for entry in block["seeds"]]
    if args.limit:
        schedule = schedule[: int(args.limit)]
    bundle = build_policy("model", generator=generator, checkpoint=args.checkpoint, model_id=args.model, compile_dense=not args.no_compile)
    policy = DoneGatePolicy(bundle["policy"], Expert())
    _log(f"collect {len(schedule)} episodes · policy {policy.version} · checkpoint {args.checkpoint}")
    summary = collect_done_gate(policy, schedule, config=generator, out=Path(args.out), label=args.label, id_tag=args.id_tag,
                                max_ticks=args.max_ticks, log=sys.stderr, describe={k: v for k, v in bundle["describe"].items() if k != "name"})
    payload = {
        "script": SCRIPT_VERSION, "generated_at": _now(), "git": _git(), "checkpoint": args.checkpoint, "seeds_file": args.seeds,
        "policy": {"name": policy.name, "version": policy.version, "inner": bundle["describe"]}, "summary": summary,
        "gpu": {"guard": guard, "memory_at_end": memory_report(), "peak_allocated_bytes": int(torch.cuda.max_memory_allocated())},
    }
    _write(Path(args.report), payload)
    done = sum(1 for row in summary["rows"] if row["done"] and row["inside"])
    _log(f"collected {summary['episodes']} episodes in {summary['wall_seconds']} s · completed (expert gate) {done}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    seeds = sub.add_parser("seeds", help="train 계열의 새 seed (E1·E2 각 100)")
    seeds.add_argument("--config", default=GENERATOR_CONFIG)
    seeds.add_argument("--cycle", type=int, default=1)
    seeds.add_argument("--per-profile-max", dest="per_profile_max", type=int, default=4000)
    seeds.add_argument("--out", required=True)
    seeds.set_defaults(func=cmd_seeds)
    run = sub.add_parser("run", help="GPU: 수집 정책으로 돌려 원 기록을 쓴다")
    run.add_argument("--checkpoint", required=True)
    run.add_argument("--model", default="Qwen/Qwen3.5-2B")
    run.add_argument("--config", default=GENERATOR_CONFIG)
    run.add_argument("--seeds", required=True)
    run.add_argument("--out", required=True)
    run.add_argument("--report", required=True)
    run.add_argument("--label", default="donegate")
    run.add_argument("--id-tag", dest="id_tag", default="r6")
    run.add_argument("--limit", type=int, default=None)
    run.add_argument("--max-ticks", dest="max_ticks", type=int, default=None)
    run.add_argument("--no-compile", dest="no_compile", action="store_true")
    run.add_argument("--threads", type=int, default=8)
    run.add_argument("--gpu-memory-fraction", dest="gpu_memory_fraction", type=float, default=DEFAULT_FRACTION)
    run.set_defaults(func=cmd_run)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
