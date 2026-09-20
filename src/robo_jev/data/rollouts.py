"""키프레임 rollout 제작 — 레코드 재생으로 snapshot을 얻고, 후보×seed rollout을 돌려 라벨과 비용을 적는다
(docs/04 §4 "Counterfactual rollout의 제작 예산", docs/06 Task 3, Task 3c-2).

    uv run python scripts/rollout_keyframes.py --dataset artifacts/datasets/d1-robot/batch-0 --limit 100 [--workers 8]

흐름: 에피소드 레코드마다 :func:`robo_jev.sim.label.select_keyframes` → 레코드의 `model_output`을 같은 하네스
조합 규칙에 다시 먹여 그 틱까지 **재생**(:func:`replay_to_keyframes`)하고 키프레임 틱의 snapshot을 얻는다 →
:func:`choose_rollout_candidates`로 후보 8개 → 후보 × paired seed의 :func:`rollout_event` → 후보별 집계 →
:func:`label_main_decision`. 결과는 `<dataset>/rollouts/`에 `keyframes.json`·`rollouts.jsonl`·`labels.jsonl`·
`costing.json`으로 쓴다. 레코드 자체는 바꾸지 않는다(라벨은 계보를 유지한 후속 버전에 붙는다, docs/04 §6 —
:func:`robo_jev.data.lineage.attach_rollout_labels`).

**재개(D1 128k).** 에피소드를 `chunk_episodes`개씩 묶어 재생·rollout하고, 묶음이 끝날 때마다 결과를 `rollouts.jsonl`·
`keyframes.jsonl`에 **덧붙이고** 완료한 에피소드마다 `progress.jsonl`에 표지(버전·rollout 수·벽시계)를 적는다. worker 풀은 한 번
만들어 묶음 사이에 다시 쓰고, 풀이 한 묶음의 rollout을 도는 동안 주 프로세스가 다음 묶음을 재생한다(재생은 단일 프로세스라 겹치지 않으면 worker가 논다). `--resume`은 표지가 있는 에피소드를 건너뛰고(표지가 없는 부분 결과는 버린다 — `--limit`에
잘린 묶음은 표지를 쓰지 않는다) 남은 것만 돌린 뒤 전체에서 라벨·비용을 다시 만든다. 표지의 버전이 지금 것과 다르면
:class:`ConfigMismatch`다.

재생의 전제는 레코드를 만든 코드·설정과 지금 것이 같다는 것이다. 그래서 먼저 레코드의 `versions`(하네스·
컨트롤러·전문가 버전과 설정 묶음의 `config_digest`)를 지금 돌아가는 것과 맞대 보고, 다르면 무엇이 다른지 말하는
:class:`ConfigMismatch`로 그 레코드를 **거절**한다(조용히 건너뛰지 않는다). 그 다음 키프레임 틱에서 다시 만든
요청의 후보 집합 버전·commitment·실행 이력·물체가 레코드와 같은지(`fidelity`)를 보고, 다르면 그 키프레임은
건너뛰되 `skipped_fidelity`로 센다.
"""

from __future__ import annotations

import argparse
import contextlib
from collections import Counter
import copy
import json
import math
import os
import random
import statistics
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from robo_jev.contracts import QUESTION_SET_V0
from robo_jev.data.episode import default_versions
from robo_jev.data.robot_episodes import read_episodes
from robo_jev.harness.robot import FIXED_KEYS, RobotHarness, load_harness_config
from robo_jev.sim.expert import Expert, load_expert_config
from robo_jev.sim.label import (
    choose_rollout_candidates,
    event_for,
    label_main_decision,
    load_events_config,
    rollout_event,
    select_keyframes,
    summarise_results,
)

__all__ = [
    "ConfigMismatch",
    "VERSION_KEYS",
    "build_jobs",
    "costing",
    "job_pool",
    "main",
    "replay_to_keyframes",
    "run",
    "run_jobs",
    "running_versions_for",
    "summarise_sweep",
    "write_outputs",
]

#: 재개용 증분 파일 (출력 디렉터리 안).
PROGRESS_FILE = "progress.jsonl"
KEYFRAMES_INCREMENTAL = "keyframes.jsonl"
ROLLOUTS_FILE = "rollouts.jsonl"

ROLLOUTS_VERSION = "rollouts-v0.1"
_QUESTIONS = tuple(QUESTION_SET_V0)

#: 레코드와 지금 돌아가는 것이 같아야 하는 `versions` 키. 코드 버전(하네스·전문가·규칙 기준군·추출기), 설정의 버전
#: 문자열(컨트롤러·레코드 직렬화·장면), 그리고 설정 묶음의 지문이다. 재생은 관측 → 상태(추출기·레코드 직렬화) →
#: 요청 → 조합(하네스·규칙) → 실행(컨트롤러·장면)을 되풀이하므로 어느 하나가 올라가도 조용한 fidelity skip이 아니라
#: 거절이어야 한다.
VERSION_KEYS = ("harness", "controller", "expert", "rules", "serializer", "extractor", "sim", "config_digest")


class ConfigMismatch(ValueError):
    """레코드를 만든 코드·설정이 지금 것과 다르다 — 재생은 그 레코드를 거절한다."""


def running_versions_for(
    *, sim_config: str, events: dict[str, Any], generator_config: str | Path | dict[str, Any] | None = None
) -> dict[str, str]:
    """지금 돌아가는 코드·설정의 버전과 설정 묶음의 지문 (사건 설정이 가리키는 하네스·전문가 설정, 생성 설정의
    `episode.*` 손잡이 — 없으면 기본 생성 설정)."""
    from robo_jev.sim.expert import Expert, load_expert_config

    followup = events["followup"]
    versions = default_versions(
        harness_config=followup["harness_config"],
        expert_config=followup["expert_config"],
        sim_config=sim_config,
        events_config=events.get("_path"),
        generator_config=generator_config,
    )
    versions["expert"] = Expert(load_expert_config(followup["expert_config"])).version
    return {key: str(versions[key]) for key in VERSION_KEYS}


def check_record_versions(record: dict[str, Any], running: dict[str, str]) -> None:
    """레코드의 `versions`가 지금 것과 다르면 :class:`ConfigMismatch` — 무엇이 어떻게 다른지 적는다."""
    recorded = record.get("versions") or {}
    differences = []
    for key in VERSION_KEYS:
        if key not in recorded:
            differences.append(f"{key}: 레코드에 없음 (지금 {running[key]})")
        elif str(recorded[key]) != running[key]:
            differences.append(f"{key}: 레코드 {recorded[key]} ≠ 지금 {running[key]}")
    if differences:
        raise ConfigMismatch(
            f"{record.get('episode_id')}: 레코드를 만든 코드·설정이 지금과 다르다 — " + "; ".join(differences)
            + ". 배치를 지금 설정으로 다시 만들거나 그 설정으로 돌린다."
        )


# --------------------------------------------------------------------------
# 재생 — 레코드의 모델 출력을 같은 조합 규칙에 다시 먹여 키프레임의 snapshot을 얻는다
# --------------------------------------------------------------------------


def replay_to_keyframes(
    record: dict[str, Any],
    indices: Sequence[int],
    *,
    sim_config: str,
    harness_config: dict[str, Any],
    control_steps: int,
    env: Any | None = None,
    running: dict[str, str] | None = None,
) -> dict[int, dict[str, Any]]:
    """레코드를 키프레임 틱까지 재생해 `{index: {"snapshot", "commitment", "fidelity", "replay_s"}}`를 만든다.

    `running`(:func:`running_versions_for`)을 주면 먼저 레코드의 `versions`와 맞대 보고 다르면
    :class:`ConfigMismatch`로 거절한다. snapshot은 그 틱의 요청을 만든 관측의 상태다(명령을 적용하기 **전**).
    `commitment`는 하네스 장부까지 든 그 틱 시작의 commitment이고, `fidelity`는 다시 만든 요청이 레코드와
    같은지(후보 집합 버전·commitment·실행 이력·물체)를 말한다.
    """
    from robo_jev.sim.environment import Environment

    if running is not None:
        check_record_versions(record, running)
    wanted = sorted(set(int(index) for index in indices))
    if not wanted:
        return {}
    profile = str(record["provenance"]["profile"])
    seed = int(record["provenance"]["seed"])
    own_env = env is None
    if own_env:
        env = Environment(config_path=sim_config, profile=profile)
    started = time.perf_counter()
    found: dict[int, dict[str, Any]] = {}
    try:
        scene = env.reset(seed=seed)
        harness = RobotHarness(harness_config)
        commitment = None
        history = None
        last = wanted[-1]
        for index, tick in enumerate(record["ticks"]):
            if index > last:
                break
            request = harness.build_request(scene, history, commitment)
            if index in wanted:
                recorded = tick["request"]
                rebuilt = request["request"]
                fidelity = {
                    "candidate_set_version": recorded["state"]["t"].get("candidate_set_version") == rebuilt["state"]["t"].get("candidate_set_version"),
                    "commitment": recorded.get("commitment") == rebuilt.get("commitment"),
                    "exec_history": recorded.get("exec_history") == rebuilt.get("exec_history"),
                    "objects": recorded["state"]["objects"] == rebuilt["state"]["objects"],
                }
                found[index] = {
                    "snapshot": env.snapshot(),
                    "commitment": copy.deepcopy(commitment),
                    "fidelity": fidelity,
                    "exact": all(fidelity.values()),
                    "replay_s": round(time.perf_counter() - started, 4),
                    "precision_mm": {str(entry["id"]): int(entry.get("precision_mm", 3)) for entry in rebuilt["state"]["objects"]},
                }
            results = {question: copy.deepcopy(tick["model_output"][question]) for question in _QUESTIONS if question in tick["model_output"]}
            out = harness.compose(request, results, commitment, int(scene["sim_time_ms"]))
            ack = None
            for step in range(control_steps):
                scene = env.step(out["command"] if step == 0 else None)
                ack = scene["ack"] or ack
            commitment = out["commitment"]
            history = {"adopted": out["adopted"], "ack": ack, "gate": out["gate"]}
    finally:
        if own_env:
            env.close()
    return found


# --------------------------------------------------------------------------
# 작업 목록
# --------------------------------------------------------------------------


def build_jobs(
    records: list[dict[str, Any]],
    events: dict[str, Any],
    *,
    limit: int | None,
    per_episode: int | None = None,
    candidates_per_keyframe: int = 8,
    sim_config: str,
    harness_config: dict[str, Any],
    control_steps: int,
    log: Any = None,
    running: dict[str, str] | None = None,
    generator_config: str | Path | dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """키프레임을 고르고 재생해 rollout 작업 `(keyframe, candidate, seed)` 목록을 만든다.

    먼저 모든 레코드의 `versions`를 지금 것(`running`, 없으면 여기서 계산)과 맞대 보고 하나라도 다르면
    :class:`ConfigMismatch`로 멈춘다. 작업 순서는 (에피소드를 돌아가며 키프레임) → seed → 후보다. 그래서
    `limit`가 작아도 첫 키프레임은 모든 후보의 paired seed를 갖는다. 에피소드 안의 키프레임 순서는 에피소드 id로 seed한
    난수로 섞는다(리뷰 1 M1) — 정렬한 순서(t 오름차순)면 부분 sweep이 모든 에피소드의 t=0 `switch` 키프레임만 보게 된다;
    섞으면 부분 sweep의 종류 혼합이 전체의 표본이 된다. 재생이 레코드와 어긋난 키프레임은 건넌다(`skipped_fidelity`).
    """
    from robo_jev.sim.environment import Environment

    if running is None:
        running = running_versions_for(sim_config=sim_config, events=events, generator_config=generator_config)
    for record in records:
        check_record_versions(record, running)
    seeds = int(events.get("seeds", 8))
    config = {"keyframes": {"per_episode": per_episode or int((events.get("keyframes") or {}).get("per_episode", 5))}}
    selected: list[dict[str, Any]] = []
    for record in records:
        frames = select_keyframes(record, config)
        random.Random(f"jobs:{record.get('episode_id')}").shuffle(frames)  # 에피소드 안 순서는 seed한 무작위 (M1)
        for frame in frames:
            frame["record"] = record
            selected.append(frame)
    # 에피소드를 돌아가며: (에피소드 안 순번, 에피소드 순서)
    order: dict[str, int] = {}
    ranked = []
    for frame in selected:
        rank = order.get(frame["episode_id"], 0)
        order[frame["episode_id"]] = rank + 1
        ranked.append((rank, records.index(frame["record"]), frame))
    ranked.sort(key=lambda item: (item[0], item[1]))

    jobs: list[dict[str, Any]] = []
    keyframes_out: list[dict[str, Any]] = []
    envs: dict[str, Any] = {}
    replay_total = 0.0
    skipped_fidelity = 0
    # 에피소드의 키프레임은 **한 번의 재생**으로 전부 snapshot한다(처음 쓰이는 순간, 그 에피소드의 모든 키프레임 색인을 한 번에) —
    # 키프레임마다 reset부터 다시 돌리면 5배다. 재생 시간은 에피소드의 첫 키프레임에 적는다.
    indices_by_episode: dict[str, list[int]] = {}
    for _rank, _position, frame in ranked:
        indices_by_episode.setdefault(frame["episode_id"], []).append(int(frame["index"]))
    replays: dict[str, dict[int, dict[str, Any]]] = {}
    try:
        for _rank, _position, frame in ranked:
            if limit is not None and len(jobs) >= limit:
                break
            record = frame["record"]
            profile = str(record["provenance"]["profile"])
            env = envs.get(profile)
            if env is None:
                env = envs[profile] = Environment(config_path=sim_config, profile=profile)
            if frame["episode_id"] not in replays:
                replays[frame["episode_id"]] = replay_to_keyframes(
                    record, indices_by_episode[frame["episode_id"]], sim_config=sim_config, harness_config=harness_config,
                    control_steps=control_steps, env=env, running=running,
                )
                replay_total += max(item["replay_s"] for item in replays[frame["episode_id"]].values())
            replayed = replays[frame["episode_id"]][int(frame["index"])]
            tick = record["ticks"][frame["index"]]
            label = next((item for item in tick.get("labels") or () if item.get("question_id") == "q_main"), None)
            keys = {entry["id"]: str(entry.get("key", "")) for entry in tick["request"]["candidates"]["q_main"]}
            expert_choice = (label or {}).get("candidate_ids", [None])[0]
            if expert_choice is not None and keys.get(expert_choice) in FIXED_KEYS:
                expert_choice = None
            commitment = replayed["commitment"]
            chosen = choose_rollout_candidates(tick, commitment, expert_choice, k=candidates_per_keyframe, harness_config=harness_config)
            holding = tick["request"]["state"]["robot"].get("holding")
            entry = {key: value for key, value in frame.items() if key != "record"}
            entry.update(
                {
                    "profile": profile,
                    "seed": int(record["provenance"]["seed"]),
                    "split": record.get("split"),
                    "candidates": chosen,
                    "keys": {cid: keys[cid] for cid in chosen},
                    "commitment": (commitment or {}).get("action_ref"),
                    "expert_choice": expert_choice,
                    "holding": holding,
                    "fidelity": replayed["fidelity"],
                    "exact": replayed["exact"],
                    "replay_s": replayed["replay_s"],
                    "snapshot_bytes": len(replayed["snapshot"]),
                }
            )
            keyframes_out.append(entry)
            if not replayed["exact"]:
                skipped_fidelity += 1
                if log is not None:
                    print(f"  skip {frame['episode_id']} t={frame['t']}: replay differs {replayed['fidelity']}", file=log, flush=True)
                continue
            keyframe_id = f"{frame['episode_id']}@{frame['t']}"
            for seed in range(seeds):
                for cid in chosen:
                    if limit is not None and len(jobs) >= limit:
                        break
                    key = keys[cid]
                    jobs.append(
                        {
                            "keyframe": keyframe_id,
                            "episode_id": frame["episode_id"],
                            "index": frame["index"],
                            "t": frame["t"],
                            "profile": profile,
                            "snapshot": replayed["snapshot"],
                            "action": {"id": cid, "action_ref": cid, "key": key, "precision_mm": replayed["precision_mm"]},
                            "event": event_for(key, holding=holding, config={**events, "sim_config": sim_config}),
                            "seed": seed,
                        }
                    )
            if log is not None:
                print(
                    f"  keyframe {keyframe_id} kind={frame['kind']} candidates={len(chosen)} jobs={len(jobs)} replay={replayed['replay_s']:.2f}s",
                    file=log,
                    flush=True,
                )
    finally:
        for env in envs.values():
            env.close()
    summary = {
        "keyframes": len(keyframes_out),
        "skipped_fidelity": skipped_fidelity,
        "replay_s_total": round(replay_total, 3),
        "running_versions": dict(running),
    }
    return jobs, keyframes_out, summary


# --------------------------------------------------------------------------
# 실행 — 직렬 또는 worker 풀 (worker마다 환경 하나를 오래 쓴다)
# --------------------------------------------------------------------------

_WORKER: dict[str, Any] = {}


def _worker_init(sim_config: str, expert_config: str) -> None:
    _WORKER["sim_config"] = sim_config
    _WORKER["expert"] = Expert(load_expert_config(expert_config))
    _WORKER["envs"] = {}
    _WORKER["rebuild_s"] = {}


def _worker_run(job: dict[str, Any]) -> dict[str, Any]:
    from robo_jev.sim.environment import Environment

    profile = str(job["profile"])
    env = _WORKER["envs"].get(profile)
    rebuild_s = None
    if env is None:
        started = time.perf_counter()
        env = _WORKER["envs"][profile] = Environment(config_path=_WORKER["sim_config"], profile=profile)
        env.restore(job["snapshot"])  # 첫 복원이 모델을 짓는다 — 재구축 비용
        rebuild_s = round(time.perf_counter() - started, 4)
    result = rollout_event(job["snapshot"], job["action"], {**job["event"], "sim_config": _WORKER["sim_config"]}, job["seed"], env=env, expert=_WORKER["expert"])
    result["job"] = {key: job[key] for key in ("keyframe", "episode_id", "index", "t", "profile", "seed")}
    result["job"]["candidate"] = job["action"]["id"]
    result["job"]["key"] = job["action"]["key"]
    result["job"]["worker_pid"] = os.getpid()
    result["job"]["env_rebuild_s"] = rebuild_s
    return result


@contextlib.contextmanager
def job_pool(workers: int, *, sim_config: str, expert_config: str):
    """worker 풀(spawn, worker마다 환경 하나)을 한 번 만들어 여러 :func:`run_jobs` 호출에 다시 쓴다. `workers <= 1`이면 `None`(직렬)."""
    if workers <= 1:
        yield None
        return
    import multiprocessing

    context = multiprocessing.get_context("spawn")
    pool = context.Pool(workers, initializer=_worker_init, initargs=(sim_config, expert_config))
    try:
        yield pool
    finally:
        pool.close()
        pool.join()


def run_jobs(
    jobs: list[dict[str, Any]], *, sim_config: str, expert_config: str, workers: int = 1, log: Any = None, pool: Any = None
) -> list[dict[str, Any]]:
    """job 목록을 돌린다. `pool`(:func:`job_pool`)을 주면 그 풀로, 없으면 `workers`에 따라 직렬 또는 이 호출만의 풀로."""
    if pool is None and workers > 1:
        with job_pool(workers, sim_config=sim_config, expert_config=expert_config) as own:
            return run_jobs(jobs, sim_config=sim_config, expert_config=expert_config, workers=workers, log=log, pool=own)
    if pool is None:
        _worker_init(sim_config, expert_config)
        results = []
        try:
            for number, job in enumerate(jobs, start=1):
                result = _worker_run(job)
                results.append(result)
                if log is not None and (number % 10 == 0 or number == len(jobs)):
                    print(f"  rollout {number}/{len(jobs)} {result['job']['key']} seed={job['seed']} → {result['outcome']} ({result['evidence']['wall_s']}s)", file=log, flush=True)
        finally:
            for env in _WORKER.get("envs", {}).values():
                env.close()
            _WORKER.clear()
        return results

    results = []
    for number, result in enumerate(pool.imap(_worker_run, jobs, chunksize=1), start=1):
        results.append(result)
        if log is not None and (number % 10 == 0 or number == len(jobs)):
            print(f"  rollout {number}/{len(jobs)} {result['job']['key']} seed={result['job']['seed']} → {result['outcome']} ({result['evidence']['wall_s']}s)", file=log, flush=True)
    return results


# --------------------------------------------------------------------------
# 라벨과 비용
# --------------------------------------------------------------------------


def label_keyframes(records: list[dict[str, Any]], keyframes: list[dict[str, Any]], results: list[dict[str, Any]], events: dict[str, Any]) -> list[dict[str, Any]]:
    by_id = {record["episode_id"]: record for record in records}
    grouped: dict[str, list[dict[str, Any]]] = {}
    for result in results:
        grouped.setdefault(result["job"]["keyframe"], []).append(result)
    labels = []
    for frame in keyframes:
        keyframe_id = f"{frame['episode_id']}@{frame['t']}"
        rollouts = grouped.get(keyframe_id)
        if not rollouts:
            continue
        tick = by_id[frame["episode_id"]]["ticks"][frame["index"]]
        summary = summarise_results(rollouts)
        label = label_main_decision(tick, summary, events)
        labels.append(
            {
                "episode_id": frame["episode_id"],
                "index": frame["index"],
                "t": frame["t"],
                "kind": frame["kind"],
                "split": frame.get("split"),
                "rollouts": len(rollouts),
                "results": summary,
                "label": label,
            }
        )
    return labels


def _percentile(values: Sequence[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = min(len(ordered) - 1, max(0, int(round(fraction * (len(ordered) - 1)))))
    return float(ordered[position])


def costing(
    results: list[dict[str, Any]],
    *,
    batch_wall_s: float,
    replay_s_total: float,
    workers: int,
    targets: dict[str, int] | None = None,
) -> dict[str, Any]:
    """첫 rollout들의 실측 → 128,000(D1)·1,280,000/2,560,000(D2)의 CPU 시간 추정.

    `cpu_hours`는 rollout당 벽시계(복원 포함, worker 하나 기준)의 합이고 `wall_hours_parallel`은 그것을 실제로 쓴
    worker 수로 나눈 것이다. 재생(키프레임당 한 번)은 따로 적는다 — 키프레임당 64개 rollout에 분할된다.
    """
    walls = [float(result["evidence"]["wall_s"]) for result in results]
    restores = [float(result["evidence"]["restore_s"]) for result in results if result["evidence"].get("restore_s") is not None]
    ticks = [int(result["evidence"]["ticks"]) for result in results]
    sizes = [len(json.dumps(result, ensure_ascii=False, separators=(",", ":")).encode("utf-8")) for result in results]
    rebuilds = [float(result["job"]["env_rebuild_s"]) for result in results if result["job"].get("env_rebuild_s") is not None]
    outcomes: dict[str, int] = {}
    reasons: dict[str, int] = {}
    by_function: dict[str, dict[str, int]] = {}
    for result in results:
        outcomes[result["outcome"]] = outcomes.get(result["outcome"], 0) + 1
        if result["reason"]:
            reasons[str(result["reason"])] = reasons.get(str(result["reason"]), 0) + 1
        function = str(result["job"]["key"]).split(":")[0]
        entry = by_function.setdefault(function, {"rollouts": 0, "success": 0, "failure": 0, "censored": 0})
        entry["rollouts"] += 1
        entry[result["outcome"]] += 1
    mean_wall = statistics.fmean(walls) if walls else 0.0
    cores = os.cpu_count() or 1
    targets = targets or {"d1_128k": 128_000, "d2_1_28m": 1_280_000, "d2_2_56m": 2_560_000}
    projections = {}
    for name, count in targets.items():
        cpu_hours = count * mean_wall / 3600.0
        projections[name] = {
            "rollouts": count,
            "cpu_hours": round(cpu_hours, 2),
            "wall_hours_with_workers_used": round(cpu_hours / max(1, workers), 2),
            "wall_hours_with_all_cores": round(cpu_hours / cores, 2),
            "sim_hours": round(count * statistics.fmean(ticks) * 0.1 / 3600.0, 2) if ticks else 0.0,
            "storage_gb": round(count * (statistics.fmean(sizes) if sizes else 0.0) / 1e9, 3),
        }
    return {
        "version": ROLLOUTS_VERSION,
        "rollouts": len(results),
        "machine": {"cpu_count": cores, "workers_used": workers, "platform": sys.platform},
        "outcomes": dict(sorted(outcomes.items())),
        "reasons": dict(sorted(reasons.items())),
        "by_function": dict(sorted(by_function.items())),
        "wall_s_per_rollout": {
            "mean": round(mean_wall, 4),
            "p50": round(_percentile(walls, 0.5), 4),
            "p95": round(_percentile(walls, 0.95), 4),
            "max": round(max(walls), 4) if walls else 0.0,
        },
        "restore_s_per_rollout": {"mean": round(statistics.fmean(restores), 4) if restores else None, "p95": round(_percentile(restores, 0.95), 4) if restores else None},
        "env_rebuild_s": {"count": len(rebuilds), "mean": round(statistics.fmean(rebuilds), 4) if rebuilds else None},
        "ticks_per_rollout": {"mean": round(statistics.fmean(ticks), 2) if ticks else 0.0, "max": max(ticks) if ticks else 0},
        "sim_seconds_per_rollout_mean": round(statistics.fmean(ticks) * 0.1, 3) if ticks else 0.0,
        "bytes_per_rollout": {"mean": round(statistics.fmean(sizes), 1) if sizes else 0.0, "max": max(sizes) if sizes else 0},
        "replay": {"total_s": round(replay_s_total, 3)},
        "batch_wall_s": round(batch_wall_s, 3),
        "throughput": {
            "rollouts_per_hour_per_worker": round(3600.0 / mean_wall, 1) if mean_wall else 0.0,
            "rollouts_per_hour_batch": round(len(results) / batch_wall_s * 3600.0, 1) if batch_wall_s else 0.0,
        },
        "projections": projections,
    }


# --------------------------------------------------------------------------
# sweep 요약 — 128k 전 관문 (ledger: 접근 시간·시작 거리로 조건화한 밀기 성공률, censoring 사유, 키프레임 종류 혼합)
# --------------------------------------------------------------------------

#: 접근 시간(s)·시작 거리(mm; 키프레임의 말단→대상 중심)의 구간 경계.
_APPROACH_EDGES_S = (1.0, 2.0, 3.0, 4.0)
_DISTANCE_EDGES_MM = (150, 250, 350, 500)


def _bucket(value: float | None, edges: Sequence[float]) -> str:
    if value is None:
        return "none"
    for edge in edges:
        if value <= edge:
            return f"<={edge:g}"
    return f">{edges[-1]:g}"


def _rate_table(groups: dict[str, dict[str, int]]) -> dict[str, dict[str, Any]]:
    """구간 → {rollouts, success, failure, censored, rate, wilson} (유효 반복 = censoring을 뺀 수, z=1)."""
    from robo_jev.sim.label import wilson_interval

    table: dict[str, dict[str, Any]] = {}
    for name, counts in sorted(groups.items()):
        successes = int(counts.get("success", 0))
        trials = successes + int(counts.get("failure", 0))
        low, high = wilson_interval(successes, trials)
        table[name] = {
            "rollouts": sum(counts.values()),
            "success": successes,
            "failure": int(counts.get("failure", 0)),
            "censored": int(counts.get("censored", 0)),
            "rate": round(successes / trials, 3) if trials else None,
            "wilson": [round(low, 3), round(high, 3)],
        }
    return table


def summarise_sweep(out: Path) -> dict[str, Any]:
    """`rollouts/` 출력(keyframes·rollouts·labels·costing)에서 128k 전 관문의 표를 만든다 (`sweep-summary.json`으로도 쓴다).

    사건별 결과, **밀기 성공률을 접근 시간(`approach_s`)·시작 거리(키프레임의 말단→대상 중심)·방향·키프레임 종류로 조건화**한
    표(Wilson 구간), 밀기 `contact_force` 실패의 시작 거리 분포, 파지 성공률의 시작 거리 표, censoring 사유, 키프레임 종류 혼합,
    라벨 신뢰도, rollout당 벽시계와 128k·D2 산정을 한 dict에 모은다. rollout을 다시 돌리지 않는다.
    """
    root = Path(out)
    results = [json.loads(line) for line in (root / "rollouts.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    keyframes = json.loads((root / "keyframes.json").read_text(encoding="utf-8"))
    labels = [json.loads(line) for line in (root / "labels.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    cost = json.loads((root / "costing.json").read_text(encoding="utf-8"))

    kind_of = {f"{frame['episode_id']}@{frame['t']}": str(frame["kind"]) for frame in keyframes}
    holding_of = {f"{frame['episode_id']}@{frame['t']}": frame.get("holding") for frame in keyframes}
    by_event: dict[str, dict[str, int]] = {}
    push_by_approach: dict[str, dict[str, int]] = {}
    push_reasons_by_approach: dict[str, dict[str, int]] = {}
    push_stage_by_direction: dict[str, dict[str, int]] = {}
    push_by_distance: dict[str, dict[str, int]] = {}
    push_by_direction: dict[str, dict[str, int]] = {}
    push_by_kind: dict[str, dict[str, int]] = {}
    grasp_by_distance: dict[str, dict[str, int]] = {}
    place_by_kind: dict[str, dict[str, int]] = {}
    contact_force_by_distance: dict[str, int] = {}
    reasons: dict[str, int] = {}
    censoring: dict[str, int] = {}
    approach_seconds: list[float] = []

    def bump(table: dict[str, dict[str, int]], name: str, outcome: str) -> None:
        table.setdefault(name, {})[outcome] = table.setdefault(name, {}).get(outcome, 0) + 1

    for result in results:
        evidence = result["evidence"]
        key = str(result["job"]["key"])
        parts = key.split(":")
        keyframe = str(result["job"]["keyframe"])
        holding = holding_of.get(keyframe)
        event = "place" if parts[0] == "place" or (parts[0] == "grasp" and holding == parts[1]) else parts[0]
        outcome = str(result["outcome"])
        bump(by_event, event, outcome)
        if result.get("reason"):
            reasons[f"{event}:{result['reason']}"] = reasons.get(f"{event}:{result['reason']}", 0) + 1
        if outcome == "censored":
            censoring[str(result["reason"])] = censoring.get(str(result["reason"]), 0) + 1
        trajectory = evidence.get("trajectory") or []
        start_ee = trajectory[0][1:4] if trajectory else None
        start_object = evidence.get("start_pose_mm")
        distance = math.dist(start_ee, start_object) if start_ee and start_object else None
        kind = kind_of.get(keyframe, "?")
        if event == "push":
            approach = evidence.get("approach_s")
            bump(push_by_approach, _bucket(approach, _APPROACH_EDGES_S), outcome)
            # 실패 이유를 접근 구간마다, 그리고 방향마다 단계(접근 중 충돌 / 밀기 중 충돌 / horizon)로 가른다 (리뷰 1 I5):
            # `approach_s`가 없는 rollout은 접촉점에 닿기 전에 끝난 것이고, 그 대부분은 내려가다 물체를 친 `contact_force`다.
            stage = "approach" if evidence.get("first_action_tick") is None else "push"
            label = outcome if outcome != "failure" else f"{stage}_{result.get('reason')}"
            bump(push_reasons_by_approach, _bucket(approach, _APPROACH_EDGES_S), label)
            bump(push_stage_by_direction, parts[2], label)
            bump(push_by_distance, _bucket(distance, _DISTANCE_EDGES_MM), outcome)
            bump(push_by_direction, parts[2], outcome)
            bump(push_by_kind, kind, outcome)
            if approach is not None:
                approach_seconds.append(float(approach))
            if result.get("reason") == "contact_force":
                name = _bucket(distance, _DISTANCE_EDGES_MM)
                contact_force_by_distance[name] = contact_force_by_distance.get(name, 0) + 1
        elif event == "grasp":
            bump(grasp_by_distance, _bucket(distance, _DISTANCE_EDGES_MM), outcome)
        else:
            bump(place_by_kind, kind, outcome)

    kinds: dict[str, int] = {}
    for frame in keyframes:
        kinds[str(frame["kind"])] = kinds.get(str(frame["kind"]), 0) + 1
    confidence: dict[str, int] = {}
    rollout_reason: dict[str, int] = {}
    for entry in labels:
        label = entry["label"]
        confidence[str(label.get("label_confidence"))] = confidence.get(str(label.get("label_confidence")), 0) + 1
        rollout_reason[str(label.get("rollout_reason"))] = rollout_reason.get(str(label.get("rollout_reason")), 0) + 1

    summary = {
        "version": ROLLOUTS_VERSION,
        "rollouts": len(results),
        "keyframes": {
            "count": len(keyframes),
            "exact": sum(1 for frame in keyframes if frame.get("exact")),
            "kinds": dict(sorted(kinds.items())),
            "labelled": len(labels),
        },
        "outcomes": cost.get("outcomes"),
        "by_event": {name: dict(sorted(counts.items())) for name, counts in sorted(by_event.items())},
        "reasons": dict(sorted(reasons.items(), key=lambda item: (-item[1], item[0]))),
        "censoring": dict(sorted(censoring.items())),
        "push_by_approach_s": _rate_table(push_by_approach),
        "push_reasons_by_approach_s": {name: dict(sorted(counts.items())) for name, counts in sorted(push_reasons_by_approach.items())},
        "push_stage_by_direction": {name: dict(sorted(counts.items())) for name, counts in sorted(push_stage_by_direction.items())},
        "push_by_start_distance_mm": _rate_table(push_by_distance),
        "push_by_direction": _rate_table(push_by_direction),
        "push_by_keyframe_kind": _rate_table(push_by_kind),
        "push_contact_force_failures_by_start_distance_mm": dict(sorted(contact_force_by_distance.items())),
        "push_approach_s": {
            "n": len(approach_seconds),
            "mean": round(statistics.fmean(approach_seconds), 2) if approach_seconds else None,
            "p50": round(statistics.median(approach_seconds), 2) if approach_seconds else None,
        },
        "grasp_by_start_distance_mm": _rate_table(grasp_by_distance),
        "place_by_keyframe_kind": _rate_table(place_by_kind),
        "labels": {"confidence": dict(sorted(confidence.items())), "rollout_reason": dict(sorted(rollout_reason.items()))},
        "wall_s_per_rollout": cost.get("wall_s_per_rollout"),
        "restore_s_per_rollout": cost.get("restore_s_per_rollout"),
        "throughput": cost.get("throughput"),
        "batch_wall_s": cost.get("batch_wall_s"),
        "replay": cost.get("replay"),
        "projections": cost.get("projections"),
        "machine": cost.get("machine"),
        "versions": cost.get("versions"),
    }
    (root / "sweep-summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def write_outputs(out: Path, *, keyframes: list[dict[str, Any]], results: list[dict[str, Any]] | None, labels: list[dict[str, Any]], cost: dict[str, Any]) -> dict[str, Path]:
    """`results=None`이면 `rollouts.jsonl`은 이미 증분으로 써진 것으로 보고 다시 쓰지 않는다."""
    out.mkdir(parents=True, exist_ok=True)
    paths = {
        "keyframes": out / "keyframes.json",
        "rollouts": out / ROLLOUTS_FILE,
        "labels": out / "labels.jsonl",
        "costing": out / "costing.json",
    }
    paths["keyframes"].write_text(json.dumps(keyframes, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if results is not None:
        with paths["rollouts"].open("w", encoding="utf-8") as handle:
            for result in results:
                handle.write(json.dumps(result, ensure_ascii=False, separators=(",", ":")) + "\n")
    with paths["labels"].open("w", encoding="utf-8") as handle:
        for entry in labels:
            handle.write(json.dumps(entry, ensure_ascii=False, separators=(",", ":")) + "\n")
    paths["costing"].write_text(json.dumps(cost, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return paths


# --------------------------------------------------------------------------
# 전체
# --------------------------------------------------------------------------


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _append_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def _load_progress(out: Path, running: dict[str, str]) -> dict[str, Any]:
    """재개: `progress.jsonl`의 표지(완료 에피소드)를 읽고 증분 파일에서 그 에피소드의 결과·키프레임만 남긴다(부분 결과는 버리고
    파일을 압축해 다시 쓴다). 표지의 버전이 지금 것과 다르면 :class:`ConfigMismatch`."""
    markers = _read_jsonl(out / PROGRESS_FILE)
    done: dict[str, dict[str, Any]] = {}
    for marker in markers:
        recorded = marker.get("running_versions") or {}
        differences = [f"{key}: 표지 {recorded.get(key)} ≠ 지금 {running[key]}" for key in VERSION_KEYS if str(recorded.get(key)) != running[key]]
        if differences:
            raise ConfigMismatch(f"{out / PROGRESS_FILE}: 앞선 실행의 버전이 지금과 다르다 — " + "; ".join(differences))
        done[str(marker["episode_id"])] = marker
    keyframes = [frame for frame in _read_jsonl(out / KEYFRAMES_INCREMENTAL) if frame["episode_id"] in done]
    results = [result for result in _read_jsonl(out / ROLLOUTS_FILE) if result["job"]["episode_id"] in done]
    _write_jsonl(out / KEYFRAMES_INCREMENTAL, keyframes)
    _write_jsonl(out / ROLLOUTS_FILE, results)
    return {
        "done": done,
        "keyframes": keyframes,
        "results": results,
        "wall_s": sum(float(marker.get("wall_s", 0.0)) for marker in done.values()),
        "replay_s": sum(float(marker.get("replay_s", 0.0)) for marker in done.values()),
        "skipped_fidelity": sum(int(marker.get("skipped_fidelity", 0)) for marker in done.values()),
    }


def run(
    dataset: Path,
    *,
    limit: int | None,
    workers: int = 1,
    events_path: str = "configs/sim/events.yaml",
    sim_config: str = "configs/sim/tidy_clutter.yaml",
    out: Path | None = None,
    per_episode: int | None = None,
    candidates_per_keyframe: int = 8,
    log: Any = None,
    generator_config: str | Path = "configs/data/d1_robot.yaml",
    resume: bool = False,
    chunk_episodes: int = 8,
    events_override: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """배치의 키프레임 rollout 전부(`limit=None`) 또는 앞 `limit`개를 돌려 `out`(기본 `<dataset>/rollouts`)에 쓴다 — 에피소드
    `chunk_episodes`개 묶음마다 증분 파일에 덧붙이고 표지를 적어 `resume=True`로 이어 돌릴 수 있다(모듈 docstring)."""
    started = time.perf_counter()
    events = {**load_events_config(events_path), "_path": str(events_path), **(events_override or {})}
    harness_config = load_harness_config(events["followup"]["harness_config"])
    control_steps = int(events["followup"]["control_steps_per_tick"])
    records = [record for _, record in read_episodes(dataset)]
    if not records:
        raise FileNotFoundError(f"에피소드가 없다: {dataset}")
    out = out or (dataset / "rollouts")
    out.mkdir(parents=True, exist_ok=True)
    running = running_versions_for(sim_config=sim_config, events=events, generator_config=generator_config)
    for record in records:
        check_record_versions(record, running)

    if resume:
        prior = _load_progress(out, running)
    else:
        for name in (PROGRESS_FILE, KEYFRAMES_INCREMENTAL, ROLLOUTS_FILE):
            (out / name).unlink(missing_ok=True)
        prior = {"done": {}, "keyframes": [], "results": [], "wall_s": 0.0, "replay_s": 0.0, "skipped_fidelity": 0}
    keyframes: list[dict[str, Any]] = list(prior["keyframes"])
    results: list[dict[str, Any]] = list(prior["results"])
    replay_s_total = float(prior["replay_s"])
    skipped_fidelity = int(prior["skipped_fidelity"])
    pending = [record for record in records if record["episode_id"] not in prior["done"]]
    if log is not None:
        print(f"rollouts: {len(records)} episodes, {len(prior['done'])} done before, {len(pending)} pending, {len(results)} results kept", file=log, flush=True)

    step = max(1, int(chunk_episodes))
    chunks = [pending[start : start + step] for start in range(0, len(pending), step)]

    def persist(number: int, chunk: list[dict[str, Any]], jobs: list[dict[str, Any]], frames: list[dict[str, Any]], summary: dict[str, Any], chunk_results: list[dict[str, Any]], cut: bool, started_at: float) -> None:
        nonlocal replay_s_total, skipped_fidelity
        replay_s_total += float(summary["replay_s_total"])
        skipped_fidelity += int(summary["skipped_fidelity"])
        keyframes.extend(frames)
        results.extend(chunk_results)
        _append_jsonl(out / KEYFRAMES_INCREMENTAL, frames)
        _append_jsonl(out / ROLLOUTS_FILE, chunk_results)
        if not cut:  # `limit`에 잘린 묶음: 완료 표지를 쓰지 않는다
            wall = time.perf_counter() - started_at
            _append_jsonl(
                out / PROGRESS_FILE,
                [
                    {
                        "episode_id": record["episode_id"],
                        "keyframes": sum(1 for frame in frames if frame["episode_id"] == record["episode_id"]),
                        "rollouts": sum(1 for result in chunk_results if result["job"]["episode_id"] == record["episode_id"]),
                        "wall_s": round(wall / len(chunk), 3),
                        "replay_s": round(float(summary["replay_s_total"]) / len(chunk), 3),
                        "skipped_fidelity": sum(1 for frame in frames if frame["episode_id"] == record["episode_id"] and not frame.get("exact")),
                        "running_versions": dict(running),
                        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    }
                    for record in chunk
                ],
            )
        if log is not None:
            outcomes = Counter(result["outcome"] for result in chunk_results)
            print(f"chunk {number}/{len(chunks)}: {len(chunk)} episodes, {len(jobs)} jobs ({dict(outcomes)}), {len(results)} results so far ({time.perf_counter() - started_at:.0f}s){' [limit]' if cut else ''}", file=log, flush=True)

    def build(chunk: list[dict[str, Any]], remaining: int | None):
        return build_jobs(
            chunk, events, limit=remaining, per_episode=per_episode, candidates_per_keyframe=candidates_per_keyframe, sim_config=sim_config,
            harness_config=harness_config, control_steps=control_steps, log=log, running=running, generator_config=generator_config,
        )

    with job_pool(workers, sim_config=sim_config, expert_config=events["followup"]["expert_config"]) as pool:
        in_flight: tuple | None = None  # (number, chunk, jobs, frames, summary, async_result, cut, started_at) — 풀이 도는 동안 다음 묶음을 재생한다
        for number, chunk in enumerate(chunks, start=1):
            planned = len(results) + (len(in_flight[2]) if in_flight else 0)
            remaining = None if limit is None else int(limit) - planned
            if remaining is not None and remaining <= 0:
                break
            chunk_started = time.perf_counter()
            jobs, frames, summary = build(chunk, remaining)  # 재생(주 프로세스) — 풀은 앞 묶음의 rollout을 돌리고 있다
            cut = remaining is not None and len(jobs) >= remaining
            if pool is None:
                chunk_results = run_jobs(jobs, sim_config=sim_config, expert_config=events["followup"]["expert_config"], workers=1, log=log)
                persist(number, chunk, jobs, frames, summary, chunk_results, cut, chunk_started)
            else:
                if in_flight is not None:
                    prev_number, prev_chunk, prev_jobs, prev_frames, prev_summary, async_result, prev_cut, prev_started = in_flight
                    persist(prev_number, prev_chunk, prev_jobs, prev_frames, prev_summary, async_result.get(), prev_cut, prev_started)
                    in_flight = None
                async_result = pool.map_async(_worker_run, jobs, chunksize=1)
                in_flight = (number, chunk, jobs, frames, summary, async_result, cut, chunk_started)
            if cut:
                break
        if in_flight is not None:
            prev_number, prev_chunk, prev_jobs, prev_frames, prev_summary, async_result, prev_cut, prev_started = in_flight
            persist(prev_number, prev_chunk, prev_jobs, prev_frames, prev_summary, async_result.get(), prev_cut, prev_started)

    labels = label_keyframes(records, keyframes, results, events)
    batch_wall_s = (time.perf_counter() - started) + float(prior["wall_s"])  # 재개면 앞선 실행의 묶음 벽시계를 더한다
    cost = costing(results, batch_wall_s=batch_wall_s, replay_s_total=replay_s_total, workers=workers)
    cost["keyframes"] = {
        "keyframes": len(keyframes), "skipped_fidelity": skipped_fidelity, "replay_s_total": round(replay_s_total, 3), "running_versions": dict(running),
        "labelled": len(labels), "events_version": events["version"], "episodes": len(records), "episodes_done": len(_read_jsonl(out / PROGRESS_FILE)),
        "resumed": bool(resume), "chunk_episodes": int(chunk_episodes),
    }
    cost["versions"] = dict(running)
    paths = write_outputs(out, keyframes=keyframes, results=None, labels=labels, cost=cost)
    return {"jobs": len(results) - len(prior["results"]), "results": results, "labels": labels, "costing": cost, "paths": paths, "keyframes": keyframes}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="scripts/rollout_keyframes.py", description="키프레임 rollout을 돌리고 라벨·비용을 적는다 (docs/04 §4).")
    parser.add_argument("--dataset", type=Path, default=Path("artifacts/datasets/d1-robot/batch-0"))
    parser.add_argument("--limit", type=int, default=100, help="돌릴 rollout 수 (비용 산정용 첫 묶음); 0이면 전부")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--resume", action="store_true", help="출력 디렉터리의 progress.jsonl에 표지가 있는 에피소드는 건너뛴다")
    parser.add_argument("--chunk-episodes", type=int, default=8, help="증분 저장·표지의 단위(에피소드 수)")
    parser.add_argument("--events", default="configs/sim/events.yaml")
    parser.add_argument("--out", type=Path, default=None, help="기본 <dataset>/rollouts")
    parser.add_argument("--generator-config", default="configs/data/d1_robot.yaml", help="레코드를 만든 생성 설정 (episode.* 손잡이가 지문에 든다)")
    parser.add_argument("--summarise", type=Path, default=None, help="rollout을 돌리지 않고 이 출력 디렉터리의 sweep 요약(sweep-summary.json)만 만든다")
    args = parser.parse_args(argv)
    if args.summarise is not None:
        summary = summarise_sweep(args.summarise)
        print(json.dumps({key: summary[key] for key in ("rollouts", "keyframes", "by_event", "censoring", "push_by_approach_s", "push_by_start_distance_mm", "wall_s_per_rollout", "throughput")}, ensure_ascii=False, indent=2))
        print(f"→ {Path(args.summarise) / 'sweep-summary.json'}")
        return 0
    outcome = run(
        args.dataset, limit=(None if args.limit <= 0 else args.limit), workers=args.workers, events_path=args.events, out=args.out, log=sys.stdout,
        generator_config=args.generator_config, resume=args.resume, chunk_episodes=args.chunk_episodes,
    )
    cost = outcome["costing"]
    print(json.dumps({key: cost[key] for key in ("rollouts", "outcomes", "wall_s_per_rollout", "restore_s_per_rollout", "env_rebuild_s", "bytes_per_rollout", "throughput", "projections", "keyframes")}, ensure_ascii=False, indent=2))
    print(f"→ {outcome['paths']['costing'].parent}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
