"""밀기 접촉 거리 A/B 드라이버 — 같은 키프레임·snapshot·후보·paired seed의 밀기 job을 하네스 설정만 다른 팔(arm)마다 돌려
방향·접촉 부위별 단계 실패 표를 만든다 (D1-prep 리뷰 2 N3·N7; `scripts/push_contact_ab.py`).

    uv run python scripts/push_contact_ab.py --dataset artifacts/datasets/d1-robot/batch-0 --workers 8 \\
        --arm 'old_30mm={"candidates": {"push_contact_mm": 30, "push_reach_mm": null}}' \\
        --arm 'h0.6={"candidates": {"push_reach_mm": null}}' --arm 'h0.7={}' \\
        --out artifacts/reports/d1-robot-push-contact-ab.json

흐름: 배치의 레코드를 지금 돌아가는 설정과 맞대 보고(:func:`robo_jev.data.rollouts.build_jobs`의 `ConfigMismatch` — 재생은 배치를
만든 설정으로만 한다) 모든 키프레임을 재생해 job을 만든 뒤 **밀기 job만** 남긴다. 팔마다 하네스 설정에 override(중첩 dict; `null`은
키 삭제)를 적용한 yaml과 그것을 가리키는 전문가 yaml을 `--scratch` 아래에 쓰고, job의 사건(`event.harness_config`)과 worker의 전문가
설정을 그 팔의 파일로 바꿔 :func:`run_jobs`로 돌린다 — snapshot은 물리 상태라 팔과 무관하고, 후속 정책(전문가 + 하네스)만 팔을 따른다.
결과 JSON은 명령줄·팔별 override·설정 sha256·방향별(`by_direction`)·접촉 부위별(`by_class`: fingers/hand × 방향) 단계 표(접근 중 /
밀기 중 × contact_force / horizon, 성공, censored)와 `push_horizon` 실패의 변위 중앙값을 적는다. 옛 A/B(리뷰 2 §b)는 스크립트 없이
돌아 재현할 수 없었다 — 이 파일이 그 드라이버다.
"""

from __future__ import annotations

import copy
import hashlib
import json
import statistics
import time
from pathlib import Path
from typing import Any

import yaml

from robo_jev.data.robot_episodes import read_episodes
from robo_jev.data.rollouts import ROLLOUTS_VERSION, build_jobs, run_jobs
from robo_jev.harness.robot import joint_key_parts, load_harness_config, push_contact_class
from robo_jev.sim.expert import load_expert_config
from robo_jev.sim.label import load_events_config

__all__ = ["apply_override", "push_jobs", "run", "summarise_arm", "write_arm_configs"]

DEFAULT_SCRATCH = Path("artifacts/scratch/d1/push-ab")
_STAGES = ("approach_contact_force", "approach_horizon", "push_contact_force", "push_horizon")


def apply_override(config: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """중첩 override를 설정에 얹는다. dict는 재귀로 합치고, `None`은 그 키를 지운다, 그 밖의 값은 덮어쓴다."""
    out = copy.deepcopy(config)
    for key, value in (override or {}).items():
        if value is None:
            out.pop(key, None)
        elif isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = apply_override(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def write_arm_configs(
    name: str, override: dict[str, Any], scratch: Path, *, harness_path: str, expert_path: str
) -> dict[str, Any]:
    """팔의 하네스 yaml(override 적용)과 그것을 가리키는 전문가 yaml을 쓴다. 돌려주는 것은 경로와 sha256."""
    root = Path(scratch) / name
    root.mkdir(parents=True, exist_ok=True)
    harness = apply_override(load_harness_config(harness_path), override)
    harness_file = root / "robot.yaml"
    harness_text = yaml.safe_dump(harness, allow_unicode=True, sort_keys=False)
    harness_file.write_text(harness_text, encoding="utf-8")
    expert = load_expert_config(expert_path)
    expert["harness_config"] = str(harness_file.resolve())
    expert_file = root / "expert.yaml"
    expert_file.write_text(yaml.safe_dump(expert, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return {
        "name": name,
        "override": copy.deepcopy(override),
        "harness_config": str(harness_file),
        "expert_config": str(expert_file),
        "harness_sha256": hashlib.sha256(harness_text.encode("utf-8")).hexdigest(),
        "harness_version": str(harness.get("version")),
    }


def _job_class(job: dict[str, Any], records_by_id: dict[str, dict[str, Any]], spec: dict[str, Any]) -> str:
    """job의 밀기 대상이 키프레임 상태에서 손가락·손몸통 어느 쪽으로 밀리는가 (기록·요약용; 팔과 무관하게 배치의 하네스 설정으로)."""
    parts = joint_key_parts(job["action"]["key"])
    state = records_by_id[job["episode_id"]]["ticks"][int(job["index"])]["request"]["state"]
    entry = next((item for item in state["objects"] if str(item["id"]) == parts[1]), None)
    if entry is None:
        return "?"
    pose = [float(value) for value in entry["pose_mm"]]
    surface = float((state.get("scene") or {}).get("work_surface_mm", 0.0))
    height = max(pose[2], surface + float(spec.get("push_height_min_mm", 0.0)))
    top = float(entry.get("top_mm", pose[2] + float(entry["obb_mm"][2]) / 2.0))
    return push_contact_class(spec, top, height)


def push_jobs(
    records: list[dict[str, Any]],
    events: dict[str, Any],
    *,
    sim_config: str,
    harness_config: dict[str, Any],
    control_steps: int,
    directions: tuple[str, ...] | None = None,
    limit_keyframes: int | None = None,
    log: Any = None,
    generator_config: str | Path | dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """배치의 모든 키프레임을 재생해 job을 만들고 밀기 job(선택: 방향 집합, 앞 `limit_keyframes`개 키프레임)만 남긴다.
    job마다 `direction`·`klass`(fingers/hand)를 붙인다."""
    if limit_keyframes is not None:
        records = list(records)
    jobs, keyframes, summary = build_jobs(
        records, events, limit=None, sim_config=sim_config, harness_config=harness_config, control_steps=control_steps, log=log,
        generator_config=generator_config,
    )
    if limit_keyframes is not None:
        kept = {f"{frame['episode_id']}@{frame['t']}" for frame in keyframes[:limit_keyframes]}
        jobs = [job for job in jobs if job["keyframe"] in kept]
        keyframes = keyframes[:limit_keyframes]
    by_id = {record["episode_id"]: record for record in records}
    spec = harness_config["candidates"]
    selected = []
    for job in jobs:
        parts = joint_key_parts(job["action"]["key"])
        if parts is None or parts[0] != "push":
            continue
        if directions is not None and parts[2] not in directions:
            continue
        selected.append({**job, "direction": parts[2], "klass": _job_class(job, by_id, spec)})
    summary = {**summary, "jobs_all": len(jobs), "push_jobs": len(selected)}
    return selected, keyframes, summary


def summarise_arm(results: list[dict[str, Any]], jobs: list[dict[str, Any]]) -> dict[str, Any]:
    """방향별·접촉 부위별 단계 표 (`summarise_sweep`의 `push_stage_by_direction`과 같은 분류) + `push_horizon`의 변위 중앙값."""
    meta = {(job["keyframe"], job["action"]["id"], int(job["seed"])): job for job in jobs}
    by_direction: dict[str, dict[str, int]] = {}
    by_class: dict[str, dict[str, int]] = {}
    outcomes: dict[str, int] = {}
    horizon_displacement: dict[str, list[float]] = {}
    walls: list[float] = []
    for result in results:
        job = result["job"]
        info = meta[(job["keyframe"], job["candidate"], int(job["seed"]))]
        direction, klass = info["direction"], info["klass"]
        evidence = result["evidence"]
        outcome = str(result["outcome"])
        outcomes[outcome] = outcomes.get(outcome, 0) + 1
        if outcome == "failure":
            stage = "approach" if evidence.get("first_action_tick") is None else "push"
            label = f"{stage}_{result.get('reason')}"
        else:
            label = outcome
        for table, name in ((by_direction, direction), (by_class, f"{klass}:{direction}")):
            row = table.setdefault(name, {"rollouts": 0})
            row["rollouts"] += 1
            row[label] = row.get(label, 0) + 1
        if label == "push_horizon" and evidence.get("displacement_along_mm") is not None:
            horizon_displacement.setdefault(f"{klass}:{direction}", []).append(float(evidence["displacement_along_mm"]))
        if evidence.get("wall_s") is not None:
            walls.append(float(evidence["wall_s"]))
    return {
        "outcomes": dict(sorted(outcomes.items())),
        "by_direction": {name: dict(sorted(row.items())) for name, row in sorted(by_direction.items())},
        "by_class": {name: dict(sorted(row.items())) for name, row in sorted(by_class.items())},
        "push_horizon_displacement_median_mm": {
            name: round(statistics.median(values), 1) for name, values in sorted(horizon_displacement.items())
        },
        "wall_s_per_rollout": round(statistics.fmean(walls), 4) if walls else None,
    }


def run(
    dataset: Path,
    arms: dict[str, dict[str, Any]],
    *,
    workers: int = 1,
    directions: tuple[str, ...] | None = None,
    limit_keyframes: int | None = None,
    out: Path | None = None,
    scratch: Path = DEFAULT_SCRATCH,
    events_path: str = "configs/sim/events.yaml",
    sim_config: str = "configs/sim/tidy_clutter.yaml",
    generator_config: str | Path = "configs/data/d1_robot.yaml",
    events_override: dict[str, Any] | None = None,
    command: list[str] | None = None,
    log: Any = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    events = {**load_events_config(events_path), "_path": str(events_path), **(events_override or {})}
    harness_path = str(events["followup"]["harness_config"])
    expert_path = str(events["followup"]["expert_config"])
    harness_config = load_harness_config(harness_path)
    records = [record for _, record in read_episodes(dataset)]
    if not records:
        raise FileNotFoundError(f"에피소드가 없다: {dataset}")
    jobs, keyframes, summary = push_jobs(
        records, events, sim_config=sim_config, harness_config=harness_config, control_steps=int(events["followup"]["control_steps_per_tick"]),
        directions=directions, limit_keyframes=limit_keyframes, log=log, generator_config=generator_config,
    )
    report: dict[str, Any] = {
        "version": ROLLOUTS_VERSION,
        "command": list(command or ()),
        "dataset": str(dataset),
        "keyframes": len(keyframes),
        "keyframes_exact": sum(1 for frame in keyframes if frame.get("exact")),
        "push_jobs": len(jobs),
        "directions": list(directions) if directions else None,
        "seeds": int(events.get("seeds", 8)),
        "running_versions": summary["running_versions"],
        "jobs_by_direction": _count(job["direction"] for job in jobs),
        "jobs_by_class": _count(f"{job['klass']}:{job['direction']}" for job in jobs),
        "arms": {},
    }
    for name, override in arms.items():
        arm = write_arm_configs(name, override, Path(scratch), harness_path=harness_path, expert_path=expert_path)
        arm_jobs = [{**job, "event": {**job["event"], "harness_config": arm["harness_config"]}} for job in jobs]
        if log is not None:
            print(f"arm {name}: {len(arm_jobs)} push jobs, harness {arm['harness_config']} ({arm['harness_sha256'][:12]})", file=log, flush=True)
        arm_started = time.perf_counter()
        results = run_jobs(arm_jobs, sim_config=sim_config, expert_config=arm["expert_config"], workers=workers, log=log)
        report["arms"][name] = {**arm, "wall_s": round(time.perf_counter() - arm_started, 1), **summarise_arm(results, jobs)}
    report["batch_wall_s"] = round(time.perf_counter() - started, 1)
    if out is not None:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_text(json.dumps(report, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return report


def _count(values) -> dict[str, int]:
    out: dict[str, int] = {}
    for value in values:
        out[value] = out.get(value, 0) + 1
    return dict(sorted(out.items()))
