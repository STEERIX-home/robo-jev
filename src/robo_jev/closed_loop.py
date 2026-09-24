"""폐루프 평가 — 장면(seed) 선택, 정책 어댑터, run, 지표, seed로 짝지은 구간 (Task R4 Stage B·C; docs/06 Task 6, docs/08 §10).

지금까지의 모든 숫자는 녹화된 에피소드를 **재생**한 오프라인 값이었다. 여기서는 정책이 실제로 시뮬레이터를 움직인다:
:func:`robo_jev.data.robot_episodes.generate_episode` 의 정책 자리에 (1) 학습된 모델(:class:`robo_jev.harness.model_policy.ModelPolicy`),
(2) 규칙 판정기, (3) 기계적 기준군, (4) 전문가 자신을 차례로 꽂고 **같은 seed 집합**을 돌린다. 전문가는 정책이 아닐 때도
틱마다 참조 답을 내고(생성기 :323, DAgger의 씨앗) 그것이 `labels`가 되므로, 폐루프 레코드에서 반응 지연·`q_stop`·안전
위반 같은 오프라인 지표를 **같은 자**(:mod:`robo_jev.evaluate`)로 잴 수 있다.

**층 (docs/06 Task 6 "기하 충분 층과 의미 판단 층").** 둘을 낸다.

* **조건 층** — 에피소드가 무엇을 요구했는가: `no_instruction_change`(E0 — 시작 지시 하나, 하네스의 기하와 후보로 충분하다)
  대 `instruction_changes`(E1·E2 — 지시가 바뀌므로 정책이 문장을 다시 읽어 대상을 바꿔야 한다). "규칙 기준군이 두 층 모두
  포화한다"는 이 층 위에서 읽는다.
* **실패 원인** — 실패한 편을 :func:`failure_cause` 로 가른다. **semantic_main** = 정책이 채택한 주 결정이 전문가 참조와
  결정적 틱에서 달랐다(결합 후보의 대상·기능이 참조 허용 집합의 어느 결합 후보와도 다른 틱이 하나라도 있거나, 참조가 결합
  후보를 허용하는데 정책이 끝까지 게이트·hold만 했다). **semantic_aux** = 주 결정은 같은데 **부가 판단**(그리퍼·경로·속도·힘)이
  참조와 `AUX_DISAGREEMENT_STREAK`틱 이상 이어서 달랐다 — 예: 파지점에 닿았는데 `closed`를 말하지 않아 팔이 열린 채 머문다
  (R4 첫 run이 실제로 그랬다). 둘 다 **모델의 판단**이 참조와 다른 것이고, 부가 판단은 오프라인 칸에서 몫이 작아 가려진다.
  **geometric** = 그런 틱이 없다 — 같은 판단을 하고도 실행이 끝나지 않았다(정체·충돌·시간). 보고서의 "semantic"은 둘의 합이다.

모든 비교는 seed로 짝지은 편 단위 부트스트랩 구간(:func:`robo_jev.evaluate.episode_bootstrap`)이고, 0을 포함하면 발견이
아니다.
"""

from __future__ import annotations

import copy
import json
import math
import statistics
import time
from collections import Counter
from pathlib import Path
from typing import Any

import yaml

from robo_jev.contracts import QUESTION_SET_V0, validate_record
from robo_jev.data.episode import aggregate as aggregate_records
from robo_jev.data.robot_episodes import (
    build_manifest,
    config_paths,
    episode_id,
    generate_episode,
    load_generator_config,
    origin_group,
    plan_tags,
    profile_cycle,
    seed_schedule,
    split_policy,
    write_episode,
)
from robo_jev.evaluate import (
    REACTION_HORIZON_TICKS,
    answer_stability,
    episode_bootstrap,
    reaction_delay,
    selective_metrics_from_stored,
    stop_timing,
)
from robo_jev.sim.controller import resolve_config_path
from robo_jev.sim.scene import build_plan, family_id

__all__ = [
    "CLOSED_LOOP_VERSION",
    "POLICY_KINDS",
    "MechanicalPolicy",
    "TimedEnvironment",
    "TimedExpert",
    "build_policy",
    "closed_loop_report",
    "condition_layer",
    "condition_metrics",
    "episode_summary",
    "failure_cause",
    "gripper_event_metrics",
    "offline_gripper_transitions",
    "load_closed_loop_config",
    "paired_success",
    "per_record_rows",
    "print_report",
    "quotas",
    "run_condition",
    "select_conditions",
    "select_seeds",
]

CLOSED_LOOP_VERSION = "cl0.1"
POLICY_KINDS = ("model", "rule", "mechanical", "expert")
#: 조건 층의 이름 (모듈 설명).
LAYERS = ("no_instruction_change", "instruction_changes")
#: 그리퍼 전환을 짝지을 때 허용하는 최대 틱 차 (전문가 라벨의 전환 허용 ±1틱보다 넉넉히; 10틱 = 1 s).
GRIPPER_MATCH_HORIZON_TICKS = 10
_GATE_KEYS = ("observe", "hold", "replan")
#: 부가 판단의 불일치를 "달랐다"로 세는 최소 연속 틱 (전문가 라벨의 전환 허용 ±1틱을 넘고, 한 틱의 흔들림은 세지 않는다).
AUX_DISAGREEMENT_STREAK = 3
_AUX = ("q_gripper", "q_path", "q_speed", "q_force")
_JOINT = ("grasp", "place", "push")


# --------------------------------------------------------------------------
# 설정·장면(seed) 선택 (B2)
# --------------------------------------------------------------------------


def load_closed_loop_config(path: str | Path) -> dict[str, Any]:
    """`configs/eval/r4-closed-loop.yaml` → 생성 설정(읽은 것)·seed 구간·조건. 모르는 키·빈 조건은 거절한다."""
    path = Path(path)
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    allowed = ("version", "generator_config", "seeds", "scan", "conditions")
    unknown = [key for key in raw if key not in allowed]
    if unknown:
        raise ValueError(f"{path}: 알 수 없는 키 {unknown} (허용: {list(allowed)})")
    for key in ("generator_config", "seeds", "conditions"):
        if not raw.get(key):
            raise ValueError(f"{path}: {key}가 필요하다")
    if raw["seeds"].get("base") is None:
        raise ValueError(f"{path}: seeds.base가 필요하다")
    conditions: dict[str, dict[str, Any]] = {}
    for name, entry in raw["conditions"].items():
        if not isinstance(entry, dict) or not entry.get("split") or int(entry.get("count", 0)) < 1:
            raise ValueError(f"{path}: conditions.{name}에는 split과 1 이상의 count가 필요하다")
        if str(entry["split"]) == "ood_test":
            raise ValueError(f"{path}: conditions.{name}: ood_test는 봉인이다 — 폐루프 장면으로 쓰지 않는다")
        conditions[str(name)] = {"split": str(entry["split"]), "count": int(entry["count"])}
    generator_path = resolve_config_path(raw["generator_config"])
    generator = load_generator_config(generator_path)
    return {
        "path": str(path), "version": raw.get("version"), "generator_config_path": str(generator_path), "generator": generator,
        "seeds": {"base": int(raw["seeds"]["base"])}, "scan": {"per_profile_max": int((raw.get("scan") or {}).get("per_profile_max", 4000))},
        "conditions": conditions,
    }


def quotas(weights: list[int], count: int) -> list[int]:
    """비중대로 정수 몫을 나눈다 (최대 나머지법; 동점은 앞 순서). [20, 40, 40]·100 → [20, 40, 40], ·26 → [5, 11, 10]."""
    total = float(sum(weights))
    raw = [count * float(weight) / total for weight in weights]
    out = [int(math.floor(value)) for value in raw]
    remainder = count - sum(out)
    order = sorted(range(len(weights)), key=lambda index: (-(raw[index] - out[index]), index))
    for index in order[:remainder]:
        out[index] += 1
    return out


def select_seeds(
    generator: dict[str, Any], sim_settings: dict[str, Any], *, split: str, count: int, base: int, per_profile_max: int = 4000,
) -> dict[str, Any]:
    """생성기의 seed 일정(프로파일 순환·프로파일별 계수기)을 `base`에서 걸으며 `split`에 떨어지는 장면을 프로파일 몫만큼 고른다.

    계획만 짓는다(:func:`build_plan` — 물리 없음). 다른 split에 떨어진 seed는 세기만 하고 **적지 않는다**(봉인 `ood_test` 포함).
    """
    profiles = [str(name) for name in generator["profiles"]]
    weights = [int(value) for value in (generator.get("profile_weights") or [1] * len(profiles))]
    quota = dict(zip(profiles, quotas(weights, count)))
    policy = split_policy(generator)
    schedule = seed_schedule({**generator, "seeds": {**generator["seeds"], "base": int(base)}}, per_profile_max * len(profile_cycle(generator)))
    chosen: list[dict[str, Any]] = []
    taken: dict[str, int] = {name: 0 for name in profiles}
    scanned = 0
    other_splits: Counter = Counter()
    for profile, seed in schedule:
        if all(taken[name] >= quota[name] for name in profiles):
            break
        if taken[profile] >= quota[profile]:
            continue
        scanned += 1
        plan = build_plan(sim_settings, int(seed), profile)
        group = origin_group(profile, plan)
        tags = plan_tags(plan)
        assigned = policy.assign(group, tags)
        if assigned != split:
            other_splits[assigned] += 1
            continue
        taken[profile] += 1
        chosen.append({
            "profile": profile, "seed": int(seed), "episode_id": episode_id(profile, int(seed)), "origin_group": group,
            "family": family_id(plan), "split": assigned, "holdout": policy.holdout_reasons(group, tags),
        })  # fmt: skip
    short = {name: quota[name] - taken[name] for name in profiles if taken[name] < quota[name]}
    return {"split": split, "count": count, "quota": quota, "by_profile": dict(taken), "shortfall": short, "scanned": scanned,
            "skipped_other_splits": dict(sorted(other_splits.items())), "seeds": chosen}


def select_conditions(config: dict[str, Any], *, manifest: str | Path | None = None) -> dict[str, Any]:
    """설정의 조건마다 :func:`select_seeds`. `manifest`(r1)를 주면 고른 계열 가운데 r1이 이미 쓴 계열의 수도 적는다."""
    generator = config["generator"]
    paths = config_paths(generator)
    sim_settings = yaml.safe_load(resolve_config_path(paths["sim_config"]).read_text(encoding="utf-8"))
    known_families: set[str] = set()
    r1_seeds: set[tuple[str, int]] = set()
    if manifest is not None and Path(manifest).is_file():
        loaded = json.loads(Path(manifest).read_text(encoding="utf-8"))
        known_families = {str(name) for name in (loaded.get("families") or {})}
        r1_seeds = {(profile, seed) for profile, seed in seed_schedule(generator, int(loaded.get("episodes") or 0))}
    out: dict[str, Any] = {"version": CLOSED_LOOP_VERSION, "generator_config": config["generator_config_path"], "seeds_base": config["seeds"]["base"], "conditions": {}}
    for name, entry in config["conditions"].items():
        block = select_seeds(generator, sim_settings, split=entry["split"], count=entry["count"], base=config["seeds"]["base"], per_profile_max=config["scan"]["per_profile_max"])
        collisions = [item for item in block["seeds"] if (item["profile"], item["seed"]) in r1_seeds]
        if collisions:
            raise ValueError(f"{name}: r1이 이미 만든 seed와 겹친다: {collisions[:3]}")
        families = sorted({item["origin_group"] for item in block["seeds"]})
        block["families"] = families
        block["families_in_r1"] = sum(1 for family in families if family in known_families)
        out["conditions"][name] = block
    return out


# --------------------------------------------------------------------------
# 정책 (B1)
# --------------------------------------------------------------------------


class MechanicalPolicy:
    """**기계적 기준군**을 `act` 꼴로 (:data:`robo_jev.evaluate.MECHANICAL_BASELINE_POLICY`): 이 틱의 commitment가 있으면 그것,
    없으면 `observe` 게이트 키 — 아무것도 읽지 않는다. `q_main`에만 정의된 정책이라 나머지는 하네스의 **기본값**을 그대로 말한다
    (게이트는 열리지 않고 정지도 없다: done 0 · instr 1 · observe 0 · retry 0 · stop 0; 부가 답은 빈 분포 → 하네스가 초기
    프로파일로 채운다). 스스로 결합 후보를 고른 적이 없으므로 루프에서는 관측·hold만 되풀이한다 — 그것이 바닥의 뜻이다."""

    name = "MechanicalPolicy"
    version = "mech0.1"

    def act(self, request: dict[str, Any], commitment: dict[str, Any] | None = None, observation: Any = None) -> dict[str, Any]:
        del commitment, observation  # 요청에 실린 투영만 본다 (오프라인 열과 같은 필드)
        model = request.get("request", request)
        entries = list(model["candidates"]["q_main"])
        ids = [str(entry["id"]) for entry in entries]
        reference = (model.get("commitment") or {}).get("action_ref")
        chosen = str(reference) if reference is not None and str(reference) in ids else None
        if chosen is None:
            chosen = next((cid for cid, entry in zip(ids, entries) if str(entry.get("key", "")).startswith("observe")), None)
        if chosen is None:
            main = {cid: 1.0 / len(ids) for cid in ids}
        else:
            main = {cid: (1.0 if cid == chosen else 0.0) for cid in ids}
        return {
            "q_main": main, "q_done": 0.0, "q_instr": 1.0, "q_observe": 0.0, "q_retry": 0.0, "q_stop": 0.0,
            "q_gripper": {}, "q_path": {}, "q_speed": {}, "q_force": {},
        }


class TimedExpert:
    """전문가를 감싸 `act`·`labels`의 벽시계를 잰다 (참조 답 계산은 배포에 없는 비용이라 관측→명령 시간에서 뺀다). 나머지는 그대로 넘긴다."""

    def __init__(self, expert: Any) -> None:
        self._expert = expert
        self.act_ms: list[float] = []
        self.labels_ms: list[float] = []

    def __getattr__(self, name: str) -> Any:
        return getattr(self._expert, name)

    def act(self, request: dict[str, Any], commitment: dict[str, Any] | None, observation: Any = None) -> dict[str, Any]:
        started = time.perf_counter()
        out = self._expert.act(request, commitment, observation)
        self.act_ms.append((time.perf_counter() - started) * 1e3)
        return out

    def labels(self, answers: dict[str, Any], request: dict[str, Any]) -> list[dict[str, Any]]:
        started = time.perf_counter()
        out = self._expert.labels(answers, request)
        self.labels_ms.append((time.perf_counter() - started) * 1e3)
        return out


class TimedEnvironment:
    """환경을 감싸 **관측→명령** 벽시계를 잰다: 틱의 마지막 제어 스텝이 관측을 돌려준 순간부터 다음 `step(command)`가 불릴 때까지
    (= 요청 만들기 + 정책 + 참조 답·라벨 + 조합 + 레코드 기록). 물리·일정은 건드리지 않는다."""

    def __init__(self, env: Any) -> None:
        self._env = env
        self.intervals_ms: list[float] = []
        self._ready_at: float | None = None

    def __getattr__(self, name: str) -> Any:
        return getattr(self._env, name)

    def reset(self, seed: int) -> dict[str, Any]:
        self.intervals_ms = []
        observation = self._env.reset(seed)
        self._ready_at = time.perf_counter()
        return observation

    def step(self, command: dict[str, Any] | None = None) -> dict[str, Any]:
        if command is not None and self._ready_at is not None:
            self.intervals_ms.append((time.perf_counter() - self._ready_at) * 1e3)
        observation = self._env.step(command)
        self._ready_at = time.perf_counter()
        return observation


def build_policy(
    kind: str, *, generator: dict[str, Any], checkpoint: str | Path | None = None, model_id: str = "Qwen/Qwen3.5-2B", compile_dense: bool = True,
) -> dict[str, Any]:
    """정책 묶음 ``{"kind", "policy", "expert", "describe"}`` — `expert`는 참조·라벨의 원천(정책이 expert면 같은 객체)."""
    from robo_jev.sim.expert import Expert, load_expert_config

    if kind not in POLICY_KINDS:
        raise ValueError(f"policy: {list(POLICY_KINDS)} 중 하나여야 한다 (받은 값: {kind!r})")
    paths = config_paths(generator)
    expert = TimedExpert(Expert(load_expert_config(paths["expert_config"])))
    if kind == "expert":
        return {"kind": kind, "policy": expert, "expert": expert, "describe": {"name": "Expert", "version": expert.version, "kind": kind}}
    if kind == "rule":
        from robo_jev.data.dagger import rule_judge_policy

        policy = rule_judge_policy()
        return {"kind": kind, "policy": policy, "expert": expert, "describe": {"name": policy.name, "version": policy.version, "kind": kind}}
    if kind == "mechanical":
        policy = MechanicalPolicy()
        return {"kind": kind, "policy": policy, "expert": expert, "describe": {"name": policy.name, "version": policy.version, "kind": kind}}
    if not checkpoint:
        raise ValueError("model: --checkpoint가 필요하다")
    from robo_jev.harness.model_policy import ModelPolicy, load_serving_judge

    bundle = load_serving_judge(checkpoint, model_id=model_id, compile_dense=compile_dense)
    policy = ModelPolicy(bundle["judge"], bundle["tokenizer"], fused=True)
    describe = {
        **policy.describe(), "kind": kind, "checkpoint": str(checkpoint), "model_id": model_id, "compile": bool(compile_dense),
        "compile_seconds": bundle["compile_seconds"], "load_seconds": bundle["load_seconds"], "tokenizer": bundle["tokenizer_name"],
        "checkpoint_manifest": {key: bundle["manifest"].get(key) for key in ("model", "contract_sha256", "serializer_version", "tokenizer")},
    }
    return {"kind": kind, "policy": policy, "expert": expert, "describe": describe}


# --------------------------------------------------------------------------
# run (B1·B4)
# --------------------------------------------------------------------------


def _quantiles(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"n": 0}
    ordered = sorted(values)

    def q(fraction: float) -> float:
        position = fraction * (len(ordered) - 1)
        low = int(position)
        high = min(low + 1, len(ordered) - 1)
        return ordered[low] + (ordered[high] - ordered[low]) * (position - low)

    return {"n": len(ordered), "mean": statistics.fmean(ordered), "p50": q(0.5), "p90": q(0.9), "p95": q(0.95), "p99": q(0.99), "max": ordered[-1],
            "over_80ms_rate": sum(v > 80.0 for v in ordered) / len(ordered), "over_100ms_rate": sum(v > 100.0 for v in ordered) / len(ordered)}


def run_condition(
    bundle: dict[str, Any], schedule: list[tuple[str, int]], *, config: dict[str, Any], out: str | Path, condition: str, label: str,
    log: Any = None, max_ticks: int | None = None,
) -> dict[str, Any]:
    """정책 하나를 seed 목록 전부에 돌려 에피소드를 쓰고(레코드 + manifest + 틱 지연 sidecar) 편별 요약을 돌려준다."""
    from robo_jev.sim.environment import Environment

    generator = config["generator"]
    paths = config_paths(generator)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    policy, expert = bundle["policy"], bundle["expert"]
    suffix = f"-r4-{label}"
    envs: dict[str, TimedEnvironment] = {}
    episodes: list[dict[str, Any]] = []
    timing_path = out / "timing.jsonl"
    started = time.perf_counter()
    obs_to_command: list[float] = []
    obs_to_command_net: list[float] = []
    policy_ms: list[float] = []
    model_ms: list[float] = []
    first_tick_model_ms: list[float] = []
    with timing_path.open("w", encoding="utf-8") as sink:
        try:
            for profile, seed in schedule:
                env = envs.get(profile)
                if env is None:
                    env = envs[profile] = TimedEnvironment(Environment(config_path=paths["sim_config"], profile=profile))
                if hasattr(policy, "reset"):
                    policy.reset()
                expert.act_ms, expert.labels_ms = [], []
                record = generate_episode(profile, seed, policy=policy, expert=expert, config=generator, env=env, max_ticks=max_ticks, id_suffix=suffix)
                validate_record(record)
                write_episode(record, out)
                rows = list(getattr(policy, "timing", []) or [])
                summary = episode_summary(record)
                ticks = len(record["ticks"])
                reference = [a + b for a, b in zip(expert.act_ms, expert.labels_ms)] if bundle["kind"] != "expert" else [0.0] * ticks
                intervals = list(env.intervals_ms)
                for index in range(ticks):
                    row = {"episode_id": record["episode_id"], "condition": condition, "index": index, "t": record["ticks"][index]["t"],
                           "obs_to_command_ms": intervals[index] if index < len(intervals) else None,
                           "reference_ms": reference[index] if index < len(reference) else None}
                    if index < len(rows):
                        row.update({key: rows[index][key] for key in rows[index] if key != "t"})
                    sink.write(json.dumps(row, ensure_ascii=False) + "\n")
                    if row["obs_to_command_ms"] is not None:
                        obs_to_command.append(float(row["obs_to_command_ms"]))
                        obs_to_command_net.append(float(row["obs_to_command_ms"]) - float(row["reference_ms"] or 0.0))
                    if index < len(rows):
                        policy_ms.append(float(rows[index]["act_ms"]))
                        (first_tick_model_ms if rows[index].get("prefix_tokens") else model_ms).append(float(rows[index]["model_ms"]))
                summary["timing"] = {"obs_to_command_ms": _quantiles(intervals) if intervals else {"n": 0}, "model_ms": _quantiles([r["model_ms"] for r in rows]) if rows else {"n": 0}}
                episodes.append(summary)
                if log is not None:
                    print(f"{record['episode_id']} {condition:<8} ticks={ticks:<3} done={summary['done']} layer={summary['layer']} cause={summary['failure_cause']} "
                          f"wall={summary['wall_s']:.1f}s model_p95={summary['timing']['model_ms'].get('p95', float('nan')):.1f}ms", file=log, flush=True)
        finally:
            for env in envs.values():
                env.close()
    wall = time.perf_counter() - started
    manifest = build_manifest(out, generator, batch_wall_s=wall)
    manifest["closed_loop"] = {"version": CLOSED_LOOP_VERSION, "policy": bundle["describe"], "condition": condition, "label": label, "episodes": len(episodes)}
    (out / "manifest.json").write_bytes((json.dumps(manifest, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))
    latency = {
        "obs_to_command_ms": _quantiles(obs_to_command), "obs_to_command_net_of_reference_ms": _quantiles(obs_to_command_net),
        "policy_act_ms": _quantiles(policy_ms), "model_ms": _quantiles(model_ms), "first_tick_model_ms": _quantiles(first_tick_model_ms),
        "gate": {"rule": "model_ms p95 ≤ 80 ms and >100 ms rate ≤ 5 % on non-prefix ticks (docs/03 §7-6)",
                 "passes": bool(model_ms) and _quantiles(model_ms)["p95"] <= 80.0 and _quantiles(model_ms)["over_100ms_rate"] <= 0.05} if model_ms else None,
    }
    return {
        "condition": condition, "label": label, "episodes_dir": str(out), "timing_file": str(timing_path), "episodes": episodes,
        "summary": {
            "episodes": len(episodes), "done": sum(1 for row in episodes if row["done"]), "ticks": sum(row["ticks"] for row in episodes),
            "wall_seconds": round(wall, 1), "sim_seconds": sum(row["sim_ms"] for row in episodes) / 1e3,
            "model_seconds": sum(r for r in model_ms) / 1e3 + sum(first_tick_model_ms) / 1e3,
        },
        "latency": latency,
    }


# --------------------------------------------------------------------------
# 편별 요약과 층 (B3)
# --------------------------------------------------------------------------


def _key_parts(key: Any) -> tuple[str, str, str, str] | None:
    parts = str(key or "").split(":")
    if len(parts) < 4 or parts[0] not in _JOINT:
        return None
    return parts[0], parts[1], parts[2], parts[3]


def _goal_version(tick: dict[str, Any]) -> int:
    goal = (tick["request"].get("state") or {}).get("goal")
    return int(goal.get("version", 1)) if isinstance(goal, dict) and goal.get("version") is not None else 1


def condition_layer(record: dict[str, Any]) -> str:
    """조건 층 (모듈 설명): **장면의 성질**이다 — 장면 계획(`evidence.scene_plan.instructions`)에 지시가 둘 이상 예정돼 있으면
    `instruction_changes`, 하나면 `no_instruction_change`. seed가 같으면 정책이 달라도 같은 층이라 seed로 짝지어 읽을 수 있다
    (빨리 끝낸 정책이 변경 시각 전에 완료했어도 층은 그대로다). 계획이 없는 레코드는 실제로 관측된 목표 버전으로 대신한다."""
    plan = (record.get("evidence") or {}).get("scene_plan") or {}
    instructions = plan.get("instructions")
    if isinstance(instructions, list) and instructions:
        return LAYERS[1] if len(instructions) > 1 else LAYERS[0]
    ticks = record["ticks"]
    return LAYERS[1] if any(_goal_version(b) > _goal_version(a) for a, b in zip(ticks, ticks[1:])) or _goal_version(ticks[0]) > 1 else LAYERS[0]


def _decision_ticks(record: dict[str, Any]) -> dict[str, Any]:
    """정책의 채택 판단 대 전문가 참조 — 주 결정의 결정적 틱 수와 부가 판단의 불일치 (모듈 설명의 실패 원인 규칙).

    주 결정: 채택 결합 후보의 (기능, 대상)이 참조 허용 집합의 어느 결합 후보와도 다르면 `wrong_action`, 참조가 결합 후보를
    허용하는데 게이트·hold를 채택했으면 `idle`. 부가 판단: 라벨이 한 값인 틱에서 정책의 **채택된** 부가 답(`adopted`의 gripper·
    path 종류·speed·force)이 다르면 불일치이고, 가장 긴 연속 불일치 길이를 질문마다 센다(`aux_streak`)."""
    wrong_action = idle = agreed = acted = expert_joint = 0
    aux_total: dict[str, int] = {qid: 0 for qid in _AUX}
    aux_agree: dict[str, int] = {qid: 0 for qid in _AUX}
    aux_streak: dict[str, int] = {qid: 0 for qid in _AUX}
    running: dict[str, int] = {qid: 0 for qid in _AUX}
    for tick in record["ticks"]:
        adopted = tick.get("adopted") or {}
        main = adopted.get("main")
        labels = {item.get("question_id"): item for item in tick.get("labels") or ()}
        label = labels.get("q_main")
        if main is not None and label is not None:
            keys = {str(entry["id"]): str(entry.get("key", "")) for entry in tick["request"]["candidates"]["q_main"]}
            allowed_joint = {_key_parts(keys.get(cid)) for cid in (label.get("candidate_ids") or ())} - {None}
            adopted_parts = _key_parts(keys.get(main))
            if allowed_joint:
                expert_joint += 1
            if adopted_parts is not None:
                acted += 1
                if allowed_joint and adopted_parts[:2] not in {parts[:2] for parts in allowed_joint}:
                    wrong_action += 1  # 다른 대상·다른 기능 — 참조와 다른 결정
                elif allowed_joint:
                    agreed += 1
            elif allowed_joint:
                idle += 1  # 참조는 결합 후보를 허용하는데 정책은 게이트·hold
        paths = {str(entry["id"]): str(entry.get("kind", "")) for entry in tick["request"]["candidates"].get("q_path") or []}
        chosen = {
            "q_gripper": adopted.get("gripper"), "q_path": adopted.get("path_kind") or paths.get(str(adopted.get("path"))),
            "q_speed": None if adopted.get("speed") is None else str(adopted.get("speed")), "q_force": None if adopted.get("force") is None else str(adopted.get("force")),
        }
        for qid in _AUX:
            item = labels.get(qid)
            if not adopted or item is None or chosen[qid] is None:
                running[qid] = 0
                continue
            wanted = [str(cid) for cid in item.get("candidate_ids", [item.get("answer")])]
            if qid == "q_path":
                wanted = [paths.get(cid, cid) for cid in wanted]
            if len(wanted) != 1:
                running[qid] = 0  # 전환 허용 구간·경유점 대안 — 판정하지 않는다
                continue
            aux_total[qid] += 1
            if str(chosen[qid]) == wanted[0]:
                aux_agree[qid] += 1
                running[qid] = 0
            else:
                running[qid] += 1
                aux_streak[qid] = max(aux_streak[qid], running[qid])
    return {"acted": acted, "expert_joint": expert_joint, "wrong_action": wrong_action, "agreed": agreed, "idle": idle,
            "aux_total": aux_total, "aux_agree": aux_agree, "aux_streak": aux_streak}


def failure_cause(record: dict[str, Any], decisions: dict[str, Any] | None = None) -> str | None:
    """실패한 편의 원인 층 (모듈 설명): `semantic_main` | `semantic_aux` | `geometric`; 완료한 편은 None."""
    if record["provenance"]["outcome"]["done"]:
        return None
    counts = decisions or _decision_ticks(record)
    if counts["wrong_action"] > 0 or (counts["expert_joint"] > 0 and counts["acted"] == 0):
        return "semantic_main"
    if any(streak >= AUX_DISAGREEMENT_STREAK for streak in counts["aux_streak"].values()):
        return "semantic_aux"
    return "geometric"


def episode_summary(record: dict[str, Any]) -> dict[str, Any]:
    """폐루프 레코드 하나의 요약 행 (성공·완료 시각·층·결정 계수·게이트·정지·거절·충돌·비용)."""
    outcome = record["provenance"]["outcome"]
    provenance = record["provenance"]
    ticks = record["ticks"]
    counts = aggregate_records([record])["per_episode"][0]
    decisions = _decision_ticks(record)
    commanded = [tick for tick in ticks if tick.get("adopted") is not None]
    rejected = [tick for tick in commanded if (tick.get("ack") or {}).get("rejected")]
    reasons = Counter(str((tick.get("ack") or {}).get("reason")) for tick in rejected)
    reflex_ticks = [index for index, tick in enumerate(ticks) if any(str(event.get("kind", "")).startswith("reflex") for event in (tick["request"]["state"].get("events") or ()))]
    stop_ticks = [index for index, tick in enumerate(ticks) if (tick.get("adopted") or {}).get("stop")]
    goal_changes = sum(1 for a, b in zip(ticks, ticks[1:]) if _goal_version(b) > _goal_version(a))
    disturbances = sum(1 for tick in ticks for event in (tick["request"]["state"].get("events") or ()) if str(event.get("kind")) == "disturbance_applied")
    plan = (record.get("evidence") or {}).get("scene_plan") or {}
    scheduled = {"instruction_changes": max(0, len(plan.get("instructions") or [1]) - 1), "disturbances": len(plan.get("disturbances") or [])}
    return {
        "episode_id": record["episode_id"], "profile": provenance["profile"], "seed": int(provenance["seed"]), "key": f"{provenance['profile']}:{provenance['seed']}",
        "origin_group": record.get("origin_group"), "split": record.get("split"),
        "done": bool(outcome["done"]), "done_tick": outcome.get("done_tick"), "first_done_tick": outcome.get("first_done_tick"),
        "ticks": len(ticks), "sim_ms": int(outcome.get("sim_ms", 0)), "terminated": outcome.get("terminated"), "stall": outcome.get("stall"),
        "target_inside_zone": outcome.get("target_inside_zone"), "wall_s": float(provenance["timing"]["wall_s"]),
        "layer": condition_layer(record), "failure_cause": failure_cause(record, decisions), "decisions": decisions,
        "goal_changes": goal_changes, "disturbances": disturbances, "scheduled": scheduled,
        "gates": counts["gates"], "switches": counts["switches"], "main_changes": counts["main_changes"], "conflicts": counts["conflicts"],
        "stops": len(stop_ticks), "reflex_ticks": len(reflex_ticks), "commanded": len(commanded), "rejected": len(rejected), "reject_reasons": dict(reasons),
        "transition_collisions": int(reasons.get("transition_collision", 0)),
    }


# --------------------------------------------------------------------------
# 사건·그리퍼·정지·안전 지표 (B3; 오프라인과 같은 자)
# --------------------------------------------------------------------------


def per_record_rows(records: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """`aggregate(..., store_predictions)`의 `per_record` 꼴로: 채택 주 결정(`adopted`), 모델 raw 답의 argmax(`model`), `q_stop`(모델 답 ≥ 0.5)."""
    adopted: list[dict[str, Any]] = []
    model: list[dict[str, Any]] = []
    stop: list[dict[str, Any]] = []
    for record in records:
        for index, tick in enumerate(record["ticks"]):
            main = (tick.get("adopted") or {}).get("main")
            if main is not None:
                adopted.append({"record_id": record["episode_id"], "tick": index, "question": "q_main", "predicted": str(main)})
            output = tick.get("model_output") or {}
            raw = output.get("q_main")
            if isinstance(raw, dict) and raw:
                model.append({"record_id": record["episode_id"], "tick": index, "question": "q_main", "predicted": max(raw, key=lambda cid: (float(raw[cid]), cid))})
            p_stop = output.get("q_stop")
            if isinstance(p_stop, dict):
                p_stop = p_stop.get("true", 1.0 - float(p_stop.get("false", 1.0)))
            if isinstance(p_stop, (int, float)):
                stop.append({"record_id": record["episode_id"], "tick": index, "question": "q_stop", "predicted": "true" if float(p_stop) >= 0.5 else "false"})
    return {"adopted": adopted, "model": model, "stop": stop}


def gripper_event_metrics(records: list[dict[str, Any]], *, horizon: int = GRIPPER_MATCH_HORIZON_TICKS) -> dict[str, Any]:
    """그리퍼 이벤트의 누락·중복·시각 오차 (docs/08 §10): 전문가 참조가 원하는 그리퍼 상태의 **전환**(라벨 `q_gripper`가 한 값이고
    직전 값과 다른 틱; 전환 허용 구간의 두 값 라벨은 건너뛴다) 대 정책이 **명령한** 그리퍼(`adopted.gripper`)의 전환을 같은
    방향끼리 `horizon`틱 안에서 짝짓는다. 누락 = 짝 없는 참조 전환, 중복 = 짝 없는 정책 전환, 시각 오차 = 짝의 틱 차."""
    expected = matched = missing = duplicate = 0
    errors: list[int] = []
    for record in records:
        ticks = record["ticks"]
        reference: list[tuple[int, str]] = []
        last: str | None = None
        for index, tick in enumerate(ticks):
            label = next((item for item in tick.get("labels") or () if item.get("question_id") == "q_gripper"), None)
            ids = [str(cid) for cid in (label.get("candidate_ids") or ())] if label else []
            if len(ids) != 1:
                continue
            if last is not None and ids[0] != last:
                reference.append((index, ids[0]))
            last = ids[0]
        policy: list[tuple[int, str]] = []
        previous: str | None = None
        for index, tick in enumerate(ticks):
            gripper = (tick.get("adopted") or {}).get("gripper")
            if gripper is None:
                continue
            if previous is not None and gripper != previous:
                policy.append((index, str(gripper)))
            previous = str(gripper)
        used: set[int] = set()
        for ref_index, state in reference:
            expected += 1
            candidates = [(abs(p_index - ref_index), position) for position, (p_index, p_state) in enumerate(policy) if p_state == state and position not in used and abs(p_index - ref_index) <= horizon]
            if candidates:
                distance, position = min(candidates)
                used.add(position)
                matched += 1
                errors.append(policy[position][0] - ref_index)
            else:
                missing += 1
        duplicate += len(policy) - len(used)
    absolute = sorted(abs(value) for value in errors)
    return {
        "reference_transitions": expected, "matched": matched, "missing": missing, "missing_rate": (missing / expected) if expected else None,
        "duplicate": duplicate, "duplicate_rate": (duplicate / (matched + duplicate)) if (matched + duplicate) else None,
        "timing_error_ticks": {"median_abs": (absolute[len(absolute) // 2] if absolute else None), "mean_signed": (statistics.fmean(errors) if errors else None),
                               "max_abs": (absolute[-1] if absolute else None)},
        "horizon_ticks": horizon,
    }


def stop_vs_reflex(records: list[dict[str, Any]]) -> dict[str, Any]:
    """`q_stop` 대 실행기 반사: 정지 틱을 원인별로(모델의 `q_stop` / 반사 사건), 반사 틱에서 모델이 그 틱이나 직전 틱에 이미 정지를
    말했는가, 그리고 정지가 끝내 무시된 채 금지 접촉이 났는가(`contact_onset`이 금지 물체에)."""
    q_stop_ticks = reflex_ticks = reflex_covered_by_model = model_stop_true = forbidden_contacts = 0
    for record in records:
        ticks = record["ticks"]
        forbidden: set[str] = set()
        for index, tick in enumerate(ticks):
            state = tick["request"]["state"]
            forbidden |= {str(item) for item in ((state.get("goal") or {}).get("forbidden_contact") or ())}
            events = state.get("events") or ()
            reflex = any(str(event.get("kind", "")).startswith("reflex") for event in events)
            p_stop = (tick.get("model_output") or {}).get("q_stop")
            if isinstance(p_stop, dict):
                p_stop = p_stop.get("true", 0.0)
            said_stop = isinstance(p_stop, (int, float)) and float(p_stop) >= 0.5
            model_stop_true += int(said_stop)
            if (tick.get("adopted") or {}).get("stop"):
                if reflex:
                    reflex_ticks += 1
                else:
                    q_stop_ticks += 1
            if reflex:
                previous = (ticks[index - 1].get("model_output") or {}).get("q_stop") if index else None
                if isinstance(previous, dict):
                    previous = previous.get("true", 0.0)
                reflex_covered_by_model += int(said_stop or (isinstance(previous, (int, float)) and float(previous) >= 0.5))
            forbidden_contacts += sum(1 for event in events if str(event.get("kind")) == "contact_onset" and str(event.get("object")) in forbidden)
    return {
        "stop_ticks_by_q_stop": q_stop_ticks, "stop_ticks_by_reflex": reflex_ticks, "reflex_ticks_where_model_had_said_stop": reflex_covered_by_model,
        "reflex_ticks_total": sum(1 for record in records for tick in record["ticks"] if any(str(e.get("kind", "")).startswith("reflex") for e in (tick["request"]["state"].get("events") or ()))),
        "model_q_stop_true_ticks": model_stop_true, "forbidden_contact_onsets": forbidden_contacts,
    }


def controller_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    commanded = sum(row["commanded"] for row in rows)
    rejected = sum(row["rejected"] for row in rows)
    reasons: Counter = Counter()
    for row in rows:
        reasons.update(row["reject_reasons"])
    return {
        "commanded_ticks": commanded, "rejected_ticks": rejected, "rejection_rate": (rejected / commanded) if commanded else None,
        "reject_reasons": dict(sorted(reasons.items())), "transition_collisions": sum(row["transition_collisions"] for row in rows),
        "conflicts": sum(row["conflicts"] for row in rows), "switches": sum(row["switches"] for row in rows), "main_changes": sum(row["main_changes"] for row in rows),
    }


def _success_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{"episode_id": row["key"], "n": 1, "graded": 1, "correct": int(bool(row["done"]))} for row in rows]


def paired_success(rows_a: list[dict[str, Any]], rows_b: list[dict[str, Any]]) -> dict[str, Any] | None:
    """두 정책의 성공률 차이의 seed로 짝지은 편 단위 부트스트랩 구간 (같은 seed 집합에서만)."""
    keys = {row["key"] for row in rows_a} & {row["key"] for row in rows_b}
    a = [row for row in rows_a if row["key"] in keys]
    b = [row for row in rows_b if row["key"] in keys]
    if not keys:
        return None
    out = episode_bootstrap(_success_rows(a), _success_rows(b))
    if out is None:
        return None
    return {"seeds": len(keys), "a": out["accuracy"], "b": out.get("control_accuracy"), "margin": out.get("margin"), "margin_ci": out.get("margin_ci"),
            "margin_includes_zero": out.get("margin_includes_zero")}


def condition_metrics(records: list[dict[str, Any]], rows: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    """정책 × 조건 하나의 지표 표 (docs/08 §10 폐루프 항목 전부)."""
    rows = rows if rows is not None else [episode_summary(record) for record in records]
    per_record = per_record_rows(records)
    done = [row for row in rows if row["done"]]
    boot = episode_bootstrap(_success_rows(rows))
    layers: dict[str, Any] = {}
    for layer in LAYERS:
        subset = [row for row in rows if row["layer"] == layer]
        if not subset:
            continue
        layer_boot = episode_bootstrap(_success_rows(subset))
        layers[layer] = {"episodes": len(subset), "done": sum(1 for row in subset if row["done"]), "success_rate": (sum(1 for row in subset if row["done"]) / len(subset)),
                         "success_ci": layer_boot["accuracy_ci"] if layer_boot else None,
                         "failure_causes": dict(Counter(row["failure_cause"] for row in subset if not row["done"]))}
    by_profile = {}
    for profile in sorted({row["profile"] for row in rows}):
        subset = [row for row in rows if row["profile"] == profile]
        by_profile[profile] = {"episodes": len(subset), "done": sum(1 for row in subset if row["done"]), "success_rate": sum(1 for row in subset if row["done"]) / len(subset)}
    completion = sorted(int(row["done_tick"]) for row in done if row.get("done_tick") is not None)
    # 한 번도 그리퍼를 닫지 않고 완료한 편 — 마지막 지시의 대상이 이미 영역 안에 있어 done 게이트가 든 편 (파지가 아니다).
    without_close = [row["episode_id"] for row, record in zip(rows, records) if row["done"] and not any((tick.get("adopted") or {}).get("gripper") == "closed" for tick in record["ticks"])]
    stored_rows = per_record["adopted"] + per_record["stop"]
    return {
        "episodes": len(rows), "done": len(done), "success_rate": (len(done) / len(rows)) if rows else None,
        "success_ci": boot["accuracy_ci"] if boot else None, "bootstrap": {"resamples": boot["resamples"], "seed": boot["seed"]} if boot else None,
        "terminated": dict(Counter(str(row["terminated"]) for row in rows)),
        "completed_without_close": without_close,
        "completion_ticks": {"median": (completion[len(completion) // 2] if completion else None), "p90": (completion[min(len(completion) - 1, int(0.9 * (len(completion) - 1)))] if completion else None), "mean": (statistics.fmean(completion) if completion else None)},
        "layers": layers, "by_profile": by_profile,
        "failure_causes": dict(Counter(row["failure_cause"] for row in rows if not row["done"])),
        "decision_ticks": {key: sum(row["decisions"][key] for row in rows) for key in ("acted", "expert_joint", "wrong_action", "agreed", "idle")},
        "aux_agreement": {qid: {"n": sum(row["decisions"]["aux_total"][qid] for row in rows), "agree": sum(row["decisions"]["aux_agree"][qid] for row in rows),
                                "rate": (sum(row["decisions"]["aux_agree"][qid] for row in rows) / max(1, sum(row["decisions"]["aux_total"][qid] for row in rows))),
                                "episodes_with_streak": sum(1 for row in rows if row["decisions"]["aux_streak"][qid] >= AUX_DISAGREEMENT_STREAK)} for qid in _AUX},
        "aux_failure_questions": dict(Counter(qid for row in rows if row["failure_cause"] == "semantic_aux" for qid in _AUX if row["decisions"]["aux_streak"][qid] >= AUX_DISAGREEMENT_STREAK)),
        "reaction_delay": {"adopted": reaction_delay(per_record["adopted"], records), "model_answer": reaction_delay(per_record["model"], records) if per_record["model"] else None},
        "stability": {"adopted": answer_stability(per_record["adopted"], records), "model_answer": answer_stability(per_record["model"], records) if per_record["model"] else None},
        "stop_timing": stop_timing(per_record["stop"], records) if per_record["stop"] else None,
        "stop_vs_reflex": stop_vs_reflex(records),
        "selective": selective_metrics_from_stored(stored_rows, records),
        "gripper_events": gripper_event_metrics(records),
        "controller": controller_metrics(rows),
        "gates": dict(sum((Counter(row["gates"]) for row in rows), Counter())),
        "events": {"goal_changes": sum(row["goal_changes"] for row in rows), "disturbances": sum(row["disturbances"] for row in rows), "reflex_ticks": sum(row["reflex_ticks"] for row in rows)},
        "cost": {"ticks": sum(row["ticks"] for row in rows), "wall_seconds": sum(row["wall_s"] for row in rows), "sim_seconds": sum(row["sim_ms"] for row in rows) / 1e3,
                 "ticks_per_episode": (sum(row["ticks"] for row in rows) / len(rows)) if rows else None, "wall_per_episode": (sum(row["wall_s"] for row in rows) / len(rows)) if rows else None},
        "horizon_ticks": REACTION_HORIZON_TICKS,
    }


#: 오프라인 재생의 `q_gripper` 예측을 세 종류의 틱으로 나눠 채점한다 (Task R4 C — "판단하는가, 실행 상태를 베끼는가").
GRIPPER_TICK_CLASSES = ("initiate", "window", "settled", "open")


def offline_gripper_transitions(report: dict[str, Any], records: list[dict[str, Any]], *, split_name: str) -> dict[str, Any]:
    """오프라인 재생 산출물(`per_record`가 있는 `q_gripper`)을 전문가의 폐루프 레코드 위에서 **틱 종류별로** 채점한다.

    * `initiate` — 라벨이 한 값 `closed`인데 **실행된** 그리퍼(`state.exec.gripper`)는 아직 `open`인 틱: "지금 닫아라"를 모델이
      스스로 내야 하는 틱(파지마다 몇 틱). 루프에서 팔이 멈춘 자리다.
    * `settled` — 라벨 `closed`이고 실행된 그리퍼도 이미 `closed`인 틱: 실행 상태를 베끼면 맞는 틱.
    * `window` — 라벨이 **두 값**(전환 허용 구간 ±`gripper_transition_tolerance_ticks`)이고 실행된 그리퍼는 아직 `open`인 틱:
      전문가가 실제로 닫기를 시작한 틱은 여기 든다(라벨은 두 값이라 정확도는 없고 `predicted_closed`만 뜻이 있다).
    * `open` — 라벨이 한 값 `open`인 틱.
    `initiate`가 거의 비고 `window`에서 `predicted_closed`가 0에 가까우면, 녹화된 데이터는 "지금 닫아라"를 한 값 라벨로 거의 묻지
    않았고 모델은 그 틱에서 닫지 않는다 — 폐루프에서 그리퍼가 한 번도 닫히지 않는 까닭이다.
    """
    table = report["evaluation"]["splits"][split_name]["model"]
    rows = (table.get("q_gripper") or {}).get("per_record") or []
    predicted = {(str(row["record_id"]), int(row["tick"])): str(row["predicted"]) for row in rows if row.get("tick") is not None}
    counts = {name: {"n": 0, "correct": 0, "predicted_closed": 0} for name in GRIPPER_TICK_CLASSES}
    episodes_with_initiate = episodes_initiate_all_wrong = 0
    for record in records:
        episode = str(record["episode_id"])
        seen = wrong = 0
        for index, tick in enumerate(record["ticks"]):
            label = next((item for item in tick.get("labels") or () if item.get("question_id") == "q_gripper"), None)
            ids = [str(cid) for cid in (label.get("candidate_ids") or ())] if label else []
            answer = predicted.get((episode, index))
            if answer is None or not ids:
                continue
            executed = str(((tick["request"].get("state") or {}).get("exec") or {}).get("gripper") or "")
            if len(ids) != 1:
                if executed != "closed":
                    counts["window"]["n"] += 1
                    counts["window"]["predicted_closed"] += int(answer == "closed")
                continue
            if ids[0] == "closed":
                kind = "initiate" if executed != "closed" else "settled"
            else:
                kind = "open"
            counts[kind]["n"] += 1
            counts[kind]["predicted_closed"] += int(answer == "closed")
            correct = answer == ids[0]
            counts[kind]["correct"] += int(correct)
            if kind == "initiate":
                seen += 1
                wrong += int(not correct)
        if seen:
            episodes_with_initiate += 1
            episodes_initiate_all_wrong += int(wrong == seen)
    out = {name: {**block, "accuracy": (block["correct"] / block["n"]) if (block["n"] and name != "window") else None} for name, block in counts.items()}
    out["window"]["predicted_closed_rate"] = (counts["window"]["predicted_closed"] / counts["window"]["n"]) if counts["window"]["n"] else None
    out["episodes_with_initiate_ticks"] = episodes_with_initiate
    out["episodes_where_every_initiate_tick_is_wrong"] = episodes_initiate_all_wrong
    out["whole_question_accuracy"] = (table.get("q_gripper") or {}).get("accuracy")
    return out


# --------------------------------------------------------------------------
# 보고서 (C)
# --------------------------------------------------------------------------


def _read_records(directory: Path) -> list[dict[str, Any]]:
    from robo_jev.data.robot_episodes import read_episodes

    return [record for _, record in read_episodes(directory)]


def _offline_columns(paths: list[Path]) -> dict[str, Any]:
    """같은 checkpoint의 오프라인 판정 칸 산출물에서 나란히 적을 값 (칸 전체·선택적 지표·사건 지표)."""
    out: dict[str, Any] = {}
    for path in paths:
        report = json.loads(path.read_text(encoding="utf-8"))
        splits = (report.get("evaluation") or {}).get("splits") or {}
        for name, table in splits.items():
            events = (table.get("event_metrics") or {}).get("model") or {}
            out[f"{path.name}:{name}"] = {
                "run_id": report.get("run_id"), "eval_set": ((report.get("evaluation") or {}).get("eval_set") or {}).get("sha256"),
                "q_main_accuracy": ((table.get("model") or {}).get("q_main") or {}).get("accuracy"),
                "selective_model": (table.get("selective") or {}).get("model"),
                "reaction_delay": events.get("reaction_delay"), "stability": events.get("stability"), "stop_timing": events.get("stop_timing"),
            }
    return out


def closed_loop_report(run_paths: list[Path], *, offline: list[Path] | None = None) -> dict[str, Any]:
    """run 요약 JSON들(`scripts/closed_loop.py run --report`) → 정책 × 조건 지표, 짝지은 성공률 차이, 오프라인 값 나란히."""
    runs: dict[str, dict[str, Any]] = {}
    for path in run_paths:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        runs[str(payload["label"])] = {"path": str(path), "policy": payload["policy"], "conditions": payload["conditions"], "gpu": payload.get("gpu")}
    conditions = sorted({name for run in runs.values() for name in run["conditions"]})
    tables: dict[str, dict[str, Any]] = {}
    rows_by: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for label, run in runs.items():
        for condition in conditions:
            block = run["conditions"].get(condition)
            if block is None:
                continue
            records = _read_records(Path(block["episodes_dir"]))
            rows = [episode_summary(record) for record in records]
            rows_by[(label, condition)] = rows
            tables.setdefault(condition, {})[label] = {**condition_metrics(records, rows), "latency": block.get("latency"), "run_summary": block.get("summary")}
    pairs: dict[str, dict[str, Any]] = {}
    labels = list(runs)
    for condition in conditions:
        for index, a in enumerate(labels):
            for b in labels[index + 1 :]:
                if (a, condition) in rows_by and (b, condition) in rows_by:
                    pairs.setdefault(condition, {})[f"{a} - {b}"] = paired_success(rows_by[(a, condition)], rows_by[(b, condition)])
                    for layer in LAYERS:
                        sub_a = [row for row in rows_by[(a, condition)] if row["layer"] == layer]
                        sub_b = [row for row in rows_by[(b, condition)] if row["layer"] == layer]
                        if sub_a and sub_b:
                            pairs[condition][f"{a} - {b} @ {layer}"] = paired_success(sub_a, sub_b)
    return {"version": CLOSED_LOOP_VERSION, "runs": {label: {"path": run["path"], "policy": run["policy"], "gpu": run["gpu"]} for label, run in runs.items()},
            "conditions": conditions, "tables": tables, "paired": pairs, "offline": _offline_columns(offline or []), "layers": list(LAYERS)}


def _f(value: Any, digits: int = 3) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _ci(block: dict[str, Any] | None) -> str:
    if not block or block.get("margin") is None:
        return "—"
    low, high = block["margin_ci"]
    return f"{block['margin']:+.3f} [{low:+.3f}, {high:+.3f}]" + (" (0 inside)" if block.get("margin_includes_zero") else "")


def print_report(report: dict[str, Any], file: Any = None) -> None:
    """보고서에 옮겨 적을 markdown 표."""
    import sys

    file = file or sys.stdout
    for condition in report["conditions"]:
        table = report["tables"][condition]
        print(f"\n### {condition}\n", file=file)
        print("| policy | n | success | 95 % CI | no-change layer | changes layer | failures main / aux / geometric | completion median ticks | switch rate | round trips | goal-change immediate / censored | q_stop caught / censored / false alarm | unsafe | reject rate | model p95 / >100 ms |", file=file)
        print("| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |", file=file)
        for label, block in table.items():
            layers = block["layers"]
            causes = block["failure_causes"]
            react = (block["reaction_delay"]["adopted"] or {}).get("goal_change") or {}
            stop = block.get("stop_timing") or {}
            latency = block.get("latency") or {}
            model_ms = (latency.get("model_ms") or {}) if latency else {}
            print(
                f"| {label} | {block['episodes']} | {_f(block['success_rate'])} | {_f(block['success_ci'][0]) if block['success_ci'] else '—'}–{_f(block['success_ci'][1]) if block['success_ci'] else '—'} "
                f"| {_f((layers.get(LAYERS[0]) or {}).get('success_rate'))} ({(layers.get(LAYERS[0]) or {}).get('done', 0)}/{(layers.get(LAYERS[0]) or {}).get('episodes', 0)}) "
                f"| {_f((layers.get(LAYERS[1]) or {}).get('success_rate'))} ({(layers.get(LAYERS[1]) or {}).get('done', 0)}/{(layers.get(LAYERS[1]) or {}).get('episodes', 0)}) "
                f"| {causes.get('semantic_main', 0)} / {causes.get('semantic_aux', 0)} / {causes.get('geometric', 0)} | {_f(block['completion_ticks']['median'])} | {_f(block['stability']['adopted']['switch_rate'])} | {block['stability']['adopted']['round_trips']} "
                f"| {_f(react.get('immediate_rate'))} / {_f(react.get('censored_rate'))} | {stop.get('reacted', '—')} / {stop.get('censored', '—')} / {_f(stop.get('false_alarm_rate'))} "
                f"| {_f(block['selective']['unsafe_action_rate'], 4)} | {_f(block['controller']['rejection_rate'], 4)} | {_f(model_ms.get('p95'), 1)} / {_f(model_ms.get('over_100ms_rate'), 4)} |",
                file=file,
            )
        print("\n| pair | seeds | a | b | margin (paired 95 %) |", file=file)
        print("| --- | ---: | ---: | ---: | --- |", file=file)
        for name, block in (report["paired"].get(condition) or {}).items():
            if block:
                print(f"| {name} | {block['seeds']} | {_f(block['a'])} | {_f(block['b'])} | {_ci(block)} |", file=file)
