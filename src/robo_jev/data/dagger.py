"""DAgger식 재라벨링 사이클 — 정책이 실행한 에피소드에 전문가의 답을 `labels`에만 붙인다 (docs/04 §7, docs/08 §7).

    uv run python scripts/dagger_cycle.py --episodes 200 --out artifacts/datasets/d1-robot/dagger-0 [--policy rule_judge]

정책은 `policy(request) -> results`(10답, 모델 출력 형식)다. 오늘은 규칙 기준군(:class:`RuleJudge`)이 그
자리에 서고, 학습 모델의 클라이언트가 같은 인터페이스로 들어온다. 사이클은 :func:`generate_episode`를 그대로
써서 실제로 실행된 것(`model_output`·`adopted`·`ack`·실행 이력·commitment)을 보존하고, 방문한 틱마다 전문가가
지금 고를 답을 `labels`에 `source: expert_v0`·`relabel: true`로 적는다. 입력의 실행 이력·commitment는 바꾸지
않는다. 집계는 정책 오류(전문가의 답이 채택된 주 결정과 다름)·복구(오류 뒤 다시 일치)·commitment 변경·게이트
방문이다.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from robo_jev.contracts import validate_record
from robo_jev.data.episode import aggregate
from robo_jev.data.robot_episodes import (
    build_manifest,
    config_paths,
    generate_episode,
    load_generator_config,
    seed_schedule,
    write_episode,
)
from robo_jev.harness.rule_judge import RULE_JUDGE_VERSION, RuleJudge
from robo_jev.sim.expert import Expert, load_expert_config

__all__ = [
    "DAGGER_VERSION",
    "DEFAULT_CYCLE_OFFSET",
    "PolicyClient",
    "count_policy_behaviour",
    "dagger_seed_schedule",
    "main",
    "relabel",
    "run_cycle",
    "rule_judge_policy",
]

DAGGER_VERSION = "dagger-v0.1"

#: 사이클마다 seed를 띄우는 기본 간격 (설정 `seeds.dagger_cycle_offset`). D1 전문가 배치(`seeds.base`부터 편수만큼)와
#: 겹치지 않을 만큼 크고, 한 사이클(200편)이 다음 사이클의 시작에 닿지 않는다.
DEFAULT_CYCLE_OFFSET = 100_000


def dagger_seed_schedule(config: dict[str, Any], episodes: int, *, cycle: int) -> list[tuple[str, int]]:
    """DAgger 사이클 `cycle`의 seed 일정 — 생성기의 일정(프로파일을 번갈아, `count`의 prefix 성질)을 그대로 두고 시작만
    `seeds.base + (cycle + 1) × seeds.dagger_cycle_offset`으로 띄운다. 전문가 에피소드와 같은 seed·같은 id를 다시 만들지
    않는다(id에는 따로 `-dagger{cycle}`도 붙는다)."""
    seeds = config["seeds"]
    offset = int(seeds.get("dagger_cycle_offset", DEFAULT_CYCLE_OFFSET))
    if offset < 1:
        raise ValueError(f"seeds.dagger_cycle_offset: 1 이상이어야 한다 (받은 값: {offset})")
    if int(cycle) < 0:
        raise ValueError(f"cycle: 0 이상이어야 한다 (받은 값: {cycle})")
    base = int(seeds["base"]) + (int(cycle) + 1) * offset
    return seed_schedule({**config, "seeds": {**seeds, "base": base}}, episodes)


class PolicyClient:
    """`policy(request) -> results`를 생성기의 정책 인터페이스(`act(request, commitment, observation)`)로 감싼다.

    commitment·관측은 넘기지 않는다 — 모델은 요청(모델 입력)만 본다. 이름·버전은 레코드의 provenance에 간다.
    """

    def __init__(self, policy: Callable[[dict[str, Any]], dict[str, Any]], *, name: str, version: str) -> None:
        self.policy = policy
        self.name = name
        self.version = version

    def act(self, request: dict[str, Any], commitment: dict[str, Any] | None = None, observation: Any = None) -> dict[str, Any]:
        del commitment, observation
        return self.policy(request)


def rule_judge_policy(config: dict[str, Any] | None = None) -> PolicyClient:
    """오늘의 정책 대역: 규칙 기준군. 하네스 블록 없이(모델 입력만으로) 답한다."""
    judge = RuleJudge(config) if config is not None else RuleJudge()

    def policy(request: dict[str, Any]) -> dict[str, Any]:
        stripped = {key: value for key, value in request.items() if key != "harness"}
        return judge(stripped)

    return PolicyClient(policy, name="RuleJudge", version=str(getattr(judge, "version", RULE_JUDGE_VERSION)))


def relabel(record: dict[str, Any], *, cycle: int, policy: PolicyClient, seed_base: int | None = None) -> dict[str, Any]:
    """라벨에 재라벨 표지를 붙이고 provenance에 사이클(과 그 사이클의 seed 시작)을 적는다. 실행된 필드는 건드리지 않는다."""
    for tick in record["ticks"]:
        for label in tick.get("labels") or ():
            label["relabel"] = True
    record["provenance"]["dagger"] = {
        "version": DAGGER_VERSION,
        "cycle": int(cycle),
        "seed_base": seed_base,
        "policy": {"name": policy.name, "version": policy.version},
        "relabel_source": "expert_v0",
    }
    return record


def count_policy_behaviour(record: dict[str, Any]) -> dict[str, Any]:
    """정책 오류·복구·commitment 변경·게이트 방문 (docs/04 §7 "정책 오류·복구 상황·commitment 변경이 얼마나").

    정책 오류 = 채택된 주 결정이 전문가 라벨의 허용 집합 밖인 틱. 복구 = 오류 틱 다음에 다시 허용 집합 안으로
    돌아온 틱. commitment 변경 = 채택 주 결정이 직전 틱과 다른 틱(`aggregate`의 `main_changes`). 게이트 방문은
    `usage.gate`에서 센다. 오류는 전문가가 게이트를 원한 틱(`gate_disagreement`)과 다른 결합 후보를 원한 틱
    (`main_disagreement`)으로 나눈다.
    """
    counts = aggregate([record])["per_episode"][0]
    errors = recoveries = 0
    gate_disagreement = main_disagreement = 0
    previous_error = False
    for tick in record["ticks"]:
        adopted = (tick.get("adopted") or {}).get("main")
        label = next((item for item in tick.get("labels") or () if item.get("question_id") == "q_main"), None)
        if adopted is None or label is None:
            previous_error = False
            continue
        allowed = set(label.get("candidate_ids") or ())
        error = adopted not in allowed
        if error:
            errors += 1
            keys = {entry["id"]: str(entry.get("key", "")) for entry in tick["request"]["candidates"]["q_main"]}
            if any(keys.get(cid) in ("observe", "hold", "replan") for cid in allowed):
                gate_disagreement += 1
            else:
                main_disagreement += 1
        elif previous_error:
            recoveries += 1
        previous_error = error
    return {
        "episode_id": record.get("episode_id"),
        "ticks": counts["ticks"],
        "policy_errors": errors,
        "policy_error_rate": round(errors / counts["ticks"], 4) if counts["ticks"] else 0.0,
        "gate_disagreement": gate_disagreement,
        "main_disagreement": main_disagreement,
        "recoveries": recoveries,
        "commitment_changes": counts["main_changes"],
        "switches": counts["switches"],
        "gates": counts["gates"],
        "stops": counts["stops"],
        "conflicts": counts["conflicts"],
        "done": bool(record["provenance"]["outcome"]["done"]),
    }


def run_cycle(
    policy: PolicyClient,
    *,
    episodes: int,
    out: Path,
    config: dict[str, Any] | None = None,
    expert: Expert | None = None,
    cycle: int = 0,
    max_ticks: int | None = None,
    log: Any = None,
) -> dict[str, Any]:
    """정책으로 `episodes`편을 실행·재라벨해 쓰고 집계를 돌려준다.

    seed 일정은 :func:`dagger_seed_schedule` — 생성기의 일정과 같은 꼴이되 사이클마다 `seeds.dagger_cycle_offset`만큼
    띄운 시작에서 센다 — 이고 id에는 `-dagger{cycle}`이 붙는다. 그래서 전문가 배치 옆에 써도 파일·manifest·키프레임
    라벨의 열쇠(id)가 겹치지 않는다. split은 장면 계열에서 나오므로(:func:`new_episode`) 같은 계열의 전문가
    에피소드와 같은 split을 받는다.
    """
    from robo_jev.sim.environment import Environment

    started = time.perf_counter()
    config = config or load_generator_config()
    paths = config_paths(config)
    expert = expert or Expert(load_expert_config(paths["expert_config"]))
    schedule = dagger_seed_schedule(config, episodes, cycle=cycle)
    seed_base = schedule[0][1] if schedule else None
    suffix = f"-dagger{int(cycle)}"
    envs: dict[str, Any] = {}
    per_episode: list[dict[str, Any]] = []
    try:
        for profile, seed in schedule:
            env = envs.get(profile)
            if env is None:
                env = envs[profile] = Environment(config_path=paths["sim_config"], profile=profile)
            record = generate_episode(
                profile, seed, policy=policy, expert=expert, config=config, env=env, max_ticks=max_ticks, id_suffix=suffix,
            )
            relabel(record, cycle=cycle, policy=policy, seed_base=seed_base)
            validate_record(record)
            write_episode(record, out)
            behaviour = count_policy_behaviour(record)
            per_episode.append(behaviour)
            if log is not None:
                print(
                    f"{record['episode_id']} ticks={behaviour['ticks']:<3} done={behaviour['done']} errors={behaviour['policy_errors']} "
                    f"recoveries={behaviour['recoveries']} changes={behaviour['commitment_changes']} gates={behaviour['gates']}",
                    file=log,
                    flush=True,
                )
    finally:
        for env in envs.values():
            env.close()

    manifest = build_manifest(out, config, batch_wall_s=time.perf_counter() - started)
    totals = {
        key: sum(int(entry[key]) for entry in per_episode)
        for key in ("ticks", "policy_errors", "gate_disagreement", "main_disagreement", "recoveries", "commitment_changes", "switches", "stops", "conflicts")
    }
    gates: dict[str, int] = {}
    for entry in per_episode:
        for gate, count in entry["gates"].items():
            gates[gate] = gates.get(gate, 0) + count
    manifest["dagger"] = {
        "version": DAGGER_VERSION,
        "cycle": int(cycle),
        "seed_base": seed_base,
        "id_suffix": suffix,
        "policy": {"name": policy.name, "version": policy.version},
        "relabel_source": "expert_v0",
        "episodes": len(per_episode),
        "done": sum(1 for entry in per_episode if entry["done"]),
        "totals": {**totals, "gates": dict(sorted(gates.items()))},
        "policy_error_rate": round(totals["policy_errors"] / totals["ticks"], 4) if totals["ticks"] else 0.0,
        "per_episode": per_episode,
    }
    (out / "manifest.json").write_bytes((json.dumps(manifest, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="scripts/dagger_cycle.py", description="정책 실행 + 전문가 재라벨 사이클 (docs/04 §7).")
    parser.add_argument("--config", type=Path, default=Path("configs/data/d1_robot.yaml"))
    parser.add_argument("--episodes", type=int, default=200, help="학습 버전마다 200편 (docs/04 §7)")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--policy", choices=["rule_judge"], default="rule_judge", help="정책 대역 (학습 모델 클라이언트는 같은 인터페이스로 붙는다)")
    parser.add_argument("--cycle", type=int, default=0)
    parser.add_argument("--max-ticks", type=int, default=None)
    args = parser.parse_args(argv)
    config = load_generator_config(args.config)
    policy = rule_judge_policy()
    manifest = run_cycle(policy, episodes=args.episodes, out=args.out, config=config, cycle=args.cycle, max_ticks=args.max_ticks, log=sys.stdout)
    dagger = manifest["dagger"]
    print(f"{dagger['episodes']}편 · done {dagger['done']} · 정책 오류율 {dagger['policy_error_rate']} · 합계 {dagger['totals']} → {args.out}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
