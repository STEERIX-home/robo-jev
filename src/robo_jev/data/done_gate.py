"""done 게이트 수집 — 거짓 done **뒤의** 상태를 방문하는 DAgger 수집 전용 정책 (Task R6 A3; docs/04 §7).

왜. R5의 모델 주행 기록에는 거짓 done 뒤가 없다 — 하네스의 done 게이트가 정책 자신의 `q_done`이라(`harness/robot.py`
`_gate`) 모델이 거짓으로 done을 말하면 꼬리 3틱 뒤 편이 끝난다. 그래서 "거짓 done 상태에서 expert라면 무엇을 하는가"(새 대상을
잡으러 간다)가 라벨로 한 번도 붙지 않았다. DAgger의 β-혼합을 **종료 판단 하나에만** 건다: :class:`DoneGatePolicy`는 모델의 답을
그대로 쓰되 하네스에 넘기는 `q_done`만 **그 틱에 expert가 내는 `q_done`**으로 바꾼다. 그러면 모델이 거짓 done을 내도 편은
끝나지 않고, 그 뒤 모델이 새 대상으로 가는지·멈추는지가 expert 참조 라벨과 함께 기록된다.

기록의 자리. 생성기(:func:`robo_jev.data.robot_episodes.generate_episode`)는 정책이 돌려준 답을 `model_output`에 쓴다 — 이 정책의
답이므로 `model_output.q_done`은 **expert의 값**(하네스가 실제로 받은 것)이다. 모델의 raw `q_done`은 틱마다 `usage.model_q_done`에,
편의 요약은 `evidence.done_gate`에 남긴다(:func:`attach_raw_done`) — 둘 다 계약 검증이 받는 자리다(`usage`·`evidence`는 dict이면
된다; `model_output`의 라벨 유출 검사와 무관하다). 층·거짓 done 셈(:mod:`robo_jev.data.done_strata`)은 `usage.model_q_done`을 먼저
읽는다.

**평가에는 절대 쓰지 않는다.** 이 정책은 종료를 expert에게 맡기므로 성공률이 뜻을 잃는다. 막는 것 셋: 정책의
`collection_only` 표지를 평가 실행기(:func:`robo_jev.closed_loop.run_condition`)가 거절하고, 평가의 정책 목록
(`closed_loop.POLICY_KINDS`)에 없으며, 수집은 **train 분할의 장면**만 받는다(:func:`collect_done_gate` — 계획을 지어 분할을 먼저 보고,
다른 분할이면 돌리기 전에 거절한다; ood_dev·ood_test 계열은 학습에 들어가지 않는다).
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from robo_jev.contracts import validate_record

__all__ = [
    "COLLECTION_NAME",
    "DONE_GATE_VERSION",
    "DoneGatePolicy",
    "attach_raw_done",
    "collect_done_gate",
    "collection_seed_base",
]

#: 감싸개 버전 — 레코드의 `provenance.policy.version`은 ``dg0.1/<안쪽 정책의 버전>``이다.
DONE_GATE_VERSION = "dg0.1"
#: 수집 방식의 이름 — DAgger 데이터셋의 `provenance.dagger.collection`(:func:`robo_jev.data.gripper_labels.build_dagger_dataset`).
COLLECTION_NAME = "expert_done_gate"


class DoneGatePolicy:
    """모델의 답 + expert의 `q_done` (모듈 설명). **수집 전용** — `collection_only`가 참이고 평가 실행기는 이 정책을 거절한다."""

    name = "DoneGatePolicy"
    collection_only = True

    def __init__(self, inner: Any, expert: Any) -> None:
        self.inner = inner
        self.expert = expert
        self.version = f"{DONE_GATE_VERSION}/{getattr(inner, 'version', 'unknown')}"
        self.raw: list[dict[str, Any]] = []

    @property
    def timing(self) -> list[dict[str, Any]]:
        return list(getattr(self.inner, "timing", []) or [])

    def reset(self) -> None:
        """새 에피소드 — 안쪽 정책을 되돌리고 raw 기록을 비운다."""
        if hasattr(self.inner, "reset"):
            self.inner.reset()
        self.raw = []

    def act(self, request: dict[str, Any], commitment: dict[str, Any] | None = None, observation: Any = None) -> dict[str, Any]:
        answers = self.inner.act(request, commitment, observation)
        # expert의 답은 참조와 같은 인자로 — `q_done`은 상태와 목표만 읽는다(`Expert._goal_satisfied`); commitment·관측은 쓰지 않는다
        reference = self.expert.act(request, commitment, observation)
        raw = answers.get("q_done")
        self.raw.append({"t": int(request["t"]), "model_q_done": None if raw is None else float(raw), "expert_q_done": float(reference["q_done"])})
        return {**answers, "q_done": float(reference["q_done"])}


def attach_raw_done(record: dict[str, Any], raw: list[dict[str, Any]], *, policy_version: str) -> dict[str, Any]:
    """틱마다 모델의 raw `q_done`을 `usage.model_q_done`에, 편 요약을 `evidence.done_gate`에 적는다. 틱 번호가 어긋나면 거절한다."""
    ticks = record["ticks"]
    if len(raw) != len(ticks) or any(int(entry["t"]) != int(tick["t"]) for entry, tick in zip(raw, ticks)):
        raise ValueError(f"{record.get('episode_id')}: raw `q_done` 기록이 틱과 맞지 않는다 ({len(raw)}개 대 틱 {len(ticks)}개)")
    for tick, entry in zip(ticks, raw):
        usage = tick.setdefault("usage", {})
        usage["model_q_done"] = entry["model_q_done"]
    evidence = record.setdefault("evidence", {})
    evidence["done_gate"] = {
        "collection": COLLECTION_NAME, "policy": policy_version,
        "rule": "the harness received the expert's q_done on every tick; the model's raw q_done is usage.model_q_done",
        "model_q_done_over_gate_ticks": sum(1 for entry in raw if entry["model_q_done"] is not None and entry["model_q_done"] >= 0.5),
        "expert_q_done_over_gate_ticks": sum(1 for entry in raw if entry["expert_q_done"] >= 0.5),
    }
    return record


def collection_seed_base(config: dict[str, Any], *, cycle: int) -> int:
    """이 사이클의 수집 seed 시작 = `dagger_seed_schedule(cycle)`의 시작 (`seeds.base + (cycle + 1) × dagger_cycle_offset`).
    r1 400100+ · R4 900100+ · R5 950100+ · R6 평가 980100+와 겹치지 않는다(사이클 1 = 600100)."""
    from robo_jev.data.dagger import DEFAULT_CYCLE_OFFSET

    seeds = config["seeds"]
    return int(seeds["base"]) + (int(cycle) + 1) * int(seeds.get("dagger_cycle_offset", DEFAULT_CYCLE_OFFSET))


def _scene_split(config: dict[str, Any], profile: str, seed: int) -> str:
    """계획만 지어(물리 없음) 그 장면의 분할을 본다 — 생성기가 에피소드를 돌리기 **전에** 정하는 것과 같은 규칙."""
    import yaml

    from robo_jev.data.robot_episodes import config_paths, origin_group, plan_tags, split_policy
    from robo_jev.sim.controller import resolve_config_path
    from robo_jev.sim.scene import build_plan

    sim = yaml.safe_load(resolve_config_path(config_paths(config)["sim_config"]).read_text(encoding="utf-8"))
    plan = build_plan(sim, int(seed), profile)
    return split_policy(config).assign(origin_group(profile, plan), plan_tags(plan))


def collect_done_gate(
    policy: DoneGatePolicy, schedule: list[tuple[str, int]], *, config: dict[str, Any], out: str | Path, label: str,
    id_tag: str = "r6", max_ticks: int | None = None, log: Any = None, describe: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """`schedule`의 장면마다 `policy`로 편을 돌려 raw `q_done`을 붙여 쓴다(레코드 + manifest). 장면은 **train 분할**이어야 한다."""
    from robo_jev.data.dagger import count_policy_behaviour
    from robo_jev.data.robot_episodes import build_manifest, config_paths, generate_episode, write_episode
    from robo_jev.sim.environment import Environment
    from robo_jev.sim.expert import Expert, load_expert_config

    if not getattr(policy, "collection_only", False):
        raise ValueError("collect_done_gate: 수집 정책(DoneGatePolicy)만 받는다")
    for profile, seed in schedule:
        split = _scene_split(config, profile, seed)
        if split != "train":
            raise ValueError(f"{profile}:{seed}: 분할이 {split}이다 — done 게이트 수집은 train 계열 장면만 돈다 (ood_dev·ood_test 계열은 학습에 넣지 않는다)")
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    paths = config_paths(config)
    reference = Expert(load_expert_config(paths["expert_config"]))
    envs: dict[str, Any] = {}
    started = time.perf_counter()
    rows: list[dict[str, Any]] = []
    try:
        for profile, seed in schedule:
            env = envs.get(profile)
            if env is None:
                env = envs[profile] = Environment(config_path=paths["sim_config"], profile=profile)
            policy.reset()
            record = generate_episode(profile, seed, policy=policy, expert=reference, config=config, env=env, max_ticks=max_ticks, id_suffix=f"-{id_tag}-{label}")
            if record["split"] != "train":  # 계획으로 본 분할과 같아야 한다 — 다르면 쓰지 않는다
                raise ValueError(f"{record['episode_id']}: 기록의 분할이 {record['split']}이다")
            attach_raw_done(record, policy.raw, policy_version=policy.version)
            validate_record(record)
            write_episode(record, out)
            behaviour = count_policy_behaviour(record)
            outcome = record["provenance"]["outcome"]
            over = record["evidence"]["done_gate"]["model_q_done_over_gate_ticks"]
            rows.append({"episode_id": record["episode_id"], "profile": profile, "seed": int(seed), "ticks": len(record["ticks"]),
                         "done": bool(outcome["done"]), "inside": bool(outcome.get("target_inside_zone")), "terminated": outcome.get("terminated"),
                         "model_q_done_over_gate_ticks": over, "policy_errors": behaviour["policy_errors"]})
            if log is not None:
                print(f"{record['episode_id']} ticks={len(record['ticks']):<3} done={outcome['done']} inside={outcome.get('target_inside_zone')} "
                      f"model_q_done≥0.5 ticks={over} errors={behaviour['policy_errors']}", file=log, flush=True)
    finally:
        for env in envs.values():
            env.close()
    wall = time.perf_counter() - started
    manifest = build_manifest(out, config, batch_wall_s=wall)
    manifest["closed_loop"] = {
        "version": DONE_GATE_VERSION, "condition": "train", "label": label, "id_tag": id_tag, "episodes": len(rows),
        "collection": COLLECTION_NAME, "collection_only": True,
        # 안쪽 정책의 서술(checkpoint·model_id·kind…)을 싣되 이름·버전은 감싸개의 것이다 — 안쪽 버전은 `inner_version`에
        "policy": {**(describe or {}), "name": policy.name, "version": policy.version, "inner_version": (describe or {}).get("version")},
        "note": "done-gate collection for DAgger training only — the harness received the expert's q_done; never an evaluation",
    }
    (out / "manifest.json").write_bytes((json.dumps(manifest, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))
    return {"episodes": len(rows), "wall_seconds": round(wall, 1), "rows": rows, "out": str(out)}
