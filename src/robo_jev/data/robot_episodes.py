"""로봇 에피소드 생성 — 전문가 + 하네스 + 실제 환경을 10Hz로 돌려 스트림 레코드를 쓴다 (docs/08 §9, docs/04 §3).

    uv run python scripts/generate_episodes.py --config configs/data/d1_robot.yaml \\
        --count 40 --out artifacts/datasets/d1-robot/batch-0 [--resume]

한 에피소드는 이렇게 만든다.

1. **split을 먼저 정한다.** 장면 계획의 구조 서명(물체 수·형상 다중집합·영역 집합)이 장면 계열
   (:func:`family_id`)이고, `origin_group = "robot/<profile>/<계열>/goal-<목표 영역>"`(:func:`origin_group`,
   장면 계획 모듈이 정의한다 — 문구 변형도 이 group의 해시로 고른다)를 계획의 태그(:func:`plan_tags`)와 함께
   :func:`robo_jev.data.split.assign_split`에 넣는다 — 에피소드를 돌리기 **전에**, 그리고 설정의
   holdout 목록(`configs/data/d1_robot.yaml`)으로 (docs/04 §5).
2. 판단 틱마다 하네스가 요청을 만들고(:meth:`RobotHarness.build_request`), **정책**이 답하고
   (D1에서는 전문가 자신, DAgger에서는 학습 모델), 하네스가 조합해(:meth:`compose`) 명령을 내고,
   실행기가 50Hz 제어 5회를 진행한다.
3. 틱마다 `model_output`(정책의 raw 답), `adopted`·`ack`·`exec_history`·`commitment`(실제로
   일어난 것), `labels`(전문가의 답, `source: expert_v0`, 게이팅 규칙 이름)를 **분리 필드**로
   적는다(docs/08 §8). 실행 이력은 어떤 경우에도 라벨로 대체하지 않는다.
4. `done` 게이트 뒤 1초(10틱)의 꼬리까지, 아니면 30초까지 돈다.
5. 배치가 끝나면 에피소드마다 **틱 대조 쌍**(:mod:`robo_jev.data.robot_contrast`; 설정 `contrast.per_episode`)을 만들어
   `contrast/records.jsonl`(`judgment-v0`)에 쓰고 manifest의 `files`·`contrast`에 적는다 — 같은 origin_group·split이다.

레코드는 :func:`robo_jev.contracts.validate_record`를 지나야 하고, 한 에피소드가 JSONL 한 줄
(`episodes/<id>/streams.jsonl` — 자동 QA가 찾는 이름)이며 `manifest.json`이 편수·틱·프로파일별·
split별·버전·시간을 적는다.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import statistics
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import yaml

from robo_jev.contracts import QUESTION_SET_V0, validate_record
from robo_jev.data.episode import aggregate, append_tick, finalize, new_episode
from robo_jev.data.robot_contrast import build_pairs, contrast_summary, default_question_texts
from robo_jev.data.split import CONCEPT_TAG, TEMPLATE_TAG, SplitPolicy, assign_split
from robo_jev.harness.robot import RobotHarness, count_records, load_harness_config
from robo_jev.sim.controller import resolve_config_path
from robo_jev.sim.expert import Expert, load_expert_config
from robo_jev.sim.scene import (
    ScenePlan,
    build_plan,
    family_id,
    family_signature,
    origin_group,
    template_entries,
)

__all__ = [
    "CONTRAST_PATH",
    "DEFAULT_CONFIG_PATH",
    "GENERATOR_VERSION",
    "MANIFEST_VERSION",
    "build_manifest",
    "write_contrast",
    "check_holdout_templates",
    "check_zone_change_excludes",
    "config_paths",
    "episode_id",
    "family_id",
    "family_signature",
    "generate_episode",
    "load_generator_config",
    "main",
    "origin_group",
    "plan_concepts",
    "plan_tags",
    "profile_cycle",
    "run",
    "sealed_goal_zones",
    "seed_schedule",
    "write_episode",
]

GENERATOR_VERSION = "gen-robot-v0.2"
MANIFEST_VERSION = "manifest-robot-v1"
DEFAULT_CONFIG_PATH = "configs/data/d1_robot.yaml"

#: 정책이 내야 하는 답. 이 밖의 키(`phase`·`expert_meta`)는 모델 출력에 넣지 않는다.
QUESTIONS = tuple(QUESTION_SET_V0)

def load_generator_config(path: str | Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    return yaml.safe_load(resolve_config_path(path).read_text(encoding="utf-8"))


def config_digest(config: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(config, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


#: 생성 설정이 이름으로 적어야 하는 설정 파일. 전부 레코드의 `versions.config_digest`에 든다.
CONFIG_PATH_KEYS = ("sim_config", "harness_config", "expert_config", "events_config")


def config_paths(config: dict[str, Any]) -> dict[str, str]:
    """생성 설정이 가리키는 장면·하네스·전문가·사건 설정의 경로. 빠진 키는 기본 경로로 대신하지 않고 거절한다 —
    레코드의 지문(`config_digest`)에 드는 파일은 설정이 이름으로 적은 것이어야 한다."""
    missing = [key for key in CONFIG_PATH_KEYS if not config.get(key)]
    if missing:
        raise ValueError(f"생성 설정에 설정 파일 경로가 없다: {missing} (configs/data/d1_robot.yaml처럼 {list(CONFIG_PATH_KEYS)} 전부를 적는다)")
    return {key: str(config[key]) for key in CONFIG_PATH_KEYS}


def check_holdout_templates(config: dict[str, Any], sim_config: dict[str, Any]) -> None:
    """봉인한 템플릿 변형 id가 장면 설정에 **실제로 있는가** (Task R1 B1).

    템플릿 id는 봉인의 열쇠다(docs/04 §5). 장면 설정의 템플릿을 고치면서 id를 바꾸면 옛 id를 적은 데이터 설정은
    아무것도 봉인하지 않게 되고, 그 사실은 OOD 비율이 조용히 내려가는 것으로만 드러난다 — 그래서 생성 전에 멈춘다.
    """
    declared = [str(name) for name in (config.get("split") or {}).get("holdout_templates") or ()]
    if not declared:
        return
    spec = sim_config.get("instruction") or {}
    known = {
        f"{family}#{template_id}"
        for family in ("v1", "v2")
        for template_id, _ in template_entries(spec, family)
    }
    unknown = sorted(name for name in declared if name not in known)
    if unknown:
        raise ValueError(
            f"split.holdout_templates에 장면 설정이 모르는 변형 id가 있다: {unknown} "
            f"(있는 것: {sorted(known)}) — 템플릿 문장을 바꾸면 봉인 id도 같이 바꾼다"
        )


#: 개념 id에서 봉인한 **목표 영역**을 읽는 접두사 (`robot:goal-zone:zoneF` → `zoneF`).
_GOAL_ZONE_CONCEPT = "robot:goal-zone:"


def sealed_goal_zones(config: dict[str, Any]) -> set[str]:
    """`split.holdout_concepts`가 봉인한 목표 영역 — 도중 지시 변경이 들여오면 안 되는 영역이다."""
    return {
        str(name)[len(_GOAL_ZONE_CONCEPT):]
        for name in (config.get("split") or {}).get("holdout_concepts") or ()
        if str(name).startswith(_GOAL_ZONE_CONCEPT)
    }


def check_zone_change_excludes(config: dict[str, Any], sim_config: dict[str, Any]) -> None:
    """도중 지시 변경이 **봉인 개념의 영역**을 들여오지 못하게 막혀 있는가 (R1 리뷰 1 M13).

    개념 봉인은 에피소드 **계보**(origin group = v1의 목표 영역)의 성질이라, 변경이 봉인 영역을 들여오면 같은
    group의 한 편만 OOD로 가고 나머지는 train에 남는다 — 400편 실측에서 **27 group이 그런 변경을 냈고 그 가운데
    13 group(32편)이 실제로 두 split에 걸쳤다**(리뷰 1 I2). `s0.3`이
    `instruction.zone_change_excludes`로 막았지만 그 목록과 `split.holdout_concepts`를 **묶는 것이 없었다**:
    장면 설정의 기본값은 빈 목록이고, 둘째 영역을 봉인하면서 한쪽만 고치면 누출이 조용히 다시 열린다.
    생성 전에 멈춘다 — 누출은 QA가 잡기 전까지 보이지 않는다.
    """
    sealed = sealed_goal_zones(config)
    if not sealed:
        return
    excluded = {str(name) for name in (sim_config.get("instruction") or {}).get("zone_change_excludes") or ()}
    missing = sorted(sealed - excluded)
    if missing:
        raise ValueError(
            f"split.holdout_concepts가 봉인한 목표 영역이 instruction.zone_change_excludes에 없다: {missing} "
            f"(지금 제외: {sorted(excluded)}) — 도중 지시 변경이 그 영역을 들여오면 한 group이 두 split에 걸친다"
        )


# --------------------------------------------------------------------------
# 장면 계열과 split — 생성 전에 (docs/04 §5)
# --------------------------------------------------------------------------


# `family_signature`·`family_id`·`origin_group`은 :mod:`robo_jev.sim.scene`이 정의한다(계획의 순수 함수; 문구 변형이 group 해시를
# 쓴다) — 여기서는 같은 이름으로 다시 내보낸다.


def plan_concepts(plan: ScenePlan) -> list[str]:
    """계획이 다루는 개념 id (docs/04 §5 개념 계열): 지시의 목표 영역 `robot:goal-zone:<zone>` (v1·v2 모두)."""
    return sorted({f"robot:goal-zone:{step.zone}" for step in plan.instructions if step.zone})


def plan_tags(plan: ScenePlan) -> tuple[str, ...]:
    """생성 **전에** 아는 holdout 태그: 지시의 문구 템플릿 변형(`template:v1#2`)과 개념(`concept:robot:goal-zone:zoneF`)."""
    templates = sorted({TEMPLATE_TAG + str(step.template) for step in plan.instructions if step.template})
    concepts = [CONCEPT_TAG + concept for concept in plan_concepts(plan)]
    return tuple(templates + concepts)


def episode_id(profile: str, seed: int, suffix: str = "") -> str:
    """`ep-<profile>-<seed 6자리>[suffix]`. 전문가 에피소드는 suffix가 없고, DAgger 사이클은 `-dagger{cycle}`을 붙인다
    (:func:`robo_jev.data.dagger.run_cycle`) — id가 파일 경로·manifest·키프레임 라벨의 열쇠라 겹치면 조용히 덮어쓴다."""
    return f"ep-{profile}-{int(seed):06d}{suffix}"


def profile_cycle(config: dict[str, Any]) -> list[str]:
    """한 바퀴의 프로파일 순서. `profile_weights`가 있으면 그 비중대로 섞어 돌린다 (Task R1 B2-iv).

    비중 [20, 40, 40](E0·E1·E2)은 최대공약수로 줄여 [1, 2, 2] → ``[E0, E1, E2, E1, E2]``가 된다 — 한 바퀴가 5편이고
    편수가 5의 배수면 비중이 정확히 맞는다. 비중이 없으면 옛 규칙(균등 번갈기)이다.
    """
    profiles = [str(name) for name in config["profiles"]]
    weights = config.get("profile_weights")
    if not weights:
        return profiles
    if len(weights) != len(profiles):
        raise ValueError(f"profile_weights는 profiles와 길이가 같아야 한다: {weights} vs {profiles}")
    counts = [int(value) for value in weights]
    if any(value <= 0 for value in counts):
        raise ValueError(f"profile_weights는 전부 양수여야 한다: {weights}")
    divisor = math.gcd(*counts) if len(counts) > 1 else counts[0]
    remaining = [value // divisor for value in counts]
    cycle: list[str] = []
    while any(remaining):
        for index, name in enumerate(profiles):
            if remaining[index]:
                cycle.append(name)
                remaining[index] -= 1
    return cycle


def seed_schedule(config: dict[str, Any], count: int) -> list[tuple[str, int]]:
    """결정적 seed 일정. 한 바퀴(:func:`profile_cycle`)를 돌며 프로파일마다 **자기 seed 계수기**를 센다:
    비중 없이 [E0, E1]이면 (E0, 100), (E1, 100), (E0, 101), …이고, [E0, E1, E2] 20/40/40이면
    (E0, 400100), (E1, 400100), (E2, 400100), (E1, 400101), (E2, 400101), (E0, 400101), …이다.

    프로파일마다 따로 세는 까닭: 한 바퀴에 같은 프로파일이 두 번 나오면 seed를 바퀴 번호로 주었을 때 **에피소드
    id가 겹친다**(`ep-E1-000400100`이 둘). 같은 설정·`count`면 언제나 같은 목록이고, `count`를 늘려도 앞부분은
    그대로다(`--resume`의 전제).
    """
    cycle = profile_cycle(config)
    base = int(config["seeds"]["base"])
    # 프로파일마다 seed 구간을 벌린다 (`seeds.profile_offset`). 같은 seed를 두 프로파일에 주면 프로파일 덮어쓰기가
    # 장면을 바꾸지 않는 경우(E1 6~10물체 · E2 7~10물체) **같은 장면이 두 split에 생긴다** — 400편 QA가 실제로
    # 그런 쌍 하나를 잡았다(`ep-E1-400142` train ↔ `ep-E2-400142` ood_test). 0이면 옛 규칙이다(D1은 E0 3물체 ·
    # E1 6~10물체라 겹치지 않았다).
    offset = int(config["seeds"].get("profile_offset", 0))
    order = {name: index for index, name in enumerate(dict.fromkeys(cycle))}
    seen: dict[str, int] = {}
    schedule: list[tuple[str, int]] = []
    for index in range(int(count)):
        name = cycle[index % len(cycle)]
        schedule.append((name, base + offset * order[name] + seen.get(name, 0)))
        seen[name] = seen.get(name, 0) + 1
    return schedule


def split_policy(config: dict[str, Any]) -> SplitPolicy:
    return SplitPolicy.from_config(config.get("split"))


# --------------------------------------------------------------------------
# 에피소드 하나
# --------------------------------------------------------------------------


def generate_episode(
    profile: str,
    seed: int,
    *,
    policy: Any,
    expert: Expert,
    config: dict[str, Any],
    env: Any | None = None,
    max_ticks: int | None = None,
    id_suffix: str = "",
) -> dict[str, Any]:
    """에피소드 하나를 10Hz로 돌려 레코드를 만든다.

    `policy`는 행동하는 쪽이다(`act(request, commitment, observation)` → 10답). D1에서는 `expert`
    자신이고, DAgger에서는 학습 모델의 클라이언트다. `expert`는 라벨의 원천이다. 둘이 같은 객체면
    답을 한 번만 계산한다. `env`를 주면 다시 쓴다(같은 모델 서명이면 reset이 물리를 다시 짓지 않는다).
    `id_suffix`는 에피소드 id 뒤에 붙는다(DAgger의 `-dagger{cycle}`; :func:`episode_id`).
    """
    from robo_jev.sim.environment import Environment

    paths = config_paths(config)
    sim_config = paths["sim_config"]
    episode_config = config.get("episode") or {}
    control_steps = int(episode_config.get("control_steps_per_tick", 5))
    tail_ticks = int(episode_config.get("tail_ticks_after_done", 10))
    harness_config = load_harness_config(paths["harness_config"])

    own_env = env is None
    if own_env:
        env = Environment(config_path=sim_config, profile=profile)
    elif getattr(env, "profile", None) != profile:
        raise ValueError(f"환경의 프로파일이 다르다: {getattr(env, 'profile', None)!r} != {profile!r}")

    started = time.perf_counter()
    try:
        scene = env.reset(seed=int(seed))
        plan = env.plan
        group = origin_group(profile, plan)
        tags = plan_tags(plan)
        splits = split_policy(config)
        tick_ms = int(env.period_ms) * control_steps
        limit = int(max_ticks if max_ticks is not None else env.max_ms // tick_ms)
        record = new_episode(
            episode_id(profile, seed, id_suffix),
            group,
            instructions=[scene["instruction"]],
            policy=splits,
            tags=tags,
        )
        harness = RobotHarness(harness_config)
        commitment = None
        history = None
        done_tick = None  # 지금 이어지는 done 연속 구간의 첫 틱
        first_done_tick = None
        done_streak = 0
        terminated = "max_ticks"
        expert_meta: list[dict[str, Any]] = []
        ticks = 0
        for tick in range(limit):
            request = harness.build_request(scene, history, commitment)
            answers = policy.act(request, commitment, scene)
            reference = answers if policy is expert else expert.act(request, commitment, scene)
            results = {question: copy.deepcopy(answers[question]) for question in QUESTIONS}
            labels = expert.labels(reference, request)
            out = harness.compose(request, results, commitment, int(scene["sim_time_ms"]))

            # 틱 안의 5 제어 스텝에서 난 사건을 **합친다** (gen-robot-v0.2, Task R1 B2-i): 마지막 스텝의 관측만
            # 남기면 스텝 1~4의 사건이 모델 입력에서 사라진다 — D1의 목표 변경 57번 중 `instruction_changed` 줄은
            # 7개뿐이었다(변경은 스텝 경계에 내려오므로 5번에 4번은 버려졌다). ACK와 같은 규칙으로 모은다.
            ack = None
            events: list[dict[str, Any]] = []
            for control_step in range(control_steps):
                scene = env.step(out["command"] if control_step == 0 else None)
                ack = scene["ack"] or ack
                events.extend(scene.get("events") or ())
            scene = {**scene, "events": events}
            usage = {
                "gate": out["gate"],
                "switch": bool(out["switch"]),
                "records": count_records(out["records"]),
                "executor": (ack or {}).get("executor"),
                "applied": bool((ack or {}).get("applied")),
                "ack_reason": (ack or {}).get("reason"),
            }
            append_tick(
                record, request, model_output=results, adopted=out["adopted"], ack=ack, labels=labels, usage=usage
            )
            expert_meta.append({"t": int(request["t"]), **reference["expert_meta"]})
            ticks = tick + 1
            commitment = out["commitment"]
            history = {"adopted": out["adopted"], "ack": ack, "gate": out["gate"]}
            # 꼬리는 **안정된** 완료 1초다: done 게이트가 이어지는 동안만 센다. 파지 판정이 한 틱 흔들리거나
            # 외란이 대상을 영역 밖으로 밀면 연속이 끊기고 에피소드는 계속된다.
            if out["gate"] == "done":
                done_streak += 1
                if done_tick is None:
                    done_tick = tick
                if first_done_tick is None:
                    first_done_tick = tick
            else:
                done_streak = 0
                done_tick = None
            if done_streak >= tail_ticks + 1:
                terminated = "done_tail"
                break
            if harness.stalled_out is not None:
                # 정체 감시가 결정으로 풀 수 없는 정체를 확인했다 (h0.9, R1 리뷰 1 C1): 팔이 물러남으로도
                # 움직이지 않았거나 진행 없이 발동만 되풀이했다. 남은 시간을 죽은 틱으로 채우는 대신
                # **명시적 이유**로 끝낸다 — `max_ms`는 "왜"를 말하지 않는다.
                terminated = "stall_exhausted"
                break
            if scene.get("episode_over"):
                terminated = "max_ms"
                break

        _tolerate_gripper_transitions(record, int((expert.label_config or {}).get("gripper_transition_tolerance_ticks", 0)))
        wall_s = time.perf_counter() - started
        outcome = _outcome(scene, plan, env, done_tick if terminated == "done_tail" else None, ticks, terminated)
        outcome["first_done_tick"] = first_done_tick
        # 정체로 끝났으면 그 까닭(`arm_pinned`·`no_progress`)을 레코드에 남긴다 — "왜 안 끝났나"를 데이터가 말한다.
        outcome["stall"] = harness.stalled_out
        versions = {"expert": expert.version, "generator": GENERATOR_VERSION, "sim": env.serializer_version}
        provenance = {
            "generator": GENERATOR_VERSION,
            "seed": int(seed),
            "profile": profile,
            "origin_group": group,
            "family": family_signature(plan),
            # 봉인 holdout의 근거 (docs/04 §5): 지시의 문구 템플릿 변형과 개념, 걸린 이유 (아니면 빈 목록).
            "instruction_templates": [str(step.template) for step in plan.instructions if step.template],
            "phrasing": [str(step.template) for step in plan.instructions if step.template],
            "concepts": plan_concepts(plan),
            "holdout": splits.holdout_reasons(group, tags),
            # 정책 클라이언트(`data.dagger.PolicyClient`)는 감싼 정책의 이름을 `name`으로 든다.
            "policy": {"name": str(getattr(policy, "name", type(policy).__name__)), "version": str(getattr(policy, "version", "unknown"))},
            "label_source": expert.label_source,
            "config_sha256": config_digest(config),
            "sim_config": sim_config,
            "timing": {"wall_s": round(wall_s, 3), "ticks": ticks, "control_steps_per_tick": control_steps},
            "outcome": outcome,
        }
        evidence = {
            "expert": {"version": expert.version, "ticks": expert_meta},
            "scene_plan": plan.to_json(),
            "disturbance_log": [dict(entry) for entry in scene.get("disturbance_log") or ()],
            "adapter": harness.adapter.evidence(),
        }
        return finalize(
            record,
            versions=versions,
            provenance=provenance,
            evidence=evidence,
            # 지문에 드는 설정: 설정이 이름으로 적은 네 파일(+ 하네스가 가리키는 컨트롤러, 기본 경로의 규칙 기준군)과
            # 이 생성 설정의 `episode.*` 손잡이.
            config_paths={**paths, "generator_config": config},
        )
    finally:
        if own_env:
            env.close()


def _tolerate_gripper_transitions(record: dict[str, Any], tolerance_ticks: int) -> None:
    """그리퍼 전환 틱 ±`tolerance_ticks`는 두 상태를 허용한다 (docs/08 §7 `q_gripper`).

    전환은 전문가 라벨의 원하는 상태가 이웃 틱과 다른 곳이다. 라벨은 그 틱의 commitment에
    조건화된 것이므로 라벨이 있는 틱끼리만 본다.
    """
    if tolerance_ticks <= 0:
        return
    ticks = record["ticks"]
    desired: list[str | None] = []
    for tick in ticks:
        label = next((item for item in tick.get("labels") or () if item["question_id"] == "q_gripper"), None)
        desired.append(str(label["candidate_ids"][0]) if label else None)
    for index, tick in enumerate(ticks):
        if desired[index] is None:
            continue
        window = [
            desired[other]
            for other in range(max(0, index - tolerance_ticks), min(len(ticks), index + tolerance_ticks + 1))
            if desired[other] is not None
        ]
        if len(set(window)) > 1:
            label = next(item for item in tick["labels"] if item["question_id"] == "q_gripper")
            label["candidate_ids"] = ["open", "closed"]
            label["rule"] = f"{label['rule']}+transition-tolerance"


def _outcome(scene: dict[str, Any], plan: ScenePlan, env: Any, done_tick: int | None, ticks: int, terminated: str) -> dict[str, Any]:
    instruction = plan.instructions[env.instruction_version - 1]
    target = next((entry for entry in scene["objects"] if entry["id"] == instruction.target), None)
    zone = next((entry for entry in scene["zones"] if entry["id"] == instruction.zone), None)
    inside = None
    if target is not None and zone is not None:
        x0, y0, x1, y1 = [float(value) for value in zone["bounds_mm"]]
        x, y = float(target["pos_mm"][0]), float(target["pos_mm"][1])
        inside = bool(min(x0, x1) <= x <= max(x0, x1) and min(y0, y1) <= y <= max(y0, y1))
    return {
        "done": done_tick is not None,
        "done_tick": done_tick,
        "ticks": ticks,
        "terminated": terminated,
        "goal_version": int(instruction.version),
        "target": instruction.target,
        "zone": instruction.zone,
        "target_inside_zone": inside,
        "holding": scene["robot"].get("holding"),
        # 놓았다는 판정(`holding`)과 별개로 손가락이 실제로 열렸는지 — 인수 검사가 해제를 단언하는 값.
        "gripper_mm": int(round(float(scene["robot"]["gripper_mm"]))),
        "sim_ms": int(scene["sim_time_ms"]),
    }


# --------------------------------------------------------------------------
# 파일과 manifest
# --------------------------------------------------------------------------


#: 틱 대조 쌍의 파일 (`judgment-v0` 레코드; 자동 QA는 `records.jsonl`을 찾는다).
CONTRAST_PATH = "contrast/records.jsonl"


def episode_path(out: Path, identifier: str) -> Path:
    """`episodes/<id>/streams.jsonl` — 자동 QA(`python -m robo_jev.data.validate`)가 찾는 이름이다."""
    return out / "episodes" / str(identifier) / "streams.jsonl"


def write_contrast(
    records: list[dict[str, Any]], out: Path, config: dict[str, Any], *, expert: Expert, log: Any = None
) -> dict[str, Any]:
    """에피소드들의 틱 대조 쌍을 `contrast/records.jsonl`에 쓰고 집계를 돌려준다 (설정 `contrast`; 없으면 종류별 1, ≤ 4).

    파일은 배치를 다시 셀 때마다 통째로 다시 쓴다(`--resume` 뒤에도 전체 에피소드에서 만든다) — 쌍은 에피소드 id로
    seed한 결정적 선택이라 같은 배치는 같은 파일이다.
    """
    spec = config.get("contrast") or {}
    per_episode = int(spec.get("per_episode", 4))
    kinds = tuple(str(kind) for kind in (spec.get("kinds") or ()))
    texts = default_question_texts(config_paths(config)["harness_config"])
    rows: list[dict[str, Any]] = []
    reasons: dict[str, int] = {}
    for record in records:
        pairs, found = build_pairs(
            record, expert=expert, per_episode=per_episode, question_texts=texts, log=log, **({"kinds": kinds} if kinds else {})
        )
        rows.extend(pairs)
        for key, value in found.items():
            reasons[key] = reasons.get(key, 0) + value
    path = out / CONTRAST_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(json.dumps(row, ensure_ascii=False, sort_keys=False, separators=(",", ":")) + "\n" for row in rows)
    path.write_bytes(payload.encode("utf-8"))
    summary = contrast_summary(rows, reasons)
    summary["per_episode"] = per_episode
    return summary


def write_episode(record: dict[str, Any], out: Path) -> Path:
    """레코드를 제 `episode_id`의 파일에 쓴다 — id가 다르면(DAgger의 suffix) 다른 파일이다."""
    path = episode_path(out, record["episode_id"])
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(record, ensure_ascii=False, sort_keys=False, separators=(",", ":")) + "\n"
    path.write_bytes(payload.encode("utf-8"))
    return path


def read_episodes(out: Path) -> list[tuple[Path, dict[str, Any]]]:
    folder = out / "episodes"
    if not folder.is_dir():
        return []
    found = []
    for path in sorted(folder.glob("*/streams.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                found.append((path, json.loads(line)))
    return found


def _percentile(values: Sequence[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = min(len(ordered) - 1, max(0, int(round(fraction * (len(ordered) - 1)))))
    return float(ordered[position])


def build_manifest(
    out: Path,
    config: dict[str, Any],
    *,
    batch_wall_s: float | None = None,
) -> dict[str, Any]:
    """디렉터리의 에피소드 전부에서 manifest를 만든다 (`--resume` 뒤에도 전체를 다시 센다)."""
    found = read_episodes(out)
    records = [record for _, record in found]
    counts = aggregate(records)
    tick_counts = [len(record["ticks"]) for record in records]
    sizes = [path.stat().st_size for path, _ in found]
    walls = [float(record["provenance"]["timing"]["wall_s"]) for record in records]

    per_profile: dict[str, dict[str, Any]] = {}
    for record in records:
        provenance = record["provenance"]
        entry = per_profile.setdefault(
            provenance["profile"], {"episodes": 0, "done": 0, "ticks": [], "wall_s": []}
        )
        entry["episodes"] += 1
        entry["done"] += int(bool(provenance["outcome"]["done"]))
        entry["ticks"].append(len(record["ticks"]))
        entry["wall_s"].append(float(provenance["timing"]["wall_s"]))
    for entry in per_profile.values():
        ticks = entry.pop("ticks")
        wall = entry.pop("wall_s")
        entry["done_rate"] = round(entry["done"] / entry["episodes"], 4) if entry["episodes"] else 0.0
        entry["ticks_mean"] = round(statistics.fmean(ticks), 1) if ticks else 0.0
        entry["ticks_p95"] = _percentile(ticks, 0.95)
        entry["wall_s_mean"] = round(statistics.fmean(wall), 3) if wall else 0.0

    families: dict[str, int] = {}
    for record in records:
        families[record["origin_group"]] = families.get(record["origin_group"], 0) + 1
    versions: dict[str, set[str]] = {}
    for record in records:
        for name, value in record["versions"].items():
            versions.setdefault(name, set()).add(str(value))

    mean_wall = statistics.fmean(walls) if walls else 0.0
    target_episodes = int(config.get("target_episodes", 400))
    manifest = {
        "version": MANIFEST_VERSION,
        "generator": GENERATOR_VERSION,
        "config_version": str(config.get("version", "")),
        "config_sha256": config_digest(config),
        "config": config,
        "episodes": len(records),
        "ticks": {
            "total": counts["ticks"],
            "mean": round(statistics.fmean(tick_counts), 1) if tick_counts else 0.0,
            "p95": _percentile(tick_counts, 0.95),
            "max": max(tick_counts) if tick_counts else 0,
        },
        "questions": counts["questions"],
        "labels": counts["labels"],
        "per_profile": dict(sorted(per_profile.items())),
        "splits": counts["splits"],
        "families": dict(sorted(families.items())),
        "holdout_prefixes": list((config.get("split") or {}).get("holdout_prefixes") or ()),
        "holdout_templates": list((config.get("split") or {}).get("holdout_templates") or ()),
        "holdout_concepts": list((config.get("split") or {}).get("holdout_concepts") or ()),
        "versions": {name: sorted(values) for name, values in sorted(versions.items())},
        "bytes": {
            "total": sum(sizes),
            "per_episode_mean_mb": round(statistics.fmean(sizes) / 1e6, 3) if sizes else 0.0,
            "per_tick_mean_kb": round(sum(sizes) / counts["ticks"] / 1e3, 2) if counts["ticks"] else 0.0,
        },
        "timing": {
            "wall_s_per_episode_mean": round(mean_wall, 3),
            "wall_s_per_episode_p95": round(_percentile(walls, 0.95), 3),
            "episodes_per_hour": round(3600.0 / mean_wall, 1) if mean_wall else 0.0,
            "projected_hours_for_target": round(target_episodes * mean_wall / 3600.0, 3) if mean_wall else 0.0,
            "target_episodes": target_episodes,
            "batch_wall_s": round(batch_wall_s, 3) if batch_wall_s is not None else None,
        },
        "decisions": {
            "gates": counts["gates"],
            "stops": counts["stops"],
            "switches": counts["switches"],
            "main_changes": counts["main_changes"],
            "conflicts": counts["conflicts"],
            "per_episode": counts["per_episode"],
        },
        # 학습 적재기(`robo_jev.sampler.load_items`)와 비로봇 manifest(`data/generate.py`)가 읽는 꼴 — 경로 → {sha256, …}.
        "files": {
            str(path.relative_to(out)): {
                "episode_id": record["episode_id"],
                "profile": record["provenance"]["profile"],
                "seed": record["provenance"]["seed"],
                "split": record["split"],
                "origin_group": record["origin_group"],
                "done": bool(record["provenance"]["outcome"]["done"]),
                "ticks": len(record["ticks"]),
                "bytes": path.stat().st_size,
                "records": 1,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
            for path, record in found
        },
    }
    contrast_file = out / CONTRAST_PATH
    if contrast_file.is_file():
        # 틱 대조 쌍(`judgment-v0`)도 적재기의 파일 목록에 든다 — 스트림과 같은 split·계열이며 schema로 갈린다.
        rows = [line for line in contrast_file.read_text(encoding="utf-8").splitlines() if line.strip()]
        manifest["files"][CONTRAST_PATH] = {
            "kind": "contrast",
            "records": len(rows),
            "bytes": contrast_file.stat().st_size,
            "sha256": hashlib.sha256(contrast_file.read_bytes()).hexdigest(),
        }
    (out / "manifest.json").write_bytes(
        (json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=False) + "\n").encode("utf-8")
    )
    return manifest


# --------------------------------------------------------------------------
# 배치
# --------------------------------------------------------------------------


def excluded_groups(config: dict[str, Any]) -> set[str]:
    """D-OOD용 (docs/04 §6): `exclude_groups_from`(manifest 경로 목록)의 `families` 키 = 이미 쓴 origin group. 그 계열의 seed는 건넌다."""
    groups: set[str] = set()
    for path in config.get("exclude_groups_from") or ():
        manifest = json.loads(Path(path).read_text(encoding="utf-8"))
        groups.update(str(name) for name in (manifest.get("families") or {}))
    return groups


def run(
    config: dict[str, Any],
    count: int,
    out: Path,
    *,
    resume: bool = False,
    log: Any = None,
) -> dict[str, Any]:
    """seed 일정대로 에피소드를 만들고 manifest를 쓴다. `resume`이면 이미 있는 seed는 건넌다.

    `exclude_groups_from`(D-OOD)이 있으면 origin group이 그 manifest들의 계열에 있는 seed는 건너뛰고(`excluded`로 센다) 일정을
    앞으로 늘려 `count`편을 채운다 — D1과 겹치지 않는 장면·목표 계열만 남는다.
    """
    from robo_jev.sim.environment import Environment

    started = time.perf_counter()
    paths = config_paths(config)
    expert = Expert(load_expert_config(paths["expert_config"]))
    exclude = excluded_groups(config)
    sim_settings = yaml.safe_load(resolve_config_path(paths["sim_config"]).read_text(encoding="utf-8"))
    # 봉인한 템플릿 id가 장면 설정에 실제로 있는지 **생성 전에** 본다 (Task R1 B1).
    check_holdout_templates(config, sim_settings)
    check_zone_change_excludes(config, sim_settings)
    schedule = seed_schedule(config, count if not exclude else count * int(config.get("exclude_schedule_factor", 8)))
    envs: dict[str, Any] = {}
    produced = skipped = excluded = 0
    try:
        for profile, seed in schedule:
            if produced + skipped >= count:
                break
            if exclude:
                group = origin_group(profile, build_plan(sim_settings, int(seed), profile))
                if group in exclude:
                    excluded += 1
                    continue
            path = episode_path(out, episode_id(profile, seed))
            if resume and path.is_file():
                skipped += 1
                continue
            env = envs.get(profile)
            if env is None:
                env = envs[profile] = Environment(config_path=paths["sim_config"], profile=profile)
            record = generate_episode(profile, seed, policy=expert, expert=expert, config=config, env=env)
            validate_record(record)
            write_episode(record, out)
            produced += 1
            if log is not None:
                outcome = record["provenance"]["outcome"]
                print(
                    f"{record['episode_id']} {record['split']:<11} ticks={outcome['ticks']:<3} "
                    f"done={outcome['done']} wall={record['provenance']['timing']['wall_s']:.1f}s",
                    file=log,
                    flush=True,
                )
    finally:
        for env in envs.values():
            env.close()
    contrast = write_contrast([record for _, record in read_episodes(out)], out, config, expert=expert, log=log)
    manifest = build_manifest(out, config, batch_wall_s=time.perf_counter() - started)
    manifest["contrast"] = contrast
    manifest["run"] = {"requested": int(count), "produced": produced, "skipped": skipped, "excluded": excluded, "resume": bool(resume)}
    (out / "manifest.json").write_bytes(
        (json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=False) + "\n").encode("utf-8")
    )
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="scripts/generate_episodes.py",
        description="로봇 에피소드 스트림 레코드를 만든다 (docs/08 §9).",
    )
    parser.add_argument("--config", type=Path, default=Path(DEFAULT_CONFIG_PATH), help="생성 설정 YAML")
    parser.add_argument("--count", type=int, required=True, help="만들 에피소드 수 (프로파일을 번갈아 센다)")
    parser.add_argument("--out", type=Path, required=True, help="episodes/*.jsonl과 manifest.json을 쓸 디렉터리")
    parser.add_argument("--resume", action="store_true", help="이미 있는 seed의 에피소드는 건넌다")
    args = parser.parse_args(argv)

    config = load_generator_config(args.config)
    manifest = run(config, args.count, args.out, resume=args.resume, log=sys.stdout)
    timing = manifest["timing"]
    print(
        f"{manifest['episodes']}편 · {manifest['ticks']['total']}틱 · split {manifest['splits']} → {args.out}"
    )
    print(
        f"  {timing['episodes_per_hour']}편/시간 · 평균 {manifest['ticks']['mean']}틱 (p95 {manifest['ticks']['p95']}) · "
        f"{manifest['bytes']['per_episode_mean_mb']}MB/편 · 400편 예상 {timing['projected_hours_for_target']}시간"
    )
    for profile, entry in manifest["per_profile"].items():
        print(f"  {profile}: {entry['episodes']}편 done {entry['done']} ({entry['done_rate']})")
    contrast = manifest.get("contrast") or {}
    print(f"  대조 쌍 {contrast.get('pairs', 0)} (종류 {contrast.get('by_kind')}, split {contrast.get('by_split')}, 삭제 {contrast.get('deletion_outcomes')}) → {CONTRAST_PATH}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
