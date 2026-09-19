"""키프레임 counterfactual rollout — 사건 실행, 키프레임·후보 선택, 주 결정 라벨 (docs/04 §4, docs/08 §7).

네 가지를 한다.

* :func:`rollout_event` — snapshot 하나에서 후보 하나를 **고정된 후속 정책**으로 실행해 `success` /
  `failure` / `censored`와 근거(docs/04 §4의 사건 기록 필드)를 돌려준다. 사건 정의(horizon·성공 기준·
  외란 분포·후속 정책 버전)는 `configs/sim/events.yaml`에 있다. 같은 snapshot + 같은 seed는 같은 궤적이고,
  seed가 다르면 자세 흔들기·컨트롤러 잡음만 다르다(paired seed).
* :func:`select_keyframes` — 에피소드마다 키프레임 5틱: 사건 틱(전환·지시 변경·관측된 물체 이동·정지) +
  무작위 채움. 종류별 포함 수를 적는다. 라벨·결과는 보지 않는다.
* :func:`choose_rollout_candidates` — 키프레임의 결합 후보 8개: 현재 commitment와 전문가 선택을 반드시
  넣고, 나머지는 접근점의 farthest-point 표본(기하 다양성)으로 채운다. 정답을 모른 채 고른다.
* :func:`label_main_decision` — docs/08 §7의 규칙: 의미 적합성 → 적합 후보 안의 성공률·신뢰구간 비교 →
  commitment 규칙(`ε`, 하한). 결과는 허용 집합 + `unknown` + `event_results`(seed별) + `label_confidence`.

후속 정책은 전문가의 부가 답 위에 주 결정을 후보 하나로 못박고 게이팅을 끈 :class:`FollowupPolicy`다.
정지 규칙(`q_stop`)은 전문가 것을 그대로 쓴다 — 후속 정책도 안전 규칙 아래에서 움직인다.
"""

from __future__ import annotations

import copy
import math
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from robo_jev.contracts import QUESTION_SET_V0
from robo_jev.harness.robot import FIXED_KEYS, RobotHarness, candidate_id, joint_key_parts, load_harness_config
from robo_jev.sim.controller import resolve_config_path
from robo_jev.sim.expert import DEGENERATE_REASONS, GATE_REASONS, Expert, load_expert_config

__all__ = [
    "DEFAULT_EVENTS_PATH",
    "EVENT_FIELDS",
    "FollowupPolicy",
    "choose_rollout_candidates",
    "event_for",
    "label_main_decision",
    "load_events_config",
    "rollout_event",
    "select_keyframes",
    "summarise_results",
    "wilson_interval",
]

DEFAULT_EVENTS_PATH = "configs/sim/events.yaml"
DEFAULT_SIM_CONFIG = "configs/sim/tidy_clutter.yaml"

#: 사건 기록이 빠짐없이 가져야 하는 필드 (docs/04 §4). `rollout_seeds`·`successes`·`failures`·`censored_trials`는
#: 후보 단위의 집계(:func:`summarise_results`)에 있고, rollout 하나의 근거에는 자기 seed와 결과가 있다.
EVENT_FIELDS = (
    "event_id",
    "observation",
    "action_ref",
    "followup_policy_version",
    "horizon_seconds",
    "success_rule",
    "randomization_distribution",
    "rollout_seed",
)

#: 밀기 방향 벡터 (하네스와 같은 이름).
_PUSH_VECTORS = {"+x": (1.0, 0.0), "-x": (-1.0, 0.0), "+y": (0.0, 1.0), "-y": (0.0, -1.0)}

#: 키프레임 사건 종류와 그것을 알아보는 규칙 (docs/04 §4 "이벤트·전술 변경 시점").
KEYFRAME_KINDS = ("switch", "instruction", "moved", "stop")

#: 규칙 라벨이 정답인 근거 — 이 틱의 `q_main` 정답은 규칙 후보이고 rollout으로 바꾸지 않는다. 게이트(완료·지시·관측)에
#: 더해 하네스가 한 틱 동안 재시도를 막은 틱(`way_retry_blocked`), 실행기 사정으로 고를 수 없는 틱(`not_executable`),
#: 목표를 실현할 후보가 목록에 없는 틱(`goal_candidate_missing`)도 그렇다: rollout은 그 차단·사정을 모른 채
#: (:func:`rollout_event`는 `history=None`으로 시작한다) 후보를 실행하므로 그 결과로 규칙의 `hold`를 뒤집으면 안 된다.
#: 퇴화 셋은 hold∉A 규칙(docs/08 §7)이라 적합 후보가 `unknown`으로 간다 — 전문가의 상수를 그대로 쓴다.
_GATE_REASONS = GATE_REASONS + DEGENERATE_REASONS

_QUESTIONS = tuple(QUESTION_SET_V0)


class _SimulatorError(Exception):
    """simulator 경계(환경 생성·복원·관측·자세 이동·step) 안에서 난 예외. 원인은 `__cause__`."""


def _simulator(call, *args, **kwargs):
    """simulator 호출을 감싼다: 안에서 난 예외는 `_SimulatorError`로 표시해 밖의 코드 결함과 가른다."""
    try:
        return call(*args, **kwargs)
    except Exception as error:
        raise _SimulatorError(f"{type(error).__name__}: {error}") from error


def load_events_config(path: str | Path = DEFAULT_EVENTS_PATH) -> dict[str, Any]:
    return yaml.safe_load(resolve_config_path(path).read_text(encoding="utf-8"))


def event_for(key: str, *, holding: str | None, config: dict[str, Any]) -> dict[str, Any]:
    """후보의 의미 키 → 사건 정의 (설정의 공통 부분을 합친 dict).

    들고 있는 대상의 `grasp` 후보는 진행 중인 결합 행동(남은 절반: 옮기기)이므로 `place` 사건이다.
    """
    parts = joint_key_parts(key)
    if parts is None:
        raise ValueError(f"결합 후보가 아니다 (사건이 없다): {key!r}")
    function, target = parts[0], parts[1]
    for spec in config["events"].values():
        applies = spec["applies_to"]
        if function not in applies["function"]:
            continue
        if "holding_target" in applies and bool(applies["holding_target"]) != (holding == target):
            continue
        followup = config["followup"]
        return {
            **copy.deepcopy(spec),
            "function": function,
            "version": str(config["version"]),
            "followup_policy_version": str(config["followup_policy_version"]),
            "randomization": copy.deepcopy(config["randomization"]),
            "harness_config": followup["harness_config"],
            "expert_config": followup["expert_config"],
            "control_steps_per_tick": int(followup["control_steps_per_tick"]),
            "wall_time_limit_s": float(followup["wall_time_limit_s"]),
        }
    raise ValueError(f"기능 {function!r}(holding={holding!r})에 맞는 사건이 없다: {key!r}")


# --------------------------------------------------------------------------
# 후속 정책
# --------------------------------------------------------------------------


class FollowupPolicy:
    """고정 후속 정책: 전문가의 부가 답·정지 규칙 위에 주 결정을 후보 하나로 못박고 게이팅을 끈다.

    `q_main`은 후보에 전문가의 선택 질량을 주고, `q_done`·`q_observe`는 거짓, `q_instr`·`q_retry`는 참이다
    (사건은 이 행동을 끝까지 실행하는 것이며 재시도도 그 일부다). 후보가 목록에 없는 틱은 전문가 답을
    그대로 낸다 — 호출자가 그것을 검열 사유로 본다.
    """

    def __init__(self, expert: Expert, key: str, *, version: str | None = None) -> None:
        self.expert = expert
        self.key = key
        self.candidate = candidate_id(key)
        self.version = version or f"followup-v0/expert-{expert.version}"

    def act(self, request: dict[str, Any], commitment: dict[str, Any] | None, observation: Any = None) -> dict[str, Any]:
        answers = self.expert.act(request, commitment, observation)
        ids = [entry["id"] for entry in request["request"]["candidates"]["q_main"]]
        if self.candidate in ids:
            answers["q_main"] = self.expert._spread(self.candidate, ids)
        high, low = float(self.expert.confidence["high"]), float(self.expert.confidence["low"])
        answers.update(q_done=low, q_instr=high, q_observe=low, q_retry=high)
        return answers


# --------------------------------------------------------------------------
# 성공 기준
# --------------------------------------------------------------------------


class _Rule:
    """틱마다 관측을 보고 `"success"` / `("failure", 이유)` / `None`을 낸다. horizon 끝은 호출자가 실패로 본다."""

    def __init__(self, event: dict[str, Any], key: str, start_scene: dict[str, Any], harness: RobotHarness) -> None:
        self.spec = event["success_rule"]
        self.kind = str(self.spec["kind"])
        parts = joint_key_parts(key)
        if parts is None:
            raise ValueError(f"결합 후보가 아니다 (사건이 없다): {key!r}")
        self.target, self.approach, self.destination = parts[1], parts[2], parts[3]
        self.start_pose = list(self._pose(start_scene))
        self.zone = next((zone for zone in start_scene["zones"] if str(zone["id"]) == self.destination), None)
        self.push_segment_mm = float(harness.candidates_config["push_segment_mm"])
        self.tick_ms = int(event.get("tick_ms", 100))
        self.streak = 0
        self.max_contact_n = 0.0
        self.recent: list[list[float]] = []

    def _pose(self, scene: dict[str, Any]) -> list[float]:
        entry = next((item for item in scene["objects"] if str(item["id"]) == self.target), None)
        return [float(value) for value in entry["pos_mm"]] if entry else [math.nan] * 3

    def observe_step(self, scene: dict[str, Any]) -> None:
        self.max_contact_n = max(self.max_contact_n, float(scene["robot"]["contact_force_n"]))

    def check(self, scene: dict[str, Any]) -> str | tuple[str, str] | None:
        pose = self._pose(scene)
        holding = scene["robot"].get("holding")
        if self.kind == "held_above":
            ticks_needed = max(1, int(round(float(self.spec["hold_seconds"]) * 1000 / self.tick_ms)))
            lifted = holding == self.target and pose[2] - self.start_pose[2] >= float(self.spec["lift_height_mm"])
            self.streak = self.streak + 1 if lifted else 0
            return "success" if self.streak >= ticks_needed else None
        if self.kind == "released_inside_zone_at_rest":
            ticks_needed = max(1, int(round(float(self.spec["rest_seconds"]) * 1000 / self.tick_ms)))
            self.recent.append(pose)
            self.recent = self.recent[-(ticks_needed + 1):]
            if holding is not None or self.zone is None:
                return None
            x0, y0, x1, y1 = [float(value) for value in self.zone["bounds_mm"]]
            inside = min(x0, x1) <= pose[0] <= max(x0, x1) and min(y0, y1) <= pose[1] <= max(y0, y1)
            at_rest = len(self.recent) > ticks_needed and all(
                math.dist(self.recent[-1], earlier) <= float(self.spec["rest_mm"]) for earlier in self.recent[:-1]
            )
            return "success" if inside and at_rest else None
        if self.kind == "displaced_along":
            if self.max_contact_n > float(self.spec["max_contact_force_n"]):
                return ("failure", "contact_force")
            along = self.displacement_along(pose)
            return "success" if along >= float(self.spec["segment_fraction"]) * self.push_segment_mm else None
        raise ValueError(f"모르는 성공 기준이다: {self.kind!r}")

    def displacement_along(self, pose: list[float]) -> float:
        vector = _PUSH_VECTORS.get(self.approach, (0.0, 0.0))
        return (pose[0] - self.start_pose[0]) * vector[0] + (pose[1] - self.start_pose[1]) * vector[1]


# --------------------------------------------------------------------------
# rollout 하나
# --------------------------------------------------------------------------


def rollout_event(
    snapshot: bytes,
    action: dict[str, Any],
    event: dict[str, Any],
    seed: int,
    *,
    env: Any | None = None,
    expert: Expert | None = None,
) -> dict[str, Any]:
    """snapshot에서 `action`(후보: `key`, `id`, 선택적 `precision_mm`)을 `event`의 후속 정책으로 실행한다.

    돌려주는 것은 `{"outcome": success|failure|censored, "reason", "evidence": {...}}`다. 정상적으로 horizon을
    관측했으나 기준에 못 미치면 `failure`(이유 `horizon`, 밀기의 `contact_force`), simulator 예외·벽시계
    한계·후속 정책이 후보를 볼 수 없는 경우는 `censored`(사유 보존, docs/04 §4).

    `env`를 주면 다시 쓴다(복원만 한다). 같은 snapshot·seed는 어느 인스턴스에서든 같은 궤적이다.
    """
    from robo_jev.sim.environment import Environment

    started = time.perf_counter()
    key = str(action["key"])
    action_ref = str(action.get("id") or action.get("action_ref") or candidate_id(key))
    limit_s = float(event.get("wall_time_limit_s", 60.0))
    control_steps = int(event.get("control_steps_per_tick", 5))
    evidence: dict[str, Any] = {
        "event_id": str(event["event_id"]),
        "event_version": str(event.get("version", "")),
        "observation": None,
        "action_ref": action_ref,
        "key": key,
        "followup_policy_version": str(event["followup_policy_version"]),
        "horizon_seconds": float(event["horizon_seconds"]),
        "success_rule": copy.deepcopy(event["success_rule"]),
        "randomization_distribution": copy.deepcopy(event["randomization"]),
        "rollout_seed": int(seed),
        "ticks": 0,
        "first_success_tick": None,
        # 접근 시간을 구간 성과와 가르는 값 (리뷰 라운드 1): 채택 국면이 approach를 벗어난 첫 틱, 대상과의 접촉이
        # 처음 시작된 틱, 그리고 그 시각(초).
        "first_action_tick": None,
        "first_contact_tick": None,
        "approach_s": None,
        "contact_s": None,
        "trajectory": [],
        "max_contact_n": 0.0,
        "start_pose_mm": None,
        "end_pose_mm": None,
        "holding_at_end": None,
        "displacement_along_mm": None,
        "lift_mm": None,
        "restore_s": None,
        "wall_s": None,
    }

    def finish(outcome: str, reason: str | None) -> dict[str, Any]:
        evidence["wall_s"] = round(time.perf_counter() - started, 4)
        return {"outcome": outcome, "reason": reason, "evidence": evidence}

    own_env = env is None
    try:
        try:
            if own_env:
                header = _simulator(Environment.describe_snapshot, snapshot)
                env = _simulator(Environment, config_path=str(event.get("sim_config") or DEFAULT_SIM_CONFIG), profile=str(header["profile"]))
            restore_started = time.perf_counter()
            _simulator(env.restore, snapshot)
            evidence["restore_s"] = round(time.perf_counter() - restore_started, 4)

            rng = np.random.default_rng([int(seed), 0x5EED])
            env.rng = rng
            scene = _simulator(env.observe)
            evidence["observation"] = {"sim_ms": int(scene["sim_time_ms"]), "tick": int(scene["tick"])}
            evidence["randomization_distribution"]["applied"] = _simulator(_apply_jitter, env, scene, action, event["randomization"], rng)
            env.setpoint_noise_mm = float(event["randomization"]["controller_noise"]["setpoint_sigma_mm"])
            scene = _simulator(env.observe)

            expert = expert or Expert(load_expert_config(event.get("expert_config") or "configs/sim/expert_v0.yaml"))
            harness = RobotHarness(load_harness_config(event.get("harness_config") or "configs/harness/robot.yaml"))
            policy = FollowupPolicy(expert, key, version=str(event["followup_policy_version"]))
            tick_ms = int(env.period_ms) * control_steps
            rule = _Rule({**event, "tick_ms": tick_ms}, key, scene, harness)
            horizon_ticks = max(1, int(round(float(event["horizon_seconds"]) * 1000 / tick_ms)))
            evidence["start_pose_mm"] = list(rule.start_pose)

            commitment = None
            history = None
            for tick in range(horizon_ticks):
                # 실행할 후보는 commitment처럼 예약한다 — 자세 흔들기로 영역 쪽 축이 바뀌어도 실행 가능하면 목록에 남는다.
                request = harness.build_request(scene, history, commitment, keep_key=key)
                if tick == 0 and policy.candidate not in {entry["id"] for entry in request["request"]["candidates"]["q_main"]}:
                    return finish("censored", "candidate_unavailable")
                answers = policy.act(request, commitment, scene)
                results = {question: answers[question] for question in _QUESTIONS}
                out = harness.compose(request, results, commitment, int(scene["sim_time_ms"]))
                ack = None
                for step in range(control_steps):
                    scene = _simulator(env.step, out["command"] if step == 0 else None)
                    ack = scene["ack"] or ack
                    rule.observe_step(scene)
                    if evidence["first_contact_tick"] is None and any(
                        str(item.get("kind")) == "contact_onset" and str(item.get("object")) == rule.target
                        for item in scene.get("events") or ()
                    ):
                        evidence["first_contact_tick"] = tick + 1
                        evidence["contact_s"] = round((tick + 1) * tick_ms / 1000.0, 2)
                commitment = out["commitment"]
                history = {"adopted": out["adopted"], "ack": ack, "gate": out["gate"]}
                evidence["ticks"] = tick + 1
                if evidence["first_action_tick"] is None and str(out["adopted"].get("phase")) not in ("approach", "none"):
                    evidence["first_action_tick"] = tick + 1
                    evidence["approach_s"] = round(tick * tick_ms / 1000.0, 2)
                evidence["trajectory"].append(_trajectory_row(scene, rule.target, out["adopted"]))
                verdict = rule.check(scene)
                if verdict == "success":
                    evidence["first_success_tick"] = tick + 1
                    break
                if isinstance(verdict, tuple):
                    _finish_evidence(evidence, scene, rule)
                    return finish("failure", verdict[1])
                if time.perf_counter() - started > limit_s:
                    _finish_evidence(evidence, scene, rule)
                    return finish("censored", "wall_time_limit")
            _finish_evidence(evidence, scene, rule)
            if evidence["first_success_tick"] is not None:
                return finish("success", None)
            return finish("failure", "horizon")
        except _SimulatorError as error:  # simulator 오류 → censoring, 사유 보존 (docs/04 §4)
            cause = error.__cause__
            evidence["error"] = str(error)
            return finish("censored", f"simulator_error:{type(cause).__name__}")
        except Exception as error:  # 그 밖의 예외는 코드 결함 — 제 이름으로 센다
            evidence["error"] = f"{type(error).__name__}: {error}"
            return finish("censored", f"pipeline_error:{type(error).__name__}")
    finally:
        if own_env and env is not None:
            env.close()


def _apply_jitter(env: Any, scene: dict[str, Any], action: dict[str, Any], randomization: dict[str, Any], rng) -> dict[str, Any]:
    """물체 자세를 보고된 정밀도 안에서 흔든다 (들고 있는 물체는 제외). 적용한 값을 돌려준다."""
    jitter = randomization["pose_jitter"]
    precision = action.get("precision_mm") or {}
    default = float(jitter.get("default_precision_mm", 0.0))
    yaw = float(jitter.get("yaw_deg", 0.0))
    holding = scene["robot"].get("holding")
    applied: dict[str, dict[str, float]] = {}
    for entry in scene["objects"]:
        object_id = str(entry["id"])
        radius = float(precision.get(object_id, default)) if jitter.get("xy_within_precision", True) else default
        dx, dy = (float(value) for value in rng.uniform(-radius, radius, size=2))
        dyaw = float(rng.uniform(-yaw, yaw))
        if object_id == holding:
            continue
        env.nudge(object_id, (dx, dy), dyaw)
        applied[object_id] = {"dx_mm": round(dx, 3), "dy_mm": round(dy, 3), "dyaw_deg": round(dyaw, 3)}
    return applied


def _trajectory_row(scene: dict[str, Any], target: str, adopted: dict[str, Any]) -> list[Any]:
    entry = next((item for item in scene["objects"] if str(item["id"]) == target), None)
    pose = [int(value) for value in entry["pos_mm"]] if entry else [0, 0, 0]
    ee = [int(value) for value in scene["robot"]["ee_pos_mm"]]
    return [
        int(scene["sim_time_ms"]),
        *ee,
        *pose,
        round(float(scene["robot"]["contact_force_n"]), 2),
        1 if scene["robot"].get("holding") == target else 0,
        str(adopted.get("phase")),
    ]


def _finish_evidence(evidence: dict[str, Any], scene: dict[str, Any], rule: _Rule) -> None:
    pose = rule._pose(scene)
    evidence["end_pose_mm"] = [round(value, 1) for value in pose]
    evidence["max_contact_n"] = round(rule.max_contact_n, 2)
    evidence["holding_at_end"] = scene["robot"].get("holding")
    evidence["displacement_along_mm"] = round(rule.displacement_along(pose), 1)
    evidence["lift_mm"] = round(pose[2] - rule.start_pose[2], 1)


# --------------------------------------------------------------------------
# 키프레임 선택 (docs/04 §4: 사건·전술 변경 틱 + 무작위, 종류별 포함률)
# --------------------------------------------------------------------------


def select_keyframes(record: dict[str, Any], config: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    """에피소드 레코드 → 키프레임 목록 `[{"t", "index", "kind", ...}]`와 종류별 포함 수.

    사건 틱: `switch`(채택 주 결정이 직전 틱과 다르다 — 첫 채택 포함), `instruction`(목표 버전 변경),
    `moved`(상태의 `object_moved` 사건), `stop`(정지 구간의 첫 틱). 게이트 틱(완료·지시·관측)은 규칙 라벨
    틱이라 뺀다. 라벨·결과·rollout은 보지 않는다. 사건 틱이 모자라면 무작위로 채우고, 넘치면 종류를 돌아가며
    뽑는다 — 둘 다 에피소드 id로 seed한 난수라 결정적이다.
    """
    spec = (config or {}).get("keyframes") or {}
    per_episode = int(spec.get("per_episode", 5))
    ticks = record["ticks"]
    eligible = [index for index, tick in enumerate(ticks) if not (tick.get("usage") or {}).get("gate")]
    eligible_set = set(eligible)

    by_kind: dict[str, list[int]] = {kind: [] for kind in KEYFRAME_KINDS}
    previous_main = None
    previous_version = None
    stopping = False
    for index, tick in enumerate(ticks):
        adopted = tick.get("adopted") or {}
        main = adopted.get("main")
        version = ((tick.get("request") or {}).get("state") or {}).get("goal", {}).get("version")
        events = ((tick.get("request") or {}).get("state") or {}).get("events") or ()
        if index in eligible_set:
            if main is not None and main != previous_main:
                by_kind["switch"].append(index)
            if previous_version is not None and version != previous_version:
                by_kind["instruction"].append(index)
            if any(str(event.get("kind")) == "object_moved" for event in events):
                by_kind["moved"].append(index)
            if adopted.get("stop") and not stopping:
                by_kind["stop"].append(index)
        if main is not None:
            previous_main = main
        previous_version = version
        stopping = bool(adopted.get("stop"))

    rng = random.Random(f"keyframes:{record.get('episode_id')}")
    chosen: list[tuple[int, str]] = []
    taken: set[int] = set()
    pools = {kind: list(indices) for kind, indices in by_kind.items()}
    for pool in pools.values():
        rng.shuffle(pool)
    # 종류를 돌아가며 하나씩 — 한 종류가 키프레임을 독식하지 않는다.
    while len(chosen) < per_episode and any(pools.values()):
        for kind in KEYFRAME_KINDS:
            while pools[kind]:
                index = pools[kind].pop()
                if index not in taken:
                    chosen.append((index, kind))
                    taken.add(index)
                    break
            if len(chosen) >= per_episode:
                break
    remaining = [index for index in eligible if index not in taken]
    rng.shuffle(remaining)
    while len(chosen) < per_episode and remaining:
        index = remaining.pop()
        chosen.append((index, "random"))
        taken.add(index)
    chosen.sort()

    inclusion = {
        kind: {"available": len(by_kind[kind]), "included": sum(1 for _, chosen_kind in chosen if chosen_kind == kind)}
        for kind in KEYFRAME_KINDS
    }
    inclusion["random"] = {"available": len(eligible), "included": sum(1 for _, kind in chosen if kind == "random")}
    keyframes = []
    for index, kind in chosen:
        tick = ticks[index]
        keyframes.append(
            {
                "episode_id": record.get("episode_id"),
                "index": index,
                "t": int(tick["t"]),
                "sim_ms": int(tick["sim_ms"]),
                "kind": kind,
                "kinds": sorted(name for name, indices in by_kind.items() if index in indices),
                "inclusion": inclusion,
            }
        )
    return keyframes


# --------------------------------------------------------------------------
# 후보 선택 (commitment + 전문가 선택 + 기하 다양성)
# --------------------------------------------------------------------------


def choose_rollout_candidates(
    tick: dict[str, Any],
    commitment: dict[str, Any] | None,
    expert_choice: str | None,
    k: int = 8,
    *,
    harness_config: dict[str, Any] | None = None,
) -> list[str]:
    """키프레임 틱의 결합 후보 가운데 rollout할 `k`개의 id.

    현재 commitment의 후보와 전문가 선택(결합 후보일 때)을 반드시 넣고, 나머지는 접근점(파지·밀기 접촉점·
    놓기점)의 farthest-point 표본으로 채운다 — 같은 지점의 프로파일 변형보다 다른 대상·방향이 먼저다.
    라벨·rollout 결과·확률은 보지 않는다. 결정적이다.
    """
    request = tick.get("request") or tick
    state = request["state"]
    entries = [entry for entry in request["candidates"]["q_main"] if str(entry.get("key", "")) not in FIXED_KEYS]
    points = {entry["id"]: _approach_point(str(entry["key"]), state, harness_config) for entry in entries}
    ids = [entry["id"] for entry in entries]
    if not ids:
        return []

    chosen: list[str] = []
    for forced in ((commitment or {}).get("action_ref"), expert_choice):
        if forced in points and forced not in chosen:
            chosen.append(forced)
    if not chosen:
        ee = [float(value) for value in state["robot"]["ee_pose_mm"]]
        chosen.append(min(ids, key=lambda cid: (math.dist(ee, points[cid]), cid)))
    while len(chosen) < min(k, len(ids)):
        remaining = [cid for cid in ids if cid not in chosen]
        chosen.append(
            max(remaining, key=lambda cid: (min(math.dist(points[cid], points[other]) for other in chosen), -_key_rank(cid, entries)))
        )
    return chosen[:k]


def _key_rank(cid: str, entries: list[dict[str, Any]]) -> int:
    return next(index for index, entry in enumerate(entries) if entry["id"] == cid)


def _approach_point(key: str, state: dict[str, Any], harness_config: dict[str, Any] | None) -> list[float]:
    """후보의 접근점 (모델 입력의 상태에서 하네스와 같은 정의로). 파지 → 윗면 위, 밀기 → 접촉점, 놓기 → 영역 중심."""
    spec = (harness_config or load_harness_config())["candidates"]
    parts = joint_key_parts(key)
    if parts is None:
        return [0.0, 0.0, 0.0]
    function, target, approach, destination = parts
    holding = state["robot"].get("holding")
    entry = next((item for item in state.get("objects") or () if str(item["id"]) == target), None)
    if function == "place" or (function == "grasp" and holding == target):
        zone = next((item for item in state.get("zones") or () if str(item["id"]) == destination), None)
        if zone is not None:
            x0, y0, x1, y1 = [float(value) for value in zone["bounds_mm"]]
            return [(x0 + x1) / 2.0, (y0 + y1) / 2.0, float(state["scene"].get("work_surface_mm", 0.0)) + 100.0]
    if entry is None:
        return [0.0, 0.0, 0.0]
    pose = [float(value) for value in entry["pose_mm"]]
    if function == "push":
        vector = _PUSH_VECTORS.get(approach, (0.0, 0.0))
        obb = [float(value) for value in entry["obb_mm"]]
        reach = math.hypot(obb[0] / 2.0, obb[1] / 2.0) + float(spec["push_contact_mm"])
        return [pose[0] - vector[0] * reach, pose[1] - vector[1] * reach, pose[2]]
    return [pose[0], pose[1], float(entry.get("top_mm", pose[2])) + float(spec["approach_clearance_mm"])]


# --------------------------------------------------------------------------
# 주 결정 라벨 (docs/08 §7)
# --------------------------------------------------------------------------


def wilson_interval(successes: int, trials: int, z: float = 1.0) -> tuple[float, float]:
    """Wilson score 구간. `trials`는 censoring을 뺀 유효 반복 수다. 0회면 (0, 1)."""
    if trials <= 0:
        return (0.0, 1.0)
    p = successes / trials
    denominator = 1.0 + z * z / trials
    centre = (p + z * z / (2.0 * trials)) / denominator
    half = z * math.sqrt(p * (1.0 - p) / trials + z * z / (4.0 * trials * trials)) / denominator
    return (max(0.0, centre - half), min(1.0, centre + half))


def summarise_results(rollouts: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """rollout 결과 목록 → 후보별 집계 `{cid: {s, f, censored, seeds, rollout_seeds, censored_reasons}}`.

    `seeds`는 seed 순서의 결과(1 성공 · 0 실패 · -1 censored)이고 `rollout_seeds`는 그 seed 값이다.
    """
    summary: dict[str, dict[str, Any]] = {}
    for result in sorted(rollouts, key=lambda item: (item["evidence"]["action_ref"], item["evidence"]["rollout_seed"])):
        evidence = result["evidence"]
        entry = summary.setdefault(
            evidence["action_ref"],
            {"s": 0, "f": 0, "censored": 0, "seeds": [], "rollout_seeds": [], "censored_reasons": [], "event_id": evidence["event_id"]},
        )
        entry["rollout_seeds"].append(int(evidence["rollout_seed"]))
        if result["outcome"] == "success":
            entry["s"] += 1
            entry["seeds"].append(1)
        elif result["outcome"] == "failure":
            entry["f"] += 1
            entry["seeds"].append(0)
        else:
            entry["censored"] += 1
            entry["seeds"].append(-1)
            entry["censored_reasons"].append(str(result["reason"]))
    return summary


def label_main_decision(tick: dict[str, Any], results: dict[str, dict[str, Any]], config: dict[str, Any] | None = None) -> dict[str, Any]:
    """docs/08 §7의 `q_main` 라벨.

    (1) `semantic_admissible`은 구조화된 목표의 규칙에서 온다 — 틱에 전문가 라벨이 있으면 그 값을, 없으면
    전문가를 다시 불러 계산한다. (2) 적합 후보 가운데 rollout이 있는 것의 성공률과 Wilson 구간(z는 설정,
    유효 반복 수는 censoring을 뺀 수)을 비교한다. (3) 현재 commitment가 적합하고 최고 후보와의 차이가 `ε` 이내이며
    하한이 기준 이상이면 그 후보만 정답. 아니면 최고 후보의 `ε` 안이고 하한이 기준 이상인 후보의 허용 집합.
    근거가 약하면(하한 기준을 넘는 후보가 없다, 전부 실패) 전문가의 선택을 낮은 신뢰도로 둔다. rollout하지
    않은 결합 후보는 `unknown`. 규칙 라벨 틱(:data:`_GATE_REASONS` — 게이트(완료·지시·관측), 재시도 차단, 실행기
    사정, 목표 후보 없음)의 정답은 규칙 후보이고 그 신뢰도 그대로이며 rollout으로 바꾸지 않는다; 퇴화 셋은 hold∉A
    규칙이라 적합 후보가 rollout이 있어도 `unknown`이다. 낮은 신뢰도의 결과는 전문가 라벨과 같은 `weight`
    (`labels.low_confidence_weight`)를 단다 — rollout 라벨이 전문가 라벨의 weight를 떨어뜨리지 않는다.

    `results`는 :func:`summarise_results`의 형태다. 돌려주는 것은 라벨 dict 하나(계약 필드 + `rule` + `source`).
    """
    spec = (config or {}).get("label") or {}
    epsilon = float(spec.get("epsilon", 0.15))
    lower_bound = float(spec.get("success_lower_bound", 0.5))
    z = float(spec.get("interval_z", 1.0))
    seeds = int((config or {}).get("seeds", 8))
    width_high = float(spec.get("width_high", 0.35))
    width_medium = float(spec.get("width_medium", 0.6))

    request = tick.get("request") or tick
    candidates = request["candidates"]["q_main"]
    keys = {entry["id"]: str(entry.get("key", "")) for entry in candidates}
    joint = [cid for cid, key in keys.items() if key not in FIXED_KEYS]

    existing = next((label for label in tick.get("labels") or () if label.get("question_id") == "q_main"), None)
    if existing is None:
        expert = Expert()
        answers = expert.act(tick, None)
        existing = next(label for label in expert.labels(answers, tick) if label["question_id"] == "q_main")
    admissible = [cid for cid in existing.get("semantic_admissible") or () if cid in keys]
    expert_choice = list(existing.get("candidate_ids") or [])
    rule_reason = str(existing.get("rule", "")).rsplit("/", 1)[-1]
    # 낮은 신뢰도의 weight: 전문가 라벨의 값이 있으면 그것, 없으면 전문가 설정의 값 (같은 단일 출처).
    low_weight = existing.get("weight")
    if low_weight is None:
        configured = float(load_expert_config().get("labels", {}).get("low_confidence_weight", 0.25))
        low_weight = configured if configured != 1.0 else None

    rolled = {cid: entry for cid, entry in results.items() if cid in keys}
    event_results = {
        cid: {"s": int(entry["s"]), "f": int(entry["f"]), "censored": int(entry.get("censored", 0)), "seeds": list(entry["seeds"])}
        for cid, entry in rolled.items()
    }
    unknown = [cid for cid in joint if cid not in rolled]
    stats: dict[str, dict[str, float]] = {}
    for cid in admissible:
        entry = rolled.get(cid)
        if entry is None:
            continue
        trials = int(entry["s"]) + int(entry["f"])
        low, high = wilson_interval(int(entry["s"]), trials, z=z)
        stats[cid] = {"p": (int(entry["s"]) / trials) if trials else 0.0, "low": low, "high": high, "n": trials, "censored": int(entry.get("censored", 0))}

    base = {
        "question_id": "q_main",
        "kind": "valid_set",
        "semantic_admissible": list(admissible),
        "event_results": event_results,
        "source": "rollout_v0",
        "rule": f"admissible-then-performance-v1+commitment-v1/wilson-z{z:g}",
        "rollout_rule": {"epsilon": epsilon, "success_lower_bound": lower_bound, "interval": "wilson", "z": z, "seeds": seeds},
    }

    def finish(candidate_ids: list[str], confidence: str, reason: str, *, extra_unknown: list[str] = ()) -> dict[str, Any]:
        label = {**base, "candidate_ids": list(candidate_ids), "label_confidence": confidence, "rollout_reason": reason}
        label["unknown"] = [cid for cid in joint if (cid in unknown or cid in extra_unknown) and cid not in candidate_ids]
        if confidence == "low" and low_weight is not None:
            label["weight"] = float(low_weight)
        return label

    if rule_reason in _GATE_REASONS and expert_choice:
        # 퇴화 틱(hold∉A): 적합 후보는 판단하지 않는다 — rollout 결과는 event_results에 남되 정규화에서 빠진다.
        extra = admissible if rule_reason in DEGENERATE_REASONS else []
        return finish(expert_choice, str(existing.get("label_confidence", "high")), f"gate:{rule_reason}", extra_unknown=extra)
    if not stats:
        return finish(expert_choice or admissible[:1] or list(keys)[:1], "low", "no_rollout_evidence")

    best = max(stats.values(), key=lambda item: item["p"])["p"]
    commitment = (request.get("commitment") or {}).get("action_ref")
    winners = [cid for cid, stat in stats.items() if best - stat["p"] <= epsilon and stat["low"] >= lower_bound]

    def confidence_of(ids: list[str]) -> str:
        widths = [stats[cid]["high"] - stats[cid]["low"] for cid in ids]
        censored = any(stats[cid]["censored"] > 0 for cid in ids)
        thin = any(stats[cid]["n"] < max(1, seeds // 2) for cid in ids)
        if thin:
            return "low"
        if max(widths) <= width_high and not censored:
            return "high"
        if max(widths) <= width_medium:
            return "medium"
        return "low"

    if commitment in stats and best - stats[commitment]["p"] <= epsilon and stats[commitment]["low"] >= lower_bound:
        return finish([commitment], confidence_of([commitment]), "commitment_kept")
    if winners:
        return finish(sorted(winners), confidence_of(winners), "performance_allowed_set" if len(winners) > 1 else "performance_unique")
    if best <= 0.0:
        fallback = [cid for cid in expert_choice if cid in admissible] or sorted(stats)
        return finish(fallback, "low", "all_failed")
    # 최고 후보들이 있으나 하한이 기준에 못 미친다 — 복수 허용, 낮은 신뢰도 (docs/08 §7 "근거가 약하면").
    top = sorted(cid for cid, stat in stats.items() if best - stat["p"] <= epsilon)
    return finish(top, "low", "weak_evidence")
