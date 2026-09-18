"""에피소드 스트림 레코드의 조립과 집계 (docs/08 §8).

데이터의 독립 단위는 **에피소드**다. 이 모듈은 세 가지만 한다.

* :func:`new_episode` — prefix(지시·질문 세트)와 split을 정해 빈 레코드를 연다. 분할은
  **장면 계열 단위로 생성 전에** 배정하므로(docs/08 §8) 계열 id 하나만 받는다.
* :func:`append_tick` — 하네스가 만든 틱 요청에 `model_output`·`adopted`·`ack`·`labels`를
  **분리 필드로** 붙여 레코드에 넣는다. 하네스 블록(기하·회계)은 레코드에 가지 않는다.
  지시가 바뀐 틱에서는 prefix에 새 지시를 덧붙인다(리셋하지 않는다, docs/08 §3.1).
* :func:`finalize` — 버전(하네스·컨트롤러·전문가·규칙·serializer·추출기)과 provenance를
  붙이고 계약 검사를 돌린다. 어기면 그 자리에서 `ValueError`다.

:func:`aggregate`는 에피소드·틱·질문 수를 센다. 세는 규칙은 자동 QA
(:mod:`robo_jev.data.validate`)와 같아야 하므로 "그 틱이 실제로 던진 질문"만 센다.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

from robo_jev.contracts import QUESTION_SET_V0, SCHEMA_STREAM, validate_record
from robo_jev.data.split import SplitPolicy, assign_split

__all__ = [
    "CONFIG_DIGEST_PARTS",
    "DEFAULT_CONFIG_PATHS",
    "REQUIRED_VERSIONS",
    "aggregate",
    "append_tick",
    "config_digest",
    "default_question_set",
    "default_versions",
    "finalize",
    "new_episode",
    "posed_questions",
    "running_config_digest",
]

#: 레코드가 반드시 적어야 하는 버전 (docs/08 §8). `expert`는 전문가 에피소드에만 붙는다. `config_digest`는
#: 레코드를 만든 설정 묶음의 지문이다 — 버전 문자열을 올리지 않은 수치 변경도 레코드에 남긴다.
REQUIRED_VERSIONS = ("harness", "controller", "rules", "serializer", "extractor", "config_digest")

#: `config_digest`에 들어가는 설정과 그 순서. 컨트롤러 설정은 하네스 설정이 가리키는 파일이다.
CONFIG_DIGEST_PARTS = ("harness", "controller", "expert", "sim", "events")

DEFAULT_CONFIG_PATHS = {
    "harness_config": "configs/harness/robot.yaml",
    "expert_config": "configs/sim/expert_v0.yaml",
    "sim_config": "configs/sim/tidy_clutter.yaml",
    "events_config": "configs/sim/events.yaml",
}

#: 요청 안에만 사는 하네스 장부. 레코드에는 넣지 않는다.
_HARNESS_BLOCK = "harness"


def config_digest(configs: dict[str, Any]) -> str:
    """설정 묶음(`{이름: 설정 dict}`)의 지문 — 키를 정렬한 canonical JSON의 sha256."""
    payload = json.dumps(configs, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def running_config_digest(
    *,
    harness_config: str | Path | None = None,
    expert_config: str | Path | None = None,
    sim_config: str | Path | None = None,
    events_config: str | Path | None = None,
) -> str:
    """지금 디스크의 하네스·컨트롤러(하네스 설정이 가리키는 파일)·전문가·장면·사건 설정의 :func:`config_digest`.

    경로를 주지 않으면 기본 설정(`DEFAULT_CONFIG_PATHS`)이다. 생성기는 자기가 실제로 쓴 경로를 준다.
    """
    import yaml

    from robo_jev.harness.robot import load_harness_config
    from robo_jev.sim.controller import load_controller_config, resolve_config_path

    def read(path: str | Path) -> Any:
        return yaml.safe_load(resolve_config_path(path).read_text(encoding="utf-8"))

    harness = load_harness_config(harness_config or DEFAULT_CONFIG_PATHS["harness_config"])
    parts = {
        "harness": harness,
        "controller": load_controller_config(harness["controller_config"]),
        "expert": read(expert_config or DEFAULT_CONFIG_PATHS["expert_config"]),
        "sim": read(sim_config or DEFAULT_CONFIG_PATHS["sim_config"]),
        "events": read(events_config or DEFAULT_CONFIG_PATHS["events_config"]),
    }
    return config_digest({name: parts[name] for name in CONFIG_DIGEST_PARTS})


def default_versions(
    *,
    harness_config: str | Path | None = None,
    expert_config: str | Path | None = None,
    sim_config: str | Path | None = None,
    events_config: str | Path | None = None,
) -> dict[str, str]:
    """설정과 모듈 상수에서 읽은 기본 버전 + 설정 묶음의 지문. 부를 때마다 설정을 다시 읽는다.

    하네스·규칙·추출기는 코드가, 컨트롤러·serializer는 설정이 단일 출처다. 한 군데서
    모아야 레코드의 `versions`가 실제로 돌아간 것을 가리킨다. 캐시하지 않는다 — 한 과정
    안에서 설정을 바꾸면 그 뒤의 레코드는 바뀐 버전을 적어야 한다. 경로를 주면 그 설정으로
    (생성기가 실제로 쓴 것), 아니면 기본 설정으로 digest를 만든다.
    """
    import yaml

    from robo_jev.harness.robot import HARNESS_VERSION, load_harness_config
    from robo_jev.harness.rule_judge import RULE_JUDGE_VERSION
    from robo_jev.perception.pointworld import EXTRACTOR_VERSION
    from robo_jev.sim.controller import load_controller_config, resolve_config_path

    harness = load_harness_config(harness_config or DEFAULT_CONFIG_PATHS["harness_config"])
    controller = load_controller_config(harness["controller_config"])
    serializer = yaml.safe_load(
        resolve_config_path(sim_config or DEFAULT_CONFIG_PATHS["sim_config"]).read_text(encoding="utf-8")
    )
    return {
        "harness": HARNESS_VERSION,
        "controller": str(controller.get("version", "c0")),
        "rules": RULE_JUDGE_VERSION,
        "serializer": str(serializer.get("version", "s0")),
        "extractor": EXTRACTOR_VERSION,
        "config_digest": running_config_digest(
            harness_config=harness_config, expert_config=expert_config, sim_config=sim_config, events_config=events_config
        ),
    }


def default_question_set() -> str:
    """기본 하네스 설정의 질문 세트 id. 하네스(앞단·설정 전체)를 만들지 않고 설정만 읽는다."""
    from robo_jev.harness.robot import load_harness_config

    config = load_harness_config()
    return str(config["question_set_id"][str(config.get("language", "ko"))])


def new_episode(
    episode_id: str,
    scene_family: str,
    *,
    instructions: list[dict[str, Any]],
    question_set: str | None = None,
    policy: SplitPolicy | None = None,
) -> dict[str, Any]:
    """빈 스트림 레코드. split은 장면 계열에서 나온다 (docs/08 §8)."""
    if not instructions:
        raise ValueError("prefix에는 시작 지시가 하나 이상 있어야 한다")
    if question_set is None:
        question_set = default_question_set()
    return {
        "schema_version": SCHEMA_STREAM,
        "episode_id": str(episode_id),
        "origin_group": str(scene_family),
        "split": assign_split(str(scene_family), policy),
        "prefix": {
            "instructions": [_instruction(item) for item in instructions],
            "question_set": str(question_set),
        },
        "ticks": [],
    }


def append_tick(
    record: dict[str, Any],
    request: dict[str, Any],
    *,
    model_output: dict[str, Any] | None = None,
    adopted: dict[str, Any] | None = None,
    ack: dict[str, Any] | None = None,
    labels: list[dict[str, Any]] | None = None,
    usage: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """틱 하나를 레코드에 넣는다. 네 출력은 끝까지 분리해 둔다 (docs/08 §8).

    `request`는 :meth:`robo_jev.harness.robot.RobotHarness.build_request`가 낸 틱이다.
    원본은 건드리지 않고 깊은 복사로 옮긴다. `usage`는 그 틱의 조합 결과(게이트·조합 기록·실행
    결과) 요약이다 — 비입력 필드(docs/04 §1 표)이며 :func:`aggregate`가 게이트·정지·전환을 센다.
    """
    tick = {
        key: copy.deepcopy(value) for key, value in request.items() if key != _HARNESS_BLOCK
    }
    if model_output is not None:
        tick["model_output"] = copy.deepcopy(model_output)
    if adopted is not None:
        tick["adopted"] = copy.deepcopy(adopted)
    if ack is not None:
        tick["ack"] = copy.deepcopy(ack)
    if labels is not None:
        tick["labels"] = copy.deepcopy(labels)
    if usage is not None:
        tick["usage"] = copy.deepcopy(usage)

    _extend_instructions(record, tick)
    record["ticks"].append(tick)
    return tick


def finalize(
    record: dict[str, Any],
    *,
    versions: dict[str, str] | None = None,
    provenance: dict[str, Any] | None = None,
    evidence: dict[str, Any] | None = None,
    validate: bool = True,
    config_paths: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """버전·provenance·evidence를 붙이고 계약을 확인한다. `config_paths`는 :func:`default_versions`의 경로 인자다."""
    if not record.get("ticks"):
        raise ValueError("틱이 하나도 없는 에피소드는 레코드가 아니다")
    record["versions"] = {**default_versions(**(config_paths or {})), **(versions or {})}
    missing = [name for name in REQUIRED_VERSIONS if not record["versions"].get(name)]
    if missing:
        raise ValueError(f"레코드에 필요한 버전이 없다: {missing}")
    if provenance is not None:
        record["provenance"] = copy.deepcopy(provenance)
    if evidence is not None:
        record["evidence"] = copy.deepcopy(evidence)
    if validate:
        validate_record(record)
    return record


def posed_questions(tick: dict[str, Any]) -> int:
    """그 틱이 실제로 던진 질문 수. 자동 QA와 같은 규칙이다."""
    candidates = (tick.get("request") or {}).get("candidates")
    if not isinstance(candidates, dict):
        return 0
    return sum(
        1
        for question_id, spec in QUESTION_SET_V0.items()
        if spec["criteria"] or question_id in candidates
    )


def aggregate(records: list[dict[str, Any]]) -> dict[str, Any]:
    """에피소드·틱·질문 수와 split별 편수 (docs/08 §8 "각각 집계한다"), 그리고 에피소드별
    게이트·정지·전환·충돌 횟수 (docs/02 §9의 결정 안정성 지표의 재료).

    게이트는 틱의 `usage.gate`(생성기가 적는다)에서, 정지·전환은 `adopted`에서, 충돌은
    `usage.records.conflict`에서 센다. 셋 다 비입력 필드다. `switches`는 하네스가 전환 틱이라고
    적은 것(게이트 틱과 hold 재채택도 들어간다)이고, `main_changes`는 채택된 주 결정이 직전 틱과
    다른 틱 수다 — 결정 안정성 지표(docs/02 §9)에 가까운 쪽은 후자다.
    """
    episodes = ticks = questions = labels = 0
    splits: dict[str, int] = {}
    gates: dict[str, int] = {}
    stops = switches = conflicts = main_changes = 0
    per_episode: list[dict[str, Any]] = []
    for record in records:
        if record.get("schema_version") != SCHEMA_STREAM:
            continue
        episodes += 1
        split = record.get("split")
        if isinstance(split, str):
            splits[split] = splits.get(split, 0) + 1
        summary = {
            "episode_id": record.get("episode_id"),
            "ticks": 0,
            "gates": {},
            "stops": 0,
            "switches": 0,
            "main_changes": 0,
            "conflicts": 0,
        }
        previous_main: str | None = None
        for tick in record.get("ticks") or ():
            ticks += 1
            summary["ticks"] += 1
            questions += posed_questions(tick)
            labels += len(tick.get("labels") or ())
            adopted = tick.get("adopted") or {}
            usage = tick.get("usage") or {}
            gate = usage.get("gate")
            if gate:
                summary["gates"][gate] = summary["gates"].get(gate, 0) + 1
            if adopted.get("stop"):
                summary["stops"] += 1
            if adopted.get("switch"):
                summary["switches"] += 1
            main = adopted.get("main")
            if previous_main is not None and main is not None and main != previous_main:
                summary["main_changes"] += 1
            if main is not None:
                previous_main = main
            summary["conflicts"] += int((usage.get("records") or {}).get("conflict", 0))
        for gate, count in summary["gates"].items():
            gates[gate] = gates.get(gate, 0) + count
        stops += summary["stops"]
        switches += summary["switches"]
        main_changes += summary["main_changes"]
        conflicts += summary["conflicts"]
        summary["gates"] = dict(sorted(summary["gates"].items()))
        per_episode.append(summary)
    return {
        "episodes": episodes,
        "ticks": ticks,
        "questions": questions,
        "labels": labels,
        "splits": dict(sorted(splits.items())),
        "gates": dict(sorted(gates.items())),
        "stops": stops,
        "switches": switches,
        "main_changes": main_changes,
        "conflicts": conflicts,
        "per_episode": per_episode,
    }


# --------------------------------------------------------------------------


def _instruction(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "version": int(item["version"]),
        "t_ms": int(item.get("t_ms", 0)),
        "text": str(item["text"]),
    }


def _extend_instructions(record: dict[str, Any], tick: dict[str, Any]) -> None:
    """지시가 바뀌면 prefix 뒤에 붙인다 (docs/08 §3.1 "리셋 없이 뒤에 추가")."""
    goal = ((tick.get("request") or {}).get("state") or {}).get("goal") or {}
    version = goal.get("version")
    if version is None:
        return
    instructions = record["prefix"]["instructions"]
    if int(version) <= int(instructions[-1]["version"]):
        return
    instructions.append(
        _instruction(
            {
                "version": int(version),
                "t_ms": int(goal.get("t_ms", tick.get("sim_ms", 0))),
                "text": str(goal.get("text", "")),
            }
        )
    )
