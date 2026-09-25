"""`q_done`의 틱 층과 거짓 done의 사실 (Task R6 A0·A1; docs/08 §7 `q_done` 라벨 `goal-zone-containment-v0`).

왜. R5의 폐루프에서 가장 큰 결함이 **거짓 done**이었다 — 하네스의 done 게이트는 정책 자신의 `q_done`이고
(`harness/robot.py` `_gate`), 모델은 참조 라벨이 거짓인 틱에서 그 답을 든다. R5 기록의 거짓 done 25편은 전부 지시가
바뀐 편(E1/E2)이고 모두 **현재 지시의 대상을 한 번도 잡지 않았다** — 모델이 이전 지시의 대상을 놓은 직후에 `q_done`이
오른다. 곧 모델은 `q_done`을 현재 지시의 대상·영역이 아니라 "놓기가 막 끝났다"는 실행 패턴에서 읽는다. 이 모듈은 그
물음을 틱 단위로 가를 **층**을 정의한다.

층 (앞 층이 이긴다, :data:`DONE_STRATA`):

* ``done_true`` — 참조 라벨이 참(대상이 영역 안에 놓여 있고 손에 없다).
* ``post_release_other`` — **지금 대상이 아닌** 물체를 놓은 뒤 ≤ :data:`RELEASE_WINDOW_TICKS` 틱(놓은 틱 = 0), 참조 거짓.
* ``old_goal_satisfied`` — 이 편의 **어느 이전 지시**(지금보다 낮은 목표 버전)의 대상이 그 지시의 영역 안에 있고 손에 없다,
  지금 목표는 미성립(참조 거짓).
* ``post_release_current_outside`` — **지금 대상**을 놓은 뒤 ≤ 창, 그 대상이 손에 없고 영역 밖이다(참조 거짓).
* ``other_false`` — 나머지 참조 거짓 틱.

"놓기"는 관측의 `robot.holding`이 물체 X에서 X가 아닌 것으로 바뀐 틱이다(:func:`release_events`) — 떨어뜨린 것도 놓기다.
"지금 대상"은 그 틱의 `state.goal.target_ref`(구조화된 목표 — 레코드에는 있고 모델 입력에는 없다, 서식 v0.4)다.
층은 **그 틱과 과거 틱만** 본다(놓기·목표 버전의 이력) — 미래 틱을 읽지 않으므로 롤아웃 중에도 같은 값이 나온다.
영역 안 판정은 전문가의 규칙(:func:`robo_jev.sim.expert._inside`) 그대로다.

창 6틱과 "놓은 틱 = 0"은 브리프가 준 세 수(g2 train 403 / 107 / 853, R5 dev_new 24/67, 본 장면 54/140)를 다시 낸
정의다(:func:`release_window_counts`, 보고서 A0) — 층의 정의를 수를 본 뒤에 고른 것이 아니라, 컨트롤러가 수를 낸 정의를
재현해 그대로 층으로 쓴 것이다.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

from robo_jev.sim.expert import _inside, _zone_distance_mm

__all__ = [
    "DONE_GATE_THRESHOLD",
    "DONE_QUESTION",
    "DONE_STRATA",
    "RELEASE_WINDOW_TICKS",
    "SEALED_SPLITS",
    "count_done_strata",
    "done_strata_by_split",
    "false_done_events",
    "false_done_facts",
    "model_q_done",
    "model_rates_by_stratum",
    "old_goal_variants",
    "recovery_after_false_done",
    "reference_done",
    "release_events",
    "release_window_counts",
    "tick_done_strata",
]

DONE_QUESTION = "q_done"
DONE_STRATA = ("done_true", "post_release_other", "old_goal_satisfied", "post_release_current_outside", "other_false")
#: 놓은 틱(0)부터 몇 틱까지를 "놓은 뒤"로 보는가 — 브리프의 수를 다시 낸 값 (모듈 설명).
RELEASE_WINDOW_TICKS = 6
#: 하네스가 `q_done`을 done 게이트로 읽는 임계값 (`configs/harness/robot.yaml` `compose.gates.done`; `p ≥ 0.5`면 게이트).
DONE_GATE_THRESHOLD = 0.5
#: 봉인 분할 — 층의 수도 적지 않는다 (docs/04 §5; 편 수만).
SEALED_SPLITS = ("ood_test",)
#: done 게이트가 편을 끝내는 연속 틱 수 = 완료 꼬리 3틱(`configs/data/r1_robot.yaml` `episode.tail_ticks_after_done`) + 1 —
#: 생성기는 done 게이트가 이만큼 이어져야 `done_tail`로 끝낸다(`robot_episodes.generate_episode`).
TERMINATING_DONE_RUN = 4


# --------------------------------------------------------------------------
# 틱에서 읽는 것
# --------------------------------------------------------------------------


def _state(tick: dict[str, Any]) -> dict[str, Any]:
    return (tick.get("request") or {}).get("state") or {}


def _goal(tick: dict[str, Any]) -> dict[str, Any]:
    return _state(tick).get("goal") or {}


def _version(tick: dict[str, Any]) -> int:
    value = _goal(tick).get("version")
    return int(value) if value is not None else 1


def _holding(tick: dict[str, Any]) -> str | None:
    value = (_state(tick).get("robot") or {}).get("holding")
    return None if value is None else str(value)


def reference_done(tick: dict[str, Any]) -> bool | None:
    """그 틱의 `q_done` 참조 라벨(전문가의 목표 평가기); 라벨이 없으면 None."""
    label = next((item for item in tick.get("labels") or () if item.get("question_id") == DONE_QUESTION), None)
    if label is None:
        return None
    answer = label.get("answer")
    if isinstance(answer, str):
        return answer == "true"
    return bool(answer)


def model_q_done(tick: dict[str, Any]) -> float | None:
    """모델의 **raw** `q_done` 확률. 수집 정책(:class:`robo_jev.data.done_gate.DoneGatePolicy`)이 하네스에 expert의 답을 넘긴
    기록에서는 `model_output.q_done`이 expert의 값이므로 `usage.model_q_done`(모델의 raw 답)을 먼저 읽는다."""
    usage = tick.get("usage") or {}
    if usage.get("model_q_done") is not None:
        return float(usage["model_q_done"])
    value = (tick.get("model_output") or {}).get(DONE_QUESTION)
    if isinstance(value, dict):
        value = value.get("true", None if "false" not in value else 1.0 - float(value["false"]))
    return None if value is None else float(value)


def _inside_zone(state: dict[str, Any], object_id: str | None, zone_id: str | None) -> bool:
    """물체가 그 영역 안에 놓여 있는가 — 전문가 `_goal_satisfied`와 같은 규칙(손에 있으면 거짓)."""
    if object_id is None or zone_id is None:
        return False
    if (state.get("robot") or {}).get("holding") == object_id:
        return False
    entry = next((item for item in state.get("objects") or () if str(item.get("id")) == str(object_id)), None)
    zone = next((item for item in state.get("zones") or () if str(item.get("id")) == str(zone_id)), None)
    if entry is None or zone is None:
        return False
    return _inside(entry["pose_mm"], zone["bounds_mm"])


def release_events(ticks: list[dict[str, Any]]) -> list[tuple[int, str]]:
    """놓기: 관측의 `robot.holding`이 물체 X에서 X가 아닌 것(없음·다른 물체)으로 바뀐 틱과 X — ``[(틱 색인, X), …]``."""
    out: list[tuple[int, str]] = []
    previous: str | None = None
    for index, tick in enumerate(ticks):
        holding = _holding(tick)
        if index > 0 and previous is not None and holding != previous:
            out.append((index, previous))
        previous = holding
    return out


# --------------------------------------------------------------------------
# 층
# --------------------------------------------------------------------------


def tick_done_strata(record: dict[str, Any], *, window: int = RELEASE_WINDOW_TICKS) -> list[str | None]:
    """틱마다 층 이름(:data:`DONE_STRATA`), `q_done` 라벨이 없는 틱은 None. 그 틱과 과거 틱만 읽는다(모듈 설명)."""
    ticks = record["ticks"]
    releases = dict(release_events(ticks))
    last_release: tuple[int, str] | None = None
    goals: dict[int, tuple[str | None, str | None]] = {}
    out: list[str | None] = []
    for index, tick in enumerate(ticks):
        if index in releases:
            last_release = (index, releases[index])
        goal = _goal(tick)
        version = _version(tick)
        goals.setdefault(version, (goal.get("target_ref"), goal.get("target_zone")))
        reference = reference_done(tick)
        if reference is None:
            out.append(None)
            continue
        if reference:
            out.append("done_true")
            continue
        state = _state(tick)
        target = goal.get("target_ref")
        released = last_release[1] if last_release is not None and index - last_release[0] <= int(window) else None
        if released is not None and released != target:
            out.append("post_release_other")
            continue
        if any(_inside_zone(state, old_target, old_zone) for old_version, (old_target, old_zone) in goals.items() if old_version < version):
            out.append("old_goal_satisfied")
            continue
        if released is not None and released == target and _holding(tick) != target and not _inside_zone(state, target, goal.get("target_zone")):
            out.append("post_release_current_outside")
            continue
        out.append("other_false")
    return out


def count_done_strata(records: list[dict[str, Any]], *, window: int = RELEASE_WINDOW_TICKS) -> dict[str, Any]:
    """층별 틱 수·그 층을 가진 편 수·라벨 틱 수·편 수."""
    strata: Counter = Counter()
    episodes_with: Counter = Counter()
    labelled = 0
    for record in records:
        found = [name for name in tick_done_strata(record, window=window) if name is not None]
        labelled += len(found)
        strata.update(found)
        episodes_with.update(set(found))
    return {
        "strata": {name: int(strata.get(name, 0)) for name in DONE_STRATA},
        "episodes_with": {name: int(episodes_with.get(name, 0)) for name in DONE_STRATA},
        "labelled_ticks": labelled, "episodes": len(records), "window_ticks": int(window),
    }


def done_strata_by_split(records: list[dict[str, Any]], *, window: int = RELEASE_WINDOW_TICKS) -> dict[str, Any]:
    """전체 + 분할별 층 수. 봉인 분할은 `by_split`에서 빼고 편 수만 `sealed`에 적는다(전체도 봉인 분할을 뺀 수다)."""
    open_records = [record for record in records if str(record.get("split")) not in SEALED_SPLITS]
    out = count_done_strata(open_records, window=window)
    out["by_split"] = {
        split: count_done_strata([record for record in open_records if str(record.get("split")) == split], window=window)
        for split in sorted({str(record.get("split")) for record in open_records})
    }
    out["sealed"] = {
        split: {"episodes": sum(1 for record in records if str(record.get("split")) == split)}
        for split in SEALED_SPLITS if any(str(record.get("split")) == split for record in records)
    }
    return out


def model_rates_by_stratum(
    records: list[dict[str, Any]], *, window: int = RELEASE_WINDOW_TICKS, threshold: float = DONE_GATE_THRESHOLD,
) -> dict[str, dict[str, Any]]:
    """층마다 모델의 raw `q_done`이 게이트 임계값 이상인 틱 수(`model_true`)와 비율 — 참조 거짓 층에서는 거짓 양성률이다."""
    out = {name: {"n": 0, "answered": 0, "model_true": 0, "rate": None, "episodes": 0} for name in DONE_STRATA}
    seen: dict[str, set[str]] = {name: set() for name in DONE_STRATA}
    for record in records:
        for tick, name in zip(record["ticks"], tick_done_strata(record, window=window)):
            if name is None:
                continue
            block = out[name]
            block["n"] += 1
            seen[name].add(str(record.get("episode_id")))
            value = model_q_done(tick)
            if value is None:
                continue
            block["answered"] += 1
            block["model_true"] += int(value >= float(threshold))
    for name, block in out.items():
        block["rate"] = (block["model_true"] / block["answered"]) if block["answered"] else None
        block["episodes"] = len(seen[name])
    return out


def release_window_counts(records: list[dict[str, Any]], *, window: int = RELEASE_WINDOW_TICKS) -> dict[str, int]:
    """놓은 뒤 ≤ 창의 틱을 (놓은 물체가 지금 대상인가) × (참조가 참인가)로 — 층의 우선순위 없이, 브리프의 세 수(403 / 107 / 853)의 정의."""
    counts = {"other_not_done": 0, "other_done": 0, "current_not_done": 0, "current_done": 0}
    for record in records:
        ticks = record["ticks"]
        releases = dict(release_events(ticks))
        last: tuple[int, str] | None = None
        for index, tick in enumerate(ticks):
            if index in releases:
                last = (index, releases[index])
            if last is None or index - last[0] > int(window):
                continue
            reference = reference_done(tick)
            if reference is None:
                continue
            kind = "current" if last[1] == _goal(tick).get("target_ref") else "other"
            counts[f"{kind}_{'done' if reference else 'not_done'}"] += 1
    return {**counts, "window_ticks": int(window)}


# --------------------------------------------------------------------------
# A0 — 거짓 done 편의 사실
# --------------------------------------------------------------------------


def old_goal_variants(record: dict[str, Any], index: int = -1) -> dict[str, Any]:
    """틱 `index`(기본: 마지막 틱)에서 "이전 지시의 목표가 성립해 있는가"를 세 정의로 — 브리프의 20이 어느 정의인지 가르려고.

    * ``immediately_previous`` — 바로 앞 목표 버전의 대상이 그 영역 안·손에 없음.
    * ``any_earlier`` — 지금보다 낮은 **어느** 목표 버전이든 그 대상이 그 영역 안·손에 없음(층 `old_goal_satisfied`의 정의).
    * ``released_objects_instruction`` — 마지막으로 놓은 물체가 어느 이전 지시의 대상이고 그 지시의 영역 안·손에 없음.
    """
    ticks = record["ticks"][: (index % len(record["ticks"])) + 1]
    tick = ticks[-1]
    state = _state(tick)
    version = _version(tick)
    goals: dict[int, tuple[str | None, str | None]] = {}
    for item in ticks:
        goal = _goal(item)
        goals.setdefault(_version(item), (goal.get("target_ref"), goal.get("target_zone")))
    earlier = sorted(value for value in goals if value < version)
    releases = release_events(ticks)
    last_released = releases[-1][1] if releases else None
    return {
        "immediately_previous": bool(earlier) and _inside_zone(state, *goals[earlier[-1]]),
        "any_earlier": any(_inside_zone(state, *goals[value]) for value in earlier),
        "released_objects_instruction": last_released is not None and any(
            goals[value][0] == last_released and _inside_zone(state, *goals[value]) for value in earlier
        ),
        "last_released": last_released,
    }


def false_done_facts(records: list[dict[str, Any]], *, threshold: float = DONE_GATE_THRESHOLD) -> dict[str, Any]:
    """거짓 done(`done ∧ ¬target_inside_zone`) 편마다의 사실과 합계 — 브리프 A0의 기제를 기록에서 다시 계산한다."""
    rows: list[dict[str, Any]] = []
    for record in records:
        outcome = (record.get("provenance") or {}).get("outcome") or {}
        if not (outcome.get("done") and not outcome.get("target_inside_zone")):
            continue
        ticks = record["ticks"]
        last = ticks[-1]
        goal = _goal(last)
        target, zone = goal.get("target_ref"), goal.get("target_zone")
        version = _version(last)
        start = next(index for index, tick in enumerate(ticks) if _version(tick) == version)
        holding = [_holding(tick) for tick in ticks]
        state = _state(last)
        entry = next((item for item in state.get("objects") or () if str(item.get("id")) == str(target)), None)
        bounds = next((item["bounds_mm"] for item in state.get("zones") or () if str(item.get("id")) == str(zone)), None)
        done_ticks = [index for index, tick in enumerate(ticks) if (tick.get("usage") or {}).get("gate") == "done"]
        first = outcome.get("first_done_tick")
        first = int(first) if first is not None else (done_ticks[0] if done_ticks else None)
        rows.append({
            "episode_id": record.get("episode_id"), "profile": (record.get("provenance") or {}).get("profile"),
            "goal_version": version, "target": target, "zone": zone,
            "target_outside_mm": None if entry is None or bounds is None else _zone_distance_mm(entry["pose_mm"], bounds),
            "held_current_target_ever": any(value == target for value in holding),
            "held_current_target_during_instruction": any(value == target for value in holding[start:]),
            "reference_false_on_every_done_tick": all(reference_done(ticks[index]) is False for index in done_ticks),
            "done_ticks": len(done_ticks), "first_done_tick": first,
            "q_done_before_first_done": model_q_done(ticks[first - 1]) if first else None,
            "q_done_first_done": model_q_done(ticks[first]) if first is not None else None,
            "q_done_done_ticks": [model_q_done(ticks[index]) for index in done_ticks],
            "old_goal": old_goal_variants(record),
            "ticks_from_last_release_to_first_done": (first - release_events(ticks[: first + 1])[-1][0]) if first is not None and release_events(ticks[: first + 1]) else None,
        })  # fmt: skip
    distances = [row["target_outside_mm"] for row in rows if row["target_outside_mm"] is not None]
    firsts = [row["q_done_first_done"] for row in rows if row["q_done_first_done"] is not None]
    befores = [row["q_done_before_first_done"] for row in rows if row["q_done_before_first_done"] is not None]
    return {
        "episodes": len(records), "false_done": len(rows), "threshold": float(threshold), "rows": rows,
        "summary": {
            "profiles": dict(sorted(Counter(str(row["profile"]) for row in rows).items())),
            "never_held_current_target": sum(1 for row in rows if not row["held_current_target_during_instruction"]),
            "never_held_current_target_in_the_episode": sum(1 for row in rows if not row["held_current_target_ever"]),
            "old_goal": {name: sum(1 for row in rows if row["old_goal"][name]) for name in ("immediately_previous", "any_earlier", "released_objects_instruction")},
            "reference_false_on_every_done_tick": sum(1 for row in rows if row["reference_false_on_every_done_tick"]),
            "target_outside_mm": {"min": min(distances), "max": max(distances)} if distances else None,
            "q_done_first_done": {"min": min(firsts), "max": max(firsts)} if firsts else None,
            "q_done_before_first_done": {"min": min(befores), "max": max(befores)} if befores else None,
            "q_done_done_ticks_min": min((value for row in rows for value in row["q_done_done_ticks"] if value is not None), default=None),
        },
    }


# --------------------------------------------------------------------------
# A2·A3 — 모델의 raw `q_done`이 거짓으로 든 사건과 그 뒤
# --------------------------------------------------------------------------


def _false_done_ticks(record: dict[str, Any], threshold: float) -> list[bool]:
    """틱마다: 모델의 raw `q_done` ≥ 임계값이고 참조가 거짓인가."""
    out: list[bool] = []
    for tick in record["ticks"]:
        value = model_q_done(tick)
        out.append(value is not None and value >= float(threshold) and reference_done(tick) is False)
    return out


def false_done_events(
    records: list[dict[str, Any]], *, threshold: float = DONE_GATE_THRESHOLD, terminating_run: int = TERMINATING_DONE_RUN,
) -> dict[str, Any]:
    """모델의 raw `q_done`이 참조 거짓 틱에서 게이트 임계값을 넘은 틱·편·사건(연속 구간). `terminating_run`틱 이상 이어진 사건은
    모델 자신의 done 게이트였다면 편을 끝냈을 사건이다(수집 정책의 기록에서는 하네스가 expert의 답을 받았으므로 편이 이어졌다)."""
    ticks = events = long_runs = 0
    per_episode: dict[str, Any] = {}
    for record in records:
        flags = _false_done_ticks(record, threshold)
        runs: list[int] = []
        length = 0
        for flag in flags + [False]:
            if flag:
                length += 1
            elif length:
                runs.append(length)
                length = 0
        if not runs:
            continue
        ticks += sum(runs)
        events += len(runs)
        long_runs += sum(1 for run in runs if run >= int(terminating_run))
        per_episode[str(record.get("episode_id"))] = {"ticks": sum(runs), "events": len(runs), "longest": max(runs), "first_tick": flags.index(True)}
    return {"ticks": ticks, "episodes": len(per_episode), "events": events, "runs_reaching_the_tail": long_runs,
            "tail_ticks": int(terminating_run), "threshold": float(threshold), "episodes_total": len(records), "per_episode": per_episode}


def recovery_after_false_done(records: list[dict[str, Any]], *, threshold: float = DONE_GATE_THRESHOLD) -> dict[str, Any]:
    """첫 거짓 done 틱(모델의 raw 답) 뒤에 모델이 **그 틱의 지금 대상**을 잡았는가(관측의 `holding`), 그리고 편이 실제로 완료했는가
    (`done ∧ target_inside_zone`) — A3의 물음 "거짓 done 뒤 새 대상으로 가는가·멈추는가"."""
    per_episode: dict[str, Any] = {}
    for record in records:
        flags = _false_done_ticks(record, threshold)
        if True not in flags:
            continue
        first = flags.index(True)
        ticks = record["ticks"]
        grasped = next((index for index in range(first + 1, len(ticks)) if _holding(ticks[index]) is not None
                        and _holding(ticks[index]) == _goal(ticks[index]).get("target_ref")), None)
        outcome = (record.get("provenance") or {}).get("outcome") or {}
        per_episode[str(record.get("episode_id"))] = {
            "first_false_done_tick": first, "grasped_current_target_at": grasped,
            "completed": bool(outcome.get("done")) and bool(outcome.get("target_inside_zone")),
        }
    return {
        "episodes_with_false_done": len(per_episode),
        "grasped_current_target_after": sum(1 for row in per_episode.values() if row["grasped_current_target_at"] is not None),
        "completed_after": sum(1 for row in per_episode.values() if row["completed"]),
        "threshold": float(threshold), "per_episode": per_episode,
    }
