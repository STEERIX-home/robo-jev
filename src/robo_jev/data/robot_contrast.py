"""로봇 틱의 대조 sibling — 같은 틱 상태에서 사실 하나를 바꿔 만든 `judgment-v0` 단일 요청 레코드 (docs/04 §3, analysis-nimble §3-2).

라벨 규칙이 코드(전문가)이므로 검사 모델 없이 만든다. 뒤집기 종류(`KINDS`):

* ``forbidden`` — 지시 대상의 금지 접촉 표지 토글(구조화된 목표의 `forbidden_contact`와 물체의 `attributes`, 같은 사실의 두
  표현). `q_main`의 `semantic_admissible`이 바뀐다 — 대상→영역 파지가 부적합해지고 지시가 모순이 된다(`q_instr` 거짓).
* ``zone_boundary`` — 완료 틱에서 놓인 대상을 영역 경계 너머 1cm로 옮긴다. `q_done`이 뒤집힌다.
* ``instruction`` — 지시를 계획의 다음 버전(v2)으로 올린다. 주 결정(`q_main`)이 바뀐다. 후보 목록은 틱의 것이므로(아래) v2의
  대상×영역 조합이 **이미 목록에 있는** 틱이나 로봇이 **다른 물체를 들고 있는** 틱(`release_held_object` 경로)만 뽑는다 —
  그 밖의 틱에서는 하네스가 v2 조합을 예약해 목록에 넣었을 것이라 틱의 목록이 새 목표에 맞지 않는다(리뷰 1 I3).

`q_gripper`를 겨냥하는 뒤집기는 없다: 부가 질문의 라벨은 commitment의 국면 프로파일에 조건화되고(commitment 없는 틱은 마스크)
든 물체 표지를 바꿔도 답이 바뀌지 않는다. 든 물체 표지를 지우는 뒤집기(`holding`)는 실현할 수 없는 상태(공중의 물체)를 만들고
`q_done`만 뒤집어 `zone_boundary`와 겹치므로 두지 않는다(리뷰 1 I2).

쌍의 한쪽인 **기본 틱**도 같은 `judgment-v0` 형태로 낸다(`state_first`, L0) — 스트림 틱이 아니라 별도 레코드다. 후보 목록은
틱의 것을 그대로 쓰되(하네스가 그 틱의 관측에서 만든 것; 바뀐 상태로 다시 열거하지 않는다 — 질문은 "이 상태와 이 후보에서의
답"이다) 순서는 레코드 seed로 섞는다(정답 위치 편향 방지, docs/04 §3). 기본 틱이나 sibling의 `q_main` 이유가 퇴화
(:data:`DEGENERATE_REASONS` — hold∉A 틱)면 쌍을 만들지 않는다. 에피소드마다 종류별 ≤ 1, 합쳐 ≤ `per_episode`이며 같은
origin_group·split·holdout 근거를 승계한다.

**삭제 analogue.** 비로봇의 삭제 검사("초점 사실을 지우면 라벨이 마스크·해당 없음")의 로봇 대응은 **대상 삭제**다: 대상을
관측에서 지우거나(`forbidden`·`zone_boundary`) 지시의 대상을 비우면(`instruction`) 전문가는 판단하지 않고 게이트로 가야 한다 —
관측(`observe_target`)·재계획(`instruction_incomplete`). 뒤집힌 질문 자체가 "모른다"가 되는 것은 아니다(boolean에는 마스크가 없고
대상이 없는 `q_done`은 거짓이다); 검사하는 것은 사실이 없을 때 결정이 게이트로 물러난다는 것이다. 만들 때 검사하고(못 지나면
쌍을 버린다) QA(:func:`deletion_outcome`)가 레코드에 적힌 전문가 버전으로 다시 돌린다.
"""

from __future__ import annotations

import copy
import random
import re
from collections.abc import Sequence
from typing import Any

from robo_jev.contracts import QUESTION_SET_V0, SCHEMA_SINGLE_REQUEST, validate_record
from robo_jev.harness.robot import joint_key_parts, load_harness_config
from robo_jev.sim.expert import DEGENERATE_REASONS, GATE_REASONS, Expert

__all__ = [
    "GENERATOR_VERSION",
    "KINDS",
    "allowed_diff_paths",
    "build_pairs",
    "contrast_summary",
    "default_question_texts",
    "deletion_outcome",
    "flip",
    "flipped_answer",
    "forget",
    "single_request_from_tick",
]

GENERATOR_VERSION = "gen-robot-contrast-v0.3"

#: 뒤집기 종류와 그것이 겨냥하는 질문.
KINDS: dict[str, str] = {
    "forbidden": "q_main",
    "zone_boundary": "q_done",
    "instruction": "q_main",
}

#: 경계 너머로 옮기는 거리(mm) — "±1cm".
BOUNDARY_STEP_MM = 10

#: 삭제 analogue의 게이트 결과 이름.
_GATE_OUTCOMES = {"observe_target": "observe_gate", "instruction_incomplete": "replan_gate"}


# --------------------------------------------------------------------------
# 틱 → 단일 요청 레코드
# --------------------------------------------------------------------------


def single_request_from_tick(
    episode: dict[str, Any],
    tick: dict[str, Any],
    state: dict[str, Any],
    *,
    request_id: str,
    expert: Expert,
    question_texts: dict[str, str],
    shuffle_seed: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """틱의 후보·이력·commitment와 (바뀐) 상태로 `judgment-v0` 레코드와 전문가의 답을 만든다.

    전문가는 모델 입력(상태·실행 이력·commitment·후보)만 본다. 라벨은 그 답에서 나오고(스트림 틱과 같은 규칙 이름), 부가
    질문은 요청의 commitment에 조건화된다(없으면 마스크).
    """
    request = tick["request"]
    model = {
        "state": copy.deepcopy(state),
        "exec_history": copy.deepcopy(request.get("exec_history")),
        "commitment": copy.deepcopy(request.get("commitment")),
        "candidates": copy.deepcopy(request["candidates"]),
    }
    answers = expert.act({"request": model}, None)
    labels = [
        {key: value for key, value in label.items() if key != "conditioned_on"}
        for label in expert.labels(answers, {"request": model})
    ]
    rng = random.Random(shuffle_seed)
    questions = []
    for question_id, spec in QUESTION_SET_V0.items():
        if spec["criteria"]:
            criteria = [dict(criterion) for criterion in spec["criteria"]]
        elif question_id in model["candidates"]:
            entries = model["candidates"][question_id]
            criteria = [
                {"id": str(entry["id"]), "description": _candidate_text(entry)} for entry in entries
            ]
            if question_id in ("q_main", "q_path"):
                # 동적 후보(주 결정·경로)의 순서는 레코드마다 섞는다 (docs/04 §3 "정답 위치의 편향을 없앤다"). 경로 후보를 하네스의
                # 정식 순서(direct·via·retreat·hold)로 두면 정답(거의 언제나 direct)이 첫 자리에 몰려 QA의 정답 위치 검사가
                # D1 규모(표본 ≥ 200)에서 걸린다 — 스트림 틱의 순서는 그대로이고 학습 증강(`permute_candidates`)이 섞는다.
                rng.shuffle(criteria)
        else:
            continue
        questions.append(
            {"id": question_id, "type": spec["type"], "instructions": question_texts[question_id], "criteria": criteria}
        )
    posed = {question["id"] for question in questions}
    labels = [label for label in labels if label["question_id"] in posed]
    record = {
        "schema_version": SCHEMA_SINGLE_REQUEST,
        "origin_group": episode["origin_group"],
        "split": episode["split"],
        "request": {"request_id": request_id, "state": copy.deepcopy(state), "questions": questions},
        "labels": labels,
    }
    return record, answers


def _candidate_text(entry: dict[str, Any]) -> str:
    """후보 줄의 설명: 결합 후보는 의미 키와 기하 네 값(서식 v0.3의 후보 줄과 같은 정보), 경로 후보는 종류(와 경유점 이름)."""
    if "key" in entry:
        parts = [str(entry["key"])]
        parts.extend(f"{name}={entry[name]}" for name in ("d", "clr", "path", "g") if name in entry)
        return " ".join(parts)
    parts = [str(entry.get("kind", ""))]
    if entry.get("ref"):
        parts.append(f"ref={entry['ref']}")
    return " ".join(parts)


# --------------------------------------------------------------------------
# 뒤집기와 삭제
# --------------------------------------------------------------------------


def _target(state: dict[str, Any]) -> dict[str, Any] | None:
    ref = (state.get("goal") or {}).get("target_ref")
    return next((entry for entry in state.get("objects") or () if str(entry["id"]) == str(ref)), None)


def _zone(state: dict[str, Any], zone_id: str | None) -> dict[str, Any] | None:
    return next((zone for zone in state.get("zones") or () if str(zone["id"]) == str(zone_id)), None)


def flip(state: dict[str, Any], kind: str, *, plan: dict[str, Any] | None = None) -> tuple[dict[str, Any], str] | None:
    """`kind`의 뒤집기를 적용한 새 상태와 초점 사실의 경로. 적용할 수 없으면 `None`."""
    out = copy.deepcopy(state)
    target = _target(out)
    if kind == "forbidden":
        if target is None:
            return None
        forbidden = list(out["goal"].get("forbidden_contact") or [])
        attributes = list(target.get("attributes") or [])
        if target["id"] in forbidden:
            forbidden.remove(target["id"])
            attributes = [item for item in attributes if item != "forbidden"]
        else:
            forbidden.append(target["id"])
            attributes.append("forbidden")
        out["goal"]["forbidden_contact"] = forbidden
        target["attributes"] = attributes
        return out, f"goal.forbidden_contact[{target['id']}]"
    if kind == "zone_boundary":
        zone = _zone(out, (out.get("goal") or {}).get("target_zone"))
        if target is None or zone is None or out["robot"].get("holding") == target["id"]:
            return None
        x0, y0, x1, y1 = [float(value) for value in zone["bounds_mm"]]
        x0, x1 = min(x0, x1), max(x0, x1)
        y0, y1 = min(y0, y1), max(y0, y1)
        pose = [float(value) for value in target["pose_mm"]]
        inside = x0 <= pose[0] <= x1 and y0 <= pose[1] <= y1
        if inside:
            # 가장 가까운 경계 너머 1cm로 나간다.
            options = [(pose[0] - x0, 0, x0 - BOUNDARY_STEP_MM), (x1 - pose[0], 0, x1 + BOUNDARY_STEP_MM),
                       (pose[1] - y0, 1, y0 - BOUNDARY_STEP_MM), (y1 - pose[1], 1, y1 + BOUNDARY_STEP_MM)]
        else:
            # 경계 밖 1cm 안에 있을 때만 안으로 들인다(축마다 가까운 경계 안쪽 1cm).
            options = []
            for axis, (lo, hi) in ((0, (x0, x1)), (1, (y0, y1))):
                if pose[axis] < lo and lo - pose[axis] <= BOUNDARY_STEP_MM * 2:
                    options.append((lo - pose[axis], axis, lo + BOUNDARY_STEP_MM))
                if pose[axis] > hi and pose[axis] - hi <= BOUNDARY_STEP_MM * 2:
                    options.append((pose[axis] - hi, axis, hi - BOUNDARY_STEP_MM))
            other = [1 - axis for _, axis, _ in options]
            options = [option for option, o in zip(options, other) if (y0 <= pose[o] <= y1 if o == 1 else x0 <= pose[o] <= x1)]
            if not options:
                return None
        _, axis, value = min(options)
        target["pose_mm"][axis] = int(round(value))
        ee = out["robot"]["ee_pose_mm"]
        for item in out.get("derived") or ():
            if item.get("object") == target["id"] and "relative_mm" in item:
                item["relative_mm"] = [int(round(target["pose_mm"][i] - ee[i])) for i in range(3)]
        return out, f"objects[{target['id']}].pose_mm[{axis}]"
    if kind == "instruction":
        instructions = (plan or {}).get("instructions") or []
        current = int(out["goal"].get("version", 1))
        step = next((item for item in instructions if int(item["version"]) == current + 1), None)
        if step is None:
            return None
        objects = {str(entry["id"]): entry for entry in out.get("objects") or ()}
        new_target = objects.get(str(step.get("target")))
        if new_target is None or step.get("target") == out["goal"].get("target_ref"):
            return None
        out["goal"].update(
            {
                "text": str(step["text"]),
                "version": int(step["version"]),
                "t_ms": int(out["t"]["sim_ms"]) if isinstance(out.get("t"), dict) and "sim_ms" in out["t"] else int(out["goal"].get("t_ms", 0)),
                "target_ref": str(step["target"]),
                "target_desc": str(new_target.get("desc", "")),
                "target_zone": str(step["zone"]),
            }
        )
        return out, "goal"
    raise ValueError(f"모르는 뒤집기 종류다: {kind!r}")


def forget(state: dict[str, Any], kind: str) -> dict[str, Any]:
    """대상 삭제: 대상 물체를 관측에서 지우거나(`forbidden`·`zone_boundary`), 지시의 대상을 비운다(`instruction`)."""
    out = copy.deepcopy(state)
    if kind == "instruction":
        out["goal"].update({"target_ref": None, "target_desc": None})
        return out
    target = _target(out)
    if target is not None:
        out["objects"] = [entry for entry in out["objects"] if entry["id"] != target["id"]]
        out["derived"] = [item for item in out.get("derived") or () if item.get("object") != target["id"]]
        if out["robot"].get("holding") == target["id"]:
            out["robot"]["holding"] = None
    return out


def deletion_outcome(state: dict[str, Any], tick_request: dict[str, Any], kind: str, expert: Expert) -> str | None:
    """초점 사실을 지운 상태에서 전문가가 게이트(관측·재계획)로 가면 그 이름, 아니면 `None`(삭제 검사 실패)."""
    model = {
        "state": forget(state, kind),
        "exec_history": tick_request.get("exec_history"),
        "commitment": tick_request.get("commitment"),
        "candidates": tick_request["candidates"],
    }
    answers = expert.act({"request": model}, None)
    reason = str(answers["expert_meta"]["main"]["reason"])
    if reason in GATE_REASONS:
        return _GATE_OUTCOMES.get(reason, reason)
    return None


def flipped_answer(record: dict[str, Any], question_id: str) -> Any:
    """그 질문의 라벨을 비교할 수 있는 값으로: `q_main`은 (허용 집합, 적합 집합), boolean은 답, 나머지는 허용 집합. 없으면 `None`."""
    label = next((item for item in record["labels"] if item["question_id"] == question_id), None)
    if label is None:
        return None
    if question_id == "q_main":
        return (tuple(sorted(label["candidate_ids"])), tuple(sorted(label.get("semantic_admissible") or ())))
    if "answer" in label:
        return label["answer"]
    return tuple(sorted(label["candidate_ids"]))


def allowed_diff_paths(kind: str, focus_field: str, parent_state: dict[str, Any]) -> frozenset[str]:
    """QA의 한 자리 검사가 로봇 sibling에 허용하는 **정확한** 잎 경로 집합 (리뷰 1 I1): 초점 사실과 그 파생값뿐이며 부모 상태의
    대상 색인으로 계산한다 — `zone_boundary`는 `state.objects[i].pose_mm[axis]`와 `state.derived[j].relative_mm[axis]`,
    `forbidden`은 `state.goal.forbidden_contact`와 `state.objects[i].attributes`(목록 길이가 바뀌므로 목록 경로),
    `instruction`은 `state.goal.{text, version, t_ms, target_ref, target_desc, target_zone}`. 다른 잎이 하나라도 다르면 위반이다."""
    if kind == "instruction":
        return frozenset(f"state.goal.{name}" for name in ("text", "version", "t_ms", "target_ref", "target_desc", "target_zone"))
    match = re.match(r"(?:goal\.forbidden_contact|objects)\[(?P<id>[^\]]+)\](?:\.pose_mm\[(?P<axis>\d)\])?$", focus_field)
    if match is None:
        return frozenset()
    object_id = match.group("id")
    index = next((i for i, entry in enumerate(parent_state.get("objects") or ()) if str(entry.get("id")) == object_id), None)
    if index is None:
        return frozenset()
    if kind == "forbidden":
        return frozenset({"state.goal.forbidden_contact", f"state.objects[{index}].attributes"})
    if kind == "zone_boundary":
        axis = match.group("axis")
        allowed = {f"state.objects[{index}].pose_mm[{axis}]"}
        derived = next((j for j, item in enumerate(parent_state.get("derived") or ()) if item.get("object") == object_id), None)
        if derived is not None:
            allowed.add(f"state.derived[{derived}].relative_mm[{axis}]")
        return frozenset(allowed)
    return frozenset()


# --------------------------------------------------------------------------
# 에피소드 → 쌍
# --------------------------------------------------------------------------


def _next_step(plan: dict[str, Any] | None, version: int) -> dict[str, Any] | None:
    for item in (plan or {}).get("instructions") or ():
        if int(item.get("version", 0)) == version + 1:
            return item
    return None


def _combination_listed(tick: dict[str, Any], target: str, zone: str) -> bool:
    """틱의 후보 목록에 `target → zone`의 파지·놓기가 있는가 (하네스의 지시 조합 예약이 넣었을 후보)."""
    for entry in tick["request"]["candidates"]["q_main"]:
        parts = joint_key_parts(entry.get("key", ""))
        if parts is not None and parts[0] in ("grasp", "place") and parts[1] == target and parts[3] == zone:
            return True
    return False


def _eligible_ticks(episode: dict[str, Any], kind: str, plan: dict[str, Any] | None = None) -> list[int]:
    """종류별로 뒤집기가 뜻이 있는 틱(에피소드 안 순번). 라벨은 읽되 결과는 보지 않는다 — 어느 틱이 어떤 상황인지만.

    `instruction`은 v2의 대상×영역 조합이 틱의 목록에 있거나 로봇이 다른 물체를 들고 있는 틱만이다(리뷰 1 I3)."""
    found: list[int] = []
    for index, tick in enumerate(episode["ticks"]):
        state = tick["request"]["state"]
        target = _target(state)
        if target is None:
            continue
        gate = (tick.get("usage") or {}).get("gate")
        holding = state["robot"].get("holding")
        commitment = tick["request"].get("commitment") or {}
        phase = str(commitment.get("phase", "none"))
        if kind == "zone_boundary":
            if gate == "done":
                found.append(index)
        elif kind == "forbidden":
            if gate is None and holding is None and phase in ("approach", "grasp", "none"):
                found.append(index)
        elif kind == "instruction":
            if gate is not None or int(state["goal"].get("version", 1)) != 1 or phase not in ("approach", "grasp", "lift", "transport"):
                continue
            step = _next_step(plan, int(state["goal"].get("version", 1)))
            if step is None or str(step.get("target")) == str(state["goal"].get("target_ref")):
                continue
            objects = {str(entry["id"]) for entry in state.get("objects") or ()}
            if str(step.get("target")) not in objects:
                continue
            holds_other = holding is not None and holding != str(step.get("target"))
            if holds_other or _combination_listed(tick, str(step.get("target")), str(step.get("zone"))):
                found.append(index)
    return found


def build_pairs(
    episode: dict[str, Any],
    *,
    expert: Expert,
    per_episode: int = 4,
    kinds: Sequence[str] = tuple(KINDS),
    question_texts: dict[str, str] | None = None,
    log: Any = None,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """에피소드 하나의 대조 쌍(기본 레코드 + sibling)들과 빠진 이유의 집계.

    종류마다 뜻이 있는 틱 가운데 하나를 에피소드 id로 seed한 난수로 고르고, 뒤집기 → 겨냥 질문의 라벨이 실제로 다른가 →
    삭제 analogue를 지나는가를 본다. 어느 하나라도 아니면 그 종류는 빠지고 이유를 센다.
    """
    texts = question_texts or {qid: str(spec["instructions"]) for qid, spec in QUESTION_SET_V0.items()}
    plan = ((episode.get("evidence") or {}).get("scene_plan")) or {}
    provenance = episode.get("provenance") or {}
    rng = random.Random(f"robot-contrast:{episode['episode_id']}")
    records: list[dict[str, Any]] = []
    reasons: dict[str, int] = {}
    made = 0
    for kind in kinds:
        if made >= per_episode:
            reasons["rate_limited"] = reasons.get("rate_limited", 0) + 1
            continue
        ticks = _eligible_ticks(episode, kind, plan)
        if not ticks:
            reasons[f"{kind}:no_tick"] = reasons.get(f"{kind}:no_tick", 0) + 1
            continue
        index = rng.choice(ticks)
        tick = episode["ticks"][index]
        state = tick["request"]["state"]
        flipped = flip(state, kind, plan=plan)
        if flipped is None:
            reasons[f"{kind}:not_applicable"] = reasons.get(f"{kind}:not_applicable", 0) + 1
            continue
        new_state, focus_field = flipped
        base_id = f"{episode['episode_id']}@{tick['t']}-{kind}-base"
        sibling_id = f"{episode['episode_id']}@{tick['t']}-{kind}"
        base, base_answers = single_request_from_tick(
            episode, tick, state, request_id=base_id, expert=expert, question_texts=texts, shuffle_seed=f"{sibling_id}:s"
        )
        sibling, sibling_answers = single_request_from_tick(
            episode, tick, new_state, request_id=sibling_id, expert=expert, question_texts=texts, shuffle_seed=f"{sibling_id}:s"
        )
        reasons_main = (str(base_answers["expert_meta"]["main"]["reason"]), str(sibling_answers["expert_meta"]["main"]["reason"]))
        if any(reason in DEGENERATE_REASONS for reason in reasons_main):
            # hold∉A 틱: 실행기 사정으로 물러난 답이라 대조가 아니다 (리뷰 1 I3).
            reasons[f"{kind}:degenerate"] = reasons.get(f"{kind}:degenerate", 0) + 1
            continue
        question_id = KINDS[kind]
        before, after = flipped_answer(base, question_id), flipped_answer(sibling, question_id)
        if before is None or after is None or before == after:
            reasons[f"{kind}:no_flip"] = reasons.get(f"{kind}:no_flip", 0) + 1
            continue
        outcome = deletion_outcome(new_state, tick["request"], kind, expert)
        if outcome is None:
            reasons[f"{kind}:deletion_failed"] = reasons.get(f"{kind}:deletion_failed", 0) + 1
            continue
        common = {
            "generator": GENERATOR_VERSION,
            "domain": "robot",
            "episode_id": episode["episode_id"],
            "t": int(tick["t"]),
            "kind": kind,
            "origin_group": episode["origin_group"],
            "seed": provenance.get("seed"),
            "profile": provenance.get("profile"),
            "language": "ko",
            "phrasing": list(provenance.get("phrasing") or []),
            "concepts": list(provenance.get("concepts") or []),
            "holdout": list(provenance.get("holdout") or []),
            "label_source": expert.label_source,
            "versions": copy.deepcopy(episode.get("versions") or {}),
        }
        base["provenance"] = {
            **common,
            "contrast": {"role": "base", "sibling_id": sibling_id, "focus_field": focus_field, "flipped_question": question_id},
        }
        base["evidence"] = {"tick_index": index}
        sibling["provenance"] = {
            **common,
            "derived_from": base_id,
            "derivation": "contrast",
            "contrast": {
                "role": "sibling",
                "sibling_id": base_id,
                "focus_field": focus_field,
                "flipped_question": question_id,
                "flipped_questions": [
                    qid for qid in QUESTION_SET_V0 if flipped_answer(base, qid) != flipped_answer(sibling, qid)
                ],
                "deletion": {"field": focus_field, "outcome": outcome, "expert_version": str(expert.version)},
            },
        }
        sibling["evidence"] = {
            "tick_index": index,
            "contrast": {
                "kind": kind,
                "main_reasons": {"base": reasons_main[0], "sibling": reasons_main[1]},
                "base_answer": _jsonable(before),
                "sibling_answer": _jsonable(after),
                "tick_request": {
                    "exec_history": copy.deepcopy(tick["request"].get("exec_history")),
                    "commitment": copy.deepcopy(tick["request"].get("commitment")),
                    "candidates": copy.deepcopy(tick["request"]["candidates"]),
                },
            },
        }
        validate_record(base)
        validate_record(sibling)
        records.extend([base, sibling])
        made += 1
        reasons[f"{kind}:paired"] = reasons.get(f"{kind}:paired", 0) + 1
        if log is not None:
            print(f"  contrast {sibling_id}: {question_id} {before!r} → {after!r}, deletion={outcome}", file=log, flush=True)
    return records, reasons


def _jsonable(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    return value


def contrast_summary(records: Sequence[dict[str, Any]], reasons: dict[str, int]) -> dict[str, Any]:
    """manifest에 적는 집계: 쌍 수, 종류별·split별 수, 삭제 결과, 빠진 이유."""
    pairs = [record for record in records if (record.get("provenance") or {}).get("derivation") == "contrast"]
    by_kind: dict[str, int] = {}
    by_split: dict[str, int] = {}
    deletion: dict[str, int] = {}
    for record in pairs:
        kind = str(record["provenance"].get("kind"))
        by_kind[kind] = by_kind.get(kind, 0) + 1
        by_split[str(record.get("split"))] = by_split.get(str(record.get("split")), 0) + 1
        outcome = str(record["provenance"]["contrast"]["deletion"]["outcome"])
        deletion[outcome] = deletion.get(outcome, 0) + 1
    return {
        "generator": GENERATOR_VERSION,
        "pairs": len(pairs),
        "records": len(records),
        "by_kind": dict(sorted(by_kind.items())),
        "by_split": dict(sorted(by_split.items())),
        "deletion_outcomes": dict(sorted(deletion.items())),
        "reasons": dict(sorted(reasons.items())),
    }


def default_question_texts(harness_config: str | None = None) -> dict[str, str]:
    """하네스 설정의 질문 문구 (계약의 ko 세트와 같다)."""
    config = load_harness_config(harness_config) if harness_config else load_harness_config()
    language = str(config.get("language", "ko"))
    return {question_id: str(texts[language]) for question_id, texts in config["questions"].items()}
