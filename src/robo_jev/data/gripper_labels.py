"""그리퍼 라벨 규칙 v2의 도구와 그 파생 데이터셋 (Task R5 Stage A; docs/08 §7 `q_gripper`, docs/04 §7 DAgger).

    uv run python -m robo_jev.data.gripper_labels derive --parent artifacts/datasets/r1-robot/r1-rollout-labels \\
        --out artifacts/datasets/r1-robot/r1-rollout-labels-g2 --early-ticks K
    uv run python -m robo_jev.data.gripper_labels dagger --source artifacts/datasets/r4-closed-loop/s18/dev \\
        --source artifacts/datasets/r4-closed-loop/466/dev --out artifacts/datasets/r5-dagger/dagger-0 --early-ticks K

왜. R4가 보인 것: 루프 안의 모델은 지시를 읽는데 파지점에서 `closed`를 한 번도 내지 못한다(참조 전환 125건 중 실행 0). 까닭은
라벨 기제다 — 옛 규칙 `_tolerate_gripper_transitions`(±1틱 두 값)는 전환 틱 t*(전문가가 "닫아라"를 처음 낸 틱) 자체를 두 값으로
만들어 R1 v0.2 train의 "지금 닫아라" 316틱 중 307을 지웠다(R4 리뷰 1). 규칙 v2(:func:`robo_jev.data.robot_episodes.
_tolerate_gripper_transitions_v2`)는 전환 틱을 한 값으로 남기고 허용은 전환 **앞** k틱만 넓힌다(이른 답의 허용 — 실행기는 readiness
미충족이면 `gripper_wait`로 보류하므로 이른 `closed`가 안전한지는 A2 실험이 잰다).

무엇을 만드는가. (1) `derive` — 부모 데이터셋(R1 v0.2 rollout 라벨판)에서 **`q_gripper` 라벨만** `evidence.expert.ticks[*].aux.gripper.
desired`(허용 적용 전의 원하는 상태)로 되돌린 뒤 v2를 적용한 파생 데이터셋(:func:`robo_jev.data.lineage.attach_rollout_labels`와 같은
꼴: 레코드 복사·`versions.labels`·`provenance.lineage`·`validate_record`·`build_manifest`·contrast 복사·분할 그대로). (2) `dagger` — R4의
**dev 조건** 모델 주행 기록(`labels` = expert 참조, `provenance.policy` = ModelPolicy)을 같은 규칙으로 재라벨한 첫 DAgger 재료. 에피소드는
다시 만들지 않는다. `data.dagger.relabel`이 하는 기록 방식(`source: expert_v0`·`relabel: true`·cycle)을 따르고, **ood_dev·ood_test 계열
기록은 거절한다**(경로·조건·split 셋 다 검사). 분할 이름은 `train`이다 — 계약의 `SPLITS`에 `dagger`가 없고 `contracts.py`는 배포 계약
digest의 조각이라(`model/contract_digest.py`) 이름을 더하면 R3a 체크포인트를 같은 자로 잴 수 없다; 장면의 원래 조건·split은
`provenance.dagger.scene_condition`·`scene_split`에 남는다. docs/04 §7의 혼합 축(기존 70 / 새 오류 계열 20 / 새 의미 계열 10)에서
이 기록은 **새 오류 계열**이므로 `provenance.material: error_family`를 단다.

틱 종류(R4 C0 표, :data:`GRIPPER_LABEL_CLASSES`): `initiate`(한 값 `closed`·실행 그리퍼 아직 open — "지금 닫아라"), `window`(두 값·실행
open), `settled`(한 값 `closed`·실행 closed — 실행 상태를 베끼면 맞는 틱), `open`(한 값 `open`), `window_closed`(두 값·실행 closed).

규칙 v2의 범위(리뷰 1 M1·M2). 넓히는 것은 원하는 상태가 **open→closed로 바뀌는 모든 전환**의 앞 k틱이지 파지만이 아니다 — g2의
봉인되지 않은 분할에서 넓힌 1,003틱 가운데 956은 `at_grasp_point`(파지) 앞, 17은 `push_with_closed_fingers`(밀기의 접근) 앞, 28은
`place_blocked`(막힌 놓기의 재닫기) 앞, 2는 `phase:lift` 재닫기 앞이다; `window_closed`로 남는 24틱은 뒤의 두 종류(실행 그리퍼가 아직
closed인 채 전문가가 열기를 원하던 틱)다. 거슬러 가는 걸음은 다른 전환·라벨 없는 틱·다른 상태에서 멈추지만 **commitment 변경에서는
멈추지 않는다** — 넓힌 틱 19개(밀기 17·lift 2)는 자기 전환 틱과 `conditioned_on`이 다르다(1.8 %). 둘 다 학습·루프에 해가 없고(이른
닫기는 실행기가 보류한다) 데이터셋은 이 규칙으로 만들어졌으므로 규칙은 그대로 두고 여기 적는다; 파지 앞으로 좁히는 것은 다음 라운드의 선택지다.

manifest의 `generator` 필드(리뷰 1 M8). `build_manifest`가 쓰는 최상위 `generator`는 **manifest를 쓴 체크아웃의 생성기 버전**(이 라운드
`gen-robot-v0.3`, 기본 라벨 규칙 v2)이고, 레코드마다의 `versions.generator`는 **그 에피소드를 만든 생성기**(g2·dagger-0 모두 `gen-robot-v0.2`)다
— 파생 데이터셋은 에피소드를 다시 만들지 않으므로 둘이 다르다. 학습 run의 identity 블록이 manifest 파일의 sha256을 들므로 이미 학습에 쓴
manifest는 고쳐 쓰지 않는다(`docs/reports/run-report-2.md` §2). 봉인 분할(`ood_test`)의 라벨 종류 수는 `by_split`에 적지 않는다(리뷰 1 M9;
편 수만 `sealed`에 든다).
"""

from __future__ import annotations

import argparse
import copy
import json
import shutil
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

from robo_jev.contracts import validate_record
from robo_jev.data.split import OOD_SPLITS

__all__ = [
    "DEFAULT_EARLY_DIRECTIONS",
    "GRIPPER_LABELS_VERSION",
    "GRIPPER_LABEL_CLASSES",
    "EarlyGripperPolicy",
    "build_dagger_dataset",
    "count_gripper_classes",
    "derive_gripper_v2_dataset",
    "desired_from_evidence",
    "early_tolerance_indices",
    "gripper_tick_class",
    "gripper_transitions",
    "main",
    "reference_gripper_schedule",
    "reset_gripper_labels",
]

GRIPPER_LABELS_VERSION = "gripper-v2"
GRIPPER_LABEL_CLASSES = ("initiate", "window", "settled", "open", "window_closed")
_QUESTION = "q_gripper"
_STATES = ("open", "closed")
#: 옛·새 규칙이 라벨의 `rule` 문자열에 붙이는 꼬리 — 되돌릴 때 뗀다.
_RULE_SUFFIXES = ("+transition-tolerance", "+early-tolerance")


def _gripper_label(tick: dict[str, Any]) -> dict[str, Any] | None:
    return next((item for item in tick.get("labels") or () if item.get("question_id") == _QUESTION), None)


# --------------------------------------------------------------------------
# 규칙 v2 — 순수 함수
# --------------------------------------------------------------------------


def gripper_transitions(desired: list[str | None]) -> list[int]:
    """전환 틱: 원하는 상태가 **직전 라벨 틱**(라벨 없는 틱은 건너뛴다)과 달라지는 첫 틱의 색인들 (open→closed·closed→open 둘 다)."""
    out: list[int] = []
    previous: str | None = None
    for index, value in enumerate(desired):
        if value is None:
            continue
        if previous is not None and value != previous:
            out.append(index)
        previous = value
    return out


#: 규칙 v2가 앞 틱을 넓히는 전환의 **방향**(전환 뒤의 상태). 기본은 `closed`(open→closed — 파지가 대부분이고 밀기 접근·`place_blocked`·lift 재닫기도
#: 든다)뿐이다 — A2 실측: 이른 `closed`는
#: 실행기가 readiness(`close_readiness_distance_mm`)로 보류해 전문가와 같은 틱에 닫히지만, 이른 `open`은 운반 국면에서
#: readiness가 하중(`open_readiness_force_n`)만 보므로 **그 자리에서 실행돼 물체를 운반 높이에서 떨어뜨린다**(docs/10 I2의 사고).
DEFAULT_EARLY_DIRECTIONS = ("closed",)


def early_tolerance_indices(desired: list[str | None], early_ticks: int, *, directions: tuple[str, ...] = DEFAULT_EARLY_DIRECTIONS) -> set[int]:
    """규칙 v2가 두 값으로 넓히는 틱: `directions`로 들어가는 각 전환 t*의 **앞** k틱(t*−1 … t*−k)을 색인으로 거슬러 가되 다른
    전환 틱·라벨 없는 틱·다른 상태를 만나면 멈춘다. 전환 틱 자체와 그 뒤는 절대 들어가지 않는다."""
    if int(early_ticks) < 0:
        raise ValueError(f"early_ticks: 0 이상이어야 한다 (받은 값: {early_ticks})")
    unknown = [name for name in directions if name not in _STATES]
    if unknown:
        raise ValueError(f"directions: {list(_STATES)} 중에서 골라야 한다 (받은 값: {unknown})")
    transitions = set(gripper_transitions(desired))
    out: set[int] = set()
    for at in transitions:
        if desired[at] not in directions:
            continue
        state = desired[at - 1] if at > 0 else None
        for back in range(1, int(early_ticks) + 1):
            index = at - back
            if index < 0 or index in transitions or desired[index] is None or desired[index] != state:
                break
            out.add(index)
    return out


# --------------------------------------------------------------------------
# 증거에서 되돌리기
# --------------------------------------------------------------------------


def desired_from_evidence(record: dict[str, Any]) -> list[str | None]:
    """틱마다 전문가가 **허용 적용 전에** 원한 그리퍼 상태(`evidence.expert.ticks[i].aux.gripper.desired`); `q_gripper` 라벨이 없는 틱은 None."""
    ticks = record["ticks"]
    evidence = ((record.get("evidence") or {}).get("expert") or {}).get("ticks")
    if not isinstance(evidence, list) or len(evidence) != len(ticks):
        raise ValueError(f"{record.get('episode_id')}: evidence.expert.ticks가 없거나 틱 수와 다르다 ({None if evidence is None else len(evidence)} ≠ {len(ticks)})")
    out: list[str | None] = []
    for index, (tick, meta) in enumerate(zip(ticks, evidence)):
        if int(meta.get("t", tick["t"])) != int(tick["t"]):
            raise ValueError(f"{record.get('episode_id')} 틱 {index}: evidence의 t {meta.get('t')} ≠ 틱의 t {tick['t']}")
        if _gripper_label(tick) is None:
            out.append(None)
            continue
        gripper = ((meta.get("aux") or {}).get("gripper") or {})
        value = gripper.get("desired")
        if value not in _STATES:
            raise ValueError(f"{record.get('episode_id')} 틱 {index}: evidence에 그리퍼의 원하는 상태가 없다 ({value!r})")
        out.append(str(value))
    return out


def _stripped_rule(rule: Any) -> str:
    text = str(rule or "")
    changed = True
    while changed:
        changed = False
        for suffix in _RULE_SUFFIXES:
            if text.endswith(suffix):
                text = text[: -len(suffix)]
                changed = True
    return text


def reset_gripper_labels(record: dict[str, Any]) -> list[int]:
    """`q_gripper` 라벨을 허용 적용 전의 한 값(`evidence`의 원하는 상태)으로 되돌린다 — 두 값 라벨·규칙 꼬리·`tolerance` 표지를 지운다.
    돌려주는 것은 바뀐 틱의 색인 목록이다 (이미 한 값이면 빈 목록)."""
    desired = desired_from_evidence(record)
    changed: list[int] = []
    for index, tick in enumerate(record["ticks"]):
        label = _gripper_label(tick)
        if label is None:
            continue
        before = (list(label.get("candidate_ids") or ()), label.get("rule"), label.get("tolerance"))
        label["candidate_ids"] = [desired[index]]
        label["rule"] = _stripped_rule(label.get("rule"))
        label.pop("tolerance", None)
        if before != (list(label["candidate_ids"]), label["rule"], None):
            changed.append(index)
    return changed


# --------------------------------------------------------------------------
# 틱 종류 (R4 C0 표의 분류)
# --------------------------------------------------------------------------


def gripper_tick_class(tick: dict[str, Any]) -> str | None:
    """라벨과 **실행된** 그리퍼(`state.exec.gripper`)로 가른 틱 종류 (:data:`GRIPPER_LABEL_CLASSES`); 라벨이 없으면 None."""
    label = _gripper_label(tick)
    if label is None:
        return None
    ids = [str(cid) for cid in (label.get("candidate_ids") or ())]
    executed = str((((tick.get("request") or {}).get("state") or {}).get("exec") or {}).get("gripper") or "")
    if len(ids) != 1:
        return "window" if executed != "closed" else "window_closed"
    if ids[0] == "closed":
        return "initiate" if executed != "closed" else "settled"
    return "open"


def _executed_closes(record: dict[str, Any]) -> int:
    count = 0
    previous: str | None = None
    for tick in record["ticks"]:
        executed = str((((tick.get("request") or {}).get("state") or {}).get("exec") or {}).get("gripper") or "")
        if previous is not None and executed == "closed" and previous != "closed":
            count += 1
        previous = executed or previous
    return count


def _label_desired(record: dict[str, Any]) -> list[str | None]:
    """증거가 있으면 증거의 원하는 상태, 없으면 한 값 라벨만(두 값은 None)."""
    try:
        return desired_from_evidence(record)
    except (KeyError, ValueError):
        out: list[str | None] = []
        for tick in record["ticks"]:
            label = _gripper_label(tick)
            ids = [str(cid) for cid in (label.get("candidate_ids") or ())] if label else []
            out.append(ids[0] if len(ids) == 1 else None)
        return out


def count_gripper_classes(records: list[dict[str, Any]]) -> dict[str, Any]:
    """레코드들의 그리퍼 틱 종류 수 + 실행된 닫기 수·라벨 전환 수·실행 닫기당 initiate 틱 수."""
    classes: Counter = Counter()
    labelled = 0
    executed_closes = 0
    transitions = 0
    for record in records:
        for tick in record["ticks"]:
            kind = gripper_tick_class(tick)
            if kind is None:
                continue
            labelled += 1
            classes[kind] += 1
        executed_closes += _executed_closes(record)
        transitions += len(gripper_transitions(_label_desired(record)))
    return {
        "classes": {name: int(classes.get(name, 0)) for name in GRIPPER_LABEL_CLASSES},
        "labelled_ticks": labelled, "executed_closes": executed_closes, "label_transitions": transitions,
        "initiate_per_executed_close": (classes.get("initiate", 0) / executed_closes) if executed_closes else None,
        "episodes": len(records),
    }


#: 봉인 분할 — 그 라벨의 종류 수도 적지 않는다 (docs/04 §5: `ood_test`는 읽기·나열·평가 금지; 편 수만 든다).
SEALED_SPLITS = ("ood_test",)


def _counts_by_split(records: list[dict[str, Any]]) -> dict[str, Any]:
    """전체 + 분할별 그리퍼 틱 종류 수. 봉인 분할은 `by_split`에서 빼고 편 수만 `sealed`에 적는다 — 전체(`classes`)는 봉인 분할을 **뺀** 레코드로 센다."""
    open_records = [r for r in records if str(r.get("split")) not in SEALED_SPLITS]
    out = count_gripper_classes(open_records)
    out["by_split"] = {split: count_gripper_classes([r for r in open_records if r.get("split") == split]) for split in sorted({str(r.get("split")) for r in open_records})}
    out["sealed"] = {split: {"episodes": sum(1 for r in records if str(r.get("split")) == split)} for split in SEALED_SPLITS if any(str(r.get("split")) == split for r in records)}
    return out


# --------------------------------------------------------------------------
# A3 — 부모에서 라벨만 바꾼 파생 데이터셋
# --------------------------------------------------------------------------


def derive_gripper_v2_dataset(parent: Path, out: Path, *, early_ticks: int, log: Any = None) -> dict[str, Any]:
    """부모 데이터셋의 레코드를 복사하되 `q_gripper` 라벨만 증거의 원하는 상태로 되돌린 뒤 규칙 v2(k = `early_ticks`)를 건다.
    다른 틱·다른 질문·입력·채택·ACK·대조 쌍·split은 바뀌지 않는다."""
    from robo_jev.data.robot_episodes import CONTRAST_PATH, _tolerate_gripper_transitions_v2, build_manifest, read_episodes, write_episode

    started = time.perf_counter()
    parent, out = Path(parent), Path(out)
    parent_manifest = json.loads((parent / "manifest.json").read_text(encoding="utf-8"))
    found = read_episodes(parent)
    if not found:
        raise FileNotFoundError(f"에피소드가 없다: {parent}")
    parents = [record for _, record in found]
    lineage = {
        "parent_dataset": str(parent), "parent_manifest_version": parent_manifest.get("version"), "parent_config_sha256": parent_manifest.get("config_sha256"),
        "parent_lineage": parent_manifest.get("lineage"), "parent_labels_version": sorted({str(r["versions"].get("labels", "expert")) for r in parents}),
        "gripper_labels": {"version": GRIPPER_LABELS_VERSION, "rule": "v2", "early_ticks": int(early_ticks)},
    }
    if out.exists():
        shutil.rmtree(out)
    children: list[dict[str, Any]] = []
    reset_total = widened_total = 0
    for record in parents:
        child = copy.deepcopy(record)
        reset = reset_gripper_labels(child)
        widened = _tolerate_gripper_transitions_v2(child, int(early_ticks))
        reset_total += len(reset)
        widened_total += len(widened)
        parent_labels = str(child["versions"].get("labels", "expert"))
        child["versions"] = {**child["versions"], "labels": f"{parent_labels}+{GRIPPER_LABELS_VERSION}"}
        child["provenance"] = {
            **child["provenance"],
            "lineage": {
                **{key: value for key, value in lineage.items() if key != "parent_labels_version"},
                "parent_lineage": (record.get("provenance") or {}).get("lineage"),
                "labels_version": f"{parent_labels}+{GRIPPER_LABELS_VERSION}",
                "gripper_label_ticks_reset": reset, "gripper_label_ticks_widened": widened,
            },
        }
        validate_record(child)
        write_episode(child, out)
        children.append(child)
    contrast = parent / CONTRAST_PATH
    if contrast.is_file():
        (out / CONTRAST_PATH).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(contrast, out / CONTRAST_PATH)
    manifest = build_manifest(out, parent_manifest["config"], batch_wall_s=time.perf_counter() - started)
    manifest["lineage"] = {**lineage, "labels_version": f"{'|'.join(lineage['parent_labels_version'])}+{GRIPPER_LABELS_VERSION}", "ticks_reset": reset_total, "ticks_widened": widened_total}
    manifest["gripper_labels"] = {"parent": _counts_by_split(parents), "derived": _counts_by_split(children)}
    manifest["rollout_labels"] = parent_manifest.get("rollout_labels")
    manifest["contrast"] = parent_manifest.get("contrast")
    manifest["run"] = parent_manifest.get("run")
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    if log is not None:
        print(f"{out}: {len(children)} episodes · reset {reset_total} · widened {widened_total} · initiate {manifest['gripper_labels']['parent']['classes']['initiate']} → {manifest['gripper_labels']['derived']['classes']['initiate']}", file=log, flush=True)
    return manifest


# --------------------------------------------------------------------------
# A4 — 모델 주행 기록의 DAgger 재료
# --------------------------------------------------------------------------


def _refuse_sealed(source: Path, *, condition: Any = None, split: Any = None, episode: Any = None) -> None:
    names = [name for name in OOD_SPLITS if name in source.parts or name == condition or name == split]
    if names:
        raise ValueError(
            f"{source}{'' if episode is None else ' ' + str(episode)}: DAgger 재료로 쓸 수 없다 — 경로·조건·split이 봉인 계열({', '.join(names)})이다 "
            f"(ood_dev·ood_test 기록은 학습에 넣지 않는다)"
        )


def build_dagger_dataset(
    sources: list[Path], out: Path, *, early_ticks: int, config: dict[str, Any], cycle: int = 0, log: Any = None,
) -> dict[str, Any]:
    """R4의 폐루프 기록 디렉터리들(`closed_loop.py run --out …/<condition>`)을 첫 DAgger 데이터셋으로 재라벨해 쓴다."""
    from robo_jev.data.dagger import DAGGER_VERSION, count_policy_behaviour
    from robo_jev.data.robot_episodes import _tolerate_gripper_transitions_v2, build_manifest, read_episodes, write_episode

    started = time.perf_counter()
    out = Path(out)
    if out.exists():
        shutil.rmtree(out)
    parents: list[dict[str, Any]] = []
    children: list[dict[str, Any]] = []
    per_episode: list[dict[str, Any]] = []
    policies: list[dict[str, Any]] = []
    scene_splits: Counter = Counter()
    scene_conditions: Counter = Counter()
    for source in [Path(item) for item in sources]:
        _refuse_sealed(source)
        manifest_path = source / "manifest.json"
        source_manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else {}
        loop = source_manifest.get("closed_loop") or {}
        condition = loop.get("condition")
        _refuse_sealed(source, condition=condition)
        found = read_episodes(source)
        if not found:
            raise FileNotFoundError(f"에피소드가 없다: {source}")
        policy_block = {**(loop.get("policy") or {})}
        policies.append({"source": str(source), "condition": condition, "label": loop.get("label"), "episodes": len(found),
                         "policy": {key: policy_block.get(key) for key in ("name", "version", "kind", "checkpoint", "model_id") if key in policy_block}})
        for _, record in found:
            _refuse_sealed(source, split=record.get("split"), episode=record.get("episode_id"))
            parents.append(record)
            child = copy.deepcopy(record)
            reset = reset_gripper_labels(child)
            widened = _tolerate_gripper_transitions_v2(child, int(early_ticks))
            scene_splits[str(record.get("split"))] += 1
            scene_conditions[str(condition)] += 1
            child["split"] = "train"
            for tick in child["ticks"]:
                for label in tick.get("labels") or ():
                    label["relabel"] = True
            child["versions"] = {**child["versions"], "labels": GRIPPER_LABELS_VERSION}
            provenance = child["provenance"]
            provenance["dagger"] = {
                "version": DAGGER_VERSION, "cycle": int(cycle), "seed_base": None,
                "policy": {**(provenance.get("policy") or {}), **{key: policy_block[key] for key in ("checkpoint", "model_id") if key in policy_block}},
                "relabel_source": "expert_v0", "source_dataset": str(source), "scene_condition": condition, "scene_split": record.get("split"),
                "gripper_labels": {"version": GRIPPER_LABELS_VERSION, "rule": "v2", "early_ticks": int(early_ticks)},
                "gripper_label_ticks_reset": reset, "gripper_label_ticks_widened": widened,
            }
            provenance["material"] = "error_family"
            validate_record(child)
            write_episode(child, out)
            children.append(child)
            behaviour = count_policy_behaviour(child)
            per_episode.append(behaviour)
            if log is not None:
                print(f"{child['episode_id']} ticks={behaviour['ticks']:<3} done={behaviour['done']} errors={behaviour['policy_errors']} reset={len(reset)} widened={len(widened)}", file=log, flush=True)
    manifest = build_manifest(out, config, batch_wall_s=time.perf_counter() - started)
    totals = {
        key: sum(int(entry[key]) for entry in per_episode)
        for key in ("ticks", "policy_errors", "gate_disagreement", "main_disagreement", "recoveries", "commitment_changes", "switches", "stops", "conflicts")
    }
    gates: Counter = Counter()
    for entry in per_episode:
        gates.update(entry["gates"])
    manifest["dagger"] = {
        "version": DAGGER_VERSION, "cycle": int(cycle), "seed_base": None, "id_suffix": None,
        "policy": policies, "relabel_source": "expert_v0", "sources": [str(Path(item)) for item in sources],
        "scene_splits": dict(sorted(scene_splits.items())), "scene_conditions": dict(sorted(scene_conditions.items())),
        "episodes": len(per_episode), "done": sum(1 for entry in per_episode if entry["done"]),
        "totals": {**totals, "gates": dict(sorted(gates.items()))},
        "policy_error_rate": round(totals["policy_errors"] / totals["ticks"], 4) if totals["ticks"] else 0.0,
        "per_episode": per_episode,
        "split_note": "split은 train이다 — 계약의 SPLITS에 dagger가 없고 contracts.py는 배포 계약 digest의 조각이라 이름을 더할 수 없다; 장면의 원래 조건·split은 provenance.dagger에 있다",
        "material": "error_family",
    }
    manifest["lineage"] = {"sources": manifest["dagger"]["sources"], "gripper_labels": {"version": GRIPPER_LABELS_VERSION, "rule": "v2", "early_ticks": int(early_ticks)}, "labels_version": GRIPPER_LABELS_VERSION}
    manifest["gripper_labels"] = {"parent": _counts_by_split(parents), "derived": _counts_by_split(children)}
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    if log is not None:
        print(f"{out}: {len(children)} episodes · initiate {manifest['gripper_labels']['parent']['classes']['initiate']} → {manifest['gripper_labels']['derived']['classes']['initiate']}", file=log, flush=True)
    return manifest


# --------------------------------------------------------------------------
# A2 — 이른 답 실험 정책
# --------------------------------------------------------------------------


def reference_gripper_schedule(record: dict[str, Any]) -> dict[str, list[int]]:
    """전문가 기록의 전환 틱을 방향별로: `closed` = open→closed 전환 색인, `open` = closed→open 전환 색인."""
    desired = desired_from_evidence(record)
    out: dict[str, list[int]] = {"closed": [], "open": []}
    for at in gripper_transitions(desired):
        out[str(desired[at])].append(at)
    return out


class EarlyGripperPolicy:
    """전문가를 감싸 `q_gripper`를 **k틱 일찍** 낸다 (Task R5 A2) — 같은 `generate_episode`, 참조·라벨은 전문가의 것.

    같은 seed의 전문가 기록에서 읽은 전환 일정(:func:`reference_gripper_schedule`)을 :meth:`begin`으로 받고, 틱 i가 어떤 전환 t*의
    앞 창(t*−k ≤ i < t*)에 들고 전문가 자신은 아직 반대 상태를 원할 때만 답을 뒤집는다(확률의 모양은 전문가의 것을 그대로 뒤집는다).
    시뮬은 결정적이라 첫 덮어쓰기까지의 상태는 참조 기록과 같으므로 "k틱 일찍"이 실제로 그 틱이다; 그 뒤 궤적이 갈리면
    (실행기가 이른 명령을 그대로 실행한 경우) 일정과 어긋난 창은 전문가의 답이 지킨다. `expert_meta`(참조)는 바꾸지 않는다.
    """

    name = "EarlyGripperExpert"

    def __init__(self, expert: Any, *, early_ticks: int, directions: tuple[str, ...] = ("closed",)) -> None:
        if int(early_ticks) < 0:
            raise ValueError(f"early_ticks: 0 이상이어야 한다 (받은 값: {early_ticks})")
        unknown = [name for name in directions if name not in _STATES]
        if unknown:
            raise ValueError(f"directions: {list(_STATES)} 중에서 골라야 한다 (받은 값: {unknown})")
        self.expert = expert
        self.early_ticks = int(early_ticks)
        self.directions = tuple(directions)
        self.version = f"early-k{self.early_ticks}-{'+'.join(self.directions)}/{getattr(expert, 'version', 'expert')}"
        self._schedule: dict[str, list[int]] = {"closed": [], "open": []}
        self._tick = 0
        self.key: Any = None
        self.overrides: list[int] = []

    def begin(self, key: Any, schedule: dict[str, list[int]]) -> None:
        """새 에피소드: 그 seed의 전환 일정을 받고 틱 계수기를 0으로."""
        self.key = key
        self._schedule = {"closed": [int(v) for v in schedule.get("closed", [])], "open": [int(v) for v in schedule.get("open", [])]}
        self._tick = 0
        self.overrides = []

    def reset(self) -> None:
        self._tick = 0
        self.overrides = []

    def _wanted_early(self, index: int, desired: str | None) -> str | None:
        for direction in self.directions:
            other = "open" if direction == "closed" else "closed"
            if desired != other:
                continue
            if any(at - self.early_ticks <= index < at for at in self._schedule.get(direction, [])):
                return direction
        return None

    def act(self, request: dict[str, Any], commitment: dict[str, Any] | None = None, observation: Any = None) -> dict[str, Any]:
        answers = self.expert.act(request, commitment, observation)
        index = self._tick
        self._tick += 1
        aux = ((answers.get("expert_meta") or {}).get("aux") or {})
        desired = (aux.get("gripper") or {}).get("desired") if aux else None
        direction = self._wanted_early(index, desired) if self.early_ticks else None
        if direction is None:
            return answers
        probabilities = dict(answers.get(_QUESTION) or {})
        high = max(probabilities.values()) if probabilities else 1.0
        low = min(probabilities.values()) if probabilities else 0.0
        other = "open" if direction == "closed" else "closed"
        self.overrides.append(index)
        return {**answers, _QUESTION: {other: low, direction: high}}


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m robo_jev.data.gripper_labels", description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    derive = sub.add_parser("derive", help="A3: 부모 데이터셋에서 q_gripper 라벨만 규칙 v2로 바꾼 파생 데이터셋")
    derive.add_argument("--parent", type=Path, required=True)
    derive.add_argument("--out", type=Path, required=True)
    derive.add_argument("--early-ticks", dest="early_ticks", type=int, required=True, help="전환 앞 허용 폭 k (A2가 정한 값)")
    dagger = sub.add_parser("dagger", help="A4: R4 dev 조건 모델 주행 기록 → 첫 DAgger 데이터셋")
    dagger.add_argument("--source", action="append", type=Path, required=True, help="closed_loop.py run의 조건 디렉터리 (반복)")
    dagger.add_argument("--out", type=Path, required=True)
    dagger.add_argument("--early-ticks", dest="early_ticks", type=int, required=True)
    dagger.add_argument("--config", type=Path, default=Path("configs/data/r1_robot.yaml"))
    dagger.add_argument("--cycle", type=int, default=0)
    args = parser.parse_args(argv)
    if args.command == "derive":
        manifest = derive_gripper_v2_dataset(args.parent, args.out, early_ticks=args.early_ticks, log=sys.stdout)
    else:
        from robo_jev.data.robot_episodes import load_generator_config

        manifest = build_dagger_dataset(args.source, args.out, early_ticks=args.early_ticks, config=load_generator_config(args.config), cycle=args.cycle, log=sys.stdout)
    print(json.dumps({"episodes": manifest["episodes"], "splits": manifest["splits"], "gripper_labels": {name: manifest["gripper_labels"][name]["classes"] for name in ("parent", "derived")}}, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
