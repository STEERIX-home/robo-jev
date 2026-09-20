"""라벨 계보 — 키프레임 rollout 라벨을 **계보를 유지한 후속 데이터 버전**에 붙인다 (docs/04 §6 D1 행, docs/08 §7·§8).

    uv run python -m robo_jev.data.lineage --dataset artifacts/datasets/d1-robot/d1 \\
        --rollouts artifacts/datasets/d1-robot/d1/rollouts --out artifacts/datasets/d1-robot/d1-rollout-labels

부모 배치(전문가 라벨)의 레코드를 그대로 복사하되, `rollouts/labels.jsonl`이 라벨한 키프레임 틱의 `q_main` 라벨을 rollout 라벨
(:func:`robo_jev.sim.label.label_main_decision` — `event_results`·`rollout_reason`·`label_confidence`·`weight`)로 바꾼다. 바뀐 전문가
라벨은 새 라벨의 `expert` 항목(`candidate_ids`·`label_confidence`·`rule`)에 남는다. 다른 틱·다른 질문·입력·채택·ACK·대조 쌍·split은
바뀌지 않는다. 레코드의 `versions.labels`가 :data:`LABELS_VERSION`, `provenance.lineage`가 부모 배치·rollout 디렉터리·버전을 말하며,
manifest의 `lineage`·`rollout_labels`가 집계(신뢰도·rollout 사유·종류·split별)를 든다. rollout의 `costing.json`이 적은 버전이 레코드의
`versions`와 다르면 :class:`robo_jev.data.rollouts.ConfigMismatch`로 거절한다.
"""

from __future__ import annotations

import argparse
import copy
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any

from robo_jev.contracts import validate_record
from robo_jev.data.robot_episodes import CONTRAST_PATH, build_manifest, read_episodes, write_episode
from robo_jev.data.rollouts import ROLLOUTS_VERSION, VERSION_KEYS, ConfigMismatch

__all__ = ["LABELS_VERSION", "VERSION_MANIFEST", "attach_rollout_labels", "main", "write_version_manifest"]

LABELS_VERSION = "labels-rollout-v1"


def attach_rollout_labels(dataset: Path, rollouts: Path, out: Path, *, log: Any = None) -> dict[str, Any]:
    started = time.perf_counter()
    dataset, rollouts, out = Path(dataset), Path(rollouts), Path(out)
    parent_manifest = json.loads((dataset / "manifest.json").read_text(encoding="utf-8"))
    cost = json.loads((rollouts / "costing.json").read_text(encoding="utf-8"))
    labels = [json.loads(line) for line in (rollouts / "labels.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    found = read_episodes(dataset)
    if not found:
        raise FileNotFoundError(f"에피소드가 없다: {dataset}")
    running = {key: str(value) for key, value in (cost.get("versions") or {}).items()}
    by_id: dict[str, dict[str, Any]] = {}
    for _, record in found:
        differences = [f"{key}: 레코드 {record['versions'].get(key)} ≠ rollout {running.get(key)}" for key in VERSION_KEYS if str(record["versions"].get(key)) != running.get(key)]
        if differences:
            raise ConfigMismatch(f"{record['episode_id']}: rollout을 만든 버전이 레코드와 다르다 — " + "; ".join(differences))
        by_id[record["episode_id"]] = copy.deepcopy(record)

    counts: dict[str, Any] = {"ticks_labelled": 0, "confidence": {}, "rollout_reason": {}, "kind": {}, "by_split": {}, "weight": {}, "rollouts": 0}
    lineage_ticks: dict[str, list[int]] = {}
    for entry in labels:
        record = by_id.get(str(entry["episode_id"]))
        if record is None:
            raise KeyError(f"rollout 라벨의 에피소드가 배치에 없다: {entry['episode_id']}")
        index = int(entry["index"])
        tick = record["ticks"][index]
        if int(tick["t"]) != int(entry["t"]):
            raise ValueError(f"{entry['episode_id']} 틱 {index}: t {tick['t']} ≠ 라벨 {entry['t']}")
        position = next((i for i, label in enumerate(tick["labels"]) if label.get("question_id") == "q_main"), None)
        if position is None:
            raise ValueError(f"{entry['episode_id']} 틱 {index}: q_main 라벨이 없다")
        old = tick["labels"][position]
        new = {
            **copy.deepcopy(entry["label"]),
            "expert": {key: old.get(key) for key in ("candidate_ids", "label_confidence", "rule", "weight") if key in old},
            "keyframe_kind": str(entry.get("kind")),
        }
        tick["labels"][position] = new
        lineage_ticks.setdefault(record["episode_id"], []).append(index)
        counts["ticks_labelled"] += 1
        counts["rollouts"] += int(entry.get("rollouts", 0))
        for name, value in (("confidence", new.get("label_confidence")), ("rollout_reason", new.get("rollout_reason")), ("kind", entry.get("kind")), ("by_split", record.get("split"))):
            counts[name][str(value)] = counts[name].get(str(value), 0) + 1
        if "weight" in new:
            counts["weight"][str(new["weight"])] = counts["weight"].get(str(new["weight"]), 0) + 1

    lineage = {
        "parent_dataset": str(dataset),
        "parent_manifest_version": parent_manifest.get("version"),
        "parent_config_sha256": parent_manifest.get("config_sha256"),
        "rollouts": str(rollouts),
        "rollouts_version": str(cost.get("version", ROLLOUTS_VERSION)),
        "events_version": (cost.get("keyframes") or {}).get("events_version"),
        "labels_version": LABELS_VERSION,
    }
    if out.exists():
        shutil.rmtree(out)
    for record in by_id.values():
        record["versions"] = {**record["versions"], "labels": LABELS_VERSION}
        record["provenance"] = {**record["provenance"], "lineage": {**lineage, "rollout_label_ticks": sorted(lineage_ticks.get(record["episode_id"], []))}}
        validate_record(record)
        write_episode(record, out)
    contrast = dataset / CONTRAST_PATH
    if contrast.is_file():
        (out / CONTRAST_PATH).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(contrast, out / CONTRAST_PATH)
    manifest = build_manifest(out, parent_manifest["config"], batch_wall_s=time.perf_counter() - started)
    manifest["lineage"] = lineage
    manifest["rollout_labels"] = {name: (dict(sorted(value.items())) if isinstance(value, dict) else value) for name, value in counts.items()}
    manifest["rollout_labels"]["episodes_with_labels"] = len(lineage_ticks)
    manifest["rollout_labels"]["keyframes_in_rollouts"] = (cost.get("keyframes") or {}).get("keyframes")
    manifest["contrast"] = parent_manifest.get("contrast")
    manifest["run"] = parent_manifest.get("run")
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    if log is not None:
        print(f"{out}: {len(by_id)} episodes, {counts['ticks_labelled']} rollout-labelled ticks, confidence {counts['confidence']}", file=log, flush=True)
    return manifest


VERSION_MANIFEST = "dataset-version-v1"


def write_version_manifest(out: Path, name: str, parts: dict[str, Path], *, qa: dict[str, Path] | None = None, notes: dict[str, Any] | None = None) -> dict[str, Any]:
    """데이터 **버전** manifest (docs/04 §6 표의 한 행): 부분 데이터셋(로봇 스트림·후속 라벨 버전·비로봇)의 manifest 경로·sha256·버전·집계와
    QA 보고서 경로를 한 파일에 모은다 — 학습 설정의 `dataset_manifests`가 가리킬 부분들의 기록이지, 적재기가 읽는 manifest는 아니다."""
    import hashlib

    entries: dict[str, Any] = {}
    for part, path in parts.items():
        path = Path(path)
        manifest = json.loads(path.read_text(encoding="utf-8"))
        counts = manifest.get("counts") or {}
        entries[part] = {
            "manifest": str(path),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "version": manifest.get("version"),
            "generator": manifest.get("generator"),
            "config_version": manifest.get("config_version"),
            "config_sha256": manifest.get("config_sha256"),
            "versions": manifest.get("versions"),
            "episodes": manifest.get("episodes"),
            "ticks": (manifest.get("ticks") or {}).get("total") if isinstance(manifest.get("ticks"), dict) else None,
            "records": counts.get("states"),
            "questions": manifest.get("questions") if isinstance(manifest.get("questions"), int) else counts.get("questions"),
            "labels": manifest.get("labels") if isinstance(manifest.get("labels"), int) else counts.get("labels"),
            "splits": manifest.get("splits") or counts.get("splits"),
            "contrast": {key: (manifest.get("contrast") or counts.get("contrast") or {}).get(key) for key in ("pairs", "by_split", "by_kind", "by_domain", "base_records")},
            "lineage": manifest.get("lineage"),
            "rollout_labels": manifest.get("rollout_labels"),
            "files": len(manifest.get("files") or {}),
        }
    version = {
        "version": VERSION_MANIFEST, "name": name, "written_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "parts": entries,
        "qa_reports": {key: str(path) for key, path in (qa or {}).items()}, "notes": notes or {},
    }
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(version, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return version


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m robo_jev.data.lineage", description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", type=Path, default=None)
    parser.add_argument("--rollouts", type=Path, default=None, help="기본 <dataset>/rollouts")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--version-manifest", action="append", default=[], metavar="PART=MANIFEST", help="데이터 버전 manifest를 쓴다 (부분 이름=manifest 경로; 반복)")
    parser.add_argument("--name", default="d1")
    parser.add_argument("--qa", action="append", default=[], metavar="PART=REPORT")
    args = parser.parse_args(argv)
    if args.version_manifest:
        parts = {item.split("=", 1)[0]: Path(item.split("=", 1)[1]) for item in args.version_manifest}
        qa = {item.split("=", 1)[0]: Path(item.split("=", 1)[1]) for item in args.qa}
        version = write_version_manifest(args.out, args.name, parts, qa=qa)
        print(json.dumps({part: {key: entry[key] for key in ("version", "episodes", "ticks", "records", "questions", "labels")} for part, entry in version["parts"].items()}, ensure_ascii=False, indent=1))
        return 0
    if args.dataset is None:
        parser.error("--dataset가 필요하다 (또는 --version-manifest)")
    manifest = attach_rollout_labels(args.dataset, args.rollouts or (args.dataset / "rollouts"), args.out, log=sys.stdout)
    print(json.dumps({"lineage": manifest["lineage"], "rollout_labels": manifest["rollout_labels"]}, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
