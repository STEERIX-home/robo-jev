"""봉인 분할을 열지 않는 에피소드 적재 (Task R7 A3; docs/04 §5 — `ood_test`는 읽기·나열·평가 금지).

평가·분석 경로(폐루프 보고의 run 디렉터리, 판정 칸의 층, `q_done` 층·루프 상태)가 에피소드 디렉터리(`episodes/<id>/streams.jsonl` +
`manifest.json`)를 읽을 때 쓰는 적재기다. 규칙은 학습 적재기(:func:`robo_jev.sampler.plan_manifest_files`, Task R7 A1)와 같다:
**manifest의 파일 항목이 적은 `split`으로 파일을 열기 전에 거른다.** manifest가 없거나 에피소드 항목에 `split`이 없으면 열지 않고는
봉인 여부를 알 수 없으므로 거절한다 — 읽고 버리지 않는다. 봉인 편은 수만 센다.

데이터 생성·파생·QA(`data.lineage`·`gripper_labels.derive`·`rollouts`·`validate`)는 봉인 분할을 **데이터로** 옮기거나 그 계약을 검사하는
정당한 경로라 이 적재기를 쓰지 않는다(`robot_episodes.read_episodes`). 이 모듈은 generator·simulator를 import하지 않는다.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

__all__ = ["SEALED_SPLITS", "episode_entries", "read_episode_ids", "read_open_episodes"]

#: 봉인 분할 — 이 적재기는 이 분할의 파일을 열지 않는다.
SEALED_SPLITS = ("ood_test",)


def episode_entries(directory: str | Path) -> dict[str, dict[str, Any]]:
    """manifest의 에피소드 항목 `{episode_id: {path, split, …}}` — 메타데이터만 읽는다. manifest가 없거나 어떤 에피소드 항목에 `split`이
    없으면 거절한다(열지 않고는 봉인 여부를 알 수 없다)."""
    directory = Path(directory)
    manifest_path = directory / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError(f"{directory}: manifest.json이 없다 — 파일을 열지 않고는 봉인 분할을 가를 수 없으므로 읽지 않는다")
    files = json.loads(manifest_path.read_text(encoding="utf-8")).get("files") or {}
    if isinstance(files, list):
        files = {str(entry.get("path")): entry for entry in files if isinstance(entry, dict)}
    out: dict[str, dict[str, Any]] = {}
    for name, entry in files.items():
        if not str(name).startswith("episodes/"):
            continue
        if not isinstance(entry, dict) or entry.get("split") is None:
            raise ValueError(f"{directory}: manifest의 {name}에 split이 없다 — 열지 않고는 봉인 여부를 알 수 없으므로 거절한다")
        episode_id = str(entry.get("episode_id") or Path(str(name)).parent.name)
        out[episode_id] = {**entry, "path": str(name), "split": str(entry["split"])}
    return out


def _read(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def read_open_episodes(directory: str | Path, *, splits: tuple[str, ...] | list[str] | None = None) -> tuple[list[dict[str, Any]], dict[str, dict[str, int]]]:
    """디렉터리의 **봉인되지 않은** 에피소드 레코드(파일 이름 순서)와 봉인 분할의 편 수. `splits`를 주면 그 분할의 파일만 연다."""
    directory = Path(directory)
    wanted = None if splits is None else {str(split) for split in splits}
    sealed: dict[str, dict[str, int]] = {}
    chosen: list[str] = []
    for episode_id, entry in episode_entries(directory).items():
        if entry["split"] in SEALED_SPLITS:
            sealed.setdefault(entry["split"], {"episodes": 0})["episodes"] += 1
            continue
        if wanted is not None and entry["split"] not in wanted:
            continue
        chosen.append(entry["path"])
    records: list[dict[str, Any]] = []
    for name in sorted(chosen):
        records.extend(_read(directory / name))
    return records, sealed


def read_episode_ids(directory: str | Path, episode_ids: list[str]) -> list[dict[str, Any]]:
    """이름이 주어진 편들을 **그 순서로** — manifest에 없는 편이나 봉인 분할의 편이 섞였으면 어느 파일도 열기 전에 거절한다."""
    directory = Path(directory)
    entries = episode_entries(directory)
    unknown = [episode_id for episode_id in episode_ids if episode_id not in entries]
    if unknown:
        raise ValueError(f"{directory}: manifest에 없는 편이다: {unknown[:5]} — split을 모르는 파일은 열지 않는다")
    sealed = [episode_id for episode_id in episode_ids if entries[episode_id]["split"] in SEALED_SPLITS]
    if sealed:
        raise ValueError(f"{directory}: 봉인 분할의 편은 열지 않는다 ({len(sealed)}편) — 평가·분석 경로는 봉인 편을 읽지 않는다")
    out: list[dict[str, Any]] = []
    for episode_id in episode_ids:
        rows = _read(directory / entries[episode_id]["path"])
        if len(rows) != 1:
            raise ValueError(f"{directory / entries[episode_id]['path']}: 에피소드 파일에는 레코드가 하나여야 한다 ({len(rows)}개)")
        out.append(rows[0])
    return out
