"""봉인 분할을 열지 않는 에피소드 적재 검사 (Task R7 A3) — 평가·분석 경로(폐루프 보고, 판정 칸 층, `q_done` 층)가 에피소드 디렉터리를
읽을 때 manifest의 `split`으로 **먼저** 거르고, 봉인(`ood_test`) 파일은 열지 않는다. 봉인 파일은 읽기 권한이 없게 만들어 둔다 — 열려고 하면
`PermissionError`가 난다."""

import json

import pytest

from robo_jev.data.sealed import SEALED_SPLITS, read_episode_ids, read_open_episodes


def _episode(episode_id: str, split: str) -> dict:
    return {"schema_version": "stream-v0", "episode_id": episode_id, "split": split, "ticks": [{"t": 0}]}


def _dataset(tmp_path, splits: dict[str, str], *, manifest: bool = True, missing_split: str | None = None) -> None:
    """`episodes/<id>/streams.jsonl` + manifest(파일마다 split·episode_id). 봉인 편의 파일은 읽을 수 없고 JSON도 아니다."""
    entries = {}
    for episode_id, split in splits.items():
        path = tmp_path / "episodes" / episode_id / "streams.jsonl"
        path.parent.mkdir(parents=True)
        if split in SEALED_SPLITS:
            path.write_text("sealed — never opened\n", encoding="utf-8")
            path.chmod(0)
        else:
            path.write_text(json.dumps(_episode(episode_id, split)) + "\n", encoding="utf-8")
        entry = {"episode_id": episode_id, "split": split, "sha256": "x"}
        if episode_id == missing_split:
            del entry["split"]
        entries[f"episodes/{episode_id}/streams.jsonl"] = entry
    entries["contrast/records.jsonl"] = {"sha256": "y"}  # 에피소드가 아닌 항목은 보지 않는다
    if manifest:
        (tmp_path / "manifest.json").write_text(json.dumps({"files": entries}), encoding="utf-8")


def _unlock(tmp_path) -> None:
    for path in (tmp_path / "episodes").glob("*/streams.jsonl"):
        path.chmod(0o600)


def test_read_open_episodes_filters_on_the_manifest_split_before_opening_and_counts_the_sealed_split_only(tmp_path):
    _dataset(tmp_path, {"ep-a": "train", "ep-b": "dev", "ep-s": "ood_test"})
    try:
        records, sealed = read_open_episodes(tmp_path)
        dev_only, _ = read_open_episodes(tmp_path, splits=("dev",))
    finally:
        _unlock(tmp_path)
    assert [record["episode_id"] for record in records] == ["ep-a", "ep-b"] and sealed == {"ood_test": {"episodes": 1}}
    assert [record["episode_id"] for record in dev_only] == ["ep-b"]


def test_read_open_episodes_refuses_a_directory_without_a_manifest_or_an_entry_without_a_split(tmp_path):
    _dataset(tmp_path / "a", {"ep-a": "train"}, manifest=False)
    with pytest.raises(ValueError, match="manifest"):
        read_open_episodes(tmp_path / "a")
    _dataset(tmp_path / "b", {"ep-a": "train", "ep-s": "ood_test"}, missing_split="ep-a")
    try:
        with pytest.raises(ValueError, match="split"):
            read_open_episodes(tmp_path / "b")
    finally:
        _unlock(tmp_path / "b")


def test_read_episode_ids_opens_the_named_episodes_in_order_and_refuses_a_sealed_or_unknown_id_without_opening_it(tmp_path):
    _dataset(tmp_path, {"ep-a": "ood_dev", "ep-b": "ood_dev", "ep-s": "ood_test"})
    try:
        assert [record["episode_id"] for record in read_episode_ids(tmp_path, ["ep-b", "ep-a"])] == ["ep-b", "ep-a"]
        with pytest.raises(ValueError, match="봉인"):
            read_episode_ids(tmp_path, ["ep-a", "ep-s"])
        with pytest.raises(ValueError, match="manifest에 없는"):
            read_episode_ids(tmp_path, ["ep-z"])
    finally:
        _unlock(tmp_path)


def test_the_closed_loop_report_reads_run_directories_through_the_manifest_and_never_opens_a_sealed_episode(tmp_path):
    """`closed_loop._read_records`(보고서가 run 디렉터리를 읽는 곳)가 manifest로 먼저 거른다 — 봉인 편이 섞인 디렉터리를 가리켜도 그 파일은 열지 않는다."""
    from robo_jev.closed_loop import _read_records

    _dataset(tmp_path, {"ep-a": "ood_dev", "ep-s": "ood_test"})
    try:
        assert [record["episode_id"] for record in _read_records(tmp_path)] == ["ep-a"]
    finally:
        _unlock(tmp_path)


def test_the_decision_cell_readers_refuse_a_sealed_episode_id_and_a_sealed_population_split(tmp_path):
    """`scripts/decision_cell_strata.py`의 편 읽기(판정 칸 q_main·q_gripper·q_done 층)는 manifest가 봉인이라 적은 편을 열지 않고 거절하고,
    분할 전체를 읽는 모집단 보기(`split_episodes`)는 봉인 분할을 청하면 거절한다."""
    import importlib.util
    import sys

    from helpers import REPO

    spec = importlib.util.spec_from_file_location("decision_cell_strata_sealed", REPO / "scripts" / "decision_cell_strata.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    _dataset(tmp_path, {"ep-a": "ood_dev", "ep-s": "ood_test"})
    try:
        for reader in (module.dataset_ticks, module.gripper_dataset_ticks, module.done_dataset_ticks):
            with pytest.raises(ValueError, match="봉인"):
                reader(tmp_path, ["ep-a", "ep-s"])
        with pytest.raises(ValueError, match="봉인"):
            module.split_episodes(tmp_path / "manifest.json", "ood_test")
        assert module.split_episodes(tmp_path / "manifest.json", "ood_dev") == ["ep-a"]
    finally:
        _unlock(tmp_path)
