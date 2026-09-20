"""검수 표본 내보내기 — 층화·quota·이중 검수·봉인 분리·파일 (D0 fixture)."""

import json

from helpers import FIXTURES

from robo_jev.data.review_sample import REVIEW_SAMPLE_VERSION, build_sample, render_sheet, write_sample


def _load(name):
    return [json.loads(line) for line in (FIXTURES / name).read_text(encoding="utf-8").splitlines()]


def test_the_review_sample_is_stratified_with_quotas_double_review_and_no_sealed_records_in_open_strata(tmp_path):
    singles, streams = _load("d0.jsonl"), _load("d0_streams.jsonl")
    for record in singles:
        record.setdefault("provenance", {})["domain"] = ["spatial", "dom", "workflow", "rules"][hash(record["origin_group"]) % 4]
    # D0 fixture에는 ood_test가 없다 — 봉인 층을 보려고 test 스트림 하나를 ood_test로 옮긴 사본을 봉인 입력으로 준다.
    sealed_streams = [dict(record, split="ood_test") for record in streams if record["split"] == "test"]
    rows, table = build_sample(streams, singles, per_stratum=4, double_review=10, seed=1, sealed_robot=sealed_streams, sealed_single=[])
    assert table["version"] == REVIEW_SAMPLE_VERSION and table["questions"] == len(rows) >= 40
    assert table["double_review"] == 10 and sum(1 for row in rows if row["double_review"]) == 10
    names = set(table["strata"])
    assert {"robot/q_main/steady", "robot/gate/steady", "robot/aux/steady", "sealed/robot"} <= names
    assert any(name.startswith("non_robot:") for name in names) and table["core_strata"] == len(names) - 1
    for name, entry in table["strata"].items():
        assert entry["sampled"] <= (4 if not name.startswith("sealed/") else 2) and entry["sampled"] + entry["short"] >= min(entry["population"], 1)
    ids = [row["sample_id"] for row in rows]
    assert len(set(ids)) == len(ids) and ids == sorted(ids)
    assert all(row["split"] != "ood_test" for row in rows if not row["sealed"]) and any(row["sealed"] for row in rows)
    for row in rows:
        assert row["label"] is not None and row["candidates"] and row["question"] and row["context"]
        assert "true_state" not in row["context"] and "rule_trace" not in row["context"]
        assert row["verdict"] == {"correct": None, "severity": None, "note": ""}
        if row["stratum"].startswith("robot/"):
            assert row["source"]["tick"] is not None and row["tick_class"] in ("steady", "event", "goal_change", "other")
    # 같은 틱에서 층마다 하나, 같은 에피소드에서 층마다 ≤ 2.
    seen = {}
    for row in rows:
        key = (row["stratum"], row["source"]["record_id"])
        seen[key] = seen.get(key, 0) + 1
    assert max(seen.values()) <= 2
    paths = write_sample(tmp_path / "sample", rows, table)
    assert all(path.is_file() for path in paths.values())
    sheet = paths["sheet"].read_text(encoding="utf-8")
    assert sheet.count("LABEL:") == len(rows) and "[이중]" in sheet and "[봉인]" in sheet and "protocol.md" in sheet
    assert "중대한 오라벨 2 %" in paths["protocol"].read_text(encoding="utf-8")
    again, _ = build_sample(streams, singles, per_stratum=4, double_review=10, seed=1, sealed_robot=sealed_streams, sealed_single=[])
    assert [(row["sample_id"], row["source"], row["question_id"]) for row in again] == [(row["sample_id"], row["source"], row["question_id"]) for row in rows]
    assert render_sheet(rows[:1], table).startswith("# D1 검수 시트")
