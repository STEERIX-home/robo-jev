"""bench-v0 정의 (Task bench0 §1) — 층화 표본이 결정적이고 쌍·T6 묶음을 깨지 않으며, 정의 파일의 기대값과 대조한다."""

import json
from collections import Counter

import pytest
from helpers import REPO

from robo_jev.bench.define import (
    allocate,
    build,
    ids_sha256,
    load_config,
    select_pilot,
    select_set,
    stratum_of,
    units_of,
)
from robo_jev.decisions.records import select_records

CONFIG = REPO / "configs" / "bench" / "bench-v0.yaml"


def _record(rid: str, group: str, language: str = "en", **meta) -> dict:
    return {
        "schema": "simgen-request/0", "id": rid, "meta": {"group": group, "language": language, **meta},
        "request": {"state": f"state {rid}", "questions": {"q": {"type": "noul", "instructions": "Will it?"}}},
        "labels": {"q": {"type": "noul", "p_yes": 0.5}},
    }


def _singles(n: int = 200) -> list[dict]:
    groups = ["P1", "P3", "P4", "P1+P5"]
    return [_record(f"r{i:04d}", groups[i % 4] if i % 7 else "P6", "ko" if i % 5 == 0 else "en") for i in range(n)]


def _pairs(n_pairs: int = 60) -> list[dict]:
    out = []
    for i in range(n_pairs):
        a, b = f"p{i:03d}a", f"p{i:03d}b"
        group = ["P1", "P3", "P5"][i % 3]
        out.append(_record(a, group, pair={"of": b, "kind": "numeric" if i % 2 else "wording", "role": "base"}))
        out.append(_record(b, group, pair={"of": a, "kind": "numeric" if i % 2 else "wording", "role": "variant"}))
    return out


def _families(n: int = 20, variants: int = 4) -> list[dict]:
    out = []
    for i in range(n):
        base = f"base-{i:03d}"
        group = ["P1", "P4"][i % 2]
        out.append(_record(base, group, t6={"base": None, "variant": "base"}))
        for k in range(variants - (i % 2)):  # 묶음 크기가 달라도 된다 (변형이 맞지 않는 장면)
            out.append(_record(f"var{k}-{i:03d}", group, t6={"base": base, "variant": f"kind{k}"}))
    return out


def test_allocate_is_proportional_and_sums_to_the_total():
    sizes = {"a": 10, "b": 25, "c": 65}
    assert allocate(sizes, 20) == {"a": 2, "b": 5, "c": 13}
    out = allocate({"x": 1, "y": 1, "z": 1}, 2)
    assert sum(out.values()) == 2 and out == {"x": 1, "y": 1, "z": 0}  # 같은 나머지는 층 이름 순
    assert allocate(sizes, 1000) == sizes


def test_record_sampling_is_deterministic_stratified_and_in_file_order():
    records = _singles()
    a = select_set(records, set_name="T1", unit="record", count=50, seed=7)
    b = select_set(records, set_name="T1", unit="record", count=50, seed=7)
    c = select_set(records, set_name="T1", unit="record", count=50, seed=8)
    assert [r["id"] for r in a] == [r["id"] for r in b] and len(a) == 50
    assert [r["id"] for r in a] != [r["id"] for r in c]
    assert [r["id"] for r in a] == sorted(r["id"] for r in a)  # 원본 파일 순서
    whole = Counter(stratum_of(r) for r in records)
    got = Counter(stratum_of(r) for r in a)
    for name, size in whole.items():
        assert abs(got[name] - 50 * size / len(records)) < 1.0  # 비례 몫 ± 1


def test_one_stratum_reduces_to_the_diagnosis_tools_sampler():
    """층이 하나면 진단 도구의 쌍 보존 표본기(select_records, --limit)와 같은 레코드를 고른다."""
    from robo_jev.bench.define import stratum_seed

    records = [_record(f"r{i:03d}", "P1+P3") for i in range(120)]
    ours = select_set(records, set_name="T4", unit="record", count=30, seed=3)
    tool = select_records(records, 30, seed=stratum_seed(3, "T4", stratum_of(records[0])))
    assert [r["id"] for r in ours] == [r["id"] for r in tool]


def test_pairs_are_never_split():
    records = _pairs()
    chosen = select_set(records, set_name="T5", unit="pair", count=15, seed=1)
    assert len(chosen) == 30
    ids = {r["id"] for r in chosen}
    for record in chosen:
        assert record["meta"]["pair"]["of"] in ids
    assert all(len(group) == 2 for group in units_of(chosen, "pair"))


def test_t6_families_come_whole():
    records = _families()
    chosen = select_set(records, set_name="T6", unit="t6_family", count=6, seed=2)
    families = units_of(chosen, "t6_family")
    assert len(families) == 6
    by_base = Counter(((r["meta"].get("t6") or {}).get("base") or r["id"]) for r in records)
    for family in families:
        base = family[0]
        assert base["meta"]["t6"]["variant"] == "base"
        assert len(family) == by_base[base["id"]]  # 기준 + 그 모든 변형


def test_all_unit_keeps_everything_and_pilot_avoids_the_bench():
    records = _singles(80)
    assert select_set(records, set_name="G2", unit="all", count=None, seed=0) == records
    bench = select_set(records, set_name="T1", unit="record", count=30, seed=0)
    pilot = select_pilot(records, [r["id"] for r in bench], count=20, seed=0)
    assert len(pilot) == 20 and not {r["id"] for r in pilot} & {r["id"] for r in bench}


def test_build_writes_source_bytes_and_rejects_a_changed_definition(tmp_path):
    source = tmp_path / "T1.jsonl"
    records = _singles(40)
    lines = [json.dumps(r, ensure_ascii=False, separators=(", ", ": ")) for r in records]  # 원본의 서식 그대로 옮겨야 한다
    source.write_text("\n".join(lines) + "\n", encoding="utf-8")
    config = {"version": "bench-test", "seed": 5, "sets": {"T1": {"source": str(source), "unit": "record", "count": 10}},
              "pilot": {"from": "T1", "count": 5}}
    manifest = build(config, root=tmp_path, out_dir=tmp_path / "out")
    written = (tmp_path / "out" / "T1.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(written) == 10 and set(written) <= set(lines)
    entry = manifest["sets"]["T1"]
    assert entry["ids_sha256"] == ids_sha256(json.loads(line)["id"] for line in written)
    assert manifest["pilot"]["records"] == 5 and manifest["pilot"]["overlap_with_bench"] == 0
    config["sets"]["T1"]["expect"] = {"ids_sha256": entry["ids_sha256"], "records": 10, "questions": 10}
    build(config, root=tmp_path, out_dir=None)  # 같은 정의 → 통과
    config["seed"] = 6
    with pytest.raises(ValueError, match="ids_sha256"):
        build(config, root=tmp_path, out_dir=None)


def test_the_committed_definition_reproduces_its_expected_selection():
    """정의 파일(커밋)의 expect를 원본에서 다시 뽑아 맞춘다 — 원본(artifacts, 커밋 안 함)이 없는 checkout에서는 건너뛴다."""
    config = load_config(CONFIG)
    missing = [spec["source"] for spec in config["sets"].values() if not (REPO / spec["source"]).is_file()]
    if missing:
        pytest.skip(f"sealed sources not on this machine: {missing[:2]}")
    manifest = build(config, root=REPO, out_dir=None, check=True)
    assert [manifest["sets"][name]["records"] for name in ("T1", "T2", "T3", "T4", "T5", "T6", "T7")] == [300, 500, 300, 300, 300, 390, 300]
    assert manifest["sets"]["T5"]["units"] == 150 and manifest["sets"]["T6"]["units"] == 30
    assert manifest["sets"]["G1"]["records"] == 314 and manifest["sets"]["G2"]["records"] == 1000
    assert len(config["models"]["default"]) == 6 and all(m.get("why") for m in config["models"]["default"])
