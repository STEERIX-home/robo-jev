"""빼 둔 칸과 봉인된 시험 세트의 정직성 (Task v0b 지시서 §4·§8).

* 학습 나눔은 빼 둔 칸을 **만들 수 없다**: 문체 log·빼 둔 상황·빼 둔 바꿔 쓰기 묶음·조합 P1+P3을 강제하면 생성이 거절된다(HeldOutError),
  T3 범위 칸은 뽑는 범위가 학습 쪽으로 좁혀진다.
* 학습 표본(그리고 만들어 둔 학습 세트 전부 — artifacts/simgen/v0b/train-50k.jsonl)을 훑어 빼 둔 칸이 0건, 요청 글의 해시가 시험 세트와
  겹치지 않는다.
* 시험 세트가 정의대로다: T2는 레코드마다 빼 둔 축 하나 이상, T3는 강제한 칸 하나만, T4는 P1+P3만, 나머지는 빼 둔 칸 없음; manifest의
  sha256이 파일과 맞다.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from simgen_helpers import sample_records

from robo_jev.simgen.catalog import GROUPS, SCENE_GROUP, default_config
from robo_jev.simgen.generate import HeldOutError, build_moment
from robo_jev.simgen.holdout import TRAIN_STYLES, cells_hit, load_holdout, ranges_for, request_hash, split_for

ARTIFACTS = Path(__file__).resolve().parents[1] / "artifacts" / "simgen" / "v0b"
TRAIN = ARTIFACTS / "train-50k.jsonl"
TESTS = ARTIFACTS / "tests"


@pytest.fixture(scope="module")
def holdout():
    return load_holdout()


def _held_cells(holdout, meta: dict) -> list[str]:
    """레코드의 meta(문체·상황·묶음·조합·T3 변수 값)만으로 다시 잰 빼 둔 칸."""
    out = []
    if meta["style"] in holdout.styles:
        out.append("style:" + meta["style"])
    if meta["setting"] in holdout.heldout_settings(meta["domain"]):
        out.append(f"setting:{meta['domain']}[{meta['setting']}]")
    if meta["pool"] == holdout.paraphrase_pool:
        out.append("pool:" + meta["pool"])
    if meta["group"] in holdout.groups:
        out.append("group:" + meta["group"])
    for cell in holdout.ranges:
        value = meta["holdout_vars"].get(cell.var)
        if meta["scene"] in cell.scenes and value is not None and cell.in_test(value, meta["body"]):
            out.append(cell.name)
    return out


# ----------------------------------------------------------------------------------------------------------------
# 생성 시점 거절
# ----------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("forced", [{"style": "log"}, {"setting": 4}, {"pool": "heldout"}, {"group": "P1+P3"}, {"scene": "p1p3_step_in_carrying_tall"}])
def test_the_training_split_cannot_build_a_held_out_cell(forced):
    with pytest.raises(HeldOutError):
        build_moment(default_config(), 777, "cafe", {"scene": "p1_step_in", **forced} if "scene" not in forced and "group" not in forced else forced)


def test_the_training_split_draws_range_variables_on_the_training_side(holdout):
    train = split_for("train", holdout)
    assert ranges_for(holdout, train, "p5_braking", "mobile_base") == {"fill_reading": (-1e9, 0.92), "robot_speed": (-1e9, 1.25)}
    assert ranges_for(holdout, train, "p6_blind_corner", "humanoid") == {"corner_lateral": (0.5, 1e9)}
    t3 = split_for("T3", holdout, cell="p3_steep_ramp", train_ranges=True)
    assert ranges_for(holdout, t3, "p3_ramp", "mobile_base") == {"ramp_angle": (7.0, 1e9), "load_aspect": (-1e9, 5.0)}


def test_training_groups_exclude_the_held_out_combination(holdout):
    assert "P1+P3" not in split_for("train", holdout).groups
    assert set(split_for("train", holdout).groups) == set(GROUPS) - set(holdout.groups)
    assert split_for("T4", holdout).groups == ("P1+P3",)


# ----------------------------------------------------------------------------------------------------------------
# 학습 표본·학습 세트 전부 훑기
# ----------------------------------------------------------------------------------------------------------------


def test_a_training_sample_hits_no_held_out_cell(holdout):
    records = sample_records()
    for r in records:
        meta = r["meta"]
        assert meta["holdout"] == [] and _held_cells(holdout, meta) == [], (r["id"], meta["holdout"], _held_cells(holdout, meta))
        assert meta["style"] in TRAIN_STYLES and meta["split"] == "train"
    # 관측에서 다시 잰 칸도 0건(meta.holdout_vars가 관측의 값과 같다)
    cfg = default_config()
    from robo_jev.simgen.generate import split_from

    for r in records[:300]:
        g = r["generator"]
        forced = {k: v for k, v in g["forced"].items() if k != "domain"}
        m = build_moment(cfg, g["seed"], g["forced"]["domain"], forced, g["variant"], split_from(g["split"]))
        assert cells_hit(holdout, m.obs) == []


def _scan(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            yield json.loads(line)


@pytest.mark.skipif(not TRAIN.exists(), reason="training set not generated yet (scripts/simgen_v0b.py train)")
def test_the_whole_training_set_hits_no_held_out_cell_and_shares_no_request_with_any_test_set(holdout):
    train_hashes = set()
    n = 0
    for r in _scan(TRAIN):
        meta = r["meta"]
        assert meta["holdout"] == [] and _held_cells(holdout, meta) == [], r["id"]
        assert meta["group"] != "P1+P3" and SCENE_GROUP[meta["scene"]] != "P1+P3"
        train_hashes.add(request_hash(r))
        n += 1
    assert n == 50_000
    assert TESTS.exists()
    overlap = 0
    for path in sorted(TESTS.glob("T*.jsonl")):
        for r in _scan(path):
            overlap += request_hash(r) in train_hashes
    assert overlap == 0


# ----------------------------------------------------------------------------------------------------------------
# 시험 세트의 정의
# ----------------------------------------------------------------------------------------------------------------


@pytest.mark.skipif(not (TESTS / "manifest.json").exists(), reason="test sets not generated yet (scripts/simgen_v0b.py tests)")
def test_sealed_test_sets_match_their_definitions_and_manifest(holdout):
    manifest = json.loads((TESTS / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["holdout"]["sha256"] == holdout.sha256
    for name, entry in manifest["sets"].items():
        path = TESTS / entry["path"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == entry["sha256"], name
        lo, hi = entry["seed_range"]
        records = list(_scan(path))
        assert len(records) == entry["records"]
        for r in records:
            meta, seed = r["meta"], r["generator"]["seed"]
            assert lo <= seed <= hi, (name, r["id"])
            held = _held_cells(holdout, meta)
            assert sorted(held) == sorted(meta["holdout"]), (name, r["id"], held, meta["holdout"])
            if name == "T2":
                assert held and all(h.split(":")[0] in ("style", "setting", "pool") for h in held), r["id"]
            elif name == "T3":
                assert held == [r["generator"]["split"]["cell"]], r["id"]
            elif name == "T4":
                assert held == ["group:P1+P3"] and meta["group"] == "P1+P3", r["id"]
            else:
                assert held == [], (name, r["id"])
