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
#: Task v0.1a — 학습 문구 판 v01 + 사실 순서 섞기로 만든 학습 세트(같은 seed·같은 설정; 시험 세트는 v0b 그대로)
TRAIN_V01 = ARTIFACTS.parent / "v01" / "train-50k.jsonl"


@pytest.fixture(scope="module")
def holdout():
    return load_holdout()


def _held_cells(holdout, meta: dict) -> list[str]:
    """레코드의 meta(문체·상황·묶음·조합·T3 변수 값)만으로 다시 잰 빼 둔 칸 — 범위 칸은 **장면 목록과 무관하게**(수정 1 I1) 변수
    이름으로 적는다('range:<변수>'): 그 변수가 관측된 레코드면 어느 장면이든 자르기를 넘었나만 본다."""
    out = []
    if meta["style"] in holdout.styles:
        out.append("style:" + meta["style"])
    if meta["setting"] in holdout.heldout_settings(meta["domain"]):
        out.append(f"setting:{meta['domain']}[{meta['setting']}]")
    if meta["pool"] == holdout.paraphrase_pool:
        out.append("pool:" + meta["pool"])
    if meta["group"] in holdout.groups:
        out.append("group:" + meta["group"])
    for var in sorted({c.var for c in holdout.ranges}):
        cell = next(c for c in holdout.ranges if c.var == var)
        value = meta["holdout_vars"].get(var)
        if value is not None and cell.in_test(value, meta["body"]):
            out.append("range:" + var)
    return out


def _named(holdout, names: list[str]) -> list[str]:
    """meta.holdout의 이름을 _held_cells와 같은 꼴로(범위 칸 이름 → 'range:<변수>')."""
    cells = {c.name: c for c in holdout.ranges}
    return sorted({("range:" + cells[n].var) if n in cells else n for n in names})


# ----------------------------------------------------------------------------------------------------------------
# 생성 시점 거절
# ----------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("forced", [{"style": "log"}, {"setting": 4}, {"pool": "heldout"}, {"group": "P1+P3"}, {"scene": "p1p3_step_in_carrying_tall"}])
def test_the_training_split_cannot_build_a_held_out_cell(forced):
    with pytest.raises(HeldOutError):
        build_moment(default_config(), 777, "cafe", {"scene": "p1_step_in", **forced} if "scene" not in forced and "group" not in forced else forced)


def test_the_training_split_draws_range_variables_on_the_training_side(holdout):
    train = split_for("train", holdout)
    ranges = ranges_for(holdout, train, "p5_braking", "mobile_base")
    assert ranges["fill_reading"] == (-1e9, 0.92) and ranges["robot_speed"] == (-1e9, 1.25)
    assert ranges_for(holdout, train, "p6_blind_corner", "humanoid")["corner_lateral"] == (0.5, 1e9)
    t3 = split_for("T3", holdout, cell="p3_steep_ramp", train_ranges=True)
    ranges = ranges_for(holdout, t3, "p3_ramp", "mobile_base")
    assert ranges["ramp_angle"] == (7.0, 1e9) and ranges["load_aspect"] == (-1e9, 5.0) and ranges["robot_speed"] == (-1e9, 1.25)


def test_range_cuts_apply_in_every_scene_not_only_the_listed_ones(holdout):
    """수정 1 I1: 칸의 장면 목록은 T3가 뽑는 곳일 뿐 — 학습 쪽 자르기는 그 변수가 관측되는 모든 장면에 건다(로봇 속도가 P3+P5·P3·P6으로
    새던 것). 목록에 없는 장면에서도 좁히고, 넘는 관측은 거절된다."""
    from robo_jev.simgen.catalog import SCENE_BODIES, SCENES

    train = split_for("train", holdout)
    speed = holdout.cell("p1_fast_robot")
    for scene in SCENES:
        for body in SCENE_BODIES[scene]:
            assert ranges_for(holdout, train, scene, body)["robot_speed"] == (-1e9, speed.cut_for(body)), (scene, body)
    assert ranges_for(holdout, train, "p1p3_step_in_carrying_tall", "humanoid")["load_aspect"] == (-1e9, 5.0)   # T4 장면도 학습 쪽 범위로
    # T3가 로봇 속도 칸 하나를 강제하면 같은 물리량의 다른 칸은 좁히지 않는다
    t3 = split_for("T3", holdout, cell="p1_fast_robot", train_ranges=True)
    assert ranges_for(holdout, t3, "p1_step_in", "mobile_base")["robot_speed"] == (1.25, 1e9)


def test_cells_that_cut_one_variable_differently_or_share_scenes_are_rejected(tmp_path):
    """같은 변수를 쓰는 칸들은 같은 쪽·같은 자르기여야 하고 T3 장면 목록이 겹치면 안 된다 — 어긴 설정 파일은 읽히지 않는다."""
    from robo_jev.simgen.holdout import HOLDOUT_PATH

    text = HOLDOUT_PATH.read_text(encoding="utf-8")
    bad_cut = text.replace("- {name: p5_fast_robot, phenomenon: P5, var: robot_speed, side: above, cut: {mobile_base: 1.25,",
                           "- {name: p5_fast_robot, phenomenon: P5, var: robot_speed, side: above, cut: {mobile_base: 1.40,")
    assert bad_cut != text
    path = tmp_path / "bad-cut.yaml"
    path.write_text(bad_cut, encoding="utf-8")
    with pytest.raises(ValueError, match="cut the same variable differently"):
        load_holdout.__wrapped__(path)
    shared = text.replace("scenes: [p5_braking, p5_turning]}", "scenes: [p5_braking, p5_turning, p1_step_in]}")
    assert shared != text
    path = tmp_path / "shared.yaml"
    path.write_text(shared, encoding="utf-8")
    with pytest.raises(ValueError, match="share T3 scenes"):
        load_holdout.__wrapped__(path)


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
        assert "robot_speed" in meta["holdout_vars"]          # 로봇 속도는 모든 장면에서 적힌다(장면 목록과 무관하게 검사하려고)
    # 관측에서 다시 잰 칸도 0건(meta.holdout_vars가 관측의 값과 같다)
    cfg = default_config()
    from robo_jev.simgen.generate import record_moment

    for r in records[:300]:
        m = record_moment(cfg, r)            # 숫자 쌍의 변형은 받아들인 후보 번호까지(수정 1 N2)
        assert cells_hit(holdout, m.obs) == []


def _scan(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            yield json.loads(line)


@pytest.mark.parametrize("train", [TRAIN, TRAIN_V01], ids=["v0b", "v01"])
def test_the_whole_training_set_hits_no_held_out_cell_and_shares_no_request_with_any_test_set(holdout, train):
    if not train.exists():
        pytest.skip(f"training set not generated yet: {train} (scripts/simgen_v0b.py train)")
    train_hashes = set()
    n = 0
    for r in _scan(train):
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
            held = sorted(_held_cells(holdout, meta))
            assert held == _named(holdout, meta["holdout"]), (name, r["id"], held, meta["holdout"])
            if name == "T2":
                assert held and all(h.split(":")[0] in ("style", "setting", "pool") for h in held), r["id"]
            elif name == "T3":
                forced = r["generator"]["split"]["cell"]
                assert meta["holdout"] == [forced] and held == ["range:" + holdout.cell(forced).var], r["id"]
            elif name == "T4":
                assert held == ["group:P1+P3"] and meta["group"] == "P1+P3", r["id"]
            else:
                assert held == [], (name, r["id"])
