"""B2 측정 스크립트의 연기 검사 — 합성 장면이 계약을 지키고 집계 구조가 유지되는지.

토큰 수 자체는 실제 tokenizer의 몫이라(`artifacts/reports/tokens-b2.json`) 여기서는 공백
tokenizer로 구조만 본다.
"""

import copy
import functools
import importlib.util
import sys
from pathlib import Path

import pytest

from helpers import D0, D0_STREAMS, read_jsonl

from robo_jev.contracts import validate_record
from robo_jev.model.serialize import serialize_request
from robo_jev.model.tokenizer import WhitespaceTokenizer

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "measure_tokens.py"


@functools.lru_cache(maxsize=1)
def script():
    spec = importlib.util.spec_from_file_location("measure_tokens", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_scene_builders_stay_in_sync_with_test_harness():
    """스크립트의 obj/observation은 tests/test_harness.py의 복제다 — 영역 목록만 다르다.

    어긋나면 B2의 합성 장면이 하네스 검사의 관측과 달라진다. 둘을 바꿀 때는 같이 바꾼다.
    """
    import test_harness

    module = script()
    assert module.obj("o0", (300, 0, -80), colour="blue") == test_harness.obj("o0", (300, 0, -80), colour="blue")
    objects = module.scene(6)
    assert module.observation(copy.deepcopy(objects), tick=3, sim_time_ms=300) == test_harness.observation(
        objects=copy.deepcopy(objects), zones=copy.deepcopy(module.ZONES), tick=3, sim_time_ms=300
    )


def test_synthetic_scenes_are_valid_streams_with_the_requested_k():
    module = script()
    assert module.SYNTHETIC_CELLS == ((6, 12), (10, 12), (10, None))  # K=32는 계약 v0.3에 없다
    for n_objects, k_cap in module.SYNTHETIC_CELLS:
        record = module.synthetic_record(n_objects, k_cap, instruction_change=True)
        # 상한을 푼 셀은 프로파일(K≤32) 밖의 상계 참고값이라 그 셀만 넓힌 상한으로 검증한다 (계약이 넓어진 것이 아니다).
        validate_record(record, limits=None if k_cap is not None else module.UNCAPPED_LIMITS)
        if k_cap is None:
            with pytest.raises(ValueError, match="프로파일 상한을 넘는다"):
                validate_record(record)
        assert len(record["ticks"]) == 2
        k = len(record["ticks"][0]["request"]["candidates"]["q_main"])
        assert k > 12 if k_cap is None else k == k_cap  # 상한을 풀면 실행 가능한 후보 전부 (상계)
        assert [item["version"] for item in record["prefix"]["instructions"]] == [1, 2]
        assert record["ticks"][1]["request"]["state"]["goal"]["version"] == 2
    ten = module.synthetic_record(10, 12, instruction_change=False)
    assert len(ten["ticks"][0]["request"]["state"]["objects"]) == 10
    assert len(ten["prefix"]["instructions"]) == 1
    # 100틱 에피소드: 지시 변경은 요청한 틱부터, commitment는 지시의 대상×영역 파지에 틱마다 이어진다.
    long = module.synthetic_episode(6, 12, instruction_change=True, ticks=40, change_tick=17)
    validate_record(long)
    assert [tick["request"]["state"]["goal"]["version"] for tick in long["ticks"]] == [1] * 17 + [2] * 23
    assert [tick["request"]["commitment"]["held_ticks"] for tick in long["ticks"][1:]] == list(range(1, 40))
    # 외란은 기하 갱신 주기(200ms = 2틱) 뒤에 상태에 나타난다.
    assert long["ticks"][module.DISTURBANCE_TICK + 2]["request"]["state"]["objects"][3]["pose_mm"][0] == long["ticks"][0]["request"]["state"]["objects"][3]["pose_mm"][0] + 30
    occluded = long["ticks"][module.OCCLUSION_TICKS[0]]["request"]["state"]["objects"][1]
    assert occluded["visible_ratio"] == 0.2 and occluded["age_ms"] > 0


def test_measurements_have_the_documented_shape():
    module = script()
    tokenizer = WhitespaceTokenizer()
    streams = module.measure_streams(tokenizer, read_jsonl(D0_STREAMS)[:1])
    assert streams["episodes"][0]["ticks"] == 100
    assert set(streams["all_ticks"]["new_tokens"]) == {"n", "mean", "p50", "p95", "max", "min"}
    assert streams["q_main_block_by_k"]
    assert set(streams["episodes"][0]["sections_mean"]) <= set(module.SECTIONS)

    singles = module.measure_singles(tokenizer, read_jsonl(D0))
    assert singles["requests"] == 64
    assert set(singles["question_tokens_by_type"]) == {"boolean", "choice", "ordinal"}

    assert module.percentile([1, 2, 3, 4, 5], 0.5) == 3
    assert module.percentile([1, 2, 3, 4, 5], 0.95) == 5


def test_synthetic_cells_profile_first_typical_refresh_intro_and_change_ticks_and_judge_the_budget():
    """합성 셀은 100틱 에피소드의 틱 종류별 토큰(첫·통상·갱신·소개·지시 변경)과 구간별 평균을 내고, 10물체·K=12 셀을
    계약 v0.3 예산(p50 ≤ 500, p95 ≤ 800, 첫 틱 ≤ 1,200, 100틱 ≤ 60K)과 대조한다. 토큰 수 자체는 실제 tokenizer의 몫이다."""
    module = script()
    tokenizer = WhitespaceTokenizer()
    cells = module.measure_synthetic(tokenizer, ticks=31)
    assert [(c["objects"], c["k_cap"], c["instruction_change"]) for c in cells] == [
        (6, 12, False), (6, 12, True), (10, 12, False), (10, 12, True), (10, "none", False), (10, "none", True),
    ]
    cell = next(c for c in cells if (c["objects"], c["k_cap"], c["instruction_change"]) == (10, 12, True))
    assert cell["ticks"] == 31 and cell["change_tick"] == 15 and cell["k_actual"] == 12
    kinds = cell["by_kind"]
    assert kinds["first"]["ticks"] == 1 and kinds["intro"]["ticks"] == 1 and kinds["instruction_change"]["ticks"] == 1
    assert kinds["refresh"]["ticks"] == 2 and kinds["typical"]["ticks"] == 31 - 5  # 10·20 갱신, 30 소개, 15 지시 변경
    assert "objects_intro" in kinds["first"]["sections"] and "objects_intro" in kinds["intro"]["sections"]
    assert "objects_intro" not in kinds["typical"]["sections"]
    assert kinds["instruction_change"]["sections"]["instruction_change"] > 0
    assert cell["chunk_100_ticks"] is None  # 100틱이 안 된다
    assert kinds["typical"]["new_tokens"]["p50"] < kinds["refresh"]["new_tokens"]["mean"] < kinds["intro"]["new_tokens"]["mean"]
    report = {"synthetic": cells}
    verdict = module.verdicts(report)
    assert set(verdict["checks"]) == {"same_instruction", "instruction_change"}
    assert set(verdict["checks"]["same_instruction"]) == {"tick_p50", "tick_p95", "first_tick", "chunk_100_ticks", "typical_tick_p50"}
    assert verdict["checks"]["same_instruction"]["chunk_100_ticks"]["holds"] is False  # 100틱 구간이 없으면 성립하지 않는다
    assert verdict["budget"] == module.BUDGET == {"tick_p50": 500, "tick_p95": 800, "first_tick": 1200, "chunk_100_ticks": 60_000}
    assert verdict["holds"] is False
    hundred = module.measure_synthetic(tokenizer, ticks=100)
    assert all(c["chunk_100_ticks"] == sum(t for t in [c["all_ticks"]["mean"] * 100]) or c["chunk_100_ticks"] > 0 for c in hundred)
    assert module.verdicts({"synthetic": hundred})["checks"]["same_instruction"]["chunk_100_ticks"]["measured"] == hundred[2]["chunk_100_ticks"]


def test_real_episode_measurement_reports_distribution_chunks_and_sections():
    module = script()
    tokenizer = WhitespaceTokenizer()
    record = module.synthetic_episode(6, 12, instruction_change=True, ticks=120, change_tick=40)
    record["provenance"] = {"profile": "E1"}
    out = module.measure_episodes(tokenizer, [record])
    assert out["episodes"] == 1 and out["ticks"] == 120 and out["chunk_100_ticks"]["n"] == 1
    assert out["per_episode"][0]["profile"] == "E1" and out["per_episode"][0]["instruction_changes"] == 1
    assert out["first_tick"]["max"] == out["per_episode"][0]["first_tick"] > 0
    assert set(out["sections_mean"]) <= set(module.SECTIONS)
    verdict = module.verdicts({"synthetic": module.measure_synthetic(tokenizer, ticks=5), "episodes": out})
    assert set(verdict["checks"]["real_episodes"]) == {"tick_p50", "tick_p95", "first_tick_max", "chunk_100_ticks_max"}


def test_instruction_change_tokens_are_part_of_the_ticks_new_tokens():
    """지시 변경 조각은 그 틱의 토큰이라 new_tokens에 이미 들어 있고 따로도 센다."""
    module = script()
    record = module.synthetic_record(6, 12, instruction_change=True)
    out = serialize_request(record, WhitespaceTokenizer(), layout="stream_l1a")
    second = module.tick_breakdown(out, out["ticks"][1])
    assert second["instruction_change"] > 0
    assert second["new_tokens_without_instruction_change"] == second["new_tokens"] - second["instruction_change"]
