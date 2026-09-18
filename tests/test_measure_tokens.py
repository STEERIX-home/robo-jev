"""B2 측정 스크립트의 연기 검사 — 합성 장면이 계약을 지키고 집계 구조가 유지되는지.

토큰 수 자체는 실제 tokenizer의 몫이라(`artifacts/reports/tokens-b2.json`) 여기서는 공백
tokenizer로 구조만 본다.
"""

import functools
import importlib.util
import sys
from pathlib import Path

from helpers import D0, D0_STREAMS, read_jsonl

from robo_jev.contracts import validate_record
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


def test_synthetic_scenes_are_valid_streams_with_the_requested_k():
    module = script()
    for k_cap in (12, 32):
        record = module.synthetic_record(6, k_cap, instruction_change=True)
        validate_record(record)
        assert len(record["ticks"]) == 2
        assert len(record["ticks"][0]["request"]["candidates"]["q_main"]) == k_cap
        assert [item["version"] for item in record["prefix"]["instructions"]] == [1, 2]
        assert record["ticks"][1]["request"]["state"]["goal"]["version"] == 2
    ten = module.synthetic_record(10, 12, instruction_change=False)
    assert len(ten["ticks"][0]["request"]["state"]["objects"]) == 10
    assert len(ten["prefix"]["instructions"]) == 1


def test_measurements_have_the_documented_shape():
    module = script()
    tokenizer = WhitespaceTokenizer()
    streams = module.measure_streams(tokenizer, read_jsonl(D0_STREAMS)[:1])
    assert streams["episodes"][0]["ticks"] == 100
    assert set(streams["all_ticks"]["new_tokens"]) == {"n", "mean", "p50", "p95", "max", "min"}
    assert streams["q_main_block_by_k"]

    singles = module.measure_singles(tokenizer, read_jsonl(D0))
    assert singles["requests"] == 64
    assert set(singles["question_tokens_by_type"]) == {"boolean", "choice", "ordinal"}

    assert module.percentile([1, 2, 3, 4, 5], 0.5) == 3
    assert module.percentile([1, 2, 3, 4, 5], 0.95) == 5
