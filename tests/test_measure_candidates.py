"""`scripts/measure_candidates.py`(Task 2b G0a, native 경로 예비 선별)의 검사 — 가짜 runner로 CPU에서, 가중치·`transformers` 없이.

runner는 결정적 시간을 돌려주는 가짜다. 여기서 보는 것은 (1) 틱 자르기가 직렬화의 경계를 따르는지, (2) warm은 prefix +
30틱을 먼저 넣고 cold는 틱마다 prefix + 이력을 다시 넣는지, (3) 알려진 시간 벡터에서의 분위수·초과율, (4) 문턱값에서의
판정, (5) 보고서가 JSON이 되고 문서의 키를 갖는지, (6) `--path stream` 거절, (7) 2B·27B config의 메모리 산정식,
(8) `candidates.yaml`의 필수 키와 40-hex revision이다. 실제 지연은 GPU의 몫(`artifacts/reports/backbone-screen.json`)이다.
"""

import copy
import functools
import importlib.util
import json
import re
import statistics
import sys

import pytest
import torch
import yaml
from helpers import D0, D0_STREAMS, REPO, read_jsonl

from robo_jev.contracts import validate_record
from robo_jev.model.serialize import TOKEN_SERIALIZER_VERSION, serialize_request
from robo_jev.model.tokenizer import WhitespaceTokenizer

SCRIPT = REPO / "scripts" / "measure_candidates.py"
CANDIDATES = REPO / "configs" / "model" / "candidates.yaml"
GIB = 2**30


@functools.lru_cache(maxsize=1)
def script():
    spec = importlib.util.spec_from_file_location("measure_candidates", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@functools.lru_cache(maxsize=1)
def measure_tokens():
    spec = importlib.util.spec_from_file_location("measure_tokens", REPO / "scripts" / "measure_tokens.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault(spec.name, module)
    spec.loader.exec_module(module)
    return module


@functools.lru_cache(maxsize=1)
def entries() -> dict[str, dict]:
    data = yaml.safe_load(CANDIDATES.read_text(encoding="utf-8"))
    return {entry["id"]: entry for entry in data["candidates"]}


class FakeRunner:
    """결정적 시간을 내는 가짜 backbone — `forward`가 받은 id와 cache 길이를 적어 둔다. cache는 넣은 id의 목록이다."""

    def __init__(self, base_ms: float = 5.0, ms_per_token: float = 0.01) -> None:
        self.base_ms, self.ms_per_token = base_ms, ms_per_token
        self.calls: list[dict] = []
        self.loaded: list[dict] = []
        self.unloaded = 0
        self._last_ms = 0.0

    def load(self, config: dict):
        self.loaded.append(config)
        return {"config": config}

    def forward(self, handle, ids: list[int], cache):
        cache = [] if cache is None else cache
        self.calls.append({"ids": list(ids), "cache_before": len(cache)})
        cache = cache + list(ids)
        self._last_ms = self.base_ms + self.ms_per_token * len(ids)
        return torch.zeros(4), cache

    def sync(self) -> None:
        return None

    def event_ms(self) -> float:
        return self._last_ms

    def cache_length(self, cache) -> int:
        return 0 if cache is None else len(cache)

    def cache_bytes(self, cache) -> int:
        return 0 if cache is None else 2 * len(cache)

    def memory(self) -> dict:
        return {"peak_allocated_bytes": 123, "peak_reserved_bytes": 456, "allocated_bytes": 7}

    def describe(self, handle) -> dict:
        return {"class": "FakeModel", "params": 42, "weight_bytes": 84, "dtype": "fake", "attn_implementation": "fake"}

    def unload(self, handle) -> None:
        self.unloaded += 1

    def environment(self) -> dict:
        return {"gpu": "fake", "kernels": {"fla": False, "causal_conv1d": False}}


def stream_record(ticks: int) -> dict:
    record = copy.deepcopy(read_jsonl(D0_STREAMS)[0])
    record["ticks"] = record["ticks"][:ticks]
    return record


# --------------------------------------------------------------------------


def test_tick_slices_follow_the_serializer_boundaries_and_truncation_is_length_only():
    module = script()
    out = serialize_request(stream_record(12), WhitespaceTokenizer(), layout="stream_l1a")
    stream = module.stream_input(out, name="ep")
    assert stream.prefix == out["tokens"][: out["prefix_end"]] and len(stream.prefix) == out["prefix_end"]
    assert stream.ticks == 12
    for index, tick in enumerate(out["ticks"]):
        ids = stream.tick_ids(index)
        assert ids == out["tokens"][tick["start"] : tick["end"]]
        assert len(ids) == measure_tokens().tick_breakdown(out, tick)["new_tokens"]
    joined = list(stream.prefix)
    for index in range(stream.ticks):
        joined.extend(stream.tick_ids(index))
    assert joined == out["tokens"]  # prefix와 틱이 빈틈없이 이어진다 (결정 토큰까지 그 틱의 것)

    short = module.stream_input(out, name="ep", truncate_tick_tokens=5)
    assert short.prefix == stream.prefix
    assert all(short.tick_ids(index) == stream.tick_ids(index)[:5] for index in range(12))
    assert short.truncated_to == 5 and short.full_tick_tokens == [len(stream.tick_ids(i)) for i in range(12)]


def test_warm_condition_feeds_prefix_plus_history_in_the_cache_before_the_first_timed_tick():
    module = script()
    runner = FakeRunner()
    out = serialize_request(stream_record(12), WhitespaceTokenizer(), layout="stream_l1a")
    stream = module.stream_input(out, name="ep")
    result = module.measure_latency(
        "fake/model", [out], "stream_l1a", True, "native", runner=runner, handle=runner.load({}), history_ticks=4, warmup=2
    )
    prefill = runner.calls[0]
    expected = list(stream.prefix)
    for index in range(4):
        expected.extend(stream.tick_ids(index))
    assert prefill == {"ids": expected, "cache_before": 0}  # prefix + 이력 4틱이 한 번에 cache로
    assert [call["ids"] for call in runner.calls[1:]] == [stream.tick_ids(index) for index in range(4, 12)]
    assert [call["cache_before"] for call in runner.calls[1:]] == [
        len(expected) + sum(len(stream.tick_ids(j)) for j in range(4, index)) for index in range(4, 12)
    ]
    ticks = result["ticks"]
    assert [tick["tick"] for tick in ticks] == list(range(6, 12))  # 이력 4 + 예열 2 뒤부터 잰다
    assert [tick["new_tokens"] for tick in ticks] == [len(stream.tick_ids(index)) for index in range(6, 12)]
    assert [tick["cache_before"] for tick in ticks] == [call["cache_before"] for call in runner.calls[3:]]
    assert all(tick["cache_after"] == tick["cache_before"] + tick["new_tokens"] for tick in ticks)
    assert all(tick["model_ms"] == pytest.approx(5.0 + 0.01 * tick["new_tokens"]) for tick in ticks)
    assert all(tick["wall_ms"] >= 0 and tick["obs_apply_ms"] >= tick["wall_ms"] for tick in ticks)
    assert result["condition"] == "stream_warm" and result["summary"]["ticks"] == 6
    assert result["summary"]["cache_length"] == {"min": ticks[0]["cache_before"], "max": ticks[-1]["cache_after"]}
    assert result["episodes"][0]["cache_bytes_end"] == 2 * ticks[-1]["cache_after"]
    assert result["episodes"][0]["name"] == "stream-0"  # 직렬화 결과만 주면 에피소드 이름이 없다
    named = module.measure_latency(
        "fake/model", [stream_record(12)], "stream_l1a", True, "native", runner=FakeRunner(), handle=None or runner.load({}),
        tokenizer=WhitespaceTokenizer(), history_ticks=4, warmup=2,
    )
    assert named["episodes"][0]["name"] == "ep-d0-001" and {tick["episode"] for tick in named["ticks"]} == {"ep-d0-001"}


def test_cold_condition_recomputes_prefix_plus_the_window_with_an_empty_cache_every_tick():
    module = script()
    out = serialize_request(stream_record(12), WhitespaceTokenizer(), layout="stream_l1a")
    stream = module.stream_input(out, name="ep")
    runner = FakeRunner()
    result = module.measure_latency(
        "fake/model", [out], "stream_l1a", False, "native", runner=runner, handle=runner.load({}),
        history_ticks=4, warmup=1, cold_window_ticks=3, cold_ticks=2, cold_warmup=1,
    )
    # cold 예열 1틱(4번 틱) 뒤 5·6번 틱을 잰다 — warm과 같은 첫 측정 틱(이력 + 예열)이다
    assert [tick["tick"] for tick in result["ticks"]] == [5, 6]
    for call, index in zip(runner.calls, (4, 5, 6)):
        window = list(stream.prefix)
        for j in range(max(0, index - 3 + 1), index + 1):
            window.extend(stream.tick_ids(j))
        assert call == {"ids": window, "cache_before": 0}
    for tick in result["ticks"]:
        assert tick["cache_before"] == 0 and tick["new_tokens"] == len(runner.calls[tick["tick"] - 4]["ids"])
        assert tick["history_tokens"] == tick["new_tokens"] - len(stream.tick_ids(tick["tick"]))
    assert result["condition"] == "stream_cold"

    # 윈도우 0 = 이력 전부 (문자 그대로의 무상태 하한)
    runner = FakeRunner()
    module.measure_latency(
        "fake/model", [out], "stream_l1a", False, "native", runner=runner, handle=runner.load({}),
        history_ticks=4, warmup=0, cold_window_ticks=0, cold_ticks=1, cold_warmup=0,
    )
    everything = list(stream.prefix)
    for j in range(5):
        everything.extend(stream.tick_ids(j))
    assert runner.calls == [{"ids": everything, "cache_before": 0}]


def test_state_first_requests_are_forwarded_whole_without_a_cache():
    module = script()
    records = read_jsonl(D0)[:6]
    runner = FakeRunner()
    tokenizer = WhitespaceTokenizer()  # id는 처음 본 순서라 같은 인스턴스로 먼저 직렬화해 둔다
    outs = [serialize_request(record, tokenizer) for record in records]
    result = module.measure_latency("fake/model", records, "state_first", False, "native", runner=runner, handle=runner.load({}), warmup=2, tokenizer=tokenizer)
    assert [call["ids"] for call in runner.calls] == [outs[0]["tokens"], outs[1]["tokens"]] + [out["tokens"] for out in outs]
    assert all(call["cache_before"] == 0 for call in runner.calls)
    assert result["condition"] == "state_first" and result["summary"]["ticks"] == 6
    assert [tick["new_tokens"] for tick in result["ticks"]] == [len(out["tokens"]) for out in outs]
    assert [tick["request_id"] for tick in result["ticks"]] == [record["request"]["request_id"] for record in records]


def test_path_stream_is_rejected_as_stage_two(capsys):
    module = script()
    with pytest.raises(ValueError, match="2단계"):
        module.measure_latency("fake/model", [], "stream_l1a", True, "stream", runner=FakeRunner())
    with pytest.raises(ValueError, match="native"):
        module.measure_latency("fake/model", [], "stream_l1a", True, "fast", runner=FakeRunner())
    with pytest.raises(SystemExit) as excinfo:  # CLI도 같은 자리에서 거절한다 — 가중치·tokenizer를 읽기 전에
        module.main(["--path", "stream"])
    assert excinfo.value.code == 2 and "2단계" in capsys.readouterr().err


def test_summary_percentiles_miss_rate_and_over_budget_rates_on_a_known_vector():
    module = script()
    model = [float(v) for v in range(10, 210, 10)]  # 10 … 200, 20개
    ticks = [
        {"tick": i, "new_tokens": 100, "cache_before": 1000 + i, "cache_after": 1100 + i, "model_ms": m, "wall_ms": m + 1, "obs_apply_ms": m + 2}
        for i, m in enumerate(model)
    ]
    summary = module.summarize_ticks(ticks)
    pct = measure_tokens().percentile
    assert summary["ticks"] == 20
    assert summary["model_ms"] == {"mean": 105.0, "p50": pct(model, 0.5), "p95": pct(model, 0.95), "p99": pct(model, 0.99), "max": 200.0, "min": 10.0}
    assert summary["model_ms"]["p50"] == 110.0 and summary["model_ms"]["p95"] == 190.0 and summary["model_ms"]["p99"] == 200.0
    assert summary["wall_ms"]["p95"] == 191.0 and summary["obs_apply_ms"]["max"] == 202.0
    assert summary["deadline_miss_rate_100ms"] == pytest.approx(11 / 20)  # obs→apply 102 … 202 (> 100)
    assert summary["over_budget_rate_80ms"] == pytest.approx(12 / 20)  # model 90 … 200 (> 80)
    assert summary["over_budget_rate_150ms"] == pytest.approx(5 / 20)  # model 160 … 200
    assert summary["new_tokens"] == {"min": 100, "p50": 100.0, "max": 100} and summary["cache_length"] == {"min": 1000, "max": 1119}
    assert module.summarize_ticks([]) == {"ticks": 0}


def test_verdict_flags_flip_strictly_above_each_threshold_and_quote_their_numbers():
    module = script()

    def conditions(p95: float, miss: float) -> dict:
        summary = {"ticks": 35, "model_ms": {"p50": p95 / 2, "p95": p95, "p99": p95, "max": p95, "mean": p95, "min": 0.0}, "deadline_miss_rate_100ms": miss}
        return {"lower": {"stream_warm": {"summary": summary}}, "v03_target": {"stream_warm": {"summary": dict(summary, deadline_miss_rate_100ms=0.0)}}}

    verdict = module.verdicts(conditions(80.0, 0.05), max_miss_rate=0.05)
    assert verdict["lower"] == {
        "profile": "lower", "condition": "stream_warm", "ticks": 35, "p95_model_ms": 80.0, "deadline_miss_rate_100ms": 0.05,
        "fails_10hz": False, "fails_5hz": False, "deadline_fail": False, "passes": True,
        "window": module.window_reading([], prefix_tokens=None, tick_tokens_mean=None),  # 틱 기록이 없으면 전부 None, 죽지 않는다
        "text": verdict["lower"]["text"],
    }
    assert "80.0" in verdict["lower"]["text"] and "0.05" in verdict["lower"]["text"] and "passes" in verdict["lower"]["text"]
    assert "window-sized: fit n/a at n/a tokens, first-5 mean n/a; cache n/a" in verdict["lower"]["text"]
    tight = module.verdicts(conditions(80.1, 0.051), max_miss_rate=0.05)
    assert tight["lower"]["fails_10hz"] and not tight["lower"]["fails_5hz"] and tight["lower"]["deadline_fail"] and not tight["lower"]["passes"]
    assert tight["v03_target"]["fails_10hz"] and not tight["v03_target"]["deadline_fail"]
    assert "80.1" in tight["lower"]["text"] and "0.051" in tight["lower"]["text"] and "fails_10hz" in tight["lower"]["text"]
    slow = module.verdicts(conditions(150.1, 0.0), max_miss_rate=0.05)
    assert slow["lower"]["fails_10hz"] and slow["lower"]["fails_5hz"]
    missing = module.verdicts({"lower": {}}, max_miss_rate=0.05)
    assert missing["lower"]["passes"] is None and "not measured" in missing["lower"]["text"] and missing["v03_target"]["passes"] is None
    assert missing["lower"]["window"]["model_ms_at_window_cache"] is None and missing["lower"]["window"]["fit_n"] == 0


def test_verdict_window_reading_fits_model_ms_on_cache_and_quotes_it_without_moving_the_flags():
    """윈도우 크기 cache의 읽기(직선 맞춤·첫 5틱 평균·cache 범위)는 문장에 인용되지만 flag는 문자 그대로의 p95가 정한다."""
    module = script()

    def tick(index: int, cache: int, ms: float, episode: str = "ep") -> dict:
        return {"tick": 35 + index, "episode": episode, "new_tokens": 50, "cache_before": cache, "cache_after": cache + 50, "model_ms": ms, "wall_ms": ms + 0.2, "obs_apply_ms": ms + 0.5}

    ticks = [tick(i, 1000 + 100 * i, 10.0 + 0.02 * (1000 + 100 * i)) for i in range(10)]  # 정확히 직선: 10 + 0.02 × cache
    result = {"ticks": ticks, "summary": module.summarize_ticks(ticks)}
    profiles = {"lower": {"prefix_tokens": 100, "tick_tokens": {"mean": 50.0}}}
    verdict = module.verdicts({"lower": {"stream_warm": result}}, max_miss_rate=0.05, profiles=profiles)["lower"]
    window = verdict["window"]
    assert window["cache_before_range"] == [1000, 1900] and window["fit_n"] == 10 and window["fit"] == "ols model_ms ~ cache_before"
    assert window["early_ticks"] == 5 and window["early_ticks_mean_ms"] == pytest.approx(34.0)  # 첫 5틱: cache 1000…1400
    assert window["window_cache_tokens"] == 100 + 29 * 50 and "profile" in window["window_cache_basis"]
    assert window["model_ms_at_window_cache"] == pytest.approx(10.0 + 0.02 * 1550) and window["fit_slope_ms_per_1k_cache"] == pytest.approx(20.0)
    assert window["fit_intercept_ms"] == pytest.approx(10.0) and window["fit_r2"] == pytest.approx(1.0)
    assert "p95 model 48.0 ms (window-sized: fit 41.0 ms at 1,550 tokens, first-5 mean 34.0 ms; cache 1,000…1,900) vs 80 ms" in verdict["text"]
    assert verdict["passes"] is True and verdict["p95_model_ms"] == 48.0  # flag는 p95(문자 그대로: 10개 중 round(0.95·9) = 최대)로

    # 프로파일 정보가 없으면 측정 틱에서 되만든다 (min cache + 29 × 평균 새 토큰) — basis에 적힌다
    bare = module.verdicts({"lower": {"stream_warm": result}}, max_miss_rate=0.05)["lower"]["window"]
    assert bare["window_cache_tokens"] == 1000 + 29 * 50 and "min(cache_before)" in bare["window_cache_basis"]
    assert bare["model_ms_at_window_cache"] == pytest.approx(10.0 + 0.02 * 2450)

    # 에피소드가 여럿이면 첫 5틱은 에피소드마다 센다; 잡음이 있으면 R² < 1
    two = [tick(i, 1000 + 100 * i, 30.0 + 0.02 * (1000 + 100 * i) + (1.0 if i % 2 else -1.0), "a") for i in range(6)] + [tick(i, 1000 + 100 * i, 30.0, "b") for i in range(6)]
    both = module.window_reading(two, prefix_tokens=100, tick_tokens_mean=50.0)
    assert both["early_ticks_mean_ms"] == pytest.approx(statistics.fmean([t["model_ms"] for t in two[:5]] + [30.0] * 5)) and 0.0 < both["fit_r2"] < 1.0

    # 퇴화: 틱 하나 → 맞춤 없음(None), 첫 틱 평균은 그 값; cache가 한 값이어도 None; 죽지 않는다
    single = module.verdicts({"lower": {"stream_warm": {"ticks": ticks[:1], "summary": module.summarize_ticks(ticks[:1])}}}, max_miss_rate=0.05, profiles=profiles)["lower"]
    assert single["window"]["model_ms_at_window_cache"] is None and single["window"]["fit_n"] == 0 and single["window"]["fit_r2"] is None
    assert single["window"]["early_ticks_mean_ms"] == pytest.approx(30.0) and single["window"]["cache_before_range"] == [1000, 1000]
    assert "fit n/a at 1,550 tokens, first-5 mean 30.0 ms; cache 1,000…1,000" in single["text"] and single["passes"] is True
    flat = module.window_reading([tick(0, 1000, 30.0), tick(1, 1000, 32.0)], prefix_tokens=100, tick_tokens_mean=50.0)
    assert flat["model_ms_at_window_cache"] is None and flat["early_ticks_mean_ms"] == pytest.approx(31.0)


def test_memory_estimates_follow_the_stated_formula_and_match_docs05_for_the_27b_window():
    module = script()
    two = entries()["Qwen/Qwen3.5-2B"]
    window = module.window_state_bytes(two, tokens=10_000)
    assert window["kv_bytes"] == 2 * 6 * 2 * 256 * 10_000 * 2
    assert window["recurrent_bytes"] == 18 * 16 * 128 * 128 * 2
    assert window["conv_bytes"] == 18 * (2 * 16 * 128 + 16 * 128) * 4 * 2
    assert window["total_bytes"] == window["kv_bytes"] + window["recurrent_bytes"] + window["conv_bytes"]
    assert window["tokens"] == 10_000 and window["estimate"] is True and "formula" in window

    big = entries()["Qwen/Qwen3.8-27B"]
    docs05 = module.window_state_bytes(big, tokens=30 * 550)  # docs/05 §4: 27B, 30틱 × 550 token → KV ≈ 1.0 GiB
    assert abs(docs05["kv_bytes"] / GIB - 1.0) <= 0.2
    with_prefix = module.window_state_bytes(big, tokens=385 + 30 * 550)
    assert abs(with_prefix["kv_bytes"] / GIB - 1.0) <= 0.2

    activation = module.activation_bytes(two, tokens=50_000)
    assert activation["bytes"] == 50_000 * 2048 * 24 * 2 and activation["tokens"] == 50_000 and activation["estimate"] is True
    estimates = module.memory_estimates(two, prefix_tokens=385, tick_tokens_mean=1800.0, history_ticks=30)
    assert estimates["window_state"]["tokens"] == 385 + 30 * 1800
    assert estimates["training_chunk_activation"]["measured_profile"]["tokens"] == 385 + 100 * 1800
    assert estimates["training_chunk_activation"]["v03_target_50k"]["tokens"] == 50_000
    assert estimates["estimate"] is True


def test_synthetic_episodes_extend_to_n_ticks_with_the_instruction_change_on_the_requested_tick():
    module = script()
    record = module.synthetic_episode(6, 12, instruction_change=True, ticks=7, change_tick=4)
    validate_record(record)
    assert len(record["ticks"]) == 7
    assert [item["version"] for item in record["prefix"]["instructions"]] == [1, 2]
    assert [tick["request"]["state"]["goal"]["version"] for tick in record["ticks"]] == [1, 1, 1, 1, 2, 2, 2]
    assert [tick["t"] for tick in record["ticks"]] == list(range(7))
    assert len(record["ticks"][0]["request"]["candidates"]["q_main"]) == 12
    assert all(tick["request"]["commitment"] is None for tick in record["ticks"])  # K=12는 그 grasp 후보가 잘려 measure_tokens와 같이 commitment 없음
    plain = module.synthetic_episode(6, 12, instruction_change=False, ticks=3)
    assert len(plain["prefix"]["instructions"]) == 1 and len(plain["ticks"]) == 3
    held = module.synthetic_episode(6, 32, instruction_change=False, ticks=4)  # K=32에는 있어 commitment가 틱마다 이어진다
    assert [tick["request"]["commitment"]["held_ticks"] for tick in held["ticks"][1:]] == [1, 2, 3]
    assert held["ticks"][0]["request"]["commitment"] is None


def test_profiles_are_built_from_the_serializer_and_v03_is_the_upper_stream_truncated_to_500():
    module = script()
    tokenizer = WhitespaceTokenizer()
    profiles = module.build_profiles(tokenizer, ticks=6, names=("d0_streams", "lower", "v03_target"), d0_streams=[stream_record(6)], d0_singles=read_jsonl(D0)[:3], change_tick=3)
    assert set(profiles) == {"d0_streams", "lower", "v03_target", "state_first"}
    assert profiles["d0_streams"]["layout"] == "stream_l1a" and len(profiles["d0_streams"]["requests"]) == 1
    assert profiles["d0_streams"]["episodes"] == ["ep-d0-001"] and profiles["lower"]["episodes"] == ["ep-synth-6-12-same"]  # 레코드의 이름
    assert profiles["lower"]["objects"] == 6 and profiles["lower"]["k_cap"] == 12 and profiles["lower"]["instruction_change"] is False
    assert profiles["v03_target"]["truncate_tick_tokens"] == 500 and profiles["v03_target"]["derived_from"] == "upper"
    assert profiles["v03_target"]["requests"][0]["tokens"] == profiles["v03_target"]["source_requests"][0]["tokens"]  # 길이만 자른다
    assert profiles["state_first"]["layout"] == "state_first" and len(profiles["state_first"]["requests"]) == 3
    for name, profile in profiles.items():
        if profile["layout"] == "stream_l1a":
            assert all(request["serializer"] == TOKEN_SERIALIZER_VERSION for request in profile["requests"])
            assert profile["tick_tokens"]["n"] == 6 * len(profile["requests"]) and profile["prefix_tokens"] > 0


def test_screen_report_is_json_serialisable_and_carries_the_documented_keys(capsys, tmp_path):
    module = script()
    tokenizer = WhitespaceTokenizer()
    profiles = module.build_profiles(tokenizer, ticks=8, names=("lower", "v03_target"), d0_singles=read_jsonl(D0)[:3], change_tick=3)
    runner = FakeRunner()
    entry = dict(entries()["Qwen/Qwen3.5-2B"], path="/nowhere", manifest={"revision": "abc", "digest": "def", "verified": "sizes+small-files"})
    settings = module.Settings(ticks=8, warmup=1, history_ticks=3, cold_window_ticks=2, cold_ticks=2, max_miss_rate=0.05, dtype="bf16")
    checkpoint = tmp_path / "screen.partial.json"
    report = module.screen([entry], profiles, runner, settings=settings, tokenizer_info={"id": "fake", "revision": None, "sha256": "0" * 64}, checkpoint=checkpoint)
    assert json.loads(checkpoint.read_text(encoding="utf-8"))["partial"] is True and "partial" not in report
    text = json.dumps(report, ensure_ascii=False)
    assert json.loads(text)["serializer"] == TOKEN_SERIALIZER_VERSION
    assert {"task", "path", "environment", "tokenizer", "serializer", "settings", "profiles", "candidates", "budgets_ms", "notes", "generated_at", "finished_at"} <= set(report)
    assert report["finished_at"] >= report["generated_at"]  # generated_at = 시작, finished_at = 끝
    assert report["path"] == "native" and report["environment"]["gpu"] == "fake" and report["settings"]["max_miss_rate"] == 0.05
    candidate = report["candidates"]["Qwen/Qwen3.5-2B"]
    assert {"config", "manifest", "loaded", "conditions", "memory", "verdict"} <= set(candidate)
    assert set(candidate["conditions"]) == {"lower", "v03_target", "state_first"}
    assert set(candidate["conditions"]["lower"]) == {"stream_warm", "stream_cold"} and set(candidate["conditions"]["state_first"]) == {"state_first"}
    for profile, conditions in candidate["conditions"].items():
        for condition, result in conditions.items():
            assert result["summary"]["ticks"] == len(result["ticks"]) > 0
            assert {"p50", "p95", "p99", "max"} <= set(result["summary"]["model_ms"])
            assert {"deadline_miss_rate_100ms", "over_budget_rate_80ms", "over_budget_rate_150ms", "cache_length"} <= set(result["summary"])
    memory = candidate["memory"]
    assert memory["weight_bytes"] == 84 and memory["peak_allocated_bytes"] == 123 and memory["estimates"]["estimate"] is True
    assert memory["cache_bytes_measured"]["lower"]["stream_warm"] > 0
    assert memory["estimates"]["window_state"]["tokens"] == memory["estimates"]["window_state"]["prefix_tokens"] + 30 * memory["estimates"]["window_state"]["tick_tokens_mean"]
    assert set(candidate["verdict"]) == {"lower", "v03_target"} and candidate["verdict"]["lower"]["passes"] in (True, False)
    window = candidate["verdict"]["lower"]["window"]
    assert window["fit_n"] == len(candidate["conditions"]["lower"]["stream_warm"]["ticks"]) and window["cache_before_range"][0] > 0
    assert window["window_cache_tokens"] == round(profiles["lower"]["prefix_tokens"] + 29 * profiles["lower"]["tick_tokens"]["mean"])
    assert {tick["episode"] for tick in candidate["conditions"]["lower"]["stream_warm"]["ticks"]} == {"ep-synth-6-12-same"}
    assert runner.loaded[0]["path"] == "/nowhere" and runner.unloaded == 1
    module.print_table(report)
    out = capsys.readouterr().out
    assert "Qwen/Qwen3.5-2B" in out and "v03_target" in out and "stream_cold" in out and "verdict" in out


def test_from_report_rebuilds_summaries_and_verdicts_and_leaves_everything_else_untouched(tmp_path, capsys):
    """`--from-report`: GPU 없이 틱 기록에서 요약·판정만 다시 만든다. 나머지 블록과 틱 기록은 그대로, finished_at은 지어내지 않는다."""
    module = script()
    profiles = module.build_profiles(WhitespaceTokenizer(), ticks=8, names=("lower", "v03_target"), d0_singles=read_jsonl(D0)[:3], change_tick=3)
    entry = dict(entries()["Qwen/Qwen3.5-2B"], path="/nowhere", manifest={"revision": "abc", "digest": "def", "verified": "sizes+small-files"})
    settings = module.Settings(ticks=8, warmup=1, history_ticks=3, cold_window_ticks=2, cold_ticks=2, max_miss_rate=0.05, dtype="bf16")
    report = module.screen([entry], profiles, FakeRunner(), settings=settings, tokenizer_info={"id": "fake", "revision": None, "sha256": "0" * 64})
    del report["finished_at"]  # 첫 실측 JSON처럼 (그 필드가 생기기 전의 것)
    candidate = "Qwen/Qwen3.5-2B"
    report["candidates"][candidate]["conditions"]["lower"]["stream_warm"]["ticks"][0]["model_ms"] = 999.0  # 틱 하나를 바꾸면 요약이 따라와야 한다
    source = tmp_path / "screen.json"
    source.write_text(json.dumps(report, ensure_ascii=False), encoding="utf-8")
    out = tmp_path / "screen-resummarised.json"
    assert module.main(["--from-report", str(source), "--report", str(out)]) == 0
    rebuilt = json.loads(out.read_text(encoding="utf-8"))
    assert rebuilt["finished_at"] is None and rebuilt["resummarised_at"] >= rebuilt["generated_at"]
    assert rebuilt["resummarised_by"]["script_version"] == module.SCRIPT_VERSION and "commit" in rebuilt["resummarised_by"]
    for key in ("environment", "settings", "profiles", "tokenizer", "serializer", "generated_at", "command", "budgets_ms", "path", "task"):
        assert rebuilt[key] == report[key], key
    before, after = report["candidates"][candidate], rebuilt["candidates"][candidate]
    for key in ("config", "manifest", "path", "loaded", "memory"):
        assert after[key] == before[key], key
    for profile, conditions in before["conditions"].items():
        for condition, result in conditions.items():
            assert after["conditions"][profile][condition]["ticks"] == result["ticks"]
    assert after["conditions"]["lower"]["stream_warm"]["summary"]["model_ms"]["max"] == 999.0
    assert after["verdict"]["lower"]["window"]["fit_n"] == len(before["conditions"]["lower"]["stream_warm"]["ticks"])
    assert after["verdict"]["lower"]["window"]["window_cache_tokens"] == round(report["profiles"]["lower"]["prefix_tokens"] + 29 * report["profiles"]["lower"]["tick_tokens"]["mean"])
    assert "window-sized: fit" in after["verdict"]["lower"]["text"]
    assert any(note.startswith("finished_at is null") for note in rebuilt["notes"]) and any("--from-report" in note for note in rebuilt["notes"])
    assert rebuilt["notes"][: len(report["notes"])] == report["notes"]
    assert "v03_target" in capsys.readouterr().out
    # --report 없이 부르면 같은 파일에 쓰고, 두 번 돌려도 notes가 늘지 않는다
    assert module.main(["--from-report", str(out)]) == 0
    again = json.loads(out.read_text(encoding="utf-8"))
    assert len(again["notes"]) == len(rebuilt["notes"]) and again["candidates"][candidate]["verdict"] == after["verdict"]
    assert source.read_text(encoding="utf-8").find("resummarised_at") < 0  # 원본은 건드리지 않았다


def test_candidates_yaml_pins_four_candidates_with_required_keys_and_resolved_revisions():
    data = yaml.safe_load(CANDIDATES.read_text(encoding="utf-8"))
    required = {"id", "role", "revision", "license", "model_type", "layer_types", "full_attention_interval", "hidden_size", "intermediate_size", "num_attention_heads", "num_key_value_heads", "head_dim", "linear_attention", "vocab_size", "tie_word_embeddings", "params_total", "params_checkpoint"}
    assert [entry["id"] for entry in data["candidates"]] == ["Qwen/Qwen3.5-2B", "Qwen/Qwen3.5-4B", "Qwen/Qwen3.5-9B", "Qwen/Qwen3.8-27B"]
    for entry in data["candidates"]:
        assert required <= set(entry), entry["id"]
        assert re.fullmatch(r"[0-9a-f]{40}", entry["revision"]), entry["id"]
        assert entry["license"] == "apache-2.0" and entry["role"] in ("main", "separate", "reference")
        assert entry["model_type"] == "qwen3_5" and entry["vocab_size"] == 248320 and entry["full_attention_interval"] == 4
        linear, full = entry["layer_types"]["linear_attention"], entry["layer_types"]["full_attention"]
        assert linear == 3 * full and entry["head_dim"] == 256 and entry["linear_attention"]["conv_kernel_dim"] == 4
        if entry["role"] != "reference":  # 받은 후보는 header에서 센 파라미터 수가 있다
            assert isinstance(entry["params_total"], int) and 0 < entry["params_total"] <= entry["params_checkpoint"]
    assert data["tokenizer"] == "Qwen/Qwen3.8-27B"
