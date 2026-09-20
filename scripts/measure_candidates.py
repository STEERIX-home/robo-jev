"""Task 2b — backbone 후보의 지연·메모리 실측 (DGX Spark, BF16): **native 경로**(G0a, 1단계)와 **stream 경로**(G0b, 2단계).

후보(`configs/model/candidates.yaml`)마다 대표 로봇 스트림 입력에 틱당 지연·deadline 초과율·메모리를 재고 10Hz 예산(docs/03:
모델 시간 80ms, 5Hz는 150ms, obs→apply deadline 100ms)과 대조한다.

* ``--path native`` (G0a) — 공식 `transformers` forward(`AutoModelForCausalLM`, `use_cache=True`). 모델 자체의 hybrid cache는
  30틱 윈도우 없이 자라므로 예비 선별용이다.
* ``--path stream`` (G0b) — Task 4의 실제 스트림 경로: :class:`robo_jev.model.backbone_qwen.QwenStreamState`(정적 prefix +
  미리 할당한 윈도우 KV, fla·causal_conv1d 상태 전달, mask 없는 flash attention) + 결정 표지 10개의 **한 배치 분기 forward**
  + pointer readout + 타입별 출력. 틱마다 obs→apply = 직렬화 조각 자르기 + H2D + 틱 몸통 forward + 분기 배치 forward +
  readout + 질문별 typed 출력 + D2H이고, 모델 시간은 두 forward와 readout을 감싼 CUDA event다. warm(prefix + 30틱을 틱마다
  `advance`로 읽은 뒤 예열 5틱)만 잰다 — 상태 경로에 cold는 뜻이 없다. cache 길이는 30틱 뒤 **구성상** 일정하고(틱마다 적는다),
  메모리는 측정 틱 동안 자라면 안 된다(`allocated_growth_bytes`, `num_alloc_retries`). 지렛대(``--levers``)는 따로·함께 잰다:
  ``baseline``(정적 버퍼 + mask 없는 sdpa; 몸통과 분기가 두 forward), ``fused``(틱 몸통과 10개 결정 분기를 **한 forward**로 —
  가중치를 틱마다 한 번 읽는다; G0b가 고른 서빙 구성), ``graphs``(분기 배치 forward의 CUDA graph 재생), ``compile``(층의 dense
  부분 `torch.compile`), ``readout_bf16``(readout을 fp32 대신 bf16으로), ``all``(fused + compile + readout_bf16 — stream 경로의
  기본 서빙 구성; 추론 전용이고 학습은 합치지 않은 stream 경로·공식 P0 forward를 쓴다). ``--levers``를 안 주면 stream 경로는
  ``all``, native 경로는 ``baseline``이다. 판정(docs/03 §7-6): `upper`와 batch-0에서 p95 모델 시간 ≤ 80 ms **이고** obs→apply
  100 ms 초과율 ≤ 0.05이면 `passes_10hz`(5 Hz는 150 ms) — 외삽 없음.

**native cache는 30틱 윈도우 없이 자란다.** 모델 자체의 hybrid cache(linear 층의 recurrent/conv state + full 층의 KV)에
틱의 새 토큰만 이어 넣는데, docs/08 §3.1의 "정적 prefix + 최근 30틱" 윈도우는 2단계 경로의 속성이라 여기 없다. 그래서
full-attention의 KV는 에피소드 길이만큼 자라고 틱 지연도 그만큼 늘어난다 — 지연마다 **그 틱 직전의 cache 길이**를
같이 적는다. 윈도우가 있을 때의 지연은 cache 길이가 비슷한 틱의 값으로 읽는다.

입력은 전부 :func:`robo_jev.model.serialize.serialize_request` 가 만든 토큰이다(손으로 쓴 텍스트 없음). 틱의 새 토큰은
직렬화의 틱 경계(``ticks[i].start … end``, 결정 토큰 포함)이고 prefix는 ``tokens[:prefix_end]``다.

프로파일(`--profiles`):

* ``d0_streams`` — `tests/fixtures/d0_streams.jsonl`의 4 에피소드(각 `--ticks` 틱으로 자른다), 틱당 ≈800 토큰.
* ``lower`` / ``upper`` / ``instruction_change`` — `scripts/measure_tokens.py`의 장면 빌더로 만든 합성 에피소드
  (물체 6·K=12 / 물체 10·K=12 / 10·K=12 + 지시 변경 틱; 계약 v0.3 서식 `ts0.5`), `--ticks` 틱까지 `append_tick`으로
  늘린다. 지시 변경 틱은 이력·예열 뒤 첫 측정 틱(`--history + --warmup`)이라 측정 안에 든다. 계약 v0.3 이전에 있던
  ``v03_target``(옛 ``upper``를 틱마다 500토큰으로 자른 길이 대역)은 은퇴했다 — 이제 ``upper`` 자체가 v0.3 서식이다.
* ``state_first`` — `tests/fixtures/d0.jsonl` 64건의 단일 요청(L0, cache 없음). 프로파일이 아니라 조건이지만 같은 표에 둔다.
* ``batch0`` (``--episodes <dir>``) — 실제 로봇 에피소드 batch(`artifacts/datasets/d1-robot/batch-0`, 40편)의 첫 `--ticks` 틱.

조건(docs/06 bullet 3):

* ``stream_warm`` — prefix + `--history`(30)틱을 cache에 넣은 뒤(한 번의 prefill), `--warmup`(5)틱을 예열로 흘리고,
  그다음 틱부터 끝까지 틱마다 새 토큰만 넣어 잰다.
* ``stream_cold`` — cache 없이 틱마다 prefix + **최근 `--cold-window-ticks`(30)틱(현재 틱을 포함해서)**을 통째로 다시
  계산한다 — docs/08 §3.1의 윈도우 그대로다 (무상태 L0식 하한; 0이면 이력 전부). warm과 같은 틱에서 시작하고 `--cold-ticks`개만 잰다 — 한 번이 수십 초라 수를 제한한다.
* ``state_first`` — 요청마다 통째로 forward, cache 없음(`--warmup`건 예열 뒤 64건 전부).

틱마다 적는 것: 새 토큰 수, 직전 cache 길이, **모델 시간**(forward 앞뒤 `torch.cuda.Event`), forward의 벽시계 시간,
**obs→apply 시간** = 틱 토큰 자르기(직렬화·토큰화는 이미 끝났다) + H2D 복사 + forward + 마지막 위치 logits의 argmax(장치)
+ D2H. `torch.cuda.synchronize()`는 지표가 요구하는 경계에서만 한다.

지표(후보 × 프로파일 × 조건): 모델·벽시계·obs→apply ms의 p50/p95/p99/max(`measure_tokens.percentile` 규약),
`deadline_miss_rate_100ms`(obs→apply > 100ms), `over_budget_rate_80ms`·`_150ms`(모델 ms), 틱 수, cache 길이 범위.
메모리(후보): 실은 가중치 바이트, `torch.cuda.max_memory_allocated()` 최대, 조건 끝의 cache 바이트(실측), 그리고 config에서
**계산한** 추정 둘(측정이 아니다 — JSON도 그렇게 적는다): (1) 윈도우 상태(prefix + 30틱, 측정한 토큰 수) — full 층
KV ``2 × L_full × kv_heads × head_dim × tokens × 2B`` + linear 층 recurrent ``L_lin × v_heads × k_dim × v_dim × 2B`` +
conv ``L_lin × (2·k_heads·k_dim + v_heads·v_dim) × kernel × 2B``(길이와 무관); (2) 10초 학습 구간의 활성 자릿수
``tokens × hidden × layers × 2B``(측정 프로파일의 100틱과 v0.3 목표 50K). 판정(후보, `lower`와 `upper` 따로,
stream_warm 기준): `fails_10hz` = p95 모델 ms > 80, `fails_5hz` = > 150, `deadline_fail` = 초과율 > `--max-miss-rate`
(기본 0.05 — docs/03 §7-6이 수치를 파일럿에 맡겨 CLI 인자다). 품질(bullet 4)로는 떨어뜨리지 않는다.

runner는 주입할 수 있다(:class:`Runner`; 기본 :class:`TransformersRunner`). 검사는 결정적 시간을 내는 가짜를 넣고 CPU에서
가중치·`transformers` 없이 돈다 — `transformers`·`fla`는 실제 runner 안에서만 import한다.

실행: `uv run python scripts/measure_candidates.py --config configs/model/candidates.yaml --path native
       [--candidates id,…] [--include-reference] [--profiles …] [--ticks N] [--report artifacts/reports/backbone-screen.json]
       [--dtype bf16] [--max-miss-rate 0.05] [--cold-ticks N] [--verify-full]`
      `uv run python scripts/measure_candidates.py --path stream --candidates Qwen/Qwen3.5-2B,Qwen/Qwen3.5-4B
       --profiles lower,upper,instruction_change --episodes artifacts/datasets/d1-robot/batch-0 --ticks 70
       --levers baseline,fused,all --report artifacts/reports/backbone-stream.json`
가중치가 없으면 `scripts/fetch_backbone.py`를 가리키는 오류로 멈춘다.
"""

from __future__ import annotations

import argparse
import copy
from collections import Counter
import dataclasses
import datetime as dt
import importlib.util
import json
import math
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Protocol

import yaml

from robo_jev.contracts import QUESTION_SET_V0
from robo_jev.gpu import DEFAULT_FRACTION, limit_gpu_memory, memory_report
from robo_jev.data.episode import append_tick, new_episode
from robo_jev.harness.robot import candidate_id
from robo_jev.model.backbone import CANDIDATES_CONFIG, FETCH_SCRIPT, backbone_root, describe_backbone
from robo_jev.model.serialize import STREAM_FORMAT, TOKEN_SERIALIZER_VERSION, WINDOW_TICKS, serialize_request
from robo_jev.model.tokenizer import available_tokenizer, describe_tokenizer, load_tokenizer
from robo_jev.sampler import manifest_files

REPO = Path(__file__).resolve().parents[1]
D0 = REPO / "tests" / "fixtures" / "d0.jsonl"
D0_STREAMS = REPO / "tests" / "fixtures" / "d0_streams.jsonl"
DEFAULT_REPORT = REPO / "artifacts" / "reports" / "backbone-screen.json"
DEFAULT_STREAM_REPORT = REPO / "artifacts" / "reports" / "backbone-stream.json"

PATHS = ("native", "stream")
PROFILES = ("d0_streams", "lower", "upper", "instruction_change")
#: 실제 에피소드 프로파일 이름 (`--episodes`).
EPISODES_PROFILE = "batch0"
STREAM_CONDITIONS = ("stream_warm", "stream_cold")
#: stream 경로의 지렛대 (`--levers`). `all`은 셋을 함께.
LEVERS = ("baseline", "graphs", "compile", "readout_bf16", "fused", "all")
#: stream 경로의 기본 서빙 구성 (G0b 리뷰 1 권고 d): fused + dense compile + bf16 readout — 추론 전용.
DEFAULT_SERVING_LEVER = "all"
LEVER_SETTINGS = {
    "baseline": {"graphs": False, "compile": False, "readout_dtype": "float32", "fused": False},
    "graphs": {"graphs": True, "compile": False, "readout_dtype": "float32", "fused": False},
    "compile": {"graphs": False, "compile": True, "readout_dtype": "float32", "fused": False},
    "readout_bf16": {"graphs": False, "compile": False, "readout_dtype": "bfloat16", "fused": False},
    # fused: 틱 몸통과 결정 분기를 한 forward로 (가중치 한 번 읽기) — 분기 forward가 따로 없으므로 graphs와 겹치지 않는다
    "fused": {"graphs": False, "compile": False, "readout_dtype": "float32", "fused": True},
    "all": {"graphs": False, "compile": True, "readout_dtype": "bfloat16", "fused": True},
}
#: 판정에 쓰는 프로파일 (docs/03 §7-6, 브리프 S2.3): `upper`와 실제 batch-0.
GATE_PROFILES = ("upper", EPISODES_PROFILE)
#: docs/03 §"지연 예산에서 역산하는 backbone 선정": 모델 시간 80ms(10Hz)·150ms(5Hz), obs→apply deadline 100ms (5 Hz의 대응 deadline은 200ms).
BUDGET_MS = {"model_10hz": 80.0, "model_5hz": 150.0, "deadline": 100.0, "deadline_5hz": 200.0}
DEFAULT_MAX_MISS_RATE = 0.05
#: 계약 v0.3의 틱당 토큰 목표(HANDOFF 결정 1, p50) — 10초 학습 구간 50K 추정의 근거.
V03_TICK_TOKENS = 500
#: 합성 프로파일 (물체 수, K 상한, 지시 변경). K 상한은 계약 v0.3의 12 — `upper`는 물체 10개다.
SYNTHETIC = {
    "lower": (6, 12, False),
    "upper": (10, 12, False),
    "instruction_change": (10, 12, True),
}
DTYPES = {"bf16": "bfloat16"}
#: 이 스크립트의 버전 — `--from-report`가 다시 요약할 때 JSON에 적는다 (요약·판정의 정의가 바뀌면 올린다).
SCRIPT_VERSION = "g0b-2.0"
#: 판정의 "윈도우 크기 cache" 읽기: 첫 측정 틱 몇 개의 평균을 함께 적는다.
EARLY_TICKS = 5


def _script(name: str):
    """`scripts/`는 패키지가 아니라 경로로 불러온다 (검사가 measure_tokens를 부르는 방식과 같다)."""
    found = sys.modules.get(name)
    if found is not None and getattr(found, "__file__", None) == str(REPO / "scripts" / f"{name}.py"):
        return found
    spec = importlib.util.spec_from_file_location(name, REPO / "scripts" / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


#: 장면 빌더·`percentile`은 `scripts/measure_tokens.py`의 것을 그대로 쓴다 (세 번째 복제를 만들지 않는다).
measure_tokens = _script("measure_tokens")
percentile = measure_tokens.percentile
read_jsonl = measure_tokens.read_jsonl


# --------------------------------------------------------------------------
# runner 규약
# --------------------------------------------------------------------------


class Runner(Protocol):
    """backbone 하나를 싣고 forward를 돌리는 쪽. 검사는 결정적 시간의 가짜를, 실행은 :class:`TransformersRunner`를 넣는다."""

    def load(self, config: dict[str, Any]) -> Any: ...  # 후보 항목(+ `path`·`dtype`) → handle
    def forward(self, handle: Any, ids: list[int], cache: Any) -> tuple[Any, Any]: ...  # (마지막 위치 logits, cache)
    def sync(self) -> None: ...
    def event_ms(self) -> float: ...  # 직전 forward의 CUDA event 시간
    def cache_length(self, cache: Any) -> int: ...
    def cache_bytes(self, cache: Any) -> int: ...
    def memory(self) -> dict[str, Any]: ...
    def describe(self, handle: Any) -> dict[str, Any]: ...
    def unload(self, handle: Any) -> None: ...
    def environment(self) -> dict[str, Any]: ...


# --------------------------------------------------------------------------
# 직렬화 결과 → 틱 입력
# --------------------------------------------------------------------------


@dataclasses.dataclass
class StreamInput:
    """직렬화된 스트림의 prefix와 틱 경계. 틱의 id는 잴 때 자른다(`tick_ids` — obs→apply의 "자르기" 몫)."""

    name: str
    tokens: list[int]
    prefix_end: int
    bounds: list[tuple[int, int]]

    @property
    def prefix(self) -> list[int]:
        return self.tokens[: self.prefix_end]

    @property
    def ticks(self) -> int:
        return len(self.bounds)

    @property
    def full_tick_tokens(self) -> list[int]:
        return [end - start for start, end in self.bounds]

    def tick_ids(self, index: int) -> list[int]:
        start, end = self.bounds[index]
        return self.tokens[start:end]


def stream_input(out: dict[str, Any], *, name: str) -> StreamInput:
    if out.get("layout") != "stream_l1a":
        raise ValueError(f"stream_l1a 직렬화 결과가 필요하다 (받은 layout: {out.get('layout')!r})")
    return StreamInput(
        name=name,
        tokens=list(out["tokens"]),
        prefix_end=int(out["prefix_end"]),
        bounds=[(int(tick["start"]), int(tick["end"])) for tick in out["ticks"]],
    )


def _serialized(request: dict[str, Any], layout: str, tokenizer: Any) -> dict[str, Any]:
    """레코드면 직렬화하고, 이미 직렬화된 결과(`tokens`·`layout`)면 그대로 쓴다 (후보마다 다시 토큰화하지 않는다)."""
    if "tokens" in request and "layout" in request:
        if request["layout"] != layout:
            raise ValueError(f"layout이 다르다: 요청 {request['layout']!r}, 측정 {layout!r}")
        return request
    if tokenizer is None:
        tokenizer = load_tokenizer(_tokenizer_id())
    out = serialize_request(request, tokenizer, layout=layout)
    out["episode_id"] = request.get("episode_id")  # 직렬화 결과에는 없다 — 레코드에서 옮긴다
    return out


def _tokenizer_id() -> str:
    found = available_tokenizer()
    if found is None:
        raise FileNotFoundError("실제 tokenizer가 없다 — `uv run python scripts/fetch_tokenizer.py`로 받는다")
    return found[0]


# --------------------------------------------------------------------------
# 한 틱 재기
# --------------------------------------------------------------------------


def _timed_forward(runner: Runner, handle: Any, ids: list[int], cache: Any) -> tuple[dict[str, float], Any]:
    """obs→apply 한 번: (자르기는 호출자가 이미 시작한 시각 안에서) H2D + forward + sync + argmax + D2H."""
    started = time.perf_counter()
    logits, cache = runner.forward(handle, ids, cache)
    runner.sync()
    wall = time.perf_counter() - started
    token = int(logits.argmax(-1).item())  # 장치에서 argmax, D2H는 `.item()`
    return {"wall_ms": wall * 1e3, "model_ms": float(runner.event_ms()), "token": token}, cache


def run_stream_warm(
    runner: Runner, handle: Any, stream: StreamInput, *, history_ticks: int, warmup: int
) -> tuple[list[dict[str, Any]], Any]:
    first_timed = history_ticks + warmup
    if stream.ticks <= first_timed:
        raise ValueError(f"{stream.name}: 틱이 {stream.ticks}개라 이력 {history_ticks} + 예열 {warmup} 뒤에 잴 틱이 없다")
    prefill = list(stream.prefix)
    for index in range(history_ticks):
        prefill.extend(stream.tick_ids(index))
    _, cache = _timed_forward(runner, handle, prefill, None)  # prefix + 이력: 한 번에, 재지 않는다
    records: list[dict[str, Any]] = []
    for index in range(history_ticks, stream.ticks):
        started = time.perf_counter()
        ids = stream.tick_ids(index)
        before = runner.cache_length(cache)
        timing, cache = _timed_forward(runner, handle, ids, cache)
        obs_apply = (time.perf_counter() - started) * 1e3
        if index < first_timed:
            continue  # 예열
        records.append(
            {
                "tick": index,
                "new_tokens": len(ids),
                "cache_before": before,
                "cache_after": runner.cache_length(cache),
                "model_ms": timing["model_ms"],
                "wall_ms": timing["wall_ms"],
                "obs_apply_ms": obs_apply,
            }
        )
    return records, cache


def run_stream_cold(
    runner: Runner,
    handle: Any,
    stream: StreamInput,
    *,
    history_ticks: int,
    warmup: int,
    window_ticks: int,
    cold_ticks: int | None,
    cold_warmup: int = 1,
) -> tuple[list[dict[str, Any]], Any]:
    """warm과 같은 틱(`history_ticks + warmup`)부터 `cold_ticks`개를 잰다. 예열은 `cold_warmup`번(한 번이 수십 초라 따로 둔다)."""
    first_timed = history_ticks + warmup
    if stream.ticks <= first_timed:
        raise ValueError(f"{stream.name}: 틱이 {stream.ticks}개라 이력 {history_ticks} + 예열 {warmup} 뒤에 잴 틱이 없다")
    timed = list(range(first_timed, stream.ticks))
    if cold_ticks is not None:
        timed = timed[: max(0, cold_ticks)]
    records: list[dict[str, Any]] = []
    cache = None
    for index in [*range(max(0, first_timed - cold_warmup), first_timed), *timed]:
        started = time.perf_counter()
        lowest = 0 if window_ticks <= 0 else max(0, index - window_ticks + 1)
        ids = list(stream.prefix)
        for j in range(lowest, index + 1):
            ids.extend(stream.tick_ids(j))
        timing, cache = _timed_forward(runner, handle, ids, None)  # 빈 cache에서 통째로
        obs_apply = (time.perf_counter() - started) * 1e3
        if index < first_timed:
            continue
        records.append(
            {
                "tick": index,
                "new_tokens": len(ids),
                "history_tokens": len(ids) - len(stream.tick_ids(index)),
                "cache_before": 0,
                "cache_after": runner.cache_length(cache),
                "model_ms": timing["model_ms"],
                "wall_ms": timing["wall_ms"],
                "obs_apply_ms": obs_apply,
            }
        )
    return records, cache


def run_state_first(runner: Runner, handle: Any, outs: list[dict[str, Any]], *, warmup: int) -> list[dict[str, Any]]:
    for out in outs[:warmup]:
        _timed_forward(runner, handle, list(out["tokens"]), None)
    records: list[dict[str, Any]] = []
    for index, out in enumerate(outs):
        started = time.perf_counter()
        ids = list(out["tokens"])
        timing, cache = _timed_forward(runner, handle, ids, None)
        obs_apply = (time.perf_counter() - started) * 1e3
        records.append(
            {
                "tick": index,
                "request_id": out.get("request_id"),
                "new_tokens": len(ids),
                "cache_before": 0,
                "cache_after": runner.cache_length(cache),
                "model_ms": timing["model_ms"],
                "wall_ms": timing["wall_ms"],
                "obs_apply_ms": obs_apply,
            }
        )
    return records


# --------------------------------------------------------------------------
# 지표
# --------------------------------------------------------------------------


def _ms_summary(values: list[float]) -> dict[str, float]:
    return {
        "mean": round(statistics.fmean(values), 2),
        "p50": percentile(values, 0.5),
        "p95": percentile(values, 0.95),
        "p99": percentile(values, 0.99),
        "max": float(max(values)),
        "min": float(min(values)),
    }


def _rate(values: list[float], threshold: float) -> float:
    return sum(1 for value in values if value > threshold) / len(values)


def summarize_ticks(ticks: list[dict[str, Any]], *, budgets: dict[str, float] = BUDGET_MS) -> dict[str, Any]:
    """조건 하나의 틱 기록 → p50/p95/p99/max, deadline 초과율, 예산 초과율, 토큰·cache 범위."""
    if not ticks:
        return {"ticks": 0}
    model = [float(tick["model_ms"]) for tick in ticks]
    wall = [float(tick["wall_ms"]) for tick in ticks]
    obs_apply = [float(tick["obs_apply_ms"]) for tick in ticks]
    tokens = [int(tick["new_tokens"]) for tick in ticks]
    deadline, ten, five = budgets["deadline"], budgets["model_10hz"], budgets["model_5hz"]
    deadline_5hz = float(budgets.get("deadline_5hz", 2 * deadline))
    out = {
        "ticks": len(ticks),
        "new_tokens": {"min": min(tokens), "p50": percentile(tokens, 0.5), "max": max(tokens)},
        "cache_length": {"min": min(int(tick["cache_before"]) for tick in ticks), "max": max(int(tick["cache_after"]) for tick in ticks)},
        "model_ms": _ms_summary(model),
        "wall_ms": _ms_summary(wall),
        "obs_apply_ms": _ms_summary(obs_apply),
        f"deadline_miss_rate_{deadline:.0f}ms": _rate(obs_apply, deadline),
        f"deadline_miss_rate_{deadline_5hz:.0f}ms": _rate(obs_apply, deadline_5hz),
        f"over_budget_rate_{ten:.0f}ms": _rate(model, ten),
        f"over_budget_rate_{five:.0f}ms": _rate(model, five),
    }
    if all("window_ticks_in_cache" in tick for tick in ticks):  # stream 경로: 윈도우가 찬 뒤 cache의 틱 수는 일정하다
        full = [tick for tick in ticks if int(tick["window_ticks_in_cache"]) >= WINDOW_TICKS]
        out["cache_constant_after_window"] = bool(full) and len({int(tick["window_ticks_in_cache"]) for tick in full}) == 1
        out["window_tokens"] = {"min": min(int(tick["window_tokens"]) for tick in ticks), "max": max(int(tick["window_tokens"]) for tick in ticks)}
    if all("allocated_bytes" in tick for tick in ticks):
        allocated = [int(tick["allocated_bytes"]) for tick in ticks]
        out["allocated_bytes"] = {"first": allocated[0], "last": allocated[-1], "max": max(allocated)}
        out["allocated_growth_bytes"] = allocated[-1] - allocated[0]
    return out


# --------------------------------------------------------------------------
# docs/06 인터페이스
# --------------------------------------------------------------------------


def measure_latency(
    model_id: str,
    requests: list[dict[str, Any]],
    layout: str,
    warm_prefix: bool,
    path: str,
    *,
    runner: Runner | None = None,
    handle: Any = None,
    tokenizer: Any = None,
    history_ticks: int = WINDOW_TICKS,
    warmup: int = 5,
    cold_window_ticks: int = WINDOW_TICKS,
    cold_ticks: int | None = None,
    cold_warmup: int = 1,
    budgets: dict[str, float] = BUDGET_MS,
) -> dict[str, Any]:
    """docs/06 Task 2b의 인터페이스: 후보 `model_id`에 `requests`를 `layout`으로 넣어 틱당 지연을 잰다.

    `requests`는 레코드(직렬화한다) 또는 이미 직렬화된 결과다. `layout="stream_l1a"`면 `warm_prefix`가 조건을 고른다
    (True = ``stream_warm``, False = ``stream_cold``); `state_first`는 cache가 없어 `warm_prefix`를 쓰지 않는다. `path`는
    ``native``만 받는다 — ``stream``은 2단계(Task 4의 실제 스트림 경로)다. `handle`이 없으면 runner로 싣고 끝에 내린다.
    돌려주는 dict: ``ticks``(틱 기록), ``summary``(:func:`summarize_ticks`), ``episodes``(에피소드별 cache 끝 길이·바이트).
    """
    if path not in PATHS:
        raise ValueError(f"path: {list(PATHS)} 중 하나여야 한다 (받은 값: {path!r})")
    if layout not in ("stream_l1a", "state_first"):
        raise ValueError(f"layout: stream_l1a 또는 state_first (받은 값: {layout!r})")
    if path == "stream":
        if layout != "stream_l1a" or not warm_prefix:
            raise ValueError("path='stream'은 stream_l1a의 warm 조건만 잰다 — 상태 경로에 cold(무상태 재계산)는 뜻이 없다")
        runner = StreamRunner() if runner is None else runner
        owned = handle is None
        if owned:
            handle = runner.load(candidate_config(model_id))
        try:
            outs = [_serialized(request, layout, tokenizer) for request in requests]
            ticks: list[dict[str, Any]] = []
            episodes: list[dict[str, Any]] = []
            for number, out in enumerate(outs):
                name = str(out.get("episode_id") or f"stream-{number}")
                records, info = run_stream_path(runner, handle, out, name=name, history_ticks=history_ticks, warmup=warmup)
                for record in records:
                    record["episode"] = name
                ticks.extend(records)
                episodes.append({"name": name, "prefix_tokens": int(out["prefix_end"]), "timed_ticks": len(records), **info})
            return {
                "model_id": model_id, "layout": layout, "condition": "stream_warm", "path": path,
                "settings": {"history_ticks": history_ticks, "warmup": warmup, "levers": runner.describe(handle).get("levers")},
                "ticks": ticks, "summary": summarize_ticks(ticks, budgets=budgets), "episodes": episodes,
            }  # fmt: skip
        finally:
            if owned:
                runner.unload(handle)
    runner = TransformersRunner() if runner is None else runner
    owned = handle is None
    if owned:
        handle = runner.load(candidate_config(model_id))
    try:
        outs = [_serialized(request, layout, tokenizer) for request in requests]
        ticks: list[dict[str, Any]] = []
        episodes: list[dict[str, Any]] = []
        if layout == "state_first":
            condition = "state_first"
            ticks = run_state_first(runner, handle, outs, warmup=warmup)
        else:
            condition = "stream_warm" if warm_prefix else "stream_cold"
            for number, out in enumerate(outs):
                stream = stream_input(out, name=str(out.get("episode_id") or f"stream-{number}"))
                if warm_prefix:
                    records, cache = run_stream_warm(runner, handle, stream, history_ticks=history_ticks, warmup=warmup)
                else:
                    records, cache = run_stream_cold(
                        runner, handle, stream, history_ticks=history_ticks, warmup=warmup, window_ticks=cold_window_ticks,
                        cold_ticks=cold_ticks, cold_warmup=cold_warmup,
                    )
                for record in records:
                    record["episode"] = stream.name
                ticks.extend(records)
                episodes.append(
                    {
                        "name": stream.name,
                        "ticks_total": stream.ticks,
                        "prefix_tokens": stream.prefix_end,
                        "timed_ticks": len(records),
                        "cache_length_end": runner.cache_length(cache),
                        "cache_bytes_end": runner.cache_bytes(cache),
                    }
                )
                del cache
        return {
            "model_id": model_id,
            "layout": layout,
            "condition": condition,
            "path": path,
            "settings": {
                "history_ticks": history_ticks,
                "warmup": warmup,
                "cold_window_ticks": cold_window_ticks if condition == "stream_cold" else None,
                "cold_ticks": cold_ticks if condition == "stream_cold" else None,
                "cold_warmup": cold_warmup if condition == "stream_cold" else None,
            },
            "ticks": ticks,
            "summary": summarize_ticks(ticks, budgets=budgets),
            "episodes": episodes,
        }
    finally:
        if owned:
            runner.unload(handle)


def candidate_config(model_id: str, config: str | Path = CANDIDATES_CONFIG, root: str | Path | None = None, *, verify_full: bool = False) -> dict[str, Any]:
    """yaml의 후보 항목 + 받아 둔 가중치의 경로·manifest 정체 (없으면 fetch 스크립트를 가리키는 오류)."""
    data = yaml.safe_load((REPO / config if not Path(config).is_absolute() else Path(config)).read_text(encoding="utf-8"))
    entry = next((item for item in data["candidates"] if item.get("id") == model_id), None)
    if entry is None:
        raise KeyError(f"{model_id}: {config}에 없는 후보다")
    described = describe_backbone(model_id, root, full=verify_full)
    return {
        **copy.deepcopy(entry),
        "path": described["path"],
        "manifest": {key: described.get(key) for key in ("revision", "digest", "bytes_total", "verified", "fetched_at")},
    }


# --------------------------------------------------------------------------
# 메모리 산정 (config에서 계산 — 측정이 아니다)
# --------------------------------------------------------------------------

WINDOW_FORMULA = (
    "kv = 2 × L_full × num_key_value_heads × head_dim × tokens × 2 B; "
    "recurrent = L_linear × linear.num_value_heads × key_head_dim × value_head_dim × 2 B (길이 무관); "
    "conv = L_linear × (2·num_key_heads·key_head_dim + num_value_heads·value_head_dim) × conv_kernel_dim × 2 B (길이 무관). "
    "runtime은 recurrent state를 float32로 둔다(mamba_ssm_dtype) — 실측 cache 바이트가 그것을 보인다"
)
ACTIVATION_FORMULA = "tokens × hidden_size × num_layers × 2 B — 층당 hidden 하나만 센 자릿수. 실제 학습 활성은 MLP 중간·attention 작업공간·분기로 몇 배다"


def window_state_bytes(entry: dict[str, Any], *, tokens: int) -> dict[str, Any]:
    layers = entry["layer_types"]
    linear = entry["linear_attention"]
    kv = 2 * int(layers["full_attention"]) * int(entry["num_key_value_heads"]) * int(entry["head_dim"]) * int(tokens) * 2
    recurrent = int(layers["linear_attention"]) * int(linear["num_value_heads"]) * int(linear["key_head_dim"]) * int(linear["value_head_dim"]) * 2
    conv_dim = 2 * int(linear["num_key_heads"]) * int(linear["key_head_dim"]) + int(linear["num_value_heads"]) * int(linear["value_head_dim"])
    conv = int(layers["linear_attention"]) * conv_dim * int(linear["conv_kernel_dim"]) * 2
    return {
        "estimate": True,
        "tokens": int(tokens),
        "kv_bytes": kv,
        "recurrent_bytes": recurrent,
        "conv_bytes": conv,
        "total_bytes": kv + recurrent + conv,
        "formula": WINDOW_FORMULA,
    }


def activation_bytes(entry: dict[str, Any], *, tokens: int) -> dict[str, Any]:
    layers = int(entry["layer_types"]["full_attention"]) + int(entry["layer_types"]["linear_attention"])
    return {"estimate": True, "tokens": int(tokens), "bytes": int(tokens) * int(entry["hidden_size"]) * layers * 2, "formula": ACTIVATION_FORMULA}


def memory_estimates(entry: dict[str, Any], *, prefix_tokens: int, tick_tokens_mean: float, history_ticks: int = WINDOW_TICKS) -> dict[str, Any]:
    """docs/06 bullet 5: 윈도우 상태(prefix + 30틱)와 10초 학습 구간(100틱, 그리고 v0.3 목표 50K)의 활성 자릿수."""
    window_tokens = int(round(prefix_tokens + history_ticks * tick_tokens_mean))
    chunk_tokens = int(round(prefix_tokens + 100 * tick_tokens_mean))
    return {
        "estimate": True,
        "window_state": {
            **window_state_bytes(entry, tokens=window_tokens),
            "prefix_tokens": int(prefix_tokens),
            "tick_tokens_mean": tick_tokens_mean,
            "history_ticks": history_ticks,
        },
        "training_chunk_activation": {
            "measured_profile": activation_bytes(entry, tokens=chunk_tokens),
            "v03_50k": activation_bytes(entry, tokens=50_000),
        },
    }


# --------------------------------------------------------------------------
# 판정 (docs/06 1단계 탈락 규칙 — lower와 upper 따로)
# --------------------------------------------------------------------------


def _ols(xs: list[float], ys: list[float]) -> dict[str, float] | None:
    """최소제곱 직선 ``y = intercept + slope·x``와 R². 점이 둘 미만이거나 x가 한 값이면 None (판정이 죽지 않는다)."""
    if len(xs) < 2:
        return None
    mean_x, mean_y = statistics.fmean(xs), statistics.fmean(ys)
    sxx = sum((x - mean_x) ** 2 for x in xs)
    if sxx == 0:
        return None
    slope = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys)) / sxx
    intercept = mean_y - slope * mean_x
    syy = sum((y - mean_y) ** 2 for y in ys)
    residual = sum((y - (intercept + slope * x)) ** 2 for x, y in zip(xs, ys))
    r2 = 1.0 - residual / syy if syy > 0 else 1.0
    return {"intercept": intercept, "slope": slope, "r2": r2, "n": len(xs)}


def window_reading(
    ticks: list[dict[str, Any]],
    *,
    prefix_tokens: float | None,
    tick_tokens_mean: float | None,
    window_ticks: int = WINDOW_TICKS,
    early_ticks: int = EARLY_TICKS,
) -> dict[str, Any]:
    """native cache가 자라는 warm 측정에서 **윈도우 크기 cache**의 읽기 — 판정 문장에 문자 그대로의 p95 옆에 적는다.

    * ``cache_before_range`` — 측정 틱의 직전 cache 길이 최소…최대.
    * ``early_ticks_mean_ms`` — 에피소드마다 첫 ``early_ticks``개 측정 틱의 모델 ms 평균 (cache ≈ prefix + 35틱).
    * ``model_ms_at_window_cache`` — ``model_ms ~ cache_before``의 최소제곱 직선을 ``prefix + (window_ticks − 1) × 평균 틱
      토큰``(= docs/08 윈도우가 찼을 때의 cache)에서 읽은 값. 기울기(ms / 1K cached tokens)·절편·R²를 같이 적는다.
      점이 둘 미만이거나 cache 길이가 한 값이면 None이다.
    """
    if not ticks:
        return {
            "cache_before_range": None, "early_ticks": early_ticks, "early_ticks_mean_ms": None,
            "window_cache_tokens": None, "window_cache_basis": None, "model_ms_at_window_cache": None,
            "fit": "ols model_ms ~ cache_before", "fit_slope_ms_per_1k_cache": None, "fit_intercept_ms": None, "fit_r2": None, "fit_n": 0,
        }
    groups: dict[Any, list[dict[str, Any]]] = {}
    for tick in ticks:
        groups.setdefault(tick.get("episode"), []).append(tick)
    early = [float(tick["model_ms"]) for group in groups.values() for tick in group[:early_ticks]]
    caches = [float(tick["cache_before"]) for tick in ticks]
    fit = _ols(caches, [float(tick["model_ms"]) for tick in ticks])
    if prefix_tokens is None or tick_tokens_mean is None:
        # 프로파일 정보가 없으면 윈도우 크기 cache를 되만들지 않는다 — 첫 측정 틱의 cache는 이미 prefix + 이력 + 예열이라
        # `min(cache_before)`를 prefix로 읽으면 윈도우의 두 배를 "윈도우 크기"라 부르게 된다 (2b-G0a 리뷰 2 M11)
        window_cache = None
        basis = "no profile info (prefix_tokens/tick_tokens_mean missing) — window-sized reading not available"
    else:
        basis = "profile prefix_tokens + (WINDOW_TICKS − 1) × profile mean tick tokens"
        window_cache = int(round(float(prefix_tokens) + (window_ticks - 1) * float(tick_tokens_mean)))
    return {
        "cache_before_range": [int(min(caches)), int(max(caches))],
        "early_ticks": early_ticks,
        "early_ticks_mean_ms": round(statistics.fmean(early), 2),
        "window_cache_tokens": window_cache,
        "window_cache_basis": basis,
        "model_ms_at_window_cache": None if fit is None or window_cache is None else round(fit["intercept"] + fit["slope"] * window_cache, 2),
        "fit": "ols model_ms ~ cache_before",
        "fit_slope_ms_per_1k_cache": None if fit is None else round(fit["slope"] * 1000, 4),
        "fit_intercept_ms": None if fit is None else round(fit["intercept"], 2),
        "fit_r2": None if fit is None else round(fit["r2"], 4),
        "fit_n": 0 if fit is None else fit["n"],
    }


def verdicts(
    conditions: dict[str, Any],
    *,
    max_miss_rate: float,
    budgets: dict[str, float] = BUDGET_MS,
    profiles: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """docs/06 1단계 탈락 규칙 — `lower`와 `upper`에서 따로, stream_warm의 **문자 그대로의 p95**(native cache가 자란
    채로)로 flag를 정한다(보수적). 그 옆에 :func:`window_reading` 의 윈도우 크기 읽기(직선 맞춤·첫 5틱 평균·cache 범위)를
    적고 문장에도 인용한다 — 문자 그대로의 flag는 그것으로 바꾸지 않는다. 대신 **`passes_10hz_window`**를 따로 둔다:
    윈도우 크기 cache로 외삽한 모델 ms(`model_ms_at_window_cache`)와 첫 5틱 평균(`early_ticks_mean_ms`)이 **둘 다** 10 Hz
    예산 이하일 때 참 — G0b의 `stream` 경로(정적 윈도우)와 같은 잣대로 비교하기 위한 기계 판독 값이며, 외삽은 측정 cache 범위
    (15.7K~31K)보다 짧은 쪽(12.8K~13.2K)이라 R²와 함께 읽는다. 둘 중 하나라도 없으면 None."""
    deadline_key = f"deadline_miss_rate_{budgets['deadline']:.0f}ms"
    out: dict[str, Any] = {}
    for profile in ("lower", "upper"):
        result = (conditions.get(profile) or {}).get("stream_warm") or {}
        summary = result.get("summary")
        if not summary or not summary.get("ticks"):
            out[profile] = {
                "profile": profile, "condition": "stream_warm", "ticks": 0, "p95_model_ms": None, deadline_key: None,
                "fails_10hz": None, "fails_5hz": None, "deadline_fail": None, "passes": None, "passes_10hz_window": None,
                "window": window_reading([], prefix_tokens=None, tick_tokens_mean=None),
                "text": f"{profile}: not measured (no stream_warm ticks)",
            }
            continue
        info = (profiles or {}).get(profile) or {}
        window = window_reading(
            result.get("ticks") or [],
            prefix_tokens=info.get("prefix_tokens"),
            tick_tokens_mean=(info.get("tick_tokens") or {}).get("mean"),
        )
        p95 = float(summary["model_ms"]["p95"])
        miss = float(summary[deadline_key])
        fails_10hz = p95 > budgets["model_10hz"]
        fails_5hz = p95 > budgets["model_5hz"]
        deadline_fail = miss > max_miss_rate
        flags = [name for name, flag in (("fails_10hz", fails_10hz), ("fails_5hz", fails_5hz), ("deadline_fail", deadline_fail)) if flag]
        at_window = window["model_ms_at_window_cache"]
        early = window["early_ticks_mean_ms"]
        passes_10hz_window = None if at_window is None or early is None else bool(at_window <= budgets["model_10hz"] and early <= budgets["model_10hz"])
        cache_range = window["cache_before_range"]
        tokens = window["window_cache_tokens"]
        sized = (
            f"window-sized: fit {'n/a' if at_window is None else f'{at_window:.1f} ms'} at {'n/a' if tokens is None else f'{tokens:,}'} tokens, "
            f"first-{window['early_ticks']} mean {'n/a' if early is None else f'{early:.1f} ms'}; "
            f"cache {'n/a' if cache_range is None else f'{cache_range[0]:,}…{cache_range[1]:,}'}"
        )
        text = (
            f"{profile} (stream_warm, {summary['ticks']} ticks): p95 model {p95:.1f} ms ({sized}) vs {budgets['model_10hz']:.0f} ms (10 Hz) / "
            f"{budgets['model_5hz']:.0f} ms (5 Hz); obs→apply miss rate {miss:.3f} vs max {max_miss_rate:.3f} → "
            + (", ".join(flags) if flags else "passes")
            + (
                "; window-sized (extrapolated fit and first-5 mean both ≤ 10 Hz budget): "
                + ("n/a" if passes_10hz_window is None else ("passes_10hz_window" if passes_10hz_window else "fails_10hz_window"))
            )
        )
        out[profile] = {
            "profile": profile,
            "condition": "stream_warm",
            "ticks": int(summary["ticks"]),
            "p95_model_ms": p95,
            deadline_key: miss,
            "fails_10hz": fails_10hz,
            "fails_5hz": fails_5hz,
            "deadline_fail": deadline_fail,
            "passes": not flags,
            # 윈도우 크기 읽기의 기계 판독 (외삽 + 첫 5틱 평균 둘 다 ≤ 10 Hz 예산) — 문자 그대로의 flag와 별개.
            "passes_10hz_window": passes_10hz_window,
            "window": window,
            "text": text,
        }
    return out


def stream_verdicts(
    conditions: dict[str, Any], *, max_miss_rate: float, budgets: dict[str, float] = BUDGET_MS, profiles: tuple[str, ...] = GATE_PROFILES
) -> dict[str, Any]:
    """stream 경로의 판정 (docs/03 §7-6, 브리프 S2.3): 프로파일마다 stream_warm의 **문자 그대로의** p95 모델 ms와 obs→apply
    100 ms 초과율 — cache가 구성상 윈도우 크기라 외삽이 없다. `passes_10hz` = p95 ≤ 80 ms **이고** 초과율 ≤ max_miss_rate;
    `passes_5hz` = p95 ≤ 150 ms이고 5 Hz의 deadline(200 ms) 초과율 ≤ max_miss_rate. `overall`은 판정 프로파일 전부(upper·batch0)의 AND다."""
    deadline_key = f"deadline_miss_rate_{budgets['deadline']:.0f}ms"
    deadline_5hz = float(budgets.get("deadline_5hz", 2 * budgets["deadline"]))
    deadline_5hz_key = f"deadline_miss_rate_{deadline_5hz:.0f}ms"
    out: dict[str, Any] = {}
    for profile in profiles:
        result = (conditions.get(profile) or {}).get("stream_warm") or {}
        summary = result.get("summary")
        if not summary or not summary.get("ticks"):
            out[profile] = {"profile": profile, "ticks": 0, "p95_model_ms": None, deadline_key: None, "passes_10hz": None, "passes_5hz": None, "cache_constant": None, "text": f"{profile}: not measured (no stream_warm ticks)"}
            continue
        p95 = float(summary["model_ms"]["p95"])
        p50 = float(summary["model_ms"]["p50"])
        miss = float(summary[deadline_key])
        miss_5hz = float(summary.get(deadline_5hz_key, miss))
        passes_10hz = p95 <= budgets["model_10hz"] and miss <= max_miss_rate
        passes_5hz = p95 <= budgets["model_5hz"] and miss_5hz <= max_miss_rate
        cache = summary.get("cache_length") or {}
        constant = bool(summary.get("cache_constant_after_window", False))
        shortfall = max(0.0, p95 - budgets["model_10hz"])
        text = (
            f"{profile} (stream_warm, {summary['ticks']} ticks, cache {cache.get('min')}…{cache.get('max')} tokens, window-sized by construction): "
            f"model p50 {p50:.1f} / p95 {p95:.1f} ms vs {budgets['model_10hz']:.0f} ms (10 Hz) / {budgets['model_5hz']:.0f} ms (5 Hz); "
            f"obs→apply p95 {float(summary['obs_apply_ms']['p95']):.1f} ms, miss rate (> {budgets['deadline']:.0f} ms) {miss:.3f} vs max {max_miss_rate:.3f} → "
            + ("passes_10hz" if passes_10hz else f"fails_10hz (p95 {shortfall:.1f} ms over)" if p95 > budgets["model_10hz"] else "fails_10hz (miss rate)")
            + ", "
            + ("passes_5hz" if passes_5hz else "fails_5hz")
            + f" (5 Hz: p95 vs {budgets['model_5hz']:.0f} ms, miss rate > {deadline_5hz:.0f} ms {miss_5hz:.3f})"
        )
        out[profile] = {
            "profile": profile, "condition": "stream_warm", "ticks": int(summary["ticks"]), "p50_model_ms": p50, "p95_model_ms": p95,
            "p95_obs_apply_ms": float(summary["obs_apply_ms"]["p95"]), deadline_key: miss, deadline_5hz_key: miss_5hz, "passes_10hz": passes_10hz, "passes_5hz": passes_5hz,
            "shortfall_10hz_ms": round(shortfall, 2), "cache_constant": constant, "text": text,
        }  # fmt: skip
    measured = [v for v in out.values() if v["passes_10hz"] is not None]
    out["overall"] = {
        "profiles": list(profiles),
        "passes_10hz": bool(measured) and all(v["passes_10hz"] for v in measured),
        "passes_5hz": bool(measured) and all(v["passes_5hz"] for v in measured),
        "measured": [v["profile"] for v in measured],
    }
    return out


# --------------------------------------------------------------------------
# 입력 — 프로파일
# --------------------------------------------------------------------------


def synthetic_episode(n_objects: int, k_cap: int | None, *, instruction_change: bool, ticks: int, change_tick: int | None = None) -> dict[str, Any]:
    """`measure_tokens.synthetic_episode` 그대로 — 같은 장면·변화 일정 빌더, commitment는 틱마다 이어진다."""
    return measure_tokens.synthetic_episode(n_objects, k_cap, instruction_change=instruction_change, ticks=ticks, change_tick=change_tick)


def _tick_tokens_summary(outs: list[dict[str, Any]]) -> dict[str, Any]:
    values: list[int] = []
    for out in outs:
        stream = stream_input(out, name="")
        values.extend(len(stream.tick_ids(index)) for index in range(stream.ticks))
    return {"n": len(values), "mean": round(statistics.fmean(values), 1), "p50": percentile(values, 0.5), "p95": percentile(values, 0.95), "max": max(values), "min": min(values)}


def read_episodes(directory: str | Path, *, limit: int | None = None) -> list[dict[str, Any]]:
    """로봇 에피소드 batch(`manifest.json`의 `files` → `episodes/<id>/streams.jsonl`)의 스트림 레코드 (manifest 순서)."""
    root = Path(directory)
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"에피소드 batch의 manifest가 없다: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    records: list[dict[str, Any]] = []
    for name in manifest_files(manifest, manifest_path):
        for record in read_jsonl(root / name):
            if record.get("schema_version") == "stream-v0":
                records.append(record)
        if limit is not None and len(records) >= limit:
            break
    return records[:limit] if limit is not None else records


def build_profiles(
    tokenizer: Any,
    *,
    ticks: int,
    names: tuple[str, ...] = PROFILES,
    d0_streams: list[dict[str, Any]] | None = None,
    d0_singles: list[dict[str, Any]] | None = None,
    change_tick: int | None = None,
    episodes: list[dict[str, Any]] | None = None,
    include_state_first: bool = True,
) -> dict[str, dict[str, Any]]:
    """프로파일 이름 → {layout, requests(직렬화 결과), 설명, 토큰 요약}. `state_first`(D0 단일 64건)는 기본으로 들어간다.
    `episodes`(실제 batch의 스트림 레코드)를 주면 `batch0` 프로파일(각 에피소드의 첫 `ticks` 틱)이 더해진다."""
    unknown = sorted(set(names) - set(PROFILES) - {EPISODES_PROFILE})
    if unknown:
        raise ValueError(f"모르는 프로파일: {unknown} (아는 것: {list(PROFILES)} + {EPISODES_PROFILE})")
    profiles: dict[str, dict[str, Any]] = {}

    def stream_profile(name: str, records: list[dict[str, Any]], description: str, **extra: Any) -> dict[str, Any]:
        outs = []
        for record in records:
            out = serialize_request(record, tokenizer, layout="stream_l1a")
            out["episode_id"] = record.get("episode_id")  # 직렬화 결과에는 없다 — 레코드에서 옮긴다
            outs.append(out)
        return {
            "layout": "stream_l1a",
            "description": description,
            "requests": outs,
            "episodes": [out.get("episode_id") for out in outs],
            "ticks_per_episode": ticks,
            "prefix_tokens": int(outs[0]["prefix_end"]) if outs else 0,
            "tick_tokens": _tick_tokens_summary(outs),
            "format": STREAM_FORMAT,
            **extra,
        }

    if "d0_streams" in names:
        records = d0_streams if d0_streams is not None else read_jsonl(D0_STREAMS)
        cut = []
        for record in records:
            record = copy.deepcopy(record)
            record["ticks"] = record["ticks"][:ticks]
            cut.append(record)
        profiles["d0_streams"] = stream_profile("d0_streams", cut, f"tests/fixtures/d0_streams.jsonl {len(cut)} episodes, first {ticks} ticks each")

    synthetic_outs: dict[str, dict[str, Any]] = {}
    for name in [name for name in SYNTHETIC if name in names]:
        n_objects, k_cap, change = SYNTHETIC[name]
        record = synthetic_episode(n_objects, k_cap, instruction_change=change, ticks=ticks, change_tick=change_tick if change else None)
        synthetic_outs[name] = stream_profile(
            name, [record],
            f"synthetic {n_objects} objects, K={k_cap}, instruction change={'tick ' + str(change_tick if change_tick is not None else max(1, ticks - 1)) if change else 'no'}, {ticks} ticks",
            objects=n_objects, k_cap=k_cap, instruction_change=change,
        )
        profiles[name] = synthetic_outs[name]

    if episodes is not None:
        cut = []
        for record in episodes:
            record = copy.deepcopy(record)
            record["ticks"] = record["ticks"][:ticks]
            cut.append(record)
        profiles[EPISODES_PROFILE] = stream_profile(EPISODES_PROFILE, cut, f"real robot episodes ({len(cut)}), first {ticks} ticks each", real=True)

    if not include_state_first:
        return profiles
    singles = d0_singles if d0_singles is not None else read_jsonl(D0)
    outs = [serialize_request(record, tokenizer) for record in singles]
    totals = [len(out["tokens"]) for out in outs]
    profiles["state_first"] = {
        "layout": "state_first",
        "description": f"tests/fixtures/d0.jsonl {len(outs)} single requests (L0, no cache)",
        "requests": outs,
        "request_tokens": {"n": len(totals), "mean": round(statistics.fmean(totals), 1), "p50": percentile(totals, 0.5), "p95": percentile(totals, 0.95), "max": max(totals), "min": min(totals)},
    }
    return profiles


# --------------------------------------------------------------------------
# 후보 전부 재기 → 보고서
# --------------------------------------------------------------------------


@dataclasses.dataclass
class Settings:
    ticks: int = 40
    warmup: int = 5
    history_ticks: int = WINDOW_TICKS
    cold_window_ticks: int = WINDOW_TICKS
    cold_ticks: int | None = 8
    cold_warmup: int = 1
    max_miss_rate: float = DEFAULT_MAX_MISS_RATE
    dtype: str = "bf16"
    verify_full: bool = False
    budgets: dict[str, float] = dataclasses.field(default_factory=lambda: dict(BUDGET_MS))
    path: str = "native"
    levers: list[str] = dataclasses.field(default_factory=lambda: ["baseline"])
    episodes_dir: str | None = None
    episodes_limit: int | None = None
    readout_rank: int = 64
    checkpoint: str | None = None


STREAM_NOTES = [
    "stream path = the real streaming path of Task 4 on the pretrained backbone (robo_jev.model.backbone_qwen.QwenStreamState): static prefix KV + a preallocated window KV buffer (no torch.cat growth; eviction by tick), DeltaNet recurrent/conv state carried through fla chunk_gated_delta_rule(initial_state)/causal_conv1d, full attention without a materialised mask (varlen flash over prefix / window / new-causal parts merged by logsumexp), the 10 decision markers as one batched single-token forward with transient reads of the common state, pointer readout and typed outputs.",
    "per tick, obs→apply = serialize slice + H2D + tick-body forward + batched decision forward + pointer readout + typed outputs for the posed questions + D2H; model_ms = CUDA events around the two forwards and the readout. Only warm (prefix + 30 history ticks fed tick by tick, then `warmup` ticks) is measured — cold is meaningless for a stateful path.",
    "cache_length is the number of tokens the attention layers can see (prefix + the last 30 ticks); after the window is full it is window-sized by construction (`cache_constant_after_window`), so no extrapolation is needed. allocated_bytes per tick and allocated_growth_bytes over the timed ticks show that memory does not grow; peak_allocated is reset after the warm-up.",
    "levers: baseline (static buffers + mask-free sdpa; tick body and the batched decision forward as two forwards), graphs (CUDA-graph replay of the batched decision forward — the tick body stays eager), compile (torch.compile of the dense parts of each layer), readout_bf16 (readout in bf16 instead of fp32), fused (tick body + the 10 decision branches in one forward, so every layer's weights are read once per tick); `all` = fused + compile + readout_bf16 (graphs cannot combine with fused: there is no separate decision forward). verdict (docs/03 §7-6) reads the first lever given; lever_verdicts the others.",
    "verdicts: passes_10hz = p95 model ms ≤ 80 and obs→apply 100 ms miss rate ≤ --max-miss-rate, on `upper` and on batch0 separately (overall = both); passes_5hz = p95 ≤ 150 ms and the 200 ms (5 Hz deadline) miss rate ≤ --max-miss-rate. The readout is untrained unless --checkpoint is given; latency does not depend on its values.",
]


NOTES = [
    "native path = official transformers AutoModelForCausalLM forward (BF16, use_cache=True); the model's own hybrid cache grows without the 30-tick window (a stage-2 property), so every latency carries the cache length before that tick.",
    "stream_cold recomputes prefix + the most recent `cold_window_ticks` ticks, the current one included (the docs/08 window), with an empty cache (stateless bound; 0 = full history) and times only `cold_ticks` ticks.",
    "stream profiles are serialized with the contract v0.3 format (serializer ts0.5: short names, object intro/dynamic split, delta ticks, trimmed candidate lines); the pre-v0.3 `v03_target` length-only stand-in is retired.",
    "memory.estimates are computed from config.json (formulas in the entries), not measured; memory.peak_allocated_bytes and cache_bytes_measured are measured.",
    "verdicts follow the docs/06 stage-1 drop rule on stream_warm p95 model ms (80 ms = 10 Hz, 150 ms = 5 Hz) and the obs→apply 100 ms miss rate against --max-miss-rate; quality (bullet 4) is out of scope.",
]


def screen(
    entries: list[dict[str, Any]],
    profiles: dict[str, dict[str, Any]],
    runner: Runner,
    *,
    settings: Settings,
    tokenizer_info: dict[str, Any],
    command: str | None = None,
    checkpoint: Path | None = None,
) -> dict[str, Any]:
    """후보 전부를 싣고 프로파일 × 조건을 잰 뒤 보고서 dict를 만든다 (JSON으로 그대로 쓴다).

    `checkpoint`를 주면 후보 하나가 끝날 때마다 그때까지의 보고서를 거기 쓴다 — 뒤 후보가 죽어도 앞 결과는 남는다.
    """
    if settings.path == "stream":
        return screen_stream(entries, profiles, runner, settings=settings, tokenizer_info=tokenizer_info, command=command, checkpoint=checkpoint)
    report: dict[str, Any] = {
        "task": "2b-g0a",
        "path": "native",
        "command": command,
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "environment": runner.environment(),
        "tokenizer": tokenizer_info,
        "serializer": TOKEN_SERIALIZER_VERSION,
        "settings": {**dataclasses.asdict(settings), "window_ticks_contract": WINDOW_TICKS, "v03_tick_tokens": V03_TICK_TOKENS},
        "budgets_ms": dict(settings.budgets),
        "profiles": {
            name: {key: value for key, value in profile.items() if key not in ("requests", "source_requests")}
            for name, profile in profiles.items()
        },
        "candidates": {},
        "notes": list(NOTES),
    }
    stream_names = [name for name, profile in profiles.items() if profile["layout"] == "stream_l1a"]
    for entry in entries:
        model_id = entry["id"]
        print(f"[G0a] {model_id}: loading ({entry.get('path')})", file=sys.stderr, flush=True)
        started = time.perf_counter()
        handle = runner.load({**{key: value for key, value in entry.items() if key != "manifest"}, "dtype": settings.dtype})
        loaded = {**runner.describe(handle), "load_seconds": round(time.perf_counter() - started, 1)}
        if report["environment"].get("attention_implementation") is None:
            report["environment"]["attention_implementation"] = loaded.get("attn_implementation")
        conditions: dict[str, dict[str, Any]] = {}
        cache_bytes: dict[str, dict[str, int]] = {}
        try:
            for name in stream_names:
                profile = profiles[name]
                conditions[name] = {}
                cache_bytes[name] = {}
                for condition in STREAM_CONDITIONS:
                    print(f"[G0a] {model_id}: {name} / {condition}", file=sys.stderr, flush=True)
                    result = measure_latency(
                        model_id, profile["requests"], "stream_l1a", condition == "stream_warm", "native",
                        runner=runner, handle=handle, history_ticks=settings.history_ticks, warmup=settings.warmup,
                        cold_window_ticks=settings.cold_window_ticks, cold_ticks=settings.cold_ticks, cold_warmup=settings.cold_warmup,
                        budgets=settings.budgets,
                    )
                    conditions[name][condition] = result
                    cache_bytes[name][condition] = max((episode["cache_bytes_end"] for episode in result["episodes"]), default=0)
                    _print_progress(model_id, name, condition, result["summary"])
            if "state_first" in profiles:
                print(f"[G0a] {model_id}: state_first", file=sys.stderr, flush=True)
                result = measure_latency(
                    model_id, profiles["state_first"]["requests"], "state_first", False, "native",
                    runner=runner, handle=handle, warmup=settings.warmup, budgets=settings.budgets,
                )
                conditions["state_first"] = {"state_first": result}
                _print_progress(model_id, "state_first", "state_first", result["summary"])
            memory = runner.memory()
        finally:
            runner.unload(handle)

        reference = "lower" if "lower" in stream_names else (stream_names[0] if stream_names else None)
        estimates: dict[str, Any] = {"estimate": True, "reference_profile": reference, "by_profile": {}}
        for name in stream_names:
            profile = profiles[name]
            estimates["by_profile"][name] = memory_estimates(entry, prefix_tokens=profile["prefix_tokens"], tick_tokens_mean=profile["tick_tokens"]["mean"])
        if reference is not None:
            estimates.update({key: value for key, value in estimates["by_profile"][reference].items() if key != "estimate"})
        report["candidates"][model_id] = {
            "config": {key: value for key, value in entry.items() if key not in ("manifest", "path")},
            "manifest": entry.get("manifest"),
            "path": entry.get("path"),
            "loaded": loaded,
            "conditions": conditions,
            "memory": {
                "weight_bytes": loaded.get("weight_bytes"),
                **memory,
                "cache_bytes_measured": cache_bytes,
                "estimates": estimates,
            },
            "verdict": verdicts(conditions, max_miss_rate=settings.max_miss_rate, budgets=settings.budgets, profiles=profiles),
        }
        if checkpoint is not None:
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            checkpoint.write_text(json.dumps({**report, "partial": True}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report["finished_at"] = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")  # `generated_at`은 시작 시각이다
    return report


def screen_stream(
    entries: list[dict[str, Any]],
    profiles: dict[str, dict[str, Any]],
    runner: Any,
    *,
    settings: Settings,
    tokenizer_info: dict[str, Any],
    command: str | None = None,
    checkpoint: Path | None = None,
) -> dict[str, Any]:
    """stream 경로: 후보 × 지렛대 × 프로파일(stream_warm)을 재고 보고서 dict를 만든다. 첫 지렛대(`baseline`)가 `conditions`·`verdict`,
    나머지는 `levers[이름]`·`lever_verdicts[이름]`에 든다. 후보 하나가 끝날 때마다 `checkpoint`에 쓴다."""
    report: dict[str, Any] = {
        "task": "2b-g0b",
        "path": "stream",
        "command": command,
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "environment": runner.environment(),
        "tokenizer": tokenizer_info,
        "serializer": TOKEN_SERIALIZER_VERSION,
        "settings": {**dataclasses.asdict(settings), "window_ticks_contract": WINDOW_TICKS, "v03_tick_tokens": V03_TICK_TOKENS},
        "budgets_ms": dict(settings.budgets),
        "profiles": {name: {key: value for key, value in profile.items() if key not in ("requests", "source_requests")} for name, profile in profiles.items()},
        "candidates": {},
        "notes": list(STREAM_NOTES),
    }
    stream_names = [name for name, profile in profiles.items() if profile["layout"] == "stream_l1a"]
    levers = list(settings.levers) or ["baseline"]
    for entry in entries:
        model_id = entry["id"]
        candidate: dict[str, Any] = {
            "config": {key: value for key, value in entry.items() if key not in ("manifest", "path")},
            "manifest": entry.get("manifest"), "path": entry.get("path"), "levers": {}, "lever_verdicts": {}, "memory": {},
        }  # fmt: skip
        for lever in levers:
            lever_settings = dict(LEVER_SETTINGS[lever])
            print(f"[G0b] {model_id}: loading ({entry.get('path')}) lever={lever} {lever_settings}", file=sys.stderr, flush=True)
            started = time.perf_counter()
            handle = runner.load({**{key: value for key, value in entry.items() if key != "manifest"}, "dtype": settings.dtype, "lever": lever_settings, "readout_rank": settings.readout_rank, "checkpoint": settings.checkpoint, "tokenizer_sha256": tokenizer_info.get("sha256")})
            loaded = {**runner.describe(handle), "load_seconds": round(time.perf_counter() - started, 1)}
            if report["environment"].get("attention_implementation") is None:
                report["environment"]["attention_implementation"] = loaded.get("attention_backend")
            conditions: dict[str, dict[str, Any]] = {}
            try:
                for name in stream_names:
                    print(f"[G0b] {model_id}/{lever}: {name} / stream_warm", file=sys.stderr, flush=True)
                    result = measure_latency(
                        model_id, profiles[name]["requests"], "stream_l1a", True, "stream",
                        runner=runner, handle=handle, history_ticks=settings.history_ticks, warmup=settings.warmup, budgets=settings.budgets,
                    )  # fmt: skip
                    conditions[name] = {"stream_warm": result}
                    _print_progress(model_id, f"{name}/{lever}", "stream_warm", result["summary"])
                memory = runner.memory()
                loaded.update({key: value for key, value in runner.describe(handle).items() if key in ("attention_backend", "kernels")})  # 첫 forward 뒤에 정해진다
                if loaded.get("attention_backend"):
                    report["environment"]["attention_implementation"] = loaded["attention_backend"]
            finally:
                runner.unload(handle)
            verdict = stream_verdicts(conditions, max_miss_rate=settings.max_miss_rate, budgets=settings.budgets)
            block = {"settings": lever_settings, "loaded": loaded, "conditions": conditions, "memory": memory, "verdict": verdict}
            if lever == levers[0]:
                candidate.update({"loaded": loaded, "conditions": conditions, "memory": {"weight_bytes": loaded.get("weight_bytes"), **memory}, "verdict": verdict, "baseline_lever": lever})
            candidate["levers"][lever] = block
            candidate["lever_verdicts"][lever] = verdict
            report["candidates"][model_id] = candidate
            if checkpoint is not None:
                checkpoint.parent.mkdir(parents=True, exist_ok=True)
                checkpoint.write_text(json.dumps({**report, "partial": True}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report["finished_at"] = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    return report


def _print_progress(model_id: str, profile: str, condition: str, summary: dict[str, Any]) -> None:
    if not summary.get("ticks"):
        return
    model = summary["model_ms"]
    print(
        f"[G0a] {model_id}: {profile}/{condition}: {summary['ticks']} ticks, model p50={model['p50']:.1f} p95={model['p95']:.1f} max={model['max']:.1f} ms, "
        f"cache {summary['cache_length']['min']}…{summary['cache_length']['max']}",
        file=sys.stderr,
        flush=True,
    )


# --------------------------------------------------------------------------
# 다시 요약 — 틱 기록만으로 (GPU 없이)
# --------------------------------------------------------------------------

_RESUMMARISED_NOTE = "summaries and verdicts were rebuilt from the per-tick records by --from-report"
_FINISHED_AT_NOTE = "finished_at is null: this run predates the field and the end time cannot be recovered from the data; generated_at is the start of screen()"


def _git_commit() -> str | None:
    import subprocess

    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, timeout=10, cwd=REPO).stdout.strip() or None
    except Exception:  # noqa: BLE001 — 없으면 None
        return None


def resummarise(report: dict[str, Any]) -> dict[str, Any]:
    """보고서 JSON의 틱 기록에서 조건 요약과 판정을 다시 만든다 — 다시 재지 않는다.

    `environment`·`settings`·`profiles`·`memory`·틱 기록은 그대로 두고 `conditions[*][*].summary`와 `verdict`만 바꾼다.
    `generated_at` 옆에 `resummarised_at`과 이 스크립트의 버전·commit을 적는다. `finished_at`은 없으면 null로 남긴다 —
    끝난 시각은 데이터에서 되찾을 수 없으므로 지어내지 않는다(notes에 적는다).
    """
    budgets = {key: float(value) for key, value in (report.get("budgets_ms") or BUDGET_MS).items()}
    max_miss_rate = float(report["settings"]["max_miss_rate"])
    profiles = report.get("profiles") or {}
    stream = report.get("path") == "stream"
    for candidate in report["candidates"].values():
        blocks = [candidate] + list((candidate.get("levers") or {}).values()) if stream else [candidate]
        for block in blocks:
            for conditions in block["conditions"].values():
                for result in conditions.values():
                    result["summary"] = summarize_ticks(result.get("ticks") or [], budgets=budgets)
            if stream:
                block["verdict"] = stream_verdicts(block["conditions"], max_miss_rate=max_miss_rate, budgets=budgets)
            else:
                block["verdict"] = verdicts(block["conditions"], max_miss_rate=max_miss_rate, budgets=budgets, profiles=profiles)
        if stream:
            candidate["lever_verdicts"] = {name: block["verdict"] for name, block in (candidate.get("levers") or {}).items()}
    stamp = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    report["resummarised_at"] = stamp
    report["resummarised_by"] = {"script_version": SCRIPT_VERSION, "commit": _git_commit()}
    report.setdefault("finished_at", None)
    static = STREAM_NOTES if stream else NOTES
    old = report.get("notes") or []
    # 스크립트가 쓰는 고정 문구는 현재 버전의 것으로 바꾼다(문구는 데이터가 아니다 — 2b-G0a 리뷰 2 M12); 그 밖의 note는 남긴다
    extra = [note for note in old[len(static):] if not note.startswith(_RESUMMARISED_NOTE) and note != _FINISHED_AT_NOTE] if len(old) >= len(static) else [
        note for note in old if not note.startswith(_RESUMMARISED_NOTE) and note != _FINISHED_AT_NOTE
    ]
    notes = list(static) + extra
    notes.append(f"{_RESUMMARISED_NOTE} on {stamp} (script {SCRIPT_VERSION}); environment, settings, profiles, memory and tick records untouched; the script's own static notes were refreshed to this version's wording")
    if report["finished_at"] is None:
        notes.append(_FINISHED_AT_NOTE)
    report["notes"] = notes
    return report


# --------------------------------------------------------------------------
# 표
# --------------------------------------------------------------------------


def _gib(value: Any) -> str:
    return "-" if value is None else f"{float(value) / 2**30:.2f} GiB"


def _mib(value: Any) -> str:
    return "-" if value is None else f"{float(value) / 2**20:.0f} MiB"


def print_stream_table(report: dict[str, Any]) -> None:
    env = report["environment"]
    print(f"\n[G0b] stream path, dtype={report['settings']['dtype']}, gpu={env.get('gpu')}, torch={env.get('torch')}, transformers={env.get('transformers')}, "
          f"fla={env.get('flash_linear_attention')} causal_conv1d={env.get('causal_conv1d')} attention={env.get('attention_implementation')}")
    print(f"       ticks/episode={report['settings']['ticks']} history={report['settings']['history_ticks']} warmup={report['settings']['warmup']} levers={report['settings'].get('levers')}\n")
    header = (
        f"{'candidate':<18}{'lever':<13}{'profile':<19}{'ticks':>6}{'tok/tick':>9}{'cache min…max':>16}{'const':>6}"
        f"{'model p50':>10}{'p95':>8}{'p99':>8}{'max':>9}{'readout p50':>12}{'obs→apply p95':>15}{'miss>100':>9}{'>80ms':>7}{'alloc Δ MiB':>12}"
    )
    print(header)
    budgets = report["budgets_ms"]
    miss_key = f"deadline_miss_rate_{budgets['deadline']:.0f}ms"
    ten_key = f"over_budget_rate_{budgets['model_10hz']:.0f}ms"
    for model_id, candidate in report["candidates"].items():
        for lever, block in candidate["levers"].items():
            for profile, conditions in block["conditions"].items():
                for result in conditions.values():
                    s = result["summary"]
                    if not s.get("ticks"):
                        print(f"{model_id:<18}{lever:<13}{profile:<19}{0:>6}")
                        continue
                    m, o = s["model_ms"], s["obs_apply_ms"]
                    readout = [t["readout_ms"] for t in result["ticks"] if t.get("readout_ms") is not None]
                    cache = f"{s['cache_length']['min']}…{s['cache_length']['max']}"
                    print(
                        f"{model_id:<18}{lever:<13}{profile:<19}{s['ticks']:>6}{s['new_tokens']['p50']:>9.0f}{cache:>16}{str(s.get('cache_constant_after_window'))[:5]:>6}"
                        f"{m['p50']:>10.1f}{m['p95']:>8.1f}{m['p99']:>8.1f}{m['max']:>9.1f}{(percentile(readout, 0.5) if readout else 0):>12.2f}{o['p95']:>15.1f}"
                        f"{s[miss_key]:>9.2f}{s[ten_key]:>7.2f}{(s.get('allocated_growth_bytes') or 0) / 2**20:>12.1f}"
                    )
            memory = block.get("memory") or {}
            print(f"{'':<18}{lever:<13}memory: peak allocated {_gib(memory.get('peak_allocated_bytes'))}, reserved {_gib(memory.get('peak_reserved_bytes'))}, alloc retries {memory.get('num_alloc_retries')}, compile {block['loaded'].get('compile_seconds')} s")
            for profile, verdict in block["verdict"].items():
                if profile != "overall":
                    print(f"{'':<18}{lever:<13}verdict {verdict['text']}")
            overall = block["verdict"].get("overall") or {}
            print(f"{'':<18}{lever:<13}overall: passes_10hz={overall.get('passes_10hz')} passes_5hz={overall.get('passes_5hz')} on {overall.get('measured')}")
        print()


def print_table(report: dict[str, Any]) -> None:
    if report.get("path") == "stream":
        print_stream_table(report)
        return
    env = report["environment"]
    kernels = env.get("kernels") or {}
    print(f"\n[G0a] native path, dtype={report['settings']['dtype']}, gpu={env.get('gpu')}, torch={env.get('torch')}, transformers={env.get('transformers')}, "
          f"fla={env.get('flash_linear_attention')} (kernels active: {kernels.get('fla')}), causal_conv1d={env.get('causal_conv1d')} (active: {kernels.get('causal_conv1d')}), "
          f"attention={env.get('attention_implementation')}")
    print(f"       tokenizer={report['tokenizer'].get('id')} @ {report['tokenizer'].get('revision')}  serializer={report['serializer']}  "
          f"ticks/episode={report['settings']['ticks']} history={report['settings']['history_ticks']} warmup={report['settings']['warmup']} "
          f"cold window={report['settings']['cold_window_ticks']} cold ticks={report['settings']['cold_ticks']}")
    print("       native cache grows without the 30-tick window: `cache` is the KV length before/after the timed ticks.\n")
    header = (
        f"{'candidate':<18}{'profile':<20}{'condition':<13}{'ticks':>6}{'tok/tick':>9}{'cache min…max':>18}"
        f"{'model p50':>10}{'p95':>8}{'p99':>8}{'max':>9}{'wall p95':>9}{'obs→apply p95':>15}{'miss>100':>9}{'>80ms':>7}{'>150ms':>8}"
    )
    print(header)
    budgets = report["budgets_ms"]
    miss_key = f"deadline_miss_rate_{budgets['deadline']:.0f}ms"
    ten_key = f"over_budget_rate_{budgets['model_10hz']:.0f}ms"
    five_key = f"over_budget_rate_{budgets['model_5hz']:.0f}ms"
    for model_id, candidate in report["candidates"].items():
        short = model_id
        for profile, conditions in candidate["conditions"].items():
            for condition, result in conditions.items():
                s = result["summary"]
                if not s.get("ticks"):
                    print(f"{short:<18}{profile:<20}{condition:<13}{0:>6}")
                    continue
                m, w, o = s["model_ms"], s["wall_ms"], s["obs_apply_ms"]
                cache = f"{s['cache_length']['min']}…{s['cache_length']['max']}"
                print(
                    f"{short:<18}{profile:<20}{condition:<13}{s['ticks']:>6}{s['new_tokens']['p50']:>9.0f}{cache:>18}"
                    f"{m['p50']:>10.1f}{m['p95']:>8.1f}{m['p99']:>8.1f}{m['max']:>9.1f}{w['p95']:>9.1f}{o['p95']:>15.1f}"
                    f"{s[miss_key]:>9.2f}{s[ten_key]:>7.2f}{s[five_key]:>8.2f}"
                )
        memory = candidate["memory"]
        est = memory.get("estimates") or {}
        window = est.get("window_state") or {}
        chunk = est.get("training_chunk_activation") or {}
        print(
            f"{short:<18}memory: weights {_gib(memory.get('weight_bytes'))}, peak allocated {_gib(memory.get('peak_allocated_bytes'))}, "
            f"peak reserved {_gib(memory.get('peak_reserved_bytes'))}, params {candidate['loaded'].get('params')}, load {candidate['loaded'].get('load_seconds')} s"
        )
        if window:
            print(
                f"{'':<18}estimate ({est.get('reference_profile')}, prefix + {window.get('history_ticks')} ticks = {window.get('tokens')} tokens): "
                f"KV {_mib(window.get('kv_bytes'))} + recurrent {_mib(window.get('recurrent_bytes'))} + conv {_mib(window.get('conv_bytes'))} = {_mib(window.get('total_bytes'))}; "
                f"10 s chunk activation {_gib(chunk.get('measured_profile', {}).get('bytes'))} at {chunk.get('measured_profile', {}).get('tokens')} tokens, "
                f"{_gib(chunk.get('v03_50k', {}).get('bytes'))} at 50K (estimates, not measurements)"
            )
        measured = {p: {c: _mib(v) for c, v in conds.items()} for p, conds in (memory.get("cache_bytes_measured") or {}).items()}
        if measured:
            print(f"{'':<18}cache bytes at the end (measured): " + "; ".join(f"{p} {conds}" for p, conds in measured.items()))
        for profile, verdict in candidate["verdict"].items():
            print(f"{'':<18}verdict {verdict['text']}")
        print()


# --------------------------------------------------------------------------
# stream 경로 — 틱 하나의 obs→apply (Task 4의 실제 경로: adapter + 배치 분기 + pointer readout + typed 출력)
# --------------------------------------------------------------------------


def tick_readout(judge: Any, state: Any, tick: dict[str, Any], branch_hidden: Any, prefix_end: int) -> tuple[Any, list[tuple[str, int]]]:
    """틱의 질문별 pointer logits를 **한 tensor**로 (질문 순서 = 결정 표지 순서 = `branch_hidden`의 행). 정적 후보의 ``h_c``는
    prefix hidden, 동적 후보는 이 틱 몸통의 hidden이다. 돌려주는 것: 이어 붙인 logits ``[ΣK]``와 (qid, K) 목록."""
    import torch

    spans: list[tuple[str, int]] = []
    rows = []
    starts = int(tick["start"])
    for qid in tick["decision_positions"]:
        boundaries = [int(b) for b in tick["candidate_boundaries"][qid]]
        static = [b for b in boundaries if b < prefix_end]
        dynamic = [b - starts for b in boundaries if b >= starts]
        if static and dynamic:
            raise ValueError(f"{qid}: 후보 경계가 prefix와 틱 몸통에 섞여 있다")
        rows.append(state.prefix_hidden[static] if static else state.hidden[dynamic])
        spans.append((qid, len(boundaries)))
    h_c = torch.cat(rows)  # [ΣK, d]
    dtype = judge.readout_dtype
    scores = judge.V(h_c.to(dtype)) @ judge.U(branch_hidden.to(dtype)).T / math.sqrt(judge.rank) + judge.bias  # [ΣK, n]
    pieces = []
    offset = 0
    for column, (_, count) in enumerate(spans):
        pieces.append(scores[offset : offset + count, column])
        offset += count
    return torch.cat(pieces), spans


def typed_from_flat(flat: Any, spans: list[tuple[str, int]], tick: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """D2H한 logits(CPU) → 질문별 typed 출력 (:func:`robo_jev.model.judge.typed_outputs`)."""
    from robo_jev.model.judge import typed_outputs

    logits: dict[str, Any] = {}
    offset = 0
    for qid, count in spans:
        logits[qid] = flat[offset : offset + count]
        offset += count
    candidates = {qid: list(tick["candidate_mapping"][qid]) for qid, _ in spans}
    return typed_outputs(logits, candidates, QUESTION_SET_V0)


def run_stream_path(
    runner: Any, handle: Any, layout: dict[str, Any], *, name: str, history_ticks: int, warmup: int
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """에피소드 하나: prefix → 이력 `history_ticks`틱(틱마다 advance + 분기 — 실제 경로 그대로) → 예열 `warmup`틱 → 나머지 틱을 잰다."""
    ticks = layout["ticks"]
    first_timed = history_ticks + warmup
    if len(ticks) <= first_timed:
        raise ValueError(f"{name}: 틱이 {len(ticks)}개라 이력 {history_ticks} + 예열 {warmup} 뒤에 잴 틱이 없다")
    state = runner.begin_episode(handle, layout)
    records: list[dict[str, Any]] = []
    for index, tick in enumerate(ticks):
        if index == first_timed:
            runner.reset_peak()
        started = time.perf_counter()
        before = state.cached_tokens
        state, timing = runner.tick(handle, state, layout, tick)
        obs_apply = (time.perf_counter() - started) * 1e3
        if index < first_timed:
            continue
        records.append(
            {
                "tick": index,
                "new_tokens": int(tick["end"]) - int(tick["start"]),
                "body_tokens": int(tick["body_end"]) - int(tick["start"]),
                "decisions": int(tick["end"]) - int(tick["body_end"]),
                "cache_before": before,
                "cache_after": state.cached_tokens,
                "window_tokens": state.window_tokens,
                "window_ticks_in_cache": len(state._ticks),
                "model_ms": timing["model_ms"],
                "wall_ms": timing["wall_ms"],
                "obs_apply_ms": obs_apply,
                "readout_ms": timing.get("readout_ms"),
                "allocated_bytes": runner.allocated(),
            }
        )
    info = {
        "ticks_total": len(ticks),
        "cache_length_end": state.cached_tokens,
        "cache_bytes_end": state.kv_bytes(),
        "peak_allocated_bytes_timed": runner.peak_allocated(),
        "allocated_bytes_end": runner.allocated(),
    }
    runner.end_episode(handle, state)
    return records, info


class StreamRunner:
    """실제 stream 경로 runner — adapter(:class:`robo_jev.model.backbone_qwen.QwenBackbone`) + Judge(pointer readout)."""

    def __init__(self, device: str = "cuda", dtype: str = "bf16") -> None:
        import torch

        if dtype not in DTYPES:
            raise ValueError(f"dtype: {list(DTYPES)} 중 하나 (받은 값: {dtype!r})")
        self.device = device
        self.dtype = getattr(torch, DTYPES[dtype])
        self._start = torch.cuda.Event(enable_timing=True)
        self._mid = torch.cuda.Event(enable_timing=True)
        self._end = torch.cuda.Event(enable_timing=True)
        self._attn: str | None = None

    def load(self, config: dict[str, Any]) -> Any:
        import torch

        from robo_jev.model.backbone_qwen import QwenBackbone
        from robo_jev.model.judge import Judge

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        lever = dict(LEVER_SETTINGS["baseline"])
        lever.update(config.get("lever") or {})
        backbone = QwenBackbone.load(config["id"], dtype=self.dtype, device=self.device, kv_mode="static")
        readout_dtype = torch.float32 if lever["readout_dtype"] == "float32" else torch.bfloat16
        judge = Judge(backbone, rank=int(config.get("readout_rank") or 64), readout="pointer", seed=1000, readout_dtype=readout_dtype)
        # **적재가 먼저, 컴파일이 나중** (G0b 리뷰 2 M-b): `torch.compile`은 감싼 module의 state_dict 키에 `_orig_mod.` 접두사를
        # 붙이므로 컴파일한 모델에 LoRA checkpoint를 실으면 `lora_*` 키가 "모델에 없는 파라미터"로 거절된다. readout만 있는
        # checkpoint는 backbone 밖이라 우연히 통과했을 뿐이다.
        if config.get("checkpoint"):
            from robo_jev.train import load_readout_checkpoint

            # 배포 계약 digest의 네 조각(tokenizer 파일 해시 포함)을 지금 체크아웃 기준으로 대조한다 (리뷰 1 I2)
            load_readout_checkpoint(judge, config["checkpoint"], tokenizer_sha256=config.get("tokenizer_sha256"))
        backbone.use_branch_graph = bool(lever["graphs"])
        compile_seconds = None
        if lever["compile"]:
            started = time.perf_counter()
            compile_seconds = backbone.compile_dense_parts()
            compile_seconds = round(time.perf_counter() - started, 1) if compile_seconds is None else compile_seconds
        return {"backbone": backbone, "judge": judge, "config": config, "lever": lever, "compile_seconds": compile_seconds}

    # -- 에피소드·틱 --

    def begin_episode(self, handle: Any, layout: dict[str, Any]) -> Any:
        from robo_jev.model.backbone_qwen import QwenStreamState

        state = QwenStreamState.initial(handle["backbone"], window_ticks=int(layout.get("window_ticks", WINDOW_TICKS)))
        return state.extend_prefix(layout["tokens"][: layout["prefix_end"]])

    def tick(self, handle: Any, state: Any, layout: dict[str, Any], tick: dict[str, Any]) -> tuple[Any, dict[str, float]]:
        import torch

        judge = handle["judge"]
        tokens = layout["tokens"]
        body = tokens[int(tick["start"]) : int(tick["body_end"])]  # 자르기 (직렬화·토큰화는 이미 끝났다)
        decisions = tokens[int(tick["body_end"]) : int(tick["end"])]
        wall_started = time.perf_counter()
        with torch.no_grad():
            self._start.record()
            if handle["lever"].get("fused") and decisions:
                state, branch_hidden = state.advance_with_branches(body, decisions)  # 몸통 + 분기 한 forward
            else:
                state = state.advance(body)  # H2D는 안에서 (틱 토큰 tensor)
                branch_hidden = state.branch_step(decisions) if decisions else None
            self._mid.record()
            if branch_hidden is not None:
                flat, spans = tick_readout(judge, state, tick, branch_hidden, int(layout["prefix_end"]))
            self._end.record()
        torch.cuda.synchronize()
        wall = (time.perf_counter() - wall_started) * 1e3
        if branch_hidden is not None:
            typed_from_flat(flat.float().cpu(), spans, tick)  # D2H 한 번 + CPU에서 softmax·선택
        return state, {"model_ms": float(self._start.elapsed_time(self._end)), "readout_ms": float(self._mid.elapsed_time(self._end)), "wall_ms": wall}

    def end_episode(self, handle: Any, state: Any) -> None:
        handle["backbone"].release_branch_graph()

    # -- 메모리·환경 --

    def reset_peak(self) -> None:
        import torch

        torch.cuda.reset_peak_memory_stats()

    def peak_allocated(self) -> int:
        import torch

        return int(torch.cuda.max_memory_allocated())

    def allocated(self) -> int:
        import torch

        return int(torch.cuda.memory_allocated())

    def memory(self) -> dict[str, Any]:
        return TransformersRunner.memory(self)  # type: ignore[arg-type]

    def describe(self, handle: Any) -> dict[str, Any]:
        backbone = handle["backbone"]
        params = list(backbone.model.parameters())
        return {
            "class": type(backbone.model).__name__,
            "adapter": type(backbone).__name__,
            "params": int(sum(p.numel() for p in params)),
            "weight_bytes": int(sum(p.numel() * p.element_size() for p in params)),
            "dtype": str(params[0].dtype),
            "attention_backend": backbone.attention_backend,
            "kernels": backbone.kernel_names(),
            "layer_types": dict(Counter(backbone.layer_types)),
            "levers": handle["lever"],
            "compile_seconds": handle.get("compile_seconds"),
            "readout_rank": handle["judge"].rank,
            "readout_dtype": str(handle["judge"].readout_dtype),
            "window_capacity": backbone.window_capacity,
        }

    def unload(self, handle: Any) -> None:
        import gc

        import torch

        handle.pop("judge", None)
        handle.pop("backbone", None)
        gc.collect()
        torch.cuda.empty_cache()

    def environment(self) -> dict[str, Any]:
        env = TransformersRunner.environment(self)  # type: ignore[arg-type]
        env["attention_implementation"] = self._attn
        env["path"] = "stream"
        return env


# --------------------------------------------------------------------------
# 실제 runner — transformers (여기서만 import)
# --------------------------------------------------------------------------

_GDN_FUNCTIONS = ("torch_chunk_gated_delta_rule", "torch_recurrent_gated_delta_rule", "causal_conv1d_fn", "causal_conv1d_update")


def _package_version(name: str) -> str | None:
    import importlib.metadata

    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def kernel_status() -> dict[str, Any]:
    """transformers의 qwen3_5 모듈이 실제로 고른 구현 — fla(Triton)·causal_conv1d(CUDA)인지, torch 참조 경로인지.

    `use_kernel_func_from_hub_with_fallback`가 import 시점에 고른 `implementation`을 closure에서 읽는다.
    """
    import inspect

    import transformers.models.qwen3_5.modeling_qwen3_5 as modeling

    status: dict[str, Any] = {"functions": {}}
    for name in _GDN_FUNCTIONS:
        try:
            nonlocals = inspect.getclosurevars(getattr(modeling, name)).nonlocals
            implementation = nonlocals.get("implementation")
            status["functions"][name] = {
                "implementation": f"{getattr(implementation, '__module__', None)}.{getattr(implementation, '__name__', None)}",
                "optimized": bool(nonlocals.get("is_new_implementation")),
            }
        except Exception as exc:  # noqa: BLE001 — 상태 보고이지 측정이 아니다
            status["functions"][name] = {"implementation": None, "optimized": None, "error": f"{type(exc).__name__}: {exc}"}
    functions = status["functions"]
    status["fla"] = all(functions[name].get("optimized") and "fla." in str(functions[name].get("implementation")) for name in _GDN_FUNCTIONS[:2])
    status["causal_conv1d"] = all(functions[name].get("optimized") and "causal_conv1d" in str(functions[name].get("implementation")) for name in _GDN_FUNCTIONS[2:])
    return status


def fla_kernel_check(device: str = "cuda") -> dict[str, Any]:
    """fla의 Triton kernel(`chunk_gated_delta_rule`)이 이 GPU에서 컴파일·실행되는지 작은 입력으로 확인하고 torch 참조와 대조한다.

    컴파일 실패는 예외 문자열을 그대로 적는다 — 조용히 느린 경로로 넘어가지 않는다.
    """
    import inspect

    import torch
    import transformers.models.qwen3_5.modeling_qwen3_5 as modeling

    try:
        from fla.ops.gated_delta_rule import chunk_gated_delta_rule
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": f"import: {type(exc).__name__}: {exc}"}
    reference = inspect.getclosurevars(modeling.torch_chunk_gated_delta_rule).nonlocals.get("torch_function")
    try:
        generator = torch.Generator(device=device).manual_seed(0)
        batch, length, heads, dim = 1, 128, 2, 64
        q = torch.randn(batch, length, heads, dim, device=device, dtype=torch.bfloat16, generator=generator)
        k = torch.randn(batch, length, heads, dim, device=device, dtype=torch.bfloat16, generator=generator)
        v = torch.randn(batch, length, heads, dim, device=device, dtype=torch.bfloat16, generator=generator)
        g = -torch.rand(batch, length, heads, device=device, dtype=torch.float32, generator=generator)
        beta = torch.rand(batch, length, heads, device=device, dtype=torch.bfloat16, generator=generator)
        out, state = chunk_gated_delta_rule(q, k, v, g, beta, output_final_state=True, use_qk_l2norm_in_kernel=True)
        torch.cuda.synchronize()
        result: dict[str, Any] = {"ok": True, "device": torch.cuda.get_device_name(0), "output_shape": list(out.shape), "state_shape": list(state.shape)}
        if reference is not None:
            ref_out, _ = reference(q, k, v, g=g, beta=beta, output_final_state=True, use_qk_l2norm_in_kernel=True)
            result["max_abs_diff_vs_torch_reference"] = float((out.float() - ref_out.float()).abs().max().item())
        return result
    except Exception as exc:  # noqa: BLE001 — 실패 자체가 발견이다
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}


class TransformersRunner:
    """공식 `transformers` forward(BF16, CUDA). `transformers`·`fla`는 여기서만 import한다."""

    def __init__(self, device: str = "cuda", dtype: str = "bf16") -> None:
        import torch

        if dtype not in DTYPES:
            raise ValueError(f"dtype: {list(DTYPES)} 중 하나 (받은 값: {dtype!r})")
        self.device = device
        self.dtype = getattr(torch, DTYPES[dtype])
        self._start = torch.cuda.Event(enable_timing=True)
        self._end = torch.cuda.Event(enable_timing=True)
        self._attn: str | None = None

    def load(self, config: dict[str, Any]) -> Any:
        import torch
        from transformers import AutoModelForCausalLM
        from transformers.utils import logging as hf_logging

        hf_logging.disable_progress_bar()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        model = AutoModelForCausalLM.from_pretrained(config["path"], dtype=self.dtype)
        model.to(self.device)
        model.eval()
        self._attn = getattr(model.config, "_attn_implementation", None)
        return {"model": model, "config": config}

    def forward(self, handle: Any, ids: list[int], cache: Any) -> tuple[Any, Any]:
        import torch

        model = handle["model"]
        input_ids = torch.tensor([ids], dtype=torch.long, device=self.device)  # H2D
        with torch.inference_mode():
            self._start.record()
            out = model(input_ids=input_ids, past_key_values=cache, use_cache=True, logits_to_keep=1)
            self._end.record()
        return out.logits[0, -1], out.past_key_values

    def sync(self) -> None:
        import torch

        torch.cuda.synchronize()

    def event_ms(self) -> float:
        return float(self._start.elapsed_time(self._end))

    def cache_length(self, cache: Any) -> int:
        return 0 if cache is None else int(cache.get_seq_length())

    def cache_bytes(self, cache: Any) -> int:
        import torch

        if cache is None:
            return 0
        total = 0
        for layer in getattr(cache, "layers", ()):
            for attribute in ("keys", "values", "conv_states", "recurrent_states"):
                value = getattr(layer, attribute, None)
                tensors = value.values() if isinstance(value, dict) else (value if isinstance(value, (list, tuple)) else [value])
                total += sum(t.numel() * t.element_size() for t in tensors if torch.is_tensor(t))
        return int(total)

    def memory(self) -> dict[str, Any]:
        import torch

        stats = torch.cuda.memory_stats()
        return {
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
            "allocated_bytes": int(torch.cuda.memory_allocated()),
            # 작은 장치에서 allocator가 free-and-retry를 했는지 보이도록 (docs/05 §4의 메모리 통과 기준의 재료)
            "num_alloc_retries": int(stats.get("num_alloc_retries", 0)),
            "num_ooms": int(stats.get("num_ooms", 0)),
        }

    def describe(self, handle: Any) -> dict[str, Any]:
        model = handle["model"]
        params = list(model.parameters())
        tied = getattr(model, "lm_head", None) is not None and model.lm_head.weight.data_ptr() == model.model.embed_tokens.weight.data_ptr()
        return {
            "class": type(model).__name__,
            "params": int(sum(p.numel() for p in params)),
            "weight_bytes": int(sum(p.numel() * p.element_size() for p in params)),
            "dtype": str(next(iter(params)).dtype),
            "attn_implementation": self._attn,
            "layer_types": dict(Counter(model.config.layer_types)),
            "tied_embeddings": bool(tied),
            "mamba_ssm_dtype": getattr(model.config, "mamba_ssm_dtype", None),
        }

    def unload(self, handle: Any) -> None:
        import gc

        import torch

        handle.pop("model", None)
        gc.collect()
        torch.cuda.empty_cache()

    def environment(self) -> dict[str, Any]:
        import platform
        import subprocess

        import torch

        driver = None
        try:
            driver = subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"], capture_output=True, text=True, timeout=10).stdout.strip() or None
        except Exception:  # noqa: BLE001
            driver = None
        free, total = torch.cuda.mem_get_info()
        return {
            "gpu": torch.cuda.get_device_name(0),
            "compute_capability": ".".join(map(str, torch.cuda.get_device_capability(0))),
            "driver": driver,
            "cuda": torch.version.cuda,
            "torch": torch.__version__,
            "triton": _package_version("triton"),
            "transformers": _package_version("transformers"),
            "flash_linear_attention": _package_version("flash-linear-attention"),
            "causal_conv1d": _package_version("causal-conv1d"),
            "python": platform.python_version(),
            "machine": platform.machine(),
            "device_memory_total_bytes": int(total),
            "device_memory_free_bytes_at_start": int(free),
            "attention_implementation": self._attn,
            "kernels": kernel_status(),
            "fla_kernel_check": fla_kernel_check(self.device),
        }


# --------------------------------------------------------------------------


def main(argv: list[str] | None = None, *, runner: Runner | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default=str(CANDIDATES_CONFIG), help="후보 목록 yaml")
    parser.add_argument("--path", default="native", choices=PATHS, help="native(G0a, 공식 forward) 또는 stream(G0b, 실제 스트림 경로)")
    parser.add_argument("--candidates", help="잴 후보 id, 쉼표로 (기본: role이 main·separate인 후보)")
    parser.add_argument("--include-reference", dest="include_reference", action="store_true", help="role=reference(27B)도 잰다")
    parser.add_argument("--profiles", default=",".join(PROFILES), help=f"프로파일, 쉼표로 (기본: {','.join(PROFILES)}; state_first는 언제나)")
    parser.add_argument("--ticks", type=int, default=Settings.ticks, help="에피소드당 틱 수 (D0는 자르고 합성은 늘린다)")
    parser.add_argument("--history", type=int, default=Settings.history_ticks, help="warm의 이력 틱 수 (cache에 먼저 넣는다)")
    parser.add_argument("--warmup", type=int, default=Settings.warmup, help="재기 전 예열 틱(요청) 수")
    parser.add_argument("--cold-window-ticks", dest="cold_window_ticks", type=int, default=Settings.cold_window_ticks, help="cold가 다시 계산하는 최근 틱 수 (0 = 이력 전부)")
    parser.add_argument("--cold-ticks", dest="cold_ticks", type=int, default=Settings.cold_ticks, help="cold에서 재는 틱 수 (음수 = 전부)")
    parser.add_argument("--cold-warmup", dest="cold_warmup", type=int, default=Settings.cold_warmup, help="cold의 예열 forward 수 (한 번이 수십 초라 warm과 따로)")
    parser.add_argument("--max-miss-rate", dest="max_miss_rate", type=float, default=DEFAULT_MAX_MISS_RATE, help="deadline 초과율 문턱 (docs/03 §7-6: 파일럿이 정한다)")
    parser.add_argument("--report", default=None, help=f"보고서 JSON (기본: native {DEFAULT_REPORT.relative_to(REPO)}, stream {DEFAULT_STREAM_REPORT.relative_to(REPO)})")
    parser.add_argument("--episodes", default=None, help="실제 로봇 에피소드 batch 디렉터리 (manifest.json) — `batch0` 프로파일")
    parser.add_argument("--episodes-limit", dest="episodes_limit", type=int, default=None, help="batch에서 쓸 에피소드 수 (기본: 전부)")
    parser.add_argument("--levers", default=None, help=f"stream 경로의 지렛대, 쉼표로 ({', '.join(LEVERS)}; 첫 것이 판정의 기준; 기본 = stream `{DEFAULT_SERVING_LEVER}`(서빙 구성), native `baseline`)")
    parser.add_argument("--readout-rank", dest="readout_rank", type=int, default=64, help="stream 경로 pointer readout의 rank")
    parser.add_argument("--checkpoint", default=None, help="stream 경로에 실을 readout checkpoint (없으면 seed 초기값 — 지연은 값에 무관)")
    parser.add_argument("--dtype", default="bf16", choices=sorted(DTYPES))
    parser.add_argument("--root", default=None, help="가중치 보관 디렉터리 (기본: artifacts/models)")
    parser.add_argument("--verify-full", dest="verify_full", action="store_true", help="safetensors 전부의 sha256을 manifest와 대조한다")
    parser.add_argument("--from-report", dest="from_report", help="이 보고서 JSON의 틱 기록으로 요약·판정만 다시 만든다 (GPU·가중치 없이; --report가 없으면 같은 파일에 쓴다)")
    parser.add_argument("--gpu-memory-fraction", dest="gpu_memory_fraction", type=float, default=DEFAULT_FRACTION, help="프로세스가 쓸 장치(통합) 메모리 몫 (robo_jev.gpu; 첫 CUDA 할당 전에 건다)")
    args = parser.parse_args(argv)

    if args.from_report:
        source = Path(args.from_report)
        report = resummarise(json.loads(source.read_text(encoding="utf-8")))
        out = Path(args.report) if args.report is not None else source  # --report를 안 주면 같은 파일에 쓴다 (2b-G0a 리뷰 2 M14a)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print_table(report)
        print(f"→ {out} (resummarised from {source})")
        return 0

    if not 0.0 <= args.max_miss_rate <= 1.0:
        parser.error("--max-miss-rate는 0~1")
    if args.levers is None:
        args.levers = DEFAULT_SERVING_LEVER if args.path == "stream" else "baseline"
    levers = tuple(item.strip() for item in args.levers.split(",") if item.strip())
    unknown_levers = sorted(set(levers) - set(LEVERS))
    if unknown_levers:
        parser.error(f"--levers: 모르는 지렛대 {unknown_levers} (아는 것: {list(LEVERS)})")
    if args.path == "native" and levers != ("baseline",):
        parser.error("--levers는 stream 경로의 것이다")

    config_path = Path(args.config)
    data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    wanted = [item.strip() for item in args.candidates.split(",")] if args.candidates else None
    entries: list[dict[str, Any]] = []
    for entry in data["candidates"]:
        if wanted is not None:
            if entry["id"] not in wanted:
                continue
        elif entry.get("role") == "reference" and not args.include_reference:
            continue
        entries.append(entry)
    if wanted is not None:
        missing = sorted(set(wanted) - {entry["id"] for entry in entries})
        if missing:
            parser.error(f"--candidates: {config_path}에 없는 후보다: {missing}")
    if not entries:
        parser.error("잴 후보가 없다")

    try:
        for entry in entries:
            entry.update(candidate_config(entry["id"], config_path, args.root, verify_full=args.verify_full))
    except (FileNotFoundError, ValueError) as exc:
        print(f"[G0a] {exc}", file=sys.stderr)
        return 1

    tokenizer_id = _tokenizer_id()
    tokenizer = load_tokenizer(tokenizer_id)
    tokenizer_info = {key: value for key, value in describe_tokenizer(tokenizer_id).items() if key in ("id", "revision", "sha256", "manifest")}
    if data.get("tokenizer") and data["tokenizer"] != tokenizer_id:
        print(f"[G0a] 경고: candidates.yaml의 tokenizer({data['tokenizer']})와 받아 둔 tokenizer({tokenizer_id})가 다르다", file=sys.stderr)

    names = tuple(item.strip() for item in args.profiles.split(",") if item.strip())
    settings = Settings(
        ticks=args.ticks, warmup=args.warmup, history_ticks=args.history, cold_window_ticks=args.cold_window_ticks,
        cold_ticks=None if args.cold_ticks is not None and args.cold_ticks < 0 else args.cold_ticks, cold_warmup=args.cold_warmup,
        max_miss_rate=args.max_miss_rate, dtype=args.dtype, verify_full=args.verify_full, path=args.path, levers=list(levers),
        episodes_dir=args.episodes, episodes_limit=args.episodes_limit, readout_rank=args.readout_rank, checkpoint=args.checkpoint,
    )
    episodes = read_episodes(args.episodes, limit=args.episodes_limit) if args.episodes else None
    print(f"[G0b] building profiles ({', '.join(names)}{' + batch0' if episodes else ''}) with {tokenizer_id}, {settings.ticks} ticks/episode", file=sys.stderr, flush=True)
    profiles = build_profiles(
        tokenizer, ticks=settings.ticks, names=names, change_tick=min(settings.history_ticks + settings.warmup, settings.ticks - 1),
        episodes=episodes, include_state_first=args.path == "native",
    )

    if runner is None:
        guard = limit_gpu_memory(args.gpu_memory_fraction)  # 첫 CUDA 할당 전에 — GB10 통합 메모리 울타리
        print(f"[G0b] gpu guard {guard} · memory at start {memory_report()}", file=sys.stderr, flush=True)
        runner = StreamRunner(dtype=settings.dtype) if args.path == "stream" else TransformersRunner(dtype=settings.dtype)
    command = "uv run python " + " ".join([str(Path(sys.argv[0]).relative_to(REPO)) if Path(sys.argv[0]).is_absolute() and str(sys.argv[0]).startswith(str(REPO)) else sys.argv[0], *(argv if argv is not None else sys.argv[1:])])
    out = Path(args.report) if args.report is not None else (DEFAULT_STREAM_REPORT if args.path == "stream" else DEFAULT_REPORT)
    report = screen(entries, profiles, runner, settings=settings, tokenizer_info=tokenizer_info, command=command, checkpoint=out.with_suffix(".partial.json"))
    report["gpu_memory_at_end"] = memory_report()
    print(f"[G0b] memory at end {report['gpu_memory_at_end']}", file=sys.stderr, flush=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    out.with_suffix(".partial.json").unlink(missing_ok=True)
    print_table(report)
    print(f"→ {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
