"""Task 2b G0a — backbone 후보의 **native 경로** 지연·메모리 예비 선별 (DGX Spark, BF16, docs/06 Task 2b 1단계).

후보(`configs/model/candidates.yaml`)마다 공식 `transformers` forward(`AutoModelForCausalLM`, `use_cache=True`)를 대표
로봇 스트림 입력에 돌려 틱당 지연·deadline 초과율·메모리를 재고, 10Hz 예산(docs/03: 모델 시간 80ms, 5Hz는 150ms,
obs→apply deadline 100ms)에 명백히 못 미치는 후보를 adapter·readout 작업 전에 거른다. 2단계(`--path stream`: 윈도우
KV·`StreamState` 분기·pointer readout)는 다른 브리프다 — 여기서는 `stream`을 거절한다.

**native cache는 30틱 윈도우 없이 자란다.** 모델 자체의 hybrid cache(linear 층의 recurrent/conv state + full 층의 KV)에
틱의 새 토큰만 이어 넣는데, docs/08 §3.1의 "정적 prefix + 최근 30틱" 윈도우는 2단계 경로의 속성이라 여기 없다. 그래서
full-attention의 KV는 에피소드 길이만큼 자라고 틱 지연도 그만큼 늘어난다 — 지연마다 **그 틱 직전의 cache 길이**를
같이 적는다. 윈도우가 있을 때의 지연은 cache 길이가 비슷한 틱의 값으로 읽는다.

입력은 전부 :func:`robo_jev.model.serialize.serialize_request` 가 만든 토큰이다(손으로 쓴 텍스트 없음). 틱의 새 토큰은
직렬화의 틱 경계(``ticks[i].start … end``, 결정 토큰 포함)이고 prefix는 ``tokens[:prefix_end]``다.

프로파일(`--profiles`):

* ``d0_streams`` — `tests/fixtures/d0_streams.jsonl`의 4 에피소드(각 `--ticks` 틱으로 자른다), 틱당 ≈800 토큰.
* ``lower`` / ``upper`` / ``instruction_change`` — `scripts/measure_tokens.py`의 장면 빌더로 만든 합성 에피소드
  (물체 6·K=12 ≈1.8K / 물체 10·K=32 ≈3.5K / 10·K=32 + 지시 변경 틱 ≈3.7K), `--ticks` 틱까지 `append_tick`으로 늘린다.
  지시 변경 틱은 이력·예열 뒤 첫 측정 틱(`--history + --warmup`)이라 측정 안에 든다.
* ``v03_target`` — ``upper`` 스트림의 **틱마다 토큰 목록을 500개로 자른 것**. 계약 v0.3(HANDOFF 결정 1, ≈500 토큰/틱)의
  **길이만의 대역**이다 — 내용은 v0.3 서식이 아니다. "≈500 토큰/틱이면 이 후보가 들어오는가"에만 답한다.
* ``state_first`` — `tests/fixtures/d0.jsonl` 64건의 단일 요청(L0, cache 없음). 프로파일이 아니라 조건이지만 같은 표에 둔다.

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
``tokens × hidden × layers × 2B``(측정 프로파일의 100틱과 v0.3 목표 50K). 판정(후보, `lower`와 `v03_target` 따로,
stream_warm 기준): `fails_10hz` = p95 모델 ms > 80, `fails_5hz` = > 150, `deadline_fail` = 초과율 > `--max-miss-rate`
(기본 0.05 — docs/03 §7-6이 수치를 파일럿에 맡겨 CLI 인자다). 품질(bullet 4)로는 떨어뜨리지 않는다.

runner는 주입할 수 있다(:class:`Runner`; 기본 :class:`TransformersRunner`). 검사는 결정적 시간을 내는 가짜를 넣고 CPU에서
가중치·`transformers` 없이 돈다 — `transformers`·`fla`는 실제 runner 안에서만 import한다.

실행: `uv run python scripts/measure_candidates.py --config configs/model/candidates.yaml --path native
       [--candidates id,…] [--include-reference] [--profiles …] [--ticks N] [--report artifacts/reports/backbone-screen.json]
       [--dtype bf16] [--max-miss-rate 0.05] [--cold-ticks N] [--verify-full]`
가중치가 없으면 `scripts/fetch_backbone.py`를 가리키는 오류로 멈춘다.
"""

from __future__ import annotations

import argparse
import copy
import dataclasses
import datetime as dt
import importlib.util
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Protocol

import yaml

from robo_jev.data.episode import append_tick, new_episode
from robo_jev.harness.robot import candidate_id
from robo_jev.model.backbone import CANDIDATES_CONFIG, FETCH_SCRIPT, backbone_root, describe_backbone
from robo_jev.model.serialize import TOKEN_SERIALIZER_VERSION, WINDOW_TICKS, serialize_request
from robo_jev.model.tokenizer import available_tokenizer, describe_tokenizer, load_tokenizer

REPO = Path(__file__).resolve().parents[1]
D0 = REPO / "tests" / "fixtures" / "d0.jsonl"
D0_STREAMS = REPO / "tests" / "fixtures" / "d0_streams.jsonl"
DEFAULT_REPORT = REPO / "artifacts" / "reports" / "backbone-screen.json"

PATHS = ("native",)
PROFILES = ("d0_streams", "lower", "upper", "instruction_change", "v03_target")
STREAM_CONDITIONS = ("stream_warm", "stream_cold")
#: docs/03 §"지연 예산에서 역산하는 backbone 선정": 모델 시간 80ms(10Hz)·150ms(5Hz), obs→apply deadline 100ms.
BUDGET_MS = {"model_10hz": 80.0, "model_5hz": 150.0, "deadline": 100.0}
DEFAULT_MAX_MISS_RATE = 0.05
#: 계약 v0.3의 틱당 토큰 목표(HANDOFF 결정 1) — `v03_target`이 자르는 길이.
V03_TICK_TOKENS = 500
#: 합성 프로파일 (물체 수, K 상한, 지시 변경).
SYNTHETIC = {
    "lower": (6, 12, False),
    "upper": (10, 32, False),
    "instruction_change": (10, 32, True),
}
DTYPES = {"bf16": "bfloat16"}
#: 이 스크립트의 버전 — `--from-report`가 다시 요약할 때 JSON에 적는다 (요약·판정의 정의가 바뀌면 올린다).
SCRIPT_VERSION = "g0a-1.1"
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
    truncated_to: int | None = None

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
        if self.truncated_to is not None:
            end = min(end, start + self.truncated_to)
        return self.tokens[start:end]


def stream_input(out: dict[str, Any], *, name: str, truncate_tick_tokens: int | None = None) -> StreamInput:
    if out.get("layout") != "stream_l1a":
        raise ValueError(f"stream_l1a 직렬화 결과가 필요하다 (받은 layout: {out.get('layout')!r})")
    return StreamInput(
        name=name,
        tokens=list(out["tokens"]),
        prefix_end=int(out["prefix_end"]),
        bounds=[(int(tick["start"]), int(tick["end"])) for tick in out["ticks"]],
        truncated_to=truncate_tick_tokens,
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
    return {
        "ticks": len(ticks),
        "new_tokens": {"min": min(tokens), "p50": percentile(tokens, 0.5), "max": max(tokens)},
        "cache_length": {"min": min(int(tick["cache_before"]) for tick in ticks), "max": max(int(tick["cache_after"]) for tick in ticks)},
        "model_ms": _ms_summary(model),
        "wall_ms": _ms_summary(wall),
        "obs_apply_ms": _ms_summary(obs_apply),
        f"deadline_miss_rate_{deadline:.0f}ms": _rate(obs_apply, deadline),
        f"over_budget_rate_{ten:.0f}ms": _rate(model, ten),
        f"over_budget_rate_{five:.0f}ms": _rate(model, five),
    }


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
    truncate_tick_tokens: int | None = None,
    budgets: dict[str, float] = BUDGET_MS,
) -> dict[str, Any]:
    """docs/06 Task 2b의 인터페이스: 후보 `model_id`에 `requests`를 `layout`으로 넣어 틱당 지연을 잰다.

    `requests`는 레코드(직렬화한다) 또는 이미 직렬화된 결과다. `layout="stream_l1a"`면 `warm_prefix`가 조건을 고른다
    (True = ``stream_warm``, False = ``stream_cold``); `state_first`는 cache가 없어 `warm_prefix`를 쓰지 않는다. `path`는
    ``native``만 받는다 — ``stream``은 2단계(Task 4의 실제 스트림 경로)다. `handle`이 없으면 runner로 싣고 끝에 내린다.
    돌려주는 dict: ``ticks``(틱 기록), ``summary``(:func:`summarize_ticks`), ``episodes``(에피소드별 cache 끝 길이·바이트).
    """
    if path == "stream":
        raise ValueError("path='stream'은 2단계(G0b, Task 4의 실제 스트림 경로 — 윈도우 KV·StreamState 분기·pointer readout)다. 이 스크립트는 native만 잰다")
    if path not in PATHS:
        raise ValueError(f"path: {list(PATHS)} 중 하나여야 한다 (받은 값: {path!r})")
    if layout not in ("stream_l1a", "state_first"):
        raise ValueError(f"layout: stream_l1a 또는 state_first (받은 값: {layout!r})")
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
                stream = stream_input(out, name=str(out.get("episode_id") or f"stream-{number}"), truncate_tick_tokens=truncate_tick_tokens)
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
                "truncate_tick_tokens": truncate_tick_tokens,
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
            "v03_target_50k": activation_bytes(entry, tokens=50_000),
        },
    }


# --------------------------------------------------------------------------
# 판정 (docs/06 1단계 탈락 규칙 — lower와 v03_target 따로)
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
    if prefix_tokens is None or tick_tokens_mean is None:
        prefix_tokens = min(caches)  # 프로파일 정보가 없으면 측정 틱에서 되만든다 — basis에 적는다
        tick_tokens_mean = statistics.fmean(int(tick["new_tokens"]) for tick in ticks)
        basis = "min(cache_before) + (WINDOW_TICKS − 1) × mean new_tokens of the timed ticks"
    else:
        basis = "profile prefix_tokens + (WINDOW_TICKS − 1) × profile mean tick tokens"
    window_cache = int(round(float(prefix_tokens) + (window_ticks - 1) * float(tick_tokens_mean)))
    fit = _ols(caches, [float(tick["model_ms"]) for tick in ticks])
    return {
        "cache_before_range": [int(min(caches)), int(max(caches))],
        "early_ticks": early_ticks,
        "early_ticks_mean_ms": round(statistics.fmean(early), 2),
        "window_cache_tokens": window_cache,
        "window_cache_basis": basis,
        "model_ms_at_window_cache": None if fit is None else round(fit["intercept"] + fit["slope"] * window_cache, 2),
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
    """docs/06 1단계 탈락 규칙 — `lower`와 `v03_target`에서 따로, stream_warm의 **문자 그대로의 p95**(native cache가 자란
    채로)로 flag를 정한다(보수적). 그 옆에 :func:`window_reading` 의 윈도우 크기 읽기(직선 맞춤·첫 5틱 평균·cache 범위)를
    적고 문장에도 인용한다 — flag는 그것으로 바꾸지 않는다."""
    deadline_key = f"deadline_miss_rate_{budgets['deadline']:.0f}ms"
    out: dict[str, Any] = {}
    for profile in ("lower", "v03_target"):
        result = (conditions.get(profile) or {}).get("stream_warm") or {}
        summary = result.get("summary")
        if not summary or not summary.get("ticks"):
            out[profile] = {
                "profile": profile, "condition": "stream_warm", "ticks": 0, "p95_model_ms": None, deadline_key: None,
                "fails_10hz": None, "fails_5hz": None, "deadline_fail": None, "passes": None,
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
            "window": window,
            "text": text,
        }
    return out


# --------------------------------------------------------------------------
# 입력 — 프로파일
# --------------------------------------------------------------------------


def synthetic_episode(n_objects: int, k_cap: int, *, instruction_change: bool, ticks: int, change_tick: int | None = None) -> dict[str, Any]:
    """`measure_tokens.synthetic_record`의 2틱을 `ticks`틱으로 늘린 합성 에피소드 — 같은 장면 빌더, commitment는 틱마다 이어진다.

    `instruction_change`면 `change_tick`(기본: 마지막 틱 앞)부터 지시 v2가 실리고 그 틱의 첫 토큰이 지시 조각이 된다.
    """
    if ticks < 1:
        raise ValueError("ticks는 1 이상")
    if change_tick is None:
        change_tick = max(1, ticks - 1)
    if not 1 <= change_tick < ticks:
        raise ValueError(f"change_tick은 1 이상 ticks({ticks}) 미만이어야 한다 (받은 값: {change_tick})")
    hrn = measure_tokens.harness_with_cap(k_cap)
    first = hrn.build_request(measure_tokens.observation(measure_tokens.scene(n_objects)), None, None)
    grasp_key = "grasp:o0:top:zoneL:slow"
    grasp = next((c for c in first["request"]["candidates"]["q_main"] if c["key"] == grasp_key), None)
    exec_history = {
        "adopted": {"main": candidate_id("hold"), "phase": "none", "path": "p0", "speed": 0, "force": 0, "gripper": "open", "stop": False},
        "ack": {"applied": True},
    }
    record = new_episode(
        f"ep-synth-{n_objects}-{k_cap}-{'chg' if instruction_change else 'same'}",
        "scene-family-b2",
        instructions=[dict(measure_tokens.observation(measure_tokens.scene(n_objects))["instruction"])],
        question_set=hrn.question_set_id(),
    )
    append_tick(record, first)
    for index in range(1, ticks):
        observation = measure_tokens.observation(measure_tokens.scene(n_objects), tick=index, sim_time_ms=100 * index)
        if instruction_change and index >= change_tick:
            observation["instruction"] = {
                "version": 2,
                "t_ms": 100 * change_tick,
                "text": "blue 상자를 오른쪽 정리 영역으로 옮기고 green 상자는 건드리지 마라",
            }
        commitment = (
            {"action_ref": grasp["id"], "key": grasp_key, "phase": "approach", "held_ticks": index, "last_switch_tick": 0}
            if grasp
            else None
        )
        append_tick(record, hrn.build_request(observation, exec_history, commitment))
    return record


def _tick_tokens_summary(outs: list[dict[str, Any]], truncate: int | None) -> dict[str, Any]:
    values: list[int] = []
    for out in outs:
        stream = stream_input(out, name="", truncate_tick_tokens=truncate)
        values.extend(len(stream.tick_ids(index)) for index in range(stream.ticks))
    return {"n": len(values), "mean": round(statistics.fmean(values), 1), "p50": percentile(values, 0.5), "p95": percentile(values, 0.95), "max": max(values), "min": min(values)}


def build_profiles(
    tokenizer: Any,
    *,
    ticks: int,
    names: tuple[str, ...] = PROFILES,
    d0_streams: list[dict[str, Any]] | None = None,
    d0_singles: list[dict[str, Any]] | None = None,
    change_tick: int | None = None,
) -> dict[str, dict[str, Any]]:
    """프로파일 이름 → {layout, requests(직렬화 결과), 설명, 토큰 요약}. `state_first`(D0 단일 64건)는 언제나 들어간다."""
    unknown = sorted(set(names) - set(PROFILES))
    if unknown:
        raise ValueError(f"모르는 프로파일: {unknown} (아는 것: {list(PROFILES)})")
    profiles: dict[str, dict[str, Any]] = {}

    def stream_profile(name: str, records: list[dict[str, Any]], description: str, truncate: int | None = None, **extra: Any) -> dict[str, Any]:
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
            "tick_tokens": _tick_tokens_summary(outs, truncate),
            "truncate_tick_tokens": truncate,
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
    needed = [name for name in SYNTHETIC if name in names] + (["upper"] if "v03_target" in names and "upper" not in names else [])
    for name in needed:
        n_objects, k_cap, change = SYNTHETIC[name]
        record = synthetic_episode(n_objects, k_cap, instruction_change=change, ticks=ticks, change_tick=change_tick if change else None)
        synthetic_outs[name] = stream_profile(
            name, [record],
            f"synthetic {n_objects} objects, K={k_cap}, instruction change={'tick ' + str(change_tick if change_tick is not None else max(1, ticks - 1)) if change else 'no'}, {ticks} ticks",
            objects=n_objects, k_cap=k_cap, instruction_change=change,
        )
        if name in names:
            profiles[name] = synthetic_outs[name]
    if "v03_target" in names:
        upper = synthetic_outs["upper"]
        profiles["v03_target"] = {
            **upper,
            "description": f"`upper` stream with every tick truncated to {V03_TICK_TOKENS} tokens — LENGTH-ONLY stand-in for contract v0.3 (not v0.3 formatting)",
            "truncate_tick_tokens": V03_TICK_TOKENS,
            "derived_from": "upper",
            "source_requests": upper["requests"],
            "tick_tokens": _tick_tokens_summary(upper["requests"], V03_TICK_TOKENS),
        }

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


NOTES = [
    "native path = official transformers AutoModelForCausalLM forward (BF16, use_cache=True); the model's own hybrid cache grows without the 30-tick window (a stage-2 property), so every latency carries the cache length before that tick.",
    "stream_cold recomputes prefix + the most recent `cold_window_ticks` ticks, the current one included (the docs/08 window), with an empty cache (stateless bound; 0 = full history) and times only `cold_ticks` ticks.",
    "v03_target is the `upper` stream with each tick's token list truncated to 500 tokens — a length-only stand-in for contract v0.3, not v0.3 formatting.",
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
                        truncate_tick_tokens=profile.get("truncate_tick_tokens"), budgets=settings.budgets,
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
    for candidate in report["candidates"].values():
        for conditions in candidate["conditions"].values():
            for result in conditions.values():
                result["summary"] = summarize_ticks(result.get("ticks") or [], budgets=budgets)
        candidate["verdict"] = verdicts(candidate["conditions"], max_miss_rate=max_miss_rate, budgets=budgets, profiles=profiles)
    stamp = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    report["resummarised_at"] = stamp
    report["resummarised_by"] = {"script_version": SCRIPT_VERSION, "commit": _git_commit()}
    report.setdefault("finished_at", None)
    notes = [note for note in report.get("notes") or [] if not note.startswith(_RESUMMARISED_NOTE) and note != _FINISHED_AT_NOTE]
    notes.append(f"{_RESUMMARISED_NOTE} on {stamp} (script {SCRIPT_VERSION}); environment, settings, profiles, memory and tick records untouched")
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


def print_table(report: dict[str, Any]) -> None:
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
                f"{_gib(chunk.get('v03_target_50k', {}).get('bytes'))} at 50K (estimates, not measurements)"
            )
        measured = {p: {c: _mib(v) for c, v in conds.items()} for p, conds in (memory.get("cache_bytes_measured") or {}).items()}
        if measured:
            print(f"{'':<18}cache bytes at the end (measured): " + "; ".join(f"{p} {conds}" for p, conds in measured.items()))
        for profile, verdict in candidate["verdict"].items():
            print(f"{'':<18}verdict {verdict['text']}")
        print()


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
        from collections import Counter

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
    parser.add_argument("--path", default="native", help="native만 (stream은 2단계)")
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
    parser.add_argument("--report", default=str(DEFAULT_REPORT))
    parser.add_argument("--dtype", default="bf16", choices=sorted(DTYPES))
    parser.add_argument("--root", default=None, help="가중치 보관 디렉터리 (기본: artifacts/models)")
    parser.add_argument("--verify-full", dest="verify_full", action="store_true", help="safetensors 전부의 sha256을 manifest와 대조한다")
    parser.add_argument("--from-report", dest="from_report", help="이 보고서 JSON의 틱 기록으로 요약·판정만 다시 만든다 (GPU·가중치 없이; --report가 없으면 같은 파일에 쓴다)")
    args = parser.parse_args(argv)

    if args.from_report:
        source = Path(args.from_report)
        report = resummarise(json.loads(source.read_text(encoding="utf-8")))
        out = Path(args.report) if args.report != str(DEFAULT_REPORT) else source
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print_table(report)
        print(f"→ {out} (resummarised from {source})")
        return 0

    if args.path == "stream":
        parser.error("--path stream은 2단계(G0b, Task 4의 실제 스트림 경로)다 — 이 스크립트는 native만 잰다")
    if args.path not in PATHS:
        parser.error(f"--path: {list(PATHS)} 중 하나 (받은 값: {args.path!r})")
    if not 0.0 <= args.max_miss_rate <= 1.0:
        parser.error("--max-miss-rate는 0~1")

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
        max_miss_rate=args.max_miss_rate, dtype=args.dtype, verify_full=args.verify_full,
    )
    print(f"[G0a] building profiles ({', '.join(names)}) with {tokenizer_id}, {settings.ticks} ticks/episode", file=sys.stderr, flush=True)
    profiles = build_profiles(tokenizer, ticks=settings.ticks, names=names, change_tick=min(settings.history_ticks + settings.warmup, settings.ticks - 1))

    runner = TransformersRunner(dtype=settings.dtype) if runner is None else runner
    command = "uv run python " + " ".join([str(Path(sys.argv[0]).relative_to(REPO)) if Path(sys.argv[0]).is_absolute() and str(sys.argv[0]).startswith(str(REPO)) else sys.argv[0], *(argv if argv is not None else sys.argv[1:])])
    out = Path(args.report)
    report = screen(entries, profiles, runner, settings=settings, tokenizer_info=tokenizer_info, command=command, checkpoint=out.with_suffix(".partial.json"))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    out.with_suffix(".partial.json").unlink(missing_ok=True)
    print_table(report)
    print(f"→ {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
