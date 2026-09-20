"""Task 2b G0b S2.5 — 단일 요청 forward의 귀속 측정: CUDA kernel 시간 vs 빈틈(launch), CUDA graph 재생, weight-read 하한 (2B/4B/9B).

2b-G0a 리뷰 Q3(HANDOFF 결정 3의 미결): `state_first`(≈186토큰, cache 없음) 한 요청의 forward 절편이 **launch-bound**(CPU가
kernel을 띄우는 사이 GPU가 노는 시간)인지 **weight-read**(가중치를 HBM/LPDDR에서 한 번 읽는 시간)인지 잰다.

* eager: 공식 forward를 CUDA event로 잰다(예열 뒤 반복 평균).
* profiler: `torch.profiler`(CUDA activity)로 같은 forward의 kernel 시간 합과 kernel 수를 세고, 빈틈 = eager − kernel 합.
* graph: 같은 forward를 `torch.cuda.CUDAGraph`로 캡처해 재생한다 — launch 빈틈이 사라진 시간. 캡처가 안 되면 예외를 적는다.
* weight-read 하한 = 가중치 바이트 / 장치 대역폭(GB10 273 GB/s, docs/03) — graph 시간이 이 값에 가까우면 weight-read가 절편이다.

실행: `uv run python scripts/attribution.py --candidates Qwen/Qwen3.5-2B,Qwen/Qwen3.5-4B,Qwen/Qwen3.5-9B --out artifacts/reports/attribution.json`
"""

from __future__ import annotations

import argparse
import datetime as dt
import gc
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any

from robo_jev.gpu import DEFAULT_FRACTION, limit_gpu_memory, memory_report

REPO = Path(__file__).resolve().parents[1]
D0 = REPO / "tests" / "fixtures" / "d0.jsonl"
#: GB10의 통합 메모리 대역폭 (docs/03 §"지연 예산": ≈273 GB/s) — weight-read 하한의 분모.
BANDWIDTH_GBPS = 273.0


def pick_request(tokenizer: Any, target_tokens: int) -> dict[str, Any]:
    from robo_jev.model.serialize import serialize_request

    records = [json.loads(line) for line in D0.read_text(encoding="utf-8").splitlines() if line.strip()]
    outs = [serialize_request(record, tokenizer) for record in records]
    return min(outs, key=lambda out: abs(len(out["tokens"]) - target_tokens))


def measure(model_id: str, request: dict[str, Any], *, repeats: int = 20, warmup: int = 5) -> dict[str, Any]:
    import torch
    from torch.profiler import ProfilerActivity, profile

    from robo_jev.model.backbone_qwen import QwenBackbone

    gc.collect()
    torch.cuda.empty_cache()
    backbone = QwenBackbone.load(model_id)
    model = backbone.text
    ids = torch.tensor([request["tokens"]], dtype=torch.long, device="cuda")
    positions = torch.tensor([request["position"]], dtype=torch.long, device="cuda")
    weight_bytes = int(sum(p.numel() * p.element_size() for p in backbone.model.parameters()))
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

    def forward() -> Any:
        return model(input_ids=ids, position_ids=positions, use_cache=False).last_hidden_state

    with torch.no_grad():
        for _ in range(warmup):
            forward()
        torch.cuda.synchronize()
        eager: list[float] = []
        walls: list[float] = []
        for _ in range(repeats):
            wall = time.perf_counter()
            start.record()
            forward()
            end.record()
            torch.cuda.synchronize()
            walls.append((time.perf_counter() - wall) * 1e3)
            eager.append(start.elapsed_time(end))
        # profiler: kernel 시간 합 vs 빈틈
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            for _ in range(3):
                forward()
            torch.cuda.synchronize()
        kernel_us = 0.0
        kernels = 0
        by_name: dict[str, float] = {}
        cpu_op_us = 0.0
        for event in prof.events():
            if event.device_type == torch.autograd.DeviceType.CUDA:
                kernel_us += event.self_device_time_total if hasattr(event, "self_device_time_total") else event.self_cuda_time_total
                kernels += 1
                by_name[event.name] = by_name.get(event.name, 0.0) + (event.self_device_time_total if hasattr(event, "self_device_time_total") else event.self_cuda_time_total)
            elif event.device_type == torch.autograd.DeviceType.CPU and event.self_cpu_time_total:
                cpu_op_us += event.self_cpu_time_total
        kernel_ms = kernel_us / 3 / 1e3
        top = sorted(by_name.items(), key=lambda item: -item[1])[:8]
        # CUDA graph
        graph_result: dict[str, Any]
        try:
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(3):
                    forward()
            torch.cuda.current_stream().wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                captured = forward()
            graph.replay()
            torch.cuda.synchronize()
            reference = forward()
            diff = float((captured.float() - reference.float()).abs().max())
            replays: list[float] = []
            for _ in range(repeats):
                start.record()
                graph.replay()
                end.record()
                torch.cuda.synchronize()
                replays.append(start.elapsed_time(end))
            graph_result = {"ok": True, "ms_mean": round(statistics.fmean(replays), 2), "ms_p50": round(statistics.median(replays), 2), "max_abs_diff_vs_eager": diff}
            del graph
        except Exception as exc:  # noqa: BLE001 — 캡처 실패 자체가 결과
            graph_result = {"ok": False, "error": f"{type(exc).__name__}: {str(exc)[:300]}"}
    eager_mean = statistics.fmean(eager)
    weight_read_ms = weight_bytes / (BANDWIDTH_GBPS * 1e9) * 1e3
    result = {
        "model_id": model_id,
        "tokens": len(request["tokens"]),
        "weight_bytes": weight_bytes,
        "eager_ms": {"mean": round(eager_mean, 2), "p50": round(statistics.median(eager), 2), "min": round(min(eager), 2)},
        "wall_ms_mean": round(statistics.fmean(walls), 2),
        "profiler": {"kernel_ms_per_forward": round(kernel_ms, 2), "kernels_per_forward": kernels // 3, "gap_ms": round(eager_mean - kernel_ms, 2), "gap_share": round(1.0 - kernel_ms / eager_mean, 3), "cpu_op_ms_per_forward": round(cpu_op_us / 3 / 1e3, 2), "top_kernels_us_per_forward": [(name, round(us / 3, 1)) for name, us in top]},
        "graph": graph_result,
        "weight_read_lower_bound_ms": round(weight_read_ms, 2),
        "bandwidth_gbps_assumed": BANDWIDTH_GBPS,
    }
    if graph_result.get("ok"):
        g = graph_result["ms_mean"]
        result["decomposition"] = {
            "launch_gap_ms": round(eager_mean - g, 2),
            "launch_gap_share_of_eager": round((eager_mean - g) / eager_mean, 3),
            "graph_ms": g,
            "weight_read_share_of_graph": round(weight_read_ms / g, 3),
            "reading": ("launch-bound" if (eager_mean - g) / eager_mean > 0.5 else "weight/compute-bound") + f" — eager {eager_mean:.1f} ms = graph {g:.1f} ms + launch gaps {eager_mean - g:.1f} ms; weight-read lower bound {weight_read_ms:.1f} ms ({weight_read_ms / g:.0%} of the graph time)",
        }
    del backbone, model
    gc.collect()
    torch.cuda.empty_cache()
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--candidates", default="Qwen/Qwen3.5-2B,Qwen/Qwen3.5-4B,Qwen/Qwen3.5-9B")
    parser.add_argument("--tokens", type=int, default=186, help="D0 단일 요청 가운데 이 토큰 수에 가장 가까운 것")
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--out", default=str(REPO / "artifacts" / "reports" / "attribution.json"))
    parser.add_argument("--gpu-memory-fraction", dest="gpu_memory_fraction", type=float, default=DEFAULT_FRACTION, help="프로세스가 쓸 장치(통합) 메모리 몫 (robo_jev.gpu; 첫 CUDA 할당 전에 건다)")
    args = parser.parse_args(argv)
    from robo_jev.model.tokenizer import available_tokenizer, load_tokenizer

    guard = limit_gpu_memory(args.gpu_memory_fraction)
    memory_start = memory_report()
    print(f"[attribution] gpu guard {guard} · memory at start {memory_start}", file=sys.stderr, flush=True)

    tokenizer = load_tokenizer(available_tokenizer()[0])
    request = pick_request(tokenizer, args.tokens)
    report: dict[str, Any] = {"task": "2b-g0b-attribution", "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"), "request": {"request_id": request.get("request_id"), "tokens": len(request["tokens"])}, "candidates": {}}
    for model_id in [item.strip() for item in args.candidates.split(",") if item.strip()]:
        print(f"[attribution] {model_id}", file=sys.stderr, flush=True)
        result = measure(model_id, request, repeats=args.repeats)
        report["candidates"][model_id] = result
        print(f"[attribution] {model_id}: eager {result['eager_ms']['mean']} ms, kernels {result['profiler']['kernel_ms_per_forward']} ms ({result['profiler']['kernels_per_forward']} launches), gap {result['profiler']['gap_ms']} ms, graph {result['graph'].get('ms_mean')} ms, weight-read ≥ {result['weight_read_lower_bound_ms']} ms", file=sys.stderr, flush=True)
    report["gpu"] = {"guard": guard, "memory_at_start": memory_start, "memory_at_end": memory_report()}
    print(f"[attribution] memory at end {report['gpu']['memory_at_end']}", file=sys.stderr, flush=True)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"→ {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
