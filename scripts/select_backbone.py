"""Task 2b G0b S3.3 — `artifacts/reports/backbone-selection.json`: 후보별 stream 지연(판정)·T0/LoRA 품질·무학습 품질·구간 메모리·귀속을 모아 선정을 적는다.

입력(모두 `artifacts/reports/`): `backbone-stream.json`(baseline, 40편 batch-0), `backbone-stream-levers.json`(지렛대, 앞 8편),
`backbone-stream-fused.json`(fused·all, 40편 batch-0 — 판정의 근거),
`chunk-memory-{2b,4b}.json`, `adapt-{2b,4b}-t0.json`, `adapt-{2b,4b}-lora.json`, `zero-shot-{2b,4b}.json`, `attribution.json`.
없는 파일은 `null`로 적는다(어느 것이 빠졌는지 보인다). 선정 문장은 `--selection-text`(파일)로 준다 — 수치는 JSON이 말하고
사람이 읽는 판단은 보고서·문서에 적는다.

실행: `uv run python scripts/select_backbone.py --selection-text .superpowers/sdd/selection.txt`
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
REPORTS = REPO / "artifacts" / "reports"
CANDIDATES = {"Qwen/Qwen3.5-2B": "2b", "Qwen/Qwen3.5-4B": "4b"}
#: Task P1 파일럿(D1 규모, 같은 step·seed·고정 평가 집합) — `artifacts/reports/p1-{2b,4b}-{mode}.json`와,
#: Task P2가 같은 조건에 **fp32 master weight**만 넣어 다시 돌린 T1(`p2-{2b}-t1-fp32.json`).
PILOT_MODES = {
    "t0": "p1-{short}-t0.json",
    "lora": "p1-{short}-lora.json",
    "t1": "p1-{short}-t1.json",
    "zero-shot": "p1-{short}-zero-shot.json",
    "t1-fp32-master": "p2-{short}-t1-fp32.json",
}
#: 판정 칸 — 편 단위 구간이 여기 붙는다 (Task P2 B2·B3).
DECISION_SPLIT, DECISION_QUESTION = "robot/ood_dev", "q_main"
#: 판정 칸만 다시 평가한 run들 (`configs/eval/pilot-decision-cell.yaml`; 학습 없음, 저장된 checkpoint).
DECISION_CELL_RUNS = {
    "2B T0 (200)": "p2-reeval-2b-t0.json",
    "2B LoRA (40)": "p2-reeval-2b-lora.json",
    "2B T1 bf16 (40)": "p2-reeval-2b-t1.json",
    "4B T0 (200)": "p2-reeval-4b-t0.json",
    "4B LoRA (40)": "p2-reeval-4b-lora.json",
    "2B T1 bf16 5 s (40)": "p2-reeval-2b-t1-bf16-5s.json",
    "2B zero-shot (stride 16)": "p2-reeval-2b-zero-shot.json",
    "4B zero-shot (stride 16)": "p2-reeval-4b-zero-shot.json",
    "2B T1 fp32 master (40)": "p2-2b-t1-fp32.json",
}

#: 판정 칸을 "라벨이 지금 commitment인가"로 가른 표 (`scripts/decision_cell_strata.py`). 집계 여유는 70 %가
#: commitment 반복인 모집단에서 잰 값이라 읽기를 희석한다 — 결정 기록은 그 층화를 함께 들고 있어야 한다.
DECISION_CELL_STRATA = "p2-decision-cell-strata.json"


def _load(name: str) -> dict[str, Any] | None:
    path = REPORTS / name
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


#: `context_shuffle_kind`가 없는 평가 JSON은 그 키를 만들기 전(G0b)의 것이다 — 그때의 로봇 스트림 열은 **지시 텍스트** 대조군,
#: 비로봇 단일 요청 열은 상태 전체 대조군이었다. 지금의 표준 열(id 재매핑 **상태** 섞기)과 같은 것으로 읽으면 안 된다.
LEGACY_CONTEXT_SHUFFLE_KIND = "unrecorded (G0b, key predates the run: robot streams = instruction/text control, non-robot singles = whole-state control)"
#: 대조군 **열이 없는** run — 무학습(학습이 없으니 섞을 것도 없다). 옛 JSON용 이름표를 붙이면 쓰지 않은 대조군을
#: 썼다고 주장하게 된다 (P1 리뷰 1 I9). 종류는 `null`로 두고 왜 없는지를 여기 적는다.
NO_CONTROL_NOTE = "no control column — untrained run (nothing was shuffled); do not read this row against another run's control"


def _eval_summary(evaluation: dict[str, Any] | None) -> dict[str, Any] | None:
    if not evaluation:
        return None
    out: dict[str, Any] = {}
    for split, table in evaluation.items():
        model = table.get("model", {})
        control = table.get("context_shuffle")
        out[split] = {
            # 상태 수와 프롬프트 수는 다른 것이다 — 무학습 표는 (틱 × 질문)마다 프롬프트 하나라 10배쯤 크다.
            "n_states": table.get("n_states"),
            "n_prompts": table.get("n_prompts"),
            "tick_stride": table.get("tick_stride"),
            "accuracy": model.get("_all", {}).get("accuracy"),
            "nll": model.get("_all", {}).get("nll"),
            "brier": model.get("_all", {}).get("brier"),
            "per_question": {q: {k: v for k, v in row.items() if k in ("n", "accuracy", "nll", "brier", "first_position_rate")} for q, row in model.items() if q != "_all"},
            "answer_change_rate": (table.get("answer_change") or {}).get("rate"),
            "context_shuffle_accuracy": (control or {}).get("_all", {}).get("accuracy"),
            # 어느 대조군인지 (D1 리뷰 2 N4): 열이 **있는데** 종류가 없으면 옛 실행의 텍스트 대조군이라고 이름으로
            # 적는다. 열 자체가 없으면(무학습) null로 두고 `control_note`로 왜 없는지를 적는다 (P1 리뷰 1 I9).
            "context_shuffle_kind": (table.get("context_shuffle_kind") or LEGACY_CONTEXT_SHUFFLE_KIND) if control else None,
            "instruction_shuffle_accuracy": (table.get("instruction_shuffle") or {}).get("_all", {}).get("accuracy"),
            "instruction_shuffle_kind": table.get("instruction_shuffle_kind"),
            "rule_judge_accuracy": (table.get("rule_judge") or {}).get("_all", {}).get("accuracy"),
            # 편(에피소드·origin_group) 단위 95 % 구간과 대조군 대비 **쌍** 구간 (Task P2 B2·B3). 구간이 0을
            # 포함하는 여유는 판정이 아니다 — P1은 이 수를 산출물로 낼 수 없었다.
            "episode_bootstrap": table.get("episode_bootstrap"),
        }
        if not control:
            out[split]["control_note"] = NO_CONTROL_NOTE
    return out


def _decision_cell() -> dict[str, Any] | None:
    """판정 칸(`robot/ood_dev` `q_main`)의 run별 모델·대조군·여유와 **편 단위 구간** — 결정 기록이 없던 수다.

    P1의 보고서는 "틱이 독립이면 ±0.023~0.034, 완전히 상관이면 ±0.23~0.35"라는 두 한계만 적을 수 있었다
    (`evaluate_items`가 레코드별 예측을 버렸다). 여기 들어가는 값은 편을 표본 단위로 재표집한 실제 구간이고,
    `margin_includes_zero`가 참인 줄의 여유는 **판정이 아니다** (Task P2 B2·B3).
    """
    rows: dict[str, Any] = {}
    missing: dict[str, Any] = {}
    for label, name in DECISION_CELL_RUNS.items():
        payload = _load(name)
        table = ((payload or {}).get("evaluation") or {}).get("splits", {}).get(DECISION_SPLIT)
        if table is None:
            # 없는 줄은 **이름으로 남긴다** — 조용히 빠지면 6줄짜리 표가 8줄이었던 것처럼 보이지 않는다 (P2 리뷰 1 M10)
            missing[label] = {"report": name, "reason": "report not produced" if payload is None else f"the report has no {DECISION_SPLIT} split"}
            continue
        cell = (table.get("episode_bootstrap") or {}).get(DECISION_QUESTION)
        rows[label] = {
            "report": name,
            "eval_set_sha256": (payload["evaluation"].get("eval_set") or {}).get("sha256"),
            "model_accuracy": (table["model"].get(DECISION_QUESTION) or {}).get("accuracy"),
            "n": (table["model"].get(DECISION_QUESTION) or {}).get("n"),
            "state_shuffle_accuracy": ((table.get("context_shuffle") or {}).get(DECISION_QUESTION) or {}).get("accuracy"),
            "instruction_shuffle_accuracy": ((table.get("instruction_shuffle") or {}).get(DECISION_QUESTION) or {}).get("accuracy"),
            "rule_judge_accuracy": ((table.get("rule_judge") or {}).get(DECISION_QUESTION) or {}).get("accuracy"),
            "episode_bootstrap": cell,
        }
    if not rows:
        return None
    strata = _load(DECISION_CELL_STRATA)
    return {
        "split": DECISION_SPLIT, "question": DECISION_QUESTION,
        "unit": (
            "episode — the 844 ticks come from 8 episodes (94/80/67/79/72/70/300/82, so ep-E1-000235 alone is 35.5 % "
            "of the cell); the independent unit is the episode, not the tick"
        ),
        "reading": (
            "This cell is ~70 % 'repeat your commitment': on 595 of the 844 ticks the expert label IS the tick's own "
            "commitment.action_ref (98.5 % of the 604 ticks that have one), and the state shuffle keeps that line "
            "verbatim, so a policy that reads nothing but the preserved fields scores 751/844 = 0.890. Every whole-cell "
            "margin below is therefore diluted by a stratum that needs no goal. Read `strata` before quoting one."
        ),
        "note": (
            "Each row's controls are that run's own. `episode_bootstrap.state_shuffle.margin_ci` is a PAIRED bootstrap "
            "over episodes (model and its control counted inside the same resample), so it is the interval of the margin "
            "itself; a margin whose interval includes 0 is not a finding. Rows re-evaluated with "
            "configs/eval/pilot-decision-cell.yaml load exactly the same 8 episodes / 844 ticks as configs/eval/pilot.yaml "
            "but turn the permutation column off, so their eval_set hash differs while the scored population does not."
        ),
        "runs": rows,
        "missing": missing,
        "strata": (
            {"source": DECISION_CELL_STRATA, **{key: strata[key] for key in ("reading", "mechanism", "runs", "missing") if key in strata}}
            if strata else {"source": DECISION_CELL_STRATA, "note": "not produced — run scripts/decision_cell_strata.py"}
        ),
    }


def build(selection_text: str | None) -> dict[str, Any]:
    stream = _load("backbone-stream.json")
    levers = _load("backbone-stream-levers.json")
    fused = _load("backbone-stream-fused.json")  # 40편 batch-0에서 fused·all (리뷰 1 I1)
    attribution = _load("attribution.json")
    out: dict[str, Any] = {
        "task": "2b-g0b-selection",
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "gate": "docs/03 §7-6: on `upper` and batch-0, p95 model time ≤ 80 ms and obs→apply miss rate (> 100 ms) ≤ 0.05 → passes_10hz; 5 Hz analogue at 150 ms with the 200 ms deadline",
        "context_shuffle_kind_note": (
            "`context_shuffle_accuracy` is the control column; `context_shuffle_kind` says which control it is. "
            f"Runs whose JSON predates the key are labelled {LEGACY_CONTEXT_SHUFFLE_KIND!r} — for those the robot-stream column is the "
            "instruction/text control (goal line, physical state and candidates kept), not the state shuffle of the current standard column, "
            "so the two are not comparable. New runs carry kind `state` plus a separate `instruction_shuffle_*` column (D1 review 2 N4)."
        ),
        "sources": {
            "stream": "backbone-stream.json" if stream else None, "levers": "backbone-stream-levers.json" if levers else None,
            "attribution": "attribution.json" if attribution else None,
            "fused_40_episodes": "backbone-stream-fused.json" if fused else None,
            "pilot_d1": "p1-{2b,4b}-{t0,lora,t1,zero-shot}.json (Task P1: D1 규모, 같은 step·seed, 고정 평가 집합 configs/eval/pilot.yaml)",
        },
        "candidates": {},
        "selection": selection_text,
    }
    for model_id, short in CANDIDATES.items():
        entry: dict[str, Any] = {"model_id": model_id}
        if stream and model_id in stream["candidates"]:
            candidate = stream["candidates"][model_id]
            entry["stream_baseline"] = {
                "verdict": candidate["verdict"],
                "profiles": {
                    profile: {k: v for k, v in conds["stream_warm"]["summary"].items() if k in ("ticks", "model_ms", "obs_apply_ms", "deadline_miss_rate_100ms", "deadline_miss_rate_200ms", "cache_length", "cache_constant_after_window", "allocated_growth_bytes")}
                    for profile, conds in candidate["conditions"].items()
                },
                "memory": candidate["memory"],
                "attention_backend": candidate["loaded"].get("attention_backend"),
            }
        if levers and model_id in levers["candidates"]:
            candidate = levers["candidates"][model_id]
            entry["stream_levers"] = {
                lever: {
                    "settings": block["settings"],
                    "verdict": {k: {"p50_model_ms": v.get("p50_model_ms"), "p95_model_ms": v.get("p95_model_ms"), "deadline_miss_rate_100ms": v.get("deadline_miss_rate_100ms"), "passes_10hz": v.get("passes_10hz"), "passes_5hz": v.get("passes_5hz")} for k, v in block["verdict"].items() if k != "overall"},
                    "overall": block["verdict"].get("overall"),
                    "compile_seconds": block["loaded"].get("compile_seconds"),
                }
                for lever, block in candidate["levers"].items()
            }
        if fused and model_id in fused["candidates"]:
            candidate = fused["candidates"][model_id]
            entry["stream_fused_40_episodes"] = {
                lever: {
                    "settings": block["settings"],
                    "profiles": {
                        profile: {k: v for k, v in conds["stream_warm"]["summary"].items() if k in ("ticks", "model_ms", "obs_apply_ms", "deadline_miss_rate_100ms", "deadline_miss_rate_200ms", "cache_length", "cache_constant_after_window", "allocated_growth_bytes")}
                        for profile, conds in block["conditions"].items()
                    },
                    "verdict": {k: {"p50_model_ms": v.get("p50_model_ms"), "p95_model_ms": v.get("p95_model_ms"), "deadline_miss_rate_100ms": v.get("deadline_miss_rate_100ms"), "passes_10hz": v.get("passes_10hz"), "passes_5hz": v.get("passes_5hz")} for k, v in block["verdict"].items() if k != "overall"},
                    "overall": block["verdict"].get("overall"),
                }
                for lever, block in candidate["levers"].items()
            }
        chunk = _load(f"chunk-memory-{short}.json")
        entry["chunk_memory"] = None if chunk is None else {"episode": chunk.get("episode"), "runs": chunk.get("runs")}
        for mode in ("t0", "lora"):
            adapt = _load(f"adapt-{short}-{mode}.json")
            entry[mode] = None if adapt is None else {
                "steps": adapt.get("steps"), "status": adapt.get("status"), "loss_first_last": adapt.get("loss_first_last"), "step_seconds": adapt.get("step_seconds"),
                "train_seconds": adapt.get("train_seconds"), "memory": adapt.get("memory"), "checkpoint": adapt.get("checkpoint"), "contract_sha256": (adapt.get("manifest") or {}).get("contract_sha256"),
                "evaluation": _eval_summary(adapt.get("evaluation")),
            }
        pilot: dict[str, Any] = {}
        for mode, pattern in PILOT_MODES.items():
            run = _load(pattern.format(short=short))
            if run is None:
                continue
            evaluation = run.get("evaluation") or {}
            pilot[mode] = {
                "steps": run.get("steps"), "status": run.get("status"), "loss_first_last": run.get("loss_first_last"),
                "step_seconds": run.get("step_seconds"), "train_seconds": run.get("train_seconds"), "tokens": run.get("tokens"),
                "memory": run.get("memory"), "checkpoint": run.get("checkpoint"), "contract_sha256": (run.get("manifest") or {}).get("contract_sha256"),
                "train_config": run.get("train_config"), "eval_config": run.get("eval_config"),
                "eval_set_sha256": (evaluation.get("eval_set") or {}).get("sha256"),
                "evaluation": _eval_summary(evaluation.get("splits")),
            }
        entry["pilot_d1"] = pilot or None
        zero = _load(f"zero-shot-{short}.json")
        entry["zero_shot"] = None if zero is None else {
            split: {"prompts": r.get("prompts"), "accuracy": r["table"]["_all"].get("accuracy"), "nll": r["table"]["_all"].get("nll"), "per_question": {q: {k: v for k, v in row.items() if k in ("n", "accuracy", "nll", "first_position_rate")} for q, row in r["table"].items() if q != "_all"}}
            for split, r in zero.get("splits", {}).items()
        }
        if attribution and model_id in attribution.get("candidates", {}):
            a = attribution["candidates"][model_id]
            entry["attribution"] = {"tokens": a.get("tokens"), "eager_ms": a.get("eager_ms"), "profiler": {k: v for k, v in a.get("profiler", {}).items() if k != "top_kernels_us_per_forward"}, "graph": a.get("graph"), "weight_read_lower_bound_ms": a.get("weight_read_lower_bound_ms"), "decomposition": a.get("decomposition")}
        out["candidates"][model_id] = entry
    out["decision_cell"] = _decision_cell()
    if attribution and "Qwen/Qwen3.5-9B" in attribution.get("candidates", {}):
        a = attribution["candidates"]["Qwen/Qwen3.5-9B"]
        out["attribution_9b"] = {"eager_ms": a.get("eager_ms"), "profiler": {k: v for k, v in a.get("profiler", {}).items() if k != "top_kernels_us_per_forward"}, "graph": a.get("graph"), "weight_read_lower_bound_ms": a.get("weight_read_lower_bound_ms"), "decomposition": a.get("decomposition")}
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--selection-text", dest="selection_text", default=None, help="선정 문단이 든 텍스트 파일")
    parser.add_argument("--out", default=str(REPORTS / "backbone-selection.json"))
    args = parser.parse_args(argv)
    text = Path(args.selection_text).read_text(encoding="utf-8").strip() if args.selection_text else None
    report = build(text)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"→ {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
