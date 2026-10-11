"""bench-v0 점수 (Task bench0 §3) — 진단 도구의 지표(:mod:`robo_jev.decisions.metrics`, :mod:`robo_jev.decisions.single_probe`)를 그대로
불러 세트·모델마다 v0 보고(`.superpowers/sdd/v0-results.md`)와 같은 지표를 낸다. 새 지표 계산은 T6 불변성(v0 보고의
`artifacts/scratch/v0-compare/t6_invariance.py`를 옮긴 것)과 세트를 묶는 합(T2+T3+T4)뿐이다.

**지표** (세트마다, 두 읽기):

* **값 없음 — 물리 결과 예측**(머리 지표): 예/아니오(noul) Brier, 점수(score) Brier와 그 "언제나 0.5" 기준선, Brier skill
  = 1 − Brier / Brier(0.5). T7은 ECE(noul·score·상태).
* **값 있음** — 행동 정확도(기대 비용 최소), 후회(기본 기준 / 적힌 기준). 비용은 우리 가치표 values-v0이고 API 모델은 그 가치표를
  모른다 — 행동 칸은 "우리 가치표 아래에서"로 읽는다.
* 상태·규칙 정확도, 틀린·옳은 권고 따르기, 전체 KL, 답한 질문의 비율(``coverage``).
* T5 — 숫자 쌍의 방향 정답률(정답이 움직인 질문에서 모델도 같은 쪽으로), 문구 쌍의 흔들림. T6 — 교란별 평균 이동(:func:`t6_invariance`).
* G1·G2 — 정확도(choice = argmax가 최선 집합 안, noul = 0.5의 같은 쪽)와 Brier, 분야별(:func:`probe_summary`).

**공정성** — API 모델은 저장된 순서 하나로 한 번 묻는다(질문마다 listing 하나: 두 읽기가 같다). 우리 모델의 "한 순서"는 균형 배치
listing들의 기대값(배치마다 1/L)이고 "순서 평균"은 배치 평균 답이다. 표는 한 순서 읽기를 쓴다(배포 조건).
"""

from __future__ import annotations

import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from robo_jev.decisions.metrics import aggregate, compare_runs, question_rows, summarize_all
from robo_jev.decisions.records import file_sha256, read_records
from robo_jev.decisions.single_probe import probe_values

__all__ = [
    "PHYSICS_SETS",
    "fake_api_predictions",
    "UNSEEN_SETS",
    "pilot_report",
    "pooled_brier",
    "probe_summary",
    "readme_table",
    "render_pilot",
    "resolve_ours",
    "set_metrics",
    "subset_predictions",
    "t6_invariance",
]

PHYSICS_SETS = ("T1", "T2", "T3", "T4", "T5", "T6", "T7")
GENERAL_SETS = ("G1", "G2")
#: 처음 보는 물리 — 새 문체·설정·문구(T2), 범위 밖(T3), 처음 보는 조합(T4). 그림 1의 x축.
UNSEEN_SETS = ("T2", "T3", "T4")
#: 값이 붙지 않는 불변 교란 — 정답이 바뀌는 key_change와 정답이 조금 바뀌는 mirror를 뺀 열 가지 (v0 보고와 같다).
VARIANT_KINDS_CHANGING = ("key_change", "mirror")
BIG_SHIFT = 0.2


# --------------------------------------------------------------------------
# 예측 고르기
# --------------------------------------------------------------------------


def subset_predictions(path: str | Path, ids: Iterable[str]) -> list[dict[str, Any]]:
    """예측 파일에서 벤치 id만 (벤치 순서)."""
    order = [str(i) for i in ids]
    wanted = set(order)
    found = {str(p["id"]): p for p in read_records(path) if str(p["id"]) in wanted}
    return [found[i] for i in order if i in found]


def resolve_ours(entry: dict[str, Any], set_name: str, *, root: str | Path) -> Path | None:
    """정의의 ``ours`` 항목 하나와 세트 → 처음 있는 후보 경로 (없으면 None). T 세트는 ``T``, G 세트는 ``G`` 목록을 본다."""
    candidates = entry.get("G" if set_name.startswith("G") else "T") or []
    for template in candidates:
        path = Path(root) / template.format(set=set_name)
        if path.is_file():
            return path
    return None


# --------------------------------------------------------------------------
# 세트 하나
# --------------------------------------------------------------------------


def _get(tree: Any, *keys: str) -> Any:
    for key in keys:
        if not isinstance(tree, dict) or key not in tree:
            return None
        tree = tree[key]
    return tree


def _skill(brier: float | None, uniform: float | None) -> float | None:
    if brier is None or not uniform:
        return None
    return 1.0 - brier / uniform


def _reading_metrics(reading: dict[str, Any]) -> dict[str, Any]:
    overall = reading["overall"]
    counts = reading["counts"]
    out: dict[str, Any] = {
        "records": counts["records"], "questions": counts["questions"], "scored": counts["scored"],
        "coverage": counts["scored"] / counts["questions"] if counts["questions"] else None,
    }
    for block in ("noul", "score"):
        entry = overall.get(block) or {}
        out[f"{block}_n"] = entry.get("n", 0)
        out[f"{block}_brier"] = entry.get("brier")
        out[f"{block}_brier_uniform"] = _get(entry, "uniform", "brier")
        out[f"{block}_skill"] = _skill(entry.get("brier"), _get(entry, "uniform", "brier"))
        out[f"{block}_ece"] = entry.get("ece")
        out[f"{block}_mae"] = entry.get("mae")
        out[f"{block}_pearson"] = entry.get("pearson")
    out["state_ece"] = _get(overall, "choice/state", "ece")
    out["action_n"] = _get(overall, "choice/action", "n") or 0
    out["action_accuracy"] = _get(overall, "choice/action", "accuracy")
    out["regret_default"] = _get(reading, "regret_by_criterion", "default", "regret_mean")
    out["regret_stated"] = _get(reading, "regret_by_criterion", "stated", "regret_mean")
    out["state_accuracy"] = _get(overall, "choice/state", "accuracy")
    out["rule_accuracy"] = _get(overall, "choice/rule", "accuracy")
    out["follow_wrong"] = _get(reading, "recommendation", "wrong", "follow_rate")
    out["follow_right"] = _get(reading, "recommendation", "right", "follow_rate")
    out["kl_all"] = _get(overall, "_all", "kl")
    out["direction_correct"] = _get(reading, "pairs", "numeric", "direction_correct_rate")
    out["numeric_moved"] = _get(reading, "pairs", "numeric", "moved")
    out["wording_shift"] = _get(reading, "pairs", "wording", "model_shift_mean")
    out["wording_flip"] = _get(reading, "pairs", "wording", "flip_rate")
    out["ci"] = {name: reading["bootstrap"][name] for name in ("noul.brier", "score.brier", "choice/action.accuracy",
                                                              "choice/action.regret_mean[default]") if name in reading.get("bootstrap", {})}
    return out


def probe_summary(records: list[dict[str, Any]], predictions: list[dict[str, Any]], *, reading: str) -> dict[str, Any]:
    """G1·G2: 질문마다 :func:`robo_jev.decisions.single_probe.probe_values` 의 정확도·Brier → 평균(질문마다 같은 무게; ``convert_single_probe.py
    compare``의 ``_all``과 같다)과 분야별, 답한 질문의 비율."""
    values = probe_values(records, predictions, reading=reading)
    total = sum(len(r["request"]["questions"]) for r in records)

    def mean(items: list[dict[str, Any]], key: str) -> float | None:
        found = [v[key] for v in items if v.get(key) is not None]
        return sum(found) / len(found) if found else None

    items = list(values.values())
    by_domain: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in items:
        by_domain[item["groups"]["domain"]].append(item)
    return {
        "questions": total, "scored": len(items), "coverage": len(items) / total if total else None,
        "accuracy": mean(items, "accuracy"), "brier": mean(items, "brier"), "chance_accuracy": mean(items, "chance"),
        "by_domain": {name: {"questions": len(group), "accuracy": mean(group, "accuracy"), "brier": mean(group, "brier")}
                      for name, group in sorted(by_domain.items())},
    }


def set_metrics(set_name: str, records: list[dict[str, Any]], predictions: list[dict[str, Any]], *, n_boot: int = 1000, seed: int = 0) -> dict[str, Any]:
    """세트 하나의 지표 (두 읽기). T 세트는 :func:`summarize_all` (진단 도구의 ``--rescore``와 같은 계산), G 세트는 :func:`probe_summary`.
    예측이 없는 레코드도 분모에 남는다(``coverage``)."""
    if set_name.startswith("G"):
        return {"kind": "general", "readings": {r: probe_summary(records, predictions, reading=r) for r in ("single_order", "order_averaged")}}
    summary = summarize_all(records, predictions, n_boot=n_boot, seed=seed)
    out: dict[str, Any] = {"kind": "physics", "readings": {r: _reading_metrics(summary["readings"][r]) for r in summary["readings"]}}
    permutation = _get(summary, "order", "permutation", "_all")
    out["order_sensitivity"] = {"shift_mean": _get(permutation, "shift_mean"), "flip_rate": _get(permutation, "flip_rate"),
                                "decisive_flip_rate": _get(permutation, "decisive_flip_rate")}
    if set_name == "T6":
        out["t6"] = t6_invariance(records, predictions)
    return out


# --------------------------------------------------------------------------
# T6 불변성 (v0 보고의 artifacts/scratch/v0-compare/t6_invariance.py)
# --------------------------------------------------------------------------


def _averaged(question: dict[str, Any]) -> Any:
    listings = question.get("listings") or [question]
    n = len(listings)
    if question["type"] == "noul":
        return sum(float(l["answer"]["p_yes"]) for l in listings) / n
    if question["type"] == "score":
        return sum(float(l["answer"]["value"]) for l in listings) / n
    out: dict[str, float] = defaultdict(float)
    for listing in listings:
        for key, value in listing["answer"]["probabilities"].items():
            out[key] += float(value) / n
    return dict(out)


def _label(label: dict[str, Any]) -> Any:
    if label["type"] == "noul":
        return float(label["p_yes"])
    if label["type"] == "score":
        return float(label["value"])
    return {k: float(v) for k, v in label["probabilities"].items()}


def _delta(a: Any, b: Any) -> Any:
    if isinstance(a, dict):
        return {k: a.get(k, 0.0) - b.get(k, 0.0) for k in set(a) | set(b)}
    return a - b


def _size(d: Any) -> float:
    return 0.5 * sum(abs(v) for v in d.values()) if isinstance(d, dict) else abs(d)


def _excess(dm: Any, dl: Any) -> float:
    if isinstance(dm, dict):
        return 0.5 * sum(abs(dm.get(k, 0.0) - dl.get(k, 0.0)) for k in set(dm) | set(dl))
    return abs(dm - dl)


def t6_invariance(records: list[dict[str, Any]], predictions: list[dict[str, Any]]) -> dict[str, Any]:
    """T6: 변형 레코드마다 ``meta.t6.map``(변형 질문 → 기준 질문)의 짝에서 순서 평균 답의 이동 |Δ|(choice는 전변동 거리)과
    정답의 이동, 초과 이동 |Δ모델 − Δ정답|. 교란 종류별 평균과, 정답이 바뀌지 않는 열 가지(``key_change``·``mirror`` 제외)의 합.
    기준과 변형 모두 답이 있는 짝만 센다."""
    by_id = {str(r["id"]): r for r in records}
    answers = {str(p["id"]): {name: _averaged(q) for name, q in p["questions"].items()} for p in predictions}
    stats: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for rid, record in by_id.items():
        t6 = (record.get("meta") or {}).get("t6") or {}
        if t6.get("variant") in (None, "base") or t6.get("identical_to_base"):
            continue
        base = by_id.get(str(t6.get("base")))
        if base is None or rid not in answers or str(base["id"]) not in answers:
            continue
        for vq, bq in (t6.get("map") or {}).items():
            if bq is None or vq not in answers[rid] or bq not in answers[str(base["id"])]:
                continue
            dl = _delta(_label(record["labels"][vq]), _label(base["labels"][bq]))
            dm = _delta(answers[rid][vq], answers[str(base["id"])][bq])
            kind = str(t6["variant"])
            stats[kind]["label"].append(_size(dl))
            stats[kind]["shift"].append(_size(dm))
            stats[kind]["excess"].append(_excess(dm, dl))

    def summary(values: dict[str, list[float]]) -> dict[str, Any]:
        shift = values.get("shift") or []
        return {"n": len(shift), "label_change": statistics.fmean(values["label"]) if values.get("label") else None,
                "shift_mean": statistics.fmean(shift) if shift else None, "excess_mean": statistics.fmean(values["excess"]) if values.get("excess") else None,
                "big_shift_share": (sum(1 for x in shift if x >= BIG_SHIFT) / len(shift)) if shift else None}

    kinds = {kind: summary(values) for kind, values in sorted(stats.items())}
    invariant: dict[str, list[float]] = defaultdict(list)
    for kind, values in stats.items():
        if kind in VARIANT_KINDS_CHANGING:
            continue
        for key, xs in values.items():
            invariant[key].extend(xs)
    return {"kinds": kinds, "invariant": summary(invariant), "invariant_kinds": sorted(k for k in stats if k not in VARIANT_KINDS_CHANGING)}


# --------------------------------------------------------------------------
# 묶음·표
# --------------------------------------------------------------------------


def pooled_brier(per_set: dict[str, Any], sets: Iterable[str] = UNSEEN_SETS, *, reading: str = "single_order",
                 blocks: Iterable[str] = ("noul", "score")) -> dict[str, Any] | None:
    """세트들의 noul·score 질문을 한데 모은 Brier(질문마다 같은 무게)와 "언제나 0.5"의 Brier, skill = 1 − Brier / Brier(0.5).
    세트 하나라도 없으면 None."""
    total = uniform = n = 0.0
    used = []
    for name in sets:
        entry = per_set.get(name)
        if not entry:
            return None
        metrics = entry["readings"][reading]
        for block in blocks:
            count = metrics.get(f"{block}_n") or 0
            if not count or metrics.get(f"{block}_brier") is None:
                continue
            total += metrics[f"{block}_brier"] * count
            uniform += metrics[f"{block}_brier_uniform"] * count
            n += count
        used.append(name)
    if not n:
        return None
    return {"sets": used, "blocks": list(blocks), "questions": int(n), "brier": total / n, "uniform_brier": uniform / n, "skill": 1.0 - total / uniform}


def _fmt(value: Any, digits: int = 3) -> str:
    return "—" if value is None or (isinstance(value, float) and math.isnan(value)) else f"{value:.{digits}f}"


def readme_table(results: dict[str, Any], *, date: str) -> str:
    """README용 Markdown 표 — 모델, 처음 보는 물리(T2+T3+T4)의 결과 예측, 일반 읽기, $/M 입력 토큰. 사실과 주의만."""
    lines = [
        "| Model | Physics skill T2–T4 ↑ | Yes/no Brier T2–T4 ↓ | Score Brier T2–T4 ↓ | G1 acc ↑ | G2 acc ↑ | $/M input |",
        "|---|---|---|---|---|---|---|",
    ]
    for model in results["models"]:
        per_set = model["sets"]
        both = pooled_brier(per_set)
        noul = pooled_brier(per_set, blocks=("noul",))
        score = pooled_brier(per_set, blocks=("score",))
        g1 = _get(per_set, "G1", "readings", "single_order", "accuracy")
        g2 = _get(per_set, "G2", "readings", "single_order", "accuracy")
        price = model.get("price_per_m_input")
        name = f"**{model['label']}**" if model.get("role") in ("ours", "baseline") else model["label"]
        lines.append(f"| {name} | {_fmt(both and both['skill'])} | {_fmt(noul and noul['brier'])} | {_fmt(score and score['brier'])} | "
                     f"{_fmt(g1)} | {_fmt(g2)} | {'self-hosted' if price is None else f'{price:.3f}'} |")
    lines += [
        "",
        f"Evaluated {date} on bench-v0 (`configs/bench/bench-v0.yaml`): T1–T7 are sealed test sets from our own physics-judgment generator "
        "(home ground — every set shares that generator), G1 is the old non-robot reading probe, G2 is public BoolQ + MultiNLI (1,000 questions). "
        "Physics skill = 1 − Brier / Brier(always 0.5) over the yes/no and score questions of T2–T4 (new style, out-of-range, unseen combination). "
        "OpenRouter models were asked once, in the stored option order, through the Decisions API (score questions as an 11-level 0.0–1.0 scale); "
        "our models' numbers are the expectation over balanced option orders (single-order reading). Action metrics use our value table and are not shown here.",
    ]
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------
# 시운전 (score 형식)
# --------------------------------------------------------------------------


def _score_block(records: list[dict[str, Any]], predictions: list[dict[str, Any]]) -> dict[str, Any]:
    rows = [r for r in question_rows(records, predictions, reading="single_order") if r.get("m") is not None and r["type"] == "score"]
    block = aggregate(rows).get("score") or {}
    keys = ("n", "brier", "mae", "pearson", "side_agreement", "mean_p", "mean_y", "ece")
    out = {key: block.get(key) for key in keys}
    out["uniform_brier"] = _get(block, "uniform", "brier")
    out["skill"] = _skill(block.get("brier"), out["uniform_brier"])
    return out


def pilot_report(
    records: list[dict[str, Any]], predictions_11: list[dict[str, Any]], predictions_2: list[dict[str, Any]], *,
    estimates: dict[int, dict[str, dict[str, Any]]], bench_tokens: dict[str, int], price: dict[str, float], n_boot: int = 1000,
) -> dict[str, Any]:
    """score 형식 시운전 한 모델: 단계 둘(그대로) 대 11단계의 score 지표, 짝지은 차(11 − 2, 레코드 bootstrap — 진단 도구의 compare_runs),
    실제 입력 토큰 대 견적과 그 비율로 본 본 실행 비용 예측."""
    out: dict[str, Any] = {"variants": {}}
    for levels, predictions in ((2, predictions_2), (11, predictions_11)):
        block = _score_block(records, predictions)
        actual = sum(int(_get(p, "api", "usage", "input_tokens") or 0) for p in predictions)
        ids = [str(p["id"]) for p in predictions]
        est_low = sum(estimates[levels][i]["low"] for i in ids if i in estimates[levels])
        est_high = sum(estimates[levels][i]["high"] for i in ids if i in estimates[levels])
        cost = sum(float(_get(p, "api", "usage", "cost") or 0.0) for p in predictions)
        block.update({"requests": len(predictions), "input_tokens": actual, "estimate_low": est_low, "estimate_high": est_high,
                      "actual_over_high": (actual / est_high) if est_high else None, "cost_usd": cost,
                      "resolved_models": sorted({_get(p, "api", "model") for p in predictions if _get(p, "api", "model")})})
        out["variants"][f"score{levels}"] = block
    comparison = compare_runs(records, predictions_11, predictions_2, n_boot=n_boot, seed=0)
    out["difference_11_minus_2"] = _get(comparison, "readings", "single_order", "differences", "score.brier")
    ratio = out["variants"]["score11"].get("actual_over_high")
    out["projected_bench_cost_usd"] = (ratio * bench_tokens["high"] * price.get("prompt", 0.0)) if ratio is not None else None
    out["estimated_bench_cost_usd"] = {"low": bench_tokens["low"] * price.get("prompt", 0.0), "high": bench_tokens["high"] * price.get("prompt", 0.0)}
    return out


def render_pilot(payload: dict[str, Any]) -> str:
    lines = [
        "# bench-v0 pilot — score format: two anchors as-is vs 11 levels", "",
        f"{payload['records']} T1 records outside the bench; only the score questions were sent. Score Brier against the posterior label "
        "(lower is better); skill = 1 − Brier / Brier(always 0.5). Δ = 11 levels − two anchors with a paired record-bootstrap 95 % interval. "
        "Tokens: the provider's reported input tokens vs our estimate (high end = the state counted once per question).", "",
        "| model | Brier (2) | Brier (11) | Δ 11 − 2 [95 %] | skill (2) | skill (11) | r (2) | r (11) | mean value (11) / label | input tokens (11) / estimate high | projected bench $ |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for model, report in payload["models"].items():
        two, eleven = report["variants"]["score2"], report["variants"]["score11"]
        diff = report.get("difference_11_minus_2") or {}
        ci = f"{diff.get('value', float('nan')):+.3f} [{diff.get('low', float('nan')):+.3f}, {diff.get('high', float('nan')):+.3f}]" if diff else "—"
        lines.append(f"| {model} | {_fmt(two['brier'])} | {_fmt(eleven['brier'])} | {ci} | {_fmt(two['skill'])} | {_fmt(eleven['skill'])} | "
                     f"{_fmt(two['pearson'], 2)} | {_fmt(eleven['pearson'], 2)} | {_fmt(eleven['mean_p'])} / {_fmt(eleven['mean_y'])} | "
                     f"{eleven['input_tokens']:,} / {eleven['estimate_high']:,} | {_fmt(report.get('projected_bench_cost_usd'), 4)} |")
    return "\n".join(lines) + "\n"


def describe_source(path: Path) -> dict[str, Any]:
    return {"path": str(path), "sha256": file_sha256(path)}


# --------------------------------------------------------------------------
# 가짜 API 예측 (그래프·점수 장치 시험 전용)
# --------------------------------------------------------------------------


def fake_api_predictions(records: list[dict[str, Any]], model: str, *, strength: float, seed: int, coverage: float = 1.0) -> list[dict[str, Any]]:
    """**시험 전용 가짜** API 예측 — 정답 쪽으로 `strength`만큼 당긴 잡음 답(0 = 무작위, 1 = 정답). 모델 이름에 ``FAKE``가 붙고
    ``api.fake``가 참이다. 실제 결과로 쓰지 않는다(그래프는 워터마크를 단다)."""
    import random

    from robo_jev.bench.openrouter import FORMAT, MODE, ORDERS

    rng = random.Random(f"{model}|{seed}")
    out = []
    for record in records:
        if rng.random() > coverage:
            continue
        questions = {}
        for name, question in record["request"]["questions"].items():
            label = (record.get("labels") or {}).get(name) or {}
            kind = question["type"]
            if kind in ("noul", "score"):
                y = float(label.get("p_yes" if kind == "noul" else "value", 0.5))
                p = min(max(strength * y + (1 - strength) * rng.random(), 0.0), 1.0)
                answer = {"p_yes": p} if kind == "noul" else {"value": p, "bins": None}
                shown = {"options": None if kind == "noul" else [f"{i / 10:.1f}" for i in range(11)],
                         "values": None if kind == "noul" else [i / 10 for i in range(11)]}
            else:
                keys = list(question["criteria"])
                q = label.get("probabilities") or {k: 1 / len(keys) for k in keys}
                noise = [rng.random() for _ in keys]
                raw = [strength * float(q.get(k, 0.0)) + (1 - strength) * z / sum(noise) for k, z in zip(keys, noise)]
                total = sum(raw) or 1.0
                answer = {"probabilities": {k: v / total for k, v in zip(keys, raw)}}
                shown = {"options": keys, "values": None}
            listing = {**shown, "codes": None, "code_probs": None, "code_mass": None, "answer": answer, "raw": None}
            questions[name] = {"type": kind, "n_listings": 1, "answer": answer, "code_mass": None, "listings": [listing]}
        out.append({"id": str(record["id"]), "model": f"FAKE {model}", "mode": MODE, "format": FORMAT, "prompt_version": f"{FORMAT}/score11",
                    "orders": ORDERS, "shuffle_seed": None, "api": {"model": None, "provider": None, "fake": True, "usage": {}}, "questions": questions})
    return out
