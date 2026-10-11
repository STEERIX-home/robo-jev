"""묶음 통계 `stats.json` (Task v0a 지시서 §6): 장면·분야·몸·문체 비율, 정답 균형, 여유 분포, 글 길이, 생성 속도, 덮은 칸.

정답 균형은 질문 종류마다 본다: noul의 p_yes 분포(0 근처·1 근처·가운데), 핵심 noul의 평균(지시서 §5: 0.3~0.7), 행동 choice의 최고 확률·
엔트로피·argmax 선택지, 상태 choice의 최고 확률, 규칙 답의 갈래. **참 결과 대조**는 noul 정답(사후 확률)을 같은 구간끼리 묶어 참 세계에서
실제로 일어난 비율과 나란히 놓는다 — 사후 표본이 생성 모형과 어긋나면 여기서 보인다(학습 정답의 보정 점검; 완전히 같을 필요는 없다:
참 세계는 여유 표본기에서, 사후는 국소적으로 평평한 사전에서 나온다).
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict

import numpy as np

from robo_jev.simgen.catalog import SimgenConfig

__all__ = ["dataset_stats"]


def _shares(values) -> dict:
    counts = Counter(values)
    total = sum(counts.values())
    return {k: {"n": n, "share": round(n / total, 4)} for k, n in sorted(counts.items(), key=lambda kv: (-kv[1], str(kv[0])))}


def _hist(values, edges) -> dict:
    counts, _ = np.histogram(values, bins=edges)
    return {f"[{lo:g},{hi:g})": int(c) for lo, hi, c in zip(edges[:-1], edges[1:], counts, strict=True)}


def _summary(values) -> dict:
    a = np.asarray(values, dtype=float)
    if len(a) == 0:
        return {"n": 0}
    q = np.quantile(a, [0.0, 0.1, 0.5, 0.9, 1.0])
    return {"n": int(len(a)), "mean": round(float(a.mean()), 4), "std": round(float(a.std()), 4), "min": round(float(q[0]), 4),
            "p10": round(float(q[1]), 4), "median": round(float(q[2]), 4), "p90": round(float(q[3]), 4), "max": round(float(q[4]), 4)}  # fmt: skip


def _entropy(probs: dict) -> float:
    return float(-sum(p * math.log(p) for p in probs.values() if p > 0))


def _design_share(cfg: SimgenConfig, name: str, body: str, k: float = 1.5) -> float:
    spec = cfg.catalog["margins"][name]
    w = cfg.catalog["margins"]["normal_weight"]
    return w * math.erf(k / math.sqrt(2.0)) + (1.0 - w) * min(1.0, k * spec["sigma"][body] / spec["half_range"][body])


def dataset_stats(records: list[dict], cfg: SimgenConfig, seconds: float | None = None) -> dict:
    n = len(records)
    meta = [r["meta"] for r in records]
    out: dict = {"records": n}
    if seconds is not None:
        out["generation"] = {"seconds": round(seconds, 2), "records_per_second": round(n / seconds, 1) if seconds > 0 else None}
    out["shares"] = {
        "scene": _shares(m["scene"] for m in meta),
        "domain": _shares(m["domain"] for m in meta),
        "body": _shares(m["body"] for m in meta),
        "style": _shares(m["style"] for m in meta),
        "phenomena": _shares("+".join(m["phenomena"]) for m in meta),
        "sensor_suite": _shares(m["sensor_suite"] for m in meta),
        "crossing_variant": _shares(m["variant"] for m in meta if m["scene"] == "p1_crossing"),
        "agent_kind": _shares(m["agent"]["kind"] for m in meta if m["agent"]),
        "agent_velocity_mode": _shares(m["agent"]["velocity_mode"] for m in meta if m["agent"]),
        "container_kind": _shares(m["container"]["kind"] for m in meta if m["container"]),
        "container_content": _shares(m["container"]["content"] for m in meta if m["container"]),
        "lid_view": _shares(m["container"]["lid_view"] for m in meta if m["container"]),
        "fill_mode": _shares(m["container"]["fill_mode"] for m in meta if m["container"]),
        "key_style": _shares(m["key_style"] for m in meta),
        "distance_unit": _shares(m["units"]["distance"] for m in meta),
        "speed_unit": _shares(m["units"]["speed"] for m in meta),
        "time_unit": _shares(m["units"]["time"] for m in meta),
        "distractors": _shares(m["distractors"] for m in meta),
    }
    out["coverage"] = {
        "domain_x_scene": {d: dict(Counter(m["scene"] for m in meta if m["domain"] == d)) for d in sorted({m["domain"] for m in meta})},
        "body_x_scene": {b: dict(Counter(m["scene"] for m in meta if m["body"] == b)) for b in sorted({m["body"] for m in meta})},
    }
    rec_present = [m["recommendation"]["present"] for m in meta]
    out["flags"] = {
        "hint_share": round(float(np.mean([m["hint"] for m in meta])), 4),
        "recommendation_share": {s: round(float(np.mean([m["recommendation"]["present"] for m in meta if m["style"] == s])), 4) for s in ("facts", "prose", "planner")},
        "wrong_recommendation_share_of_present": round(float(np.mean([m["recommendation"]["wrong"] for m in meta if m["recommendation"]["present"]])), 4) if any(rec_present) else None,
        "criterion_share_of_action_questions": round(float(np.mean([c is not None for m in meta for c in m["criteria"].values()])), 4),
        "criteria": _shares(c for m in meta for c in m["criteria"].values() if c is not None),
        "pairs": _shares(m["pair"]["kind"] for m in meta if m["pair"]),
        "paired_record_share": round(sum(1 for m in meta if m["pair"]) / n, 4),
        "margin_redraw_share": round(float(np.mean([m["margin_redraws"] > 0 for m in meta])), 4),
    }
    out["questions"] = _question_stats(records)
    out["labels"] = _label_stats(records)
    out["margins"] = _margin_stats(records, cfg)
    out["text_length"] = {"all": _summary([len(r["request"]["state"]) for r in records])} | {
        s: _summary([len(r["request"]["state"]) for r in records if r["meta"]["style"] == s]) for s in ("facts", "prose", "planner")
    }
    out["text_length"]["share_400_to_1500"] = round(float(np.mean([400 <= len(r["request"]["state"]) <= 1500 for r in records])), 4)
    out["truth_check"] = _truth_check(records)
    return out


def _question_stats(records: list[dict]) -> dict:
    per_request = [len(r["request"]["questions"]) for r in records]
    kinds = Counter()
    for r in records:
        for lab in r["labels"].values():
            kinds[lab.get("kind", lab["type"])] += 1
    return {"per_request": _shares(per_request), "kinds": dict(kinds), "options_per_action": _shares(len(lab["probabilities"]) for r in records for lab in r["labels"].values() if lab.get("kind") == "action")}


def _label_stats(records: list[dict]) -> dict:
    noul = [lab["p_yes"] for r in records for lab in r["labels"].values() if lab["type"] == "noul"]
    key = [r["labels"][r["meta"]["key_question"]]["p_yes"] for r in records]
    score = [lab["value"] for r in records for lab in r["labels"].values() if lab["type"] == "score"]
    edges = np.linspace(0, 1, 11)
    key_by_scene = defaultdict(list)
    for r, v in zip(records, key, strict=True):
        key_by_scene[r["meta"]["scene"]].append(v)
    action_top, action_entropy, argmax_by_scene, gaps = [], [], defaultdict(Counter), []
    for r in records:
        for name, lab in r["labels"].items():
            if lab.get("kind") != "action":
                continue
            probs = lab["probabilities"]
            best = max(probs, key=probs.get)
            action_top.append(max(probs.values()))
            action_entropy.append(_entropy(probs))
            argmax_by_scene[r["meta"]["scene"]][r["meta"]["options"][name][best]] += 1
            ec = sorted(lab["expected_cost"].values())
            gaps.append(ec[1] - ec[0] if len(ec) > 1 else 0.0)
    state_top = [max(lab["probabilities"].values()) for r in records for lab in r["labels"].values() if lab.get("kind") == "state"]
    rule_answers = Counter()
    for r in records:
        for name, lab in r["labels"].items():
            if lab.get("kind") == "rule":
                answer = max(lab["probabilities"], key=lab["probabilities"].get)
                rule_answers[r["meta"]["basis"][name]["classes"][answer]] += 1
    return {
        "noul": {"summary": _summary(noul), "histogram": _hist(noul, edges), "share_below_0.05": round(float(np.mean(np.array(noul) < 0.05)), 4),
                 "share_above_0.95": round(float(np.mean(np.array(noul) > 0.95)), 4), "share_0.2_to_0.8": round(float(np.mean((np.array(noul) >= 0.2) & (np.array(noul) <= 0.8))), 4)},  # fmt: skip
        "key_noul": {"summary": _summary(key), "histogram": _hist(key, edges), "mean_by_scene": {s: round(float(np.mean(v)), 4) for s, v in sorted(key_by_scene.items())}},
        "score": {"summary": _summary(score), "histogram": _hist(score, edges)},
        "action": {
            "top_probability": _summary(action_top), "top_probability_histogram": _hist(action_top, np.linspace(0, 1, 11)),
            "share_top_above_0.95": round(float(np.mean(np.array(action_top) > 0.95)), 4), "entropy": _summary(action_entropy),
            "expected_cost_gap_best_to_second": _summary(gaps),
            "argmax_option_by_scene": {s: dict(c.most_common()) for s, c in sorted(argmax_by_scene.items())},
        },
        "state": {"top_probability": _summary(state_top)},
        "rule_answers": dict(rule_answers),
    }


def _margin_stats(records: list[dict], cfg: SimgenConfig) -> dict:
    by_name: dict[str, list] = defaultdict(list)
    for r in records:
        m = r["meta"]
        if (m["pair"] and m["pair"]["role"] == "variant") or m["margin_redraws"]:
            continue
        for name, value in m["margins"].items():
            by_name[name].append((value, m["body"]))
    out = {}
    for name, rows in sorted(by_name.items()):
        values = [v for v, _ in rows]
        normalized = [v / cfg.catalog["margins"][name]["sigma"][b] for v, b in rows]
        out[name] = {
            "unit": cfg.catalog["margins"][name]["unit"],
            "summary": _summary(values),
            "normalized_by_sigma": _summary(normalized),
            "share_abs_below_1.5_sigma": round(float(np.mean(np.abs(normalized) < 1.5)), 4),
            "design_share_abs_below_1.5_sigma": round(float(np.mean([_design_share(cfg, name, b) for _, b in rows])), 4),
            "share_negative": round(float(np.mean(np.array(values) < 0)), 4),
        }
    return out


def _truth_check(records: list[dict]) -> dict:
    """noul 사후 확률 구간별: 평균 정답 대 참 세계에서 실제로 일어난 비율."""
    pairs = [(lab["p_yes"], bool(r["truth"]["realized"][name])) for r in records for name, lab in r["labels"].items() if lab["type"] == "noul"]
    p = np.array([a for a, _ in pairs])
    y = np.array([b for _, b in pairs], dtype=float)
    edges = np.linspace(0, 1, 6)
    table = {}
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        mask = (p >= lo) & ((p < hi) if hi < 1 else (p <= hi))
        if mask.any():
            table[f"[{lo:.1f},{hi:.1f}]"] = {"n": int(mask.sum()), "mean_label": round(float(p[mask].mean()), 4), "realized_rate": round(float(y[mask].mean()), 4)}
    brier = float(np.mean((p - y) ** 2)) if len(p) else None
    action_hits = []
    for r in records:
        for name, lab in r["labels"].items():
            if lab.get("kind") == "action":
                best = max(lab["probabilities"], key=lab["probabilities"].get)
                action_hits.append(best == r["truth"]["realized"][name]["best"] or r["truth"]["realized"][name]["costs"][best] == min(r["truth"]["realized"][name]["costs"].values()))
    return {
        "noul_by_label_bin": table,
        "noul_mean_label": round(float(p.mean()), 4),
        "noul_realized_rate": round(float(y.mean()), 4),
        "noul_brier_vs_truth": round(brier, 4) if brier is not None else None,
        "action_label_best_is_cheapest_in_hindsight": round(float(np.mean(action_hits)), 4),
    }
