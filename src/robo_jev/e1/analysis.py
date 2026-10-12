"""Task E1 analysis — situation reading, paired bootstrap and the pre-declared interpretation rule (progress.md, 2026-10-12 ~19:05 KST).

**Primary metric: situation reading.** For every question group — the same question id in the same family (``meta.scene``: an
engine scene for A/B, a T8b family for C/D) — the Pearson r between the model's answer (the order-averaged p: ``answer.p_yes`` of a
yes/no, ``answer.value`` of a score, as the diagnosis tool stores it) and the label; groups with n ≥ 20 (counted on the paired
analysis set) are pooled, weighted by n. A group whose r is undefined (no variance) is left out.

**Paired bootstrap over moments** (2,000 draws): the moments of a pair of conditions (A/B: the A moments whose B rewrite passed;
C/D: the 480 T8b moments) are resampled with replacement; each draw recomputes every fixed group's r on the resampled items (with
their multiplicity) and pools by the resampled n — so Δformat_engine = r_B − r_A and Δformat_T8b = r_C − r_D are paired on the same
moments, and r_C's interval comes from the same draws. The two models use the same draws. Percentile 95 % intervals.

**Rule** (:func:`apply_rule`, mechanical): H1 iff Δformat_engine ≤ −0.20 with its interval below −0.10 (upper bound < −0.10) and
Δformat_T8b ≥ +0.20 with its interval above +0.10 (lower bound > +0.10); H2 iff Δformat_engine > −0.10 and r_C < 0.20; BOTH iff
Δformat_engine ≤ −0.20 and r_C < 0.20; otherwise INCONCLUSIVE. The rule's labels are not exclusive as written (H1 and BOTH can hold
together when r_D < 0); if more than one holds, all are reported and the overlap is flagged rather than resolved here.
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

__all__ = ["MIN_GROUP_N", "Item", "apply_rule", "build_items", "paired_bootstrap", "pearson", "situation_reading"]

MIN_GROUP_N = 20
#: An item: (moment key, group (family, question id), label y, answer p).
Item = tuple[str, tuple[str, str], float, float]


def _scalar(entry: Mapping[str, Any] | None) -> float | None:
    if not isinstance(entry, Mapping):
        return None
    for key in ("p_yes", "value"):
        if entry.get(key) is not None:
            return float(entry[key])
    return None


def build_items(records: Sequence[Mapping[str, Any]], predictions: Sequence[Mapping[str, Any]], *, moment_key, keep: set[str] | None = None) -> list[Item]:
    """Yes/no and score items of one condition: (moment key, (family, question id), label, order-averaged answer). ``moment_key``
    maps a record to its moment (the pairing key); ``keep`` limits to these moment keys. A question without a prediction raises —
    every analysed question must have been scored."""
    predicted = {str(p["id"]): p for p in predictions}
    out: list[Item] = []
    for record in records:
        key = moment_key(record)
        if keep is not None and key not in keep:
            continue
        family = str(record["meta"]["scene"])
        questions = (predicted.get(str(record["id"])) or {}).get("questions") or {}
        for name, question in record["request"]["questions"].items():
            if question["type"] not in ("noul", "score"):
                continue
            y = _scalar(record["labels"][name])
            p = _scalar((questions.get(name) or {}).get("answer"))
            if y is None or p is None:
                raise ValueError(f"{record['id']} {name}: missing label or prediction")
            out.append((key, (family, name), y, p))
    return out


def pearson(x: Sequence[float], y: Sequence[float], w: Sequence[float] | None = None) -> float | None:
    """(Weighted) Pearson r; None if fewer than 3 items (by weight) or no variance."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    w = np.ones_like(x) if w is None else np.asarray(w, float)
    total = w.sum()
    if total < 3:
        return None
    mx, my = (w * x).sum() / total, (w * y).sum() / total
    vx, vy = (w * (x - mx) ** 2).sum(), (w * (y - my) ** 2).sum()
    if vx <= 1e-15 or vy <= 1e-15:
        return None
    return float((w * (x - mx) * (y - my)).sum() / math.sqrt(vx * vy))


def _groups(items: Iterable[Item]) -> dict[tuple[str, str], list[Item]]:
    out: dict[tuple[str, str], list[Item]] = defaultdict(list)
    for item in items:
        out[item[1]].append(item)
    return dict(out)


def situation_reading(items: Sequence[Item], *, groups: Sequence[tuple[str, str]] | None = None, min_n: int = MIN_GROUP_N) -> dict[str, Any]:
    """Pooled within-group r (n-weighted over groups with n ≥ ``min_n``, or over the given ``groups``) and the per-group table."""
    by_group = _groups(items)
    chosen = list(groups) if groups is not None else sorted(g for g, its in by_group.items() if len(its) >= min_n)
    table, num, den = {}, 0.0, 0
    for g in chosen:
        its = by_group.get(g, [])
        r = pearson([i[3] for i in its], [i[2] for i in its])
        table["|".join(g)] = {"n": len(its), "r": r, "mean_answer": float(np.mean([i[3] for i in its])) if its else None,
                              "mean_label": float(np.mean([i[2] for i in its])) if its else None}
        if r is not None:
            num += len(its) * r
            den += len(its)
    return {"r": (num / den) if den else None, "n_items": den, "groups_used": sum(1 for v in table.values() if v["r"] is not None),
            "groups": table, "all_groups": {"|".join(g): len(its) for g, its in sorted(by_group.items())}}


def _boot_r(items: Sequence[Item], groups: Sequence[tuple[str, str]], moment_index: dict[str, int], weights: np.ndarray) -> np.ndarray:
    """Pooled r for every draw: ``weights`` is draws × moments (how often each moment was drawn)."""
    draws = weights.shape[0]
    num, den = np.zeros(draws), np.zeros(draws)
    by_group = _groups(items)
    for g in groups:
        its = by_group.get(g, [])
        if not its:
            continue
        idx = np.array([moment_index[i[0]] for i in its])
        y = np.array([i[2] for i in its])
        p = np.array([i[3] for i in its])
        w = weights[:, idx]                                   # draws × items
        n = w.sum(1)
        safe = np.maximum(n, 1e-12)
        mp, my = (w * p).sum(1) / safe, (w * y).sum(1) / safe
        dp, dy = p[None, :] - mp[:, None], y[None, :] - my[:, None]
        cov, vp, vy = (w * dp * dy).sum(1), (w * dp * dp).sum(1), (w * dy * dy).sum(1)
        ok = (n >= 3) & (vp > 1e-15) & (vy > 1e-15)
        r = np.where(ok, cov / np.sqrt(np.where(ok, vp * vy, 1.0)), 0.0)
        num += np.where(ok, n * r, 0.0)
        den += np.where(ok, n, 0.0)
    return np.where(den > 0, num / np.maximum(den, 1e-12), np.nan)


def _interval(samples: np.ndarray) -> tuple[float | None, float | None]:
    samples = samples[np.isfinite(samples)]
    if not samples.size:
        return None, None
    return float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))


def paired_bootstrap(items_x: Sequence[Item], items_y: Sequence[Item], *, n_boot: int = 2000, seed: int = 0,
                     min_n: int = MIN_GROUP_N) -> dict[str, Any]:
    """Situation reading of two conditions on the same moments, and Δ = r_y − r_x with a paired bootstrap over moments. The groups
    (n ≥ ``min_n``) are fixed from condition x's full analysis set (both conditions ask the same questions of the same moments)."""
    moments_x = sorted({i[0] for i in items_x})
    if moments_x != sorted({i[0] for i in items_y}):
        raise ValueError("the two conditions do not cover the same moments")
    gx, gy = _groups(items_x), _groups(items_y)
    if {g: len(v) for g, v in gx.items()} != {g: len(v) for g, v in gy.items()}:
        raise ValueError("the two conditions do not ask the same questions")
    groups = sorted(g for g, its in gx.items() if len(its) >= min_n)
    full_x = situation_reading(items_x, groups=groups)
    full_y = situation_reading(items_y, groups=groups)
    index = {m: i for i, m in enumerate(moments_x)}
    rng = np.random.default_rng(int(seed))
    sample = rng.integers(0, len(moments_x), size=(int(n_boot), len(moments_x)))
    weights = np.zeros((int(n_boot), len(moments_x)))
    np.add.at(weights, (np.repeat(np.arange(int(n_boot)), len(moments_x)), sample.ravel()), 1.0)
    bx, by = _boot_r(items_x, groups, index, weights), _boot_r(items_y, groups, index, weights)
    delta = by - bx
    rx, ry = full_x["r"], full_y["r"]
    lo, hi = _interval(delta)
    return {"moments": len(moments_x), "groups": ["|".join(g) for g in groups], "n_boot": int(n_boot), "seed": int(seed),
            "x": {**full_x, "low": _interval(bx)[0], "high": _interval(bx)[1]},
            "y": {**full_y, "low": _interval(by)[0], "high": _interval(by)[1]},
            "delta": {"value": (ry - rx) if rx is not None and ry is not None else None, "low": lo, "high": hi}}


def apply_rule(delta_engine: Mapping[str, Any], delta_t8b: Mapping[str, Any], r_c: float) -> dict[str, Any]:
    """The pre-declared rule, mechanically. Inputs: {"value", "low", "high"} for the two deltas and r_C's point value."""
    de, de_hi = float(delta_engine["value"]), float(delta_engine["high"])
    dt, dt_lo = float(delta_t8b["value"]), float(delta_t8b["low"])
    conditions = {
        "H1": {"delta_engine <= -0.20": de <= -0.20, "delta_engine interval upper < -0.10": de_hi < -0.10,
               "delta_t8b >= +0.20": dt >= 0.20, "delta_t8b interval lower > +0.10": dt_lo > 0.10},
        "H2": {"delta_engine > -0.10": de > -0.10, "r_C < 0.20": r_c < 0.20},
        "BOTH": {"delta_engine <= -0.20": de <= -0.20, "r_C < 0.20": r_c < 0.20},
    }
    holds = [name for name, parts in conditions.items() if all(parts.values())]
    verdict = holds[0] if len(holds) == 1 else ("INCONCLUSIVE" if not holds else " + ".join(holds) + " (rule overlap — not resolved)")
    return {"verdict": verdict, "holds": holds, "conditions": conditions,
            "inputs": {"delta_engine": dict(delta_engine), "delta_t8b": dict(delta_t8b), "r_C": r_c}}
