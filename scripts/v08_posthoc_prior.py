#!/usr/bin/env python
"""Task v0.8b §1.5 — the post-hoc prior-removal estimate on v0.7's predictions: how large P1 could be (reported only, never a gate; CPU).

    artifacts/scratch/v08b/cpu.sh posthoc scripts/v08_posthoc_prior.py        # → artifacts/v08/posthoc/estimate.{json,md}

**The priors** (v08-plan §2) come from v0.7's training input ``artifacts/simgen/v07/train-v07.jsonl``, read in code (counts and means only).
Per family — engine records (``meta.body``), MuJoCo-author records (``meta.scene``), and the two pooled ("physics") — and per question type
(yes/no = ``noul``, ``score``): the label band shares (cut at 1/3 and 2/3), the mean label μ, and μ_bal = the mean of the three band means
(the mean label when every band carries the same total weight — what the FP arm's H-prior weights do within a family).

**The shift.** Every yes/no and score answer of every listing (p_yes, or the score's value) of v0.7-s0 and v0.7-s1 on A4, T9 and T8b (the
existing Spark predictions — nothing is re-scored) is moved by the prior's logit: ``a' = σ(logit(a) + logit(μ_bal) − logit(μ))`` — the Bayes
correction of a probability trained under label prior μ to the band-balanced prior μ_bal. The main estimate uses the pooled physics prior of
the answer's type; the engine-only and MuJoCo-only priors give the sensitivity range.

**The reading** is P1's: DC on the call view (v0.5's truth classes; T8b's from B2's alternative-prior labels), v0.7's arm = seed means,
Δ_s = DC(shifted) − DC(as scored) per set, the mean of the three with one draw-wise bootstrap (each set resampled by its pair clusters with
seed 0 + its index; 2,000 draws). Also CA and GAP before and after (does the shift buy DC with accuracy?), and the untrained base's DC as the
reference P1 points toward. A post-hoc shift is an upper-bound-like stand-in: training with balanced weights need not move the answers this
way, and it moves every answer by the same logit, so read it as the expected size of P1, not a prediction of the verdict.
"""

from __future__ import annotations

import argparse
import copy
import datetime as dt
import functools
import importlib.util
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

SCRIPT_VERSION = "v08-posthoc-1"
ARTIFACTS = REPO / "artifacts"
TRAIN_V07 = ARTIFACTS / "simgen" / "v07" / "train-v07.jsonl"
SETS = ("A4", "T9", "T8b")
V07_SPARK = ARTIFACTS / "reports" / "v07" / "spark"
SEEDS = (0, 1)
FAMILIES = ("engine", "mujoco", "physics")
TYPES = ("noul", "score")
EPS = 1e-6


@functools.lru_cache(maxsize=None)
def _script(name: str):
    spec = importlib.util.spec_from_file_location(f"v08posthoc_{Path(name).stem}", REPO / "scripts" / name)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _log(message: str) -> None:
    print(f"[v08-posthoc {dt.datetime.now().strftime('%H:%M:%S')}] {message}", file=sys.stderr, flush=True)


def family_of(record: dict[str, Any]) -> str | None:
    meta = record.get("meta") or {}
    if "body" in meta:
        return "engine"
    if "scene" in meta:
        return "mujoco"
    return None


def band(y: float) -> str:
    return "low" if y < 1.0 / 3.0 else "high" if y > 2.0 / 3.0 else "middle"


def label_value(label: Any) -> float | None:
    if not isinstance(label, dict):
        return None
    for key in ("p_yes", "value"):
        if isinstance(label.get(key), (int, float)) and not isinstance(label.get(key), bool):
            return float(label[key])
    return None


def priors(path: Path) -> dict[str, Any]:
    """Per family and type: band shares, μ, μ_bal (module docstring) and the logit shift — read line by line, counts and sums only."""
    sums: dict[tuple[str, str, str], list[float]] = defaultdict(lambda: [0.0, 0])
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            fam = family_of(record)
            if fam is None:
                continue
            for name, question in record["request"]["questions"].items():
                kind = question.get("type")
                if kind not in TYPES:
                    continue
                y = label_value((record.get("labels") or {}).get(name))
                if y is None:
                    continue
                for f in (fam, "physics"):
                    entry = sums[(f, kind, band(y))]
                    entry[0] += y
                    entry[1] += 1
    out: dict[str, Any] = {}
    for f in FAMILIES:
        for kind in TYPES:
            counts = {b: sums[(f, kind, b)][1] for b in ("low", "middle", "high")}
            total = sum(counts.values())
            if not total:
                continue
            means = {b: (sums[(f, kind, b)][0] / counts[b]) if counts[b] else None for b in counts}
            mu = sum(sums[(f, kind, b)][0] for b in counts) / total
            present = [m for m in means.values() if m is not None]
            mu_bal = sum(present) / len(present)
            out[f"{f}/{kind}"] = {"questions": total, "band_shares": {b: counts[b] / total for b in counts}, "band_means": means, "mu": mu,
                                  "mu_balanced": mu_bal, "logit_shift": logit(mu_bal) - logit(mu)}
    return out


def logit(p: float) -> float:
    p = min(max(float(p), EPS), 1.0 - EPS)
    return math.log(p / (1.0 - p))


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def shift_predictions(predictions: list[dict[str, Any]], shifts: dict[str, float]) -> list[dict[str, Any]]:
    """A copy of diagnosis predictions with every listing's yes/no p_yes and score value moved by the type's logit shift."""
    out = []
    for prediction in predictions:
        p = copy.deepcopy(prediction)
        for question in (p.get("questions") or {}).values():
            kind = question.get("type")
            if kind not in shifts:
                continue
            key = "p_yes" if kind == "noul" else "value"
            for listing in question.get("listings") or []:
                answer = listing.get("answer") or {}
                if answer.get(key) is not None:
                    answer[key] = sigmoid(logit(answer[key]) + shifts[kind])
            answer = question.get("answer") or {}
            if answer.get(key) is not None:
                answer[key] = sigmoid(logit(answer[key]) + shifts[kind])
        out.append(p)
    return out


def set_records() -> dict[str, list[dict[str, Any]]]:
    from robo_jev.decisions.records import read_records

    paths = _script("verdict_v06.py").SET_PATHS
    sets = {name: read_records(paths[name]) for name in SETS}
    sets["T8b"], _ = _script("verdict_v06.py").t8b_with_b2(sets["T8b"])
    return sets


def calls(name: str, records: list[dict[str, Any]], predictions: list[dict[str, Any]]) -> dict[tuple, tuple]:
    clusters = _script("verdict_v04.py").pair_clusters(records)
    return _script("verdict_v05.py").call_items(name, records, predictions, clusters=clusters)


def draws_dc(sources: list[dict[tuple, tuple]], *, n_boot: int, seed: int) -> tuple[list[dict[str, float]], np.ndarray]:
    """Point readings (DC, CA, GAP) of each source on their common items and the DC bootstrap draws (cluster draws shared by the sources)."""
    v5 = _script("verdict_v05.py")
    keys = sorted(set.intersection(*(set(s) for s in sources)))
    sums, counts, clusters = v5._cluster_arrays(keys, sources)
    points = []
    for s in sums:
        m = v5._metrics(s.sum(0), counts.sum(0))
        points.append({k: float(m[k]) for k in ("DC", "CA", "GAP", "LU", "LC")})
    w = v5._draw_weights(clusters, n_boot, seed)
    n = w @ counts
    draws = np.stack([np.asarray(v5._metrics(np.einsum("bc,ctk->btk", w, s), n)["DC"], dtype=float) for s in sums])
    return points, draws


def estimate(prior_key: str, prior: dict[str, Any], sets: dict[str, list], preds: dict[str, dict[int, list]], base: dict[str, list], *,
             n_boot: int, seed: int) -> dict[str, Any]:
    """The P1-like reading for one prior: per set Δ = DC(shifted v0.7 arm) − DC(v0.7 arm), the mean with the draw-wise bootstrap."""
    from robo_jev.bench.system1 import _interval

    v6 = _script("verdict_v06.py")
    shifts = {kind: prior[f"{prior_key}/{kind}"]["logit_shift"] for kind in TYPES}
    per_set, deltas = {}, []
    for index, name in enumerate(SETS):
        before = [calls(name, sets[name], preds[name][s]) for s in SEEDS]
        after = [calls(name, sets[name], shift_predictions(preds[name][s], shifts)) for s in SEEDS]
        arm_before, arm_after = v6.arm_calls(*before), v6.arm_calls(*after)
        base_calls = calls(name, sets[name], base[name])
        points, draws = draws_dc([arm_after, arm_before, base_calls], n_boot=n_boot, seed=seed + index)
        delta = draws[0] - draws[1]
        low, high = _interval(delta)
        per_set[name] = {"DC_v07": points[1]["DC"], "DC_v07_shifted": points[0]["DC"], "delta_DC": points[0]["DC"] - points[1]["DC"], "low": low,
                         "high": high, "CA_v07": points[1]["CA"], "CA_v07_shifted": points[0]["CA"], "GAP_v07": points[1]["GAP"],
                         "GAP_v07_shifted": points[0]["GAP"], "DC_base": points[2]["DC"], "v07_minus_base_DC": points[1]["DC"] - points[2]["DC"],
                         "questions": len(set(arm_after) & set(arm_before) & set(base_calls))}
        deltas.append(delta)
    mean_draws = np.mean(np.stack(deltas), axis=0)
    low, high = _interval(mean_draws)
    value = float(np.mean([per_set[s]["delta_DC"] for s in SETS]))
    return {"prior": prior_key, "logit_shift": shifts, "per_set": per_set, "P1_estimate": {"value": value, "low": low, "high": high},
            "mean_v07_minus_base_DC": float(np.mean([per_set[s]["v07_minus_base_DC"] for s in SETS])),
            "mean_delta_CA": float(np.mean([per_set[s]["CA_v07_shifted"] - per_set[s]["CA_v07"] for s in SETS])),
            "mean_delta_GAP": float(np.mean([per_set[s]["GAP_v07_shifted"] - per_set[s]["GAP_v07"] for s in SETS]))}


def main(argv: list[str] | None = None) -> int:
    from robo_jev.bench.gates_v08 import POSTHOC_DIR
    from robo_jev.decisions.records import file_sha256, read_records

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=POSTHOC_DIR / "estimate.json")
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    _log("reading the label priors of train-v07 (counts and means only)")
    prior = priors(TRAIN_V07)
    sets = set_records()
    preds = {name: {s: read_records(V07_SPARK / f"{name}-v07s{s}-2b.predictions.jsonl") for s in SEEDS} for name in SETS}
    base = {name: read_records(path) for name, path in _script("verdict_v07.py").BASE_SPARK.items() if name in SETS}
    results = {key: estimate(key, prior, sets, preds, base, n_boot=args.n_boot, seed=args.seed) for key in FAMILIES}
    main_result = results["physics"]
    payload = {"kind": "v08-posthoc-prior-estimate", "script_version": SCRIPT_VERSION, "reported_only": True,
               "train_v07": {"path": str(TRAIN_V07.relative_to(REPO)), "sha256": file_sha256(TRAIN_V07)}, "priors": prior,
               "P1_estimate": main_result["P1_estimate"], "per_set": main_result["per_set"], "by_prior": results,
               "range_over_priors": {"low": min(r["P1_estimate"]["value"] for r in results.values()),
                                     "high": max(r["P1_estimate"]["value"] for r in results.values())},
               "inputs": {name: [str((V07_SPARK / f"{name}-v07s{s}-2b.predictions.jsonl").relative_to(REPO)) for s in SEEDS] for name in SETS},
               "notes": ["v0.7's existing Spark predictions on A4, T9, T8b (B2 truth); nothing re-scored; the main estimate uses the pooled physics "
                         "prior per answer type, engine-only and MuJoCo-only priors give the range",
                         "a uniform logit shift of every answer — an upper-bound-like stand-in for removing the label prior, not a forecast of FP"],
               "n_boot": args.n_boot, "seed": args.seed, "created": dt.datetime.now().astimezone().isoformat(timespec="seconds")}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")
    lines = ["# Post-hoc prior-removal estimate on v0.7 (reported only — the expected size of P1)", "",
             f"Shift: a' = σ(logit a + logit μ_bal − logit μ) on every yes/no and score answer of v0.7-s0 / -s1 (Spark, as scored); "
             f"{args.n_boot} draw-wise cluster draws.", "", "| prior | shift noul | shift score | P1-like mean ΔDC [95 %] | ΔCA | ΔGAP |", "|---|---|---|---|---|---|"]
    for key, r in results.items():
        e = r["P1_estimate"]
        lines.append(f"| {key} | {r['logit_shift']['noul']:+.3f} | {r['logit_shift']['score']:+.3f} | {e['value']:+.3f} [{e['low']:+.3f}, {e['high']:+.3f}] | "
                     f"{r['mean_delta_CA']:+.3f} | {r['mean_delta_GAP']:+.3f} |")
    lines += ["", "Per set (pooled physics prior):", "", "| set | DC v0.7 | DC shifted | Δ [95 %] | DC base | CA v0.7 → shifted | GAP v0.7 → shifted |",
              "|---|---|---|---|---|---|---|"]
    for name, e in main_result["per_set"].items():
        lines.append(f"| {name} | {e['DC_v07']:.3f} | {e['DC_v07_shifted']:.3f} | {e['delta_DC']:+.3f} [{e['low']:+.3f}, {e['high']:+.3f}] | {e['DC_base']:.3f} | "
                     f"{e['CA_v07']:.3f} → {e['CA_v07_shifted']:.3f} | {e['GAP_v07']:.3f} → {e['GAP_v07_shifted']:.3f} |")
    lines += ["", "Priors (train-v07):", "", "| family / type | questions | low | middle | high | μ | μ_bal |", "|---|---|---|---|---|---|---|"]
    for key, p in prior.items():
        s = p["band_shares"]
        lines.append(f"| {key} | {p['questions']:,} | {s['low']:.3f} | {s['middle']:.3f} | {s['high']:.3f} | {p['mu']:.3f} | {p['mu_balanced']:.3f} |")
    args.out.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    _log("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
