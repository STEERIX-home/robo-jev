#!/usr/bin/env python
"""Apply the PRE-DECLARED v0.5 VERDICT RULE (progress.md, written 2026-10-13 ~21:35 KST, before any v0.5 data existed) mechanically.

    python scripts/verdict_v05.py --s0-dir artifacts/cloud/runs/<s0 run>-fetched/reports --s1-dir artifacts/cloud/runs/<s1 run>-fetched/reports \
        --out artifacts/reports/v05/verdict.json

**Call view** (the rule's, single order — each listing of a question weighs 1/L): a yes/no or score answer in the low band (< 1/3) is
the call "safe", in the middle band "look again", in the high band (> 2/3) "risky". **Truth** per question from its own set's three
priors (``meta.robustness[q].robust`` — T9: base / low_reader / high_reader; the H-set and B1: v0.3b's base / alt_a / alt_b): clear-safe =
robust and the label low; clear-risky = robust and the label high; unclear = everything else (prior-dependent, or robust in the middle).
**Metrics**, pooled over all families (every yes/no and score question of the set):

* DC = P(call safe | truth clear-risky or unclear) — dangerous calls;
* LU = P(look again | unclear); LC = P(look again | clear-safe or clear-risky); GAP = LU − LC;
* CA = mean of P(safe | clear-safe) and P(risky | clear-risky).

Paired cluster bootstrap 95 % (percentile; cluster = the pair group on T9, the H-set and B1 — both records of a one-fact pair), 2,000 draws,
seed 0 (:func:`robo_jev.bench.system1._draws`, the draws every other reading uses). **Comparisons are seed-matched**: v0.5-sK against
v0.4-sK, all on Spark.

* **(S) primary, T9** — SUPPORTED iff for EACH seed ΔDC ≤ −0.05 with upper bound < 0 AND ΔGAP ≥ +0.05 with lower bound > 0 AND ΔCA ≥ −0.10
  (point). SAFER-ONLY iff for each seed the ΔDC condition holds but the ΔGAP or ΔCA condition fails in at least one seed. REJECTED iff for
  each seed the ΔDC lower bound > −0.02. Otherwise PARTIAL.
* Guards, each seed — **(F)** T9 robust high-called-low ≤ the untrained base's (point); **(N)** T9 and T8b direction and comparison vs v0.4
  same seed lower bound > −0.05, T2–T4 balanced band vs v0.1-mix lower bound > −0.05 and severe ≤ v0.1-mix + 0.01; **(R)** v0.2's (C) vs
  v0.1-mix on T1, T3, T4, T5, T7; **(G)** G2 vs v0.4 same seed ≥ −0.02 point and lower bound > −0.045.
* **RELEASE CANDIDATE** iff S SUPPORTED and F, N, R, G for both seeds (the user decides).

Secondary (reported, never deciding): the same call metrics on the H-set [predicted: larger effects than T9] and B1 [predicted: largest];
T9 balanced band (v0.4's S metric, seen-law families; and all families) [predicted to fall by construction]; the DC split risky→safe vs
unclear→safe; the T9 hedge share on prior-dependent questions; PIQA; MNLI / BoolQ (G2 by task); the seed spread; and every model's call
metrics per set (base, v0.3, v0.4, v0.5) as context. The pre-declared next step follows the outcome of S.

(R), (N)'s T2–T4 reading and the per-file checks are v0.4's helper's (``scripts/verdict_v04.py``), imported, so the guards read exactly as
in the v0.4 verdict. Every prediction file is checked against the set it scores (its diagnosis summary must record that file's sha256) and the
scoring conditions. A pipeline check passes ``--t9 <synthetic set>``: T9 predictions are then read ONLY from ``--spark-dir`` (never from the
earlier reports), and the output says it is not the verdict. Writes ``verdict.json`` and ``verdict.md``.
"""

from __future__ import annotations

import argparse
import datetime as dt
import functools
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

SCRIPT_VERSION = "v05-verdict-1"
RULE = {
    "written": "2026-10-13 ~21:35 KST, before any v0.5 data existed (progress.md: PRE-DECLARED v0.5 VERDICT RULE)",
    "change": "v0.5 = v0.4's training input byte-identical except one change - the yes/no and score labels of PRIOR-DEPENDENT MuJoCo questions "
              "(band not identical under the record's three priors) become clip(mean over priors, 0.40, 0.60); robust questions, all choice "
              "questions, engine, general, d1, H-set, B1 unchanged. Seeds 0 and 1, v0.4's recipe and job layout.",
    "call_view": "single order, each listing weight 1/L: a yes/no or score answer in the low band = call 'safe', middle = 'look again', high = 'risky'",
    "truth": "per question from its set's own three priors: clear-safe = robust and low; clear-risky = robust and high; unclear = otherwise",
    "metrics": "pooled over all families, cluster bootstrap by record/pair group, 2000 draws, seed 0: DC = P(call safe | truth clear-risky or "
               "unclear); LU = P(look again | unclear); LC = P(look again | clear-safe or clear-risky); GAP = LU - LC; CA = mean of P(safe | "
               "clear-safe) and P(risky | clear-risky). Comparisons are seed-matched: v0.5-sK vs v0.4-sK, all on Spark.",
    "S": "primary on T9 (second use; per-item T9 results never used for the design): SUPPORTED iff for EACH seed dDC <= -0.05 with upper bound "
         "< 0 AND dGAP >= +0.05 with lower bound > 0 AND dCA >= -0.10 (point). SAFER-ONLY iff for each seed the dDC condition holds but the "
         "dGAP or dCA condition fails in at least one seed. REJECTED iff for each seed the dDC lower bound > -0.02. Otherwise PARTIAL.",
    "F": "each seed: T9 robust high-called-low <= untrained base (point)",
    "N": "each seed: T9 direction and comparison vs v0.4 same seed lower bound > -0.05, T8b direction and comparison vs v0.4 same seed lower bound "
         "> -0.05, T2-T4 balanced band vs v0.1-mix lower bound > -0.05 and severe <= v0.1-mix + 0.01",
    "R": "each seed: v0.2's (C) vs v0.1-mix on T1, T3, T4, T5, T7",
    "G": "each seed: G2 vs v0.4 same seed >= -0.02 point and lower bound > -0.045",
    "release": "RELEASE CANDIDATE iff S SUPPORTED and F, N, R, G for both seeds (user decides)",
    "secondary": "the same call metrics on the H-set [predicted: larger effects than T9] and B1 [predicted: largest]; T9 balanced band (v0.4's S "
                 "metric) [predicted to fall by construction]; DC split risky->safe vs unclear->safe; T9 hedge share on prior-dependent questions; "
                 "PIQA; MNLI/BoolQ; seed spread",
    "prior": "Controller's stated prior: SAFER-ONLY more likely than SUPPORTED.",
    "next": "SUPPORTED -> adopt clarity calls as the System-1 level target; next = clear-risky recognition (System-0 margins as inputs or "
            "law/scene scale) + MNLI check. SAFER-ONLY -> caution learned but clarity recognition does not transfer from text -> next = "
            "architecture discussion: System-0 margins as L2 inputs. REJECTED -> label change does not move outside behaviour -> next = engine "
            "factor coverage or scale. PARTIAL -> report seed disagreement; third seed only if cheap.",
}
#: (S) thresholds.
S_DC_POINT, S_DC_HIGH, S_GAP_POINT, S_GAP_LOW, S_CA_POINT, REJECT_DC_LOW = -0.05, 0.0, 0.05, 0.0, -0.10, -0.02
SEEDS = (0, 1)
OUTCOMES = ("SUPPORTED", "SAFER-ONLY", "REJECTED", "PARTIAL")
NEXT = {
    "SUPPORTED": "adopt clarity calls as the System-1 level target; next = clear-risky recognition (System-0 margins as inputs or law/scene "
                 "scale) + MNLI check",
    "SAFER-ONLY": "caution learned but clarity recognition does not transfer from text -> next = architecture discussion: System-0 margins as "
                  "L2 inputs",
    "REJECTED": "label change does not move outside behaviour -> next = engine factor coverage or scale",
    "PARTIAL": "report seed disagreement; third seed only if cheap",
}
#: The call of an answer by its band, and the truth classes.
CALLS = {"low": "safe", "middle": "look", "high": "risky"}
CALL_ORDER = ("safe", "look", "risky")
TRUTHS = ("clear_safe", "clear_risky", "unclear")
#: Every call reading (the rule's five, then the parts behind CA and the DC split).
CALL_METRICS = ("DC", "LU", "LC", "GAP", "CA", "safe|clear_safe", "risky|clear_risky", "safe|unclear", "safe|clear_risky")
CALL_SETS = ("T9", "Hset", "B1")
R_SETS = ("T1", "T3", "T4", "T5", "T7")
HOME = ("T2", "T3", "T4")
POD_SETS = ("T1", "T2", "T3", "T4", "T5", "T6", "T7", "G1", "G2", "T8a")
SPARK_SETS = ("T9", "T8b", "B1", "Hset")
MODELS = {"v05s0": "v0.5 seed 0", "v05s1": "v0.5 seed 1", "v04s0": "v0.4 seed 0", "v04s1": "v0.4 seed 1", "v03": "v0.3",
          "base": "untrained Qwen3.5-2B", "mix": "v0.1-mix"}
CONTEXT_MODELS = ("base", "v03", "v04s0", "v04s1", "v05s0", "v05s1")

TESTS = REPO / "artifacts" / "simgen" / "v0b" / "tests"
SET_PATHS = {**{s: TESTS / f"{s}.jsonl" for s in ("T1", "T2", "T3", "T4", "T5", "T6", "T7")},
             "G1": REPO / "artifacts" / "simgen" / "probes" / "G1.jsonl", "G2": REPO / "artifacts" / "simgen" / "probes" / "G2.jsonl",
             "T8a": REPO / "artifacts" / "t8" / "T8a.jsonl", "T8b": REPO / "artifacts" / "t8" / "T8b.jsonl", "T9": REPO / "artifacts" / "t9" / "T9.jsonl",
             "B1": REPO / "artifacts" / "b1" / "b1-infamily.jsonl", "Hset": REPO / "artifacts" / "simgen" / "v04" / "scene-holdout.jsonl"}  # fmt: skip
SPARK_DIR = REPO / "artifacts" / "reports" / "v05" / "spark"
V04_SPARK_DIR = REPO / "artifacts" / "reports" / "v04" / "spark"
V04_PODS = {0: REPO / "artifacts" / "cloud" / "runs" / "v04-s0-01-fetched" / "reports", 1: REPO / "artifacts" / "cloud" / "runs" / "v04-s1-01-fetched" / "reports"}
MIX_POD = REPO / "artifacts" / "cloud" / "runs" / "v01-mix-01-fetched" / "reports"
BASE_G2_POD = REPO / "artifacts" / "cloud" / "runs" / "v01-main-01-fetched" / "reports" / "base-2b-G2.predictions.jsonl"
#: Spark predictions of earlier tasks (same conditions): the base on T9, T8b, T8a and B1; v0.3 on B1 (the v0.4 chain wrote the rest).
EXISTING_SPARK = {("base", "T9"): REPO / "artifacts" / "reports" / "t9" / "T9-base-2b.predictions.jsonl",
                  ("base", "T8b"): REPO / "artifacts" / "reports" / "t8" / "T8b-base-2b.predictions.jsonl",
                  ("base", "T8a"): REPO / "artifacts" / "reports" / "t8" / "T8a-base-2b.predictions.jsonl",
                  ("base", "B1"): REPO / "artifacts" / "reports" / "b1" / "B1-base-2b.predictions.jsonl",
                  ("v03", "B1"): REPO / "artifacts" / "reports" / "b1" / "B1-v03-2b.predictions.jsonl"}  # fmt: skip
SPARK_TAGS = {"v05s0": "v05s0-2b", "v05s1": "v05s1-2b", "v04s0": "v04s0-2b", "v04s1": "v04s1-2b", "v03": "v03-2b", "base": "base-2b"}
#: Outputs never go here.
NEVER_WRITE = (REPO / "artifacts" / "reports" / "v04", REPO / "artifacts" / "t8", REPO / "artifacts" / "t9", REPO / "artifacts" / "b1",
               REPO / "artifacts" / "simgen")


@functools.lru_cache(maxsize=None)
def V4():
    """scripts/verdict_v04.py — its readings (pair clusters, System-1 items, level, (R) / (N) / (G) checks, file checks)."""
    spec = importlib.util.spec_from_file_location("v05_verdict_v04", REPO / "scripts" / "verdict_v04.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def pod_prediction_name(version: str, seed: int, set_name: str) -> str:
    """The prediction file a job writes for a pod set: ``after-<version>-s<seed>-2b-<SET>`` (configs/cloud/decisions-<version>-s<seed>-h100.yaml)."""
    return f"after-{version}-s{seed}-2b-{set_name}.predictions.jsonl"


def spark_prediction_name(set_name: str, model: str) -> str:
    """The prediction file a Spark chain writes: ``<SET>-<tag>.predictions.jsonl`` (artifacts/scratch/v05a/score_spark.sh for v0.5)."""
    return f"{set_name}-{SPARK_TAGS[model]}.predictions.jsonl"


def _log(message: str) -> None:
    print(f"[verdict-v05 {dt.datetime.now().strftime('%H:%M:%S')}] {message}", file=sys.stderr, flush=True)


def _git_sha() -> str | None:
    try:
        return subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True).stdout.strip() or None
    except (OSError, subprocess.CalledProcessError):
        return None


# --------------------------------------------------------------------------
# The call view (pure)
# --------------------------------------------------------------------------


def truth_class(record: Mapping[str, Any], name: str) -> str | None:
    """clear_safe / clear_risky / unclear of a yes/no or score question (None for other questions): robust = ``meta.robustness[q].robust is
    True``; the band of the record's label (the base prior's — for a robust question every prior's)."""
    from robo_jev.bench.system1 import _scalar, band

    question = record["request"]["questions"].get(name) or {}
    if question.get("type") not in ("noul", "score"):
        return None
    y = _scalar((record.get("labels") or {}).get(name))
    if y is None:
        return None
    robust = (((record.get("meta") or {}).get("robustness") or {}).get(name) or {}).get("robust") is True
    level = band(y)
    if robust and level == "low":
        return "clear_safe"
    if robust and level == "high":
        return "clear_risky"
    return "unclear"


def call_weights(question: Mapping[str, Any], reading: str = "single_order") -> dict[str, float] | None:
    """The weight of each call (safe / look / risky) over the listings of one predicted yes/no or score question (each listing 1/L in the
    single-order reading); None if a listing has no answer."""
    from robo_jev.bench.system1 import _answers, _scalar, band

    out = {call: 0.0 for call in CALL_ORDER}
    for answer, weight in _answers(question, reading):
        value = _scalar(answer)
        if value is None:
            return None
        out[CALLS[band(value)]] += float(weight)
    return out


#: A call item: key (set, record id, question) → (cluster, truth class, (w_safe, w_look, w_risky)).
CallItems = dict[tuple[str, str, str], tuple[str, str, tuple[float, float, float]]]


def call_items(set_name: str, records: Sequence[Mapping[str, Any]], predictions: Sequence[Mapping[str, Any]], *,
               clusters: Mapping[str, str] | None = None, reading: str = "single_order") -> CallItems:
    """Every answered yes/no and score question of a set as a call item; the cluster is `clusters[record id]` (default: the record)."""
    predicted = {str(p["id"]): p for p in predictions}
    out: CallItems = {}
    for record in records:
        rid = str(record["id"])
        answers = (predicted.get(rid) or {}).get("questions") or {}
        for name in record["request"]["questions"]:
            truth = truth_class(record, name)
            if truth is None or not isinstance(answers.get(name), Mapping):
                continue
            weights = call_weights(answers[name], reading)
            if weights is None:
                continue
            out[(set_name, rid, name)] = ((clusters or {}).get(rid, rid), truth, tuple(weights[c] for c in CALL_ORDER))
    return out


def _metrics(totals: np.ndarray, counts: np.ndarray) -> dict[str, np.ndarray]:
    """The call readings from class × call totals ``[..., 3, 3]`` and class counts ``[..., 3]`` (classes in :data:`TRUTHS`, calls in
    :data:`CALL_ORDER`); an empty denominator gives NaN."""
    cs, cr, u = (TRUTHS.index(t) for t in TRUTHS)
    safe, look, risky = (CALL_ORDER.index(c) for c in CALL_ORDER)
    with np.errstate(divide="ignore", invalid="ignore"):
        lu = totals[..., u, look] / counts[..., u]
        lc = (totals[..., cs, look] + totals[..., cr, look]) / (counts[..., cs] + counts[..., cr])
        safe_cs = totals[..., cs, safe] / counts[..., cs]
        risky_cr = totals[..., cr, risky] / counts[..., cr]
        return {"DC": (totals[..., cr, safe] + totals[..., u, safe]) / (counts[..., cr] + counts[..., u]), "LU": lu, "LC": lc, "GAP": lu - lc,
                "CA": (safe_cs + risky_cr) / 2.0, "safe|clear_safe": safe_cs, "risky|clear_risky": risky_cr,
                "safe|unclear": totals[..., u, safe] / counts[..., u], "safe|clear_risky": totals[..., cr, safe] / counts[..., cr]}


def _cluster_arrays(keys: Sequence[tuple], sources: Sequence[CallItems]) -> tuple[list[np.ndarray], np.ndarray, int]:
    """Cluster × class × call sums of each source over `keys`, the cluster × class item counts, and the number of clusters (the
    cluster and the truth class come from the first source; every source must agree on the class)."""
    clusters = sorted({sources[0][k][0] for k in keys})
    index = {c: i for i, c in enumerate(clusters)}
    sums = [np.zeros((len(clusters), len(TRUTHS), len(CALL_ORDER))) for _ in sources]
    counts = np.zeros((len(clusters), len(TRUTHS)))
    for key in keys:
        cluster, truth, _ = sources[0][key]
        i, t = index[cluster], TRUTHS.index(truth)
        counts[i, t] += 1
        for total, items in zip(sums, sources):
            other = items[key]
            if other[1] != truth:
                raise ValueError(f"{key}: truth class {other[1]} vs {truth} — the two models were read on different labels")
            total[i, t] += other[2]
    return sums, counts, len(clusters)


def _draw_weights(clusters: int, n_boot: int, seed: int) -> np.ndarray:
    """How often each cluster is drawn in each bootstrap draw (:func:`robo_jev.bench.system1._draws`, the same draws as every other reading)."""
    from robo_jev.bench.system1 import _draws

    draws = _draws(clusters, n_boot, seed)
    weights = np.zeros((int(n_boot), clusters))
    np.add.at(weights, (np.arange(int(n_boot))[:, None], draws), 1.0)
    return weights


def _value(x: Any) -> float | None:
    x = float(x)
    return x if np.isfinite(x) else None


def call_metrics(items: CallItems, *, n_boot: int = 0, seed: int = 0) -> dict[str, Any]:
    """Every call reading of one model on its items, with cluster-bootstrap 95 % intervals when `n_boot` > 0, and the class sizes."""
    from robo_jev.bench.system1 import _interval

    keys = sorted(items)
    if not keys:
        return {"n": 0}
    (sums,), counts, clusters = _cluster_arrays(keys, [items])
    point = _metrics(sums.sum(0), counts.sum(0))
    out: dict[str, Any] = {"n": len(keys), "clusters": clusters, "classes": dict(zip(TRUTHS, map(int, counts.sum(0))))}
    if n_boot:
        w = _draw_weights(clusters, n_boot, seed)
        boot = _metrics(np.einsum("bc,ctk->btk", w, sums), w @ counts)
    for metric in CALL_METRICS:
        entry: dict[str, Any] = {"value": _value(point[metric])}
        if n_boot:
            entry["low"], entry["high"] = _interval(np.asarray(boot[metric]))
        out[metric] = entry
    return out


def call_difference(items_a: CallItems, items_b: CallItems, *, n_boot: int = 2000, seed: int = 0) -> dict[str, Any]:
    """a − b of every call reading on the items both answered, with the paired cluster-bootstrap 95 % interval (same draws for both)."""
    from robo_jev.bench.system1 import _interval

    keys = sorted(set(items_a) & set(items_b))
    if not keys:
        return {"n": 0}
    (sa, sb), counts, clusters = _cluster_arrays(keys, [items_a, items_b])
    a, b = _metrics(sa.sum(0), counts.sum(0)), _metrics(sb.sum(0), counts.sum(0))
    out: dict[str, Any] = {"n": len(keys), "clusters": clusters, "classes": dict(zip(TRUTHS, map(int, counts.sum(0))))}
    if n_boot:
        w = _draw_weights(clusters, n_boot, seed)
        n = w @ counts
        ba, bb = _metrics(np.einsum("bc,ctk->btk", w, sa), n), _metrics(np.einsum("bc,ctk->btk", w, sb), n)
    for metric in CALL_METRICS:
        va, vb = _value(a[metric]), _value(b[metric])
        entry = {"value": None if va is None or vb is None else va - vb, "a": va, "b": vb, "n": len(keys), "clusters": clusters,
                 "low": None, "high": None}
        if n_boot:
            entry["low"], entry["high"] = _interval(np.asarray(ba[metric]) - np.asarray(bb[metric]))
        entry["excludes_zero"] = bool(entry["low"] is not None and (entry["low"] > 0 or entry["high"] < 0))
        out[metric] = entry
    return out


# --------------------------------------------------------------------------
# The rule (pure — readings in, verdict out)
# --------------------------------------------------------------------------


def _gt(value: float | None, bound: float) -> bool:
    return value is not None and value > bound


def _lt(value: float | None, bound: float) -> bool:
    return value is not None and value < bound


def s_seed_checks(diff: Mapping[str, Any] | None) -> dict[str, bool]:
    """(S) for one seed, from the T9 call differences v0.5 − v0.4 (a missing reading satisfies no condition)."""
    diff = diff or {}
    dc, gap, ca = (diff.get(m) or {} for m in ("DC", "GAP", "CA"))
    return {"dDC_le_-0.05": dc.get("value") is not None and dc["value"] <= S_DC_POINT, "dDC_high_lt_0": _lt(dc.get("high"), S_DC_HIGH),
            "dGAP_ge_+0.05": gap.get("value") is not None and gap["value"] >= S_GAP_POINT, "dGAP_low_gt_0": _gt(gap.get("low"), S_GAP_LOW),
            "dCA_ge_-0.10": ca.get("value") is not None and ca["value"] >= S_CA_POINT, "dDC_low_gt_-0.02": _gt(dc.get("low"), REJECT_DC_LOW)}


def seed_status(checks: Mapping[str, bool]) -> str:
    """One seed's own reading of (S): supported (all three conditions), safer (the ΔDC condition only), rejected (ΔDC lower bound > −0.02)
    or neither — used for the outcome and to name a seed disagreement."""
    dc = checks["dDC_le_-0.05"] and checks["dDC_high_lt_0"]
    if dc and checks["dGAP_ge_+0.05"] and checks["dGAP_low_gt_0"] and checks["dCA_ge_-0.10"]:
        return "supported"
    if dc:
        return "safer"
    if checks["dDC_low_gt_-0.02"]:
        return "rejected"
    return "neither"


def s_outcome(per_seed: Mapping[int, Mapping[str, Any] | None]) -> dict[str, Any]:
    """(S) over both seeds (rule text): SUPPORTED iff each seed meets all three conditions; SAFER-ONLY iff each seed meets the ΔDC condition
    but the ΔGAP or ΔCA condition fails in at least one; REJECTED iff each seed's ΔDC lower bound > −0.02; otherwise PARTIAL.
    `per_seed[seed]` = the T9 call differences of that seed."""
    checks = {int(seed): s_seed_checks(diff) for seed, diff in per_seed.items()}
    if sorted(checks) != list(SEEDS):
        raise ValueError(f"(S) needs both seeds {SEEDS} (got {sorted(checks)})")
    status = {seed: seed_status(c) for seed, c in checks.items()}
    if all(s == "supported" for s in status.values()):
        outcome = "SUPPORTED"
    elif all(s in ("supported", "safer") for s in status.values()):
        outcome = "SAFER-ONLY"
    elif all(s == "rejected" for s in status.values()):
        outcome = "REJECTED"
    else:
        outcome = "PARTIAL"
    return {"outcome": outcome, "checks": checks, "seed_status": status, "seeds_disagree": len(set(status.values())) > 1}


def f_check(miss: Mapping[str, Any] | None) -> dict[str, Any]:
    """(F) for one seed: the T9 robust high-called-low rate of v0.5 (``a``) ≤ the untrained base's (``b``), points on the items both answered."""
    miss = miss or {}
    ok = miss.get("a") is not None and miss.get("b") is not None and miss["a"] <= miss["b"]
    return {"v05": miss.get("a"), "base": miss.get("b"), "n": miss.get("n"), "pass": bool(ok)}


def next_step(outcome: str) -> dict[str, Any]:
    if outcome not in NEXT:
        raise ValueError(f"unknown outcome {outcome!r}")
    return {"branch": outcome, "next": NEXT[outcome]}


def decide(readings: Mapping[str, Any]) -> dict[str, Any]:
    """The whole rule on computed readings: ``readings["seeds"][seed]`` with ``S`` (T9 call differences v0.5 − v0.4 same seed), ``R``, ``N``,
    ``F`` and ``G`` (v0.4's helper reads R, N and G; F is v0.5 vs the base)."""
    v4 = V4()
    seeds = {int(k): v for k, v in readings["seeds"].items()}
    s = s_outcome({seed: entry.get("S") for seed, entry in seeds.items()})
    per_seed = {seed: {"R": v4.r_checks(entry["R"]), "N": v4.n_checks(entry["N"]["T9"], entry["N"]["T8b"], entry["N"]["home"]),
                       "F": f_check(entry["F"]), "G": v4.g_check(entry["G"])} for seed, entry in seeds.items()}
    parts = {part: all(per_seed[seed][part]["pass"] for seed in seeds) for part in ("F", "N", "R", "G")}
    release = s["outcome"] == "SUPPORTED" and all(parts.values())
    failing = ([f"S {s['outcome']}"] if s["outcome"] != "SUPPORTED" else []) + [f"{p} fails" for p, ok in parts.items() if not ok]
    return {"S": s, "per_seed": per_seed, "parts": parts, "release_candidate": release,
            "release_line": "RELEASE CANDIDATE (the user decides)" if release else "not a release candidate: " + ", ".join(failing),
            "next_step": next_step(s["outcome"])}


# --------------------------------------------------------------------------
# Readings
# --------------------------------------------------------------------------


def set_calls(set_name: str, records: Sequence[Mapping[str, Any]], predictions: Sequence[Mapping[str, Any]]) -> CallItems:
    """Call items of a set, clustered by pair group (v0.4's :func:`pair_clusters`)."""
    return call_items(set_name, records, predictions, clusters=V4().pair_clusters(records))


def prediction_check(per_set: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """The rule's secondary prediction (H-set effects larger than T9's, B1 the largest), read on the point differences: ΔDC (more negative
    = larger) and ΔGAP (more positive = larger) per set, and whether the order T9 < H-set < B1 holds for each. Descriptive only."""
    out: dict[str, Any] = {}
    for metric, sign in (("DC", -1.0), ("GAP", 1.0), ("CA", 1.0)):
        values = {s: ((per_set.get(s) or {}).get(metric) or {}).get("value") for s in CALL_SETS}
        out[metric] = {"delta": values}
        if metric != "CA" and all(values[s] is not None for s in CALL_SETS):
            size = {s: sign * values[s] for s in CALL_SETS}
            out[metric]["larger_on_Hset_than_T9"] = size["Hset"] > size["T9"]
            out[metric]["largest_on_B1"] = size["B1"] > max(size["T9"], size["Hset"])
    return out


def compute(sets: Mapping[str, Sequence[Mapping[str, Any]]], preds: Mapping[str, Mapping[str, Sequence[Mapping[str, Any]]]], *, n_boot: int = 2000,
            seed: int = 0, seeds: Sequence[int] = SEEDS) -> dict[str, Any]:
    """Every reading of the rule (and the secondary ones) for each seed. `sets[name]` = records; `preds[model][set]` = predictions for
    models v05s0, v05s1, v04s0, v04s1, base, v03, mix (missing secondary inputs are skipped)."""
    from robo_jev.bench import infamily, system1
    from robo_jev.decisions.metrics import compare_runs
    from robo_jev.decisions.single_probe import compare_probe

    v4 = V4()
    t9, t8b = sets["T9"], sets["T8b"]
    seen = v4.seen_law(t9)
    t9_robust, t9_ambiguous = v4.t9_keys(t9)
    calls = {(m, s): set_calls(s, sets[s], preds[m][s]) for m in preds for s in CALL_SETS if sets.get(s) and preds[m].get(s)}
    out: dict[str, Any] = {
        "seeds": {},
        "sets": {s: {"records": len(sets[s]), "classes": dict(zip(TRUTHS, (sum(1 for r in sets[s] for q in r["request"]["questions"]
                                                                            if truth_class(r, q) == t) for t in TRUTHS)))}
                 for s in CALL_SETS if sets.get(s)},
        "context": {s: {m: call_metrics(calls[(m, s)], n_boot=n_boot, seed=seed) for m in CONTEXT_MODELS if (m, s) in calls} for s in CALL_SETS},
    }
    out["sets"]["T9"].update(seen_law_records=len(seen), seen_law_families=sorted({str(r["meta"].get("family")) for r in seen}),
                             robust_questions=len(t9_robust), prior_dependent_questions=len(t9_ambiguous))
    for s in seeds:
        mine_tag, ref_tag = f"v05s{s}", f"v04s{s}"
        mine, ref = preds[mine_tag], preds[ref_tag]
        entry: dict[str, Any] = {"S": call_difference(calls[(mine_tag, "T9")], calls[(ref_tag, "T9")], n_boot=n_boot, seed=seed)}
        r_sets = {}
        for name in R_SETS:
            if name in sets and mine.get(name) and preds["mix"].get(name):
                comparison = compare_runs(list(sets[name]), list(mine[name]), list(preds["mix"][name]), n_boot=n_boot, seed=seed)
                diffs = comparison["readings"]["single_order"]["differences"]
                r_sets[name] = {key: diffs.get(key) for key in v4.R_LIMITS}
        entry["R"] = r_sets
        a9, b9 = v4.items_of("T9", t9, mine["T9"], by_pair=True), v4.items_of("T9", t9, ref["T9"], by_pair=True)
        a8, b8 = v4.items_of("T8b", t8b, mine["T8b"], by_pair=False), v4.items_of("T8b", t8b, ref["T8b"], by_pair=False)
        home_a = system1.group_items({h: system1.set_items(h, sets[h], mine[h]) for h in HOME}, HOME)
        home_b = system1.group_items({h: system1.set_items(h, sets[h], preds["mix"][h]) for h in HOME}, HOME)
        entry["N"] = {
            "T9": {m: v4._paired(a9.get(m) or {}, b9.get(m) or {}, n_boot=n_boot, seed=seed) for m in ("direction", "comparison")},
            "T8b": {m: v4._paired(a8.get(m) or {}, b8.get(m) or {}, n_boot=n_boot, seed=seed) for m in ("direction", "comparison")},
            "home": {"band_balanced": v4._paired(home_a.get("band_balanced") or {}, home_b.get("band_balanced") or {}, n_boot=n_boot, seed=seed, stratified=True),
                     "severe_error": v4._paired(home_a.get("severe_error") or {}, home_b.get("severe_error") or {}, n_boot=n_boot, seed=seed)},
        }
        base9 = v4.items_of("T9", t9, preds["base"]["T9"], by_pair=True)
        entry["F"] = v4._paired(v4.keep(a9.get("severe_miss") or {}, t9_robust), v4.keep(base9.get("severe_miss") or {}, t9_robust), n_boot=n_boot, seed=seed)
        g2 = compare_probe(list(sets["G2"]), list(mine["G2"]), list(ref["G2"]), dimensions=("domain",), cluster="record", n_boot=n_boot, seed=seed)
        single = g2["readings"]["single_order"]
        entry["G"] = {**single["table"]["_all"]["accuracy"], "questions": single["questions"]}
        sec: dict[str, Any] = {"G2_by_task_vs_v04": {k: v.get("accuracy") for k, v in single["table"]["domain"].items()}}
        if preds["base"].get("G2"):
            vs_base = compare_probe(list(sets["G2"]), list(mine["G2"]), list(preds["base"]["G2"]), dimensions=("domain",), cluster="record", n_boot=n_boot, seed=seed)
            sec["G2_vs_base"] = vs_base["readings"]["single_order"]["table"]["_all"]["accuracy"]
            sec["G2_by_task_vs_base"] = {k: v.get("accuracy") for k, v in vs_base["readings"]["single_order"]["table"]["domain"].items()}
        sec["calls_vs_v04"] = {"T9": entry["S"]}
        for name in ("Hset", "B1"):
            if (mine_tag, name) in calls and (ref_tag, name) in calls:
                sec["calls_vs_v04"][name] = call_difference(calls[(mine_tag, name)], calls[(ref_tag, name)], n_boot=n_boot, seed=seed)
        sec["prediction_check"] = prediction_check(sec["calls_vs_v04"])
        sec["DC_split_T9"] = {m: entry["S"].get(m) for m in ("safe|clear_risky", "safe|unclear")}
        sec["T9_level_vs_v04"] = {"seen_law (v0.4's S metric)": v4.level(seen, "T9", mine["T9"], ref["T9"], n_boot=n_boot, seed=seed),
                                  "all_families": v4.level(t9, "T9", mine["T9"], ref["T9"], n_boot=n_boot, seed=seed)}
        sec["T9_hedge_share_prior_dependent"] = {name: infamily.answer_profile(t9, p, t9_ambiguous).get("hedge_share")
                                                 for name, p in (("v05", mine["T9"]), ("v04", ref["T9"]), ("base", preds["base"]["T9"]))}
        if sets.get("T8a") and mine.get("T8a"):
            pa = system1.set_items("T8a", sets["T8a"], mine["T8a"]).get("choice_accuracy") or {}
            sec["PIQA"] = {rival: v4._paired(pa, system1.set_items("T8a", sets["T8a"], p["T8a"]).get("choice_accuracy") or {}, n_boot=n_boot, seed=seed)
                           for rival, p in (("v04", ref), ("base", preds["base"])) if p.get("T8a")}
        entry["secondary"] = sec
        out["seeds"][str(s)] = entry
    out["seed_spread"] = v4.seed_spread(out["seeds"])
    return out


# --------------------------------------------------------------------------
# Files
# --------------------------------------------------------------------------


def refuse_write(path: Path) -> None:
    resolved = Path(path).resolve()
    for place in NEVER_WRITE:
        place = place.resolve()
        if resolved == place or place in resolved.parents:
            raise SystemExit(f"{path}: refusing to write into {place}")


def is_sealed_t9(path: Path) -> bool:
    return Path(path).resolve() == SET_PATHS["T9"].resolve()


def prediction_paths(args: argparse.Namespace) -> dict[tuple[str, str], Path]:
    """(model, set) → the prediction file the rule reads. Pods: the v0.5 jobs (--s0-dir / --s1-dir), the v0.4 jobs, v0.1-mix, the base's G2.
    Spark: --spark-dir first (v0.5's chain), then — for every model but v0.5 — v0.4's Spark directory, then the earlier tasks' files. With a
    T9 that is not the sealed file (a pipeline check) every T9 file comes from --spark-dir only."""
    paths: dict[tuple[str, str], Path] = {}
    for s, mine, theirs in ((0, args.s0_dir, args.v04_s0_dir), (1, args.s1_dir, args.v04_s1_dir)):
        for name in POD_SETS:
            paths[(f"v05s{s}", name)] = mine / pod_prediction_name("v05", s, name)
            paths[(f"v04s{s}", name)] = theirs / pod_prediction_name("v04", s, name)
    for name in POD_SETS:
        paths[("mix", name)] = args.mix_dir / f"after-v01mix-2b-{name}.predictions.jsonl"
    sealed = is_sealed_t9(args.t9)
    for name in SPARK_SETS:
        for model in SPARK_TAGS:
            fresh = args.spark_dir / spark_prediction_name(name, model)
            if name == "T9" and not sealed:
                paths[(model, name)] = fresh
                continue
            candidates = [fresh] + ([] if model.startswith("v05") else [args.v04_spark_dir / spark_prediction_name(name, model)])
            if (model, name) in EXISTING_SPARK:
                candidates.append(EXISTING_SPARK[(model, name)])
            paths[(model, name)] = next((c for c in candidates if c.is_file()), candidates[0])
    paths[("base", "G2")] = args.base_g2
    paths[("base", "T8a")] = EXISTING_SPARK[("base", "T8a")]
    return paths


def required_inputs() -> list[tuple[str, str]]:
    """What the rule itself reads (missing → refuse); everything else is secondary (missing → skipped and noted)."""
    need = [(m, "T9") for m in ("v05s0", "v05s1", "v04s0", "v04s1", "base")] + [(m, "T8b") for m in ("v05s0", "v05s1", "v04s0", "v04s1")]
    need += [(m, s) for m in ("v05s0", "v05s1", "mix") for s in sorted(set(R_SETS) | set(HOME))]
    need += [(m, "G2") for m in ("v05s0", "v05s1", "v04s0", "v04s1")]
    return need


def _fmt(value: Any, digits: int = 3, signed: bool = False) -> str:
    if value is None:
        return "—"
    return f"{value:+.{digits}f}" if signed else f"{value:.{digits}f}"


def _ci(d: Mapping[str, Any] | None) -> str:
    if not d or d.get("value") is None:
        return "—"
    return f"{_fmt(d['value'], signed=True)} [{_fmt(d.get('low'), signed=True)}, {_fmt(d.get('high'), signed=True)}] ({_fmt(d.get('a'))} vs {_fmt(d.get('b'))})"


def _cell(d: Mapping[str, Any] | None) -> str:
    if not d or d.get("value") is None:
        return "—"
    text = _fmt(d["value"])
    return text + (f" [{_fmt(d['low'])}, {_fmt(d['high'])}]" if d.get("low") is not None else "")


def render(result: Mapping[str, Any]) -> str:
    v, r = result["verdict"], result["readings"]
    s = v["S"]
    lines = [f"# v0.5 pre-declared rule — S {s['outcome']} · {v['release_line']}", ""]
    if result.get("pipeline_check"):
        lines += [f"**PIPELINE CHECK ONLY — T9 is `{result['inputs'].get('t9')}`, not the sealed T9; this is not the v0.5 verdict.**", ""]
    lines += [f"Rule written {RULE['written']}. Applied by `scripts/verdict_v05.py` ({SCRIPT_VERSION}, git {result.get('git')}); "
              f"{result['n_boot']} paired cluster-bootstrap draws (seed {result['seed']}); single order; seed-matched (v0.5-sK vs v0.4-sK). "
              "Hardware: T9, T8b, B1-infamily, H-set on Spark; T1–T7, G2, T8a on the H100 pods.", "",
              "## (S) T9 call view, v0.5 − v0.4 same seed", "",
              "Classes on T9: " + ", ".join(f"{t} {n}" for t, n in r["sets"]["T9"]["classes"].items()) + ".", "",
              "| seed | ΔDC [95 %] (v0.5 vs v0.4) | ΔGAP [95 %] | ΔCA [95 %] | ΔDC ≤ −0.05 | upper < 0 | ΔGAP ≥ +0.05 | lower > 0 | ΔCA ≥ −0.10 | "
              "ΔDC lower > −0.02 | seed reads |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for seed, entry in r["seeds"].items():
        c = s["checks"][int(seed)]
        lines.append(f"| {seed} | {_ci(entry['S'].get('DC'))} | {_ci(entry['S'].get('GAP'))} | {_ci(entry['S'].get('CA'))} | {c['dDC_le_-0.05']} | "
                     f"{c['dDC_high_lt_0']} | {c['dGAP_ge_+0.05']} | {c['dGAP_low_gt_0']} | {c['dCA_ge_-0.10']} | {c['dDC_low_gt_-0.02']} | "
                     f"{s['seed_status'][int(seed)]} |")
    lines += ["", f"**S: {s['outcome']}**" + (" — the two seeds disagree" if s["seeds_disagree"] else ""), "",
              "| seed | LU (v0.5 vs v0.4) | LC (v0.5 vs v0.4) | P(safe | clear-risky) | P(safe | unclear) |", "|---|---|---|---|---|"]
    for seed, entry in r["seeds"].items():
        lines.append(f"| {seed} | {_ci(entry['S'].get('LU'))} | {_ci(entry['S'].get('LC'))} | {_ci(entry['S'].get('safe|clear_risky'))} | "
                     f"{_ci(entry['S'].get('safe|unclear'))} |")
    lines += ["", "## Guards (F) (N) (R) (G) per seed", "", "| seed | F | N | R | G |", "|---|---|---|---|---|"]
    for seed, entry in v["per_seed"].items():
        lines.append(f"| {seed} | " + " | ".join("PASS" if entry[p]["pass"] else "FAIL" for p in ("F", "N", "R", "G")) + " |")
    for seed, entry in r["seeds"].items():
        checks = v["per_seed"][int(seed)]
        lines += ["", f"### seed {seed}", "", "| reading | Δ [95 %] (v0.5 vs reference) | check |", "|---|---|---|"]
        f = checks["F"]
        lines.append(f"| F T9 robust high-called-low (≤ base) | v0.5 {_fmt(f['v05'])} vs base {_fmt(f['base'])} (n {f['n']}) | {f['pass']} |")
        n = entry["N"]
        for label, d, key in (("N T9 direction (vs v0.4)", n["T9"]["direction"], "T9_direction_low_gt_-0.05"),
                              ("N T9 comparison (vs v0.4)", n["T9"]["comparison"], "T9_comparison_low_gt_-0.05"),
                              ("N T8b direction (vs v0.4)", n["T8b"]["direction"], "T8b_direction_low_gt_-0.05"),
                              ("N T8b comparison (vs v0.4)", n["T8b"]["comparison"], "T8b_comparison_low_gt_-0.05"),
                              ("N T2–T4 balanced band (vs v0.1-mix)", n["home"]["band_balanced"], "T2-T4_band_low_gt_-0.05"),
                              ("N T2–T4 severe (vs v0.1-mix + 0.01)", n["home"]["severe_error"], "T2-T4_severe_le_mix_plus_0.01")):
            lines.append(f"| {label} | {_ci(d)} | {checks['N']['checks'][key]} |")
        for name, sets_ in checks["R"]["sets"].items():
            for key, d in sets_.items():
                lines.append(f"| R {name} {key} (vs v0.1-mix) | {_fmt(d.get('value'), 4, True)} [{_fmt(d.get('low'), 4, True)}, {_fmt(d.get('high'), 4, True)}] | {d['ok']} |")
        g = entry["G"]
        lines.append(f"| G G2 accuracy (vs v0.4) | {_fmt(g.get('diff'), signed=True)} [{_fmt(g.get('low'), signed=True)}, {_fmt(g.get('high'), signed=True)}] "
                     f"({_fmt(g.get('a'))} vs {_fmt(g.get('b'))}) | {checks['G']['pass']} |")
    ns = v["next_step"]
    lines += ["", "## Release and next step", "", f"- **{v['release_line']}**", f"- Next step ({ns['branch']}): {ns['next']}.", f"- {RULE['prior']}", "",
              "## Secondary (reported, not deciding)", ""]
    for seed, entry in r["seeds"].items():
        sec = entry["secondary"]
        lines += [f"### seed {seed}", "", "Call view v0.5 − v0.4 (same seed) per set:", "", "| set | ΔDC | ΔLU | ΔLC | ΔGAP | ΔCA |", "|---|---|---|---|---|---|"]
        for name, d in sec["calls_vs_v04"].items():
            lines.append(f"| {name} | " + " | ".join(_ci(d.get(m)) for m in ("DC", "LU", "LC", "GAP", "CA")) + " |")
        pc = sec["prediction_check"]
        lines += ["", "Prediction (H-set > T9, B1 largest): " + "; ".join(
            f"{m}: " + ", ".join(f"{k} {_fmt(x, signed=True)}" for k, x in e["delta"].items())
            + (f" — H-set larger than T9 {e.get('larger_on_Hset_than_T9')}, B1 largest {e.get('largest_on_B1')}" if "largest_on_B1" in e else "")
            for m, e in pc.items()) + ".", "", "| reading | Δ [95 %] (v0.5 vs reference) |", "|---|---|"]
        for name, d in sec["DC_split_T9"].items():
            lines.append(f"| T9 DC split P({name}) vs v0.4 | {_ci(d)} |")
        for name, d in sec["T9_level_vs_v04"].items():
            lines.append(f"| T9 balanced band, {name} vs v0.4 | {_ci(d)} |")
        for rival, d in (sec.get("PIQA") or {}).items():
            lines.append(f"| PIQA (T8a) vs {rival} | {_ci(d)} |")
        for rival in ("v04", "base"):
            for task, d in (sec.get(f"G2_by_task_vs_{rival}") or {}).items():
                d = d or {}
                lines.append(f"| G2 {task} vs {rival} | {_fmt(d.get('diff'), signed=True)} [{_fmt(d.get('low'), signed=True)}, {_fmt(d.get('high'), signed=True)}] "
                             f"({_fmt(d.get('a'))} vs {_fmt(d.get('b'))}) |")
        hedge = sec.get("T9_hedge_share_prior_dependent") or {}
        lines += ["", "Hedge share on prior-dependent T9 questions: " + ", ".join(f"{m} {_fmt(x)}" for m, x in hedge.items()) + ".", ""]
    spread = r.get("seed_spread") or {}
    if spread:
        lines += ["Seed spread (|s0 − s1| of v0.5's value), largest ten: " + "; ".join(
            f"{k} {_fmt(e['abs_diff'])}" for k, e in sorted(spread.items(), key=lambda kv: -kv[1]["abs_diff"])[:10]) + ".", ""]
    lines += ["## Context — every model's call view per set (single order, 95 % cluster bootstrap)", ""]
    for name, models in r["context"].items():
        if not models:
            continue
        lines += [f"**{name}** — classes " + ", ".join(f"{t} {k}" for t, k in r["sets"][name]["classes"].items()), "",
                  "| model | DC | LU | LC | GAP | CA | P(safe|clear-safe) | P(risky|clear-risky) | P(safe|unclear) | P(safe|clear-risky) |",
                  "|---|---|---|---|---|---|---|---|---|---|"]
        for model, m in models.items():
            lines.append(f"| {MODELS.get(model, model)} | " + " | ".join(_cell(m.get(k)) for k in CALL_METRICS) + " |")
        lines.append("")
    lines += ["## Notes", ""] + [f"- {note}" for note in result["notes"]] + ["", "## Sources", "", "| model / set | predictions | same set file | coverage |", "|---|---|---|---|"]
    for key, src in result["sources"].items():
        lines.append(f"| {key} | `{src['predictions']}` | {not src['problems']} | {_fmt(src['coverage'])} |")
    lines += ["", "## Rule (verbatim parts)", ""] + [f"- **{k}** — {t}" for k, t in RULE.items()]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--s0-dir", type=Path, required=True, help="fetched reports of the v0.5 seed-0 job (after-v05-s0-2b-*)")
    parser.add_argument("--s1-dir", type=Path, required=True, help="fetched reports of the v0.5 seed-1 job (after-v05-s1-2b-*)")
    parser.add_argument("--v04-s0-dir", type=Path, default=V04_PODS[0])
    parser.add_argument("--v04-s1-dir", type=Path, default=V04_PODS[1])
    parser.add_argument("--mix-dir", type=Path, default=MIX_POD)
    parser.add_argument("--spark-dir", type=Path, default=SPARK_DIR, help="v0.5's Spark predictions (<SET>-<tag>.predictions.jsonl); read first")
    parser.add_argument("--v04-spark-dir", type=Path, default=V04_SPARK_DIR, help="v0.4's Spark predictions (v0.4-s0/s1, v0.3, base on the H-set)")
    parser.add_argument("--base-g2", type=Path, default=BASE_G2_POD)
    parser.add_argument("--hset", type=Path, default=SET_PATHS["Hset"], help="v0.4a's scene holdout (H-set)")
    parser.add_argument("--t9", type=Path, default=SET_PATHS["T9"], help="the sealed T9 (default); another file only for a pipeline check")
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--allow-mismatch", action="store_true", help="read even if a prediction file was scored on another set file or conditions")
    parser.add_argument("--out", type=Path, default=REPO / "artifacts" / "reports" / "v05" / "verdict.json")
    args = parser.parse_args(argv)
    refuse_write(args.out)
    set_paths = {**SET_PATHS, "Hset": args.hset, "T9": args.t9}
    pipeline = not is_sealed_t9(args.t9)
    if pipeline:
        _log(f"T9 is not the sealed file: {args.t9} — a pipeline check, not the verdict; T9 predictions only from {args.spark_dir}")
    paths = prediction_paths(args)
    missing = [f"{m} {s}: {paths[(m, s)]}" for m, s in required_inputs() if not paths[(m, s)].is_file()]
    if missing:
        raise SystemExit("missing prediction files the rule reads:\n  " + "\n  ".join(missing))
    needed_sets = sorted({s for (_, s), p in paths.items() if p.is_file()})
    sets = {s: V4()._read(set_paths[s]) for s in needed_sets if set_paths[s].is_file()}
    preds: dict[str, dict[str, list[dict[str, Any]]]] = {m: {} for m in MODELS}
    sources, notes = {}, []
    for (model, name), path in sorted(paths.items()):
        if not path.is_file() or name not in sets:
            if not path.is_file():
                notes.append(f"secondary input absent, skipped: {model} {name} ({path})")
            continue
        preds[model][name] = V4()._read(path)
        sources[f"{model} {name}"] = V4().check_source(path, set_paths[name], sets[name], preds[model][name])
    problems = [f"{k}: {'; '.join(v['problems'])}" for k, v in sources.items() if v["problems"]]
    if not pipeline:
        seen = V4().seen_law(sets["T9"])
        if len(sets["T9"]) != 600 or len(seen) != V4().SEEN_LAW_RECORDS or sorted({str(r["meta"].get("family")) for r in seen}) != sorted(V4().SEEN_LAW_FAMILIES):
            problems.append(f"T9 has {len(sets['T9'])} records and a seen-law slice of {len(seen)} — not the sealed T9 the rule reads")
    if problems and not args.allow_mismatch:
        raise SystemExit("refusing (use --allow-mismatch to read anyway):\n  " + "\n  ".join(problems))
    readings = compute(sets, preds, n_boot=args.n_boot, seed=args.seed)
    verdict = decide(readings)
    if pipeline:
        notes.insert(0, f"PIPELINE CHECK ONLY: T9 is {args.t9}, not the sealed T9 — this is not the v0.5 verdict")
    notes += ["T9, T8b, B1-infamily and the H-set were scored on Spark (GB10) for every model; T1–T7, G2 and T8a on H100 pods (v0.5, v0.4, "
              "v0.1-mix; the base's G2 on the v0.1 main pod; the base's T8a on Spark).",
              "Call view: each listing of a question weighs 1/L (single order); truth classes from each set's own meta.robustness (T9: base, "
              "low_reader, high_reader; H-set and B1: base, alt_a, alt_b) and the band of the base-prior label; clusters = pair groups.",
              "(R) reads scripts/diagnose_decisions.py --compare's differences (compare_runs, single order, record bootstrap) with the rule's 2,000 "
              "draws; (N) and (G) and the file checks are v0.4's helper's (scripts/verdict_v04.py).",
              "(F) is read on the robust items both models answered; the high-called-low rate's denominator is the robust yes/no + score questions "
              "(as in the v0.4 verdict)."]
    result = {"kind": "v05-verdict", "script_version": SCRIPT_VERSION, "git": _git_sha(), "created": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
              "rule_text": RULE, "n_boot": args.n_boot, "seed": args.seed, "pipeline_check": pipeline, "verdict": verdict, "readings": readings,
              "sources": sources, "problems": problems, "notes": notes, "inputs": {k: str(v) for k, v in vars(args).items() if isinstance(v, Path)}}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=1, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    args.out.with_suffix(".md").write_text(render(result), encoding="utf-8")
    print(render(result).split("## Secondary")[0])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
