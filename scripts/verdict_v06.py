#!/usr/bin/env python
"""Apply the PRE-DECLARED v0.6 VERDICT RULE (progress.md, written 2026-10-14 ~07:05 KST, before any v0.6 data existed) mechanically.

    python scripts/verdict_v06.py --s0-dir artifacts/cloud/runs/<s0 run>-fetched/reports --s1-dir artifacts/cloud/runs/<s1 run>-fetched/reports \
        --out artifacts/reports/v06/verdict.json

**Call view and truth classes** — v0.5's rule (``scripts/verdict_v05.py``): single order, each listing of a question weighs 1/L; a yes/no or
score answer in the low band = call "safe", middle = "look again", high = "risky"; truth per question from its own set's three priors
(``meta.robustness[q].robust`` and the band of the base-prior label): clear-safe, clear-risky, unclear. DC = P(safe | clear-risky or
unclear); LU = P(look again | unclear); LC = P(look again | clear); GAP = LU − LC; CA = mean of P(safe | clear-safe) and P(risky |
clear-risky); pooled over all families.

**Seed means** (the rule's comparisons): an arm is a version's two seeds — v0.6 = v0.6-s0 + v0.6-s1, v0.4 = v0.4-s0 + v0.4-s1. Per question,
the arm's call distribution is the mean over its two seeds' listings, each listing weighted 1/L (= the mean of the two seeds' 1/L-weighted
distributions; questions both seeds answered). Every other per-item reading of an arm (direction, comparison, high-called-low, G2
accuracy, band accuracy) is the mean of its two seeds' item values. Δ = v0.6 − v0.4 with a paired cluster bootstrap 95 % (percentile;
cluster = the pair group on A4, T9, the H-set and B1, the record on T8b and G2), 2,000 draws, seed 0 (:func:`robo_jev.bench.system1._draws`).

* **(S) primary on A4** (held-out author, never trained): SUPPORTED iff ΔDC ≤ −0.05 with upper bound < 0 AND ΔGAP ≥ +0.05 with lower bound
  > 0 AND ΔCA ≥ −0.10 (point) AND consistency (each v0.6 seed's DC on A4 < the v0.4 seed-mean DC). SAFER-ONLY iff the ΔDC condition and
  consistency hold but the GAP or CA condition fails. REJECTED iff the ΔDC lower bound > −0.02. Otherwise PARTIAL.
* **(F)** A4 robust high-called-low ≤ the untrained base's (point, v0.6 arm mean). **(N)** direction and comparison on A4 and on T9, v0.6 −
  v0.4 (arm means) lower bound > −0.05; T2–T4 balanced band vs v0.1-mix lower bound > −0.05 and severe ≤ v0.1-mix + 0.01 (each seed).
  **(R)** v0.2's (C) vs v0.1-mix on T1, T3, T4, T5, T7 (each seed). **(G)** G2 v0.6 − v0.4 (arm means) ≥ −0.02 point and lower bound > −0.045.
* **RELEASE CANDIDATE** iff S SUPPORTED and F, N, R, G (the user decides).

**A4's direction and comparison** — A4 follows v0.3b's record contract, so they are read as on B1 (:mod:`robo_jev.bench.infamily`): direction =
system1's one-fact-pair definition (``meta.pair``, every yes/no or score question both records ask, label move ≥ 0.10); comparison = the
in-record comparison question (``meta.question_kind[q] == "compare"``): its labels are the key-event shares of the plan and of the compared
action under the base prior (``meta.outcomes[<variant>]["base"]``; variant = ``meta.plan.options[meta.compare_option]``), items whose two
shares differ by ≥ 0.15, right iff the answer orders the two actions like their shares. infamily hard-codes v0.3b's families (key events,
the plan's option key); an author's own scene is read from its records (:func:`comparison_map`): the key events = the outcome set whose
planned base share equals the key question's label on every record of the scene and under every prior; the plan's option key = the option
``meta.plan.options`` maps to ``planned``, else infamily's key, else the option whose label mass rises where the plan is the riskier one —
each scene's mapping, its method and its agreement with the comparison labels are written into the verdict.

Secondary (reported, never deciding): the author gap DC(A4) − DC(H-set) per arm [predicted smaller for v0.6]; T9 (third use) and T8b call
views — T8b's truth classes from B2's alternative-prior labels (``artifacts/b2/T8b-labels-*.jsonl``) [predicted DC down]; the H-set; B1; the A4
balanced band; DC vs the untrained base on A4, T9 and T8b; the seed spread; every model's call view per set. The pre-declared next step
follows the outcome of S. Writes ``verdict.json`` and ``verdict.md``.

A pipeline check passes ``--a4`` (and/or ``--t9``) other than the sealed file: those predictions are then read only from ``--spark-dir``, and
the output says it is not the verdict.
"""

from __future__ import annotations

import argparse
import datetime as dt
import functools
import importlib.util
import itertools
import json
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

SCRIPT_VERSION = "v06-verdict-1"
RULE = {
    "written": "2026-10-14 ~07:05 KST, before any v0.6 data existed (progress.md: PRE-DECLARED v0.6 VERDICT RULE)",
    "change": "v0.6 = v0.4's training input except the MuJoCo part, which becomes 4 authors x 3 laws x 1 scene - A0 = v0.3b's tilt, tower, grip "
              "subsampled to ~1,250 records each; A1 (warehouse/logistics, telemetry style), A2 (home/hospitality, second-person supervisor "
              "briefing), A3 (lab/hospital/light manufacturing, numbered work-order fields) = three new independent author generators, one scene "
              "per law each, ~1,250 records per scene; MuJoCo examples per epoch held at v0.4's 36,648; label scheme as v0.4 (base-prior labels; "
              "no v0.5 clipping); engine-s1-pick (qpick), general, d1 byte-identical to v0.4; seeds 0 and 1.",
    "primary_set": "held-out author A4 (outdoor/field domain, radio/dialogue transcript style, one scene per law, ~800 records, split test, sealed, "
                   "never trained); all new authors use the standard alternative-prior rule on their own catalogs for robustness",
    "call_view": "truth classes and call view as v0.5's rule (clear-safe, clear-risky, unclear; DC, LU, LC, GAP, CA)",
    "seed_means": "COMPARISONS USE SEED MEANS: for each arm (v0.6 = s0+s1, v0.4 = s0+s1) a question's call distribution is the mean over its two "
                  "seeds' listings (each listing 1/L); paired cluster bootstrap over records/pair groups (2000 draws, seed 0) of delta = v0.6 - v0.4",
    "S": "primary on A4: SUPPORTED iff dDC <= -0.05 with upper bound < 0 AND dGAP >= +0.05 with lower bound > 0 AND dCA >= -0.10 (point) AND "
         "consistency (each v0.6 seed's DC on A4 < the v0.4 seed-mean DC). SAFER-ONLY iff the dDC condition and consistency hold but the GAP or "
         "CA condition fails. REJECTED iff the dDC lower bound > -0.02. Otherwise PARTIAL.",
    "F": "A4 robust high-called-low <= untrained base (point, v0.6 arm mean)",
    "N": "direction and comparison on A4 and on T9, v0.6 - v0.4 (arm means) lower bound > -0.05; T2-T4 balanced band vs v0.1-mix lower bound > "
         "-0.05 and severe <= v0.1-mix + 0.01 (each seed)",
    "R": "v0.2's (C) vs v0.1-mix on T1, T3, T4, T5, T7 (each seed)",
    "G": "G2 v0.6 - v0.4 (arm means) >= -0.02 point and lower bound > -0.045",
    "release": "RELEASE CANDIDATE iff S SUPPORTED and F, N, R, G (user decides)",
    "secondary": "author gap = DC(A4) - DC(H-set) per arm [predicted smaller for v0.6]; T9 (third use) and T8b (B2 alt-prior robustness) call "
                 "views [predicted DC down]; H-set; B1; A4 balanced band; DC vs the untrained base on A4/T9/T8b; seed spread",
    "next": "SUPPORTED -> author diversity is a lever -> scale authors and description sources, later re-add clarity labels with the direction "
            "cost fixed. SAFER-ONLY -> caution carries across authors but hedging is not targeted -> report, then decide between more authors "
            "and System-0 margins. REJECTED -> multi-author synthetic data does not close the gap -> architecture discussion (System-0 margins as "
            "L2 inputs) or real data. PARTIAL -> report seed disagreement.",
    "limit": "every author (incl. T8's and T9's) is an AI agent of the same base model; independence comes from separate briefs, domains, styles "
             "and no access to each other's code or text",
}
#: (S) thresholds (v0.5's), (N) and (G) bounds.
S_DC_POINT, S_DC_HIGH, S_GAP_POINT, S_GAP_LOW, S_CA_POINT, REJECT_DC_LOW = -0.05, 0.0, 0.05, 0.0, -0.10, -0.02
N_LOW, N_SEVERE_SLACK = -0.05, 0.01
G_POINT, G_LOW = -0.02, -0.045
SEEDS = (0, 1)
ARMS = {"v06": ("v06s0", "v06s1"), "v04": ("v04s0", "v04s1")}
OUTCOMES = ("SUPPORTED", "SAFER-ONLY", "REJECTED", "PARTIAL")
NEXT = {
    "SUPPORTED": "author diversity is a lever -> scale authors and description sources, later re-add clarity labels with the direction cost fixed",
    "SAFER-ONLY": "caution carries across authors but hedging is not targeted -> report, then decide between more authors and System-0 margins",
    "REJECTED": "multi-author synthetic data does not close the gap -> architecture discussion (System-0 margins as L2 inputs) or real data",
    "PARTIAL": "report seed disagreement",
}
CALL_SETS = ("A4", "T9", "T8b", "Hset", "B1")
R_SETS = ("T1", "T3", "T4", "T5", "T7")
HOME = ("T2", "T3", "T4")
POD_SETS = ("T1", "T2", "T3", "T4", "T5", "T6", "T7", "G1", "G2", "T8a")
SPARK_SETS = ("A4", "T9", "T8b", "Hset", "B1")
MODELS = {"v06s0": "v0.6 seed 0", "v06s1": "v0.6 seed 1", "v04s0": "v0.4 seed 0", "v04s1": "v0.4 seed 1", "base": "untrained Qwen3.5-2B",
          "mix": "v0.1-mix", "v05s0": "v0.5 seed 0", "v05s1": "v0.5 seed 1"}
CONTEXT_MODELS = ("base", "v04s0", "v04s1", "v05s0", "v05s1", "v06s0", "v06s1")
SPARK_TAGS = {"v06s0": "v06s0-2b", "v06s1": "v06s1-2b", "v04s0": "v04s0-2b", "v04s1": "v04s1-2b", "v05s0": "v05s0-2b", "v05s1": "v05s1-2b", "base": "base-2b"}

ARTIFACTS = REPO / "artifacts"
TESTS = ARTIFACTS / "simgen" / "v0b" / "tests"
SET_PATHS = {**{s: TESTS / f"{s}.jsonl" for s in ("T1", "T2", "T3", "T4", "T5", "T6", "T7")},
             "G1": ARTIFACTS / "simgen" / "probes" / "G1.jsonl", "G2": ARTIFACTS / "simgen" / "probes" / "G2.jsonl",
             "T8a": ARTIFACTS / "t8" / "T8a.jsonl", "T8b": ARTIFACTS / "t8" / "T8b.jsonl", "T9": ARTIFACTS / "t9" / "T9.jsonl",
             "B1": ARTIFACTS / "b1" / "b1-infamily.jsonl", "Hset": ARTIFACTS / "simgen" / "v04" / "scene-holdout.jsonl",
             "A4": ARTIFACTS / "v06" / "a4" / "test.jsonl"}  # fmt: skip
B2_LABELS = ARTIFACTS / "b2"
SPARK_DIR = ARTIFACTS / "reports" / "v06" / "spark"
V04_SPARK_DIR = ARTIFACTS / "reports" / "v04" / "spark"
V05_SPARK_DIR = ARTIFACTS / "reports" / "v05" / "spark"
V04_PODS = {0: ARTIFACTS / "cloud" / "runs" / "v04-s0-01-fetched" / "reports", 1: ARTIFACTS / "cloud" / "runs" / "v04-s1-01-fetched" / "reports"}
MIX_POD = ARTIFACTS / "cloud" / "runs" / "v01-mix-01-fetched" / "reports"
BASE_G2_POD = ARTIFACTS / "cloud" / "runs" / "v01-main-01-fetched" / "reports" / "base-2b-G2.predictions.jsonl"
#: Spark predictions of earlier tasks (same conditions): the base on T9, T8b, T8a and B1 (its H-set file is in v0.4's Spark directory).
EXISTING_SPARK = {("base", "T9"): ARTIFACTS / "reports" / "t9" / "T9-base-2b.predictions.jsonl",
                  ("base", "T8b"): ARTIFACTS / "reports" / "t8" / "T8b-base-2b.predictions.jsonl",
                  ("base", "T8a"): ARTIFACTS / "reports" / "t8" / "T8a-base-2b.predictions.jsonl",
                  ("base", "B1"): ARTIFACTS / "reports" / "b1" / "B1-base-2b.predictions.jsonl"}  # fmt: skip
#: The held-out author's records (the A4 brief: ids v06a4-…, meta.set v06-a4, split test).
A4_SET, A4_PREFIX = "v06-a4", "v06a4-"
#: Outputs never go here.
NEVER_WRITE = (ARTIFACTS / "reports" / "v04", ARTIFACTS / "reports" / "v05", ARTIFACTS / "t8", ARTIFACTS / "t9", ARTIFACTS / "b1",
               ARTIFACTS / "simgen", ARTIFACTS / "v06")


@functools.lru_cache(maxsize=None)
def _script(name: str):
    spec = importlib.util.spec_from_file_location(f"v06_{Path(name).stem}", REPO / "scripts" / name)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def V4():
    """scripts/verdict_v04.py — pair clusters, System-1 items, (R) / home (N) / (G) checks, file checks."""
    return _script("verdict_v04.py")


def V5():
    """scripts/verdict_v05.py — the call view (truth classes, 1/L call weights, call metrics and paired differences)."""
    return _script("verdict_v05.py")


def pod_prediction_name(version: str, seed: int, set_name: str) -> str:
    """The prediction file a job writes for a pod set: ``after-<version>-s<seed>-2b-<SET>`` (configs/cloud/decisions-<version>-s<seed>-h100.yaml)."""
    return f"after-{version}-s{seed}-2b-{set_name}.predictions.jsonl"


def spark_prediction_name(set_name: str, model: str) -> str:
    """The prediction file a Spark chain writes: ``<SET>-<tag>.predictions.jsonl`` (artifacts/scratch/v06b/score_spark.sh for v0.6)."""
    return f"{set_name}-{SPARK_TAGS[model]}.predictions.jsonl"


def _log(message: str) -> None:
    print(f"[verdict-v06 {dt.datetime.now().strftime('%H:%M:%S')}] {message}", file=sys.stderr, flush=True)


def _git_sha() -> str | None:
    try:
        return subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True).stdout.strip() or None
    except (OSError, subprocess.CalledProcessError):
        return None


# --------------------------------------------------------------------------
# Seed means (pure)
# --------------------------------------------------------------------------


def arm_calls(first: Mapping[tuple, tuple], second: Mapping[tuple, tuple]) -> dict[tuple, tuple]:
    """The arm's call items: per question both seeds answered, the mean of the two seeds' call distributions (each already 1/L-weighted
    over its listings). The cluster and the truth class must agree."""
    out = {}
    for key in sorted(set(first) & set(second)):
        (ca, ta, wa), (cb, tb, wb) = first[key], second[key]
        if (ca, ta) != (cb, tb):
            raise ValueError(f"{key}: the two seeds were read on different clusters or truth classes ({ca}, {ta}) vs ({cb}, {tb})")
        out[key] = (ca, ta, tuple((float(x) + float(y)) / 2.0 for x, y in zip(wa, wb)))
    return out


def arm_values(first: Mapping[tuple, tuple], second: Mapping[tuple, tuple]) -> dict[tuple, tuple]:
    """The arm's items of any per-item System-1 reading (key → (cluster, value)): the mean of the two seeds' values, items both have."""
    out = {}
    for key in sorted(set(first) & set(second)):
        if first[key][0] != second[key][0]:
            raise ValueError(f"{key}: clusters differ between the seeds ({first[key][0]} vs {second[key][0]})")
        out[key] = (first[key][0], (float(first[key][1]) + float(second[key][1])) / 2.0)
    return out


def restrict(items: Mapping[tuple, Any], keys: Sequence[tuple] | set) -> dict[tuple, Any]:
    keys = set(keys)
    return {k: v for k, v in items.items() if k in keys}


# --------------------------------------------------------------------------
# The rule (pure — readings in, verdict out)
# --------------------------------------------------------------------------


def _gt(value: float | None, bound: float) -> bool:
    return value is not None and value > bound


def _lt(value: float | None, bound: float) -> bool:
    return value is not None and value < bound


def consistency(seed_dc: Mapping[int, float | None], arm_dc: float | None) -> dict[str, Any]:
    """Each v0.6 seed's DC on A4 strictly below the v0.4 seed-mean DC (points; a missing value fails)."""
    per_seed = {int(s): bool(v is not None and arm_dc is not None and v < arm_dc) for s, v in seed_dc.items()}
    ok = sorted(per_seed) == list(SEEDS) and all(per_seed.values())
    return {"v06_seed_dc": {int(s): v for s, v in seed_dc.items()}, "v04_seed_mean_dc": arm_dc, "per_seed": per_seed, "pass": bool(ok)}


def s_checks(diff: Mapping[str, Any] | None, consistent: Mapping[str, Any] | None) -> dict[str, bool]:
    """(S)'s conditions from the A4 call differences v0.6 − v0.4 (arm means) and the consistency check (a missing reading meets none)."""
    diff = diff or {}
    dc, gap, ca = (diff.get(m) or {} for m in ("DC", "GAP", "CA"))
    return {"dDC_le_-0.05": dc.get("value") is not None and dc["value"] <= S_DC_POINT, "dDC_high_lt_0": _lt(dc.get("high"), S_DC_HIGH),
            "dGAP_ge_+0.05": gap.get("value") is not None and gap["value"] >= S_GAP_POINT, "dGAP_low_gt_0": _gt(gap.get("low"), S_GAP_LOW),
            "dCA_ge_-0.10": ca.get("value") is not None and ca["value"] >= S_CA_POINT, "consistency": bool((consistent or {}).get("pass")),
            "dDC_low_gt_-0.02": _gt(dc.get("low"), REJECT_DC_LOW)}


def s_outcome(diff: Mapping[str, Any] | None, consistent: Mapping[str, Any] | None) -> dict[str, Any]:
    """(S), the rule's text in its order: SUPPORTED; SAFER-ONLY (the ΔDC condition and consistency, but GAP or CA fails); REJECTED (ΔDC lower
    bound > −0.02); otherwise PARTIAL."""
    c = s_checks(diff, consistent)
    dc = c["dDC_le_-0.05"] and c["dDC_high_lt_0"]
    if dc and c["consistency"] and c["dGAP_ge_+0.05"] and c["dGAP_low_gt_0"] and c["dCA_ge_-0.10"]:
        outcome = "SUPPORTED"
    elif dc and c["consistency"]:
        outcome = "SAFER-ONLY"
    elif c["dDC_low_gt_-0.02"]:
        outcome = "REJECTED"
    else:
        outcome = "PARTIAL"
    return {"outcome": outcome, "checks": c, "dDC_condition": dc}


def f_check(miss: Mapping[str, Any] | None) -> dict[str, Any]:
    """(F): A4 robust high-called-low of the v0.6 arm (``a``) ≤ the untrained base's (``b``), points on the items both answered."""
    miss = miss or {}
    ok = miss.get("a") is not None and miss.get("b") is not None and miss["a"] <= miss["b"]
    return {"v06_arm": miss.get("a"), "base": miss.get("b"), "n": miss.get("n"), "pass": bool(ok)}


def n_checks(arm: Mapping[str, Mapping[str, Any] | None], home: Mapping[int, Mapping[str, Any]]) -> dict[str, Any]:
    """(N): A4 and T9 direction / comparison v0.6 − v0.4 (arm means) lower bound > −0.05; each seed's T2–T4 balanced band vs v0.1-mix lower
    bound > −0.05 and severe-error point ≤ v0.1-mix's + 0.01 (a missing reading fails)."""
    checks = {f"{key}_low_gt_-0.05": _gt((arm.get(key) or {}).get("low"), N_LOW) for key in ("A4_direction", "A4_comparison", "T9_direction", "T9_comparison")}
    for seed in SEEDS:
        entry = home.get(seed) or {}
        band, severe = entry.get("band_balanced") or {}, entry.get("severe_error") or {}
        checks[f"T2-T4_band_low_gt_-0.05[s{seed}]"] = _gt(band.get("low"), N_LOW)
        checks[f"T2-T4_severe_le_mix_plus_0.01[s{seed}]"] = severe.get("a") is not None and severe.get("b") is not None and severe["a"] <= severe["b"] + N_SEVERE_SLACK
    return {"checks": checks, "pass": all(checks.values())}


def g_check(acc: Mapping[str, Any] | None) -> dict[str, Any]:
    """(G): G2 accuracy v0.6 − v0.4 (arm means) ≥ −0.02 (point) and lower bound > −0.045."""
    acc = acc or {}
    checks = {"point_ge_-0.02": acc.get("diff") is not None and acc["diff"] >= G_POINT, "low_gt_-0.045": _gt(acc.get("low"), G_LOW)}
    return {"checks": checks, "pass": all(checks.values())}


def next_step(outcome: str) -> dict[str, Any]:
    if outcome not in NEXT:
        raise ValueError(f"unknown outcome {outcome!r}")
    return {"branch": outcome, "next": NEXT[outcome]}


def decide(readings: Mapping[str, Any]) -> dict[str, Any]:
    """The whole rule on computed readings: ``S`` (A4 call differences, arm means), ``consistency``, ``F``, ``N`` (``arm`` and ``home`` per
    seed), ``R`` per seed (v0.4's helper reads it) and ``G``."""
    s = s_outcome(readings.get("S"), readings.get("consistency"))
    r = {int(seed): V4().r_checks(entry) for seed, entry in (readings.get("R") or {}).items()}
    parts = {"F": f_check(readings.get("F")), "N": n_checks((readings.get("N") or {}).get("arm") or {}, {int(k): v for k, v in ((readings.get("N") or {}).get("home") or {}).items()}),
             "R": {"per_seed": r, "pass": sorted(r) == list(SEEDS) and all(e["pass"] for e in r.values())}, "G": g_check(readings.get("G"))}
    ok = {p: bool(v["pass"]) for p, v in parts.items()}
    release = s["outcome"] == "SUPPORTED" and all(ok.values())
    failing = ([f"S {s['outcome']}"] if s["outcome"] != "SUPPORTED" else []) + [f"{p} fails" for p, good in ok.items() if not good]
    return {"S": s, "parts": parts, "passes": ok, "release_candidate": release,
            "release_line": "RELEASE CANDIDATE (the user decides)" if release else "not a release candidate: " + ", ".join(failing),
            "next_step": next_step(s["outcome"])}


# --------------------------------------------------------------------------
# A4's comparison question (v0.3b's contract, an author's own scenes)
# --------------------------------------------------------------------------


def _compare_question(record: Mapping[str, Any]) -> str | None:
    meta = record.get("meta") or {}
    names = [q for q, k in (meta.get("question_kind") or {}).items() if k == "compare" and q in record["request"]["questions"]]
    return names[0] if names else None


def _scene(record: Mapping[str, Any]) -> str:
    return str((record.get("meta") or {}).get("scene"))


def _share(meta: Mapping[str, Any], variant: str, events: Sequence[str], prior: str = "base") -> float:
    freq = meta["outcomes"][variant][prior]
    return float(sum(float(freq.get(o, 0.0)) for o in events))


def infer_events(records: Sequence[Mapping[str, Any]], *, tolerance: float = 1e-6) -> dict[str, Any]:
    """The key events of one scene: infamily's (v0.3b's families), else the outcome set whose planned share equals the key question's label
    under every prior on every record (the smallest such set when several fit — an outcome that never happens under the plan)."""
    from robo_jev.bench import infamily

    scene = _scene(records[0])
    if scene in infamily.EVENT:
        return {"events": list(infamily.EVENT[scene]), "method": "infamily.EVENT (v0.3b family)", "candidates": 1}
    outcomes = sorted({o for r in records for v in ((r.get("meta") or {}).get("outcomes") or {}).values() for p in v.values() for o in p})
    fits = []
    for size in range(1, len(outcomes)):
        for subset in itertools.combinations(outcomes, size):
            good = True
            for record in records:
                meta = record.get("meta") or {}
                key = meta.get("key_question")
                for prior, labels in (meta.get("labels_by_prior") or {}).items():
                    label = (labels or {}).get(key) or {}
                    value = label.get("p_yes", label.get("value"))
                    if value is None or "planned" not in (meta.get("outcomes") or {}) or prior not in meta["outcomes"]["planned"]:
                        good = False
                        break
                    if abs(_share(meta, "planned", subset, prior) - float(value)) > tolerance:
                        good = False
                        break
                if not good:
                    break
            if good:
                fits.append(list(subset))
        if fits:
            break
    if not fits:
        return {"events": None, "method": "no outcome set reproduces the key question's labels", "candidates": 0}
    return {"events": fits[0], "method": "inferred: the outcome set whose planned share = the key question's label under every prior", "candidates": len(fits)}


def infer_plan_key(records: Sequence[Mapping[str, Any]], events: Sequence[str]) -> dict[str, Any]:
    """The plan's option key in the scene's comparison question: the option ``meta.plan.options`` maps to ``planned``, else infamily's
    (v0.3b), else the option (not the compared one) whose label mass is larger where the plan is the riskier action by ≥ 0.15."""
    from robo_jev.bench import infamily
    from robo_jev.bench.system1 import COMPARISON_MIN

    scene = _scene(records[0])
    votes: Counter = Counter()
    mass: Counter = Counter()
    for record in records:
        meta = record.get("meta") or {}
        name = _compare_question(record)
        if name is None:
            continue
        option = meta.get("compare_option")
        keys = [k for k in (record["request"]["questions"][name].get("criteria") or {}) if k != option]
        planned = [k for k in keys if (meta.get("plan") or {}).get("options", {}).get(k) == "planned"]
        if len(planned) == 1:
            votes[("plan.options", planned[0])] += 1
            continue
        if scene in infamily.PLAN_KEY and infamily.PLAN_KEY[scene] in keys:
            votes[("infamily.PLAN_KEY", infamily.PLAN_KEY[scene])] += 1
            continue
        variant = (meta.get("plan") or {}).get("options", {}).get(option)
        if variant is None:
            continue
        moved = _share(meta, variant, events) - _share(meta, "planned", events)
        if moved <= -COMPARISON_MIN:
            probs = (record.get("labels") or {}).get(name, {}).get("probabilities") or {}
            for k in keys:
                mass[k] += float(probs.get(k, 0.0))
    if votes:
        (method, key), n = votes.most_common(1)[0]
        return {"plan_key": key, "method": method, "agreeing_records": n, "records": sum(votes.values())}
    if mass:
        key = mass.most_common(1)[0][0]
        return {"plan_key": key, "method": "label mass where the plan is riskier", "mass": dict(mass)}
    return {"plan_key": None, "method": "no comparison question with a plan-riskier label"}


def comparison_map(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Per scene: key events and the plan's option key (:func:`infer_events`, :func:`infer_plan_key`), and how often the comparison labels
    order the two actions like their key-event shares (items with shares ≥ 0.15 apart) — the mapping's own check."""
    from robo_jev.bench.system1 import COMPARISON_MIN

    by_scene: dict[str, list] = defaultdict(list)
    for record in records:
        if _compare_question(record) is not None:
            by_scene[_scene(record)].append(record)
    out = {}
    for scene, rows in sorted(by_scene.items()):
        events = infer_events(rows)
        plan = infer_plan_key(rows, events["events"] or [])
        agree = items = 0
        if events["events"] and plan["plan_key"]:
            for record in rows:
                meta = record["meta"]
                name, option = _compare_question(record), meta.get("compare_option")
                variant = (meta.get("plan") or {}).get("options", {}).get(option)
                moved = _share(meta, variant, events["events"]) - _share(meta, "planned", events["events"])
                if abs(moved) < COMPARISON_MIN - 1e-9 or moved == 0:
                    continue
                probs = (record.get("labels") or {}).get(name, {}).get("probabilities") or {}
                order = float(probs.get(option, 0.0)) - float(probs.get(plan["plan_key"], 0.0))
                items += 1
                agree += int(order != 0 and (order > 0) == (moved > 0))
        out[scene] = {"records": len(rows), "events": events, "plan": plan, "label_agreement": agree / items if items else None, "items": items}
    return out


def comparison_items(set_name: str, records: Sequence[Mapping[str, Any]], predictions: Sequence[Mapping[str, Any]], mapping: Mapping[str, Any], *,
                     clusters: Mapping[str, str], reading: str = "single_order") -> dict[tuple, tuple]:
    """infamily's comparison reading with a per-scene mapping (:func:`comparison_map`): value = the weight of the listings whose answer orders
    the plan and the compared action like their key-event shares (≥ 0.15 apart); an equal answer misses. Cluster = `clusters[record id]`."""
    from robo_jev.bench import system1
    from robo_jev.bench.system1 import COMPARISON_MIN

    predicted = {str(p["id"]): p for p in predictions}
    out = {}
    for record in records:
        name = _compare_question(record)
        scene = mapping.get(_scene(record)) or {}
        events, plan_key = (scene.get("events") or {}).get("events"), (scene.get("plan") or {}).get("plan_key")
        if name is None or not events or not plan_key:
            continue
        meta, rid = record["meta"], str(record["id"])
        option = meta.get("compare_option")
        variant = (meta.get("plan") or {}).get("options", {}).get(option)
        if option is None or variant is None:
            continue
        moved = _share(meta, variant, events) - _share(meta, "planned", events)
        if abs(moved) < COMPARISON_MIN - 1e-9 or moved == 0:
            continue
        answer = ((predicted.get(rid) or {}).get("questions") or {}).get(name)
        if not isinstance(answer, Mapping):
            continue
        value = 0.0
        for listing, weight in system1._answers(answer, reading):
            probs = listing.get("probabilities") or {}
            order = float(probs.get(option, 0.0)) - float(probs.get(plan_key, 0.0))
            if order != 0 and (order > 0) == (moved > 0):
                value += weight
        out[(set_name, rid, name)] = (clusters.get(rid, rid), value)
    return out


# --------------------------------------------------------------------------
# Readings
# --------------------------------------------------------------------------


def g2_values(records: Sequence[Mapping[str, Any]], predictions: Sequence[Mapping[str, Any]]) -> dict[tuple, dict[str, Any]]:
    from robo_jev.decisions.single_probe import probe_values

    return probe_values(list(records), list(predictions), reading="single_order")


def mean_probe_values(first: Mapping[tuple, Mapping[str, Any]], second: Mapping[tuple, Mapping[str, Any]]) -> dict[tuple, dict[str, Any]]:
    """An arm's G2 question values: the mean of the two seeds' accuracy / Brier / code mass (groups and clusters from the first)."""
    from robo_jev.decisions.single_probe import PROBE_METRICS

    out = {}
    for key in sorted(set(first) & set(second)):
        a, b = first[key], second[key]
        entry = {k: a[k] for k in ("groups", "clusters", "chance") if k in a}
        for metric in PROBE_METRICS:
            entry[metric] = None if a.get(metric) is None or b.get(metric) is None else (float(a[metric]) + float(b[metric])) / 2.0
        out[key] = entry
    return out


def g2_arm_comparison(records: Sequence[Mapping[str, Any]], arm_a: Sequence[Sequence[Mapping[str, Any]]], arm_b: Sequence[Sequence[Mapping[str, Any]]], *,
                      n_boot: int = 2000, seed: int = 0) -> dict[str, Any]:
    """G2 accuracy of arm a − arm b (seed means per question), paired record bootstrap (``single_probe._paired_cell``), overall and by task."""
    from robo_jev.decisions.single_probe import _paired_cell

    ids = set.intersection(*({str(p["id"]) for p in preds} for preds in (*arm_a, *arm_b)))
    kept = [r for r in records if str(r.get("id")) in ids]
    a = mean_probe_values(g2_values(kept, arm_a[0]), g2_values(kept, arm_a[1]))
    b = mean_probe_values(g2_values(kept, arm_b[0]), g2_values(kept, arm_b[1]))
    common = sorted(set(a) & set(b))
    overall = _paired_cell(common, a, b, cluster="record", n_boot=n_boot, rng_seed=seed)
    by_task = {}
    for task in sorted({a[k]["groups"]["domain"] for k in common}):
        by_task[task] = _paired_cell([k for k in common if a[k]["groups"]["domain"] == task], a, b, cluster="record", n_boot=n_boot, rng_seed=seed)["accuracy"]
    return {**(overall["accuracy"] or {}), "questions": len(common), "records": len(kept), "by_task": by_task}


def _boot_dc(items_by_arm: Sequence[Mapping[tuple, tuple]], *, n_boot: int, seed: int) -> tuple[list[float], np.ndarray]:
    """DC of each source on their common items, and its bootstrap draws (paired across the sources; cluster bootstrap)."""
    v5 = V5()
    keys = sorted(set.intersection(*(set(i) for i in items_by_arm)))
    sums, counts, clusters = v5._cluster_arrays(keys, list(items_by_arm))
    points = [float(v5._metrics(s.sum(0), counts.sum(0))["DC"]) for s in sums]
    w = v5._draw_weights(clusters, n_boot, seed)
    n = w @ counts
    draws = np.stack([v5._metrics(np.einsum("bc,ctk->btk", w, s), n)["DC"] for s in sums])
    return points, draws


def author_gap(a4: Mapping[str, Mapping[tuple, tuple]], hset: Mapping[str, Mapping[tuple, tuple]], *, n_boot: int = 2000, seed: int = 0) -> dict[str, Any]:
    """The author gap DC(A4) − DC(H-set) per arm (and model), and Δgap = gap(v0.6) − gap(v0.4) with a bootstrap that draws A4's and the
    H-set's clusters independently (A4 with `seed`, the H-set with `seed` + 1), paired across the arms within each set."""
    from robo_jev.bench.system1 import _interval

    names = [n for n in a4 if n in hset]
    out: dict[str, Any] = {"per_model": {}, "seeds": {"A4": seed, "Hset": seed + 1}}
    if not names:
        return out
    pa, da = _boot_dc([a4[n] for n in names], n_boot=n_boot, seed=seed)
    ph, dh = _boot_dc([hset[n] for n in names], n_boot=n_boot, seed=seed + 1)
    for i, name in enumerate(names):
        out["per_model"][name] = {"DC_A4": pa[i], "DC_Hset": ph[i], "gap": pa[i] - ph[i], **dict(zip(("low", "high"), _interval(da[i] - dh[i])))}
    if "v06" in names and "v04" in names:
        i, j = names.index("v06"), names.index("v04")
        low, high = _interval((da[i] - dh[i]) - (da[j] - dh[j]))
        value = (pa[i] - ph[i]) - (pa[j] - ph[j])
        out["delta_gap_v06_minus_v04"] = {"value": value, "low": low, "high": high, "smaller_for_v06": value < 0}
    return out


def t8b_with_b2(t8b: Sequence[Mapping[str, Any]], labels_dir: Path = B2_LABELS) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """T8b's records with ``meta.robustness`` from B2's alternative-prior labels (band identical under the original prior — the stored
    labels —, alt_a and alt_b; :func:`robo_jev.outside.robustness.band_robustness`), in memory, and the reproduction check."""
    from robo_jev.decisions.records import file_sha256
    from robo_jev.outside.robustness import band_robustness, reproduction, with_robustness

    rows, manifest = _script("b2_verdict.py").load_labels(labels_dir, file_sha256(SET_PATHS["T8b"]))
    robustness = band_robustness(list(t8b), {p: rows[p] for p in ("alt_a", "alt_b")})
    repro = reproduction(list(t8b), rows["original"])
    return with_robustness(list(t8b), robustness), {"labels": str(labels_dir), "reproduction_share": repro["share"], "priors": manifest.get("priors")}


def _call_reading(items: Mapping[tuple, tuple], *, n_boot: int, seed: int) -> dict[str, Any]:
    return V5().call_metrics(dict(items), n_boot=n_boot, seed=seed)


def compute(sets: Mapping[str, Sequence[Mapping[str, Any]]], preds: Mapping[str, Mapping[str, Sequence[Mapping[str, Any]]]], *, n_boot: int = 2000,
            seed: int = 0) -> dict[str, Any]:
    """Every reading of the rule (and the secondary ones). `sets[name]` = records (T8b with B2's robustness); `preds[model][set]` =
    predictions for v06s0, v06s1, v04s0, v04s1, base, mix (+ v05s0 / v05s1 for context); missing secondary inputs are skipped."""
    from robo_jev.bench import infamily, system1
    from robo_jev.decisions.metrics import compare_runs

    v4, v5 = V4(), V5()
    a4, t9 = sets["A4"], sets["T9"]
    clusters = {name: v4.pair_clusters(sets[name]) for name in CALL_SETS if sets.get(name)}
    calls = {(m, s): v5.call_items(s, sets[s], preds[m][s], clusters=clusters[s]) for m in preds for s in CALL_SETS if sets.get(s) and preds[m].get(s)}
    for arm, (first, second) in ARMS.items():
        for s in CALL_SETS:
            if (first, s) in calls and (second, s) in calls:
                calls[(arm, s)] = arm_calls(calls[(first, s)], calls[(second, s)])
    out: dict[str, Any] = {"sets": {s: {"records": len(sets[s]), "clusters": len(set(clusters[s].values())),
                                        "classes": dict(Counter(t for r in sets[s] for q in r["request"]["questions"] if (t := v5.truth_class(r, q))))}
                                    for s in CALL_SETS if sets.get(s)}}
    # (S)
    s_diff = v5.call_difference(calls[("v06", "A4")], calls[("v04", "A4")], n_boot=n_boot, seed=seed)
    keys = sorted(set(calls[("v06", "A4")]) & set(calls[("v04", "A4")]))
    seed_dc = {s: v5.call_metrics(restrict(calls[(f"v06s{s}", "A4")], keys))["DC"]["value"] for s in SEEDS}
    out["S"] = s_diff
    out["consistency"] = consistency(seed_dc, s_diff["DC"]["b"] if s_diff.get("DC") else None)
    out["S_per_seed"] = {str(s): v5.call_difference(restrict(calls[(f"v06s{s}", "A4")], keys), calls[("v04", "A4")], n_boot=n_boot, seed=seed) for s in SEEDS}
    # (F)
    robust_a4 = infamily.robust_keys(a4)
    miss = {m: v4.keep(v4.items_of("A4", a4, preds[m]["A4"], by_pair=True).get("severe_miss") or {}, robust_a4) for m in ("v06s0", "v06s1", "base")}
    out["F"] = v4._paired(arm_values(miss["v06s0"], miss["v06s1"]), miss["base"], n_boot=n_boot, seed=seed)
    # (N)
    mapping = comparison_map(a4)
    out["A4_comparison_map"] = mapping
    a4_items = {m: v4.items_of("A4", a4, preds[m]["A4"], by_pair=True) for m in ("v06s0", "v06s1", "v04s0", "v04s1", "base")}
    for m in a4_items:
        a4_items[m]["comparison_compare_question"] = comparison_items("A4", a4, preds[m]["A4"], mapping, clusters=clusters["A4"])
    t9_items = {m: v4.items_of("T9", t9, preds[m]["T9"], by_pair=True) for m in ("v06s0", "v06s1", "v04s0", "v04s1") if preds[m].get("T9")}

    def arm_diff(items: Mapping[str, Mapping[str, Any]], metric: str, *, stratified: bool = False) -> dict[str, Any] | None:
        a = arm_values(items["v06s0"].get(metric) or {}, items["v06s1"].get(metric) or {})
        b = arm_values(items["v04s0"].get(metric) or {}, items["v04s1"].get(metric) or {})
        return v4._paired(a, b, n_boot=n_boot, seed=seed, stratified=stratified)

    home = {}
    for s in SEEDS:
        mine = preds[f"v06s{s}"]
        home_a = system1.group_items({h: system1.set_items(h, sets[h], mine[h]) for h in HOME}, HOME)
        home_b = system1.group_items({h: system1.set_items(h, sets[h], preds["mix"][h]) for h in HOME}, HOME)
        home[str(s)] = {"band_balanced": v4._paired(home_a.get("band_balanced") or {}, home_b.get("band_balanced") or {}, n_boot=n_boot, seed=seed, stratified=True),
                        "severe_error": v4._paired(home_a.get("severe_error") or {}, home_b.get("severe_error") or {}, n_boot=n_boot, seed=seed)}
    out["N"] = {"arm": {"A4_direction": arm_diff(a4_items, "direction"), "A4_comparison": arm_diff(a4_items, "comparison_compare_question"),
                        "T9_direction": arm_diff(t9_items, "direction") if len(t9_items) == 4 else None,
                        "T9_comparison": arm_diff(t9_items, "comparison") if len(t9_items) == 4 else None},
                "home": home}
    # (R)
    r = {}
    for s in SEEDS:
        mine, per_set = preds[f"v06s{s}"], {}
        for name in R_SETS:
            if sets.get(name) and mine.get(name) and preds["mix"].get(name):
                comparison = compare_runs(list(sets[name]), list(mine[name]), list(preds["mix"][name]), n_boot=n_boot, seed=seed)
                diffs = comparison["readings"]["single_order"]["differences"]
                per_set[name] = {key: diffs.get(key) for key in v4.R_LIMITS}
        r[str(s)] = per_set
    out["R"] = r
    # (G)
    out["G"] = g2_arm_comparison(sets["G2"], [preds["v06s0"]["G2"], preds["v06s1"]["G2"]], [preds["v04s0"]["G2"], preds["v04s1"]["G2"]], n_boot=n_boot, seed=seed)
    # secondary
    sec: dict[str, Any] = {}
    sec["calls_v06_minus_v04"] = {s: v5.call_difference(calls[("v06", s)], calls[("v04", s)], n_boot=n_boot, seed=seed)
                                  for s in CALL_SETS if ("v06", s) in calls and ("v04", s) in calls}
    sec["calls_vs_base"] = {s: {arm: v5.call_difference(calls[(arm, s)], calls[("base", s)], n_boot=n_boot, seed=seed) for arm in ARMS if (arm, s) in calls}
                            for s in ("A4", "T9", "T8b") if ("base", s) in calls}
    sec["author_gap"] = author_gap({arm: calls[(arm, "A4")] for arm in (*ARMS, *CONTEXT_MODELS) if (arm, "A4") in calls},
                                   {arm: calls[(arm, "Hset")] for arm in (*ARMS, *CONTEXT_MODELS) if (arm, "Hset") in calls}, n_boot=n_boot, seed=seed)
    sec["A4_balanced_band"] = {"v06_minus_v04": arm_diff(a4_items, "band_balanced", stratified=True),
                               "v06_minus_base": v4._paired(arm_values(a4_items["v06s0"].get("band_balanced") or {}, a4_items["v06s1"].get("band_balanced") or {}),
                                                            a4_items["base"].get("band_balanced") or {}, n_boot=n_boot, seed=seed, stratified=True),
                               "v04_minus_base": v4._paired(arm_values(a4_items["v04s0"].get("band_balanced") or {}, a4_items["v04s1"].get("band_balanced") or {}),
                                                            a4_items["base"].get("band_balanced") or {}, n_boot=n_boot, seed=seed, stratified=True)}
    sec["A4_direction_comparison_vs_base"] = {metric: v4._paired(arm_values(a4_items["v06s0"].get(metric) or {}, a4_items["v06s1"].get(metric) or {}),
                                                                 a4_items["base"].get(metric) or {}, n_boot=n_boot, seed=seed)
                                              for metric in ("direction", "comparison_compare_question")}
    ambiguous = infamily.ambiguous_keys(a4)
    sec["A4_hedge_share_prior_dependent"] = {m: infamily.answer_profile(a4, preds[m]["A4"], ambiguous).get("hedge_share")
                                             for m in ("v06s0", "v06s1", "v04s0", "v04s1", "base")}
    sec["G2_by_task_v06_minus_v04"] = out["G"].pop("by_task")
    out["secondary"] = sec
    # context: every model's (and arm's) call view per set
    out["context"] = {s: {m: _call_reading(calls[(m, s)], n_boot=n_boot, seed=seed) for m in (*CONTEXT_MODELS, *ARMS) if (m, s) in calls} for s in CALL_SETS}
    spread = {}
    for s in CALL_SETS:
        for arm, (first, second) in ARMS.items():
            if (first, s) in calls and (second, s) in calls:
                a, b = out["context"][s].get(first) or {}, out["context"][s].get(second) or {}
                spread[f"{s}/{arm}"] = {m: abs(a[m]["value"] - b[m]["value"]) if a.get(m, {}).get("value") is not None and b.get(m, {}).get("value") is not None else None
                                        for m in ("DC", "LU", "LC", "GAP", "CA")}
    out["seed_spread"] = spread
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


def is_sealed(set_name: str, path: Path) -> bool:
    return Path(path).resolve() == SET_PATHS[set_name].resolve()


def prediction_paths(args: argparse.Namespace) -> dict[tuple[str, str], Path]:
    """(model, set) → the prediction file the rule reads. Pods: the v0.6 jobs (--s0-dir / --s1-dir), the v0.4 jobs, v0.1-mix, the base's G2.
    Spark: --spark-dir first (the v0.6 chain: A4 for v0.6, v0.4 and the base; T9, T8b, H-set, B1 for v0.6), then — for every model but
    v0.6 — v0.4's and v0.5's Spark directories, then the earlier tasks' files. A4, and a T9 that is not the sealed file, only from --spark-dir."""
    paths: dict[tuple[str, str], Path] = {}
    for s, mine, theirs in ((0, args.s0_dir, args.v04_s0_dir), (1, args.s1_dir, args.v04_s1_dir)):
        for name in POD_SETS:
            paths[(f"v06s{s}", name)] = mine / pod_prediction_name("v06", s, name)
            paths[(f"v04s{s}", name)] = theirs / pod_prediction_name("v04", s, name)
    for name in POD_SETS:
        paths[("mix", name)] = args.mix_dir / f"after-v01mix-2b-{name}.predictions.jsonl"
    sealed_t9 = is_sealed("T9", args.t9)
    for name in SPARK_SETS:
        for model in SPARK_TAGS:
            fresh = args.spark_dir / spark_prediction_name(name, model)
            if name == "A4" or (name == "T9" and not sealed_t9):
                paths[(model, name)] = fresh
                continue
            candidates = [fresh] + ([] if model.startswith("v06") else [args.v04_spark_dir / spark_prediction_name(name, model),
                                                                        args.v05_spark_dir / spark_prediction_name(name, model)])
            if (model, name) in EXISTING_SPARK:
                candidates.append(EXISTING_SPARK[(model, name)])
            paths[(model, name)] = next((c for c in candidates if c.is_file()), candidates[0])
    paths[("base", "G2")] = args.base_g2
    paths[("base", "T8a")] = EXISTING_SPARK[("base", "T8a")]
    return paths


def required_inputs() -> list[tuple[str, str]]:
    """What the rule itself reads (missing → refuse); everything else is secondary (missing → skipped and noted)."""
    need = [(m, "A4") for m in ("v06s0", "v06s1", "v04s0", "v04s1", "base")] + [(m, "T9") for m in ("v06s0", "v06s1", "v04s0", "v04s1")]
    need += [(m, s) for m in ("v06s0", "v06s1", "mix") for s in sorted(set(R_SETS) | set(HOME))]
    need += [(m, "G2") for m in ("v06s0", "v06s1", "v04s0", "v04s1")]
    return need


def a4_problems(records: Sequence[Mapping[str, Any]]) -> list[str]:
    """The sealed A4 must be the held-out author's test split (ids v06a4-…, meta.set v06-a4, split test)."""
    bad = [str(r.get("id")) for r in records if not str(r.get("id")).startswith(A4_PREFIX) or (r.get("meta") or {}).get("set") != A4_SET
           or (r.get("meta") or {}).get("split") != "test"]
    return [f"A4: {len(bad)} records are not the held-out author's test records (first {bad[:3]})"] if bad else []


def _fmt(value: Any, digits: int = 3, signed: bool = False) -> str:
    if value is None:
        return "—"
    return f"{value:+.{digits}f}" if signed else f"{value:.{digits}f}"


def _ci(d: Mapping[str, Any] | None, *, key: str = "value") -> str:
    if not d or d.get(key) is None:
        return "—"
    return f"{_fmt(d[key], signed=True)} [{_fmt(d.get('low'), signed=True)}, {_fmt(d.get('high'), signed=True)}] ({_fmt(d.get('a'))} vs {_fmt(d.get('b'))})"


def _cell(d: Mapping[str, Any] | None) -> str:
    if not d or d.get("value") is None:
        return "—"
    return _fmt(d["value"]) + (f" [{_fmt(d['low'])}, {_fmt(d['high'])}]" if d.get("low") is not None else "")


def render(result: Mapping[str, Any]) -> str:
    v, r = result["verdict"], result["readings"]
    s = v["S"]
    lines = [f"# v0.6 pre-declared rule — S {s['outcome']} · {v['release_line']}", ""]
    if result.get("pipeline_check"):
        lines += [f"**PIPELINE CHECK ONLY — {result['pipeline_check']}; this is not the v0.6 verdict.**", ""]
    lines += [f"Rule written {RULE['written']}. Applied by `scripts/verdict_v06.py` ({SCRIPT_VERSION}, git {result.get('git')}); {result['n_boot']} paired "
              f"cluster-bootstrap draws (seed {result['seed']}); single order; **seed means** (arm v0.6 = s0 + s1 vs arm v0.4 = s0 + s1). Hardware: A4, T9, "
              "T8b, B1-infamily, H-set on Spark; T1–T7, G2, T8a on the H100 pods.", "",
              "## (S) A4 call view, v0.6 − v0.4 (arm means)", "",
              "Classes on A4: " + ", ".join(f"{t} {n}" for t, n in r["sets"]["A4"]["classes"].items()) + f"; clusters {r['sets']['A4']['clusters']}.", "",
              "| reading | Δ [95 %] (v0.6 vs v0.4) |", "|---|---|"]
    for metric in ("DC", "GAP", "CA", "LU", "LC", "safe|clear_risky", "safe|unclear"):
        lines.append(f"| {metric} | {_ci(r['S'].get(metric))} |")
    c = s["checks"]
    cons = r["consistency"]
    lines += ["", "| ΔDC ≤ −0.05 | upper < 0 | ΔGAP ≥ +0.05 | lower > 0 | ΔCA ≥ −0.10 | consistency | ΔDC lower > −0.02 |", "|---|---|---|---|---|---|---|",
              f"| {c['dDC_le_-0.05']} | {c['dDC_high_lt_0']} | {c['dGAP_ge_+0.05']} | {c['dGAP_low_gt_0']} | {c['dCA_ge_-0.10']} | {c['consistency']} | {c['dDC_low_gt_-0.02']} |",
              "", "Consistency: v0.6 seed DC on A4 " + ", ".join(f"s{k} {_fmt(x)}" for k, x in cons["v06_seed_dc"].items())
              + f" vs the v0.4 seed-mean DC {_fmt(cons['v04_seed_mean_dc'])} → {cons['per_seed']}.", "", f"**S: {s['outcome']}**", "",
              "Per seed (each v0.6 seed vs the v0.4 arm — for a seed disagreement):", "", "| seed | ΔDC | ΔGAP | ΔCA |", "|---|---|---|---|"]
    for seed, d in r["S_per_seed"].items():
        lines.append(f"| {seed} | {_ci(d.get('DC'))} | {_ci(d.get('GAP'))} | {_ci(d.get('CA'))} |")
    p = v["parts"]
    lines += ["", "## Guards", "", "| guard | reading | pass |", "|---|---|---|",
              f"| F A4 robust high-called-low (v0.6 arm ≤ base) | v0.6 {_fmt(p['F']['v06_arm'])} vs base {_fmt(p['F']['base'])} (n {p['F']['n']}) | {p['F']['pass']} |"]
    n = r["N"]
    for key in ("A4_direction", "A4_comparison", "T9_direction", "T9_comparison"):
        lines.append(f"| N {key} (v0.6 − v0.4 arms, lower > −0.05) | {_ci(n['arm'].get(key))} | {p['N']['checks'][f'{key}_low_gt_-0.05']} |")
    for seed, entry in n["home"].items():
        lines.append(f"| N T2–T4 balanced band s{seed} (vs v0.1-mix) | {_ci(entry['band_balanced'])} | {p['N']['checks'][f'T2-T4_band_low_gt_-0.05[s{seed}]']} |")
        lines.append(f"| N T2–T4 severe s{seed} (≤ v0.1-mix + 0.01) | {_ci(entry['severe_error'])} | {p['N']['checks'][f'T2-T4_severe_le_mix_plus_0.01[s{seed}]']} |")
    for seed, entry in p["R"]["per_seed"].items():
        for name, sets_ in entry["sets"].items():
            for key, d in sets_.items():
                lines.append(f"| R s{seed} {name} {key} (vs v0.1-mix, upper) | {_fmt(d.get('value'), 4, True)} [{_fmt(d.get('low'), 4, True)}, {_fmt(d.get('high'), 4, True)}] | {d['ok']} |")
    g = r["G"]
    lines.append(f"| G G2 accuracy (v0.6 − v0.4 arms) | {_fmt(g.get('diff'), signed=True)} [{_fmt(g.get('low'), signed=True)}, {_fmt(g.get('high'), signed=True)}] "
                 f"({_fmt(g.get('a'))} vs {_fmt(g.get('b'))}) | {p['G']['pass']} |")
    ns = v["next_step"]
    lines += ["", "## Release and next step", "", f"- **{v['release_line']}**", f"- Next step ({ns['branch']}): {ns['next']}.", f"- Known limit: {RULE['limit']}.", "",
              "## Secondary (reported, not deciding)", "", "Call view v0.6 − v0.4 (arm means) per set:", "", "| set | ΔDC | ΔLU | ΔLC | ΔGAP | ΔCA |", "|---|---|---|---|---|---|"]
    sec = r["secondary"]
    for name, d in sec["calls_v06_minus_v04"].items():
        lines.append(f"| {name} | " + " | ".join(_ci(d.get(m)) for m in ("DC", "LU", "LC", "GAP", "CA")) + " |")
    lines += ["", "DC vs the untrained base:", "", "| set | arm | ΔDC vs base |", "|---|---|---|"]
    for name, arms in sec["calls_vs_base"].items():
        for arm, d in arms.items():
            lines.append(f"| {name} | {arm} | {_ci(d.get('DC'))} |")
    gap = sec["author_gap"]
    lines += ["", "Author gap DC(A4) − DC(H-set) (predicted smaller for v0.6):", "", "| model / arm | DC A4 | DC H-set | gap [95 %] |", "|---|---|---|---|"]
    for name, e in gap.get("per_model", {}).items():
        lines.append(f"| {MODELS.get(name, name)} | {_fmt(e['DC_A4'])} | {_fmt(e['DC_Hset'])} | {_fmt(e['gap'], signed=True)} [{_fmt(e.get('low'), signed=True)}, {_fmt(e.get('high'), signed=True)}] |")
    if gap.get("delta_gap_v06_minus_v04"):
        d = gap["delta_gap_v06_minus_v04"]
        lines.append(f"\nΔgap v0.6 − v0.4: {_fmt(d['value'], signed=True)} [{_fmt(d['low'], signed=True)}, {_fmt(d['high'], signed=True)}] (smaller for v0.6: {d['smaller_for_v06']}).")
    lines += ["", "| reading | Δ [95 %] |", "|---|---|"]
    for key, d in sec["A4_balanced_band"].items():
        lines.append(f"| A4 balanced band {key} | {_ci(d)} |")
    for key, d in sec["A4_direction_comparison_vs_base"].items():
        lines.append(f"| A4 {key} v0.6 arm − base | {_ci(d)} |")
    for task, d in (sec.get("G2_by_task_v06_minus_v04") or {}).items():
        d = d or {}
        lines.append(f"| G2 {task} v0.6 − v0.4 (arms) | {_fmt(d.get('diff'), signed=True)} [{_fmt(d.get('low'), signed=True)}, {_fmt(d.get('high'), signed=True)}] "
                     f"({_fmt(d.get('a'))} vs {_fmt(d.get('b'))}) |")
    hedge = sec.get("A4_hedge_share_prior_dependent") or {}
    lines += ["", "Hedge share on prior-dependent A4 questions: " + ", ".join(f"{m} {_fmt(x)}" for m, x in hedge.items()) + ".", "",
              "Seed spread (|s0 − s1| per arm): " + "; ".join(f"{k} " + ", ".join(f"{m} {_fmt(x)}" for m, x in e.items()) for k, e in r["seed_spread"].items()) + ".", "",
              "A4 comparison mapping (per scene): " + "; ".join(f"{scene}: events {m['events']['events']} ({m['events']['method']}), plan key "
                                                                f"{m['plan']['plan_key']} ({m['plan']['method']}), label agreement {_fmt(m['label_agreement'])} on {m['items']} items"
                                                                for scene, m in r["A4_comparison_map"].items()) + ".", "",
              "## Context — every model's call view per set (single order, 95 % cluster bootstrap)", ""]
    for name, models in r["context"].items():
        if not models:
            continue
        lines += [f"**{name}** — classes " + ", ".join(f"{t} {k}" for t, k in r["sets"][name]["classes"].items()), "",
                  "| model | DC | LU | LC | GAP | CA | P(safe|clear-safe) | P(risky|clear-risky) | P(safe|unclear) | P(safe|clear-risky) |", "|---|---|---|---|---|---|---|---|---|---|"]
        for model, m in models.items():
            lines.append(f"| {MODELS.get(model, model + ' (arm, seed mean)')} | " + " | ".join(_cell(m.get(k)) for k in V5().CALL_METRICS) + " |")
        lines.append("")
    lines += ["## Notes", ""] + [f"- {note}" for note in result["notes"]] + ["", "## Sources", "", "| model / set | predictions | same set file | coverage |", "|---|---|---|---|"]
    for key, src in result["sources"].items():
        lines.append(f"| {key} | `{src['predictions']}` | {not src['problems']} | {_fmt(src['coverage'])} |")
    lines += ["", "## Rule (verbatim parts)", ""] + [f"- **{k}** — {t}" for k, t in RULE.items()]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--s0-dir", type=Path, required=True, help="fetched reports of the v0.6 seed-0 job (after-v06-s0-2b-*)")
    parser.add_argument("--s1-dir", type=Path, required=True, help="fetched reports of the v0.6 seed-1 job (after-v06-s1-2b-*)")
    parser.add_argument("--v04-s0-dir", type=Path, default=V04_PODS[0])
    parser.add_argument("--v04-s1-dir", type=Path, default=V04_PODS[1])
    parser.add_argument("--mix-dir", type=Path, default=MIX_POD)
    parser.add_argument("--spark-dir", type=Path, default=SPARK_DIR, help="the v0.6 chain's Spark predictions (<SET>-<tag>.predictions.jsonl); read first")
    parser.add_argument("--v04-spark-dir", type=Path, default=V04_SPARK_DIR)
    parser.add_argument("--v05-spark-dir", type=Path, default=V05_SPARK_DIR)
    parser.add_argument("--base-g2", type=Path, default=BASE_G2_POD)
    parser.add_argument("--a4", type=Path, default=SET_PATHS["A4"], help="the sealed A4 (default); another file only for a pipeline check")
    parser.add_argument("--t9", type=Path, default=SET_PATHS["T9"], help="the sealed T9 (default); another file only for a pipeline check")
    parser.add_argument("--b2-labels", type=Path, default=B2_LABELS)
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--allow-mismatch", action="store_true", help="read even if a prediction file was scored on another set file or conditions")
    parser.add_argument("--out", type=Path, default=ARTIFACTS / "reports" / "v06" / "verdict.json")
    args = parser.parse_args(argv)
    refuse_write(args.out)
    set_paths = {**SET_PATHS, "A4": args.a4, "T9": args.t9}
    pipeline = [f"{name} is {path}, not the sealed file" for name, path in (("A4", args.a4), ("T9", args.t9)) if not is_sealed(name, path)]
    if pipeline:
        _log("pipeline check: " + "; ".join(pipeline) + f" — those predictions only from {args.spark_dir}")
    paths = prediction_paths(args)
    missing = [f"{m} {s}: {paths[(m, s)]}" for m, s in required_inputs() if not paths[(m, s)].is_file()]
    if missing:
        raise SystemExit("missing prediction files the rule reads:\n  " + "\n  ".join(missing))
    needed = sorted({s for (_, s), p in paths.items() if p.is_file()} | {"A4", "T9", "G2"})
    sets = {s: V4()._read(set_paths[s]) for s in needed if set_paths[s].is_file()}
    notes: list[str] = []
    if "T8b" in sets:
        sets["T8b"], b2 = t8b_with_b2(sets["T8b"], args.b2_labels)
        notes.append(f"T8b truth classes from B2's alternative-prior labels ({b2['labels']}; original-prior reproduction share {b2['reproduction_share']:.4f}).")
    preds: dict[str, dict[str, list[dict[str, Any]]]] = {m: {} for m in MODELS}
    sources = {}
    for (model, name), path in sorted(paths.items()):
        if not path.is_file() or name not in sets:
            if not path.is_file():
                notes.append(f"secondary input absent, skipped: {model} {name} ({path})")
            continue
        preds[model][name] = V4()._read(path)
        sources[f"{model} {name}"] = V4().check_source(path, set_paths[name], sets[name], preds[model][name])
    problems = [f"{k}: {'; '.join(v['problems'])}" for k, v in sources.items() if v["problems"]]
    if is_sealed("A4", args.a4):
        problems += a4_problems(sets["A4"])
    if is_sealed("T9", args.t9) and len(sets["T9"]) != 600:
        problems.append(f"T9 has {len(sets['T9'])} records — not the sealed T9")
    if problems and not args.allow_mismatch:
        raise SystemExit("refusing (use --allow-mismatch to read anyway):\n  " + "\n  ".join(problems))
    readings = compute(sets, preds, n_boot=args.n_boot, seed=args.seed)
    verdict = decide(readings)
    if pipeline:
        notes.insert(0, "PIPELINE CHECK ONLY: " + "; ".join(pipeline) + " — this is not the v0.6 verdict")
    weak = [scene for scene, m in readings["A4_comparison_map"].items() if m["label_agreement"] is None or m["label_agreement"] < 0.9]
    if weak:
        notes.append(f"A4 comparison mapping: label agreement below 0.9 (or unread) in scenes {weak} — check the mapping before reading N's A4 comparison")
    notes += ["A4, T9, T8b, B1-infamily and the H-set were scored on Spark (GB10); T1–T7, G2 and T8a on H100 pods (v0.6, v0.4, v0.1-mix; the base's G2 "
              "on the v0.1 main pod).",
              "Seed means: an arm's item = the mean of its two seeds' item values (call distributions: each seed's listings weighted 1/L), on the items "
              "both seeds answered; arm comparisons pair the items both arms have. Consistency compares each v0.6 seed's DC with the v0.4 arm's DC "
              "on the same items.",
              "(R) reads compare_runs (single order, record bootstrap) per seed vs v0.1-mix; (G) is single_probe._paired_cell on the arms' mean "
              "question values (record clusters); (F) is read on the robust A4 items the v0.6 arm and the base both answered."]
    result = {"kind": "v06-verdict", "script_version": SCRIPT_VERSION, "git": _git_sha(), "created": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
              "rule_text": RULE, "n_boot": args.n_boot, "seed": args.seed, "pipeline_check": "; ".join(pipeline) or None, "verdict": verdict,
              "readings": readings, "sources": sources, "problems": problems, "notes": notes,
              "inputs": {k: str(v) for k, v in vars(args).items() if isinstance(v, Path)}}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=1, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    args.out.with_suffix(".md").write_text(render(result), encoding="utf-8")
    print(render(result).split("## Secondary")[0])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
