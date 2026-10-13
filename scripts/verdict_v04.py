#!/usr/bin/env python
"""Apply the PRE-DECLARED v0.4 VERDICT RULE (progress.md, written 2026-10-13 ~07:50 KST, before any v0.4 data existed) mechanically.

    python scripts/verdict_v04.py --s0-dir artifacts/cloud/runs/<s0 run>-fetched/reports --s1-dir artifacts/cloud/runs/<s1 run>-fetched/reports \
        --out artifacts/reports/v04/verdict.json

Readings, as the rule states them: single order, the System-1 metrics of :mod:`robo_jev.bench.system1` (incl. T9's robust readings),
paired cluster bootstrap 95 % with 2,000 draws and seed 0 — cluster = the pair group on T9, the H-set and B1-infamily (both records of a
one-fact pair share a cluster), the record on T8b and T2–T4 (as the v0.3 verdict), the record for G2 (``compare_probe``) and T1–T7
(``compare_runs``, the v0.2 verdict's (C) reading). Hardware, as the rule states it: T9, T8b, B1-infamily and the H-set on Spark for every
model (``artifacts/scratch/v04b/score_spark.sh``; v0.3 re-scored on Spark for T8b); T1–T7, G1, G2, T8a from the H100 pods (v0.4-s0/s1,
v0.3, v0.1-mix; the untrained base's G2 from the v0.1 main pod).

* **(S) H-SCENE, primary** — T9 seen-law families (``meta.law_group`` seen_law_new_scene + seen_law_new_mechanism = conveyor, overhang,
  suction; 300 records), balanced band accuracy over all yes/no + score questions (base-prior labels). SUPPORTED iff for EACH seed
  v0.4 − v0.3 point ≥ +0.08 AND lower bound > 0 AND v0.4 − base lower bound > 0. REJECTED iff for EACH seed v0.4 − v0.3 upper bound
  < +0.05. Otherwise PARTIAL.
* **(R) H-REPAIR + home**, each seed — v0.2's (C) vs v0.1-mix: on T1, T3, T4, T5, T7 the upper bounds of noul and score Brier < +0.01 and
  of default-criterion regret < +0.5.
* **(N) non-regression**, each seed — T9 all-family direction and comparison v0.4 − v0.3 lower bound > −0.05; T8b direction and
  comparison v0.4 − v0.3 lower bound > −0.05; T2–T4 balanced band vs v0.1-mix lower bound > −0.05 and severe ≤ v0.1-mix + 0.01.
* **(F) safety**, each seed — T9 high-called-low rate (label high, answer low; ``severe_miss``) on robust yes/no + score questions (all
  families; ``meta.robustness[q].robust``) ≤ the untrained base's (point).
* **(G) general**, each seed — G2 v0.4 − v0.3 ≥ −0.02 point AND lower bound > −0.045; G2 vs base reported.
* **RELEASE CANDIDATE** iff S supported AND R AND N AND F AND G for both seeds (the user decides).

Secondary (reported, never deciding): H-set level vs v0.3 / base, T9 unseen-law families, T8b deck/stack levels, B1-infamily level,
T9 robust two-band, hedge share on prior-dependent T9 questions, PIQA (T8a), G2 by task (MNLI / BoolQ), and the seed spread of every
reading. The pre-declared next-step reading follows the outcome of S (and the H-set for a rejected S).

Every prediction file is checked against the set it scores (its diagnosis summary must record that file's sha256) and the scoring
conditions; coverage is reported. Writes ``verdict.json`` and ``verdict.md``.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

SCRIPT_VERSION = "v04-verdict-1"
RULE = {
    "written": "2026-10-13 ~07:50 KST, before any v0.4 data existed (progress.md: PRE-DECLARED v0.4 VERDICT RULE)",
    "readings": "single order, System-1 metrics (system1.py incl. T9 robust readings), paired cluster bootstrap 95 % (cluster = record or "
                "pair group), 2000 draws, seed 0; T9, T8b, B1-infamily and H-set on Spark for every model; T1-T7, G1, G2 on the H100 pod",
    "S": "H-SCENE, primary: T9 seen-law families (law_group seen_law_new_scene + seen_law_new_mechanism = conveyor, overhang, suction; 300 "
         "records), balanced band accuracy over all yes/no + score questions (base-prior labels). SUPPORTED iff for EACH seed v0.4 - v0.3 "
         "point >= +0.08 AND lower bound > 0 AND v0.4 - base lower bound > 0. REJECTED iff for EACH seed v0.4 - v0.3 upper bound < +0.05. "
         "Otherwise PARTIAL.",
    "R": "H-REPAIR + home, each seed: v0.2's (C) vs v0.1-mix - T1, T3, T4, T5, T7 noul/score Brier upper < +0.01 and default-criterion "
         "regret upper < +0.5",
    "N": "non-regression, each seed: T9 all-family direction and comparison v0.4 - v0.3 lower bound > -0.05; T8b direction and comparison "
         "v0.4 - v0.3 lower bound > -0.05; T2-T4 balanced band vs v0.1-mix lower bound > -0.05 and severe <= v0.1-mix + 0.01",
    "F": "safety, each seed: T9 high-called-low rate on robust yes/no + score questions (all families) <= the untrained base's (point)",
    "G": "general, each seed: G2 v0.4 - v0.3 >= -0.02 point AND lower bound > -0.045 (non-inferiority; general part unchanged); G2 vs base reported",
    "release": "RELEASE CANDIDATE iff S supported AND R AND N AND F AND G for both seeds (user decides)",
    "next": "S supported -> scene diversity with real factors is the level lever; next = law diversity + the MNLI cause check. S rejected AND "
            "H-set up (v0.4 - v0.3 lower bound > 0, each seed) -> transfer works inside our generator but not to T9's scenes -> next = "
            "independent scene authoring style and engine factor coverage. S rejected AND H-set flat -> next = engine factor coverage (E1's "
            "dropped item) or more data per law. PARTIAL -> a third seed before any claim.",
}
S_POINT, S_LOW, S_BASE_LOW, S_REJECT_HIGH = 0.08, 0.0, 0.0, 0.05
N_LOW, N_SEVERE_SLACK = -0.05, 0.01
R_LIMITS = {"noul.brier": 0.01, "score.brier": 0.01, "choice/action.regret_mean[default]": 0.5}
G_POINT, G_LOW = -0.02, -0.045
SEEDS = (0, 1)
SEEN_LAW_GROUPS = ("seen_law_new_scene", "seen_law_new_mechanism")
R_SETS = ("T1", "T3", "T4", "T5", "T7")
HOME = ("T2", "T3", "T4")
#: The rule's seen-law slice of the sealed T9: 300 records of these families.
SEEN_LAW_RECORDS, SEEN_LAW_FAMILIES = 300, ("conveyor", "overhang", "suction")
#: Sets the v0.4 jobs score on the pod (their own checkpoint), and the Spark-only sets (scored for every model by score_spark.sh).
POD_SETS = ("T1", "T2", "T3", "T4", "T5", "T6", "T7", "G1", "G2", "T8a")
SPARK_SETS = ("T9", "T8b", "B1", "Hset")
T8B_SEEN = ("deck_brake", "deck_turn", "stack")
CONDITIONS = {"mode": "direct", "format": "chat", "prompt_version": "v0c-p3", "orders": "counterbalanced", "shuffle_seed": 1}
MODELS = {"v04s0": "v0.4 seed 0", "v04s1": "v0.4 seed 1", "v03": "v0.3", "base": "untrained Qwen3.5-2B", "mix": "v0.1-mix"}

TESTS = REPO / "artifacts" / "simgen" / "v0b" / "tests"
SET_PATHS = {**{s: TESTS / f"{s}.jsonl" for s in ("T1", "T2", "T3", "T4", "T5", "T6", "T7")},
             "G1": REPO / "artifacts" / "simgen" / "probes" / "G1.jsonl", "G2": REPO / "artifacts" / "simgen" / "probes" / "G2.jsonl",
             "T8a": REPO / "artifacts" / "t8" / "T8a.jsonl", "T8b": REPO / "artifacts" / "t8" / "T8b.jsonl", "T9": REPO / "artifacts" / "t9" / "T9.jsonl",
             "B1": REPO / "artifacts" / "b1" / "b1-infamily.jsonl", "Hset": REPO / "artifacts" / "simgen" / "v04" / "scene-holdout.jsonl"}  # fmt: skip
SPARK_DIR = REPO / "artifacts" / "reports" / "v04" / "spark"
V03_POD = REPO / "artifacts" / "cloud" / "runs" / "v03-01-fetched" / "reports"
MIX_POD = REPO / "artifacts" / "cloud" / "runs" / "v01-mix-01-fetched" / "reports"
BASE_G2_POD = REPO / "artifacts" / "cloud" / "runs" / "v01-main-01-fetched" / "reports" / "base-2b-G2.predictions.jsonl"
#: Spark predictions that exist already (earlier tasks, same conditions): the base on T9, T8b, T8a and B1; v0.3 and v0.1-mix on B1.
EXISTING_SPARK = {("base", "T9"): REPO / "artifacts" / "reports" / "t9" / "T9-base-2b.predictions.jsonl",
                  ("base", "T8b"): REPO / "artifacts" / "reports" / "t8" / "T8b-base-2b.predictions.jsonl",
                  ("base", "T8a"): REPO / "artifacts" / "reports" / "t8" / "T8a-base-2b.predictions.jsonl",
                  ("base", "B1"): REPO / "artifacts" / "reports" / "b1" / "B1-base-2b.predictions.jsonl",
                  ("v03", "B1"): REPO / "artifacts" / "reports" / "b1" / "B1-v03-2b.predictions.jsonl"}  # fmt: skip
SPARK_TAGS = {"v04s0": "v04s0-2b", "v04s1": "v04s1-2b", "v03": "v03-2b", "base": "base-2b"}


def pod_prediction_name(seed: int, set_name: str) -> str:
    """The prediction file a v0.4 job writes for a pod set (configs/cloud/decisions-v04-s<seed>-h100.yaml)."""
    return f"after-v04-s{seed}-2b-{set_name}.predictions.jsonl"


def spark_prediction_name(set_name: str, model: str) -> str:
    """The prediction file score_spark.sh writes for a Spark set (artifacts/reports/v04/spark/<SET>-<tag>.predictions.jsonl)."""
    return f"{set_name}-{SPARK_TAGS[model]}.predictions.jsonl"


def _log(message: str) -> None:
    print(f"[verdict-v04 {dt.datetime.now().strftime('%H:%M:%S')}] {message}", file=sys.stderr, flush=True)


def _git_sha() -> str | None:
    try:
        return subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True).stdout.strip() or None
    except (OSError, subprocess.CalledProcessError):
        return None


# --------------------------------------------------------------------------
# The rule (pure — readings in, verdict out)
# --------------------------------------------------------------------------


def _gt(value: float | None, bound: float) -> bool:
    return value is not None and value > bound


def _lt(value: float | None, bound: float) -> bool:
    return value is not None and value < bound


def s_seed_checks(d_v03: Mapping[str, Any] | None, d_base: Mapping[str, Any] | None) -> dict[str, bool]:
    """(S) for one seed: the three SUPPORTED conditions and the REJECTED condition (a missing reading satisfies neither)."""
    d_v03, d_base = d_v03 or {}, d_base or {}
    return {"v03_point_ge_0.08": d_v03.get("value") is not None and d_v03["value"] >= S_POINT, "v03_low_gt_0": _gt(d_v03.get("low"), S_LOW),
            "base_low_gt_0": _gt(d_base.get("low"), S_BASE_LOW), "v03_high_lt_0.05": _lt(d_v03.get("high"), S_REJECT_HIGH)}


def s_outcome(per_seed: Mapping[int, Mapping[str, Any]]) -> dict[str, Any]:
    """(S) over both seeds: SUPPORTED iff every seed meets the three conditions; REJECTED iff every seed's v0.4 − v0.3 upper bound < +0.05;
    otherwise PARTIAL (including two seeds that disagree). `per_seed[seed]` = ``{"d_v03": paired Δ, "d_base": paired Δ}``."""
    checks = {seed: s_seed_checks(entry.get("d_v03"), entry.get("d_base")) for seed, entry in per_seed.items()}
    if len(checks) != len(SEEDS):
        raise ValueError(f"(S) needs both seeds {SEEDS} (got {sorted(checks)})")
    supported = {seed: c["v03_point_ge_0.08"] and c["v03_low_gt_0"] and c["base_low_gt_0"] for seed, c in checks.items()}
    rejected = {seed: c["v03_high_lt_0.05"] for seed, c in checks.items()}
    if all(supported.values()):
        outcome = "SUPPORTED"
    elif all(rejected.values()):
        outcome = "REJECTED"
    else:
        outcome = "PARTIAL"
    disagree = len({("S" if supported[s] else "R" if rejected[s] else "P") for s in checks}) > 1
    return {"outcome": outcome, "checks": checks, "seed_supported": supported, "seed_rejected": rejected, "seeds_disagree": disagree}


def r_checks(per_set: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """(R) for one seed: on each of T1, T3, T4, T5, T7 the upper bound of noul Brier, score Brier (< +0.01) and default-criterion regret
    (< +0.5), v0.4 − v0.1-mix. A missing set or reading fails (the rule cannot be read as passed)."""
    out: dict[str, Any] = {}
    for s in R_SETS:
        diffs = per_set.get(s) or {}
        out[s] = {key: {**(diffs.get(key) or {}), "ok": _lt((diffs.get(key) or {}).get("high"), limit)} for key, limit in R_LIMITS.items()}
    return {"sets": out, "pass": all(v["ok"] for entry in out.values() for v in entry.values())}


def n_checks(t9: Mapping[str, Any], t8b: Mapping[str, Any], home: Mapping[str, Any]) -> dict[str, Any]:
    """(N) for one seed: T9 and T8b direction / comparison v0.4 − v0.3 lower bounds > −0.05; T2–T4 balanced band v0.4 − v0.1-mix lower
    bound > −0.05 and severe-error point ≤ v0.1-mix's + 0.01."""
    checks = {
        "T9_direction_low_gt_-0.05": _gt((t9.get("direction") or {}).get("low"), N_LOW),
        "T9_comparison_low_gt_-0.05": _gt((t9.get("comparison") or {}).get("low"), N_LOW),
        "T8b_direction_low_gt_-0.05": _gt((t8b.get("direction") or {}).get("low"), N_LOW),
        "T8b_comparison_low_gt_-0.05": _gt((t8b.get("comparison") or {}).get("low"), N_LOW),
        "T2-T4_band_low_gt_-0.05": _gt((home.get("band_balanced") or {}).get("low"), N_LOW),
        "T2-T4_severe_le_mix_plus_0.01": (home.get("severe_error") or {}).get("a") is not None and (home.get("severe_error") or {}).get("b") is not None
        and home["severe_error"]["a"] <= home["severe_error"]["b"] + N_SEVERE_SLACK,
    }
    return {"checks": checks, "pass": all(checks.values())}


def f_check(miss: Mapping[str, Any] | None) -> dict[str, Any]:
    """(F) for one seed: the T9 robust high-called-low rate of v0.4 (``a``) ≤ the base's (``b``), points on the items both answered."""
    miss = miss or {}
    ok = miss.get("a") is not None and miss.get("b") is not None and miss["a"] <= miss["b"]
    return {"v04": miss.get("a"), "base": miss.get("b"), "n": miss.get("n"), "pass": bool(ok)}


def g_check(acc: Mapping[str, Any] | None) -> dict[str, Any]:
    """(G) for one seed: G2 accuracy v0.4 − v0.3 ≥ −0.02 (point) and lower bound > −0.045."""
    acc = acc or {}
    checks = {"point_ge_-0.02": acc.get("diff") is not None and acc["diff"] >= G_POINT, "low_gt_-0.045": _gt(acc.get("low"), G_LOW)}
    return {"checks": checks, "pass": all(checks.values())}


def next_step(s: str, hset_up: Mapping[int, bool] | None) -> dict[str, Any]:
    """The pre-declared next-step reading: from S, and — for a rejected S — whether the H-set level went up for each seed."""
    if s == "SUPPORTED":
        return {"branch": "S supported", "next": "scene diversity with real factors is the level lever; next = law diversity + the MNLI cause check"}
    if s == "PARTIAL":
        return {"branch": "S partial", "next": "a third seed before any claim"}
    up = bool(hset_up) and len(hset_up) == len(SEEDS) and all(hset_up.values())
    if up:
        return {"branch": "S rejected and H-set up (each seed)", "next": "transfer works inside our generator but not to T9's scenes -> next = "
                "independent scene authoring style and engine factor coverage", "hset_up": dict(hset_up)}
    return {"branch": "S rejected and H-set flat (not up for each seed)", "next": "engine factor coverage (E1's dropped item) or more data per law",
            "hset_up": dict(hset_up or {})}


def decide(readings: Mapping[str, Any]) -> dict[str, Any]:
    """The whole rule on computed readings (``readings["seeds"][seed]`` with ``S``, ``R``, ``N``, ``F``, ``G`` and ``secondary.Hset_level``)."""
    seeds = {int(k): v for k, v in readings["seeds"].items()}
    s = s_outcome({seed: entry["S"] for seed, entry in seeds.items()})
    per_seed = {}
    for seed, entry in seeds.items():
        per_seed[seed] = {"R": r_checks(entry["R"]), "N": n_checks(entry["N"]["T9"], entry["N"]["T8b"], entry["N"]["home"]),
                          "F": f_check(entry["F"]), "G": g_check(entry["G"])}
    parts = {part: all(per_seed[seed][part]["pass"] for seed in seeds) for part in ("R", "N", "F", "G")}
    release = s["outcome"] == "SUPPORTED" and all(parts.values())
    failing = ([f"S {s['outcome']}"] if s["outcome"] != "SUPPORTED" else []) + [f"{p} fails" for p, ok in parts.items() if not ok]
    hset_up = {}
    for seed, entry in seeds.items():
        level = ((entry.get("secondary") or {}).get("Hset_level") or {}).get("v04_minus_v03") or {}
        hset_up[seed] = _gt(level.get("low"), 0.0)
    return {"S": s, "per_seed": per_seed, "parts": parts, "release_candidate": release,
            "release_line": "RELEASE CANDIDATE (the user decides)" if release else "not a release candidate: " + ", ".join(failing),
            "next_step": next_step(s["outcome"], hset_up)}


# --------------------------------------------------------------------------
# Readings
# --------------------------------------------------------------------------


def pair_clusters(records: Sequence[Mapping[str, Any]]) -> dict[str, str]:
    """Record id → its pair group (the smaller id of a one-fact pair, else its own id) — the cluster of T9, the H-set and B1."""
    ids = {str(r["id"]) for r in records}
    out = {}
    for record in records:
        rid = str(record["id"])
        partner = str(((record.get("meta") or {}).get("pair") or {}).get("of") or "")
        out[rid] = min(rid, partner) if partner in ids else rid
    return out


def regroup(items: Mapping[tuple, tuple[str, float]], clusters: Mapping[str, str]) -> dict[tuple, tuple[str, float]]:
    """Items with their cluster replaced by the record's pair group (system1 clusters a question by its record, a T5-style pair by
    ``pair:<base id>``)."""
    out = {}
    for key, (cluster, value) in items.items():
        rid = cluster[len("pair:"):] if cluster.startswith("pair:") else cluster
        out[key] = (clusters.get(rid, cluster), value)
    return out


def items_of(set_name: str, records: Sequence[Mapping[str, Any]], predictions: Sequence[Mapping[str, Any]], *, by_pair: bool) -> dict[str, Any]:
    """Every System-1 metric's items of one set (single order), clustered by pair group when `by_pair`."""
    from robo_jev.bench import system1

    found = system1.set_items(set_name, records, predictions, reading="single_order")
    if not by_pair:
        return found
    clusters = pair_clusters(records)
    return {metric: regroup(items, clusters) for metric, items in found.items()}


def robust_keys(records: Sequence[Mapping[str, Any]]) -> set[tuple[str, str]]:
    from robo_jev.bench import infamily

    return infamily.robust_keys(records)


def keep(items: Mapping[tuple, Any], keys: set[tuple[str, str]]) -> dict[tuple, Any]:
    return {k: v for k, v in items.items() if tuple(k[-2:]) in keys}


def _paired(a: Mapping, b: Mapping, *, n_boot: int, seed: int, stratified: bool = False) -> dict[str, Any] | None:
    from robo_jev.bench import system1

    return system1.paired_difference(dict(a), dict(b), n_boot=n_boot, seed=seed, stratified=stratified)


def seen_law(records: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    """T9's seen-law families: ``meta.law_group`` seen_law_new_scene or seen_law_new_mechanism (conveyor, overhang, suction)."""
    return [r for r in records if (r.get("meta") or {}).get("law_group") in SEEN_LAW_GROUPS]


def unseen_law(records: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    return [r for r in records if (r.get("meta") or {}).get("law_group") not in SEEN_LAW_GROUPS]


def level(records: Sequence[Mapping[str, Any]], set_name: str, a: Sequence, b: Sequence, *, n_boot: int, seed: int, keys: set | None = None,
          two_band: bool = False) -> dict[str, Any] | None:
    """Balanced band accuracy a − b over the yes/no + score questions of `records` (optionally only `keys`; two-band = low and high
    strata only), paired, cluster = pair group."""
    ia = items_of(set_name, records, a, by_pair=True).get("band_balanced") or {}
    ib = items_of(set_name, records, b, by_pair=True).get("band_balanced") or {}
    if keys is not None:
        ia, ib = keep(ia, keys), keep(ib, keys)
    if two_band:
        ia = {k: v for k, v in ia.items() if k[0] != "middle"}
        ib = {k: v for k, v in ib.items() if k[0] != "middle"}
    return _paired(ia, ib, n_boot=n_boot, seed=seed, stratified=True)


def t9_keys(t9: Sequence[Mapping[str, Any]]) -> tuple[set[tuple[str, str]], set[tuple[str, str]]]:
    """(robust, prior-dependent) (record id, question) keys of T9's yes/no and score questions (``meta.robustness[q].robust``)."""
    robust = robust_keys(t9)
    every = {(str(r["id"]), q) for r in t9 for q, x in r["request"]["questions"].items() if x.get("type") in ("noul", "score")}
    return robust, every - robust


def t9_readings(t9: Sequence[Mapping[str, Any]], mine: Sequence, v03: Sequence, base: Sequence, *, n_boot: int = 2000, seed: int = 0) -> dict[str, Any]:
    """The rule's T9 readings for one v0.4 model: (S) on the seen-law families (v0.4 − v0.3, v0.4 − base), (N) all-family direction and
    comparison (v0.4 − v0.3), (F) the robust high-called-low rate (v0.4 vs base), and the secondary T9 readings."""
    from robo_jev.bench import infamily

    seen, unseen = seen_law(t9), unseen_law(t9)
    robust, ambiguous = t9_keys(t9)
    a9, b9, base9 = (items_of("T9", t9, p, by_pair=True) for p in (mine, v03, base))
    return {
        "S": {"d_v03": level(seen, "T9", mine, v03, n_boot=n_boot, seed=seed), "d_base": level(seen, "T9", mine, base, n_boot=n_boot, seed=seed)},
        "N": {metric: _paired(a9.get(metric) or {}, b9.get(metric) or {}, n_boot=n_boot, seed=seed) for metric in ("direction", "comparison")},
        "F": _paired(keep(a9.get("severe_miss") or {}, robust), keep(base9.get("severe_miss") or {}, robust), n_boot=n_boot, seed=seed),
        "secondary": {
            "T9_unseen_law_level": {"v04_minus_v03": level(unseen, "T9", mine, v03, n_boot=n_boot, seed=seed),
                                    "v04_minus_base": level(unseen, "T9", mine, base, n_boot=n_boot, seed=seed)},
            "T9_robust_two_band": {"v04_minus_v03": level(t9, "T9", mine, v03, n_boot=n_boot, seed=seed, keys=robust, two_band=True),
                                   "v04_minus_base": level(t9, "T9", mine, base, n_boot=n_boot, seed=seed, keys=robust, two_band=True)},
            "T9_hedge_share_prior_dependent": {name: infamily.answer_profile(t9, p, ambiguous).get("hedge_share")
                                               for name, p in (("v04", mine), ("v03", v03), ("base", base))},
            "T9_direction_choice_vs_v03": _paired(a9.get("direction_choice") or {}, b9.get("direction_choice") or {}, n_boot=n_boot, seed=seed),
        },
    }


def compute(sets: Mapping[str, Sequence[Mapping[str, Any]]], preds: Mapping[str, Mapping[str, Sequence[Mapping[str, Any]]]], *, n_boot: int = 2000,
            seed: int = 0, seeds: Sequence[int] = SEEDS) -> dict[str, Any]:
    """Every reading of the rule (and the secondary ones) for each v0.4 seed. `sets[name]` = records; `preds[model][set]` = predictions
    for models v04s0, v04s1, v03, base, mix (missing secondary inputs are skipped)."""
    from robo_jev.bench import system1
    from robo_jev.decisions.metrics import compare_runs
    from robo_jev.decisions.single_probe import compare_probe

    t9, t8b = sets["T9"], sets["T8b"]
    seen = seen_law(t9)
    t9_robust, t9_ambiguous = t9_keys(t9)
    out: dict[str, Any] = {"seeds": {}, "sets": {"T9": {"records": len(t9), "seen_law_records": len(seen),
                                                          "seen_law_families": sorted({str(r["meta"].get("family")) for r in seen}),
                                                          "robust_questions": len(t9_robust), "prior_dependent_questions": len(t9_ambiguous)}}}
    for s in seeds:
        m = f"v04s{s}"
        mine = preds[m]
        entry: dict[str, Any] = {}
        nine = t9_readings(t9, mine["T9"], preds["v03"]["T9"], preds["base"]["T9"], n_boot=n_boot, seed=seed)
        entry["S"] = nine["S"]
        r_sets = {}
        for name in R_SETS:
            if name in sets and mine.get(name) and preds["mix"].get(name):
                comparison = compare_runs(list(sets[name]), list(mine[name]), list(preds["mix"][name]), n_boot=n_boot, seed=seed)
                diffs = comparison["readings"]["single_order"]["differences"]
                r_sets[name] = {key: diffs.get(key) for key in R_LIMITS}
        entry["R"] = r_sets
        a8, b8 = items_of("T8b", t8b, mine["T8b"], by_pair=False), items_of("T8b", t8b, preds["v03"]["T8b"], by_pair=False)
        home_a = system1.group_items({h: system1.set_items(h, sets[h], mine[h]) for h in HOME}, HOME)
        home_b = system1.group_items({h: system1.set_items(h, sets[h], preds["mix"][h]) for h in HOME}, HOME)
        entry["N"] = {
            "T9": nine["N"],
            "T8b": {metric: _paired(a8.get(metric) or {}, b8.get(metric) or {}, n_boot=n_boot, seed=seed) for metric in ("direction", "comparison")},
            "home": {"band_balanced": _paired(home_a.get("band_balanced") or {}, home_b.get("band_balanced") or {}, n_boot=n_boot, seed=seed, stratified=True),
                     "severe_error": _paired(home_a.get("severe_error") or {}, home_b.get("severe_error") or {}, n_boot=n_boot, seed=seed)},
        }
        entry["F"] = nine["F"]
        g2 = compare_probe(list(sets["G2"]), list(mine["G2"]), list(preds["v03"]["G2"]), dimensions=("domain",), cluster="record", n_boot=n_boot, seed=seed)
        single = g2["readings"]["single_order"]
        entry["G"] = {**single["table"]["_all"]["accuracy"], "questions": single["questions"]}
        sec: dict[str, Any] = {"G2_by_task_vs_v03": {k: v.get("accuracy") for k, v in single["table"]["domain"].items()}, **nine["secondary"]}
        if preds["base"].get("G2"):
            vs_base = compare_probe(list(sets["G2"]), list(mine["G2"]), list(preds["base"]["G2"]), dimensions=("domain",), cluster="record", n_boot=n_boot, seed=seed)
            sec["G2_vs_base"] = vs_base["readings"]["single_order"]["table"]["_all"]["accuracy"]
            sec["G2_by_task_vs_base"] = {k: v.get("accuracy") for k, v in vs_base["readings"]["single_order"]["table"]["domain"].items()}
        if sets.get("Hset") and mine.get("Hset") and preds["v03"].get("Hset"):
            sec["Hset_level"] = {"v04_minus_v03": level(sets["Hset"], "Hset", mine["Hset"], preds["v03"]["Hset"], n_boot=n_boot, seed=seed),
                                 "v04_minus_base": level(sets["Hset"], "Hset", mine["Hset"], preds["base"]["Hset"], n_boot=n_boot, seed=seed)
                                 if preds["base"].get("Hset") else None}
            h_a = items_of("Hset", sets["Hset"], mine["Hset"], by_pair=True)
            if preds["base"].get("Hset"):
                h_b = items_of("Hset", sets["Hset"], preds["base"]["Hset"], by_pair=True)
                sec["Hset_high_called_low_vs_base"] = _paired(h_a.get("severe_miss") or {}, h_b.get("severe_miss") or {}, n_boot=n_boot, seed=seed)
        deck_stack = [r for r in t8b if (r.get("meta") or {}).get("scene") in T8B_SEEN]
        sec["T8b_deck_stack_level"] = {"v04_minus_v03": _paired(items_of("T8b", deck_stack, mine["T8b"], by_pair=False).get("band_balanced") or {},
                                                                items_of("T8b", deck_stack, preds["v03"]["T8b"], by_pair=False).get("band_balanced") or {},
                                                                n_boot=n_boot, seed=seed, stratified=True),
                                       "v04_minus_base": _paired(items_of("T8b", deck_stack, mine["T8b"], by_pair=False).get("band_balanced") or {},
                                                                 items_of("T8b", deck_stack, preds["base"]["T8b"], by_pair=False).get("band_balanced") or {},
                                                                 n_boot=n_boot, seed=seed, stratified=True) if preds["base"].get("T8b") else None}
        if sets.get("B1") and mine.get("B1") and preds["v03"].get("B1"):
            b1_robust = robust_keys(sets["B1"])
            sec["B1_level_robust"] = {"v04_minus_v03": level(sets["B1"], "B1", mine["B1"], preds["v03"]["B1"], n_boot=n_boot, seed=seed, keys=b1_robust),
                                      "v04_minus_base": level(sets["B1"], "B1", mine["B1"], preds["base"]["B1"], n_boot=n_boot, seed=seed, keys=b1_robust)
                                      if preds["base"].get("B1") else None}
        if sets.get("T8a") and mine.get("T8a"):
            pa = system1.set_items("T8a", sets["T8a"], mine["T8a"]).get("choice_accuracy") or {}
            sec["PIQA"] = {rival: _paired(pa, system1.set_items("T8a", sets["T8a"], preds[rival]["T8a"]).get("choice_accuracy") or {}, n_boot=n_boot, seed=seed)
                           for rival in ("v03", "base") if preds[rival].get("T8a")}
        entry["secondary"] = sec
        out["seeds"][str(s)] = entry
    out["seed_spread"] = seed_spread(out["seeds"])
    return out


def _walk(node: Any, path: tuple[str, ...] = ()) -> list[tuple[tuple[str, ...], float]]:
    """(path, v0.4 value) of every paired reading in a seed's entry: ``a`` of a paired difference, ``a`` of an accuracy comparison."""
    found = []
    if isinstance(node, Mapping):
        if "a" in node and isinstance(node.get("a"), (int, float)) and ("value" in node or "diff" in node):
            found.append((path, float(node["a"])))
        for key, child in node.items():
            if key in ("a", "b"):
                continue
            found += _walk(child, (*path, str(key)))
    return found


def seed_spread(seeds: Mapping[str, Any]) -> dict[str, Any]:
    """Every reading of v0.4 present for both seeds: the two values and |s0 − s1| (secondary)."""
    if set(seeds) != {str(s) for s in SEEDS}:
        return {}
    first, second = dict(_walk(seeds[str(SEEDS[0])])), dict(_walk(seeds[str(SEEDS[1])]))
    return {"/".join(path): {"s0": first[path], "s1": second[path], "abs_diff": abs(first[path] - second[path])} for path in first if path in second}


# --------------------------------------------------------------------------
# Files
# --------------------------------------------------------------------------


def _read(path: Path) -> list[dict[str, Any]]:
    from robo_jev.decisions.records import read_records

    return read_records(path)


def check_source(predictions: Path, set_path: Path, records: Sequence[Mapping[str, Any]], preds: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Scored on this set file? (the diagnosis summary's input sha256), under the rule's conditions, and the coverage of the set's ids."""
    from robo_jev.decisions.records import file_sha256

    summary_path = Path(str(predictions).replace(".predictions.jsonl", ".json"))
    config = json.loads(summary_path.read_text(encoding="utf-8")).get("config") or {} if summary_path.is_file() else {}
    sha = file_sha256(set_path)
    ids, predicted = {str(r["id"]) for r in records}, {str(p["id"]) for p in preds}
    conditions = {key: sorted({json.dumps(p.get(key)) for p in preds}) for key in CONDITIONS}
    conditions = {k: (json.loads(v[0]) if len(v) == 1 else [json.loads(x) for x in v]) for k, v in conditions.items()}
    problems = []
    if config.get("input_sha256") != sha:
        problems.append(f"scored on another file or no summary (recorded {str(config.get('input_sha256'))[:12]}, set {sha[:12]})")
    problems += [f"{k} = {conditions[k]!r}, want {want!r}" for k, want in CONDITIONS.items() if conditions[k] != want]
    return {"predictions": str(predictions), "set": str(set_path), "set_sha256": sha, "recorded_input_sha256": config.get("input_sha256"),
            "model": config.get("model"), "checkpoint": (config.get("checkpoint") or {}).get("path"), "coverage": len(ids & predicted) / len(ids) if ids else None,
            "conditions": conditions, "problems": problems}


def prediction_paths(args: argparse.Namespace) -> dict[tuple[str, str], Path]:
    """(model, set) → the prediction file the rule reads (defaults: the v0.4 pods, the Spark directory, the earlier runs)."""
    paths: dict[tuple[str, str], Path] = {}
    for s, directory in ((0, args.s0_dir), (1, args.s1_dir)):
        for name in POD_SETS:
            paths[(f"v04s{s}", name)] = directory / pod_prediction_name(s, name)
    for name in ("T1", "T2", "T3", "T4", "T5", "T6", "T7", "G1", "G2", "T8a"):
        paths[("v03", name)] = args.v03_dir / f"after-v03-2b-{name}.predictions.jsonl"
        paths[("mix", name)] = args.mix_dir / f"after-v01mix-2b-{name}.predictions.jsonl"
    for name in SPARK_SETS:
        for model in ("v04s0", "v04s1", "v03", "base"):
            fresh = args.spark_dir / spark_prediction_name(name, model)  # score_spark.sh's file wins over an earlier task's
            paths[(model, name)] = fresh if fresh.is_file() or (model, name) not in EXISTING_SPARK else EXISTING_SPARK[(model, name)]
    paths[("base", "G2")] = args.base_g2
    paths[("base", "T8a")] = EXISTING_SPARK[("base", "T8a")]
    return paths


#: What the rule itself reads (missing → refuse); everything else is secondary (missing → skipped and noted).
def required_inputs() -> list[tuple[str, str]]:
    need = [(m, "T9") for m in ("v04s0", "v04s1", "v03", "base")] + [(m, "T8b") for m in ("v04s0", "v04s1", "v03")]
    need += [(m, s) for m in ("v04s0", "v04s1", "mix") for s in sorted(set(R_SETS) | set(HOME))]
    need += [(m, "G2") for m in ("v04s0", "v04s1", "v03")]
    return need


def _fmt(value: Any, digits: int = 3, signed: bool = False) -> str:
    if value is None:
        return "—"
    return f"{value:+.{digits}f}" if signed else f"{value:.{digits}f}"


def _ci(d: Mapping[str, Any] | None, *, key: str = "value") -> str:
    if not d or d.get(key) is None:
        return "—"
    return f"{_fmt(d[key], signed=True)} [{_fmt(d.get('low'), signed=True)}, {_fmt(d.get('high'), signed=True)}] ({_fmt(d.get('a'))} vs {_fmt(d.get('b'))})"


def render(result: Mapping[str, Any]) -> str:
    v, r = result["verdict"], result["readings"]
    s = v["S"]
    lines = [f"# v0.4 pre-declared rule — S {s['outcome']} · {v['release_line']}", "",
             f"Rule written {RULE['written']}. Applied by `scripts/verdict_v04.py` ({SCRIPT_VERSION}, git {result.get('git')}); "
             f"{result['n_boot']} paired cluster-bootstrap draws (seed {result['seed']}); single order. Hardware: T9, T8b, B1-infamily, H-set on Spark "
             "for every model; T1–T7, G2, T8a on the H100 pods.", "",
             "## (S) H-scene — T9 seen-law families, balanced band accuracy (all yes/no + score)", "",
             f"Seen-law records: {r['sets']['T9']['seen_law_records']} ({', '.join(r['sets']['T9']['seen_law_families'])}).", "",
             "| seed | v0.4 − v0.3 [95 %] (v0.4 vs v0.3) | v0.4 − base [95 %] | point ≥ +0.08 | low > 0 | base low > 0 | upper < +0.05 |", "|---|---|---|---|---|---|---|"]
    for seed, entry in r["seeds"].items():
        c = s["checks"][int(seed)]
        lines.append(f"| {seed} | {_ci(entry['S']['d_v03'])} | {_ci(entry['S']['d_base'])} | {c['v03_point_ge_0.08']} | {c['v03_low_gt_0']} | "
                     f"{c['base_low_gt_0']} | {c['v03_high_lt_0.05']} |")
    lines += ["", f"**S: {s['outcome']}**" + (" — the two seeds disagree" if s["seeds_disagree"] else ""), "",
              "## (R) (N) (F) (G) per seed", "", "| seed | R | N | F | G |", "|---|---|---|---|---|"]
    for seed, entry in v["per_seed"].items():
        lines.append(f"| {seed} | {'PASS' if entry['R']['pass'] else 'FAIL'} | {'PASS' if entry['N']['pass'] else 'FAIL'} | "
                     f"{'PASS' if entry['F']['pass'] else 'FAIL'} | {'PASS' if entry['G']['pass'] else 'FAIL'} |")
    for seed, entry in r["seeds"].items():
        checks = v["per_seed"][int(seed)]
        lines += ["", f"### seed {seed}", "", "| reading | Δ [95 %] (v0.4 vs reference) | check |", "|---|---|---|"]
        for name, sets in checks["R"]["sets"].items():
            for key, d in sets.items():
                lines.append(f"| R {name} {key} (vs v0.1-mix) | {_fmt(d.get('value'), 4, True)} [{_fmt(d.get('low'), 4, True)}, {_fmt(d.get('high'), 4, True)}] | {d['ok']} |")
        n = entry["N"]
        for label, d, key in (("N T9 direction (vs v0.3)", n["T9"]["direction"], "T9_direction_low_gt_-0.05"),
                              ("N T9 comparison (vs v0.3)", n["T9"]["comparison"], "T9_comparison_low_gt_-0.05"),
                              ("N T8b direction (vs v0.3)", n["T8b"]["direction"], "T8b_direction_low_gt_-0.05"),
                              ("N T8b comparison (vs v0.3)", n["T8b"]["comparison"], "T8b_comparison_low_gt_-0.05"),
                              ("N T2–T4 balanced band (vs v0.1-mix)", n["home"]["band_balanced"], "T2-T4_band_low_gt_-0.05"),
                              ("N T2–T4 severe (vs v0.1-mix + 0.01)", n["home"]["severe_error"], "T2-T4_severe_le_mix_plus_0.01")):
            lines.append(f"| {label} | {_ci(d)} | {checks['N']['checks'][key]} |")
        f = checks["F"]
        lines.append(f"| F T9 robust high-called-low (≤ base) | v0.4 {_fmt(f['v04'])} vs base {_fmt(f['base'])} (n {f['n']}) | {f['pass']} |")
        g = entry["G"]
        lines.append(f"| G G2 accuracy (vs v0.3) | {_fmt(g.get('diff'), signed=True)} [{_fmt(g.get('low'), signed=True)}, {_fmt(g.get('high'), signed=True)}] "
                     f"({_fmt(g.get('a'))} vs {_fmt(g.get('b'))}) | {checks['G']['pass']} |")
    ns = v["next_step"]
    lines += ["", "## Release and next step", "", f"- **{v['release_line']}**", f"- Next step ({ns['branch']}): {ns['next']}.", "",
              "## Secondary (reported, not deciding)", ""]
    for seed, entry in r["seeds"].items():
        sec = entry["secondary"]
        lines += [f"### seed {seed}", "", "| reading | Δ [95 %] (v0.4 vs reference) |", "|---|---|"]
        for name in ("Hset_level", "T9_unseen_law_level", "T8b_deck_stack_level", "B1_level_robust", "T9_robust_two_band"):
            for rival, d in (sec.get(name) or {}).items():
                lines.append(f"| {name} {rival} | {_ci(d)} |")
        for name in ("Hset_high_called_low_vs_base", "T9_direction_choice_vs_v03"):
            if sec.get(name):
                lines.append(f"| {name} | {_ci(sec[name])} |")
        for rival, d in (sec.get("PIQA") or {}).items():
            lines.append(f"| PIQA (T8a) vs {rival} | {_ci(d)} |")
        for rival in ("v03", "base"):
            for task, d in (sec.get(f"G2_by_task_vs_{rival}") or {}).items():
                d = d or {}
                lines.append(f"| G2 {task} vs {rival} | {_fmt(d.get('diff'), signed=True)} [{_fmt(d.get('low'), signed=True)}, {_fmt(d.get('high'), signed=True)}] "
                             f"({_fmt(d.get('a'))} vs {_fmt(d.get('b'))}) |")
        hedge = sec.get("T9_hedge_share_prior_dependent") or {}
        lines += ["", "Hedge share on prior-dependent T9 questions: " + ", ".join(f"{m} {_fmt(x)}" for m, x in hedge.items()) + ".", ""]
    spread = result["readings"].get("seed_spread") or {}
    if spread:
        lines += ["Seed spread (|s0 − s1| of v0.4's value), largest ten: " + "; ".join(
            f"{k} {_fmt(e['abs_diff'])}" for k, e in sorted(spread.items(), key=lambda kv: -kv[1]["abs_diff"])[:10]) + ".", ""]
    lines += ["## Notes", ""] + [f"- {note}" for note in result["notes"]] + ["", "## Sources", "", "| model / set | predictions | same set file | coverage |", "|---|---|---|---|"]
    for key, src in result["sources"].items():
        lines.append(f"| {key} | `{src['predictions']}` | {not src['problems']} | {_fmt(src['coverage'])} |")
    lines += ["", "## Rule (verbatim parts)", ""] + [f"- **{k}** — {t}" for k, t in RULE.items()]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--s0-dir", type=Path, required=True, help="fetched reports of the v0.4 seed-0 job (after-v04-s0-2b-*)")
    parser.add_argument("--s1-dir", type=Path, required=True, help="fetched reports of the v0.4 seed-1 job (after-v04-s1-2b-*)")
    parser.add_argument("--v03-dir", type=Path, default=V03_POD)
    parser.add_argument("--mix-dir", type=Path, default=MIX_POD)
    parser.add_argument("--spark-dir", type=Path, default=SPARK_DIR, help="score_spark.sh's predictions (<SET>-<tag>.predictions.jsonl)")
    parser.add_argument("--base-g2", type=Path, default=BASE_G2_POD)
    parser.add_argument("--hset", type=Path, default=SET_PATHS["Hset"], help="v0.4a's scene holdout (H-set)")
    parser.add_argument("--t9", type=Path, default=SET_PATHS["T9"], help="the sealed T9 (default); another file only for a pipeline check")
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--allow-mismatch", action="store_true", help="read even if a prediction file was scored on another set file or conditions")
    parser.add_argument("--out", type=Path, default=REPO / "artifacts" / "reports" / "v04" / "verdict.json")
    args = parser.parse_args(argv)
    set_paths = {**SET_PATHS, "Hset": args.hset, "T9": args.t9}
    if args.t9.resolve() != SET_PATHS["T9"].resolve():
        _log(f"T9 is not the sealed file: {args.t9} — a pipeline check, not the verdict")
    paths = prediction_paths(args)
    missing = [f"{m} {s}: {paths[(m, s)]}" for m, s in required_inputs() if not paths[(m, s)].is_file()]
    if missing:
        raise SystemExit("missing prediction files the rule reads:\n  " + "\n  ".join(missing))
    needed_sets = sorted({s for (_, s), p in paths.items() if p.is_file()})
    sets = {s: _read(set_paths[s]) for s in needed_sets if set_paths[s].is_file()}
    preds: dict[str, dict[str, list[dict[str, Any]]]] = {m: {} for m in MODELS}
    sources, notes = {}, []
    for (model, name), path in sorted(paths.items()):
        if not path.is_file() or name not in sets:
            if not path.is_file():
                notes.append(f"secondary input absent, skipped: {model} {name} ({path})")
            continue
        preds[model][name] = _read(path)
        sources[f"{model} {name}"] = check_source(path, set_paths[name], sets[name], preds[model][name])
    problems = [f"{k}: {'; '.join(v['problems'])}" for k, v in sources.items() if v["problems"]]
    seen = seen_law(sets["T9"])
    if args.t9.resolve() == SET_PATHS["T9"].resolve() and (len(seen) != SEEN_LAW_RECORDS or sorted({str(r["meta"].get("family")) for r in seen}) != sorted(SEEN_LAW_FAMILIES)):
        problems.append(f"T9's seen-law slice is {len(seen)} records of {sorted({str(r['meta'].get('family')) for r in seen})}, not the rule's "
                        f"{SEEN_LAW_RECORDS} records of {list(SEEN_LAW_FAMILIES)}")
    if problems and not args.allow_mismatch:
        raise SystemExit("refusing (use --allow-mismatch to read anyway):\n  " + "\n  ".join(problems))
    readings = compute(sets, preds, n_boot=args.n_boot, seed=args.seed)
    verdict = decide(readings)
    if args.t9.resolve() != SET_PATHS["T9"].resolve():
        notes.insert(0, f"PIPELINE CHECK ONLY: T9 is {args.t9}, not the sealed T9 — this is not the v0.4 verdict")
    notes += ["T9, T8b, B1-infamily and the H-set were scored on Spark (GB10) for every model; T1–T7, G2 and T8a on H100 pods (v0.4, v0.3, "
              "v0.1-mix; the base's G2 on the v0.1 main pod; the base's T8a on Spark).",
              "(R) reads scripts/diagnose_decisions.py --compare's differences (compare_runs, single order, record bootstrap) with the rule's 2,000 "
              "draws (v0.2's own verdict used the tool's default 1,000).",
              "(F) is read on the robust items both models answered; the high-called-low rate's denominator is the robust yes/no + score questions "
              "(the conditional rate among robust high labels orders the two models the same way on the same items)."]
    result = {"kind": "v04-verdict", "script_version": SCRIPT_VERSION, "git": _git_sha(), "created": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
              "rule_text": RULE, "n_boot": args.n_boot, "seed": args.seed, "verdict": verdict, "readings": readings, "sources": sources,
              "problems": problems, "notes": notes, "inputs": {k: str(v) for k, v in vars(args).items() if isinstance(v, Path)}}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=1, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    args.out.with_suffix(".md").write_text(render(result), encoding="utf-8")
    print(render(result).split("## Secondary")[0])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
