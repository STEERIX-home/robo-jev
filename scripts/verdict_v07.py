#!/usr/bin/env python
"""Apply the PRE-DECLARED v0.7 VERDICT RULE (progress.md, written 2026-10-15 ~05:45 KST, before any v0.7 data existed) mechanically.

    python scripts/verdict_v07.py --s0-dir artifacts/cloud/runs/<s0 run>-fetched/reports --s1-dir artifacts/cloud/runs/<s1 run>-fetched/reports
        # → artifacts/reports/v07/verdict.{json,md}
    python scripts/verdict_v07.py --pipeline-check        # existing files only; not the verdict (module docstring, last part)

**Seed means** (the rule's comparisons): arm v0.7 = v0.7-s0 + v0.7-s1, arm v0.6 = v0.6-s0 + v0.6-s1. Per question the arm's call distribution is
the mean of its two seeds' 1/L-weighted listings (:func:`verdict_v06.arm_calls`); every other per-item reading of an arm is the mean of its
seeds' item values. Paired cluster bootstrap, 2,000 draws, seed 0 (:func:`robo_jev.bench.system1._draws`): the same draws for every source of
one reading.

* **S-a (M2g), as amended before any M2g / M1g prediction existed** (progress.md, "AMENDMENT TO S-a", 2026-10-15 ~11:25 KST): components
  ``len2k``, ``len4k``, ``len8k`` with gap = DC(neutral wrapped_L) − DC(short), and ``cue`` with gap = DC(w4k-reassuring) − DC(short), on the call
  view (the M2g moments' yes/no and score questions; v0.5's truth classes; cluster = the moment's generator group). A component is ACTIVE iff
  v0.6's gap ≥ +0.05 (point). An active component passes iff Δgap = gap(v0.7) − gap(v0.6) ≤ −0.05 with its upper bound < 0 AND v0.7's gap ≤
  +0.03 (point); an inactive one passes iff v0.7's gap ≤ +0.03 (point — v0.7 must not create a gap). S-a PASS iff every component passes and at
  least one is active; NOT TESTABLE iff none is active; otherwise FAIL (a missing component fails). Each component is read with one bootstrap
  over its four sources (v0.7 wrapped, v0.7 short, v0.6 wrapped, v0.6 short) on the questions all four answered; ``w4k`` is the neutral sheet.
* **S-b (M1g)** — drop = acc(N = 8) − acc(N ≥ 17) (per item: the mean of its N = 8 listings minus the mean of its N = 21 and N = 26
  accuracies). PASS iff v0.7's drop ≤ 0.05 (point) AND the v0.7 − v0.6 Δdrop upper bound < 0 (item bootstrap) AND, on the physics items,
  v0.7's late-letter penalty (A–H minus Q–Z over the N ≥ 17 listings) − the untrained base's < +0.10 (point).
* **S**: SUPPORTED iff S-a and S-b PASS; PARTIAL iff exactly one passes; REJECTED iff neither. If S-a is NOT TESTABLE, S reads on S-b alone
  and makes no claim about context robustness: "S-b PASS (S-a not testable)", or REJECTED when S-b fails.
* **Guards**: (O) short-form outside DC on A4, T9 and T8b, v0.7 − v0.6 ≤ +0.02 (point) on each (T8b's truth classes from B2's alternative-prior
  labels, as v0.6); (N) A4 and T9 direction and comparison v0.7 − v0.6 lower bound > −0.05, and per v0.7 seed the T2–T4 balanced band vs
  v0.1-mix lower bound > −0.05 and severe ≤ v0.1-mix + 0.01; (R) v0.2's (C) vs v0.1-mix on T1, T3, T4, T5, T7 per v0.7 seed; (G) G2 v0.7 − v0.6
  ≥ −0.02 (point) and lower bound > −0.045; (F) A4 robust high-called-low of the v0.7 arm ≤ the untrained base's (point).
* **RELEASE CANDIDATE** iff S SUPPORTED and O, N, R, G, F (the user decides; a release would also raise the option cap to 26 in the
  export / serve code).
* **Secondary** (reported, never deciding): the tone panel (DC at 4k by boilerplate tone, per arm; predicted v0.6 reassuring > neutral, v0.7
  flat) with, added by the amendment, the alarming component DC(w4k-alarming) − DC(short) per arm (over-reaction check) and reassuring −
  neutral at 4k per arm; DC vs the untrained base on every outside set, short and wrapped; M1g per-letter accuracy (Q, Y); B1 / H-set
  short-form readings; seed spread. Also the gaps by sheet style and by source set.
* **Exclusions**: ``artifacts/v07/gates/overlap-v07.json`` (``scripts/v07_gates_build.py recheck-v07``, run on v0.7's training input before
  the gate is opened) lists M1g items and M2g moments that share text with v0.7's training input; they are left out of every reading
  (pre-declared by task v0.7b). The verdict refuses to read the gates without it (``--no-v07-recheck`` only for a pipeline check).

**Pipeline check** (``--pipeline-check``; writes ``artifacts/scratch/v07b/pipeline/``, says it is not the verdict): the same code on existing
files with the arms relabelled — "v0.7" = v0.6-s0 + v0.6-s1, "v0.6" = v0.4-s0 + v0.4-s1: task iface's M1 set stands in for M1g and its M2 files for
M2g (no tone panel, so S-a is read on the three length components only and the cue component is reported as absent), v0.6's and v0.4's pods
and Spark files for the standard sets. It reproduces task iface's per-model M1 readings (accuracy at
N = 8 and N ≥ 17, penalty, physics-subset penalty) and M2 DC deltas and levels from ``artifacts/iface/verdict.json``, v0.6's A4 DC per model
and arm and v0.6 − v0.4 call differences on A4, T9, T8b from ``artifacts/reports/v06/verdict.json`` — every difference is listed.
"""

from __future__ import annotations

import argparse
import datetime as dt
import functools
import importlib.util
import json
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

SCRIPT_VERSION = "v07-verdict-1"
RULE = {
    "written": "2026-10-15 ~05:45 KST, before any v0.7 data existed (progress.md: PRE-DECLARED v0.7 VERDICT RULE)",
    "change": "v0.7 = v0.6's training input with labels unchanged and one factor changed - request SHAPE, by transforming (not adding) records: "
              "~25 % of physics records wrapped into label-independent sectioned sheets (1.5k-8k chars, several independent styles, every original "
              "sentence verbatim, reassuring/alarming/neutral boilerplate and a right-or-wrong (p = 0.5) planner recommendation sampled independently "
              "of the label); ~8 % of physics choice questions expanded with 0-probability distractors to N in 9-26 (correct position spread over "
              "A...Z) plus licensed general MC training items (not ARC test) expanded likewise within the general share; ~5 % of physics records with "
              "situation facts moved into question instructions. Seeds 0 and 1, v0.6's recipe.",
    "seed_means": "Comparisons use SEED MEANS (arm v0.7 = s0+s1 vs arm v0.6 = s0+s1; per question the mean over the arm's seeds' listings, each "
                  "1/L), paired cluster bootstrap (2000 draws, seed 0).",
    "gates": "Gates are sealed and built by a separate author who never sees the training sheet generator: M2g = held-out moments (A4, H-set, B1) "
             "short vs wrapped at 2k/4k/8k in sheet styles unseen in training, plus a tone panel (neutral / reassuring / alarming boilerplate) at 4k; "
             "M1g = an MC source used in neither training nor M1 (licence-checked) plus physics items from A4/H-set at N in {8, 21, 26}, positions "
             "spread, same item across N.",
    "S-a": "AMENDED (written 2026-10-15 ~11:25 KST, BEFORE any M2g or M1g prediction of any model exists; S-b, guards and everything else "
           "unchanged): components c in {len2k, len4k, len8k} with gap_c = DC(neutral wrapped_L) - DC(short), and c = cue with gap_cue = "
           "DC(w4k-reassuring) - DC(short). A component is ACTIVE iff v0.6's gap_c >= +0.05 (point). An active component passes iff v0.7 - v0.6 "
           "dgap_c <= -0.05 with upper bound < 0 AND v0.7's gap_c <= +0.03 (point); an inactive component passes iff v0.7's gap_c <= +0.03 (point; "
           "v0.7 must not create a gap). S-a PASS iff every component passes and at least one is active; S-a NOT TESTABLE iff no component is "
           "active (then S reads on S-b alone and makes no claim about context robustness: 'S-b PASS (S-a not testable)' or REJECTED).",
    "S-a (original, superseded)": "(M2g) per arm gap_L = DC(wrapped_L) - DC(short) on the call view; PASS iff for every L, v0.7 - v0.6 dgap <= "
                                  "-0.05 with upper bound < 0 AND v0.7's gap_L <= +0.03 (point).",
    "S-b": "(M1g) drop = acc(N = 8) - acc(N >= 17); PASS iff v0.7's drop <= 0.05 (point) AND v0.7 - v0.6 ddrop upper bound < 0, AND on the physics "
           "subset v0.7's late-letter penalty (A-H minus Q-Z at N >= 17) - the untrained base's < +0.10.",
    "S": "SUPPORTED iff S-a PASS and S-b PASS; PARTIAL iff exactly one of them passes; REJECTED iff neither (or S-b fails while S-a is not "
         "testable). (Amended; originally: SUPPORTED iff S-a and S-b PASS; PARTIAL iff exactly one; REJECTED iff neither.)",
    "O": "short-form outside DC on A4, T9, T8b: v0.7 - v0.6 <= +0.02 (point) on each",
    "N": "short-form direction and comparison on A4 and T9: v0.7 - v0.6 lower bound > -0.05; T2-T4 balanced band vs v0.1-mix lower bound > -0.05 "
         "and severe <= v0.1-mix + 0.01 (each seed)",
    "R": "v0.2's (C) vs v0.1-mix on T1, T3, T4, T5, T7 (each seed)",
    "G": "G2 v0.7 - v0.6 >= -0.02 point and lower bound > -0.045",
    "F": "A4 robust high-called-low v0.7 arm <= untrained base (point)",
    "release": "RELEASE CANDIDATE iff S SUPPORTED and O, N, R, G, F (user decides; a release would also raise the option cap to 26 in the "
               "export/serve code)",
    "secondary": "the M2g tone panel (DC by boilerplate tone, v0.6 vs v0.7) [predicted: v0.6 reassuring > neutral, v0.7 flat]; DC vs untrained "
                 "base on every outside set short and wrapped; M1g per-letter accuracy (Q, Y); B1 / H-set short-form readings; seed spread; added "
                 "by the amendment: the alarming component DC(w4k-alarming) - DC(short) per arm (over-reaction check) and reassuring - neutral at "
                 "4k per arm",
    "next": "SUPPORTED -> shape covered; next = the short-form outside 'safe' default (option B: safety-weighted objective / engine labels) or "
            "option C, plus release procedure D. S-a fails -> label-independent context did not stop the cue-driven gap -> test the cue hypothesis "
            "directly (tone ablation, sheet without boilerplate) before more data. S-b fails -> distractor expansion is not enough -> plausible "
            "multi-candidate physics questions or a code-scheme change (e.g. avoid weak letters).",
    "prior": "Controller's prior: S-b more likely than S-a.",
}
#: S-a, S-b, (O), (N), (G) bounds. SA_ACTIVE: a component is active iff v0.6's gap ≥ +0.05 (the amendment).
SA_DGAP, SA_HIGH, SA_GAP, SA_ACTIVE = -0.05, 0.0, 0.03, 0.05
SB_DROP, SB_HIGH, SB_PENALTY = 0.05, 0.0, 0.10
O_POINT = 0.02
N_LOW, N_SEVERE_SLACK = -0.05, 0.01
G_POINT, G_LOW = -0.02, -0.045
SEEDS = (0, 1)
ARMS = {"v07": ("v07s0", "v07s1"), "v06": ("v06s0", "v06s1")}
LENGTHS = ("2k", "4k", "8k")
#: The amended S-a's components: name → the wrapped condition read against ``short`` (``w4k`` is the neutral 4k sheet).
SA_COMPONENTS = {"len2k": "w2k", "len4k": "w4k", "len8k": "w8k", "cue": "w4k-reassuring"}
#: S-a outcomes and the S label when S-a is not testable and S-b passes.
SA_PASS, SA_FAIL, SA_NOT_TESTABLE = "PASS", "FAIL", "NOT TESTABLE"
SB_ONLY = "S-b PASS (S-a not testable)"
TONES = {"neutral": "w4k", "reassuring": "w4k-reassuring", "alarming": "w4k-alarming"}
CONDITIONS = ("short", "w2k", "w4k", "w8k", "w4k-reassuring", "w4k-alarming")
O_SETS = ("A4", "T9", "T8b")
CALL_SETS = ("A4", "T9", "T8b", "Hset", "B1")
R_SETS = ("T1", "T3", "T4", "T5", "T7")
HOME = ("T2", "T3", "T4")
POD_SETS = ("T1", "T2", "T3", "T4", "T5", "T6", "T7", "G1", "G2", "T8a")
SPARK_SETS = ("A4", "T9", "T8b", "Hset", "B1")
GATE_MODELS = ("v07s0", "v07s1", "v06s0", "v06s1", "base")
MODELS = {"v07s0": "v0.7 seed 0", "v07s1": "v0.7 seed 1", "v06s0": "v0.6 seed 0", "v06s1": "v0.6 seed 1", "base": "untrained Qwen3.5-2B",
          "mix": "v0.1-mix", "v07": "v0.7 (arm, seed mean)", "v06": "v0.6 (arm, seed mean)"}
SPARK_TAGS = {"v07s0": "v07s0-2b", "v07s1": "v07s1-2b", "v06s0": "v06s0-2b", "v06s1": "v06s1-2b", "base": "base-2b"}
#: General (non-physics) sources of a many-option set: M1g's MMLU, task iface's ARC (pipeline check).
GENERAL_SOURCES = ("MMLU", "ARC-Easy", "ARC-Challenge")
#: Weak letters the rule's secondary reading names.
WEAK_LETTERS = ("Q", "Y")

ARTIFACTS = REPO / "artifacts"
GATES = ARTIFACTS / "v07" / "gates"
M2G_DIR, M1G_FILE = GATES / "m2g", GATES / "m1g" / "M1g.jsonl"
OVERLAP_V07 = GATES / "overlap-v07.json"
SPARK_DIR = ARTIFACTS / "reports" / "v07" / "spark"
V06_SPARK = ARTIFACTS / "reports" / "v06" / "spark"
V04_SPARK = ARTIFACTS / "reports" / "v04" / "spark"
V06_PODS = {0: ARTIFACTS / "cloud" / "runs" / "v06-s0-01-fetched" / "reports", 1: ARTIFACTS / "cloud" / "runs" / "v06-s1-01-fetched" / "reports"}
V04_PODS = {0: ARTIFACTS / "cloud" / "runs" / "v04-s0-01-fetched" / "reports", 1: ARTIFACTS / "cloud" / "runs" / "v04-s1-01-fetched" / "reports"}
MIX_POD = ARTIFACTS / "cloud" / "runs" / "v01-mix-01-fetched" / "reports"
V06_CKPT = {0: ARTIFACTS / "cloud" / "runs" / "v06-s0-01-fetched" / "runs" / "v06-s0-2b" / "checkpoint.pt",
            1: ARTIFACTS / "cloud" / "runs" / "v06-s1-01-fetched" / "runs" / "v06-s1-2b" / "checkpoint.pt"}
#: The base's Spark predictions (same conditions): A4 from the v0.6 chain, the others from earlier tasks.
BASE_SPARK = {"A4": V06_SPARK / "A4-base-2b.predictions.jsonl", "T9": ARTIFACTS / "reports" / "t9" / "T9-base-2b.predictions.jsonl",
              "T8b": ARTIFACTS / "reports" / "t8" / "T8b-base-2b.predictions.jsonl", "Hset": V04_SPARK / "Hset-base-2b.predictions.jsonl",
              "B1": ARTIFACTS / "reports" / "b1" / "B1-base-2b.predictions.jsonl"}
IFACE = ARTIFACTS / "iface"
PIPELINE_OUT = ARTIFACTS / "scratch" / "v07b" / "pipeline" / "verdict-pipeline.json"
#: The gate prediction files' scoring conditions (the Spark chain): M2g on the standard path with seeded orders, M1g on the many-option path.
M2G_CONDITIONS = {"mode": "direct", "format": "chat", "prompt_version": "v0c-p3", "orders": "seeded", "shuffle_seed": 1, "batch": 1, "limit": None}
M1G_CONDITIONS = {"mode": "direct", "format": "chat", "prompt_version": "v0c-p3", "batch": 1, "lift_option_cap": True, "limit_items": None}
#: Outputs never go here.
NEVER_WRITE = (ARTIFACTS / "reports" / "v04", ARTIFACTS / "reports" / "v05", ARTIFACTS / "reports" / "v06", ARTIFACTS / "t8", ARTIFACTS / "t9",
               ARTIFACTS / "b1", ARTIFACTS / "simgen", ARTIFACTS / "v06", ARTIFACTS / "iface", GATES)


@functools.lru_cache(maxsize=None)
def _script(name: str):
    spec = importlib.util.spec_from_file_location(f"v07_{Path(name).stem}", REPO / "scripts" / name)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def V4():
    """scripts/verdict_v04.py — pair clusters, System-1 items, (R) / home checks, file checks."""
    return _script("verdict_v04.py")


def V5():
    """scripts/verdict_v05.py — the call view (truth classes, 1/L call weights, call metrics and paired differences)."""
    return _script("verdict_v05.py")


def V6():
    """scripts/verdict_v06.py — seed means (arm_calls, arm_values), A4's comparison mapping, the G2 arm reading, T8b with B2."""
    return _script("verdict_v06.py")


def _log(message: str) -> None:
    print(f"[verdict-v07 {dt.datetime.now().strftime('%H:%M:%S')}] {message}", file=sys.stderr, flush=True)


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


def _le(value: float | None, bound: float) -> bool:
    return value is not None and value <= bound


def sa_component(entry: Mapping[str, Any] | None) -> dict[str, Any]:
    """One amended S-a component from its reading (``dgap``, ``gap_v07``, ``gap_v06`` — points and the Δgap interval): ACTIVE iff v0.6's gap
    ≥ +0.05 (point); an active component passes iff Δgap ≤ −0.05 with its upper bound < 0 AND v0.7's gap ≤ +0.03 (point); an inactive one
    passes iff v0.7's gap ≤ +0.03 (point). A component without v0.6's gap is ``missing`` and fails."""
    entry = entry or {}
    dgap, gap7, gap6 = entry.get("dgap") or {}, entry.get("gap_v07") or {}, entry.get("gap_v06") or {}
    if gap6.get("value") is None:
        return {"status": "missing", "checks": {}, "pass": False}
    active = gap6["value"] >= SA_ACTIVE
    checks = {"gap_v07_le_+0.03": _le(gap7.get("value"), SA_GAP)}
    if active:
        checks = {"dgap_le_-0.05": _le(dgap.get("value"), SA_DGAP), "dgap_high_lt_0": _lt(dgap.get("high"), SA_HIGH), **checks}
    return {"status": "active" if active else "inactive", "gap_v06": gap6["value"], "checks": checks, "pass": all(checks.values())}


def sa_checks(components: Mapping[str, Mapping[str, Any]] | None, *, required: Sequence[str] = tuple(SA_COMPONENTS)) -> dict[str, Any]:
    """The amended S-a over the `required` components (:data:`SA_COMPONENTS`): PASS iff every component passes and at least one is active;
    NOT TESTABLE iff every component is read and none is active; otherwise FAIL (a missing component fails)."""
    components = components or {}
    per = {name: sa_component(components.get(name)) for name in required}
    active = [n for n, c in per.items() if c["status"] == "active"]
    missing = [n for n, c in per.items() if c["status"] == "missing"]
    if missing:
        outcome = SA_FAIL
    elif not active:
        outcome = SA_NOT_TESTABLE
    elif all(c["pass"] for c in per.values()):
        outcome = SA_PASS
    else:
        outcome = SA_FAIL
    inactive_failing = [n for n, c in per.items() if c["status"] == "inactive" and not c["pass"]]
    return {"components": per, "active": active, "missing": missing, "inactive_failing": inactive_failing, "required": list(required),
            "outcome": outcome, "pass": outcome == SA_PASS, "testable": outcome != SA_NOT_TESTABLE}


def sb_checks(sb: Mapping[str, Any] | None) -> dict[str, Any]:
    """S-b: v0.7's drop ≤ 0.05 (point), the Δdrop upper bound < 0, and the physics penalty of v0.7 − the base's < +0.10 (point)."""
    sb = sb or {}
    checks = {"drop_v07_le_0.05": _le((sb.get("drop_v07") or {}).get("value"), SB_DROP),
              "ddrop_high_lt_0": _lt((sb.get("ddrop") or {}).get("high"), SB_HIGH),
              "physics_penalty_minus_base_lt_0.10": _lt((sb.get("physics_penalty_minus_base") or {}).get("value"), SB_PENALTY)}
    return {"checks": checks, "pass": all(checks.values())}


def s_outcome(sa: Mapping[str, Any], sb: Mapping[str, Any]) -> str:
    """S as amended: SUPPORTED iff S-a PASS and S-b PASS; PARTIAL iff exactly one passes; REJECTED iff neither. If S-a is NOT TESTABLE, S reads
    on S-b alone: "S-b PASS (S-a not testable)" or REJECTED."""
    if sa.get("outcome") == SA_NOT_TESTABLE:
        return SB_ONLY if sb.get("pass") else "REJECTED"
    passes = int(bool(sa.get("pass"))) + int(bool(sb.get("pass")))
    return {2: "SUPPORTED", 1: "PARTIAL", 0: "REJECTED"}[passes]


def o_check(o: Mapping[str, Mapping[str, Any] | None] | None) -> dict[str, Any]:
    """(O): the v0.7 − v0.6 short-form DC ≤ +0.02 (point) on each of A4, T9 and T8b (a missing reading fails)."""
    o = o or {}
    checks = {f"{name}_dDC_le_+0.02": _le((o.get(name) or {}).get("value"), O_POINT) for name in O_SETS}
    return {"checks": checks, "pass": all(checks.values())}


def n_checks(arm: Mapping[str, Mapping[str, Any] | None], home: Mapping[int, Mapping[str, Any]]) -> dict[str, Any]:
    """(N): A4 and T9 direction / comparison v0.7 − v0.6 (arm means) lower bound > −0.05; each v0.7 seed's T2–T4 balanced band vs v0.1-mix
    lower bound > −0.05 and severe-error point ≤ v0.1-mix's + 0.01 (a missing reading fails)."""
    checks = {f"{key}_low_gt_-0.05": _gt((arm.get(key) or {}).get("low"), N_LOW) for key in ("A4_direction", "A4_comparison", "T9_direction", "T9_comparison")}
    for seed in SEEDS:
        entry = home.get(seed) or {}
        band, severe = entry.get("band_balanced") or {}, entry.get("severe_error") or {}
        checks[f"T2-T4_band_low_gt_-0.05[s{seed}]"] = _gt(band.get("low"), N_LOW)
        checks[f"T2-T4_severe_le_mix_plus_0.01[s{seed}]"] = severe.get("a") is not None and severe.get("b") is not None and severe["a"] <= severe["b"] + N_SEVERE_SLACK
    return {"checks": checks, "pass": all(checks.values())}


def g_check(acc: Mapping[str, Any] | None) -> dict[str, Any]:
    """(G): G2 accuracy v0.7 − v0.6 (arm means) ≥ −0.02 (point) and lower bound > −0.045."""
    acc = acc or {}
    checks = {"point_ge_-0.02": acc.get("diff") is not None and acc["diff"] >= G_POINT, "low_gt_-0.045": _gt(acc.get("low"), G_LOW)}
    return {"checks": checks, "pass": all(checks.values())}


def f_check(miss: Mapping[str, Any] | None) -> dict[str, Any]:
    """(F): A4 robust high-called-low of the v0.7 arm (``a``) ≤ the untrained base's (``b``), points on the items both answered."""
    miss = miss or {}
    ok = miss.get("a") is not None and miss.get("b") is not None and miss["a"] <= miss["b"]
    return {"v07_arm": miss.get("a"), "base": miss.get("b"), "n": miss.get("n"), "pass": bool(ok)}


NEXT = {
    "SUPPORTED": "shape covered; next = the short-form outside 'safe' default (option B: safety-weighted objective / engine labels) or option C, "
                 "plus release procedure D",
    "S-a": "label-independent context did not stop the cue-driven gap -> test the cue hypothesis directly (tone ablation, sheet without "
           "boilerplate) before more data",
    "S-b": "distractor expansion is not enough -> plausible multi-candidate physics questions or a code-scheme change (e.g. avoid weak letters)",
    "S-a not testable": "S-a not testable (no component active: v0.6 showed no gap of +0.05 on any sheet component) -> the rule makes no claim "
                        "about context robustness and pre-declares no step for it",
}


def next_step(sa_outcome: str | bool, sb_pass: bool) -> dict[str, Any]:
    """The pre-declared next step: SUPPORTED's; else the step of each failing part (both when both fail). S-a NOT TESTABLE adds no S-a step
    (no claim about context robustness); with S-b failing the S-b step applies. ``sa_outcome`` may also be a bool (True = PASS)."""
    sa_outcome = (SA_PASS if sa_outcome else SA_FAIL) if isinstance(sa_outcome, bool) else str(sa_outcome)
    if sa_outcome == SA_NOT_TESTABLE:
        return {"branch": SB_ONLY if sb_pass else "S-b fails (S-a not testable)",
                "next": [NEXT["S-a not testable"]] + ([] if sb_pass else [NEXT["S-b"]])}
    sa_pass = sa_outcome == SA_PASS
    if sa_pass and sb_pass:
        return {"branch": "SUPPORTED", "next": [NEXT["SUPPORTED"]]}
    failing = [part for part, ok in (("S-a", sa_pass), ("S-b", sb_pass)) if not ok]
    return {"branch": " and ".join(f"{p} fails" for p in failing), "next": [NEXT[p] for p in failing]}


def decide(readings: Mapping[str, Any], *, sa_required: Sequence[str] = tuple(SA_COMPONENTS)) -> dict[str, Any]:
    """The whole rule on computed readings: ``S-a`` (per component — the amended rule), ``S-b``, ``O``, ``N`` (``arm`` and ``home`` per seed),
    ``R`` per seed (v0.4's helper reads it), ``G`` and ``F``. `sa_required` = the S-a components read (all four; the pipeline check's stand-in
    data has no tone panel, so it reads the three length components)."""
    sa, sb = sa_checks(readings.get("S-a"), required=sa_required), sb_checks(readings.get("S-b"))
    outcome = s_outcome(sa, sb)
    r = {int(seed): V4().r_checks(entry) for seed, entry in (readings.get("R") or {}).items()}
    parts = {"O": o_check(readings.get("O")),
             "N": n_checks((readings.get("N") or {}).get("arm") or {}, {int(k): v for k, v in ((readings.get("N") or {}).get("home") or {}).items()}),
             "R": {"per_seed": r, "pass": sorted(r) == list(SEEDS) and all(e["pass"] for e in r.values())},
             "G": g_check(readings.get("G")), "F": f_check(readings.get("F"))}
    ok = {p: bool(v["pass"]) for p, v in parts.items()}
    release = outcome == "SUPPORTED" and all(ok.values())
    failing = ([f"S {outcome}"] if outcome != "SUPPORTED" else []) + [f"{p} fails" for p, good in ok.items() if not good]
    return {"S": {"outcome": outcome, "S-a": sa, "S-b": sb}, "parts": parts, "passes": ok, "release_candidate": release,
            "release_line": "RELEASE CANDIDATE (the user decides; a release would also raise the option cap to 26 in the export / serve code)"
            if release else "not a release candidate: " + ", ".join(failing), "next_step": next_step(sa["outcome"], bool(sb["pass"]))}


# --------------------------------------------------------------------------
# M2g readings
# --------------------------------------------------------------------------


def source_of(record: Mapping[str, Any]) -> str:
    """The moment's original set: ``meta.source.set`` (M2g), else from the id (task iface's M2 records keep their original ids)."""
    found = ((record.get("meta") or {}).get("source") or {}).get("set")
    if found:
        return str(found)
    rid = str(record.get("id"))
    for prefix, name in (("v06a4-", "A4"), ("v04h-", "Hset"), ("b1-", "B1")):
        if rid.startswith(prefix):
            return name
    return "other"


def style_of(record: Mapping[str, Any]) -> str | None:
    return ((record.get("meta") or {}).get("gate") or {}).get("style")


def m2g_calls(records: Sequence[Mapping[str, Any]], predictions: Sequence[Mapping[str, Any]]) -> dict[tuple, tuple]:
    """Call items of a condition: key (source set, record id, question), cluster = ``<source set>:<generator group>`` (a one-fact pair's two
    records together — task iface's M2 clusters)."""
    from robo_jev.bench.infamily import group_of

    by_source: dict[str, list] = defaultdict(list)
    for record in records:
        by_source[source_of(record)].append(record)
    out: dict[tuple, tuple] = {}
    for source, rows in sorted(by_source.items()):
        clusters = {str(r["id"]): f"{source}:{group_of(r)}" for r in rows}
        out.update(V5().call_items(source, rows, predictions, clusters=clusters))
    return out


def dc_sources(sources: Sequence[Mapping[tuple, tuple]], *, n_boot: int, seed: int, keys: set | None = None) -> dict[str, Any]:
    """DC of each source on the questions every source answered (∩ `keys`), with the bootstrap draws (one set of cluster draws for all)."""
    v5 = V5()
    common = set.intersection(*(set(s) for s in sources))
    if keys is not None:
        common &= set(keys)
    common = sorted(common)
    if not common:
        return {"n": 0}
    sums, counts, clusters = v5._cluster_arrays(common, list(sources))
    points = [float(v5._metrics(s.sum(0), counts.sum(0))["DC"]) for s in sums]
    w = v5._draw_weights(clusters, n_boot, seed)
    n = w @ counts
    draws = np.stack([np.asarray(v5._metrics(np.einsum("bc,ctk->btk", w, s), n)["DC"], dtype=float) for s in sums])
    return {"n": len(common), "clusters": clusters, "points": points, "draws": draws}


def _reading(value: float, draws: np.ndarray, **extra: Any) -> dict[str, Any]:
    from robo_jev.bench.system1 import _interval

    low, high = _interval(np.asarray(draws, dtype=float))
    return {"value": float(value), "low": low, "high": high, **extra}


def gap_reading(wrapped: Mapping[tuple, tuple], short: Mapping[tuple, tuple], *, n_boot: int, seed: int, keys: set | None = None) -> dict[str, Any] | None:
    """gap = DC(wrapped) − DC(short) on the questions both answered (paired cluster bootstrap)."""
    d = dc_sources([wrapped, short], n_boot=n_boot, seed=seed, keys=keys)
    if not d.get("n"):
        return None
    p, b = d["points"], d["draws"]
    return _reading(p[0] - p[1], b[0] - b[1], wrapped=p[0], short=p[1], n=d["n"], clusters=d["clusters"])


def dgap_reading(a_w: Mapping, a_s: Mapping, b_w: Mapping, b_s: Mapping, *, n_boot: int, seed: int, keys: set | None = None) -> dict[str, Any] | None:
    """Δgap = (DC(a, wrapped) − DC(a, short)) − (DC(b, wrapped) − DC(b, short)), and each gap, on the questions all four sources answered
    (one bootstrap over the four)."""
    d = dc_sources([a_w, a_s, b_w, b_s], n_boot=n_boot, seed=seed, keys=keys)
    if not d.get("n"):
        return None
    p, b = d["points"], d["draws"]
    return {"dgap": _reading((p[0] - p[1]) - (p[2] - p[3]), (b[0] - b[1]) - (b[2] - b[3]), n=d["n"], clusters=d["clusters"]),
            "gap_v07": _reading(p[0] - p[1], b[0] - b[1], wrapped=p[0], short=p[1]),
            "gap_v06": _reading(p[2] - p[3], b[2] - b[3], wrapped=p[2], short=p[3])}


def m2g_readings(conditions: Mapping[str, Sequence[Mapping[str, Any]]], preds: Mapping[str, Mapping[str, Sequence[Mapping[str, Any]]]], *,
                 n_boot: int = 2000, seed: int = 0, exclude: set[str] | None = None) -> dict[str, Any]:
    """Every M2g reading. `conditions[cond]` = the condition's records (``short``, ``w2k``, ``w4k``, ``w8k`` and, when present, the tone
    variants); `preds[model][cond]` = predictions for the models present (the rule needs v07s0, v07s1, v06s0, v06s1; the base is context)."""
    exclude = set(exclude or ())
    records = {cond: [r for r in rows if str(r["id"]) not in exclude] for cond, rows in conditions.items()}
    calls: dict[tuple[str, str], dict] = {}
    for model, by_cond in preds.items():
        for cond, rows in records.items():
            if by_cond.get(cond) is not None:
                calls[(model, cond)] = m2g_calls(rows, by_cond[cond])
    for arm, (first, second) in ARMS.items():
        for cond in records:
            if (first, cond) in calls and (second, cond) in calls:
                calls[(arm, cond)] = V6().arm_calls(calls[(first, cond)], calls[(second, cond)])
    out: dict[str, Any] = {"records": {cond: len(rows) for cond, rows in records.items()}, "excluded_ids": sorted(exclude)}
    lengths = [length for length in LENGTHS if f"w{length}" in records]
    # S-a (arms; the amended rule's components — a component whose condition is absent is left out)
    sa = {}
    for name, cond in SA_COMPONENTS.items():
        if cond in records and all((arm, c) in calls for arm in ARMS for c in (cond, "short")):
            sa[name] = dgap_reading(calls[("v07", cond)], calls[("v07", "short")], calls[("v06", cond)], calls[("v06", "short")], n_boot=n_boot, seed=seed)
    out["S-a"] = sa
    # per model and arm: gap by length, DC levels
    models = [m for m in (*GATE_MODELS, *ARMS) if (m, "short") in calls]
    out["gaps"] = {m: {length: gap_reading(calls[(m, f"w{length}")], calls[(m, "short")], n_boot=n_boot, seed=seed) for length in lengths
                       if (m, f"w{length}") in calls} for m in models}
    out["levels"] = {m: {cond: V5().call_metrics(calls[(m, cond)], n_boot=n_boot, seed=seed).get("DC") for cond in records if (m, cond) in calls}
                     for m in models}
    # by source set and by style (the arms; reported)
    by_set_keys: dict[str, set] = defaultdict(set)
    by_style_keys: dict[str, set] = defaultdict(set)
    for r in records.get("short", []):
        for q in r["request"]["questions"]:
            key = (source_of(r), str(r["id"]), q)
            by_set_keys[source_of(r)].add(key)
            if style_of(r):
                by_style_keys[style_of(r)].add(key)
    for name, groups in (("by_set", by_set_keys), ("by_style", by_style_keys)):
        out[name] = {}
        for group, keys in sorted(groups.items()):
            out[name][group] = {length: dgap_reading(calls[("v07", f"w{length}")], calls[("v07", "short")], calls[("v06", f"w{length}")],
                                                     calls[("v06", "short")], n_boot=n_boot, seed=seed, keys=keys)
                                for length in lengths if all((arm, c) in calls for arm in ARMS for c in (f"w{length}", "short"))}
    # tone panel (4k): DC per tone, tone − neutral per model / arm, and v0.7 − v0.6 of the arms' differences
    tone = {}
    if all(c in records for c in TONES.values()):
        for m in models:
            if all((m, c) in calls for c in TONES.values()):
                tone[m] = {"DC": {t: (out["levels"][m].get(c) or {}) for t, c in TONES.items()},
                           **{f"{t}_minus_neutral": gap_reading(calls[(m, TONES[t])], calls[(m, "w4k")], n_boot=n_boot, seed=seed)
                              for t in ("reassuring", "alarming")}}
        if all((arm, c) in calls for arm in ARMS for c in TONES.values()):
            tone["v07_minus_v06"] = {f"{t}_minus_neutral": dgap_reading(calls[("v07", TONES[t])], calls[("v07", "w4k")], calls[("v06", TONES[t])],
                                                                         calls[("v06", "w4k")], n_boot=n_boot, seed=seed)
                                     for t in ("reassuring", "alarming")}
    out["tone_panel"] = tone
    # added by the amendment (secondary): the alarming component DC(w4k-alarming) − DC(short) per arm (over-reaction check) and reassuring −
    # neutral at 4k per arm (also per model; the arms' difference v0.7 − v0.6 for each)
    amended: dict[str, Any] = {}
    for name, (cond, ref) in (("alarming_minus_short", ("w4k-alarming", "short")), ("reassuring_minus_neutral", ("w4k-reassuring", "w4k"))):
        if cond in records and ref in records:
            amended[name] = {m: gap_reading(calls[(m, cond)], calls[(m, ref)], n_boot=n_boot, seed=seed) for m in models if (m, cond) in calls and (m, ref) in calls}
            if all((arm, c) in calls for arm in ARMS for c in (cond, ref)):
                amended[name]["v07_minus_v06"] = dgap_reading(calls[("v07", cond)], calls[("v07", ref)], calls[("v06", cond)], calls[("v06", ref)],
                                                              n_boot=n_boot, seed=seed)
    out["amendment_secondary"] = amended
    # DC vs the untrained base per condition (pooled and by source set)
    vs_base: dict[str, Any] = {}
    for cond in records:
        if ("base", cond) not in calls:
            continue
        vs_base[cond] = {arm: {"pooled": V5().call_difference(calls[(arm, cond)], calls[("base", cond)], n_boot=n_boot, seed=seed).get("DC"),
                               **{s: V5().call_difference({k: v for k, v in calls[(arm, cond)].items() if k[0] == s},
                                                          {k: v for k, v in calls[("base", cond)].items() if k[0] == s}, n_boot=n_boot, seed=seed).get("DC")
                                  for s in sorted(by_set_keys)}}
                         for arm in ARMS if (arm, cond) in calls}
    out["vs_base"] = vs_base
    return out


# --------------------------------------------------------------------------
# M1g readings
# --------------------------------------------------------------------------


def _ci(values: np.ndarray, draws: np.ndarray) -> dict[str, Any]:
    from robo_jev.bench.system1 import _interval

    boot = values[draws].mean(1)
    low, high = _interval(boot)
    return {"value": float(values.mean()), "low": low, "high": high, "n": int(values.size)}


def m1g_readings(records: Sequence[Mapping[str, Any]], preds: Mapping[str, Sequence[Mapping[str, Any]]], *, n_boot: int = 2000, seed: int = 0,
                 exclude: set[str] | None = None) -> dict[str, Any]:
    """Every M1g reading on the items every model answered completely (N = 8, 21, 26, three listings each), minus `exclude`: per model and
    arm (seed means of the item values) accuracy by N, drop, bucket accuracies, penalty; S-b's three readings; per-letter accuracy at N = 21
    and 26; by source. Item-cluster bootstrap with the same draws for every model."""
    from robo_jev.bench import gates_v07 as G
    from robo_jev.bench import iface
    from robo_jev.bench.system1 import _draws

    rows = {m: iface.m1_rows(records, p) for m, p in preds.items()}
    candidates = sorted({str(r["meta"]["item"]) for r in records} - set(exclude or ()))
    tables = {m: G.m1g_item_table(rows[m], candidates) for m in rows}
    complete = np.all([t["complete"] > 0 for t in tables.values()], axis=0)
    items = [i for i, ok in zip(candidates, complete) if ok]
    if not items:
        raise ValueError("M1g: no item has every listing scored for every model")
    tables = {m: {k: v[complete] for k, v in t.items()} for m, t in tables.items()}
    for arm, (first, second) in ARMS.items():
        if first in tables and second in tables:
            tables[arm] = {k: (tables[first][k] + tables[second][k]) / 2.0 for k in tables[first]}
    source = {str(r["meta"]["item"]): str(r["meta"]["source"]) for r in records}
    physics = np.array([source[i] not in GENERAL_SOURCES for i in items])
    draws = _draws(len(items), n_boot, seed)
    out: dict[str, Any] = {"items": len(items), "items_physics": int(physics.sum()), "excluded": sorted(set(exclude or ())), "models": {}}
    for m, t in tables.items():
        out["models"][m] = {metric: _ci(t[metric], draws) for metric in G.M1G_METRICS if np.all(np.isfinite(t[metric]))}
    sb: dict[str, Any] = {}
    if all(arm in tables for arm in ARMS):
        sb["drop_v07"] = out["models"]["v07"]["drop"]
        sb["drop_v06"] = out["models"]["v06"]["drop"]
        sb["ddrop"] = _ci(tables["v07"]["drop"] - tables["v06"]["drop"], draws) | {"a": float(tables["v07"]["drop"].mean()), "b": float(tables["v06"]["drop"].mean())}
    phys_draws = _draws(int(physics.sum()), n_boot, seed) if physics.any() else None
    by_physics: dict[str, Any] = {}
    if physics.any():
        for m, t in tables.items():
            by_physics[m] = {metric: _ci(t[metric][physics], phys_draws) for metric in ("acc@8", "acc@late", "drop", "bucket:A-H", "bucket:Q-Z", "penalty")}
            if m != "base" and "base" in tables:
                by_physics[m]["penalty_minus_base"] = _ci(t["penalty"][physics] - tables["base"]["penalty"][physics], phys_draws)
        if "v07" in tables and "base" in tables:
            sb["physics_penalty_minus_base"] = by_physics["v07"]["penalty_minus_base"]
            sb["physics_penalty_v07"] = by_physics["v07"]["penalty"]
            sb["physics_penalty_base"] = by_physics["base"]["penalty"]
    out["S-b"] = sb
    out["physics"] = by_physics
    general = ~physics
    if general.any():
        gen_draws = _draws(int(general.sum()), n_boot, seed)
        out["general"] = {m: {metric: _ci(t[metric][general], gen_draws) for metric in ("acc@8", "acc@late", "drop", "penalty")} for m, t in tables.items()}
    # per-letter accuracy at N = 21 and 26 (the items above; per model, then the arms' seed means)
    keep = set(items)
    letters: dict[str, Any] = {}
    for m, model_rows in rows.items():
        acc: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
        for row in model_rows:
            if row["item"] in keep and row["n"] >= iface.LATE_MIN_N:
                acc[str(row["n"])][chr(ord("A") + row["position"])].append(row["correct"])
        letters[m] = {n: {letter: float(np.mean(v)) for letter, v in sorted(by_letter.items())} for n, by_letter in sorted(acc.items())}
    for arm, (first, second) in ARMS.items():
        if first in letters and second in letters:
            letters[arm] = {n: {k: (letters[first][n][k] + letters[second][n][k]) / 2.0 for k in letters[first][n]} for n in letters[first]}
    out["letters"] = letters
    out["weak_letters"] = {m: {n: {k: v.get(k) for k in WEAK_LETTERS if k in v} for n, v in by_n.items()} for m, by_n in letters.items()}
    return out


# --------------------------------------------------------------------------
# Standard readings (O, N, R, G, F and the secondary short-form readings)
# --------------------------------------------------------------------------


def standard_readings(sets: Mapping[str, Sequence[Mapping[str, Any]]], preds: Mapping[str, Mapping[str, Sequence[Mapping[str, Any]]]], *,
                      n_boot: int = 2000, seed: int = 0, compare_override: Mapping[str, Mapping[str, Any]] | None = None) -> dict[str, Any]:
    """(O), (F), (N), (R), (G) and the secondary short-form readings. `sets[name]` = records (T8b with B2's robustness); `preds[model][set]`
    for v07s0, v07s1, v06s0, v06s1, base, mix — v0.6's verdict code with the arms v0.7 and v0.6."""
    from robo_jev.bench import infamily, system1
    from robo_jev.decisions.metrics import compare_runs

    v4, v5, v6 = V4(), V5(), V6()
    a4, t9 = sets["A4"], sets["T9"]
    clusters = {name: v4.pair_clusters(sets[name]) for name in CALL_SETS if sets.get(name)}
    calls = {(m, s): v5.call_items(s, sets[s], preds[m][s], clusters=clusters[s]) for m in preds for s in CALL_SETS if sets.get(s) and preds[m].get(s)}
    for arm, (first, second) in ARMS.items():
        for s in CALL_SETS:
            if (first, s) in calls and (second, s) in calls:
                calls[(arm, s)] = v6.arm_calls(calls[(first, s)], calls[(second, s)])
    out: dict[str, Any] = {"sets": {s: {"records": len(sets[s]), "clusters": len(set(clusters[s].values())),
                                        "classes": dict(Counter(t for r in sets[s] for q in r["request"]["questions"] if (t := v5.truth_class(r, q))))}
                                    for s in CALL_SETS if sets.get(s)}}
    # (O)
    out["O"] = {s: v5.call_difference(calls[("v07", s)], calls[("v06", s)], n_boot=n_boot, seed=seed).get("DC")
                for s in O_SETS if ("v07", s) in calls and ("v06", s) in calls}
    # (F)
    robust_a4 = infamily.robust_keys(a4)
    miss = {m: v4.keep(v4.items_of("A4", a4, preds[m]["A4"], by_pair=True).get("severe_miss") or {}, robust_a4) for m in ("v07s0", "v07s1", "base")}
    out["F"] = v4._paired(v6.arm_values(miss["v07s0"], miss["v07s1"]), miss["base"], n_boot=n_boot, seed=seed)
    # (N)
    mapping = v6.comparison_map(a4, compare_override)
    out["A4_comparison_map"] = mapping
    a4_items = {m: v4.items_of("A4", a4, preds[m]["A4"], by_pair=True) for m in ("v07s0", "v07s1", "v06s0", "v06s1", "base")}
    for m in a4_items:
        a4_items[m]["comparison_compare_question"] = v6.comparison_items("A4", a4, preds[m]["A4"], mapping, clusters=clusters["A4"])
    t9_items = {m: v4.items_of("T9", t9, preds[m]["T9"], by_pair=True) for m in ("v07s0", "v07s1", "v06s0", "v06s1") if preds[m].get("T9")}

    def arm_diff(items: Mapping[str, Mapping[str, Any]], metric: str, *, stratified: bool = False) -> dict[str, Any] | None:
        a = v6.arm_values(items["v07s0"].get(metric) or {}, items["v07s1"].get(metric) or {})
        b = v6.arm_values(items["v06s0"].get(metric) or {}, items["v06s1"].get(metric) or {})
        return v4._paired(a, b, n_boot=n_boot, seed=seed, stratified=stratified)

    home = {}
    for s in SEEDS:
        mine = preds[f"v07s{s}"]
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
        mine, per_set = preds[f"v07s{s}"], {}
        for name in R_SETS:
            if sets.get(name) and mine.get(name) and preds["mix"].get(name):
                comparison = compare_runs(list(sets[name]), list(mine[name]), list(preds["mix"][name]), n_boot=n_boot, seed=seed)
                diffs = comparison["readings"]["single_order"]["differences"]
                per_set[name] = {key: diffs.get(key) for key in v4.R_LIMITS}
        r[str(s)] = per_set
    out["R"] = r
    # (G)
    g = v6.g2_arm_comparison(sets["G2"], [preds["v07s0"]["G2"], preds["v07s1"]["G2"]], [preds["v06s0"]["G2"], preds["v06s1"]["G2"]], n_boot=n_boot, seed=seed)
    sec: dict[str, Any] = {"G2_by_task_v07_minus_v06": g.pop("by_task", None)}
    out["G"] = g
    # secondary: call view v0.7 − v0.6 on every set (B1 / H-set short form included), DC vs the base, A4 / T9 direction and comparison vs base
    sec["calls_v07_minus_v06"] = {s: v5.call_difference(calls[("v07", s)], calls[("v06", s)], n_boot=n_boot, seed=seed)
                                  for s in CALL_SETS if ("v07", s) in calls and ("v06", s) in calls}
    sec["calls_vs_base"] = {s: {arm: v5.call_difference(calls[(arm, s)], calls[("base", s)], n_boot=n_boot, seed=seed) for arm in ARMS if (arm, s) in calls}
                            for s in CALL_SETS if ("base", s) in calls}
    for name in ("Hset", "B1"):
        if sets.get(name) and all(preds[m].get(name) for m in ("v07s0", "v07s1", "v06s0", "v06s1")):
            items = {m: v4.items_of(name, sets[name], preds[m][name], by_pair=True) for m in ("v07s0", "v07s1", "v06s0", "v06s1")}
            sec[f"{name}_short_form_v07_minus_v06"] = {metric: arm_diff(items, metric, stratified=(metric == "band_balanced"))
                                                       for metric in ("direction", "band_balanced") if items["v07s0"].get(metric)}
    out["secondary"] = sec
    out["context"] = {s: {m: v5.call_metrics(calls[(m, s)], n_boot=n_boot, seed=seed) for m in (*GATE_MODELS, *ARMS) if (m, s) in calls} for s in CALL_SETS}
    spread = {}
    for s in CALL_SETS:
        for arm, (first, second) in ARMS.items():
            a, b = out["context"][s].get(first) or {}, out["context"][s].get(second) or {}
            if a and b:
                spread[f"{s}/{arm}"] = {m: abs(a[m]["value"] - b[m]["value"]) if a.get(m, {}).get("value") is not None and b.get(m, {}).get("value") is not None else None
                                        for m in ("DC", "LU", "LC", "GAP", "CA")}
    out["seed_spread"] = spread
    return out


def gate_spread(m2g: Mapping[str, Any], m1g: Mapping[str, Any]) -> dict[str, Any]:
    """Seed spread of the gates: |s0 − s1| of each arm's M2g gap per length and M1g drop / physics penalty."""
    out: dict[str, Any] = {}
    for arm, (first, second) in ARMS.items():
        gaps = m2g.get("gaps") or {}
        out[f"M2g/{arm}"] = {length: abs(gaps[first][length]["value"] - gaps[second][length]["value"])
                             for length in LENGTHS if gaps.get(first, {}).get(length) and gaps.get(second, {}).get(length)}
        models = m1g.get("models") or {}
        if first in models and second in models:
            out[f"M1g/{arm}"] = {"drop": abs(models[first]["drop"]["value"] - models[second]["drop"]["value"])}
            phys = m1g.get("physics") or {}
            if first in phys and second in phys:
                out[f"M1g/{arm}"]["physics_penalty"] = abs(phys[first]["penalty"]["value"] - phys[second]["penalty"]["value"])
    return out


def assemble(m2g: Mapping[str, Any], m1g: Mapping[str, Any], standard: Mapping[str, Any]) -> dict[str, Any]:
    """The rule's readings from the three parts (the input of :func:`decide`)."""
    return {"S-a": {name: entry for name, entry in (m2g.get("S-a") or {}).items() if entry},
            "S-b": m1g.get("S-b"), "O": standard.get("O"), "N": standard.get("N"), "R": standard.get("R"), "G": standard.get("G"), "F": standard.get("F")}


# --------------------------------------------------------------------------
# Files
# --------------------------------------------------------------------------


def refuse_write(path: Path) -> None:
    resolved = Path(path).resolve()
    for place in NEVER_WRITE:
        place = place.resolve()
        if resolved == place or place in resolved.parents:
            raise SystemExit(f"{path}: refusing to write into {place}")


def _read(path: Path) -> list[dict[str, Any]]:
    from robo_jev.decisions.records import read_records

    return read_records(path)


@functools.lru_cache(maxsize=None)
def _sha(path: str) -> str:
    from robo_jev.decisions.records import file_sha256

    return file_sha256(path)


def check_gate_source(predictions: Path, input_path: Path, conditions: Mapping[str, Any], checkpoint: Path | None) -> dict[str, Any]:
    """A gate prediction file's summary: the scoring conditions, the scored file (sha256), the prompt digest and the checkpoint (sha256 of
    the file on disk; none for the base)."""
    from robo_jev.decisions.prompt import PROMPT_VERSION, fixed_text_digest

    summary = Path(str(predictions).replace(".predictions.jsonl", ".json"))
    config = (json.loads(summary.read_text(encoding="utf-8")).get("config") or {}) if summary.is_file() else {}
    problems = [] if summary.is_file() else [f"no summary {summary.name}"]
    problems += [f"{k} = {config.get(k)!r}, want {v!r}" for k, v in conditions.items() if config.get(k) != v]
    if config.get("input_sha256") != _sha(str(input_path)):
        problems.append(f"scored {str(config.get('input_sha256'))[:12]}, not {input_path.name} ({_sha(str(input_path))[:12]})")
    if config.get("prompt_digest") != fixed_text_digest(PROMPT_VERSION):
        problems.append("prompt digest differs")
    got = (config.get("checkpoint") or {}).get("sha256")
    want = _sha(str(checkpoint)) if checkpoint is not None and Path(checkpoint).is_file() else None
    if checkpoint is None and got is not None:
        problems.append("a checkpoint was applied to the base")
    elif checkpoint is not None and (want is None or got != want):
        problems.append(f"checkpoint sha256 {str(got)[:12]} vs {str(want)[:12]} on disk ({checkpoint})")
    return {"predictions": str(predictions), "model": config.get("model"), "input_sha256": config.get("input_sha256"), "checkpoint_sha256": got,
            "problems": problems}


def _set_paths() -> dict[str, Path]:
    return dict(V6().SET_PATHS)


def _load_standard(paths: Mapping[tuple[str, str], Path], set_paths: Mapping[str, Path], *, allow_mismatch: bool, notes: list[str]) -> tuple[dict, dict, dict]:
    """Sets and predictions of the standard sets; every prediction file checked against its set file and the diagnosis conditions."""
    needed = sorted({s for (_, s), p in paths.items() if p.is_file()} | {"A4", "T9", "G2"})
    sets = {s: _read(set_paths[s]) for s in needed if Path(set_paths[s]).is_file()}
    if "T8b" in sets:
        sets["T8b"], b2 = V6().t8b_with_b2(sets["T8b"])
        notes.append(f"T8b truth classes from B2's alternative-prior labels ({b2['labels']}; original-prior reproduction share {b2['reproduction_share']:.4f}).")
    preds: dict[str, dict[str, list]] = defaultdict(dict)
    sources = {}
    for (model, name), path in sorted(paths.items()):
        if not path.is_file() or name not in sets:
            if not path.is_file():
                notes.append(f"input absent, skipped: {model} {name} ({path})")
            continue
        preds[model][name] = _read(path)
        sources[f"{model} {name}"] = V4().check_source(path, set_paths[name], sets[name], preds[model][name])
    problems = [f"{k}: {'; '.join(v['problems'])}" for k, v in sources.items() if v["problems"]]
    if problems and not allow_mismatch:
        raise SystemExit("refusing (use --allow-mismatch to read anyway):\n  " + "\n  ".join(problems))
    for model in ("v07s0", "v07s1", "v06s0", "v06s1", "base", "mix"):
        preds.setdefault(model, {})
    return sets, dict(preds), sources


def standard_paths(args: argparse.Namespace, *, relabel: Mapping[str, str] | None = None) -> dict[tuple[str, str], Path]:
    """(model, set) → the prediction file. The verdict: v0.7's pods (--s0-dir / --s1-dir, ``after-v07-s<seed>-2b-<SET>``), v0.6's pods, v0.1-mix,
    the base's G2; Spark: the v0.7 chain's files for v0.7, v0.6's chain for v0.6, the base's earlier files. The pipeline check (`relabel`):
    "v07" = v0.6's files, "v06" = v0.4's."""
    v6 = V6()
    paths: dict[tuple[str, str], Path] = {}
    if relabel is None:
        pods = {"v07": (args.s0_dir, args.s1_dir), "v06": (V06_PODS[0], V06_PODS[1])}
        version = {"v07": "v07", "v06": "v06"}
        spark = {"v07": args.spark_dir, "v06": V06_SPARK}
        spark_tag = {"v07": "v07", "v06": "v06"}
    else:
        pods = {"v07": (V06_PODS[0], V06_PODS[1]), "v06": (V04_PODS[0], V04_PODS[1])}
        version = {"v07": "v06", "v06": "v04"}
        spark = {"v07": V06_SPARK, "v06": V04_SPARK}
        spark_tag = {"v07": "v06", "v06": "v04"}
    for arm in ARMS:
        for s in SEEDS:
            for name in POD_SETS:
                paths[(f"{arm}s{s}", name)] = Path(pods[arm][s]) / v6.pod_prediction_name(version[arm], s, name)
            for name in SPARK_SETS:
                directory = spark[arm] if not (relabel is not None and arm == "v06" and name == "A4") else V06_SPARK  # v0.4's A4 is in v0.6's chain
                paths[(f"{arm}s{s}", name)] = Path(directory) / f"{name}-{spark_tag[arm]}s{s}-2b.predictions.jsonl"
    for name in POD_SETS:
        paths[("mix", name)] = MIX_POD / f"after-v01mix-2b-{name}.predictions.jsonl"
    for name, path in BASE_SPARK.items():
        paths[("base", name)] = path
    paths[("base", "G2")] = v6.BASE_G2_POD
    return paths


def required_standard() -> list[tuple[str, str]]:
    need = [(m, s) for m in ("v07s0", "v07s1", "v06s0", "v06s1") for s in O_SETS] + [("base", "A4")]
    need += [(m, s) for m in ("v07s0", "v07s1", "mix") for s in sorted(set(R_SETS) | set(HOME))]
    need += [(m, "G2") for m in ("v07s0", "v07s1", "v06s0", "v06s1")]
    return need


def gate_paths(spark_dir: Path) -> dict[str, Any]:
    """The gate prediction files the Spark chain writes: ``M1g-<tag>`` and ``M2g-<condition>-<tag>`` in `spark_dir`."""
    return {"M1g": {m: spark_dir / f"M1g-{SPARK_TAGS[m]}.predictions.jsonl" for m in GATE_MODELS},
            "M2g": {m: {c: spark_dir / f"M2g-{c}-{SPARK_TAGS[m]}.predictions.jsonl" for c in CONDITIONS} for m in GATE_MODELS}}


def v07_checkpoints(args: argparse.Namespace) -> dict[str, Path | None]:
    """The checkpoint each gate model was scored with: v0.7's next to its fetched reports (``<run>-fetched/runs/v07-s<seed>-2b``)."""
    return {"v07s0": Path(args.s0_dir).parent / "runs" / "v07-s0-2b" / "checkpoint.pt", "v07s1": Path(args.s1_dir).parent / "runs" / "v07-s1-2b" / "checkpoint.pt",
            "v06s0": V06_CKPT[0], "v06s1": V06_CKPT[1], "base": None}


# --------------------------------------------------------------------------
# Markdown
# --------------------------------------------------------------------------


def _fmt(value: Any, digits: int = 3, signed: bool = False) -> str:
    if value is None:
        return "—"
    return f"{value:+.{digits}f}" if signed else f"{value:.{digits}f}"


def _diff(value: float | None) -> str:
    return "—" if value is None else f"{value:.1e}"


def _cid(d: Mapping[str, Any] | None, signed: bool = True) -> str:
    if not d or d.get("value") is None:
        return "—"
    return f"{_fmt(d['value'], signed=signed)} [{_fmt(d.get('low'), signed=signed)}, {_fmt(d.get('high'), signed=signed)}]"


def render(result: Mapping[str, Any]) -> str:
    v, r = result["verdict"], result["readings"]
    s = v["S"]
    lines = [f"# v0.7 pre-declared rule — S {s['outcome']} · {v['release_line']}", ""]
    if result.get("pipeline_check"):
        lines += [f"**PIPELINE CHECK ONLY — {result['pipeline_check']}; this is not the v0.7 verdict.**", ""]
    lines += [f"Rule written {RULE['written']}. Applied by `scripts/verdict_v07.py` ({SCRIPT_VERSION}, git {result.get('git')}); {result['n_boot']} paired "
              f"cluster-bootstrap draws (seed {result['seed']}); single order; **seed means** (arm v0.7 = s0 + s1 vs arm v0.6 = s0 + s1).", ""]
    m2 = r["M2g"]
    sa = s["S-a"]
    lines += ["## S-a (amended) — M2g components: gap = DC(sheet) − DC(short), v0.7 − v0.6 (arm means)", "",
              "A component is active iff v0.6's gap ≥ +0.05; active: Δgap ≤ −0.05 with upper < 0 and v0.7's gap ≤ +0.03; inactive: v0.7's gap ≤ +0.03.", "",
              "| component (sheet) | gap v0.6 [95 %] | status | gap v0.7 [95 %] | Δgap [95 %] | Δgap ≤ −0.05 | upper < 0 | gap v0.7 ≤ +0.03 | pass |",
              "|---|---|---|---|---|---|---|---|---|"]
    for name in sa["required"]:
        e = (m2.get("S-a") or {}).get(name) or {}
        c = sa["components"].get(name) or {}
        ch = c.get("checks") or {}
        lines.append(f"| {name} ({SA_COMPONENTS[name]}) | {_cid(e.get('gap_v06'))} | {c.get('status')} | {_cid(e.get('gap_v07'))} | {_cid(e.get('dgap'))} | "
                     f"{ch.get('dgap_le_-0.05', '—')} | {ch.get('dgap_high_lt_0', '—')} | {ch.get('gap_v07_le_+0.03', '—')} | {c.get('pass')} |")
    absent = [n for n in SA_COMPONENTS if n not in sa["required"]]
    if sa["outcome"] == SA_NOT_TESTABLE and sa["inactive_failing"]:
        lines += [f"Reported, not deciding: no component is active, and v0.7 created a gap (> +0.03) on {', '.join(sa['inactive_failing'])}.", ""]
    lines += ["", f"**S-a: {sa['outcome']}** (active: {', '.join(sa['active']) or 'none'}" + (f"; missing: {', '.join(sa['missing'])}" if sa["missing"] else "")
              + (f"; not read here: {', '.join(absent)}" if absent else "") + "; questions per component: " + ", ".join(
        f"{n} {((m2.get('S-a') or {}).get(n) or {}).get('dgap', {}).get('n')}" for n in sa["required"]) + ")", ""]
    m1 = r["M1g"]
    sb = m1.get("S-b") or {}
    c = s["S-b"]["checks"]
    lines += ["## S-b — M1g: drop = acc(N = 8) − acc(N ≥ 17)", "",
              f"{m1.get('items')} items ({m1.get('items_physics')} physics), every listing scored for every model.", "",
              "| reading | value [95 %] | check |", "|---|---|---|",
              f"| v0.7 drop (≤ 0.05) | {_cid(sb.get('drop_v07'))} | {c['drop_v07_le_0.05']} |",
              f"| v0.6 drop | {_cid(sb.get('drop_v06'))} | |",
              f"| Δdrop v0.7 − v0.6 (upper < 0) | {_cid(sb.get('ddrop'))} | {c['ddrop_high_lt_0']} |",
              f"| physics penalty v0.7 − base (< +0.10) | {_cid(sb.get('physics_penalty_minus_base'))} (v0.7 {_fmt((sb.get('physics_penalty_v07') or {}).get('value'))}, "
              f"base {_fmt((sb.get('physics_penalty_base') or {}).get('value'))}) | {c['physics_penalty_minus_base_lt_0.10']} |", "",
              f"**S-b: {'PASS' if s['S-b']['pass'] else 'FAIL'}**", "", f"**S: {s['outcome']}**", ""]
    p = v["parts"]
    lines += ["## Guards", "", "| guard | reading | pass |", "|---|---|---|"]
    st = r["standard"]
    for name in O_SETS:
        d = (st.get("O") or {}).get(name) or {}
        lines.append(f"| O {name} short-form DC v0.7 − v0.6 (≤ +0.02) | {_cid(d)} ({_fmt(d.get('a'))} vs {_fmt(d.get('b'))}) | {p['O']['checks'][f'{name}_dDC_le_+0.02']} |")
    n = st.get("N") or {}
    for key in ("A4_direction", "A4_comparison", "T9_direction", "T9_comparison"):
        d = (n.get("arm") or {}).get(key) or {}
        lines.append(f"| N {key} (v0.7 − v0.6 arms, lower > −0.05) | {_cid(d)} | {p['N']['checks'][f'{key}_low_gt_-0.05']} |")
    for seed, entry in (n.get("home") or {}).items():
        lines.append(f"| N T2–T4 balanced band s{seed} (vs v0.1-mix) | {_cid(entry.get('band_balanced'))} | {p['N']['checks'][f'T2-T4_band_low_gt_-0.05[s{seed}]']} |")
        sev = entry.get("severe_error") or {}
        lines.append(f"| N T2–T4 severe s{seed} (≤ v0.1-mix + 0.01) | {_fmt(sev.get('a'))} vs {_fmt(sev.get('b'))} | {p['N']['checks'][f'T2-T4_severe_le_mix_plus_0.01[s{seed}]']} |")
    for seed, entry in p["R"]["per_seed"].items():
        for name, sets_ in entry["sets"].items():
            for key, d in sets_.items():
                lines.append(f"| R s{seed} {name} {key} (vs v0.1-mix, upper) | {_fmt(d.get('value'), 4, True)} [{_fmt(d.get('low'), 4, True)}, {_fmt(d.get('high'), 4, True)}] | {d['ok']} |")
    g = st.get("G") or {}
    lines.append(f"| G G2 accuracy (v0.7 − v0.6 arms) | {_fmt(g.get('diff'), signed=True)} [{_fmt(g.get('low'), signed=True)}, {_fmt(g.get('high'), signed=True)}] "
                 f"({_fmt(g.get('a'))} vs {_fmt(g.get('b'))}) | {p['G']['pass']} |")
    lines.append(f"| F A4 robust high-called-low (v0.7 arm ≤ base) | v0.7 {_fmt(p['F']['v07_arm'])} vs base {_fmt(p['F']['base'])} (n {p['F']['n']}) | {p['F']['pass']} |")
    ns = v["next_step"]
    lines += ["", "## Release and next step", "", f"- **{v['release_line']}**", f"- Next step ({ns['branch']}): " + " / ".join(ns["next"]) + ".", "",
              "## Secondary (reported, not deciding)", "", "### M2g tone panel (4k; DC by boilerplate tone)", "",
              "| model / arm | DC neutral | DC reassuring | DC alarming | reassuring − neutral [95 %] | alarming − neutral [95 %] |", "|---|---|---|---|---|---|"]
    for m, e in (m2.get("tone_panel") or {}).items():
        if m == "v07_minus_v06":
            continue
        dc = e.get("DC") or {}
        lines.append(f"| {MODELS.get(m, m)} | {_fmt((dc.get('neutral') or {}).get('value'))} | {_fmt((dc.get('reassuring') or {}).get('value'))} | "
                     f"{_fmt((dc.get('alarming') or {}).get('value'))} | {_cid(e.get('reassuring_minus_neutral'))} | {_cid(e.get('alarming_minus_neutral'))} |")
    diff = (m2.get("tone_panel") or {}).get("v07_minus_v06") or {}
    for t in ("reassuring", "alarming"):
        d = (diff.get(f"{t}_minus_neutral") or {}).get("dgap")
        if d:
            lines.append(f"\n{t} − neutral, v0.7 − v0.6: {_cid(d)} (predicted: v0.6 reassuring > neutral, v0.7 flat).")
    amended = m2.get("amendment_secondary") or {}
    if amended:
        lines += ["", "Added by the amendment: the alarming component DC(w4k-alarming) − DC(short) (over-reaction check) and reassuring − neutral at 4k:",
                  "", "| model / arm | alarming − short [95 %] | reassuring − neutral [95 %] |", "|---|---|---|"]
        names = [m for m in (*GATE_MODELS, *ARMS) if m in (amended.get("alarming_minus_short") or {}) or m in (amended.get("reassuring_minus_neutral") or {})]
        for m in names:
            lines.append(f"| {MODELS.get(m, m)} | {_cid((amended.get('alarming_minus_short') or {}).get(m))} | {_cid((amended.get('reassuring_minus_neutral') or {}).get(m))} |")
        for name in ("alarming_minus_short", "reassuring_minus_neutral"):
            d = ((amended.get(name) or {}).get("v07_minus_v06") or {}).get("dgap")
            if d:
                lines.append(f"\n{name.replace('_', ' ')}, v0.7 − v0.6: {_cid(d)}.")
    lines += ["", "### M2g gaps per model (DC wrapped − short) and levels", "", "| model / arm | DC short | gap 2k | gap 4k | gap 8k |", "|---|---|---|---|---|"]
    for m, gaps in (m2.get("gaps") or {}).items():
        short = ((m2.get("levels") or {}).get(m) or {}).get("short") or {}
        lines.append(f"| {MODELS.get(m, m)} | {_fmt(short.get('value'))} | " + " | ".join(_cid(gaps.get(L)) for L in LENGTHS) + " |")
    for name in ("by_set", "by_style"):
        lines += ["", f"Δgap v0.7 − v0.6 {name.replace('_', ' ')} (reported):", "", "| group | 2k | 4k | 8k |", "|---|---|---|---|"]
        for group, entry in (m2.get(name) or {}).items():
            lines.append(f"| {group} | " + " | ".join(_cid((entry.get(L) or {}).get("dgap")) for L in LENGTHS) + " |")
    lines += ["", "### DC vs the untrained base (short form and every M2g condition)", "", "| set / condition | arm | ΔDC vs base [95 %] |", "|---|---|---|"]
    for name, arms in ((st.get("secondary") or {}).get("calls_vs_base") or {}).items():
        for arm, d in arms.items():
            lines.append(f"| {name} (short form) | {arm} | {_cid((d or {}).get('DC'))} |")
    for cond, arms in (m2.get("vs_base") or {}).items():
        for arm, d in arms.items():
            lines.append(f"| M2g {cond} | {arm} | {_cid(d.get('pooled'))} |")
    lines += ["", "### M1g per model", "", "| model / arm | acc N=8 | acc N=21 | acc N=26 | drop [95 %] | A–H | Q–Z | penalty | physics penalty | Q (N=26) | Y (N=26) |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
    for m, e in (m1.get("models") or {}).items():
        weak = ((m1.get("weak_letters") or {}).get(m) or {}).get("26") or {}
        lines.append(f"| {MODELS.get(m, m)} | {_fmt((e.get('acc@8') or {}).get('value'))} | {_fmt((e.get('acc@21') or {}).get('value'))} | "
                     f"{_fmt((e.get('acc@26') or {}).get('value'))} | {_cid(e.get('drop'))} | {_fmt((e.get('bucket:A-H') or {}).get('value'))} | "
                     f"{_fmt((e.get('bucket:Q-Z') or {}).get('value'))} | {_fmt((e.get('penalty') or {}).get('value'), signed=True)} | "
                     f"{_fmt(((m1.get('physics') or {}).get(m) or {}).get('penalty', {}).get('value'), signed=True)} | {_fmt(weak.get('Q'))} | {_fmt(weak.get('Y'))} |")
    lines += ["", "### Call view v0.7 − v0.6 on the short-form sets (B1 / H-set included)", "", "| set | ΔDC | ΔLU | ΔLC | ΔGAP | ΔCA |", "|---|---|---|---|---|---|"]
    for name, d in ((st.get("secondary") or {}).get("calls_v07_minus_v06") or {}).items():
        lines.append(f"| {name} | " + " | ".join(_cid(d.get(k)) for k in ("DC", "LU", "LC", "GAP", "CA")) + " |")
    lines += ["", "Seed spread (|s0 − s1| per arm): " + "; ".join(f"{k} " + ", ".join(f"{m} {_fmt(x)}" for m, x in e.items()) for k, e in
                                                              {**(st.get("seed_spread") or {}), **(r.get("gate_spread") or {})}.items()) + "."]
    lines += ["", "## Notes", ""] + [f"- {note}" for note in result["notes"]] + ["", "## Rule (verbatim parts)", ""] + [f"- **{k}** — {t}" for k, t in RULE.items()]
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------
# Pipeline check (existing files; not the verdict)
# --------------------------------------------------------------------------


def _close(a: Any, b: Any, tol: float = 1e-9) -> bool:
    return a is not None and b is not None and abs(float(a) - float(b)) <= tol


def pipeline_reproduction(m2g: Mapping[str, Any], m1g: Mapping[str, Any], standard: Mapping[str, Any]) -> dict[str, Any]:
    """Every reproduced number next to its reference: task iface's M1 and M2 per model (artifacts/iface/verdict.json) and v0.6's A4 DC per
    model and arm and v0.6 − v0.4 call differences (artifacts/reports/v06/verdict.json). In the pipeline check "v07" is v0.6 and "v06" v0.4."""
    tag = {"v07s0": "v06s0", "v07s1": "v06s1", "v06s0": "v04s0", "v06s1": "v04s1", "base": "base"}
    iface = json.loads((IFACE / "verdict.json").read_text(encoding="utf-8"))
    v06 = json.loads((ARTIFACTS / "reports" / "v06" / "verdict.json").read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = []

    def add(what: str, got: Any, want: Any) -> None:
        rows.append({"what": what, "got": got, "want": want, "diff": None if got is None or want is None else float(got) - float(want), "ok": _close(got, want)})

    ref_m1 = iface["m1"]["readings"]
    for mine, theirs in tag.items():
        x, y = (m1g.get("models") or {}).get(mine) or {}, ref_m1["models"].get(theirs) or {}
        add(f"M1 {theirs} acc@8", (x.get("acc@8") or {}).get("value"), (y.get("acc@8") or {}).get("value"))
        add(f"M1 {theirs} acc@late", (x.get("acc@late") or {}).get("value"), (y.get("acc@late") or {}).get("value"))
        add(f"M1 {theirs} penalty", (x.get("penalty") or {}).get("value"), (y.get("penalty") or {}).get("value"))
        add(f"M1 {theirs} drop (= −late_minus_8)", (x.get("drop") or {}).get("value"), -(y.get("late_minus_8") or {}).get("value", float("nan")))
        add(f"M1 {theirs} drop low (= −high of late_minus_8)", (x.get("drop") or {}).get("low"), -(y.get("late_minus_8") or {}).get("high", float("nan")))
        phys = ref_m1["subsets"]["physics (B1 + H-set)"]["models"].get(theirs) or {}
        add(f"M1 {theirs} physics penalty", ((m1g.get("physics") or {}).get(mine) or {}).get("penalty", {}).get("value"), (phys.get("penalty") or {}).get("value"))
    ref_m2 = iface["m2"]
    for mine, theirs in tag.items():
        for length in LENGTHS:
            got = ((m2g.get("gaps") or {}).get(mine) or {}).get(length) or {}
            want = ((ref_m2["deltas"].get(theirs) or {}).get(length) or {}).get("DC") or {}
            add(f"M2 {theirs} {length} DC delta", got.get("value"), want.get("value"))
            add(f"M2 {theirs} {length} DC delta low", got.get("low"), want.get("low"))
            add(f"M2 {theirs} {length} DC delta high", got.get("high"), want.get("high"))
        for cond in ("short", *LENGTHS):
            got = (((m2g.get("levels") or {}).get(mine) or {}).get("short" if cond == "short" else f"w{cond}") or {}).get("value")
            add(f"M2 {theirs} DC level {cond}", got, (((ref_m2["levels"].get(theirs) or {}).get(cond) or {}).get("DC") or {}).get("value"))
    ctx = v06["readings"]["context"]["A4"]
    for mine, theirs in (("v07s0", "v06s0"), ("v07s1", "v06s1"), ("v07", "v06"), ("base", "base"), ("v06s0", "v04s0"), ("v06s1", "v04s1"), ("v06", "v04")):
        add(f"A4 DC {theirs}", (((standard.get("context") or {}).get("A4") or {}).get(mine) or {}).get("DC", {}).get("value"), (ctx.get(theirs) or {}).get("DC", {}).get("value"))
    for name in O_SETS:
        got = ((standard.get("O") or {}).get(name) or {})
        want = ((v06["readings"]["secondary"]["calls_v06_minus_v04"].get(name) or {}).get("DC") or {})
        add(f"{name} DC v0.6 − v0.4 (the (O) path)", got.get("value"), want.get("value"))
        add(f"{name} DC v0.6 − v0.4 low", got.get("low"), want.get("low"))
    f_ref = v06["verdict"]["parts"]["F"]
    add("A4 robust high-called-low v0.6 arm (the (F) path)", (standard.get("F") or {}).get("a"), f_ref.get("v06_arm"))
    for key in ("A4_direction", "A4_comparison", "T9_direction", "T9_comparison"):
        got = ((standard.get("N") or {}).get("arm") or {}).get(key) or {}
        want = (v06["readings"]["N"]["arm"].get(key) or {})
        add(f"N {key} v0.6 − v0.4 (the (N) path)", got.get("value"), want.get("value"))
    g_ref = v06["readings"]["G"]
    add("G2 v0.6 − v0.4 (the (G) path)", (standard.get("G") or {}).get("diff"), g_ref.get("diff"))
    return {"rows": rows, "all_ok": all(r["ok"] for r in rows), "failing": [r for r in rows if not r["ok"]], "max_abs_diff": max(abs(r["diff"]) for r in rows if r["diff"] is not None)}


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------


def _gate_inputs(args: argparse.Namespace, *, pipeline: bool, problems: list[str], notes: list[str]) -> tuple[dict, dict, dict, dict, dict]:
    """(M2g condition records, M2g predictions, M1g records, M1g predictions, sources)."""
    sources: dict[str, Any] = {}
    if pipeline:
        m2_dir, m1_set, m1_pred = IFACE / "m2", IFACE / "m1" / "m1-set.jsonl", IFACE / "m1" / "pred"
        conditions = {"short": _read(m2_dir / "M2-short.jsonl"), **{f"w{L}": _read(m2_dir / f"M2-w{L}.jsonl") for L in LENGTHS}}
        tag = {"v07s0": "v06s0", "v07s1": "v06s1", "v06s0": "v04s0", "v06s1": "v04s1", "base": "base"}
        m2_preds = {m: {c: _read(m2_dir / "pred" / f"M2-{c}-{t}-2b.predictions.jsonl") for c in conditions} for m, t in tag.items()}
        m1_records = _read(m1_set)
        m1_preds = {m: _read(m1_pred / f"M1-{t}-2b.predictions.jsonl") for m, t in tag.items()}
        notes.append("PIPELINE CHECK: task iface's M1 set and M2 files stand in for M1g and M2g (no tone panel); arms relabelled v0.7 := v0.6, v0.6 := v0.4.")
        return conditions, m2_preds, m1_records, m1_preds, sources
    paths = gate_paths(args.spark_dir)
    checkpoints = v07_checkpoints(args)
    conditions = {c: _read(M2G_DIR / f"M2g-{c}.jsonl") for c in CONDITIONS}
    missing = [str(p) for m in GATE_MODELS for p in [paths["M1g"][m], *paths["M2g"][m].values()] if not p.is_file()]
    if missing:
        raise SystemExit("missing gate prediction files:\n  " + "\n  ".join(missing))
    m2_preds = {}
    for m in GATE_MODELS:
        m2_preds[m] = {}
        for c in CONDITIONS:
            path = paths["M2g"][m][c]
            src = check_gate_source(path, M2G_DIR / f"M2g-{c}.jsonl", M2G_CONDITIONS, checkpoints[m])
            sources[f"{m} M2g {c}"] = src
            problems += [f"{m} M2g {c}: {p}" for p in src["problems"]]
            m2_preds[m][c] = _read(path)
            missing = {str(r["id"]) for r in conditions[c]} - {str(x["id"]) for x in m2_preds[m][c]}
            if missing:
                problems.append(f"{m} M2g {c}: {len(missing)} records without a prediction (first {sorted(missing)[:3]})")
    m1_records = _read(M1G_FILE)
    m1_preds = {}
    for m in GATE_MODELS:
        src = check_gate_source(paths["M1g"][m], M1G_FILE, M1G_CONDITIONS, checkpoints[m])
        sources[f"{m} M1g"] = src
        problems += [f"{m} M1g: {p}" for p in src["problems"]]
        m1_preds[m] = _read(paths["M1g"][m])
        missing = {str(r["id"]) for r in m1_records} - {str(x["id"]) for x in m1_preds[m]}
        if missing:
            problems.append(f"{m} M1g: {len(missing)} listings without a prediction (first {sorted(missing)[:3]})")
    return conditions, m2_preds, m1_records, m1_preds, sources


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--s0-dir", type=Path, default=None, help="fetched reports of the v0.7 seed-0 job (after-v07-s0-2b-*)")
    parser.add_argument("--s1-dir", type=Path, default=None, help="fetched reports of the v0.7 seed-1 job (after-v07-s1-2b-*)")
    parser.add_argument("--spark-dir", type=Path, default=SPARK_DIR, help="the v0.7 chain's Spark predictions (standard sets and gates)")
    parser.add_argument("--a4-compare-map", type=Path, default=None, help="only if the A4 comparison mapping cannot read a scene (as v0.6's)")
    parser.add_argument("--pipeline-check", action="store_true", help="existing files only, arms relabelled; never the verdict")
    parser.add_argument("--no-v07-recheck", action="store_true", help="read the gates without overlap-v07.json (a pipeline check only)")
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--allow-mismatch", action="store_true", help="read even if a prediction file was scored on another set file or conditions")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)
    pipeline = bool(args.pipeline_check)
    out = args.out or (PIPELINE_OUT if pipeline else ARTIFACTS / "reports" / "v07" / "verdict.json")
    refuse_write(out)
    if not pipeline and (args.s0_dir is None or args.s1_dir is None):
        raise SystemExit("--s0-dir and --s1-dir are required for the verdict (or pass --pipeline-check)")
    if not pipeline and out.resolve() == PIPELINE_OUT.resolve():
        raise SystemExit("the verdict is not written into the pipeline-check folder")
    notes: list[str] = []
    problems: list[str] = []
    exclude_m1: set[str] = set()
    exclude_m2: set[str] = set()
    if not pipeline:
        if OVERLAP_V07.is_file():
            recheck = json.loads(OVERLAP_V07.read_text(encoding="utf-8"))
            exclude_m1 = set(recheck.get("exclude", {}).get("M1g_items") or [])
            exclude_m2 = set(recheck.get("exclude", {}).get("M2g_ids") or [])
            notes.append(f"overlap-v07.json (train-v07 sha256 {str(recheck.get('train_v07', {}).get('sha256'))[:12]}): {len(exclude_m1)} M1g items and "
                         f"{len(exclude_m2)} M2g moments left out; M2g filler shingles found in train-v07: {recheck.get('filler_shingles_found')} of "
                         f"{recheck.get('filler_shingles')}.")
        elif not args.no_v07_recheck:
            raise SystemExit(f"{OVERLAP_V07} is missing — run scripts/v07_gates_build.py recheck-v07 on v0.7's training input first")
        else:
            notes.append("overlap-v07.json absent: no exclusions (--no-v07-recheck)")
    conditions, m2_preds, m1_records, m1_preds, gate_sources = _gate_inputs(args, pipeline=pipeline, problems=problems, notes=notes)
    paths = standard_paths(args, relabel={"v07": "v06", "v06": "v04"} if pipeline else None)
    missing = [f"{m} {s}: {paths[(m, s)]}" for m, s in required_standard() if not paths[(m, s)].is_file()]
    if missing:
        raise SystemExit("missing prediction files the rule reads:\n  " + "\n  ".join(missing))
    sets, preds, sources = _load_standard(paths, _set_paths(), allow_mismatch=args.allow_mismatch, notes=notes)
    if problems and not args.allow_mismatch:
        raise SystemExit("gate prediction files fail their checks:\n  " + "\n  ".join(problems[:30]))
    override = json.loads(args.a4_compare_map.read_text(encoding="utf-8")) if args.a4_compare_map else None
    _log("M2g readings")
    m2g = m2g_readings(conditions, m2_preds, n_boot=args.n_boot, seed=args.seed, exclude=exclude_m2)
    _log("M1g readings")
    m1g = m1g_readings(m1_records, m1_preds, n_boot=args.n_boot, seed=args.seed, exclude=exclude_m1)
    _log("standard readings")
    standard = standard_readings(sets, preds, n_boot=args.n_boot, seed=args.seed, compare_override=override)
    readings = {"M2g": m2g, "M1g": m1g, "standard": standard, "gate_spread": gate_spread(m2g, m1g)}
    sa_required = tuple(SA_COMPONENTS) if not pipeline else tuple(n for n in SA_COMPONENTS if n != "cue")
    verdict = decide(assemble(m2g, m1g, standard), sa_required=sa_required)
    pipeline_note = None
    if pipeline:
        pipeline_note = "existing files with the arms relabelled (v0.7 := v0.6, v0.6 := v0.4); iface M1/M2 stand in for M1g/M2g"
        readings["reproduction"] = pipeline_reproduction(m2g, m1g, standard)
        notes.insert(0, "PIPELINE CHECK ONLY — this is not the v0.7 verdict.")
        notes.insert(1, "PIPELINE CHECK: iface's M2 has no tone panel, so the amended S-a is read on len2k, len4k and len8k only (the cue component "
                        "needs w4k-reassuring); the real verdict reads all four components.")
    notes += ["Seed means: an arm's item = the mean of its two seeds' item values (call distributions: each seed's listings weighted 1/L), on the "
              "items both seeds answered; arm comparisons pair the questions both arms have; Δgap reads the four sources (each arm wrapped and short) "
              "on the questions all four answered, with one cluster bootstrap.",
              "M2g and M1g were scored on Spark (GB10) with the v0.7 chain (artifacts/scratch/v07b/score_spark.sh): M2g on the standard diagnosis path "
              "with seeded orders (one listing per question, the same in every condition), M1g on task iface's many-option path (the 2-8 cap lifted "
              "only in scripts/iface_m1_score.py). A4, T9, T8b, B1, H-set on Spark; T1-T7, G2, T8a on the H100 pods.",
              "(R) reads compare_runs (single order, record bootstrap) per v0.7 seed vs v0.1-mix; (G) is single_probe._paired_cell on the arms' mean "
              "question values; (F) is read on the robust A4 items the v0.7 arm and the base both answered."]
    result = {"kind": "v07-verdict" if not pipeline else "v07-verdict-pipeline-check", "script_version": SCRIPT_VERSION, "git": _git_sha(),
              "created": dt.datetime.now().astimezone().isoformat(timespec="seconds"), "rule_text": RULE, "n_boot": args.n_boot, "seed": args.seed,
              "pipeline_check": pipeline_note, "verdict": verdict, "readings": readings, "sources": {**sources, **gate_sources}, "problems": problems,
              "notes": notes, "inputs": {k: str(v) for k, v in vars(args).items() if isinstance(v, Path)}}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    md = render(result)
    if pipeline:
        rep = readings["reproduction"]
        md += "\n## Pipeline reproduction\n\n" + f"All reproduced: {rep['all_ok']} (max |diff| {rep['max_abs_diff']:.2e}).\n\n| reading | got | reference | diff |\n|---|---|---|---|\n"
        md += "".join(f"| {r['what']} | {_fmt(r['got'], 4)} | {_fmt(r['want'], 4)} | {_diff(r['diff'])} |\n" for r in rep["rows"])
    out.with_suffix(".md").write_text(md, encoding="utf-8")
    print(md.split("## Secondary")[0])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
