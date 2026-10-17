#!/usr/bin/env python
"""Apply the PRE-DECLARED v0.9 VERDICT RULE (progress.md, written 2026-10-17 ~07:56 KST, before any v0.9 data existed) mechanically.

    python scripts/verdict_v09.py --c-s0-dir <C seed-0 run>-fetched/reports --c-s1-dir … --cn-s0-dir … --cn-s1-dir …
        # → artifacts/reports/v09/verdict.{json,md} (and a16s-curves.svg)
    python scripts/verdict_v09.py --pipeline-check     # existing v0.8 files only; not the verdict (last part below)

**Arms and seed means.** C = v0.9C-s0 + v0.9C-s1 (train-v08 + the order-consistency loss), CN = v0.9CN-s0 + v0.9CN-s1 (C + the
measurement-literacy records), F = v0.8F-s0 + v0.8F-s1 (the reference; already scored — its v0.8 Spark files are reused wherever the same set
and conditions exist, and only N9g is new). An arm's per-item value is the mean of its two seeds' values on the items both seeds have (call
view: :func:`verdict_v06.arm_calls`); arm comparisons pair the items both arms have. Paired cluster bootstrap, 2,000 draws, seed 0
(:func:`robo_jev.bench.system1._draws`). Inclusive bounds (≤, ≥) compare with a 1e-12 tolerance, so float rounding of an exact tie never flips
them; strict bounds (<, >) compare as they are. A missing reading fails its check.

* **P-num1 (CN − C, gate A16s)**: accuracy (the top option = the question's rule option) on the 324 x / y / z questions, single order seed 1
  under serve's conditions on v0.8's path (:func:`robo_jev.bench.a16s_v08.sweep_rows`, reading ``single-s1``); unit = an axis question,
  cluster = its A16s request. PASS iff Δ ≥ +0.25 AND its lower bound > 0 AND CN's accuracy in the 1–8 cm band (A16s bands ``1-3`` and
  ``3-8``, i.e. 1 ≤ |d| < 8 cm; both signs, both number formats, the three axes pooled) ≥ 0.60 (point).
* **P-num2 (CN − C, gate N9g)**: accuracy (the top option of reading ``single-s1``, ties to the record's earlier key, is the key;
  :func:`robo_jev.bench.n9g_v09.n9g_rows`); unit and cluster = the record (one question each). PASS iff Δ ≥ +0.20 AND its lower bound > 0.
* **H-num**: SUPPORTED iff P-num1 and P-num2; PARTIAL iff exactly one; REJECTED iff neither.
* **P-cons (C − F)**: the flip rate between two order seeds, the mean of five sources with equal weight — the M8g bins 2–10, 11–26, 27–60
  and 61–255 (listing-0 records, the record order against the package's seeded order seed 2; a flip = a different top option — v0.8's
  :func:`verdict_v08.flip_values`) and B8 (every tagged question, single order seed 1 against seed 2; v0.8's flip definition). Per source Δ =
  C − F on the units both arms have. ONE draw-wise bootstrap: the M8g items are resampled jointly for the four bins (cluster = the M8g item,
  draw seed = seed), the B8 requests separately (cluster = the request, draw seed = seed + 1); per draw the five Δ are averaged. PASS iff the
  mean ≤ −0.10 AND its upper bound < 0.
* **H-cons**: SUPPORTED iff P-cons AND M8g accuracy C − F ≥ −0.02 (point; every listing in the record order, on the listings both arms'
  seeds scored) AND M8g accuracy at N ≤ 26 (the N = 6 and N = 18 listings) C − F ≥ −0.02 (point); otherwise NOT SUPPORTED.
* **GUARDS** — for CN − C and for C − F, reported with pass / fail; the rule gives them no effect on the outcome labels: outside DC (the mean
  of A4, T9 and T8b with B2's truth classes; each set drawn by its own pair clusters with draw seed = seed + its index, averaged per draw —
  v0.8's :func:`verdict_v08.mean_of_sets`) ≤ +0.02 (point); outside CA (the same mean) ≥ −0.05 (point); T2–T4 balanced band (pods; stratified
  paired bootstrap, arm means) lower bound > −0.05, and severe errors a ≤ b + 0.01 (arm means); G2 (:func:`verdict_v06.g2_arm_comparison`)
  ≥ −0.02 (point) with lower bound > −0.045.
* **NEXT STEPS** as pre-declared: H-num REJECTED; H-num PARTIAL with N9g passing and A16s failing (A16s = P-num1 as a whole); H-num PARTIAL
  with A16s passing and N9g failing; H-num SUPPORTED (CN is the release candidate — the user decides; A16 is compared with v0.7: CN's and
  v0.7's A16 rule agreement are printed); H-cons NOT SUPPORTED. **RELEASE**: the user decides.

**Reported, never deciding**: A16 agreement and the stay rate per axis and band (C, CN, F, v0.7, the base; v0.8's
:func:`verdict_v08.a16_readings`); the untrained base on N9g; N9g per family, domain (and seen / unseen domain group), number format, wording,
near-boundary kind and option count (:func:`robo_jev.bench.n9g_v09.n9g_summary`), per model and arm, with CN − C and C − F per family;
R14 and S16 flips (serve's three orders); Jev agreement on the comparison set; M8g accuracy per bin and per source; ECE (M8g per bin, N9g
pooled and per family; an arm's ECE = the mean of its seeds'); A16s per band for every model and arm (+ the curves, ``a16s-curves.svg``);
B8 flips per model and arm; the short-form call context per set (H-set and B1 included).

**Reuse.** v0.8's helper (``scripts/verdict_v08.py``) is loaded under a private module name (:func:`V8`) whose globals ``ARMS``, ``MODELS``,
``SPARK_TAGS`` and ``LABELS`` are set to v0.9's, so its arm-aware readings (calls, flips, M8g per bin / per source, R14, S16, A16, A16s,
Jev) read v0.9's arms; the real ``verdict_v08`` module is never touched.

**Exclusions**: ``artifacts/v08/gates/overlap-v08.json`` (recheck-v08 on train-v08 — C's and F's input) and ``artifacts/v09/overlap-v09.json``
(recheck-v09 on train-v09n — CN's input); every comparison leaves out the union. The verdict refuses without either (``--no-v09-recheck`` only
for tests and pipeline checks) and checks that the C runs trained on the input overlap-v08.json read and the CN runs on the one
overlap-v09.json read (``metrics.json`` → ``manifest.data.file_sha256``).

**Refusals.** N9g's, A16s's and B8's frozen files must be unchanged (no override). A prediction or score file that fails its checks (set,
scored file, readings, frozen record, checkpoint, prompt digest) is refused unless ``--allow-mismatch``; so is a rule comparison across bundle
versions (B8: F's reused v0.8 files against C's; A16s and N9g: C against CN must carry the same ``bundle_sha256``) or conditions (A16s: C's
and CN's files scored with serve's adapter, as P-num1 says; N9g: C and CN alike; the same order key). A model the rule does not read whose
file fails its checks is left out and listed (v0.8's handling of reported sets).

**Pipeline check** (``--pipeline-check``; writes ``artifacts/scratch/v09b/verdict/``, says it is not the verdict): from the existing v0.8
files, with stand-in arms, it reproduces v0.8's own numbers (``artifacts/reports/v08/verdict.json``): A16s — F's and FP's accuracy in total
and in every band cell, and the P-num1 path with C := F, CN := FP (Δ = FP − F; CN's 1–8 cm accuracy against the count-weighted mean of v0.8's
1–3 and 3–8 cells); M8g and B8 flips — F's per-bin flip rate, FP's, the B8 pooled flips of both, and the P-cons path with C := FP, F := F
(each arm's five-source mean against the mean of v0.8's five numbers; Δ with its interval printed); the C − F guard path with C := FP — the
outside DC mean (v0.8's P1, with its interval and per set), the CA mean, the T2–T4 band and severe errors and G2 against v0.8's guards.
"""

from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import functools
import importlib.util
import json
import math
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from robo_jev.bench import gates_v08 as G8  # noqa: E402
from robo_jev.bench import n9g_v09 as N9  # noqa: E402

SCRIPT_VERSION = "v09-verdict-1"
RULE = {
    "written": "2026-10-17 ~07:56 KST, BEFORE any v0.9 data exists (progress.md: PRE-DECLARED v0.9 VERDICT RULE); user: '좋아 추천대로 진행해' "
               "— plan v09-plan.md, phase cap -> $650, training consent for four runs",
    "REFERENCE": "v0.8 F (runs v08f-s0-01 / v08f-s1-01, already scored).",
    "ARMS": "(seeds 0 and 1 each; v0.8's recipe otherwise; seed means; paired cluster bootstrap 2000 draws seed 0): C = train-v08.jsonl "
            "(unchanged, sha bffd7494…) + an ORDER-CONSISTENCY LOSS. The loss: JS divergence between the model's distributions over option "
            "contents across permuted copies of a choice question (one cyclic shift and the reverse). Settings: λ = 0.5, on 30 % of choice "
            "questions, drawn by a fixed seed. The share may be lowered only to fit the budget; the final value is recorded before training. "
            "CN = C + MEASUREMENT-LITERACY records, replacing an equal number of general-mix records (same total; target 6–8 % of records). "
            "Families: threshold / step (sign plus ordered thresholds → category); interval membership; multi-constraint selection (4–20 "
            "options); comparison / ordering; unit conversion inside rules. Domains: distance, speed, angle, time, mass, temperature, percent. "
            "Mixed number formats, dense near thresholds, and wording variants that put option words into descriptions. robo-jev-web's arm "
            "template (buildServoQuestions) is NOT used.",
    "GATES": "N9g (new, sealed): a numeric-rule gate built by a separate author who never sees the generator. Unseen templates and domains, "
             "covering the five families, with multi-constraint selection at 4–20 options. Labels are exact by construction. From v0.8: A16s "
             "(the arm sweep), A16 (bench arm requests), M8g and B8. R14 is reported only.",
    "H-num (CN − C)": "P-num1: A16s accuracy (single order seed 1, serve's conditions, v0.8's path) ≥ +0.25 with lower bound > 0, AND CN's "
                      "accuracy in the 1–8 cm band (both signs pooled) ≥ 0.60 (point). P-num2: N9g accuracy ≥ +0.20 with lower bound > 0. "
                      "Outcome: SUPPORTED iff P-num1 and P-num2; PARTIAL iff exactly one; REJECTED iff neither.",
    "H-cons (C − F)": "P-cons: the flip rate between two order seeds, as the mean of the four M8g bins and B8 (five sources, equal weight), "
                      "≤ −0.10 with upper bound < 0. AND M8g accuracy ≥ −0.02 (point), AND M8g accuracy at N ≤ 26 (N 6 and 18) ≥ −0.02 "
                      "(point). Outcome: SUPPORTED iff all three hold; otherwise NOT SUPPORTED.",
    "GUARDS (for CN − C and C − F, each reported)": "outside DC (A4 / T9 / T8b-B2 mean) ≤ +0.02 (point); outside CA (mean) ≥ −0.05 (point); "
                                                    "T2–T4 balanced band lower bound > −0.05, and severe errors ≤ +0.01 (arm means); G2 ≥ "
                                                    "−0.02 (point) with lower bound > −0.045.",
    "REPORTED, NOT DECIDING": "A16 agreement and stay rate per axis and band; the untrained base on N9g; N9g per family, domain and format; "
                              "R14 and S16 flips; Jev agreement on the comparison set; M8g accuracy per bin and per source; ECE.",
    "PRE-DECLARED NEXT STEPS": "H-num REJECTED: stated numeric rules are not learnable at 2B in one pass with this data. Next: a 4B backbone "
                               "trial on the same data, or System 0 computing such steps. H-num PARTIAL with N9g passing and A16s failing: a "
                               "template-transfer gap. Next: paraphrased arm-style rules. H-num PARTIAL with A16s passing and N9g failing: "
                               "narrow learning. Next: broader families. H-num SUPPORTED: CN is the release candidate (the user decides; A16 "
                               "is compared with v0.7). Next: the label prior through data balancing, not loss weights. H-cons NOT SUPPORTED: "
                               "a larger share or λ, or test-time order averaging for bench questions.",
    "RELEASE": "the user decides.",
}

# --- the rule's bounds (never changed) ---
PNUM1_DELTA, PNUM1_LOW, PNUM1_BAND = 0.25, 0.0, 0.60
PNUM2_DELTA, PNUM2_LOW = 0.20, 0.0
PCONS_DELTA, PCONS_HIGH = -0.10, 0.0
ACC_HOLD = -0.02
DC_POINT, CA_POINT = 0.02, -0.05
BAND_LOW, SEVERE_SLACK = -0.05, 0.01
G_POINT, G_LOW = -0.02, -0.045
#: Inclusive bounds are compared with this tolerance (module docstring).
TOL = 1e-12

SEEDS = (0, 1)
V09_MODELS = ("v09Cs0", "v09Cs1", "v09CNs0", "v09CNs1")
F_MODELS = ("v08Fs0", "v08Fs1")
#: The rule's arms (C, CN; F the reference) and v0.7 (reported references only: A16 is compared with v0.7).
ARMS = {"C": ("v09Cs0", "v09Cs1"), "CN": ("v09CNs0", "v09CNs1"), "F": ("v08Fs0", "v08Fs1"), "v07": ("v07s0", "v07s1")}
RULE_ARMS = ("C", "CN", "F")
MODELS = (*V09_MODELS, *F_MODELS, "v07s0", "v07s1", "base")
LABELS = {"v09Cs0": "v0.9C seed 0", "v09Cs1": "v0.9C seed 1", "v09CNs0": "v0.9CN seed 0", "v09CNs1": "v0.9CN seed 1", "v08Fs0": "v0.8F seed 0",
          "v08Fs1": "v0.8F seed 1", "v07s0": "v0.7 seed 0", "v07s1": "v0.7 seed 1", "base": "untrained Qwen3.5-2B", "C": "C (arm, seed mean)",
          "CN": "CN (arm, seed mean)", "F": "F (v0.8, arm, seed mean)", "v07": "v0.7 (arm, seed mean)"}
SPARK_TAGS = {m: f"{m}-2b" for m in MODELS}
#: The runs whose pods (T2–T4, G2) the guards read.
POD_MODELS = (*V09_MODELS, *F_MODELS)
POD_ARGS = {"v09Cs0": "c_s0_dir", "v09Cs1": "c_s1_dir", "v09CNs0": "cn_s0_dir", "v09CNs1": "cn_s1_dir", "v08Fs0": "f_s0_dir", "v08Fs1": "f_s1_dir"}
OUTSIDE_SETS = ("A4", "T9", "T8b")
CALL_SETS = ("A4", "T9", "T8b", "Hset", "B1")
HOME = ("T2", "T3", "T4")
POD_SETS = ("T2", "T3", "T4", "G2")
BIN_NAMES = tuple(name for name, _, _ in G8.BINS)
#: P-cons's five sources, equal weight.
CONS_SOURCES = tuple(f"M8g {name}" for name in BIN_NAMES) + ("B8",)
A16S = "A16s"
A16S_READINGS = ("single-s1", "single-s2")
#: P-num1's band: 1 ≤ |d| < 8 cm = the A16s reading bands 1–3 and 3–8.
A16S_BAND_1_8 = ("1-3", "3-8")
N9G = "N9g"
B8_READINGS = ("balanced-s1", "single-s1", "single-s2")
R14_READINGS = ("spaces-0", "spaces-1", "spaces-2")
R14_INPUT = "R14-v08-sample.jsonl"
A16, A16Y, S16 = "A16", "A16y", "S16"
R16_FILES = {A16: "A16.jsonl", A16Y: "A16-y-ablation.jsonl", S16: "S16.jsonl"}
R16_READINGS = {A16: ("single-s1",), A16Y: ("single-s1",), S16: R14_READINGS}
M8G_KINDS = {"order": "M8g", "seed2": "M8g-seed2"}
A16S_SVG = "a16s-curves.svg"
#: The A16s figure's series slots (a16s_v08.curves_svg draws the slots F, FP, v07, base): C, CN, the v0.8 F reference and the base.
A16S_FIGURE_SLOTS = {"F": ("C", "v0.9 C (consistency loss)"), "FP": ("CN", "v0.9 CN (+ measurement literacy)"),
                     "v07": ("F", "v0.8 F (reference)"), "base": ("base", "untrained base")}

ARTIFACTS = REPO / "artifacts"
SPARK_DIR = ARTIFACTS / "reports" / "v09" / "spark"
V08_SPARK = ARTIFACTS / "reports" / "v08" / "spark"
V08_VERDICT = ARTIFACTS / "reports" / "v08" / "verdict.json"
F_PODS = {0: ARTIFACTS / "cloud" / "runs" / "v08f-s0-01-fetched" / "reports", 1: ARTIFACTS / "cloud" / "runs" / "v08f-s1-01-fetched" / "reports"}
V07_CKPT = {0: ARTIFACTS / "cloud" / "runs" / "v07-s0-01-fetched" / "runs" / "v07-s0-2b" / "checkpoint.pt",
            1: ARTIFACTS / "cloud" / "runs" / "v07-s1-01-fetched" / "runs" / "v07-s1-2b" / "checkpoint.pt"}
OUT = ARTIFACTS / "reports" / "v09" / "verdict.json"
PIPELINE_OUT = ARTIFACTS / "scratch" / "v09b" / "verdict" / "verdict-pipeline.json"
#: The gates' files (module constants so tests can point them elsewhere).
N9G_DIR = N9.N9G_DIR
OVERLAP_V08 = G8.OVERLAP_V08
OVERLAP_V09 = N9.OVERLAP_V09
#: Outputs never go here.
NEVER_WRITE = (ARTIFACTS / "v08", ARTIFACTS / "v09" / "n9g", ARTIFACTS / "simgen", ARTIFACTS / "t8", ARTIFACTS / "t9", ARTIFACTS / "b1", ARTIFACTS / "b2",
               ARTIFACTS / "v06", ARTIFACTS / "v07", ARTIFACTS / "iface", ARTIFACTS / "reports" / "v04", ARTIFACTS / "reports" / "v05",
               ARTIFACTS / "reports" / "v06", ARTIFACTS / "reports" / "v07", ARTIFACTS / "reports" / "v08", ARTIFACTS / "cloud")


@functools.lru_cache(maxsize=None)
def V8():
    """scripts/verdict_v08.py under a PRIVATE module name, its globals ``ARMS`` / ``MODELS`` / ``SPARK_TAGS`` / ``LABELS`` set to v0.9's (module
    docstring, Reuse). Its functions read those globals at call time, so its arm-aware readings aggregate v0.9's arms; the module that other
    scripts and tests import as ``verdict_v08`` is a separate instance and keeps v0.8's values."""
    spec = importlib.util.spec_from_file_location("v09_verdict_v08", REPO / "scripts" / "verdict_v08.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module.ARMS = dict(ARMS)
    module.MODELS = tuple(MODELS)
    module.SPARK_TAGS = dict(SPARK_TAGS)
    module.LABELS = dict(LABELS)
    return module


def V4():
    return V8().V4()


def V5():
    return V8().V5()


def V6():
    return V8().V6()


def V7():
    return V8().V7()


def _log(message: str) -> None:
    print(f"[verdict-v09 {dt.datetime.now().strftime('%H:%M:%S')}] {message}", file=sys.stderr, flush=True)


def _git_sha() -> str | None:
    try:
        return subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True).stdout.strip() or None
    except (OSError, subprocess.CalledProcessError):
        return None


# --------------------------------------------------------------------------
# The rule (pure — readings in, verdict out)
# --------------------------------------------------------------------------


def _num(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def _le(value: Any, bound: float) -> bool:
    value = _num(value)
    return value is not None and value <= bound + TOL


def _ge(value: Any, bound: float) -> bool:
    value = _num(value)
    return value is not None and value >= bound - TOL


def _lt(value: Any, bound: float) -> bool:
    value = _num(value)
    return value is not None and value < bound


def _gt(value: Any, bound: float) -> bool:
    value = _num(value)
    return value is not None and value > bound


def pnum1_check(r: Mapping[str, Any] | None) -> dict[str, Any]:
    """P-num1: A16s accuracy CN − C ≥ +0.25 (point) AND its lower bound > 0 AND CN's 1–8 cm accuracy ≥ 0.60 (point)."""
    r = r or {}
    delta, band = r.get("delta") or {}, r.get("band_CN") or {}
    checks = {"A16s_CN_minus_C_ge_+0.25": _ge(delta.get("value"), PNUM1_DELTA), "A16s_low_gt_0": _gt(delta.get("low"), PNUM1_LOW),
              "CN_1-8cm_accuracy_ge_0.60": _ge(band.get("value"), PNUM1_BAND)}
    return {"value": _num(delta.get("value")), "low": _num(delta.get("low")), "band_CN": _num(band.get("value")), "checks": checks,
            "pass": all(checks.values())}


def pnum2_check(r: Mapping[str, Any] | None) -> dict[str, Any]:
    """P-num2: N9g accuracy CN − C ≥ +0.20 (point) AND its lower bound > 0."""
    r = r or {}
    checks = {"N9g_CN_minus_C_ge_+0.20": _ge(r.get("value"), PNUM2_DELTA), "N9g_low_gt_0": _gt(r.get("low"), PNUM2_LOW)}
    return {"value": _num(r.get("value")), "low": _num(r.get("low")), "checks": checks, "pass": all(checks.values())}


def hnum_outcome(p1_pass: bool, p2_pass: bool) -> str:
    """SUPPORTED iff P-num1 and P-num2; PARTIAL iff exactly one; REJECTED iff neither."""
    return {2: "SUPPORTED", 1: "PARTIAL", 0: "REJECTED"}[int(bool(p1_pass)) + int(bool(p2_pass))]


def pcons_check(r: Mapping[str, Any] | None) -> dict[str, Any]:
    """P-cons: the five-source mean flip rate C − F ≤ −0.10 (point) AND its upper bound < 0."""
    r = r or {}
    checks = {"flip_mean_le_-0.10": _le(r.get("value"), PCONS_DELTA), "flip_mean_high_lt_0": _lt(r.get("high"), PCONS_HIGH)}
    return {"value": _num(r.get("value")), "high": _num(r.get("high")), "checks": checks, "pass": all(checks.values())}


def accuracy_holds(acc: Mapping[str, Any] | None, acc26: Mapping[str, Any] | None) -> dict[str, Any]:
    """The accuracy holds of H-cons: M8g accuracy C − F ≥ −0.02 (point) AND M8g accuracy at N ≤ 26 (N 6 and 18) C − F ≥ −0.02 (point)."""
    checks = {"M8g_accuracy_ge_-0.02": _ge((acc or {}).get("value"), ACC_HOLD), "M8g_accuracy_N<=26_ge_-0.02": _ge((acc26 or {}).get("value"), ACC_HOLD)}
    return {"accuracy": _num((acc or {}).get("value")), "accuracy_n26": _num((acc26 or {}).get("value")), "checks": checks, "pass": all(checks.values())}


def hcons_outcome(pcons_pass: bool, holds_pass: bool) -> str:
    """SUPPORTED iff all three hold (P-cons and both accuracy holds); otherwise NOT SUPPORTED."""
    return "SUPPORTED" if pcons_pass and holds_pass else "NOT SUPPORTED"


def guard_checks(g: Mapping[str, Any] | None) -> dict[str, Any]:
    """One comparison's guards (a − b): outside DC mean ≤ +0.02 (point); outside CA mean ≥ −0.05 (point); T2–T4 balanced band lower bound >
    −0.05 and severe errors a ≤ b + 0.01 (arm means); G2 ≥ −0.02 (point) with lower bound > −0.045. A missing reading fails. They do not change
    the outcome labels."""
    g = g or {}
    dc, ca, band = g.get("DC") or {}, g.get("CA") or {}, g.get("band") or {}
    severe, g2 = g.get("severe") or {}, g.get("G2") or {}
    sa, sb = _num(severe.get("a")), _num(severe.get("b"))
    checks = {"outside_DC_mean_le_+0.02": _le(dc.get("value"), DC_POINT), "outside_CA_mean_ge_-0.05": _ge(ca.get("value"), CA_POINT),
              "T2-T4_band_low_gt_-0.05": _gt(band.get("low"), BAND_LOW),
              "T2-T4_severe_a_le_b_plus_0.01": sa is not None and sb is not None and sa <= sb + SEVERE_SLACK + TOL,
              "G2_point_ge_-0.02": _ge(g2.get("diff"), G_POINT), "G2_low_gt_-0.045": _gt(g2.get("low"), G_LOW)}
    return {"checks": checks, "pass": all(checks.values())}


NEXT = {
    "H-num REJECTED": "stated numeric rules are not learnable at 2B in one pass with this data. Next: a 4B backbone trial on the same data, or "
                      "System 0 computing such steps",
    "H-num PARTIAL with N9g passing and A16s failing": "a template-transfer gap. Next: paraphrased arm-style rules",
    "H-num PARTIAL with A16s passing and N9g failing": "narrow learning. Next: broader families",
    "H-num SUPPORTED": "CN is the release candidate (the user decides; A16 is compared with v0.7). Next: the label prior through data balancing, "
                       "not loss weights",
    "H-cons NOT SUPPORTED": "a larger share or λ, or test-time order averaging for bench questions",
}
RELEASE_LINE = "RELEASE: the user decides."


def next_steps(hnum: str, p1_pass: bool, p2_pass: bool, hcons_supported: bool, a16_vs_v07: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """The pre-declared next steps: one H-num branch (P-num1 stands for "A16s"), and the H-cons branch when H-cons is NOT SUPPORTED (no step
    is pre-declared for H-cons SUPPORTED). Under H-num SUPPORTED the A16 comparison with v0.7 is printed with it."""
    branches: list[str] = []
    if hnum == "SUPPORTED":
        branches.append("H-num SUPPORTED")
    elif hnum == "REJECTED":
        branches.append("H-num REJECTED")
    elif p2_pass and not p1_pass:
        branches.append("H-num PARTIAL with N9g passing and A16s failing")
    else:
        branches.append("H-num PARTIAL with A16s passing and N9g failing")
    if not hcons_supported:
        branches.append("H-cons NOT SUPPORTED")
    notes = []
    if hnum == "SUPPORTED":
        cmp_ = a16_vs_v07 or {}
        notes.append("A16 compared with v0.7 (rule agreement, single order seed 1, serve's conditions): "
                     + (f"CN {cmp_['CN']:.3f} vs v0.7 {cmp_['v07']:.3f} (CN − v0.7 {cmp_['CN'] - cmp_['v07']:+.3f}); the stored v0.7 endpoint answers "
                        f"{cmp_.get('v07_stored') if cmp_.get('v07_stored') is None else format(cmp_['v07_stored'], '.3f')}"
                        if cmp_.get("CN") is not None and cmp_.get("v07") is not None else "unread (A16 for CN or v0.7 is missing)"))
    if hcons_supported:
        notes.append("H-cons SUPPORTED: no next step is pre-declared for it")
    return {"branches": branches, "next": [NEXT[b] for b in branches], "notes": notes}


def decide(readings: Mapping[str, Any]) -> dict[str, Any]:
    """The whole rule on computed readings: ``P-num1`` ({delta, band_CN}), ``P-num2``, ``P-cons``, ``M8g accuracy``, ``M8g accuracy N<=26``,
    ``guards`` ({"CN-C": …, "C-F": …}) and ``A16 vs v0.7`` (printed under H-num SUPPORTED)."""
    p1, p2 = pnum1_check(readings.get("P-num1")), pnum2_check(readings.get("P-num2"))
    hnum = hnum_outcome(p1["pass"], p2["pass"])
    pc = pcons_check(readings.get("P-cons"))
    holds = accuracy_holds(readings.get("M8g accuracy"), readings.get("M8g accuracy N<=26"))
    hcons = hcons_outcome(pc["pass"], holds["pass"])
    guards = {name: guard_checks((readings.get("guards") or {}).get(name)) for name in ("CN-C", "C-F")}
    return {"H-num": hnum, "H-cons": hcons, "P-num1": p1, "P-num2": p2, "P-cons": pc, "accuracy_holds": holds, "guards": guards,
            "next_steps": next_steps(hnum, p1["pass"], p2["pass"], hcons == "SUPPORTED", readings.get("A16 vs v0.7")),
            "release_line": RELEASE_LINE, "guards_note": "the guards are reported with pass / fail; the rule does not let them change the outcome labels"}


# --------------------------------------------------------------------------
# Decision readings
# --------------------------------------------------------------------------


def _arm_of(values: Mapping[str, Any], arm: str, arms: Mapping[str, tuple[str, str]], merge) -> Any:
    first, second = arms[arm]
    if first in values and second in values:
        return merge(values[first], values[second])
    return None


def pnum1_reading(records: Sequence[Mapping[str, Any]], scores: Mapping[str, Mapping[str, Any]], *, arms: Mapping[str, tuple[str, str]] = ARMS,
                  a: str = "CN", b: str = "C", n_boot: int = 2000, seed: int = 0) -> dict[str, Any]:
    """P-num1's readings: A16s accuracy a − b (reading ``single-s1``; unit = (request, axis), cluster = the request; seed means) and each arm's
    accuracy in the 1–8 cm band (bands 1–3 and 3–8; both signs, formats and axes pooled) — `a`'s is the rule's."""
    from robo_jev.bench import a16s_v08 as S

    rows = {m: S.sweep_rows(records, (s or {}).get("answers") or {}, reading=A16S_READINGS[0], second=A16S_READINGS[1]) for m, s in scores.items()}
    arm_rows = {arm: S.seed_mean_rows(rows[x], rows[y]) for arm, (x, y) in arms.items() if x in rows and y in rows}
    out: dict[str, Any] = {"records": len(records), "arms": sorted(arm_rows), "missing": [x for x in (a, b) if x not in arm_rows]}
    band: dict[str, Any] = {}
    for arm, chosen in arm_rows.items():
        in_band = [float(r["correct"]) for r in chosen if r["band"] in A16S_BAND_1_8]
        band[arm] = {"value": float(np.mean(in_band)) if in_band else None, "n": len(in_band),
                     "total": float(np.mean([float(r["correct"]) for r in chosen])) if chosen else None, "questions": len(chosen)}
    out["band_1_8"] = band
    out["band_CN"] = band.get(a) or {"value": None, "n": 0}
    if out["missing"]:
        out["delta"] = {"value": None}
        return out
    va = {(r["id"], r["axis"]): float(r["correct"]) for r in arm_rows[a]}
    vb = {(r["id"], r["axis"]): float(r["correct"]) for r in arm_rows[b]}
    cluster = {k: k[0] for k in set(va) | set(vb)}
    out["delta"] = V8()._cluster_diff(va, vb, cluster, n_boot=n_boot, seed=seed)
    return out


def pnum2_reading(records: Sequence[Mapping[str, Any]], scores: Mapping[str, Mapping[str, Any]], *, arms: Mapping[str, tuple[str, str]] = ARMS,
                  a: str = "CN", b: str = "C", n_boot: int = 2000, seed: int = 0) -> dict[str, Any]:
    """P-num2's reading: N9g accuracy a − b (reading ``single-s1``; unit and cluster = the record; seed means)."""
    rows = {m: N9.n9g_rows(records, (s or {}).get("answers") or {}, reading=N9.READINGS[0], second=N9.READINGS[1]) for m, s in scores.items()}
    arm_rows = {arm: N9.seed_mean_rows(rows[x], rows[y]) for arm, (x, y) in arms.items() if x in rows and y in rows}
    missing = [x for x in (a, b) if x not in arm_rows]
    if missing:
        return {"value": None, "missing": missing}
    va = {r["id"]: float(r["correct"]) for r in arm_rows[a]}
    vb = {r["id"]: float(r["correct"]) for r in arm_rows[b]}
    return {**V8()._cluster_diff(va, vb, {k: k for k in set(va) | set(vb)}, n_boot=n_boot, seed=seed), "missing": []}


def m8g_model_rows(records: Sequence[Mapping[str, Any]], preds: Mapping[str, Mapping[str, Sequence[Mapping[str, Any]]]]) -> tuple[dict, dict]:
    """Per model: the record-order rows (every listing; :func:`verdict_v08.m8g_rows`, which refuses a reordered file) and the per-record flips
    (record order vs seed 2, on the records scored in both — the seed-2 file holds the listing-0 records)."""
    v8 = V8()
    rows, flips = {}, {}
    for m, p in preds.items():
        if p.get("order"):
            rows[m] = v8.m8g_rows(records, p["order"], kind="order")
            if p.get("seed2"):
                flips[m] = v8.flip_values(rows[m], v8.m8g_rows(records, p["seed2"], kind="seed2"))
    return rows, flips


def b8_flip_values(units: Sequence[Mapping[str, Any]], scores: Mapping[str, Any]) -> dict[tuple, float]:
    """One model's B8 flips per tagged question ((env, request, question) → 1 if the top option differs between the single orders seed 1
    and seed 2; ties to the request's earlier option) — v0.8's flip definition (:func:`verdict_v08.flip_readings`)."""
    v8 = V8()
    out = {}
    for u in units:
        first, second = v8._answer(scores, u["id"], u["question"], "single-s1"), v8._answer(scores, u["id"], u["question"], "single-s2")
        if first is not None and second is not None:
            out[(u["env"], u["id"], u["question"])] = float(v8.argmax_key(first["probabilities"], u["order"]) != v8.argmax_key(second["probabilities"], u["order"]))
    return out


def five_source_mean(sources: Sequence[Mapping[str, Any]], *, n_boot: int, seed: int) -> dict[str, Any]:
    """The mean of the sources' a − b with ONE draw-wise bootstrap. Each source: ``name``, ``group`` (sources of a group share their cluster
    draws — the four M8g bins are one group over the M8g items), ``a`` / ``b`` (unit → value; the units both have are used) and ``cluster``
    (unit → cluster). Group i in the order the groups first appear is drawn with seed + i. Per draw, each source's Δ = (Σ a − Σ b) / n over the
    drawn clusters (with repeats), and the Δ are averaged; draws in which a source has no unit are dropped (NaN)."""
    v8 = V8()
    groups: dict[str, list[str]] = {}
    keys_of: dict[str, list] = {}
    for s in sources:
        keys = sorted(set(s["a"]) & set(s["b"]))
        keys_of[s["name"]] = keys
        clusters = groups.setdefault(s["group"], [])
        clusters.extend(s["cluster"][k] for k in keys)
    group_clusters = {g: sorted(set(c)) for g, c in groups.items()}
    group_seed = {g: seed + i for i, g in enumerate(groups)}
    draws = {g: v8._draws(len(c), n_boot, group_seed[g]) for g, c in group_clusters.items() if n_boot and c}
    per_source: dict[str, Any] = {}
    deltas, delta_draws = [], []
    for s in sources:
        keys = keys_of[s["name"]]
        clusters = group_clusters[s["group"]]
        index = {c: i for i, c in enumerate(clusters)}
        sa, sb, n = np.zeros(len(clusters)), np.zeros(len(clusters)), np.zeros(len(clusters))
        for k in keys:
            i = index[s["cluster"][k]]
            sa[i] += float(s["a"][k])
            sb[i] += float(s["b"][k])
            n[i] += 1
        if not keys:
            per_source[s["name"]] = {"value": None, "a": None, "b": None, "n": 0, "clusters": 0}
            deltas.append(None)
            continue
        va, vb = float(sa.sum() / n.sum()), float(sb.sum() / n.sum())
        entry = {"value": va - vb, "a": va, "b": vb, "n": len(keys), "clusters": int((n > 0).sum()), "group": s["group"],
                 "draw_seed": group_seed[s["group"]], "low": None, "high": None}
        deltas.append(va - vb)
        if s["group"] in draws:
            d = draws[s["group"]]
            with np.errstate(invalid="ignore", divide="ignore"):
                dd = (sa[d].sum(1) - sb[d].sum(1)) / n[d].sum(1)
            entry["low"], entry["high"] = v8._interval(dd)
            delta_draws.append(dd)
        per_source[s["name"]] = entry
    complete = bool(sources) and all(d is not None for d in deltas)
    out: dict[str, Any] = {"value": float(np.mean(deltas)) if complete else None, "low": None, "high": None, "per_source": per_source,
                           "group_seeds": group_seed, "complete": complete, "sources": [s["name"] for s in sources]}
    if complete:
        out["a"] = float(np.mean([per_source[s["name"]]["a"] for s in sources]))
        out["b"] = float(np.mean([per_source[s["name"]]["b"] for s in sources]))
        if n_boot and len(delta_draws) == len(sources):
            out["low"], out["high"] = v8._interval(np.mean(np.stack(delta_draws), axis=0))
    return out


def pcons_reading(m8g_records: Sequence[Mapping[str, Any]], m8g_preds: Mapping[str, Mapping[str, Sequence[Mapping[str, Any]]]], b8_units: Sequence[Mapping[str, Any]],
                  b8_scores: Mapping[str, Mapping[str, Any]], *, arms: Mapping[str, tuple[str, str]] = ARMS, a: str = "C", b: str = "F",
                  n_boot: int = 2000, seed: int = 0) -> dict[str, Any]:
    """P-cons (a − b): the five sources' flip rates (each arm = the seed mean on the units both its seeds have) and their draw-wise mean
    (:func:`five_source_mean`; M8g items drawn jointly with `seed`, B8 requests with `seed` + 1); also each arm's five-source mean."""
    v8 = V8()
    _, flips = m8g_model_rows(m8g_records, m8g_preds)
    meta = {str(r["id"]): r["meta"] for r in m8g_records}
    arm_flips = {arm: v8._seed_mean(flips[x], flips[y]) for arm, (x, y) in arms.items() if x in flips and y in flips}
    b8_flips = {m: b8_flip_values(b8_units, s) for m, s in b8_scores.items()}
    b8_arm = {arm: v8._seed_mean(b8_flips[x], b8_flips[y]) for arm, (x, y) in arms.items() if x in b8_flips and y in b8_flips}
    missing = sorted({x for x in (a, b) if x not in arm_flips} | {f"B8 {x}" for x in (a, b) if x not in b8_arm})
    if missing:
        return {"value": None, "missing": missing}
    sources = []
    for name in BIN_NAMES:
        def in_bin(values: Mapping[str, float]) -> dict[str, float]:
            return {rid: v for rid, v in values.items() if str(meta[rid].get("bin") or G8.bin_of(int(meta[rid]["n"]))) == name}

        sources.append({"name": f"M8g {name}", "group": "M8g", "a": in_bin(arm_flips[a]), "b": in_bin(arm_flips[b]),
                        "cluster": {rid: str(meta[rid]["item"]) for rid in meta}})
    sources.append({"name": "B8", "group": "B8", "a": b8_arm[a], "b": b8_arm[b], "cluster": {k: k[1] for k in set(b8_arm[a]) | set(b8_arm[b])}})
    out = five_source_mean(sources, n_boot=n_boot, seed=seed)
    out["missing"] = []
    out["arms"] = {"a": a, "b": b}
    return out


def m8g_accuracy_readings(m8g_records: Sequence[Mapping[str, Any]], m8g_preds: Mapping[str, Mapping[str, Sequence[Mapping[str, Any]]]], *,
                          arms: Mapping[str, tuple[str, str]] = ARMS, a: str = "C", b: str = "F", n_boot: int = 2000, seed: int = 0) -> dict[str, Any]:
    """H-cons's accuracy holds (a − b): M8g accuracy over every listing in the record order, and over the N ≤ 26 listings (N 6 and 18), on the
    listings both arms' seeds scored (seed means); item-cluster intervals are reported, the rule reads points."""
    v8 = V8()
    rows, _ = m8g_model_rows(m8g_records, m8g_preds)
    arm_rows = {arm: v8._arm_rows(rows[x], rows[y]) for arm, (x, y) in arms.items() if x in rows and y in rows}
    missing = [x for x in (a, b) if x not in arm_rows]
    if missing:
        return {"all": {"value": None}, "n26": {"value": None}, "missing": missing}
    common = sorted(set(arm_rows[a]) & set(arm_rows[b]))
    out: dict[str, Any] = {"missing": []}
    for name, keys in (("all", common), ("n26", [k for k in common if int(arm_rows[a][k]["n"]) <= 26])):
        out[name] = v8._cluster_diff({k: arm_rows[a][k]["correct"] for k in keys}, {k: arm_rows[b][k]["correct"] for k in keys},
                                     {k: arm_rows[a][k]["item"] for k in keys}, n_boot=n_boot, seed=seed)
    return out


# --------------------------------------------------------------------------
# Guards
# --------------------------------------------------------------------------


def home_readings(sets: Mapping[str, Sequence[Mapping[str, Any]]], preds: Mapping[str, Mapping[str, Sequence[Mapping[str, Any]]]], *, a: str, b: str,
                  arms: Mapping[str, tuple[str, str]] = ARMS, n_boot: int = 2000, seed: int = 0) -> dict[str, Any]:
    """T2–T4 (pods) a − b, arm means: the balanced band (stratified paired bootstrap, v0.7's / v0.8's home reading) and severe errors."""
    from robo_jev.bench import system1

    v4, v6 = V4(), V6()
    items = {}
    for model in {*arms[a], *arms[b]}:
        if all(sets.get(h) and (preds.get(model) or {}).get(h) for h in HOME):
            items[model] = system1.group_items({h: system1.set_items(h, sets[h], preds[model][h]) for h in HOME}, HOME)
    arm_items = {}
    for arm in (a, b):
        first, second = arms[arm]
        if first in items and second in items:
            arm_items[arm] = {metric: v6.arm_values(items[first].get(metric) or {}, items[second].get(metric) or {}) for metric in ("band_balanced", "severe_error")}
    out: dict[str, Any] = {"models": sorted(items)}
    if a in arm_items and b in arm_items:
        out["band"] = v4._paired(arm_items[a]["band_balanced"], arm_items[b]["band_balanced"], n_boot=n_boot, seed=seed, stratified=True)
        out["severe"] = v4._paired(arm_items[a]["severe_error"], arm_items[b]["severe_error"], n_boot=n_boot, seed=seed)
    return out


def g2_reading(g2: Sequence[Mapping[str, Any]] | None, preds: Mapping[str, Mapping[str, Sequence[Mapping[str, Any]]]], *, a: str, b: str,
               arms: Mapping[str, tuple[str, str]] = ARMS, n_boot: int = 2000, seed: int = 0) -> dict[str, Any] | None:
    """G2 accuracy a − b (arm means per question; :func:`verdict_v06.g2_arm_comparison`)."""
    models = (*arms[a], *arms[b])
    if not g2 or not all((preds.get(m) or {}).get("G2") for m in models):
        return None
    return V6().g2_arm_comparison(list(g2), [preds[arms[a][0]]["G2"], preds[arms[a][1]]["G2"]], [preds[arms[b][0]]["G2"], preds[arms[b][1]]["G2"]],
                                  n_boot=n_boot, seed=seed)


def guard_readings(calls: Mapping[tuple[str, str], Mapping], pod_sets: Mapping[str, Sequence], pod_preds: Mapping[str, Mapping[str, Sequence]], *,
                   a: str, b: str, n_boot: int = 2000, seed: int = 0) -> dict[str, Any]:
    """One comparison's guard readings: outside DC and CA (three-set means, v0.8's path), T2–T4 band and severe errors, G2."""
    v8 = V8()
    home = home_readings(pod_sets, pod_preds, a=a, b=b, n_boot=n_boot, seed=seed)
    return {"DC": v8.mean_of_sets(calls, a, b, metric="DC", n_boot=n_boot, seed=seed),
            "CA": v8.mean_of_sets(calls, a, b, metric="CA", n_boot=n_boot, seed=seed),
            "band": home.get("band"), "severe": home.get("severe"), "home_models": home.get("models"),
            "G2": g2_reading(pod_sets.get("G2"), pod_preds, a=a, b=b, n_boot=n_boot, seed=seed)}


# --------------------------------------------------------------------------
# Reported readings
# --------------------------------------------------------------------------


def n9g_reported(records: Sequence[Mapping[str, Any]], scores: Mapping[str, Mapping[str, Any]], *, n_boot: int = 2000, seed: int = 0) -> dict[str, Any]:
    """N9g per model and arm (:func:`robo_jev.bench.n9g_v09.n9g_summary`; an arm's ECE = the mean of its seeds'), the base included; CN − C
    and C − F per family with record-cluster intervals; each arm − the base (points)."""
    rows = {m: N9.n9g_rows(records, (s or {}).get("answers") or {}, reading=N9.READINGS[0], second=N9.READINGS[1]) for m, s in scores.items()}
    out: dict[str, Any] = {"records": len(records), "models": {}}
    for m in [x for x in MODELS if x in rows]:
        out["models"][m] = N9.n9g_summary(rows[m])
    arm_rows = {}
    for arm, (x, y) in ARMS.items():
        if x in rows and y in rows:
            arm_rows[arm] = N9.seed_mean_rows(rows[x], rows[y])
            summary = N9.n9g_summary(arm_rows[arm])
            eces = [out["models"][s].get("ece") for s in (x, y)]
            summary["ece"] = float(np.mean(eces)) if all(e is not None for e in eces) else None
            summary["ece_by_family"] = {f: (float(np.mean([out["models"][s]["ece_by_family"].get(f) for s in (x, y)]))
                                            if all(out["models"][s]["ece_by_family"].get(f) is not None for s in (x, y)) else None)
                                        for f in N9.FAMILIES if f in out["models"][x].get("ece_by_family", {})}
            out["models"][arm] = summary
    v8 = V8()
    diffs: dict[str, Any] = {}
    for a, b in (("CN", "C"), ("C", "F")):
        if a in arm_rows and b in arm_rows:
            entry = {}
            for family in ("all", *N9.FAMILIES):
                va = {r["id"]: float(r["correct"]) for r in arm_rows[a] if family == "all" or r["family"] == family}
                vb = {r["id"]: float(r["correct"]) for r in arm_rows[b] if family == "all" or r["family"] == family}
                if set(va) & set(vb):
                    entry[family] = v8._cluster_diff(va, vb, {k: k for k in set(va) | set(vb)}, n_boot=n_boot, seed=seed)
            diffs[f"{a} − {b}"] = entry
    out["differences"] = diffs
    base = ((out["models"].get("base") or {}).get("total") or {}).get("accuracy", {}).get("value")
    out["minus_base"] = {arm: (((out["models"].get(arm) or {}).get("total") or {}).get("accuracy", {}).get("value") - base)
                         for arm in RULE_ARMS if base is not None and (((out["models"].get(arm) or {}).get("total") or {}).get("accuracy", {}).get("value")) is not None}
    return out


def standard_context(calls: Mapping[tuple[str, str], Mapping], *, n_boot: int = 2000, seed: int = 0) -> dict[str, Any]:
    """The short-form call context (reported): call metrics per model and arm per set, and CN − C, C − F per set (H-set and B1 included)."""
    v5 = V5()
    out: dict[str, Any] = {"context": {}, "CN − C": {}, "C − F": {}, "vs_base_DC": {}}
    for s in CALL_SETS:
        out["context"][s] = {m: v5.call_metrics(calls[(m, s)], n_boot=n_boot, seed=seed) for m in (*MODELS, *ARMS) if (m, s) in calls}
        for a, b in (("CN", "C"), ("C", "F")):
            if (a, s) in calls and (b, s) in calls:
                out[f"{a} − {b}"][s] = v5.call_difference(calls[(a, s)], calls[(b, s)], n_boot=n_boot, seed=seed)
        if ("base", s) in calls:
            out["vs_base_DC"][s] = {arm: v5.call_difference(calls[(arm, s)], calls[("base", s)], n_boot=n_boot, seed=seed).get("DC")
                                    for arm in RULE_ARMS if (arm, s) in calls}
    return out


@contextlib.contextmanager
def _figure_names():
    """a16s_v08.curves_svg draws fixed series slots (F, FP, v07, base) and names them from its SERIES_NAME; for v0.9's figure the slots carry
    C, CN, the v0.8 F reference and the base, so their names are set for the call and restored afterwards."""
    from robo_jev.bench import a16s_v08 as S

    saved = dict(S.SERIES_NAME)
    S.SERIES_NAME.update({slot: name for slot, (_, name) in A16S_FIGURE_SLOTS.items()})
    try:
        yield S
    finally:
        S.SERIES_NAME.clear()
        S.SERIES_NAME.update(saved)


def a16s_figure(reading: Mapping[str, Any] | None) -> str | None:
    """The A16s curves (:func:`robo_jev.bench.a16s_v08.curves_svg`): C, CN, the v0.8 F reference (seed means, their seeds thin) and the
    untrained base, dashed; None when none of them was read."""
    models = (reading or {}).get("models") or {}
    series = {slot: models[key]["curves"] for slot, (key, _) in A16S_FIGURE_SLOTS.items() if key in models}
    if not series:
        return None
    seeds = {slot: models[key]["seed_curves"] for slot, (key, _) in A16S_FIGURE_SLOTS.items() if key in models and models[key].get("seed_curves")}
    with _figure_names() as S:
        return S.curves_svg(series, seeds=seeds, title="A16s — the arm's distance sweep (v0.9 verdict; P-num1 reads accuracy)")


# --------------------------------------------------------------------------
# Compute (in-memory inputs → readings and the verdict)
# --------------------------------------------------------------------------


def a16_vs_v07(a16: Mapping[str, Any] | None) -> dict[str, Any]:
    """CN's and v0.7's A16 rule agreement (arm seed means) and the stored v0.7 endpoint answers' — printed under H-num SUPPORTED."""
    models = (a16 or {}).get("models") or {}

    def agreement(entry: Mapping[str, Any] | None) -> float | None:
        return _num((((entry or {}).get("summary") or {}).get("agreement") or {}).get("value"))

    return {"CN": agreement(models.get("CN")), "v07": agreement(models.get("v07")),
            "v07_stored": agreement((a16 or {}).get("reference_v07_stored"))}


def compute(*, standard_sets: Mapping[str, Sequence], standard_preds: Mapping[str, Mapping[str, Sequence]], pod_sets: Mapping[str, Sequence],
            pod_preds: Mapping[str, Mapping[str, Sequence]], a16s: Mapping[str, Any], n9g: Mapping[str, Any], m8g: Mapping[str, Any],
            b8: Mapping[str, Any], a16: Mapping[str, Any] | None = None, s16: Mapping[str, Any] | None = None, r14: Mapping[str, Any] | None = None,
            jev: Mapping[str, Any] | None = None, n_boot: int = 2000, seed: int = 0) -> dict[str, Any]:
    """Every reading and the verdict from in-memory inputs (exclusions already applied to the records): `a16s` / `n9g` = {records, scores
    (model → bench score file)}; `m8g` = {records, preds (model → {"order", "seed2"})}; `b8` = {tags, records (by id), scores, exclude};
    `a16` = {records: {A16, A16y}, scores: {A16, A16y}, stored} (reported); `s16` = {records, scores, stored}; `r14` = {tags, record_ids,
    scores}; `jev` = {records, jev_preds, scores}; a reported part may be None (absent) or {"refused": problems}."""
    v8 = V8()
    calls = v8.calls_of(standard_sets, standard_preds, CALL_SETS)
    pnum1 = pnum1_reading(a16s["records"], a16s["scores"], n_boot=n_boot, seed=seed)
    pnum2 = pnum2_reading(n9g["records"], n9g["scores"], n_boot=n_boot, seed=seed)
    b8_units = v8.b8_units(b8["tags"], b8["records"], exclude=b8.get("exclude") or ())
    pcons = pcons_reading(m8g["records"], m8g["preds"], b8_units, b8["scores"], n_boot=n_boot, seed=seed)
    accuracy = m8g_accuracy_readings(m8g["records"], m8g["preds"], n_boot=n_boot, seed=seed)
    guards = {"CN-C": guard_readings(calls, pod_sets, pod_preds, a="CN", b="C", n_boot=n_boot, seed=seed),
              "C-F": guard_readings(calls, pod_sets, pod_preds, a="C", b="F", n_boot=n_boot, seed=seed)}
    readings: dict[str, Any] = {"P-num1": pnum1, "P-num2": pnum2, "P-cons": pcons, "M8g accuracy": accuracy, "guards": guards}
    # reported, never deciding
    readings["N9g"] = n9g_reported(n9g["records"], n9g["scores"], n_boot=n_boot, seed=seed)
    readings["A16s"] = v8.a16s_readings(a16s["records"], a16s["scores"])
    readings["M8g"] = v8.m8g_readings(m8g["records"], m8g["preds"], n_boot=n_boot, seed=seed)
    readings["B8 flips"] = v8.flip_readings(b8_units, b8["scores"], group="env")
    readings["standard"] = standard_context(calls, n_boot=n_boot, seed=seed)
    if a16 and not a16.get("refused"):
        readings["A16"] = v8.a16_readings(a16["records"][A16], (a16.get("scores") or {}).get(A16) or {}, (a16.get("scores") or {}).get(A16Y) or {},
                                          (a16.get("stored") or {}).get(A16) or {})
    else:
        readings["A16"] = {"refused": a16["refused"]} if a16 else {"absent": "no A16 files"}
    if s16 and not s16.get("refused"):
        readings["S16"] = v8.s16_readings(s16["records"], s16.get("scores") or {}, s16.get("stored") or {})
    else:
        readings["S16"] = {"refused": s16["refused"]} if s16 else {"absent": "no S16 files"}
    if r14 and not r14.get("refused"):
        readings["R14"] = v8.r14_readings(v8.r14_units(r14["tags"], r14.get("record_ids")), r14.get("scores") or {})
    else:
        readings["R14"] = {"refused": r14["refused"]} if r14 else {"absent": "no R14 files"}
    readings["Jev"] = v8.jev_readings(jev["records"], jev["jev_preds"], jev["scores"]) if jev else {"absent": "no Jev comparison files"}
    rule_inputs = {"P-num1": {"delta": pnum1.get("delta"), "band_CN": pnum1.get("band_CN")}, "P-num2": pnum2,
                   "P-cons": {k: pcons.get(k) for k in ("value", "low", "high")}, "M8g accuracy": accuracy.get("all"),
                   "M8g accuracy N<=26": accuracy.get("n26"),
                   "guards": {name: {"DC": g["DC"], "CA": g["CA"], "band": g.get("band"), "severe": g.get("severe"), "G2": g.get("G2")}
                              for name, g in guards.items()},
                   "A16 vs v0.7": a16_vs_v07(readings.get("A16"))}
    return {"verdict": decide(rule_inputs), "readings": readings}


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


def source_dir(model: str, set_name: str, spark_dir: Path, v08_spark: Path) -> Path:
    """Where a model's score file of a set lives: v0.9's chain for C and CN (every set) and for N9g (every model); v0.8's chain otherwise (F,
    v0.7 and the base — the same set and conditions, reused)."""
    return Path(spark_dir) if model in V09_MODELS or set_name == N9G else Path(v08_spark)


def checkpoints(dirs: Mapping[str, Path]) -> dict[str, Path | None]:
    """Each run's checkpoint next to its fetched reports; v0.7's; None for the base."""
    v8 = V8()
    out: dict[str, Path | None] = {m: v8.pod_checkpoint(Path(d)) for m, d in dirs.items()}
    out.update({"v07s0": V07_CKPT[0], "v07s1": V07_CKPT[1], "base": None})
    return out


def read_overlaps(*, required: bool) -> tuple[dict[str, set[str]], dict[str, Any], list[str]]:
    """The union of the exclusions of overlap-v08.json (train-v08) and overlap-v09.json (train-v09n) per gate, the training-input sha256 each
    recheck read, and notes. Refuses without either file when `required`."""
    keys = {"n9g": "excluded_n9g", "items": "excluded_items", "b8": "excluded_b8", "r14": "excluded_r14", "a16": "excluded_a16",
            "s16": "excluded_s16", "a16s": "excluded_a16s"}
    out: dict[str, set[str]] = {k: set() for k in keys}
    trained: dict[str, Any] = {"train_v08": None, "train_v09n": None}
    notes: list[str] = []
    for path, name, field in ((OVERLAP_V08, "overlap-v08.json", "train_v08"), (OVERLAP_V09, "overlap-v09.json", "train_v09n")):
        if not Path(path).is_file():
            if required:
                raise SystemExit(f"{path} is missing — run the recheck ({'recheck-v08' if field == 'train_v08' else 'scripts/v09_gates_build.py recheck-v09'}) "
                                 "on the training input first")
            notes.append(f"{name} absent: no exclusions from it (--no-v09-recheck)")
            continue
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        for key, src in keys.items():
            out[key] |= {str(x) for x in data.get(src) or []}
        entry = data.get(field)
        trained[field] = entry.get("sha256") if isinstance(entry, Mapping) else None
        notes.append(f"{name} (training input sha256 {str(trained[field])[:12]}): " + ", ".join(f"{k} {len({str(x) for x in data.get(src) or []})}"
                                                                                             for k, src in keys.items()) + " left out.")
    return out, trained, notes


def trained_input_problems(cks: Mapping[str, Path | None], trained: Mapping[str, Any]) -> list[str]:
    """The C runs trained on the input overlap-v08.json read (train-v08) and the CN runs on the one overlap-v09.json read (train-v09n): each
    run's metrics.json (next to its checkpoint) → manifest.data.file_sha256."""
    problems = []
    for model in V09_MODELS:
        want = trained.get("train_v09n" if model.startswith("v09CN") else "train_v08")
        if want is None:
            continue
        ck = cks.get(model)
        metrics = Path(ck).parent / "metrics.json" if ck else None
        got = None
        if metrics is not None and metrics.is_file():
            got = ((json.loads(metrics.read_text(encoding="utf-8")).get("manifest") or {}).get("data") or {}).get("file_sha256")
        if got != want:
            problems.append(f"{model}: trained on {str(got)[:12]}, the recheck read {str(want)[:12]} — the exclusions do not cover its training input")
    return problems


def load_bench(set_name: str, input_path: Path, readings: Sequence[str], models: Sequence[str], cks: Mapping[str, Path | None], *, spark_dir: Path,
               v08_spark: Path, frozen: Any = None, required: Sequence[str] = (), problems: list[str], notes: list[str],
               left_out: list[str]) -> dict[str, Any]:
    """Bench score files (:func:`verdict_v08.check_bench_source`: kind, set, tag, scored file, readings, frozen record, checkpoint) of one set,
    each from its model's directory (:func:`source_dir`). For a `required` model (the rule reads it) a missing file or a failing check is a
    problem; any other model's failing file is left out and listed in `left_out` (v0.8's handling of reported sets)."""
    v8 = V8()
    out = {}
    for m in models:
        path = source_dir(m, set_name, spark_dir, v08_spark) / f"{set_name}-{SPARK_TAGS[m]}.json"
        if not path.is_file():
            (problems if m in required else notes).append(f"{'missing' if m in required else 'absent'}: {path}")
            continue
        data, found = v8.check_bench_source(path, set_name=set_name, tag=SPARK_TAGS[m], input_path=input_path, checkpoint=cks.get(m), readings=readings,
                                            frozen=frozen)
        if found and m not in required:
            left_out += [f"{m} {set_name}: {p}" for p in found]
            continue
        problems += [f"{m} {set_name}: {p}" for p in found]
        out[m] = data
    return out


def bundle_problems(scores: Mapping[str, Mapping[str, Any]], models: Sequence[str], *, set_name: str) -> list[str]:
    """The rule's comparison on `set_name` reads these models' bench files: they must carry the same bundle (``bundle_sha256``) — otherwise
    the comparison would cross bundle versions (e.g. F's v0.8 files against C's)."""
    found = {m: (scores.get(m) or {}).get("bundle_sha256") for m in models if m in scores}
    if len(set(found.values())) > 1:
        return [f"{set_name}: bench files of different bundles — " + ", ".join(f"{m} {str(v)[:12]}" for m, v in found.items())]
    return []


def condition_problems(scores: Mapping[str, Mapping[str, Any]], models: Sequence[str], *, set_name: str, serve: bool | None = None) -> list[str]:
    """The rule's comparison on `set_name` reads these models' bench files under one condition: the same serve's-adapter use (and, when
    `serve` is given, that use — P-num1 reads A16s under serve's conditions) and the same order key."""
    used = {m: bool(((scores.get(m) or {}).get("serve_adapter") or {}).get("used")) for m in models if m in scores}
    keys = {m: (scores.get(m) or {}).get("key") for m in models if m in scores}
    problems = []
    if len(set(used.values())) > 1 or (serve is not None and any(u != serve for u in used.values())):
        problems.append(f"{set_name}: serve's adapter {'required' if serve else 'mixed'} — " + ", ".join(f"{m} {u}" for m, u in used.items()))
    if len(set(keys.values())) > 1:
        problems.append(f"{set_name}: the files were scored with different order keys — " + ", ".join(f"{m} {k}" for m, k in keys.items()))
    return problems


def load_frozen_dir(check, directory: Path, name: str) -> dict[str, Any]:
    """A frozen gate directory's ``frozen-before-scoring.json`` after its freeze check — refuses (no override) when a file changed."""
    problems = check(directory)
    if problems:
        raise SystemExit(f"{name}'s frozen files changed or are missing — refusing (no override): " + "; ".join(problems))
    return json.loads((Path(directory) / "frozen-before-scoring.json").read_text(encoding="utf-8"))


def load_m8g(cks: Mapping[str, Path | None], *, spark_dir: Path, v08_spark: Path, models: Sequence[str], required: Sequence[str], problems: list[str],
             notes: list[str], left_out: list[str]) -> dict[str, dict[str, list]]:
    """M8g prediction files per model and kind (record order; seed 2), each from its model's directory, checked by
    :func:`verdict_v08.check_m8g_source` (the gate file, the kind's orders, the prompt digest, the checkpoint); a failing file of a model the
    rule does not read (not `required`) is left out and listed."""
    v8 = V8()
    out: dict[str, dict[str, list]] = {}
    for m in models:
        for kind, stem in M8G_KINDS.items():
            path = source_dir(m, "M8g", spark_dir, v08_spark) / f"{stem}-{SPARK_TAGS[m]}.predictions.jsonl"
            if not path.is_file():
                (problems if m in required else notes).append(f"{'missing' if m in required else 'absent'}: {path}")
                continue
            found = v8.check_m8g_source(path, kind, cks.get(m))
            if found and m not in required:
                left_out += [f"{m} {stem}: {p}" for p in found]
                continue
            problems += [f"{m} {stem}: {p}" for p in found]
            out.setdefault(m, {})[kind] = _read(path)
    return out


def standard_paths(spark_dir: Path, v08_spark: Path) -> dict[tuple[str, str], Path]:
    """(model, set) → the Spark short-form prediction file: C and CN (v0.9's chain), F (v0.8's), the base (v0.7's references)."""
    paths: dict[tuple[str, str], Path] = {}
    for m in (*V09_MODELS, *F_MODELS):
        for s in CALL_SETS:
            paths[(m, s)] = source_dir(m, s, spark_dir, v08_spark) / f"{s}-{SPARK_TAGS[m]}.predictions.jsonl"
    for s, path in V7().BASE_SPARK.items():
        paths[("base", s)] = Path(path)
    return paths


def required_files(spark_dir: Path, v08_spark: Path) -> list[str]:
    """The files the rule reads — the verdict refuses if one is missing."""
    from robo_jev.bench import a16s_v08

    paths = standard_paths(spark_dir, v08_spark)
    need = [paths[(m, s)] for m in (*V09_MODELS, *F_MODELS) for s in OUTSIDE_SETS]
    need += [Path(spark_dir) / f"{set_name}-{SPARK_TAGS[m]}.json" for m in V09_MODELS for set_name in (N9G, A16S)]
    for m in (*ARMS["C"], *ARMS["F"]):
        directory = source_dir(m, "M8g", spark_dir, v08_spark)
        need += [directory / f"{stem}-{SPARK_TAGS[m]}.predictions.jsonl" for stem in M8G_KINDS.values()] + [directory / f"B8-{SPARK_TAGS[m]}.json"]
    need += [Path(N9G_DIR) / "N9g.jsonl", Path(N9G_DIR) / "frozen-before-scoring.json", a16s_v08.A16S_DIR / "A16s.jsonl",
             a16s_v08.A16S_DIR / "frozen-before-scoring.json", G8.B8_FILE, G8.B8_DIR / "tags.json", G8.B8_DIR / "frozen-before-scoring.json", G8.M8G_FILE]
    return [str(p) for p in need if not Path(p).is_file()]


def load_pods(dirs: Mapping[str, Path], *, allow_mismatch: bool) -> tuple[dict, dict, dict]:
    """T2, T3, T4, G2 records and the six runs' pod predictions (each checked like the Spark files)."""
    v8 = V8()
    set_paths = dict(V6().SET_PATHS)
    sets = {s: _read(set_paths[s]) for s in POD_SETS}
    preds: dict[str, dict[str, list]] = {}
    sources = {}
    for model, directory in dirs.items():
        preds[model] = {}
        for s in POD_SETS:
            path = v8.pod_file(Path(directory), s)
            preds[model][s] = _read(path)
            sources[f"{model} {s}"] = V4().check_source(path, set_paths[s], sets[s], preds[model][s])
    problems = [f"{k}: {'; '.join(v['problems'])}" for k, v in sources.items() if v["problems"]]
    if problems and not allow_mismatch:
        raise SystemExit("refusing (use --allow-mismatch to read anyway):\n  " + "\n  ".join(problems))
    return sets, preds, sources


# --------------------------------------------------------------------------
# Markdown
# --------------------------------------------------------------------------


def _fmt(value: Any, digits: int = 3, signed: bool = False) -> str:
    return V8()._fmt(value, digits, signed)


def _cid(d: Mapping[str, Any] | None, signed: bool = True, key: str = "value") -> str:
    return V8()._cid(d, signed, key)


def _yes(flag: bool) -> str:
    return "PASS" if flag else "FAIL"


def _render_n9g(n9g: Mapping[str, Any]) -> list[str]:
    lines = ["", "### N9g — the numeric-rule gate (per model and arm; the untrained base included)", ""]
    models = n9g.get("models") or {}
    order = [m for m in (*RULE_ARMS, *MODELS) if m in models]
    lines += ["| model / arm | accuracy (n) | mean p(key) | flips s1 / s2 | ECE | seen domains | unseen domains |", "|---|---|---|---|---|---|---|"]
    for m in order:
        e = models[m]
        groups = e.get("by_domain_group") or {}
        lines.append(f"| {LABELS.get(m, m)} | {_fmt(e['total']['accuracy']['value'])} ({e['total']['accuracy']['n']}) | {_fmt(e['total']['p_key']['value'])} | "
                     f"{_fmt(e['total']['flips']['value'])} | {_fmt(e.get('ece'))} | {_fmt(((groups.get('seen domain') or {}).get('accuracy') or {}).get('value'))} | "
                     f"{_fmt(((groups.get('unseen domain') or {}).get('accuracy') or {}).get('value'))} |")
    for field, title in (("by_family", "family"), ("by_domain", "domain"), ("by_format", "number format"), ("by_wording", "wording"),
                         ("by_near", "near-boundary kind"), ("by_options", "options")):
        keys = sorted({k for m in order for k in (models[m].get(field) or {})})
        if not keys:
            continue
        lines += ["", f"Accuracy by {title}:", "", "| model / arm | " + " | ".join(keys) + " |", "|---|" + "---|" * len(keys)]
        for m in order:
            cells = [_fmt((((models[m].get(field) or {}).get(k) or {}).get("accuracy") or {}).get("value")) for k in keys]
            lines.append(f"| {LABELS.get(m, m)} | " + " | ".join(cells) + " |")
    for name, entry in (n9g.get("differences") or {}).items():
        lines += ["", f"{name} per family (record-cluster bootstrap): " + "; ".join(f"{f} {_cid(d)}" for f, d in entry.items()) + "."]
    if n9g.get("minus_base"):
        lines.append("Arm − untrained base (accuracy, points): " + ", ".join(f"{a} {v:+.3f}" for a, v in n9g["minus_base"].items()) + ".")
    return lines


def _render_a16s(a16s: Mapping[str, Any]) -> list[str]:
    from robo_jev.bench import a16s_v08 as S

    lines = ["", "### A16s — accuracy per band (single order seed 1; cells + / −)", ""]
    models = a16s.get("models") or {}
    order = [m for m in (*RULE_ARMS, "v07", *MODELS) if m in models]
    bands = a16s.get("bands") or [name for name, _, _ in S.READ_BANDS]
    lines += ["| model / arm | " + " | ".join(bands) + " | all | stay | flips |", "|---|" + "---|" * (len(bands) + 3)]
    for m in order:
        sm = models[m]["summary"]
        by = sm.get("by_band") or {}
        cells = [f"{_fmt(((by.get(b + '|+') or {}).get('accuracy') or {}).get('value'), 2)} / {_fmt(((by.get(b + '|-') or {}).get('accuracy') or {}).get('value'), 2)}"
                 for b in bands]
        total = sm.get("total") or {}
        lines.append(f"| {LABELS.get(m, m)} | " + " | ".join(cells) + f" | {_fmt((total.get('accuracy') or {}).get('value'), 3)} | "
                     f"{_fmt((total.get('stay') or {}).get('value'), 2)} | {_fmt((total.get('flips') or {}).get('value'), 2)} |")
    return lines


def _render_a16(a16: Mapping[str, Any]) -> list[str]:
    v8 = V8()
    lines = ["", "### A16 — the arm's bench requests (single order seed 1, serve's conditions; reported)", ""]
    if a16.get("absent") or a16.get("refused"):
        return lines + [f"A16: {a16.get('absent') or 'refused — ' + '; '.join(a16['refused'])}."]
    lines += ["| model / arm | agreement | x | y | z | stay rate 0.2–3 cm | y without '(up is plus)' |", "|---|---|---|---|---|---|---|"]
    rows = [("v0.7 endpoint (stored)", a16.get("reference_v07_stored") or {})] + [(LABELS.get(m, m), e) for m, e in (a16.get("models") or {}).items()]
    for label, entry in rows:
        sm = entry.get("summary") or {}
        if not sm:
            continue
        ax = sm.get("agreement_by_axis") or {}
        ab = ((entry.get("ablation_y") or {}).get("without_phrase") or {}).get("agreement")
        lines.append(f"| {label} | {v8._rate_cell(sm.get('agreement'))} | {v8._rate_cell(ax.get('x'))} | {v8._rate_cell(ax.get('y'))} | {v8._rate_cell(ax.get('z'))} | "
                     f"{v8._rate_cell(sm.get('stay_rate_small'))} | {v8._rate_cell(ab)} |")
    lines += ["", "Agreement and stay rate per axis × signed band (arms):", "", "| arm | axis:band | agreement | stay rate |", "|---|---|---|---|"]
    for m, entry in (a16.get("models") or {}).items():
        if m not in ARMS:
            continue
        sm = entry.get("summary") or {}
        for band, d in (sm.get("agreement_by_axis_band") or {}).items():
            lines.append(f"| {LABELS.get(m, m)} | {band} | {v8._rate_cell(d)} | {v8._rate_cell((sm.get('stay_rate_by_axis_band') or {}).get(band))} |")
    return lines


def _render_flips(title: str, part: Mapping[str, Any], group_name: str) -> list[str]:
    lines = ["", f"### {title} (serve's three orders; reported)", ""]
    if part.get("absent") or part.get("refused"):
        return lines + [f"{part.get('absent') or 'refused — ' + '; '.join(part['refused'])}."]
    lines += [f"| model / arm | pooled flip rate (questions) | per {group_name} |", "|---|---|---|"]
    for m, e in (part.get("models") or {}).items():
        groups = e.get("per_env") or e.get("per_group") or {}
        lines.append(f"| {LABELS.get(m, m)} | {_fmt(e['pooled'].get('flip_rate'))} ({e['pooled'].get('questions')}) | "
                     + ", ".join(f"{g} {_fmt(x.get('flip_rate'))}" for g, x in groups.items()) + " |")
    return lines


def render(result: Mapping[str, Any]) -> str:
    v, r = result["verdict"], result["readings"]
    lines = [f"# v0.9 pre-declared rule — H-num {v['H-num']} · H-cons {v['H-cons']}", ""]
    if result.get("pipeline_check"):
        lines += [f"**PIPELINE CHECK ONLY — {result['pipeline_check']}; this is not the v0.9 verdict.**", ""]
    lines += [f"Rule written {RULE['written']}. Applied by `scripts/verdict_v09.py` ({SCRIPT_VERSION}, git {result.get('git')}); {result['n_boot']} paired "
              f"cluster-bootstrap draws (seed {result['seed']}); seed means (C = v0.9C s0 + s1, CN = v0.9CN s0 + s1, F = v0.8F s0 + s1).", ""]
    p1 = r.get("P-num1") or {}
    d1 = p1.get("delta") or {}
    c1 = v["P-num1"]["checks"]
    band = p1.get("band_1_8") or {}
    lines += ["## H-num (CN − C)", "", "### P-num1 — gate A16s (single order seed 1, serve's conditions; unit = an axis question, cluster = the request)", "",
              "| reading | CN | C | CN − C [95 %] | questions | clusters |", "|---|---|---|---|---|---|",
              f"| accuracy | {_fmt(d1.get('a'))} | {_fmt(d1.get('b'))} | {_cid(d1)} | {d1.get('n', '—')} | {d1.get('clusters', '—')} |", "",
              "1–8 cm band (bands 1–3 and 3–8, both signs pooled): " + "; ".join(f"{LABELS.get(a, a)} {_fmt(e.get('value'))} (n {e.get('n')})" for a, e in band.items()) + ".", "",
              f"Δ ≥ +0.25: {c1['A16s_CN_minus_C_ge_+0.25']} · lower > 0: {c1['A16s_low_gt_0']} · CN 1–8 cm ≥ 0.60: {c1['CN_1-8cm_accuracy_ge_0.60']} → "
              f"**P-num1 {_yes(v['P-num1']['pass'])}**", ""]
    d2 = r.get("P-num2") or {}
    c2 = v["P-num2"]["checks"]
    lines += ["### P-num2 — gate N9g (single order seed 1; unit = cluster = the record)", "", "| reading | CN | C | CN − C [95 %] | questions |", "|---|---|---|---|---|",
              f"| accuracy | {_fmt(d2.get('a'))} | {_fmt(d2.get('b'))} | {_cid(d2)} | {d2.get('n', '—')} |", "",
              f"Δ ≥ +0.20: {c2['N9g_CN_minus_C_ge_+0.20']} · lower > 0: {c2['N9g_low_gt_0']} → **P-num2 {_yes(v['P-num2']['pass'])}**", "",
              f"## H-num outcome: **{v['H-num']}**", ""]
    pc = r.get("P-cons") or {}
    lines += ["## H-cons (C − F)", "", "### P-cons — flip rate between two order seeds, the mean of five sources (M8g items drawn jointly for the four bins, "
              "B8 requests separately; averaged per draw)", "", "| source | C | F | C − F [95 %] | units | clusters | draw seed |", "|---|---|---|---|---|---|---|"]
    for name in CONS_SOURCES:
        e = (pc.get("per_source") or {}).get(name) or {}
        lines.append(f"| {name} | {_fmt(e.get('a'))} | {_fmt(e.get('b'))} | {_cid(e)} | {e.get('n', '—')} | {e.get('clusters', '—')} | {e.get('draw_seed', '—')} |")
    cc = v["P-cons"]["checks"]
    acc = (r.get("M8g accuracy") or {})
    hc = v["accuracy_holds"]["checks"]
    lines += [f"| **mean of five** | {_fmt(pc.get('a'))} | {_fmt(pc.get('b'))} | {_cid(pc)} | | | |", "",
              f"Mean ≤ −0.10: {cc['flip_mean_le_-0.10']} · upper < 0: {cc['flip_mean_high_lt_0']} → **P-cons {_yes(v['P-cons']['pass'])}**", "",
              "| accuracy hold (C − F, points) | C | F | C − F [95 %, items] | listings | check |", "|---|---|---|---|---|---|",
              f"| M8g, every listing (≥ −0.02) | {_fmt((acc.get('all') or {}).get('a'))} | {_fmt((acc.get('all') or {}).get('b'))} | {_cid(acc.get('all'))} | "
              f"{(acc.get('all') or {}).get('n', '—')} | {hc['M8g_accuracy_ge_-0.02']} |",
              f"| M8g, N ≤ 26 (N 6 and 18; ≥ −0.02) | {_fmt((acc.get('n26') or {}).get('a'))} | {_fmt((acc.get('n26') or {}).get('b'))} | {_cid(acc.get('n26'))} | "
              f"{(acc.get('n26') or {}).get('n', '—')} | {hc['M8g_accuracy_N<=26_ge_-0.02']} |", "", f"## H-cons outcome: **{v['H-cons']}**", "",
              "## Guards (reported with pass / fail — they do not change the outcome labels)", ""]
    for name in ("CN-C", "C-F"):
        g = (r.get("guards") or {}).get(name) or {}
        gc = v["guards"][name]["checks"]
        sev, g2 = g.get("severe") or {}, g.get("G2") or {}
        lines += [f"**{name.replace('-', ' − ')}**", "", "| guard | reading | pass |", "|---|---|---|",
                  f"| outside DC, mean of A4 / T9 / T8b-B2 (≤ +0.02) | {_cid(g.get('DC'))} | {gc['outside_DC_mean_le_+0.02']} |",
                  f"| outside CA, mean (≥ −0.05) | {_cid(g.get('CA'))} | {gc['outside_CA_mean_ge_-0.05']} |",
                  f"| T2–T4 balanced band (lower > −0.05) | {_cid(g.get('band'))} | {gc['T2-T4_band_low_gt_-0.05']} |",
                  f"| T2–T4 severe errors (a ≤ b + 0.01) | {_fmt(sev.get('a'))} vs {_fmt(sev.get('b'))} | {gc['T2-T4_severe_a_le_b_plus_0.01']} |",
                  f"| G2 accuracy (≥ −0.02, lower > −0.045) | {_fmt(g2.get('diff'), signed=True)} [{_fmt(g2.get('low'), signed=True)}, {_fmt(g2.get('high'), signed=True)}] | "
                  f"{gc['G2_point_ge_-0.02'] and gc['G2_low_gt_-0.045']} |", ""]
        per = (g.get("DC") or {}).get("per_set") or {}
        lines.append("Per set ΔDC: " + "; ".join(f"{s} {_cid(per.get(s))}" for s in OUTSIDE_SETS) + ".")
        lines.append("")
    ns = v["next_steps"]
    lines += ["## Next steps (pre-declared) and release", ""] + [f"- **{b}** → {n}." for b, n in zip(ns["branches"], ns["next"])]
    lines += [f"- {n}." for n in ns.get("notes") or []] + [f"- {v['release_line']}", "", "## Reported, not deciding"]
    lines += _render_n9g(r.get("N9g") or {})
    lines += _render_a16s(r.get("A16s") or {})
    if (result.get("figures") or {}).get(A16S):
        lines.append(f"\nThe curves: `{result['figures'][A16S]}` (C, CN, the v0.8 F reference — seed means, their seeds thin — and the untrained base, dashed).")
    lines += _render_a16(r.get("A16") or {})
    m8 = r.get("M8g") or {}
    lines += ["", "### M8g per bin (record order; reported)", "", "| model / arm | bin | listings | accuracy [95 %] | ECE | flip rate (records) | acc two / single |",
              "|---|---|---|---|---|---|---|"]
    for model, entry in (m8.get("per_bin") or {}).items():
        for name, e in entry.items():
            lines.append(f"| {LABELS.get(model, model)} | {name} | {e['listings']} | {_cid(e.get('accuracy'), signed=False)} | {_fmt(e.get('ece'))} | "
                         f"{_fmt(e.get('flip_rate'))} ({e.get('flip_records')}) | {_fmt(e.get('acc_two'))} / {_fmt(e.get('acc_single'))} |")
    lines += ["", "M8g per source (accuracy per bin): " + "; ".join(
        f"{LABELS.get(m, m)} " + ", ".join(f"{src} {b} {_fmt(x['accuracy'])}" for src, by_bin in e.items() for b, x in by_bin.items())
        for m, e in (m8.get("per_source") or {}).items() if m in ARMS) + "."]
    bf = r.get("B8 flips") or {}
    lines += ["", "### B8 flips (single order seed 1 vs seed 2, every tagged question; reported)", "", "| model / arm | pooled (n) | per env |", "|---|---|---|"]
    for m, e in (bf.get("pooled") or {}).items():
        lines.append(f"| {LABELS.get(m, m)} | {_fmt(e.get('value'))} ({e.get('questions')}) | "
                     + ", ".join(f"{g} {_fmt(x['value'])}" for g, x in ((bf.get('per_group') or {}).get(m) or {}).items()) + " |")
    lines += _render_flips("R14 — robo-jev-web's collected bench requests", r.get("R14") or {}, "environment")
    lines += _render_flips("S16 — soccer", r.get("S16") or {}, "group")
    jev = r.get("Jev") or {}
    lines += ["", "### Jev top-1 agreement (comparison set, not sealed; reported)", ""]
    if jev.get("absent"):
        lines.append(f"absent: {jev['absent']}")
    else:
        lines += ["| model | part / reading | agreement | questions | accuracy |", "|---|---|---|---|---|"]
        for model, e in (jev.get("models") or {}).items():
            for key, x in e.items():
                lines.append(f"| {LABELS.get(model, model)} | {key} | {_fmt(x['agreement'])} | {x['questions']} | {_fmt(x['accuracy'])} |")
    st = r.get("standard") or {}
    lines += ["", "### Short-form context (call view)", "", "| set | model / arm | DC | CA | GAP |", "|---|---|---|---|---|"]
    for s, by_model in (st.get("context") or {}).items():
        for model, e in by_model.items():
            lines.append(f"| {s} | {LABELS.get(model, model)} | {_cid(e.get('DC'), signed=False)} | {_cid(e.get('CA'), signed=False)} | {_cid(e.get('GAP'))} |")
    for name in ("CN − C", "C − F"):
        lines += ["", f"{name} per set: " + "; ".join(f"{s} ΔDC {_cid((d or {}).get('DC'))}, ΔCA {_cid((d or {}).get('CA'))}" for s, d in (st.get(name) or {}).items()) + "."]
    lines += ["", "## Notes", ""] + [f"- {note}" for note in result.get("notes") or []] + ["", "## Rule (verbatim parts)", ""] + [f"- **{k}** — {t}" for k, t in RULE.items()]
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------
# Pipeline check (existing v0.8 files, stand-in arms; not the verdict)
# --------------------------------------------------------------------------


def _close(a: Any, b: Any, tol: float = 1e-12) -> bool:
    return _num(a) is not None and _num(b) is not None and abs(float(a) - float(b)) <= tol


def pipeline_check(*, n_boot: int, seed: int, out: Path, v08_spark: Path = V08_SPARK, v08_verdict: Path = V08_VERDICT) -> dict[str, Any]:
    """The decision paths of P-num1 and P-cons on v0.8's own files with stand-in arms (module docstring), against v0.8's verdict.json."""
    from robo_jev.bench import a16s_v08 as S

    v8 = V8()
    ref = json.loads(Path(v08_verdict).read_text(encoding="utf-8"))["readings"]
    exclusions = v8.read_exclusions(OVERLAP_V08) if Path(OVERLAP_V08).is_file() else {"items": [], "b8": [], "a16s": []}
    stand = {"F": ("v08Fs0", "v08Fs1"), "FP": ("v08FPs0", "v08FPs1")}
    models = (*stand["F"], *stand["FP"])
    rows: list[dict[str, Any]] = []

    def add(what: str, got: Any, want: Any) -> None:
        rows.append({"what": what, "got": _num(got), "want": _num(want), "diff": None if _num(got) is None or _num(want) is None else float(got) - float(want),
                     "ok": _close(got, want)})

    # A16s: every model's and arm's accuracy (total and every band cell) through the P-num1 path's rows, and the P-num1 path itself
    a16s_records = [r for r in _read(S.A16S_DIR / "A16s.jsonl") if str(r["id"]) not in set(exclusions.get("a16s") or [])]
    a16s_scores = {m: json.loads((Path(v08_spark) / f"A16s-{m}-2b.json").read_text(encoding="utf-8")) for m in models}
    sweep = {m: S.sweep_rows(a16s_records, a16s_scores[m]["answers"], reading=A16S_READINGS[0], second=A16S_READINGS[1]) for m in models}
    for name, (x, y) in stand.items():
        mine = S.sweep_summary(S.seed_mean_rows(sweep[x], sweep[y]))
        theirs = ref["A16s"]["models"][name]["summary"]
        add(f"A16s {name} accuracy (total)", mine["total"]["accuracy"]["value"], theirs["total"]["accuracy"]["value"])
        for key in sorted(theirs["by_band"]):
            add(f"A16s {name} accuracy [{key}]", (mine["by_band"].get(key) or {}).get("accuracy", {}).get("value"), theirs["by_band"][key]["accuracy"]["value"])
    for m in models:
        add(f"A16s {m} accuracy (total)", S.sweep_summary(sweep[m])["total"]["accuracy"]["value"], ref["A16s"]["models"][m]["summary"]["total"]["accuracy"]["value"])
    p1 = pnum1_reading(a16s_records, a16s_scores, arms=stand, a="FP", b="F", n_boot=n_boot, seed=seed)
    tot = {k: ref["A16s"]["models"][k]["summary"]["total"]["accuracy"]["value"] for k in stand}
    add("P-num1 path: Δ = FP − F (stand-ins CN := FP, C := F)", p1["delta"]["value"], tot["FP"] - tot["F"])
    add("P-num1 path: CN arm accuracy (FP)", p1["delta"]["a"], tot["FP"])
    add("P-num1 path: C arm accuracy (F)", p1["delta"]["b"], tot["F"])
    for name in stand:
        by = ref["A16s"]["models"][name]["summary"]["by_band"]
        cells = [by[b]["accuracy"] for b in A16S_BAND_1_8]
        weighted = sum(c["value"] * c["n"] for c in cells) / sum(c["n"] for c in cells)
        add(f"P-num1 path: {name} 1–8 cm accuracy (v0.8's 1–3 and 3–8 cells, count-weighted)", p1["band_1_8"][name]["value"], weighted)
    # M8g and B8 flips through the P-cons path (C := FP, F := F)
    m8g_records = [r for r in _read(G8.M8G_FILE) if str(r["meta"]["item"]) not in set(exclusions.get("items") or [])]
    m8g_preds = {m: {kind: _read(Path(v08_spark) / f"{stem}-{m}-2b.predictions.jsonl") for kind, stem in M8G_KINDS.items()} for m in models}
    tags = json.loads((G8.B8_DIR / "tags.json").read_text(encoding="utf-8"))
    b8_records = {str(r["id"]): r for r in _read(G8.B8_FILE)}
    b8_scores = {m: json.loads((Path(v08_spark) / f"B8-{m}-2b.json").read_text(encoding="utf-8")) for m in models}
    units = v8.b8_units(tags, b8_records, exclude=exclusions.get("b8") or ())
    pc = pcons_reading(m8g_records, m8g_preds, units, b8_scores, arms={"C": stand["FP"], "F": stand["F"]}, a="C", b="F", n_boot=n_boot, seed=seed)
    want_f = {f"M8g {b}": ref["M8g"]["flip_F_by_bin"][b]["value"] for b in BIN_NAMES}
    want_f["B8"] = ref["B8"]["flips"]["pooled"]["F"]["value"]
    want_fp = {f"M8g {b}": ref["M8g"]["per_bin"]["FP"][b]["flip_rate"] for b in BIN_NAMES}
    want_fp["B8"] = ref["B8"]["flips"]["pooled"]["FP"]["value"]
    for name in CONS_SOURCES:
        e = pc["per_source"][name]
        add(f"P-cons path: F flip rate, {name}", e["b"], want_f[name])
        add(f"P-cons path: FP flip rate (stand-in C), {name}", e["a"], want_fp[name])
        add(f"P-cons path: FP − F, {name}", e["value"], want_fp[name] - want_f[name])
    add("P-cons path: F five-source mean", pc["b"], float(np.mean([want_f[n] for n in CONS_SOURCES])))
    add("P-cons path: FP five-source mean", pc["a"], float(np.mean([want_fp[n] for n in CONS_SOURCES])))
    add("P-cons path: Δ five-source mean (FP − F)", pc["value"], float(np.mean([want_fp[n] - want_f[n] for n in CONS_SOURCES])))
    # the guard path (C − F) with C := FP: v0.8's P1 (the DC mean), its CA guard, the T2–T4 band and severe errors, G2
    stand_c = {"v09Cs0": "v08FPs0", "v09Cs1": "v08FPs1"}
    paths = {}
    for s in OUTSIDE_SETS:
        for m in F_MODELS:
            paths[(m, s)] = Path(v08_spark) / f"{s}-{m}-2b.predictions.jsonl"
        for new_name, old_name in stand_c.items():
            paths[(new_name, s)] = Path(v08_spark) / f"{s}-{old_name}-2b.predictions.jsonl"
    std_sets, std_preds, _ = v8.load_standard(paths, allow_mismatch=False, notes=[])
    calls = v8.calls_of(std_sets, std_preds, OUTSIDE_SETS)
    runs = ARTIFACTS / "cloud" / "runs"
    pod_sets, pod_preds, _ = load_pods({"v09Cs0": runs / "v08p-s0-01-fetched" / "reports", "v09Cs1": runs / "v08p-s1-01-fetched" / "reports",
                                        "v08Fs0": F_PODS[0], "v08Fs1": F_PODS[1]}, allow_mismatch=False)
    guards = guard_readings(calls, pod_sets, pod_preds, a="C", b="F", n_boot=n_boot, seed=seed)
    for key in ("value", "low", "high"):
        add(f"guard path: outside DC mean FP − F {key} (v0.8's P1)", guards["DC"].get(key), ref["standard"]["P1"].get(key))
        add(f"guard path: outside CA mean FP − F {key} (v0.8's CA guard)", guards["CA"].get(key), ref["guards"]["CA"].get(key))
        add(f"guard path: T2–T4 balanced band FP − F {key}", (guards.get("band") or {}).get(key), (ref["guards"].get("band") or {}).get(key))
    for s in OUTSIDE_SETS:
        add(f"guard path: {s} ΔDC FP − F", ((guards["DC"].get("per_set") or {}).get(s) or {}).get("value"), ((ref["standard"]["P1"].get("per_set") or {}).get(s) or {}).get("value"))
    for key in ("a", "b"):
        add(f"guard path: T2–T4 severe errors {'FP' if key == 'a' else 'F'}", (guards.get("severe") or {}).get(key), (ref["guards"].get("severe") or {}).get(key))
    for key in ("diff", "low", "high"):
        add(f"guard path: G2 FP − F {key}", (guards.get("G2") or {}).get(key), (ref["guards"].get("G2") or {}).get(key))
    reproduction = {"rows": rows, "all_ok": all(r["ok"] for r in rows), "failing": [r for r in rows if not r["ok"]],
                    "max_abs_diff": max(abs(r["diff"]) for r in rows if r["diff"] is not None), "numbers": len(rows)}
    acc = m8g_accuracy_readings(m8g_records, m8g_preds, arms={"C": stand["FP"], "F": stand["F"]}, a="C", b="F", n_boot=n_boot, seed=seed)
    result = {"kind": "v09-verdict-pipeline-check", "script_version": SCRIPT_VERSION, "git": _git_sha(),
              "created": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
              "note": "PIPELINE CHECK ONLY — not the v0.9 verdict: v0.8's files with stand-in arms (P-num1: C := F, CN := FP; P-cons and the C − F "
                      "guards: C := FP, F := F)",
              "n_boot": n_boot, "seed": seed, "reproduction": reproduction,
              "side_readings": {"P-num1 path (FP − F on A16s)": {k: p1["delta"].get(k) for k in ("value", "low", "high", "n", "clusters")},
                                "P-num1 path 1–8 cm": p1["band_1_8"],
                                "P-cons path (FP − F)": {k: pc.get(k) for k in ("value", "low", "high", "a", "b", "group_seeds")},
                                "M8g accuracy FP − F": {k: (acc.get("all") or {}).get(k) for k in ("value", "low", "high", "n")},
                                "M8g accuracy N<=26 FP − F": {k: (acc.get("n26") or {}).get(k) for k in ("value", "low", "high", "n")}},
              "exclusions": {k: len(v or []) for k, v in exclusions.items() if k != "train_sha256"}}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    side = result["side_readings"]
    md = ["# v0.9 verdict helper — PIPELINE CHECK ONLY (not the v0.9 verdict)", "", result["note"] + ".", "",
          f"Reproduced: {reproduction['all_ok']} ({len(rows)} numbers against artifacts/reports/v08/verdict.json; max |diff| {reproduction['max_abs_diff']:.2e}).", "",
          f"Side readings on the stand-in arms: P-num1 path FP − F {_cid(side['P-num1 path (FP − F on A16s)'])}; 1–8 cm accuracy "
          + ", ".join(f"{k} {_fmt(v.get('value'))} (n {v.get('n')})" for k, v in side["P-num1 path 1–8 cm"].items())
          + f"; P-cons path FP − F {_cid(side['P-cons path (FP − F)'])} (FP {_fmt(pc.get('a'))}, F {_fmt(pc.get('b'))}; draw seeds {pc.get('group_seeds')}); "
          f"M8g accuracy FP − F {_cid(side['M8g accuracy FP − F'])}, N ≤ 26 {_cid(side['M8g accuracy N<=26 FP − F'])}.", "",
          "| reading | got | v0.8's verdict | diff |", "|---|---|---|---|"]
    md += [f"| {r['what']} | {_fmt(r['got'], 4)} | {_fmt(r['want'], 4)} | {'—' if r['diff'] is None else format(r['diff'], '.1e')} |" for r in rows]
    out.with_suffix(".md").write_text("\n".join(md) + "\n", encoding="utf-8")
    return result


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    from robo_jev.bench import a16s_v08, r14_v08, r16_v08

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    for name in ("c-s0-dir", "c-s1-dir", "cn-s0-dir", "cn-s1-dir"):
        parser.add_argument(f"--{name}", type=Path, default=None, help="fetched reports of the v0.9 run (<run>-fetched/reports; pods T2–T4, G2)")
    parser.add_argument("--f-s0-dir", type=Path, default=F_PODS[0], help="v0.8 F seed 0's fetched reports (default: v08f-s0-01)")
    parser.add_argument("--f-s1-dir", type=Path, default=F_PODS[1], help="v0.8 F seed 1's fetched reports (default: v08f-s1-01)")
    parser.add_argument("--spark-dir", type=Path, default=SPARK_DIR, help="the v0.9 chain's Spark outputs")
    parser.add_argument("--v08-spark-dir", type=Path, default=V08_SPARK, help="the v0.8 chain's Spark outputs (F, v0.7, the base; reused)")
    parser.add_argument("--pipeline-check", action="store_true", help="existing v0.8 files only, stand-in arms; never the verdict")
    parser.add_argument("--no-v09-recheck", action="store_true", help="read without overlap-v08.json / overlap-v09.json (tests and pipeline checks only)")
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--allow-mismatch", action="store_true", help="read even if a file fails its checks (never the frozen-file checks)")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)
    if args.pipeline_check:
        out = args.out or PIPELINE_OUT
        refuse_write(out)
        result = pipeline_check(n_boot=args.n_boot, seed=args.seed, out=out, v08_spark=args.v08_spark_dir)
        print(out.with_suffix(".md").read_text(encoding="utf-8"))
        return 0 if result["reproduction"]["all_ok"] else 1
    out = args.out or OUT
    refuse_write(out)
    if out.resolve() == PIPELINE_OUT.resolve():
        raise SystemExit("the verdict is not written into the pipeline-check folder")
    missing_dirs = [f"--{POD_ARGS[m].replace('_', '-')}" for m in V09_MODELS if getattr(args, POD_ARGS[m]) is None]
    if missing_dirs:
        raise SystemExit(f"required: {' '.join(missing_dirs)} (the four v0.9 runs' fetched reports)")
    missing = required_files(args.spark_dir, args.v08_spark_dir)
    if missing:
        raise SystemExit("missing files the rule reads:\n  " + "\n  ".join(missing))
    v8 = V8()
    notes: list[str] = []
    problems: list[str] = []
    n9g_frozen = load_frozen_dir(N9.check_frozen, Path(N9G_DIR), "N9g")
    a16s_frozen = load_frozen_dir(a16s_v08.check_frozen, a16s_v08.A16S_DIR, "A16s")
    b8_frozen, b8_problems = v8.check_b8_frozen()
    if b8_problems:
        raise SystemExit("B8's tags are not the frozen ones — refusing (no override):\n  " + "\n  ".join(b8_problems))
    exclusions, trained, overlap_notes = read_overlaps(required=not args.no_v09_recheck)
    notes += overlap_notes
    dirs = {m: Path(getattr(args, POD_ARGS[m])) for m in POD_MODELS}
    cks = checkpoints(dirs)
    problems += trained_input_problems(cks, trained)
    sets, preds, sources = v8.load_standard(standard_paths(args.spark_dir, args.v08_spark_dir), allow_mismatch=args.allow_mismatch, notes=notes)
    pod_sets, pod_preds, pod_sources = load_pods(dirs, allow_mismatch=args.allow_mismatch)
    left_out: list[str] = []
    common = {"spark_dir": args.spark_dir, "v08_spark": args.v08_spark_dir, "problems": problems, "notes": notes, "left_out": left_out}
    # the rule's gates
    n9g_file = Path(N9G_DIR) / "N9g.jsonl"
    n9g_scores = load_bench(N9G, n9g_file, N9.READINGS, (*V09_MODELS, *F_MODELS, "base"), cks, frozen=n9g_frozen, required=V09_MODELS, **common)
    n9g_records = [r for r in _read(n9g_file) if str(r["id"]) not in exclusions["n9g"]]
    a16s_file = a16s_v08.A16S_DIR / "A16s.jsonl"
    a16s_scores = load_bench(A16S, a16s_file, A16S_READINGS, MODELS, cks, frozen=a16s_frozen, required=V09_MODELS, **common)
    a16s_records = [r for r in _read(a16s_file) if str(r["id"]) not in exclusions["a16s"]]
    b8_scores = load_bench("B8", G8.B8_FILE, B8_READINGS, (*V09_MODELS, *F_MODELS, "base", "v07s0", "v07s1"), cks, frozen=b8_frozen,
                           required=(*ARMS["C"], *ARMS["F"]), **common)
    m8g_preds = load_m8g(cks, models=(*V09_MODELS, *F_MODELS, "base"), required=(*ARMS["C"], *ARMS["F"]), **common)
    problems += bundle_problems(b8_scores, (*ARMS["C"], *ARMS["F"]), set_name="B8")
    problems += bundle_problems(a16s_scores, (*ARMS["C"], *ARMS["CN"]), set_name="A16s")
    problems += bundle_problems(n9g_scores, (*ARMS["C"], *ARMS["CN"]), set_name="N9g")
    problems += condition_problems(a16s_scores, (*ARMS["C"], *ARMS["CN"]), set_name="A16s", serve=True)
    problems += condition_problems(n9g_scores, (*ARMS["C"], *ARMS["CN"]), set_name="N9g")
    for name, found in (("B8", b8_scores), ("A16s", a16s_scores), ("N9g", n9g_scores)):
        bundles = {str((d or {}).get("bundle_sha256"))[:12] for d in found.values()}
        if len(bundles) > 1:
            notes.append(f"{name}: the score files carry {len(bundles)} bundles ({sorted(bundles)}); the rule's comparisons were checked above.")
    # reported parts (a freeze failure refuses only that part)
    a16 = s16 = r14 = jev = None
    r16_problems = r16_v08.check_frozen(r16_v08.R16_DIR)
    if r16_problems:
        a16 = s16 = {"refused": r16_problems}
        notes.append("A16 / S16's frozen files changed — their readings are refused (the rest of the verdict stands): " + "; ".join(r16_problems))
    else:
        r16_frozen = json.loads((r16_v08.R16_DIR / "frozen-before-scoring.json").read_text(encoding="utf-8"))
        rec = {name: [r for r in _read(r16_v08.R16_DIR / f) if str(r["id"]) not in exclusions["s16" if name == S16 else "a16"]] for name, f in R16_FILES.items()}
        scores = {name: load_bench(name, r16_v08.R16_DIR / R16_FILES[name], R16_READINGS[name], MODELS, cks, frozen=r16_frozen, **common) for name in R16_FILES}
        stored = {name: json.loads((r16_v08.R16_DIR / f"{name}-stored-v07.json").read_text(encoding="utf-8")).get("answers") or {} for name in (A16, S16)}
        a16 = {"records": {A16: rec[A16], A16Y: rec[A16Y]}, "scores": {A16: scores[A16], A16Y: scores[A16Y]}, "stored": stored}
        s16 = {"records": rec[S16], "scores": scores[S16], "stored": stored[S16]}
    r14_problems = r14_v08.check_frozen(r14_v08.R14_DIR)
    if r14_problems:
        r14 = {"refused": r14_problems}
        notes.append("R14's frozen files changed — its readings are refused (the rest of the verdict stands): " + "; ".join(r14_problems))
    else:
        r14_frozen = json.loads((r14_v08.R14_DIR / "frozen-before-scoring.json").read_text(encoding="utf-8"))
        r14 = {"tags": json.loads((r14_v08.R14_DIR / "tags.json").read_text(encoding="utf-8")),
               "record_ids": [str(r["id"]) for r in _read(r14_v08.R14_DIR / R14_INPUT) if str(r["id"]) not in exclusions["r14"]],
               "scores": load_bench("R14", r14_v08.R14_DIR / R14_INPUT, R14_READINGS, MODELS, cks, frozen=r14_frozen, **common)}
    jev_set, jev_answers = G8.JEV_DIR / "jev-set.jsonl", G8.JEV_DIR / "jev-answers.predictions.jsonl"
    if jev_set.is_file() and jev_answers.is_file():
        jev = {"records": _read(jev_set), "jev_preds": _read(jev_answers), "scores": load_bench("Jev", jev_set, ("single-s1",), MODELS, cks, **common)}
    if left_out:
        notes.append("Score files left out (a model the rule does not read; the file fails its checks): " + "; ".join(left_out) + ".")
    if problems and not args.allow_mismatch:
        raise SystemExit("files fail their checks (use --allow-mismatch to read anyway; never the frozen-file checks):\n  " + "\n  ".join(problems[:40]))
    _log("computing readings")
    m8g_records = [r for r in _read(G8.M8G_FILE) if str(r["meta"]["item"]) not in exclusions["items"]]
    b8_records = {str(r["id"]): r for r in _read(G8.B8_FILE)}
    tags = json.loads((G8.B8_DIR / "tags.json").read_text(encoding="utf-8"))
    computed = compute(standard_sets=sets, standard_preds=preds, pod_sets=pod_sets, pod_preds=pod_preds,
                       a16s={"records": a16s_records, "scores": a16s_scores}, n9g={"records": n9g_records, "scores": n9g_scores},
                       m8g={"records": m8g_records, "preds": m8g_preds}, b8={"tags": tags, "records": b8_records, "scores": b8_scores, "exclude": sorted(exclusions["b8"])},
                       a16=a16, s16=s16, r14=r14, jev=jev, n_boot=args.n_boot, seed=args.seed)
    notes += ["Seed means: an arm's per-item value = the mean of its seeds' values on the items both seeds have (call view: the mean of the 1/L-weighted "
              "listings); arm comparisons pair the items both arms have.",
              "Reused v0.8 files: F's (and v0.7's / the base's) Spark files of the same sets and conditions are read from the v0.8 chain; only N9g was "
              "scored for F and the base in the v0.9 chain.",
              "P-num1: unit = an A16s axis question, cluster = its request; CN's 1–8 cm accuracy = A16s bands 1–3 and 3–8 (1 <= |d| < 8 cm), both signs, "
              "formats and axes pooled.",
              "P-num2: N9g, one question per record; unit = cluster = the record.",
              "P-cons: the four M8g bins resampled jointly by item (draw seed = seed), B8 by request (seed + 1); the five differences averaged per draw.",
              "Guards (both comparisons): DC and CA as v0.8's P1 path (per-set seeds = seed + index); T2–T4 stratified paired; G2 as v0.6's.",
              "Inclusive bounds compare with a 1e-12 tolerance; strict bounds as they are."]
    result = {"kind": "v09-verdict", "script_version": SCRIPT_VERSION, "git": _git_sha(), "created": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
              "rule_text": RULE, "n_boot": args.n_boot, "seed": args.seed, "pipeline_check": None, **computed,
              "sources": {**sources, **pod_sources}, "frozen": {"N9g": n9g_frozen, "A16s": a16s_frozen, "B8": b8_frozen}, "problems": problems, "left_out": left_out,
              "exclusions": {k: sorted(v) for k, v in exclusions.items()}, "trained_inputs": trained, "notes": notes,
              "inputs": {k: str(v) for k, v in vars(args).items() if isinstance(v, Path)}}
    figure = a16s_figure(computed["readings"].get(A16S))
    if figure is not None:
        result["figures"] = {A16S: str(out.with_name(A16S_SVG))}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    if figure is not None:
        out.with_name(A16S_SVG).write_text(figure, encoding="utf-8")
    md = render(result)
    out.with_suffix(".md").write_text(md, encoding="utf-8")
    print(md.split("## Reported, not deciding")[0])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
