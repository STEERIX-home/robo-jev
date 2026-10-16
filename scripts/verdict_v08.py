#!/usr/bin/env python
"""Apply the PRE-DECLARED v0.8 VERDICT RULE (progress.md, written 2026-10-16 ~12:43 KST, before any v0.8 data existed) mechanically.

    python scripts/verdict_v08.py --f-s0-dir <F seed-0 run>-fetched/reports --f-s1-dir … --fp-s0-dir … --fp-s1-dir …
        # → artifacts/reports/v08/verdict.{json,md}
    python scripts/verdict_v08.py --pipeline-check     # existing v0.6 / v0.7 short-form files only; not the verdict (last part below)

**Arms and seed means.** F = v0.8F-s0 + v0.8F-s1 (format package), FP = v0.8FP-s0 + v0.8FP-s1 (format + H-prior loss weights), v0.7 = v0.7-s0 +
v0.7-s1. Per question an arm's call distribution is the mean of its two seeds' 1/L-weighted listings (:func:`verdict_v06.arm_calls`); every
other per-item reading of an arm is the mean of its seeds' item values. Paired cluster bootstrap, 2,000 draws, seed 0
(:func:`robo_jev.bench.system1._draws`).

* **P1 — short-form outside DC** (H-prior, FP − F): per set s ∈ (A4, T9, T8b with B2's truth classes) Δ_s = DC(FP arm) − DC(F arm) on the
  questions both arms answered (call view of v0.5; cluster = the one-fact pair). ONE draw-wise bootstrap: each set is resampled by its own
  clusters with its own draw seed (seed + the set's index in (A4, T9, T8b) — the precedent of v0.6's author gap, which drew A4 and the H-set
  independently; the arms are paired within a set), and the three Δ are averaged per draw. PASS iff the mean ≤ −0.05 (point) AND the upper
  bound of the mean < 0.
* **P2 — gate B8** (fresh bench-shaped requests built by robo-jev-web's own builders, tags frozen before scoring): a unit = a tagged question
  of plates ``move``, café ``intent`` / ``attend`` / ``request`` or delivery ``manoeuvre`` (pooled; the tube cell's ``work`` lists are reported
  only); cluster = the request. Per model the top option = the argmax of the balanced-order answer (reading ``balanced-s1``; ties → the earlier
  option of the request); passive-top = the top option is tagged PASSIVE; planner agreement = the top option is the request's planner
  recommendation (questions with one). An arm's unit value = the mean of its seeds' indicators. PASS iff FP − F passive-top share ≤ −0.15
  (point) AND FP − F planner agreement ≥ −0.05 (point).
* **OUTCOME**: SUPPORTED iff P1 and P2; PARTIAL iff exactly one; REJECTED iff neither.
* **GUARDS (FP − F)** — reported with pass / fail; the rule does not let them change the outcome label: outside CA, the mean of the three
  sets' differences ≥ −0.05 (point), and GAP ≥ −0.03 (point; read like CA, as the mean of the three sets' differences); T2–T4 balanced band
  (pods; stratified paired bootstrap, as v0.7's home reading) lower bound > −0.05, and severe errors FP ≤ F + 0.01 (arm means); G2 ≥ −0.02
  (point) with lower bound > −0.045 (:func:`verdict_v06.g2_arm_comparison`).
* **FORMAT (arm F, gate M8g)** PASS iff (a) in each of the bins 27–60 and 61–255, F's top-1 accuracy on listings whose correct option carries
  a two-letter code ≥ 0.8 × its top-1 accuracy on listings with a single-letter code (points; compared as acc_two ≥ 0.8 · acc_single) AND (b)
  F's accuracy at N ≤ 26 ≥ v0.7's − 0.02 (point). **O-F (release guard)**: F − v0.7 short-form outside DC, the mean of the three sets, ≤ +0.02
  (point).
* **NEXT STEPS** as pre-declared (SUPPORTED; P1 fails; P2 fails while P1 passes; FORMAT fails), with the consistency-loss condition as
  amended before any v0.8 data (progress.md, "AMENDMENT TO THE v0.8 NEXT-STEP CONDITION", 2026-10-16 ~13:39 KST): the order-consistency loss
  goes into v0.9 iff F's flip rate between two order seeds is > 0.20 in ANY source — each M8g bin (2–10, 11–26, 27–60, 61–255; record
  order vs the seeded order seed 2, listing-0 records), B8 (every tagged question, single order seed 1 vs seed 2) and, when scored, R14 —
  robo-jev-web's collected bench requests (``R14-<tag>.json``; serve's three orders spaces-0/1/2; a flip = the top option is not the same in
  all three; :mod:`robo_jev.bench.r14_v08`). It replaces "in bins ≥ 27" and is
  applied as written: not conditional on FORMAT passing (FORMAT's own step applies as well when it fails); the output states FORMAT's status
  and what the original wording (which required FORMAT to pass) would give. A source that cannot be read is listed as unread, never counted
  as ≤ 0.20 (TRIGGERED / NOT TRIGGERED / UNREAD). **RELEASE**: the user decides (arm and seed).

Inclusive bounds (≤, ≥) are compared with a 1e-12 tolerance, so float rounding of an exact tie never flips an inclusive bound; strict bounds
(<, >) are compared as they are.

**Reported, never deciding**: per option-count bin (2–10, 11–26, 27–60, 61–255) on M8g per model and arm — accuracy, top-label ECE (10
equal-width confidence bins), NLL of the key (= KL to the one-hot key; M8g has no teacher distribution), flip rate between the two order seeds
(the record order against the package's seeded order, seed 2, on the same listing-0 records; a flip = a different top option), position RStd
(the standard deviation, ddof 0, over blocks of 8 consecutive positions of the block's top-1 accuracy; blocks with ≥ 5 listings); the per-bin
flip rate on B8 (single order seed 1 against seed 2; pooled, per environment and per bin); R14 (reported only — it enters neither P1, P2 nor
FORMAT): flip rate across serve's three orders, pick-rate RStd, passive-top share, planner agreement, top-probability spread and the web's
override clearance, per environment × question, per environment and pooled; a label-free
pick-rate RStd for every source (per block of 8 shown positions of the seed-1 single order — M8g: the record order — the ratio of the block's
share of top-1 picks to its share of shown options; the standard deviation, ddof 0, over blocks shown by ≥ 5 questions; ≈ 0 = no position
preference; an arm pools both seeds' picks); KL(teacher ‖ model) on the choice questions of the standard sets whose labels are not
one-hot (bin 2–10); single against balanced on M8g bins ≤ 60; Jev top-1 agreement (the comparison set; not sealed); the 20 storage-bin
probe cases of rel071; the real long lists (tube cell ``work``, plates ``move``); the teacher label spread; the post-hoc prior-removal
estimate (``artifacts/v08/posthoc/estimate.json``).

**A16 and S16** (reported only — robo-jev-web's requests of 2026-10-16, :mod:`robo_jev.bench.r16_v08`; they enter neither P1, P2, FORMAT nor
the amended flip condition, whose third source stays R14, the 10-14 requests): A16 — the arm's servo decisions (single order seed 1, serve's
conditions): rule agreement (total, per axis, per axis × signed band), sign errors per axis, the stay rate per band and on 0.2–3 cm, the
boundary sensitivity (±0.2 read as stay), v0.7's stored endpoint answers as a reference row, and the y-phrase ablation (y without " (up is
plus)") next to the same model's y readings; S16 — soccer, movement (15 options) and gaze (6 options) separately: the flip rate across serve's
three orders, the top-probability spread and the pick-rate RStd, with v0.7's stored answers' spread as a reference.

**Exclusions**: ``artifacts/v08/gates/overlap-v08.json`` (``scripts/v08_gates_build.py recheck-v08``, run on v0.8's training input before
the gates are opened) lists M8g items and B8 requests that share text with v0.8's training input; they are left out of every reading. The
verdict refuses to read the gates without it (``--no-v08-recheck`` only for tests and pipeline checks).

**Pipeline check** (``--pipeline-check``; writes ``artifacts/scratch/v08b/verdict/``, says it is not the verdict): P1's and the CA / GAP
guard's path on existing files with the arms relabelled — F := v0.6-s0 + v0.6-s1, FP := v0.7-s0 + v0.7-s1 (their Spark short-form files on
A4, T9, T8b) — reproduces v0.7's verdict (``artifacts/reports/v07/verdict.json``): the per-set v0.7 − v0.6 DC with its interval (each set
drawn with seed 0, as v0.7 did) and every seed's and arm's DC, CA and GAP. P1's draw-wise mean of these stand-in arms is printed as a side
reading.
"""

from __future__ import annotations

import argparse
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

SCRIPT_VERSION = "v08-verdict-1"
RULE = {
    "written": "2026-10-16 ~12:43 KST, BEFORE any v0.8 data exists (progress.md: PRE-DECLARED v0.8 VERDICT RULE); user: '좋아 추천대로 모두 진행' "
               "-> two arms, consistency loss deferred, Jev agreement calls allowed; training consent given earlier, '2,3번은 동의'",
    "DESIGN": "one training input, train-v08 = v0.7's input with one transform added, the FORMAT package. Choice questions take 2-255 options, N "
              "log-uniform, with enough large-N (61-255) examples. The correct option's position is spread over every code. A share of choice "
              "questions show codes drawn at random from the 255-code pool, so every code is trained as an answer. Records are transformed, never "
              "added. Labels are v0.7's. The code pool = the bundle's CODES[:255] (A-Z, then the first 229 two-letter codes of the 466 list). At "
              "inference, codes are assigned in order (A-Z first), so prompts with <= 26 options are unchanged.",
    "ARMS": "F = format only; FP = format + H-PRIOR LOSS WEIGHTS: within each question family (source x question), every label class carries "
            "equal total weight, with mean weight 1 per family and weights capped (the cap fixed and reported before training). CLASSES: yes/no "
            "and score: the label's band (cut at 1/3 and 2/3); level questions: the best level; action questions: the best option's kind (stop / "
            "wait / hold, continue, other move; rule fixed in code and tested); other choice questions: the answer key. Positions and codes are "
            "not weighted (the format package handles them in both arms).",
    "RECIPE": "seeds 0 and 1 per arm; v0.7's recipe otherwise; seed means; paired cluster bootstrap, 2,000 draws, seed 0.",
    "P1": "short-form outside DC: the mean of the three set-level differences (A4, T9, and T8b with B2 truth) <= -0.05, with the upper bound of "
          "that mean < 0. Each set is resampled by clusters, and the three are averaged per draw.",
    "P2": "fresh bench-shaped requests, gate B8: built by robo-jev-web's own builders from fresh states; plates move, cafe intent / attend / "
          "request and delivery manoeuvre pooled; a question is the unit and a request the cluster; tags frozen before scoring; the rel07 check's "
          "30 requests are excluded. Passive-top share <= -0.15 (point) AND planner agreement >= -0.05 (point).",
    "OUTCOME": "SUPPORTED iff P1 and P2; PARTIAL iff exactly one; REJECTED iff neither.",
    "GUARDS (FP - F)": "outside CA, mean of the three sets >= -0.05 (point), and GAP >= -0.03 (point); T2-T4 balanced band: lower bound > -0.05, "
                       "and severe errors <= F's + 0.01 (arm means); G2 >= -0.02 (point), with lower bound > -0.045.",
    "FORMAT READING": "(arm F, against v0.7 where stated), on gate M8g: a sealed many-option set. Sources: a licensed MC source used in neither "
                      "training, M1 nor M1g, plus fresh physics items. N in {6, 18, 40, 120, 255}, the same item across N, correct positions "
                      "spread over all codes including two-letter ones. FORMAT PASS iff both: in each of the bins 27-60 and 61-255, F's top-1 "
                      "accuracy on items whose correct option carries a two-letter code >= 0.8 x its top-1 accuracy on items with a "
                      "single-letter code (point); F's accuracy at N <= 26 >= v0.7's - 0.02 (point).",
    "RELEASE GUARD (O-F)": "F - v0.7 short-form outside DC, mean of the three sets, <= +0.02 (point).",
    "REPORTED, NOT DECIDING": "per bin (2-10, 11-26, 27-60, 61-255): accuracy, top-label ECE, KL to the teacher where teacher distributions "
                              "exist, flip rate between two order seeds, position RStd; single against balanced (bins <= 60); Jev top-1 "
                              "agreement on a separate non-sealed comparison set, via OpenRouter with data collection denied; the 20 "
                              "storage-bin probe cases from rel071; real long lists: tubes `work` (~46) and plates `move`; teacher label "
                              "spread (action labels: median top probability ~ 0.61; measured 2026-10-16); a post-hoc prior-removal estimate "
                              "on v0.7's predictions, for the expected size of P1.",
    "PRE-DECLARED NEXT STEPS": "SUPPORTED: the weights stay; next is the remaining outside gap and plausible many-candidate choice. P1 fails: "
                               "the 'safe' default is not a label-prior effect, so next is System-0 margins as input (C) or real data. P2 "
                               "fails while P1 passes: plausible same-family action candidates. FORMAT fails: more large-N examples and a "
                               "higher random-code share. FORMAT passes but F's flip rate in bins >= 27 > 0.20: the consistency loss (KL/JS "
                               "over permuted copies) goes into v0.9.",
    "PRE-DECLARED NEXT STEPS - superseded part": "the last condition above ('FORMAT passes but F's flip rate in bins >= 27 > 0.20') is "
                                                  "superseded by the AMENDMENT TO THE v0.8 NEXT-STEP CONDITION below; its text is kept as written.",
    "AMENDMENT TO THE v0.8 NEXT-STEP CONDITION": "(written 2026-10-16 ~13:39 KST, BEFORE any v0.8 data exists; relayed by the user; the verdict "
                                                 "criteria - P1, P2, the guards, FORMAT PASS, O-F - are unchanged). WHY: robo-jev-serve / "
                                                 "robo-jev-web report that on v0.7's real bench requests, even questions with <= 26 options change "
                                                 "their top-1 with the option order in 0.5-1.0 of cases: plates move 11/14, delivery offset 13/17, "
                                                 "humanoid gaze 12/20, cafe say 10/20. NEW CONDITION: the order-consistency loss (KL/JS over "
                                                 "permuted copies) goes into v0.9 if F's flip rate between two order seeds is > 0.20 in ANY of: the "
                                                 "M8g bins 2-10, 11-26, 27-60 and 61-255; the fresh bench-shaped gate B8; robo-jev-web's collected "
                                                 "bench requests, if the user provides them before scoring. This replaces 'in bins >= 27'. "
                                                 "REPORTED: flip rate and RStd for all of these.",
    "R14 (controller, 2026-10-16)": "robo-jev-web's collected requests are in artifacts/bench/robojev-requests-2026-10-14/ (zip sha256 "
                                    "c344da09...). After dedup: lab 116, cafe 118, delivery 207, humanoid 119, forklift 640. They were handed to v0.8b "
                                    "as the reported set R14, measured under serve's conditions: 1-option choices split out at p = 1, and forklift / "
                                    "humanoid `offset` asked as a 5-level choice. (R14 is the amended condition's third source - the collected bench "
                                    "requests - and enters neither P1, P2 nor FORMAT.)",
    "REPORTED, NOT DECIDING - additions (controller, relayed 2026-10-16 ~14:08 KST, before any v0.8 scoring; P1, P2, FORMAT and every bound "
    "unchanged)": "(1) the passive-top share split into its two parts on B8 and on R14 - hold (stop / wait / hold / do nothing / hand to a "
                  "person) and continue (continue unchanged / as planned / keep going), by robo_jev.bench.r14_v08.passive_kind of each option's "
                  "tag rule; B8: per model and arm, pooled over the P2 units and per env, and FP - F of each with request-cluster intervals next "
                  "to P2 (reported; P2 still reads the frozen PASSIVE tag); R14: per model and arm, per env and per env x question, a mean over "
                  "the three orders. Why: v0.8a's H-prior action classes are three - stop / wait / hold, down-weighted (share 0.45 -> 0.33); "
                  "continue, slightly up-weighted (0.29 -> 0.33); other move, up-weighted (0.27 -> 0.33) - while the PASSIVE tag, following "
                  "rel07, counts 'continue unchanged' as passive. (2) M8g per source (general = MMLU against physics = PHYS) in each bin, per "
                  "model and arm: accuracy, acc_two, acc_single, the FORMAT margin acc_two - 0.8 x acc_single, flip rate and NLL (FORMAT "
                  "unchanged). Note: in training, physics questions reach at most about 57-139 options and only general items reach 255.",
    "A16 and S16 (controller, relayed for task v0.8c on 2026-10-16; files built and frozen 15:12 KST)": "two REPORTED sets of robo-jev-web's "
        "2026-10-16 requests - A16 (the arm: 90 servo decisions, six questions each; the rule written in every x / y / z question) and S16 "
        "(soccer: 55 requests, six players' movement (15 options) and gaze (6 options)). They enter neither P1, P2, FORMAT nor the amended flip "
        "condition - its third source stays R14 (10-14) only. A16: rule agreement (total, per axis, per axis x signed band), sign errors per "
        "axis, the stay rate per band and on small distances (0.2-3 cm), the boundary sensitivity (+-0.2 read as stay), the reference row 'v0.7 "
        "endpoint (stored)', and the y-phrase ablation (y agreement and sign errors without ' (up is plus)' next to the same model's with it; "
        "serve's reported v0.7 change: agreement 27 -> 50 of 90, sign errors 27 -> 4). S16: per group the flip rate across serve's three orders, "
        "the top-probability spread and the label-free pick-rate RStd, with v0.7's stored answers' spread as a reference.",
    "RELEASE": "the user decides (arm and seed). The bundle takes CODES[:255] with MAX_OPTIONS = 255; requests with <= 26 options keep "
               "identical prompts; format, weights.class and prompt v0c-p3 are unchanged.",
    "CONTROLLER'S PRIOR": "FORMAT likely passes; P1 uncertain; P2 more likely than P1.",
}

# --- the rule's bounds (never changed) ---
P1_POINT, P1_HIGH = -0.05, 0.0
P2_PASSIVE, P2_PLANNER = -0.15, -0.05
CA_POINT, GAP_POINT = -0.05, -0.03
BAND_LOW, SEVERE_SLACK = -0.05, 0.01
G_POINT, G_LOW = -0.02, -0.045
FORMAT_RATIO, FORMAT_N26 = 0.8, -0.02
OF_POINT = 0.02
FLIP_LIMIT = 0.20
#: Inclusive bounds are compared with this tolerance (module docstring).
TOL = 1e-12

SEEDS = (0, 1)
ARMS = {"F": ("v08Fs0", "v08Fs1"), "FP": ("v08FPs0", "v08FPs1"), "v07": ("v07s0", "v07s1")}
V08_MODELS = ("v08Fs0", "v08Fs1", "v08FPs0", "v08FPs1")
MODELS = (*V08_MODELS, "v07s0", "v07s1", "base")
LABELS = {"v08Fs0": "v0.8F seed 0", "v08Fs1": "v0.8F seed 1", "v08FPs0": "v0.8FP seed 0", "v08FPs1": "v0.8FP seed 1", "v07s0": "v0.7 seed 0",
          "v07s1": "v0.7 seed 1", "base": "untrained Qwen3.5-2B", "F": "F (arm, seed mean)", "FP": "FP (arm, seed mean)",
          "v07": "v0.7 (arm, seed mean)", "v06s0": "v0.6 seed 0", "v06s1": "v0.6 seed 1"}
SPARK_TAGS = {m: f"{m}-2b" for m in MODELS}
P1_SETS = ("A4", "T9", "T8b")
CALL_SETS = ("A4", "T9", "T8b", "Hset", "B1")
HOME = ("T2", "T3", "T4")
POD_SETS = ("T2", "T3", "T4", "G2")
POD_ARGS = {"v08Fs0": "f_s0_dir", "v08Fs1": "f_s1_dir", "v08FPs0": "fp_s0_dir", "v08FPs1": "fp_s1_dir"}
P2_ENVS = tuple(G8.P2_TARGETS)
FORMAT_BINS = ("27-60", "61-255")
BIN_NAMES = tuple(name for name, _, _ in G8.BINS)
B8_READINGS = ("balanced-s1", "single-s1", "single-s2")
#: The amended consistency-loss condition's sources (F's flip rate between two order seeds); R14 joins them when its score files exist.
CONSISTENCY_SOURCES = tuple(f"M8g {name}" for name, _, _ in G8.BINS) + ("B8",)
R14 = "R14"
#: R14 (robo-jev-web's collected bench requests, :mod:`robo_jev.bench.r14_v08`): serve's three order seeds — 0, 1, 2 spaces after the state.
R14_READINGS = ("spaces-0", "spaces-1", "spaces-2")
R14_INPUT = "R14-v08-sample.jsonl"
#: A16 / S16 (robo-jev-web's 2026-10-16 requests; reported only): the scored files and each score file's readings.
A16, A16Y, S16 = "A16", "A16y", "S16"
A16_READING = "single-s1"
R16_FILES = {A16: "A16.jsonl", A16Y: "A16-y-ablation.jsonl", S16: "S16.jsonl"}
R16_READINGS = {A16: (A16_READING,), A16Y: (A16_READING,), S16: R14_READINGS}
SERVE_A16_ABLATION = "serve's reported v0.7 change with ' (up is plus)' removed from the y question: agreement 27 -> 50 of 90, sign errors 27 -> 4"
R16_SEPARATE = ("A16 and S16 (robo-jev-web's 2026-10-16 requests) are reported separately: they enter neither P1, P2, FORMAT nor this condition, "
                "whose third source stays R14 (the 10-14 requests)")
#: The shown position of an answer code (A = 0, …, Z = 25, then the runner's two-letter codes).
CODE_POSITION = {code: i for i, code in enumerate(G8.RUNNER_CODES)}
M8G_KINDS = {"order": "M8g", "seed2": "M8g-seed2", "balanced": "M8g-balanced"}
#: What each M8g prediction file must have been scored with (the summary's config).
M8G_CONDITIONS = {"order": {"order_seed": None, "orders": "seeded"}, "seed2": {"order_seed": 2, "orders": "seeded"},
                  "balanced": {"order_seed": None, "orders": "counterbalanced"}}
RSTD_BLOCK, RSTD_MIN = 8, 5
ECE_BINS = 10
#: The rule's reported fact (measured 2026-10-16): the teacher's action labels are spread.
TEACHER_SPREAD = "teacher label spread: action labels' median top probability ~ 0.61 (measured 2026-10-16; the reason answers are diffuse)"
#: M8g per source (reported): the training range of each source's option counts (relayed 2026-10-16 ~14:08 KST).
PER_SOURCE_NOTE = "in training, physics questions reach at most about 57-139 options and only general items reach 255"
#: robo-jev-web runs a model's choice instead of the planner's only at p >= 0.55 and >= 0.15 above the recommendation (context, as rel07).
HOLD_P, HOLD_AHEAD = 0.55, 0.15

ARTIFACTS = REPO / "artifacts"
SPARK_DIR = ARTIFACTS / "reports" / "v08" / "spark"
V07_SPARK = ARTIFACTS / "reports" / "v07" / "spark"
V06_SPARK = ARTIFACTS / "reports" / "v06" / "spark"
V07_CKPT = {0: ARTIFACTS / "cloud" / "runs" / "v07-s0-01-fetched" / "runs" / "v07-s0-2b" / "checkpoint.pt",
            1: ARTIFACTS / "cloud" / "runs" / "v07-s1-01-fetched" / "runs" / "v07-s1-2b" / "checkpoint.pt"}
OUT = ARTIFACTS / "reports" / "v08" / "verdict.json"
PIPELINE_OUT = ARTIFACTS / "scratch" / "v08b" / "verdict" / "verdict-pipeline.json"
TAGS_FILE, FROZEN_FILE = G8.B8_DIR / "tags.json", G8.B8_DIR / "frozen-before-scoring.json"
JEV_SET, JEV_ANSWERS = G8.JEV_DIR / "jev-set.jsonl", G8.JEV_DIR / "jev-answers.predictions.jsonl"
PROBE_ROWS = G8.PROBES_DIR / "bins.jsonl"
POSTHOC_FILE = G8.POSTHOC_DIR / "estimate.json"
#: Outputs never go here.
NEVER_WRITE = (ARTIFACTS / "v08", ARTIFACTS / "simgen", ARTIFACTS / "t8", ARTIFACTS / "t9", ARTIFACTS / "b1", ARTIFACTS / "b2", ARTIFACTS / "v06",
               ARTIFACTS / "v07", ARTIFACTS / "iface", ARTIFACTS / "reports" / "v04", ARTIFACTS / "reports" / "v05", ARTIFACTS / "reports" / "v06",
               ARTIFACTS / "reports" / "v07", ARTIFACTS / "cloud")


@functools.lru_cache(maxsize=None)
def _script(name: str):
    spec = importlib.util.spec_from_file_location(f"v08_{Path(name).stem}", REPO / "scripts" / name)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def V4():
    """scripts/verdict_v04.py — pair clusters, paired differences, prediction-file checks."""
    return _script("verdict_v04.py")


def V5():
    """scripts/verdict_v05.py — the call view (truth classes, 1/L call weights, call metrics and paired differences)."""
    return _script("verdict_v05.py")


def V6():
    """scripts/verdict_v06.py — seed means (arm_calls, arm_values), the G2 arm reading, T8b with B2, the set paths."""
    return _script("verdict_v06.py")


def V7():
    """scripts/verdict_v07.py — the untrained base's Spark files."""
    return _script("verdict_v07.py")


def _log(message: str) -> None:
    print(f"[verdict-v08 {dt.datetime.now().strftime('%H:%M:%S')}] {message}", file=sys.stderr, flush=True)


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


def p1_check(p1: Mapping[str, Any] | None) -> dict[str, Any]:
    """P1: the mean of the three sets' FP − F DC ≤ −0.05 (point) AND the upper bound of the mean < 0 (a missing reading fails)."""
    p1 = p1 or {}
    checks = {"mean_le_-0.05": _le(p1.get("value"), P1_POINT), "mean_high_lt_0": _lt(p1.get("high"), P1_HIGH)}
    return {"value": _num(p1.get("value")), "high": _num(p1.get("high")), "checks": checks, "pass": all(checks.values())}


def p2_check(p2: Mapping[str, Any] | None) -> dict[str, Any]:
    """P2: FP − F passive-top share ≤ −0.15 (point) AND FP − F planner agreement ≥ −0.05 (point) on B8 (a missing reading fails)."""
    p2 = p2 or {}
    passive, planner = (p2.get("passive_top") or {}).get("value"), (p2.get("planner") or {}).get("value")
    checks = {"passive_top_le_-0.15": _le(passive, P2_PASSIVE), "planner_ge_-0.05": _ge(planner, P2_PLANNER)}
    return {"passive_top": _num(passive), "planner": _num(planner), "checks": checks, "pass": all(checks.values())}


def outcome_of(p1_pass: bool, p2_pass: bool) -> str:
    """SUPPORTED iff P1 and P2; PARTIAL iff exactly one; REJECTED iff neither."""
    return {2: "SUPPORTED", 1: "PARTIAL", 0: "REJECTED"}[int(bool(p1_pass)) + int(bool(p2_pass))]


def guard_checks(g: Mapping[str, Any] | None) -> dict[str, Any]:
    """The guards (FP − F): outside CA mean ≥ −0.05 and GAP mean ≥ −0.03 (points); T2–T4 balanced band lower bound > −0.05 and severe errors
    FP ≤ F + 0.01 (arm means); G2 ≥ −0.02 (point) with lower bound > −0.045. A missing reading fails. They do not change the outcome label."""
    g = g or {}
    ca, gap, band = g.get("CA") or {}, g.get("GAP") or {}, g.get("band") or {}
    severe, g2 = g.get("severe") or {}, g.get("G2") or {}
    sa, sb = _num(severe.get("a")), _num(severe.get("b"))
    checks = {"outside_CA_mean_ge_-0.05": _ge(ca.get("value"), CA_POINT), "outside_GAP_mean_ge_-0.03": _ge(gap.get("value"), GAP_POINT),
              "T2-T4_band_low_gt_-0.05": _gt(band.get("low"), BAND_LOW),
              "T2-T4_severe_FP_le_F_plus_0.01": sa is not None and sb is not None and sa <= sb + SEVERE_SLACK + TOL,
              "G2_point_ge_-0.02": _ge(g2.get("diff"), G_POINT), "G2_low_gt_-0.045": _gt(g2.get("low"), G_LOW)}
    return {"checks": checks, "pass": all(checks.values())}


def format_check(fmt: Mapping[str, Any] | None) -> dict[str, Any]:
    """FORMAT PASS iff in each of the bins 27–60 and 61–255 F's accuracy on two-letter-coded listings ≥ 0.8 × its accuracy on single-letter-coded
    listings (points), AND F's accuracy at N ≤ 26 ≥ v0.7's − 0.02 (point). A missing reading fails."""
    fmt = fmt or {}
    checks = {}
    for name in FORMAT_BINS:
        entry = (fmt.get("bins") or {}).get(name) or {}
        two, single = _num((entry.get("acc_two") or {}).get("value")), _num((entry.get("acc_single") or {}).get("value"))
        checks[f"{name}_two_ge_0.8_x_single"] = two is not None and single is not None and two >= FORMAT_RATIO * single - TOL
    checks["N<=26_F_minus_v07_ge_-0.02"] = _ge((fmt.get("n26") or {}).get("value"), FORMAT_N26)
    return {"checks": checks, "pass": all(checks.values())}


def of_check(of: Mapping[str, Any] | None) -> dict[str, Any]:
    """O-F (release guard): F − v0.7 short-form outside DC, the mean of the three sets, ≤ +0.02 (point)."""
    of = of or {}
    ok = _le(of.get("value"), OF_POINT)
    return {"value": _num(of.get("value")), "checks": {"mean_le_+0.02": ok}, "pass": ok}


NEXT = {
    "SUPPORTED": "the weights stay; next is the remaining outside gap and plausible many-candidate choice",
    "P1 fails": "the 'safe' default is not a label-prior effect, so next is System-0 margins as input (C) or real data",
    "P2 fails while P1 passes": "plausible same-family action candidates",
    "FORMAT fails": "more large-N examples and a higher random-code share",
    "FORMAT passes but F's flip rate in bins >= 27 > 0.20": "the consistency loss (KL/JS over permuted copies) goes into v0.9",
    "F's flip rate > 0.20 in a source (amendment)": "the order-consistency loss (KL/JS over permuted copies) goes into v0.9",
}
CONSISTENCY_BRANCH = "F's flip rate > 0.20 in a source (amendment)"


def consistency_check(flips: Mapping[str, Any] | None, *, r14_provided: bool) -> dict[str, Any]:
    """The amended consistency-loss condition: F's flip rate between two order seeds > 0.20 in ANY source — each M8g bin, B8 and, when its
    score files exist, R14 (robo-jev-web's collected bench requests). A source without a reading is listed as unread (never counted as ≤
    0.20): TRIGGERED iff a read source is over the bound; NOT TRIGGERED iff every source is read and none is over; otherwise UNREAD."""
    flips = flips or {}
    sources = list(CONSISTENCY_SOURCES) + ([R14] if r14_provided else [])
    read = {name: _num(flips.get(name)) for name in sources if _num(flips.get(name)) is not None}
    unread = [name for name in sources if name not in read]
    over = [name for name, value in read.items() if value > FLIP_LIMIT]
    status = "TRIGGERED" if over else ("UNREAD" if unread else "NOT TRIGGERED")
    return {"status": status, "triggered": bool(over), "over": over, "read": read, "unread": unread, "sources": sources,
            "r14": "scored" if r14_provided else "absent (no R14-<tag>.json: the condition reads on M8g and B8 only)",
            "reported_separately": R16_SEPARATE}


def next_steps(p1_pass: bool, p2_pass: bool, format_pass: bool, consistency: Mapping[str, Any] | None,
               flip_f_ge27: float | None = None) -> dict[str, Any]:
    """The pre-declared next steps with the amended consistency-loss condition: one step for H-prior (SUPPORTED; else P1 fails; else P2 fails
    while P1 passes), FORMAT's own step when FORMAT fails, and the consistency loss whenever the amended condition is TRIGGERED (applied as
    written: not conditional on FORMAT). The notes state FORMAT's status and what the original wording ("FORMAT passes but F's flip rate in
    bins ≥ 27 > 0.20", which required FORMAT to pass) would give, so either reading can be applied."""
    branches: list[str] = []
    if p1_pass and p2_pass:
        branches.append("SUPPORTED")
    elif not p1_pass:
        branches.append("P1 fails")
    else:
        branches.append("P2 fails while P1 passes")
    if not format_pass:
        branches.append("FORMAT fails")
    consistency = consistency or {}
    if consistency.get("triggered"):
        branches.append(CONSISTENCY_BRANCH)
    read = ", ".join(f"{k} {v:.3f}" for k, v in (consistency.get("read") or {}).items()) or "none"
    notes = [f"consistency-loss condition (amendment): {consistency.get('status', 'UNREAD')} — F's flip rate read on {read}"
             + (f"; over 0.20: {', '.join(consistency['over'])}" if consistency.get("over") else "")
             + (f"; unread: {', '.join(consistency['unread'])}" if consistency.get("unread") else "")
             + f"; R14 (robo-jev-web's collected bench requests) {consistency.get('r14', 'absent')}",
             f"FORMAT {'PASS' if format_pass else 'FAIL'}: the amendment's condition is applied as written, without FORMAT; the original wording "
             "required FORMAT to pass" + (f" (with F's flip rate in bins >= 27 {flip_f_ge27:.3f}, the original reading would "
                                           f"{'' if format_pass and flip_f_ge27 > FLIP_LIMIT else 'not '}add the consistency step)"
                                           if flip_f_ge27 is not None else " (F's flip rate in bins >= 27 unread)")]
    return {"branches": branches, "next": [NEXT[b] for b in branches], "notes": notes}


RELEASE_LINE = "RELEASE: the user decides (arm and seed)."


def decide(readings: Mapping[str, Any]) -> dict[str, Any]:
    """The whole rule on computed readings: ``P1``, ``P2``, ``guards``, ``FORMAT``, ``O-F``, the amended consistency condition's F flip
    rates per source (``consistency_flips``; ``r14_provided``) and ``flip_F_ge27`` (the original wording's reading, reported)."""
    p1, p2 = p1_check(readings.get("P1")), p2_check(readings.get("P2"))
    outcome = outcome_of(p1["pass"], p2["pass"])
    guards, fmt, of = guard_checks(readings.get("guards")), format_check(readings.get("FORMAT")), of_check(readings.get("O-F"))
    flip = _num(readings.get("flip_F_ge27"))
    consistency = consistency_check(readings.get("consistency_flips"), r14_provided=bool(readings.get("r14_provided")))
    original = {"format_pass": fmt["pass"], "flip_F_ge27": flip, "would_add_step": bool(fmt["pass"] and flip is not None and flip > FLIP_LIMIT),
                "readable": flip is not None}
    return {"outcome": outcome, "P1": p1, "P2": p2, "guards": guards, "FORMAT": fmt, "O-F": of, "flip_F_ge27": flip, "consistency": consistency,
            "consistency_original_reading": original, "next_steps": next_steps(p1["pass"], p2["pass"], fmt["pass"], consistency, flip),
            "release_line": RELEASE_LINE, "guards_note": "the guards are reported with pass / fail; the rule does not let them change the outcome label"}


# --------------------------------------------------------------------------
# Bootstrap helpers
# --------------------------------------------------------------------------


def _interval(samples: np.ndarray) -> tuple[float | None, float | None]:
    from robo_jev.bench.system1 import _interval as interval

    return interval(np.asarray(samples, dtype=float))


def _draws(clusters: int, n_boot: int, seed: int) -> np.ndarray:
    from robo_jev.bench.system1 import _draws as draws

    return draws(clusters, n_boot, seed)


def _reading(value: float | None, draws: np.ndarray | None, **extra: Any) -> dict[str, Any]:
    low, high = _interval(draws) if draws is not None and np.size(draws) else (None, None)
    return {"value": _num(value), "low": low, "high": high, **extra}


def _cluster_mean(values: Mapping[str, float], cluster_of: Mapping[str, str], *, n_boot: int, seed: int) -> dict[str, Any]:
    """The mean of per-unit values (each unit once) with a cluster-bootstrap 95 % interval."""
    keys = sorted(values)
    if not keys:
        return {"value": None, "n": 0, "clusters": 0}
    clusters = sorted({cluster_of[k] for k in keys})
    index = {c: i for i, c in enumerate(clusters)}
    sums, counts = np.zeros(len(clusters)), np.zeros(len(clusters))
    for k in keys:
        sums[index[cluster_of[k]]] += float(values[k])
        counts[index[cluster_of[k]]] += 1
    out = {"value": float(sums.sum() / counts.sum()), "n": len(keys), "clusters": len(clusters), "low": None, "high": None}
    if n_boot:
        d = _draws(len(clusters), n_boot, seed)
        with np.errstate(invalid="ignore", divide="ignore"):
            out["low"], out["high"] = _interval(sums[d].sum(1) / counts[d].sum(1))
    return out


def _cluster_diff(a: Mapping[str, float], b: Mapping[str, float], cluster_of: Mapping[str, str], *, n_boot: int, seed: int) -> dict[str, Any]:
    """a − b of per-unit values on the units both have, with a paired cluster-bootstrap 95 % interval."""
    keys = sorted(set(a) & set(b))
    if not keys:
        return {"value": None, "n": 0, "clusters": 0}
    clusters = sorted({cluster_of[k] for k in keys})
    index = {c: i for i, c in enumerate(clusters)}
    sa, sb, n = np.zeros(len(clusters)), np.zeros(len(clusters)), np.zeros(len(clusters))
    for k in keys:
        i = index[cluster_of[k]]
        sa[i] += float(a[k])
        sb[i] += float(b[k])
        n[i] += 1
    va, vb = float(sa.sum() / n.sum()), float(sb.sum() / n.sum())
    out = {"value": va - vb, "a": va, "b": vb, "n": len(keys), "clusters": len(clusters), "low": None, "high": None}
    if n_boot:
        d = _draws(len(clusters), n_boot, seed)
        with np.errstate(invalid="ignore", divide="ignore"):
            out["low"], out["high"] = _interval((sa[d].sum(1) - sb[d].sum(1)) / n[d].sum(1))
    return out


def _seed_mean(first: Mapping[str, float], second: Mapping[str, float]) -> dict[str, float]:
    """An arm's per-unit values: the mean of its two seeds' values, on the units both seeds have."""
    return {k: (float(first[k]) + float(second[k])) / 2.0 for k in sorted(set(first) & set(second))}


# --------------------------------------------------------------------------
# Standard readings: P1, the CA / GAP guards, O-F, context (call view, short form)
# --------------------------------------------------------------------------


def calls_of(sets: Mapping[str, Sequence[Mapping[str, Any]]], preds: Mapping[str, Mapping[str, Sequence[Mapping[str, Any]]]],
             set_names: Sequence[str] = CALL_SETS) -> dict[tuple[str, str], dict]:
    """Call items per (model, set) (v0.5's call view, pair clusters) and per (arm, set) (seed means of the call distributions)."""
    v4, v5, v6 = V4(), V5(), V6()
    calls: dict[tuple[str, str], dict] = {}
    for s in set_names:
        if not sets.get(s):
            continue
        clusters = v4.pair_clusters(sets[s])
        for model, by_set in preds.items():
            if by_set.get(s):
                calls[(model, s)] = v5.call_items(s, sets[s], by_set[s], clusters=clusters)
        for arm, (first, second) in ARMS.items():
            if (first, s) in calls and (second, s) in calls:
                calls[(arm, s)] = v6.arm_calls(calls[(first, s)], calls[(second, s)])
    return calls


def metric_draws(items_a: Mapping[tuple, tuple], items_b: Mapping[tuple, tuple], *, metric: str, n_boot: int, seed: int) -> dict[str, Any] | None:
    """a − b of one call metric on the questions both answered, with its cluster-bootstrap draws (a and b paired, `seed`)."""
    v5 = V5()
    keys = sorted(set(items_a) & set(items_b))
    if not keys:
        return None
    (sa, sb), counts, clusters = v5._cluster_arrays(keys, [items_a, items_b])
    pa, pb = _num(v5._metrics(sa.sum(0), counts.sum(0))[metric]), _num(v5._metrics(sb.sum(0), counts.sum(0))[metric])
    out: dict[str, Any] = {"value": None if pa is None or pb is None else pa - pb, "a": pa, "b": pb, "n": len(keys), "clusters": clusters,
                           "seed": seed, "draws": None, "low": None, "high": None}
    if n_boot:
        w = v5._draw_weights(clusters, n_boot, seed)
        n = w @ counts
        da = np.asarray(v5._metrics(np.einsum("bc,ctk->btk", w, sa), n)[metric], dtype=float)
        db = np.asarray(v5._metrics(np.einsum("bc,ctk->btk", w, sb), n)[metric], dtype=float)
        out["draws"] = da - db
        out["low"], out["high"] = _interval(da - db)
    return out


def mean_of_sets(calls: Mapping[tuple[str, str], Mapping], a: str, b: str, *, metric: str, n_boot: int, seed: int,
                 set_seeds: Sequence[int] | None = None, sets: Sequence[str] = P1_SETS) -> dict[str, Any]:
    """The rule's three-set reading of a − b (P1, the CA / GAP guards, O-F): per set the difference on the questions both answered, each set
    resampled by its own clusters with its own draw seed (default seed + the set's index — v0.6's author-gap precedent), and the per-set
    differences averaged per draw. A set without both readings makes the mean missing."""
    seeds = list(set_seeds) if set_seeds is not None else [seed + i for i in range(len(sets))]
    per_set: dict[str, Any] = {}
    draws = []
    for s, s_seed in zip(sets, seeds):
        if (a, s) not in calls or (b, s) not in calls:
            per_set[s] = None
            continue
        d = metric_draws(calls[(a, s)], calls[(b, s)], metric=metric, n_boot=n_boot, seed=s_seed)
        per_set[s] = None if d is None else {k: v for k, v in d.items() if k != "draws"}
        if d is not None and d.get("draws") is not None:
            draws.append(d["draws"])
    values = [(per_set[s] or {}).get("value") for s in sets]
    complete = all(v is not None for v in values)
    mean = float(np.mean(values)) if complete else None
    low = high = None
    if complete and n_boot and len(draws) == len(sets):
        low, high = _interval(np.mean(np.stack(draws), axis=0))
    return {"value": mean, "low": low, "high": high, "metric": metric, "a": a, "b": b, "per_set": per_set, "set_seeds": dict(zip(sets, seeds)),
            "complete": complete}


def standard_readings(sets: Mapping[str, Sequence[Mapping[str, Any]]], preds: Mapping[str, Mapping[str, Sequence[Mapping[str, Any]]]], *,
                      n_boot: int = 2000, seed: int = 0) -> dict[str, Any]:
    """P1, the CA / GAP guards and O-F (three-set means), and the context: call metrics per model and arm per set, FP − F and F − v0.7 per
    set (B1 and the H-set included), the seed spread and KL to the teacher on the choice questions with spread labels."""
    calls = calls_of(sets, preds)
    v5 = V5()
    out: dict[str, Any] = {"sets": {s: {"records": len(sets[s])} for s in CALL_SETS if sets.get(s)}}
    out["P1"] = mean_of_sets(calls, "FP", "F", metric="DC", n_boot=n_boot, seed=seed)
    out["CA"] = mean_of_sets(calls, "FP", "F", metric="CA", n_boot=n_boot, seed=seed)
    out["GAP"] = mean_of_sets(calls, "FP", "F", metric="GAP", n_boot=n_boot, seed=seed)
    out["O-F"] = mean_of_sets(calls, "F", "v07", metric="DC", n_boot=n_boot, seed=seed)
    out["FP_minus_F"] = {s: v5.call_difference(calls[("FP", s)], calls[("F", s)], n_boot=n_boot, seed=seed)
                         for s in CALL_SETS if ("FP", s) in calls and ("F", s) in calls}
    out["F_minus_v07"] = {s: v5.call_difference(calls[("F", s)], calls[("v07", s)], n_boot=n_boot, seed=seed)
                          for s in P1_SETS if ("F", s) in calls and ("v07", s) in calls}
    out["vs_base"] = {s: {arm: v5.call_difference(calls[(arm, s)], calls[("base", s)], n_boot=n_boot, seed=seed).get("DC")
                          for arm in ARMS if (arm, s) in calls} for s in CALL_SETS if ("base", s) in calls}
    out["context"] = {s: {m: v5.call_metrics(calls[(m, s)], n_boot=n_boot, seed=seed) for m in (*MODELS, *ARMS) if (m, s) in calls} for s in CALL_SETS}
    spread = {}
    for s in CALL_SETS:
        for arm, (first, second) in ARMS.items():
            a, b = out["context"][s].get(first) or {}, out["context"][s].get(second) or {}
            if a and b:
                spread[f"{s}/{arm}"] = {k: abs(a[k]["value"] - b[k]["value"]) if a.get(k, {}).get("value") is not None and b.get(k, {}).get("value") is not None
                                        else None for k in ("DC", "LU", "LC", "GAP", "CA")}
    out["seed_spread"] = spread
    out["teacher_kl"] = teacher_kl(sets, preds)
    return out


def _kl(p: Mapping[str, float], q: Mapping[str, float]) -> float:
    """KL(p ‖ q) in nats over p's keys (q clipped at 1e-12)."""
    total = 0.0
    for k, pk in p.items():
        pk = float(pk)
        if pk > 0:
            total += pk * math.log(pk / max(float(q.get(k, 0.0)), 1e-12))
    return total


def teacher_kl(sets: Mapping[str, Sequence[Mapping[str, Any]]], preds: Mapping[str, Mapping[str, Sequence[Mapping[str, Any]]]]) -> dict[str, Any]:
    """KL(teacher ‖ model) on the choice questions whose label is a spread distribution (top < 1): per question the mean over its listings,
    then the mean over questions — per option-count bin, per model and arm (an arm = the mean of its seeds' per-question values)."""
    from robo_jev.decisions.metrics import listings_of

    per_model: dict[str, dict[tuple, float]] = {}
    bins_of: dict[tuple, str] = {}
    for model, by_set in preds.items():
        values: dict[tuple, float] = {}
        for s in CALL_SETS:
            if not sets.get(s) or not by_set.get(s):
                continue
            predicted = {str(p["id"]): p for p in by_set[s]}
            for record in sets[s]:
                rid = str(record["id"])
                for name, question in record["request"]["questions"].items():
                    label = (record.get("labels") or {}).get(name)
                    if question.get("type") != "choice" or not isinstance(label, Mapping) or not isinstance(label.get("probabilities"), Mapping):
                        continue
                    teacher = {k: float(v) for k, v in label["probabilities"].items()}
                    if not teacher or max(teacher.values()) >= 1.0 - 1e-9:
                        continue
                    answer = ((predicted.get(rid) or {}).get("questions") or {}).get(name)
                    if not isinstance(answer, Mapping):
                        continue
                    kls = [_kl(teacher, l["answer"]["probabilities"]) for l in listings_of(dict(answer))
                           if isinstance((l.get("answer") or {}).get("probabilities"), Mapping)]
                    if kls:
                        values[(s, rid, name)] = float(np.mean(kls))
                        bins_of[(s, rid, name)] = G8.bin_of(len(teacher))
        per_model[model] = values
    for arm, (first, second) in ARMS.items():
        if first in per_model and second in per_model:
            per_model[arm] = {k: (per_model[first][k] + per_model[second][k]) / 2.0 for k in set(per_model[first]) & set(per_model[second])}
    out: dict[str, Any] = {}
    for model, values in per_model.items():
        by_bin: dict[str, list[float]] = defaultdict(list)
        for k, v in values.items():
            by_bin[bins_of[k]].append(v)
        out[model] = {b: {"value": float(np.mean(v)), "questions": len(v)} for b, v in sorted(by_bin.items())}
    return out


# --------------------------------------------------------------------------
# Pod guards: T2–T4 balanced band and severe errors, G2
# --------------------------------------------------------------------------


def home_readings(sets: Mapping[str, Sequence[Mapping[str, Any]]], preds: Mapping[str, Mapping[str, Sequence[Mapping[str, Any]]]], *,
                  n_boot: int = 2000, seed: int = 0) -> dict[str, Any]:
    """T2–T4 (pods) FP − F, arm means: the balanced band (stratified paired bootstrap, v0.7's home reading) and severe errors."""
    from robo_jev.bench import system1

    v4, v6 = V4(), V6()
    items = {}
    for model in V08_MODELS:
        if all(sets.get(h) and (preds.get(model) or {}).get(h) for h in HOME):
            items[model] = system1.group_items({h: system1.set_items(h, sets[h], preds[model][h]) for h in HOME}, HOME)
    out: dict[str, Any] = {"models": sorted(items)}
    arm_items = {}
    for arm in ("F", "FP"):
        first, second = ARMS[arm]
        if first in items and second in items:
            arm_items[arm] = {metric: v6.arm_values(items[first].get(metric) or {}, items[second].get(metric) or {}) for metric in ("band_balanced", "severe_error")}
    if "F" in arm_items and "FP" in arm_items:
        out["band"] = v4._paired(arm_items["FP"]["band_balanced"], arm_items["F"]["band_balanced"], n_boot=n_boot, seed=seed, stratified=True)
        out["severe"] = v4._paired(arm_items["FP"]["severe_error"], arm_items["F"]["severe_error"], n_boot=n_boot, seed=seed)
    return out


def g2_reading(g2: Sequence[Mapping[str, Any]] | None, preds: Mapping[str, Mapping[str, Sequence[Mapping[str, Any]]]], *, n_boot: int = 2000,
               seed: int = 0) -> dict[str, Any] | None:
    """G2 accuracy FP − F (arm means per question; :func:`verdict_v06.g2_arm_comparison`)."""
    if not g2 or not all((preds.get(m) or {}).get("G2") for m in V08_MODELS):
        return None
    return V6().g2_arm_comparison(list(g2), [preds["v08FPs0"]["G2"], preds["v08FPs1"]["G2"]], [preds["v08Fs0"]["G2"], preds["v08Fs1"]["G2"]],
                                  n_boot=n_boot, seed=seed)


# --------------------------------------------------------------------------
# B8: P2, flips, long lists
# --------------------------------------------------------------------------


def argmax_key(probabilities: Mapping[str, Any], order: Sequence[str]) -> str | None:
    """The most likely option; ties go to the earlier option of `order` (keys missing from the answer count as 0)."""
    best, value = None, -math.inf
    for key in order:
        p = float(probabilities.get(key, 0.0)) if isinstance(probabilities, Mapping) else 0.0
        if p > value:
            best, value = key, p
    return best


def _kinds(options: Sequence[Mapping[str, Any]]) -> dict[str, str | None]:
    """Per option of a tagged question: ``hold`` / ``continue`` for a PASSIVE option (by its tag rule, :func:`robo_jev.bench.r14_v08.passive_kind`;
    ``unmapped`` if the rule has no kind), None for an ACTIVE one or a level option."""
    from robo_jev.bench.r14_v08 import passive_kind

    return {str(o["key"]): ((passive_kind(o.get("rule")) or "unmapped") if o.get("tag") == "PASSIVE" else None) for o in options}


def b8_units(tags: Mapping[str, Any], records: Mapping[str, Mapping[str, Any]], *, exclude: Iterable[str] = ()) -> list[dict[str, Any]]:
    """Every tagged question of B8 (requests in `exclude` left out): request, env, question, option order (the request's), tag per option,
    planner (None if the question has none), option count and bin, and whether it is a P2 unit (plates / café / delivery)."""
    exclude = set(exclude)
    out = []
    for entry in tags.get("requests") or []:
        rid = str(entry["id"])
        if rid in exclude:
            continue
        env = str(entry.get("env"))
        for name, q in (entry.get("questions") or {}).items():
            options = [o["key"] for o in q.get("options") or []]
            record_q = ((records.get(rid) or {}).get("request") or {}).get("questions", {}).get(name) or {}
            order = list(record_q.get("criteria") or options)
            if set(order) != set(options):
                raise ValueError(f"B8 {rid} {name}: the tagged options differ from the request's")
            out.append({"id": rid, "env": env, "question": name, "order": order, "tags": {o["key"]: o["tag"] for o in q["options"]},
                        "kinds": _kinds(q["options"]),
                        "borderline": {o["key"]: bool(o.get("borderline")) for o in q["options"]}, "planner": q.get("planner"), "n": len(order),
                        "bin": G8.bin_of(len(order)), "p2": env in P2_ENVS})
    return out


def _answer(scores: Mapping[str, Any] | None, rid: str, name: str, reading: str) -> Mapping[str, Any] | None:
    found = ((((scores or {}).get("answers") or {}).get(rid) or {}).get(name) or {}).get(reading)
    return found if isinstance(found, Mapping) and isinstance(found.get("probabilities"), Mapping) else None


def b8_unit_values(units: Sequence[Mapping[str, Any]], scores: Mapping[str, Any], *, reading: str = "balanced-s1") -> dict[str, dict[tuple, float]]:
    """Per unit (env, request, question) of one model: passive_top, planner (agreement; units with a planner) and passive_mass."""
    out: dict[str, dict[tuple, float]] = {"passive_top": {}, "passive_top_hold": {}, "passive_top_continue": {}, "planner": {}, "passive_mass": {}}
    for u in units:
        answer = _answer(scores, u["id"], u["question"], reading)
        if answer is None:
            continue
        probs = answer["probabilities"]
        top = argmax_key(probs, u["order"])
        key = (u["env"], u["id"], u["question"])
        out["passive_top"][key] = float(u["tags"][top] == "PASSIVE")
        kind = (u.get("kinds") or {}).get(top)
        out["passive_top_hold"][key] = float(kind == "hold")
        out["passive_top_continue"][key] = float(kind == "continue")
        if kind == "unmapped":
            out.setdefault("passive_top_unmapped", {})[key] = 1.0
        out["passive_mass"][key] = float(sum(float(probs.get(k, 0.0)) for k, t in u["tags"].items() if t == "PASSIVE"))
        if u["planner"] is not None:
            out["planner"][key] = float(top == u["planner"])
    return out


def b8_readings(tags: Mapping[str, Any], records: Mapping[str, Mapping[str, Any]], scores: Mapping[str, Mapping[str, Any]], *,
                n_boot: int = 2000, seed: int = 0, exclude: Iterable[str] = ()) -> dict[str, Any]:
    """P2 (FP − F on the pooled P2 units), the levels per model and arm and per environment, the per-bin flip rate (single seed 1 vs seed 2),
    the long lists and the bench's hold-rule context."""
    units = b8_units(tags, records, exclude=exclude)
    p2_units = [u for u in units if u["p2"]]
    cluster = {(u["env"], u["id"], u["question"]): u["id"] for u in units}
    values = {m: b8_unit_values(p2_units, s) for m, s in scores.items()}
    for arm, (first, second) in ARMS.items():
        if first in values and second in values:
            values[arm] = {metric: _seed_mean(values[first][metric], values[second][metric]) for metric in values[first]}
    out: dict[str, Any] = {"units": len(units), "p2_units": len(p2_units), "requests": len({u["id"] for u in p2_units}),
                           "excluded_requests": sorted(set(exclude)), "per_env_units": dict(Counter(u["env"] for u in p2_units)),
                           "planner_units": sum(u["planner"] is not None for u in p2_units)}
    if "FP" in values and "F" in values:
        out["P2"] = {metric: _cluster_diff(values["FP"][metric], values["F"][metric], cluster, n_boot=n_boot, seed=seed)
                     for metric in ("passive_top", "planner", "passive_mass")}
        out["P2_per_env"] = {env: {metric: _cluster_diff({k: v for k, v in values["FP"][metric].items() if k[0] == env},
                                                         {k: v for k, v in values["F"][metric].items() if k[0] == env}, cluster, n_boot=n_boot, seed=seed)
                                   for metric in ("passive_top", "planner", "passive_top_hold", "passive_top_continue")} for env in P2_ENVS}
        # reported next to P2 (P2 reads the frozen PASSIVE tag): the passive-top share's two parts, hold and continue
        out["P2_split"] = {metric: _cluster_diff(values["FP"][metric], values["F"][metric], cluster, n_boot=n_boot, seed=seed)
                           for metric in ("passive_top_hold", "passive_top_continue")}
    out["levels"] = {m: {metric: _cluster_mean(v[metric], cluster, n_boot=n_boot, seed=seed) for metric in v} for m, v in values.items()}
    out["levels_per_env"] = {m: {env: {metric: _cluster_mean({k: x for k, x in v[metric].items() if k[0] == env}, cluster, n_boot=0, seed=seed)["value"]
                                       for metric in v} for env in P2_ENVS} for m, v in values.items()}
    out["flips"] = flip_readings(units, scores, group="env")  # every tagged question: the P2 units and the tube cell's lists
    out["long_lists"] = long_lists(units, scores)
    out["hold_rule"] = hold_rule(p2_units, scores)
    return out


def pick_rstd(picks: Sequence[tuple[int, int]], *, block: int = RSTD_BLOCK, min_n: int = RSTD_MIN) -> dict[str, Any]:
    """The label-free pick-rate RStd: `picks` = (shown position of the top-1 option, options shown) per question. Per block of `block`
    consecutive shown positions the ratio (top-1 picks landing in the block / options shown in the block) / (all picks / all options shown);
    the standard deviation (ddof 0) over blocks shown by ≥ `min_n` questions. ≈ 0 means no position preference."""
    questions, options, picked = Counter(), Counter(), Counter()
    for position, n in picks:
        for b in range((int(n) - 1) // block + 1):
            questions[b] += 1
            options[b] += min(block, int(n) - block * b)
        picked[int(position) // block] += 1
    total_picks, total_shown = len(picks), sum(int(n) for _, n in picks)
    if not total_picks:
        return {"value": None, "blocks": 0, "questions": 0}
    base = total_picks / total_shown
    ratios = {b: (picked[b] / options[b]) / base for b in sorted(questions) if questions[b] >= min_n}
    return {"value": float(np.std(list(ratios.values()))) if len(ratios) >= 2 else None, "blocks": len(ratios), "questions": total_picks,
            "ratios": {str(b): round(r, 4) for b, r in ratios.items()}}


def _single_pick(answer: Mapping[str, Any] | None, order: Sequence[str]) -> tuple[int, int] | None:
    """(shown position of the top-1 option, options shown) of a single-order answer that carries its ``codes`` map."""
    if answer is None or not isinstance(answer.get("codes"), Mapping):
        return None
    codes = answer["codes"]
    top = argmax_key(answer["probabilities"], order)
    if top not in codes or codes[top] not in CODE_POSITION:
        return None
    return CODE_POSITION[codes[top]], len(codes)


def flip_readings(units: Sequence[Mapping[str, Any]], scores: Mapping[str, Mapping[str, Any]], *, group: str = "env") -> dict[str, Any]:
    """Flips between the single order seed 1 and seed 2 (a flip = a different top option; ties to the earlier option of the request), pooled,
    per `group` value and per option-count bin, per model and arm (an arm's question value = the mean of its seeds'), and the label-free
    pick-rate RStd of the seed-1 single order (pooled and per bin; an arm pools both seeds' picks). `units` carry id, question, order, bin
    and the `group` field."""
    flips: dict[str, dict[tuple, float]] = {}
    picks: dict[str, dict[tuple, tuple[int, int]]] = {}
    info = {}
    for model, s in scores.items():
        values, shown = {}, {}
        for u in units:
            key = (u.get(group), u["id"], u["question"])
            info[key] = u
            a, b = _answer(s, u["id"], u["question"], "single-s1"), _answer(s, u["id"], u["question"], "single-s2")
            if a is not None and b is not None:
                values[key] = float(argmax_key(a["probabilities"], u["order"]) != argmax_key(b["probabilities"], u["order"]))
            pick = _single_pick(a, u["order"])
            if pick is not None:
                shown[key] = pick
        if values:
            flips[model] = values
        if shown:
            picks[model] = shown
    for arm, (first, second) in ARMS.items():
        if first in flips and second in flips:
            flips[arm] = _seed_mean(flips[first], flips[second])
    out: dict[str, Any] = {"pooled": {}, "per_group": {}, "by_bin": {}, "pick_rstd": {}, "pick_rstd_by_bin": {}}
    for model, values in flips.items():
        out["pooled"][model] = {"value": float(np.mean(list(values.values()))), "questions": len(values)}
        for field, target in ((group, "per_group"), ("bin", "by_bin")):
            grouped: dict[str, list[float]] = defaultdict(list)
            for k, v in values.items():
                grouped[str(info[k].get(field))].append(v)
            out[target][model] = {g: {"value": float(np.mean(v)), "questions": len(v)} for g, v in sorted(grouped.items())}
    pick_lists = {m: list(v.items()) for m, v in picks.items()}
    for arm, (first, second) in ARMS.items():
        if first in picks and second in picks:
            pick_lists[arm] = pick_lists[first] + pick_lists[second]
    for model, items in pick_lists.items():
        out["pick_rstd"][model] = pick_rstd([p for _, p in items])
        by_bin: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for k, p in items:
            by_bin[str(info[k]["bin"])].append(p)
        out["pick_rstd_by_bin"][model] = {b: pick_rstd(v) for b, v in sorted(by_bin.items())}
    return out


def long_lists(units: Sequence[Mapping[str, Any]], scores: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """The real long lists — tube cell ``work`` and plates ``move``: per model the mass on two-letter-coded options in the single order (seed
    1), top-1 agreement single vs balanced, planner agreement (balanced) and passive-top share (balanced)."""
    out: dict[str, Any] = {}
    for env, name in (("tubes", "work"), ("plates", "move")):
        chosen = [u for u in units if u["env"] == env and u["question"] == name]
        if not chosen:
            continue
        out[f"{env} {name}"] = {"questions": len(chosen), "options": sorted({u["n"] for u in chosen}), "models": {}}
        for model, s in scores.items():
            two, agree, planner, passive = [], [], [], []
            for u in chosen:
                single, balanced = _answer(s, u["id"], name, "single-s1"), _answer(s, u["id"], name, "balanced-s1")
                if single is not None and isinstance(single.get("codes"), Mapping):
                    two.append(sum(float(single["probabilities"].get(k, 0.0)) for k, c in single["codes"].items() if len(str(c)) == 2))
                if single is not None and balanced is not None:
                    agree.append(float(argmax_key(single["probabilities"], u["order"]) == argmax_key(balanced["probabilities"], u["order"])))
                if balanced is not None:
                    top = argmax_key(balanced["probabilities"], u["order"])
                    passive.append(float(u["tags"][top] == "PASSIVE"))
                    if u["planner"] is not None:
                        planner.append(float(top == u["planner"]))
            out[f"{env} {name}"]["models"][model] = {
                "two_letter_mass_single": float(np.mean(two)) if two else None, "top_single_eq_balanced": float(np.mean(agree)) if agree else None,
                "planner_agreement_balanced": float(np.mean(planner)) if planner else None, "passive_top_balanced": float(np.mean(passive)) if passive else None}
    return out


def hold_rule(units: Sequence[Mapping[str, Any]], scores: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """Context (as rel07): on questions with a planner recommendation, how often the bench would run the model's balanced-order choice instead
    of the planner's (p ≥ 0.55 and ≥ 0.15 above the recommendation)."""
    out = {}
    for model, s in scores.items():
        n = runs = passive = 0
        for u in units:
            answer = _answer(s, u["id"], u["question"], "balanced-s1")
            if answer is None or u["planner"] is None:
                continue
            probs = answer["probabilities"]
            top = argmax_key(probs, u["order"])
            n += 1
            if top != u["planner"] and float(probs.get(top, 0.0)) >= HOLD_P and float(probs.get(top, 0.0)) - float(probs.get(u["planner"], 0.0)) >= HOLD_AHEAD:
                runs += 1
                passive += u["tags"][top] == "PASSIVE"
        out[model] = {"questions": n, "overrides_run": runs, "passive_overrides_run": passive}
    return out


# --------------------------------------------------------------------------
# M8g: FORMAT and the per-bin readings
# --------------------------------------------------------------------------


def m8g_rows(records: Sequence[Mapping[str, Any]], predictions: Sequence[Mapping[str, Any]], *, kind: str = "order") -> dict[str, dict[str, Any]]:
    """One row per scored M8g record: its design (item, source, N, position, two-letter, bin, listing) and the model's top option, top-1
    correctness, p(key), confidence (top probability) and NLL of the key. `kind`: ``order`` (the record's own order — checked), ``seed2``
    (another order of the same options) or ``balanced`` (the question's average answer)."""
    question = G8.M8G_QUESTION if hasattr(G8, "M8G_QUESTION") else "q"
    by_id = {str(p["id"]): p for p in predictions}
    out = {}
    for record in records:
        rid = str(record["id"])
        found = by_id.get(rid)
        if found is None:
            continue
        meta = record["meta"]
        order = list(record["request"]["questions"][question]["criteria"])
        q = found["questions"][question]
        if kind == "balanced":
            probs = {k: float(v) for k, v in q["answer"]["probabilities"].items()}
            top = argmax_key(probs, order)
            top_position = None
        else:
            listing = q["listings"][0]
            options, code_probs = list(listing["options"]), [float(x) for x in listing["code_probs"]]
            if kind == "order" and options != order:
                raise ValueError(f"{rid}: the scored option order differs from the record's")
            if sorted(options) != sorted(order):
                raise ValueError(f"{rid}: the scored options differ from the record's")
            probs = dict(zip(options, code_probs))
            top = options[int(np.argmax(code_probs))]  # the first maximum = the earlier option shown
            top_position = int(np.argmax(code_probs))
        correct = str(meta["correct"])
        p_key = float(probs.get(correct, 0.0))
        out[rid] = {"id": rid, "item": str(meta["item"]), "source": str(meta.get("source")), "n": int(meta["n"]), "position": int(meta["position"]),
                    "two_letter": bool(meta.get("two_letter", int(meta["position"]) >= 26)), "bin": str(meta.get("bin") or G8.bin_of(int(meta["n"]))),
                    "listing": int(meta.get("listing", 0)), "top": top, "top_position": top_position, "correct": float(top == correct), "p_key": p_key,
                    "confidence": float(max(probs.values())), "nll": -math.log(max(p_key, 1e-12))}
    return out


def _arm_rows(first: Mapping[str, Mapping[str, Any]], second: Mapping[str, Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    """An arm's M8g rows: per record both seeds scored, the seeds' mean of correct / p(key) / NLL / confidence (design fields from the first)."""
    out = {}
    for rid in sorted(set(first) & set(second)):
        a, b = first[rid], second[rid]
        out[rid] = {**a, **{k: (float(a[k]) + float(b[k])) / 2.0 for k in ("correct", "p_key", "nll", "confidence")}, "top": None}
    return out


def _ece(confidence: Sequence[float], correct: Sequence[float], bins: int = ECE_BINS) -> float | None:
    """Top-label expected calibration error over equal-width confidence bins."""
    conf, acc = np.asarray(confidence, float), np.asarray(correct, float)
    if not conf.size:
        return None
    index = np.minimum((conf * bins).astype(int), bins - 1)
    total = 0.0
    for b in range(bins):
        mask = index == b
        if mask.any():
            total += mask.sum() / conf.size * abs(acc[mask].mean() - conf[mask].mean())
    return float(total)


def _rstd(positions: Sequence[int], correct: Sequence[float], *, block: int = RSTD_BLOCK, min_n: int = RSTD_MIN) -> dict[str, Any]:
    """Position RStd: the standard deviation (ddof 0) over blocks of `block` consecutive positions of the block's top-1 accuracy (blocks with ≥
    `min_n` listings)."""
    by_block: dict[int, list[float]] = defaultdict(list)
    for pos, c in zip(positions, correct):
        by_block[int(pos) // block].append(float(c))
    used = {b: float(np.mean(v)) for b, v in by_block.items() if len(v) >= min_n}
    return {"value": float(np.std(list(used.values()))) if len(used) >= 2 else None, "blocks": len(used)}


def _item_mean(rows: Sequence[Mapping[str, Any]], field: str, *, n_boot: int, seed: int) -> dict[str, Any]:
    return _cluster_mean({r["id"]: float(r[field]) for r in rows}, {r["id"]: r["item"] for r in rows}, n_boot=n_boot, seed=seed)


def format_reading(rows: Mapping[str, Mapping[str, Mapping[str, Any]]], *, n_boot: int = 2000, seed: int = 0) -> dict[str, Any]:
    """FORMAT on the M8g listings every rule model scored (F's two seeds and v0.7's): (a) per bin 27–60 and 61–255, F's accuracy on two-letter-
    coded vs single-letter-coded listings (and acc_two − 0.8 · acc_single with an item-cluster bootstrap interval); (b) N ≤ 26: F − v0.7 accuracy
    (item-cluster paired bootstrap)."""
    need = (*ARMS["F"], *ARMS["v07"])
    missing = [m for m in need if m not in rows]
    if missing:
        return {"missing_models": missing}
    common = set.intersection(*(set(rows[m]) for m in need))
    f = {k: v for k, v in _arm_rows(rows[ARMS["F"][0]], rows[ARMS["F"][1]]).items() if k in common}
    v07 = {k: v for k, v in _arm_rows(rows[ARMS["v07"][0]], rows[ARMS["v07"][1]]).items() if k in common}
    out: dict[str, Any] = {"listings": len(common), "bins": {}}
    for name in FORMAT_BINS:
        chosen = [r for r in f.values() if r["bin"] == name]
        two, single = [r for r in chosen if r["two_letter"]], [r for r in chosen if not r["two_letter"]]
        entry: dict[str, Any] = {"acc_two": _item_mean(two, "correct", n_boot=n_boot, seed=seed), "acc_single": _item_mean(single, "correct", n_boot=n_boot, seed=seed)}
        a, s = entry["acc_two"]["value"], entry["acc_single"]["value"]
        entry["ratio"] = (a / s) if a is not None and s else None
        entry["margin"] = _margin(two, single, n_boot=n_boot, seed=seed)
        out["bins"][name] = entry
    small = [k for k, r in f.items() if r["n"] <= 26]
    out["n26"] = _cluster_diff({k: f[k]["correct"] for k in small}, {k: v07[k]["correct"] for k in small}, {k: f[k]["item"] for k in small},
                               n_boot=n_boot, seed=seed)
    return out


def _margin(two: Sequence[Mapping[str, Any]], single: Sequence[Mapping[str, Any]], *, n_boot: int, seed: int) -> dict[str, Any]:
    """acc_two − 0.8 · acc_single (the FORMAT condition's margin, ≥ 0 passes) with an item-cluster bootstrap interval."""
    items = sorted({r["item"] for r in (*two, *single)})
    if not two or not single:
        return {"value": None}
    index = {i: j for j, i in enumerate(items)}
    st, nt, ss, ns = (np.zeros(len(items)) for _ in range(4))
    for r in two:
        st[index[r["item"]]] += r["correct"]
        nt[index[r["item"]]] += 1
    for r in single:
        ss[index[r["item"]]] += r["correct"]
        ns[index[r["item"]]] += 1
    value = float(st.sum() / nt.sum() - FORMAT_RATIO * ss.sum() / ns.sum())
    low = high = None
    if n_boot:
        d = _draws(len(items), n_boot, seed)
        with np.errstate(invalid="ignore", divide="ignore"):
            low, high = _interval(st[d].sum(1) / nt[d].sum(1) - FORMAT_RATIO * ss[d].sum(1) / ns[d].sum(1))
    return {"value": value, "low": low, "high": high, "items": len(items)}


def flip_values(main: Mapping[str, Mapping[str, Any]], seed2: Mapping[str, Mapping[str, Any]]) -> dict[str, float]:
    """Per record scored in both orders: 1 if the top option differs (record order vs the seeded order seed 2)."""
    return {rid: float(main[rid]["top"] != seed2[rid]["top"]) for rid in sorted(set(main) & set(seed2))}


def m8g_readings(records: Sequence[Mapping[str, Any]], preds: Mapping[str, Mapping[str, Sequence[Mapping[str, Any]]]], *, n_boot: int = 2000,
                 seed: int = 0, exclude: Iterable[str] = ()) -> dict[str, Any]:
    """FORMAT, F's flip rate in bins ≥ 27 and the reported per-bin readings. `preds[model][kind]` with kind ∈ ``order``, ``seed2``,
    ``balanced``; records of excluded items are left out."""
    exclude = set(exclude)
    kept = [r for r in records if str(r["meta"]["item"]) not in exclude]
    meta = {str(r["id"]): r["meta"] for r in kept}
    bin_of_id = {rid: str(m.get("bin") or G8.bin_of(int(m["n"]))) for rid, m in meta.items()}
    rows = {m: m8g_rows(kept, p["order"], kind="order") for m, p in preds.items() if p.get("order")}
    seed2 = {m: m8g_rows(kept, p["seed2"], kind="seed2") for m, p in preds.items() if p.get("seed2")}
    balanced = {m: m8g_rows(kept, p["balanced"], kind="balanced") for m, p in preds.items() if p.get("balanced")}
    out: dict[str, Any] = {"records": len(kept), "items": len({r["meta"]["item"] for r in kept}), "excluded_items": sorted(exclude),
                           "models": sorted(rows), "FORMAT": format_reading(rows, n_boot=n_boot, seed=seed)}
    # the common listings of every model present (per-bin readings compare models on the same listings)
    common = set.intersection(*(set(v) for v in rows.values())) if rows else set()
    flips = {m: flip_values(rows[m], seed2[m]) for m in seed2 if m in rows}
    arm_rows = {arm: _arm_rows(rows[a], rows[b]) for arm, (a, b) in ARMS.items() if a in rows and b in rows}
    arm_flips = {arm: _seed_mean(flips[a], flips[b]) for arm, (a, b) in ARMS.items() if a in flips and b in flips}
    f_flips = arm_flips.get("F")
    if f_flips:
        big = [v for rid, v in f_flips.items() if int(meta[rid]["n"]) >= 27]
        out["flip_F_ge27"] = {"value": float(np.mean(big)) if big else None, "records": len(big)}
    else:
        out["flip_F_ge27"] = {"value": None, "records": 0}
    # the amended consistency condition: F's flip rate in each bin, on every record both F seeds scored in both orders
    out["flip_F_by_bin"] = {}
    for name in BIN_NAMES:
        values = [v for rid, v in (f_flips or {}).items() if bin_of_id.get(rid) == name]
        out["flip_F_by_bin"][name] = {"value": float(np.mean(values)) if values else None, "records": len(values)}
    per_bin: dict[str, Any] = {}
    for model, model_rows in {**rows, **arm_rows}.items():
        chosen = [r for rid, r in model_rows.items() if rid in common]
        entry = {}
        for name in BIN_NAMES:
            in_bin = [r for r in chosen if r["bin"] == name]
            if not in_bin:
                continue
            fl = flips.get(model) or arm_flips.get(model) or {}
            fl_bin = [v for rid, v in fl.items() if bin_of_id.get(rid) == name]  # every record the model (arm: both seeds) scored in both orders
            entry[name] = {"listings": len(in_bin), "accuracy": _item_mean(in_bin, "correct", n_boot=n_boot, seed=seed),
                           "ece": _ece([r["confidence"] for r in in_bin], [r["correct"] for r in in_bin]),
                           "nll": float(np.mean([r["nll"] for r in in_bin])), "p_key": float(np.mean([r["p_key"] for r in in_bin])),
                           "flip_rate": float(np.mean(fl_bin)) if fl_bin else None, "flip_records": len(fl_bin),
                           "rstd": _rstd([r["position"] for r in in_bin], [r["correct"] for r in in_bin]),
                           "acc_two": float(np.mean([r["correct"] for r in in_bin if r["two_letter"]])) if any(r["two_letter"] for r in in_bin) else None,
                           "acc_single": float(np.mean([r["correct"] for r in in_bin if not r["two_letter"]])) if any(not r["two_letter"] for r in in_bin) else None}
        if model in ARMS:  # an arm's ECE: the mean of its seeds' ECE on the same listings
            a, b = ARMS[model]
            for name in entry:
                eces = [per_bin.get(s, {}).get(name, {}).get("ece") for s in (a, b)]
                entry[name]["ece"] = float(np.mean(eces)) if all(e is not None for e in eces) else None
        # label-free pick-rate RStd on the record-order listing (an arm pools both seeds' picks)
        sources_ = [rows[s] for s in ARMS[model]] if model in ARMS else [rows[model]]
        for name in entry:
            entry[name]["pick_rstd"] = pick_rstd([(src[rid]["top_position"], src[rid]["n"]) for src in sources_ for rid in src
                                                  if rid in common and src[rid]["bin"] == name])
        per_bin[model] = entry
    out["per_bin"] = per_bin
    # reported: general (MMLU) against physics (PHYS) per bin, per model and arm, on the same listings as the per-bin readings
    per_source: dict[str, Any] = {}
    for model, model_rows in {**rows, **arm_rows}.items():
        fl = flips.get(model) or arm_flips.get(model) or {}
        entry: dict[str, Any] = {}
        for source in sorted({r["source"] for r in model_rows.values()}):
            for name in BIN_NAMES:
                in_bin = [r for rid, r in model_rows.items() if rid in common and r["source"] == source and r["bin"] == name]
                if not in_bin:
                    continue
                two = [r["correct"] for r in in_bin if r["two_letter"]]
                single = [r["correct"] for r in in_bin if not r["two_letter"]]
                acc_two = float(np.mean(two)) if two else None
                acc_single = float(np.mean(single)) if single else None
                fl_bin = [v for rid, v in fl.items() if bin_of_id.get(rid) == name and str(meta[rid].get("source")) == source]
                entry.setdefault(source, {})[name] = {
                    "listings": len(in_bin), "accuracy": float(np.mean([r["correct"] for r in in_bin])), "acc_two": acc_two, "acc_single": acc_single,
                    "margin": (acc_two - FORMAT_RATIO * acc_single) if acc_two is not None and acc_single is not None else None,
                    "flip_rate": float(np.mean(fl_bin)) if fl_bin else None, "flip_records": len(fl_bin), "nll": float(np.mean([r["nll"] for r in in_bin]))}
        per_source[model] = entry
    out["per_source"] = per_source
    out["per_source_note"] = PER_SOURCE_NOTE
    by_source: dict[str, Any] = {}
    for model, model_rows in {**rows, **arm_rows}.items():
        acc: dict[tuple[str, int], list[float]] = defaultdict(list)
        for rid, r in model_rows.items():
            if rid in common:
                acc[(r["source"], r["n"])].append(r["correct"])
        by_source[model] = {f"{s} N={n}": float(np.mean(v)) for (s, n), v in sorted(acc.items())}
    out["by_source"] = by_source
    single_vs_balanced: dict[str, Any] = {}
    for model, b_rows in balanced.items():
        if model not in rows:
            continue
        shared = sorted(set(b_rows) & set(rows[model]))
        entry = {}
        for name in BIN_NAMES:
            ids = [rid for rid in shared if rows[model][rid]["bin"] == name]
            if ids:
                entry[name] = {"records": len(ids), "single": float(np.mean([rows[model][rid]["correct"] for rid in ids])),
                               "balanced": float(np.mean([b_rows[rid]["correct"] for rid in ids]))}
        single_vs_balanced[model] = entry
    out["single_vs_balanced"] = single_vs_balanced
    return out


def r14_units(tags: Mapping[str, Any], record_ids: Iterable[str] | None = None) -> list[dict[str, Any]]:
    """R14's scored questions: every question of tags.json (the model questions after serve's adapter — 1-option choices left out, scores with
    ≥ 2 criteria asked as level choices) of the requests in `record_ids` (the scored sample), with the option order (the tags' = the adapted
    request's criteria order), tag per option (None for a level question), planner (None unless one of the options), option count and bin."""
    wanted = None if record_ids is None else {str(i) for i in record_ids}
    out = []
    for entry in tags.get("requests") or []:
        rid = str(entry["id"])
        if wanted is not None and rid not in wanted:
            continue
        for name, q in (entry.get("questions") or {}).items():
            order = [str(o["key"]) for o in q.get("options") or []]
            if len(order) < 2:
                continue
            planner = q.get("planner") if q.get("planner") in order else None
            out.append({"id": rid, "env": str(entry.get("env")), "question": name, "order": order, "level": bool(q.get("level")),
                        "tags": {str(o["key"]): o.get("tag") for o in q["options"]}, "kinds": _kinds(q["options"]), "planner": planner,
                        "n": len(order), "bin": G8.bin_of(len(order))})
    return out


def r14_values(unit: Mapping[str, Any], scores: Mapping[str, Any] | None) -> dict[str, Any] | None:
    """One model's values on one R14 question, over serve's three orders (None unless all three were scored): flip (the top option is not the
    same in all three), passive-top share and planner agreement (means over the orders; None for a level question / a question without a
    planner option), override clearance (the web's rule: top ≠ planner, p_top ≥ 0.55 and ≥ 0.15 above the planner's), the top probabilities
    and the picks (shown position from each order's codes map, options shown)."""
    answers = [_answer(scores, unit["id"], unit["question"], reading) for reading in R14_READINGS]
    if any(a is None for a in answers):
        return None
    tops = [argmax_key(a["probabilities"], unit["order"]) for a in answers]
    p_top = [float(a["probabilities"].get(t, 0.0)) for a, t in zip(answers, tops)]
    tagged = not unit["level"] and all(unit["tags"].get(k) in ("PASSIVE", "ACTIVE") for k in unit["order"])
    planner = unit["planner"]
    overrides = None
    if planner is not None:
        overrides = float(np.mean([t != planner and p >= HOLD_P - TOL and p - float(a["probabilities"].get(planner, 0.0)) >= HOLD_AHEAD - TOL
                                   for a, t, p in zip(answers, tops, p_top)]))
    picks = [pick for pick in (_single_pick(a, unit["order"]) for a in answers) if pick is not None]
    kinds = unit.get("kinds") or {}
    return {"flip": float(len(set(tops)) > 1), "passive": float(np.mean([unit["tags"][t] == "PASSIVE" for t in tops])) if tagged else None,
            "passive_hold": float(np.mean([kinds.get(t) == "hold" for t in tops])) if tagged else None,
            "passive_continue": float(np.mean([kinds.get(t) == "continue" for t in tops])) if tagged else None,
            "planner": float(np.mean([t == planner for t in tops])) if planner is not None else None, "override": overrides, "top_probs": p_top,
            "picks": picks}


def _merge_r14(a: Mapping[str, Any], b: Mapping[str, Any]) -> dict[str, Any]:
    """An arm's values on a question: the mean of its seeds' per-question values; the top probabilities and picks of both seeds pooled."""
    mean = {k: (None if a[k] is None or b[k] is None else (a[k] + b[k]) / 2.0)
            for k in ("flip", "passive", "passive_hold", "passive_continue", "planner", "override")}
    return {**mean, "top_probs": list(a["top_probs"]) + list(b["top_probs"]), "picks": list(a["picks"]) + list(b["picks"])}


def _r14_summary(values: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    def mean_of(field: str) -> tuple[float | None, int]:
        found = [v[field] for v in values if v[field] is not None]
        return (float(np.mean(found)) if found else None), len(found)

    flip, n = mean_of("flip")
    passive, n_passive = mean_of("passive")
    hold, _ = mean_of("passive_hold")
    keep, _ = mean_of("passive_continue")
    planner, n_planner = mean_of("planner")
    override, _ = mean_of("override")
    probs = [p for v in values for p in v["top_probs"]]
    spread = ({"mean": float(np.mean(probs)), "median": float(np.median(probs)), "q25": float(np.percentile(probs, 25)),
               "q75": float(np.percentile(probs, 75)), "n": len(probs)} if probs else None)
    return {"questions": n, "flip_rate": flip, "passive_top": passive, "passive_top_hold": hold, "passive_top_continue": keep,
            "passive_questions": n_passive, "planner_agreement": planner,
            "planner_questions": n_planner, "override_clearance": override, "top_prob": spread,
            "pick_rstd": pick_rstd([pick for v in values for pick in v["picks"]])}


def r14_readings(units: Sequence[Mapping[str, Any]], scores: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """R14's readings per model and arm (an arm = the seeds' per-question means; spreads and picks pooled over both seeds): pooled over every
    scored question (all environments, choice and level), per environment and per environment × question."""
    per_model: dict[str, dict[tuple, dict[str, Any]]] = {}
    for model, s in scores.items():
        values = {}
        for u in units:
            v = r14_values(u, s)
            if v is not None:
                values[(u["env"], u["id"], u["question"])] = v
        per_model[model] = values
    for arm, (first, second) in ARMS.items():
        if first in per_model and second in per_model:
            per_model[arm] = {k: _merge_r14(per_model[first][k], per_model[second][k]) for k in sorted(set(per_model[first]) & set(per_model[second]))}
    out: dict[str, Any] = {"questions": len(units), "requests": len({u["id"] for u in units}), "level_questions": sum(u["level"] for u in units),
                           "planner_questions": sum(u["planner"] is not None for u in units), "per_env_questions": dict(Counter(u["env"] for u in units)),
                           "models": {}}
    for model, values in per_model.items():
        envs = sorted({k[0] for k in values})
        pairs = sorted({(k[0], k[2]) for k in values})
        out["models"][model] = {"pooled": _r14_summary(list(values.values())),
                                "per_env": {env: _r14_summary([v for k, v in values.items() if k[0] == env]) for env in envs},
                                "per_env_question": {f"{env} {name}": _r14_summary([v for k, v in values.items() if (k[0], k[2]) == (env, name)])
                                                     for env, name in pairs}}
    return out


def r14_check_summary(*places: Path) -> dict[str, Any] | None:
    """The R14 pipeline check (scripts/v08_r14_check.py: serve's sample, v0.7-s0, against serve's own counts) — its summary, if present."""
    for place in places:
        path = Path(place) / "r14-check.json"
        if path.is_file():
            data = json.loads(path.read_text(encoding="utf-8"))
            return {"path": str(path), **{k: data.get(k) for k in ("compared", "same_counted", "exact_flips", "max_abs_flips_diff")}}
    return None


def load_r14(spark_dir: Path, cks: Mapping[str, Path | None], notes: list[str], *, directory: Path | None = None) -> dict[str, Any] | None:
    """R14's frozen files and score files (``R14-<tag>.json``): None when no score file exists; ``{"refused": problems}`` when the frozen files
    changed (:func:`robo_jev.bench.r14_v08.check_frozen`) — the R14 readings are refused, not the whole verdict; a score file that fails its
    checks (set, scored sample, three readings, frozen record, checkpoint) is left out and listed."""
    from robo_jev.bench import r14_v08

    directory = Path(directory) if directory is not None else Path(r14_v08.R14_DIR)
    if not sorted(Path(spark_dir).glob("R14-*.json")):
        return None
    problems = r14_v08.check_frozen(directory)
    if problems:
        return {"refused": problems}
    frozen = json.loads((directory / "frozen-before-scoring.json").read_text(encoding="utf-8"))
    scores, left_out = {}, []
    for model in MODELS:
        path = Path(spark_dir) / f"R14-{SPARK_TAGS[model]}.json"
        if not path.is_file():
            notes.append(f"absent: {path}")
            continue
        data, found = check_bench_source(path, set_name=R14, tag=SPARK_TAGS[model], input_path=directory / R14_INPUT, checkpoint=cks.get(model),
                                         readings=R14_READINGS, frozen=frozen)
        if found:
            left_out += [f"{model}: {p}" for p in found]
            continue
        scores[model] = data
    tags = json.loads((directory / "tags.json").read_text(encoding="utf-8"))
    record_ids = [str(r["id"]) for r in _read(directory / R14_INPUT)]
    return {"tags": tags, "record_ids": record_ids, "scores": scores, "frozen": frozen, "left_out": left_out,
            "check": r14_check_summary(Path(spark_dir), directory)}


# --------------------------------------------------------------------------
# A16 and S16 — robo-jev-web's 2026-10-16 requests (reported only)
# --------------------------------------------------------------------------


def a16_check_summary(*places: Path) -> list[str]:
    """The A16 pipeline check (scripts/v08_a16_check.py: v0.7's stored answers against the rule and serve's numbers, our v0.7-s0 scoring, the
    y-phrase ablation) — its summary lines, if ``a16-check.json`` is present."""
    for place in places:
        path = Path(place) / "a16-check.json"
        if not path.is_file():
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        lines = []
        stored = (data.get("stored") or {}).get("main") or {}
        if stored:
            lines.append(f"stored v0.7 answers: agreement {stored.get('total')} (x {stored.get('x')}, y {stored.get('y')}, z {stored.get('z')}); "
                         f"sign errors {stored.get('sign_errors')}; reproduces serve's numbers: {(data.get('stored') or {}).get('reproduces_serve')}")
        ours = (data.get("ours") or {}).get("main") or {}
        if ours:
            lines.append(f"our scoring ({(data.get('ours') or {}).get('model')}): agreement {ours.get('total')} (x {ours.get('x')}, y {ours.get('y')}, "
                         f"z {ours.get('z')}); sign errors {ours.get('sign_errors')}")
        after = (data.get("ablation_y") or {}).get("after") or {}
        if after:
            lines.append(f"y without ' (up is plus)' ({(data.get('ablation_y') or {}).get('before_source')} before): agreement {after.get('y')}, "
                         f"y sign errors {(after.get('sign_errors') or {}).get('y')}")
        return [f"{path}: " + line for line in lines]
    return []


def load_r16(spark_dir: Path, cks: Mapping[str, Path | None], notes: list[str], *, directory: Path | None = None,
             exclude: Mapping[str, Iterable[str]] | None = None) -> dict[str, Any] | None:
    """A16 / S16's frozen files, v0.7's stored answers and the score files (``A16-<tag>.json``, ``A16y-<tag>.json``, ``S16-<tag>.json``): None
    when the frozen directory is absent; ``{"refused": problems}`` when its files changed (:func:`robo_jev.bench.r16_v08.check_frozen`) — only
    these two sets are refused; a score file that fails its checks (set, scored file, readings, the frozen record, checkpoint) is left out and
    listed. ``exclude`` = the overlap recheck's ``a16`` / ``s16`` record ids."""
    from robo_jev.bench import r16_v08

    directory = Path(directory) if directory is not None else Path(r16_v08.R16_DIR)
    if not (directory / "frozen-before-scoring.json").is_file():
        return None
    problems = r16_v08.check_frozen(directory)
    if problems:
        return {"refused": problems}
    frozen = json.loads((directory / "frozen-before-scoring.json").read_text(encoding="utf-8"))
    exclude = {k: {str(x) for x in v} for k, v in (exclude or {}).items()}
    records = {name: [r for r in _read(directory / R16_FILES[name]) if str(r["id"]) not in exclude.get("s16" if name == S16 else "a16", set())]
               for name in R16_FILES}
    scores: dict[str, dict[str, Any]] = {name: {} for name in R16_FILES}
    left_out = []
    for name, readings in R16_READINGS.items():
        for model in MODELS:
            path = Path(spark_dir) / f"{name}-{SPARK_TAGS[model]}.json"
            if not path.is_file():
                notes.append(f"absent: {path}")
                continue
            data, found = check_bench_source(path, set_name=name, tag=SPARK_TAGS[model], input_path=directory / R16_FILES[name], checkpoint=cks.get(model),
                                             readings=readings, frozen=frozen)
            if found:
                left_out += [f"{name} {model}: {p}" for p in found]
                continue
            scores[name][model] = data
    stored = {name: json.loads((directory / f"{name}-stored-v07.json").read_text(encoding="utf-8")).get("answers") or {} for name in (A16, S16)}
    return {"records": records, "scores": scores, "stored": stored, "frozen": frozen, "left_out": left_out,
            "excluded": {k: sorted(v) for k, v in exclude.items()}, "check": a16_check_summary(Path(spark_dir), directory)}


def a16_readings(records: Sequence[Mapping[str, Any]], scores: Mapping[str, Mapping[str, Any]], ablation: Mapping[str, Mapping[str, Any]],
                 stored: Mapping[str, Any]) -> dict[str, Any]:
    """A16 per model and arm (an arm's rows = :func:`robo_jev.bench.r16_v08.seed_mean_rows` of its seeds'): the rule's readings
    (:func:`~robo_jev.bench.r16_v08.arm_summary` — agreement total, per axis and per axis × signed band, sign errors per axis, the stay rate per
    band and on 0.2–3 cm), the boundary sensitivity (±0.2 read as stay) per axis, the stored v0.7 endpoint answers as a reference row, and the
    y-phrase ablation: y agreement and y sign errors without " (up is plus)" next to the same model's with it."""
    from robo_jev.bench import r16_v08 as R

    def with_arms(rows: dict[str, list]) -> dict[str, list]:
        for arm, (first, second) in ARMS.items():
            if first in rows and second in rows:
                rows[arm] = R.seed_mean_rows(rows[first], rows[second])
        return rows

    main = with_arms({m: R.arm_rows(records, s.get("answers") or {}, reading=A16_READING) for m, s in scores.items()})
    edge = with_arms({m: R.arm_rows(records, s.get("answers") or {}, reading=A16_READING, boundary_stay=True) for m, s in scores.items()})
    without = with_arms({m: R.arm_rows(records, s.get("answers") or {}, reading=A16_READING, axes=("y",)) for m, s in ablation.items()})
    out: dict[str, Any] = {"records": len(records), "models": {}, "serve_ablation_v07": SERVE_A16_ABLATION}

    def boundary(rows: list) -> dict[str, Any]:
        summary = R.arm_summary(rows)
        return {"agreement": summary["agreement"], "agreement_by_axis": summary["agreement_by_axis"]}

    for model in [m for m in (*MODELS, *ARMS) if m in main or m in without]:
        entry: dict[str, Any] = {}
        if model in main:
            entry["summary"] = R.arm_summary(main[model])
            entry["boundary_stay"] = boundary(edge[model])
        if model in without:
            after = R.arm_summary(without[model])
            before = (entry.get("summary") or {})
            entry["ablation_y"] = {"with_phrase": {"agreement": (before.get("agreement_by_axis") or {}).get("y"),
                                                   "sign_errors": (before.get("sign_errors_by_axis") or {}).get("y")},
                                   "without_phrase": {"agreement": after["agreement_by_axis"].get("y"), "sign_errors": after["sign_errors_by_axis"].get("y")}}
        out["models"][model] = entry
    stored_rows = R.arm_rows(records, stored, reading="stored")
    out["reference_v07_stored"] = {"summary": R.arm_summary(stored_rows),
                                   "boundary_stay": boundary(R.arm_rows(records, stored, reading="stored", boundary_stay=True))}
    return out


def _spread(values: Sequence[float]) -> dict[str, Any] | None:
    if not values:
        return None
    return {"mean": float(np.mean(values)), "median": float(np.median(values)), "q25": float(np.percentile(values, 25)),
            "q75": float(np.percentile(values, 75)), "n": len(values)}


def s16_units(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """S16's questions (every question of every request): the group (movement / gaze — ``meta.groups``) stands where R14 has the environment,
    the option order is the record's criteria order; no tags, no planner."""
    out = []
    for record in records:
        groups = (record.get("meta") or {}).get("groups") or {}
        for name, question in record["request"]["questions"].items():
            order = [str(k) for k in (question.get("criteria") or {})]
            if question.get("type") != "choice" or len(order) < 2:
                continue
            out.append({"id": str(record["id"]), "env": str(groups.get(name) or ("gaze" if str(name).endswith("_look") else "movement")),
                        "question": name, "order": order, "level": False, "tags": {k: None for k in order}, "kinds": {}, "planner": None,
                        "n": len(order), "bin": G8.bin_of(len(order))})
    return out


def s16_readings(records: Sequence[Mapping[str, Any]], scores: Mapping[str, Mapping[str, Any]], stored: Mapping[str, Any]) -> dict[str, Any]:
    """S16 per model and arm, pooled and per group (movement, 15 options; gaze, 6): the flip rate across serve's three orders, the
    top-probability spread over question × order and the pick-rate RStd (R14's readings); and the stored v0.7 answers' top-probability spread
    (one order) as a reference."""
    units = s16_units(records)
    base = r14_readings(units, scores)
    out: dict[str, Any] = {"questions": len(units), "requests": len({u["id"] for u in units}), "per_group_questions": dict(Counter(u["env"] for u in units)),
                           "models": {m: {"pooled": v["pooled"], "per_group": v["per_env"]} for m, v in base["models"].items()}}
    ref: dict[str, list[float]] = defaultdict(list)
    for u in units:
        answer = (((stored.get(u["id"]) or {}).get(u["question"]) or {}).get("stored")) or {}
        probs = answer.get("probabilities") if isinstance(answer, Mapping) else None
        if isinstance(probs, Mapping) and probs:
            top = argmax_key(probs, u["order"])
            ref[u["env"]].append(float(probs.get(top, 0.0)))
            ref["pooled"].append(float(probs.get(top, 0.0)))
    out["reference_v07_stored_top_prob"] = {group: _spread(values) for group, values in sorted(ref.items())}
    return out


# --------------------------------------------------------------------------
# Jev, probes, post-hoc estimate (reported)
# --------------------------------------------------------------------------


def _targets(record: Mapping[str, Any]) -> list[str]:
    target = (record.get("meta") or {}).get("target")
    if target:
        return [str(t) for t in target]
    return [n for n, q in record["request"]["questions"].items() if q.get("type") == "choice"]


def jev_readings(records: Sequence[Mapping[str, Any]], jev_preds: Sequence[Mapping[str, Any]], scores: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """Jev top-1 agreement on the comparison set: per model, part (bench / many) and reading, the share of target choice questions where the
    model's top option is Jev's (Jev's top = the argmax of its stored probabilities); Jev's and each model's accuracy on the many part."""
    jev = {str(p["id"]): p for p in jev_preds}
    units = []
    for record in records:
        rid = str(record["id"])
        meta = record.get("meta") or {}
        for name in _targets(record):
            question = record["request"]["questions"].get(name) or {}
            if question.get("type") != "choice":
                continue
            answer = ((jev.get(rid) or {}).get("questions") or {}).get(name)
            probs = ((answer or {}).get("answer") or {}).get("probabilities")
            order = list(question.get("criteria") or {})
            units.append({"id": rid, "question": name, "part": str(meta.get("part") or "bench"), "order": order,
                          "jev_top": argmax_key(probs, order) if isinstance(probs, Mapping) else None,
                          "correct": meta.get("correct") if str(meta.get("part")) == "many" else None})
    out: dict[str, Any] = {"questions": len(units), "jev_answered": sum(u["jev_top"] is not None for u in units), "parts": dict(Counter(u["part"] for u in units))}
    many = [u for u in units if u["part"] == "many" and u["jev_top"] is not None and u["correct"] is not None]
    out["jev_accuracy_many"] = {"value": float(np.mean([u["jev_top"] == u["correct"] for u in many])) if many else None, "questions": len(many)}
    models: dict[str, Any] = {}
    for model, s in scores.items():
        entry: dict[str, Any] = {}
        for part in sorted({u["part"] for u in units}):
            for reading in ("single-s1", "balanced-s1"):
                agree, acc = [], []
                for u in units:
                    if u["part"] != part:
                        continue
                    answer = _answer(s, u["id"], u["question"], reading)
                    if answer is None:
                        continue
                    top = argmax_key(answer["probabilities"], u["order"])
                    if u["jev_top"] is not None:
                        agree.append(float(top == u["jev_top"]))
                    if u["correct"] is not None:
                        acc.append(float(top == u["correct"]))
                if agree or acc:
                    entry[f"{part}/{reading}"] = {"agreement": float(np.mean(agree)) if agree else None, "questions": len(agree),
                                                  "accuracy": float(np.mean(acc)) if acc else None}
        models[model] = entry
    out["models"] = models
    return out


def probe_readings(rows: Sequence[Mapping[str, Any]], scores: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """The rel071 storage-bin cases: per model, case and reading, p(the right bin), its rank and (single orders) the mass on two-letter codes."""
    out: dict[str, Any] = {"cases": len(rows), "models": {}}
    for model, s in scores.items():
        cases = {}
        for row in rows:
            rid = str(row["id"])
            name = next(iter(row["request"]["questions"]))
            right = str(row["meta"]["right"])
            order = list(row["request"]["questions"][name]["criteria"])
            per = {}
            for reading, answer in (((((s or {}).get("answers") or {}).get(rid) or {}).get(name)) or {}).items():
                probs = answer.get("probabilities") or {}
                ranked = sorted(order, key=lambda k: (-float(probs.get(k, 0.0)), order.index(k)))
                entry = {"p_right": float(probs.get(right, 0.0)), "rank_right": ranked.index(right) + 1, "top": ranked[0]}
                if isinstance(answer.get("codes"), Mapping):
                    entry["two_letter_mass"] = float(sum(float(probs.get(k, 0.0)) for k, c in answer["codes"].items() if len(str(c)) == 2))
                    entry["right_code"] = answer["codes"].get(right)
                per[reading] = entry
            cases[rid] = {"n": row["meta"].get("n"), "position": row["meta"].get("position"), "kind": row["meta"].get("kind"), "readings": per}
        out["models"][model] = cases
    return out


def posthoc_reading(path: Path = POSTHOC_FILE) -> dict[str, Any]:
    """The post-hoc prior-removal estimate (reported context; written by task v0.8b before any v0.8 model existed)."""
    if not Path(path).is_file():
        return {"absent": str(path)}
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return {"path": str(path), "P1_estimate": data.get("P1_estimate"), "per_set": data.get("per_set"), "notes": data.get("notes")}


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


def check_b8_frozen(b8_file: Path = G8.B8_FILE, tags_file: Path = TAGS_FILE, frozen_file: Path = FROZEN_FILE) -> tuple[dict[str, Any] | None, list[str]]:
    """The frozen B8 tags: (frozen-before-scoring.json's content, problems) — B8.jsonl and tags.json must have the sha256 recorded there."""
    problems = [f"missing {p}" for p in (b8_file, tags_file, frozen_file) if not Path(p).is_file()]
    if problems:
        return None, problems
    frozen = json.loads(Path(frozen_file).read_text(encoding="utf-8"))
    if frozen.get("b8_sha256") != _sha(str(b8_file)):
        problems.append(f"B8.jsonl sha256 {_sha(str(b8_file))[:12]} != frozen {str(frozen.get('b8_sha256'))[:12]}")
    if frozen.get("tags_sha256") != _sha(str(tags_file)):
        problems.append(f"tags.json sha256 {_sha(str(tags_file))[:12]} != frozen {str(frozen.get('tags_sha256'))[:12]}")
    return frozen, problems


def pod_file(directory: Path, set_name: str) -> Path:
    """The pod's prediction file of a set: the unique ``*-<SET>.predictions.jsonl`` in the fetched reports directory."""
    found = sorted(Path(directory).glob(f"*-{set_name}.predictions.jsonl"))
    if len(found) != 1:
        raise SystemExit(f"{directory}: {len(found)} files match *-{set_name}.predictions.jsonl (want exactly one)")
    return found[0]


def pod_checkpoint(directory: Path) -> Path | None:
    """The run's checkpoint next to its fetched reports (``<run>-fetched/runs/*/checkpoint.pt``), if exactly one."""
    found = sorted((Path(directory).parent / "runs").glob("*/checkpoint.pt"))
    return found[0] if len(found) == 1 else None


def _checkpoint_problems(config_checkpoint: Mapping[str, Any] | None, checkpoint: Path | None) -> list[str]:
    got = (config_checkpoint or {}).get("sha256")
    if checkpoint is None:
        return ["a checkpoint was applied to the base"] if got is not None else []
    if not Path(checkpoint).is_file():
        return [f"checkpoint {checkpoint} not on disk"]
    want = _sha(str(checkpoint))
    return [] if got == want else [f"checkpoint sha256 {str(got)[:12]} vs {want[:12]} on disk ({checkpoint})"]


def check_m8g_source(predictions: Path, kind: str, checkpoint: Path | None, gate: Path = G8.M8G_FILE) -> list[str]:
    """An M8g prediction file's summary: scored on the gate file, with the kind's order seed and orders, prompt v0c-p3, the model's checkpoint."""
    from robo_jev.decisions.prompt import PROMPT_VERSION, fixed_text_digest

    summary = Path(str(predictions).replace(".predictions.jsonl", ".json"))
    if not summary.is_file():
        return [f"no summary {summary.name}"]
    config = json.loads(summary.read_text(encoding="utf-8")).get("config") or {}
    problems = [f"{k} = {config.get(k)!r}, want {v!r}" for k, v in M8G_CONDITIONS[kind].items() if config.get(k) != v]
    if config.get("input_sha256") != _sha(str(gate)):
        problems.append(f"scored {str(config.get('input_sha256'))[:12]}, not {Path(gate).name} ({_sha(str(gate))[:12]})")
    if config.get("prompt_digest") not in (None, fixed_text_digest(PROMPT_VERSION)):
        problems.append("prompt digest differs")
    return problems + _checkpoint_problems(config.get("checkpoint"), checkpoint)


def check_bench_source(path: Path, *, set_name: str, tag: str, input_path: Path | None, checkpoint: Path | None, readings: Sequence[str],
                       frozen: Mapping[str, Any] | None = None) -> tuple[dict[str, Any] | None, list[str]]:
    """A bench score file (B8 / Jev / Probe): its kind, set, tag, scored input (sha256), checkpoint, readings and — for B8 — the frozen tags."""
    if not Path(path).is_file():
        return None, [f"missing {path}"]
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    problems = []
    if data.get("kind") != "v08-bench-scores":
        problems.append(f"kind {data.get('kind')!r}")
    if data.get("set") != set_name:
        problems.append(f"set {data.get('set')!r}, want {set_name!r}")
    if data.get("tag") not in (None, tag):
        problems.append(f"tag {data.get('tag')!r}, want {tag!r}")
    if input_path is not None and data.get("input_sha256") != _sha(str(input_path)):
        problems.append(f"scored {str(data.get('input_sha256'))[:12]}, not {Path(input_path).name}")
    missing = [r for r in readings if r not in (data.get("readings") or {})]
    if missing:
        problems.append(f"readings missing: {missing}")
    if frozen is not None and data.get("frozen") != frozen:
        problems.append("its frozen tags differ from frozen-before-scoring.json (scored before the freeze, or on other tags)")
    return data, problems + _checkpoint_problems(data.get("checkpoint"), checkpoint)


def read_exclusions(path: Path = G8.OVERLAP_V08) -> dict[str, list[str]]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return {"items": [str(x) for x in data.get("excluded_items") or []], "b8": [str(x) for x in data.get("excluded_b8") or []],
            "a16": [str(x) for x in data.get("excluded_a16") or []], "s16": [str(x) for x in data.get("excluded_s16") or []],
            "train_sha256": data.get("train_v08", {}).get("sha256") if isinstance(data.get("train_v08"), Mapping) else data.get("train_sha256")}


def checkpoints(args: argparse.Namespace) -> dict[str, Path | None]:
    out: dict[str, Path | None] = {m: pod_checkpoint(getattr(args, POD_ARGS[m])) for m in V08_MODELS}
    out.update({"v07s0": V07_CKPT[0], "v07s1": V07_CKPT[1], "base": None})
    return out


def standard_paths(spark_dir: Path) -> dict[tuple[str, str], Path]:
    """(model, set) → the Spark short-form prediction file: v0.8 models (every call set), v0.7 (A4, T9, T8b), the base (v0.7's references)."""
    paths: dict[tuple[str, str], Path] = {}
    for m in V08_MODELS:
        for s in CALL_SETS:
            paths[(m, s)] = Path(spark_dir) / f"{s}-{SPARK_TAGS[m]}.predictions.jsonl"
    for seed in SEEDS:
        for s in P1_SETS:
            paths[(f"v07s{seed}", s)] = V07_SPARK / f"{s}-v07s{seed}-2b.predictions.jsonl"
    for s, path in V7().BASE_SPARK.items():
        paths[("base", s)] = Path(path)
    return paths


def required(args: argparse.Namespace) -> list[str]:
    """The files the rule reads (P1, P2, guards, FORMAT, O-F) — the verdict refuses if one is missing."""
    paths = standard_paths(args.spark_dir)
    need = [paths[(m, s)] for m in (*V08_MODELS, "v07s0", "v07s1") for s in P1_SETS]
    need += [Path(args.spark_dir) / f"B8-{SPARK_TAGS[m]}.json" for m in V08_MODELS]
    need += [Path(args.spark_dir) / f"M8g-{SPARK_TAGS[m]}.predictions.jsonl" for m in (*ARMS["F"], *ARMS["v07"])]
    need += [G8.B8_FILE, TAGS_FILE, FROZEN_FILE, G8.M8G_FILE]
    return [str(p) for p in need if not Path(p).is_file()]


def load_standard(paths: Mapping[tuple[str, str], Path], *, allow_mismatch: bool, notes: list[str], b2: bool = True) -> tuple[dict, dict, dict]:
    """Sets (T8b with B2's truth classes) and the short-form predictions that exist; each checked against its set file and the diagnosis
    conditions (:func:`verdict_v04.check_source`)."""
    set_paths = dict(V6().SET_PATHS)
    names = sorted({s for (_, s), p in paths.items() if Path(p).is_file()})
    sets = {s: _read(set_paths[s]) for s in names}
    if "T8b" in sets and b2:
        sets["T8b"], info = V6().t8b_with_b2(sets["T8b"])
        notes.append(f"T8b truth classes from B2's alternative-prior labels ({info['labels']}; original-prior reproduction share {info['reproduction_share']:.4f}).")
    preds: dict[str, dict[str, list]] = defaultdict(dict)
    sources = {}
    for (model, s), path in sorted(paths.items()):
        if not Path(path).is_file():
            notes.append(f"absent, skipped: {model} {s} ({path})")
            continue
        preds[model][s] = _read(path)
        sources[f"{model} {s}"] = V4().check_source(Path(path), set_paths[s], sets[s], preds[model][s])
    problems = [f"{k}: {'; '.join(v['problems'])}" for k, v in sources.items() if v["problems"]]
    if problems and not allow_mismatch:
        raise SystemExit("refusing (use --allow-mismatch to read anyway):\n  " + "\n  ".join(problems))
    return sets, dict(preds), sources


def load_pods(args: argparse.Namespace, *, allow_mismatch: bool) -> tuple[dict, dict, dict]:
    """T2, T3, T4, G2 records and the four v0.8 runs' pod predictions (each checked like the Spark files)."""
    set_paths = dict(V6().SET_PATHS)
    sets = {s: _read(set_paths[s]) for s in POD_SETS}
    preds: dict[str, dict[str, list]] = {}
    sources = {}
    for model in V08_MODELS:
        directory = getattr(args, POD_ARGS[model])
        preds[model] = {}
        for s in POD_SETS:
            path = pod_file(directory, s)
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
    value = _num(value) if not isinstance(value, str) else value
    if value is None:
        return "—"
    if isinstance(value, str):
        return value
    return f"{value:+.{digits}f}" if signed else f"{value:.{digits}f}"


def _cid(d: Mapping[str, Any] | None, signed: bool = True, key: str = "value") -> str:
    if not d or _num(d.get(key)) is None:
        return "—"
    return f"{_fmt(d[key], signed=signed)} [{_fmt(d.get('low'), signed=signed)}, {_fmt(d.get('high'), signed=signed)}]"


def _p2_row(name: str, d: Mapping[str, Any] | None, check: Any) -> str:
    d = d or {}
    return f"| {name} | {_fmt(d.get('a'))} | {_fmt(d.get('b'))} | {_cid(d)} | {check} |"


def _long_list(model: str, x: Mapping[str, Any]) -> str:
    return (f"{LABELS.get(model, model)} two-letter mass {_fmt(x['two_letter_mass_single'])}, single = balanced {_fmt(x['top_single_eq_balanced'])}, "
            f"planner {_fmt(x['planner_agreement_balanced'])}, passive-top {_fmt(x['passive_top_balanced'])}")


def _r14_row(label: str, e: Mapping[str, Any]) -> str:
    tp = e.get("top_prob") or {}
    return (f"| {label} | {e.get('questions')} | {_fmt(e.get('flip_rate'))} | {_fmt((e.get('pick_rstd') or {}).get('value'))} | "
            f"{_fmt(e.get('passive_top'))} ({e.get('passive_questions')}) | {_fmt(e.get('passive_top_hold'))} / {_fmt(e.get('passive_top_continue'))} | "
            f"{_fmt(e.get('planner_agreement'))} ({e.get('planner_questions')}) | "
            f"{_fmt(tp.get('mean'))} / {_fmt(tp.get('median'))} [{_fmt(tp.get('q25'))}, {_fmt(tp.get('q75'))}] | {_fmt(e.get('override_clearance'))} |")


def _render_r14(r14: Mapping[str, Any]) -> list[str]:
    """R14 — robo-jev-web's collected bench requests (reported; the amended condition's third source)."""
    lines = ["", "### R14 — robo-jev-web's collected bench requests (reported only; the consistency condition's third source)", ""]
    if r14.get("absent"):
        return lines + [f"absent: {r14['absent']} — the consistency condition reads on M8g and B8 only."]
    if r14.get("refused"):
        return lines + ["**refused**: the frozen R14 files changed — " + "; ".join(r14["refused"]) + " (R14 is listed unread; the rest of the verdict stands)."]
    lines.append(f"{r14.get('questions')} questions in {r14.get('requests')} requests (per env {r14.get('per_env_questions')}; {r14.get('level_questions')} level "
                 f"questions; {r14.get('planner_questions')} with a planner option); serve's three orders spaces-0 / 1 / 2; frozen {r14.get('frozen')}.")
    if r14.get("left_out"):
        lines.append("Score files left out: " + "; ".join(r14["left_out"]) + ".")
    check = r14.get("check")
    if check:
        lines.append(f"Pipeline check (serve's sample, v0.7-s0, against serve's own counts): compared {check.get('compared')}, same counted "
                     f"{check.get('same_counted')}, exact flips {check.get('exact_flips')}, max |Δ flips| {check.get('max_abs_flips_diff')}.")
    header = ["| model / arm | questions | flip rate | pick-rate RStd | passive-top (n) | hold / continue | planner agreement (n) | "
              "top p mean / median [q25, q75] | override clearance |", "|---|---|---|---|---|---|---|---|---|"]
    lines += ["", "Pooled (every environment, choice and level questions):", "", *header]
    models = r14.get("models") or {}
    for model in [m for m in (*MODELS, *ARMS) if m in models]:
        lines.append(_r14_row(LABELS.get(model, model), models[model]["pooled"]))
    lines += ["", "Per environment:", "", *header]
    for model in [m for m in (*MODELS, *ARMS) if m in models]:
        for env, e in models[model]["per_env"].items():
            lines.append(_r14_row(f"{LABELS.get(model, model)} · {env}", e))
    lines += ["", "Per environment × question (arms):", "", *header]
    for model in [m for m in ARMS if m in models]:
        for name, e in models[model]["per_env_question"].items():
            lines.append(_r14_row(f"{LABELS.get(model, model)} · {name}", e))
    return lines


def _rate_cell(d: Mapping[str, Any] | None) -> str:
    d = d or {}
    return "—" if d.get("value") is None else f"{_fmt(d['value'])} ({_fmt(d.get('count'), 1)}/{d.get('n')})"


def _render_r16(a16: Mapping[str, Any], s16: Mapping[str, Any]) -> list[str]:
    """A16 and S16 — robo-jev-web's 2026-10-16 requests (reported only)."""
    lines = ["", "### A16 and S16 — robo-jev-web's 2026-10-16 requests (reported only)", "", f"{R16_SEPARATE}.", ""]
    if a16.get("absent") or a16.get("refused"):
        reason = a16.get("absent") or ("**refused**: the frozen files changed — " + "; ".join(a16["refused"]) + " (listed unread; the rest of the verdict stands)")
        return lines + [f"A16 / S16: {reason}."]
    lines.append(f"Frozen {a16.get('frozen')}; excluded {a16.get('excluded')}." + (f" Score files left out: {'; '.join(a16['left_out'])}." if a16.get("left_out") else ""))
    for line in a16.get("check") or []:
        lines.append(f"A16 pipeline check — {line}.")
    lines += ["", "**A16 — the arm** (single order seed 1, serve's conditions; the rule written in each x / y / z question):", "",
              "| model / arm | agreement | x | y | z | sign errors x / y / z | stay rate 0.2–3 cm | boundary ±0.2 as stay: total (x) |", "|---|---|---|---|---|---|---|---|"]

    def a16_row(label: str, entry: Mapping[str, Any]) -> str:
        sm, bs = entry.get("summary") or {}, entry.get("boundary_stay") or {}
        ax, se = sm.get("agreement_by_axis") or {}, sm.get("sign_errors_by_axis") or {}
        return (f"| {label} | {_rate_cell(sm.get('agreement'))} | {_rate_cell(ax.get('x'))} | {_rate_cell(ax.get('y'))} | {_rate_cell(ax.get('z'))} | "
                f"{' / '.join(_rate_cell(se.get(a)) for a in ('x', 'y', 'z'))} | {_rate_cell(sm.get('stay_rate_small'))} | "
                f"{_rate_cell(bs.get('agreement'))} ({_rate_cell((bs.get('agreement_by_axis') or {}).get('x'))}) |")

    lines.append(a16_row("v0.7 endpoint (stored)", a16.get("reference_v07_stored") or {}))
    for model, entry in (a16.get("models") or {}).items():
        if entry.get("summary"):
            lines.append(a16_row(LABELS.get(model, model), entry))
    lines += ["", "Agreement per axis × signed band and the stay rate there (arms and the stored reference):", "",
              "| model / arm | axis:band | agreement | stay rate |", "|---|---|---|---|"]
    for model, entry in [("v0.7 endpoint (stored)", a16.get("reference_v07_stored") or {})] + [(LABELS.get(m, m), e) for m, e in (a16.get("models") or {}).items()
                                                                                                  if m in ARMS]:
        sm = entry.get("summary") or {}
        for band, d in (sm.get("agreement_by_axis_band") or {}).items():
            lines.append(f"| {model} | {band} | {_rate_cell(d)} | {_rate_cell((sm.get('stay_rate_by_axis_band') or {}).get(band))} |")
    lines += ["", f"y-phrase ablation (y without \" (up is plus)\", A16y, next to the same model's y with it; {a16.get('serve_ablation_v07', SERVE_A16_ABLATION)}):", "",
              "| model / arm | y agreement with / without | y sign errors with / without |", "|---|---|---|"]
    for model, entry in (a16.get("models") or {}).items():
        ab = entry.get("ablation_y")
        if ab:
            lines.append(f"| {LABELS.get(model, model)} | {_rate_cell(ab['with_phrase']['agreement'])} / {_rate_cell(ab['without_phrase']['agreement'])} | "
                         f"{_rate_cell(ab['with_phrase']['sign_errors'])} / {_rate_cell(ab['without_phrase']['sign_errors'])} |")
    lines += ["", f"**S16 — soccer** ({s16.get('questions')} questions in {s16.get('requests')} requests; per group {s16.get('per_group_questions')}; serve's three orders):", "",
              "| model / arm | group | questions | flip rate | top p mean / median [q25, q75] | pick-rate RStd |", "|---|---|---|---|---|---|"]
    ref = s16.get("reference_v07_stored_top_prob") or {}
    for group, sp in ref.items():
        lines.append(f"| v0.7 endpoint (stored, one order) | {group} | {(sp or {}).get('n')} | — | {_fmt((sp or {}).get('mean'))} / {_fmt((sp or {}).get('median'))} "
                     f"[{_fmt((sp or {}).get('q25'))}, {_fmt((sp or {}).get('q75'))}] | — |")
    for model in [m for m in (*MODELS, *ARMS) if m in (s16.get("models") or {})]:
        entry = s16["models"][model]
        for group, e in [("pooled", entry["pooled"]), *entry["per_group"].items()]:
            tp = e.get("top_prob") or {}
            lines.append(f"| {LABELS.get(model, model)} | {group} | {e.get('questions')} | {_fmt(e.get('flip_rate'))} | {_fmt(tp.get('mean'))} / {_fmt(tp.get('median'))} "
                         f"[{_fmt(tp.get('q25'))}, {_fmt(tp.get('q75'))}] | {_fmt((e.get('pick_rstd') or {}).get('value'))} |")
    return lines


def render(result: Mapping[str, Any]) -> str:
    v, r = result["verdict"], result["readings"]
    lines = [f"# v0.8 pre-declared rule — {v['outcome']} · FORMAT {'PASS' if v['FORMAT']['pass'] else 'FAIL'}", ""]
    if result.get("pipeline_check"):
        lines += [f"**PIPELINE CHECK ONLY — {result['pipeline_check']}; this is not the v0.8 verdict.**", ""]
    lines += [f"Rule written {RULE['written']}. Applied by `scripts/verdict_v08.py` ({SCRIPT_VERSION}, git {result.get('git')}); {result['n_boot']} paired "
              f"cluster-bootstrap draws (seed {result['seed']}); seed means (arm F = v0.8F s0 + s1, arm FP = v0.8FP s0 + s1, arm v0.7 = v0.7 s0 + s1).", ""]
    st = r.get("standard") or {}
    p1 = st.get("P1") or {}
    lines += ["## P1 — short-form outside DC, FP − F (mean of A4, T9, T8b-B2; each set drawn by its own clusters, averaged per draw)", "",
              "| set | DC FP | DC F | Δ [95 %] | questions | clusters | draw seed |", "|---|---|---|---|---|---|---|"]
    for s in P1_SETS:
        e = (p1.get("per_set") or {}).get(s) or {}
        lines.append(f"| {s} | {_fmt(e.get('a'))} | {_fmt(e.get('b'))} | {_cid(e)} | {e.get('n', '—')} | {e.get('clusters', '—')} | {(p1.get('set_seeds') or {}).get(s, '—')} |")
    c = v["P1"]["checks"]
    lines += [f"| **mean** | | | {_cid(p1)} | | | |", "", f"Mean ≤ −0.05: {c['mean_le_-0.05']} · upper < 0: {c['mean_high_lt_0']} → **P1 {'PASS' if v['P1']['pass'] else 'FAIL'}**", ""]
    b8 = r.get("B8") or {}
    p2 = b8.get("P2") or {}
    lines += ["## P2 — gate B8 (fresh bench-shaped requests), FP − F, balanced orders", "",
              f"{b8.get('p2_units', '—')} questions in {b8.get('requests', '—')} requests (per env {b8.get('per_env_units')}); {b8.get('planner_units', '—')} with a planner "
              f"recommendation; excluded requests {len(b8.get('excluded_requests') or [])}.", "",
              "| reading | FP | F | FP − F [95 %] | check |", "|---|---|---|---|---|",
              _p2_row("passive-top share (≤ −0.15)", p2.get("passive_top"), v["P2"]["checks"]["passive_top_le_-0.15"]),
              _p2_row("planner agreement (≥ −0.05)", p2.get("planner"), v["P2"]["checks"]["planner_ge_-0.05"]),
              _p2_row("passive mass (reported)", p2.get("passive_mass"), ""),
              _p2_row("passive-top: hold — stop / wait / hold (reported)", (b8.get("P2_split") or {}).get("passive_top_hold"), ""),
              _p2_row("passive-top: continue unchanged (reported)", (b8.get("P2_split") or {}).get("passive_top_continue"), ""),
              "", f"**P2 {'PASS' if v['P2']['pass'] else 'FAIL'}**", "", f"## Outcome: **{v['outcome']}**", "",
              "## Guards (FP − F; reported with pass / fail — they do not change the outcome label)", "", "| guard | reading | pass |", "|---|---|---|"]
    gc = v["guards"]["checks"]
    g = r.get("guards") or {}
    lines.append(f"| outside CA, mean of three (≥ −0.05) | {_cid(g.get('CA'))} | {gc['outside_CA_mean_ge_-0.05']} |")
    lines.append(f"| outside GAP, mean of three (≥ −0.03) | {_cid(g.get('GAP'))} | {gc['outside_GAP_mean_ge_-0.03']} |")
    lines.append(f"| T2–T4 balanced band (lower > −0.05) | {_cid(g.get('band'))} | {gc['T2-T4_band_low_gt_-0.05']} |")
    sev = g.get("severe") or {}
    lines.append(f"| T2–T4 severe errors (FP ≤ F + 0.01) | FP {_fmt(sev.get('a'))} vs F {_fmt(sev.get('b'))} | {gc['T2-T4_severe_FP_le_F_plus_0.01']} |")
    g2 = g.get("G2") or {}
    lines.append(f"| G2 accuracy (≥ −0.02, lower > −0.045) | {_fmt(g2.get('diff'), signed=True)} [{_fmt(g2.get('low'), signed=True)}, {_fmt(g2.get('high'), signed=True)}] | "
                 f"{gc['G2_point_ge_-0.02'] and gc['G2_low_gt_-0.045']} |")
    m8 = r.get("M8g") or {}
    fm = m8.get("FORMAT") or {}
    fc = v["FORMAT"]["checks"]
    lines += ["", "## FORMAT — arm F on gate M8g", "", f"{fm.get('listings', '—')} listings scored by F's and v0.7's seeds.", "",
              "| bin | acc two-letter [95 %] | acc single-letter [95 %] | ratio | two − 0.8 · single [95 %] | check |", "|---|---|---|---|---|---|"]
    for name in FORMAT_BINS:
        e = (fm.get("bins") or {}).get(name) or {}
        lines.append(f"| {name} | {_cid(e.get('acc_two'), signed=False)} | {_cid(e.get('acc_single'), signed=False)} | {_fmt(e.get('ratio'))} | {_cid(e.get('margin'))} | "
                     f"{fc[f'{name}_two_ge_0.8_x_single']} |")
    n26 = fm.get("n26") or {}
    lines += [f"| N ≤ 26: F − v0.7 (≥ −0.02) | F {_fmt(n26.get('a'))} vs v0.7 {_fmt(n26.get('b'))} | | | {_cid(n26)} | {fc['N<=26_F_minus_v07_ge_-0.02']} |", "",
              f"**FORMAT {'PASS' if v['FORMAT']['pass'] else 'FAIL'}**", ""]
    cons = v.get("consistency") or {}
    orig = v.get("consistency_original_reading") or {}
    lines += ["## Consistency-loss condition (amendment, 2026-10-16 ~13:39 KST, before any v0.8 data): F's flip rate between two order seeds > 0.20 "
              "in ANY source", "", "| source | F flip rate | over 0.20 |", "|---|---|---|"]
    for name in cons.get("sources") or []:
        value = (cons.get("read") or {}).get(name)
        lines.append(f"| {name} | {_fmt(value) if value is not None else 'unread'} | {'—' if value is None else value > FLIP_LIMIT} |")
    lines += ["", f"**{cons.get('status', 'UNREAD')}** (R14: {cons.get('r14')}). Applied as written: not conditional on "
              f"FORMAT (FORMAT {'PASS' if v['FORMAT']['pass'] else 'FAIL'}). The original wording ('FORMAT passes but F's flip rate in bins >= 27 > "
              f"0.20') required FORMAT to pass: F's flip rate in bins ≥ 27 {_fmt(orig.get('flip_F_ge27'))}, so that reading would "
              f"{'' if orig.get('would_add_step') else 'not '}add the consistency step" + ("" if orig.get("readable") else " (unread)") + ".", "",
              f"{cons.get('reported_separately', R16_SEPARATE)}.", ""]
    of = st.get("O-F") or {}
    lines += ["## O-F — release guard: F − v0.7 short-form outside DC (mean of the three sets, ≤ +0.02)", "",
              "| set | F − v0.7 [95 %] |", "|---|---|"] + [f"| {s} | {_cid((of.get('per_set') or {}).get(s))} |" for s in P1_SETS]
    lines += [f"| **mean** | {_cid(of)} |", "", f"**O-F {'PASS' if v['O-F']['pass'] else 'FAIL'}**", "",
              "## Next steps (pre-declared) and release", ""]
    ns = v["next_steps"]
    lines += [f"- **{b}** → {n}." for b, n in zip(ns["branches"], ns["next"])] + [f"- {n}." for n in ns.get("notes") or []]
    lines += [f"- {v['release_line']}", "", "## Reported, not deciding", "", "### M8g per bin", "",
              "| model / arm | bin | listings | accuracy [95 %] | ECE | NLL of the key | flip rate (records) | RStd (label) | RStd (pick rate) | acc two / single |",
              "|---|---|---|---|---|---|---|---|---|---|"]
    for model, entry in (m8.get("per_bin") or {}).items():
        for name, e in entry.items():
            lines.append(f"| {LABELS.get(model, model)} | {name} | {e['listings']} | {_cid(e.get('accuracy'), signed=False)} | {_fmt(e.get('ece'))} | {_fmt(e.get('nll'))} | "
                         f"{_fmt(e.get('flip_rate'))} ({e.get('flip_records')}) | {_fmt((e.get('rstd') or {}).get('value'))} | {_fmt((e.get('pick_rstd') or {}).get('value'))} | "
                         f"{_fmt(e.get('acc_two'))} / {_fmt(e.get('acc_single'))} |")
    lines += ["", "Single against balanced (M8g bins ≤ 60, the same records):", "", "| model | bin | records | single | balanced |", "|---|---|---|---|---|"]
    for model, entry in (m8.get("single_vs_balanced") or {}).items():
        for name, e in entry.items():
            lines.append(f"| {LABELS.get(model, model)} | {name} | {e['records']} | {_fmt(e['single'])} | {_fmt(e['balanced'])} |")
    lines += ["", "### M8g per source (general = MMLU, physics = PHYS; reported — FORMAT unchanged)", "",
              f"Note: {m8.get('per_source_note', PER_SOURCE_NOTE)}.", "",
              "| model / arm | source | bin | listings | accuracy | acc two / single | margin two − 0.8 · single | flip rate (records) | NLL |", "|---|---|---|---|---|---|---|---|---|"]
    for model, by_source in (m8.get("per_source") or {}).items():
        for source, by_bin in by_source.items():
            for name, e in by_bin.items():
                lines.append(f"| {LABELS.get(model, model)} | {source} | {name} | {e['listings']} | {_fmt(e['accuracy'])} | {_fmt(e['acc_two'])} / {_fmt(e['acc_single'])} | "
                             f"{_fmt(e['margin'], signed=True)} | {_fmt(e['flip_rate'])} ({e['flip_records']}) | {_fmt(e['nll'])} |")
    lines += ["", "### B8 levels per model (balanced orders, P2 questions)", "",
              "| model / arm | passive-top | hold | continue | planner agreement | passive mass |", "|---|---|---|---|---|---|"]
    for model, e in (b8.get("levels") or {}).items():
        lines.append(f"| {LABELS.get(model, model)} | {_cid(e.get('passive_top'), signed=False)} | {_cid(e.get('passive_top_hold'), signed=False)} | "
                     f"{_cid(e.get('passive_top_continue'), signed=False)} | {_cid(e.get('planner'), signed=False)} | {_cid(e.get('passive_mass'), signed=False)} |")
    lines += ["", "Per environment (hold / continue): " + "; ".join(
        f"{LABELS.get(m, m)} " + ", ".join(f"{env} {_fmt(x.get('passive_top_hold'))} / {_fmt(x.get('passive_top_continue'))}" for env, x in e.items())
        for m, e in (b8.get("levels_per_env") or {}).items()) + "."]
    lines += ["", "### Flip rates and pick-rate RStd per source (single order seed 1 vs seed 2; RStd on the seed-1 single order)", ""]
    sources_ = [("B8 (every tagged question)", b8.get("flips") or {})]
    for title, f in sources_:
        lines += ["", f"{title}:", "", "| model / arm | pooled flip (n) | per group | per bin | pick-rate RStd pooled | per bin |", "|---|---|---|---|---|---|"]
        for model in [m for m in (*MODELS, *ARMS) if m in (f.get("pooled") or {}) or m in (f.get("pick_rstd") or {})]:
            pooled = (f.get("pooled") or {}).get(model) or {}
            groups = ", ".join(f"{g} {_fmt(x['value'])}" for g, x in ((f.get("per_group") or {}).get(model) or {}).items())
            bins = ", ".join(f"{b} {_fmt(x['value'])}" for b, x in ((f.get("by_bin") or {}).get(model) or {}).items())
            prs = (f.get("pick_rstd") or {}).get(model) or {}
            prb = ", ".join(f"{b} {_fmt(x.get('value'))}" for b, x in ((f.get("pick_rstd_by_bin") or {}).get(model) or {}).items())
            lines.append(f"| {LABELS.get(model, model)} | {_fmt(pooled.get('value'))} ({pooled.get('questions', 0)}) | {groups or '—'} | {bins or '—'} | "
                         f"{_fmt(prs.get('value'))} | {prb or '—'} |")
    lines += _render_r14(r.get("R14") or {})
    lines += _render_r16(r.get("A16") or {}, r.get("S16") or {})
    lines += ["", "Long lists:", ""]
    for name, e in (b8.get("long_lists") or {}).items():
        lines.append(f"- {name} ({e['questions']} questions, {e['options']} options): " + "; ".join(_long_list(m, x) for m, x in e["models"].items()))
    lines += ["", "Bench hold rule (context, as rel07): " + "; ".join(f"{LABELS.get(m, m)} {e['overrides_run']}/{e['questions']} ({e['passive_overrides_run']} passive)"
                                                                     for m, e in (b8.get("hold_rule") or {}).items()) + "."]
    jev = r.get("Jev") or {}
    lines += ["", "### Jev top-1 agreement (comparison set, not sealed)", ""]
    if jev.get("absent"):
        lines.append(f"absent: {jev['absent']}")
    else:
        lines.append(f"{jev.get('questions')} questions, Jev answered {jev.get('jev_answered')}; Jev's accuracy on the many part {_fmt((jev.get('jev_accuracy_many') or {}).get('value'))}.")
        lines += ["", "| model | part / reading | agreement | questions | accuracy |", "|---|---|---|---|---|"]
        for model, e in (jev.get("models") or {}).items():
            for key, x in e.items():
                lines.append(f"| {LABELS.get(model, model)} | {key} | {_fmt(x['agreement'])} | {x['questions']} | {_fmt(x['accuracy'])} |")
    probes = r.get("probes") or {}
    lines += ["", "### rel071 storage-bin probe cases", ""]
    if probes.get("absent"):
        lines.append(f"absent: {probes['absent']}")
    else:
        lines += ["| model | case | reading | p(right) | rank | two-letter mass |", "|---|---|---|---|---|---|"]
        for model, cases in (probes.get("models") or {}).items():
            for rid, e in cases.items():
                for reading, x in e["readings"].items():
                    lines.append(f"| {LABELS.get(model, model)} | {rid} | {reading} | {_fmt(x['p_right'], 4)} | {x['rank_right']} | {_fmt(x.get('two_letter_mass'), 4)} |")
    lines += ["", "### Short-form context", "", "| set | model / arm | DC | CA | GAP |", "|---|---|---|---|---|"]
    for s, by_model in (st.get("context") or {}).items():
        for model, e in by_model.items():
            lines.append(f"| {s} | {LABELS.get(model, model)} | {_cid(e.get('DC'), signed=False)} | {_cid(e.get('CA'), signed=False)} | {_cid(e.get('GAP'))} |")
    lines += ["", "FP − F per set (H-set and B1 reported): " + "; ".join(f"{s} ΔDC {_cid((d or {}).get('DC'))}, ΔCA {_cid((d or {}).get('CA'))}, ΔGAP {_cid((d or {}).get('GAP'))}"
                                                                      for s, d in (st.get("FP_minus_F") or {}).items()) + "."]
    lines += ["", "KL(teacher ‖ model) on choice questions with spread labels: " + "; ".join(
        f"{LABELS.get(m, m)} " + ", ".join(f"{b} {_fmt(x['value'])} (n {x['questions']})" for b, x in e.items()) for m, e in (st.get("teacher_kl") or {}).items()) + "."]
    ph = r.get("posthoc") or {}
    lines += ["", f"- {TEACHER_SPREAD}.", f"- Post-hoc prior-removal estimate (expected size of P1, v0.7's predictions): "
              + (f"{_cid(ph.get('P1_estimate'))}" if ph.get("P1_estimate") else f"absent ({ph.get('absent')})") + "."]
    lines += ["", "## Notes", ""] + [f"- {note}" for note in result.get("notes") or []] + ["", "## Rule (verbatim parts)", ""] + [f"- **{k}** — {t}" for k, t in RULE.items()]
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------
# Pipeline check (existing v0.6 / v0.7 short-form files; not the verdict)
# --------------------------------------------------------------------------


def _sci(value: Any) -> str:
    return "—" if _num(value) is None else f"{float(value):.1e}"


def _close(a: Any, b: Any, tol: float = 1e-9) -> bool:
    return _num(a) is not None and _num(b) is not None and abs(float(a) - float(b)) <= tol


def pipeline_check(*, n_boot: int, seed: int, out: Path) -> dict[str, Any]:
    """P1's and the CA / GAP guard's path with F := v0.6 arm and FP := v0.7 arm on A4, T9, T8b-B2 (their Spark files), against v0.7's verdict."""
    notes: list[str] = []
    paths = {}
    for seed_ in SEEDS:
        for s in P1_SETS:
            paths[(f"v08Fs{seed_}", s)] = V06_SPARK / f"{s}-v06s{seed_}-2b.predictions.jsonl"
            paths[(f"v08FPs{seed_}", s)] = V07_SPARK / f"{s}-v07s{seed_}-2b.predictions.jsonl"
    sets, preds, sources = load_standard(paths, allow_mismatch=False, notes=notes)
    calls = calls_of(sets, preds, P1_SETS)
    v5 = V5()
    reading = {"P1": mean_of_sets(calls, "FP", "F", metric="DC", n_boot=n_boot, seed=seed),
               "CA": mean_of_sets(calls, "FP", "F", metric="CA", n_boot=n_boot, seed=seed),
               "GAP": mean_of_sets(calls, "FP", "F", metric="GAP", n_boot=n_boot, seed=seed),
               "P1_v07_draws": mean_of_sets(calls, "FP", "F", metric="DC", n_boot=n_boot, seed=seed, set_seeds=[seed] * len(P1_SETS))}
    v07 = json.loads((ARTIFACTS / "reports" / "v07" / "verdict.json").read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = []

    def add(what: str, got: Any, want: Any) -> None:
        rows.append({"what": what, "got": _num(got), "want": _num(want), "diff": None if _num(got) is None or _num(want) is None else float(got) - float(want),
                     "ok": _close(got, want)})

    for s in P1_SETS:
        ref = v07["readings"]["standard"]["O"][s]
        mine = v5.call_difference(calls[("FP", s)], calls[("F", s)], n_boot=n_boot, seed=seed)["DC"]
        for key in ("value", "low", "high", "a", "b"):
            add(f"{s} DC v0.7 − v0.6 {key} (v0.7's O; seed {seed} per set)", mine.get(key), ref.get(key))
        p1s = reading["P1"]["per_set"][s]
        add(f"{s} DC v0.7 − v0.6 point (P1 path)", p1s.get("value"), ref.get("value"))
        p1v = reading["P1_v07_draws"]["per_set"][s]
        add(f"{s} DC v0.7 − v0.6 low (P1 path, seed {seed} per set)", p1v.get("low"), ref.get("low"))
        add(f"{s} DC v0.7 − v0.6 high (P1 path, seed {seed} per set)", p1v.get("high"), ref.get("high"))
        ctx = v07["readings"]["standard"]["context"][s]
        for mine_name, theirs in (("v08Fs0", "v06s0"), ("v08Fs1", "v06s1"), ("F", "v06"), ("v08FPs0", "v07s0"), ("v08FPs1", "v07s1"), ("FP", "v07")):
            metrics = v5.call_metrics(calls[(mine_name, s)], n_boot=0)
            for metric in ("DC", "CA", "GAP"):
                add(f"{s} {metric} {theirs}", (metrics.get(metric) or {}).get("value"), ((ctx.get(theirs) or {}).get(metric) or {}).get("value"))
        for metric in ("CA", "GAP"):
            want = (((v07["readings"]["standard"].get("secondary") or {}).get("calls_v07_minus_v06") or {}).get(s) or {}).get(metric) or {}
            if want:
                add(f"{s} {metric} v0.7 − v0.6 point (guard path)", reading[metric]["per_set"][s].get("value"), want.get("value"))
    reproduction = {"rows": rows, "all_ok": all(r["ok"] for r in rows), "failing": [r for r in rows if not r["ok"]],
                    "max_abs_diff": max(abs(r["diff"]) for r in rows if r["diff"] is not None)}
    result = {"kind": "v08-verdict-pipeline-check", "script_version": SCRIPT_VERSION, "git": _git_sha(), "created": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
              "note": "PIPELINE CHECK ONLY — not the v0.8 verdict: F := v0.6-s0 + v0.6-s1, FP := v0.7-s0 + v0.7-s1 (Spark short-form A4, T9, T8b-B2)",
              "n_boot": n_boot, "seed": seed, "readings": {k: v for k, v in reading.items()}, "reproduction": reproduction, "sources": sources, "notes": notes}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    md = ["# v0.8 verdict helper — PIPELINE CHECK ONLY (not the v0.8 verdict)", "", result["note"] + ".", "",
          f"Reproduced: {reproduction['all_ok']} (max |diff| {reproduction['max_abs_diff']:.2e}, {len(rows)} numbers against artifacts/reports/v07/verdict.json).", "",
          f"Side reading — P1's path on the stand-in arms (\"FP − F\" = v0.7 − v0.6): mean {_cid(reading['P1'])} (draw seeds {reading['P1']['set_seeds']}); "
          f"CA mean {_cid(reading['CA'])}; GAP mean {_cid(reading['GAP'])}.", "", "| reading | got | reference | diff |", "|---|---|---|---|"]
    md += [f"| {r['what']} | {_fmt(r['got'], 4)} | {_fmt(r['want'], 4)} | {_sci(r['diff'])} |" for r in rows]
    out.with_suffix(".md").write_text("\n".join(md) + "\n", encoding="utf-8")
    return result


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------


def compute(standard_sets: Mapping[str, Sequence], standard_preds: Mapping[str, Mapping[str, Sequence]], pod_sets: Mapping[str, Sequence],
            pod_preds: Mapping[str, Mapping[str, Sequence]], b8: Mapping[str, Any], m8g: Mapping[str, Any], *, jev: Mapping[str, Any] | None = None,
            probes: Mapping[str, Any] | None = None, r14: Mapping[str, Any] | None = None, r16: Mapping[str, Any] | None = None,
            posthoc: Path | None = None, n_boot: int = 2000, seed: int = 0) -> dict[str, Any]:
    """Every reading and the verdict from in-memory inputs. `b8` = {tags, records (by id), scores (by model), exclude}; `m8g` = {records,
    preds (model → kind → predictions), exclude}; `jev` = {records, jev_preds, scores}; `probes` = {rows, scores}; `r14` = :func:`load_r14`'s
    result ({tags, record_ids, scores, …}, or {refused} — then R14 is listed unread; None: no R14 score files); `r16` = :func:`load_r16`'s
    result (A16 / S16, reported only: {records, scores, stored, …}, {refused}, or None when absent)."""
    standard = standard_readings(standard_sets, standard_preds, n_boot=n_boot, seed=seed)
    home = home_readings(pod_sets, pod_preds, n_boot=n_boot, seed=seed)
    g2 = g2_reading(pod_sets.get("G2"), pod_preds, n_boot=n_boot, seed=seed)
    b8r = b8_readings(b8["tags"], b8["records"], b8["scores"], n_boot=n_boot, seed=seed, exclude=b8.get("exclude") or ())
    m8r = m8g_readings(m8g["records"], m8g["preds"], n_boot=n_boot, seed=seed, exclude=m8g.get("exclude") or ())
    guards = {"CA": {k: standard["CA"].get(k) for k in ("value", "low", "high", "per_set")},
              "GAP": {k: standard["GAP"].get(k) for k in ("value", "low", "high", "per_set")},
              "band": home.get("band"), "severe": home.get("severe"), "G2": g2}
    readings = {"standard": standard, "home": home, "guards": guards, "B8": b8r, "M8g": m8r,
                "Jev": jev_readings(jev["records"], jev["jev_preds"], jev["scores"]) if jev else {"absent": "no Jev comparison files"},
                "probes": probe_readings(probes["rows"], probes["scores"]) if probes else {"absent": "no probe files"},
                "R14": (({"refused": r14["refused"]} if r14.get("refused") else
                         {**r14_readings(r14_units(r14["tags"], r14.get("record_ids")), r14.get("scores") or {}),
                          "left_out": r14.get("left_out") or [], "check": r14.get("check"), "frozen": (r14.get("frozen") or {}).get("frozen")})
                        if r14 else {"absent": "no R14-<tag>.json score files"}),
                **r16_part(r16),
                "posthoc": posthoc_reading(posthoc) if posthoc is not None else posthoc_reading()}
    flips = {f"M8g {name}": ((m8r.get("flip_F_by_bin") or {}).get(name) or {}).get("value") for name in BIN_NAMES}
    flips["B8"] = (((b8r.get("flips") or {}).get("pooled") or {}).get("F") or {}).get("value")
    if r14:
        flips[R14] = ((((readings["R14"].get("models") or {}).get("F")) or {}).get("pooled") or {}).get("flip_rate")
    rule_inputs = {"P1": {k: standard["P1"].get(k) for k in ("value", "low", "high")}, "P2": b8r.get("P2"), "guards": guards,
                   "FORMAT": m8r.get("FORMAT"), "O-F": {k: standard["O-F"].get(k) for k in ("value", "low", "high")},
                   "flip_F_ge27": (m8r.get("flip_F_ge27") or {}).get("value"), "consistency_flips": flips, "r14_provided": r14 is not None}
    return {"verdict": decide(rule_inputs), "readings": readings}


def r16_part(r16: Mapping[str, Any] | None) -> dict[str, Any]:
    """The A16 and S16 readings (reported only; they enter neither P1, P2, FORMAT nor the amended condition)."""
    if not r16:
        return {"A16": {"absent": "no frozen A16 / S16 files"}, "S16": {"absent": "no frozen A16 / S16 files"}}
    if r16.get("refused"):
        return {"A16": {"refused": r16["refused"]}, "S16": {"refused": r16["refused"]}}
    common = {"left_out": r16.get("left_out") or [], "frozen": (r16.get("frozen") or {}).get("frozen"), "excluded": r16.get("excluded") or {},
              "note": R16_SEPARATE}
    a16 = a16_readings(r16["records"][A16], r16["scores"].get(A16) or {}, r16["scores"].get(A16Y) or {}, (r16.get("stored") or {}).get(A16) or {})
    s16 = s16_readings(r16["records"][S16], r16["scores"].get(S16) or {}, (r16.get("stored") or {}).get(S16) or {})
    return {"A16": {**a16, **common, "check": r16.get("check") or []}, "S16": {**s16, **common}}


def _scores(spark_dir: Path, set_name: str, input_path: Path | None, readings: Sequence[str], cks: Mapping[str, Path | None], *, frozen=None,
            models: Sequence[str] = MODELS, required_models: Sequence[str] = (), problems: list[str], notes: list[str]) -> dict[str, Any]:
    out = {}
    for m in models:
        path = Path(spark_dir) / f"{set_name}-{SPARK_TAGS[m]}.json"
        if not path.is_file():
            (problems if m in required_models else notes).append(f"{'missing' if m in required_models else 'absent'}: {path}")
            continue
        data, found = check_bench_source(path, set_name=set_name, tag=SPARK_TAGS[m], input_path=input_path, checkpoint=cks.get(m), readings=readings, frozen=frozen)
        problems += [f"{m} {set_name}: {p}" for p in found]
        out[m] = data
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    for name in ("f-s0-dir", "f-s1-dir", "fp-s0-dir", "fp-s1-dir"):
        parser.add_argument(f"--{name}", type=Path, default=None, help="fetched reports of the run (<run>-fetched/reports; pods T2–T4, G2)")
    parser.add_argument("--spark-dir", type=Path, default=SPARK_DIR, help="the v0.8 chain's Spark outputs")
    parser.add_argument("--pipeline-check", action="store_true", help="existing v0.6 / v0.7 short-form files only, arms relabelled; never the verdict")
    parser.add_argument("--no-v08-recheck", action="store_true", help="read the gates without overlap-v08.json (tests and pipeline checks only)")
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--allow-mismatch", action="store_true", help="read even if a prediction file fails its checks (never the frozen-tag check)")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)
    if args.pipeline_check:
        out = args.out or PIPELINE_OUT
        refuse_write(out)
        result = pipeline_check(n_boot=args.n_boot, seed=args.seed, out=out)
        print(out.with_suffix(".md").read_text(encoding="utf-8"))
        return 0 if result["reproduction"]["all_ok"] else 1
    out = args.out or OUT
    refuse_write(out)
    if out.resolve() == PIPELINE_OUT.resolve():
        raise SystemExit("the verdict is not written into the pipeline-check folder")
    missing_dirs = [f"--{POD_ARGS[m].replace('_', '-')}" for m in V08_MODELS if getattr(args, POD_ARGS[m]) is None]
    if missing_dirs:
        raise SystemExit(f"required: {' '.join(missing_dirs)} (the four v0.8 runs' fetched reports)")
    missing = required(args)
    if missing:
        raise SystemExit("missing files the rule reads:\n  " + "\n  ".join(missing))
    notes: list[str] = []
    problems: list[str] = []
    exclusions = {"items": [], "b8": [], "a16": [], "s16": []}
    if G8.OVERLAP_V08.is_file():
        exclusions = read_exclusions()
        notes.append(f"overlap-v08.json (training input sha256 {str(exclusions.get('train_sha256'))[:12]}): {len(exclusions['items'])} M8g items and "
                     f"{len(exclusions['b8'])} B8 requests left out.")
    elif not args.no_v08_recheck:
        raise SystemExit(f"{G8.OVERLAP_V08} is missing — run scripts/v08_gates_build.py recheck-v08 on v0.8's training input first")
    else:
        notes.append("overlap-v08.json absent: no exclusions (--no-v08-recheck)")
    frozen, frozen_problems = check_b8_frozen()
    if frozen_problems:
        raise SystemExit("B8's tags are not the frozen ones — refusing:\n  " + "\n  ".join(frozen_problems))
    if not getattr(G8, "OPTION_RANGE_VERSION_CONFIRMED", True):
        notes.append(f"the gates carry the opt-in version {G8.OPTION_RANGE_VERSION!r}, not yet marked as v0.8a's confirmed name.")
    cks = checkpoints(args)
    sets, preds, sources = load_standard(standard_paths(args.spark_dir), allow_mismatch=args.allow_mismatch, notes=notes)
    pod_sets, pod_preds, pod_sources = load_pods(args, allow_mismatch=args.allow_mismatch)
    b8_records = {str(r["id"]): r for r in _read(G8.B8_FILE)}
    tags = json.loads(TAGS_FILE.read_text(encoding="utf-8"))
    b8_scores = _scores(args.spark_dir, "B8", G8.B8_FILE, B8_READINGS, cks, frozen=frozen, required_models=V08_MODELS, problems=problems, notes=notes)
    m8g_records = _read(G8.M8G_FILE)
    m8g_preds: dict[str, dict[str, list]] = {}
    for m in MODELS:
        for kind, stem in M8G_KINDS.items():
            path = Path(args.spark_dir) / f"{stem}-{SPARK_TAGS[m]}.predictions.jsonl"
            if not path.is_file():
                notes.append(f"absent: {path}")
                continue
            problems += [f"{m} {stem}: {p}" for p in check_m8g_source(path, kind, cks.get(m))]
            m8g_preds.setdefault(m, {})[kind] = _read(path)
    jev = None
    if JEV_SET.is_file() and JEV_ANSWERS.is_file():
        jev_scores = _scores(args.spark_dir, "Jev", JEV_SET, ("single-s1",), cks, problems=problems, notes=notes)
        jev = {"records": _read(JEV_SET), "jev_preds": _read(JEV_ANSWERS), "scores": jev_scores}
    probes = None
    if PROBE_ROWS.is_file():
        probes = {"rows": _read(PROBE_ROWS), "scores": _scores(args.spark_dir, "Probe", PROBE_ROWS, (), cks, problems=problems, notes=notes)}
    r14 = load_r14(args.spark_dir, cks, notes)
    r16 = load_r16(args.spark_dir, cks, notes, exclude={"a16": exclusions.get("a16") or [], "s16": exclusions.get("s16") or []})
    if r16 and r16.get("refused"):
        notes.append("A16 / S16's frozen files changed — their readings are refused (the rest of the verdict stands): " + "; ".join(r16["refused"]))
    if r14 and r14.get("refused"):
        notes.append("R14's frozen files changed — its readings are refused (the rest of the verdict stands): " + "; ".join(r14["refused"]))
    if problems and not args.allow_mismatch:
        raise SystemExit("gate / bench files fail their checks:\n  " + "\n  ".join(problems[:40]))
    _log("computing readings")
    computed = compute(sets, preds, pod_sets, pod_preds, {"tags": tags, "records": b8_records, "scores": b8_scores, "exclude": exclusions["b8"]},
                       {"records": m8g_records, "preds": m8g_preds, "exclude": exclusions["items"]}, jev=jev, probes=probes, r14=r14, r16=r16,
                       n_boot=args.n_boot, seed=args.seed)
    notes += ["Seed means: an arm's call distribution per question = the mean of its seeds' 1/L-weighted listings; every other per-item value = the "
              "mean of its seeds' values, on the items both seeds have; arm comparisons pair the items both arms have.",
              "P1, CA, GAP and O-F: each set drawn by its own pair clusters with draw seed = seed + its index in (A4, T9, T8b) (v0.6's author-gap "
              "precedent), the arms paired within a set, the three differences averaged per draw.",
              "P2: a B8 question is the unit, its request the cluster; the top option is the balanced-order argmax (ties to the earlier option). "
              "P2 reads the frozen PASSIVE tag; its two parts — hold (stop / wait / hold) and continue (continue unchanged) — are reported next "
              "to it (robo_jev.bench.r14_v08.passive_kind of each option's tag rule).",
              "FORMAT: listings scored by both F seeds and both v0.7 seeds; item-cluster bootstrap intervals are reported, the rule reads points.",
              "Inclusive bounds compare with a 1e-12 tolerance; strict bounds as they are."]
    result = {"kind": "v08-verdict", "script_version": SCRIPT_VERSION, "git": _git_sha(), "created": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
              "rule_text": RULE, "n_boot": args.n_boot, "seed": args.seed, "pipeline_check": None, **computed,
              "sources": {**sources, **pod_sources}, "frozen": frozen, "problems": problems, "notes": notes,
              "inputs": {k: str(v) for k, v in vars(args).items() if isinstance(v, Path)}}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    md = render(result)
    out.with_suffix(".md").write_text(md, encoding="utf-8")
    print(md.split("## Reported, not deciding")[0])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
