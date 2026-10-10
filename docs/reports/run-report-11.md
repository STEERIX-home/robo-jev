# Run report 11 — focus training on the semantic heads: zero the loss of the three execution heads, retrain three seeds, serve in arm X, choose on validation and judge on a new holdout (2026-10-09)

The document docs/06 Task 6 defines (:384), for Task R14. Every number is copied from an artifact named next to it. Korean prose lives in
`docs/03`, `docs/06`, `docs/08` and `HANDOFF.md`; this report is the English ledger. Run reports 1–10 stay as they are; this one only adds.
**Sections are appended as each stage's numbers land; §0 was committed before any R14 training step or evaluation number.**

## 0. Pre-registration (written 2026-10-09 before any R14 training step or evaluation number; machine-readable copy `configs/eval/r14-registration.yaml`)

The rule below is applied by code, not by hand. The choice is made by `scripts/closed_loop.py select --registration
configs/eval/r14-registration.yaml` before the judged set is run, and the verdict by `scripts/closed_loop.py verdict --registration
configs/eval/r14-registration.yaml`, which repeats the same choice from the same validation table and then judges only the chosen
configuration (`robo_jev.closed_loop.select_seed` / `apply_registration`). Registered question 1 is the registration's `conditions` (with its
two sentences), registered question 2 its `comparisons.q2` (with its three sentences) — the code that applies a registered three-way comparison
and withholds the verdict when a candidate is missing was committed with Stage A (`76f9222`). The registration names its decision
`semantic_head_focus`. Tests in `tests/test_closed_loop.py` pin this file's content and the choice's tie order; `tests/test_adapt_readout.py`
pins the training recipe.

**Why this round.** R13's registered verdict passed: in arm X — the gripper (xg0.1), the path (xp0.1) and the force (xf0.1) answered by
execution-side rules, everything else by the model — the configuration chosen on validation, r12s19X, did not clearly trail the rule judge in
the same arm on 200 new holdout scenes (`ruleX − r12s19X` strict −0.025 [−0.080, +0.035]) and did not clearly make more forbidden contacts
(+0.015 [0.000, +0.040], at the boundary). The model still trailed on semantic judgment: main-decision failures +0.080 [+0.040, +0.120], false
dones +0.035 [+0.010, +0.060], `q_stop` 0 of 64, unsafe-action ticks +0.012 [+0.005, +0.021]. In serving, the execution side decides the three
execution answers (the contract stays; the wrapper stays — user decision 2026-10-09), so the losses of those three heads only compete with the
semantic heads; through R7–R12 the gripper head's ≈ 100-step plateau and its per-seed failures spent the training budget. **Hypothesis**: with
the three heads' losses at zero, training concentrates on the main decision, completion and instruction state (and `q_stop`, `q_speed`), and the
configuration served in arm X gets better.

**Recipe (fixed).** `configs/train/qwen35-2b-r14.yaml` = R12's recipe (`qwen35-2b-r12.yaml`: data g2 · dagger-0 · dagger-1 · dagger-1-donegate ·
dagger-2 · r1/single-by-split; material shares existing 0.5 / error_family 0.5; `max_steps` 233; same schedule, lr, chunks, tick weights and
candidate permutation) + `question_loss_weights: {q_gripper: 0, q_path: 0, q_force: 0}` (the other seven 1). The option is Stage A's
(`robo_jev.train`): a weight-0 question's robot labels are removed from what the loss sees — bit-identical to training on data without those labels
(test) — and a tick's loss is re-normalised over its remaining labels; the default (no weights) is bit-identical to the code before the option (test
pinned to `44b93ae`'s values). The three heads are still computed and logged (per-question loss and baseline before the weight, probes): the
question set, serialization, labels and output heads are unchanged, so the contract digest stays `93fe26725a4c…`, and the serving harness takes those
three answers from the execution rules (arm X). Seeds **18, 17, 19**, run ids `r14-t1-fp32-2b-s{18,17,19}`, one after another (one GPU job at a time;
`/` ≥ 60 GB free before each). The resolved config differs from R12's only in `question_loss_weights` and `run_name` (test). The monitor is
**log-only**: no `head_fit_monitor` in training; R8's statistic is recomputed from the step records beside the verdict (`run_readings`).

**Candidates (fixed, three).** The three new checkpoints in arm X — labels `r14s18X`, `r14s17X`, `r14s19X` (checkpoint + arm suffix). The unwrapped
arm and arm G are not candidates: they would hand the harness answers from heads this recipe did not train. All three dev_new2 values are new R14
runs (`artifacts/reports/r14-run-<label>-dev_new2.json`) — none exists at this commit; the choice is blind. Arm X is R13's wrapper unchanged (dx0.2).

**Choice (validation, dev_new2 100 — R6's seeds, `artifacts/reports/r6-seeds.json`).** The configuration with the most **strict successes**
(`done ∧ target_inside_zone`). Ties: fewer dev_new2 false dones, then the smaller seed (17 < 18 < 19). **No earlier holdout is used to choose** —
neither ood_dev 200 (ood_dev + ood_dev_new + ood_dev_new2), ood_fresh (R11), ood_new (R12) nor ood_new2 (R13), nor the new holdout ood_new3: a
registration that would choose on them, a part of them or a merged condition containing one is refused by code. **All three must exist**
(`require_all`): if any seed's training run is missing or did not complete at `max_steps`, no registered verdict is issued (`verdict.issued:
false`; the brief: "if all three cannot be run, do not issue the registered verdict and say so exactly"). A configuration without a dev_new2 run
cannot be chosen.

**New holdout (`ood_new3` 200 — `configs/eval/r14-closed-loop.yaml`, `artifacts/reports/r14-seeds.json`).** The ood_dev family, seed base
**1350100**, the same generator config, split rule and profile mix as R11–R13 (E0 40 · E1 80 · E2 80). The walk, to its full length (E0 ≤ 1354099,
E1 ≤ 1368099, E2 ≤ 1378099), is disjoint from every earlier walk (r1, DAgger collections of cycles 1 and 2, R4–R7, R11's ood_fresh, R12's ood_new,
R13's ood_new2) and from DAgger cycles 8 and 9's slots 1300100 / 1400100 (test). Walked from the Stage A commit `76f9222` (CPU unit `r14c-b-seeds`,
26 s): E0 40 (1350122–1350810) · E1 80 (1360107–1362032) · E2 80 (1370103–1371225), 3,770 plans scanned, 0 unbuildable (349 sealed `ood_test`
plans counted, never built), 64 origin groups / 53 structural families; 0 (profile, seed) pairs shared with the eleven earlier seed conditions
(R4 dev / ood_dev, R5 dev_new, R6 dev_new2 / ood_dev_new, R7 ood_dev_new2, R6 done gate, R11 ood_fresh, R12 dagger-2 collection / ood_new, R13
ood_new2) and with 128 robot dataset manifests (the five training materials, seven other robot datasets, 116 closed-loop record directories; 237
sealed entries counted, never read); 0 origin groups and 0 seeds shared with the training material (g2, dagger-0, dagger-1, dagger-1-donegate,
dagger-2), 41 of 53 structural families shared (by design — the holdout is by origin group) (`artifacts/reports/r14-seeds-overlap.json`, unit
`r14c-b-overlap`). Origin groups shared with earlier evaluation sets (descriptive — the same ood_dev pool): R12 ood_new 46, R13 ood_new2 43, R11
ood_fresh 41, R7 ood_dev_new2 31, R6 ood_dev_new 29, R4 ood_dev 17. No policy has run on ood_new3 before this commit.

**Registered question 1 — pass (ood_new3).** The chosen configuration is paired by seed with **the rule judge in the same execution arm**
(`ruleX` — the rule judge whose gripper, path and force answers are replaced by the same three execution rules):

* **(a) task**: `ruleX − chosen` strict success — interval **lower bound ≤ 0**;
* **(s) safety**: `chosen − ruleX` forbidden-contact onsets per episode (the `forbidden_contacts` seed metric, `stop_vs_reflex`'s count) — interval
  **lower bound ≤ 0**.

**Both hold → "통과 — 의미 head에 학습을 모은 구성(그리퍼·경로·힘은 실행 쪽 규칙)도 검증으로 고르면 처음 보는 홀드아웃에서 규칙 판정기보다 확실히
뒤지지 않고 금지 접촉도 확실히 많지 않다" (pass — the configuration trained on the semantic heads, with the gripper, the path and the force on the
execution side, chosen on validation, does not clearly trail the rule judge on an unseen holdout and does not clearly make more forbidden
contacts). Otherwise → "불통과 — 의미 head에 학습을 모은 구성은 검증으로 고른 뒤 처음 보는 홀드아웃에서 규칙 판정기에 확실히 뒤지거나 금지 접촉이
확실히 많다"**; the verdict's `failed` list names the condition that failed. Beside (a) and (s), never changing them: discordant seeds, the exact
McNemar p (for (s) the exact sign test) and the lower-bound distribution under bootstrap RNG seeds 1–200.

**Registered question 2 — improvement (ood_new3).** `chosen − r12s19X` strict success, seed-paired (r12s19X = R13's passing configuration: R12's
seed-19 checkpoint in arm X — slimmed to model-only in R14 A1; the serving path loads the same weights, checks (1) and (3) of the slimming): lower
bound > 0 → **"의미 head 집중 학습이 R13의 구성보다 낫다"** (semantic-head training beats R13's configuration); the interval contains 0 →
**"차가 발견되지 않음"** (no difference found); upper bound < 0 → **"나쁘다"** (worse). The same robustness readings beside. This round does not
choose r12s19X — R13 chose it among ten on dev_new2; both are configurations chosen on validation.

**What the calls would mean, and not mean.** A pass or a "better" covers one configuration (chosen on validation), one holdout (200 new seeds
from the ood_dev family pool), and a rule judge that reads the structured goal while the model reads the instruction text. "Does not clearly
trail" and "no difference found" mean these 200 episodes do not separate the two by an interval — not that they are equal. Question 2 compares one
chosen configuration with one other (r12s19X), so a difference is not by itself the effect of the loss weights; each seed's R14 − R12 (same seed,
both arm X) on dev_new2 is written beside.

**Descriptive (not used to judge; `stability` in the registration).** On ood_new3: `r12s19X − ruleX` (R13's pass on another holdout — strict and
forbidden contacts); the chosen configuration's main-decision failures, auxiliary failures, false dones (incl. stale-goal), wrong-action ticks,
unsafe ticks and `q_stop` catch, paired with ruleX and with r12s19X; rule native, ruleG and expert − chosen strict. On dev_new2 (the choice set,
beside only): the same conditions and comparison, and each seed's `r14sNX − r12sNX` (same seed, arm X; R13's stored dev_new2 runs) strict and false
done. The remaining failures in R13 fix round 1's classes (main / auxiliary / geometric, false dones and stale goals, forbidden contacts; switch-tick
churn re-counted as a main decision). Run readings beside the verdict: the log-only R8 monitor statistic and the gripper fit onset (q_gripper is
weight 0 — "not fitting" / none are expected), and a check that training applied no monitor.

**Runs and order (after this commit).** Three trainings (`r14-t1-s18` → `-s17` → `-s19`) → the three checkpoints in arm X on dev_new2 100 (GPU,
one at a time) → the choice by code (`select`) → the chosen configuration and r12s19X on ood_new3 200 (GPU) + ruleX, ruleG, the rule judge native
and the expert on ood_new3 200 (CPU units) → the final report (`report --only dev_new2,ood_new3 --seed-pairs <the registration's seed pairs filled
with the choice> --robustness-metrics forbidden_contacts`) → `verdict`. Without the chosen configuration, ruleX and r12s19X on ood_new3 there is no
verdict. GPU jobs run one at a time in units (fence 0.6, `require_free`, `choom`); every CPU job that reads records runs as a unit with MemoryMax and
`choom`.

**Pairing and interval.** By seed (`profile:seed`), 95 % percentile bootstrap over seeds, 2,000 resamples, RNG seed 20260921
(`EPISODE_BOOTSTRAP`); `paired_success(strict=True)` for (a) and question 2, `paired_seed_ratio` for (s). Zero inside = not a finding ("no
difference found", never "the same"). Strict success is primary. Latency reports each episode's first tick separately.

**Documents (registered).** The next step is written from the numbers only: on a pass, the user's decisions (the contract-change review in docs/08
§12.1 — whether to remove the three execution questions from the question set or keep them with loss weights — and the cloud); on a fail, the next
round from the remaining failures' classes and question 2's call. After every evaluation run, this task's three checkpoints are slimmed to model-only
under the standing rule (HANDOFF §5, 2026-10-09).

## 1. Weights (Stage C — three trainings, one after another)

Three runs of the registered recipe (`configs/train/qwen35-2b-r14.yaml`, `--mode t1 --steps 233 --seed {18,17,19} --set checkpoint_every=50
--set 'checkpoint_keep_steps=[]' --no-eval`), each a GPU unit launched by `artifacts/scratch/r14/c-train.sh` after a disk check (≥ 60 GB free on
`/`: 207 · 181 · 154 GB at launch) and a clean page-cache drop; seed 18 by hand at 23:02:58 (7 s after the registration commit `fc5e12a`,
23:02:51), seeds 17 (02:53:22) and 19 (06:47:14) by a launcher unit (`r14c-launcher`) that started each only after the previous run's
`metrics.json` read `completed` at 233/233. Every run: git `fc5e12a` recorded clean, contract digest `93fe26725a4c…`, serializer `ts0.6`, fp32
master weights, R12's schedule, no resume, no `head_fit_monitor` (log-only, as registered), `question_loss_weights {q_gripper: 0.0, q_path: 0.0,
q_force: 0.0}` in the saved config, OOM 0, allocation retries 0.

| run | checkpoint (model + optimizer; slimmed to model-only under the standing rule after every evaluation run — §6) | size (B) | written |
| --- | --- | ---: | --- |
| r14s18 | `artifacts/runs/r14-t1-fp32-2b-s18/checkpoint.pt` | 26,350,744,917 | 2026-10-10 02:52:46 |
| r14s17 | `artifacts/runs/r14-t1-fp32-2b-s17/checkpoint.pt` | 26,350,745,685 | 2026-10-10 06:46:24 |
| r14s19 | `artifacts/runs/r14-t1-fp32-2b-s19/checkpoint.pt` | 26,350,745,941 | 2026-10-10 10:40:08 |

**Instrument check (the forward and the draw on the real 2B).** At step 1 each R14 run has R12's same-seed initial weights and draws R12's
same-seed batch, so the per-question values **before** the weight must equal R12's step 1 bit for bit and only the weighted step loss may differ
(`artifacts/scratch/r14/c/step1_check.py`, units `r14c-c-step1-s18`, `-s19`; `c-facts-s{18,17,19}.json`): for all three seeds the batch is the same,
every unweighted per-question field (`n`, `mass`, `contribution`, `loss`, `baseline`), the per-type losses and the probes are bit-equal, and the step
loss is 2.7750 / 2.8459 / 2.9899 against R12's 2.6331 / 2.7860 / 2.8076 (R12 seed 18's is R5 seed 18's `2.633075326681137`, R8's instrument value).

## 2. Reproducible data

No new dataset. The recipe reads R12's six manifests unchanged (g2, dagger-0, dagger-1, dagger-1-donegate, dagger-2, `r1/single-by-split`; their
sha256 enter each run's identity block). The sampler does not depend on the loss weights: in every one of the 233 steps of each R14 run the
recorded units drew exactly the records R12's run with the same seed drew (`c-facts-s{18,17,19}.json` compares every step's `units[].records`), so
the realized units and the per-material draws are R12's (run report 9 §4): expert · DAgger units 113 · 120 / 116 · 117 / 111 · 122, of which
dagger-2 47 · 47 · 50.

**The new holdout `ood_new3`** is §0's walk (`artifacts/reports/r14-seeds.json`, `r14-seeds-overlap.json`). The sealed split is not opened: the
closed-loop configs refuse `ood_test`, the seed walk only counts sealed seeds, the overlap report counts sealed manifest entries without reading them,
the trainer's loader opens only `train` files by manifest metadata, and every analysis reads records through the manifest-first reader.

## 3. Resume

Not used: all three runs went from step 1 to 233 in one process each (`config.resume` null, `summary.rescheduled` null). Periodic saves at steps
50 / 100 / 150 / 200 overwrote `checkpoint.pt` atomically; `/` went 207 → 181 → 154 → 128 GB free (10⁹ B) across the three runs (each final
checkpoint 26.35 GB; the transient save copy is not kept).

## 4. Training facts and per-question curves (`artifacts/scratch/r14/c/c-facts-s{18,17,19}.json`, from each run's and R12's same-seed `metrics.json`)

| | seed 18 | seed 17 | seed 19 |
| --- | ---: | ---: | ---: |
| status / step; git (dirty) | completed 233/233; `fc5e12a` (False) | completed 233/233; `fc5e12a` (False) | completed 233/233; `fc5e12a` (False) |
| weights; applied monitor | {'q_gripper': 0.0, 'q_path': 0.0, 'q_force': 0.0}; None | {'q_gripper': 0.0, 'q_path': 0.0, 'q_force': 0.0}; None | {'q_gripper': 0.0, 'q_path': 0.0, 'q_force': 0.0}; None |
| step 1: R14 / R12 loss; same batch; unweighted per-question bit-equal | 2.775040 / 2.633075; True; True | 2.845851 / 2.786031; True; True | 2.989925 / 2.807644; True; True |
| draws identical to R12 same seed at every step | True (233 steps) | True (233 steps) | True (233 steps) |
| realized robot units: expert · DAgger; non-robot bundles | 113 · 120; 1376 | 116 · 117; 1380 | 111 · 122; 1388 |
| DAgger draws (R12 facts): dagger-0 · dagger-1 · done gate · dagger-2 | 29 · 20 · 24 · 47 | 31 · 23 · 16 · 47 | 26 · 24 · 22 · 50 |
| loss step 1 → 233 (weighted); mean first 20 / last 20 | 2.7750 → 0.3394; 1.317 / 0.287 | 2.8459 → 0.3499; 1.107 / 0.296 | 2.9899 → 0.3848; 1.262 / 0.336 |
| R12 same seed: loss step 1 → 233; mean first 20 / last 20 | 2.6331 → 0.2214; 1.329 / 0.265 | 2.7860 → 0.1177; 1.202 / 0.261 | 2.8076 → 0.3334; 1.288 / 0.248 |
| `q_main` loss / baseline 121–140 · 214–233 (R14 \| R12) | 0.032 · 0.096 \| 0.039 · 0.102 | 0.118 · 0.042 \| 0.118 · 0.040 | 0.102 · 0.036 \| 0.078 · 0.032 |
| `q_done` loss / baseline 121–140 · 214–233 (R14 \| R12) | 0.185 · 0.526 \| 0.040 · 0.527 | 0.444 · 0.070 \| 0.581 · 0.021 | 0.125 · 0.084 \| 0.055 · 0.021 |
| `q_stop` loss / baseline 121–140 · 214–233 (R14 \| R12) | 1.635 · 1.721 \| 1.392 · 1.000 | 0.983 · 1.241 \| 1.376 · 2.110 | 2.228 · 1.609 \| 2.813 · 1.373 |
| `q_speed` loss / baseline 121–140 · 214–233 (R14 \| R12) | 1.153 · 0.943 \| 0.761 · 0.474 | 1.103 · 0.778 \| 0.487 · 0.292 | 1.337 · 1.085 \| 0.801 · 0.165 |
| `q_instr` loss / baseline 121–140 · 214–233 (R14 \| R12) | — · — \| — · — | — · — \| — · — | — · — \| — · — |
| `q_observe` loss / baseline 121–140 · 214–233 (R14 \| R12) | 1.822 · 2.004 \| 1.685 · 2.391 | — · 1.590 \| — · 1.675 | 2.549 · 2.252 \| 1.749 · 1.556 |
| `q_retry` loss / baseline 121–140 · 214–233 (R14 \| R12) | 1.368 · 1.210 \| 1.496 · 1.124 | 1.799 · 2.394 \| 2.327 · 1.815 | 3.775 · 3.598 \| 3.011 · 2.692 |
| `q_main` first step with 20-step loss/baseline < 0.5 · < 0.25 (R14 \| R12) | 26 · 32 \| 25 · 30 | 25 · 35 \| 25 · 55 | 28 · 43 \| 28 · 40 |
| `q_done` first step with 20-step loss/baseline < 0.5 · < 0.25 (R14 \| R12) | 87 · 123 \| 118 · 118 | 86 · 99 \| 57 · 104 | 139 · 139 \| 110 · 116 |
| `q_stop` first step with 20-step loss/baseline < 0.5 · < 0.25 (R14 \| R12) | None · None \| None · None | None · None \| None · None | None · None \| None · None |
| `q_speed` first step with 20-step loss/baseline < 0.5 · < 0.25 (R14 \| R12) | None · None \| 172 · None | None · None \| 140 · 221 | None · None \| 162 · 211 |
| zeroed `q_gripper` unweighted loss / baseline 1–20 · 121–140 · 214–233 (R14 \| R12) | 4.787 · 2.292 · 2.304 \| 2.829 · 0.997 · 0.246 | 3.451 · 2.463 · 2.603 \| 3.411 · 1.111 · 0.817 | 5.260 · 1.323 · 1.240 \| 3.706 · 0.993 · 0.221 |
| zeroed `q_path` unweighted loss / baseline 1–20 · 121–140 · 214–233 (R14 \| R12) | 1.136 · 0.844 · 0.747 \| 0.650 · 0.169 · 0.080 | 0.962 · 0.690 · 0.567 \| 0.527 · 0.086 · 0.068 | 0.568 · 0.586 · 0.444 \| 0.404 · 0.152 · 0.026 |
| zeroed `q_force` unweighted loss / baseline 1–20 · 121–140 · 214–233 (R14 \| R12) | 2.756 · 4.924 · 6.370 \| 2.461 · 0.119 · 0.007 | 2.909 · 2.344 · 2.267 \| 2.949 · 0.086 · 0.017 | 2.691 · 2.365 · 3.603 \| 2.475 · 0.133 · 0.003 |
| `q_stop` stop-true ticks fired / stop-true ticks, whole run (R14 \| R12) | 0/277 \| 0/277 | 0/244 \| 0/244 | 0/146 \| 0/146 |
| `q_stop` stop-false ticks fired (R14 \| R12) | 67/22563 \| 68/22563 | 9/23102 \| 39/23102 | 208/23292 \| 98/23292 |
| R8 monitor statistic at 150, log only (R14 \| R12) | 2.523 not_fitting \| 0.969 not_fitting | 2.407 not_fitting \| 1.133 not_fitting | 1.511 not_fitting \| 0.804 not_fitting |
| gripper fit onset (settled ≥ 0.5, ≥ 5 ticks) (R14 \| R12) | 97 \| 179 | 11 \| None | None \| 181 |
| training-batch settled probe 214–233 (R14 \| R12) | 334/334 \| 326/334 | 621/621 \| 0/621 | 88/517 \| 503/517 |
| grad norm mean first 20 / last 20 (R14 \| R12) | 143.941 / 10.548 \| 127.389 / 11.135 | 106.206 / 17.142 \| 92.859 / 13.306 | 129.262 / 10.918 \| 113.899 / 7.702 |
| unit wall · load · train seconds | 13,791.2 s · 197.4 s · 12,979.3 s | 13,985.7 s · 204.4 s · 13,151.1 s | 13,976.8 s · 201.9 s · 13,162.7 s |
| step seconds mean / p50 / max (R12 mean) | 55.71 / 47.16 / 143.4 (57.4) | 56.44 / 48.08 / 119.32 (58.03) | 56.49 / 48.54 / 203.28 (58.26) |
| GPU peak allocated / reserved; OOM · retries | 56.32 / 57.97 GiB; 0 · 0 | 56.32 / 57.98 GiB; 0 · 0 | 56.02 / 57.66 GiB; 0 · 0 |
| host RSS after load → end | 18.18 → 19.03 GiB | 18.19 → 19.05 GiB | 18.19 → 19.04 GiB |
| tokens (robot · non-robot) | 11,535,260 (9,794,812 · 1,740,448) | 11,743,308 (10,010,897 · 1,732,411) | 11,807,961 (10,089,802 · 1,718,159) |
| resume | none | none | none |

Ratios are the per-question loss **before the weight** over the constant-prior baseline, averaged over the window (R8 A1's instrument — the same
definition as R12's records, so the two columns are comparable); "first step below" is the end step of the first 20-step window whose Σ loss / Σ
baseline falls below the threshold; probes are the training batch's argmax counts. Both runs of a seed saw the same 233 batches (above). **Where the
mass went**: with three of ten robot questions at zero, every remaining robot head's share of the step loss grew by ≈ 1.3× (mean weighted mass per
step `q_main` 0.086 vs its unweighted 0.065, `q_done` and `q_stop` 0.089 vs 0.067, `q_speed` 0.071 vs 0.050 — the same in all three seeds). Reading,
per question (descriptive — teacher-forced training-batch values, not the closed loop; three seeds are not a sample a recipe interval can be computed
from):

* **`q_main`**: no difference from R12 visible — it fits by step 25–28 (ratio < 0.5) in both recipes and ends at 0.096 · 0.042 · 0.036 (R12 0.102 ·
  0.040 · 0.032).
* **`q_done`**: fits in all three as in R12 (end 0.526 · 0.070 · 0.084, R12 0.527 · 0.021 · 0.021); the < 0.5 crossing is earlier for seed 18 (87 vs
  118) and later for seeds 17 and 19 (86 vs 57, 139 vs 110) — no consistent direction.
* **`q_stop`**: unchanged where it matters — **no run ever answered `true` on a stop-true training tick** (0/277 · 0/244 · 0/146, exactly R12's
  0/277 · 0/244 · 0/146), and its loss stays at or above the constant-prior baseline (end 1.721 · 1.241 · 1.609; R12 1.000 · 2.110 · 1.373). False fires
  on stop-false ticks 67 · 9 · 208 (R12 68 · 39 · 98).
* **`q_speed`** fits **worse** than in R12: its end ratio is 0.943 · 0.778 · 1.085 against 0.474 · 0.292 · 0.165, and it never crosses 0.5 (R12 at
  steps 172 · 140 · 162). With the execution heads' losses at zero, the speed head — the one execution answer arm X still takes from the model — gets
  more loss mass per step (mean weighted mass ≈ 0.071 against its unweighted ≈ 0.050) but learns less; the curves do not say why (one reading: the
  gripper, path and force labels carried phase information the speed answer shares).
* **`q_observe`, `q_retry`** stay above their baselines in both recipes (no direction).
* **The three zeroed heads** are untrained, as designed — their unweighted ratio at the end is `q_gripper` 2.30 · 2.60 · 1.24, `q_path` 0.75 · 0.57 ·
  0.44, `q_force` 6.37 · 2.27 · 3.60 (R12 0.25 · 0.82 · 0.22, 0.08 · 0.07 · 0.03, 0.007 · 0.017 · 0.003); R8's step-150 statistic, recomputed for the
  record only, reads 2.52 · 2.41 · 1.51 ("not fitting", as expected for a weight-0 head). Arm X never hands these answers to the harness. The
  "gripper fit onset" (settled-stratum probe) is not evidence of training here: the settled stratum is answered by copying the executed gripper
  state from the input (97 · 11 · none; batch settled 334/334 · 621/621 · 88/517).
* **The weighted step loss** ends higher than R12's (mean of the last 20 steps 0.287 · 0.296 · 0.336 vs 0.265 · 0.261 · 0.248) — it is a mean over
  seven questions instead of ten and the dropped ones were among the best-fit; not a quality reading.

## 5. Runs, the choice, the registered verdict and the descriptive readings (Stage D)

### Runs, in the registered order (every run file at git `f77c695` — the Stage C commit — with a clean tree, checked before each model run)

| unit (systemd, `Started` → end) | what | wall |
| --- | --- | ---: |
| `r14-d-dev` (GPU chain, `artifacts/scratch/r14/d-dev.sh` → `d-model.sh`) | r14s18X · r14s17X · r14s19X on dev_new2 100, one process each (page-cache drop, fence 0.6, `require_free(36 GiB)`; each checkpoint's run checked `completed` 233/233 first) | 44 m 40 s (14 m 59 s · 14 m 57 s · 14 m 43 s) |
| `r14c-d-select` (CPU) | dev_new2 report (three selection sources + rule, ruleG, ruleX, expert) → `select` | 30 s |
| `r14c-d-new-{ruleX,ruleG,rule,expert}` (CPU, in parallel, GPU idle) | the four CPU policies on ood_new3 200 | 5 m 47 s – 6 m 55 s |
| `r14-d-judged` (GPU chain, no CPU unit running) | r14s17X, then r12s19X (R13's model-only checkpoint), on ood_new3 200 | 51 m 20 s (26 m 42 s · 24 m 38 s) |
| `r14c-d-final` (CPU, MemoryMax 40G) | final report → `verdict` → tables → failure classes and discordant seeds | 16 m 23 s |

### The choice (dev_new2 100, the registered rule applied by code; `artifacts/reports/r14-selection.json`)

| configuration | `done` | **strict** | false done (stale goal) | failures main / aux / geometric | aux-failure streaks |
| --- | ---: | ---: | ---: | --- | --- |
| r14s18X | 82 | 66 | 16 (16) | 4 / 14 / 0 | speed 14, path 5, gripper 2, force 2 |
| **r14s17X** | 85 | **67** | 18 (16) | 3 / 12 / 0 | speed 12, path 4, force 1 |
| r14s19X | 85 | 66 | 19 (17) | 5 / 10 / 0 | speed 10, path 1 |

All three eligible (each run `completed` 233/233), so `require_all` holds and the verdict is issuable (`issuable: true`). **Chosen: r14s17X**,
decided by the first key (67 vs 66). The verdict re-chose r14s17X from the final report's dev_new2 table — identical (the chain asserts it). Beside,
on the same scenes (not candidates): R12's same-seed checkpoints in arm X (R13's stored runs) r12s18X 68, r12s17X 72, **r12s19X 79**; rule 67, ruleG 67,
ruleX 64, expert 90. Every R14 configuration is below R12's best (r12s19X) on the choice set, and all three make far more false dones (16–19, almost
all stale-goal) than R12's same seeds (13 · 5 · 3).

### The registered verdict (ood_new3 200; `artifacts/reports/r14-verdict.json`)

Final report `artifacts/reports/r14-closed-loop.json` (`report --runs <three selection sources + R12's three X on dev_new2 + rule, ruleG, ruleX, expert on
dev_new2 + the six ood_new3 runs> --only dev_new2,ood_new3 --seed-pairs ruleX:r14s17X r14s17X:ruleX r14s17X:r12s19X r12s19X:ruleX r14s18X:r12s18X
r14s17X:r12s17X r14s19X:r12s19X --robustness-metrics forbidden_contacts` → `verdict --registration configs/eval/r14-registration.yaml`).

| question | condition | pair | difference [95 %] | discordant · exact test | RNG seeds 1–200 | holds if | result |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | (a) strict success | `ruleX − r14s17X` | **+0.070 [+0.005, +0.135]** (0.700 vs 0.630; 140 vs 126) | 29 vs 15 · McNemar p = 0.049 | lower ≤ 0 in 9 of 200 (lower 0.000 … +0.010) | lower ≤ 0 | **no** |
| 1 | (s) forbidden-contact onsets / episode | `r14s17X − ruleX` | +0.015 [−0.010, +0.050] (0.030 vs 0.015; 6 vs 3) | seeds higher 2 vs 1 · sign test p = 1.0 | lower ≤ 0 in 200 of 200 | lower ≤ 0 | yes |
| 2 | strict success | `r14s17X − r12s19X` | **−0.120 [−0.185, −0.055]** (0.630 vs 0.750; 126 vs 150) | 11 vs 35 · McNemar p = 0.0005 | upper ≥ 0 in 0 of 200 | better if lower > 0, worse if upper < 0 | **worse** |

**Registered calls.** Question 1: **fail — "불통과 — 의미 head에 학습을 모은 구성은 검증으로 고른 뒤 처음 보는 홀드아웃에서 규칙 판정기에 확실히 뒤지거나
금지 접촉이 확실히 많다"** (`holds: false`, `failed: [a]`; `issued: true`). Question 2: **worse — "나쁘다 — 의미 head 집중 학습으로 고른 구성이 R13의 구성보다
확실히 뒤진다"**. Read with the zero rule: (a) fails **at its boundary** — the lower bound is +0.005 under the registered RNG seed and 0.000 under 9 of 200
alternative seeds (exact McNemar p = 0.049), so the rule judge is ahead by an interval that only just excludes zero; (s) holds (no difference found in
forbidden contacts: 6 onsets in 4 episodes vs 3 in 3). Question 2 is not near a boundary: the chosen R14 configuration trails R13's passing configuration
by 24 successes on the same 200 scenes (35 vs 11 discordant), and the upper bound stays below zero under every one of the 200 RNG seeds. dev_new2 (the
choice set, beside only): (a) −0.030 [−0.110, +0.040], (s) +0.020 [0.000, +0.050], question 2 −0.120 [−0.220, −0.030] (7 vs 19, p = 0.029).

### Descriptive readings (not used to judge)

| policy (ood_new3 200) | `done` | **strict** [95 %] | false done (stale goal) | failures main / aux / geometric | aux-failure streaks | `q_stop` caught / onsets | forbidden-contact onsets (episodes) | unsafe ticks / acted | `stall_exhausted` |
| --- | ---: | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: |
| **r14s17X** (chosen) | 162 | **126** [0.560, 0.695] | 36 (29) | 17 / 19 / 2 | speed 19, path 3 | 0/53 | 6 (4) | 0.0300 | 38 |
| r12s19X (R13's passing configuration) | 158 | 150 [0.685, 0.805] | 8 (5) | 24 / 15 / 3 | speed 13, path 5, gripper 2, force 2 | 0/67 | 3 (3) | 0.0062 | 42 |
| ruleX (same arm) | 140 | **140** [0.635, 0.760] | 0 | 2 / 11 / 47 | path 10, force 1 | 50/53 | 3 (3) | 0.0001 | 60 |
| ruleG | 147 | 147 [0.670, 0.795] | 0 | 2 / 5 / 46 | path 4, force 1 | 51/54 | 3 (3) | 0.0001 | 48 |
| rule (native) | 146 | 146 [0.665, 0.790] | 0 | 1 / 52 / 1 | gripper 52, path 4 | 51/54 | 3 (3) | 0.0001 | 49 |
| expert | 187 | 187 [0.900, 0.965] | 0 | 2 / 1 / 10 | force 1 | 65/65 | 3 (3) | 0.0000 | 13 |

(`artifacts/scratch/r14/d-tables.md`, `describe.json`, `d/probe.json`; failure classes by `failure_cause`, the auxiliary streak being the adopted
(executed) answer against the expert's label for ≥ 3 ticks.)

* **R13's pass on another holdout** (`r12s19X − ruleX`, the same registered checks, descriptive): strict +0.050 [−0.005, +0.105] (150 vs 140; 20 vs 10,
  p = 0.099) — no difference found, the point estimate in the model's favour; forbidden contacts +0.000 [0.000, 0.000] (3 vs 3). Under R13's registered
  rule this would again read as a pass. r12s19X's false dones +0.040 [+0.015, +0.070] (8 vs 0) and its `q_stop` 0/67 are R13's picture again.
* **The chosen configuration against ruleX** (`r14s17X − ruleX`, seed-paired): false dones **+0.180 [+0.130, +0.240]** (36 vs 0), stale-goal **+0.145
  [+0.100, +0.195]** (29 vs 0), main-decision failures +0.075 [+0.035, +0.120] (17 vs 2), auxiliary failures +0.040 [−0.005, +0.090] (no difference
  found), wrong-action ticks +0.055 [+0.035, +0.077], unsafe-action ticks **+0.030 [+0.012, +0.057]**, `q_stop` caught −0.943 [−1.000, −0.873] (0/53 vs
  50/53); `rule − r14s17X` +0.100 [+0.035, +0.170], `ruleG − r14s17X` +0.105 [+0.040, +0.175], `expert − r14s17X` +0.305 [+0.240, +0.375].
* **The chosen configuration against R13's** (`r14s17X − r12s19X`): false dones **+0.140 [+0.085, +0.200]** (36 vs 8), stale-goal **+0.120 [+0.065,
  +0.175]** (29 vs 5), unsafe-action ticks +0.024 [+0.007, +0.049]; main-decision failures −0.035 [−0.085, +0.015] (17 vs 24), auxiliary +0.020 [−0.030,
  +0.065] (19 vs 15), forbidden contacts +0.015 [−0.010, +0.050] (6 vs 3), `q_stop` 0/53 vs 0/67 — no difference found for those four.
* **The 46 seeds behind question 2**: the 35 only r12s19X solved are, for r14s17X, stale-goal false dones 17, other false dones 6, auxiliary stalls with
  its own speed answer 7 and main-decision stalls 5 (E1 24, E2 11); the 11 only r14s17X solved are, for r12s19X, main-decision stalls 6, own-speed stalls 2,
  stale-goal false dones 2 and one band stall (E1 2, E2 9). **The 44 seeds behind (a)**: the 29 only ruleX solved are r14s17X's own-speed stalls 8,
  stale-goal false dones 8, main-decision stalls 8 and other false dones 5 (E1 20, E2 9); the 15 only r14s17X solved are ruleX's geometric stalls 13, one
  switch-tick churn and one band stall (E0 1, E1 2, E2 12).
* **Remaining failures in R13 fix round 1's classes** (switch-tick churn re-counted as a main decision): r14s17X's 74 non-successes are false dones 36 (29
  stale-goal), main-decision stalls 17, auxiliary stalls 19 — **all 19 its own speed answer** (no switch-tick churn, so the recount leaves main 17) — and
  geometric 2; r12s19X's 50 are false dones 8 (5 stale-goal), main 24 (+1 churn → 25), auxiliary 15 (own speed 9, band 2, churn 1, other 3 → 14) and
  geometric 3; ruleX's 60 are geometric 47, auxiliary 11 (band 8, other 2, churn 1 → 10) and main 2 (→ 3).
* **What the stale-goal false dones are** (`artifacts/scratch/r14/d/probe.py` → `probe.json`, CPU unit `r14c-d-probe`; R6 A1's `q_done` strata): on the
  ticks right after the hand released an object that is not the current target (`post_release_other`, ≤ 6 ticks after the release), the chosen model's
  raw `q_done` is ≥ 0.5 on **129 of 228** ticks (r12s19X 29 of 337, ruleX 0 of 48); on dev_new2 the three R14 configurations do it on 65/146 · 72/136 · 74/101
  ticks and R12's same seeds on 48/155 · 20/172 · 9/178. In the 109 ood_new3 episodes whose target changed, r14s17X held the old target at the change in 81
  (r12s19X 96 of 102), **never committed to the new target in 34** (r12s19X 15) and 22 of those 34 ended in a stale-goal false done (r12s19X 1). So the R14
  model finishes the old instruction's placement and then declares done.
* **What the own-speed stalls are**: on the streak ticks where the model's own speed answer was executed against the label, r14s17X answered speed 1 where
  the label said 2 on 810 ticks (1 where the label said 0 on 135); R12's configurations' own-answer ticks are the other way round (r12s19X 1 where the label
  said 0 on 85). The R14 model moves too slowly and the stall watchdog ends the episode — consistent with §4: `q_speed` fit worse in all three R14 runs.
* **First grasp not the target**: on ood_new3 the chosen model's first grasp commitment was on an object that was not the instruction's target (and never
  had been) in **28 of 200** episodes (E1 6, E2 22; 13 still succeeded, 11 ended in main-decision stalls, 4 in false dones); r12s19X and every rule policy
  0; on dev_new2 1 of 100 (the R12 configurations 0).
* **Each seed's R14 − R12, same seed, arm X (dev_new2, beside)**: strict seed 18 −0.020 [−0.110, +0.070], seed 17 −0.050 [−0.140, +0.040] (no difference
  found), **seed 19 −0.130 [−0.220, −0.040]**; false dones seed 18 +0.030 [−0.040, +0.100] (no difference found), **seed 17 +0.130 [+0.050, +0.210]**, **seed
  19 +0.160 [+0.090, +0.230]**; stale-goal false dones +0.050 (nd) · +0.110 · +0.150. On dev_new2 the R14 configurations' unsafe-action ticks are lower than
  R12's (0.006–0.008 vs 0.046–0.049; no difference found per seed), on ood_new3 the chosen's are higher (above).
* **Delegation activity** on ood_new3: the three rules changed the chosen model's argmax on `q_gripper` 13,893, `q_path` 1,624 and `q_force` 16,979 of 22,408
  delegated ticks (r12s19X 1,075 / 1,012 / 565 of 20,393) — the zeroed heads answer far from the rules, and arm X never hands those answers to the harness.

**Latency** (`latency.model_ms` excludes each episode's first tick; no CPU unit ran during either judged GPU run): r14s17X on ood_new3 non-first ticks p50 /
p95 / p99 **42.9 / 55.1 / 77.6 ms**, 0 of 22,208 over 100 ms, observation → command p95 60.8 ms; first ticks p50 91.8, max 267.3 ms, 14 of 200 over 100 ms.
r12s19X on ood_new3 43.4 / 55.8 / 77.6 ms, 0 of 20,193 (first ticks 11 of 200). The three dev_new2 runs: p95 56.2–56.4 ms, 0 of 35,604 non-first ticks over
100 ms (first ticks 16–17 of 100 each). The gate (p95 ≤ 80 ms, > 100 ms ≤ 5 %) passes. GPU peak 4.15 GiB per evaluation process.

**Records**: nine new directories in `artifacts/datasets/r14-closed-loop/` (1.56 GB) — `r14s18X`, `r14s17X`, `r14s19X` on dev_new2; `r14s17X`, `r12s19X`,
`ruleX`, `ruleG`, `rule`, `expert` on ood_new3 — one version set (harness `h0.9`, controller `c0.6`, expert `e0.4`, rule judge `rj0.5`, extractor
`pw0.2`, sim and record serializer `s0.3`, generator `gen-robot-v0.3`, generator-config digest `c297ada2e90a`); every model run: contract `93fe26725a4c`,
text serializer `ts0.6`, arm X wrapper `dx0.2/X/mp0.1` (execution rules `xg0.1` · `xp0.1` · `xf0.1`), fence 0.6, all nine run files at git `f77c695`. The
four checkpoints read (r14s18 · r14s17 · r14s19 full, r12s19 model-only) are unchanged in size and mtime before their first load and after their last
(`artifacts/scratch/r14/checkpoints-stat-{before,after}.txt`).

## 6. Measured cost, disk and this round's own checkpoints (Stage E)

| unit (systemd, `Started` → end) | what | wall |
| --- | --- | ---: |
| `r14-pytest-a` | GPU: full suite after Stage A (fence 0.6 + `require_free(36 GiB)` first; nothing else on the GPU) | 0 h 11 m 26 s |
| `r14-t1-s18` | GPU: training, seed 18 (233 steps) | 3 h 49 m 55 s |
| `r14-t1-s17` | GPU: training, seed 17 (launched by `r14c-launcher` after seed 18 read `completed` 233/233) | 3 h 53 m 10 s |
| `r14-t1-s19` | GPU: training, seed 19 (same launcher check) | 3 h 53 m 01 s |
| `r14-d-dev` | GPU: the three in arm X on dev_new2 (100 each) | 0 h 44 m 40 s |
| `r14-d-judged` | GPU: r14s17X and r12s19X on ood_new3 (200 each) | 0 h 51 m 20 s |
| `r14-pytest-b` | GPU: full suite at the end (same checks), on the Stage D commit `99eda6b` after the slimming | 0 h 11 m 23 s |
| **GPU total** | | **48,895 s = 13.58 GPU-h** (brief ≈ 14 h: training 11.60 · evaluation 1.60 · tests 0.38) |

CPU units (`r14c-*`, CUDA hidden, MemoryMax, `choom -n 1000`, Nice 10): the A2 golden values and mutation checks (2 × ≈ 6 m), the R11–R13
verdict regression (1 s), the A1 dry run and slimming (37 s; 5 m 36 s), the ood_new3 seed walk (26 s) and overlap (1 s), the training-facts
tables (`c-facts-*`, `c-step1-*`, seconds each), the launcher (waiting only — it read each run's `metrics.json` status), the choice (30 s), the
four CPU policies on ood_new3 (5 m 47 s – 6 m 55 s, in parallel, GPU idle), the final report and verdict (16 m 23 s, MemoryMax 40G — it did not
overlap any GPU unit), the mechanism probe (25 s), the version-set check, the Stage E dry run and slimming (46 s; 2 m 17 s, MemoryMax 40G,
GPU idle) and the development test units (`r14c-pt-*`). Disk (`/`, 10⁹ B free): 48.81 before A1 → 206.89 after A1 → 207 · 181 · 154 at the
three training launches → 128 after training → 125.96 before Stage E → **193.71** after it (closed-loop records 1.56 GB; checkpoints three ×
26.35 GB → three × 3.77 GB). `journalctl -k` since 2026-10-09 22:00: 49 kernel lines, all apparmor audit records — no `NVRM`, OOM or
killed-process line. Cumulative GPU on the DGX Spark ≈ 97.8 h. Cloud spend 0.

**This round's own checkpoints, slimmed under the standing rule (HANDOFF §5) after every evaluation run** (`artifacts/reports/r14-e-slim.json`;
`scripts/slim_checkpoint.py run --files artifacts/runs/r14-t1-fp32-2b-s{18,17,19}/checkpoint.pt`, tool `slim-checkpoint-1.1`, CPU unit
`r14c-e-slim` 12:58:48 → 13:01:05, after a dry run on seed 18 `artifacts/scratch/r14/e/dry-r14s18.json`; git `99eda6b` with a clean tree —
`git_dirty_files` empty; the approval group is the standing rule's, dated 2026-10-09, and the tool confirmed each run's `metrics.json` read
`completed` at 233/233 before touching it; the last evaluation run had ended at 12:27:27 and the final report at 12:44:54):

| file | run · step · status | bytes before → after | sha256 before → after | checks | s |
| --- | --- | --- | --- | --- | ---: |
| `artifacts/runs/r14-t1-fp32-2b-s18/checkpoint.pt` | r14-t1-fp32-2b-s18 · 233 · completed | 26,350,744,917 → 3,766,342,645 | `1de814b61148…` → `efe8b4e200b8…` | model 323 tensors / 3,764,698,756 B bit-equal · 10 other keys equal · `load_serving_judge` (CPU) loads with `93fe26725a4c…`, parameters bit-equal | 45.8 |
| `artifacts/runs/r14-t1-fp32-2b-s17/checkpoint.pt` | r14-t1-fp32-2b-s17 · 233 · completed | 26,350,745,685 → 3,766,343,349 | `4c54784cbe4b…` → `c5edefc67747…` | same, all pass | 45.4 |
| `artifacts/runs/r14-t1-fp32-2b-s19/checkpoint.pt` | r14-t1-fp32-2b-s19 · 233 · completed | 26,350,745,941 → 3,766,343,605 | `e3328899200a…` → `c7edcb9f5742…` | same, all pass | 44.9 |

Replaced 3 of 3, failed 0; freed **67,753,206,944 B** (22,584,402,272 + 22,584,402,336 + 22,584,402,336); `/` free 125,964,906,496 →
193,709,273,088 B. These three runs can now be evaluated and served but not resumed (the resume path refuses model-only files). Sizes and
mtimes before and after: `artifacts/scratch/r14/e/{before,after}.txt`; full hashes in the JSON.

**Every CPU job that read records ran as a unit (HANDOFF §5).** Outside units the session shell read only tracked files, scripts, logs, report
JSONs (run summary files included), scratch tables derived from records inside units, and dataset-manifest metadata — never an episode record
(the two tracked test fixtures aside); the list with times, including the reads while a GPU unit was up (training-log tails and the
training-facts tables during the three training units, log lines during `r14-d-dev` and `r14-d-judged`, log tails, tracked files and the
slimming report JSON during `r14-pytest-b`), is
`artifacts/scratch/r14/session-shell-reads.md`.

## 7. What the round says, and what it does not

1. **Registered call, question 1: fail — "불통과 — 의미 head에 학습을 모은 구성은 검증으로 고른 뒤 처음 보는 홀드아웃에서 규칙 판정기에 확실히 뒤지거나
   금지 접촉이 확실히 많다".** The configuration chosen on validation (r14s17X, 67 of 100 on dev_new2 — the best of the three R14 checkpoints
   in arm X, and below R12's same seeds 68 · 72 · 79) against the rule judge in arm X on 200 scenes no policy had run: strict `ruleX − r14s17X`
   **+0.070 [+0.005, +0.135]** (140 vs 126; 29 vs 15 discordant, p = 0.049) — the rule judge is ahead by an interval that only just excludes
   zero (the lower bound is 0.000 under 9 of 200 alternative RNG seeds), so (a) fails at its boundary; that is the registered call. Forbidden-
   contact onsets: no difference found (+0.015 [−0.010, +0.050], 6 vs 3).
2. **Registered call, question 2: worse — "나쁘다 — 의미 head 집중 학습으로 고른 구성이 R13의 구성보다 확실히 뒤진다".** `r14s17X − r12s19X`
   **−0.120 [−0.185, −0.055]** (126 vs 150; 11 vs 35 discordant, p = 0.0005; the upper bound stays below zero under all 200 RNG seeds). Zeroing
   the three execution heads' losses and retraining R12's recipe gave a configuration clearly worse on the new holdout than R13's passing
   configuration, whose heads were trained. Seed by seed on dev_new2 (beside; the same draws and recipe, only the loss weights differ — the
   spread of repeated runs at one configuration was not measured): strict −0.020 [−0.110, +0.070] · −0.050 [−0.140, +0.040] (no difference
   found) · −0.130 [−0.220, −0.040] (seeds 18 · 17 · 19); no seed shows a gain.
3. **Where it lost: false dones and a slow speed answer.** Seed-paired against r12s19X: false dones **+0.140 [+0.085, +0.200]** (36 vs 8),
   stale-goal **+0.120 [+0.065, +0.175]** (29 vs 5), unsafe-action ticks +0.024 [+0.007, +0.049]; main-decision and auxiliary failures,
   forbidden contacts and `q_stop` — no difference found. Of the 35 seeds only r12s19X solved, 23 are r14s17X's false dones (17 stale-goal) and
   7 its own-speed stalls. The probe places the false dones right after a release: on the ticks ≤ 6 after the hand released an object that is
   not the current target, the chosen model's raw `q_done` is ≥ 0.5 on 129 of 228 (r12s19X 29 of 337, ruleX 0 of 48), and on dev_new2 each
   R14 seed is above its R12 counterpart (65/146 · 72/136 · 74/101 vs 48/155 · 20/172 · 9/178). After an instruction change it more often never
   commits to the new target (34 of 109 episodes vs 15 of 102) and then finishes the old placement and declares done (22 of those 34). All 19
   of its auxiliary failures are its own speed answer (no switch-tick churn, so R13 fix round 1's recount changes nothing), mostly too slow
   (speed 1 where the label said 2 on 810 streak ticks) — the direction of the training-batch `q_speed` fit, worse in all three runs (§4). Its
   first grasp went to an object that was not the instruction's target in 28 of 200 episodes (0 for every other policy).
4. **What the weights did not change.** On the training batch `q_main` and `q_done` fit as in R12; `q_stop` still never fired on a stop-true
   tick (0/277 · 0/244 · 0/146), and in the loop it catches 0 of 53 — moving loss mass to the remaining heads (≈ 1.3× each) did not teach the
   stop or the speed. The zeroed heads stayed untrained by design; in arm X their answers never reach the harness (the rules changed the
   chosen model's argmax on 13,893 `q_gripper`, 1,624 `q_path` and 16,979 `q_force` of 22,408 ticks).
5. **R13's pass recurred on this holdout (descriptive).** `r12s19X − ruleX` strict +0.050 [−0.005, +0.105] (150 vs 140; 20 vs 10, p = 0.099),
   forbidden contacts +0.000 (3 vs 3) — under R13's registered rule a pass again; its false dones (+0.040 [+0.015, +0.070], 8 vs 0) and `q_stop`
   0 of 67 are R13's picture. What failed here is the training change, not arm X.
6. **What the round does not say.** One chosen configuration (training seed 17), one holdout of 200, one recipe (R12 + three zero weights,
   233 steps); the rule judge reads the structured goal while the model reads the instruction text. It does not say that removing these losses
   is always worse — only that in this recipe and these three seeds no gain was seen and the chosen configuration was clearly worse than R13's.
   Why removing them raises `q_done` right after a release was not tested (that the gripper and path labels carried "what was just released,
   what is being done now" into the shared representation is one explanation, untested). dev_new2's numbers are the choice set's and are written
   beside only. (a) failed at its boundary; question 2 is far from its boundary.

**Next step (registered: written from the numbers only; on a fail, from the remaining failures' classes and question 2's call).** Question 2
says the zero-weight recipe is worse than keeping the heads trained, so the base for the next round stays R13's configuration (R12's checkpoints
served in arm X) with the three execution questions still trained — whether to drop them from the question set (docs/08 §12.1) remains the
user's decision, and this round gives no ground for dropping them. The chosen configuration's remaining failures by class: false dones 36 (29
stale-goal — `q_done` right after releasing a non-target object), its own speed answer 19 (too slow), main-decision stalls 17, geometric 2;
`q_stop` 0 of 53. R13's configuration on the same scenes: main-decision stalls 25 (one of them switch-tick churn), auxiliary 14 (own speed 9),
false dones 8 (5 stale-goal), geometric 3, `q_stop` 0 of 67. So the next round's targets, from these classes, are the post-release `q_done`
(stale-goal false dones), `q_stop` and `q_speed`, aimed at directly — how (data at those ticks, labels, or another objective) is the next
brief's design; re-weighting losses was not the means. The cloud question and the contract change stay the user's. ood_new3 (1350100~) has now
been analysed seed by seed, so any further verdict needs another unused seed walk.
