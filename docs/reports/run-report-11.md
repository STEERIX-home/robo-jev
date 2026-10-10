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
