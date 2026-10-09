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
