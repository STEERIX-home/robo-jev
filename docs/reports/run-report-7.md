# Run report 7 — delegation: the gripper's timing (and the stop) handed to the execution layer, measured on existing checkpoints without training (2026-10-02)

The document docs/06 Task 6 defines (:384), this time for an **evaluation-only** round (Task R10): no training, no new checkpoint, no
new dataset. Every number is copied from an artifact named next to it. Korean prose lives in `docs/06`, `docs/08` and `HANDOFF.md`;
this report is the English ledger. Run reports 1–6 stay as they are; this one only adds. **Sections are appended as each stage's numbers
land; §0 was committed before any R10 closed-loop number existed.**

## 0. Pre-registration (written 2026-10-02 before any R10 closed-loop number; machine-readable copy `configs/eval/r10-registration.yaml`)

The rule below is applied by code, not by hand: `scripts/closed_loop.py verdict --report artifacts/reports/r10-closed-loop.json
--registration configs/eval/r10-registration.yaml --out artifacts/reports/r10-verdict.json` runs `robo_jev.closed_loop.apply_registration`
on the closed-loop report. The registration names its decision `delegation` (not `cloud`), so the verdict is written as `decision` and
`holds` next to the registered sentence; the registered predictions and descriptive readings sit beside it and never change it (tests in
`tests/test_closed_loop.py` pin both behaviours and this file's content).

**Why this round.** The closed-loop failures of R4–R9 gather into two strands. (1) **Unstable execution-level heads**: the gripper head
sits at its constant-prior level for ≈ 100 steps in every seed and fits late (onsets 132 / 139 / 152), the seed-18 run taken to the end
executed 68 of 701 reference gripper transitions, and `q_stop` caught no stop event in any of three seeds. (2) **Seed spread of the
semantic judgment**: one recipe gives holdout strict 90 / 27 / 28 of 200. Yet the expert's gripper decision is a **geometric rule**
(`Expert._gripper`: per-phase default state, close only within `grasp_ready_mm` of the committed target's grasp point, open at place only
on a descending path, closed fist for pushes), its stop is mostly force and contact rules, and the controller already has close readiness
and a force reflex; the rule judge's main failures were its own gripper threshold too. **Hypothesis**: the gripper's timing (and the force
stop) belongs to the execution layer, not the semantic layer; handing it to an execution-side rule removes a large part of the model's
failures, and what remains is the model's actual semantic judgment. This round measures that hypothesis **without training**, on the
existing checkpoints.

**Mechanism — a policy wrapper, not a harness change.** The harness version is part of the contract digest (`93fe26725a4c…`), so changing
the harness would make every existing checkpoint unloadable. The delegation therefore lives in the policy slot
(`robo_jev.delegation.DelegatedExecutionPolicy`, version `dx0.1`; Stage A commit `5a00ac3`):

* **Arm G** — the wrapped policy's answers are passed to the harness unchanged except `q_gripper`, which is replaced by the **execution
  gripper rule** (`ExecutionGripperRule`, `xg0.1`): the expert's e0.4 `_gripper` computed from the current commitment and the observed
  state only (the request's commitment projection, candidate keys and path flags, `state.robot` and the target's observed pose and top) —
  no goal, no instruction text, no simulator observation, no expert object. It equals the expert's recorded gripper decision on every tick
  of every stored R4–R9 closed-loop record (49 directories, 3,852 episodes, **419,245 / 419,245 ticks**, all twelve reasons;
  `artifacts/reports/r10-a1-rule-agreement.json`) and a live `Expert._gripper` on hand-made requests covering every branch.
* **Arm GS** — G plus `q_stop` always false (0.0): stopping is the executor's force reflex only (the harness sends any tick carrying a
  reflex event to its stop branch regardless of the answer).
* The wrapped model never sees the commitment, the observation, the rule's answer or the expert's answers: it receives exactly the
  generator's arguments, as on the unwrapped path (`ModelPolicy` ignores commitment and observation). Records keep what the harness
  received in `model_output`; the wrapped policy's raw answers and the rule's decision per tick are in `usage.delegation`, the episode
  summary in `evidence.delegation`, the arm and inner policy in `provenance.policy`.
* The unwrapped path is byte-identical: a CPU test re-runs R9 seed 17's dev_new2 episode `ep-E1-990376-r9-r9s17` through the closed-loop
  runner with its own recorded answers and writes the stored file byte for byte (only `provenance.timing.wall_s` differs); the expert
  wrapped in G reproduces R6's stored expert episode `ep-E1-990297-r6-expert` byte for byte apart from the wrapper's own fields.

**Policies.** Checkpoints are used as they are (the model-only slimmed files and the full ones; the evaluation path accepts both):
**r5** (`artifacts/runs/r5-t1-fp32-2b-s18/checkpoint.pt` — the best run fixed before R10: ood_dev 200 strict 132), **r6**
(`artifacts/runs/r6-t1-fp32-2b-s18`), **r9s17** (`artifacts/runs/r9-t1-fp32-2b-s17` — R9's chosen seed; second evidence), **r9s18c**
(`artifacts/runs/r9-t1-fp32-2b-s18c/r8-t1-fp32-2b-s18` — the seed whose gripper never closed: 633 of 701 reference transitions missing),
**r8s19** (`artifacts/runs/r8-t1-fp32-2b-s19` — target selection failed: first grasp commitment on a non-instructed object in 139 of 200),
and the CPU policies **rule** (rj0.5), **expert** (e0.4) and **mechanical**. Labels are the policy's label plus the arm (`r5G`, `ruleG`,
`r9s17GS`, …), episode ids end in `-r10-<label>`, records go to `artifacts/datasets/r10-closed-loop/<label>/<condition>/`. **Native values
are not re-run**: they are the stored records of R4/R6/R7 (expert, rule, mechanical, r5, r6), R8 (r8s19) and R9 (r9s17, r9s18c) on the
same seeds. The expert wrapped in G must produce the same episodes as the unwrapped expert (its rule is the expert's own); any difference is
reported as a wrapper defect before any other number is read.

**Scenes.** Judged set **`ood_dev200`** = R4's 26 ood_dev seeds + R6's 74 + R7's 100 (`--merge ood_dev200=ood_dev,ood_dev_new,ood_dev_new2`;
the same 200 as R7–R9; 0 origin groups shared with training). Second evidence **`dev_new2`** 100 (R6's new dev seeds) for r5G, r9s17G,
ruleG and expertG only — written beside, never judged. The sealed split is not opened (the closed-loop configs refuse `ood_test`).

**Runs and order (the brief's priority).** G: r5 and rule (the primary comparison) → r9s17 → r9s18c, r8s19, r6 on ood_dev 200; GS: r5,
r9s17 (GPU) and rule (CPU) on ood_dev 200; dev_new2: r5G, r9s17G (GPU), ruleG, expertG (CPU); then expertG and mechanicalG on ood_dev 200.
GPU jobs run one at a time in one unit (fence 0.6, `require_free`, `choom`); CPU policies run as `systemd-run` units with MemoryMax and
`choom`. Anything not run is written as missing; without both runs of the primary comparison there is no verdict.

**Verdict (holdout, `ood_dev200`) — one condition, the brief's:**
* **(a)** `ruleG − r5G` strict success (`done ∧ target_inside_zone`), seed-paired: interval **lower bound ≤ 0**.

**Holds → "with execution delegated, the model (r5) does not clearly trail the rule judge — recommendation: review a contract change (model =
semantic judgment; gripper timing = execution layer); the decision is the user's". Fails → "with execution delegated, the rule judge is still
ahead — the bottleneck is semantic judgment".** Next to (a): discordant seeds, exact McNemar p and the bound distribution under bootstrap RNG
seeds 1–200 (they do not change the verdict). dev_new2's value of (a) is written beside only. **Why r5 is the primary model**: it is the
best run fixed before R10 (ood_dev 200 strict 132); r9s17 is the second evidence (`ruleG − r9s17G`, beside).

**Registered predictions (beside the verdict; they do not change it).** The brief predicts that r9s18c (the seed that never closed the
gripper) rises a lot and that r8s19 (target selection failure) rises less. "A lot" is measured relative to r8s19, with no absolute threshold:
* **P1** `r9s18cG − r9s18c` strict: lower bound > 0 (it rises).
* **P2** `(r9s18cG − r9s18c) − (r8s19G − r8s19)` strict, seed-paired over the 200 seeds the four runs share
  (`robo_jev.closed_loop.paired_effect_difference`): lower bound > 0 (r8s19 rises less than r9s18c).

**Descriptive readings (not used to judge).** The delegation effect per policy, seed-paired with native: `policyG − policy` strict for r5,
r6, r9s17, r9s18c, r8s19, rule, expert (identity check) and mechanical; false done for the five models and the rule judge; gripper
transitions missing (share of reference transitions) and duplicate (per episode) per seed. GS − G (the value of dropping the semantic stop)
for r5, r9s17 and rule: strict, unsafe ticks (an acted tick whose stop label is true counts as a violation, so GS raises it **by
construction**), forbidden-object contact onsets per episode, reflex-event ticks per episode, `q_stop` catch (0 by construction). Second
evidence `ruleG − r9s17G` strict. The tables per policy and arm (strict/done, false dones, gripper transitions executed/missing/duplicate,
`q_stop` and reflex, unsafe, failure classes main / aux / geometric, first-grasp target selection) and what failures remain are written in §4.
**Reading rule:** under G, gripper agreement with the reference is close to perfect **by construction** — the rule and the reference labels are
the same expert rule — so it confirms that the delegation works, not that the model is able; what remains under G (main-decision, auxiliary
path/speed/force and geometric failures, false dones, target selection) is the model's semantic judgment.

**Documents (registered).** If (a) holds, docs/08 gets a section "Proposal: q_gripper to the execution layer" written **as a proposal only**
(not applied — what changes and what stays, the harness version and digest impact, what happens to existing checkpoints); it covers
`q_gripper` only — GS's values are written beside as evidence for a later decision on `q_stop`, and R10 makes no `q_stop` proposal. If (a)
does not hold, no proposal section is written and this report says the results do not support it.

**Pairing and interval.** By seed (`profile:seed`), 95 % percentile bootstrap over seeds, 2,000 resamples, RNG seed 20260921
(`EPISODE_BOOTSTRAP`); `paired_success(strict=True)`, `paired_false_done`, `paired_seed_ratio`, `paired_effect_difference`. Zero inside = not
a finding. Strict success is primary. Latency reports each episode's first tick separately.

## 1. Weights (no training)

R10 trains nothing and writes no checkpoint. The five checkpoints are read as they are by the serving loader
(`robo_jev.harness.model_policy.load_serving_judge`: contract digest `93fe26725a4c…` checked against this checkout before `compile`;
`fused` + dense compile, fp32 readout — the same path as run reports 1–6). Size and mtime recorded during the GPU chain
(`artifacts/scratch/r10/checkpoints-stat-before.txt`, written at 10:49:41 — after the first two model loads, not before the first one)
and after the last load (`…-after.txt`, 14:17:27) are identical, and every mtime predates R10 (2026-09-29/30), so no R10 process wrote
a checkpoint:

| label | checkpoint | kind | size (B) | mtime |
| --- | --- | --- | ---: | --- |
| r5 | `artifacts/runs/r5-t1-fp32-2b-s18/checkpoint.pt` | model-only (slimmed in R8 A2) | 3,765,327,925 | 2026-09-29 17:09:25 |
| r6 | `artifacts/runs/r6-t1-fp32-2b-s18/checkpoint.pt` | model-only (slimmed in R8 A2) | 3,765,449,845 | 2026-09-29 17:10:12 |
| r9s17 | `artifacts/runs/r9-t1-fp32-2b-s17/checkpoint.pt` | model + optimizer | 26,350,501,141 | 2026-09-30 21:36:57 |
| r9s18c | `artifacts/runs/r9-t1-fp32-2b-s18c/r8-t1-fp32-2b-s18/checkpoint.pt` | model + optimizer | 26,350,502,421 | 2026-09-30 16:36:25 |
| r8s19 | `artifacts/runs/r8-t1-fp32-2b-s19/checkpoint.pt` | model + optimizer | 26,350,500,437 | 2026-09-30 00:02:24 |

## 2. Reproducible data

No dataset was built. R10 writes only closed-loop records (`artifacts/datasets/r10-closed-loop/<label>/<condition>/`, id tag `r10`), which no
training reads. Scenes are R7's: `ood_dev200` = `artifacts/reports/r4-seeds.json` `ood_dev` 26 + `r6-seeds.json` `ood_dev_new` 74 +
`r7-seeds.json` `ood_dev_new2` 100; `dev_new2` = `r6-seeds.json` 100; one process per model covers the three ood_dev seed files
(`scripts/closed_loop.py run --seeds …` with three files, conditions merged). The generator config is `configs/data/r1_robot.yaml` in all three
closed-loop configs. The native rows are the stored records of R4 (`r4-closed-loop/{expert,rule,mechanical}/ood_dev`), R5 (`r5/ood_dev`), R6
(`r6-closed-loop/*`), R7 (`r7-closed-loop/*/ood_dev_new2`), R8 (`r8s19`) and R9 (`r9s17`, `r9s18c`) — read through the manifest-first reader; no
sealed file was opened. **Caveat carried from R5:** R4's 26 ood_dev records of expert, rule and mechanical carry gripper label rule **v1**; every
R5-onwards record (and every R10 record) carries v2. Labels are computed after the fact from the expert's reference and do not change behaviour
(the expert wrapped in G reproduces those 26 trajectories exactly, §4), but the label-derived columns (reference gripper transitions,
auxiliary-disagreement streaks and the failure class built on them) use different label rules on those 26 seeds of those three natives.
Strict success, `done` and false dones do not depend on labels.

## 3. Resume

Not applicable — no training. The unwrapped evaluation path is the one thing that had to stay byte-identical; a CPU test re-runs R9 seed 17's
stored dev_new2 episode `ep-E1-990376-r9-r9s17` through `run_condition` with its own recorded answers and writes the same bytes
(`tests/test_delegation.py::test_the_unwrapped_path_reproduces_a_stored_r9_record_byte_for_byte`; only `provenance.timing.wall_s` differs).

## 4. Closed loop, the registered verdict and the delegation effect (Stage C)

Same serving path, harness (`h0.9`), controller (`c0.6`) and expert (`e0.4`) as run reports 1–6. Report
`artifacts/reports/r10-closed-loop.json` (`closed_loop.py report --merge ood_dev200=ood_dev,ood_dev_new,ood_dev_new2 --only ood_dev200,dev_new2
--seed-pairs <the registration's eleven> --effects r9s18cG:r9s18c:r8s19G:r8s19`; native rows from the stored run files of R4–R9, the R10 rows
from the registered `artifacts/reports/r10-run-*.json` — the two exploratory dev_new2 run files are not inputs), verdict
`artifacts/reports/r10-verdict.json` (`closed_loop.py verdict --registration configs/eval/r10-registration.yaml`; both rebuilt from the
registered runs only in fix round 1, §5), the exploratory dev_new2 runs in their own labelled report `artifacts/reports/r10-exploratory.json`
(their native rows and the rule judge beside; feeds nothing registered), tables printed by `artifacts/scratch/r10/tables.py`, failure and target-selection counts by
`artifacts/scratch/r10/failures.py`, per-seed flips by `flips.py`, the expert identity check by `identity_check.py`, the false-done mechanism by
R9's `artifacts/scratch/r9/fix1/false_done_mechanism.py` — all reading records through the manifest-first reader.

**The wrapper did what it was registered to do.** The expert wrapped in G (`expertG`) produced **the same episodes as the unwrapped expert on all
300 scenes** (`artifacts/scratch/r10/identity-check.json`): on the 274 scenes whose stored expert records carry label rule v2 (R6/R7) the records
are identical apart from the wrapper's own fields, the id label and the wall clock; on R4's 26 the trajectories are identical and what
differs is the label rule (v1 vs v2: the tick labels and the label-rule mark), the generator version (`gen-robot-v0.2` → `v0.3`) and the
expert-config digest (R5 added the label-rule keys) — `artifacts/scratch/r10/fix1/generator-versions.json`. The rule changed the wrapped
expert's gripper answer on **0** of 32,372 delegated ticks. Strict 182 / 90 = the unwrapped expert's 182 / 90, every seed-paired difference
0 [0, 0].

**What the wrapped model sees.** Its contract input, which the wrapper leaves as it is: the request, including the commitment projection
(`serialize.py` `_commitment_line`) and the executed-command history (`exec_history … gripper=…`, `state.exec.gripper` / `gripper_wait`) —
under G that history carries the gripper the harness adopted from the rule on earlier ticks. The wrapper adds nothing to that input, and
the model never receives the rule's or the expert's answer for the current tick: the wrapper calls the model first, with the generator's own
objects, and replaces `q_gripper` afterwards. (§0's "never sees the commitment" means the harness commitment argument, which `ModelPolicy`
ignores; the projection in the request is part of every model input by contract.)

### The registered verdict (ood_dev 200)

| condition | pair | difference [95 %] | holds if | holds? |
| --- | --- | --- | --- | --- |
| (a) strict success | `ruleG − r5G` | **+0.095 [+0.030, +0.160]** (0.720 vs 0.625) | lower ≤ 0 | **no** |

Beside (a): discordant seeds 34 (rule judge only) vs 15 (r5G only), exact McNemar p = 0.0094; the lower bound stays above zero under all 200
alternative bootstrap RNG seeds (min +0.020). **The registered call: "실행을 맡겨도 규칙 판정기가 앞선다 — 병목은 의미 판단" (with execution
delegated, the rule judge is still ahead — the bottleneck is semantic judgment).** dev_new2, beside only: +0.070 [−0.010, +0.150] (11 vs 4,
p = 0.118 — it would hold there). **Registered predictions (beside):** P1 `r9s18cG − r9s18c` strict **+0.530 [+0.460, +0.600] — holds**; P2
`(r9s18cG − r9s18c) − (r8s19G − r8s19)` **+0.485 [+0.410, +0.565] — holds** (r8s19's own effect +0.045). Per the registration, **no contract-change
proposal is written** (the results do not support moving `q_gripper` to the execution layer as the way past the rule judge).

**How to read the gap.** Natively the same pair read `rule − r5` **+0.065 [0.000, +0.130]** — zero at the interval's edge, so no
difference was found there; with execution delegated the pair excludes zero (+0.095 [+0.030, +0.160]). That is not evidence that delegation
opened (or failed to close) a gap: neither policy's own delegation effect is a finding — rule judge −0.005 [−0.015, 0.000], r5 −0.035
[−0.075, 0.000], zero at the edge of both — so the native and delegated readings differ by less than these 200 seeds resolve. What the rule
judge is ahead on under G is seed-paired below — false dones and main decisions, not the gripper.

### By policy and arm (ood_dev 200; `done` · **strict** [95 %] · false done · failures main / aux / geometric · gripper reference / executed / missing / duplicate · `q_stop` caught / onsets · stop ticks by `q_stop` / reflex · forbidden-contact onsets · unsafe · `stall_exhausted`)

| policy · arm | `done` | **strict** | false done | main / aux / geom | gripper ref / exec / miss / dup | `q_stop` caught; stop ticks q / reflex | forbidden | unsafe | stalls |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| expert | 182 | **182** [0.870, 0.950] | 0 | 2 / 0 / 16 | 509 / 495 / 14 / 33 | 65/65; 88 / 38 | 23 | 0.0000 | 17 |
| expertG | 182 | **182** [0.870, 0.950] | 0 | 2 / 0 / 16 | 511 / 497 / 14 / 31 † | 65/65; 88 / 38 | 23 | 0.0000 | 17 |
| rule | 145 | **145** [0.665, 0.785] | 0 | 2 / 51 / 2 | 395 / 384 / 11 / 759 | 31/43; 66 / 26 | 9 | 0.0005 | 49 |
| ruleG | 144 | **144** [0.660, 0.780] | 0 | 1 / 5 / 50 | 396 / 383 / 13 / 3 | 29/41; 63 / 24 | 9 | 0.0005 | 51 |
| ruleGS | 145 | **145** [0.665, 0.785] | 0 | 2 / 4 / 49 | 396 / 385 / 11 / 2 | 0/38; 0 / 26 | 9 | 0.0036 | 50 |
| r5 | 162 | **132** [0.590, 0.725] | 30 | 17 / 13 / 8 | 514 / 404 / 110 / 64 | 21/36; 101 / 27 | 3 | 0.0074 | 38 |
| r5G | 149 | **125** [0.555, 0.690] | 24 | 23 / 14 / 14 | 503 / 403 / 100 / 34 | 28/44; 209 / 36 | 5 | 0.0072 | 51 |
| r5GS | 159 | **134** [0.605, 0.735] | 25 | 24 / 12 / 5 | 516 / 418 / 98 / 26 | 0/40; 0 / 34 | 6 | 0.0112 | 41 |
| r6 | 134 | **122** [0.540, 0.675] | 12 | 28 / 38 / 0 | 752 / 623 / 129 / 439 | 3/92; 0 / 75 | 24 | 0.0413 | 65 |
| r6G | 167 | **139** [0.635, 0.760] | 28 | 18 / 13 / 2 | 544 / 482 / 62 / 42 | 1/82; 1 / 60 | 25 | 0.0417 | 33 |
| r9s17 | 145 | **90** [0.380, 0.515] | 55 | 33 / 22 / 0 | 431 / 362 / 69 / 153 | 0/67; 0 / 34 | 15 | 0.0778 | 54 |
| r9s17G | 152 | **92** [0.390, 0.525] | 60 | 30 / 16 / 2 | 420 / 390 / 30 / 31 | 0/59; 0 / 26 | 14 | 0.0768 | 48 |
| r9s17GS | 152 | **92** [0.390, 0.525] | 60 | 30 / 16 / 2 | 420 / 390 / 30 / 31 | 0/59; 0 / 26 | 14 | 0.0768 | 48 |
| r9s18c | 36 | **27** [0.090, 0.185] | 9 | 41 / 123 / 0 | 701 / 68 / 633 / 73 | 0/20; 0 / 16 | 1 | 0.0192 | 164 |
| r9s18cG | 147 | **133** [0.600, 0.730] | 14 | 31 / 22 / 0 | 519 / 440 / 79 / 29 | 0/38; 0 / 42 | 3 | 0.0125 | 53 |
| r8s19 | 84 | **28** [0.095, 0.190] | 56 | 91 / 25 / 0 | 336 / 188 / 148 / 91 | 0/67; 0 / 41 | 24 | 0.1765 | 116 |
| r8s19G | 141 | **37** [0.130, 0.245] | 104 | 52 / 7 / 0 | 405 / 352 / 53 / 26 | 0/96; 0 / 35 | 31 | 0.1778 | 59 |
| mechanical · mechanicalG | 0 · 0 | **0** · **0** | 0 · 0 | 200 / 0 / 0 · same | 0 | — | 0 | — | 200 · 200 |

† R4's 26 expert records carry label rule v1 (§2); with v2 labels the same trajectories show two more reference transitions — the
label-rule difference, not a behaviour difference. r9s17GS is the same 200 episodes as r9s17G because r9s17's `q_stop` never reached 0.5.

**Why reference gripper transitions still go missing under G.** The delegated `q_gripper` is the reference's own rule on every tick, yet
r5G misses 100 of 503 reference transitions (ruleG 13 / 396; even expertG misses 14 of its own 511). The `missing` column counts *adopted*
transitions (`adopted.gripper`, matched within ±10 ticks), and the harness applies the gripper answer only on a tick that keeps the main
decision: on a switch tick it discards all four auxiliary answers and keeps the current gripper (docs/08 §5.4), and stop and gate ticks do
the same. Traced per transition (fix round 1, `artifacts/scratch/r10/fix1/probe.json`), r5G's 100 are: **52** transitions (50 of them closes)
that the harness never adopted within the window — on 280 of the 318 window ticks where the delegated answer was the target state, r5's own
`q_main` was switching candidates (stops 13, gates 6, other ticks 19), and in 23 of the 52 the episode ended within 10 ticks of the
reference transition; **43** opens that followed those same never-executed closes and found the gripper already open; **5** with a
transition in the window that the matcher had paired with a neighbouring reference transition. ruleG's 13 are 8 never adopted (switch 5,
own `q_stop` 2, stall escape 1) + 5 such opens. So under G the column measures how r5's other answers gate adoption, not gripper
disagreement — which is zero by construction (§0's reading rule).

### The delegation effect per policy, seed-paired with native (ood_dev 200)

| policy | strict native → G | `policyG − policy` strict [95 %] | discordant (G only vs native only) | false done native → G (difference) | gripper missing share · duplicates per episode (G − native) |
| --- | ---: | --- | ---: | --- | --- |
| **r9s18c** (gripper never closed) | 27 → **133** | **+0.530 [+0.460, +0.600]** | 106 vs 0 | 9 → 14 (+0.025, 0 inside) | **−0.751** [−0.845, −0.642] · −0.220 [−0.430, −0.035] |
| **r6** (early close, opens in transport) | 122 → **139** | **+0.085 [+0.020, +0.150]** | 32 vs 15 | 12 → 28 (**+0.080** [+0.040, +0.120]) | −0.058 (0 inside) · **−1.985** [−2.905, −1.335] |
| r8s19 (target selection) | 28 → 37 | +0.045 [+0.005, +0.090] | 14 vs 5 | 56 → **104** (**+0.240** [+0.175, +0.305]) | **−0.310** [−0.416, −0.193] · −0.325 [−0.730, −0.070] |
| r9s17 | 90 → 92 | +0.010 [−0.015, +0.040] (0 inside) | 5 vs 3 | 55 → 60 (+0.025, 0 inside) | −0.089 [−0.132, −0.044] · −0.610 [−0.910, −0.370] |
| **r5** (the registered primary model) | 132 → 125 | −0.035 [−0.075, 0.000] (0 at the edge) | 4 vs 11 | 30 → 24 (−0.030 [−0.060, −0.005]) | −0.015 (0 inside) · −0.150 [−0.275, −0.025] |
| rule judge | 145 → 144 | −0.005 [−0.015, 0.000] (0 at the edge) | 0 vs 1 | 0 → 0 | +0.005 (0 inside) · −3.780 [−6.471, −1.590] |
| expert | 182 → 182 | 0 [0, 0] | 0 vs 0 | 0 → 0 | 0 · 0 |
| mechanical | 0 → 0 | 0 [0, 0] | — | — | — |

**Reading.** (1) **Delegation repairs a broken gripper head; for the working ones no gain was found, and on dev_new2 r5 lost.** r9s18c — the
seed whose head never learned "close now" — goes from 27 to 133 strict with 106 seeds gained and none lost (the rule changed its gripper answer
on 2,139 of 20,773 ticks; at the grasp point it had said `open` on 46 % of the ticks). r6 — whose head closed early (`closed` on 82 % of the
not-yet-at-the-grasp-point ticks) and said `open` during transport on 21 % — gains 17 net. For r5 (125 vs 132) and r9s17 (92 vs 90), whose
heads worked, no difference was found on ood_dev 200 (r5 −0.035 [−0.075, 0.000], zero at the edge; r9s17 +0.010 [−0.015, +0.040]) — which is
not the same as "unchanged" —, and on dev_new2 r5 fell: r5G − r5 **−0.080 [−0.150, −0.010]** (r9s17 0.000 [−0.050, +0.050]). So the delegated
rule is not neutral for r5 — one explanation, not tested here, is that r5's own timing is part of how it was trained and DAgger-collected.
(2) **It exposes the semantic failures it does not cause.** r8s19 now
completes grasps (missing share −0.310) — of the wrong object: its first grasp commitment is on a non-instructed object in **139 of 200**
episodes in both arms, and its false dones nearly double (56 → 104, 102 of them after a grasp commitment on a non-target). r6G's extra
completions bring 16 more false dones. (3) **The rule judge's "gripper threshold" failures were a classifier artifact, not gripper failures.**
Natively the rule judge commanded `closed` with the gripper open on 2,489 ticks; on the next tick the executor reported `gripper_wait:
readiness` (the close held) on 2,280 of them and closed on 209 — the real grasps. The label-based classifier counted those held early
closes as ≥ 3-tick disagreements and filed 51 failures as `semantic_aux` (gripper). Under G (gripper commands = the label's source) none of the
51 seeds recovers: 46 are filed `geometric`, 5 `semantic_aux` on `q_path`, and the rule judge loses one more seed (`E2:920145`). (8 of the 51
are R4 ood_dev seeds whose native labels follow rule v1 — §2 —, so on those 8 the reclassification also crosses label rules v1 → v2.) On
dev_new2 the same holds for the 29 of its 30 native auxiliary failures that carry a gripper streak (the 30th, `E2:1000377`, is a `q_path`
streak): under G 27 are filed `geometric`, 3 `semantic_aux` on `q_path`, none recovers (67 → 67). R4–R9's reading that "the rule judge's main
failures were its own gripper threshold (rj0.5)" is therefore withdrawn: its 55 native failures on ood_dev 200 are execution stalls and
timeouts (49 `stall_exhausted`, 6 `max_ms`; under G 56 = 51 + 5), its first grasp commitment is on the instructed object in all 200 episodes,
and it has no false done. The same artifact sits in the models' native `semantic_aux` counts wherever they
closed early (r6 most) — part of why the aux failure shares drop under G, where the gripper streaks they were filed under cannot occur (r6
0.190 → 0.065, r9s18c 0.615 → 0.110, rule 0.255 → 0.025, all excluding zero).

**What remains under G — the semantic judgment (ood_dev 200).**

| arm | strict | false done (after a non-target last grasp commitment; of those, an earlier instruction's target) | first grasp commitment on a non-instructed object | main-decision failures | aux failures (streak questions) | geometric | `wrong_action` ticks / acted |
| --- | ---: | --- | ---: | ---: | --- | ---: | ---: |
| ruleG | 144 | 0 | 0 | 1 | 5 (path 5) | 50 | 0.000 |
| r6G | 139 | 28 (26; 18) | 12 | 18 | 13 (speed 11, path 8, force 5, gripper 3) | 2 | 0.084 |
| r9s18cG | 133 | 14 (13; 8) | 6 | 31 | 22 (force 21, path 19, speed 18, gripper 2) | 0 | 0.075 |
| r5G | 125 | 24 (19; 18) | 0 | 23 | 14 (path 11, speed 5, force 5, gripper 3) | 14 | 0.037 |
| r9s17G | 92 | 60 (56; 31) | 41 | 30 | 16 (path 11, speed 4, force 4, gripper 1) | 2 | 0.235 |
| r8s19G | 37 | 104 (102; 25) | 139 | 52 | 7 (path 5) | 0 | 0.563 |

Seed-paired against the primary model, `ruleG − r5G` on ood_dev 200: false done **−0.120 [−0.165, −0.080]** (0 vs 24 seeds), main-decision
failures −0.110 [−0.155, −0.065], wrong-action ticks −0.037 [−0.055, −0.022], unsafe ticks −0.007 [−0.018, −0.000] (upper bound −0.0003,
zero excluded), gripper duplicates
−0.155 [−0.250, −0.070] — the rule judge's lead over r5G is false dones and main decisions, not the gripper. The false dones are mostly
**stale goals**: of r5G's 24, 19 follow a last grasp commitment on a non-target and 18 of those objects had been an earlier instruction's target
(R9 found the same for R5, 22 of 24). Second evidence `ruleG − r9s17G` +0.260 [+0.195, +0.330] (dev_new2 +0.170 [+0.090, +0.250]).

**Beside, not judged (post hoc).** Under G, for two of the five models no difference from the rule judge was found on ood_dev 200 — r6G 139
(`ruleG − r6G` +0.025 [−0.045, +0.090]) and r9s18cG 133 (`ruleG − r9s18cG` +0.055 [−0.015, +0.120]; `r5G − r6G` −0.070 [−0.135, −0.010],
`r5G − r9s18cG` −0.040 [−0.105, +0.025]) — and in the GS arm `ruleGS − r5GS` reads +0.055 [−0.005, +0.120]. None of these is the registered
comparison: r6 and r9s18c were not the primary model, and picking the best of five (or the better arm) after seeing the holdout is selection on
the judged set — what R9's registration forbids — so these are observations, not findings. Read with the zero rule they do not show that
r6G or r9s18cG matches the rule judge — only that these 200 seeds do not separate them, and both keep their own semantic failures (false dones
28 and 14, main-decision failures 18 and 31); and r5 gains from dropping its own semantic `q_stop` (GS, below: +0.045 [+0.015, +0.080]),
after which no difference from the rule judge was found either (`ruleGS − r5GS` above). Two **unregistered, exploratory** dev_new2 runs were
made after seeing these numbers (r6G and r9s18cG; table below; `artifacts/reports/r10-exploratory.json`).

### GS — the value of dropping the semantic stop (ood_dev 200, `policyGS − policyG`)

| policy | strict GS − G | `q_stop` caught (by construction) | unsafe ticks (by construction) | forbidden-contact onsets / episode | reflex-event ticks / episode | stop ticks by `q_stop` G → GS |
| --- | --- | --- | --- | --- | --- | --- |
| r5 | 125 → 134: **+0.045 [+0.015, +0.080]** (10 vs 1, p = 0.012) | 28/44 → 0/40 | +0.004 [+0.001, +0.008] | +0.005 [0.000, +0.015] (0 inside) | −0.025 [−0.065, 0.000] (0 inside) | 209 → 0 |
| rule | 144 → 145: +0.005 [−0.010, +0.025] (0 inside) | 29/41 → 0/38 | +0.003 [+0.001, +0.006] | 0 [0, 0] | +0.010 [−0.010, +0.030] (0 inside) | 63 → 0 |
| r9s17 | identical episodes (its `q_stop` never fired) | 0/59 → 0/59 | 0 | 0 | 0 | 0 → 0 |

No increase in forbidden contacts or reflex events was found on these scenes when the semantic stop was dropped (onsets 9 → 9 for the rule
judge, 5 → 6 for r5; the intervals contain zero), and for r5 it **helped**: r5's `q_stop` fired on 209 ticks in r5G while catching 28 of 44
onsets, and r5GS gains 10 seeds and loses 1. Traced per seed (fix round 1, `artifacts/scratch/r10/fix1/probe.json`): in r5G all ten gained
seeds had ended `stall_exhausted` after 9–14 stop ticks of r5's own `q_stop` each, and every one of those 130 ticks carries a false `q_stop`
label (r5G failure classes 7 geometric, 2 main-decision, 1 auxiliary) — consistent with false alarms holding the arm until the stall
monitor ended the episode. The `unsafe` rise is the metric's definition (an acted tick whose stop label is true), not a contact. **Caution**: these scenes have few protected contacts (forbidden onsets 9–31 per 200 episodes, all policies), the
force reflex is the executor's, and GS was registered as descriptive — this is evidence for a later decision on `q_stop`, not a finding that
the semantic stop is unnecessary.

### dev_new2 (second evidence, beside; registered rows r5G, r9s17G, ruleG, expertG)

| policy · arm | strict native → G | `policyG − policy` [95 %] | false done native → G |
| --- | ---: | --- | ---: |
| r5 | 68 → 60 | **−0.080 [−0.150, −0.010]** | 11 → 11 |
| r9s17 | 50 → 50 | 0.000 [−0.050, +0.050] | 24 → 24 |
| rule | 67 → 67 | 0 [0, 0] | 0 → 0 |
| expert | 90 → 90 | 0 [0, 0] | 0 → 0 |
| r6 (exploratory, unregistered) | 69 → 74 | +0.050 [−0.050, +0.150] | 3 → 15 (+0.120 [+0.050, +0.190]) |
| r9s18c (exploratory, unregistered) | 14 → 61 | +0.470 [+0.370, +0.570] | 3 → 5 (+0.020 [−0.030, +0.080]) |

`ruleG − r5G` on dev_new2 +0.070 [−0.010, +0.150] (beside the verdict), `ruleG − r9s17G` +0.170 [+0.090, +0.250]. The two exploratory rows
come from `artifacts/reports/r10-exploratory.json` (marked `"registered": false`), not from the registered report or the verdict.

**Latency** (`latency.model_ms` excludes each episode's first tick; the wrapper adds no model time — the rule is a few arithmetic operations):
non-first ticks p95 54.5–57.4 ms over all 25 GPU condition runs (160,342 ticks); **4 ticks over 100 ms**, all in r5G's ood_dev / ood_dev_new
conditions (1 of 2,449 and 3 of 6,111), which ran while the four CPU policy units were up (10:20–10:31; r5G's ood_dev_new2 began at 10:31:38
and overlapped the last of them, expertG, by ≈ 5 s); **0** in the other 151,782, including the conditions that overlapped later CPU units
(the preview, the early CPU test pass, the preliminary report) or the out-of-unit analyses of §5; first ticks 2–28 per condition over 100 ms
(26–100 first ticks each, max 295.4 ms). The gate (p95 ≤ 80 ms, > 100 ms ≤ 5 %) passes everywhere.

## 5. Measured cost

| unit (systemd, `Started` → end) | what | wall |
| --- | --- | ---: |
| `r10-smoke` | GPU smoke: r5G, one ood_dev episode, 20 ticks (output in scratch, not in any table) | 0 h 00 m 38 s |
| `r10-c-models` | the registered GPU chain: r5G · r9s17G · r9s18cG · r8s19G · r6G · r5GS · r9s17GS on ood_dev 200 (19:20–25:18 each), r5G · r9s17G on dev_new2 (11:41 · 10:30) | 2 h 53 m 14 s |
| `r10-x-models` | **unregistered, exploratory**: r6G and r9s18cG on dev_new2 (12:45 · 15:13) | 0 h 27 m 58 s |
| `r10-pytest` | full suite, GPU visible, no other GPU job | 0 h 10 m 18 s |
| **GPU total** | | **12,728 s = 3.54 GPU-h** (registered runs 2.90 h; brief ≈ 3.3 h) |

CPU units (`r10c-*`, CUDA hidden, MemoryMax, `choom`, Nice 10): ruleG 10 m 20 s, expertG 11 m 25 s, ruleGS 6 m 34 s, mechanicalG 10 m 52 s (all on
ood_dev 200 and, for ruleG / expertG, dev_new2; run concurrently with the GPU chain's first model), a preview of the primary pair (5 m 17 s), an
early CPU-only test pass (7 m 48 s; 1 environment failure with CUDA hidden — `test_stream_runner_loads_the_checkpoint_before_compiling_so_lora_keys_survive`
needs CUDA — and 1 skip), the preliminary report (32 m 30 s) and the first final report (33 m 03 s; it reproduced every preliminary number —
1,185 paired blocks, 16 seed-pair blocks and the effect block identical — but also took the two exploratory dev_new2 run files as inputs, so
nine dev_new2 slots of its verdict artifact — P1 and eight stability readings for r6 / r9s18c — were filled from them, unlabelled). Stage A's
rule agreement check: 65 s. Cloud spend 0. Disk: 128 → 126 GB free (closed-loop records ≈ 3 GB). The five checkpoints are unchanged in size
and mtime (`artifacts/scratch/r10/checkpoints-stat-{before,after}.txt`, §1). Cumulative GPU on the DGX Spark ≈ 66.3 h.

**Fix round 1 (review 1; CPU only, GPU 0).** `r10c-fix1-reports` (31 m 39 s, MemoryMax 40G, peak 23.2 GB) rebuilt `r10-closed-loop.json` and
`r10-verdict.json` from the registered runs only — identical to the preliminary pass apart from time stamps, the git hash and the report path
(`artifacts/scratch/r10/fix1/compare.json`), primary block and call unchanged, the nine dev_new2 slots "not available" again — and wrote the
exploratory artifact `artifacts/reports/r10-exploratory.json`; the replaced files are kept in `artifacts/scratch/r10/fix1/replaced/`.
`r10c-fix1-probe` (34 s; gripper transitions, GS seeds, the rule judge's auxiliary failures), `r10c-fix1-footprint` (26 s),
`r10c-fix1-genver`, `r10c-fix1-latency` (≈ 1 s each) and `r10c-fix1-tests` (`tests/test_delegation.py` + `tests/test_closed_loop.py`, CUDA
hidden: 74 passed in 22 s) checked the facts used above.

**Not every CPU job ran as a unit (review 1 I5).** While the GPU chain was up (10:23–12:33), 17 short analysis calls — 21 Python processes,
1–28 s each: `failures.py`, `flips.py`, `identity_check.py`, R9's false-done script, an inline scan, a one-record trace and a one-test pytest
run — read stored records from the session's shell without a unit, MemoryMax or `choom` (only `nice -n 10` and CUDA hidden), against
HANDOFF §5's rule; three of them overlapped a 26 GB checkpoint load (11:02 r9s18cG, 11:27 r8s19G, 12:33 r9s17GS). Re-run in a unit with the
same arguments, the two largest peak at 2.30 and 1.90 GiB RSS; the kernel log for the day has no `NVRM` or OOM line. The commands as run are
in `artifacts/scratch/r10/fix1/out-of-unit-commands.md`; HANDOFF §5 records the lapse.

## 6. What the round says, and what it does not

1. **Registered call: with execution delegated, the rule judge is still ahead of r5 — the bottleneck is semantic judgment** (`ruleG − r5G`
   +0.095 [+0.030, +0.160], 34 vs 15, p = 0.009, robust under all 200 RNG seeds). The rule judge's lead over r5G is false dones (0 vs 24 seeds,
   −0.120 [−0.165, −0.080]) and main decisions (−0.110 [−0.155, −0.065]); the false dones are mostly stale goals (18 of r5G's 24 carried an
   earlier instruction's target). **No contract-change proposal was written** (registered: only if (a) holds).
2. **The gripper's timing is an execution matter in a narrower sense than the hypothesis.** Handing it to the expert's geometric rule repairs
   a model whose gripper head failed — r9s18c 27 → 133 (P1 holds), r6 122 → 139. For the models whose head worked no difference was found on
   ood_dev 200 (r5 −0.035 [−0.075, 0.000], r9s17 +0.010 [−0.015, +0.040]), and r5 fell on dev_new2 (−0.080 [−0.150, −0.010]). So the
   R4–R9 instability of the gripper head *can* be engineered away at serving time, and it was what ruined r9s18c; but it is not what separates
   the best model from the rule judge.
3. **A correction to R4–R9's reading of the rule judge**: its 51 "gripper" failures on ood_dev 200 (and 29 of its 30 auxiliary failures on
   dev_new2) were label disagreements on early `closed` commands that the executor's close readiness held (2,280 of 2,489 such ticks on
   ood_dev 200); under G none of them recovers and they are execution stalls. The label-based failure classifier over-attributes failures to
   `semantic_aux` (gripper) for any policy that closes early. R4's dev 100 failures were not re-run; the correction reaches them by analogy.
4. **Delegation exposes semantic failures rather than causing them**: r8s19 completes the wrong task (false dones 56 → 104, first grasp on a
   non-instructed object 139 / 200 in both arms; P2 holds), r6G's extra completions bring 16 more false dones.
5. **`q_stop` (GS, descriptive)**: dropping the semantic stop raised r5 by +0.045 [+0.015, +0.080] (in r5G its `q_stop` made 209 stop ticks for
   28 of 44 onsets caught; all 10 seeds gained had ended `stall_exhausted` in r5G after 9–14 false-alarm stop ticks each). For the rule judge
   no difference was found (+0.005 [−0.010, +0.025]) and r9s17's episodes were identical (its `q_stop` never reached 0.5); no increase in
   forbidden-contact onsets (9 → 9, 5 → 6) or reflex events was found. This is evidence for a later `q_stop` decision, not a finding that the
   semantic stop is unnecessary — the scenes have few protected contacts.
6. **Post hoc, not judged**: for r6G and r9s18cG no difference from the rule judge was found on ood_dev 200 (+0.025 [−0.045, +0.090], +0.055
   [−0.015, +0.120]); the exploratory dev_new2 runs, made after seeing those numbers, read r6G 74 (rule judge 67: `ruleG − r6G` −0.070 [−0.150,
   0.000]) and r9s18cG 61 (+0.060 [−0.030, +0.150]) (`artifacts/reports/r10-exploratory.json`). Selection on the judged set is exactly what the
   registration avoided; a fair test of "the best delegated model" needs a choice made on dev_new2 and judged on fresh holdout scenes.

**Recommendation for the next round (from these numbers):** keep the gripper delegation available as an execution-side safety net for a
checkpoint whose gripper head failed (it is exactly the label's source and removes that failure mode — r9s18c 27 → 133), but not as a free
default: for r5, whose head worked, no gain was found on ood_dev 200 and it cost 8 seeds net on dev_new2 (11 lost, 3 gained; −0.080
[−0.150, −0.010]). Point
training at the semantic failures that remain under it — false done after a non-target grasp (stale goals) and target selection, which DAgger
cycles 0–1 have not fixed — before any cloud spend. Whether to make the delegation part of the contract (a harness change, h0.10, which
re-digests every checkpoint) is the user's decision; R10 did not register evidence that it closes the gap to the rule judge.
