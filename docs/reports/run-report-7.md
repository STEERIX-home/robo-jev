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
`fused` + dense compile, fp32 readout — the same path as run reports 1–6). Size and mtime before the first R10 load
(`artifacts/scratch/r10/checkpoints-stat-before.txt`) and after the last (`…-after.txt`) are identical:

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
from `artifacts/reports/r10-run-*.json`), verdict `artifacts/reports/r10-verdict.json` (`closed_loop.py verdict --registration
configs/eval/r10-registration.yaml`), tables printed by `artifacts/scratch/r10/tables.py`, failure and target-selection counts by
`artifacts/scratch/r10/failures.py`, per-seed flips by `flips.py`, the expert identity check by `identity_check.py`, the false-done mechanism by
R9's `artifacts/scratch/r9/fix1/false_done_mechanism.py` — all reading records through the manifest-first reader.

**The wrapper did what it was registered to do.** The expert wrapped in G (`expertG`) produced **the same episodes as the unwrapped expert on all
300 scenes** (`artifacts/scratch/r10/identity-check.json`): on the 274 scenes whose stored expert records carry label rule v2 (R6/R7) the records
are identical apart from the wrapper's own fields, the id label and the wall clock; on R4's 26 the trajectories are identical and only the
labels differ (v1 vs v2). The rule changed the wrapped expert's gripper answer on **0** of 32,372 delegated ticks. Strict 182 / 90 = the
unwrapped expert's 182 / 90, every seed-paired difference 0 [0, 0].

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

**How to read the gap.** Natively the same pair read `rule − r5` **+0.065 [0.000, +0.130]** — zero at the interval's edge, so not a finding.
Delegation did not close it: it moved the rule judge by −0.005 and r5 by −0.035 [−0.075, 0.000], and the delegated pair excludes zero. What the
rule judge is ahead on is seed-paired below — false dones and main decisions, not the gripper.

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

**Reading.** (1) **Delegation repairs a broken gripper head completely and does nothing for a working one.** r9s18c — the seed whose head never
learned "close now" — goes from 27 to 133 strict with 106 seeds gained and none lost (the rule changed its gripper answer on 2,139 of 20,773
ticks; at the grasp point it had said `open` on 46 % of the ticks). r6 — whose head closed early (`closed` on 82 % of the not-yet-at-the-grasp-
point ticks) and said `open` during transport on 21 % — gains 17 net. r5 (125 vs 132) and r9s17 (92 vs 90), whose heads worked, do not move
(r5's −0.035 has zero at the interval's edge; on dev_new2 r5G − r5 is **−0.080 [−0.150, −0.010]**, so the delegated rule is not neutral for r5:
r5's own timing is part of how it was trained and DAgger-collected). (2) **It exposes the semantic failures it does not cause.** r8s19 now
completes grasps (missing share −0.310) — of the wrong object: its first grasp commitment is on a non-instructed object in **139 of 200**
episodes in both arms, and its false dones nearly double (56 → 104, 102 of them after a grasp commitment on a non-target). r6G's extra
completions bring 16 more false dones. (3) **The rule judge's "gripper threshold" failures were a classifier artifact, not gripper failures.**
Natively the rule judge commanded `closed` with the gripper open on 2,489 ticks; on the next tick the executor reported `gripper_wait:
readiness` (the close held) on 2,280 of them and closed on 209 — the real grasps. The label-based classifier counted those held early
closes as ≥ 3-tick disagreements and filed 51 failures as `semantic_aux` (gripper). Under G (gripper commands = the label's source) none of the
51 seeds recovers: 46 are filed `geometric`, 5 `semantic_aux` on `q_path`, and the rule judge loses one more seed (`E2:920145`). R4–R9's reading
that "the rule judge's main failures were its own gripper threshold (rj0.5)" is therefore withdrawn: its 55 failures are execution stalls
(`stall_exhausted` 51) with correct targets and no false done. The same artifact sits in the models' native `semantic_aux` counts wherever they
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
failures −0.110 [−0.155, −0.065], wrong-action ticks −0.037 [−0.055, −0.022], unsafe ticks −0.007 [−0.018, 0.000], gripper duplicates
−0.155 [−0.250, −0.070] — the rule judge's lead over r5G is false dones and main decisions, not the gripper. The false dones are mostly
**stale goals**: of r5G's 24, 19 follow a last grasp commitment on a non-target and 18 of those objects had been an earlier instruction's target
(R9 found the same for R5, 22 of 24). Second evidence `ruleG − r9s17G` +0.260 [+0.195, +0.330] (dev_new2 +0.170 [+0.090, +0.250]).

**Beside, not judged (post hoc).** Under G two of the five models are within the interval of the rule judge on ood_dev 200 — r6G 139
(`ruleG − r6G` +0.025 [−0.045, +0.090]) and r9s18cG 133 (`ruleG − r9s18cG` +0.055 [−0.015, +0.120]; `r5G − r6G` −0.070 [−0.135, −0.010],
`r5G − r9s18cG` −0.040 [−0.105, +0.025]) — and in the GS arm `ruleGS − r5GS` reads +0.055 [−0.005, +0.120]. None of these is the registered
comparison: r6 and r9s18c were not the primary model, and picking the best of five (or the better arm) after seeing the holdout is selection on
the judged set — what R9's registration forbids — so these are observations, not findings. They say two things the verdict does not: the gap is
not a property of every checkpoint (r6G and r9s18cG are not distinguishable from the rule judge here, with their own semantic failures — false
dones 28 and 14, main-decision failures 18 and 31), and part of r5G's gap is its own semantic `q_stop` (GS, below). Two **unregistered,
exploratory** dev_new2 runs were made after seeing these numbers (r6G and r9s18cG; table below).

### GS — the value of dropping the semantic stop (ood_dev 200, `policyGS − policyG`)

| policy | strict GS − G | `q_stop` caught (by construction) | unsafe ticks (by construction) | forbidden-contact onsets / episode | reflex-event ticks / episode | stop ticks by `q_stop` G → GS |
| --- | --- | --- | --- | --- | --- | --- |
| r5 | 125 → 134: **+0.045 [+0.015, +0.080]** (10 vs 1, p = 0.012) | 28/44 → 0/40 | +0.004 [+0.001, +0.008] | +0.005 [0.000, +0.015] (0 inside) | −0.025 [−0.065, 0.000] (0 inside) | 209 → 0 |
| rule | 144 → 145: +0.005 [−0.010, +0.025] (0 inside) | 29/41 → 0/38 | +0.003 [+0.001, +0.006] | 0 [0, 0] | +0.010 [−0.010, +0.030] (0 inside) | 63 → 0 |
| r9s17 | identical episodes (its `q_stop` never fired) | 0/59 → 0/59 | 0 | 0 | 0 | 0 → 0 |

Dropping the semantic stop cost nothing measurable in forbidden contacts or reflex events on these scenes (9 → 9 for the rule judge, 5 → 6
for r5), and for r5 it **helped**: r5's `q_stop` fired on 209 ticks in r5G while catching 28 of 44 onsets — false alarms that stall the arm
(r5GS gains 10 seeds, 7 of them `geometric` failures in r5G). The `unsafe` rise is the metric's definition (an acted tick whose stop label is
true), not a contact. **Caution**: these scenes have few protected contacts (forbidden onsets 9–31 per 200 episodes, all policies), the
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

`ruleG − r5G` on dev_new2 +0.070 [−0.010, +0.150] (beside the verdict), `ruleG − r9s17G` +0.170 [+0.090, +0.250].

**Latency** (`latency.model_ms` excludes each episode's first tick; the wrapper adds no model time — the rule is a few arithmetic operations):
non-first ticks p95 54.5–57.4 ms over all 25 GPU condition runs (160,342 ticks); **4 ticks over 100 ms**, all in r5G's ood_dev / ood_dev_new
conditions (4 of 8,560 — the only conditions that ran while the four CPU policy units were up, 10:20–10:31), **0** in the other 151,782;
first ticks 2–28 per condition over 100 ms (26–100 first ticks each, max 295.4 ms). The gate (p95 ≤ 80 ms, > 100 ms ≤ 5 %) passes everywhere.

