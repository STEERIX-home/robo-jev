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
