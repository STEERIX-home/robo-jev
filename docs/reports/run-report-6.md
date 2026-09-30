# Run report 6 — one recipe's seed distribution: seed 18 run to the end, seed 17 added, one seed chosen on the validation set (2026-09-30)

The document docs/06 Task 6 defines (:384) for the seventh and eighth trained checkpoints of this project (Task R9). Every number
is copied from an artifact named next to it. Korean prose lives in `docs/06`, `docs/08` and `HANDOFF.md`; this report is the
English ledger. Run reports 1–5 stay as they are; this one only adds. **Sections are appended as each stage's numbers land;
§0 was committed before any R9 training step or number existed.**

## 0. Pre-registration (written 2026-09-30 before any R9 training step or number; machine-readable copy `configs/eval/r9-registration.yaml`)

The rule below is applied by code, not by hand: `scripts/closed_loop.py verdict --report artifacts/reports/r9-closed-loop.json
--registration configs/eval/r9-registration.yaml --out artifacts/reports/r9-verdict.json` first reads each candidate's training
run (`groups[].run/metrics.json` — completed at its `max_steps` or not; `run_readings`), then `robo_jev.closed_loop.apply_registration`
chooses one seed on the validation set (`select_seed`) and applies R7's three conditions **to the chosen seed only**. Tests in
`tests/test_closed_loop.py` pin the choice (most strict successes, then fewer false dones, then the smaller seed), that the choice
reads nothing of the judged set, that a registration choosing on the judged set or its parts is refused, that a candidate whose run
did not complete cannot be chosen, that the conditions read the chosen seed's pairs only, the registered sentences, and the
run readings (on R8's published curves they return R8's numbers).

**Why this round.** R8 (run report 5) ran one recipe — R5's recipe with the DAgger cycle-1 material, `configs/train/qwen35-2b-r8.yaml`
— at seeds 18 and 19. **Seed 18** was stopped at step 150 by the registered head-fit monitor (`q_gripper` 0.723 × its baseline ≥ 0.70):
by the rule, but a **boundary call** (under every consistent alternative definition it sits within 0.014 of the value the constant
was calibrated on, and behaviourally it was not r7-like — settled 0.886 offline against r7's 0.043). Its **step-150 full checkpoint**
(`artifacts/runs/r8-t1-fp32-2b-s18/checkpoint.pt`, 26.35 GB with the optimizer, status `stopped_head_not_fitting`) still exists.
**Seed 19** completed and its closed loop collapsed (ood_dev 200 strict 28; the first grasp went to a non-instructed object in 139
of 200 episodes; offline instruction-shuffle margin +0.064, direction still above zero). R3a's seeds 17 / 18 / 19 read +0.111 /
+0.247 / +0.128 on an older recipe. The lesson was *do not judge a recipe by one seed* — and this recipe has no seed distribution yet.

R9 does **not change the recipe**. It runs seed 18 to the end (did the boundary stop discard a good seed?), adds seed 17 from
scratch, and so measures the distribution over **three seeds** (17, 18, 19) of one recipe. It then tests, under a rule registered
here, a practical procedure: **run several seeds, choose one on a validation set, judge only the chosen one on the holdout.**

**Recipe — unchanged** (`configs/train/qwen35-2b-r8.yaml`): Qwen3.5-2B bf16 with fp32 master weights, full text backbone + fp32
pointer readout rank 64, `backbone_lr` 1e-5, `readout_lr` 3e-4, 5-second TBPTT chunks, 30-tick window, tick weights 0.25 / 2 / 2 / 1,
candidate permutation seeded by the run seed, material shares 0.7 / 0.2 / 0.1 (→ 0.78 / 0.22 of the robot units), **`max_steps` 233**,
`checkpoint_every` 50. Data: the g2 expert episodes, DAgger-0, DAgger-1 and the done-gate collection (one `error_family` bucket) and
the non-robot split-by-split set — R8's five manifests, `splits: [train]`, no ood_dev-family record.

**Runs** (closed-loop labels in bold):
* **`r8s19`** — R8 seed 19, as it is (233 steps, completed; `artifacts/runs/r8-t1-fp32-2b-s19`; loop records
  `artifacts/datasets/r8-closed-loop/r8s19/`). Nothing is re-run.
* **`r9s18c`** — seed 18 **resumed from R8's step-150 checkpoint on the same schedule** (`max_steps` 233, no reschedule) to step 233.
  A resumed run keeps the checkpoint's run id (`r8-t1-fp32-2b-s18`), so the new run directory is set through `artifacts_dir`:
  **`artifacts/runs/r9-t1-fp32-2b-s18c/r8-t1-fp32-2b-s18/`**. R8's directory is read only: size, mtime and sha256 of its files are
  recorded before the launch (`artifacts/scratch/r9/r8-runs-before.txt`) and compared after. No trainer allowance is needed: the
  resume path does not refuse a `stopped_head_not_fitting` checkpoint — confirmed on CPU before this registration
  (`tests/test_resume.py::test_a_run_the_head_fit_monitor_stopped_resumes_into_a_new_directory_on_the_same_schedule`: the stopped run's
  files stay byte-identical, the run continues into the new directory, and steps after the stop equal an uninterrupted run bit for bit).
* **`r9s17`** — the same recipe at **seed 17 from scratch** (`--seed 17`, run id `r9-t1-fp32-2b-s17`).
* **The monitor is log-only in both new runs.** They train with `head_fit_monitor: null` (a resume-free key; the per-question losses and
  probes are still logged every step). R8's statistic (`q_gripper`, step 150, window 20, ratio 0.70) is recomputed from `metrics.json`
  and written beside (`run_readings.monitor_log`); it stops nothing, chooses nothing and judges nothing — R8's review found its
  per-episode baseline a weak instrument for this decision.

**Scenes.** Judged set **`ood_dev200`** = R4's 26 ood_dev seeds + R6's 74 + R7's 100 (`--merge ood_dev200=ood_dev,ood_dev_new,ood_dev_new2`;
0 origin groups shared with training). Validation set **`dev_new2`** 100 (R6's new dev seeds; 27 of 31 origin groups are training
material). Rows: expert · rule judge · mechanical · r5 · r6 from the stored R4 / R6 / R7 runs, `r8s19` from R8, `r9s18c` and `r9s17` new
(all four loop conditions each: ood_dev, ood_dev_new, ood_dev_new2, dev_new2; id tag `r9`).

**Choosing the seed (validation).** Among `r8s19`, `r9s18c` and `r9s17`, the seed with the **most strict successes on dev_new2**
(`done ∧ target_inside_zone`, 100 episodes). Ties: fewer dev_new2 false dones, then the smaller seed number. **No ood_dev 200 number
is used to choose**: `select_seed` reads only the dev_new2 table, and `load_registration` refuses a registration whose choosing
condition is the judged set or one of the conditions merged into it. A candidate whose run did not complete at its `max_steps` cannot
be chosen (the reason is written); the candidates must have run the same dev_new2 scenes (the code checks the shared seed count).

**Verdict (holdout) — for the chosen seed s only, on `ood_dev200`, R7's three conditions:**
* **(a)** `rule − s` strict success: interval **lower bound ≤ 0**.
* **(b)** `s − r5` false-done rate: interval **upper bound < 0**.
* **(c)** `s − r5` duplicate gripper transitions per episode: interval **lower bound ≤ 0**; **and** `s − r5` `q_stop` catch rate:
  interval **upper bound ≥ 0**.

**All four hold → "the procedure — run several seeds, choose on the validation set — works; the cloud is worth using to run seeds in
parallel (provider, account and budget remain the user's decision)". Otherwise → "the chosen seed did not pass — the next round is
training stability (the gripper head's flat phase, `q_stop`)."** Next to (a) and (b): discordant seeds, exact McNemar p and the bound
distribution under bootstrap RNG seeds 1–200 — they do not change the verdict. dev_new2's values of the same conditions are written
beside only; it is the set the seed was chosen on, so it is not evidence for the verdict.

**Distribution readout (descriptive — not used to choose or to judge).** For all three seeds: dev_new2 and ood_dev 200 strict success,
false done, gripper duplicates and `q_stop` catch (`distribution` in the verdict file); the primary-stratum instruction-shuffle margin on
the decision cell (hash `6a3b69131243`; `scripts/decision_cell_strata.py --runs r9`, with R5 and R8's rows read from their stored
reports); seed-paired differences `r9s18c − r8s19`, `r9s17 − r8s19`, `r9s18c − r9s17` (strict, false done, duplicates, `q_stop`); the
log-only monitor value; and the step at which the gripper head began to fit. **Gripper fit onset (definition):** on the in-training
probe, count only steps with ≥ 5 `settled` ticks; the onset is the earliest counted step from which every counted step to the end of
the record has argmax accuracy ≥ 0.5 (`robo_jev.closed_loop.gripper_fit_onset`). On R8's curves this definition returns seed 19 → 132
and seed 18 (to step 150) → 142 — R8's published readings. A seed-paired difference whose interval contains zero means no difference
between the seeds was found, not that they are equal; three seeds are not a sample from which to build an interval for the recipe.

**Also reported, not part of the rule.** Stage B: the trainer's resume checks (contract digest; config differences only in resume-free
keys; the identity block), continuity across step 150 → 151 (learning rates on one cosine schedule, the sampler cursor, the units
by material, optimizer steps reaching 233, status `completed`), the per-question losses after step 150 (does the gripper fit that began
at step 142 continue?), wall and peak memory, R8's files before/after. Stage C: seed 17's step-1 loss and first draws, realized units by
material, the per-question curves, the gripper onset, `q_stop`, wall, peak GiB, s/step. Stage D: offline `q_gripper`
settled / initiate / open, `q_stop` onsets and `q_done` strata per seed on both cells; seed-paired `s − r5` loop tables; latency
(first ticks separately).

**Pairing and interval.** By seed (`profile:seed`), 95 % percentile bootstrap over seeds, 2,000 resamples, RNG seed 20260921
(`EPISODE_BOOTSTRAP`); `paired_success(strict=True)`, `paired_false_done`, `paired_seed_ratio`. Zero inside = not a finding. Strict success is
primary. Latency reports each episode's first tick separately.

**Order and budget (brief).** A (this commit) → B (≈ 1.3 GPU-h) → D for `r9s18c` → C (≈ 4.1 GPU-h) → D for `r9s17` → E; GPU ≈ 7.5 h.
Running D for `r9s18c` before C follows the brief's priority order (A → B → D(`r9s18c` loop) → C → D(`r9s17`) → offline cells → rest)
and changes nothing registered.
