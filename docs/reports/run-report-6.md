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

## 1. Trained weights (Stages B and C)

| run | steps | seed | checkpoint | contract digest | training | log-only monitor (R8's statistic: step 150, `q_gripper`, steps 131–150) | gripper fit onset (registered definition) |
| --- | ---: | ---: | --- | --- | --- | --- | ---: |
| **`r9s17`** (run id `r9-t1-fp32-2b-s17`) | **233 — completed** in one process | 17 | `artifacts/runs/r9-t1-fp32-2b-s17/checkpoint.pt` (model + optimizer) | `93fe26725a4c…` | inside the chain unit `r9-rest` (17:41:39 → 21:37:06 = **14,127 s = 3.92 GPU-h**; train 13,375.3 s, 57.40 s/step, p50 47.54, load 140.1 s), peak **56.05 GiB** allocated (60,185,186,816 B) / 57.68 GiB reserved, OOMs 0, host RSS 13.45 → 14.30 GiB; loss 2.672 → **0.304**, mean of the last 20 steps 0.365 | 0.3662 / 0.5688 = **0.644** — would have *passed* R8's monitor; stops nothing | **139** |
| **`r9s18c`** (run id `r8-t1-fp32-2b-s18`, the checkpoint's) | **233 — completed on one schedule**: 1–150 by R8's unit, 151–233 resumed (`summary.rescheduled: null`) | 18 | `artifacts/runs/r9-t1-fp32-2b-s18c/r8-t1-fp32-2b-s18/checkpoint.pt` (model + optimizer, 26,350,502,421 B) | `93fe26725a4c…` | unit `r9-t1-s18c` wall 15:04:15 → 16:36:47 = **5,552 s = 1.54 GPU-h** (load 164.3 s; steps 151–233 5,155.0 s = 62.1 s/step), peak **56.05 GiB** allocated (60,185,186,816 B) / 57.66 GiB reserved, OOMs 0, host RSS 35.63 GiB after the load (the resume holds the checkpoint state on the host); loss 0.526 (150) → **0.149** (233), mean of the last 20 steps 0.287 | 0.3682 / 0.5094 = 0.723 — R8's own steps, unchanged; stops nothing | **152** (to step 150 only: 142, R8's reading) |

**`r9s18c` — seed 18 run to the end.** The resume was the plain resume path (same `max_steps`, no reschedule) from R8's step-150 full
checkpoint, written to a new directory; R8's run directories were read only — size, mtime and sha256 of their files were recorded
before the launch and are identical after it (`artifacts/scratch/r9/r8-runs-{before,after}.txt`; checkpoint sha256 `ff2a682bc4ee…`).
The per-question curve after step 150 (`metrics.json`, window means; ratio = mean loss / mean constant-prior baseline):

| window | `q_gripper` loss / baseline (ratio) | settled | initiate ("close now") | open | `q_stop` ratio (true ticks fired) | `q_main` ratio |
| --- | --- | ---: | ---: | ---: | --- | ---: |
| 131–150 (R8's, before the stop) | 0.368 / 0.509 (0.723) | 266/593 = 0.449 | 0/32 | 1,205/1,233 | 3.48 (0/39) | 0.031 |
| 151–170 | 0.306 / 0.530 (0.578) | 611/698 = 0.875 | 1/43 | 1,375/1,472 | 5.74 (0/2) | 0.051 |
| 171–190 | 0.261 / 0.563 (0.463) | 525/607 = 0.865 | 4/48 | 1,077/1,130 | 1.35 (0/12) | 0.070 |
| 191–210 | 0.233 / 0.563 (0.413) | 627/688 = 0.911 | 12/31 | 960/1,134 | 2.06 (0/24) | 0.043 |
| 211–233 | 0.235 / 0.535 (0.440) | 694/774 = 0.897 | 2/62 | 1,339/1,488 | 1.41 (0/35) | 0.045 |

**The gripper fit that began at step 142 continued** — the head stayed fitted on the settled ticks for all 83 resumed steps (0.865–0.911
per window; ratio to the constant head 0.41–0.58), and the registered onset over the full record is step **152**: the first resumed
step's batch (`ep-E2-420242`) read settled 14/52, so the durable run of ≥ 0.5 steps starts at 152 (through step 150 alone the same
definition gives 142). It fitted less cleanly than seed 19 did at the end (seed 19's last 23 steps: settled 613/613, ratio 0.10–0.29;
here 0.90 and 0.44), "close now" stayed weak (initiate 19/184 over steps 151–233), and some `open` ticks were answered `closed` in the
last windows (open 0.847–0.900). `q_stop` never fired on a true tick (0 of 73 in the resumed steps; 0 of 209 over the run) and stayed
above its constant head. The main-decision head stayed fitted (`q_main` 0.04–0.07 of its uniform baseline). Realized draws over the 233
steps: expert **192** (0.82 epoch) · DAgger **41** (dagger-0 13 · dagger-1 15 · done-gate 13; expected ≈ 181 / 52 — the seeded material
coin 1.7 SD below the DAgger mean, like seed 19's 1.8 SD); raw-false-done episodes drawn: dagger-1 3 of 22, done-gate 1 of 21; tokens
11,758,944.

**`r9s17` — seed 17 from scratch** (unit `r9-rest`, on commit `0278238`, `git.dirty: false`; `--seed 17` sets the candidate
permutation seed too). **Step 1 drew the same units as R2's / R3a's seed-17 runs** (`ep-E0-400131` and the non-robot bundle starting
`rules-0133-0`) and its loss is **2.6720494627952576** against their 2.6720505952835083 — 1.1 × 10⁻⁶ apart because R2 trained on the v1
gripper labels (R8 saw the same for seed 19); step 2 is the first DAgger draw (`ep-E1-960163-r5-r5`), a bucket R2 did not have. Realized
draws: expert **178** (0.76 epoch) · DAgger **55** (dagger-0 21 · dagger-1 14 · done-gate 20; expected ≈ 181 / 52 — on the mean, where seeds
18 and 19 came out 1.7 and 1.8 SD below it); raw-false-done episodes drawn: dagger-1 1 of 22, done-gate 3 of 21; non-robot 1,436 records; tokens
11,578,292. The same shape as the other two seeds — a flat gripper head for ≈ 100 steps (settled 0/606 in steps 41–60 and 0/649 in 101–120,
with short-lived fits at steps 61–100 that collapsed again), then a late fit: settled 57/68 at step 139 and ≥ 0.5 on every counted step
after it, so the **registered onset is 139** (seed 19 132, seed 18 152 over its full record). Unlike seeds 18 and 19, **seed 17 then learned
"close now"** in the training batch: initiate 123/227 over steps 151–233 (76/94 in the last window; seed 18c 19/184, seed 19 34/151), with
settled 622/622 in the last window and open 4,591/4,750 over steps 151–233. `q_stop` fired on 2 of 184 true ticks, both at step 1 (the untrained head), and on none after; it
stays above its constant head (ratio 0.97–5.44). The main head fit early and ended looser than the other two (`q_main` 0.09 of its uniform
baseline in steps 211–233; seeds 18c / 19 0.045 / 0.048).

Side by side (training batch, window means; ratio = mean loss / mean constant-prior baseline):

| window | seed 17: `q_gripper` ratio · settled · initiate | seed 18 (to 150: R8, then `r9s18c`) | seed 19 (R8) |
| --- | --- | --- | --- |
| 41–60 | 1.217 · 0/606 · 0/67 | 1.173 · 36/564 · 2/72 | 1.296 · 145/577 · 31/46 |
| 101–120 | 1.025 · 0/649 · 0/77 | 1.048 · 50/729 · 2/39 | 0.965 · 1/642 · 0/33 |
| 131–150 (R8's monitor window) | **0.644** · 441/721 · 21/57 | **0.723** · 266/593 · 0/32 | **0.592** · 571/704 · 9/60 |
| 151–170 | 0.320 · 692/717 · 29/46 | 0.578 · 611/698 · 1/43 | 0.231 · 559/597 · 14/43 |
| 191–210 | 0.163 · 561/561 · 18/51 | 0.413 · 627/688 · 12/31 | 0.102 · 721/722 · 6/28 |
| 211–233 | 0.143 · 622/622 · **76/94** | 0.440 · 694/774 · 2/62 | 0.186 · 613/613 · 12/43 |

(Every window's row for all three seeds: `artifacts/scratch/r9/c-facts-s{17,18c,19}.json`, printed side by side by
`artifacts/scratch/r9/curves_side_by_side.py`.)

## 3. Resume verification (Stage B)

**The resume path accepts a monitor-stopped unit, and no trainer change was made.** `robo_jev.checkpoint.load_checkpoint` and
`Trainer._load` do not read `status`; a CPU test committed with §0 stops a tiny run with the monitor, resumes it with
`head_fit_monitor: null` into another `artifacts_dir` and matches an uninterrupted run bit for bit, with the stopped run's files
byte-identical. On the GPU, unit `r9-t1-s18c` (on the registration commit `9437d57`, `git.dirty: false`) resumed
`artifacts/runs/r8-t1-fp32-2b-s18/checkpoint.pt` with the command `scripts/adapt_readout.py --config configs/train/qwen35-2b-r8.yaml
--mode t1 --steps 233 --seed 18 --set checkpoint_every=50 --set 'checkpoint_keep_steps=[]' --set head_fit_monitor=null --set
resume=artifacts/runs/r8-t1-fp32-2b-s18/checkpoint.pt --set artifacts_dir=artifacts/runs/r9-t1-fp32-2b-s18c --run-id
r8-t1-fp32-2b-s18 --no-eval`. The trainer's own checks passed on launch (`check_contract` on the digest; `resume_config_differences` —
the keys that differ are the resume-free `resume`, `artifacts_dir` and `head_fit_monitor`, and the path keys `model_config` and
`dataset_manifests[].path`, which now point into the R9 worktree and are compared by content; the identity block — data, tokenizer,
serializer, question set, model — equals R8's). This is the same check R5 applied to its plain resume; the GPU resume gate (`p1_acceptance.py --gate t1`) was
not re-run — the licence is R2 A1's verdict, as in R5.

**The seam at step 150 → 151** (`metrics.json`): steps 1–150 of the new record are R8's history byte for byte (every field, including
step seconds); the learning rates continue on the one cosine schedule — backbone 3.0948e-6 → **3.0293e-6**, readout 9.284e-5 → 9.088e-5,
exactly seed 19's values at the same steps; the sampler cursor continues — `drawn` 300 → 302 (one robot episode and one non-robot bundle
per step), robot/existing 123 → 124, robot/error_family 27 → 27, non-robot records 876 → 882 — with no unit repeated or skipped; the
optimizer reached step 233 and the run ended `completed`; the loss reads 0.526 (150) → 0.511 (151) → 0.218 (152), gradient norm 24.16 →
24.04 → 8.88.
