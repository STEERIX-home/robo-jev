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

## 2. Reproducible data

The training data list is R8's, unchanged (five manifests, `splits: [train]`): robot train labels with gripper rule v2
(`artifacts/datasets/r1-robot/r1-rollout-labels-g2/manifest.json` `4d433bae6d22…`, 233 train episodes), DAgger-0
(`r5-dagger/dagger-0` `8b205ee13993…`), DAgger-1 (`r6-dagger/dagger-1` `6351ec247d35…`), the done-gate collection
(`r6-dagger/dagger-1-donegate` `6b44fc3e3492…`) — one `error_family` bucket — and the non-robot split-by-split set
(`r1/single-by-split` `cfd632dd568f…`); items loaded 3,453 = 833 stream + 2,620 single in both new runs; tokenizer `0997f410c57a…`, serializer
`ts0.6`. No ood_dev-family record is in any training manifest, and no sealed file was opened by training, evaluation or analysis code
(manifest-first loaders; the closed-loop configs refuse `ood_test`). **No new dataset was built.** R9 writes only closed-loop records
(`artifacts/datasets/r9-closed-loop/{r9s18c,r9s17}/{ood_dev,ood_dev_new,ood_dev_new2,dev_new2}/`, id tag `r9`), which no training reads. The
closed-loop scenes are R7's: `ood_dev200` = `r4-seeds.json` 26 + `r6-seeds.json` 74 + `r7-seeds.json` 100; `dev_new2` = `r6-seeds.json` 100.
R8's run directories were read, never written (size, mtime and sha256 before and after, §1). No checkpoint was slimmed.

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

## 4. Per-question quality (offline; the decision cell `ood_dev` 26 episodes / 3,162 ticks, hash `6a3b69131243`; the `dev` cell 42 / 4,168, `9b484441b23b`)

Same serving loader and cells as run reports 2–5 (`artifacts/reports/r9-{reeval,dev}-2b-t1-fp32-r9s{18c,17}.json`, strata
`artifacts/reports/r9-{decision,dev}-cell-strata.json` from `scripts/decision_cell_strata.py --runs r9` / `r9dev`; R5 and R8's rows are their
stored reports). Contract digest `93fe26725a4c…` in all four new reports.

| checkpoint (decision cell) | primary stratum (235) | **instruction-shuffle margin** [95 %] | state-shuffle margin | `grasp` (97): model / instr. margin | LOO (26) excluding 0 (worst) | whole `q_main` | `q_gripper` v2 initiate (43) / settled / open / whole | `q_done` post_release_other | `q_stop` onsets / 10 | goal-change immediate |
| --- | ---: | --- | ---: | --- | --- | ---: | --- | ---: | ---: | ---: |
| **seed 17 — `r9s17` (233)** | 0.830 | **+0.196 [+0.133, +0.253]** | +0.183 | 0.825 / +0.454 [+0.273, +0.636] | 26/26 (+0.175) | 0.987 | **0.209** / 0.994 / 0.987 / 0.978 | 0.952 | **0** | 0.778 |
| **seed 18 — `r9s18c` (150 → 233)** | 0.843 | **+0.217 [+0.148, +0.287]** | +0.174 | 0.887 / +0.536 [+0.358, +0.722] | 26/26 (+0.157) | 0.985 | **0.023** / 0.929 / 0.901 / 0.898 | 1.000 | **0** | 0.844 |
| **seed 19 — `r8s19` (233, R8)** | 0.749 | **+0.064 [+0.020, +0.104]** | +0.055 | 0.577 / +0.113 [+0.014, +0.206] | 26/26 (+0.041) | 0.981 | **0.372** / 0.994 / 0.962 / 0.964 | 0.984 | **0** | 0.667 |
| seed 18 stopped at step 150 (R8, for the pair below) | 0.855 | +0.209 [+0.156, +0.258] | +0.187 | 0.835 / +0.505 | 26/26 | 0.986 | 0.047 / 0.886 / 0.936 / 0.905 | 0.742 | 0 | 0.822 |
| R5 (seed 18, R5 recipe) | 0.902 | +0.298 [+0.232, +0.354] | +0.298 | 0.948 / +0.670 | 26/26 | 0.992 | 0.674 / 0.996 / 0.981 / 0.982 | 0.984 | 5 | 0.911 |

Paired by episode (registered bootstrap): **`r9s18c − r8s19`** primary +0.094 [+0.026, +0.158], instruction margin **+0.153 [+0.090, +0.220]**,
initiate `closed` −0.349 [−0.460, −0.222]; **`r9s17 − r8s19`** primary +0.081 [+0.033, +0.130], instruction margin **+0.132 [+0.083, +0.178]**,
initiate −0.163 [−0.333, +0.000] (0 inside); **`r9s18c − r9s17`** primary +0.013 [−0.054, +0.078], instruction margin +0.021 [−0.048, +0.094]
(both 0 inside), initiate −0.186 [−0.349, −0.026]; each seed − R5: instruction margin −0.102 [−0.158, −0.039] (17), −0.081 [−0.121, −0.034]
(18c), −0.234 [−0.285, −0.180] (19). **Continued − stopped (seed 18, step 233 − step 150)**: primary −0.013 [−0.044, +0.019], instruction
margin +0.009 [−0.024, +0.043], initiate −0.023 [−0.075, +0.000] — all three contain zero: the 83 extra steps changed nothing that this cell
measures. The `dev` cell (42 episodes, a replication; 19 of 21 origin groups are training material) repeats the pattern: instruction margin
+0.192 [+0.141, +0.246] (17), +0.254 [+0.185, +0.317] (18c), +0.116 [+0.067, +0.172] (19); initiate 0.267 / 0.033 / 0.417; `q_stop` 0 of 14 in
all three (R5 9 / 14); all 42 leave-one-out refits exclude zero for every seed; `r9s18c − r9s17` instruction margin +0.062 [+0.016, +0.106].

**Reading.** On the held-out cell **seeds 17 and 18 read instructions about equally** (+0.196 / +0.217, paired difference 0 inside) and
**both clearly better than seed 19** (+0.132 / +0.153 over it, both exclude zero): R8's seed 19 was the weak one of the three, not the recipe's
level. None of the three reaches R5 (seed 18, R5 recipe, +0.298). The gripper's "close now" ticks separate the seeds the other way on this
cell — seed 18 0.023, seed 17 0.209, seed 19 0.372 (R5 0.674) — and `q_stop` catches none of the ten stop onsets in any seed.

## 5. Closed loop, the registered choice and the verdict (Stage D)

Same serving path, harness, controller and expert as run reports 1–5 (`ModelPolicy`, `fused` + compile). `artifacts/reports/r9-closed-loop.json`
(`closed_loop.py report --merge ood_dev200=ood_dev,ood_dev_new,ood_dev_new2 --only ood_dev200,dev_new2 --seed-pairs <the registration's nine
pairs>`; label order expert · rule · mechanical · r8s19 · r9s18c · r9s17 · r6 · r5), records `artifacts/datasets/r9-closed-loop/r9s{18c,17}/`,
verdict `artifacts/reports/r9-verdict.json` (`closed_loop.py verdict --registration configs/eval/r9-registration.yaml`); the report → verdict
path had been run end to end on stand-in rows first (`artifacts/scratch/r9/dryrun/`: it reproduced R7's published `rule − r6` +0.115 [+0.035,
+0.195] and `r7 − r6` duplicates +2.290 [+0.545, +4.110] through the new code).

| policy | ood_dev 200: `done` / **strict** (false done) | dev_new2 100: `done` / **strict** (false done) | failures main / aux / geom (ood) | first grasp on a non-instructed object (ood · dev) | gripper transitions: reference / executed / missing / duplicate (ood) | `q_stop` onsets caught (ood · dev) | wrong-action / unsafe ticks per acted tick (ood) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| expert | 0.910 / **182** (0) | 0.900 / **90** (0) | 2 / 0 / 16 | 0 · — | 509 / 495 / 14 / 33 | 65/65 · 41/41 | — |
| rule judge | 0.725 / **145** (0) | 0.670 / **67** (0) | 2 / 51 / 2 | 0 · — | 395 / 384 / 11 / 759 | 31/43 · 25/30 | 0.006 / 0.001 |
| model R5 (seed 18, R5 recipe) | 0.810 / **132** (30) | 0.790 / **68** (11) | 17 / 13 / 8 | 1 · 0 | 514 / 404 / 110 / 64 | 21/36 · 13/31 | 0.041 / 0.007 |
| **seed 17 — `r9s17`** | **0.725 / 90 (55)** | **0.740 / 50 (24)** | 33 / 22 / 0 | 41 · 2 | 431 / 362 / 69 / 153 | **0/67 · 0/22** | 0.228 / 0.078 |
| **seed 18 — `r9s18c`** | **0.180 / 27 (9)** | **0.170 / 14 (3)** | 41 / **123** / 0 | 6 · 0 | 701 / 68 / **633** / 73 | **0/20 · 0/5** | 0.050 / 0.019 |
| **seed 19 — `r8s19` (R8)** | **0.420 / 28 (56)** | **0.510 / 26 (25)** | **91** / 25 / 0 | **139** · 33 | 336 / 188 / 148 / 91 | **0/67 · 0/31** | 0.565 / 0.177 |

(First-grasp counts: the first tick whose `commitment` is a `grasp` on an object other than that tick's instructed target,
`artifacts/scratch/r9/target_selection.py` → `target-selection.json`; it reproduces R8's published 139 / 200 for seed 19 and 1 for R5.)

**The registered choice (dev_new2 only).** Strict successes on dev_new2: `r9s17` **50**, `r8s19` 26, `r9s18c` 14 — all three runs completed
at `max_steps` 233 and ran the same 100 scenes (the code checked the shared seed count) — so **seed 17 (`r9s17`) was chosen, decided by the
first key (strict)**; no tie-break was needed. Beside it, not part of the choice: on dev_new2 `r9s17 − r8s19` strict +0.240 [+0.130, +0.350] and
`r9s18c − r9s17` −0.360 [−0.470, −0.260] (both exclude zero).

**The registered verdict (ood_dev 200, the chosen seed only):**

| condition | pair | difference [95 %] | holds if | holds? |
| --- | --- | --- | --- | --- |
| (a) strict success | `rule − r9s17` | **+0.275 [+0.205, +0.340]** | lower ≤ 0 | **no** |
| (b) false-done rate | `r9s17 − r5` | **+0.125 [+0.060, +0.190]** | upper < 0 | **no** |
| (c₁) duplicate gripper transitions per episode | `r9s17 − r5` | **+0.445 [+0.190, +0.730]** | lower ≤ 0 | **no** |
| (c₂) `q_stop` onsets caught | `r9s17 − r5` | **−0.583 [−0.750, −0.379]** | upper ≥ 0 | **no** |

Beside (a): discordant seeds 61 (rule only) vs 6 (seed 17 only), exact McNemar p = 1.5 × 10⁻¹², the lower bound above zero under all 200
alternative bootstrap RNG seeds. Beside (b): 36 vs 11, p = 3.5 × 10⁻⁴, the upper bound above zero under all 200. **All four conditions fail →
the registered call: "the chosen seed did not pass — the next round is training stability (the gripper head's flat phase, `q_stop`)".** The
cloud stays closed, and not on a knife-edge. On dev_new2 (the set the seed was chosen on, beside only) the chosen seed fails (a) +0.170
[+0.090, +0.250], (b) +0.130 [+0.060, +0.210] and (c₂) −0.419 [−0.625, −0.235] and holds (c₁) +0.160 [+0.000, +0.320].

**Where the chosen seed fails.** Seed 17 declares done as often as the rule judge (ood_dev 200 `done` 145 vs 145: `rule − r9s17` `done`
+0.000 [−0.080, +0.075]) but **55 of its 145 dones are false** (27.5 % of the episodes; R5 30, rule 0): in 49 of the 55 the last grasp
before the done tick was on an object other than the instructed one — it moves the wrong object into the zone and declares done (seed 19's
failure, milder: its first grasp is on a non-instructed object in 41 of 200 episodes against seed 19's 139). Its gripper works in the loop
(362 of 431 reference transitions executed; seed 18c 68 of 701) but fires extra transitions (153 duplicates; R5 64), and `q_stop` catches none of
the 67 stop onsets. Per seed against R5 (ood_dev 200): wrong-action ticks +0.187 [+0.136, +0.242], unsafe ticks +0.070 [+0.037, +0.110],
gripper-streak episodes +0.240 [+0.150, +0.325]; the aux-failure share +0.045 [−0.005, +0.095] contains zero.

**Distribution readout (three seeds of one recipe; descriptive, not used to choose or judge).**

| | seed 17 (`r9s17`) | seed 18 (`r9s18c`) | seed 19 (`r8s19`) |
| --- | ---: | ---: | ---: |
| dev_new2 strict / false done | **50** / 24 | 14 / 3 | 26 / 25 |
| ood_dev 200 strict / false done | **90** / 55 | 27 / 9 | 28 / 56 |
| ood_dev 200 duplicate gripper transitions / missing | 153 / 69 | 73 / **633** | 91 / 148 |
| `q_stop` onsets caught (ood · dev) | 0/67 · 0/22 | 0/20 · 0/5 | 0/67 · 0/31 |
| main failure class (ood) | wrong object → false done (main 33, aux 22) | gripper never closes (aux 123, main 41) | target selection (main 91, aux 25) |
| offline instruction-shuffle margin (decision cell) | +0.196 [+0.133, +0.253] | +0.217 [+0.148, +0.287] | +0.064 [+0.020, +0.104] |
| offline `q_gripper` initiate / settled | 0.209 / 0.994 | 0.023 / 0.929 | 0.372 / 0.994 |
| gripper fit onset (training batch, registered) | 139 | 152 (142 to step 150) | 132 |
| R8's step-150 statistic (log only) | 0.644 (would pass) | 0.723 (stopped in R8) | 0.592 (passed) |

Seed-paired differences on ood_dev 200 (registered bootstrap): strict `r9s17 − r8s19` **+0.310 [+0.230, +0.385]**, `r9s18c − r9s17` **−0.315
[−0.380, −0.255]**, `r9s18c − r8s19` −0.005 [−0.070, +0.060] (0 inside); false done `r9s18c − r8s19` −0.235 [−0.305, −0.165], `r9s18c − r9s17`
−0.230 [−0.295, −0.165], `r9s17 − r8s19` −0.005 [−0.085, +0.080] (0 inside); duplicates `r9s18c − r9s17` −0.400 [−0.740, −0.090], the other two
contain zero; `q_stop` catch 0 in every seed (all differences 0 [0, 0]). dev_new2 repeats every sign that is a finding except duplicates
(`r9s18c − r9s17` −0.160 [−0.390, +0.100]); there `r9s18c − r8s19` strict is −0.120 [−0.230, −0.020].

**Latency** (`latency.model_ms` excludes each episode's first tick): seed 17 non-first ticks p50 / p95 / p99 43.2–43.6 / 54.8–56.6 / 77.1–81.6 ms,
**0 of 24,505 over 100 ms**; first ticks p50 91.5–94.4 ms, **32 of 300** over 100 ms (max 298.6 ms); seed 18c non-first 43.6–44.4 / 56.4–57.7 /
81.5–84.2 ms, **0 of 27,767** over 100 ms, first ticks 35 of 300 over (max 294.8 ms). The gate (p95 ≤ 80 ms, > 100 ms ≤ 5 %) passes on either
reading.

## 6. Measured cost, the G1 recommendation and the seed distribution

| item | GPU-h | source |
| --- | ---: | --- |
| R2 … R8 (run reports 1–5) | ≈ 55.1 | `docs/reports/run-report-5.md` §6 |
| **R9** (unit wall clocks: B `r9-t1-s18c` 1 h 32 m 32 s · chain `r9-rest` 6 h 01 m 14 s = D `r9s18c` 1 h 04 m 47 s + C seed 17 3 h 55 m 27 s + D `r9s17` 1 h 01 m 00 s · full test suite `r9-pytest` 10 m 01 s = 27,827 s) | **7.73** | `journalctl --user`, `artifacts/scratch/r9/*.log` |
| total on the DGX Spark GB10 | **≈ 62.8** | — |

CPU this round: the stand-in dry run of the report → verdict path (23 min, plus a first attempt stopped after ≈ 22 min to protect the
resumed training's memory), the `r9s18c` preview report (10 min), the final strata, report and verdict (25 min), analyses in minutes. Cloud
spend: 0. Disk: `/` had 178 GB free at the start and **128 GB at the end** (two 26.35 GB checkpoints with their optimizer state, 0.5 GB of
loop records); nothing was deleted or slimmed, and R8's run directories are byte-identical.

**G1, decided by the rule registered before the numbers (§0; applied by `scripts/closed_loop.py verdict`, `artifacts/reports/r9-verdict.json`).**
The choice on dev_new2 took **seed 17** (strict 50 against 26 and 14). On ood_dev 200 it fails all four conditions: (a) `rule − r9s17` strict
**+0.275 [+0.205, +0.340]** (61 vs 6 discordant, p = 1.5 × 10⁻¹², lower bound > 0 under all 200 RNG seeds); (b) false-done rate **+0.125 [+0.060,
+0.190]** over R5; (c₁) duplicates **+0.445 [+0.190, +0.730]**; (c₂) `q_stop` **−0.583 [−0.750, −0.379]**. **The registered call is "fail": the
procedure (run several seeds, choose on the validation set) did not produce a seed that passes the holdout; the next round is training
stability — the gripper head's flat phase and `q_stop`.** The cloud stays closed, and not on a knife-edge. Provider, account and first paid
run remain the user's decisions in any case.

**The seed distribution, from the numbers (descriptive).**
1. **The choice did its job, and it was the whole story on the validation set only.** Seed 17 is the best of the three on both sets —
   dev_new2 strict 50 / 26 / 14 and ood_dev 200 90 / 28 / 27, `r9s17 − r8s19` +0.310 [+0.230, +0.385] and `r9s17 − r9s18c` +0.315 [+0.255,
   +0.380] on the holdout — so choosing on dev_new2 picked the seed that also ranks first on ood_dev 200. What it picked is still far from the
   bar: 90 strict against the rule judge's 145 and R5's 132, because 55 of its 145 dones are false (49 after carrying a non-instructed object).
2. **Each seed of this recipe fails in its own way.** Seed 19 picks the wrong target (first grasp on a non-instructed object in 139 of 200;
   offline instruction margin +0.064); seed 18, run to the end, reads instructions well (+0.217; first grasp wrong in 6 of 200) but its gripper
   never learned "close now" (offline initiate 1/43; 633 of 701 reference gripper transitions missing in the loop); seed 17 reads well (+0.196)
   and closes the gripper (362 of 431 transitions executed) but declares done after moving the wrong object and fires extra gripper
   transitions. Seeds 17 and 18 read instructions equally on the held-out cell (paired +0.021 [−0.048, +0.094]) and both better than seed 19
   (+0.132 / +0.153, both exclude zero) — R8's weak seed 19 was the tail of the distribution, not its centre. None reaches R5 (+0.298).
3. **Seed 18's boundary stop did not discard a seed that passes.** Run to step 233 on the same schedule it is statistically indistinguishable
   from its stopped state on the offline cell (primary −0.013 [−0.044, +0.019], margin +0.009 [−0.024, +0.043], initiate −0.023 [−0.075, +0.000])
   and in the loop it ties seed 19 on ood_dev 200 (27 vs 28 strict, −0.005 [−0.070, +0.060]) and is the weakest of the three on dev_new2 (14).
   The gripper fit that began at step 142 continued on the settled ticks, but "close now" never came.
4. **The gripper head's flat phase is the recipe's, not a seed's.** All three seeds sat at the constant-prior level for ≈ 100 steps and fit
   late (onsets 132 / 139 / 142–152); R8's step-150 statistic read 0.592 / 0.644 / 0.723 — the seed it stopped was the latest to fit, and
   whether a seed then learns "close now" (training-batch initiate 123/227 for seed 17, 34/151 and 19/184 for 19 and 18 over steps 151–233)
   is what separated the loops. The realized draws split the same way after the fit (an observation, not a finding — the material
   coin, episode order and candidate permutation are tied together within a seed): over steps 151–233 seed 17 drew 21 DAgger episodes, 8
   of them from the initiate-heavy dagger-0, against 14 (4) for seed 18 and 11 (3) for seed 19; by step 150 the three mixes were close
   (expert · DAgger 116 · 34 / 123 · 27 / 121 · 29).
5. **`q_stop` is a consistent failure of this recipe**: no true stop tick fired after step 1 in any of the three trainings, 0 of 10 onsets
   offline and 0 caught in the loop for every seed (67 / 20 / 67 onsets on ood_dev 200).

**Recommendation for the next Spark round (from these numbers):** training stability of the auxiliary heads before any cloud run — the
≈ 100-step flat phase of the gripper head (all three seeds) and "close now" (whether it is learned decides whether the loop can grasp),
`q_stop` (never learned beyond its prior), and the false dones after a wrong-object grasp (seed 17's 55, 49 of them after the last grasp was on a
non-instructed object). A seed-selection procedure is only worth scaling out once a seed can pass the holdout; three seeds showed that the
spread is wide (strict 27–90 of 200) but its best is below R5's single seed.
