# Run report 5 — auxiliary-head stability: R5's recipe with DAgger cycle-1 material at seeds 18 and 19, per-question losses while training (2026-09-29)

The document docs/06 Task 6 defines (:384) for the fifth and sixth trained checkpoints of this project (Task R8). Every number
is copied from an artifact named next to it. Korean prose lives in `docs/06`, `docs/08` and `HANDOFF.md`; this report is the
English ledger. Run reports 1–4 stay as they are; this one only adds. **Sections are appended as each stage's numbers land;
§0 was committed before any Stage C or D number existed.**

## 0. Pre-registration (written 2026-09-29 before any Stage C or D number; machine-readable copy `configs/eval/r8-registration.yaml`)

The rule below is applied by code, not by hand: `scripts/closed_loop.py verdict --report artifacts/reports/r8-closed-loop.json
--registration configs/eval/r8-registration.yaml --out artifacts/reports/r8-verdict.json` first reads each seed's head-fit
monitor from its run (`artifacts/runs/r8-t1-fp32-2b-s{18,19}/metrics.json`, `monitor_verdicts` — it refuses to judge if the
monitor the training applied differs from the registered one) and then runs `robo_jev.closed_loop.apply_registration` on the
report (R7's function, extended with seed groups, a stopped-seed rule and the stability readings; tests in
`tests/test_closed_loop.py` pin every bound, the per-seed grouping, the stopped seed and the monitor check).

**Why this round.** From R5 to R6 to R7 the main-decision head (instruction reading) was robust to recipe changes
(primary-stratum instruction-shuffle margin +0.298 / +0.374 / +0.370) but the auxiliary heads were not: the gripper head went
right (R5, settled 99.6 %) → hasty (R6, more duplicate transitions) → never fit during training (R7, settled 4.3 %, loop strict
0/200), and offline `q_stop` caught 5 → 0 → 0 of 10 stop onsets. Every comparison so far was **one seed**. The best closed
loop so far is r5 (ood_dev 200 strict 132, false done 30; rule judge 145, expert 182); r6 repaired the false dones (12) at
strict 122. This round runs **R5's recipe unchanged** plus the **cycle-1 DAgger material**, at **two seeds**, and logs the
per-question losses so the heads' fit is watched during training for the first time.

**Recipe** (`configs/train/qwen35-2b-r8.yaml`; a test pins that against R5's resolved config only the data list, the run name,
the kept-step list and the monitor differ, and against R7's only the mixture, `max_steps`, the name and the monitor). Qwen3.5-2B
bf16 with fp32 master weights, full text backbone + fp32 pointer readout rank 64, `backbone_lr` 1e-5, `readout_lr` 3e-4,
5-second TBPTT chunks, 30-tick window, tick weights 0.25 / 2 / 2 / 1, candidate permutation seeded by the run seed,
**material shares at the default 0.7 / 0.2 / 0.1** (the empty new-semantic bucket renormalizes them to 0.78 / 0.22 of the robot
units), **`max_steps` 233** (R5's schedule), `checkpoint_every` 50, no kept steps. Data: the g2 expert episodes (233 train),
DAgger-0 (200), DAgger-1 (200) and the done-gate collection (200) — the three DAgger sets as one `error_family` bucket — and the
non-robot split-by-split set. Expected draws: expert ≈ 181, DAgger ≈ 52 (of which cycle-1 ≈ 35). **Seeds 18 and 19**, run ids
`r8-t1-fp32-2b-s18` / `-s19`, closed-loop labels `r8s18` / `r8s19`. The instrument check: seed 18's step-1 loss must equal R5
seed 18's `2.633075326681137` (the logging does not change training).

**Head-fit monitor (soft stop).** At step 150, if the `q_gripper` per-question weighted loss averaged over steps 131–150 is
≥ **0.70 ×** its baseline averaged over the same steps, the run stops (status `stopped_head_not_fitting`), its metrics and
checkpoint are kept and its closed loop is skipped; **a stopped seed fails all of its cloud conditions**. The baseline is the
loss of a constant-prior head that outputs the step's label marginal (Task R8 A1; the step's robot batch is one episode).
**The constant was changed before the first training step, as the brief allows only after A3:** the brief's 0.95 → 0.70.
Why (A3, `artifacts/reports/r8-a3-teacher-forced.json`): r7 run teacher-forced on the ten g2 train episodes it drew last
(steps 286–304) has a `q_gripper` weighted loss of 0.3899 against a baseline of 0.5306 — **0.735 × baseline** — while its head
does not work on those very episodes (settled 13/336 = 0.039, initiate 0/11); the training computation and the C2 evaluation
computation agree to the argmax on all 1,018 ticks. 0.95 would read that head as fitting. r5 on the same ten episodes (which it
never drew) is at 0.067 × baseline (settled 336/336, initiate 9/11). 0.70 is the largest round constant below r7's measured
state and ten times r5's.

**Scenes.** Primary **`ood_dev200`** = R4's 26 ood_dev seeds + R6's 74 + R7's 100 (`r4-seeds.json` `ood_dev`, `r6-seeds.json`
`ood_dev_new`, `r7-seeds.json` `ood_dev_new2`), merged by `report --merge ood_dev200=ood_dev,ood_dev_new,ood_dev_new2`; 0 origin
groups shared with training. Secondary **`dev_new2`** 100 (R6), beside the verdict only.

**Rows.** expert · rule judge · mechanical and r5 · r6: the R4 / R6 / R7 runs (as in run report 4). r8s18 and r8s19: this
task, all four conditions (ood_dev, ood_dev_new, ood_dev_new2, dev_new2).

**Pairing and interval.** By seed (`profile:seed`), 95 % percentile bootstrap over seeds, 2,000 resamples, RNG seed 20260921
(`EPISODE_BOOTSTRAP`); `paired_success(strict=True)`, `paired_false_done`, `paired_seed_ratio`. Zero inside = not a finding.
Strict success is primary. Latency reports each episode's first tick separately.

**Cloud rule — R7's three conditions, for each seed s ∈ {r8s18, r8s19}, on `ood_dev200`:**
* **(a)** `rule − s` strict success: interval **lower bound ≤ 0**.
* **(b)** `s − r5` false-done rate: interval **upper bound < 0**.
* **(c)** `s − r5` duplicate gripper transitions per episode: interval **lower bound ≤ 0**; **and** `s − r5` `q_stop` catch rate:
  interval **upper bound ≥ 0**.

**The cloud is recommended only if both seeds pass all three** (all eight registered conditions hold). A seed stopped by the
monitor, or without a completed run, fails its four. Next to (a) and (b) for each seed: discordant seeds, exact McNemar p and the
bound distribution under bootstrap RNG seeds 1–200 — they do not change the verdict. If the cloud is recommended, the provider,
account and first paid run remain the user's decision.

**Stability reading (beside the verdict, not part of it).** `r8s18 − r8s19` paired strict success and false-done rate; the
seed-level auxiliary differences (duplicate transitions, gripper-streak episodes, `q_stop` catch rate, unsafe ticks, auxiliary
failures); both seeds' monitor results (the per-question loss curves against their baselines). An interval containing zero
means no difference between the seeds was found, not that they are equal. `dev_new2` is secondary evidence.

**Also reported, not part of the rule.** Stage C: realized units by material, the per-question loss curves against their
baselines (`q_gripper`, `q_stop`, `q_main`), the in-training gripper and stop probes, wall and peak memory. Stage D offline, per
seed, on the same cells (hashes `6a3b69131243`, `9b484441b23b`): the primary-stratum instruction-shuffle margin, `q_gripper`
settled / initiate / open, `q_stop` onsets, the `q_done` strata. Seed-paired `r8 − r5` closed-loop tables.

## 1. Trained weights (Stage C)

| run | steps | seed | checkpoint | contract digest | training | head-fit monitor (step 150, q_gripper, steps 131–150) |
| --- | ---: | ---: | --- | --- | --- | --- |
| **`r8-t1-fp32-2b-s19`** | **233 — completed** (the monitor let it continue) | 19 | `artifacts/runs/r8-t1-fp32-2b-s19/checkpoint.pt` (model + optimizer) | `93fe26725a4c…` | 4 h 03 m 54 s inside the chain unit `r8-rest` = **4.06 GPU-h** (train 13,884.8 s, 59.59 s/step, load 137.2 s), peak 56.01 GiB, loss 3.092 → 0.182 | weighted loss 0.3065 / baseline 0.5181 = **0.592 < 0.70 → fits** |
| **`r8-t1-fp32-2b-s18`** | **150 of 233 — stopped by the monitor** (`stopped_head_not_fitting`) | 18 | `artifacts/runs/r8-t1-fp32-2b-s18/checkpoint.pt` (step 150; model + optimizer — a resume unit, not a finished run) | `93fe26725a4c…` | unit wall 8,985 s = **2.50 GPU-h** (train 8,485.4 s, 56.57 s/step, load 137.3 s), peak 56.01 GiB, loss 2.633 → 0.526 | weighted loss 0.3682 / baseline 0.5094 = **0.723 ≥ 0.70 → not fitting**; closed loop skipped; **fails its four cloud conditions** |

Seed 18: **step-1 loss `2.633075326681137` — identical to R5 seed 18**, so the per-question logging did not change training (the
draws equal R5's for steps 1–6 and diverge at the first DAgger draw, step 7). Realized draws in its 150 steps: expert 123 (0.53
epoch), DAgger 27 (dagger-0 9 · dagger-1 9 · done-gate 9). The main-decision head fit early (`q_main` at 0.03–0.04 of its uniform
baseline from step ≈ 61); the gripper head said `open` on nearly every tick from step ≈ 21 to ≈ 140 (settled 0/621 in steps 81–100;
loss 1.0–1.2 × the constant-prior head's) — R7's collapse on R5's recipe and seed — and began to fit only at step ≈ 142 (settled
266/593 in steps 131–150, almost all of it in the last nine steps). `q_stop` stayed above its constant head throughout (ratio 1.3–5.8;
0 of 136 true ticks fired). Beside the rule, not changing it: two window steps were g2 episodes without any `closed` label (baseline
0); without them the statistic would be 0.674, and the brief's original 0.95 would have passed the run. Curves:
`artifacts/runs/r8-t1-fp32-2b-s18/metrics.json` (`steps[].loss_by_question`, `steps[].probes`).

Seed 19: realized draws expert **193** (0.83 epoch) and DAgger **40** (dagger-0 13 · dagger-1 17 · done-gate 10) — the seeded
material coin came out 1.8 SD below the expected 52 DAgger draws. Its gripper head sat at the constant-prior level and said `open`
almost everywhere until step ≈ 120 (settled 1/642 in steps 101–120) and then fit: window ratios 0.62 (121–140), **0.592** (131–150,
the monitor), 0.23 · 0.29 · 0.10 · 0.19 afterwards; in its last 23 steps the training batch's settled ticks were 613/613 right and
the "close now" (initiate) ticks 12/43. `q_stop` fired on 0 of 180 true ticks during training and ended near its constant head
(ratio 0.95). The run was launched by a chain on HEAD `65d38d5` with uncommitted edits outside `src/` only (`git.dirty: true`;
`git diff 65d38d5 -- src/` is empty). **Both seeds show the same shape — a flat gripper head for ≈ 100 steps, then a late fit
whose onset differs by ≈ 20 steps (seed 19 ≈ 120, seed 18 ≈ 142); the step-150 window falls on either side of it (0.592 / 0.723).**

## 2. Reproducible data

The training data list is R7's, unchanged (five manifests; test-pinned): robot train labels with gripper rule v2
(`artifacts/datasets/r1-robot/r1-rollout-labels-g2/manifest.json` `4d433bae6d22…`, 233 train episodes), DAgger-0
(`r5-dagger/dagger-0` `8b205ee13993…`), DAgger-1 (`r6-dagger/dagger-1` `6351ec247d35…`), the done-gate collection
(`r6-dagger/dagger-1-donegate` `6b44fc3e3492…`) — the three DAgger sets as one `error_family` bucket — and the non-robot
split-by-split set (`r1/single-by-split` `cfd632dd568f…`). Items loaded 3,453 = 833 stream + 2,620 single (both seeds). No
ood_dev-family record is in any training manifest; no sealed file was opened by training, evaluation or analysis code. The
closed-loop scenes are R7's (`ood_dev200` = `r4-seeds.json` 26 + `r6-seeds.json` 74 + `r7-seeds.json` 100; `dev_new2` 100).
**Checkpoint slimming (Task R8 A2, user-approved):** the nine finished-run files of R2–R7 now hold the model weights only
(26.35 → 3.77 GB each; `artifacts/reports/r8-a2-slim.json`); they evaluate and serve as before but cannot be resumed.

## 3. Resume verification

Nothing was resumed. Seed 18 stopped by the registered monitor at step 150 (its `checkpoint.pt` is a full resume unit at step
150 — the monitor is a stopping rule, not a crash); seed 19 completed in one process. The model-only files of A2 are refused by
the resume path with a named reason (tests at the loader and at `Trainer(resume=…)`).

## 4. Per-question quality (offline; the decision cell `ood_dev` 26 episodes / 3,162 ticks, hash `6a3b69131243`; the `dev` cell `9b484441b23b`)

Seed 18's rows are its **stopped state at step 150** (evaluated for the record; they do not enter the rule). R5–R7 rows are their
stored reports on the same cells. `artifacts/reports/r8-reeval-2b-t1-fp32-r8s{18,19}.json`, `r8-dev-2b-t1-fp32-r8s{18,19}.json`,
strata `r8-decision-cell-strata.json`, `r8-dev-cell-strata.json`.

| checkpoint (decision cell) | primary stratum (235) | **instruction-shuffle margin** [95 %] | `grasp` (97) | whole `q_main` | `q_gripper` initiate (43) / settled (1,015) / open (1,860) | `q_stop` onsets caught / 10 | goal-change immediate |
| --- | ---: | ---: | ---: | ---: | --- | ---: | ---: |
| **R8 seed 19 (233)** | 0.749 | **+0.064 [+0.020, +0.104]** | **0.577** | 0.981 | 0.372 / **0.994** / 0.962 | **0** | 0.667 |
| R8 seed 18 (stopped at 150) | 0.855 | +0.209 [+0.156, +0.258] | 0.835 | 0.986 | **0.047** / 0.886 / 0.936 | 0 | 0.822 |
| R7 | 0.881 | +0.370 [+0.241, +0.483] | 0.948 | 0.982 | 0.279 / 0.043 / 0.990 | 0 | 0.844 |
| R6 | 0.877 | +0.374 [+0.268, +0.465] | 0.928 | 0.990 | 0.837 / 0.897 / 0.916 | 0 | 0.956 |
| R5 | 0.902 | +0.298 [+0.232, +0.354] | 0.948 | 0.992 | 0.674 / 0.996 / 0.981 | 5 | 0.911 |
| **R8 s19 − R5** (paired by episode) | −0.153 [−0.206, −0.101] | −0.234 [−0.285, −0.180] | | | initiate `closed` −0.302 [−0.487, −0.140] | | |
| **R8 s18 − R5** | −0.047 [−0.084, −0.009] | −0.089 [−0.123, −0.055] | | | initiate `closed` −0.628 [−0.744, −0.512] | | |
| **R8 s18 − R8 s19** | +0.106 [+0.064, +0.153] | **+0.145 [+0.100, +0.192]** | | | initiate `closed` −0.326 [−0.444, −0.200] | | |

Every row's primary margin survives each of the 26 leave-one-episode-out refits. The `dev` cell (42 episodes, a replication, 19
of 21 origin groups are training material) repeats the ordering: primary instruction-shuffle margin R8 s19 +0.116 [+0.067,
+0.172] (`grasp` 0.716), R8 s18 +0.221 [+0.156, +0.285], R5 +0.268; `q_gripper` initiate / settled 0.417 / 0.998 (s19) and 0.000 /
0.882 (s18); `q_stop` 0 of 14 onsets for both seeds (R5 9 / 14). The `q_done` strata of seed 19 equal R5's (`post_release_other`
0.984); seed 18's stopped state is lower (0.742).

**Reading.** Seed 19, trained to the end, has the **weakest instruction reading of any checkpoint since R2's**: on the primary
stratum its margin over the instruction-shuffled control is +0.064 — still above zero, a quarter of R5's — and on the `grasp`
ticks it picks the instructed object 57.7 % of the time (R5 94.8 %). Its gripper head is R5-like on the settled ticks (0.994) and
weak on "close now" (0.372). Seed 18 stopped at step 150 already read instructions better (+0.209) than seed 19 did at 233, while
its gripper head had not yet learned "close now" (0.047). `q_stop` catches none of the stop onsets in either seed.

## 5. Closed-loop success (seed 19 only — seed 18's loop was skipped by the registered rule)

Same serving path, harness, controller and expert as run reports 1–4. `artifacts/reports/r8-closed-loop.json` (`closed_loop.py
report --merge ood_dev200=ood_dev,ood_dev_new,ood_dev_new2 --only ood_dev200,dev_new2 --seed-pairs r8s19:r5 rule:r8s19`),
records `artifacts/datasets/r8-closed-loop/r8s19/`.

| policy | ood_dev 200: `done` / **strict** (false done) | dev_new2 100: `done` / **strict** (false done) | failures main / aux / geom (ood) | gripper duplicates (ood · dev) | `q_stop` onsets caught (ood · dev) | wrong-action / unsafe ticks per acted tick (ood) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| expert | 0.910 / **182** (0) | 0.900 / **90** (0) | 2 / 0 / 16 | 33 · 15 | 65/65 · 41/41 | — |
| rule judge | 0.725 / **145** (0) | 0.670 / **67** (0) | 2 / 51 / 2 | 759 · 491 | 31/43 · 25/30 | 0.006 / 0.001 |
| model R5 | 0.810 / **132** (30) | 0.790 / **68** (11) | 17 / 13 / 8 | 64 · 27 | 21/36 · 13/31 | 0.041 / 0.007 |
| model R6 | 0.670 / **122** (12) | 0.720 / **69** (3) | 28 / 38 / 0 | 439 · 190 | 3/92 · 0/59 | 0.074 / 0.041 |
| **model R8 seed 19** | **0.420 / 28 (56)** | **0.510 / 26 (25)** | **91** / 25 / 0 | 91 · 46 | **0/67 · 0/31** | **0.565 / 0.177** |
| model R8 seed 18 | — (stopped at step 150; no loop) | — | — | — | — | — |

Paired by seed on ood_dev 200 (registered bootstrap): **rule − R8s19 strict +0.585 [+0.510, +0.660]** (discordant 122 vs 5, exact
McNemar p < 0.001, lower bound > 0 under all 200 RNG seeds); **R8s19 − R5 false-done rate +0.130 [+0.050, +0.205]** (46 vs 20, p =
0.002); R8s19 − R5 strict −0.520 [−0.600, −0.440]; per seed, R8s19 − R5: duplicate transitions +0.135 [−0.140, +0.570] (0 inside),
`q_stop` onsets caught −0.583 [−0.750, −0.379], wrong-action ticks +0.524 [+0.458, +0.592], unsafe ticks +0.169 [+0.118, +0.220],
main-decision failures +0.370 [+0.295, +0.445], gripper-streak episodes +0.485 [+0.395, +0.570]. dev_new2 repeats every sign
(rule − R8s19 strict +0.410 [+0.310, +0.510]; false done +0.140 [+0.040, +0.240]).

**The loop failure is the main decision, and it is target selection.** R8s19's **first grasp commitment is on an object other than
the instructed target in 139 of the 200 ood_dev episodes** (R5: 1 of 200); 59.8 % of its adopted `grasp` ticks name a non-target
object (R5 4.1 %), and 48 of its 56 false dones followed a non-target grasp (it moves the wrong object into the zone and declares
done). Example `ep-E0-1050128-r8-r8s19` (E0, one fixed instruction "put the gray cylinder in the left zone; do not go near the
red box, do not touch the green box"): the model commits at tick 0 to `grasp:o1:top:zoneL` — the wrong, fragile object — with
probability 1.000 and keeps it until the stall watchdog ends the episode. This is the offline `grasp` weakness (57.7 %) compounded by
commitment inertia in the loop. Latency (non-first ticks): p50 / p95 / p99 43.2–43.7 / 55.4–56.5 / 79.0–82.1 ms, 1 of 21,892
over 100 ms; first ticks p50 91.5–94.6 ms, 40 of 300 over 100 ms (max 303 ms) — overall 0.18 %; the gate passes.
