# Run report 3 — DAgger cycle 1 and the `q_done` instrument: the false done, the states after it, and the closed loop again (2026-09-27)

The document docs/06 Task 6 defines (:384) — *trained weights, reproducible data, resume verification, per-question
quality and measured cost, presented together* — for the third trained checkpoint of this project, the one that answers
run report 2's G1 recommendation ("one more Spark round first: DAgger cycle 1 on the R5 records and a check of the
`q_done` label/gate, then open the cloud once the false-done rate and the aux streaks move"). Every number is copied
from an artifact named next to it. Korean prose lives in `docs/06`, `docs/08` and `HANDOFF.md`; this report is the
English ledger. Run reports 1 and 2 stay as they are; this one only adds.

## 1. Trained weights

| run | steps | seed | checkpoint | contract digest | training | evaluated by |
| --- | ---: | ---: | --- | --- | --- | --- |
| **`r6-t1-fp32-2b-s18`** | 233 = one schedule, one process (no resume) | 18 | `artifacts/runs/r6-t1-fp32-2b-s18/checkpoint.pt` (24.5 GiB; model + optimizer + fp32 masters) | `93fe26725a4c…` (unchanged) | 14,130.1 s of unit wall = **3.93 GPU-h** (train seconds 13,413.1; 57.57 s/step), peak 56.31 GiB, loss 2.633 → 0.140 | R6 C2 (cells), **R6 D2 (closed loop)** |
| `r5-t1-fp32-2b-s18` (the baseline) | 233 | 18 | `artifacts/runs/r5-t1-fp32-2b-s18/checkpoint.pt` | same | 4.01 GPU-h (run report 2) | R6 B1 (cell with `q_done`), **R6 D2 (new seeds)** |

Recipe: R5's (`configs/train/qwen35-2b-r6.yaml` extends `qwen35-2b-r5.yaml`) — Qwen3.5-2B bf16 with fp32 master
weights, fp32 pointer readout rank 64, full text backbone, `backbone_lr` 1e-5, `readout_lr` 3e-4, 5-second TBPTT chunks,
30-tick window, tick weights 0.25 / 2 / 2 / 1, seed 18, `max_steps` 233. **Two things differ, and a test pins that it is
exactly these two**: the data list (five manifests, below) and `sampler.material_shares` existing **0.5** / error family
**0.5** / new semantic family 0.0 (R5 used the default 0.7 / 0.2 / 0.1, renormalised to 0.78 / 0.22). Realized robot
draws (`metrics.json` `summary.sampler.units`): **expert 111 episodes (0.48 epoch of 233) and DAgger 122 (0.20 epoch of
600: dagger-0 39, dagger-1 37, done-gate 46)**; only 2 of dagger-1's 20 false-done episodes and 7 of the done-gate set's
21 episodes with raw false-done events were drawn. Non-robot 1,376 bundles, tokens 11.55 M.

## 2. Reproducible data

| dataset | what | manifest (sha256 of the manifest file) | version |
| --- | --- | --- | --- |
| robot train labels, gripper rule v2 (unchanged from R5) | the 400 R1 v0.2 episodes, `q_gripper` under rule v2; train 233 episodes (the sealed split is copied as data and never read) | `artifacts/datasets/r1-robot/r1-rollout-labels-g2/manifest.json` `4d433bae6d22…` | `labels-rollout-v1+gripper-v2` |
| DAgger cycle 0 (unchanged from R5) | R4's dev-condition model-driven records, 200 episodes, 14,597 ticks | `artifacts/datasets/r5-dagger/dagger-0/manifest.json` `8b205ee13993…` | `gripper-v2`, `dagger-v0.1` cycle 0 |
| **DAgger cycle 1 — model loops** | R5's dev-condition model-driven records (R5 dev_new 100 + R4 dev seeds 100), relabelled with the same builder; 18,384 ticks, 34 families, 155 `done` of which **20 false dones**; the model's raw `q_done` ≥ 0.5 on 87 reference-False ticks (22 episodes); `provenance.dagger.collection = model_loop` | `artifacts/datasets/r6-dagger/dagger-1/manifest.json` `6351ec247d35…` (sources `artifacts/datasets/r5-closed-loop/r5/{dev_new,dev}`) | `gripper-v2`, `dagger-v0.1` cycle 1 |
| **DAgger cycle 1 — done-gate collection** | 200 new train-family episodes (E1 100 + E2 100, seeds 600100+) driven by the R5 model with the harness receiving the **expert's** `q_done` (collection-only `DoneGatePolicy`, never an evaluation); 20,191 ticks, 117 families; the model's raw `q_done` false-positive on 143 ticks in 21 episodes, 12 of them runs long enough to have ended the episode under the model's own gate; `provenance.dagger.collection = expert_done_gate` | `artifacts/datasets/r6-dagger/dagger-1-donegate/manifest.json` `6b44fc3e3492…` (raw records `artifacts/scratch/r6/donegate-raw/train`, run `artifacts/reports/r6-donegate-run.json`) | `gripper-v2`, `dagger-v0.1` cycle 1 |
| non-robot single requests (unchanged) | 2,000 records | `artifacts/datasets/r1/single/manifest.json` `46e7e361719b…` | R1 |
| closed-loop scenes (R6) | `dev_new2` 100 new dev seeds and `ood_dev_new` 74 new ood_dev seeds (base 980100; E0/E1/E2 = 20/40/40); `ood_dev 100` = R4's 26 ood_dev seeds + the 74 | `artifacts/reports/r6-seeds.json`, `artifacts/reports/r4-seeds.json`; overlaps `artifacts/reports/r6-seeds-overlap.json`, `r6-r4-seeds-overlap.json` | `cl0.1`, ids `-r6-<label>` |

QA: `artifacts/reports/r6-dagger-dagger-1-qa.json` and `r6-dagger-dagger-1-donegate-qa.json` — 0 contract violations,
0 QA violations, 0 leaks. Both cycle-1 datasets are `split: train` and `material: error_family` (docs/04 §7's new-error
axis; the contract's `SPLITS` has no `dagger` and `contracts.py` is a digest piece). No ood_dev-family record is in any
training manifest: the done-gate collection refuses any scene whose split is not `train`, and the ood_dev 100 scenes
share **0** origin groups with the four training manifests (dev_new2 shares 27 of its 31 — unseen seeds of mostly
trained-on families).

## 3. Resume verification

Nothing to resume this time: the run completed in one process (`summary.rescheduled` None, status `completed`, step 233)
with periodic atomic saves at steps 50 / 100 / 150 / 200 (10:08:46, 11:00:46, 11:52:20, 12:42:50) and the final at
13:15:18 — four periodic saves and the final one were written with ≥ 64 GB free between saves, which is what the brief's disk gate
(≥ 65 GB before launch; one 26.35 GB checkpoint next to the previous one during a save) was for. The first launch was
refused by the memory fence before any allocation (`require_free` 60 GiB read 44.2 GiB free: on the GB10 the page cache
counts against `mem_get_info`); evicting the clean page cache of the artifact files (`posix_fadvise(DONTNEED)`, no file
changed) raised MemFree from 44 to 101 GiB and the second launch ran. The license for the T1 resume path is unchanged (R2 A1's gate, R3a
A2's registered spread; R5's first long GPU walk of the plain resume).

## 4. Per-question quality (offline — the decision cell is `ood_dev`, 26 episodes / 3,162 ticks, hash `6a3b69131243`, R2/R3a/R5's ruler)

| checkpoint | primary stratum (label ≠ commitment, 235) | **instruction-shuffle margin** (paired 95 %) | `grasp` (97) | whole `q_main` | `q_gripper` initiate (43, v2): `closed` | `q_gripper` on single-`open` ticks: accuracy | `q_gripper` whole (parent) | `q_done` `post_release_other` (62) | `q_stop` onsets caught / 10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **R6** | 0.877 [0.842, 0.909] | **+0.374 [+0.268, +0.465]** | 0.928 | 0.990 | 0.837 [0.711, 0.947] | **0.916** | **0.905** | 0.952 | **0** |
| R5 (B1, same suite) | 0.902 [0.874, 0.933] | +0.298 [+0.232, +0.354] | 0.948 | 0.992 | 0.674 [0.565, 0.789] | 0.981 | 0.980 | 0.984 | 5 |
| R6 − R5 (paired by episode) | −0.026 [−0.060, 0.000] | **+0.077 [−0.017, +0.183]** | | | +0.163 [+0.049, +0.270] | | | −0.032 [−0.079, 0.000] | |
| rule judge (ceiling for goal reading) | 0.477 | — | 0.959 | 0.812 | 0.977 | — | — | 1.000 | 9 |
| mechanical baseline (floor) | 0.077 | — | 0.000 | 0.931 | — | — | — | — | — |

Reading (R6 C2). The instruction-shuffle margin on the primary stratum **excludes zero and does not regress** against R5
(all 26 × 3 leave-one-episode-out refits exclude zero; the paired difference contains zero). Two judgments moved the
wrong way, and they are the loop's regressions seen offline: the gripper head now says `closed` on 8.4 % of single-`open`
ticks (R5 1.9 %) — the initiate stratum rose, but its shuffled controls answer `closed` even more often (0.884 / 0.884
/ 0.930), so it is still read from execution state — and the stop head is silent (0 of 10 onsets, R5 5 of 10). The new
`q_done` strata (docs/08 §7) see almost nothing on this cell: its expert-driven episodes hold only 35 ticks (5 episodes)
of the loop's false-done state, and both checkpoints answer them correctly — the closed loop is the instrument for that
judgment. The non-selecting `dev` cell (hash `9b484441b23b`; 19 of its 21 origin groups are now training material)
replicates the sign — +0.221 [+0.157, +0.282] — with a small paired regression against R5 (−0.047 [−0.085, −0.010]) and
the same gripper and stop shifts (`open` accuracy 0.895 vs 0.974; `q_stop` 0 of 14 onsets vs 9 of 14). Sources:
`artifacts/reports/r6-reeval-2b-t1-fp32-{r5,r6}.json`, `r6-dev-2b-t1-fp32-r6.json`, `r6-decision-cell-strata.json`,
`r6-dev-cell-strata.json`.

## 5. Closed-loop success (R6 D — the third time the model moves the robot)

Setup as in run reports 1–2 (the same `ModelPolicy` serving path, `h0.9` / `c0.6` / `ts0.6`, expert reference labels) on
two scene sets: **ood_dev 100** — R4's 26 ood_dev seeds plus 74 new ones from base 980100, sharing **no** origin group with
any training manifest (the only family holdout, and the E1 ruler) — and **dev_new2 100**, new dev seeds whose origin groups
are mostly trained-on (27 of 31; secondary evidence). R5's and R4's dev scenes are training material now and were not run.

| policy | ood_dev 100: `done` / **strict** (false done) | dev_new2 100: `done` / **strict** (false done) | gripper transitions executed / reference, duplicates (ood · dev) | `q_stop` onsets caught (ood · dev) | unsafe (ood · dev) |
| --- | ---: | ---: | ---: | ---: | ---: |
| expert | 0.950 / **95** (0) | 0.900 / **90** (0) | 254/257, 22 · 272/283, 15 | 37/37 · 41/41 | 0 · 0 |
| rule judge | 0.720 / **72** (0) | 0.670 / **67** (0) | 185/189, 191 · 195/219, 491 | 12/18 · 25/30 | 0.05 % · 0.01 % |
| mechanical | 0 / 0 | 0 / 0 | — | — | — |
| model R5 (labels v2 + DAgger-0) | 0.790 / **63** (**16**) | 0.790 / **68** (**11**) | 202/259, 36 · 208/229, 27 | 12/20 · 13/31 | 1.07 % · 0.21 % |
| **model R6 (+ DAgger-1 + done gate, shares 0.5/0.5)** | 0.630 / **60** (**3**) | 0.720 / **69** (**3**) | 321/388, **175** · 370/454, **190** | **3/47 · 0/59** | **2.70 % · 1.32 %** |

Paired by seed on ood_dev 100: **rule − R6 strict +0.120 [+0.010, +0.230]**; **R6 − R5 false-done rate −0.130 [−0.210,
−0.050]**; R6 − R5 strict −0.030 [−0.140, +0.080]; R6 − R5 `done` −0.160 [−0.260, −0.060]. On dev_new2: rule − R6 strict
−0.020 [−0.140, +0.090]; false done −0.080 [−0.150, −0.010]; R6 − R5 strict +0.010 [−0.100, +0.120]. Seed by seed, R6 turns
16 of R5's 27 false dones into strict successes and loses 32 of R5's 131 strict successes (20 to auxiliary failures, 8 to
main-decision failures, 4 to new false dones). Failure classes, ood_dev 100: R6 main 15 / aux 22 / geometric 0 (R5 10 / 7 /
4). Stability: round trips 54 / 38 (R5 22 / 8), goal-change immediate reaction 0.839 / 0.938 (R5 0.818 / 0.915). Latency in
the loop, non-first ticks: p50 / p95 / p99 44.1–45.1 / 56.7–57.6 / 81.0–82.7 ms, **0 of 25,066 over 100 ms**; first ticks
(prefix build) p50 ≈ 93 ms, max 279 ms, 25 of 200 over 100 ms; overall 0.10 % — the gate passes. Sources:
`artifacts/reports/r6-closed-loop.json`, `r6-run-*.json`, `artifacts/datasets/r6-closed-loop/`,
`.superpowers/sdd/task-r6-report.md` D2.

## 6. Measured cost and the G1 recommendation

| item | GPU-h | source |
| --- | ---: | --- |
| R2 + R3a + R4 (run report 1) | 28.56 | `docs/reports/run-report-1.md` §6 |
| R5 (run report 2) | 5.29 | `docs/reports/run-report-2.md` §6 |
| **R6** (unit wall clocks: done-gate collection 25 m 50 s incl. smoke · B1 + r5 loops 32 m 32 s · training 3 h 55 m 33 s (+ 1 s refused) · r6 loops + two cells 1 h 01 m 08 s = 21,304 s) | **5.92** | `journalctl --user`, `artifacts/scratch/r6/*.log` |
| total on the DGX Spark GB10 | **≈ 39.8** | — |

CPU this round is minutes. Cloud spend: 0. The disk was again the binding resource: the training could not start until
the Qwen3.5-9B and Qwen3.8-27B weights were deleted with the user's approval (re-fetchable from their manifest entries).

**G1, decided by the rule registered before the numbers.** Open the cloud only if, on ood_dev 100, (a) the rule judge is
not strictly ahead of R6 (paired strict interval contains zero or is negative) **and** (b) R6's false-done rate is below
R5's (paired interval excluding zero). (b) holds: **−0.130 [−0.210, −0.050]** (16 → 3). (a) does not: **rule − R6 strict
+0.120 [+0.010, +0.230]** (72 vs 60). **The cloud stays closed.** The bottleneck, named by the numbers: the false done is
fixed and instruction reading holds, but the same run lost 32 of R5's 131 strict successes to a regression of the other
judgments — gripper duplicate transitions 36 → 175, `q_stop` 12 of 20 → 3 of 47 onsets (offline 5 → 0 of 10), unsafe
actions 1.07 % → 2.70 %, wrong-action ticks 390 → 547 on ood_dev 100. The recipe's one change is the most plausible cause
and remains a hypothesis at one seed: the 0.5 share cut the expert exposure from 0.78 to 0.48 epoch and spread 122 DAgger
draws over 600 episodes, of which only 9 of the 41 carrying a false-done state were drawn. The next Spark round should keep
the expert exposure at least at R5's level (R5's shares with the cycle-1 material in the pool, or more steps), put the
false-done material in a small weighted bucket instead of a uniform pool, and treat `q_stop` and the gripper duplicates
as regression gates next to the false-done rate. R3a's seed spread (+0.111 … +0.247 on the primary margin) is the noise a
single run carries.
