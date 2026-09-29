# Run report 4 — R5's expert exposure with R6's DAgger exposure: was the auxiliary regression the mixture ratio, and does the model pass the rule judge? (2026-09-29)

The document docs/06 Task 6 defines (:384) for the fourth trained checkpoint of this project (Task R7). Every number is
copied from an artifact named next to it. Korean prose lives in `docs/06`, `docs/08` and `HANDOFF.md`; this report is the
English ledger. Run reports 1–3 stay as they are; this one only adds. **Sections are appended as each stage's numbers
land; §0 was committed before any Stage C or D number existed.**

## 0. Pre-registration (written 2026-09-29 07:15 KST, before any Stage C or D number; machine-readable copy `configs/eval/r7-registration.yaml`)

The rule below is applied by code, not by hand: `scripts/closed_loop.py verdict --report artifacts/reports/r7-closed-loop.json
--registration configs/eval/r7-registration.yaml --out artifacts/reports/r7-verdict.json` runs
`robo_jev.closed_loop.apply_registration`, which reads exactly the pairs, metrics and bounds registered here (tests in
`tests/test_closed_loop.py` pin every bound at its edge, the flipped-pair reading and the refusal of a report whose
bootstrap is not the registered one).

**Why this round.** R6 cut the false-done rate (ood_dev 100: 16 → 3) but left strict success where it was and regressed
three auxiliary judgments with seed-paired intervals excluding zero (gripper disagreement streaks, duplicate gripper
transitions, `q_stop` onsets caught). R6 turned one recipe knob, the material shares 0.5 / 0.5, which cut the expert
exposure from 182 draws (0.78 epoch) to 111 (0.48 epoch). Hypothesis (one run, one seed): less expert exposure weakened
the auxiliary judgments and `q_stop`. Alternative: the DAgger material's auxiliary labels (conditioned on the model's own
commitments) made the gripper hasty. **R7 separates the two**: shares existing 0.6 / error family 0.4 and `max_steps` 304,
so the expected draws are expert ≈ 182 (R5's level) and DAgger ≈ 122 (R6's level). `r7 − r6` then holds the DAgger
exposure fixed and adds ≈ 71 expert draws.

**Scenes.**
* **Primary: `ood_dev200`** = R4's 26 ood_dev seeds (`artifacts/reports/r4-seeds.json`, condition `ood_dev`) + R6's 74 new
  ood_dev seeds (`artifacts/reports/r6-seeds.json`, `ood_dev_new`) + this task's 100 new ood_dev seeds
  (`artifacts/reports/r7-seeds.json`, `ood_dev_new2`, seed base 1050100). One table via `closed_loop.py report --merge
  ood_dev200=ood_dev,ood_dev_new,ood_dev_new2` (a seed present in two parts is refused). No origin group of these scenes is
  in any training manifest.
* **Secondary: `dev_new2`** (R6's 100 new dev seeds) — written next to the verdict, never used to make it.

**Rows.** expert · rule judge · mechanical: R4's records on the 26, R6's on `ood_dev_new` and `dev_new2`, this task's (B2) on
`ood_dev_new2`. r5: R5's records on the 26, R6's on `ood_dev_new` and `dev_new2`, this task's (B3) on `ood_dev_new2`. r6: R6's
records on the 26, `ood_dev_new` and `dev_new2`, this task's (B3) on `ood_dev_new2`. r7: this task, all of them.

**Metrics (per seed, pooled over seeds).** strict success = `done ∧ target_inside_zone`; false-done rate = seeds with `done ∧
¬target_inside_zone` over **all** seeds; duplicate gripper transitions per episode (`seed_metrics` `gripper_duplicates`);
`q_stop` catch rate = onsets caught / onsets (`q_stop_caught`); gripper-streak episodes = episodes with a ≥ 3-tick gripper
disagreement streak (`gripper_streak`).

**Intervals.** Paired by seed (`profile:seed`), 95 % percentile bootstrap over seeds, **2,000 resamples, RNG seed 20260921**
(`robo_jev.evaluate.EPISODE_BOOTSTRAP`) — `paired_success(strict=True)`, `paired_false_done`, `paired_seed_ratio`. An interval
that contains zero is not a finding.

**Cloud rule — recommend the cloud (a seed set of the R7 recipe) only if all three hold on `ood_dev200`:**
* **(a)** `rule − r7` strict success: interval **lower bound ≤ 0**.
* **(b)** `r7 − r5` false-done rate: interval **upper bound < 0** (R6's repair holds).
* **(c) regression gate**: `r7 − r5` duplicate gripper transitions per episode: the interval is **not entirely above zero**
  (lower bound ≤ 0); **and** `r7 − r5` `q_stop` catch rate: the interval is **not entirely below zero** (upper bound ≥ 0).

Next to (a) and (b), not changing the verdict: discordant seeds, the exact two-sided McNemar p, and the distribution of the
lower/upper bound under bootstrap RNG seeds 1–200 (`closed_loop.paired_robustness`). If the cloud is recommended, the
provider, account and first paid run remain the user's decision. If not, the failing condition(s) are named with numbers.
One seed: R3a's seed spread on the primary instruction margin (+0.111 … +0.247) is the noise a single run carries.

**Cause call (registered) — `r7 − r6` on `ood_dev200`, three per-seed metrics:** duplicate gripper transitions per episode
and gripper-streak episodes count as *toward r7* when the interval's **upper bound < 0**; the `q_stop` catch rate when its
**lower bound > 0**. All three toward r7 → write "**R6의 퇴행은 expert 노출 비율 때문**" (R6's regression was the
expert-exposure ratio). None → write "**노출 비율로 설명되지 않음 — DAgger 부가 라벨이 다음 표적**" (not explained by the
exposure ratio — the DAgger auxiliary labels are the next target). Mixed → the call is written per judgment (each metric
toward r7 "explained by the exposure ratio", each other one "not explained — DAgger auxiliary labels next target") and the
overall call is "mixed". An interval excluding zero **in r6's favour** counts as not explained and is named a new
regression of r7. The same values on `dev_new2` are written beside it.

**Also reported, not part of the rule.** Stage C2 (offline, same cell hash `6a3b69131243`): the primary-stratum
instruction-shuffle margin (does it exclude zero; paired against r6 +0.374 and r5 +0.298), the `q_done` strata, the
`q_gripper` initiate and `open` ticks, the `q_stop` onsets; the `dev` cell as a family-overlap caveat. Latency reports the
first tick of each episode separately.

## 1. Trained weights

| run | steps | seed | checkpoint | contract digest | training | evaluated by |
| --- | ---: | ---: | --- | --- | --- | --- |
| **`r7-t1-fp32-2b-s18`** | 304 = one schedule, one process (no resume) | 18 | `artifacts/runs/r7-t1-fp32-2b-s18/checkpoint.pt` (24.5 GiB; model + optimizer + fp32 masters) | `93fe26725a4c…` (unchanged) | 18,361.1 s of unit wall = **5.10 GPU-h** (train seconds 17,384.0; 57.18 s/step), peak 56.05 GiB, loss 2.633 → 0.128 | R7 C2 (cells), **R7 D (closed loop)** |
| `r6-t1-fp32-2b-s18`, `r5-t1-fp32-2b-s18` (baselines) | 233 each | 18 | `artifacts/runs/r{6,5}-t1-fp32-2b-s18/checkpoint.pt` | same | run reports 3 and 2 | R7 B3 (the 100 new ood_dev seeds) |

Recipe: R6's (`configs/train/qwen35-2b-r7.yaml` extends `qwen35-2b-r6.yaml`) — Qwen3.5-2B bf16 with fp32 master weights, fp32
pointer readout rank 64, full text backbone, `backbone_lr` 1e-5, `readout_lr` 3e-4, 5-second TBPTT chunks, 30-tick window,
tick weights 0.25 / 2 / 2 / 1, seed 18, the same five data manifests (the non-robot one in its split-by-split layout). **Two
recipe numbers differ, and a test pins that it is exactly these two** (plus the manifest path and the run name):
`sampler.material_shares` existing **0.6** / error family **0.4** / new semantic family 0.0 (R6 0.5 / 0.5) and `max_steps`
**304** (R6 233). Designed draws: expert 0.6 × 304 ≈ 182 (R5's level), DAgger 0.4 × 304 ≈ 122 (R6's level). **Realized**
(`metrics.json` `summary.sampler.units` and `steps[].units`): **expert 195 (0.84 epoch of 233) and DAgger 109** (dagger-0 42,
dagger-1 33, done-gate 34) — the seeded material coin came out 1.5 SD from its mean, so against R6 this run has **+84 expert
draws and −13 DAgger draws** rather than "+71 and the same". Of the 43 episodes whose model raw `q_done` rose on a
reference-False tick, **10** were drawn (3 + 7; R6 9). Non-robot 1,818 bundles, tokens 14.99 M (R6 11.55 M: 71 more steps).

## 2. Reproducible data

The training data list is R6's, with one change of **layout, not content**: the non-robot set is the split-by-split
re-layout of A2. The robot manifests are unchanged.

| dataset | what | manifest (sha256 of the manifest file) | version |
| --- | --- | --- | --- |
| robot train labels, gripper rule v2 (unchanged since R5) | the 400 R1 v0.2 episodes, `q_gripper` under rule v2; train 233 episodes | `artifacts/datasets/r1-robot/r1-rollout-labels-g2/manifest.json` `4d433bae6d22…` | `labels-rollout-v1+gripper-v2` |
| DAgger cycle 0 (unchanged since R5) | R4's dev-condition model-driven records, 200 episodes | `artifacts/datasets/r5-dagger/dagger-0/manifest.json` `8b205ee13993…` | `gripper-v2`, `dagger-v0.1` cycle 0 |
| DAgger cycle 1 — model loops (unchanged since R6) | R5's dev-condition model-driven records, 200 episodes | `artifacts/datasets/r6-dagger/dagger-1/manifest.json` `6351ec247d35…` | cycle 1 |
| DAgger cycle 1 — done-gate collection (unchanged since R6) | 200 train-family episodes with the expert's `q_done` at the harness | `artifacts/datasets/r6-dagger/dagger-1-donegate/manifest.json` `6b44fc3e3492…` | cycle 1 |
| **non-robot single requests, split by split (A2)** | the same 4,200 states as `r1/single` (`46e7e361719b…`; the run reports 1–3 wrote "2,000 records" — that is D1's pilot set; this one has 4,200 states, 2,620 of them `train`), regenerated with the same generator, config and seed and written one file per split; the one-file sha256 **`2ed780fd…` equals the old manifest's**, so the records are byte-identical and the old file was not opened | `artifacts/datasets/r1/single-by-split/manifest.json` `cfd632dd568f…` | `gen-single-v0.3.0`, `layout: by-split` |
| closed-loop scenes (R7) | `ood_dev_new2` 100 new ood_dev seeds (base 1050100; E0/E1/E2 20/40/40; 41 origin groups, **0** shared with training); **ood_dev 200** = R4's 26 + R6's 74 + these 100 | `artifacts/reports/r7-seeds.json` (+ `r4-seeds.json`, `r6-seeds.json`); overlaps `r7-seeds-overlap.json`, `r7-seeds-crosscheck.json` | `cl0.1`, ids `-r7-<label>` |

**Seal hygiene (Task R7 A1–A3).** From R2 to R6 the training loader hashed and parsed every file a manifest listed and only
then dropped the records outside the requested split — every training run on the R1 robot corpora (33 sealed episode
files) and on the non-robot `r1/single` (all splits in one `records.jsonl`) opened and parsed sealed records, and so did the
evaluation-cell loader. The impact was nil: in the old code the split test came right after the parse and before tagging,
serialization, `Item` creation and indexing, so a dropped record had no tokens and no index; R6 loaded exactly the
train-split counts (3,453 = 833 + 2,620), and the new loader — which is tested never to open a sealed file — reproduces
the decision cell's identity hash `6a3b69131243`. The loader now decides which files to open from the manifest before
opening any, refuses entries that do not say their split, and cannot load the sealed split; the evaluation and analysis
readers of episode directories follow the same rule (`robo_jev.data.sealed`). An old run cannot be resumed with the new
loader: its non-robot manifest is refused, and on the re-layout the dataset identity and the sampler's record sources differ.
The audit of every other path is in `.superpowers/sdd/task-r7-report.md` A3.

## 3. Resume verification

Nothing to resume: the run completed in one process (`summary.rescheduled` None, status `completed`, step 304) with periodic
atomic saves at steps 50 / 100 / 150 / 200 / 250 / 300 and the final one (08:56:40, 09:42:57, 10:31:46, 11:25:53, 12:15:05,
13:05:28, 13:10:57; `artifacts/scratch/r7/t1-polls.log`). The disk gate held: 67.86 GB free at launch (≥ 65 GB), 39 GiB between
saves, 18 GiB once in the middle of the step-200 save (the computed peak is ≈ 15 GB). **An old run cannot be resumed with the new
loader** (A1): its non-robot manifest (`r1/single`, one mixed file) is refused before any file is opened, and on the
split-by-split layout the dataset identity and the sampler's record sources differ (`records.jsonl` → `records.train.jsonl`);
a test pins the refusal. The license for the T1 resume path is unchanged (R2 A1, R3a A2, R5's first long walk).

## 4. Per-question quality (offline — the decision cell is `ood_dev`, 26 episodes / 3,162 ticks, hash `6a3b69131243`; the `dev` cell `9b484441b23b`)

| checkpoint | primary stratum (235) | **instruction-shuffle margin** (paired 95 %) | whole `q_main` | `q_gripper` whole (parent) | `q_gripper` initiate (43): `closed` | `q_gripper` settled (1,015): accuracy | `q_gripper` single-`open` (1,860): accuracy | `q_stop` onsets caught / 10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **R7** | 0.881 [0.849, 0.914] | **+0.370 [+0.241, +0.483]** | 0.982 | **0.680** | **0.279** | **0.043** | 0.990 | **0** |
| R6 | 0.877 [0.842, 0.909] | +0.374 [+0.268, +0.465] | 0.990 | 0.905 | 0.837 | 0.897 | 0.916 | 0 |
| R5 | 0.902 [0.874, 0.933] | +0.298 [+0.232, +0.354] | 0.992 | 0.980 | 0.674 | 0.996 | 0.981 | 5 |
| R7 − R6 (paired by episode) | +0.004 [−0.031, +0.046] | −0.004 [−0.058, +0.046] | | | −0.558 [−0.718, −0.400] | | | |
| R7 − R5 (paired by episode) | −0.021 [−0.052, +0.009] | +0.072 [−0.041, +0.199] | | | −0.395 [−0.571, −0.234] | | | |

Reading (R7 C2). **Instruction reading is intact**: the primary-stratum margin excludes zero, every leave-one-episode-out refit
excludes zero, and it does not move against R6 or R5 (both paired differences contain zero); the non-selecting `dev` cell
replicates the sign (+0.257 [+0.182, +0.328]; paired against R6 +0.036 [−0.012, +0.083]). **The gripper head collapsed to
"open"**: r7 answers `closed` on 3.1 % of the labelled ticks (R6 38.9 %, R5 37.4 %), 4.3 % right on the ticks where the gripper
is already closed on the object — a copy of the execution state that R5 gets 99.6 % right — and 27.9 % on the ticks where the
reference says "close now"; the `dev` cell repeats it (settled 0.035, initiate 0.233). `q_stop` catches none of the 10 (dev: 14)
stop onsets, like R6. The `q_done` strata show no offline regression (`post_release_other` 0.984). Sources:
`artifacts/reports/r7-reeval-2b-t1-fp32-r7.json`, `r7-dev-2b-t1-fp32-r7.json`, `r7-decision-cell-strata.json`,
`r7-dev-cell-strata.json`.

## 5. Closed-loop success (R7 D — the fourth time the model moves the robot)

Setup as in run reports 1–3 (the same `ModelPolicy` serving path, `h0.9` / `c0.6` / `ts0.6`, expert reference labels).
**ood_dev 200** = R4's 26 + R6's 74 + R7's 100 new ood_dev seeds (0 origin groups shared with training; the E1 ruler);
**dev_new2 100** (R6's new dev seeds; secondary). `artifacts/reports/r7-closed-loop.json` (`--merge
ood_dev200=ood_dev,ood_dev_new,ood_dev_new2 --seed-pairs r7:r5 r7:r6 rule:r7 r6:r5`); records `artifacts/datasets/r7-closed-loop/`.

| policy | ood_dev 200: `done` / **strict** (false done) | dev_new2 100: `done` / **strict** (false done) | gripper duplicates (ood · dev) | `q_stop` onsets caught (ood · dev) | failures main / aux / geom (ood) |
| --- | ---: | ---: | ---: | ---: | ---: |
| expert | 0.910 / **182** (0) | 0.900 / **90** (0) | 33 · 15 | 65/65 · 41/41 | 2 / 0 / 16 |
| rule judge | 0.725 / **145** (0) | 0.670 / **67** (0) | 759 · 491 | 31/43 · 25/30 | 2 / 51 / 2 |
| mechanical | 0 / 0 | 0 / 0 | — | — | 200 / 0 / 0 |
| model R5 | 0.810 / **132** (**30**) | 0.790 / **68** (11) | 64 · 27 | 21/36 · 13/31 | 17 / 13 / 8 |
| model R6 | 0.670 / **122** (**12**) | 0.720 / **69** (3) | 439 · 190 | 3/92 · 0/59 | 28 / 38 / 0 |
| **model R7 (shares 0.6/0.4, 304 steps)** | **0.015 / 0 (3)** | **0.060 / 0 (6)** | **897 · 388** | **0/113 · 0/101** | 45 / 151 / 1 |

**The R7 checkpoint does not complete the task in the loop**: strict success 0 of 200 and 0 of 100; its nine `done`s are all
false dones. 244 of its 300 episodes end by the stall watchdog (218 with the arm pinned), because at the grasp point the
gripper head does not say `closed` — r7's raw P(closed) never exceeds 0.59 on R4's 26 seeds (mean 0.43 on the ticks where the
reference says "close now", 0.52 on ticks where the gripper is already closed on the object; r6 0.55 / 0.78, r5 0.66 / 0.97),
and when it wobbles across 0.5 the gripper flickers (897 duplicate transitions on ood_dev 200). The offline cell confirms it
is the model (§4). The data pipeline was ruled out by replaying R6's training with the new loader and the re-laid-out
non-robot set — 233 of 233 steps drew the same records with the same token counts (`artifacts/scratch/r7/replay-r6.json`).

Paired by seed (ood_dev 200 / dev_new2): **rule − R7 strict +0.725 [+0.665, +0.785]** / +0.670 [+0.570, +0.760] (discordant 145
vs 0); **R7 − R5 false-done rate −0.135 [−0.185, −0.085]** / −0.050 [−0.130, +0.030]; R7 − R5 strict −0.660 [−0.725, −0.590];
R7 − R6 strict −0.610 [−0.675, −0.540]. Per seed, R7 − R5 on ood_dev 200: gripper-streak episodes +0.665 [+0.595, +0.735],
**duplicate transitions per episode +4.165 [+2.640, +5.795]**, **`q_stop` onsets caught −0.583 [−0.750, −0.379]**, auxiliary
failures +0.690 [+0.625, +0.760]; R7 − R6: duplicates **+2.290 [+0.545, +4.110]**, gripper-streak episodes −0.030 [−0.080, +0.020]
(0 inside), `q_stop` −0.033 [−0.098, 0.000] (0 inside). The lower false-done rate is not a repaired `q_done` — r7 rarely holds
an object long enough to reach a done state. For reference on the larger set: rule − R6 strict **+0.115 [+0.035, +0.195]** (R6's
knife-edge on 100 seeds is settled on 200), R6 − R5 strict −0.050 [−0.125, +0.025] (0 inside), false done −0.090 [−0.145, −0.035].

Latency (non-first ticks): R7 p50 / p95 / p99 46.4–46.9 / 57.4–58.9 / 82.0–84.7 ms, **0 of 42,786 over 100 ms**; first ticks
(prefix build) p50 ≈ 93 ms, max 291 ms, 26 of 300 over 100 ms — overall 0.06 %; the gate passes.

## 6. Measured cost and the G1 recommendation

| item | GPU-h | source |
| --- | ---: | --- |
| R2 + R3a + R4 (run report 1) | 28.56 | `docs/reports/run-report-1.md` §6 |
| R5 (run report 2) | 5.29 | `docs/reports/run-report-2.md` §6 |
| R6 (run report 3) | 5.92 | `docs/reports/run-report-3.md` §6 |
| **R7** (unit wall clocks: B3 r5 · r6 on the new seeds 26 m 27 s · training 5 h 06 m 04 s · D + C2 1 h 22 m 37 s · full test suite 8 m 13 s = 25,401 s) | **7.06** | `journalctl --user`, `artifacts/scratch/r7/*.log` |
| total on the DGX Spark GB10 | **≈ 46.8** | — |

CPU this round: minutes (the three baseline policies 3.5–5 min each in parallel, reports and strata minutes). Cloud spend: 0.
Disk: `/` had 68.45 GB free at the start and 40.83 GB at the end (one 26.35 GB checkpoint, ≈ 1.2 GB of loop records, the
43 MB re-layout); nothing outside this task's own outputs was deleted.

**G1, decided by the rule registered before the numbers (§0; applied by `scripts/closed_loop.py verdict`,
`artifacts/reports/r7-verdict.json`).** On ood_dev 200: (a) `rule − R7` strict **+0.725 [+0.665, +0.785]** — fails (discordant
seeds 145 vs 0; the lower bound is above zero under all 200 alternative bootstrap RNG seeds); (b) `R7 − R5` false-done rate
**−0.135 [−0.185, −0.085]** — holds, but only because R7 almost never reaches a done state; (c) `R7 − R5` duplicate gripper
transitions **+4.165 [+2.640, +5.795]** and `q_stop` catch rate **−0.583 [−0.750, −0.379]** — both fail. **Under the registered
procedure the cloud stays closed** — this time not on a knife-edge. The dev_new2 values fail all four conditions.

**Cause call, as registered (`R7 − R6` on ood_dev 200).** Duplicates per episode +2.290 [+0.545, +4.110] (against R7 — a new
regression), gripper-streak episodes −0.030 [−0.080, +0.020] and `q_stop` −0.033 [−0.098, 0.000] (both contain zero): none moves
toward R7, so the registered sentence applies — **"노출 비율로 설명되지 않음 — DAgger 부가 라벨이 다음 표적"** (not explained by
the exposure ratio — the DAgger auxiliary labels are the next target).

**What the call cannot carry.** The realized contrast was +84 expert / −13 DAgger draws, not the designed +71 / 0, and R7 did
not repeat R6's regression at another size — its gripper head collapsed to `open` (offline 3.1 % `closed`, 4.3 % right on
ticks where the gripper is already closed on the object; R5 99.6 %), which neither hypothesis predicts: R7 drew fewer DAgger
episodes than R6 and more expert episodes than R5. With instruction reading unchanged (+0.370) and the data pipeline verified
(R6's 233 steps replayed identically with the new loader), the round's finding is that **the auxiliary heads are unstable across
recipe changes at one seed** (gripper: right → hasty → never closes; `q_stop` 5 → 0 → 0 of 10 offline; R3a: `q_stop` 1 / 8 / 0
across three seeds). **Recommendation for the next Spark round:** make that instability measurable before attributing it —
seed repeats of one recipe, per-question training losses and a cheap in-training probe on the `q_gripper` settled / initiate
strata and the `q_stop` onsets — and only then the DAgger-auxiliary-label question the registered call names. One seed: R3a's
seed spread on the primary instruction margin (+0.111 … +0.247) is the noise a single run carries on the *stable* head.
