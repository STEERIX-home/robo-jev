# Run report 2 — the first DAgger checkpoint: gripper labels v2, the model's own hovering states, and the closed loop again (2026-09-25)

The document docs/06 Task 6 defines (:384) — *trained weights, reproducible data, resume verification, per-question
quality and measured cost, presented together* — for the second trained checkpoint of this project, the one that
answers run report 1's G1 recommendation ("repair the auxiliary-judgment labels and the evaluation cell, then a
DAgger cycle on the R4 records, before scaling to the cloud"). Every number is copied from an artifact named next
to it; nothing is transcribed from memory. Korean prose lives in `docs/06`, `docs/08` and `HANDOFF.md`; this report
is the English ledger. Run report 1 stays as it is; this one only adds.

## 1. Trained weights

| run | steps | seed | checkpoint | contract digest | training | evaluated by |
| --- | ---: | ---: | --- | --- | --- | --- |
| **`r5-t1-fp32-2b-s18`** | 233 = one schedule (1–40 by unit `r5-t1`, 41–233 by `r5-t1-resume` from the step-40 checkpoint) | 18 | `artifacts/runs/r5-t1-fp32-2b-s18/checkpoint.pt` (24.5 GiB; model + optimizer + fp32 masters) | `93fe26725a4c…` (unchanged) | 14,438 s of unit wall = **4.01 GPU-h** (train seconds 13,391.7; 57.48 s/step), peak 56.05 GiB, loss 2.633 → 0.150 | R5 C2 (cells), **R5 D2 (closed loop)** |
| `r3a-t1-fp32-2b-s18` (the comparison) | 233 | 18 | `artifacts/runs/r3a-t1-fp32-2b-s18/checkpoint.pt` | same | 14,944.5 s (4.15 GPU-h), peak 56.05 GiB, loss 2.633 → 0.130 | R3a B, R4, **R5 B1 / D2 (new scenes)** |

Recipe: identical to seed 18's (`configs/train/qwen35-2b-r2.yaml` inherited by `configs/train/qwen35-2b-r5.yaml`) —
Qwen3.5-2B bf16 with fp32 master weights, fp32 pointer readout rank 64, full text backbone, `backbone_lr` 1e-5,
`readout_lr` 3e-4, 5-second TBPTT chunks, 30-tick window, accumulation 2, tick weights steady 0.25 / event 2 /
goal_change 2, `max_steps` 233 (the same compute as seed 18, so the difference is attributable to the data). **Only
the data list differs**: the v2-labelled parent corpus, DAgger cycle 0 tagged `material: error_family`, and the
non-robot set. The sampler's 70/20/10 material axis (docs/04 §7), renormalised over the two present buckets,
drew **182 expert episodes (0.78 epoch of 233) and 51 DAgger episodes (0.26 epoch of 200)** in the 233 robot
units (`metrics.json` `summary.sampler.units`); seed 18 saw every expert episode once. Tokens 11.57 M against
12.22 M (DAgger episodes are shorter). The run was interrupted at its step-50 checkpoint save by a full disk
(`ENOSPC`; the root filesystem was 99 % full) and resumed from the step-40 checkpoint; the crash cost ≈ 16 min of
GPU (0.27 h); the resumed run made no periodic save (see §3 and `.superpowers/sdd/task-r5-report.md` C0/C1).

## 2. Reproducible data

| dataset | what | manifest (sha256 of the manifest file) | version |
| --- | --- | --- | --- |
| robot train labels, **gripper rule v2** | the 400 R1 v0.2 episodes of `r1-rollout-labels` with only the `q_gripper` labels changed: reset from `evidence.expert.ticks[*].aux.gripper.desired` and re-widened under rule v2 (transition tick single-valued, the 2 ticks before a close transition `{open, closed}`); 42,509 ticks, splits train 233 / dev 42 / calibration 51 / test 15 / ood_dev 26 / ood_test 33 (sealed, copied as data, never read); train initiate ticks 9 → 316 | `artifacts/datasets/r1-robot/r1-rollout-labels-g2/manifest.json` `4d433bae6d22…` (parent `r1-rollout-labels` `d4bc06b14a4a…`, config sha `d2f61e51cc3c…`) | `labels-rollout-v1+gripper-v2` (generator of the records `gen-robot-v0.2`; rule `gen-robot-v0.3`) |
| **DAgger cycle 0** | R4's dev-condition model-driven records, 200 episodes (seed 18 100 + 466 100), 14,597 ticks, executed answers/ACKs/commitments untouched, labels = expert reference under rule v2, `relabel: true`, split `train`, `material: error_family`; initiate 3,367 · window 444 · settled 0 · open 9,813; policy-error rate 12.1 % | `artifacts/datasets/r5-dagger/dagger-0/manifest.json` `8b205ee13993…` (sources `artifacts/datasets/r4-closed-loop/{s18,466}/dev`) | `gripper-v2`, `dagger-v0.1` cycle 0 |
| non-robot single requests | 2,000 records, unchanged | `artifacts/datasets/r1/single/manifest.json` `46e7e361719b…` | R1 |
| closed-loop scenes (R5) | dev_new 100 **new** seeds (950100+, same generator config, split rule and profile mix E0/E1/E2 = 20/40/40 as R4; 1,104 scanned, 19 of 29 families in r1) | `artifacts/reports/r5-seeds.json`, `artifacts/datasets/r5-closed-loop/<policy>/dev_new/manifest.json` | `cl0.1`, ids `-r5-<label>` |
| closed-loop scenes reused from R4 | ood_dev 26 (never trained on) and dev 100 (the DAgger scenes — "seen") | `artifacts/reports/r4-seeds.json`, `artifacts/datasets/r5-closed-loop/r5/{ood_dev,dev}/` | `cl0.1` |

QA: `artifacts/reports/r1-robot-r1-rollout-labels-g2-qa.json` and `r5-dagger-dagger-0-qa.json` — 0 contract violations,
0 QA violations, 0 leaks. The early-answer experiment behind k = 2 is `artifacts/reports/r5-a2-early-gripper.json`
(100 dev seeds, five policies, seed-paired).

## 3. Resume verification

The license is unchanged (R2 A1's `t1` gate on the exact and parameter criteria, R3a A2's registered spread) — and
this run is the **first long walk of the plain resume path on a GPU**: unit `r5-t1` died at its step-50 save
(`OSError: [Errno 28] No space left on device`, 15 GB free on a 916 GB disk carrying 240 GB of prior checkpoints),
and `r5-t1-resume` continued from `checkpoint-step40.pt` with the same `max_steps` (the keys that differ —
`resume`, `checkpoint_every`, `checkpoint_keep_steps` — are `RESUME_FREE_KEYS`; `summary.rescheduled` is `None`).
Across step 40 → 41 the cosine schedule continues (backbone lr 9.609e-6 → 9.581e-6), the sampler cursor continues
(`drawn` 80 → 82; units robot/existing 27 → 28, robot/error_family 13 → 13, non-robot 239 → 244 — one robot unit and
one non-robot bundle per step, nothing repeated or skipped), the run ends at optimizer step 233 with status
`completed`, and the loss (0.374 → 0.628 → 0.669 → 0.545 at steps 40/41/42/50) sits inside the ±25 % no-restart
spread R2 measured at these losses. The trainer's own checks on resume (contract digest, config differences,
identity block) passed. The GPU gate (`p1_acceptance.py --gate t1`) was **not re-run** this round (the box was
never idle for a second GPU job). The step-40 file was deleted after the resumed process had loaded it, to leave
51 GB for the final save; the *reschedule* path (`resume_reschedule`) still has no GPU run behind it.

## 4. Per-question quality (offline, R3a's cells — the decision cell is `ood_dev`, 26 episodes / 3,162 ticks, hash `6a3b69131243`)

| checkpoint | primary stratum (label ≠ commitment, 235) | **instruction-shuffle margin** (paired 95 %) | `grasp` (97) | whole `q_main` | `q_gripper` **initiate** (43, v2 labels): predicts `closed` | its shuffled controls (state / instr. / commit.) | `q_gripper` whole (v2) | `q_path` | `q_speed` | `q_stop` | `q_stop` onsets caught / 10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **R5** (`r5-t1-fp32-2b-s18`) | **0.902** [0.874, 0.933] | **+0.298 [+0.232, +0.354]** | **0.948** | 0.992 | **0.674 [0.565, 0.789]** | 0.721 / 0.651 / 0.674 (margins contain zero) | 0.982 | 0.990 | 0.893 | 0.997 | 5 |
| seed 18 (R3a) | 0.889 [0.855, 0.925] | +0.247 [+0.179, +0.308] | 0.876 | 0.991 | **0.000 [0.000, 0.000]** | 0.000 / 0.000 / 0.000 | 0.981 | 0.986 | 0.892 | 0.995 | 8 |
| rule judge (ceiling for goal reading) | 0.477 | — | 0.959 | 0.812 | 0.977 | — | 0.913 | 0.994 | 1.000 | 0.983 | 9 |
| mechanical baseline (floor) | 0.077 | — | 0.000 | 0.931 | — | — | — | — | — | — | — |

Reading (R5 C2). The instruction-shuffle margin on the primary stratum widens rather than regresses (+0.247 →
+0.298; all 26 leave-one-episode-out refits exclude zero for all three controls; the non-selecting `dev` cell,
hash `9b484441b23b`, reads +0.268 [+0.212, +0.324] against +0.261). The `q_gripper` initiate stratum — which the
parent labels did not contain at all (0 ticks) and rule v2 gives 43 ticks in every one of the 26 episodes — goes
from 0 of 43 `closed` answers to 29 of 43; but the shuffled controls answer `closed` as often, so the answer is
read from execution state (phase, hold count, the arm's stillness), not from the object's geometry the shuffles
replace; the controller's readiness is what makes that safe in the loop (R5 A2). Sources:
`artifacts/reports/r5-reeval-2b-t1-fp32-{s18,r5}.json`, `r5-dev-2b-t1-fp32-r5.json`, `r5-decision-cell-strata.json`,
`r5-dev-cell-strata.json`, `.superpowers/sdd/task-r5-report.md` B1/C2.

## 5. Closed-loop success (R5 D — the second time the model moves the robot)

Setup as in run report 1 §5 (the same `ModelPolicy` serving path, `h0.9` / `c0.6` / `ts0.6`, expert reference
labels), on three scene sets: **100 new dev seeds** (950100+, harder than R4's — the expert completes 87),
**R4's 26 ood_dev seeds** (never trained on) and **R4's 100 dev seeds** (the DAgger scenes; "seen").

| policy | new dev 100: `done` [95 %] / `done ∧ inside` (false done) | ood_dev 26: `done` / strict (false) | seen dev 100: `done` / strict (false) | gripper transitions executed / reference (new / ood / seen) |
| --- | ---: | ---: | ---: | ---: |
| expert | 0.870 [0.800, 0.930] / 87 (0) | 1.000 / 26 (0) | 0.970 / 97 (0) | 212/222 · 70/70 · 255/255 |
| rule judge | 0.810 [0.730, 0.880] / 81 (0) | 0.692 [0.500, 0.846] / 18 (0) | 0.800 / 80 (0) | 187/193 · 44/46 · 194/206 |
| mechanical | 0.000 / 0 | 0.000 / 0 | 0.000 / 0 | — |
| model seed 18, 1 epoch (R3a) | **0.000** / 0 | 0.000 / 0 | 0.000 / 0 | **0/117 · 0/44 · 0/125** |
| **model R5 (labels v2 + DAgger-0, 1 epoch)** | **0.710 [0.620, 0.800] / 64 (7)** | **0.885 [0.731, 1.000] / 18 (5)** | **0.840 [0.770, 0.910] / 71 (13)** | **170/205 · 57/64 · 210/338** |

Paired by seed: **R5 − seed 18 = +0.710 [+0.620, +0.800]** (strict +0.640 [+0.550, +0.730]) on the new scenes,
**+0.885 [+0.731, +1.000]** (strict +0.692 [+0.500, +0.846]) on ood_dev, +0.840 [+0.770, +0.910] (strict +0.710) on
the seen scenes. **Rule judge − R5 = +0.100 [+0.010, +0.190]** (strict +0.170 [+0.100, +0.250]) on the new scenes —
the goal-reading rule baseline is still ahead where generalisation is measured; on ood_dev −0.192 [−0.423, 0.000]
(strict 0.000 [−0.192, +0.192], a tie); on the seen scenes −0.040 [−0.140, +0.060] (strict +0.090 [+0.010, +0.170]).
Expert − R5 +0.160 [+0.090, +0.240] / +0.115 [0.000, +0.269] / +0.130 [+0.060, +0.200]. Layers: rule = expert on
E0 everywhere; R5 completes 19/22 E0 and 52/78 E1+E2 on the new scenes.

Why the remaining 29 new-scene episodes fail (R5 D2): 15 semantic-aux (22 episodes with a `q_gripper`
disagreement streak, 17 with a `q_path` streak — 35 of 205 transitions missing, 33 duplicates), 6 semantic-main
(185 wrong-action ticks of 7,910, 2.3 %; goal-change reaction on the tick 91.5 %), 8 geometric (the expert itself
fails 12 of these scenes geometrically). **False dones are the new largest gap**: 7 of R5's 71 `done` episodes on the
new scenes, 5 of 23 on ood_dev and 13 of 84 on the seen scenes end with the target outside its zone — the model's
`q_done` fires where the reference label is False (the expert and the rule judge never do). `q_stop`: 18 of 26
onsets caught on the new scenes, false alarms 1.8 % (147 stop ticks from `q_stop` against 20 reflexes); unsafe
0.55 % (seed 18 0.37 %); rejections 0. Latency in the loop (docs/03 §7-6): p50 / p95 / p99 42.8 / 55.7 / 80.9 ms,
0 of 20,757 ticks over 100 ms across the three sets. Sources: `artifacts/reports/r5-closed-loop.json`,
`artifacts/datasets/r5-closed-loop/`, `.superpowers/sdd/task-r5-report.md` D2.

## 6. Measured cost and the G1 recommendation

| item | GPU-h | source |
| --- | ---: | --- |
| R2 + R3a + R4 (run report 1) | 28.56 | `docs/reports/run-report-1.md` §6 |
| **R5** (unit wall clocks, `Started` → `Consumed`: B1 eval 12 m 46 s · `r5-t1` 50 m 47 s (died at the step-50 save) · `r5-t1-resume` 3 h 09 m 51 s · `r5-after` six jobs 1 h 04 m 15 s = 19,060 s) | **5.29** | `journalctl --user`, `artifacts/scratch/r5/*.log` |
| total on the DGX Spark GB10 | **≈ 33.9** | — |

CPU this round is minutes (A2 five policies ≈ 4 min in parallel, the two derived datasets ≈ 6 min, three CPU
policies on the new scenes ≈ 5 min). Cloud spend: 0. The disk, not the GPU, was the binding resource: the root
filesystem reached 99 % (240 GB of prior checkpoints under `artifacts/runs`), the training died once, and the
resumed run could keep no periodic checkpoint.

**Recommendation on G1 (from the numbers only).** Run report 1 said: do not scale until the auxiliary-judgment
labels and cell are repaired. They are, and the repair moved the loop from 0/126 to 64–71 % strict on unseen
scenes and 18/26 on the sealed-concept holdout while the instruction-reading margin widened — so a cloud run on
this recipe is no longer pointless. It is still not the next step. Two numbers argue for one more Spark round
first: the rule judge beats this single seed on the new scenes (+0.100 [+0.010, +0.190]; strict +0.170), and the
largest remaining failure is a false-done rate of 10–15 % of completions plus auxiliary streaks on model-driven
states — both addressable by a second DAgger cycle on the R5 records (226 model-driven episodes that now contain
completions, false dones, drops and flaps) and a check of the `q_done` label/gate, for ≈ 5 GPU-h here. Open the
cloud (three seeds, two epochs) after cycle 1 shows those two rates moving; R3a's seed spread (+0.111 … +0.247 on
the primary margin) remains the noise floor any cloud comparison must clear, and one seed's 0.71 is not yet the
recipe's number.
