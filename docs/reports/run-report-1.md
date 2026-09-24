# Run report 1 — the first trained robojev checkpoint, from weights to the closed loop (2026-09-24)

This is the document docs/06 Task 6 defines as its completion condition (:384): *directly trained weights, reproducible
data, resume verification, per-question quality and measured cost, presented together* — extended by Task R4 with the
first closed-loop success rates. Every number here is copied from an artifact named next to it; nothing is
transcribed from memory. Korean prose lives in `docs/06` and `HANDOFF.md`; this report is the English ledger.

## 1. Trained weights

| run | steps | seed | checkpoint | contract digest | training | evaluated by |
| --- | ---: | ---: | --- | --- | --- | --- |
| `r3a-t1-fp32-2b-s18` | 233 = 1 epoch | 18 | `artifacts/runs/r3a-t1-fp32-2b-s18/checkpoint.pt` (24.5 GiB; model + optimizer + fp32 masters) | `93fe26725a4c…` | 14,944.5 s (4.15 GPU-h), peak 56.05 GiB, loss 2.633 → 0.130 | R3a B1a/B3, **R4 (closed loop)** |
| `r3a-t1-fp32-2b-466` | 466 = 2 epochs (fresh run) | 17 | `artifacts/runs/r3a-t1-466/r3a-t1-fp32-2b-466/checkpoint.pt` | same contract | 29,205.5 s (8.11 GPU-h), loss 2.672 → 0.102 | R3a C3/C4, **R4** |
| `r2-t1-fp32-2b` | 233 | 17 | `artifacts/runs/r2-t1-fp32-2b/checkpoint.pt` | same contract | 15,555.9 s (4.32 GPU-h) | R2 B |
| `r3a-t1-fp32-2b-s19` | 233 | 19 | `artifacts/runs/r3a-t1-fp32-2b-s19/checkpoint.pt` | same contract | 14,860.8 s (4.13 GPU-h) | R3a B1b |

Recipe (all four): Qwen3.5-2B backbone in bf16 with fp32 master weights (`MasterWeightAdamW`), fp32 pointer readout
rank 64, full text backbone trained (`text_backbone_and_readout`), `backbone_lr` 1e-5, `readout_lr` 3e-4, 5-second
TBPTT chunks, 30-tick window, gradient accumulation 2, `configs/train/qwen35-2b-r2.yaml`. The deployment contract
digest binds the serializer and contract sources, the harness version (`h0.9`) and the tokenizer file hash
(`Qwen/Qwen3.8-27B` `tokenizer.json`, `0997f410…`); `load_readout_checkpoint` refuses a checkpoint whose digest differs
from the checkout that serves it, and R4's loader (`load_serving_judge`) goes through that function.

## 2. Reproducible data

| dataset | what | manifest | version |
| --- | --- | --- | --- |
| robot train / eval corpus | 400 expert episodes, 42,509 ticks, splits train 233 / dev 42 / calibration 51 / test 15 / ood_dev 26 / ood_test 33 (sealed, never read) | `artifacts/datasets/r1-robot/r1/manifest.json`, config sha256 `d2f61e51cc3c…` | `r1-robot-v0.2` (generator `gen-robot-v0.2`, harness `h0.9`, expert `e0.4`, controller `c0.6`, record serializer `s0.3`) |
| robot train labels | the same 400 episodes with 128k-plan keyframe rollout labels (100,640 realised) | `artifacts/datasets/r1-robot/r1-rollout-labels/manifest.json` | `d1-rollout-labels` lineage |
| non-robot single requests | 2,000 records (951 base + 760 contrast pairs) | `artifacts/datasets/r1/single/manifest.json` | R1 |
| **closed-loop scenes (R4)** | dev 100 + ood_dev 26 **new** seeds (900100+, same split rule, E0/E1/E2 = 20/40/40), five policies × 126 episodes | `artifacts/reports/r4-seeds.json`, `artifacts/datasets/r4-closed-loop/<policy>/<condition>/manifest.json` | `cl0.1` |

The seed schedule, the split assignment (family hash + holdout tags) and the scene plans are deterministic functions of
the config, so any of these sets can be rebuilt from the config file alone; the manifests carry per-file sha256.

## 3. Resume verification

R2 A1: the `t1` resume gate (3 steps + a real process restart + 3 steps) passes on the criteria that discriminate —
sampler position, drawn units, optimizer step count, no missing tensor, parameter max-abs 5.959e-4 ≤ 0.01, relative
L2 3.518e-3 ≤ 0.05 — with the loss difference (0.315) kept as a diagnostic next to the no-restart spread of the same
path (0.079–0.446 over 6 steps; R3a A2 registered four pairings). R3a A1 added the CUDA generator to the saved unit.
What has **not** been walked on a GPU: the documented resume-and-reschedule path (`resume_reschedule`); R3a's
466-step run was launched as a continuation and ran from scratch because `run_training` dropped the `resume` key
(fixed with a test, ≈4.0 GPU-h lost). CPU tests cover the path; no GPU run does.

## 4. Per-question quality (offline, R3a's cells — the decision cell is `ood_dev`, 26 episodes / 3,162 ticks)

| checkpoint | primary stratum (label ≠ commitment, 235 ticks) | instruction-shuffle margin (paired 95 %) | `grasp` (97) | whole cell | `q_gripper` | `q_path` | `q_speed` | `q_stop` | unsafe | `q_stop` onsets caught / 10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| seed 18, 1 epoch | 0.889 | **+0.247 [+0.179, +0.308]** | 0.876 | 0.991 (`q_main`), 0.986 (all) | 0.996 | 0.986 | 0.892 | 0.995 | 0.0036 | 8 |
| seed 17, 2 epochs (466) | 0.860 | +0.191 [+0.137, +0.247] | 0.845 | 0.989 / 0.996 | 0.998 | 0.997 | 0.983 | 0.998 | 0.0026 | 6 |
| seed 17, 1 epoch (R2) | 0.762 | +0.111 [+0.065, +0.158] | 0.629 | 0.982 | — | — | — | — | 0.0092 | 1 |
| seed 19, 1 epoch | 0.804 | +0.128 [+0.084, +0.175] | 0.722 | 0.985 | — | — | — | — | 0.0099 | 0 |
| rule judge (ceiling for goal reading) | 0.477 | — | 0.959 | 0.812 | 0.903 | 0.994 | 1.000 | 0.983 | 0.0003 | 9 |
| mechanical baseline (floor) | 0.077 | — | 0.000 | 0.931 | — | — | — | — | 0.0000 | — |

Reading (R3a D1/D2): the instruction-shuffle margin excludes zero at all three seeds, so "the model reads the
instruction" is not seed-dependent; its size is known to within a factor of two (+0.111 … +0.247); `grasp` carries
it; `q_stop` is seed noise (1 / 8 / 0 onsets). Sources: `artifacts/reports/r3a-reeval-2b-t1-fp32-{s18,466}.json`,
`r3a-decision-cell-strata.json`, `.superpowers/sdd/task-r3a-report.md`.

## 5. Closed-loop success (R4 — the first time the model moves the robot)

Setup: `generate_episode` with the policy seat filled by the checkpoint (`ModelPolicy`: incremental ts0.6 stream
serialization, static prefix KV + 30-tick window, ten decision branches in one fused forward, batched pointer readout,
`compile` on the dense parts, fp32 readout), the expert's reference answer recorded at every tick as `labels`, the
harness `h0.9` and controller `c0.6` untouched. Before running, the loop policy was shown to reproduce the offline
evaluation **bit-for-bit** on three recorded dev episodes (3,890 answers, probability difference 0; 389/389 against
R3a's stored predictions) — `artifacts/reports/r4-a3-s18.json`.

| policy | dev 100 success (seed-level 95 % CI) | ood_dev 26 | paired vs seed-18 model (dev) | paired vs expert (dev) |
| --- | ---: | ---: | --- | --- |
| expert (the policy that made the data) | **0.970** [0.930, 1.000] | **1.000** | +0.970 [+0.930, +1.000] | — |
| rule judge (`rj0.5`, reads the structured goal) | **0.800** [0.720, 0.870] | **0.692** [0.500, 0.846] | **+0.800 [+0.720, +0.870]** | −0.170 [−0.250, −0.100] |
| mechanical baseline (commitment else observe) | 0.000 | 0.000 | 0.000 [0, 0] | −0.970 |
| **model, seed 18, 1 epoch** | **0.000** [0, 0] | **0.000** | — | −0.970 [−1.000, −0.930] |
| model, seed 17, 2 epochs (466) | 0.020 [0.000, 0.050] | 0.000 | +0.020 [+0.000, +0.050] (contains 0) | −0.950 [−0.990, −0.890] |

Layers (a property of the scene): in the no-instruction-change layer (E0, 20 + 5 seeds) the rule judge equals the
expert (20/20, 5/5; paired 0.000 [0, 0]); in the instruction-change layer (E1/E2, 80 + 21) it does not (60/80 vs 77/80,
−0.213 [−0.312, −0.125]; 13/21 vs 21/21, −0.381 [−0.619, −0.190]). **The rule baseline saturates one layer, not
both, so docs/06's trigger for a task redesign is not met.** The models complete nothing in either layer; the 466
run's two completions are episodes whose final instruction target already sat in its zone (zero closed-gripper ticks).

**Why the model fails, from the records** (`artifacts/reports/r4-closed-loop.json`, seed 18, dev): the adopted main
action agrees with the expert reference on 6,463 of the 7,041 ticks where the reference allows a joint action
(91.8 %; 2.2 % wrong target/function, 6.0 % gating while the reference would act); after a goal change the adopted
action is the new reference on the change tick in 95.9 % (rule judge 79.9 %); switch rate 0.018, 0 round trips. But
the model **never initiates the gripper close** — 0 of 125 reference gripper transitions executed (0 of 44 in
ood_dev) — so every episode hovers open at the grasp point until the stall guard ends it (`stall_exhausted`, 126/126).
Failure attribution: 90 semantic-auxiliary (81 with a `q_gripper` disagreement streak), 8 semantic-main, 2 geometric.
The same checkpoint scores `q_gripper` 0.996 offline: in recorded expert episodes the executed gripper state is in the
input from the tick after the expert closed, and the label agrees with it; the tick where "close now" must be
*initiated* is a few per grasp and half-covered by the label's ±1-tick tolerance. Replaying the expert's own R4
episodes through the checkpoint scores those initiate ticks directly (§5a). Other loop rows: `q_stop` 6 of 8 onsets
caught (median 0 ticks), false alarms 1.1 % of 7,230 quiet ticks — 82 stop ticks from `q_stop` against 4 from the
controller reflex; unsafe-action rate 1.60 % (105 forbidden-target ticks, 25 stop-ignored); controller rejections
0.08 % (`unreachable`), transition collisions 0.

### 5a. Offline on the same seeds (`artifacts/reports/r4-offline-loop-seeds-s18.json`, `r4-transitions-s18.json`)

The expert's own R4 episodes (dev 11,128 ticks, ood_dev 2,715; eval-set hash `53e77e69fb54`) replayed through the
seed-18 checkpoint: `q_main` 0.994 / 0.996 (rule judge 0.815 / 0.761, mechanical 0.932 / 0.934), **`q_gripper` 0.997
[0.994, 1.000] / 1.000**, `q_path` 0.988 / 0.990, `q_stop` 0.997 / 0.994, unsafe 0.14 % / 0.27 %. Split by what the
gripper tick asks: **initiate** (label single `closed`, executed gripper still open) **3 ticks** in dev, 0 in ood_dev —
the model gets the 3; **window** (two-valued label under the ±1-tick tolerance, executed still open — where the expert
actually started closing) 271 / 72 ticks — the model predicts `closed` on **1 / 271 and 0 / 72**; settled (executed
already closed) 3,477 / 952 — 0.992 / 1.000; open 6,595 / 1,495 — 0.9997 / 0.9993. The R1 v0.2 **training** split has
9 single-valued initiate ticks among 23,192 gripper ticks (7,895 settled-closed, 614 window ticks around 301
executed closes). The offline 0.996 and the loop's 0 of 125 executed transitions are the same model read on
different ticks.

### 5b. Latency in the loop (docs/03 §7-6)

Seed 18, dev, 7,171 non-prefix ticks, CUDA events: model **p50 43.6 / p95 56.8 / p99 82.7 / max 94.7 ms, 0 ticks over
100 ms**; first tick of an episode (prefix ≈1.4K tokens) p50 94 ms; observation → command p50 47.9 / p95 63.4 / p99
100.7 ms including the reference computation (harness ≈2 ms). **Passes** (p95 ≤ 80, > 100 ms ≤ 5 %). GPU peak
allocation 4.45 GiB.

## 6. Measured cost and the G1 recommendation

| item | GPU-h | source |
| --- | ---: | --- |
| R2 (gate, T1 233 steps, evaluations, diagnostics) | 10.07 | `.superpowers/sdd/task-r2-report.md` |
| R3a (two seeds, one 466 run, evaluations) | 17.84 | `task-r3a-report.md` D5 |
| **R4** (A3 2.0 min · seed-18 loop 11.7 min · 466 loop 11.2 min · same-seed offline replay ≈13 min) | **≈0.63** | `artifacts/scratch/r4/*.log`, `r4-run-*.json` |
| total on the DGX Spark GB10 | **≈28.5** | — |

CPU: the three CPU policies ran 126 episodes each in 3.7 / 4.2 / 6.2 minutes; a model episode costs ≈5.3 s wall
(73 ticks × 45 ms model + simulation). Cloud spend: 0 (R3b's launcher is ready; no instance was created).

**Recommendation on G1 (from the numbers only).** Do not scale seeds or data to the cloud on this recipe yet. The
closed loop shows a failure the offline cells score at 0.996 — an auxiliary judgment (initiate the gripper close) that
the recorded data lets the model answer by copying the executed state, so more of the same data at more seeds would
raise the offline number and leave the loop at 0/126. What has to change first is the instrument and the labels
around that judgment: an evaluation cell that scores the initiate ticks on their own (§5a is its first form), a label
whose ±1-tick tolerance does not erase the transition, and — since the same auxiliary mechanism plausibly holds for
`q_path` — a DAgger cycle on the R4 records (126 × 2 episodes of model-driven states with expert reference labels,
the first such data in the project) so the model sees the hovering states it creates. The main decision does not
need the cloud to be believed: it already agrees with the expert on 92 % of decisive ticks in the loop and reads the
instruction on the tick it changes. If G1 is expanded later, the seed spread R3a measured (margin +0.111 … +0.247 at
one epoch) is the noise floor any cloud comparison has to clear.
