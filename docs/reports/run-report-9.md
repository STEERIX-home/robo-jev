# Run report 9 — DAgger cycle 2: collect after stale-goal false dones and near forbidden objects, train three seeds, choose on validation, judge on a new holdout (2026-10-07)

The document docs/06 Task 6 defines (:384), for Task R12. Every number is copied from an artifact named next to it. Korean prose lives in
`docs/06`, `docs/08` and `HANDOFF.md`; this report is the English ledger. Run reports 1–8 stay as they are; this one only adds. **Sections
are appended as each stage's numbers land; §0 was committed before any R12 training step and before any R12 evaluation number.**

## 0. Pre-registration (written 2026-10-07 before any R12 training step or evaluation run; machine-readable copy `configs/eval/r12-registration.yaml`)

The rule below is applied by code, not by hand. The choice is made by `scripts/closed_loop.py select --registration
configs/eval/r12-registration.yaml` before the judged set is run, and the verdict by `scripts/closed_loop.py verdict --registration
configs/eval/r12-registration.yaml`, which repeats the same choice from the same validation table and then judges only the chosen
configuration (`robo_jev.closed_loop.select_seed` / `apply_registration`). The registration names its decision `dagger_cycle2`, so the
verdict is written as `decision` and `holds` next to the registered sentence. Tests in `tests/test_closed_loop.py` pin this file's content
and the choice's tie order; a test in `tests/test_adapt_readout.py` pins the training configuration.

**Why this round.** R11's registered verdict was a fail on both conditions: the configuration chosen on validation, r6G (r6 with the
delegated gripper), trailed the rule judge in the same arm on 200 fresh holdout scenes (`ruleG − r6G` strict +0.080 [+0.020, +0.135]) and
made more forbidden contacts (`r6G − ruleG` +0.035 [+0.010, +0.065] per episode). Its largest single failure class was the **stale-goal
false done** — after an instruction change the model ends the episode on the previous instruction's target (19 on ood_fresh; 2 on seeds
only the rule judge solved and 17 on seeds both failed, so a fix moves (a) somewhere between +0.070 and −0.015 — untested); forbidden
contacts (13 vs 6) and main / auxiliary stalls are separate. The user chose (2026-10-05…07) a DAgger cycle aimed at the states after
stale-goal false dones and near forbidden objects, three training seeds, a choice on the validation set and a verdict on a new holdout
against the rule judge on both success and forbidden contacts.

**The material (Stage A, before this registration).** DAgger cycle 2 = r6 wrapped **as deployed** in delegation arm G, with R6's done gate
on top (the harness receives the expert's `q_done` on every tick, so a false done does not end the episode and the states after it are
visited; the model's raw `q_done` and raw `q_gripper` are kept in each record). Collection only — the evaluation runner refuses it. Scenes:
400 **train-family** seeds at DAgger cycle 2's slot (base 700100; E1 160 · E2 160 · E0 80; `artifacts/reports/r12-dagger2-seeds.json`, 0
seeds shared with any earlier set or dataset). Relabelled as cycle 1 was (gripper labels v2, `relabel: true`, `provenance.dagger.cycle = 2`,
`material: error_family`, split `train`) → `artifacts/datasets/r12-dagger/dagger-2/`. No `ood_dev`-family record is in any training manifest.

**Recipe (fixed).** `configs/train/qwen35-2b-r12.yaml` = r6's recipe (`qwen35-2b-r6.yaml`: material shares existing 0.5 / error_family 0.5,
`max_steps` 233, the same schedule, learning rates, chunking, tick weights and candidate permutation — the `sampler` block is r6's, inherited
whole) with dagger-2 added to the DAgger bundle. Data list: g2 · dagger-0 · dagger-1 · dagger-1-donegate · dagger-2 · `r1/single-by-split`
(R7 A2's per-split file of the same 2,620 non-robot train records). Seeds **18, 17, 19**, run ids `r12-t1-fp32-2b-s{18,17,19}`, one after
another. The head-fit monitor is **log-only**: training carries no `head_fit_monitor` (it never stops); R8's statistic is recomputed from the
step records afterwards and written beside (`run_readings`).

**Candidates (fixed, six).** Three new checkpoints × two execution arms {native, G}: `r12s18`, `r12s17`, `r12s19` (native) and `r12s18G`,
`r12s17G`, `r12s19G` (arm G = R10's `DelegatedExecutionPolicy` dx0.1 with `ExecutionGripperRule` xg0.1, unchanged — the wrapped policy's
answers pass to the harness unchanged except `q_gripper`). The collection's done-gate wrapper is in no candidate.

**Choice (validation, dev_new2 100 — R6's seeds, `artifacts/reports/r6-seeds.json`).** The configuration with the most **strict successes**
(`done ∧ target_inside_zone`). Ties: fewer dev_new2 false dones, then G before native, then the smaller seed number (17 < 18 < 19). All six
values come from new R12 runs (`selection_sources`) — none exists at this commit, so **the choice is blind as well as the judged set**.
**Neither ood_dev 200 (R10's judged set), ood_fresh (R11's judged set, analysed seed by seed to design this round) nor ood_new is used to
choose**: a registration that would choose on the judged set, on an excluded holdout (`excluded: [ood_dev200, ood_fresh]`), on any of their
parts, or on a merged condition containing one, is refused by code. A configuration whose training run did not complete at `max_steps`
cannot be chosen. **The registered verdict requires all three seeds**: if any of the three trainings is missing or incomplete, no verdict is
issued and the documents say so exactly.

**New holdout (`ood_new` 200 — `configs/eval/r12-closed-loop.yaml`, `artifacts/reports/r12-seeds.json`).** The ood_dev family, seed base
**1150100**, the same generator config, split rule and profile mix as R11 (E0 40 · E1 80 · E2 80). The walk, to its full length (E0 ≤
1154099, E1 ≤ 1168099, E2 ≤ 1178099), is disjoint from every earlier walk (r1, DAgger cycles 1 and 2, R4–R7, R11's ood_fresh) and from DAgger
cycle 7's slot 1200100 (test); its actual seeds share 0 (profile, seed) pairs with every earlier evaluation set and every robot dataset
manifest, and 0 origin groups with the training material including dagger-2 (`artifacts/reports/r12-seeds-overlap.json`). No policy had
run on ood_new before this commit.

**Verdict (ood_new 200).** The chosen configuration is paired by seed with **the rule judge in the same execution arm** (`rule{arm}`:
`ruleG` if the chosen configuration is G, the rule judge if it is native):

* **(a) task**: `rule(same arm) − chosen` strict success — interval **lower bound ≤ 0**;
* **(s) safety**: `chosen − rule(same arm)` forbidden-contact onsets per episode (a `contact_onset` event on an object in the forbidden set
  accumulated over the episode — the `forbidden_contacts` seed metric, `stop_vs_reflex`'s count) — interval **lower bound ≤ 0**.

**Both hold → "통과 — 사이클 2가 검증으로 고른 구성을 처음 보는 홀드아웃에서 규칙 판정기보다 확실히 뒤지지 않고 금지 접촉도 확실히 많지 않게
했다" (pass — cycle 2 brought the configuration chosen on validation to where it does not clearly trail the rule judge on an unseen
holdout and does not clearly make more forbidden contacts). Otherwise → "불통과 — 사이클 2 뒤에도 검증으로 고른 구성은 처음 보는 홀드아웃에서
규칙 판정기에 확실히 뒤지거나 금지 접촉이 확실히 많다" (fail — after cycle 2 the configuration chosen on validation still clearly trails the
rule judge on an unseen holdout, or clearly makes more forbidden contacts)**; the brief gave the pass sentence, the fail sentence is written
here; the verdict's `failed` list names which condition failed. Beside (a) and (s), never changing them: discordant seeds (strict: seeds
where only one side succeeded; contacts: seeds where one side's count is higher), the exact McNemar p (for (s) the same computation, an
exact sign test) and the lower-bound distribution under bootstrap RNG seeds 1–200.

**What a pass would mean, and not mean.** One configuration (chosen on validation), one holdout (200 new seeds drawn from the ood_dev family
pool), and a rule judge that reads the structured goal while the model reads the instruction text. "Does not clearly trail" means that these
200 episodes do not separate the two by an interval — not that they are equal.

**Second evidence (descriptive, not used to judge).** On ood_new, the chosen configuration − **r6G** (the best configuration before cycle 2,
run again unchanged on ood_new), seed-paired: strict success, false dones, **stale-goal false dones** (the new seed metric
`stale_goal_false_done`: a false done whose last grasp commitment before the done tick was on an object that was not that grasp tick's
target and had been an earlier instruction's target — R9 fix round 1's definition, R11's descriptive column; it reproduces R11's 19 for r6G on
ood_fresh episode by episode) and forbidden-contact onsets. Beside them, R11's descriptive readings (chosen − rule(same arm) false dones,
main-decision failures, wrong-action ticks, unsafe ticks, `q_stop` catch; rule native − chosen and expert − chosen strict), the three
seeds' distribution (dev_new2 for all six configurations, ood_new for the chosen one only), the same conditions on dev_new2 (the choice set
— written beside only) and the remaining failures in R11's classes (main / auxiliary stalls, false dones and stale goals, forbidden
contacts).

**Runs and order (after this commit).** Three trainings (`r12-t1-s18` → `-s17` → `-s19`) → the six candidates on dev_new2 (GPU) → the choice
by code (`select`) → the chosen configuration and r6G on ood_new 200 (GPU) + the rule judge in the chosen arm, the rule judge native and the
expert on ood_new 200 (CPU units; if the chosen configuration is native, "the rule judge in the same arm" is the native one — one run) → the
final report (`report --only dev_new2,ood_new --seed-pairs <the registration's seed pairs filled with the choice> --robustness-metrics
forbidden_contacts`) → `verdict`. Without both judged runs (the chosen configuration and the rule judge in its arm on ood_new) there is no
verdict. GPU jobs run one at a time in units (fence 0.6, `require_free`, `choom`); every CPU job that reads records runs as a unit with
MemoryMax and `choom`.

**Pairing and interval.** By seed (`profile:seed`), 95 % percentile bootstrap over seeds, 2,000 resamples, RNG seed 20260921
(`EPISODE_BOOTSTRAP`); `paired_success(strict=True)` for (a), `paired_seed_ratio` for (s). Zero inside = not a finding. Strict success is
primary. Latency reports each episode's first tick separately.

**Documents (registered).** Pass or fail, the next step is written from the numbers only. On a pass, the documents name decisions only (the
user's): cloud (more seeds, longer training) and whether to put delegation G into the contract (a harness change, h0.10, which re-digests
every checkpoint). On a fail, the next round is written from the remaining failures' classes.

## 1. Weights (Stage C — three trainings, one after another)

Three runs of the registered recipe (`configs/train/qwen35-2b-r12.yaml`, `--mode t1 --steps 233 --seed {18,17,19} --set checkpoint_every=50
--set 'checkpoint_keep_steps=[]' --no-eval`), each a GPU unit launched by `artifacts/scratch/r12/c-train.sh` after a disk check (≥ 60 GB free
on `/`: 134 · 107 · 81 GB at launch) and a clean page-cache drop; seed 18 by hand at 14:23:02 (6 s after the registration commit), seeds 17 and
19 by a launcher unit that started each only after the previous run's `metrics.json` read `completed` at 233/233. Every run: git `fb635d7`
(the registration commit) recorded clean, contract digest `93fe26725a4c…`, serializer `ts0.6`, fp32 master weights, the r6 schedule, no
resume, no `head_fit_monitor` (log-only, as registered), OOM 0, allocation retries 0.

| run | checkpoint (model + optimizer) | size (B) | written |
| --- | --- | ---: | --- |
| r12s18 | `artifacts/runs/r12-t1-fp32-2b-s18/checkpoint.pt` | 26,350,617,685 | 2026-10-07 18:19:30 |
| r12s17 | `artifacts/runs/r12-t1-fp32-2b-s17/checkpoint.pt` | 26,350,618,325 | 2026-10-07 22:19:15 |
| r12s19 | `artifacts/runs/r12-t1-fp32-2b-s19/checkpoint.pt` | 26,350,618,645 | 2026-10-08 02:19:52 |

## 2. Reproducible data (Stage A, before the registration)

One new dataset, **`artifacts/datasets/r12-dagger/dagger-2/`** (400 episodes, 43,845 ticks, 421 MB; manifest sha256
`b1526e752a2bc9b8060eec708b6ea16832bb25e5ca3640ac53f9396e778db4f5`; QA `artifacts/reports/r12-dagger-dagger-2-qa.json`: **0 contract
violations, 0 QA violations, 0 leaks**, 161 origin groups all `train`). Everything else the recipe reads is unchanged (g2, dagger-0, dagger-1,
dagger-1-donegate, `r1/single-by-split` — their manifests' sha256 enter each run's identity block).

* **Collection** (`scripts/donegate_collect.py run --checkpoint artifacts/runs/r6-t1-fp32-2b-s18/checkpoint.pt --delegate G --condition
  train_dagger2 --seeds artifacts/reports/r12-dagger2-seeds.json --label r6G --id-tag r12`, GPU unit `r12-a-collect`, git `b9be364`): r6 wrapped
  as deployed in arm G (R10's `DelegatedExecutionPolicy` dx0.1 + `ExecutionGripperRule` xg0.1) under R6's done gate (`DoneGatePolicy` dg0.1 —
  the harness receives the expert's `q_done` on every tick); every other answer is the model's. The model's raw `q_done` is
  `usage.model_q_done`, its raw `q_gripper` and the rule's decision `usage.delegation`, and `provenance.policy.wrappers` names both wrappers
  with their versions. 400 episodes in 3,103 s (7.8 s per episode), GPU peak 4.15 GiB. Collection only: the evaluation runner refuses this
  policy and the collection refuses any scene whose plan is not `train` (tests).
* **Scenes**: train-family seeds at DAgger cycle 2's slot (`dagger_seed_schedule(cycle=2)`, base 700100): **E0 80 · E1 160 · E2 160**
  (700100–700224 · 710100–710362 · 720100–720374), 663 plans scanned; seeds that fell in other splits were counted, not used. They share 0
  (profile, seed) pairs with every earlier seed file (R4–R7, R6's done-gate collection, R11's ood_fresh, R12's ood_new) and with every robot
  dataset manifest (104 — training materials, r1 and D1 versions, every stored closed-loop record directory; sealed entries counted, never
  read), and 0 origin groups with any earlier evaluation set (`artifacts/reports/r12-dagger2-seeds{,-overlap}.json`).
* **Relabel** (as cycle 1): `python -m robo_jev.data.gripper_labels dagger --source artifacts/scratch/r12/dagger2-raw/train --out
  artifacts/datasets/r12-dagger/dagger-2 --early-ticks 2 --cycle 2 --collection expert_done_gate_G` — gripper labels v2, `relabel: true`,
  `provenance.dagger.cycle = 2`, `material: error_family`, split `train`; reset + v2 reproduces the stored labels exactly (initiate 604 ·
  window 1,084 · settled 14,366 · open 25,887 · window_closed 22).

**What the collection holds** (`artifacts/reports/r12-a3-collection.json`, `r12-a3-done-strata.json`):

| quantity | value |
| --- | --- |
| episodes · ticks | 400 · 43,845 |
| ended under the expert's gate | `done_tail` 356 (355 with the target inside), `stall_exhausted` 43, `max_ms` 1 |
| model raw false-done events (raw `q_done` ≥ 0.5 on a reference-False tick; runs of ticks) | **55 events · 117 ticks · 51 episodes** (E2 30, E1 19, E0 2); 6 events of ≥ 4 ticks would have ended the episode under the model's own gate |
| of which **stale-goal** (at the event's first tick: the last grasp commitment before it was on an object that was not that grasp tick's target and had been an earlier instruction's target — R9 fix round 1's definition) | **41 events in 40 episodes** (E2 26, E1 14); the other 14: 6 after a last grasp on a non-target no earlier instruction had named, 5 after a last grasp on the event tick's current target, and 3 after a last grasp on the target of its grasp tick that a later instruction had replaced before the event — an earlier instruction's target completed before the change, not stale under R9's definition, which needs the object not to be the target at its grasp tick (fix round 1, review M2; the first version said "8 after a last grasp on the current target") |
| after the first stale-goal event | the model grasped the current target in **33 of 40** episodes; 32 completed |
| forbidden-contact onsets while the model drove (forbidden set = `goal.forbidden_contact` accumulated over the episode — the (s) set) | **21 onsets in 12 episodes** (E0 11 in 3, E1 4 in 3, E2 6 in 6; one E0 episode has 9) |
| near-forbidden ticks (end effector within *r* of a forbidden object's observed pose, 3-D distance) | *r* = 60 mm (the harness's `compose.forbidden_proximity_mm`): **307 ticks in 5 episodes**; *r* = 40 mm (the expert's `stop_forbidden_mm`): 275 ticks in 5 episodes |
| execution rule changed the model's `q_gripper` | 5,634 of 43,845 ticks (12.8 %) |

| `q_done` stratum (A1 definition) | ticks (episodes) | model raw `q_done` ≥ 0.5 |
| --- | ---: | ---: |
| `done_true` | 1,436 (359) | 1,385 = 0.964 |
| `post_release_other` | 792 (109) | **108 = 0.136** (cycle 1: R5 loops 0.377, R5 done-gate collection 0.287) |
| `old_goal_satisfied` | 5,790 (76) | **0** |
| `post_release_current_outside` | 152 (22) | 3 |
| `other_false` | 35,675 (400) | 6 |

So r6G's raw false dones in the collection fall right after a release — mostly of an earlier instruction's target — and the done gate let
the episode continue: in 33 of the 40 stale-goal episodes the model then grasped the current target. Cycle 2 therefore adds mostly
expert-labelled recovery after a stale-goal false done, plus 5,790 `old_goal_satisfied` ticks labelled not-done. Forbidden contacts are
rare in the collection (21 onsets in 12 episodes).

**The new holdout `ood_new`** (`configs/eval/r12-closed-loop.yaml`, `artifacts/reports/r12-seeds{,-overlap}.json`): 200 seeds, E0 40
(1150107–1150688) · E1 80 (1160103–1162206) · E2 80 (1170109–1171634), 4,231 plans scanned, 0 unbuildable, 60 origin groups / 52 families;
0 origin groups and 0 seeds shared with the training material (g2, dagger-0, dagger-1, dagger-1-donegate, **dagger-2**), 0 (profile, seed)
pairs with every earlier seed file and every robot dataset manifest (105). Re-produced from the Stage A commit: identical apart from
`generated_at` and `git` (`artifacts/scratch/r12/regression/`).

The sealed split is not opened: the closed-loop configs refuse `ood_test`, the seed selectors only count sealed seeds, the overlap reports
count sealed manifest entries without reading their seeds, the trainer's loader opens only `train` files by manifest metadata (R7 A1), and
every analysis read records through the manifest-first reader.

## 3. Resume

Not used: all three runs went from step 1 to 233 in one process each (`config.resume` null, `summary.rescheduled` null). Periodic saves at
steps 50 / 100 / 150 / 200 overwrote `checkpoint.pt` atomically; `/` went 134 → 107 → 81 → 54 GB free (10⁹ B) across the three runs (each final
checkpoint 26.35 GB; the transient save copy is not kept).

## 4. Training facts and per-question curves (`artifacts/scratch/r12/c-facts-s{18,17,19}.json`, from each run's `metrics.json`)

| | seed 18 | seed 17 | seed 19 |
| --- | ---: | ---: | ---: |
| status / step | completed 233/233 | completed 233/233 | completed 233/233 |
| git (dirty) | `fb635d7` (False) | `fb635d7` (False) | `fb635d7` (False) |
| realized robot units: expert · DAgger | 113 · 120 | 116 · 117 | 111 · 122 |
| DAgger draws: dagger-0 · dagger-1 · done gate · **dagger-2** | 29 · 20 · 24 · **47** | 31 · 23 · 16 · **47** | 26 · 24 · 22 · **50** |
| epochs of pool: expert · dagger-2 | 0.485 · 0.117 | 0.498 · 0.117 | 0.476 · 0.125 |
| drawn episodes with a raw false done: dagger-1 · done gate · dagger-2 (of 22 · 21 · 51) | 1 · 5 · 6 | 1 · 1 · 5 | 1 · 2 · 5 |
| drawn episodes with a stale-goal event: dagger-1 · done gate · dagger-2 (of 13 · 15 · 40) | 1 · 2 · 4 | 1 · 1 · 4 | 1 · 1 · 4 |
| non-robot bundles | 1376 | 1380 | 1388 |
| loss step 1 → 233; mean first 20 / last 20 | 2.6331 → 0.2214; 1.329 / 0.265 | 2.7860 → 0.1177; 1.202 / 0.261 | 2.8076 → 0.3334; 1.288 / 0.248 |
| `q_gripper` loss / baseline: 131–150 · 214–233 | 0.969 · 0.246 | 1.133 · 0.817 | 0.804 · 0.221 |
| R8 monitor statistic at step 150 (log only) | 0.969 → not_fitting | 1.133 → not_fitting | 0.804 → not_fitting |
| gripper fit onset (registered definition) | 179 | None | 181 |
| training-batch settled probe 214–233 | 326/334 = 0.976 | 0/621 = 0.000 | 503/517 = 0.973 |
| `q_stop` loss / baseline 214–233; stop-true ticks fired / stop-true ticks, whole run | 1.000; 0/277 | 2.110; 0/244 | 1.373; 0/146 |
| `q_done` loss / baseline: 131–150 · 214–233 | 0.047 · 0.527 | 0.062 · 0.021 | 1.081 · 0.021 |
| `q_main` loss / baseline 214–233 | 0.102 | 0.040 | 0.032 |
| unit wall · load · train seconds | 14,190.6 s · 199.7 s · 13,375.3 s | 14,329.9 s · 199.9 s · 13,521.3 s | 14,376.4 s · 201.8 s · 13,575.1 s |
| step seconds mean / p50 / max | 57.4 / 48.36 / 150.87 | 58.03 / 48.53 / 124.6 | 58.26 / 49.18 / 219.15 |
| GPU peak allocated / reserved; OOM · retries | 56.31 / 57.97 GiB; 0 · 0 | 56.31 / 57.97 GiB; 0 · 0 | 56.01 / 57.66 GiB; 0 · 0 |
| host RSS after load → end | 18.18 → 19.04 GiB | 18.2 → 19.08 GiB | 18.18 → 19.04 GiB |
| items loaded (stream + single) | 3,853 (1,233 + 2,620) | 3,853 (1,233 + 2,620) | 3,853 (1,233 + 2,620) |
| tokens (robot · non-robot) | 11,535,260 (9,794,812 · 1,740,448) | 11,743,308 (10,010,897 · 1,732,411) | 11,807,961 (10,089,802 · 1,718,159) |
| resume | none | none | none |

Ratios are the per-question weighted loss over the constant-prior baseline, averaged over the window (R8 A1's instrument); probes are the
training batch's argmax counts. Reading, per question (descriptive — three seeds are not a sample a recipe interval can be computed from):

* **Material**: the 0.5 share drew 117–122 DAgger units per run, of which **dagger-2 47 · 47 · 50** (≈ 0.12 epoch of its 400 episodes, as
  designed: 0.5 × 233 × 400/1,000 ≈ 47); expert exposure fell to 0.48–0.50 epoch, as in r6. Of dagger-2's 40 episodes with a stale-goal event,
  **4 were drawn in each run** (of its 51 with any raw false done, 6 · 5 · 5).
* **`q_gripper`**: the ≈ 100-step flat phase at the constant-prior level is there in all three seeds (ratio 1.0–1.3 over steps 21–140). It
  then fit late in seeds 18 and 19 (registered onset **179** and **181**; batch `settled` 0.976 / 0.973 at the end) and **never fit in seed
  17** (onset none; `settled` 0/621 over steps 214–233, ratio 0.817 — the R7 / R9-seed-18 shape). R8's step-150 statistic, recomputed for the
  record only, reads 0.969 · 1.133 · 0.804 — all three above R8's 0.70, so under R8's rule all three would have been stopped; here nothing
  stopped (registered log-only). Under arm G the gripper is the execution rule's, so seed 17's head matters only for its native arm.
* **`q_stop`**: no run ever answered `true` on a stop-true training tick (0/277 · 0/244 · 0/146), and its loss sits at or above the
  constant-prior baseline at the end (1.000 · 2.110 · 1.373) — the same failure as R6–R9.
* **`q_done`**: fit in all three (end-of-run ratio 0.527 · 0.021 · 0.021; seed 18's last windows carry two high-loss steps, its 151–190 ratio
  0.06–0.20).
* **`q_main`**: fit early in all three (ratio ≤ 0.24 from step 21–40 on; 0.10 · 0.04 · 0.03 at the end).

## 5. The choice, the registered verdict and the descriptive readings (Stage D)

**Runs, in the registered order, all after the three trainings.** GPU chain `r12-d-dev` (`artifacts/scratch/r12/d-dev.sh` → `d-model.sh`, one
process per candidate, each with a page-cache drop, the 0.6 fence and `require_free(36 GiB)`; 02:21:05 → 03:35:38): the six candidates on
dev_new2 100 (R6's seeds; R10/R11's dev_new2 command with `--id-tag r12`) → CPU unit `r12c-d-select` (03:35:55 → 03:36:43; `report --runs
<the registration's selection_sources + rule, ruleG, expert, r6G on dev_new2> --only dev_new2` → `artifacts/reports/r12-selection-report.json`;
`select --registration configs/eval/r12-registration.yaml` → `artifacts/reports/r12-selection.json`) → CPU units `r12c-d-new-{rule,expert,ruleG}`
on ood_new (03:36:53 → 03:43:44, in parallel, GPU idle) → GPU chain `r12-d-judged` (from 03:43:50, no CPU unit running): the chosen
configuration, then r6G, on ood_new 200.

### The choice (dev_new2 100, the registered rule applied by code)

| configuration | checkpoint · arm | `done` | **strict** | false done |
| --- | --- | ---: | ---: | ---: |
| **r12s19G** | r12s19 · G | 73 | **69** | 4 |
| r12s17G | r12s17 · G | 69 | 65 | 4 |
| r12s18G | r12s18 · G | 77 | 62 | 15 |
| r12s19 | r12s19 · native | 61 | 58 | 3 |
| r12s18 | r12s18 · native | 56 | 44 | 12 |
| r12s17 | r12s17 · native | 0 | 0 | 0 |

All six eligible (the three training runs completed at 233/233; no run applied a monitor). **Chosen: r12s19G**, decided by the first key
(69 vs 65); no tie rule was needed. r12s17 native completes nothing — its gripper head never fit (§4) — while the same checkpoint in arm G is
second. Beside, on the same scenes (not candidates): the rule judge 67 (native and G), the expert 90 and r6G 74 (R10's exploratory dev_new2
run) — the best new configuration is 5 below r6G on validation.

### The registered verdict (ood_new 200; `artifacts/reports/r12-verdict.json`)

Final report `artifacts/reports/r12-closed-loop.json` (CPU unit `r12c-d-final`, `artifacts/scratch/r12/final-reports.sh`, 04:30:57 → 04:41:48:
`report --runs <six selection sources + rule, ruleG, expert, r6G on dev_new2 + the five ood_new runs> --only dev_new2,ood_new --seed-pairs
ruleG:r12s19G r12s19G:ruleG r12s19G:r6G --robustness-metrics forbidden_contacts` → `verdict --registration configs/eval/r12-registration.yaml`).
The verdict re-chose r12s19G from the final report's dev_new2 table — identical to `select`'s choice (the chain asserts it).

| condition | pair | difference [95 %] | discordant · exact test | RNG seeds 1–200 | holds if | holds? |
| --- | --- | --- | --- | --- | --- | --- |
| (a) strict success | `ruleG − r12s19G` | **+0.080 [+0.025, +0.135]** (0.760 vs 0.680; 152 vs 136) | 25 vs 9 · McNemar p = 0.009 | lower ≤ 0 in 0 of 200 (min +0.020) | lower ≤ 0 | **no** |
| (s) forbidden-contact onsets / episode | `r12s19G − ruleG` | **+0.005 [+0.000, +0.015]** (0.020 vs 0.015; 4 vs 3 onsets) | seeds higher 1 vs 0 · sign test p = 1.000 | lower ≤ 0 in 200 of 200 | lower ≤ 0 | **yes** |

**Registered call: fail — "불통과 — 사이클 2 뒤에도 검증으로 고른 구성은 처음 보는 홀드아웃에서 규칙 판정기에 확실히 뒤지거나 금지 접촉이 확실히
많다"** (`failed: ['a']`): the chosen configuration still clearly trails the rule judge in its arm on strict success; on forbidden contacts these
200 episodes do not separate the two (that is not "equal"). dev_new2 (the choice set, beside only): (a) −0.020 [−0.120, +0.070] (11 vs 13),
(s) +0.000 [−0.030, +0.030].

### Second evidence and descriptive readings (not used to judge)

| policy (ood_new 200) | `done` | **strict** [95 %] | false done (stale goal) | failures main / aux / geometric | first grasp on a non-instructed object | `q_stop` caught / onsets | forbidden-contact onsets (episodes) | unsafe ticks / acted |
| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| **r12s19G** (chosen) | 141 | **136** [0.610, 0.745] | **5 (3)** | 23 / 32 / 4 | 4 | 0/52 | **4** (4) | 0.0120 |
| ruleG (same arm) | 153 | **152** [0.700, 0.815] | 1 (0) | 0 / 9 / 38 | 0 | 43/50 | 3 (3) | 0.0000 |
| rule (native) | 154 | 153 [0.705, 0.820] | 1 (0) | 0 / 45 / 1 | 0 | 40/47 | 3 (3) | 0.0000 |
| expert | 182 | 181 [0.860, 0.945] | 1 (0) | 0 / 1 / 17 | 0 | 53/53 | 4 (4) | 0.0000 |
| **r6G** (before cycle 2) | 168 | **139** [0.630, 0.760] | **29 (16)** | 18 / 10 / 4 | 16 | 1/79 | **17** (9) | 0.0348 |

(Failure classes and first grasps by `artifacts/scratch/r12/failures.py` → `failures.json`; stale goal by the seed metric and R9's
`false_done_mechanism.py` → `artifacts/scratch/r12/false-done-mechanism.json`, which agree; tables `artifacts/scratch/r12/d-tables.md`.)

* **Second evidence — the chosen configuration against the best configuration before cycle 2 (`r12s19G − r6G`, seed-paired on ood_new)**:
  strict **−0.015 [−0.070, +0.045]** (16 vs 19 discordant, p = 0.74 — no difference found); false dones **−0.120 [−0.170, −0.075]** (1 vs 25
  seeds, p = 8 × 10⁻⁷); stale-goal false dones **−0.065 [−0.100, −0.035]** (3 vs 16); forbidden-contact onsets **−0.065 [−0.155, −0.005]** per
  episode (4 vs 17 — 7 of r6G's 17 on one seed, E2:1171395; by seed 1 vs 6, sign test p = 0.125; the upper bound stays below zero under all 200
  RNG seeds — the per-seed sign test does not reach 0.05); beside: auxiliary failures +0.110 [+0.055, +0.165], `stall_exhausted` endings 58 vs
  31, **+0.135 [+0.075, +0.195]**, main-decision failures +0.025 [−0.025, +0.080] (no difference found), missing reference gripper transitions
  +0.139 [+0.030, +0.236], wrong-action ticks −0.046 [−0.078, −0.015], unsafe ticks −0.023 [−0.047, −0.001]. On dev_new2 the same pair reads
  strict −0.050 [−0.150, +0.050], false dones −0.110 [−0.180, −0.050], stale-goal −0.090 [−0.150, −0.040]. *(Fix round 1, review I2 — what this
  compares:)* one configuration — training seed 19, the best of six on dev_new2 — against r6G, training seed 18. Seed-paired with r6G on
  dev_new2 in the same arm (`artifacts/scratch/r12/fix1/verify.json`), the false-done reduction appears for **seed 17** (4 vs 15, −0.11 [−0.17,
  −0.05]; stale-goal 4 vs 12, −0.08 [−0.14, −0.03]) and **seed 19** (4 vs 15, −0.11 [−0.18, −0.05]; stale-goal 3 vs 12) but **not for the seed-18
  retrain** (15 vs 15, 0.00 [−0.05, +0.05]; stale-goal 13 vs 12, +0.01 [−0.02, +0.04]; its strict success fell, −0.12 [−0.20, −0.05], 2 vs 14
  discordant, p = 0.004); forbidden contacts show no difference on dev_new2 for any of the three (3 vs 3, 3 vs 3, 1 vs 3). The reduction is seen
  in two of three seeds; with the material, the training seed and the selection bound together it is not shown to be an effect of cycle 2.
* **The 25 seeds behind the strict gap** (`artifacts/scratch/r12/discordant.json`): the seeds only ruleG solved are r12s19G stalls — 14 after an
  auxiliary disagreement (13 with a `q_path` streak; one, E1:1160457, with only `q_force` and `q_gripper` streaks — fix round 1, review M8; over
  all 32 auxiliary failures the streaks are `q_path` 29 · `q_speed` 10 · `q_force` 3 · `q_gripper` 2), 10 after a main-decision disagreement (9
  `stall_exhausted`, 1 `max_ms`) and 1 geometric — **none is a false done**; the 9 seeds only r12s19G solved are all rule-judge geometric
  stalls. Of r12s19G's 64 failures: 32 auxiliary, 23 main-decision, 5 false dones (3 stale goal, 1 other, 1 the shared edge below), 4
  geometric. By profile (strict, r12s19G vs ruleG): E0 39 vs 40, E1 67 vs 69, **E2 30 vs 43**; seed-paired `ruleG − r12s19G` (fix round 1,
  review I4): E2 **+0.163 [+0.038, +0.288]**, E0 + E1 +0.025 [−0.025, +0.075], the difference of the two gaps **+0.138 [+0.008, +0.258]**
  (stratified bootstrap — E2 and E0 + E1 seeds resampled separately, 2,000 resamples, RNG seed 20260921); 19 of the 25 ruleG-only seeds are E2.
  Against r6G the 16 r12s19G-only seeds are r6G false dones (9 — 3 stale goal, 5 another non-target, 1 other) and stalls (7); the 19 r6G-only
  seeds are r12s19G stalls or timeouts (18) and 1 false done — the chosen configuration's wins over r6G are mostly r6G false dones and its losses
  its own stalls.
* Seed-paired `r12s19G − ruleG` beside the verdict: false dones **+0.020 [+0.005, +0.040]** (5 vs 1; 4 vs 0 seeds — the false-done rate still
  differs, although none of the 25 ruleG-only seeds is a false done: r12s19G's four false dones besides the shared edge fall on seeds where ruleG
  also failed, 3 geometric and 1 auxiliary stall, so turning them into successes would move (a) only from +0.080 to +0.060 — fix round 1, review
  I3), main-decision failures **+0.115 [+0.075, +0.160]**, auxiliary failures +0.115 [+0.065, +0.165], wrong-action ticks +0.032 [+0.018,
  +0.048], unsafe ticks +0.012 [+0.004, +0.023], `q_stop` catch **−0.860 [−0.952, −0.755]**; `rule − r12s19G` strict +0.085 [+0.030, +0.140];
  `expert − r12s19G` +0.225 [+0.170, +0.285]; `ruleG − rule` −0.005 [−0.015, 0.000] (zero inside; fix round 1, review M3 — the first version
  printed a sign-flipped −0.000).
* **The three seeds** (`distribution`; strict on dev_new2, native · G): seed 18 44 · 62 (false dones 12 · 15 — 13 of the G arm's 15 are stale
  goal), seed 17 **0** · 65 (its gripper head never fit — every native episode ends `stall_exhausted`; delegation rescues it), seed 19 58 · 69 (3
  · 4). The false-done reduction is seed-dependent: seeds 17 and 19 under G have 4 false dones each on dev_new2 (each −0.11 against r6G's 15),
  seed 18 under G 15 (0.00 [−0.05, +0.05]). Only r12s19G ran on ood_new (136 strict, 5 false dones).
* **One false done is shared by every policy, the expert included**: E1:1161544 — the first instruction (o6 → zoneL) is complete at the done
  tick (71) and at the last tick, but the second instruction is scheduled at 7,420 ms, inside the 0.3 s done tail that ends the episode at
  7,500 ms, so the outcome is scored against the new target (o0) — the same generator timing edge as R11's E1:1110410
  (`artifacts/scratch/r12/check/edge_probe.py`, unit `r12c-d-edge-probe`). All five policies end it at tick 71, so it cancels in every
  seed-paired comparison; nothing was changed.

**Latency** (`latency.model_ms` excludes each episode's first tick; no CPU unit ran during either judged GPU run): r12s19G on ood_new non-first
ticks p50 / p95 / p99 **43.4 / 56.4 / 81.5 ms**, 1 of 18,070 over 100 ms, observation → command p95 62.8 ms; first ticks p50 93.0, max 270.9 ms,
18 of 200 over 100 ms. r6G on ood_new 43.7 / 56.6 / 81.6 ms, 0 of 19,467 (first ticks 24 of 200). The six dev_new2 runs: p95 56.8–57.9 ms, 2 of
57,374 non-first ticks over 100 ms (both r12s17G), first ticks 10–22 of 100 over 100 ms. The gate (p95 ≤ 80 ms, > 100 ms ≤ 5 %) passes.

**Records.** All eleven new closed-loop directories (`artifacts/datasets/r12-closed-loop/`, 1.5 GB) carry one version set — harness `h0.9`,
controller `c0.6`, expert `e0.4`, rule judge `rj0.5`, generator `gen-robot-v0.3`, generator-config digest `c297ada2e90a` (R10/R11's) — and every
model run file records contract `93fe26725a4c`, serializer `ts0.6`, fence 0.6, git `fb635d7` (the registration commit). The four checkpoints
read (r12s18, r12s17, r12s19, r6) have the same size and mtime before their first load and after the last report
(`artifacts/scratch/r12/checkpoints-stat-{before,after}.txt`).

## 6. Measured cost

| unit (systemd, `Started` → end) | what | wall |
| --- | --- | ---: |
| `r12-a-smoke` | GPU: collection smoke, 2 episodes | 0 h 00 m 43 s |
| `r12-a-collect` | GPU: DAgger cycle 2 collection, 400 episodes (r6G under the done gate) | 0 h 52 m 27 s |
| `r12-t1-s18` | GPU: training, seed 18, 233 steps | 3 h 56 m 34 s |
| `r12-t1-s17` | GPU: training, seed 17, 233 steps | 3 h 58 m 54 s |
| `r12-t1-s19` | GPU: training, seed 19, 233 steps | 3 h 59 m 41 s |
| `r12-d-dev` | GPU: the six candidates on dev_new2 (100 each) | 1 h 14 m 33 s |
| `r12-d-judged` | GPU: r12s19G and r6G on ood_new (200 each) | 0 h 46 m 45 s |
| `r12-pytest` | GPU: full suite (fence 0.6 + `require_free(36 GiB)` checked first; nothing else on the GPU) | 0 h 09 m 38 s |
| **GPU total** | | **53,955 s = 14.99 GPU-h** (brief ≈ 15 h) |

CPU units (`r12c-*`, CUDA hidden, MemoryMax, `choom -n 1000`, Nice 10): the two seed walks and their re-checks (5–30 s each), the two overlap
reports, the smoke checks, the stale-goal regression on R11's records (13 s), the relabel / QA / A3 tables / strata / ood_new overlap chain
(`r12c-a3`, 1 m 38 s), the training facts (`r12c-c-facts-*`, ≈ 15 s per seed), the choice (`r12c-d-select`, 48 s), rule / expert / ruleG on
ood_new (6 m 34 s – 6 m 51 s, in parallel, GPU idle), the final report and verdict (`r12c-d-final`, 10 m 51 s, MemoryMax 40G — it did not
overlap any GPU unit), the discordant and edge probes, and the development test units (`r12c-pt-*`). Disk: `/` 134 → 52 GB free (10⁹ B; three
checkpoints 3 × 26.35 GB, dagger-2 421 MB, closed-loop records 1.5 GB, scratch 617 MB). `journalctl -k` since 2026-10-07 13:00: 49 kernel lines,
no `NVRM`, OOM or killed-process line (the same query shows R9's seven `NVRM` lines of 2026-09-30). Cumulative GPU on the DGX Spark ≈ 82.0 h.
Cloud spend 0.

**Every CPU job that read records ran as a unit (HANDOFF §5).** Outside units the session shell read only tracked files, scripts, logs, report
JSONs and dataset-manifest metadata — never an episode record; the list with times, including the reads that happened while a GPU unit was up
(report JSONs during `r12-a-smoke`, `r12-a-collect`, `r12-t1-s18`, `r12-t1-s17`, `r12-t1-s19` and `r12-d-dev`; two manifests' first `files`
entry keys during `r12-a-collect`; `metrics.json` git/status fields; during `r12-pytest` the verdict JSON, the eleven R12 closed-loop manifests
and the `versions` of three manifests — R12's r12s19G/ood_new again, R11's r6G/ood_fresh and dagger-2), is
`artifacts/scratch/r12/session-shell-reads.md` — completed in fix round 1 (review M1: the first version stopped at 22:27 and so missed
22:37 during `r12-t1-s19`, 02:21 during `r12-d-dev` and 04:42–04:43 during `r12-pytest`; it now runs to the handback).

## 7. What the round says, and what it does not

1. **Registered call: fail — "불통과 — 사이클 2 뒤에도 검증으로 고른 구성은 처음 보는 홀드아웃에서 규칙 판정기에 확실히 뒤지거나 금지 접촉이 확실히
   많다", on condition (a) only.** The configuration chosen on validation (r12s19G, 69 of 100 on dev_new2, the highest of six) trails the rule
   judge in its arm on 200 scenes no policy had run — `ruleG − r12s19G` strict **+0.080 [+0.025, +0.135]** (152 vs 136; 25 vs 9 discordant, p =
   0.009; above zero under all 200 alternative bootstrap seeds). Condition (s) holds: forbidden-contact onsets `r12s19G − ruleG` **+0.005
   [+0.000, +0.015]** per episode (4 vs 3) — these 200 episodes do not show r12s19G making clearly more forbidden contacts, which is not "equal".
2. **The chosen configuration made fewer false dones, stale-goal false dones and forbidden contacts than r6G, and no difference in strict
   success was found — the reduction is not shown to be an effect of cycle 2 as such** (rewritten in fix round 1, reviews I1 and I2; the
   first version read "Cycle 2 did what it aimed at, and the success count did not move"). Against r6G on the same 200 scenes (second
   evidence, seed-paired): false dones −0.120 [−0.170, −0.075] (5 vs 29), stale-goal false dones −0.065 [−0.100, −0.035] (3 vs 16),
   forbidden-contact onsets −0.065 [−0.155, −0.005] (4 vs 17 — 7 of r6G's 17 on one seed; by seed 1 vs 6, sign test p = 0.125), unsafe ticks
   −0.023 [−0.047, −0.001]; strict success −0.015 [−0.070, +0.045] (136 vs 139) — no difference found. Where it did not end falsely it more
   often stalled: auxiliary failures +0.110 [+0.055, +0.165] (32 vs 10), `stall_exhausted` 58 vs 31, +0.135 [+0.075, +0.195]; seed by seed,
   its 16 wins over r6G are r6G false dones (9) and stalls (7), its 19 losses its own stalls or timeouts (18) and one false done. This is one
   configuration (training seed 19, the best of six on dev_new2) against r6G (training seed 18). On dev_new2, seed-paired with r6G in the same
   arm, the false-done reduction appears for seeds 17 and 19 (−0.11 each) and not for the seed-18 retrain (15 vs 15, 0.00 [−0.05, +0.05]; its
   strict success fell, −0.12 [−0.20, −0.05]); forbidden contacts show no difference there for any seed. Seen in two of three seeds.
3. **The 25 seeds behind the strict gap to the rule judge are stalls, not false dones — the false-done rate still differs** (descriptive;
   rewritten in fix round 1, review I3: the first version said "the remaining gap … is stalls, not false dones"). The 25 seeds only the rule
   judge solved are r12s19G stalls after auxiliary disagreements (14 — 13 with a `q_path` streak, one with only `q_force` and `q_gripper`
   streaks) and after main-decision disagreements (10; one ran out of time, `max_ms`), plus 1 geometric — none is a false done; the 9 seeds only
   r12s19G solved are rule-judge geometric stalls. Seed-paired against ruleG: main-decision failures +0.115 [+0.075, +0.160], auxiliary +0.115
   [+0.065, +0.165], and **false dones +0.020 [+0.005, +0.040]** (5 vs 1) — r12s19G's four false dones besides the shared edge fall on seeds
   the rule judge also failed, so turning them into successes would move (a) only from +0.080 to +0.060. The strict gap is in E2: +0.163
   [+0.038, +0.288] (43 vs 30) against +0.025 [−0.025, +0.075] on E0 + E1 (109 vs 106 — E0 40 vs 39, E1 69 vs 67), the difference
   +0.138 [+0.008, +0.258] (stratified bootstrap); 19 of the 25 rule-only seeds are E2.
4. **The stop is unchanged**: `q_stop` caught 0 of 52 stop onsets (the rule judge 43 of 50; `−0.860 [−0.952, −0.755]`), and in training no seed
   ever answered `true` on a stop-true tick. Arm G delegates only the gripper; the stop is still the model's.
5. **The three seeds** (descriptive, three is not a sample a recipe interval can be computed from): seed 17's gripper head never fit (native 0
   of 100 on dev_new2; under G 65), seed 18's and 19's fit late (onset 179 · 181) — R8's step-150 statistic would have stopped all three; the
   false-done reduction is seed-dependent (dev_new2 false dones under G: seed 18 15, seeds 17 and 19 4 each; r6G 15 — −0.11 for seeds 17 and 19,
   0.00 [−0.05, +0.05] for seed 18; fix round 1, review I2). Validation chose seed 19's G arm, 69, below r6G's 74 on the same set.
6. **A scene-timing edge of the generator, again**: E1:1161544's second instruction is scheduled inside the done tail, so a completed first
   instruction is scored against the new target — a false done for all five policies. It cancels in every seed-paired comparison; the harness
   and generator were not touched.
7. **What the round does not say.** Only the chosen configuration (and r6G, as the registered second evidence) ran on ood_new, so the other five
   configurations are not ranked there; one holdout of 200; one recipe at three seeds; the rule judge reads the structured goal while the model
   reads the instruction text. Whether cycle 2 caused the chosen configuration's fewer false dones and contacts is not shown — one configuration
   against r6G, and the seed-18 retrain shows no false-done reduction on dev_new2 (fix round 1, review I2; the first version said "the false dones
   and contacts it aimed at fell"); what the round shows is that the chosen configuration does not yet match the rule judge on success.

**Next round (registered: written from the numbers only).** The strict gap that remains is the stalls after auxiliary (mostly `q_path`) and
main-decision disagreements — all 25 seeds the rule judge won — concentrated in E2 (+0.163 [+0.038, +0.288] against +0.025 [−0.025, +0.075]
on E0 + E1), with `q_stop` still catching nothing. None of those 25 seeds ends in a false done, but the false-done rate still differs (+0.020
[+0.005, +0.040], 5 vs 1); forbidden contacts are not separated on this holdout (+0.005 [+0.000, +0.015] — which is not "equal"). Which part of
the chosen configuration's lower false-done count is cycle 2's is open: the seed-18 retrain shows none on dev_new2 (fix round 1, reviews I2
and I3). ood_new (1150100~) has now been analysed seed by seed, so the next
verdict needs another unused seed walk. No contract-change proposal (delegation G into the harness) is made by this round, and nothing here
argues for cloud spend.
