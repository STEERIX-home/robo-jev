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
| of which **stale-goal** (at the event's first tick: the last grasp commitment before it was on an object that was not that grasp tick's target and had been an earlier instruction's target — R9 fix round 1's definition) | **41 events in 40 episodes** (E2 26, E1 14); the other 14: 6 after a last grasp on another non-target, 8 after a last grasp on the current target |
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

