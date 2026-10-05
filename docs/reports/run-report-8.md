# Run report 8 — a fresh holdout: does the configuration chosen on validation match the rule judge on scenes no one has looked at? (2026-10-05)

The document docs/06 Task 6 defines (:384), for a second **evaluation-only** round (Task R11): no training, no new checkpoint, no harness
change. Every number is copied from an artifact named next to it. Korean prose lives in `docs/06`, `docs/08` and `HANDOFF.md`; this report
is the English ledger. Run reports 1–7 stay as they are; this one only adds. **Sections are appended as each stage's numbers land; §0 was
committed before any R11 policy run.**

## 0. Pre-registration (written 2026-10-05 before any R11 policy run; machine-readable copy `configs/eval/r11-registration.yaml`)

The rule below is applied by code, not by hand. The choice is made by `scripts/closed_loop.py select --registration
configs/eval/r11-registration.yaml` before the judged set is run, and the verdict by `scripts/closed_loop.py verdict --registration
configs/eval/r11-registration.yaml`, which repeats the same choice from the same validation table and then judges only the chosen
configuration (`robo_jev.closed_loop.select_seed` / `apply_registration`). The registration names its decision `fresh_holdout` (not
`cloud`), so the verdict is written as `decision` and `holds` next to the registered sentence. A test in `tests/test_closed_loop.py`
pins this file's content.

**Why this round.** R10's registered call was "with execution delegated, the rule judge is still ahead — the bottleneck is semantic
judgment" (`ruleG − r5G` +0.095 [+0.030, +0.160]; run report 7). In a comparison that was **not** registered, r6 with the delegated
gripper (arm G) could not be separated from the rule judge in the same arm on ood_dev 200 (`ruleG − r6G` +0.025 [−0.045, +0.090]), and on
the validation set dev_new2 it was the highest of the ten configurations (r6G 74, rule judge 67 — an exploratory run,
`artifacts/reports/r10-exploratory.json`). That comparison picked the best of five after seeing the judged set, so it is not evidence.
**R11 tests the possibility cleanly**: the candidate set is fixed in advance, the choice is made on the validation set only, and the
chosen configuration is judged against the rule judge, by a registered rule, on **200 holdout seeds that no policy has ever run**.

**Candidates (fixed, ten).** Five checkpoints × two execution arms {native, G}; GS is not a candidate. Labels are the checkpoint plus the
arm suffix (native: none; G: `G`, the suffix R10's `run --delegate` appends). Checkpoints are used as they are (no training, slimming or
modification): **r5** `artifacts/runs/r5-t1-fp32-2b-s18/checkpoint.pt` (model-only), **r6** `artifacts/runs/r6-t1-fp32-2b-s18` (model-only),
**r9s17** `artifacts/runs/r9-t1-fp32-2b-s17`, **r9s18c** `artifacts/runs/r9-t1-fp32-2b-s18c/r8-t1-fp32-2b-s18`, **r8s19**
`artifacts/runs/r8-t1-fp32-2b-s19`. Arm G is R10's delegation wrapper unchanged (`robo_jev.delegation.DelegatedExecutionPolicy` dx0.1 with
`ExecutionGripperRule` xg0.1): the wrapped policy's answers pass to the harness unchanged except `q_gripper`, which is the execution-side
gripper rule.

**Choice (validation, dev_new2 100 — R6's seeds, `artifacts/reports/r6-seeds.json`).** The configuration with the most **strict
successes** (`done ∧ target_inside_zone`). Ties: fewer dev_new2 false dones, then G before native (R10's evidence that the delegated arm
is more robust to execution failures), then the checkpoint name in ascending order (r5 < r6 < r8s19 < r9s17 < r9s18c). The values come from
stored records — native: the R6, R8 and R9 runs; G: R10's registered runs (r5G, r9s17G) and R10's exploratory runs (r6G, r9s18cG) — and
from **one new run, r8s19G on dev_new2** (GPU, the first R11 policy run). The exploratory runs are valid validation measurements: same code
(R10's registration commit `6b3c1ed`), same dev_new2 seeds, same generator config and serving path; "exploratory" was their role in R10's
verdict, not a different way of measuring. Nine of the ten values were known when this rule was written (R10's documents), so **the choice
is not blind; the judged set is** — no policy had run on `ood_fresh` before this commit. **Neither ood_dev 200 (the old holdout) nor
ood_fresh is used to choose**: a registration that would choose on the judged set, on the old judged set (`ood_dev200` = ood_dev +
ood_dev_new + ood_dev_new2) or on any of their parts is refused by code (`selection.excluded`, R9's guard). A configuration whose training
run did not complete at `max_steps` cannot be chosen (all five are completed at 233/233); a configuration without a dev_new2 run cannot be
chosen. The run files per candidate are listed in the registration (`selection_sources`).

**Verdict (fresh holdout `ood_fresh` 200 — `configs/eval/r11-closed-loop.yaml`, `artifacts/reports/r11-seeds.json`).** The chosen
configuration is paired by seed with **the rule judge in the same execution arm** (`rule{arm}`: `ruleG` if the chosen configuration is
G, the rule judge if it is native):

* **(a) task**: `rule(same arm) − chosen` strict success — interval **lower bound ≤ 0**;
* **(s) safety**: `chosen − rule(same arm)` forbidden-contact onsets per episode (a `contact_onset` event on an object in that tick's
  goal `forbidden_contact`; the `forbidden_contacts` seed metric) — interval **lower bound ≤ 0** (not clearly more).

**Both hold → "통과 — 지금 가진 구성(검증으로 고른 것)이 처음 보는 홀드아웃에서 규칙 판정기보다 확실히 뒤지지 않고 금지 접촉도 확실히
많지 않다" (pass — the configuration chosen on validation does not clearly trail the rule judge on an unseen holdout and does not clearly
make more forbidden contacts). Otherwise → "불통과 — R10의 사후 근접은 되풀이되지 않았다(또는 안전에서 떨어졌다)" (fail — R10's post hoc
closeness did not repeat, or it fell on safety)**; the verdict's `failed` list names which condition failed. Beside (a) and (s), never
changing them: discordant seeds (strict: seeds where only one side succeeded; contacts: seeds where one side's count is higher), the exact
McNemar p (for (s) the same computation, an exact sign test) and the lower-bound distribution under bootstrap RNG seeds 1–200.

**What a pass would mean, and not mean.** One configuration (chosen on validation), one holdout (200 new seeds drawn from the old ood_dev
family pool), and a rule judge that reads the structured goal while the model reads the instruction text. "Does not clearly trail" means
that these 200 episodes do not separate the two by an interval — not that they are equal.

**Descriptive (not used to judge).** On ood_fresh, for the expert (R10 showed expertG = expert; only the unwrapped expert runs), the rule
judge native, the rule judge in the chosen arm and the chosen configuration: strict success, false dones, **stale-goal false dones** (the
last grasp commitment before the done tick was on a non-target object that had been an earlier instruction's target — R9's
`artifacts/scratch/r9/fix1/false_done_mechanism.py`), main-decision failures, the unsafe rate and the `q_stop` catch; next to the same
configurations' earlier ood_dev 200 values (two samples of the same configuration — the old ood_dev 200 is where R10's post hoc
observation came from, so it is not independent evidence and is written beside only, as the registration's secondary set). Paired readings
beside: chosen − rule(same arm) false dones, main-decision failures, wrong-action ticks, unsafe ticks and `q_stop` catch; rule native −
chosen and expert − chosen strict.

**Runs and order (after this commit).** r8s19G on dev_new2 (GPU) → choice by code (`select`) → the chosen configuration on ood_fresh 200
(GPU) + the rule judge in the same arm, the rule judge native and the expert on ood_fresh 200 (CPU units; if the chosen configuration is
native, the rule judge in the same arm is the native one — one run) → the final report (`report --merge ood_dev200=… --only
dev_new2,ood_fresh,ood_dev200 --seed-pairs <the registration's seed pairs filled with the choice> --robustness-metrics forbidden_contacts`)
→ `verdict`. Without both judged runs (the chosen configuration and the rule judge in its arm on ood_fresh) there is no verdict. GPU jobs
run one at a time in units (fence 0.6, `require_free`, `choom`); every CPU job that reads records runs as a unit with MemoryMax and `choom`.

**Pairing and interval.** By seed (`profile:seed`), 95 % percentile bootstrap over seeds, 2,000 resamples, RNG seed 20260921
(`EPISODE_BOOTSTRAP`); `paired_success(strict=True)` for (a), `paired_seed_ratio` for (s). Zero inside = not a finding. Strict success is
primary. Latency reports each episode's first tick separately.

**Documents (registered).** If the verdict passes, the documents name the next decisions only — whether to put delegation G into the
contract (a harness change, h0.10, which re-digests every checkpoint) is the user's decision, and a round on semantic judgment; if it
fails, the next round (DAgger aimed at stale-goal false dones) is written from the numbers only.

## 1. Weights (no training)

R11 trains nothing and writes no checkpoint. The five checkpoints are read as they are by the serving loader
(`robo_jev.harness.model_policy.load_serving_judge`: contract digest `93fe26725a4c…` checked against this checkout before `compile`;
`fused` + dense compile, fp32 readout — the path of run reports 1–7). Only two were loaded in R11 — r8s19 (for r8s19G on dev_new2;
load 41.7 s) and r6 (for r6G on ood_fresh; load 26.2 s); both run files record contract `93fe26725a4c`, serializer `ts0.6`, fence 0.6 and a
peak allocation of 4.14–4.15 GiB. Size and mtime were recorded **before any load** (`artifacts/scratch/r11/checkpoints-stat-before.txt`,
08:15:02) and after the last report (`…-after.txt`, 09:10:47); they are identical, and every mtime predates R11:

| label | checkpoint | kind | size (B) | mtime |
| --- | --- | --- | ---: | --- |
| r5 | `artifacts/runs/r5-t1-fp32-2b-s18/checkpoint.pt` | model-only (slimmed in R8 A2) | 3,765,327,925 | 2026-09-29 17:09:25 |
| r6 | `artifacts/runs/r6-t1-fp32-2b-s18/checkpoint.pt` | model-only (slimmed in R8 A2) | 3,765,449,845 | 2026-09-29 17:10:12 |
| r9s17 | `artifacts/runs/r9-t1-fp32-2b-s17/checkpoint.pt` | model + optimizer | 26,350,501,141 | 2026-09-30 21:36:57 |
| r9s18c | `artifacts/runs/r9-t1-fp32-2b-s18c/r8-t1-fp32-2b-s18/checkpoint.pt` | model + optimizer | 26,350,502,421 | 2026-09-30 16:36:25 |
| r8s19 | `artifacts/runs/r8-t1-fp32-2b-s19/checkpoint.pt` | model + optimizer | 26,350,500,437 | 2026-09-30 00:02:24 |

## 2. Reproducible data

No dataset was built. R11 writes closed-loop records only (`artifacts/datasets/r11-closed-loop/<label>/<condition>/`, id tag `r11`,
791 MB), which no training reads. All five new directories carry one version set — harness `h0.9`, controller `c0.6`, expert `e0.4`, rule
judge `rj0.5`, generator `gen-robot-v0.3`, generator-config digest `c297ada2e90a` (R10's expertG digest) — and every run file records git
`8164947` (the registration commit). Scenes:

* **dev_new2** (validation, 100) — R6's seeds (`artifacts/reports/r6-seeds.json`); nine candidates from stored runs (the registration's
  `selection_sources`), r8s19G new, run with R10's dev_new2 command (`--config configs/eval/r7-closed-loop.yaml`, which only supplies the
  generator config).
* **ood_fresh** (the judged set, 200) — `configs/eval/r11-closed-loop.yaml`, `artifacts/reports/r11-seeds.json` (Stage A: base 1100100,
  E0 40 / E1 80 / E2 80, 0 seeds shared with any earlier set or dataset, 0 origin groups shared with training material; one seed, E2
  1120536, cannot be built by the scene generator and was skipped and listed by the selector). 43 of its 59 origin groups (177 of the 200
  seeds) also appear in the old ood_dev 200 — the same ood_dev family pool, by design; 16 groups (23 seeds) are new. Beyond the overlap
  report (fix round 1, review M8; `artifacts/scratch/r11/fix1/verify.json`): the D1 manifests (batch-0, d1, d1-rollout-labels), the r1
  v0.1 manifests (r1-v0.1, r1-rollout-labels-v0.1) and every stored closed-loop record directory — 94 manifests — share 0 (profile, seed)
  pairs with ood_fresh; d1 and d1-rollout-labels each share one train-split origin-group name, and none of the five checkpoints trained on
  D1.
* **ood_dev200** (the old holdout, the registration's secondary set) — the stored R4/R6/R7/R10 records of the same four policies.

The sealed split is not opened: the closed-loop configs refuse `ood_test`, the selector only counts sealed seeds, the overlap report counts
sealed manifest entries without reading their seeds, and every analysis read records through the manifest-first reader.

## 3. Resume

Not applicable — no training. R11 changed no runner, wrapper or record-writing code; the byte-identity tests of R10 (an R9 record replayed
byte for byte; the expert wrapped in G reproducing R6's expert record) pass in the full suite.

## 4. The choice, the registered verdict and the descriptive readings (Stage C)

**Runs, in the registered order, all after the registration commit (08:14:41).** GPU unit `r11-c-dev-r8s19G` (r8s19G on dev_new2,
08:15:06 → 08:25:54) → CPU unit `r11c-c-select` (the dev_new2 report `artifacts/reports/r11-selection-report.json` from the registered
sources, and `select` → `artifacts/reports/r11-selection.json`, 08:26:06 → 08:27:16) → CPU units `r11c-c-fresh-{ruleG,rule,expert}` on
ood_fresh (08:27:26 → 08:33:58, in parallel, with the GPU idle) → GPU unit `r11-c-fresh-r6G` (08:34:07 → 08:58:02, no CPU unit running) →
CPU unit `r11c-c-final` (`artifacts/scratch/r11/final-reports.sh`: `report --merge ood_dev200=… --only dev_new2,ood_fresh,ood_dev200
--seed-pairs ruleG:r6G r6G:ruleG --robustness-metrics forbidden_contacts` → `artifacts/reports/r11-closed-loop.json`; `verdict
--registration configs/eval/r11-registration.yaml` → `artifacts/reports/r11-verdict.json`; 08:58:14 → 09:10:47). The verdict re-chose r6G
from the final report's dev_new2 table, identical to `select`'s choice (the chain asserts it).

### The choice (dev_new2 100, the registered rule applied by code)

| configuration | checkpoint · arm | source | `done` | **strict** | false done |
| --- | --- | --- | ---: | ---: | ---: |
| **r6G** | r6 · G | R10 exploratory run | 89 | **74** | 15 |
| r6 | r6 · native | R6 run | 72 | 69 | 3 |
| r5 | r5 · native | R6 run | 79 | 68 | 11 |
| r9s18cG | r9s18c · G | R10 exploratory run | 66 | 61 | 5 |
| r5G | r5 · G | R10 registered run | 71 | 60 | 11 |
| r9s17G | r9s17 · G | R10 registered run | 74 | 50 | 24 |
| r9s17 | r9s17 · native | R9 run | 74 | 50 | 24 |
| **r8s19G** | r8s19 · G | **R11 run (new)** | 73 | 36 | 37 |
| r8s19 | r8s19 · native | R8 run | 51 | 26 | 25 |
| r9s18c | r9s18c · native | R9 run | 17 | 14 | 3 |

All ten eligible (the five training runs completed at 233/233). **Chosen: r6G**, decided by the first key (74 vs 69); no tie rule was
needed. The new run r8s19G (36 strict; main / aux / geometric failures 12 / 15 / 0; forbidden-contact onsets 12; unsafe 0.100) does not
change the order. Beside, on the same scenes: the rule judge 67 (native and G), the expert 90.

### The registered verdict (ood_fresh 200)

| condition | pair | difference [95 %] | discordant · exact test | RNG seeds 1–200 | holds if | holds? |
| --- | --- | --- | --- | --- | --- | --- |
| (a) strict success | `ruleG − r6G` | **+0.080 [+0.020, +0.135]** (0.805 vs 0.725; 161 vs 145) | 28 vs 12 · McNemar p = 0.017 | lower ≤ 0 in 0 of 200 (min +0.015) | lower ≤ 0 | **no** |
| (s) forbidden-contact onsets / episode | `r6G − ruleG` | **+0.035 [+0.010, +0.065]** (0.065 vs 0.030; 13 vs 6 onsets) | seeds higher 6 vs 0 · sign test p = 0.031 | lower ≤ 0 in 0 of 200 (min +0.010) | lower ≤ 0 | **no** |

*How (s) counts (fix round 1, review M2):* `stop_vs_reflex` adds an onset when a `contact_onset` event's object is in the forbidden set
accumulated over the episode up to that tick; §0's wording ("that tick's goal `forbidden_contact`") is narrower. On R11's records the two
definitions give the same counts — ood_fresh 13 vs 6, ood_dev 200 25 vs 9, 0 seeds differ for any of the four policies
(`artifacts/scratch/r11/fix1/verify.json`).

**Registered call: fail — "불통과 — R10의 사후 근접은 되풀이되지 않았다(또는 안전에서 떨어졌다)"** (`failed: ['a', 's']` — both parts
of the sentence apply: the rule judge in the same arm is ahead on strict success, and the chosen configuration makes more forbidden
contacts). Per the registration the documents write the next round from the numbers only.

**The old holdout, beside (secondary `ood_dev200`; same records R10 read).** (a) `ruleG − r6G` +0.025 [−0.045, +0.090] (26 vs 21,
p = 0.56) — R10's post hoc reading, reproduced; (s) `r6G − ruleG` **+0.080 [+0.020, +0.160]** (25 vs 9 onsets; seeds higher 9 vs 1,
p = 0.021; lower > 0 under all 200 RNG seeds) — so on the old holdout the configuration would also have failed the safety condition; R10's
post hoc comparison looked at strict success only.

### Descriptive readings (not used to judge)

| policy | strict: ood_fresh · ood_dev200 | false done (last grasp on a non-target; of those, an earlier instruction's target) | main / aux / geometric failures | first grasp commitment on a non-instructed object | `q_stop` caught / onsets | forbidden-contact onsets | unsafe ticks / acted |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **r6G** (chosen) | **145** · 139 | **28 (25; 19)** · 28 (26; 18) | 15 / 11 / 1 · 18 / 13 / 2 | 17 · 12 | 1/52 · 1/82 | **13** · 25 | 0.0356 · 0.0417 |
| ruleG (same arm) | **161** · 144 | 1 (0; 0) · 0 | 0 / 2 / 36 · 1 / 5 / 50 | 0 · 0 | 29/36 · 29/41 | **6** · 9 | 0.0002 · 0.0005 |
| rule (native) | 162 · 145 | 1 (0; 0) · 0 | 0 / 37 / 0 · 2 / 51 / 2 | 0 · 0 | 34/41 · 31/43 | 6 · 9 | 0.0002 · 0.0005 |
| expert | 189 · 182 | 1 (0; 0) · 0 | 1 / 0 / 9 · 2 / 0 / 16 | 0 · 0 | 48/48 · 65/65 | 6 · 23 | 0 · 0 |

(Stale-goal false dones by R9's `false_done_mechanism.py`, `artifacts/scratch/r11/false-done-mechanism.json`; first-grasp and failure
counts by `artifacts/scratch/r11/failures.py`; tables by `artifacts/scratch/r11/tables.py` → `c-tables.md`.)

* **One false done is shared by every policy, the expert included**: E1:1110410 — the first instruction is completed (target o3 observed
  inside zoneR at the done tick and at the last tick), but the scene's second instruction is scheduled at exactly 9,700 ms, the instant the
  0.3 s done tail ends, so the episode's outcome is scored against the new target (o0) and `target_inside_zone` is false
  (`artifacts/scratch/r11/probe/`). All four policies end that episode at the same tick, so it cancels in every seed-paired comparison;
  it is a scene-timing edge of the generator, not a policy decision (R4–R10 never met it; nothing was changed).
* **Seed-paired, `r6G − ruleG` on ood_fresh** (`seed_pairs`): false dones **+0.135 [+0.090, +0.185]** (27 vs 0 seeds), main-decision
  failures +0.075 [+0.040, +0.115], wrong-action ticks +0.061 [+0.039, +0.085], unsafe ticks +0.035 [+0.016, +0.058], `q_stop` catch
  −0.786 [−0.923, −0.639], gripper duplicates +0.130 [+0.065, +0.210], missing reference gripper transitions +0.068 [+0.009, +0.135];
  reflex-event ticks −0.010 [−0.160, +0.100] (no difference found). Beside: `rule − r6G` strict +0.085 [+0.025, +0.145], `expert − r6G`
  +0.220 [+0.165, +0.280], `ruleG − rule` −0.005 [−0.015, 0.000] (no difference found).
* **Two samples of the same configuration** (rewritten in fix round 1, review I2 — the first version read point estimates as a finding).
  The two sets have different seeds, so the comparison is unpaired; each set bootstrapped separately (2,000 resamples, RNG seed 20260921;
  `artifacts/scratch/r11/fix1/verify.json`), strict ood_fresh − ood_dev 200 is: expert +0.035 [−0.015, +0.085] (189 vs 182) and r6G
  +0.030 [−0.060, +0.120] (145 vs 139) — no difference found —; the rule judge +0.085 [+0.005, +0.165] (162 vs 145) and ruleG +0.085
  [+0.005, +0.165] (161 vs 144) — borderline: the lower bound is at or below zero under 84 and 105 of 200 alternative RNG seeds
  (`fix1/i2-sensitivity.json`), and the review's draw order puts ruleG's at +0.000. The `ruleG − r6G` gap went from +0.025 to +0.080, a
  change of +0.055 [−0.030, +0.150] — no difference found. So "the fresh draw was easier for every policy" and "the gap widened" are not
  findings. r6G's false dones are the same in number and kind on both (28; 19 and 18 after a stale goal), its main-decision failures 15
  vs 18, its `q_stop` catches 1 stop onset in either set; its forbidden-contact onsets are 13 vs 25 and the rule judge's 6 vs 9.
* On dev_new2 (the choice set — not evidence): `ruleG − r6G` −0.070 [−0.150, 0.000] (4 vs 11 rule-judge-only vs r6G-only seeds, the order
  used everywhere here — fix round 1, M5; as R10's exploratory run read it) and
  forbidden contacts 3 vs 1 (+0.020 [0.000, +0.050]) — the validation set did not show the safety difference (too few protected contacts
  in 100 scenes).

**Latency** (`latency.model_ms` excludes each episode's first tick; no CPU unit ran during either GPU run — a session-shell read of 31
run-summary JSONs with existence checks did run at 08:16:52 during r8s19G's run, §5): r6G on ood_fresh non-first
ticks p50 / p95 / p99 **43.7 / 56.1 / 80.7 ms**, 0 of 19,418 over 100 ms, observation → command p95 62.1 ms; first ticks p50 93.5, max
277.2 ms, 18 of 200 over 100 ms. r8s19G on dev_new2: 43.5 / 56.3 / 81.9 ms, 0 of 8,092; first ticks 21 of 100 over 100 ms (max 275.6 ms).
The gate (p95 ≤ 80 ms, > 100 ms ≤ 5 %) passes.

## 5. Measured cost

| unit (systemd, `Started` → end) | what | wall |
| --- | --- | ---: |
| `r11-c-dev-r8s19G` | GPU: r8s19G on dev_new2 (100), the only new validation run | 0 h 10 m 48 s |
| `r11-c-fresh-r6G` | GPU: the chosen configuration r6G on ood_fresh (200) | 0 h 23 m 55 s |
| `r11-pytest` | full suite (fence 0.6 + `require_free(36 GiB)` checked first; GPU visible, no other GPU job) | 0 h 10 m 12 s |
| **GPU total** | | **2,695 s = 0.75 GPU-h** (brief ≈ 1 h) |

CPU units (`r11c-*`, CUDA hidden, MemoryMax, `choom`, Nice 10): the seed walk (25 s; a first attempt stopped after 8 s on the unbuildable
seed, §2), the overlap report, the seed re-check from the Stage A commit (25 s), the regression check of R7–R10's verdicts (4 s, plus the
same R7 verdict with `main`'s code), the choice (`r11c-c-select`, 1 m 10 s), ruleG / rule / expert on ood_fresh (6 m 05 s, 6 m 05 s,
6 m 32 s, in parallel, GPU idle), the final report and verdict (`r11c-c-final`, 12 m 33 s, MemoryMax 40G — it overlapped the test unit,
which is not a measurement), three probes (13 s, 3 s, 1 s) and the development test runs (`r11c-pt-*`, 2–40 s each). Disk 126 → 125 GB
free (closed-loop records 791 MB). Cumulative GPU on the DGX Spark ≈ 67.0 h. Cloud spend 0. *(Fix round 1, review M7:)* the two units that
overlapped, `r11c-c-final` (MemoryMax 40G) and `r11-pytest` (MemoryMax 90G), together budgeted 130 GiB on a 121 GiB box — `gpu-chain.sh`
checks only GPU units before it launches; the kernel log for the day has no OOM or `NVRM` line. Fix round 1's checks ran as CPU units
(`r11c-fix1-verify`, `r11c-fix1-i2`, test units), GPU 0.

**Every CPU job that read records ran as a unit (HANDOFF §5; R9 and R10 broke this rule).** Outside units, from the session shell, only
report and metadata files were read (list completed in fix round 1, review M1): before the first unit (07:46–07:57, no GPU job up) the
run-summary JSONs, `r10-exploratory.json`, the earlier seed / overlap / crosscheck JSONs, the `files` metadata of five dataset manifests (r1,
g2, dagger-0, dagger-1, the done-gate collection) and, at 07:54, the five training runs' `metrics.json`; **at 08:16:52, during GPU unit
`r11-c-dev-r8s19G`**, a `python3` that loaded 31 run-summary JSONs and checked that their episode directories and `manifest.json` files
exist (`stat` only — no manifest or record opened); after the GPU runs (from 09:11), the selection, verdict, table and probe outputs
(JSON / markdown) and the run-summary rows behind `artifacts/scratch/r11/probe/discordant.json`. No episode record was opened outside a
unit.

## 6. What the round says, and what it does not

1. **Registered call: fail — "불통과 — R10의 사후 근접은 되풀이되지 않았다(또는 안전에서 떨어졌다)", on both conditions.** The configuration
   chosen on validation (r6G, 74 of 100 on dev_new2, the highest of ten) trails the rule judge in the same arm on 200 fresh holdout scenes
   — `ruleG − r6G` strict **+0.080 [+0.020, +0.135]** (161 vs 145; 28 vs 12 discordant, p = 0.017; the lower bound stays above zero under all
   200 alternative bootstrap seeds) — and makes more forbidden contacts — `r6G − ruleG` **+0.035 [+0.010, +0.065]** per episode (13 vs 6 onsets;
   6 vs 0 seeds, sign test p = 0.031; likewise robust). R10's post hoc closeness (+0.025 [−0.045, +0.090] on ood_dev 200) did not repeat.
2. **The old holdout would have failed the safety condition too.** On the same records R10 read, `r6G − ruleG` forbidden contacts is +0.080
   [+0.020, +0.160] (25 vs 9; 9 vs 1 seeds). R10's post hoc comparison looked at strict success only; the safety condition the user added
   for R11 separates the two on both holdouts.
3. **Where the gap is (descriptive) — semantic judgment, as R10 said.** The 28 seeds only the rule judge solved are r6G stalls after
   main-decision (9) or auxiliary (9) disagreements and r6G false dones (10; 2 of them on an earlier instruction's target); the 12 seeds only
   r6G solved are all rule-judge execution stalls (`geometric`; 12 of 12 `stall_exhausted`, 11 `arm_pinned`). Of r6G's 55 failures, 53
   are semantic — 27 false dones (19 after a last grasp on an earlier instruction's target, 6 on another non-target, 2 other), 15
   main-decision and 11 auxiliary failures —, 1 is the shared generator edge (item 6) and 1 is geometric (fix round 1, M4: the first version
   counted the edge as semantic, "54"); of the rule judge's 39, 36 are geometric stalls or timeouts (plus the same edge and 2 auxiliary).
   r6G's false dones are the same in number and kind on both holdouts (28 and 28; 19 and 18 stale-goal). Of its 19 stale-goal false dones,
   2 fall on seeds only the rule judge solved and 17 on seeds both failed — on those 17 the rule judge in the same arm stalled every time
   (`stall_exhausted`: 16 `arm_pinned`, 1 `no_progress`) and the expert finished 16. The gap is in the instruction-change profiles (E1 65
   vs 72, E2 40 vs 49; E0 40 vs 40).
4. **Safety**: r6's `q_stop` catches 1 of 52 stop onsets (the rule judge 29 of 36) and its unsafe-tick rate is 0.036 (0.0002); arm G
   delegates only the gripper, so the stop is still the model's. Whether the missing stops cause the extra forbidden contacts was not tested.
5. **Two samples, no finding on "easier"** (rewritten in fix round 1, review I2). Unpaired, each set bootstrapped separately: strict
   ood_fresh − ood_dev 200 is expert +0.035 [−0.015, +0.085], r6G +0.030 [−0.060, +0.120], the rule judge +0.085 [+0.005, +0.165] and
   ruleG +0.085 [+0.005, +0.165] (borderline — lower bound ≤ 0 under 84 and 105 of 200 RNG seeds), and the change of the `ruleG − r6G` gap
   +0.055 [−0.030, +0.150]. Under the zero rule neither "the fresh draw was easier for every policy" nor "the gap widened" holds; the
   registered call does not rest on either.
6. **A scene-timing edge of the generator**: in E1:1110410 the second instruction is scheduled at the instant the done tail ends, so a
   completed first instruction is scored against the new target — a "false done" for every policy, the expert included. It cancels in every
   seed-paired comparison and changed nothing here; the harness and generator were not touched (a fix would be a generator change for a
   later round).
7. **What the round does not say.** Only the chosen configuration ran on the fresh holdout, so the other nine are not ranked there; the
   choice used dev_new2 values of which nine were known in advance (the choice was not blind, the judged set was); one holdout of 200; and
   the rule judge reads the structured goal while the model reads the instruction text. It does not say delegation is useless — R10 showed
   it repairs a collapsed gripper head — only that the best configuration we have does not match the rule judge on unseen scenes.

**Next round (registered: on a fail, written from the numbers only) — DAgger aimed at stale-goal false dones,** with what the numbers say
about its reach (corrected in fix round 1, review I1 — the first version read the 2 / 17 split backwards): false dones are r6G's largest
single failure class (27 of 200 besides the shared edge; 19 stale-goal) and DAgger cycles 0–1 did not remove them. (a) = (rule-judge-only
− r6G-only) / 200 = (28 − 12) / 200. Of the 19 stale-goal false dones, 2 are on rule-judge-only seeds and 17 on seeds both failed, where
the rule judge stalled every time and the expert finished 16 — a success there would be an r6G-only seed. Every stale-goal false done a fix
turns into a success therefore lowers (a) by 0.005, whichever group it is in: turning the 2 gives +0.070, turning all 19 gives −0.015. That
range is untested — it depends on how many of those episodes a fixed policy would actually complete. r6G's stalls after main-decision or
auxiliary disagreements (18 of the 28 rule-judge-only seeds) and its forbidden contacts (13 vs 6; `q_stop` 1 of 52 — the safety condition
(s)) are separate targets. **The next verdict needs a new, unused seed walk** (fix round 1, review I3): ood_fresh (1100100~) has now been
analysed seed by seed and that analysis is designing the next round, so judging it on ood_fresh again would bring back the post hoc
problem R11 was set up to remove. No contract-change proposal (delegation G into the harness) is supported by this round, and nothing here
argues for cloud spend.
