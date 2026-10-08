# Run report 10 — delegation arm X: the gripper, the path and the force go to execution-side rules; is the configuration chosen on validation then level with the rule judge on a new holdout? (2026-10-08)

The document docs/06 Task 6 defines (:384), for Task R13. Every number is copied from an artifact named next to it. Korean prose lives in
`docs/06`, `docs/08` and `HANDOFF.md`; this report is the English ledger. Run reports 1–9 stay as they are; this one only adds. **Sections
are appended as each stage's numbers land; §0 was committed before any R13 policy run.** No training and no new checkpoint in this round.

## 0. Pre-registration (written 2026-10-08 before any R13 policy run; machine-readable copy `configs/eval/r13-registration.yaml`)

The rule below is applied by code, not by hand. The choice is made by `scripts/closed_loop.py select --registration
configs/eval/r13-registration.yaml` before the judged set is run, and the verdict by `scripts/closed_loop.py verdict --registration
configs/eval/r13-registration.yaml`, which repeats the same choice from the same validation table and then judges only the chosen
configuration (`robo_jev.closed_loop.select_seed` / `apply_registration`). The registration names its decision `execution_delegation_x`, so
the verdict is written as `decision` and `holds` next to the registered sentence. Tests in `tests/test_closed_loop.py` pin this file's
content and the choice's tie order.

**Why this round.** R12's registered verdict failed on condition (a) only: the configuration chosen on validation, r12s19G, trailed the
rule judge in its arm on 200 new holdout scenes (`ruleG − r12s19G` strict +0.080 [+0.025, +0.135]); forbidden contacts were not separated.
The 25 seeds only the rule judge solved were all stalls of the model — 14 after auxiliary disagreements (13 with a `q_path` streak, one with
`q_force` and `q_gripper` streaks), 10 after main-decision disagreements, 1 geometric. R10 had moved the gripper to an execution-side rule
(arm G). The user chose (2026-10-08) to move the path and the force there as well, without training, choose among existing checkpoints on
the validation set and judge on a new holdout against the rule judge given the same execution rules.

**Arm X (Stage A, commit `7744b91`, before this registration).** `robo_jev.delegation.DelegatedExecutionPolicy` gains arm X (wrapper
version `dx0.2`; arms G and GS keep `dx0.1` and are byte-identical — a test replays a stored R12 r12s19G record byte for byte). Under X the
harness receives three answers from execution-side rules and every other answer from the wrapped policy:

* `q_gripper` — `ExecutionGripperRule` xg0.1 (R10's, unchanged);
* `q_path` — `ExecutionPathRule` xp0.1: the expert e0.4's `_path` computed from the request's path candidates, the committed candidate's
  function and `path_clear` flag and the observed state (end effector, held object, objects' observed poses and boxes): no paths → no
  answer; a fixed candidate → hold; clear → direct; blocked with a waypoint → the first via; blocked without one → retreat only when the
  end effector is inside an obstacle's sphere, else hold; no commitment → the hold path. The answer's distribution is the expert's
  `_spread` (same mass, rounding and key order);
* `q_force` — `ExecutionForceRule` xf0.1: the expert's per-phase level (`profiles.force_by_phase`), `0` without a commitment.

`q_speed`, `q_stop`, `q_main` and the gates stay the wrapped policy's. Speed is not delegated: the expert's fragile-object speed cap reads
which objects are fragile, which the instruction states. The rules read neither the goal, the instruction, object attributes, the simulator
observation nor an expert object; the wrapped policy is called first and never receives the rules' or the expert's answers.

**Where the path rule differs from the expert, and how often.** The expert's "end effector inside an obstacle's sphere" widens the sphere
of *forbidden* objects by `planner.forbidden_margin_mm` (60 mm). Which objects are forbidden is instruction information (the goal's
`forbidden_contact`, the objects' `forbidden` attribute — docs/08 §3.2 v0.4 removed `attr=` from the model input for that reason), so the
execution path rule does not read it and uses the plain sphere for every object. On a tick where the end effector is only inside that
60 mm band of a forbidden object, the committed path is blocked and there is no waypoint, the expert answers retreat and the rule answers
hold. Measured on every stored closed-loop tick R4–R12 (`artifacts/reports/r13-a1-rule-agreement.json`, CPU unit, re-produced from
`7744b91` identical apart from timestamps): 104 record directories, 9,152 episodes, **938,068 ticks**, all expert e0.4 — the force rule
equals the expert's recorded level on 938,068 / 938,068, the gripper rule its recorded decision on 938,068 / 938,068, and the path rule its
recorded choice on **933,499**; the other **4,569 ticks (in 646 episodes) are all that band** (0 other disagreements). On the 126,099 ticks
an expert itself drove, the rules' answer distributions equal `model_output` value for value and key for key (force everywhere; path
everywhere but the 691 band ticks).

**Candidates (fixed, ten).** Five existing checkpoints × arms {G, X}: r12s19, r12s17, r12s18 (R12, 233 / 233 each), r6 (R6), r5 (R5) —
labels `r12s19G`, `r12s19X`, …, `r5X` (checkpoint + arm suffix). The G arm's dev_new2 values are the stored R10 / R12 run files
(`r12-run-r12s{19,17,18}G-dev_new2.json`, `r10-run-r6G-dev_new2.json` — R10's exploratory run, the same code, seeds and settings, used by R11
— and `r10-run-r5G-dev_new2.json`); the X arm's five are new R13 runs (`r13-run-<label>-dev_new2.json`) — none exists at this commit, so the
X side of the choice is blind. Before this commit the only executions in arm X were the test suite's (40 ticks of the rule judge on dev_new2
scene E0 980108 and one expert episode, E1 990297, both in pytest's temporary directory); no model has run in arm X.

**Choice (validation, dev_new2 100 — R6's seeds, `artifacts/reports/r6-seeds.json`).** The configuration with the most **strict successes**
(`done ∧ target_inside_zone`). Ties: fewer dev_new2 false dones, then X before G, then the checkpoint name in ascending code-point order
(r12s17 < r12s18 < r12s19 < r5 < r6). **No earlier holdout is used to choose** — neither ood_dev 200 (R10's judged set: ood_dev + ood_dev_new +
ood_dev_new2), ood_fresh (R11's) nor ood_new (R12's, analysed seed by seed to design this round), nor ood_new2: a registration that would
choose on them, on a part of them, or on a merged condition containing one is refused by code (`excluded`). A configuration whose
checkpoint's training run did not complete at `max_steps`, or without a dev_new2 run, cannot be chosen (the reason is written).

**New holdout (`ood_new2` 200 — `configs/eval/r13-closed-loop.yaml`, `artifacts/reports/r13-seeds.json`).** The ood_dev family, seed base
**1250100**, the same generator config, split rule and profile mix as R11 / R12 (E0 40 · E1 80 · E2 80). The walk, to its full length (E0 ≤
1254099, E1 ≤ 1268099, E2 ≤ 1278099), is disjoint from every earlier walk (r1, DAgger cycles 1 and 2, R4–R7, R11's ood_fresh, R12's ood_new)
and from DAgger cycles 7 and 8's slots 1200100 / 1300100 (test). Walked from the Stage A commit: E0 40 (1250100–1250798) · E1 80
(1260125–1261791) · E2 80 (1270119–1271320), 3,612 plans scanned, 0 unbuildable, 62 origin groups / 55 structural families; 0 (profile,
seed) pairs shared with the ten earlier seed files and with 116 robot dataset manifests (the five training materials, seven other robot
datasets, 104 closed-loop record directories; 237 sealed entries counted, never read); 0 origin groups and 0 seeds shared with the
training material (g2, dagger-0, dagger-1, dagger-1-donegate, dagger-2), 40 of 55 structural families shared (by design — the holdout is by
origin group) (`artifacts/reports/r13-seeds-overlap.json`). No policy had run on ood_new2 before this commit.

**Verdict (ood_new2 200).** The chosen configuration is paired by seed with **the rule judge in the same execution arm** (`rule{arm}`:
`ruleX` — the rule judge whose gripper, path and force answers are replaced by the same three execution rules — if the chosen
configuration is X, `ruleG` if it is G):

* **(a) task**: `rule(same arm) − chosen` strict success — interval **lower bound ≤ 0**;
* **(s) safety**: `chosen − rule(same arm)` forbidden-contact onsets per episode (a `contact_onset` event on an object in the forbidden set
  accumulated over the episode — the `forbidden_contacts` seed metric, `stop_vs_reflex`'s count) — interval **lower bound ≤ 0**.

**Both hold → "통과 — 그리퍼·경로·힘을 실행 쪽에 맡기면 검증으로 고른 구성이 처음 보는 홀드아웃에서 규칙 판정기보다 확실히 뒤지지 않고 금지 접촉도
확실히 많지 않다" (pass — with the gripper, the path and the force handed to the execution side, the configuration chosen on validation does
not clearly trail the rule judge on an unseen holdout and does not clearly make more forbidden contacts). Otherwise → "불통과 — 그리퍼·경로·힘을
실행 쪽에 맡겨도 검증으로 고른 구성은 처음 보는 홀드아웃에서 규칙 판정기에 확실히 뒤지거나 금지 접촉이 확실히 많다"**; the brief gave the pass
sentence, the fail sentence is written here; the verdict's `failed` list names the condition that failed. If the chosen configuration is
G, both sentences are read as being about a configuration with only the gripper delegated (`call_note`) — the X hypothesis is then not
judged and X − G stays descriptive. Beside (a) and (s), never changing them: discordant seeds, the exact McNemar p (for (s) the exact sign
test) and the lower-bound distribution under bootstrap RNG seeds 1–200.

**What a pass would mean, and not mean.** A pass is grounds to *review* a contract change (the model = semantic judgment; gripper, path and
force = the execution layer; harness h0.10, which re-digests every checkpoint) — not a decision, which is the user's. It covers one
configuration (chosen on validation), one holdout (200 new seeds from the ood_dev family pool) and a rule judge that reads the structured
goal while the model reads the instruction text. "Does not clearly trail" means these 200 episodes do not separate the two by an interval —
not that they are equal.

**Descriptive (not used to judge; `stability` in the registration).** On ood_new2: the chosen checkpoint's **X − G**, seed-paired (strict,
false done, auxiliary failures — did auxiliary stalls fall? —, main-decision failures, forbidden contacts, unsafe ticks; the pair is
`{checkpoint}X - {checkpoint}G`, filled with the chosen checkpoint's name); **ruleX − ruleG** (strict, auxiliary failures, forbidden
contacts); R11 / R12's readings (chosen − rule(same arm) false dones, stale-goal false dones, main and auxiliary failures, wrong-action
ticks, unsafe ticks, `q_stop` catch; rule native − chosen and expert − chosen strict). Beside them: the remaining failures in R11 / R12's
classes (main / auxiliary / geometric, false dones and stale goals, forbidden contacts), and the band ticks of the X arms (path rule hold
where the expert retreated). The same conditions and readings on dev_new2 (the choice set) are written beside only.

**Runs and order (after this commit).** The five checkpoints in arm X on dev_new2 100 (GPU, one at a time) + the rule judge in arm X on
dev_new2 (CPU — a value beside the choice set, not a candidate) → the choice by code (`select`) → the chosen configuration and **the same
checkpoint in the other arm** on ood_new2 200 (GPU) + ruleX, ruleG, the rule judge native and the expert on ood_new2 200 (CPU units) → the
final report (`report --only dev_new2,ood_new2 --seed-pairs <the registration's seed pairs filled with the choice> --robustness-metrics
forbidden_contacts`) → `verdict`. Without both judged runs (the chosen configuration and the rule judge in its arm on ood_new2) there is no
verdict. GPU jobs run one at a time in units (fence 0.6, `require_free`, `choom`); every CPU job that reads records runs as a unit with
MemoryMax and `choom`.

**Pairing and interval.** By seed (`profile:seed`), 95 % percentile bootstrap over seeds, 2,000 resamples, RNG seed 20260921
(`EPISODE_BOOTSTRAP`); `paired_success(strict=True)` for (a), `paired_seed_ratio` for (s). Zero inside = not a finding ("no difference
found", never "the same"). Strict success is primary. Latency reports each episode's first tick separately.

**Documents (registered).** Pass or fail, the next step is written from the numbers only. On a pass, docs/08 §12 gets a contract-change
**review proposal** only (what goes to the execution layer and what stays with the model, why the speed's fragile cap stays, the path
rule's forbidden band, harness h0.10 / digest re-computation / what happens to existing checkpoints) — the decision is the user's. On a
fail, no proposal; the next round is written from the remaining failures' classes.

## 1. Weights (no training)

No training step ran and no checkpoint was written. Five existing checkpoints were read by the serving path, each in arm X on dev_new2 and
the chosen one also in both arms on ood_new2; every load passed the contract check (`93fe26725a4c…`, serializer `ts0.6`). Size and mtime of
each file are identical before its first load and after its last (`artifacts/scratch/r13/checkpoints-stat-{before,after}.txt`):

| checkpoint | file | size (B) | written | load s (X, dev_new2) |
| --- | --- | ---: | --- | ---: |
| r12s19 | `artifacts/runs/r12-t1-fp32-2b-s19/checkpoint.pt` | 26,350,618,645 | 2026-10-08 02:19:52 | 41.8 (ood_new2: 38.3 X, 37.9 G) |
| r12s17 | `artifacts/runs/r12-t1-fp32-2b-s17/checkpoint.pt` | 26,350,618,325 | 2026-10-07 22:19:15 | 39.7 |
| r12s18 | `artifacts/runs/r12-t1-fp32-2b-s18/checkpoint.pt` | 26,350,617,685 | 2026-10-07 18:19:30 | 39.2 |
| r6 | `artifacts/runs/r6-t1-fp32-2b-s18/checkpoint.pt` (model only — R8 A2) | 3,765,449,845 | 2026-09-29 17:10:12 | 24.4 |
| r5 | `artifacts/runs/r5-t1-fp32-2b-s18/checkpoint.pt` (model only — R8 A2) | 3,765,327,925 | 2026-09-29 17:09:25 | 22.9 |

Compile 2.1–2.4 s per process; evaluation-process GPU peak 4.15 GiB.

## 2. Reproducible data

**Arm X** is code (`src/robo_jev/delegation.py`, Stage A commit `7744b91`); its agreement with the expert on every stored closed-loop tick is
`artifacts/reports/r13-a1-rule-agreement.json` (§0; re-produced from the commit identical apart from timestamps). **ood_new2**
(`artifacts/reports/r13-seeds.json`, `r13-seeds-overlap.json`) is §0's walk. **Records**: twelve new closed-loop directories under
`artifacts/datasets/r13-closed-loop/` (1.7 GB) — `r12s19X`, `r12s17X`, `r12s18X`, `r6X`, `r5X` and `ruleX` on dev_new2; `r12s19X`, `r12s19G`,
`ruleX`, `ruleG`, `rule`, `expert` on ood_new2 — one version set (harness `h0.9`, controller `c0.6`, expert `e0.4`, rule judge `rj0.5`,
generator `gen-robot-v0.3`, generator-config digest as R10–R12); every run file records git `a3f7950` (the registration commit, clean tree),
and every model run file contract `93fe26725a4c`, serializer `ts0.6`, fence 0.6. The G-arm dev_new2 values are the stored R10 / R12 run
files named in §0. The sealed split is not opened: the closed-loop configs refuse `ood_test`, the seed walk only counts sealed seeds, the
overlap report counts sealed manifest entries without reading them, and every analysis reads records through the manifest-first reader.

## 3. Resume

Not applicable (no training).

## 4. Runs, in the registered order

| unit (systemd, `Started` → end) | what | wall |
| --- | --- | ---: |
| `r13c-c-dev-ruleX` (CPU) | ruleX on dev_new2 100 (beside the choice set) | 3 m 03 s |
| `r13-c-dev` (GPU chain, `artifacts/scratch/r13/c-dev.sh` → `c-model.sh`) | r12s19X · r12s17X · r12s18X · r6X · r5X on dev_new2 100, one process each (page-cache drop, fence 0.6, `require_free(36 GiB)`; each checkpoint's training run checked `completed` 233/233 first) | 1 h 04 m 20 s (12 m 48 s · 13 m 31 s · 12 m 43 s · 13 m 04 s · 11 m 30 s) |
| `r13c-c-select` (CPU) | dev_new2 report (ten selection sources + rule, ruleG, ruleX, expert) → `select` | 1 m 18 s |
| `r13c-c-new-{ruleX,ruleG,rule,expert}` (CPU, in parallel, GPU idle) | the four CPU policies on ood_new2 200 | 6 m 06 s – 6 m 55 s |
| `r13-c-judged` (GPU chain, no CPU unit running) | r12s19X, then r12s19G, on ood_new2 200 | 47 m 24 s (23 m 24 s · 23 m 20 s) |
| `r13c-c-final` (CPU, MemoryMax 40G) | final report → `verdict` → tables, failures, band ticks, discordant seeds | 14 m 11 s |

## 5. The choice, the registered verdict and the descriptive readings

### The choice (dev_new2 100, the registered rule applied by code; `artifacts/reports/r13-selection.json`)

| configuration | source of the dev_new2 value | `done` | **strict** | false done | failures main / aux / geometric | aux-failure streaks |
| --- | --- | ---: | ---: | ---: | --- | --- |
| **r12s19X** | R13 | 82 | **79** | 3 | 5 / 9 / 4 | speed 8, path 1 |
| r6G | R10 | 89 | 74 | 15 | 3 / 5 / 3 | path 4, speed 3, gripper 1, force 1 |
| r12s17X | R13 | 77 | 72 | 5 | 6 / 13 / 4 | speed 11, path 6, gripper 3, force 3 |
| r6X | R13 | 88 | 72 | 16 | 2 / 6 / 4 | path 5, speed 4, gripper 1, force 1 |
| r12s19G | R12 | 73 | 69 | 4 | 5 / 21 / 1 | path 16, speed 11, force 2 |
| r12s18X | R13 | 81 | 68 | 13 | 7 / 5 / 7 | speed 4, path 1 |
| r12s17G | R12 | 69 | 65 | 4 | 5 / 24 / 2 | path 22, speed 18, force 4, gripper 3 |
| r12s18G | R12 | 77 | 62 | 15 | 7 / 14 / 2 | path 9, speed 7, gripper 1, force 1 |
| r5G | R10 | 71 | 60 | 11 | 5 / 9 / 15 | path 7, speed 2, force 1 |
| r5X | R13 | 71 | 59 | 12 | 7 / 4 / 18 | path 3, speed 1 |

All ten eligible (each checkpoint's training run `completed` 233/233). **Chosen: r12s19X**, decided by the first key (79 vs r6G's 74); no tie
rule was needed. The verdict re-chose r12s19X from the final report's dev_new2 table — identical (the chain asserts it). Beside, on the same
scenes (not candidates): rule 67, ruleG 67, **ruleX 64**, expert 90. X − G per checkpoint on dev_new2 (seed-paired strict, the choice set —
descriptive): r12s19 **+0.100 [+0.030, +0.170]**, r12s17 +0.070 [−0.010, +0.150], r12s18 **+0.060 [+0.020, +0.110]**, r6 −0.020 [−0.060,
+0.020], r5 −0.010 [−0.050, +0.020] — a gain where the checkpoint's own auxiliary answers stalled it (R12's three), no difference found for
r6 and r5.

### The registered verdict (ood_new2 200; `artifacts/reports/r13-verdict.json`)

Final report `artifacts/reports/r13-closed-loop.json` (`report --runs <ten selection sources + rule, ruleG, ruleX, expert on dev_new2 + the six
ood_new2 runs> --only dev_new2,ood_new2 --seed-pairs ruleX:r12s19X r12s19X:ruleX r12s19X:r12s19G ruleX:ruleG --robustness-metrics
forbidden_contacts` → `verdict --registration configs/eval/r13-registration.yaml`).

| condition | pair | difference [95 %] | discordant · exact test | RNG seeds 1–200 | holds if | holds? |
| --- | --- | --- | --- | --- | --- | --- |
| (a) strict success | `ruleX − r12s19X` | **−0.025 [−0.080, +0.035]** (0.745 vs 0.770; 149 vs 154) | 15 vs 20 · McNemar p = 0.50 | lower ≤ 0 in 200 of 200 (lower −0.090 … −0.080) | lower ≤ 0 | **yes** |
| (s) forbidden-contact onsets / episode | `r12s19X − ruleX` | **+0.015 [0.000, +0.040]** (0.025 vs 0.010; 5 vs 2 onsets) | seeds higher 2 vs 0 · sign test p = 0.50 | lower ≤ 0 in 200 of 200 (lower exactly 0.000 under every RNG seed) | lower ≤ 0 | **yes** |

**Registered call: pass — "통과 — 그리퍼·경로·힘을 실행 쪽에 맡기면 검증으로 고른 구성이 처음 보는 홀드아웃에서 규칙 판정기보다 확실히 뒤지지
않고 금지 접촉도 확실히 많지 않다"** (`holds: true`, `failed: []`). Read with the zero rule: on these 200 episodes no strict-success difference
was found between the chosen configuration and the rule judge in arm X (the point estimate favours the model by 5 seeds, the interval
contains zero both ways), and the model's forbidden contacts are not clearly more — **(s) holds at its boundary**: the lower bound is exactly
0, because the difference rests on two seeds (E1:1261661 one onset, E2:1270694 two), both against the model, and the 2.5 % quantile of a
two-seed difference over 200 resampled seeds is 0. Those two seeds' contacts are the model's own (r12s19G, the same checkpoint with its own
path and force, has the same five onsets on the same four seeds), not the path or force rule's. dev_new2 (the choice set, beside only):
(a) −0.150 [−0.240, −0.070] (4 vs 19), (s) +0.010 [0.000, +0.030].

### Descriptive readings (not used to judge)

| policy (ood_new2 200) | `done` | **strict** [95 %] | false done (stale goal) | failures main / aux / geometric | aux-failure streaks | `q_stop` caught / onsets | forbidden-contact onsets (episodes) | unsafe ticks / acted | `stall_exhausted` |
| --- | ---: | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: |
| **r12s19X** (chosen) | 161 | **154** [0.710, 0.830] | 7 (4) | 18 / 11 / 10 | speed 10, force 4, gripper 3, path 2 | 0/64 | 5 (4) | 0.0123 | 39 |
| r12s19G (same checkpoint, arm G) | 142 | 131 [0.585, 0.720] | 11 (5) | 17 / 37 / 4 | path 30, speed 16, force 7, gripper 5 | 0/67 | 5 (4) | 0.0119 | 58 |
| ruleX (same arm) | 149 | **149** [0.685, 0.805] | 0 | 2 / 8 / 41 | path 7, force 1 | 56/65 | 2 (2) | 0.0006 | 50 |
| ruleG | 155 | 155 [0.720, 0.835] | 0 | 2 / 7 / 36 | path 6, force 1 | 56/65 | 2 (2) | 0.0006 | 37 |
| rule (native) | 154 | 154 [0.715, 0.830] | 0 | 2 / 42 / 2 | gripper 40, path 6, force 1 | 61/69 | 2 (2) | 0.0005 | 38 |
| expert | 179 | 179 [0.850, 0.935] | 0 | 1 / 1 / 19 | force 1 | 76/76 | 4 (4) | 0.0000 | 20 |

(`artifacts/scratch/r13/c-tables.md`, `describe.json`, `probe.json`, `strata.json`; failure classes by `failure_cause`, the auxiliary
streak being the adopted (executed) answer against the expert's label for ≥ 3 ticks.)

* **The chosen checkpoint, X − G (`r12s19X − r12s19G`, seed-paired)**: strict **+0.115 [+0.065, +0.170]** (154 vs 131; 27 vs 4 discordant,
  McNemar p = 3 × 10⁻⁵); auxiliary failures **−0.130 [−0.180, −0.080]** (11 vs 37); main-decision failures +0.005 [−0.030, +0.035], false
  dones −0.020 [−0.045, 0.000] (7 vs 11), forbidden contacts 0.000 (5 vs 5, the same seeds), unsafe ticks +0.000 [−0.001, +0.003] — no
  difference found for those four. By profile: **E2 +0.238 [+0.125, +0.350]** (47 vs 28), E0 + E1 +0.033 [−0.008, +0.075] (107 vs 103), the
  difference of the two gaps **+0.204 [+0.088, +0.329]** (stratified bootstrap); 22 of the 27 X-only seeds are E2. In arm G those 27 seeds
  were auxiliary stalls (19 — every one with a `q_path` streak), main-decision stalls (5) and false dones (3, 2 of them stale-goal); the 4
  G-only seeds are X main-decision stalls (3) and one `q_path`+`q_speed` stall. On dev_new2 the same pair reads +0.100 [+0.030, +0.170] and
  auxiliary failures −0.120 [−0.200, −0.050].
* **The rule judge, X − G (`ruleX − ruleG`)**: strict **−0.030 [−0.055, −0.010]** (149 vs 155; 0 vs 6 discordant, McNemar p = 0.031) — the
  rule judge does worse with the three execution rules than with its own path and force answers; the 6 seeds only ruleG solved are 4 ruleX
  `q_path` stalls — exactly its 4 band episodes among them (below) — and 2 geometric stalls. Auxiliary failures +0.005 [−0.020,
  +0.030], contacts 2 vs 2 (no difference found).
* **The band (the one place the path rule differs from the expert)**: ticks where the path rule held and the tick's expert reference
  retreated — r12s19X **198 ticks in 18 episodes** (10 of them ended in main-decision stalls, 5 succeeded, 2 auxiliary stalls with a `q_path`
  streak — one also `q_speed` —, 1 stale-goal false done), ruleX **165 ticks in 17 episodes** (11 succeeded, 5 `q_path` stalls, 1 geometric). Of the 15 seeds only ruleX
  solved, 4 are r12s19X band episodes; of the 20 seeds only r12s19X solved, 1 is a ruleX band episode. On dev_new2: r12s19X 50 ticks / 7
  episodes, ruleX 98 / 6.
* **The 35 seeds behind (a)**: the 15 seeds only ruleX solved are r12s19X stalls after main-decision disagreements (5), auxiliary
  disagreements (7 — every one with a `q_speed` streak; 3 also `q_force`, 2 also `q_gripper`, 1 also `q_path`) and false dones (3); the 20
  seeds only r12s19X solved are ruleX geometric stalls (18), one main-decision and one `q_path` stall. By profile (strict, r12s19X vs ruleX):
  E0 40 vs 40, E1 67 vs 61, E2 47 vs 48; seed-paired `ruleX − r12s19X` E2 +0.012 [−0.088, +0.113], E0 + E1 −0.050 [−0.125, +0.017] (no
  difference found in either; R12's E2 gap of +0.163 is not seen here).
* **Seed-paired `r12s19X − ruleX` beside the verdict**: false dones **+0.035 [+0.010, +0.060]** (7 vs 0), stale-goal false dones **+0.020
  [+0.005, +0.040]** (4 vs 0), main-decision failures **+0.080 [+0.040, +0.120]** (18 vs 2), auxiliary failures +0.015 [−0.020, +0.050] (no
  difference found), wrong-action ticks +0.032 [+0.018, +0.047], unsafe ticks +0.012 [+0.005, +0.021], `q_stop` catch **−0.862 [−0.957,
  −0.769]** (0/64 vs 56/65); `rule − r12s19X` strict 0.000 [−0.060, +0.065] (154 vs 154); `expert − r12s19X` **+0.125 [+0.075, +0.175]**.
  The rule judge's failures are geometric (41 of 51); the model's are semantic (18 main, 7 false dones) and its auxiliary residue is `q_speed`
  — the one execution answer it still gives.
* **Residual auxiliary streaks in arm X** are mostly `q_speed` (the model's own answer). The `q_force`, `q_gripper` and `q_path` streaks that
  remain (4, 3, 2) compare the *executed* answer with the expert's label: the harness caps the force near forbidden objects, keeps the
  gripper and discards auxiliary answers on switch ticks, and reroutes blocked paths — the expert itself shows such a streak once (force).

**Latency** (`latency.model_ms` excludes each episode's first tick; no CPU unit ran during either judged GPU run): r12s19X on ood_new2
non-first ticks p50 / p95 / p99 **43.4 / 55.9 / 80.0 ms**, 0 of 19,517 over 100 ms, observation → command p95 61.7 ms; first ticks p50 92.0,
max 244.9 ms, 10 of 200 over 100 ms. r12s19G on ood_new2 43.4 / 55.7 / 78.0 ms, 0 of 18,917 (first ticks 13 of 200). The five X runs on
dev_new2: p95 56.7–57.2 ms, 1 of 49,985 non-first ticks over 100 ms (r5X), first ticks 6–13 of 100 over 100 ms. The gate (p95 ≤ 80 ms,
> 100 ms ≤ 5 %) passes. The three rules run on the CPU after the model, outside `model_ms`; observation → command p95 is 61.7 ms (X) vs
61.6 ms (G) on ood_new2.

## 6. Measured cost

| unit (systemd, `Started` → end) | what | wall |
| --- | --- | ---: |
| `r13-pytest-a` | GPU: full suite after Stage A (fence 0.6 + `require_free(36 GiB)` first; nothing else on the GPU) | 0 h 09 m 56 s |
| `r13-c-dev` | GPU: the five checkpoints in arm X on dev_new2 (100 each) | 1 h 04 m 20 s |
| `r13-c-judged` | GPU: r12s19X and r12s19G on ood_new2 (200 each) | 0 h 47 m 24 s |
| `r13-pytest-b` | GPU: full suite at the end (same checks) | 0 h 09 m 52 s |
| **GPU total** | | **7,892 s = 2.19 GPU-h** (brief ≈ 2–3 h) |

CPU units (`r13c-*`, CUDA hidden, MemoryMax, `choom -n 1000`, Nice 10): the agreement check over every stored R4–R12 closed-loop tick (2 m 51 s,
then 2 m 42 s from the commit), the ood_new2 seed walk (26 s) and overlap, ruleX on dev_new2 (3 m 03 s), the choice (1 m 18 s), the four CPU
policies on ood_new2 (6 m 06 s – 6 m 55 s, in parallel, GPU idle), the final report and verdict (14 m 11 s, MemoryMax 40G — it did not
overlap any GPU unit), the band / contact probe (21 s) and the per-profile intervals (2 s), and the development test units (`r13c-pt-*`).
Disk: `/` 49 → 46 GB free (10⁹ B; closed-loop records 1.7 GB, scratch ≈ 0.2 GB). `journalctl -k` since 2026-10-08 15:16: 9 kernel lines, no
`NVRM`, OOM or killed-process line (the same query shows R9's seven `NVRM` lines of 2026-09-30). Cumulative GPU on the DGX Spark ≈ 84.2 h.
Cloud spend 0.

**Every CPU job that read records ran as a unit (HANDOFF §5).** Outside units the session shell read only tracked files, scripts, logs, report
JSONs, scratch tables derived from records, and dataset-manifest metadata — never an episode record; the list with times, including the reads
while a GPU unit was up (log tails during `r13-pytest-a`, `r13-c-dev` and `r13-c-judged`; six run files' `episodes_dir` fields during
`r13-c-dev`; `r13-selection.json` during `r13-c-judged`), is `artifacts/scratch/r13/session-shell-reads.md`.

## 7. What the round says, and what it does not

1. **Registered call: pass — "통과 — 그리퍼·경로·힘을 실행 쪽에 맡기면 검증으로 고른 구성이 처음 보는 홀드아웃에서 규칙 판정기보다 확실히 뒤지지
   않고 금지 접촉도 확실히 많지 않다".** The configuration chosen on validation (r12s19X, 79 of 100 on dev_new2, the highest of ten) against
   the rule judge given the same three execution rules, on 200 scenes no policy had run: strict `ruleX − r12s19X` **−0.025 [−0.080, +0.035]**
   (149 vs 154; 15 vs 20 discordant, p = 0.50) — no difference found, which is not "equal" and not "better"; forbidden-contact onsets
   `r12s19X − ruleX` **+0.015 [0.000, +0.040]** (5 vs 2) — not clearly more, with the lower bound exactly at 0: two seeds, both against the model,
   carry the whole difference, and the same two seeds carry the same contacts when that checkpoint drives its own path and force (arm G). A
   pass is grounds to review a contract change (§12 of docs/08 has the proposal); it is not the decision, which is the user's.
2. **Handing the path and the force to the execution side removed this checkpoint's auxiliary stalls.** The same checkpoint, X − G on
   ood_new2: strict **+0.115 [+0.065, +0.170]** (154 vs 131; 27 vs 4 discordant), auxiliary failures **−0.130 [−0.180, −0.080]**; 19 of the 27
   seeds it gained were `q_path` stalls in arm G. The gain sits in E2 (+0.238 [+0.125, +0.350] against +0.033 [−0.008, +0.075] on E0 + E1),
   where R12's gap to the rule judge sat. It depends on the checkpoint: on dev_new2 X − G is +0.100 [+0.030, +0.170] (r12s19), +0.070 [−0.010,
   +0.150] (r12s17), +0.060 [+0.020, +0.110] (r12s18), −0.020 [−0.060, +0.020] (r6), −0.010 [−0.050, +0.020] (r5) — a gain for the three R12
   checkpoints whose auxiliary heads stalled them, no difference found for r6 and r5.
3. **What separates the model from the rule judge in arm X is semantic.** Seed-paired `r12s19X − ruleX`: false dones +0.035 [+0.010, +0.060]
   (7 vs 0; 4 stale-goal), main-decision failures +0.080 [+0.040, +0.120] (18 vs 2), `q_stop` caught 0 of 64 (the rule judge 56 of 65),
   unsafe ticks +0.012 [+0.005, +0.021]; auxiliary failures +0.015 [−0.020, +0.050] (no difference found — the model's residue is `q_speed`,
   the one execution answer it still gives). The rule judge's failures are geometric (41 of 51). The two lose different seeds — 15 seeds only
   the rule judge solved (the model's main-decision stalls 5, `q_speed` stalls 7, false dones 3) and 20 only the model solved (the rule
   judge's geometric stalls 18) — and on these 200 the counts did not separate them. The expert still leads the model by +0.125 [+0.075,
   +0.175].
4. **The path rule's information boundary has a measurable cost.** The rule does not read which objects are forbidden (instruction
   information), so in the 60 mm forbidden band it holds where the expert retreats (§0). The rule judge pays for it: `ruleX − ruleG` strict
   −0.030 [−0.055, −0.010] (0 vs 6 discordant; 4 of the 6 are exactly its band episodes, ended in `q_path` stalls). The model meets the band
   too (198 ticks in 18 episodes; 4 of its 15 losses to ruleX are band episodes). The harness (h0.9, unchanged) already receives the
   structured forbidden constraints and uses them for its planner margins and force cap; whether an execution-layer path rule may read them
   is part of the review, not settled here.
5. **The stop is unchanged** — `q_stop` caught 0 of 64 stop onsets; arm X does not delegate it.
6. **What the round does not say.** One configuration (training seed 19, chosen on validation), one holdout of 200, a rule judge that reads
   the structured goal while the model reads the instruction text, and a comparison inside arm X (both sides use the same execution
   rules). dev_new2's (a) −0.150 [−0.240, −0.070] is the choice set's, favoured by the choice, and is written beside only. The second-best
   configuration on dev_new2 was r6G (74) — an arm G configuration; X's advantage there was checkpoint-dependent. The rule judge in arm X is
   weaker than in its own arm (149 vs 155) — the verdict compares the model with that judge, as registered; against ruleG (a cross-arm
   comparison, descriptive) the model reads +0.005 [−0.055, +0.070] (155 vs 154), no difference found either.

**Next step (registered: written from the numbers only).** The decisions are the user's: whether to review the contract change in docs/08
§12 (the model = semantic judgment; gripper, path and force = the execution layer; harness h0.10, which re-digests every checkpoint; the
forbidden band), and the cloud question. For the model, what remains is semantic — main-decision stalls, stale-goal false dones, `q_stop`
— and `q_speed`. ood_new2 (1250100~) has now been analysed seed by seed, so any further verdict needs another unused seed walk.
