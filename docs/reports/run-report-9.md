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
