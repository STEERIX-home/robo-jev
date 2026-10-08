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
