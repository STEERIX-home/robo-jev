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
