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
