# Run report 5 — auxiliary-head stability: R5's recipe with DAgger cycle-1 material at seeds 18 and 19, per-question losses while training (2026-09-29)

The document docs/06 Task 6 defines (:384) for the fifth and sixth trained checkpoints of this project (Task R8). Every number
is copied from an artifact named next to it. Korean prose lives in `docs/06`, `docs/08` and `HANDOFF.md`; this report is the
English ledger. Run reports 1–4 stay as they are; this one only adds. **Sections are appended as each stage's numbers land;
§0 was committed before any Stage C or D number existed.**

## 0. Pre-registration (written 2026-09-29 before any Stage C or D number; machine-readable copy `configs/eval/r8-registration.yaml`)

The rule below is applied by code, not by hand: `scripts/closed_loop.py verdict --report artifacts/reports/r8-closed-loop.json
--registration configs/eval/r8-registration.yaml --out artifacts/reports/r8-verdict.json` first reads each seed's head-fit
monitor from its run (`artifacts/runs/r8-t1-fp32-2b-s{18,19}/metrics.json`, `monitor_verdicts` — it refuses to judge if the
monitor the training applied differs from the registered one) and then runs `robo_jev.closed_loop.apply_registration` on the
report (R7's function, extended with seed groups, a stopped-seed rule and the stability readings; tests in
`tests/test_closed_loop.py` pin every bound, the per-seed grouping, the stopped seed and the monitor check).

**Why this round.** From R5 to R6 to R7 the main-decision head (instruction reading) was robust to recipe changes
(primary-stratum instruction-shuffle margin +0.298 / +0.374 / +0.370) but the auxiliary heads were not: the gripper head went
right (R5, settled 99.6 %) → hasty (R6, more duplicate transitions) → never fit during training (R7, settled 4.3 %, loop strict
0/200), and offline `q_stop` caught 5 → 0 → 0 of 10 stop onsets. Every comparison so far was **one seed**. The best closed
loop so far is r5 (ood_dev 200 strict 132, false done 30; rule judge 145, expert 182); r6 repaired the false dones (12) at
strict 122. This round runs **R5's recipe unchanged** plus the **cycle-1 DAgger material**, at **two seeds**, and logs the
per-question losses so the heads' fit is watched during training for the first time.

**Recipe** (`configs/train/qwen35-2b-r8.yaml`; a test pins that against R5's resolved config only the data list, the run name,
the kept-step list and the monitor differ, and against R7's only the mixture, `max_steps`, the name and the monitor). Qwen3.5-2B
bf16 with fp32 master weights, full text backbone + fp32 pointer readout rank 64, `backbone_lr` 1e-5, `readout_lr` 3e-4,
5-second TBPTT chunks, 30-tick window, tick weights 0.25 / 2 / 2 / 1, candidate permutation seeded by the run seed,
**material shares at the default 0.7 / 0.2 / 0.1** (the empty new-semantic bucket renormalizes them to 0.78 / 0.22 of the robot
units), **`max_steps` 233** (R5's schedule), `checkpoint_every` 50, no kept steps. Data: the g2 expert episodes (233 train),
DAgger-0 (200), DAgger-1 (200) and the done-gate collection (200) — the three DAgger sets as one `error_family` bucket — and the
non-robot split-by-split set. Expected draws: expert ≈ 181, DAgger ≈ 52 (of which cycle-1 ≈ 35). **Seeds 18 and 19**, run ids
`r8-t1-fp32-2b-s18` / `-s19`, closed-loop labels `r8s18` / `r8s19`. The instrument check: seed 18's step-1 loss must equal R5
seed 18's `2.633075326681137` (the logging does not change training).

**Head-fit monitor (soft stop).** At step 150, if the `q_gripper` per-question weighted loss averaged over steps 131–150 is
≥ **0.70 ×** its baseline averaged over the same steps, the run stops (status `stopped_head_not_fitting`), its metrics and
checkpoint are kept and its closed loop is skipped; **a stopped seed fails all of its cloud conditions**. The baseline is the
loss of a constant-prior head that outputs the step's label marginal (Task R8 A1; the step's robot batch is one episode).
**The constant was changed before the first training step, as the brief allows only after A3:** the brief's 0.95 → 0.70.
Why (A3, `artifacts/reports/r8-a3-teacher-forced.json`): r7 run teacher-forced on the ten g2 train episodes it drew last
(steps 286–304) has a `q_gripper` weighted loss of 0.3899 against a baseline of 0.5306 — **0.735 × baseline** — while its head
does not work on those very episodes (settled 13/336 = 0.039, initiate 0/11); the training computation and the C2 evaluation
computation agree to the argmax on all 1,018 ticks. 0.95 would read that head as fitting. r5 on the same ten episodes (which it
never drew) is at 0.067 × baseline (settled 336/336, initiate 9/11). 0.70 is the largest round constant below r7's measured
state and ten times r5's.

**Scenes.** Primary **`ood_dev200`** = R4's 26 ood_dev seeds + R6's 74 + R7's 100 (`r4-seeds.json` `ood_dev`, `r6-seeds.json`
`ood_dev_new`, `r7-seeds.json` `ood_dev_new2`), merged by `report --merge ood_dev200=ood_dev,ood_dev_new,ood_dev_new2`; 0 origin
groups shared with training. Secondary **`dev_new2`** 100 (R6), beside the verdict only.

**Rows.** expert · rule judge · mechanical and r5 · r6: the R4 / R6 / R7 runs (as in run report 4). r8s18 and r8s19: this
task, all four conditions (ood_dev, ood_dev_new, ood_dev_new2, dev_new2).

**Pairing and interval.** By seed (`profile:seed`), 95 % percentile bootstrap over seeds, 2,000 resamples, RNG seed 20260921
(`EPISODE_BOOTSTRAP`); `paired_success(strict=True)`, `paired_false_done`, `paired_seed_ratio`. Zero inside = not a finding.
Strict success is primary. Latency reports each episode's first tick separately.

**Cloud rule — R7's three conditions, for each seed s ∈ {r8s18, r8s19}, on `ood_dev200`:**
* **(a)** `rule − s` strict success: interval **lower bound ≤ 0**.
* **(b)** `s − r5` false-done rate: interval **upper bound < 0**.
* **(c)** `s − r5` duplicate gripper transitions per episode: interval **lower bound ≤ 0**; **and** `s − r5` `q_stop` catch rate:
  interval **upper bound ≥ 0**.

**The cloud is recommended only if both seeds pass all three** (all eight registered conditions hold). A seed stopped by the
monitor, or without a completed run, fails its four. Next to (a) and (b) for each seed: discordant seeds, exact McNemar p and the
bound distribution under bootstrap RNG seeds 1–200 — they do not change the verdict. If the cloud is recommended, the provider,
account and first paid run remain the user's decision.

**Stability reading (beside the verdict, not part of it).** `r8s18 − r8s19` paired strict success and false-done rate; the
seed-level auxiliary differences (duplicate transitions, gripper-streak episodes, `q_stop` catch rate, unsafe ticks, auxiliary
failures); both seeds' monitor results (the per-question loss curves against their baselines). An interval containing zero
means no difference between the seeds was found, not that they are equal. `dev_new2` is secondary evidence.

**Also reported, not part of the rule.** Stage C: realized units by material, the per-question loss curves against their
baselines (`q_gripper`, `q_stop`, `q_main`), the in-training gripper and stop probes, wall and peak memory. Stage D offline, per
seed, on the same cells (hashes `6a3b69131243`, `9b484441b23b`): the primary-stratum instruction-shuffle margin, `q_gripper`
settled / initiate / open, `q_stop` onsets, the `q_done` strata. Seed-paired `r8 − r5` closed-loop tables.
