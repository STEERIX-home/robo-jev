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

## 1. Trained weights

| run | steps | seed | checkpoint | contract digest | training | evaluated by |
| --- | ---: | ---: | --- | --- | --- | --- |
| **`r7-t1-fp32-2b-s18`** | 304 = one schedule, one process (no resume) | 18 | `artifacts/runs/r7-t1-fp32-2b-s18/checkpoint.pt` (24.5 GiB; model + optimizer + fp32 masters) | `93fe26725a4c…` (unchanged) | 18,361.1 s of unit wall = **5.10 GPU-h** (train seconds 17,384.0; 57.18 s/step), peak 56.05 GiB, loss 2.633 → 0.128 | R7 C2 (cells), **R7 D (closed loop)** |
| `r6-t1-fp32-2b-s18`, `r5-t1-fp32-2b-s18` (baselines) | 233 each | 18 | `artifacts/runs/r{6,5}-t1-fp32-2b-s18/checkpoint.pt` | same | run reports 3 and 2 | R7 B3 (the 100 new ood_dev seeds) |

Recipe: R6's (`configs/train/qwen35-2b-r7.yaml` extends `qwen35-2b-r6.yaml`) — Qwen3.5-2B bf16 with fp32 master weights, fp32
pointer readout rank 64, full text backbone, `backbone_lr` 1e-5, `readout_lr` 3e-4, 5-second TBPTT chunks, 30-tick window,
tick weights 0.25 / 2 / 2 / 1, seed 18, the same five data manifests (the non-robot one in its split-by-split layout). **Two
recipe numbers differ, and a test pins that it is exactly these two** (plus the manifest path and the run name):
`sampler.material_shares` existing **0.6** / error family **0.4** / new semantic family 0.0 (R6 0.5 / 0.5) and `max_steps`
**304** (R6 233). Designed draws: expert 0.6 × 304 ≈ 182 (R5's level), DAgger 0.4 × 304 ≈ 122 (R6's level). **Realized**
(`metrics.json` `summary.sampler.units` and `steps[].units`): **expert 195 (0.84 epoch of 233) and DAgger 109** (dagger-0 42,
dagger-1 33, done-gate 34) — the seeded material coin came out 1.5 SD from its mean, so against R6 this run has **+84 expert
draws and −13 DAgger draws** rather than "+71 and the same". Of the 43 episodes whose model raw `q_done` rose on a
reference-False tick, **10** were drawn (3 + 7; R6 9). Non-robot 1,818 bundles, tokens 14.99 M (R6 11.55 M: 71 more steps).

## 2. Reproducible data

The training data list is R6's, with one change of **layout, not content**: the non-robot set is the split-by-split
re-layout of A2. The robot manifests are unchanged.

| dataset | what | manifest (sha256 of the manifest file) | version |
| --- | --- | --- | --- |
| robot train labels, gripper rule v2 (unchanged since R5) | the 400 R1 v0.2 episodes, `q_gripper` under rule v2; train 233 episodes | `artifacts/datasets/r1-robot/r1-rollout-labels-g2/manifest.json` `4d433bae6d22…` | `labels-rollout-v1+gripper-v2` |
| DAgger cycle 0 (unchanged since R5) | R4's dev-condition model-driven records, 200 episodes | `artifacts/datasets/r5-dagger/dagger-0/manifest.json` `8b205ee13993…` | `gripper-v2`, `dagger-v0.1` cycle 0 |
| DAgger cycle 1 — model loops (unchanged since R6) | R5's dev-condition model-driven records, 200 episodes | `artifacts/datasets/r6-dagger/dagger-1/manifest.json` `6351ec247d35…` | cycle 1 |
| DAgger cycle 1 — done-gate collection (unchanged since R6) | 200 train-family episodes with the expert's `q_done` at the harness | `artifacts/datasets/r6-dagger/dagger-1-donegate/manifest.json` `6b44fc3e3492…` | cycle 1 |
| **non-robot single requests, split by split (A2)** | the same 4,200 states as `r1/single` (`46e7e361719b…`; the run reports 1–3 wrote "2,000 records" — that is D1's pilot set; this one has 4,200 states, 2,620 of them `train`), regenerated with the same generator, config and seed and written one file per split; the one-file sha256 **`2ed780fd…` equals the old manifest's**, so the records are byte-identical and the old file was not opened | `artifacts/datasets/r1/single-by-split/manifest.json` `cfd632dd568f…` | `gen-single-v0.3.0`, `layout: by-split` |
| closed-loop scenes (R7) | `ood_dev_new2` 100 new ood_dev seeds (base 1050100; E0/E1/E2 20/40/40; 41 origin groups, **0** shared with training); **ood_dev 200** = R4's 26 + R6's 74 + these 100 | `artifacts/reports/r7-seeds.json` (+ `r4-seeds.json`, `r6-seeds.json`); overlaps `r7-seeds-overlap.json`, `r7-seeds-crosscheck.json` | `cl0.1`, ids `-r7-<label>` |

**Seal hygiene (Task R7 A1–A3).** From R2 to R6 the training loader hashed and parsed every file a manifest listed and only
then dropped the records outside the requested split — every training run on the R1 robot corpora (33 sealed episode
files) and on the non-robot `r1/single` (all splits in one `records.jsonl`) opened and parsed sealed records, and so did the
evaluation-cell loader. The impact was nil: in the old code the split test came right after the parse and before tagging,
serialization, `Item` creation and indexing, so a dropped record had no tokens and no index; R6 loaded exactly the
train-split counts (3,453 = 833 + 2,620), and the new loader — which is tested never to open a sealed file — reproduces
the decision cell's identity hash `6a3b69131243`. The loader now decides which files to open from the manifest before
opening any, refuses entries that do not say their split, and cannot load the sealed split; the evaluation and analysis
readers of episode directories follow the same rule (`robo_jev.data.sealed`). An old run cannot be resumed with the new
loader: its non-robot manifest is refused, and on the re-layout the dataset identity and the sampler's record sources differ.
The audit of every other path is in `.superpowers/sdd/task-r7-report.md` A3.
