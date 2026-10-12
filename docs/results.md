# robo-jev — detailed results

This page keeps the numbers and caveats behind the [README](../README.md). All comparisons are paired (same records, same prompts), single option order unless noted, with 95 % record-bootstrap intervals in brackets. Sealed test sets:

| Set | Measures |
|---|---|
| T1 | same distribution as training |
| T2 | unseen text style, settings and wording |
| T3 | unseen variable ranges (faster, taller, older sightings) |
| T4 | an unseen combination of phenomena |
| T5 | one-fact minimal pairs — does the judgment move the right way? |
| T6 | perturbations: paraphrase, option order, keys, formatting, units, number format, mirror, fact order |
| T7 | calibration |
| G1 | general reading: non-robot rules, workflows, page layouts, spatial facts (mostly Korean) |
| G2 | general reading: public BoolQ + MultiNLI (1,000 questions) |

The physics sets come from our own simulation engine: the training generator cannot produce their held-out cells, but they share its phenomena and text skins. Tests outside this engine are still ahead.

## First trained version (v0)

The 2B backbone (Qwen3.5-2B) was trained once (LoRA, 50,000 generated requests, one epoch, 62 minutes on one H100) and scored on the sealed sets with the same prompts as the untrained model. The table shows untrained → trained, single option order (the deployed condition), with the paired 95 % interval of the difference:

| Set | yes/no Brier | score Brier | action regret (no stated criterion) | action accuracy |
|---|---|---|---|---|
| T1 same distribution | 0.180 → 0.034 [−0.152, −0.141] | 0.190 → 0.012 | 9.9 → 0.8 | 0.37 → 0.71 |
| **T2 unseen style, settings, wording** | 0.167 → 0.059 [−0.115, −0.102] | 0.183 → 0.018 [−0.169, −0.160] | 12.4 → 1.6 [−11.8, −9.9] | 0.34 → 0.67 |
| T3 unseen variable ranges | 0.169 → 0.053 | 0.181 → 0.017 | 8.2 → 1.2 | 0.39 → 0.66 |
| **T4 unseen combination** | 0.197 → 0.025 [−0.178, −0.166] | 0.204 → 0.021 [−0.190, −0.176] | 22.8 → 1.4 [−23.1, −19.9] | 0.29 → 0.75 |
| T7 calibration (ECE, yes/no / score) | 0.134 / 0.193 → 0.014 / 0.004 | | | |

- **It stops deferring.** The untrained model follows the planner's recommendation about two times in three whether it is right or wrong. The trained model follows wrong recommendations 5–12 % of the time and right ones as often as before (about 65 %).
- **It stops depending on option order.** Answer flips when only the option listing changes fell from 0.53 to 0.08. Confident flips (both answers decisive) fell from 0.12 to 0.
- **Changing one fact moves it the right way.** In minimal pairs it follows the moved label 90 % of the time, versus 55 % before training.
- **Size alone does not do it.** The untrained 4B is no better than the untrained 2B (T2 yes/no Brier 0.207 vs 0.167).

**Weaknesses v0 exposed:**
- **Wording.** A counterfactual question ("if you recentre it first, does it come off the stack?") is answered perfectly in the training wordings but worse than untrained in T2's held-out wording. Training saw only three wordings per question.
- **Fact order.** Shuffling the order of facts moves the trained model's answers more than the untrained model's (mean shift 0.071 vs 0.034).
- **General reading got worse.** Measured after the physics sets:
  - On 2,082 non-robot reading questions (rules, workflows, page layouts, spatial facts; mostly Korean), accuracy fell from 0.484 to 0.429, a drop of 0.055 [0.030, 0.083].
  - On 1,000 public questions (BoolQ and MultiNLI), it fell from 0.739 to 0.650, a drop of 0.090 [0.062, 0.116]. Nearly all of that loss is on MultiNLI, where the trained model rarely answers "contradiction".
  - The clearest mechanism is a learned *stop* reflex. Asked which action a written rule requires, the trained model answers "stop immediately" in 97 of 105 questions; the rule requires stopping in 8, and the untrained model answers stop in 37.
  - Physics training taught it that stopping is usually safe, and that now overrides what the text says.
- **Not yet tested outside this generator family.** All seven sets come from the same engine with held-out cells, so outside tests are still to come.

## v0.1 result

v0.1 ran as two variants, each trained once with v0's recipe on the same 50,000 moments with the same labels:
- **v0.1-main (data fixes only):** 6–10 training wordings per question instead of 3, and fact order shuffled in half of the texts.
- **v0.1-mix:** the same, plus about 2.7 % of training examples from the non-robot reading set's training split.

The pass rule was written before any v0.1 data existed. It compares each variant with v0 on the same records and hardware, and has three parts:
- (A) better on T2's held-out wordings;
- (B) fact-order sensitivity no higher than the untrained model's 0.034;
- (C) no loss elsewhere beyond small fixed margins.

| | (A) held-out wording, yes/no Brier vs v0 | (B) fact-order shift | (C) no loss elsewhere | verdict |
|---|---|---|---|---|
| v0.1-main | −0.013 [−0.017, −0.008] | 0.028 | default-criterion action regret up on T1 (+0.39 [+0.14, +0.72]) and T5 (+0.28 [+0.09, +0.50]) | does not pass |
| v0.1-mix | −0.017 [−0.023, −0.012] | 0.027 | regret up on T1 (+0.44) and T4 (+0.49); T1 yes/no Brier +0.008 | does not pass |

- **Wording fixed.** The question v0 broke on — "if you shift it to the centre first, does it come off the stack?" in the held-out wording — went from 0.611 (v0) to 0.146 (v0.1-mix); the untrained model scores 0.173.
- **Fact order fixed.** The shift fell from 0.071 to 0.027.
- **Small in-distribution cost.** Default-criterion action regret on T1 rose from 0.84 to about 1.2–1.3; the untrained model scores 9.9. A likely cause: T1 and T7 are asked in the original wordings, and those now make up only about a third of the English training wordings instead of all of them.
- **General reading**, measured on the same hardware against the untrained 2B:

  | | G1 | G2 (public) |
  |---|---|---|
  | v0 | −0.057 | −0.088 |
  | v0.1-main | −0.061 | −0.131 |
  | v0.1-mix | +0.304* | −0.061 |

  - \* G1 is in the mix's own family, so this gain overstates it.
  - Broader physics wordings made public sentence inference worse (MultiNLI −0.25).
  - The 2.7 % general data won back part of it (+0.071 against v0.1-main).

## Benchmark: decision models on OpenRouter (bench-v0)

The same requests went to six decision models on [OpenRouter](https://openrouter.ai/models?output_modalities=decisions) through the Decisions API, and every model was scored with the same tools.
- **Physics:** a fixed stratified sample of the sealed physics sets (2,390 requests).
- **General reading:** the two general-reading checks (1,314 requests).
- **Cost:** $1.64 for all six.
- **Data collection:** denied, so requests went only to providers that do not store or train on them.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="figures/bench-v0-scatter-dark.svg">
  <img alt="Physical judgment vs general reading on bench-v0: our trained 2B models lead on the physics sets, the OpenRouter decision models lead on public reading" src="figures/bench-v0-scatter-light.svg">
</picture>

| Model | Physics skill T2–T4 ↑ | Yes/no Brier T2–T4 ↓ | Score Brier T2–T4 ↓ | Answered T2–T4 | G1 acc ↑ | G2 acc ↑ | $/M input |
|---|---|---|---|---|---|---|---|
| **Qwen3.5-2B (untrained)** | -0.148 | 0.177 | 0.186 | 100% | 0.485 | 0.738 | self-hosted |
| **ours v0 (2B)** | 0.777 | 0.048 | 0.019 | 100% | 0.428 | 0.650 | self-hosted |
| **ours v0.1-main (2B)** | 0.795 | 0.043 | 0.018 | 100% | 0.424 | 0.607 | self-hosted |
| **ours v0.1-mix (2B)** | 0.795 | 0.043 | 0.019 | 100% | 0.790 | 0.677 | self-hosted |
| liquid/d1 | 0.167 | 0.138 | 0.122 | 100% | 0.778 | 0.926 | 0.040 |
| microsoft/microsoft-decision-1 | 0.044 | 0.162 | 0.136 | 100% | 0.787 | 0.915 | 0.042 |
| openai/gpt-6-luna-decisions | -0.020 | 0.194 | 0.118 | 76% | 0.768 | 0.863 | 0.100 |
| ~typesafe/jev-latest | 0.277 | 0.123 | 0.102 | 100% | 0.791 | 0.909 | 0.042 |
| togethercomputer/tev1-4b-experimental | 0.027 | 0.162 | 0.141 | 100% | 0.684 | 0.903 | 0.042 |
| upstage/solar-decide | -0.150 | 0.144 | 0.230 | 100% | 0.769 | 0.864 | 0.050 |

- **Physics (T2–T4).** These sets come from our own engine — home ground.
  - The trained 2B models predict the simulated outcomes far better than any OpenRouter model: physics skill 0.78–0.80, against 0.28 for the best (Jev).
  - Most OpenRouter models score near "always answer 0.5", like the untrained 2B.
- **Public reading (G2).** The OpenRouter models lead: 0.86–0.93, against 0.61–0.68 for ours.
  - The 2B backbone starts lower (0.74).
  - Physics training has cost part of that so far; see [v0.1](#v01-result).
- **Refusals.** GPT-6 Luna Decisions declined about a fifth of the physics requests (507 of 2,390, "refused to answer") and none of the reading requests; its scores cover the rest.
- **How each side was asked.**
  - OpenRouter models were asked once each, in the stored option order. Ours are averaged over balanced option orders.
  - Score questions were sent with their two anchors as written. The API accepts at most 10 levels, and an 11-level variant made no difference on the one model that accepted it.
  - Action-choice metrics depend on our value table, so they are left out here.
  - Definition: `configs/bench/bench-v0.yaml`; run with `scripts/bench.py`.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="figures/bench-v0-sets-dark.svg">
  <img alt="Per-set results on bench-v0: yes/no Brier on T1–T7 and reading accuracy on G1 and G2 for every model" src="figures/bench-v0-sets-light.svg">
</picture>

## Outside test (T8)

T8a is PIQA validation (1,838 two-choice questions). T8b has 480 robot moments in four families: braking deck, turning deck, release over a stack edge, and push. Each moment is written independently of our engine, and its labels come from 64 MuJoCo rollouts over hidden friction and weight distribution. The OpenRouter models were asked once each with data collection denied (cost $0.29). The test was built and scored on 2026-10-12.

Single-order reading (deployment condition) unless noted: our models' numbers are expectations over balanced option orders; OpenRouter models were asked once in the stored order (data collection denied). Intervals are record-bootstrap 95 %.

### T8b composition

| family | moments | questions | majority outcome of the planned action |
|---|---|---|---|
| deck_brake | 150 | 600 | slide 50, stay 50, tip 50 |
| deck_turn | 90 | 360 | slide 30, stay 30, tip 30 |
| stack | 120 | 360 | fall 60, stay 60 |
| push | 120 | 360 | tip 60, upright 60 |

### Scores

| model | T8a PIQA acc ↑ | T8a acc (order-avg) | T8a first-option share | T8b physics skill ↑ | T8b skill (order-avg) | T8b yes/no Brier ↓ | T8b score Brier ↓ | T8b outcome acc ↑ | T8b answered | bench-v0 skill T2–T4 |
|---|---|---|---|---|---|---|---|---|---|---|
| **Qwen3.5-2B (untrained)** | 0.662 [0.649, 0.675] | 0.789 | 0.77 | -0.115 [-0.147, -0.084] | 0.026 | 0.217 | 0.207 | 0.347 | 1.00 | -0.148 |
| **ours v0 (2B)** | 0.700 [0.685, 0.717] | 0.736 | 0.49 | -0.431 [-0.549, -0.318] | -0.428 | 0.241 | 0.340 | 0.335 | 1.00 | 0.777 |
| **ours v0.1-mix (2B)** | 0.686 [0.670, 0.702] | 0.743 | 0.61 | -0.307 [-0.412, -0.213] | -0.296 | 0.232 | 0.286 | 0.335 | 1.00 | 0.795 |
| ~typesafe/jev-latest | 0.946 [0.936, 0.955] | 0.946 | 0.51 | 0.124 [+0.069, +0.179] | 0.124 | 0.181 | 0.142 | 0.492 | 1.00 | 0.277 |
| microsoft/microsoft-decision-1 | 0.915 [0.902, 0.928] | 0.915 | 0.51 | -0.253 [-0.337, -0.168] | -0.253 | 0.259 | 0.201 | 0.450 | 1.00 | 0.044 |
| togethercomputer/tev1-4b-experimental | 0.848 [0.831, 0.864] | 0.848 | 0.48 | -0.043 [-0.086, +0.001] | -0.043 | 0.203 | 0.193 | 0.362 | 1.00 | 0.027 |
| openai/gpt-6-luna-decisions | 0.920 [0.908, 0.932] | 0.920 | 0.54 | -0.086 [-0.169, -0.004] | -0.086 | 0.249 | 0.121 | 0.371 | 0.83 | -0.020 |
| liquid/d1 | 0.936 [0.925, 0.947] | 0.936 | 0.49 | 0.065 [+0.005, +0.127] | 0.065 | 0.190 | 0.158 | 0.425 | 1.00 | 0.167 |
| upstage/solar-decide | 0.899 [0.886, 0.913] | 0.899 | 0.53 | -0.042 [-0.096, +0.010] | -0.042 | 0.169 | 0.261 | 0.546 | 1.00 | -0.150 |
| _static calculator (reference)_ | — | — | — | 0.856 [+0.817, +0.888] | 0.856 | 0.027 | 0.028 | 0.958 | 1.00 | — |

### T8b skill by family and question kind (single order)

| model | deck_brake | deck_turn | stack | push | planned | counterfactual | yes/no only | score only |
|---|---|---|---|---|---|---|---|---|
| Qwen3.5-2B (untrained) | -0.058 | -0.041 | -0.184 | -0.205 | -0.146 | -0.082 | -0.136 | -0.073 |
| ours v0 (2B) | -0.178 | -0.054 | -1.214 | -0.563 | -0.246 | -0.630 | -0.262 | -0.766 |
| ours v0.1-mix (2B) | -0.108 | -0.008 | -0.502 | -0.685 | -0.156 | -0.469 | -0.217 | -0.485 |
| ~typesafe/jev-latest | 0.167 | 0.137 | 0.371 | -0.100 | 0.022 | 0.235 | 0.053 | 0.265 |
| microsoft/microsoft-decision-1 | -0.058 | -0.669 | 0.018 | -0.321 | -0.335 | -0.164 | -0.359 | -0.042 |
| togethercomputer/tev1-4b-experimental | 0.020 | -0.005 | -0.029 | -0.161 | -0.108 | 0.027 | -0.062 | -0.004 |
| openai/gpt-6-luna-decisions | -0.046 | -0.287 | 0.029 | 0.065 | -0.409 | 0.321 | -0.324 | 0.375 |
| liquid/d1 | 0.151 | 0.000 | 0.393 | -0.200 | -0.044 | 0.182 | 0.006 | 0.182 |
| upstage/solar-decide | -0.233 | 0.081 | 0.258 | -0.097 | 0.119 | -0.215 | 0.115 | -0.354 |
| static calculator (reference) | 0.810 | 0.975 | 0.776 | 0.864 | 0.863 | 0.849 | 0.857 | 0.854 |

### Ours vs the untrained base (paired, same questions)

| model | T8a Δacc (single) | T8a Δacc (order-avg) | T8b Δskill (single) | T8b Δskill (order-avg) | Δskill deck_brake | Δskill deck_turn | Δskill stack | Δskill push |
|---|---|---|---|---|---|---|---|---|
| ours v0 (2B) | +0.039 [+0.023, +0.056] | -0.053 [-0.075, -0.032] | -0.316 [-0.412, -0.225] | -0.454 [-0.550, -0.362] | -0.120 [-0.261, +0.012] | -0.013 [-0.206, +0.160] | -1.030 [-1.254, -0.805] | -0.357 [-0.551, -0.174] |
| ours v0.1-mix (2B) | +0.024 [+0.008, +0.041] | -0.047 [-0.069, -0.024] | -0.192 [-0.274, -0.115] | -0.322 [-0.405, -0.244] | -0.050 [-0.170, +0.062] | +0.033 [-0.123, +0.175] | -0.318 [-0.458, -0.173] | -0.480 [-0.699, -0.273] |

### Ours vs the best OpenRouter model of each set (paired, questions both answered, single order)

| model | T8a Δacc vs best | T8b Δskill vs best |
|---|---|---|
| Qwen3.5-2B (untrained) | -0.284 [-0.298, -0.269] (~typesafe/jev-latest) | -0.239 [-0.279, -0.200] (~typesafe/jev-latest) |
| ours v0 (2B) | -0.245 [-0.262, -0.227] (~typesafe/jev-latest) | -0.555 [-0.651, -0.467] (~typesafe/jev-latest) |
| ours v0.1-mix (2B) | -0.260 [-0.276, -0.243] (~typesafe/jev-latest) | -0.431 [-0.518, -0.349] (~typesafe/jev-latest) |

OpenRouter spend (T8 ledger): $0.2854 over 14704 calls (uncertain $0.0001).
