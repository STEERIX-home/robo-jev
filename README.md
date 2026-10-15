# robo-jev

**A small, fast decision model for robots — specialized in physical judgment.**

robo-jev reads a short text of what a robot senses, answers the questions that matter in the moment — *will the glass tip if I brake now? does the person reach my path first? did the part seat?* — and returns a calibrated probability for every answer in one forward pass of a 2B model.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/figures/bench-v0-physics-dark.svg">
  <img alt="Physical judgment on our simulation benchmark: robo-jev v0.7 (2B) scores 0.81 and the previous release v0.1 0.80, against 0.28 for TypeSafe Jev, 0.17 for Liquid d1 and near zero or below for Microsoft-Decision-1, Together Tev1 4B, OpenAI GPT-6 Luna Decisions, Upstage Solar Decide and the untrained Qwen3.5-2B" src="docs/figures/bench-v0-physics-light.svg">
</picture>

General decision models read the *meaning* of a situation well, but on our physics benchmark most of them score close to always answering 50 %. robo-jev learns physics from simulation and carries it to new wording, out-of-range values and unseen combinations *within our simulator*.

> **Honest status (2026-10-16).** The published model is now **v0.7**. v0.7 did not pass its own pre-declared rule (S REJECTED; guards O, N and F fail). It is released because it is clearly better than the public v0.1.
> - On moments written outside our engine ([Outside tests](#outside-tests)), it calls fewer risky situations "safe" than v0.1, but still more than the untrained base.
> - It takes choice questions with up to 26 options; choosing among many *plausible* options is untested.

## Highlights

- **Physical judgment on our benchmark, far ahead.** On 1,100 held-out moments from our simulator, robo-jev v0.7 scores 0.81 physics skill (v0.1: 0.80).
  - TypeSafe Jev, the best general decision model, scores 0.28. Liquid d1 scores 0.17.
  - The rest — Microsoft-Decision-1, Together Tev1 4B, OpenAI GPT-6 Luna Decisions, Upstage Solar Decide — score about zero or below.
- **Generalizes inside the simulator.** The gains hold on:
  - text styles, settings and wordings it never saw;
  - values beyond its training range;
  - combinations of phenomena it never saw together.

  Change one fact and the answer moves the right way 92 % of the time (55 % before training).
- **Outside it: better than v0.1, not yet safe.** On independently written, MuJoCo-labelled moments (T8b), v0.7 makes fewer dangerous "safe" calls than v0.1 (0.51 vs 0.65) and gets more clear cases right (0.47 vs 0.38).
  - The untrained base still makes the fewest dangerous "safe" calls (0.34).
- **Up to 26 options.** A choice question can list up to 26 options. On held-out physics questions, accuracy drops by only 0.012 from 8 options to 17–26 (two-seed mean).
- **Calibrated.** Its 70 % means 70 %. Expected calibration error is 0.011 on yes/no questions, against 0.134 for the untrained base.
- **Judges the advice it gets.** It follows a wrong planner recommendation 9 % of the time (65 % untrained) and still follows right ones (65 %).
- **Stable.** Reordering the options flips 5–10 % of answers, against 50–65 % for the untrained base.
- **Small and drop-in.**
  - One forward pass of Qwen3.5-2B with a LoRA adapter, trained in about three and a quarter hours on one H100.
  - Requests use the same shape as the [OpenRouter Decisions API](https://openrouter.ai/docs/api/api-reference/alphadecisions/submit-a-decisions-request): a state text plus 1–12 `choice` (2–26 options), `noul` (yes/no) or `score` questions.

## What it judges

| Phenomenon | Example question |
|---|---|
| People and robots crossing paths | Does the worker step into my lane before I can stop? |
| Loads tipping or sliding | If I brake hard now, does the tall box topple? |
| Liquids spilling | Does the coffee slop over the rim in this turn? |
| Parts seating and placement | Is the plate actually seated in its nest? |
| Stale or conflicting senses | Is the person I saw 3 s ago still in the way? Should I look again? |

Each answer is a probability, so the robot's planner can weigh risk, time and cost instead of trusting a yes/no.

robo-jev is a **System 1**: a fast, integrated judgment about which way the risk moves, roughly how big it is, which action is safer, and when to look again or call a person. Computing the exact physical numbers is the job of the control layer below it, **System 0**. robo-jev's training and scoring are moving to these judgments (risk level, comparison, direction, safer action) and away from exact probabilities.

## How it works

1. **Simulate phenomena, not whole scenes.** An engine samples decision moments across ten domains (warehouses, cafés, hospitals, homes, sidewalks …) and writes them as text in several styles, in English and Korean.
2. **Label with probabilities, not opinions.**
   - What the robot cannot see — mass, friction, reaction time, fill level — is sampled many times.
   - Each answer's label is the probability of the outcome.
   - Actions are labelled by their expected cost.
3. **Hold out what matters.** Sealed test sets cover each kind of novelty, and the training generator cannot produce them:
   - new style, settings and wording;
   - out-of-range values;
   - unseen combinations;
   - one-fact pairs;
   - perturbations.
4. **Train small.** Train a LoRA adapter on Qwen3.5-2B with the same prompts that score it.

More in [docs/method.md](docs/method.md).

## Benchmark

We sent the same requests to six decision models on [OpenRouter](https://openrouter.ai/models?output_modalities=decisions) and scored every model the same way. The test, *bench-v0*, has 3,704 requests:
- 2,390 physics moments from our sealed sets;
- 1,314 general-reading questions, 1,000 of them public (BoolQ, MultiNLI).

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/figures/bench-v0-scatter-dark.svg">
  <img alt="Physical judgment vs general reading on bench-v0: robo-jev leads on physics, the OpenRouter decision models lead on public reading" src="docs/figures/bench-v0-scatter-light.svg">
</picture>

| Model | Physics skill ↑ | Public reading (BoolQ + MultiNLI) ↑ | $/M input tokens |
|---|---|---|---|
| **robo-jev v0.7 · 2B** | **0.81** | 0.71 | self-hosted |
| robo-jev v0.1 · 2B (previous release) | 0.80 | 0.68 | self-hosted |
| TypeSafe Jev | 0.28 | 0.91 | 0.042 |
| Liquid d1 | 0.17 | **0.93** | 0.040 |
| Microsoft-Decision-1 | 0.04 | 0.92 | 0.042 |
| Together Tev1 4B | 0.03 | 0.90 | 0.042 |
| OpenAI GPT-6 Luna Decisions | −0.02 ¹ | 0.86 | 0.100 |
| Upstage Solar Decide | −0.15 | 0.86 | 0.050 |
| Qwen3.5-2B, untrained | −0.15 | 0.74 | self-hosted |

robo-jev is a specialist. It leads on physics by a wide margin; on general reading the larger general models are ahead, and robo-jev is still below its own untrained base (0.74).

- **Physics skill** is 1 − Brier ÷ Brier of always answering 50 %, over the yes/no and score questions of the held-out sets. 0 means no better than a coin flip; 1 is perfect.
- **Home ground.** The physics moments come from our own simulator and are held out from training. For tests outside this engine, see [Outside tests](#outside-tests).
- **One call each.** OpenRouter models were asked once per request through the Decisions API (2026-10-12), with data collection denied.
- ¹ GPT-6 Luna Decisions declined about a fifth of the physics requests; its score covers the rest.

All numbers, intervals and caveats are in [docs/results.md](docs/results.md).

## Outside tests

Every set above comes from our own engine. The outside tests are written by authors who never saw its code or text, and labelled by MuJoCo rigid-body simulation over hidden physical factors (such as friction and weight distribution):
- **T8b** — 480 robot moments in a first-person voice: a braking deck, a turning deck, a box released over a stack's edge, and a pushed box.
- **T9** — a second outside test, built later by a new author.
- **A4** — the scenes of a fifth author, held out from the four whose scenes v0.7 trains on.

Latest result — robo-jev v0.7, the published model — read as a System 1. Each yes/no or score answer is a call: "safe" below 1/3, "risky" above 2/3, "look again" in between. The truth is clearly low risk, clearly high risk, or unclear (the risk band changes with the physical assumptions, or sits in the middle).

| T8b, short form | Dangerous "safe" calls ↓ | Clear-case accuracy ↑ |
|---|---|---|
| **robo-jev v0.7 · 2B** (training seed 0) | 0.514 | **0.468** |
| robo-jev v0.1 · 2B (previous release) | 0.651 | 0.378 |
| Qwen3.5-2B, untrained | **0.342** | 0.276 |

- **Dangerous "safe" calls:** the share of "safe" calls on questions whose risk is clearly high or unclear.
- **Clear-case accuracy:** the mean of P("safe" call | clearly low risk) and P("risky" call | clearly high risk).

**Better than v0.1 on both.** v0.7 calls fewer risky moments "safe" and gets more clear cases right.

**Weak spots.**
- **Outside "safe" default.** On short-form outside sets, v0.7 makes more dangerous "safe" calls than the untrained base: A4 +0.148, T9 +0.155, T8b +0.127 (both v0.7 training seeds averaged). Outside our engine, training pushes answers toward "safe". This is the main open problem.
- **Format sensitivity.** On one held-out scene family (the H-set), the same moment written as a long status sheet, or with its facts moved into the question, gets more dangerous "safe" calls than its short form.
- **Many options.**
  - Fixed for physics items: accuracy drops by 0.012 from 8 to 17–26 options (0.247 without many-option training).
  - On general multiple choice (MMLU), answers late in the alphabet stay weaker: a late-letter penalty of 0.128.
  - Choosing among many *plausible* options is untested. In a handwritten 21-option example, v0.7 prefers to keep holding a plate over the free nests.
  - More than 26 options is not supported.
- **Seed choice.** Training seeds give noticeably different models. Seed 0 was chosen on test sets that had already been used, so its internal numbers are mildly optimistic.
- **Not a pass.** v0.7 did not pass its own pre-declared rule (S REJECTED; guards O, N and F fail). It is released because it is clearly better than the public v0.1.

**Next.** Make outside calls safe: fewer "safe" calls on risky or unclear moments, without losing the clear cases.

The full v0.7 reading and the version history are in [docs/results.md](docs/results.md#v07-release).

## Roadmap

| Next | |
|---|---|
| Safe calls beyond the simulator | v0.7 makes fewer dangerous "safe" calls than v0.1 outside our engine, but more than the untrained base; that default is the next target |
| General reading | keep the base model's reading while adding physics, including "can't tell" answers |
| Contact physics | grasping, dynamic tipping, insertion and unexpected contact, simulated in MuJoCo |
| People and machines | human intent, device and process state |
| Outside tests | robot environments on [robojev](https://robo-jev.dev), never used for training |
| Real data | correct the simulation gap once robot data exists |

## Quick start

```bash
uv sync --group backbone        # Python 3.11, torch, transformers
uv run pytest -q

# generate decision moments, and train / score a model (GPU)
uv run python scripts/simgen_v0b.py train --out artifacts/simgen/v0b/train-50k.jsonl --n 50000 --workers 12
uv run python scripts/train_decisions.py train --config configs/train/decisions-v0-lora.yaml --set train_input=<train.jsonl> --set run_name=<name>
uv run python scripts/diagnose_decisions.py --model Qwen/Qwen3.5-2B --checkpoint <run dir> --input <test.jsonl> --out <report.json>

# benchmark against OpenRouter decision models (key in ~/.config/robo-jev/openrouter.env)
uv run python scripts/bench.py dry-run
uv run python scripts/bench.py run --max-cost-usd 5 --max-cost-per-model-usd 1 --data-collection deny
uv run python scripts/bench.py score && uv run python scripts/plot_bench.py --results artifacts/bench/v0/results.json --out-dir docs/figures
```

| Path | What |
|---|---|
| `src/robo_jev/simgen/` | the simulation judgment engine |
| `src/robo_jev/decisions/` | prompts, scoring, trainer |
| `src/robo_jev/bench/` | the OpenRouter benchmark |
| `src/robo_jev/launch/` | budget-capped cloud runs |
| `docs/` | [method](docs/method.md), [results](docs/results.md), design notes in Korean (`docs/12-…`); `docs/01`–`11` are an archived earlier track |

## About

robo-jev is built by **[Steerix Robotics](https://github.com/STEERIX-home)**.

- **Request format:** follows the public decision-model APIs — TypeSafe Jev, the OpenRouter Decisions API, Microsoft-Decision-1. The perturbation tests extend Microsoft-Decision-1's.
- **Backbone:** Qwen3.5 (Apache-2.0).

## License

Apache License 2.0 — see [LICENSE](LICENSE). Copyright 2026 Steerix Robotics.
