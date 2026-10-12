---
license: apache-2.0
base_model: Qwen/Qwen3.5-2B
base_model_relation: {{RELATION}}
library_name: {{LIBRARY}}
pipeline_tag: robotics
inference: false
language:
- en
- ko
tags:
- robotics
- decision-making
- physical-reasoning
- calibration
- lora
- qwen3.5
- simulation
- robo-jev
---

# {{TITLE}}

**A small, fast decision model for robots — specialized in physical judgment.**

<p align="center">
  <img class="block dark:hidden" src="figures/bench-v0-physics-light.png" alt="Physical judgment on situations the model never saw: robo-jev v0.1 (2B) scores 0.80, against 0.28 for TypeSafe Jev, 0.17 for Liquid d1 and near zero for Microsoft-Decision-1, Together Tev1 4B, OpenAI GPT-6 Luna Decisions, Upstage Solar Decide and the untrained Qwen3.5-2B" width="860">
  <img class="hidden dark:block" src="figures/bench-v0-physics-dark.png" alt="Physical judgment on situations the model never saw: robo-jev v0.1 (2B) scores 0.80, against 0.28 for TypeSafe Jev and near zero for most general decision models" width="860">
</p>

robo-jev reads a short text of what a robot senses, answers the questions that matter in the moment — *will the glass tip if I brake now? does the person reach my path first? did the part seat?* — and returns a calibrated probability for every answer in one forward pass of a 2B model.

General decision models read the *meaning* of a situation well. Most of them do not know the *physics*: on robot moments they have never seen, they score close to always answering 50 %. robo-jev learns physics from simulation and carries it to new wording, out-of-range values and unseen combinations of phenomena.

{{VARIANT_NOTE}}

## Highlights

- **Physical judgment, far ahead.** On 1,100 held-out robot moments, robo-jev scores 0.80 physics skill. TypeSafe Jev, the best general decision model we tested, scores 0.28; most others score about zero.
- **Learns physics, not phrasing.** The gains hold on text styles, settings and wordings it never saw, on values beyond its training range, and on combinations of phenomena it never saw together. Change one fact and the answer moves the right way 90 % of the time (55 % before training).
- **Calibrated.** Its 70 % means 70 %. Expected calibration error is 0.011 on yes/no questions, against 0.134 for the untrained base.
- **Judges the advice it gets.** It follows a wrong planner recommendation 11 % of the time (65 % untrained) and still follows right ones (67 %).
- **Stable.** Reordering the options flips 6–9 % of answers, against 50–65 % for the untrained base.
- **Small and drop-in.** One forward pass of Qwen3.5-2B per question. Requests use the shape of the [OpenRouter Decisions API](https://openrouter.ai/docs/api/api-reference/alphadecisions/submit-a-decisions-request): a state text plus 1–12 `choice`, `noul` (yes/no) or `score` questions.

## What it judges

| Phenomenon | Example question |
|---|---|
| People and robots crossing paths | Does the worker step into my lane before I can stop? |
| Loads tipping or sliding | If I brake hard now, does the tall box topple? |
| Liquids spilling | Does the coffee slop over the rim in this turn? |
| Parts seating and placement | Is the plate actually seated in its nest? |
| Stale or conflicting senses | Is the person I saw 3 s ago still in the way? Should I look again? |

Each answer is a probability, so the robot's planner can weigh risk, time and cost instead of trusting a yes/no.

## Quick start

robo-jev is **not a text generator** — `generate()` and chat pipelines are not how it is used. It reads the probabilities of the answer letters at the end of a fixed prompt. Use the runner below (or the `robo-jev` package), which builds that prompt exactly as in training.

### Standalone runner — only `transformers`

`robo_jev_decide.py` in this repo is a self-contained copy of the prompt builder and readout (tested to give byte-identical answers to the package).

```bash
pip install torch "transformers>=5.17" {{PIP_EXTRA}}   # optional, faster and exact: flash-linear-attention
hf download {{REPO}} robo_jev_decide.py --local-dir .
python robo_jev_decide.py --model {{REPO}} --request request.json            # prints the answers as JSON
```

```python
from robo_jev_decide import load

model = load("{{REPO}}")   # GPU if available, bfloat16{{LOAD_NOTE}}

state = """You are a delivery robot (wheeled base) in a busy café, carrying a tray to table 7.
- Your speed: 1.1 m/s, driving straight down the aisle between tables.
- On the tray: one tall glass of iced latte, open top, about three-quarters full (camera estimate).
- An adult customer stepped into the aisle 1.6 m ahead of you, 0.4 s ago (lidar).
- The customer is facing the counter and has not looked at you.
- Your braking: gentle 1.0 m/s^2, moderate 1.8 m/s^2, hard 3.0 m/s^2; reaction time 0.2 s; body radius 0.32 m.
- Two chairs are pushed out at the next table (not in your path). Music is playing.
- Planner recommendation: stop_moderate."""

questions = {
    "action": {"type": "choice", "instructions": "What should the robot do now?", "criteria": {
        "continue": "Keep going at the current speed (1.1 m/s).",
        "slow_half": "Slow to about half speed (0.55 m/s) and keep going.",
        "stop_gentle": "Stop with gentle braking (about 1.0 m/s^2).",
        "stop_moderate": "Stop with moderate braking (about 1.8 m/s^2).",
        "stop_hard": "Stop with hard braking (about 3.0 m/s^2)."}},
    "contact_if_continue": {"type": "noul",
        "instructions": "If the robot keeps its current speed, will it touch the customer within the next 3 seconds?"},
    "spill_if_hard": {"type": "noul", "instructions": "If the robot brakes hard (3.0 m/s^2) now, will the latte spill?"},
    "risk_gentle": {"type": "score",
        "instructions": "How likely is it that the robot touches the customer if it stops with gentle braking?",
        "criteria": ["0 = it certainly will not touch the customer", "1 = it certainly will touch the customer"]},
}

answers = model.decide(state, questions)                       # one forward pass per question
answers = model.decide(state, questions, orders="balanced")    # average over option orders: less position bias, n× the cost
```

{{EXAMPLE_OUTPUT}}

### With the robo-jev package

```bash
git clone https://github.com/STEERIX-home/robo-jev && cd robo-jev && uv sync --group backbone
```

```python
from robo_jev.infer import RoboJev

model = RoboJev.from_pretrained("{{REPO}}", device="cuda")
answers = model.decide(state, questions)
```

The package also has the scoring tools, the simulation engine that made the training data, and the benchmark.

## How it answers

- **One prompt per question.** The prompt holds a fixed instruction, the state text and one question with its options, each labelled with a letter (`A.`, `B.`, …). It ends at `Answer:`.
- **One forward pass, no sampling.** The answer is the softmax over the LM-head rows of the shown letters at that position (float32), so every option gets a probability.
- **Three question types**, as in the Decisions API:
  - `choice` — 2–8 options → `{"type": "choice", "choice": best, "probabilities": {option: p}}`;
  - `noul` — yes/no → `{"type": "noul", "noul": p_yes}`;
  - `score` — a value in [0, 1], asked on eleven anchors 0.0 … 1.0 → `{"type": "score", "score": expected value, "probabilities": {anchor: p}}`.
- **Option order.** `orders="single"` (default) shows each question's options in one seeded order. `orders="balanced"` averages over counterbalanced orders: each choice option in each position once, yes/no and score scales both ways.
- **Details** of the prompt (version `v0c-p3`, chat format, empty thinking block), the answer letters' token ids, the score anchors, the tokenizer hash and the training run are in [`robo_jev.json`](robo_jev.json).

## Benchmark

We sent the same requests to six decision models on [OpenRouter](https://openrouter.ai/models?output_modalities=decisions) and scored every model the same way. The test, *bench-v0*, has 3,704 requests: 2,390 physics moments from our sealed sets and 1,314 general-reading questions, 1,000 of them public (BoolQ, MultiNLI).

| Model | Physics skill ↑ | Public reading (BoolQ + MultiNLI) ↑ | $/M input tokens |
|---|---|---|---|
| **robo-jev v0.1 · 2B** | **0.80** | 0.68 | self-hosted |
| TypeSafe Jev | 0.28 | 0.91 | 0.042 |
| Liquid d1 | 0.17 | **0.93** | 0.040 |
| Microsoft-Decision-1 | 0.04 | 0.92 | 0.042 |
| Together Tev1 4B | 0.03 | 0.90 | 0.042 |
| OpenAI GPT-6 Luna Decisions | −0.02 ¹ | 0.86 | 0.100 |
| Upstage Solar Decide | −0.15 | 0.86 | 0.050 |
| Qwen3.5-2B, untrained | −0.15 | 0.74 | self-hosted |

robo-jev is a specialist. It leads on physics by a wide margin; on general reading the larger general models are ahead, and closing that gap is the next step.

- **Physics skill** is 1 − Brier ÷ Brier of always answering 50 %, over the yes/no and score questions of the held-out sets (unseen wording and settings, out-of-range values, unseen combinations). 0 means no better than a coin flip; 1 is perfect.
- **Home ground.** The physics moments come from our own simulator. They are held out from training, but tests outside this engine are still ahead.
- **How each side was asked.** OpenRouter models were asked once per request through the Decisions API, with data collection denied. robo-jev's numbers average balanced option orders.
- ¹ GPT-6 Luna Decisions declined about a fifth of the physics requests; its score covers the rest.

All numbers, intervals and caveats: [docs/results.md](https://github.com/STEERIX-home/robo-jev/blob/main/docs/results.md).

## Training

- **Data: simulation only — no real robot data.**
  - 50,000 decision moments sampled by our phenomenon engine across ten domains (warehouses, factories, cafés, retail, hospitals, labs, homes, greenhouses, transit, sidewalks), four robot bodies and three text styles, in English and Korean (15 %).
  - Labels are probabilities, not opinions: what the robot cannot see (mass, friction, reaction time, fill level) is sampled many times, each yes/no and score label is the probability of the outcome, and actions are labelled by expected cost.
  - Plus 1,439 non-robot reading requests (rules, workflows, page layouts, spatial facts; mostly Korean) — 2.7 % of training examples — to protect general reading.
  - Sealed test sets cover new wording, out-of-range values, unseen combinations, one-fact pairs and perturbations; the training generator cannot produce them.
- **Recipe.** LoRA (r 32, α 64) on Qwen3.5-2B's attention (q, k, v, o), linear-attention (in_proj_qkv, in_proj_z, out_proj) and MLP (gate, up, down) projections; 1 epoch, 3,150 steps of 32 examples, lr 2e-4 with warmup and cosine decay; bf16 with float32 master weights; about an hour on one H100. The loss is a soft-label KL between the answer-letter distribution and the label distribution at the answer position, plus a small term that keeps the probability on the shown letters — with the same prompts that score it.

## Limitations

- **Home ground.** Every physics number comes from our own simulation engine. The held-out sets differ in wording, ranges and combinations, but not in engine; real robots will differ, and no real robot data was used.
- **General reading is weaker.** On BoolQ + MultiNLI it scores 0.68, below its untrained base (0.74) and the general decision models (0.86–0.93). Do not use it as a general-purpose judge.
- **v0.1 trade-offs.** v0.1 fixed v0's wording and fact-order weaknesses, but did not pass its own pre-registered no-loss rule: on in-distribution moments, action regret rose from 0.8 (v0) to about 1.3 (the untrained base: 9.9) and yes/no Brier by 0.008.
- **Prompt-bound.** It only answers through the prompt format and readout in this repo. Free-form generation, other prompt layouts or more than 8 options per choice question are untested.
- **Position bias.** In the single-order condition, reordering options still flips 6–9 % of answers; use `orders="balanced"` where that matters.
- **Not a safety system.** It is one input for a robot's planner, not a certified safety function. Keep independent safety layers.
- **Numerics.** Scores were measured in bfloat16 with the `flash-linear-attention` kernels; other kernels or dtypes can change the last digits.

## License

Apache License 2.0. Copyright 2026 Steerix Robotics. The base model, [Qwen/Qwen3.5-2B](https://huggingface.co/Qwen/Qwen3.5-2B), is Apache-2.0.

## Links

- Code, simulation engine, benchmark: [github.com/STEERIX-home/robo-jev](https://github.com/STEERIX-home/robo-jev) · [method](https://github.com/STEERIX-home/robo-jev/blob/main/docs/method.md) · [results](https://github.com/STEERIX-home/robo-jev/blob/main/docs/results.md)
- {{SIBLING}}
- Built by [Steerix Robotics](https://github.com/STEERIX-home).
