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
  <img class="block dark:hidden" src="figures/bench-v0-physics-light.png" alt="Physical judgment on our simulation benchmark: robo-jev v0.7 (2B) scores 0.81 and the previous release v0.1 0.80, against 0.28 for TypeSafe Jev, 0.17 for Liquid d1 and near zero or below for Microsoft-Decision-1, Together Tev1 4B, OpenAI GPT-6 Luna Decisions, Upstage Solar Decide and the untrained Qwen3.5-2B" width="860">
  <img class="hidden dark:block" src="figures/bench-v0-physics-dark.png" alt="Physical judgment on our simulation benchmark: robo-jev v0.7 (2B) scores 0.81 and v0.1 0.80, against 0.28 for TypeSafe Jev and near zero for most general decision models" width="860">
</p>

robo-jev reads a short text of what a robot senses, answers the questions that matter in the moment — *will the glass tip if I brake now? does the person reach my path first? did the part seat?* — and returns a calibrated probability for every answer in one forward pass of a 2B model.

General decision models read the *meaning* of a situation well, but on our physics benchmark most of them score close to always answering 50 %. robo-jev learns physics from simulation and carries it to new wording, out-of-range values and unseen combinations *within our simulator*. Outside it, it is not there yet — see [Release status](#release-status).

{{VARIANT_NOTE}}

> **v0.7 did not pass its own pre-declared rule (S REJECTED; guards O, N and F fail). It is released because it is clearly better than the public v0.1.** Its biggest weakness: on moments written outside our engine, it calls more risky situations "safe" than the untrained base does. Read [Release status](#release-status) and [Limitations](#limitations) before you rely on it.

## What's new since v0.1

Same base model, same LoRA recipe, same prompt. The training data changed:

- **Physics labelled in MuJoCo, across scenes.** Three physical laws — tip or slide under acceleration or tilt, centre of mass past an edge, friction-limited holding — in twelve scenes, with pairs that change one physical factor.
- **Four independent authors.** Those scenes were written by four separate authors, each with its own domain and text style. A fifth author's scenes are held out as a test (A4). The authors are AI agents with separate briefs; none saw another's code or text.
- **Deployment-shaped requests.**
  - About a quarter of the physics moments come wrapped in long status sheets (1,500–8,000 characters) with neutral, reassuring or alarming boilerplate and a planner note that is right only half the time.
  - About 8 % of physics choice questions, and of the general multiple-choice items, are expanded to 9–26 options.
  - About 5 % move the situation facts into the questions.
- **Up to 26 options.** Both runners now take choice questions with 2–26 options (v0.1: 2–8), one answer letter (A–Z) each.

## Highlights

- **Physical judgment on our benchmark, far ahead.** On 1,100 held-out moments from our simulator, v0.7 scores 0.81 physics skill (v0.1: 0.80). TypeSafe Jev, the best general decision model we tested, scores 0.28; most others score about zero.
- **Learns physics, not phrasing — inside the simulator.** The gains hold on text styles, settings and wordings it never saw, on values beyond its training range, and on combinations of phenomena it never saw together. Change one fact and the answer moves the right way 92 % of the time (v0.1: 90 %; untrained: 55 %).
- **Many options.** On held-out physics questions, accuracy drops by only 0.012 from 8 options to 17–26 (two-seed mean).
- **Calibrated.** Its 70 % means 70 %. Expected calibration error is 0.011 on yes/no questions, against 0.134 for the untrained base.
- **Judges the advice it gets.** It follows a wrong planner recommendation 9 % of the time (65 % untrained) and still follows right ones (65 %).
- **Stable.** Reordering the options flips 5–10 % of answers, against 50–65 % for the untrained base.
- **Small and drop-in.** One forward pass of Qwen3.5-2B per question. Requests use the shape of the [OpenRouter Decisions API](https://openrouter.ai/docs/api/api-reference/alphadecisions/submit-a-decisions-request): a state text plus 1–12 `choice`, `noul` (yes/no) or `score` questions.

## Release status

**The pre-declared rule.** Before any v0.7 data existed, we wrote down how v0.7 would be judged: did reshaping the training requests (long status sheets, many options, facts in the questions) make the model robust to request shape, without losing ground elsewhere? v0.7 did not pass its own pre-declared rule (S REJECTED; guards O, N and F fail). It is released because it is clearly better than the public v0.1. The full reading is in [docs/results.md](https://github.com/STEERIX-home/robo-jev/blob/main/docs/results.md#v07-release).

**Outside our simulator.** T8b has 480 robot moments written independently of our engine, labelled by MuJoCo rollouts. Each yes/no or score answer is read as a call — below 1/3 is "safe", above 2/3 is "risky", in between is "look again" — and compared with the truth: clearly low risk, clearly high risk, or unclear (the risk band changes with the physical assumptions, or sits in the middle).

| T8b, short form, same truth classes | Dangerous "safe" calls ↓ | Clear-case accuracy ↑ |
|---|---|---|
| **robo-jev v0.7** (this release, training seed 0) | 0.514 | **0.468** |
| robo-jev v0.1 (previous release) | 0.651 | 0.378 |
| Qwen3.5-2B, untrained | **0.342** | 0.276 |

- **Dangerous "safe" calls:** the share of "safe" calls on questions whose risk is clearly high or unclear.
- **Clear-case accuracy:** the mean of P("safe" call | clearly low risk) and P("risky" call | clearly high risk).
- v0.7 beats v0.1 on both. The untrained base still makes the fewest dangerous "safe" calls: outside our engine, training pushes answers toward "safe".

**Many options** (sealed test: the same item asked with 8, 21 and 26 options, the right answer spread over A–Z):

| | Accuracy drop, 8 → 17–26 options ↓ | Late-letter penalty ↓ |
|---|---|---|
| Physics items from held-out scenes and authors | 0.012 (0.247 without many-option training) | 0.029 (0.433 without) |
| General multiple choice (MMLU) | 0.070 (0.060 without; untrained base 0.084) | 0.128 (0.113 without; untrained base 0.270) |

The late-letter penalty is the accuracy when the right answer is A–H minus when it is Q–Z, at 17 or more options. "Without many-option training" is v0.7's predecessor, trained on the same data without the request-shape changes. Every number averages two training seeds per version. Physics is fixed; general multiple choice is not.

## Quick start

robo-jev is **not a text generator** — `generate()` and chat pipelines are not how it is used. It reads the probabilities of the answer letters at the end of a fixed prompt. Use the runner below (or the `robo-jev` package), which builds that prompt exactly as in training.

**Limits** (both runners; a request outside them is refused whole):
- state text ≤ 16,000 characters;
- 1–12 questions per request;
- 2–26 options per `choice` question.

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

### Many options

A choice question may list up to 26 options. This lab request asks where to set down an open, full 96-well plate, with 21 candidate places:

<details>
<summary>The 21-option request</summary>

```python
state = """You are a lab robot arm at a liquid-handling bench, holding a 96-well plate in your gripper 15 cm above the deck.
- The plate: standard footprint 128 x 86 mm, no lid, every well about 80 % full of water-based buffer (camera estimate).
- Your gripper holds the plate level; you move it at 0.25 m/s between deck positions.
- The scheduler asks you to set this plate down within the next 10 s so the next run can start.
- The deck positions below are as the deck camera saw them 1 s ago.
- A technician stands at the far end of the bench, 1.2 m away, not reaching toward the deck."""

questions = {
    "move": {"type": "choice", "instructions": "Where should the robot set the plate down now?", "criteria": {
        "nest_a1": "Deck nest A1: empty, level, plate nest with all four corner guides.",
        "nest_a2": "Deck nest A2: another plate is already in it.",
        "nest_a3": "Deck nest A3: empty and level, but one corner guide is broken off.",
        "nest_b1": "Deck nest B1: empty and level; a used pipette tip lies inside the nest.",
        "magnet": "Magnetic bead stand: empty, level, plate nest with guides.",
        "sealer": "Plate sealer input tray: empty, level, ready.",
        "reader_tray": "Plate reader tray: extended, empty, level.",
        "shaker_on": "Shaker platform: running at 600 rpm right now.",
        "shaker_off": "Shaker platform: stopped, clamps open.",
        "heater_hot": "Heating block: at 95 °C.",
        "cooler": "Cooling block at 4 °C; the next run needs the plate at room temperature.",
        "stack_top": "On top of a stack of four lidded plates, no nest.",
        "ledge": "Shelf ledge 60 mm deep (narrower than the plate).",
        "tilted_rack": "Rack slot tilted about 15 degrees (worn rails).",
        "centrifuge": "Centrifuge bucket: still spinning down, lid open.",
        "edge_bench": "Bare bench top, 3 cm from the front edge, no nest.",
        "tech_hand": "Hand the plate to the technician at the far end of the bench.",
        "floor": "Set it on the floor next to the bench.",
        "trash": "Waste bin.",
        "hold": "Keep holding it level and wait 10 s.",
        "nest_b2": "Deck nest B2: empty, level, plate nest with all four corner guides; next to the shaker."}},
    "spill_on_shaker": {"type": "noul", "instructions": "If the robot sets the plate on the running shaker (600 rpm) now, "
                                                        "will buffer slop out of the wells within 5 seconds?"},
}
answers = model.decide(state, questions)
```

</details>

{{MANY_OPTIONS_OUTPUT}}

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

- **One prompt per question.** The prompt holds a fixed instruction, the state text and one question with its options, each labelled with a letter (`A.`, `B.`, … up to `Z.`). It ends at `Answer:`.
- **One forward pass, no sampling.** The answer is the softmax over the LM-head rows of the shown letters at that position (float32), so every option gets a probability.
- **Three question types**, as in the Decisions API:
  - `choice` — 2–26 options → `{"type": "choice", "choice": best, "probabilities": {option: p}}`;
  - `noul` — yes/no → `{"type": "noul", "noul": p_yes}`;
  - `score` — a value in [0, 1], asked on eleven anchors 0.0 … 1.0 → `{"type": "score", "score": expected value, "probabilities": {anchor: p}}`.
- **Option order.** `orders="single"` (default) shows each question's options in one seeded order. `orders="balanced"` averages over counterbalanced orders: each choice option in each position once, yes/no and score scales both ways.
- **Details** of the prompt (version `v0c-p3`, chat format, empty thinking block), the token ids of all 26 answer letters, the score anchors, the limits, the tokenizer hash and the training run are in [`robo_jev.json`](robo_jev.json).

## Benchmark

We sent the same requests to six decision models on [OpenRouter](https://openrouter.ai/models?output_modalities=decisions) and scored every model the same way. The test, *bench-v0*, has 3,704 requests: 2,390 physics moments from our sealed sets and 1,314 general-reading questions, 1,000 of them public (BoolQ, MultiNLI).

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

robo-jev is a specialist. It leads on physics by a wide margin; on general reading the larger general models are ahead, and robo-jev is still below its own untrained base.

- **Physics skill** is 1 − Brier ÷ Brier of always answering 50 %, over the yes/no and score questions of the held-out sets (unseen wording and settings, out-of-range values, unseen combinations). 0 means no better than a coin flip; 1 is perfect.
- **Home ground.** The physics moments come from our own simulator. They are held out from training, but they share its phenomena and text skins. For tests outside this engine, see [Release status](#release-status).
- **How each side was asked.** OpenRouter models were asked once per request through the Decisions API, on 2026-10-12, with data collection denied. robo-jev's numbers average balanced option orders.
- ¹ GPT-6 Luna Decisions declined about a fifth of the physics requests; its score covers the rest.

All numbers, intervals and caveats: [docs/results.md](https://github.com/STEERIX-home/robo-jev/blob/main/docs/results.md).

## Training

- **Data: simulation only — no real robot data.** 116,890 training records:
  - **50,000 engine moments.** Our phenomenon engine samples decision moments across ten domains (warehouses, factories, cafés, retail, hospitals, labs, homes, greenhouses, transit, sidewalks), four robot bodies and three text styles, in English and Korean (15 %). The moments also carry derived System-1 questions: risk bands and comparisons.
  - **18,680 MuJoCo records** from the four authors (14,998 moments, about a quarter of them used twice), labelled by rollouts over hidden physical factors.
  - **46,771 general multiple-choice records** from licensed public training sets — CommonsenseQA, HellaSwag, Social IQa, QASC, CosmosQA (35,000 items, about a third used twice) — and 1,439 non-robot reading requests (rules, workflows, page layouts, spatial facts; mostly Korean), to protect general reading.
  - Labels are probabilities, not opinions: what the robot cannot see (mass, friction, reaction time, fill level) is sampled many times, each yes/no and score label is the probability of the outcome, and actions are labelled by expected cost.
  - Sealed test sets cover new wording, out-of-range values, unseen combinations, one-fact pairs, perturbations, held-out scenes and independent authors; none of them is in the training data.
- **Recipe.** LoRA (r 32, α 64) on Qwen3.5-2B's attention (q, k, v, o), linear-attention (in_proj_qkv, in_proj_z, out_proj) and MLP (gate, up, down) projections; 1 epoch, 6,306 steps of 32 examples, lr 2e-4 with warmup and cosine decay; bf16 with float32 master weights; about 3¼ hours on one H100. The loss is a soft-label KL between the answer-letter distribution and the label distribution at the answer position, plus a small term that keeps the probability on the shown letters — with the same prompts that score it.

## Limitations

- **Outside "safe" default.** On short-form outside sets — moments written independently of our engine — the trained model makes more dangerous "safe" calls than the untrained base: A4 +0.148, T9 +0.155, T8b +0.127 (both v0.7 training seeds averaged; A4 is the held-out author, T9 a separately built outside test). It is better than v0.1 here, but not yet safe.
- **Format sensitivity.** On one held-out scene family (the H-set), the same moment written as a long status sheet, or with its facts moved into the question, gets more dangerous "safe" calls than its short form.
- **Late-letter penalty.** On general multiple choice with many options, answers late in the alphabet are weaker: a late-letter penalty of 0.128.
- **More than 26 options is not supported.** The runners refuse a choice question with 27 or more options.
- **Seed choice.** Training seeds give noticeably different models. Seed 0 was chosen on test sets that had already been used, so its internal numbers are mildly optimistic.
- **Home ground.** Every bench-v0 physics number comes from our own simulation engine. Real robots will differ, and no real robot data was used.
- **General reading is weaker.** On BoolQ + MultiNLI it scores 0.71, below its untrained base (0.74) and the general decision models (0.86–0.93). Do not use it as a general-purpose judge.
- **Prompt-bound.** It only answers through the prompt format and readout in this repo. Free-form generation and other prompt layouts are untested.
- **Position bias.** In the single-order condition, reordering options still flips 5–10 % of answers; use `orders="balanced"` where that matters.
- **Not a safety system.** It is one input for a robot's planner, not a certified safety function. Keep independent safety layers.
- **Numerics.** Scores were measured in bfloat16 with the `flash-linear-attention` kernels; other kernels or dtypes can change the last digits.

## License

Apache License 2.0. Copyright 2026 Steerix Robotics. The base model, [Qwen/Qwen3.5-2B](https://huggingface.co/Qwen/Qwen3.5-2B), is Apache-2.0.

## Links

- Code, simulation engine, benchmark: [github.com/STEERIX-home/robo-jev](https://github.com/STEERIX-home/robo-jev) · [method](https://github.com/STEERIX-home/robo-jev/blob/main/docs/method.md) · [results](https://github.com/STEERIX-home/robo-jev/blob/main/docs/results.md)
- {{SIBLING}}
- Built by [Steerix Robotics](https://github.com/STEERIX-home).
