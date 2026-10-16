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

**A fast, single-pass 2B System 1 for robots' physical judgment.**

<p align="center">
  <img class="block dark:hidden" src="figures/bench-v0-physics-light.png" alt="Physical judgment on robo-jev's benchmark of held-out situations: robo-jev v0.7 (2B) 0.81, TypeSafe Jev 0.28, Liquid d1 0.17, Microsoft-Decision-1 0.04, Together Tev1 4B 0.03, OpenAI GPT-6 Luna Decisions −0.02, Upstage Solar Decide −0.15, untrained Qwen3.5-2B −0.15" width="860">
  <img class="hidden dark:block" src="figures/bench-v0-physics-dark.png" alt="Physical judgment on robo-jev's benchmark of held-out situations: robo-jev v0.7 (2B) 0.81, against 0.28 for TypeSafe Jev and near zero or below for the other decision models" width="860">
</p>

robo-jev reads a short text of what a robot senses and answers the questions that matter in the moment — *will the glass tip if I brake now? does the person reach my path first? did the part seat?* — with a probability for every answer.

{{VARIANT_NOTE}}

## Benchmark

| Model | Physical judgment ↑ | General reading (BoolQ + MultiNLI) ↑ |
|---|---|---|
| **robo-jev v0.7 · 2B** | **0.81** | 0.71 |
| TypeSafe Jev | 0.28 | 0.91 |
| Liquid d1 | 0.17 | **0.93** |
| Microsoft-Decision-1 | 0.04 | 0.92 |
| Together Tev1 4B | 0.03 | 0.90 |
| OpenAI GPT-6 Luna Decisions | −0.02 ¹ | 0.86 |
| Upstage Solar Decide | −0.15 | 0.86 |
| Qwen3.5-2B, untrained | −0.15 | 0.74 |

Physical judgment is skill against always answering 50 % (0 = a coin flip, 1 = perfect) on robo-jev's physical-judgment benchmark: 1,100 held-out situations — wordings, value ranges and combinations of phenomena the model never saw in training. The decision models were asked through the [OpenRouter Decisions API](https://openrouter.ai/models?output_modalities=decisions). ¹ It declined about a fifth of the physical-judgment requests; its score covers the rest.

## Why it matters

- **Fast.** One forward pass per question — no sampling, no reasoning tokens. On one L4 GPU a short request answers in about 50 ms, a dozen questions in about 0.1 s.
- **Small.** 2B parameters: it runs beside a robot's planner on a single small GPU.
- **Drop-in.** The Decisions API's request shape — a state text and up to 12 questions — and a calibrated probability for every answer.

## Quick usage

`robo_jev_decide.py` in this repo builds robo-jev's prompt and reads its answer; it needs only `transformers`{{PIP_EXTRA_NOTE}}.

```bash
pip install torch "transformers>=5.17" {{PIP_EXTRA}}
hf download {{REPO}} robo_jev_decide.py --local-dir .
python robo_jev_decide.py --model {{REPO}} --request request.json      # prints the answers as JSON
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

answers = model.decide(state, questions)   # {"action": {"type": "choice", "choice": …, "probabilities": {…}}, …}
```

{{EXAMPLE_OUTPUT}}

**Limits:** `choice` questions with 2–466 options; 1–12 questions per request; state text ≤ 16,000 characters. Options past the 26th get two-letter codes (AA, AB, …) that were not part of training, and the model almost never picks them in the default order; `orders="balanced"` lets every option compete, at one forward pass per option. With the robo-jev package (up to 26 options): `RoboJev.from_pretrained("{{REPO}}").decide(state, questions)`.

*Research preview, tuned on simulated physical judgment; less reliable outside simulation, and not a safety system.*

## Links

- {{SIBLING}}
- The robojev bench: [robo-jev.dev](https://robo-jev.dev)
- Code: [github.com/STEERIX-home/robo-jev](https://github.com/STEERIX-home/robo-jev)
- Apache License 2.0 (the base, [Qwen/Qwen3.5-2B](https://huggingface.co/Qwen/Qwen3.5-2B), is Apache-2.0). Built by [Steerix Robotics](https://github.com/STEERIX-home).
