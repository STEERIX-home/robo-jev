# robo-jev — method

How robo-jev is built: where it sits in a robot, why physics needs its own data, the simulation judgment engine, held-out evaluation and training. Results are in [results.md](results.md); the full design notes (Korean) are in [12-sim-judgment-data-engine.md](12-sim-judgment-data-engine.md).

## Where it sits: three layers

| Layer | Job | Speed | In robo-jev |
|---|---|---|---|
| **L3** plan | What to do and in what order | slow; an LLM can reason here | given |
| **L2** judgment | While moving: *what now? is it safe? will it spill? should I look again?* — typed questions answered with probabilities | about once a second or faster | **this model** |
| **L1** execution | Turn each answer into motion with the robot's own controllers | tens to hundreds of Hz | given |

The request format is the one decision models already use: a **state** (text, what the senses report, at most 16,000 characters) and **1–12 questions**, each `choice` (options with descriptions), `noul` (yes/no) or `score` (0–1). The answer is a probability per option, a probability of yes, or a value.

## Why physics needs its own data

We scored untrained Qwen3.5 2B and 4B on generated physical decision requests with the same reading a deployed decision model would use. This is a **preview** on 300 generated records (not yet the sealed test sets); intervals are paired 95 % record-bootstrap intervals.

| | 2B | 4B |
|---|---|---|
| Yes/no Brier minus "always answer 0.5" (one option order, as deployed) | +0.023 [+0.015, +0.032] — worse | +0.061 [+0.047, +0.075] — worse |
| Same, averaged over option orders | +0.006 [−0.002, +0.015] — no difference found | +0.031 [+0.017, +0.045] — worse |
| Action accuracy minus random pick (0.23) | +0.085 [+0.048, +0.124] | +0.153 [+0.106, +0.199] |
| p(yes) shift when "Yes" is listed first | +0.25 | +0.32 |
| Verdict changes between two listings of the same question | 56 % | 31 % |
| Wrong planner recommendations followed | 61 % | 73 % |

Without a planner recommendation in the text, the 2B's action choice was about as good as a random pick in the first preview round (0.26 vs 0.24). The only scene type where either model clearly beats answering 0.5 is people crossing the robot's path. In short: the base models lean on option position and on the planner's advice, not on the physics.

## How the data is made: the simulation judgment engine

The full design is in [`docs/12-sim-judgment-data-engine.md`](docs/12-sim-judgment-data-engine.md) (Korean). In brief:

**1. Decision moments, not episodes.** Most L2 judgments are about the next 1–5 seconds. In our earlier tabletop episodes only 4 % of ticks needed a decision; the rest repeated the same answer. So the engine builds one decision moment at a time and rolls each option forward for a few seconds.

**2. Phenomena, not situations.** Situations are endless; the physics that decides robot judgments is a short list. Each phenomenon is a parametric scene that can be computed in closed form or (later) simulated in MuJoCo.

| | Phenomenon | Decided by | Version |
|---|---|---|---|
| P1 | Timing and collision | distance, speed, reaction time, stopping distance, the other agent's intent | v0 |
| P2 | Grasp and slip | grip force, friction, mass, centre-of-mass offset | v1 (MuJoCo) |
| P3 | Tipping and sliding on a moving support | centre-of-mass height, base width, acceleration, friction | v0 |
| P4 | Seating and alignment checks | measured height/tilt/force against tolerances | v0 |
| P5 | Spilling contents | fill level, container size, acceleration, tilt | v0 |
| P6 | Stale sightings, blind spots, sensor conflict | age of the last sighting, occlusion, sensor reliability | v0 |
| P7 | Body limits | top speed, deceleration, payload, reach | v0 |
| P8 | Human motion and intent | gestures, gaze, speech, approach | v0.5 |
| P9 | Device and process state | commands, replies, timing, jams | v0.5 |
| P10 | Unexpected contact | measured vs expected force | v1 (MuJoCo) |

**3. Meaning from the LLM side, physics from the engine.** A *scenario skin* (a café robot carrying coffee, a forklift at a blind corner, a cobot arm and a technician's hand, …) is mapped to physical distributions through a *property catalog* (a mug of hot coffee → size, mass, fill range; a running child → speed and a wide spread of intents). Ten domains are covered evenly — not only industrial sites.

**4. Labels are posterior probabilities.** The robot cannot see friction, the true fill level or a person's intent. So the engine draws K worlds consistent with what the robot observed, rolls every option forward in each, and labels:

- yes/no → how often the event happens across those worlds;
- score → the expected value;
- **action choice → a softmax over each option's expected cost**, so the most probable option is always the one with the lowest expected cost. (Not "how often each option was best": that rewards options that save a little time in most worlds but cause a serious incident in a few.)
- state choice → the posterior probability of each state.

A label of 0/1 from the single true world would teach the model to guess what the observation cannot reveal. Posterior labels teach the best probability that can be given from what is visible — that is where calibration comes from.

**5. No leakage, by construction.** The text writer only ever receives the observation, never the world; tests check that two different worlds with the same observation produce byte-identical text. (The earlier track lost weeks to goal fields leaking into the model input.) Some requests carry a planner recommendation, and some of those recommendations are deliberately wrong.

**6. Diversity aimed at the decision boundary.** Composition of phenomena × domains × bodies × sensor suites × question families × text styles (English-first, about 15 % Korean); sampling concentrated near the margin where the outcome flips; minimal pairs that move one fact across the boundary (and pairs that change only the wording, so the model reads numbers, not words); oversampled rare hazards.

## Held-out evaluation

Test sets are generated separately and sealed (sha256 manifest). The training generator is **unable** to produce any held-out cell; a scan of the full training set checks for zero held-out cells and zero request overlaps.

| Set | Measures |
|---|---|
| T1 | same distribution as training |
| T2 | unseen text style, unseen settings and paraphrases — generalization on the meaning side |
| T3 | unseen variable ranges (faster, taller, older sightings) — physical extrapolation |
| T4 | an unseen combination of phenomena (people stepping in while the robot carries tall loads) — compositional generalization |
| T5 | minimal pairs — does the judgment move in the right direction? |
| T6 | perturbations: Microsoft-Decision-1's eight (paraphrase, option order, keys, formatting, …) plus physical ones (units, number formats, left/right mirror, fact order) — invariance |
| T7 | calibration (probabilities vs posterior frequencies) |

**Central hypothesis:** what is learned from phenomena transfers to unseen situations. It is supported if the trained model beats the untrained one on **T2 and T4** (paired intervals excluding zero). If it improves only on T1, it has learned the generator's style, and the data is fixed before anything is scaled.

## Training

- **Same reading for diagnosis and training.** One prompt per question (Qwen chat template, one-letter answer codes, ending in `Answer:`); training prompts are byte-identical to diagnosis prompts. Option order, Yes/No order and score direction are re-shuffled for every training example, so position bias cannot be learned.
- **Loss:** soft-label KL between the label distribution and the model's probabilities over that question's answer codes, at the answer position only; a small auxiliary term keeps probability mass on the shown codes.
- **Modes:** LoRA (bf16 adapter compute with fp32 master weights) or full fine-tuning with fp32 master weights.
- **Smoke test (in-distribution only, not the hypothesis test):** after a short LoRA run on 1,600 generated records, on 400 held-back records of the same distribution the 2B went from worse than "always 0.5" on yes/no to clearly better (−0.087 [−0.113, −0.061] vs the 0.5 baseline), action accuracy 0.31 → 0.54, state accuracy 0.46 → 0.85, the "Yes first" shift +0.25 → −0.01, and listing flips 56 % → 13 %.
- **Compute:** development on a DGX Spark (GB10); training and evaluation run on RunPod H100 through a budget-capped launcher that deletes its pod when done and lets the pod delete itself if the launcher dies (LoRA bf16 on one H100: 11,215 tokens/s, about 6× the GB10; one epoch of 50k requests in 62 minutes).
