# robojev — a real-time System One for decisions in the physical world

robojev is a research project by Steerix Robotics: a **learned judgment model that decides, many times per second, what a robot should do next** — which action to commit to, whether to stop, whether to grip, which path and speed — from a structured description of the world, without generating text.

It is a *System One* model in the sense that TypeSafe introduced with Jev: the input is a typed state plus a set of questions with candidate answers, and the output is one probability distribution per question, read out in a single forward pass. What is new here is the setting. The state is a physical scene that changes every 100 ms, the questions repeat on every tick, and the answers drive a real controller.

> Status (2026-09-19): the whole pipeline that can be verified without a GPU is implemented, reviewed and tested — contracts, data generators, simulator and controller, streaming harness, scripted expert, rollout labels, serialization, a tiny reference model, and a training loop with exact resume. **No real backbone has been trained yet, and no latency claim has been measured on target hardware.** See [Status](#status) for what is and is not verified.

## Why

Robot task intelligence today lives in large vision-language models that plan in seconds. Execution lives in controllers that run at hundreds of hertz but understand nothing. Between them is a gap: the moment when a grasp fails, an instruction changes, an object moves, or a person steps in — decisions that need *meaning* (what the instruction wants, what is forbidden, what has already been tried) at the speed of a reflex, not of a paragraph.

robojev is that layer: **semantic System One at 10 Hz.** It does not reason in chains of thought and it does not compute torques. It reads the situation and returns typed judgments that a deterministic executor can act on immediately.

```text
L3  task intelligence  (VLM / planner, seconds)     goals, constraints, plans
        ↓
L2  robojev            (10 Hz stream, typed)        action commitment · stop · gripper · path · speed · force
        ↓
L1  executor           (50 Hz, scripted / RL)       blending, leases, reflexes, gripper events
        ↓
L0  physics / motors   (500 Hz)
```

## Core ideas

- **The harness asks, the model judges.** A harness turns the world into a structured state (objects with tracked ids, poses, precision, visibility; zones; robot state; executed history) and enumerates *candidate* joint actions (`grasp o3 → zone L`, `push o5 −x`, …) with geometry it computed itself. The model never generates candidates or text; it scores them. Anything a classical computation can answer exactly (reachability, clearance, distances) is given to the model, not asked of it.
- **Ten typed questions per tick, answered in parallel.** The main decision is a choice over joint-action candidates (K ≤ 32); four gating booleans (done, instruction sufficient, observe more, retry ok); stop; desired gripper state; path (direct / via waypoint / retreat / hold); speed and force as ordinals. Auxiliary answers are conditioned on the commitment fixed at the start of the tick, so the model is never asked "which speed" for an action it might switch away from in the same tick.
- **Pointer readout, not classification.** Every decision position reads all candidates through a shared bilinear readout `z_ik = (U h_d)ᵀ (V h_c) / √r + b`, so candidates can change meaning and count on every request and the same weights serve robot ticks and general single-request judgments alike. Candidate-branch readout (one path per candidate) is kept only as a comparison group.
- **Streaming state.** A robot episode is one append-only token stream: a static prefix, then per tick `[state][executed history][dynamic candidates]`, from whose end the ten decision positions branch as *transient* one-token forks. The next tick continues from the pre-branch state. Full-attention layers see the static prefix plus a 30-tick window; linear-attention (DeltaNet) layers carry the whole episode. Training uses truncated BPTT over 10-second chunks so the model trains on exactly the state it will infer with.
- **Composition rules and a real executor contract.** Validity checks, stop-first ordering, gating, hysteresis on the main decision (switch only when the same challenger beats the incumbent by δ for m ticks), auxiliary-answer application, and command generation are deterministic and versioned. The controller enforces observation deadlines, geometry-age tolerances, issue leases, blending, stop transitions, reflexes and idempotent gripper events, and every acknowledgement is recorded exactly as executed.
- **Information boundary.** Nothing the front end could not know reaches the model: occluded objects keep their last observed pose, simulator events never leak, labels and evidence live outside the input area and a validator rejects any request that carries them. The executed history is never replaced by a label, in expert episodes or in DAgger relabeling.
- **Labels from evidence, not decree.** Main-decision labels come from semantic admissibility (instruction, destination, forbidden contacts) → measured outcomes of counterfactual rollouts at keyframes (paired seeds, censoring preserved) → a commitment rule. Candidates that were not rolled out are `unknown` and trained with a partial-label loss. A rule-based judge and a tiny from-scratch scorer serve as baselines so that no result is credited to "understanding" that pattern matching would also achieve.

## What is built

| Area | Modules | Verified by |
| --- | --- | --- |
| Contracts | `contracts.py` — request/label schema, validation, `model_input` boundary | contract and boundary tests; D0 fixtures (64 single requests, 4 synthetic streams) |
| Non-robot data | `data/generate.py`, `domains.py`, `split.py`, `validate.py` — four domains, origin-group splits, automatic QA | 2,000 regenerated records, zero QA violations |
| Simulator and executor | `sim/environment.py` (MuJoCo / robosuite, E0/E1 scenes, disturbances, instruction changes, snapshot/restore), `sim/controller.py` (executor contract) | E0 closed loop completes with the rule baseline alone |
| Streaming harness | `harness/robot.py` (candidates, pruning, waypoint planner, composition v0, commands), `harness/rule_judge.py`, `perception/pointworld.py` | regression tests for every defect found by external review; E0 seeds complete |
| Expert and data pipeline | `sim/expert.py`, `data/robot_episodes.py`, `sim/label.py`, `data/rollouts.py`, `data/dagger.py`, `configs/sim/events.yaml` | E0 and E1 episodes complete end to end; first 100 rollouts costed (0.345 s each → 128k ≈ 12 CPU-hours) |
| Model (CPU reference) | `model/serialize.py` (real tokenizer, two layouts), `model/attention.py`, `model/hybrid.py` (differentiable DeltaNet/conv state fork, tiny hybrid), `model/stream.py`, `model/judge.py`, `loss.py` | incremental == from-scratch computation in FP32 and float64; branch isolation bitwise; the plan's mandated tests verbatim |
| Training (CPU reference) | `train.py`, `sampler.py`, `checkpoint.py` — TBPTT, mixed sampler, atomic checkpoints, identity-checked resume | 20 continuous steps == 10 + process restart + 10, bit-identical |

`uv run pytest -q` → 898 passed, 1 expected failure (an E1 seed the current candidate cap cannot solve; see below).

## Status

What is verified and what is not, per area. Nothing below the line is claimed.

| | Verified | Not yet |
| --- | --- | --- |
| Robot closed loop | E0 (static scenes) with rule judge and scripted expert; E1 (instruction change + disturbance) on 3 seeds | E1 at scale; real-robot front end (3D reconstruction) |
| Data | first batch of 40 episodes; rollout machinery and costing | D1 (400 episodes, 128k rollouts) — withheld until the candidate-space decision below; human review of D0 |
| Model | contracts, masks, state forking, readout and losses on a random-weight fixture | any pretrained backbone; hybrid kernels; BF16 |
| Training | step, TBPTT, sampler, checkpoint, resume on the fixture | GPU training; real-backbone results; calibration |
| Latency | token counts with the real tokenizer | the 10 Hz / 100 ms gate, which must be measured on deployment-class hardware (DGX Spark / Jetson) |

Two measurements changed the plan and are still open decisions:

1. **Token budget.** With the current serialization a tick costs 1,764–3,712 tokens (estimate was 450–700), so a 10-second training chunk is 176K–371K tokens. A compact format with delta ticks (contract v0.3) is a prerequisite for both the latency budget and cloud training.
2. **Candidate cap.** With two speed profiles and four push directions per object, almost every scene hits K = 32, and in 44% of E1 ticks the instructed target×zone action is pruned away, leaving a degenerate `hold` label (down-weighted for now). Removing the profile from the joint key, restricting push directions, and a goal-relevance pre-filter are the proposed fix.

## Repository

```text
docs/        research design (01–06), external reviews (07, 09, 10, 11), and the canonical stream contract (08)
src/robo_jev/
  contracts.py  data/  sim/  harness/  perception/      # world, harness, expert, data pipeline
  model/  loss.py  sampler.py  train.py  checkpoint.py   # model contracts and CPU training path
configs/     harness, controller, simulator, expert, events, data, model fixture, training
scripts/     generate_episodes · rollout_keyframes · dagger_cycle · measure_tokens · fetch_tokenizer
tests/       898 tests; fixtures under tests/fixtures
HANDOFF.md   how to continue on another machine; what lives outside git
```

`docs/08-streaming-io-and-data-contract.md` is the source of truth for the robot stream. Where any other document disagrees with it, 08 wins.

## Quick start

```bash
uv sync                                   # Python 3.11; MuJoCo + robosuite 1.5, torch (CPU on macOS)
uv run python scripts/fetch_tokenizer.py  # Qwen tokenizer.json into artifacts/ (pinned by manifest)
uv run pytest -q                          # 898 passed, 1 xfailed

uv run python scripts/generate_episodes.py --config configs/data/d1_robot.yaml --count 4 --out artifacts/datasets/d1-robot/smoke
uv run python scripts/rollout_keyframes.py --limit 20
uv run python -m robo_jev.train --config configs/train/tiny_cpu.yaml
```

## Roadmap

1. Decide contract v0.3 (compact serialization, delta ticks, candidate space, allowed-set labels), seal semantic holdouts, regenerate the first batch, re-measure tokens.
2. Measure candidate backbones (2–4B, 9B) on deployment-class hardware with representative inputs; select by speed *and* zero-shot judgment quality.
3. First trained model in the cloud: readout-only, then backbone + readout, on non-robot data plus robot episodes; evaluate on held-out question semantics and calibration.
4. Full D1 rollouts, DAgger cycles, closed-loop comparison against the rule baseline with decision-stability metrics.

## Acknowledgements

The design follows the public description of TypeSafe's Jev / System One models (structured state in, typed probabilistic judgments out, one pass). The open minimal option scorers (jevlike, cua-s1) informed the tiny baseline and the shuffled-context control. robosuite and MuJoCo provide the simulator.
