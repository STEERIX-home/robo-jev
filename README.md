# robojev — a real-time System One for decisions in the physical world

robojev is a research project by Steerix Robotics: a **learned judgment model that decides, many times per second, what a robot should do next** — which action to commit to, whether to stop, whether to grip, which path and speed — from a structured description of the world, without generating text.

It is a *System One* model in the sense that TypeSafe introduced with Jev: the input is a typed state plus a set of questions with candidate answers, and the output is one probability distribution per question, read out in a single forward pass. What is new here is the setting. The state is a physical scene that changes every 100 ms, the questions repeat on every tick, and the answers drive a real controller.

> Status (2026-09-19): the whole pipeline that can be verified without a GPU is implemented, reviewed and tested — contracts, data generators, simulator and controller, streaming harness, scripted expert, rollout labels, serialization, a tiny reference model, and a training loop with exact resume. **No real backbone has been trained yet. Latency has now been measured on the deployment-class GPU (DGX Spark GB10) with the official BF16 forward of Qwen3.5-2B/4B/9B: at the current serialization every candidate is ≥7× over the 10 Hz budget, and at the planned ≈500-token tick only the 2B is within reach (≈83–123 ms vs 80 ms).** See [Status](#status) for what is and is not verified.

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
| Data | place-phase fix (h0.5/e0.4/c0.6: free placement point inside the zone, observed-bottom place height, no `open` on hold/retreat place ticks — E1 sweep seeds 1–24/29/43 23/26 → 26/26, gripper events above the place height 16/30 → 0/33); batch-0 regenerated (40 episodes, E0 20/20 and E1 20/20 completed, 3,455 ticks, 0.71 MB/episode, 2,800 episodes/h); contrast siblings as a first-class generator output with a deletion QA gate (non-robot: one fact flipped per base record, pilot 2,000 records → 767 pairs, 0 deletion failures; robot: tick-level pairs in `contrast/records.jsonl`, batch-0 115 pairs); pre-128k sweep on batch-0 (9,880 rollouts: place 93.7 %, grasp 54 %, push 49 % with success conditioned on approach time and start distance and the failure stage split per direction — the no-contact push failures are approach-stage collisions on ±y where the hand body overlaps the object top, fixed by per-axis/hand-body contact offsets in h0.6; censoring 0.56 %, 0.49 s/rollout → 128k ≈ 17 CPU-h); a place stall guard (`place_stalled` after 15 readiness-wait ticks without descent, point excluded) and the rule baseline mirrored to e0.4 (rj0.5); sealed holdouts (template variants, one concept per domain, robot zoneF goal + one E1 family; `ood_dev`/`ood_test` by family hash) with a zero-leakage QA check; rollout machinery and re-costing (0.33 s/rollout) | D1 (400 episodes, 128k rollouts); human review of D0 |
| Model | contracts, masks, state forking, readout and losses on a random-weight fixture; native BF16 forward of Qwen3.5-2B/4B/9B (27B as reference) on GB10 with the fla and causal-conv1d kernels active | any *trained* backbone; the stream path (window, forking, pointer readout) on a real backbone |
| Training | step, TBPTT, sampler, checkpoint, resume on the fixture | GPU training; real-backbone results; calibration |
| Latency | token counts with the real tokenizer under contract v0.3 (10 objects, K=12: p50 332 / p95 599 / first tick 880 tokens, 1,331 with the episode prefix; 37K per 10-s chunk); native-path screen on DGX Spark (`scripts/measure_candidates.py`, 2026-09-19): p95 model time per tick 928 / 2,299 / 2,447 / 5,640 ms (2B / 4B / 9B / 27B) at the old format, 123 / 305 / 378 / 886 ms at a 500-token tick (≈83 / 197 / 278 ms extrapolated to a window-sized cache), and the v0.3 re-run for 2B/4B (`artifacts/reports/backbone-screen-v03.json`): at ≈400–440 tokens per tick, 2B p50 86–88 / p95 97–103 ms with the growing native cache (literal flags `fails_10hz`, and `deadline_fail` on upper/instruction_change: miss 0.11/0.14), ≈65–68 ms extrapolated to a window-sized cache (OLS below the measured 15.7–31K range, R² 0.25–0.59; first-5-tick mean 71–74 ms; `passes_10hz_window` true), 4B 195–200 / 225–247 ms (≈144–149 ms extrapolated, `passes_10hz_window` false) | the 10 Hz / 100 ms gate on the real stream path (stage 2: static window KV, CUDA graphs, an attention kernel without a materialized mask) |

Three measurements changed the plan; all three are now decided:

1. **Token budget (decided 2026-09-19, contract v0.3).** With the old serialization a tick cost 1,764–3,712 tokens (estimate was 450–700). Contract v0.3 — short field names, an object intro/dynamic split with delta ticks (intro every 30 ticks, dynamic on change or every 10), zones and scene in the prefix, a compact goal reference and key-based candidate lines — measures p50 332 / p95 599 / first tick 880 tokens (1,331 with the uncacheable 451-token episode prefix ≈ 80 ms on the 2B) at 10 objects and K=12 (37K per 10-s chunk; the regenerated batch-0 reads p50 393 / p95 566), inside the 500 / 800 / 1,200 / 60K budget. Details: docs/08 §3.
2. **Candidate cap (decided 2026-09-19, contract v0.3).** The speed profile left the joint key (`q_speed` answers speed), pushes are enumerated only toward the zones, the cap is K ≤ 12 (9 + 3 reserved) and the instructed target×zone combination is reserved before the round-robin spread (it reads the instruction, never the label). On the E1 sweep (seeds 1–24, 29, 43) the degenerate-`hold` rate went from 49.7% to 0% and completion from 20/26 to 23/26 (like-for-like on the 22 seeds whose plan the A6 scene fix did not change: 16/22 → 19/22; the 4 changed seeds were 4/4 before — one trivially at tick 0 — and 4/4 after). Non-keyframe ticks now carry a planner-cost allowed set (τ = 0.15) and executor-caused `hold` ticks follow a hold∉A rule; sealed holdouts (docs/04 §5) went in with the same change.
3. **Backbone (decided 2026-09-19).** The Spark screen fits per-request model time as an intercept plus a token cost — 26.5 / 51.6 / 93.6 ms + 38.6 / 107.7 / 152 ms per 1K new tokens for 2B / 4B / 9B — and ≈2.1 ms per 1K cached tokens for attention, so the 9B is 3.5–4.7× over budget at a 500-token tick and no single lever closes that. Qwen3.5-2B (primary) and Qwen3.5-4B (5 Hz fallback) go to stage 2; the 9B leaves the 10 Hz track (FP8 deferred until a profiler attribution in stage 2), the 27B is excluded. Full numbers: `artifacts/reports/backbone-screen.json`, docs/03.

## Repository

```text
docs/        research design (01–06), external reviews (07, 09, 10, 11), and the canonical stream contract (08)
src/robo_jev/
  contracts.py  data/  sim/  harness/  perception/      # world, harness, expert, data pipeline
  model/  loss.py  sampler.py  train.py  checkpoint.py   # model contracts and CPU training path
configs/     harness, controller, simulator, expert, events, data, model fixture, training
scripts/     generate_episodes · rollout_keyframes · dagger_cycle · measure_tokens · fetch_tokenizer · fetch_backbone · measure_candidates
tests/       962 tests; fixtures under tests/fixtures
HANDOFF.md   how to continue on another machine; what lives outside git
```

`docs/08-streaming-io-and-data-contract.md` is the source of truth for the robot stream. Where any other document disagrees with it, 08 wins.

## Quick start

```bash
uv sync                                   # Python 3.11; MuJoCo + robosuite 1.5, torch (CPU on macOS)
uv run python scripts/fetch_tokenizer.py  # Qwen tokenizer.json into artifacts/ (pinned by manifest)
uv run pytest -q                          # 962 passed (tokenizer present: 0 skipped, 0 xfailed)

uv run python scripts/generate_episodes.py --config configs/data/d1_robot.yaml --count 4 --out artifacts/datasets/d1-robot/smoke
uv run python scripts/rollout_keyframes.py --limit 20
uv run python -m robo_jev.train --config configs/train/tiny_cpu.yaml
```

## Roadmap

1. Decide contract v0.3 (compact serialization, delta ticks, candidate space, allowed-set labels), seal semantic holdouts, regenerate the first batch, re-measure tokens.
2. ~~Measure candidate backbones (2–4B, 9B) on deployment-class hardware with representative inputs~~ — done for speed (native path, 2026-09-19; see Status). Still open: zero-shot judgment quality, and stage 2 on the real stream path with the 2B/4B.
3. First trained model in the cloud: readout-only, then backbone + readout, on non-robot data plus robot episodes; evaluate on held-out question semantics and calibration.
4. Full D1 rollouts, DAgger cycles, closed-loop comparison against the rule baseline with decision-stability metrics.

## Acknowledgements

The design follows the public description of TypeSafe's Jev / System One models (structured state in, typed probabilistic judgments out, one pass). The open minimal option scorers (jevlike, cua-s1) informed the tiny baseline and the shuffled-context control. robosuite and MuJoCo provide the simulator.
