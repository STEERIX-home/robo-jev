"""From MuJoCo outcome frequencies to A3's labels, robustness and pair metadata (record contract of v0.3b, A3's own names).

Per record and prior (``base`` is the training label, :data:`robo_jev.authors.a3.catalog.PRIORS`):

* **key** (yes/no) — the share of the key events (every outcome but the mildest) under the plan; **severe** (yes/no) — the share of
  the severe outcome; **score_cf** (0–1) — the key share at the counterfactual value of the decisive number;
* **direction** (choice, ``higher`` / ``lower`` / ``unchanged``) and **compare** (choice, ``as_written`` / the compared alternative /
  ``no_difference``) — a soft three-way reading of a change Δ in the key share (plan → counterfactual; plan → alternative):
  P(up) = σ((Δ − m)/s), P(down) = σ((−Δ − m)/s), P(same) = the rest, m = :data:`MARGIN` (= system1's direction threshold 0.1),
  s = :data:`SOFTNESS`. For ``compare``, "up" means the alternative is the riskier one;
* **action** (choice) — expected cost of each option (outcome shares × the scene's cost table) and probabilities
  softmax(−cost / :data:`TEMPERATURE`);
* **outcome** (choice) — the outcome shares under the plan.

**Robustness** per question: yes/no and score — the band (low < 1/3 ≤ middle ≤ 2/3 < high) under each prior, robust iff identical;
choices — the argmax under each prior (ties → the first option in the question's order), robust iff identical, plus ``no_reversal``
for the direction and compare questions (no prior points up while another points down).
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

from robo_jev.authors.a3 import catalog as cat

__all__ = ["MARGIN", "SOFTNESS", "TEMPERATURE", "band", "direction_word", "record_labels", "robustness", "three_way"]

MARGIN = 0.10
SOFTNESS = 0.04
TEMPERATURE = 0.08
FLAT = 0.10


def band(x: float) -> str:
    x = float(x)
    if x < 1.0 / 3.0:
        return "low"
    if x > 2.0 / 3.0:
        return "high"
    return "middle"


def _sigmoid(z: float) -> float:
    return 1.0 / (1.0 + math.exp(-z)) if z >= 0 else math.exp(z) / (1.0 + math.exp(z))


def three_way(delta: float) -> tuple[float, float, float]:
    """(P up, P down, P same) of a change `delta` in the key share."""
    up = _sigmoid((float(delta) - MARGIN) / SOFTNESS)
    down = _sigmoid((-float(delta) - MARGIN) / SOFTNESS)
    return up, down, max(0.0, 1.0 - up - down)


def _normalised(probs: Mapping[str, float]) -> dict[str, float]:
    """Round to 6 decimals and make the sum exactly 1 (the largest absorbs the rounding)."""
    out = {k: round(float(v), 6) for k, v in probs.items()}
    top = max(out, key=lambda k: out[k])
    out[top] = round(1.0 - sum(v for k, v in out.items() if k != top), 6)
    return out


def direction_word(delta: float) -> str:
    return "up" if delta >= FLAT - 1e-12 else ("down" if delta <= -FLAT + 1e-12 else "flat")


def record_labels(*, freq: Mapping[str, Mapping[str, Mapping[str, float]]], outcomes: Sequence[str], key_events: Sequence[str], severe: str,
                  costs: Mapping[str, float], names: Mapping[str, str], options: Mapping[str, str], compare_option: str,
                  plan_option: str = "as_written", same_option: str = "no_difference") -> dict[str, dict[str, dict]]:
    """``{prior: {question: label}}`` of one record from ``freq[variant][prior][outcome]`` (variants ``planned``, ``cf`` and the
    option variants named in `options`: option key → variant)."""
    out: dict[str, dict[str, dict]] = {}
    for prior in cat.PRIORS:
        def key_share(variant: str) -> float:
            return float(sum(float(freq[variant][prior][o]) for o in key_events))

        plan_key = key_share("planned")
        labels: dict[str, dict] = {
            names["key"]: {"type": "noul", "p_yes": round(plan_key, 6)},
            names["severe"]: {"type": "noul", "p_yes": round(float(freq["planned"][prior][severe]), 6)},
            names["score_cf"]: {"type": "score", "value": round(key_share("cf"), 6)},
        }
        up, down, same = three_way(key_share("cf") - plan_key)
        labels[names["direction"]] = {"type": "choice", "kind": "state",
                                      "probabilities": _normalised({"higher": up, "lower": down, "unchanged": same})}
        up, down, same = three_way(key_share(options[compare_option]) - plan_key)
        labels[names["compare"]] = {"type": "choice", "kind": "state",
                                    "probabilities": _normalised({plan_option: down, compare_option: up, same_option: same})}
        expected = {opt: round(sum(float(freq[variant][prior][o]) * float(costs[o]) for o in outcomes), 6) for opt, variant in options.items()}
        best = min(expected.values())
        weights = {opt: math.exp(-(c - best) / TEMPERATURE) for opt, c in expected.items()}
        total = sum(weights.values())
        labels[names["action"]] = {"type": "choice", "kind": "action", "probabilities": _normalised({k: w / total for k, w in weights.items()}),
                                   "expected_cost": expected}
        labels[names["outcome"]] = {"type": "choice", "kind": "state",
                                    "probabilities": _normalised({o: float(freq["planned"][prior][o]) for o in outcomes})}
        out[prior] = labels
    return out


def _argmax(probs: Mapping[str, float], order: Sequence[str]) -> str:
    top = max(float(probs[k]) for k in order)
    return next(k for k in order if float(probs[k]) >= top - 1e-9)


def robustness(by_prior: Mapping[str, Mapping[str, Mapping[str, Any]]], question_order: Mapping[str, Sequence[str]],
               reversal: Mapping[str, tuple[str, str]]) -> dict[str, dict]:
    """Per question (module docstring). `question_order[q]` = the option order of a choice question; `reversal[q]` = its (up, down)
    option keys for the direction and compare questions."""
    priors = list(cat.PRIORS)
    out: dict[str, dict] = {}
    for name, label in by_prior["base"].items():
        kind = label["type"]
        if kind in ("noul", "score"):
            field = "p_yes" if kind == "noul" else "value"
            values = {p: float(by_prior[p][name][field]) for p in priors}
            bands = {p: band(v) for p, v in values.items()}
            same = len(set(bands.values())) == 1
            out[name] = {"type": kind, "min": min(values.values()), "max": max(values.values()), "bands": bands, "band_same": same, "robust": same}
            continue
        order = list(question_order[name])
        argmax = {p: _argmax(by_prior[p][name]["probabilities"], order) for p in priors}
        same = len(set(argmax.values())) == 1
        entry = {"type": "choice", "argmax": argmax, "argmax_same": same, "robust": same,
                 "min": {o: min(float(by_prior[p][name]["probabilities"][o]) for p in priors) for o in order},
                 "max": {o: max(float(by_prior[p][name]["probabilities"][o]) for p in priors) for o in order}}
        if name in reversal:
            up, down = reversal[name]
            entry["no_reversal"] = not (up in argmax.values() and down in argmax.values())
        out[name] = entry
    return out
