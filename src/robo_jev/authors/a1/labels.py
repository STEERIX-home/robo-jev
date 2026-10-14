"""Author A1 — labels from per-draw MuJoCo outcomes (K = 32 shared draw points under each prior).

* yes/no (``noul``): the share of draws with the event — the key event (any failure) or the severe outcome — under the plan.
* score (``score_cf``): the key-event share at the counterfactual value of the decisive number.
* direction (choice ``state``: higher / lower / unchanged) and comparison (choice ``state``: plan riskier / alternative riskier /
  no difference): the per-draw paired change of the key event, Δ (mean over the K shared draws) with its standard error s
  (sample SD / √K, floored at 1/64): P(higher) = Φ((Δ − 0.10) / s), P(lower) = Φ((−Δ − 0.10) / s), the rest "unchanged".
* safest action (choice ``action``): expected cost = Σ outcome share × outcome cost + the action's own cost; probabilities =
  softmax(−cost / 0.08).
* outcome category (choice ``state``): the outcome shares under the plan.

Robustness per question: a yes/no or score question is robust iff its band (system1's low < 1/3 ≤ middle ≤ 2/3 < high) is the
same under the three priors; a choice question iff its argmax is; direction and comparison also record ``no_reversal`` (no prior
says the opposite of another).
"""

from __future__ import annotations

import math
from statistics import NormalDist
from typing import Any, Mapping, Sequence

import numpy as np

from robo_jev.bench.system1 import band

MARGIN = 0.10
SE_FLOOR = 1.0 / 64.0
ACTION_TEMPERATURE = 0.08
Z_CONTRAST = 1.96
_ND = NormalDist()


def _round(x: float) -> float:
    return round(float(x), 6)


def normalized(probs: Mapping[str, float]) -> dict[str, float]:
    """Rounded to 6 decimals, summing to 1 within 1e-6 (the last key takes the remainder)."""
    keys = list(probs)
    total = sum(max(0.0, float(probs[k])) for k in keys)
    vals = [_round(max(0.0, float(probs[k])) / total) for k in keys]
    vals[-1] = _round(max(0.0, 1.0 - sum(vals[:-1])))
    out = dict(zip(keys, vals))
    if abs(sum(out.values()) - 1.0) > 1e-6:  # the remainder hit zero: move the excess off the largest
        top = max(keys, key=lambda k: out[k])
        out[top] = _round(out[top] - (sum(out.values()) - 1.0))
    return out


def shares(outcomes: Sequence[str], categories: Sequence[str]) -> dict[str, float]:
    n = len(outcomes)
    return {c: _round(sum(1 for o in outcomes if o == c) / n) for c in categories}


def indicator(outcomes: Sequence[str], events: Sequence[str]) -> np.ndarray:
    return np.array([1.0 if o in events else 0.0 for o in outcomes])


def paired(diff: np.ndarray) -> tuple[float, float]:
    """Mean and standard error (floored) of per-draw paired differences."""
    k = len(diff)
    mean = float(np.mean(diff))
    se = float(np.std(diff, ddof=1) / math.sqrt(k)) if k > 1 else 0.0
    return mean, max(se, SE_FLOOR)


def three_way(diff: np.ndarray) -> tuple[float, float, float]:
    """(P(up), P(down), P(about the same)) of a paired change (module docstring)."""
    d, s = paired(diff)
    up, down = _ND.cdf((d - MARGIN) / s), _ND.cdf((-d - MARGIN) / s)
    return up, down, max(0.0, 1.0 - up - down)


def contrast(diff: np.ndarray) -> str:
    """Direction of a paired change by a two-sided z-test at 1.96 s: ``up`` / ``down`` / ``flat``."""
    d, s = paired(diff)
    if d > Z_CONTRAST * s:
        return "up"
    if d < -Z_CONTRAST * s:
        return "down"
    return "flat"


def move_direction(delta: float) -> str:
    return "up" if delta >= MARGIN - 1e-9 else ("down" if delta <= -MARGIN + 1e-9 else "flat")


def action_label(option_shares: Mapping[str, Mapping[str, float]], outcome_cost: Mapping[str, float],
                 action_cost: Mapping[str, float]) -> tuple[dict[str, float], dict[str, float]]:
    """(probabilities, expected costs) of the safest-action question."""
    costs = {opt: _round(sum(float(sh[o]) * float(outcome_cost[o]) for o in outcome_cost) + float(action_cost[opt]))
             for opt, sh in option_shares.items()}
    low = min(costs.values())
    weights = {opt: math.exp(-(c - low) / ACTION_TEMPERATURE) for opt, c in costs.items()}
    return normalized(weights), costs


def robust_scalar(by_prior: Mapping[str, float]) -> dict[str, Any]:
    bands = {p: band(v) for p, v in by_prior.items()}
    same = len(set(bands.values())) == 1
    return {"min": _round(min(by_prior.values())), "max": _round(max(by_prior.values())), "bands": bands, "band_same": same, "robust": same}


def argmax(probs: Mapping[str, float]) -> str:
    top = max(probs.values())
    return next(k for k, v in probs.items() if v == top)


def robust_choice(by_prior: Mapping[str, Mapping[str, float]], *, opposite: tuple[str, str] | None = None) -> dict[str, Any]:
    arg = {p: argmax(v) for p, v in by_prior.items()}
    keys = list(next(iter(by_prior.values())))
    out: dict[str, Any] = {"argmax": arg, "argmax_same": len(set(arg.values())) == 1, "robust": len(set(arg.values())) == 1,
                           "min": {k: _round(min(v[k] for v in by_prior.values())) for k in keys},
                           "max": {k: _round(max(v[k] for v in by_prior.values())) for k in keys}}
    if opposite is not None:
        out["no_reversal"] = not (opposite[0] in arg.values() and opposite[1] in arg.values())
    return out
