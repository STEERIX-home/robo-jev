"""Author A4 — groups (a record and its one-fact partner), MuJoCo rollouts over the shared draws, labels and robustness.

**Group.** :func:`build_group` draws a scene's visible facts, aims the decisive number at a target key share (``design.aim``), then
makes the partner: about 2/3 **factor pairs** (one non-decisive stated fact changes — a material/contents word or one other number)
and 1/3 **far-number pairs** (the decisive number moves to a second aim at least 0.3 away). The counterfactual value of the decisive
number (the score and direction questions) is *near* (a small step) or *far* (a third aim) and is shared by both records; so are the
alternative actions' stated parameters. Everything is a pure function of (scene, group index, seed).

**Rollouts.** :func:`simulate_group` runs every distinct (variant facts, prior) of the group's two records over the K shared draws
(identical hypotheticals are run once). A group with any rollout flagged ``unstable`` is dropped by the caller.

**Labels** (:func:`record_labels`), per prior, from the outcome frequencies:

* key / severe (yes/no) — the share of draws with the key event / the severe event under the plan;
* score at the counterfactual value — the key share at the counterfactual number;
* direction (more / less / about the same, key event, counterfactual vs plan) and comparison (plan / alternative / about the same) —
  paired over the shared draws: mean difference d of the key indicators, s = max(sd / √K, 0.02); P(up) = Φ((d − 0.1) / s),
  P(down) = Φ((−0.1 − d) / s), P(same) = the rest;
* action — expected cost of each option = Σ outcome frequency · outcome cost + the action's cost; probabilities = the share of 400 paired
  bootstrap resamples of the draws in which the option has the lowest mean cost (ties split);
* outcome — the outcome frequencies under the plan.

**Robustness**: a yes/no or score question is robust iff its band (low < 1/3 ≤ middle ≤ 2/3 < high) is the same under the three priors;
a choice question iff its argmax is; direction and comparison also record ``no_reversal`` (no prior says "up" while another says "down").
"""

from __future__ import annotations

import hashlib
import json
import math
from statistics import NormalDist
from typing import Any

import numpy as np

from robo_jev.authors.a4 import design as D
from robo_jev.authors.a4 import sims as S

__all__ = ["CHANGE_MIN", "build_group", "record_labels", "simulate_group"]

#: A paired change counts as a move ("up" / "down") from this size on (key share).
CHANGE_MIN = 0.1
BOOTSTRAP = 400
_PHI = NormalDist().cdf


def derive(*parts: Any) -> int:
    """A 63-bit integer seed from the parts (sha256)."""
    digest = hashlib.sha256("|".join(str(p) for p in parts).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") >> 1


def _far_target(rng: np.random.Generator, target: float) -> float:
    for _ in range(100):
        t = float(rng.uniform(0.03, 0.97))
        if abs(t - target) >= 0.3:
            return t
    return 0.97 if target < 0.5 else 0.03


def build_group(scene: str, index: int, seed: int) -> dict[str, Any] | None:
    """One group (two records' visible facts, pair, counterfactual, options); None if no valid group came out of 60 attempts."""
    spec = D.SCENE_SPECS[scene]
    group_seed = derive("a4", scene, index, seed)
    draws_seed = derive("a4-draws", scene, index, seed)
    rng = np.random.default_rng(group_seed)
    points = D.draw_points(draws_seed, scene)
    for attempt in range(60):
        vis = D.sample_visible(scene, rng)
        if scene == "ledge":
            vis["overhang"] = vis["length"] / 2  # placeholder for the factor checks; replaced by the aim
        hid = D.hidden_values(scene, D.prior_specs(scene, vis)["base"], points)
        target = float(rng.uniform(0.03, 0.97))
        value, share = D.aim(scene, vis, hid, target)
        if value is None or abs(share - target) > 0.15:
            continue
        vis[spec.decisive] = value
        partner = dict(vis)
        if rng.random() < 2.0 / 3.0:
            factor = D.choose(rng, spec.factors)
            change = D.factor_change(scene, vis, factor, rng)
            if change is None:
                continue
            new, fact = change
            partner[factor] = new
            pair = {"field": factor, "from": vis[factor], "to": new, "closeness": None, "factor": factor, "fact": fact}
            hid_p = D.hidden_values(scene, D.prior_specs(scene, partner)["base"], points)
        else:
            t2 = _far_target(rng, target)
            value2, share2 = D.aim(scene, vis, hid, t2)
            if value2 is None or value2 == value:
                continue
            partner[spec.decisive] = value2
            pair = {"field": spec.decisive, "from": value, "to": value2, "closeness": "far", "factor": None, "fact": "number"}
            hid_p = hid
        if rng.random() < 0.5:
            cf_kind = "near"
            step = float(rng.uniform(*spec.near)) * (value if spec.relative else 1.0)
            sign = 1.0 if rng.random() < 0.5 else -1.0
            cf = spec.round(value + sign * step)
            if cf == value:
                cf = spec.round(value - sign * step)
            if cf == value:
                continue
        else:
            cf_kind = "far"
            cf, _ = D.aim(scene, vis, hid, _far_target(rng, target))
            if cf is None or cf == value:
                continue
        options = D.options_values(scene, vis, rng)
        if scene == "bank":
            low = min(vis["slope"], partner["slope"])
            flat = max(1.0, low - options["opt_flatter"]["flat_drop"])
            if flat >= low:
                continue
            options["opt_flatter"]["flat_slope"] = flat
        if scene == "ledge" and min(vis["overhang"], partner["overhang"]) <= options["opt_short"]["short_by"] + 0.01:
            options["opt_short"]["short_by"] = round(max(0.05, min(vis["overhang"], partner["overhang"]) / 2), 2)
        alternatives = [k for k in D.OPTIONS[scene] if k != D.PLAN_KEY[scene]]
        compare = alternatives[int(rng.integers(len(alternatives)))]
        aims = {"key": share, "target": target, "partner_key": float(np.mean(D.key_analytic(scene, D.variant_facts(scene, partner, "planned", {}), hid_p)[0]))}
        aims["cf"] = float(np.mean(D.key_analytic(scene, D.variant_facts(scene, {**vis, spec.decisive: cf}, "planned", {}), hid)[0]))
        return {"scene": scene, "index": int(index), "seed": int(seed), "group_seed": group_seed, "draws_seed": draws_seed, "attempt": attempt,
                "records": [vis, partner], "pair": pair, "cf_kind": cf_kind, "cf_value": cf, "options": options, "compare_option": compare,
                "aims": aims}
    return None


# --------------------------------------------------------------------------
# Rollouts
# --------------------------------------------------------------------------


def _variants(group: dict[str, Any], member: int) -> dict[str, dict[str, Any]]:
    vis = group["records"][member]
    return D.resolve_variants(group["scene"], vis, group["options"], group["cf_value"])


def _job_key(phys: dict[str, Any], specs: dict[str, Any]) -> str:
    return json.dumps({"phys": phys, "specs": specs}, sort_keys=True, default=float)


def _run(scene: str, phys: dict[str, Any], hid: dict[str, np.ndarray], cache: dict) -> tuple[list[str], int]:
    """The K rollouts of one hypothetical under one prior's hidden values. The bank and ledge models are compiled per draw (the hidden
    centre of mass is compiled in — :mod:`.sims`); the anchor's robot has a fixed centre of mass and its model is reused."""
    outcomes, unstable = [], 0
    for k in range(D.K):
        if scene == "bank":
            down = phys["dims"][0]
            rig = S.bank_scene(phys["dims"], phys["mass"], (float(hid["com_down"][k]) * down, 0.0, float(hid["com_z"][k])))
            result = S.run_bank(rig, slope_deg=phys["slope"] + float(hid["slope_dev"][k]), mu=float(hid["mu"][k]))
        elif scene == "ledge":
            cx = float(hid["com_x"][k]) * (-1.0 if phys["mirror"] else 1.0)
            rig = S.ledge_scene(phys["dims"], phys["mass"], phys["drop"], (cx, 0.0, float(hid["com_z"][k])), phys["overhang"] + float(hid["stop_dev"][k]))
            result = S.run_ledge(rig, mu=float(hid["mu"][k]))
        else:
            key = tuple(phys["dims"]) + (phys["mass"],)
            if key not in cache:
                cache[key] = S.anchor_scene(phys["dims"], phys["mass"])
            result = S.run_anchor(cache[key], robot_slope_deg=phys["robot_slope"], mu=float(hid["mu"][k]),
                                  load_mass=phys["load_mass"] * float(hid["mass_ratio"][k]), load_slope_deg=phys["load_slope"],
                                  drag=float(hid["drag"][k]), start_accel=phys["accel"], line_speed=phys["line_speed"])
        outcomes.append(result["outcome"])
        unstable += int(bool(result["unstable"]))
    return outcomes, unstable


def simulate_group(group: dict[str, Any]) -> dict[str, Any]:
    """Every variant × prior of both records → ``{"draws": [{variant: {prior: [outcome per draw]}} per record], "rollouts", "unstable"}``."""
    scene = group["scene"]
    points = D.draw_points(group["draws_seed"], scene)
    cache: dict = {}
    done: dict[str, tuple[list[str], int]] = {}
    out: list[dict[str, dict[str, list[str]]]] = []
    rollouts = unstable = 0
    for member, vis in enumerate(group["records"]):
        specs = D.prior_specs(scene, vis)
        variants = _variants(group, member)
        per: dict[str, dict[str, list[str]]] = {}
        for variant in variants:
            phys = D.variant_facts(scene, vis, variant, variants)
            per[variant] = {}
            for prior in D.PRIORS:
                key = _job_key(phys, specs[prior])
                if key not in done:
                    hid = D.hidden_values(scene, specs[prior], points)
                    done[key] = _run(scene, phys, hid, cache)
                    rollouts += D.K
                    unstable += done[key][1]
                per[variant][prior] = list(done[key][0])
        out.append(per)
    return {"draws": out, "rollouts": rollouts, "unstable": unstable}


# --------------------------------------------------------------------------
# Labels
# --------------------------------------------------------------------------


def _freq(scene: str, outcomes: list[str]) -> dict[str, float]:
    return {o: outcomes.count(o) / len(outcomes) for o in D.OUTCOMES[scene]}


def _indicator(scene: str, outcomes: list[str], events: tuple[str, ...]) -> np.ndarray:
    return np.array([1.0 if o in events else 0.0 for o in outcomes])


def _round_probs(probs: dict[str, float]) -> dict[str, float]:
    """Round to 6 decimals and make the sum exactly 1 (the largest entry absorbs the remainder)."""
    out = {k: round(max(0.0, float(v)), 6) for k, v in probs.items()}
    top = max(out, key=lambda k: out[k])
    out[top] = round(out[top] + 1.0 - sum(out.values()), 6)
    return out


def paired_change(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    """(up, down, same) probabilities that the key share moves from `a` to `b` (paired over the shared draws; module docstring)."""
    diff = np.asarray(b, dtype=float) - np.asarray(a, dtype=float)
    mean = float(diff.mean())
    se = max(float(diff.std(ddof=0)) / math.sqrt(len(diff)), 0.02)
    up = _PHI((mean - CHANGE_MIN) / se)
    down = _PHI((-CHANGE_MIN - mean) / se)
    return {"up": up, "down": down, "same": max(0.0, 1.0 - up - down)}


def _action(scene: str, per_option: dict[str, list[str]], seed: int) -> tuple[dict[str, float], dict[str, float]]:
    """Expected costs and bootstrap 'cheapest' probabilities of the options (module docstring)."""
    names = list(per_option)
    costs = np.array([[D.OUTCOME_COST[scene][o] + D.ACTION_COST[scene][name] for o in per_option[name]] for name in names])
    expected = {name: round(float(costs[i].mean()), 6) for i, name in enumerate(names)}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, costs.shape[1], size=(BOOTSTRAP, costs.shape[1]))
    means = costs[:, idx].mean(axis=2)  # options × resamples
    best = means.min(axis=0)
    ties = np.isclose(means, best[None, :], atol=1e-12)
    share = (ties / ties.sum(axis=0, keepdims=True)).mean(axis=1)
    return _round_probs({name: float(share[i]) for i, name in enumerate(names)}), expected


def band(x: float) -> str:
    return "low" if x < 1 / 3 else ("high" if x > 2 / 3 else "middle")


def _direction_word(delta: float) -> str:
    return "up" if delta >= CHANGE_MIN - 1e-12 else ("down" if delta <= -CHANGE_MIN + 1e-12 else "flat")


def record_labels(group: dict[str, Any], sim: dict[str, Any], member: int, names: dict[str, str]) -> dict[str, Any]:
    """Labels by prior, outcomes, robustness and the in-record contrast of one record. `names` maps the question kinds (key, severe,
    score_cf, direction, compare, action, outcome) to this record's question names."""
    scene = group["scene"]
    draws = sim["draws"][member]
    event, severe = D.EVENT[scene], D.SEVERE[scene]
    options = D.OPTIONS[scene]
    compare_variant = options[group["compare_option"]]
    outcomes = {variant: {prior: _freq(scene, draws[variant][prior]) for prior in D.PRIORS} for variant in draws}
    by_prior: dict[str, dict[str, Any]] = {}
    for p_index, prior in enumerate(D.PRIORS):
        planned = draws["planned"][prior]
        key_plan = _indicator(scene, planned, event)
        key_cf = _indicator(scene, draws["cf"][prior], event)
        key_cmp = _indicator(scene, draws[compare_variant][prior], event)
        moved = paired_change(key_plan, key_cf)
        contrast = paired_change(key_plan, key_cmp)
        probs, expected = _action(scene, {name: draws[variant][prior] for name, variant in options.items()},
                                  derive("a4-action", group["draws_seed"], member, p_index))
        by_prior[prior] = {
            names["key"]: {"type": "noul", "p_yes": float(key_plan.mean())},
            names["severe"]: {"type": "noul", "p_yes": float(_indicator(scene, planned, severe).mean())},
            names["score_cf"]: {"type": "score", "value": float(key_cf.mean())},
            names["direction"]: {"type": "choice", "kind": "state",
                                 "probabilities": _round_probs({"goes_up": moved["up"], "goes_down": moved["down"], "stays_similar": moved["same"]})},
            names["compare"]: {"type": "choice", "kind": "state",
                               "probabilities": _round_probs({"the_plan": contrast["down"], group["compare_option"]: contrast["up"], "no_difference": contrast["same"]})},
            names["action"]: {"type": "choice", "kind": "action", "probabilities": probs, "expected_cost": expected},
            names["outcome"]: {"type": "choice", "kind": "state", "probabilities": _round_probs(outcomes["planned"][prior])},
        }
    robustness = {}
    for name in by_prior["base"]:
        entries = {prior: by_prior[prior][name] for prior in D.PRIORS}
        if entries["base"]["type"] in ("noul", "score"):
            values = {p: float(e.get("p_yes", e.get("value"))) for p, e in entries.items()}
            bands = {p: band(v) for p, v in values.items()}
            robustness[name] = {"type": entries["base"]["type"], "min": min(values.values()), "max": max(values.values()), "bands": bands,
                                "band_same": len(set(bands.values())) == 1, "robust": len(set(bands.values())) == 1}
        else:
            probs = {p: e["probabilities"] for p, e in entries.items()}
            argmax = {p: max(v, key=lambda k: (v[k], -list(v).index(k))) for p, v in probs.items()}
            keys = list(probs["base"])
            entry = {"type": "choice", "argmax": argmax, "argmax_same": len(set(argmax.values())) == 1, "robust": len(set(argmax.values())) == 1,
                     "min": {k: min(probs[p][k] for p in D.PRIORS) for k in keys}, "max": {k: max(probs[p][k] for p in D.PRIORS) for k in keys}}
            if name == names["direction"]:
                entry["no_reversal"] = not {"goes_up", "goes_down"} <= set(argmax.values())
            if name == names["compare"]:
                entry["no_reversal"] = not {"the_plan", group["compare_option"]} <= set(argmax.values())
            robustness[name] = entry
    contrast_by_prior = {}
    for prior in D.PRIORS:
        plan = float(_indicator(scene, draws["planned"][prior], event).mean())
        other = float(_indicator(scene, draws[compare_variant][prior], event).mean())
        contrast_by_prior[prior] = _direction_word(plan - other)
    return {"labels_by_prior": by_prior, "outcomes": outcomes, "robustness": robustness, "contrast_direction_by_prior": contrast_by_prior}


def pair_meta(group: dict[str, Any], sim: dict[str, Any], member: int, ids: tuple[str, str]) -> dict[str, Any]:
    """``meta.pair`` of one record (both records carry the same numbers; ``role`` base / variant)."""
    scene = group["scene"]
    keys = {prior: [float(_indicator(scene, sim["draws"][m]["planned"][prior], D.EVENT[scene]).mean()) for m in (0, 1)] for prior in D.PRIORS}
    directions = {prior: _direction_word(v[1] - v[0]) for prior, v in keys.items()}
    pair = group["pair"]
    return {"of": ids[1 - member], "role": "base" if member == 0 else "variant", "kind": "numeric", "field": pair["field"], "from": pair["from"],
            "to": pair["to"], "closeness": pair["closeness"], "factor": pair["factor"], "fact": pair["fact"], "key_by_prior": keys,
            "key_delta": keys["base"][1] - keys["base"][0], "direction_by_prior": directions, "direction_same": len(set(directions.values())) == 1,
            "no_reversal": not {"up", "down"} <= set(directions.values())}
