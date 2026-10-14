"""Author A2 — plan groups, label them in MuJoCo, assemble records (``simgen-request/0``, the v0.3b record contract's structure).

**Group** = a base record and its one-fact partner. Planning (:func:`plan_group`, pure, no MuJoCo):

1. sample the stated facts of a scene; draw a target key probability uniform in [0.02, 0.98]; solve the decisive number with the
   scene's static model on 600 base-prior samples so the static key probability hits the target (``static_aim``); round it as a
   person would state it;
2. the counterfactual value of the decisive number: ``near`` (× or ÷ 1.12–1.30) or ``far`` (× or ÷ 1.5–2.2), half each;
3. the partner: a third are **far-number pairs** (the decisive number × or ÷ 1.6–2.4, towards the other side of 0.5); two thirds are
   **one-factor pairs** (one material / surface / contents / build word, one other number, or the decisive number moved a little);
4. one alternative action, valid for both members — the comparison question's option and the second action option; the third
   action option is the scene's fallback (no physics: its expected cost is its fixed action cost).

Labelling (:func:`label_group`): every distinct physical setup of the group (planned, counterfactual and alternative of both
members — a decisive-number pair shares its counterfactual setup) × the three prior settings × K = 32 shared hidden draws.

Assembly (:func:`assemble_group`): outcome frequencies → labels under each prior (key, severe, score at the counterfactual value,
direction and comparison as soft three-way choices from the paired per-draw differences, action probabilities from expected costs,
outcome category), robustness, the pair block and the text.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from typing import Any, Mapping, Sequence

import numpy as np

from robo_jev.authors.a2 import catalog as C
from robo_jev.authors.a2 import text as T
from robo_jev.authors.a2.scenes import SCENES, hidden_values

__all__ = ["GENERATOR", "K", "SET", "assemble_group", "group_seed", "label_group", "plan_group", "soft_three"]

GENERATOR = "v06-a2-mujoco-v1"
SET = "v06-a2"
SCHEMA = "simgen-request/0"
K = 32
AIM_SAMPLES = 600
#: Soft three-way choice from paired differences: "about the same" margin and a floor on the spread.
SAME_MARGIN = 0.08
SPREAD_FLOOR = 0.04
#: Action probabilities: softmax of −expected cost / τ.
ACTION_TAU = 0.12
#: Direction / contrast threshold on a key-probability move.
MOVE_MIN = 0.10
BAND_EDGES = (1.0 / 3.0, 2.0 / 3.0)
FAR_PAIR_SHARE = 1.0 / 3.0


def group_seed(scene: str, index: int) -> int:
    return int.from_bytes(hashlib.sha256(f"v06-a2|{scene}|{int(index)}".encode()).digest()[:8], "little") >> 1


def _weighted(rng: np.random.Generator, options: Sequence[tuple[str, str, str, float]]) -> tuple[str, str, str, float]:
    weights = np.array([o[3] for o in options], dtype=float)
    return options[int(rng.choice(len(options), p=weights / weights.sum()))]


def _alt_variant(alt: str) -> str:
    return "opt_" + alt


def plan_group(scene: str, index: int) -> dict[str, Any]:
    """Facts, decisive number, counterfactual, partner and alternative of one group (module docstring). Deterministic in (scene, index)."""
    spec = SCENES[scene]
    seed = group_seed(scene, index)
    rng = np.random.default_rng(seed)
    draws_seed = int(rng.integers(1, 2**62))
    # wording and flavour (venue, names) come from their own streams: editing the text never moves the physics of a plan
    style = T.choices(np.random.default_rng([seed, 101]))
    flavour = np.random.default_rng([seed, 202])
    dec = spec.decisive
    for attempt in range(400):
        facts = spec.sample(rng, flavour)
        target = float(rng.uniform(0.02, 0.98))
        aim_draws = spec.draws(np.random.default_rng([seed, attempt]), AIM_SAMPLES)
        aim_vals = hidden_values(spec, spec.ranges(facts), aim_draws, "base")
        raw = spec.solve(facts, aim_vals, target)
        if not spec.decisive_ok(raw):
            continue
        facts[dec] = spec.round_decisive(raw)
        aim_key = spec.static_key(facts, aim_vals)
        # counterfactual value
        cf_kind = "near" if rng.random() < 0.5 else "far"
        factor = float(rng.uniform(1.12, 1.30) if cf_kind == "near" else rng.uniform(1.5, 2.2))
        moved = spec.risk_up(facts[dec], factor) if rng.random() < 0.5 else spec.risk_down(facts[dec], factor)
        if not spec.decisive_ok(moved):
            continue
        cf = spec.round_decisive(moved)
        if cf == facts[dec]:
            continue
        aim_cf = spec.static_key({**facts, dec: cf}, aim_vals)
        # partner
        if rng.random() < FAR_PAIR_SHARE:
            factor = float(rng.uniform(1.6, 2.4))
            new = spec.risk_down(facts[dec], factor) if aim_key >= 0.5 else spec.risk_up(facts[dec], factor)
            if not spec.decisive_ok(new):
                continue
            new = spec.round_decisive(new)
            pair = {"field": dec, "fact": "number", "factor": None, "closeness": "far", "one_factor": False}
        else:
            field, fact, factor_name, _ = _weighted(rng, spec.one_factor_options(facts))
            if factor_name == "decisive":
                factor = float(rng.uniform(1.12, 1.30))
                new = spec.risk_up(facts[dec], factor) if rng.random() < 0.5 else spec.risk_down(facts[dec], factor)
                if not spec.decisive_ok(new):
                    continue
                new = spec.round_decisive(new)
                pair = {"field": dec, "fact": "number", "factor": None, "closeness": "near", "one_factor": True}
            else:
                try:
                    new = spec.change_one_factor(facts, field, rng)
                except LookupError:
                    continue
                pair = {"field": field, "fact": fact, "factor": factor_name, "closeness": None, "one_factor": True}
        if new == facts[pair["field"]]:
            continue
        partner = {**facts, pair["field"]: new}
        if partner[dec] == cf:
            continue
        alts = [a for a in spec.alternatives(facts) if a in spec.alternatives(partner)]
        alt = str(rng.choice(alts))
        partner_vals = hidden_values(spec, spec.ranges(partner), aim_draws, "base")
        return {"scene": scene, "index": int(index), "seed": seed, "draws_seed": draws_seed, "attempts": attempt + 1, "style": style,
                "facts": facts, "partner": partner, "pair": {**pair, "from": facts[pair["field"]], "to": new}, "cf": cf, "cf_kind": cf_kind,
                "target": target, "aim_key": aim_key, "aim_cf": aim_cf, "aim_key_partner": spec.static_key(partner, partner_vals),
                "aim_cf_partner": spec.static_key({**partner, dec: cf}, partner_vals), "alt": alt}
    raise RuntimeError(f"{scene} group {index}: no valid plan in 400 attempts")


def member_variants(plan: Mapping[str, Any], member: str) -> dict[str, dict[str, Any]]:
    """Variant name → full facts of one member (``planned``, ``cf``, the alternative's ``opt_<alt>``)."""
    spec = SCENES[plan["scene"]]
    facts = plan["facts"] if member == "base" else plan["partner"]
    return {"planned": dict(facts), "cf": {**facts, spec.decisive: plan["cf"]},
            _alt_variant(plan["alt"]): {**facts, **spec.apply_alternative(facts, plan["alt"])}}


_SCENE_CACHE: dict = {}


def label_group(plan: Mapping[str, Any]) -> dict[str, Any]:
    """MuJoCo outcomes of every distinct setup of a group under the three priors (K shared draws). Outcome codes are the index of the
    outcome in the scene's outcome tuple, one character per draw; ``metric`` keeps the rollout's physical reading per draw (turntable:
    metres moved; tiltback: the hand's pull in N; wallpress: metres dropped)."""
    spec = SCENES[plan["scene"]]
    started = time.process_time()
    draws = spec.draws(np.random.default_rng(plan["draws_seed"]), K)
    setups: dict[str, dict[str, Any]] = {}
    for member in ("base", "pair"):
        for facts in member_variants(plan, member).values():
            setups.setdefault(spec.setup_key(facts), facts)
    metric_name = {"turntable": "moved_m", "tiltback": "pull_n", "wallpress": "drop_m"}[plan["scene"]]
    out: dict[str, Any] = {}
    unstable = 0
    for key, facts in setups.items():
        ranges = spec.ranges(facts)
        codes, metrics = {}, {}
        for prior in C.PRIORS:
            vals = hidden_values(spec, ranges, draws, prior)
            row, mrow = [], []
            for k in range(K):
                result = spec.run(_SCENE_CACHE, facts, {name: float(vals[name][k]) for name in vals})
                row.append(str(spec.outcomes.index(result["outcome"])))
                mrow.append(round(float(result[metric_name]), 4))
                unstable += int(bool(result.get("unstable")))
            codes[prior], metrics[prior] = "".join(row), mrow
        out[key] = {"codes": codes, "metric": metrics}
    return {"scene": plan["scene"], "index": plan["index"], "setups": out, "unstable": unstable, "cpu_s": round(time.process_time() - started, 3)}


# --------------------------------------------------------------------------
# Labels
# --------------------------------------------------------------------------


def band(x: float) -> str:
    return "low" if x < BAND_EDGES[0] else ("high" if x > BAND_EDGES[1] else "middle")


def _phi(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _normalise(probs: Mapping[str, float]) -> dict[str, float]:
    """Rounded to 6 decimals with the remainder put on the largest entry, so the sum is exactly 1 within 1e-9."""
    keys = list(probs)
    values = [max(0.0, float(probs[k])) for k in keys]
    total = sum(values)
    values = [round(v / total, 6) for v in values]
    top = int(np.argmax(values))
    values[top] = round(values[top] + (1.0 - sum(values)), 6)
    return dict(zip(keys, values))


def soft_three(diff: np.ndarray) -> tuple[float, float, float]:
    """(P(up), P(down), P(about the same)) of a paired per-draw difference of key indicators: a normal reading of its mean with the
    standard error (floored at :data:`SPREAD_FLOOR`) against the "about the same" margin :data:`SAME_MARGIN`."""
    diff = np.asarray(diff, dtype=float)
    mean = float(diff.mean())
    se = float(diff.std() / math.sqrt(len(diff)))
    spread = math.hypot(se, SPREAD_FLOOR)
    up = _phi((mean - SAME_MARGIN) / spread)
    down = _phi((-SAME_MARGIN - mean) / spread)
    return up, down, max(0.0, 1.0 - up - down)


def _direction(delta: float) -> str:
    return "up" if delta >= MOVE_MIN else ("down" if delta <= -MOVE_MIN else "flat")


def _robust_scalar(values: Mapping[str, float]) -> dict[str, Any]:
    bands = {p: band(v) for p, v in values.items()}
    same = len(set(bands.values())) == 1
    return {"type": None, "min": min(values.values()), "max": max(values.values()), "bands": bands, "band_same": same, "robust": same}


def _robust_choice(probs: Mapping[str, Mapping[str, float]], *, ordered: tuple[str, str] | None) -> dict[str, Any]:
    argmax = {p: max(v, key=lambda k: (v[k], -list(v).index(k))) for p, v in probs.items()}
    same = len(set(argmax.values())) == 1
    keys = list(next(iter(probs.values())))
    out: dict[str, Any] = {"type": "choice", "argmax": argmax, "argmax_same": same, "robust": same,
                           "min": {k: min(v[k] for v in probs.values()) for k in keys}, "max": {k: max(v[k] for v in probs.values()) for k in keys}}
    if ordered is not None:
        a, b = ordered
        out["no_reversal"] = not ({a, b} <= set(argmax.values()))
    return out


def _indicator(codes: str, spec: Any, events: Sequence[str]) -> np.ndarray:
    idx = {str(spec.outcomes.index(e)) for e in events}
    return np.array([1.0 if c in idx else 0.0 for c in codes])


def _freq(codes: str, spec: Any) -> dict[str, float]:
    return {o: round(codes.count(str(i)) / len(codes), 6) for i, o in enumerate(spec.outcomes)}


def member_labels(plan: Mapping[str, Any], labelled: Mapping[str, Any], member: str) -> dict[str, Any]:
    """Outcomes and labels by prior of one member (question names from :data:`robo_jev.authors.a2.text.QUESTION_KINDS`)."""
    scene = plan["scene"]
    spec = SCENES[scene]
    variants = member_variants(plan, member)
    alt_v = _alt_variant(plan["alt"])
    codes = {v: labelled["setups"][spec.setup_key(f)]["codes"] for v, f in variants.items()}
    kinds = T.QUESTION_KINDS[scene]
    names = {kind: name for name, kind in kinds.items()}
    more, less, same = T.DIRECTION_KEYS[scene]
    same_cmp = T.SAME_KEYS[scene]
    outcome_keys = T.OUTCOME_KEYS[scene]
    outcomes = {v: {p: _freq(codes[v][p], spec) for p in C.PRIORS} for v in variants}
    by_prior: dict[str, dict[str, Any]] = {}
    contrast = {}
    for p in C.PRIORS:
        x_plan = _indicator(codes["planned"][p], spec, spec.key)
        x_cf = _indicator(codes["cf"][p], spec, spec.key)
        x_alt = _indicator(codes[alt_v][p], spec, spec.key)
        up, down, flat = soft_three(x_cf - x_plan)
        a_up, a_down, a_flat = soft_three(x_alt - x_plan)
        cost = {o: spec.costs[o] for o in spec.outcomes}
        ec = {spec.plan_key: sum(outcomes["planned"][p][o] * cost[o] for o in spec.outcomes),
              plan["alt"]: spec.action_costs[plan["alt"]] + sum(outcomes[alt_v][p][o] * cost[o] for o in spec.outcomes),
              spec.fallback: spec.action_costs[spec.fallback]}
        ec = {k: round(v, 6) for k, v in ec.items()}
        low = min(ec.values())
        soft = _normalise({k: math.exp(-(v - low) / ACTION_TAU) for k, v in ec.items()})
        by_prior[p] = {
            names["key"]: {"type": "noul", "p_yes": round(float(x_plan.mean()), 6)},
            names["severe"]: {"type": "noul", "p_yes": round(float(_indicator(codes["planned"][p], spec, spec.severe).mean()), 6)},
            names["score_cf"]: {"type": "score", "value": round(float(x_cf.mean()), 6)},
            names["direction"]: {"type": "choice", "kind": "state", "probabilities": _normalise({more: up, less: down, same: flat})},
            names["compare"]: {"type": "choice", "kind": "state",
                               "probabilities": _normalise({spec.plan_key: a_down, plan["alt"]: a_up, same_cmp: a_flat})},
            names["action"]: {"type": "choice", "kind": "action", "probabilities": soft, "expected_cost": ec},
            names["outcome"]: {"type": "choice", "kind": "state",
                               "probabilities": _normalise({outcome_keys[o]: outcomes["planned"][p][o] for o in spec.outcomes})},
        }
        contrast[p] = _direction(float(x_cf.mean() - x_plan.mean()))
    robustness = {}
    for name, kind in kinds.items():
        if kind in ("key", "severe", "score_cf"):
            field = "value" if kind == "score_cf" else "p_yes"
            entry = _robust_scalar({p: by_prior[p][name][field] for p in C.PRIORS})
            entry["type"] = by_prior["base"][name]["type"]
            robustness[name] = entry
        else:
            ordered = {"direction": (more, less), "compare": (spec.plan_key, plan["alt"])}.get(kind)
            robustness[name] = _robust_choice({p: by_prior[p][name]["probabilities"] for p in C.PRIORS}, ordered=ordered)
    return {"variants": variants, "outcomes": outcomes, "labels_by_prior": by_prior, "robustness": robustness, "contrast": contrast}


def _variant_changes(variants: Mapping[str, Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    base = variants["planned"]
    return {v: {k: f[k] for k in f if f[k] != base.get(k)} for v, f in variants.items()}


def _visible(spec: Any, facts: Mapping[str, Any]) -> dict[str, Any]:
    out = dict(facts)
    out["dims"] = [round(float(x), 4) for x in spec.dims(facts)]
    if spec.name == "turntable":
        out["item_name"] = C.TT_ITEMS[facts["item"]]["name"]
        out["turn_deg"] = C.TT_TURNS[facts["turn"]][0]
    elif spec.name == "tiltback":
        out["piece_name"] = C.TB_PIECES[facts["piece"]]["name"]
    else:
        out["item_name"] = C.WP_ITEMS[facts["item"]]["name"]
    return out


def assemble_group(plan: Mapping[str, Any], labelled: Mapping[str, Any], *, git: str | None, mujoco_version: str) -> list[dict[str, Any]]:
    """The two records of a group (base first)."""
    scene = plan["scene"]
    spec = SCENES[scene]
    kinds = T.QUESTION_KINDS[scene]
    names = {kind: name for name, kind in kinds.items()}
    base_id = f"v06a2-{scene}-{int(plan['index']):06d}"
    ids = {"base": base_id, "pair": base_id + "p"}
    parts = {m: member_labels(plan, labelled, m) for m in ("base", "pair")}
    key_name = names["key"]
    key_by_prior = {p: [parts["base"]["labels_by_prior"][p][key_name]["p_yes"], parts["pair"]["labels_by_prior"][p][key_name]["p_yes"]]
                    for p in C.PRIORS}
    pair_dirs = {p: _direction(v[1] - v[0]) for p, v in key_by_prior.items()}
    alt_v = _alt_variant(plan["alt"])
    records = []
    for member in ("base", "pair"):
        part = parts[member]
        facts = part["variants"]["planned"]
        changes = _variant_changes(part["variants"])
        changes["planned"] = {}
        state, questions = T.render(scene, facts, plan["style"], plan["cf"], plan["alt"], changes[alt_v])
        aim_key = plan["aim_key"] if member == "base" else plan["aim_key_partner"]
        aim_cf = plan["aim_cf"] if member == "base" else plan["aim_cf_partner"]
        meta = {
            "set": SET, "split": "train", "scene": scene, "group": scene, "law": spec.law,
            "phenomena": {"A": ["acceleration", "slide", "tip"], "B": ["centre_of_mass", "edge"], "C": ["friction", "holding"]}[spec.law],
            "language": "en", "domain": facts["domain"], "style": "supervisor_briefing", "author": "a2",
            "key_question": key_name, "question_kind": dict(kinds),
            "key_events": list(spec.key), "severe_events": list(spec.severe), "plan_key": spec.plan_key,
            "visible": _visible(spec, facts),
            "hidden_prior": spec.hidden_prior(facts),
            "prior_settings": dict(C.PRIOR_SETTINGS),
            "variants": changes,
            "plan": {"decisive": spec.decisive, "aim_key": aim_key, "aim_cf": aim_cf, "cf_kind": plan["cf_kind"], "cf_value": plan["cf"],
                     "options": {spec.plan_key: "planned", plan["alt"]: alt_v, spec.fallback: None},
                     "option_values": {alt_v: changes[alt_v]}, "action_costs": dict(spec.action_costs), "outcome_costs": dict(spec.costs)},
            "outcomes": part["outcomes"],
            "labels_by_prior": part["labels_by_prior"],
            "robustness": part["robustness"],
            "contrast_direction_by_prior": part["contrast"],
            "contrast_direction_same": len(set(part["contrast"].values())) == 1,
            "contrast_no_reversal": not ({"up", "down"} <= set(part["contrast"].values())),
            "compare_option": plan["alt"],
            "action_options": [spec.plan_key, plan["alt"], spec.fallback],
            "static_aim": {"key": aim_key, "cf": aim_cf, "cf_kind": plan["cf_kind"], "target": plan["target"] if member == "base" else None},
            "pair": {"of": ids["pair" if member == "base" else "base"], "role": "base" if member == "base" else "variant", "kind": "numeric",
                     "field": plan["pair"]["field"], "from": plan["pair"]["from"], "to": plan["pair"]["to"],
                     "closeness": plan["pair"]["closeness"], "factor": plan["pair"]["factor"], "fact": plan["pair"]["fact"],
                     "one_factor": plan["pair"]["one_factor"], "key_by_prior": key_by_prior,
                     "key_delta": round(key_by_prior["base"][1] - key_by_prior["base"][0], 6), "direction_by_prior": pair_dirs,
                     "direction_same": len(set(pair_dirs.values())) == 1, "no_reversal": not ({"up", "down"} <= set(pair_dirs.values()))},
        }
        record = {
            "schema": SCHEMA,
            "id": ids[member],
            "generator": {"name": GENERATOR, "family": scene, "group": int(plan["index"]), "seed": int(plan["seed"]),
                          "draws_seed": int(plan["draws_seed"]), "k": K, "priors": list(C.PRIORS), "mujoco": mujoco_version, "git": git,
                          "member": member},
            "meta": meta,
            "request": {"state": state, "questions": questions},
            "labels": part["labels_by_prior"]["base"],
        }
        records.append(json.loads(json.dumps(record)))
    return records
