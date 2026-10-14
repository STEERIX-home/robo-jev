"""Author A1 — groups (a base record and its one-fact partner), their MuJoCo rollouts and the records.

**Plan** (:func:`plan_group`, no MuJoCo): stated facts, a target key probability ``aim_key`` ~ U(0.02, 0.98) turned into the
decisive number through the scene's closed-form surrogate (so the key labels come out roughly flat), the counterfactual value
(``near``: a small step, about 2/5; ``far``: aimed at a key probability at least 0.35 away), and the partner: a **one-factor**
pair (about 2/3 — one stated word or one non-decisive number changes; of eight random candidate changes the one whose
surrogate key is nearest a fresh U(0.02, 0.98) target, so partners are not piled at 0 and 1) or a **far-number** pair (the decisive number moves to a
key probability at least 0.40 away). The draw points (``draws_seed``: a K × D Latin hypercube) are shared by the three priors,
every variant and both members.

**Label** (:func:`simulate_group`, MuJoCo): every member × variant (planned, cf, the two alternative actions) × prior × draw is
one rollout; identical rollouts (e.g. the shared counterfactual of a far-number pair) run once. Results are cached per group
(JSON), so labelling is incremental and resumable.

**Assemble** (:func:`records_of_group`): the two records in the shared contract structure (``meta.question_kind``,
``labels_by_prior``, ``robustness``, ``pair``, ``plan``, ``outcomes``, ``variants``, …).
"""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Any, Mapping

import mujoco
import numpy as np

from robo_jev.authors.a1 import labels as L
from robo_jev.authors.a1 import sims
from robo_jev.authors.a1.scenes import DIR_KEYS, PLAN_KEY, PRIORS, SCENES, TIE_KEY, Scene, lhs

AUTHOR = "a1"
SET = "v06-a1"
GENERATOR = "v06a1-mujoco-v1"
K = 32
SURROGATE_N = 256
MASTER_SEED = 61_2026_1014
NEAR_STEP = {"mast": (0.2, 0.5), "dock": (0.02, 0.06), "ramp": (0.5, 1.5)}
AIM_TRIES, AIM_TOLERANCE = 30, 0.10
FACTOR_CANDIDATES = 8
PRIOR_SETTINGS = {
    "base": "catalog ranges of the stated words (friction, contents mass or centre of mass, placement sigma)",
    "alt_a": "friction ranges x0.7; mass and centre-of-mass ranges 1.5x wider about the same centre; placement sigma x1.5",
    "alt_b": "friction ranges x1.3; mass and centre-of-mass ranges half as wide about the same centre; placement sigma x0.75",
}


def seed_of(*parts: Any) -> int:
    return int.from_bytes(hashlib.sha256("|".join(str(p) for p in parts).encode()).digest()[:8], "big") >> 1


def record_id(scene: str, index: int, member: str) -> str:
    return f"v06a1-{scene}-{index:06d}" + ("p" if member == "pair" else "")


# --------------------------------------------------------------------------
# Plan
# --------------------------------------------------------------------------


def _far_target(rng: random.Random, aim: float, gap: float) -> float:
    """A target key probability at least `gap` away from `aim`, uniform over the allowed part of [0.02, 0.98]."""
    sides = [(a, b) for a, b in ((0.02, aim - gap), (aim + gap, 0.98)) if b > a]
    if not sides:
        return 0.02 if aim > 0.5 else 0.98
    x = rng.uniform(0.0, sum(b - a for a, b in sides))
    for a, b in sides:
        if x <= b - a:
            return a + x
        x -= b - a
    return sides[-1][1]


def plan_group(scene_name: str, index: int, master_seed: int = MASTER_SEED) -> dict[str, Any]:
    sc = SCENES[scene_name]
    seed = seed_of("v06a1", scene_name, index, master_seed)
    draws_seed = seed_of("v06a1-draws", scene_name, index, master_seed)
    rng = random.Random(seed)
    tpl = rng.randrange(sc.templates)
    u_surr = lhs(SURROGATE_N, len(sc.hidden_names), seed_of("v06a1-surrogate", scene_name, index))
    aim_key = rng.uniform(0.02, 0.98)
    # stated facts that can reach the target within the decisive number's range (else draw them again, at most AIM_TRIES times)
    for attempt in range(AIM_TRIES):
        visible = sc.sample_visible(rng)
        if scene_name == "ramp":
            visible["grade"] = 0.0
            visible["b_grade"] = 0.0
        value, static_key = sc.aim(visible, aim_key, u_surr)
        if abs(static_key - aim_key) <= AIM_TOLERANCE:
            break
    visible[sc.decisive] = value
    if scene_name == "ramp":
        lo, hi = sc.bounds(visible)
        visible["b_grade"] = sc.quantize(min(hi, max(lo, value * rng.uniform(0.35, 1.6) + rng.uniform(-2.0, 2.0))))
    # counterfactual value of the decisive number
    lo, hi = sc.bounds(visible)
    cf_kind = "near" if rng.random() < 0.4 else "far"
    cf_value, aim_cf = None, None
    if cf_kind == "far":
        aim_cf = _far_target(rng, static_key, 0.35)
        cf_value, aim_cf_static = sc.aim(visible, aim_cf, u_surr)
        if cf_value == value:
            cf_kind = "near"
    if cf_kind == "near":
        a, b = NEAR_STEP[scene_name]
        step = rng.uniform(a, b) * (1 if rng.random() < 0.5 else -1)
        cf_value = sc.quantize(min(hi, max(lo, value + step)))
        if cf_value == value:
            cf_value = sc.quantize(min(hi, max(lo, value - step)))
    aim_cf_static = sc.surrogate_at(visible, cf_value, u_surr)
    # partner
    pair_kind = "one_factor" if rng.random() < 2.0 / 3.0 else "far"
    partner = dict(visible)
    if pair_kind == "far":
        aim_p = _far_target(rng, static_key, 0.40)
        new, _ = sc.aim(visible, aim_p, u_surr)
        if new == value:
            new = sc.quantize(min(hi, max(lo, value + (hi - lo) * (0.3 if value < (lo + hi) / 2 else -0.3))))
        field, fact = sc.decisive, "number"
        partner[field] = new
    else:
        # one stated fact changes; among a few candidate changes keep the one whose surrogate key is nearest a fresh flat target
        target = rng.uniform(0.02, 0.98)
        best = None
        for _ in range(FACTOR_CANDIDATES):
            cand = sc.factor_change(visible, rng)
            trial = dict(visible)
            trial[cand[0]] = cand[1]
            gap = abs(sc.surrogate_at(trial, trial[sc.decisive], u_surr) - target)
            if best is None or gap < best[0]:
                best = (gap, cand)
        field, new, fact = best[1]
        partner[field] = new
    return {"scene": scene_name, "index": int(index), "seed": seed, "draws_seed": draws_seed, "template": tpl, "visible": visible,
            "aim_attempts": attempt + 1,
            "partner": partner, "aim_key": aim_key, "static_key": static_key, "static_key_partner": sc.surrogate_at(partner, partner[sc.decisive], u_surr),
            "cf_kind": cf_kind, "cf_value": cf_value, "aim_cf": aim_cf if aim_cf is not None else aim_cf_static, "static_cf": aim_cf_static,
            "pair": {"kind": pair_kind, "field": field, "from": visible[field], "to": new, "fact": fact}}


# --------------------------------------------------------------------------
# Variants and rollouts
# --------------------------------------------------------------------------


def variants_of(sc: Scene, visible: Mapping[str, Any], cf_value: float) -> dict[str, dict[str, Any]]:
    """Variant name → overrides of the stated facts: planned, cf, and one per alternative action (``opt_<action>``)."""
    out: dict[str, dict[str, Any]] = {"planned": {}, "cf": {sc.decisive: cf_value}}
    for action, overrides in sc.action_overrides(visible).items():
        if action != sc.actions[0]:
            out[f"opt_{action}"] = dict(overrides)
    return out


def _signature(scene: str, spec: Mapping[str, Any], state: Mapping[str, Any], hidden: list[dict[str, float]]) -> str:
    rows = [{k: round(float(v), 12) for k, v in h.items()} for h in hidden]
    return hashlib.sha256(json.dumps([scene, spec, state, rows], sort_keys=True).encode()).hexdigest()


def simulate_group(plan: Mapping[str, Any]) -> dict[str, Any]:
    """Every rollout of a group → ``{"members": {member: {variant: {prior: [outcome, …]}}}, "unstable": n, "rollouts": n}``."""
    sc = SCENES[plan["scene"]]
    u = lhs(K, len(sc.hidden_names), plan["draws_seed"])
    cache: dict[str, list[str]] = {}
    unstable = rollouts = 0
    members: dict[str, Any] = {}
    for member, visible in (("base", plan["visible"]), ("pair", plan["partner"])):
        out: dict[str, Any] = {}
        for variant, overrides in variants_of(sc, visible, plan["cf_value"]).items():
            v = sc.apply(visible, overrides)
            spec, state = sc.spec(v), sc.state(v)
            out[variant] = {}
            for prior in PRIORS:
                hidden = sc.hidden(v, prior, u)
                sig = _signature(sc.name, spec, state, hidden)
                if sig not in cache:
                    results = [sims.rollout(sc.name, spec, h, state) for h in hidden]
                    rollouts += len(results)
                    unstable += sum(1 for r in results if r["unstable"])
                    cache[sig] = [r["outcome"] for r in results]
                out[variant][prior] = cache[sig]
        members[member] = out
    return {"members": members, "unstable": unstable, "rollouts": rollouts}


def group_path(root: Path, scene: str, index: int) -> Path:
    return Path(root) / scene / f"{index:06d}.json"


def label_group(root: Path, scene: str, index: int, master_seed: int = MASTER_SEED) -> dict[str, Any]:
    """Plan + simulate one group and cache it (skips a group already cached)."""
    path = group_path(root, scene, index)
    if path.exists():
        return json.loads(path.read_text())
    plan = plan_group(scene, index, master_seed)
    result = {"plan": plan, **simulate_group(plan), "mujoco": mujoco.__version__}
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(result))
    tmp.replace(path)
    return result


# --------------------------------------------------------------------------
# Records
# --------------------------------------------------------------------------


def _member_labels(sc: Scene, outs: Mapping[str, Mapping[str, list[str]]], variants: Mapping[str, Any]) -> dict[str, Any]:
    """labels_by_prior, outcomes, robustness and the contrast directions of one member."""
    q = sc.q
    key, sev = sc.key_events, (sc.severe,)
    compare_variant = f"opt_{sc.compare_option}"
    option_variant = {a: ("planned" if a == sc.actions[0] else f"opt_{a}") for a in sc.actions}
    outcomes = {var: {p: L.shares(outs[var][p], sc.outcomes) for p in PRIORS} for var in variants}
    by_prior: dict[str, Any] = {}
    contrast_by_prior: dict[str, str] = {}
    for p in PRIORS:
        planned = L.indicator(outs["planned"][p], key)
        cf = L.indicator(outs["cf"][p], key)
        alt = L.indicator(outs[compare_variant][p], key)
        up, down, same = L.three_way(cf - planned)
        alt_up, alt_down, alt_same = L.three_way(alt - planned)
        probs, costs = L.action_label({a: outcomes[option_variant[a]][p] for a in sc.actions}, sc.outcome_cost, sc.action_cost)
        by_prior[p] = {
            q["key"]: {"type": "noul", "p_yes": round(float(planned.mean()), 6)},
            q["severe"]: {"type": "noul", "p_yes": round(float(L.indicator(outs["planned"][p], sev).mean()), 6)},
            q["score_cf"]: {"type": "score", "value": round(float(cf.mean()), 6)},
            q["direction"]: {"type": "choice", "kind": "state", "probabilities": L.normalized(dict(zip(DIR_KEYS, (up, down, same))))},
            q["compare"]: {"type": "choice", "kind": "state",
                           "probabilities": L.normalized({PLAN_KEY: alt_down, sc.compare_option: alt_up, TIE_KEY: alt_same})},
            q["action"]: {"type": "choice", "kind": "action", "probabilities": probs, "expected_cost": costs},
            q["outcome"]: {"type": "choice", "kind": "state", "probabilities": L.normalized(dict(outcomes["planned"][p]))},
        }
        contrast_by_prior[p] = L.contrast(planned - alt)
    robustness: dict[str, Any] = {}
    for kind, name in q.items():
        labels = {p: by_prior[p][name] for p in PRIORS}
        if kind in ("key", "severe"):
            robustness[name] = {"type": "noul", **L.robust_scalar({p: v["p_yes"] for p, v in labels.items()})}
        elif kind == "score_cf":
            robustness[name] = {"type": "score", **L.robust_scalar({p: v["value"] for p, v in labels.items()})}
        else:
            opposite = {"direction": ("higher", "lower"), "compare": (PLAN_KEY, sc.compare_option)}.get(kind)
            robustness[name] = {"type": "choice", **L.robust_choice({p: v["probabilities"] for p, v in labels.items()}, opposite=opposite)}
    directions = set(contrast_by_prior.values())
    return {"by_prior": by_prior, "outcomes": outcomes, "robustness": robustness, "contrast": contrast_by_prior,
            "contrast_same": len(directions) == 1, "contrast_no_reversal": not {"up", "down"} <= directions}


def records_of_group(cached: Mapping[str, Any], git: str | None = None) -> list[dict[str, Any]]:
    plan = cached["plan"]
    sc = SCENES[plan["scene"]]
    index = plan["index"]
    members = {"base": plan["visible"], "pair": plan["partner"]}
    built = {}
    for member, visible in members.items():
        variants = variants_of(sc, visible, plan["cf_value"])
        built[member] = (visible, variants, _member_labels(sc, cached["members"][member], variants))
    pair_meta_common = {
        "kind": "numeric", "field": plan["pair"]["field"], "from": plan["pair"]["from"], "to": plan["pair"]["to"],
        "closeness": "far" if plan["pair"]["kind"] == "far" else "one_factor",
        "factor": None if plan["pair"]["kind"] == "far" else plan["pair"]["field"], "fact": plan["pair"]["fact"],
    }
    key_q = sc.q["key"]
    key_by_prior = {p: [built["base"][2]["by_prior"][p][key_q]["p_yes"], built["pair"][2]["by_prior"][p][key_q]["p_yes"]] for p in PRIORS}
    dirs = {p: L.move_direction(v[1] - v[0]) for p, v in key_by_prior.items()}
    pair_meta_common.update({"key_by_prior": key_by_prior, "key_delta": round(key_by_prior["base"][1] - key_by_prior["base"][0], 6),
                             "direction_by_prior": dirs, "direction_same": len(set(dirs.values())) == 1,
                             "no_reversal": not {"up", "down"} <= set(dirs.values())})
    out = []
    for member, (visible, variants, lab) in built.items():
        rid = record_id(sc.name, index, member)
        other = record_id(sc.name, index, "pair" if member == "base" else "base")
        options = {a: ("planned" if a == sc.actions[0] else f"opt_{a}") for a in sc.actions}
        static_key = plan["static_key"] if member == "base" else plan["static_key_partner"]
        aim_key = plan["aim_key"] if member == "base" else static_key
        variant_priors = {}
        for var, overrides in variants.items():
            pr = sc.priors(sc.apply(visible, overrides))
            if pr != sc.priors(visible):
                variant_priors[var] = pr
        meta: dict[str, Any] = {
            "set": SET, "split": "train", "author": AUTHOR, "scene": sc.name, "group": sc.name, "law": sc.law, "phenomena": list(sc.phenomena),
            "language": "en", "domain": "warehouse", "style": "telemetry", "template": plan["template"],
            "key_question": key_q, "question_kind": {name: kind for kind, name in sc.q.items()},
            "key_events": list(sc.key_events), "severe_outcome": sc.severe, "plan_option": PLAN_KEY,
            "visible": dict(visible), "hidden_prior": sc.priors(visible), "prior_settings": dict(PRIOR_SETTINGS),
            "variants": variants,
            "plan": {"decisive": sc.decisive, "aim_key": aim_key, "aim_cf": plan["aim_cf"], "cf_kind": plan["cf_kind"], "cf_value": plan["cf_value"],
                     "options": options, "option_values": {v: o for v, o in variants.items() if v.startswith("opt_")}},
            "outcomes": lab["outcomes"], "labels_by_prior": lab["by_prior"], "robustness": lab["robustness"],
            "contrast_direction_by_prior": lab["contrast"], "contrast_direction_same": lab["contrast_same"],
            "contrast_no_reversal": lab["contrast_no_reversal"], "compare_option": sc.compare_option, "action_options": list(sc.actions),
            "static_aim": {"key": static_key, "cf": plan["static_cf"], "cf_kind": plan["cf_kind"]},
            "pair": {"of": other, "role": "base" if member == "base" else "variant", **pair_meta_common},
        }
        if variant_priors:
            meta["variant_hidden_prior"] = variant_priors
        record = {
            "schema": "simgen-request/0", "id": rid,
            "generator": {"name": GENERATOR, "family": sc.name, "group": index, "seed": plan["seed"], "draws_seed": plan["draws_seed"], "k": K,
                          "priors": list(PRIORS), "mujoco": cached.get("mujoco", mujoco.__version__), "git": git, "member": member},
            "meta": meta,
            "request": {"state": sc.render(visible, plan["template"]), "questions": sc.questions(visible, plan["cf_value"], plan["template"])},
            "labels": lab["by_prior"]["base"],
        }
        out.append(record)
    return out
