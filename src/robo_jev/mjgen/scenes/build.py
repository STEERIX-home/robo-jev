"""v0.4a pipeline — candidate groups, one-number and one-factor pairs, MuJoCo labels under three priors, records, balanced selection
and statistics for the new scenes (Task v0.4a §1). The rules are v0.3b's (:mod:`robo_jev.mjgen.families`, :mod:`robo_jev.mjgen.generate`),
generalised from three fixed families to a registry of scenes:

* **groups and hidden draws** — one scrambled Latin hypercube of K points per group (seed = group seed ^ 0xD3A5), mapped through each
  prior setting (``base``, ``alt_a``: friction × 0.7 and ranges 1.5 × wider, ``alt_b``: friction × 1.3 and ranges half as wide;
  placement errors × 1.5 / × 0.75), so every prior, variant and pair member sees the same hidden worlds;
* **aiming** — the decisive number solves "static key-event rate under the base prior = a target drawn uniformly on [−0.06, 1.06]";
  the counterfactual asked inside the record is aimed at a second target (far in 80 %, close in 20 %);
* **pairs** — a far-aimed one-number pair in a share of the base run; at selection every kept group without one gets a one-factor
  partner (a material, wetness or contents word, or one physical number), so about two thirds of the pairs are one-factor pairs;
* **labels** — event shares per variant and prior; direction and comparison from the paired per-draw differences
  (:func:`robo_jev.mjgen.generate.contrast_probs`); safest action = softmax(−(cost − min)/0.05), cost = P(event) + 0.5·P(severe);
  robustness across the three priors (:func:`robo_jev.mjgen.generate._robust`).
"""

from __future__ import annotations

import collections
import copy
import dataclasses
import hashlib
import json
import math
import random
import statistics
from typing import Any, Callable, Iterable, Sequence

import numpy as np

from robo_jev.decisions.records import SCHEMA, validate_record
from robo_jev.mjgen import families as fam
from robo_jev.mjgen import generate as gen
from robo_jev.mjgen.scenes.base import Scene

__all__ = [
    "GENERATOR",
    "candidate",
    "factor_pair",
    "group_records",
    "group_seed",
    "label_group",
    "select_balanced",
    "static_rate",
    "stats",
]

GENERATOR = "v04a-mujoco-scenes-v1"
PERSON = {"log": "third", "sheet": "third", "brief": "second", "self": "first"}


def _registry() -> dict[str, Scene]:
    from robo_jev.mjgen.scenes import SCENES

    return SCENES


def spec(name: str) -> Scene:
    return _registry()[name]


def group_seed(master: int, scene: str, index: int) -> int:
    """Seed of one group: the first 8 bytes of sha256("v04a|master|scene|index")."""
    return int.from_bytes(hashlib.sha256(f"v04a|{int(master)}|{scene}|{int(index)}".encode("utf-8")).digest()[:8], "big")


# --------------------------------------------------------------------------
# Static rates (aiming only)
# --------------------------------------------------------------------------


def _draws(scene: Scene, visible: dict[str, Any], seed: int, k: int, prior: str = "base", cache: dict | None = None) -> list[dict[str, float]]:
    hidden = scene.hidden_for(visible)
    names = sorted(hidden)
    key = (tuple(names), int(k), int(seed))
    if cache is not None and key in cache:
        unit = cache[key]
    else:
        unit = fam.unit_draws(names, k, seed)
        if cache is not None:
            cache[key] = unit
    return fam.draw_values(fam.prior_ranges(hidden, prior), unit, names)


def static_rate(scene: Scene, visible: dict[str, Any], draws: Sequence[dict[str, float]]) -> float:
    """Share of draws whose static outcome is the key event."""
    return sum(1 for d in draws if scene.static_outcome(visible, d) in scene.event) / len(draws)


def _rate_fn(scene: Scene, visible: dict[str, Any], field: str, seed: int, k: int, cache: dict) -> Callable[[float], float]:
    """Static key rate as a function of one stated number (the hidden draws re-mapped when the number moves a prior)."""

    def rate(value: float) -> float:
        trial = copy.deepcopy(visible)
        fam._set_field(trial, field, value)
        return static_rate(scene, trial, _draws(scene, trial, seed, k, cache=cache))

    return rate


# --------------------------------------------------------------------------
# Candidate groups
# --------------------------------------------------------------------------


def candidate(name: str, index: int, master: int, *, k: int = fam.DEFAULT_K, pair_share: float = 0.33) -> fam.Group:
    """Candidate group `index` of scene `name` (deterministic in (master, name, index, k))."""
    scene = spec(name)
    seed = group_seed(master, name, index)
    rng = random.Random(seed)
    cache: dict = {}
    for _ in range(200):
        visible, decisive = scene.sample(rng)
        field = decisive["field"]
        rate_at = _rate_fn(scene, visible, field, seed ^ 0xD3A5, k, cache)
        aim_key = fam._target(rng)
        value = fam.solve_value(rate_at, decisive["lo"], decisive["hi"], min(max(aim_key, 1e-3), 1.0), increasing=decisive["increasing"])
        if aim_key > 1.0:  # clear-cut: push past the last threshold
            value = value * (1.15 if decisive["increasing"] else 0.85)
        elif aim_key < 0.0:
            value = value * (0.85 if decisive["increasing"] else 1.15)
        value = fam._round(value, decisive["step"])
        if not decisive["lo"] <= value <= decisive["hi"]:
            continue
        fam._set_field(visible, field, value)
        if not scene.accept(visible):
            continue
        aim_cf, cf_kind = fam._second_target(rng, aim_key)
        cf_value = fam._round(fam.solve_value(rate_at, decisive["lo"], decisive["hi"], max(aim_cf, 1e-3), increasing=decisive["increasing"]), decisive["step"])
        if abs(cf_value - value) < decisive["step"] / 2:
            cf_value = fam._round(value + decisive["step"] * (1 if rng.random() < 0.5 else -1) * rng.choice((1, 2, 3)), decisive["step"])
        trial = copy.deepcopy(visible)
        fam._set_field(trial, field, cf_value)
        if not decisive["lo"] <= cf_value <= decisive["hi"] or not scene.accept(trial):
            continue
        group = fam.Group(family=name, index=index, seed=seed, visible=visible, hidden=scene.hidden_for(visible), variants={}, plan={},
                          style=rng.choice(fam.STYLES), wording=rng.getrandbits(32), mix=fam._mix(rng))
        group.plan = {"decisive": field, "aim_key": aim_key, "aim_cf": aim_cf, "cf_kind": cf_kind, "cf_value": cf_value,
                      "decisive_span": [decisive["lo"], decisive["hi"], decisive["step"], decisive["increasing"]]}
        group.variants = {"planned": {}, "cf": {field: cf_value}}
        scene.options(group, rng)
        group.plan["option_values"] = {v: dict(group.variants[v]) for v in group.plan["options"].values() if v != "planned"}
        if rng.random() < pair_share:
            pair = number_pair(group, rng, k=k)
            if pair is not None:
                group.pair = pair
        draws = _draws(scene, visible, seed ^ 0xD3A5, k, cache=cache)
        group.notes = {"static_key": static_rate(scene, visible, draws), "static_cf": rate_at(cf_value)}
        return group
    raise RuntimeError(f"{name} #{index}: no valid scene in 200 tries")


def number_pair(group: fam.Group, rng: random.Random, *, k: int = fam.DEFAULT_K) -> dict[str, Any] | None:
    """A far-aimed one-number pair (``meta.pair``): one stated number changed so the static key rate moves to a far target; None when
    no valid value is found. Near or free pairs are not built (v0.3b dropped them at selection)."""
    scene = spec(group.family)
    if not scene.pair_fields:
        return None
    roll = rng.random()
    acc = 0.0
    name = scene.pair_fields[-1][0]
    for field, share in scene.pair_fields:
        acc += share
        if roll < acc:
            name = field
            break
    v = group.visible
    if name == "decisive":
        name = group.plan["decisive"]
    if not name.startswith("dims.") and v.get(name) is None:  # e.g. a stated mass when the text gives only a range
        return None
    current = fam._get_field(v, name)
    if name == group.plan["decisive"]:
        lo, hi, step, increasing = group.plan["decisive_span"]
    else:
        lo, hi, step = scene.pair_spans[name](current, v)
        increasing = None
    rate = _rate_fn(scene, v, name, group.seed ^ 0xD3A5, k, {})
    if increasing is None:
        increasing = rate(lo) <= rate(hi)
    aim_key = min(1.0, max(0.0, group.plan["aim_key"]))
    aim_pair, closeness = fam._second_target(rng, aim_key, near_share=0.0)
    value = fam._round(fam.solve_value(rate, lo, hi, max(aim_pair, 1e-3), increasing=increasing), step)
    if abs(value - current) < 1.5 * step or not lo <= value <= hi:
        return None
    if name == group.plan["decisive"] and abs(value - group.plan["cf_value"]) < 1e-9:
        return None
    trial = copy.deepcopy(v)
    fam._set_field(trial, name, value)
    if not scene.accept(trial):
        return None
    return {"field": name, "from": current, "value": value, "aim": aim_pair, "closeness": closeness}


def factor_pair(group: fam.Group, rng: random.Random, *, k: int = fam.DEFAULT_K) -> dict[str, Any] | None:
    """A paired record that changes exactly one physical factor of the base scene (``scene.factors``). Word factors take the alternative
    whose static key rate moves furthest (a random one among those moving ≥ 0.25); ``aim`` numbers are aimed at a far static target;
    ``choice`` numbers (outside the static model: ramp-up time, release height, run speed …) take a large change and MuJoCo decides how
    much it matters. None when the scene offers no valid change."""
    scene = spec(group.family)
    v = group.visible
    names = list(scene.factors)
    rng.shuffle(names)
    cache: dict = {}
    seed = group.seed ^ 0xD3A5
    base_rate = static_rate(scene, v, _draws(scene, v, seed, k, cache=cache))
    for factor in names:
        entry = scene.factors[factor]
        field, how = entry["field"], entry["how"]
        current = fam._get_field(v, field) if field in v or field.startswith("dims.") else None
        if how == "word":
            candidates = [c for c in scene.factor_choices(group, factor, current) if c != current]
            if not candidates:
                continue
            scored = []
            for value in candidates:
                trial = copy.deepcopy(v)
                fam._set_field(trial, field, value)
                if not scene.accept(trial):
                    continue
                scored.append((abs(static_rate(scene, trial, _draws(scene, trial, seed, k, cache=cache)) - base_rate), value))
            if not scored:
                continue
            movers = [value for delta, value in scored if delta >= 0.25]
            value = rng.choice(movers) if movers else max(scored, key=lambda dv: (dv[0], str(dv[1])))[1]
        elif how == "aim":
            lo, hi, step = entry["span"](current, v)
            rate = _rate_fn(scene, v, field, seed, k, cache)
            if abs(rate(lo) - rate(hi)) < 0.05:  # the static model says this number does not matter here (e.g. the fork spread in a stop)
                continue
            aim, _ = fam._second_target(rng, base_rate, near_share=0.0)
            increasing = rate(lo) <= rate(hi)
            value = fam._round(fam.solve_value(rate, lo, hi, max(aim, 1e-3), increasing=increasing), step)
            if abs(value - current) < 1.5 * step:
                continue
        elif how == "choice":
            candidates = [c for c in scene.factor_choices(group, factor, current) if c != current]
            if not candidates:
                continue
            value = rng.choice(candidates)
        else:
            raise ValueError(f"factor kind {how!r}")
        trial = copy.deepcopy(v)
        fam._set_field(trial, field, value)
        if value == current or not scene.accept(trial):
            continue
        if field == group.plan["decisive"] and abs(float(value) - group.plan["cf_value"]) < 1e-9:
            continue
        return {"field": field, "from": current, "value": value, "aim": None, "closeness": "factor", "factor": factor, "fact": entry["fact"]}
    return None


# --------------------------------------------------------------------------
# MuJoCo labels
# --------------------------------------------------------------------------


def _effective(group: fam.Group, member: str, overrides: dict[str, Any]) -> dict[str, Any]:
    visible = copy.deepcopy(group.visible_for(member))
    for name, value in overrides.items():
        fam._set_field(visible, name, value)
    return visible


def label_group(group: fam.Group, k: int = fam.DEFAULT_K, *, rigs: dict[str, Any] | None = None, only: Sequence[str] | None = None) -> dict[str, Any]:
    """Roll out every variant of both members under all three priors on the group's shared unit draws (v0.3b's
    :func:`robo_jev.mjgen.families.label_group`). Identical (scene, prior) specs are rolled once."""
    rigs = {} if rigs is None else rigs
    scene = spec(group.family)
    draws_seed = group.seed ^ 0xD3A5
    cache: dict[str, list[dict[str, Any]]] = {}
    members = ("base", "pair") if group.pair is not None else ("base",)
    if only is not None:
        members = tuple(m for m in members if m in only)
    out: dict[str, Any] = {"k": int(k), "draws_seed": int(draws_seed), "members": {}, "rollouts": 0, "flags": {"unstable": 0}}
    for member in members:
        visible_member = group.visible_for(member)
        hidden = scene.hidden_for(visible_member)
        names = sorted(hidden)
        unit = fam.unit_draws(names, k, draws_seed)
        entry: dict[str, Any] = {"hidden": {p: fam.prior_ranges(hidden, p) for p in fam.PRIORS}, "variants": {}}
        for name, overrides in group.variants.items():
            visible = _effective(group, member, overrides)
            per_prior: dict[str, list[str]] = {}
            freq: dict[str, dict[str, float]] = {}
            for prior in fam.PRIORS:
                ranges = entry["hidden"][prior]
                key = scene.physics_key(visible) + "|" + json.dumps(ranges, sort_keys=True)
                if key not in cache:
                    draws = fam.draw_values(ranges, unit, names)
                    cache[key] = [scene.roll(visible, d, rigs) for d in draws]
                    out["rollouts"] += len(draws)
                    out["flags"]["unstable"] += sum(1 for r in cache[key] if r.get("unstable"))
                rolls = cache[key]
                per_prior[prior] = [r["outcome"] for r in rolls]
                freq[prior] = {o: sum(1 for r in rolls if r["outcome"] == o) / len(rolls) for o in scene.outcomes}
            entry["variants"][name] = {"outcomes": per_prior, "freq": freq}
        out["members"][member] = entry
    return out


# --------------------------------------------------------------------------
# Records
# --------------------------------------------------------------------------


def _events(scene: Scene, outcomes: Sequence[str]) -> list[bool]:
    return [o in scene.event for o in outcomes]


def _action_keys(group: fam.Group) -> list[str]:
    scene = spec(group.family)
    keys = list(group.plan["options"])
    if scene.dominant_option and random.Random(group.wording ^ 0xAC71).random() < 0.5:
        keys = [k for k in keys if k != scene.dominant_option]
    return keys


def _cf_option(group: fam.Group) -> str:
    keys = [k for k, v in group.plan["options"].items() if v != "planned"]
    return random.Random(group.wording ^ 0xC0DE).choice(keys)


def _labels(group: fam.Group, variants: dict[str, Any], prior: str, questions: dict[str, Any], cf_option: str) -> dict[str, Any]:
    scene = spec(group.family)

    def ev(name: str) -> list[bool]:
        return _events(scene, variants[name]["outcomes"][prior])

    def freq(name: str) -> dict[str, float]:
        return variants[name]["freq"][prior]

    out: dict[str, Any] = {}
    for kind, qname in scene.question_names.items():
        if qname not in questions:
            continue
        if kind == "key":
            out[qname] = {"type": "noul", "p_yes": float(np.mean(ev("planned")))}
        elif kind == "severe":
            out[qname] = {"type": "noul", "p_yes": float(freq("planned")[scene.severe])}
        elif kind == "score_cf":
            out[qname] = {"type": "score", "value": float(np.mean(ev("cf")))}
        elif kind == "outcome":
            probs = {key: float(freq("planned")[o]) for key, o in scene.outcome_keys.items()}
            out[qname] = {"type": "choice", "kind": "state", "probabilities": gen._normalise(probs)}
        elif kind == "direction":
            c = gen.contrast_probs(ev("planned"), ev("cf"))
            out[qname] = {"type": "choice", "kind": "state", "probabilities": {"more_likely": c["more"], "less_likely": c["less"], "about_same": c["same"]}}
        elif kind == "compare":
            other = group.plan["options"][cf_option]
            c = gen.contrast_probs(ev("planned"), ev(other))
            out[qname] = {"type": "choice", "kind": "state", "probabilities": {scene.planned_key: c["less"], cf_option: c["more"], "about_same": c["same"]}}
        elif kind == "action":
            costs = {}
            for key in questions[qname]["criteria"]:
                vname = group.plan["options"][key]
                costs[key] = round(float(np.mean(ev(vname))) + gen.SEVERE_WEIGHT * float(freq(vname)[scene.severe]), 6)
            out[qname] = {"type": "choice", "kind": "action", "probabilities": gen._softmax_costs(costs), "expected_cost": costs}
    return out


SET_NAME = "v04a-mujoco-scenes"
SPLIT = "train"
ID_PREFIX = "v04a"


def group_records(group: fam.Group, labelled: dict[str, Any], *, mujoco_version: str, git: str | None = None, set_name: str = SET_NAME,
                  split: str = SPLIT, id_prefix: str = ID_PREFIX) -> list[dict[str, Any]]:
    """One labelled group → its records (the base, and the paired record when there is one) — v0.3b's record layout, plus ``meta.law``."""
    scene = spec(group.family)
    names = scene.question_names
    cf_option = _cf_option(group)
    action_keys = _action_keys(group)
    members = ("base", "pair") if group.pair is not None else ("base",)
    built: dict[str, dict[str, Any]] = {}
    for member in members:
        visible = group.visible_for(member)
        state = scene.state_text(visible, group.style, random.Random(group.wording ^ 0x57A7E))
        questions = scene.questions(visible, group.mix, group.plan, PERSON[group.style], random.Random(group.wording ^ 0x0E57), cf_option, action_keys)
        variants = labelled["members"][member]["variants"]
        by_prior = {p: _labels(group, variants, p, questions, cf_option) for p in fam.PRIORS}
        robustness = {q: gen._robust(questions[q], {p: by_prior[p][q] for p in fam.PRIORS}) for q in questions}
        key = names["key"]
        contrast = {p: gen.direction(float(np.mean(_events(scene, variants["cf"]["outcomes"][p]))) - by_prior[p][key]["p_yes"]) for p in fam.PRIORS}
        kinds = {qname: kind for kind, qname in names.items() if qname in questions}
        record = {
            "schema": SCHEMA,
            "id": f"{id_prefix}-{group.family}-{group.index:06d}" + ("p" if member == "pair" else ""),
            "generator": {"name": GENERATOR, "family": group.family, "law": scene.law, "group": group.index, "seed": group.seed,
                          "draws_seed": labelled["draws_seed"], "k": labelled["k"], "priors": list(fam.PRIORS), "mujoco": mujoco_version, "git": git,
                          "member": member},
            "meta": {
                "set": set_name, "split": split, "scene": group.family, "law": scene.law, "group": group.family, "phenomena": [scene.law, group.family],
                "language": "en", "domain": visible.get("domain"), "style": group.style, "key_question": key, "question_kind": kinds,
                "visible": visible, "hidden_prior": {p: gen._jsonable(r) for p, r in labelled["members"][member]["hidden"].items()},
                "prior_settings": {p: fam.PRIOR_SETTINGS[p]["what"] for p in fam.PRIORS},
                "variants": {name: group.variants[name] for name in variants}, "plan": gen._jsonable(group.plan),
                "outcomes": {name: entry["freq"] for name, entry in variants.items()},
                "labels_by_prior": by_prior, "robustness": robustness,
                "contrast_direction_by_prior": contrast, "contrast_direction_same": len(set(contrast.values())) == 1,
                "contrast_no_reversal": not {"up", "down"} <= set(contrast.values()),
                "compare_option": cf_option if names["compare"] in questions else None,
                "action_options": action_keys if names["action"] in questions else None,
                "static_aim": {"key": group.plan.get("aim_key"), "cf": group.plan.get("aim_cf"), "cf_kind": group.plan.get("cf_kind")},
                "pair": None,
            },
            "request": {"state": state, "questions": questions},
            "labels": by_prior["base"],
        }
        problems = validate_record(record)
        if problems:
            raise ValueError(f"{record['id']}: {problems}")
        built[member] = record
    if group.pair is not None:
        base, pair = built["base"], built["pair"]
        key = names["key"]
        by_prior = {p: (base["meta"]["labels_by_prior"][p][key]["p_yes"], pair["meta"]["labels_by_prior"][p][key]["p_yes"]) for p in fam.PRIORS}
        directions = {p: gen.direction(b - a) for p, (a, b) in by_prior.items()}
        common = {"kind": "numeric", "field": group.pair["field"], "from": group.pair["from"], "to": group.pair["value"],
                  "closeness": group.pair["closeness"], "factor": group.pair.get("factor"), "fact": group.pair.get("fact", "number"),
                  "key_by_prior": by_prior, "key_delta": by_prior["base"][1] - by_prior["base"][0],
                  "direction_by_prior": directions, "direction_same": len(set(directions.values())) == 1,
                  "no_reversal": not {"up", "down"} <= set(directions.values())}
        base["meta"]["pair"] = {"of": pair["id"], "role": "base", **common}
        pair["meta"]["pair"] = {"of": base["id"], "role": "variant", **common}
    return [built[m] for m in members]


# --------------------------------------------------------------------------
# Balanced selection
# --------------------------------------------------------------------------


def select_balanced(items: Iterable[tuple[fam.Group, dict[str, Any]]], targets: dict[str, int], *, bins: int = 10) -> tuple[list[tuple[fam.Group, dict[str, Any]]], dict[str, Any]]:
    """Per scene keep `targets[scene]` groups so the base record's key label (base prior) is spread evenly over `bins` bins of [0, 1]
    (candidate order inside a bin; a short bin gives its slots to the others in turn). Groups with a rollout flagged unstable are dropped
    (v0.3b's :func:`robo_jev.mjgen.generate.select_balanced`)."""
    by_scene: dict[str, list[tuple[fam.Group, dict[str, Any]]]] = collections.defaultdict(list)
    dropped: collections.Counter[str] = collections.Counter()
    for group, labelled in items:
        if labelled["flags"]["unstable"]:
            dropped[group.family] += 1
            continue
        by_scene[group.family].append((group, labelled))
    kept: list[tuple[fam.Group, dict[str, Any]]] = []
    report: dict[str, Any] = {}
    for name, total in targets.items():
        scene = spec(name)
        pool = sorted(by_scene.get(name, []), key=lambda gl: gl[0].index)
        binned: dict[int, list[tuple[fam.Group, dict[str, Any]]]] = collections.defaultdict(list)
        for group, labelled in pool:
            events = _events(scene, labelled["members"]["base"]["variants"]["planned"]["outcomes"]["base"])
            binned[min(bins - 1, int(np.mean(events) * bins))].append((group, labelled))
        share = {b: total // bins + (1 if b < total % bins else 0) for b in range(bins)}
        taken = {b: binned[b][: share[b]] for b in range(bins)}
        missing = total - sum(len(v) for v in taken.values())
        while missing > 0:
            spare = [b for b in range(bins) if len(binned[b]) > len(taken[b])]
            if not spare:
                break
            for b in spare:
                if missing <= 0:
                    break
                taken[b].append(binned[b][len(taken[b])])
                missing -= 1
        chosen = sorted((gl for b in range(bins) for gl in taken[b]), key=lambda gl: gl[0].index)
        kept.extend(chosen)
        report[name] = {"candidates": len(pool) + dropped[name], "dropped_unstable": dropped[name], "pool_by_bin": [len(binned[b]) for b in range(bins)],
                        "kept_by_bin": [len(taken[b]) for b in range(bins)], "target": total, "kept": len(chosen), "shortfall": max(0, total - len(chosen))}
    return kept, report


# --------------------------------------------------------------------------
# Statistics
# --------------------------------------------------------------------------


def _band_profile(values: Sequence[float]) -> dict[str, Any]:
    bands = collections.Counter(gen.band(v) for v in values)
    n = len(values)
    return {"n": n, "low": bands["low"], "middle": bands["middle"], "high": bands["high"],
            "share": {b: round(bands[b] / n, 4) for b in ("low", "middle", "high")} if n else None,
            "exact_0_or_1": sum(1 for v in values if v <= 0.0 or v >= 1.0)}


def stats(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Per scene, per law and pooled: composition, key-label and all yes/no + score histograms and band profiles, robust share by question
    kind, in-record contrasts, pair moves (by design and by factor, with direction agreement across priors) and styles."""
    out: dict[str, Any] = {"records": len(records), "by_scene": {}, "by_law": {}}
    scenes = sorted({r["meta"]["scene"] for r in records})
    laws = sorted({r["meta"]["law"] for r in records})
    groups = [("by_scene", s, lambda r, s=s: r["meta"]["scene"] == s) for s in scenes]
    groups += [("by_law", law, lambda r, law=law: r["meta"]["law"] == law) for law in laws]
    groups += [("by_law", "all", lambda r: True)]
    for section, name, keep in groups:
        rows = [r for r in records if keep(r)]
        if not rows:
            continue
        key_values = [r["labels"][r["meta"]["key_question"]]["p_yes"] for r in rows]
        binary = [float(l.get("p_yes", l.get("value"))) for r in rows for l in r["labels"].values() if l["type"] in ("noul", "score")]
        yesno = [float(l["p_yes"]) for r in rows for l in r["labels"].values() if l["type"] == "noul"]
        score = [float(l["value"]) for r in rows for l in r["labels"].values() if l["type"] == "score"]
        q_types = collections.Counter(q["type"] for r in rows for q in r["request"]["questions"].values())
        q_kinds = collections.Counter(r["meta"]["question_kind"][q] for r in rows for q in r["request"]["questions"])
        robust_by_kind: dict[str, list[int]] = collections.defaultdict(lambda: [0, 0])
        for r in rows:
            for q, entry in r["meta"]["robustness"].items():
                kind = r["meta"]["question_kind"][q]
                robust_by_kind[kind][0] += int(entry["robust"])
                robust_by_kind[kind][1] += 1
        inside = []
        for r in rows:
            sc = spec(r["meta"]["scene"])
            planned = r["meta"]["outcomes"]["planned"]["base"]
            cf = r["meta"]["outcomes"]["cf"]["base"]
            inside.append(sum(cf[o] for o in sc.event) - sum(planned[o] for o in sc.event))
        pairs = [r for r in rows if r["meta"]["pair"] and r["meta"]["pair"]["role"] == "base"]
        deltas = [abs(r["meta"]["pair"]["key_delta"]) for r in pairs]
        by_design: dict[str, list[float]] = collections.defaultdict(list)
        by_factor: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
        by_field: dict[str, list[float]] = collections.defaultdict(list)
        for r in pairs:
            p = r["meta"]["pair"]
            by_design[p["closeness"]].append(abs(p["key_delta"]))
            if p.get("factor"):
                by_factor[f'{r["meta"]["scene"]}/{p["factor"]}'].append(p)
            else:
                by_field[f'{r["meta"]["scene"]}/{p["field"]}'].append(abs(p["key_delta"]))
        outcome_answers: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
        for r in rows:
            for q, lab in r["labels"].items():
                kind = r["meta"]["question_kind"][q]
                if lab["type"] == "choice" and kind in ("outcome", "action", "compare", "direction"):
                    probs = lab["probabilities"]
                    outcome_answers[kind][max(probs, key=lambda k: (probs[k], k))] += 1
        out[section][name] = {
            "records": len(rows), "groups": len({(r["meta"]["scene"], r["generator"]["group"]) for r in rows}),
            "questions": sum(q_types.values()), "question_types": dict(sorted(q_types.items())), "question_kinds": dict(sorted(q_kinds.items())),
            "key_label_hist10": gen._hist(key_values), "key_label_mean": statistics.fmean(key_values),
            "key_band_profile": _band_profile(key_values), "yes_no_band_profile": _band_profile(yesno), "score_band_profile": _band_profile(score),
            "binary_label_hist10": gen._hist(binary), "uncertain_share": sum(1 for v in binary if 0.1 <= v <= 0.9) / len(binary) if binary else None,
            "styles": dict(sorted(collections.Counter(r["meta"]["style"] for r in rows).items())),
            "argmax_answers": {k: dict(sorted(v.items())) for k, v in sorted(outcome_answers.items())},
            "contrast_inside": {"abs_delta_ge_0.1": sum(1 for d in inside if abs(d) >= 0.1) / len(inside), "abs_delta_hist10": gen._hist([abs(d) for d in inside]),
                                "direction_same_all_priors": sum(1 for r in rows if r["meta"]["contrast_direction_same"]) / len(rows),
                                "no_reversal_all_priors": sum(1 for r in rows if r["meta"]["contrast_no_reversal"]) / len(rows)},
            "pairs": {"pairs": len(pairs), "records_in_pairs": sum(1 for r in rows if r["meta"]["pair"]),
                      "share_of_records": sum(1 for r in rows if r["meta"]["pair"]) / len(rows),
                      "factor_pair_share": (sum(1 for r in pairs if r["meta"]["pair"]["closeness"] == "factor") / len(pairs)) if pairs else None,
                      "abs_delta_ge_0.1": (sum(1 for d in deltas if d >= 0.1) / len(deltas)) if deltas else None,
                      "abs_delta_median": statistics.median(deltas) if deltas else None, "abs_delta_hist10": gen._hist(deltas),
                      "by_design": {f: {"pairs": len(v), "abs_delta_ge_0.1": sum(1 for d in v if d >= 0.1) / len(v), "median": statistics.median(v)}
                                    for f, v in sorted(by_design.items())},
                      "by_number_field": {f: {"pairs": len(v), "abs_delta_ge_0.1": sum(1 for d in v if d >= 0.1) / len(v), "median": statistics.median(v)}
                                          for f, v in sorted(by_field.items())},
                      "by_factor": {f: gen._factor_summary(v) for f, v in sorted(by_factor.items())},
                      "direction_same_all_priors": (sum(1 for r in pairs if r["meta"]["pair"]["direction_same"]) / len(pairs)) if pairs else None,
                      "no_reversal_all_priors": (sum(1 for r in pairs if r["meta"]["pair"]["no_reversal"]) / len(pairs)) if pairs else None},
            "robust": {"questions": sum(v[1] for v in robust_by_kind.values()), "robust": sum(v[0] for v in robust_by_kind.values()),
                       "share": sum(v[0] for v in robust_by_kind.values()) / max(1, sum(v[1] for v in robust_by_kind.values())),
                       "by_kind": {k: {"questions": v[1], "robust": v[0], "share": v[0] / v[1]} for k, v in sorted(robust_by_kind.items())}},
            "state_chars_median": statistics.median(len(r["request"]["state"]) for r in rows),
        }
    return out
