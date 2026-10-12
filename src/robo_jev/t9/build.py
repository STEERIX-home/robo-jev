"""T9 build — candidate groups, MuJoCo labels under three priors, records, robustness, pairs and balanced selection (Task T9 §1).

**Group.** One candidate group = a base moment and its **one-fact partner** (one stated number or word changed), sharing everything
else: the same sentence variants (text seed), the same question changes (``cf``, the gentler ``alt`` action, the ``dirv`` change) and
the same hidden draws. :func:`candidate` builds it with the family's textbook calculator only:

1. sample a visible moment; aim its decisive number so the calculator's base-prior key rate hits a target (targets stratified over
   tenths: group i aims at a uniform draw inside tenth ``i mod 10``);
2. partner: a factor of the family's list moved so the calculator's key rate moves by 0.2–0.55 (words: the alternative that moves
   it most, or the item's twin — the same item with other contents);
3. ``cf``: another stated number, aimed at a fresh uniform target; ``alt``: the gentler action (a ratio of the plan's number, or the
   item set further in); ``dirv``: by group index the intended answer cycles through about-the-same (a fact that should not matter,
   ×2 or ×½), likelier and less likely (a number moved 0.2–0.5 in that direction, or the word that moves it most);
4. costs, drawn from their own seeded stream (:func:`draw_costs`), independent of every label: a failure costs log-uniform 10–300 s
   of delay, the gentler action adds 5–30 s, the safe step 5–40 s.

**Labels** (:func:`label_group`): every variant (plan, cf, alt, dirv) of both moments under every prior (base, low_reader,
high_reader; :mod:`robo_jev.t9.priors`) on the group's K shared draws → outcome frequencies. A group with any unstable rollout is
dropped.

**Records** (:func:`records_of`): labels from the base prior —

* yes/no and score: the key event's share (``key`` under the plan, ``cf``, ``alt``);
* ``dir`` (``kind: state``): Δ = key share(dirv) − key share(plan); one-hot ``likelier`` if Δ ≥ +0.10, ``less_likely`` if Δ ≤ −0.10,
  else ``no_real_change``;
* ``act`` (``kind: action``): expected cost = share × failure cost + added seconds (plan 0, gentler ``alt`` s, safe step ``secure``
  s with no risk); probability 1 on the cheapest option (ties split); ``expected_cost`` stored;
* ``outcome`` (``kind: state``): the plan's outcome shares.

``meta.labels_by_prior[prior][question]`` holds the same labels under each prior; ``meta.robustness[question]`` the values / bands or
argmax per prior and ``robust`` (band or argmax identical under all three). ``meta.pair`` links the two records of a group (``kind:
numeric`` — a one-fact pair, as T5), with the key share per prior and whether its direction holds under all priors.

**Selection** (:func:`select`): per family, groups with a stable rollout set, equal numbers per tenth of the base record's key label
(base prior), the rest filled from the nearest tenths.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from typing import Any, Sequence

from robo_jev.t9 import physics as ph
from robo_jev.t9 import priors as P
from robo_jev.t9 import text as T
from robo_jev.t9.families import FAMILIES, Change, Factor, Family

__all__ = ["GENERATOR", "candidate", "group_seed", "label_group", "records_of", "select", "variant_vis"]

GENERATOR = "t9-mujoco-v1"
DEFAULT_K = 40
DIR_MIN = 0.10
VARIANTS = ("plan", "cf", "alt", "dirv")
DIR_INTENTS = ("no_real_change", "likelier", "less_likely")


def group_seed(master: int, family_name: str, index: int) -> int:
    digest = hashlib.sha256(f"{master}|{family_name}|{index}".encode()).hexdigest()
    return int(digest[:12], 16)


def _apply(fam: Family, vis: dict[str, Any], change: Change | None) -> dict[str, Any] | None:
    if change is None:
        return dict(vis)
    if change.kind == "word":
        return fam.apply_word(vis, change.field, change.value)
    out = dict(vis)
    out[change.field] = change.value
    return out


def variant_vis(fam: Family, vis: dict[str, Any], spec: dict[str, Any], variant: str) -> dict[str, Any]:
    """The visible configuration of one question variant of a moment."""
    if variant == "plan":
        return dict(vis)
    if variant == "alt":
        out = dict(vis)
        out[fam.alt_field] = spec["alt"]
        return out
    change = spec[variant]
    out = _apply(fam, vis, change)
    if out is None:
        raise ValueError(f"{fam.name}: change {change} does not apply")
    return out


def _change_from(data: dict[str, Any] | None) -> Change | None:
    return None if data is None else Change(data["field"], data["value"], data["kind"])


def _change_json(change: Change | None) -> dict[str, Any] | None:
    return None if change is None else {"field": change.field, "value": change.value, "kind": change.kind}


class _Draws:
    """Base-prior draws of any variant of a group (the group's shared unit draws mapped through the variant's word ranges)."""

    def __init__(self, fam: Family, units: Any):
        self.fam, self.units, self.names = fam, units, fam.hidden_names()

    def __call__(self, vis: dict[str, Any], prior: str = "base") -> list[dict[str, float]]:
        return P.draws_for(self.fam.words(vis), self.fam.fixed(vis), prior, self.units, self.names)


def _move_number(fam: Family, vis: dict[str, Any], factor: Factor, target: float, draws: _Draws) -> Change | None:
    if factor.scale:
        return None
    lo, hi = factor.lo, factor.hi
    if factor.field == "lip":
        hi = min(hi, 0.8 * vis["radius"])
    if factor.field == "height" and fam.name == "knock":
        from robo_jev.t9.families import KNOCK_OBJECTS

        hi = min(hi, 0.95 * KNOCK_OBJECTS[vis["object"]]["dims"][2])
    if factor.field == "over" and fam.name == "overhang":
        hi = min(hi, 0.48 * vis["board_len"])
    if hi <= lo:
        return None
    moved = fam.aim(vis, factor.field, lo, hi, factor.sign, target, draws, digits=factor.digits)
    value = moved[factor.field]
    if value == vis[factor.field] or not fam.valid(moved):
        return None
    return Change(factor.field, value, "num")


def _move_word(fam: Family, vis: dict[str, Any], factor: Factor, direction: int, draws: _Draws, base_rate: float) -> tuple[Change, float] | None:
    best = None
    for word in fam.word_choices(factor.field):
        if word == vis.get(factor.field):
            continue
        moved = fam.apply_word(vis, factor.field, word)
        if moved is None or not fam.valid(moved):
            continue
        delta = fam.rate(moved, draws(moved)) - base_rate
        score = delta * direction if direction else abs(delta)
        if best is None or score > best[1]:
            best = (Change(factor.field, word, "word"), score, delta)
    return None if best is None else (best[0], best[2])


#: Action costs (seconds of delay), drawn per group from their own seeded stream — independent of every label: a failure costs
#: log-uniform 10–300 s, the gentler action adds 5–30 s, the safe step 5–40 s (whole seconds). The ranges were chosen once so the
#: cheapest option splits about 42 / 26 / 32 % between plan, gentler action and safe step on T9's shares.
COST_FAIL = (10.0, 300.0)
COST_ALT = (5, 30)
COST_SECURE = (5, 40)


def draw_costs(seed: int) -> dict[str, float]:
    rng = random.Random(int(seed) ^ 0xC057)
    return {"fail": float(round(math.exp(rng.uniform(math.log(COST_FAIL[0]), math.log(COST_FAIL[1]))))), "alt": float(rng.randint(*COST_ALT)),
            "secure": float(rng.randint(*COST_SECURE))}


def candidate(family_name: str, index: int, master: int, k: int = DEFAULT_K, tries: int = 40, reasons: Counter | None = None) -> dict[str, Any]:
    """One candidate group (module docstring), built with the calculator only (no MuJoCo). `reasons` counts why tries failed."""
    reasons = reasons if reasons is not None else Counter()
    fam = FAMILIES[family_name]
    seed = group_seed(master, family_name, index)
    rng = random.Random(seed)
    draws_seed = seed ^ 0x7A9C
    units = P.unit_draws(fam.hidden_names(), k, draws_seed)
    draws = _Draws(fam, units)
    target = (index % 10 + rng.random()) / 10
    for attempt in range(tries):
        vis = fam.sample(rng)
        field_name, lo, hi, sign = fam.aim_spec(vis)
        if field_name == "inset":
            hi = min(hi, vis["over"])
        vis = fam.aim(vis, field_name, lo, hi, sign, target, draws, digits=3 if field_name not in ("tilt", "rate") else 1)
        if not fam.valid(vis):
            reasons["invalid after aim"] += 1
            continue
        base_rate = fam.rate(vis, draws(vis))
        if abs(base_rate - target) > 0.2:
            reasons["target out of reach"] += 1
            continue
        # partner: one fact moved by 0.2–0.55 (or the word that moves it most)
        partner = None
        factors = [f for f in fam.partner_factors if f.field not in (fam.alt_field, field_name)]
        rng.shuffle(factors)
        for factor in factors:
            if factor.kind == "num":
                up = base_rate < 0.35 or (base_rate <= 0.65 and rng.random() < 0.5)
                goal = min(0.98, base_rate + rng.uniform(0.2, 0.55)) if up else max(0.02, base_rate - rng.uniform(0.2, 0.55))
                change = _move_number(fam, vis, factor, goal, draws)
                if change is None:
                    continue
                moved = _apply(fam, vis, change)
                if abs(fam.rate(moved, draws(moved)) - base_rate) >= 0.12:
                    partner = change
                    break
            else:
                found = _move_word(fam, vis, factor, 0, draws, base_rate)
                if found is not None and abs(found[1]) >= 0.12:
                    partner = found[0]
                    break
        if partner is None:
            reasons["no partner"] += 1
            continue
        taken = {partner.field, fam.alt_field}
        cf_options = [f for f in fam.cf_factors if f.field not in taken]
        rng.shuffle(cf_options)
        # when every other number is taken, the counterfactual may ask about another value of the action's own number
        cf_options += [f for f in fam.cf_factors if f.field == fam.alt_field and f.field != partner.field]
        cf = None
        for factor in cf_options:
            cf = _move_number(fam, vis, factor, rng.random(), draws)
            if cf is not None:
                break
        if cf is None:
            reasons["no cf"] += 1
            continue
        intent = DIR_INTENTS[(index // 10 + index) % 3]
        if intent == "less_likely" and base_rate < 0.25:
            intent = "likelier"
        elif intent == "likelier" and base_rate > 0.75:
            intent = "less_likely"
        dirv = None
        options = [f for f in fam.dir_factors if f.field not in taken | {cf.field, field_name} and vis.get(f.field) is not None]
        rng.shuffle(options)
        base_partner = _apply(fam, vis, partner)
        for factor in options:
            if intent == "no_real_change":
                if factor.sign != 0 or not factor.scale:
                    continue
                ratio = rng.choice(factor.same_ratio)
                value = round(vis[factor.field] * ratio, factor.digits)
                dirv = Change(factor.field, value, "num")
                break
            if factor.sign == 0 and factor.kind == "num":
                continue
            direction = 1 if intent == "likelier" else -1
            if factor.kind == "word":
                found = _move_word(fam, vis, factor, direction, draws, base_rate)
                if found is not None and found[1] * direction >= 0.15:
                    dirv = found[0]
                    break
                continue
            goal = base_rate + direction * rng.uniform(0.2, 0.5)
            if not 0.02 <= goal <= 0.98:
                continue
            change = _move_number(fam, vis, factor, goal, draws)
            if change is None:
                continue
            moved = _apply(fam, vis, change)
            if (fam.rate(moved, draws(moved)) - base_rate) * direction >= 0.12:
                dirv = change
                break
        if dirv is None:
            reasons[f"no dirv ({intent})"] += 1
            continue
        # every change must apply to the partner as well (a word twin or a range limit could refuse it)
        alt_param = fam.alt_param(rng)
        spec = {"cf": cf, "alt": fam.alt_from(vis, alt_param), "dirv": dirv}
        try:
            for moment in (vis, base_partner):
                for variant in VARIANTS:
                    v = variant_vis(fam, moment, spec, variant)
                    if not fam.valid(v) and variant != "alt":
                        raise ValueError("invalid variant")
        except (ValueError, KeyError, TypeError):
            reasons["invalid variant"] += 1
            continue
        costs = draw_costs(seed)
        return {
            "family": family_name, "index": index, "seed": seed, "draws_seed": draws_seed, "k": k, "target": round(target, 4),
            "attempt": attempt, "base": vis, "partner": _change_json(partner), "cf": _change_json(cf), "alt": spec["alt"], "alt_param": alt_param,
            "dirv": _change_json(dirv), "dir_intent": intent, "costs": costs, "aimed": field_name,
            "calc_key": {"base": base_rate, "partner": fam.rate(base_partner, draws(base_partner))}, "failed_tries": dict(reasons),
        }
    raise RuntimeError(f"{family_name} #{index}: no valid candidate in {tries} tries ({dict(reasons)})")


AIM_DRAWS = 16
AIM_STEPS = 7
AIM_TOL = 0.1


def _mujoco_rate(fam: Family, vis: dict[str, Any], draws: Sequence[dict[str, float]], scenes: dict[str, Any]) -> float:
    spec = fam.scene(vis)
    key = json.dumps(spec, sort_keys=True, default=list)
    if key not in scenes:
        scenes[key] = ph.make_scene(fam.name, spec)
    action = fam.action(vis)
    hits = sum(scenes[key].run(action, fam.physics_draw(vis, d))["outcome"] in fam.key for d in draws)
    return hits / len(draws)


def refine(group: dict[str, Any]) -> dict[str, Any]:
    """Move the plan's decisive number so the **MuJoCo** key rate on the first :data:`AIM_DRAWS` base-prior draws of the group comes
    within :data:`AIM_TOL` of the group's target (bisection, at most :data:`AIM_STEPS` rollout batches, between the calculator's value
    and the family's bound on the side the rate must move). The calculator aims first (cheap); this step corrects its bias near the
    boundary. The gentler action is recomputed from its stored parameter; a refinement that makes any variant invalid is dropped."""
    fam = FAMILIES[group["family"]]
    units = P.unit_draws(fam.hidden_names(), int(group["k"]), int(group["draws_seed"]))
    draws_all = _Draws(fam, units)
    vis = dict(group["base"])
    field_name, lo, hi, sign = fam.aim_spec(vis)
    if field_name == "inset":
        hi = min(hi, vis["over"])
    target = float(group["target"])
    scenes: dict[str, Any] = {}

    def rate(v: dict[str, Any]) -> float:
        return _mujoco_rate(fam, v, draws_all(v)[:AIM_DRAWS], scenes)

    start = rate(vis)
    best = (abs(start - target), vis[field_name], start)
    steps = 0
    if best[0] > AIM_TOL:
        up = (start < target) == (sign > 0)
        a, b = (float(vis[field_name]), hi) if up else (lo, float(vis[field_name]))
        digits = 1 if field_name in ("tilt", "rate") else 3
        for _ in range(AIM_STEPS):
            steps += 1
            mid = round((a + b) / 2, digits)
            trial = dict(vis)
            trial[field_name] = mid
            r = rate(trial)
            if abs(r - target) < best[0]:
                best = (abs(r - target), mid, r)
            if best[0] <= AIM_TOL:
                break
            if (r < target) == (sign > 0):
                a = mid
            else:
                b = mid
    out = dict(group)
    out["aim"] = {"field": field_name, "calculator_value": vis[field_name], "value": best[1], "mujoco_rate_start": start, "mujoco_rate": best[2],
                  "draws": AIM_DRAWS, "steps": steps}
    if best[1] != vis[field_name]:
        new = dict(vis)
        new[field_name] = best[1]
        spec = spec_of({**group, "base": new})
        if fam.alt_field == field_name or field_name == "inset" and fam.name == "overhang":
            spec["alt"] = fam.alt_from(new, group["alt_param"])
        try:
            for moment in moments_of({**group, "base": new}).values():
                for variant in VARIANTS:
                    if variant != "alt" and not fam.valid(variant_vis(fam, moment, spec, variant)):
                        raise ValueError("invalid after refinement")
        except (ValueError, KeyError, TypeError):
            out["aim"]["dropped"] = True
            return out
        out["base"] = new
        out["alt"] = spec["alt"]
    return _refine_direction(out, fam, draws_all, scenes)


def _refine_direction(group: dict[str, Any], fam: Family, draws_all: "_Draws", scenes: dict[str, Any]) -> dict[str, Any]:
    """For a numeric ``dirv`` aimed at likelier / less likely: if on the aim draws its MuJoCo move from the plan is under 0.15 the
    intended way, move the number further (bisection toward the factor's bound, at most :data:`AIM_STEPS` batches) until it is; keep
    the original when no value inside the bound gets there."""
    intent, dirv = group["dir_intent"], group["dirv"]
    if intent == "no_real_change" or dirv is None or dirv["kind"] != "num":
        return group
    factor = next((f for f in fam.dir_factors if f.field == dirv["field"]), None)
    if factor is None or factor.sign == 0:
        return group
    vis = dict(group["base"])
    sign = 1 if intent == "likelier" else -1

    def rate(v: dict[str, Any]) -> float:
        return _mujoco_rate(fam, v, draws_all(v)[:AIM_DRAWS], scenes)

    base_rate = rate(vis)
    moved = dict(vis)
    moved[dirv["field"]] = dirv["value"]
    if not fam.valid(moved):
        return group
    now = rate(moved) - base_rate
    record = {"intent": intent, "value_before": dirv["value"], "move_before": now, "steps": 0}
    if now * sign >= 0.15:
        return {**group, "dir_aim": record}
    # the number must move further in the direction that raises (sign·factor.sign > 0) or lowers the share
    up = (factor.sign * sign) > 0
    lo, hi = (float(dirv["value"]), factor.hi) if up else (factor.lo, float(dirv["value"]))
    if dirv["field"] == "lip":
        hi = min(hi, 0.8 * vis["radius"])
    best = None
    for _ in range(AIM_STEPS):
        record["steps"] += 1
        mid = round((lo + hi) / 2, factor.digits)
        trial = dict(vis)
        trial[dirv["field"]] = mid
        if not fam.valid(trial):
            if up:
                hi = mid
            else:
                lo = mid
            continue
        move = rate(trial) - base_rate
        if move * sign >= 0.15:
            best = (mid, move)
            if up:
                hi = mid
            else:
                lo = mid
        elif up:
            lo = mid
        else:
            hi = mid
    if best is None:
        return {**group, "dir_aim": record}
    record.update({"value": best[0], "move": best[1]})
    return {**group, "dirv": {**dirv, "value": best[0]}, "dir_aim": record}


def moments_of(group: dict[str, Any]) -> dict[str, dict[str, Any]]:
    fam = FAMILIES[group["family"]]
    base = dict(group["base"])
    partner = _apply(fam, base, _change_from(group["partner"]))
    return {"a": base, "b": partner}


def spec_of(group: dict[str, Any]) -> dict[str, Any]:
    return {"cf": _change_from(group["cf"]), "alt": group["alt"], "dirv": _change_from(group["dirv"])}


def label_group(group: dict[str, Any]) -> dict[str, Any]:
    """MuJoCo outcomes of every variant of both moments under every prior (module docstring). Returns ``{moment: {variant: {prior:
    {"freq": {outcome: share}, "codes": str}}}}`` (``codes``: one digit per draw, the outcome's index in the family's order), plus
    the number of unstable rollouts and the CPU seconds."""
    import time

    fam = FAMILIES[group["family"]]
    outcomes = ph.OUTCOMES[fam.name]
    units = P.unit_draws(fam.hidden_names(), int(group["k"]), int(group["draws_seed"]))
    draws = _Draws(fam, units)
    spec = spec_of(group)
    scenes: dict[str, ph._Scene] = {}
    out: dict[str, Any] = {"labels": {}, "unstable": 0}
    start = time.process_time()
    for moment_name, vis in moments_of(group).items():
        per_variant: dict[str, Any] = {}
        for variant in VARIANTS:
            v = variant_vis(fam, vis, spec, variant)
            scene_spec = fam.scene(v)
            key = json.dumps(scene_spec, sort_keys=True, default=list)
            if key not in scenes:
                scenes[key] = ph.make_scene(fam.name, scene_spec)
            scene = scenes[key]
            action = fam.action(v)
            per_prior = {}
            for prior in P.PRIORS:
                codes = []
                for draw in draws(v, prior):
                    result = scene.run(action, fam.physics_draw(v, draw))
                    out["unstable"] += int(bool(result.get("unstable")))
                    codes.append(str(outcomes.index(result["outcome"])))
                counts = Counter(codes)
                per_prior[prior] = {"freq": {o: counts.get(str(i), 0) / len(codes) for i, o in enumerate(outcomes)}, "codes": "".join(codes)}
            per_variant[variant] = per_prior
        out["labels"][moment_name] = per_variant
    out["cpu_s"] = round(time.process_time() - start, 3)
    return out


# --------------------------------------------------------------------------
# Records
# --------------------------------------------------------------------------


def _key_share(fam: Family, freq: dict[str, float]) -> float:
    return float(sum(freq[o] for o in fam.key))


def _band(x: float) -> str:
    return "low" if x < 1 / 3 else ("high" if x > 2 / 3 else "middle")


def _dir_class(delta: float) -> str:
    return "likelier" if delta >= DIR_MIN - 1e-12 else ("less_likely" if delta <= -DIR_MIN + 1e-12 else "no_real_change")


def _argmax(probabilities: dict[str, float]) -> list[str]:
    top = max(probabilities.values())
    return sorted(k for k, v in probabilities.items() if abs(v - top) <= 1e-12)


def _labels_for(fam: Family, per_variant: dict[str, Any], prior: str, costs: dict[str, float]) -> dict[str, Any]:
    names = T.NAMES[fam.name]
    share = {variant: _key_share(fam, per_variant[variant][prior]["freq"]) for variant in VARIANTS}
    labels: dict[str, Any] = {
        names["key"]: {"type": "noul", "p_yes": round(share["plan"], 6)},
        names["cf"]: {"type": "score", "value": round(share["cf"], 6)},
        names["alt"]: {"type": "noul", "p_yes": round(share["alt"], 6)},
    }
    cls = _dir_class(share["dirv"] - share["plan"])
    labels[names["dir"]] = {"type": "choice", "kind": "state", "probabilities": {k: (1.0 if k == cls else 0.0) for k in T.DIR_OPTIONS},
                            "delta": round(share["dirv"] - share["plan"], 6)}
    plan_key, alt_key, safe_key = T.ACT_OPTIONS[fam.name]
    expected = {plan_key: share["plan"] * costs["fail"], alt_key: share["alt"] * costs["fail"] + costs["alt"], safe_key: costs["secure"]}
    expected = {k: round(v, 6) for k, v in expected.items()}
    best = _argmax({k: -v for k, v in expected.items()})
    labels[names["act"]] = {"type": "choice", "kind": "action", "probabilities": {k: (1.0 / len(best) if k in best else 0.0) for k in expected},
                            "expected_cost": expected}
    freq = per_variant["plan"][prior]["freq"]
    options = T.OUTCOME_OPTIONS[fam.name]
    probs = {key: round(freq[o], 6) for key, o in zip(options, ph.OUTCOMES[fam.name])}
    total = sum(probs.values())
    probs = {k: v / total for k, v in probs.items()}
    labels[names["outcome"]] = {"type": "choice", "kind": "state", "probabilities": probs}
    return labels


def _robustness(by_prior: dict[str, dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name, label in by_prior["base"].items():
        if label["type"] in ("noul", "score"):
            key = "p_yes" if label["type"] == "noul" else "value"
            values = {p: by_prior[p][name][key] for p in P.PRIORS}
            bands = {p: _band(v) for p, v in values.items()}
            out[name] = {"by_prior": values, "min": min(values.values()), "max": max(values.values()), "bands": bands,
                         "band_same": len(set(bands.values())) == 1}
            out[name]["robust"] = out[name]["band_same"]
        else:
            tops = {p: _argmax(by_prior[p][name]["probabilities"]) if label.get("kind") != "action"
                    else _argmax({k: -v for k, v in by_prior[p][name]["expected_cost"].items()}) for p in P.PRIORS}
            same = len({tuple(v) for v in tops.values()}) == 1
            out[name] = {"argmax": tops, "argmax_same": same, "robust": same}
    return out


def records_of(group: dict[str, Any], labelled: dict[str, Any], *, mujoco_version: str, commit: str | None = None) -> list[dict[str, Any]]:
    """The two records of a labelled group (module docstring)."""
    fam = FAMILIES[group["family"]]
    names = T.NAMES[fam.name]
    spec = spec_of(group)
    moments = moments_of(group)
    ids = {m: f"t9-{fam.name}-{int(group['index']):04d}{m}" for m in moments}
    costs = draw_costs(int(group["seed"]))
    by_prior_all = {m: {p: _labels_for(fam, labelled["labels"][m], p, costs) for p in P.PRIORS} for m in moments}
    key = names["key"]
    shares = {m: {p: by_prior_all[m][p][key]["p_yes"] for p in P.PRIORS} for m in moments}
    moves = {p: shares["b"][p] - shares["a"][p] for p in P.PRIORS}
    direction = {p: ("up" if d > 0.05 else ("down" if d < -0.05 else "flat")) for p, d in moves.items()}
    partner = _change_from(group["partner"])
    out = []
    for m, vis in moments.items():
        state, questions = T.render(fam.name, vis, {"cf": spec["cf"], "alt": spec["alt"], "dirv": spec["dirv"]}, costs, int(group["seed"]))
        by_prior = by_prior_all[m]
        units_ranges = {p: P.prior_ranges(fam.words(vis), p) for p in P.PRIORS}
        meta = {
            "set": "T9", "scene": fam.name, "family": fam.name, "law": fam.law, "law_group": fam.group,
            "events": [names["key"], names["cf"], names["alt"]], "outcome_question": names["outcome"], "direction_question": names["dir"],
            "action_question": names["act"], "criteria": {names["act"]: "least expected cost: failure share x stated cost + added seconds"},
            "question_variant": {names["key"]: "plan", names["cf"]: "cf", names["alt"]: "alt", names["dir"]: "dirv", names["act"]: "plan+alt",
                                 names["outcome"]: "plan"},
            "visible": vis, "changes": {"cf": _change_json(spec["cf"]), "alt": {"field": fam.alt_field, "value": spec["alt"]},
                                        "dirv": _change_json(spec["dirv"]), "dir_intent": group["dir_intent"]},
            "costs": costs, "hidden_prior": units_ranges, "fixed_hidden": fam.fixed(vis), "hidden_names": list(fam.hidden_names()),
            "outcome_names": list(ph.OUTCOMES[fam.name]), "key_outcomes": list(fam.key),
            "outcomes": {variant: {p: labelled["labels"][m][variant][p]["freq"] for p in P.PRIORS} for variant in VARIANTS},
            "labels_by_prior": by_prior, "robustness": _robustness(by_prior),
            "pair": {"of": ids["b" if m == "a" else "a"], "kind": "numeric", "role": "base" if m == "a" else "variant",
                     "field": partner.field, "from": moments["a"][partner.field], "to": moments["b"][partner.field], "factor_kind": partner.kind,
                     "key_by_prior": {p: [shares["a"][p], shares["b"][p]] for p in P.PRIORS}, "key_delta": round(moves["base"], 6),
                     "direction_by_prior": direction, "direction_same": len(set(direction.values())) == 1,
                     "no_reversal": not ({"up", "down"} <= set(direction.values()))},
            "target": group["target"], "aimed": group["aimed"], "calc_key": group["calc_key"].get("base" if m == "a" else "partner"),
        }
        out.append({
            "schema": "simgen-request/0", "id": ids[m], "request": {"state": state, "questions": questions},
            "labels": by_prior["base"], "meta": meta,
            "generator": {"name": GENERATOR, "commit": commit, "mujoco": mujoco_version, "group": int(group["index"]), "seed": int(group["seed"]),
                          "draws_seed": int(group["draws_seed"]), "k": int(group["k"]), "priors": list(P.PRIORS)},
        })
    return out


# --------------------------------------------------------------------------
# Selection
# --------------------------------------------------------------------------


def select(groups: Sequence[dict[str, Any]], per_family: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Equal numbers of groups per tenth of the base record's key label (base prior) in each family; short tenths are filled from
    the nearest tenths (ties: the lower tenth first), then by group index. `groups` carry ``family``, ``index``, ``key_a`` and
    ``unstable``."""
    by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for g in groups:
        if not g.get("unstable"):
            by_family[g["family"]].append(g)
    chosen: list[dict[str, Any]] = []
    report: dict[str, Any] = {}
    for fam_name, pool in sorted(by_family.items()):
        bins: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for g in sorted(pool, key=lambda g: g["index"]):
            bins[min(9, int(g["key_a"] * 10))].append(g)
        quota = {b: per_family // 10 + (1 if b < per_family % 10 else 0) for b in range(10)}
        taken: list[dict[str, Any]] = []
        for b in range(10):
            taken += bins[b][: quota[b]]
        left = per_family - len(taken)
        if left > 0:
            spare = []
            for b in range(10):
                spare += [(min(abs(b - c) for c in range(10) if len(bins[c]) < quota[c]) if any(len(bins[c]) < quota[c] for c in range(10)) else 0, b, g)
                          for g in bins[b][quota[b]:]]
            spare.sort(key=lambda t: (t[0], t[1], t[2]["index"]))
            taken += [g for _, _, g in spare[:left]]
        chosen += taken
        report[fam_name] = {"pool": len(pool), "kept": len(taken), "pool_by_tenth": [len(bins[b]) for b in range(10)],
                            "kept_by_tenth": [sum(1 for g in taken if min(9, int(g["key_a"] * 10)) == b) for b in range(10)]}
    return chosen, report
