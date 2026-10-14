"""A3 generator — plan groups, label them in MuJoCo, assemble ``simgen-request/0`` records (set ``v06-a3``, split ``train``).

**Group** (:func:`plan_group`, deterministic in (scene, group index)): one base record and its partner (``…p``).

1. Draw K = 32 shared uniform points for the scene's hidden quantities (``draws_seed``) — every prior, variant and both pair members
   map the same points through their own ranges (:mod:`robo_jev.authors.a3.catalog`).
2. Draw a target key share t ~ U(0, 1) and states until the analytic aim (:meth:`robo_jev.authors.a3.scenes.Scene.aim`) puts the
   decisive stated number at t — the key labels then spread flat over 0–1 (MuJoCo gives the label; the aim is stored as ``static_aim``).
3. Pair design: **one-factor** (2/3) — exactly one stated fact changes (a material / condition / contents / model / method word, or one
   number: a non-decisive number or a near change of the decisive one), drawn from the scene's catalog weights, down-weighted × 0.2
   when the aimed partner share would sit outside [0.05, 0.95]; **far-number** (1/3) — the decisive number moves to a value whose
   aimed key share is ≥ 0.45 away.
4. Counterfactual value of the decisive number (shared by both records): **near** (40 %, a small change) or **far** (60 %, aimed ≥ 0.3
   away); it never equals either record's own value.
5. Two alternatives valid for both records (half the groups get one written as riskier), one of them compared with the plan.

**Labelling** (:func:`label_group`): every distinct state of the group (planned, counterfactual and the two options, per record) ×
3 priors × 32 draws in MuJoCo; outcome codes per draw are kept. **Assembly** (:func:`assemble_group`) builds the two records.
"""

from __future__ import annotations

import hashlib
import json
import math
import multiprocessing as mp
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import mujoco
import numpy as np

from robo_jev.authors.a3 import catalog as cat
from robo_jev.authors.a3 import labels as lab
from robo_jev.authors.a3 import text
from robo_jev.authors.a3.scenes import SCENES, Scene

__all__ = ["GENERATOR", "K", "SET", "assemble_group", "label_group", "plan_group", "plan_scene", "state_id"]

SET = "v06-a3"
GENERATOR = "v06-a3-mujoco-v1"
AUTHOR = "a3"
K = 32
MASTER_SEED = 20261014
SCENE_INDEX = {"indexdial": 1, "mayostand": 2, "holddown": 3}
ONE_FACTOR_SHARE = 2.0 / 3.0
FAR_PAIR_GAP = 0.45
FAR_CF_GAP = 0.30
NEAR_CF_SHARE = 0.4
#: A state is kept only when its aimed key share is this close to the drawn target (the far partner: + 0.03).
AIM_TOLERANCE = 0.12
#: Weight of a one-factor change whose aimed partner share falls outside [0.05, 0.95] (relative to the catalog weight).
EXTREME_PARTNER_WEIGHT = 0.2
QUESTION_KIND_LABEL = {"key": "key", "severe": "severe", "score_cf": "score_cf", "direction": "direction", "compare": "compare",
                       "action": "action", "outcome": "outcome"}


def state_id(state: Mapping[str, Any]) -> str:
    return hashlib.sha256(json.dumps(state, sort_keys=True).encode("utf-8")).hexdigest()[:16]


def _rng(scene: str, group: int, salt: int = 0) -> np.random.Generator:
    return np.random.default_rng([MASTER_SEED, SCENE_INDEX[scene], int(group), int(salt)])


def _choose(rng: np.random.Generator, items: Sequence[Any], weights: Sequence[float]) -> Any:
    w = np.asarray(weights, dtype=float)
    return items[int(rng.choice(len(items), p=w / w.sum()))]


def _far_target(rng: np.random.Generator, t: float, gap: float) -> float:
    low, high = max(0.0, t - gap), max(0.0, (1.0 - gap) - t)
    if low + high <= 0:
        return 0.0 if t > 0.5 else 1.0
    x = rng.uniform(0.0, low + high)
    return x if x < low else t + gap + (x - low)


def _near_cf(sc: Scene, rng: np.random.Generator, value: float) -> float:
    sign = 1.0 if rng.random() < 0.5 else -1.0
    if sc.name == "mayostand":
        return value + sign * rng.uniform(20, 50)
    if sc.name == "indexdial":
        return value * math.exp(sign * rng.uniform(0.10, 0.25))
    return value * math.exp(sign * rng.uniform(0.08, 0.20))


def plan_group(scene_name: str, group: int) -> dict[str, Any]:
    """The group's plan (module docstring) — pure function of (scene, group)."""
    sc = SCENES[scene_name]
    rng = _rng(scene_name, group)
    seed = int(rng.integers(2**62))
    draws_seed = int(rng.integers(2**62))
    points = cat.uniform_points(draws_seed, K, sc.hidden_names)
    design = "one_factor" if rng.random() < ONE_FACTOR_SHARE else "far_number"
    target = float(rng.uniform(0.0, 1.0))
    for attempt in range(600):
        if attempt and attempt % 150 == 0:
            target = float(rng.uniform(0.0, 1.0))
        state = sc.sample_state(rng)
        aimed = sc.aim(state, points, target)
        if aimed is None or abs(aimed[1] - target) > AIM_TOLERANCE:
            continue
        base = sc.with_value(state, aimed[0])
        lo, hi = sc.decisive_bounds(base)
        # --- partner ---
        if design == "far_number":
            t2 = _far_target(rng, target, FAR_PAIR_GAP)
            aimed2 = sc.aim(base, points, t2)
            if (aimed2 is None or abs(aimed2[1] - t2) > AIM_TOLERANCE + 0.03 or abs(aimed2[0] - aimed[0]) < 2 * sc.step - 1e-9
                    or abs(aimed2[1] - aimed[1]) < 0.3):
                continue
            pair = sc.with_value(base, aimed2[0])
            field, fact, factor = sc.decisive, "number", None
        else:
            changes = sc.one_factor_changes(base, rng)
            if not changes:
                continue
            # keep partners off the 0 / 1 ends when the scene allows it (the base is aimed flat; a one-fact change of a steep
            # scene would otherwise land most partners on an end)
            weights = []
            for c_field, c_value, _, c_weight in changes:
                p_partner = sc.proxy(sc.apply_change(base, c_field, c_value), points)
                weights.append(c_weight * (1.0 if 0.05 <= p_partner <= 0.95 else EXTREME_PARTNER_WEIGHT))
            field, value, fact, _ = _choose(rng, changes, weights)
            pair = sc.apply_change(base, field, value)
            factor = field
        if sc.name == "mayostand" and not (lo <= pair[sc.decisive] <= hi):
            continue
        # --- counterfactual value of the decisive number (shared) ---
        own = {round(base[sc.decisive], 6), round(pair[sc.decisive], 6)}
        cf_kind = "near" if rng.random() < NEAR_CF_SHARE else "far"
        cf_value = None
        for _ in range(12):
            if cf_kind == "far":
                aimed_cf = sc.aim(base, points, _far_target(rng, aimed[1], FAR_CF_GAP))
                candidate = aimed_cf[0] if aimed_cf else _near_cf(sc, rng, base[sc.decisive])
            else:
                candidate = _near_cf(sc, rng, base[sc.decisive])
            candidate = min(max(round(round(candidate / sc.step) * sc.step, 6), lo), hi)
            if all(abs(candidate - v) >= sc.step - 1e-9 for v in own):
                cf_value = candidate
                break
        if cf_value is None:
            continue
        # --- alternatives valid for both records ---
        valid = [a for a in sc.alternatives() if a.valid(base) and a.valid(pair)]
        if len(valid) < 2:
            continue
        risky = [a for a in valid if a.riskier]
        safe = [a for a in valid if not a.riskier]
        if risky and safe and rng.random() < 0.5:
            chosen = [risky[int(rng.integers(len(risky)))], safe[int(rng.integers(len(safe)))]]
        else:
            idx = rng.permutation(len(valid))[:2]
            chosen = [valid[int(i)] for i in idx]
        order = rng.permutation(2)
        chosen = [chosen[int(i)] for i in order]
        compare = chosen[int(rng.integers(2))]
        style = text.pick_style(scene_name, rng)
        from_value, to_value = sc.changed_value(base, field), sc.changed_value(pair, field)
        cf_base = sc.with_value(base, cf_value)
        cf_pair = sc.with_value(pair, cf_value)
        return {
            "scene": scene_name, "group": int(group), "seed": seed, "draws_seed": draws_seed, "attempts": attempt + 1,
            "design": design, "field": field, "fact": fact, "factor": factor, "from": from_value, "to": to_value,
            "target": target, "base_state": base, "pair_state": pair,
            "aim": {"base": sc.proxy(base, points), "pair": sc.proxy(pair, points), "cf_base": sc.proxy(cf_base, points), "cf_pair": sc.proxy(cf_pair, points)},
            "cf": {"value": cf_value, "kind": cf_kind}, "options": [a.key for a in chosen], "compare_option": compare.key, "style": style,
        }
    raise RuntimeError(f"{scene_name} group {group}: no plan after 600 attempts")


def _plan_worker(args: tuple[str, int]) -> dict[str, Any]:
    return plan_group(*args)


def plan_scene(scene_name: str, groups: int, *, workers: int = 1) -> list[dict[str, Any]]:
    """Plans of groups 0 … `groups` − 1 (each a pure function of (scene, group), so the worker count does not change them)."""
    jobs = [(scene_name, g) for g in range(groups)]
    if workers <= 1:
        return [plan_group(*job) for job in jobs]
    with mp.get_context("fork").Pool(int(workers)) as pool:
        return pool.map(_plan_worker, jobs, chunksize=4)


def member_variants(sc: Scene, plan: Mapping[str, Any], member: str) -> dict[str, dict[str, Any]]:
    """Visible overrides per variant of one record (``planned``, ``cf``, ``opt_<key>``)."""
    state = plan["base_state"] if member == "base" else plan["pair_state"]
    alternatives = {a.key: a for a in sc.alternatives()}
    out: dict[str, dict[str, Any]] = {"planned": {}, "cf": {sc.decisive: plan["cf"]["value"]}}
    for key in plan["options"]:
        out[f"opt_{key}"] = alternatives[key].apply(state)
    return out


def member_states(sc: Scene, plan: Mapping[str, Any], member: str) -> dict[str, dict[str, Any]]:
    state = plan["base_state"] if member == "base" else plan["pair_state"]
    out = {}
    for variant, overrides in member_variants(sc, plan, member).items():
        full = dict(state)
        full.update(overrides)
        out[variant] = full
    return out


def label_group(plan: Mapping[str, Any]) -> dict[str, Any]:
    """MuJoCo outcome codes of every distinct state of the group × prior × draw."""
    t0 = time.perf_counter()
    sc = SCENES[plan["scene"]]
    points = cat.uniform_points(plan["draws_seed"], K, sc.hidden_names)
    states: dict[str, dict[str, Any]] = {}
    for member in ("base", "pair"):
        for full in member_states(sc, plan, member).values():
            states.setdefault(state_id(full), full)
    outcomes: dict[str, dict[str, list[int]]] = {}
    for sid, full in states.items():
        outcomes[sid] = {}
        for prior in cat.PRIORS:
            values = sc.values(full, prior, points)
            codes = []
            for k in range(K):
                result = sc.simulate(full, {q: float(v[k]) for q, v in values.items()})
                codes.append(sc.outcomes.index(result["outcome"]))
            outcomes[sid][prior] = codes
    return {"scene": plan["scene"], "group": plan["group"], "states": states, "outcomes": outcomes, "seconds": round(time.perf_counter() - t0, 3),
            "rollouts": len(states) * len(cat.PRIORS) * K}


def _freq(sc: Scene, codes: Sequence[int]) -> dict[str, float]:
    return {o: sum(1 for c in codes if c == i) / len(codes) for i, o in enumerate(sc.outcomes)}


def _key(sc: Scene, freq: Mapping[str, float]) -> float:
    return float(sum(freq[o] for o in sc.key_events))


def assemble_group(plan: Mapping[str, Any], labelled: Mapping[str, Any], *, git: str | None = None) -> list[dict[str, Any]]:
    """The group's two records (base first)."""
    sc = SCENES[plan["scene"]]
    names = text.QUESTION_NAMES[sc.name]
    alternatives = {a.key: a for a in sc.alternatives()}
    action_options = [text.PLAN_OPTION, *plan["options"]]
    option_variant = {text.PLAN_OPTION: "planned", **{k: f"opt_{k}" for k in plan["options"]}}
    words = [(text.PLAN_OPTION, "")] + [(k, alternatives[k].words) for k in plan["options"]]
    compare = plan["compare_option"]
    questions = text.questions(sc.name, plan["style"], cf_value=plan["cf"]["value"], options=words, compare=(compare, alternatives[compare].words))
    question_kind = {names[k]: QUESTION_KIND_LABEL[k] for k in text.KINDS}
    order = {name: list(q["criteria"]) for name, q in questions.items() if q["type"] == "choice"}
    reversal = {names["direction"]: ("higher", "lower"), names["compare"]: (compare, text.PLAN_OPTION)}
    rid = {m: f"v06a3-{sc.name}-{plan['group']:06d}" + ("" if m == "base" else "p") for m in ("base", "pair")}
    built: dict[str, dict[str, Any]] = {}
    keys_by_prior: dict[str, dict[str, float]] = {}
    for member in ("base", "pair"):
        states = member_states(sc, plan, member)
        freq = {variant: {prior: _freq(sc, labelled["outcomes"][state_id(full)][prior]) for prior in cat.PRIORS} for variant, full in states.items()}
        by_prior = lab.record_labels(freq=freq, outcomes=sc.outcomes, key_events=sc.key_events, severe=sc.severe, costs=sc.costs, names=names,
                                     options=option_variant, compare_option=compare, plan_option=text.PLAN_OPTION,
                                     same_option=text.SAME_OPTION["compare"])
        robust = lab.robustness(by_prior, order, reversal)
        keys_by_prior[member] = {p: _key(sc, freq["planned"][p]) for p in cat.PRIORS}
        contrast = {p: lab.direction_word(_key(sc, freq["cf"][p]) - _key(sc, freq["planned"][p])) for p in cat.PRIORS}
        state = states["planned"]
        base_specs = sc.hidden_specs(state)
        variants = member_variants(sc, plan, member)
        aim_key = plan["aim"]["base" if member == "base" else "pair"]
        aim_cf = plan["aim"]["cf_base" if member == "base" else "cf_pair"]
        meta = {
            "set": SET, "split": "train", "author": AUTHOR, "law": sc.law, "scene": sc.name, "group": sc.name, "phenomena": list(sc.phenomena),
            "language": "en", "domain": sc.domain, "style": "work_order_checklist", "key_question": names["key"],
            "question_kind": question_kind, "key_events": list(sc.key_events), "severe_event": sc.severe, "outcome_names": list(sc.outcomes),
            "plan_option": text.PLAN_OPTION, "costs": dict(sc.costs),
            "visible": {**state, "domain": sc.domain},
            "hidden_prior": cat.prior_ranges(base_specs), "prior_settings": dict(cat.PRIOR_SETTINGS),
            "variants": variants,
            "plan": {"decisive": sc.decisive, "aim_key": aim_key, "aim_cf": aim_cf, "cf_kind": plan["cf"]["kind"], "cf_value": plan["cf"]["value"],
                     "options": dict(option_variant), "option_values": {v: variants[v] for v in option_variant.values() if v != "planned"}},
            "outcomes": freq,
            "labels_by_prior": by_prior,
            "robustness": robust,
            "contrast_direction_by_prior": contrast,
            "contrast_direction_same": len(set(contrast.values())) == 1,
            "contrast_no_reversal": not ("up" in contrast.values() and "down" in contrast.values()),
            "compare_option": compare, "action_options": action_options,
            "static_aim": {"key": aim_key, "cf": aim_cf, "cf_kind": plan["cf"]["kind"]},
        }
        record = {
            "schema": "simgen-request/0", "id": rid[member],
            "generator": {"name": GENERATOR, "family": sc.name, "group": int(plan["group"]), "seed": int(plan["seed"]),
                          "draws_seed": int(plan["draws_seed"]), "k": K, "priors": list(cat.PRIORS), "mujoco": mujoco.__version__, "git": git,
                          "member": member},
            "meta": meta,
            "request": {"state": text.state_text(sc.name, state, plan["style"]), "questions": questions},
            "labels": by_prior["base"],
        }
        built[member] = record
    deltas = {p: keys_by_prior["pair"][p] - keys_by_prior["base"][p] for p in cat.PRIORS}
    directions = {p: lab.direction_word(d) for p, d in deltas.items()}
    for member in ("base", "pair"):
        built[member]["meta"]["pair"] = {
            "of": rid["pair" if member == "base" else "base"], "role": "base" if member == "base" else "variant", "kind": "numeric",
            "design": plan["design"], "field": plan["field"], "from": plan["from"], "to": plan["to"],
            "closeness": "far" if plan["design"] == "far_number" else "one_factor", "factor": plan["factor"], "fact": plan["fact"],
            "key_by_prior": {p: [round(keys_by_prior["base"][p], 6), round(keys_by_prior["pair"][p], 6)] for p in cat.PRIORS},
            "key_delta": round(deltas["base"], 6), "direction_by_prior": directions, "direction_same": len(set(directions.values())) == 1,
            "no_reversal": not ("up" in directions.values() and "down" in directions.values()),
        }
    return [built["base"], built["pair"]]


# --------------------------------------------------------------------------
# files
# --------------------------------------------------------------------------


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    out = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                out.append(json.loads(line))
    return out


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    n = 0
    with tmp.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=False) + "\n")
            n += 1
    tmp.replace(path)
    return n


def _label_worker(plan: Mapping[str, Any]) -> dict[str, Any]:
    return label_group(plan)


def label_scene(plans: Sequence[Mapping[str, Any]], out_path: Path, *, workers: int, log=sys.stderr) -> dict[str, Any]:
    """Label every plan not yet in `out_path` (resumable: one line per finished group, appended and flushed)."""
    done = {int(row["group"]) for row in read_jsonl(out_path)}
    todo = [p for p in plans if int(p["group"]) not in done]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    finished = 0
    if todo:
        with out_path.open("a", encoding="utf-8") as handle, mp.get_context("fork").Pool(int(workers)) as pool:
            for result in pool.imap_unordered(_label_worker, todo, chunksize=1):
                handle.write(json.dumps(result) + "\n")
                handle.flush()
                finished += 1
                if finished % 10 == 0 or finished == len(todo):
                    rate = (time.perf_counter() - t0) / finished
                    print(f"[a3 label {todo[0]['scene']}] {finished}/{len(todo)} groups, {rate:.2f} s/group wall, ETA {rate * (len(todo) - finished) / 60:.1f} min",
                          file=log, flush=True)
    return {"already": len(done), "labelled": finished, "wall_s": round(time.perf_counter() - t0, 1)}
