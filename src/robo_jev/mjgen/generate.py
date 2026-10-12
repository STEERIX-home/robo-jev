"""v0.3b records — labelled groups → ``simgen-request/0`` records, one-fact pairs, robustness across priors, balanced selection and
statistics (Task v0.3b §1.3–1.6, controller update 2026-10-12).

**Labels.** Every question is labelled from the per-draw outcomes of the variants it is about, under each prior setting
(:data:`robo_jev.mjgen.families.PRIORS`); ``labels`` = the ``base`` labels and ``meta.labels_by_prior`` keeps all three. The raw
outcome frequencies of every variant under every prior are in ``meta.outcomes`` (later transforms derive coarse risk bands and other
comparisons from them).

* key yes/no — share of draws where the key event (a failure: slide or topple · slip or drop · a box falls) happens under the plan;
* severe yes/no — share of draws with the severe outcome (topple · drop · a lower box goes too);
* what happens (``choice``, ``kind: state``) — the outcome shares under the plan;
* direction (``choice``, ``kind: state``; more likely / less likely / about the same) and comparison of two actions (``choice``,
  ``kind: state``) — from the paired per-draw differences d_k = event_B − event_A on the shared draws (:func:`contrast_probs`):
  P(more) = P(Δ > 0.1), P(less) = P(Δ < −0.1), the rest "about the same", with Δ ~ Normal(mean d, sd d / √K) (sd floored at
  1/(2K)) — K-sample uncertainty, not opinion;
* safest action (``choice``, ``kind: action``) — expected cost of each option = P(key event) + 0.5 · P(severe outcome) under that
  option, probabilities = softmax(−(cost − min cost) / 0.05), ``expected_cost`` stored; in a seeded half of the grip and tower
  records the option that is safest by construction is left out (:data:`DOMINANT_OPTION`);
* score — the key-event share at the counterfactual value, with both anchors written out.

**Robustness** (``meta.robustness``, per question): yes/no and score → min/max over the three priors and whether the risk band
(low < 1/3 ≤ middle ≤ 2/3 < high) is the same under all; choices → the argmax under each prior and whether it is the same under all
(min/max per option kept); direction and comparison questions also carry ``no_reversal`` (no prior picks the opposite side — one
may say "about the same" when both values saturate). Counterfactual contrasts: the direction of the key-event change (up / down / flat within ±0.05) from the
plan to the counterfactual value, and for paired records from one partner to the other (``meta.pair.direction_by_prior``), and
whether it is the same under all priors.
"""

from __future__ import annotations

import collections
import dataclasses
import math
import random
import statistics
from typing import Any, Iterable, Sequence

import numpy as np
from scipy.stats import norm

from robo_jev.decisions.records import SCHEMA, validate_record
from robo_jev.mjgen import families as fam
from robo_jev.mjgen import text as tx

__all__ = [
    "BANDS",
    "DELTA",
    "QUESTION_NAMES",
    "band",
    "cap_still_pairs",
    "contrast_probs",
    "direction",
    "group_records",
    "select_balanced",
    "stats",
]

DELTA = 0.1
TAU = 0.05
SEVERE_WEIGHT = 0.5
BANDS = (1 / 3, 2 / 3)
FLAT = 0.05
QUESTION_NAMES = {
    "tilt": {"key": "moves_at_plan", "direction": "pitch_change_effect", "outcome": "load_after_tilt", "action": "safest_way_to_carry",
             "compare": "riskier_orientation", "score_cf": "risk_at_other_pitch", "severe": "topples_at_plan"},
    "grip": {"key": "loses_grip", "direction": "force_change_effect", "outcome": "item_after_move", "action": "safest_grip_plan",
             "compare": "riskier_grip_plan", "score_cf": "slip_risk_other_force", "severe": "falls_out"},
    "tower": {"key": "box_falls", "direction": "offset_change_effect", "outcome": "stack_after_release", "action": "safest_stacking",
              "compare": "riskier_stacking", "score_cf": "fall_risk_other_offset", "severe": "lower_box_goes"},
}
OUTCOME_KEYS = {"tilt": {"stays": "stay", "slides": "slide", "topples": "tip"}, "grip": {"held": "hold", "slipped": "slip", "dropped": "drop"},
                "tower": {"all_stay": "stands", "top_falls": "top", "more_fall": "more"}}
PLANNED_KEY = {"tilt": "as_it_stands", "grip": "as_planned", "tower": "as_planned"}
TILT_OPTION_KEY = {"upright": "upright", "lengthwise": "lengthwise", "crosswise": "crosswise"}


def band(value: float) -> str:
    return "low" if value < BANDS[0] else ("high" if value > BANDS[1] else "middle")


def direction(delta: float) -> str:
    return "up" if delta > FLAT else ("down" if delta < -FLAT else "flat")


def contrast_probs(first: Sequence[bool], second: Sequence[bool]) -> dict[str, float]:
    """(more, less, same) for "is the event more likely under `second` than under `first`?" on shared draws (module docstring)."""
    d = np.asarray(second, dtype=float) - np.asarray(first, dtype=float)
    k = d.size
    mean = float(d.mean())
    sd = float(d.std(ddof=1)) if k > 1 else 0.0
    se = max(sd / math.sqrt(k), 0.5 / k)
    more = float(1.0 - norm.cdf((DELTA - mean) / se))
    less = float(norm.cdf((-DELTA - mean) / se))
    return _normalise({"more": more, "less": less, "same": max(0.0, 1.0 - more - less)})


def _normalise(probs: dict[str, float]) -> dict[str, float]:
    """Round to 6 decimals so the sum is exactly 1 (within 1e-9): the rounding remainder goes to the largest entry."""
    out = {k: round(max(0.0, v), 6) for k, v in probs.items()}
    biggest = max(out, key=lambda key: out[key])
    out[biggest] = round(out[biggest] + 1.0 - sum(out.values()), 6)
    return out


def _softmax_costs(costs: dict[str, float]) -> dict[str, float]:
    low = min(costs.values())
    weights = {k: math.exp(-(c - low) / TAU) for k, c in costs.items()}
    total = sum(weights.values())
    return _normalise({k: w / total for k, w in weights.items()})


def _events(family: str, outcomes: Sequence[str]) -> list[bool]:
    return [o in fam.EVENT[family] for o in outcomes]


#: options that are safest in most groups by construction (pulling the top box back; a much firmer grip with a faster move): in a
#: seeded half of the records the safest-action question leaves them out, so the question is not answerable by a fixed choice
#: (interim check: tower "top_back" safest in 93 %, grip "firmer_faster" in 85 % of the 3-option questions)
DOMINANT_OPTION = {"grip": "firmer_faster", "tower": "top_back"}


def _action_keys(group: fam.Group) -> list[str]:
    """Option keys of the record's safest-action question (seeded by the wording seed, so a pair shows the same options)."""
    keys = list(group.plan["options"])
    dominant = DOMINANT_OPTION.get(group.family)
    if dominant and random.Random(group.wording ^ 0xAC71).random() < 0.5:
        keys = [k for k in keys if k != dominant]
    return keys


def _cf_option(group: fam.Group) -> str:
    """The alternative action the comparison question is about (seeded by the wording seed, so a pair asks about the same one)."""
    keys = [k for k, v in group.plan["options"].items() if v != "planned"]
    return random.Random(group.wording ^ 0xC0DE).choice(keys)


def _labels(group: fam.Group, variants: dict[str, Any], prior: str, questions: dict[str, Any], cf_option: str) -> dict[str, Any]:
    family = group.family
    names = QUESTION_NAMES[family]

    def ev(name: str) -> list[bool]:
        return _events(family, variants[name]["outcomes"][prior])

    def freq(name: str) -> dict[str, float]:
        return variants[name]["freq"][prior]

    out: dict[str, Any] = {}
    for kind, qname in names.items():
        if qname not in questions:
            continue
        if kind == "key":
            out[qname] = {"type": "noul", "p_yes": float(np.mean(ev("planned")))}
        elif kind == "severe":
            out[qname] = {"type": "noul", "p_yes": float(freq("planned")[fam.SEVERE[family]])}
        elif kind == "score_cf":
            out[qname] = {"type": "score", "value": float(np.mean(ev("cf")))}
        elif kind == "outcome":
            probs = {key: float(freq("planned")[o]) for key, o in OUTCOME_KEYS[family].items()}
            out[qname] = {"type": "choice", "kind": "state", "probabilities": _normalise(probs)}
        elif kind == "direction":
            c = contrast_probs(ev("planned"), ev("cf"))
            out[qname] = {"type": "choice", "kind": "state", "probabilities": {"more_likely": c["more"], "less_likely": c["less"], "about_same": c["same"]}}
        elif kind == "compare":
            other = group.plan["options"][cf_option]
            c = contrast_probs(ev("planned"), ev(other))
            out[qname] = {"type": "choice", "kind": "state", "probabilities": {PLANNED_KEY[family]: c["less"], cf_option: c["more"], "about_same": c["same"]}}
        elif kind == "action":
            costs = {}
            for key in questions[qname]["criteria"]:
                vname = group.plan["options"][key]
                costs[key] = round(float(np.mean(ev(vname))) + SEVERE_WEIGHT * float(freq(vname)[fam.SEVERE[family]]), 6)
            out[qname] = {"type": "choice", "kind": "action", "probabilities": _softmax_costs(costs), "expected_cost": costs}
    return out


def _robust(question: dict[str, Any], by_prior: dict[str, dict[str, Any]]) -> dict[str, Any]:
    labels = [by_prior[p] for p in fam.PRIORS]
    if question["type"] in ("noul", "score"):
        values = [float(l.get("p_yes", l.get("value"))) for l in labels]
        bands = [band(v) for v in values]
        return {"type": question["type"], "min": min(values), "max": max(values), "bands": dict(zip(fam.PRIORS, bands)),
                "band_same": len(set(bands)) == 1, "robust": len(set(bands)) == 1}
    argmax = {p: max(l["probabilities"], key=lambda k: (l["probabilities"][k], k)) for p, l in zip(fam.PRIORS, labels)}
    keys = list(labels[0]["probabilities"])
    out = {"type": "choice", "argmax": argmax, "argmax_same": len(set(argmax.values())) == 1, "robust": len(set(argmax.values())) == 1,
           "min": {k: min(l["probabilities"][k] for l in labels) for k in keys}, "max": {k: max(l["probabilities"][k] for l in labels) for k in keys}}
    if "about_same" in keys and len(keys) == 3:
        # direction / comparison: the two sides never both win (one prior may say "about the same" when both values saturate)
        out["no_reversal"] = len(set(argmax.values()) - {"about_same"}) <= 1
    return out


def _record_id(group: fam.Group, member: str) -> str:
    return f"v03b-{group.family}-{group.index:06d}" + ("p" if member == "pair" else "")


def group_records(group: fam.Group, labelled: dict[str, Any], *, mujoco_version: str, git: str | None = None) -> list[dict[str, Any]]:
    """One labelled group → its records (the base, and the paired record when there is one)."""
    if group.visible.get("robot") in fam.ROBOT_RENAMES:
        group = dataclasses.replace(group, visible={**group.visible, "robot": fam.ROBOT_RENAMES[group.visible["robot"]]})
    family = group.family
    names = QUESTION_NAMES[family]
    cf_option = _cf_option(group)
    action_keys = _action_keys(group)
    members = ("base", "pair") if group.pair is not None else ("base",)
    built: dict[str, dict[str, Any]] = {}
    for member in members:
        visible = group.visible_for(member)
        state = tx.state_text(family, visible, group.style, group.wording)
        questions = tx.questions(family, visible, group.style, group.wording, group.mix, {**group.plan, "action_keys": action_keys}, cf_option)
        variants = labelled["members"][member]["variants"]
        by_prior = {p: _labels(group, variants, p, questions, cf_option) for p in fam.PRIORS}
        robustness = {q: _robust(questions[q], {p: by_prior[p][q] for p in fam.PRIORS}) for q in questions}
        key = names["key"]
        contrast = {p: direction(float(np.mean(_events(family, variants["cf"]["outcomes"][p]))) - by_prior[p][key]["p_yes"]) for p in fam.PRIORS}
        kinds = {qname: kind for kind, qname in names.items() if qname in questions}
        record = {
            "schema": SCHEMA,
            "id": _record_id(group, member),
            "generator": {"name": fam.GENERATOR, "family": family, "group": group.index, "seed": group.seed, "draws_seed": labelled["draws_seed"],
                          "k": labelled["k"], "priors": list(fam.PRIORS), "mujoco": mujoco_version, "git": git, "member": member},
            "meta": {
                "set": "v03b-mujoco-train", "split": "train", "scene": family, "group": family, "phenomena": [family], "language": "en",
                "domain": visible.get("domain"), "style": group.style, "key_question": key, "question_kind": kinds,
                "visible": visible, "hidden_prior": {p: _jsonable(r) for p, r in labelled["members"][member]["hidden"].items()},
                "prior_settings": {p: fam.PRIOR_SETTINGS[p]["what"] for p in fam.PRIORS},
                "variants": {name: group.variants[name] for name in variants}, "plan": _jsonable(group.plan),
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
        directions = {p: direction(b - a) for p, (a, b) in by_prior.items()}
        common = {"kind": "numeric", "field": group.pair["field"], "from": group.pair["from"], "to": group.pair["value"],
                  "closeness": group.pair["closeness"], "key_by_prior": by_prior, "key_delta": by_prior["base"][1] - by_prior["base"][0],
                  "direction_by_prior": directions, "direction_same": len(set(directions.values())) == 1,
                  "no_reversal": not {"up", "down"} <= set(directions.values())}
        base["meta"]["pair"] = {"of": pair["id"], "role": "base", **common}
        pair["meta"]["pair"] = {"of": base["id"], "role": "variant", **common}
    return [built[m] for m in members]


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, float) and math.isinf(value):
        return None
    return value


# --------------------------------------------------------------------------
# Balanced selection
# --------------------------------------------------------------------------

#: at most this share of the kept pairs may be "designed still" — aimed at a small change (``near``) or changing a number the
#: outcome barely depends on (``free``: mass on a slope, the travel distance of a lift, a box's mass in a two-box stack …). The
#: rest are aimed far, so the label moves in most pairs (T8: the trained model barely reacted to changed numbers).
STILL_PAIR_SHARE = 0.2


def cap_still_pairs(items: Sequence[tuple[fam.Group, dict[str, Any]]], *, share: float = STILL_PAIR_SHARE) -> tuple[list[tuple[fam.Group, dict[str, Any]]], dict[str, Any]]:
    """Per family keep every far-aimed pair and the first (candidate order) near/free pairs up to `share` of the kept pairs; the
    other near/free groups lose their paired record (the base record stays, unpaired). A rule on the pair's design, not its outcome."""
    out: list[tuple[fam.Group, dict[str, Any]]] = []
    report: dict[str, Any] = {}
    by_family: dict[str, list[tuple[fam.Group, dict[str, Any]]]] = collections.defaultdict(list)
    for group, labelled in items:
        by_family[group.family].append((group, labelled))
    for family, rows in by_family.items():
        rows = sorted(rows, key=lambda gl: gl[0].index)
        far = sum(1 for g, _ in rows if g.pair is not None and g.pair["closeness"] == "far")
        allowed = int(math.floor(share / (1.0 - share) * far))
        kept_still = 0
        stripped = collections.Counter()
        for group, labelled in rows:
            if group.pair is not None and group.pair["closeness"] != "far":
                if kept_still >= allowed:
                    stripped[group.pair["closeness"]] += 1
                    group = dataclasses.replace(group, pair=None)
                    labelled = {**labelled, "members": {"base": labelled["members"]["base"]}}
                else:
                    kept_still += 1
            out.append((group, labelled))
        report[family] = {"far": far, "still_allowed": allowed, "still_kept": kept_still, "stripped": dict(stripped)}
    return out, report



def select_balanced(items: Iterable[tuple[fam.Group, dict[str, Any]]], targets: dict[str, int], *, bins: int = 10) -> tuple[list[tuple[fam.Group, dict[str, Any]]], dict[str, Any]]:
    """Per family keep `targets[family]` groups so the base record's key label (base prior) is spread evenly over `bins` equal bins of
    [0, 1] (candidate order inside a bin; a bin that runs short gives its slots to the others in turn). Groups with a rollout flagged
    unstable are dropped."""
    by_family: dict[str, list[tuple[fam.Group, dict[str, Any]]]] = collections.defaultdict(list)
    dropped: collections.Counter[str] = collections.Counter()
    for group, labelled in items:
        if labelled["flags"]["unstable"]:
            dropped[group.family] += 1
            continue
        by_family[group.family].append((group, labelled))
    kept: list[tuple[fam.Group, dict[str, Any]]] = []
    report: dict[str, Any] = {}
    for family, total in targets.items():
        pool = sorted(by_family.get(family, []), key=lambda gl: gl[0].index)
        binned: dict[int, list[tuple[fam.Group, dict[str, Any]]]] = collections.defaultdict(list)
        for group, labelled in pool:
            events = _events(family, labelled["members"]["base"]["variants"]["planned"]["outcomes"]["base"])
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
        report[family] = {"candidates": len(pool) + dropped[family], "dropped_unstable": dropped[family],
                          "pool_by_bin": [len(binned[b]) for b in range(bins)], "kept_by_bin": [len(taken[b]) for b in range(bins)],
                          "target": total, "kept": len(chosen), "shortfall": max(0, total - len(chosen))}
    return kept, report


# --------------------------------------------------------------------------
# Statistics
# --------------------------------------------------------------------------


def _hist(values: Sequence[float], bins: int = 10) -> list[int]:
    out = [0] * bins
    for v in values:
        out[min(bins - 1, int(float(v) * bins))] += 1
    return out


def stats(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Composition, label histograms, contrast-pair statistics and robustness counts (per family and pooled)."""
    out: dict[str, Any] = {"records": len(records), "by_family": {}}
    families = sorted({r["meta"]["scene"] for r in records})
    for family in families + ["all"]:
        rows = [r for r in records if family == "all" or r["meta"]["scene"] == family]
        names = None if family == "all" else QUESTION_NAMES[family]
        key_values = [r["labels"][r["meta"]["key_question"]]["p_yes"] for r in rows]
        binary = [float(l.get("p_yes", l.get("value"))) for r in rows for l in r["labels"].values() if l["type"] in ("noul", "score")]
        q_types = collections.Counter(q["type"] for r in rows for q in r["request"]["questions"].values())
        q_kinds = collections.Counter(r["meta"]["question_kind"][q] for r in rows for q in r["request"]["questions"])
        robust_by_kind: dict[str, list[int]] = collections.defaultdict(lambda: [0, 0])
        no_reversal: dict[str, list[int]] = collections.defaultdict(lambda: [0, 0])
        for r in rows:
            for q, entry in r["meta"]["robustness"].items():
                kind = r["meta"]["question_kind"][q]
                robust_by_kind[kind][0] += int(entry["robust"])
                robust_by_kind[kind][1] += 1
                if "no_reversal" in entry:
                    no_reversal[kind][0] += int(entry["no_reversal"])
                    no_reversal[kind][1] += 1
        robust_total = [sum(v[0] for v in robust_by_kind.values()), sum(v[1] for v in robust_by_kind.values())]
        # (a) counterfactual inside the record: the key event at the plan vs at the counterfactual value (base prior)
        inside = []
        for r in rows:
            fam_name = r["meta"]["scene"]
            planned = r["meta"]["outcomes"]["planned"]["base"]
            cf = r["meta"]["outcomes"]["cf"]["base"]
            ev = fam.EVENT[fam_name]
            inside.append(sum(cf[o] for o in ev) - sum(planned[o] for o in ev))
        direction_argmax = collections.Counter()
        for r in rows:
            qname = QUESTION_NAMES[r["meta"]["scene"]]["direction"]
            probs = r["labels"][qname]["probabilities"]
            direction_argmax[max(probs, key=lambda k: (probs[k], k))] += 1
        # (b) paired records (each pair once)
        pairs = [r for r in rows if r["meta"]["pair"] and r["meta"]["pair"]["role"] == "base"]
        deltas = [abs(r["meta"]["pair"]["key_delta"]) for r in pairs]
        by_field: dict[str, list[float]] = collections.defaultdict(list)
        by_design: dict[str, list[float]] = collections.defaultdict(list)
        for r in pairs:
            by_field[r["meta"]["pair"]["field"].replace(".", "_")].append(abs(r["meta"]["pair"]["key_delta"]))
            by_design[r["meta"]["pair"]["closeness"]].append(abs(r["meta"]["pair"]["key_delta"]))
        in_pairs = sum(1 for r in rows if r["meta"]["pair"])
        out["by_family"][family] = {
            "records": len(rows), "groups": len({(r["meta"]["scene"], r["generator"]["group"]) for r in rows}),
            "questions": sum(q_types.values()), "question_types": dict(sorted(q_types.items())), "question_kinds": dict(sorted(q_kinds.items())),
            "key_label_hist10": _hist(key_values), "key_label_mean": statistics.fmean(key_values) if key_values else None,
            "binary_label_hist10": _hist(binary), "uncertain_share": (sum(1 for v in binary if 0.1 <= v <= 0.9) / len(binary)) if binary else None,
            "styles": dict(collections.Counter(r["meta"]["style"] for r in rows)),
            "contrast_inside": {"records": len(inside), "share_with": 1.0 if rows else None,
                                "abs_delta_ge_0.1": (sum(1 for d in inside if abs(d) >= 0.1) / len(inside)) if inside else None,
                                "abs_delta_hist10": _hist([abs(d) for d in inside]), "direction_label_argmax": dict(direction_argmax),
                                "direction_same_all_priors": (sum(1 for r in rows if r["meta"]["contrast_direction_same"]) / len(rows)) if rows else None,
                                "no_reversal_all_priors": (sum(1 for r in rows if r["meta"]["contrast_no_reversal"]) / len(rows)) if rows else None},
            "pairs": {"pairs": len(pairs), "records_in_pairs": in_pairs, "share_of_records": in_pairs / len(rows) if rows else None,
                      "abs_delta_ge_0.1": (sum(1 for d in deltas if d >= 0.1) / len(deltas)) if deltas else None,
                      "abs_delta_hist10": _hist(deltas), "abs_delta_median": statistics.median(deltas) if deltas else None,
                      "by_field": {f: {"pairs": len(v), "abs_delta_ge_0.1": sum(1 for d in v if d >= 0.1) / len(v), "median": statistics.median(v)}
                                   for f, v in sorted(by_field.items())},
                      "by_design": {f: {"pairs": len(v), "abs_delta_ge_0.1": sum(1 for d in v if d >= 0.1) / len(v), "median": statistics.median(v)}
                                    for f, v in sorted(by_design.items())},
                      "direction_same_all_priors": (sum(1 for r in pairs if r["meta"]["pair"]["direction_same"]) / len(pairs)) if pairs else None,
                      "no_reversal_all_priors": (sum(1 for r in pairs if r["meta"]["pair"]["no_reversal"]) / len(pairs)) if pairs else None},
            "robust": {"questions": robust_total[1], "robust": robust_total[0], "share": robust_total[0] / robust_total[1] if robust_total[1] else None,
                       "by_kind": {k: {"questions": v[1], "robust": v[0], "share": v[0] / v[1]} for k, v in sorted(robust_by_kind.items())},
                       "no_reversal_by_kind": {k: {"questions": v[1], "no_reversal": v[0], "share": v[0] / v[1]} for k, v in sorted(no_reversal.items())}},
        }
        if names is None:
            out["by_family"][family].pop("contrast_inside", None) if not rows else None
    return out
