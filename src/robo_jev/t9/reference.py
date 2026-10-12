"""T9 references — the textbook calculator and constant answers, written as predictions (Task T9 §1-4).

* **Calculator** (:func:`calculator_prediction`) — every question answered by the family's closed-form calculator
  (:meth:`robo_jev.t9.families.Family.calc`: rolling with rolling resistance and a lip-impact energy balance, a driven pendulum,
  an impulsive strike, a rocking block under the belt's pulse, lever statics, friction-limited suction) on the record's **K base-prior
  draws** (rebuilt from ``generator.draws_seed`` and the words in ``meta.visible``), exactly as T8's static calculator — so it knows
  the ranges behind the words (privileged). ``means=True`` evaluates it once at the **prior means** instead (a single best guess per
  hidden quantity; answers are 0 or 1). Answers: the key event's share for yes/no and score; the direction class of the share's change
  (the label rule); the cheapest action under the same stated costs; the outcome shares.
* **Constants** (:func:`constant_prediction`) — ``half``: 0.5 for every yes/no and score, even odds on every choice; ``low``: 0.1, "no
  real change", the plan, nothing happens (the "nothing will happen" answer); ``high``: 0.9, "likelier", the safe step, the key
  outcome (the alarmist answer); ``gentler``: as ``half``, but always the gentler action (the cheapest option in most moments).

Predictions use the diagnosis tool's format (one listing per question) so :mod:`robo_jev.bench.system1` reads them like a model's.
"""

from __future__ import annotations

from typing import Any

from robo_jev.t9 import priors as P
from robo_jev.t9 import text as T
from robo_jev.t9.build import VARIANTS, _change_from, _dir_class, variant_vis
from robo_jev.t9.families import FAMILIES

__all__ = ["CONSTANTS", "calculator_prediction", "constant_prediction"]

CONSTANTS = ("half", "low", "high", "gentler")


def _prediction(record: dict[str, Any], answers: dict[str, dict[str, Any]], model: str) -> dict[str, Any]:
    questions = {}
    for name, question in record["request"]["questions"].items():
        answer = answers[name]
        options = list(question["criteria"]) if question["type"] == "choice" else None
        questions[name] = {"type": question["type"], "n_listings": 1, "answer": answer, "code_mass": None,
                           "listings": [{"options": options, "values": None, "codes": None, "code_probs": None, "code_mass": None, "answer": answer}]}
    return {"id": record["id"], "model": model, "mode": "reference", "format": "closed-form", "prompt_version": None, "orders": "stored",
            "shuffle_seed": None, "questions": questions}


def _answers_from_shares(record: dict[str, Any], shares: dict[str, float], outcome_shares: dict[str, float]) -> dict[str, dict[str, Any]]:
    meta = record["meta"]
    fam = FAMILIES[meta["family"]]
    names = T.NAMES[fam.name]
    costs = meta["costs"]
    cls = _dir_class(shares["dirv"] - shares["plan"])
    plan_key, alt_key, safe_key = T.ACT_OPTIONS[fam.name]
    expected = {plan_key: shares["plan"] * costs["fail"], alt_key: shares["alt"] * costs["fail"] + costs["alt"], safe_key: costs["secure"]}
    low = min(expected.values())
    best = [k for k, v in expected.items() if abs(v - low) <= 1e-12]
    keys = T.OUTCOME_OPTIONS[fam.name]
    return {
        names["key"]: {"p_yes": shares["plan"]},
        names["cf"]: {"value": shares["cf"], "bins": None},
        names["alt"]: {"p_yes": shares["alt"]},
        names["dir"]: {"probabilities": {k: (1.0 if k == cls else 0.0) for k in T.DIR_OPTIONS}},
        names["act"]: {"probabilities": {k: (1.0 / len(best) if k in best else 0.0) for k in expected}},
        names["outcome"]: {"probabilities": {key: outcome_shares[o] for key, o in zip(keys, meta["outcome_names"])}},
    }


def calculator_prediction(record: dict[str, Any], *, means: bool = False) -> dict[str, Any]:
    """The calculator's answers to one record (module docstring)."""
    meta = record["meta"]
    fam = FAMILIES[meta["family"]]
    spec = {"cf": _change_from(meta["changes"]["cf"]), "alt": meta["changes"]["alt"]["value"], "dirv": _change_from(meta["changes"]["dirv"])}
    names = fam.hidden_names()
    units = P.unit_draws(names, int(record["generator"]["k"]), int(record["generator"]["draws_seed"]))
    shares: dict[str, float] = {}
    outcome_shares: dict[str, float] = {}
    for variant in VARIANTS:
        vis = variant_vis(fam, meta["visible"], spec, variant)
        if means:
            ranges = {**P.prior_ranges(fam.words(vis), "base"), **fam.fixed(vis)}
            draws = [{n: (float(ranges[n][0]) + float(ranges[n][1])) / 2 for n in names}]
        else:
            draws = P.draws_for(fam.words(vis), fam.fixed(vis), "base", units, names)
        outcomes = fam.calc(vis, draws)
        shares[variant] = sum(o in fam.key for o in outcomes) / len(outcomes)
        if variant == "plan":
            outcome_shares = {o: outcomes.count(o) / len(outcomes) for o in meta["outcome_names"]}
    return _prediction(record, _answers_from_shares(record, shares, outcome_shares), "calculator-means" if means else "calculator")


def constant_prediction(record: dict[str, Any], kind: str) -> dict[str, Any]:
    """A constant answer (module docstring) to one record."""
    if kind not in CONSTANTS:
        raise ValueError(f"kind: one of {CONSTANTS}")
    meta = record["meta"]
    fam = FAMILIES[meta["family"]]
    names = T.NAMES[fam.name]
    value = {"half": 0.5, "low": 0.1, "high": 0.9, "gentler": 0.5}[kind]
    answers: dict[str, dict[str, Any]] = {names["key"]: {"p_yes": value}, names["cf"]: {"value": value, "bins": None}, names["alt"]: {"p_yes": value}}
    choices = {
        names["dir"]: ("no_real_change" if kind == "low" else "likelier", list(T.DIR_OPTIONS)),
        names["act"]: (T.ACT_OPTIONS[fam.name][0] if kind == "low" else T.ACT_OPTIONS[fam.name][2], list(T.ACT_OPTIONS[fam.name])),
        names["outcome"]: (T.OUTCOME_OPTIONS[fam.name][0] if kind == "low" else T.OUTCOME_OPTIONS[fam.name][-1], list(T.OUTCOME_OPTIONS[fam.name])),
    }
    if kind == "gentler":
        choices[names["act"]] = (T.ACT_OPTIONS[fam.name][1], list(T.ACT_OPTIONS[fam.name]))
    for name, (pick, keys) in choices.items():
        if kind == "half" or (kind == "gentler" and name != names["act"]):
            answers[name] = {"probabilities": {k: 1.0 / len(keys) for k in keys}}
        else:
            answers[name] = {"probabilities": {k: (1.0 if k == pick else 0.0) for k in keys}}
    return _prediction(record, answers, f"const-{kind}")
