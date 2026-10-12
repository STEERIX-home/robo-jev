"""T8b analysis reference — a quasi-static textbook calculator on the same hidden draws (not a model under test).

For every T8b record it rebuilds the moment's K hidden draws (same seed as the labels) and decides each draw with closed-form
statics instead of MuJoCo:

* deck (braking at a or a curve with v²/R = a): tipping threshold a_tip = g·(b/2 − c)/h_c (b = base along the push of the inertial
  force, c = centre-of-mass offset toward the tipping edge, h_c = centre-of-mass height), sliding threshold a_slide = μ·g;
  ``stay`` if a ≤ min, else ``tip`` if a_tip < a_slide, else ``slide`` — no duration effect (a short stop above a_tip still counts
  as a tip here).
* stack: falls iff the centre of mass lands past the edge (overhang − L/2 + c + placement error > 0) — no drop dynamics.
* push: tips iff the bar height exceeds h* = μ_b·L + (b/2 − c)(1 − μ·μ_b)/μ — no start-up dynamics and no limit on the push
  travel (the MuJoCo push is displacement-limited, so a high bar may only lean the box, which then rocks back).

Its predictions are written in the diagnosis tool's prediction format (one listing per question) so the same scoring code reads
them. Its use is interpretation: the Brier between the calculator and the labels measures how much of the MuJoCo outcome is
dynamics rather than statics, and its skill is what exact statics with the generator's own priors would score. It is privileged
(it knows the prior ranges behind the words), so it is a reference, not a competitor.
"""

from __future__ import annotations

from typing import Any

from robo_jev.outside import physics as ph
from robo_jev.outside.moments import OUTCOMES, Moment, hidden_draws

__all__ = ["static_outcome", "static_prediction"]

G = ph.GRAVITY


def _moment(record: dict[str, Any]) -> Moment:
    meta = record["meta"]
    hidden = {k: tuple(v) for k, v in meta["hidden_prior"].items()}
    return Moment(meta["scene"], int(record["generator"]["candidate"]), int(record["generator"]["seed"]), meta["target"], meta["visible"], hidden,
                  meta["variants"])


def static_outcome(moment: Moment, variant: str, draw: dict[str, float]) -> str:
    """Closed-form outcome of one hidden draw for one question variant (module docstring)."""
    v, spec = moment.visible, moment.variants[variant]
    length, width, height = v["dims"]
    if moment.family in ("deck_brake", "deck_turn"):
        if spec["kind"] == "brake":
            accel, base = spec["decel"], length
        else:
            accel, base = spec["speed"] ** 2 / spec["radius"], width
        toward_edge = draw["cx"] * base  # + = toward the tipping edge — the sign convention of moments._com
        a_tip = G * (base / 2 - toward_edge) / (draw["kh"] * height)
        a_slide = draw["mu"] * G
        if accel <= min(a_tip, a_slide):
            return "stay"
        return "tip" if a_tip < a_slide else "slide"
    if moment.family == "stack":
        beyond = spec["overhang"] - length / 2 + draw["cx"] * length + draw["eps"]
        return "fall" if beyond > 0 else "stay"
    mu, mu_b = draw["mu"], ph.BAR_FRICTION
    threshold = mu_b * length + (length / 2 - draw["cx"] * length) * (1 - mu * mu_b) / mu
    return "tip" if spec["height"] > threshold else "upright"


def static_prediction(record: dict[str, Any], *, k: int | None = None) -> dict[str, Any]:
    """One record → a prediction line (diagnosis format, one listing per question) from the static calculator."""
    moment = _moment(record)
    k = int(k or record["generator"]["k"])
    draws = hidden_draws(moment.hidden, k, int(record["generator"]["draws_seed"]))
    outcomes = OUTCOMES[moment.family]
    freq: dict[str, dict[str, float]] = {}
    for name in moment.variants:
        decided = [static_outcome(moment, name, d) for d in draws]
        freq[name] = {o: decided.count(o) / len(decided) for o in outcomes}
    questions: dict[str, Any] = {}
    for name, question in record["request"]["questions"].items():
        variant = record["meta"]["question_variant"][name]
        f = freq[variant]
        if question["type"] == "choice":
            answer: dict[str, Any] = {"probabilities": {"stays": f["stay"], "slides": f["slide"], "topples": f["tip"]}}
            shown = {"options": ["stays", "slides", "topples"], "values": None}
        else:
            event = "stay" if name == "keeps_place" else ("fall" if moment.family == "stack" else "tip")
            value = f[event]
            answer = {"p_yes": value} if question["type"] == "noul" else {"value": value, "bins": None}
            shown = {"options": None, "values": None}
        listing = {**shown, "codes": None, "code_probs": None, "code_mass": None, "answer": answer}
        questions[name] = {"type": question["type"], "n_listings": 1, "answer": answer, "code_mass": None, "listings": [listing]}
    return {"id": record["id"], "model": "static-reference", "mode": "reference", "format": "closed-form", "prompt_version": None,
            "orders": "stored", "shuffle_seed": None, "questions": questions}
