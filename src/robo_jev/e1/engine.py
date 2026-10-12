"""Task E1 condition A — fresh engine moments in the engine's own format.

About 400 moments from the v0b generator in the training distribution: the training split (train settings, no held-out style,
setting, phrase pool or combination), phrasing ``v0b``, ``fact_shuffle`` 0, and the four P3 tipping/falling scenes. Seeds are new:
they start at :data:`SEED_START` (3,000,000), above every range in ``configs/simgen/holdout-v0b.yaml`` (train 1,000,000–1,999,999,
T1–T7 2,000,000–2,699,999; the training generator's spare shifts stay inside the train range). These moments are not sealed.

Design choices inside the brief (recorded in the report):

* **English only** (``forced lang = en``): B re-voices English ("you" → "I") and T8b is English; the training split's 15 % Korean
  would add a language factor to a format comparison.
* **Single moments** (no one-fact pairs): the training set's ~10 % pair units are a dataset device, and the bootstrap resamples
  independent moments.
* **Equal shares per scene** (the four scenes in turn, the domain cycling over the domains that have a body for the scene).

Each record keeps only its **tipping/falling yes/no and score questions** (:func:`tip_fall_questions`): event recipes whose event is
``fall`` (topple over or slide off, collisions counted when stated), ``fall_motion`` (the same, motion only), ``tip`` or ``topple``
(a placed item falls off the stack). Records without one are dropped. Every question is scored on its own prompt (state + that
question), so the dropped questions change no answer.
"""

from __future__ import annotations

import copy
from typing import Any, Iterable

__all__ = ["ID_PREFIX", "P3_SCENES", "SEED_START", "SPARE_START", "TIP_FALL_EVENTS", "generate_a", "plan_a", "question_event", "restrict",
           "tip_fall_questions"]

P3_SCENES = ("p3_carrying_tall", "p3_place_stack", "p3_ramp", "p3p5_carrying_mixed")
#: Event names (yes/no ``event``, score ``name``) that ask whether the load tips over or falls.
TIP_FALL_EVENTS = frozenset({"fall", "fall_motion", "tip", "topple"})
SEED_START = 3_000_000
#: Spare seeds (a moment that cannot be built inside the training split moves here, in order).
SPARE_START = 3_050_000
SEED_END = 3_099_999
ID_PREFIX = "e1a"


def question_event(record: dict[str, Any], name: str) -> str | None:
    """The event a yes/no or score question asks about (from ``meta.basis``), or None (choices, quantity scores, world nouls)."""
    question = record["request"]["questions"][name]
    basis = ((record.get("meta") or {}).get("basis") or {}).get(name) or {}
    if question.get("type") not in ("noul", "score") or basis.get("source") != "event":
        return None
    return basis.get("event") if question["type"] == "noul" else basis.get("name")


def tip_fall_questions(record: dict[str, Any]) -> list[str]:
    """Names of the record's tipping/falling yes/no and score questions, in request order."""
    return [name for name in record["request"]["questions"] if question_event(record, name) in TIP_FALL_EVENTS]


def restrict(record: dict[str, Any], names: Iterable[str]) -> dict[str, Any]:
    """A copy of the record with only these questions (request, labels, truth.realized, meta.basis / options / criteria)."""
    keep = list(names)
    out = copy.deepcopy(record)
    out["request"]["questions"] = {n: out["request"]["questions"][n] for n in keep}
    out["labels"] = {n: out["labels"][n] for n in keep}
    realized = (out.get("truth") or {}).get("realized")
    if isinstance(realized, dict):
        out["truth"]["realized"] = {n: realized[n] for n in keep if n in realized}
    meta = out.get("meta") or {}
    for key in ("basis", "options", "criteria"):
        if isinstance(meta.get(key), dict):
            meta[key] = {n: v for n, v in meta[key].items() if n in keep}
    meta["e1_questions"] = {"kept": keep, "dropped": [n for n in record["request"]["questions"] if n not in keep]}
    return out


def plan_a(cfg, n_per_scene: int) -> list[dict[str, Any]]:
    """Units in turn over the four scenes (unit i → scene i mod 4); the domain cycles over the domains that can do the scene;
    seed = :data:`SEED_START` + i."""
    from robo_jev.simgen.catalog import DOMAINS, SCENE_BODIES, SCENE_GROUP

    domains = {s: [d for d in DOMAINS if any(cfg.domain(d)["bodies"].get(b, 0) > 0 for b in SCENE_BODIES[s])] for s in P3_SCENES}
    units = []
    for i in range(n_per_scene * len(P3_SCENES)):
        scene = P3_SCENES[i % len(P3_SCENES)]
        j = i // len(P3_SCENES)
        units.append({"index": i, "scene": scene, "group": SCENE_GROUP[scene], "domain": domains[scene][j % len(domains[scene])],
                      "seed": SEED_START + i})
    if units and units[-1]["seed"] >= SPARE_START:
        raise ValueError("too many units for the main seed block")
    return units


def generate_a(cfg, units: list[dict[str, Any]], *, git: str | None = None, log=None) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Build every unit in the training split (v0b phrasing, fact_shuffle 0) → (records restricted to their tipping/falling
    questions, report). A unit that cannot be built inside the split takes the next spare seed."""
    from robo_jev.simgen.generate import HeldOutError, assemble, build_moment, split_spec
    from robo_jev.simgen.holdout import load_holdout, split_for

    holdout = load_holdout()
    for name, (lo, hi) in holdout.seed_ranges.items():
        if not (SEED_END < lo or hi < SEED_START):
            raise ValueError(f"E1 seeds {SEED_START}..{SEED_END} overlap the {name} range {lo}..{hi}")
    split = split_for("train", holdout)
    spare = SPARE_START
    records, replaced, dropped = [], [], []
    for unit in units:
        forced = {"group": unit["group"], "scene": unit["scene"], "lang": "en"}
        seed = unit["seed"]
        for _ in range(40):
            try:
                moment = build_moment(cfg, seed, unit["domain"], forced, None, split)
                break
            except HeldOutError:
                replaced.append({"unit": unit["index"], "seed": seed, "spare": spare})
                seed, spare = spare, spare + 1
                if seed > SEED_END:
                    raise
        else:
            raise HeldOutError(f"unit {unit}: no feasible seed")
        record = assemble(cfg, moment, unit["index"], git=git, id_prefix=ID_PREFIX, forced=forced, split_spec=split_spec(split),
                          phrasing="v0b", fact_shuffle=0.0)
        names = tip_fall_questions(record)
        if not names:
            dropped.append(record["id"])
            continue
        records.append(restrict(record, names))
        if log is not None and len(records) % 50 == 0:
            log(f"A: {len(records)} records")
    report = {"units": len(units), "records": len(records), "dropped_no_tip_fall_question": dropped, "replaced_seeds": replaced,
              "seed_start": SEED_START, "spare_start": SPARE_START, "seed_end": SEED_END, "split": "train", "phrasing": "v0b", "fact_shuffle": 0.0,
              "language": "en", "scenes": list(P3_SCENES)}
    return records, report
