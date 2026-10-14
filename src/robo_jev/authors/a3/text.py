"""A3's wording: the state as a numbered work order (formal, imperative fields) and the questions as numbered review items.

One **style** per group (:func:`pick_style`) fixes the header, the field labels, the review-item prefix and one phrasing per question
kind, so the two records of a pair differ only in the one stated fact that the pair changes. All text is English with metric units.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from robo_jev.authors.a3 import catalog as cat

__all__ = ["QUESTION_NAMES", "pick_style", "questions", "state_text"]

#: Question names per scene, in request order: key, severe, score at the counterfactual value, direction, comparison, safest action,
#: outcome category.
QUESTION_NAMES = {
    "indexdial": {"key": "wo_part_moves_on_plate", "severe": "wo_part_falls_over", "score_cf": "wo_move_odds_at_other_time",
                  "direction": "wo_index_time_effect", "compare": "wo_riskier_plan_or_change", "action": "wo_safest_way_to_index",
                  "outcome": "wo_part_end_state"},
    "mayostand": {"key": "wo_stand_lifts_or_tips", "severe": "wo_stand_tips_over", "score_cf": "wo_lift_odds_at_other_spot",
                  "direction": "wo_setdown_spot_effect", "compare": "wo_riskier_plan_or_change", "action": "wo_safest_way_to_set_down",
                  "outcome": "wo_stand_end_state"},
    "holddown": {"key": "wo_part_shifts_in_clamp", "severe": "wo_part_breaks_free", "score_cf": "wo_shift_odds_at_other_force",
                 "direction": "wo_press_force_effect", "compare": "wo_riskier_plan_or_change", "action": "wo_safest_way_to_press",
                 "outcome": "wo_part_end_state"},
}
KINDS = ("key", "severe", "score_cf", "direction", "compare", "action", "outcome")
PLAN_OPTION = "as_written"
SAME_OPTION = {"direction": "unchanged", "compare": "no_difference"}
PREFIXES = ("Check {n}:", "Item {n} -", "R{n}.", "Review {n}:")

# --------------------------------------------------------------------------
# formatting
# --------------------------------------------------------------------------


def _kg(v: float) -> str:
    v = float(v)
    return f"{v:.1f}" if abs(v * 10 - round(v * 10)) < 1e-6 else f"{v:.2f}"


def _s(v: float) -> str:
    return f"{float(v):.2f}"


def _int(v: float) -> str:
    return str(int(round(float(v))))


def _dims(d: Sequence[float]) -> str:
    return " x ".join(_int(v) for v in d) + " mm"


# --------------------------------------------------------------------------
# per-scene words
# --------------------------------------------------------------------------

DIAL_BASE_PHRASE = {"bare_alu": "bare machined-aluminium base", "cast_iron": "as-cast iron base", "powder_coat": "powder-coated steel base",
                    "gf_nylon": "glass-filled nylon base", "rubber_feet": "four rubber feet", "hard_anod": "hard-anodised aluminium base"}
DIAL_PLATE_PHRASE = {"dry": "clean and dry", "coolant": "has coolant mist on it", "oil": "has a thin oil film", "dust": "has grinding dust on it",
                     "mat": "covered by an anti-slip mat"}
DIAL_CELLS = ("Cell 2", "Cell 5", "kitting cell K1", "assembly cell B", "cobot cell 7", "test-fixture cell 3")
STAND_SITES = ("OR 3", "OR 6", "theatre 2 prep", "sterile services", "day-surgery suite", "endoscopy room 1")
CLAMP_SITES = ("BME workshop", "device assembly cell", "light-assembly cell", "prototype lab cell")

HEADERS = {
    "indexdial": ("WORK ORDER {wo} / {site}", "WO {wo} | {site} | dial index", "Job sheet {wo} ({site})"),
    "mayostand": ("WORK ORDER {wo} / {site}", "WO {wo} | {site} | stand load", "Task sheet {wo} ({site})"),
    "holddown": ("WORK ORDER {wo} / {site}", "WO {wo} | {site} | press-fit", "Job card {wo} ({site})"),
}
LABELS = {
    "indexdial": (("Task", "Part", "Position", "Contact", "Index"), ("Operation", "Item", "Placement", "Surfaces", "Motion")),
    "mayostand": (("Task", "Stand", "Load", "Set-down", "Release"), ("Job", "Equipment", "Item", "Placement", "Hand-off")),
    "holddown": (("Operation", "Part", "Nest", "Hold-down", "Press"), ("Step", "Workpiece", "Seat", "Clamping", "Force")),
}
SITES = {"indexdial": DIAL_CELLS, "mayostand": STAND_SITES, "holddown": CLAMP_SITES}


def pick_style(scene: str, rng: np.random.Generator) -> dict[str, Any]:
    """The group's text choices (shared by both records of the pair)."""
    return {
        "header": int(rng.integers(len(HEADERS[scene]))),
        "labels": int(rng.integers(len(LABELS[scene]))),
        "prefix": int(rng.integers(len(PREFIXES))),
        "site": int(rng.integers(len(SITES[scene]))),
        "wo": int(rng.integers(1000, 9999)),
        "q": {kind: int(rng.integers(2)) for kind in KINDS},
        "task": int(rng.integers(2)),
    }


def _fields(labels: Sequence[str], bodies: Sequence[str]) -> str:
    return "\n".join(f"{i}. {label}: {body}" for i, (label, body) in enumerate(zip(labels, bodies), start=1))


def state_text(scene: str, state: Mapping[str, Any], style: Mapping[str, Any]) -> str:
    header = HEADERS[scene][style["header"]].format(wo=style["wo"], site=SITES[scene][style["site"]])
    labels = LABELS[scene][style["labels"]]
    if scene == "indexdial":
        from robo_jev.authors.a3.scenes import DIAL_PARTS

        part = DIAL_PARTS[state["part"]][0]
        build = cat.DIAL_BUILD[state["build"]][0]
        side = "radial" if state["orient"] == "radial" else "along the travel"
        task = ("index the dial one station; part rides loose, no nest.", "advance the dial one station; part stands free on it.")[style["task"]]
        bodies = [
            task,
            f"{part}, {_dims(state['dims_mm'])}, {_kg(state['mass_kg'])} kg, {build}.",
            f"{_int(state['radius_mm'])} mm from the dial centre, long side {side}; {cat.DIAL_PLACEMENT[state['placement']].words}.",
            f"{DIAL_BASE_PHRASE[state['base']]} on the plate; plate {DIAL_PLATE_PHRASE[state['plate']]}.",
            f"{_int(state['angle_deg'])} deg in {_s(state['index_s'])} s, cycloidal cam.",
        ]
    elif scene == "mayostand":
        kit = cat.STAND_SET[state["set"]].words
        release = ("lower it to touch the tray, then let go." if int(state["release_mm"]) == 0
                   else f"let go {_int(state['release_mm'])} mm above the tray.")
        task = ("put the set on the Mayo stand tray; casters locked.", "place the instrument set on the stand tray for the case.")[style["task"]]
        bodies = [
            task,
            f"{cat.STAND_MODEL[state['stand']].words}; tray {_int(state['tray_mm'])} mm long at {_int(state['height_mm'])} mm high; "
            f"base {_int(state['reach_mm'])} mm ahead of the post.",
            f"{kit} in a {_dims(state['dims_mm'])} container along the tray; {cat.STAND_PACKING[state['packing']].words}.",
            f"container centre {_int(state['setdown_mm'])} mm out from the post; {cat.STAND_METHOD[state['method']].words}.",
            release,
        ]
    elif scene == "holddown":
        from robo_jev.authors.a3.scenes import CLAMP_PARTS

        part = CLAMP_PARTS[state["part"]][0]
        task = ("cobot presses a fitting into the side of the part (horizontal push); no end stop behind it.",
                "press a fitting in from the side with the cobot; nothing but friction holds the part.")[style["task"]]
        bodies = [
            task,
            f"{part}, {_dims(state['dims_mm'])}, {_kg(state['mass_kg'])} kg, flat in the nest.",
            f"{cat.CLAMP_NEST[state['nest']].words}.",
            f"pneumatic, set {_int(state['clamp_n'])} N; {cat.CLAMP_PAD[state['pad']].words}; {cat.CLAMP_CONDITION[state['condition']].words}.",
            f"peak {_int(state['press_n'])} N over about {_s(state['press_s'])} s.",
        ]
    else:
        raise ValueError(f"unknown scene {scene!r}")
    return header + "\n" + _fields(labels, bodies)


# --------------------------------------------------------------------------
# questions
# --------------------------------------------------------------------------

_Q = {
    "indexdial": {
        "key": ("{p} Will the part slide 10 mm+, lift 10 deg+ or fall?", "{p} Does the part shift (10 mm+), rock up (10 deg+) or fall?"),
        "severe": ("{p} Will the part fall over?", "{p} Does the part topple?"),
        "score_cf": ("{p} Rate 0-1: part moves if the index took {cf} s.", "{p} 0-1 risk of the part moving with a {cf} s index?"),
        "score_scale": ["0 = no chance", "1 = certain"],
        "direction": ("{p} Index in {cf} s instead: chance the part moves?", "{p} If the same index took {cf} s, the part moving becomes:"),
        "compare": ("{p} Which risks the part moving more?", "{p} More likely to move the part: as written or the change?"),
        "action": ("{p} Safest way to run the index?", "{p} Pick the safest way to index."),
        "outcome": ("{p} Likely end state of the part?", "{p} Part when the dial stops?"),
        "outcome_options": {"stays_put": "", "slides_off_mark": "slides 10 mm+", "rocks_back": "lifts 10 deg+, settles", "falls_over": ""},
    },
    "mayostand": {
        "key": ("{p} At release, will the stand lift off its rear feet or tip?", "{p} Does letting go rock the stand off its rear feet or tip it?"),
        "severe": ("{p} Will the stand tip over?", "{p} Does the stand go over?"),
        "score_cf": ("{p} Rate 0-1: stand lifts or tips with the centre at {cf} mm.", "{p} 0-1 risk of lift or tip if the centre were {cf} mm out?"),
        "score_scale": ["0 = no chance", "1 = certain"],
        "direction": ("{p} Centre at {cf} mm instead: chance of lifting or tipping?", "{p} If the centre went to {cf} mm, lifting or tipping becomes:"),
        "compare": ("{p} Which risks lifting or tipping more?", "{p} More likely to rock or tip the stand: as written or the change?"),
        "action": ("{p} Safest way to finish the set-down?", "{p} Pick the safest way to place the set."),
        "outcome": ("{p} Likely result at release?", "{p} Stand behaviour at release?"),
        "outcome_options": {"stays_level": "", "rocks": "rear feet lift, then settle", "tips_over": ""},
    },
    "holddown": {
        "key": ("{p} Will the part shift 1 mm or more in the clamp?", "{p} Does the press move the part 1 mm or more under the hold-down?"),
        "severe": ("{p} Will the part break free (8 mm or more)?", "{p} Is the part shoved out of the clamp (8 mm+)?"),
        "score_cf": ("{p} Rate 0-1: part shifts with a {cf} N peak.", "{p} 0-1 risk of a 1 mm+ shift if the peak were {cf} N?"),
        "score_scale": ["0 = no chance", "1 = certain"],
        "direction": ("{p} Peak {cf} N instead: chance the part shifts?", "{p} If the peak were {cf} N, a shift becomes:"),
        "compare": ("{p} Which risks the part shifting more?", "{p} More likely to move the part: as written or the change?"),
        "action": ("{p} Safest way to run the press?", "{p} Pick the safest way to press."),
        "outcome": ("{p} Likely result for the part?", "{p} Part after the press?"),
        "outcome_options": {"stays_put": "", "creeps": "moves 1-8 mm", "breaks_free": "moves 8 mm+"},
    },
}
_DIRECTION_OPTIONS = ({"higher": "", "lower": "", "unchanged": ""}, {"higher": "", "lower": "", "unchanged": "about the same"})


def _cf_text(scene: str, value: float) -> str:
    return _s(value) if scene == "indexdial" else _int(value)


def questions(scene: str, style: Mapping[str, Any], *, cf_value: float, options: Sequence[tuple[str, str]], compare: tuple[str, str]) -> dict[str, dict]:
    """The request's questions. `options` = the action options (key, words), the plan first; `compare` = the compared alternative."""
    q = _Q[scene]
    names = QUESTION_NAMES[scene]
    variant = style["q"]
    prefix_template = PREFIXES[style["prefix"]]
    cf = _cf_text(scene, cf_value)
    out: dict[str, dict] = {}
    for n, kind in enumerate(KINDS, start=1):
        p = prefix_template.format(n=n)
        text = q[kind][variant[kind]].format(p=p, cf=cf)
        if kind in ("key", "severe"):
            out[names[kind]] = {"type": "noul", "instructions": text}
        elif kind == "score_cf":
            out[names[kind]] = {"type": "score", "instructions": text, "criteria": list(q["score_scale"])}
        elif kind == "direction":
            out[names[kind]] = {"type": "choice", "instructions": text, "criteria": dict(_DIRECTION_OPTIONS[variant[kind]])}
        elif kind == "compare":
            out[names[kind]] = {"type": "choice", "instructions": text,
                                "criteria": {PLAN_OPTION: "", compare[0]: compare[1], SAME_OPTION["compare"]: ""}}
        elif kind == "action":
            out[names[kind]] = {"type": "choice", "instructions": text,
                                "criteria": {key: ("" if key == PLAN_OPTION else words) for key, words in options}}
        else:
            out[names[kind]] = {"type": "choice", "instructions": text, "criteria": dict(q["outcome_options"])}
    return out
