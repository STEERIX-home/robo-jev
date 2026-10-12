"""Task E1 conditions C and D — T8b moments in the engine's format (C) and in their own format (D), same questions.

**C** renders every T8b moment **deterministically from ``meta.visible``** (no LLM, no randomness) in the engine's *prose* style
(:func:`robo_jev.simgen.narrate._render_prose`): a setting line, "You are …" identity, then the robot and load sentences in the
second person, with the engine's sentence patterns and units —

* load: "It is 47 cm tall with a base 19 cm deep …" (load sizes, overhangs, heights and drops in cm, as the engine's load and stack
  facts write them); speeds in m/s, accelerations in m/s², times in s;
* braking: "Your brakes give …"; turning: "a 90° turn to the left with a radius of …"; stacking: "You are holding … over …, about to
  set it down", "your placement is accurate to about … (one standard deviation)", "When you lifted it, … felt heavier toward …";
* questions in the engine's second-person templates ("If you {action}, will {load} tip over?", "How likely is {load} to … if you
  {action}?") with the action's number in brackets as the engine writes its options ("brake as planned (3.89 m/s²)").

Every T8b fact is kept — surface, weight distribution, jerk time, the free deck, the release height, the push travel and the
back-away — and nothing is added except the engine's opening setting line, which states ``meta.visible["domain"]`` (a T8b visible
value that T8b's own text never prints). :func:`check_c` tests that every ``meta.visible`` value appears (numbers by exact unit
conversion; identifiers through their phrase) and that every number of D's text appears in C's text (same rounding).

**Questions.** Only the T8b question kinds that map 1:1 to an engine yes/no or score about tipping / falling — the s1score module's
:data:`robo_jev.bench.system1.T8B_EVENTS` (deck: planned tip-over yes/no + tip-over score at another braking rate or curve speed;
stack: falls off yes/no + falls-off score further out + falls-off yes/no further in; push: tip-over yes/no, at another height, and
the score at a third height). ``keeps_place`` (stays put: neither slides 2 cm nor tips) and the stays/slides/topples choice have
no 1:1 engine counterpart and are left out. Labels and question semantics are T8b's, unchanged.

**D** is ``artifacts/t8/T8b.jsonl`` restricted to the same questions (ids, text and labels unchanged).

C and D are T8b-derived and sealed: they are scored locally only and never sent to any API.
"""

from __future__ import annotations

import copy
import math
import re
from typing import Any

__all__ = ["C_PREFIX", "KEPT_QUESTIONS", "check_c", "quantities", "render_c", "restrict_d"]

C_PREFIX = "e1c"

#: T8b questions kept in C and D (= robo_jev.bench.system1.T8B_EVENTS; a test checks they agree).
KEPT_QUESTIONS: dict[str, tuple[str, ...]] = {
    "deck_brake": ("falls_over", "falls_other_braking"),
    "deck_turn": ("falls_over", "falls_other_speed"),
    "stack": ("falls_off", "falls_if_further_out", "falls_if_further_in"),
    "push": ("tips_over", "tips_at_other_height", "tip_risk_other_height"),
}

#: The engine's opening setting line per T8b domain (states the visible ``domain``).
SETTINGS = {"warehouse": "A warehouse.", "hospital": "A hospital.", "office": "An office.", "home": "A private home.",
            "retail": "A retail store.", "lab": "A lab."}
#: T8b phrases behind the visible identifiers (the same words T8b's text uses; first-person ones put into the second person).
DECK_SURFACES = {"rubber_mat": "a ribbed rubber mat", "painted_steel": "bare painted steel", "plastic_liner": "a smooth plastic liner",
                 "plywood": "plain plywood", "foam_pad": "an anti-slip foam pad"}
FILLS = {"even": "Its weight looks evenly spread.", "unknown": "You cannot tell how its contents are arranged.",
         "bottom_heavy": "Its heaviest contents sit near the bottom.", "top_heavy": "It is heavier near the top than near the bottom.",
         "planter": "The pot is full of damp soil, so most of the weight is low down."}
STACK_SPREADS = {"even": "When you lifted it, its weight felt evenly spread from end to end.",
                 "outer_heavy": "When you lifted it, it felt heavier toward the end that will stick out.",
                 "inner_heavy": "When you lifted it, it felt heavier toward the end that will rest on the top.",
                 "unknown": "Its contents slide around, so the weight could sit toward either end."}
PUSH_FLOORS = {"concrete": "polished concrete", "vinyl": "smooth vinyl flooring", "rubber": "rubber floor matting", "carpet": "low-pile carpet",
               "epoxy": "an epoxy-coated floor"}
PUSH_SPREADS = {"even": "Its weight seems evenly spread.", "unknown": "You do not know how the weight is arranged inside.",
                "far_heavy": "It is heavier on the side facing away from you.", "near_heavy": "It is heavier on the side facing you.",
                "bottom_heavy": "Its heaviest part is at the bottom."}
#: Constant facts of T8b's text (not in ``meta.visible``): the push robot backs away this far.
PUSH_BACK_AWAY_M = 0.30


# --------------------------------------------------------------------------
# Numbers
# --------------------------------------------------------------------------


def _num(value: float, max_decimals: int = 3) -> str:
    """Shortest exact decimal (up to ``max_decimals``) — 0.19 → '0.19', 19.000000000000004 → '19', 9.5 → '9.5'."""
    for decimals in range(max_decimals + 1):
        text = f"{value:.{decimals}f}"
        if abs(float(text) - value) < 1e-9:
            return text
    return f"{value:.{max_decimals}f}"


def _cm(metres: float) -> str:
    return f"{_num(round(metres * 100.0, 6), 2)} cm"


def _ms(value: float) -> str:
    return f"{value:.2f} m/s"


def _acc(value: float) -> str:
    return f"{value:.2f} m/s²"


def _s(value: float) -> str:
    return f"{_num(value, 3)} s" if round(value, 2) != value else f"{value:.2f} s"


def _kg(value: float) -> str:
    return f"{value:.1f} kg"


def _m(value: float) -> str:
    return f"{value:.2f} m"


def _a(name: str) -> str:
    return ("an " if name[0].lower() in "aeiou" else "a ") + name


def _the(name: str) -> str:
    """'the …' reference, closed with a comma when the name itself contains one ('the tall square glass bottle, full and capped,')."""
    return f"the {name}," if "," in name else f"the {name}"


# --------------------------------------------------------------------------
# C: states and questions
# --------------------------------------------------------------------------


def _deck(v: dict[str, Any], family: str) -> tuple[str, dict[str, dict[str, Any]]]:
    length, width, height = v["dims"]
    name = v["load_name"]
    first = [SETTINGS[v["domain"]], f"You are {_a(v['robot'])}.", f"You are moving straight ahead at {_ms(v['speed'])}."]
    if family == "deck_brake":
        stop_time = v["speed"] / v["decel"]
        first.append(f"You are about to brake to a standstill: your brakes give {_acc(v['decel'])}, reached within {_s(v['jerk_time'])} and "
                     f"held until you stand still; the stop takes about {stop_time:.2f} s.")
    first.append(f"In the middle of your deck, on {DECK_SURFACES[v['surface']]}, stands {_a(name)}. It is box-shaped, {_cm(height)} tall with a base "
                 f"{_cm(length)} deep (front to back) and {_cm(width)} wide (side to side), and weighs about {_kg(v['mass'])}. {FILLS[v['fill']]}")
    second = ["Your deck is clear and wide enough on every side that it cannot slide off an edge, and there is nothing it could lean against."]
    if family == "deck_turn":
        second.append(f"Next you reach a {int(v['angle_deg'])}° turn to the {v['direction']} with a radius of {_m(v['radius'])}, which you take without "
                      f"slowing down; the steering eases in over the first {_s(v['steer_ramp'])} and eases out over the last {_s(v['steer_ramp'])}.")
    state = " ".join(first) + "\n\n" + " ".join(second)
    ref = _the(name)
    tip = ["0 = it certainly stays upright", "1 = it certainly tips over"]
    if family == "deck_brake":
        questions = {
            "falls_over": {"type": "noul", "instructions": f"If you brake as planned ({_acc(v['decel'])}), will {ref} tip over?"},
            "falls_other_braking": {"type": "score", "instructions": f"How likely is {ref} to tip over if you brake from the same speed at "
                                    f"{_acc(v['other_decel'])} instead?", "criteria": tip},
        }
    else:
        questions = {
            "falls_over": {"type": "noul", "instructions": f"If you take the turn without slowing ({_ms(v['speed'])}), will {ref} tip over?"},
            "falls_other_speed": {"type": "score", "instructions": f"How likely is {ref} to tip over if you take the same turn at {_ms(v['other_speed'])} "
                                  "instead?", "criteria": tip},
        }
    return state, questions


def _overhang(overhang: float) -> str:
    if overhang > 0.0005:
        return f"its far end sticking out {_cm(overhang)} past the edge"
    if overhang < -0.0005:
        return f"its far end {_cm(-overhang)} short of the edge"
    return "its far end exactly at the edge"


def _stack(v: dict[str, Any]) -> tuple[str, dict[str, dict[str, Any]]]:
    length, width, height = v["dims"]
    item, support, short = v["item_name"], v["support_name"], v["support_short"]
    first = [SETTINGS[v["domain"]], f"You are {_a(v['robot'])}.", f"You are holding {_a(item)} by its top over {support}, about to set it down."]
    second = [f"The flat top of the {short} is {_cm(v['support_top'])} long and {_cm(v['support_height'])} above the floor; the {item} is {_cm(length)} long, "
              f"{_cm(width)} wide and {_cm(height)} tall, and weighs about {_kg(v['mass'])}. {STACK_SPREADS[v['spread']]}",
              f"You plan to lay it lengthwise along that top, square to the edge, with {_overhang(v['overhang'])} and the rest of it on the top. "
              f"Your gripper opens when its underside is about {_cm(v['drop'])} above the top, and your placement is accurate to about {_cm(v['sigma'])} "
              "(one standard deviation)."]
    state = " ".join(first) + "\n\n" + " ".join(second)
    ref = _the(item)
    questions = {
        "falls_off": {"type": "noul", "instructions": f"If you set it down as planned, will {ref} fall off the {short}?"},
        "falls_if_further_out": {"type": "score", "instructions": f"How likely is {ref} to fall off the {short} if you set it down with "
                                 f"{_overhang(v['overhang_out'])} instead?", "criteria": ["0 = it certainly stays on the top", "1 = it certainly falls off"]},
        "falls_if_further_in": {"type": "noul", "instructions": f"If you set it down with {_overhang(v['overhang_in'])} instead, will {ref} fall off the "
                                f"{short}?"},
    }
    return state, questions


def _push(v: dict[str, Any]) -> tuple[str, dict[str, dict[str, Any]]]:
    length, width, height = v["dims"]
    box = v["box_name"]
    first = [SETTINGS[v["domain"]], f"You are {_a(v['robot'])} with a smooth, rigid plastic bumper bar across your front."]
    second = [f"{_a(box)[0].upper() + _a(box)[1:]} stands ahead of you on {PUSH_FLOORS[v['floor']]}. It is box-shaped, {_cm(length)} deep in the direction "
              f"you push, {_cm(width)} wide and {_cm(height)} tall, and weighs about {_kg(v['mass'])}. {PUSH_SPREADS[v['spread']]}",
              f"You plan to push it {_cm(v['travel'])} forward at {_ms(v['speed'])}, with the bar pressing on its near face {_cm(v['push_height'])} above "
              f"the floor, and then back away {_cm(PUSH_BACK_AWAY_M)}."]
    state = " ".join(first) + "\n\n" + " ".join(second)
    ref = _the(box)
    questions = {
        "tips_over": {"type": "noul", "instructions": f"If you push as planned and then back away, will {ref} end up tipped over?"},
        "tips_at_other_height": {"type": "noul", "instructions": f"If the bar met it {_cm(v['push_height_noul'])} above the floor instead (the same push, "
                                 f"then backing away), will {ref} end up tipped over?"},
        "tip_risk_other_height": {"type": "score", "instructions": f"How likely is {ref} to end up tipped over if the bar met it "
                                  f"{_cm(v['push_height_score'])} above the floor instead?",
                                  "criteria": ["0 = it certainly ends up standing on its base", "1 = it certainly ends up tipped over"]},
    }
    return state, questions


def render_c(record: dict[str, Any]) -> dict[str, Any]:
    """T8b record → its C record (engine prose, the kept questions, T8b's labels). Deterministic: same input, same bytes."""
    meta = record["meta"]
    family = meta["scene"]
    v = meta["visible"]
    if family in ("deck_brake", "deck_turn"):
        state, questions = _deck(v, family)
    elif family == "stack":
        state, questions = _stack(v)
    elif family == "push":
        state, questions = _push(v)
    else:
        raise ValueError(f"unknown T8b family {family!r}")
    kept = KEPT_QUESTIONS[family]
    if tuple(questions) != kept:
        raise AssertionError(f"{record['id']}: rendered questions {list(questions)} != kept {list(kept)}")
    for name in kept:
        if record["request"]["questions"][name]["type"] != questions[name]["type"]:
            raise AssertionError(f"{record['id']}: {name} type changed")
    out = {
        "schema": record["schema"],
        "id": f"{C_PREFIX}-{record['id']}",
        "generator": {"name": "e1-c-render-v1", "source": record["id"], "source_generator": record.get("generator")},
        "meta": {**copy.deepcopy(meta), "e1": {"condition": "C", "source_id": record["id"], "questions": list(kept)}},
        "request": {"state": state, "questions": questions},
        "labels": {name: copy.deepcopy(record["labels"][name]) for name in kept},
    }
    return out


def restrict_d(record: dict[str, Any]) -> dict[str, Any]:
    """T8b record → its D record (the kept questions only; id, text and labels unchanged)."""
    kept = KEPT_QUESTIONS[record["meta"]["scene"]]
    out = copy.deepcopy(record)
    out["request"]["questions"] = {n: out["request"]["questions"][n] for n in kept}
    out["labels"] = {n: out["labels"][n] for n in kept}
    out["meta"]["e1"] = {"condition": "D", "source_id": record["id"], "questions": list(kept)}
    return out


# --------------------------------------------------------------------------
# Checks
# --------------------------------------------------------------------------

#: number + unit → (dimension, factor to SI)
_UNITS = {"m/s²": ("acc", 1.0), "m/s": ("speed", 1.0), "km/h": ("speed", 1 / 3.6), "cm": ("length", 0.01), "mm": ("length", 0.001), "m": ("length", 1.0),
          "kg": ("mass", 1.0), "ms": ("time", 0.001), "s": ("time", 1.0), "°": ("angle", 1.0)}
_QTY = re.compile(r"(?<![\w.])(-?\d+(?:\.\d+)?)\s?(m/s²|m/s|km/h|cm|mm|ms|kg|°|m|s)(?![\w/²])")


def quantities(text: str) -> list[tuple[str, float, int, str]]:
    """Number-with-unit tokens → (dimension, SI value, decimals as printed, unit)."""
    out = []
    for number, unit in _QTY.findall(text):
        dim, factor = _UNITS[unit]
        decimals = len(number.split(".")[1]) if "." in number else 0
        out.append((dim, float(number) * factor, decimals, unit))
    return out


def _texts(record: dict[str, Any]) -> str:
    request = record["request"]
    parts = [request["state"]]
    for q in request["questions"].values():
        parts.append(q["instructions"])
        if isinstance(q.get("criteria"), list):
            parts.extend(q["criteria"])
    return "\n".join(parts)


def _has_quantity(found: list[tuple[str, float, int, str]], dim: str, si: float) -> bool:
    return any(d == dim and math.isclose(v, si, rel_tol=1e-9, abs_tol=1e-9) for d, v, _, _ in found)


def check_c(c: dict[str, Any], t8b: dict[str, Any]) -> list[str]:
    """Problems with a C record (empty = passes): (1) every ``meta.visible`` value appears in C's text — numbers by exact unit
    conversion, identifiers through their phrase; (2) every number-with-unit of D's text (the kept questions) appears in C's text
    at D's printed precision; (3) same question names, types and labels as T8b."""
    problems: list[str] = []
    text = _texts(c)
    low = text.lower()
    found = quantities(text)
    v = t8b["meta"]["visible"]
    family = t8b["meta"]["scene"]
    numeric = {"dims": "length", "mass": "mass", "speed": "speed", "decel": "acc", "jerk_time": "time", "other_decel": "acc", "radius": "length",
               "angle_deg": "angle", "steer_ramp": "time", "other_speed": "speed", "support_top": "length", "support_height": "length", "sigma": "length",
               "drop": "length", "travel": "length", "push_height": "length", "push_height_noul": "length", "push_height_score": "length"}
    for key, value in v.items():
        if key in numeric:
            for x in (value if isinstance(value, list) else [value]):
                if not _has_quantity(found, numeric[key], float(x)):
                    problems.append(f"visible {key}={x} not in the text")
        elif key in ("overhang", "overhang_out", "overhang_in"):
            phrase = _overhang(float(value))
            if phrase.lower() not in low:
                problems.append(f"visible {key}={value} ('{phrase}') not in the text")
        elif key in ("robot", "load_name", "item_name", "box_name", "support_name", "support_short", "direction"):
            if str(value).lower() not in low:
                problems.append(f"visible {key}={value!r} not in the text")
        elif key == "domain":
            if SETTINGS[value].lower().rstrip(".") not in low or str(value).lower() not in low:
                problems.append(f"visible domain={value!r} not in the text")
        elif key == "shape":
            if f"{value}-shaped" not in low:
                problems.append(f"visible shape={value!r} not in the text")
        elif key in ("load", "item", "box", "support"):
            name = v.get({"load": "load_name", "item": "item_name", "box": "box_name", "support": "support_name"}[key])
            if not name or str(name).lower() not in low:
                problems.append(f"visible {key}={value!r}: its name is not in the text")
        elif key == "surface":
            if DECK_SURFACES[value].lower() not in low:
                problems.append(f"visible surface={value!r} not in the text")
        elif key == "fill":
            if FILLS[value].lower() not in low:
                problems.append(f"visible fill={value!r} not in the text")
        elif key == "spread":
            phrase = (STACK_SPREADS if family == "stack" else PUSH_SPREADS)[value]
            if phrase.lower() not in low:
                problems.append(f"visible spread={value!r} not in the text")
        elif key == "floor":
            if PUSH_FLOORS[value].lower() not in low:
                problems.append(f"visible floor={value!r} not in the text")
        else:
            problems.append(f"visible {key!r} has no check")
    d = restrict_d(t8b)
    for dim, si, decimals, unit in quantities(_texts(d)):
        factor = _UNITS[unit][1]
        if not any(dd == dim and round(vv / factor, decimals) == round(si / factor, decimals) for dd, vv, _, _ in found):
            problems.append(f"D's {round(si / factor, decimals)} {unit} has no counterpart in C")
    if list(c["request"]["questions"]) != list(d["request"]["questions"]):
        problems.append("question names differ from D")
    for name, q in c["request"]["questions"].items():
        if q["type"] != d["request"]["questions"][name]["type"]:
            problems.append(f"{name}: type differs from D")
    if c["labels"] != d["labels"]:
        problems.append("labels differ from D")
    if re.search(r"\b(I|I'm|me|my|mine)\b", text):
        problems.append("first-person words in C (the engine writes in the second person)")
    return problems
