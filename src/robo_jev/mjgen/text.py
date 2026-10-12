"""v0.3b text — four narration styles and the question wording (Task v0.3b §1.2), written for this task.

Nothing here is taken from T8b's wording or from :mod:`robo_jev.simgen` phrase banks (a test parses the imports; the overlap check
in :mod:`robo_jev.mjgen.overlap` compares sentences and word n-grams with T8b and finds none shared at five words or more). Styles:

* ``log`` — an operations log in the third person ("Unit: …; Cargo: …; Packing note: …");
* ``brief`` — a second-person task briefing ("You are …; your deck …");
* ``sheet`` — key–value lines;
* ``self`` — the robot speaking in the first person (its own sentences, not T8b's).

Every number the physics needs is written with its unit, once; hidden values (friction, where the weight sits inside, the exact
mass of an unknown fill, placement errors) are never written — only the words that set their prior. A paired record (one changed
number) is rendered from the same wording seed, so its text differs from its partner's only in that number.

Questions favour System-1 judgments (controller update 2026-10-12): which action is safest (``choice``, action), direction of change
("more likely, less likely or about the same", ``choice``), comparisons of two actions (``choice``), what happens (``choice``,
outcome), plus the key yes/no question, a 0–1 rating with both anchors written out, and a yes/no on the severe outcome. No question
asks for a number.
"""

from __future__ import annotations

import random
from typing import Any

from robo_jev.mjgen.families import GRIP_JERK, GRIP_PADS, GRIP_SURFACES, TILT_SURFACES

__all__ = ["DIRECTION_KEYS", "OPTION_TEXT", "questions", "state_text"]

DIRECTION_KEYS = ("more_likely", "less_likely", "about_same")
_DIRECTION = {"more_likely": "more likely than with the plan", "less_likely": "less likely than with the plan",
              "about_same": "about the same (the chance changes by less than 0.1)"}


def _only(options: dict[str, str], plan: dict[str, Any]) -> dict[str, str]:
    """The options the record's safest-action question offers (``plan["action_keys"]``; all of them when absent)."""
    keys = plan.get("action_keys")
    return dict(options) if keys is None else {k: v for k, v in options.items() if k in keys}


def _n(value: float, digits: int = 2) -> str:
    return f"{value:.{digits}f}"


def _cm(value: float) -> str:
    cm = round(value * 100, 1)
    return f"{cm:.0f} cm" if abs(cm - round(cm)) < 1e-9 else f"{cm:.1f} cm"


def _a(word: str) -> str:
    return ("an " if word[0].lower() in "aeiou" else "a ") + word


# --------------------------------------------------------------------------
# tilt
# --------------------------------------------------------------------------

TILT_FILL_TEXT = {
    "level": ("its contents are spread evenly", "its contents fill it uniformly"),
    "low": ("its heavy contents sit at the bottom", "its weight is concentrated low down"),
    "high": ("the bulk of its load rides up near the top", "its heaviest contents ride near the top"),
    "unsure": ("its packing is unknown", "no one has recorded how it was packed"),
    "fore": ("its heavier end faces the front of the deck", "its weight is biased toward the deck's front"),
    "aft": ("its heavier end faces the rear of the deck", "its weight is biased toward the deck's rear"),
}


def _tilt_plan(v: dict[str, Any], person: str) -> str:
    angle, dur = f"{v['angle']:.1f}°", f"{v['duration']:.1f} s"
    if v["storyline"] == "bed":
        way = "rises" if v["direction"] == "up" else "dips"
        subject = {"third": "The bed", "second": "Your bed", "first": "My bed"}[person]
        return f"{subject} is set to tilt so that its front end {way}: the angle goes from level to {angle} over {dur} and is then held there."
    way = "climbing" if v["direction"] == "up" else "going down"
    subject = {"third": "The unit is", "second": "You are", "first": "I'm"}[person]
    deck = {"third": "its deck", "second": "your deck", "first": "my deck"}[person]
    nose = "front up" if v["direction"] == "up" else "front down"
    return (f"{subject} heading onto a ramp, {way}: as the wheels roll onto it, {deck} pitches {nose} from level to {angle} within {dur}, "
            "and keeps that pitch on the slope.")


def _tilt_size(v: dict[str, Any]) -> str:
    length, width, height = v["dims"]
    return f"{_n(height)} m high on a base {_n(length)} m long (front–rear) and {_n(width)} m across"


def _tilt_state(v: dict[str, Any], style: str, rng: random.Random) -> str:
    length, width, height = v["dims"]
    fill = rng.choice(TILT_FILL_TEXT[v["fill"]])
    surface = TILT_SURFACES[v["surface"]]["text"] + (", still wet from mopping" if v.get("wet") else "")
    size = _tilt_size(v)
    if style == "log":
        return (f"Log, {v['domain']} floor. Unit: {_a(v['robot'])}. Cargo: one {v['load_name']} standing upright and unstrapped at the centre of "
                f"the deck; {size}; mass {v['mass']:.1f} kg. Packing note: {fill}. Deck covering: {surface}. The deck runs far beyond the cargo on "
                f"all sides, with no wall or neighbour within reach. {_tilt_plan(v, 'third')}")
    if style == "brief":
        return (f"You are {_a(v['robot'])} in {_a(v['domain'])}. {_a(v['load_name']).capitalize()} stands upright and loose at the centre of your "
                f"deck: {size}, {v['mass']:.1f} kg. Packing: {fill}. Your deck is covered with {surface}; no strap holds the load, nothing stands "
                f"near it, and the deck extends well past it all round. {_tilt_plan(v, 'second')}")
    if style == "sheet":
        story = "tilting bed" if v["storyline"] == "bed" else "ramp"
        return "\n".join((
            f"robot: {v['robot']} ({v['domain']})", f"load: {v['load_name']}, upright, unstrapped, centred on the deck",
            f"load height: {_n(height)} m; base: {_n(length)} m front–rear × {_n(width)} m across", f"load mass: {v['mass']:.1f} kg",
            f"packing: {fill}", f"deck surface: {surface}", "deck clearance: ample on every side, no walls or neighbours",
            f"manoeuvre: {story} — {_tilt_plan(v, 'third')}",
        ))
    return (f"This is the {v['robot']} reporting from the {v['domain']}. A {v['load_name']} rides loose and upright at the centre of my deck, "
            f"{size}, weighing {v['mass']:.1f} kg. As for its packing: {fill}. My deck surface is {surface}; the deck reaches well past the load "
            f"on every side and nothing stands beside it. {_tilt_plan(v, 'first')}")


def _tilt_questions(v: dict[str, Any], mix: list[str], plan: dict[str, Any], person: str, rng: random.Random, cf_option: str) -> dict[str, Any]:
    load = v["load_name"]
    angle, cf = f"{v['angle']:.1f}°", f"{plan['cf_value']:.1f}°"
    deck = {"third": "the deck", "second": "your deck", "first": "my deck"}[person]
    out: dict[str, Any] = {}
    out["moves_at_plan"] = {"type": "noul", "instructions": rng.choice((
        f"Once {deck} reaches {angle}, does the {load} slide 3 cm or more, or topple over?",
        f"At {angle} of pitch, will the {load} shift (3 cm or more along the deck) or fall over?"))}
    out["pitch_change_effect"] = {"type": "choice", "instructions": rng.choice((
        f"Suppose {deck} pitched to {cf} instead of {angle}. Would the {load} then be more likely, less likely, or about as likely to slide or topple?",
        f"Compared with the plan of {angle}, what would a pitch of {cf} do to the risk that the {load} slides or topples?")),
        "criteria": dict(_DIRECTION)}
    if "outcome" in mix:
        out["load_after_tilt"] = {"type": "choice", "instructions": f"What happens to the {load} as {deck} goes to {angle} and holds there?",
                                  "criteria": {"stays": "no tipping and under 3 cm of movement", "slides": "a slide of 3 cm or more, still upright",
                                               "topples": "tips onto a side"}}
    if "action" in mix:
        out["safest_way_to_carry"] = {"type": "choice", "instructions": f"For this {angle} pitch, which way of carrying the {load} is safest?",
                                      "criteria": _only(OPTION_TEXT["tilt"], plan)}
    if "compare" in mix:
        laid = OPTION_TEXT["tilt"][cf_option].replace("lay it", "laid")
        out["riskier_orientation"] = {"type": "choice", "instructions": f"At {angle}, which is more likely to slide or topple: the {load} as it stands now, "
                                      f"or the {load} {laid}?",
                                      "criteria": {"as_it_stands": "the upright load, as planned", cf_option: "the re-oriented load",
                                                   "about_same": "about the same (within 0.1)"}}
    if "score_cf" in mix:
        out["risk_at_other_pitch"] = {"type": "score", "instructions": f"Rate the chance that the {load} slides or topples if the pitch were {cf} "
                                      f"rather than {angle}.", "criteria": ["0 = it will not move at all", "1 = it is sure to slide or topple"]}
    if "severe" in mix:
        out["topples_at_plan"] = {"type": "noul", "instructions": f"At {angle}, does the {load} actually tip over (not just slide)?"}
    return out


# --------------------------------------------------------------------------
# grip
# --------------------------------------------------------------------------


def _mass_text(v: dict[str, Any]) -> str:
    if v["mass_kind"] == "about":
        return f"about {v['mass']:.2f} kg"
    return f"somewhere between {v['mass_lo']:.2f} kg and {v['mass_hi']:.2f} kg, depending on how full it is (unknown)"


def _grip_move(v: dict[str, Any], person: str) -> str:
    subject = {"third": "The arm", "second": "You", "first": "I"}[person]
    verb = {"lift": "lift it straight up", "lower": "lower it straight down", "swing": "carry it sideways, level"}[v["move"]]
    jerk = v.get("jerk_time", GRIP_JERK)
    return (f"{subject} will {verb} by {_n(v['distance'])} m with a top speed of {v['peak']:.1f} m/s, speeding up and braking at "
            f"{v['accel']:.1f} m/s², each time taking {jerk:.2f} s to get to that rate.")


def _grip_size(v: dict[str, Any]) -> str:
    length, width, height = v["dims"]
    return f"{_n(height)} m tall, {_n(length)} m in length and {_n(width)} m across the grip"


def _grip_state(v: dict[str, Any], style: str, rng: random.Random) -> str:
    length, width, height = v["dims"]
    pads = GRIP_PADS[v["pads"]]["text"]
    surface = GRIP_SURFACES[v["surface"]]["text"] + (", damp to the touch" if v.get("wet") else "")
    size = _grip_size(v)
    force = f"{v['squeeze']:.1f} N"
    if style == "log":
        return (f"Log, {v['domain']}. Unit: {_a(v['robot'])} fitted with {pads}. Held item: {_a(v['item_name'])} ({size}), its surface "
                f"{surface}, mass {_mass_text(v)}. Grasp: pads centred on its two side faces, clamping force {force}. Next move: {_grip_move(v, 'third')}")
    if style == "brief":
        return (f"You are {_a(v['robot'])} with {pads}, holding {_a(v['item_name'])} by its sides, pads centred on them. The item is {size}, "
                f"covered in {surface}, and weighs {_mass_text(v)}. You clamp it with {force}. {_grip_move(v, 'second')}")
    if style == "sheet":
        return "\n".join((
            f"robot: {v['robot']} ({v['domain']})", f"finger pads: {pads}", f"item: {v['item_name']}",
            f"item height / length / width across the grip: {_n(height)} m / {_n(length)} m / {_n(width)} m", f"item surface: {surface}",
            f"item mass: {_mass_text(v)}", "grasp: side faces, pads centred", f"clamping force: {force}", f"move: {_grip_move(v, 'third')}",
        ))
    return (f"Here is my situation in the {v['domain']}: as {_a(v['robot'])} with {pads}, I have {_a(v['item_name'])} pinched by its side "
            f"faces, pads centred. It measures {size}, its outside is {surface}, and it weighs {_mass_text(v)}. My clamping force is {force}. "
            f"{_grip_move(v, 'first')}")


def _grip_options(v: dict[str, Any], plan: dict[str, Any]) -> dict[str, str]:
    opts = plan["option_values"]
    return {"as_planned": f"clamp at {v['squeeze']:.1f} N and move at {v['accel']:.1f} m/s², as planned",
            "firmer_faster": f"clamp at {opts['opt_firm_fast']['squeeze']:.1f} N but move at {opts['opt_firm_fast']['accel']:.1f} m/s²",
            "lighter_slower": f"clamp at {opts['opt_light_slow']['squeeze']:.1f} N and move at only {opts['opt_light_slow']['accel']:.1f} m/s²"}


def _grip_questions(v: dict[str, Any], mix: list[str], plan: dict[str, Any], person: str, rng: random.Random, cf_option: str) -> dict[str, Any]:
    item = v["item_name"]
    move = {"lift": "lift", "lower": "lowering", "swing": "sideways carry"}[v["move"]]
    hold = {"third": "the fingers", "second": "your fingers", "first": "my fingers"}[person]
    force, cf = f"{v['squeeze']:.1f} N", f"{plan['cf_value']:.1f} N"
    out: dict[str, Any] = {}
    out["loses_grip"] = {"type": "noul", "instructions": rng.choice((
        f"During this {move}, does the {item} slip 1 cm or more between {hold}, or fall out?",
        f"Will the {item} shift 1 cm or more in {hold} (or drop) during the {move}?"))}
    out["force_change_effect"] = {"type": "choice", "instructions": rng.choice((
        f"If the clamping force were {cf} instead of {force}, would the {item} be more likely, less likely or about as likely to slip or drop?",
        f"Compared with clamping at {force}, how would {cf} change the risk that the {item} slips or drops during the {move}?")),
        "criteria": dict(_DIRECTION)}
    if "outcome" in mix:
        out["item_after_move"] = {"type": "choice", "instructions": f"How does the {item} end up after the {move}?",
                                  "criteria": {"held": "still held, shifted under 1 cm", "slipped": "slipped 1 cm or more but still held",
                                               "dropped": "fell out of the grip"}}
    if "action" in mix:
        out["safest_grip_plan"] = {"type": "choice", "instructions": f"Which of these plans is safest for the {item}?",
                                   "criteria": _only(_grip_options(v, plan), plan)}
    if "compare" in mix:
        options = _grip_options(v, plan)
        out["riskier_grip_plan"] = {"type": "choice", "instructions": f"Which is more likely to make the {item} slip or drop: the plan as it stands "
                                    f"({options['as_planned']}), or the other plan ({options[cf_option]})?",
                                    "criteria": {"as_planned": "the plan as it stands", cf_option: "the other plan", "about_same": "about the same (within 0.1)"}}
    if "score_cf" in mix:
        out["slip_risk_other_force"] = {"type": "score", "instructions": f"Rate the chance of the {item} slipping or dropping if the clamping force "
                                        f"were {cf} rather than {force}.", "criteria": ["0 = the grip will not give at all", "1 = it is sure to slip or drop"]}
    if "severe" in mix:
        out["falls_out"] = {"type": "noul", "instructions": f"Does the {item} fall out of {hold} during the {move}?"}
    return out


# --------------------------------------------------------------------------
# tower
# --------------------------------------------------------------------------

TOWER_CONTENTS_TEXT = {
    "even": "packed evenly", "shifted": "its contents have slid toward one end, which end unknown",
    "front": "heavier at its front end", "back": "heavier at its back end",
}


def _offset_text(offset: float) -> str:
    if abs(offset) < 1e-9:
        return "centred on the box below it"
    if offset > 0:
        return f"centred {_cm(offset)} toward the front of the box below it"
    return f"centred {_cm(-offset)} toward the back of the box below it"


def _tower_state(v: dict[str, Any], style: str, rng: random.Random) -> str:
    boxes = v["boxes"]
    n = len(boxes)
    lines = []
    for j, b in enumerate(boxes):
        length, width, height = b["dims"]
        where = "on the floor" if j == 0 else _offset_text(b["offset"])
        lines.append((j, b, f"{_n(length)} m front–back by {_n(width)} m by {_n(height)} m high", where))
    release = _cm(v["drop"])
    sigma = _cm(v["sigma"])
    if style == "log":
        parts = [f"Log, {v['domain']}. Unit: {_a(v['robot'])}, building a stack of {n} boxes, bottom to top."]
        for j, b, size, where in lines:
            parts.append(f"Box {j + 1}: {b['name']}, {size}, {b['mass']:.1f} kg, {TOWER_CONTENTS_TEXT[b['contents']]}; {where}.")
        parts.append(f"The lower boxes are already in place; the top box drops from {release} up onto the box under it. Placement scatter: "
                     f"typically ±{sigma}. Front = the side facing the aisle.")
        return " ".join(parts)
    if style == "brief":
        parts = [f"You are {_a(v['robot'])} in {_a(v['domain'])}, stacking {n} boxes on the floor."]
        for j, b, size, where in lines:
            parts.append(f"{'At the bottom' if j == 0 else ('On top' if j == n - 1 else 'Next')} is {_a(b['name'])} ({size}, {b['mass']:.1f} kg, "
                         f"{TOWER_CONTENTS_TEXT[b['contents']]}), {where}.")
        parts.append(f"Everything below the top box is already placed; the top box drops from {release} up onto the one beneath it, and you "
                     f"typically put a box down up to {sigma} off target. 'Front' means the aisle side.")
        return " ".join(parts)
    if style == "sheet":
        out = [f"robot: {v['robot']} ({v['domain']})", f"task: stack of {n} boxes on the floor (front = aisle side)"]
        for j, b, size, where in lines:
            out.append(f"box {j + 1} ({'bottom' if j == 0 else ('top' if j == n - 1 else 'middle')}): {b['name']}; {size}; {b['mass']:.1f} kg; "
                       f"{TOWER_CONTENTS_TEXT[b['contents']]}; {where}")
        out += [f"release: top box dropped from {release} up onto the box below; lower boxes already in place", f"placement scatter: typically ±{sigma}"]
        return "\n".join(out)
    parts = [f"I'm {_a(v['robot'])} working in the {v['domain']}, finishing a stack of {n} boxes on the floor (front means the aisle side)."]
    for j, b, size, where in lines:
        parts.append(f"{'Lowest' if j == 0 else ('Uppermost' if j == n - 1 else 'In the middle')}: {_a(b['name'])}, {size}, {b['mass']:.1f} kg, "
                     f"{TOWER_CONTENTS_TEXT[b['contents']]}, {where}.")
    parts.append(f"The lower boxes are placed already. I open my gripper with the top box {release} up from the box under it, and I usually set a "
                 f"box down no more than {sigma} away from my target.")
    return " ".join(parts)


def _tower_options(v: dict[str, Any], plan: dict[str, Any]) -> dict[str, str]:
    opts = plan["option_values"]
    n = len(v["boxes"])
    back = opts["opt_back"][f"boxes.{n - 1}.offset"]
    out = {"as_planned": f"place the top box {_offset_text(v['boxes'][-1]['offset'])}, as planned",
           "top_back": f"place the top box {_offset_text(back)} instead"}
    shift = opts["opt_shift"]
    if n == 3:
        out["rearranged"] = (f"move the middle box so it is {_offset_text(shift['boxes.1.offset'])} and put the top box "
                             f"{_offset_text(shift[f'boxes.{n - 1}.offset'])}")
    else:
        out["rearranged"] = "turn the top box end for end before placing it, same position"
    return out


def _tower_questions(v: dict[str, Any], mix: list[str], plan: dict[str, Any], person: str, rng: random.Random, cf_option: str) -> dict[str, Any]:
    n = len(v["boxes"])
    top = v["boxes"][-1]["offset"]
    cf = plan["cf_value"]
    release = {"third": "the unit lets go", "second": "you let go", "first": "I let go"}[person]
    out: dict[str, Any] = {}
    out["box_falls"] = {"type": "noul", "instructions": rng.choice((
        f"After {release} of the top box, does any box come off the stack?",
        "Once the top box is released, does the stack lose at least one box?"))}
    out["offset_change_effect"] = {"type": "choice", "instructions": rng.choice((
        f"If the top box were centred {_cm(cf)} toward the front of the box below instead of {_cm(top)}, would the stack be more likely, "
        "less likely or about as likely to lose a box?",
        f"Compared with the planned {_cm(top)}, how would centring the top box {_cm(cf)} forward change the risk that a box comes off?")),
        "criteria": dict(_DIRECTION)}
    if "outcome" in mix:
        out["stack_after_release"] = {"type": "choice", "instructions": "What does the stack look like a few seconds after the release?",
                                      "criteria": {"all_stay": "every box stays where it was put", "top_falls": "only the top box comes off",
                                                   "more_fall": "a lower box goes over as well"}}
    options = _tower_options(v, plan)
    if "action" in mix:
        out["safest_stacking"] = {"type": "choice", "instructions": "Which of these ways to finish the stack is safest?", "criteria": _only(options, plan)}
    if "compare" in mix:
        out["riskier_stacking"] = {"type": "choice", "instructions": f"Which is more likely to end with a box on the floor: the plan as it stands, or "
                                   f"to {options[cf_option]}?",
                                   "criteria": {"as_planned": "the plan as it stands", cf_option: "the other arrangement", "about_same": "about the same (within 0.1)"}}
    if "score_cf" in mix:
        out["fall_risk_other_offset"] = {"type": "score", "instructions": f"Rate the chance that a box comes off if the top box were centred {_cm(cf)} "
                                         f"toward the front of the box below rather than {_cm(top)}.",
                                         "criteria": ["0 = every box will stay put", "1 = a box is sure to come off"]}
    if "severe" in mix:
        out["lower_box_goes"] = {"type": "noul", "instructions": "Does a box below the top one go over as well?" if n == 3
                                 else "Does the bottom box go over as well?"}
    return out


OPTION_TEXT = {
    "tilt": {"upright": "keep it standing upright as it is", "lengthwise": "lay it down so its top end points to the front of the deck",
             "crosswise": "lay it on its side so its top end points across the deck"},
}
PERSON = {"log": "third", "sheet": "third", "brief": "second", "self": "first"}


def state_text(family: str, visible: dict[str, Any], style: str, wording: int) -> str:
    rng = random.Random(wording ^ 0x57A7E)
    return {"tilt": _tilt_state, "grip": _grip_state, "tower": _tower_state}[family](visible, style, rng)


def questions(family: str, visible: dict[str, Any], style: str, wording: int, mix: list[str], plan: dict[str, Any], cf_option: str) -> dict[str, Any]:
    rng = random.Random(wording ^ 0x0E57)
    build = {"tilt": _tilt_questions, "grip": _grip_questions, "tower": _tower_questions}[family]
    return build(visible, mix, plan, PERSON[style], rng, cf_option)
