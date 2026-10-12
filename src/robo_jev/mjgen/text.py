"""v0.3b text — four narration styles and the question wording (Task v0.3b §1.2), written for this task.

Nothing here is taken from T8b's wording or from :mod:`robo_jev.simgen` phrase banks (a test parses the imports; the overlap check
in :mod:`robo_jev.mjgen.overlap` compares sentences and word n-grams with T8b). Styles:

* ``log`` — an operations log in the third person ("Unit: …; Cargo: …; Planned: …");
* ``brief`` — a second-person task briefing ("You are …; your deck …");
* ``sheet`` — key–value lines;
* ``self`` — the robot speaking in the first person (its own sentences, not T8b's).

Every number the physics needs is written with its unit; hidden values (friction, where the weight sits inside, the exact mass of
an unknown fill, placement errors) are never written — only the words that set their prior. A paired record (one changed number)
is rendered from the same wording seed, so its text differs from its partner's only in that number.

Questions favour System-1 judgments (controller update 2026-10-12): which action is safest (``choice``, action), direction of change
("more likely, less likely or about the same", ``choice``), comparisons of two actions (``choice``), what happens (``choice``,
outcome), plus the key yes/no question, a 0–1 score with both anchors written out, and a yes/no on the severe outcome. No question
asks for a number.
"""

from __future__ import annotations

import random
from typing import Any

from robo_jev.mjgen.families import GRIP_PADS, GRIP_SURFACES, TILT_SURFACES

__all__ = ["DIRECTION_KEYS", "OPTION_TEXT", "questions", "state_text"]

DIRECTION_KEYS = ("more_likely", "less_likely", "about_same")


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
    "level": ("its contents are spread evenly", "it is packed evenly throughout"),
    "low": ("the heavy part of its contents sits at the bottom", "its weight is concentrated low down"),
    "high": ("most of its weight is up near the top", "its heavy contents ride near the top"),
    "unsure": ("there is no telling how its contents are distributed", "how its contents are distributed is not known"),
    "fore": ("its heavier end faces the front of the deck", "its weight is biased toward the deck's front"),
    "aft": ("its heavier end faces the rear of the deck", "its weight is biased toward the deck's rear"),
}


def _tilt_plan(v: dict[str, Any], person: str) -> str:
    angle, dur = f"{v['angle']:.1f}°", f"{v['duration']:.1f} s"
    if v["storyline"] == "bed":
        way = "rises" if v["direction"] == "up" else "dips"
        subject = {"third": "The bed", "second": "Your bed", "first": "My bed"}[person]
        return f"{subject} is about to tilt so that its front end {way}: the angle goes from level to {angle} over {dur} and is then held there."
    climb = "climb" if v["direction"] == "up" else "go down"
    subject = {"third": "The unit is", "second": "You are", "first": "I am"}[person]
    deck = {"third": "its deck", "second": "your deck", "first": "my deck"}[person]
    nose = "front up" if v["direction"] == "up" else "front down"
    return (f"{subject} about to {climb} a ramp: as the wheels roll onto it, {deck} pitches {nose} from level to {angle} within {dur}, "
            "and keeps that pitch on the slope.")


def _tilt_state(v: dict[str, Any], style: str, rng: random.Random) -> str:
    length, width, height = v["dims"]
    fill = rng.choice(TILT_FILL_TEXT[v["fill"]])
    surface = TILT_SURFACES[v["surface"]]["text"]
    size = f"{_n(length)} m long (front–rear), {_n(width)} m wide and {_n(height)} m high"
    if style == "log":
        return (f"Log, {v['domain']} floor. Unit: {_a(v['robot'])}. Cargo: one {v['load_name']} standing upright and unstrapped at the centre of "
                f"the deck, {size}, mass {v['mass']:.1f} kg. Packing note: {fill}. Deck covering: {surface}. The deck extends far past the cargo "
                f"in every direction and there is nothing beside it to lean on. {_tilt_plan(v, 'third')}")
    if style == "brief":
        return (f"You are {_a(v['robot'])} in {_a(v['domain'])}. {_a(v['load_name']).capitalize()} stands upright and loose at the centre of your "
                f"deck: {size}, {v['mass']:.1f} kg. As far as you know, {fill}. Your deck is covered with {surface}; nothing ties the load down, "
                f"nothing is beside it, and there is deck to spare all round it. {_tilt_plan(v, 'second')}")
    if style == "sheet":
        story = "tilting bed" if v["storyline"] == "bed" else "ramp"
        return "\n".join((
            f"robot: {v['robot']} ({v['domain']})", f"load: {v['load_name']}, upright, unstrapped, centred on the deck",
            f"load size (length front-rear / width / height): {_n(length)} m / {_n(width)} m / {_n(height)} m", f"load mass: {v['mass']:.1f} kg",
            f"contents: {fill}", f"deck surface: {surface}", "deck clearance: ample on every side, no walls or neighbours",
            f"manoeuvre: {story} — {_tilt_plan(v, 'third')}",
        ))
    return (f"This is the {v['robot']} reporting from the {v['domain']}. A {v['load_name']} rides loose and upright at the centre of my deck; "
            f"it is {size} and weighs {v['mass']:.1f} kg. From what I can tell, {fill}. The deck under it is {surface}, with plenty of room "
            f"around it and nothing it could rest against. {_tilt_plan(v, 'first')}")


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
        "criteria": {"more_likely": "more likely than at the planned pitch", "less_likely": "less likely than at the planned pitch",
                     "about_same": "about the same (the chance changes by less than 0.1)"}}
    if "outcome" in mix:
        out["load_after_tilt"] = {"type": "choice", "instructions": f"What happens to the {load} as {deck} goes to {angle} and holds there?",
                                  "criteria": {"stays": "it stays upright and moves less than 3 cm", "slides": "it slides 3 cm or more but stays upright",
                                               "topples": "it tips over"}}
    if "action" in mix:
        out["safest_way_to_carry"] = {"type": "choice", "instructions": f"For this {angle} pitch, which way of carrying the {load} is safest?",
                                      "criteria": OPTION_TEXT["tilt"]}
    if "compare" in mix:
        other = cf_option
        out["riskier_orientation"] = {"type": "choice", "instructions": f"At {angle}, which is more likely to slide or topple: the {load} as it stands now, "
                                      f"or the {load} {OPTION_TEXT['tilt'][other].replace('lay it', 'laid').replace('keep it', 'kept')}?",
                                      "criteria": {"as_it_stands": "the upright load as planned", other: "the re-oriented load",
                                                   "about_same": "about the same (within 0.1)"}}
    if "score_cf" in mix:
        out["risk_at_other_pitch"] = {"type": "score", "instructions": f"How likely is the {load} to slide or topple if the pitch were {cf} instead?",
                                      "criteria": ["0 = it certainly stays put and upright", "1 = it certainly slides or topples"]}
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
    return (f"{subject} will {verb} by {_n(v['distance'])} m with a top speed of {v['peak']:.1f} m/s, speeding up and braking at "
            f"{v['accel']:.1f} m/s².")


def _grip_state(v: dict[str, Any], style: str, rng: random.Random) -> str:
    length, width, height = v["dims"]
    pads = GRIP_PADS[v["pads"]]["text"]
    surface = GRIP_SURFACES[v["surface"]]["text"]
    size = f"{_n(length)} m long, {_n(width)} m across the grip and {_n(height)} m tall"
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
            f"item size (length / across grip / height): {_n(length)} m / {_n(width)} m / {_n(height)} m", f"item surface: {surface}",
            f"item mass: {_mass_text(v)}", "grasp: side faces, pads centred", f"clamping force: {force}", f"move: {_grip_move(v, 'third')}",
        ))
    return (f"Here is my situation in the {v['domain']}: as {_a(v['robot'])} with {pads}, I have {_a(v['item_name'])} pinched by its side "
            f"faces, pads centred. It is {size}, its outside is {surface}, and it weighs {_mass_text(v)}. My clamping force is {force}. "
            f"{_grip_move(v, 'first')}")


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
        "criteria": {"more_likely": "more likely than with the planned force", "less_likely": "less likely than with the planned force",
                     "about_same": "about the same (the chance changes by less than 0.1)"}}
    if "outcome" in mix:
        out["item_after_move"] = {"type": "choice", "instructions": f"How does the {item} end up after the {move}?",
                                  "criteria": {"held": "still held, shifted less than 1 cm", "slipped": "slipped 1 cm or more but still held",
                                               "dropped": "fell out of the grip"}}
    if "action" in mix:
        out["safest_grip_plan"] = {"type": "choice", "instructions": f"Which of these plans is safest for the {item}?", "criteria": _grip_options(v, plan)}
    if "compare" in mix:
        options = _grip_options(v, plan)
        out["riskier_grip_plan"] = {"type": "choice", "instructions": f"Which is more likely to make the {item} slip or drop: the plan as it stands "
                                    f"({options['as_planned']}), or {options[cf_option]}?",
                                    "criteria": {"as_planned": "the plan as it stands", cf_option: "the other plan", "about_same": "about the same (within 0.1)"}}
    if "score_cf" in mix:
        out["slip_risk_other_force"] = {"type": "score", "instructions": f"With a clamping force of {cf} instead, how likely is the {item} to slip or drop?",
                                        "criteria": ["0 = it certainly stays put in the grip", "1 = it certainly slips or drops"]}
    if "severe" in mix:
        out["falls_out"] = {"type": "noul", "instructions": f"Does the {item} fall out of {hold} during the {move}?"}
    return out


def _grip_options(v: dict[str, Any], plan: dict[str, Any]) -> dict[str, str]:
    opts = plan["option_values"]
    unit = "m/s²"
    return {"as_planned": f"clamp at {v['squeeze']:.1f} N and move at {v['accel']:.1f} {unit} as planned",
            "firmer_faster": f"clamp at {opts['opt_firm_fast']['squeeze']:.1f} N but move at {opts['opt_firm_fast']['accel']:.1f} {unit}",
            "lighter_slower": f"clamp at {opts['opt_light_slow']['squeeze']:.1f} N and move at only {opts['opt_light_slow']['accel']:.1f} {unit}"}


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
        lines.append((j, b, f"{_n(length)} m front–back, {_n(width)} m wide, {_n(height)} m high", where))
    release = f"{_cm(v['drop'])}"
    sigma = f"{_cm(v['sigma'])}"
    if style == "log":
        parts = [f"Log, {v['domain']}. Unit: {_a(v['robot'])}, building a stack of {n} boxes, bottom to top."]
        for j, b, size, where in lines:
            parts.append(f"Box {j + 1}: {b['name']}, {size}, {b['mass']:.1f} kg, {TOWER_CONTENTS_TEXT[b['contents']]}; {where}.")
        parts.append(f"The lower boxes are already in place; the top box is let go {release} above the box under it. Placements land within about "
                     f"{sigma} of the aim. Front = the side facing the aisle.")
        return " ".join(parts)
    if style == "brief":
        parts = [f"You are {_a(v['robot'])} in {_a(v['domain'])}, stacking {n} boxes on the floor."]
        for j, b, size, where in lines:
            parts.append(f"{'At the bottom' if j == 0 else ('On top' if j == n - 1 else 'Next')} is {_a(b['name'])} ({size}, {b['mass']:.1f} kg, "
                         f"{TOWER_CONTENTS_TEXT[b['contents']]}), {where}.")
        parts.append(f"Everything below the top box is already placed; you release the top box {release} above the one beneath it, and your "
                     f"placements are usually within {sigma} of where you aim. 'Front' means the aisle side.")
        return " ".join(parts)
    if style == "sheet":
        out = [f"robot: {v['robot']} ({v['domain']})", f"task: stack of {n} boxes on the floor (front = aisle side)"]
        for j, b, size, where in lines:
            out.append(f"box {j + 1} ({'bottom' if j == 0 else ('top' if j == n - 1 else 'middle')}): {b['name']}; {size}; {b['mass']:.1f} kg; "
                       f"{TOWER_CONTENTS_TEXT[b['contents']]}; {where}")
        out += [f"release: top box let go {release} above the box below; lower boxes already in place", f"placement accuracy: within about {sigma}"]
        return "\n".join(out)
    parts = [f"I'm {_a(v['robot'])} working in the {v['domain']}, and I am finishing a stack of {n} boxes on the floor (front means the aisle side)."]
    for j, b, size, where in lines:
        parts.append(f"{'Lowest' if j == 0 else ('Uppermost' if j == n - 1 else 'In the middle')}: {_a(b['name'])}, {size}, {b['mass']:.1f} kg, "
                     f"{TOWER_CONTENTS_TEXT[b['contents']]}, {where}.")
    parts.append(f"The lower boxes are placed already. I open my gripper with the top box {release} above the box under it, and my placements "
                 f"usually come within {sigma} of my aim.")
    return " ".join(parts)


def _tower_questions(v: dict[str, Any], mix: list[str], plan: dict[str, Any], person: str, rng: random.Random, cf_option: str) -> dict[str, Any]:
    boxes = v["boxes"]
    n = len(boxes)
    top = boxes[-1]["offset"]
    cf = plan["cf_value"]
    release = {"third": "the unit lets go", "second": "you let go", "first": "I let go"}[person]
    out: dict[str, Any] = {}
    out["box_falls"] = {"type": "noul", "instructions": rng.choice((
        f"After {release} of the top box, does any box fall off the stack?",
        f"Once the top box is released, does the stack lose at least one box?"))}
    out["offset_change_effect"] = {"type": "choice", "instructions": rng.choice((
        f"If the top box were centred {_cm(cf)} toward the front of the box below instead of {_cm(top)}, would the stack be more likely, "
        "less likely or about as likely to lose a box?",
        f"Compared with the planned {_cm(top)}, how would centring the top box {_cm(cf)} forward change the risk that a box falls?")),
        "criteria": {"more_likely": "more likely than with the planned placement", "less_likely": "less likely than with the planned placement",
                     "about_same": "about the same (the chance changes by less than 0.1)"}}
    if "outcome" in mix:
        out["stack_after_release"] = {"type": "choice", "instructions": "What does the stack look like a few seconds after the release?",
                                      "criteria": {"all_stay": "every box stays where it was put", "top_falls": "only the top box falls",
                                                   "more_fall": "a lower box goes over as well"}}
    options = _tower_options(v, plan)
    if "action" in mix:
        out["safest_stacking"] = {"type": "choice", "instructions": "Which of these ways to finish the stack is safest?", "criteria": options}
    if "compare" in mix:
        out["riskier_stacking"] = {"type": "choice", "instructions": f"Which is more likely to end with a box on the floor: the plan as it stands, or "
                                   f"to {options[cf_option]}?",
                                   "criteria": {"as_planned": "the plan as it stands", cf_option: "the other arrangement", "about_same": "about the same (within 0.1)"}}
    if "score_cf" in mix:
        out["fall_risk_other_offset"] = {"type": "score", "instructions": f"If the top box were centred {_cm(cf)} toward the front of the box below "
                                         "instead, how likely is it that a box falls?",
                                         "criteria": ["0 = every box certainly stays put", "1 = a box certainly falls"]}
    if "severe" in mix:
        out["lower_box_goes"] = {"type": "noul", "instructions": "Does a box below the top one go over as well?" if n == 3
                                 else "Does the bottom box go over as well?"}
    return out


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
