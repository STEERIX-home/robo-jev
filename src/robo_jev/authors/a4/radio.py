"""Author A4 — radio transcripts between a field robot and its supervisor, and the questions (English, metric units).

The state is a short exchange of ``SUPERVISOR:`` / ``ROBOT:`` turns with the facts spread over them. Every wording choice (template,
callsign, station, place, question phrasing) comes from the group's text seed, so the two records of a one-fact pair read the same
except for the one fact that changes. All names, phrasings and option keys are this author's own.
"""

from __future__ import annotations

from typing import Any

import numpy as np


__all__ = ["QUESTION_NAMES", "render"]

#: Question names per scene and kind (the key event, the severe event, the score at the counterfactual number, the direction of
#: the counterfactual, the comparison, the action, the outcome category).
QUESTION_NAMES = {
    "bank": {"key": "load_shifts_after_release", "severe": "load_tips_over", "score_cf": "shift_chance_other_grade",
             "direction": "steeper_or_gentler_effect", "compare": "which_setdown_riskier", "action": "best_way_to_leave_it",
             "outcome": "what_the_load_does"},
    "ledge": {"key": "nose_drops_over_edge", "severe": "robot_goes_over_edge", "score_cf": "nose_drop_chance_other_stop",
              "direction": "stop_point_effect", "compare": "which_approach_riskier", "action": "best_way_to_look_over",
              "outcome": "where_the_robot_ends_up"},
    "anchor": {"key": "robot_slips_on_ramp", "severe": "robot_dragged_down_ramp", "score_cf": "slip_chance_other_load",
               "direction": "heavier_or_lighter_effect", "compare": "which_haul_riskier", "action": "best_way_to_haul",
               "outcome": "how_the_robot_holds"},
}
QUESTION_KIND = {"key": "key", "severe": "severe", "score_cf": "score_cf", "direction": "direction", "compare": "compare", "action": "action",
                 "outcome": "outcome"}

STATIONS = {"orchard": ("Orchard desk", "Packhouse"), "farm": ("Farm office", "Yard base"), "construction": ("Site control", "Gatehouse"),
            "marina": ("Harbour office", "Dock master"), "park": ("Park depot", "Ranger base"), "sidewalk": ("Street ops", "Kerbside desk")}
CALLSIGNS = ("Rover Six", "Mule Two", "Kestrel", "Badger Three", "Unit Nine", "Wren One", "Otter Four", "Heron Two", "Fox Five", "Marten Seven",
             "Lark Eight", "Stoat One")
BANK_PLACES = {"orchard": ("bank at the end of row {n}", "slope below the windbreak"), "farm": ("grass bank by the silage clamp", "field edge above the ditch"),
               "construction": ("spoil embankment", "side of the cutting"), "marina": ("bank above the slipway", "slope behind the boat store"),
               "park": ("lawn bank above the pond", "slope below the bandstand"), "sidewalk": ("verge beside the footpath", "sloping lawn by the bus stop")}
LEDGE_PURPOSE = {"orchard": "to spray the lower row", "farm": "to check the drain outlet", "construction": "to film the footing below",
                 "marina": "to read the gauge on the wall below", "park": "to look at the outlet below", "sidewalk": "to scan the grate below"}
ANCHOR_WHERE = {"marina": "at the head of the slipway", "construction": "at the top of the ramp into the dig", "farm": "at the top of the track from the lower field",
                "orchard": "at the top of the terrace ramp", "park": "on the bank above the stream"}
BANK_CONTENTS_TEXT = {
    "packed full, level with the top": "packed full of {filler}, level with the top",
    "half full, settled at the bottom": "half full of {filler}, settled at the bottom",
    "nearly empty, one heavy offcut on the floor": "nearly empty, just one heavy offcut lying on the floor of it",
    "heaped up over the rim": "{filler} heaped up over the rim",
    "full, with a coil of chain strapped on the lid": "full of {filler}, with a coil of chain strapped on the lid",
    "half full, all shoved to one side": "half full of {filler}, all shoved to one side",
}
LEDGE_PAYLOAD_TEXT = {
    "front tank full": "my front tank is full",
    "front tank half full": "my front tank is about half full",
    "running empty, battery pack in the back": "I'm running empty and the battery pack sits at the back",
    "front tool tray loaded": "the tool tray on my front is loaded",
    "counterweight plates on the rear": "I have counterweight plates bolted on the rear",
    "load spread evenly": "my load is spread evenly",
    "camera mast folded forward": "the camera mast is folded forward over my nose",
}


def _n(x: float) -> str:
    """A number as said on the radio: no trailing .0."""
    x = float(x)
    return str(int(round(x))) if abs(x - round(x)) < 1e-9 else f"{x:g}"


def _cm(x: float) -> str:
    return str(int(round(float(x) * 100)))


def _choices(group: dict[str, Any]) -> dict[str, Any]:
    from robo_jev.authors.a4.label import derive

    rng = np.random.default_rng(derive("a4-text", group["group_seed"]))
    domain = group["records"][0]["domain"]
    return {"template": int(rng.integers(3)), "station": STATIONS[domain][int(rng.integers(2))], "callsign": CALLSIGNS[int(rng.integers(len(CALLSIGNS)))],
            "place": int(rng.integers(2)), "row": int(rng.integers(3, 40)), "q": [int(rng.integers(2)) for _ in range(7)], "close": int(rng.integers(3))}


# --------------------------------------------------------------------------
# States
# --------------------------------------------------------------------------


def _bank_state(group: dict[str, Any], vis: dict[str, Any], c: dict[str, Any]) -> str:
    st, cs = c["station"], c["callsign"]
    load = vis["load"]
    place = BANK_PLACES[vis["domain"]][c["place"]].format(n=c["row"])
    contents = BANK_CONTENTS_TEXT[vis["contents"]].format(filler=vis["filler"])
    across = vis["orientation"] == "across"
    orient = "long side across the slope" if across else "long side pointing straight down the slope"
    patch = group["options"]["opt_patch"]
    other = f"{_n(patch['patch_slope'])} degrees, {_n(patch['patch_distance'])} metres"
    size = f"{_cm(vis['long'])} by {_cm(vis['short'])} centimetres and {_cm(vis['height'])} tall, {_n(vis['mass'])} kilograms"
    if c["template"] == 0:
        lines = [f"SUPERVISOR: {cs}, {st}. Where are you leaving the {load}?",
                 f"ROBOT: {st}, {cs}. On the {place}. Ground is {vis['ground']}, and my tilt sensor reads {_n(vis['slope'])} degrees there.",
                 "SUPERVISOR: What's in it?",
                 f"ROBOT: It's {contents}. {size}.",
                 "SUPERVISOR: How will you put it down?",
                 f"ROBOT: Set it with the {orient}, let go, and go back for the next one. The other patch I could use reads {other} along."]
    elif c["template"] == 1:
        lines = [f"ROBOT: {st}, {cs}, ready to drop the {load} on the {place}.",
                 f"SUPERVISOR: Go ahead, {cs}. What's the grade?",
                 f"ROBOT: {_n(vis['slope'])} degrees, {vis['ground']} underfoot.",
                 "SUPERVISOR: Size and weight?",
                 f"ROBOT: {size}. It's {contents}.",
                 "SUPERVISOR: Which way round?",
                 f"ROBOT: {orient[0].upper() + orient[1:]}. If you want it elsewhere, there's a patch at {_n(patch['patch_slope'])} degrees about "
                 f"{_n(patch['patch_distance'])} metres along. I'll leave it unattended."]
    else:
        lines = [f"SUPERVISOR: {cs}, {st}. I have you on the {place}. Survey says {_n(vis['slope'])} degrees there, {vis['ground']}. Confirm?",
                 f"ROBOT: Confirmed, {_n(vis['slope'])} degrees, {vis['ground']}.",
                 f"SUPERVISOR: And the {load}?",
                 f"ROBOT: {size}, {contents}. Plan is to stand it with the {orient} and leave it there.",
                 "SUPERVISOR: Any other spot?",
                 f"ROBOT: One patch at about {_n(patch['patch_slope'])} degrees, {_n(patch['patch_distance'])} metres further on."]
    return "\n".join(lines)


def _ledge_state(group: dict[str, Any], vis: dict[str, Any], c: dict[str, Any]) -> str:
    st, cs = c["station"], c["callsign"]
    payload = LEDGE_PAYLOAD_TEXT[vis["payload"]]
    purpose = LEDGE_PURPOSE[vis["domain"]]
    build = f"{_cm(vis['length'])} long, {_cm(vis['width'])} wide, {_cm(vis['height'])} high, {_n(vis['mass'])} kilograms"
    if c["template"] == 0:
        lines = [f"SUPERVISOR: {cs}, {st}. What are you doing at the {vis['edge']}?",
                 f"ROBOT: {st}, {cs}. I need to look over it {purpose}. The drop is {_cm(vis['drop'])} centimetres. I'll creep forward until my nose is "
                 f"{_cm(vis['overhang'])} centimetres past the edge and stop there.",
                 "SUPERVISOR: What's the ground like at the top?",
                 f"ROBOT: {vis['ground'][0].upper() + vis['ground'][1:]} right up to the edge.",
                 f"SUPERVISOR: Remind me of your build, {cs}.",
                 f"ROBOT: {build}. {payload[0].upper() + payload[1:]}."]
    elif c["template"] == 1:
        lines = [f"ROBOT: {st}, {cs}. Approaching the {vis['edge']} {purpose}.",
                 "SUPERVISOR: How far out are you going?",
                 f"ROBOT: Nose {_cm(vis['overhang'])} centimetres past the edge, then I stop and hold. The drop below is {_cm(vis['drop'])} centimetres.",
                 "SUPERVISOR: Surface?",
                 f"ROBOT: {vis['ground'][0].upper() + vis['ground'][1:]}.",
                 "SUPERVISOR: And your load?",
                 f"ROBOT: {payload[0].upper() + payload[1:]}. For the record I'm {build}."]
    else:
        lines = [f"SUPERVISOR: {cs}, {st}. I see you heading for the {vis['edge']}. The drop there is {_cm(vis['drop'])} centimetres.",
                 f"ROBOT: Copy. I'm {build}, and {payload}.",
                 "SUPERVISOR: What's the plan?",
                 f"ROBOT: Creep out until my nose is {_cm(vis['overhang'])} centimetres past the edge, {purpose}. The top is {vis['ground']}.",
                 "SUPERVISOR: Understood. Report when stopped."]
    return "\n".join(lines)


def _anchor_state(group: dict[str, Any], vis: dict[str, Any], c: dict[str, Any]) -> str:
    st, cs = c["station"], c["callsign"]
    where = ANCHOR_WHERE[vis["domain"]]
    preset = group["options"]["opt_start"]["start_time"]
    apron = group["options"]["opt_apron"]["apron_distance"]
    load = f"{vis['load']} at about {_n(vis['load_mass'])} kilograms, {vis['mass_source']}"
    winch = f"line speed {_n(vis['line_speed'])} metres a second, reached in {_n(vis['start_time'])} seconds"
    if c["template"] == 0:
        lines = [f"SUPERVISOR: {cs}, {st}. Ready to haul?",
                 f"ROBOT: {st}, {cs}. Parked {where}, brakes locked, nose uphill. The ramp under me is {_n(vis['slope'])} degrees, {vis['ramp']}. "
                 f"I weigh {_n(vis['mass'])} kilograms.",
                 "SUPERVISOR: What's on the line?",
                 f"ROBOT: A {load}. It's on the same ramp below me.",
                 "SUPERVISOR: Winch setting?",
                 f"ROBOT: Start preset one, {winch}. Preset two would take {_n(preset)} seconds. There's flat apron {_n(apron)} metres behind me."]
    elif c["template"] == 1:
        lines = [f"ROBOT: {st}, {cs}. In position {where} to winch up a {vis['load']}.",
                 "SUPERVISOR: Weight on the line?",
                 f"ROBOT: About {_n(vis['load_mass'])} kilograms, {vis['mass_source']}. The load sits on the ramp below me.",
                 "SUPERVISOR: And where you're parked?",
                 f"ROBOT: {_n(vis['slope'])} degrees of {vis['ramp']}, brakes on, facing up. I'm {_n(vis['mass'])} kilograms myself.",
                 "SUPERVISOR: How will you start it?",
                 f"ROBOT: Preset one, {winch}. Preset two is {_n(preset)} seconds to line speed, and the flat apron is {_n(apron)} metres back."]
    else:
        lines = [f"SUPERVISOR: {cs}, {st}. You're {where} with a {load}?",
                 f"ROBOT: Affirmative. The load is on the ramp below me. I'm braked on {vis['ramp']} at {_n(vis['slope'])} degrees, "
                 f"{_n(vis['mass'])} kilograms.",
                 "SUPERVISOR: Start settings?",
                 f"ROBOT: Preset one, {winch}. The other preset would be {_n(preset)} seconds. Flat apron is {_n(apron)} metres behind me if needed."]
    return "\n".join(lines)


# --------------------------------------------------------------------------
# Questions
# --------------------------------------------------------------------------


def _q(qtype: str, instructions: str, criteria: Any = None) -> dict[str, Any]:
    out: dict[str, Any] = {"type": qtype, "instructions": instructions}
    if criteria is not None:
        out["criteria"] = criteria
    return out


def _bank_questions(group: dict[str, Any], vis: dict[str, Any], c: dict[str, Any]) -> dict[str, dict[str, Any]]:
    n = QUESTION_NAMES["bank"]
    load, q = vis["load"], c["q"]
    cf = _n(group["cf_value"])
    patch = _n(group["options"]["opt_patch"]["patch_slope"])
    turned = ("Give it a quarter turn first, so the long side points down the slope" if vis["orientation"] == "across"
              else "Give it a quarter turn first, so the long side runs across the slope")
    moved = f"Take it to the other patch, at {patch} degrees"
    alt_text = {"turned_first": "giving it a quarter turn first" + turned[len("Give it a quarter turn first"):],
                "other_patch": f"taking it to the other patch, at {patch} degrees"}
    labels = {"turned_first": turned, "other_patch": moved}
    alt = group["compare_option"]
    return {
        n["key"]: _q("noul", (f"Once the robot lets go, will the {load} move from where it was set, by sliding or tipping?",
                              f"After release, does the {load} slide or tip at all?")[q[0]]),
        n["severe"]: _q("noul", (f"Will the {load} tip over after the robot lets go?", f"Does the {load} end up toppled over?")[q[1]]),
        n["score_cf"]: _q("score", (f"Say the ground there sloped at {cf} degrees. Give the chance that the {load} slides or tips once released.",
                                    f"Rate the chance that the {load} moves after release if the grade were {cf} degrees.")[q[2]],
                          ["0 = it surely stays where it is set", "1 = it surely slides or tips"]),
        n["direction"]: _q("choice", (f"Suppose the slope were {cf} degrees rather than {_n(vis['slope'])}. Does that raise, lower, or barely change the chance "
                                      f"that the {load} moves after release?",
                                      f"Compared with {_n(vis['slope'])} degrees, how would a slope of {cf} degrees change the chance that the {load} moves?")[q[3]],
                           {"goes_up": "More likely to move", "goes_down": "Less likely to move", "stays_similar": "About as likely"}),
        n["compare"]: _q("choice", (f"Which set-down is more likely to end with the {load} moving: the plan as stated, or {alt_text[alt]}?",
                                    f"Riskier for the {load}: the stated plan or {alt_text[alt]}? Pick the one more likely to slide or tip.")[q[4]],
                         {"the_plan": "The plan as stated", alt: labels[alt], "no_difference": "About the same"}),
        n["action"]: _q("choice", (f"Which way of leaving the {load} is best? A slide means a trip back, a toppled {load} is much worse, the quarter turn "
                                   "costs a few seconds, and the other patch costs a short carry.",
                                   f"Best choice for the {load}? Weigh a slide (a trip back), a topple (far worse), a quarter turn (a few seconds) and "
                                   "the carry to the other patch (a little longer).")[q[5]],
                        {"the_plan": "Set it down as planned", "turned_first": turned, "other_patch": moved}),
        n["outcome"]: _q("choice", (f"What will the {load} do after release?", f"Once let go, how does the {load} end up?")[q[6]],
                         {"stays_put": "Stays where it was set", "slides_down": "Slides down the slope", "tips_over": "Tips over"}),
    }


def _ledge_questions(group: dict[str, Any], vis: dict[str, Any], c: dict[str, Any]) -> dict[str, dict[str, Any]]:
    n = QUESTION_NAMES["ledge"]
    q = c["q"]
    cf = _cm(group["cf_value"])
    edge = vis["edge"]
    change = float(group["options"]["opt_stop"]["stop_change"])
    step = _cm(abs(change))
    stop = f"Stop {step} centimetres shorter" if change < 0 else f"Go {step} centimetres further out"
    reverse = "Turn round and back out to the same overhang, rear first"
    alt_text = {"other_stop": (f"stopping {step} centimetres shorter" if change < 0 else f"going {step} centimetres further out"),
                "reverse_in": "turning round and backing out to the same overhang, rear first"}
    labels = {"other_stop": stop, "reverse_in": reverse}
    alt = group["compare_option"]
    return {
        n["key"]: _q("noul", ("Once it stops there, will the robot tip forward over the edge?",
                              f"Stopped at that point, does the robot's nose drop over the {edge}?")[q[0]]),
        n["severe"]: _q("noul", (f"Will the robot go right over the {edge} and end up below it?",
                                 f"Does the robot fall all the way off the {edge}?")[q[1]]),
        n["score_cf"]: _q("score", (f"Say it had stopped with its nose {cf} centimetres over the edge. Give the chance that it tips forward.",
                                    f"Rate the chance of tipping forward over the edge for a stop {cf} centimetres out.")[q[2]],
                          ["0 = it surely stays level on top", "1 = it surely tips forward over the edge"]),
        n["direction"]: _q("choice", (f"Suppose it stopped {cf} centimetres past the edge rather than {_cm(vis['overhang'])}. Does that raise, lower, or barely "
                                      "change the chance of tipping forward?",
                                      f"How does stopping {cf} centimetres out, instead of {_cm(vis['overhang'])}, change the chance of tipping forward?")[q[3]],
                           {"goes_up": "More likely to tip", "goes_down": "Less likely to tip", "stays_similar": "About as likely"}),
        n["compare"]: _q("choice", (f"Which is more likely to end with the robot tipping over the edge: the plan as stated, or {alt_text[alt]}?",
                                    f"Riskier at the {edge}: the stated plan or {alt_text[alt]}?")[q[4]],
                         {"the_plan": "The plan as stated", alt: labels[alt], "no_difference": "About the same"}),
        n["action"]: _q("choice", ("Which way to do it is best? Nosing down needs a recovery, going over the edge is far worse, moving the stop point "
                                   "costs a little, and turning round costs time.",
                                   "Best plan at the edge? Weigh a nose-down (a recovery job), going over (far worse), a different stop point (a little) "
                                   "and turning round (time).")[q[5]],
                        {"the_plan": "Go out as planned", "other_stop": stop, "reverse_in": reverse}),
        n["outcome"]: _q("choice", ("Where does the robot end up?", "How does the robot finish once it stops?")[q[6]],
                         {"holds": "Stays level on top", "noses_down": "Tips forward, nose resting on the lower level",
                          "goes_over": "Goes right over the edge"}),
    }


def _anchor_questions(group: dict[str, Any], vis: dict[str, Any], c: dict[str, Any]) -> dict[str, dict[str, Any]]:
    n = QUESTION_NAMES["anchor"]
    q = c["q"]
    cf = _n(group["cf_value"])
    preset = _n(group["options"]["opt_start"]["start_time"])
    other = f"Switch to the other preset, {preset} seconds to line speed"
    apron = "Back up to the flat apron and haul from there"
    alt_text = {"other_start": f"the other preset ({preset} seconds to line speed)", "from_apron": "backing up to the flat apron first and hauling from there"}
    labels = {"other_start": other, "from_apron": apron}
    alt = group["compare_option"]
    return {
        n["key"]: _q("noul", ("When the winch starts, will the robot slide down the ramp at all, even slightly?",
                              "Does the robot lose its grip and slip when the haul starts?")[q[0]]),
        n["severe"]: _q("noul", ("Will the robot be dragged down the ramp and keep sliding?",
                                 "Does the load drag the robot down the ramp without it stopping?")[q[1]]),
        n["score_cf"]: _q("score", (f"Say the load were about {cf} kilograms. Give the chance that the robot slips when the winch starts.",
                                    f"Rate the chance of the robot slipping with {cf} kilograms on the line.")[q[2]],
                          ["0 = it surely holds without moving", "1 = it surely slips"]),
        n["direction"]: _q("choice", (f"Suppose the load were about {cf} kilograms rather than {_n(vis['load_mass'])}. Does that raise, lower, or barely change "
                                      "the chance that the robot slips?",
                                      f"How does a load of about {cf} kilograms, instead of {_n(vis['load_mass'])}, change the chance that the robot slips?")[q[3]],
                           {"goes_up": "More likely to slip", "goes_down": "Less likely to slip", "stays_similar": "About as likely"}),
        n["compare"]: _q("choice", (f"Which haul is more likely to make the robot slip: the plan as stated, or {alt_text[alt]}?",
                                    f"Riskier for the robot's grip: the stated plan or {alt_text[alt]}?")[q[4]],
                         {"the_plan": "The plan as stated", alt: labels[alt], "no_difference": "About the same"}),
        n["action"]: _q("choice", ("Which way to haul is best? A short slip is a nuisance, being dragged down the ramp is far worse, switching presets "
                                   "costs a little time, and moving to the apron costs more.",
                                   "Best way to run this haul? Weigh a short slip (a nuisance), being dragged (far worse), switching presets (a little "
                                   "time) and moving to the apron (more time).")[q[5]],
                        {"the_plan": "Haul as planned", "other_start": other, "from_apron": apron}),
        n["outcome"]: _q("choice", ("How does the robot fare when the winch starts?", "What happens to the robot once the haul begins?")[q[6]],
                         {"holds_fast": "Holds without moving", "creeps": "Slides a little, then holds", "dragged": "Gets dragged down the ramp"}),
    }


STATES = {"bank": _bank_state, "ledge": _ledge_state, "anchor": _anchor_state}
QUESTIONS = {"bank": _bank_questions, "ledge": _ledge_questions, "anchor": _anchor_questions}


def render(group: dict[str, Any], member: int) -> tuple[str, dict[str, dict[str, Any]]]:
    """(state, questions) of one record of a group."""
    scene = group["scene"]
    vis = group["records"][member]
    choices = _choices(group)
    return STATES[scene](group, vis, choices), QUESTIONS[scene](group, vis, choices)
