"""T9 text — the robot's own words for a moment and its six questions (Task T9 §1-1 "Text", "Questions").

Written for T9 (no phrase bank or sentence from the engine, v0.3b or T8b; the overlap check in :mod:`robo_jev.t9.overlap` compares
every sentence and 5-gram). The voice: a robot thinking aloud just before it acts — what it is holding or about to touch, what it
can tell about it, what it plans, in short first-person sentences with the numbers the physics needs. Hidden quantities are never
written as numbers; only the words behind them (a surface, the contents, a strap, the pump's state) are.

Every record asks the same six kinds of question (names per family in :data:`NAMES`):

* ``key`` (yes/no) — the key event under the plan;
* ``cf`` (score, 0–1 with both ends written out) — the key event with one stated number changed;
* ``alt`` (yes/no) — the key event under a gentler action (the comparison of two actions);
* ``dir`` (choice) — one number or word changed: is the key event clearly likelier (by ≥ 10 points), clearly less likely, or within
  10 points;
* ``act`` (choice) — plan, gentler action or a safe but slow step, with the failure's cost stated in seconds of delay;
* ``outcome`` (choice) — the three outcome categories under the plan.
"""

from __future__ import annotations

import random
from typing import Any

from robo_jev.t9 import families as F

__all__ = ["DIR_OPTIONS", "NAMES", "OUTCOME_OPTIONS", "ACT_OPTIONS", "render"]

#: Question names per family: key, cf, alt, dir, act, outcome.
NAMES: dict[str, dict[str, str]] = {
    "roll": {"key": "over_lip", "cf": "over_lip_rating", "alt": "over_lip_gentler", "dir": "over_lip_shift", "act": "tilt_choice", "outcome": "roll_result"},
    "swing": {"key": "touches_upright", "cf": "touch_rating", "alt": "touches_if_slower", "dir": "touch_shift", "act": "swing_choice", "outcome": "swing_result"},
    "knock": {"key": "knocked_over", "cf": "knock_rating", "alt": "knocked_if_slower", "dir": "knock_shift", "act": "sweep_choice", "outcome": "knock_result"},
    "conveyor": {"key": "belt_topples", "cf": "belt_topple_rating", "alt": "topples_if_gentler", "dir": "belt_shift", "act": "belt_choice", "outcome": "belt_result"},
    "overhang": {"key": "anything_drops", "cf": "drop_rating", "alt": "drops_if_further_in", "dir": "drop_shift", "act": "placing_choice", "outcome": "shelf_result"},
    "suction": {"key": "panel_lost", "cf": "panel_loss_rating", "alt": "lost_if_gentler", "dir": "panel_shift", "act": "move_choice", "outcome": "panel_result"},
}
#: Direction options (the label rule: change of the key event's share ≥ +0.10 / ≤ −0.10 / in between).
DIR_OPTIONS = {
    "likelier": "Clearly more likely than now, by at least ten percentage points",
    "less_likely": "Clearly less likely than now, by at least ten percentage points",
    "no_real_change": "Within ten percentage points of now, either way",
}
#: Outcome option keys per family, in the order of :data:`robo_jev.t9.physics.OUTCOMES`.
OUTCOME_OPTIONS: dict[str, tuple[str, str, str]] = {
    "roll": ("stays_put", "stopped_by_lip", "over_the_lip"),
    "swing": ("misses_it", "grazes_it", "hits_it"),
    "knock": ("stays_standing", "slides_upright", "falls_over"),
    "conveyor": ("keeps_its_spot", "skids_along", "tips_over"),
    "overhang": ("all_stays", "item_drops_off", "board_tips_off"),
    "suction": ("holds_firm", "creeps_on_cup", "comes_off"),
}
#: Action option keys per family: plan, gentler action, safe but slow step.
ACT_OPTIONS: dict[str, tuple[str, str, str]] = {
    "roll": ("tilt_as_planned", "tilt_less", "stow_it_first"),
    "swing": ("move_as_planned", "move_slower", "lower_it_first"),
    "knock": ("sweep_as_planned", "sweep_slower", "clear_it_first"),
    "conveyor": ("run_as_planned", "ramp_gently", "lift_it_off_first"),
    "overhang": ("place_as_planned", "place_further_in", "push_board_back_first"),
    "suction": ("move_as_planned", "accelerate_gently", "add_second_cup_first"),
}


def _cm(x: float) -> str:
    """Metres → centimetres with at most one decimal ("12", "0.8", "3.5")."""
    text = f"{100 * float(x):.1f}"
    return text[:-2] if text.endswith(".0") else text


def _num(x: float, digits: int = 2) -> str:
    text = f"{float(x):.{digits}f}"
    return text.rstrip("0").rstrip(".") if "." in text else text


def _pick(rng: random.Random, *options: str) -> str:
    return options[rng.randrange(len(options))]


def _scale(lo_text: str, hi_text: str) -> list[str]:
    return [f"0 means {lo_text}", f"1 means {hi_text}"]


# ==========================================================================================================================
# roll
# ==========================================================================================================================


def _roll(vis: dict[str, Any], ch: dict[str, Any], costs: dict[str, float], rng: random.Random) -> tuple[str, dict[str, Any]]:
    item = F.ROLL_ITEMS[vis["item"]]
    short = item["name"].split(" of ")[0].split()[-1] if " of " in item["name"] else item["name"].split()[-1]
    thing = f"the {short}"
    surface = F.ROLL_SURFACES[vis["surface"]]["text"]
    lip, dist, tilt = vis["lip"], vis["dist"], vis["tilt"]
    if vis["context"] == "ramp":
        opener = _pick(rng, "Parcel cart here, stopped at the foot of the dock ramp.", "I am the dock cart, lined up at the bottom of the loading ramp.")
        where = "my open deck"
    else:
        opener = _pick(rng, "I am a serving arm with a flat tray locked in my gripper.", "Serving arm here; a flat tray is clamped level in my gripper.")
        where = "the tray"
    if item["shape"] == "capsule":
        body = (f"On {where} lies a {item['name']}, {item['contents']}, resting on its side with its length across the way it would roll. "
                f"It is {_cm(2 * vis['radius'])} cm across and {_cm(vis['length'])} cm long, and weighs roughly {_num(vis['mass'])} kg.")
    else:
        body = (f"On {where} sits a loose {item['name']}, {item['contents']}, {_cm(2 * vis['radius'])} cm across and roughly "
                f"{_num(vis['mass'], 3)} kg.")
    floor = _pick(rng, f"Underneath it is {surface}.", f"The floor it rests on is {surface}.")
    edge = (f"Downhill of it, once I tilt, runs a rounded lip {_cm(lip)} cm high; {thing} meets the floor {_cm(dist)} cm short of the lip's foot. "
            "Nothing else holds it in place.")
    if vis["context"] == "ramp":
        plan = (f"Climbing the ramp will tip my deck back by {_num(tilt, 1)} degrees. The tilt comes on over {_num(vis['ramp'], 1)} s as my wheels "
                f"go up, and I stay on the slope for about {_num(vis['hold'], 0)} s.")
    else:
        plan = (f"To slide the tray through a low hatch I mean to tip it {_num(tilt, 1)} degrees toward that lip, reaching the angle in "
                f"{_num(vis['ramp'], 1)} s and holding it for {_num(vis['hold'], 0)} s.")
    state = " ".join([opener, body, floor, edge, plan])
    names = NAMES["roll"]
    q: dict[str, Any] = {}
    q[names["key"]] = {"type": "noul", "instructions": _pick(rng, f"If I go ahead with this tilt, will {thing} end up over the lip?",
                                                             f"Going ahead with this tilt, does {thing} end up beyond the lip?")}
    cf = ch["cf"]
    if cf.field == "tilt":
        what = f"the tilt were {_num(cf.value, 1)} degrees instead of {_num(tilt, 1)}"
    elif cf.field == "lip":
        what = f"the lip stood {_cm(cf.value)} cm high instead of {_cm(lip)} cm"
    else:
        what = f"{thing} lay {_cm(cf.value)} cm from the lip's foot instead of {_cm(dist)} cm"
    q[names["cf"]] = {"type": "score", "instructions": f"Keep everything else, but say {what}. How likely would {thing} be to end up over the lip?",
                      "criteria": _scale(f"{thing} surely stays behind the lip", f"{thing} surely ends up over it")}
    alt_tilt = ch["alt"]
    gentle = (f"take the long back ramp, which tips my deck only {_num(alt_tilt, 1)} degrees" if vis["context"] == "ramp"
              else f"tip the tray only {_num(alt_tilt, 1)} degrees and ease it through more slowly")
    q[names["alt"]] = {"type": "noul", "instructions": f"Suppose instead I {gentle}. Will {thing} end up over the lip then?"}
    q[names["dir"]] = _dir_question(_roll_dir_text(ch["dirv"], vis, thing), f"{thing} ends up over the lip")
    q[names["act"]] = _act_question(
        f"Which do I do? Losing {thing} over the lip would cost me as much as {_num(costs['fail'], 0)} s of delay.",
        "roll", ["Tilt as planned (no extra time)", f"{gentle[0].upper() + gentle[1:]} (adds {_num(costs['alt'], 0)} s)",
                 f"First move {thing} into my lidded crate, where it cannot roll (adds {_num(costs['secure'], 0)} s)"])
    q[names["outcome"]] = _outcome_question(f"After the tilt is held as planned, where is {thing}?", "roll", [
        "Still where it lay, less than a centimetre away", "It rolled down and stopped at the lip, still aboard", "Over the lip and off"])
    return state, q


def _roll_dir_text(change: F.Change, vis: dict[str, Any], thing: str) -> str:
    if change.field == "mass":
        return f"Picture {thing} weighing {_num(change.value)} kg instead of {_num(vis['mass'])} kg, same size and everything else unchanged."
    if change.field == "length":
        return f"Picture {thing} {_cm(change.value)} cm long instead of {_cm(vis['length'])} cm, same diameter, contents and everything else."
    if change.field == "lip":
        return f"Picture the lip {_cm(change.value)} cm high instead of {_cm(vis['lip'])} cm, with nothing else changed."
    if change.field == "dist":
        return f"Picture {thing} lying {_cm(change.value)} cm from the lip's foot instead of {_cm(vis['dist'])} cm, with nothing else changed."
    return f"Picture a tilt of {_num(change.value, 1)} degrees instead of {_num(vis['tilt'], 1)}, with nothing else changed."


# ==========================================================================================================================
# swing
# ==========================================================================================================================


def _swing(vis: dict[str, Any], ch: dict[str, Any], costs: dict[str, float], rng: random.Random) -> tuple[str, dict[str, Any]]:
    load = F.SWING_LOADS[vis["load"]]
    thing = "the " + load["name"].split(" of ")[0]
    b, w, h = load["dims"]
    strap = F.SWING_STRAPS[vis["strap"]]["text"]
    opener = {"gantry": "Overhead gantry speaking: my hook is running along the aisle with a load slung beneath it.",
              "arm": "Mobile arm here, carrying a load that dangles from the hook on my wrist.",
              "rail": "I am the ceiling-rail carrier, moving a hanging load down the bay."}[vis["context"]]
    body = (f"The load is a {load['name']}, {_cm(b)} cm long in the direction I travel, {_cm(w)} cm wide and {_cm(h)} cm tall, "
            f"weighing about {_num(vis['mass'], 1)} kg.")
    hang = _pick(rng, f"It hangs on {strap}; from hook to the top of the load is {_cm(vis['strap_len'])} cm when it is not moving.",
                 f"The strap is {strap}, and at rest {_cm(vis['strap_len'])} cm of it runs between the hook and the top of the load.")
    plan = (f"I am travelling at {_num(vis['speed'])} m/s and will brake to a stop at {_num(vis['decel'], 1)} m/s², building up to that "
            f"braking within a tenth of a second.")
    gap = (f"Right where I stop stands the {vis['obstacle']}: hanging straight down, the front face of {thing} would be "
           f"{_cm(vis['gap'])} cm short of it.")
    state = " ".join([opener, body, hang, plan, gap])
    names = NAMES["swing"]
    obstacle = "the " + vis["obstacle"].split(" of ")[0] if " of " in vis["obstacle"] else "the " + vis["obstacle"]
    q: dict[str, Any] = {}
    q[names["key"]] = {"type": "noul", "instructions": _pick(rng, f"After I stop like this, will {thing} swing into {obstacle}?",
                                                             f"Stopping this way, does {thing} make contact with {obstacle} as it swings?")}
    cf = ch["cf"]
    if cf.field == "speed":
        what = f"my travel speed were {_num(cf.value)} m/s, not {_num(vis['speed'])} m/s, braking just as planned"
    elif cf.field == "strap_len":
        what = f"the hook-to-load length were {_cm(cf.value)} cm instead of {_cm(vis['strap_len'])} cm"
    else:
        what = f"the hanging load stopped {_cm(cf.value)} cm short of {obstacle} instead of {_cm(vis['gap'])} cm"
    q[names["cf"]] = {"type": "score", "instructions": f"Leave the rest alone and suppose {what}. How likely would {thing} be to make contact then?",
                      "criteria": _scale(f"it surely never touches {obstacle}", f"it surely makes contact")}
    q[names["alt"]] = {"type": "noul", "instructions": (f"If I slowed to {_num(ch['alt'])} m/s before braking the same way, would {thing} still make "
                                                        f"contact with {obstacle}?")}
    q[names["dir"]] = _dir_question(_swing_dir_text(ch["dirv"], vis, thing, obstacle), f"{thing} makes contact with {obstacle}")
    q[names["act"]] = _act_question(
        f"Which do I do? A knock against {obstacle} would cost me as much as {_num(costs['fail'], 0)} s of delay.",
        "swing", ["Carry on and stop as planned (no extra time)", f"Slow to {_num(ch['alt'])} m/s first, then brake the same way (adds {_num(costs['alt'], 0)} s)",
                  f"Lower {thing} onto a trolley and push it the last stretch (adds {_num(costs['secure'], 0)} s)"])
    q[names["outcome"]] = _outcome_question(f"Once I have stopped as planned, what happens between {thing} and {obstacle}?", "swing", [
        "No contact at all", f"It just grazes {obstacle}, moving slower than {_num(F.ph.STRIKE_SPEED)} m/s at the touch",
        f"It hits {obstacle} at {_num(F.ph.STRIKE_SPEED)} m/s or more"])
    return state, q


def _swing_dir_text(change: F.Change, vis: dict[str, Any], thing: str, obstacle: str) -> str:
    if change.field == "mass":
        return f"Picture {thing} weighing {_num(change.value, 1)} kg instead of {_num(vis['mass'], 1)} kg, same size, same strap, everything else unchanged."
    if change.field == "strap_len":
        return f"Picture {_cm(change.value)} cm between hook and load instead of {_cm(vis['strap_len'])} cm, with nothing else changed."
    if change.field == "speed":
        return f"Picture my travel speed being {_num(change.value)} m/s, not {_num(vis['speed'])} m/s, braking just as planned and nothing else changed."
    return f"Picture the hanging load stopping {_cm(change.value)} cm short of {obstacle} instead of {_cm(vis['gap'])} cm, nothing else changed."


# ==========================================================================================================================
# knock
# ==========================================================================================================================


def _knock(vis: dict[str, Any], ch: dict[str, Any], costs: dict[str, float], rng: random.Random) -> tuple[str, dict[str, Any]]:
    obj = F.KNOCK_OBJECTS[vis["object"]]
    b, w, h = obj["dims"]
    thing = "the " + obj["name"].split(" of ")[0]
    surface = F.KNOCK_SURFACES[vis["surface"]]["text"]
    limb = {"forearm": "my forearm", "gripper": "the side of my open gripper", "elbow": "my elbow"}[vis["context"]]
    opener = _pick(rng, f"I am clearing a shelf, and on my next motion {limb} will clip something on the way past.",
                   f"Shelf work: my planned path brings {limb} into light contact with an item standing in the way.")
    body = (f"It is a {obj['name']}, {obj['contents']}, standing on {surface}. Its footprint is {_cm(b)} cm in the direction I move by "
            f"{_cm(w)} cm across; it rises {_cm(h)} cm, with a weight of roughly {_num(vis['mass'])} kg.")
    plan = (f"{limb[0].upper() + limb[1:]} moves at {_num(vis['speed'])} m/s and meets its face {_cm(vis['height'])} cm above the shelf. "
            f"The moment I feel the touch I brake, coming to rest within {_cm(vis['stop'])} cm, then pull back 10 cm.")
    state = " ".join([opener, body, plan])
    names = NAMES["knock"]
    q: dict[str, Any] = {}
    q[names["key"]] = {"type": "noul", "instructions": _pick(rng, f"Once this is over, is {thing} lying knocked over?",
                                                             f"After the touch and my pull-back, has {thing} fallen over?")}
    cf = ch["cf"]
    what = (f"the point of touch sat {_cm(cf.value)} cm high on its face, not {_cm(vis['height'])} cm" if cf.field == "height"
            else f"after the touch I came to rest within {_cm(cf.value)} cm instead of {_cm(vis['stop'])} cm")
    q[names["cf"]] = {"type": "score", "instructions": f"Change only one thing: say {what}. How likely would {thing} be to fall over?",
                      "criteria": _scale(f"{thing} surely stays on its base", f"{thing} surely falls over")}
    q[names["alt"]] = {"type": "noul", "instructions": f"If I made the same pass slowed to {_num(ch['alt'])} m/s, would {thing} fall over?"}
    q[names["dir"]] = _dir_question(_knock_dir_text(ch["dirv"], vis, thing), f"{thing} falls over")
    q[names["act"]] = _act_question(
        f"Which do I do? {thing[0].upper() + thing[1:]} falling over would cost me as much as {_num(costs['fail'], 0)} s of delay.",
        "knock", ["Make the pass as planned (no extra time)", f"Make the pass at {_num(ch['alt'])} m/s (adds {_num(costs['alt'], 0)} s)",
                  f"Pick {thing} up and set it aside first (adds {_num(costs['secure'], 0)} s)"])
    q[names["outcome"]] = _outcome_question(f"After the pass as planned, what has become of {thing}?", "knock", [
        "Still standing within a centimetre of where it was", "Still upright but pushed a centimetre or more along the shelf", "Knocked over"])
    return state, q


def _knock_dir_text(change: F.Change, vis: dict[str, Any], thing: str) -> str:
    if change.field == "mass":
        return f"Picture {thing} weighing {_num(change.value)} kg instead of {_num(vis['mass'])} kg, same size and weight layout, everything else unchanged."
    if change.field == "height":
        return f"Picture the point of touch sitting {_cm(change.value)} cm high on its face, not {_cm(vis['height'])} cm, nothing else changed."
    if change.field == "speed":
        return f"Picture my pass running {_num(change.value)} m/s, not {_num(vis['speed'])} m/s, nothing else changed."
    if change.field == "stop":
        return f"Picture me coming to rest within {_cm(change.value)} cm of the touch instead of {_cm(vis['stop'])} cm, nothing else changed."
    return f"Picture it standing on {F.KNOCK_SURFACES[change.value]['text']} instead, nothing else changed."


# ==========================================================================================================================
# conveyor
# ==========================================================================================================================


def _conveyor(vis: dict[str, Any], ch: dict[str, Any], costs: dict[str, float], rng: random.Random) -> tuple[str, dict[str, Any]]:
    item = F.CONVEYOR_ITEMS[vis["item"]]
    thing = "the " + item["name"].split(" of ")[0]
    b, w, h = F.FAMILIES["conveyor"].dims(vis)
    belt = F.CONVEYOR_BELTS[vis["belt"]]["text"]
    opener = {"packing": "I run the infeed conveyor of my packing cell and decide how it starts and stops.",
              "sorting": "Sorting station robot: the takeaway belt in front of me answers to my commands."}[vis["context"]]
    body = (f"Standing upright on it is a {item['name']}, {item['contents']}: {_cm(b)} cm deep along the belt, {_cm(w)} cm across and "
            f"{_cm(h)} cm high, roughly {_num(vis['mass'])} kg in weight. The belt is {belt}.")
    if vis["event"] == "stop":
        plan = (f"The belt now runs at {_num(vis['speed'])} m/s, and I am about to halt it at {_num(vis['rate'], 1)} m/s², reaching that rate "
                f"within {_num(vis['ramp'])} s.")
    else:
        plan = (f"The belt is standing still, and I am about to start it, speeding up at {_num(vis['rate'], 1)} m/s² — I get to that rate in "
                f"{_num(vis['ramp'])} s — until it runs at {_num(vis['speed'])} m/s.")
    state = " ".join([opener, body, plan])
    names = NAMES["conveyor"]
    verb = "halt" if vis["event"] == "stop" else "start"
    q: dict[str, Any] = {}
    q[names["key"]] = {"type": "noul", "instructions": _pick(rng, f"If I {verb} the belt this way, does {thing} end up tipped over?",
                                                             f"With this {'halt' if verb == 'halt' else 'start'}, is {thing} lying on its side afterwards?")}
    cf = ch["cf"]
    what = (f"I used {_num(cf.value, 1)} m/s² instead of {_num(vis['rate'], 1)} m/s²" if cf.field == "rate"
            else f"the belt speed were {_num(cf.value)} m/s instead of {_num(vis['speed'])} m/s at the same rate")
    q[names["cf"]] = {"type": "score", "instructions": f"Alter just one thing: suppose {what}. How likely would {thing} be to tip over?",
                      "criteria": _scale(f"{thing} surely stays on its base", f"{thing} surely tips over")}
    past = "halted" if verb == "halt" else "started"
    q[names["alt"]] = {"type": "noul", "instructions": f"If I {past} the belt more softly, with {_num(ch['alt'], 1)} m/s², would {thing} tip over?"}
    q[names["dir"]] = _dir_question(_conveyor_dir_text(ch["dirv"], vis, thing), f"{thing} tips over")
    q[names["act"]] = _act_question(
        f"Which do I do? {thing[0].upper() + thing[1:]} tipping over would cost me as much as {_num(costs['fail'], 0)} s of delay.",
        "conveyor", [f"{verb.capitalize()} the belt as planned (no extra time)", f"{verb.capitalize()} it at {_num(ch['alt'], 1)} m/s² (adds {_num(costs['alt'], 0)} s)",
                     f"Take {thing} off the belt first and put it back afterwards (adds {_num(costs['secure'], 0)} s)"])
    q[names["outcome"]] = _outcome_question(f"After the {'halt' if verb == 'halt' else 'start'} as planned, what has {thing} done?", "conveyor", [
        "Kept its spot on the belt, upright", "Skidded two centimetres or more along the belt but stayed upright", "Tipped over"])
    return state, q


def _conveyor_dir_text(change: F.Change, vis: dict[str, Any], thing: str) -> str:
    if change.field == "mass":
        return f"Picture {thing} weighing {_num(change.value)} kg instead of {_num(vis['mass'])} kg, with the weight spread the same way and nothing else changed."
    if change.field == "rate":
        return f"Picture a rate of {_num(change.value, 1)} m/s² instead of {_num(vis['rate'], 1)} m/s², nothing else changed."
    if change.field == "speed":
        return f"Picture a belt speed of {_num(change.value)} m/s instead of {_num(vis['speed'])} m/s at the same rate, nothing else changed."
    if change.field == "turned":
        side = "its wider side" if change.value == "wide" else "its narrower side"
        return f"Picture {thing} turned a quarter turn, so that {side} faces along the belt, nothing else changed."
    return f"Picture the belt being {F.CONVEYOR_BELTS[change.value]['text']} instead, nothing else changed."


# ==========================================================================================================================
# overhang
# ==========================================================================================================================


def _overhang(vis: dict[str, Any], ch: dict[str, Any], costs: dict[str, float], rng: random.Random) -> tuple[str, dict[str, Any]]:
    item = F.OVERHANG_ITEMS[vis["item"]]
    board = F.OVERHANG_BOARDS[vis["board"]]["text"]
    thing = "the " + item["name"].split(" of ")[0]
    d, w, _ = item["dims"]
    opener = {"pantry": "I am putting the shopping away on the pantry shelves.", "workshop": "I am tidying the workshop shelving."}[vis["context"]]
    board_s = (f"A loose {board} lies on a shelf, {_cm(vis['board_len'])} cm long, {_cm(vis['board_w'])} cm wide and {_cm(vis['board_t'])} cm "
               f"thick, with {_cm(vis['over'])} cm of it reaching beyond the shelf's front lip. Nothing fixes it down.")
    contents = f", {item['contents']}" if item["contents"] else ""
    weight = (f"about {_num(item['mass'], 1)} kg" if item["stated"] else f"about {_num(item['mass'], 1)} kg when full")
    item_s = (f"In my gripper is a {item['name']}{contents}, {_cm(d)} cm along the board by {_cm(w)} cm and {_cm(vis['item_height'])} cm high; "
              f"it should weigh {weight}.")
    inset = vis["inset"]
    where = (f"its middle {_cm(inset)} cm in from the board's outer end" if inset > 0.0005 else
             ("its middle right at the board's outer end" if inset > -0.0005 else f"its middle {_cm(-inset)} cm beyond the board's outer end"))
    plan = (f"I plan to set it down flat on the sticking-out part with {where}, letting go just above the board. My placing is good to "
            f"within {_cm(vis['place_tol'])} cm.")
    state = " ".join([opener, board_s, item_s, plan])
    names = NAMES["overhang"]
    q: dict[str, Any] = {}
    q[names["key"]] = {"type": "noul", "instructions": _pick(rng, f"If I set {thing} down as planned, does the board or {thing} end up falling off?",
                                                             f"Placing it as planned, does anything (the board or {thing}) come down off the shelf?")}
    cf = ch["cf"]
    if cf.field == "over":
        what = f"the board reached {_cm(cf.value)} cm beyond the shelf's lip, not {_cm(vis['over'])} cm"
    else:
        what = f"I set {thing} with its middle {_cm(cf.value)} cm in from the board's end instead of {_cm(inset)} cm" if cf.value > 0 else \
            f"I set {thing} with its middle {_cm(-cf.value)} cm beyond the board's end instead"
    q[names["cf"]] = {"type": "score", "instructions": f"Only one thing different: suppose {what}. How likely would something fall off then?",
                      "criteria": _scale("everything surely stays up", "something surely comes down")}
    q[names["alt"]] = {"type": "noul", "instructions": (f"If I set {thing} with its middle {_cm(ch['alt'])} cm in from the board's outer end instead, "
                                                        "would the board or it fall off?")}
    q[names["dir"]] = _dir_question(_overhang_dir_text(ch["dirv"], vis, thing), "the board or the item comes down")
    q[names["act"]] = _act_question(
        f"Which do I do? Anything landing on the floor would cost me as much as {_num(costs['fail'], 0)} s of delay.",
        "overhang", ["Place it as planned (no extra time)", f"Place it with its middle {_cm(ch['alt'])} cm in from the end (adds {_num(costs['alt'], 0)} s to clear room)",
                     f"Slide the board fully onto the shelf before placing (adds {_num(costs['secure'], 0)} s)"])
    q[names["outcome"]] = _outcome_question(f"After I let go as planned, what is the result?", "overhang", [
        "Board and item both stay put", f"{thing[0].upper() + thing[1:]} topples off the board's end; the board stays", "The board tips off the shelf"])
    return state, q


def _overhang_dir_text(change: F.Change, vis: dict[str, Any], thing: str) -> str:
    if change.field == "item_height":
        return f"Picture {thing} {_cm(change.value)} cm tall instead of {_cm(vis['item_height'])} cm, same weight and footprint, nothing else changed."
    if change.field == "over":
        return f"Picture the board reaching {_cm(change.value)} cm beyond the lip, not {_cm(vis['over'])} cm, nothing else changed."
    if change.field == "inset":
        return (f"Picture {thing} set with its middle {_cm(change.value)} cm in from the board's end instead of {_cm(vis['inset'])} cm, nothing else changed."
                if change.value > 0 else f"Picture {thing} set with its middle {_cm(-change.value)} cm beyond the board's end, nothing else changed.")
    if change.field == "board_t":
        return f"Picture the board {_cm(change.value)} cm thick instead of {_cm(vis['board_t'])} cm, same material and outline, nothing else changed."
    if change.field == "board_len":
        return (f"Picture the board {_cm(change.value)} cm long instead of {_cm(vis['board_len'])} cm, still reaching {_cm(vis['over'])} cm beyond "
                "the lip, nothing else changed.")
    return f"Picture the board being a {F.OVERHANG_BOARDS[change.value]['text']} of the same size instead, nothing else changed."


# ==========================================================================================================================
# suction
# ==========================================================================================================================


def _suction(vis: dict[str, Any], ch: dict[str, Any], costs: dict[str, float], rng: random.Random) -> tuple[str, dict[str, Any]]:
    panel = F.SUCTION_PANELS[vis["panel"]]
    pl, pw, pt = panel["dims"]
    thing = "the " + panel["name"].split()[-1]
    opener = {"glazing": "Glazing cell, vacuum gripper on my wrist.", "kitting": "I am the kitting robot, holding a flat part on my suction cup."}[vis["context"]]
    body = (f"Hanging flat under one {_cm(vis['cup'])} cm cup is a {panel['name']}, {_cm(pl)} by {_cm(pw)} cm and {_num(1000 * pt, 0)} mm thick, "
            f"about {_num(panel['mass'], 1)} kg. Its upper face is {F.SUCTION_FACES[vis['face']]['text']}, and "
            f"{F.SUCTION_PUMPS[vis['pump']]['text']}.")
    off = (f"I gripped it {_cm(vis['offset'])} cm off its centre, toward the front." if vis["offset"] >= 0.0005 else "I gripped it at its centre.")
    plan = (f"Next I carry it {_num(vis['carry'], 0)} m sideways: I accelerate by {_num(vis['rate'], 1)} m/s² to a pace of {_num(vis['speed'])} m/s, "
            f"keep that pace, then slow by the same {_num(vis['rate'], 1)} m/s².")
    state = " ".join([opener, body, off, plan])
    names = NAMES["suction"]
    q: dict[str, Any] = {}
    q[names["key"]] = {"type": "noul", "instructions": _pick(rng, f"During this carry, does {thing} come off my cup?",
                                                             f"Over the whole carry as planned, will {thing} part from the cup?")}
    cf = ch["cf"]
    if cf.field == "rate":
        what = f"I sped up and slowed down at {_num(cf.value, 1)} m/s² instead of {_num(vis['rate'], 1)} m/s²"
    elif cf.field == "speed":
        what = f"the top speed were {_num(cf.value)} m/s instead of {_num(vis['speed'])} m/s, at the same rates"
    else:
        what = (f"I had gripped it {_cm(cf.value)} cm off its centre instead" if cf.value >= 0.0005 else "I had gripped it right at its centre instead")
    q[names["cf"]] = {"type": "score", "instructions": f"With the rest unchanged, suppose {what}. How likely would {thing} be to come off?",
                      "criteria": _scale(f"{thing} surely stays on the cup", f"{thing} surely comes off")}
    q[names["alt"]] = {"type": "noul", "instructions": f"If I sped up and slowed down more gently, by {_num(ch['alt'], 1)} m/s², would {thing} come off?"}
    q[names["dir"]] = _dir_question(_suction_dir_text(ch["dirv"], vis, thing), f"{thing} comes off the cup")
    q[names["act"]] = _act_question(
        f"Which do I do? {thing[0].upper() + thing[1:]} coming off would cost me as much as {_num(costs['fail'], 0)} s of delay.",
        "suction", ["Carry it as planned (no extra time)", f"Speed up and slow down at {_num(ch['alt'], 1)} m/s² (adds {_num(costs['alt'], 0)} s)",
                    f"Fit a second cup and regrip before moving (adds {_num(costs['secure'], 0)} s)"])
    q[names["outcome"]] = _outcome_question(f"By the end of the carry as planned, how has {thing} fared?", "suction", [
        "Stayed put on the cup (crept less than 3 mm)", "Crept 3 mm or more across the cup but is still held", "Came off the cup"])
    return state, q


def _suction_dir_text(change: F.Change, vis: dict[str, Any], thing: str) -> str:
    if change.field == "carry":
        return f"Picture the carry being {_num(change.value, 0)} m long instead of {_num(vis['carry'], 0)} m, same speeds and rates, nothing else changed."
    if change.field == "rate":
        return f"Picture rates of {_num(change.value, 1)} m/s² instead of {_num(vis['rate'], 1)} m/s², nothing else changed."
    if change.field == "cup":
        return f"Picture a {_cm(change.value)} cm cup instead of the {_cm(vis['cup'])} cm one, same pump, nothing else changed."
    return f"Picture its upper face being {F.SUCTION_FACES[change.value]['text']} instead, nothing else changed."


# ==========================================================================================================================
# shared question builders
# ==========================================================================================================================


def _dir_question(change_text: str, clause: str) -> dict[str, Any]:
    return {"type": "choice", "instructions": f"{change_text} With that change, do the odds rise, fall, or hold within ten points of the moment as it "
                                              f"stands, for this outcome: {clause}?", "criteria": dict(DIR_OPTIONS)}


def _act_question(head: str, family_name: str, texts: list[str]) -> dict[str, Any]:
    keys = ACT_OPTIONS[family_name]
    return {"type": "choice", "instructions": head, "criteria": dict(zip(keys, texts))}


def _outcome_question(head: str, family_name: str, texts: list[str]) -> dict[str, Any]:
    keys = OUTCOME_OPTIONS[family_name]
    return {"type": "choice", "instructions": head, "criteria": dict(zip(keys, texts))}


_RENDER = {"roll": _roll, "swing": _swing, "knock": _knock, "conveyor": _conveyor, "overhang": _overhang, "suction": _suction}


def render(family_name: str, vis: dict[str, Any], changes: dict[str, Any], costs: dict[str, float], seed: int) -> tuple[str, dict[str, Any]]:
    """(state text, questions) of one moment. `changes`: ``cf`` (:class:`~robo_jev.t9.families.Change`), ``alt`` (the gentler action's
    number), ``dirv`` (Change). `costs`: ``fail`` (s of delay a failure costs), ``alt`` and ``secure`` (s added). `seed` picks the
    sentence variants (both records of a pair use the same seed, so they read alike but for the changed fact)."""
    return _RENDER[family_name](vis, changes, costs, random.Random(int(seed)))
