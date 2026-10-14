"""Author A2 — wording. Every state is a short second-person briefing from a human supervisor ("You're about to …"), informal, in short
paragraphs, with everyday hedges ("about", "roughly", "-ish"), English and metric units. Questions are this author's own.

Template choices are drawn once per group (:func:`choices`) and both members of a pair are rendered with the same choices, so a
one-fact pair differs only in the changed fact.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from robo_jev.authors.a2 import catalog as C

__all__ = ["choices", "render"]

#: Number of template variants per slot (choices draw an index per slot).
SLOTS = {"open": 8, "plan": 4, "item": 3, "surface": 3, "close": 6, "hedge": 4, "q_key": 3, "q_severe": 3, "q_score": 2, "q_dir": 2,
         "q_cmp": 2, "q_act": 3, "q_out": 2}

OPENERS = ("", "Quick one before you go. ", "Heads up. ", "Okay, next job. ", "Right, listen up. ", "One for you. ", "Before you start: ",
           "Alright. ")
CLOSERS = ("", "", "", " Shout if anything looks off.", " Tell me how you see it.", " I'll be right here.")
HEDGES = ("about", "roughly", "around", "about")


def choices(rng: np.random.Generator) -> dict[str, int]:
    return {slot: int(rng.integers(n)) for slot, n in SLOTS.items()}


def _num(x: float) -> str:
    """A stated number: integers without a decimal point, others with one decimal."""
    x = float(x)
    return str(int(round(x))) if abs(x - round(x)) < 1e-9 else f"{x:.1f}"


def _hedge(ch: Mapping[str, int], shift: int = 0) -> str:
    return HEDGES[(ch["hedge"] + shift) % len(HEDGES)]


def _a(number: str) -> str:
    """The article before a spoken number ("an 8", "an 11.5", "an 18", "an 80", "a 9")."""
    whole = number.split(".")[0]
    return "an" if whole.startswith("8") or whole in ("11", "18") else "a"


def _open(ch: Mapping[str, int], plan: str) -> str:
    """An opener only in front of a plan that starts with "You're" (others carry their own lead-in)."""
    return (OPENERS[ch["open"]] if plan.startswith("You're") else "") + plan


# --------------------------------------------------------------------------
# A — turntable
# --------------------------------------------------------------------------

_TT_SHORT = {"bottle": "bottle", "vase": "vase"}
_TT_SHORT_ITEM = {"carafe": "carafe", "jug": "jug", "flowerbox": "flower box"}


def _tt_short(facts: Mapping[str, Any]) -> str:
    return _TT_SHORT_ITEM.get(facts["item"], _TT_SHORT[C.TT_ITEMS[facts["item"]]["family"]])


def _turntable(facts: Mapping[str, Any], ch: Mapping[str, int], cf: float, alt: str, alt_values: Mapping[str, Any]) -> tuple[str, dict]:
    spec = C.TT_ITEMS[facts["item"]]
    short = _tt_short(facts)
    h, h2 = _hedge(ch), _hedge(ch, 1)
    turn = C.TT_TURNS[facts["turn"]][1]
    t = _num(facts["duration"])
    spin = facts["spin_dir"]
    venue, seat = facts["venue"], facts["seat"]
    plans = (
        f"You're about to spin the lazy Susan on {venue} so seat {seat} gets the {short}: {turn} {spin}, done in {h} {t} s.",
        f"You're about to give the lazy Susan at {venue} {turn} {spin}, {h} {t} seconds start to stop, so seat {seat} can reach the {short}.",
        f"Next you'll turn the lazy Susan on {venue} by {turn} to bring the {short} to seat {seat}. You've timed it at {h} {t} s.",
        f"You're going to swing the lazy Susan at {venue} {turn} {spin} for seat {seat}. Planned time: {t}-ish seconds.",
    )
    side = _num(facts["side_cm"])
    contents = C.TT_CONTENTS[spec["family"]][facts["contents"]][0]
    base = C.TT_BASES[facts["base"]][0]
    if facts["orient"] == "square":
        size = f"{h2} {side} cm square and {_num(facts['height_cm'])} cm tall"
        lie = ""
    else:
        other = _num(round(facts["side_cm"] * facts["ratio"] * 2) / 2)
        size = f"{h2} {side} by {other} cm and {_num(facts['height_cm'])} cm tall"
        lie = (", long side pointing at the centre" if facts["orient"] == "radial" else ", long side along the rim")
    r = facts["radius_cm"]
    items = (
        f"It's a {spec['name']}, {size}, {contents}. It stands {h} {r} cm out from the centre{lie}, on {base}.",
        f"About the {short}: {spec['name']}, {size}, and it's {contents}. It has {base} and sits {h} {r} cm from the middle{lie}.",
        f"On it: a {spec['name']}, {contents}, {size}, {r}-ish cm from the centre{lie}, on {base}.",
    )
    top, cond = C.TT_TOPS[facts["top"]][0], C.TT_CONDITIONS[facts["condition"]][0]
    surfaces = (f" The top's {top} and {cond}.", f" Top: {top}, {cond}.", f" That {top} top is {cond}.")
    state = (_open(ch, plans[ch["plan"]]) + "\n\n" + items[ch["item"]] + surfaces[ch["surface"]]).strip()
    cfs = _num(cf)
    if alt == "turn_slower":
        alt_label = f"turn slower, in {h} {_num(alt_values['duration'])} s"
    elif alt == "move_it_inward":
        alt_label = f"first slide it in to {h} {C.TT_INWARD_CM} cm from the centre"
    else:
        other = {**C.TT_TURNS, **C.TT_LONG_TURNS}[C.TT_OTHER_WAY[facts["turn"]]][1]
        alt_label = f"go the other way round ({other}), same time"
    questions = {
        "shifts_during_turn": {"type": "noul", "instructions": (
            f"As planned, will the {short} skid or tip at all?", f"Does the {short} move during the turn (slide or go over)?",
            f"Turning it as planned, will the {short} shift on the top?")[ch["q_key"]]},
        "tips_over_during_turn": {"type": "noul", "instructions": (
            f"Will the {short} actually tip over?", f"Will the {short} be knocked flat?", f"As planned, will the {short} topple?")[ch["q_severe"]]},
        "shift_risk_other_time": {"type": "score", "instructions": (
            f"Rate the chance the {short} shifts if the turn took {h} {cfs} s instead.",
            f"Take {_a(cfs)} {cfs}-second turn instead: what are the odds the {short} skids or tips?")[ch["q_score"]],
            "criteria": ["0 = stays put for sure", "1 = shifts for sure"]},
        "if_turn_took_other_time": {"type": "choice", "instructions": (
            f"If the turn took {h} {cfs} s rather than {t} s, is a shift more or less likely?",
            f"What does {_a(cfs)} {cfs}-second turn do to the chance of the {short} shifting?")[ch["q_dir"]],
            "criteria": {"riskier": "more likely", "safer": "less likely", "no_change": "about the same"}},
        "riskier_option": {"type": "choice", "instructions": (
            f"Which is likelier to end with the {short} shifting?", f"Which way makes a skid or tip more likely?")[ch["q_cmp"]],
            "criteria": {"turn_as_planned": "turning as planned", alt: alt_label, "roughly_equal": "about the same"}},
        "best_move": {"type": "choice", "instructions": (
            f"Your best move? A spill or a smashed {short} costs far more than a few seconds.",
            f"Which would you do? Keep the {short} safe without wasting the table's time.",
            "Pick one. Breakage and mess are the expensive part; a bit of time isn't.")[ch["q_act"]],
            "criteria": {"turn_as_planned": "spin as planned", alt: alt_label, "carry_it_round": "carry it round by hand (slow)"}},
        "what_the_item_does": {"type": "choice", "instructions": (
            f"What does the {short} do during the turn?", f"How does the {short} come through it?")[ch["q_out"]],
            "criteria": {"stays_put": "stays put", "skids": "skids but stays upright", "tips_over": "tips over"}},
    }
    return state, questions


# --------------------------------------------------------------------------
# B — tiltback
# --------------------------------------------------------------------------


def _tb_alt(alt: str) -> str:
    if alt == "tip_it_sideways":
        return "tip it onto a side edge instead"
    if alt == "empty_top_first":
        return "clear the top first"
    if alt == "load_top_first":
        return "load the floor boxes up top first (saves a trip)"
    raise ValueError(alt)


def _tiltback(facts: Mapping[str, Any], ch: Mapping[str, int], cf: float, alt: str, alt_values: Mapping[str, Any]) -> tuple[str, dict]:
    spec = C.TB_PIECES[facts["piece"]]
    name = spec["name"]
    h, h2 = _hedge(ch), _hedge(ch, 1)
    th = _num(facts["tilt_deg"])
    venue, task = facts["venue"], facts["task"]
    plans = (
        f"You're about to tip the {name} in {venue} back {h} {th}° on its back edge so I can {task}, holding it by the top front edge.",
        f"Next job: the {name} in {venue}. Tilt it back {h} {th}° on its rear edge and hold the top while we {task}.",
        f"You're going to lean the {name} in {venue} back, {th}-ish degrees, and hold it by the top so I can {task}.",
        f"We need to {task}, so you'll tip the {name} in {venue} back {h} {th} degrees and hold it from the front.",
    )
    build = C.TB_BUILDS[facts["build"]][0]
    contents = spec["contents"][facts["contents"]]
    height_m = f"{facts['height_cm'] / 100:.2f}".rstrip("0").rstrip(".")
    size = f"{h2} {height_m} m tall, {facts['depth_cm']} cm deep, {facts['width_cm']} cm wide"
    items = (f"It's {build}, {size}; {contents}.", f"The {name} is {build}, {size}; {contents}.", f"It's {size}, {build}. Loaded: {contents}.")
    floor = C.TB_FLOORS[facts["floor"]][0]
    limit = _num(facts["limit_n"])
    surfaces = (f" Floor's {floor}. Up there your arm holds back {h} {limit} N.", f" It's on {floor}; your arm is good for {h} {limit} N up there.",
                f" Floor: {floor}. Your arm tops out at {limit}-ish N there.")
    state = (_open(ch, plans[ch["plan"]]) + "\n\n" + items[ch["item"]] + surfaces[ch["surface"]]).strip()
    cfs = _num(cf)
    alt_label = _tb_alt(alt)
    questions = {
        "past_balance_point": {"type": "noul", "instructions": (
            "At that angle, is it past its balance point (hanging on your hand)?",
            f"Tipped {th}°, will it want to keep going over rather than drop back?",
            "At the planned angle, is its weight behind the back edge?")[ch["q_key"]]},
        "gets_away_from_you": {"type": "noul", "instructions": (
            "Will it get away from you and go over backwards?", "Does it go over despite your grip?",
            "Will your arm fail to hold it, so it falls back?")[ch["q_severe"]]},
        "past_point_at_other_angle": {"type": "score", "instructions": (
            f"Rate the chance it's past that point if tipped to {h} {cfs}° instead.",
            f"Take {_a(cfs)} {cfs}° tilt instead: what are the odds it's past that point?")[ch["q_score"]],
            "criteria": ["0 = settles back for sure", "1 = past the point for sure"]},
        "if_tipped_other_angle": {"type": "choice", "instructions": (
            f"Tipped to {h} {cfs}° versus the planned {th}°, is it more or less likely to be past that point?",
            f"What does {_a(cfs)} {cfs}° tilt do to the chance it's past that point?")[ch["q_dir"]],
            "criteria": {"riskier": "more likely", "safer": "less likely", "no_change": "about the same"}},
        "riskier_option": {"type": "choice", "instructions": (
            "Which is likelier to leave it past its balance point?", "Which way gets it past its balance point more easily?")[ch["q_cmp"]],
            "criteria": {"tip_as_planned": "tipping as planned", alt: alt_label, "about_the_same": "about the same"}},
        "best_move": {"type": "choice", "instructions": (
            "Smartest way to do it? It going over is far worse than a few extra minutes.",
            "Which would you do? Nobody wants it crashing down, but the room's due today.",
            "Pick one. A fall is the expensive part; a bit of time isn't.")[ch["q_act"]],
            "criteria": {"tip_as_planned": "tip it as planned", alt: alt_label, "use_the_lifter": "use the lever lifter (slow, no tipping)"}},
        "where_it_ends_up": {"type": "choice", "instructions": (
            "While you hold it there, how does it end up?", "How does it behave at that angle?")[ch["q_out"]],
            "criteria": {"settles_forward": "wants to drop back onto its feet", "hangs_on_hand": "hangs back on your hand", "goes_over": "goes over"}},
    }
    return state, questions


# --------------------------------------------------------------------------
# C — wallpress
# --------------------------------------------------------------------------

_WP_SHORT = {"mirror": "mirror", "print": "print", "pinboard": "pinboard", "cabinet": "cabinet", "headboard": "headboard", "clock": "clock"}


def _wp_alt(alt: str, facts: Mapping[str, Any], h: str) -> str:
    if alt == "use_both_pads":
        return f"press with both pads, {h} {_num(facts['force_n'])} N each"
    if alt == "rubber_sheet_behind":
        return "put a rubber sheet behind it, same push"
    if alt == "felt_pad_instead":
        return "swap to the felt-faced pad (kinder to the finish)"
    raise ValueError(alt)


def _wallpress(facts: Mapping[str, Any], ch: Mapping[str, int], cf: float, alt: str, alt_values: Mapping[str, Any]) -> tuple[str, dict]:
    spec = C.WP_ITEMS[facts["item"]]
    name, short = spec["name"], _WP_SHORT[facts["item"]]
    h, h2 = _hedge(ch), _hedge(ch, 1)
    n = _num(facts["force_n"])
    pad = C.WP_PADS[facts["pad"]][0]
    venue, helper = facts["venue"], facts["helper"]
    plans = (
        f"You're about to hold the {name} flat on the wall in {venue} while {helper} marks the holes. Only you hold it up: {pad} in the middle, {h} {n} N.",
        f"Job for you: press the {name} against the wall in {venue} so {helper} can fix it. {pad.capitalize()}, dead centre, {h} {n} N, nothing else holding it.",
        f"You're going to pin the {name} to the wall in {venue} with {pad} at {n}-ish newtons while {helper} gets the drill. No ledge under it.",
        f"{helper} needs the {name} held to the wall in {venue}. You'll push {pad} into the middle at {h} {n} N, nothing underneath.",
    )
    build = C.WP_BUILDS[facts["build"]][0]
    back = C.WP_BACKS[facts["back"]][0] if facts["back"] in C.WP_BACKS else "a rubber sheet"
    size = f"{_num(facts['thick_cm'])} cm thick and {h2} {facts['width_cm']} by {facts['height_cm']} cm"
    items = (f"It's {size}, {build}, with {back} on the back.", f"The {short} is {size}, {build}, with {back} behind.",
             f"It's {size}, {build}; {back} on the back.")
    wall = C.WP_WALLS[facts["wall"]][0]
    rx = _num(facts["reflex"])
    surfaces = (f" Wall's {wall}. If it slides, your reflex pushes up to {h} {rx}× as hard.",
                f" Wall: {wall}. Your slip reflex can raise the push to {h} {rx} times.",
                f" It's going on {wall}. If it creeps, your reflex ramps to {rx}-ish times the push.")
    state = (_open(ch, plans[ch["plan"]]) + "\n\n" + items[ch["item"]] + surfaces[ch["surface"]]).strip()
    cfs = _num(cf)
    alt_label = _wp_alt(alt, facts, h)
    questions = {
        "slips_at_all": {"type": "noul", "instructions": (
            f"With that push, will the {short} slip down the wall at all?", f"Held as planned, does the {short} slide down even a little?",
            f"Will the {short} move down the wall while you hold it?")[ch["q_key"]]},
        "drops_off_wall": {"type": "noul", "instructions": (
            "Will it get away from you and drop, not just slip a bit?", f"Does the {short} fall, even with your reflex?",
            f"Will the {short} slide right out from under the pad?")[ch["q_severe"]]},
        "slip_risk_other_push": {"type": "score", "instructions": (
            f"Rate the chance it slips at all if you pushed {h} {cfs} N instead.",
            f"Take {_a(cfs)} {cfs} N push instead: what are the odds the {short} slides?")[ch["q_score"]],
            "criteria": ["0 = won't budge", "1 = slips for sure"]},
        "if_push_was_other": {"type": "choice", "instructions": (
            f"Pushing {h} {cfs} N, not {n} N: is a slip more or less likely?",
            f"What does {_a(cfs)} {cfs} N push do to the chance of the {short} slipping?")[ch["q_dir"]],
            "criteria": {"riskier": "more likely", "safer": "less likely", "no_change": "about the same"}},
        "riskier_option": {"type": "choice", "instructions": (
            f"Which is likelier to end with the {short} slipping?", "Which way lets it slip more easily?")[ch["q_cmp"]],
            "criteria": {"hold_as_planned": "holding as planned", alt: alt_label, "about_the_same": "about the same"}},
        "best_move": {"type": "choice", "instructions": (
            f"Your best call? A dropped {short} costs far more than a minute of prep.",
            f"Which would you do? Don't drop it, but {helper} is waiting.",
            "Pick one. A fall is the expensive part; a bit of time isn't.")[ch["q_act"]],
            "criteria": {"hold_as_planned": "hold it as planned", alt: alt_label, "batten_first": "screw a batten under it first (slow, can't slip)"}},
        "how_the_hold_goes": {"type": "choice", "instructions": ("How does the hold go?", f"What happens to the {short}?")[ch["q_out"]],
            "criteria": {"holds_still": "stays put", "slips_then_caught": "slips, then your reflex catches it", "drops": "slides away and drops"}},
    }
    return state, questions


#: Outcome name → the option key of the outcome-category question, per scene.
OUTCOME_KEYS = {
    "turntable": {"stays": "stays_put", "skids": "skids", "topples": "tips_over"},
    "tiltback": {"rests": "settles_forward", "hangs": "hangs_on_hand", "over": "goes_over"},
    "wallpress": {"holds": "holds_still", "slips": "slips_then_caught", "drops": "drops"},
}
#: Question names by kind, per scene (meta.question_kind).
QUESTION_KINDS = {
    "turntable": {"shifts_during_turn": "key", "tips_over_during_turn": "severe", "shift_risk_other_time": "score_cf",
                  "if_turn_took_other_time": "direction", "riskier_option": "compare", "best_move": "action", "what_the_item_does": "outcome"},
    "tiltback": {"past_balance_point": "key", "gets_away_from_you": "severe", "past_point_at_other_angle": "score_cf",
                 "if_tipped_other_angle": "direction", "riskier_option": "compare", "best_move": "action", "where_it_ends_up": "outcome"},
    "wallpress": {"slips_at_all": "key", "drops_off_wall": "severe", "slip_risk_other_push": "score_cf", "if_push_was_other": "direction",
                  "riskier_option": "compare", "best_move": "action", "how_the_hold_goes": "outcome"},
}
#: The option keys of the direction question: (more, less, same).
DIRECTION_KEYS = {"turntable": ("riskier", "safer", "no_change"), "tiltback": ("riskier", "safer", "no_change"),
                  "wallpress": ("riskier", "safer", "no_change")}
#: The "about the same" key of the comparison question.
SAME_KEYS = {"turntable": "roughly_equal", "tiltback": "about_the_same", "wallpress": "about_the_same"}


def render(scene: str, facts: Mapping[str, Any], ch: Mapping[str, int], cf: float, alt: str, alt_values: Mapping[str, Any]) -> tuple[str, dict]:
    """(state, questions) of one record."""
    fn = {"turntable": _turntable, "tiltback": _tiltback, "wallpress": _wallpress}[scene]
    return fn(facts, ch, cf, alt, alt_values)
