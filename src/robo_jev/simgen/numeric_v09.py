"""Task v0.9a — measurement-literacy records (arm CN of the PRE-DECLARED v0.9 VERDICT RULE): stated numeric rules applied to stated numbers.

Every record is a Decisions request (``simgen-request/0``) whose questions are choice questions with exact one-hot labels **by
construction**: the generator draws the numbers, renders them, and labels the rendered values with exact rational arithmetic
(:class:`fractions.Fraction`, so a reading of ``1.35 m`` against a limit of ``135 cm`` compares exactly). ``meta.numeric`` records, per
question, every rendered number string as it appears in the text and the rule's structure, so that an independent re-labeller (the tests)
can find each number in the text, parse it with its own parser and unit table, and recompute the answer.

**Five families** (:data:`FAMILIES`), each with several paraphrased templates of this module's own (:data:`TEMPLATES`):

* ``step`` — threshold / step: the sign of a signed reading picks the direction, its size against ordered thresholds picks the step
  category, a dead band (when the rule has one) picks the hold option. Threshold sets, category names, direction words, key formats and
  boundary conventions (``at least`` vs ``more than``) vary; 1–3 readings (questions) per record share the rule.
* ``interval`` — interval membership: inside / below / above a window (closed, open, half-open, target ± tolerance), or which named zone
  holds the reading.
* ``select`` — multi-constraint selection among 4–20 candidates with 3–5 numeric and status attributes: exactly one candidate meets every
  constraint and each other candidate breaks one or two (often by a hair).
* ``compare`` — comparison / ordering: largest, smallest, closest to or farthest from a target, second largest / smallest, and the right
  ascending or descending order.
* ``convert`` — unit conversion inside rules: the rule in one unit and the reading in another (cm against m, s against min, g against
  kg, km/h against m/s, K against °C, a fraction against a percentage), as a step rule, a limit check or a comparison.

**Domains** (:data:`DOMAINS`): distance, speed, angle, time, mass, temperature, percent. **Number formats** (:class:`Style`): integers and
decimals (fixed decimals with trailing zeros or the shortest form), a leading ``+`` on positive signed readings or not, ASCII or Unicode
minus, a space between sign and digits or not, a space before the unit or not, thousands separators or not. **Values** are dense near every
threshold (within ±5 % or ±0.05 of it, on both sides) and spread over every band; a few sit exactly on a threshold where the rule's wording
decides it. **Wording**: option words (``plus``, ``minus``, a category or direction word) appear in the descriptions as distractors — a sign
convention named right next to readings of the other sign, irrelevant panel labels and log lines.

**Not used**: robo-jev-web's arm template (``buildServoQuestions``) and its sentences — the arm requests A16 / A16s stay a transfer test. A
check (:func:`arm_rule_problems`) refuses the arm's exact rule (thresholds 0.2 / 1 / 3 / 8 / 20 cm, or a category set holding three or more of
its names); the assembly scans every record's word shingles against A16 and A16s and reads only counts.

Labels are one-hot (``kind`` ``rule`` for rules, ``state`` for comparisons); choice questions have 2–26 options (``select`` 4–20); a record
with more than 8 options carries the 255-option opt-in; about a quarter of the questions show random pool codes (v0.8's rule,
:func:`robo_jev.simgen.format_v08.codes_selected`).
"""

from __future__ import annotations

import hashlib
import itertools
import math
import random
import re
from dataclasses import dataclass
from decimal import Decimal
from fractions import Fraction
from typing import Any, Callable, Sequence

__all__ = [
    "ARM_CATEGORY_NAMES",
    "ARM_THRESHOLDS_CM",
    "DOMAINS",
    "FAMILIES",
    "FAMILY_SHARES",
    "ID_PREFIX",
    "SET_NAME",
    "TEMPLATES",
    "UNITS",
    "VERSION",
    "Style",
    "arm_rule_problems",
    "generate",
    "make_record",
    "render",
    "schedule",
]

VERSION = "numeric-v09-1"
SET_NAME = "v09-numeric"
ID_PREFIX = "v09num-"
FAMILIES = ("step", "interval", "select", "compare", "convert")
DOMAINS = ("distance", "speed", "angle", "time", "mass", "temperature", "percent")
#: The family mix of the CN records.
FAMILY_SHARES = {"step": 0.30, "interval": 0.15, "select": 0.25, "compare": 0.15, "convert": 0.15}
#: The arm workcell's rule (A16 / A16s): never generated.
ARM_THRESHOLDS_CM = (Fraction(1, 5), Fraction(1), Fraction(3), Fraction(8), Fraction(20))
ARM_CATEGORY_NAMES = frozenset({"far", "mid", "near", "small", "fine", "stay"})
#: Exact-boundary readings (a value equal to a threshold, decided by the rule's wording) — a small share of drawn values.
BOUNDARY_SHARE = 0.03
#: v0.8's random-code share of choice questions (the same rule as the rest of the input).
CODES_SHARE = 0.25
#: The first index of a family's spare records (drawn when the assembly refuses a record that shares text with a sealed set).
SPARE_START = 900000


# --------------------------------------------------------------------------
# numbers, units and rendering
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Unit:
    """A display unit: its symbol, its domain and the exact factor (and offset) to the domain's base unit."""

    key: str
    domain: str
    symbol: str
    factor: Fraction
    offset: Fraction = Fraction(0)
    attach: bool = False  # the symbol follows the digits without a space ("12°")

    def to_base(self, value: Fraction) -> Fraction:
        return value * self.factor + self.offset

    def from_base(self, value: Fraction) -> Fraction:
        return (value - self.offset) / self.factor


UNITS: dict[str, Unit] = {u.key: u for u in (
    Unit("mm", "distance", "mm", Fraction(1, 1000)), Unit("cm", "distance", "cm", Fraction(1, 100)), Unit("m", "distance", "m", Fraction(1)),
    Unit("km", "distance", "km", Fraction(1000)),
    Unit("mm/s", "speed", "mm/s", Fraction(1, 1000)), Unit("cm/s", "speed", "cm/s", Fraction(1, 100)), Unit("m/s", "speed", "m/s", Fraction(1)),
    Unit("km/h", "speed", "km/h", Fraction(5, 18)),
    Unit("deg", "angle", "°", Fraction(1), attach=True), Unit("degrees", "angle", "degrees", Fraction(1)),
    Unit("ms", "time", "ms", Fraction(1, 1000)), Unit("s", "time", "s", Fraction(1)), Unit("min", "time", "min", Fraction(60)),
    Unit("h", "time", "h", Fraction(3600)),
    Unit("g", "mass", "g", Fraction(1, 1000)), Unit("kg", "mass", "kg", Fraction(1)), Unit("t", "mass", "t", Fraction(1000)),
    Unit("C", "temperature", "°C", Fraction(1)), Unit("K", "temperature", "K", Fraction(1), offset=Fraction(-27315, 100)),
    Unit("pct", "percent", "%", Fraction(1)), Unit("frac", "percent", "", Fraction(100)),
)}  # fmt: skip


@dataclass(frozen=True)
class Style:
    """How a record writes its numbers (module docstring)."""

    plus: bool
    minus: str
    sign_space: bool
    unit_space: bool
    thousands: bool
    fixed: bool

    def describe(self) -> dict[str, Any]:
        return {"plus": self.plus, "minus": "unicode" if self.minus == "−" else "ascii", "sign_space": self.sign_space,
                "unit_space": self.unit_space, "thousands": self.thousands, "fixed": self.fixed}


def draw_style(rnd: random.Random) -> Style:
    return Style(plus=rnd.random() < 0.5, minus="−" if rnd.random() < 0.3 else "-", sign_space=rnd.random() < 0.1,
                 unit_space=rnd.random() < 0.75, thousands=rnd.random() < 0.6, fixed=rnd.random() < 0.4)


def quantize(x: float | Fraction, decimals: int) -> Fraction:
    """`x` rounded to `decimals` places, exactly."""
    scale = 10 ** int(decimals)
    return Fraction(round(Fraction(x) * scale), scale)


def decimals_of(value: Fraction) -> int:
    """The fewest decimals that write `value` exactly (≤ 6)."""
    for d in range(7):
        if (value * 10**d).denominator == 1:
            return d
    raise ValueError(f"{value} needs more than 6 decimals")


def render(value: Fraction, unit: str, style: Style, decimals: int | None = None, *, signed: bool = False) -> str:
    """The text of a number with its unit: `decimals` places (None: the fewest that write it exactly; with ``style.fixed`` the given
    places are kept with trailing zeros), the style's sign, separators and unit spacing. ``signed``: a reading whose sign matters (a
    positive one may carry ``+``)."""
    value = Fraction(value)
    exact = decimals_of(value)
    places = exact if decimals is None else max(int(decimals), exact) if style.fixed else exact
    magnitude = abs(value)
    number = Decimal(magnitude.numerator) / Decimal(magnitude.denominator)
    digits = format(number, f"{',' if style.thousands else ''}.{places}f")
    sign = style.minus if value < 0 else ("+" if signed and value > 0 and style.plus else "")
    if sign and style.sign_space:
        sign += " "
    u = UNITS[unit]
    if not u.symbol:
        return sign + digits
    space = "" if u.attach else (" " if style.unit_space else "")
    return f"{sign}{digits}{space}{u.symbol}"


_NICE = (Fraction(1), Fraction(6, 5), Fraction(3, 2), Fraction(2), Fraction(5, 2), Fraction(3), Fraction(4), Fraction(5), Fraction(6),
         Fraction(15, 2), Fraction(8))


def nice(x: float, rnd: random.Random, *, rough: float = 0.35) -> Fraction:
    """A threshold near `x`: a 'nice' number (1, 1.2, 1.5, 2, 2.5, 3, 4, 5, 6, 7.5, 8 × 10^k) or, with probability `rough`, two significant
    digits (0.35, 1.7, 6.5, 14, 37 …)."""
    if x <= 0:
        raise ValueError(x)
    exponent = math.floor(math.log10(x))
    if rnd.random() < rough:
        return Fraction(round(x / 10 ** (exponent - 1))) * Fraction(10) ** (exponent - 1)
    mantissa = Fraction(x) / Fraction(10) ** exponent
    best = min(_NICE + (Fraction(10),), key=lambda m: abs(m - mantissa))
    return best * Fraction(10) ** exponent


def draw_thresholds(rnd: random.Random, k: int, low: float, high: float) -> list[Fraction]:
    """`k` strictly increasing thresholds between `low` and `high` (display unit), successive ratios ≥ 1.4, each a nice number."""
    for _ in range(200):
        ratios = [rnd.uniform(1.5, 4.0) for _ in range(k - 1)]
        span = math.prod(ratios)
        top = high / span
        if top <= low:
            continue
        first = math.exp(rnd.uniform(math.log(low), math.log(top)))
        values = [first]
        for r in ratios:
            values.append(values[-1] * r)
        out = [nice(v, rnd) for v in values]
        if all(b / a >= Fraction(7, 5) for a, b in zip(out, out[1:])) and all(decimals_of(t) <= 3 for t in out):
            return out
    raise RuntimeError("could not draw thresholds")


def near(rnd: random.Random, t: Fraction, *, decimals: int, side: int | None = None, allow_equal: bool = False) -> Fraction:
    """A value within ±5 % or ±0.05 of `t` (display unit), on `side` (+1 above, −1 below, None: either), with `decimals` places; never
    equal to `t` unless `allow_equal`."""
    side = side if side is not None else rnd.choice((-1, 1))
    for _ in range(50):
        if rnd.random() < 0.5:
            delta = float(t) * rnd.uniform(0.002, 0.05)
        else:
            delta = rnd.uniform(0.004, 0.05)
        value = quantize(float(t) + side * delta, decimals)
        if value == t and not allow_equal:
            value = t + side * Fraction(1, 10**decimals)
        if (value - t) * side > 0 or (allow_equal and value == t):
            return value
    return t + side * Fraction(1, 10**decimals)


def unit_of(rnd: random.Random, choices: Sequence[str]) -> str:
    return rnd.choice(list(choices))


def _seed(*parts: Any) -> random.Random:
    digest = hashlib.sha256("|".join(str(p) for p in (VERSION, *parts)).encode("utf-8")).digest()
    return random.Random(int.from_bytes(digest[:8], "big"))


def _cap(text: str) -> str:
    return text[:1].upper() + text[1:]


def _join(items: Sequence[str], conj: str = "and") -> str:
    items = list(items)
    if len(items) <= 1:
        return "".join(items)
    return ", ".join(items[:-1]) + f" {conj} " + items[-1]


# --------------------------------------------------------------------------
# distractor wording (option words in descriptions)
# --------------------------------------------------------------------------

_DISTRACTORS = (
    "The {w} button on the side panel is taped over and not in use.",
    "An old sticker on the housing still reads '{w}'.",
    "The previous shift logged a '{w}' adjustment on a different unit.",
    "The {w} lamp on the console only shows the backup mode.",
    "Someone has pencilled '{w}?' on the clipboard; it was never confirmed.",
    "The manual's index lists '{w}' under spare parts.",
    "A colleague jokes that this unit only knows '{w}'.",
)


def distractor_lines(rnd: random.Random, words: Sequence[str], *, rate: float = 0.6) -> list[str]:
    """0–2 sentences that name option words in an irrelevant way (module docstring)."""
    out: list[str] = []
    if not words or rnd.random() >= rate:
        return out
    for _ in range(rnd.choice((1, 1, 2))):
        out.append(rnd.choice(_DISTRACTORS).format(w=rnd.choice(list(words)).replace("_", " ")))
    return out


# --------------------------------------------------------------------------
# family: step (threshold / step)
# --------------------------------------------------------------------------

STEP_DOMAINS: dict[str, dict[str, Any]] = {
    "distance": {
        "units": ("mm", "cm", "m"), "range": {"mm": (0.5, 400.0), "cm": (0.1, 60.0), "m": (0.02, 6.0)},
        "scenes": [("A floor-cleaning robot follows a painted guide line", "offset from the line", ("front", "rear")),
                   ("A camera dolly has to stop at marked positions on its rail", "distance to the next mark", ("dolly_a", "dolly_b")),
                   ("A paint nozzle is lined up over a seam before each pass", "seam offset", ("nozzle_1", "nozzle_2", "nozzle_3")),
                   ("A label printer centres its head over each parcel", "head offset", ("lane_1", "lane_2")),
                   ("A docking cart creeps toward its charging contacts", "gap to the contacts", ("left_contact", "right_contact"))],
        "pairs": [("left", "right", "shift left", "shift right"), ("forward", "back", "move forward", "move back"),
                  ("raise", "lower", "raise it", "lower it"), ("plus", "minus", "move toward plus", "move toward minus"),
                  ("extend", "retract", "extend", "retract"), ("outward", "inward", "move outward", "move inward")],
    },
    "speed": {
        "units": ("m/s", "km/h", "cm/s"), "range": {"m/s": (0.02, 4.0), "km/h": (0.1, 15.0), "cm/s": (0.5, 300.0)},
        "scenes": [("A conveyor belt should run at its set speed", "speed difference to the setting", ("belt_1", "belt_2")),
                   ("A delivery robot keeps pace with the person it follows", "speed gap to the person", ("robot",)),
                   ("A winch reels cable at a commanded rate", "rate error", ("winch_a", "winch_b")),
                   ("Two shuttles on a test track should match a reference speed", "speed error", ("shuttle_1", "shuttle_2", "shuttle_3"))],
        "pairs": [("faster", "slower", "speed up", "slow down"), ("up", "down", "raise the speed", "lower the speed"),
                  ("plus", "minus", "add speed", "take speed off"), ("accelerate", "brake", "accelerate", "brake")],
    },
    "angle": {
        "units": ("deg", "degrees"), "range": {"deg": (0.1, 60.0), "degrees": (0.1, 60.0)},
        "scenes": [("A solar tracker turns its panels toward the sun", "pointing error", ("row_1", "row_2", "row_3")),
                   ("A security camera pans to face each door", "pan error", ("camera",)),
                   ("A boat autopilot holds a compass heading", "heading error", ("helm",)),
                   ("A tilting table has to stay level", "tilt", ("x_side", "y_side"))],
        "pairs": [("cw", "ccw", "turn clockwise", "turn counterclockwise"), ("left", "right", "turn left", "turn right"),
                  ("up", "down", "tilt up", "tilt down"), ("plus", "minus", "rotate the plus way", "rotate the minus way")],
    },
    "time": {
        "units": ("s", "ms", "min"), "range": {"s": (0.05, 300.0), "ms": (2.0, 4000.0), "min": (0.2, 60.0)},
        "scenes": [("A traffic-light controller keeps its cycle in step with the next junction", "offset to the next junction", ("main_road", "side_road")),
                   ("A dosing pump fires on a fixed schedule", "timing error", ("pump_1", "pump_2")),
                   ("A photo booth flash has to fire with the shutter", "flash delay error", ("flash",)),
                   ("A shuttle bus tries to keep to its timetable", "schedule deviation", ("bus_12", "bus_14"))],
        "pairs": [("advance", "delay", "fire earlier", "fire later"), ("earlier", "later", "shift earlier", "shift later"),
                  ("shorten", "lengthen", "shorten the wait", "lengthen the wait"), ("plus", "minus", "add time", "take time off")],
    },
    "mass": {
        "units": ("g", "kg"), "range": {"g": (0.2, 3000.0), "kg": (0.01, 40.0)},
        "scenes": [("A filling line puts a set amount of coffee into each bag", "weight difference to the target", ("bag",)),
                   ("A batching scale weighs out flour for each mix", "batch error", ("hopper_1", "hopper_2")),
                   ("A drone carries two payload pods that should match", "imbalance", ("pod_left", "pod_right")),
                   ("A bakery robot portions dough", "portion error", ("line_a", "line_b", "line_c"))],
        "pairs": [("add", "remove", "add material", "remove material"), ("plus", "minus", "add some", "take some away"),
                  ("top_up", "trim", "top up", "trim off"), ("more", "less", "put in more", "take out some")],
    },
    "temperature": {
        "units": ("C",), "range": {"C": (0.05, 40.0)},
        "scenes": [("A greenhouse heater holds each bed near its setpoint", "deviation from the setpoint", ("bed_a", "bed_b", "bed_c")),
                   ("An incubator corrects small temperature errors", "temperature error", ("upper_shelf", "lower_shelf")),
                   ("A heated print bed tracks its target temperature", "bed error", ("bed",)),
                   ("A pharmacy fridge keeps a set temperature", "drift from the setting", ("door_side", "back_wall"))],
        "pairs": [("heat", "cool", "heat", "cool"), ("warm", "chill", "warm up", "chill"), ("up", "down", "raise the setpoint", "lower the setpoint"),
                  ("plus", "minus", "add heat", "remove heat")],
    },
    "percent": {
        "units": ("pct",), "range": {"pct": (0.1, 60.0)},
        "scenes": [("A water tank controller keeps the fill level near its target", "level difference to the target", ("tank_1", "tank_2")),
                   ("A humidifier holds the room humidity near its setting", "humidity error", ("room",)),
                   ("A battery bank balances the charge of its packs", "charge imbalance", ("pack_a", "pack_b", "pack_c")),
                   ("A valve controller tracks a commanded opening", "opening error", ("valve_1", "valve_2"))],
        "pairs": [("fill", "drain", "fill", "drain"), ("raise", "lower", "raise it", "lower it"), ("open", "close", "open further", "close further"),
                  ("plus", "minus", "increase", "decrease")],
    },
}
CATEGORY_SETS: tuple[tuple[str, ...], ...] = (
    ("tiny", "small", "medium", "large", "huge"), ("fine", "medium", "coarse"), ("nudge", "hop", "jump", "leap"),
    ("slight", "moderate", "strong", "maximum"), ("trim", "small", "medium", "big", "full"), ("low", "mid", "high"),
    ("gentle", "normal", "hard", "max"), ("short", "medium", "long", "extra"), ("light", "medium", "heavy"),
    ("level1", "level2", "level3", "level4", "level5"), ("one", "two", "three", "four"), ("s", "m", "l", "xl"),
)
HOLD_NAMES = ("hold", "keep", "none", "leave", "no_change", "idle", "ok", "wait", "pass", "rest")
STEP_TEMPLATES = ("step-bullets", "step-prose", "step-table", "step-instructions", "step-checklist", "step-telemetry")


def _step_keys(rnd: random.Random, pos: str, neg: str, cats: Sequence[str]) -> tuple[dict[tuple[str, str], str], str]:
    fmt = rnd.choices(("dir_cat", "cat_dir", "dir_n"), weights=(0.6, 0.25, 0.15))[0]
    keys = {}
    for sign, word in (("+", pos), ("-", neg)):
        for i, cat in enumerate(cats, start=1):
            keys[(sign, cat)] = {"dir_cat": f"{word}_{cat}", "cat_dir": f"{cat}_{word}", "dir_n": f"{word}{i}"}[fmt]
    return keys, fmt


def step_bands(names: Sequence[str], texts: Sequence[str], inclusive: str) -> list[tuple[str, str]]:
    """(band name, its range in words) for a step rule: `names` = the bands from the smallest size up (the hold option first when the rule
    has one), `texts` = the thresholds between them as written (one fewer than `names`). ``lower``: a threshold value belongs to the band
    above it (``at least``); ``upper``: to the band below it (``more than``)."""
    if len(texts) != len(names) - 1:
        raise ValueError("one threshold between each two bands")
    out = []
    for i, name in enumerate(names):
        low = texts[i - 1] if i > 0 else None
        high = texts[i] if i < len(texts) else None
        if inclusive == "lower":
            words = f"below {high}" if low is None else (f"at {low} or more" if high is None else f"from {low} to under {high}")
        else:
            words = f"{high} or less" if low is None else (f"more than {low}" if high is None else f"more than {low} up to {high}")
        out.append((name, words))
    return out


def _step_band_lines(template: str, names: Sequence[str], texts: Sequence[str], inclusive: str, q: str) -> list[str]:
    """The size rule in the template's words (`names` from the smallest size up, `texts` the thresholds between them)."""
    bands = step_bands(names, texts, inclusive)
    if template in ("step-bullets", "step-prose", "step-instructions"):
        return ["; ".join(f"{name} {words}" for name, words in bands)]
    if template == "step-table":
        return [f"  {words}: {name}" for name, words in bands]
    if template == "step-checklist":
        word = "at least" if inclusive == "lower" else "more than"
        lines = []
        for n, i in enumerate(range(len(names) - 1, 0, -1), start=1):
            lead = "If" if n == 1 else "Otherwise, if"
            lines.append(f"{n}. {lead} the {q} is {word} {texts[i - 1]} in size, use {names[i]}.")
        lines.append(f"{len(names)}. Otherwise use {names[0]}.")
        return lines
    if template == "step-telemetry":
        lo, hi = ("<", "≥") if inclusive == "lower" else ("≤", ">")
        mid = "≤" if inclusive == "lower" else "<"
        parts = []
        for i, name in enumerate(names):
            if i == 0:
                parts.append(f"|e| {lo} {texts[0]} → {name}")
            elif i == len(names) - 1:
                parts.append(f"|e| {hi} {texts[-1]} → {name}")
            else:
                parts.append(f"{texts[i - 1]} {mid} |e| {lo} {texts[i]} → {name}")
        return ["; ".join(parts)]
    raise ValueError(template)


def step_label(value: Fraction, thresholds: Sequence[Fraction], *, inclusive: str, hold: bool) -> tuple[str, int | None]:
    """(sign — "+", "-", or "0" for the hold option —, category index) of a reading under a step rule: the thresholds its size passes
    (``>=`` for ``lower``, ``>`` for ``upper``) count the bands; with a dead band none passed means hold."""
    size = abs(value)
    passed = sum(1 for t in thresholds if (size >= t if inclusive == "lower" else size > t))
    if hold:
        if passed == 0:
            return "0", None
        return ("+" if value > 0 else "-"), passed - 1
    if value == 0:
        raise ValueError("a zero reading has no direction under a rule without a dead band")
    return ("+" if value > 0 else "-"), passed


def _step_value(rnd: random.Random, thresholds: Sequence[Fraction], hold: bool, decimals: int, top: Fraction) -> Fraction:
    """A signed reading: half near a threshold (either side), a few exactly on one (the wording decides), the rest anywhere in a band;
    never 0 without a dead band."""
    sign = rnd.choice((-1, 1))
    roll = rnd.random()
    if roll < 0.5:
        size = near(rnd, rnd.choice(list(thresholds)), decimals=decimals)
    elif roll < 0.5 + BOUNDARY_SHARE:
        size = rnd.choice(list(thresholds))
    else:
        bounds = [Fraction(0), *thresholds, max(top, thresholds[-1] * 2)]
        band = rnd.randrange(len(bounds) - 1)
        size = quantize(rnd.uniform(float(bounds[band]), float(bounds[band + 1])), decimals)
    if size <= 0:
        size = Fraction(0) if hold else Fraction(1, 10**decimals)
    return size * sign


def make_step(rnd: random.Random, index: int) -> dict[str, Any]:
    domain = rnd.choice(DOMAINS)
    spec = STEP_DOMAINS[domain]
    unit = unit_of(rnd, spec["units"])
    low, high = spec["range"][unit]
    scene, q, channels_all = rnd.choice(spec["scenes"])
    pos, neg, pos_phrase, neg_phrase = rnd.choice(spec["pairs"])
    if rnd.random() < 0.5:  # which sign maps to which direction is stated in the text
        pos, neg, pos_phrase, neg_phrase = neg, pos, neg_phrase, pos_phrase
    for _ in range(200):
        cat_set = rnd.choice(CATEGORY_SETS)
        k = rnd.randint(2, min(5, len(cat_set)))
        first = rnd.randint(0, len(cat_set) - k)
        cats = list(cat_set[first:first + k])
        hold_name = rnd.choice(HOLD_NAMES) if rnd.random() < 0.8 else None
        thresholds = draw_thresholds(rnd, k if hold_name else k - 1, low, high)
        names = [pos, neg, *cats] + ([hold_name] if hold_name else [])
        if len(set(names)) == len(names) and not arm_rule_problems(unit, thresholds, cats + ([hold_name] if hold_name else []), directions=(pos, neg)):
            break
    else:  # pragma: no cover
        raise RuntimeError("could not draw a step rule")
    inclusive = rnd.choice(("lower", "upper"))
    template = rnd.choice(STEP_TEMPLATES)
    style = draw_style(rnd)
    t_dec = max(decimals_of(t) for t in thresholds)
    keys, key_fmt = _step_keys(rnd, pos, neg, cats)
    if hold_name in set(keys.values()):
        hold_name = "hold_still"
    channels = rnd.sample(list(channels_all), rnd.randint(1, len(channels_all)))
    values = [_step_value(rnd, thresholds, bool(hold_name), min(3, t_dec + rnd.choice((0, 1, 1, 2))), Fraction(high) * Fraction(8, 5))
              for _ in channels]
    value_dec = max(decimals_of(v) for v in values)
    t_texts = [render(t, unit, style, t_dec) for t in thresholds]
    v_texts = [render(v, unit, style, value_dec, signed=True) for v in values]
    shown = (lambda i: f"size {i + 1}") if key_fmt == "dir_n" else (lambda i: cats[i])
    band_names = ([hold_name] if hold_name else []) + [shown(i) for i in range(k)]
    criteria: dict[str, str] = {}
    desc_mode = rnd.choices(("verb_cat", "cat_verb", "plain", "empty"), weights=(0.35, 0.3, 0.2, 0.15))[0]
    for sign, phrase in (("+", pos_phrase), ("-", neg_phrase)):
        for i, cat in enumerate(cats):
            criteria[keys[(sign, cat)]] = {"verb_cat": f"{_cap(phrase)} by a {shown(i)} step.", "cat_verb": f"{_cap(shown(i))} correction: {phrase}.",
                                           "plain": f"{_cap(phrase)} ({shown(i)}).", "empty": ""}[desc_mode]
    if hold_name:
        criteria[hold_name] = {"verb_cat": "Leave it as it is.", "cat_verb": "No correction this cycle.", "plain": "Do nothing.", "empty": ""}[desc_mode]
    order = list(criteria)
    rnd.shuffle(order)
    criteria = {key: criteria[key] for key in order}
    sign_line = rnd.choice((
        f"a positive {q} calls for {pos} ({pos_phrase}), a negative one for {neg} ({neg_phrase})",
        f"positive readings mean {pos}, negative readings mean {neg}",
        f"+ means {pos}, - means {neg}",
        f"{pos} when the {q} is above zero, {neg} when it is below zero",
    ))
    band_lines = _step_band_lines(template, band_names, t_texts, inclusive, q)
    noise = distractor_lines(rnd, [pos, neg, *cats] + ([hold_name] if hold_name else []))
    readings = [f"{ch.replace('_', ' ')}: {vt}" for ch, vt in zip(channels, v_texts)]
    u_symbol = UNITS[unit].symbol
    if template == "step-bullets":
        state = "\n".join([f"{scene}.", f"{_cap(q)} now ({u_symbol}): " + "; ".join(readings) + ".", "How each correction is chosen:",
                           f"- direction: {sign_line};", f"- size (the {q} without its sign): {band_lines[0]}.", *noise])
    elif template == "step-prose":
        state = " ".join([f"{scene}.", f"The controller sizes each correction from the {q} alone: {band_lines[0]}.",
                          f"The sign only sets the direction — {sign_line}.", *noise, "Latest values: " + ", ".join(readings) + "."])
    elif template == "step-table":
        state = "\n".join([f"{scene}.", f"Correction table for the {q} (size taken without its sign):", *band_lines,
                           f"Direction: {sign_line}.", *noise, "Measured now — " + " | ".join(readings)])
    elif template == "step-instructions":
        state = "\n".join([f"{scene}.", *(f"{_cap(ch.replace('_', ' '))} {q}: {vt}" for ch, vt in zip(channels, v_texts)), *noise])
    elif template == "step-checklist":
        state = "\n".join([f"{scene}.", f"Decide each correction by checking from the largest step down ({q}, size without sign):", *band_lines,
                           f"Direction follows the sign: {sign_line}.", *noise, "Readings: " + "; ".join(readings) + "."])
    else:  # telemetry
        short = re.sub(r"[^a-z]+", "_", q.split(" ")[0].lower())
        state = "\n".join([f"{scene}.", "[telemetry] " + " ".join(f"{ch}.{short}={vt}" for ch, vt in zip(channels, v_texts)),
                           f"policy, e = {q}: {band_lines[0]}", f"sign: + → {pos}, - → {neg}", *noise])
    questions, labels, qmeta = {}, {}, {}
    convention = f" ({pos} means a positive reading.)" if rnd.random() < 0.3 else ""
    for ch, value, vt in zip(channels, values, v_texts):
        name = f"q_{ch}"[:40]
        where = ch.replace("_", " ")
        if template == "step-instructions":
            instructions = f"Rule for the {q}: {band_lines[0]}; {sign_line}. Which correction does the {where} need?{convention}"
        else:
            instructions = rnd.choice((f"Which correction should be applied to the {where}?", f"Pick the step for the {where}.",
                                       f"What does the rule give for the {where}?", f"Choose the adjustment for the {where}.")) + convention
        sign, cat_index = step_label(value, thresholds, inclusive=inclusive, hold=bool(hold_name))
        answer = hold_name if sign == "0" else keys[(sign, cats[cat_index])]
        questions[name] = {"type": "choice", "instructions": instructions, "criteria": dict(criteria)}
        labels[name] = _one_hot(criteria, answer, kind="rule")
        qmeta[name] = {"kind": "step", "value": {"text": vt, "unit": unit}, "answer": answer}
    rule = {"unit": unit, "inclusive": inclusive, "thresholds": t_texts, "categories": list(cats), "hold": hold_name, "sign": {"+": pos, "-": neg},
            "keys": {f"{s}{c}": key for (s, c), key in keys.items()}, "key_format": key_fmt}
    return {"family": "step", "template": template, "domain": domain, "style": style, "state": state, "questions": questions, "labels": labels,
            "numeric": {"rule": rule, "questions": qmeta}}


# --------------------------------------------------------------------------
# family: interval membership
# --------------------------------------------------------------------------

INTERVAL_DOMAINS: dict[str, dict[str, Any]] = {
    "distance": {"units": ("mm", "cm"), "range": {"mm": (2.0, 600.0), "cm": (0.5, 80.0)},
                 "things": [("tread depth of each tyre", ("front_left", "front_right", "rear")), ("thickness of each parcel", ("parcel_1", "parcel_2")),
                            ("gap under each door", ("door_a", "door_b", "door_c"))]},
    "speed": {"units": ("m/s", "km/h"), "range": {"m/s": (0.1, 5.0), "km/h": (0.5, 20.0)},
              "things": [("speed of each cart in the aisle", ("cart_1", "cart_2")), ("belt speed", ("belt",)), ("walking pace of the tour robot", ("tour_robot",))]},
    "angle": {"units": ("deg",), "range": {"deg": (0.5, 80.0)}, "signed": True,
              "things": [("tilt of each panel", ("panel_1", "panel_2", "panel_3")), ("ramp angle", ("ramp",)), ("camera tilt", ("camera_a", "camera_b"))]},
    "time": {"units": ("s", "min"), "range": {"s": (1.0, 600.0), "min": (0.5, 90.0)},
             "things": [("curing time of each joint", ("joint_1", "joint_2")), ("response time of the door sensor", ("sensor",)),
                        ("charging break of each robot", ("robot_a", "robot_b", "robot_c"))]},
    "mass": {"units": ("g", "kg"), "range": {"g": (5.0, 5000.0), "kg": (0.1, 60.0)},
             "things": [("weight of each parcel for this chute", ("parcel_1", "parcel_2", "parcel_3")), ("dose in each cup", ("cup_a", "cup_b")),
                        ("load on the tray", ("tray",))]},
    "temperature": {"units": ("C",), "range": {"C": (-30.0, 90.0)}, "signed": True,
                    "things": [("temperature in each fridge", ("fridge_1", "fridge_2")), ("freezer temperature", ("freezer",)),
                               ("water temperature in each tank", ("tank_a", "tank_b", "tank_c"))]},
    "percent": {"units": ("pct",), "range": {"pct": (1.0, 99.0)},
                "things": [("charge of each battery", ("battery_1", "battery_2", "battery_3")), ("room humidity", ("room",)),
                           ("fill level of each silo", ("silo_a", "silo_b"))]},
}
WINDOW_NAMES = (("inside", "below", "above"), ("ok", "too_low", "too_high"), ("accept", "low", "high"), ("in_band", "under", "over"),
                ("pass", "fail_low", "fail_high"), ("within", "short", "excess"))
ZONE_NAMES = (("zone_a", "zone_b", "zone_c", "zone_d", "zone_e", "zone_f", "zone_g", "zone_h"),
              ("green", "yellow", "orange", "red", "purple", "grey", "white", "black"), ("band_1", "band_2", "band_3", "band_4", "band_5", "band_6", "band_7", "band_8"),
              ("calm", "mild", "brisk", "strong", "severe", "extreme", "storm", "gale"))
INTERVAL_TEMPLATES = ("interval-closed", "interval-open", "interval-halfopen", "interval-tolerance", "interval-zones")


def interval_label(value: Fraction, low: Fraction, high: Fraction, *, low_in: bool, high_in: bool) -> str:
    """"below" / "inside" / "above" of `value` against a window (each end included or not)."""
    if value < low or (value == low and not low_in):
        return "below"
    if value > high or (value == high and not high_in):
        return "above"
    return "inside"


def zone_label(value: Fraction, edges: Sequence[Fraction], *, lower_inclusive: bool) -> int:
    """The index of the zone (0 … len(edges)) of `value` with zones split at `edges` (ascending); an edge value goes up when
    `lower_inclusive` (each zone includes its lower limit), down otherwise."""
    count = 0
    for edge in edges:
        if value > edge or (value == edge and lower_inclusive):
            count += 1
    return count


def _window(rnd: random.Random, spec: dict[str, Any], unit: str, template: str) -> tuple[Fraction, Fraction, dict[str, Fraction]]:
    """(low, high, the written numbers) of an acceptance window in the domain's range (signed domains may straddle zero)."""
    low_r, high_r = spec["range"][unit]
    span = high_r - low_r
    if template == "interval-tolerance":
        centre = quantize(rnd.uniform(low_r + 0.2 * span, high_r - 0.2 * span), rnd.choice((0, 1)))
        if centre <= 0 and not spec.get("signed"):
            centre = Fraction(1)
        tol = nice(max(abs(float(centre)) * rnd.uniform(0.01, 0.15), 0.05), rnd)
        return centre - tol, centre + tol, {"centre": centre, "tolerance": tol}
    if spec.get("signed"):
        centre = quantize(rnd.uniform(low_r + 0.1 * span, high_r - 0.1 * span), rnd.choice((0, 1)))
        half = nice(rnd.uniform(0.02 * span, 0.15 * span), rnd)
        return centre - half, centre + half, {"low": centre - half, "high": centre + half}
    for _ in range(100):
        a = nice(rnd.uniform(low_r, low_r + 0.6 * span), rnd, rough=0.5)
        b = nice(float(a) * rnd.uniform(1.3, 3.5), rnd, rough=0.5)
        if a < b <= Fraction(high_r) * Fraction(6, 5) and (unit != "pct" or b <= 100):
            return a, b, {"low": a, "high": b}
    return Fraction(low_r).limit_denominator(10), Fraction(high_r).limit_denominator(10), {"low": Fraction(low_r).limit_denominator(10),
                                                                                           "high": Fraction(high_r).limit_denominator(10)}


def make_interval(rnd: random.Random, index: int) -> dict[str, Any]:
    domain = rnd.choice(DOMAINS)
    spec = INTERVAL_DOMAINS[domain]
    unit = unit_of(rnd, spec["units"])
    low_r, high_r = spec["range"][unit]
    thing, items_all = rnd.choice(spec["things"])
    template = rnd.choice(INTERVAL_TEMPLATES)
    style = draw_style(rnd)
    signed = bool(spec.get("signed"))
    items = rnd.sample(list(items_all), rnd.randint(1, len(items_all)))
    questions, labels, qmeta = {}, {}, {}
    if template != "interval-zones":
        names = rnd.choice(WINDOW_NAMES)
        inside, below, above = names
        low, high, written = _window(rnd, spec, unit, template)
        low_in, high_in = {"interval-closed": (True, True), "interval-open": (False, False), "interval-halfopen": (True, False),
                           "interval-tolerance": (True, True)}[template]
        dec = max(decimals_of(x) for x in written.values())
        texts = {key: render(x, unit, style, dec if key in ("low", "high") else None, signed=signed and x < 0) for key, x in written.items()}
        included = [edge for edge, flag in ((low, low_in), (high, high_in)) if flag]
        values = []
        for _ in items:
            vd = min(3, dec + rnd.choice((0, 1, 1, 2)))
            roll = rnd.random()
            if roll < 0.5:
                edge, side = rnd.choice(((low, -1), (low, 1), (high, -1), (high, 1)))
                v = near(rnd, edge, decimals=vd, side=side)
            elif roll < 0.5 + 2 * BOUNDARY_SHARE and included:  # exactly on a limit only where the limit is included
                v = rnd.choice(included)
            else:
                width = float(high - low)
                lo, hi = float(low) - 0.6 * width, float(high) + 0.6 * width
                if not signed:
                    lo = max(lo, float(low) * 0.3)
                if unit == "pct":
                    hi = min(hi, 100.0)
                v = quantize(rnd.uniform(lo, hi), vd)
                if (v == low and not low_in) or (v == high and not high_in):  # never exactly on an excluded limit
                    v += Fraction(1, 10**vd)
            if not signed and v <= 0:
                v = quantize(float(low) * 0.5, vd) or Fraction(1, 10**vd)
            values.append(v)
        vdec = max(decimals_of(v) for v in values)
        v_texts = [render(v, unit, style, vdec, signed=signed and v != 0) for v in values]
        if template == "interval-closed":
            rule = rnd.choice((f"The {thing} is acceptable from {texts['low']} to {texts['high']}, both limits included.",
                               f"Allowed: from {texts['low']} up to and including {texts['high']}; a value on either limit counts as {inside}."))
        elif template == "interval-open":
            rule = rnd.choice((f"The {thing} must be more than {texts['low']} and less than {texts['high']}.",
                               f"Window for the {thing}: strictly between {texts['low']} and {texts['high']} (limits excluded)."))
        elif template == "interval-halfopen":
            rule = rnd.choice((f"The {thing} must be at least {texts['low']} and under {texts['high']}.",
                               f"Accepted range for the {thing}: from {texts['low']} (included) up to {texts['high']} (not included)."))
        else:
            rule = rnd.choice((f"Target for the {thing}: {texts['centre']} with a tolerance of ± {texts['tolerance']} (limits included).",
                               f"The {thing} should be {texts['centre']}, give or take {texts['tolerance']}; a reading right on the limit still passes."))
        lines = [f"Check of the {thing}.", rule, *(f"{it.replace('_', ' ')}: {vt}" for it, vt in zip(items, v_texts))]
        option_desc = {inside: rnd.choice(("Within the allowed range.", "Inside the window.", "")),
                       below: rnd.choice(("Too low for the range.", "Under the range.", "")),
                       above: rnd.choice(("Too high for the range.", "Over the range.", ""))}
        order = list(option_desc)
        rnd.shuffle(order)
        criteria = {key: option_desc[key] for key in order}
        for it, v, vt in zip(items, values, v_texts):
            name = f"q_{it}"[:40]
            answer = {"inside": inside, "below": below, "above": above}[interval_label(v, low, high, low_in=low_in, high_in=high_in)]
            questions[name] = {"type": "choice", "instructions": rnd.choice((f"Where does the {it.replace('_', ' ')} reading fall?",
                                                                             f"Is the {it.replace('_', ' ')} inside the range, below it or above it?",
                                                                             f"Classify the {it.replace('_', ' ')}.")), "criteria": dict(criteria)}
            labels[name] = _one_hot(criteria, answer, kind="rule")
            qmeta[name] = {"kind": "window", "value": {"text": vt, "unit": unit}, "answer": answer}
        window = ({"centre": texts["centre"], "tolerance": texts["tolerance"]} if template == "interval-tolerance"
                  else {"low": texts["low"], "high": texts["high"]})
        numeric_rule = {"unit": unit, "window": {**window, "low_in": low_in, "high_in": high_in},
                        "names": {"inside": inside, "below": below, "above": above}}
        noise_words = list(names)
    else:
        zone_count = rnd.randint(3, 8)
        names = list(rnd.choice(ZONE_NAMES)[:zone_count])
        span = high_r - low_r
        if signed:
            start = quantize(rnd.uniform(low_r, low_r + 0.4 * span), 0)
        else:
            start = nice(rnd.uniform(max(low_r, 0.05), low_r + 0.3 * span), rnd)
        edges = [start]
        for _ in range(zone_count - 2):
            edges.append(edges[-1] + nice(max(span * rnd.uniform(0.03, 0.15), 0.1), rnd))
        dec = max(decimals_of(e) for e in edges)
        lower_inclusive = rnd.random() < 0.5
        e_texts = [render(e, unit, style, dec, signed=signed and e < 0) for e in edges]
        values = []
        for _ in items:
            vd = min(3, dec + rnd.choice((0, 1, 1)))
            roll = rnd.random()
            if roll < 0.55:
                v = near(rnd, rnd.choice(edges), decimals=vd)
            elif roll < 0.61:
                v = rnd.choice(edges)
            else:
                width = float(edges[-1] - edges[0]) or 1.0
                lo = float(edges[0]) - 0.4 * width
                v = quantize(rnd.uniform(max(lo, float(edges[0]) * 0.2) if not signed else lo, float(edges[-1]) + 0.4 * width), vd)
            if not signed and v <= 0:
                v = quantize(float(edges[0]) * 0.5, vd) or Fraction(1, 10**vd)
            values.append(v)
        vdec = max(decimals_of(v) for v in values)
        v_texts = [render(v, unit, style, vdec, signed=signed and v != 0) for v in values]
        conv = "each zone includes its lower limit" if lower_inclusive else "each zone includes its upper limit"
        zone_lines = []
        for z in range(zone_count):
            if z == 0:
                zone_lines.append(f"{names[0]}: {'below' if lower_inclusive else 'up to'} {e_texts[0]}")
            elif z == zone_count - 1:
                zone_lines.append(f"{names[z]}: {'from' if lower_inclusive else 'above'} {e_texts[-1]}")
            else:
                zone_lines.append(f"{names[z]}: {e_texts[z - 1]} to {e_texts[z]}")
        lines = [f"Zones for the {thing} ({conv}):", *zone_lines, *(f"{it.replace('_', ' ')} reads {vt}" for it, vt in zip(items, v_texts))]
        criteria = {n: rnd.choice((f"The {n.replace('_', ' ')} zone.", "")) for n in names}
        order = list(criteria)
        rnd.shuffle(order)
        criteria = {key: criteria[key] for key in order}
        for it, v, vt in zip(items, values, v_texts):
            name = f"q_{it}"[:40]
            answer = names[zone_label(v, edges, lower_inclusive=lower_inclusive)]
            questions[name] = {"type": "choice", "instructions": rnd.choice((f"Which zone is the {it.replace('_', ' ')} in?",
                                                                             f"Name the zone of the {it.replace('_', ' ')} reading.")),
                               "criteria": dict(criteria)}
            labels[name] = _one_hot(criteria, answer, kind="rule")
            qmeta[name] = {"kind": "zones", "value": {"text": vt, "unit": unit}, "answer": answer}
        numeric_rule = {"unit": unit, "zones": names, "edges": e_texts, "lower_inclusive": lower_inclusive}
        noise_words = names
    lines += distractor_lines(rnd, noise_words, rate=0.4)
    return {"family": "interval", "template": template, "domain": domain, "style": style, "state": "\n".join(lines), "questions": questions,
            "labels": labels, "numeric": {"rule": numeric_rule, "questions": qmeta}}


# --------------------------------------------------------------------------
# family: multi-constraint selection
# --------------------------------------------------------------------------

#: Scenarios: the item word, the attributes (name, unit, low, high, places, the allowed constraint operators) and the status values
#: (the first is the one the task needs). Bins, boxes, widths and heights are not used here.
SELECT_SCENARIOS: tuple[dict[str, Any], ...] = (
    {"what": "battery pack for the night shift", "item": "pack", "attrs": [("charge", "pct", 5, 100, 0, ("ge", "gt")), ("cell temperature", "C", 12, 55, 1, ("le", "lt")),
     ("mass", "kg", 0.8, 6.0, 1, ("le", "lt")), ("age", "min", 30, 900, 0, ("le",))], "status": ("ready", "charging", "locked", "faulty")},
    {"what": "route for the delivery robot", "item": "route", "attrs": [("length", "m", 80, 1500, 0, ("le", "lt")), ("steepest slope", "deg", 0.5, 12, 1, ("le", "lt")),
     ("expected time", "min", 2, 40, 1, ("le", "lt")), ("crowd level", "pct", 0, 100, 0, ("le", "lt"))], "status": ("dry", "wet", "icy", "closed")},
    {"what": "drone for the parcel", "item": "drone", "attrs": [("payload limit", "kg", 0.5, 8.0, 1, ("ge", "gt")), ("flight time left", "min", 3, 45, 0, ("ge", "gt")),
     ("wind limit", "km/h", 10, 60, 0, ("ge", "gt")), ("battery", "pct", 5, 100, 0, ("ge",))], "status": ("ready", "grounded", "updating")},
    {"what": "conveyor lane for the next tote", "item": "lane", "attrs": [("belt speed", "m/s", 0.2, 3.0, 2, ("ge", "le")), ("occupancy", "pct", 0, 100, 0, ("le", "lt")),
     ("queue wait", "s", 0, 240, 0, ("le", "lt"))], "status": ("open", "jammed", "paused")},
    {"what": "incubator slot for the samples", "item": "slot", "attrs": [("temperature", "C", 20, 45, 1, ("between",)), ("humidity", "pct", 30, 99, 0, ("ge", "between")),
     ("free time", "h", 1, 72, 0, ("ge", "gt"))], "status": ("closed", "open", "cleaning")},
    {"what": "charging station for the cart", "item": "station", "attrs": [("distance", "m", 5, 400, 0, ("le", "lt")), ("queue", "min", 0, 60, 0, ("le", "lt")),
     ("charge rate", "pct", 5, 60, 0, ("ge", "gt"))], "status": ("free", "busy", "out_of_service")},
    {"what": "camera mount for the inspection", "item": "mount", "attrs": [("pan range", "deg", 10, 340, 0, ("ge",)), ("distance to the line", "m", 0.5, 12, 1, ("le", "between")),
     ("coverage", "pct", 20, 100, 0, ("ge", "gt"))], "status": ("online", "offline", "booting")},
    {"what": "delivery cart for the heavy order", "item": "cart", "attrs": [("top speed", "km/h", 3, 25, 0, ("ge",)), ("load limit", "kg", 20, 400, 0, ("ge", "gt")),
     ("battery", "pct", 5, 100, 0, ("ge", "gt")), ("distance away", "m", 2, 300, 0, ("le",))], "status": ("idle", "on_route", "maintenance")},
    {"what": "oven for the next tray", "item": "oven", "attrs": [("temperature", "C", 140, 260, 0, ("between", "ge")), ("time until free", "min", 0, 50, 0, ("le", "lt")),
     ("load", "pct", 0, 100, 0, ("le", "lt"))], "status": ("on", "off", "cleaning")},
    {"what": "time slot for the maintenance job", "item": "slot", "attrs": [("start in", "min", 5, 600, 0, ("le", "ge")), ("length", "min", 10, 240, 0, ("ge",)),
     ("expected load", "pct", 0, 100, 0, ("le", "lt"))], "status": ("open", "reserved", "blocked")},
    {"what": "lift for the cart", "item": "lift", "attrs": [("wait", "s", 0, 300, 0, ("le", "lt")), ("free load", "kg", 50, 1500, 0, ("ge", "gt")),
     ("car speed", "m/s", 0.3, 2.5, 1, ("ge",))], "status": ("running", "stopped", "inspection")},
    {"what": "ramp for the wheelchair robot", "item": "ramp", "attrs": [("slope", "deg", 1, 15, 1, ("le", "lt")), ("length", "m", 2, 30, 1, ("le",)),
     ("surface grip", "pct", 30, 100, 0, ("ge", "gt"))], "status": ("clear", "blocked", "wet")},
)
SELECT_TEMPLATES = ("select-options", "select-list", "select-instructions", "select-table")
_OP_WORDS = {"ge": "at least", "gt": "more than", "le": "at most", "lt": "less than"}


def _satisfies(value: Fraction, op: str, a: Fraction, b: Fraction | None = None) -> bool:
    if op == "ge":
        return value >= a
    if op == "gt":
        return value > a
    if op == "le":
        return value <= a
    if op == "lt":
        return value < a
    if op == "between":
        return a <= value <= b
    raise ValueError(op)


def make_select(rnd: random.Random, index: int) -> dict[str, Any]:
    scenario = rnd.choice(SELECT_SCENARIOS)
    template = rnd.choice(SELECT_TEMPLATES)
    style = draw_style(rnd)
    n = rnd.randint(4, 20)
    attrs = list(scenario["attrs"])
    m = rnd.randint(2, len(attrs))
    used = rnd.sample(attrs, m)
    use_status = rnd.random() < 0.75
    constraints = []
    for name, unit, low, high, places, ops in used:
        op = rnd.choice(ops)
        if op == "between":
            a = quantize(rnd.uniform(low, low + (high - low) * 0.5), places)
            b = quantize(float(a) + rnd.uniform(0.15, 0.4) * (high - low), places)
            if b <= a:
                b = a + Fraction(1, 10**places)
            constraints.append({"attr": name, "unit": unit, "op": op, "a": a, "b": b, "places": places, "low": low, "high": high})
        else:
            a = quantize(rnd.uniform(low + (high - low) * 0.2, low + (high - low) * 0.8), places)
            constraints.append({"attr": name, "unit": unit, "op": op, "a": a, "b": None, "places": places, "low": low, "high": high})
    status_needed = scenario["status"][0]

    def draw(c: dict[str, Any], ok: bool) -> Fraction:
        places = c["places"]
        step = Fraction(1, 10**places)
        lo, hi = Fraction(c["low"]), Fraction(c["high"])
        for _ in range(200):
            if rnd.random() < 0.5:  # near a limit
                edge = c["a"] if c["b"] is None or rnd.random() < 0.5 else c["b"]
                v = near(rnd, edge, decimals=places, allow_equal=rnd.random() < 0.1)
            else:
                v = quantize(rnd.uniform(float(lo), float(hi)), places)
            if v < 0 or (c["unit"] == "pct" and v > 100):
                continue
            if _satisfies(v, c["op"], c["a"], c["b"]) == ok:
                return v
        # fall back: one step past (or inside) the limit
        a = c["a"]
        if c["op"] in ("ge", "gt"):
            return (a + step if c["op"] == "gt" else a) if ok else a - step
        if c["op"] in ("le", "lt"):
            return (a - step if c["op"] == "lt" else a) if ok else a + step
        return a if ok else a - step

    keys = [f"{scenario['item']}_{i}" for i in rnd.sample(range(1, 100), n)] if rnd.random() < 0.7 else \
        [f"{scenario['item']}_{c}" for c in rnd.sample([chr(97 + i) for i in range(26)], n)]
    winner = rnd.randrange(n)
    candidates = []
    for i in range(n):
        if i == winner:
            broken: set[int] = set()
            status_broken = False
        else:
            slots = list(range(len(constraints))) + (["status"] if use_status else [])
            count = 1 if rnd.random() < 0.6 or len(slots) < 2 else 2
            picked = rnd.sample(slots, count)
            broken = {p for p in picked if p != "status"}
            status_broken = "status" in picked
        values = {}
        for j, c in enumerate(constraints):
            values[c["attr"]] = draw(c, j not in broken)
        for name, unit, low, high, places, _ in attrs:  # attributes without a constraint: any value
            if name not in values and rnd.random() < 0.6:
                values[name] = quantize(rnd.uniform(low, high), places)
        status = rnd.choice(scenario["status"][1:]) if status_broken else (status_needed if use_status else rnd.choice(scenario["status"]))
        candidates.append({"key": keys[i], "values": values, "status": status})
    # exactly one candidate meets every constraint
    def passes(cand: dict[str, Any]) -> bool:
        if use_status and cand["status"] != status_needed:
            return False
        return all(_satisfies(cand["values"][c["attr"]], c["op"], c["a"], c["b"]) for c in constraints)

    assert sum(passes(c) for c in candidates) == 1 and passes(candidates[winner])
    units = {name: unit for name, unit, *_ in attrs}
    places_of = {name: places for name, _, _, _, places, _ in attrs}

    def text_of(name: str, value: Fraction) -> str:
        return render(value, units[name], style, places_of[name])

    c_texts = []
    for c in constraints:
        if c["op"] == "between":
            c_texts.append({"attr": c["attr"], "op": "between", "a": text_of(c["attr"], c["a"]), "b": text_of(c["attr"], c["b"])})
        else:
            c_texts.append({"attr": c["attr"], "op": c["op"], "a": text_of(c["attr"], c["a"])})
    needs = []
    for c in c_texts:
        if c["op"] == "between":
            needs.append(f"a {c['attr']} between {c['a']} and {c['b']} (inclusive)")
        else:
            needs.append(f"a {c['attr']} of {_OP_WORDS[c['op']]} {c['a']}")
    if use_status:
        needs.append(f"status {status_needed.replace('_', ' ')}")
    need_sentence = f"The {scenario['what']} needs " + _join(needs) + "."
    attr_order = [a[0] for a in attrs]

    def describe(cand: dict[str, Any]) -> str:
        parts = [f"{name} {text_of(name, cand['values'][name])}" for name in attr_order if name in cand["values"]]
        parts.append(f"status {cand['status'].replace('_', ' ')}")
        return ", ".join(parts)

    rows = [(cand["key"], describe(cand)) for cand in candidates]
    if template == "select-options":
        state = need_sentence
        criteria = {k: d for k, d in rows}
        instructions = rnd.choice((f"Which {scenario['item']} meets every requirement?", f"Pick the {scenario['item']} to use.",
                                   f"Choose the only {scenario['item']} that fits all the requirements."))
    elif template == "select-list":
        state = "\n".join([need_sentence, f"Candidates ({n}):", *(f"- {k}: {d}" for k, d in rows)])
        criteria = {k: rnd.choice((f"Use {k.replace('_', ' ')}.", "")) for k, _ in rows}
        instructions = rnd.choice((f"Which {scenario['item']} should be used?", f"Pick the {scenario['item']} that meets all requirements."))
    elif template == "select-instructions":
        state = "\n".join([f"Candidates for the {scenario['what']}:", *(f"{k}: {d}" for k, d in rows)])
        criteria = {k: "" for k, _ in rows}
        instructions = f"{need_sentence} Which {scenario['item']} fits?"
    else:  # table
        header = " | ".join(["name", *attr_order, "status"])
        lines = [header]
        for cand in candidates:
            cells = [cand["key"]] + [text_of(name, cand["values"][name]) if name in cand["values"] else "n/a" for name in attr_order]
            lines.append(" | ".join(cells + [cand["status"].replace("_", " ")]))
        state = "\n".join([need_sentence, *lines])
        criteria = {k: "" for k, _ in rows}
        instructions = rnd.choice((f"Which {scenario['item']} meets all the requirements?", "Which row fits?"))
    noise = distractor_lines(rnd, [k for k, _ in rows], rate=0.25)
    if noise:
        state = state + "\n" + "\n".join(noise)
    answer = candidates[winner]["key"]
    name = "q_choose"
    meta_candidates = {cand["key"]: {"values": {a: text_of(a, v) for a, v in cand["values"].items()}, "status": cand["status"]} for cand in candidates}
    return {"family": "select", "template": template, "domain": UNITS[units[constraints[0]["attr"]]].domain, "style": style, "state": state,
            "questions": {name: {"type": "choice", "instructions": instructions, "criteria": criteria}},
            "labels": {name: _one_hot(criteria, answer, kind="rule")},
            "numeric": {"rule": {"constraints": c_texts, "status": status_needed if use_status else None, "units": {a: units[a] for a in units}},
                        "questions": {name: {"kind": "select", "candidates": meta_candidates, "answer": answer}}}}


# --------------------------------------------------------------------------
# family: comparison / ordering
# --------------------------------------------------------------------------

COMPARE_DOMAINS: dict[str, dict[str, Any]] = {
    "distance": {"units": ("mm", "cm", "m"), "range": {"mm": (-80, 900), "cm": (-30, 250), "m": (-5, 40)}, "signed": True,
                 "items": ("sensor_a", "sensor_b", "sensor_c", "sensor_d", "sensor_e", "sensor_f", "sensor_g", "sensor_h"), "what": "measured gap"},
    "speed": {"units": ("m/s", "km/h"), "range": {"m/s": (0.0, 6.0), "km/h": (0.0, 30.0)}, "signed": False,
              "items": ("cart_1", "cart_2", "cart_3", "cart_4", "cart_5", "cart_6", "cart_7", "cart_8"), "what": "speed"},
    "angle": {"units": ("deg",), "range": {"deg": (-45, 45)}, "signed": True,
              "items": ("panel_n", "panel_e", "panel_s", "panel_w", "panel_ne", "panel_nw", "panel_se", "panel_sw"), "what": "tilt"},
    "time": {"units": ("s", "min"), "range": {"s": (0.5, 400), "min": (0.5, 90)}, "signed": False,
             "items": ("task_a", "task_b", "task_c", "task_d", "task_e", "task_f", "task_g", "task_h"), "what": "duration"},
    "mass": {"units": ("g", "kg"), "range": {"g": (1, 4000), "kg": (0.1, 80)}, "signed": False,
             "items": ("parcel_1", "parcel_2", "parcel_3", "parcel_4", "parcel_5", "parcel_6", "parcel_7", "parcel_8"), "what": "weight"},
    "temperature": {"units": ("C",), "range": {"C": (-25, 60)}, "signed": True,
                    "items": ("probe_1", "probe_2", "probe_3", "probe_4", "probe_5", "probe_6", "probe_7", "probe_8"), "what": "temperature"},
    "percent": {"units": ("pct",), "range": {"pct": (0, 100)}, "signed": False,
                "items": ("tank_a", "tank_b", "tank_c", "tank_d", "tank_e", "tank_f", "tank_g", "tank_h"), "what": "fill level"},
}
COMPARE_MODES = ("largest", "smallest", "closest", "farthest", "second_largest", "second_smallest", "order_asc", "order_desc")
COMPARE_TEMPLATES = ("compare-list", "compare-inline", "compare-options")


def compare_answer(mode: str, values: dict[str, Fraction], target: Fraction | None = None) -> Any:
    """The item (or the ordering, a tuple of items) a comparison asks for."""
    items = list(values)
    if mode == "largest":
        return max(items, key=lambda k: values[k])
    if mode == "smallest":
        return min(items, key=lambda k: values[k])
    if mode == "closest":
        return min(items, key=lambda k: abs(values[k] - target))
    if mode == "farthest":
        return max(items, key=lambda k: abs(values[k] - target))
    if mode == "second_largest":
        return sorted(items, key=lambda k: values[k])[-2]
    if mode == "second_smallest":
        return sorted(items, key=lambda k: values[k])[1]
    if mode == "order_asc":
        return tuple(sorted(items, key=lambda k: values[k]))
    if mode == "order_desc":
        return tuple(sorted(items, key=lambda k: values[k], reverse=True))
    raise ValueError(mode)


def _distinct_values(rnd: random.Random, count: int, low: float, high: float, places: int, *, close: bool) -> list[Fraction]:
    """`count` distinct values; with `close`, two of them differ by a hair (one or a few last-place units) to make the comparison count."""
    out: list[Fraction] = []
    while len(out) < count:
        v = quantize(rnd.uniform(low, high), places)
        if v not in out:
            out.append(v)
    if close and count >= 2:
        i = rnd.randrange(count)
        gap = Fraction(rnd.randint(1, 4), 10**places)
        candidate = out[i] + gap * rnd.choice((-1, 1))
        if candidate not in out and low <= candidate <= high:
            out[(i + 1) % count] = candidate
    return out


def make_compare(rnd: random.Random, index: int) -> dict[str, Any]:
    domain = rnd.choice(DOMAINS)
    spec = COMPARE_DOMAINS[domain]
    unit = unit_of(rnd, spec["units"])
    low, high = spec["range"][unit]
    template = rnd.choice(COMPARE_TEMPLATES)
    style = draw_style(rnd)
    count = rnd.randint(3, 8)
    modes = rnd.sample(COMPARE_MODES, rnd.choice((1, 1, 2)))
    if any(m.startswith("order") for m in modes):
        count = min(count, rnd.choice((3, 4)))
        if template == "compare-options":  # an ordering question's options show orders, not readings: the readings go in the state
            template = rnd.choice(("compare-list", "compare-inline"))
    names = rnd.sample(list(spec["items"]), count)
    for _ in range(100):
        places_list = [rnd.choice((0, 1, 1, 2)) for _ in range(count)]
        places = max(places_list)
        values_list = _distinct_values(rnd, count, low, high, places, close=rnd.random() < 0.6)
        # mixed decimal places: some values written with fewer places (2.5 vs 2.45)
        values_list = [quantize(v, p) if rnd.random() < 0.4 else v for v, p in zip(values_list, places_list)]
        if len(set(values_list)) < count:
            continue
        values = dict(zip(names, values_list))
        target = None
        if any(m in ("closest", "farthest") for m in modes):
            target = quantize(rnd.uniform(low, high), rnd.choice((0, 1)))
            dists = sorted(abs(v - target) for v in values_list)
            if dists[0] == dists[1] or dists[-1] == dists[-2]:
                continue
        break
    else:  # pragma: no cover
        raise RuntimeError("could not draw comparison values")
    signed = spec["signed"]
    texts = {k: render(v, unit, style, None, signed=signed and v != 0) for k, v in values.items()}
    target_text = render(target, unit, style, None, signed=signed and target is not None and target != 0) if target is not None else None
    what = spec["what"]
    if template == "compare-list":
        state = "\n".join([f"Latest {what} readings:", *(f"- {k.replace('_', ' ')}: {texts[k]}" for k in names)])
    elif template == "compare-inline":
        state = f"The {what} of each one: " + "; ".join(f"{k.replace('_', ' ')} {texts[k]}" for k in names) + "."
    else:
        state = f"Each option shows a {what} reading."
    if target_text:
        state += f"\nReference value: {target_text}."
    state = "\n".join([state, *distractor_lines(rnd, ["largest", "smallest", "first", "last"], rate=0.2)])
    questions, labels, qmeta = {}, {}, {}
    for mode in modes:
        answer = compare_answer(mode, values, target)
        if mode.startswith("order"):
            perms = list(itertools.permutations(names))
            rnd.shuffle(perms)
            wanted = tuple(answer)
            chosen = [wanted] + [p for p in perms if p != wanted][: rnd.randint(3, min(5, len(perms) - 1))]
            rnd.shuffle(chosen)
            sep = " < " if mode == "order_asc" else " > "
            criteria = {f"order_{i + 1}": sep.join(x.replace("_", " ") for x in p) for i, p in enumerate(chosen)}
            answer_key = f"order_{chosen.index(wanted) + 1}"
            instructions = ("Which line puts the readings from lowest to highest?" if mode == "order_asc"
                            else "Which line puts the readings from highest to lowest?")
            orders = {f"order_{i + 1}": list(p) for i, p in enumerate(chosen)}
        else:
            if template == "compare-options":
                criteria = {k: f"{what} {texts[k]}" for k in names}
            else:
                criteria = {k: rnd.choice((f"{k.replace('_', ' ')}", "")) for k in names}
            answer_key = answer
            word = {"largest": f"the highest {what}", "smallest": f"the lowest {what}", "closest": f"the {what} closest to the reference value",
                    "farthest": f"the {what} farthest from the reference value", "second_largest": f"the second-highest {what}",
                    "second_smallest": f"the second-lowest {what}"}[mode]
            instructions = rnd.choice((f"Which one has {word}?", f"Pick the one with {word}."))
            orders = None
        name = f"q_{mode}"
        questions[name] = {"type": "choice", "instructions": instructions, "criteria": criteria}
        labels[name] = _one_hot(criteria, answer_key, kind="state")
        qmeta[name] = {"kind": "compare", "mode": mode, "items": dict(texts), "unit": unit, "target": target_text, "orders": orders,
                       "answer": answer_key}
    return {"family": "compare", "template": template, "domain": domain, "style": style, "state": state, "questions": questions, "labels": labels,
            "numeric": {"rule": {"unit": unit}, "questions": qmeta}}


# --------------------------------------------------------------------------
# family: unit conversion inside rules
# --------------------------------------------------------------------------

#: (domain, rule unit, reading units, a rule-unit range for the limits)
CONVERT_PAIRS = (
    ("distance", "m", ("cm", "mm"), (0.05, 5.0)), ("distance", "cm", ("mm", "m"), (0.5, 300.0)), ("distance", "km", ("m",), (0.2, 15.0)),
    ("time", "min", ("s",), (0.5, 30.0)), ("time", "s", ("ms", "min"), (0.2, 240.0)), ("time", "h", ("min",), (0.25, 10.0)),
    ("mass", "kg", ("g",), (0.05, 25.0)), ("mass", "g", ("kg",), (20.0, 5000.0)), ("mass", "t", ("kg",), (0.2, 8.0)),
    ("speed", "km/h", ("m/s",), (1.0, 40.0)), ("speed", "m/s", ("km/h", "cm/s"), (0.1, 8.0)),
    ("temperature", "K", ("C",), (250.0, 360.0)), ("temperature", "C", ("K",), (-20.0, 80.0)),
    ("percent", "pct", ("frac",), (5.0, 95.0)), ("percent", "frac", ("pct",), (0.05, 0.95)),
)
CONVERT_TEMPLATES = ("convert-step", "convert-limit", "convert-compare", "convert-window")
CONVERT_STEP_NAMES = (("stop", "slow", "go"), ("wait", "prepare", "start"), ("red", "amber", "green"), ("reject", "check", "accept"),
                      ("low", "normal", "high"), ("hold", "trickle", "fast"))


def _reading_in(value_rule: Fraction, rule_unit: str, reading_unit: str, places: int) -> Fraction:
    """A value given in the rule unit, written in the reading unit with `places` decimals (rounded; the label uses the written value)."""
    return quantize(UNITS[reading_unit].from_base(UNITS[rule_unit].to_base(Fraction(value_rule))), places)


def _shift(rule_unit: str, reading_unit: str) -> int:
    """How many decimal places a number gains going from the rule unit to the reading unit (−2 for m → cm)."""
    factor = UNITS[rule_unit].factor / UNITS[reading_unit].factor
    return -round(math.log10(float(factor)))


def _convert_thresholds(rnd: random.Random, domain: str, rule_unit: str, k: int, low: float, high: float) -> list[Fraction]:
    if domain == "temperature":
        found: set[Fraction] = set()
        while len(found) < k:
            found.add(quantize(rnd.uniform(low, high) * 2, 0) / 2)
        return sorted(found)
    if rule_unit == "frac":
        found = set()
        while len(found) < k:
            found.add(quantize(rnd.uniform(low, high), 2))
        out = sorted(found)
        return out
    return draw_thresholds(rnd, k, low, high)


def make_convert(rnd: random.Random, index: int) -> dict[str, Any]:
    domain, rule_unit, reading_units, (low, high) = rnd.choice(CONVERT_PAIRS)
    reading_unit = rnd.choice(reading_units)
    template = rnd.choice(CONVERT_TEMPLATES)
    style = draw_style(rnd)
    ru, vu = UNITS[rule_unit], UNITS[reading_unit]
    questions, labels, qmeta = {}, {}, {}
    what = {"distance": "gap", "time": "duration", "mass": "load", "speed": "speed", "temperature": "temperature", "percent": "fill level"}[domain]
    of_full = " of full" if rule_unit == "frac" else ""
    if template in ("convert-step", "convert-limit", "convert-window"):
        k = 1 if template == "convert-limit" else 2
        thresholds = _convert_thresholds(rnd, domain, rule_unit, k, low, high)
        t_dec = max(decimals_of(t) for t in thresholds)
        t_texts = [render(t, rule_unit, style, t_dec, signed=t < 0) for t in thresholds]
        base_t = [ru.to_base(t) for t in thresholds]
        op = rnd.choice(("le", "lt")) if template == "convert-limit" else None
        item_names = rnd.sample(("unit_1", "unit_2", "unit_3", "left", "right", "north", "south"), rnd.randint(1, 3))
        readings = []
        for _ in item_names:
            places = max(0, min(3, t_dec + rnd.choice((0, 1, 1, 2)) + _shift(rule_unit, reading_unit)))
            for _attempt in range(50):
                roll = rnd.random()
                t = rnd.choice(thresholds)
                if roll < 0.55:
                    in_rule = near(rnd, t, decimals=min(6, t_dec + 3))
                elif roll < 0.6:
                    in_rule = t
                else:
                    width = float(thresholds[-1] - thresholds[0]) or abs(float(thresholds[0])) or 1.0
                    in_rule = Fraction(rnd.uniform(float(thresholds[0]) - width, float(thresholds[-1]) + width))
                v = _reading_in(in_rule, rule_unit, reading_unit, places)
                if domain != "temperature" and v <= 0:
                    continue
                if op == "lt" and vu.to_base(v) == base_t[0]:  # "must stay below": a reading exactly on the limit is left out
                    continue
                break
            readings.append(v)
        rdec = max(decimals_of(v) for v in readings)
        v_texts = [render(v, reading_unit, style, rdec, signed=v < 0) for v in readings]
        base_vals = [vu.to_base(v) for v in readings]
        if template == "convert-step":
            names = rnd.choice(CONVERT_STEP_NAMES)
            inclusive = rnd.choice(("lower", "upper"))
            bands = step_bands(names, [t + of_full for t in t_texts], inclusive)
            rule = f"Rule for the {what}: " + "; ".join(f"{name} {words}" for name, words in bands) + "."
            criteria = {n: "" for n in names}
            order = list(criteria)
            rnd.shuffle(order)
            criteria = {key: criteria[key] for key in order}
            for it, vt, b in zip(item_names, v_texts, base_vals):
                passed = sum(1 for t in base_t if (b >= t if inclusive == "lower" else b > t))
                answer = names[passed]
                name = f"q_{it}"
                questions[name] = {"type": "choice", "instructions": rnd.choice((f"What does the rule say for {it.replace('_', ' ')}?",
                                                                                 f"Which option fits {it.replace('_', ' ')}?")), "criteria": dict(criteria)}
                labels[name] = _one_hot(criteria, answer, kind="rule")
                qmeta[name] = {"kind": "convert_step", "value": {"text": vt, "unit": reading_unit}, "answer": answer}
            numeric_rule = {"rule_unit": rule_unit, "thresholds": t_texts, "names": list(names), "inclusive": inclusive}
        elif template == "convert-limit":
            ok, bad = rnd.choice((("within", "over"), ("ok", "too_much"), ("fits", "too_long"), ("allowed", "not_allowed")))
            rule = (f"The {what} may be at most {t_texts[0]}{of_full}." if op == "le" else f"The {what} must stay below {t_texts[0]}{of_full}.")
            criteria = {ok: "", bad: ""} if rnd.random() < 0.5 else {bad: "", ok: ""}
            for it, vt, b in zip(item_names, v_texts, base_vals):
                answer = ok if (b <= base_t[0] if op == "le" else b < base_t[0]) else bad
                name = f"q_{it}"
                questions[name] = {"type": "choice", "instructions": rnd.choice((f"Is {it.replace('_', ' ')} within the limit?",
                                                                                 f"Check {it.replace('_', ' ')} against the limit.")), "criteria": dict(criteria)}
                labels[name] = _one_hot(criteria, answer, kind="rule")
                qmeta[name] = {"kind": "convert_limit", "value": {"text": vt, "unit": reading_unit}, "answer": answer}
            numeric_rule = {"rule_unit": rule_unit, "limit": t_texts[0], "op": op, "names": {"pass": ok, "fail": bad}}
        else:  # window
            inside, below, above = rnd.choice(WINDOW_NAMES)
            rule = f"Accept a {what} from {t_texts[0]}{of_full} to {t_texts[1]}{of_full}, both limits included."
            criteria = {inside: "", below: "", above: ""}
            order = list(criteria)
            rnd.shuffle(order)
            criteria = {key: criteria[key] for key in order}
            for it, vt, b in zip(item_names, v_texts, base_vals):
                answer = {"inside": inside, "below": below, "above": above}[interval_label(b, base_t[0], base_t[1], low_in=True, high_in=True)]
                name = f"q_{it}"
                questions[name] = {"type": "choice", "instructions": f"Where does {it.replace('_', ' ')} fall?", "criteria": dict(criteria)}
                labels[name] = _one_hot(criteria, answer, kind="rule")
                qmeta[name] = {"kind": "convert_window", "value": {"text": vt, "unit": reading_unit}, "answer": answer}
            numeric_rule = {"rule_unit": rule_unit, "window": {"low": t_texts[0], "high": t_texts[1], "low_in": True, "high_in": True},
                            "names": {"inside": inside, "below": below, "above": above}}
        state = "\n".join([rule, "Readings:", *(f"{it.replace('_', ' ')}: {vt}" for it, vt in zip(item_names, v_texts)),
                           *distractor_lines(rnd, list(criteria), rate=0.3)])
    else:  # compare in mixed units
        count = rnd.randint(3, 6)
        names = rnd.sample(("item_a", "item_b", "item_c", "item_d", "item_e", "item_f", "item_g"), count)
        units_here = [rule_unit, *reading_units]
        mode = rnd.choice(("largest", "smallest"))
        for _ in range(200):
            shown = []
            for _i in range(count):
                u = rnd.choice(units_here)
                places = max(0, min(3, 2 + _shift(rule_unit, u)))
                shown.append((u, _reading_in(Fraction(rnd.uniform(low, high)), rule_unit, u, places)))
            base = [UNITS[u].to_base(v) for u, v in shown]
            if len(set(base)) == count and (domain == "temperature" or all(v > 0 for _, v in shown)):
                break
        else:  # pragma: no cover
            raise RuntimeError("could not draw mixed-unit values")
        texts = {n: render(v, u, style, None, signed=v < 0) for n, (u, v) in zip(names, shown)}
        answer = compare_answer(mode, dict(zip(names, base)))
        state = "\n".join([f"Recorded {what} values (mixed units):", *(f"- {n.replace('_', ' ')}: {texts[n]}{' of full' if u == 'frac' else ''}"
                                                                         for n, (u, _) in zip(names, shown))])
        criteria = {n: "" for n in names}
        name = "q_" + mode
        questions[name] = {"type": "choice", "instructions": f"Which one has the {'highest' if mode == 'largest' else 'lowest'} {what}?",
                           "criteria": criteria}
        labels[name] = _one_hot(criteria, answer, kind="state")
        qmeta[name] = {"kind": "compare", "mode": mode, "items": dict(texts), "units": {n: u for n, (u, _) in zip(names, shown)}, "target": None,
                       "orders": None, "answer": answer}
        numeric_rule = {"rule_unit": rule_unit}
    return {"family": "convert", "template": template, "domain": domain, "style": style, "state": state, "questions": questions, "labels": labels,
            "numeric": {"rule": numeric_rule, "questions": qmeta}}


# --------------------------------------------------------------------------
# records
# --------------------------------------------------------------------------

MAKERS: dict[str, Callable[[random.Random, int], dict[str, Any]]] = {
    "step": make_step, "interval": make_interval, "select": make_select, "compare": make_compare, "convert": make_convert,
}
TEMPLATES = {"step": STEP_TEMPLATES, "interval": INTERVAL_TEMPLATES, "select": SELECT_TEMPLATES, "compare": COMPARE_TEMPLATES,
             "convert": CONVERT_TEMPLATES}


def _one_hot(criteria: dict[str, str], answer: str, *, kind: str) -> dict[str, Any]:
    if answer not in criteria:
        raise ValueError(f"answer {answer!r} is not an option ({list(criteria)[:5]} …)")
    return {"type": "choice", "kind": kind, "probabilities": {k: (1.0 if k == answer else 0.0) for k in criteria}}


def arm_rule_problems(unit: str, thresholds: Sequence[Fraction], names: Sequence[str], *, directions: Sequence[str] = ()) -> list[str]:
    """Why a step rule would copy the arm workcell's rule or its option names (module docstring): its threshold set in cm; three or more of
    its category names in one rule; its hold option ``stay``; or the directions ``plus`` / ``minus`` with any of its category names (which
    would make its option keys, ``plus_fine`` and the like)."""
    problems = []
    if unit == "cm" and tuple(Fraction(t) for t in thresholds) == ARM_THRESHOLDS_CM:
        problems.append("the arm's thresholds 0.2 / 1 / 3 / 8 / 20 cm")
    if sum(1 for n in set(names) if n in ARM_CATEGORY_NAMES) >= 3:
        problems.append("three or more of the arm's category names")
    if "stay" in names:
        problems.append("the arm's hold option 'stay'")
    if {"plus", "minus"} & set(directions) and ARM_CATEGORY_NAMES & set(names):
        problems.append("plus / minus with an arm category name (the arm's option keys)")
    return problems


def schedule(count: int) -> list[str]:
    """The family of each of `count` records, in record order: the family shares rounded to whole counts (largest remainder), interleaved
    by a seeded shuffle."""
    raw = {f: FAMILY_SHARES[f] * count for f in FAMILIES}
    counts = {f: int(math.floor(v)) for f, v in raw.items()}
    for f in sorted(FAMILIES, key=lambda f: raw[f] - counts[f], reverse=True)[: count - sum(counts.values())]:
        counts[f] += 1
    out = [f for f in FAMILIES for _ in range(counts[f])]
    _seed("schedule", count).shuffle(out)
    return out


def make_record(family: str, index: int, *, record_id: str | None = None) -> dict[str, Any]:
    """Record `index` of `family` (deterministic): a Decisions request with one-hot labels and ``meta.numeric`` (module docstring). Random
    codes (a quarter of the questions) and the 255-option opt-in follow v0.8's rules."""
    from robo_jev.decisions.records import OPTION_RANGE, OPTION_RANGE_MARK_255
    from robo_jev.simgen.format_v08 import codes_selected, draw_codes

    rnd = _seed("record", family, index)
    made = MAKERS[family](rnd, index)
    rid = record_id or f"{ID_PREFIX}{family}-{index:06d}"
    meta: dict[str, Any] = {"set": SET_NAME, "domain": f"numeric:{made['domain']}", "language": "en", "split": "train",
                            "numeric": {"version": VERSION, "family": family, "template": made["template"], "domain": made["domain"],
                                        "style": made["style"].describe(), **made["numeric"]}}
    codes = {}
    for name, question in made["questions"].items():
        if codes_selected(rid, name):
            codes[name] = draw_codes(rid, name, len(question["criteria"]))
    if codes:
        meta["codes"] = codes
    if any(len(q["criteria"]) > OPTION_RANGE[1] for q in made["questions"].values()):
        meta["option_range"] = dict(OPTION_RANGE_MARK_255)
    return {"schema": "simgen-request/0", "id": rid, "generator": {"version": VERSION, "family": family, "template": made["template"], "index": index},
            "meta": meta, "request": {"state": made["state"], "questions": made["questions"]}, "labels": made["labels"]}


def generate(count: int, *, start: int = 0, id_ok: Callable[[str], bool] | None = None) -> list[dict[str, Any]]:
    """`count` records in schedule order (module docstring). `id_ok` (the assembly: not a dev record of the trainer's split, not an
    existing id) is asked for each id; a refused id gets a suffix ``-b``, ``-c`` … until accepted."""
    families = schedule(count)
    per_family: dict[str, int] = {f: 0 for f in FAMILIES}
    out = []
    for family in families:
        index = start + per_family[family]
        per_family[family] += 1
        rid = f"{ID_PREFIX}{family}-{index:06d}"
        if id_ok is not None:
            suffix = 0
            while not id_ok(rid):
                suffix += 1
                rid = f"{ID_PREFIX}{family}-{index:06d}-{chr(97 + suffix)}"
        out.append(make_record(family, index, record_id=rid))
    return out
