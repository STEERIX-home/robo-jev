"""Task v0.7a — deployment-shaped requests made from v0.6's training records, labels unchanged (hypothesis H-shape).

The PRE-DECLARED v0.7 VERDICT RULE (progress.md, 2026-10-15) changes one factor — request **shape** — by *transforming* records of
v0.6's input (never adding any). This module holds the pure parts; ``scripts/assemble_v07.py`` streams the input through them.

**Assignment** (:func:`assign_kind`) — a physics record (engine or MuJoCo) that the trainer does not hold out as dev draws one uniform
number from ``sha256(SHAPE_VERSION | kind | <shuffle key>)``: below :data:`SHEET_SHARE` (0.25) it becomes a **sheet**, below
``SHEET_SHARE + FACTS_SHARE`` (0.30) it gets its **facts moved into the questions' instructions**, otherwise its state is unchanged. The
shuffle key (:func:`robo_jev.decisions.records.shuffle_key`) is shared by the two records of a one-fact pair, so a pair gets one shape and
the same context; a second-pass copy (``<id>~r2``) has its own key and its own draw. Nothing here reads a label, except the LOCAL PLAN
line (:func:`plan_line`), which by the rule names the best action with p = 0.5 and a wrong one otherwise.

**Sheets** (:func:`build_sheet`) — the original state, cut into 1–3 contiguous chunks at safe boundaries (:func:`cut_points`: the start
of a top-level line, or a sentence start inside a one-paragraph state, never before a pronoun or a connective), is placed **verbatim and in
order** into a sectioned sheet of 1,500–8,000 characters in one of :data:`STYLES` (six independent layouts). The other sections are
context sampled from the record's key alone — role, robot status (subsystems that never bear on the physics), camera tracks of unrelated
far-away objects, routine log lines, overheard speech not addressed to the robot, and site notices whose **tone** (neutral / reassuring /
alarming, :data:`TONES`) is drawn independently of the labels. The filler budget is fixed per key from the group's longest state, so the
two records of a pair get the same filler. :func:`sheet_problems` proves the verbatim placement twice: every sentence of the original is
found in order, and the recorded chunk spans reproduce the original exactly.

**Facts in instructions** (:func:`facts_request`) — every question's instructions get the whole original situation (its lines stripped and
joined by single spaces, each sentence verbatim) wrapped around the original instructions in one of :data:`FACTS_TEMPLATES`; the state
becomes a minimal role or JSON-like status line (:data:`MINIMAL_STATES`). :func:`facts_problems` checks it.

**Many options** (:func:`expand_choice`, :class:`DistractorPool`) — about 8 % of the physics choice questions (English records; question
families action, outcome, direction, comparison) and about 8 % of the general records are expanded to N ∈ 9–26 options with distractors
drawn from **other records' options of other answer families** (:data:`ALLOWED`: an outcome can never answer "what should you do", an action
never "what does the item do", and so on), sharing no content word with the question's own options and instructions, never a "same /
equal / as planned / first / second" option, and — for direction and comparison questions — never an option with a comparison word.
Distractors get probability 0 (and, where the label stores expected costs, the worst own option's cost — a convention for the validator;
the trainer reads only the probabilities). The prompt sorts and permutes the option keys per training example, so the correct option's
letter is spread over A… for the chosen N. A record with more than 8 options carries the opt-in :data:`OPTION_RANGE_MARK`
(:func:`robo_jev.decisions.records.option_range_for`).
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

from robo_jev.decisions.records import OPTION_RANGE_MARK, WIDE_OPTION_RANGE

__all__ = [
    "ALLOWED",
    "FACTS_SHARE",
    "FACTS_TEMPLATES",
    "GENERAL_MANY_P",
    "MINIMAL_STATES",
    "N_RANGE",
    "PHYSICS_FAMILIES",
    "PHYSICS_MANY_P",
    "SHAPE_VERSION",
    "SHEET_CHARS",
    "SHEET_SHARE",
    "STYLES",
    "TONES",
    "DistractorPool",
    "PoolEntry",
    "assign_kind",
    "build_sheet",
    "content_words",
    "cut_points",
    "draw_count",
    "expand_choice",
    "facts_problems",
    "facts_request",
    "filler_texts",
    "GENERAL_SOURCES_ALLOWED",
    "general_distractors",
    "general_eligible",
    "general_source",
    "plan_line",
    "plan_pick",
    "question_family",
    "sentence_units",
    "sheet_problems",
    "unit",
]

SHAPE_VERSION = "v07-shape-1"
SHEET_SHARE = 0.25
FACTS_SHARE = 0.05
#: Per eligible question / record: P(expanded). Physics: 8 % of all physics choice questions over the eligible ≈ 47 % (English, families
#: action / outcome / direction / comparison) → 0.17. General: 8 % of the general records over the eligible ≈ 42 % → 0.19.
PHYSICS_MANY_P = 0.17
GENERAL_MANY_P = 0.19
N_RANGE = (9, 26)
SHEET_CHARS = (1500, 8000)
#: Log-uniform target length of a sheet (characters) — more short sheets than long ones, the whole range covered.
_TARGET = (1700, 7950)
TONES = ("neutral", "reassuring", "alarming")
PLAN_RIGHT_P = 0.5
#: An action question has a known best option when its top probability is at least this and some option is clearly wrong (≤ 0.1).
PLAN_BEST_MIN = 0.6
PLAN_WRONG_MAX = 0.1


def unit(*parts: Any) -> float:
    """A uniform number in [0, 1) from the sha256 of the parts — the only randomness of the transform besides :func:`rng`."""
    digest = hashlib.sha256("|".join(str(p) for p in parts).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / 2.0**64


def rng(*parts: Any) -> random.Random:
    digest = hashlib.sha256("|".join(str(p) for p in parts).encode("utf-8")).digest()
    return random.Random(int.from_bytes(digest[:8], "big"))


def assign_kind(key: str) -> str | None:
    """``"sheet"`` (p = :data:`SHEET_SHARE`), ``"facts"`` (p = :data:`FACTS_SHARE`) or None — a function of the record's shuffle key only."""
    u = unit(SHAPE_VERSION, "kind", key)
    if u < SHEET_SHARE:
        return "sheet"
    if u < SHEET_SHARE + FACTS_SHARE:
        return "facts"
    return None


def draw_count(key: str, name: str) -> int:
    """N ∈ :data:`N_RANGE` for an expanded question (uniform)."""
    low, high = N_RANGE
    return low + min(high - low, int(unit(SHAPE_VERSION, "n", key, name) * (high - low + 1)))


# --------------------------------------------------------------------------
# State units, cuts and the verbatim proof
# --------------------------------------------------------------------------

_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")
#: A unit that starts with one of these leans on the unit before it — no cut there.
_ANAPHORIC = re.compile(
    r"^(it|its|it's|they|their|theirs|them|this|these|that|those|he|she|his|her|hers|there|both|also|then|which|so|but|and|or|"
    r"the latter|the former|such|another|other|its own|each|here|now|meanwhile|however|still|yet)\b", re.IGNORECASE)
_ANAPHORIC_KO = ("그 ", "그것", "그는", "그녀", "그의", "그들", "그러", "그리고", "이 ", "이것", "이는", "저 ", "또한", "하지만", "따라서", "그래서")


def sentence_units(state: str) -> list[str]:
    """The checker's units of a state: each non-empty line, stripped, split after . ! ? before whitespace — every sentence of the state."""
    out = []
    for line in str(state).split("\n"):
        for piece in _SENTENCE_END.split(line.strip()):
            if piece.strip():
                out.append(piece.strip())
    return out


def _leans_back(text: str) -> bool:
    head = text.lstrip()
    return bool(_ANAPHORIC.match(head)) or head.startswith(_ANAPHORIC_KO) or head[:1].islower()


def cut_points(state: str) -> list[int]:
    """Offsets where a state may be cut into chunks. Between lines: before a top-level line (not indented — an indented line belongs to
    the block above it), unless the line before ends with ":" (a header) or the new line starts with a pronoun or a connective
    (:data:`_ANAPHORIC`). Inside a line: only in a paragraph-like state (at most three non-empty lines) and a line that is not a bullet or
    a numbered item — before a sentence that starts with a capital letter and does not lean on the one before."""
    text = str(state)
    lines = text.split("\n")
    paragraph_like = sum(1 for line in lines if line.strip()) <= 3
    out: list[int] = []
    pos = 0
    previous = ""
    for i, line in enumerate(lines):
        start = pos
        top = bool(line.strip()) and not line[:1].isspace()
        if i > 0 and top and not previous.rstrip().endswith(":") and not _ANAPHORIC.match(line.lstrip()) \
                and not line.lstrip().startswith(_ANAPHORIC_KO):
            out.append(start)
        if top and paragraph_like and not re.match(r"^\s*([-*•·]|\d+[.)])\s", line):
            for match in _SENTENCE_END.finditer(line):
                cut = start + match.end()
                if cut < start + len(line.rstrip()) and not _leans_back(text[cut:]):
                    out.append(cut)
        if line.strip():
            previous = line
        pos += len(line) + 1
    end = len(text.rstrip())
    return sorted({c for c in out if 0 < c < end})


def split_state(state: str, chunks: int, rnd: random.Random) -> list[tuple[int, int]]:
    """`chunks` contiguous ranges (start, end) covering the state, cut at :func:`cut_points` (fewer chunks if there are fewer cuts)."""
    cuts = cut_points(state)
    k = max(1, min(int(chunks), len(cuts) + 1))
    chosen = sorted(rnd.sample(cuts, k - 1)) if k > 1 else []
    bounds = [0, *chosen, len(state)]
    return [(bounds[i], bounds[i + 1]) for i in range(len(bounds) - 1)]


def sheet_problems(original: str, sheet: str, *, cuts: Sequence[int] | None = None, spans: Sequence[Sequence[int]] | None = None) -> list[str]:
    """Every way the sheet fails to hold the original state verbatim (empty = proved): (1) every sentence unit of the original
    (:func:`sentence_units`) is found in the sheet, in order; (2) with the recorded `cuts` (offsets in the original) and `spans` (offsets of
    the chunks in the sheet), each chunk ``original[cut_i:cut_{i+1}].strip()`` equals ``sheet[span_i]`` and the chunks cover the original;
    (3) the sheet's length is within :data:`SHEET_CHARS`."""
    problems = []
    position = 0
    for piece in sentence_units(original):
        found = sheet.find(piece, position)
        if found < 0:
            problems.append(f"sentence not found in order: {piece[:60]!r}")
            break
        position = found + len(piece)
    if cuts is not None or spans is not None:
        bounds = [0, *list(cuts or []), len(original)]
        if spans is None or len(spans) != len(bounds) - 1:
            problems.append(f"{len(bounds) - 1} chunks but {0 if spans is None else len(spans)} spans")
        else:
            last = -1
            for i, (start, end) in enumerate(spans):
                want = original[bounds[i]:bounds[i + 1]].strip()
                if sheet[start:end] != want:
                    problems.append(f"chunk {i} is not verbatim at its span")
                if start < last:
                    problems.append(f"chunk {i} is placed before chunk {i - 1}")
                last = end
            if "".join(original[bounds[i]:bounds[i + 1]] for i in range(len(bounds) - 1)) != original:
                problems.append("the chunks do not cover the original")
    low, high = SHEET_CHARS
    if not low <= len(sheet) <= high:
        problems.append(f"sheet has {len(sheet)} characters (want {low}..{high})")
    return problems


# --------------------------------------------------------------------------
# Context (label-independent filler) — written for this task
# --------------------------------------------------------------------------

#: Words no filler line may contain (physics of the scenes, the robot's motion, people near its path, instructions to the robot).
FILLER_FORBIDDEN = re.compile(
    r"\b(load|loads|cargo|payload|tray|trays|deck|grip|gripper|grippers|clamp|surface|floor|floors|wet|slippery|icy|ice|slope|ramp|ramps|"
    r"brake|brakes|braking|speed|speeds|turn|turning|stop|stops|stopping|slow|slower|fast|faster|path|aisle|lane|corridor|collide|collision|"
    r"contact|tip|tips|tipping|topple|slide|slides|sliding|slip|spill|spills|fall|falls|falling|child|children|pallet|pallets|box|boxes|"
    r"crate|crates|tote|totes|jug|cup|cups|bin|bins|shelf|shelves|rack|racks|careful|caution|wait|hurry|move|moving|carry|carrying|push|"
    r"lift|lifts|weight|heavy|friction|tilt|accelerat\w*|decelerat\w*)\b", re.IGNORECASE)

_UNIT_PREFIX = ("RX", "AMR", "unit", "node", "bot", "SR", "MX", "LV")
_STATUS = (
    lambda r: f"battery: {r.randint(35, 98)} %",
    lambda r: "battery health: good",
    lambda r: f"processor usage: {r.randint(9, 64)} %",
    lambda r: f"processor temperature: {r.randint(39, 63)} °C",
    lambda r: f"uplink: ok, {r.randint(6, 95)} ms round trip",
    lambda r: f"wifi access point: AP-{r.randint(1, 48):02d}",
    lambda r: f"firmware: {r.randint(3, 9)}.{r.randint(0, 19)}.{r.randint(0, 40)}",
    lambda r: f"map revision: r{r.randint(100, 980)}",
    lambda r: "localisation quality: good",
    lambda r: f"odometer since commissioning: {r.randint(120, 9800)} km",
    lambda r: f"uptime: {r.randint(0, 71)} h {r.randint(0, 59):02d} min",
    lambda r: f"last full self-test: {r.randint(1, 13)} days ago, passed",
    lambda r: f"log upload queue: {r.randint(0, 40)} files",
    lambda r: f"storage used: {r.randint(18, 77)} %",
    lambda r: "safety controller: ok",
    lambda r: "safety interlock: closed (normal)",
    lambda r: "fleet manager link: connected",
    lambda r: f"time since last charge: {r.randint(0, 9)} h {r.randint(0, 59):02d} min",
    lambda r: "software watchdog: ok",
    lambda r: "status light: steady blue",
    lambda r: f"speaker volume: {r.choice((20, 30, 40, 50, 60))} %",
    lambda r: "front display: route summary",
    lambda r: f"clock drift vs server: {r.randint(1, 40)} ms",
    lambda r: "camera self-check: ok",
    lambda r: "lidar self-check: ok",
    lambda r: f"next scheduled service: in {r.randint(2, 60)} days",
)
_TRACK_OBJECTS = (
    "wall clock", "fire extinguisher cabinet", "parked cleaning cart", "notice board", "vending machine", "closed fire door",
    "potted plant", "water cooler", "office printer", "coat stand", "folded stepladder against a wall", "parked scrubber machine, powered off",
    "first-aid cabinet", "hose reel on a wall", "charging dock, unoccupied", "parked hand truck", "umbrella stand", "wall-mounted screen",
    "glass partition", "mail lockers", "electrical panel, closed", "hand sanitiser station", "parked bicycle", "flag on a pole outside",
    "seated worker behind a glass partition", "technician on the upper walkway", "courier at the front desk", "guard inside the gatehouse",
    "cleaner on the mezzanine", "receptionist at the counter",
)
_TRACK_WHERE = (
    "rear", "rear left", "rear right", "far side of the glass", "upper level", "outside the window", "behind a closed door",
    "beyond the partition", "far end of the hall, behind the counter", "across the atrium, on the gallery",
)
_TRACK_MOTION = ("static", "static", "static", "receding", "parked", "stationary")
_LOG_LINES = (
    lambda r: f"map tile {r.randint(1, 240)} refreshed from the fleet server",
    lambda r: f"uplink roamed to AP-{r.randint(1, 48):02d}",
    lambda r: f"job queue synced ({r.randint(0, 9)} jobs pending for other units)",
    lambda r: f"fire door {r.randint(1, 12)} reported closed",
    lambda r: f"elevator {r.randint(1, 4)} called by unit {r.choice(_UNIT_PREFIX)}-{r.randint(10, 99)}",
    lambda r: f"charging dock {r.randint(1, 9)} reported free",
    lambda r: "heartbeat to fleet manager ok",
    lambda r: f"log archive uploaded ({r.randint(2, 180)} MB)",
    lambda r: f"printer at desk {r.randint(1, 30)} reported out of paper",
    lambda r: f"badge reader at entrance {r.randint(1, 6)} updated",
    lambda r: "ambient light sensor recalibrated",
    lambda r: f"time sync with server ({r.randint(1, 30)} ms offset)",
    lambda r: f"maintenance ticket #{r.randint(1000, 9999)} closed for unit {r.choice(_UNIT_PREFIX)}-{r.randint(10, 99)}",
    lambda r: "weather feed: no change",
    lambda r: "status light self-test passed",
    lambda r: f"fleet roster updated ({r.randint(3, 30)} units online)",
    lambda r: f"configuration backup written (rev {r.randint(10, 400)})",
    lambda r: "display brightness adjusted to room light",
    lambda r: f"network scan: {r.randint(2, 15)} access points visible",
    lambda r: "audio pipeline restarted after an update",
)
_SPEECH_NONE = ("(no speech detected in the last {n} s)", "(microphone idle since the last cycle)", "(no voice input for {n} s)")
_SPEECH_LINES = (
    "Has anyone seen the blue folder?", "The meeting moved to room four.", "Can you send me the rota for next week?",
    "I'll grab a coffee after this call.", "Did the parcel for accounts arrive?", "Remember to sign the visitors' book.",
    "We need more printer paper upstairs.", "The heating is on again in the east wing.", "Lunch is at half past twelve today.",
    "Is the projector in room two working?", "I think the train was late again.", "Who is on call this weekend?",
)
_NOTICE = {
    "neutral": (
        "The cafeteria opens at the usual time today.", "The fleet software update is scheduled for next week.",
        "Quarterly inventory counts continue in another building.", "Parking permits are being renewed this month.",
        "Please file timesheets by Friday.", "The badge readers at the east entrance were replaced.",
        "Network maintenance window: Sunday night.", "Training sessions for new staff run on Thursdays.",
        "The visitor car park will be repainted next month.", "Recycling collection moves to Tuesdays from next week.",
    ),
    "reassuring": (
        "All units passed their morning safety checks.", "No incidents have been logged on this site for {n} days.",
        "Operations are running smoothly and on schedule today.", "Last week's site safety audit found no issues.",
        "Supervisors report a calm, routine shift so far.", "Every unit in the fleet reports healthy status.",
        "Near-miss reports are at their lowest level this year.", "The team is fully staffed and everything is under control.",
        "Today's risk assessment for the site: low.", "Management thanks everyone for another incident-free month.",
    ),
    "alarming": (
        "Safety alert: an unrelated incident in another building is under investigation.",
        "{n} near-miss reports were filed across the site this month.",
        "Reminder: last week's accident review found missed warnings in another area.",
        "An elevated hazard notice was issued after an incident elsewhere on site.",
        "Insurance inspectors visit tomorrow; every incident will be escalated.",
        "Two units were taken out of service yesterday after faults (not this unit).",
        "The safety team warns that incident numbers are rising this quarter.",
        "Management has flagged this week as high risk for errors.", "Today's risk assessment for the site: high.",
        "Emergency drill results were poor; a review is under way.",
    ),
}
_ROLE = (
    "You are the judgment layer of unit {unit}. Answer each question from the situation in this sheet; the controller below you handles exact motion.",
    "On-board risk judgment for {unit}. Report what is likely to happen next; you do not drive the motors.",
    "This sheet goes to the judgment model of unit {unit} every cycle; the model answers the questions attached to it.",
    "Judgment service for fleet unit {unit}. Read the sheet, then answer the questions that follow it.",
    "You assist unit {unit} with short-horizon judgment calls. The questions come after this sheet.",
)
_PLAN_NONE = ("none this cycle", "no recommendation (planner idle)", "— (nothing proposed)", "not available", "no proposal yet")


def _unit_name(r: random.Random) -> str:
    return f"{r.choice(_UNIT_PREFIX)}-{r.randint(10, 99)}{r.choice(('', 'A', 'B', 'C'))}"


def filler_texts() -> list[str]:
    """Every fixed filler string (templates with their placeholders filled with a sample value) — for the forbidden-word test."""
    r = random.Random(0)
    out = [f(r) for f in _STATUS] + list(_TRACK_OBJECTS) + list(_TRACK_WHERE) + list(_TRACK_MOTION) + [f(r) for f in _LOG_LINES]
    out += list(_SPEECH_LINES) + [f"{style.header} {style.bullet} {' '.join(style.names.values())} {' '.join(style.cont)}" for style in STYLES]
    out += list(FACTS_TEMPLATES) + [f(random.Random(0), "RX-10") for f in MINIMAL_STATES]
    out += [s.format(n=30) for s in _SPEECH_NONE] + [s.format(n=40) for lines in _NOTICE.values() for s in lines]
    out += [s.format(unit="RX-10") for s in _ROLE] + list(_PLAN_NONE)
    return out


# --------------------------------------------------------------------------
# Sheet styles
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Style:
    """One sheet layout: section names, how a header, a bullet and a key/value line look, and the section order (``S1``–``S3`` are the
    original chunks; optional sections may be dropped)."""

    name: str
    names: Mapping[str, str]
    header: str  # format with {name}
    bullet: str  # format with {text}
    sep: str  # between sections
    order: tuple[str, ...]
    cont: tuple[str, str]  # names of the 2nd and 3rd chunk sections


STYLES: tuple[Style, ...] = (
    Style("console", {"role": "ROLE", "status": "ROBOT STATUS", "s1": "CURRENT SITUATION", "tracks": "CAMERA TRACKS", "log": "EARLIER OBSERVATIONS",
                      "speech": "SPEECH", "plan": "LOCAL PLAN", "notice": "SITE NOTICE"},
          "=== {name} ===", "- {text}", "\n", ("notice", "role", "status", "s1", "tracks", "s2", "log", "s3", "speech", "plan"),
          ("SITUATION (CONTINUED)", "MORE ON THE SITUATION")),
    Style("markdown", {"role": "Role", "status": "Unit health", "s1": "What is happening now", "tracks": "Vision objects", "log": "Event log",
                       "speech": "Audio", "plan": "Planner", "notice": "Bulletin"},
          "## {name}", "* {text}", "\n", ("role", "s1", "s2", "status", "tracks", "log", "s3", "speech", "plan", "notice"),
          ("Situation, continued", "Further detail")),
    Style("brackets", {"role": "assignment", "status": "sys", "s1": "scene", "tracks": "perception.tracks", "log": "history",
                       "speech": "mic", "plan": "local_plan", "notice": "site_bulletin"},
          "[{name}]", "{text}", "\n", ("role", "status", "notice", "s1", "s2", "tracks", "log", "speech", "s3", "plan"),
          ("scene.more", "scene.extra")),
    Style("memo", {"role": "WHO YOU ARE", "status": "UNIT CONDITION", "s1": "TASK AND SITUATION", "tracks": "WHAT THE CAMERAS SEE",
                   "log": "SHIFT LOG", "speech": "VOICES NEARBY", "plan": "PLANNER SUGGESTION", "notice": "FROM THE SUPERVISOR"},
          "{name}:", "  {text}", "\n\n", ("notice", "role", "s1", "status", "s2", "tracks", "log", "speech", "s3", "plan"),
          ("SITUATION, CONTINUED", "ADDITIONAL FACTS")),
    Style("numbered", {"role": "ROLE", "status": "UNIT STATUS", "s1": "CURRENT TASK", "tracks": "PERCEPTION", "log": "EVENT RECORD",
                       "speech": "SPEECH TRANSCRIPT", "plan": "LOCAL PLAN", "notice": "NOTICES"},
          "{index}. {name}", "   {index}.{sub} {text}", "\n", ("role", "status", "notice", "s1", "tracks", "s2", "log", "speech", "s3", "plan"),
          ("CURRENT TASK (CONT.)", "FURTHER FACTS")),
    Style("tagged", {"role": "ROLE", "status": "SYS", "s1": "NOW", "tracks": "TRACKS", "log": "EVENTS", "speech": "AUDIO", "plan": "PLAN",
                     "notice": "MSG"},
          "#{name}", "  · {text}", "\n", ("role", "status", "tracks", "s1", "log", "s2", "speech", "notice", "s3", "plan"),
          ("NOW+", "NOW++")),
)
@dataclass
class SheetPlan:
    """Everything about a sheet that depends on the key only (pair members share it)."""

    style: Style
    tone: str
    target: int
    chunks: int
    sections: dict[str, bool]
    seed: int
    unit_name: str
    plan_mode: str  # "line" | "none" | "omit" (decided with the record's plan availability)
    meta: dict[str, Any] = field(default_factory=dict)


def sheet_plan(key: str) -> SheetPlan:
    r = rng(SHAPE_VERSION, "sheet", key)
    style = STYLES[int(unit(SHAPE_VERSION, "style", key) * len(STYLES))]
    tone = TONES[int(unit(SHAPE_VERSION, "tone", key) * len(TONES))]
    low, high = _TARGET
    target = int(round(math.exp(math.log(low) + unit(SHAPE_VERSION, "length", key) * (math.log(high) - math.log(low)))))
    u = unit(SHAPE_VERSION, "chunks", key)
    chunks = 1 if u < 0.5 else 2 if u < 0.85 else 3
    sections = {"status": r.random() < 0.9, "tracks": r.random() < 0.85, "log": r.random() < 0.9, "speech": r.random() < 0.6,
                "notice": True, "role": r.random() < 0.9}
    return SheetPlan(style=style, tone=tone, target=target, chunks=chunks, sections=sections, seed=r.getrandbits(63),
                     unit_name=_unit_name(r), plan_mode="omit")


def plan_pick(record: Mapping[str, Any]) -> dict[str, Any] | None:
    """The LOCAL PLAN recommendation a record may carry — None when it has no action question (MuJoCo ``meta.question_kind`` action,
    engine label kind action) or its state already carries a planner recommendation (engine ``meta.recommendation.present``). Both are
    facts of the record's structure, never of its labels, so whether a sheet *can* show a recommendation says nothing about the labels.

    The rule's recommendation: when the first action question has a **known best** option (top probability ≥ :data:`PLAN_BEST_MIN`, a
    unique top, and some option ≤ :data:`PLAN_WRONG_MAX`), the best with p = :data:`PLAN_RIGHT_P`, otherwise a random clearly wrong
    option (``right`` True / False); without a known best, a uniformly random option of the question (``right`` None — neither right nor
    wrong is defined). The coin and the draws come from the record's id. The option is named by its key (shown in the prompt) and its
    description."""
    meta = record.get("meta") or {}
    if (meta.get("recommendation") or {}).get("present"):
        return None
    kinds = meta.get("question_kind") or {}
    for name, question in record["request"]["questions"].items():
        label = (record.get("labels") or {}).get(name) or {}
        if question.get("type") != "choice" or not (label.get("kind") == "action" or kinds.get(name) == "action"):
            continue
        probs = label.get("probabilities") or {}
        if not probs:
            continue
        ranked = sorted(probs, key=lambda k: (-float(probs[k]), k))
        best = ranked[0]
        wrong = sorted(k for k in probs if float(probs[k]) <= PLAN_WRONG_MAX)
        known = float(probs[best]) >= PLAN_BEST_MIN and bool(wrong) and (len(ranked) < 2 or float(probs[ranked[1]]) < float(probs[best]))
        if known:
            right = unit(SHAPE_VERSION, "plan", record["id"]) < PLAN_RIGHT_P
            pick = best if right else wrong[int(unit(SHAPE_VERSION, "plan-wrong", record["id"]) * len(wrong))]
        else:
            right = None
            keys = sorted(probs)
            pick = keys[int(unit(SHAPE_VERSION, "plan-any", record["id"]) * len(keys))]
        return {"question": name, "option": pick, "right": right, "best": best, "known_best": known,
                "description": str(question["criteria"].get(pick) or "")}
    return None


def plan_line(pick: Mapping[str, Any] | None, *, key: str, style: Style) -> str:
    """The LOCAL PLAN body (one or two lines) for a pick (or a "no recommendation" line)."""
    r = rng(SHAPE_VERSION, "plan-text", key)
    if pick is None:
        return f"recommendation: {r.choice(_PLAN_NONE)}"
    desc = " ".join(str(pick.get("description") or "").split()).rstrip(" .;,")
    confidence = f"{r.uniform(0.52, 0.97):.2f}"
    forms = (
        f"recommended next action: {pick['option']}" + (f" ({desc})" if desc else ""),
        f"planner proposes {pick['option']}" + (f" — {desc}" if desc else "") + f"; planner confidence {confidence}",
        f"suggested option: {pick['option']}",
        f"next step queued by the local planner: {pick['option']}" + (f" ({desc})" if desc else "") + ", not yet executed",
    )
    return forms[int(unit(SHAPE_VERSION, "plan-form", key) * len(forms))]


def _render(style: Style, slot_name: str, index: int, lines: Sequence[str], *, prose: bool = False) -> str:
    """One section: its header and its lines (bullets, or the verbatim chunk when `prose`)."""
    head = style.header.format(name=slot_name, index=index)
    if prose:
        return head + "\n" + "\n".join(lines)
    body = [style.bullet.format(text=line, index=index, sub=j + 1) for j, line in enumerate(lines)]
    return head + ("\n" + "\n".join(body) if body else "")


def _filler_line(slot: str, r: random.Random, tone: str, clock: list[int]) -> str:
    if slot == "status":
        return r.choice(_STATUS)(r)
    if slot == "tracks":
        return (f"track {r.randint(100, 9999):04d}: {r.choice(_TRACK_OBJECTS)} | {r.randint(7, 31)}.{r.randint(0, 9)} m | "
                f"{r.choice(_TRACK_WHERE)} | {r.choice(_TRACK_MOTION)} | conf {r.uniform(0.6, 0.99):.2f}")
    if slot == "log":
        clock[0] += r.randint(4, 95)
        seconds = clock[0]
        return f"[-{seconds // 60:02d}:{seconds % 60:02d}] {r.choice(_LOG_LINES)(r)}"
    if slot == "speech":
        clock[1] += r.randint(10, 80)
        return f"[-{clock[1]}s] overheard, not addressed to this unit: \"{r.choice(_SPEECH_LINES)}\""
    if slot == "notice":
        return r.choice(_NOTICE[tone]).format(n=r.randint(3, 400))
    raise ValueError(slot)


def build_sheet(state: str, *, key: str, reference_length: int, pick: Mapping[str, Any] | None,
                plan_wanted: bool = True) -> tuple[str, dict[str, Any]]:
    """The sheet for one record (module docstring) → (sheet text, info: style, tone, target, cuts, spans, sections, plan, length).

    `reference_length` = the longest state of the record's group: a filler line is kept while the sheet, counted as if it held a state of
    that length, stays within the target — so the filler depends on the key and the group only, and the two records of a pair get the
    same filler. `pick` = :func:`plan_pick` of the record (None: a "no recommendation" line or no LOCAL PLAN section)."""
    plan = sheet_plan(key)
    style = plan.style
    r = random.Random(plan.seed)
    ranges = split_state(state, plan.chunks, rng(SHAPE_VERSION, "cut", key))
    chunks = [state[a:b].strip() for a, b in ranges]
    own = sum(len(c) for c in chunks)
    if own > int(reference_length):
        raise ValueError(f"reference_length {reference_length} is shorter than the state's own text ({own})")
    # the LOCAL PLAN block: a record that can carry a recommendation (structure only) shows one with p = 0.5, else "no recommendation"
    # or no block (p = 0.25 each); any other record: "no recommendation" or no block (p = 0.5 each) — one coin per key
    u = unit(SHAPE_VERSION, "plan-mode", key)
    if pick is not None:
        plan_mode = "line" if u < 0.5 else "none" if u < 0.75 else "omit"
    else:
        plan_mode = "none" if u < 0.5 else "omit"
    if not plan_wanted:
        plan_mode = "omit"
    present = [s for s in style.order if (s in ("s1", "s2", "s3") and int(s[1]) <= len(chunks)) or (s == "plan" and plan_mode != "omit")
               or (s not in ("s1", "s2", "s3", "plan") and plan.sections.get(s, True))]
    lines: dict[str, list[str]] = {s: [] for s in present if s not in ("s1", "s2", "s3")}
    clock = [r.randint(3, 40), r.randint(2, 30)]
    if "role" in lines:
        lines["role"].append(r.choice(_ROLE).format(unit=plan.unit_name))
    if "notice" in lines:
        lines["notice"].append(_filler_line("notice", r, plan.tone, clock))
    if "speech" in lines and r.random() < 0.4:
        lines["speech"].append(r.choice(_SPEECH_NONE).format(n=r.choice((20, 30, 45, 60, 90))))
    if "plan" in lines:
        lines["plan"].append(plan_line(pick if plan_mode == "line" else None, key=key, style=style))
    for slot in ("status", "tracks", "log", "speech"):  # every present filler section shows at least one line
        if slot in lines and not lines[slot]:
            lines[slot].append(_filler_line(slot, r, plan.tone, clock))

    def render() -> tuple[str, list[tuple[int, int]]]:
        parts: list[str] = []
        spans_local: list[tuple[int, int]] = []
        length = 0
        for index, slot in enumerate(present, start=1):
            if parts:
                length += len(style.sep)
            if slot in ("s1", "s2", "s3"):
                i = int(slot[1]) - 1
                name = style.names["s1"] if i == 0 else style.cont[i - 1]
                head = style.header.format(name=name, index=index) + "\n"
                spans_local.append((length + len(head), length + len(head) + len(chunks[i])))
                block = head + chunks[i]
            else:
                block = _render(style, style.names[slot], index, lines[slot])
            parts.append(block)
            length += len(block)
        return style.sep.join(parts), spans_local

    text, spans = render()
    growable = [s for s in ("status", "tracks", "log", "speech", "notice") if s in lines and not (s == "speech" and lines[s][0].startswith("("))]
    weights = {"status": 2, "tracks": 4, "log": 5, "speech": 1, "notice": 1}
    def once(slot: str, line: str) -> str:  # a status field appears once (by its name), a notice once (whole line)
        return line.split(":", 1)[0] if slot == "status" else line

    used = {once(s, line) for s in ("status", "notice") for line in lines.get(s, [])}
    stalled = 0
    while growable and stalled < 60:
        slot = r.choices(growable, weights=[weights[s] for s in growable])[0]
        line = _filler_line(slot, r, plan.tone, clock)
        if slot in ("status", "notice") and once(slot, line) in used:
            stalled += 1
            continue
        lines[slot].append(line)
        trial, trial_spans = render()
        if len(trial) - own + int(reference_length) > plan.target:
            lines[slot].pop()
            stalled += 1
            continue
        text, spans = trial, trial_spans
        if slot in ("status", "notice"):
            used.add(once(slot, line))
        stalled = 0
    topped = 0
    spare = next((s for s in ("log", "tracks", "status", "notice") if s in lines), None)
    while len(text) < SHEET_CHARS[0] and spare is not None:  # rare: a record much shorter than its pair partner — top up to the minimum
        lines[spare].append(_filler_line(spare, r, plan.tone, clock))
        text, spans = render()
        topped += 1
    info = {"version": SHAPE_VERSION, "kind": "sheet", "style": style.name, "tone": plan.tone, "target": plan.target,
            "reference_length": int(reference_length), "chunks": len(chunks), "cuts": [a for a, _ in ranges[1:]],
            "spans": [list(s) for s in spans], "sections": present, "plan": plan_mode,
            "plan_eligible": pick is not None,
            "plan_pick": ({k: pick[k] for k in ("question", "option", "right", "known_best")} if plan_mode == "line" and pick else None),
            "topped_up_lines": topped, "length": len(text)}
    return text, info


# --------------------------------------------------------------------------
# Facts in instructions
# --------------------------------------------------------------------------

FACTS_TEMPLATES = (
    "Context: {facts} — {instruction}",
    "Given that: {facts} Now: {instruction}",
    "Facts: {facts} | Ask: {instruction}",
    "{instruction} Known facts: {facts}",
    "Situation report: {facts} Question for you: {instruction}",
)
MINIMAL_STATES = (
    lambda r, u: f"ROLE: judgment module (L2) of unit {u} — the facts come with each question.",
    lambda r, u: json.dumps({"node": "judge-l2", "unit": u, "mode": "auto", "battery_pct": r.randint(35, 98), "link": "ok"}),
    lambda r, u: f"unit {u} · status nominal · situation facts are given per question below",
    lambda r, u: json.dumps({"role": "judgment", "unit": u, "cycle": r.randint(1000, 99999), "facts": "in the questions"}),
    lambda r, u: f"[role] on-board judgment for {u}; read the facts in each question.",
)


def facts_text(state: str) -> str:
    """The situation as one paragraph: the state's non-empty lines, stripped, joined by single spaces (every sentence verbatim)."""
    return " ".join(line.strip() for line in str(state).split("\n") if line.strip())


def facts_request(record: Mapping[str, Any], *, key: str) -> tuple[str, dict[str, str], dict[str, Any]]:
    """(minimal state, question name → new instructions, info) for one record (module docstring). The template and the minimal state
    are drawn from the key (pair members share them)."""
    r = rng(SHAPE_VERSION, "facts", key)
    template_index = int(unit(SHAPE_VERSION, "facts-template", key) * len(FACTS_TEMPLATES))
    template = FACTS_TEMPLATES[template_index]
    state_index = int(unit(SHAPE_VERSION, "facts-state", key) * len(MINIMAL_STATES))
    minimal = MINIMAL_STATES[state_index](r, _unit_name(r))
    facts = facts_text(record["request"]["state"])
    out = {name: template.format(facts=facts, instruction=str(q.get("instructions") or ""))
           for name, q in record["request"]["questions"].items()}
    return minimal, out, {"version": SHAPE_VERSION, "kind": "facts", "template": template_index, "minimal_state": state_index,
                          "facts_chars": len(facts)}


def facts_problems(original: Mapping[str, Any], new: Mapping[str, Any]) -> list[str]:
    """Every way the facts transform broke its promise (empty = proved): each question's new instructions hold every sentence unit of the
    original state in order and the original instructions verbatim; the new state is non-empty and shares no sentence unit (≥ 3 words)
    with the original state; the question names are the same."""
    problems = []
    units = sentence_units(original["request"]["state"])
    old_q, new_q = original["request"]["questions"], new["request"]["questions"]
    if list(old_q) != list(new_q):
        problems.append("question names or order changed")
        return problems
    for name in old_q:
        text = str(new_q[name].get("instructions") or "")
        position = 0
        for piece in units:
            found = text.find(piece, position)
            if found < 0:
                problems.append(f"{name}: sentence not found in order: {piece[:60]!r}")
                break
            position = found + len(piece)
        if str(old_q[name].get("instructions") or "") not in text:
            problems.append(f"{name}: the original instructions are not kept verbatim")
    state = str(new["request"]["state"])
    if not state.strip():
        problems.append("the minimal state is empty")
    leaked = [u for u in units if len(u.split()) >= 3 and u in state]
    if leaked:
        problems.append(f"the minimal state still holds {len(leaked)} original sentence(s)")
    return problems


# --------------------------------------------------------------------------
# Many options
# --------------------------------------------------------------------------

_STOP = frozenset(
    "a an the of to in on at by for with from into onto over under as is are was were be been being it its this that these those there their "
    "they them he she his her you your yours we our i me my and or but if then than so not no do does did done can could will would should "
    "may might must shall have has had having all any each every some very just only also again still yet about around roughly approx "
    "approximately per via one two three four five six seven eight nine ten s m cm mm km ms h min kg n g deg".split())
#: Words that make an option read as a comparison or a level — never a distractor of a direction or comparison question.
COMPARISON_WORDS = re.compile(
    r"\b(more|less|most|least|higher|lower|high|low|increase\w*|decrease\w*|up|down|same|equal\w*|unchanged|change\w*|differ\w*|"
    r"riskier|risk\w*|safer|safe|likel\w*|unlikely|chance\w*|better|worse|faster|slower|greater|smaller|larger|bigger|rise\w*|"
    r"first|second|plan\w*|odds)\b", re.IGNORECASE)
#: Options that never become distractors anywhere: "same"-type, plan references, meta references.
_POOL_EXCLUDED = re.compile(
    r"\b(same|equal|equally|unchanged|no change|no difference|neither|either|both|none|nothing|as planned|as written|as commanded|"
    r"as briefed|as is|current plan|planned|plan|question|first|second|above|below)\b", re.IGNORECASE)
#: Engine state questions about an object's own state (not an agent's next move or a sensor choice) — the ``est_obj`` family.
OBJECT_STATE_NAMES = frozenset({"fill_level", "how_full", "how_it_sits", "its_state", "condition_now", "lid_on", "has_lid", "heavy_side",
                                "weight_side"})
PHYSICS_FAMILIES = ("act", "out", "dir", "cmp")
#: Question family → the families its distractors may come from (module docstring). Direction options never go to action or outcome
#: questions: the audit of the first build found "goes down" (≈ falls) under "what does the vase do?", a key-only "lower" under an action
#: list (≈ lower the lift) and "safer: less likely" under "which plan is safest?".
ALLOWED: dict[str, tuple[str, ...]] = {
    "act": ("out", "rule", "band", "est_obj"),
    "out": ("act", "band"),
    "dir": ("act", "out", "est_obj"),
    "cmp": ("out", "est_obj", "rule", "band"),
}
#: Question families whose distractors must carry no comparison word.
_NO_COMPARISON = frozenset({"dir", "cmp"})
#: Words that could make a non-action option read as something the robot could do ("keeps rolling" ≈ continue, "stops inside the window"
#: ≈ stop) — never a distractor of an action question.
ACTION_WORDS = re.compile(
    r"\b(stop\w*|keep\w*|hold\w*|continu\w*|mov\w*|go|goes|going|went|carr\w*|wait\w*|slow\w*|brak\w*|turn\w*|lift\w*|lower\w*|"
    r"rais\w*|push\w*|pull\w*|press\w*|place\w*|put\w*|set\w*|leav\w*|drop\w*|back\w*|ask\w*|call\w*|check\w*|look\w*|"
    r"steer\w*|yield\w*|reseat\w*|redo|retry|undo|pause\w*|halt\w*|clamp\w*|grip\w*|release\w*)\b", re.IGNORECASE)
#: Words that could make a non-outcome option read as what the item does ("Hold your position for now" ≈ stays put, "Come to rest" ≈
#: settles) — never a distractor of an outcome question.
OUTCOME_WORDS = re.compile(
    r"\b(hold\w*|stay\w*|remain\w*|keep\w*|stop\w*|rest\w*|settl\w*|fall\w*|tip\w*|slid\w*|slip\w*|skid\w*|drop\w*|toppl\w*|"
    r"roll\w*|level|steady|still|stable|stand\w*|halt\w*|pause\w*|lean\w*|shift\w*|mov\w*|rock\w*|lift\w*|spill\w*|break\w*|"
    r"crash\w*|hit\w*|catch\w*|caught)\b", re.IGNORECASE)


def _stem(word: str) -> str:
    for suffix in ("ingly", "edly", "ing", "ied", "ies", "ed", "es", "s", "ly"):
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            word = word[: -len(suffix)] + ("y" if suffix in ("ies", "ied") else "")
            break
    if len(word) > 3 and word.endswith("e"):
        word = word[:-1]
    if len(word) > 3 and word[-1] == word[-2] and word[-1] not in "aeiou":
        word = word[:-1]
    return word


def content_words(text: str) -> set[str]:
    """Stemmed content words of an English text (letters only, stop words and units dropped)."""
    return {_stem(w) for w in re.findall(r"[a-z]+", str(text).lower().replace("_", " ")) if w not in _STOP and len(w) > 1}


def _norm(text: str) -> str:
    return " ".join(re.sub(r"[\W_]+", " ", str(text).lower()).split())


def question_family(record: Mapping[str, Any], name: str, part: str) -> str | None:
    """The answer family of a choice question: ``act`` / ``out`` / ``dir`` / ``cmp`` (MuJoCo ``meta.question_kind``; engine label kind
    ``action`` and the derived ``s1_cmp_*``), ``band`` (engine ``s1_band_*``), ``rule`` (engine label kind ``rule``), ``est_obj`` (engine
    object-state questions), ``est`` (other engine state questions) — None for anything else."""
    question = record["request"]["questions"][name]
    if question.get("type") != "choice":
        return None
    label = (record.get("labels") or {}).get(name) or {}
    if part == "mujoco":
        return {"action": "act", "outcome": "out", "direction": "dir", "compare": "cmp"}.get(((record.get("meta") or {}).get("question_kind") or {}).get(name))
    if part == "engine":
        if name.startswith("s1_band"):
            return "band"
        if name.startswith("s1_cmp"):
            return "cmp"
        kind = label.get("kind")
        if kind == "action":
            return "act"
        if kind == "rule":
            return "rule"
        if kind == "state":
            return "est_obj" if re.sub(r"_\d+$", "", name) in OBJECT_STATE_NAMES else "est"
    return None


def scene_of(record: Mapping[str, Any], part: str) -> str:
    meta = record.get("meta") or {}
    return f"{meta.get('set') or part}:{meta.get('scene') or meta.get('group') or 'all'}"


@dataclass(frozen=True)
class PoolEntry:
    family: str
    key: str
    description: str
    scene: str
    words: frozenset
    norm: str
    comparison: bool
    action: bool
    outcome: bool


class DistractorPool:
    """Distractor candidates by family (English physics options of the training records), in first-seen file order (deterministic)."""

    def __init__(self) -> None:
        self.entries: dict[str, list[PoolEntry]] = {}
        self._seen: set[tuple[str, str, str]] = set()
        self.excluded = 0

    def add(self, family: str, key: str, description: str, scene: str) -> None:
        text = f"{key.replace('_', ' ')} {description}"
        norm = _norm(description) or _norm(key)
        marker = (family, key, norm)
        if marker in self._seen:
            return
        self._seen.add(marker)
        if _POOL_EXCLUDED.search(text.replace("_", " ")):
            self.excluded += 1
            return
        plain = text.replace("_", " ")
        self.entries.setdefault(family, []).append(PoolEntry(family, str(key), str(description), scene, frozenset(content_words(text)), norm,
                                                             bool(COMPARISON_WORDS.search(plain)), bool(ACTION_WORDS.search(plain)),
                                                             bool(OUTCOME_WORDS.search(plain))))

    def add_record(self, record: Mapping[str, Any], part: str) -> None:
        if (record.get("meta") or {}).get("language") != "en":
            return
        scene = scene_of(record, part)
        for name, question in record["request"]["questions"].items():
            family = question_family(record, name, part)
            if family is None or family == "est":
                continue
            for key, description in question["criteria"].items():
                self.add(family, str(key), str(description), scene)

    def sizes(self) -> dict[str, int]:
        return {family: len(entries) for family, entries in sorted(self.entries.items())}

    def draw(self, family: str, *, count: int, own_keys: Iterable[str], own_norms: Iterable[str], words: Iterable[str], scene: str,
             rnd: random.Random, max_attempts: int = 6000) -> list[PoolEntry] | None:
        """`count` distractors for a question of `family` (:data:`ALLOWED`), round-robin over the allowed families in a seeded order; each
        with a key unseen in the question, a text unseen, a scene other than the question's, no content word shared with `words` (the
        question's own options and instructions), no comparison word for direction / comparison questions (:data:`COMPARISON_WORDS`), no
        action word for action questions (:data:`ACTION_WORDS`) and no outcome word for outcome questions (:data:`OUTCOME_WORDS`). None if
        the pool cannot supply `count`."""
        families = [f for f in ALLOWED[family] if self.entries.get(f)]
        if not families or count <= 0:
            return [] if count <= 0 else None
        rnd.shuffle(families)
        keys, norms, banned = set(map(str, own_keys)), set(own_norms), set(words)
        chosen: list[PoolEntry] = []
        attempts = 0
        turn = 0
        while len(chosen) < count and attempts < max_attempts:
            source = families[turn % len(families)]
            turn += 1
            attempts += 1
            entry = self.entries[source][rnd.randrange(len(self.entries[source]))]
            if entry.key in keys or entry.norm in norms or entry.scene == scene or entry.words & banned:
                continue
            if family in _NO_COMPARISON and entry.comparison:
                continue
            if (family == "act" and entry.action) or (family == "out" and entry.outcome):
                continue
            chosen.append(entry)
            keys.add(entry.key)
            norms.add(entry.norm)
        return chosen if len(chosen) == count else None


def expand_choice(question: Mapping[str, Any], label: Mapping[str, Any], distractors: Sequence[tuple[str, str]]) -> tuple[dict[str, Any], dict[str, Any]]:
    """(question, label) with the distractors (key, description) appended: probability 0 each; ``expected_cost`` (if any) gets the worst
    own option's cost for each. Own options, their order, descriptions and probabilities are unchanged."""
    new_q = json.loads(json.dumps(question))
    new_l = json.loads(json.dumps(label))
    for key, description in distractors:
        if key in new_q["criteria"]:
            raise ValueError(f"distractor key {key!r} is already an option")
        new_q["criteria"][key] = description
        new_l["probabilities"][key] = 0.0
    costs = new_l.get("expected_cost")
    if isinstance(costs, dict) and costs:
        worst = max(float(v) for v in label["expected_cost"].values())
        for key, _ in distractors:
            costs[key] = worst
    return new_q, new_l


# -- general items --------------------------------------------------------

_META_OPTION = re.compile(r"\b(none of the above|all of the above|both|neither|none)\b", re.IGNORECASE)
_PRONOUN_START = re.compile(r"^(they|it|its|this|these|those|he|she|his|her|their|there|that|him|them|we|you|i)\b", re.IGNORECASE)


def general_eligible(record: Mapping[str, Any]) -> bool:
    """A general item that may be expanded: one choice question, its correct option a phrase of ≥ 3 words (a drawn short answer could be
    a second right answer to a short-answer item), and no "none / all of the above"-type option among its own."""
    questions = record["request"]["questions"]
    if len(questions) != 1:
        return False
    (name, question), = questions.items()
    if question.get("type") != "choice":
        return False
    probs = record["labels"][name]["probabilities"]
    gold = max(probs, key=lambda k: float(probs[k]))
    if len(str(question["criteria"][gold]).split()) < 3:
        return False
    return not any(_META_OPTION.search(str(v)) for v in question["criteria"].values())


def general_candidate(text: str) -> bool:
    """A general option that may serve as a distractor: ≥ 3 words, no digit, no pronoun start, no meta option."""
    text = " ".join(str(text).split())
    return (len(text.split()) >= 3 and not re.search(r"\d", text) and not _PRONOUN_START.match(text) and not _META_OPTION.search(text))


#: General source → the sources its distractors may come from: never its own (same-style answers), never a source whose answers are of
#: the same kind — Social IQa and CosmosQA answers are both people's feelings and reasons ("happy it happened" could answer "how would
#: Quinn feel?"), QASC and CommonsenseQA answers are both short science / everyday noun phrases (audit of the first build).
GENERAL_SOURCES_ALLOWED: dict[str, tuple[str, ...]] = {
    "hellaswag": ("qasc", "commonsense_qa", "cosmos_qa"),
    "social_i_qa": ("hellaswag", "qasc", "commonsense_qa"),
    "cosmos_qa": ("hellaswag", "qasc", "commonsense_qa"),
    "qasc": ("hellaswag", "social_i_qa", "cosmos_qa"),
    "commonsense_qa": ("hellaswag", "social_i_qa", "cosmos_qa"),
}


def general_source(record: Mapping[str, Any]) -> str:
    return str(((record.get("meta") or {}).get("source") or {}).get("set") or "")


def general_distractors(record: Mapping[str, Any], pool: Sequence[tuple[str, str, str]], *, count: int, rnd: random.Random,
                        max_attempts: int = 6000) -> list[str] | None:
    """`count` option texts from other general records (`pool` = (origin, source, text)) of a compatible source
    (:data:`GENERAL_SOURCES_ALLOWED`), sharing no content word with the item's state, its own options or the distractors drawn before (the
    item's topic never lends an answer), none equal to an own option. None if not enough."""
    (name, question), = record["request"]["questions"].items()
    origin = str((record.get("meta") or {}).get("origin_group") or record.get("id"))
    allowed = set(GENERAL_SOURCES_ALLOWED.get(general_source(record), ()))
    banned = content_words(record["request"]["state"]) | set().union(*(content_words(v) for v in question["criteria"].values()))
    norms = {_norm(v) for v in question["criteria"].values()}
    chosen: list[str] = []
    attempts = 0
    while len(chosen) < count and attempts < max_attempts:
        attempts += 1
        source_origin, source, text = pool[rnd.randrange(len(pool))]
        if source_origin == origin or source not in allowed or _norm(text) in norms:
            continue
        words = content_words(text)
        if not words or words & banned:
            continue
        chosen.append(text)
        banned |= words
        norms.add(_norm(text))
    return chosen if len(chosen) == count else None


def option_range_mark() -> dict[str, Any]:
    """The opt-in a record with more than 8 options carries (``meta.option_range``)."""
    return dict(OPTION_RANGE_MARK)


assert WIDE_OPTION_RANGE == (2, 26)
