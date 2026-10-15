"""Task v0.7b — the sealed v0.7 shape gates M2g (sheet-wrapped held-out moments) and M1g (many options), pure parts.

The PRE-DECLARED v0.7 VERDICT RULE (progress.md, 2026-10-15 ~05:45 KST, before any v0.7 data existed) reads two gates built by an author who
never sees the v0.7 training sheet generator (task v0.7a). This module builds them; nothing here loads a model. The build runs in
``scripts/v07_gates_build.py``, the scoring in ``artifacts/scratch/v07b/score_spark.sh`` and the reading in ``scripts/verdict_v07.py``.

**M2g — sheet-wrapped held-out moments** (S-a reads DC(wrapped_L) − DC(short) on the call view).

* *Moments*: whole one-fact pairs of A4 (the held-out author's sealed test — its second use, as pre-declared), the H-set (the v0.4 scene
  holdout) and B1-infamily, never trained on. Every condition carries the record's **call-view questions only** (its yes/no and score
  questions — the only questions any M2g reading uses), byte-identical to the original's, with the original labels; prompts are built per
  question, so dropping the choice questions changes no kept prompt (decided before any data, for GPU time).
* *Identity* (:func:`gate_record`): id ``v07m2g-<original id>`` (the partner in ``meta.pair.of`` gets the same prefix, so the order key — the
  smaller id of the pair — stays shared by the pair and is the same in every condition), ``meta.set`` = ``v07-m2g`` (sealed:
  :mod:`robo_jev.data.sealed`), ``meta.source`` = the original set, id and ``meta.set``; ``meta.gate.style`` = the pair's sheet style. Every
  other field is the original's. The six condition files hold the same ids, questions, labels and meta; only the state differs.
* *Sheets* (:func:`wrap_state`): three styles written for this gate (:data:`STYLES`) — a numbered work order (``ticket``), a shift-handover
  memo in prose (``handover``) and a console dump with key=value status lines, a pipe-table track list and an event feed (``console``). The
  original state goes **verbatim, once, as one contiguous block** into the style's report section; every original sentence is therefore kept
  verbatim and in order (:func:`sheet_problems` proves it per sheet). The context — status lines, tracks (people and objects ≥ 3.5 m away,
  static or moving away), log lines, a planner line with no recommendation — is drawn from pools seeded only by the pair's order key and
  the style, sized for the pair's longer state, so both records of a pair (which differ in their labels) get exactly the same filler; a
  longer sheet extends the shorter one (the same lines plus more). No filler line uses a word of :data:`FACT_WORDS` (task and physics
  words) or of :data:`TONE_WORDS` (evaluative words); the only evaluative text is the tone slot.
* *Tone slot*: three sentences of a site bulletin, chosen by the pair's key from three parallel banks (:data:`TONE_BANKS`): ``neutral``
  (administrative notices), ``reassuring`` and ``alarming`` (site-level mood, never about the moment). The main wrapped forms use the
  neutral bank; the **tone panel** at 4k swaps only the three sentences (the rest of the sheet is identical — checked), so ``w4k`` is the
  panel's neutral member.
* *Lengths*: ≈ 2,000 / 4,000 / 8,000 characters (:data:`LENGTHS`), filled round-robin over the style's three growing lists while the sheet
  for the pair's longer state and the longest of the three tone blocks stays within the target.

**M1g — many options** (S-b reads acc(N = 8) − acc(N ≥ 17) and the physics late-letter penalty).

* *General items* (:func:`mmlu_items`): MMLU test questions (cais/mmlu, MIT licence; used in no training input and not in M1), filtered as
  task iface's ARC items (exactly four options, none pointing at other options, no picture, no negative stem, no number in any option) plus:
  no option that points at lettered or numbered options or declines to answer, no stem asking for a rare property or a rejection ("SELDOM",
  "would be rejected by" — :data:`_NEGATION_EXTRA`), no symbol-heavy options (formulas), stem ≤ 300 characters, every option ≤ 100
  characters, the moral-scenarios subject left out (templated paired verdicts); the measured answer is a phrase of ≥ 3 words. Shown like the general training mix shows CommonsenseQA (``question:
  "<stem>"``, one fixed instruction, keys ``o1 … oN`` given by a seeded permutation).
* *Physics items*: the direction question of A4 and H-set records with a single clear best option (band-robust, top probability ≥ 0.9 —
  :func:`robo_jev.bench.iface.physics_items`), one per one-fact pair, H-set pairs that task iface's M1 used left out.
* *Distractors* from other items of the same source, nested across N, the same item and answer at every N: general — task iface's
  topic-distant draw (:func:`robo_jev.bench.iface.draw_general`); physics — options of other questions of the item's own set first, then of
  the sibling set, one per key the item does not have, never an option that states a direction or an equality (a paraphrase of an answer —
  :data:`PHYSICS_ANSWERLIKE`) and never a context-bound option ("the alternative") (:func:`draw_physics`).
* *Levels and positions*: N ∈ :data:`M1G_LEVELS` = {8, 21, 26}, three listings each — at N = 8 one per third of the list, at N = 21 / 26 one per
  bucket A–H / I–P / Q–Z (Q–U at 21) — cycling within the band so every letter is used about equally (:func:`robo_jev.bench.iface.positions`).
"""

from __future__ import annotations

import copy
import hashlib
import random
import re
from collections import defaultdict
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from robo_jev.bench import iface
from robo_jev.data.general_sets import normalize
from robo_jev.decisions.records import SCHEMA, shuffle_key

__all__ = [
    "CONDITIONS", "FACT_WORDS", "LENGTHS", "M1G_LEVELS", "M1G_PREFIX", "M1G_SET", "M2G_PREFIX", "M2G_SET", "PHYSICS_ANSWERLIKE", "STYLES",
    "TONES", "TONE_BANKS", "TONE_WORDS", "assign_styles", "call_view_questions", "draw_physics", "expand_item", "gate_record", "m1g_item_table",
    "mmlu_items", "sentences", "sheet_counts", "sheet_problems", "tone_lines", "tone_skeleton", "wrap_record", "wrap_state",
]

M2G_SET, M2G_PREFIX = "v07-m2g", "v07m2g-"
M1G_SET, M1G_PREFIX = "v07-m1g", "v07m1g-"
#: Sheet lengths (characters of the wrapped state).
LENGTHS = {"2k": 2000, "4k": 4000, "8k": 8000}
TONES = ("neutral", "reassuring", "alarming")
#: The tone panel's length; its neutral member is the main ``w4k`` condition.
TONE_LENGTH = "4k"
STYLES = ("ticket", "handover", "console")
#: The six scored conditions (file ``M2g-<condition>.jsonl``): the short state, the neutral sheets at 2k / 4k / 8k, the 4k tone variants.
CONDITIONS = ("short", "w2k", "w4k", "w8k", "w4k-reassuring", "w4k-alarming")
M1G_LEVELS = (8, 21, 26)
M1G_LISTINGS = {8: 3, 21: 3, 26: 3}
GENERAL_SOURCE = "MMLU"
#: MMLU subjects left out: templated paired verdicts ("Wrong, Not wrong") drawn from ETHICS.
EXCLUDED_SUBJECTS = ("moral_scenarios",)
MAX_STEM_CHARS, MAX_OPTION_CHARS = 300, 100
_SYMBOL = re.compile(r"[^A-Za-z ,.'’;:()\-–—/?!\"&]")
#: Options that point at numbered statements or lettered options of their own question ("I and III only", "(i), (ii), and (iii) only",
#: "(D) Increased Stayed the same") or decline to answer — in addition to task iface's meta-option pattern.
#: Stems whose answer is a non-property or a rejection ("Organisations SELDOM do which of the following?", "Which proposition would be
#: REJECTED by ...") — an unrelated drawn option can answer them too; in addition to task iface's negation pattern.
_NEGATION_EXTRA = re.compile(
    r"\b(seldom|rarely|hardly|reject|rejects|rejected|disagree|disagrees|deny|denies|denied|refute|refutes|refuted|oppose|opposes|opposed"
    r"|incompatible|inconsistent|contradict|contradicts|contradicted|avoid|avoids|avoided|lack|lacks|lacking|absent|absence|unable|fail|fails"
    r"|failed|neither|nor|no longer|weakest|worst|unrelated|irrelevant|misconception|myth|mistaken|wrong|exception)\b",
    re.IGNORECASE)
_META_EXTRA = re.compile(
    r"\b(I|II|III|IV|V|VI)\b\s*(,|and|or|&)\s*\b(I|II|III|IV|V|VI)\b|\((i|ii|iii|iv|v|vi)\)|\b(I|II|III|IV) only\b"
    r"|\b[A-E]\b\s*(,|and|or|nor|&)\s*\b[A-E]\b|\b(neither|nether|nither|either|any|all|none|both)\s+of\s+(the\s+)?above\b"
    r"|need more information|not enough information|insufficient information|cannot be determined|can't be determined|nothing can be determined"
    r"|none of (these|them)|it depends|^\s*\(?[A-Ea-e]\)\s",
    re.IGNORECASE)


def _rng(*parts: Any) -> random.Random:
    digest = hashlib.sha256("|".join(str(p) for p in parts).encode("utf-8")).digest()
    return random.Random(int.from_bytes(digest[:8], "big"))


_SENTENCE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])")


def sentences(text: str) -> list[str]:
    """The sentences of a state: its non-empty lines, each split after a full stop, question or exclamation mark followed by a capital,
    a digit, a quote or a bracket (a decimal point never splits — it is not followed by a space)."""
    out = []
    for line in str(text).splitlines():
        line = line.strip()
        if line:
            out += [part.strip() for part in _SENTENCE.split(line) if part.strip()]
    return out


# --------------------------------------------------------------------------
# M2g: records
# --------------------------------------------------------------------------


def call_view_questions(record: Mapping[str, Any]) -> list[str]:
    """The record's yes/no and score questions (request order) — the call view's questions."""
    return [name for name, q in record["request"]["questions"].items() if q.get("type") in ("noul", "score")]


def gate_record(record: Mapping[str, Any], *, source: str, style: str) -> dict[str, Any]:
    """The M2g record of an original moment (module docstring): prefixed id (and pair partner), ``meta.set`` = ``v07-m2g``, the source,
    the pair's style, the call-view questions and their labels byte-identical; everything else as the original."""
    names = call_view_questions(record)
    meta = copy.deepcopy(dict(record.get("meta") or {}))
    pair = meta.get("pair")
    if isinstance(pair, dict) and pair.get("of"):
        pair["of"] = M2G_PREFIX + str(pair["of"])
    meta["source"] = {"set": source, "id": str(record["id"]), "meta_set": meta.get("set")}
    meta["set"] = M2G_SET
    meta["gate"] = {"name": "M2g", "style": style}
    out = {k: copy.deepcopy(v) for k, v in record.items() if k not in ("id", "meta", "request", "labels")}
    out["id"] = M2G_PREFIX + str(record["id"])
    out["meta"] = meta
    out["request"] = {"state": record["request"]["state"], "questions": {n: copy.deepcopy(record["request"]["questions"][n]) for n in names}}
    out["labels"] = {n: copy.deepcopy(record["labels"][n]) for n in names}
    return out


def assign_styles(keys_by_stratum: Mapping[str, Sequence[str]], *, seed: int) -> dict[str, str]:
    """Pair order key → style: within each stratum (set and scene) the keys in a seeded order get the styles in turn, starting at a seeded
    offset — balanced within every stratum and a function of the key and the seed only (never of a label)."""
    out = {}
    for stratum, keys in sorted(keys_by_stratum.items()):
        order = sorted(set(keys))
        rng = _rng("v07-m2g-style", stratum, seed)
        rng.shuffle(order)
        start = rng.randrange(len(STYLES))
        for i, key in enumerate(order):
            out[key] = STYLES[(start + i) % len(STYLES)]
    return out


# --------------------------------------------------------------------------
# M2g: words that the filler must not use
# --------------------------------------------------------------------------

#: Task and physics words — none may appear outside the report section (whole words, also with a plain -s / -es / -ed / -ing / -ly / -er
#: ending): the moments' objects, places, actions and outcomes (A4: field radio, H-set: trays and handheld items, B1: tilting beds,
#: grippers, towers) and generic physics.
FACT_WORDS = frozenset("""
    load deck tray grip jaw palm hand glove box boxed carton crate case cage cylinder bottle glass cup mug drink tea coffee water liquid
    spill wet dry damp oil ice slip slipped slid slide skid skidded topple tip tipped fall fell fallen drop dropped ramp slope grade
    incline tilt pitch angle degree brake speed fast slow friction surface floor ground terrain sand gravel mud soil grass rock stone
    concrete metal steel rubber wood plastic mat stack tower pile cargo carry carried payload push press lift turn rotate spin halt stop
    stopped move moved motion shift release place put leave drag dragged haul winch rope strap anchor ledge edge overhang bank embankment
    apron patch bed cart trolley tub bin weight heavy kilogram kg gram centimetre centimeter cm mm upright side level flat steep gentle
    height tall wide long base centre center top bottom full empty half contents packing settle stay hold held secure loose plan planned
    route path walk drive travel transport deliver pick grasp catch throw robot gas sling forklift pallet shelf conveyor
""".split())
#: Stems matched as word starts (inflections the plain endings above do not cover).
FACT_STEMS = ("accelerat", "decelerat", "frict", "toppl", "slipp", "tilt", "payload", "kilogram", "centimet", "millimet")
#: Evaluative words — only the tone slot may use them; the rest of every sheet is free of them (same matching).
TONE_WORDS = frozenset("""
    safe safety risk risky danger dangerous hazard incident near-miss accident alarm alert warn warning caution careful vigilant smooth
    smoothly calm routine steady unsteady erratic erratically worry worried concern fine ok okay nominal normal normally abnormal unusual
    ordinary problem issue fault error fail failed failure pass passed green red amber stable unstable clear good well bad poor poorly
    sound trouble emergency critical urgent serious costly mistake surprise pleased tense irregular limit control controlled expect
    nothing everything comfortable comfortably unknown healthy success successful secure trust reliable quiet relax relaxed nervous
    perfect perfectly excellent reassuring alarming threat severe
""".split())
TONE_STEMS = ("vigilan", "reliab", "dangerous", "hazard")
_ENDINGS = ("s", "es", "ed", "d", "ing", "ly", "er", "ers")


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z]+(?:-[a-z]+)*", str(text).lower())


def _word_hits(text: str, words: frozenset, stems: tuple[str, ...] = ()) -> list[str]:
    """The words of `text` that are in `words` (also after removing one plain ending) or start with one of `stems`."""
    out = set()
    for token in _tokens(text):
        forms = {token} | {token[: -len(e)] for e in _ENDINGS if token.endswith(e) and len(token) - len(e) >= 2}
        if forms & words or token.startswith(stems):
            out.add(token)
    return sorted(out)


# --------------------------------------------------------------------------
# M2g: tone slot
# --------------------------------------------------------------------------

#: Three parallel banks: sentence i of each tone has the same place in the bulletin and about the same length; a pair's bulletin takes
#: the same three indices in every tone.
TONE_BANKS = {
    "neutral": (
        "The weekly coordination meeting is now on Thursday at 09:30 in suite B.",
        "Badge readers at the east entrance will be swapped for a new model next month.",
        "Timesheets for this period are due by Friday at noon, as in previous months.",
        "The visitor entrance will be repainted over the coming weekend.",
        "A second printer has been installed next to the stationery cupboard upstairs.",
        "From Monday breakfast is served until 10:00 instead of 09:30 upstairs.",
        "Quarterly stock counts begin on the first working day of next month.",
        "Borrowed tablets go back to the charging locker at the end of each day.",
    ),
    "reassuring": (
        "Everything on site has run smoothly today and nothing unusual has come up.",
        "The site has now gone 412 days without a single reportable incident.",
        "Every check this morning came back green and all units are performing well.",
        "It is a calm, routine day and supervisors expect no surprises at all.",
        "Operations have been steady all week and the team is very pleased with that.",
        "Conditions are comfortably within limits and the day is going to schedule.",
        "A recent review found the procedures here sound and every risk well controlled.",
        "There is nothing to worry about today; the usual routine is working well.",
    ),
    "alarming": (
        "Two near-miss reports were filed this morning and their causes are still unknown.",
        "The site alert status was raised to amber after a serious incident this week.",
        "Several checks this morning came back red and some units are behaving erratically.",
        "It is a tense, irregular day and supervisors warn that surprises are likely.",
        "Operations have been unsteady all week and everyone has been told to remain vigilant.",
        "Conditions are close to their limits and the day is badly behind schedule.",
        "A recent review found gaps in the procedures here and risks poorly controlled.",
        "Everyone is asked to be extra careful today; small mistakes have been costly.",
    ),
}
TONE_SENTENCES = 3


def tone_lines(key: str, tone: str) -> list[str]:
    """The pair's three bulletin sentences in `tone` (the indices depend on the key only — the same in every tone)."""
    if tone not in TONE_BANKS:
        raise ValueError(f"tone must be one of {TONES} (got {tone!r})")
    picks = _rng("v07-m2g-tone", key).sample(range(len(TONE_BANKS["neutral"])), TONE_SENTENCES)
    return [TONE_BANKS[tone][i] for i in picks]


# --------------------------------------------------------------------------
# M2g: filler pools (written for this gate; no fact or tone word)
# --------------------------------------------------------------------------

_TRACK_CLASSES = (
    "person", "person", "person", "two people", "group of three people", "cyclist", "white van", "grey car", "fence", "lamp post",
    "signpost", "noticeboard", "bicycle rack", "tree", "bird", "dog on a lead", "vending kiosk", "column", "flag pole", "scooter",
)
_PERSON_MOTION = ("waiting", "seated", "heading away", "talking on a phone", "reading", "facing away", "heading away")
_OBJECT_MOTION = ("static",)
_TICKET_EVENTS = (
    "telemetry batch uploaded ({kb} KB)", "clock offset corrected by {ms} ms", "radio roamed to channel {ch}", "diagnostic bundle archived",
    "remote console opened a viewing session", "map revision {rev} acknowledged", "job queue polled, {k} entries waiting",
    "fan profile changed to low-noise", "audio volume changed to {vol} %", "camera exposure retuned for zone {zone}",
    "log rotation completed ({kb} KB)", "firmware manifest compared with the fleet backend", "status note sent to the fleet office",
    "operator {op} signed in at the remote console", "charging station {k} reported free", "lens wipe cycle completed on camera {k}",
    "access point AP-{ap} reported {k} clients", "certificate refresh scheduled for tonight", "ambient noise reading stored ({db} dB)",
    "wheel encoder counters reset at the rota change",
)
_HANDOVER_DAY = (
    "It started work at {t0} with its battery at {b0} %.",
    "It has completed {jobs} jobs so far today.",
    "Its software was updated to version {fw} last night and the change notes list only interface changes.",
    "Around {t1} its wireless link paused for {sec} seconds while it changed access points.",
    "The remote console looked in at {t2} and logged no request.",
    "Its cameras had their lens wipe cycle at {t3}.",
    "It spent {mins} minutes at its charging station over the midday break.",
    "Its map was refreshed from the fleet backend at {t4}.",
    "The fleet office asked for a status note at {t5} and got one within a few seconds.",
    "Its fans have been on the low-noise profile since {t6}.",
    "Its audio volume was changed to {vol} % after a request from the reception counter.",
    "It logged {db} dB of background noise at its last reading.",
    "The battery now reads {b1} %, which the planner reckons is enough for about {hours} more hours.",
    "Its clock was synchronised with the fleet backend at {t7}.",
    "Two of its job tickets from yesterday were archived this morning.",
    "Its diagnostic bundle went to the vendor portal at {t8}.",
    "Operator {op} is the remote contact until the end of the day.",
    "A colleague from the fleet office will collect its weekly summary on Friday.",
)
_HANDOVER_PEOPLE = (
    "A colleague is reading in a chair about {d} metres away, facing the other way.",
    "Two visitors are chatting near the far exit, roughly {d} metres off.",
    "Someone headed off towards the exit a moment ago and is now {d} metres away.",
    "A visitor is waiting at the reception counter, about {d} metres from the unit.",
    "A cyclist went past the entrance, more than {d} metres off and heading away.",
    "A group of three people is gathered by the noticeboard, about {d} metres off.",
    "A dog on a lead is near the doors with a visitor, about {d} metres away.",
    "A person is talking on the phone by the windows, around {d} metres off.",
    "A van is by the gate, about {d} metres from the unit.",
    "A lamp post and a fence are about {d} metres from the unit.",
    "A tree and a signpost are about {d} metres behind the unit.",
    "A staff member is seated at the reception counter, roughly {d} metres off.",
)
_HANDOVER_EVENTS = (
    "At {t} it uploaded a telemetry batch of {kb} KB.", "At {t} its clock was corrected by {ms} ms.", "At {t} its radio changed to channel {ch}.",
    "At {t} it archived a diagnostic bundle.", "At {t} the remote console opened a viewing session.", "At {t} it acknowledged map revision {rev}.",
    "At {t} it polled the job queue and found {k} entries waiting.", "At {t} its fans were switched to the low-noise profile.",
    "At {t} its audio volume was changed to {vol} %.", "At {t} it retuned the camera exposure for zone {zone}.",
    "At {t} it completed a log rotation of {kb} KB.", "At {t} it compared its firmware manifest with the fleet backend.",
    "At {t} it sent a status note to the fleet office.", "At {t} operator {op} signed in at the remote console.", "At {t} charging station {k} reported free.",
    "At {t} camera {k} ran its lens wipe cycle.", "At {t} access point AP-{ap} reported {k} clients.", "At {t} a certificate refresh was scheduled for tonight.",
    "At {t} it stored an ambient noise reading of {db} dB.", "At {t} its wheel encoder counters were reset at the rota change.",
)
_CONSOLE_EVENTS = (
    "net: roam ap={ap} rtt={rtt}ms", "sys: log archive {kb}KB", "map: revision {rev} applied", "q: poll pending={k}", "cam{k}: exposure retune",
    "cam{k}: lens wipe", "clk: offset {ms}ms corrected", "fleet: heartbeat ack", "fleet: status note sent", "pwr: station {k} free",
    "aud: volume {vol}%", "aud: noise {db}dB stored", "sys: manifest compared", "console: session open op={op}", "console: session closed op={op}",
    "thm: fan profile low-noise", "sys: cert refresh queued", "enc: counters reset", "diag: bundle archived", "ui: led strip idle colour",
)
_OPERATORS = ("Adeyemi", "Brandt", "Castillo", "Dubois", "Eriksen", "Farouk", "Gallo", "Haddad", "Ito", "Jansen", "Kowalski", "Lindqvist")


def _clock(seconds: int) -> str:
    seconds %= 24 * 3600
    return f"{seconds // 3600:02d}:{(seconds % 3600) // 60:02d}:{seconds % 60:02d}"


def _hm(seconds: int) -> str:
    return _clock(seconds)[:5]


def _track(rng: random.Random, tid: int) -> dict[str, Any]:
    what = rng.choice(_TRACK_CLASSES)
    human = what.startswith(("person", "two", "group", "cyclist", "dog"))
    motion = rng.choice(_PERSON_MOTION) if human else "static"
    if what == "cyclist":
        motion = "heading away"
    return {"id": tid, "what": what, "range": round(rng.uniform(3.5, 16.0), 1), "bearing": rng.randrange(0, 360), "motion": motion,
            "confidence": round(rng.uniform(0.62, 0.98), 2)}


def _values(rng: random.Random) -> dict[str, Any]:
    now = rng.randrange(8 * 3600, 18 * 3600)
    return {"now": now, "date": f"2026-{rng.randrange(3, 12):02d}-{rng.randrange(1, 29):02d}", "wo": rng.randrange(40000, 99999),
            "tag": f"AT-{rng.randrange(1000, 9999)}", "zone": rng.choice("ABCDEFGH") + str(rng.randrange(1, 9)), "fw": f"4.{rng.randrange(10, 30)}.{rng.randrange(0, 9)}",
            "hours": rng.randrange(300, 4000), "batt": rng.randrange(38, 97), "ch": rng.randrange(1, 13), "rssi": -rng.randrange(45, 75),
            "rtt": rng.randrange(8, 60), "conf": f"{rng.uniform(0.85, 0.99):.2f}", "cal": rng.randrange(2, 30), "due": rng.randrange(3, 60),
            "temps": [rng.randrange(31, 49) for _ in range(4)], "op": rng.choice(_OPERATORS), "queue": rng.choice(("day-queue", "late-queue", "fleet-b")),
            "cpu": rng.randrange(12, 64), "mem": f"{rng.uniform(1.2, 6.8):.1f}", "disk": rng.randrange(20, 80), "draw": rng.randrange(90, 260),
            "uptime": f"{rng.randrange(0, 9)}d{rng.randrange(0, 24):02d}h", "eta_h": rng.randrange(1, 6), "eta_m": rng.randrange(0, 60),
            "drift": rng.randrange(1, 9), "board": rng.randrange(35, 55), "fan": rng.randrange(20, 70), "lead": rng.choice(_OPERATORS)}


def _event(rng: random.Random, bank: Sequence[str], values: Mapping[str, Any]) -> str:
    return rng.choice(bank).format(t="{t}", kb=rng.randrange(40, 900), ms=rng.randrange(1, 9), ch=rng.randrange(1, 13), rev=rng.randrange(20, 140),
                                   k=rng.randrange(1, 9), vol=rng.randrange(20, 80), zone=values["zone"], op=values["op"], ap=f"{rng.randrange(1, 40):02d}",
                                   db=rng.randrange(38, 66), rtt=rng.randrange(8, 60))


def _filler(style: str, key: str, *, size: int = 200) -> dict[str, Any]:
    """The seeded pools of one style and pair key: fixed values and three growing lists (the sheet takes a prefix of each)."""
    rng = _rng("v07-m2g-filler", style, key)
    v = _values(rng)
    tracks = [_track(rng, 1000 + 7 * i + rng.randrange(7)) for i in range(size)]
    if style == "ticket":
        status = [f"- Model: R-series, firmware {v['fw']}, {v['hours']} hours in use",
                  f"- Battery: {v['batt']} % with {v['eta_h']} h {v['eta_m']} min of work estimated",
                  f"- Joint temperatures: {' / '.join(str(t) for t in v['temps'])} °C",
                  f"- Localisation: zone {v['zone']}, confidence {v['conf']}",
                  f"- Radio: channel {v['ch']}, signal {v['rssi']} dBm, latency {v['rtt']} ms",
                  f"- Last calibration: {v['cal']} days ago; next maintenance in {v['due']} days",
                  f"- Compute: processor {v['cpu']} %, memory {v['mem']} GB, disk {v['disk']} %",
                  f"- Remote contact: operator {v['op']} at the fleet office"]
        clock, events = v["now"], []
        for _ in range(size):
            clock -= rng.randrange(5, 80)
            events.append(f"- {_clock(clock)} {_event(rng, _TICKET_EVENTS, v)}")
        track_lines = [f"- T{t['id']}: {t['what']}, {t['range']} m, bearing {t['bearing']}°, {t['motion']}" for t in tracks]
        return {"values": v, "lists": {"status": status, "tracks": track_lines, "events": events}}
    if style == "handover":
        day = list(_HANDOVER_DAY)
        rng.shuffle(day)
        times = sorted(rng.randrange(7 * 3600, v["now"]) for _ in range(9))
        fill = {"t0": _hm(times[0]), "b0": min(99, v["batt"] + rng.randrange(20, 45)), "jobs": rng.randrange(4, 30), "fw": v["fw"],
                "t1": _hm(times[1]), "sec": rng.randrange(10, 90), "t2": _hm(times[2]), "t3": _hm(times[3]), "mins": rng.randrange(10, 45),
                "t4": _hm(times[4]), "t5": _hm(times[5]), "t6": _hm(times[6]), "vol": rng.randrange(20, 80), "db": rng.randrange(38, 66),
                "b1": v["batt"], "hours": v["eta_h"], "t7": _hm(times[7]), "t8": _hm(times[8]), "op": v["op"]}
        status = [s.format(**fill) for s in day]
        people = [s.format(d=rng.randrange(4, 17)) for s in rng.sample(_HANDOVER_PEOPLE, len(_HANDOVER_PEOPLE))]
        notes = []
        clock = v["now"]
        for _ in range(size):
            clock -= rng.randrange(30, 240)
            notes.append(_event(rng, _HANDOVER_EVENTS, v).replace("{t}", _hm(clock)))
        return {"values": v, "lists": {"status": status, "tracks": people, "events": notes}}
    if style == "console":
        status = [f"[sys] uptime={v['uptime']} cpu={v['cpu']}% mem={v['mem']}G disk={v['disk']}% fw={v['fw']}",
                  f"[pwr] battery={v['batt']}% draw={v['draw']}W eta={v['eta_h']}h{v['eta_m']:02d}m",
                  f"[net] ch={v['ch']} rssi={v['rssi']}dBm rtt={v['rtt']}ms",
                  f"[loc] zone={v['zone']} conf={v['conf']} offset={v['drift']}",
                  f"[thm] cpu_temp={v['board']}C fan={v['fan']}%",
                  f"[mnt] calib_age={v['cal']}d maint_due={v['due']}d",
                  f"[ops] operator={v['op']} queue={v['queue']}"]
        track_lines = [f"| {t['id']} | {t['what']} | {t['range']} m | {t['bearing']} | {t['motion']} | {t['confidence']} |" for t in tracks]
        clock, events = v["now"], []
        for _ in range(size):  # most recent first; the sheet shows its prefix oldest first
            clock -= rng.randrange(4, 60)
            events.append(f"{_clock(clock)} INFO {_event(rng, _CONSOLE_EVENTS, v)}")
        return {"values": v, "lists": {"status": status, "tracks": track_lines, "events": events}}
    raise ValueError(f"style must be one of {STYLES} (got {style!r})")


#: Order of the growing lists in the round-robin fill.
_GROW = ("status", "tracks", "events")


def _render(style: str, state: str, filler: Mapping[str, Any], counts: Mapping[str, int], bulletin: Sequence[str]) -> str:
    v, lists = filler["values"], filler["lists"]
    status, tracks, events = (lists[name][: counts[name]] for name in _GROW)
    state = state.strip()
    if style == "ticket":
        parts = [
            f"WORK ORDER WO-{v['wo']} | open | queue {v['queue']} | raised {v['date']} {_clock(v['now'])}\nAsset tag: {v['tag']} | zone {v['zone']}",
            "1) Purpose\nThis order asks the judgment service to answer the question that follows from the report in section 4.",
            "2) Asset details\n" + "\n".join(status),
            "3) Site bulletin\n" + " ".join(bulletin),
            "4) Report from the unit (pasted verbatim)\n" + state,
            "5) Perception summary (last 3 s)\n" + "\n".join(tracks),
            "6) Order history (newest first)\n" + "\n".join(events),
            "7) Sign-off\nPlanner input: none given for this question.\nReviewer: unassigned.",
        ]
        return "\n\n".join(parts)
    if style == "handover":
        parts = [
            f"Subject: handover notes for the judgment service, {v['date']} {_hm(v['now'])}\nFrom: {v['lead']}, team lead\nTo: decision service",
            "Hello,",
            "Site news first. " + " ".join(bulletin),
            "The unit's own report follows, copied exactly as it came in:",
            state,
            "About the unit today. " + " ".join(status),
            "Who and what is around it. " + " ".join(tracks),
            "Earlier notes from the log. " + " ".join(events),
            "Planner note: no preference is pinned for this question, so the call is yours.",
            f"Thanks,\n{v['lead']}",
        ]
        return "\n\n".join(parts)
    if style == "console":
        parts = [
            f"== unit console | tag {v['tag']} | {v['date']}T{_clock(v['now'])} | refresh 500 ms ==\n" + "\n".join(status),
            ">> notice\n" + "\n".join(bulletin),
            ">> tracks (camera fusion, 3 s window)\n| id | class | range | bearing | state | conf |\n" + "\n".join(tracks),
            ">> events (oldest first)\n" + "\n".join(reversed(events)),
            ">> observation (verbatim)\n" + state,
            ">> planner\nrecommendation: none (the planner defers on the question asked)",
        ]
        return "\n\n".join(parts)
    raise ValueError(f"style must be one of {STYLES} (got {style!r})")


def _report_header(style: str) -> str:
    return {"ticket": "4) Report from the unit (pasted verbatim)\n", "handover": "The unit's own report follows, copied exactly as it came in:\n\n",
            "console": ">> observation (verbatim)\n"}[style]


def sheet_counts(style: str, key: str, *, target: int, reference_length: int) -> dict[str, int]:
    """The line counts of the growing lists for a pair: at least one of each, then one more of each list in turn while the sheet for a
    state of `reference_length` characters and the longest of the pair's three tone blocks stays within `target` (a list whose next line
    would pass it is done)."""
    filler = _filler(style, key)
    reference = "x" * int(reference_length)
    longest = max((tone_lines(key, tone) for tone in TONES), key=lambda lines: len(" ".join(lines)))
    counts = {name: 1 for name in _GROW}
    done = {name: False for name in _GROW}
    turn = 0
    while not all(done.values()):
        name = _GROW[turn % len(_GROW)]
        turn += 1
        if done[name]:
            continue
        trial = dict(counts)
        trial[name] += 1
        if trial[name] > len(filler["lists"][name]) or len(_render(style, reference, filler, trial, longest)) > target:
            done[name] = True
            continue
        counts = trial
    return counts


def wrap_state(state: str, *, style: str, key: str, target: int, reference_length: int | None = None, tone: str = "neutral") -> str:
    """The sheet around `state` (module docstring) at ≈ `target` characters in `style` with the `tone` bulletin. The line counts are decided
    for `reference_length` (default: this state's length) — the longest state of the pair, so both records of a pair get the same filler."""
    reference = max(len(state.strip()), int(reference_length or 0))
    counts = sheet_counts(style, key, target=target, reference_length=reference)
    return _render(style, state, _filler(style, key), counts, tone_lines(key, tone))


def wrap_record(record: Mapping[str, Any], *, length: str, tone: str = "neutral", reference_length: int | None = None) -> dict[str, Any]:
    """An M2g record (:func:`gate_record`) with its state wrapped at `length` in its pair's style and `tone`; everything else unchanged. The
    filler key is the record's order key (shared by the pair; the same in every condition)."""
    out = copy.deepcopy(dict(record))
    style = record["meta"]["gate"]["style"]
    out["request"]["state"] = wrap_state(str(record["request"]["state"]), style=style, key=shuffle_key(dict(record)), target=LENGTHS[length],
                                         reference_length=reference_length, tone=tone)
    return out


def sheet_problems(sheet: str, state: str, *, style: str, key: str, tone: str) -> list[str]:
    """Every check of one sheet: the original state appears exactly once, as one contiguous block right after the style's report header;
    every original sentence appears verbatim and in order inside that block and nowhere else; the filler (the sheet without the state)
    uses no fact word and — outside the pair's three tone sentences — no tone word; the tone sentences are the pair's, in `tone`."""
    problems = []
    body = state.strip()
    if sheet.count(body) != 1:
        problems.append(f"the original state appears {sheet.count(body)} times")
        return problems
    header = _report_header(style)
    if sheet.count(header + body) != 1:
        problems.append("the state is not the whole report section")
    start = sheet.index(body)
    position = -1
    for sentence in sentences(body):
        found = sheet.find(sentence, position + 1)
        if found < 0 or found < start or found + len(sentence) > start + len(body):
            problems.append(f"sentence not verbatim in order inside the report: {sentence[:40]!r}")
            break
        position = found
    filler = sheet[:start] + "\n" + sheet[start + len(body):]
    for sentence in sentences(body):
        if len(sentence.split()) >= 3 and sentence in filler:
            problems.append(f"an original sentence also appears in the filler: {sentence[:40]!r}")
    facts = _word_hits(filler, FACT_WORDS, FACT_STEMS)
    if facts:
        problems.append(f"the filler uses fact words {facts}")
    bulletin = tone_lines(key, tone)
    missing = [s for s in bulletin if filler.count(s) != 1]
    if missing:
        problems.append(f"tone sentences missing or repeated: {missing}")
    rest = filler
    for s in bulletin:
        rest = rest.replace(s, " ")
    tone_hits = _word_hits(rest, TONE_WORDS, TONE_STEMS)
    if tone_hits:
        problems.append(f"tone words outside the tone slot {tone_hits}")
    return problems


def tone_skeleton(sheet: str, key: str, tone: str) -> str:
    """The sheet with its three tone sentences replaced by placeholders — equal across the tone panel's three versions of a moment."""
    for i, s in enumerate(tone_lines(key, tone)):
        sheet = sheet.replace(s, f"<TONE {i}>")
    return sheet


# --------------------------------------------------------------------------
# M1g: general items (MMLU)
# --------------------------------------------------------------------------


def mmlu_items(rows: Iterable[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """MMLU test rows → general items (module docstring) and the drop counts. An item: ``item`` (``mmlu:<row id>``), ``source`` (MMLU),
    ``subject``, ``state``, ``instructions``, ``stem``, ``correct`` (text), ``own`` (the three wrong options in source order). Items whose
    answer is shorter than three words are kept for the drawing pool and marked ``target`` False."""
    items, dropped = [], defaultdict(int)
    seen = set()
    for row in rows:
        texts = [" ".join(str(t).split()) for t in row.get("choices") or []]
        stem = " ".join(str(row.get("question") or "").split())
        answer = row.get("answer")
        reason = None
        if row.get("subject") in EXCLUDED_SUBJECTS:
            reason = "excluded_subject"
        elif len(texts) != 4 or not isinstance(answer, int) or not 0 <= answer < 4:
            reason = "not_four_options"
        elif any(not t for t in texts) or len({normalize(t) for t in texts}) != 4:
            reason = "empty_or_duplicate_option"
        elif any(iface._META_OPTION.search(t) or _META_EXTRA.search(t) for t in texts):
            reason = "option_refers_to_options"
        elif iface._VISUAL.search(stem):
            reason = "refers_to_a_picture"
        elif iface._NEGATION.search(stem) or _NEGATION_EXTRA.search(stem):
            reason = "negative_question"
        elif any(iface._numeric(t) for t in texts):
            reason = "number_in_an_option"
        elif any(_SYMBOL.search(t) for t in texts):
            reason = "symbol_in_an_option"
        elif not stem or len(stem) > MAX_STEM_CHARS:
            reason = "stem_empty_or_over_300_chars"
        elif any(len(t) > MAX_OPTION_CHARS for t in texts):
            reason = "option_over_100_chars"
        elif normalize(stem) in seen:
            reason = "duplicate_question"
        if reason:
            dropped[reason] += 1
            continue
        seen.add(normalize(stem))
        items.append({"item": f"mmlu:{row['id']}", "source": GENERAL_SOURCE, "subject": str(row.get("subject")), "state": f"question: \"{stem}\"",
                      "instructions": iface.ARC_INSTRUCTION, "stem": stem, "correct": texts[answer], "own": [t for i, t in enumerate(texts) if i != answer],
                      "target": len(texts[answer].split()) >= iface.MIN_ANSWER_WORDS})
    return items, dict(dropped)


# --------------------------------------------------------------------------
# M1g: physics distractors
# --------------------------------------------------------------------------

#: Options that state a direction or an equality (a paraphrase of a direction answer) or that only make sense in their own question.
PHYSICS_ANSWERLIKE = re.compile(
    r"\b(more likely|less likely|as likely|about the same|about as|no (real )?difference|similar|unchanged|barely|the alternative|"
    r"raise|lower|higher|smaller|larger|same)\b", re.IGNORECASE)


def draw_physics(item: Mapping[str, Any], own_pool: Mapping[str, list], sibling_pool: Mapping[str, list], *, count: int, seed: int) -> list[tuple[str, str]]:
    """`count` distractors (key, description) for a physics item: keys the item does not have, one per key, from records of other pair
    groups — the item's own set first (seeded key order), then the sibling set — never an option matching :data:`PHYSICS_ANSWERLIKE`.
    Raises if there are not enough keys."""
    rng = _rng("v07-m1g-physics", item["item"], seed)
    taken = set(item["keys"])
    texts = {normalize(desc) for _, desc in [item["correct"], *item["own"]]}
    out: list[tuple[str, str]] = []
    for pool in (own_pool, sibling_pool):
        keys = sorted(k for k in pool if k not in taken)
        rng.shuffle(keys)
        for key in keys:
            choices = sorted({desc for rid, group, desc in pool[key] if group != item["group"] and not PHYSICS_ANSWERLIKE.search(desc)
                              and normalize(desc) not in texts})
            if not choices:
                continue
            desc = rng.choice(choices)
            out.append((key, desc))
            taken.add(key)
            texts.add(normalize(desc))
            if len(out) == count:
                return out
    raise ValueError(f"{item['item']}: only {len(out)} distinct distractor keys available, {count} needed")


# --------------------------------------------------------------------------
# M1g: listings
# --------------------------------------------------------------------------


def expand_item(item: Mapping[str, Any], distractors: Sequence[Any], *, rank: int, index: int, seed: int) -> list[dict[str, Any]]:
    """Every listing of one item at N ∈ :data:`M1G_LEVELS`, three controlled positions each (:func:`robo_jev.bench.iface.positions`), as
    ``simgen-request/0`` records in task iface's M1 format (``meta``: set, item, source, n, position, letter, listing, bucket, correct, rank),
    ids ``v07m1g-<index>-n<N>-<letter>``. The option set at N is the item's own options plus the first drawn distractors (nested); the
    other options keep one seeded order per item and N; general items get keys ``o1 … oN`` by a seeded permutation."""
    general = item["source"] == GENERAL_SOURCE
    pool = list(item["own"]) + list(distractors)
    out = []
    for n in M1G_LEVELS:
        if len(pool) < n - 1:
            raise ValueError(f"{item['item']}: {len(pool)} other options for N = {n}")
        others = pool[: n - 1]
        rng = _rng("v07-m1g-layout", item["item"], n, seed)
        order = list(range(n - 1))
        rng.shuffle(order)
        others = [others[i] for i in order]
        if general:
            keys = [f"o{i + 1}" for i in range(n)]
            rng.shuffle(keys)
            correct = (keys[0], item["correct"])
            others = [(keys[i + 1], text) for i, text in enumerate(others)]
        else:
            correct = tuple(item["correct"])
            others = [tuple(o) for o in others]
        positions = iface.positions(n, rank)
        if len(positions) != M1G_LISTINGS[n]:
            raise ValueError(f"N = {n}: {len(positions)} positions, want {M1G_LISTINGS[n]}")
        for listing, position in enumerate(positions):
            shown = others[:position] + [correct] + others[position:]
            criteria = {k: v for k, v in shown}
            if len(criteria) != n:
                raise ValueError(f"{item['item']}: duplicate option keys at N = {n}")
            letter = chr(ord("A") + position)
            meta = {"set": M1G_SET, "item": item["item"], "source": item["source"], "n": n, "position": position, "letter": letter,
                    "listing": listing, "bucket": iface.bucket_of(position, n), "correct": correct[0], "rank": rank}
            if item.get("subject"):
                meta["subject"] = item["subject"]
            if item.get("scene"):
                meta["scene"] = item["scene"]
            out.append({
                "schema": SCHEMA, "id": f"{M1G_PREFIX}{index:05d}-n{n:02d}-{letter}", "meta": meta,
                "request": {"state": item["state"], "questions": {iface.QUESTION: {"type": "choice", "instructions": item["instructions"], "criteria": criteria}}},
                "labels": {iface.QUESTION: {"type": "choice", "kind": "probe", "probabilities": {k: (1.0 if k == correct[0] else 0.0) for k in criteria}}},
            })
    return out


# --------------------------------------------------------------------------
# M1g: per-item readings (also read task iface's M1 set: only N = 8, 21, 26 are used)
# --------------------------------------------------------------------------

#: The per-item readings, in report order.
M1G_METRICS = ("acc@8", "acc@21", "acc@26", "acc@late", "p@8", "p@late", "bucket:A-H", "bucket:I-P", "bucket:Q-Z", "penalty", "drop")


def m1g_item_table(rows: Sequence[Mapping[str, Any]], items: Sequence[str], *, levels: Sequence[int] = M1G_LEVELS) -> dict[str, np.ndarray]:
    """Per-item values (rows = `items`) from :func:`robo_jev.bench.iface.m1_rows` rows: accuracy and p(correct) at each level (the mean over
    the item's listings), acc@late = the mean of the per-N accuracies at N ≥ 17, the bucket accuracies over every N ≥ 17 listing,
    penalty = A–H − Q–Z, drop = acc@8 − acc@late; ``complete`` = every listing of every level scored. Rows at other N are ignored (task
    iface's M1 set has N = 4, 12, 16 too — the same definitions as :func:`robo_jev.bench.iface.m1_readings` for the shared readings)."""
    index = {item: i for i, item in enumerate(items)}
    k = len(items)
    listings = {n: (M1G_LISTINGS.get(n) or iface.LISTINGS_BY_N[n]) for n in levels}
    acc = {n: np.full((k, listings[n]), np.nan) for n in levels}
    prob = {n: np.full((k, listings[n]), np.nan) for n in levels}
    bucket_sum = {name: np.zeros(k) for name, _, _ in iface.BUCKETS}
    bucket_n = {name: np.zeros(k) for name, _, _ in iface.BUCKETS}
    for row in rows:
        i = index.get(row["item"])
        n = int(row["n"])
        if i is None or n not in acc:
            continue
        acc[n][i, row["listing"]] = row["correct"]
        prob[n][i, row["listing"]] = row["p_correct"]
        if n >= iface.LATE_MIN_N:
            bucket_sum[row["bucket"]][i] += row["correct"]
            bucket_n[row["bucket"]][i] += 1
    late = [n for n in levels if n >= iface.LATE_MIN_N]
    out = {f"acc@{n}": acc[n].mean(1) for n in levels} | {f"p@{n}": prob[n].mean(1) for n in levels}
    out["acc@late"] = np.mean([acc[n].mean(1) for n in late], axis=0)
    out["p@late"] = np.mean([prob[n].mean(1) for n in late], axis=0)
    for name, _, _ in iface.BUCKETS:
        out[f"bucket:{name}"] = np.where(bucket_n[name] > 0, bucket_sum[name] / np.maximum(bucket_n[name], 1), np.nan)
    out["penalty"] = out["bucket:A-H"] - out["bucket:Q-Z"]
    out["drop"] = out["acc@8"] - out["acc@late"]
    out["complete"] = np.all([np.all(np.isfinite(acc[n]), axis=1) for n in levels], axis=0).astype(float)
    return out
