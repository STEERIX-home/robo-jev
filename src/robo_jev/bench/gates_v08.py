"""Task v0.8b — the sealed v0.8 gates B8 (fresh bench-shaped requests) and M8g (many options up to 255), shared constants and pure parts.

The PRE-DECLARED v0.8 VERDICT RULE (progress.md, 2026-10-16 ~12:43 KST, before any v0.8 data existed) reads two gates built by an author who
never sees the v0.8 training input or its code (task v0.8a). This module holds what every part of task v0.8b shares — the set names and
prefixes, the files, the 255-option opt-in, the answer codes and the option-count bins — and the M8g build (many options). The B8 build
(robo-jev-web's own request builders under Node, option tags frozen before scoring) is :mod:`robo_jev.bench.b8_v08`. Nothing here loads a model.

**The 255-option opt-in.** Task v0.8a defines it: a record carries ``meta.option_range = {"version": <name>, "max": 255}``. Its version name
is the ONE constant :data:`OPTION_RANGE_VERSION` below (the controller relays v0.8a's name; until then it is a placeholder, flagged by
:data:`OPTION_RANGE_VERSION_CONFIRMED`). Every gate record takes the mark from :func:`option_range_mark`, so setting the constant (and the
flag) and rebuilding is the whole change — deterministic: ``scripts/v08_gates_build.py build-m8g`` then ``merge-jev``, and for B8
``scripts/v08_b8_build.py convert --force-before-scoring`` then ``tag-freeze --force-before-scoring`` (allowed only while no model has been
scored on B8; the new hashes are frozen again).

**Codes.** Options are shown with the codes in order — A–Z first, then the two-letter codes — exactly as at inference: the runner's
``CODES[:255]`` (:data:`CODES_255`, from :mod:`robo_jev.hf.robo_jev_decide`, the file the Hub carries).
"""

from __future__ import annotations

import hashlib
import math
import random
import re
from pathlib import Path
from typing import Any

from robo_jev.hf.robo_jev_decide import CODES as RUNNER_CODES

__all__ = [
    "ARTIFACTS", "B8_DIR", "B8_FILE", "B8_PREFIX", "B8_SET", "BINS", "CODES_255", "GATES", "JEV_DIR", "M8G_DIR", "M8G_FILE", "M8G_LEVELS",
    "M8G_PREFIX", "M8G_SET", "OPTION_RANGE_MAX", "OPTION_RANGE_VERSION", "OPTION_RANGE_VERSION_CONFIRMED", "OVERLAP_V08", "P2_TARGETS",
    "KNOWN_KEY_ERRORS", "PLANNED", "POSTHOC_DIR", "PROBES_DIR", "bin_of", "code_of", "draw_general_many", "expand_m8g", "gate_problems", "m8g_position",
    "option_range_mark", "physics_item", "refers_to_options", "rng", "two_letter",
]

REPO = Path(__file__).resolve().parents[3]
ARTIFACTS = REPO / "artifacts"

#: The sealed sets (registered in :mod:`robo_jev.data.sealed`): never trained on, never sent to any API.
B8_SET, B8_PREFIX = "v08-b8", "v08b8-"
M8G_SET, M8G_PREFIX = "v08-m8g", "v08m8g-"

GATES = ARTIFACTS / "v08" / "gates"
B8_DIR = GATES / "b8"
B8_FILE = B8_DIR / "B8.jsonl"
M8G_DIR = GATES / "m8g"
M8G_FILE = M8G_DIR / "M8g.jsonl"
#: ``scripts/v08_gates_build.py recheck-v08`` — gate items sharing text with v0.8's training input (counts and hashes only).
OVERLAP_V08 = GATES / "overlap-v08.json"
#: Not sealed (it is sent to OpenRouter): the Jev comparison set and Jev's stored answers.
JEV_DIR = ARTIFACTS / "v08" / "jev"
#: Reported only: the rel071 storage-bin probe cases.
PROBES_DIR = ARTIFACTS / "v08" / "probes"
#: Reported only: the post-hoc prior-removal estimate on v0.7's predictions.
POSTHOC_DIR = ARTIFACTS / "v08" / "posthoc"

#: The 255-option opt-in's version name (module docstring) — task v0.8a's, relayed by the controller on 2026-10-16 (~14:08 KST):
#: ``meta.option_range = {"version": "options-255-1", "max": 255}``.
OPTION_RANGE_VERSION = "options-255-1"
#: True once the relayed name is set above (the build records it; the Spark check refuses an unconfirmed name).
OPTION_RANGE_VERSION_CONFIRMED = True
OPTION_RANGE_MAX = 255

#: The answer codes in the order options are shown (A–Z, then the runner's two-letter codes).
CODES_255: tuple[str, ...] = tuple(RUNNER_CODES[:OPTION_RANGE_MAX])

#: Option-count bins of the rule's reported readings (inclusive bounds).
BINS: tuple[tuple[str, int, int], ...] = (("2-10", 2, 10), ("11-26", 11, 26), ("27-60", 27, 60), ("61-255", 61, 255))

#: M8g option counts: one per bin, two in the top bin.
M8G_LEVELS = (6, 18, 40, 120, 255)

#: P2's questions per environment (the rule: plates move, café intent / attend / request, delivery manoeuvre — pooled).
P2_TARGETS = {"plates": ("move",), "cafe": ("intent", "attend", "request"), "delivery": ("manoeuvre",)}
#: The question each environment's planner recommends for (café attend / request have no recommendation).
PLANNED = {"plates": "move", "cafe": "intent", "delivery": "manoeuvre", "tubes": "work"}


def option_range_mark() -> dict[str, Any]:
    """The 255-option opt-in every v0.8b gate record carries (``meta.option_range``)."""
    return {"version": OPTION_RANGE_VERSION, "max": OPTION_RANGE_MAX}


def code_of(position: int) -> str:
    """The code shown at a 0-based position (A–Z, then two-letter codes)."""
    return CODES_255[int(position)]


def two_letter(position: int) -> bool:
    """Whether the code at a 0-based position has two letters (position ≥ 26)."""
    return int(position) >= 26


def bin_of(n: int) -> str:
    """The rule's option-count bin of a question with `n` options."""
    for name, low, high in BINS:
        if low <= int(n) <= high:
            return name
    raise ValueError(f"{n} options: outside every bin (2-255)")


def rng(*parts: Any) -> random.Random:
    """A seeded generator keyed by `parts` (sha256 of their joined text)."""
    digest = hashlib.sha256("|".join(str(p) for p in parts).encode("utf-8")).digest()
    return random.Random(int.from_bytes(digest[:8], "big"))


# --------------------------------------------------------------------------
# M8g — many options (FORMAT): the design
# --------------------------------------------------------------------------
#
# * Items: MMLU test questions (MIT licence; filtered exactly as M1g's — :func:`robo_jev.bench.gates_v07.mmlu_items`) that are in neither M1g
#   nor any training input, plus fresh physics items written for this gate (:func:`physics_item` — four families of plausible same-family
#   candidates, exactly one of which meets every stated physical constraint).
# * N ∈ :data:`M8G_LEVELS`, the same item and answer at every N; the option set at N is the item's own options plus the first drawn
#   distractors (nested across N); the other options keep one seeded order per item and N.
# * Two listings per item and N (:func:`m8g_position`): at N ≤ 26 one in each half of the list; at N ≥ 40 listing 0 puts the correct option
#   on a single-letter code (A–Z) and listing 1 on a two-letter code, both spread over their band with the item's rank — so in bins 27–60
#   and 61–255 every item is read with both kinds of code (the FORMAT reading compares the two on the same items), and over the items the
#   correct position covers the whole list, A to the last code.
# * Codes are assigned in order at scoring (A–Z first, then :data:`CODES_255`): the records fix the shown order (scored with seed None).

GENERAL_SOURCE = "MMLU"
PHYSICS_SOURCE = "PHYS"
M8G_LISTINGS = 2
M8G_QUESTION = "q"
#: The first draws of a general item follow task iface's strict rule (no content word shared with the item's question and options or with
#: an earlier draw, one per source question); the rest only stay topic-distant from the item (no shared content word with its question and
#: options), one per source question — at N = 255 the strict rule runs out of words.
STRICT_DRAWS = 25
PHYSICS_FAMILIES = ("bins", "shelves", "spots", "doors")
#: Position strides (the first one coprime with the band width is used, so the ranks walk every position of the band).
POSITION_STRIDES = (11, 13, 17, 19, 23)


def m8g_position(n: int, rank: int, listing: int) -> int:
    """The 0-based position of the correct option (module design): N ≤ 26 — listing 0 in the first half, listing 1 in the second; N ≥ 27 —
    listing 0 on A–Z, listing 1 on a two-letter code; spread over the band with the rank (and shifted by the level)."""
    n, rank, listing = int(n), int(rank), int(listing)
    if listing not in (0, 1):
        raise ValueError(f"listing {listing}: 0 or 1")
    if n <= 26:
        half = (n + 1) // 2
        lo, hi = (0, half) if listing == 0 else (half, n)
    else:
        lo, hi = (0, 26) if listing == 0 else (26, n)
    level = M8G_LEVELS.index(n) if n in M8G_LEVELS else 0
    width = hi - lo
    stride = next(s for s in POSITION_STRIDES if math.gcd(s, width) == 1)  # coprime: consecutive ranks walk the whole band
    return lo + (rank * stride + level) % width


#: Options that point at other options without M1g's patterns catching them ("All the above", "both of these", "all are correct" — M1g's
#: filters need "of" before "above"): after shuffling into a long list such an option has no referent, so an item with one (key or not) is
#: neither measured nor drawn from (found by the M8g ambiguity audit, 2026-10-16: 13 of M1g's 5,072 candidates).
REFERS_TO_OPTIONS = re.compile(
    r"\b(all|none|both|neither|either|any)\s+(of\s+)?(the\s+)?(above|below|previous|preceding|following|other)\b"
    r"|\b(all|none|both)\s+(of\s+)?(these|those|them)\b|\b(all|none)\s+are\s+(correct|true|right)\b"
    r"|\b(above|below)\s+(options?|answers?|choices?|statements?)\b|\bboth\s+\(?[a-d]\)?\s+and\s+\(?[a-d]\)?\b", re.IGNORECASE)


#: MMLU items whose own key is wrong, found by the M8g ambiguity audit before sealing (2026-10-16) — never measured: ``mmlu-test-13841``
#: (virology: the key says Nipah came from humans; its reservoir is fruit bats — the source row's answer index is 2, so not a mapping slip).
KNOWN_KEY_ERRORS = ("mmlu:mmlu-test-13841",)


def refers_to_options(item: dict[str, Any]) -> bool:
    """Whether any option of a general item (the key or its own wrong options) points at other options (:data:`REFERS_TO_OPTIONS`)."""
    return any(REFERS_TO_OPTIONS.search(str(t)) for t in [item["correct"], *item["own"]])


def draw_general_many(item: dict[str, Any], index: list[dict[str, Any]], *, count: int, seed: int, strict: int = STRICT_DRAWS) -> list[str]:
    """`count` distractor texts for a general item, from the wrong options of other questions (`index` = :func:`robo_jev.bench.iface.general_index`
    of the drawing pool): the first `strict` follow task iface's rule (the other question's topic shares no content word with this item's
    topic or with earlier draws; the text neither); the rest only share no content word with this item's topic; one per source question,
    no repeated text, never the item's own options; candidates in a seeded order. Raises if there are not enough."""
    from robo_jev.bench import iface
    from robo_jev.data.general_sets import normalize

    r = rng("v08-m8g-general", item["item"], seed)
    topic = iface._topic(item)
    blocked = set(topic)
    taken = {normalize(t) for t in [item["correct"], *item["own"]]}
    used: set[str] = {item["item"]}
    order = list(range(len(index)))
    r.shuffle(order)
    out: list[str] = []
    for phase in ("strict", "loose"):
        for i in order:
            if len(out) >= (min(strict, count) if phase == "strict" else count):
                break
            other = index[i]
            if other["item"] in used or not other["wrong"] or other["topic"] & (blocked if phase == "strict" else topic):
                continue
            fence = blocked if phase == "strict" else topic
            found = [(t, tokens) for t, tokens in other["wrong"] if tokens and not tokens & fence and normalize(t) not in taken]
            if not found:
                continue
            text, tokens = found[r.randrange(len(found))]
            out.append(text)
            taken.add(normalize(text))
            used.add(other["item"])
            if phase == "strict":
                blocked |= tokens
    if len(out) < count:
        raise ValueError(f"{item['item']}: only {len(out)} general distractors found, {count} needed")
    return out


# --------------------------------------------------------------------------
# M8g — fresh physics items (written for this gate)
# --------------------------------------------------------------------------


def _pick(r: random.Random, values: Any) -> Any:
    values = list(values)
    return values[r.randrange(len(values))]


def _frange(lo: float, hi: float, step: float) -> list[float]:
    out, x = [], lo
    while x <= hi + 1e-9:
        out.append(round(x, 2))
        x += step
    return out


def _physics_spec(family: str, r: random.Random) -> dict[str, Any]:
    """One scenario of a family: the state, the instruction, the key prefix, how to write a candidate, and its constraints — each
    constraint as (name, ok values, violating values)."""
    if family == "bins":
        material = _pick(r, ("steel fittings", "ceramic tiles", "bottled reagents", "brass valves", "printed manuals", "glass jars", "motor parts", "cable reels"))
        m = _pick(r, (6, 8, 9, 11, 12, 14, 15, 18, 20, 22))
        w = _pick(r, (30, 34, 38, 40, 44, 48))
        d, h = w - _pick(r, (4, 6, 8, 10)), _pick(r, (18, 20, 24, 26, 30))
        reach = _pick(r, (1.6, 1.7, 1.8, 1.9, 2.0))
        state = (f"You are a mobile manipulator in a parts store, holding a closed box of {material} in both grippers.\n"
                 f"- The box weighs {m} kg and measures {w} x {d} x {h} cm; a bin's opening must be at least {w} cm wide for it to go in.\n"
                 f"- Your arm can set a box into a bin whose floor is at most {reach:.1f} m above the ground.\n"
                 "- Store it in a bin that is empty, in service and open, that holds its weight, that it fits into and that you can reach now.\n"
                 "- The aisle is clear; nobody is near you.")
        low = max(5, -(-int(1.5 * m) // 5) * 5)
        constraints = [("rating", list(range(low, low + 35, 5)), list(range(max(2, round(0.3 * m)), int(0.6 * m) + 1))),
                       ("opening", list(range(w + 8, w + 31, 2)), list(range(round(0.5 * w), int(0.85 * w) + 1, 2))),
                       ("height", _frange(0.3, reach - 0.4, 0.1), _frange(reach + 0.3, reach + 0.9, 0.1)),
                       ("status", ["empty with its door open"], ["full to the top", "with its door jammed shut", "marked out of service",
                                                                  "already holding a box"])]
        return {"state": state, "instructions": "Which bin should the robot put the box into?", "prefix": "bin", "noun": "Bin",
                "write": lambda v: f"rated for {v['rating']} kg, opening {v['opening']} cm wide, floor {v['height']:.1f} m above the ground, {v['status']}.",
                "constraints": constraints}
    if family == "shelves":
        contents = _pick(r, ("engine parts", "paint cans", "bagged cement", "water bottles", "machine tools", "battery packs"))
        m = _pick(r, (40, 60, 80, 90, 120, 150, 180))
        d = _pick(r, (60, 70, 80, 90, 100))
        lift = _pick(r, (2.0, 2.4, 2.8, 3.2))
        state = (f"You are a warehouse robot with a crate of {contents} on your forks.\n"
                 f"- The crate weighs {m} kg and is {d} cm deep.\n"
                 f"- Your forks lift a load no higher than {lift:.1f} m.\n"
                 "- The crate must rest on a shelf that carries its weight, is at least as deep as the crate, and is level and dry so it cannot slide off.")
        low = -(-int(1.4 * m) // 10) * 10
        constraints = [("load", list(range(low, low + 200, 10)), list(range(max(10, round(0.3 * m / 5) * 5), int(0.65 * m) + 1, 5))),
                       ("depth", list(range(d + 5, d + 41, 5)), list(range(d - 40, d - 11, 4))),
                       ("height", _frange(0.2, lift - 0.4, 0.2), _frange(lift + 0.4, lift + 1.6, 0.2)),
                       ("surface", ["level and dry"], ["tilted 8° toward the aisle", "slick with spilled hydraulic oil", "already stacked to the beam",
                                                       "cracked and taped off"])]
        return {"state": state, "instructions": "Which shelf should the robot set the crate on?", "prefix": "shelf", "noun": "Shelf",
                "write": lambda v: f"holds up to {v['load']} kg, {v['depth']} cm deep, {v['height']:.1f} m up, {v['surface']}.",
                "constraints": constraints}
    if family == "spots":
        cart = _pick(r, ("a loaded tool cart", "a cart of gas cylinders", "a laundry cart", "a parts trolley", "a cart of paint drums"))
        m = _pick(r, (120, 160, 200, 250, 300, 400))
        cw = _pick(r, (60, 70, 80, 90))
        smax = _pick(r, (3, 4, 5, 6))
        state = (f"You are towing {cart} that weighs {m} kg and is {cw} cm wide, and you must leave it parked while you fetch a pallet.\n"
                 f"- Its brakes hold it on a floor slope of at most {smax}°; on a wet or oily floor its wheels slide.\n"
                 "- Park it where it fits, where it stays put, and where the spot is free.")
        constraints = [("slope", list(range(0, smax - 1)), list(range(smax + 4, smax + 13))),
                       ("width", list(range(cw + 15, cw + 61, 5)), list(range(cw - 30, cw - 7, 4))),
                       ("surface", ["dry concrete", "dry rubber matting", "dry painted concrete"], ["freshly mopped wet tiles", "a film of spilled oil",
                                                                                                    "a puddle of spilled coolant"]),
                       ("free", ["free"], ["taken by a pallet jack", "taken by a parked forklift"])]
        return {"state": state, "instructions": "Which spot should the robot park the cart in?", "prefix": "spot", "noun": "Spot",
                "write": lambda v: f"floor slope {v['slope']}°, {v['width']} cm wide, {v['surface']}, {v['free']}.", "constraints": constraints}
    if family == "doors":
        load = _pick(r, ("a parcel cage", "a pallet of boxes", "a hospital bed", "a pallet of drinks", "a rack of garments"))
        w = _pick(r, (70, 80, 90, 100, 110))
        step = _pick(r, (2, 3, 4))
        state = (f"You are a delivery robot moving {load} that is {w} cm wide to the loading bay; every door listed leads there.\n"
                 f"- Your wheels climb a step of at most {step} cm.\n"
                 "- Take a door the load fits through and that you can pass now.")
        constraints = [("width", list(range(w + 10, w + 51, 5)), list(range(w - 30, w - 5, 4))),
                       ("threshold", list(range(0, step)), list(range(step + 3, step + 11))),
                       ("state", ["open"], ["locked", "blocked by stacked boxes on the far side", "closed for floor cleaning"])]
        return {"state": state, "instructions": "Which door should the robot take?", "prefix": "door", "noun": "Door",
                "write": lambda v: (f"{v['width']} cm clear width, " + ("flush threshold" if v["threshold"] == 0 else f"threshold {v['threshold']} cm high")
                                    + f", {v['state']}."), "constraints": constraints}
    raise ValueError(f"family {family!r}: one of {PHYSICS_FAMILIES}")


def physics_item(family: str, index: int, *, seed: int, count: int = OPTION_RANGE_MAX - 1, numbers: tuple[int, int] = (100, 1000)) -> dict[str, Any]:
    """A fresh physics item: a scenario of `family` and `count` + 1 candidates of the same kind — the correct one meets every constraint with
    a margin, every distractor breaks exactly one (the broken constraint cycles in a seeded order, so the first draws break different ones).
    Candidates are numbered with distinct numbers from `numbers` (keys ``<prefix>_<number>``; M8g uses three digits, the Jev side four, so
    their texts never coincide); ``pool`` is in draw order (nested across N)."""
    r = rng("v08-m8g-physics", family, index, seed)
    spec = _physics_spec(family, r)
    names = [c[0] for c in spec["constraints"]]
    numbers = r.sample(range(*numbers), count + 1)

    def candidate(number: int, broken: str | None) -> tuple[str, str]:
        values = {name: (_pick(r, bad) if name == broken else _pick(r, ok)) for name, ok, bad in spec["constraints"]}
        return f"{spec['prefix']}_{number}", f"{spec['noun']} {number}: " + spec["write"](values)

    correct = candidate(numbers[0], None)
    cycle = list(names)
    pool, broken = [], []
    for i in range(count):
        if i % len(cycle) == 0:
            r.shuffle(cycle)
        broken.append(cycle[i % len(cycle)])
        pool.append(candidate(numbers[i + 1], broken[-1]))
    texts = [correct[1]] + [t for _, t in pool]
    if len(set(texts)) != len(texts):  # the numbers differ, so texts never repeat; keep the check
        raise ValueError(f"{family}-{index}: repeated candidate text")
    return {"item": f"phys:{family}-{index:03d}", "source": PHYSICS_SOURCE, "family": family, "state": spec["state"],
            "instructions": spec["instructions"], "correct": correct, "pool": pool, "broken": broken, "constraints": names}


# --------------------------------------------------------------------------
# M8g — listings
# --------------------------------------------------------------------------


def expand_m8g(item: dict[str, Any], distractors: list[Any], *, rank: int, index: int, seed: int, levels: tuple[int, ...] = M8G_LEVELS,
               listings: int = M8G_LISTINGS, prefix: str = M8G_PREFIX, set_name: str = M8G_SET) -> list[dict[str, Any]]:
    """Every listing of one item at N ∈ `levels` (module design) as ``simgen-request/0`` records (question ``q``, one-hot label of kind
    ``probe``; ``meta``: set, item, source, family / subject, n, position, code, two_letter, listing, correct, rank, bin, option_range), ids
    ``<prefix><index>-n<N>-<code>``. General items: the item's three own wrong options, then the drawn texts; keys ``o1 … oN`` by a seeded
    permutation. Physics items: the drawn candidates (key, text)."""
    from robo_jev.decisions.records import SCHEMA

    general = item["source"] == GENERAL_SOURCE
    pool = (list(item["own"]) + list(distractors)) if general else list(distractors)
    out = []
    for n in levels:
        if len(pool) < n - 1:
            raise ValueError(f"{item['item']}: {len(pool)} other options for N = {n}")
        others = pool[: n - 1]
        r = rng("v08-m8g-layout", item["item"], n, seed)
        order = list(range(n - 1))
        r.shuffle(order)
        others = [others[i] for i in order]
        if general:
            keys = [f"o{i + 1}" for i in range(n)]
            r.shuffle(keys)
            correct = (keys[0], item["correct"])
            others = [(keys[i + 1], text) for i, text in enumerate(others)]
        else:
            correct = tuple(item["correct"])
            others = [tuple(o) for o in others]
        for listing in range(listings):
            position = m8g_position(n, rank, listing) if listings == M8G_LISTINGS else m8g_position(n, rank, rank % 2)
            shown = others[:position] + [correct] + others[position:]
            criteria = {k: v for k, v in shown}
            if len(criteria) != n:
                raise ValueError(f"{item['item']}: duplicate option keys at N = {n}")
            code = code_of(position)
            meta = {"set": set_name, "item": item["item"], "source": item["source"], "n": n, "position": position, "code": code,
                    "two_letter": two_letter(position), "listing": listing, "correct": correct[0], "rank": rank, "bin": bin_of(n),
                    "option_range": option_range_mark()}
            for extra in ("subject", "family"):
                if item.get(extra):
                    meta[extra] = item[extra]
            out.append({
                "schema": SCHEMA, "id": f"{prefix}{index:05d}-n{n:03d}-{code}", "meta": meta,
                "request": {"state": item["state"], "questions": {M8G_QUESTION: {"type": "choice", "instructions": item["instructions"], "criteria": criteria}}},
                "labels": {M8G_QUESTION: {"type": "choice", "kind": "probe", "probabilities": {k: (1.0 if k == correct[0] else 0.0) for k in criteria}}},
            })
    return out


def gate_problems(record: dict[str, Any]) -> list[str]:
    """:func:`robo_jev.decisions.records.validate_record` with the option range read from this task's opt-in: a record must carry
    :func:`option_range_mark` exactly and its choice questions 2–255 options; every other check stays. (Main before task v0.8a does not know
    the 255 mark, so the range problems ``validate_record`` raises for it are replaced by this check — nothing else is lifted.)"""
    from robo_jev.decisions.records import validate_record

    mark = (record.get("meta") or {}).get("option_range")
    problems = [p for p in validate_record(record, require_labels=bool(record.get("labels")))
                if not p.startswith("meta.option_range") and "choice needs" not in p]
    if mark != option_range_mark():
        problems.append(f"meta.option_range {mark!r} is not this task's 255-option opt-in {option_range_mark()!r}")
    for name, question in ((record.get("request") or {}).get("questions") or {}).items():
        if isinstance(question, dict) and question.get("type") == "choice":
            n = len(question.get("criteria") or {})
            if not 2 <= n <= OPTION_RANGE_MAX:
                problems.append(f"question {name!r}: choice needs 2..{OPTION_RANGE_MAX} options (has {n})")
    if problems:
        return problems
    # the labels were skipped where the range check failed inside validate_record — check them now
    from robo_jev.decisions.records import _validate_label  # noqa: PLC0415 — the module's own label check

    labels = record.get("labels") or {}
    for name, question in record["request"]["questions"].items():
        if labels and question.get("type") == "choice" and len(question.get("criteria") or {}) > 8:
            problems += _validate_label(str(name), question, labels.get(name))
    return problems
