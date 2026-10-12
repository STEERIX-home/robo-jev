"""Fact-preservation checks for narrated training moments (Task v0.3a).

An LLM retells a training moment's ``request.state`` in another voice or format. Every rewrite must keep every fact, or it is
dropped. The checks here are automatic and deterministic. They compare the rewrite with the original state text only — no meta,
labels or hidden values are needed — so they read exactly what the model will read.

**State checks** (:func:`check_state`; each returns a list of problems, empty = pass):

* ``numbers`` — every number-with-unit of the original appears in the rewrite with the same value and unit, **as often as in the
  original** (a multiset: two facts that share a value, "front 4 cm, side 4 cm", must both survive). Values compare as decimals
  (``0.80`` = ``0.8``); units compare after spelling aliases (``seconds`` = ``s``, ``m/s^2`` = ``m/s²``, ``초`` = ``s``). No unit
  conversion is accepted: the rewrite keeps the original units. ``±`` is part of a number (a tolerance is not a value).
* ``new_numbers`` — the rewrite holds no number that the original does not: a number with a unit must exist with that unit, a bare
  number must exist with some unit. Times of day (``7:45``) are numbers too.
* ``names`` — every named agent and object of the original (catalog phrases from the skin cards: people, vehicles, robots, the
  robot's own body name, containers, loads, stacked items, placed items and targets, obstacles, occluders, blockers, corners,
  ramps, decks, sensor-confusing objects, detours, stack spots) appears in the rewrite word for word; articles, possessives,
  case and punctuation are ignored. Korean phrases match as substrings (particles attach to the last noun).
* ``language`` — English stays English (no Hangul), Korean stays Korean (Hangul ≥ 70 % of letters outside unit symbols).
* ``length`` — the rewrite's length over the original's stays inside the voice's bounds, and under the format limit.
* ``directions`` — the set of side words (left/right, 왼쪽/오른쪽) is the same, and every number that sits next to a side word
  in the original ("12 cm left of …") still sits next to the same side word in the rewrite (a swapped side is caught).
* ``fill`` / ``lid`` — the fill-level band ("about three-quarters full", "약 4분의 3") and the lid state (open / lid on / cannot
  see) of a carried container are the same.
* ``judgement`` — the rewrite adds no prediction, advice or safety judgement word that the original lacks ("dangerous",
  "should", "likely", "will tip", "위험", "해야" …): a voice may not answer the questions.
* ``log_format`` — no timestamped event-log lines (``[−0.85 s] …``): that format is the held-out style of the sealed test T2.

**Question checks** (:func:`check_question` — re-voicing "you" → "I/my" or → "the robot"): same type, same option keys in the same
order, same number of score anchors; per text field the numbers, new numbers and names checks above; the content words other than
person words are kept (a re-voiced question may change only the person and the verb agreement — hypothetical actions and time
windows stay). A question that fails keeps its original text.
"""

from __future__ import annotations

import functools
import re
import unicodedata
from collections import Counter
from decimal import Decimal, InvalidOperation
from typing import Any, Iterable

__all__ = [
    "CHECK_NAMES",
    "action_items",
    "Mention",
    "check_question",
    "check_state",
    "copy_share",
    "direction_bindings",
    "entity_catalog",
    "extract_numbers",
    "fill_bands",
    "found_entities",
    "hangul_share",
    "lid_states",
    "normalize_words",
    "number_surfaces",
]

CHECK_NAMES = ("numbers", "new_numbers", "names", "language", "length", "directions", "fill", "lid", "actions", "judgement", "log_format", "copy")

#: An item of the original's list of possible actions is kept when this share of its content words (Korean: character bigrams) occurs
#: together in one line or sentence of the rewrite.
ACTION_KEEP_SHARE = 0.75

#: A rewrite whose word 4-grams are more than this share copied from the original is too close to add diversity.
MAX_COPY_SHARE = 0.5

#: Decisions format limit on ``state`` (``record.MAX_STATE_CHARS``).
MAX_STATE_CHARS = 16_000

# --------------------------------------------------------------------------
# numbers
# --------------------------------------------------------------------------

#: Canonical unit → spellings. Latin spellings need a word boundary after them; Hangul ones attach directly ("12분", "0.5초").
#: Korean counters and labels (번, 호, 층 …) are ID-like: they normalise to the bare number ("4번 도크" = "dock 4").
_UNIT_ALIASES: dict[str, tuple[str, ...]] = {
    "m/s²": ("m/s²", "m/s^2", "m/s2", "m·s⁻²", "m s⁻²", "metres per second squared", "meters per second squared",
             "metre per second squared", "meter per second squared", "metres per second per second", "meters per second per second"),
    "m/s": ("m/s", "metres per second", "meters per second", "metre per second", "meter per second", "metres a second",
            "meters a second", "미터/초"),
    "mm/s": ("mm/s", "millimetres per second", "millimeters per second", "millimetre per second", "millimeter per second"),
    "cm/s": ("cm/s", "centimetres per second", "centimeters per second"),
    "km/h": ("km/h", "kph", "km/hr", "kilometres per hour", "kilometers per hour", "kilometre per hour", "kilometer per hour", "킬로미터/시"),
    "mm": ("mm", "millimetres", "millimeters", "millimetre", "millimeter", "밀리미터"),
    "cm": ("cm", "centimetres", "centimeters", "centimetre", "centimeter", "센티미터", "센티"),
    "m": ("m", "metres", "meters", "metre", "meter", "미터"),
    "ms": ("ms", "milliseconds", "millisecond", "msec", "밀리초"),
    "s": ("s", "sec", "secs", "seconds", "second", "-second", "초"),
    "min": ("min", "mins", "minutes", "minute", "-minute", "분"),
    "h": ("h", "hr", "hrs", "hours", "hour", "-hour", "시간"),
    "°C": ("°C", "℃", "degrees Celsius", "degrees C", "deg C", "°c"),
    "°": ("°", "deg", "degrees", "degree", "도"),
    "%": ("%", "percent", "per cent", "퍼센트", "프로"),
    "N": ("N", "newtons", "newton", "뉴턴"),
    "kg": ("kg", "kilograms", "kilogram", "kilos", "킬로그램"),
    "g": ("g", "grams", "gram", "그램"),
    "ml": ("ml", "mL", "millilitres", "milliliters", "millilitre", "milliliter", "밀리리터"),
    "l": ("l", "L", "litres", "liters", "litre", "liter", "리터"),
    "×": ("×", "x", "times", "배"),
    "": ("분의", "번지", "번째", "번", "호실", "호", "층", "병동", "동", "조", "편", "개", "명", "륜", "축", "웰", "대", "단", "칸", "열", "줄"),
}
#: Spellings that need a word boundary after them (Latin letters) vs. Hangul / symbols that do not.
_UNIT_TABLE: list[tuple[str, str, bool]] = sorted(
    ((alias, canon, bool(re.match(r"[A-Za-z]", alias[-1]))) for canon, aliases in _UNIT_ALIASES.items() for alias in aliases),
    key=lambda item: -len(item[0]),
)
#: Spellings matched case-insensitively (words); unit symbols ("N", "m", "s", "ms", "L") are case-sensitive.
_CASELESS = {alias for alias, _, _ in _UNIT_TABLE if len(alias) > 3 and alias.isascii() and alias.replace(" ", "").isalpha()}

#: A number: optional ± / sign, digits with optional thousands separators and decimals; not glued to Latin letters or digits on the
#: left ("B4", "LH718" stay codes). A time of day "7:45" is one token.
_NUMBER_RE = re.compile(
    r"(?<![A-Za-z0-9.,:])(?P<time>\d{1,2}:\d{2})(?![\d:])"
    r"|(?<![A-Za-z0-9.,])(?P<pm>±\s?)?(?P<sign>[−-](?=\d))?(?P<num>\d{1,3}(?:,\d{3})+(?![\d])|\d+)(?P<frac>\.\d+)?"
)
_SIGN_OK_BEFORE = re.compile(r"(^|[\s(\[{:;,=/])$")
_ENGLISH_NUMBER_WORDS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18,
    "nineteen": 19, "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "ninety": 90, "hundred": 100,
}
#: An English number word directly followed by a time or distance unit counts as that number ("two minutes" = "2 minutes"), so a
#: rewrite that writes it as digits is not a new number.
_WORD_UNIT_RE = re.compile(
    r"\b(" + "|".join(_ENGLISH_NUMBER_WORDS) + r")[\s-](minutes?|seconds?|hours?|metres?|meters?|centimetres?|centimeters?)\b",
    re.IGNORECASE,
)


class Mention(tuple):
    """One number mention: ``(value, unit, pm)`` — value a normalised decimal string, unit canonical ("" = bare), pm True for ±."""

    __slots__ = ()

    def __new__(cls, value: str, unit: str, pm: bool = False):
        return super().__new__(cls, (value, unit, pm))

    @property
    def value(self) -> str:
        return self[0]

    @property
    def unit(self) -> str:
        return self[1]

    @property
    def pm(self) -> bool:
        return self[2]

    def __repr__(self) -> str:
        return f"{'±' if self.pm else ''}{self.value}{(' ' + self.unit) if self.unit else ''}"


def _decimal(text: str) -> str:
    try:
        value = Decimal(text.replace(",", ""))
    except InvalidOperation:
        return text
    value = value.normalize()
    out = format(value, "f")
    return "0" if out in ("-0", "−0") else out


def _unit_at(text: str, pos: int) -> tuple[str, int]:
    """The canonical unit starting at `pos` (after at most one space) and the end position; ("", pos) if none."""
    start = pos
    if start < len(text) and text[start] in "   ":
        start += 1
    elif start + 1 < len(text) and text[start] == "-" and text[start + 1].isalpha():
        start += 1  # "250-ml", "45-degree"
    for alias, canon, latin in _UNIT_TABLE:
        end = start + len(alias)
        piece = text[start:end]
        if alias in _CASELESS:
            if piece.lower() != alias.lower():
                continue
        elif piece != alias:
            continue
        if alias in ("x",) and start != pos:  # "0.8x" only when glued; "3 x 4" is not a unit
            continue
        if alias == "도" and start != pos:  # "21도" only when glued (도 is also a particle and a word start)
            continue
        if latin and end < len(text) and re.match(r"[A-Za-z0-9]", text[end]):
            continue
        if not latin and alias[-1].isascii() and alias[-1].isalnum() and end < len(text) and re.match(r"[A-Za-z0-9]", text[end]):
            continue
        return canon, end
    return "", pos


def _replace_number_words(text: str) -> str:
    return _WORD_UNIT_RE.sub(lambda m: f"{_ENGLISH_NUMBER_WORDS[m.group(1).lower()]} {m.group(2)}", text)


def extract_numbers(text: str) -> list[Mention]:
    """Every number mention of a text, in order (see the module docstring)."""
    text = unicodedata.normalize("NFC", _replace_number_words(str(text)))
    out: list[Mention] = []
    for match in _NUMBER_RE.finditer(text):
        if match.group("time"):
            hours, minutes = match.group("time").split(":")
            out.append(Mention(f"{int(hours)}:{minutes}", "time"))
            continue
        sign = match.group("sign") or ""
        if sign and not _SIGN_OK_BEFORE.search(text[: match.start("sign")]):
            sign = ""  # "AMR-2", "3-4": a hyphen inside a word is not a minus sign
        value = _decimal(("-" if sign else "") + match.group("num") + (match.group("frac") or ""))
        unit, _ = _unit_at(text, match.end())
        out.append(Mention(value, unit, bool(match.group("pm"))))
    return out


def number_surfaces(text: str) -> list[tuple[Mention, str]]:
    """Every number mention with its surface string as written (``"0.6°"``, ``"±10.0 mm"``, ``"4번"``)."""
    text = unicodedata.normalize("NFC", _replace_number_words(str(text)))
    out = []
    mentions = iter(extract_numbers(text))
    for match in _NUMBER_RE.finditer(text):
        mention = next(mentions)
        if match.group("time"):
            out.append((mention, match.group("time")))
            continue
        _, end = _unit_at(text, match.end())
        out.append((mention, text[match.start() : end].strip()))
    return out


def _number_problems(original: str, rewrite: str) -> tuple[list[str], list[str]]:
    want = Counter(extract_numbers(original))
    have = Counter(extract_numbers(rewrite))
    missing = []
    for mention, count in sorted(want.items(), key=repr):
        if have[mention] < count:
            missing.append(f"{mention!r} ×{count - have[mention]}")
    values = {m.value for m in want}
    pairs = {(m.value, m.unit) for m in want}
    new = []
    for mention in sorted(have, key=repr):
        if mention.unit == "":
            if mention.value not in values:
                new.append(repr(mention))
        elif (mention.value, mention.unit) not in pairs:
            new.append(repr(mention))
    return missing, new


# --------------------------------------------------------------------------
# names (catalog phrases)
# --------------------------------------------------------------------------

_DETERMINERS = {"a", "an", "the", "your", "my", "its", "their", "his", "her", "our", "this", "that", "these", "those"}
_WORD_RE = re.compile(r"[a-z0-9]+(?:['’][a-z]+)?")


def normalize_words(text: str) -> list[str]:
    """Lower-case words of an English text without articles and possessive determiners; ``snake_case`` keys split into words."""
    text = unicodedata.normalize("NFKC", str(text)).lower().replace("_", " ").replace("’", "'")
    return [w for w in _WORD_RE.findall(text) if w not in _DETERMINERS]


def _squash(text: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", str(text))).strip()


def _contains_words(haystack: list[str], needle: list[str]) -> bool:
    if not needle:
        return True
    n = len(needle)
    first = needle[0]
    for i, word in enumerate(haystack):
        if word == first and haystack[i : i + n] == needle:
            return True
    return False


def _entry_text(entry: Any, key: str = "text") -> str:
    return entry if isinstance(entry, str) else entry[key]


@functools.lru_cache(maxsize=1)
def entity_catalog() -> dict[str, dict[str, str]]:
    """Language → {phrase: category} of the named agents and objects on the skin cards (English card + its Korean phrases)."""
    from robo_jev.simgen.catalog import SKINS_PATH, _read_yaml
    from robo_jev.simgen.korean import ko_skins

    skins = _read_yaml(SKINS_PATH)
    english: dict[str, str] = {}

    def add(phrase: str, category: str) -> None:
        phrase = _squash(phrase)
        if phrase and phrase not in english:
            english[phrase] = category

    for names in skins.get("body_names", {}).values():
        for n in names:
            add(_entry_text(n), "body")
    for entries in skins.get("decks", {}).values():
        for e in entries:
            add(e["text"], "deck")
    for card in skins.get("domains", {}).values():
        for names in card.get("body_names", {}).values():
            for n in names:
                add(_entry_text(n), "body")
        for kind, entries in (card.get("descriptors") or {}).items():
            for e in entries:
                add(e["np"], "agent")
                add(e["ref"], "agent_ref")
        for e in card.get("containers", []):
            for p in e["phrases"]:
                add(p, "container")
        for key, category in (("loads", "load"), ("stack_items", "stack_item")):
            for e in card.get(key, []):
                for p in e["phrases"]:
                    add(p, category)
        for e in card.get("p4_items", []):
            add(e["item"], "p4_item")
            add(e["target"], "p4_target")
        for key in ("obstacles", "bench_obstacles", "occluders", "blockers", "ramps", "ramp_ends", "detours", "stack_spots", "corners"):
            for e in card.get(key, []):
                add(_entry_text(e), key.rstrip("s"))
        for e in card.get("conflicts", []):
            for p in e["phrases"]:
                add(p, "conflict")
    ko = ko_skins()
    korean: dict[str, str] = {}
    for phrase, category in english.items():
        options = []
        if phrase in ko.text:
            options.append(ko.text[phrase])
        if phrase in ko.descriptors:
            options += [ko.descriptors[phrase]["np"], ko.descriptors[phrase]["ref"]]
        if phrase in ko.objects:
            options += [ko.objects[phrase]["np"], ko.objects[phrase]["ref"]]
        for option in options:
            option = _squash(option)
            if option and option not in korean:
                korean[option] = category
    # descriptor refs and object refs that are only reachable through an np
    for entry in list(ko.descriptors.values()) + list(ko.objects.values()):
        for key in ("np", "ref"):
            option = _squash(entry[key])
            if option and option not in korean:
                korean[option] = "agent_ref" if key == "ref" else "agent"
    return {"en": english, "ko": korean}


def _ko_word_start_find(text: str, phrase: str) -> bool:
    """A Korean phrase occurs starting at a word start (start, space, punctuation or a digit before it; particles may follow)."""
    start = text.find(phrase)
    while start >= 0:
        if start == 0 or not ("\uac00" <= text[start - 1] <= "\ud7a3" or text[start - 1].isalpha()):
            return True
        start = text.find(phrase, start + 1)
    return False


def found_entities(text: str, language: str) -> list[str]:
    """The catalog phrases that occur in a text (longest first; a phrase inside a longer found phrase is still listed)."""
    catalog = entity_catalog()["ko" if language == "ko" else "en"]
    if language == "ko":
        squashed = _squash(text)
        return sorted((p for p in catalog if len(p) >= 2 and _ko_word_start_find(squashed, p)), key=lambda p: (-len(p), p))
    words = normalize_words(text)
    out = []
    for phrase in catalog:
        needle = normalize_words(phrase)
        if needle and _contains_words(words, needle):
            out.append(phrase)
    return sorted(out, key=lambda p: (-len(p), p))


def _name_problems(original: str, rewrite: str, language: str) -> list[str]:
    wanted = found_entities(original, language)
    if language == "ko":
        squashed = _squash(rewrite)
        return [p for p in wanted if not _ko_word_start_find(squashed, p)]
    words = normalize_words(rewrite)
    return [p for p in wanted if not _contains_words(words, normalize_words(p))]


# --------------------------------------------------------------------------
# language, length
# --------------------------------------------------------------------------

_UNIT_SYMBOL_RE = re.compile(r"\b(?:m/s²|m/s|km/h|mm/s|mm|cm|ms|m|s|N|ml|kg|g|°C|min|h)\b")


def hangul_share(text: str) -> float:
    """Hangul letters / (Hangul + Latin letters), unit symbols and JSON-ish snake_case keys removed."""
    text = _UNIT_SYMBOL_RE.sub(" ", str(text))
    text = re.sub(r'"[a-z0-9_]+"\s*:', " ", text)
    hangul = sum(1 for ch in text if "가" <= ch <= "힣" or "ㄱ" <= ch <= "ㆎ")
    latin = sum(1 for ch in text if ch.isascii() and ch.isalpha())
    return hangul / (hangul + latin) if hangul + latin else 0.0


def _language_problems(rewrite: str, language: str) -> list[str]:
    share = hangul_share(rewrite)
    if language == "ko":
        return [] if share >= 0.70 else [f"hangul share {share:.2f} < 0.70"]
    return [] if share == 0.0 else [f"hangul in an English rewrite ({share:.2f})"]


def _length_problems(original: str, rewrite: str, bounds: tuple[float, float]) -> list[str]:
    ratio = len(rewrite) / max(1, len(original))
    out = []
    if not bounds[0] <= ratio <= bounds[1]:
        out.append(f"length ratio {ratio:.2f} outside [{bounds[0]}, {bounds[1]}]")
    if len(rewrite) > MAX_STATE_CHARS:
        out.append(f"{len(rewrite)} characters > {MAX_STATE_CHARS}")
    if not rewrite.strip():
        out.append("empty rewrite")
    return out


# --------------------------------------------------------------------------
# directions
# --------------------------------------------------------------------------

_SIDE_WORDS = {"en": {"left": "left", "right": "right", "lefthand": "left", "righthand": "right"},
               "ko": {"왼쪽": "left", "오른쪽": "right", "좌측": "left", "우측": "right", "왼편": "left", "오른편": "right"}}
_TOKEN_RE = re.compile(r"\n|[;!?]|[^\s;!?,:()\[\]{}\"|]+")


#: "right" / "left" that are not sides: "right now", "right away", "all right", "…, right?", "has left", "nothing left". Each
#: pattern's group "w" is the word to blank out.
_NOT_SIDE_RES = tuple(re.compile(p, re.IGNORECASE) for p in (
    r"\b(?P<w>right)\s+(?:now|away|there|here|after|before|behind|ahead|next|then|beside|by|up|down|into|onto|under|over|across|along|through|until|when|where|as|this|that|off|in front|at the moment)\b",
    r"\ball (?P<w>right)\b",
    r"\b(?P<w>right)\s*\?",
    r",\s*(?P<w>right)\s*(?:[?!.,]|$)",
    r"\b(?P<w>right)-of-way\b",
    r"\b(?:is|are|was|were|has|have|had|be|been|nothing|anything|something|time|room|space|seconds|metres|meters)\s+(?P<w>left)\b",
    r"\b(?P<w>left)\s+(?:over|behind|sticking|unattended|running|alone|out)\b",
))


def _mask_non_sides(text: str) -> str:
    for pattern in _NOT_SIDE_RES:
        text = pattern.sub(lambda m: m.group(0)[: m.start("w") - m.start()] + "x" * len(m.group("w")) + m.group(0)[m.end("w") - m.start():], text)
    return text


def _side_of(token: str, language: str) -> str | None:
    if language == "ko":
        for word, side in _SIDE_WORDS["ko"].items():
            if word in token:
                return side
        return None
    word = token.lower().strip("'’-_").replace("_", "-")
    for part in re.split(r"[-_]", word):
        if part in _SIDE_WORDS["en"]:
            return _SIDE_WORDS["en"][part]
    return None


def _tokens(text: str, language: str) -> list[tuple[str, Any]]:
    """Text → tokens: ("break", None) at line and sentence ends, ("num", Mention) for numbers, ("side", side), ("word", None).
    A unit symbol right after a number is not a token of its own (so "12 cm left" puts the side next to the number)."""
    out: list[tuple[str, Any]] = []
    text = _replace_number_words(str(text)).replace("_", " ")
    if language != "ko":
        text = _mask_non_sides(text)
    for match in _TOKEN_RE.finditer(text):
        token = match.group(0)
        if token == "\n" or token in ";!?":
            out.append(("break", None))
            continue
        sentence_end = token.endswith(".") and not re.search(r"\d\.$", token)
        token = token.rstrip(".")
        if token:
            numbers = extract_numbers(token)
            side = _side_of(token, language)
            if numbers:
                out.extend(("num", m) for m in numbers)
                if side:
                    out.append(("side", side))
            elif side:
                out.append(("side", side))
            elif out and out[-1][0] == "num" and _unit_at(" " + token, 0)[0] and _unit_at(" " + token, 0)[1] >= len(token):
                pass  # the unit of the number just before ("12 cm")
            else:
                out.append(("word", None))
        if sentence_end:
            out.append(("break", None))
    return out


def _nearest(tokens: list[tuple[str, Any]], index: int, kind: str, window: int, *, cross_breaks: bool = False) -> list[tuple[int, int]]:
    """The nearest tokens of `kind` to token `index` within `window` tokens (not across a line or sentence break unless
    `cross_breaks`) → [(distance, token index)]. ± tolerances are never candidates."""
    found: list[tuple[int, int]] = []
    for step in (-1, 1):
        j, distance = index + step, 0
        while 0 <= j < len(tokens):
            if tokens[j][0] == "break":
                if not cross_breaks:
                    break
                j += step
                continue
            if kind == "num" and tokens[j][0] == "num" and tokens[j][1].pm:
                j += step
                continue
            distance += 1
            if distance > window:
                break
            if tokens[j][0] == kind:
                found.append((distance, j))
                break
            j += step
    if not found:
        return []
    best = min(d for d, _ in found)
    return [(d, j) for d, j in found if d == best]


def _nearest_side(tokens: list[tuple[str, Any]], index: int, window: int, *, cross_breaks: bool = False) -> str | None:
    """The side word nearest to token `index` (None if none within the window, or a left/right tie)."""
    sides = {tokens[j][1] for _, j in _nearest(tokens, index, "side", window, cross_breaks=cross_breaks)}
    return sides.pop() if len(sides) == 1 else None


def direction_bindings(text: str, language: str, window: int = 4) -> list[tuple[Mention, str]]:
    """Numbers bound to a side word: each side word binds the number(s) nearest to it ("12 cm left", "왼쪽으로 12 cm") — a number
    further away than another number from that side word is not bound to it."""
    tokens = _tokens(text, language)
    bound: dict[int, set[str]] = {}
    for i, (kind, value) in enumerate(tokens):
        if kind == "side":
            for _, j in _nearest(tokens, i, "num", window):
                bound.setdefault(j, set()).add(value)
    return [(tokens[j][1], sides.pop()) for j, sides in sorted(bound.items()) if len(sides) == 1]


#: The vertical sign of a reading ("1.8 mm above nominal", "기준보다 1.8 mm 높음"): the set of words must be the same.
_VERTICAL = {"en": {"above": r"\babove\b", "below": r"\bbelow\b"},
             "ko": {"above": r"높음|높습니다|높다|높아요|높게|더 높", "below": r"낮음|낮습니다|낮다|낮아요|낮게|더 낮"}}


def _direction_problems(original: str, rewrite: str, language: str) -> list[str]:
    problems = []
    flags = 0 if language == "ko" else re.IGNORECASE
    table = _VERTICAL["ko" if language == "ko" else "en"]
    vertical_o = {name for name, pattern in table.items() if re.search(pattern, original, flags)}
    vertical_r = {name for name, pattern in table.items() if re.search(pattern, rewrite, flags)}
    if vertical_o != vertical_r:
        problems.append(f"above/below {sorted(vertical_o)} → {sorted(vertical_r)}")
    sides_o = {v for k, v in _tokens(original, language) if k == "side"}
    sides_r = {v for k, v in _tokens(rewrite, language) if k == "side"}
    if sides_o != sides_r:
        problems.append(f"side words {sorted(sides_o)} → {sorted(sides_r)}")
    tokens_r = _tokens(rewrite, language)
    for mention, side in direction_bindings(original, language):
        ok = False
        for i, (kind, value) in enumerate(tokens_r):
            if kind == "num" and value == mention and _nearest_side(tokens_r, i, 8, cross_breaks=True) == side:
                ok = True
                break
        if not ok:
            problems.append(f"{mention!r} no longer next to '{side}'")
    return problems


# --------------------------------------------------------------------------
# fill band and lid state
# --------------------------------------------------------------------------

_FILL_EN = (
    ("lt_third", r"less than (?:a |one )?third"),
    ("third", r"(?<!less than )(?<!less than a )(?<!two-)(?<!two )\b(?:a|one)[ -]third full"),
    ("half", r"\bhalf[ -]full|\bhalf of its"),
    ("two_thirds", r"two[ -]thirds"),
    ("three_quarters", r"three[ -]quarters|three[ -]fourths"),
    ("almost_full", r"almost full"),
    ("near_brim", r"nearly to the brim|nearly brim"),
    ("brim", r"(?<!nearly )(?<!nearly to )full to the brim|brim[ -]full|filled to the brim"),
)
_FILL_KO = (
    ("lt_third", r"3분의 ?1\s*(?:미만|이 안|가 안|도 안|에 못)"),
    ("third", r"3분의 ?1(?!\s*(?:미만|이 안|가 안|도 안|에 못))"),
    ("half", r"절반[^.\n;|]{0,6}(?:차\s?있|차있|채워|찬 상태|차 있)|(?:채움|채워진 정도|담긴 양|내용물)\s*[:：]?\s*(?:약\s*|대략\s*)?절반|—\s*(?:약\s*)?절반\s*[;,.)]"),
    ("two_thirds", r"3분의 ?2"),
    ("three_quarters", r"4분의 ?3"),
    ("almost_full", r"거의 가득"),
    ("near_brim", r"가장자리 가까이"),
    ("brim", r"가장자리까지 가득"),
)


def fill_bands(text: str, language: str) -> set[str]:
    """The fill-level bands a text names (catalog FILL_BINS / FILL_KO wording)."""
    table = _FILL_KO if language == "ko" else _FILL_EN
    flags = 0 if language == "ko" else re.IGNORECASE
    return {name for name, pattern in table if re.search(pattern, str(text), flags)}


_LID_EN = (
    ("unknown", r"(?:cannot|can't|can not|could not|couldn't|unable to|not able to|no way to|impossible to)(?: be)? (?:see|seen|tell|told|confirm|confirmed|make out|made out|determine|determined|verify|verified|check|checked)[^.\n;]{0,40}\blid"
                r"|\blid\b[^.\n;]{0,30}(?:not visible|unknown|unclear|cannot tell|can't tell|cannot be seen|can't be seen|not be seen|unconfirmed|not known|undetermined|unsure|uncertain|not confirmed)"
                r"|whether (?:or not )?(?:it|there|the [a-z-]+) (?:has|is|have) (?:a |its |the )?lid|whether (?:the |a )?lid is (?:on|present|there|fitted)"
                r"|(?:tell|see|know|sure) if (?:it|there) (?:has|is) (?:a |its )?lid"
                r"|\blid\b[\"']?\s*[:=]\s*(?:null|[\"'](?:unknown|not visible|unclear|cannot tell|can't tell|cannot be seen|not seen)[^\"']*[\"'])"),
    ("open", r"\bno lid\b|\bwithout (?:a |any )?lid\b|\blid\b[\"']?(?:\s*[:=]\s*|\s+is\s+|\s+status\s*[:=]?\s*|\s+)[\"']?(?:none|no|off|absent|not present|false|open)\b"
             r"|\"has_lid\"\s*:\s*false|\blidless\b|\bunlidded\b|\buncovered\b"
             r"|\bnot lidded\b|\bhas no cover\b|\blid is off\b|\bno cover\b"),
    ("lid", r"\b(?:has|with|wearing) (?:a |its )?lid on\b"
            r"|(?<!no )(?<!without a )(?<!without )(?<!not have a )(?<!n't have a )\blid\b[\"']?(?:\s*[:=]\s*|\s+is\s+|\s+status\s*[:=]?\s*|\s+)[\"']?(?:on|yes|present|closed|fitted|true)\b"
            r"|\"has_lid\"\s*:\s*true"
            r"|(?<!un)(?<!not )\blidded\b|\bhas a lid\b(?! or)|\bwith a lid\b|\blid is on\b|\bcovered by (?:a |its )?lid\b|\bcapped\b"),
)
_LID_KO = (
    ("unknown", r"뚜껑[^.\n]{0,12}(?:보이지 않|알 수 없|모름|확인(?:이)? (?:안|불가|되지 않|할 수 없)|불분명|안 보|확인 불가)|뚜껑이 있는지|뚜껑[\"']?\s*[:：]?\s*[\"']?\s*(?:미상|불명)"
                r"|\"lid\"\s*:\s*(?:null|\"(?:알 수 없음|보이지 않음|확인 불가|미상|불명)[^\"]*\")"),
    ("open", r"뚜껑(?:이|은|도)? ?없|뚜껑[\"']?\s*[:：]?\s*[\"']?\s*없음|뚜껑 미착용|뚜껑이 열려|뚜껑 없이|\"lid\"\s*:\s*(?:false|\"(?:없음|없다|없어요|미착용)\")"),
    ("lid", r"뚜껑(?:이|을|은)? ?(?:덮여|덮인|닫혀|씌워|있음|있습니다|있다|있어|있고|있는 상태)|뚜껑[\"']?\s*[:：]\s*[\"']?\s*있음|뚜껑 덮음"
            r"|\"lid\"\s*:\s*(?:true|\"(?:있음|덮임|닫힘|덮여 있음)\")"),
)


def lid_states(text: str, language: str) -> set[str]:
    """The lid states a text states: ``unknown`` (cannot see), ``open`` (no lid) and/or ``lid`` (lid on)."""
    table = _LID_KO if language == "ko" else _LID_EN
    flags = 0 if language == "ko" else re.IGNORECASE
    text = str(text)
    found = set()
    for name, pattern in table:
        hits = list(re.finditer(pattern, text, flags))
        if not hits:
            continue
        if name == "lid" and "unknown" in found:
            # "cannot see whether it has a lid" also contains "has a lid" — keep only hits outside the unknown phrases
            unknown_spans = [m.span() for m in re.finditer(table[0][1], text, flags)]
            hits = [h for h in hits if not any(a <= h.start() < b + 15 for a, b in unknown_spans)]
            if language == "ko":
                hits = [h for h in hits if not re.match(r"뚜껑이 있는지", text[h.start():h.start() + 8])]
        if hits:
            found.add(name)
    return found


# --------------------------------------------------------------------------
# judgement words and the T2 log format
# --------------------------------------------------------------------------

#: Judgement word groups: (name, words a rewrite may not add, words in the original that allow them). A group is flagged when the
#: rewrite has its words and the original has none of the allowing words — restating a spec, a rule or the planner's advice is fine.
_JUDGEMENT_EN = (
    ("obligation", r"\b(?:should|shouldn't|must|ought|needs? to|ha(?:ve|s) to|required to)\b",
     r"\b(?:should|must|ought|needs? to|ha(?:ve|s) to|required|requires?|spec|rule|limit|means within|tolerance)\b"),
    ("danger", r"\bdanger(?:ous)?\b|\bhazard(?:ous)?\b|\brisk(?:y|s)?\b|\bunsafe\b|\bsafe(?:ly|r|st)?\b|\bcareful(?:ly)?\b|\bcaution(?:ary|ious|iously)?\b"
               r"|\bwarn(?:ing|s)?\b|\balert\b|\bemergency\b", None),
    ("likelihood", r"\b(?:likely|unlikely|probabl[ey]|possibly|perhaps|chances? (?:of|are|is)|odds)\b"
                   r"|\bpossible\b(?!\s+(?:actions?|options?|moves?|manoeuvres?|maneuvers?|responses?|choices?|next steps?|steps?))", None),
    ("outcome", r"\bcollid(?:e|es|ing)\b|\bcollision\b|\bcrash(?:es|ing)?\b|\btopple(?:s|d)?\b|\btip(?:s|ping)? over\b|\bspill(?:s|ing|ed)?\b"
                r"|\bwill (?:hit|tip|fall|spill|slide|touch|collide|crash|topple)\b", None),
    ("advice", r"\bbest (?:option|choice|move|action)\b|\brecommend(?:s|ed|ation)?\b|\bplanner\b|\badvi(?:ce|ses?|sed)\b|\bsuggest(?:s|ed|ion)?\b", None),
)
_JUDGEMENT_KO = (
    ("obligation", r"해야|하여야|필요가 있", r"해야|하여야|필요|규격|규칙|기준|제한|허용"),
    ("danger", r"위험|안전하|안전한|조심|주의(?:해|하|가 필요)|경고|비상", None),
    ("likelihood", r"아마|가능성이 (?:높|크|있)|것 같|듯하", None),
    ("outcome", r"충돌|부딪(?:힐|칠)|넘어질|쏟아질|쏟을|엎질러질|미끄러질|떨어질", None),
    ("advice", r"계획기|플래너|권(?:고|장|하|합|해|했|할|한)|조언|제안|바람직|하는 (?:게|것이) 좋", None),
)
#: Words that name the planner's advice. When the original has advice, the rewrite may restate it ("the planner says the robot
#: should brake …"): sentences that name the planner are then not searched for judgement words.
_PLANNER_RE = {"en": re.compile(r"\bplanner|\badvi(?:ce|ses?|sed)\b|\brecommend|\bsuggest", re.IGNORECASE),
               "ko": re.compile(r"계획기|플래너|권(?:고|장|하|합|해|했|할|한)|조언|제안")}


def _judgement_problems(original: str, rewrite: str, language: str) -> list[str]:
    groups = _JUDGEMENT_KO if language == "ko" else _JUDGEMENT_EN
    flags = 0 if language == "ko" else re.IGNORECASE
    planner = _PLANNER_RE["ko" if language == "ko" else "en"]
    if planner.search(original):  # the advice may be restated: leave out the sentences that name the planner
        pieces = re.split(r"(?<=[.!?。])\s+|\n", rewrite)
        rewrite = "\n".join(p for p in pieces if not planner.search(p))
    out = []
    for name, pattern, allow in groups:
        hit = re.search(pattern, rewrite, flags)
        if hit and not re.search(allow or pattern, original, flags):
            out.append(f"{name}: {hit.group(0)}")
    return out


_LOG_LINE_RE = re.compile(r"^\s*[-*•]?\s*\[\s*(?:[−\-+]?\s*\d[\d.,:]*\s*[a-zA-Z초분]*|now|t\s*=?[^\]]*|지금)\s*\]", re.MULTILINE | re.IGNORECASE)


def copy_share(original: str, rewrite: str, language: str) -> float:
    """Share of the rewrite's word 4-grams (Korean: character 6-grams without spaces) that occur in the original."""
    if language == "ko":
        a, b = re.sub(r"\s+", "", str(original)), re.sub(r"\s+", "", str(rewrite))
        grams_o = {a[i : i + 6] for i in range(len(a) - 5)}
        grams_r = [b[i : i + 6] for i in range(len(b) - 5)]
    else:
        wo = re.findall(r"[a-z0-9.]+", str(original).lower())
        wr = re.findall(r"[a-z0-9.]+", str(rewrite).lower())
        grams_o = {tuple(wo[i : i + 4]) for i in range(len(wo) - 3)}
        grams_r = [tuple(wr[i : i + 4]) for i in range(len(wr) - 3)]
    if not grams_r:
        return 0.0
    return sum(g in grams_o for g in grams_r) / len(grams_r)


_ACTION_HEADERS = ("AVAILABLE ACTIONS", "가능한 동작")
_ACTION_STOP = {"at", "to", "the", "a", "an", "and", "or", "of", "your", "my", "its", "it", "you", "i", "me", "robot", "robot's", "unit",
                "then", "first", "about", "as", "can", "on", "in", "with", "for", "by"}


def action_items(text: str) -> list[str]:
    """The items of an engine text's list of possible actions (``AVAILABLE ACTIONS`` / ``가능한 동작`` followed by ``- item`` lines)."""
    lines = str(text).split("\n")
    for i, line in enumerate(lines):
        if line.strip() in _ACTION_HEADERS:
            items = []
            for follow in lines[i + 1 :]:
                stripped = follow.strip()
                if not stripped.startswith("- "):
                    break
                items.append(stripped[2:].strip())
            return items
    return []


def _action_units(text: str, language: str) -> Counter:
    if language == "ko":
        chars = re.sub(r"\s+", "", str(text))
        return Counter(chars[i : i + 2] for i in range(len(chars) - 1))
    words = re.findall(r"[a-z0-9]+(?:'[a-z]+)?", unicodedata.normalize("NFKC", str(text)).lower().replace("’", "'"))
    return Counter(_stem(w) for w in words if w not in _ACTION_STOP)


def _action_problems(original: str, rewrite: str, language: str) -> list[str]:
    items = action_items(original)
    if not items:
        return []
    pieces = [p for p in re.split(r"\n|(?<=[.!?;])\s+", str(rewrite)) if p.strip()]
    units = [_action_units(p, language) for p in pieces]
    missing = []
    for item in items:
        want = _action_units(item, language)
        total = sum(want.values())
        if not total:
            continue
        best = max((sum(min(c, have[k]) for k, c in want.items()) / total for have in units), default=0.0)
        if best < ACTION_KEEP_SHARE:
            missing.append(item)
    return [f"action not kept: {m}" for m in missing]


def _log_format_problems(rewrite: str) -> list[str]:
    hits = _LOG_LINE_RE.findall(rewrite)
    return [f"{len(hits)} timestamped log lines (the T2 held-out style)"] if len(hits) >= 2 else []


# --------------------------------------------------------------------------
# state and question checks
# --------------------------------------------------------------------------


def check_state(original: str, rewrite: str, language: str, bounds: tuple[float, float], *,
                copy_limit: float | None = MAX_COPY_SHARE) -> dict[str, list[str]]:
    """All state checks → {check name: problems}; the rewrite passes when every list is empty. ``copy_limit=None`` skips the copy
    check (a JSON dump is a new format whatever phrases its values reuse)."""
    missing, new = _number_problems(original, rewrite)
    fill_o, fill_r = fill_bands(original, language), fill_bands(rewrite, language)
    lid_o, lid_r = lid_states(original, language), lid_states(rewrite, language)
    return {
        "numbers": missing,
        "new_numbers": new,
        "names": _name_problems(original, rewrite, language),
        "language": _language_problems(rewrite, language),
        "length": _length_problems(original, rewrite, bounds),
        "directions": _direction_problems(original, rewrite, language),
        "fill": [] if fill_o == fill_r else [f"fill {sorted(fill_o)} → {sorted(fill_r)}"],
        "lid": [] if lid_o == lid_r else [f"lid {sorted(lid_o)} → {sorted(lid_r)}"],
        "actions": _action_problems(original, rewrite, language),
        "judgement": _judgement_problems(original, rewrite, language),
        "log_format": _log_format_problems(rewrite),
        "copy": [] if copy_limit is None or (share := copy_share(original, rewrite, language)) <= copy_limit else [f"{share:.2f} of the 4-grams copied"],
    }


_PERSON_EN = {
    "you", "your", "yours", "yourself", "i", "me", "my", "mine", "myself", "it", "its", "itself", "robot", "robot's", "unit",
    "am", "are", "is", "was", "were", "do", "does", "did", "have", "has", "i'm", "you're", "it's", "i've", "you've", "i'd", "you'd",
    "i'll", "you'll", "a", "an", "the", "of", "for", "to", "own",
}
#: A Korean person token: 당신 / 나·내·저·제 / 로봇 with an optional particle (Korean verbs do not change with the person).
_PERSON_KO_RE = re.compile(r"^(?:당신|나|내|저|제|로봇|이 로봇|본 로봇)(?:은|는|이|가|의|을|를|에게|에게서|한테|도|와|과|만|께서|로서|로)?[.,?!]*$")


def _stem(word: str) -> str:
    """Undo the third-person singular -s (brakes → brake, touches → touch, carries → carry); the same on both sides."""
    if len(word) > 4 and word.endswith("ies"):
        return word[:-3] + "y"
    if len(word) > 4 and re.search(r"(?:ch|sh|ss|x|z|o)es$", word):
        return word[:-2]
    if len(word) > 3 and word.endswith("s") and not word.endswith(("ss", "us", "is")):
        return word[:-1]
    return word


def _content_words(text: str, language: str) -> Counter:
    if language == "ko":  # 어절 other than person tokens
        return Counter(t for t in _squash(text).split(" ") if t and not _PERSON_KO_RE.match(t))
    words = re.findall(r"[a-z0-9]+(?:'[a-z]+)?", unicodedata.normalize("NFKC", str(text)).lower().replace("’", "'"))
    return Counter(_stem(w) for w in words if w not in _PERSON_EN)


def _content_problems(original: str, rewrite: str, language: str) -> list[str]:
    want, have = _content_words(original, language), _content_words(rewrite, language)
    lost = sorted(k for k, c in want.items() if have[k] < c)
    added = sorted(k for k, c in have.items() if want[k] < c)
    out = []
    if lost:
        out.append(f"content words lost {lost[:6]}")
    if len(added) > 2:
        out.append(f"content words added {added[:6]}")
    return out


def _text_problems(original: str, rewrite: str, language: str) -> list[str]:
    if not isinstance(rewrite, str) or not rewrite.strip():
        return ["empty text"]
    missing, new = _number_problems(original, rewrite)
    out = [f"number missing {m}" for m in missing] + [f"new number {n}" for n in new]
    out += [f"name missing {n}" for n in _name_problems(original, rewrite, language)]
    out += _language_problems(rewrite, language)
    out += _content_problems(original, rewrite, language)
    if _direction_problems(original, rewrite, language):
        out.append("side words changed")
    return out


def check_question(original: dict[str, Any], revoiced: Any, language: str) -> list[str]:
    """Problems of one re-voiced question (empty = keep the re-voiced text). `revoiced` is the LLM's
    ``{"instructions", "criteria"?}`` for that question."""
    if not isinstance(revoiced, dict):
        return ["not an object"]
    problems = [f"instructions: {p}" for p in _text_problems(original["instructions"], revoiced.get("instructions"), language)]
    kind = original["type"]
    criteria = original.get("criteria")
    new_criteria = revoiced.get("criteria", criteria if kind == "noul" else None)
    if kind == "choice":
        if not isinstance(new_criteria, dict) or list(new_criteria) != list(criteria):
            return problems + ["option keys changed"]
        for key, text in criteria.items():
            problems += [f"option {key}: {p}" for p in _text_problems(text, new_criteria[key], language)]
    elif kind == "score":
        if not isinstance(new_criteria, list) or len(new_criteria) != len(criteria or []):
            return problems + ["score anchors changed"]
        for i, (old, new) in enumerate(zip(criteria, new_criteria)):
            problems += [f"anchor {i}: {p}" for p in _text_problems(old, new, language)]
    elif new_criteria not in (None, criteria):
        problems.append("noul gained criteria")
    return problems


def summarize(results: Iterable[dict[str, list[str]]]) -> dict[str, int]:
    """Number of rewrites failing each check."""
    counts = Counter()
    for result in results:
        for name, problems in result.items():
            if problems:
                counts[name] += 1
    return dict(counts)
