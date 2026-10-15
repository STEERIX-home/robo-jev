"""Task v0.7f — gate M2g-F: the sealed M2g moments with the situation facts moved into each question's instructions (pure parts).

The PRE-DECLARED v0.7 SECONDARY READING — FACTS IN INSTRUCTIONS (progress.md, written 2026-10-15 ~19:57 KST, before this gate was built and
before any v0.7 gate prediction existed) reads this gate next to M2g-short. Real bench requests of this form keep the state tiny and carry a
few hundred characters of facts in every question's instructions (the request-shape audit: servo and soccer, "a JSON object" state and
250–700-char facts). This module turns M2g-short into that form; nothing here loads a model. The build and the gate check run in
``scripts/v07_gates_build.py`` (``build-facts`` / ``check-facts``), the scoring in ``artifacts/scratch/v07f/score_facts.sh`` and the reading
in ``scripts/verdict_v07_facts.py``. The author never opened task v0.7a's training transform; the status lines and templates below were written
from the audit's description only.

* *Identity* (:func:`facts_record`): every record keeps its id, its pair, its questions' names, types and criteria, its labels and every other
  field of M2g-short (so the option-order key — the pair's smaller id — and the seeded option order are M2g-short's); ``meta.gate`` gains
  ``condition`` = ``facts``, ``template`` and ``status``.
* *State*: the pair's status line (:data:`STATUS_FORMS` — a role line, a JSON-like status, a key=value line), which carries no fact: no word of
  :data:`robo_jev.bench.gates_v07.FACT_WORDS` or :data:`~robo_jev.bench.gates_v07.TONE_WORDS`, nothing about the moment, nothing evaluative,
  and no pointer to where the situation is (no word of :data:`POINTER_WORDS` — real tiny states never say "see the question").
* *Instructions*: every question's instructions are the pair's template (:data:`TEMPLATES`) filled with the original state's sentences
  (:func:`robo_jev.bench.gates_v07.sentences` — verbatim, in order, joined by one space, which equals the state with its line breaks shown as
  spaces) and the question's original instruction (verbatim). The templates differ in where the facts sit: before the instruction (``lead``),
  after it (``tail``) or around it (``split`` — the first half of the facts, the instruction, the rest; the cut is after the first half of
  the state's lines, or of its sentences for a one-line state, the same place for both records of a pair). The prompt builder shows an
  instruction on one line (:func:`robo_jev.decisions.prompt._one_line`), so the stored text — one line, single spaces — is exactly what the
  model reads.
* *Label independence* (:func:`assign`): the template and the status line are functions of the pair's order key (its stratum and the seed)
  only — within every stratum (set × scene) the templates, the status lines and their 9 combinations are each balanced to ±1 (a seeded Latin
  order) — so both records of a pair (which differ in their labels) get the same added text (:func:`added_text` cuts the original sentences
  and instructions out of a record; the pair's two cuts are byte-identical).
* *Proofs* (:func:`facts_problems`): the state is the pair's status line, holds no original sentence of 3 or more words and no fact, tone or
  pointer word; every original
  sentence is found verbatim and in order in every question's instructions, the original instruction exactly once, and the instructions are
  exactly the template's rendering; questions, options and labels equal M2g-short's; ``validate_record`` passes.
* *Overlap* (:func:`fixed_pieces`, :func:`touching_shingles`): every fixed run of the templates and status lines with 2 or more words, and every
  8-word shingle of a record's text that touches added text, is checked against the training inputs by the build (counts only).
"""

from __future__ import annotations

import copy
import json
import math
import string
from typing import Any, Mapping, Sequence

from robo_jev.bench.gates_v07 import FACT_STEMS, FACT_WORDS, M2G_PREFIX, M2G_SET, TONE_STEMS, TONE_WORDS, _rng, _word_hits, sentences
from robo_jev.data.general_sets import normalize
from robo_jev.decisions.records import validate_record

__all__ = [
    "ADDED_GATE_KEYS", "CONDITION", "FILE_NAME", "PLACEMENTS", "POINTER_WORDS", "SLOTS", "STATUS_FORMS", "TEMPLATES", "added_text", "assign", "facts_problems",
    "facts_record", "facts_split", "fixed_pieces", "latin_cells", "piece_words", "render", "template_parts", "text_problems", "touching_shingles",
]

#: The condition name (``meta.gate.condition``) and the gate file's name (``artifacts/v07/gates/m2gf/``).
CONDITION = "facts"
FILE_NAME = "M2g-facts.jsonl"
#: The keys this gate adds to ``meta.gate`` (everything else in a record is M2g-short's).
ADDED_GATE_KEYS = ("condition", "template", "status")
#: Template slots: the whole facts, the two halves of the facts (``split``) and the question's original instruction.
SLOTS = ("facts", "facts_a", "facts_b", "instruction")
PLACEMENTS = ("before", "after", "around")

#: Minimal status lines (the whole state of an M2g-F record). Written for this gate; no fact, no tone word, nothing evaluative, and no
#: pointer to where the situation is (revised after the controller's review, before any M2g-F prediction existed: the first version said
#: "particulars are written into each question" — a hint real tiny states never give).
STATUS_FORMS = {
    "role": "Judgment desk — inbound query from a deployed unit.",
    "json": '{"desk": "judgment", "sender": "deployed unit"}',
    "kv": "desk=judgment | sender=deployed-unit",
}
#: Words a status line must not use — they would point to where the situation is (whole words, also with a plain ending).
POINTER_WORDS = frozenset("""
    question instruction particular detail fact context situation report account body text attached inside within below above see each
    here following note info information
""".split())
#: Templates of every question's instructions (``str.format`` fields from :data:`SLOTS`) and where the facts sit relative to the instruction.
TEMPLATES = {
    "lead": {"placement": "before", "text": "Unit's account, as relayed: {facts} Desk query: {instruction}"},
    "tail": {"placement": "after", "text": "{instruction} (Judge it from the unit's account: {facts})"},
    "split": {"placement": "around", "text": "{facts_a} Desk query: {instruction} Rest of the unit's account: {facts_b}"},
}


def template_parts(name: str) -> list[tuple[str, str | None]]:
    """The template as (literal text, slot or None) parts in order — the literal runs and the slots between them."""
    if name not in TEMPLATES:
        raise ValueError(f"template must be one of {list(TEMPLATES)} (got {name!r})")
    parts: list[tuple[str, str | None]] = []
    for literal, field, spec, conversion in string.Formatter().parse(TEMPLATES[name]["text"]):
        if literal:
            parts.append((literal, None))
        if field is not None:
            if field not in SLOTS or spec or conversion:
                raise ValueError(f"template {name!r}: unknown slot {field!r}")
            parts.append(("", field))
    return parts


def piece_words(text: str) -> int:
    """The number of words of a fixed piece: whitespace-separated tokens holding a letter or a digit."""
    return sum(1 for token in str(text).split() if any(ch.isalnum() for ch in token))


def fixed_pieces() -> list[dict[str, Any]]:
    """Every fixed piece of added text: each maximal literal run of each template (between slots, with its punctuation, stripped) and each
    status line (a status line has no slot — it is one run). ``words`` = :func:`piece_words`; the build checks pieces of 2 words or more."""
    out = []
    for name in TEMPLATES:
        for literal, slot in template_parts(name):
            if slot is None and literal.strip():
                out.append({"source": f"template:{name}", "text": literal.strip(), "words": piece_words(literal)})
    for name, line in STATUS_FORMS.items():
        out.append({"source": f"status:{name}", "text": line, "words": piece_words(line)})
    return out


def text_problems() -> list[str]:
    """Checks of the gate's own text: at least 3 status lines and 3 templates; the templates place the facts before, after and around the
    instruction, each holds the instruction once and the facts once (whole, or both halves in order); no status line and no template literal
    uses a fact word or a tone word (:mod:`robo_jev.bench.gates_v07`'s lists and matching) and no status line a pointer word
    (:data:`POINTER_WORDS`); every slot is set off from its neighbours by a space or punctuation (word boundaries survive)."""
    problems = []
    if len(STATUS_FORMS) < 3 or len(TEMPLATES) < 3:
        problems.append("fewer than 3 status lines or templates")
    if {t["placement"] for t in TEMPLATES.values()} != set(PLACEMENTS):
        problems.append("the templates do not place the facts before, after and around the instruction")
    for name, spec in TEMPLATES.items():
        slots = [slot for _, slot in template_parts(name) if slot]
        if slots.count("instruction") != 1 or slots not in (["facts", "instruction"], ["instruction", "facts"], ["facts_a", "instruction", "facts_b"]):
            problems.append(f"template {name!r}: slots {slots}")
        elif {"before": ["facts", "instruction"], "after": ["instruction", "facts"], "around": ["facts_a", "instruction", "facts_b"]}[spec["placement"]] != slots:
            problems.append(f"template {name!r}: placement {spec['placement']!r} does not match its slots {slots}")
        parts = template_parts(name)
        for i, (_, slot) in enumerate(parts):
            if slot is None:
                continue
            before = parts[i - 1][0] if i > 0 else " "
            after = parts[i + 1][0] if i + 1 < len(parts) else " "
            if before and before[-1].isalnum() or after and after[0].isalnum():
                problems.append(f"template {name!r}: slot {slot!r} touches a letter")
        literal = " ".join(text for text, slot in parts if slot is None)
        for words, stems, what in ((FACT_WORDS, FACT_STEMS, "fact"), (TONE_WORDS, TONE_STEMS, "tone")):
            hits = _word_hits(literal, words, stems)
            if hits:
                problems.append(f"template {name!r} uses {what} words {hits}")
    for name, line in STATUS_FORMS.items():
        for words, stems, what in ((FACT_WORDS, FACT_STEMS, "fact"), (TONE_WORDS, TONE_STEMS, "tone"), (POINTER_WORDS, (), "pointer")):
            hits = _word_hits(line, words, stems)
            if hits:
                problems.append(f"status {name!r} uses {what} words {hits}")
        if "\n" in line or line != " ".join(line.split()):
            problems.append(f"status {name!r} is not one single-spaced line")
    return problems


# --------------------------------------------------------------------------
# Records
# --------------------------------------------------------------------------


def _lines(state: str) -> list[str]:
    return [line.strip() for line in str(state).splitlines() if line.strip()]


def facts_split(state: str) -> int:
    """Where ``split`` cuts the facts: the number of sentences in the first ⌈L/2⌉ lines of a state of L ≥ 2 lines, else ⌈n/2⌉ of its n
    sentences. Both halves are non-empty."""
    units = sentences(state)
    lines = _lines(state)
    if len(units) < 2:
        raise ValueError("a state of one sentence cannot be split")
    k = sum(len(sentences(line)) for line in lines[: math.ceil(len(lines) / 2)]) if len(lines) >= 2 else math.ceil(len(units) / 2)
    return min(max(k, 1), len(units) - 1)


def render(template: str, facts: Sequence[str], instruction: str, split: int) -> str:
    """The instructions of one question: the template filled with the facts (sentences joined by one space; ``split`` = where ``facts_a``
    ends) and the instruction."""
    values = {"facts": " ".join(facts), "facts_a": " ".join(facts[:split]), "facts_b": " ".join(facts[split:]), "instruction": instruction}
    return "".join(literal if slot is None else values[slot] for literal, slot in template_parts(template))


def facts_record(record: Mapping[str, Any], *, template: str, status: str) -> dict[str, Any]:
    """The M2g-F record of an M2g-short record (module docstring): the status line as the state, every question's instructions rendered
    from `template`, ``meta.gate`` + condition / template / status; everything else unchanged (the input is not modified)."""
    if template not in TEMPLATES or status not in STATUS_FORMS:
        raise ValueError(f"unknown template {template!r} or status {status!r}")
    out = copy.deepcopy(dict(record))
    state = str(record["request"]["state"])
    facts = sentences(state)
    split = facts_split(state)
    out["request"]["state"] = STATUS_FORMS[status]
    for name, question in out["request"]["questions"].items():
        question["instructions"] = render(template, facts, str(record["request"]["questions"][name]["instructions"]), split)
    gate = dict((out.get("meta") or {}).get("gate") or {})
    gate.update({"condition": CONDITION, "template": template, "status": status})
    out.setdefault("meta", {})["gate"] = gate
    return out


def _bare(record: Mapping[str, Any]) -> dict[str, Any]:
    """The record without what this gate changes: the state, the instructions, the added ``meta.gate`` keys."""
    out = copy.deepcopy(dict(record))
    request = out.get("request") or {}
    request["state"] = None
    for question in (request.get("questions") or {}).values():
        if isinstance(question, dict):
            question["instructions"] = None
    gate = (out.get("meta") or {}).get("gate")
    if isinstance(gate, dict):
        for key in ADDED_GATE_KEYS:
            gate.pop(key, None)
    return out


def facts_problems(record: Mapping[str, Any], original: Mapping[str, Any]) -> list[str]:
    """Every check of one M2g-F record against its M2g-short record (module docstring, *Proofs*). Empty = passes."""
    problems: list[str] = []
    if record.get("id") != original.get("id"):
        problems.append(f"id {record.get('id')!r} is not M2g-short's {original.get('id')!r}")
    meta = record.get("meta") or {}
    gate = meta.get("gate") or {}
    template, status = gate.get("template"), gate.get("status")
    if gate.get("condition") != CONDITION or template not in TEMPLATES or status not in STATUS_FORMS:
        problems.append(f"meta.gate is not a facts gate entry: {({k: gate.get(k) for k in ADDED_GATE_KEYS})}")
    if meta.get("set") != M2G_SET or not str(record.get("id", "")).startswith(M2G_PREFIX):
        problems.append("not under M2g's sealed set and prefix")
    if _bare(record) != _bare(original):
        problems.append("a field other than the state, the instructions and meta.gate.{condition, template, status} differs from M2g-short's")
    if list((record.get("request") or {}).get("questions") or {}) != list(original["request"]["questions"]):
        problems.append("question names or their order differ from M2g-short's")
    if record.get("labels") != original.get("labels"):
        problems.append("labels differ from M2g-short's")
    state = str((record.get("request") or {}).get("state") or "")
    facts = sentences(original["request"]["state"])
    if status in STATUS_FORMS and state != STATUS_FORMS[status]:
        problems.append("the state is not the record's status line")
    left = [s for s in facts if len(s.split()) >= 3 and s in state]
    if left:
        problems.append(f"{len(left)} original sentence(s) of 3+ words left in the state (first {left[0][:40]!r})")
    for words, stems, what in ((FACT_WORDS, FACT_STEMS, "fact"), (TONE_WORDS, TONE_STEMS, "tone"), (POINTER_WORDS, (), "pointer")):
        hits = _word_hits(state, words, stems)
        if hits:
            problems.append(f"the state uses {what} words {hits}")
    split = facts_split(original["request"]["state"])
    for name, question in ((record.get("request") or {}).get("questions") or {}).items():
        text = str((question or {}).get("instructions") or "")
        if name not in original["request"]["questions"]:
            continue
        instruction = str(original["request"]["questions"][name]["instructions"])
        position = 0
        for sentence in facts:
            found = text.find(sentence, position)
            if found < 0:
                problems.append(f"{name}: an original sentence is not verbatim in order in the instructions: {sentence[:40]!r}")
                break
            position = found + len(sentence)
        if text.count(instruction) != 1:
            problems.append(f"{name}: the original instruction appears {text.count(instruction)} times")
        if template in TEMPLATES and text != render(template, facts, instruction, split):
            problems.append(f"{name}: the instructions are not the template's rendering of the original facts and instruction")
        if text != " ".join(text.split()):
            problems.append(f"{name}: the instructions are not one single-spaced line (the prompt would show other text)")
    problems += validate_record(dict(record))
    return problems


def added_text(record: Mapping[str, Any], original: Mapping[str, Any]) -> str:
    """What the gate adds to a record, independent of the template definitions: the state, and per question the instructions with the original
    instruction cut out (``<I>``) and every original sentence cut out in order (``<S>``), as one JSON string. Raises if a piece is missing."""
    facts = sentences(original["request"]["state"])
    out: dict[str, Any] = {"state": record["request"]["state"], "questions": {}}
    for name, question in record["request"]["questions"].items():
        text = str(question["instructions"])
        instruction = str(original["request"]["questions"][name]["instructions"])
        start = text.find(instruction)
        if start < 0:
            raise ValueError(f"{record.get('id')} {name}: the original instruction is missing")
        spans = [(start, start + len(instruction), "<I>")]
        masked = text[:start] + "\x00" * len(instruction) + text[start + len(instruction):]
        position = 0
        for sentence in facts:
            found = masked.find(sentence, position)
            if found < 0:
                raise ValueError(f"{record.get('id')} {name}: an original sentence is missing")
            spans.append((found, found + len(sentence), "<S>"))
            position = found + len(sentence)
        pieces, cursor = [], 0
        for begin, end, mark in sorted(spans):
            pieces += [text[cursor:begin], mark]
            cursor = end
        pieces.append(text[cursor:])
        out["questions"][name] = "".join(pieces)
    return json.dumps(out, ensure_ascii=False, sort_keys=True)


def latin_cells(templates: Sequence[str], status: Sequence[str]) -> list[tuple[str, str]]:
    """The 9 (template, status) cells in a Latin order: block b (b = 0, 1, 2) pairs template i with status (i + b) mod 3 — every block of 3
    holds each template once and each status line once, the 3 blocks hold every cell once."""
    if len(templates) != 3 or len(status) != 3:
        raise ValueError("the Latin order needs 3 templates and 3 status lines")
    return [(templates[i], status[(i + b) % 3]) for b in range(3) for i in range(3)]


def assign(keys_by_stratum: Mapping[str, Sequence[str]], *, seed: int) -> dict[str, dict[str, str]]:
    """Pair order key → {template, status}: within each stratum (set and scene) the keys in a seeded order take the cells of
    :func:`latin_cells` (templates and status lines in seeded orders) in turn from a seeded block — so within every stratum the templates,
    the status lines and their 9 combinations are each balanced to ±1. A function of the key, its stratum and the seed only (never of a
    label)."""
    out: dict[str, dict[str, str]] = {}
    for stratum, keys in sorted(keys_by_stratum.items()):
        rng = _rng("v07f-m2gf-assign", stratum, seed)
        order = sorted(set(keys))
        rng.shuffle(order)
        templates, status = list(TEMPLATES), list(STATUS_FORMS)
        rng.shuffle(templates)
        rng.shuffle(status)
        cells = latin_cells(templates, status)
        start = 3 * rng.randrange(3)
        for i, key in enumerate(order):
            template, line = cells[(start + i) % len(cells)]
            out[key] = {"template": template, "status": line}
    return out


def _segments(record: Mapping[str, Any], original: Mapping[str, Any], name: str) -> list[tuple[str, bool]]:
    """One question's instructions as (text, added?) segments in order (the template's literal runs are added text)."""
    template = record["meta"]["gate"]["template"]
    state = original["request"]["state"]
    facts = sentences(state)
    split = facts_split(state)
    values = {"facts": " ".join(facts), "facts_a": " ".join(facts[:split]), "facts_b": " ".join(facts[split:]),
              "instruction": str(original["request"]["questions"][name]["instructions"])}
    return [(literal, True) if slot is None else (values[slot], False) for literal, slot in template_parts(template)]


def touching_shingles(record: Mapping[str, Any], original: Mapping[str, Any], size: int = 8) -> set[str]:
    """The `size`-word shingles (normalised as :func:`robo_jev.bench.iface.shingles`) of the record's texts — its state and every question's
    instructions — that hold at least one word of added text (the status line or a template literal). A text of at most `size` words gives
    itself (as ``shingles`` does)."""
    texts = [[(str(record["request"]["state"]), True)]] + [_segments(record, original, name) for name in record["request"]["questions"]]
    out: set[str] = set()
    for segments in texts:
        words: list[tuple[str, bool]] = []
        for text, added in segments:
            words += [(w, added) for w in normalize(text).split()]
        if not words:
            continue
        if len(words) <= size:
            if any(added for _, added in words):
                out.add(" ".join(w for w, _ in words))
            continue
        for i in range(len(words) - size + 1):
            window = words[i : i + size]
            if any(added for _, added in window):
                out.add(" ".join(w for w, _ in window))
    return out
