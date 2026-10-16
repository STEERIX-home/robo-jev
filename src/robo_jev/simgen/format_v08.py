"""Task v0.8a — the FORMAT package: v0.7's training records with many-option questions of up to 255 options and random answer codes,
labels unchanged (the PRE-DECLARED v0.8 VERDICT RULE, progress.md 2026-10-16). Pure parts; ``scripts/assemble_v08.py`` streams the input.

**What changes** (records are transformed, never added; ids, order, labels, v0.7's sheets and facts and every untouched line stay):

* **N, the option count** — v0.7's expanded questions (the same questions: ≈ 8 % of the English physics choice questions of the families
  action / outcome / direction / comparison, and ≈ 8 % of the general records) are expanded again from their own options with N drawn
  **log-uniform over 9..255** (:func:`draw_n`, per shuffle key and question). The distractors come from other training records' options
  **as v0.7 drew them** (:mod:`robo_jev.simgen.shape` — other answer families, another scene, no content word shared with the question,
  the same word filters), with two changes that large N needs:

  - physics (:meth:`KeyIndex.draw`): every distinct key the allowed families offer is a candidate (each key once — option keys are unique
    within a question), taken in a seeded round-robin over the families, with one passing description per key. A question whose families
    offer fewer distinct keys than N − own gets all of them; its N is then smaller than drawn (recorded as ``short``; the v0.7 pools hold
    ≈ 50 keys usable under an action or a comparison question, ≈ 90 under an outcome question, ≈ 145 under a direction question);
  - general (:func:`general_distractors`): a drawn text may share words with the other drawn texts (v0.7 also kept them apart from each
    other, which cannot reach 255); it still shares none with the item's state or own options.

  Distractors get probability 0 (and, where the label stores expected costs, the worst own option's cost, as in v0.7).
* **General items are re-keyed** (:func:`rekey`): v0.7 gave the own options their keys ``o1``…``oK`` (K ≤ 8) and the distractors ``o9``…,
  so a key past ``o8`` was never correct — a shortcut that grows with N. v0.8 gives all N options of an expanded general item the keys
  ``o1``…``oN`` in a seeded random order; the label moves with its option (same texts, same probabilities).
* **Random codes** — about 25 % of ALL choice questions (:data:`CODES_SHARE`; any N, the ordinary 2–8-option ones included, every part,
  English and Korean) show codes drawn at random without replacement from the 255-code pool (:func:`draw_codes`, in the drawn order —
  ``meta.codes[question]``, one code per shown position, :mod:`robo_jev.decisions.codes`). The trainer's order augmentation moves the correct
  option over the positions, so over the epochs every pool code is trained as an answer. All other questions show A, B, … as before.
* A record with more than 8 options carries the 255-option opt-in (``meta.option_range`` =
  :data:`robo_jev.decisions.records.OPTION_RANGE_MARK_255`); ``meta.format`` records what v0.8 did. v0.7's ``meta.shape`` is kept as v0.7
  wrote it (its ``expanded[...].to`` is v0.7's N; ``meta.format.expanded[...].to`` is the current one).

Every draw is a function of ``sha256(FORMAT_VERSION | … | shuffle key | question)``; a one-fact pair shares its shuffle key and so its N,
distractors and codes. Records the trainer holds out as dev are never transformed.
"""

from __future__ import annotations

import hashlib
import math
import random
from collections import defaultdict
from typing import Any, Iterable, Mapping, Sequence

from robo_jev.decisions.codes import CODE_POOL
from robo_jev.simgen import shape

__all__ = [
    "BINS",
    "CODES_SHARE",
    "FORMAT_VERSION",
    "N_RANGE",
    "KeyIndex",
    "bin_of",
    "codes_selected",
    "draw_codes",
    "draw_n",
    "general_distractors",
    "rekey",
    "rng",
    "unit",
]

FORMAT_VERSION = "v08-format-1"
#: The option count of an expanded question: log-uniform over 9..255 (the code pool's size).
N_RANGE = (9, 255)
#: The share of all choice questions that show random pool codes.
CODES_SHARE = 0.25
#: The reporting bins of N (the brief): 9–26 (single letters), 27–60, 61–255.
BINS = ((2, 8), (9, 26), (27, 60), (61, 255))


def unit(*parts: Any) -> float:
    """A uniform number in [0, 1) from the sha256 of :data:`FORMAT_VERSION` and the parts."""
    digest = hashlib.sha256("|".join(str(p) for p in (FORMAT_VERSION, *parts)).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / 2.0**64


def rng(*parts: Any) -> random.Random:
    digest = hashlib.sha256("|".join(str(p) for p in (FORMAT_VERSION, *parts)).encode("utf-8")).digest()
    return random.Random(int.from_bytes(digest[:8], "big"))


def draw_n(key: str, name: str) -> int:
    """N log-uniform over :data:`N_RANGE` (a continuous log-uniform draw over [9, 256), floored)."""
    low, high = N_RANGE
    value = math.exp(math.log(low) + unit("n", key, name) * (math.log(high + 1) - math.log(low)))
    return max(low, min(high, int(value)))


def bin_of(n: int) -> str:
    for low, high in BINS:
        if low <= int(n) <= high:
            return f"{low}-{high}"
    raise ValueError(f"N = {n} is outside {BINS}")


def codes_selected(key: str, name: str) -> bool:
    """Whether a choice question shows random codes (p = :data:`CODES_SHARE`)."""
    return unit("codes", key, name) < CODES_SHARE


def draw_codes(key: str, name: str, count: int) -> list[str]:
    """`count` pool codes drawn at random without replacement, in the drawn order (one per shown position)."""
    if not 1 <= int(count) <= len(CODE_POOL):
        raise ValueError(f"{count} options — the code pool has {len(CODE_POOL)} codes")
    return rng("codes-draw", key, name, int(count)).sample(list(CODE_POOL), int(count))


class KeyIndex:
    """The v0.7 distractor pool (:class:`robo_jev.simgen.shape.DistractorPool`) indexed by family and key — the physics draw of v0.8
    (module docstring)."""

    def __init__(self, pool: shape.DistractorPool) -> None:
        self.by_family: dict[str, dict[str, list[shape.PoolEntry]]] = {}
        for family, entries in pool.entries.items():
            index: dict[str, list[shape.PoolEntry]] = defaultdict(list)
            for entry in entries:
                index[entry.key].append(entry)
            self.by_family[family] = dict(index)

    def keys(self, family: str) -> dict[str, int]:
        """Distinct keys of the families a question of `family` may draw from (:data:`robo_jev.simgen.shape.ALLOWED`), with entry counts."""
        out: dict[str, int] = {}
        for source in shape.ALLOWED[family]:
            for key, entries in self.by_family.get(source, {}).items():
                out[key] = out.get(key, 0) + len(entries)
        return out

    def draw(self, family: str, *, count: int, own_keys: Iterable[str], own_norms: Iterable[str], words: Iterable[str], scene: str,
             rnd: random.Random) -> list[shape.PoolEntry]:
        """Up to `count` distractors for a question of `family`: v0.7's filters (a key and a text unseen in the question, another scene, no
        content word shared with `words`, no comparison word under a direction or comparison question, no action word under an action
        question, no outcome word under an outcome question), each distinct key at most once, taken in a seeded round-robin over the
        allowed families; per key the first passing entry in a seeded order. Fewer than `count` when the families run out of keys."""
        if count <= 0:
            return []
        families = [f for f in shape.ALLOWED[family] if self.by_family.get(f)]
        rnd.shuffle(families)
        keys_taken, norms, banned = set(map(str, own_keys)), set(own_norms), set(words)
        queues: dict[str, list[str]] = {}
        for source in families:
            keys = sorted(self.by_family[source])
            rnd.shuffle(keys)
            queues[source] = keys
        chosen: list[shape.PoolEntry] = []
        while len(chosen) < count and any(queues.values()):
            for source in families:
                if len(chosen) >= count:
                    break
                queue = queues[source]
                while queue:
                    key = queue.pop()
                    if key in keys_taken:
                        continue
                    entries = self.by_family[source][key]
                    start = rnd.randrange(len(entries))
                    found = None
                    for offset in range(len(entries)):
                        entry = entries[(start + offset) % len(entries)]
                        if entry.norm in norms or entry.scene == scene or entry.words & banned:
                            continue
                        if family in shape._NO_COMPARISON and entry.comparison:
                            continue
                        if (family == "act" and entry.action) or (family == "out" and entry.outcome):
                            continue
                        found = entry
                        break
                    if found is None:
                        continue
                    chosen.append(found)
                    keys_taken.add(found.key)
                    norms.add(found.norm)
                    break
        return chosen


def general_distractors(record: Mapping[str, Any], pool: Sequence[tuple[str, str, str]], *, count: int, rnd: random.Random,
                        max_attempts: int = 60000) -> list[str] | None:
    """`count` option texts for a general item (``pool`` = (origin, source, text) of the other general records, as v0.7): a compatible
    source (:data:`robo_jev.simgen.shape.GENERAL_SOURCES_ALLOWED`), another item, a text not equal to an own option or an earlier draw, and
    no content word shared with the item's state or own options (module docstring: drawn texts may share words with each other). None if not
    enough."""
    (name, question), = record["request"]["questions"].items()
    origin = str((record.get("meta") or {}).get("origin_group") or record.get("id"))
    allowed = set(shape.GENERAL_SOURCES_ALLOWED.get(shape.general_source(record), ()))
    banned = shape.content_words(record["request"]["state"]) | set().union(*(shape.content_words(v) for v in question["criteria"].values()))
    norms = {shape._norm(v) for v in question["criteria"].values()}
    chosen: list[str] = []
    attempts = 0
    while pool and len(chosen) < count and attempts < max_attempts:
        attempts += 1
        source_origin, source, text = pool[rnd.randrange(len(pool))]
        if source_origin == origin or source not in allowed:
            continue
        norm = shape._norm(text)
        if norm in norms:
            continue
        words = shape.content_words(text)
        if not words or words & banned:
            continue
        chosen.append(text)
        norms.add(norm)
    return chosen if len(chosen) == count else None


def rekey(own: Sequence[tuple[str, str]], drawn: Sequence[str], rnd: random.Random) -> tuple[dict[str, str], dict[str, str]]:
    """(new criteria — own options first, in their order, then the drawn texts —, old own key → new key): all N options get the keys
    ``o1``…``oN`` in a seeded random order (module docstring)."""
    n = len(own) + len(drawn)
    keys = [f"o{i}" for i in range(1, n + 1)]
    rnd.shuffle(keys)
    criteria: dict[str, str] = {}
    mapping: dict[str, str] = {}
    for (old, text), new in zip(own, keys):
        criteria[new] = text
        mapping[str(old)] = new
    for text, new in zip(drawn, keys[len(own):]):
        criteria[new] = text
    return criteria, mapping
