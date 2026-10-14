"""Task iface — the pre-declared interface measurements M1 (many options) and M2 (bench-shaped long states).

The rule (progress.md, "PRE-DECLARED INTERFACE MEASUREMENTS M1 / M2", written 2026-10-14 ~21:05 KST before any M1/M2 data existed) is
quoted in :data:`RULE`; this module holds the pure parts: building the two sets and applying the two readings. Nothing here loads a model;
the scoring runs in ``scripts/iface_m1_score.py`` (M1) and ``scripts/diagnose_decisions.py`` (M2, the standard path), and
``scripts/iface_verdict.py`` reads the predictions.

**M1 set** — unambiguous multiple-choice items that are in no training input, each shown with N ∈ :data:`N_LEVELS` options:

* *general items*: ARC test questions (Easy and Challenge; :func:`arc_items`) with exactly four options, no option that points at other
  options ("all of the above", "A and B"), no question about a missing picture, no negatively phrased question ("all of the following
  EXCEPT", "which is NOT" — any unrelated drawn option would answer it) and no number in any option (a drawn "1 year" would also answer a
  "365 days" question). Shown like the general training mix: state ``question: "<stem>"``, one fixed instruction
  (:data:`ARC_INSTRUCTION`), option keys ``o1 … oN``. Keys are given to the options by a seeded permutation per item and N, so a key
  carries no position or answer information, and they stay with their option across the three listings of that N.
* *physics items* (:func:`physics_items`): the one-fact *direction* question of B1-infamily and H-set records (more likely / less likely /
  about the same) whose label has a single clear best option — band-robust under the record's three priors and a base-prior top probability
  ≥ :data:`PHYSICS_MIN_TOP` — with the record's own state and instruction, one record per one-fact pair. Only the direction kind: the outcome,
  comparison and action options of the other families paraphrase each other ("unmoved" / "stays", "secure" / "held"), so a drawn option
  could be a second right answer; no other kind's option states a direction.
* *distractors* are drawn from **other items of the same source** and appended after the item's own wrong options in a fixed order, so the
  option set at N is nested in the set at every larger N and the item and its answer are the same at every N. General: wrong options of other
  ARC questions from a different topic (no content word shared between the two questions *and their options* — a question naming "metal"
  never lends "aluminum" to an item whose options say "metal"), sharing no content word with the item's question, its options or the
  distractors drawn before, no number, not starting with a pronoun ("They make their own food" would bind to the new question's subject),
  each from a different question, and a phrase of at least three words (:func:`draw_general`). Measured general items answer in a phrase
  of at least three words (:func:`general_target`): a bare noun drawn from an unrelated question can be a second right answer to a category
  question ("heat" for "which of these do all cells release? — energy"; an audit of 40 short-answer items at N = 26 found 3–5 such cases). Physics: options of the other questions of the same set
  with a key the item does not have, one per key (keys must be unique in a request); B1 has 18 such keys and the H-set 11, so the item's own
  set comes first and the sibling physics set (same generator, same formats) supplies the rest (:func:`draw_physics`).
* *positions* (:func:`positions`): the correct option at controlled positions — at N ≥ 17 three listings per item, one in each bucket
  A–H / I–P / Q–Z (Q–U at N = 21); at N = 8 three, one in each third of the list; at the descriptive levels N = 4, 12, 16 one listing
  (:data:`LISTINGS_BY_N`, fixed for GPU time before any data); within a band the position cycles over the items so every letter is used
  about equally. The other options keep one seeded order per item and N (the same in the three listings); the
  correct option is inserted at the listing's position. The prompt code shows the request order (``seed=None``): no reshuffle.
* the 2–8 option cap of the Decisions records (:data:`robo_jev.decisions.records.OPTION_RANGE`) is **not** changed: :func:`m1_problems`
  applies it unless the caller passes ``lift_cap=True`` (``scripts/iface_m1_score.py --lift-option-cap``), and then allows 2–26 — the
  number of one-token answer codes.

**M1 readings** (:func:`m1_readings`, :func:`m1_outcome`): accuracy (argmax over the shown codes = correct; ties go to the earlier option,
as :func:`robo_jev.decisions.metrics.choice_metrics`), the probability on the correct option, accuracy by the correct option's bucket at
N ≥ 17 (both N = 21 and N = 26), late-letter penalty = acc(A–H) − acc(Q–Z); per item first (the mean over its listings), then the mean over
items; 95 % percentile bootstrap with the item as the cluster (2,000 draws, seed 0 — :func:`robo_jev.bench.system1._draws`), paired across
models on the same draws.

**M2 sheets** (:func:`wrap_state`): the record's state goes verbatim into one section (``CURRENT SITUATION``) of a sectioned sheet — ROLE,
ROBOT STATUS, CAMERA TRACKS (objects unrelated to the task, all ≥ 3 m away and static or moving away), CURRENT SITUATION, EARLIER
OBSERVATIONS (log lines about routine events), LOCAL PLAN (a neutral recommendation) — filled to ≈ 2,000 / 4,000 / 8,000 characters.
The filler is seeded by the record's order key (:func:`robo_jev.decisions.records.shuffle_key`), so the two records of a one-fact pair get
the same filler (line counts decided for the pair's longest state) and differ only in the fact, and a longer sheet extends the shorter one
(the same track and log lines plus more). Every
filler line is written for this task; none mentions loads, decks, trays, grippers, surfaces, liquids, slopes or the robot's motion. The
questions, labels, ids and ``meta`` are unchanged, so the prompt code shows the same option orders as for the short state.

**M2 readings** (:func:`m2_outcome`): paired differences wrapped − short per trained model and length; the rule's gate reads direction,
comparison and call-view DC.
"""

from __future__ import annotations

import hashlib
import math
import random
import re
from collections import defaultdict
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from robo_jev.data.general_sets import normalize
from robo_jev.decisions.records import OPTION_RANGE, SCHEMA, shuffle_key, validate_record

__all__ = [
    "ARC_INSTRUCTION",
    "BUCKETS",
    "GENERAL_SOURCES",
    "LATE_MIN_N",
    "LENGTHS",
    "LIFTED_RANGE",
    "LISTINGS",
    "LISTINGS_BY_N",
    "M1_SET",
    "N_LEVELS",
    "PHYSICS_MIN_TOP",
    "QUESTION",
    "RULE",
    "arc_items",
    "bucket_of",
    "content_tokens",
    "draw_general",
    "draw_physics",
    "expand_item",
    "general_index",
    "general_target",
    "m1_outcome",
    "m1_problems",
    "m1_readings",
    "m1_rows",
    "m2_model_reading",
    "m2_outcome",
    "pair_reference_lengths",
    "physics_items",
    "physics_pool",
    "positions",
    "shingles",
    "select_physics",
    "sheet_problems",
    "sheet_sections",
    "wrap_record",
    "wrap_state",
]

#: The pre-declared rule, verbatim parts (progress.md, 2026-10-14 ~21:05 KST, before any M1/M2 data or result existed).
RULE = {
    "written": "2026-10-14 ~21:05 KST, before any M1/M2 data or result existed (progress.md: PRE-DECLARED INTERFACE MEASUREMENTS M1 / M2)",
    "models": "untrained Qwen3.5-2B, v0.1-mix (the published v0.1), v0.4-s0, v0.4-s1, v0.6-s0, v0.6-s1; prompt v0c-p3, chat, Spark",
    "M1_set": "unambiguous multiple-choice items NOT in any training input (e.g. ARC / OpenBookQA test items, each checked against the general "
              "training mix) plus physics choice items from B1 / H-set, each expanded with distractors drawn from other items to N in {4, 8, 12, "
              "16, 21, 26} options, the correct option placed at controlled positions spread over A...Z (same item, same answer across N). "
              "Readout as the model's own (softmax over the shown letter codes), the 2-8 cap lifted only inside the measurement script.",
    "M1_metrics": "accuracy (argmax = correct), probability on the correct option, accuracy by the correct option's position bucket (A-H / I-P / "
                  "Q-Z) at N >= 17, late-letter penalty = acc(A-H) - acc(Q-Z)",
    "M1_reading": "FINE-TUNE BIAS iff a trained model's late-letter penalty exceeds the untrained base's by >= 0.10 with a 95 % bootstrap interval "
                  "excluding 0 -> the next version must train varied option counts (2-26); GENERALISES iff a trained model's accuracy at N >= 17 is "
                  "within 0.05 of its N = 8 accuracy on the same items and its late-letter penalty is within 0.05 of the base's -> raising the cap "
                  "plus a light mix suffices; otherwise MIXED.",
    "M2_set": "B1 and H-set records (labels known, never trained) with the state wrapped, all original facts kept verbatim in one section, into "
              "bench-like sheets (role header, robot status, camera track list with unrelated objects, earlier-observation log lines, a LOCAL "
              "PLAN block with a neutral recommendation) at ~2k, 4k and 8k characters; the same questions.",
    "M2_metrics": "paired metrics short vs wrapped: direction, comparison, call-view DC, balanced band, choice accuracy",
    "M2_reading": "LONG-STATE GAP iff any of direction / comparison drops >= 0.05 or DC rises >= 0.05 at a length with the interval excluding 0, "
                  "for the trained models -> the next version needs long, sectioned states in training; NO GAP iff all within +-0.03; otherwise MIXED.",
    "scope": "Neither measurement trains anything or touches T9 or A4.",
}

M1_SET = "iface-m1"
QUESTION = "q"
N_LEVELS = (4, 8, 12, 16, 21, 26)
#: The bucket readings (and the late-letter penalty) use the listings with N at least this.
LATE_MIN_N = 17
#: The rule's position buckets: (name, first position, end) — 0-based, end exclusive (Q-Z is cut at N).
BUCKETS = (("A-H", 0, 8), ("I-P", 8, 16), ("Q-Z", 16, 26))
#: Listings per item and N: three controlled positions where the rule reads (N = 8, and N ≥ 17 — one per bucket), one elsewhere (the
#: descriptive levels 4, 12, 16 — chosen for GPU time before any M1 data existed).
LISTINGS = 3
LISTINGS_BY_N = {4: 1, 8: 3, 12: 1, 16: 1, 21: 3, 26: 3}
#: Inside the M1 script only: 2..26 options (A-Z are verified single tokens at the answer boundary).
LIFTED_RANGE = (2, 26)
ARC_INSTRUCTION = "Which option answers the question best?"
GENERAL_SOURCES = ("ARC-Easy", "ARC-Challenge")
#: A physics item's label must give its best option at least this probability (base prior) — "a single clear best option".
PHYSICS_MIN_TOP = 0.9
#: General targets answer in a phrase or sentence, and drawn distractors are phrases or sentences too: a bare noun drawn from an unrelated
#: question can be a second right answer to a category question ("heat" for "which of these do all cells release? — energy").
MIN_ANSWER_WORDS = 3
MIN_DISTRACTOR_WORDS = 3
#: M2 sheet lengths (characters of the wrapped state) by name.
LENGTHS = {"2k": 2000, "4k": 4000, "8k": 8000}

_STOP = frozenset(
    """a about above across after again against all also although always am among an and another any are around as at away be because been
    before being below best between both but by can could did do does doing done down during each either else enough even ever every
    following for from further get gets got had has have having he her here hers him his how however i if in into is it its itself just
    least less like likely made make makes many may me might more most much must my near nearly neither never no nor not now of off often
    on once only onto or other others our out over own per perhaps put rather really same should shown since so some something such than
    that the their them themselves then there these they this those though through thus to too toward towards under until up upon us use
    used uses using usually very was way we well were what whatever when where whether which while who whom whose why will with within
    without would yet you your best better""".split()
)
_NUMBER_WORDS = frozenset(
    "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen twenty "
    "thirty forty fifty sixty seventy eighty ninety hundred thousand million billion half quarter third dozen once twice double triple".split()
)
_META_OPTION = re.compile(
    r"\b(all|none|both|neither) of (the |these |those )?(above|these|those|them|choices|options|answers)\b"
    r"|\b(choices?|options?|answers?) \(?[a-e1-5]\)?(?![a-z])|^\(?[a-e]\)? (and|or) \(?[a-e]\)?\.?$",
    re.IGNORECASE,
)
_VISUAL = re.compile(
    r"\b(diagram|diagrams|picture|pictured|pictures|figure|illustration|illustrated|drawing|photograph|shown (below|above|here|at right|at left))\b"
    r"|\b(graph|table|chart|map|model) (below|above|shown)\b",
    re.IGNORECASE,
)
_DIGIT = re.compile(r"\d")
#: Negatively phrased questions ("all of the following EXCEPT", "which is NOT") — any unrelated drawn option would answer them.
_NEGATION = re.compile(r"\b(except|exception|not|least|false|incorrect|untrue|never|cannot|unlikely|without)\b|n't\b", re.IGNORECASE)
#: Options that start with a pronoun ("They make their own food") bind to whatever question they are moved to.
_PRONOUN_START = re.compile(r"^(they|it|its|this|these|those|he|she|his|her|their|there|that)\b", re.IGNORECASE)


def _rng(*parts: Any) -> random.Random:
    digest = hashlib.sha256("|".join(str(p) for p in parts).encode("utf-8")).digest()
    return random.Random(int.from_bytes(digest[:8], "big"))


def _stem(token: str) -> str:
    if token.endswith("ies") and len(token) > 4:
        return token[:-3] + "y"
    if token.endswith(("ches", "shes", "sses", "xes", "zes")) and len(token) > 4:
        return token[:-2]
    if token.endswith("s") and not token.endswith(("ss", "us", "is")) and len(token) > 3:
        token = token[:-1]
    if token.endswith("ing") and len(token) > 5:
        return token[:-3]
    if token.endswith("ed") and len(token) > 4:
        return token[:-2]
    return token


def content_tokens(text: str) -> set[str]:
    """Lower-cased word tokens of `text` without stop words, crudely stemmed; numbers and number words always count."""
    out = set()
    for token in re.findall(r"[a-z0-9]+", str(text).lower().replace("'s ", " ")):
        if token.isdigit() or token in _NUMBER_WORDS:
            out.add(token)
        elif len(token) >= 3 and token not in _STOP:
            out.add(_stem(token))
    return out


def _numeric(text: str) -> bool:
    return bool(_DIGIT.search(text)) or bool(set(re.findall(r"[a-z]+", text.lower())) & _NUMBER_WORDS)


def shingles(text: str, size: int = 8) -> set[str]:
    """The `size`-word windows of the normalised text (the whole text if shorter)."""
    words = normalize(text).split()
    if not words:
        return set()
    if len(words) <= size:
        return {" ".join(words)}
    return {" ".join(words[i : i + size]) for i in range(len(words) - size + 1)}


# --------------------------------------------------------------------------
# M1 items
# --------------------------------------------------------------------------


def arc_items(rows: Iterable[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """ARC rows → general items (module docstring) and the drop counts. An item: ``item``, ``source`` (ARC-Easy / ARC-Challenge), ``state``,
    ``instructions``, ``correct`` (text), ``own`` (the three wrong options in source order), ``stem``, ``wrong_pool`` (the wrong options,
    for drawing distractors for other items)."""
    items, dropped = [], defaultdict(int)
    seen = set()
    for row in rows:
        texts = [" ".join(str(t).split()) for t in (row.get("choices") or {}).get("text") or []]
        labels = [str(x) for x in (row.get("choices") or {}).get("label") or []]
        stem = " ".join(str(row.get("question") or "").split())
        key = str(row.get("answerKey") or "")
        reason = None
        if len(texts) != 4 or len(labels) != 4:
            reason = "not_four_options"
        elif key not in labels:
            reason = "no_gold"
        elif any(not t for t in texts) or len({normalize(t) for t in texts}) != 4:
            reason = "empty_or_duplicate_option"
        elif any(_META_OPTION.search(t) for t in texts):
            reason = "option_refers_to_options"
        elif _VISUAL.search(stem):
            reason = "refers_to_a_picture"
        elif _NEGATION.search(stem):
            reason = "negative_question"
        elif any(_numeric(t) for t in texts):
            reason = "number_in_an_option"
        elif normalize(stem) in seen:
            reason = "duplicate_question"
        if reason:
            dropped[reason] += 1
            continue
        seen.add(normalize(stem))
        gold = labels.index(key)
        items.append({
            "item": f"arc:{row['id']}", "source": str(row.get("subset")), "state": f"question: \"{stem}\"", "instructions": ARC_INSTRUCTION,
            "stem": stem, "correct": texts[gold], "own": [t for i, t in enumerate(texts) if i != gold],
        })
    return items, dict(dropped)


def _top(label: Mapping[str, Any]) -> tuple[str, float, float]:
    probs = sorted(((float(v), k) for k, v in (label.get("probabilities") or {}).items()), reverse=True)
    return probs[0][1], probs[0][0], (probs[1][0] if len(probs) > 1 else 0.0)


def _pair_group(record: Mapping[str, Any]) -> str:
    return shuffle_key(dict(record))


def physics_items(records: Sequence[Mapping[str, Any]], *, source: str) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Direction questions with a single clear best option (module docstring), every eligible record (one per pair is chosen later by
    :func:`select_m1`). An item: ``item``, ``source``, ``record``, ``question``, ``group`` (the pair's order key), ``scene``, ``state``,
    ``instructions``, ``correct`` (key, description), ``own`` (the other options, request order), ``keys`` (all own keys)."""
    items, counts = [], defaultdict(int)
    for record in records:
        meta = record.get("meta") or {}
        for name, kind in (meta.get("question_kind") or {}).items():
            if kind != "direction" or name not in record["request"]["questions"]:
                continue
            counts["direction_questions"] += 1
            question = record["request"]["questions"][name]
            label = (record.get("labels") or {}).get(name) or {}
            robust = ((meta.get("robustness") or {}).get(name) or {}).get("robust") is True
            best, top, _ = _top(label)
            if not robust:
                counts["not_robust"] += 1
                continue
            if top < PHYSICS_MIN_TOP:
                counts["top_below_0.9"] += 1
                continue
            criteria = question["criteria"]
            items.append({
                "item": f"{source}:{record['id']}:{name}", "source": source, "record": str(record["id"]), "question": name,
                "group": _pair_group(record), "scene": str(meta.get("scene")), "state": record["request"]["state"],
                "instructions": question["instructions"], "correct": (best, criteria[best]),
                "own": [(k, criteria[k]) for k in criteria if k != best], "keys": sorted(criteria),
            })
            counts["eligible"] += 1
    return items, dict(counts)


def physics_pool(records: Sequence[Mapping[str, Any]]) -> dict[str, list[tuple[str, str, str]]]:
    """Key → (record id, pair group, description) of every choice option of a set (the drawing pool)."""
    pool: dict[str, list[tuple[str, str, str]]] = defaultdict(list)
    for record in records:
        group = _pair_group(record)
        for question in record["request"]["questions"].values():
            if question.get("type") == "choice":
                for key, desc in question["criteria"].items():
                    pool[str(key)].append((str(record["id"]), group, " ".join(str(desc).split())))
    return dict(pool)


def draw_physics(item: Mapping[str, Any], own_pool: Mapping[str, list], sibling_pool: Mapping[str, list], *, count: int, seed: int) -> list[tuple[str, str]]:
    """`count` distractors (key, description) for a physics item: keys the item does not have, one per key, from records of other pair
    groups — the item's own set first (seeded key order), then the sibling set's remaining keys. Raises if there are not enough keys."""
    rng = _rng("iface-m1-physics", item["item"], seed)
    taken = set(item["keys"])
    out: list[tuple[str, str]] = []
    for pool in (own_pool, sibling_pool):
        keys = sorted(k for k in pool if k not in taken)
        rng.shuffle(keys)
        for key in keys:
            choices = sorted({desc for rid, group, desc in pool[key] if group != item["group"]})
            if not choices:
                continue
            out.append((key, rng.choice(choices)))
            taken.add(key)
            if len(out) == count:
                return out
    raise ValueError(f"{item['item']}: only {len(out)} distinct distractor keys available, {count} needed")


def _topic(item: Mapping[str, Any]) -> set[str]:
    """The content words of a general item's question and all four options — its topic for the distance check."""
    return content_tokens(item["stem"]).union(*(content_tokens(t) for t in [item["correct"], *item["own"]]))


def general_target(item: Mapping[str, Any]) -> bool:
    """A general item that may be measured: its answer is a phrase or sentence (≥ :data:`MIN_ANSWER_WORDS` words)."""
    return len(str(item["correct"]).split()) >= MIN_ANSWER_WORDS


def general_index(items: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """The drawing pool of the general items: per item its topic words (question and options) and its wrong options with their words
    (number-free phrases of ≥ :data:`MIN_DISTRACTOR_WORDS` words that do not start with a pronoun or point at other options)."""
    out = []
    for item in items:
        wrong = [(t, content_tokens(t)) for t in item["own"] if not _numeric(t) and not _META_OPTION.search(t) and not _PRONOUN_START.search(t)
                 and len(t.split()) >= MIN_DISTRACTOR_WORDS]
        out.append({"item": item["item"], "topic": _topic(item), "wrong": wrong})
    return out


def draw_general(item: Mapping[str, Any], index: Sequence[Mapping[str, Any]], *, count: int, seed: int) -> list[str]:
    """`count` distractor texts for a general item (module docstring): wrong options of other questions whose topic (question and all
    options) shares no content word with this item's topic — so a question about "metal" never lends "aluminum" to a question whose
    options say "metal" —, sharing no content word with this item's topic or with the distractors drawn before, not equal to any option
    taken, each from a different question; candidates in a seeded order. Raises if there are not enough."""
    rng = _rng("iface-m1-general", item["item"], seed)
    own = _topic(item)
    taken = {normalize(t) for t in [item["correct"], *item["own"]]}
    order = list(range(len(index)))
    rng.shuffle(order)
    out: list[str] = []
    for i in order:
        other = index[i]
        if other["item"] == item["item"] or other["topic"] & own or not other["wrong"]:
            continue
        candidates = [(t, tokens) for t, tokens in other["wrong"] if not tokens & own and normalize(t) not in taken and tokens]
        if not candidates:
            continue
        text, tokens = candidates[rng.randrange(len(candidates))]
        out.append(text)
        taken.add(normalize(text))
        own |= tokens  # later draws share no content word with earlier ones either (the source's topic stays allowed: other options
        #                of that question are not taken — one per question)
        if len(out) == count:
            return out
    raise ValueError(f"{item['item']}: only {len(out)} general distractors found, {count} needed")


def bucket_of(position: int, n: int) -> str | None:
    """The rule's bucket of a 0-based position (only for N ≥ 17)."""
    if n < LATE_MIN_N:
        return None
    for name, lo, hi in BUCKETS:
        if lo <= position < min(hi, n):
            return name
    raise ValueError(f"position {position} outside N = {n}")


def _bands(n: int) -> list[tuple[int, int]]:
    if n >= LATE_MIN_N:
        return [(lo, min(hi, n)) for _, lo, hi in BUCKETS]
    if LISTINGS_BY_N.get(n, LISTINGS) == 1:
        return [(0, n)]
    edges = [0, math.ceil(n / 3), math.ceil(2 * n / 3), n]
    return [(edges[j], edges[j + 1]) for j in range(LISTINGS)]


def positions(n: int, rank: int) -> list[int]:
    """The controlled positions (0-based) of the correct option for an item of rank `rank` at N = `n` (:data:`LISTINGS_BY_N`): one per
    bucket (N ≥ 17), one per third (N = 8), or one anywhere in the list (N = 4, 12, 16) — cycling inside the band with the rank (and the
    level) so every position is used about equally."""
    level = N_LEVELS.index(n) if n in N_LEVELS else 0
    return [lo + (rank + level) % (hi - lo) for lo, hi in _bands(n)]


def expand_item(item: Mapping[str, Any], distractors: Sequence[Any], *, rank: int, index: int, seed: int) -> list[dict[str, Any]]:
    """Every listing of one item: N ∈ :data:`N_LEVELS` × its controlled positions (:func:`positions`), as ``simgen-request/0`` records (the options in the shown
    order; ``meta`` carries the design). `distractors` = the drawn ones (texts for general items, (key, description) for physics)."""
    general = item["source"] in GENERAL_SOURCES
    pool = list(item["own"]) + list(distractors)
    out = []
    for n in N_LEVELS:
        if len(pool) < n - 1:
            raise ValueError(f"{item['item']}: {len(pool)} distractors for N = {n}")
        others = pool[: n - 1]
        rng = _rng("iface-m1-layout", item["item"], n, seed)
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
        for listing, position in enumerate(positions(n, rank)):
            shown = others[:position] + [correct] + others[position:]
            criteria = {k: v for k, v in shown}
            if len(criteria) != n:
                raise ValueError(f"{item['item']}: duplicate option keys at N = {n}")
            letter = chr(ord("A") + position)
            out.append({
                "schema": SCHEMA, "id": f"{M1_SET}-{index:05d}-n{n:02d}-{letter}",
                "meta": {"set": M1_SET, "item": item["item"], "source": item["source"], "n": n, "position": position, "letter": letter,
                         "listing": listing, "bucket": bucket_of(position, n), "correct": correct[0], "rank": rank},
                "request": {"state": item["state"], "questions": {QUESTION: {"type": "choice", "instructions": item["instructions"], "criteria": criteria}}},
                "labels": {QUESTION: {"type": "choice", "kind": "probe", "probabilities": {k: (1.0 if k == correct[0] else 0.0) for k in criteria}}},
            })
    return out


def m1_problems(record: Mapping[str, Any], *, lift_cap: bool = False) -> list[str]:
    """:func:`robo_jev.decisions.records.validate_record`; with `lift_cap` (only the M1 script, by an explicit flag) its 2–8 option-range
    problem is replaced by a 2–26 check — every other check stays."""
    problems = validate_record(dict(record))
    if not lift_cap:
        return problems
    low, high = OPTION_RANGE
    out = []
    for name, question in record["request"]["questions"].items():
        n = len(question.get("criteria") or {}) if question.get("type") == "choice" else None
        cap_message = f"question {name!r}: choice needs {low}..{high} options (has {n})"
        if n is not None and not LIFTED_RANGE[0] <= n <= LIFTED_RANGE[1]:
            out.append(f"question {name!r}: choice needs {LIFTED_RANGE[0]}..{LIFTED_RANGE[1]} options even with the cap lifted (has {n})")
        problems = [p for p in problems if p != cap_message]
    return problems + out


# --------------------------------------------------------------------------
# M1 readings
# --------------------------------------------------------------------------


def m1_rows(records: Sequence[Mapping[str, Any]], predictions: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """One row per scored listing: item, source, N, position, bucket, correct (argmax = correct, ties to the earlier option),
    p_correct, code_mass. `predictions` are :func:`robo_jev.decisions.zero_shot.diagnose_records` outputs (one listing per record)."""
    by_id = {str(p["id"]): p for p in predictions}
    rows = []
    for record in records:
        meta = record["meta"]
        found = by_id.get(str(record["id"]))
        if found is None:
            continue
        listing = found["questions"][QUESTION]["listings"][0]
        options, probs = list(listing["options"]), [float(x) for x in listing["code_probs"]]
        if options != list(record["request"]["questions"][QUESTION]["criteria"]):
            raise ValueError(f"{record['id']}: the scored option order differs from the record's")
        target = options.index(meta["correct"])
        best = int(np.argmax(probs))  # first maximum = the earlier option
        rows.append({"id": str(record["id"]), "item": meta["item"], "source": meta["source"], "n": int(meta["n"]), "position": int(meta["position"]),
                     "bucket": meta.get("bucket"), "listing": int(meta["listing"]), "correct": float(best == target), "p_correct": probs[target],
                     "code_mass": listing.get("code_mass"), "prompt_tokens": listing.get("prompt_tokens")})
    return rows


def _item_table(rows: Sequence[Mapping[str, Any]], items: Sequence[str]) -> dict[str, np.ndarray]:
    """Per-item values (rows = `items`): acc and p at each N, bucket accuracies at N ≥ 17, the late accuracy (all N ≥ 17 listings)."""
    index = {item: i for i, item in enumerate(items)}
    k = len(items)
    acc = {n: np.full((k, LISTINGS_BY_N[n]), np.nan) for n in N_LEVELS}
    prob = {n: np.full((k, LISTINGS_BY_N[n]), np.nan) for n in N_LEVELS}
    mass = {n: np.full((k, LISTINGS_BY_N[n]), np.nan) for n in N_LEVELS}
    bucket_sum = {name: np.zeros(k) for name, _, _ in BUCKETS}
    bucket_n = {name: np.zeros(k) for name, _, _ in BUCKETS}
    for row in rows:
        i = index.get(row["item"])
        if i is None:
            continue
        n, j = row["n"], row["listing"]
        acc[n][i, j] = row["correct"]
        prob[n][i, j] = row["p_correct"]
        mass[n][i, j] = np.nan if row.get("code_mass") is None else float(row["code_mass"])
        if n >= LATE_MIN_N:
            bucket_sum[row["bucket"]][i] += row["correct"]
            bucket_n[row["bucket"]][i] += 1
    complete = np.all([np.all(np.isfinite(acc[n]), axis=1) for n in N_LEVELS], axis=0)
    out = {f"acc@{n}": acc[n].mean(1) for n in N_LEVELS} | {f"p@{n}": prob[n].mean(1) for n in N_LEVELS}
    for n in N_LEVELS:
        finite = np.isfinite(mass[n])
        count = finite.sum(1)
        out[f"mass@{n}"] = np.where(count > 0, np.where(finite, mass[n], 0.0).sum(1) / np.maximum(count, 1), np.nan)
    for name, _, _ in BUCKETS:
        out[f"bucket:{name}"] = np.where(bucket_n[name] > 0, bucket_sum[name] / np.maximum(bucket_n[name], 1), np.nan)
    late = [n for n in N_LEVELS if n >= LATE_MIN_N]
    out["acc@late"] = np.mean([acc[n].mean(1) for n in late], axis=0)
    out["p@late"] = np.mean([prob[n].mean(1) for n in late], axis=0)
    out["penalty"] = out["bucket:A-H"] - out["bucket:Q-Z"]
    out["late_minus_8"] = out["acc@late"] - out["acc@8"]
    out["complete"] = complete.astype(float)
    return out


#: The readings of one model, in report order.
M1_METRICS = tuple([f"acc@{n}" for n in N_LEVELS] + ["acc@late"] + [f"p@{n}" for n in N_LEVELS] + ["p@late"]
                   + [f"bucket:{name}" for name, _, _ in BUCKETS] + ["penalty", "late_minus_8"] + [f"mass@{n}" for n in N_LEVELS])


def _ci(values: np.ndarray, draws: np.ndarray) -> dict[str, Any]:
    boot = values[draws].mean(1)
    return {"value": float(values.mean()), "low": float(np.percentile(boot, 2.5)), "high": float(np.percentile(boot, 97.5)), "n": int(values.size)}


def m1_readings(rows_by_model: Mapping[str, Sequence[Mapping[str, Any]]], *, base: str, n_boot: int = 2000, seed: int = 0,
                subsets: Mapping[str, set[str]] | None = None) -> dict[str, Any]:
    """Every M1 reading per model on the items every model answered completely, with item-cluster bootstrap intervals (the same draws for
    every model and the paired differences), plus the per-subset breakdowns (`subsets`: name → items) without changing the draws of the
    pooled reading."""
    from robo_jev.bench.system1 import _draws

    tables_all = {}
    common: set[str] | None = None
    for model, rows in rows_by_model.items():
        items = sorted({r["item"] for r in rows})
        table = _item_table(rows, items)
        done = {item for item, ok in zip(items, table["complete"]) if ok}
        common = done if common is None else common & done
        tables_all[model] = rows
    items = sorted(common or set())
    if not items:
        raise ValueError("M1: no item has every listing scored for every model")
    tables = {model: _item_table(rows, items) for model, rows in tables_all.items()}
    draws = _draws(len(items), n_boot, seed)
    out: dict[str, Any] = {"items": len(items), "n_boot": int(n_boot), "seed": int(seed), "models": {}, "paired_vs_base": {}, "subsets": {}}
    for model, table in tables.items():
        out["models"][model] = {metric: _ci(table[metric], draws) for metric in M1_METRICS if np.all(np.isfinite(table[metric]))}
    for model, table in tables.items():
        if model == base:
            continue
        diff = table["penalty"] - tables[base]["penalty"]
        out["paired_vs_base"][model] = {"penalty": _ci(diff, draws) | {"a": float(table["penalty"].mean()), "b": float(tables[base]["penalty"].mean())},
                                        "acc@late": _ci(table["acc@late"] - tables[base]["acc@late"], draws),
                                        "acc@8": _ci(table["acc@8"] - tables[base]["acc@8"], draws)}
    for name, members in (subsets or {}).items():
        keep = np.array([item in members for item in items])
        if not keep.any():
            continue
        sub_draws = _draws(int(keep.sum()), n_boot, seed)
        entry = {"items": int(keep.sum()), "models": {}}
        for model, table in tables.items():
            entry["models"][model] = {metric: _ci(table[metric][keep], sub_draws) for metric in
                                      ("acc@4", "acc@8", "acc@16", "acc@late", "p@8", "p@late", "bucket:A-H", "bucket:I-P", "bucket:Q-Z", "penalty", "late_minus_8")}
            if model != base:
                entry["models"][model]["penalty_minus_base"] = _ci(table["penalty"][keep] - tables[base]["penalty"][keep], sub_draws)
        out["subsets"][name] = entry
    out["item_list_sha256"] = hashlib.sha256("\n".join(items).encode("utf-8")).hexdigest()
    return out


#: M1 thresholds (rule text).
BIAS_DELTA, GENERAL_ACC, GENERAL_PENALTY = 0.10, 0.05, 0.05


def m1_model_outcome(paired: Mapping[str, Any], own: Mapping[str, Any]) -> dict[str, Any]:
    """One trained model: FINE-TUNE BIAS iff Δpenalty (model − base) ≥ 0.10 with the 95 % interval excluding 0; GENERALISES iff |acc(N ≥ 17)
    − acc(N = 8)| ≤ 0.05 and |Δpenalty| ≤ 0.05 (points); otherwise MIXED."""
    d = paired["penalty"]
    gap = own["late_minus_8"]
    checks = {"dpenalty_ge_0.10": d["value"] >= BIAS_DELTA, "dpenalty_interval_excludes_0": d["low"] > 0 or d["high"] < 0,
              "abs_late_minus_8_le_0.05": abs(gap["value"]) <= GENERAL_ACC, "abs_dpenalty_le_0.05": abs(d["value"]) <= GENERAL_PENALTY}
    if checks["dpenalty_ge_0.10"] and checks["dpenalty_interval_excludes_0"]:
        outcome = "FINE-TUNE BIAS"
    elif checks["abs_late_minus_8_le_0.05"] and checks["abs_dpenalty_le_0.05"]:
        outcome = "GENERALISES"
    else:
        outcome = "MIXED"
    return {"outcome": outcome, "checks": checks, "dpenalty": d, "late_minus_8": gap}


M1_NEXT = {"FINE-TUNE BIAS": "the next version must train varied option counts (2-26)",
           "GENERALISES": "raising the cap plus a light mix suffices", "MIXED": "neither reading holds (report the parts)"}


def m1_outcome(readings: Mapping[str, Any], trained: Sequence[str]) -> dict[str, Any]:
    """The rule over the trained models: FINE-TUNE BIAS iff any trained model reads FINE-TUNE BIAS; GENERALISES iff every trained model
    reads GENERALISES; otherwise MIXED (the per-model readings are kept)."""
    per_model = {m: m1_model_outcome(readings["paired_vs_base"][m], readings["models"][m]) for m in trained}
    names = [r["outcome"] for r in per_model.values()]
    if any(n == "FINE-TUNE BIAS" for n in names):
        outcome = "FINE-TUNE BIAS"
    elif names and all(n == "GENERALISES" for n in names):
        outcome = "GENERALISES"
    else:
        outcome = "MIXED"
    return {"outcome": outcome, "next": M1_NEXT[outcome], "per_model": per_model,
            "aggregation": "FINE-TUNE BIAS if any trained model shows it; GENERALISES only if every trained model does; else MIXED"}


# --------------------------------------------------------------------------
# M2 sheets
# --------------------------------------------------------------------------

_ROLE = (
    "ROLE",
    "- Judgment module (System 1) for the robot described under CURRENT SITUATION.",
    "- A planner picks routes and a controller runs the motors; this module answers questions about the next few seconds.",
    "- This sheet is assembled by the onboard reporter and refreshed about twice a second.",
)
_PLAN = (
    "LOCAL PLAN",
    "- Step in progress: the one described under CURRENT SITUATION.",
    "- Planner recommendation: none. The planner has no preference on the question asked and leaves it to this module.",
    "- Plan source: onboard planner, refreshed 0.4 s ago.",
)
_STATUS = (
    "- Battery: {battery} %, discharging, about {hours} h {minutes} min of work left.",
    "- Motor temperatures: normal, warmest joint {temp} °C.",
    "- Localisation: good, map match {match}.",
    "- Network: connected to access point {ap}, round trip {latency} ms.",
    "- Safety circuit: armed and clear; bumpers clear.",
    "- Last self-check: passed {selfcheck} min ago.",
    "- Operating mode: autonomous; remote supervisor {supervisor} on call.",
    "- Onboard clock: {clock}.",
)
_OBJECTS = (
    "potted plant", "wall clock", "fire extinguisher on the wall", "exit sign", "closed door", "office chair", "waste bin", "recycling bin",
    "notice board", "coat rack", "window", "ceiling light", "wall-mounted screen", "bookshelf", "folded ladder against the wall",
    "umbrella stand", "printer", "parked trolley", "bench", "row of chairs", "standing lamp", "wall poster", "vending machine",
    "first-aid cabinet", "radiator", "electrical cabinet", "hand sanitiser station", "pillar", "wall socket",
    "framed picture", "light switch", "air vent", "rolled-up rug against the wall", "noticeboard stand",
)
_PEOPLE = ("person standing", "person seated", "person walking", "two people talking", "staff member at a desk", "visitor reading a sign")
_SIDES = ("left", "right", "far left", "far right", "behind", "behind left", "behind right")
_LOG = (
    "route segment {k} completed on schedule",
    "map tile {k} refreshed from the site server",
    "network handover to access point {ap}",
    "operator {supervisor} acknowledged the shift start",
    "ambient noise level {noise} dB",
    "camera exposure adjusted for the room lighting",
    "door {k} on the site map reported open, then closed",
    "battery reading {battery} %",
    "lidar health check passed",
    "time synchronised with the site server (offset {offset} ms)",
    "software heartbeat received from the fleet manager",
    "no new obstacles reported on the site map",
    "status report sent to the fleet manager",
    "wheel odometry recalibrated against the map",
    "speaker volume set to {volume} %",
    "task queue checked: {k} tasks waiting",
    "temperature sensor on the main board reads {temp} °C",
    "log upload to the site server finished",
    "front camera lens check passed",
    "person detector model reloaded",
)


def _clock(rng: random.Random) -> int:
    return rng.randrange(8 * 3600, 18 * 3600)


def _hms(seconds: int) -> str:
    seconds %= 24 * 3600
    return f"{seconds // 3600:02d}:{(seconds % 3600) // 60:02d}:{seconds % 60:02d}"


def _filler(key: str, *, tracks: int = 400, logs: int = 400) -> dict[str, Any]:
    """The seeded filler of one order key: status lines, a long track list and a long log (most recent first)."""
    rng = _rng("iface-m2-filler", key)
    now = _clock(rng)
    values = {"battery": rng.randrange(35, 96), "hours": rng.randrange(1, 6), "minutes": rng.randrange(0, 60), "temp": rng.randrange(34, 52),
              "match": f"{rng.uniform(0.90, 0.99):.2f}", "ap": f"AP-{rng.randrange(1, 40):02d}", "latency": rng.randrange(8, 60),
              "selfcheck": rng.randrange(5, 90), "supervisor": rng.choice(("Kim", "Lee", "Park", "Choi", "Jung", "Kang", "Cho", "Yoon")),
              "clock": _hms(now)}
    status = ["ROBOT STATUS"] + [line.format(**values) for line in _STATUS]
    track_lines = []
    for t in range(tracks):
        tid = 100 + t * 3 + rng.randrange(3)
        if rng.random() < 0.2:
            what = rng.choice(_PEOPLE)
            motion = rng.choice(("moving away", "static", "static"))
        else:
            what = rng.choice(_OBJECTS)
            motion = "static"
        track_lines.append(f"- track {tid}: {what}, {rng.uniform(3.0, 15.0):.1f} m {rng.choice(_SIDES)}, {motion}, confidence {rng.uniform(0.6, 0.99):.2f}")
    log_lines = []
    clock = now
    for _ in range(logs):
        clock -= rng.randrange(4, 75)
        event = rng.choice(_LOG).format(k=rng.randrange(1, 30), ap=f"AP-{rng.randrange(1, 40):02d}", supervisor=values["supervisor"],
                                        noise=rng.randrange(38, 66), battery=values["battery"] + rng.randrange(0, 4), offset=rng.randrange(1, 9),
                                        volume=rng.randrange(20, 80), temp=rng.randrange(35, 55))
        log_lines.append(f"- {_hms(clock)} {event}")
    return {"status": status, "tracks": track_lines, "logs": log_lines}


def _sheet(state: str, filler: Mapping[str, Any], n_tracks: int, n_logs: int) -> str:
    parts = ["\n".join(_ROLE), "\n".join(filler["status"]),
             "\n".join(["CAMERA TRACKS (front and side cameras, last 2 s; nothing listed is within 3 m of the robot)"] + list(filler["tracks"][:n_tracks])),
             "CURRENT SITUATION\n" + state.strip(),
             "\n".join(["EARLIER OBSERVATIONS (oldest first)"] + list(reversed(filler["logs"][:n_logs]))),
             "\n".join(_PLAN)]
    return "\n\n".join(parts)


def wrap_state(state: str, *, key: str, target: int, reference_length: int | None = None) -> str:
    """The bench-like sheet around `state` (module docstring) at ≈ `target` characters: at least one track and one log line, then a track
    line and a log line in turn while the sheet stays at or under `target` (a kind whose next line would pass it is done). The line counts
    are decided for a state of `reference_length` characters (default: this state's) — the longest state of a one-fact pair, so both
    records of the pair get exactly the same filler."""
    filler = _filler(key)
    reference = "x" * max(len(state.strip()), int(reference_length or 0))
    counts = [1, 1]  # track lines, log lines
    done = [False, False]
    turn = 0
    while not all(done):
        kind = turn % 2
        turn += 1
        if done[kind]:
            continue
        trial = list(counts)
        trial[kind] += 1
        if trial[0] > len(filler["tracks"]) or trial[1] > len(filler["logs"]):
            raise ValueError(f"filler too short for {target} characters")
        if len(_sheet(reference, filler, *trial)) > target:
            done[kind] = True
            continue
        counts = trial
    return _sheet(state, filler, *counts)


def wrap_record(record: Mapping[str, Any], *, length: str, reference_length: int | None = None) -> dict[str, Any]:
    """The record with its state wrapped at `length` (:data:`LENGTHS`); id, meta (incl. the pair), questions and labels unchanged. The
    filler key is the record's order key and the line counts follow `reference_length` (the pair's longest state), so a one-fact pair
    shares its filler exactly."""
    out = {k: v for k, v in record.items()}
    request = dict(record["request"])
    request["state"] = wrap_state(str(record["request"]["state"]), key=shuffle_key(dict(record)), target=LENGTHS[length], reference_length=reference_length)
    out["request"] = request
    return out


def pair_reference_lengths(records: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    """Record id → the longest (stripped) state among the records sharing its order key (a one-fact pair; a lone record: its own)."""
    longest: dict[str, int] = defaultdict(int)
    for record in records:
        key = shuffle_key(dict(record))
        longest[key] = max(longest[key], len(str(record["request"]["state"]).strip()))
    return {str(r["id"]): longest[shuffle_key(dict(r))] for r in records}


#: Words that must not appear in a sheet outside its CURRENT SITUATION section (the filler is about nothing the questions ask).
FILLER_FORBIDDEN = (
    "load", "deck", "tray", "grip", "gripper", "jaw", "palm", "glove", "box", "carton", "glass", "cup", "drink", "tea", "water", "spill",
    "wet", "dry", "slip", "slide", "skid", "topple", "tip", "fall", "drop", "ramp", "slope", "tilt", "pitch", "brake", "speed", "accelerat",
    "decelerat", "friction", "surface", "floor", "stack", "cargo", "carry", "payload", "push", "press", "lift", "turn", "halt", "stop",
)


def sheet_sections(sheet: str) -> dict[str, str]:
    """A wrapped state's sections by header line (the first line of each blank-line-separated block)."""
    out = {}
    for block in sheet.split("\n\n"):
        head, _, body = block.partition("\n")
        out[head.split(" (")[0]] = body
    return out


def sheet_problems(sheet: str, state: str) -> list[str]:
    """Checks of one wrapped state: the original state appears exactly once and is the whole CURRENT SITUATION section; the six sections in
    order; no filler line uses a word of :data:`FILLER_FORBIDDEN` (as a word start)."""
    problems = []
    if sheet.count(state.strip()) != 1:
        problems.append("the original state does not appear exactly once")
    sections = sheet_sections(sheet)
    names = ["ROLE", "ROBOT STATUS", "CAMERA TRACKS", "CURRENT SITUATION", "EARLIER OBSERVATIONS", "LOCAL PLAN"]
    if list(sections) != names:
        problems.append(f"sections {list(sections)}")
    elif sections["CURRENT SITUATION"] != state.strip():
        problems.append("CURRENT SITUATION is not the original state verbatim")
    filler = "\n".join(body for name, body in sections.items() if name != "CURRENT SITUATION").lower()
    found = sorted({w for w in FILLER_FORBIDDEN if re.search(r"\b" + re.escape(w), filler)})
    if found:
        problems.append(f"filler mentions {found}")
    if not re.search(r"^- track ", sections.get("CAMERA TRACKS", ""), flags=re.MULTILINE) or not re.search(r"^- \d\d:\d\d:\d\d ", sections.get("EARLIER OBSERVATIONS", ""), flags=re.MULTILINE):
        problems.append("no track line or no log line")
    return problems


def select_physics(items: Sequence[Mapping[str, Any]], count: int, *, seed: int) -> list[dict[str, Any]]:
    """One item per pair group (seeded), then `count` items spread evenly over the scenes (a scene with too few gives all it has and the
    others fill the rest), in a seeded order."""
    rng = _rng("iface-m1-physics-select", seed)
    by_group: dict[str, list] = defaultdict(list)
    for item in items:
        by_group[item["group"]].append(item)
    one = [sorted(members, key=lambda x: x["item"])[rng.randrange(len(members))] for _, members in sorted(by_group.items())]
    by_scene: dict[str, list] = defaultdict(list)
    for item in one:
        by_scene[item["scene"]].append(item)
    for members in by_scene.values():
        rng.shuffle(members)
    scenes = sorted(by_scene)
    chosen: list = []
    quota = {s: count // len(scenes) + (1 if i < count % len(scenes) else 0) for i, s in enumerate(scenes)}
    for s in scenes:
        chosen += by_scene[s][: quota[s]]
    rest = [item for s in scenes for item in by_scene[s][quota[s]:]]
    rng.shuffle(rest)
    chosen += rest[: max(0, count - len(chosen))]
    if len(chosen) < count:
        raise ValueError(f"only {len(chosen)} physics items for {count}")
    return [dict(item) for item in chosen]


# --------------------------------------------------------------------------
# M2 reading
# --------------------------------------------------------------------------

#: M2 thresholds (rule text).
GAP_DELTA, NO_GAP_BAND = 0.05, 0.03
#: The gate metrics and the direction that counts as harm (sign of wrapped − short).
GATE = {"direction": -1, "comparison": -1, "DC": +1}
M2_NEXT = {"LONG-STATE GAP": "the next version needs long, sectioned states in training", "NO GAP": "no long-state change needed (rule text)",
           "MIXED": "neither reading holds (report the parts)"}


def m2_model_reading(deltas: Mapping[str, Mapping[str, Mapping[str, Any]]]) -> dict[str, Any]:
    """One trained model: `deltas[length][metric]` = paired wrapped − short (``value``, ``low``, ``high``). A length triggers the gap if
    direction or comparison drops ≥ 0.05, or DC rises ≥ 0.05, with the interval excluding 0; every gate delta within ±0.03 (points)
    counts towards NO GAP."""
    triggers, within = [], True
    checks: dict[str, Any] = {}
    for length, metrics in deltas.items():
        for metric, sign in GATE.items():
            d = metrics.get(metric) or {}
            value, low, high = d.get("value"), d.get("low"), d.get("high")
            if value is None:
                checks[f"{length}:{metric}"] = {"value": None, "harm": False, "within_0.03": False}
                within = False
                continue
            excludes = low is not None and high is not None and (low > 0 or high < 0)
            harm = (sign * value >= GAP_DELTA) and excludes
            ok = abs(value) <= NO_GAP_BAND
            within &= ok
            checks[f"{length}:{metric}"] = {"value": value, "low": low, "high": high, "harm": bool(harm), "within_0.03": bool(ok)}
            if harm:
                triggers.append(f"{length}:{metric}")
    return {"triggers": triggers, "all_within_0.03": bool(within), "checks": checks}


def m2_outcome(per_model: Mapping[str, Mapping[str, Mapping[str, Mapping[str, Any]]]]) -> dict[str, Any]:
    """The rule over the trained models: LONG-STATE GAP iff any trained model triggers at any length; NO GAP iff every gate delta of every
    trained model at every length is within ±0.03; otherwise MIXED."""
    readings = {m: m2_model_reading(d) for m, d in per_model.items()}
    if any(r["triggers"] for r in readings.values()):
        outcome = "LONG-STATE GAP"
    elif readings and all(r["all_within_0.03"] for r in readings.values()):
        outcome = "NO GAP"
    else:
        outcome = "MIXED"
    return {"outcome": outcome, "next": M2_NEXT[outcome], "per_model": readings,
            "aggregation": "LONG-STATE GAP if any trained model shows a harmful gate delta at any length; NO GAP only if every gate delta "
                           "(direction, comparison, DC) of every trained model at every length is within +-0.03; else MIXED"}
