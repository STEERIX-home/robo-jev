"""System-1 derived questions for engine training records (Task v0.3c §1-1).

robo-jev is a System 1: it judges risk levels, comparisons, the direction of a change and the safer action; exact numbers are
System 0's job (docs/12 §16, 2026-10-12). The engine labels every yes/no and score question with a posterior probability. This
transform adds **derived choice questions** to a training record that ask for those coarse judgments directly. The original
questions, their labels and every other field stay exactly as they are; derived questions are appended after them.

**Risk band** (``s1_band_<i>``, one per event question). An event question is a yes/no question or an event-probability score —
the questions :mod:`robo_jev.bench.system1` bands as events (``band_balanced:event``: every yes/no question and every score whose
recipe source is not ``quantity``). The derived question repeats the original instruction and asks for the band of its probability:
low < 1/3 ≤ medium ≤ 2/3 < high (:data:`robo_jev.bench.system1.BAND_EDGES`). The label is a soft distribution over the bands from the
label probability p: one-hot away from the edges, a linear ramp within :data:`SMOOTH` of an edge (at the edge itself 50/50), so a
borderline case is never hard-labelled. Its sum is exactly 1 (rounded to 6 decimals, the remainder on the largest entry).

**Comparison** (``s1_cmp_<i>_<j>``, one per pair). Two event questions of the record about the same event under different hypothetical
actions, paired by the engine's recipe identity exactly as :func:`robo_jev.bench.system1.basis_event_pairs` pairs them for the
comparison metric (same ``meta.basis`` source and event, different options). Only pairs whose two questions ask in the same polarity
(neither or both negated) are used, so "that outcome" means the same thing in both quoted questions (train-50k has no negated
question). The question quotes both originals as the *first* and *second* case — which one is first is a seeded coin per record and
pair (:func:`_first_is`), so the label is not tied to the engine's question order — and asks under which action the outcome is more
likely. The label comes from |Δp| (the two label probabilities): "about the same" when |Δp| < :data:`SAME_BELOW`, with the same linear
ramp of half-width :data:`SMOOTH` around it (|Δp| ≤ 0.05 → same; ≥ 0.15 → the larger side; 0.1 → 50/50).

**Budget.** A request holds at most 12 questions (the Decisions limit), so derived questions fill the room left after the originals in
this priority: the band of the record's key question (``meta.key_question`` — always an event question in train-50k), then the
comparisons (pair order of the request), then the other bands (request order). ``meta.system1.dropped`` counts what did not fit.
Derived questions share the trainer's choice slot with the original choice questions (the trainer picks two questions of different
types per record and epoch), so they displace action, state, rule and look-again examples — not yes/no or score ones;
:func:`exposure` counts this. An optional cap ``max_derived`` (off by default) limits the derived questions per record.

**Records.** The id stays (the trainer's dev split and pair keys are unchanged). Derived labels have ``kind: state`` (a distribution
over discrete answers, as the MuJoCo set's direction and comparison questions). ``meta.question_families`` names the derived questions'
families (``s1:band`` / ``s1:comparison`` — the diagnosis tool's per-family metrics read it); ``meta.system1`` records the transform
version, the derived questions (kind, source questions, label probabilities) and the budget drops. ``meta.basis`` is left as the
engine wrote it (derived questions have no engine recipe).

**Wording.** English and Korean frames (``meta.language``); new text is only the fixed frames below. A test checks them — and the whole
derived instructions of real records — against the held-out wording bans of T2 (:func:`robo_jev.simgen.phrasing.language_wide_hits`).

**Sealed sets.** Only training-split engine records are accepted (``meta.split`` and ``generator.split.name`` are ``train``); a record
from T1–T7 or any other split raises :class:`SealedRecordError`. Nothing here reads a sealed file.
"""

from __future__ import annotations

import copy
import hashlib
from collections import Counter
from typing import Any, Iterable, Mapping

from robo_jev.bench.system1 import BAND_EDGES, basis_event_pairs, band

__all__ = [
    "BAND_KEYS",
    "COMPARE_KEYS",
    "FRAMES",
    "MAX_QUESTIONS",
    "SAME_BELOW",
    "SMOOTH",
    "VERSION",
    "SealedRecordError",
    "band_label",
    "compare_label",
    "comparison_candidates",
    "derived_texts",
    "event_questions",
    "exposure",
    "plan_derived",
    "summarize",
    "transform_record",
    "with_pick_rule",
]

VERSION = "s1q-v1"
#: Decisions limit on questions per request.
MAX_QUESTIONS = 12
#: Half-width of the linear ramp around a band edge and around the "about the same" edge.
SMOOTH = 0.05
#: "About the same" below this |Δp|.
SAME_BELOW = 0.1
BAND_KEYS = ("low", "medium", "high")
COMPARE_KEYS = ("first", "second", "same")
_ROUND = 6

#: Fixed frames per language. ``{q}`` is the original instruction; ``{a}``/``{b}`` the first/second quoted question.
FRAMES: dict[str, dict[str, Any]] = {
    "en": {
        "band_noul": "{q} How likely is a yes? Pick one of three levels.",
        "band_score": "{q} Instead of a number, pick one of three levels.",
        "band_options": {"low": "less than 1 in 3", "medium": "between 1 in 3 and 2 in 3", "high": "more than 2 in 3"},
        "compare": ('Two what-if questions about this moment ask about the same outcome under different actions. First: "{a}" '
                    'Second: "{b}" Under which action is that outcome more likely?'),
        "compare_options": {"first": "the action in the first question", "second": "the action in the second question",
                            "same": "about equally likely (less than 0.1 apart)"},
    },
    "ko": {
        "band_noul": "{q} 답이 ‘예’일 가능성은 얼마나 됩니까? 세 단계 가운데 하나를 고르십시오.",
        "band_score": "{q} 수치 대신 세 단계 가운데 하나를 고르십시오.",
        "band_options": {"low": "3분의 1 미만", "medium": "3분의 1에서 3분의 2 사이", "high": "3분의 2 초과"},
        "compare": ('이 순간에 대해 두 가정 질문이 서로 다른 행동에서 같은 일을 묻습니다. 첫째: "{a}" 둘째: "{b}" '
                    '그 일은 어느 행동에서 더 일어나기 쉽습니까?'),
        "compare_options": {"first": "첫째 질문의 행동", "second": "둘째 질문의 행동", "same": "거의 같음(차이 0.1 미만)"},
    },
}


class SealedRecordError(ValueError):
    """A record that is not a training-split engine record (a sealed test record, or something else)."""


# --------------------------------------------------------------------------
# Labels
# --------------------------------------------------------------------------


def _normalise(probs: Mapping[str, float]) -> dict[str, float]:
    """Round to 6 decimals and put the rounding remainder on the largest entry, so the sum is exactly 1 (within 1e-9)."""
    out = {k: round(min(1.0, max(0.0, float(v))), _ROUND) for k, v in probs.items()}
    biggest = max(out, key=lambda key: out[key])
    out[biggest] = round(out[biggest] + 1.0 - sum(out.values()), _ROUND)
    return out


def _ramp(x: float, edge: float) -> float:
    """Share of the side above `edge`: 0 at or below edge − SMOOTH, 1 at or above edge + SMOOTH, linear between (0.5 at the edge)."""
    return min(1.0, max(0.0, (float(x) - (edge - SMOOTH)) / (2.0 * SMOOTH)))


def band_label(p: float) -> dict[str, float]:
    """Soft band label of a probability p (module docstring): low / medium / high."""
    p = float(p)
    if not 0.0 <= p <= 1.0:
        raise ValueError(f"a band label needs a probability in [0, 1] (got {p!r})")
    above_low, above_high = _ramp(p, BAND_EDGES[0]), _ramp(p, BAND_EDGES[1])
    return _normalise({"low": 1.0 - above_low, "medium": above_low - above_high, "high": above_high})


def compare_label(p_first: float, p_second: float) -> dict[str, float]:
    """Soft comparison label (module docstring): first / second / same from the two probabilities."""
    a, b = float(p_first), float(p_second)
    for value in (a, b):
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"a comparison label needs probabilities in [0, 1] (got {a!r}, {b!r})")
    apart = _ramp(abs(b - a), SAME_BELOW)
    larger = "second" if b > a else "first"
    probs = {"first": 0.0, "second": 0.0, "same": 1.0 - apart}
    probs[larger] += apart
    return _normalise(probs)


# --------------------------------------------------------------------------
# Which questions
# --------------------------------------------------------------------------


def _check_train(record: Mapping[str, Any]) -> None:
    meta = record.get("meta") or {}
    split = ((record.get("generator") or {}).get("split") or {}).get("name")
    if meta.get("split") != "train" or split != "train":
        raise SealedRecordError(f"{record.get('id')}: only training-split engine records are transformed (meta.split {meta.get('split')!r}, "
                                f"generator.split {split!r})")
    if (meta.get("system1") or {}).get("version"):
        raise ValueError(f"{record.get('id')}: already transformed ({meta['system1']['version']})")


def _probability(record: Mapping[str, Any], name: str) -> float:
    label = (record.get("labels") or {})[name]
    value = label.get("p_yes", label.get("value"))
    if value is None:
        raise ValueError(f"{record.get('id')}/{name}: no probability label")
    return float(value)


def event_questions(record: Mapping[str, Any]) -> list[str]:
    """Names of the record's event questions in request order: yes/no questions and event-probability scores (``meta.basis`` source not
    ``quantity`` — the questions :mod:`robo_jev.bench.system1` bands as events)."""
    basis = (record.get("meta") or {}).get("basis") or {}
    out = []
    for name, question in record["request"]["questions"].items():
        qtype = question.get("type")
        recipe = basis.get(name) or {}
        if qtype == "noul" or (qtype == "score" and recipe.get("source") not in (None, "quantity")):
            out.append(str(name))
    return out


def comparison_candidates(record: Mapping[str, Any]) -> list[tuple[str, str]]:
    """Pairs of event questions about the same event under different actions (:func:`robo_jev.bench.system1.basis_event_pairs`), same
    polarity only, in request order."""
    order = {name: i for i, name in enumerate(record["request"]["questions"])}
    out = []
    for _, (_, qa, na), (_, qb, nb) in basis_event_pairs([record]):
        if na != nb:
            continue
        out.append(tuple(sorted((qa, qb), key=order.__getitem__)))
    return sorted(set(out), key=lambda pair: (order[pair[0]], order[pair[1]]))


def _first_is(record_id: str, a: str, b: str) -> str:
    """Which of the pair is quoted first: a seeded coin per record and pair (deterministic)."""
    digest = hashlib.sha256(f"{VERSION}|first|{record_id}|{a}|{b}".encode("utf-8")).digest()
    return a if digest[0] % 2 == 0 else b


def plan_derived(record: Mapping[str, Any], *, max_derived: int | None = None) -> tuple[list[dict[str, Any]], Counter]:
    """The derived questions to add (``{"kind", "of"}`` in priority order, within the 12-question budget and at most `max_derived` when
    given — an optional cap, off by default) and the drops by kind."""
    questions = record["request"]["questions"]
    events = event_questions(record)
    key = (record.get("meta") or {}).get("key_question")
    wanted = []
    if key in events:
        wanted.append({"kind": "band", "of": [key]})
    rid = str(record.get("id"))
    for a, b in comparison_candidates(record):
        first = _first_is(rid, a, b)
        wanted.append({"kind": "comparison", "of": [first, b if first == a else a]})
    wanted += [{"kind": "band", "of": [name]} for name in events if name != key]
    room = max(0, MAX_QUESTIONS - len(questions))
    if max_derived is not None:
        room = min(room, int(max_derived))
    return wanted[:room], Counter(item["kind"] for item in wanted[room:])


# --------------------------------------------------------------------------
# Text
# --------------------------------------------------------------------------


def _frames(record: Mapping[str, Any]) -> dict[str, Any]:
    language = (record.get("meta") or {}).get("language") or "en"
    if language not in FRAMES:
        raise ValueError(f"{record.get('id')}: no System-1 frames for language {language!r}")
    return FRAMES[language]


def _instruction(record: Mapping[str, Any], name: str) -> str:
    return " ".join(str(record["request"]["questions"][name]["instructions"]).split())


def derived_texts(record: Mapping[str, Any], item: Mapping[str, Any]) -> dict[str, Any]:
    """The request entry (``type``, ``instructions``, ``criteria``) of one planned derived question."""
    frames = _frames(record)
    if item["kind"] == "band":
        (source,) = item["of"]
        frame = frames["band_noul"] if record["request"]["questions"][source]["type"] == "noul" else frames["band_score"]
        return {"type": "choice", "instructions": frame.format(q=_instruction(record, source)), "criteria": dict(frames["band_options"])}
    first, second = item["of"]
    text = frames["compare"].format(a=_instruction(record, first), b=_instruction(record, second))
    return {"type": "choice", "instructions": text, "criteria": dict(frames["compare_options"])}


# --------------------------------------------------------------------------
# Transform
# --------------------------------------------------------------------------


def _name(record: Mapping[str, Any], item: Mapping[str, Any]) -> str:
    index = {name: i for i, name in enumerate(record["request"]["questions"])}
    if item["kind"] == "band":
        return f"s1_band_{index[item['of'][0]]}"
    a, b = sorted(index[name] for name in item["of"])
    return f"s1_cmp_{a}_{b}"


def transform_record(record: Mapping[str, Any], *, max_derived: int | None = None) -> dict[str, Any]:
    """A training record → a new record with the derived questions appended (module docstring). The input is not modified.
    `max_derived` (optional, off by default): at most this many derived questions per record (same priority)."""
    _check_train(record)
    plan, dropped = plan_derived(record, max_derived=max_derived)
    out = copy.deepcopy(dict(record))
    questions, labels = out["request"]["questions"], out["labels"]
    derived: dict[str, Any] = {}
    families = dict(out["meta"].get("question_families") or {})
    for item in plan:
        name = _name(record, item)
        if name in questions:
            raise ValueError(f"{record.get('id')}: derived question name {name!r} clashes with an original question")
        probabilities = [_probability(record, source) for source in item["of"]]
        if item["kind"] == "band":
            label = band_label(probabilities[0])
        else:
            label = compare_label(*probabilities)
        questions[name] = derived_texts(record, item)
        labels[name] = {"type": "choice", "kind": "state", "probabilities": label}
        families[name] = f"s1:{item['kind']}"
        derived[name] = {"kind": item["kind"], "of": list(item["of"]), "p": probabilities}
    out["meta"]["question_families"] = families
    out["meta"]["system1"] = {"version": VERSION, "original_questions": len(record["request"]["questions"]), "derived": derived,
                              "dropped": dict(sorted(dropped.items())), "smooth": SMOOTH, "same_below": SAME_BELOW}
    if max_derived is not None:
        out["meta"]["system1"]["max_derived"] = int(max_derived)
    return out


# --------------------------------------------------------------------------
# Statistics
# --------------------------------------------------------------------------


def _argmax_set(probs: Mapping[str, float]) -> list[str]:
    top = max(probs.values())
    return [k for k, v in probs.items() if v == top]


def summarize(records: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Derived-question statistics of transformed records: counts per kind and language, per record, budget drops, label distributions
    (hard = one-hot, soft = split; mean label mass per answer; the band of p for bands)."""
    n = 0
    per_kind: Counter = Counter()
    per_lang: Counter = Counter()
    per_record: Counter = Counter()
    dropped: Counter = Counter()
    truncated = 0
    hard: dict[str, Counter] = {"band": Counter(), "comparison": Counter()}
    soft: Counter = Counter()
    mass: dict[str, Counter] = {"band": Counter(), "comparison": Counter()}
    p_band: Counter = Counter()
    sizes: Counter = Counter()
    for record in records:
        n += 1
        meta = record["meta"]
        info = meta.get("system1") or {}
        derived = info.get("derived") or {}
        per_record[len(derived)] += 1
        sizes[len(record["request"]["questions"])] += 1
        if info.get("dropped"):
            truncated += 1
            dropped.update(info["dropped"])
        for name, item in derived.items():
            kind = item["kind"]
            per_kind[kind] += 1
            per_lang[(meta.get("language") or "en", kind)] += 1
            probs = record["labels"][name]["probabilities"]
            mass[kind].update(probs)
            top = _argmax_set(probs)
            if max(probs.values()) >= 1.0 - 1e-9:
                hard[kind][top[0]] += 1
            else:
                soft[kind] += 1
            if kind == "band":
                p_band[band(item["p"][0])] += 1
    out: dict[str, Any] = {
        "records": n, "derived": dict(per_kind), "derived_total": sum(per_kind.values()),
        "derived_per_record_mean": sum(per_kind.values()) / n if n else None,
        "derived_per_record": dict(sorted(per_record.items())), "questions_per_record": dict(sorted(sizes.items())),
        "by_language": {f"{lang}:{kind}": count for (lang, kind), count in sorted(per_lang.items())},
        "records_truncated_by_budget": truncated, "dropped_by_budget": dict(sorted(dropped.items())),
        "labels": {},
    }
    for kind, keys in (("band", BAND_KEYS), ("comparison", COMPARE_KEYS)):
        total = per_kind[kind]
        out["labels"][kind] = {
            "hard": {k: hard[kind][k] for k in keys}, "soft": soft[kind],
            "hard_share": {k: hard[kind][k] / total for k in keys} if total else None,
            "soft_share": soft[kind] / total if total else None,
            "mean_mass": {k: mass[kind][k] / total for k in keys} if total else None,
        }
    out["labels"]["band"]["band_of_p"] = {k: p_band[k] for k in ("low", "middle", "high")}
    return out


def with_pick_rule(record: Mapping[str, Any]) -> dict[str, Any]:
    """Task v0.4b — the action repair as a data layout: a copy of a transformed record whose derived questions are recorded as the
    secondary questions of the trainer's pick (``meta.question_pick``, :func:`robo_jev.decisions.examples.record_pick`). The trainer then
    picks the record's q examples from the original questions exactly as for the untransformed record, and trains each derived question
    in addition in the epochs where the plain pick (the v0.3 pick) chooses it: the original questions get their v0.2 / v0.1-mix exposure
    back and the derived questions keep their v0.3 exposure. Nothing else changes (the field goes last in ``meta``)."""
    from robo_jev.decisions.examples import QUESTION_PICK_VERSION

    derived = list(((record.get("meta") or {}).get("system1") or {}).get("derived") or {})
    if not derived:
        raise ValueError(f"{record.get('id')}: no derived System-1 questions (meta.system1.derived) — nothing to record")
    if (record.get("meta") or {}).get("question_pick") is not None:
        raise ValueError(f"{record.get('id')}: already carries meta.question_pick")
    out = copy.deepcopy(dict(record))
    out["meta"]["question_pick"] = {"version": QUESTION_PICK_VERSION, "secondary": derived}
    return out


def exposure(records: Iterable[Mapping[str, Any]], *, q: int, seed: int = 0, epoch: int = 0) -> dict[str, int]:
    """Examples of one training epoch by question category under the trainer's own pick (:func:`robo_jev.decisions.examples.record_pick`
    with the trainer's seed stream — :func:`~robo_jev.decisions.examples.select_questions`, or the recorded pick rule when a record carries
    one) — what the derived questions displace when they share the choice slot."""
    from robo_jev.decisions.examples import record_pick

    counts: Counter = Counter()
    for record in records:
        names = record_pick(dict(record), q=q, seed=seed, epoch=epoch)
        basis = (record.get("meta") or {}).get("basis") or {}
        families = (record.get("meta") or {}).get("question_families") or {}
        for name in names:
            if name in families and str(families[name]).startswith("s1:"):
                counts[str(families[name])] += 1
                continue
            qtype = record["request"]["questions"][name]["type"]
            recipe = basis.get(name) or {}
            counts[f"{qtype}:{recipe.get('kind') if qtype == 'choice' else recipe.get('source')}"] += 1
    return dict(sorted(counts.items()))
