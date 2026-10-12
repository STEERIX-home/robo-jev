"""General-reading training mix from permissive public train splits (Task v0.2 §1-1).

**Why.** v0.1-mix reads physics well but lost general reading: on G2 (sealed BoolQ + MultiNLI sample) it is −0.061 below the
untrained 2B, mostly because it avoids "contradiction". v0.2 adds general multiple-choice reading data to the training mix. The
tasks must differ from G2 and from the outside test T8, so G2 stays an independent check:

* **no** BoolQ, MultiNLI, SNLI, ANLI or any other NLI set (G2), **no** PIQA (T8a), no boolean (yes/no) tasks at all — BoolQ is G2's
  yes/no half, so the general mix is choice-only;
* **permissive licences only** (MIT / Apache-2.0 / CC-BY), verified on the Hugging Face dataset card (metadata or the card's
  "Licensing Information" text) and recorded per source; no NC, no SA;
* **train splits only.**

**Sources** (:data:`SOURCES`, pinned revisions): CommonsenseQA (MIT), HellaSwag (MIT), Social IQa (CC-BY-4.0), QASC (CC-BY-4.0),
CosmosQA (CC-BY-4.0). Candidates left out and why are in :data:`EXCLUDED` (the manifest repeats them).

**Conversion** — the existing two-step path: a public row → a ``judgment-v0`` record (the shape of
:mod:`robo_jev.data.public_sets`; checked by :func:`robo_jev.contracts.validate_record`) → :func:`robo_jev.decisions.single_probe.convert_record`
→ a ``simgen-request/0`` decisions record that the diagnosis tool and the trainer read as is. Per row:

* **state** — the row's text fields as ``key: "<text>"`` lines (single_probe's lossless state text). Field names avoid G2's
  (``passage``/``premise``/``hypothesis``): CosmosQA's passage is ``context``.
* **one choice question** — instruction picked from three fixed wordings per source (seeded by source and row), options = the row's
  answers. The option order is **permuted with a seed per row** before the keys ``o1 … oN`` are given, so a key carries no
  information about the answer (the prompt then sorts the keys and shuffles them again for display).
* **label** — one-hot on the gold answer (``kind: probe`` — the label kind of converted non-robot choices; metrics put it in
  ``choice/other``).
* **dropped rows** (counted): a missing or unknown gold label, an empty option, two options with the same normalised text, an
  option count outside 2–8, a state over 16,000 characters, an exact duplicate (normalised state + options) of an earlier row.

**Overlap with sealed sets** (:func:`text_units`): every record is reduced to normalised text units — the whole state, every state
line value and list item with ≥ 5 words, every option with ≥ 5 words (instructions are fixed templates and are not units). A row is
dropped if any of its units occurs in G1, G2, T1–T7, T8 (every ``*.jsonl`` under ``artifacts/simgen/t8/`` when it exists) or the
PIQA validation strings (T8a's source, read only for this check). The manifest records the counts and the digests.

**Sampling and order.** Per source, a seeded random sample of ``per_source`` eligible rows (the sample order is random, not file
order). The file interleaves the sources round by round, each round in a seeded source order, so **any prefix is balanced across
sources** — the two training doses (v02-a ≈ 10 %, v02-b ≈ 25 % of examples) take prefixes of this one file
(``scripts/convert_single_probe.py mix --general``).

Entry point: ``scripts/convert_general_sets.py`` (downloads parquet with ``huggingface_hub``; reading parquet needs ``pyarrow``:
``uv run --with pyarrow``). This module imports neither ``pyarrow`` nor the network at import time.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

__all__ = [
    "EXCLUDED",
    "GENERAL_VERSION",
    "INSTRUCTIONS",
    "MIN_UNIT_WORDS",
    "SOURCES",
    "build_general",
    "convert_row",
    "convert_source",
    "interleave",
    "normalize",
    "raw_units",
    "record_units",
    "sealed_unit_sets",
    "text_units",
    "unit_digest",
]

GENERAL_VERSION = "general-sets-v0.2"
#: A state line value, list item or option counts as an overlap unit only from this many words on (shorter strings — "bank",
#: "yes" — would collide by chance).
MIN_UNIT_WORDS = 5

_HF = "https://huggingface.co/datasets"

#: The sources: Hugging Face dataset, pinned revision (for script datasets the parquet conversion ``refs/convert/parquet``), the
#: train file, the licence as the card states it, where the card states it, and the task.
SOURCES: dict[str, dict[str, Any]] = {
    "commonsense_qa": {
        "repo": "tau/commonsense_qa", "revision": "94630fe30dad47192a8546eb75f094926d47e155", "file": "data/train-00000-of-00001.parquet",
        "split": "train", "license": "MIT", "license_evidence": "card metadata 'license: mit' and card text 'The dataset is licensed under the MIT License'",
        "card": f"{_HF}/tau/commonsense_qa", "task": "commonsense question answering, 5 options",
        "citation": "Talmor et al. 2019, CommonsenseQA",
    },
    "hellaswag": {
        "repo": "Rowan/hellaswag", "revision": "218ec52e09a7e7462a5400043bb9a69a41d06b76", "file": "data/train-00000-of-00001.parquet",
        "split": "train", "license": "MIT", "license_evidence": "card text 'Licensing Information: MIT https://github.com/rowanz/hellaswag/blob/master/LICENSE'",
        "card": f"{_HF}/Rowan/hellaswag", "task": "plausible continuation of a described activity, 4 options",
        "citation": "Zellers et al. 2019, HellaSwag",
    },
    "social_i_qa": {
        "repo": "allenai/social_i_qa", "revision": "537a2ec8ec565adc0b70b70752893e59e024df26", "file": "default/train/0000.parquet",
        "split": "train", "license": "CC-BY-4.0", "license_evidence": "card text 'Licensing Information: CC-BY-4.0' (parquet conversion refs/convert/parquet of 8835ceb)",
        "card": f"{_HF}/allenai/social_i_qa", "task": "commonsense about people in a short situation, 3 options",
        "citation": "Sap et al. 2019, Social IQa",
    },
    "qasc": {
        "repo": "allenai/qasc", "revision": "a34ba204eb9a33b919c10cc08f4f1c8dae5ec070", "file": "data/train-00000-of-00001.parquet",
        "split": "train", "license": "CC-BY-4.0", "license_evidence": "card metadata 'license: cc-by-4.0' and card text 'released under CC BY 4.0'",
        "card": f"{_HF}/allenai/qasc", "task": "two given facts composed to answer a science question, 8 options",
        "citation": "Khot et al. 2020, QASC",
    },
    "cosmos_qa": {
        "repo": "allenai/cosmos_qa", "revision": "ed50fe83db62b355759053b1cd5eef344ee371e0", "file": "default/train/0000.parquet",
        "split": "train", "license": "CC-BY-4.0",
        "license_evidence": "card metadata 'license: cc-by-4.0' and card text 'licensed under CC BY 4.0' (parquet conversion refs/convert/parquet of 28d9d5e)",
        "card": f"{_HF}/allenai/cosmos_qa", "task": "reading a short everyday passage with commonsense, 4 options",
        "citation": "Huang et al. 2019, Cosmos QA",
    },
}

#: Candidates considered and left out (the manifest repeats this).
EXCLUDED: dict[str, str] = {
    "boolq": "G2 — the sealed general probe's yes/no half",
    "multinli / snli / anli / wnli / scitail / any NLI": "NLI family — G2's MultiNLI half must stay independent",
    "piqa": "reserved for the outside test T8 (T8a)",
    "boolean tasks (e.g. CommonsenseQA 2.0, StrategyQA)": "yes/no like BoolQ — kept out so G2's BoolQ half stays independent",
    "openbookqa": "the HF card states no licence (metadata 'unknown', 'More Information Needed'); upstream GitHub LICENSE is Apache-2.0, "
                  "but the card does not confirm it — left out under the card rule",
    "winogrande": "the HF card states no licence ('More Information Needed'); upstream GitHub LICENSE is Apache-2.0 — left out under the card rule",
    "ai2_arc": "CC-BY-SA-4.0 (share-alike)",
    "sciq": "CC-BY-NC-3.0 (non-commercial)",
    "race / dream / reclor": "non-commercial research licences",
}

#: Three instruction wordings per source (one is picked per row with a seed). None of them is a G2 wording.
INSTRUCTIONS: dict[str, tuple[str, ...]] = {
    "commonsense_qa": (
        "Which option answers the question best?",
        "Pick the answer that fits the question.",
        "Which choice is the most sensible answer to the question?",
    ),
    "hellaswag": (
        "Which ending continues the text most plausibly?",
        "How does the text most likely go on?",
        "Pick the most sensible continuation of the text.",
    ),
    "social_i_qa": (
        "Which answer fits the question about this situation best?",
        "Given the situation, which answer is most likely right?",
        "Pick the most fitting answer to the question about the people in the situation.",
    ),
    "qasc": (
        "Using the two facts, which option answers the question?",
        "Combine the facts: which choice answers the question?",
        "Which option follows from the facts as the answer to the question?",
    ),
    "cosmos_qa": (
        "Which answer to the question does the context support best?",
        "Based on the context, which answer is most likely right?",
        "Pick the answer to the question that fits the context best.",
    ),
}

_QUESTION_ID = "q_general"


def _seeded(*parts: Any) -> random.Random:
    digest = hashlib.sha256("|".join(str(p) for p in parts).encode("utf-8")).digest()
    return random.Random(int.from_bytes(digest[:8], "big"))


def normalize(text: str) -> str:
    """Lower case, every run of non-word characters → one space (works for English and Korean)."""
    return " ".join(re.sub(r"[\W_]+", " ", str(text).lower()).split())


def _clean(text: Any) -> str:
    return " ".join(str(text if text is not None else "").split())


class _Drop(Exception):
    """This row is not converted — the reason is args[0]."""


# --------------------------------------------------------------------------
# rows → judgment-v0
# --------------------------------------------------------------------------


def _fields(name: str, row: dict[str, Any]) -> tuple[dict[str, Any], list[str], int]:
    """(state fields, options in source order, index of the gold option) for one source row."""
    if name == "commonsense_qa":
        labels, texts = list(row["choices"]["label"]), list(row["choices"]["text"])
        key = str(row.get("answerKey") or "")
        if key not in labels:
            raise _Drop("no_gold")
        return {"question": _clean(row["question"])}, [_clean(t) for t in texts], labels.index(key)
    if name == "hellaswag":
        label = str(row.get("label") if row.get("label") is not None else "")
        if not label.isdigit():
            raise _Drop("no_gold")
        state = {"topic": _clean(row.get("activity_label")), "text": _clean(row.get("ctx"))}
        return state, [_clean(t) for t in row["endings"]], int(label)
    if name == "social_i_qa":
        label = str(row.get("label") or "")
        if label not in ("1", "2", "3"):
            raise _Drop("no_gold")
        return {"situation": _clean(row["context"]), "question": _clean(row["question"])}, [_clean(row[f"answer{c}"]) for c in "ABC"], int(label) - 1
    if name == "qasc":
        labels, texts = list(row["choices"]["label"]), list(row["choices"]["text"])
        key = str(row.get("answerKey") or "")
        if key not in labels:
            raise _Drop("no_gold")
        state = {"facts": [_clean(row.get("fact1")), _clean(row.get("fact2"))], "question": _clean(row["question"])}
        if not all(state["facts"]):
            raise _Drop("empty_field")
        return state, [_clean(t) for t in texts], labels.index(key)
    if name == "cosmos_qa":
        label = row.get("label")
        if not isinstance(label, int) or not 0 <= label <= 3:
            raise _Drop("no_gold")
        return {"context": _clean(row["context"]), "question": _clean(row["question"])}, [_clean(row[f"answer{i}"]) for i in range(4)], int(label)
    raise ValueError(f"unknown general source {name!r}")


def convert_row(name: str, index: int, row: dict[str, Any]) -> dict[str, Any]:
    """One source row → a ``judgment-v0`` record (one choice question, the options permuted with a per-row seed and keyed ``o1 … oN``).
    Raises ``_Drop`` (reason in ``args[0]``) for rows the module docstring drops."""
    state, options, gold = _fields(name, row)
    if not all(isinstance(v, list) or v for v in state.values()):
        raise _Drop("empty_field")
    if not 2 <= len(options) <= 8:
        raise _Drop("option_count")
    if not all(options):
        raise _Drop("empty_option")
    if len({normalize(o) for o in options}) != len(options):
        raise _Drop("duplicate_options")
    order = list(range(len(options)))
    _seeded(GENERAL_VERSION, name, index, "options").shuffle(order)
    keys = [f"o{i + 1}" for i in range(len(options))]
    criteria = [{"id": keys[pos], "description": options[source]} for pos, source in enumerate(order)]
    answer = keys[order.index(gold)]
    instruction = INSTRUCTIONS[name][_seeded(GENERAL_VERSION, name, index, "instruction").randrange(len(INSTRUCTIONS[name]))]
    rid = f"{name}-train-{index:05d}"
    return {
        "schema_version": "judgment-v0",
        "origin_group": f"public/{name}/train/{index:05d}",
        "split": "train",
        "request": {"request_id": rid, "state": state, "questions": [{"id": _QUESTION_ID, "type": "choice", "instructions": instruction, "criteria": criteria}]},
        "labels": [{"question_id": _QUESTION_ID, "kind": "single", "answer": answer, "source": f"{name}/train", "label_confidence": "high"}],
        "provenance": {"generator": GENERAL_VERSION, "domain": f"public:{name}", "public_source": name, "source_split": "train", "source_index": int(index),
                       "language": "en"},
    }  # fmt: skip


def convert_source(name: str, rows: Iterable[dict[str, Any]]) -> tuple[list[dict[str, Any]], Counter]:
    """All rows of one source → (decisions records in row order, counts: ``rows``, ``kept``, ``drop:<reason>``). Each record passes
    the judgment-v0 contract, then :func:`robo_jev.decisions.single_probe.convert_record`; exact duplicates (normalised state +
    options) of an earlier row are dropped."""
    from robo_jev.contracts import validate_record as validate_source
    from robo_jev.decisions.records import validate_record
    from robo_jev.decisions.single_probe import convert_record

    out: list[dict[str, Any]] = []
    counts: Counter = Counter()
    seen: set[str] = set()
    for index, row in enumerate(rows):
        counts["rows"] += 1
        try:
            source = convert_row(name, index, row)
        except _Drop as drop:
            counts[f"drop:{drop.args[0]}"] += 1
            continue
        validate_source(source)
        converted, found = convert_record(source)
        if len(converted) != 1:
            counts["drop:" + next((k[2:] for k in found if k.startswith(("r:", "q:"))), "not_converted")] += 1
            continue
        (record,) = converted
        problems = validate_record(record)
        if problems:
            raise ValueError(f"{record['id']}: {problems[:3]}")
        key = normalize(record["request"]["state"]) + "||" + "|".join(sorted(normalize(t) for t in record["request"]["questions"][_QUESTION_ID]["criteria"].values()))
        if key in seen:
            counts["drop:duplicate_row"] += 1
            continue
        seen.add(key)
        record["meta"]["source"] = {"set": name, "row": int(index)}
        out.append(record)
        counts["kept"] += 1
    return out, counts


# --------------------------------------------------------------------------
# overlap units
# --------------------------------------------------------------------------


def _strings(value: Any) -> Iterable[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from _strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from _strings(item)


_KEYED_LINE = re.compile(r"^(?:- )?([^:\n]{1,60}): (.*)$")


def _line_strings(line: str) -> list[str]:
    """A state line → its text values: ``key: <json>`` lines give the JSON's strings, ``- <json>`` list lines likewise, other lines
    are text themselves."""
    body = line[2:] if line.startswith("- ") else line
    match = _KEYED_LINE.match(line)
    for candidate in ((match.group(2),) if match else ()) + (body,):
        try:
            value = json.loads(candidate)
        except (ValueError, TypeError):
            continue
        strings = list(_strings(value))
        if strings or not isinstance(value, (int, float)):
            return strings
    return [match.group(2) if match else body]


def raw_units(strings: Iterable[str], *, min_words: int = MIN_UNIT_WORDS) -> set[str]:
    """Normalised strings with at least `min_words` words."""
    out = set()
    for text in strings:
        norm = normalize(text)
        if len(norm.split()) >= min_words:
            out.add(norm)
    return out


def text_units(state: str, options: Iterable[str] = ()) -> set[str]:
    """Overlap units of one request: the whole normalised state, every state line value / list item with ≥ 5 words, every option
    text with ≥ 5 words (module docstring)."""
    units = {normalize(state)} if normalize(state) else set()
    for line in str(state).split("\n"):
        units |= raw_units(_line_strings(line))
    units |= raw_units(options)
    return units


def record_units(record: dict[str, Any]) -> set[str]:
    """:func:`text_units` of a decisions record (``request.state`` + every choice option description)."""
    request = record.get("request") or {}
    options = []
    for question in (request.get("questions") or {}).values():
        criteria = question.get("criteria")
        if isinstance(criteria, dict):
            options.extend(str(v) for v in criteria.values())
    return text_units(str(request.get("state") or ""), options)


def _hash(unit: str) -> str:
    return hashlib.sha256(unit.encode("utf-8")).hexdigest()


def unit_digest(units: Iterable[str]) -> str:
    """sha256 over the sorted unit hashes — a fingerprint of a set's text units (manifest)."""
    digest = hashlib.sha256()
    for h in sorted({_hash(u) for u in units}):
        digest.update(h.encode("ascii") + b"\n")
    return digest.hexdigest()


def sealed_unit_sets(paths: dict[str, Path], extra: dict[str, set[str]] | None = None) -> dict[str, set[str]]:
    """Name → the set of unit **hashes** of a sealed JSONL file (decisions records, or any JSONL whose lines have ``request.state``),
    plus already-built raw unit sets in `extra` (hashed here)."""
    out: dict[str, set[str]] = {}
    for name, path in paths.items():
        hashes: set[str] = set()
        with Path(path).open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    hashes |= {_hash(u) for u in record_units(json.loads(line))}
        out[name] = hashes
    for name, units in (extra or {}).items():
        out[name] = {_hash(u) for u in units}
    return out


# --------------------------------------------------------------------------
# build
# --------------------------------------------------------------------------


def interleave(per_source: dict[str, list[dict[str, Any]]], *, seed: int) -> list[dict[str, Any]]:
    """Round-robin over the sources, each round in a seeded source order: any prefix holds about the same number of records of
    every source (a source that runs out simply stops contributing)."""
    names = sorted(per_source)
    rounds = max((len(v) for v in per_source.values()), default=0)
    out = []
    for r in range(rounds):
        order = list(names)
        _seeded(GENERAL_VERSION, seed, "round", r).shuffle(order)
        out.extend(per_source[name][r] for name in order if r < len(per_source[name]))
    return out


def build_general(
    rows: dict[str, Sequence[dict[str, Any]]], *, per_source: int, seed: int, sealed: dict[str, set[str]],
    log: Callable[[str], None] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Source rows → (the interleaved general records, a per-source report). `sealed` maps a sealed set's name to its unit hashes
    (:func:`sealed_unit_sets`); rows sharing any unit with any of them are dropped before sampling."""
    sealed_all = set().union(*sealed.values()) if sealed else set()
    chosen: dict[str, list[dict[str, Any]]] = {}
    report: dict[str, Any] = {}
    for name in sorted(rows):
        records, counts = convert_source(name, rows[name])
        eligible, hits = [], Counter()
        for record in records:
            hashes = {_hash(u) for u in record_units(record)}
            touched = [s for s, units in sealed.items() if hashes & units]
            if touched:
                for s in touched:
                    hits[s] += 1
                continue
            eligible.append(record)
        if per_source > len(eligible):
            raise ValueError(f"{name}: {per_source} records asked, only {len(eligible)} eligible")
        picked = _seeded(GENERAL_VERSION, seed, name, "sample").sample(eligible, per_source)
        chosen[name] = picked
        answers = Counter(next(k for k, p in r["labels"][_QUESTION_ID]["probabilities"].items() if p == 1.0) for r in picked)
        report[name] = {
            "rows": counts["rows"], "converted": counts["kept"], "dropped": {k[5:]: v for k, v in sorted(counts.items()) if k.startswith("drop:")},
            "overlap_dropped": dict(sorted(hits.items())), "eligible": len(eligible), "sampled": len(picked),
            "gold_key_counts": dict(sorted(answers.items())),
        }  # fmt: skip
        if log:
            log(f"{name}: {counts['rows']} rows → {counts['kept']} converted → {len(eligible)} eligible (overlap {dict(hits)}) → {len(picked)} sampled")
    records = interleave(chosen, seed=seed)
    report["_check"] = {"sealed_sets": sorted(sealed), "sealed_units": len(sealed_all)}
    return records, report
