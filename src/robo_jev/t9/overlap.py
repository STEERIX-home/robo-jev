"""T9 overlap — T9 against every training input and T8b (Task T9 §1-3), streamed so the large files are never held in memory.

For every compared file each record is reduced to the units below and only *membership in T9's own sets* is tested (T9 is small),
so the training files are read once, line by line, and nothing from them is kept but counts and a few example hashes:

* **ids** — record ids equal to a T9 id;
* **hashes** — sha256 of the whole request (canonical JSON), of the state text and of every question instruction / option text /
  scale text (whitespace-normalised);
* **sentences** — every state, instruction and option text split into sentences, lower-cased with whitespace collapsed: exact
  matches, and matches after every number is replaced by ``<n>`` (the same template with other numbers);
* **5-grams** (against T8b and the MuJoCo training set — the sets written like T9, outside the engine's templates) — lower-cased word
  5-grams with numbers as ``<n>``; 4-grams are reported, not judged (runs such as "<n> m s and" are not wording);
* **catalog words** — every T9 catalog name (items, surfaces, straps, panels, …) searched as a phrase in T8b's and the MuJoCo set's
  texts (reported).

The verdict requires zero ids, hashes and exact sentences against every file, zero number-normalised sentences against T8b and the
MuJoCo set, and zero 5-grams against T8b and the MuJoCo set.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

__all__ = ["NGRAM_WRITTEN_LIKE_T9", "T9Units", "catalog_names", "compare_file", "ngrams", "sentences", "units_of", "verdict"]

_NUM = re.compile(r"[-+]?\d+(?:[.,]\d+)?")
_WORD = re.compile(r"<n>|[a-z]+(?:['’-][a-z]+)*")
_SPLIT = re.compile(r"(?<=[.!?;:])\s+|\n+")
#: Files whose 5-grams are judged (sets written like T9 — outside the engine's templates).
NGRAM_WRITTEN_LIKE_T9 = ("T8b", "mujoco-train")


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _norm(text: str) -> str:
    return " ".join(str(text).lower().split())


def _template(text: str) -> str:
    return _NUM.sub("<n>", _norm(text))


def sentences(text: str) -> list[str]:
    return [part.strip() for part in _SPLIT.split(str(text)) if part.strip()]


def ngrams(text: str, n: int) -> set[tuple[str, ...]]:
    words = _WORD.findall(_NUM.sub(" <n> ", str(text).lower()))
    return {tuple(words[i : i + n]) for i in range(len(words) - n + 1)}


def _texts(record: dict[str, Any]) -> tuple[str, list[str]]:
    request = record.get("request") or {}
    state = str(request.get("state") or "")
    pieces: list[str] = []
    for question in (request.get("questions") or {}).values():
        if not isinstance(question, dict):
            continue
        pieces.append(str(question.get("instructions") or ""))
        criteria = question.get("criteria")
        if isinstance(criteria, dict):
            pieces += [str(v) for v in criteria.values()]
        elif isinstance(criteria, list):
            pieces += [str(v) for v in criteria]
    return state, [p for p in pieces if p.strip()]


def _request_hash(record: dict[str, Any]) -> str:
    return _sha(json.dumps(record.get("request"), ensure_ascii=False, sort_keys=True, separators=(",", ":")))


def units_of(record: dict[str, Any]) -> dict[str, Any]:
    state, pieces = _texts(record)
    sents = [s for text in [state, *pieces] for s in sentences(text)]
    return {"id": str(record.get("id")), "request": _request_hash(record), "state": _sha(_norm(state)),
            "pieces": {_sha(_norm(p)) for p in pieces}, "sentences": {_norm(s) for s in sents}, "templates": {_template(s) for s in sents},
            "texts": [state, *pieces]}


class T9Units:
    """T9's own sets (ids, hashes, sentences, n-grams)."""

    def __init__(self, records: Sequence[dict[str, Any]], sizes: Iterable[int] = (4, 5, 6)):
        self.ids: set[str] = set()
        self.requests: set[str] = set()
        self.states: set[str] = set()
        self.pieces: set[str] = set()
        self.sentences: set[str] = set()
        self.templates: set[str] = set()
        self.sizes = tuple(sizes)
        self.grams: dict[int, set[tuple[str, ...]]] = {n: set() for n in self.sizes}
        for record in records:
            u = units_of(record)
            self.ids.add(u["id"])
            self.requests.add(u["request"])
            self.states.add(u["state"])
            self.pieces |= u["pieces"]
            self.sentences |= u["sentences"]
            self.templates |= u["templates"]
            for text in u["texts"]:
                for n in self.sizes:
                    self.grams[n] |= ngrams(text, n)


def _lines(path: Path) -> Iterator[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def compare_file(t9: T9Units, name: str, path: Path, *, grams: bool, names: Sequence[str] = ()) -> dict[str, Any]:
    """One file against T9 (module docstring)."""
    counts: Counter = Counter()
    shared_sentences: set[str] = set()
    shared_templates: set[str] = set()
    shared_grams: dict[int, set[tuple[str, ...]]] = defaultdict(set)
    name_hits: Counter = Counter()
    lowered_names = [(n, re.compile(r"\b" + re.escape(n.lower()) + r"\b")) for n in names]
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for raw in handle:
            digest.update(raw)
    for record in _lines(path):
        counts["records"] += 1
        u = units_of(record)
        counts["ids"] += u["id"] in t9.ids
        counts["requests"] += u["request"] in t9.requests
        counts["states"] += u["state"] in t9.states
        counts["instruction_or_option_texts"] += len(u["pieces"] & t9.pieces)
        hit = u["sentences"] & t9.sentences
        shared_sentences |= hit
        shared_templates |= u["templates"] & t9.templates
        if grams or lowered_names:
            for text in u["texts"]:
                if grams:
                    for n in t9.sizes:
                        shared_grams[n] |= ngrams(text, n) & t9.grams[n]
                low = text.lower()
                for original, pattern in lowered_names:
                    if original.lower() in low and pattern.search(low):
                        name_hits[original] += 1
    out: dict[str, Any] = {
        "name": name, "path": str(path), "sha256": digest.hexdigest(), "records": counts["records"],
        "shared": {"ids": counts["ids"], "requests": counts["requests"], "states": counts["states"],
                   "instruction_or_option_texts": counts["instruction_or_option_texts"], "sentences": len(shared_sentences),
                   "sentences_number_normalised": len(shared_templates)},
        "examples": {"sentences": sorted(shared_sentences)[:10], "sentences_number_normalised": sorted(shared_templates)[:10]},
    }
    if grams:
        out["ngrams"] = {str(n): {"t9": len(t9.grams[n]), "shared": len(shared_grams[n]),
                                  "share_of_t9": round(len(shared_grams[n]) / len(t9.grams[n]), 6) if t9.grams[n] else 0.0,
                                  "examples": [" ".join(g) for g in sorted(shared_grams[n])[:15]]} for n in t9.sizes}
    if lowered_names:
        out["catalog_names_found"] = dict(sorted(name_hits.items()))
    return out


def catalog_names() -> list[str]:
    """Every name in T9's catalogs (items, surfaces, straps, obstacles, boards, panels, faces, pumps)."""
    from robo_jev.t9 import families as F

    out: set[str] = set()
    for table in (F.ROLL_ITEMS, F.SWING_LOADS, F.KNOCK_OBJECTS, F.CONVEYOR_ITEMS, F.OVERHANG_ITEMS, F.SUCTION_PANELS):
        out |= {str(v["name"]) for v in table.values()}
    for table in (F.ROLL_SURFACES, F.SWING_STRAPS, F.KNOCK_SURFACES, F.CONVEYOR_BELTS, F.SUCTION_FACES, F.SUCTION_PUMPS):
        out |= {str(v["text"]) for v in table.values()}
    out |= {str(v["text"]) for v in F.OVERHANG_BOARDS.values()}
    out |= set(F.SWING_OBSTACLES)
    return sorted(out)


def verdict(reports: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Zero ids, hashes and exact sentences everywhere; zero number-normalised sentences and 5-grams against the sets written like T9."""
    checks: dict[str, bool] = {}
    for rep in reports:
        s = rep["shared"]
        checks[f"{rep['name']}: ids, hashes"] = s["ids"] == 0 and s["requests"] == 0 and s["states"] == 0 and s["instruction_or_option_texts"] == 0
        checks[f"{rep['name']}: sentences"] = s["sentences"] == 0
        if rep["name"] in NGRAM_WRITTEN_LIKE_T9:
            checks[f"{rep['name']}: number-normalised sentences"] = s["sentences_number_normalised"] == 0
            checks[f"{rep['name']}: 5-grams and longer"] = all(v["shared"] == 0 for n, v in rep.get("ngrams", {}).items() if int(n) >= 5)
    return {"checks": checks, "passed": all(checks.values())}
