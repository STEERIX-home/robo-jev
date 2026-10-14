"""Author A2 — overlap of the A2 records with the existing sets, in code only.

Other sets are read here and reduced to hashes, sentence sets and word 5-gram sets; nothing of theirs is printed or written. The
report keeps counts and, where something is shared, the shared fragments **of A2's own text**.

* ids, request hashes (sha256 of the request JSON with sorted keys), state hashes and instruction hashes — 0 shared allowed;
* sentences (A2's states and question texts split at . ? ! and line breaks, lower-cased, whitespace collapsed) — 0 shared allowed;
* word 5-grams (lower-cased words) of A2's text against T8b, T9 and the H-set — 0 shared allowed.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

__all__ = ["COMPARE_SETS", "NGRAM_SETS", "compare", "ngrams", "sentences", "texts"]

#: Name → path (relative to the repository) of every set the A2 records are compared with.
COMPARE_SETS = {
    **{f"T{i}": f"artifacts/simgen/v0b/tests/T{i}.jsonl" for i in range(1, 8)},
    "G1": "artifacts/simgen/probes/G1.jsonl",
    "G2": "artifacts/simgen/probes/G2.jsonl",
    "T8a": "artifacts/t8/T8a.jsonl",
    "T8b": "artifacts/t8/T8b.jsonl",
    "T9": "artifacts/t9/T9.jsonl",
    "B1": "artifacts/b1/b1-infamily.jsonl",
    "Hset": "artifacts/simgen/v04/scene-holdout.jsonl",
    "v04train": "artifacts/simgen/v04/train-v04.jsonl",
}
#: The sets the 5-gram rule applies to.
NGRAM_SETS = ("T8b", "T9", "Hset")
_WORD = re.compile(r"[a-z0-9]+(?:['’][a-z]+)?")
_SPLIT = re.compile(r"(?<=[.?!])\s+|\n+")


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def texts(record: Mapping[str, Any]) -> list[str]:
    """Every natural-language string of a record's request: the state, each question's instructions and its criteria."""
    request = record.get("request") or {}
    out = [str(request.get("state") or "")]
    for question in (request.get("questions") or {}).values():
        out.append(str(question.get("instructions") or ""))
        criteria = question.get("criteria")
        if isinstance(criteria, Mapping):
            out += [str(v) for v in criteria.values()]
        elif isinstance(criteria, list):
            out += [str(v) for v in criteria]
    return [t for t in out if t.strip()]


def sentences(text: str) -> set[str]:
    out = set()
    for part in _SPLIT.split(text):
        norm = " ".join(part.lower().split())
        if _WORD.search(norm):
            out.add(norm)
    return out


def ngrams(text: str, n: int = 5) -> set[tuple[str, ...]]:
    words = _WORD.findall(text.lower())
    return {tuple(words[i : i + n]) for i in range(len(words) - n + 1)}


def _profile(records: Iterable[Mapping[str, Any]], *, with_ngrams: bool) -> dict[str, Any]:
    ids, req, state, instr, sent, grams = set(), set(), set(), set(), set(), set()
    for record in records:
        ids.add(str(record.get("id")))
        request = record.get("request") or {}
        req.add(_sha(json.dumps(request, sort_keys=True, ensure_ascii=False)))
        state.add(_sha(str(request.get("state") or "")))
        for question in (request.get("questions") or {}).values():
            instr.add(_sha(str(question.get("instructions") or "")))
        for text in texts(record):
            sent |= sentences(text)
            if with_ngrams:
                grams |= ngrams(text)
    return {"ids": ids, "requests": req, "states": state, "instructions": instr, "sentences": sent, "ngrams": grams}


def _read(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def compare(mine: Sequence[Mapping[str, Any]], repo: Path, sets: Mapping[str, str] = COMPARE_SETS) -> dict[str, Any]:
    """Counts of shared ids / hashes / sentences / 5-grams per set, and A2's own shared fragments (sentences, 5-grams)."""
    own = _profile(mine, with_ngrams=True)
    own_sentence_hashes = {_sha(s): s for s in own["sentences"]}
    out: dict[str, Any] = {"a2": {"records": len(mine), "sentences": len(own["sentences"]), "ngrams": len(own["ngrams"])}, "sets": {}}
    for name, rel in sets.items():
        path = repo / rel
        if not path.is_file():
            out["sets"][name] = {"missing": rel}
            continue
        records = _read(path)
        theirs = _profile(records, with_ngrams=name in NGRAM_SETS)
        their_sentence_hashes = {_sha(s) for s in theirs["sentences"]}
        shared_sentences = sorted(own_sentence_hashes[h] for h in set(own_sentence_hashes) & their_sentence_hashes)
        row: dict[str, Any] = {"records": len(records), "shared_ids": len(own["ids"] & theirs["ids"]),
                               "shared_requests": len(own["requests"] & theirs["requests"]),
                               "shared_states": len(own["states"] & theirs["states"]),
                               "shared_instructions": len(own["instructions"] & theirs["instructions"]),
                               "shared_sentences": len(shared_sentences), "a2_shared_sentences": shared_sentences}
        if name in NGRAM_SETS:
            shared = sorted(" ".join(g) for g in own["ngrams"] & theirs["ngrams"])
            row["shared_5grams"] = len(shared)
            row["a2_shared_5grams"] = shared
        out["sets"][name] = row
    return out
