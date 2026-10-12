"""T8a — PIQA validation as decision requests (Task T8 §1-1).

PIQA (Bisk et al. 2020, "PIQA: Reasoning about Physical Commonsense in Natural Language") gives a goal and two
solutions; exactly one works. Source files are the official validation split from the PIQA homepage
(``valid.jsonl`` — ``goal``, ``sol1``, ``sol2`` — and ``valid-labels.lst`` — 0 = sol1, 1 = sol2).

**License.** The Hugging Face dataset card (``ybisk/piqa``) lists the license as *unknown*; the PIQA homepage
(https://yonatanbisk.com/piqa/) states the Academic Free License v3.0 (AFL-3.0). Both are recorded in the manifest.

**Conversion** (one record per validation item, file order):

* ``request.state`` = ``"Goal: <goal>"`` — the goal as written, nothing else (no robot framing, no hint).
* one ``choice`` question ``solution`` with the instruction :data:`INSTRUCTIONS` and two options whose descriptions are
  the two solutions as written (whitespace collapsed to single spaces): keys ``sol1`` and ``sol2`` (the dataset's own
  field names, stored in that order).
* label: one-hot on the correct solution, ``kind: "probe"`` (scored in the ``choice/other`` block, like the public
  reading sets G1/G2 — accuracy = argmax on the correct solution).
* ``meta``: ``domain = "public:piqa"``, ``language = "en"``, ``origin_group = "public/piqa/<index>"``, ``split =
  "validation"``, ``source_index``, ``set = "T8a"``, and ``questions.solution`` (source type, family, label kind).

**Sealed.** Never used for training: no training configuration points at the T8 files, and the v0.2 training mix
deliberately excludes PIQA.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from robo_jev.decisions.records import MAX_STATE_CHARS, SCHEMA, validate_record

__all__ = [
    "CONVERTER_VERSION",
    "INSTRUCTIONS",
    "LICENSE",
    "QUESTION",
    "SOURCE",
    "convert",
    "piqa_record",
    "read_source",
]

CONVERTER_VERSION = "t8a-piqa-v1"
QUESTION = "solution"
INSTRUCTIONS = "Which solution achieves the goal?"
SOURCE = {
    "name": "PIQA",
    "split": "validation",
    "homepage": "https://yonatanbisk.com/piqa/",
    "files": {"valid.jsonl": "https://yonatanbisk.com/piqa/data/valid.jsonl", "valid-labels.lst": "https://yonatanbisk.com/piqa/data/valid-labels.lst"},
    "dataset_card": "https://huggingface.co/datasets/ybisk/piqa/raw/main/README.md",
    "citation": "Bisk, Zellers, Le Bras, Gao, Choi. PIQA: Reasoning about Physical Commonsense in Natural Language. AAAI 2020.",
}
LICENSE = {
    "dataset_card": "unknown (license: unknown; 'Licensing Information: Unknown')",
    "homepage": "Academic Free License v3.0 (AFL-3.0), stated on https://yonatanbisk.com/piqa/",
    "use_here": "evaluation only (sealed outside test); not redistributed in git — records live under artifacts/",
}


def _one_line(text: Any) -> str:
    return " ".join(str(text).split())


def read_source(rows_path: str | Path, labels_path: str | Path) -> list[tuple[dict[str, Any], int]]:
    """``valid.jsonl`` + ``valid-labels.lst`` → [(row, label)] in file order. The two files must have the same length."""
    rows = [json.loads(line) for line in Path(rows_path).read_text(encoding="utf-8").splitlines() if line.strip()]
    labels = [int(token) for token in Path(labels_path).read_text(encoding="utf-8").split()]
    if len(rows) != len(labels):
        raise ValueError(f"PIQA: {len(rows)} rows but {len(labels)} labels")
    if any(label not in (0, 1) for label in labels):
        raise ValueError("PIQA labels must be 0 or 1")
    return list(zip(rows, labels))


def piqa_record(index: int, row: dict[str, Any], label: int) -> dict[str, Any]:
    """One PIQA item → one ``simgen-request/0`` record (module docstring)."""
    goal, sol1, sol2 = (_one_line(row.get(key, "")) for key in ("goal", "sol1", "sol2"))
    if not goal or not sol1 or not sol2:
        raise ValueError(f"PIQA item {index}: empty goal or solution")
    state = f"Goal: {goal}"
    if len(state) > MAX_STATE_CHARS:
        raise ValueError(f"PIQA item {index}: state longer than {MAX_STATE_CHARS} characters")
    correct = "sol1" if int(label) == 0 else "sol2"
    rid = f"piqa-val-{index:05d}"
    record = {
        "schema": SCHEMA,
        "id": rid,
        "generator": {"converter": CONVERTER_VERSION, "source": "PIQA validation (valid.jsonl + valid-labels.lst)"},
        "meta": {
            "set": "T8a", "domain": "public:piqa", "language": "en", "origin_group": f"public/piqa/{index:05d}", "split": "validation",
            "source_record": rid, "source_index": int(index),
            "questions": {QUESTION: {"source_type": "choice", "source": "piqa/validation", "family": "piqa/validation", "label_kind": "single",
                                     "label_confidence": "high", "options_trimmed": None}},
        },
        "request": {
            "state": state,
            "questions": {QUESTION: {"type": "choice", "instructions": INSTRUCTIONS, "criteria": {"sol1": sol1, "sol2": sol2}}},
        },
        "labels": {QUESTION: {"type": "choice", "kind": "probe", "probabilities": {"sol1": 1.0 if correct == "sol1" else 0.0,
                                                                                   "sol2": 1.0 if correct == "sol2" else 0.0}}},
    }
    problems = validate_record(record)
    if problems:
        raise ValueError(f"PIQA item {index}: {problems}")
    return record


def convert(items: Iterable[tuple[dict[str, Any], int]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """[(row, label)] → (records, summary). Summary: counts, label balance, state/solution length stats."""
    records = [piqa_record(i, row, label) for i, (row, label) in enumerate(items)]
    n = len(records)
    sol2 = sum(1 for r in records if r["labels"][QUESTION]["probabilities"]["sol2"] == 1.0)
    lengths = [len(r["request"]["state"]) for r in records]
    sols = [len(text) for r in records for text in r["request"]["questions"][QUESTION]["criteria"].values()]
    return records, {
        "records": n, "label_sol1": n - sol2, "label_sol2": sol2,
        "state_chars": {"mean": sum(lengths) / n if n else None, "max": max(lengths) if n else None},
        "solution_chars": {"mean": sum(sols) / len(sols) if sols else None, "max": max(sols) if sols else None},
    }
