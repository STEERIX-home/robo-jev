#!/usr/bin/env python
"""Task v0.7b (M1g) — download the MMLU test split (Hendrycks et al. 2021) and write its rows as JSONL.

    PY=<venv with pyarrow + huggingface_hub> HF_HUB_OFFLINE=0 artifacts/scratch/v07b/cpu.sh fetch-mc scripts/v07_fetch_mc.py \\
        --out-dir artifacts/v07/gates/m1g/source

Standalone on purpose (no ``robo_jev`` import): reading parquet needs ``pyarrow``, which is not a project dependency, so this step runs in a
separate small venv (as ``scripts/iface_fetch_arc.py`` does for ARC). Only the **test** split of the ``all`` configuration is read (never
``auxiliary_train``, which repackages ARC, OpenBookQA, RACE and MCTest); the gate never trains on anything.

Why MMLU: M1g needs a multiple-choice source used in neither training nor M1 — not ARC (M1), not OpenBookQA (no licence on its card), not a
general-train source (CommonsenseQA, HellaSwag, Social IQa, QASC, CosmosQA), not G2's BoolQ / MultiNLI, not PIQA (T8a). MMLU's card states
``license: mit`` in its metadata and "Licensing Information: MIT License" in its text (upstream github.com/hendrycks/test LICENSE); the
moral-scenarios subject draws from ETHICS (also MIT) and is not used by the gate.

Writes ``<out-dir>/mmlu-test.jsonl`` (one row per question: ``id`` = ``mmlu-test-<row index>``, ``subject``, ``question``, ``choices``,
``answer`` = the index of the right choice) and ``<out-dir>/mmlu-test.fetch.json`` (repository, resolved revision, file sha256 and bytes, the
card's licence lines, row count, citation).
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import re
from pathlib import Path

REPO_ID = "cais/mmlu"
DEFAULT_REVISION = "main"
FILE = "all/test-00000-of-00001.parquet"
CARD = "README.md"
CITATION = "Hendrycks et al. 2021, Measuring Massive Multitask Language Understanding (ICLR 2021, arXiv:2009.03300)"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _revision_of(path: Path) -> str | None:
    parts = path.parts
    return parts[parts.index("snapshots") + 1] if "snapshots" in parts else None


def card_licence(card: str) -> dict:
    """The licence lines of a dataset card: the YAML ``license:`` entries and the "Licensing Information" section's first line."""
    head = card.split("\n---", 1)[0]
    found = re.search(r"^license:[ \t]*(\S*)[ \t]*\n((?:[ \t]*-[ \t]*\S+[ \t]*\n)*)", head + "\n", flags=re.MULTILINE)
    metadata = ([found.group(1)] if found and found.group(1) else []) + (re.findall(r"-[ \t]*(\S+)", found.group(2)) if found else [])
    section = re.search(r"#+\s*Licensing Information\s*\n+([^\n]+)", card)
    return {"metadata": metadata, "text": section.group(1).strip() if section else None}


def main(argv: list[str] | None = None) -> int:
    from huggingface_hub import hf_hub_download
    import pyarrow.parquet as pq

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    args = parser.parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    cache = args.out_dir / "hf-cache"
    card_path = Path(hf_hub_download(REPO_ID, CARD, repo_type="dataset", revision=args.revision, cache_dir=str(cache)))
    licence = card_licence(card_path.read_text(encoding="utf-8"))
    path = Path(hf_hub_download(REPO_ID, FILE, repo_type="dataset", revision=args.revision, cache_dir=str(cache)))
    table = pq.read_table(path).to_pylist()
    rows = []
    for index, row in enumerate(table):
        rows.append({"id": f"mmlu-test-{index:05d}", "subject": str(row.get("subject") or ""), "question": row["question"],
                     "choices": [str(c) for c in (row.get("choices") or [])], "answer": int(row["answer"])})
    out = args.out_dir / "mmlu-test.jsonl"
    with out.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    record = {
        "kind": "v07b-mc-fetch", "repo": REPO_ID, "repo_type": "dataset", "requested_revision": args.revision, "revision": _revision_of(card_path),
        "card": {"file": CARD, "sha256": _sha256(card_path), "license_metadata": licence["metadata"], "license_text": licence["text"]},
        "license": licence["metadata"][0] if licence["metadata"] else None, "citation": CITATION, "split": "test", "config": "all",
        "file": {"name": FILE, "sha256": _sha256(path), "bytes": path.stat().st_size, "revision": _revision_of(path), "rows": len(table)},
        "out": {"path": str(out), "sha256": _sha256(out), "rows": len(rows)},
        "use": "evaluation only (task v0.7b, sealed gate M1g); never trained on; nothing derived is committed or published",
        "created": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
    }
    (args.out_dir / "mmlu-test.fetch.json").write_text(json.dumps(record, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({k: record[k] for k in ("revision", "license", "card", "file")}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
