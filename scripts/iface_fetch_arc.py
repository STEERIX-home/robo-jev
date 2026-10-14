#!/usr/bin/env python
"""Task iface (M1) — download the ARC test splits (AI2 Reasoning Challenge, Easy and Challenge) and write their rows as JSONL.

    PY=<venv with pyarrow + huggingface_hub> artifacts/scratch/iface/cpu.sh fetch-arc scripts/iface_fetch_arc.py --out-dir artifacts/iface/m1/source

Standalone on purpose (no ``robo_jev`` import): reading parquet needs ``pyarrow``, which is not a project dependency, so this step runs in a
separate small venv (as ``scripts/convert_general_sets.py`` runs under ``uv run --with pyarrow``). Only the **test** split of each subset is
read; the measurement never trains on anything.

Source: ``allenai/ai2_arc`` on the Hugging Face Hub (the dataset card states ``license: cc-by-sa-4.0``; Clark et al. 2018, "Think you have
Solved Question Answering? Try ARC, the AI2 Reasoning Challenge"). The project's general training mix left ARC out because of the share-alike
term (artifacts/simgen/v02/general-train.manifest.json, ``excluded``) — so ARC is not in any training input, which is what M1 needs; using
it for evaluation with attribution is within CC-BY-SA-4.0, and nothing derived from it is committed or published (artifacts/ is not in git).

Writes ``<out-dir>/arc-test.jsonl`` (one row per question: ``id``, ``subset``, ``question``, ``choices`` {``text``, ``label``}, ``answerKey``)
and ``<out-dir>/arc-test.fetch.json`` (repository, resolved revision, files with sha256 and bytes, the card's licence line, row counts).
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import re
from pathlib import Path

REPO_ID = "allenai/ai2_arc"
#: The revision to read; "main" resolves to a commit whose sha is written into the fetch record (and the cache path).
DEFAULT_REVISION = "main"
FILES = {"ARC-Easy": "ARC-Easy/test-00000-of-00001.parquet", "ARC-Challenge": "ARC-Challenge/test-00000-of-00001.parquet"}
CARD = "README.md"
CITATION = "Clark et al. 2018, Think you have Solved Question Answering? Try ARC, the AI2 Reasoning Challenge (arXiv:1803.05457)"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _revision_of(path: Path) -> str | None:
    """The commit sha in a hub cache path (``…/snapshots/<sha>/…``)."""
    parts = path.parts
    return parts[parts.index("snapshots") + 1] if "snapshots" in parts else None


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
    card = card_path.read_text(encoding="utf-8")
    head = card.split("\n---", 1)[0]  # the YAML metadata block at the top of the card
    found = re.search(r"^license:[ \t]*(\S*)[ \t]*\n((?:[ \t]*-[ \t]*\S+[ \t]*\n)*)", head + "\n", flags=re.MULTILINE)
    licence = ([found.group(1)] if found and found.group(1) else []) + (re.findall(r"-[ \t]*(\S+)", found.group(2)) if found else [])
    rows, files = [], {}
    for subset, name in FILES.items():
        path = Path(hf_hub_download(REPO_ID, name, repo_type="dataset", revision=args.revision, cache_dir=str(cache)))
        files[subset] = {"file": name, "sha256": _sha256(path), "bytes": path.stat().st_size, "revision": _revision_of(path)}
        table = pq.read_table(path).to_pylist()
        for row in table:
            choices = row.get("choices") or {}
            rows.append({"id": str(row["id"]), "subset": subset, "question": row["question"],
                         "choices": {"text": list(choices.get("text") or []), "label": list(choices.get("label") or [])},
                         "answerKey": str(row.get("answerKey") or "")})
        files[subset]["rows"] = len(table)
    out = args.out_dir / "arc-test.jsonl"
    with out.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    record = {
        "kind": "iface-arc-fetch", "repo": REPO_ID, "repo_type": "dataset", "requested_revision": args.revision,
        "revision": _revision_of(card_path), "card": {"file": CARD, "sha256": _sha256(card_path), "license_lines": licence},
        "license": licence[0] if licence else None, "citation": CITATION, "split": "test", "files": files,
        "out": {"path": str(out), "sha256": _sha256(out), "rows": len(rows)},
        "use": "evaluation only (task iface, M1); never trained on; nothing derived is committed or published",
        "created": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
    }
    (args.out_dir / "arc-test.fetch.json").write_text(json.dumps(record, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({k: record[k] for k in ("revision", "license", "files")}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
