"""공개 세트 → `judgment-v0` `state_first` 레코드 변환 (analysis-nimble §3-7, Task D1 stage D3; **학습 manifest에는 넣지 않는다**).

    uv run --with pyarrow python scripts/convert_public_sets.py --out artifacts/datasets/public [--scratch artifacts/scratch/d1/public]

비로봇 4분야의 일반 판단력 sanity용 외부 세트다: BoolQ validation(3,270; 지문·질문 → boolean)과 MultiNLI dev_matched(9,815; 전제·가설 →
entailment/neutral/contradiction 3후보 choice). 레코드는 계약 검사를 지나고(`request.state`는 자유 dict, 라벨은 `single`), `split`은
원천의 dev이므로 `dev`, `origin_group`은 `public/<세트>/<행 번호>`, provenance가 원천·URL·sha256·라이선스를 든다. manifest는 단일 요청
manifest와 같은 꼴(`files: {records.jsonl: {sha256, …}}`)이라 평가 적재기가 읽을 수 있지만 `trainer_manifest: false`·`purpose: evaluation-only`가
적혀 있고 어떤 학습 설정도 이 경로를 가리키지 않는다. 원천 파일은 `--scratch`에 받고(BoolQ parquet는 pyarrow가 있어야 읽는다; MultiNLI는 zip
안의 jsonl을 `zipfile`로 읽는다), 닿지 않으면 그 세트만 건너뛰고 이유를 manifest에 적는다.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
import time
import urllib.request
import zipfile
from pathlib import Path
from typing import Any

from robo_jev.contracts import SCHEMA_SINGLE_REQUEST, validate_record

__all__ = ["PUBLIC_SETS_VERSION", "SOURCES", "boolq_record", "convert", "main", "multinli_record", "write_public_set"]

PUBLIC_SETS_VERSION = "public-sets-v0.1"
SOURCES = {
    "boolq": {
        "url": "https://huggingface.co/datasets/google/boolq/resolve/main/data/validation-00000-of-00001.parquet",
        "file": "boolq-validation.parquet",
        "license": "CC BY-SA 3.0",
        "citation": "Clark et al. 2019, BoolQ",
        "split": "validation",
    },
    "multinli": {
        "url": "https://cims.nyu.edu/~sbowman/multinli/multinli_1.0.zip",
        "file": "multinli_1.0.zip",
        "member": "multinli_1.0/multinli_1.0_dev_matched.jsonl",
        "license": "OANC / CC BY-SA 3.0 (mixed, see the source README)",
        "citation": "Williams et al. 2018, MultiNLI",
        "split": "dev_matched",
    },
}
_NLI_CRITERIA = [
    {"id": "c_entailment", "description": "the hypothesis follows from the premise (entailment)"},
    {"id": "c_neutral", "description": "the hypothesis may or may not be true given the premise (neutral)"},
    {"id": "c_contradiction", "description": "the hypothesis contradicts the premise (contradiction)"},
]
_BOOLEAN = [{"id": "true", "description": "yes"}, {"id": "false", "description": "no"}]


def boolq_record(index: int, row: dict[str, Any]) -> dict[str, Any]:
    """BoolQ 한 행 (`question`·`passage`·`answer`) → 레코드."""
    return {
        "schema_version": SCHEMA_SINGLE_REQUEST,
        "origin_group": f"public/boolq/{index:05d}",
        "split": "dev",
        "request": {
            "request_id": f"boolq-{index:05d}",
            "state": {"passage": str(row["passage"]), "question": str(row["question"])},
            "questions": [{"id": "q_boolq", "type": "boolean", "instructions": "Given the passage, is the answer to the question yes?", "criteria": list(_BOOLEAN)}],
        },
        "labels": [{"question_id": "q_boolq", "kind": "single", "answer": bool(row["answer"]), "source": "boolq/validation", "label_confidence": "high"}],
        "provenance": {"generator": PUBLIC_SETS_VERSION, "domain": "public:boolq", "public_source": "boolq", "source_split": "validation", "source_index": int(index), "language": "en"},
    }


def multinli_record(index: int, row: dict[str, Any]) -> dict[str, Any] | None:
    """MultiNLI 한 행 (`sentence1`·`sentence2`·`gold_label`) → 레코드. gold가 `-`(합의 없음)면 None."""
    gold = str(row.get("gold_label", "-"))
    if gold not in ("entailment", "neutral", "contradiction"):
        return None
    return {
        "schema_version": SCHEMA_SINGLE_REQUEST,
        "origin_group": f"public/multinli/{index:05d}",
        "split": "dev",
        "request": {
            "request_id": f"multinli-{index:05d}",
            "state": {"premise": str(row["sentence1"]), "hypothesis": str(row["sentence2"]), "genre": str(row.get("genre", ""))},
            "questions": [{"id": "q_nli", "type": "choice", "instructions": "What is the relation of the hypothesis to the premise?", "criteria": [dict(item) for item in _NLI_CRITERIA]}],
        },
        "labels": [{"question_id": "q_nli", "kind": "single", "answer": f"c_{gold}", "source": "multinli/dev_matched", "label_confidence": "high"}],
        "provenance": {"generator": PUBLIC_SETS_VERSION, "domain": "public:multinli", "public_source": "multinli", "source_split": "dev_matched", "source_index": int(index), "language": "en", "pair_id": row.get("pairID")},
    }


def _download(url: str, target: Path, log: Any = None) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_file() and target.stat().st_size > 0:
        return target
    if log is not None:
        print(f"downloading {url} → {target}", file=log, flush=True)
    with urllib.request.urlopen(url, timeout=120) as response, target.open("wb") as handle:  # noqa: S310
        while True:
            chunk = response.read(1 << 20)
            if not chunk:
                break
            handle.write(chunk)
    return target


def _boolq_rows(path: Path) -> list[dict[str, Any]]:
    import pyarrow.parquet as pq  # `uv run --with pyarrow`

    table = pq.read_table(path)
    return table.to_pylist()


def _multinli_rows(path: Path, member: str) -> list[dict[str, Any]]:
    with zipfile.ZipFile(path) as archive, archive.open(member) as handle:
        return [json.loads(line) for line in io.TextIOWrapper(handle, encoding="utf-8") if line.strip()]


def convert(name: str, rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """행 목록 → (레코드, 집계). 레코드는 계약 검사를 지난다."""
    records = []
    skipped = 0
    for index, row in enumerate(rows):
        record = boolq_record(index, row) if name == "boolq" else multinli_record(index, row)
        if record is None:
            skipped += 1
            continue
        validate_record(record)
        records.append(record)
    answers: dict[str, int] = {}
    for record in records:
        answer = str(record["labels"][0]["answer"])
        answers[answer] = answers.get(answer, 0) + 1
    return records, {"rows": len(rows), "records": len(records), "skipped_no_gold": skipped, "answers": dict(sorted(answers.items()))}


def write_public_set(out: Path, name: str, records: list[dict[str, Any]], summary: dict[str, Any], *, source: dict[str, Any], source_file: Path | None) -> dict[str, Any]:
    out = Path(out) / name
    out.mkdir(parents=True, exist_ok=True)
    path = out / "records.jsonl"
    payload = "".join(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n" for record in records).encode("utf-8")
    path.write_bytes(payload)
    manifest = {
        "version": "manifest-v0",
        "generator": PUBLIC_SETS_VERSION,
        "purpose": "evaluation-only",
        "trainer_manifest": False,
        "note": "공개 세트의 sanity 평가용 변환본 — 어떤 학습 manifest·설정도 이 경로를 가리키지 않는다 (analysis-nimble §3-7).",
        "source": {**{key: value for key, value in source.items() if key != "file"}, "downloaded_file": str(source_file) if source_file else None,
                   "sha256": hashlib.sha256(Path(source_file).read_bytes()).hexdigest() if source_file and Path(source_file).is_file() else None},
        "counts": {"states": len(records), "questions": len(records), "labels": len(records), "splits": {"dev": len(records)}, **summary},
        "files": {"records.jsonl": {"records": len(records), "bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}},
        "written_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="scripts/convert_public_sets.py", description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=Path("artifacts/datasets/public"))
    parser.add_argument("--scratch", type=Path, default=Path("artifacts/scratch/d1/public"))
    parser.add_argument("--sets", default="boolq,multinli")
    args = parser.parse_args(argv)
    status: dict[str, Any] = {}
    for name in (part.strip() for part in args.sets.split(",")):
        source = SOURCES[name]
        try:
            file = _download(source["url"], Path(args.scratch) / source["file"], log=sys.stdout)
            rows = _boolq_rows(file) if name == "boolq" else _multinli_rows(file, source["member"])
            records, summary = convert(name, rows)
            manifest = write_public_set(args.out, name, records, summary, source=source, source_file=file)
            status[name] = {"ok": True, **manifest["counts"]}
            print(f"{name}: {summary} → {Path(args.out) / name}")
        except Exception as error:  # 닿지 않거나 읽을 수 없으면 그 세트만 건너뛴다 (선택 항목)
            status[name] = {"ok": False, "error": f"{type(error).__name__}: {error}"}
            print(f"{name}: skipped — {type(error).__name__}: {error}", file=sys.stderr)
    Path(args.out).mkdir(parents=True, exist_ok=True)
    (Path(args.out) / "status.json").write_text(json.dumps({"version": PUBLIC_SETS_VERSION, "sets": status, "written_at": time.strftime("%Y-%m-%dT%H:%M:%S")}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0 if any(entry.get("ok") for entry in status.values()) else 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
