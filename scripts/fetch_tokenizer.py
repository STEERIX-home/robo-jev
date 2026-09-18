"""backbone 후보의 `tokenizer.json`을 `artifacts/tokenizers/<id>/`(git 제외)에 받는다.

후보는 :data:`robo_jev.model.tokenizer.TOKENIZER_CANDIDATES` 순서(첫 후보 Qwen3.8-27B, 다음
Qwen3.5-9B, Qwen3-8B)이고, Hub에 있는 첫 후보를 받은 뒤 **실제로 쓴 id·revision SHA·파일
해시**를 `manifest.json`에 적는다 (docs/03 §1 "revision SHA·파일 해시를 manifest에 고정").
`transformers`는 쓰지 않고 `huggingface_hub`로 파일만 받는다.

실행: `uv run python scripts/fetch_tokenizer.py [--id Qwen/Qwen3.5-9B] [--root artifacts/tokenizers]`
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import sys
from pathlib import Path

from robo_jev.model.tokenizer import (
    ARTIFACT_DIR,
    MANIFEST_NAME,
    TOKENIZER_CANDIDATES,
    TOKENIZER_FILE,
)

#: 함께 받아 두는 작은 부속 파일. 없어도 실패가 아니다.
_OPTIONAL_FILES = ("tokenizer_config.json",)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fetch(identifier: str, root: Path) -> dict:
    """후보 하나를 받는다. Hub에 없거나 받을 수 없으면 예외를 그대로 낸다."""
    from huggingface_hub import HfApi, hf_hub_download

    info = HfApi().model_info(identifier)
    target = root / identifier
    target.mkdir(parents=True, exist_ok=True)
    files: dict[str, str] = {}
    main = Path(
        hf_hub_download(identifier, TOKENIZER_FILE, revision=info.sha, local_dir=str(target))
    )
    files[TOKENIZER_FILE] = _sha256(main)
    for name in _OPTIONAL_FILES:
        try:
            extra = Path(hf_hub_download(identifier, name, revision=info.sha, local_dir=str(target)))
        except Exception:  # noqa: BLE001 — 부속 파일은 없어도 된다
            continue
        files[name] = _sha256(extra)
    return {"id": identifier, "revision": info.sha, "files": files, "path": str(main)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--id", dest="identifier", help="이 id만 받는다 (기본: 후보 순서대로 첫 성공)")
    parser.add_argument("--root", default=str(ARTIFACT_DIR), help="보관 디렉터리")
    args = parser.parse_args(argv)

    root = Path(args.root)
    candidates = [args.identifier] if args.identifier else list(TOKENIZER_CANDIDATES)
    tried: list[dict] = []
    for identifier in candidates:
        try:
            result = fetch(identifier, root)
        except Exception as exc:  # noqa: BLE001 — 다음 후보로 넘어가되 이유는 남긴다
            tried.append({"id": identifier, "error": f"{type(exc).__name__}: {exc}"[:300]})
            print(f"[fetch_tokenizer] {identifier}: 실패 — {type(exc).__name__}", file=sys.stderr)
            continue
        manifest = {
            **result,
            "fetched_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
            "candidates": list(candidates),
            "tried": tried,
        }
        root.mkdir(parents=True, exist_ok=True)
        (root / MANIFEST_NAME).write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(f"[fetch_tokenizer] 사용한 tokenizer: {identifier} @ {result['revision']}")
        print(f"[fetch_tokenizer] 파일: {result['path']} sha256={result['files'][TOKENIZER_FILE]}")
        return 0

    print(f"[fetch_tokenizer] 받을 수 있는 후보가 없다: {tried}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
