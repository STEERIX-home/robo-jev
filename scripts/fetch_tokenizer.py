"""backbone 후보의 `tokenizer.json`을 `artifacts/tokenizers/<id>/`(git 제외)에 받는다.

후보는 :data:`robo_jev.model.tokenizer.TOKENIZER_CANDIDATES` 순서(첫 후보 Qwen3.8-27B, 다음
Qwen3.5-9B, Qwen3-8B)다. 기본 호출은 **첫 후보만** 받고, 받지 못하면 실패한다 — 다른 모델의 tokenizer로
넘어가는 것은 `--allow-fallback`을 준 경우뿐이다(다른 tokenizer는 토큰 수·후보 경계·어휘가 다르므로
조용히 바뀌면 안 된다). 받은 뒤 **실제로 쓴 id·revision SHA·파일 해시**를 `manifest.json`에 적는다
(docs/03 §1 "revision SHA·파일 해시를 manifest에 고정"). `transformers`는 쓰지 않고 `huggingface_hub`로
파일만 받는다.

**같은 버전의 재현 (리뷰 11 S5).** 다음 머신은 이전 manifest의 id·revision·sha256으로 같은 파일을 받는다:

* `--from-manifest <이전 manifest.json>` — id·revision·`files["tokenizer.json"]`을 그 manifest에서 읽는다
  (`--id`·`--revision`·`--expect-sha256`을 따로 주면 그것이 우선한다).
* `--revision <commit SHA·tag·branch>` — 그 revision을 받는다(manifest에는 해석된 commit SHA를 적는다).
* `--expect-sha256 <hex>` — 받은 `tokenizer.json`의 sha256이 다르면 실패하고 받은 파일을 지운다.

고정(`--revision`·`--expect-sha256`·`--from-manifest`)이나 `--id`는 `--allow-fallback`과 함께 쓸 수 없다 —
고정은 특정 id의 것이라 다른 후보로 넘어갈 수 없다. 적재 쪽(:func:`robo_jev.model.tokenizer.load_tokenizer`)은
읽을 때마다 이 manifest의 해시와 파일을 대조한다.

실행:
  uv run python scripts/fetch_tokenizer.py                       # 첫 후보의 현재 revision
  uv run python scripts/fetch_tokenizer.py --from-manifest manifest.json   # 인계받은 manifest와 같은 파일
  uv run python scripts/fetch_tokenizer.py --id Qwen/Qwen3.5-9B --revision <sha> --expect-sha256 <hex>
  uv run python scripts/fetch_tokenizer.py --allow-fallback      # 첫 후보가 없으면 다음 후보로 (이전 기본 동작)
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path
from typing import Any

from robo_jev.model.tokenizer import (
    ARTIFACT_DIR,
    MANIFEST_NAME,
    TOKENIZER_CANDIDATES,
    TOKENIZER_FILE,
    sha256_of_file,
)

#: 함께 받아 두는 작은 부속 파일. 없어도 실패가 아니다.
_OPTIONAL_FILES = ("tokenizer_config.json",)


class HashMismatch(ValueError):
    """받은 `tokenizer.json`의 sha256이 `--expect-sha256`과 다르다."""


class HubClient:
    """`huggingface_hub`의 두 호출만 감싼다 — 검사는 같은 두 메서드를 가진 가짜를 넣는다."""

    def revision(self, identifier: str, revision: str | None) -> str:
        """`revision`(commit SHA·tag·branch; None이면 현재 main)이 가리키는 commit SHA."""
        from huggingface_hub import HfApi

        return str(HfApi().model_info(identifier, revision=revision).sha)

    def download(self, identifier: str, filename: str, revision: str, target: Path) -> Path:
        from huggingface_hub import hf_hub_download

        return Path(hf_hub_download(identifier, filename, revision=revision, local_dir=str(target)))


def fetch(
    identifier: str,
    root: Path,
    *,
    revision: str | None = None,
    expect_sha256: str | None = None,
    hub: Any | None = None,
) -> dict:
    """후보 하나를 `root/<id>/`에 받는다. Hub에 없거나 받을 수 없으면 예외를 그대로 낸다.

    `revision`이 있으면 그 revision(해석된 commit SHA를 적는다), 없으면 현재 revision. `expect_sha256`이 있으면
    받은 `tokenizer.json`의 해시가 같아야 하며, 다르면 받은 파일을 지우고 :class:`HashMismatch`다.
    """
    hub = HubClient() if hub is None else hub
    resolved = hub.revision(identifier, revision)
    target = root / identifier
    target.mkdir(parents=True, exist_ok=True)
    main = hub.download(identifier, TOKENIZER_FILE, resolved, target)
    digest = sha256_of_file(main)
    if expect_sha256 is not None and digest != expect_sha256.lower():
        main.unlink()
        raise HashMismatch(
            f"{identifier}@{resolved}: {TOKENIZER_FILE}의 sha256이 기대와 다르다 (받은 파일 {digest}, 기대 {expect_sha256.lower()}) "
            "— 받은 파일은 지웠다. revision이 다르거나 그 revision의 파일이 바뀐 것이다"
        )
    files: dict[str, str] = {TOKENIZER_FILE: digest}
    for name in _OPTIONAL_FILES:
        try:
            extra = hub.download(identifier, name, resolved, target)
        except Exception:  # noqa: BLE001 — 부속 파일은 없어도 된다
            continue
        files[name] = sha256_of_file(extra)
    return {
        "id": identifier,
        "revision": resolved,
        "files": files,
        "path": str(main),
        "pinned": {"revision": revision, "sha256": None if expect_sha256 is None else expect_sha256.lower()},
    }


def _from_manifest(path: Path, parser: argparse.ArgumentParser) -> dict[str, str]:
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        parser.error(f"--from-manifest {path}: 읽을 수 없다 ({type(exc).__name__}: {exc})")
    files = record.get("files") if isinstance(record, dict) else None
    digest = files.get(TOKENIZER_FILE) if isinstance(files, dict) else None
    missing = [name for name, value in (("id", record.get("id")), ("revision", record.get("revision")), (f"files.{TOKENIZER_FILE}", digest)) if not value]
    if missing:
        parser.error(f"--from-manifest {path}: 필요한 항목이 없다: {missing} (id·revision·files.{TOKENIZER_FILE})")
    return {"id": str(record["id"]), "revision": str(record["revision"]), "sha256": str(digest)}


def main(argv: list[str] | None = None, *, hub: Any | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--id", dest="identifier", help="이 id만 받는다 (기본: 첫 후보)")
    parser.add_argument("--root", default=str(ARTIFACT_DIR), help="보관 디렉터리")
    parser.add_argument("--revision", help="받을 revision (commit SHA·tag·branch). 기본: Hub의 현재 revision")
    parser.add_argument("--expect-sha256", dest="expect_sha256", help="받은 tokenizer.json의 sha256이 이것과 달라야 하면 실패한다")
    parser.add_argument("--from-manifest", dest="from_manifest", help="이전 manifest.json에서 id·revision·sha256을 읽는다")
    parser.add_argument("--allow-fallback", dest="allow_fallback", action="store_true", help="첫 후보를 받지 못하면 다음 후보로 넘어간다 (고정·--id와 함께 쓸 수 없다)")
    args = parser.parse_args(argv)

    if args.allow_fallback and (args.identifier or args.revision or args.expect_sha256 or args.from_manifest):
        parser.error("--allow-fallback은 고정하지 않은 기본 후보 목록에서만 쓴다 — --id·--revision·--expect-sha256·--from-manifest와 함께 쓸 수 없다")
    identifier, revision, expect = args.identifier, args.revision, args.expect_sha256
    if args.from_manifest:
        previous = _from_manifest(Path(args.from_manifest), parser)
        identifier, revision, expect = identifier or previous["id"], revision or previous["revision"], expect or previous["sha256"]

    root = Path(args.root)
    candidates = [identifier] if identifier else (list(TOKENIZER_CANDIDATES) if args.allow_fallback else [TOKENIZER_CANDIDATES[0]])
    tried: list[dict] = []
    for candidate in candidates:
        try:
            result = fetch(candidate, root, revision=revision, expect_sha256=expect, hub=hub)
        except Exception as exc:  # noqa: BLE001 — 이유를 남기고 (허용됐으면) 다음 후보로
            reason = f"{type(exc).__name__}: {exc}"
            tried.append({"id": candidate, "error": reason[:300]})
            print(f"[fetch_tokenizer] {candidate}: 실패 — {reason}", file=sys.stderr)
            continue
        manifest = {
            **result,
            "fetched_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
            "candidates": list(candidates),
            "tried": tried,
            "fallback_allowed": bool(args.allow_fallback),
        }
        root.mkdir(parents=True, exist_ok=True)
        (root / MANIFEST_NAME).write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        pinned = " (고정: " + ", ".join(f"{k}={v}" for k, v in result["pinned"].items() if v) + ")" if any(result["pinned"].values()) else ""
        print(f"[fetch_tokenizer] 사용한 tokenizer: {candidate} @ {result['revision']}{pinned}")
        print(f"[fetch_tokenizer] 파일: {result['path']} sha256={result['files'][TOKENIZER_FILE]}")
        print(f"[fetch_tokenizer] manifest: {root / MANIFEST_NAME} — 다른 머신에서 같은 파일: --from-manifest {root / MANIFEST_NAME}")
        return 0

    if not args.allow_fallback and not identifier:
        print(f"[fetch_tokenizer] 첫 후보 {candidates[0]}를 받지 못했다. 다른 후보로 넘어가려면 --allow-fallback을 준다 (tokenizer가 바뀐다)", file=sys.stderr)
    print(f"[fetch_tokenizer] 받을 수 있는 후보가 없다: {tried}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
