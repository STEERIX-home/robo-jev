"""tokenizer 적재 — backbone의 실제 `tokenizer.json`과 구조 검사용 공백 tokenizer.

모델 코드는 `transformers`를 쓰지 않는다. backbone 후보의 `tokenizer.json`만
`scripts/fetch_tokenizer.py`가 `artifacts/tokenizers/<id>/`(git 제외)에 받아 두고, 여기서는
그 파일을 :class:`tokenizers.Tokenizer`로 읽는다. 어느 id·revision을 실제로 받았고 파일 해시가
무엇인지는 보관 디렉터리의 ``manifest.json``이 말한다.

**해시 대조.** :func:`load_tokenizer` 는 읽기 전에 그 파일을 기술하는 manifest를 찾아(:func:`tokenizer_manifest_for`
— root의 manifest는 `id`로 `<root>/<id>/tokenizer.json`을 가리키고, 파일과 같은 디렉터리의 manifest는 id 없이도
그 파일을 기술한다) `files["tokenizer.json"]`의 sha256과 실제 파일을 대조하고 다르면 `ValueError`다 — 다른
revision의 파일이 같은 자리에 놓이면 토큰 수·후보 경계·어휘가 조용히 달라지기 때문이다. 기술하는 manifest가
없는 파일(검사용 tokenizer 등)은 대조 없이 읽히고 :func:`describe_tokenizer` 가 id·revision·manifest를 `None`으로
적는다 — 학습 manifest는 어느 경우든 파일의 sha256을 run의 정체로 적는다.

직렬화(:mod:`robo_jev.model.serialize`)가 tokenizer에 요구하는 것은 두 가지뿐이다.

* ``encode(text, add_special_tokens=False)`` → ``.ids``(정수 목록)와 ``.tokens``가 있는 객체
* ``encode_batch(texts, add_special_tokens=False)`` → 위 객체의 목록

:class:`WhitespaceTokenizer`는 같은 규약을 지키는 검사용 대역이다. 공백으로 나누되 줄바꿈을
토큰으로 남기므로 "후보 줄의 마지막 토큰"이라는 경계 규칙이 실제 tokenizer와 같은 모양으로
성립한다. 토큰 수는 실제 값이 아니므로 예산 측정(B2)에는 쓰지 않는다.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

__all__ = [
    "ARTIFACT_DIR",
    "FETCH_SCRIPT",
    "MANIFEST_NAME",
    "TOKENIZER_CANDIDATES",
    "TOKENIZER_FILE",
    "WhitespaceTokenizer",
    "available_tokenizer",
    "describe_tokenizer",
    "load_tokenizer",
    "resolve_tokenizer_file",
    "sha256_of_file",
    "tokenizer_manifest_for",
    "tokenizer_root",
]

#: backbone 후보의 tokenizer, 선호 순서 (docs/03 §1: 첫 후보 Qwen3.8-27B, 비교 후보 순).
TOKENIZER_CANDIDATES = ("Qwen/Qwen3.8-27B", "Qwen/Qwen3.5-9B", "Qwen/Qwen3-8B")

#: 저장소 뿌리 기준의 tokenizer 보관 위치 (git 제외).
ARTIFACT_DIR = Path("artifacts/tokenizers")
MANIFEST_NAME = "manifest.json"
TOKENIZER_FILE = "tokenizer.json"
FETCH_SCRIPT = "scripts/fetch_tokenizer.py"

_PACKAGE_ROOT = Path(__file__).resolve().parents[3]


def tokenizer_root(root: str | Path | None = None) -> Path:
    """tokenizer 보관 디렉터리. 현재 작업 디렉터리에 없으면 패키지 뿌리에서 찾는다."""
    if root is not None:
        return Path(root)
    if ARTIFACT_DIR.is_dir():
        return ARTIFACT_DIR
    return _PACKAGE_ROOT / ARTIFACT_DIR


def sha256_of_file(path: str | Path) -> str:
    """파일의 sha256 (fetch가 manifest에 적는 것과 적재의 대조가 같은 함수를 쓴다)."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_tokenizer_file(path_or_id: str | Path, root: str | Path | None = None) -> Path:
    """`path_or_id`가 가리키는 `tokenizer.json`: (1) 그 파일, (2) 그 파일이 든 디렉터리, (3) Hub id
    (`Qwen/Qwen3.8-27B` → `<root>/Qwen/Qwen3.8-27B/tokenizer.json`). 없으면 받는 방법을 적은
    `FileNotFoundError`를 낸다 — 조용히 다른 tokenizer로 바꾸지 않는다."""
    candidate = Path(path_or_id)
    if candidate.is_file():
        return candidate
    if candidate.is_dir() and (candidate / TOKENIZER_FILE).is_file():
        return candidate / TOKENIZER_FILE
    by_id = tokenizer_root(root) / str(path_or_id) / TOKENIZER_FILE
    if by_id.is_file():
        return by_id
    raise FileNotFoundError(
        f"tokenizer를 찾을 수 없다: {path_or_id!r} ({by_id}). "
        f"`uv run python {FETCH_SCRIPT}`로 받는다 (후보: {list(TOKENIZER_CANDIDATES)})"
    )


def _read_manifest(path: Path) -> dict | None:
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return record if isinstance(record, dict) else None


def tokenizer_manifest_for(file: str | Path) -> tuple[Path, dict] | None:
    """`file`(tokenizer.json)을 기술하는 manifest의 (경로, 내용). 없으면 `None`.

    같은 디렉터리의 ``manifest.json``은 id 없이도 그 파일을 기술한다(전송한 tokenizer.json + manifest.json 한 쌍).
    위쪽 디렉터리(보관 root)의 ``manifest.json``은 `id`가 `<그 디렉터리>/<id>/tokenizer.json`으로 이 파일을 가리킬
    때만(또는 `path`가 이 파일일 때만) 그 파일의 것이다 — 다른 id의 파일은 기술하지 않는다.
    """
    target = Path(file).resolve()
    for depth, base in enumerate([target.parent, *target.parents[1:6]]):
        candidate = base / MANIFEST_NAME
        if not candidate.is_file():
            continue
        record = _read_manifest(candidate)
        if record is None:
            continue
        identifier = record.get("id")
        described = (base / str(identifier) / TOKENIZER_FILE).resolve() if identifier else None
        recorded = record.get("path")
        if depth == 0 or described == target or (recorded and Path(str(recorded)).resolve() == target):
            return candidate, record
    return None


def describe_tokenizer(path_or_id: str | Path, root: str | Path | None = None) -> dict[str, Any]:
    """읽을 tokenizer 파일의 정체 — ``{kind, file, sha256, id, revision, manifest}`` — 를 manifest와 대조해 돌려준다.

    기술하는 manifest가 있고 그 `files["tokenizer.json"]`이 실제 파일의 sha256과 다르면 `ValueError`
    (어느 파일·manifest·두 해시인지, 다시 받는 방법을 적는다). manifest가 없으면 id·revision·manifest는 `None`이다.
    """
    file = resolve_tokenizer_file(path_or_id, root)
    actual = sha256_of_file(file)
    found = tokenizer_manifest_for(file)
    if found is None:
        return {"kind": "tokenizers", "file": str(file), "sha256": actual, "id": None, "revision": None, "manifest": None}
    manifest_path, record = found
    files = record.get("files")
    expected = files.get(TOKENIZER_FILE) if isinstance(files, dict) else None
    identifier = record.get("id")
    if expected and str(expected).lower() != actual:
        label = f"{identifier} " if identifier else ""
        raise ValueError(
            f"tokenizer {label}{file}: sha256이 manifest({manifest_path})와 다르다 "
            f"(파일 {actual[:12]}…, manifest {str(expected)[:12]}…) — 다른 revision의 파일이거나 손상된 파일이다. "
            f"manifest의 revision·해시로 다시 받는다: `uv run python {FETCH_SCRIPT} --from-manifest {manifest_path}`"
        )
    return {
        "kind": "tokenizers",
        "file": str(file),
        "sha256": actual,
        "id": str(identifier) if identifier else None,
        "revision": str(record["revision"]) if record.get("revision") else None,
        "manifest": str(manifest_path),
    }


def load_tokenizer(path_or_id: str | Path, root: str | Path | None = None) -> Any:
    """`tokenizer.json`을 읽는다 (:func:`resolve_tokenizer_file` 의 세 꼴).

    읽기 전에 :func:`describe_tokenizer` 로 그 파일을 기술하는 manifest의 sha256과 대조한다 — 다르면
    `ValueError`, 없으면 `FileNotFoundError`. 조용히 다른 tokenizer로 바꾸지 않는다.
    """
    from tokenizers import Tokenizer

    described = describe_tokenizer(path_or_id, root)
    return Tokenizer.from_file(described["file"])


def available_tokenizer(root: str | Path | None = None) -> tuple[str, Path] | None:
    """받아 둔 실제 tokenizer의 (id, 파일 경로). 없으면 `None`.

    `manifest.json`이 있으면 그것이 정본이고, 없으면 후보 순서대로 디렉터리를 찾는다.
    """
    base = tokenizer_root(root)
    manifest = base / MANIFEST_NAME
    if manifest.is_file():
        record = json.loads(manifest.read_text(encoding="utf-8"))
        identifier = str(record.get("id", ""))
        path = base / identifier / TOKENIZER_FILE
        if identifier and path.is_file():
            return identifier, path
    for identifier in TOKENIZER_CANDIDATES:
        path = base / identifier / TOKENIZER_FILE
        if path.is_file():
            return identifier, path
    return None


# --------------------------------------------------------------------------
# 검사용 공백 tokenizer
# --------------------------------------------------------------------------

_PIECES = re.compile(r"\n|[^\s]+")


@dataclass
class _Encoding:
    ids: list[int]
    tokens: list[str]


@dataclass
class WhitespaceTokenizer:
    """공백으로 나누고 줄바꿈은 토큰으로 남기는 대역. id는 처음 본 순서로 준다."""

    vocab: dict[str, int] = field(default_factory=dict)

    def encode(self, text: str, add_special_tokens: bool = False) -> _Encoding:
        del add_special_tokens  # 특수 토큰이 없다
        tokens = _PIECES.findall(text)
        ids = [self.vocab.setdefault(token, len(self.vocab)) for token in tokens]
        return _Encoding(ids=ids, tokens=tokens)

    def encode_batch(self, texts: list[str], add_special_tokens: bool = False) -> list[_Encoding]:
        return [self.encode(text, add_special_tokens=add_special_tokens) for text in texts]

    def decode(self, ids: list[int]) -> str:
        """조각을 공백으로 잇되 줄바꿈 앞뒤에는 공백을 넣지 않는다."""
        inverse = {index: token for token, index in self.vocab.items()}
        out: list[str] = []
        for index in ids:
            token = inverse[index]
            if out and token != "\n" and out[-1] != "\n":
                out.append(" ")
            out.append(token)
        return "".join(out)
