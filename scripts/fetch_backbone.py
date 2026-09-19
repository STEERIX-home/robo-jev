"""backbone 후보의 가중치를 `artifacts/models/<id>/`(git 제외)에 받고 파일 해시를 manifest에 적는다 (Task 2b G0a).

`scripts/fetch_tokenizer.py`와 같은 꼴이다: `huggingface_hub`로 파일만 받고(`transformers`는 쓰지 않는다), 받은 뒤
**실제로 쓴 id·revision SHA·파일별 sha256·크기**를 `artifacts/models/manifest.json`의 ``models[id]``에 적는다. 받는
파일은 `config.json`, `*.safetensors`, `model.safetensors.index.json`, `generation_config.json`(없어도 된다)이다 —
tokenizer는 `artifacts/tokenizers`의 것을 쓰므로 받지 않는다.

후보는 `configs/model/candidates.yaml`에 고정되어 있다(id·revision·라이선스·층 구성). 기본 호출은 그 yaml의
`revision`(commit SHA)을 받고, 받은 뒤 해석된 commit SHA·모델 카드의 라이선스·safetensors header에서 **센** 파라미터
수(`params_total` = 언어 모델 + lm_head, `params_checkpoint` = vision·MTP까지)를 yaml에 써 넣는다(docs/06 Task 2b
"revision·라이선스·층 구성을 고정"). 받은 config.json의 구조 값이 yaml과 다르면 실패한다 — 조용히 바뀐 후보를
재지 않는다. yaml에 없는 id는 받지 않는다(후보 목록이 고정의 단일 출처다).

**같은 버전의 재현.** 다음 머신은 이전 manifest의 id·revision·해시로 같은 파일을 받는다:

* `--from-manifest <이전 manifest.json>` — 그 manifest의 후보 전부(`--id`를 주면 그것만)를 같은 revision으로 받고
  파일별 sha256을 대조한다.
* `--revision <commit SHA·tag·branch>` — yaml 대신 그 revision을 받는다(manifest·yaml에는 해석된 commit SHA를 적는다).
* `--expect-sha256 <hex>` — 받은 **파일 묶음의 지문**(:func:`robo_jev.model.backbone.fileset_digest`, manifest의
  `digest`)이 다르면 실패하고 받은 파일을 지운다. 파일이 여럿이라 tokenizer의 파일 하나 해시 대신 묶음 지문이다.

적재 쪽(:func:`robo_jev.model.backbone.describe_backbone`, 측정 스크립트가 부른다)은 읽기 전에 manifest의 크기와
작은 파일의 sha256을 대조한다(`--verify-full`이면 safetensors 전부).

실행:
  uv run python scripts/fetch_backbone.py --id Qwen/Qwen3.5-2B                          # yaml의 revision
  uv run python scripts/fetch_backbone.py --id Qwen/Qwen3.5-9B --revision <sha> --expect-sha256 <hex>
  uv run python scripts/fetch_backbone.py --from-manifest artifacts/models/manifest.json  # 인계받은 manifest 전부
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import sys
from pathlib import Path
from typing import Any

import yaml

from robo_jev.model.backbone import (
    ARTIFACT_DIR,
    CANDIDATES_CONFIG,
    FILE_PATTERNS,
    MANIFEST_NAME,
    STRUCTURE_KEYS,
    config_structure,
    count_params,
    fileset_digest,
    hash_files,
    matches_patterns,
    matching_files,
    read_manifest,
)

_HEX40 = re.compile(r"^[0-9a-f]{40}$")


class HashMismatch(ValueError):
    """받은 파일(묶음)의 sha256이 기대와 다르다."""


class StructureMismatch(ValueError):
    """받은 config.json의 구조 값이 candidates.yaml의 항목과 다르다."""


class HubClient:
    """`huggingface_hub`의 네 호출만 감싼다 — 검사는 같은 네 메서드를 가진 가짜를 넣는다."""

    def files(self, identifier: str, revision: str) -> list[str]:
        """그 revision의 저장소 파일 목록 — 받을 묶음(패턴에 맞는 것)을 미리 안다."""
        from huggingface_hub import HfApi

        return [str(name) for name in HfApi().list_repo_files(identifier, revision=revision)]

    def revision(self, identifier: str, revision: str | None) -> str:
        """`revision`(commit SHA·tag·branch; None이면 현재 main)이 가리키는 commit SHA."""
        from huggingface_hub import HfApi

        return str(HfApi().model_info(identifier, revision=revision).sha)

    def license(self, identifier: str, revision: str) -> str | None:
        """모델 카드의 `license` (없으면 None)."""
        from huggingface_hub import HfApi

        card = HfApi().model_info(identifier, revision=revision).card_data
        value = getattr(card, "license", None) if card is not None else None
        return str(value) if value else None

    def download(self, identifier: str, patterns: list[str], revision: str, target: Path) -> Path:
        from huggingface_hub import snapshot_download

        return Path(snapshot_download(identifier, revision=revision, local_dir=str(target), allow_patterns=patterns))


def fetch(
    identifier: str,
    root: Path,
    *,
    revision: str | None = None,
    expect_sha256: str | None = None,
    expect_files: dict[str, dict[str, Any]] | None = None,
    hub: Any | None = None,
) -> dict:
    """후보 하나를 `root/<id>/`에 받는다. Hub에 없거나 받을 수 없으면 예외를 그대로 낸다.

    `revision`이 있으면 그 revision(해석된 commit SHA를 적는다), 없으면 현재 revision. `expect_sha256`(묶음 지문)이나
    `expect_files`(이전 manifest의 파일별 sha256)가 있으면 받은 파일이 같아야 하며, 다르면 받은 파일을 지우고
    :class:`HashMismatch`다.
    """
    hub = HubClient() if hub is None else hub
    resolved = hub.revision(identifier, revision)
    expected = sorted(name for name in hub.files(identifier, resolved) if matches_patterns(name))
    if "config.json" not in expected or not any(name.endswith(".safetensors") for name in expected):
        raise FileNotFoundError(f"{identifier}@{resolved}: Hub의 그 revision에 config.json과 safetensors가 있어야 한다 (있는 것: {expected})")
    target = root / identifier
    target.mkdir(parents=True, exist_ok=True)
    hub.download(identifier, list(FILE_PATTERNS), resolved, target)
    # 같은 디렉터리에 남아 있던 다른 revision의 조각(패턴에는 맞지만 이 묶음에 없는 파일)은 지문에 들어가면 안 된다.
    for name in matching_files(target):
        if name not in expected:
            (target / name).unlink()
            print(f"[fetch_backbone] {identifier}: 이 revision의 묶음에 없는 파일을 지웠다: {name}", file=sys.stderr)
    files = hash_files(target)
    missing = [name for name in expected if name not in files]
    if missing:
        raise FileNotFoundError(f"{identifier}@{resolved}: 받지 못한 파일이 있다: {missing}")
    digest = fileset_digest(files)

    problems: list[str] = []
    for name, expected in (expect_files or {}).items():
        actual = files.get(name)
        if actual is None:
            problems.append(f"{name}: 받은 파일에 없다")
        elif actual["sha256"] != str(expected["sha256"]).lower():
            problems.append(f"{name}: sha256 {actual['sha256'][:12]}… ≠ 기대 {str(expected['sha256'])[:12]}…")
    if expect_sha256 is not None and digest != expect_sha256.lower():
        problems.append(f"묶음 지문 {digest[:12]}… ≠ 기대 {expect_sha256.lower()[:12]}…")
    if problems:
        for name in files:
            (target / name).unlink()
        raise HashMismatch(
            f"{identifier}@{resolved}: 받은 파일이 기대와 다르다 — " + "; ".join(problems)
            + " — 받은 파일은 지웠다. revision이 다르거나 그 revision의 파일이 바뀐 것이다"
        )

    params = count_params([target / name for name in files if name.endswith(".safetensors")])
    return {
        "id": identifier,
        "revision": resolved,
        "license": hub.license(identifier, resolved),
        "files": files,
        "digest": digest,
        "bytes_total": sum(int(spec["bytes"]) for spec in files.values()),
        "params": params,
        "path": str(target),
        "pinned": {"revision": revision, "sha256": None if expect_sha256 is None else expect_sha256.lower()},
    }


# --------------------------------------------------------------------------
# candidates.yaml
# --------------------------------------------------------------------------


def read_candidates(path: Path) -> tuple[str, dict[str, Any]]:
    """(머리말 주석, 내용). 머리말은 첫 항목 줄 앞의 `#`·빈 줄이다 — 다시 쓸 때 그대로 남긴다."""
    text = path.read_text(encoding="utf-8")
    header_lines: list[str] = []
    for line in text.splitlines(keepends=True):
        if line.startswith("#") or not line.strip():
            header_lines.append(line)
        else:
            break
    data = yaml.safe_load(text)
    if not isinstance(data, dict) or not isinstance(data.get("candidates"), list):
        raise ValueError(f"{path}: `candidates` 목록이 있어야 한다")
    return "".join(header_lines), data


def candidate_entry(data: dict[str, Any], identifier: str) -> dict[str, Any] | None:
    return next((entry for entry in data["candidates"] if entry.get("id") == identifier), None)


def check_structure(entry: dict[str, Any], config: dict[str, Any]) -> None:
    """yaml 항목의 구조 값이 받은 config.json과 같아야 한다 — 다르면 :class:`StructureMismatch`."""
    actual = config_structure(config)
    different = {key: (entry.get(key), actual[key]) for key in STRUCTURE_KEYS if entry.get(key) != actual[key]}
    if different:
        raise StructureMismatch(
            f"{entry['id']}: candidates.yaml과 받은 config.json의 구조가 다르다 (yaml, config): {different}"
        )


def write_candidates(path: Path, header: str, data: dict[str, Any]) -> None:
    body = yaml.safe_dump(data, sort_keys=False, allow_unicode=True, width=120)
    path.write_text(header + body, encoding="utf-8")


# --------------------------------------------------------------------------


def _pinned_revision(entry: dict[str, Any] | None) -> str | None:
    value = entry.get("revision") if entry else None
    return str(value) if isinstance(value, str) and _HEX40.match(value) else None


def main(argv: list[str] | None = None, *, hub: Any | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--id", dest="identifier", help="받을 후보 id (candidates.yaml에 있어야 한다)")
    parser.add_argument("--root", default=str(ARTIFACT_DIR), help="보관 디렉터리")
    parser.add_argument("--config", default=str(CANDIDATES_CONFIG), help="후보 목록 yaml — revision을 읽고 해석된 값·라이선스·파라미터 수를 써 넣는다")
    parser.add_argument("--revision", help="받을 revision (commit SHA·tag·branch). 기본: yaml의 revision, 없으면 Hub의 현재 revision")
    parser.add_argument("--expect-sha256", dest="expect_sha256", help="받은 파일 묶음의 지문(manifest의 digest)이 이것과 달라야 하면 실패한다")
    parser.add_argument("--from-manifest", dest="from_manifest", help="이전 manifest.json의 revision·파일 해시로 같은 파일을 받는다 (--id가 없으면 그 manifest의 후보 전부)")
    args = parser.parse_args(argv)

    if not args.identifier and not args.from_manifest:
        parser.error("--id 또는 --from-manifest가 필요하다")
    if args.from_manifest and len([x for x in (args.revision, args.expect_sha256) if x]) and not args.identifier:
        parser.error("--from-manifest로 여러 후보를 받을 때는 --revision·--expect-sha256을 줄 수 없다 (--id로 하나를 고른다)")

    config_path = Path(args.config)
    header, data = read_candidates(config_path)
    root = Path(args.root)

    previous: dict[str, Any] = {}
    if args.from_manifest:
        try:
            previous = json.loads(Path(args.from_manifest).read_text(encoding="utf-8")).get("models") or {}
        except (OSError, ValueError) as exc:
            parser.error(f"--from-manifest {args.from_manifest}: 읽을 수 없다 ({type(exc).__name__}: {exc})")
        if not previous:
            parser.error(f"--from-manifest {args.from_manifest}: `models` 항목이 없다")
    identifiers = [args.identifier] if args.identifier else list(previous)

    for identifier in identifiers:
        entry = candidate_entry(data, identifier)
        if entry is None:
            print(f"[fetch_backbone] {identifier}: {config_path}에 없는 후보다 — 후보 목록에 먼저 넣는다", file=sys.stderr)
            return 1
        earlier = previous.get(identifier) or {}
        revision = args.revision or earlier.get("revision") or _pinned_revision(entry)
        expect = args.expect_sha256 or earlier.get("digest")
        expect_files = earlier.get("files") if earlier else None
        try:
            result = fetch(identifier, root, revision=revision, expect_sha256=expect, expect_files=expect_files, hub=hub)
            config = json.loads((root / identifier / "config.json").read_text(encoding="utf-8"))
            check_structure(entry, config)
        except Exception as exc:  # noqa: BLE001 — 이유를 남기고 실패한다 (다른 후보로 넘어가지 않는다)
            print(f"[fetch_backbone] {identifier}: 실패 — {type(exc).__name__}: {exc}", file=sys.stderr)
            return 1

        manifest = read_manifest(root) if (root / MANIFEST_NAME).is_file() else {"models": {}}
        manifest["models"][identifier] = {
            **result,
            "fetched_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        }
        manifest["root"] = str(root.resolve())
        manifest["updated_at"] = manifest["models"][identifier]["fetched_at"]
        root.mkdir(parents=True, exist_ok=True)
        (root / MANIFEST_NAME).write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

        # 받는 데 수십 분이 걸리므로 yaml은 쓰기 직전에 다시 읽는다 — 그 사이의 편집(머리말 등)을 덮어쓰지 않는다.
        header, data = read_candidates(config_path)
        entry = candidate_entry(data, identifier)
        if entry is None:
            print(f"[fetch_backbone] {identifier}: 받는 동안 {config_path}에서 항목이 사라졌다 — yaml을 갱신하지 않는다", file=sys.stderr)
            return 1
        entry["revision"] = result["revision"]
        entry["license"] = result["license"]
        entry["params_total"] = result["params"]["text"]
        entry["params_checkpoint"] = result["params"]["total"]
        write_candidates(config_path, header, data)

        pinned = " (고정: " + ", ".join(f"{k}={v}" for k, v in result["pinned"].items() if v) + ")" if any(result["pinned"].values()) else ""
        print(f"[fetch_backbone] 받음: {identifier} @ {result['revision']}{pinned} license={result['license']}")
        print(
            f"[fetch_backbone] 파일 {len(result['files'])}개 {result['bytes_total'] / 1e9:.2f} GB, 지문 {result['digest']}, "
            f"파라미터 text={result['params']['text']:,} 전체={result['params']['total']:,} (vision={result['params']['vision']:,}, mtp={result['params']['mtp']:,})"
        )
        print(f"[fetch_backbone] manifest: {root / MANIFEST_NAME}; yaml 갱신: {config_path} — 다른 머신에서 같은 파일: --from-manifest {root / MANIFEST_NAME} --id {identifier}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
