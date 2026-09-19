"""`scripts/fetch_tokenizer.py` 검사 — 가짜 Hub로 고정 revision·해시 대조·fallback 규칙·manifest 형식을 본다 (리뷰 11 S5).

다음 머신이 같은 tokenizer를 같은 버전으로 다시 받을 수 있어야 한다: 이전 manifest의 id·revision·sha256을
`--from-manifest`(또는 `--revision`·`--expect-sha256`)로 넘기면 그 revision을 받고 파일 해시를 대조한다. 첫 후보를
받지 못했을 때 다른 모델의 tokenizer로 넘어가는 것은 `--allow-fallback`을 준 경우뿐이다. 네트워크는 쓰지 않는다.
"""

import functools
import importlib.util
import json
import sys
from pathlib import Path

import pytest
from helpers import REPO
from test_tokenizer import sha256_of, tiny_tokenizer_json

from robo_jev.model.tokenizer import MANIFEST_NAME, TOKENIZER_CANDIDATES, describe_tokenizer, load_tokenizer

SCRIPT = REPO / "scripts" / "fetch_tokenizer.py"
FIRST, SECOND, THIRD = TOKENIZER_CANDIDATES


@functools.lru_cache(maxsize=1)
def script():
    spec = importlib.util.spec_from_file_location("fetch_tokenizer", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class FakeHub:
    """id → {revision SHA → {파일명: bytes}}. `heads`는 revision을 주지 않았을 때의 현재 revision(Hub의 `main`)."""

    def __init__(self, repos: dict[str, dict[str, dict[str, bytes]]], heads: dict[str, str]) -> None:
        self.repos, self.heads = repos, heads
        self.calls: list[tuple] = []

    def revision(self, identifier: str, revision: str | None) -> str:
        self.calls.append(("revision", identifier, revision))
        if identifier not in self.repos:
            raise LookupError(f"{identifier}: Hub에 없다")
        sha = self.heads[identifier] if revision is None else revision
        if sha not in self.repos[identifier]:
            raise LookupError(f"{identifier}@{revision}: 없는 revision")
        return sha

    def download(self, identifier: str, filename: str, revision: str, target: Path) -> Path:
        self.calls.append(("download", identifier, filename, revision))
        files = self.repos[identifier][revision]
        if filename not in files:
            raise FileNotFoundError(f"{identifier}@{revision}: {filename} 없음")
        target.mkdir(parents=True, exist_ok=True)
        path = target / filename
        path.write_bytes(files[filename])
        return path


@pytest.fixture(scope="module")
def tokenizer_bytes(tmp_path_factory) -> dict[str, bytes]:
    """두 revision의 서로 다른 `tokenizer.json` (어휘가 다르다)."""
    base = tmp_path_factory.mktemp("tokenizer-bytes")
    tiny_tokenizer_json(base / "old.json")
    tiny_tokenizer_json(base / "new.json", vocab={"[UNK]": 0, "world": 1, "hello": 2, "\n": 3, "extra": 4})
    return {"old": (base / "old.json").read_bytes(), "new": (base / "new.json").read_bytes()}


def hub_with_every_candidate(tokenizer_bytes) -> FakeHub:
    repos = {
        FIRST: {
            "1111111111111111111111111111111111111111": {"tokenizer.json": tokenizer_bytes["old"], "tokenizer_config.json": b"{}"},
            "2222222222222222222222222222222222222222": {"tokenizer.json": tokenizer_bytes["new"], "tokenizer_config.json": b"{}"},
        },
        SECOND: {"3333333333333333333333333333333333333333": {"tokenizer.json": tokenizer_bytes["new"]}},
        THIRD: {"4444444444444444444444444444444444444444": {"tokenizer.json": tokenizer_bytes["old"]}},
    }
    heads = {FIRST: "2222222222222222222222222222222222222222", SECOND: "3333333333333333333333333333333333333333", THIRD: "4444444444444444444444444444444444444444"}
    return FakeHub(repos, heads)


def read_manifest(root: Path) -> dict:
    return json.loads((root / MANIFEST_NAME).read_text(encoding="utf-8"))


def test_default_fetch_takes_the_first_candidate_at_its_current_revision_and_writes_the_manifest(tmp_path, tokenizer_bytes, capsys):
    hub = hub_with_every_candidate(tokenizer_bytes)
    root = tmp_path / "tokenizers"
    assert script().main(["--root", str(root)], hub=hub) == 0
    manifest = read_manifest(root)
    new_sha = sha256_of(root / FIRST / "tokenizer.json")
    assert (root / FIRST / "tokenizer.json").read_bytes() == tokenizer_bytes["new"]
    assert manifest["id"] == FIRST and manifest["revision"] == "2222222222222222222222222222222222222222"
    assert manifest["files"] == {"tokenizer.json": new_sha, "tokenizer_config.json": sha256_of(root / FIRST / "tokenizer_config.json")}
    assert manifest["path"] == str(root / FIRST / "tokenizer.json") and manifest["fetched_at"]
    assert manifest["candidates"] == [FIRST] and manifest["tried"] == []
    assert manifest["pinned"] == {"revision": None, "sha256": None} and manifest["fallback_allowed"] is False
    assert hub.calls[0] == ("revision", FIRST, None) and ("download", FIRST, "tokenizer.json", manifest["revision"]) in hub.calls
    # 받은 파일은 곧바로 manifest와 대조되어 읽힌다
    assert load_tokenizer(FIRST, root=root).encode("extra", add_special_tokens=False).ids == [4]
    assert describe_tokenizer(FIRST, root=root)["revision"] == manifest["revision"]
    out = capsys.readouterr().out
    assert FIRST in out and manifest["revision"] in out and new_sha in out


def test_a_failed_first_candidate_is_an_error_unless_fallback_is_allowed_explicitly(tmp_path, tokenizer_bytes, capsys):
    hub = hub_with_every_candidate(tokenizer_bytes)
    del hub.repos[FIRST]
    root = tmp_path / "tokenizers"
    assert script().main(["--root", str(root)], hub=hub) == 1
    assert not (root / MANIFEST_NAME).exists() and not (root / SECOND).exists()
    err = capsys.readouterr().err
    assert FIRST in err and "--allow-fallback" in err and "LookupError" in err
    assert [call for call in hub.calls if call[0] == "download"] == []

    assert script().main(["--root", str(root), "--allow-fallback"], hub=hub) == 0
    manifest = read_manifest(root)
    assert manifest["id"] == SECOND and manifest["revision"] == "3333333333333333333333333333333333333333"
    assert manifest["candidates"] == list(TOKENIZER_CANDIDATES) and manifest["fallback_allowed"] is True
    assert [entry["id"] for entry in manifest["tried"]] == [FIRST] and "LookupError" in manifest["tried"][0]["error"]
    assert load_tokenizer(SECOND, root=root).encode("extra", add_special_tokens=False).ids == [4]


def test_a_pinned_revision_and_expected_hash_reproduce_the_same_file_and_a_mismatch_leaves_nothing(tmp_path, tokenizer_bytes, capsys):
    hub = hub_with_every_candidate(tokenizer_bytes)
    root = tmp_path / "tokenizers"
    (tmp_path / "old.json").write_bytes(tokenizer_bytes["old"])
    (tmp_path / "new.json").write_bytes(tokenizer_bytes["new"])
    old_sha, new_sha = sha256_of(tmp_path / "old.json"), sha256_of(tmp_path / "new.json")
    old = "1111111111111111111111111111111111111111"

    assert script().main(["--root", str(root), "--revision", old, "--expect-sha256", old_sha], hub=hub) == 0
    manifest = read_manifest(root)
    assert manifest["id"] == FIRST and manifest["revision"] == old and manifest["files"]["tokenizer.json"] == old_sha
    assert manifest["pinned"] == {"revision": old, "sha256": old_sha}
    assert (root / FIRST / "tokenizer.json").read_bytes() == tokenizer_bytes["old"]
    assert ("download", FIRST, "tokenizer.json", old) in hub.calls and ("revision", FIRST, old) in hub.calls
    assert load_tokenizer(FIRST, root=root).encode("hello", add_special_tokens=False).ids == [1]

    # 기대 해시와 다른 파일: 실패하고, 받은 파일을 남기지 않으며, 이전 manifest는 그대로다
    assert script().main(["--root", str(root), "--revision", "2222222222222222222222222222222222222222", "--expect-sha256", old_sha], hub=hub) == 1
    err = capsys.readouterr().err
    assert "sha256" in err and old_sha in err and new_sha in err
    assert read_manifest(root) == manifest
    assert not (root / FIRST / "tokenizer.json").exists()
    # 대소문자는 가리지 않는다; 해시만 고정하고 revision은 현재 것을 받을 수도 있다 (해시가 맞을 때만 성공)
    assert script().main(["--root", str(root), "--expect-sha256", new_sha.upper()], hub=hub) == 0
    assert read_manifest(root)["revision"] == "2222222222222222222222222222222222222222"
    assert read_manifest(root)["pinned"] == {"revision": None, "sha256": new_sha}


def test_from_manifest_reads_id_revision_and_hash_from_a_previous_machines_manifest(tmp_path, tokenizer_bytes):
    """인계받은 manifest(지금까지의 형식 그대로)만으로 같은 id·revision을 받고 해시를 대조한다."""
    hub = hub_with_every_candidate(tokenizer_bytes)
    (tmp_path / "old.json").write_bytes(tokenizer_bytes["old"])
    old_sha = sha256_of(tmp_path / "old.json")
    previous = {
        "id": FIRST,
        "revision": "1111111111111111111111111111111111111111",
        "files": {"tokenizer.json": old_sha, "tokenizer_config.json": "0" * 64},
        "path": "/on/the/previous/machine/tokenizer.json",
        "fetched_at": "2026-09-18T16:50:26+00:00",
        "candidates": list(TOKENIZER_CANDIDATES),
        "tried": [],
    }
    handed_over = tmp_path / "handoff-manifest.json"
    handed_over.write_text(json.dumps(previous, indent=2), encoding="utf-8")
    root = tmp_path / "tokenizers"
    assert script().main(["--root", str(root), "--from-manifest", str(handed_over)], hub=hub) == 0
    manifest = read_manifest(root)
    assert manifest["id"] == FIRST and manifest["revision"] == previous["revision"]
    assert manifest["files"]["tokenizer.json"] == old_sha and manifest["pinned"] == {"revision": previous["revision"], "sha256": old_sha}
    assert manifest["candidates"] == [FIRST] and manifest["fallback_allowed"] is False
    assert ("download", FIRST, "tokenizer.json", previous["revision"]) in hub.calls
    assert describe_tokenizer(FIRST, root=root)["sha256"] == old_sha

    # 인계 manifest의 파일이 Hub의 그 revision과 다르면 실패한다 (해시 불일치)
    hub.repos[FIRST][previous["revision"]]["tokenizer.json"] = tokenizer_bytes["new"]
    assert script().main(["--root", str(root), "--from-manifest", str(handed_over)], hub=hub) == 1
    assert read_manifest(root) == manifest  # 실패한 시도는 manifest를 바꾸지 않는다

    incomplete = tmp_path / "incomplete.json"
    incomplete.write_text(json.dumps({"id": FIRST}), encoding="utf-8")
    with pytest.raises(SystemExit):
        script().main(["--root", str(root), "--from-manifest", str(incomplete)], hub=hub)


def test_fallback_cannot_be_combined_with_a_pin_or_a_single_id(tmp_path, tokenizer_bytes):
    hub = hub_with_every_candidate(tokenizer_bytes)
    root = str(tmp_path / "tokenizers")
    for extra in (["--revision", "1" * 40], ["--expect-sha256", "f" * 64], ["--id", FIRST], ["--from-manifest", "x.json"]):
        with pytest.raises(SystemExit) as excinfo:
            script().main(["--root", root, "--allow-fallback", *extra], hub=hub)
        assert excinfo.value.code == 2
    assert hub.calls == []
