"""tokenizer 적재 검사 — 실제 `tokenizer.json` 적재와 구조 검사용 공백 tokenizer.

실제 backbone tokenizer는 `scripts/fetch_tokenizer.py`가 받아 둔 것만 쓴다. 없으면 그 검사만
이유를 적고 건너뛰고, 공백 tokenizer가 구조 검사를 언제나 돌게 한다.
"""

import hashlib
import json

import pytest
from tokenizers import Tokenizer, models, pre_tokenizers

from robo_jev.model.tokenizer import (
    FETCH_SCRIPT,
    MANIFEST_NAME,
    TOKENIZER_CANDIDATES,
    WhitespaceTokenizer,
    available_tokenizer,
    describe_tokenizer,
    load_tokenizer,
)


def tiny_tokenizer_json(path, vocab: dict[str, int] | None = None) -> None:
    """검사용 진짜 `tokenizer.json` — 단어 단위 어휘 하나."""
    vocab = {"[UNK]": 0, "hello": 1, "world": 2, "\n": 3} if vocab is None else vocab
    tokenizer = Tokenizer(models.WordLevel(vocab=vocab, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tokenizer.save(str(path))


def sha256_of(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# --------------------------------------------------------------------------
# 공백 tokenizer (구조 검사용)
# --------------------------------------------------------------------------


def test_whitespace_tokenizer_keeps_newlines_as_tokens():
    tokenizer = WhitespaceTokenizer()
    encoded = tokenizer.encode("a b\nc\n", add_special_tokens=False)
    assert encoded.tokens == ["a", "b", "\n", "c", "\n"]
    assert len(encoded.ids) == 5
    assert encoded.ids[2] == encoded.ids[4]  # 같은 조각은 같은 id


def test_whitespace_tokenizer_ids_are_stable_within_an_instance():
    tokenizer = WhitespaceTokenizer()
    first = tokenizer.encode("x y", add_special_tokens=False).ids
    second = tokenizer.encode("y x", add_special_tokens=False).ids
    assert first == list(reversed(second))


def test_whitespace_tokenizer_batch_matches_single():
    tokenizer = WhitespaceTokenizer()
    texts = ["one two\n", "three", ""]
    batch = tokenizer.encode_batch(texts, add_special_tokens=False)
    assert [item.ids for item in batch] == [
        tokenizer.encode(text, add_special_tokens=False).ids for text in texts
    ]
    assert batch[2].ids == []


def test_whitespace_tokenizer_decodes_back_to_text():
    tokenizer = WhitespaceTokenizer()
    text = "goal text=a b\nobject id=o1\n"
    assert tokenizer.decode(tokenizer.encode(text, add_special_tokens=False).ids) == text


# --------------------------------------------------------------------------
# 실제 tokenizer.json 적재
# --------------------------------------------------------------------------


def test_load_tokenizer_from_a_file_or_its_directory(tmp_path):
    tiny_tokenizer_json(tmp_path / "tokenizer.json")
    by_file = load_tokenizer(tmp_path / "tokenizer.json")
    by_dir = load_tokenizer(tmp_path)
    assert by_file.encode("hello world", add_special_tokens=False).ids == [1, 2]
    assert by_dir.encode("hello world", add_special_tokens=False).ids == [1, 2]


def test_load_tokenizer_by_id_reads_the_artifact_root(tmp_path):
    target = tmp_path / "Some" / "Model"
    target.mkdir(parents=True)
    tiny_tokenizer_json(target / "tokenizer.json")
    tokenizer = load_tokenizer("Some/Model", root=tmp_path)
    assert tokenizer.encode("world", add_special_tokens=False).ids == [2]


def test_missing_tokenizer_names_the_fetch_script(tmp_path):
    with pytest.raises(FileNotFoundError, match=FETCH_SCRIPT):
        load_tokenizer("Absent/Model", root=tmp_path)


def test_available_tokenizer_follows_the_manifest(tmp_path):
    assert available_tokenizer(root=tmp_path) is None

    target = tmp_path / "Qwen" / "Qwen3-8B"
    target.mkdir(parents=True)
    tiny_tokenizer_json(target / "tokenizer.json")
    # manifest가 없으면 후보 순서대로 디렉터리를 찾는다.
    found = available_tokenizer(root=tmp_path)
    assert found is not None and found[0] == "Qwen/Qwen3-8B"
    assert found[1] == target / "tokenizer.json"

    # manifest가 있으면 그것이 정본이다.
    (tmp_path / MANIFEST_NAME).write_text(
        json.dumps({"id": "Qwen/Qwen3-8B", "revision": "abc"}), encoding="utf-8"
    )
    found = available_tokenizer(root=tmp_path)
    assert found == ("Qwen/Qwen3-8B", target / "tokenizer.json")


def test_candidate_order_is_the_documented_one():
    assert TOKENIZER_CANDIDATES == ("Qwen/Qwen3.8-27B", "Qwen/Qwen3.5-9B", "Qwen/Qwen3-8B")


def test_real_tokenizer_loads_when_fetched():
    found = available_tokenizer()
    if found is None:
        pytest.skip(f"실제 tokenizer가 없다 — `uv run python {FETCH_SCRIPT}`로 받는다")
    tokenizer = load_tokenizer(found[0])
    ids = tokenizer.encode("object id=o1 pose_mm=310,-40,742\n", add_special_tokens=False).ids
    assert ids and all(isinstance(value, int) for value in ids)


# --------------------------------------------------------------------------
# manifest와의 해시 대조 (리뷰 11 S5)
# --------------------------------------------------------------------------


def existing_manifest(identifier: str, tokenizer_json, revision: str = "abc123") -> dict:
    """`scripts/fetch_tokenizer.py`가 지금까지 써 온 manifest 형식 그대로."""
    return {
        "id": identifier,
        "revision": revision,
        "files": {"tokenizer.json": sha256_of(tokenizer_json), "tokenizer_config.json": "0" * 64},
        "path": "/somewhere/else/on/the/fetching/machine/tokenizer.json",
        "fetched_at": "2026-09-18T16:50:26+00:00",
        "candidates": list(TOKENIZER_CANDIDATES),
        "tried": [],
    }


def test_load_tokenizer_verifies_the_file_hash_against_the_manifest_that_describes_it(tmp_path):
    """받아 둔 tokenizer는 root의 `manifest.json`(id → `<root>/<id>/tokenizer.json`)이 기술한다. id로 열든, 디렉터리나
    파일 경로로 열든 그 manifest의 sha256과 실제 파일을 대조하고 다르면 거절한다 — 다른 revision의 파일이 같은
    자리에 놓이면 토큰 수·어휘가 조용히 달라지기 때문이다."""
    target = tmp_path / "Some" / "Model"
    target.mkdir(parents=True)
    tiny_tokenizer_json(target / "tokenizer.json")
    digest = sha256_of(target / "tokenizer.json")
    (tmp_path / MANIFEST_NAME).write_text(json.dumps(existing_manifest("Some/Model", target / "tokenizer.json")), encoding="utf-8")
    references = ("Some/Model", target, target / "tokenizer.json")
    for reference in references:
        assert load_tokenizer(reference, root=tmp_path).encode("hello world", add_special_tokens=False).ids == [1, 2]
    assert describe_tokenizer("Some/Model", root=tmp_path) == {
        "kind": "tokenizers", "file": str(target / "tokenizer.json"), "sha256": digest,
        "id": "Some/Model", "revision": "abc123", "manifest": str(tmp_path / MANIFEST_NAME),
    }  # fmt: skip
    assert describe_tokenizer(target / "tokenizer.json", root=tmp_path)["id"] == "Some/Model"

    tiny_tokenizer_json(target / "tokenizer.json", vocab={"[UNK]": 0, "world": 1, "hello": 2, "\n": 3})  # 다른 어휘
    changed = sha256_of(target / "tokenizer.json")
    assert changed != digest
    for reference in references:
        with pytest.raises(ValueError, match="sha256") as excinfo:
            load_tokenizer(reference, root=tmp_path)
        message = str(excinfo.value)
        assert "Some/Model" in message and digest[:12] in message and changed[:12] in message
        assert str(tmp_path / MANIFEST_NAME) in message and FETCH_SCRIPT in message
    with pytest.raises(ValueError, match="sha256"):
        describe_tokenizer("Some/Model", root=tmp_path)
    # manifest를 실제 파일에 맞추면(다시 받거나 해시를 갱신) 읽힌다
    record = existing_manifest("Some/Model", target / "tokenizer.json")
    (tmp_path / MANIFEST_NAME).write_text(json.dumps(record), encoding="utf-8")
    assert load_tokenizer("Some/Model", root=tmp_path).encode("hello", add_special_tokens=False).ids == [2]
    assert describe_tokenizer("Some/Model", root=tmp_path)["sha256"] == changed


def test_a_manifest_next_to_a_bare_tokenizer_file_describes_it_and_a_foreign_manifest_does_not(tmp_path):
    """파일과 같은 디렉터리의 manifest(전송한 tokenizer.json + manifest.json 한 쌍)는 id 없이도 그 파일을 기술한다.
    root manifest가 다른 id를 가리키면 그 파일은 기술되지 않은 것이고, 대조 없이 읽히되 `describe_tokenizer`가
    id·revision·manifest를 `None`으로 적는다."""
    tiny_tokenizer_json(tmp_path / "tokenizer.json")
    digest = sha256_of(tmp_path / "tokenizer.json")
    (tmp_path / MANIFEST_NAME).write_text(json.dumps({"revision": "r1", "files": {"tokenizer.json": digest}}), encoding="utf-8")
    assert load_tokenizer(tmp_path / "tokenizer.json").encode("world", add_special_tokens=False).ids == [2]
    assert describe_tokenizer(tmp_path)["manifest"] == str(tmp_path / MANIFEST_NAME)
    assert describe_tokenizer(tmp_path)["revision"] == "r1" and describe_tokenizer(tmp_path)["id"] is None
    (tmp_path / MANIFEST_NAME).write_text(json.dumps({"revision": "r1", "files": {"tokenizer.json": "f" * 64}}), encoding="utf-8")
    with pytest.raises(ValueError, match="sha256"):
        load_tokenizer(tmp_path)

    root = tmp_path / "root"
    described = root / "Some" / "Model"
    other = root / "Other" / "Model"
    for directory in (described, other):
        directory.mkdir(parents=True)
        tiny_tokenizer_json(directory / "tokenizer.json")
    (root / MANIFEST_NAME).write_text(json.dumps(existing_manifest("Some/Model", described / "tokenizer.json")), encoding="utf-8")
    assert load_tokenizer("Other/Model", root=root).encode("hello", add_special_tokens=False).ids == [1]
    assert describe_tokenizer("Other/Model", root=root) == {
        "kind": "tokenizers", "file": str(other / "tokenizer.json"), "sha256": sha256_of(other / "tokenizer.json"),
        "id": None, "revision": None, "manifest": None,
    }  # fmt: skip
    tiny_tokenizer_json(described / "tokenizer.json", vocab={"[UNK]": 0, "x": 1})
    with pytest.raises(ValueError, match="sha256"):
        load_tokenizer("Some/Model", root=root)
    assert load_tokenizer("Other/Model", root=root).encode("hello", add_special_tokens=False).ids == [1]


def test_the_fetched_real_tokenizer_matches_its_manifest():
    found = available_tokenizer()
    if found is None:
        pytest.skip(f"실제 tokenizer가 없다 — `uv run python {FETCH_SCRIPT}`로 받는다")
    described = describe_tokenizer(found[0])
    assert described["id"] == found[0] and described["file"] == str(found[1]) and described["manifest"]
    assert described["revision"] and described["sha256"] == sha256_of(found[1])
