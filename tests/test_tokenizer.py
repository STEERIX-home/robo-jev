"""tokenizer 적재 검사 — 실제 `tokenizer.json` 적재와 구조 검사용 공백 tokenizer.

실제 backbone tokenizer는 `scripts/fetch_tokenizer.py`가 받아 둔 것만 쓴다. 없으면 그 검사만
이유를 적고 건너뛰고, 공백 tokenizer가 구조 검사를 언제나 돌게 한다.
"""

import json

import pytest
from tokenizers import Tokenizer, models, pre_tokenizers

from robo_jev.model.tokenizer import (
    FETCH_SCRIPT,
    MANIFEST_NAME,
    TOKENIZER_CANDIDATES,
    WhitespaceTokenizer,
    available_tokenizer,
    load_tokenizer,
)


def tiny_tokenizer_json(path) -> None:
    """검사용 진짜 `tokenizer.json` — 단어 단위 어휘 하나."""
    vocab = {"[UNK]": 0, "hello": 1, "world": 2, "\n": 3}
    tokenizer = Tokenizer(models.WordLevel(vocab=vocab, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tokenizer.save(str(path))


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
