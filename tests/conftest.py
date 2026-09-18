"""pytest fixture만 둔다. 도우미는 `tests/helpers.py`에 있다."""

import json

import pytest
from helpers import D0, D0_MANIFEST, D0_STREAMS, read_jsonl


@pytest.fixture(scope="module")
def manifest() -> dict:
    return json.loads(D0_MANIFEST.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def singles() -> list[dict]:
    return read_jsonl(D0)


@pytest.fixture(scope="module")
def streams() -> list[dict]:
    return read_jsonl(D0_STREAMS)
