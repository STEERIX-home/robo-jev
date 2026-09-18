"""pytest fixture만 둔다. 도우미는 `tests/helpers.py`에 있다."""

import json
import warnings

import pytest
from helpers import D0, D0_MANIFEST, D0_STREAMS, read_jsonl

# robosuite 1.5.2의 `robosuite/__init__.py`는 `__logo__` 문자열에 잘못된 escape sequence
# (`'\ '`)를 갖고 있다. 새 venv에서 그 파일을 **처음** 바이트코드로 컴파일할 때 Python 3.11은
# DeprecationWarning(3.12+는 SyntaxWarning)을 내고, `filterwarnings = error` 아래에서는 그것이
# SyntaxError가 되어 수집이 멈춘다(`.pyc`가 생긴 뒤에는 재현되지 않는다). 그 import 한 번에만
# 두 경고를 무시한다 — 전역 ignore가 아니다. robosuite 상한을 올릴 때 이 조각이 아직
# 필요한지 확인한다.
with warnings.catch_warnings():
    warnings.simplefilter("ignore", DeprecationWarning)
    warnings.simplefilter("ignore", SyntaxWarning)
    import robosuite  # noqa: F401


@pytest.fixture(scope="module")
def manifest() -> dict:
    return json.loads(D0_MANIFEST.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def singles() -> list[dict]:
    return read_jsonl(D0)


@pytest.fixture(scope="module")
def streams() -> list[dict]:
    return read_jsonl(D0_STREAMS)
