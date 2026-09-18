"""검사 공용 도우미.

fixture 파일 경로, 빌더 적재, 키 스캔처럼 두 검사 파일이 함께 쓰는 것만 둔다.
pytest fixture는 `tests/conftest.py`에 있다.
"""

import functools
import importlib.util
import json
import sys
from pathlib import Path

FIXTURES = Path(__file__).parent / "fixtures"
D0 = FIXTURES / "d0.jsonl"
D0_STREAMS = FIXTURES / "d0_streams.jsonl"
D0_MANIFEST = FIXTURES / "d0_manifest.json"
BUILDER = FIXTURES / "build_d0.py"


def all_keys(node) -> set[str]:
    """중첩 구조 안의 모든 키 이름. 정보 경계 검사가 쓴다."""
    keys: set[str] = set()
    if isinstance(node, dict):
        for key, value in node.items():
            keys.add(key)
            keys |= all_keys(value)
    elif isinstance(node, list):
        for value in node:
            keys |= all_keys(value)
    return keys


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


@functools.lru_cache(maxsize=1)
def load_builder():
    """빌더는 패키지가 아니라 스크립트라서 경로로 불러온다."""
    spec = importlib.util.spec_from_file_location("build_d0", BUILDER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclass가 자기 모듈을 찾을 수 있어야 한다
    spec.loader.exec_module(module)
    return module
