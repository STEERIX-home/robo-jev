"""배포 계약 digest — 직렬화 코드·질문 세트·하네스 버전·tokenizer의 지문 (Task 2b G0b S1.1, analysis-nimble §3-4).

학습된 readout(과 LoRA)은 **어떤 토큰 서식·질문 세트·표지·tokenizer**로 만든 입력에 맞춰진 것이다. 그 계약이 조용히
바뀐 체크아웃에서 checkpoint를 싣거나 서빙하면 숫자만 그럴듯한 잘못된 판단이 나온다. 그래서 학습 manifest와 모든
checkpoint에 ``contract_sha256``을 적고, 적재·서빙은 지금 체크아웃의 digest와 대조해 다르면 **거절**한다
(Nimble의 ``schema_config.json``이 프롬프트 코드 sha256으로 하는 것과 같다).

digest = sha256( ``serialize.py`` 소스 ‖ ``contracts.py`` 소스 ‖ 하네스 버전 문자열 ‖ tokenizer 파일 sha256 ), 조각마다
``이름\\n길이\\n내용``으로 구분해 잇는다. 조각별 sha256도 함께 적어 어느 조각이 달라졌는지 말할 수 있게 한다.
하네스 버전은 `configs/harness/robot.yaml`의 ``version``(코드의 ``HARNESS_VERSION``과 같아야 한다 — 하네스 검사가
대조)이다: 학습 코드는 하네스를 import하지 않으므로(docs/06 §1) 설정 파일의 문자열을 읽는다.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any

__all__ = ["CONTRACT_PARTS", "contract_digest", "contract_differences", "harness_version"]

_PACKAGE_ROOT = Path(__file__).resolve().parents[3]
_MODEL_DIR = Path(__file__).resolve().parent
#: digest에 드는 조각, 순서대로.
CONTRACT_PARTS = ("serialize_py", "contracts_py", "harness_version", "tokenizer_sha256")
_HARNESS_CONFIG = _PACKAGE_ROOT / "configs" / "harness" / "robot.yaml"


def harness_version(config: str | Path = _HARNESS_CONFIG) -> str:
    """`configs/harness/robot.yaml`의 ``version:`` 문자열 (yaml 전체를 파싱하지 않고 그 줄만 읽는다)."""
    text = Path(config).read_text(encoding="utf-8")
    match = re.search(r"^version:\s*(\S+)\s*$", text, flags=re.MULTILINE)
    if match is None:
        raise ValueError(f"{config}: `version:` 줄이 없다")
    return match.group(1).strip("'\"")


def contract_digest(
    tokenizer_sha256: str,
    *,
    serialize_source: bytes | None = None,
    contracts_source: bytes | None = None,
    harness: str | None = None,
) -> dict[str, Any]:
    """``{"sha256", "parts": {조각: sha256}, "harness_version", "tokenizer_sha256"}``.

    기본은 이 체크아웃의 소스·설정을 읽는다. 검사는 조각을 직접 넣어(한 바이트 바꾼 사본 등) digest가 바뀌는지 본다.
    """
    serialize_source = (_MODEL_DIR / "serialize.py").read_bytes() if serialize_source is None else serialize_source
    contracts_source = (_MODEL_DIR.parent / "contracts.py").read_bytes() if contracts_source is None else contracts_source
    harness = harness_version() if harness is None else str(harness)
    pieces = {
        "serialize_py": serialize_source,
        "contracts_py": contracts_source,
        "harness_version": harness.encode("utf-8"),
        "tokenizer_sha256": str(tokenizer_sha256).lower().encode("utf-8"),
    }
    digest = hashlib.sha256()
    parts: dict[str, str] = {}
    for name in CONTRACT_PARTS:
        content = pieces[name]
        parts[name] = hashlib.sha256(content).hexdigest()
        digest.update(f"{name}\n{len(content)}\n".encode("utf-8"))
        digest.update(content)
    return {"sha256": digest.hexdigest(), "parts": parts, "harness_version": harness, "tokenizer_sha256": str(tokenizer_sha256).lower()}


def contract_differences(saved: dict[str, Any] | None, current: dict[str, Any]) -> list[str]:
    """저장된 digest 블록과 지금 것이 다른 조각 이름들 (같으면 빈 목록). 저장된 블록이 없으면 ``["missing"]``."""
    if not isinstance(saved, dict) or "sha256" not in saved:
        return ["missing"]
    if saved["sha256"] == current["sha256"]:
        return []
    saved_parts = saved.get("parts") or {}
    out = [name for name in CONTRACT_PARTS if saved_parts.get(name) != current["parts"].get(name)]
    return out or ["sha256"]
