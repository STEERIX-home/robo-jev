"""상황 옷 카드의 항목 고르기 — 의미 일관성 규칙 (리뷰 1 m1, skins-v0.1 머리말).

카드 목록의 항목은 문자열(어디에나 쓴다)이거나 제약을 가진 dict다:

* ``tags``   — 고른 상황(setting)의 태그와 하나라도 겹쳐야 쓴다(식품 공장에 에폭시 경화제가 나오지 않게).
* ``bodies`` — 그 몸일 때만 쓴다(고정 팔에 '배터리 61 %'가 나오지 않게).
* ``scenes`` — 그 장면일 때만 쓴다(과일 따는 팔은 비커를 나르는 장면에 나오지 않는다).

이 모듈은 장면 생성(:mod:`~robo_jev.simgen.scenes`)과 설정 검사(:mod:`~robo_jev.simgen.catalog`)가 같이 쓴다 — 고르는 규칙과 검사하는
규칙이 같다. 세계·관측 타입을 모른다.
"""

from __future__ import annotations

__all__ = ["compatible", "entry_ok", "entry_text", "setting_allows", "setting_tags"]


def entry_text(entry, key: str = "text") -> str:
    """항목의 글(문자열 항목이면 그대로)."""
    return entry if isinstance(entry, str) else entry[key]


def entry_ok(entry, tags: tuple[str, ...] | list[str], body: str | None, scene: str | None) -> bool:
    """항목이 그 상황 태그·몸·장면에서 쓸 수 있나."""
    if isinstance(entry, str):
        return True
    want = entry.get("tags")
    if want and not set(want) & set(tags):
        return False
    bodies = entry.get("bodies")
    if bodies and body is not None and body not in bodies:
        return False
    scenes = entry.get("scenes")
    if scenes and scene is not None and scene not in scenes:
        return False
    return True


def compatible(entries, tags, body: str | None, scene: str | None) -> list:
    """쓸 수 있는 항목만(순서 유지)."""
    return [e for e in entries if entry_ok(e, tags, body, scene)]


def setting_allows(entry, body: str) -> bool:
    """상황이 그 몸을 허락하나(상황의 bodies — 약국·지하철 승강장에는 지게차가 없다)."""
    bodies = entry.get("bodies") if isinstance(entry, dict) else None
    return not bodies or body in bodies


def setting_tags(card: dict, index: int) -> tuple[str, ...]:
    entry = card["settings"][index]
    return tuple(entry.get("tags", ())) if isinstance(entry, dict) else ()
