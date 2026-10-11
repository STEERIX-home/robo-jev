"""문구 판(phrasing): v0b(봉인된 시험 세트와 v0b 학습 세트의 문구)와 v01(학습 묶음만 넓힌 판) — Task v0.1a §1-1.

* **판.** ``v0b`` = :mod:`~robo_jev.simgen.phrases`·:mod:`~robo_jev.simgen.phrases_ko_data` 의 ``train``·``heldout`` 목록 그대로(바이트 단위로
  손대지 않는다). ``v01`` = 학습 묶음 ``train_v01`` = 기존 ``train`` + 새 문구(:mod:`~robo_jev.simgen.phrases_v01`,
  :mod:`~robo_jev.simgen.phrases_ko_v01`). 빼 둔 묶음(``heldout`` — T2 전용)은 두 판이 같다. 기본은 ``v0b``이고, ``v01``은 학습 나눔에서만
  쓸 수 있다(시험 세트 생성 경로는 늘 v0b — :func:`robo_jev.simgen.generate.assemble` 이 막는다).
* **난수.** v01의 고르기는 주 난수 흐름(``questions``)을 건드리지 않는다: v0b 묶음 크기로 뽑던 수는 그대로 뽑아 버리고(소비가 같다), 넓은
  묶음에서의 고르기는 레코드 seed와 열쇠 이름의 별도 흐름(:func:`pick_stream`)에서 한다. 그래서 같은 seed면 질문 이름·조리법·선택지·순서·
  정답이 v0b와 바이트 단위로 같고 글만 다르다(시험).
* **빼 둔 문구에 가까이 가지 않기**(시험이 고정): 새 문구는 같은 열쇠의 빼 둔 문구와 정규화(소문자, 문장부호 제거, 자리 표시 → ``{}``) 뒤
  같지 않고, 낱말(한국어는 어절) 단위 :class:`difflib.SequenceMatcher` 유사도가 0.8 미만이며, 빼 둔 문구에만 있는 낱말 3-gram(그 열쇠의
  빼 둔 문구에는 있고 v0b 학습 문구에는 없는 것)을 쓰지 않는다. 자리 표시는 그 열쇠의 v0b 학습 문구가 쓰는 것만 쓴다.

관측·세계를 모른다(문구 표만 다룬다).
"""

from __future__ import annotations

import difflib
import re
from functools import lru_cache

__all__ = [
    "DEFAULT_PHRASING",
    "MAX_SIMILARITY",
    "PHRASINGS",
    "TABLES",
    "bank_digest",
    "base_tables",
    "counterfactual_keys",
    "description_table",
    "fact_order_stats",
    "forbidden_ngrams",
    "meaning_violations",
    "new_tables",
    "ngrams",
    "normalize",
    "pick_stream",
    "pick_table",
    "placeholders",
    "proximity",
    "required_new",
    "similarity",
    "tokens",
    "violations",
    "wide",
    "without_text",
    "wording_stats",
]

PHRASINGS = ("v0b", "v01")
DEFAULT_PHRASING = "v0b"
#: 새 문구와 같은 열쇠의 빼 둔 문구 사이 낱말 유사도의 상한(미만이어야 한다)
MAX_SIMILARITY = 0.8
#: 문구 표의 이름: 질문 지시문·점수 기준 열쇠, 선택지 설명(갈래별)
TABLES = ("instructions", "desc.main", "desc.arm", "desc.head_on", "desc.p6", "desc.p4", "desc.turn", "desc.turn_full")

_SLOT = re.compile(r"\{([A-Za-z_][A-Za-z0-9_]*)(?::[^}]*)?\}")
_PUNCT = re.compile(r"[^\w{}\s]", re.UNICODE)


def placeholders(template: str) -> set[str]:
    """자리 표시 이름(한국어 조사 꼴 ``{ref:이}`` 는 이름 ``ref``)."""
    return set(_SLOT.findall(template))


def normalize(text: str) -> str:
    """소문자, 자리 표시 → ``{}``, 문장부호 → 빈칸, 빈칸 하나로(영어·한국어 같은 규칙)."""
    text = _SLOT.sub(" {} ", text.lower())
    text = _PUNCT.sub(" ", text)
    return " ".join(text.split())


def tokens(text: str) -> list[str]:
    """낱말(영어) 또는 어절(한국어) — 정규화한 글을 빈칸으로 나눈 것."""
    return normalize(text).split()


def ngrams(words: list[str], n: int = 3) -> set[tuple[str, ...]]:
    return {tuple(words[i : i + n]) for i in range(len(words) - n + 1)}


def similarity(a: str, b: str) -> float:
    """낱말(어절) 단위 SequenceMatcher 유사도."""
    return difflib.SequenceMatcher(None, tokens(a), tokens(b), autojunk=False).ratio()


# ----------------------------------------------------------------------------------------------------------------
# 표
# ----------------------------------------------------------------------------------------------------------------


@lru_cache(maxsize=2)
def base_tables(lang: str) -> dict[str, dict[str, dict[str, list[str]]]]:
    """v0b 문구 표: {표 이름: {열쇠: {"train": [...], "heldout": [...]}}} — 회전 표는 열쇠 하나(``turn``·``turn_full``)."""
    if lang == "ko":
        from robo_jev.simgen import phrases_ko_data as d

        return {
            "instructions": d.KO, "desc.main": d.DESCRIPTIONS_KO, "desc.arm": d.DESCRIPTIONS_ARM_KO, "desc.head_on": d.DESCRIPTIONS_HEAD_ON_KO,
            "desc.p6": d.DESCRIPTIONS_P6_KO, "desc.p4": d.DESCRIPTIONS_P4_KO, "desc.turn": {"turn": d.DESCRIPTIONS_TURN_KO},
            "desc.turn_full": {"turn_full": d.DESCRIPTIONS_TURN_FULL_KO},
        }  # fmt: skip
    from robo_jev.simgen import phrases as p

    return {
        "instructions": p.EN, "desc.main": p._DESCRIPTIONS, "desc.arm": p._DESCRIPTIONS_ARM, "desc.head_on": p._DESCRIPTIONS_HEAD_ON,
        "desc.p6": p._DESCRIPTIONS_P6, "desc.p4": p._DESCRIPTIONS_P4, "desc.turn": {"turn": p._DESCRIPTIONS_TURN},
        "desc.turn_full": {"turn_full": p._DESCRIPTIONS_TURN_FULL},
    }  # fmt: skip


@lru_cache(maxsize=2)
def new_tables(lang: str) -> dict[str, dict[str, list[str]]]:
    """v01의 새 문구: {표 이름: {열쇠: [새 문구 …]}}."""
    if lang == "ko":
        from robo_jev.simgen import phrases_ko_v01 as m
    else:
        from robo_jev.simgen import phrases_v01 as m
    return {"instructions": m.NEW, **{f"desc.{name}": table for name, table in m.NEW_DESCRIPTIONS.items()}}


def wide(lang: str, table: str, key: str) -> list[str]:
    """v01 학습 묶음(``train_v01``) = v0b 학습 묶음 + 새 문구(앞부분은 v0b와 같은 순서)."""
    return list(base_tables(lang)[table][key]["train"]) + list(new_tables(lang)[table].get(key, ()))


def description_table(canon: str, scene: str, body: str) -> tuple[str, str]:
    """선택지 설명이 쓰는 표와 그 안의 열쇠(:func:`robo_jev.simgen.phrases.option_descriptions` 와 같은 고름 — 두 언어 같다)."""
    from robo_jev.simgen import phrases as p

    if canon.startswith("turn_"):
        return ("desc.turn_full", "turn_full") if int(canon.split("_")[1]) >= 100 else ("desc.turn", "turn")
    if scene.startswith("p4") and canon in p._DESCRIPTIONS_P4:
        return "desc.p4", canon
    if body == "arm" and canon in p._DESCRIPTIONS_ARM:
        return "desc.arm", canon
    if scene == "p1_head_on" and canon in p._DESCRIPTIONS_HEAD_ON:
        return "desc.head_on", canon
    if scene.startswith(("p6", "p1p6")) and canon in p._DESCRIPTIONS_P6:
        return "desc.p6", canon
    return "desc.main", canon


def pick_stream(seed: int, name: str):
    """v01 문구 고르기의 별도 흐름(레코드 seed + 열쇠 이름) — 주 흐름과 독립."""
    from robo_jev.simgen.spec import stream

    return stream(seed, "phrasing-v01", name)


# ----------------------------------------------------------------------------------------------------------------
# 규칙(시험이 쓴다)
# ----------------------------------------------------------------------------------------------------------------


@lru_cache(maxsize=1)
def counterfactual_keys() -> frozenset[str]:
    """영어 가정 질문 열쇠: v0b 학습 문구가 ``{action}``(가정 행동)을 쓰는 열쇠 — noul 17·score 22."""
    from robo_jev.simgen import phrases as p

    return frozenset(k for k, bank in p.EN.items() if any("{action}" in t for t in bank["train"]))


def required_new(lang: str, table: str, key: str) -> int:
    """열쇠마다 새 문구의 최소 수: 영어 가정 질문(``{action}``을 쓰는 열쇠) 6, 그 밖 영어 4, 한국어 2."""
    if lang == "ko":
        return 2
    if table == "instructions" and key in counterfactual_keys():
        return 6
    return 4


def forbidden_ngrams(lang: str, table: str, key: str) -> set[tuple[str, ...]]:
    """그 열쇠의 빼 둔 문구에는 있고 v0b 학습 문구에는 없는 낱말 3-gram."""
    bank = base_tables(lang)[table][key]
    held = set().union(*(ngrams(tokens(t)) for t in bank["heldout"]))
    train = set().union(*(ngrams(tokens(t)) for t in bank["train"]))
    return held - train


def proximity(lang: str, table: str, key: str, template: str) -> tuple[float, list[str]]:
    """(같은 열쇠의 빼 둔 문구와의 최대 유사도, 쓴 금지 3-gram)."""
    bank = base_tables(lang)[table][key]
    best = max(similarity(template, h) for h in bank["heldout"])
    banned = forbidden_ngrams(lang, table, key)
    hits = sorted(" ".join(g) for g in ngrams(tokens(template)) if g in banned)
    return best, hits


_COLLISION_EN = re.compile(r"collision|bump|knock", re.I)
_WINDOW_EN = re.compile(r"3 s(?:econds)?\b|3 seconds|next 3", re.I)
_COLLISION_KO = re.compile(r"충돌|부딪")
_IMPACT = ("noul.spill_impact", "noul.rim_impact", "noul.fall_impact", "score.headroom_impact", "score.rim_impact", "score.spill_impact", "score.fall_impact")
_NO_COLLISION = ("noul.spill_slosh", "noul.rim_slosh", "score.headroom_slosh", "score.rim_slosh", "score.spill_slosh", "noul.fall", "noul.tip", "noul.slide",
                 "score.fall", "score.slide")  # fmt: skip
_CONTACT = ("noul.contact", "noul.contact_any", "noul.contact_obstacle", "noul.contact_emerge", "noul.contact_stale", "score.contact", "score.contact_stale",
            "score.contact_emerge", "score.contact_obstacle", "score.closeness")  # fmt: skip
#: 모델 입력에 나오면 안 되는 숨은 이름(누출 시험 test_simgen_leak.HIDDEN_NAMES의 부분 — 문구에 낱말로 들어갈 수 있는 것)
_HIDDEN = ("intent", "slosh", "true_", "posterior", "k_ref")
#: 한국어: 자리 표시 바로 뒤에 그대로 적으면 안 되는 바뀌는 조사(받침에 따라 꼴이 바뀐다 — {name:조사}로 쓴다)
_KO_VARIABLE_JOSA = re.compile(r"\{[A-Za-z_][A-Za-z0-9_]*\}(?:이|가|은|는|을|를|과|와|으로|로|아|야)")


def meaning_violations(lang: str, table: str, key: str, template: str) -> list[str]:
    """뜻 규칙(v0b 수정의 I2·창·누출 — 문구 표 수준): 충돌을 세는 열쇠는 충돌을 말하고 세지 않는 열쇠는 말하지 않는다, 닿음·가까움은
    3초 창, 가까움은 지금 간격도 세고 '지나가다'를 쓰지 않는다, 숨은 이름 없음, 한국어 조사 자리 표시."""
    out = []
    lower = template.lower()
    if any(h in lower for h in _HIDDEN):
        out.append("hidden name")
    if table == "instructions":
        if lang == "ko":
            if key in _IMPACT and not _COLLISION_KO.search(template):
                out.append("impact key must name the collision")
            if key in _NO_COLLISION and _COLLISION_KO.search(template):
                out.append("motion/slosh key must not mention a collision")
            if key in _CONTACT and "3초" not in template:
                out.append("contact/closeness needs the 3-second window")
            if key == "score.closeness" and ("지금" not in template or re.search("지나가|지나치", template)):
                out.append("closeness must count the gap now and not say 'pass'")
        else:
            if key in _IMPACT and not _COLLISION_EN.search(template):
                out.append("impact key must name the collision")
            if key in _NO_COLLISION and _COLLISION_EN.search(template):
                out.append("motion/slosh key must not mention a collision")
            if key in _CONTACT and not _WINDOW_EN.search(template):
                out.append("contact/closeness needs the 3-second window")
            if key == "score.closeness" and ("now" not in lower or re.search(r"\bpass\b", template)):
                out.append("closeness must count the gap now and not say 'pass'")
    if lang == "ko" and _KO_VARIABLE_JOSA.search(template):
        out.append("variable particle written after a slot (use {name:조사})")
    if lang == "ko" and table.startswith("desc.") and re.search(r"\{[A-Za-z_][A-Za-z0-9_]*:", template):
        out.append("descriptions take no particle slots")
    return out


def violations(lang: str, table: str, key: str, new: list[str]) -> list[str]:
    """한 열쇠의 새 문구 목록에 대한 위반(비면 통과): 수, 중복, 빼 둔 문구 근접·3-gram, 자리 표시, 뜻 규칙."""
    bank = base_tables(lang)[table].get(key)
    if bank is None:
        return [f"{lang}:{table}:{key}: unknown key"]
    out = []
    where = f"{lang}:{table}:{key}"
    need = required_new(lang, table, key)
    if len(new) < need:
        out.append(f"{where}: {len(new)} new wordings, need >= {need}")
    if len(bank["train"]) + len(new) < (7 if lang == "en" else 0):
        out.append(f"{where}: fewer than 7 training wordings in total")
    allowed = set().union(*(placeholders(t) for t in bank["train"]))
    seen = {normalize(t) for t in bank["train"]}
    held = {normalize(t) for t in bank["heldout"]}
    for template in new:
        norm = normalize(template)
        if norm in held:
            out.append(f"{where}: equals a held-out wording: {template!r}")
        if norm in seen:
            out.append(f"{where}: duplicate wording: {template!r}")
        seen.add(norm)
        best, hits = proximity(lang, table, key, template)
        if best >= MAX_SIMILARITY:
            out.append(f"{where}: similarity {best:.3f} to a held-out wording: {template!r}")
        if hits:
            out.append(f"{where}: held-out-only 3-grams {hits}: {template!r}")
        extra = placeholders(template) - allowed
        if extra:
            out.append(f"{where}: placeholders {sorted(extra)} not used by the v0b training wordings: {template!r}")
        out += [f"{where}: {v}: {template!r}" for v in meaning_violations(lang, table, key, template)]
    return out


# ----------------------------------------------------------------------------------------------------------------
# 통계(학습 세트의 문구 쓰임 — meta.phrasing.picks에서)
# ----------------------------------------------------------------------------------------------------------------


def pick_table(name: str) -> tuple[str, str]:
    """고른 문구의 이름(:attr:`QuestionSet.picks` 의 열쇠) → (표, 열쇠). 설명은 ``desc.<갈래>.<정식 선택지>``(회전은 표 열쇠 하나)."""
    if not name.startswith("desc."):
        return "instructions", name
    _, group, canon = name.split(".", 2)
    table = f"desc.{group}"
    return (table, group) if group in ("turn", "turn_full") else (table, canon)


def wording_stats(records) -> dict:
    """언어 × 표 × 열쇠마다: 쓰인 횟수, v0b가 썼을 문구 수(같은 seed의 주 흐름 번호), v01이 쓴 문구 수, 새 문구의 몫 — v01 레코드의
    ``meta.phrasing.picks``([열쇠, v0b 번호, v01 번호])에서."""
    from collections import defaultdict

    uses: dict = defaultdict(lambda: {"uses": 0, "v0b": set(), "v01": set(), "new": 0})
    for record in records:
        lang = record["meta"]["language"]
        for name, i_v0b, i_v01 in (record["meta"].get("phrasing") or {}).get("picks") or ():
            table, key = pick_table(name)
            row = uses[(lang, table, key)]
            row["uses"] += 1
            row["v0b"].add(i_v0b)
            row["v01"].add(i_v01)
            row["new"] += i_v01 >= len(base_tables(lang)[table][key]["train"])
    out: dict = {}
    for (lang, table, key), row in sorted(uses.items()):
        size_v0b = len(base_tables(lang)[table][key]["train"])
        out.setdefault(lang, {}).setdefault(table, {})[key] = {
            "uses": row["uses"], "bank_v0b": size_v0b, "bank_v01": len(wide(lang, table, key)), "used_v0b": len(row["v0b"]),
            "used_v01": len(row["v01"]), "new_share": round(row["new"] / row["uses"], 4),
        }  # fmt: skip
    return out


def fact_order_stats(records) -> dict:
    """사실 순서 섞기의 비율(장치를 켠 비율, 글이 실제로 바뀐 비율 — 문체별)."""
    from collections import Counter

    on, changed, total = Counter(), Counter(), Counter()
    for record in records:
        fo = record["meta"].get("fact_order")
        style = record["meta"]["style"]
        total[style] += 1
        if fo and fo["shuffled"]:
            on[style] += 1
            changed[style] += bool(fo["changed"])
    n = sum(total.values())
    return {
        "records": n, "shuffled_share": round(sum(on.values()) / n, 4) if n else None, "changed_share": round(sum(changed.values()) / n, 4) if n else None,
        "by_style": {s: {"n": total[s], "shuffled_share": round(on[s] / total[s], 4), "changed_share": round(changed[s] / total[s], 4)} for s in sorted(total)},
    }  # fmt: skip


def bank_digest(lang: str) -> str:
    """새 문구 표의 sha256(manifest에 적는다)."""
    import hashlib
    import json

    return hashlib.sha256(json.dumps(new_tables(lang), ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()


def without_text(record: dict) -> dict:
    """맞대기용 레코드: 요청 글(상태 글·지시문·선택지 설명 — 질문 이름·종류·선택지 열쇠·점수 기준은 남긴다), v01 장부(``meta.phrasing``·
    ``meta.fact_order``·``generator.phrasing``·``generator.fact_shuffle``)와 생성기 git을 뺀 것. 같은 seed의 v0b·v01 레코드는 이것이 같아야
    한다(정답·참 세계·계산 과정·장부 전부)."""
    import json

    out = json.loads(json.dumps(record))
    out["generator"].pop("git", None)
    for key in ("phrasing", "fact_shuffle"):
        out["generator"].pop(key, None)
    for key in ("phrasing", "fact_order"):
        out["meta"].pop(key, None)
    request = out.pop("request")
    out["questions"] = {name: {"type": q["type"], "criteria": sorted(q["criteria"]) if isinstance(q.get("criteria"), dict) else q.get("criteria")}
                        for name, q in request["questions"].items()}  # fmt: skip
    return out
