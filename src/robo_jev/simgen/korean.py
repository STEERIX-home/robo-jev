"""한국어 도우미 (Task v0b 지시서 §3): 조사 고르기, 조사 자리 표시가 있는 문구 채우기, 상황 옷의 한국어 문구 표.

* :func:`josa` — 낱말의 끝소리(받침)에 맞는 조사: 이/가, 은/는, 을/를, 과/와, 으로/로(ㄹ 받침은 '로'), 아/야, 이다/다(…이에요 대신 쓰지 않는다).
  끝 글자가 한글이 아니면 읽는 소리로 정한다(숫자: 일·삼·육·칠·팔·십·영처럼 받침이 있는 것, 영문 대문자: 엘·엠·엔·알 …, 단위: m(미터) 등).
* :func:`kfmt` — ``{name}``·``{name:조사}`` 자리 표시(예: ``{ref:이}`` → '피커가', ``{container:을}`` → '머그잔을'). 조사는 늘 받침이
  있을 때의 꼴로 적는다(이·은·을·과·으로·아).
* :class:`KoSkins` — ``configs/simgen/skins-ko-v0.2.yaml``(영어 문구 → 한국어)을 읽고, 상황 옷 문구를 한국어로 바꾼다. :func:`skin_strings`
  는 카드에서 한국어가 있어야 하는 영어 문구를 갈래별로 모은다(시험이 빠짐없이 있는지 본다).

세계·관측 타입을 모른다(맨 아래 층).
"""

from __future__ import annotations

import functools
import re
from dataclasses import dataclass
from pathlib import Path

import yaml

__all__ = ["KO_PATH", "KoSkins", "has_batchim", "josa", "kfmt", "ko_skins", "skin_strings"]

KO_PATH = Path(__file__).resolve().parents[3] / "configs" / "simgen" / "skins-ko-v0.2.yaml"

# 숫자를 읽을 때의 끝소리: 0 영, 1 일, 2 이, 3 삼, 4 사, 5 오, 6 육, 7 칠, 8 팔, 9 구 — (받침 있음, 받침이 ㄹ인가)
_DIGIT = {"0": (True, False), "1": (True, True), "2": (False, False), "3": (True, False), "4": (False, False), "5": (False, False),
          "6": (True, False), "7": (True, True), "8": (True, True), "9": (False, False)}  # fmt: skip
# 영문 글자 이름의 끝소리: L 엘·M 엠·N 엔·R 알(받침), 그 밖은 받침 없음
_LETTER = {"L": (True, True), "M": (True, False), "N": (True, False), "R": (True, True)}
# 단위·약어를 읽는 소리(끝에 올 때): m 미터, cm 센티미터, mm 밀리미터, s 초, ms 밀리초, km/h 킬로미터, m/s 미터, kg 킬로그램, N 뉴턴, % 퍼센트
_UNITS = {"m": (False, False), "cm": (False, False), "mm": (False, False), "s": (False, False), "ms": (False, False), "km/h": (False, False),
          "m/s": (False, False), "mm/s": (False, False), "kg": (True, False), "N": (True, False), "%": (False, False), "°": (False, False),
          "m/s²": (False, False)}  # fmt: skip
_PAIRS = {"이": ("이", "가"), "은": ("은", "는"), "을": ("을", "를"), "과": ("과", "와"), "아": ("아", "야"), "으로": ("으로", "로"), "이다": ("이다", "다")}


def _final(word: str) -> tuple[bool, bool]:
    """(받침이 있나, 그 받침이 ㄹ인가) — 낱말의 끝 소리."""
    text = word.rstrip()
    for unit in sorted(_UNITS, key=len, reverse=True):
        if text.endswith(unit) and (len(text) == len(unit) or not text[-len(unit) - 1].isalpha()):
            return _UNITS[unit]
    for ch in reversed(text):
        if "가" <= ch <= "힣":
            code = (ord(ch) - 0xAC00) % 28
            return code != 0, code == 8
        if ch.isdigit():
            return _DIGIT[ch]
        if ch.isalpha():
            return _LETTER.get(ch.upper(), (False, False))
        # 괄호·따옴표·마침표는 건너뛰고 그 앞 글자로
    return False, False


def has_batchim(word: str) -> bool:
    return _final(word)[0]


def josa(word: str, particle: str) -> str:
    """낱말 뒤에 붙일 조사(받침 있을 때의 꼴로 부른다: 이·은·을·과·아·으로·이다)."""
    with_final, without = _PAIRS[particle]
    final, rieul = _final(word)
    if particle == "으로":
        return "로" if (not final or rieul) else "으로"
    return with_final if final else without


_SLOT = re.compile(r"\{([A-Za-z_][A-Za-z0-9_]*)(?::(이|은|을|과|아|으로|이다))?\}")


def kfmt(template: str, **values) -> str:
    """``{name}``·``{name:조사}`` 자리 표시를 채운다(조사는 채운 낱말의 받침에 맞춘다)."""

    def fill(match: re.Match) -> str:
        name, particle = match.group(1), match.group(2)
        if name not in values:
            raise KeyError(name)
        value = str(values[name])
        return value + josa(value, particle) if particle else value

    return _SLOT.sub(fill, template)


# ----------------------------------------------------------------------------------------------------------------
# 상황 옷의 한국어 문구
# ----------------------------------------------------------------------------------------------------------------


def _text(entry, key: str = "text") -> str:
    return entry if isinstance(entry, str) else entry[key]


def skin_strings(skins: dict) -> dict[str, list[str]]:
    """상황 옷 카드에서 한국어가 있어야 하는 영어 문구(갈래별, 중복 없이 처음 나온 순서).

    * ``text`` — 그대로 바꾸는 문구: 상황(문장), 몸 이름·나르는 곳(…에), 과제(‘-는’으로 끝나 '중입니다' 앞에 오는 마디), 장소·장애물·가림·
      막힘·경사로·경사로 끝·돌아가는 길·쌓을 곳·모퉁이·센서 단서 물체·받침 표면·놓는 물건·놓는 곳(명사구), 방해 문구(문장),
      오래된 카메라의 까닭(문장).
    * ``descriptors`` — 상대 묘사(np: 처음 소개하는 명사구, ref: 다시 가리키는 짧은 명사).
    * ``objects`` — 용기·키 큰 물건·쌓을 물건 문구(np, ref).
    * ``cues`` — 단서 문장 틀({Ref}/{ref} → {ref:조사}).
    """
    out: dict[str, list[str]] = {"text": [], "descriptors": [], "objects": [], "cues": []}
    seen: dict[str, set] = {k: set() for k in out}

    def add(kind: str, value: str) -> None:
        if value and value not in seen[kind]:
            seen[kind].add(value)
            out[kind].append(value)

    for names in skins.get("body_names", {}).values():
        for n in names:
            add("text", _text(n))
    for where in skins.get("carry_default", {}).values():
        add("text", where)
    for entries in skins.get("decks", {}).values():
        for e in entries:
            add("text", e["text"])
    for entries in skins.get("stale_camera", {}).values():
        for e in entries:
            add("text", e)
    for group in skins.get("cue_phrases", {}).values():
        for phrases in group.values():
            for p in phrases:
                add("cues", p)
    for card in skins.get("domains", {}).values():
        add("text", card["name"])
        for s in card["settings"]:
            add("text", _text(s))
        for names in card.get("body_names", {}).values():
            for n in names:
                add("text", _text(n))
                if isinstance(n, dict):
                    if n.get("carry"):
                        add("text", n["carry"])
                    for key in ("tasks", "carry_tasks"):
                        for task in n.get(key, []):
                            add("text", _text(task))
        for places in card.get("places", {}).values():
            for p in places:
                add("text", _text(p))
        for key in ("tasks", "carry_tasks"):
            for entries in card.get(key, {}).values():
                for e in entries:
                    add("text", _text(e))
        for key in ("obstacles", "bench_obstacles", "occluders", "blockers", "distractors", "ramps", "ramp_ends", "detours", "stack_spots", "corners"):
            for e in card.get(key, []):
                add("text", _text(e))
        for e in card.get("conflicts", []):
            for p in e["phrases"]:
                add("text", p)
        for e in card.get("p4_items", []):
            add("text", e["item"])
            add("text", e["target"])
        for kind, entries in card.get("descriptors", {}).items():
            for d in entries:
                add("descriptors", d["np"])
        for key in ("containers", "loads", "stack_items"):
            for e in card.get(key, []):
                for p in e["phrases"]:
                    add("objects", p)
    return out


@dataclass(frozen=True)
class KoSkins:
    """영어 상황 옷 문구 → 한국어."""

    version: str
    text: dict
    descriptors: dict
    objects: dict
    cues: dict

    def t(self, english: str) -> str:
        return self.text[english]

    def descriptor(self, english_np: str) -> tuple[str, str]:
        d = self.descriptors[english_np]
        return d["np"], d["ref"]

    def obj(self, english: str) -> tuple[str, str]:
        d = self.objects[english]
        return d["np"], d["ref"]

    def cue(self, english_template: str) -> str:
        return self.cues[english_template]

    def missing(self, skins: dict) -> dict[str, list[str]]:
        """카드에 있는데 한국어가 없는 문구(갈래별)."""
        want = skin_strings(skins)
        have = {"text": self.text, "descriptors": self.descriptors, "objects": self.objects, "cues": self.cues}
        return {k: [s for s in v if s not in have[k]] for k, v in want.items() if any(s not in have[k] for s in v)}


@functools.lru_cache(maxsize=2)
def ko_skins(path: Path = KO_PATH) -> KoSkins:
    with path.open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    return KoSkins(data["version"], data.get("text", {}), data.get("descriptors", {}), data.get("objects", {}), data.get("cues", {}))
