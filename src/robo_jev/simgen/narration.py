"""Varied narration of training moments (Task v0.3a) — the same physical moment told in many voices and formats, every fact kept.

**Why.** The outside test T8 showed that v0.1 learned our engine's way of *writing*, not physics that carries over. v0.3 attacks
this with diversity. This module covers the text side: an LLM retells each moment's ``request.state`` in another voice, and an
automatic checker (:mod:`robo_jev.simgen.narration_checks`) drops every rewrite that loses, changes or adds a fact. **Labels,
truth, rationale and meta of the source record are copied unchanged**; only ``request.state`` (and, for voices that change the
person, the question texts) are new, and ``meta.narration`` records how.

**Voices** (:data:`VOICES`) — ten per language, chosen for breadth along four axes, not to resemble any test set:

* *person* — first person (the robot narrates), second person (someone talks to the robot, as the engine's texts do), third person
  (someone reports about the robot), no person (data formats), and a two-speaker dialogue;
* *format* — prose, labelled fields, a JSON dump, a table, bullet fragments, a dialogue;
* *register* — radio-clipped, technical log, formal report, informal note, chatty talk;
* *length* — from telegraphic (≈ 0.4× the original) to long-winded (≈ 2×).

Korean moments stay Korean, with their own set of ten (Korean registers: 반말 독백, 무전체, 개조식 일지, 해요체 메모, 합니다체 보고 …).
The engine's held-out T2 style (a time-stamped event log) is excluded on purpose: the ``log_format`` check rejects time-stamped
lines, so T2 keeps measuring an unseen style. No sealed text (T1–T7, G1, G2, T8a, T8b) is ever used as an example or sent.

**Questions.** For a voice that changes the person (first or third person), the questions are re-voiced by the same call: only
"you/your" → "I/my" (or → "the robot/its"; Korean 당신 → 나/로봇) with verb agreement; option keys, numbers, hypothetical actions and
time windows are unchanged (:func:`~robo_jev.simgen.narration_checks.check_question`). A question that fails keeps its original text.

**Sealed-set guard** (:class:`SealedIndex`). Before any request is sent, the source record and every text in the request are
checked against hashes of the sealed sets: the whole-state and whole-request hashes of T1–T7, G1, G2, T8a and T8b, and the text
units (:func:`robo_jev.data.general_sets.record_units` — the state, its lines and options of ≥ 5 words) of the sets written outside
our engine (G1, G2, T8a, T8b). A hit raises :class:`SealedTextError` and nothing is sent. Every sent request is logged with these
hashes, and :func:`audit_request_log` re-checks the log after a run.

**Plan** (:func:`build_plan`). Source moments are ordered by a stratified permutation over (scene, language), so any prefix of the
plan is a proportional stratified sample; each moment gets two different voices of its language, balanced within each stratum,
and a per-voice fact-order hint. The pilot is the plan's first 200 moments; the scale run continues the same order until the cap.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from robo_jev.simgen import narration_checks as checks

__all__ = [
    "CHAT_URL",
    "NARRATION_VERSION",
    "ORDER_HINTS",
    "PROMPT_VERSION",
    "SealedIndex",
    "SealedTextError",
    "VOICES",
    "Voice",
    "assemble_record",
    "audit_request_log",
    "build_messages",
    "build_plan",
    "chat_payload",
    "keep_list",
    "parse_reply",
    "payload_sha256",
    "render_state",
    "state_sha256",
]

NARRATION_VERSION = "v03a"
CHAT_URL = "https://openrouter.ai/api/v1/chat/completions"


@dataclass(frozen=True)
class Voice:
    """One voice: who speaks and how. ``person`` decides question re-voicing (``first``/``third`` re-voice; others keep "you")."""

    key: str
    language: str
    person: str  # first | second | third | none
    family: str  # prose | fields | data | dialogue
    description: str
    bounds: tuple[float, float]  # rewrite length / original length
    json_state: bool = False


_EN = (
    Voice("robot_monologue", "en", "first", "prose",
          "The robot itself narrates the moment in the first person (I, me, my), present tense, in plain spoken sentences, as if "
          "thinking aloud while it works.", (0.6, 2.0)),
    Voice("radio_call", "en", "third", "prose",
          "An operator radios a status report about the robot to a control room: clipped spoken phrases, few articles; radio words "
          "such as 'copy' or 'over' are fine. Refer to the robot as 'the robot' or by its kind, never by a number or call sign.",
          (0.45, 1.8)),
    Voice("maintenance_log", "en", "third", "fields",
          "A maintenance and safety log entry: labelled fields (for example 'Unit:', 'Location:', 'Load:', 'Observed:', "
          "'Remarks:'), terse technical register, present tense, the robot called 'the unit' or 'the robot'. No time stamps, no "
          "entry numbers.", (0.45, 1.8)),
    Voice("supervisor_note", "en", "second", "prose",
          "A floor supervisor's quick note left for the robot, addressing it as 'you': informal, short sentences, everyday words.",
          (0.45, 1.8)),
    Voice("sensor_json", "en", "none", "data",
          "A JSON dump of the robot's state and perception: one nested object with snake_case keys (for example 'self', 'motion', "
          "'braking', 'load', 'perceived', 'notes', 'planner', 'actions'); every measurement is a string that keeps the number "
          "immediately followed by its unit exactly as written (\"0.65 m/s\"); descriptions and remarks are short strings.",
          (0.6, 2.7), True),
    Voice("field_table", "en", "none", "fields",
          "A plain-text table of 'field | value' rows grouped under a few headings; fragments, not sentences, except for remarks. "
          "Put each unit right after its number in the value column.", (0.45, 1.9)),
    Voice("terse_bullets", "en", "second", "fields",
          "Terse telegraphic bullet points: as few words as possible, fragments not sentences, one fact per bullet; where a person "
          "is needed the robot is 'you'.", (0.35, 1.4)),
    Voice("chatty_colleague", "en", "second", "prose",
          "A long, chatty, informal paragraph from a co-worker talking to the robot ('you'): long winding sentences, filler words "
          "and asides — but no new facts and no opinion about what will happen or what to do.", (0.8, 2.5)),
    Voice("formal_report", "en", "third", "prose",
          "A formal technical description in the third person ('the robot …'): long complex sentences, passive voice where "
          "natural, precise register.", (0.7, 2.3)),
    Voice("control_dialogue", "en", "second", "dialogue",
          "A short radio exchange between a control-room operator (lines start 'Control:') and the robot (lines start 'Robot:'); "
          "the robot answers in the first person and Control addresses it as 'you'. Every fact is said by one of them.",
          (0.7, 2.4)),
)

_KO = (
    Voice("ko_monologue", "ko", "first", "prose",
          "로봇이 스스로 1인칭(나, 내)으로 지금 상황을 혼잣말하듯 서술한다. 반말 평서체(-다, -는 중이다), 현재 시제. "
          "(The robot narrates in the first person, plain informal declaratives.)", (0.6, 2.0)),
    Voice("ko_radio", "ko", "third", "prose",
          "관제실로 보내는 무전 보고. 짧게 끊는 구어체, '이상', '확인 바람' 같은 무전 말투 허용. 로봇은 '로봇' 또는 그 종류 이름으로 "
          "부르고 번호나 호출 부호를 붙이지 않는다. (An operator's clipped radio report about the robot.)", (0.45, 1.8)),
    Voice("ko_work_log", "ko", "third", "fields",
          "작업·점검 일지 항목. '기체:', '위치:', '적재물:', '관측:', '비고:' 같은 항목 이름과 개조식 명사형 종결(-함, -임, -중). "
          "시각이나 일지 번호는 쓰지 않는다. (A maintenance log entry in Korean bullet-report style.)", (0.45, 1.8)),
    Voice("ko_handover_note", "ko", "second", "prose",
          "교대하는 동료가 로봇에게 남긴 인수인계 메모. 해요체, 짧은 문장, 일상어; 로봇은 '당신'으로 부르거나 주어를 생략한다. "
          "(A colleague's handover note to the robot, polite informal.)", (0.45, 1.8)),
    Voice("ko_json", "ko", "none", "data",
          "로봇 상태와 인식을 담은 JSON 덤프: 영어 snake_case 열쇠(예: 'self', 'motion', 'braking', 'load', 'perceived', "
          "'notes', 'planner', 'actions')와 한국어 문자열 값. 측정값은 숫자 바로 뒤에 원래 단위를 그대로 붙인 문자열(\"0.65 m/s\"). "
          "(A JSON dump with English keys and Korean values.)", (0.6, 2.7), True),
    Voice("ko_table", "ko", "none", "fields",
          "'항목 | 값' 줄로 된 표를 몇 개의 제목 아래로 묶는다. 비고 말고는 문장이 아닌 짧은 구. 단위는 값 칸의 숫자 바로 뒤에. "
          "(A plain-text field | value table.)", (0.45, 1.9)),
    Voice("ko_terse", "ko", "second", "fields",
          "아주 짧은 메모 조각: 명사구 위주, 한 줄에 사실 하나, 문장을 완성하지 않는다. (Terse note fragments.)", (0.35, 1.4)),
    Voice("ko_chatty", "ko", "second", "prose",
          "동료가 로봇에게 수다스럽게 말을 거는 긴 문단: 해요체 구어, 길게 이어지는 문장, 군말과 곁가지 — 그러나 새로운 사실도, "
          "무엇이 일어날지나 무엇을 할지에 대한 의견도 없다. (A long chatty paragraph.)", (0.8, 2.5)),
    Voice("ko_formal_report", "ko", "third", "prose",
          "격식 있는 기술 보고서체(합니다체), 3인칭('로봇은 …'), 길고 복잡한 문장. (A formal third-person technical report.)",
          (0.7, 2.3)),
    Voice("ko_dialogue", "ko", "second", "dialogue",
          "관제('관제:')와 로봇('로봇:')의 짧은 무전 교신. 로봇은 1인칭으로 답하고 관제는 로봇을 '당신'으로 부른다. 모든 사실을 둘 중 "
          "하나가 말한다. (A control-room / robot radio exchange.)", (0.7, 2.4)),
)

#: Language → voice key → :class:`Voice`.
VOICES: dict[str, dict[str, Voice]] = {"en": {v.key: v for v in _EN}, "ko": {v.key: v for v in _KO}}

#: Per-call fact-order hints (seeded per moment and voice): the same voice opens in different places.
ORDER_HINTS = (
    "Open with the people, vehicles and objects around the robot, then the robot's own motion and limits.",
    "Open with the robot's own motion and braking, then what it perceives, then the rest.",
    "Open with the place and the task, then give the remaining facts in an order of your choosing.",
    "Open with the notes, side remarks or the planner's advice, then everything else.",
    "Use an order of your own choosing that differs clearly from the original order.",
)

_PERSON_RULES = {
    ("en", "first"): "change only 'you' to 'I' or 'me', 'your' to 'my', 'yours' to 'mine', 'yourself' to 'myself', and fix the verb "
                     "agreement (do you → do I, are you → am I, you were → I was)",
    ("en", "third"): "change only 'you' to 'the robot' (or 'it' once the robot is named in the same sentence), 'your' to 'the robot's' "
                     "or 'its', 'yourself' to 'itself', and fix the verb agreement (do you stop → does the robot stop)",
    ("ko", "first"): "'당신'이 든 어절만 1인칭으로 바꾼다(당신이 → 내가, 당신은 → 나는, 당신의 → 나의, 당신을 → 나를, 당신에게 → 나에게). "
                     "다른 어절은 어미까지 한 글자도 바꾸지 않는다(-합니까?, -하십시오 그대로); '당신'이 없는 글은 그대로 돌려준다",
    ("ko", "third"): "'당신'이 든 어절만 '로봇'으로 바꾼다(당신이 → 로봇이, 당신은 → 로봇은, 당신의 → 로봇의, 당신을 → 로봇을, 당신에게 → 로봇에게). "
                     "다른 어절은 어미까지 한 글자도 바꾸지 않는다(-합니까?, -하십시오 그대로); '당신'이 없는 글은 그대로 돌려준다",
}

_SYSTEM = """You rewrite short robot-perception texts for a training corpus. The same moment must be told in a different voice and \
format, while every fact stays exactly the same. The reader must be able to recover every fact of the original from your text.

Hard rules:
1. Keep every number exactly as written, followed by its unit written the same way ("0.65 m/s" stays "0.65 m/s", "850 ms" stays \
"850 ms", "±1.06 N" keeps its ±). Never convert units, round, add up, compare or compute. Write numbers as digits. If a number appears \
several times in the original (for example in the planner's advice and again in the list of actions), it must appear as many times.
2. Add no number that is not in the original: no times of day, dates, durations, ids, unit or call-sign numbers, counts or percentages.
3. Keep every name of a person, vehicle, robot, object, container, load, surface and place with its full description word for word \
(articles and capital letters may change). The exact phrases to keep are listed under KEEP. Keep the setting (where, and the time of \
day if given), the robot's identity and its task.
4. Keep every relation exactly: which person or object each fact belongs to (never merge or split them), what is seen now versus \
last seen or hidden, what lies in which direction, above/below, short of/past, and every required stop or limit.
5. Keep every observation: directions (left/right, ahead, toward/away), what people or vehicles are doing or showing, how full \
something is (keep the fill wording), the lid exactly as stated (no lid / lid on / cannot see whether it has a lid), which sides are \
clear, blocked or hidden, how old each sensor reading is, tolerances, the notes and side remarks, the planner's recommendation (if \
any), and the list of possible actions (if any).
6. Add nothing: no new facts, no causes, no predictions, no advice, no safety judgements (never say what will happen, what is \
risky, likely or safe, or what should be done), and do not answer or hint at any question. If the original has no planner \
recommendation, do not mention a planner or any advice at all.
7. Rebuild every sentence in the voice's own words and structure; do not copy the original's sentences or its layout. Only the \
KEEP phrases and the numbers stay word for word.
8. No time-stamped log lines such as "[-0.85 s] ...", and no numbering of rows, options or list items.
9. Write in {language_name}. Output one JSON object and nothing else."""

_USER = """VOICE: {description}
PERSON: {person_rule}
ORDER: {order}
LENGTH: about {length} the original.

KEEP (word for word, at least once each): {keep}
NUMBERS (each exactly as written, as many times as listed): {numbers}

ORIGINAL TEXT:
<<<
{state}
>>>
{questions_block}
Return JSON: {schema}"""

_QUESTIONS_BLOCK = """
QUESTIONS to re-voice for the same person: {rule}. Change nothing else — keep every other word, every number, every action, every \
time window, every name and every option key exactly. If a text has no second-person word, return it unchanged.
<<<
{questions}
>>>
"""

_PERSON_TEXT = {
    "first": "the robot speaks about itself in the first person (I, me, my).",
    "second": "the robot is addressed as 'you' where a person is needed.",
    "third": "the robot is spoken about in the third person ('the robot', 'it', 'its'), never 'you' and never by a number.",
    "none": "a data format; no narrator. Where a sentence is needed, keep the robot as 'you' or leave the subject out.",
}
_PERSON_TEXT_KO = {
    "first": "로봇이 1인칭(나, 내)으로 자기 이야기를 한다.",
    "second": "로봇을 '당신'으로 부르거나 주어를 생략한다.",
    "third": "로봇을 3인칭('로봇', '로봇의')으로 말한다. '당신'은 쓰지 않고 번호도 붙이지 않는다.",
    "none": "데이터 형식이며 화자가 없다. 문장이 필요하면 로봇을 '당신'으로 두거나 주어를 생략한다.",
}
_LENGTH = {(0.35, 1.4): "half as long as", (0.45, 1.8): "as long as or shorter than", (0.45, 1.9): "as long as",
           (0.6, 2.0): "as long as or a little longer than", (0.6, 2.7): "one to two times as long as", (0.7, 2.3): "one and a half times as long as",
           (0.7, 2.4): "one and a half times as long as", (0.8, 2.5): "one and a half to two times as long as"}

PROMPT_VERSION = hashlib.sha256(
    json.dumps([_SYSTEM, _USER, _QUESTIONS_BLOCK, sorted(map(list, _PERSON_RULES.items())), _PERSON_TEXT, _PERSON_TEXT_KO, ORDER_HINTS,
                sorted([list(k), v] for k, v in _LENGTH.items()),
                [[v.key, v.person, v.description, v.bounds] for lang in VOICES.values() for v in lang.values()]],
               ensure_ascii=False, sort_keys=True).encode("utf-8")
).hexdigest()[:12]


def revoices_questions(voice: Voice) -> bool:
    return voice.person in ("first", "third")


def keep_list(state: str, language: str) -> tuple[list[str], list[str]]:
    """The phrases and numbers the rewrite must keep — computed from the observation text only (catalog names + fill wording;
    every number mention with its count)."""
    names = checks.found_entities(state, language)
    for band in sorted(checks.fill_bands(state, language)):
        names.append(_FILL_WORDING[language][band])
    # a phrase inside a longer kept phrase is implied; list the longest ones
    if language == "ko":
        names = [n for n in names if not any(n != m and n in m for m in names)]
    else:
        words = {n: checks.normalize_words(n) for n in names}
        names = [n for n in names if not any(n != m and _sub(words[n], words[m]) for m in names)]
    surfaces = [surface for _, surface in checks.number_surfaces(state)]
    merged: list[str] = []
    for surface in surfaces:  # a Korean fraction "4분의 3" is one item
        if merged and merged[-1].endswith("분의"):
            merged[-1] = f"{merged[-1]} {surface}"
        else:
            merged.append(surface)
    counts = Counter(merged)
    numbers = [f'"{surface}"' + (f" ×{counts[surface]}" if counts[surface] > 1 else "") for surface in dict.fromkeys(merged)]
    return names, numbers


def _sub(small: list[str], big: list[str]) -> bool:
    n = len(small)
    return any(big[i : i + n] == small for i in range(len(big) - n + 1)) if n else True


_FILL_WORDING = {
    "en": {"lt_third": "less than a third full", "third": "about a third full", "half": "about half full", "two_thirds": "about two-thirds full",
           "three_quarters": "about three-quarters full", "almost_full": "almost full", "near_brim": "nearly to the brim", "brim": "full to the brim"},
    "ko": {"lt_third": "3분의 1 미만", "third": "약 3분의 1", "half": "약 절반", "two_thirds": "약 3분의 2", "three_quarters": "약 4분의 3",
           "almost_full": "거의 가득", "near_brim": "가장자리 가까이", "brim": "가장자리까지 가득"},
}


def _question_payload(questions: dict[str, Any]) -> dict[str, Any]:
    out = {}
    for name, q in questions.items():
        entry: dict[str, Any] = {"instructions": q["instructions"]}
        if q["type"] in ("choice", "score") and q.get("criteria") is not None:
            entry["criteria"] = q["criteria"]
        out[name] = entry
    return out


def build_messages(record: dict[str, Any], voice: Voice, order_hint: str) -> list[dict[str, str]]:
    """The chat messages for one rewrite. Only ``request.state`` and the question texts of the record are used."""
    language = voice.language
    state = record["request"]["state"]
    names, numbers = keep_list(state, language)
    person_text = (_PERSON_TEXT_KO if language == "ko" else _PERSON_TEXT)[voice.person]
    state_shape = "{ ...the JSON dump as an object... }" if voice.json_state else '"...the rewritten text as one string..."'
    if revoices_questions(voice):
        questions_block = _QUESTIONS_BLOCK.format(
            rule=_PERSON_RULES[(language, voice.person)],
            questions=json.dumps(_question_payload(record["request"]["questions"]), ensure_ascii=False, indent=1),
        )
        schema = ('{"state": ' + state_shape + ', "questions": {"<each question name above>": {"instructions": "...", '
                  '"criteria": <the same option keys, or the same list, re-voiced; omit for questions without criteria>}}} — '
                  '"questions" sits at the top level next to "state", never inside it, with exactly the question names given.')
    else:
        questions_block = ""
        schema = '{"state": ' + state_shape + '} — nothing else.'
    user = _USER.format(
        description=voice.description, person_rule=person_text, order=order_hint, length=_LENGTH.get(voice.bounds, "as long as"),
        keep="; ".join(f'"{n}"' for n in names) or "(none)", numbers=", ".join(numbers) or "(none)", state=state,
        questions_block=questions_block, schema=schema,
    )
    system = _SYSTEM.format(language_name="Korean (한국어)" if language == "ko" else "English")
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def chat_payload(model: str, messages: list[dict[str, str]], *, max_tokens: int, temperature: float = 0.9, seed: int | None = None,
                 data_collection: str = "deny") -> dict[str, Any]:
    """OpenRouter chat-completions body (JSON object output, reasoning off, usage with cost)."""
    payload: dict[str, Any] = {
        "model": model, "messages": messages, "max_tokens": int(max_tokens), "temperature": float(temperature),
        "response_format": {"type": "json_object"}, "reasoning": {"enabled": False}, "usage": {"include": True},
        "provider": {"data_collection": data_collection},
    }
    if seed is not None:
        payload["seed"] = int(seed)
    return payload


def payload_sha256(payload: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------
# reply → state / questions
# --------------------------------------------------------------------------


def render_state(value: Any) -> str:
    """A JSON-voice state object → its text (2-space indented JSON, non-ASCII kept)."""
    return json.dumps(value, ensure_ascii=False, indent=2)


def _json_object(text: str, *, last: bool = False) -> Any:
    """The JSON object of a reply: the whole text, a fenced block, or (``last``) the last complete object in the text."""
    text = text.strip()
    fence = re.match(r"^```(?:json)?\s*(.*?)\s*```$", text, re.DOTALL)
    if fence:
        text = fence.group(1)
    try:
        return json.loads(text)
    except ValueError:
        decoder = json.JSONDecoder()
        found = []
        i = text.find("{")
        while i >= 0:
            try:
                obj, end = decoder.raw_decode(text, i)
            except ValueError:
                i = text.find("{", i + 1)
                continue
            found.append(obj)
            if not last:
                return obj
            i = text.find("{", end)  # top-level objects only: skip the inside of the one just read
        if found:
            return found[-1]
        raise ValueError("no JSON object in the reply")


def parse_reply(content: str, voice: Voice) -> tuple[str | None, dict[str, Any] | None, list[str]]:
    """The model's reply → (state text, re-voiced questions or None, problems)."""
    try:
        data = _json_object(content or "")
    except ValueError as error:
        return None, None, [f"reply is not JSON ({str(error)[:80]})"]
    if not isinstance(data, dict) or "state" not in data:
        return None, None, ["reply has no 'state'"]
    raw = data["state"]
    if voice.json_state:
        if isinstance(raw, str):  # a JSON object written as a string is fine if it parses
            try:
                raw = json.loads(raw)
            except ValueError:
                return None, None, ["JSON voice: state is not a JSON object"]
        if not isinstance(raw, (dict, list)):
            return None, None, ["JSON voice: state is not a JSON object"]
        state = render_state(raw)
    elif isinstance(raw, str):
        state = raw.strip()
    else:
        return None, None, ["state is not a string"]
    questions = data.get("questions") if revoices_questions(voice) else None
    if revoices_questions(voice) and not isinstance(questions, dict):
        return state, None, ["no re-voiced questions"]
    return state, questions, []


# --------------------------------------------------------------------------
# sealed-set guard
# --------------------------------------------------------------------------


class SealedTextError(RuntimeError):
    """A request would carry text of a sealed set — it was not sent."""


def _norm_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text)).strip()


def state_sha256(text: str) -> str:
    return hashlib.sha256(_norm_text(text).encode("utf-8")).hexdigest()


def _request_hash(record: dict[str, Any]) -> str:
    from robo_jev.simgen.holdout import request_hash

    return request_hash(record)


def _short(h: str) -> str:
    return h[:16]


@dataclass
class SealedIndex:
    """Hashes of the sealed sets. ``states``/``requests``: name → whole-state / whole-request sha256 of every record; ``units``:
    name → text-unit hashes (16-hex prefixes) of the sets written outside our engine. ``files``: name → (path, sha256)."""

    states: dict[str, set[str]]
    requests: dict[str, set[str]]
    units: dict[str, set[str]]
    files: dict[str, dict[str, str]]

    #: Sets written outside our engine: shared text units with them are refused. T1–T7 share the engine's templates with the
    #: training data, so they are matched on whole texts only.
    OUTSIDE = ("G1", "G2", "T8a", "T8b", "T9")

    @classmethod
    def build(cls, paths: dict[str, Path]) -> "SealedIndex":
        from robo_jev.data.general_sets import record_units

        states: dict[str, set[str]] = {}
        requests: dict[str, set[str]] = {}
        units: dict[str, set[str]] = {}
        files: dict[str, dict[str, str]] = {}
        for name, path in paths.items():
            path = Path(path)
            digest = hashlib.sha256()
            s, r, u = set(), set(), set()
            with path.open("rb") as handle:
                for raw in handle:
                    digest.update(raw)
                    if not raw.strip():
                        continue
                    record = json.loads(raw)
                    request = record.get("request") or {}
                    s.add(state_sha256(request.get("state", "")))
                    r.add(_request_hash(record))
                    if name in cls.OUTSIDE:
                        u |= {_short(hashlib.sha256(unit.encode("utf-8")).hexdigest()) for unit in record_units(record)}
            states[name], requests[name], units[name] = s, r, u
            files[name] = {"path": str(path), "sha256": digest.hexdigest()}
        return cls(states, requests, units, files)

    def to_json(self) -> dict[str, Any]:
        return {"files": self.files, "states": {k: sorted(v) for k, v in self.states.items()},
                "requests": {k: sorted(v) for k, v in self.requests.items()}, "units": {k: sorted(v) for k, v in self.units.items()}}

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> "SealedIndex":
        return cls({k: set(v) for k, v in data["states"].items()}, {k: set(v) for k, v in data["requests"].items()},
                   {k: set(v) for k, v in data["units"].items()}, dict(data["files"]))

    def text_units(self, text: str) -> set[str]:
        from robo_jev.data.general_sets import text_units

        return {_short(hashlib.sha256(unit.encode("utf-8")).hexdigest()) for unit in text_units(text)}

    def hits(self, *, source: dict[str, Any] | None = None, texts: Iterable[str] = ()) -> list[str]:
        """Sealed sets touched by a source record (whole request / whole state) or by any text (whole text; units of the outside
        sets). Empty = safe to send."""
        out = []
        if source is not None:
            rh = _request_hash(source)
            sh = state_sha256(source["request"]["state"])
            out += [f"{name}: source request" for name, hashes in self.requests.items() if rh in hashes]
            out += [f"{name}: source state" for name, hashes in self.states.items() if sh in hashes]
        for text in texts:
            th = state_sha256(text)
            out += [f"{name}: text" for name, hashes in self.states.items() if th in hashes]
            units = self.text_units(text)
            out += [f"{name}: {len(units & hashes)} text units" for name, hashes in self.units.items() if units & hashes]
        return sorted(set(out))

    def guard(self, *, source: dict[str, Any], texts: Iterable[str]) -> None:
        hit = self.hits(source=source, texts=texts)
        if hit:
            raise SealedTextError(f"{source.get('id')}: refusing to send sealed text ({'; '.join(hit[:4])})")


def sent_text_hashes(index: SealedIndex, texts: Iterable[str]) -> dict[str, Any]:
    """What the request log keeps about the texts of one request: whole-text hashes and outside-set unit hashes (16-hex)."""
    texts = list(texts)
    return {"text_sha256": [state_sha256(t) for t in texts], "units": sorted(set().union(*(index.text_units(t) for t in texts))) if texts else []}


def audit_request_log(log_path: Path, index: SealedIndex, *, source_ids: set[str] | None = None) -> dict[str, Any]:
    """Re-check a request log: no logged whole-text hash is a sealed state, no logged unit is in an outside set, every source id is
    a training id. → ``{"requests", "violations": [...]}``."""
    violations = []
    n = 0
    all_states = set().union(*index.states.values()) if index.states else set()
    all_requests = set().union(*index.requests.values()) if index.requests else set()
    outside = set().union(*index.units.values()) if index.units else set()
    path = Path(log_path)
    if path.is_file():
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                entry = json.loads(line)
                if "sent" not in entry:
                    continue
                n += 1
                sent = entry["sent"]
                where = f"{entry.get('source_id')}/{entry.get('voice')}"
                if sent.get("source_request_sha256") in all_requests:
                    violations.append(f"{where}: source request is sealed")
                for h in sent.get("text_sha256", []):
                    if h in all_states:
                        violations.append(f"{where}: a sent text equals a sealed state")
                if set(sent.get("units", [])) & outside:
                    violations.append(f"{where}: sent text shares units with an outside sealed set")
                if source_ids is not None and entry.get("source_id") not in source_ids:
                    violations.append(f"{where}: source id is not a training id")
    return {"requests": n, "violations": violations}


# --------------------------------------------------------------------------
# plan (stratified order, voices, order hints)
# --------------------------------------------------------------------------


def _rng(*parts: Any) -> random.Random:
    return random.Random(hashlib.sha256("|".join(map(str, parts)).encode("utf-8")).hexdigest())


def build_plan(rows: list[dict[str, Any]], *, seed: int = 0) -> list[dict[str, Any]]:
    """Source rows ``{id, scene, group, language}`` → plan entries in stratified order: ``{rank, source_id, scene, group, language,
    voices: [v1, v2], order: {voice: hint index}}``. Strata = (scene, language); every prefix is a proportional stratified sample."""
    strata: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        strata[(row["scene"], row["language"])].append(row)
    keyed = []
    for key in sorted(strata):
        members = sorted(strata[key], key=lambda r: r["id"])
        _rng(NARRATION_VERSION, seed, "stratum", *key).shuffle(members)
        n = len(members)
        voices = list(VOICES[key[1]])
        cycle: list[str] = []
        for j, row in enumerate(members):
            if len(cycle) < 2:  # refill with a fresh permutation (each voice once per cycle)
                fresh = list(voices)
                _rng(NARRATION_VERSION, seed, "voices", *key, j).shuffle(fresh)
                if cycle and fresh[0] == cycle[0]:
                    fresh.append(fresh.pop(0))
                cycle += fresh
            pair = [cycle.pop(0), cycle.pop(0)]
            jitter = _rng(NARRATION_VERSION, seed, "jitter", row["id"]).random()
            order = {v: _rng(NARRATION_VERSION, seed, "order", row["id"], v).randrange(len(ORDER_HINTS)) for v in pair}
            keyed.append(((j + jitter) / n, row["id"], {"source_id": row["id"], "scene": row["scene"], "group": row["group"],
                                                         "language": row["language"], "voices": pair, "order": order}))
    keyed.sort(key=lambda item: (item[0], item[1]))
    return [{"rank": i, **entry} for i, (_, _, entry) in enumerate(keyed)]


# --------------------------------------------------------------------------
# narrated record
# --------------------------------------------------------------------------


def narrated_id(source_id: str, voice: str) -> str:
    return f"{source_id}~{voice}"


def assemble_record(source: dict[str, Any], *, voice: Voice, state: str, questions: dict[str, Any] | None,
                    question_status: dict[str, str], narration: dict[str, Any]) -> dict[str, Any]:
    """The narrated record: the source with a new ``request`` and ``meta.narration``; ``labels``, ``truth``, ``rationale``,
    ``generator`` and every other ``meta`` field are the source's, unchanged."""
    request_questions = {}
    for name, q in source["request"]["questions"].items():
        if question_status.get(name) == "revoiced":
            new = dict(q)
            new["instructions"] = questions[name]["instructions"]
            if q["type"] in ("choice", "score"):
                new["criteria"] = questions[name]["criteria"]
            request_questions[name] = new
        else:
            request_questions[name] = q
    meta = dict(source["meta"])
    meta["narration"] = narration
    out = {
        "schema": source["schema"], "id": narrated_id(source["id"], voice.key), "generator": source["generator"], "meta": meta,
        "request": {"state": state, "questions": request_questions}, "labels": source["labels"], "truth": source["truth"],
        "rationale": source["rationale"],
    }
    return out
