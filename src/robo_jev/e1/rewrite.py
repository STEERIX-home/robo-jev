"""Task E1 condition B — the A moments rewritten by an LLM into first-person narration by the robot.

One OpenRouter chat call per A record (:func:`build_messages`): the state becomes a first-person narration in plain prose, and each
question is re-voiced to match ("you" → "I"); every number and unit, every object and every hypothetical action is to be kept. The
prompt has no examples (nothing from T8 or any test set). Score scales carry no "you" in A and are copied unchanged.

**Automatic checks** (:func:`check_b`; a record failing any is dropped, and the failure rate is reported by reason):

* ``numbers`` — the multiset of number-with-unit tokens (number exactly as printed, with the unit that follows it or none) is
  identical, in the state and in every question separately;
* ``objects`` — every object named in an A field (the load, its deck surface, the stacked item and its spot, the container, the stop
  target, the ramp, its end and the detour — from the moment's observation) is named in the same B field;
* ``keys`` — the same question names (types and score scales are copied from A);
* ``voice`` — the rewrite is first person: no "you"/"your" left, and a first-person word in the state and in every question;
* ``parse`` — the reply is one JSON object of the requested shape.

**Sealed sets.** :class:`SealedIndex` holds the whole-state sha256 of every sealed record (T1–T7, G1, G2, T8a, T8b) and the sha256 of
every sentence of the sets written outside our engine (G1, G2, T8a, T8b); :func:`assert_sendable` refuses a request whose state or
any sentence matches, before anything is sent.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

__all__ = ["B_PREFIX", "PROMPT_VERSION", "SYSTEM_PROMPT", "SealedIndex", "SealedTextError", "assert_sendable", "b_record", "build_messages",
           "check_b", "number_tokens", "object_phrases", "parse_reply"]

B_PREFIX = "e1b"
PROMPT_VERSION = "e1-b-v1"

SYSTEM_PROMPT = """You rewrite descriptions of one moment in a robot's work. The input is a JSON object with a "state" (a description addressed to the robot as "you") and "questions" addressed to the robot.

Rewrite the state as a first-person narration spoken by the robot itself: the robot says "I", "me" and "my". Write plain narrative prose: sentences in one or more paragraphs, with no bullet points, no headings, no labels such as "CONTEXT:" and no lists.

Rules (they are checked automatically):
1. Keep every fact, including incidental remarks, any planner recommendation and every action the robot could take.
2. Keep every number exactly as written, together with its unit exactly as written: the same digits and decimals (1.80 stays 1.80) and the same unit symbol (m/s², km/h, cm, mm, ms, s, °, %, …). Do not convert units, round, spell numbers out as words, write number words as digits, or add any number.
3. Keep the name of every object, place and thing exactly as written; only "a", "the" and "your" (which becomes "my") may change.
4. Do not add facts, interpretations, predictions, warnings, advice or opinions.
5. Never address the robot as "you": the robot is "I".

Then rewrite each question in the same first-person voice of the robot: "you" becomes "I", "your" becomes "my". Keep its meaning, its numbers and units, its objects and its hypothetical action exactly; only the person changes. Do not answer the questions.

Reply with one JSON object only, no other text: {"state": "<the narration>", "questions": {"<question name>": "<the rewritten question>", ...}} with exactly the same question names as the input."""


class SealedTextError(RuntimeError):
    """A request would carry sealed text — nothing is sent."""


# --------------------------------------------------------------------------
# Sealed sets
# --------------------------------------------------------------------------


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.?!])\s+|\n+", text)
    return [" ".join(p.split()).lower() for p in parts if len(p.split()) >= 6]


class SealedIndex:
    """Hashes of sealed text: whole states of every sealed set, sentences (≥ 6 words) of the sets written outside our engine."""

    def __init__(self, states: set[str], sentences: set[str], files: dict[str, str]):
        self.states, self.sentences, self.files = states, sentences, files

    @classmethod
    def build(cls, files: dict[str, Path], outside: Iterable[str]) -> "SealedIndex":
        outside = set(outside)
        states, sentences, shas = set(), set(), {}
        for name, path in files.items():
            data = Path(path).read_bytes()
            shas[name] = hashlib.sha256(data).hexdigest()
            for line in data.decode("utf-8").splitlines():
                if not line.strip():
                    continue
                record = json.loads(line)
                state = (record.get("request") or {}).get("state")
                if not isinstance(state, str):
                    continue
                states.add(_sha(state))
                if name in outside:
                    sentences.update(_sha(s) for s in _sentences(state))
                    for q in ((record.get("request") or {}).get("questions") or {}).values():
                        sentences.update(_sha(s) for s in _sentences(str(q.get("instructions") or "")))
        return cls(states, sentences, shas)


def assert_sendable(record: dict[str, Any], texts: Iterable[str], index: SealedIndex) -> None:
    """Refuse (raise :class:`SealedTextError`) if the record is T8-derived or any outgoing text matches sealed text."""
    rid = str(record.get("id"))
    if rid.startswith(("t8", "e1c", "e1d")) or (record.get("meta") or {}).get("set") in ("T8a", "T8b"):
        raise SealedTextError(f"{rid}: T8-derived record — never sent")
    if rid.startswith("t9") or (record.get("meta") or {}).get("set") == "T9":
        raise SealedTextError(f"{rid}: T9 record — sealed, never sent to any API")
    state = record["request"]["state"]
    if _sha(state) in index.states:
        raise SealedTextError(f"{rid}: the state is a sealed state")
    for text in texts:
        if _sha(text) in index.states:
            raise SealedTextError(f"{rid}: outgoing text equals a sealed state")
        for sentence in _sentences(text):
            if _sha(sentence) in index.sentences:
                raise SealedTextError(f"{rid}: outgoing sentence matches a sentence of a sealed outside set")


# --------------------------------------------------------------------------
# Prompt and reply
# --------------------------------------------------------------------------


def build_messages(record: dict[str, Any]) -> list[dict[str, str]]:
    """The chat messages for one A record (state + question texts only — no id, meta or labels)."""
    request = record["request"]
    payload = {"state": request["state"], "questions": {name: q["instructions"] for name, q in request["questions"].items()}}
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": json.dumps(payload, ensure_ascii=False, indent=1)}]


def parse_reply(text: str) -> dict[str, Any]:
    """Reply text → {"state": str, "questions": {name: str}} (a fenced or surrounded JSON object is accepted); ``ValueError`` else."""
    body = text.strip()
    fence = re.search(r"```(?:json)?\s*(\{.*\})\s*```", body, re.S)
    if fence:
        body = fence.group(1)
    elif not body.startswith("{"):
        start, end = body.find("{"), body.rfind("}")
        if start < 0 or end <= start:
            raise ValueError("no JSON object in the reply")
        body = body[start:end + 1]
    data = json.loads(body)
    if not isinstance(data, dict) or not isinstance(data.get("state"), str) or not isinstance(data.get("questions"), dict):
        raise ValueError("reply is not {state: str, questions: {...}}")
    if not data["state"].strip() or not all(isinstance(v, str) and v.strip() for v in data["questions"].values()):
        raise ValueError("empty state or question text")
    return {"state": data["state"].strip(), "questions": {str(k): v.strip() for k, v in data["questions"].items()}}


# --------------------------------------------------------------------------
# Checks
# --------------------------------------------------------------------------

#: Units that may follow a number (longest first); the value is the unit's normal form.
_UNIT_FORMS = (("m/s²", "m/s²"), ("m/s^2", "m/s²"), ("m/s2", "m/s²"), ("m/s", "m/s"), ("km/h", "km/h"), ("°C", "°C"), ("° C", "°C"), ("°", "°"),
               ("%", "%"), ("cm", "cm"), ("mm", "mm"), ("ml", "ml"), ("ms", "ms"), ("-minutes", "minutes"), ("-minute", "minutes"),
               ("minutes", "minutes"), ("minute", "minutes"), ("min", "minutes"), ("s", "s"), ("m", "m"), ("kg", "kg"), ("N", "N"))
_UNIT_RE = "|".join(re.escape(u) for u, _ in _UNIT_FORMS)
_NUMBER = re.compile(r"(?<![\w.])(\d+(?:\.\d+)?)(?:\s?(" + _UNIT_RE + r")(?![A-Za-z²/]))?")
_UNIT_NORMAL = dict(_UNIT_FORMS)


def number_tokens(text: str) -> Counter:
    """Multiset of (number as printed, unit or '') tokens."""
    return Counter((number, _UNIT_NORMAL.get(unit or "", "")) for number, unit in _NUMBER.findall(text))


def _norm(text: str) -> str:
    """Lower case, punctuation to spaces, 'your' read as 'my' (B is first person)."""
    text = re.sub(r"\byour\b", "my", text.lower())
    return " ".join(re.sub(r"[^\w°%²/.\- ]+", " ", text).split())


def _core(phrase: str) -> str:
    """Object phrase without a leading article or 'your' (B writes 'my')."""
    words = _norm(phrase).split()
    while words and words[0] in ("a", "an", "the", "your", "my"):
        words = words[1:]
    return " ".join(words)


def object_phrases(cfg, record: dict[str, Any]) -> list[str]:
    """Names of the physical objects of the moment (rebuilt from the record's generator fields), longest first, without articles."""
    from robo_jev.simgen.generate import record_moment

    obs = record_moment(cfg, record).obs
    found = []
    if obs.load is not None:
        found += [obs.load.phrase, obs.load.ref, obs.load.deck_phrase]
    if obs.stack is not None:
        found += [obs.stack.phrase, obs.stack.ref, obs.stack.spot]
    if obs.container is not None:
        found += [obs.container.phrase] + ([obs.container.ref] if obs.container.ref else [])
    env = obs.env
    found += [x for x in (env.obstacle, env.ramp, env.ramp_end, env.detour) if x]
    cores = {_core(p) for p in found if p and _core(p)}
    return sorted(cores, key=lambda s: (-len(s), s))


_SECOND = re.compile(r"\b(you|your|yours|yourself|yourselves|you're|you've|you'll|you'd)\b", re.I)
_FIRST = re.compile(r"\bI\b|\b(?i:me|my|mine|myself)\b")


def _objects_in(text: str, objects: list[str]) -> set[str]:
    norm = _norm(text)
    return {o for o in objects if o in norm}


def check_b(a: dict[str, Any], reply: dict[str, Any], objects: list[str]) -> dict[str, list[str]]:
    """Problems by check (empty dict = passes)."""
    problems: dict[str, list[str]] = {}

    def add(kind: str, message: str) -> None:
        problems.setdefault(kind, []).append(message)

    qa = a["request"]["questions"]
    if set(reply["questions"]) != set(qa):
        add("keys", f"question names {sorted(reply['questions'])} != {sorted(qa)}")
        return problems
    fields = [("state", a["request"]["state"], reply["state"])] + [(name, qa[name]["instructions"], reply["questions"][name]) for name in qa]
    for field, before, after in fields:
        nb, na = number_tokens(before), number_tokens(after)
        if nb != na:
            add("numbers", f"{field}: missing {sorted((nb - na).elements())} added {sorted((na - nb).elements())}")
        lost = _objects_in(before, objects) - _objects_in(after, objects)
        if lost:
            add("objects", f"{field}: missing {sorted(lost)}")
        if _SECOND.search(after):
            add("voice", f"{field}: second person left ({_SECOND.search(after).group(0)!r})")
        if not _FIRST.search(after):
            add("voice", f"{field}: no first-person word")
    return problems


def b_record(a: dict[str, Any], reply: dict[str, Any], *, model: str, objects: list[str], request_sha256: str, resolved: str | None) -> dict[str, Any]:
    """The B record: B's state and question texts; question types, score scales, labels and meta from A."""
    questions = {}
    for name, q in a["request"]["questions"].items():
        entry = {"type": q["type"], "instructions": reply["questions"][name]}
        if "criteria" in q:
            entry["criteria"] = q["criteria"]
        questions[name] = entry
    meta = json.loads(json.dumps(a["meta"]))
    meta["e1"] = {"condition": "B", "source_id": a["id"], "model": model, "resolved_model": resolved, "prompt_version": PROMPT_VERSION,
                  "request_sha256": request_sha256, "objects": objects}
    return {"schema": a["schema"], "id": f"{B_PREFIX}-{a['id'].split('-', 1)[1]}", "generator": a.get("generator"), "meta": meta,
            "request": {"state": reply["state"], "questions": questions}, "labels": a["labels"]}
