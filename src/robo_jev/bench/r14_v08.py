"""Task v0.8b — R14: robo-jev-web's collected bench requests (2026-10-14), a reported evaluation set measured under robo-jev-serve's conditions.

The user provided the requests robo-jev-web's pages sent to ``/v1/decisions`` (``artifacts/bench/robojev-requests-2026-10-14/``; lab, café,
delivery, humanoid, forklift; captured with robo-jev-web branch codex/robojev-direct, e5a3b3a). R14 is reported only: it is the third source
of the amended next-step condition (F's flip rate between order seeds > 0.20 → the consistency loss goes into v0.9) and enters neither P1,
P2 nor FORMAT. Its files are sealed for training (never trained on) but not never-sent — the bench has already sent them.

**Serve's conditions** (robo-jev-serve @ :data:`SERVE_COMMIT`, read-only clone; ``src/robo_jev_serve/{trivial,scores,engine,app}.py`` and
HANDOFF.md appendix B), replicated exactly:

* *dedup*: per environment, the first line of each distinct ``json.dumps(line["request"], sort_keys=True)`` (the whole request, ``model``
  included) — serve's counts: lab 116, café 118, delivery 207, humanoid 119, forklift 640;
* *1-option choices* are answered by the server with p = 1 and never reach the model (``trivial.split``); here they are left out of every
  reading (a question with one option cannot flip);
* *score questions with ≥ 2 criteria* are asked as a level choice, keys ``"0"``…``"n−1"``, the criteria sentences as written
  (``scores.to_choice``) — humanoid and forklift ``offset`` (5 levels);
* *orders*: serve's endpoint answers in the bundle's default single order (seed 1, key = sha256 of the state as received); serve varied the
  order by appending 0, 1 or 2 spaces to the state — the key changes, the prompt (built from ``state.strip()``) does not: readings
  ``spaces-0``, ``spaces-1``, ``spaces-2``;
* *serve's sample* (its appendix B, the pipeline check): ``lines[::max(1, len(lines) // 20)][:20]`` per environment; serve counted the choice
  questions of the original request with ≥ 2 options, a flip = more than one distinct top-1 over the three orders.

**Our sample** for the v0.8 readings: up to :data:`SAMPLE_PER_ENV` requests per environment, evenly spaced over the deduplicated requests
(index ⌊i · n / K⌋). **Tags**: every choice option of every sampled request, PASSIVE / ACTIVE by the rel07 rules, then B8's, then the R14 rules
below (first match wins; every option must match; frozen before scoring). **Planner**: the request's own recommendation, parsed from its
LOCAL PLAN text (:func:`planner_of`).
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from robo_jev.bench import b8_v08
from robo_jev.bench.gates_v08 import ARTIFACTS, option_range_mark
from robo_jev.decisions.records import SCHEMA

__all__ = [
    "ORDERS", "R14_DIR", "R14_ENVS", "R14_PREFIX", "R14_RULES", "R14_SET", "R14_SOURCE", "SAMPLE_PER_ENV", "SERVE_COMMIT", "SERVE_SAMPLE",
    "PASSIVE_KIND", "check_frozen", "dedup", "even_sample", "model_questions", "passive_kind", "planner_of", "read_env", "serve_sample", "split_trivial", "tag", "tags_payload",
    "to_level_choice", "to_record",
]

R14_SET, R14_PREFIX = "v08-r14", "v08r14-"
R14_DIR = ARTIFACTS / "v08" / "r14"
R14_ALL = R14_DIR / "R14-all.jsonl"
R14_SOURCE = ARTIFACTS / "bench" / "robojev-requests-2026-10-14" / "robojev-requests-2026-10-14"
R14_ZIP_SHA256_PREFIX = "c344da09fb89119b"
R14_ENVS = ("lab", "cafe", "delivery", "humanoid", "forklift")
SERVE_COMMIT = "7b54004bb335c2abdcd3675054221a194166cf3b"
SERVE_SAMPLE = 20
SAMPLE_PER_ENV = 40
#: The three order seeds: 0, 1 and 2 trailing spaces on the state (serve's appendix B).
ORDERS = ("spaces-0", "spaces-1", "spaces-2")

#: R14's added tag rules (after rel07's 38 and B8's 13, first match wins). Speeds: stopping is PASSIVE, any moving speed ACTIVE (the slow
#: ones borderline); speech and announcements ACTIVE (a person addressed), saying nothing PASSIVE; offsets: the planned line PASSIVE
#: (borderline: continue unchanged), a side ACTIVE; the mast left to the planner or held PASSIVE; gaze targets ACTIVE; reader commands
#: ACTIVE, sending nothing PASSIVE; corrections (go to a pose) ACTIVE; humanoid / forklift manoeuvres ACTIVE, standing and watching PASSIVE,
#: keep walking PASSIVE (borderline, as rel07's "Carry on").
R14_RULES: list[tuple[str, str, str, bool]] = [
    # café advance / say / stand
    ("v08r14-cafe-step-aside", "ACTIVE", r"^Step about [\d.]+ m to your (left|right), out of someone's way", True),
    ("v08r14-cafe-walk-normal", "ACTIVE", r"^Walk at the normal pace", False),
    ("v08r14-cafe-walk-carefully", "ACTIVE", r"^Walk carefully", True),
    ("v08r14-do-not-move-the-feet", "PASSIVE", r"^Do not move the feet this second", False),
    ("v08r14-say-nothing", "PASSIVE", r"^Say nothing", False),
    ("v08r14-say-one-moment", "ACTIVE", r"^\"One moment, please\"", True),
    ("v08r14-say-words", "ACTIVE", r"^\"[^\"]+\" — (to|offering)", False),
    ("v08r14-cafe-stand-side", "ACTIVE", r"^On the customer's (left|right)\.", False),
    # delivery announce / offset / pace
    ("v08r14-announce-nothing", "PASSIVE", r"^nothing$", False),
    ("v08r14-announce-chime", "ACTIVE", r"^a short chime", False),
    ("v08r14-announce-words", "ACTIVE", r"^\"[^\"]+\" — ", False),
    ("v08r14-offset-on-the-line", "PASSIVE", r"^on the walking line", True),
    ("v08r14-offset-to-a-side", "ACTIVE", r"^[\d.]+ m to the (left|right)", False),
    ("v08r14-pace-planners", "ACTIVE", r"^the planner's pace", True),
    ("v08r14-pace-speed", "ACTIVE", r"^[\d.]+ m/s — the door", True),
    # forklift advance / maneuver / mast
    ("v08r14-forklift-creep", "ACTIVE", r"^Creep slowly", True),
    ("v08r14-forklift-proceed", "ACTIVE", r"^Proceed at the plan's nominal speed", False),
    ("v08r14-forklift-brake", "PASSIVE", r"^Brake to a standstill", False),
    ("v08r14-forklift-docking", "ACTIVE", r"^Run the planned docking manoeuvre", False),
    ("v08r14-forklift-confirm", "ACTIVE", r"^Reverse gently with the lifted pallet", False),
    ("v08r14-forklift-insert", "ACTIVE", r"^Creep straight along the pocket axis", False),
    ("v08r14-forklift-lift", "ACTIVE", r"^Stop and raise the mast", True),
    ("v08r14-forklift-retry", "ACTIVE", r"^Abandon the current manoeuvre", False),
    ("v08r14-mast-to-the-planner", "PASSIVE", r"^Let the planner handle the mast", True),
    ("v08r14-mast-hold", "PASSIVE", r"^Hold the mast height", False),
    # humanoid advance / gaze / maneuver
    ("v08r14-humanoid-walk-normal", "ACTIVE", r"^Walk the plan at normal speed", False),
    ("v08r14-humanoid-walk-carefully", "ACTIVE", r"^Walk the plan carefully", True),
    ("v08r14-look-at", "ACTIVE", r"^Look (at|along) ", False),
    ("v08r14-humanoid-approach", "ACTIVE", r"^Walk to the carton", False),
    ("v08r14-humanoid-keep-walking", "PASSIVE", r"^Keep walking the planned path", True),
    ("v08r14-humanoid-cross", "ACTIVE", r"^Cross the forklift lane", False),
    ("v08r14-humanoid-squat", "ACTIVE", r"^Squat (at the table and take|and set)", False),
    ("v08r14-humanoid-pick-other", "ACTIVE", r"^Give up on this carton and go for another", False),
    ("v08r14-humanoid-wait-at-edge", "PASSIVE", r"^Stop at the edge of the forklift lane", True),
    ("v08r14-humanoid-wait-here", "PASSIVE", r"^Stand still this second", False),
    # lab advance / correction / reader
    ("v08r14-lab-back-off", "ACTIVE", r"^Back off:", True),
    ("v08r14-lab-go-on", "ACTIVE", r"^Go on at the usual speed", True),
    ("v08r14-lab-half-speed", "ACTIVE", r"^The next move at half speed", False),
    ("v08r14-lab-halt", "PASSIVE", r"^Halt the move under way", False),
    ("v08r14-lab-camera-pose", "ACTIVE", r"^Go where the camera's latest look puts the station", False),
    ("v08r14-lab-taught-pose", "ACTIVE", r"^Go to the station's taught pose", False),
    ("v08r14-reader-nothing", "PASSIVE", r"^Send the reader nothing", False),
    ("v08r14-reader-command", "ACTIVE", r"^(Open|Close) the reader's drawer|^Start a read of the plate", False),
]
RULES = list(b8_v08.RULES) + R14_RULES
#: A reported reading, not a tag (controller, 2026-10-16 ~14:08 KST, before any v0.8 scoring; P2 still reads the frozen PASSIVE tag): every
#: PASSIVE rule of rel07, B8 and R14 split into ``hold`` (stop / wait / hold / do nothing / hand to a person) and ``continue`` (continue
#: unchanged / as planned / keep going) — v0.8a's H-prior action classes weigh the two differently (stop / wait / hold down, continue up).
PASSIVE_KIND = {
    # rel07
    "wait": "hold", "let-the-move-go-on": "continue", "hold-still-for-a-look": "hold", "hold-for-the-researcher": "hold", "stay": "hold",
    "carry-on": "continue", "stop-where-you-are": "hold", "call-the-researcher": "hold", "call-the-operator": "hold", "go-on-with-the-pass": "continue",
    "look-ahead-standing-still": "hold", "look-round-the-room": "hold", "nothing-new": "hold", "something-the-cafe-does-not-make": "hold",
    "not-made-out": "hold",
    # B8
    "v08-hold-this-second": "hold", "v08-go-on-with-the-work": "continue", "v08-set-a-tube-aside": "hold",
    # R14
    "v08r14-do-not-move-the-feet": "hold", "v08r14-say-nothing": "hold", "v08r14-announce-nothing": "hold", "v08r14-offset-on-the-line": "continue",
    "v08r14-forklift-brake": "hold", "v08r14-mast-to-the-planner": "continue", "v08r14-mast-hold": "hold", "v08r14-humanoid-keep-walking": "continue",
    "v08r14-humanoid-wait-at-edge": "hold", "v08r14-humanoid-wait-here": "hold", "v08r14-lab-halt": "hold", "v08r14-reader-nothing": "hold",
}


def passive_kind(rule_id: str) -> str | None:
    """``hold`` or ``continue`` for an option tagged PASSIVE by `rule_id` (:data:`PASSIVE_KIND`); None for an ACTIVE rule."""
    return PASSIVE_KIND.get(str(rule_id))
COMPILED = [(rid, cls, re.compile(pat, re.IGNORECASE), borderline) for rid, cls, pat, borderline in RULES]
_PLANNER = re.compile(r"\(planner recommends: ([A-Za-z0-9_]+)\)")
_LAB_PLANNER = {"move": re.compile(r"The recommended move: ([A-Za-z0-9_]+)\b"), "reader": re.compile(r"The recommended reader command: ([A-Za-z0-9_]+)\b"),
                "correction": re.compile(r"The recommended correction: ([A-Za-z0-9_]+)\b")}
#: The question a "(planner recommends: …)" line belongs to, per environment.
_PLANNED_QUESTION = {"cafe": "intent", "delivery": "manoeuvre", "humanoid": "maneuver", "forklift": "maneuver"}


def tag(text: str) -> tuple[str, str, bool]:
    """(rule id, class, borderline) of an option text — first match of rel07's, B8's, then R14's rules; no match is an error."""
    for rid, cls, pat, borderline in COMPILED:
        if pat.search(str(text)):
            return rid, cls, borderline
    raise ValueError(f"no rule matches the option text: {text!r}")


def read_env(env: str, source: Path = R14_SOURCE) -> list[dict[str, Any]]:
    path = Path(source) / f"{env}.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def dedup(lines: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Serve's dedup: the first line of each distinct ``json.dumps(line["request"], sort_keys=True)`` (``model`` included), in file order."""
    seen: set[str] = set()
    out = []
    for line in lines:
        key = json.dumps(line["request"], sort_keys=True)
        if key not in seen:
            seen.add(key)
            out.append(dict(line))
    return out


def serve_sample(lines: Sequence[Any], n: int = SERVE_SAMPLE) -> list[int]:
    """The indices serve's appendix B sent: ``lines[::max(1, len(lines) // n)][:n]``."""
    return list(range(0, len(lines), max(1, len(lines) // n)))[:n]


def even_sample(lines: Sequence[Any], k: int = SAMPLE_PER_ENV) -> list[int]:
    """Up to `k` indices evenly spaced over the whole list: ⌊i · n / k⌋."""
    n = len(lines)
    k = min(int(k), n)
    return [(i * n) // k for i in range(k)]


def split_trivial(questions: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """serve's ``trivial.split``: (questions sent to the model, 1-option choices the server answers with p = 1)."""
    single = {name: q for name, q in questions.items() if isinstance(q, dict) and q.get("type") == "choice" and isinstance(q.get("criteria"), dict)
              and len(q["criteria"]) == 1}
    return {name: q for name, q in questions.items() if name not in single}, single


def to_level_choice(questions: Mapping[str, Any]) -> tuple[dict[str, Any], set[str]]:
    """serve's ``scores.to_choice``: a score with ≥ 2 criteria becomes a choice with keys "0"…"n−1" and the criteria sentences as written."""
    asked: dict[str, Any] = {}
    converted: set[str] = set()
    for name, question in questions.items():
        if isinstance(question, dict) and question.get("type") == "score" and isinstance(question.get("criteria"), list) and len(question["criteria"]) >= 2:
            asked[name] = {"type": "choice", "instructions": question.get("instructions"),
                           "criteria": {str(i): str(text) for i, text in enumerate(question["criteria"])}}
            converted.add(name)
        else:
            asked[name] = question
    return asked, converted


def model_questions(questions: Mapping[str, Any]) -> dict[str, Any]:
    """What R14 scores of a request, under serve's adapter: the choice questions with ≥ 2 options after the 1-option split and the score →
    level-choice conversion (yes/no questions and 1-option choices are not read). ``{"asked": {name: question}, "converted": [...],
    "trivial": [...], "skipped": [...]}`` (``asked`` in the request's question order)."""
    rest, single = split_trivial(questions)
    asked, converted = to_level_choice(rest)
    chosen = {name: q for name, q in asked.items() if q.get("type") == "choice" and len(q.get("criteria") or {}) >= 2}
    return {"asked": chosen, "converted": sorted(converted), "trivial": sorted(single), "skipped": sorted(set(asked) - set(chosen))}


def planner_of(env: str, state: str) -> dict[str, str]:
    """The request's own recommendation per question, from its LOCAL PLAN text: lab's "The recommended move / reader command / correction:
    X"; café intent, delivery manoeuvre, humanoid and forklift maneuver from "(planner recommends: X)"."""
    out: dict[str, str] = {}
    if env == "lab":
        for question, pattern in _LAB_PLANNER.items():
            found = pattern.search(state)
            if found:
                out[question] = found.group(1)
        return out
    found = _PLANNER.search(state)
    if found and env in _PLANNED_QUESTION:
        out[_PLANNED_QUESTION[env]] = found.group(1)
    return out


def to_record(env: str, index: int, line: Mapping[str, Any], *, source_line: int) -> dict[str, Any]:
    """A deduplicated request as a sealed Decisions record (``v08r14-<env>-NNNN``, no labels); the request is the page's ``state`` and
    ``questions`` exactly (``model`` kept in meta)."""
    request = line["request"]
    return {"schema": SCHEMA, "id": f"{R14_PREFIX}{env}-{index:04d}",
            "meta": {"set": R14_SET, "env": env, "route": line.get("route"), "status": line.get("status"), "asked": line.get("asked"),
                     "reply_model": line.get("replyModel"), "request_model": request.get("model"), "source": {"file": f"{env}.jsonl", "line": source_line},
                     "dedup_index": index, "option_range": option_range_mark()},
            "request": {"state": request["state"], "questions": request["questions"]}}


def tags_payload(records: Sequence[Mapping[str, Any]], *, files: Mapping[str, str], created: str) -> dict[str, Any]:
    """Tags of every choice option of the given records' model questions (after serve's adapter; level choices carry no tag) and each
    question's planner recommendation."""
    out, counts = [], {}
    for record in records:
        env = record["meta"]["env"]
        plan = planner_of(env, record["request"]["state"])
        mq = model_questions(record["request"]["questions"])
        entry = {"id": record["id"], "env": env, "questions": {}}
        for name, question in mq["asked"].items():
            level = name in mq["converted"]
            options = []
            for key, text in question["criteria"].items():
                if level:
                    options.append({"key": key, "text": text, "tag": None, "rule": "level", "borderline": False})
                    continue
                rid, cls, borderline = tag(text)
                options.append({"key": key, "text": text, "tag": cls, "rule": rid, "borderline": borderline})
                counts[f"{env}/{cls}"] = counts.get(f"{env}/{cls}", 0) + 1
            planner = plan.get(name)
            entry["questions"][name] = {"n_options": len(options), "level": level, "options": options, "planner": planner if planner in question["criteria"] else None,
                                        "planner_text": planner, "planner_tag": next((o["tag"] for o in options if o["key"] == planner), None)}
        entry["trivial"] = mq["trivial"]
        out.append(entry)
    return {"kind": "v08-r14-tags", "classes": b8_v08.CLASSES if hasattr(b8_v08, "CLASSES") else None,
            "rules": [{"id": rid, "class": cls, "pattern": pat, "borderline": b, "origin": "r14" if rid.startswith("v08r14-") else
                       ("v0.8b" if rid.startswith("v08-") else "rel07")} for rid, cls, pat, b in RULES],
            "files": dict(files), "created": created, "requests": out, "counts": counts}


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def check_frozen(directory: str | Path = R14_DIR) -> list[str]:
    """Problems with the frozen R14 files (empty = unchanged since frozen): every file named in ``frozen-before-scoring.json`` must have its
    recorded sha256."""
    directory = Path(directory)
    frozen_path = directory / "frozen-before-scoring.json"
    if not frozen_path.is_file():
        return [f"missing {frozen_path.name} in {directory}"]
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    problems = []
    for name, want in (frozen.get("files") or {}).items():
        path = directory / name
        if not path.is_file():
            problems.append(f"missing {name}")
        elif sha256_file(path) != want:
            problems.append(f"{name} changed after it was frozen")
    return problems
