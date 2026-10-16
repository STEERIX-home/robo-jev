"""Task v0.8c — A16 (the arm's servo decisions) and S16 (soccer): robo-jev-web's requests of 2026-10-16, reported v0.8 evaluation sets.

robo-jev-serve sent the files (``artifacts/bench/robojev-requests-2026-10-16/``, zip sha256 ``90dd1afb…``; captured with robo-jev v0.7 =
``steerix/robo-jev-2b@cc0b99f2`` answering from the real endpoint). Both sets are reported only — they enter neither P1, P2, FORMAT nor the
amended flip condition (its third source stays R14, the 10-14 requests) — and are sealed for training (never trained on) but not never-sent:
the bench has already sent them.

**A16 — the arm** (``arm-decisions.jsonl``, 90 requests): the decisions bodies robo-jev-web's server sends robo-jev for the servo (the state is
a constant protocol line, the facts are in the questions — ``buildServoQuestions``), six questions each (intent 10, x / y / z 11, wrist 5,
fingers 3 options). ``arm-servo.jsonl`` holds the same 90 calls as the page sent them, with v0.7's answers (checked: same order, every
distance in the question text equals ``facts.move`` rounded to 0.1).

* *The rule* of every x / y / z question is written in its text: "The gripper still has to travel D cm along A. … plus for a positive
  distance, minus for a negative one. Take the largest step that does not overshoot: far (20) when the distance is at least 20 cm, mid (8)
  when at least 8, near (3) when at least 3, small (1) when at least 1, fine (0.4) when between 0.2 and 1 cm, and stay when it is within
  0.2 cm." Parsing (:data:`DISTANCE`): ``still has to travel ([+-]?\\d+(?:\\.\\d+)?) cm along ([xyz])`` → the signed distance D. Label
  (:func:`arm_label`, the controller's thresholds, which reproduce serve's v0.7 numbers): |D| ≥ 20 far, ≥ 8 mid, ≥ 3 near, ≥ 1 small, ≥ 0.2
  fine, below 0.2 stay; the sign of D picks plus / minus. **Boundary**: 50 of the 90 x distances are exactly −0.2; "within 0.2 cm" in the
  text would make them stay (:func:`arm_label` with ``boundary_stay=True`` — reported as a sensitivity), the controller's "below 0.2 is stay"
  makes them minus_fine (the main reading).
* *Bands* (:func:`arm_band`): stay (< 0.2), 0.2–1, 1–3, 3–8, 8–20, ≥ 20 cm, signed.
* *The phrase ablation* (:func:`ablate_y`): a copy of each request with only the y question, " (up is plus)" removed from its text.

**S16 — soccer** (``soccer.jsonl``, 55 requests): 12 questions each — six players' movement (15 options) and gaze (``<player>_look``, 6
options); no labels. Read across serve's three orders (0, 1, 2 trailing spaces on the state: the key changes, the prompt does not).
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from robo_jev.bench.gates_v08 import ARTIFACTS, option_range_mark
from robo_jev.decisions.records import SCHEMA

__all__ = [
    "A16_PREFIX", "A16_SET", "AXES", "BANDS", "DISTANCE", "R16_DIR", "R16_SOURCE", "S16_PREFIX", "S16_SET", "UP_IS_PLUS", "ablate_y",
    "arm_band", "arm_distance", "arm_label", "arm_record", "arm_rows", "arm_summary", "check_frozen", "seed_mean_rows", "soccer_group",
    "soccer_record", "stored_answers",
]

A16_SET, A16_PREFIX = "v08-a16", "v08a16-"
S16_SET, S16_PREFIX = "v08-s16", "v08s16-"
R16_DIR = ARTIFACTS / "v08" / "r16"
R16_SOURCE = ARTIFACTS / "bench" / "robojev-requests-2026-10-16" / "robojev-requests-2026-10-16"
R16_ZIP_SHA256_PREFIX = "90dd1afb"
AXES = ("x", "y", "z")
DISTANCE = re.compile(r"still has to travel ([+-]?\d+(?:\.\d+)?) cm along ([xyz])")
UP_IS_PLUS = " (up is plus)"
#: Distance bands (inclusive lower bound, exclusive upper bound) and the step each one gets.
BANDS: tuple[tuple[str, float, float, str], ...] = (("<0.2", 0.0, 0.2, "stay"), ("0.2-1", 0.2, 1.0, "fine"), ("1-3", 1.0, 3.0, "small"),
                                                    ("3-8", 3.0, 8.0, "near"), ("8-20", 8.0, 20.0, "mid"), (">=20", 20.0, float("inf"), "far"))
#: The bands where the rule moves but by little — "the stay rate on small distances" is read on them.
SMALL_BANDS = ("0.2-1", "1-3")


def arm_distance(text: str) -> tuple[str, float]:
    """(axis, signed distance in cm) parsed from an x / y / z question's text (:data:`DISTANCE`); raises if the sentence is missing."""
    found = DISTANCE.search(str(text))
    if not found:
        raise ValueError(f"no 'still has to travel D cm along A' in {str(text)[:80]!r}")
    return found.group(2), float(found.group(1))


def arm_band(d: float) -> str:
    """The unsigned band name of a distance (:data:`BANDS`)."""
    m = abs(float(d))
    for name, low, high, _ in BANDS:
        if low <= m < high:
            return name
    raise ValueError(d)


def arm_label(d: float, *, boundary_stay: bool = False) -> str:
    """The rule's option for a signed distance (module docstring): |d| ≥ 20 far, ≥ 8 mid, ≥ 3 near, ≥ 1 small, ≥ 0.2 fine, below 0.2 stay;
    with ``boundary_stay`` a distance of exactly ±0.2 is stay ("within 0.2 cm")."""
    m = abs(float(d))
    if m < 0.2 or (boundary_stay and m <= 0.2):
        return "stay"
    step = next(step for name, low, high, step in BANDS if low <= m < high)
    return f"{'plus' if d > 0 else 'minus'}_{step}"


def _sign(option: str) -> str | None:
    return option.split("_", 1)[0] if option.startswith(("plus_", "minus_")) else None


def arm_record(index: int, line: Mapping[str, Any], *, source_line: int) -> dict[str, Any]:
    """An arm request as a sealed Decisions record (``v08a16-NNNN``, no labels; ``meta.rule`` = per axis the parsed distance, the rule's
    option under both boundary readings and the signed band)."""
    request = line["request"]
    rule = {}
    for axis in AXES:
        parsed_axis, d = arm_distance(request["questions"][axis]["instructions"])
        if parsed_axis != axis:
            raise ValueError(f"line {source_line}: question {axis} speaks of axis {parsed_axis}")
        rule[axis] = {"distance": d, "label": arm_label(d), "label_boundary_stay": arm_label(d, boundary_stay=True), "band": arm_band(d),
                      "sign": None if arm_label(d) == "stay" else ("plus" if d > 0 else "minus")}
    return {"schema": SCHEMA, "id": f"{A16_PREFIX}{index:04d}",
            "meta": {"set": A16_SET, "env": "arm", "route": line.get("route"), "status": line.get("status"), "reply_model": line.get("replyModel"),
                     "request_model": request.get("model"), "source": {"file": "arm-decisions.jsonl", "line": source_line}, "rule": rule,
                     "option_range": option_range_mark()},
            "request": {"state": request["state"], "questions": request["questions"]}}


def ablate_y(record: Mapping[str, Any]) -> dict[str, Any]:
    """The phrase ablation: the record with only its y question, " (up is plus)" removed from the text (same id, ``meta.ablation``)."""
    question = dict(record["request"]["questions"]["y"])
    text = str(question["instructions"])
    if text.count(UP_IS_PLUS) != 1:
        raise ValueError(f"{record['id']}: ' (up is plus)' appears {text.count(UP_IS_PLUS)} times in the y question")
    question["instructions"] = text.replace(UP_IS_PLUS, "")
    meta = {**record["meta"], "ablation": {"question": "y", "removed": UP_IS_PLUS}, "target": ["y"]}
    return {**record, "meta": meta, "request": {"state": record["request"]["state"], "questions": {"y": question}}}


def soccer_group(name: str) -> str:
    """``gaze`` for a ``<player>_look`` question, else ``movement``."""
    return "gaze" if str(name).endswith("_look") else "movement"


def soccer_record(index: int, line: Mapping[str, Any], *, source_line: int) -> dict[str, Any]:
    """A soccer request as a sealed Decisions record (``v08s16-NNNN``, no labels; ``meta.groups`` = movement / gaze per question)."""
    request = line["request"]
    return {"schema": SCHEMA, "id": f"{S16_PREFIX}{index:04d}",
            "meta": {"set": S16_SET, "env": "soccer", "route": line.get("route"), "status": line.get("status"), "reply_model": line.get("replyModel"),
                     "request_model": request.get("model"), "source": {"file": "soccer.jsonl", "line": source_line},
                     "groups": {name: soccer_group(name) for name in request["questions"]}, "option_range": option_range_mark()},
            "request": {"state": request["state"], "questions": request["questions"]}}


def stored_answers(records: Sequence[Mapping[str, Any]], servo_lines: Sequence[Mapping[str, Any]], *, reading: str = "stored") -> dict[str, Any]:
    """v0.7's answers stored in ``arm-servo.jsonl`` in the score-file shape ``{record id: {question: {reading: {choice, probabilities}}}}``,
    after checking that line i is record i's call (every ``facts.move`` rounds to the record's parsed distance)."""
    if len(records) != len(servo_lines):
        raise ValueError(f"{len(records)} records vs {len(servo_lines)} servo lines")
    out: dict[str, Any] = {}
    for record, line in zip(records, servo_lines):
        move = (line.get("request") or {}).get("facts", {}).get("move") or {}
        for axis in AXES:
            if round(float(move[axis]), 1) != record["meta"]["rule"][axis]["distance"]:
                raise ValueError(f"{record['id']}: servo line's {axis} {move[axis]} is not the record's {record['meta']['rule'][axis]['distance']}")
        out[str(record["id"])] = {name: {reading: {"choice": a.get("choice"), "probabilities": a.get("probabilities")}}
                                  for name, a in (line.get("answers") or {}).items()}
    return out


def arm_rows(records: Sequence[Mapping[str, Any]], answers: Mapping[str, Any], *, reading: str, axes: Iterable[str] = AXES,
             boundary_stay: bool = False) -> list[dict[str, Any]]:
    """One row per (record, axis) the model answered: distance, band, the rule's option, the model's top option and the indicators
    ``agree``, ``sign_error`` (the rule moves, the model moves the other way; None when either says stay) and ``top_is_stay``."""
    rows = []
    for record in records:
        rid = str(record["id"])
        for axis in axes:
            answer = (((answers.get(rid) or {}).get(axis) or {}).get(reading)) or None
            if not answer or answer.get("choice") is None:
                continue
            rule = record["meta"]["rule"][axis]
            label = rule["label_boundary_stay"] if boundary_stay else rule["label"]
            top = str(answer["choice"])
            sign_error = None if label == "stay" or top == "stay" else float(_sign(top) != _sign(label))
            rows.append({"id": rid, "axis": axis, "distance": rule["distance"], "band": rule["band"], "label": label, "top": top,
                         "agree": float(top == label), "sign_error": sign_error, "top_is_stay": float(top == "stay"),
                         "p_top": float(max((answer.get("probabilities") or {top: 1.0}).values()))})
    return rows


def seed_mean_rows(first: Sequence[Mapping[str, Any]], second: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """An arm's rows: per (record, axis) both seeds answered, the mean of the two seeds' indicators (a sign error counts only where both
    seeds' value is defined; otherwise the defined one, or None)."""
    b = {(r["id"], r["axis"]): r for r in second}
    out = []
    for r in first:
        other = b.get((r["id"], r["axis"]))
        if other is None:
            continue
        signs = [x for x in (r["sign_error"], other["sign_error"]) if x is not None]
        out.append({**{k: r[k] for k in ("id", "axis", "distance", "band", "label")}, "top": None,
                    "agree": (r["agree"] + other["agree"]) / 2.0, "top_is_stay": (r["top_is_stay"] + other["top_is_stay"]) / 2.0,
                    "sign_error": sum(signs) / len(signs) if signs else None, "p_top": (r["p_top"] + other["p_top"]) / 2.0})
    return out


def _rate(values: Iterable[float | None]) -> dict[str, Any]:
    kept = [float(v) for v in values if v is not None]
    return {"value": sum(kept) / len(kept) if kept else None, "n": len(kept), "count": sum(kept)}


def arm_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """The A16 readings of one model or arm: rule agreement (total, per axis, per axis × signed band), sign errors (per axis; over the rows
    where both the rule and the model move), the stay rate per band and on small distances (0.2–3 cm)."""
    by_axis: dict[str, list] = defaultdict(list)
    by_band: dict[str, list] = defaultdict(list)
    for r in rows:
        by_axis[r["axis"]].append(r)
        sign = "" if r["label"] == "stay" else ("+" if r["distance"] > 0 else "-")
        by_band[f"{r['axis']}:{sign}{r['band']}"].append(r)
    small = [r for r in rows if r["band"] in SMALL_BANDS]
    return {
        "agreement": _rate(r["agree"] for r in rows),
        "agreement_by_axis": {a: _rate(r["agree"] for r in v) for a, v in sorted(by_axis.items())},
        "agreement_by_axis_band": {k: _rate(r["agree"] for r in v) for k, v in sorted(by_band.items())},
        "sign_errors_by_axis": {a: _rate(r["sign_error"] for r in v) for a, v in sorted(by_axis.items())},
        "sign_errors": _rate(r["sign_error"] for r in rows),
        "stay_rate_by_axis_band": {k: _rate(r["top_is_stay"] for r in v) for k, v in sorted(by_band.items())},
        "stay_rate_small": _rate(r["top_is_stay"] for r in small),
        "stay_rate_small_by_axis": {a: _rate(r["top_is_stay"] for r in small if r["axis"] == a) for a in AXES},
        "rows": len(rows),
    }


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def check_frozen(directory: str | Path = R16_DIR) -> list[str]:
    """Problems with the frozen A16 / S16 files (empty = unchanged since frozen): every file in ``frozen-before-scoring.json``."""
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
