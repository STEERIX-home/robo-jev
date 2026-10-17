#!/usr/bin/env python
"""Task v0.9b — the N9g self-check: relabel every N9g question independently of its generator (CPU).

    artifacts/scratch/v09b/cpu.sh selfcheck scripts/v09_n9g_selfcheck.py [--input artifacts/v09/n9g/N9g.jsonl] [--out <json>]

It never imports the generator (:mod:`robo_jev.bench.n9g_v09`). From each record it takes the rendered texts (state, question, options)
and the record's ``meta.spec`` — which only says which strings are the reading, the bounds and the candidates, and which option key stands
for which band — and then:

1. **finds every spec string in the rendered texts**, with its unit, where it belongs (the reading in the state, bounds in the rule, each
   candidate's numbers in its own option, each item's number in its own fact line), and checks that every number written anywhere in the
   texts is one of them (no stray number);
2. **parses** every number with its own parser (signs ``+`` / ``-`` / U+2212, ``$``, thousands commas, no leading zero) into
   :class:`fractions.Fraction`;
3. **reads the rule's structure from the text** where the text decides it: a threshold rule's step ↔ lower-bound pairing (clause by
   clause), the sign convention (the positive clause names the positive action), the window ↔ name pairing, a comparison's direction (its
   superlative), the ordering options against their names;
4. **relabels** with its own logic and its own unit table (conversion: the reading converted into the rule's unit), and checks the label
   equals ``meta.correct`` and the one-hot ``labels``;
5. checks that **no reading sits on a bound** (no label depends on an inclusive / exclusive reading), that exactly **one** candidate of a
   selection meets every constraint, that comparisons have **no ties**, and that every ``near`` claim holds (``rel``: within 5 % of a bound;
   ``abs``: within 0.05).

Exit 0 iff every record passes; the JSON report lists problems per record.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from fractions import Fraction
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

DEFAULT_INPUT = REPO / "artifacts" / "v09" / "n9g" / "N9g.jsonl"
NUMBER = re.compile(r"(?<![\w.])([+\-−]?)(\$?)((?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?|\.\d+)(?![\w])")
FIVE_PERCENT, ABS_NEAR = Fraction(5, 100), Fraction(5, 100)


def parse(text: str) -> Fraction:
    """One written number → Fraction (own parser)."""
    found = NUMBER.fullmatch(text.strip())
    if not found:
        raise ValueError(f"not a number: {text!r}")
    sign, _, body = found.groups()
    value = Fraction(body.replace(",", ""))
    return -value if sign in ("-", "−") else value


def numbers_in(text: str) -> list[Fraction]:
    out = []
    for m in NUMBER.finditer(text):
        sign, _, body = m.groups()
        value = Fraction(body.replace(",", ""))
        out.append(-value if sign in ("-", "−") else value)
    return out


def with_unit(number: str, unit: str) -> re.Pattern:
    """The pattern of a written number with its unit (money: $ before the digits, after a sign)."""
    if unit == "$":
        sign = number[0] if number[:1] in "+-−" else ""
        return re.compile(re.escape(sign + "$" + number[len(sign):]) + r"(?!\d|[.,]\d)")
    if unit in ("fraction", ""):
        return re.compile(r"(?<![\d.])" + re.escape(number) + r"(?!\d|[.,]\d)")
    return re.compile(r"(?<![\d.,])" + re.escape(number) + r"\s?" + re.escape(unit))


#: Own conversion table: reading unit → (rule unit, a function to the rule unit).
TO_RULE = {
    ("mm", "cm"): lambda x: x / 10, ("cm", "m"): lambda x: x / 100, ("mm", "m"): lambda x: x / 1000, ("g", "kg"): lambda x: x / 1000,
    ("s", "min"): lambda x: x / 60, ("ms", "s"): lambda x: x / 1000, ("min", "h"): lambda x: x / 60, ("mL", "L"): lambda x: x / 1000,
    ("mV", "V"): lambda x: x / 1000, ("mA", "A"): lambda x: x / 1000, ("kPa", "bar"): lambda x: x / 100, ("kPa", "MPa"): lambda x: x / 1000,
    ("Pa", "kPa"): lambda x: x / 1000, ("cents", "$"): lambda x: x / 100, ("fraction", "%"): lambda x: x * 100,
    ("cm/s", "m/s"): lambda x: x / 100, ("mm/s", "m/s"): lambda x: x / 1000, ("m/min", "m/s"): lambda x: x / 60,
    ("K", "°C"): lambda x: x - Fraction(27315, 100),
}
#: Own superlatives (direction of an "extreme" comparison).
HIGH_WORDS = ("farthest", "fastest", "steepest", "longest", "heaviest", "warmest", "most", "highest", "most expensive")
LOW_WORDS = ("nearest", "slowest", "least steep", "shortest", "lightest", "coldest", "least", "lowest", "cheapest")


def direction(phrase: str) -> str | None:
    """``high`` / ``low`` from a superlative phrase (own word lists), None if it says neither or both."""
    phrase = phrase.lower()
    high, low = any(w in phrase for w in HIGH_WORDS), any(w in phrase for w in LOW_WORDS)
    return None if high == low else ("high" if high else "low")


def _near_ok(kind: Any, value: Fraction, bounds: list[Fraction]) -> bool:
    if kind == "rel":
        return any(b != 0 and abs(value - b) <= abs(b) * FIVE_PERCENT for b in bounds)
    if kind == "abs":
        return any(abs(value - b) <= ABS_NEAR for b in bounds)
    return True


def check_record(record: dict[str, Any]) -> tuple[str | None, list[str]]:
    """(the relabelled key, problems) of one N9g record."""
    problems: list[str] = []
    meta = record.get("meta") or {}
    spec = meta.get("spec") or {}
    questions = (record.get("request") or {}).get("questions") or {}
    if list(questions) != ["q"]:
        return None, [f"questions {list(questions)} (want exactly 'q')"]
    q = questions["q"]
    criteria = q.get("criteria") or {}
    if q.get("type") != "choice" or not 2 <= len(criteria) <= 20:
        problems.append(f"type {q.get('type')} / {len(criteria)} options")
    state, instructions = record["request"]["state"], q.get("instructions") or ""
    rule_text = state + "\n" + instructions
    label = ((record.get("labels") or {}).get("q") or {}).get("probabilities") or {}
    if set(label) != set(criteria) or sorted(label.values()) != [0.0] * (len(label) - 1) + [1.0]:
        problems.append("the label is not one-hot over the options")
    hot = next((k for k, v in label.items() if v == 1.0), None)
    if hot != meta.get("correct"):
        problems.append(f"labels say {hot}, meta.correct {meta.get('correct')}")
    kind = spec.get("kind")
    allowed: list[Fraction] = []
    relabel = None
    try:
        if kind == "threshold":
            relabel = _threshold(spec, state, rule_text, criteria, problems, allowed, meta)
        elif kind in ("band", "tolerance"):
            relabel = _band(spec, state, rule_text, criteria, problems, allowed, meta)
        elif kind == "windows":
            relabel = _windows(spec, state, criteria, problems, allowed, meta)
        elif kind == "selection":
            relabel = _selection(spec, rule_text, criteria, problems, allowed)
        elif kind in ("extreme", "second", "closest", "order"):
            relabel = _comparison(spec, state, instructions, criteria, problems, allowed)
        elif kind == "conversion":
            relabel = _conversion(spec, state, rule_text, criteria, problems, allowed, meta)
        else:
            problems.append(f"unknown spec kind {kind!r}")
    except (ValueError, KeyError, IndexError, ZeroDivisionError) as error:
        problems.append(f"could not relabel: {error!r}")
    written = numbers_in(state) + numbers_in(instructions) + [x for v in criteria.values() for x in numbers_in(v)]
    stray = sorted({str(x) for x in written if x not in set(allowed)})
    if stray:
        problems.append(f"numbers in the texts that are not the spec's: {stray[:5]}")
    if relabel is not None and relabel != meta.get("correct"):
        problems.append(f"relabelled {relabel}, stored {meta.get('correct')}")
    return relabel, problems


def _need(pattern: re.Pattern, text: str, what: str, problems: list[str]) -> None:
    if not pattern.search(text):
        problems.append(f"{what} ({pattern.pattern!r}) not found where it belongs")


def _threshold(spec, state, rule_text, criteria, problems, allowed, meta):
    unit = spec["unit"]
    value = parse(spec["value_text"])
    reading_line = [line for line in state.split("\n") if with_unit(spec["value_text"], unit).search(line)]
    if len(reading_line) != 1:
        problems.append("the reading is not on exactly one state line")
    thresholds = [parse(t) for t in spec["thresholds_text"]]
    if thresholds != sorted(thresholds) or len(set(thresholds)) != len(thresholds):
        problems.append("thresholds not strictly ascending")
    for text in spec["thresholds_text"]:
        _need(with_unit(text, unit), rule_text, f"threshold {text} in the reading's unit", problems)
    allowed += [value, *thresholds]
    keys = spec["keys"]
    sizes = [k[len(spec["pos_action"]) + 1:] for k in keys["pos"]]
    if [k[len(spec["neg_action"]) + 1:] for k in keys["neg"]] != sizes:
        problems.append("the two directions' step names differ")
    clauses = [c for c in re.split(r"[;\n]", rule_text) if c.strip()]
    for k, size in enumerate(sizes):
        found = [c for c in clauses if re.search(rf"\b{re.escape(size)} step\b", c)]
        if len(found) != 1:
            problems.append(f"step {size!r}: {len(found)} rule clauses")
            continue
        first = numbers_in(found[0])
        if not first or first[0] != thresholds[k]:
            problems.append(f"step {size!r}: its clause starts at {first[:1]}, want the lower bound {thresholds[k]}")
    sign = next((c for c in re.split(r"(?<=[.;])\s", rule_text) if "positive" in c and "negative" not in c), None)
    neg_sign = next((c for c in re.split(r"(?<=[.;])\s", rule_text) if "negative" in c and "positive" not in c), None)
    stem = lambda action: action.split("_")[-1] if action.startswith(("turn_", "shift_")) else action.split("_")[0]
    if not sign or stem(spec["pos_action"])[:5] not in sign.lower():
        problems.append(f"the positive clause does not name {spec['pos_action']}: {sign!r}")
    if not neg_sign or stem(spec["neg_action"])[:5] not in neg_sign.lower():
        problems.append(f"the negative clause does not name {spec['neg_action']}: {neg_sign!r}")
    for key in keys["pos"]:
        if not criteria.get(key, "").lower().startswith(stem(spec["pos_action"])[:4]) and stem(spec["pos_action"]) not in criteria.get(key, "").lower():
            problems.append(f"option {key} does not read as the positive action")
    size_of = abs(value)
    if size_of in thresholds:
        problems.append("the reading sits on a threshold")
    step = sum(1 for t in thresholds if size_of >= t)
    if not _near_ok(meta.get("near"), size_of, thresholds):
        problems.append(f"near={meta.get('near')} does not hold")
    if step == 0:
        return keys["hold"]
    return (keys["pos"] if value > 0 else keys["neg"])[step - 1]


def _band(spec, state, rule_text, criteria, problems, allowed, meta):
    unit = spec["unit"]
    value = parse(spec["value_text"])
    _need(with_unit(spec["value_text"], unit), state, "the reading", problems)
    if spec["kind"] == "band":
        lo, hi = parse(spec["lo_text"]), parse(spec["hi_text"])
        allowed += [lo, hi]
        for text in (spec["lo_text"], spec["hi_text"]):
            _need(with_unit(text, unit) if unit != "$" else re.compile(re.escape("$") + re.escape(text) + r"(\.\d+)?"), rule_text, "a band end", problems)
    else:
        c, t = parse(spec["center_text"]), parse(spec["tol_text"])
        allowed += [c, t]
        lo, hi = c - t, c + t
        if "plus or minus" not in rule_text:
            problems.append("a tolerance rule without 'plus or minus'")
        for text in (spec["center_text"], spec["tol_text"]):
            _need(with_unit(text, unit), rule_text, f"the tolerance's {text} in the reading's unit", problems)
    allowed.append(value)
    if not lo < hi:
        problems.append("an empty band")
    keys = spec["keys"]
    for key, word in zip(keys, ("Below", "Inside", "Above")):
        if not criteria[key].startswith(word):
            problems.append(f"option {key} does not start with {word!r}")
    if value in (lo, hi):
        problems.append("the reading sits on a band end")
    if not _near_ok(meta.get("near"), value, [lo, hi]):
        problems.append(f"near={meta.get('near')} does not hold")
    return keys[0] if value < lo else keys[2] if value > hi else keys[1]


def _windows(spec, state, criteria, problems, allowed, meta):
    unit = spec["unit"]
    value = parse(spec["value_text"])
    allowed.append(value)
    _need(with_unit(spec["value_text"], unit), state, "the reading", problems)
    inside, edges = [], []
    for w in spec["windows"]:
        lo, hi = parse(w["lo_text"]), parse(w["hi_text"])
        allowed += [lo, hi]
        edges += [lo, hi]
        name = criteria[w["key"]].rstrip(".")
        line = [l for l in state.split("\n") if l.startswith(f"- {name}:")]
        if len(line) != 1 or numbers_in(line[0]) != [lo, hi]:
            problems.append(f"window {w['key']}: its line does not give {lo} to {hi}")
        elif not all(with_unit(x, unit).search(line[0]) for x in (w["lo_text"], w["hi_text"])):
            problems.append(f"window {w['key']}: its bounds are not in the reading's unit {unit}")
        if lo < value < hi:
            inside.append(w["key"])
    if value in edges:
        problems.append("the reading sits on a window edge")
    if len(inside) > 1:
        problems.append(f"the reading is in {len(inside)} windows")
    if not criteria[spec["none"]].startswith("None of"):
        problems.append("the none option does not say so")
    if not _near_ok(meta.get("near"), value, edges):
        problems.append(f"near={meta.get('near')} does not hold")
    return inside[0] if inside else spec["none"]


def _selection(spec, rule_text, criteria, problems, allowed):
    attrs = spec["attrs"]
    bounds = {}
    for a in attrs:
        texts = a["bound_text"] if isinstance(a["bound_text"], list) else [a["bound_text"]]
        parsed = [parse(x) for x in texts]
        bounds[a["attr"]] = parsed
        allowed += parsed
        for x in texts:
            pattern = (re.compile(re.escape("$") + re.escape(x) + r"(\.\d+)?") if a["unit"] == "$" else with_unit(x, a["unit"]))
            _need(pattern, rule_text, f"bound {a['attr']}", problems)
    meeting = []
    for key, cand in spec["candidates"].items():
        text = criteria.get(key, "")
        ok = True
        for a in attrs:
            vt = cand["values_text"][a["attr"]]
            _need(with_unit(vt, a["unit"]), text, f"{key} {a['attr']}", problems)
            v = parse(vt)
            allowed.append(v)
            b = bounds[a["attr"]]
            if v in b:
                problems.append(f"{key} {a['attr']} sits on its bound")
            if a["op"] == "min":
                ok &= v > b[0]
            elif a["op"] == "max":
                ok &= v < b[0]
            else:
                ok &= b[0] < v < b[1]
        if not text.rstrip(".").endswith(cand["status"]):
            problems.append(f"{key}: the status is not at the end of its text")
        ok &= cand["status"] == spec["status_ok"]
        if ok:
            meeting.append(key)
    if set(spec["candidates"]) != set(criteria):
        problems.append("candidates and options differ")
    if spec["status_ok"] not in rule_text:
        problems.append("the rule does not name the good status")
    if len(meeting) != 1:
        problems.append(f"{len(meeting)} candidates meet every constraint")
        return None
    return meeting[0]


def _comparison(spec, state, instructions, criteria, problems, allowed):
    unit = spec["unit"]
    values = {}
    lines = state.split("\n")
    for name, vt in spec["items"].items():
        v = parse(vt)
        values[name] = v
        allowed.append(v)
        own = [l for l in lines if l.split(" ")[1:2] == [name.capitalize()] and with_unit(vt, unit).search(l)]
        if len(own) != 1:
            problems.append(f"item {name}: no single fact line with {vt}")
    if len(set(values.values())) != len(values):
        problems.append("tied values")
    kind = spec["kind"]
    text = instructions.lower()
    if kind == "order":
        found = re.search(r"from (.+?) to (.+?)\?", text)
        first = direction(found.group(1)) if found else None
        if found is None or first is None or direction(found.group(2)) in (None, first):
            problems.append("no order direction in the question")
        ascending = first == "low"
        right = sorted(values, key=lambda n: values[n], reverse=not ascending)
        hits = []
        for key, names in spec["orders"].items():
            if criteria[key].rstrip(".") != ", then ".join(n.capitalize() for n in names):
                problems.append(f"{key}: its text does not match its order")
            if names == right:
                hits.append(key)
        if len(hits) != 1:
            problems.append(f"{len(hits)} orderings are right")
            return None
        return hits[0]
    noun = spec["noun"]
    if kind == "closest":
        target = parse(spec["target_text"])
        allowed.append(target)
        _need(with_unit(spec["target_text"], unit), instructions, "the target", problems)
        gaps = sorted((abs(v - target), n) for n, v in values.items())
        if gaps[0][0] == gaps[1][0]:
            problems.append("a tie for closest")
        return f"{noun}_{gaps[0][1]}"
    if kind == "second":
        found = re.search(r",\s*([^,.]+?) first\.", text)
        way = direction(found.group(1)) if found else None
        if way is None:
            problems.append("no ranking direction")
        high = way == "high"
        ranked = sorted(values, key=lambda n: values[n], reverse=high)
        return f"{noun}_{ranked[1]}"
    way = direction(text)
    if way is None:
        problems.append(f"cannot tell the direction of {instructions!r}")
    high = way == "high"
    best = max(values, key=lambda n: values[n]) if high else min(values, key=lambda n: values[n])
    return f"{noun}_{best}"


def _conversion(spec, state, rule_text, criteria, problems, allowed, meta):
    ru, rd = spec["rule_unit"], spec["read_unit"]
    reading = parse(spec["value_text"])
    allowed.append(reading)
    _need(with_unit(spec["value_text"], "" if rd == "fraction" else rd), state, "the reading", problems)
    bounds = [parse(b) for b in spec["bounds_text"]]
    allowed += bounds
    if bounds != sorted(bounds):
        problems.append("bounds not ascending")
    positions = []
    for text in spec["bounds_text"]:
        pattern = (re.compile(re.escape("$") + re.escape(text) + r"(\.\d+)?") if ru == "$" else with_unit(text, ru))
        found = pattern.search(rule_text)
        if not found:
            problems.append(f"bound {text} {ru} not in the rule")
        else:
            positions.append(found.start())
    if positions != sorted(positions):
        problems.append("the rule does not give its bounds in ascending order")
    convert = TO_RULE.get((rd, ru))
    if convert is None:
        problems.append(f"no own conversion for {rd} -> {ru}")
        return None
    value = convert(reading)
    if value in bounds:
        problems.append("the converted reading sits on a bound")
    if not _near_ok(meta.get("near"), value, bounds):
        problems.append(f"near={meta.get('near')} does not hold in the rule's unit")
    index = sum(1 for b in bounds if value > b)
    keys = spec["keys"]
    if len(keys) != len(bounds) + 1:
        problems.append("options do not match the bounds")
    return keys[index]


def check(records: list[dict[str, Any]]) -> dict[str, Any]:
    problems = {}
    relabelled = Counter()
    for record in records:
        key, found = check_record(record)
        relabelled[str(key == (record.get("meta") or {}).get("correct"))] += 1
        if found:
            problems[str(record.get("id"))] = found
    ids = [str(r.get("id")) for r in records]
    whole = Counter(json.dumps(r.get("request"), sort_keys=True, ensure_ascii=False) for r in records)
    general = []
    if len(set(ids)) != len(ids):
        general.append("duplicate ids")
    if max(whole.values() or [0]) > 1:
        general.append("duplicate requests")
    return {"records": len(records), "relabel_matches": relabelled.get("True", 0), "records_with_problems": len(problems),
            "general_problems": general, "problems": problems, "ok": not problems and not general}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)
    records = [json.loads(line) for line in args.input.read_text(encoding="utf-8").splitlines() if line.strip()]
    result = {"input": str(args.input), **check(records)}
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "problems"}, indent=1))
    for rid, found in list(result["problems"].items())[:15]:
        print(rid, found)
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
