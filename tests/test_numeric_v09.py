"""The measurement-literacy records of arm CN (Task v0.9a, :mod:`robo_jev.simgen.numeric_v09`).

**The re-labeller here is independent of the generator**: it has its own number parser and unit table, finds every number of a question
in the record's text (state, instructions, option descriptions; a match must not be part of a longer number), reads the rule's wording
(``at least`` / ``more than``, limits included or not, which sign means which direction), recomputes the answer with its own logic from the
numbers it parsed, and compares it with the one-hot label. ``meta.numeric`` only tells it which written number plays which role. It runs on
a generated sample of every family and template here and on **every** numeric record of the built input when that exists.

Also: records validate; labels are one-hot; families, templates, domains and number formats are all used; step readings are dense near the
thresholds; the arm workcell's rule is never generated; random codes and the many-option opt-in follow v0.8's rules.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from fractions import Fraction
from pathlib import Path

import pytest
from helpers import REPO

from robo_jev.decisions.records import validate_record
from robo_jev.simgen import numeric_v09 as N

BUILT = REPO / "artifacts" / "simgen" / "v09" / "train-v09n.jsonl"

# -- the re-labeller's own number parser and unit table ---------------------------------------------------------------------------------------

#: symbol → (domain, factor to the domain's base unit, offset) — written here independently of the generator's table.
UNIT_TABLE = {
    "mm": ("distance", Fraction(1, 1000), 0), "cm": ("distance", Fraction(1, 100), 0), "m": ("distance", Fraction(1), 0),
    "km": ("distance", Fraction(1000), 0), "mm/s": ("speed", Fraction(1, 1000), 0), "cm/s": ("speed", Fraction(1, 100), 0),
    "m/s": ("speed", Fraction(1), 0), "km/h": ("speed", Fraction(1000, 3600), 0), "°": ("angle", Fraction(1), 0),
    "degrees": ("angle", Fraction(1), 0), "ms": ("time", Fraction(1, 1000), 0), "s": ("time", Fraction(1), 0), "min": ("time", Fraction(60), 0),
    "h": ("time", Fraction(3600), 0), "g": ("mass", Fraction(1, 1000), 0), "kg": ("mass", Fraction(1), 0), "t": ("mass", Fraction(1000), 0),
    "°C": ("temperature", Fraction(1), 0), "K": ("temperature", Fraction(1), Fraction(-27315, 100)), "%": ("percent", Fraction(1), 0),
    "": ("percent", Fraction(100), 0),
}
_NUMBER = re.compile(r"^(?P<sign>[+\-−]?) ?(?P<int>\d{1,3}(?:,\d{3})+|\d+)(?:\.(?P<frac>\d+))? ?(?P<unit>.*)$")


def parse(text: str) -> tuple[Fraction, str]:
    """A written number with its unit → (value in that unit, unit symbol)."""
    found = _NUMBER.match(text)
    assert found, f"not a number: {text!r}"
    digits = found["int"].replace(",", "") + (found["frac"] or "")
    value = Fraction(int(digits), 10 ** len(found["frac"] or ""))
    if found["sign"] in ("-", "−"):
        value = -value
    unit = found["unit"]
    assert unit in UNIT_TABLE, f"unknown unit in {text!r}"
    return value, unit


def base(text: str) -> Fraction:
    value, unit = parse(text)
    _, factor, offset = UNIT_TABLE[unit]
    return value * factor + offset


def appears(needle: str, haystack: str) -> bool:
    """`needle` occurs in `haystack` and not as part of a longer number (no digit, separator or sign right before or after it)."""
    for found in re.finditer(re.escape(needle), haystack):
        before = haystack[found.start() - 1] if found.start() > 0 else " "
        after = haystack[found.end()] if found.end() < len(haystack) else " "
        tail = haystack[found.end():found.end() + 2]
        if before.isdigit() or before in ".,+-−" or after.isdigit() or re.match(r"[.,]\d", tail):
            continue
        return True
    return False


def everything(record: dict, name: str | None = None) -> str:
    """What the model sees for question `name` (the state, its instructions and its option descriptions) — every question's when None."""
    request = record["request"]
    parts = [request["state"]]
    for qname, q in request["questions"].items():
        if name is not None and qname != name:
            continue
        parts.append(q["instructions"])
        parts.extend(str(v) for v in q["criteria"].values())
    return "\n".join(parts)


#: The question whose prompt the checks read (set by :func:`relabel` for each question in turn).
_CURRENT: dict[str, str | None] = {"name": None}


def found(record: dict, *texts: str) -> None:
    text = everything(record, _CURRENT["name"])
    for t in texts:
        assert appears(t, text), f"{record['id']}/{_CURRENT['name']}: {t!r} is not in the question's prompt"


def any_phrase(record: dict, phrases: list[str]) -> bool:
    text = everything(record, _CURRENT["name"])
    return any(p in text for p in phrases)


# -- the re-labeller ------------------------------------------------------------------------------------------------------------------------------


def relabel_step(record: dict, name: str, rule: dict, q: dict) -> str:
    found(record, *rule["thresholds"], q["value"]["text"])
    t = [base(x) for x in rule["thresholds"]]
    assert t == sorted(t) and len(set(t)) == len(t)
    first, last = rule["thresholds"][0], rule["thresholds"][-1]
    if rule["inclusive"] == "lower":  # the wording says a threshold value belongs to the band above it
        assert any_phrase(record, [f"at {last} or more", f"at least {last}", f"≥ {last}"]), "lower: the top band starts AT its threshold"
        assert any_phrase(record, [f"below {first}", f"at least {first}", f"< {first}"]), "lower: the bottom band ends BELOW the first threshold"
    else:
        assert any_phrase(record, [f"more than {last}", f"> {last}"]), "upper: the top band starts ABOVE its threshold"
        assert any_phrase(record, [f"{first} or less", f"more than {first}", f"≤ {first}"]), "upper: the bottom band includes the first threshold"
    pos, neg = rule["sign"]["+"], rule["sign"]["-"]
    text = everything(record, name)
    assert re.search(rf"\b{re.escape(pos)}\b", text) and re.search(rf"\b{re.escape(neg)}\b", text), "both directions are named"
    v = base(q["value"]["text"])
    size = abs(v)
    count = sum(1 for x in t if (size >= x if rule["inclusive"] == "lower" else size > x))
    cats = rule["categories"]
    if rule["hold"]:
        if count == 0:
            return rule["hold"]
        cat = cats[count - 1]
    else:
        assert v != 0
        cat = cats[count]
    word = pos if v > 0 else neg
    key = rule["keys"][("+" if v > 0 else "-") + cat]
    assert word in key, f"the key {key!r} names its direction {word!r}"
    assert cat in key or key.endswith(str(cats.index(cat) + 1)), f"the key {key!r} names its size"
    return key


def _window_verdict(v: Fraction, low: Fraction, high: Fraction, low_in: bool, high_in: bool) -> str:
    if v < low or (v == low and not low_in):
        return "below"
    if v > high or (v == high and not high_in):
        return "above"
    return "inside"


def relabel_window(record: dict, name: str, rule: dict, q: dict) -> str:
    w = rule["window"]
    if "centre" in w:
        found(record, w["centre"], w["tolerance"], q["value"]["text"])
        centre, tol = base(w["centre"]), base(w["tolerance"]) - UNIT_TABLE[parse(w["tolerance"])[1]][2]
        low, high = centre - tol, centre + tol
        assert any_phrase(record, ["limits included", "still passes"])
        flags = (w["low_in"], w["high_in"])
        assert flags == (True, True)
    else:
        found(record, w["low"], w["high"], q["value"]["text"])
        low, high = base(w["low"]), base(w["high"])
        flags = (w["low_in"], w["high_in"])
        words = {(True, True): ["both limits included", "up to and including"], (False, False): ["more than", "limits excluded"],
                 (True, False): ["at least", "(not included)"]}[flags]
        assert any_phrase(record, words), f"the wording states the limits {flags}"
    verdict = _window_verdict(base(q["value"]["text"]), low, high, *flags)
    return rule["names"][verdict]


def relabel_zones(record: dict, name: str, rule: dict, q: dict) -> str:
    found(record, *rule["edges"], q["value"]["text"])
    edges = [base(e) for e in rule["edges"]]
    lower = rule["lower_inclusive"]
    assert any_phrase(record, ["includes its lower limit" if lower else "includes its upper limit"])
    v = base(q["value"]["text"])
    index = sum(1 for e in edges if v > e or (v == e and lower))
    return rule["zones"][index]


_OPS = {"ge": ("at least", lambda v, a: v >= a), "gt": ("more than", lambda v, a: v > a), "le": ("at most", lambda v, a: v <= a),
        "lt": ("less than", lambda v, a: v < a)}


def _candidate_text(record: dict, name: str, key: str) -> str:
    """Where a candidate's attributes are written: its option description, or its line / table row in the state (or the instructions)."""
    desc = record["request"]["questions"][name]["criteria"][key]
    if "status" in desc:
        return desc
    for line in (record["request"]["state"] + "\n" + record["request"]["questions"][name]["instructions"]).splitlines():
        if re.match(rf"^(- )?{re.escape(key)}(:| \|)", line):
            return line
    raise AssertionError(f"{record['id']}: no line for {key}")


def relabel_select(record: dict, name: str, rule: dict, q: dict) -> str:
    passing = []
    for c in rule["constraints"]:
        if c["op"] == "between":
            assert any_phrase(record, [f"a {c['attr']} between {c['a']} and {c['b']} (inclusive)"])
        else:
            assert any_phrase(record, [f"a {c['attr']} of {_OPS[c['op']][0]} {c['a']}"])
    if rule["status"]:
        assert any_phrase(record, [f"status {rule['status'].replace('_', ' ')}"])
    for key, cand in q["candidates"].items():
        segment = _candidate_text(record, name, key)
        assert f"{cand['status'].replace('_', ' ')}" in segment
        ok = not rule["status"] or cand["status"] == rule["status"]
        for c in rule["constraints"]:
            written = cand["values"][c["attr"]]
            assert appears(written, segment), f"{key}: {written!r} not in its description"
            v = base(written)
            if c["op"] == "between":
                ok = ok and base(c["a"]) <= v <= base(c["b"])
            else:
                ok = ok and _OPS[c["op"]][1](v, base(c["a"]))
        if ok:
            passing.append(key)
    assert len(passing) == 1, f"{record['id']}: {len(passing)} candidates meet every constraint"
    return passing[0]


def relabel_compare(record: dict, name: str, rule: dict, q: dict) -> str:
    found(record, *q["items"].values())
    values = {k: base(t) for k, t in q["items"].items()}
    assert len(set(values.values())) == len(values)
    mode = q["mode"]
    ranked = sorted(values, key=values.get)
    if mode in ("closest", "farthest"):
        found(record, q["target"])
        target = base(q["target"])
        distance = {k: abs(v - target) for k, v in values.items()}
        best = sorted(distance, key=distance.get)
        assert distance[best[0]] != distance[best[1]] and distance[best[-1]] != distance[best[-2]]
        return best[0] if mode == "closest" else best[-1]
    if mode.startswith("order"):
        wanted = ranked if mode == "order_asc" else ranked[::-1]
        matches = [k for k, order in q["orders"].items() if list(order) == wanted]
        assert len(matches) == 1
        sep = " < " if mode == "order_asc" else " > "
        assert record["request"]["questions"][name]["criteria"][matches[0]] == sep.join(x.replace("_", " ") for x in wanted)
        return matches[0]
    return {"largest": ranked[-1], "smallest": ranked[0], "second_largest": ranked[-2], "second_smallest": ranked[1]}[mode]


def relabel_convert(record: dict, name: str, rule: dict, q: dict) -> str:
    v = base(q["value"]["text"])
    found(record, q["value"]["text"])
    if q["kind"] == "convert_step":
        found(record, *rule["thresholds"])
        t = [base(x) for x in rule["thresholds"]]
        last = rule["thresholds"][-1]
        assert any_phrase(record, [f"at {last}", f"more than {last}"])
        lower = rule["inclusive"] == "lower"
        assert any_phrase(record, ["or more"] if lower else ["or less"])
        return rule["names"][sum(1 for x in t if (v >= x if lower else v > x))]
    if q["kind"] == "convert_limit":
        found(record, rule["limit"])
        limit = base(rule["limit"])
        assert any_phrase(record, ["at most"] if rule["op"] == "le" else ["stay below"])
        ok = v <= limit if rule["op"] == "le" else v < limit
        return rule["names"]["pass" if ok else "fail"]
    w = rule["window"]
    found(record, w["low"], w["high"])
    assert any_phrase(record, ["both limits included"])
    return rule["names"][_window_verdict(v, base(w["low"]), base(w["high"]), True, True)]


def relabel(record: dict) -> dict[str, str]:
    """question → the answer the re-labeller computes."""
    numeric = record["meta"]["numeric"]
    out = {}
    for name, q in numeric["questions"].items():
        _CURRENT["name"] = name  # every check reads this question's own prompt text
        kind = q["kind"]
        if kind == "step":
            out[name] = relabel_step(record, name, numeric["rule"], q)
        elif kind == "window":
            out[name] = relabel_window(record, name, numeric["rule"], q)
        elif kind == "zones":
            out[name] = relabel_zones(record, name, numeric["rule"], q)
        elif kind == "select":
            out[name] = relabel_select(record, name, numeric["rule"], q)
        elif kind == "compare":
            out[name] = relabel_compare(record, name, numeric["rule"], q)
        else:
            out[name] = relabel_convert(record, name, numeric["rule"], q)
    return out


def label_of(record: dict, name: str) -> str:
    probabilities = record["labels"][name]["probabilities"]
    ones = [k for k, p in probabilities.items() if p == 1.0]
    assert len(ones) == 1 and all(p in (0.0, 1.0) for p in probabilities.values()), "one-hot"
    return ones[0]


def check(record: dict) -> None:
    assert validate_record(record) == [], validate_record(record)[:3]
    assert set(record["meta"]["numeric"]["questions"]) == set(record["request"]["questions"])
    answers = relabel(record)
    for name, answer in answers.items():
        assert answer == label_of(record, name), f"{record['id']}/{name}: re-labelled {answer!r}, labelled {label_of(record, name)!r}"


# -- generated sample ----------------------------------------------------------------------------------------------------------------------------

SAMPLE = 600


@pytest.fixture(scope="module")
def sample() -> list[dict]:
    return [N.make_record(family, i) for family in N.FAMILIES for i in range(SAMPLE)]


def test_every_generated_record_relabels_exactly(sample):
    for record in sample:
        check(record)


def test_the_relabeller_catches_a_wrong_label_and_a_wrong_number(sample):
    record = json.loads(json.dumps(next(r for r in sample if r["meta"]["numeric"]["family"] == "step")))
    name = next(iter(record["labels"]))
    answer = label_of(record, name)
    other = next(k for k in record["labels"][name]["probabilities"] if k != answer)
    record["labels"][name]["probabilities"] = {k: (1.0 if k == other else 0.0) for k in record["labels"][name]["probabilities"]}
    with pytest.raises(AssertionError):
        check(record)
    record = json.loads(json.dumps(next(r for r in sample if r["meta"]["numeric"]["family"] == "select")))
    q = record["meta"]["numeric"]["questions"]["q_choose"]
    key = q["answer"]
    attr, text = next(iter(q["candidates"][key]["values"].items()))
    q["candidates"][key]["values"][attr] = "9" + text  # a number the text does not show
    with pytest.raises(AssertionError):
        check(record)


def test_families_templates_domains_and_formats_are_all_used(sample):
    templates = Counter(r["meta"]["numeric"]["template"] for r in sample)
    assert set(templates) == {t for ts in N.TEMPLATES.values() for t in ts}
    domains = Counter(r["meta"]["numeric"]["domain"] for r in sample)
    assert set(domains) == set(N.DOMAINS)
    styles = [r["meta"]["numeric"]["style"] for r in sample]
    for key in ("plus", "sign_space", "unit_space", "thousands", "fixed"):
        assert {s[key] for s in styles} == {True, False}
    assert {s["minus"] for s in styles} == {"ascii", "unicode"}
    text = "\n".join(everything(r) for r in sample)
    for piece in ("+", "−", ".0", "°", "°C", " %", "%", "km/h", "m/s", " ms", " min", " kg", " g", "K"):
        assert piece in text
    assert re.search(r"\d,\d{3}", text) and re.search(r"\d{4}", text), "with and without thousands separators"
    assert re.search(r"\d(cm|mm|kg|s|%)\b", text) and re.search(r"\d (cm|mm|kg|s|%)", text), "with and without a space before the unit"
    sizes = Counter(len(q["criteria"]) for r in sample for q in r["request"]["questions"].values())
    assert min(sizes) >= 2 and max(sizes) <= 26
    select = Counter(len(r["request"]["questions"]["q_choose"]["criteria"]) for r in sample if r["meta"]["numeric"]["family"] == "select")
    assert set(select) == set(range(4, 21)), "4–20 candidates"


def test_step_readings_are_dense_near_the_thresholds_and_cover_every_band(sample):
    near = bands = total = 0
    seen = Counter()
    for r in sample:
        numeric = r["meta"]["numeric"]
        if numeric["family"] != "step":
            continue
        t = [base(x) for x in numeric["rule"]["thresholds"]]
        for q in numeric["questions"].values():
            v = abs(base(q["value"]["text"]))
            total += 1
            near += any(abs(v - x) <= max(abs(x) * Fraction(5, 100), Fraction(5, 100) * N.UNITS[q["value"]["unit"]].factor) or v == x for x in t)
            seen[sum(1 for x in t if v >= x)] += 1
            bands += 1
    assert near / total >= 0.4, f"only {near}/{total} readings sit near a threshold"
    assert set(range(5)) <= set(seen), f"every band is used: {sorted(seen.items())}"


def test_exact_boundary_readings_follow_the_wording():
    t = [Fraction(1), Fraction(5), Fraction(12)]
    assert N.step_label(Fraction(5), t, inclusive="lower", hold=True) == ("+", 1)
    assert N.step_label(Fraction(5), t, inclusive="upper", hold=True) == ("+", 0)
    assert N.step_label(Fraction(-1), t, inclusive="lower", hold=True) == ("-", 0)
    assert N.step_label(Fraction(-1), t, inclusive="upper", hold=True) == ("0", None)
    assert N.step_label(Fraction(12), t[1:], inclusive="lower", hold=False) == ("+", 2)
    assert N.step_label(Fraction(1, 10), t[1:], inclusive="lower", hold=False) == ("+", 0)
    assert N.interval_label(Fraction(3), Fraction(3), Fraction(7), low_in=False, high_in=True) == "below"
    assert N.zone_label(Fraction(4), [Fraction(2), Fraction(4)], lower_inclusive=True) == 2
    assert N.zone_label(Fraction(4), [Fraction(2), Fraction(4)], lower_inclusive=False) == 1


def test_numbers_render_in_every_format():
    style = N.Style(plus=True, minus="−", sign_space=False, unit_space=True, thousands=True, fixed=True)
    assert N.render(Fraction(12345, 10), "g", style, 2, signed=True) == "+1,234.50 g"
    assert N.render(Fraction(-3, 10), "cm", style, 2, signed=True) == "−0.30 cm"
    plain = N.Style(plus=False, minus="-", sign_space=True, unit_space=False, thousands=False, fixed=False)
    assert N.render(Fraction(12345, 10), "g", plain, 2, signed=True) == "1234.5g"
    assert N.render(Fraction(-3, 10), "cm", plain, 2, signed=True) == "- 0.3cm"
    assert N.render(Fraction(7), "deg", plain) == "7°" and N.render(Fraction(42, 100), "frac", plain) == "0.42"
    for text in ("+1,234.50 g", "−0.30 cm", "1234.5g", "- 0.3cm", "7°", "0.42"):
        parse(text)


def test_the_arm_workcells_rule_is_never_generated(sample):
    assert N.arm_rule_problems("cm", N.ARM_THRESHOLDS_CM, ["a", "b"]) and N.arm_rule_problems("m", [Fraction(1)], ["far", "near", "fine"])
    assert N.arm_rule_problems("s", [Fraction(1)], ["hold", "fine"], directions=("plus", "minus"))
    assert N.arm_rule_problems("s", [Fraction(1)], ["stay", "large"], directions=("heat", "cool"))
    assert not N.arm_rule_problems("s", [Fraction(1)], ["hold", "fine"], directions=("heat", "cool"))
    for r in sample:
        rule = r["meta"]["numeric"]["rule"]
        if r["meta"]["numeric"]["family"] != "step":
            continue
        names = rule["categories"] + ([rule["hold"]] if rule["hold"] else [])
        assert not N.arm_rule_problems(rule["unit"], [parse(x)[0] for x in rule["thresholds"]], names, directions=tuple(rule["sign"].values()))
        for key in rule["keys"].values():
            assert not re.fullmatch(r"(plus|minus)_(far|mid|near|small|fine)", key) and key != "stay"
        text = everything(r).lower()
        assert "servo" not in text and "gripper" not in text and "up is plus" not in text


def test_codes_and_the_many_option_opt_in_follow_v08(sample):
    from robo_jev.decisions.records import OPTION_RANGE_MARK_255

    questions = [(r, name) for r in sample for name in r["request"]["questions"]]
    with_codes = sum(1 for r, name in questions if name in (r["meta"].get("codes") or {}))
    assert abs(with_codes / len(questions) - 0.25) < 0.03
    for r in sample:
        wide = any(len(q["criteria"]) > 8 for q in r["request"]["questions"].values())
        assert (r["meta"].get("option_range") == OPTION_RANGE_MARK_255) is wide
        assert r["meta"]["set"] == N.SET_NAME and r["id"].startswith(N.ID_PREFIX)


def test_the_schedule_holds_the_family_shares_and_ids_avoid_refused_names():
    order = N.schedule(1000)
    counts = Counter(order)
    assert sum(counts.values()) == 1000 and all(abs(counts[f] - 1000 * s) <= 1 for f, s in N.FAMILY_SHARES.items())
    refused = {"v09num-step-000000", "v09num-select-000000"}
    records = N.generate(40, id_ok=lambda rid: rid not in refused)
    ids = [r["id"] for r in records]
    assert len(set(ids)) == 40 and not refused & set(ids)
    assert any(i.endswith("-b") for i in ids)


@pytest.mark.skipif(not BUILT.is_file(), reason="the v0.9 input is built by scripts/assemble_v09.py")
def test_the_built_input_relabels_every_numeric_record():
    count = 0
    with BUILT.open(encoding="utf-8") as handle:
        for line in handle:
            if '"v09-numeric"' not in line:
                continue
            record = json.loads(line)
            if (record.get("meta") or {}).get("set") != N.SET_NAME:
                continue
            check(record)
            count += 1
    manifest = json.loads(BUILT.with_name("train-v09n.manifest.json").read_text(encoding="utf-8"))
    assert count == manifest["numeric_records"] > 0
