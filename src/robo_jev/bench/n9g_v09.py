"""Task v0.9b — N9g: the sealed numeric-rule gate of the PRE-DECLARED v0.9 VERDICT RULE (measurement literacy, P-num2).

The v0.9 rule (progress.md, 2026-10-17 ~07:56 KST, before any v0.9 data existed) reads a numeric-rule gate "built by a separate author who
never sees the generator. Unseen templates and domains, covering the five families, with multi-constraint selection at 4–20 options. Labels
are exact by construction." This module is that gate's design, its builder and its readings. It was written by task v0.9b, which never opened
task v0.9a's worktree, generator or ``artifacts/simgen/v09/``; every template and phrasing here is this task's own.

**Families** (:data:`FAMILIES`; one choice question ``q`` per record, exactly one right option by construction):

* ``threshold`` — a signed reading, a stated sign convention and ordered magnitude thresholds → a direction and a step size (or "no
  action" below the smallest threshold): 5, 7, 9 or 11 options;
* ``interval`` — membership: one band (below / inside / above), a target with a ± tolerance, or several named windows with gaps (one window
  or "none of them"): 3–5 options;
* ``selection`` — multi-constraint selection: candidates with several stated numbers (and a status); exactly one meets every stated
  constraint, every other breaks exactly one (cycled, often by a small margin): 4–20 options;
* ``comparison`` — the highest / lowest, the second when ranked (four or more readings), the closest to a target, or the order of three or
  four readings, in each domain's own words (farthest / nearest, coldest / warmest, …; signed values, decimal traps such as 9.11 against
  9.9): 3–8 options;
* ``conversion`` — a rule stated in one unit and a reading in another (mm / cm / m, g / kg, ms / s / min / h, mL / L, mV / V, mA / A, Pa /
  kPa / bar / MPa, cents / dollars, % / fraction of full, cm/s / mm/s / m/min / m/s, K / °C): 2–4 options.

**Domains** (:data:`DOMAINS`): the v0.9 generator's stated list — distance, speed, angle, time, mass, temperature, percent (:data:`SEEN_DOMAINS`)
— and five outside it (:data:`UNSEEN_DOMAINS`: pressure, voltage, volume, currency, current), so domain transfer is read.

**Values** (:func:`n9g_records`): every band of every rule is used; about half of the threshold / interval / conversion readings sit near a
boundary — within ±5 % of it (``near = "rel"``) or within ±0.05 absolute (``near = "abs"``) — never on it, so the label never depends on an
inclusive / exclusive reading. Number formats (:data:`FORMATS`, one per question for the judged numbers): plain, explicit sign, Unicode minus,
two fixed decimals, integer, no leading zero, thousands separator, and (comparison / selection) mixed — each number with its own decimals. Wording (:data:`WORDINGS`): ``plain``, or ``distractor`` — a sentence that
uses option words in another sense and says nothing about the rule: in threshold items the words of several options (step names, direction
words); in interval items a note from another unit quoting a WRONG option's label; in selection / comparison items the name of a WRONG
candidate or item; in conversion items a faded sign naming a WRONG option — the answer of a reader who skips the unit conversion whenever that
differs from the key (lures, as the arm's "(up is plus)").

**Labels** are exact by construction (:class:`decimal.Decimal` arithmetic) and re-derived independently from the rendered texts by
``scripts/v09_n9g_selfcheck.py`` (its own number parser, unit table and rule logic over :class:`fractions.Fraction`).

**Scoring** (the Spark chain): the bundle's runner under serve's conditions — the default key, single order seed 1 (the deployed order; the
rule's reading) and seed 2 (flips, reported) — ``scripts/v08_bench_score.py --set N9g``. Accuracy = the top option (ties to the earlier key
of the record) is the key. Cluster = the record (one question per record).

Sealed for training and never sent anywhere (:mod:`robo_jev.data.sealed`): ``v09-n9g`` / ``v09n9g-`` / ``artifacts/v09/n9g/N9g.jsonl``.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

REPO = Path(__file__).resolve().parents[3]
ARTIFACTS = REPO / "artifacts"

N9G_SET, N9G_PREFIX = "v09-n9g", "v09n9g-"
V09_DIR = ARTIFACTS / "v09"
N9G_DIR = V09_DIR / "n9g"
N9G_FILE = N9G_DIR / "N9g.jsonl"
N9G_FROZEN = N9G_DIR / "frozen-before-scoring.json"
#: ``scripts/v09_gates_build.py recheck-v09`` — N9g and the v0.8 gates against CN's training input (train-v09n.jsonl; counts and hashes).
OVERLAP_V09 = V09_DIR / "overlap-v09.json"
#: The one question of every N9g record.
N9G_QUESTION = "q"
FAMILIES = ("threshold", "interval", "selection", "comparison", "conversion")
#: The v0.9 generator's stated domains (progress.md, the v0.9 rule) and five outside that list.
SEEN_DOMAINS = ("distance", "speed", "angle", "time", "mass", "temperature", "percent")
UNSEEN_DOMAINS = ("pressure", "voltage", "volume", "currency", "current")
DOMAINS = SEEN_DOMAINS + UNSEEN_DOMAINS
FORMATS = ("plain", "signed", "unicode", "fixed2", "int", "bare", "thousands")
WORDINGS = ("plain", "distractor")
#: The scored readings: the deployed single order (seed 1; P-num2) and a second order seed (flips; reported).
READINGS = ("single-s1", "single-s2")
#: The 26-option opt-in of :mod:`robo_jev.decisions.records` (N9g questions have at most 20 options).
OPTION_RANGE = {"version": "options-26-1", "max": 26}
SEED = 20261017


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def rng(*parts: Any) -> random.Random:
    """A seeded generator keyed by `parts` (sha256 of their joined text)."""
    digest = hashlib.sha256("|".join(str(p) for p in parts).encode("utf-8")).digest()
    return random.Random(int.from_bytes(digest[:8], "big"))


def check_frozen(directory: str | Path = N9G_DIR) -> list[str]:
    """Problems with the frozen N9g files (empty = unchanged since ``frozen-before-scoring.json`` was written)."""
    directory = Path(directory)
    frozen_path = directory / "frozen-before-scoring.json"
    if not frozen_path.is_file():
        return [f"missing {frozen_path.name} in {directory}"]
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    problems = []
    files = frozen.get("files") or {}
    if "N9g.jsonl" not in files:
        problems.append("frozen-before-scoring.json does not name N9g.jsonl")
    for name, want in files.items():
        path = directory / name
        if not path.is_file():
            problems.append(f"missing {name}")
        elif sha256_file(path) != want:
            problems.append(f"{name} changed after it was frozen")
    return problems


# --------------------------------------------------------------------------
# Readings (pure; the verdict and the chain's glance use them)
# --------------------------------------------------------------------------


def top_option(probabilities: Mapping[str, Any], order: Sequence[str]) -> str | None:
    """The most likely option; ties go to the earlier key of `order` (the record's criteria order)."""
    best, value = None, float("-inf")
    for key in order:
        p = float(probabilities.get(key, 0.0)) if isinstance(probabilities, Mapping) else 0.0
        if p > value:
            best, value = key, p
    return best


def n9g_rows(records: Sequence[Mapping[str, Any]], answers: Mapping[str, Any], *, reading: str = READINGS[0],
             second: str | None = READINGS[1]) -> list[dict[str, Any]]:
    """One row per answered N9g question of one model: the design fields, ``correct`` (top = key), ``p_key``, ``confidence`` (top
    probability) and ``flip`` (the top differs between `reading` and `second`; None without the second reading). `answers` is a bench score
    file's ``answers`` ({record id: {question: {reading: {"choice", "probabilities", …}}}})."""
    rows = []
    for record in records:
        rid = str(record["id"])
        meta = record.get("meta") or {}
        question = record["request"]["questions"][N9G_QUESTION]
        order = [str(k) for k in question["criteria"]]
        got = ((answers.get(rid) or {}).get(N9G_QUESTION)) or {}
        first = got.get(reading)
        if not isinstance(first, Mapping) or not isinstance(first.get("probabilities"), Mapping):
            continue
        probs = {str(k): float(v) for k, v in first["probabilities"].items()}
        top = top_option(probs, order)
        other = got.get(second) if second else None
        flip = None
        if isinstance(other, Mapping) and isinstance(other.get("probabilities"), Mapping):
            flip = float(top_option(other["probabilities"], order) != top)
        key = str(meta["correct"])
        rows.append({"id": rid, "family": meta.get("family"), "template": meta.get("template"), "domain": meta.get("domain"),
                     "domain_seen": bool(meta.get("domain_seen")), "format": meta.get("format"), "wording": meta.get("wording"),
                     "near": meta.get("near"), "n": len(order), "top": top, "correct": float(top == key), "p_key": float(probs.get(key, 0.0)),
                     "confidence": float(max(probs.values())) if probs else 0.0, "flip": flip})
    return rows


def seed_mean_rows(first: Sequence[Mapping[str, Any]], second: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """An arm's rows: per question both seeds answered, the mean of ``correct``, ``p_key``, ``confidence`` and ``flip`` (over the defined)."""
    other = {r["id"]: r for r in second}
    out = []
    for r in first:
        o = other.get(r["id"])
        if o is None:
            continue
        row = {k: r[k] for k in ("id", "family", "template", "domain", "domain_seen", "format", "wording", "near", "n")}
        row["top"] = None
        for key in ("correct", "p_key", "confidence"):
            row[key] = (float(r[key]) + float(o[key])) / 2.0
        flips = [v for v in (r["flip"], o["flip"]) if v is not None]
        row["flip"] = sum(flips) / len(flips) if flips else None
        out.append(row)
    return out


def _mean(values: Iterable[float | None]) -> dict[str, Any]:
    kept = [float(v) for v in values if v is not None]
    return {"value": sum(kept) / len(kept) if kept else None, "n": len(kept)}


def ece(confidence: Sequence[float], correct: Sequence[float], bins: int = 10) -> float | None:
    """Top-label expected calibration error over equal-width confidence bins."""
    if not confidence:
        return None
    by_bin: dict[int, list[tuple[float, float]]] = defaultdict(list)
    for c, a in zip(confidence, correct):
        by_bin[min(int(float(c) * bins), bins - 1)].append((float(c), float(a)))
    total = len(confidence)
    return float(sum(len(v) / total * abs(sum(a for _, a in v) / len(v) - sum(c for c, _ in v) / len(v)) for v in by_bin.values()))


def n9g_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Accuracy (and mean p(key), flips) pooled and per family, domain, seen / unseen domain, format, wording, near-boundary kind and option
    count; top-label ECE pooled and per family (a seed-mean row's confidence is the seeds' mean — reported only)."""
    def block(chosen: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        return {"accuracy": _mean(r["correct"] for r in chosen), "p_key": _mean(r["p_key"] for r in chosen), "flips": _mean(r["flip"] for r in chosen)}

    out: dict[str, Any] = {"total": block(rows), "ece": ece([r["confidence"] for r in rows], [r["correct"] for r in rows])}
    for field, name in (("family", "by_family"), ("domain", "by_domain"), ("format", "by_format"), ("wording", "by_wording"), ("near", "by_near")):
        groups: dict[str, list] = defaultdict(list)
        for r in rows:
            groups[str(r.get(field))].append(r)
        out[name] = {g: block(v) for g, v in sorted(groups.items())}
    seen = defaultdict(list)
    for r in rows:
        seen["seen domain" if r["domain_seen"] else "unseen domain"].append(r)
    out["by_domain_group"] = {g: block(v) for g, v in sorted(seen.items())}
    family_seen = defaultdict(list)
    for r in rows:
        family_seen[f"{r['family']} / {'seen' if r['domain_seen'] else 'unseen'}"].append(r)
    out["by_family_domain_group"] = {g: block(v) for g, v in sorted(family_seen.items())}
    sizes = defaultdict(list)
    for r in rows:
        n = int(r["n"])
        sizes["2-5" if n <= 5 else "6-9" if n <= 9 else "10-14" if n <= 14 else "15-20"].append(r)
    out["by_options"] = {g: block(v) for g, v in sorted(sizes.items())}
    out["ece_by_family"] = {f: ece([r["confidence"] for r in rows if r["family"] == f], [r["correct"] for r in rows if r["family"] == f])
                            for f in FAMILIES if any(r["family"] == f for r in rows)}
    return out


def design_counts(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Counts of the built set per family, domain, format, wording, near kind, option count and template."""
    metas = [r.get("meta") or {} for r in records]
    return {"questions": len(records), "by_family": dict(Counter(m.get("family") for m in metas)),
            "by_domain": dict(Counter(m.get("domain") for m in metas)),
            "by_domain_group": dict(Counter("unseen" if m.get("domain") in UNSEEN_DOMAINS else "seen" for m in metas)),
            "by_format": dict(Counter(m.get("format") for m in metas)), "by_wording": dict(Counter(m.get("wording") for m in metas)),
            "by_near": dict(Counter(str(m.get("near")) for m in metas)),
            "by_options": dict(sorted(Counter(len(r["request"]["questions"][N9G_QUESTION]["criteria"]) for r in records).items())),
            "by_template": dict(sorted(Counter(m.get("template") for m in metas).items())),
            "distractor_names_key": {f: {"distractor_items": sum(1 for m in metas if m.get("family") == f and m.get("wording") == "distractor"),
                                         "names_the_key": sum(1 for m in metas if m.get("family") == f and m.get("distractor_names_key"))}
                                     for f in FAMILIES}}


# --------------------------------------------------------------------------
# Build — numbers (exact: decimal.Decimal) and their written forms
# --------------------------------------------------------------------------

from decimal import ROUND_HALF_UP, Decimal  # noqa: E402

D = Decimal
MINUS = "−"
#: Number formats valid per family (the judged numbers of a question share one format; ``mixed`` = comparison / selection numbers written
#: with their own natural decimals, e.g. 9.9 next to 9.11).
FAMILY_FORMATS = {
    "threshold": ("plain", "signed", "unicode", "fixed2", "int", "bare"),
    "interval": ("plain", "fixed2", "int", "bare", "unicode", "thousands"),
    "selection": ("plain", "fixed2", "int", "mixed"),
    "comparison": ("plain", "fixed2", "int", "bare", "unicode", "mixed", "thousands"),
    "conversion": ("plain", "fixed2", "int", "bare", "thousands"),
}
FORMATS = FORMATS + ("mixed",)


def dtext(value: Decimal) -> str:
    """A number as a rule writes it: plain decimal, no trailing zeros, no exponent, ASCII minus."""
    s = format(Decimal(value), "f")
    if "." in s:
        s = s.rstrip("0").rstrip(".")
    return "0" if s in ("", "-0") else s


def places_of(value: Decimal) -> int:
    s = dtext(value)
    return len(s.split(".")[1]) if "." in s else 0


def quant(value: Decimal, places: int) -> Decimal:
    return Decimal(value).quantize(Decimal(1).scaleb(-places), rounding=ROUND_HALF_UP)


def valid_format(fmt: str, values: Sequence[Decimal]) -> bool:
    """Whether every value can be written in `fmt` exactly."""
    values = [Decimal(v) for v in values]
    if fmt == "int":
        return all(v == v.to_integral_value() for v in values)
    if fmt == "fixed2":
        return all(places_of(v) <= 2 for v in values)
    if fmt == "bare":
        return all(0 < abs(v) < 1 for v in values)
    if fmt == "thousands":
        return all(abs(v) >= 1000 for v in values)
    if fmt == "unicode":
        return any(v < 0 for v in values)
    if fmt == "mixed":
        return len({places_of(v) for v in values}) >= 2
    return True


def write(value: Decimal, fmt: str, *, places: int | None = None, signed_family: bool = False) -> str:
    """`value` written in `fmt` (exact; raises if it cannot be). ``plain`` uses `places` decimals (default: the value's own), ``signed`` and
    ``unicode`` write the sign of every non-zero number in a signed family (``unicode`` uses U+2212 for minus; outside a signed family
    ``unicode`` only changes the minus), ``fixed2`` two decimals, ``int`` no decimals, ``bare`` no leading zero, ``thousands`` comma groups,
    ``mixed`` the value's own decimals."""
    value = Decimal(value)
    if not valid_format(fmt, [value]) and fmt not in ("unicode", "mixed"):
        raise ValueError(f"{value} cannot be written as {fmt}")
    neg = value < 0
    mag = -value if neg else value
    own = places_of(value)
    if fmt == "fixed2":
        body = format(quant(mag, 2), "f")
    elif fmt == "int":
        body = format(quant(mag, 0), "f")
    elif fmt == "bare":
        body = dtext(mag)[1:] if dtext(mag).startswith("0.") else dtext(mag)
    elif fmt == "thousands":
        p = own if places is None else places
        body = f"{quant(mag, p):,.{p}f}"
    elif fmt == "mixed":
        body = dtext(mag)
    else:
        p = own if places is None else max(places, own)
        body = format(quant(mag, p), "f")
    if quant(Decimal(body.replace(",", "")), own) != quant(mag, own):
        raise ValueError(f"{value} written as {body!r} does not read back")
    if fmt == "unicode":
        sign = MINUS if neg else ("+" if signed_family and value != 0 else "")
    elif fmt == "signed":
        sign = "-" if neg else ("+" if value != 0 else "")
    else:
        sign = "-" if neg else ""
    return sign + body


def pick_format(r: random.Random, family: str, values: Sequence[Decimal], exclude: Sequence[str] = ()) -> str:
    """A format valid for every judged value: ``plain`` / the explicit-sign forms / ``fixed2`` are always considered (when valid); ``int``,
    ``bare``, ``thousands``, ``mixed`` and ``unicode`` only where the numbers allow them — so their counts follow the values. `exclude`
    leaves out forms that read unnaturally for a unit (money with one decimal, cents with decimals)."""
    pool = [f for f in FAMILY_FORMATS[family] if valid_format(f, values) and f not in exclude]
    special = [f for f in pool if f in ("int", "bare", "thousands", "mixed", "unicode")]
    if special and r.random() < 0.5:
        return r.choice(special)
    common = [f for f in pool if f not in special] or pool
    return r.choice(common)


def unit_text(number: str, unit: str) -> str:
    """A number with its unit: currency before (``$12.50``, ``-$3.40``), %, °, °C after without / with a space as usual."""
    if unit == "$":
        return (number[0] + "$" + number[1:]) if number[:1] in ("-", "+", MINUS) else "$" + number
    if unit in ("%", "°"):
        return number + unit
    return f"{number} {unit}"


def _near_value(r: random.Random, bound: Decimal, *, inside_above: bool, kind: str, places: int, lo: Decimal | None,
                hi: Decimal | None) -> tuple[Decimal, int] | None:
    """A value near `bound` on the band's side (above it when `inside_above`), strictly inside (lo, hi) and never equal to a bound: ``rel`` —
    within 5 % of the bound; ``abs`` — within 0.05 of it (needs 2 decimals). Returns (value, places) or None."""
    bound = Decimal(bound)
    for _ in range(40):
        if kind == "abs":
            p = max(places, 2)
            d = Decimal(r.choice((1, 2, 3, 4, 5))) / 100
        else:
            p = places
            u = Decimal(r.randint(8, 50)) / 1000
            d = quant(bound * u, p)
            if d == 0:
                p = min(p + 1, 3)
                d = quant(bound * u, p)
            if d == 0 or d > bound * Decimal("0.05"):
                continue
        value = quant(bound + d if inside_above else bound - d, p)
        if value == bound or value <= 0:
            continue
        if kind == "rel" and abs(value - bound) > bound * Decimal("0.05"):
            continue
        if kind == "abs" and abs(value - bound) > Decimal("0.05"):
            continue
        if (lo is not None and value <= lo) or (hi is not None and value >= hi):
            continue
        return value, p
    return None


def _inside_value(r: random.Random, lo: Decimal, hi: Decimal, places: int) -> Decimal | None:
    """A value strictly inside (lo, hi), away from both ends (the middle 80 %), with `places` decimals."""
    lo, hi = Decimal(lo), Decimal(hi)
    span = hi - lo
    for _ in range(40):
        x = lo + span * (Decimal(r.randint(100, 900)) / 1000)
        value = quant(x, places)
        if lo < value < hi:
            return value
    return None


def band_value(r: random.Random, lo: Decimal, hi: Decimal | None, *, near: str | None, edges: Sequence[str], places: int,
               top_factor: Decimal = Decimal("2.5")) -> tuple[Decimal, int, str | None]:
    """A positive magnitude in the band [lo, hi) (hi None = open above; lo 0 = the bottom band): near one of `edges` (``lo`` / ``hi``) when
    `near` is ``rel`` / ``abs``, else inside. Never equal to lo or hi. Returns (value, places, near kind actually used)."""
    lo = Decimal(lo)
    top = hi if hi is not None else quant(lo * top_factor, places) if lo > 0 else Decimal(1)
    if near:
        for edge in r.sample(list(edges), len(edges)):
            if edge == "lo" and lo == 0:
                continue
            found = _near_value(r, lo if edge == "lo" else Decimal(hi), inside_above=(edge == "lo"), kind=near, places=places,
                                lo=lo if edge == "hi" else None, hi=hi if edge == "lo" else None)
            if found:
                value, p = found
                if value > lo and (hi is None or value < hi):
                    return value, p, near
    low = lo if lo > 0 else top * Decimal("0.05")
    for p in (places, places + 1, places + 2):
        value = _inside_value(r, low if lo > 0 else Decimal(0), top, p)
        if value is not None and value > lo and (hi is None or value < hi) and value > 0:
            return value, p, None
    raise ValueError(f"no value in [{lo}, {hi}) at {places} decimals")


def value_near(r: random.Random, bound: Decimal, side: int, kind: str, places: int) -> tuple[Decimal, int] | None:
    """A value next to `bound` on `side` (+1 above, −1 below): ``rel`` — within 5 % of |bound|; ``abs`` — within 0.05 (2 decimals at least).
    Never equal to the bound. Returns (value, places) or None."""
    bound = Decimal(bound)
    for _ in range(40):
        if kind == "abs":
            p = max(places, 2)
            d = Decimal(r.choice((1, 2, 3, 4, 5))) / 100
        else:
            p = places
            d = quant(abs(bound) * Decimal(r.randint(8, 50)) / 1000, p)
            while d == 0 and p < 3:
                p += 1
                d = quant(abs(bound) * Decimal(r.randint(8, 50)) / 1000, p)
            if d == 0 or d > abs(bound) * Decimal("0.05"):
                continue
        value = quant(bound + side * d, p)
        if value == bound or (value - bound) * side <= 0:
            continue
        if kind == "rel" and abs(value - bound) > abs(bound) * Decimal("0.05"):
            continue
        if kind == "abs" and abs(value - bound) > Decimal("0.05"):
            continue
        return value, p
    return None


def value_inside(r: random.Random, lo: Decimal, hi: Decimal, places: int) -> tuple[Decimal, int] | None:
    """A value strictly inside (lo, hi), in its middle 80 %; more decimals if the band is too narrow for `places`."""
    lo, hi = Decimal(lo), Decimal(hi)
    for p in (places, places + 1, places + 2):
        for _ in range(30):
            value = quant(lo + (hi - lo) * Decimal(r.randint(100, 900)) / 1000, p)
            if lo < value < hi:
                return value, p
    return None


def region_value(r: random.Random, lo: Decimal | None, hi: Decimal | None, *, near: str | None, places: int,
                 outer: tuple[Decimal, Decimal]) -> tuple[Decimal, int, str | None]:
    """A value strictly inside the region (lo, hi) (None = open; `outer` bounds an open side): near one of its finite ends when `near` is set
    (and possible), else inside. Returns (value, places, near kind used)."""
    lo_x = Decimal(outer[0]) if lo is None else Decimal(lo)
    hi_x = Decimal(outer[1]) if hi is None else Decimal(hi)
    if near:
        ends = [(lo, +1)] if lo is not None else []
        ends += [(hi, -1)] if hi is not None else []
        r.shuffle(ends)
        for bound, side in ends:
            found = value_near(r, bound, side, near, places)
            if found and lo_x < found[0] < hi_x and (lo is None or found[0] > lo) and (hi is None or found[0] < hi):
                return found[0], found[1], near
    found = value_inside(r, lo_x, hi_x, places)
    if found is None:
        raise ValueError(f"no value inside ({lo}, {hi})")
    return found[0], found[1], None


def ladder(text: str) -> list[Decimal]:
    return [Decimal(x) for x in text.split()]


# --------------------------------------------------------------------------
# Family 1 — threshold / step: a signed reading, a sign convention, ordered thresholds on its size
# --------------------------------------------------------------------------

SIZE_WORDS = {"v1": ("trim", "light", "medium", "heavy", "maximum"), "v2": ("slight", "gentle", "moderate", "strong", "full"),
              "v3": ("minor", "modest", "major", "severe", "extreme")}
SIZE_DISTRACTORS = {"v1": "A heavy toolbox sits under the light beside the trim panel.",
                    "v2": "Earlier the night shift gave the cabinet door a slight push; a strong lamp lights the full rack.",
                    "v3": "The list of minor repairs is pinned above the shelf of major spare parts."}

THRESHOLD_SCENARIOS: tuple[dict[str, Any], ...] = (
    {"name": "bath", "domain": "temperature", "unit": "°C", "role": "You are the lab robot that tends a water bath for sample tubes.",
     "reading": "The bath's temperature error reads {v}.", "noun": "temperature error",
     "sign": "A positive error means the bath is warmer than its setpoint, so it needs cooling; a negative error means it is colder, so it needs heating.",
     "pos": "cool", "neg": "heat", "pos_verb": "Cool the bath", "neg_verb": "Heat the bath", "hold": "hold", "hold_text": "Leave the bath's temperature control as it is.",
     "ladders": ("0.2 0.5 1 2 4", "0.3 0.8 1.5 3 6", "0.25 0.6 1.2 2.5 5"), "distractor": "Heat-proof gloves hang beside the cool-air vent."},
    {"name": "line", "domain": "pressure", "unit": "kPa", "role": "You are the robot that watches the nitrogen line feeding a glovebox.",
     "reading": "The line's pressure deviation reads {v}.", "noun": "pressure deviation",
     "sign": "A positive deviation means the line is above its set pressure and should be vented; a negative one means it is below and needs the booster pump.",
     "pos": "vent", "neg": "pump", "pos_verb": "Vent the line", "neg_verb": "Run the booster pump", "hold": "no_change",
     "hold_text": "Change nothing on the line.", "ladders": ("1 3 8 20 50", "2 5 10 25 60", "1.5 4 9 18 40"),
     "distractor": "A pump cart stands under the vent hood by the door."},
    {"name": "rail", "domain": "voltage", "unit": "V", "role": "You are the test robot trimming a bench power supply for a sensor board.",
     "reading": "The supply rail's deviation from nominal reads {v}.", "noun": "deviation",
     "sign": "A positive deviation means the rail is above nominal and must be lowered; a negative one means it is below nominal and must be raised.",
     "pos": "lower", "neg": "raise", "pos_verb": "Lower the rail", "neg_verb": "Raise the rail", "hold": "keep",
     "hold_text": "Keep the supply setting.", "ladders": ("0.05 0.1 0.25 0.5 1", "0.02 0.08 0.2 0.4 0.9", "0.1 0.3 0.6 1.2 2.4"),
     "distractor": "Spare fuses are in the lower drawer; the raise-and-lock lever on the bench is painted yellow."},
    {"name": "heading", "domain": "angle", "unit": "°", "role": "You are the steering check of a warehouse robot following a painted lane.",
     "reading": "The heading error reads {v}.", "noun": "heading error",
     "sign": "A positive error means the robot points to the right of the lane, so it must turn left; a negative error means it points to the left, so it must turn right.",
     "pos": "turn_left", "neg": "turn_right", "pos_verb": "Turn left", "neg_verb": "Turn right", "hold": "straight",
     "hold_text": "Keep driving straight.", "ladders": ("1 3 8 15 30", "0.5 2 5 12 25", "2 4 10 20 45"),
     "distractor": "A worker on the left carries a box to the right-hand rack."},
    {"name": "dock", "domain": "distance", "unit": "mm", "role": "You are the docking controller lining a mobile robot up with its charging pin.",
     "reading": "The lateral offset from the pin reads {v}.", "noun": "offset",
     "sign": "A positive offset means the robot sits right of the pin and must shift left; a negative offset means it sits left of the pin and must shift right.",
     "pos": "shift_left", "neg": "shift_right", "pos_verb": "Shift left", "neg_verb": "Shift right", "hold": "stay_put",
     "hold_text": "Stay put and dock.", "ladders": ("0.5 1.5 4 10 25", "0.3 1.2 3.5 9 22", "0.4 1 2.5 6 15"),
     "distractor": "The left bumper has a scuff; the right camera was wiped this morning."},
    {"name": "dose", "domain": "mass", "unit": "g", "role": "You are the powder-dosing robot at a balance in a formulation lab.",
     "reading": "Dosed mass minus target reads {v}.", "noun": "dosing error",
     "sign": "A positive error means too much powder is in the cup, so some must be removed; a negative error means too little, so more must be added.",
     "pos": "remove", "neg": "add", "pos_verb": "Remove powder", "neg_verb": "Add powder", "hold": "accept",
     "hold_text": "Accept the dose as it is.", "ladders": ("0.1 0.5 2 5 12", "0.2 0.6 1.5 4 10", "0.05 0.25 1 3 8"),
     "distractor": "A remove-before-use cap and an add-on label lie beside the balance."},
    {"name": "tank", "domain": "volume", "unit": "mL", "role": "You are the robot that keeps a rinse tank at its working volume.",
     "reading": "Tank volume minus target reads {v}.", "noun": "volume error",
     "sign": "A positive error means the tank holds more than the target and must be drained; a negative one means it holds less and must be filled.",
     "pos": "drain", "neg": "fill", "pos_verb": "Drain the tank", "neg_verb": "Fill the tank", "hold": "leave",
     "hold_text": "Leave the tank alone.", "ladders": ("5 20 50 120 300", "10 25 60 150 400", "2 10 40 100 250"),
     "distractor": "The floor drain was cleaned; the fill hose is coiled on its hook."},
    {"name": "skew", "domain": "time", "unit": "ms", "role": "You are the timing supervisor of two robot arms that must hand over parts in sync.",
     "reading": "Arm B's clock offset against arm A reads {v}.", "noun": "clock offset",
     "sign": "A positive offset means arm B runs ahead, so its next cycle must be delayed; a negative offset means it runs behind, so its next cycle must be advanced.",
     "pos": "delay", "neg": "advance", "pos_verb": "Delay arm B", "neg_verb": "Advance arm B", "hold": "in_sync",
     "hold_text": "Leave both arms as they are.", "ladders": ("1 4 10 25 60", "2 5 15 40 100", "0.5 2 6 15 50"),
     "distractor": "A delay notice for the advance shipment is pinned to the board."},
    {"name": "belt", "domain": "speed", "unit": "m/s", "role": "You are the robot that keeps a sorting belt at its target speed.",
     "reading": "Belt speed minus target reads {v}.", "noun": "speed error",
     "sign": "A positive error means the belt runs too fast and must slow down; a negative error means it runs too slow and must speed up.",
     "pos": "slow", "neg": "speed_up", "pos_verb": "Slow the belt", "neg_verb": "Speed the belt up", "hold": "steady",
     "hold_text": "Hold the belt's speed.", "ladders": ("0.01 0.03 0.08 0.15 0.3", "0.02 0.05 0.1 0.2 0.5", "0.005 0.02 0.05 0.12 0.25"),
     "distractor": "A slow-moving forklift passes the speed sign at the aisle end."},
    {"name": "grip", "domain": "current", "unit": "A", "role": "You are the gripper supervisor of a robot holding glass vials.",
     "reading": "Gripper motor current minus its holding target reads {v}.", "noun": "current error",
     "sign": "A positive error means the gripper squeezes harder than it should and must loosen; a negative error means it squeezes too little and must tighten.",
     "pos": "loosen", "neg": "tighten", "pos_verb": "Loosen the grip", "neg_verb": "Tighten the grip", "hold": "keep_grip",
     "hold_text": "Keep the grip as it is.", "ladders": ("0.05 0.15 0.4 0.8 1.5", "0.1 0.25 0.5 1 2", "0.02 0.1 0.3 0.6 1.2"),
     "distractor": "A chart on the wall shows how to loosen and tighten the vise bolts."},
    {"name": "humid", "domain": "percent", "unit": "%", "role": "You are the climate robot of a seed-storage room.",
     "reading": "Relative humidity minus target reads {v} (percentage points).", "noun": "humidity error",
     "sign": "A positive error means the air is too damp, so the robot must dry it; a negative error means it is too dry, so the robot must humidify it.",
     "pos": "dry", "neg": "humidify", "pos_verb": "Dry the air", "neg_verb": "Humidify the air", "hold": "steady",
     "hold_text": "Leave the climate unit as it is.", "ladders": ("1 3 6 10 20", "0.5 2 5 8 15", "2 4 8 12 25"),
     "distractor": "A dry-erase board lists the humidifier's service dates."},
)
THRESHOLD_STYLES = ("prose", "table", "firstfit", "chain")


def _a(word: str) -> str:
    """The indefinite article for a step name ("an extreme step")."""
    return ("an " if word[:1].lower() in "aeiou" else "a ") + word


def _q(t: Decimal, unit: str) -> str:
    """A rule's bound with its unit (money with two decimals unless whole: $12.50, $80)."""
    if unit == "$" and Decimal(t) != Decimal(t).to_integral_value():
        return unit_text(format(quant(Decimal(t), 2), "f"), unit)
    return unit_text(dtext(t), unit)


def threshold_rule(style: str, sc: Mapping[str, Any], sizes: Sequence[str], thresholds: Sequence[Decimal]) -> tuple[list[str], str]:
    """(state lines, question text) of a threshold rule in one of :data:`THRESHOLD_STYLES`."""
    u, noun, K = sc["unit"], sc["noun"], len(thresholds)
    if style == "prose":
        parts = [f"no correction while it is under {_q(thresholds[0], u)}"]
        parts += [f"{_a(sizes[k])} step from {_q(thresholds[k], u)} up to {_q(thresholds[k + 1], u)}" for k in range(K - 1)]
        parts.append(f"{_a(sizes[K - 1])} step at {_q(thresholds[K - 1], u)} or more")
        return [], (f"{sc['sign']} Choose the step from the size of the {noun}, whatever its sign: " + "; ".join(parts)
                    + ". Which command should the robot send?")
    if style == "table":
        lines = [f"Correction table (size of the {noun}, either sign):", f"- {_q(thresholds[-1], u)} or more: {sizes[K - 1]} step"]
        lines += [f"- {_q(thresholds[k], u)} to under {_q(thresholds[k + 1], u)}: {sizes[k]} step" for k in range(K - 2, -1, -1)]
        lines += [f"- under {_q(thresholds[0], u)}: no correction", sc["sign"]]
        return lines, "Which command fits the reading?"
    if style == "firstfit":
        lines = [f"Step rule (go down the list and use the first line that fits the size of the {noun}):"]
        lines += [f"- at least {_q(thresholds[k], u)} -> {sizes[k]} step" for k in range(K - 1, -1, -1)]
        lines += ["- anything smaller -> no correction", sc["sign"]]
        return lines, "Which correction should the robot make now?"
    if style == "chain":
        parts = [f"{_a(sizes[K - 1])} step if the size of the {noun} is at least {_q(thresholds[K - 1], u)}"]
        parts += [f"otherwise {_a(sizes[k])} step if it is at least {_q(thresholds[k], u)}" for k in range(K - 2, -1, -1)]
        parts.append("otherwise no correction")
        return [], f"{sc['sign']} Use " + "; ".join(parts) + ". Which option is right?"
    raise ValueError(style)


def make_threshold(r: random.Random, sc: Mapping[str, Any], *, style: str, k_count: int, sign: int, step: int, near: str | None,
                   wording: str) -> dict[str, Any]:
    """One threshold question: band `step` (0 = no correction, 1…K = the K steps) and direction `sign` (+1 / −1)."""
    vocab = r.choice(sorted(SIZE_WORDS))
    sizes = SIZE_WORDS[vocab][:k_count]
    thresholds = ladder(r.choice(sc["ladders"]))[:k_count]
    base = max(places_of(t) for t in thresholds)
    places = min(base + r.choice((0, 0, 1)), 3)
    lo = Decimal(0) if step == 0 else thresholds[step - 1]
    hi = thresholds[step] if step < k_count else None
    magnitude, places, used = region_value(r, lo if step > 0 else None, hi, near=near, places=places,
                                           outer=(Decimal(0), (thresholds[-1] * Decimal("2.5")) if hi is None else hi))
    if step == 0 and magnitude <= 0:
        raise ValueError("a zero reading")
    value = magnitude * sign
    fmt = pick_format(r, "threshold", [value])
    written = write(value, fmt, places=places, signed_family=True)
    if fmt == "plain" and "." not in written:
        fmt = "int"
    keys = {"hold": sc["hold"], "pos": [f"{sc['pos']}_{s}" for s in sizes], "neg": [f"{sc['neg']}_{s}" for s in sizes]}
    options = [(sc["hold"], "No correction: " + sc["hold_text"][0].lower() + sc["hold_text"][1:])]
    for k, s in enumerate(sizes):
        options.append((keys["pos"][k], f"{sc['pos_verb']}: {s} step."))
        options.append((keys["neg"][k], f"{sc['neg_verb']}: {s} step."))
    correct = sc["hold"] if step == 0 else (keys["pos"] if sign > 0 else keys["neg"])[step - 1]
    rule_lines, question = threshold_rule(style, sc, sizes, thresholds)
    lines = [sc["role"]]
    aside = [SIZE_DISTRACTORS[vocab], sc["distractor"]] if wording == "distractor" else []
    lines += aside + rule_lines + [sc["reading"].format(v=unit_text(written, sc["unit"]))]
    band = "hold" if step == 0 else f"{'+' if sign > 0 else '-'}{sizes[step - 1]}"
    spec = {"kind": "threshold", "unit": sc["unit"], "value_text": written, "thresholds_text": [dtext(t) for t in thresholds], "keys": keys,
            "pos_action": sc["pos"], "neg_action": sc["neg"], "style": style, "vocab": vocab}
    return {"family": "threshold", "template": f"threshold-{sc['name']}-{style}", "domain": sc["domain"], "domains": [sc["domain"]],
            "format": fmt, "near": used, "band": band, "state": "\n".join(lines), "instructions": question, "options": options, "correct": correct,
            "spec": spec, "aside": " ".join(aside)}


# --------------------------------------------------------------------------
# Family 2 — interval membership: one band, a target ± tolerance, or named windows with gaps
# --------------------------------------------------------------------------

INTERVAL_SCENARIOS: tuple[dict[str, Any], ...] = (
    {"name": "cell", "domain": "voltage", "unit": "V", "role": "You are the battery-swap robot at a charging station.",
     "reading": "The pack's resting voltage reads {v}.", "kinds": ("band", "windows"),
     "bands": ("3.2 4.15", "11.6 12.9", "22.8 25.6", "46.5 52.5"),
     "band_text": "A pack goes back into service only if its resting voltage is between {lo} and {hi}.",
     "keys": (("too_low", "Below the range: send the pack to the recovery charger."), ("accept", "Inside the range: put the pack back into service."),
              ("too_high", "Above the range: let the pack rest before any use.")),
     "windows": (("dock_a", "Dock A", "11.5 12.8"), ("dock_b", "Dock B", "23.0 25.6"), ("dock_c", "Dock C", "46.0 51.2"), ("dock_d", "Dock D", "70.0 76.5")),
     "window_text": "Each charging dock takes only packs whose resting voltage lies in its window:", "none": ("manual_check", "None of the docks: hand the pack to a technician."),
     "window_question": "Which dock should the pack go to?", "distractor": "The accept stamp lies on the low shelf next to Dock B's high stool."},
    {"name": "tire", "domain": "pressure", "unit": "kPa", "role": "You are the inspection robot at the tire check of an AGV fleet.",
     "reading": "The front-left tire reads {v}.", "kinds": ("band", "tolerance"), "bands": ("210 240", "180 205", "300 340"),
     "band_text": "The tire passes if its pressure is between {lo} and {hi}.",
     "keys": (("inflate", "Below the range: inflate the tire."), ("pass", "Inside the range: the tire passes."),
              ("deflate", "Above the range: let some air out.")),
     "tolerances": (("2.4 0.15", "bar"), ("3.1 0.2", "bar"), ("1.8 0.1", "bar")),
     "tolerance_text": "The tire passes if its pressure is {c} within plus or minus {t}.",
     "distractor": "Someone left an inflate-and-pass checklist on the deflated spare wheel."},
    {"name": "fridge", "domain": "temperature", "unit": "°C", "role": "You are the pharmacy robot that checks storage fridges and freezers.",
     "reading": "The unit's air temperature reads {v}.", "kinds": ("band", "windows"), "bands": ("2 8", "-25 -15", "15 25"),
     "band_text": "Stock may stay in this unit only while its air temperature is between {lo} and {hi}.",
     "keys": (("too_cold", "Below the range: move the stock and report a cold fault."), ("in_range", "Inside the range: leave the stock where it is."),
              ("too_warm", "Above the range: move the stock and report a warm fault.")),
     "windows": (("zone_freezer", "Freezer zone", "-30 -18"), ("zone_chilled", "Chilled zone", "0 4"), ("zone_cool", "Cool zone", "8 15"),
                 ("zone_room", "Room zone", "18 25")),
     "window_text": "The store's zones are defined by air temperature:", "none": ("no_zone", "None of the zones: report the reading as unclassified."),
     "window_question": "Which zone does this unit belong to?", "distractor": "A cool-blue sticker on the freezer door says the room checks are due."},
    {"name": "humid", "domain": "percent", "unit": "%", "role": "You are the climate robot of an archive room.",
     "reading": "Relative humidity reads {v}.", "kinds": ("band",), "bands": ("40 60", "30 45", "45 55", "35 50"),
     "band_text": "The room is fine while relative humidity is between {lo} and {hi}.",
     "keys": (("humidify", "Below the range: start the humidifier."), ("fine", "Inside the range: change nothing."),
              ("dehumidify", "Above the range: start the dehumidifier.")),
     "distractor": "A note says the fine-paper boxes must never touch the humidifier."},
    {"name": "parcel", "domain": "mass", "unit": "kg", "role": "You are the sorting robot at the end of an inbound parcel line.",
     "reading": "The scale under the parcel reads {v}.", "kinds": ("windows",),
     "windows": (("chute_a", "Chute A", "0.5 2.0"), ("chute_b", "Chute B", "2.5 6.0"), ("chute_c", "Chute C", "7.0 15.0"), ("chute_d", "Chute D", "16.0 30.0")),
     "window_text": "Each chute takes parcels in its mass window:", "none": ("manual_desk", "None of the chutes: send the parcel to the manual desk."),
     "window_question": "Where should the parcel go?", "distractor": "Chute C's light is brighter than the manual desk lamp."},
    {"name": "cycle", "domain": "time", "unit": "s", "role": "You are the cell supervisor timing a screw-driving robot.",
     "reading": "The last cycle took {v}.", "kinds": ("band", "tolerance"), "bands": ("42 48", "8.5 9.5", "115 125"),
     "band_text": "A cycle is normal when it takes between {lo} and {hi}.",
     "keys": (("too_fast", "Below the range: flag a skipped step."), ("normal", "Inside the range: carry on."),
              ("too_slow", "Above the range: flag a stuck step.")), "tolerances": (("45 1.5", "s"), ("12 0.4", "s"), ("90 2.5", "s")),
     "tolerance_text": "A cycle is normal when it takes {c} plus or minus {t}.", "distractor": "A sticker on the driver reads 'normal torque only'."},
    {"name": "fill", "domain": "volume", "unit": "mL", "role": "You are the inspection robot after a bottle-filling machine.",
     "reading": "The bottle's fill volume reads {v}.", "kinds": ("tolerance", "band"), "tolerances": (("500 5", "mL"), ("250 3", "mL"), ("1000 8", "mL")),
     "tolerance_text": "A bottle is in spec if it holds {c} within plus or minus {t}.", "bands": ("745 755", "1495 1510", "330 334"),
     "band_text": "A bottle is in spec if it holds between {lo} and {hi}.",
     "keys": (("underfilled", "Below the range: divert the bottle as underfilled."), ("in_spec", "Inside the range: let the bottle through."),
              ("overfilled", "Above the range: divert the bottle as overfilled.")),
     "distractor": "The in-spec bin is next to the overfilled waste drum."},
    {"name": "quote", "domain": "currency", "unit": "$", "role": "You are the purchasing robot of a small workshop.",
     "reading": "The new quote is {v}.", "kinds": ("band",), "bands": ("12.50 14.00", "80 95", "1250 1400", "3.20 3.80"),
     "band_text": "A quote is accepted automatically when it is between {lo} and {hi}.",
     "keys": (("check_low", "Below the range: check the quote for a mistake."), ("accept", "Inside the range: accept the quote."),
              ("ask_manager", "Above the range: send the quote to the manager.")),
     "distractor": "The manager left a note to check the accept tray before lunch."},
    {"name": "motor", "domain": "current", "unit": "A", "role": "You are the health monitor of a conveyor's drive motor.",
     "reading": "The motor draws {v}.", "kinds": ("band",), "bands": ("0.8 1.6", "2.5 4.0", "0.15 0.35", "5.5 7.2"),
     "band_text": "The drive is healthy while it draws between {lo} and {hi}.",
     "keys": (("underload", "Below the range: check for a slipping belt."), ("healthy", "Inside the range: no action."),
              ("overload", "Above the range: check for a jam.")), "distractor": "The healthy-belt poster hangs over the overload relay."},
    {"name": "conveyor", "domain": "speed", "unit": "m/s", "role": "You are the robot that audits belt speeds in a parcel hub.",
     "reading": "The belt moves at {v}.", "kinds": ("band", "tolerance"), "bands": ("0.45 0.6", "1.2 1.5", "0.25 0.32"),
     "band_text": "A belt is on speed when it runs between {lo} and {hi}.", "tolerances": (("0.5 0.03", "m/s"), ("1.8 0.1", "m/s")),
     "tolerance_text": "A belt is on speed when it runs at {c} within plus or minus {t}.",
     "keys": (("too_slow", "Below the range: raise the drive setting."), ("on_speed", "Inside the range: no change."),
              ("too_fast", "Above the range: lower the drive setting.")), "distractor": "A slow-down sign hangs where the fast lane ends."},
    {"name": "gap", "domain": "distance", "unit": "mm", "role": "You are the gauging robot checking door gaps on a car body.",
     "reading": "The measured gap is {v}.", "kinds": ("tolerance", "band"), "tolerances": (("4.0 0.3", "mm"), ("3.5 0.1", "mm"), ("6.0 0.5", "mm")),
     "tolerance_text": "A gap is good when it is {c} within plus or minus {t}.", "bands": ("3.6 4.4", "11.5 12.5", "24.5 25.5"),
     "band_text": "A gap is good when it is between {lo} and {hi}.",
     "keys": (("too_tight", "Below the range: mark the door as too tight."), ("good", "Inside the range: pass the door."),
              ("too_wide", "Above the range: mark the door as too wide.")), "distractor": "A wide strip of tape marks the good-parts rack."},
    {"name": "tilt", "domain": "angle", "unit": "°", "role": "You are the stability check of a mobile robot carrying a tray of cups.",
     "reading": "The tray's tilt reads {v} (negative means tilted back, positive tilted forward).", "kinds": ("band",),
     "bands": ("-2.0 2.0", "-1.5 1.5", "-5 3", "-3 1"), "band_text": "The tray counts as level while its tilt is between {lo} and {hi}.",
     "keys": (("tilted_back", "Below the range: tilted back too far, slow down and level the tray."),
              ("level", "Inside the range: keep going."), ("tilted_forward", "Above the range: tilted forward too far, stop and level the tray.")),
     "distractor": "The level indicator on the forward panel was replaced last week."},
)


def make_interval(r: random.Random, sc: Mapping[str, Any], *, kind: str, target: int, near: str | None, wording: str) -> dict[str, Any]:
    """One interval question. ``band`` / ``tolerance``: target 0 below, 1 inside, 2 above. ``windows``: target 0…W−1 = a window, W = none."""
    u = sc["unit"]
    lines = [sc["role"]]
    aside = ""
    if kind in ("band", "tolerance"):
        if kind == "band":
            lo, hi = ladder(r.choice(sc["bands"]))
            rule = sc["band_text"].format(lo=_q(lo, u), hi=_q(hi, u))
            extra = {"lo_text": dtext(lo), "hi_text": dtext(hi)}
            places = max(places_of(lo), places_of(hi))
        else:
            text, tu = r.choice(sc["tolerances"])
            u = tu  # the reading is written in the tolerance's own unit (the tire's tolerance is in bar, its band in kPa)
            c, t = ladder(text)
            lo, hi = c - t, c + t
            rule = sc["tolerance_text"].format(c=unit_text(dtext(c), tu), t=unit_text(dtext(t), tu))
            extra = {"center_text": dtext(c), "tol_text": dtext(t)}
            places = max(places_of(c), places_of(t))
        places = min(places + r.choice((0, 0, 1)), 3)
        span = hi - lo
        region = [(None, lo), (lo, hi), (hi, None)][target]
        outer = (lo - span if lo - span > 0 or lo < 0 else lo / 2, hi + span)
        value, places, used = region_value(r, region[0], region[1], near=near, places=places, outer=outer)
        money = u == "$"
        fmt = pick_format(r, "interval", [value], exclude=("bare", "unicode", "mixed") if money else ())
        if money and fmt == "plain":
            fmt = "fixed2" if places_of(value) <= 2 else fmt
        written = write(value, fmt, places=places)
        options = list(sc["keys"])
        correct = options[target][0]
        lines += [rule, sc["reading"].format(v=unit_text(written, u))]
        question = r.choice(("What should the robot do?", "Which outcome applies?", "Which action fits the reading?"))
        spec = {"kind": kind, "unit": u, "value_text": written, "keys": [k for k, _ in options], **extra}
        band = ("below", "inside", "above")[target]
    else:
        count = r.choice((3, 3, 4)) if len(sc["windows"]) >= 4 else 3
        windows = list(sc["windows"])[:count]
        bounds = [ladder(w[2]) for w in windows]
        places = max(places_of(x) for b in bounds for x in b)
        places = min(places + r.choice((0, 0, 1)), 3)
        if target < count:
            lo, hi = bounds[target]
            value, places, used = region_value(r, lo, hi, near=near, places=places, outer=(lo, hi))
            correct = windows[target][0]
            band = windows[target][0]
        else:
            gaps = [(None, bounds[0][0])] + [(bounds[i][1], bounds[i + 1][0]) for i in range(count - 1)] + [(bounds[-1][1], None)]
            gap = r.choice(gaps)
            first, last = bounds[0], bounds[-1]
            outer = (first[0] - (first[1] - first[0]) if sc["domain"] == "temperature" else first[0] / 2, last[1] + (last[1] - last[0]) / 2)
            value, places, used = region_value(r, gap[0], gap[1], near=near, places=places, outer=outer)
            correct = sc["none"][0]
            band = "none"
        fmt = pick_format(r, "interval", [value])
        written = write(value, fmt, places=places)
        listed = [f"- {name}: {_q(b[0], u)} to {_q(b[1], u)}" for (_, name, _), b in zip(windows, bounds)]
        lines += [sc["window_text"], *listed, sc["reading"].format(v=unit_text(written, u))]
        options = [(key, f"{name}.") for key, name, _ in windows] + [sc["none"]]
        question = sc["window_question"]
        spec = {"kind": "windows", "unit": u, "value_text": written, "windows": [{"key": k, "lo_text": dtext(b[0]), "hi_text": dtext(b[1])}
                                                                               for (k, _, _), b in zip(windows, bounds)], "none": sc["none"][0]}
    if fmt == "plain" and "." not in written:
        fmt = "int"
    if wording == "distractor":  # a lure: the label of a WRONG option on a note about another unit
        wrong = r.choice([(k, text) for k, text in options if k != correct])
        label = wrong[1].split(":")[0].rstrip(".") if kind == "windows" and wrong[0] != sc["none"][0] else wrong[0].replace("_", " ").capitalize()
        aside = f"A note on the clipboard still reads \"{label}\"; it is from yesterday's check of another unit."
        lines.insert(1, aside)
    return {"family": "interval", "template": f"interval-{sc['name']}-{kind}", "domain": sc["domain"], "domains": [sc["domain"]], "format": fmt,
            "near": used, "band": band, "state": "\n".join(lines), "instructions": question, "options": options, "correct": correct, "spec": spec,
            "aside": aside}


# --------------------------------------------------------------------------
# Family 3 — multi-constraint selection (4–20 options): exactly one candidate meets every stated constraint
# --------------------------------------------------------------------------

NAMES = ("amber", "birch", "cedar", "dune", "ember", "fern", "garnet", "harbor", "iris", "jade", "kestrel", "lagoon", "maple", "nectar",
         "onyx", "pebble", "quartz", "raven", "sable", "tundra", "umber", "violet", "willow", "yarrow", "zephyr")
SELECTION_SIZES = (4, 5, 6, 8, 10, 12, 14, 16, 18, 20)

#: Each scenario: attributes (name, unit, decimals, op ∈ min / max / range), the status (good value, bad values), bound alternatives, the
#: rule text (bounds as {attr} / {attr_lo} / {attr_hi}) and the candidate text ({Name}, {attr}, {status}).
SELECTION_SCENARIOS: tuple[dict[str, Any], ...] = (
    {"name": "charger", "domains": ("voltage", "current", "distance"), "noun": "Charger", "key": "charger",
     "role": "You are a delivery robot looking for a charger for your battery pack.",
     "attrs": (("volt", "V", 1, "range"), ("amp", "A", 1, "min"), ("cable", "m", 1, "min")), "status": ("free and working", ("in use", "out of order")),
     "bounds": ({"volt": "23.0 25.6", "amp": "6", "cable": "1.5"}, {"volt": "46.0 51.2", "amp": "10", "cable": "2.5"},
                {"volt": "11.6 12.8", "amp": "4", "cable": "1.2"}),
     "rule": ("The pack accepts a charging voltage from {volt_lo} to {volt_hi} and needs a charger that supplies at least {amp}. You are parked "
              "{cable} from the socket wall, so the charger's cable must be at least {cable} long. Use only a charger that is free and working."),
     "item": "Charger {Name}: {volt} output, up to {amp}, {cable} cable, {status}.", "question": "Which charger should the robot use?",
     "aside": "Charger {Name} has a new label on its front."},
    {"name": "hose", "domains": ("pressure", "distance"), "noun": "Hose", "key": "hose",
     "role": "You are a maintenance robot fitting a hose to a paint sprayer.",
     "attrs": (("press", "bar", 1, "min"), ("length", "m", 1, "min"), ("bore", "mm", 1, "range")), "status": ("intact", ("cracked", "cut through")),
     "bounds": ({"press": "6", "length": "8", "bore": "9 11"}, {"press": "10", "length": "12", "bore": "6 8"}, {"press": "4.5", "length": "5", "bore": "12 14"}),
     "rule": ("The sprayer runs at {press}. Pick a hose rated for at least {press}, at least {length} long, with an inner bore from {bore_lo} "
              "to {bore_hi}. The hose must be intact."),
     "item": "Hose {Name}: rated {press}, {length} long, {bore} bore, {status}.", "question": "Which hose should the robot fit?",
     "aside": "Hose {Name} hangs on the second hook."},
    {"name": "container", "domains": ("volume", "temperature", "mass"), "noun": "Can", "key": "can",
     "role": "You are a cleaning robot about to collect hot rinse water.",
     "attrs": (("cap", "L", 1, "min"), ("heat", "°C", 0, "min"), ("tare", "kg", 1, "max")), "status": ("clean and undamaged", ("dirty", "dented at the rim")),
     "bounds": ({"cap": "12", "heat": "80", "tare": "3.5"}, {"cap": "25", "heat": "65", "tare": "6"}, {"cap": "6.5", "heat": "95", "tare": "2"}),
     "rule": ("You must collect {cap} of rinse water at {heat}. Choose a can that holds at least {cap}, is rated for at least {heat}, and "
              "weighs no more than {tare} empty, so your arm can still lift it full. The can must be clean and undamaged."),
     "item": "Can {Name}: holds {cap}, rated to {heat}, {tare} empty, {status}.", "question": "Which can should the robot take?",
     "aside": "Can {Name} has a blue lid."},
    {"name": "offer", "domains": ("currency", "time", "distance"), "noun": "Offer", "key": "offer",
     "role": "You are the purchasing robot replacing a worn bearing on a lathe.",
     "attrs": (("price", "$", 2, "max"), ("hours", "h", 0, "max"), ("tol", "mm", 3, "max")), "status": ("in stock", ("out of stock", "not in stock")),
     "bounds": ({"price": "48.00", "hours": "36", "tol": "0.05"}, {"price": "120.00", "hours": "24", "tol": "0.02"}, {"price": "15.50", "hours": "72", "tol": "0.1"}),
     "rule": ("The budget is {price} at most, the part must arrive within {hours}, and its tolerance must be {tol} or tighter (a smaller "
              "number is tighter). Only an offer that is in stock can be ordered."),
     "item": "Offer {Name}: {price}, delivery in {hours}, tolerance {tol}, {status}.", "question": "Which offer should the robot order?",
     "aside": "Offer {Name} came in by email this morning."},
    {"name": "route", "domains": ("distance", "angle", "time"), "noun": "Route", "key": "route",
     "role": "You are a delivery robot choosing a route to the loading bay.",
     "attrs": (("len", "m", 0, "max"), ("slope", "°", 1, "max"), ("mins", "min", 1, "max")), "status": ("open and clear", ("closed for cleaning", "blocked by pallets")),
     "bounds": ({"len": "250", "slope": "6", "mins": "4"}, {"len": "400", "slope": "4.5", "mins": "7"}, {"len": "120", "slope": "8", "mins": "2.5"}),
     "rule": ("Your battery allows at most {len} of driving, your load tips on slopes steeper than {slope}, and the truck leaves in {mins}, so "
              "the route must take no longer than that. The route must be open and clear."),
     "item": "Route {Name}: {len} long, steepest slope {slope}, {mins} of driving, {status}.", "question": "Which route should the robot take?",
     "aside": "Route {Name} passes the canteen."},
    {"name": "cabinet", "domains": ("temperature", "volume"), "noun": "Cabinet", "key": "cabinet",
     "role": "You are a hospital robot storing a box of medicine.",
     "attrs": (("temp", "°C", 1, "range"), ("free", "L", 1, "min")), "status": ("latched with its alarm off", ("not latched (door ajar)", "latched with its alarm on")),
     "bounds": ({"temp": "2 8", "free": "15"}, {"temp": "-25 -15", "free": "40"}, {"temp": "15 25", "free": "8"}),
     "rule": ("The box must be kept between {temp_lo} and {temp_hi} and takes {free} of space. Use a cabinet set inside that range, with at "
              "least {free} free, whose door is latched with its alarm off."),
     "item": "Cabinet {Name}: set to {temp}, {free} free, {status}.", "question": "Which cabinet should the robot use?",
     "aside": "Cabinet {Name} has a glass door."},
    {"name": "actuator", "domains": ("current", "speed", "distance"), "noun": "Actuator", "key": "actuator",
     "role": "You are the engineering robot choosing a linear actuator for a sliding gate.",
     "attrs": (("amp", "A", 1, "max"), ("speed", "mm/s", 0, "min"), ("stroke", "mm", 0, "min")), "status": ("in stock", ("out of stock", "not in stock")),
     "bounds": ({"speed": "40", "stroke": "150", "amp": "2.5"}, {"speed": "25", "stroke": "300", "amp": "4"}, {"speed": "60", "stroke": "100", "amp": "1.8"}),
     "rule": ("The gate's controller can feed at most {amp}. The actuator must move at least {speed} and have a stroke of at least {stroke}. "
              "It must be in stock."),
     "item": "Actuator {Name}: draws {amp}, moves {speed}, {stroke} stroke, {status}.", "question": "Which actuator should the robot pick?",
     "aside": "Actuator {Name} comes in a grey housing."},
    {"name": "battery", "domains": ("percent", "temperature", "mass"), "noun": "Battery", "key": "battery",
     "role": "You are a drone-servicing robot picking a spare battery for the next flight.",
     "attrs": (("charge", "%", 0, "min"), ("cell", "°C", 1, "range"), ("weight", "kg", 2, "max")), "status": ("sealed and undamaged", ("swollen", "with a cracked case")),
     "caps": {"charge": "100"},
     "bounds": ({"charge": "80", "cell": "15 35", "weight": "1.45"}, {"charge": "90", "cell": "10 30", "weight": "0.95"}, {"charge": "75", "cell": "18 32", "weight": "2.2"}),
     "rule": ("The flight needs a battery with at least {charge} charge, a cell temperature between {cell_lo} and {cell_hi}, and a weight of "
              "no more than {weight}. The battery must be sealed and undamaged."),
     "item": "Battery {Name}: {charge} charged, cells at {cell}, {weight}, {status}.", "question": "Which battery should the robot fit?",
     "aside": "Battery {Name} has a red handle."},
)


def _attr_value(r: random.Random, op: str, bound: Any, places: int, *, ok: bool, near: bool, cap: Decimal | None = None) -> Decimal:
    """A candidate's attribute: inside the allowed side (`ok`) or outside it, near the bound (within 5 %) or clearly; never above `cap`
    (e.g. a charge of at most 100 %)."""
    for _ in range(60):
        value = _attr_draw(r, op, bound, places, ok=ok, near=near)
        if value is not None and (cap is None or value <= cap):
            return value
    raise ValueError(f"no {'allowed' if ok else 'breaking'} value for {op} {bound} at {places} decimals (cap {cap})")


def _attr_draw(r: random.Random, op: str, bound: Any, places: int, *, ok: bool, near: bool) -> Decimal | None:
    for _ in range(60):
        if op == "range":
            lo, hi = bound
            if ok:
                if near:
                    found = value_near(r, lo if r.random() < 0.5 else hi, +1 if r.random() < 0.5 else -1, "rel", places)
                    if not found or not lo < found[0] < hi or places_of(found[0]) > places:
                        continue
                    return found[0]
                found = value_inside(r, lo, hi, places)
                if found and places_of(found[0]) <= places:
                    return found[0]
                continue
            side = r.choice((-1, 1))
            edge = lo if side < 0 else hi
            if near:
                found = value_near(r, edge, side, "rel", places)
                if not found or places_of(found[0]) > places:
                    continue
                return found[0]
            span = hi - lo
            value = quant(edge + side * span * Decimal(r.randint(30, 120)) / 100, places)
            if (value < lo or value > hi) and abs(value - edge) >= Decimal(1).scaleb(-places):
                return value
            continue
        b = Decimal(bound)
        above = (op == "min") == ok
        if near:
            found = value_near(r, b, +1 if above else -1, "rel", places)
            if not found or places_of(found[0]) > places:
                continue
            return found[0]
        factor = Decimal(r.randint(110, 160)) / 100 if above else Decimal(r.randint(40, 90)) / 100
        value = quant(b * factor, places)
        if (value > b if above else value < b) and value > 0:
            return value
    return None


def make_selection(r: random.Random, sc: Mapping[str, Any], *, n: int, wording: str) -> dict[str, Any]:
    """One selection question with `n` candidates: the correct one meets every constraint; each other breaks exactly one (attributes and
    the status cycled in a seeded order), about half of the numeric breaks within 5 % of the bound."""
    bounds = r.choice(sc["bounds"])
    attrs = sc["attrs"]
    parsed = {a: (tuple(ladder(bounds[a])) if op == "range" else Decimal(bounds[a])) for a, _, _, op in attrs}
    names = r.sample(NAMES, n)
    constraints = [a for a, _, _, _ in attrs] + ["status"]
    cycle: list[str] = []
    broken = []
    for i in range(n - 1):
        if i % len(constraints) == 0:
            cycle = list(constraints)
            r.shuffle(cycle)
        broken.append(cycle[i % len(constraints)])
    correct_at = r.randrange(n)
    rows = []
    others = iter(broken)
    for i, name in enumerate(names):
        breaks = None if i == correct_at else next(others)
        values = {}
        for a, unit, places, op in attrs:
            cap = Decimal(sc["caps"][a]) if a in (sc.get("caps") or {}) else None
            values[a] = _attr_value(r, op, parsed[a], places, ok=(breaks != a), near=r.random() < (0.55 if breaks == a else 0.3), cap=cap)
        status = sc["status"][0] if breaks != "status" else r.choice(sc["status"][1])
        rows.append({"name": name, "breaks": breaks, "values": values, "status": status})
    numbers = [v for row in rows for v in row["values"].values()]
    fmt = pick_format(r, "selection", numbers)

    def number(a: str, value: Decimal) -> str:
        unit, places = next((u, p) for x, u, p, _ in attrs if x == a)
        if unit == "$" and fmt != "int":
            return write(value, "fixed2")  # money: always cents (never $33.6)
        return write(value, fmt, places=places)

    def written(a: str, value: Decimal) -> str:
        return unit_text(number(a, value), next(u for x, u, _, _ in attrs if x == a))

    rule_fields = {}
    for a, unit, places, op in attrs:
        if op == "range":
            rule_fields[f"{a}_lo"], rule_fields[f"{a}_hi"] = _q(parsed[a][0], unit), _q(parsed[a][1], unit)
        else:
            rule_fields[a] = _q(parsed[a], unit)
    options, cand_spec = [], {}
    for row in rows:
        key = f"{sc['key']}_{row['name']}"
        texts = {a: written(a, row["values"][a]) for a in row["values"]}
        options.append((key, sc["item"].format(Name=row["name"].capitalize(), status=row["status"], **texts)))
        cand_spec[key] = {"values_text": {a: number(a, row["values"][a]) for a in row["values"]},
                          "status": row["status"], "breaks": row["breaks"]}
    correct = f"{sc['key']}_{rows[correct_at]['name']}"
    lines = [sc["role"]]
    aside = sc["aside"].format(Name=r.choice([n for i, n in enumerate(names) if i != correct_at]).capitalize()) if wording == "distractor" else ""
    if aside:
        lines.append(aside)
    lines.append(sc["rule"].format(**rule_fields))
    spec = {"kind": "selection", "attrs": [{"attr": a, "unit": u, "op": op, "bound_text": ([dtext(x) for x in parsed[a]] if op == "range" else dtext(parsed[a]))}
                                           for a, u, _, op in attrs], "status_ok": sc["status"][0], "candidates": cand_spec}
    near_breaks = sum(1 for row in rows if row["breaks"] not in (None, "status") and _is_near(row["values"][row["breaks"]], parsed[row["breaks"]]))
    return {"family": "selection", "template": f"selection-{sc['name']}", "domain": sc["domains"][0], "domains": list(sc["domains"]), "format": fmt,
            "near": "rel" if near_breaks else None, "band": f"n{n}", "state": "\n".join(lines), "instructions": sc["question"], "options": options,
            "correct": correct, "spec": spec, "aside": aside}


def _is_near(value: Decimal, bound: Any) -> bool:
    edges = list(bound) if isinstance(bound, tuple) else [bound]
    return any(abs(value - e) <= abs(e) * Decimal("0.05") for e in edges)


# --------------------------------------------------------------------------
# Family 4 — comparison / ordering
# --------------------------------------------------------------------------

COMPARE_DOMAINS: dict[str, dict[str, Any]] = {
    "distance": {"unit": "m", "noun": "shelf", "plural": "shelves", "role": "You are a stock-picking robot planning your next stop.", "fact": "Shelf {Name} is {v} from the dock.",
                 "what": "distance from the dock", "high": "is farthest from the dock", "low": "is nearest to the dock", "range": ("1", "60"), "places": 1,
                 "big": ("mm", "1200", "9800")},
    "speed": {"unit": "m/s", "noun": "conveyor", "role": "You are the robot balancing the load across a hub's conveyors.",
              "fact": "Conveyor {Name} runs at {v}.", "what": "speed", "high": "runs fastest", "low": "runs slowest", "range": ("0.2", "2.5"), "places": 2},
    "angle": {"unit": "°", "noun": "ramp", "role": "You are a cart robot that must avoid steep ramps.", "fact": "Ramp {Name} rises at {v}.",
              "what": "slope", "high": "is steepest", "low": "is the least steep", "range": ("1", "15"), "places": 1},
    "time": {"unit": "min", "noun": "job", "role": "You are the scheduler robot of a print farm.", "fact": "Job {Name} needs {v}.",
             "what": "time needed", "high": "takes the longest", "low": "takes the shortest time", "range": ("2", "95"), "places": 1},
    "mass": {"unit": "kg", "noun": "tote", "role": "You are a picking robot that must lift totes in a set order.", "fact": "Tote {Name} weighs {v}.",
             "what": "mass", "high": "is heaviest", "low": "is lightest", "range": ("0.4", "24"), "places": 2},
    "temperature": {"unit": "°C", "noun": "freezer", "role": "You are the cold-chain robot checking freezers.", "fact": "Freezer {Name} reads {v}.",
                    "what": "temperature", "high": "is warmest", "low": "is coldest", "range": ("-32", "4"), "places": 1, "signed": True},
    "percent": {"unit": "%", "noun": "robot", "role": "You are the fleet manager deciding which robot charges next.",
                "fact": "Robot {Name} has {v} charge left.", "what": "charge left", "high": "has the most charge left", "low": "has the least charge left",
                "range": ("4", "97"), "places": 1},
    "pressure": {"unit": "kPa", "noun": "tire", "role": "You are the tire-check robot of an AGV fleet.", "fact": "Tire {Name} reads {v}.",
                 "what": "pressure", "high": "has the highest pressure", "low": "has the lowest pressure", "range": ("180", "260"), "places": 1},
    "voltage": {"unit": "V", "noun": "cell", "role": "You are the battery-balancing robot of a storage rack.", "fact": "Cell {Name} reads {v}.",
                "what": "voltage", "high": "has the highest voltage", "low": "has the lowest voltage", "range": ("3.1", "4.2"), "places": 3},
    "volume": {"unit": "L", "noun": "tank", "role": "You are the robot that tops up coolant tanks.", "fact": "Tank {Name} holds {v}.",
               "what": "volume", "high": "holds the most coolant", "low": "holds the least coolant", "range": ("5", "80"), "places": 1,
               "big": ("mL", "1200", "8800")},
    "currency": {"unit": "$", "noun": "quote", "role": "You are the purchasing robot comparing repair quotes.", "fact": "Quote {Name} is {v}.",
                 "what": "price", "high": "is the most expensive", "low": "is the cheapest", "range": ("8", "120"), "places": 2,
                 "big": ("$", "1100", "4900")},
    "current": {"unit": "A", "noun": "motor", "role": "You are the monitor of a line of fan motors.", "fact": "Motor {Name} draws {v}.",
                "what": "current draw", "high": "draws the most current", "low": "draws the least current", "range": ("0.3", "6"), "places": 2},
}
COMPARE_KINDS = ("extreme", "second", "closest", "order")
#: The words a ranking or an ordering uses per domain: (the high end, the low end) — "farthest first", "from coldest to warmest".
COMPARE_RANK_WORDS = {"distance": ("farthest", "nearest"), "speed": ("fastest", "slowest"), "angle": ("steepest", "least steep"),
                      "time": ("longest", "shortest"), "mass": ("heaviest", "lightest"), "temperature": ("warmest", "coldest"),
                      "percent": ("most charge", "least charge"), "pressure": ("highest", "lowest"), "voltage": ("highest", "lowest"),
                      "volume": ("most coolant", "least coolant"), "currency": ("most expensive", "cheapest"), "current": ("highest draw", "lowest draw")}
#: The distractor wording's aside per domain: one item's name (an option word), nothing about its number.
COMPARE_ASIDES = {"distance": "Shelf {Name} was restocked on Monday.", "speed": "Conveyor {Name} got a new belt last week.",
                  "angle": "Ramp {Name} has a yellow handrail.", "time": "Job {Name} was sent in by the night shift.",
                  "mass": "Tote {Name} has a blue lid.", "temperature": "Freezer {Name} has a glass door.", "percent": "Robot {Name} has a red bumper.",
                  "pressure": "Tire {Name} was rotated last month.", "voltage": "Cell {Name} sits at the end of the rack.",
                  "volume": "Tank {Name} has a new level label.", "currency": "Quote {Name} came from a new supplier.",
                  "current": "Motor {Name} was cleaned last on Monday."}


def _distinct_values(r: random.Random, n: int, lo: Decimal, hi: Decimal, places: int, *, trap: bool, close: bool) -> list[Decimal]:
    """`n` distinct values in [lo, hi]: a decimal trap (same whole part, different numbers of decimals: 9.9 / 9.11 / 9.09), close values
    (within 5 % of each other) or spread values."""
    for _ in range(200):
        if trap:
            whole = quant(lo + (hi - lo) * Decimal(r.randint(100, 800)) / 1000, 0)
            pool = {whole + Decimal(r.randint(1, 99)) / Decimal(10 ** r.choice((1, 2, 2))) for _ in range(n * 3)}
            values = [v for v in pool if lo <= v <= hi]
            values = [quant(v, places_of(v)) for v in values]
            values = list(dict.fromkeys(values))
            if len(values) >= n and len({places_of(v) for v in values[:n]}) >= 2:
                chosen = r.sample(values, n)
                if len({places_of(v) for v in chosen}) >= 2:
                    return chosen
            continue
        if close:
            center = lo + (hi - lo) * Decimal(r.randint(200, 800)) / 1000
            values = {quant(center * (1 + Decimal(r.randint(-40, 40)) / 1000), places) for _ in range(n * 4)}
        else:
            values = {quant(lo + (hi - lo) * Decimal(r.randint(0, 1000)) / 1000, places) for _ in range(n * 4)}
        values = [v for v in values if lo <= v <= hi and v != 0]
        if len(values) >= n:
            return r.sample(sorted(values), n)
    raise ValueError("no distinct values")


def make_comparison(r: random.Random, domain: str, *, kind: str, n: int, wording: str, big: bool = False, trap: bool = False,
                    close: bool = False) -> dict[str, Any]:
    """One comparison question over `n` named readings of `domain`: the highest / lowest (``extreme``), the second when ranked (``second``),
    the closest to a target (``closest``), or the right ordering of three or four readings (``order``; 6 options)."""
    d = COMPARE_DOMAINS[domain]
    unit, places = d["unit"], d["places"]
    lo, hi = Decimal(d["range"][0]), Decimal(d["range"][1])
    if big and "big" in d:
        unit, lo, hi, places = d["big"][0], Decimal(d["big"][1]), Decimal(d["big"][2]), 0
    count = n if kind != "order" else r.choice((3, 4))
    values = _distinct_values(r, count + (1 if kind == "closest" else 0), lo, hi, places, trap=trap, close=close)
    target = None
    if kind == "closest":
        target = values.pop()
        diffs = sorted(abs(v - target) for v in values)
        if diffs[0] == diffs[1] or diffs[0] == 0:
            raise ValueError("a tie for closest")
    names = r.sample(NAMES, count)
    money = unit == "$"
    fmt = "mixed" if trap and not money else pick_format(r, "comparison", values + ([target] if target is not None else []),
                                                         exclude=("mixed", "bare") if money else ())
    if money and fmt not in ("int", "thousands"):
        places = 2
    written = [write(v, fmt, places=places) for v in values]
    if fmt == "plain" and all("." not in w for w in written):
        fmt = "int"
    facts = [d["fact"].format(Name=nm.capitalize(), v=unit_text(w, unit)) for nm, w in zip(names, written)]
    high = r.random() < 0.5
    order_idx = sorted(range(count), key=lambda i: values[i], reverse=high)
    lines = [d["role"], *facts]
    aside = ""
    if wording == "distractor":
        right_name = None
        if kind == "extreme":
            right_name = names[order_idx[0]]
        elif kind == "second":
            right_name = names[order_idx[1]]
        elif kind == "closest":
            right_name = names[min(range(count), key=lambda i: abs(values[i] - target))]
        aside = COMPARE_ASIDES[domain].format(Name=r.choice([n for n in names if n != right_name]).capitalize())
        lines.insert(1, aside)
    if kind in ("extreme", "second", "closest"):
        options = [(f"{d['noun']}_{nm}", f"{d['noun'].capitalize()} {nm.capitalize()}.") for nm in names]
        if kind == "extreme":
            pick = order_idx[0]
            question = f"Which {d['noun']} {d['high'] if high else d['low']}?"
        elif kind == "second":
            pick = order_idx[1]
            hi_word, lo_word = COMPARE_RANK_WORDS[domain]
            question = (f"Rank the {d.get('plural', d['noun'] + 's')} by {d['what']}, {hi_word if high else lo_word} first. "
                        f"Which {d['noun']} comes second?")
        else:
            target_text = write(target, "plain" if fmt in ("mixed", "bare") else fmt, places=places)
            pick = min(range(count), key=lambda i: abs(values[i] - target))
            question = f"Which {d['noun']}'s {d['what']} is closest to {unit_text(target_text, unit)}?"
        correct = options[pick][0]
        band = kind if kind != "extreme" else ("highest" if high else "lowest")
    else:
        ascending = not high
        right = sorted(range(count), key=lambda i: values[i], reverse=not ascending)
        perms = _orderings(r, right)
        options = [(f"order_{i + 1}", ", then ".join(names[j].capitalize() for j in perm) + ".") for i, perm in enumerate(perms)]
        correct = options[perms.index(right)][0]
        hi_word, lo_word = COMPARE_RANK_WORDS[domain]
        question = (f"Which list gives the {d.get('plural', d['noun'] + 's')} in order of {d['what']}, from "
                    f"{f'{lo_word} to {hi_word}' if ascending else f'{hi_word} to {lo_word}'}?")
        band = "ascending" if ascending else "descending"
    spec = {"kind": kind, "unit": unit, "items": {nm: w for nm, w in zip(names, written)}, "high": high,
            "target_text": (write(target, "plain" if fmt in ("mixed", "bare") else fmt, places=places) if target is not None else None),
            "noun": d["noun"]}
    if kind == "order":
        spec["orders"] = {k: [names[j] for j in perm] for (k, _), perm in zip(options, perms)}
    near = "close" if close else ("trap" if trap else None)
    return {"family": "comparison", "template": f"comparison-{kind}", "domain": domain, "domains": [domain], "format": fmt, "near": near,
            "band": band, "state": "\n".join(lines), "instructions": question, "options": options, "correct": correct, "spec": spec,
            "aside": aside, "aside_names": {n.capitalize(): f"{d['noun']}_{n}" for n in names} if kind != "order" else {}}


def _orderings(r: random.Random, right: list[int]) -> list[list[int]]:
    """Six orderings: every permutation of three items, or for four items the right one and five near misses (two neighbours swapped,
    the reverse, a rotation), shuffled."""
    import itertools

    if len(right) == 3:
        perms = [list(p) for p in itertools.permutations(right)]
    else:
        a = right
        candidates = [a, [a[1], a[0], a[2], a[3]], [a[0], a[2], a[1], a[3]], [a[0], a[1], a[3], a[2]], list(reversed(a)), a[1:] + a[:1],
                      [a[3], a[1], a[2], a[0]]]
        perms = []
        for p in candidates:
            if p not in perms:
                perms.append(p)
        perms = perms[:6]
    r.shuffle(perms)
    return perms


# --------------------------------------------------------------------------
# Family 5 — unit conversion inside rules: the rule in one unit, the reading in another
# --------------------------------------------------------------------------

CONVERSION_SCENARIOS: tuple[dict[str, Any], ...] = (
    {"name": "lidgap", "domain": "distance", "rule_unit": "cm", "read_unit": "mm", "factor": "10",
     "role": "You are a bin-picking robot waiting for a bin lid to open.", "reading": "The lid-gap sensor reads {v}.",
     "limit": ("The arm may reach into the bin only when the lid gap is at least {t}.", (("wait", "Wait: the gap is too narrow."), ("reach_in", "Reach into the bin."))),
     "thresholds": ("12", "8.5", "15"), "dir": "min"},
    {"name": "payload", "domain": "mass", "rule_unit": "kg", "read_unit": "g", "factor": "1000",
     "role": "You are a lifting robot about to pick a sealed box.", "reading": "The scale under the box reads {v}.",
     "steps": ("Grip rule: use the soft grip under {t1}, the normal grip from {t1} up to {t2}, the firm grip from {t2} up to {t3}, and call "
               "a person at {t3} or more.", (("soft_grip", "Soft grip."), ("normal_grip", "Normal grip."), ("firm_grip", "Firm grip."),
                                             ("call_person", "Call a person to lift it."))),
     "ladders": ("0.5 2 5", "0.8 2.5 6", "1.2 3 7.5")},
    {"name": "door", "domain": "time", "rule_unit": "min", "read_unit": "s", "factor": "60",
     "role": "You are the cold-room robot watching the door.", "reading": "The door has been open for {v}.",
     "limit": ("If the cold-room door has been open longer than {t}, close it; otherwise leave it for the worker.",
               (("leave_open", "Leave the door for the worker."), ("close_door", "Close the door."))), "thresholds": ("2.5", "1.5", "4"), "dir": "max"},
    {"name": "settle", "domain": "time", "rule_unit": "s", "read_unit": "ms", "factor": "1000",
     "role": "You are the camera robot waiting for a part to stop wobbling.", "reading": "The part has been still for {v}.",
     "limit": ("Take the photo only after the part has been still for at least {t}.", (("wait", "Keep waiting."), ("take_photo", "Take the photo."))),
     "thresholds": ("1.2", "0.8", "2.5"), "dir": "min"},
    {"name": "sump", "domain": "volume", "rule_unit": "L", "read_unit": "mL", "factor": "1000",
     "role": "You are the robot that runs a coolant pump.", "reading": "The sump level sensor reads {v}.",
     "band": ("The pump may run only while the sump holds between {lo} and {hi}.", (("too_low", "Below the range: stop and refill."),
                                                                                     ("run", "Inside the range: run the pump."),
                                                                                     ("too_high", "Above the range: stop and drain."))),
     "bands": ("2.5 6", "1.2 3.5", "8 14")},
    {"name": "probe", "domain": "voltage", "rule_unit": "V", "read_unit": "mV", "factor": "1000",
     "role": "You are the test robot checking a sensor's supply.", "reading": "The supply reads {v}.",
     "band": ("The sensor is powered correctly when its supply is between {lo} and {hi}.", (("undervolt", "Below the range: report undervoltage."),
                                                                                             ("powered_ok", "Inside the range: continue the test."),
                                                                                             ("overvolt", "Above the range: cut the power."))),
     "bands": ("3.2 3.4", "4.75 5.25", "1.7 1.9")},
    {"name": "stall", "domain": "current", "rule_unit": "A", "read_unit": "mA", "factor": "1000",
     "role": "You are the safety monitor of a robot's wrist motor.", "reading": "The wrist motor draws {v}.",
     "limit": ("Stop the wrist if its motor draws more than {t}.", (("keep_moving", "Keep moving."), ("stop_wrist", "Stop the wrist."))),
     "thresholds": ("1.5", "0.8", "2.2"), "dir": "max"},
    {"name": "press", "domain": "pressure", "rule_unit": "bar", "read_unit": "kPa", "factor": "100",
     "role": "You are the robot running a pneumatic press.", "reading": "The press gauge reads {v}.",
     "steps": ("Pressure bands: under {t1} is low, from {t1} up to {t2} is normal, from {t2} up to {t3} is high, and {t3} or more is critical.",
               (("low", "Low: raise the supply."), ("normal", "Normal: carry on."), ("high", "High: slow the cycle."), ("critical", "Critical: stop the press."))),
     "ladders": ("4 6 7.5", "2.5 4 5.5", "5 6.5 8")},
    {"name": "hydraulic", "domain": "pressure", "rule_unit": "MPa", "read_unit": "kPa", "factor": "1000",
     "role": "You are the robot watching a hydraulic lift.", "reading": "The lift's pressure reads {v}.",
     "limit": ("Stop lifting if the pressure exceeds {t}.", (("keep_lifting", "Keep lifting."), ("stop_lift", "Stop lifting."))),
     "thresholds": ("2.5", "1.8", "3.2"), "dir": "max"},
    {"name": "refund", "domain": "currency", "rule_unit": "$", "read_unit": "cents", "factor": "100",
     "role": "You are the kiosk robot handling a refund.", "reading": "The refund request is for {v}.",
     "limit": ("You may refund up to {t} on your own; larger refunds need a manager.", (("refund", "Refund it now."), ("call_manager", "Call a manager."))),
     "thresholds": ("12.50", "8.00", "20.00"), "dir": "max"},
    {"name": "charge", "domain": "percent", "rule_unit": "%", "read_unit": "fraction", "factor": "0.01",
     "role": "You are a floor-cleaning robot deciding what to do next.", "reading": "Your battery monitor reports {v} of a full charge.",
     "band3": ("Below {lo}, return to the dock now; from {lo} up to {hi}, finish the current room and then charge; at {hi} or more, keep cleaning.",
               (("return_now", "Return to the dock now."), ("finish_then_charge", "Finish the room, then charge."), ("keep_cleaning", "Keep cleaning."))),
     "bands": ("20 45", "15 40", "25 60")},
    {"name": "walk", "domain": "speed", "rule_unit": "m/s", "read_unit": "cm/s", "factor": "100",
     "role": "You are a mobile robot driving near people.", "reading": "Your wheel odometry reads {v}.",
     "limit": ("Near people your speed must not exceed {t}.", (("speed_ok", "Speed is fine: carry on."), ("slow_down", "Slow down."))),
     "thresholds": ("0.8", "0.5", "1.2"), "dir": "max"},
    {"name": "feed", "domain": "speed", "rule_unit": "m/s", "read_unit": "m/min", "factor": "60",
     "role": "You are the robot feeding sheet into a cutter.", "reading": "The feed roller reports {v}.",
     "band": ("The cutter needs a feed between {lo} and {hi}.", (("feed_slow", "Below the range: speed the feed up."), ("feed_ok", "Inside the range: no change."),
                                                                  ("feed_fast", "Above the range: slow the feed down."))),
     "bands": ("0.2 0.35", "0.5 0.8", "0.12 0.2")},
    {"name": "resin", "domain": "temperature", "rule_unit": "°C", "read_unit": "K", "offset": "273.15",
     "role": "You are the robot curing resin parts in an oven.", "reading": "The oven's thermocouple reads {v}.",
     "band": ("Cure only while the oven is between {lo} and {hi}.", (("too_cool", "Below the range: keep heating, do not load."),
                                                                      ("cure", "Inside the range: load the parts."), ("too_hot", "Above the range: vent the oven."))),
     "bands": ("60 75", "80 95", "45 55")},
)


def make_conversion(r: random.Random, sc: Mapping[str, Any], *, target: int, near: str | None, wording: str) -> dict[str, Any]:
    """One conversion question: the rule's bounds in ``rule_unit``, the reading in ``read_unit`` (reading = value × factor, or value +
    offset). ``limit``: target 0 below the bound, 1 above; ``band`` / ``band3``: 0 below, 1 inside, 2 above; ``steps``: 0…3."""
    kind = next(k for k in ("limit", "steps", "band", "band3") if k in sc)
    rule_text, options = sc[kind]
    ru = sc["rule_unit"]
    if kind == "limit":
        bounds = [Decimal(r.choice(sc["thresholds"]))]
        rule = rule_text.format(t=_q(bounds[0], ru))
    elif kind == "steps":
        bounds = ladder(r.choice(sc["ladders"]))
        rule = rule_text.format(t1=_q(bounds[0], ru), t2=_q(bounds[1], ru), t3=_q(bounds[2], ru))
    else:
        bounds = ladder(r.choice(sc["bands"]))
        rule = rule_text.format(lo=_q(bounds[0], ru), hi=_q(bounds[1], ru))
    target = target % len(options)
    places = min(max(places_of(b) for b in bounds) + r.choice((0, 0, 1)), 3)
    edges = [None, *bounds, None]
    lo, hi = edges[target], edges[target + 1]
    first, last = bounds[0], bounds[-1]
    outer = (first / 2 if first > 0 else first - abs(first), last * Decimal("1.6"))
    if "offset" in sc and near == "abs":
        near = "rel"  # within 0.05 °C a reading in K would test the 273.15 offset's decimals, not the rule
    value, places, used = region_value(r, lo, hi, near=near, places=places, outer=outer)
    if "offset" in sc:
        reading = value + Decimal(sc["offset"])
    else:
        reading = value * Decimal(sc["factor"])
    reading = quant(reading, places_of(reading))
    fmt = pick_format(r, "conversion", [reading], exclude=("fixed2",) if sc["read_unit"] in ("cents", "mV", "mA", "ms", "g", "mL", "mm") else ())
    written = write(reading, fmt)
    if fmt == "plain" and "." not in written:
        fmt = "int"
    shown = written if sc["read_unit"] == "fraction" else unit_text(written, sc["read_unit"])
    lines = [sc["role"]]
    aside = ""
    if wording == "distractor":
        blind = sum(1 for b in bounds if reading > b)  # the answer of a reader who skips the unit conversion
        lure = blind if blind != target else r.choice([i for i in range(len(options)) if i != target])
        aside = f"A sign by the station reads \"{options[lure][0].replace('_', ' ').capitalize()}\" in faded letters; it belongs to an old line."
        lines.append(aside)
    lines += [rule, sc["reading"].format(v=shown)]
    question = r.choice(("What should the robot do now?", "Which option follows from the rule?", "Which action is right?"))
    spec = {"kind": "conversion", "rule": kind, "rule_unit": ru, "read_unit": sc["read_unit"], "value_text": written,
            "bounds_text": [dtext(b) for b in bounds], "keys": [k for k, _ in options], "dir": sc.get("dir"),
            "factor": sc.get("factor"), "offset": sc.get("offset")}
    band = options[target][0]
    return {"family": "conversion", "template": f"conversion-{sc['name']}", "domain": sc["domain"], "domains": [sc["domain"]], "format": fmt,
            "near": used, "band": band, "state": "\n".join(lines), "instructions": question, "options": list(options), "correct": options[target][0],
            "spec": spec, "aside": aside}


# --------------------------------------------------------------------------
# The plan and the records
# --------------------------------------------------------------------------

#: Questions per family (per scenario / domain × count).
PLAN = {"threshold": 16, "interval": 13, "selection": 19, "comparison": 13, "conversion": 11}
NEAR_CYCLE = (None, "rel", None, "abs")


def _attempt(make: Any, seed_parts: tuple, **kwargs: Any) -> dict[str, Any]:
    """Run a family maker with a seeded generator; on a rare impossible draw (a tie, no value at those decimals) reseed (bounded)."""
    last = None
    for attempt in range(25):
        try:
            return make(rng(*seed_parts, attempt), **kwargs)
        except ValueError as error:
            last = error
    raise ValueError(f"{seed_parts}: {last}")


def plan_items(seed: int = SEED) -> list[dict[str, Any]]:
    """Every N9g question as a design dict (family maker output + wording), in family order."""
    items: list[dict[str, Any]] = []
    for si, sc in enumerate(THRESHOLD_SCENARIOS):
        for j in range(PLAN["threshold"]):
            k_count = (2, 3, 4, 5)[j % 4]
            style = THRESHOLD_STYLES[(j // 4 + si) % 4]
            step = (j * 3 + si) % (k_count + 1)
            sign = 1 if (j + si) % 2 == 0 else -1
            near = NEAR_CYCLE[(j + si) % 4]
            wording = WORDINGS[(j // 2 + si) % 2]
            item = _attempt(lambda r, **kw: make_threshold(r, sc, **kw), ("threshold", sc["name"], j, seed), style=style, k_count=k_count,
                            sign=sign, step=step, near=near, wording=wording)
            items.append({**item, "wording": wording})
    for si, sc in enumerate(INTERVAL_SCENARIOS):
        for j in range(PLAN["interval"]):
            kind = sc["kinds"][j % len(sc["kinds"])]
            span = (len(sc["windows"]) + 1) if kind == "windows" else 3
            target = (j * 2 + si) % span
            near = NEAR_CYCLE[(j + si) % 4]
            wording = WORDINGS[(j // 2 + si) % 2]
            item = _attempt(lambda r, **kw: make_interval(r, sc, **kw), ("interval", sc["name"], j, seed), kind=kind, target=target,
                            near=near, wording=wording)
            items.append({**item, "wording": wording})
    for si, sc in enumerate(SELECTION_SCENARIOS):
        for j in range(PLAN["selection"]):
            n = SELECTION_SIZES[(j + si) % len(SELECTION_SIZES)]
            wording = WORDINGS[(j // 2 + si) % 2]
            item = _attempt(lambda r, **kw: make_selection(r, sc, **kw), ("selection", sc["name"], j, seed), n=n, wording=wording)
            items.append({**item, "wording": wording})
    for si, domain in enumerate(COMPARE_DOMAINS):
        for j in range(PLAN["comparison"]):
            kind = COMPARE_KINDS[(j + si) % len(COMPARE_KINDS)]
            n = (3, 4, 5, 6, 7, 8)[(j + 2 * si) % 6]
            if kind == "second" and n == 3:
                n = 4
            flavour = (j + si) % 4
            big = "big" in COMPARE_DOMAINS[domain] and (j + si) % 5 == 0
            wording = WORDINGS[(j // 2 + si) % 2]
            item = _attempt(lambda r, **kw: make_comparison(r, domain, **kw), ("comparison", domain, j, seed), kind=kind, n=n, wording=wording,
                            big=big, trap=(flavour == 1 and not big), close=(flavour == 3))
            items.append({**item, "wording": wording})
    for si, sc in enumerate(CONVERSION_SCENARIOS):
        for j in range(PLAN["conversion"]):
            near = NEAR_CYCLE[(j + si) % 4]
            wording = WORDINGS[(j // 2 + si) % 2]
            item = _attempt(lambda r, **kw: make_conversion(r, sc, **kw), ("conversion", sc["name"], j, seed), target=j + si, near=near,
                            wording=wording)
            items.append({**item, "wording": wording})
    return items


#: Words of option keys that name a kind of thing, not an option (a cue counts only on a distinctive word).
GENERIC_KEY_WORDS = {"dock", "chute", "zone", "charger", "hose", "can", "offer", "route", "cabinet", "actuator", "battery", "shelf", "conveyor",
                     "ramp", "job", "tote", "freezer", "robot", "tire", "cell", "tank", "quote", "motor", "order", "too", "the", "turn", "shift", "speed"}


def distractor_names_key(item: Mapping[str, Any]) -> bool:
    """Whether the distractor wording names the right option: a candidate / item name equal to the key's, or a distinctive word of the key
    (not :data:`GENERIC_KEY_WORDS`) in the distractor sentences (reported in the manifest — the distractors are option words in another sense,
    so sometimes they name the key and sometimes a wrong option)."""
    aside = str(item.get("aside") or "")
    names = item.get("aside_names") or {}
    if names:
        return any(name in aside and key == item["correct"] for name, key in names.items())
    if item["family"] == "conversion":  # the faded sign quotes one option
        return '"' in aside and aside.split('"')[1].lower().replace(" ", "_") == str(item["correct"])
    if item["family"] == "interval":  # the note quotes one option's label (a window's name or a key)
        label = aside.split('"')[1] if '"' in aside else ""
        texts = dict(item["options"])
        return label.lower().replace(" ", "_") == str(item["correct"]) or label == str(texts.get(item["correct"], "")).split(":")[0].rstrip(".")
    words = [w for w in str(item["correct"]).lower().split("_") if len(w) >= 3 and w not in GENERIC_KEY_WORDS]
    if item["family"] == "selection":
        words = [str(item["correct"]).split("_", 1)[1]]
    return any(re.search(rf"\b{re.escape(w)}", aside.lower()) for w in words)


def to_record(index: int, item: Mapping[str, Any]) -> dict[str, Any]:
    """A design dict as a ``simgen-request/0`` record (one choice question ``q``, a one-hot label of kind ``probe``)."""
    from robo_jev.decisions.records import SCHEMA

    keys = [k for k, _ in item["options"]]
    if len(set(keys)) != len(keys) or item["correct"] not in keys:
        raise ValueError(f"{item['template']}: bad options")
    meta = {"set": N9G_SET, "family": item["family"], "template": item["template"], "domain": item["domain"], "domains": list(item["domains"]),
            "domain_seen": item["domain"] in SEEN_DOMAINS, "format": item["format"], "wording": item["wording"],
            "distractor_names_key": distractor_names_key(item) if item["wording"] == "distractor" else None, "near": item["near"],
            "band": item["band"], "n_options": len(keys), "correct": item["correct"], "target": [N9G_QUESTION], "spec": item["spec"],
            "option_range": dict(OPTION_RANGE)}
    return {"schema": SCHEMA, "id": f"{N9G_PREFIX}{index:04d}", "meta": meta,
            "request": {"state": item["state"], "questions": {N9G_QUESTION: {"type": "choice", "instructions": item["instructions"],
                                                                             "criteria": {k: v for k, v in item["options"]}}}},
            "labels": {N9G_QUESTION: {"type": "choice", "kind": "probe", "probabilities": {k: (1.0 if k == item["correct"] else 0.0) for k in keys}}}}


def n9g_records(seed: int = SEED) -> list[dict[str, Any]]:
    """The whole gate (deterministic for a seed); every record passes :func:`robo_jev.decisions.records.validate_record`."""
    from robo_jev.decisions.records import validate_record

    records = [to_record(i, item) for i, item in enumerate(plan_items(seed))]
    bad = {r["id"]: p for r in records if (p := validate_record(r))}
    if bad:
        raise ValueError(f"{len(bad)} N9g records fail validate_record: {list(bad.items())[:3]}")
    whole = Counter(json.dumps(r["request"], sort_keys=True, ensure_ascii=False) for r in records)
    if max(whole.values()) > 1:
        raise ValueError("two N9g requests are identical")
    return records
