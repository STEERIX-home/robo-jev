"""Task v0.8d — A16s: the arm's distance sweep, a reported v0.8 set (synthetic; derived from A16's requests; eval-only, sealed for training).

robo-jev-serve sent one A16 y question (phrase removed) to v0.7 with only the remaining distance changed: the answer is right only at ≥ 25 cm;
over −20…+15 cm it is mostly "stay"; at −25…−40 cm P(correct option) is 0.26–0.31; z behaves the same. A16s draws that curve for all seven
models (the untrained base included) to tell a 2B limit from a training effect. Reported only — it enters nothing in the rule.

**Design** (controller, 2026-10-16 ~16:20 KST):

* *Template*: A16's requests (``artifacts/v08/r16/A16.jsonl``; the constant protocol state). In each request the x, y and z remaining
  distances are set to three sweep values, the intent question's "remaining travel x …, y …, z … cm" facts are rewritten to match, the y
  question loses " (up is plus)" (robo-jev-web's current form, 6429c5c), and the wrist and fingers questions stay as recorded.
* *Grid* (:data:`MAGNITUDES` × sign): 27 magnitudes from 0.1 to 40 cm, both signs — 54 values — never an exact boundary (0.2, 1, 3, 8, 20).
* *Formats* (:func:`format_distance`): ``web`` — the sign always written, shortest decimal (``+12``, ``-12``, ``+0.15``; robo-jev-web's own);
  ``plain`` — no plus sign, at least one decimal (``12.0``, ``-12.0``, ``0.15``). One format per request (its three axes and its intent facts).
* *Packing*: per format, the 54 values go to x, y and z by three independent seeded permutations, so each (value, axis, format) occurs once:
  54 values × 3 axes × 2 formats = 324 axis questions in 108 requests (template = A16 record k mod 90).
* *Labels by the question's own rule* (:func:`label`): the sign picks plus / minus; |d| ≥ 20 far, ≥ 8 mid, ≥ 3 near, ≥ 1 small,
  0.2 < |d| < 1 fine, |d| ≤ 0.2 stay (the grid has no boundary value, so the ±0.2 reading never matters here).
* *Scoring*: serve's conditions — the bundle's runner, default key (sha256 of the constant state), single order seed 1 — plus order seed 2
  for flips; only the x, y and z questions are read (``meta.target``; each question's prompt does not depend on the others).

**Readings** (:func:`sweep_rows`, :func:`sweep_summary`, :func:`curves`): per model and arm (seed means), accuracy and mean P(correct option)
per signed distance and per band (:data:`READ_BANDS`: < 1, 1–3, 3–8, 8–20, 20–25, ≥ 25), by sign and by number format; the stay rate per
band; sign errors; flips between the two order seeds. :func:`curves_svg` draws accuracy and P(correct) against the signed distance.
"""

from __future__ import annotations

import html
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from robo_jev.bench.gates_v08 import ARTIFACTS, option_range_mark, rng
from robo_jev.bench.r16_v08 import DISTANCE, UP_IS_PLUS
from robo_jev.decisions.records import SCHEMA

__all__ = [
    "A16S_DIR", "A16S_PREFIX", "A16S_SET", "AXES", "FORMATS", "MAGNITUDES", "READINGS", "READ_BANDS", "build_records", "check_frozen", "curves",
    "curves_svg", "format_distance", "label", "read_band", "sweep_rows", "sweep_summary", "sweep_values",
]

A16S_SET, A16S_PREFIX = "v08-a16s", "v08a16s-"
A16S_DIR = ARTIFACTS / "v08" / "a16s"
AXES = ("x", "y", "z")
MAGNITUDES = (0.1, 0.15, 0.3, 0.5, 0.8, 0.95, 1.05, 1.5, 2.0, 2.5, 2.95, 3.05, 4.0, 5.0, 6.0, 7.0, 7.9, 8.1, 10.0, 12.0, 15.0, 18.0, 19.9, 20.1,
              25.0, 30.0, 40.0)
BOUNDARIES = (0.2, 1.0, 3.0, 8.0, 20.0)
FORMATS = ("web", "plain")
#: The two order seeds read (serve's single order, then a second seed for flips).
READINGS = ("single-s1", "single-s2")
#: Reading bands of |d| (inclusive lower, exclusive upper).
READ_BANDS: tuple[tuple[str, float, float], ...] = (("<1", 0.0, 1.0), ("1-3", 1.0, 3.0), ("3-8", 3.0, 8.0), ("8-20", 8.0, 20.0),
                                                    ("20-25", 20.0, 25.0), (">=25", 25.0, float("inf")))
SEED = 20261016
_TRAVEL = re.compile(r"remaining travel x ([+-]?\d+(?:\.\d+)?), y ([+-]?\d+(?:\.\d+)?), z ([+-]?\d+(?:\.\d+)?) cm")


def sweep_values() -> list[float]:
    """The 54 signed sweep values: every magnitude, positive then negative."""
    return [m for m in MAGNITUDES] + [-m for m in MAGNITUDES]


def format_distance(d: float, kind: str) -> str:
    """``web``: the sign always written, shortest decimal (``+12``, ``-0.15``); ``plain``: no plus sign, at least one decimal (``12.0``,
    ``-12.0``, ``0.15``). Both parse back to the same number."""
    d = float(d)
    if kind == "web":
        text = format(d, "+g")
    elif kind == "plain":
        text = repr(d)
    else:
        raise ValueError(f"format {kind!r}: one of {FORMATS}")
    if float(text) != d:
        raise ValueError(f"{d} written as {text!r} does not read back")
    return text


def label(d: float) -> str:
    """The question's own rule: the sign picks plus / minus; |d| ≥ 20 far, ≥ 8 mid, ≥ 3 near, ≥ 1 small, 0.2 < |d| < 1 fine, |d| ≤ 0.2 stay."""
    m = abs(float(d))
    if m <= 0.2:
        return "stay"
    step = "far" if m >= 20 else "mid" if m >= 8 else "near" if m >= 3 else "small" if m >= 1 else "fine"
    return f"{'plus' if d > 0 else 'minus'}_{step}"


def read_band(d: float) -> str:
    m = abs(float(d))
    return next(name for name, low, high in READ_BANDS if low <= m < high)


def _axis_text(text: str, axis: str, value: str) -> str:
    found = DISTANCE.search(text)
    if not found or found.group(2) != axis:
        raise ValueError(f"no distance sentence for axis {axis} in {text[:60]!r}")
    out = text[: found.start(1)] + value + text[found.end(1):]
    if axis == "y":
        if out.count(UP_IS_PLUS) != 1:
            raise ValueError("the y question should carry ' (up is plus)' once")
        out = out.replace(UP_IS_PLUS, "")
    return out


def build_records(templates: Sequence[Mapping[str, Any]], *, seed: int = SEED) -> list[dict[str, Any]]:
    """The sweep (module docstring) from A16's records: per format three seeded permutations of :func:`sweep_values` (x, y, z), request i of the
    format takes the i-th value of each; the template is A16 record k mod 90 (k = the request's index over both formats)."""
    values = sweep_values()
    out: list[dict[str, Any]] = []
    for kind in FORMATS:
        perms = {}
        for axis in AXES:
            order = list(values)
            rng("v08-a16s", seed, kind, axis).shuffle(order)
            perms[axis] = order
        for i in range(len(values)):
            k = len(out)
            template = templates[k % len(templates)]
            questions = json.loads(json.dumps(template["request"]["questions"]))
            sweep = {}
            for axis in AXES:
                d = perms[axis][i]
                text = format_distance(d, kind)
                questions[axis]["instructions"] = _axis_text(questions[axis]["instructions"], axis, text)
                sweep[axis] = {"distance": d, "text": text, "label": label(d), "band": read_band(d)}
            intent = questions["intent"]["instructions"]
            if not _TRAVEL.search(intent):
                raise ValueError(f"{template['id']}: no 'remaining travel' facts in the intent question")
            questions["intent"]["instructions"] = _TRAVEL.sub(
                f"remaining travel x {sweep['x']['text']}, y {sweep['y']['text']}, z {sweep['z']['text']} cm", intent)
            out.append({"schema": SCHEMA, "id": f"{A16S_PREFIX}{k:04d}",
                        "meta": {"set": A16S_SET, "env": "arm", "template": template["id"], "format": kind, "sweep": sweep, "target": list(AXES),
                                 "y_phrase": "removed (robo-jev-web's current form, 6429c5c)", "option_range": option_range_mark()},
                        "request": {"state": template["request"]["state"], "questions": questions}})
    return out


# --------------------------------------------------------------------------
# Readings
# --------------------------------------------------------------------------


def _sign(option: str) -> str | None:
    return option.split("_", 1)[0] if option.startswith(("plus_", "minus_")) else None


def sweep_rows(records: Sequence[Mapping[str, Any]], answers: Mapping[str, Any], *, reading: str = "single-s1",
               second: str | None = "single-s2") -> list[dict[str, Any]]:
    """One row per answered axis question: distance, format, axis, label, band, the model's top option and the values ``correct``,
    ``p_correct``, ``stay``, ``sign_error`` (None unless the rule and the model both move) and ``flip`` (None without the second reading)."""
    rows = []
    for record in records:
        rid = str(record["id"])
        for axis in AXES:
            got = (answers.get(rid) or {}).get(axis) or {}
            first = got.get(reading)
            if not first or first.get("choice") is None:
                continue
            sweep = record["meta"]["sweep"][axis]
            top = str(first["choice"])
            probs = first.get("probabilities") or {}
            other = got.get(second) if second else None
            sign_error = None if sweep["label"] == "stay" or top == "stay" else float(_sign(top) != _sign(sweep["label"]))
            rows.append({"id": rid, "axis": axis, "format": record["meta"]["format"], "distance": float(sweep["distance"]), "label": sweep["label"],
                         "band": sweep["band"], "top": top, "correct": float(top == sweep["label"]),
                         "p_correct": float(probs.get(sweep["label"], 0.0)), "stay": float(top == "stay"), "sign_error": sign_error,
                         "flip": None if not other or other.get("choice") is None else float(str(other["choice"]) != top)})
    return rows


def seed_mean_rows(first: Sequence[Mapping[str, Any]], second: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """An arm's rows: per (record, axis) both seeds answered, the mean of each value (``sign_error`` / ``flip`` over the defined ones)."""
    b = {(r["id"], r["axis"]): r for r in second}
    out = []
    for r in first:
        o = b.get((r["id"], r["axis"]))
        if o is None:
            continue
        row = {k: r[k] for k in ("id", "axis", "format", "distance", "label", "band")}
        row["top"] = None
        for key in ("correct", "p_correct", "stay"):
            row[key] = (r[key] + o[key]) / 2.0
        for key in ("sign_error", "flip"):
            vals = [v for v in (r[key], o[key]) if v is not None]
            row[key] = sum(vals) / len(vals) if vals else None
        out.append(row)
    return out


def _mean(values: Iterable[float | None]) -> dict[str, Any]:
    kept = [float(v) for v in values if v is not None]
    return {"value": sum(kept) / len(kept) if kept else None, "n": len(kept)}


def sweep_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Every reading of one model or arm: per band (pooled, by sign, by format, by sign × format) accuracy, mean P(correct), the stay rate,
    sign errors and flips; and the totals."""
    def block(chosen: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        return {"accuracy": _mean(r["correct"] for r in chosen), "p_correct": _mean(r["p_correct"] for r in chosen),
                "stay": _mean(r["stay"] for r in chosen), "sign_errors": _mean(r["sign_error"] for r in chosen),
                "flips": _mean(r["flip"] for r in chosen)}

    groups: dict[str, list] = defaultdict(list)
    for r in rows:
        sign = "+" if r["distance"] > 0 else "-"
        for key in (r["band"], f"{r['band']}|{sign}", f"{r['band']}|{r['format']}", f"{r['band']}|{sign}|{r['format']}"):
            groups[key].append(r)
    return {"total": block(rows), "by_band": {k: block(v) for k, v in sorted(groups.items())}, "rows": len(rows)}


def curves(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Accuracy and mean P(correct) per signed distance — pooled over the three axes and both formats, and per format."""
    by_d: dict[float, list] = defaultdict(list)
    by_df: dict[tuple[float, str], list] = defaultdict(list)
    for r in rows:
        by_d[r["distance"]].append(r)
        by_df[(r["distance"], r["format"])].append(r)
    points = []
    for d in sorted(by_d):
        chosen = by_d[d]
        entry = {"distance": d, "n": len(chosen), "accuracy": sum(r["correct"] for r in chosen) / len(chosen),
                 "p_correct": sum(r["p_correct"] for r in chosen) / len(chosen), "stay": sum(r["stay"] for r in chosen) / len(chosen)}
        for kind in FORMATS:
            sub = by_df.get((d, kind)) or []
            if sub:
                entry[f"accuracy_{kind}"] = sum(r["correct"] for r in sub) / len(sub)
                entry[f"p_correct_{kind}"] = sum(r["p_correct"] for r in sub) / len(sub)
        points.append(entry)
    return {"points": points}


# --------------------------------------------------------------------------
# The figure (static SVG; light and dark; native tooltips)
# --------------------------------------------------------------------------

#: Series styles: the first three categorical slots (validated all-pairs, light and dark) for the arms; the untrained base as a neutral
#: dashed reference line, always labelled.
SERIES_STYLE = {"F": ("--s1", "#2a78d6", "#3987e5"), "FP": ("--s2", "#eb6834", "#d95926"), "v07": ("--s3", "#1baf7a", "#199e70"),
                "base": ("--ref", "#52514e", "#c3c2b7")}
SERIES_NAME = {"F": "v0.8 F (format)", "FP": "v0.8 FP (format + prior weights)", "v07": "v0.7", "base": "untrained base"}


def _x(d: float, left: float, width: float) -> float:
    """Signed distance → x: a symmetric log scale, |d| from 0.1 to 40 on each side of a gap at 0."""
    hi = math.log10(40 / 0.05)
    u = math.copysign(math.log10(abs(d) / 0.05), d)
    return left + width * (u + hi) / (2 * hi) if abs(d) >= 0.1 else left + width / 2


def curves_svg(series: Mapping[str, Mapping[str, Any]], *, seeds: Mapping[str, Sequence[Mapping[str, Any]]] | None = None,
               title: str = "A16s — the arm's distance sweep") -> str:
    """Two stacked panels — accuracy and mean P(correct option) — against the signed remaining distance (symmetric log scale; the rule's
    thresholds marked). `series` = {"F" | "FP" | "v07" | "base": curves(...)} (the arms' seed means and the base); `seeds` = the per-seed
    curves of each arm, drawn thin in the arm's hue. One y axis per panel (0–1), hairline grid, a legend, native hover tooltips on every point;
    light and dark through the SVG's own style."""
    width, panel_h, left, right, top = 760, 230, 64, 150, 78
    plot_w = width - left - right
    gap = 54
    height = top + 2 * panel_h + gap + 56
    order = [k for k in ("F", "FP", "v07", "base") if k in series]
    css = ["svg{font-family:system-ui,-apple-system,'Segoe UI',sans-serif;}", ".bg{fill:#fcfcfb}", ".ink{fill:#0b0b0b}", ".ink2{fill:#52514e}",
           ".mut{fill:#898781}", ".grid{stroke:#e1e0d9;stroke-width:1}", ".axis{stroke:#c3c2b7;stroke-width:1}", ".thr{stroke:#c3c2b7;stroke-width:1}"]
    dark = [".bg{fill:#1a1a19}", ".ink{fill:#ffffff}", ".ink2{fill:#c3c2b7}", ".grid{stroke:#2c2c2a}", ".axis{stroke:#383835}", ".thr{stroke:#383835}"]
    for key in order:
        var, light_hex, dark_hex = SERIES_STYLE[key]
        css.append(f".c-{key}{{stroke:{light_hex}}} .f-{key}{{fill:{light_hex}}}")
        dark.append(f".c-{key}{{stroke:{dark_hex}}} .f-{key}{{fill:{dark_hex}}}")
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" height="{height}" role="img" '
           f'aria-label="{html.escape(title)}: accuracy and P(correct option) against the remaining distance, per model">',
           "<style>" + "".join(css) + "@media (prefers-color-scheme: dark){" + "".join(dark) + "}</style>",
           f'<rect class="bg" x="0" y="0" width="{width}" height="{height}"/>',
           f'<text class="ink" x="{left}" y="24" font-size="15" font-weight="600">{html.escape(title)}</text>',
           f'<text class="ink2" x="{left}" y="42" font-size="12">Accuracy and P(the rule\'s option) against the remaining distance; '
           f'untrained base dashed. Hover a point for its value.</text>']
    # legend (one row)
    lx = left
    for key in order:
        dash = ' stroke-dasharray="6 4"' if key == "base" else ""
        out.append(f'<line class="c-{key}" x1="{lx}" y1="54" x2="{lx + 22}" y2="54" stroke-width="2"{dash} stroke-linecap="round"/>')
        out.append(f'<text class="ink2" x="{lx + 28}" y="58" font-size="12">{html.escape(SERIES_NAME[key])}</text>')
        lx += 40 + 6.2 * len(SERIES_NAME[key])
    ticks = [-40, -20, -8, -3, -1, -0.2, 0.2, 1, 3, 8, 20, 40]
    for p, (metric, name) in enumerate((("accuracy", "Accuracy (top option = the rule's option)"), ("p_correct", "Mean P(the rule's option)"))):
        y0 = top + 18 + p * (panel_h + gap)
        out.append(f'<text class="ink" x="{left}" y="{y0 - 6}" font-size="13" font-weight="600">{html.escape(name)}</text>')
        ph = panel_h - 20
        for v in (0, 0.25, 0.5, 0.75, 1.0):
            yy = y0 + ph * (1 - v)
            out.append(f'<line class="grid" x1="{left}" y1="{yy:.1f}" x2="{left + plot_w}" y2="{yy:.1f}"/>')
            out.append(f'<text class="mut" x="{left - 8}" y="{yy + 4:.1f}" font-size="11" text-anchor="end">{v:g}</text>')
        for t in ticks:
            xx = _x(t, left, plot_w)
            out.append(f'<line class="thr" x1="{xx:.1f}" y1="{y0}" x2="{xx:.1f}" y2="{y0 + ph}"/>')
            out.append(f'<text class="mut" x="{xx:.1f}" y="{y0 + ph + 14}" font-size="10" text-anchor="middle">{t:+g}</text>')
        out.append(f'<line class="axis" x1="{left}" y1="{y0 + ph}" x2="{left + plot_w}" y2="{y0 + ph}"/>')
        if p == 1:
            caption = "remaining distance (cm; stay ≤ 0.2 < fine < 1 ≤ small < 3 ≤ near < 8 ≤ mid < 20 ≤ far)"
            out.append(f'<text class="mut" x="{left + plot_w / 2}" y="{y0 + ph + 32}" font-size="11" text-anchor="middle">{html.escape(caption)}</text>')
        for key in [k for k in order if seeds and seeds.get(k)]:
            for seed_curve in seeds[key]:
                pts = [(p_["distance"], p_[metric]) for p_ in seed_curve.get("points") or [] if p_.get(metric) is not None]
                for side in (-1, 1):
                    seg = [(d, v) for d, v in pts if (d > 0) == (side > 0)]
                    if len(seg) > 1:
                        path = " ".join(f"{_x(d, left, plot_w):.1f},{y0 + ph * (1 - v):.1f}" for d, v in seg)
                        out.append(f'<polyline class="c-{key}" points="{path}" fill="none" stroke-width="1" stroke-opacity="0.45" stroke-linejoin="round"/>')
        ends = []
        for key in order:
            pts = [(p_["distance"], p_[metric], p_["n"]) for p_ in series[key].get("points") or [] if p_.get(metric) is not None]
            dash = ' stroke-dasharray="6 4"' if key == "base" else ""
            for side in (-1, 1):
                seg = [(d, v) for d, v, _ in pts if (d > 0) == (side > 0)]
                if len(seg) > 1:
                    path = " ".join(f"{_x(d, left, plot_w):.1f},{y0 + ph * (1 - v):.1f}" for d, v in seg)
                    out.append(f'<polyline class="c-{key}" points="{path}" fill="none" stroke-width="2" stroke-linejoin="round" stroke-linecap="round"{dash}/>')
            for d, v, n in pts:
                cx, cy = _x(d, left, plot_w), y0 + ph * (1 - v)
                out.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="6" fill="transparent"><title>{html.escape(SERIES_NAME[key])}, d = {d:+g} cm: '
                           f'{metric.replace("_", " ")} {v:.2f} (n {n})</title></circle>')
            right_pts = [(d, v) for d, v, _ in pts if d > 0]
            if right_pts:
                ends.append((y0 + ph * (1 - right_pts[-1][1]), key))
        ends.sort()
        if all(b[0] - a[0] >= 13 for a, b in zip(ends, ends[1:])):  # direct end labels only when they do not collide (else the legend)
            for yy, key in ends:
                out.append(f'<text class="ink2" x="{left + plot_w + 8}" y="{yy + 4:.1f}" font-size="11">{html.escape(SERIES_NAME[key].split(" (")[0])}</text>')
    out.append("</svg>")
    return "\n".join(out) + "\n"


def sha256_file(path: str | Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def check_frozen(directory: str | Path = A16S_DIR) -> list[str]:
    """Problems with the frozen A16s files (empty = unchanged since frozen)."""
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
