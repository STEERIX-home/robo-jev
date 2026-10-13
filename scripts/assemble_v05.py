#!/usr/bin/env python
"""Task v0.5a — the v0.5 training input: v0.4's input byte for byte, except one label change (hypothesis H-clarity, option C).

    # the whole assembly, one command (CPU unit, ≈ 10 min, ≤ 32 GB):
    python scripts/assemble_v05.py all --out-dir artifacts/simgen/v05
    # the steps one by one: labels (the transform + change report), then system1_v03.py overlap, then exposure

**The one change** (PRE-DECLARED v0.5 VERDICT RULE, progress.md). For every **MuJoCo** record — v0.3b's families and v0.4a's scenes,
``meta.set`` ``v03b-mujoco-train`` / ``v04a-mujoco-scenes``, cross-checked with ``generator.name`` and its three priors — and every
**yes/no** (``noul``) or **score** question whose band is not identical under the record's three priors (``meta.robustness[q].robust``
false; re-derived here from ``meta.labels_by_prior`` with :func:`robo_jev.bench.system1.band` and required to agree), the label becomes

    clip(mean of the three prior labels, 0.40, 0.60)        (the exact mean of the three values, rounded once to a float)

written into ``p_yes`` (yes/no) or ``value`` (score). The band [0.40, 0.60] lies inside the middle band [1/3, 2/3], so every
prior-dependent question is labelled "unclear — look again". A robust question keeps its label.

**Everything else is v0.4's, byte for byte**: every choice question; every non-MuJoCo record (engine-s1-pick incl. its
``meta.question_pick``, general, d1); the record order and ids; the whole ``meta`` of every record (``labels_by_prior`` and
``robustness`` keep the per-prior values); the second-pass repeats (``<id>~r2``), transformed exactly like their originals. A MuJoCo
record none of whose questions is prior-dependent keeps its line; a changed record is re-serialised in the style its line was written in
(checked: the unchanged parse re-serialises to the original bytes), so its line differs only in the changed numbers.

**Checks** (``labels`` writes them into ``train-v05.changes.json`` / ``.md``; ``all`` adds the rest to ``train-v05.assembly.json``):
only those labels change (every other field, the masked records and the id order hash identically); band counts before and after; the
sha256 of every unchanged part (engine, d1, general, the untouched MuJoCo lines) is identical; the trainer's exposure per seed (action
examples, derived questions, every category, steps) equals v0.4's — label values never enter the pick —, with the number of trained
examples whose label changed (the dose); and the sealed checks (T1–T7, G1, G2, T8a, T8b, T9, B1, H-set) give 0 hits.
"""

from __future__ import annotations

import argparse
import datetime as dt
import functools
import hashlib
import importlib.util
import json
import subprocess
import sys
from collections import Counter, defaultdict
from fractions import Fraction
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

SCRIPT_VERSION = "v05a-assemble-1"
#: The rule's name in manifests and change reports.
RULE_VERSION = "clarity-1"
CODE_PATHS = ("scripts/assemble_v05.py", "scripts/assemble_v04.py", "scripts/system1_v03.py", "src/robo_jev/bench/system1.py",
              "src/robo_jev/decisions/examples.py")
V04 = REPO / "artifacts" / "simgen" / "v04"
TRAIN_V04 = V04 / "train-v04.jsonl"
#: The v0.4 input (progress.md: FINAL v0.4 INPUT; train-v04.manifest.json).
TRAIN_V04_SHA = "89bca7153924bdced8347a0de1489dc9cd59b975bf4da5c36f7c80d99bf6da4d"
EXPOSURE_V04 = V04 / "train-v04.exposure.json"
OUT_DIR = REPO / "artifacts" / "simgen" / "v05"
TRAIN_CONFIG = REPO / "configs" / "train" / "decisions-v0-lora-h100.yaml"
B1 = REPO / "artifacts" / "b1" / "b1-infamily.jsonl"
HSET = V04 / "scene-holdout.jsonl"
#: The rule's clip band ("unclear — look again"), inside the middle band [1/3, 2/3].
CLIP_LOW, CLIP_HIGH = 0.40, 0.60
#: MuJoCo sets of the v0.4 input → their generator; every record of these sets carries the three priors below.
MUJOCO_GENERATORS = {"v03b-mujoco-train": "v03b-mujoco-train-v1", "v04a-mujoco-scenes": "v04a-mujoco-scenes-v1"}
PRIORS = ("base", "alt_a", "alt_b")
SCALAR_KEY = {"noul": "p_yes", "score": "value"}
BANDS = ("low", "middle", "high")
#: Places never written to.
NEVER_WRITE = (REPO / "artifacts" / "simgen" / "v03", V04, REPO / "artifacts" / "t8", REPO / "artifacts" / "t9", REPO / "artifacts" / "b1",
               REPO / "artifacts" / "reports" / "v04", REPO / "artifacts" / "simgen" / "v0b" / "tests", REPO / "artifacts" / "simgen" / "probes")


def _log(message: str) -> None:
    print(f"[v05a {dt.datetime.now().strftime('%H:%M:%S')}] {message}", file=sys.stderr, flush=True)


@functools.lru_cache(maxsize=None)
def _v04():
    """scripts/assemble_v04.py (its part classifier, exposure report, config reader and file helpers)."""
    spec = importlib.util.spec_from_file_location("v05_assemble_v04", REPO / "scripts" / "assemble_v04.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _git() -> dict[str, Any]:
    def run(*args: str) -> str | None:
        try:
            return subprocess.run(["git", "-C", str(REPO), *args], capture_output=True, text=True, check=True).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return None

    status = run("status", "--porcelain", "--", *CODE_PATHS)
    return {"commit": run("rev-parse", "--short", "HEAD"), "dirty": bool(status) if status is not None else None}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(payload, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    temp.replace(path)


def _raw_lines(path: Path) -> Iterator[bytes]:
    with Path(path).open("rb") as handle:
        for raw in handle:
            if raw.strip():
                yield raw if raw.endswith(b"\n") else raw + b"\n"


def refuse_write(path: Path) -> None:
    """Outputs never go to v0.3's or v0.4's files, the sealed sets, B1 or v0.4's reports."""
    resolved = Path(path).resolve()
    for place in NEVER_WRITE:
        place = place.resolve()
        if resolved == place or place in resolved.parents:
            raise SystemExit(f"{path}: refusing to write into {place}")


# --------------------------------------------------------------------------
# The rule (pure)
# --------------------------------------------------------------------------


def band(value: float) -> str:
    from robo_jev.bench.system1 import band as system1_band

    return system1_band(value)


def exact_mean(values: Sequence[float]) -> float:
    """The mean of floats computed exactly (as fractions) and rounded once — the mean of three equal values is that value."""
    return float(sum(Fraction(float(v)) for v in values) / len(values))


def clarity_value(values: Sequence[float]) -> float:
    """The rule's label of a prior-dependent question: clip(mean of the three prior labels, 0.40, 0.60) — the exact mean
    (:func:`exact_mean`), no other rounding."""
    if len(values) != len(PRIORS):
        raise ValueError(f"the rule averages the {len(PRIORS)} priors {PRIORS} (got {len(values)} values)")
    return min(CLIP_HIGH, max(CLIP_LOW, exact_mean(values)))


def mujoco_kind(record: Mapping[str, Any]) -> str | None:
    """The MuJoCo set of a record (``meta.set``, cross-checked with ``generator.name`` and ``generator.priors``), or None for any other
    record. A record that names a MuJoCo generator outside its set, or a MuJoCo set without its generator or priors, is refused."""
    meta = record.get("meta") or {}
    generator = record.get("generator") if isinstance(record.get("generator"), Mapping) else {}
    name = meta.get("set")
    gen_name = generator.get("name")
    if name in MUJOCO_GENERATORS:
        if gen_name != MUJOCO_GENERATORS[name] or list(generator.get("priors") or []) != list(PRIORS):
            raise ValueError(f"{record.get('id')}: meta.set {name!r} but generator {gen_name!r} with priors {generator.get('priors')!r} "
                             f"(want {MUJOCO_GENERATORS[name]!r} with {list(PRIORS)})")
        return str(name)
    if gen_name in MUJOCO_GENERATORS.values():
        raise ValueError(f"{record.get('id')}: generator {gen_name!r} outside its MuJoCo set (meta.set {name!r})")
    if isinstance(meta.get("labels_by_prior"), Mapping):
        raise ValueError(f"{record.get('id')}: labels by prior outside the MuJoCo sets (meta.set {name!r}) — the rule would not see it")
    return None


def question_plan(record: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Every yes/no and score question of a MuJoCo record: its prior labels, bands, robustness (recorded and re-derived, which must
    agree), the label (which must be the base prior's) and the rule's new value (None for a robust question)."""
    meta = record.get("meta") or {}
    rid = record.get("id")
    by_prior = meta.get("labels_by_prior")
    robustness = meta.get("robustness")
    if not isinstance(by_prior, Mapping) or sorted(by_prior) != sorted(PRIORS) or not isinstance(robustness, Mapping):
        raise ValueError(f"{rid}: a MuJoCo record needs meta.labels_by_prior under {list(PRIORS)} and meta.robustness")
    out = []
    for name, question in record["request"]["questions"].items():
        qtype = question.get("type")
        if qtype not in SCALAR_KEY:
            continue
        key = SCALAR_KEY[qtype]
        label = (record.get("labels") or {}).get(name)
        if not isinstance(label, Mapping) or set(label) != {"type", key} or label.get("type") != qtype:
            raise ValueError(f"{rid}/{name}: a {qtype} label must be {{'type', {key!r}}} (got {label!r})")
        values = []
        for prior in PRIORS:
            entry = (by_prior[prior] or {}).get(name)
            if not isinstance(entry, Mapping) or entry.get(key) is None:
                raise ValueError(f"{rid}/{name}: no {key} under prior {prior!r}")
            values.append(float(entry[key]))
        if dict(by_prior["base"][name]) != dict(label):
            raise ValueError(f"{rid}/{name}: the label is not the base prior's ({label!r} vs {by_prior['base'][name]!r})")
        bands = [band(v) for v in values]
        same = len(set(bands)) == 1
        recorded = robustness.get(name) or {}
        if recorded.get("robust") is not same or recorded.get("band_same", same) is not same:
            raise ValueError(f"{rid}/{name}: meta.robustness says robust={recorded.get('robust')!r} but the bands under {list(PRIORS)} are {bands}")
        out.append({"question": name, "type": qtype, "key": key, "old": float(label[key]), "priors": dict(zip(PRIORS, values)),
                    "bands": dict(zip(PRIORS, bands)), "robust": same, "new": None if same else clarity_value(values)})
    return out


def transform_record(record: Mapping[str, Any]) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    """(the transformed record — None when no label changes —, the question plan). Only ``labels[q][p_yes|value]`` of prior-dependent
    yes/no and score questions are written; nothing else (the input is not modified)."""
    plan = question_plan(record)
    changed = [entry for entry in plan if entry["new"] is not None and entry["new"] != entry["old"]]
    if not changed:
        return None, plan
    out = json.loads(json.dumps(record))
    for entry in changed:
        out["labels"][entry["question"]][entry["key"]] = entry["new"]
    return out, plan


# --------------------------------------------------------------------------
# Lines
# --------------------------------------------------------------------------

#: The JSON styles of the input's lines: the mix writes its second-pass copies with json.dumps(…, ensure_ascii=False); the generators'
#: files (copied byte for byte) use the same default separators; the engine's records use the compact form.
STYLES = {"default": {"ensure_ascii": False}, "compact": {"ensure_ascii": False, "separators": (",", ":"), "allow_nan": False}}


def line_style(raw: bytes, record: Mapping[str, Any]) -> str:
    """The style that re-serialises `record` to exactly `raw` (refuses a line no style reproduces)."""
    text = raw.decode("utf-8").rstrip("\n")
    for style, kwargs in STYLES.items():
        if json.dumps(record, **kwargs) == text:
            return style
    raise ValueError(f"{record.get('id')}: the line is in no known JSON style — refusing to re-serialise it")


def dumps_style(record: Mapping[str, Any], style: str) -> bytes:
    return (json.dumps(record, **STYLES[style]) + "\n").encode("utf-8")


def _masked(record: Mapping[str, Any], plan: Sequence[Mapping[str, Any]]) -> bytes:
    """The record with the label value of every prior-dependent yes/no and score question replaced by None — identical before and after
    the transform iff nothing else changed."""
    out = json.loads(json.dumps(record))
    for entry in plan:
        if not entry["robust"]:
            out["labels"][entry["question"]][entry["key"]] = None
    return json.dumps(out, ensure_ascii=False, sort_keys=True).encode("utf-8")


def _empty_bands() -> dict[str, dict[str, int]]:
    return {row: {col: 0 for col in BANDS} for row in BANDS}


def transform_file(in_path: Path, out_path: Path, *, mujoco_sets: Sequence[str] = tuple(MUJOCO_GENERATORS)) -> dict[str, Any]:
    """Stream `in_path` → `out_path` under the rule; returns the change report (counts, band tables, digests, checks). Every check that
    fails raises; `out_path` is replaced only when every check passed (the stream goes to a temporary file first)."""
    v04 = _v04()
    sets = set(mujoco_sets)
    digests = {side: defaultdict(hashlib.sha256) for side in ("before", "after")}
    counts: Counter = Counter()
    ids = {side: hashlib.sha256() for side in ("before", "after")}
    masked = {side: hashlib.sha256() for side in ("before", "after")}
    untouched = {side: hashlib.sha256() for side in ("before", "after")}
    choice = hashlib.sha256()
    bands_before = {"robust": Counter(), "prior_dependent": Counter()}
    bands_after = {"robust": Counter(), "prior_dependent": Counter()}
    transition = _empty_bands()
    plain_mean = _empty_bands()
    by_part: dict[str, Counter] = defaultdict(Counter)
    new_value_profile = Counter()
    originals: dict[str, dict[str, float]] = {}
    repeats: dict[str, tuple[str, dict[str, float]]] = {}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    temp = out_path.with_name(out_path.name + ".tmp")
    try:
        with temp.open("wb") as handle:
            for raw in _raw_lines(in_path):
                record = json.loads(raw)
                rid = str(record["id"])
                kind = mujoco_kind(record)
                part = "mujoco" if kind is not None and kind in sets else v04._part(record, sets)
                if kind is not None and kind not in sets:
                    raise ValueError(f"{rid}: MuJoCo set {kind!r} is not in the transform's sets {sorted(sets)}")
                if part == "other":
                    raise ValueError(f"{rid}: a record of no known part (engine, mujoco, d1, general)")
                counts[f"records:{part}"] += 1
                ids["before"].update(rid.encode("utf-8") + b"\n")
                if part != "mujoco":
                    handle.write(raw)
                    digests["before"][part].update(raw)
                    digests["after"][part].update(raw)
                    ids["after"].update(rid.encode("utf-8") + b"\n")
                    continue
                style = line_style(raw, record)
                counts[f"style:{style}"] += 1
                new, plan = transform_record(record)
                line = raw if new is None else dumps_style(new, style)
                after = record if new is None else json.loads(line)
                if new is not None:  # the new line holds exactly the new values, and nothing but those values moved
                    back = json.loads(line)
                    for entry in plan:
                        if entry["new"] is not None and back["labels"][entry["question"]][entry["key"]] != entry["new"]:
                            raise ValueError(f"{rid}/{entry['question']}: the written label is not the rule's value")
                handle.write(line)
                digests["before"]["mujoco"].update(raw)
                digests["after"]["mujoco"].update(line)
                ids["after"].update(str(after["id"]).encode("utf-8") + b"\n")
                masked["before"].update(_masked(record, plan) + b"\n")
                masked["after"].update(_masked(after, plan) + b"\n")
                for name, question in record["request"]["questions"].items():
                    if question.get("type") == "choice":
                        choice.update(json.dumps([rid, name, record["labels"][name], after["labels"][name]], sort_keys=True).encode("utf-8"))
                        if record["labels"][name] != after["labels"][name]:
                            raise ValueError(f"{rid}/{name}: a choice label changed")
                if new is None:
                    untouched["before"].update(raw)
                    untouched["after"].update(line)
                    counts["mujoco_records_unchanged"] += 1
                else:
                    counts["mujoco_records_changed"] += 1
                repeat_of = (record.get("meta") or {}).get("repeat_of")
                pass_name = "repeat" if repeat_of else "first"
                values = {e["question"]: (e["new"] if e["new"] is not None else e["old"]) for e in plan}
                if repeat_of:
                    repeats[rid] = (str(repeat_of), values)
                else:
                    originals[rid] = values
                for entry in plan:
                    old_band = band(entry["old"])
                    tag = "robust" if entry["robust"] else "prior_dependent"
                    new_value = entry["old"] if entry["new"] is None else entry["new"]
                    new_band = band(new_value)
                    bands_before[tag][old_band] += 1
                    bands_after[tag][new_band] += 1
                    tally = by_part[f"{pass_name}:{kind}:{entry['type']}"]
                    tally["questions"] += 1
                    tally[tag] += 1
                    if not entry["robust"]:
                        transition[old_band][new_band] += 1
                        plain_mean[old_band][band(exact_mean(list(entry["priors"].values())))] += 1
                        moved = entry["new"] != entry["old"]
                        tally["value_changed"] += int(moved)
                        tally[f"from_{old_band}"] += 1
                        counts["labels_changed"] += int(moved)
                        counts["prior_dependent_value_kept"] += int(not moved)
                        new_value_profile["clip_low_0.40" if entry["new"] == CLIP_LOW else "clip_high_0.60" if entry["new"] == CLIP_HIGH else "inside"] += 1
                    elif new_band != old_band or new_value != entry["old"]:
                        raise ValueError(f"{rid}/{entry['question']}: a robust label moved")
        mismatched = [rid for rid, (of, values) in repeats.items() if originals.get(of) != values]
        if mismatched:
            raise ValueError(f"{len(mismatched)} second-pass repeats are not transformed like their originals (first {mismatched[:3]})")
        if masked["before"].hexdigest() != masked["after"].hexdigest() or ids["before"].hexdigest() != ids["after"].hexdigest():
            raise ValueError("something besides the prior-dependent yes/no and score labels changed (masked records or the id order differ)")
        temp.replace(out_path)
    finally:
        if temp.exists():
            temp.unlink()
    parts = {part: {"records": counts[f"records:{part}"], "sha256_before": digests["before"][part].hexdigest(),
                    "sha256_after": digests["after"][part].hexdigest(),
                    "identical": digests["before"][part].hexdigest() == digests["after"][part].hexdigest()}
             for part in sorted({k.split(":", 1)[1] for k in counts if k.startswith("records:")})}
    total = {tag: sum(c.values()) for tag, c in bands_before.items()}
    return {
        "records": sum(v["records"] for v in parts.values()), "parts": parts,
        "mujoco": {
            "records": counts["records:mujoco"], "records_changed": counts["mujoco_records_changed"],
            "records_unchanged": counts["mujoco_records_unchanged"], "repeats": len(repeats), "repeats_transformed_like_originals": True,
            "line_styles": {k.split(":", 1)[1]: v for k, v in counts.items() if k.startswith("style:")},
            "questions": {"yes_no_and_score": total["robust"] + total["prior_dependent"], "robust": total["robust"],
                          "prior_dependent": total["prior_dependent"], "labels_changed": counts["labels_changed"],
                          "prior_dependent_value_already_the_rules": counts["prior_dependent_value_kept"]},
            "bands_before": {tag: {b: c[b] for b in BANDS} for tag, c in bands_before.items()},
            "bands_after": {tag: {b: c[b] for b in BANDS} for tag, c in bands_after.items()},
            "bands_before_all": {b: bands_before["robust"][b] + bands_before["prior_dependent"][b] for b in BANDS},
            "bands_after_all": {b: bands_after["robust"][b] + bands_after["prior_dependent"][b] for b in BANDS},
            "prior_dependent_transition": transition,
            "prior_dependent_new_values": dict(new_value_profile),
            "reference_plain_mean_band": {"what": "the band of the plain across-prior mean (no clip) by the label's band — the weaker "
                                                  "operationalisation the plan (§1) set aside", "rows": plain_mean},
            "by_pass_set_type": {k: dict(v) for k, v in sorted(by_part.items())},
        },
        "digests": {
            "id_order": {"before": ids["before"].hexdigest(), "after": ids["after"].hexdigest(), "identical": True},
            "mujoco_masked": {"what": "every MuJoCo record with the value of each prior-dependent yes/no / score label set to None (sorted keys)",
                              "before": masked["before"].hexdigest(), "after": masked["after"].hexdigest(), "identical": True},
            "mujoco_untouched_lines": {"what": "the lines of MuJoCo records with no prior-dependent yes/no / score question, file order",
                                       "records": counts["mujoco_records_unchanged"], "before": untouched["before"].hexdigest(),
                                       "after": untouched["after"].hexdigest(),
                                       "identical": untouched["before"].hexdigest() == untouched["after"].hexdigest()},
            "mujoco_choice_labels": {"what": "every MuJoCo choice label before and after (all equal; checked per question)", "sha256": choice.hexdigest()},
        },
    }


# --------------------------------------------------------------------------
# labels — the transform, its manifest and change report
# --------------------------------------------------------------------------

RULE_TEXT = ("PRE-DECLARED v0.5 VERDICT RULE (progress.md, 2026-10-13 ~21:35 KST): v0.4's training input byte-identical except the yes/no "
             "and score labels of prior-dependent MuJoCo questions (band not identical under the record's three priors), which become "
             "clip(mean over priors, 0.40, 0.60); robust questions, all choice questions, engine, general, d1, H-set, B1 unchanged")


def render_changes(report: Mapping[str, Any]) -> str:
    m = report["mujoco"]
    q = m["questions"]
    lines = ["# v0.5 training input — the one label change", "",
             f"`{report['out']['path']}` sha256 `{report['out']['sha256']}` ({report['records']:,} records) from `{report['input']['path']}` "
             f"sha256 `{report['input']['sha256']}`. Rule {RULE_VERSION}: {RULE_TEXT}.", "",
             "## Parts", "", "| part | records | sha256 before | sha256 after | identical |", "|---|---|---|---|---|"]
    for part, entry in report["parts"].items():
        lines.append(f"| {part} | {entry['records']:,} | `{entry['sha256_before'][:16]}…` | `{entry['sha256_after'][:16]}…` | {entry['identical']} |")
    d = report["digests"]
    lines += ["", f"Unchanged inside the MuJoCo part: the id order (`{d['id_order']['before'][:16]}…`, both files), every record with the "
              f"prior-dependent label values masked (`{d['mujoco_masked']['before'][:16]}…`, both), the {d['mujoco_untouched_lines']['records']:,} "
              f"lines of records without a prior-dependent yes/no or score question (`{d['mujoco_untouched_lines']['before'][:16]}…`, both), every "
              "choice label.", "",
              "## MuJoCo yes/no and score questions", "",
              f"{m['records']:,} records ({m['repeats']:,} second-pass repeats, each transformed like its original); {m['records_changed']:,} "
              f"records changed, {m['records_unchanged']:,} unchanged. Questions {q['yes_no_and_score']:,}: robust {q['robust']:,} (unchanged), "
              f"prior-dependent {q['prior_dependent']:,} → labels changed {q['labels_changed']:,} "
              f"({q['prior_dependent_value_already_the_rules']:,} already held the rule's value).", "",
              "| band | before: all | after: all | before: robust | after: robust | before: prior-dependent | after: prior-dependent |",
              "|---|---|---|---|---|---|---|"]
    for b in BANDS:
        lines.append(f"| {b} | {m['bands_before_all'][b]:,} | {m['bands_after_all'][b]:,} | {m['bands_before']['robust'][b]:,} | "
                     f"{m['bands_after']['robust'][b]:,} | {m['bands_before']['prior_dependent'][b]:,} | {m['bands_after']['prior_dependent'][b]:,} |")
    lines += ["", "Prior-dependent questions, the label's band before → after (rows: before):", "", "| before | → low | → middle | → high |",
              "|---|---|---|---|"]
    for row, cols in m["prior_dependent_transition"].items():
        lines.append(f"| {row} | " + " | ".join(f"{cols[c]:,}" for c in BANDS) + " |")
    nv = m["prior_dependent_new_values"]
    ref = m["reference_plain_mean_band"]["rows"]
    lines += ["", f"New values: clipped to 0.40 {nv.get('clip_low_0.40', 0):,}, clipped to 0.60 {nv.get('clip_high_0.60', 0):,}, inside "
              f"{nv.get('inside', 0):,}.", "", "Reference (not used): the plain across-prior mean's band — " + "; ".join(
                  f"{row} → " + ", ".join(f"{c} {cols[c]:,}" for c in BANDS) for row, cols in ref.items()) + ".", "",
              "By pass, set and type:", "", "| pass:set:type | questions | robust | prior-dependent | value changed | from low | from middle | from high |",
              "|---|---|---|---|---|---|---|---|"]
    for key, tally in m["by_pass_set_type"].items():
        lines.append(f"| {key} | {tally.get('questions', 0):,} | {tally.get('robust', 0):,} | {tally.get('prior_dependent', 0):,} | "
                     f"{tally.get('value_changed', 0):,} | {tally.get('from_low', 0):,} | {tally.get('from_middle', 0):,} | {tally.get('from_high', 0):,} |")
    return "\n".join(lines) + "\n"


def cmd_labels(args: argparse.Namespace) -> int:
    v04 = _v04()
    refuse_write(args.out)
    sha = v04._check_recorded(args.input, required=args.input.resolve() == TRAIN_V04.resolve())
    if args.input.resolve() == TRAIN_V04.resolve() and sha != TRAIN_V04_SHA:
        raise SystemExit(f"{args.input}: sha256 {sha[:12]}… is not the v0.4 input {TRAIN_V04_SHA[:12]}…")
    _log(f"labels: {args.input} ({sha[:12]}…) → {args.out}")
    try:
        report = transform_file(args.input, args.out, mujoco_sets=args.mujoco_set or tuple(MUJOCO_GENERATORS))
    except ValueError as error:
        raise SystemExit(f"refusing: {error}") from error
    out_sha = _sha256(args.out)
    head = {"kind": "v05-train-changes", "script_version": SCRIPT_VERSION, "rule_version": RULE_VERSION, "rule": RULE_TEXT,
            "clip": [CLIP_LOW, CLIP_HIGH], "priors": list(PRIORS), "mujoco_sets": dict(MUJOCO_GENERATORS), "git": _git(),
            "input": {"path": str(args.input), "sha256": sha}, "out": {"path": str(args.out), "sha256": out_sha, "bytes": args.out.stat().st_size}}
    report = {**head, **report, "created": dt.datetime.now().astimezone().isoformat(timespec="seconds")}
    stem = args.out.stem
    _write_json(args.out.with_name(f"{stem}.changes.json"), report)
    args.out.with_name(f"{stem}.changes.md").write_text(render_changes(report), encoding="utf-8")
    manifest = {"kind": "v05-train", "script_version": SCRIPT_VERSION, "git": _git(), "sha256": out_sha,
                "out": {"path": str(args.out), "sha256": out_sha, "bytes": args.out.stat().st_size, "records": report["records"]},
                "input": {"path": str(args.input), "sha256": sha, "manifest": str(args.input.with_name(args.input.stem + ".manifest.json"))},
                "rule": {"version": RULE_VERSION, "text": RULE_TEXT, "clip": [CLIP_LOW, CLIP_HIGH], "mean": "exact mean of the three prior labels (fractions), rounded once to a float",
                         "robust": "meta.robustness[q].robust (= band identical under base, alt_a, alt_b; re-derived and checked)",
                         "fields": "labels[q].p_yes (noul) / labels[q].value (score) of MuJoCo records only"},
                "parts": report["parts"], "changes": str(args.out.with_name(f"{stem}.changes.json")),
                "created": dt.datetime.now().astimezone().isoformat(timespec="seconds")}  # fmt: skip
    _write_json(args.out.with_name(f"{stem}.manifest.json"), manifest)
    q = report["mujoco"]["questions"]
    _log(f"labels: {report['records']} records, MuJoCo {report['mujoco']['records']} ({report['mujoco']['records_changed']} changed); "
         f"questions {q['yes_no_and_score']} = robust {q['robust']} + prior-dependent {q['prior_dependent']} (labels changed {q['labels_changed']}); "
         f"parts identical {[(p, e['identical']) for p, e in report['parts'].items()]} → {args.out} {out_sha[:16]}…")
    return 0


# --------------------------------------------------------------------------
# exposure — the trainer's view, v0.5 against v0.4, per seed
# --------------------------------------------------------------------------


def rule_plans(records: Sequence[Mapping[str, Any]], *, mujoco_sets: set[str]) -> dict[str, dict[str, dict[str, Any]]]:
    """The rule's question plan (:func:`question_plan`) of every MuJoCo record of the **untransformed** input: id → question → entry."""
    v04 = _v04()
    return {str(r["id"]): {e["question"]: e for e in question_plan(r)} for r in records if v04._part(r, mujoco_sets) == "mujoco"}


def dose(records: Sequence[Mapping[str, Any]], plans: Mapping[str, Mapping[str, Mapping[str, Any]]], *, config: Mapping[str, Any],
         seeds: Sequence[int], mujoco_sets: set[str]) -> dict[str, Any]:
    """Per seed, the trainer's epoch-0 MuJoCo examples of the **transformed** input (its own dev split and pick) by question type, and how
    many of them are yes/no or score questions whose label the rule changed (by the label's band before). `plans` = :func:`rule_plans` of
    the untransformed input; every transformed record must hold exactly the plan's labels (checked)."""
    from robo_jev.decisions.examples import record_pick, split_dev

    v04 = _v04()
    q = int(config["questions_per_request"])
    mujoco = [r for r in records if v04._part(r, mujoco_sets) == "mujoco"]
    if {str(r["id"]) for r in mujoco} != set(plans):
        raise ValueError("the transformed input's MuJoCo records are not the untransformed input's")
    for record in mujoco:
        for name, entry in plans[str(record["id"])].items():
            want = entry["old"] if entry["new"] is None else entry["new"]
            if record["labels"][name][entry["key"]] != want:
                raise ValueError(f"{record['id']}/{name}: label {record['labels'][name]!r} is not the rule's {want!r}")
    train, _ = split_dev(mujoco, fraction=config["dev_fraction"], salt=config["dev_salt"])
    out = {}
    for seed in seeds:
        tally: Counter = Counter()
        for record in train:
            for name in record_pick(record, q=q, seed=int(seed), epoch=0):
                qtype = record["request"]["questions"][name]["type"]
                tally[f"examples:{qtype}"] += 1
                entry = plans[str(record["id"])].get(name)
                if entry is not None and not entry["robust"]:
                    tally["prior_dependent"] += 1
                    tally[f"prior_dependent_from_{band(entry['old'])}"] += 1
                    tally["label_changed"] += int(entry["new"] != entry["old"])
        total = sum(v for k, v in tally.items() if k.startswith("examples:"))
        out[str(seed)] = {"mujoco_examples": total, **dict(sorted(tally.items())),
                          "label_changed_share_of_mujoco": tally["label_changed"] / total if total else None}
    return out


def cmd_exposure(args: argparse.Namespace) -> int:
    from robo_jev.decisions.train import load_training_records

    v04 = _v04()
    config = v04._config(args.train_config)
    sets = set(args.mujoco_set or MUJOCO_GENERATORS)
    reports = {}
    plans: dict[str, Any] = {}
    for name, path in (("v04", args.reference), ("v05", args.input)):
        records, validation = load_training_records(path)
        if validation["invalid"]:
            raise SystemExit(f"{path}: {validation['invalid']} records fail the trainer's validation")
        entry = v04.exposure_report(records, config=config, seeds=args.seeds, mujoco_sets=sets)
        entry["input"] = {"path": str(path), "sha256": _sha256(path), "validation": validation}
        if name == "v04":
            plans = rule_plans(records, mujoco_sets=sets)
        else:
            try:
                entry["dose"] = dose(records, plans, config=config, seeds=args.seeds, mujoco_sets=sets)
            except ValueError as error:
                raise SystemExit(f"refusing: {error}") from error
        reports[name] = entry
        del records
        _log(f"exposure {name}: " + "; ".join(f"seed {s}: {e['examples_per_epoch']} steps {e['steps']} action {e['action']} derived {e['derived']}"
                                               for s, e in entry["seeds"].items()))
    stored = json.loads(args.stored.read_text(encoding="utf-8")) if args.stored and args.stored.is_file() else None
    fields = ("examples_per_epoch", "steps", "engine_by_category", "action", "derived", "band", "comparison")
    same = {}
    for seed in map(str, args.seeds):
        mine, theirs = reports["v05"]["seeds"][seed], reports["v04"]["seeds"][seed]
        same[seed] = {"vs_v04_same_code": {f: mine[f] == theirs[f] for f in fields},
                      "vs_v04_stored": ({f: mine[f] == stored["seeds"][seed][f] for f in fields} if stored and seed in stored.get("seeds", {}) else None)}
    split_same = {k: reports["v05"][k] == reports["v04"][k] for k in ("records", "train_records", "dev_records", "dev_used", "dev_parts", "train_by_part")}
    out = {"kind": "v05-exposure", "script_version": SCRIPT_VERSION, "git": _git(), "train_config": str(args.train_config), "mujoco_sets": sorted(sets),
           "v05": reports["v05"], "v04": reports["v04"], "stored_v04": str(args.stored) if stored else None, "identical": same, "split_identical": split_same,
           "pass": all(all(v.values()) for s in same.values() for v in s.values() if v is not None) and all(split_same.values()),
           "created": dt.datetime.now().astimezone().isoformat(timespec="seconds")}
    _write_json(args.out, out)
    _log(f"exposure: identical to v0.4 per seed {same} · split {split_same} · dose {reports['v05']['dose']}")
    return 0 if out["pass"] else 2


# --------------------------------------------------------------------------
# all
# --------------------------------------------------------------------------


def _run(argv: list[str]) -> int:
    _log("$ " + " ".join(argv))
    return subprocess.run([sys.executable, *argv], cwd=REPO).returncode


def cmd_all(args: argparse.Namespace) -> int:
    out = args.out_dir
    refuse_write(out)
    out.mkdir(parents=True, exist_ok=True)
    train = out / "train-v05.jsonl"
    me = str(Path(__file__).resolve().relative_to(REPO))
    if _run([me, "labels", "--input", str(args.input), "--out", str(train)]) != 0:
        raise SystemExit("labels failed")
    missing = [str(p) for p in (B1, HSET) if not p.is_file()]
    if missing:
        raise SystemExit(f"sealed files missing: {missing} — every sealed set is checked")
    overlap = ["scripts/system1_v03.py", "overlap", "--input", str(train), "--out", str(out / "train-v05.overlap.json"),
               "--sealed-extra", f"B1={B1}", "--sealed-extra", f"H-set={HSET}", *sum((["--mujoco-set", s] for s in MUJOCO_GENERATORS), [])]
    overlap_rc = _run(overlap)
    exposure_rc = _run([me, "exposure", "--input", str(train), "--reference", str(args.input), "--stored", str(EXPOSURE_V04),
                        "--seeds", "0", "1", "--train-config", str(args.train_config), "--out", str(out / "train-v05.exposure.json")])
    changes = json.loads((out / "train-v05.changes.json").read_text(encoding="utf-8"))
    overlap_report = json.loads((out / "train-v05.overlap.json").read_text(encoding="utf-8"))
    exposure = json.loads((out / "train-v05.exposure.json").read_text(encoding="utf-8"))
    summary = {"kind": "v05-assembly", "script_version": SCRIPT_VERSION, "git": _git(),
               "files": {"train-v05.jsonl": {"path": str(train), "sha256": _sha256(train)}, "input": {"path": str(args.input), "sha256": _sha256(args.input)}},
               "unchanged_parts_identical": {p: e["identical"] for p, e in changes["parts"].items() if p != "mujoco"},
               "labels_changed": changes["mujoco"]["questions"]["labels_changed"], "prior_dependent": changes["mujoco"]["questions"]["prior_dependent"],
               "overlap": overlap_report["total_hits"], "sealed_checked": sorted(overlap_report["sealed"]),
               "exposure_identical_to_v04": exposure["pass"], "dose": exposure["v05"]["dose"],
               "pass": overlap_rc == 0 and exposure_rc == 0 and overlap_report["total_hits"] == 0 and exposure["pass"]
                       and all(e["identical"] for p, e in changes["parts"].items() if p != "mujoco"),
               "created": dt.datetime.now().astimezone().isoformat(timespec="seconds")}  # fmt: skip
    _write_json(out / "train-v05.assembly.json", summary)
    _log(f"done: {train} sha256 {summary['files']['train-v05.jsonl']['sha256'][:16]}… · overlap hits {summary['overlap']} · "
         f"exposure identical {summary['exposure_identical_to_v04']} · pass {summary['pass']}")
    return 0 if summary["pass"] else 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    labels = sub.add_parser("labels", help="train-v04 → train-v05 (+ manifest, change report): the one label change")
    labels.add_argument("--input", type=Path, default=TRAIN_V04)
    labels.add_argument("--out", type=Path, required=True)
    labels.add_argument("--mujoco-set", action="append", default=[], help="MuJoCo sets the rule applies to (default: v0.3b's and v0.4a's)")
    labels.set_defaults(func=cmd_labels)
    ex = sub.add_parser("exposure", help="the trainer's epoch-0 view of v0.5 and v0.4 per seed (must be identical) and the label-change dose")
    ex.add_argument("--input", type=Path, required=True)
    ex.add_argument("--reference", type=Path, default=TRAIN_V04)
    ex.add_argument("--stored", type=Path, default=EXPOSURE_V04, help="v0.4's recorded exposure report (also compared)")
    ex.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    ex.add_argument("--train-config", type=Path, default=TRAIN_CONFIG)
    ex.add_argument("--mujoco-set", action="append", default=[])
    ex.add_argument("--out", type=Path, required=True)
    ex.set_defaults(func=cmd_exposure)
    every = sub.add_parser("all", help="labels → sealed overlap check → exposure (one command)")
    every.add_argument("--input", type=Path, default=TRAIN_V04)
    every.add_argument("--train-config", type=Path, default=TRAIN_CONFIG)
    every.add_argument("--out-dir", type=Path, default=OUT_DIR)
    every.set_defaults(func=cmd_all)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
