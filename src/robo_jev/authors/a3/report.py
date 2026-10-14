"""A3 composition statistics and the data manifest (what the set holds — no model, no API)."""

from __future__ import annotations

import datetime as dt
import hashlib
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from robo_jev.authors.a3 import catalog as cat
from robo_jev.authors.a3 import labels as lab

__all__ = ["composition", "manifest", "truth_class"]


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def truth_class(record: Mapping[str, Any], name: str) -> str | None:
    """v0.5 call-view truth of a yes/no or score question: clear_safe / clear_risky / unclear."""
    label = record["labels"].get(name) or {}
    value = label.get("p_yes", label.get("value"))
    if value is None:
        return None
    robust = bool(((record["meta"].get("robustness") or {}).get(name) or {}).get("robust"))
    level = lab.band(value)
    if robust and level == "low":
        return "clear_safe"
    if robust and level == "high":
        return "clear_risky"
    return "unclear"


def _hist(values: Sequence[float]) -> list[int]:
    counts, _ = np.histogram(np.clip(np.asarray(values, dtype=float), 0, 1), bins=10, range=(0.0, 1.0))
    return [int(c) for c in counts]


def composition(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_scene: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for r in records:
        by_scene[r["meta"]["scene"]].append(r)
    out: dict[str, Any] = {"records": len(records), "scenes": {}}
    for scene, rows in by_scene.items():
        kinds = rows[0]["meta"]["question_kind"]
        inverse = {kind: name for name, kind in kinds.items()}
        bases = [r for r in rows if r["meta"]["pair"]["role"] == "base"]
        entry: dict[str, Any] = {"records": len(rows), "groups": len(bases)}
        entry["pairs"] = {
            "design": dict(Counter(r["meta"]["pair"]["design"] for r in bases)),
            "fact": dict(Counter(r["meta"]["pair"]["fact"] for r in bases)),
            "field": dict(Counter(r["meta"]["pair"]["field"] for r in bases)),
            "direction_base": dict(Counter(r["meta"]["pair"]["direction_by_prior"]["base"] for r in bases)),
            "direction_same_share": float(np.mean([r["meta"]["pair"]["direction_same"] for r in bases])),
            "moves_ge_0.1_share": float(np.mean([abs(r["meta"]["pair"]["key_delta"]) >= 0.1 - 1e-9 for r in bases])),
        }
        for kind in ("key", "severe", "score_cf"):
            name = inverse[kind]
            field = "value" if kind == "score_cf" else "p_yes"
            values = [float(r["labels"][name][field]) for r in rows]
            entry[kind] = {"mean": float(np.mean(values)), "deciles": _hist(values),
                           "bands": dict(Counter(lab.band(v) for v in values)),
                           "robust_share": float(np.mean([r["meta"]["robustness"][name]["robust"] for r in rows])),
                           "truth": dict(Counter(truth_class(r, name) for r in rows))}
        for kind in ("direction", "compare", "action", "outcome"):
            name = inverse[kind]
            argmax = Counter()
            for r in rows:
                probs = r["labels"][name]["probabilities"]
                order = list(r["request"]["questions"][name]["criteria"])
                top = max(probs.values())
                key = next(k for k in order if probs[k] >= top - 1e-9)
                if kind in ("compare", "action"):
                    key = "plan" if key == "as_written" else ("no_difference" if key == "no_difference" else "alternative")
                argmax[key] += 1
            entry[kind] = {"argmax": dict(argmax), "robust_share": float(np.mean([r["meta"]["robustness"][name]["robust"] for r in rows]))}
        aims = [(float(r["meta"]["static_aim"]["key"]), float(r["labels"][inverse["key"]]["p_yes"])) for r in rows]
        diff = np.array([b - a for a, b in aims])
        entry["aim_vs_label"] = {"mean_error": float(diff.mean()), "mean_abs_error": float(np.abs(diff).mean()),
                                 "corr": float(np.corrcoef(np.array(aims).T)[0, 1]) if len(aims) > 2 else None}
        entry["truth_all_scalar"] = dict(Counter(truth_class(r, inverse[k]) for r in rows for k in ("key", "severe", "score_cf")))
        entry["alternatives"] = dict(Counter(k for r in bases for k in r["meta"]["action_options"][1:]))
        out["scenes"][scene] = entry
    total_truth = Counter()
    for scene in out["scenes"].values():
        total_truth.update(scene["truth_all_scalar"])
    out["truth_all_scalar"] = dict(total_truth)
    return out


def manifest(records: Sequence[Mapping[str, Any]], path: Path, *, git: str | None, scenes: Sequence[str], labels_dir: Path) -> dict[str, Any]:
    ids = [r["id"] for r in records]
    return {
        "set": "v06-a3", "split": "train", "author": "a3", "file": path.name, "sha256": _sha(path), "records": len(records),
        "groups": len({(r["meta"]["scene"], r["generator"]["group"]) for r in records}), "unique_ids": len(set(ids)) == len(ids),
        "scenes": {s: sum(1 for r in records if r["meta"]["scene"] == s) for s in scenes},
        "id_pattern": "v06a3-<scene>-<group:06d>[p]", "generator": records[0]["generator"]["name"] if records else None, "git": git,
        "k": 32, "priors": list(cat.PRIORS), "training_label_prior": "base",
        "label_files": {p.name: _sha(p) for p in sorted(labels_dir.glob("*.jsonl"))},
        "created": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "infamily_registration": {
            "EVENT": {s: list(records_for(records, s)[0]["meta"]["key_events"]) for s in scenes if records_for(records, s)},
            "PLAN_KEY": {s: "as_written" for s in scenes},
        },
    }


def records_for(records: Sequence[Mapping[str, Any]], scene: str) -> list[Mapping[str, Any]]:
    return [r for r in records if r["meta"]["scene"] == scene]
