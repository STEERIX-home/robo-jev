"""Author A1 — from cached rollouts to ``train.jsonl``: records, the flat-key selection, the manifest and the stats.

**Selection.** More groups are labelled than kept (720 per scene for 625). Groups with any numerically unstable rollout are
dropped first; then groups are removed greedily, one at a time, so that the key labels of the kept records (both members of
every pair) are as close to flat as possible over ten equal bins of [0, 1] (least squares against the uniform count). Pairs
are never split.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from robo_jev.authors.a1 import build
from robo_jev.bench.system1 import band, is_clear

SCENE_ORDER = ("mast", "dock", "ramp")
BINS = 10


def load_cached(root: Path, scene: str) -> list[dict[str, Any]]:
    out = []
    for path in sorted((Path(root) / scene).glob("*.json")):
        out.append(json.loads(path.read_text()))
    return out


def _bin(x: float) -> int:
    return min(BINS - 1, int(float(x) * BINS))


def select_flat(key_pairs: Sequence[tuple[float, float]], keep: int) -> list[int]:
    """Indices of `keep` groups (each contributing two key labels) whose pooled key labels are closest to flat over
    :data:`BINS` bins — greedy removal (module docstring). Deterministic."""
    alive = list(range(len(key_pairs)))
    counts = np.zeros(BINS)
    for a, b in key_pairs:
        counts[_bin(a)] += 1
        counts[_bin(b)] += 1
    while len(alive) > keep:
        target = 2 * (len(alive) - 1) / BINS
        best, best_cost = None, None
        for pos, g in enumerate(alive):
            a, b = key_pairs[g]
            trial = counts.copy()
            trial[_bin(a)] -= 1
            trial[_bin(b)] -= 1
            cost = float(((trial - target) ** 2).sum())
            if best_cost is None or cost < best_cost - 1e-12:
                best, best_cost = pos, cost
        g = alive.pop(best)
        a, b = key_pairs[g]
        counts[_bin(a)] -= 1
        counts[_bin(b)] -= 1
    return alive


def build_records(root: Path, *, per_scene: int, git: str | None) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """All kept records (scene order, then group index) and the selection summary."""
    records: list[dict[str, Any]] = []
    summary: dict[str, Any] = {}
    for scene in SCENE_ORDER:
        cached = load_cached(root, scene)
        stable = [c for c in cached if int(c.get("unstable", 0)) == 0]
        groups = [build.records_of_group(c, git=git) for c in stable]
        key_q = groups[0][0]["meta"]["key_question"] if groups else None
        pairs = [(g[0]["labels"][key_q]["p_yes"], g[1]["labels"][key_q]["p_yes"]) for g in groups]
        kept = sorted(select_flat(pairs, min(per_scene, len(groups))))
        for i in kept:
            records.extend(groups[i])
        summary[scene] = {"labelled_groups": len(cached), "unstable_groups": len(cached) - len(stable), "kept_groups": len(kept),
                          "dropped_for_flatness": len(groups) - len(kept), "rollouts": int(sum(int(c["rollouts"]) for c in cached)),
                          "kept_indices_sha256": hashlib.sha256(json.dumps([groups[i][0]["generator"]["group"] for i in kept]).encode()).hexdigest()}
    return records, summary


def write_jsonl(path: Path, records: Iterable[Mapping[str, Any]]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=False) + "\n")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return digest


# --------------------------------------------------------------------------
# Stats
# --------------------------------------------------------------------------


def _hist(values: Sequence[float]) -> list[int]:
    return [int(x) for x in np.histogram(np.asarray(values, dtype=float), bins=BINS, range=(0.0, 1.0))[0]]


def stats(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Composition: records, key-label profile, bands per question kind, robust and clear shares, pairs, choice argmaxes."""
    out: dict[str, Any] = {"records": len(records), "by_scene": {}}
    by_scene: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for r in records:
        by_scene[r["meta"]["scene"]].append(r)
    for scene, recs in list(by_scene.items()) + [("all", list(records))]:
        kinds: dict[str, dict[str, Any]] = defaultdict(lambda: {"n": 0, "bands": Counter(), "robust": 0, "clear": 0, "values": []})
        argmax: dict[str, Counter] = defaultdict(Counter)
        robust_choice: Counter = Counter()
        for r in recs:
            meta = r["meta"]
            for name, kind in meta["question_kind"].items():
                label = r["labels"][name]
                rob = meta["robustness"][name]
                entry = kinds[kind]
                entry["n"] += 1
                entry["robust"] += int(bool(rob["robust"]))
                if label["type"] in ("noul", "score"):
                    y = float(label.get("p_yes", label.get("value")))
                    entry["bands"][band(y)] += 1
                    entry["clear"] += int(is_clear(y))
                    entry["values"].append(y)
                else:
                    probs = label["probabilities"]
                    top = max(probs, key=probs.get)
                    argmax[kind][top] += 1
                    robust_choice[kind] += int(bool(rob["robust"]))
        scene_out: dict[str, Any] = {"records": len(recs), "kinds": {}}
        for kind, entry in kinds.items():
            n = entry["n"]
            row: dict[str, Any] = {"n": n, "robust_share": round(entry["robust"] / n, 4)}
            if entry["values"]:
                row.update({"band_shares": {b: round(entry["bands"][b] / n, 4) for b in ("low", "middle", "high")},
                            "clear_share": round(entry["clear"] / n, 4), "hist10": _hist(entry["values"]), "mean": round(float(np.mean(entry["values"])), 4)})
            else:
                row["argmax"] = dict(argmax[kind])
            scene_out["kinds"][kind] = row
        pairs = [r for r in recs if r["meta"]["pair"]["role"] == "base"]
        scene_out["pairs"] = {
            "groups": len(pairs), "closeness": dict(Counter(r["meta"]["pair"]["closeness"] for r in pairs)),
            "fields": dict(Counter(r["meta"]["pair"]["field"] for r in pairs)), "facts": dict(Counter(r["meta"]["pair"]["fact"] for r in pairs)),
            "key_moves_ge_0.1": sum(1 for r in pairs if abs(r["meta"]["pair"]["key_delta"]) >= 0.1 - 1e-9),
            "direction_same_all_priors": sum(1 for r in pairs if r["meta"]["pair"]["direction_same"]),
        }
        scene_out["cf_kind"] = dict(Counter(r["meta"]["plan"]["cf_kind"] for r in recs))
        scene_out["outcome_mass_planned_base"] = {
            o: round(float(np.mean([r["meta"]["outcomes"]["planned"]["base"][o] for r in recs])), 4)
            for o in recs[0]["meta"]["outcomes"]["planned"]["base"]} if recs and scene != "all" else {}
        scene_out["robust_share_scalar"] = round(
            sum(int(bool(r["meta"]["robustness"][q]["robust"])) for r in recs for q, k in r["meta"]["question_kind"].items() if k in ("key", "severe", "score_cf"))
            / max(1, sum(1 for r in recs for k in r["meta"]["question_kind"].values() if k in ("key", "severe", "score_cf"))), 4)
        out["by_scene"][scene] = scene_out
    return out
