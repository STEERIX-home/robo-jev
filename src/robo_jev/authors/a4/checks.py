"""Author A4 — checks of the held-out test: validation, physics sanity, overlap.

* :func:`validate` — the trainer's loader reads and passes every record (:func:`robo_jev.decisions.records.validate_record`); ids, set,
  split, pairs and the contract fields are consistent; :func:`robo_jev.bench.infamily.robust_keys`, ``direction_items`` and
  ``comparison_items`` run on a sample with synthetic predictions (the labels as answers — every direction and comparison item must be
  right — and noisy answers over two listings), with this author's key events and plan key registered in the in-family tables for the
  call (``infamily.EVENT`` / ``infamily.PLAN_KEY`` are keyed by family; the records also carry them as ``meta.key_events`` /
  ``meta.plan_option``); the v0.5 call view (``scripts/verdict_v05.py``: truth classes, call items clustered by pair, call metrics with a
  bootstrap and a paired difference) runs.
* :func:`sanity` — monotonicity sweeps of the key label (and of the severe label where the physics is monotone) over a stated number
  with the record's own 32 base-prior draws (0 label-level violations wanted); the analytic threshold of each law against MuJoCo
  (canonical bisections: sliding at atan μ, tipping at atan(b/h), the centre of mass past the edge at L/2 − c, the haul's peak and
  steady tension against μ·m·g·cos θ − m·g·sin θ), and draw-level agreement of the labelled draws with the closed form away from the
  threshold; determinism (a re-run gives identical outcomes and identical records).
* :func:`overlap` — 0 shared ids, request / state / instruction hashes and sentences against T1–T7, G1, G2, T8a, T8b, T9, B1, the H-set
  and the v0.4 training input; no shared word 5-grams with T8b, T9 and the H-set; and **0 shared text units** in the v0.6 assembly's own
  definition (:func:`robo_jev.data.general_sets.record_units`: the whole state, every state line value with ≥ 5 words, every option text
  with ≥ 5 words, normalised) against all of those sets, the whole v0.4 training input and the training authors' files
  (``artifacts/v06/a{1,2,3}/train.jsonl``). The other sets are read in code only and nothing of their text is kept or reported: a hit is
  reported by count, by the unit's sha256, and by where it sits in this author's records (the state, or ``question.option``).
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterator
from unittest import mock

import numpy as np

from robo_jev.authors.a4 import build as B
from robo_jev.authors.a4 import design as D
from robo_jev.authors.a4 import label as L
from robo_jev.authors.a4 import sims as S

__all__ = ["overlap", "render_sanity", "sanity", "validate"]


# --------------------------------------------------------------------------
# validate
# --------------------------------------------------------------------------


def _synthetic(records: list[dict[str, Any]], *, noise: float, seed: int, listings: int) -> list[dict[str, Any]]:
    """Predictions in the diagnosis format: the labels plus noise, over `listings` listings per question."""
    from robo_jev.bench.system1 import oracle_predictions

    if noise == 0 and listings == 1:
        return oracle_predictions(records)
    rng = np.random.default_rng(seed)
    out = []
    for record in records:
        questions = {}
        for name, question in record["request"]["questions"].items():
            label = record["labels"][name]
            entries = []
            for _ in range(listings):
                if question["type"] == "choice":
                    keys = list(question["criteria"])
                    raw = np.array([float(label["probabilities"][k]) for k in keys]) + rng.uniform(0, noise, len(keys))
                    answer: dict[str, Any] = {"probabilities": {k: float(v) for k, v in zip(keys, raw / raw.sum())}}
                    options = keys
                else:
                    value = float(label.get("p_yes", label.get("value")))
                    value = float(np.clip(value + rng.normal(0, noise), 0, 1))
                    answer = {"p_yes": value} if question["type"] == "noul" else {"value": value, "bins": None}
                    options = None
                entries.append({"options": options, "values": None, "answer": answer})
            questions[name] = {"type": question["type"], "n_listings": listings, "answer": entries[0]["answer"], "listings": entries}
        out.append({"id": record["id"], "model": f"synthetic-{noise}", "questions": questions})
    return out


def _script(repo: Path, name: str):
    import importlib.util
    import sys

    spec = importlib.util.spec_from_file_location(f"a4check_{Path(name).stem}", repo / "scripts" / name)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def validate(path: Path, repo: Path, *, sample: int = 200) -> dict[str, Any]:
    from robo_jev.bench import infamily, system1
    from robo_jev.decisions.records import read_records, select_records, validate_record

    records = read_records(path)
    problems = {r["id"]: validate_record(r) for r in records}
    bad = {k: v for k, v in problems.items() if v}
    ids = [r["id"] for r in records]
    by_id = {r["id"]: r for r in records}
    contract = Counter()
    for r in records:
        meta = r["meta"]
        contract["prefix"] += not r["id"].startswith(B.ID_PREFIX + "-")
        contract["set"] += meta["set"] != B.SET_NAME
        contract["split"] += meta["split"] != B.SPLIT
        partner = by_id.get(meta["pair"]["of"])
        contract["pair_partner_missing"] += partner is None
        if partner is not None:
            contract["pair_partner_mismatch"] += partner["meta"]["pair"]["of"] != r["id"] or partner["generator"]["group"] != r["generator"]["group"]
        contract["labels_not_base"] += r["labels"] != meta["labels_by_prior"]["base"]
        contract["question_kind_mismatch"] += set(meta["question_kind"]) != set(r["request"]["questions"])
        for field in ("question_kind", "key_question", "labels_by_prior", "robustness", "pair", "plan", "outcomes", "variants", "compare_option",
                      "action_options", "hidden_prior", "prior_settings", "static_aim", "contrast_direction_by_prior"):
            contract[f"missing_{field}"] += field not in meta
    contract = {k: v for k, v in contract.items() if v}
    picked = select_records(records, sample, seed=0)
    oracle = _synthetic(picked, noise=0.0, seed=0, listings=1)
    noisy = _synthetic(picked, noise=0.25, seed=1, listings=2)
    tables = {"EVENT": {s: tuple(v) for s, v in D.EVENT.items()}, "PLAN_KEY": dict(D.PLAN_KEY)}
    with mock.patch.dict(infamily.EVENT, tables["EVENT"]), mock.patch.dict(infamily.PLAN_KEY, tables["PLAN_KEY"]):
        robust = infamily.robust_keys(records)
        direction_oracle = infamily.direction_items("A4", picked, oracle)
        comparison_oracle = infamily.comparison_items("A4", picked, oracle)
        direction_noisy = infamily.direction_items("A4", picked, noisy)
        comparison_noisy = infamily.comparison_items("A4", picked, noisy)
        comparison_all = infamily.comparison_items("A4", records, _synthetic(records, noise=0.0, seed=0, listings=1))
    direction_all = infamily.direction_items("A4", records, _synthetic(records, noise=0.0, seed=0, listings=1))
    verdict = _script(repo, "verdict_v05.py")
    calls_a = verdict.set_calls("A4", records, _synthetic(records, noise=0.2, seed=2, listings=2))
    calls_b = verdict.set_calls("A4", records, _synthetic(records, noise=0.3, seed=3, listings=2))
    metrics = verdict.call_metrics(calls_a, n_boot=200, seed=0)
    diff = verdict.call_difference(calls_a, calls_b, n_boot=200, seed=0)
    classes = Counter(verdict.truth_class(r, q) for r in records for q in r["request"]["questions"] if verdict.truth_class(r, q))
    s1 = system1.set_items("A4", picked, oracle)

    def mean(items: dict) -> float | None:
        return float(np.mean([v for _, v in items.values()])) if items else None

    summary = {
        "records": len(records), "loader_problems": len(bad), "unique_ids": len(set(ids)) == len(ids), "contract_problems": contract,
        "robust_keys": len(robust), "yes_no_and_score_questions": sum(1 for r in records for q in r["request"]["questions"].values() if q["type"] in ("noul", "score")),
        "sample_records": len(picked),
        "direction_items_oracle": {"n": len(direction_oracle), "mean": mean(direction_oracle)},
        "comparison_items_oracle": {"n": len(comparison_oracle), "mean": mean(comparison_oracle)},
        "direction_items_noisy": {"n": len(direction_noisy), "mean": mean(direction_noisy)},
        "comparison_items_noisy": {"n": len(comparison_noisy), "mean": mean(comparison_noisy)},
        "direction_items_all_oracle": {"n": len(direction_all), "mean": mean(direction_all)},
        "comparison_items_all_oracle": {"n": len(comparison_all), "mean": mean(comparison_all)},
        "call_view": {"truth_classes": dict(classes), "items": len(calls_a), "clusters": metrics.get("clusters"),
                      "DC": metrics["DC"], "GAP": metrics["GAP"], "CA": metrics["CA"], "paired_difference_DC": diff["DC"]},
        "system1_metrics_with_items": sorted(s1),
        "in_family_tables_needed": tables,
    }
    summary["pass"] = bool(not bad and summary["unique_ids"] and not contract and direction_oracle and comparison_oracle
                           and summary["direction_items_oracle"]["mean"] == 1.0 and summary["comparison_items_oracle"]["mean"] == 1.0
                           and summary["direction_items_all_oracle"]["mean"] == 1.0 and summary["comparison_items_all_oracle"]["mean"] == 1.0
                           and len(calls_a) == summary["yes_no_and_score_questions"])
    return {"summary": summary, "loader_problems": dict(list(bad.items())[:20])}


# --------------------------------------------------------------------------
# sanity
# --------------------------------------------------------------------------

#: Monotonicity sweeps: (scene, field, values as offsets or factors from the record's value, the event, expected direction).
SWEEPS = {
    "bank": (("slope", "abs", (-8, -6, -4, -2, 0, 2, 4, 6, 8), "key", +1), ("slope", "abs", (-8, -6, -4, -2, 0, 2, 4, 6, 8), "severe", +1),
             ("height", "rel", (0.7, 0.8, 0.9, 1.0, 1.15, 1.3, 1.5), "key", +1)),
    "ledge": (("overhang", "abs", (-0.16, -0.12, -0.08, -0.04, 0, 0.04, 0.08, 0.12, 0.16), "key", +1),
              ("length", "rel", (0.8, 0.9, 1.0, 1.1, 1.2), "key", -1), ("drop", "abs", (-0.1, -0.05, 0, 0.05, 0.1, 0.2), "severe", +1)),
    "anchor": (("load_mass", "rel", (0.5, 0.65, 0.8, 0.9, 1.0, 1.1, 1.25, 1.5, 2.0), "key", +1),
               ("load_mass", "rel", (0.5, 0.65, 0.8, 0.9, 1.0, 1.1, 1.25, 1.5, 2.0), "severe", +1),
               ("slope", "abs", (-4, -2, 0, 2, 4), "key", +1), ("start_time", "rel", (0.5, 0.75, 1.0, 1.5, 2.0, 3.0), "key", -1)),
}


def _valid(scene: str, vis: dict[str, Any]) -> bool:
    if scene == "bank":
        return 0.0 <= vis["slope"] <= 45 and vis["height"] > 0.05
    if scene == "ledge":
        return 0.0 <= vis["overhang"] < 0.95 * vis["length"] and vis["drop"] > 0.05
    return vis["load_mass"] > 0 and 1 <= vis["slope"] <= 25 and vis["start_time"] > 0.05


def _sweep_task(task: tuple[str, dict[str, Any], str, str, tuple, str, int, int]) -> dict[str, Any]:
    scene, vis, field, mode, steps, event, sign, draws_seed = task
    points = D.draw_points(draws_seed, scene)
    values, shares = [], []
    for step in steps:
        facts = dict(vis)
        facts[field] = round(vis[field] + step, 6) if mode == "abs" else round(vis[field] * step, 6)
        if not _valid(scene, facts):
            continue
        hid = D.hidden_values(scene, D.prior_specs(scene, facts)["base"], points)
        phys = D.variant_facts(scene, facts, "planned", {})
        outcomes, _ = L._run(scene, phys, hid, {})
        events = D.EVENT[scene] if event == "key" else D.SEVERE[scene]
        values.append(facts[field])
        shares.append(sum(o in events for o in outcomes) / len(outcomes))
    violations = [(values[i], values[i + 1], shares[i], shares[i + 1]) for i in range(len(shares) - 1) if sign * (shares[i + 1] - shares[i]) < -1e-12]
    return {"scene": scene, "field": field, "event": event, "sign": sign, "values": values, "shares": shares, "violations": violations}


def _threshold_tasks() -> list[tuple[str, dict[str, Any]]]:
    tasks = []
    for mu in (0.3, 0.5, 0.7):
        tasks.append(("bank_slide", {"mu": mu, "dims": (0.40, 0.60, 0.40), "com_z": 0.0}))
    for com_z, mu in ((0.0, 0.95), (0.06, 0.95), (0.12, 0.95)):
        tasks.append(("bank_tip", {"mu": mu, "dims": (0.30, 0.60, 0.50), "com_z": com_z}))
    for cx in (-0.1, 0.0, 0.1):
        tasks.append(("ledge_edge", {"cx": cx, "dims": (1.0, 0.6, 0.5), "drop": 0.3, "mu": 0.5}))
    for mu, accel in ((0.45, 0.8), (0.65, 0.4)):
        tasks.append(("anchor_peak", {"mu": mu, "accel": accel}))
        tasks.append(("anchor_steady", {"mu": mu, "accel": accel}))
    return tasks


def _threshold_task(task: tuple[str, dict[str, Any]]) -> dict[str, Any]:
    """Bisect MuJoCo's transition and compare it with the closed form."""
    kind, p = task
    if kind in ("bank_slide", "bank_tip"):
        down, _, height = p["dims"]
        ratio = (down / 2) / (height / 2 + p["com_z"])
        analytic = math.degrees(math.atan(p["mu"] if kind == "bank_slide" else ratio))
        want = "slides_down" if kind == "bank_slide" else "tips_over"

        def event(x: float) -> bool:
            return S.run_bank(S.bank_scene(p["dims"], 20.0, (0.0, 0.0, p["com_z"])), slope_deg=x, mu=p["mu"])["outcome"] == want

        lo, hi = analytic - 6.0, analytic + 6.0
        statement = "slides at atan(mu)" if kind == "bank_slide" else "tips at atan(b/h) when mu > b/h"
    elif kind == "ledge_edge":
        analytic = p["dims"][0] / 2 - p["cx"]

        def event(x: float) -> bool:
            return S.run_ledge(S.ledge_scene(p["dims"], 100.0, p["drop"], (p["cx"], 0.0, 0.0), x), mu=p["mu"])["outcome"] != "holds"

        lo, hi = analytic - 0.1, analytic + 0.1
        statement = "the nose drops once the centre of mass is past the edge: overhang > L/2 - c"
    else:
        theta, m_r, drag, speed = 10.0, 400.0, 0.04, 0.4
        g = 9.81
        capacity = (p["mu"] * math.cos(math.radians(theta)) - math.sin(math.radians(theta))) * m_r * g
        per_kg = g * (math.sin(math.radians(theta)) + drag * math.cos(math.radians(theta)))
        analytic = capacity / (per_kg + (p["accel"] if kind == "anchor_peak" else 0.0))
        rig = S.anchor_scene((1.2, 0.8, 0.6), m_r)

        def event(x: float) -> bool:
            out = S.run_anchor(rig, robot_slope_deg=theta, mu=p["mu"], load_mass=x, load_slope_deg=theta, drag=drag, start_accel=p["accel"],
                               line_speed=speed)["outcome"]
            return out != "holds_fast" if kind == "anchor_peak" else out == "dragged"

        lo, hi = analytic * 0.7, analytic * 1.3
        statement = ("slips (>= 5 mm) once the peak haul tension exceeds mu m g cos - m g sin" if kind == "anchor_peak"
                     else "dragged once the steady haul tension exceeds mu m g cos - m g sin (one-sided: a long slide during the start also "
                          "counts as dragged, so MuJoCo's threshold may sit below the closed form)")
    if event(lo) or not event(hi):
        return {"kind": kind, "params": p, "analytic": analytic, "mujoco": None, "statement": statement, "bracket_failed": True}
    for _ in range(14):
        mid = (lo + hi) / 2
        if event(mid):
            hi = mid
        else:
            lo = mid
    found = (lo + hi) / 2
    return {"kind": kind, "params": {k: list(v) if isinstance(v, tuple) else v for k, v in p.items()}, "analytic": analytic, "mujoco": found,
            "relative_gap": (found - analytic) / analytic, "statement": statement}


def _agreement(work: list[dict[str, Any]]) -> dict[str, Any]:
    """Draw-level agreement of the labelled planned draws (base prior) with the closed form (bank, ledge key; anchor key and severe),
    overall and away from the threshold."""
    out: dict[str, Any] = {}
    for scene in D.SCENES:
        agree = total = agree_far = total_far = 0
        sev_agree = sev_total = 0
        for w in work:
            if w["scene"] != scene or not w.get("group"):
                continue
            g = w["group"]
            points = D.draw_points(g["draws_seed"], scene)
            for member, vis in enumerate(g["records"]):
                hid = D.hidden_values(scene, D.prior_specs(scene, vis)["base"], points)
                phys = D.variant_facts(scene, vis, "planned", {})
                key, severe = D.key_analytic(scene, phys, hid)
                margin = _margin(scene, phys, hid)
                outcomes = w["sim"]["draws"][member]["planned"]["base"]
                for k, o in enumerate(outcomes):
                    hit = (o in D.EVENT[scene]) == bool(key[k])
                    agree += hit
                    total += 1
                    if margin[k] > 0.15:
                        agree_far += hit
                        total_far += 1
                    if scene == "anchor":
                        sev_agree += (o in D.SEVERE[scene]) == bool(severe[k])
                        sev_total += 1
        out[scene] = {"draws": total, "key_agreement": agree / max(1, total), "draws_away_from_threshold": total_far,
                      "key_agreement_away_from_threshold": agree_far / max(1, total_far)}
        if sev_total:
            out[scene]["severe_agreement"] = sev_agree / sev_total
    return out


def _margin(scene: str, phys: dict[str, Any], hid: dict[str, np.ndarray]) -> np.ndarray:
    """Relative distance of each draw from its key threshold (0 = on it)."""
    if scene == "bank":
        down, _, height = phys["dims"]
        ratio = np.maximum(down / 2 - hid["com_down"] * down, 1e-6) / np.maximum(height / 2 + hid["com_z"], 1e-6)
        tan = np.tan(np.radians(np.clip(phys["slope"] + hid["slope_dev"], 0, 89)))
        limit = np.minimum(hid["mu"], ratio)
        return np.abs(tan - limit) / limit
    if scene == "ledge":
        cx = -hid["com_x"] if phys["mirror"] else hid["com_x"]
        return np.abs(phys["overhang"] + hid["stop_dev"] - (phys["dims"][0] / 2 - cx)) / (phys["dims"][0] / 2)
    g = 9.81
    th_r, th_l = math.radians(phys["robot_slope"]), math.radians(phys["load_slope"])
    capacity = (hid["mu"] * math.cos(th_r) - math.sin(th_r)) * phys["mass"] * g
    load = phys["load_mass"] * hid["mass_ratio"]
    peak = load * g * (math.sin(th_l) + hid["drag"] * math.cos(th_l)) + load * phys["accel"]
    return np.abs(peak - capacity) / np.maximum(np.abs(capacity), 1.0)


def _determinism(work: list[dict[str, Any]], *, git: str | None) -> dict[str, Any]:
    out = {}
    for scene in D.SCENES:
        w = next(x for x in work if x["scene"] == scene and x.get("group"))
        group = L.build_group(scene, w["index"], w["seed"])
        same_group = json.dumps(group, sort_keys=True, default=float) == json.dumps(w["group"], sort_keys=True, default=float)
        sim = L.simulate_group(group)
        same_draws = sim["draws"] == w["sim"]["draws"]
        a = B.group_records(group, sim, git=git)
        b = B.group_records(copy.deepcopy(w["group"]), w["sim"], git=git)
        out[scene] = {"group": w["index"], "same_group": same_group, "same_draws": same_draws,
                      "same_records": json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)}
    return out


def sanity(*, workers: int, input_path: Path, scenarios: int = 6) -> dict[str, Any]:
    from multiprocessing import get_context

    from robo_jev.decisions.records import read_records

    records = [r for r in read_records(input_path) if r["meta"]["pair"]["role"] == "base"]
    work_dir = input_path.parent / "work"
    work = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(work_dir.glob("*/*.json"))]
    rng = np.random.default_rng(0)
    tasks = []
    for scene in D.SCENES:
        pool = [r for r in records if r["meta"]["scene"] == scene]
        chosen = [pool[i] for i in sorted(rng.choice(len(pool), size=min(scenarios, len(pool)), replace=False))]
        for r in chosen:
            for field, mode, steps, event, sign in SWEEPS[scene]:
                tasks.append((scene, dict(r["meta"]["visible"]), field, mode, steps, event, sign, int(r["generator"]["draws_seed"])))
    with get_context("fork").Pool(workers) as pool:
        sweeps = pool.map(_sweep_task, tasks)
        thresholds = pool.map(_threshold_task, _threshold_tasks())
    agreement = _agreement(work)
    determinism = _determinism(work, git=None)
    violations = sum(len(s["violations"]) for s in sweeps)
    summary = {
        "sweeps": len(sweeps), "sweep_points": sum(len(s["values"]) for s in sweeps), "label_level_violations": violations,
        "violations_by": dict(Counter(f"{s['scene']}:{s['field']}:{s['event']}" for s in sweeps for _ in s["violations"])),
        "thresholds": [{k: t[k] for k in ("kind", "analytic", "mujoco", "statement") if k in t} | {"relative_gap": t.get("relative_gap")} for t in thresholds],
        "max_threshold_gap": max(abs(t.get("relative_gap") or 1.0) for t in thresholds if t["kind"] != "anchor_steady"),
        "anchor_steady_one_sided": all((t.get("relative_gap") or 1.0) <= 0.05 for t in thresholds if t["kind"] == "anchor_steady"),
        "agreement": agreement, "determinism": determinism,
    }
    summary["pass"] = bool(violations == 0 and summary["max_threshold_gap"] < 0.05 and summary["anchor_steady_one_sided"] and all(all(v.values()) for v in
                           ({k: x for k, x in d.items() if k != "group"} for d in determinism.values())))
    return {"summary": summary, "sweeps": sweeps, "thresholds": thresholds}


def render_sanity(report: dict[str, Any]) -> str:
    s = report["summary"]
    lines = ["# A4 physics sanity", "", f"- sweeps: {s['sweeps']} ({s['sweep_points']} points, 32 base-prior draws each); label-level violations: "
             f"**{s['label_level_violations']}** {s['violations_by'] or ''}", "", "## Analytic thresholds (MuJoCo bisection vs closed form)", "",
             "| law check | closed form | MuJoCo | gap | rule |", "|---|---|---|---|---|"]
    for t in s["thresholds"]:
        gap = t.get("relative_gap")
        lines.append(f"| {t['kind']} | {t['analytic']:.4g} | {('%.4g' % t['mujoco']) if t.get('mujoco') is not None else '—'} | "
                     f"{('%+.1f %%' % (100 * gap)) if gap is not None else '—'} | {t['statement']} |")
    lines += ["", "## Draw-level agreement with the closed form (planned, base prior)", "", "| scene | draws | key agreement | away from threshold (> 15 %) | severe |",
              "|---|---|---|---|---|"]
    for scene, a in s["agreement"].items():
        lines.append(f"| {scene} | {a['draws']} | {a['key_agreement']:.3f} | {a['key_agreement_away_from_threshold']:.3f} ({a['draws_away_from_threshold']}) | "
                     f"{a.get('severe_agreement', float('nan')):.3f} |")
    lines += ["", "## Determinism", ""] + [f"- {scene}: {d}" for scene, d in s["determinism"].items()]
    lines += ["", "## Sweeps", "", "| scene | field | event | values | shares | violations |", "|---|---|---|---|---|---|"]
    for sw in report["sweeps"]:
        lines.append(f"| {sw['scene']} | {sw['field']} | {sw['event']} | {', '.join('%g' % v for v in sw['values'])} | "
                     f"{', '.join('%.2f' % v for v in sw['shares'])} | {len(sw['violations'])} |")
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------
# overlap
# --------------------------------------------------------------------------

_SENTENCE = re.compile(r"(?<=[.?!])\s+|\n+")
_WORD = re.compile(r"[a-z0-9]+(?:[.'-][a-z0-9]+)*")


def _h(text: str) -> bytes:
    return hashlib.sha256(text.encode("utf-8")).digest()[:16]


def _norm(text: str) -> str:
    return " ".join(str(text).lower().split())


def _texts(record: dict[str, Any]) -> Iterator[str]:
    request = record.get("request") or {}
    if isinstance(request.get("state"), str):
        yield request["state"]
    for question in (request.get("questions") or {}).values():
        if not isinstance(question, dict):
            continue
        if isinstance(question.get("instructions"), str):
            yield question["instructions"]
        criteria = question.get("criteria")
        if isinstance(criteria, dict):
            yield from (str(v) for v in criteria.values())
        elif isinstance(criteria, list):
            yield from (str(v) for v in criteria)


def _sentences(record: dict[str, Any]) -> set[str]:
    out = set()
    for text in _texts(record):
        for part in _SENTENCE.split(text):
            part = _norm(part)
            if len(part.split()) >= 2:
                out.add(part)
    return out


def _grams(record: dict[str, Any], n: int = 5) -> set[tuple[str, ...]]:
    out = set()
    for text in _texts(record):
        words = _WORD.findall(text.lower())
        out |= {tuple(words[i: i + n]) for i in range(len(words) - n + 1)}
    return out


def _signature(record: dict[str, Any]) -> dict[str, set[bytes]]:
    request = record.get("request") or {}
    instr = {_h(_norm(q.get("instructions", ""))) for q in (request.get("questions") or {}).values() if isinstance(q, dict) and q.get("instructions")}
    return {"ids": {str(record.get("id")).encode()}, "requests": {_h(json.dumps(request, sort_keys=True, ensure_ascii=False))},
            "states": {_h(_norm(request.get("state", "")))} if request.get("state") else set(), "instructions": instr}


def _stream(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


#: The training authors' files (v0.6) — checked for shared text units only (the assembly's definition).
AUTHOR_TRAIN = {"A1-train": ("artifacts", "v06", "a1", "train.jsonl"), "A2-train": ("artifacts", "v06", "a2", "train.jsonl"),
                "A3-train": ("artifacts", "v06", "a3", "train.jsonl")}


def _unit_hash(unit: str) -> str:
    """The assembly's unit hash (sha256 hex of the normalised unit) — comparable with its manifests."""
    return hashlib.sha256(unit.encode("utf-8")).hexdigest()


def my_unit_index(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """unit hash → {"records": [my record ids], "where": sorted locations} for every text unit of my records (``record_units``); a location
    is ``state`` or ``<question>.<option key>``. Holds my own text only, never another set's."""
    from robo_jev.data.general_sets import raw_units, record_units, text_units

    out: dict[str, dict[str, Any]] = {}
    for record in records:
        request = record["request"]
        located: dict[str, set[str]] = {}
        for unit in text_units(str(request.get("state") or "")):
            located.setdefault(_unit_hash(unit), set()).add("state")
        for name, question in request["questions"].items():
            for key, text in (question.get("criteria") or {}).items() if isinstance(question.get("criteria"), dict) else ():
                for unit in raw_units([str(text)]):
                    located.setdefault(_unit_hash(unit), set()).add(f"{name}.{key}")
        units = {_unit_hash(u) for u in record_units(record)}
        if units != set(located):
            raise ValueError(f"{record['id']}: unit locations do not cover record_units")
        for h, where in located.items():
            entry = out.setdefault(h, {"records": [], "where": set()})
            entry["records"].append(str(record["id"]))
            entry["where"] |= where
    return out


def unit_overlap(mine: dict[str, dict[str, Any]], refs: dict[str, Path]) -> dict[str, Any]:
    """Shared text units (hashes) between my index (:func:`my_unit_index`) and each reference file, streamed; only counts and hashes leave
    this function."""
    from robo_jev.data.general_sets import record_units

    per: dict[str, Any] = {}
    hits: dict[str, dict[str, int]] = {}
    for name, ref in refs.items():
        if not ref.is_file():
            per[name] = {"absent": str(ref)}
            continue
        shared: Counter = Counter()
        count = 0
        for record in _stream(ref):
            count += 1
            for unit in record_units(record):
                h = _unit_hash(unit)
                if h in mine:
                    shared[h] += 1
        per[name] = {"records": count, "shared_units": len(shared), "their_records_with_a_shared_unit_occurrences": int(sum(shared.values())),
                     "my_records_with_a_shared_unit": len({rid for h in shared for rid in mine[h]["records"]})}
        for h, n in shared.items():
            hits.setdefault(h, {})[name] = n
    shared_units = [{"sha256": h, "where_in_a4": sorted(mine[h]["where"]), "a4_records": len(mine[h]["records"]), "occurrences_by_set": by_set}
                    for h, by_set in sorted(hits.items())]
    return {"per_set": per, "shared_units": shared_units, "total_shared_units": len(hits), "absent": [k for k, v in per.items() if "absent" in v]}


def overlap(path: Path, repo: Path) -> dict[str, Any]:
    from robo_jev.decisions.records import read_records

    mine = read_records(path)
    my_sig = {k: set() for k in ("ids", "requests", "states", "instructions")}
    for r in mine:
        for k, v in _signature(r).items():
            my_sig[k] |= v
    my_sentences = {}
    for r in mine:
        for s in _sentences(r):
            my_sentences.setdefault(_h(s), s)
    my_grams = {}
    for r in mine:
        for g in _grams(r):
            my_grams.setdefault(_h(" ".join(g)), " ".join(g))
    tests = repo / "artifacts" / "simgen" / "v0b" / "tests"
    refs = {**{f"T{i}": tests / f"T{i}.jsonl" for i in range(1, 8)}, "G1": repo / "artifacts" / "simgen" / "probes" / "G1.jsonl",
            "G2": repo / "artifacts" / "simgen" / "probes" / "G2.jsonl", "T8a": repo / "artifacts" / "t8" / "T8a.jsonl", "T8b": repo / "artifacts" / "t8" / "T8b.jsonl",
            "T9": repo / "artifacts" / "t9" / "T9.jsonl", "B1": repo / "artifacts" / "b1" / "b1-infamily.jsonl",
            "H-set": repo / "artifacts" / "simgen" / "v04" / "scene-holdout.jsonl", "train-v04": repo / "artifacts" / "simgen" / "v04" / "train-v04.jsonl"}
    gram_sets = ("T8b", "T9", "H-set")
    per: dict[str, Any] = {}
    shared_sentences: Counter = Counter()
    shared_grams: Counter = Counter()
    for name, ref in refs.items():
        if not ref.is_file():
            per[name] = {"absent": str(ref)}
            continue
        hits = {k: 0 for k in my_sig}
        sentence_hits: set[bytes] = set()
        gram_hits: set[bytes] = set()
        count = 0
        for record in _stream(ref):
            count += 1
            for k, v in _signature(record).items():
                hits[k] += len(v & my_sig[k])
            for s in _sentences(record):
                key = _h(s)
                if key in my_sentences:
                    sentence_hits.add(key)
            if name in gram_sets:
                for g in _grams(record):
                    key = _h(" ".join(g))
                    if key in my_grams:
                        gram_hits.add(key)
        per[name] = {"records": count, "shared": hits, "shared_sentences": len(sentence_hits)}
        for key in sentence_hits:
            shared_sentences[my_sentences[key]] += 1
        if name in gram_sets:
            per[name]["shared_5grams"] = len(gram_hits)
            for key in gram_hits:
                shared_grams[my_grams[key]] += 1
    total = sum(sum(v["shared"].values()) + v["shared_sentences"] for v in per.values() if "shared" in v)
    grams = sum(v.get("shared_5grams", 0) for v in per.values())
    summary = {"my_records": len(mine), "my_sentences": len(my_sentences), "my_5grams": len(my_grams),
               "per_set": {k: ({**v["shared"], "sentences": v["shared_sentences"], **({"5grams": v["shared_5grams"]} if "shared_5grams" in v else {}),
                                "records": v["records"]} if "shared" in v else v) for k, v in per.items()},
               "shared_total": total, "shared_5grams_total": grams, "absent": [k for k, v in per.items() if "absent" in v]}
    unit_refs = {**refs, **{name: repo.joinpath(*parts) for name, parts in AUTHOR_TRAIN.items()}}
    index = my_unit_index(mine)
    units = unit_overlap(index, unit_refs)
    summary["text_units"] = {"definition": "robo_jev.data.general_sets.record_units (the v0.6 assembly's): whole normalised state, every state "
                                           "line value with >= 5 words, every option text with >= 5 words",
                             "my_units": len(index), "per_set": units["per_set"], "total_shared_units": units["total_shared_units"],
                             "absent": units["absent"]}
    summary["pass"] = bool(total == 0 and grams == 0 and not summary["absent"] and units["total_shared_units"] == 0 and not units["absent"])
    return {"summary": summary, "shared_units": units["shared_units"],
            "my_shared_sentences": {"count": len(shared_sentences), "sha256": sorted(_unit_hash(x) for x in shared_sentences)},
            "my_shared_5grams": {"count": len(shared_grams), "sha256": sorted(_unit_hash(x) for x in shared_grams)}}
