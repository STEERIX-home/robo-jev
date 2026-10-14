"""A3 checks: record validation against the trainer and the scorers, overlap with every reference set (in code — only counts and A3's
own text are ever printed), and the physics sanity suite (monotonicity sweeps, analytic thresholds per law, determinism)."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import multiprocessing as mp
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

import numpy as np

from robo_jev.authors.a3 import build, sim
from robo_jev.authors.a3 import catalog as cat
from robo_jev.authors.a3.scenes import SCENES

__all__ = ["REFERENCE_SETS", "overlap_report", "physics_sanity", "sanity_markdown", "validate"]

# --------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------


def _noisy_predictions(records: Sequence[Mapping[str, Any]], *, seed: int, noise: float) -> list[dict[str, Any]]:
    """Synthetic predictions: the labels plus noise, in the prediction format the scorers read (one listing per question)."""
    rng = np.random.default_rng(seed)
    out = []
    for record in records:
        questions = {}
        for name, question in record["request"]["questions"].items():
            label = record["labels"][name]
            if question["type"] == "choice":
                keys = list(question["criteria"])
                raw = np.array([float(label["probabilities"][k]) for k in keys]) + rng.dirichlet(np.ones(len(keys))) * noise
                answer: dict[str, Any] = {"probabilities": {k: float(v) for k, v in zip(keys, raw / raw.sum())}}
                options: list[str] | None = keys
            else:
                value = float(label.get("p_yes", label.get("value")))
                value = float(np.clip(value + rng.normal(0.0, noise), 0.0, 1.0))
                answer = {"p_yes": value} if question["type"] == "noul" else {"value": value, "bins": None}
                options = None
            questions[name] = {"type": question["type"], "n_listings": 1, "answer": answer,
                               "listings": [{"options": options, "values": None, "answer": answer}]}
        out.append({"id": record["id"], "model": f"synthetic-{noise}", "questions": questions})
    return out


def _verdict_v05(repo: Path) -> Any:
    spec = importlib.util.spec_from_file_location("a3_verdict_v05", repo / "scripts" / "verdict_v05.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def validate(path: Path, repo: Path, *, sample: int = 400) -> dict[str, Any]:
    """Every record through the trainer's loader and format check; a pair-preserving sample through the prompt builder (real
    tokenizer, chat format, counterbalanced listings, displayed-line check), the in-family readers (robust keys, direction,
    comparison) and the v0.5 call view with synthetic predictions."""
    from robo_jev.authors import a3
    from robo_jev.bench import infamily, system1
    from robo_jev.decisions import examples
    from robo_jev.decisions.prompt import check_displayed_lines, record_prompts
    from robo_jev.decisions.records import pair_groups, read_records, select_records, validate_record
    from robo_jev.model.tokenizer import available_tokenizer, load_tokenizer

    records = read_records(path)
    problems = [(r["id"], p) for r in records for p in validate_record(r)]
    groups = pair_groups(records)
    picked = select_records(records, sample, seed=0)
    out: dict[str, Any] = {"records": len(records), "format_problems": len(problems), "problem_examples": problems[:5],
                           "pair_groups": len(groups), "pair_group_sizes": dict(Counter(len(g) for g in groups)), "sample": len(picked)}
    # prompts as the trainer builds them
    found = available_tokenizer()
    if found is not None:
        tokenizer = load_tokenizer(found[1])
        lengths, line_problems, n_prompts = [], 0, 0
        for record in picked:
            prompts = record_prompts(record, tokenizer, seed=1, fmt="chat", orders="counterbalanced")
            n_prompts += len(prompts)
            line_problems += sum(len(check_displayed_lines(p)) for p in prompts)
            lengths += [len(p["tokens"]) for p in prompts]
        dev = sum(1 for r in records if examples.is_dev(r, fraction=examples.DEFAULT_DEV_FRACTION, salt=examples.DEFAULT_DEV_SALT))
        out["prompts"] = {"tokenizer": found[0], "prompts": n_prompts, "displayed_line_problems": line_problems,
                          "per_prompt_tokens": {"median": float(np.median(lengths)), "max": int(max(lengths))}, "dev_records": dev}
    # in-family readers (A3's families registered in-process)
    a3.register_infamily()
    readers: dict[str, Any] = {}
    robust = infamily.robust_keys(picked)
    readers["robust_keys"] = len(robust)
    readers["ambiguous_keys"] = len(infamily.ambiguous_keys(picked))
    for label, noise in (("oracle", 0.0), ("noisy_0.15", 0.15), ("noisy_0.35", 0.35)):
        predictions = _noisy_predictions(picked, seed=1, noise=noise)
        direction = infamily.direction_items("A3", picked, predictions)
        comparison = infamily.comparison_items("A3", picked, predictions)
        bands = system1.question_items("A3", picked, predictions)
        readers[label] = {
            "direction_items": len(direction), "direction": float(np.mean([v for _, v in direction.values()])) if direction else None,
            "direction_clusters": len({c for c, _ in direction.values()}),
            "comparison_items": len(comparison), "comparison": float(np.mean([v for _, v in comparison.values()])) if comparison else None,
            "band_accuracy": float(np.mean([v for _, v in bands["band_accuracy"].values()])),
            "band_balanced_robust_items": len(bands.get("band_balanced_robust", {})),
            "safe_choice": float(np.mean([v for _, v in bands["safe_choice"].values()])) if bands.get("safe_choice") else None,
        }
    out["infamily"] = readers
    # v0.5 call view
    verdict = _verdict_v05(repo)
    calls = {}
    for label, noise in (("oracle", 0.0), ("noisy_0.35", 0.35)):
        predictions = _noisy_predictions(picked, seed=2, noise=noise)
        items = verdict.set_calls("A3", picked, predictions)
        metrics = verdict.call_metrics(items, n_boot=200, seed=0)
        calls[label] = {"n": metrics["n"], "clusters": metrics["clusters"], "classes": metrics["classes"],
                        **{m: metrics[m]["value"] for m in ("DC", "LU", "LC", "GAP", "CA")}}
    out["call_view"] = calls
    ok = (not problems and out.get("prompts", {}).get("displayed_line_problems", 0) == 0 and readers["oracle"]["direction"] == 1.0
          and readers["oracle"]["comparison"] == 1.0 and calls["oracle"]["n"] > 0)
    out["summary"] = {"records": len(records), "format_problems": len(problems), "pair_groups": len(groups),
                      "prompt_line_problems": out.get("prompts", {}).get("displayed_line_problems"),
                      "per_prompt_tokens": out.get("prompts", {}).get("per_prompt_tokens"),
                      "infamily_oracle": {k: readers["oracle"][k] for k in ("direction_items", "direction", "comparison_items", "comparison")},
                      "call_view_oracle": calls["oracle"], "call_view_noisy": calls["noisy_0.35"], "ok": bool(ok)}
    return out


# --------------------------------------------------------------------------
# Overlap
# --------------------------------------------------------------------------

#: Reference sets (paths relative to the repository root). ``fivegram`` marks the sets that must share no word 5-gram with A3.
REFERENCE_SETS = {
    **{f"T{i}": ("artifacts/simgen/v0b/tests/T%d.jsonl" % i, False) for i in range(1, 8)},
    "G1": ("artifacts/simgen/probes/G1.jsonl", False), "G2": ("artifacts/simgen/probes/G2.jsonl", False),
    "T8a": ("artifacts/t8/T8a.jsonl", False), "T8b": ("artifacts/t8/T8b.jsonl", True), "T9": ("artifacts/t9/T9.jsonl", True),
    "B1": ("artifacts/b1/b1-infamily.jsonl", False), "Hset": ("artifacts/simgen/v04/scene-holdout.jsonl", True),
    "train_v04": ("artifacts/simgen/v04/train-v04.jsonl", False),
}
_WORD = re.compile(r"[a-z0-9]+(?:['.-][a-z0-9]+)*")
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?;:])\s+|\n+")


def _h(text: str) -> str:
    return hashlib.sha256(" ".join(text.split()).encode("utf-8")).hexdigest()


def _strings(record: Mapping[str, Any]) -> tuple[str, list[str], list[str]]:
    """(state, instructions, option / scale texts) of a request."""
    request = record.get("request") or {}
    state = str(request.get("state") or "")
    instructions, options = [], []
    for question in (request.get("questions") or {}).values():
        if not isinstance(question, Mapping):
            continue
        if question.get("instructions"):
            instructions.append(str(question["instructions"]))
        criteria = question.get("criteria")
        if isinstance(criteria, Mapping):
            options += [str(v) for v in criteria.values() if str(v).strip()]
        elif isinstance(criteria, list):
            options += [str(v) for v in criteria if str(v).strip()]
    return state, instructions, options


def _sentences(texts: Iterable[str]) -> set[str]:
    out = set()
    for text in texts:
        for piece in _SENTENCE_SPLIT.split(text):
            norm = " ".join(piece.lower().split()).strip(" .;:!?")
            if len(norm.split()) >= 3:
                out.add(norm)
    return out


def _fivegrams(texts: Iterable[str]) -> set[tuple[str, ...]]:
    out = set()
    for text in texts:
        words = _WORD.findall(text.lower())
        out.update(tuple(words[i : i + 5]) for i in range(len(words) - 4))
    return out


def _stream(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def overlap_report(path: Path, repo: Path) -> dict[str, Any]:
    """Shared ids, request / state / instruction hashes and sentences with every reference set, and shared word 5-grams with T8b, T9
    and the H-set. The reference texts are read here only; the report holds counts and A3's own shared fragments."""
    mine = list(_stream(path))
    my_ids = {r["id"] for r in mine}
    my_requests = {_h(json.dumps(r["request"], sort_keys=True)) for r in mine}
    my_states, my_instr, my_texts = set(), set(), []
    for r in mine:
        state, instructions, options = _strings(r)
        my_states.add(_h(state))
        my_instr.update(_h(i) for i in instructions)
        my_texts += [state, *instructions, *options]
    my_sentences = _sentences(my_texts)
    my_grams = _fivegrams(my_texts)
    out: dict[str, Any] = {"mine": {"records": len(mine), "states": len(my_states), "instructions": len(my_instr), "sentences": len(my_sentences),
                                    "fivegrams": len(my_grams)}, "sets": {}}
    for name, (rel, fivegram) in REFERENCE_SETS.items():
        ref = repo / rel
        if not ref.exists():
            out["sets"][name] = {"missing": True}
            continue
        ids, requests, states, instr = set(), set(), set(), set()
        sentence_hits: set[str] = set()
        gram_hits: set[tuple[str, ...]] = set()
        n = 0
        for record in _stream(ref):
            n += 1
            ids.add(str(record.get("id")))
            if isinstance(record.get("request"), Mapping):
                requests.add(_h(json.dumps(record["request"], sort_keys=True)))
            state, instructions, options = _strings(record)
            states.add(_h(state))
            instr.update(_h(i) for i in instructions)
            texts = [state, *instructions, *options]
            sentence_hits |= my_sentences & _sentences(texts)
            if fivegram:
                gram_hits |= my_grams & _fivegrams(texts)
        entry = {"records": n, "shared_ids": len(my_ids & ids), "shared_requests": len(my_requests & requests),
                 "shared_states": len(my_states & states), "shared_instructions": len(my_instr & instr), "shared_sentences": len(sentence_hits),
                 "shared_sentence_fragments_of_a3": sorted(sentence_hits)[:20]}
        if fivegram:
            entry["shared_fivegrams"] = len(gram_hits)
            entry["shared_fivegram_fragments_of_a3"] = [" ".join(g) for g in sorted(gram_hits)[:40]]
        out["sets"][name] = entry
    fields = ("shared_ids", "shared_requests", "shared_states", "shared_instructions", "shared_sentences")
    totals = {f: sum(int(e.get(f, 0)) for e in out["sets"].values()) for f in fields}
    totals["shared_fivegrams_T8b_T9_Hset"] = sum(int(e.get("shared_fivegrams", 0)) for e in out["sets"].values())
    totals["missing_sets"] = [k for k, e in out["sets"].items() if e.get("missing")]
    totals["clean"] = all(v == 0 for k, v in totals.items() if k not in ("missing_sets", "clean")) and not totals["missing_sets"]
    out["summary"] = totals
    return out


# --------------------------------------------------------------------------
# Physics sanity
# --------------------------------------------------------------------------

SWEEPS = {
    # scene: [(field, values-builder, risk_up)]
    "indexdial": [("index_s", "decisive", False), ("radius_mm", "radius", True), ("mu_scale", "mu", False)],
    "mayostand": [("setdown_mm", "decisive", True), ("release_mm", "release", True), ("base_extra_kg", "base", False)],
    "holddown": [("press_n", "decisive", True), ("clamp_n", "clamp", False), ("mu_scale", "mu", False)],
}


def _sweep_values(sc: Any, state: Mapping[str, Any], field: str, n: int) -> list[float]:
    if field == "index_s":
        return [round(float(v), 3) for v in np.exp(np.linspace(math.log(0.3), math.log(3.0), n))]
    if field == "radius_mm":
        return [float(v) for v in np.linspace(150, 480, n)]
    if field == "mu_scale":
        return [float(v) for v in np.exp(np.linspace(math.log(0.5), math.log(2.0), n))]
    if field == "setdown_mm":
        lo, hi = sc.decisive_bounds(state)
        return [float(v) for v in np.linspace(lo, hi, n)]
    if field == "release_mm":
        return [float(v) for v in np.linspace(0, 60, n)]
    if field == "base_extra_kg":
        return [float(v) for v in np.linspace(0, 8, n)]
    if field == "press_n":
        return [float(v) for v in np.exp(np.linspace(math.log(10), math.log(1500), n))]
    if field == "clamp_n":
        return [float(v) for v in np.exp(np.linspace(math.log(30), math.log(800), n))]
    raise ValueError(field)


def _apply_sweep(sc: Any, state: Mapping[str, Any], values: Mapping[str, float], field: str, x: float) -> tuple[dict, dict]:
    st, vals = dict(state), dict(values)
    if field == "mu_scale":
        for q in ("mu", "mu_nest", "mu_pad"):
            if q in vals:
                vals[q] = vals[q] * x
    elif field == "base_extra_kg":
        vals["base_kg"] = vals["base_kg"] + x
    else:
        st[field] = x
    return st, vals


def _sweep_job(args: tuple[str, str, int, int, int]) -> dict[str, Any]:
    scene_name, field, seed, n_values, k = args
    sc = SCENES[scene_name]
    rng = np.random.default_rng(seed)
    state = sc.sample_state(rng)
    if scene_name == "mayostand":
        lo, hi = sc.decisive_bounds(state)
        state["setdown_mm"] = float(round(rng.uniform(lo, hi) / 5) * 5)
    elif scene_name == "indexdial":
        state["index_s"] = float(round(rng.uniform(0.5, 1.5), 2))
    elif scene_name == "holddown":
        state["press_n"] = float(round(rng.uniform(40, 400) / 5) * 5)
    points = cat.uniform_points(seed, k, sc.hidden_names)
    values = sc.values(state, "base", points)
    xs = _sweep_values(sc, state, field, n_values)
    key = np.zeros((len(xs), k), dtype=bool)
    severe = np.zeros((len(xs), k), dtype=bool)
    for i, x in enumerate(xs):
        for j in range(k):
            st, vals = _apply_sweep(sc, state, {q: float(v[j]) for q, v in values.items()}, field, x)
            outcome = sc.simulate(st, vals)["outcome"]
            key[i, j] = outcome in sc.key_events
            severe[i, j] = outcome == sc.severe
    return {"scene": scene_name, "field": field, "seed": seed, "xs": xs, "key": key.mean(1).tolist(), "severe": severe.mean(1).tolist(),
            "key_draws": key.astype(int).tolist(), "severe_draws": severe.astype(int).tolist()}


def _violations(series: Sequence[float], risk_up: bool) -> int:
    d = np.diff(np.asarray(series, dtype=float))
    return int(np.sum(d < -1e-9)) if risk_up else int(np.sum(d > 1e-9))


def _draw_violations(draws: Sequence[Sequence[int]], risk_up: bool) -> int:
    arr = np.asarray(draws, dtype=int)
    return sum(_violations(arr[:, j], risk_up) for j in range(arr.shape[1]))


def _threshold_job(args: tuple[str, int]) -> dict[str, Any]:
    """Analytic-threshold checks on one random case per call (law-specific; module docstring of :mod:`.sim`)."""
    scene_name, seed = args
    rng = np.random.default_rng(10_000 + seed)
    sc = SCENES[scene_name]
    out: dict[str, Any] = {"scene": scene_name, "seed": seed, "checks": []}
    if scene_name == "indexdial":
        # sliding μg: a low, wide part (no tipping) — below 0.9 μg never slides; tipping b/h: a tall part on high friction
        for mode in ("slide", "tip"):
            state = sc.sample_state(rng)
            if mode == "slide":
                state["dims_mm"] = [160, 150, 50]
                values = {"mu": float(rng.uniform(0.2, 0.7)), "kh": 0.5, "ex": 0.0, "dr": 0.0}
                thr = values["mu"] * sim.GRAVITY
            else:
                state["dims_mm"] = [90, 80, 300]
                values = {"mu": 1.4, "kh": float(rng.uniform(0.45, 0.6)), "ex": 0.0, "dr": 0.0}
                thr = sim.GRAVITY * 0.04 / (values["kh"] * 0.3)
            radius = state["radius_mm"] / 1000
            coef = sim.dial_peak_accel(state["angle_deg"], 1.0, radius)
            for ratio in (0.6, 0.9, 1.8, 2.5):
                state["index_s"] = math.sqrt(coef / (thr * ratio))
                outcome = sc.simulate(state, values)["outcome"]
                # below the static threshold nothing may happen; well above it (1.8×, past the measured dynamic factor's p90 of
                # 1.51) the part must move
                expect = None if 0.9 < ratio < 1.8 else (ratio >= 1.8)
                if mode == "slide" and ratio >= 1.8:
                    # slipping ≥ 10 mm also needs time: require the point-mass slip model to agree before expecting it
                    slip = sim.dial_slip(state["angle_deg"], state["index_s"], np.array([radius]), np.array([values["mu"]]))[0]
                    expect = True if slip >= 2 * sim.DIAL_SHIFT_M else None
                event = outcome in sc.key_events
                out["checks"].append({"law": f"A_{mode}", "ratio": ratio, "expect": expect, "event": event, "outcome": outcome,
                                      "ok": None if expect is None else event == expect})
    elif scene_name == "mayostand":
        # centre of mass past the support: released at 0 mm, combined centre of mass ±15 mm about the front feet line
        state = sc.sample_state(rng)
        state["release_mm"] = 0
        base, load = float(rng.uniform(1.6, 9.5)), float(rng.uniform(2.0, 11.0))
        length = state["dims_mm"][0] / 1000
        ms, com, _ = sim.stand_inertial(state["reach_mm"] / 1000, state["tray_mm"] / 1000, state["height_mm"] / 1000, base)
        x_star = state["reach_mm"] / 1000 + ms * (state["reach_mm"] / 1000 - com[0]) / load
        lo, hi = sc.decisive_bounds(state)
        for offset in (-0.015, 0.015):
            x = x_star + offset
            if not (lo / 1000 <= x <= hi / 1000):
                continue
            st = dict(state)
            st["setdown_mm"] = x * 1000
            outcome = sc.simulate(st, {"load_kg": load, "com_frac": 0.0, "base_kg": base, "dx": 0.0})["outcome"]
            expect = offset > 0
            out["checks"].append({"law": "B_support", "offset_m": offset, "expect_tip": expect, "outcome": outcome,
                                  "ok": (outcome == "tips_over") == expect and (expect or outcome == "stays_level")})
    else:
        # friction limit vs applied load: peak ≤ 0.9 capacity holds; ≥ 1.15 capacity breaks free
        state = sc.sample_state(rng)
        values = {"mu_nest": float(rng.uniform(0.1, 0.8)), "mu_pad": float(rng.uniform(0.14, 0.9)), "force_factor": float(rng.uniform(0.5, 1.0))}
        capacity = sim.clamp_capacity(mu_nest=values["mu_nest"], mu_pad=values["mu_pad"], clamp_force=values["force_factor"] * state["clamp_n"],
                                      mass=state["mass_kg"])
        for ratio, expect in ((0.6, "stays_put"), (0.9, "stays_put"), (1.15, "moves"), (1.5, "breaks_free")):
            st = dict(state)
            st["press_n"] = capacity * ratio
            outcome = sc.simulate(st, values)["outcome"]
            ok = outcome == expect if expect != "moves" else outcome in sc.key_events
            out["checks"].append({"law": "C_friction", "ratio": ratio, "expect": expect, "outcome": outcome, "ok": ok})
    return out


def _determinism_job(args: tuple[str, int]) -> dict[str, Any]:
    scene_name, group = args
    plan_a = build.plan_group(scene_name, group)
    plan_b = build.plan_group(scene_name, group)
    same_plan = json.dumps(plan_a, sort_keys=True) == json.dumps(plan_b, sort_keys=True)
    sc = SCENES[scene_name]
    state = plan_a["base_state"]
    points = cat.uniform_points(plan_a["draws_seed"], 4, sc.hidden_names)
    values = sc.values(state, "base", points)
    runs = [[json.dumps(sc.simulate(state, {q: float(v[k]) for q, v in values.items()}), sort_keys=True) for k in range(4)] for _ in range(2)]
    return {"scene": scene_name, "group": group, "same_plan": same_plan, "same_rollouts": runs[0] == runs[1]}


def physics_sanity(*, workers: int = 4, quick: bool = False) -> dict[str, Any]:
    t0 = time.perf_counter()
    n_states, n_values, k = (3, 8, 6) if quick else (10, 14, 12)
    sweep_jobs = [(scene, field, 500 + 37 * i, n_values, k) for scene, specs in SWEEPS.items() for field, _, _ in specs for i in range(n_states)]
    threshold_jobs = [(scene, i) for scene in SCENES for i in range(4 if quick else 24)]
    determinism_jobs = [(scene, g) for scene in SCENES for g in (0, 7)]
    with mp.get_context("fork").Pool(int(workers)) as pool:
        sweeps = pool.map(_sweep_job, sweep_jobs, chunksize=1)
        thresholds = pool.map(_threshold_job, threshold_jobs, chunksize=1)
        determinism = pool.map(_determinism_job, determinism_jobs, chunksize=1)
    risk = {(scene, field): up for scene, specs in SWEEPS.items() for field, _, up in specs}
    mono: dict[str, Any] = defaultdict(lambda: {"sweeps": 0, "key_label_violations": 0, "key_draw_violations": 0, "severe_label_violations": 0,
                                                "severe_draw_violations": 0})
    for s in sweeps:
        entry = mono[f"{s['scene']}:{s['field']}"]
        up = risk[(s["scene"], s["field"])]
        entry["sweeps"] += 1
        entry["key_label_violations"] += _violations(s["key"], up)
        entry["key_draw_violations"] += _draw_violations(s["key_draws"], up)
        entry["severe_label_violations"] += _violations(s["severe"], up)
        entry["severe_draw_violations"] += _draw_violations(s["severe_draws"], up)
        entry.setdefault("examples", []).append({"xs": s["xs"], "key": [round(v, 3) for v in s["key"]], "severe": [round(v, 3) for v in s["severe"]]})
    for entry in mono.values():
        entry["examples"] = entry["examples"][:2]
    checks = [c for t in thresholds for c in t["checks"]]
    by_law: dict[str, Counter] = defaultdict(Counter)
    for c in checks:
        by_law[c["law"]]["checked" if c["ok"] is not None else "not_asserted"] += 1
        if c["ok"] is False:
            by_law[c["law"]]["failed"] += 1
    summary = {
        "key_label_violations": sum(e["key_label_violations"] for e in mono.values()),
        "key_draw_violations": sum(e["key_draw_violations"] for e in mono.values()),
        "severe_label_violations": {k_: e["severe_label_violations"] for k_, e in mono.items() if e["severe_label_violations"]},
        "threshold_checks": {law: dict(c) for law, c in by_law.items()},
        "threshold_failures": sum(c.get("failed", 0) for c in by_law.values()),
        "determinism": all(d["same_plan"] and d["same_rollouts"] for d in determinism),
        "sweeps": len(sweeps), "seconds": round(time.perf_counter() - t0, 1),
    }
    return {"summary": summary, "monotonicity": dict(mono), "thresholds": checks, "determinism": determinism,
            "settings": {"states_per_sweep_field": n_states, "values_per_sweep": n_values, "draws": k}}


def sanity_markdown(result: Mapping[str, Any]) -> str:
    s = result["summary"]
    lines = ["# A3 physics sanity", "", f"Sweeps: {s['sweeps']} ({result['settings']}); {s['seconds']} s.", "",
             "## Monotonicity (key label must never move against the physics)", "",
             "| scene:factor | sweeps | key label violations | key draw violations | severe label violations | severe draw violations |",
             "|---|---|---|---|---|---|"]
    for name, e in result["monotonicity"].items():
        lines.append(f"| {name} | {e['sweeps']} | {e['key_label_violations']} | {e['key_draw_violations']} | {e['severe_label_violations']} | "
                     f"{e['severe_draw_violations']} |")
    lines += ["", "## Analytic thresholds", "", "| law | checked | not asserted | failed |", "|---|---|---|---|"]
    for law, c in s["threshold_checks"].items():
        lines.append(f"| {law} | {c.get('checked', 0)} | {c.get('not_asserted', 0)} | {c.get('failed', 0)} |")
    lines += ["", f"Determinism (plan twice, rollouts twice): {s['determinism']}.", ""]
    return "\n".join(lines) + "\n"
