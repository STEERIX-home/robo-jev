#!/usr/bin/env python
"""Author A2 (v0.6 author-diversity experiment) — plan, label (MuJoCo), assemble and report the A2 training set.

    python scripts/author_a2.py plan --scene turntable --groups 660
    python scripts/author_a2.py label --scene turntable --workers 4          # resumable: skips groups already labelled
    python scripts/author_a2.py assemble --groups 625                          # -> artifacts/v06/a2/train.jsonl + manifest.json
    python scripts/author_a2.py report                                         # stats, prompt lengths, scorer checks, overlap
    python scripts/author_a2.py sanity --workers 4                             # monotonicity sweeps, analytic thresholds, determinism

MuJoCo runs only inside a capped unit, e.g.::

    systemd-run --user --unit=a2-label-turntable --collect -p MemoryMax=6G \\
        choom -n 1000 -- <venv python> scripts/author_a2.py label --scene turntable --workers 4

Writes only under ``artifacts/v06/a2/`` (work files in ``artifacts/v06/a2/work/``).
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Sequence

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

OUT = REPO / "artifacts" / "v06" / "a2"
WORK = OUT / "work"
SCENE_NAMES = ("turntable", "tiltback", "wallpress")


def _log(message: str) -> None:
    print(f"[a2 {dt.datetime.now().strftime('%H:%M:%S')}] {message}", file=sys.stderr, flush=True)


def _git() -> str | None:
    try:
        return subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    out = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:  # a line cut by a stopped unit: ignored, the group is relabelled
                    continue
    return out


def plan_path(scene: str, tag: str = "") -> Path:
    return WORK / f"plan-{scene}{'-' + tag if tag else ''}.jsonl"


def labels_path(scene: str, tag: str = "") -> Path:
    return WORK / f"labels-{scene}{'-' + tag if tag else ''}.jsonl"


#: The label file each scene's records are built from. wallpress was relabelled after the physics audit of 2026-10-14 (stale bounding
#: volumes in MuJoCo's midphase after a run-time centre-of-mass write; the load–world pair of this scene goes through the midphase):
#: its labels now come from the ``bvhfix`` run; the stored pre-fix file ``labels-wallpress.jsonl`` is kept for the comparison.
LABEL_TAGS = {"turntable": "", "tiltback": "", "wallpress": "bvhfix"}


def current_labels_path(scene: str) -> Path:
    return labels_path(scene, LABEL_TAGS[scene])


def cmd_plan(args: argparse.Namespace) -> int:
    from robo_jev.authors.a2.generate import plan_group

    WORK.mkdir(parents=True, exist_ok=True)
    plans = [plan_group(args.scene, i) for i in range(args.start, args.start + args.groups)]
    existing = {p["index"]: p for p in _read_jsonl(plan_path(args.scene, args.tag))}
    for p in plans:
        existing[p["index"]] = p
    with plan_path(args.scene, args.tag).open("w", encoding="utf-8") as handle:
        for index in sorted(existing):
            handle.write(json.dumps(existing[index], sort_keys=True) + "\n")
    _log(f"{args.scene}: {len(existing)} group plans in {plan_path(args.scene, args.tag)}")
    return 0


def _label_one(plan: dict[str, Any]) -> dict[str, Any]:
    from robo_jev.authors.a2.generate import label_group

    return label_group(plan)


def cmd_label(args: argparse.Namespace) -> int:
    import multiprocessing as mp

    if not os.environ.get("INVOCATION_ID") and not args.allow_outside_unit:
        raise SystemExit("MuJoCo labelling runs only inside a capped systemd-run unit (see the module docstring)")
    plans = _read_jsonl(plan_path(args.scene, args.tag))
    out_path = labels_path(args.scene, args.label_tag if args.label_tag is not None else args.tag)
    done = {r["index"] for r in _read_jsonl(out_path)}
    todo = [p for p in plans if p["index"] not in done]
    if args.limit is not None:
        todo = todo[: args.limit]
    _log(f"{args.scene}: {len(plans)} planned, {len(done)} labelled, {len(todo)} to do with {args.workers} workers -> {out_path.name}")
    started = time.time()
    with out_path.open("a", encoding="utf-8") as handle, mp.get_context("fork").Pool(args.workers) as pool:
        for n, result in enumerate(pool.imap_unordered(_label_one, todo, chunksize=1), 1):
            handle.write(json.dumps(result, sort_keys=True) + "\n")
            handle.flush()
            if n % 10 == 0 or n == len(todo):
                rate = n / max(time.time() - started, 1e-9)
                _log(f"{args.scene}: {n}/{len(todo)} groups ({rate * 60:.1f}/min, eta {(len(todo) - n) / max(rate, 1e-9) / 60:.1f} min)")
    return 0


def cmd_assemble(args: argparse.Namespace) -> int:
    import mujoco

    from robo_jev.authors.a2.generate import assemble_group
    from robo_jev.decisions.records import validate_record

    git = args.git or _git()
    scene_git = dict(item.split("=", 1) for item in (args.scene_git or []))
    records, manifest_scenes = [], {}
    for scene in args.scenes:
        plans = {p["index"]: p for p in _read_jsonl(plan_path(scene))}
        labelled = {r["index"]: r for r in _read_jsonl(current_labels_path(scene))}
        rechecked = json.loads(unstable_path(scene).read_text(encoding="utf-8")) if unstable_path(scene).is_file() else {"groups": []}
        flung = {g["index"] for g in rechecked["groups"] if g["verdict"] == "flung"}
        used, dropped, kept_flung = [], [], []
        for index in sorted(plans):
            if len(used) >= args.groups:
                break
            lab = labelled.get(index)
            if lab is None:
                continue
            if lab["unstable"] and index not in flung:
                dropped.append(index)
                continue
            if lab["unstable"]:
                kept_flung.append(index)
            group = assemble_group(plans[index], lab, git=scene_git.get(scene, git), mujoco_version=mujoco.__version__)
            for record in group:
                problems = validate_record(record)
                if problems:
                    raise SystemExit(f"{record['id']}: {problems[:3]}")
            records.extend(group)
            used.append(index)
        manifest_scenes[scene] = {"labels": str(current_labels_path(scene).relative_to(REPO)), "git": scene_git.get(scene, git),
                                  "groups": len(used), "records": 2 * len(used), "first_index": used[0] if used else None,
                                  "last_index": used[-1] if used else None, "dropped_unstable": dropped, "kept_flung_topples": kept_flung,
                                  "labelled_cpu_s": round(sum(labelled[i]["cpu_s"] for i in used), 1)}
        _log(f"{scene}: {len(used)} groups, {2 * len(used)} records (dropped as unstable: {len(dropped)})")
    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / "train.jsonl"
    with out.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    from robo_jev.decisions.records import file_sha256

    manifest = {"kind": "v06-a2-train", "set": "v06-a2", "split": "train", "author": "a2", "path": str(out.relative_to(REPO)),
                "sha256": file_sha256(out), "records": len(records), "scenes": manifest_scenes, "git": git,
                "mujoco": mujoco.__version__, "created": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
                "k": 32, "priors": ["base", "alt_a", "alt_b"], "generator": "v06-a2-mujoco-v1", "label_code_git": args.label_git,
                "labelling": "scripts/author_a2.py label, 4 worker processes in one capped systemd-run --user unit (MemoryMax 8G, choom -n 1000, "
                             "OPENBLAS/OMP threads 1); plans in artifacts/v06/a2/work/plan-<scene>.jsonl, per-draw outcome codes in "
                             "artifacts/v06/a2/work/labels-<scene>.jsonl",
                "label_rule": "training label = the base prior's outcome frequency over K = 32 shared hidden draws; labels_by_prior and "
                              "robustness carry alt_a / alt_b (standard alternative-prior rule on A2's catalog)"}
    relabel = OUT / "relabel-wallpress.json"
    if relabel.is_file():
        summary = json.loads(relabel.read_text(encoding="utf-8"))
        manifest["relabel"] = {"scene": "wallpress", "date": "2026-10-14", "why": summary["why"], "fix": summary["fix"],
                               "previous_sha256": summary["previous_train_sha256"], "comparison": "artifacts/v06/a2/relabel-wallpress.json",
                               "headline": summary["headline"]}
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8")
    _log(f"wrote {len(records)} records to {out}")
    return 0


# --------------------------------------------------------------------------
# Reports
# --------------------------------------------------------------------------


def _band(x: float) -> str:
    return "low" if x < 1 / 3 else ("high" if x > 2 / 3 else "middle")


def _share(counter: dict, total: int) -> dict:
    return {k: round(v / total, 4) for k, v in sorted(counter.items())} if total else {}


def _load_verdict_v05():
    import importlib.util

    spec = importlib.util.spec_from_file_location("a2_verdict_v05", REPO / "scripts" / "verdict_v05.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _synthetic_predictions(records: list[dict[str, Any]], seed: int) -> list[dict[str, Any]]:
    import numpy as np

    rng = np.random.default_rng(seed)
    out = []
    for record in records:
        questions = {}
        for name, question in record["request"]["questions"].items():
            if question["type"] == "choice":
                keys = list(question["criteria"])
                answer = {"probabilities": dict(zip(keys, map(float, rng.dirichlet(np.ones(len(keys))))))}
                options = keys
            else:
                value = float(rng.random())
                answer = {"p_yes": value} if question["type"] == "noul" else {"value": value, "bins": None}
                options = None
            questions[name] = {"type": question["type"], "n_listings": 1, "answer": answer, "listings": [{"options": options, "values": None, "answer": answer}]}
        out.append({"id": record["id"], "model": "synthetic", "questions": questions})
    return out


def _prompt_lengths(records: list[dict[str, Any]]) -> dict[str, Any]:
    import numpy as np

    from robo_jev.decisions.prompt import record_prompts
    from robo_jev.model.tokenizer import available_tokenizer, load_tokenizer, sha256_of_file

    found = available_tokenizer()
    tok = load_tokenizer(found[0])
    out: dict[str, Any] = {"tokenizer": {"id": found[0], "sha256": sha256_of_file(found[1])},
                           "measure": "state_plus_questions = the state, then per question its instructions and its option lines "
                                      "('key: description') or score criteria, joined by newlines; model_prompt = robo_jev.decisions.prompt "
                                      "record_prompts (chat, direct, one listing), prefix + one question", "scenes": {}}
    by_scene: dict[str, list[dict]] = {}
    for record in records:
        by_scene.setdefault(record["meta"]["scene"], []).append(record)
    for scene, rows in by_scene.items():
        request_lengths, prompt_lengths = [], []
        for record in rows:
            q = record["request"]
            parts = [q["state"]]
            for question in q["questions"].values():
                crit = question.get("criteria")
                lines = crit if isinstance(crit, list) else [f"{k}: {v}" for k, v in (crit or {}).items()]
                parts.append(question["instructions"] + "\n" + "\n".join(lines))
            request_lengths.append(len(tok.encode("\n".join(parts), add_special_tokens=False).ids))
        for record in rows[::10]:
            prompt_lengths += [len(p["tokens"]) for p in record_prompts(record, tok, seed=1, fmt="chat", mode="direct")]
        out["scenes"][scene] = {
            "state_plus_questions": {"median": float(np.median(request_lengths)), "p90": float(np.percentile(request_lengths, 90)),
                                     "max": int(max(request_lengths)), "min": int(min(request_lengths)), "n": len(request_lengths)},
            "model_prompt_per_question_chat_direct": {"median": float(np.median(prompt_lengths)), "max": int(max(prompt_lengths)),
                                                      "n": len(prompt_lengths), "sampled_records": len(rows[::10])},
        }
    return out


def cmd_report(args: argparse.Namespace) -> int:
    from collections import Counter, defaultdict

    import numpy as np

    from robo_jev.authors.a2 import registered_with_infamily
    from robo_jev.authors.a2.overlap import compare
    from robo_jev.bench import infamily, system1
    from robo_jev.decisions.train import load_training_records

    path = OUT / "train.jsonl"
    records = _read_jsonl(path)
    valid, summary = load_training_records(path)
    stats: dict[str, Any] = {"records": len(records), "loader": {"read": summary["read"], "valid": len(valid), "invalid": summary["invalid"]}}
    v5 = _load_verdict_v05()
    per_scene: dict[str, Any] = {}
    for scene in SCENE_NAMES:
        rows = [r for r in records if r["meta"]["scene"] == scene]
        if not rows:
            continue
        kinds = rows[0]["meta"]["question_kind"]
        name = {k: q for q, k in kinds.items()}
        groups = {r["generator"]["group"] for r in rows}
        base = [r for r in rows if r["generator"]["member"] == "base"]
        entry: dict[str, Any] = {"records": len(rows), "groups": len(groups), "domains": dict(Counter(r["meta"]["domain"] for r in rows))}
        pair_kind = Counter()
        for r in base:
            pr = r["meta"]["pair"]
            pair_kind["far_number" if pr["closeness"] == "far" else ("one_factor_near_decisive" if pr["closeness"] == "near" else f"one_factor_{pr['fact']}:{pr['field']}")] += 1
        entry["pairs"] = {"kinds": dict(sorted(pair_kind.items())), "one_factor_share": round(sum(1 for r in base if r["meta"]["pair"]["one_factor"]) / len(base), 4)}
        deltas = np.array([r["meta"]["pair"]["key_delta"] for r in base])
        entry["pairs"].update({"mean_abs_key_delta": round(float(np.abs(deltas).mean()), 4),
                               "moves_at_least_0.1": round(float(np.mean(np.abs(deltas) >= 0.1)), 4),
                               "direction_same_over_priors": round(float(np.mean([r["meta"]["pair"]["direction_same"] for r in base])), 4),
                               "no_reversal": round(float(np.mean([r["meta"]["pair"]["no_reversal"] for r in base])), 4)})
        labels = {}
        for kind in ("key", "severe", "score_cf"):
            field = "value" if kind == "score_cf" else "p_yes"
            values = np.array([r["labels"][name[kind]][field] for r in rows])
            labels[kind] = {"mean": round(float(values.mean()), 4), "bands": _share(Counter(_band(v) for v in values), len(values)),
                            "hist10": np.histogram(values, bins=10, range=(0, 1))[0].tolist(),
                            "exact_0": int(np.sum(values == 0)), "exact_1": int(np.sum(values == 1))}
        entry["labels"] = labels
        entry["robust_share"] = {kind: round(float(np.mean([r["meta"]["robustness"][name[kind]]["robust"] for r in rows])), 4) for kind in kinds.values()}
        truth = Counter(v5.truth_class(r, q) for r in rows for q in r["request"]["questions"] if v5.truth_class(r, q))
        entry["truth_classes"] = dict(truth)
        argmax = {}
        for kind in ("direction", "compare", "action", "outcome"):
            argmax[kind] = dict(Counter(max(r["labels"][name[kind]]["probabilities"].items(), key=lambda kv: kv[1])[0] for r in rows))
        entry["choice_argmax"] = argmax
        entry["alternatives"] = dict(Counter(r["meta"]["compare_option"] for r in base))
        entry["static_aim_vs_key"] = {"mean_aim": round(float(np.mean([r["meta"]["static_aim"]["key"] for r in base])), 4),
                                      "mean_key": round(float(np.mean([r["labels"][name["key"]]["p_yes"] for r in base])), 4),
                                      "corr": round(float(np.corrcoef([r["meta"]["static_aim"]["key"] for r in base], [r["labels"][name["key"]]["p_yes"] for r in base])[0, 1]), 4)}
        per_scene[scene] = entry
    stats["scenes"] = per_scene
    all_keys = [r["labels"][r["meta"]["key_question"]]["p_yes"] for r in records]
    stats["key_label_hist10_all"] = np.histogram(all_keys, bins=10, range=(0, 1))[0].tolist()
    stats["robust_share_all_scalar"] = round(float(np.mean([r["meta"]["robustness"][q]["robust"] for r in records for q, k in r["meta"]["question_kind"].items()
                                                              if k in ("key", "severe", "score_cf")])), 4)
    # scorer checks on a sample with synthetic predictions (and the labels themselves as answers)
    sample = [r for r in records if r["generator"]["group"] % 5 == 0]
    synthetic = _synthetic_predictions(sample, seed=0)
    oracle = system1.oracle_predictions(sample)
    with registered_with_infamily():
        comparison = infamily.comparison_items("A2", sample, synthetic)
        comparison_oracle = infamily.comparison_items("A2", sample, oracle)
    calls = v5.set_calls("A2", sample, synthetic)
    calls_oracle = v5.set_calls("A2", sample, oracle)
    stats["scorer_checks"] = {
        "sample_records": len(sample),
        "infamily_robust_keys": len(infamily.robust_keys(sample)),
        "infamily_ambiguous_keys": len(infamily.ambiguous_keys(sample)),
        "infamily_direction_items": len(infamily.direction_items("A2", sample, synthetic)),
        "infamily_direction_oracle_mean": round(float(np.mean([v for _, v in infamily.direction_items("A2", sample, oracle).values()])), 4),
        "infamily_comparison_items": len(comparison),
        "infamily_comparison_oracle_mean": round(float(np.mean([v for _, v in comparison_oracle.values()])), 4) if comparison_oracle else None,
        "verdict_v05_call_items": len(calls),
        "verdict_v05_classes": v5.call_metrics(calls)["classes"],
        "verdict_v05_oracle": {m: v5.call_metrics(calls_oracle)[m]["value"] for m in ("DC", "LU", "LC", "GAP", "CA")},
        "verdict_v05_synthetic": {m: v5.call_metrics(calls)[m]["value"] for m in ("DC", "LU", "LC", "GAP", "CA")},
        "note": "infamily.EVENT / PLAN_KEY do not know A2's scene names; the comparison reading registers robo_jev.authors.a2.EVENT / PLAN_KEY "
                "for the call (registered_with_infamily) — the same entries are in every record as meta.key_events / meta.plan_key",
    }
    # every pair's two briefings differ only in the sentence(s) carrying the changed fact; the questions differ only in numbers that
    # quote the decisive value (direction question) when the decisive number is the changed fact
    by_id = {r["id"]: r for r in records}
    over, same_questions = 0, 0
    for r in records:
        if r["generator"]["member"] != "base":
            continue
        partner = by_id[r["meta"]["pair"]["of"]]
        sa = r["request"]["state"].replace("\n", " ").split(". ")
        sb = partner["request"]["state"].replace("\n", " ").split(". ")
        over += len(sa) != len(sb) or sum(x != y for x, y in zip(sa, sb)) > 2
        same_questions += r["request"]["questions"].keys() == partner["request"]["questions"].keys()
    n_pairs = sum(1 for r in records if r["generator"]["member"] == "base")
    stats["pair_text_check"] = {"pairs": n_pairs, "states_differing_in_more_than_2_sentences": over, "same_question_names": same_questions}
    stats["prompt_lengths"] = _prompt_lengths(records)
    cpu = {}
    for scene in SCENE_NAMES:
        labelled = _read_jsonl(current_labels_path(scene))
        if labelled:
            cpu[scene] = {"groups_labelled": len(labelled), "cpu_s": round(sum(r["cpu_s"] for r in labelled), 1),
                          "rollouts": sum(len(s["codes"]) * len(next(iter(s["codes"].values()))) for r in labelled for s in r["setups"].values()),
                          "unstable": sum(r["unstable"] for r in labelled)}
    stats["cpu"] = cpu
    (OUT / "stats.json").write_text(json.dumps(stats, indent=1) + "\n", encoding="utf-8")
    _log(f"stats -> {OUT / 'stats.json'}")
    overlap = compare(records, REPO)
    overlap["rule"] = "0 shared ids, request / state / instruction hashes and sentences against every set; 0 shared word 5-grams with T8b, T9, H-set"
    overlap["pass"] = all(all(v == 0 for k, v in row.items() if k.startswith("shared")) for row in overlap["sets"].values() if "missing" not in row)
    overlap["missing"] = [n for n, row in overlap["sets"].items() if "missing" in row]
    (OUT / "overlap.json").write_text(json.dumps(overlap, indent=1) + "\n", encoding="utf-8")
    _log(f"overlap -> {OUT / 'overlap.json'} (pass: {overlap['pass']}, missing sets: {overlap['missing']})")
    return 0


def _sweep_one(job: tuple[dict[str, Any], str, tuple, str]) -> dict[str, Any]:
    from robo_jev.authors.a2.sanity import sweep, violations

    plan, field, factors, direction = job
    result = sweep(plan, field, factors)
    result.update({"scene": plan["scene"], "index": plan["index"], "direction": direction, "violations": violations(result["freqs"], direction)})
    return result


def cmd_sanity(args: argparse.Namespace) -> int:
    import multiprocessing as mp

    import numpy as np

    from robo_jev.authors.a2.sanity import SWEEPS, determinism, static_agreement, threshold_checks

    if not os.environ.get("INVOCATION_ID"):
        raise SystemExit("MuJoCo sanity runs only inside a capped systemd-run unit")
    out: dict[str, Any] = {"determinism": determinism(), "thresholds": threshold_checks()}
    _log(f"determinism {out['determinism']}; thresholds ok {sum(r['ok'] for r in out['thresholds'])}/{len(out['thresholds'])}")
    jobs = []
    for scene in SCENE_NAMES:
        plans = _read_jsonl(plan_path(scene))[: args.plans]
        for plan in plans:
            for field, factors, direction in SWEEPS[scene]:
                jobs.append((plan, field, factors, direction))
    with mp.get_context("fork").Pool(args.workers) as pool:
        sweeps = list(pool.imap_unordered(_sweep_one, jobs, chunksize=1))
    out["sweeps"] = sorted(sweeps, key=lambda r: (r["scene"], r["field"], r["index"]))
    out["sweep_summary"] = {f"{s}:{f}": {"sweeps": sum(1 for r in sweeps if r["scene"] == s and r["field"] == f),
                                         "label_level_violations": sum(r["violations"] for r in sweeps if r["scene"] == s and r["field"] == f)}
                            for s in SCENE_NAMES for f, _, _ in SWEEPS[s]}
    agreement = {}
    for scene in SCENE_NAMES:
        plans = {p["index"]: p for p in _read_jsonl(plan_path(scene))}
        rows = []
        for lab in _read_jsonl(current_labels_path(scene)):
            rows += static_agreement(plans[lab["index"]], lab)
        if not rows:
            continue
        margin = np.array([m for m, _ in rows])
        ok = np.array([a for _, a in rows])
        bins = [(-np.inf, -0.3), (-0.3, -0.1), (-0.1, 0.0), (0.0, 0.1), (0.1, 0.3), (0.3, np.inf)]
        agreement[scene] = {"rollouts": len(rows), "agree": round(float(ok.mean()), 4),
                            "by_margin": {f"{lo}..{hi}": {"n": int(((margin >= lo) & (margin < hi)).sum()),
                                                          "agree": round(float(ok[(margin >= lo) & (margin < hi)].mean()), 4) if ((margin >= lo) & (margin < hi)).any() else None}
                                          for lo, hi in bins}}
    out["static_agreement"] = agreement
    (OUT / "sanity.json").write_text(json.dumps(out, indent=1, default=float) + "\n", encoding="utf-8")
    _log(f"sanity -> {OUT / 'sanity.json'}; sweep violations {sum(r['violations'] for r in sweeps)} over {len(sweeps)} sweeps")
    return 0


#: A flagged turntable rollout counts as a physical "flung" topple (kept) when the item toppled and travelled at least this far (m).
FLUNG_MIN_M = 0.3


def unstable_path(scene: str) -> Path:
    return WORK / f"unstable-{scene}.json"


def _recheck_one(plan: dict[str, Any]) -> dict[str, Any]:
    import numpy as np

    from robo_jev.authors.a2 import catalog as C
    from robo_jev.authors.a2.generate import _SCENE_CACHE, K, member_variants
    from robo_jev.authors.a2.scenes import SCENES, hidden_values

    spec = SCENES[plan["scene"]]
    draws = spec.draws(np.random.default_rng(plan["draws_seed"]), K)
    flagged, seen = [], set()
    for member in ("base", "pair"):
        for variant, facts in member_variants(plan, member).items():
            key = spec.setup_key(facts)
            if key in seen:
                continue
            seen.add(key)
            for prior in C.PRIORS:
                vals = hidden_values(spec, spec.ranges(facts), draws, prior)
                for k in range(K):
                    r = spec.run(_SCENE_CACHE, facts, {n: float(vals[n][k]) for n in vals})
                    if r.get("unstable"):
                        flagged.append({"member": member, "variant": variant, "prior": prior, "draw": k, "outcome": r["outcome"],
                                        "moved_m": round(float(r.get("moved_m", 0.0)), 3), "max_tilt": round(float(r.get("max_tilt", 0.0)), 1)})
    flung = bool(flagged) and all(f["outcome"] == "topples" and f["moved_m"] >= FLUNG_MIN_M for f in flagged)
    return {"index": plan["index"], "flagged": flagged, "verdict": "flung" if flung else "unstable"}


def cmd_recheck(args: argparse.Namespace) -> int:
    """Re-run the groups whose labelling tripped the rise guard and keep the per-rollout detail (MuJoCo; capped unit only)."""
    import multiprocessing as mp

    if not os.environ.get("INVOCATION_ID"):
        raise SystemExit("MuJoCo re-check runs only inside a capped systemd-run unit")
    plans = {p["index"]: p for p in _read_jsonl(plan_path(args.scene))}
    todo = [plans[r["index"]] for r in _read_jsonl(current_labels_path(args.scene)) if r["unstable"]]
    with mp.get_context("fork").Pool(args.workers) as pool:
        results = sorted(pool.imap_unordered(_recheck_one, todo, chunksize=1), key=lambda r: r["index"])
    unstable_path(args.scene).write_text(json.dumps({"scene": args.scene, "flung_min_m": FLUNG_MIN_M, "groups": results}, indent=1) + "\n",
                                         encoding="utf-8")
    _log(f"{args.scene}: {len(results)} flagged groups re-checked, {sum(r['verdict'] == 'flung' for r in results)} are flung topples only")
    return 0


def cmd_summary(args: argparse.Namespace) -> int:
    """Markdown tables from manifest.json, stats.json, sanity.json and overlap.json (printed and written to summary.md)."""
    manifest = json.loads((OUT / "manifest.json").read_text(encoding="utf-8"))
    stats = json.loads((OUT / "stats.json").read_text(encoding="utf-8"))
    sanity = json.loads((OUT / "sanity.json").read_text(encoding="utf-8")) if (OUT / "sanity.json").is_file() else None
    overlap = json.loads((OUT / "overlap.json").read_text(encoding="utf-8"))
    L = [f"# A2 summary — {manifest['records']} records, sha256 {manifest['sha256'][:12]}…", ""]
    L += ["| scene | groups | records | dropped (unstable) | kept flung topples | CPU s (labelled groups) |", "|---|---|---|---|---|---|"]
    for scene, m in manifest["scenes"].items():
        cpu = stats.get("cpu", {}).get(scene, {})
        L.append(f"| {scene} | {m['groups']} | {m['records']} | {len(m['dropped_unstable'])} | {len(m.get('kept_flung_topples', []))} | "
                 f"{cpu.get('cpu_s')} ({cpu.get('groups_labelled')} groups, {cpu.get('rollouts')} rollouts) |")
    L += ["", "| scene | key mean | key bands L/M/H | severe bands L/M/H | score bands L/M/H | key hist (10 bins) |", "|---|---|---|---|---|---|"]
    for scene, e in stats["scenes"].items():
        b = lambda k: "/".join(f"{e['labels'][k]['bands'].get(x, 0):.2f}" for x in ("low", "middle", "high"))
        L.append(f"| {scene} | {e['labels']['key']['mean']:.3f} | {b('key')} | {b('severe')} | {b('score_cf')} | {e['labels']['key']['hist10']} |")
    L += ["", "| scene | robust key | severe | score | direction | compare | action | outcome | truth classes (clear-safe / clear-risky / unclear) |",
          "|---|---|---|---|---|---|---|---|---|"]
    for scene, e in stats["scenes"].items():
        r = e["robust_share"]
        t = e["truth_classes"]
        L.append(f"| {scene} | {r['key']:.2f} | {r['severe']:.2f} | {r['score_cf']:.2f} | {r['direction']:.2f} | {r['compare']:.2f} | {r['action']:.2f} | "
                 f"{r['outcome']:.2f} | {t.get('clear_safe', 0)} / {t.get('clear_risky', 0)} / {t.get('unclear', 0)} |")
    L += ["", "| scene | pair kinds | one-factor share | mean abs key move | moves ≥ 0.1 | direction same over priors | no reversal |", "|---|---|---|---|---|---|---|"]
    for scene, e in stats["scenes"].items():
        pr = e["pairs"]
        L.append(f"| {scene} | {pr['kinds']} | {pr['one_factor_share']:.2f} | {pr['mean_abs_key_delta']:.3f} | {pr['moves_at_least_0.1']:.2f} | "
                 f"{pr['direction_same_over_priors']:.2f} | {pr['no_reversal']:.2f} |")
    L += ["", "| scene | direction argmax | compare argmax | action argmax | outcome argmax | static aim vs key (base members) |", "|---|---|---|---|---|---|"]
    for scene, e in stats["scenes"].items():
        c = e["choice_argmax"]
        L.append(f"| {scene} | {c['direction']} | {c['compare']} | {c['action']} | {c['outcome']} | {e['static_aim_vs_key']} |")
    L += ["", "| scene | state + questions median / p90 / max | model prompt per question median / max |", "|---|---|---|"]
    for scene, e in stats["prompt_lengths"]["scenes"].items():
        a, b = e["state_plus_questions"], e["model_prompt_per_question_chat_direct"]
        L.append(f"| {scene} | {a['median']:.0f} / {a['p90']:.0f} / {a['max']} | {b['median']:.0f} / {b['max']} |")
    L += ["", f"Loader: {stats['loader']}. Pair text check: {stats.get('pair_text_check')}. Scorer checks: {stats['scorer_checks']}.", ""]
    L += [f"Overlap pass: {overlap['pass']} (missing sets: {overlap['missing']}).", "", "| set | records | shared ids / requests / states / instructions / sentences / 5-grams |", "|---|---|---|"]
    for name, row in overlap["sets"].items():
        L.append(f"| {name} | {row.get('records')} | {row.get('shared_ids')} / {row.get('shared_requests')} / {row.get('shared_states')} / "
                 f"{row.get('shared_instructions')} / {row.get('shared_sentences')} / {row.get('shared_5grams', '—')} |")
    if sanity:
        L += ["", f"Determinism: {sanity['determinism']}.", "", "| law | check | expect | got | ok |", "|---|---|---|---|---|"]
        for row in sanity["thresholds"]:
            L.append(f"| {row['law']} | {row['check']} | {row['expect']} | {row['got']} | {row['ok']} |")
        L += ["", "| sweep | sweeps | label-level violations |", "|---|---|---|"]
        for key, row in sanity["sweep_summary"].items():
            L.append(f"| {key} | {row['sweeps']} | {row['label_level_violations']} |")
        L += ["", "| scene | rollouts | agree with the static law | by static margin |", "|---|---|---|---|"]
        for scene, row in sanity["static_agreement"].items():
            L.append(f"| {scene} | {row['rollouts']} | {row['agree']:.3f} | " + "; ".join(f"{k}: {v['agree']} (n {v['n']})" for k, v in row["by_margin"].items()) + " |")
    text = "\n".join(L) + "\n"
    (OUT / "summary.md").write_text(text, encoding="utf-8")
    print(text)
    return 0


# --------------------------------------------------------------------------
# Wallpress relabel after the physics audit (stale bounding volumes)
# --------------------------------------------------------------------------

RELABEL_WHY = ("robo_jev.outside.physics.Scene.set_hidden writes the load's centre of mass (body_ipos) at run time, but MuJoCo keeps a "
               "body's bounding volumes (bvh_aabb) in its compile-time inertial frame and its midphase places them with the current one; "
               "only body pairs that go through the midphase are exposed (one body with >= 2 geoms). A2 wallpress: the load-world pair (the "
               "world holds the floor and the wall); turntable and tiltback: no midphase pair.")
RELABEL_FIX = ("WallpressScene compiles Scene(..., refresh_bvh=True): the load's bounding volumes are re-expressed after every centre-of-mass "
               "write (main b94d734); every wallpress group relabelled with the same plans, draws, variants, priors and K = 32")


def _flips(old: str, new: str, spec: Any, events: Sequence[str]) -> tuple[int, int]:
    """(event -> no event, no event -> event) between two outcome-code strings of the same draws."""
    idx = {str(spec.outcomes.index(e)) for e in events}
    lost = sum(1 for a, b in zip(old, new) if a in idx and b not in idx)
    gained = sum(1 for a, b in zip(old, new) if a not in idx and b in idx)
    return lost, gained


def cmd_compare_labels(args: argparse.Namespace) -> int:
    """Pre-fix vs relabelled wallpress (no MuJoCo): rollout-level outcome changes and record-level label changes of the used groups."""
    from collections import Counter

    import numpy as np

    from robo_jev.authors.a2 import catalog as C
    from robo_jev.authors.a2.generate import assemble_group
    from robo_jev.authors.a2.scenes import SCENES
    from robo_jev.decisions.records import file_sha256

    scene = "wallpress"
    spec = SCENES[scene]
    old = {r["index"]: r for r in _read_jsonl(labels_path(scene))}
    new = {r["index"]: r for r in _read_jsonl(labels_path(scene, LABEL_TAGS[scene]))}
    plans = {p["index"]: p for p in _read_jsonl(plan_path(scene))}
    common = sorted(set(old) & set(new))
    if len(common) != len(plans):
        raise SystemExit(f"{scene}: {len(common)} groups in both label files, {len(plans)} planned — relabel incomplete")
    # rollout level (every distinct setup of every planned group, three priors, K draws)
    total = changed = identical_groups = 0
    transitions: Counter = Counter()
    key_lost = key_gained = sev_lost = sev_gained = 0
    metric_abs = []
    for index in common:
        a, b = old[index]["setups"], new[index]["setups"]
        if set(a) != set(b):
            raise SystemExit(f"{scene} group {index}: the two runs have different setups")
        same_group = True
        for key in a:
            for prior in C.PRIORS:
                ca, cb = a[key]["codes"][prior], b[key]["codes"][prior]
                total += len(ca)
                for x, y in zip(ca, cb):
                    if x != y:
                        changed += 1
                        transitions[f"{spec.outcomes[int(x)]}->{spec.outcomes[int(y)]}"] += 1
                        same_group = False
                lo, ga = _flips(ca, cb, spec, spec.key)
                key_lost, key_gained = key_lost + lo, key_gained + ga
                lo, ga = _flips(ca, cb, spec, spec.severe)
                sev_lost, sev_gained = sev_lost + lo, sev_gained + ga
                metric_abs += [abs(float(u) - float(v)) for u, v in zip(a[key]["metric"][prior], b[key]["metric"][prior])]
        identical_groups += same_group
    metric_abs = np.array(metric_abs)
    rollout = {"groups": len(common), "rollouts": total, "outcome_changed": changed, "outcome_changed_share": round(changed / total, 6),
               "transitions": dict(transitions.most_common()), "key_event_lost": key_lost, "key_event_gained": key_gained,
               "key_flip_share": round((key_lost + key_gained) / total, 6), "severe_event_lost": sev_lost, "severe_event_gained": sev_gained,
               "groups_bit_identical_codes": identical_groups,
               "drop_m_abs_change": {"mean": round(float(metric_abs.mean()), 6), "p99": round(float(np.percentile(metric_abs, 99)), 6),
                                     "max": round(float(metric_abs.max()), 6), "nonzero_share": round(float(np.mean(metric_abs > 0)), 6)}}
    # record level (the groups the training file uses)
    v5 = _load_verdict_v05()
    used = json.loads((OUT / "manifest.json").read_text(encoding="utf-8"))["scenes"][scene]
    indices = [i for i in sorted(plans) if i not in set(used["dropped_unstable"])][: used["groups"]]
    scalar_abs: dict[str, list[float]] = {}
    scalar_signed: dict[str, list[float]] = {}
    band_changed: Counter = Counter()
    n_scalar: Counter = Counter()
    truth_changes: Counter = Counter()
    robust_changes = 0
    choice_tv: dict[str, list[float]] = {}
    argmax_changed: Counter = Counter()
    n_choice: Counter = Counter()
    records_changed = 0
    for index in indices:
        before = assemble_group(plans[index], old[index], git=None, mujoco_version="")
        after = assemble_group(plans[index], new[index], git=None, mujoco_version="")
        for ra, rb in zip(before, after):
            records_changed += ra["labels"] != rb["labels"] or ra["meta"]["labels_by_prior"] != rb["meta"]["labels_by_prior"]
            for name, kind in ra["meta"]["question_kind"].items():
                la, lb = ra["labels"][name], rb["labels"][name]
                if kind in ("key", "severe", "score_cf"):
                    field = "value" if kind == "score_cf" else "p_yes"
                    d = float(lb[field]) - float(la[field])
                    scalar_abs.setdefault(kind, []).append(abs(d))
                    scalar_signed.setdefault(kind, []).append(d)
                    n_scalar[kind] += 1
                    band_changed[kind] += _band(float(la[field])) != _band(float(lb[field]))
                    ta, tb = v5.truth_class(ra, name), v5.truth_class(rb, name)
                    if ta != tb:
                        truth_changes[f"{ta}->{tb}"] += 1
                    robust_changes += ra["meta"]["robustness"][name]["robust"] != rb["meta"]["robustness"][name]["robust"]
                else:
                    pa, pb = la["probabilities"], lb["probabilities"]
                    choice_tv.setdefault(kind, []).append(0.5 * sum(abs(float(pa[k]) - float(pb[k])) for k in pa))
                    n_choice[kind] += 1
                    argmax_changed[kind] += max(pa, key=pa.get) != max(pb, key=pb.get)
    record = {
        "groups": len(indices), "records": 2 * len(indices), "records_with_any_label_change": records_changed,
        "scalar": {k: {"n": n_scalar[k], "mean_abs_delta": round(float(np.mean(scalar_abs[k])), 6),
                       "mean_signed_delta": round(float(np.mean(scalar_signed[k])), 6), "max_abs_delta": round(float(np.max(scalar_abs[k])), 6),
                       "band_changed": band_changed[k], "band_changed_share": round(band_changed[k] / n_scalar[k], 6)} for k in n_scalar},
        "scalar_all": {"n": sum(n_scalar.values()), "mean_abs_delta": round(float(np.mean([x for v in scalar_abs.values() for x in v])), 6),
                       "mean_signed_delta": round(float(np.mean([x for v in scalar_signed.values() for x in v])), 6),
                       "band_changed_share": round(sum(band_changed.values()) / sum(n_scalar.values()), 6)},
        "truth_class_changes": dict(truth_changes), "robust_flag_changes": robust_changes,
        "choice": {k: {"n": n_choice[k], "mean_total_variation": round(float(np.mean(choice_tv[k])), 6), "argmax_changed": argmax_changed[k]}
                   for k in n_choice},
    }
    headline = (f"{changed} of {total} wallpress rollouts ({100 * changed / total:.3f} %) changed outcome; key-event flips {key_lost} lost / "
                f"{key_gained} gained; {records_changed} of {2 * len(indices)} used records changed any label; yes/no+score mean |delta| "
                f"{record['scalar_all']['mean_abs_delta']:.4f}, band changes {100 * record['scalar_all']['band_changed_share']:.2f} %, "
                f"truth-class changes {sum(truth_changes.values())}")
    out = {"scene": scene, "why": RELABEL_WHY, "fix": RELABEL_FIX, "previous_train_sha256": args.previous_sha or file_sha256(OUT / "train.jsonl"),
           "old_labels": str(labels_path(scene).relative_to(REPO)), "new_labels": str(labels_path(scene, LABEL_TAGS[scene]).relative_to(REPO)),
           "rollout_level": rollout, "record_level": record, "headline": headline}
    (OUT / "relabel-wallpress.json").write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    _log(headline)
    return 0


class _FlaggedCache(dict):
    """A scene cache whose new scenes re-express the load's bounding volumes after every centre-of-mass write (the audit's flag)."""

    def __setitem__(self, key: Any, value: Any) -> None:
        value.scene.refresh_bvh = True
        super().__setitem__(key, value)


def _bitcheck_one(job: tuple[str, dict[str, Any], dict[str, Any]]) -> dict[str, Any]:
    import numpy as np

    from robo_jev.authors.a2 import catalog as C
    from robo_jev.authors.a2.generate import K, member_variants
    from robo_jev.authors.a2.scenes import SCENES, Wallpress, hidden_values

    mode, plan, stored = job
    spec = SCENES[plan["scene"]]
    draws = spec.draws(np.random.default_rng(plan["draws_seed"]), K)
    if mode == "flag":  # turntable / tiltback: without vs with the flag, and both vs the stored codes
        plain, flagged = {}, _FlaggedCache()
        runs = (("plain", spec, plain), ("flagged", spec, flagged))
    else:  # wallpress: the pre-fix rollouts (flag off) must reproduce the stored pre-fix codes; and flag off vs on, full results
        off = Wallpress()
        off.refresh_bvh = False
        runs = (("flag_off", off, {}), ("flag_on", spec, {}))
    out = {"scene": plan["scene"], "index": plan["index"], "rollouts": 0, "identical_with_and_without_flag": 0, "match_stored_codes": 0,
           "match_stored_metric": 0}
    seen = set()
    for member in ("base", "pair"):
        for facts in member_variants(plan, member).values():
            key = spec.setup_key(facts)
            if key in seen:
                continue
            seen.add(key)
            metric = {"turntable": "moved_m", "tiltback": "pull_n", "wallpress": "drop_m"}[plan["scene"]]
            for prior in C.PRIORS:
                vals = hidden_values(spec, spec.ranges(facts), draws, prior)
                for k in range(K):
                    values = {n: float(vals[n][k]) for n in vals}
                    results = [s.run(cache, facts, values) for _, s, cache in runs]
                    out["rollouts"] += 1
                    out["identical_with_and_without_flag"] += results[0] == results[1]
                    code = str(spec.outcomes.index(results[0]["outcome"]))
                    out["match_stored_codes"] += code == stored["setups"][key]["codes"][prior][k]
                    out["match_stored_metric"] += round(float(results[0][metric]), 4) == stored["setups"][key]["metric"][prior][k]
    return out


def cmd_bitcheck(args: argparse.Namespace) -> int:
    """MuJoCo (capped unit): midphase pairs of the three scenes; turntable and tiltback rollouts with and without the bounding-volume
    flag on a sample of groups (must be bit-identical and reproduce the stored codes); wallpress with the flag off on a sample (must
    reproduce the stored pre-fix codes, i.e. the relabel changes nothing but the flag)."""
    import multiprocessing as mp

    import mujoco

    from robo_jev.authors.a2 import physics as PH

    if not os.environ.get("INVOCATION_ID"):
        raise SystemExit("MuJoCo bit check runs only inside a capped systemd-run unit")
    names = {}
    for scene, obj in (("turntable", PH.TurntableScene((0.065, 0.065, 0.24), 0.9)), ("tiltback", PH.TiltbackScene((0.32, 0.8, 1.8), 50.0)),
                       ("wallpress", PH.WallpressScene((0.02, 0.5, 0.6), 5.0, 0.5))):
        model = obj.scene.model
        name = lambda b: mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b) or "world"
        names[scene] = [(name(a), name(b)) for a, b in obj.scene.midphase_pairs()]
    jobs = []
    for scene, mode in (("turntable", "flag"), ("tiltback", "flag"), ("wallpress", "off")):
        plans = {p["index"]: p for p in _read_jsonl(plan_path(scene))}
        stored = {r["index"]: r for r in _read_jsonl(labels_path(scene))}  # the stored (pre-fix) labels of every scene
        picked = sorted(plans)[:: max(1, len(plans) // (args.groups if mode == "flag" else args.wallpress_groups))]
        jobs += [(mode, plans[i], stored[i]) for i in picked]
    with mp.get_context("fork").Pool(args.workers) as pool:
        results = list(pool.imap_unordered(_bitcheck_one, jobs, chunksize=1))
    summary = {}
    for scene in ("turntable", "tiltback", "wallpress"):
        rows = [r for r in results if r["scene"] == scene]
        summary[scene] = {"groups": len(rows), "rollouts": sum(r["rollouts"] for r in rows),
                          "identical_with_and_without_flag": sum(r["identical_with_and_without_flag"] for r in rows),
                          "match_stored_codes": sum(r["match_stored_codes"] for r in rows),
                          "match_stored_metric_4dp": sum(r["match_stored_metric"] for r in rows)}
    out = {"midphase_pairs": names, "check": summary,
           "note": "turntable / tiltback: every rollout of the sampled groups (all setups, three priors, K = 32) run without and with "
                   "refresh_bvh; full result dicts compared. wallpress: the sampled groups re-run with the flag off (against the stored "
                   "pre-fix codes) and on (full result dicts compared with the flag-off run)."}
    (OUT / "bitcheck.json").write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    _log(f"midphase pairs {names}; {summary}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("plan")
    p.add_argument("--scene", choices=SCENE_NAMES, required=True)
    p.add_argument("--groups", type=int, required=True)
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--tag", default="", help="a separate plan / label file pair (e.g. a pilot)")
    p.set_defaults(fn=cmd_plan)
    p = sub.add_parser("label")
    p.add_argument("--scene", choices=SCENE_NAMES, required=True)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--allow-outside-unit", action="store_true", help="tests only")
    p.add_argument("--tag", default="")
    p.add_argument("--label-tag", default=None, help="write the labels to labels-<scene>-<label tag>.jsonl (plans from the --tag file)")
    p.set_defaults(fn=cmd_label)
    p = sub.add_parser("assemble")
    p.add_argument("--scenes", nargs="+", default=list(SCENE_NAMES))
    p.add_argument("--groups", type=int, default=625)
    p.add_argument("--git", default=None)
    p.add_argument("--label-git", default=None, help="the commit whose physics produced the labels (recorded in the manifest)")
    p.add_argument("--scene-git", action="append", default=None, help="SCENE=SHA: the generator commit of one scene's records")
    p.set_defaults(fn=cmd_assemble)
    p = sub.add_parser("recheck")
    p.add_argument("--scene", choices=SCENE_NAMES, required=True)
    p.add_argument("--workers", type=int, default=4)
    p.set_defaults(fn=cmd_recheck)
    p = sub.add_parser("compare-labels")
    p.add_argument("--previous-sha", default=None, help="sha256 of the train.jsonl built from the pre-fix labels")
    p.set_defaults(fn=cmd_compare_labels)
    p = sub.add_parser("bitcheck")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--groups", type=int, default=40, help="sampled groups per scene for turntable and tiltback")
    p.add_argument("--wallpress-groups", type=int, default=20)
    p.set_defaults(fn=cmd_bitcheck)
    p = sub.add_parser("report")
    p.set_defaults(fn=cmd_report)
    p = sub.add_parser("summary")
    p.set_defaults(fn=cmd_summary)
    p = sub.add_parser("sanity")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--plans", type=int, default=15, help="group plans per scene to sweep")
    p.set_defaults(fn=cmd_sanity)
    args = parser.parse_args(argv)
    if getattr(args, "workers", 1) > 8:
        raise SystemExit("at most 8 workers")
    return int(args.fn(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
