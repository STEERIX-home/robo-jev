#!/usr/bin/env python
"""Task T8 — outside physics tests (no training).

    # T8a: PIQA validation → decision requests (artifacts/t8/T8a.jsonl + manifest). Source files are read from --source
    # (downloaded from the PIQA homepage with --download when missing).
    python scripts/t8.py piqa --source artifacts/scratch/t8/piqa [--download]
    # T8b: robot moments labelled by MuJoCo rollouts (CPU; run in a capped unit) → artifacts/t8/T8b.jsonl + manifest
    python scripts/t8.py moments --workers 12
    # physics sanity report (monotonicity, analytic tipping, determinism) → artifacts/t8/T8b-sanity.json
    python scripts/t8.py sanity
    # scoring tables (CPU): our models' diagnosis predictions + the OpenRouter predictions written by scripts/bench.py
    python scripts/t8.py score

Our models are scored with ``scripts/diagnose_decisions.py`` (prompt v0c-p3, counterbalanced, ``--checkpoint`` for the
adapters); the OpenRouter models with ``scripts/bench.py run --config configs/bench/t8.yaml`` (data collection denied,
its own ledger capped at $1.00). The T8 files are sealed: no training configuration points at them.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import subprocess
import sys
import urllib.request
from pathlib import Path
from typing import Any, Sequence

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

SCRIPT_VERSION = "t8-1"
DEFAULT_OUT = REPO / "artifacts" / "t8"


def _log(message: str) -> None:
    print(f"[t8 {dt.datetime.now().strftime('%H:%M:%S')}] {message}", file=sys.stderr, flush=True)


def _git_sha() -> str | None:
    try:
        return subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True).stdout.strip() or None
    except (OSError, subprocess.CalledProcessError):
        return None


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(payload, ensure_ascii=False, indent=1, default=str) + "\n", encoding="utf-8")
    temp.replace(path)


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    with temp.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    temp.replace(path)


def cmd_piqa(args: argparse.Namespace) -> int:
    from robo_jev.bench.define import ids_sha256
    from robo_jev.decisions.records import file_sha256
    from robo_jev.outside.piqa import CONVERTER_VERSION, LICENSE, SOURCE, convert, read_source

    source = Path(args.source)
    source.mkdir(parents=True, exist_ok=True)
    for name, url in SOURCE["files"].items():
        target = source / name
        if not target.is_file():
            if not args.download:
                raise SystemExit(f"{target}: missing — pass --download to fetch it from {url}")
            _log(f"downloading {url} → {target}")
            with urllib.request.urlopen(url, timeout=120) as response:  # noqa: S310 — fixed public URL
                target.write_bytes(response.read())
    records, summary = convert(read_source(source / "valid.jsonl", source / "valid-labels.lst"))
    out = Path(args.out) / "T8a.jsonl"
    _write_jsonl(out, records)
    card = source / "hf-README.md"
    manifest = {
        "set": "T8a", "what": "PIQA validation (goal + two solutions; pick the one that works) as decision requests", "converter": CONVERTER_VERSION,
        "source": {**SOURCE, "sha256": {name: file_sha256(source / name) for name in SOURCE["files"]},
                   "dataset_card_sha256": file_sha256(card) if card.is_file() else None},
        "license": LICENSE, "summary": summary, "file": str(out.relative_to(REPO)) if out.is_relative_to(REPO) else str(out),
        "sha256": file_sha256(out), "ids_sha256": ids_sha256(r["id"] for r in records), "sealed": True,
        "script_version": SCRIPT_VERSION, "git": _git_sha(), "created": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
    }
    _write_json(Path(args.out) / "T8a.manifest.json", manifest)
    _log(f"T8a: {summary['records']} records (sol1 {summary['label_sol1']} · sol2 {summary['label_sol2']}) → {out} · sha256 {manifest['sha256'][:12]}")
    return 0


#: T8b composition: moments kept per family (equal numbers per majority outcome of the planned action) and candidates labelled.
MOMENT_TARGETS = {"deck_brake": 150, "deck_turn": 90, "stack": 120, "push": 120}
OVERSAMPLE = {"deck_brake": 1.8, "deck_turn": 1.8, "stack": 2.4, "push": 1.8}


def _label_job(job: tuple[str, int, int, int]) -> tuple[Any, dict[str, Any], float]:
    import time

    from robo_jev.outside.moments import candidate, label_moment

    family, index, master, k = job
    started = time.perf_counter()
    moment = candidate(family, index, master)
    return moment, label_moment(moment, k=k), time.perf_counter() - started


def _pool_map(func: Any, jobs: list[Any], workers: int) -> list[Any]:
    if workers <= 1:
        return [func(job) for job in jobs]
    import multiprocessing

    with multiprocessing.get_context("fork").Pool(workers) as pool:
        return pool.map(func, jobs, chunksize=2)


def cmd_moments(args: argparse.Namespace) -> int:
    import collections
    import statistics

    import mujoco

    from robo_jev.bench.define import ids_sha256
    from robo_jev.decisions.records import file_sha256
    from robo_jev.outside import physics as ph
    from robo_jev.outside.moments import FAMILIES, GENERATOR, OUTCOMES, moment_record, select_balanced

    targets = {family: int(round(MOMENT_TARGETS[family] * args.scale)) for family in FAMILIES}
    jobs = [(family, index, args.seed, args.k) for family in FAMILIES for index in range(int(math.ceil(targets[family] * OVERSAMPLE[family])))]
    _log(f"labelling {len(jobs)} candidates with K={args.k} on {args.workers} workers (targets {targets})")
    started = dt.datetime.now()
    results = _pool_map(_label_job, jobs, args.workers)
    seconds = (dt.datetime.now() - started).total_seconds()
    _log(f"labelled {len(results)} candidates in {seconds:.0f} s")
    labelled = [(moment, label) for moment, label, _ in results]
    kept, selection = select_balanced(labelled, targets)
    numbers: dict[str, int] = collections.Counter()
    records = []
    git = _git_sha()
    for moment, label in kept:
        records.append(moment_record(moment, label, number=numbers[moment.family], mujoco_version=mujoco.__version__, git=git))
        numbers[moment.family] += 1
    out = Path(args.out) / "T8b.jsonl"
    _write_jsonl(out, records)
    audit = [{"family": m.family, "index": m.index, "seed": m.seed, "target": m.target, "majority": lab["majority"],
              "outcomes": {name: v["freq"] for name, v in lab["variants"].items()}, "seconds": round(sec, 3), "kept": any(m is km for km, _ in kept)}
             for m, lab, sec in results]
    _write_jsonl(Path(args.out) / "T8b.candidates.jsonl", audit)
    # composition and label statistics
    composition: dict[str, Any] = {}
    for family in FAMILIES:
        rows = [r for r in records if r["meta"]["scene"] == family]
        majority = collections.Counter(r["meta"]["majority"] for r in rows)
        values = [float(l.get("p_yes", l.get("value"))) for r in rows for l in r["labels"].values() if l["type"] in ("noul", "score")]
        composition[family] = {
            "moments": len(rows), "majority": {o: majority.get(o, 0) for o in OUTCOMES[family]},
            "questions": sum(len(r["request"]["questions"]) for r in rows),
            "binary_labels": len(values), "uncertain_share": (sum(1 for v in values if 0.1 <= v <= 0.9) / len(values)) if values else None,
            "label_mean": statistics.fmean(values) if values else None,
            "domains": dict(collections.Counter(r["meta"]["domain"] for r in rows)),
            "loads": dict(collections.Counter(r["meta"]["visible"].get("load") or r["meta"]["visible"].get("item") or r["meta"]["visible"].get("box")
                                              for r in rows)),
        }
    examples = {}
    for family in FAMILIES:
        first = next(((m, lab) for m, lab in kept if m.family == family), None)
        if first is None:
            continue
        v = first[0].visible
        if family.startswith("deck"):
            xml = ph.deck_xml(v["shape"], v["dims"], v["mass"])
        elif family == "stack":
            xml = ph.stack_xml(v["dims"], v["mass"], (v["support_top"], max(0.3, v["dims"][1] + 0.1), v["support_height"]))
        else:
            xml = ph.push_xml(v["dims"], v["mass"], v["push_height"])
        examples[family] = xml
    q_types = collections.Counter(q["type"] for r in records for q in r["request"]["questions"].values())
    manifest = {
        "set": "T8b", "what": "robot moments labelled by MuJoCo rollouts (outside our closed-form engine); labels = outcome frequencies over K hidden draws",
        "generator": GENERATOR, "mujoco": mujoco.__version__, "k": args.k, "master_seed": args.seed, "scale": args.scale, "oversample": OVERSAMPLE,
        "targets": targets, "selection": selection, "composition": composition, "records": len(records),
        "questions": {"total": sum(q_types.values()), **dict(sorted(q_types.items()))},
        "physics": {"timestep": ph.TIMESTEP, "integrator": "implicitfast", "cone": "elliptic", "impratio": 10, "solref": ph.SOLREF, "condim": 3,
                    "multiccd": True, "tip_deg": ph.TIP_DEG, "slide_m": ph.SLIDE_M, "fall_tilt_deg": ph.FALL_TILT_DEG, "deck_mass": ph.DECK_MASS,
                    "bar_friction": ph.BAR_FRICTION, "hidden_draws": "scrambled Latin hypercube per moment (seed = moment seed ^ 0x5EED)"},
        "xml_examples": examples, "file": "artifacts/t8/T8b.jsonl", "sha256": file_sha256(out), "ids_sha256": ids_sha256(r["id"] for r in records),
        "candidates_file": "artifacts/t8/T8b.candidates.jsonl", "label_seconds": round(seconds, 1), "workers": args.workers, "sealed": True,
        "script_version": SCRIPT_VERSION, "git": git, "created": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
    }
    _write_json(Path(args.out) / "T8b.manifest.json", manifest)
    _log(f"T8b: {len(records)} moments, {manifest['questions']['total']} questions → {out} · sha256 {manifest['sha256'][:12]} · selection {selection}")
    return 0


def _sanity_job(job: tuple[str, Any]) -> Any:
    from robo_jev.outside import sanity

    kind, payload = job
    return getattr(sanity, kind)(**payload)


def cmd_sanity(args: argparse.Namespace) -> int:
    import mujoco

    from robo_jev.outside import sanity

    jobs = sanity.jobs(master=args.seed, k=args.k, moments=args.moments)
    _log(f"sanity: {len(jobs)} jobs on {args.workers} workers")
    results = _pool_map(_sanity_job, jobs, args.workers)
    report = sanity.summarize(jobs, results)
    report.update({"mujoco": mujoco.__version__, "k": args.k, "master_seed": args.seed, "script_version": SCRIPT_VERSION, "git": _git_sha(),
                   "created": dt.datetime.now().astimezone().isoformat(timespec="seconds")})
    _write_json(Path(args.out) / "T8b-sanity.json", report)
    _log(f"sanity → {Path(args.out) / 'T8b-sanity.json'}: " + json.dumps(report["verdict"]))
    return 0 if report["verdict"]["passed"] else 2


#: Our models: tag in T8 report files, label, bench-v0 tag.
OURS = (("base-2b", "Qwen3.5-2B (untrained)", "qwen3.5-2b"), ("v0-2b", "ours v0 (2B)", "robojev-v0"), ("v01mix-2b", "ours v0.1-mix (2B)", "robojev-v0.1-mix"))
BASE_TAG = "base-2b"


def _load_models(args: argparse.Namespace, sets: list[str]) -> list[dict[str, Any]]:
    """Our three models (diagnosis predictions) and the OpenRouter models (bench.py predictions), in report order."""
    from robo_jev.bench.define import load_config
    from robo_jev.bench.openrouter import safe_name
    from robo_jev.decisions.records import read_records

    models = []
    for tag, label, bench_tag in OURS:
        entry = {"tag": tag, "label": label, "role": "baseline" if tag == BASE_TAG else "ours", "bench_tag": bench_tag, "predictions": {}}
        for set_name in sets:
            path = Path(args.reports) / f"{set_name}-{tag}.predictions.jsonl"
            if path.is_file():
                entry["predictions"][set_name] = read_records(path)
        models.append(entry)
    config = load_config(args.config)
    for model in (config.get("models") or {}).get("default") or []:
        name = safe_name(model["id"])
        entry = {"tag": name, "label": model["id"], "role": "api", "bench_tag": name, "predictions": {}}
        for set_name in sets:
            path = Path(args.api) / name / f"{set_name}.predictions.jsonl"
            if path.is_file():
                entry["predictions"][set_name] = read_records(path)
        models.append(entry)
    return models


def cmd_score(args: argparse.Namespace) -> int:
    from robo_jev.decisions.records import file_sha256, read_records
    from robo_jev.outside.score import FAMILY_ORDER, composition, fmt, fmt_ci, paired_skill, piqa_metrics, piqa_paired, skill_rows, t8b_metrics

    sets = {name: read_records(Path(args.data) / f"{name}.jsonl") for name in ("T8a", "T8b") if (Path(args.data) / f"{name}.jsonl").is_file()}
    models = _load_models(args, list(sets))
    bench = json.loads(Path(args.bench_results).read_text(encoding="utf-8")) if Path(args.bench_results).is_file() else {"models": []}
    bench_skill = {m["tag"]: ((m.get("pooled") or {}).get("unseen") or {}).get("skill") for m in bench["models"]}
    base = next(m for m in models if m["tag"] == BASE_TAG)
    results: dict[str, Any] = {"kind": "t8-results", "created": dt.datetime.now().astimezone().isoformat(timespec="seconds"), "n_boot": args.n_boot,
                               "sets": {name: {"records": len(recs), "sha256": file_sha256(Path(args.data) / f"{name}.jsonl")} for name, recs in sets.items()},
                               "models": []}
    if "T8b" in sets:
        results["sets"]["T8b"]["composition"] = composition(sets["T8b"])
    for model in models:
        entry = {k: model[k] for k in ("tag", "label", "role", "bench_tag")}
        entry["bench_v0_skill"] = bench_skill.get(model["bench_tag"])
        if "T8a" in model["predictions"]:
            entry["T8a"] = piqa_metrics(sets["T8a"], model["predictions"]["T8a"], n_boot=args.n_boot)
            if model is not base and "T8a" in base["predictions"]:
                entry["T8a_vs_base"] = piqa_paired(sets["T8a"], model["predictions"]["T8a"], base["predictions"]["T8a"], n_boot=args.n_boot)
        if "T8b" in model["predictions"]:
            entry["T8b"] = t8b_metrics(sets["T8b"], model["predictions"]["T8b"], n_boot=args.n_boot)
            if model is not base and "T8b" in base["predictions"]:
                entry["T8b_vs_base"] = {r: paired_skill(skill_rows(sets["T8b"], model["predictions"]["T8b"], reading=r),
                                                        skill_rows(sets["T8b"], base["predictions"]["T8b"], reading=r), n_boot=args.n_boot)
                                        for r in ("single_order", "order_averaged")}
                entry["T8b_vs_base_by_family"] = {f: paired_skill([x for x in skill_rows(sets["T8b"], model["predictions"]["T8b"], reading="single_order") if x["family"] == f],
                                                                   [x for x in skill_rows(sets["T8b"], base["predictions"]["T8b"], reading="single_order") if x["family"] == f],
                                                                   n_boot=args.n_boot) for f in FAMILY_ORDER}
        results["models"].append(entry)
    ledger = Path(args.ledger)
    if ledger.is_file():
        data = json.loads(ledger.read_text(encoding="utf-8"))
        results["openrouter"] = {"ledger": str(ledger), "totals": data.get("totals"),
                                 "models": {k: {"spent_usd": v.get("spent_usd"), "calls": v.get("calls")} for k, v in (data.get("models") or {}).items()}}
    out = Path(args.out)
    _write_json(out, results)
    out.with_suffix(".md").write_text(_render(results, fmt, fmt_ci, FAMILY_ORDER), encoding="utf-8")
    _log(f"scored {len(models)} models on {list(sets)} → {out} (+ .md)")
    return 0


def _render(results: dict[str, Any], fmt: Any, fmt_ci: Any, families: Sequence[str]) -> str:
    lines = ["# T8 — outside physics tests: results", "",
             "Single-order reading (deployment condition) unless noted: our models' numbers are expectations over balanced option orders; "
             "OpenRouter models were asked once in the stored order (data collection denied). Intervals are record-bootstrap 95 %.", ""]
    comp = (results["sets"].get("T8b") or {}).get("composition") or {}
    if comp:
        lines += ["## T8b composition", "", "| family | moments | questions | majority outcome of the planned action |", "|---|---|---|---|"]
        for family in families:
            c = comp.get(family) or {}
            lines.append(f"| {family} | {c.get('moments', 0)} | {c.get('questions', 0)} | "
                         + ", ".join(f"{k} {v}" for k, v in sorted((c.get("majority") or {}).items())) + " |")
        lines.append("")
    lines += ["## Scores", "",
              "| model | T8a PIQA acc ↑ | T8a acc (order-avg) | T8a first-option share | T8b physics skill ↑ | T8b skill (order-avg) | T8b yes/no Brier ↓ "
              "| T8b score Brier ↓ | T8b outcome acc ↑ | T8b answered | bench-v0 skill T2–T4 |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
    for m in results["models"]:
        a, b = m.get("T8a") or {}, m.get("T8b") or {}
        a1, a2 = a.get("single_order") or {}, a.get("order_averaged") or {}
        b1, b2 = b.get("single_order") or {}, b.get("order_averaged") or {}
        name = f"**{m['label']}**" if m["role"] != "api" else m["label"]
        lines.append(f"| {name} | {fmt(a1.get('accuracy'))}{fmt_ci(a1.get('accuracy_ci'))} | {fmt(a2.get('accuracy'))} | {fmt(a.get('first_option_share'), 2)} | "
                     f"{fmt((b1.get('pooled') or {}).get('skill'))}{fmt_ci(b1.get('skill_ci'), signed=True)} | {fmt((b2.get('pooled') or {}).get('skill'))} | "
                     f"{fmt(b1.get('noul_brier'))} | {fmt(b1.get('score_brier'))} | {fmt(b1.get('outcome_accuracy'))} | "
                     f"{fmt(b1.get('coverage'), 2)} | {fmt(m.get('bench_v0_skill'))} |")
    lines += ["", "## T8b skill by family and question kind (single order)", "",
              "| model | " + " | ".join(families) + " | planned | counterfactual | yes/no only | score only |", "|---|" + "---|" * (len(families) + 4)]
    for m in results["models"]:
        b1 = (m.get("T8b") or {}).get("single_order") or {}
        cells = [fmt(((b1.get("by_family") or {}).get(f) or {}).get("skill")) for f in families]
        cells += [fmt(((b1.get("by_kind") or {}).get(k) or {}).get("skill")) for k in ("planned", "counterfactual")]
        cells += [fmt(((b1.get("by_block") or {}).get(k) or {}).get("skill")) for k in ("noul", "score")]
        lines.append(f"| {m['label']} | " + " | ".join(cells) + " |")
    lines += ["", "## Ours vs the untrained base (paired, same questions)", "",
              "| model | T8a Δacc (single) | T8a Δacc (order-avg) | T8b Δskill (single) | T8b Δskill (order-avg) | "
              + " | ".join(f"Δskill {f}" for f in families) + " |", "|---|---|---|---|---|" + "---|" * len(families)]
    for m in results["models"]:
        if m["role"] != "ours":
            continue
        pa = m.get("T8a_vs_base") or {}
        pb = m.get("T8b_vs_base") or {}

        def acc(reading: str) -> str:
            cell = (pa.get(reading) or {}).get("accuracy") or {}
            return f"{fmt(cell.get('diff'), signed=True)}{fmt_ci((cell.get('low'), cell.get('high')), signed=True)}" if cell else "—"

        def sk(cell: dict[str, Any] | None) -> str:
            return f"{fmt(cell.get('value'), signed=True)}{fmt_ci((cell.get('low'), cell.get('high')), signed=True)}" if cell else "—"

        fam = m.get("T8b_vs_base_by_family") or {}
        lines.append(f"| {m['label']} | {acc('single_order')} | {acc('order_averaged')} | {sk(pb.get('single_order'))} | {sk(pb.get('order_averaged'))} | "
                     + " | ".join(sk(fam.get(f)) for f in families) + " |")
    spend = results.get("openrouter")
    if spend:
        lines += ["", f"OpenRouter spend (T8 ledger): ${spend['totals'].get('spent_usd', 0):.4f} over {spend['totals'].get('calls')} calls "
                  f"(uncertain ${spend['totals'].get('uncertain_usd', 0):.4f})."]
    return "\n".join(lines) + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    piqa = sub.add_parser("piqa", help="T8a: PIQA validation → decision requests")
    piqa.add_argument("--source", type=Path, default=REPO / "artifacts" / "scratch" / "t8" / "piqa", help="directory with valid.jsonl + valid-labels.lst")
    piqa.add_argument("--download", action="store_true", help="fetch missing source files from the PIQA homepage")
    piqa.add_argument("--out", type=Path, default=DEFAULT_OUT)
    piqa.set_defaults(func=cmd_piqa)
    moments = sub.add_parser("moments", help="T8b: robot moments labelled by MuJoCo (CPU — run in a capped unit)")
    moments.add_argument("--k", type=int, default=64, help="hidden draws (rollouts) per question variant")
    moments.add_argument("--seed", type=int, default=20261012)
    moments.add_argument("--scale", type=float, default=1.0, help="multiply the per-family targets (smoke runs)")
    moments.add_argument("--workers", type=int, default=8)
    moments.add_argument("--out", type=Path, default=DEFAULT_OUT)
    moments.set_defaults(func=cmd_moments)
    check = sub.add_parser("sanity", help="T8b physics sanity report: monotonicity, analytic thresholds, determinism")
    check.add_argument("--k", type=int, default=16)
    check.add_argument("--moments", type=int, default=12, help="candidate moments per family for the monotonicity sweeps")
    check.add_argument("--seed", type=int, default=20261012)
    check.add_argument("--workers", type=int, default=8)
    check.add_argument("--out", type=Path, default=DEFAULT_OUT)
    check.set_defaults(func=cmd_sanity)
    score = sub.add_parser("score", help="scoring tables for our models and the OpenRouter models (CPU)")
    score.add_argument("--data", type=Path, default=DEFAULT_OUT, help="directory with T8a.jsonl and T8b.jsonl")
    score.add_argument("--reports", type=Path, default=REPO / "artifacts" / "reports" / "t8", help="our models' <set>-<tag>.predictions.jsonl")
    score.add_argument("--api", type=Path, default=REPO / "artifacts" / "bench" / "t8" / "api", help="bench.py predictions per model")
    score.add_argument("--config", type=Path, default=REPO / "configs" / "bench" / "t8.yaml")
    score.add_argument("--bench-results", type=Path, default=REPO / "artifacts" / "bench" / "v0" / "results.json")
    score.add_argument("--ledger", type=Path, default=REPO / "artifacts" / "bench" / "t8" / "openrouter-ledger.json")
    score.add_argument("--n-boot", type=int, default=2000)
    score.add_argument("--out", type=Path, default=DEFAULT_OUT / "results.json")
    score.set_defaults(func=cmd_score)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
