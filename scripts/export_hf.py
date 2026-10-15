#!/usr/bin/env python
"""Tasks hf0 and rel07 — export a robo-jev release (now v0.7, :data:`robo_jev.hf.export.TAG`) to Hugging Face folders, check them against
the original scoring path, upload them privately, and verify the uploaded copies from a clean cache.

    # hero chart PNGs (cairosvg through uv — not a project dependency)
    uv run python scripts/export_hf.py figures --out artifacts/scratch/rel07/figures-card
    # adapter + merged folders (CPU unit: loads the base in float32 for the merge); the data manifests default to v0.7's (--manifest)
    uv run python scripts/export_hf.py export --checkpoint <run>/checkpoint.pt --out artifacts/hf/v0.7 --base-files <dir> --figures artifacts/scratch/rel07/figures-card
    # equivalence (GPU unit): (i) the original path writes reference predictions, then (ii) adapter and (iii) merged are scored and compared;
    # the sample is hf0's (50 T1, 25 T2, 25 G2) plus --many-options records with 9–26 options from the v0.7 training input (read in code only)
    uv run python scripts/export_hf.py sample --out artifacts/scratch/rel07/equiv-140.jsonl
    uv run python scripts/diagnose_decisions.py --model Qwen/Qwen3.5-2B --checkpoint <ckpt> --input <equiv-140.jsonl> --out <equiv-i.json>
    uv run python scripts/export_hf.py equivalence --input <equiv-140.jsonl> --reference <equiv-i.predictions.jsonl> \\
        --model adapter=artifacts/hf/v0.7/robo-jev-2b-lora --model merged=artifacts/hf/v0.7/robo-jev-2b --example <request.json> --out <equiv.json>
    # model cards with the equivalence run's real example answers
    uv run python scripts/export_hf.py cards --root artifacts/hf/v0.7 --equivalence <equiv.json>
    # upload (private repos only) + the release tag
    uv run python scripts/export_hf.py upload --root artifacts/hf/v0.7 --out <upload.json>
    # clean-cache download + RoboJev.decide() and the standalone script vs the local folders, incl. the 21- and 26-option requests (GPU unit)
    uv run python scripts/export_hf.py verify-hub --root artifacts/hf/v0.7 --input tests/fixtures/decisions/requests.jsonl --limit 5 --out <verify.json>

GPU commands put the memory fence (``--gpu-memory-fraction``, default 0.6) up before the first CUDA allocation and check the free memory
(:func:`robo_jev.gpu.require_free`) before every model load. The Hugging Face token is read from ``~/.config/robo-jev/huggingface.env``
only and is never printed. Every exported text file is scanned for local paths and secrets (:func:`robo_jev.hf.export.check_public_text`)
by ``export``, ``cards`` and ``upload``.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import random
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from robo_jev.gpu import DEFAULT_FRACTION, limit_gpu_memory, memory_report, require_free  # noqa: E402

SCRIPT_VERSION = "rel07-export-1"
FIGURES = ("bench-v0-physics-light", "bench-v0-physics-dark")
MERGED_DIR, ADAPTER_DIR = "robo-jev-2b", "robo-jev-2b-lora"
#: Free unified memory to require before loading one 2B model (the scoring tools' value for 2B).
MIN_FREE = 20 * 2**30
EQUIVALENCE_SETS = (("T1", "artifacts/simgen/v0b/tests/T1.jsonl", 50), ("T2", "artifacts/simgen/v0b/tests/T2.jsonl", 25),
                    ("G2", "artifacts/simgen/probes/G2.jsonl", 25))
#: The many-option part of the equivalence sample (Task rel07): records of the v0.7 training input that carry the many-option opt-in — not
#: sealed; read in code only (nothing of their text is printed).
MANY_OPTIONS_SOURCE = "artifacts/simgen/v07/train-v07.jsonl"
#: The source kinds the many-option sample cycles through (by record id: the engine's ``v0b-…``, the MuJoCo authors', the general MC items).
MANY_OPTIONS_KINDS = ("engine", "mujoco", "general")
#: The handwritten 21- and 26-option requests (tests/fixtures): the bundle check in ``equivalence``, the card, and ``verify-hub``.
MANY_OPTIONS_REQUESTS = "tests/fixtures/decisions/many-options.jsonl"
#: The data manifests ``robo_jev.json`` records for v0.7 (label → path; absolute local paths inside are made repo-relative).
DATA_MANIFESTS = (
    ("training_input", "artifacts/simgen/v07/train-v07.manifest.json"),
    ("mix", "artifacts/simgen/v06/train-v06.manifest.json"),
    ("mujoco", "artifacts/simgen/v06/mujoco-v06.manifest.json"),
    ("engine", "artifacts/simgen/v04/engine-s1-pick.manifest.json"),
    ("simulated", "artifacts/simgen/v02/train-50k.manifest.json"),
    ("general", "artifacts/simgen/v02/general-train.manifest.json"),
    ("general_reading", "artifacts/simgen/probes/G1-train-mix.manifest.json"),
)


def _log(message: str) -> None:
    print(f"[export_hf {dt.datetime.now().strftime('%H:%M:%S')}] {message}", file=sys.stderr, flush=True)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=1, default=str) + "\n", encoding="utf-8")


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


# --------------------------------------------------------------------------
# figures, export
# --------------------------------------------------------------------------


def _figures(args: argparse.Namespace) -> int:
    args.out.mkdir(parents=True, exist_ok=True)
    code = (
        "import cairosvg, sys\n"
        "for src, dst in zip(sys.argv[1::2], sys.argv[2::2]):\n"
        "    cairosvg.svg2png(url=src, write_to=dst, scale=2.0)\n"
    )
    pairs = []
    for name in FIGURES:
        pairs += [str(REPO / "docs" / "figures" / f"{name}.svg"), str(args.out / f"{name}.png")]
    subprocess.run(["uv", "run", "--no-project", "--with", "cairosvg", "python", "-c", code, *pairs], check=True, cwd=REPO)
    _log(f"wrote {[Path(p).name for p in pairs[1::2]]} → {args.out}")
    return 0


def _export(args: argparse.Namespace) -> int:
    import torch
    from transformers import AutoModelForCausalLM

    from robo_jev.hf.export import (
        BASE_ID, BASE_REVISION, TAG, build_metadata, build_peft_model, check_no_pickle, check_public_text, load_adapter_state, read_training_checkpoint,
        render_card, save_adapter, save_merged, write_common_files,
    )
    from robo_jev.infer import file_sha256
    from robo_jev.model.backbone import describe_backbone
    from robo_jev.model.tokenizer import available_tokenizer, describe_tokenizer, load_tokenizer

    started = time.perf_counter()
    state = read_training_checkpoint(args.checkpoint)
    saved = state["model"]
    _log(f"checkpoint {args.checkpoint}: run {state.get('run_id')} step {state.get('step')}, {len(saved)} LoRA tensors")
    checkpoint_sha = file_sha256(args.checkpoint)
    tokenizer_id, tokenizer_file = available_tokenizer()
    tokenizer_info = describe_tokenizer(tokenizer_id)
    training_sha = ((state["manifest"].get("tokenizer") or {}).get("sha256"))
    if tokenizer_info["sha256"] != training_sha:
        raise SystemExit(f"local tokenizer {tokenizer_info['sha256'][:12]} is not the training tokenizer {str(training_sha)[:12]}")
    tokenizer = load_tokenizer(tokenizer_id)
    base = describe_backbone(BASE_ID, full=True)
    if base.get("revision") != BASE_REVISION:
        raise SystemExit(f"local base revision {base.get('revision')} is not the pinned {BASE_REVISION}")
    if base.get("digest") != (state["manifest"]["model"]["base"] or {}).get("digest"):
        raise SystemExit("local base digest differs from the training base")
    _log(f"base {BASE_ID}@{BASE_REVISION[:7]} verified ({base['verified']}) at {base['path']}")
    data_manifests = {}
    for spec in args.manifest or [f"{label}={path}" for label, path in DATA_MANIFESTS]:
        label, path = spec.split("=", 1)
        path = Path(path) if Path(path).is_absolute() else REPO / path
        if not path.is_file():
            raise SystemExit(f"data manifest {label}: {path} is missing")
        data_manifests[label] = json.loads(path.read_text(encoding="utf-8"))
    _log(f"{TAG}: data manifests {sorted(data_manifests)}")
    if args.data_stats is not None and Path(args.data_stats).is_file():
        stats = json.loads(Path(args.data_stats).read_text(encoding="utf-8"))
        data_manifests["simulated_shares"] = {k: {kk: vv["share"] for kk, vv in stats["shares"][k].items()} for k in ("domain", "phenomena", "style", "body")}
        data_manifests["simulated_shares"]["korean"] = stats.get("korean_share")
    files = {f"figures/{name}.png": args.figures / f"{name}.png" for name in FIGURES} if args.figures else {}
    out = Path(args.out)
    if out.exists():
        shutil.rmtree(out)
    adapter_dir, merged_dir = out / ADAPTER_DIR, out / MERGED_DIR
    adapter_dir.mkdir(parents=True)
    merged_dir.mkdir(parents=True)

    # base in float32 (exact copy of the bf16 weights) → LoRA (float32) → checkpoint values → adapter (bf16 file) → merge (fp32) → bf16
    model = AutoModelForCausalLM.from_pretrained(base["path"], dtype=torch.float32)
    _log(f"loaded the base in float32 ({type(model).__name__}) in {time.perf_counter() - started:.0f} s")
    peft_model = build_peft_model(model, state["manifest"]["model"]["lora"])
    names = load_adapter_state(peft_model, saved)
    adapter = save_adapter(peft_model, saved, adapter_dir)
    _log(f"adapter: {len(names)} tensors mapped, saved {adapter}")
    merged = save_merged(peft_model, merged_dir)
    _log(f"merged: {merged}")
    common = {"tokenizer_json": tokenizer_file, "tokenizer_config": args.base_files / "tokenizer_config.json",
              "chat_template": args.base_files / "chat_template.jinja", "figures": files}
    listing = {}
    for variant, folder, weights in (("adapter", adapter_dir, {"kind": "peft-lora", **adapter}), ("merged", merged_dir, {"kind": "merged", **merged})):
        card = render_card(variant)  # the example's real output goes in with `cards --equivalence` after the GPU check
        metadata = build_metadata(state, variant=variant, checkpoint_path=args.checkpoint, checkpoint_sha256=checkpoint_sha, tokenizer=tokenizer,
                                  tokenizer_sha256=tokenizer_info["sha256"], repo=REPO, weights=weights, data_manifests=data_manifests)
        write_common_files(folder, metadata=metadata, card=card, **common)
        listing[variant] = {"folder": str(folder), "files": {f: (folder / f).stat().st_size for f in check_no_pickle(folder)},
                            "public_text_scanned": check_public_text(folder)}
    _write_json(out / "export.json", {"script_version": SCRIPT_VERSION, "tag": TAG, "checkpoint": str(args.checkpoint), "checkpoint_sha256": checkpoint_sha,
                                      "adapter": adapter, "merged": merged, "listing": listing, "seconds": round(time.perf_counter() - started, 1)})
    _log(f"export done in {time.perf_counter() - started:.0f} s → {out}")
    return 0


# --------------------------------------------------------------------------
# equivalence
# --------------------------------------------------------------------------


def example_markdown(answers: dict[str, Any], *, variant: str) -> str:
    """The model card's example output: the real answers of this repo's weights (``equivalence --example``), rounded for reading."""

    def short(value: Any) -> Any:
        if isinstance(value, float):
            return round(value, 3)
        if isinstance(value, dict):
            return {k: short(v) for k, v in value.items()}
        return value

    lines = [json.dumps({name: short(answer)}, ensure_ascii=False)[1:-1] for name, answer in answers.items()]
    weights = "the merged weights" if variant == "merged" else "Qwen/Qwen3.5-2B + this adapter"
    return (f"The answers for this example ({weights}, bfloat16, one GPU, `orders=\"single\"`; rounded to three decimals here):\n\n"
            "```json\n{\n " + ",\n ".join(lines) + "\n}\n```")


def _cards(args: argparse.Namespace) -> int:
    from robo_jev.hf.export import check_no_pickle, check_public_text, render_card

    examples = json.loads(args.equivalence.read_text(encoding="utf-8")).get("examples", {}) if args.equivalence is not None else {}
    for variant, folder in (("adapter", args.root / ADAPTER_DIR), ("merged", args.root / MERGED_DIR)):
        example = example_markdown(examples[variant]["single"], variant=variant) if variant in examples else ""
        (folder / "README.md").write_text(render_card(variant, example_output=example), encoding="utf-8")
        check_no_pickle(folder)
        check_public_text(folder)
        _log(f"{folder}/README.md written ({'with' if example else 'without'} the example output); public-text scan clean")
    return 0


def many_options_kind(record_id: str) -> str:
    """The source of a v0.7 training record by its id: the engine (``v0b-…``), a MuJoCo author (``v03b…``, ``v06a…``) or a general item."""
    rid = str(record_id)
    if rid.startswith("v0b-"):
        return "engine"
    if rid.startswith(("v03b", "v06a")):
        return "mujoco"
    return "general"


def pick_many_options(rows: list[tuple[int, str, int]], count: int, seed: int) -> list[int]:
    """Choose `count` of the many-option candidates ``(line, kind, N)`` so that N (the record's largest choice) covers 9–26 evenly and the
    source kind rotates: the i-th pick asks for N = 9 + i mod 18 and kind ``MANY_OPTIONS_KINDS[(i + i // 18) mod 3]`` (any kind when that
    cell is empty), a seeded random candidate of the cell, never twice the same line. Returns the chosen lines in file order."""
    cells: dict[tuple[int, str], list[int]] = {}
    for line, kind, n in rows:
        cells.setdefault((n, kind), []).append(line)
    chosen: list[int] = []
    for i in range(count):
        n = 9 + i % 18
        wanted = MANY_OPTIONS_KINDS[(i + i // 18) % len(MANY_OPTIONS_KINDS)]
        for kind in (wanted, *[k for k in MANY_OPTIONS_KINDS if k != wanted]):
            free = sorted(set(cells.get((n, kind), [])) - set(chosen))
            if free:
                chosen.append(random.Random(f"rel07-many-options|{n}|{kind}|{seed}|{i}").choice(free))
                break
        else:
            raise SystemExit(f"no many-option record with {n} options left")
    return sorted(chosen)


def _sample(args: argparse.Namespace) -> int:
    """The equivalence records: hf0's seeded random sample of 50 T1, 25 T2 and 25 G2 records (file order kept within each set), then
    `--many-options` records of the v0.7 training input with 9–26 options (:func:`pick_many_options`). The training records are read in
    code only — only ids, kinds and option counts are logged. A side file ``<out>.sets.json`` names each record's set."""
    from robo_jev.decisions.records import OPTION_RANGE_MARK

    out, sets = [], {}
    for label, path, count in EQUIVALENCE_SETS:
        records = _read_jsonl(REPO / path)
        picked = sorted(random.Random(f"hf0-equivalence|{label}|{args.seed}").sample(range(len(records)), count))
        out += [records[i] for i in picked]
        sets.update({str(records[i]["id"]): {"set": label} for i in picked})
        _log(f"{label}: {count} of {len(records)} records")
    if args.many_options:
        source = args.many_source if args.many_source.is_absolute() else REPO / args.many_source
        rows = []
        with source.open(encoding="utf-8") as handle:  # pass 1: candidates (line, kind, N) — nothing of the text is kept
            for line_no, line in enumerate(handle):
                record = json.loads(line)
                if (record.get("meta") or {}).get("option_range") != OPTION_RANGE_MARK or "~r" in str(record["id"]):
                    continue
                n = max(len(q["criteria"]) for q in record["request"]["questions"].values() if q.get("type") == "choice")
                rows.append((line_no, many_options_kind(record["id"]), n))
        chosen = set(pick_many_options(rows, args.many_options, args.seed))
        picked = []
        with source.open(encoding="utf-8") as handle:  # pass 2: the chosen lines
            for line_no, line in enumerate(handle):
                if line_no in chosen:
                    picked.append(json.loads(line))
        for record in picked:
            kind = many_options_kind(record["id"])
            n = max(len(q["criteria"]) for q in record["request"]["questions"].values() if q.get("type") == "choice")
            sets[str(record["id"])] = {"set": "many", "kind": kind, "options": n,
                                       "shape": ((record.get("meta") or {}).get("shape") or {}).get("kind") or "plain"}
        out += picked
        counts = {kind: sum(1 for r in picked if many_options_kind(r["id"]) == kind) for kind in MANY_OPTIONS_KINDS}
        _log(f"many options: {len(picked)} of {len(rows)} candidates from {args.many_source} · kinds {counts} · "
             f"N {sorted(sets[str(r['id'])]['options'] for r in picked)}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in out), encoding="utf-8")
    _write_json(sets_path(args.out), sets)
    _log(f"{len(out)} records → {args.out} (sets → {sets_path(args.out)})")
    return 0


def sets_path(sample: Path) -> Path:
    return sample.with_name(sample.stem + ".sets.json")


def _set_of(record_id: str, sets: dict[str, Any] | None = None) -> str:
    """The record's set from the sample's side file; without one: T1/T2 ids are ``v0b-T1-…``/``v0b-T2-…`` and everything else is G2."""
    rid = str(record_id)
    if sets and rid in sets:
        return str(sets[rid]["set"])
    return rid.split("-")[1] if rid.startswith("v0b-T") else "G2"


def compare_to_reference(reference: list[dict[str, Any]], scored: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Listing-level code probabilities and averaged answers of one path vs the reference predictions (same records, listings, order)."""
    import math

    rows: dict[str, dict[str, list[float]]] = {}
    argmax_flips: dict[str, int] = {}
    listings: dict[str, int] = {}
    decisions: dict[str, list[int]] = {}  # per type: [questions, decisions changed] — choice argmax; yes/no and score: the side of 0.5
    exact = True
    for prediction in reference:
        ours = scored[str(prediction["id"])]
        for name, question in prediction["questions"].items():
            kind = question["type"]
            mine = ours[name]
            if len(mine["listings"]) != len(question["listings"]):
                raise ValueError(f"{prediction['id']}/{name}: {len(mine['listings'])} listings vs reference {len(question['listings'])}")
            block = rows.setdefault(kind, {"listing": [], "answer": []})
            for a, b in zip(mine["listings"], question["listings"]):
                if a["options"] != b["options"]:
                    raise ValueError(f"{prediction['id']}/{name}: shown options differ {a['options']} vs {b['options']}")
                block["listing"] += [abs(x - y) for x, y in zip(a["code_probs"], b["code_probs"])]
                exact = exact and list(a["code_probs"]) == list(b["code_probs"])
                listings[kind] = listings.get(kind, 0) + 1
                argmax_flips[kind] = argmax_flips.get(kind, 0) + int(
                    max(range(len(a["code_probs"])), key=a["code_probs"].__getitem__) != max(range(len(b["code_probs"])), key=b["code_probs"].__getitem__)
                )
            ref, got = question["answer"], mine["answer"]
            if kind == "choice":
                block["answer"] += [abs(got["probabilities"][k] - ref["probabilities"][k]) for k in ref["probabilities"]]
                changed = max(ref["probabilities"], key=ref["probabilities"].get) != max(got["probabilities"], key=got["probabilities"].get)
            elif kind == "noul":
                block["answer"].append(abs(got["p_yes"] - ref["p_yes"]))
                changed = (got["p_yes"] >= 0.5) != (ref["p_yes"] >= 0.5)
            else:
                block["answer"].append(abs(got["value"] - ref["value"]))
                block["answer"] += [abs(got["bins"][k] - ref["bins"][k]) for k in ref["bins"]]
                changed = (got["value"] >= 0.5) != (ref["value"] >= 0.5)
            tally = decisions.setdefault(kind, [0, 0])
            tally[0] += 1
            tally[1] += int(changed)
    summary = {}
    for kind, block in sorted(rows.items()):
        summary[kind] = {
            "listings": listings[kind], "probabilities": len(block["listing"]),
            "listing_max": max(block["listing"]), "listing_mean": math.fsum(block["listing"]) / len(block["listing"]),
            "answer_max": max(block["answer"]), "answer_mean": math.fsum(block["answer"]) / len(block["answer"]),
            "argmax_flips": argmax_flips[kind], "questions": decisions[kind][0], "decisions_changed": decisions[kind][1],
        }
    every = [v for block in rows.values() for v in block["listing"]]
    summary["all"] = {"listing_max": max(every), "listing_mean": math.fsum(every) / len(every), "probabilities": len(every),
                      "argmax_flips": sum(argmax_flips.values()), "listings": sum(listings.values()),
                      "questions": sum(t[0] for t in decisions.values()), "decisions_changed": sum(t[1] for t in decisions.values())}
    return {"by_type": summary, "byte_identical": exact}


def _equivalence(args: argparse.Namespace) -> int:
    guard = limit_gpu_memory(args.gpu_memory_fraction)
    _log(f"gpu guard {guard} · memory {memory_report()}")
    import torch

    from robo_jev.decisions.records import read_records, shuffle_key
    from robo_jev.infer import RoboJev

    records = {str(r["id"]): r for r in read_records(args.input)}
    reference = _read_jsonl(args.reference)
    missing = [p["id"] for p in reference if str(p["id"]) not in records]
    if missing or len(reference) != len(records):
        raise SystemExit(f"reference predictions and input records differ ({len(reference)} vs {len(records)}; missing {missing[:3]})")
    sets_file = args.sets or sets_path(args.input)
    sets = json.loads(sets_file.read_text(encoding="utf-8")) if sets_file.is_file() else None
    bundle_records = read_records(REPO / args.bundle_requests) if args.bundle_requests else []
    report: dict[str, Any] = {"input": str(args.input), "reference": str(args.reference), "records": len(records), "sets": str(sets_file) if sets else None,
                              "paths": {}, "by_set": {}, "script_version": SCRIPT_VERSION,
                              "created": dt.datetime.now().astimezone().isoformat(timespec="seconds")}
    for spec in args.model:
        label, folder = spec.split("=", 1)
        dropped = drop_page_cache(folder, args.base_model)
        _log(f"{label}: dropped the page cache of {dropped / 2**30:.1f} GiB of model files · memory {memory_report()}")
        require_free(MIN_FREE, what=f"load {label}")
        started = time.perf_counter()
        model = RoboJev.from_pretrained(folder, device="cuda", base_model=args.base_model)
        load_seconds = time.perf_counter() - started
        _log(f"{label}: loaded {model.name} in {load_seconds:.0f} s · memory {memory_report()}")
        started = time.perf_counter()
        scored = {}
        for prediction in reference:
            record = records[str(prediction["id"])]
            request = record["request"]
            scored[str(record["id"])] = model.score_request(request["state"], request["questions"], orders="balanced", key=shuffle_key(record),
                                                            seed=args.shuffle_seed)
        seconds = time.perf_counter() - started
        result = compare_to_reference(reference, scored)
        result.update({"model": model.name, "folder": folder, "load_seconds": round(load_seconds, 1), "seconds": round(seconds, 1)})
        report["paths"][label] = result
        names = sorted({_set_of(p["id"], sets) for p in reference})
        report["by_set"][label] = {
            name: compare_to_reference([p for p in reference if _set_of(p["id"], sets) == name], scored)["by_type"]["all"] for name in names
        }
        _log(f"{label}: {result['by_type']['all']} byte-identical {result['byte_identical']} ({seconds:.0f} s) · by set "
             + "; ".join(f"{name} max {v['listing_max']:.4g} flips {v['argmax_flips']}/{v['listings']}" for name, v in report["by_set"][label].items()))
        if args.save_predictions:
            _write_json(args.out.with_suffix(f".{label}.scored.json"), scored)
        if args.example is not None:  # the model card's example, called as a user would (default key and seed)
            example = json.loads(args.example.read_text(encoding="utf-8"))
            report.setdefault("examples", {})[label] = {
                orders: model.decide(example["state"], example["questions"], orders=orders) for orders in ("single", "balanced")
            }
        package = {str(r["id"]): model.decide(r["request"]["state"], r["request"]["questions"]) for r in bundle_records}  # as a user calls it
        del model
        torch.cuda.empty_cache()
        if bundle_records:  # the shipped runner of this folder, loaded and called as a user would, on the 21- and 26-option requests
            report.setdefault("bundle", {})[label] = run_bundle(label, Path(folder), bundle_records, package, base_model=args.base_model)
    report["gpu"] = {"guard": guard, "memory_at_end": memory_report(), "peak_allocated_bytes": int(torch.cuda.max_memory_allocated())}
    _write_json(args.out, report)
    _log(f"→ {args.out}")
    return 0


def _load_standalone(path: Path) -> Any:
    """Import a ``robo_jev_decide.py`` file as its own module (the copy shipped in an export folder or downloaded from the Hub), without
    writing bytecode next to it."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(f"robo_jev_decide_{abs(hash(str(path)))}", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    previous, sys.dont_write_bytecode = sys.dont_write_bytecode, True  # no __pycache__ next to the shipped file (it would be uploaded)
    try:
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = previous
    return module


def run_bundle(label: str, folder: Path, records: list[dict[str, Any]], package: dict[str, Any], *, base_model: str | None) -> dict[str, Any]:
    """Answer `records` with the folder's own ``robo_jev_decide.py`` (``load()`` + ``decide()``, default key and seed, single order) and
    compare every answer with the package's (``RoboJev.decide`` on the same folder): JSON equality of every float."""
    import torch

    from robo_jev.infer import file_sha256

    script = folder / "robo_jev_decide.py"
    runner_module = _load_standalone(script)
    dropped = drop_page_cache(folder, base_model)
    _log(f"bundle {label}: dropped the page cache of {dropped / 2**30:.1f} GiB · memory {memory_report()}")
    require_free(MIN_FREE, what=f"load bundle {label}")
    started = time.perf_counter()
    runner = runner_module.load(str(folder), device="cuda", base_model=base_model if (folder / "adapter_config.json").is_file() else None)
    out: dict[str, Any] = {"script": str(script), "script_sha256": file_sha256(script), "limits": runner_module.LIMITS, "records": {}}
    for record in records:
        request = record["request"]
        answers = runner.decide(request["state"], request["questions"])
        rid = str(record["id"])
        options = {name: len(q["criteria"]) for name, q in request["questions"].items() if q.get("type") == "choice"}
        out["records"][rid] = {"options": options, "bundle": answers, "package": package.get(rid),
                               "identical": _answers_equal(answers, package[rid]) if rid in package else None}
        _log(f"bundle {label} · {rid} ({options}): " + "; ".join(
            f"{name}={a.get('choice', a.get('noul', a.get('score')))}" for name, a in answers.items()) + f" · identical to RoboJev {out['records'][rid]['identical']}")
    out["seconds"] = round(time.perf_counter() - started, 1)
    out["all_identical"] = all(entry["identical"] for entry in out["records"].values())
    del runner
    torch.cuda.empty_cache()
    return out


# --------------------------------------------------------------------------
# upload, verify
# --------------------------------------------------------------------------


def _upload(args: argparse.Namespace) -> int:
    from huggingface_hub import HfApi

    from robo_jev.hf.export import ADAPTER_REPO, MERGED_REPO, TAG, check_no_pickle, check_public_text
    from robo_jev.hf.hub import ensure_private_repo, load_token, redact, remote_listing, tag_release, upload

    for folder in (args.root / ADAPTER_DIR, args.root / MERGED_DIR):  # before any network call
        check_no_pickle(folder)
        check_public_text(folder)
        meta = json.loads((folder / "robo_jev.json").read_text(encoding="utf-8"))
        if meta.get("version") != TAG:
            raise SystemExit(f"{folder}: robo_jev.json says {meta.get('version')}, this upload tags {TAG}")
    api = HfApi(token=load_token(args.token_file))
    try:
        me = api.whoami()
        orgs = [org.get("name") for org in me.get("orgs", [])]
        _log(f"authenticated as {me.get('name')} (orgs {orgs})")
        result: dict[str, Any] = {"user": me.get("name"), "orgs": orgs, "tag": TAG, "repos": {}}
        for repo_id, folder in ((ADAPTER_REPO, args.root / ADAPTER_DIR), (MERGED_REPO, args.root / MERGED_DIR)):
            check_no_pickle(folder)
            result["repos"][repo_id] = {"repo": ensure_private_repo(api, repo_id)}
            _log(f"{repo_id}: private repo ready ({result['repos'][repo_id]['repo']})")
            message = args.message or (f"robo-jev {TAG} (training seed 0): weights, tokenizer, robo_jev.json, model card, standalone runner "
                                       "(choice questions with 2-26 options)")
            result["repos"][repo_id]["upload"] = upload(api, repo_id, folder, message=message)
            _log(f"{repo_id}: uploaded {folder}")
            result["repos"][repo_id]["tag"] = tag_release(api, repo_id, TAG, message=f"robo-jev {TAG}")
            result["repos"][repo_id]["remote"] = remote_listing(api, repo_id)
            if result["repos"][repo_id]["remote"]["private"] is not True:
                raise SystemExit(f"{repo_id} is not private after upload")
            _log(f"{repo_id}: tag {result['repos'][repo_id]['tag']} · {len(result['repos'][repo_id]['remote']['files'])} files, private")
        leftovers = [m.id for m in api.list_models(author=me.get("name")) if "robo-jev" in m.id]
        result["user_namespace_robo_jev_repos"] = leftovers
        _log(f"robo-jev repos under {me.get('name')}: {leftovers or 'none'}")
    except Exception as error:  # noqa: BLE001 — scrub anything token-like before it reaches the log
        raise SystemExit(f"upload failed: {redact(error)}") from None
    _write_json(args.out, result)
    return 0


def drop_page_cache(*roots: str | Path) -> int:
    """Evict the page cache of the large files under `roots` (fsync, then ``posix_fadvise(DONTNEED)``); returns the bytes covered.

    On the GB10's unified memory the free figure that :func:`robo_jev.gpu.require_free` reads tracks MemFree, and a few GB of fresh
    downloads or model reads sit in the page cache — clean, reclaimable pages that still push that figure under the threshold. Dropping our
    own files' pages restores the honest figure instead of lowering the threshold."""
    covered = 0
    for root in roots:
        root = Path(root)
        files = [root] if root.is_file() else [p for p in root.rglob("*") if p.is_file()] if root.is_dir() else []
        for path in files:
            if path.is_symlink() or path.stat().st_size < (1 << 20):
                continue
            fd = os.open(path, os.O_RDONLY)
            try:
                os.fsync(fd)
                os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
                covered += path.stat().st_size
            finally:
                os.close(fd)
    return covered


def _answers_equal(a: dict[str, Any], b: dict[str, Any]) -> bool:
    return json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)


def _verify_hub(args: argparse.Namespace) -> int:
    guard = limit_gpu_memory(args.gpu_memory_fraction)
    _log(f"gpu guard {guard} · memory {memory_report()}")
    import torch

    from robo_jev.decisions.records import read_records, shuffle_key
    from robo_jev.hf.export import ADAPTER_REPO, MERGED_REPO, TAG
    from robo_jev.hf.hub import load_token, redact
    from robo_jev.infer import RoboJev, file_sha256

    records = read_records(args.input)[: args.limit]
    many = read_records(REPO / args.many_options) if args.many_options else []  # the 21- and 26-option requests, always included
    records += [r for r in many if str(r["id"]) not in {str(x["id"]) for x in records}]
    args.cache_parent.mkdir(parents=True, exist_ok=True)
    cache = Path(tempfile.mkdtemp(prefix=f"{TAG}-clean-cache-", dir=args.cache_parent))
    token = load_token(args.token_file)
    report: dict[str, Any] = {"records": [r["id"] for r in records], "many_option_records": [r["id"] for r in many], "cache": str(cache), "tag": TAG,
                              "runs": {}}

    def ready(what: str) -> None:
        dropped = drop_page_cache(cache, args.root, args.base_model)
        _log(f"{what}: dropped the page cache of {dropped / 2**30:.1f} GiB of model files · memory {memory_report()}")
        require_free(MIN_FREE, what=what)

    def run_robojev(label: str, source: str, **kwargs: Any) -> dict[str, Any]:
        ready(f"load {label}")
        started = time.perf_counter()
        model = RoboJev.from_pretrained(source, device="cuda", **kwargs)
        answers = {str(r["id"]): model.decide(r["request"]["state"], r["request"]["questions"], key=shuffle_key(r), orders=args.orders)
                   for r in records}
        report["runs"][label] = {"model": model.name, "seconds": round(time.perf_counter() - started, 1)}
        _log(f"{label}: {model.name} answered {len(answers)} records in {report['runs'][label]['seconds']} s")
        del model
        torch.cuda.empty_cache()
        return answers

    try:
        local = {
            "merged": run_robojev("local merged", str(args.root / MERGED_DIR)),
            "adapter": run_robojev("local adapter", str(args.root / ADAPTER_DIR), base_model=args.base_model),
        }
        hub = {
            "merged": run_robojev("hub merged", MERGED_REPO, revision=TAG, token=token, cache_dir=cache),
            "adapter": run_robojev("hub adapter", ADAPTER_REPO, revision=TAG, token=token, cache_dir=cache),
        }
        requests = cache / "requests.jsonl"
        requests.write_text("".join(json.dumps({"id": r["id"], "key": shuffle_key(r), "request": r["request"]}, ensure_ascii=False) + "\n"
                                    for r in records), encoding="utf-8")
        standalone = {}
        for variant, repo_id in (("merged", MERGED_REPO), ("adapter", ADAPTER_REPO)):
            ready(f"standalone {variant}")
            script = Path(hub_snapshot(repo_id, cache, token)) / "robo_jev_decide.py"
            # the standalone script, as a user would run it — in a child process with the same fence, the clean cache and the token in its env
            launcher = (f"import runpy, sys, torch; torch.cuda.set_per_process_memory_fraction({args.gpu_memory_fraction}); "
                        f"sys.argv = [{str(script)!r}] + sys.argv[1:]; runpy.run_path({str(script)!r}, run_name='__main__')")
            env = {**os.environ, "HF_HOME": str(cache / "home"), "HF_HUB_CACHE": str(cache), "HF_TOKEN": token, "PYTHONPATH": ""}
            started = time.perf_counter()
            done = subprocess.run([sys.executable, "-c", launcher, "--model", repo_id, "--revision", TAG, "--requests", str(requests), "--key-field", "key",
                                   "--orders", args.orders], capture_output=True, text=True, env=env, cwd=str(cache))
            if done.returncode != 0:
                raise RuntimeError(f"standalone {variant} failed: {redact(done.stderr[-2000:])}")
            standalone[variant] = {line["id"]: line["answers"] for line in map(json.loads, done.stdout.splitlines()) if line}
            report["runs"][f"standalone {variant}"] = {"seconds": round(time.perf_counter() - started, 1), "script": str(script),
                                                       "script_sha256": file_sha256(script)}
            _log(f"standalone {variant}: {len(standalone[variant])} records in {report['runs'][f'standalone {variant}']['seconds']} s")
    except Exception as error:  # noqa: BLE001
        raise SystemExit(f"verify failed: {redact(error)}") from None
    checks = {}
    for variant in ("merged", "adapter"):
        checks[f"hub {variant} RoboJev == local"] = all(_answers_equal(hub[variant][k], local[variant][k]) for k in local[variant])
        checks[f"hub {variant} standalone == local"] = all(_answers_equal(standalone[variant][k], local[variant][k]) for k in local[variant])
        for record in many:  # every option of the many-option choices got a probability, from the fresh download, in both runners
            rid = str(record["id"])
            for runner, answers in (("RoboJev", hub[variant]), ("standalone", standalone[variant])):
                checks[f"hub {variant} {runner} answers {rid}"] = all(
                    len(answers.get(rid, {}).get(name, {}).get("probabilities") or {}) == len(q["criteria"])
                    for name, q in record["request"]["questions"].items() if q.get("type") == "choice"
                )
    report["checks"] = checks
    report["passed"] = all(checks.values())
    report["answers"] = {"local": local, "hub": hub, "standalone": standalone}
    report["downloaded"] = sorted(str(p.relative_to(cache)) for p in cache.rglob("*") if p.is_file() and "blobs" not in p.parts)[:200]
    _write_json(args.out, report)
    _log(f"checks {checks} → {args.out}")
    if not args.keep_cache:
        shutil.rmtree(cache, ignore_errors=True)
    return 0 if report["passed"] else 1


def hub_snapshot(repo_id: str, cache: Path, token: str) -> str:
    from huggingface_hub import snapshot_download

    from robo_jev.hf.export import TAG

    return snapshot_download(repo_id, revision=TAG, token=token, cache_dir=cache)


# --------------------------------------------------------------------------


def _parser() -> argparse.ArgumentParser:
    from robo_jev.hf.hub import DEFAULT_TOKEN_FILE

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("figures", help="rasterize the hero chart SVGs to PNG")
    p.add_argument("--out", type=Path, required=True)
    p = sub.add_parser("export", help="checkpoint → adapter and merged folders")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--base-files", type=Path, required=True, help="tokenizer_config.json and chat_template.jinja of Qwen/Qwen3.5-2B at the pinned revision")
    p.add_argument("--figures", type=Path, default=None, help="folder with the hero chart PNGs (from `figures`)")
    p.add_argument("--manifest", action="append", default=None, metavar="LABEL=PATH",
                   help="a data manifest for robo_jev.json (repeatable; default: v0.7's — " + ", ".join(label for label, _ in DATA_MANIFESTS) + ")")
    p.add_argument("--data-stats", type=Path, default=REPO / "artifacts/simgen/v02/train-50k.stats.json",
                   help="the simulated part's stats (domain, phenomena, style, body shares; Korean share)")
    p = sub.add_parser("cards", help="re-render README.md (model cards) in an existing export")
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--equivalence", type=Path, default=None, help="the equivalence report with the example's answers per variant")
    p = sub.add_parser("sample", help="the equivalence records (50 T1, 25 T2, 25 G2, plus many-option training records)")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--many-options", type=int, default=40, help="many-option records (9–26 options) to add; 0 = hf0's sample only")
    p.add_argument("--many-source", type=Path, default=Path(MANY_OPTIONS_SOURCE), help="where the many-option records come from (read in code only)")
    for name in ("equivalence", "verify-hub"):
        p = sub.add_parser(name)
        p.add_argument("--gpu-memory-fraction", type=float, default=DEFAULT_FRACTION)
        p.add_argument("--base-model", default=str(REPO / "artifacts/models/Qwen/Qwen3.5-2B"),
                       help="local copy of the base for the adapter (equivalence; verify-hub's local run) — the hub run always uses the pinned Hub base")
        p.add_argument("--out", type=Path, required=True)
        p.add_argument("--input", type=Path, required=True)
    p = sub.choices["equivalence"]
    p.add_argument("--reference", type=Path, required=True, help="the original path's .predictions.jsonl (diagnose_decisions.py --checkpoint)")
    p.add_argument("--model", action="append", required=True, help="label=folder (repeatable)")
    p.add_argument("--shuffle-seed", type=int, default=1)
    p.add_argument("--save-predictions", action="store_true")
    p.add_argument("--example", type=Path, default=None, help="a {state, questions} JSON to answer with each model (the model card's example)")
    p.add_argument("--sets", type=Path, default=None, help="the sample's set file (default: <input>.sets.json when it exists)")
    p.add_argument("--bundle-requests", default=MANY_OPTIONS_REQUESTS,
                   help="requests answered through each folder's own robo_jev_decide.py and compared with RoboJev ('' to skip)")
    p = sub.choices["verify-hub"]
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--limit", type=int, default=5)
    p.add_argument("--many-options", default=MANY_OPTIONS_REQUESTS, help="requests with many options, always checked ('' to skip)")
    p.add_argument("--orders", choices=("single", "balanced"), default="single")
    p.add_argument("--token-file", type=Path, default=DEFAULT_TOKEN_FILE)
    p.add_argument("--cache-parent", type=Path, default=REPO / "artifacts/scratch/rel07")
    p.add_argument("--keep-cache", action="store_true")
    p = sub.add_parser("upload", help="private repos + upload + tag")
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--token-file", type=Path, default=DEFAULT_TOKEN_FILE)
    p.add_argument("--message", default=None, help="the Hub commit message (default: names the release tag)")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    handlers = {"figures": _figures, "export": _export, "cards": _cards, "sample": _sample, "equivalence": _equivalence, "upload": _upload, "verify-hub": _verify_hub}
    return handlers[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
