#!/usr/bin/env python
"""Task hf0 — export robo-jev v0.1 to Hugging Face folders, check them against the original scoring path, upload them privately, and
verify the uploaded copies from a clean cache.

    # hero chart PNGs (cairosvg through uv — not a project dependency)
    uv run python scripts/export_hf.py figures --out artifacts/hf/figures
    # adapter + merged folders (CPU unit: loads the base in float32 for the merge)
    uv run python scripts/export_hf.py export --checkpoint <run>/checkpoint.pt --out artifacts/hf/v0.1 --base-files <dir> --figures artifacts/hf/figures
    # equivalence (GPU unit): (i) the original path writes reference predictions, then (ii) adapter and (iii) merged are scored and compared
    uv run python scripts/export_hf.py sample --out artifacts/scratch/hf0/equiv-100.jsonl
    uv run python scripts/diagnose_decisions.py --model Qwen/Qwen3.5-2B --checkpoint <ckpt> --input <equiv-100.jsonl> --out <equiv-i.json>
    uv run python scripts/export_hf.py equivalence --input <equiv-100.jsonl> --reference <equiv-i.predictions.jsonl> \\
        --model adapter=artifacts/hf/v0.1/robo-jev-2b-lora --model merged=artifacts/hf/v0.1/robo-jev-2b --out <equiv.json>
    # upload (private repos only) + v0.1 tag
    uv run python scripts/export_hf.py upload --root artifacts/hf/v0.1 --out <upload.json>
    # clean-cache download + RoboJev.decide() and the standalone script vs the local folders (GPU unit)
    uv run python scripts/export_hf.py verify-hub --root artifacts/hf/v0.1 --input tests/fixtures/decisions/requests.jsonl --limit 5 --out <verify.json>

GPU commands put the memory fence (``--gpu-memory-fraction``, default 0.6) up before the first CUDA allocation and check the free memory
(:func:`robo_jev.gpu.require_free`) before every model load. The Hugging Face token is read from ``~/.config/robo-jev/huggingface.env``
only and is never printed.
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

SCRIPT_VERSION = "hf0-export-1"
FIGURES = ("bench-v0-physics-light", "bench-v0-physics-dark")
MERGED_DIR, ADAPTER_DIR = "robo-jev-2b", "robo-jev-2b-lora"
#: Free unified memory to require before loading one 2B model (the scoring tools' value for 2B).
MIN_FREE = 20 * 2**30
EQUIVALENCE_SETS = (("T1", "artifacts/simgen/v0b/tests/T1.jsonl", 50), ("T2", "artifacts/simgen/v0b/tests/T2.jsonl", 25),
                    ("G2", "artifacts/simgen/probes/G2.jsonl", 25))


def _log(message: str) -> None:
    print(f"[hf0 {dt.datetime.now().strftime('%H:%M:%S')}] {message}", file=sys.stderr, flush=True)


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
        BASE_ID, BASE_REVISION, build_metadata, build_peft_model, check_no_pickle, load_adapter_state, read_training_checkpoint, render_card,
        save_adapter, save_merged, write_common_files,
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
    for label, path in (("mix", args.mix_manifest), ("simulated", args.data_manifest), ("general_reading", args.probe_manifest)):
        if path is not None and Path(path).is_file():
            data_manifests[label] = json.loads(Path(path).read_text(encoding="utf-8"))
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
        listing[variant] = {"folder": str(folder), "files": {f: (folder / f).stat().st_size for f in check_no_pickle(folder)}}
    _write_json(out / "export.json", {"script_version": SCRIPT_VERSION, "checkpoint": str(args.checkpoint), "checkpoint_sha256": checkpoint_sha,
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
    from robo_jev.hf.export import check_no_pickle, render_card

    examples = json.loads(args.equivalence.read_text(encoding="utf-8")).get("examples", {}) if args.equivalence is not None else {}
    for variant, folder in (("adapter", args.root / ADAPTER_DIR), ("merged", args.root / MERGED_DIR)):
        example = example_markdown(examples[variant]["single"], variant=variant) if variant in examples else ""
        (folder / "README.md").write_text(render_card(variant, example_output=example), encoding="utf-8")
        check_no_pickle(folder)
        _log(f"{folder}/README.md written ({'with' if example else 'without'} the example output)")
    return 0


def _sample(args: argparse.Namespace) -> int:
    """The equivalence records: a seeded random sample of 50 T1, 25 T2 and 25 G2 records (file order kept within each set)."""
    out = []
    for label, path, count in EQUIVALENCE_SETS:
        records = _read_jsonl(REPO / path)
        picked = sorted(random.Random(f"hf0-equivalence|{label}|{args.seed}").sample(range(len(records)), count))
        out += [records[i] for i in picked]
        _log(f"{label}: {count} of {len(records)} records")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in out), encoding="utf-8")
    _log(f"{len(out)} records → {args.out}")
    return 0


def _set_of(record_id: str) -> str:
    """T1/T2 ids are ``v0b-T1-…``/``v0b-T2-…``; everything else in the equivalence sample is G2 (``boolq-…``, ``mnli-…``)."""
    rid = str(record_id)
    return rid.split("-")[1] if rid.startswith("v0b-T") else "G2"


def compare_to_reference(reference: list[dict[str, Any]], scored: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Listing-level code probabilities and averaged answers of one path vs the reference predictions (same records, listings, order)."""
    import math

    rows: dict[str, dict[str, list[float]]] = {}
    argmax_flips: dict[str, int] = {}
    listings: dict[str, int] = {}
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
            elif kind == "noul":
                block["answer"].append(abs(got["p_yes"] - ref["p_yes"]))
            else:
                block["answer"].append(abs(got["value"] - ref["value"]))
                block["answer"] += [abs(got["bins"][k] - ref["bins"][k]) for k in ref["bins"]]
    summary = {}
    for kind, block in sorted(rows.items()):
        summary[kind] = {
            "listings": listings[kind], "probabilities": len(block["listing"]),
            "listing_max": max(block["listing"]), "listing_mean": math.fsum(block["listing"]) / len(block["listing"]),
            "answer_max": max(block["answer"]), "answer_mean": math.fsum(block["answer"]) / len(block["answer"]),
            "argmax_flips": argmax_flips[kind],
        }
    every = [v for block in rows.values() for v in block["listing"]]
    summary["all"] = {"listing_max": max(every), "listing_mean": math.fsum(every) / len(every), "probabilities": len(every),
                      "argmax_flips": sum(argmax_flips.values()), "listings": sum(listings.values())}
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
    report: dict[str, Any] = {"input": str(args.input), "reference": str(args.reference), "records": len(records), "paths": {},
                              "by_set": {}, "script_version": SCRIPT_VERSION, "created": dt.datetime.now().astimezone().isoformat(timespec="seconds")}
    for spec in args.model:
        label, folder = spec.split("=", 1)
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
        report["by_set"][label] = {
            name: compare_to_reference([p for p in reference if _set_of(p["id"]) == name], scored)["by_type"]["all"]
            for name in sorted({_set_of(p["id"]) for p in reference})
        }
        _log(f"{label}: {result['by_type']['all']} byte-identical {result['byte_identical']} ({seconds:.0f} s)")
        if args.save_predictions:
            _write_json(args.out.with_suffix(f".{label}.scored.json"), scored)
        if args.example is not None:  # the model card's example, called as a user would (default key and seed)
            example = json.loads(args.example.read_text(encoding="utf-8"))
            report.setdefault("examples", {})[label] = {
                orders: model.decide(example["state"], example["questions"], orders=orders) for orders in ("single", "balanced")
            }
        del model
        torch.cuda.empty_cache()
    report["gpu"] = {"guard": guard, "memory_at_end": memory_report(), "peak_allocated_bytes": int(torch.cuda.max_memory_allocated())}
    _write_json(args.out, report)
    _log(f"→ {args.out}")
    return 0


# --------------------------------------------------------------------------
# upload, verify
# --------------------------------------------------------------------------


def _upload(args: argparse.Namespace) -> int:
    from huggingface_hub import HfApi

    from robo_jev.hf.export import ADAPTER_REPO, MERGED_REPO, TAG, check_no_pickle
    from robo_jev.hf.hub import ensure_private_repo, load_token, redact, remote_listing, tag_release, upload

    api = HfApi(token=load_token(args.token_file))
    try:
        me = api.whoami()
        orgs = [org.get("name") for org in me.get("orgs", [])]
        _log(f"authenticated as {me.get('name')} (orgs {orgs})")
        result: dict[str, Any] = {"user": me.get("name"), "orgs": orgs, "repos": {}}
        for repo_id, folder in ((ADAPTER_REPO, args.root / ADAPTER_DIR), (MERGED_REPO, args.root / MERGED_DIR)):
            check_no_pickle(folder)
            result["repos"][repo_id] = {"repo": ensure_private_repo(api, repo_id)}
            _log(f"{repo_id}: private repo ready ({result['repos'][repo_id]['repo']})")
            result["repos"][repo_id]["upload"] = upload(api, repo_id, folder, message=args.message)
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


def _answers_equal(a: dict[str, Any], b: dict[str, Any]) -> bool:
    return json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)


def _verify_hub(args: argparse.Namespace) -> int:
    guard = limit_gpu_memory(args.gpu_memory_fraction)
    _log(f"gpu guard {guard} · memory {memory_report()}")
    import torch

    from robo_jev.decisions.records import read_records, shuffle_key
    from robo_jev.hf.export import ADAPTER_REPO, MERGED_REPO, TAG
    from robo_jev.hf.hub import load_token, redact
    from robo_jev.infer import RoboJev

    records = read_records(args.input)[: args.limit]
    cache = Path(tempfile.mkdtemp(prefix="hf0-clean-cache-", dir=args.cache_parent))
    token = load_token(args.token_file)
    report: dict[str, Any] = {"records": [r["id"] for r in records], "cache": str(cache), "tag": TAG, "runs": {}}

    def run_robojev(label: str, source: str, **kwargs: Any) -> dict[str, Any]:
        require_free(MIN_FREE, what=f"load {label}")
        started = time.perf_counter()
        model = RoboJev.from_pretrained(source, device="cuda", **kwargs)
        answers = {str(r["id"]): model.decide(r["request"]["state"], r["request"]["questions"], key=shuffle_key(r), orders=args.orders)
                   for r in records}
        report["runs"][label] = {"model": model.name, "seconds": round(time.perf_counter() - started, 1)}
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
            require_free(MIN_FREE, what=f"standalone {variant}")
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
            report["runs"][f"standalone {variant}"] = {"seconds": round(time.perf_counter() - started, 1), "script": str(script)}
    except Exception as error:  # noqa: BLE001
        raise SystemExit(f"verify failed: {redact(error)}") from None
    checks = {}
    for variant in ("merged", "adapter"):
        checks[f"hub {variant} RoboJev == local"] = all(_answers_equal(hub[variant][k], local[variant][k]) for k in local[variant])
        checks[f"hub {variant} standalone == local"] = all(_answers_equal(standalone[variant][k], local[variant][k]) for k in local[variant])
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
    p.add_argument("--mix-manifest", type=Path, default=REPO / "artifacts/simgen/v01-mix/train-50k-mix.manifest.json")
    p.add_argument("--data-manifest", type=Path, default=REPO / "artifacts/simgen/v01/train-50k.manifest.json")
    p.add_argument("--data-stats", type=Path, default=REPO / "artifacts/simgen/v01/train-50k.stats.json")
    p.add_argument("--probe-manifest", type=Path, default=REPO / "artifacts/simgen/probes/G1-train-mix.manifest.json")
    p = sub.add_parser("cards", help="re-render README.md (model cards) in an existing export")
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--equivalence", type=Path, default=None, help="the equivalence report with the example's answers per variant")
    p = sub.add_parser("sample", help="the equivalence records (50 T1, 25 T2, 25 G2)")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--seed", type=int, default=0)
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
    p = sub.choices["verify-hub"]
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--limit", type=int, default=5)
    p.add_argument("--orders", choices=("single", "balanced"), default="single")
    p.add_argument("--token-file", type=Path, default=DEFAULT_TOKEN_FILE)
    p.add_argument("--cache-parent", type=Path, default=REPO / "artifacts/scratch/hf0")
    p.add_argument("--keep-cache", action="store_true")
    p = sub.add_parser("upload", help="private repos + upload + tag")
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--token-file", type=Path, default=DEFAULT_TOKEN_FILE)
    p.add_argument("--message", default="robo-jev v0.1 (v0.1-mix): weights, tokenizer, robo_jev.json, model card, standalone runner")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    handlers = {"figures": _figures, "export": _export, "cards": _cards, "sample": _sample, "equivalence": _equivalence, "upload": _upload, "verify-hub": _verify_hub}
    return handlers[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
