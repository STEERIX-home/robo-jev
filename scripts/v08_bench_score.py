#!/usr/bin/env python
"""Task v0.8b — one model answers bench-shaped requests through the bundle's runner (B8, the Jev comparison set, the storage-bin probes; GPU unit).

    artifacts/scratch/v08b/gpu.sh b8-base scripts/v08_bench_score.py --model Qwen/Qwen3.5-2B --set B8 --tag base-2b \\
        --input artifacts/v08/gates/b8/B8.jsonl --frozen artifacts/v08/gates/b8/frozen-before-scoring.json --out artifacts/reports/v08/spark/B8-base-2b.json
    … --checkpoint <run>-fetched/runs/<name>/checkpoint.pt      (a trained LoRA, applied to the pinned base as the diagnosis tool applies it)

**The path** is the bench's: the committed bundle runner ``src/robo_jev/hf/robo_jev_decide.py`` (``RoboJevRunner`` — the file the Hub carries
and robo-jev-serve runs; prompt v0c-p3; options shown with A–Z, then the two-letter codes of its ``CODES`` — the same first 255 codes v0.8
trains with), loaded by path, wrapping the backbone as task rel07's bench check did. Each record's target questions (``meta.target``; else
every choice question) are answered one question per call (a question's prompt does not depend on the others), in each reading:

* ``single-s<K>`` — ``orders="single"``, seed K, the runner's default key (sha256 of the state) — the deployed order is ``single-s1``;
* ``balanced-s<K>`` — ``orders="balanced"`` (every option in every position once), seed K;
* ``order-none`` — ``orders="single"``, seed None: the request's own order (the probe's position cases).

Readings come from the record (``meta.readings``) or ``--readings``. Single readings also store the code each option was shown with.
**B8**: refuses unless ``--frozen`` is given and the tags and B8 file are byte-identical to what was frozen before scoring; the frozen record
is copied into the output. Writes one JSON: ``{"kind": "v08-bench-scores", "set", "model", "tag", "checkpoint", "input", "input_sha256",
"frozen", "bundle", "bundle_sha256", "readings", "answers": {record id: {question: {reading: {"choice", "probabilities", "codes"?}}}}, …}``.
GPU rules: fence 0.6 before the first CUDA allocation, the page cache of the model files dropped and ``require_free`` (20 GiB) before loading.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib.util
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from robo_jev.gpu import DEFAULT_FRACTION, limit_gpu_memory, memory_report, require_free  # noqa: E402

SCRIPT_VERSION = "v08-bench-score-1"
BASE = "Qwen/Qwen3.5-2B"
#: robo-jev-serve's commit whose adapter R14 replicates (r14_v08.SERVE_COMMIT).
SERVE_COMMIT = "7b54004bb335c2abdcd3675054221a194166cf3b"
BUNDLE = REPO / "src" / "robo_jev" / "hf" / "robo_jev_decide.py"
MIN_FREE = 20 * 2**30
SETS = ("B8", "Jev", "Probe", "R14", "R14serve", "A16", "A16y", "S16", "A16s", "Collected", "Other")
#: Reported sets frozen in artifacts/v08/r16/ (task v0.8c): sealed for training, already sent by the bench.
R16_SETS = ("A16", "A16y", "S16")
_READING = re.compile(r"^(single|balanced)-s(\d+)$|^order-none$|^spaces-(\d)$")


def _log(message: str) -> None:
    print(f"[v08-bench {dt.datetime.now().strftime('%H:%M:%S')}] {message}", file=sys.stderr, flush=True)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_reading(name: str) -> dict[str, Any]:
    """``single-s1`` → {"name", "orders": "single", "seed": 1}; ``order-none`` → seed None; ``spaces-K`` → robo-jev-serve's order seeds:
    the deployed single order (seed 1, the default key) of the state with K trailing spaces — the key (sha256 of the state as received)
    changes, the prompt (built from ``state.strip()``) does not."""
    found = _READING.match(str(name))
    if not found:
        raise ValueError(f"reading {name!r}: single-s<K>, balanced-s<K>, order-none or spaces-<K>")
    if name == "order-none":
        return {"name": name, "orders": "single", "seed": None}
    if found.group(3) is not None:
        return {"name": name, "orders": "single", "seed": 1, "spaces": int(found.group(3))}
    return {"name": name, "orders": found.group(1), "seed": int(found.group(2))}


def readings_of(record: dict[str, Any], default: list[str]) -> list[dict[str, Any]]:
    given = (record.get("meta") or {}).get("readings")
    if given:
        return [{"name": r["name"], "orders": r["orders"], "seed": r["seed"], **({"spaces": r["spaces"]} if "spaces" in r else {})} for r in given]
    return [parse_reading(name) for name in default]


def normalise_collected(lines: list[str]) -> list[dict[str, Any]]:
    """A user's JSONL of collected requests → records: a line may be a record ``{id, request: {state, questions}}``, ``{request: {...}}`` or a
    raw Decisions body ``{state, questions, ...}``; a missing id becomes ``collected-<N>`` with N the 1-based physical line number (blank
    lines skipped but counted) — the verdict's numbering."""
    out = []
    for number, text in enumerate(lines, 1):
        if not text.strip():
            continue
        line = json.loads(text)
        request = line.get("request", line)
        out.append({"id": str(line.get("id") or f"collected-{number}"), "meta": dict(line.get("meta") or {}),
                    "request": {"state": request["state"], "questions": request["questions"]}})
    return out


def serve_adapted(record: dict[str, Any]) -> dict[str, Any]:
    """robo-jev-serve's adapter (:func:`robo_jev.bench.r14_v08.model_questions`): the record with only the questions serve sends to the model
    that R14 reads — 1-option choices split off (answered with p = 1 by the server), scores with ≥ 2 criteria asked as level choices —
    and ``meta.target`` = those questions; ``meta.serve`` = what was converted, split off or skipped."""
    from robo_jev.bench.r14_v08 import model_questions

    mq = model_questions(record["request"]["questions"])
    own = (record.get("meta") or {}).get("target")  # a record's own target (e.g. A16s: x, y, z) stays: serve's adapter only drops questions
    target = [name for name in (own or mq["asked"]) if name in mq["asked"]]
    meta = {**(record.get("meta") or {}), "target": target, "serve": {k: mq[k] for k in ("converted", "trivial", "skipped")}}
    return {**record, "meta": meta, "request": {"state": record["request"]["state"], "questions": {n: mq["asked"][n] for n in target}}}


def targets_of(record: dict[str, Any]) -> list[str]:
    questions = record["request"]["questions"]
    target = (record.get("meta") or {}).get("target")
    names = list(target) if target else [n for n, q in questions.items() if q.get("type") == "choice"]
    missing = [n for n in names if n not in questions]
    if missing:
        raise ValueError(f"{record.get('id')}: target questions {missing} not in the request")
    return names


def check_frozen(input_path: Path, frozen_path: Path | None, set_name: str) -> dict[str, Any] | None:
    """B8 and R14: the frozen-before-scoring record, after checking the scored file and its tags against it (refuses otherwise)."""
    if set_name in ("R14", "R14serve"):
        from robo_jev.bench.r14_v08 import check_frozen as r14_frozen

        if frozen_path is None or not Path(frozen_path).is_file():
            raise SystemExit("R14 is scored only against frozen tags: pass --frozen <artifacts/v08/r14/frozen-before-scoring.json>")
        frozen = json.loads(Path(frozen_path).read_text(encoding="utf-8"))
        problems = r14_frozen(Path(frozen_path).parent)
        if sha256(input_path) not in (frozen.get("files") or {}).values():
            problems.append(f"{input_path} is not one of the frozen R14 files")
        if problems:
            raise SystemExit("refusing to score R14: " + "; ".join(problems))
        return frozen
    if set_name in R16_SETS:
        from robo_jev.bench.r16_v08 import check_frozen as r16_frozen

        if frozen_path is None or not Path(frozen_path).is_file():
            raise SystemExit(f"{set_name} is scored only against its frozen files: pass --frozen <artifacts/v08/r16/frozen-before-scoring.json>")
        frozen = json.loads(Path(frozen_path).read_text(encoding="utf-8"))
        problems = r16_frozen(Path(frozen_path).parent)
        if sha256(input_path) not in (frozen.get("files") or {}).values():
            problems.append(f"{input_path} is not one of the frozen A16 / S16 files")
        if problems:
            raise SystemExit(f"refusing to score {set_name}: " + "; ".join(problems))
        return frozen
    if set_name == "A16s":  # task v0.8d: the arm's distance sweep, frozen in artifacts/v08/a16s/
        from robo_jev.bench.a16s_v08 import check_frozen as a16s_frozen

        if frozen_path is None or not Path(frozen_path).is_file():
            raise SystemExit("A16s is scored only against its frozen file: pass --frozen <artifacts/v08/a16s/frozen-before-scoring.json>")
        frozen = json.loads(Path(frozen_path).read_text(encoding="utf-8"))
        problems = a16s_frozen(Path(frozen_path).parent)
        if sha256(input_path) not in (frozen.get("files") or {}).values():
            problems.append(f"{input_path} is not the frozen A16s file")
        if problems:
            raise SystemExit("refusing to score A16s: " + "; ".join(problems))
        return frozen
    if set_name != "B8":
        return None
    if frozen_path is None or not Path(frozen_path).is_file():
        raise SystemExit("B8 is scored only against frozen tags: pass --frozen <frozen-before-scoring.json>")
    frozen = json.loads(Path(frozen_path).read_text(encoding="utf-8"))
    tags = Path(frozen_path).with_name("tags.json")
    problems = []
    if sha256(input_path) != frozen.get("b8_sha256"):
        problems.append(f"{input_path} is not the frozen B8 file")
    if not tags.is_file() or sha256(tags) != frozen.get("tags_sha256"):
        problems.append(f"{tags} is not the frozen tags file")
    if problems:
        raise SystemExit("refusing to score B8: " + "; ".join(problems))
    return frozen


def drop_page_cache(*roots: Path) -> int:
    """Evict our big model files' page cache (fsync + POSIX_FADV_DONTNEED) so require_free reads the real free memory (task rel07)."""
    covered = 0
    for root in roots:
        files = [root] if root.is_file() else [q for q in root.rglob("*") if q.is_file()] if root.is_dir() else []
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


def bundle_module():
    spec = importlib.util.spec_from_file_location("robo_jev_decide_v08b", BUNDLE)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    previous, sys.dont_write_bytecode = sys.dont_write_bytecode, True
    try:
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = previous
    return module


def answer_record(runner: Any, bundle: Any, tokenizer: Any, record: dict[str, Any], readings: list[dict[str, Any]]) -> tuple[dict[str, Any], int, int]:
    """{question: {reading: answer}} of one record, the prompts and the prompt tokens it took."""
    state = record["request"]["state"]
    out: dict[str, Any] = {}
    prompts = tokens = 0
    for name in targets_of(record):
        question = {name: record["request"]["questions"][name]}
        for reading in readings:
            shown_state = state + " " * int(reading.get("spaces") or 0)  # serve's order seeds: the key changes, the prompt does not
            shown = bundle.build_prompts(shown_state, question, tokenizer, seed=reading["seed"], orders=reading["orders"])
            prompts += len(shown)
            tokens += sum(len(p["tokens"]) for p in shown)
            answer = runner.decide(shown_state, question, orders=reading["orders"], seed=reading["seed"])[name]
            entry = {"choice": answer.get("choice"), "probabilities": answer.get("probabilities")} if answer["type"] == "choice" else dict(answer)
            if reading["orders"] == "single" and answer["type"] == "choice":
                entry["codes"] = dict(zip(shown[0]["options"], shown[0]["codes"]))
            out.setdefault(name, {})[reading["name"]] = entry
    return out, prompts, tokens


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default=BASE)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--set", choices=SETS, required=True)
    parser.add_argument("--tag", required=True, help="the model's tag in the file names (v08Fs0-2b, …, base-2b)")
    parser.add_argument("--frozen", type=Path, default=None, help="B8: frozen-before-scoring.json (required)")
    parser.add_argument("--readings", nargs="+", default=["single-s1"], help="when a record does not name its readings")
    parser.add_argument("--limit", type=int, default=None, help="smoke tests: the first K records")
    parser.add_argument("--env", nargs="+", default=None, help="only records of these meta.env values (smoke tests)")
    parser.add_argument("--serve-adapter", action="store_true", help="robo-jev-serve's adapter (R14): split 1-option choices, scores as level choices")
    parser.add_argument("--merged", type=Path, default=None, help="a merged HF folder loaded with the bundle's load() instead of base + checkpoint")
    parser.add_argument("--gpu-memory-fraction", type=float, default=DEFAULT_FRACTION)
    args = parser.parse_args(argv)
    for name in args.readings:
        parse_reading(name)
    guard = limit_gpu_memory(args.gpu_memory_fraction)  # before the first CUDA allocation
    _log(f"gpu guard {guard} · memory {memory_report()}")
    from robo_jev.decisions.records import read_records

    frozen = check_frozen(args.input, args.frozen, args.set)
    if args.set == "Collected":
        records = normalise_collected(Path(args.input).read_text(encoding="utf-8").splitlines())
    else:
        records = read_records(args.input)
    if args.env:
        records = [r for r in records if (r.get("meta") or {}).get("env") in set(args.env)]
    if args.limit:
        records = records[: args.limit]
    if args.serve_adapter:
        records = [serve_adapted(r) for r in records]
    if args.set not in ("Probe", "R14", "R14serve", "A16s", *R16_SETS):
        from robo_jev.data.sealed import sealed_record_set

        sealed = sorted({sealed_record_set(r) for r in records} - {None})
        if sealed and args.set != "B8":
            raise SystemExit(f"{args.input}: records of sealed sets {sealed} — only B8 is scored here as a sealed gate")
    import torch

    from robo_jev.decisions.train import apply_checkpoint
    from robo_jev.model.backbone_qwen import QwenBackbone
    from robo_jev.model.tokenizer import available_tokenizer, describe_tokenizer, load_tokenizer

    bundle = bundle_module()
    tokenizer_id = available_tokenizer()[0]
    tokenizer, tokenizer_sha = load_tokenizer(tokenizer_id), describe_tokenizer(tokenizer_id)["sha256"]
    big = [REPO / "artifacts" / "models" / "Qwen" / "Qwen3.5-2B"] + ([args.checkpoint] if args.checkpoint else [])
    dropped = drop_page_cache(*[Path(p) for p in big])
    _log(f"dropped the page cache of {dropped / 2**30:.1f} GiB · memory {memory_report()}")
    require_free(MIN_FREE, what=f"load {args.model}")
    started = time.perf_counter()
    info = None
    if args.merged is not None:  # the released merged folder, as the Hub and robo-jev-serve load it
        loaded = bundle.load(str(args.merged), device="cuda")
        runner, backbone = bundle.RoboJevRunner(loaded.model, tokenizer), None
        info = {"mode": "merged", "path": str(args.merged), "sha256": sha256(Path(args.merged) / "model.safetensors")}
    else:
        backbone = QwenBackbone.load(args.model)
        if args.checkpoint is not None:
            info = apply_checkpoint(backbone, args.checkpoint, tokenizer_sha256=tokenizer_sha, prompt_version=bundle.PROMPT_VERSION)
            info = {"run_id": info.get("run_id"), "step": info.get("step"), "mode": info.get("mode"), "path": str(args.checkpoint), "sha256": sha256(args.checkpoint)}
        runner = bundle.RoboJevRunner(backbone.model, tokenizer)
    load_seconds = time.perf_counter() - started
    _log(f"loaded {args.model}{' + ' + str(args.checkpoint) if args.checkpoint else ''} in {load_seconds:.1f} s · memory {memory_report()}")
    answers: dict[str, Any] = {}
    readings: dict[str, Any] = {}
    prompts = tokens = 0
    started = time.perf_counter()
    for i, record in enumerate(records, 1):
        wanted = readings_of(record, args.readings)
        for r in wanted:
            readings[r["name"]] = {"orders": r["orders"], "seed": r["seed"]}
        answers[str(record["id"])], p, t = answer_record(runner, bundle, tokenizer, record, wanted)
        prompts, tokens = prompts + p, tokens + t
        if i % 25 == 0:
            _log(f"{i}/{len(records)} records · {prompts} prompts · {tokens:,} tokens · {time.perf_counter() - started:.0f} s")
    seconds = time.perf_counter() - started
    label = args.model if info is None else (f"merged:{args.merged}" if info.get("mode") == "merged" else f"{args.model}+{info['mode']}:{info['run_id']}@{info['step']}")
    out = {"kind": "v08-bench-scores", "set": args.set, "model": label,
           "tag": args.tag, "checkpoint": info, "input": str(args.input), "input_sha256": sha256(args.input), "records": len(records),
           "frozen": frozen, "bundle": str(BUNDLE.relative_to(REPO)), "bundle_sha256": sha256(BUNDLE), "prompt_version": bundle.PROMPT_VERSION,
           "serve_adapter": {"used": bool(args.serve_adapter), "serve_commit": SERVE_COMMIT if args.serve_adapter else None,
                             "per_record": {str(r["id"]): r["meta"].get("serve") for r in records} if args.serve_adapter else None},
           "env_filter": args.env, "merged": str(args.merged) if args.merged else None,
           "key": "the runner's default (sha256 of the state)", "readings": readings, "answers": answers, "prompts": prompts, "prompt_tokens": tokens,
           "seconds": round(seconds, 1), "load_seconds": round(load_seconds, 1), "ms_per_token": round(1000 * seconds / max(1, tokens), 4),
           "tokenizer_sha256": tokenizer_sha, "script_version": SCRIPT_VERSION, "gpu": {"guard": guard, "memory_at_end": memory_report()},
           "created": dt.datetime.now().astimezone().isoformat(timespec="seconds")}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    _log(f"{len(records)} records, {prompts} prompts, {tokens:,} tokens in {seconds:.0f} s ({out['ms_per_token']} ms/token) → {args.out}")
    del runner, backbone
    torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
