#!/usr/bin/env python
"""Task v0.8a — the prompt-identity proof: records without the v0.8 metas get byte-identical prompts and targets from the new code and the
base code (``88653e1``). Hashes only — no prompt text is printed or written (the sealed sets are read in code only).

    # the same command twice: once with the base code on PYTHONPATH (git archive 88653e1 src), once with this branch's
    python scripts/v08_prompt_identity.py digest --out base.json   # PYTHONPATH=<base>/src
    python scripts/v08_prompt_identity.py digest --out new.json    # PYTHONPATH=<worktree>/src
    python scripts/v08_prompt_identity.py compare base.json new.json

``digest`` reads a seeded sample of records of every set (train-v07 and each sealed test set) and hashes, per set, every prompt of
:func:`robo_jev.decisions.prompt.record_prompts` (chat, direct, prompt v0c-p3; the seeded order and the counterbalanced listings, as the
diagnosis tool builds them) and, for the training input, every training example of :func:`robo_jev.decisions.examples.build_examples` (two
order seeds) and every dev-style example (counterbalanced listings with their targets): the shown options, lines, values, codes, code ids,
tokens, text and target. It uses only functions both code versions have. ``compare`` checks the two digests are equal set by set.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

ARTIFACTS = Path("/home/rodex_dgx/Desktop/projects/robo-jev/artifacts")
SETS = {
    "train-v07": (ARTIFACTS / "simgen" / "v07" / "train-v07.jsonl", 3000),
    **{f"T{i}": (ARTIFACTS / "simgen" / "v0b" / "tests" / f"T{i}.jsonl", 150) for i in range(1, 8)},
    "G1": (ARTIFACTS / "simgen" / "probes" / "G1.jsonl", 150), "G2": (ARTIFACTS / "simgen" / "probes" / "G2.jsonl", 150),
    "T8a": (ARTIFACTS / "t8" / "T8a.jsonl", 150), "T8b": (ARTIFACTS / "t8" / "T8b.jsonl", 150), "T9": (ARTIFACTS / "t9" / "T9.jsonl", 150),
    "B1": (ARTIFACTS / "b1" / "b1-infamily.jsonl", 150), "H-set": (ARTIFACTS / "simgen" / "v04" / "scene-holdout.jsonl", 150),
    "A4": (ARTIFACTS / "v06" / "a4" / "test.jsonl", 150), "M1": (ARTIFACTS / "iface" / "m1" / "m1-set.jsonl", 150),
    "M2-short": (ARTIFACTS / "iface" / "m2" / "M2-short.jsonl", 100), "M2-w8k": (ARTIFACTS / "iface" / "m2" / "M2-w8k.jsonl", 100),
    "M2g": (ARTIFACTS / "v07" / "gates" / "m2g" / "M2g-all-conditions.jsonl", 150), "M1g": (ARTIFACTS / "v07" / "gates" / "m1g" / "M1g.jsonl", 150),
    "M2g-F": (ARTIFACTS / "v07" / "gates" / "m2gf" / "M2g-facts.jsonl", 150),
}
FIELDS = ("question", "type", "listing", "n_listings", "options", "option_lines", "values", "codes", "code_ids", "tokens", "prefix_tokens", "text",
          "prefix_chars", "format", "mode", "version")


def _sample(path: Path, size: int, seed: str) -> list[dict]:
    lines = [line for line in path.open(encoding="utf-8") if line.strip()]
    chosen = sorted(random.Random(seed).sample(range(len(lines)), min(size, len(lines))))
    return [json.loads(lines[i]) for i in chosen]


def cmd_digest(args: argparse.Namespace) -> int:
    import robo_jev
    from robo_jev.decisions import examples as E
    from robo_jev.decisions import prompt as P
    from robo_jev.model.tokenizer import available_tokenizer, load_tokenizer

    tokenizer = load_tokenizer(available_tokenizer()[0])
    out = {"code": str(Path(robo_jev.__file__).resolve()), "sets": {}}
    for name, (path, size) in SETS.items():
        records = _sample(path, size, f"v08a-identity|{name}")
        digest = hashlib.sha256()
        count = 0
        meta_free = 0
        for record in records:
            meta = record.get("meta") or {}
            if "codes" in meta or meta.get("option_range") not in (None, {"version": "options-26-1", "max": 26}):
                continue  # only records without the v0.8 metas (none of these sets has any)
            meta_free += 1
            for orders in ("seeded", "counterbalanced"):
                for p in P.record_prompts(record, tokenizer, seed=1, fmt="chat", mode="direct", orders=orders):
                    digest.update(json.dumps({k: p[k] for k in FIELDS}, sort_keys=True).encode("utf-8"))
                    count += 1
            if name.startswith("train"):
                names = list(record["request"]["questions"])
                for seed in (0, 7):
                    for e in E.build_examples(record, names, tokenizer, order_seed=E.derive_seed(seed, 0, record["id"], "order")):
                        digest.update(json.dumps({k: e[k] for k in FIELDS + ("target", "clamped")}, sort_keys=True).encode("utf-8"))
                        count += 1
                for e in E.dev_examples([record], tokenizer):
                    digest.update(json.dumps({k: e[k] for k in FIELDS + ("target", "clamped")}, sort_keys=True).encode("utf-8"))
                    count += 1
        out["sets"][name] = {"records": len(records), "records_without_v08_metas": meta_free, "prompts_and_examples": count, "sha256": digest.hexdigest()}
        print(f"{name}: {len(records)} records, {count} prompts/examples", file=sys.stderr, flush=True)
    Path(args.out).write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    a, b = (json.loads(Path(p).read_text(encoding="utf-8")) for p in (args.base, args.new))
    rows = {name: {"base": a["sets"].get(name), "new": b["sets"].get(name), "identical": a["sets"].get(name) == b["sets"].get(name)}
            for name in sorted(set(a["sets"]) | set(b["sets"]))}
    result = {"base_code": a["code"], "new_code": b["code"], "sets": rows, "all_identical": all(r["identical"] for r in rows.values()),
              "prompts_and_examples": sum((r["new"] or {}).get("prompts_and_examples", 0) for r in rows.values())}
    if args.out:
        Path(args.out).write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "sets"}))
    return 0 if result["all_identical"] else 2


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    d = sub.add_parser("digest")
    d.add_argument("--out", required=True)
    d.set_defaults(func=cmd_digest)
    c = sub.add_parser("compare")
    c.add_argument("base")
    c.add_argument("new")
    c.add_argument("--out", default=None)
    c.set_defaults(func=cmd_compare)
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
