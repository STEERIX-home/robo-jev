#!/usr/bin/env python
"""Task v0.8a — the v0.8 training input: v0.7's input with the FORMAT package, labels unchanged (PRE-DECLARED v0.8 VERDICT RULE).

    # the whole assembly, one command (CPU unit, ≤ 48 GB): transform → sealed checks → exposure, tokens, time, codes, H-prior table
    python scripts/assemble_v08.py all --out-dir artifacts/simgen/v08
    # the steps one by one
    python scripts/assemble_v08.py transform --out artifacts/simgen/v08/train-v08.jsonl
    python scripts/assemble_v08.py overlap --input artifacts/simgen/v08/train-v08.jsonl --out-dir artifacts/simgen/v08
    python scripts/assemble_v08.py exposure --input artifacts/simgen/v08/train-v08.jsonl --out artifacts/simgen/v08/train-v08.exposure.json
    python scripts/assemble_v08.py weights --input artifacts/simgen/v08/train-v08.jsonl --out artifacts/simgen/v08/train-v08.prior-weights.json

**The transform** (:mod:`robo_jev.simgen.format_v08` holds the pure parts): every record of ``train-v07.jsonl`` (sha256 20dd9bfc…) is read
in lockstep with its source in ``train-v06.jsonl`` (sha256 42d6bd15…, same ids in the same order — v0.7's own input, which holds every
question's original options and instructions). v0.7's distractor pools and pair facts are rebuilt exactly as v0.7 built them
(``assemble_v07.Plan`` over v0.6's records). Records are **transformed, never added**:

* v0.7's expanded questions get a new N, log-uniform over 9..255, and new distractors drawn by v0.7's rules (physics: every distinct key
  the allowed families offer, fewer when they run out — recorded as short; general: texts may share words with each other); expanded general
  items are re-keyed ``o1``…``oN`` in a seeded order (labels move with their options);
* ≈ 25 % of all choice questions show random pool codes (``meta.codes``);
* a record with more than 8 options carries the 255-option opt-in; ``meta.format`` records what was done.

**Everything else is v0.7's, byte for byte**: ids, order, labels (an expanded question's own options and probabilities are v0.7's — v0.6's —
and its distractors get 0), v0.7's sheets, facts and ``meta.shape``, every untouched line. Records the trainer holds out as dev are never
transformed. ``transform`` proves it record by record.

``overlap`` runs the sealed checks (T1–T7, G1, G2, T8a, T8b, T9, B1, the H-set, A4, the iface M1 / M2 sets, M2g, M1g and M2g-F; plus v0.7b's
M1g / M2g unit-and-shingle scan and v0.7f's M2g-F piece scan, hashes and counts only). ``exposure`` proves the trainer's epoch view equals
v0.7's, measures N per bin, the correct option's position and code (per pool code, both seeds), the token growth (the trainer's packing,
assigned codes included) and the expected training time and cost, scaled from v0.7's **measured** pods. ``weights`` writes the H-prior class
table and weight statistics per family (the FP arm's trainer option) for both seeds.
"""

from __future__ import annotations

import argparse
import datetime as dt
import functools
import hashlib
import importlib.util
import json
import math
import random
import statistics
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

SCRIPT_VERSION = "v08a-assemble-1"
CODE_PATHS = ("scripts/assemble_v08.py", "src/robo_jev/simgen/format_v08.py", "src/robo_jev/decisions/codes.py", "src/robo_jev/decisions/records.py",
              "src/robo_jev/decisions/prompt.py", "src/robo_jev/decisions/examples.py", "src/robo_jev/decisions/prior_weights.py",
              "scripts/assemble_v07.py", "src/robo_jev/simgen/shape.py")
ARTIFACTS = REPO / "artifacts"
TRAIN_V06 = ARTIFACTS / "simgen" / "v06" / "train-v06.jsonl"
TRAIN_V06_SHA = "42d6bd150ce286bb0aff66c400e7a485443a25f5c52c2cf644b91292c6901aa4"
TRAIN_V07 = ARTIFACTS / "simgen" / "v07" / "train-v07.jsonl"
TRAIN_V07_SHA = "20dd9bfcd0ac6b479842e4ec372d242d9bddc6a0b5001e78f8a75e97f26fabc7"
EXPOSURE_V07 = ARTIFACTS / "simgen" / "v07" / "train-v07.exposure.json"
OUT_DIR = ARTIFACTS / "simgen" / "v08"
TRAIN_CONFIG = REPO / "configs" / "train" / "decisions-v0-lora-h100.yaml"
PHYSICS = ("engine", "mujoco")
#: v0.7's measured pods (artifacts/cloud/runs/v07-s{0,1}-01*.json and the fetched runs' metrics.json): the training steps' seconds, tokens
#: and padded tokens (dev passes excluded), the dev seconds, the whole train step, the pod's lifetime and its upload.
V07_MEASURED = {
    "0": {"step_seconds": 10922.804, "tokens": 119650977, "padded_tokens": 182503618, "dev_seconds": 643.795, "train_step_seconds": 11703.062,
          "pod_seconds": 23397.0, "upload_seconds": 1547.0, "upload_bytes": 949798591, "diagnoses_seconds": 9727.0},
    "1": {"step_seconds": 11011.969, "tokens": 119708499, "padded_tokens": 182627491, "dev_seconds": 645.223, "train_step_seconds": 11797.348,
          "pod_seconds": 23527.0, "upload_seconds": 1541.0, "upload_bytes": 949798591, "diagnoses_seconds": 9714.0},
}
HOURLY_USD = 3.99
RUN_CAP_USD = 45.0
#: Places never written to (the brief): v0.3–v0.7 inputs, t8 / t9 / b1 / v06 / v07 / iface, reports.
NEVER_WRITE = tuple(ARTIFACTS / p for p in ("simgen/v03", "simgen/v04", "simgen/v05", "simgen/v06", "simgen/v07", "simgen/v0b/tests", "simgen/probes",
                                            "simgen/v02", "t8", "t9", "b1", "v06", "v07", "v08", "iface", "reports"))


def _log(message: str) -> None:
    print(f"[v08a {dt.datetime.now().strftime('%H:%M:%S')}] {message}", file=sys.stderr, flush=True)


@functools.lru_cache(maxsize=None)
def _script(name: str):
    spec = importlib.util.spec_from_file_location(f"v08a_{Path(name).stem}", REPO / "scripts" / name)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _git() -> dict[str, Any]:
    def run(*args: str) -> str | None:
        try:
            return subprocess.run(["git", "-C", str(REPO), *args], capture_output=True, text=True, check=True).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return None

    status = run("status", "--porcelain", "--", *CODE_PATHS)
    return {"commit": run("rev-parse", "--short", "HEAD"), "dirty": bool(status) if status is not None else None}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(payload, ensure_ascii=False, indent=1, default=str) + "\n", encoding="utf-8")
    temp.replace(path)


def _raw_lines(path: Path) -> Iterator[bytes]:
    with Path(path).open("rb") as handle:
        for raw in handle:
            if raw.strip():
                yield raw if raw.endswith(b"\n") else raw + b"\n"


def refuse_write(path: Path) -> None:
    resolved = Path(path).resolve()
    for place in NEVER_WRITE:
        place = place.resolve()
        if resolved == place or place in resolved.parents:
            raise SystemExit(f"{path}: refusing to write into {place}")


def _config(path: Path) -> dict[str, Any]:
    return _script("assemble_v04.py")._config(path)


def part_of(record: Mapping[str, Any]) -> str:
    return _script("assemble_v07.py").part_of(record)


def _bin(n: int) -> str:
    from robo_jev.simgen.format_v08 import bin_of

    return bin_of(n)


# --------------------------------------------------------------------------
# transform — one record (pure given the plan)
# --------------------------------------------------------------------------


class Plan:
    """v0.7's plan (``assemble_v07.Plan`` over v0.6's records: the distractor pools, the general pool, each expanded question's group facts —
    the union over a pair of its own keys, option texts and words) and the physics key index of v0.8."""

    def __init__(self, config: Mapping[str, Any]) -> None:
        self.v07 = _script("assemble_v07.py").Plan(config)
        self.config = dict(config)
        self.index = None

    def scan(self, record: Mapping[str, Any]) -> None:
        self.v07.scan(record)

    def finish(self) -> None:
        from robo_jev.simgen.format_v08 import KeyIndex

        self.index = KeyIndex(self.v07.pool)

    def dev(self, record: Mapping[str, Any]) -> bool:
        return self.v07.dev(record)


def _own(question: Mapping[str, Any], count: int) -> list[tuple[str, str]]:
    return list(question["criteria"].items())[: int(count)]


def transform_record(v06: Mapping[str, Any], v07: Mapping[str, Any], plan: Plan) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    """(the v0.8 record — None when nothing changes —, what happened). Pure given the plan (module docstring)."""
    from robo_jev.decisions.records import OPTION_RANGE, OPTION_RANGE_MARK_255, shuffle_key
    from robo_jev.simgen import format_v08 as F
    from robo_jev.simgen import shape

    if str(v06.get("id")) != str(v07.get("id")):
        raise ValueError(f"v0.6 / v0.7 out of step: {v06.get('id')} vs {v07.get('id')}")
    part = part_of(v07)
    key = shuffle_key(dict(v07))
    event: dict[str, Any] = {"part": part, "dev": False, "expanded": {}, "codes": [], "choice_questions": 0}
    if plan.dev(v07):
        event["dev"] = True
        return None, event
    new = json.loads(json.dumps(v07))
    meta07 = v07.get("meta") or {}
    expanded07 = ((meta07.get("shape") or {}).get("expanded")) or {}
    changed = False
    for name, info in expanded07.items():
        own_n = int(info["from"])
        q06, l06 = v06["request"]["questions"][name], v06["labels"][name]
        q07 = v07["request"]["questions"][name]
        if _own(q07, own_n) != list(q06["criteria"].items()):
            raise ValueError(f"{v07['id']}/{name}: v0.7's own options are not v0.6's")
        drawn = F.draw_n(key, name)
        entry: dict[str, Any] = {"family": info.get("family"), "from": own_n, "drawn": drawn, "v07_to": int(info["to"])}
        if part in PHYSICS:
            group = plan.v07.questions[(key, name)]
            picks = plan.index.draw(info["family"], count=drawn - own_n, own_keys=group["keys"], own_norms=group["norms"], words=group["words"],
                                    scene=shape.scene_of(v06, part), rnd=F.rng("draw", key, name))
            if own_n + len(picks) < F.N_RANGE[0]:  # cannot reach 9 options: the question keeps v0.7's expansion
                event.setdefault("kept_v07", []).append(name)
                continue
            base_q = dict(q07)
            base_q["criteria"] = dict(q06["criteria"])
            new_q, new_l = shape.expand_choice(base_q, l06, [(e.key, e.description) for e in picks])
            entry["sources"] = dict(Counter(e.family for e in picks))
            entry["rekeyed"] = False
        else:
            texts, count = None, drawn - own_n
            while texts is None and count >= max(1, F.N_RANGE[0] - own_n):
                texts = F.general_distractors(v06, plan.v07.general_pool, count=count, rnd=F.rng("draw-general", key, count))
                if texts is None:
                    count = int(count * 0.8)
            if texts is None:  # cannot reach 9 options: the item keeps v0.7's expansion
                event.setdefault("kept_v07", []).append(name)
                continue
            criteria, mapping = F.rekey(_own(q06, own_n), texts, F.rng("rekey", key))
            new_q = json.loads(json.dumps(q07))
            new_q["criteria"] = criteria
            new_l = json.loads(json.dumps(l06))
            probabilities = {mapping[k]: p for k, p in l06["probabilities"].items()}
            new_l["probabilities"] = {k: probabilities.get(k, 0.0) for k in criteria}
            if isinstance(l06.get("expected_cost"), dict) and l06["expected_cost"]:
                worst = max(float(v) for v in l06["expected_cost"].values())
                costs = {mapping[k]: v for k, v in l06["expected_cost"].items()}
                new_l["expected_cost"] = {k: costs.get(k, worst) for k in criteria}
            entry["rekeyed"] = True
            entry["mapping"] = mapping
        entry["to"] = len(new_q["criteria"])
        entry["short"] = entry["to"] < drawn
        new["request"]["questions"][name], new["labels"][name] = new_q, new_l
        event["expanded"][name] = entry
        changed = True
    codes: dict[str, list[str]] = {}
    for name, question in new["request"]["questions"].items():
        if question.get("type") != "choice":
            continue
        event["choice_questions"] += 1
        if F.codes_selected(key, name):
            codes[name] = F.draw_codes(key, name, len(question["criteria"]))
    event["codes"] = list(codes)
    if codes:
        changed = True
    if not changed:
        return None, event
    meta = new.setdefault("meta", {})
    wide = any(q.get("type") == "choice" and len(q["criteria"]) > OPTION_RANGE[1] for q in new["request"]["questions"].values())
    if wide:
        meta["option_range"] = dict(OPTION_RANGE_MARK_255)
    else:
        meta.pop("option_range", None)
    if codes:
        meta["codes"] = codes
    meta["format"] = {"version": F.FORMAT_VERSION,
                      "expanded": {n: {k: e[k] for k in ("family", "from", "drawn", "to", "short", "rekeyed")} for n, e in event["expanded"].items()},
                      "codes": sorted(codes)}
    return new, event


def _norm_option(key: str, description: str) -> str:
    """An option's text as the distractor pool dedups it (the description, or the key when the description is empty)."""
    from robo_jev.simgen import shape

    return shape._norm(description) or shape._norm(key)


def record_problems(v06: Mapping[str, Any], v07: Mapping[str, Any], new: Mapping[str, Any], event: Mapping[str, Any]) -> list[str]:
    """Every way a v0.8 record breaks the rule (empty = fine): everything but the format package is v0.7's (module docstring)."""
    from robo_jev.decisions.codes import CODE_POOL
    from robo_jev.decisions.records import OPTION_RANGE, OPTION_RANGE_MARK_255, validate_record
    from robo_jev.simgen.format_v08 import N_RANGE

    problems = []
    if list(v07) != list(new) or any(v07[k] != new[k] for k in v07 if k not in ("request", "labels", "meta")):
        problems.append("a top-level field other than request / labels / meta changed")
    old_meta, new_meta = v07.get("meta") or {}, new.get("meta") or {}
    kept = {k: v for k, v in new_meta.items() if k not in ("format", "codes", "option_range")}
    if kept != {k: v for k, v in old_meta.items() if k != "option_range"}:
        problems.append("meta changed beyond meta.format / meta.codes / meta.option_range")
    if [k for k in new_meta if k in old_meta and k != "option_range"] != [k for k in old_meta if k != "option_range"]:
        problems.append("the order of v0.7's meta keys changed")
    if new["request"]["state"] != v07["request"]["state"] or set(new["request"]) != set(v07["request"]):
        problems.append("the state (v0.7's sheet / facts / original) or the request keys changed")
    oq, nq = v07["request"]["questions"], new["request"]["questions"]
    if list(oq) != list(nq) or set(v07["labels"]) != set(new["labels"]):
        problems.append("question names / order or label names changed")
        return problems
    expanded = event.get("expanded") or {}
    for name in oq:
        a, b = oq[name], nq[name]
        if {k: v for k, v in a.items() if k != "criteria"} != {k: v for k, v in b.items() if k != "criteria"}:
            problems.append(f"{name}: a question field other than the options changed (instructions are v0.7's)")
        la, lb = v07["labels"][name], new["labels"][name]
        if name not in expanded:
            if a.get("criteria") != b.get("criteria") or la != lb:
                problems.append(f"{name}: options or label changed without an expansion")
            continue
        e = expanded[name]
        q06, l06 = v06["request"]["questions"][name], v06["labels"][name]
        mapping = e.get("mapping") or {k: k for k in q06["criteria"]}
        for old_key, text in q06["criteria"].items():
            new_key = mapping.get(old_key)
            if new_key not in b["criteria"] or b["criteria"][new_key] != text:
                problems.append(f"{name}: own option {old_key!r} is not kept as {new_key!r}")
            elif lb["probabilities"].get(new_key) != l06["probabilities"][old_key]:
                problems.append(f"{name}: own probability of {old_key!r} changed")
        extra = [k for k in b["criteria"] if k not in set(mapping.values())]
        if any(lb["probabilities"].get(k) != 0.0 for k in extra):
            problems.append(f"{name}: a distractor has probability above 0")
        if {k: v for k, v in lb.items() if k not in ("probabilities", "expected_cost")} != {k: v for k, v in l06.items() if k not in ("probabilities", "expected_cost")}:
            problems.append(f"{name}: a label field changed")
        if "expected_cost" in l06 and any(lb["expected_cost"].get(mapping[k]) != v for k, v in l06["expected_cost"].items()):
            problems.append(f"{name}: an own expected cost changed")
        if not N_RANGE[0] <= len(b["criteria"]) <= N_RANGE[1] or len(b["criteria"]) != e["to"]:
            problems.append(f"{name}: {len(b['criteria'])} options outside 9..255")
        own_norms = {_norm_option(k, v) for k, v in q06["criteria"].items()}
        drawn_norms = [_norm_option(k, b["criteria"][k]) for k in extra]
        if len(set(drawn_norms)) != len(drawn_norms) or own_norms & set(drawn_norms):
            problems.append(f"{name}: a distractor repeats an option's text")
    codes = new_meta.get("codes") or {}
    if sorted(codes) != sorted(event.get("codes") or []):
        problems.append("meta.codes differs from the transform's record")
    for name, value in codes.items():
        if any(c not in CODE_POOL for c in value) or len(set(value)) != len(value) or len(value) != len(nq[name]["criteria"]):
            problems.append(f"meta.codes {name!r} is not a draw of distinct pool codes, one per option")
    wide = any(q.get("type") == "choice" and len(q["criteria"]) > OPTION_RANGE[1] for q in nq.values())
    if wide != (new_meta.get("option_range") == OPTION_RANGE_MARK_255) or (not wide and "option_range" in new_meta):
        problems.append("meta.option_range does not match the option counts")
    problems += [f"validation: {p}" for p in validate_record(dict(new))]
    return problems


# --------------------------------------------------------------------------
# transform — the input, its manifest and change report
# --------------------------------------------------------------------------


class Report:
    """Counts of what the transform did: shares, N per bin and family, codes, re-keying, audit samples."""

    def __init__(self) -> None:
        self.c: Counter = Counter()
        self.n_hist: Counter = Counter()
        self.n_by: dict[str, Counter] = defaultdict(Counter)
        self.drawn_hist: Counter = Counter()
        self.short_by: Counter = Counter()
        self.codes_n: Counter = Counter()
        self.pool_seen: Counter = Counter()
        self.audit: dict[str, list[dict[str, Any]]] = {"physics": [], "general": []}
        self._rng = random.Random(f"{SCRIPT_VERSION}|audit")
        self._seen = Counter()

    def add(self, v06: Mapping[str, Any], v07: Mapping[str, Any], new: Mapping[str, Any] | None, event: Mapping[str, Any]) -> None:
        part = event["part"]
        self.c[f"records:{part}"] += 1
        if event.get("dev"):
            self.c[f"dev:{part}"] += 1
        self.c[f"choice_questions:{part}"] += event.get("choice_questions", 0)
        if new is None:
            return
        self.c[f"changed:{part}"] += 1
        for name in event.get("codes") or []:
            self.c[f"codes:{part}"] += 1
            n = len(new["request"]["questions"][name]["criteria"])
            self.codes_n[_bin(n)] += 1
            for code in new["meta"]["codes"][name]:
                self.pool_seen[code] += 1
        for name, e in (event.get("expanded") or {}).items():
            fam = e.get("family") or "general"
            self.c[f"expanded:{part}"] += 1
            self.n_hist[e["to"]] += 1
            self.drawn_hist[_bin(e["drawn"])] += 1
            self.n_by[f"{part}:{fam}"][_bin(e["to"])] += 1
            if e["short"]:
                self.short_by[f"{part}:{fam}"] += 1
            if e.get("rekeyed"):
                self.c["rekeyed:general"] += 1
            for src, k in (e.get("sources") or {}).items():
                self.c[f"distractor_source:{fam}<-{src}"] += k
            bucket = "general" if part == "general" else "physics"
            self._seen[bucket] += 1
            size = 40 if bucket == "general" else 100
            sample = {"id": new["id"], "question": name, "family": fam, "n": e["to"], "drawn": e["drawn"],
                      "instructions": v06["request"]["questions"][name]["instructions"],
                      "own": {k: [v, v06["labels"][name]["probabilities"][k]] for k, v in v06["request"]["questions"][name]["criteria"].items()},
                      "distractors_sample": [v for k, v in new["request"]["questions"][name]["criteria"].items()
                                             if k not in set((e.get("mapping") or {k2: k2 for k2 in v06["request"]["questions"][name]["criteria"]}).values())][:30]}
            if bucket == "general":
                sample["state"] = v06["request"]["state"]
                sample["mapping"] = e.get("mapping")
            target = self.audit[bucket]
            if len(target) < size:
                target.append(sample)
            else:
                j = self._rng.randrange(self._seen[bucket])
                if j < size:
                    target[j] = sample
        for name in event.get("kept_v07") or []:
            self.c[f"kept_v07:{part}"] += 1


def transform_file(v06_path: Path, v07_path: Path, out_path: Path, *, config: Mapping[str, Any]) -> dict[str, Any]:
    """Pass 1 (v0.7's plan over v0.6's records) and pass 2 (transform, check, write) → `out_path`, replaced only when the whole stream
    passed."""
    v05 = _script("assemble_v05.py")
    plan = Plan(config)
    _log(f"pass 1: {v06_path}")
    for raw in _raw_lines(v06_path):
        plan.scan(json.loads(raw))
    plan.finish()
    _log(f"pass 1 done: pools {plan.v07.pool.sizes()}, general pool {len(plan.v07.general_pool)}, expanded-question slots {len(plan.v07.questions)}")
    report = Report()
    digests = {side: defaultdict(hashlib.sha256) for side in ("before", "after")}
    untouched = hashlib.sha256()
    ids = {side: hashlib.sha256() for side in ("before", "after")}
    styles: Counter = Counter()
    failures: list[str] = []
    out_path.parent.mkdir(parents=True, exist_ok=True)
    temp = out_path.with_name(out_path.name + ".tmp")
    try:
        with temp.open("wb") as handle:
            for raw06, raw07 in zip(_raw_lines(v06_path), _raw_lines(v07_path), strict=True):
                v06, v07 = json.loads(raw06), json.loads(raw07)
                new, event = transform_record(v06, v07, plan)
                report.add(v06, v07, new, event)
                part = event["part"]
                ids["before"].update(str(v07["id"]).encode("utf-8") + b"\n")
                if new is None:
                    line = raw07
                    untouched.update(raw07)
                else:
                    problems = record_problems(v06, v07, new, event)
                    if problems:
                        failures.append(f"{v07['id']}: {problems[:3]}")
                        if len(failures) >= 20:
                            raise ValueError(f"records break the rule: {failures}")
                    style = v05.line_style(raw07, v07)
                    styles[style] += 1
                    line = v05.dumps_style(new, style)
                    if json.loads(line) != new:
                        raise ValueError(f"{v07['id']}: the written line does not read back as the transformed record")
                ids["after"].update(str(json.loads(line)["id"]).encode("utf-8") + b"\n")
                digests["before"][part].update(raw07)
                digests["after"][part].update(line)
                handle.write(line)
        if failures:
            raise ValueError(f"{len(failures)} records break the rule: {failures[:5]}")
        if ids["before"].hexdigest() != ids["after"].hexdigest():
            raise ValueError("the id order changed")
        temp.replace(out_path)
    finally:
        if temp.exists():
            temp.unlink()
    c = report.c
    choice = sum(c[f"choice_questions:{p}"] for p in ("engine", "mujoco", "general", "d1"))
    physics_choice = sum(c[f"choice_questions:{p}"] for p in PHYSICS)
    codes_total = sum(c[f"codes:{p}"] for p in ("engine", "mujoco", "general", "d1"))
    by_bin = Counter(_bin(n) for n, k in report.n_hist.items() for _ in range(k))
    shares = {
        "expanded_of_physics_choice_questions": sum(c[f"expanded:{p}"] for p in PHYSICS) / physics_choice,
        "expanded_of_general_records": c["expanded:general"] / c["records:general"],
        "random_codes_of_all_choice_questions": codes_total / choice,
        "random_codes_by_part": {p: (c[f"codes:{p}"] / c[f"choice_questions:{p}"]) if c[f"choice_questions:{p}"] else None
                                 for p in ("engine", "mujoco", "general", "d1")},
    }
    parts = {part: {"sha256_before": digests["before"][part].hexdigest(), "sha256_after": digests["after"][part].hexdigest(),
                    "identical": digests["before"][part].hexdigest() == digests["after"][part].hexdigest(), "records": c[f"records:{part}"],
                    "changed": c[f"changed:{part}"], "dev": c[f"dev:{part}"]} for part in sorted(digests["before"])}
    from robo_jev.decisions.codes import CODE_POOL
    from robo_jev.simgen import format_v08 as F
    from robo_jev.simgen import shape

    return {
        "records": sum(c[k] for k in c if k.startswith("records:")), "parts": parts,
        "digests": {"id_order": {"before": ids["before"].hexdigest(), "after": ids["after"].hexdigest(), "identical": True},
                    "untouched_lines": untouched.hexdigest()},
        "line_styles_of_changed_records": dict(styles),
        "plan": {"pools": plan.v07.pool.sizes(), "general_pool": len(plan.v07.general_pool),
                 "distinct_keys_by_question_family": {fam: len(plan.index.keys(fam)) for fam in ("act", "out", "dir", "cmp")}},
        "counts": dict(sorted(c.items())), "shares": shares,
        "many_options": {"n_histogram": {str(k): v for k, v in sorted(report.n_hist.items())}, "n_by_bin": dict(sorted(by_bin.items())),
                         "drawn_by_bin": dict(sorted(report.drawn_hist.items())),
                         "n_by_part_family_bin": {k: dict(sorted(v.items())) for k, v in sorted(report.n_by.items())},
                         "short_by_part_family": dict(sorted(report.short_by.items())),
                         "distractor_sources": {k.split(":", 1)[1]: v for k, v in sorted(c.items()) if k.startswith("distractor_source:")},
                         "kept_v07": {k.split(":", 1)[1]: v for k, v in c.items() if k.startswith("kept_v07:")},
                         "rule": {"n_range": list(F.N_RANGE), "n_draw": "log-uniform over [9, 256), floored", "allowed": shape.ALLOWED}},
        "random_codes": {"questions": codes_total, "share": codes_total / choice, "by_bin": dict(sorted(report.codes_n.items())),
                         "pool_codes_shown": len(report.pool_seen), "pool_size": len(CODE_POOL),
                         "shown_per_code": {"min": min(report.pool_seen.get(k, 0) for k in CODE_POOL), "max": max(report.pool_seen.values())}},
        "audit": report.audit,
    }


RULE_TEXT = ("PRE-DECLARED v0.8 VERDICT RULE (progress.md, 2026-10-16 ~12:43 KST): one training input, train-v08 = v0.7's input with the "
             "FORMAT package — choice questions take 2–255 options (N log-uniform, enough large-N examples), the correct position spread over "
             "every code, a share of choice questions with random codes from the 255-code pool; records transformed, never added; labels v0.7's")


def render_changes(report: Mapping[str, Any]) -> str:
    s, m, r = report["shares"], report["many_options"], report["random_codes"]
    lines = ["# v0.8 training input — the FORMAT package", "",
             f"`{report['out']['path']}` sha256 `{report['out']['sha256']}` ({report['records']:,} records) from `{report['input']['v07']}` "
             f"(sha256 `{report['input']['v07_sha256']}`), read in lockstep with v0.6's `{report['input']['v06']}`. {RULE_TEXT}.", "",
             "## Parts", "", "| part | records | dev | changed | identical |", "|---|---|---|---|---|"]
    for part, e in report["parts"].items():
        lines.append(f"| {part} | {e['records']:,} | {e['dev']:,} | {e['changed']:,} | {e['identical']} |")
    lines += ["", "## Shares", ""] + [f"* {k}: {v}" for k, v in s.items()]
    lines += ["", "## Many options", "", f"N by bin: {m['n_by_bin']} (drawn: {m['drawn_by_bin']})", "",
              f"N by part, family and bin: {m['n_by_part_family_bin']}", "", f"Short of the drawn N (the families ran out of keys): {m['short_by_part_family']}", "",
              f"Distractor sources: {m['distractor_sources']}", "", "## Random codes", "",
              f"{r['questions']:,} questions ({r['share']:.4f} of all choice questions); by bin {r['by_bin']}; pool codes shown {r['pool_codes_shown']} of "
              f"{r['pool_size']}; shown per code min {r['shown_per_code']['min']} max {r['shown_per_code']['max']}", ""]
    return "\n".join(lines) + "\n"


def cmd_transform(args: argparse.Namespace) -> int:
    refuse_write(args.out)
    sha06, sha07 = _sha256(args.v06), _sha256(args.input)
    if args.v06.resolve() == TRAIN_V06.resolve() and sha06 != TRAIN_V06_SHA:
        raise SystemExit(f"{args.v06}: sha256 {sha06[:12]}… is not v0.6's input")
    if args.input.resolve() == TRAIN_V07.resolve() and sha07 != TRAIN_V07_SHA:
        raise SystemExit(f"{args.input}: sha256 {sha07[:12]}… is not v0.7's input")
    config = _config(args.train_config)
    _log(f"transform: {args.input} ({sha07[:12]}…) + {args.v06} ({sha06[:12]}…) → {args.out}")
    try:
        report = transform_file(args.v06, args.input, args.out, config=config)
    except ValueError as error:
        raise SystemExit(f"refusing: {error}") from error
    out_sha = _sha256(args.out)
    from robo_jev.decisions.records import OPTION_RANGE_MARK_255

    head = {"kind": "v08-train-changes", "script_version": SCRIPT_VERSION, "rule": RULE_TEXT, "git": _git(),
            "input": {"v07": str(args.input), "v07_sha256": sha07, "v06": str(args.v06), "v06_sha256": sha06},
            "out": {"path": str(args.out), "sha256": out_sha, "bytes": args.out.stat().st_size},
            "train_config": {"path": str(args.train_config), **{k: config[k] for k in ("dev_fraction", "dev_salt", "questions_per_request")}}}
    audit = report.pop("audit")
    full = {**head, **report, "created": dt.datetime.now().astimezone().isoformat(timespec="seconds")}
    stem = args.out.stem
    _write_json(args.out.with_name(f"{stem}.changes.json"), full)
    args.out.with_name(f"{stem}.changes.md").write_text(render_changes(full), encoding="utf-8")
    _write_json(args.out.with_name(f"{stem}.audit.json"), {"kind": "v08-distractor-audit-sample", "what": "reservoir samples of expanded questions "
                "(physics 100, general 40): instructions, own options with label probabilities, up to 30 distractors (general: the state and the key map)", **audit})
    manifest = {"kind": "v08-train", "script_version": SCRIPT_VERSION, "git": _git(), "sha256": out_sha,
                "out": {"path": str(args.out), "sha256": out_sha, "bytes": args.out.stat().st_size, "records": report["records"]},
                "input": {"v07": {"path": str(args.input), "sha256": sha07}, "v06": {"path": str(args.v06), "sha256": sha06}},
                "rule": RULE_TEXT, "shares": report["shares"], "option_range_mark": OPTION_RANGE_MARK_255, "parts": report["parts"],
                "changes": str(args.out.with_name(f"{stem}.changes.json")), "created": dt.datetime.now().astimezone().isoformat(timespec="seconds")}
    _write_json(args.out.with_name(f"{stem}.manifest.json"), manifest)
    _log(f"transform: {report['records']} records; shares {report['shares']} → {args.out} {out_sha[:16]}…")
    return 0


# --------------------------------------------------------------------------
# overlap — the sealed checks
# --------------------------------------------------------------------------

B1 = ARTIFACTS / "b1" / "b1-infamily.jsonl"
HSET = ARTIFACTS / "simgen" / "v04" / "scene-holdout.jsonl"
A4 = ARTIFACTS / "v06" / "a4" / "test.jsonl"
IFACE = {"M1": ARTIFACTS / "iface" / "m1" / "m1-set.jsonl", "M2-short": ARTIFACTS / "iface" / "m2" / "M2-short.jsonl",
         "M2-w2k": ARTIFACTS / "iface" / "m2" / "M2-w2k.jsonl", "M2-w4k": ARTIFACTS / "iface" / "m2" / "M2-w4k.jsonl",
         "M2-w8k": ARTIFACTS / "iface" / "m2" / "M2-w8k.jsonl"}
GATES_V07 = ARTIFACTS / "v07" / "gates"
V07_GATES = {"M2g": GATES_V07 / "m2g" / "M2g-all-conditions.jsonl", "M1g": GATES_V07 / "m1g" / "M1g.jsonl", "M2g-F": GATES_V07 / "m2gf" / "M2g-facts.jsonl"}
MUJOCO_SETS = ("v03b-mujoco-train", "v06-a1", "v06-a2", "v06-a3")


def _hash16(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _texts(record: Mapping[str, Any]) -> list[str]:
    """Every text a record shows the model: the state, each instruction, each option key and description, each score criterion."""
    request = record.get("request") or {}
    out = [str(request.get("state") or "")]
    for question in (request.get("questions") or {}).values():
        out.append(str(question.get("instructions") or ""))
        criteria = question.get("criteria")
        if isinstance(criteria, dict):
            out += [str(k) for k in criteria] + [str(v) for v in criteria.values()]
        elif isinstance(criteria, list):
            out += [str(v) for v in criteria]
    return out


def text_subset(new_path: Path, old_path: Path) -> dict[str, Any]:
    """Every distinct text of the new input (state, instructions, option keys and descriptions) is a text of the old one — the format
    package moves texts around and adds none, so every text-level sealed check of the old input carries over (hashes only). The one kind
    of string the package does make is a re-keyed general item's option key ``o<k>`` with k up to 255 (v0.7's went to ``o26``): such keys
    are counted apart and allowed."""
    import re

    rekey = re.compile(r"o\d{1,3}")
    old: set[str] = set()
    for raw in _raw_lines(old_path):
        old.update(_hash16(t) for t in _texts(json.loads(raw)))
    new: set[str] = set()
    new_keys: set[str] = set()
    for raw in _raw_lines(new_path):
        for t in _texts(json.loads(raw)):
            h = _hash16(t)
            if h not in old:
                (new_keys if rekey.fullmatch(t) else new).add(t if rekey.fullmatch(t) else h)
    return {"distinct_new_texts_not_in_old": len(new), "new_option_keys_o_k": sorted(new_keys, key=lambda k: int(k[1:])),
            "subset": not new}


def outside_units(paths: Mapping[str, Path], template_paths: Sequence[Path]) -> dict[str, set[str]]:
    """Per set, the hashes of its text units (:func:`robo_jev.data.general_sets.record_units`) that are not B1 / H-set units — the text
    written outside our generators (as v0.7a's iface check). Hashes only."""
    from robo_jev.data.general_sets import record_units

    template = {_hash16(u) for path in template_paths for raw in _raw_lines(path) for u in record_units(json.loads(raw))}
    return {name: {_hash16(u) for raw in _raw_lines(path) for u in record_units(json.loads(raw))} - template for name, path in paths.items()}


def gate_scans(train: Path) -> dict[str, Any]:
    """v0.7b's M1g / M2g scan (units and 8-word shingles of the MMLU candidates, whole states of the physics and M2g moments, the M2g filler's
    shingles) and v0.7f's M2g-F scan (its fixed template and status pieces, the shingles touching its added text) against `train` —
    counts and hashes only, nothing of the gates printed or written."""
    from robo_jev.bench import gates_v07 as G
    from robo_jev.bench import gates_v07f as GF
    from robo_jev.decisions.records import read_records

    gb = _script("v07_gates_build.py")
    m1_items = read_records(GATES_V07 / "m1g" / "m1g-items.jsonl")
    m2_short = read_records(GATES_V07 / "m2g" / "M2g-short.jsonl")
    m2_all = read_records(GATES_V07 / "m2g" / "M2g-all-conditions.jsonl")
    general = [{"item": i["item"], "state": i["state"], "stem": i["stem"], "correct": i["correct"], "own": list(i["own"]) + [str(d) for d in i["drawn"]]}
               for i in m1_items if i["source"] == G.GENERAL_SOURCE]
    states = {f"m2g:{r['id']}": r["request"]["state"] for r in m2_short}
    states.update({i["item"]: i["state"] for i in m1_items if i["source"] != G.GENERAL_SOURCE})
    units, grams = gb.candidates(general, states)
    lines = {line for r in m2_all for line in r["request"]["state"].splitlines() if line.strip()}
    short_lines = {line for r in m2_short for line in r["request"]["state"].splitlines() if line.strip()}
    found = gb.scan([train], units, grams, gb.filler_shingles(lines - short_lines))
    facts = read_records(V07_GATES["M2g-F"])
    by_id = {str(r["id"]): r for r in m2_short}
    pieces = [piece["text"] for piece in GF.fixed_pieces() if piece["words"] >= 2]
    fgrams: set[str] = set()
    for r in facts:
        fgrams |= GF.touching_shingles(r, by_id[str(r["id"])])
    controls = [c for c in gb.CONTROL_PIECES if c not in pieces]
    fscan = gb.facts_scan([train], pieces + controls, fgrams)
    piece_hits = {p: v for p, v in fscan["pieces"].items() if p in pieces and v["exact"]}
    return {"m1g_m2g": {"hits": len(found["hits"]), "files": found["files"], "filler_shingles": found["filler_shingles"],
                        "filler_shingles_found": len(found["filler_shingles_found"]),
                        "exclude_if_any": {"M1g_items": len([k for k in found["hits"] if not k.startswith("m2g:")]),
                                           "M2g_moments": len([k for k in found["hits"] if k.startswith("m2g:")])}},
            "m2g_facts": {"pieces": len(pieces), "pieces_with_exact_hits": len(piece_hits), "shingles": len(fgrams), "shingles_found": len(fscan["gram_hits"]),
                          "controls_found": {c: fscan["pieces"][c]["exact"] for c in controls}, "files": fscan["files"]}}


def cmd_overlap(args: argparse.Namespace) -> int:
    from robo_jev.data.general_sets import record_units
    from robo_jev.simgen.holdout import request_hash
    from robo_jev.simgen.narration import state_sha256

    out_dir = args.out_dir
    refuse_write(out_dir)
    sealed_files = (B1, HSET, A4, *IFACE.values(), *V07_GATES.values())
    missing = [str(p) for p in sealed_files if not p.is_file()]
    if missing:
        raise SystemExit(f"sealed files missing: {missing} — every sealed set is checked")
    stem = args.input.stem
    sets = sum((["--mujoco-set", s] for s in MUJOCO_SETS), [])
    extra = sum((["--sealed-extra", f"{name}={path}"] for name, path in {**IFACE, **V07_GATES}.items()), [])
    rc = subprocess.run([sys.executable, "scripts/system1_v03.py", "overlap", "--input", str(args.input), "--out", str(out_dir / f"{stem}.overlap.json"),
                         "--sealed-extra", f"B1={B1}", "--sealed-extra", f"H-set={HSET}", "--sealed-outside", f"A4={A4}", *extra, *sets], cwd=REPO).returncode
    _log("outside units of iface M1 / M2 and the v0.7 gates (hashes only)")
    outside = outside_units({**IFACE, **V07_GATES}, (B1, HSET))
    hits: Counter = Counter()
    by_part: dict[str, Counter] = defaultdict(Counter)
    index = {name: {"ids": set(), "requests": set(), "states": set()} for name in {**IFACE, **V07_GATES}}
    for name, path in {**IFACE, **V07_GATES}.items():
        for raw in _raw_lines(path):
            r = json.loads(raw)
            index[name]["ids"].add(str(r["id"]))
            index[name]["requests"].add(request_hash(r))
            index[name]["states"].add(state_sha256(r["request"]["state"]))
    for raw in _raw_lines(args.input):
        record = json.loads(raw)
        part = part_of(record)
        changed = "format" in (record.get("meta") or {})
        units = {_hash16(u) for u in record_units(record)}
        for name, hashes in outside.items():
            if units & hashes:
                hits[name] += 1
                by_part[f"{part}:{'changed' if changed else 'untouched'}"][name] += 1
        for name, entry in index.items():
            if str(record["id"]) in entry["ids"] or request_hash(record) in entry["requests"] or state_sha256(record["request"]["state"]) in entry["states"]:
                hits[f"{name}:id_request_state"] += 1
    _log("v0.7b's M1g / M2g scan and v0.7f's M2g-F scan")
    scans = gate_scans(args.input)
    _log("text subset: every text of the input is a text of train-v07")
    subset = text_subset(args.input, args.reference)
    report = {"kind": "v08-sealed-extra", "script_version": SCRIPT_VERSION, "git": _git(), "input": {"path": str(args.input), "sha256": _sha256(args.input)},
              "what": "text units of the iface M1 / M2 sets and the v0.7 gates M2g, M1g, M2g-F that are not B1 / H-set template units, matched by hash "
                      "against every record; ids, whole requests and whole states too; v0.7b's and v0.7f's own scans of the gates against this "
                      "input; and the text-subset proof against train-v07. Hashes and counts only — no text read out.",
              "outside_units": {name: len(v) for name, v in outside.items()}, "records_with_hits": dict(hits), "by_part": {k: dict(v) for k, v in by_part.items()},
              "gate_scans": scans, "text_subset": {**subset, "reference": {"path": str(args.reference), "sha256": _sha256(args.reference)}},
              "total_hits": sum(hits.values()) + scans["m1g_m2g"]["hits"] + scans["m2g_facts"]["pieces_with_exact_hits"] + scans["m2g_facts"]["shingles_found"],
              "system1_overlap_rc": rc, "created": dt.datetime.now().astimezone().isoformat(timespec="seconds")}
    _write_json(out_dir / f"{stem}.sealed-extra.json", report)
    overlap = json.loads((out_dir / f"{stem}.overlap.json").read_text(encoding="utf-8"))
    controls_ok = all(scans["m2g_facts"]["controls_found"].values())
    _log(f"overlap: system1_v03 hits {overlap.get('total_hits')} (rc {rc}) · outside-unit / gate hits {report['total_hits']} {dict(hits)} · "
         f"text subset {subset['subset']} · facts-scan controls found {controls_ok}")
    return 0 if rc == 0 and report["total_hits"] == 0 and subset["subset"] and controls_ok else 2


# --------------------------------------------------------------------------
# exposure — the trainer's view, N, positions and codes, tokens, time
# --------------------------------------------------------------------------


def _shown_codes(record: Mapping[str, Any], name: str, n: int) -> list[str]:
    from robo_jev.decisions.codes import CODE_POOL, assigned_codes

    return assigned_codes(record).get(name) or list(CODE_POOL[:n])


def format_exposure(records: Sequence[Mapping[str, Any]], *, config: Mapping[str, Any], seeds: Sequence[int]) -> dict[str, Any]:
    """Per seed, the trainer's epoch-0 choice examples: N per bin (all, expanded), the correct option's (argmax label) shown position as a
    fraction of N and its code — per pool code, for all choice examples and for those with random codes — and the random-code share."""
    from robo_jev.decisions.codes import CODE_POOL, assigned_codes
    from robo_jev.decisions.examples import epoch_plan, split_dev
    from robo_jev.decisions.prompt import option_order
    from robo_jev.decisions.prior_weights import best_key
    from robo_jev.decisions.records import shuffle_key

    q = int(config["questions_per_request"])
    train, _ = split_dev(list(records), fraction=config["dev_fraction"], salt=config["dev_salt"])
    out: dict[str, Any] = {}
    for seed in seeds:
        tally: Counter = Counter()
        n_bins: Counter = Counter()
        n_bins_many: Counter = Counter()
        code_all: Counter = Counter()
        code_random: Counter = Counter()
        code_by_bin: dict[str, Counter] = defaultdict(Counter)
        decile: dict[str, Counter] = defaultdict(Counter)
        for index, name, order_seed in epoch_plan(train, seed=int(seed), epoch=0, q=q, augment=bool(config.get("augment", True))):
            record = train[index]
            question = record["request"]["questions"][name]
            tally["examples"] += 1
            if question.get("type") != "choice":
                continue
            tally["choice"] += 1
            keys = sorted(question["criteria"])
            n = len(keys)
            shown = [keys[i] for i in option_order(shuffle_key(record), name, n, order_seed)]
            position = shown.index(best_key(record["labels"][name]["probabilities"]))
            assigned = assigned_codes(record).get(name)
            code = (assigned or CODE_POOL)[position]
            b = _bin(n)
            n_bins[b] += 1
            if name in (((record.get("meta") or {}).get("format") or {}).get("expanded") or {}):
                n_bins_many[b] += 1
            code_all[code] += 1
            code_by_bin[b][len(code)] += 1
            decile[b][min(9, int(10 * position / n))] += 1
            if assigned:
                tally["choice_random_codes"] += 1
                code_random[code] += 1
        out[str(seed)] = {
            "examples": tally["examples"], "choice_examples": tally["choice"], "random_code_examples": tally["choice_random_codes"],
            "random_code_share": tally["choice_random_codes"] / tally["choice"], "n_by_bin": dict(sorted(n_bins.items())),
            "expanded_n_by_bin": dict(sorted(n_bins_many.items())),
            "correct_code_letters_vs_two_letter_by_bin": {b: {"one_letter": v.get(1, 0), "two_letter": v.get(2, 0)} for b, v in sorted(code_by_bin.items())},
            "correct_position_decile_by_bin": {b: [v.get(i, 0) for i in range(10)] for b, v in sorted(decile.items())},
            "correct_per_code_all": {"min": min(code_all.get(c, 0) for c in CODE_POOL), "max": max(code_all.get(c, 0) for c in CODE_POOL),
                                     "codes_never_correct": [c for c in CODE_POOL if not code_all.get(c)],
                                     "min_two_letter": min(code_all.get(c, 0) for c in CODE_POOL[26:]),
                                     "median_two_letter": statistics.median(code_all.get(c, 0) for c in CODE_POOL[26:]),
                                     "counts": {c: code_all.get(c, 0) for c in CODE_POOL}},
            "correct_per_code_random": {"min": min(code_random.get(c, 0) for c in CODE_POOL), "max": max(code_random.get(c, 0) for c in CODE_POOL),
                                        "codes_never_correct": [c for c in CODE_POOL if not code_random.get(c)]},
        }
    return out


def token_simulation(records: Sequence[Mapping[str, Any]], *, config: Mapping[str, Any], seeds: Sequence[int], tokenizer: Any,
                     exact_sample: int = 400) -> dict[str, Any]:
    """Per seed, the epoch-0 steps exactly as the trainer forms them (v0.7a's simulation, with each question's assigned codes): every
    example's prompt length (the record's prefix tokens + the question block's tokens — order-independent: each option line tokenizes on its
    own, which a sample of exact :func:`robo_jev.decisions.examples.build_examples` prompts checks), each step's micro-batches by the trainer's
    :func:`robo_jev.decisions.train.pack_micro_batches`, padded tokens; the longest example; and the dev prompts' tokens."""
    from robo_jev.decisions.codes import assigned_codes
    from robo_jev.decisions.examples import build_examples, dev_examples, epoch_plan, split_dev
    from robo_jev.decisions.prompt import _prefix as _prompt_prefix
    from robo_jev.decisions.prompt import build_prompts
    from robo_jev.decisions.records import shuffle_key
    from robo_jev.decisions.train import pack_micro_batches

    q = int(config["questions_per_request"])
    fmt, version = config["format"], config["prompt_version"]
    batch, micro = int(config["batch_examples"]), int(config["micro_batch_tokens"])
    train, dev = split_dev(list(records), fraction=config["dev_fraction"], salt=config["dev_salt"])
    dev_used = dev[: config["dev_max_records"]] if config.get("dev_max_records") else dev
    lengths: dict[tuple[int, str], int] = {}
    prefixes: dict[int, int] = {}

    def length(index: int, name: str) -> int:
        cached = lengths.get((index, name))
        if cached is None:
            record = train[index]
            if index not in prefixes:
                prefixes[index] = len(tokenizer.encode(_prompt_prefix(record["request"]["state"], fmt), add_special_tokens=False).ids)
            sub = {"state": "x", "questions": {name: record["request"]["questions"][name]}}
            codes = assigned_codes(record).get(name)
            (prompt,) = build_prompts(sub, tokenizer, key=shuffle_key(record), seed=1, fmt=fmt, mode="direct", orders="seeded", version=version,
                                      codes={name: codes} if codes else None)
            cached = lengths[(index, name)] = prefixes[index] + len(prompt["tokens"]) - prompt["prefix_tokens"]
        return cached

    out: dict[str, Any] = {}
    for seed in seeds:
        plan = epoch_plan(train, seed=int(seed), epoch=0, q=q, augment=True)
        tokens = padded = steps = micro_batches = 0
        longest = 0
        by_bin_tokens: Counter = Counter()
        by_bin_count: Counter = Counter()
        for start in range(0, len(plan), batch):
            chunk = plan[start:start + batch]
            ls = [length(i, n) for i, n, _ in chunk]
            groups = pack_micro_batches(ls, micro)
            tokens += sum(ls)
            padded += sum(len(g) * max(ls[i] for i in g) for g in groups)
            micro_batches += len(groups)
            steps += 1
            longest = max(longest, max(ls))
            for (i, n, _), length_i in zip(chunk, ls):
                question = train[i]["request"]["questions"][n]
                b = _bin(len(question["criteria"])) if question.get("type") == "choice" else "noul/score"
                by_bin_tokens[b] += length_i
                by_bin_count[b] += 1
        rnd = random.Random(f"{SCRIPT_VERSION}|exact|{seed}")
        mismatches = 0
        wide = [entry for entry in plan if len(train[entry[0]]["request"]["questions"][entry[1]].get("criteria") or {}) > 8
                or assigned_codes(train[entry[0]]).get(entry[1])]
        sample = rnd.sample(plan, min(exact_sample, len(plan))) + rnd.sample(wide, min(exact_sample, len(wide)))
        for i, n, order_seed in sample:
            (example,) = build_examples(train[i], [n], tokenizer, order_seed=order_seed, fmt=fmt, version=version)
            mismatches += int(len(example["tokens"]) != length(i, n))
        out[str(seed)] = {"steps": steps, "examples": len(plan), "tokens": tokens, "padded_tokens": padded, "micro_batches": micro_batches,
                          "longest_example": longest, "mean_tokens_by_bin": {b: by_bin_tokens[b] / by_bin_count[b] for b in sorted(by_bin_count)},
                          "examples_by_bin": dict(sorted(by_bin_count.items())),
                          "exact_check": {"sample": len(sample), "length_mismatches": mismatches,
                                          "what": "a uniform sample of the plan plus a sample of its examples with more than 8 options or assigned codes"}}
    dev_prompts = dev_examples(dev_used, tokenizer, fmt=fmt, version=version)
    out["dev"] = {"records": len(dev_used), "prompts": len(dev_prompts), "tokens": sum(len(p["tokens"]) for p in dev_prompts)}
    return out


def time_and_budget(sim: Mapping[str, Any], *, input_bytes: int, v07_bytes: int = 837650913) -> dict[str, Any]:
    """Per seed: v0.7's measured seconds per padded token × this input's padded tokens (+ v0.7's measured dev and loading seconds — the dev
    records are the same — and a 10 % margin for the longer sequences' attention) → the train step; the pod = v0.7's measured pod minus its
    train step and upload, plus this train step and this upload (v0.7's measured upload rate) → hours and $ at :data:`HOURLY_USD`."""
    out: dict[str, Any] = {}
    for seed, m in V07_MEASURED.items():
        if seed not in sim:
            continue
        rate = m["step_seconds"] / m["padded_tokens"]
        steps_seconds = rate * sim[seed]["padded_tokens"]
        other = m["train_step_seconds"] - m["step_seconds"] - m["dev_seconds"]
        train_seconds = steps_seconds + m["dev_seconds"] + other
        with_margin = train_seconds * 1.10
        upload_bytes = m["upload_bytes"] - v07_bytes + int(input_bytes)
        upload_seconds = m["upload_seconds"] * upload_bytes / m["upload_bytes"]
        pod_seconds = m["pod_seconds"] - m["train_step_seconds"] - m["upload_seconds"] + with_margin + upload_seconds
        out[seed] = {"seconds_per_padded_token_v07": rate, "padded_tokens_ratio_vs_v07_measured": sim[seed]["padded_tokens"] / m["padded_tokens"],
                     "tokens_ratio_vs_v07_measured": sim[seed]["tokens"] / m["tokens"], "step_minutes": steps_seconds / 60,
                     "train_minutes": train_seconds / 60, "train_minutes_with_margin": with_margin / 60, "upload_minutes": upload_seconds / 60,
                     "pod_hours": pod_seconds / 3600, "pod_usd": pod_seconds / 3600 * HOURLY_USD, "within_run_cap": pod_seconds / 3600 * HOURLY_USD <= RUN_CAP_USD,
                     "v07_pod_hours": m["pod_seconds"] / 3600, "v07_pod_usd": m["pod_seconds"] / 3600 * HOURLY_USD}
    return out


def cmd_exposure(args: argparse.Namespace) -> int:
    from robo_jev.decisions.train import load_training_records

    v04 = _script("assemble_v04.py")
    config = _config(args.train_config)
    stored = json.loads(args.stored.read_text(encoding="utf-8"))
    tokenizer = None
    if not args.no_tokens:
        from robo_jev.model.tokenizer import available_tokenizer, load_tokenizer

        tokenizer = load_tokenizer(available_tokenizer()[0])
    records, validation = load_training_records(args.input)
    if validation["invalid"]:
        raise SystemExit(f"{args.input}: {validation['invalid']} records fail the trainer's validation ({validation['problems']})")
    entry = v04.exposure_report(records, config=config, seeds=args.seeds, mujoco_sets=set(MUJOCO_SETS))
    entry["input"] = {"path": str(args.input), "sha256": _sha256(args.input), "validation": validation}
    fields = ("examples_per_epoch", "steps", "engine_by_category", "action", "derived", "band", "comparison")
    v07 = stored["v07"]
    same = {seed: {f: entry["seeds"][seed][f] == v07["seeds"][seed][f] for f in fields} for seed in map(str, args.seeds)}
    split_same = {k: entry[k] == v07[k] for k in ("records", "train_records", "dev_records", "dev_used", "dev_parts", "train_by_part")}
    _log("format exposure")
    fmt = format_exposure(records, config=config, seeds=args.seeds)
    out: dict[str, Any] = {"kind": "v08-exposure", "script_version": SCRIPT_VERSION, "git": _git(), "train_config": str(args.train_config),
                           "v08": entry, "identical_to_v07_stored": same, "split_identical": split_same, "format": fmt}
    if tokenizer is not None:
        _log("tokens")
        sim = token_simulation(records, config=config, seeds=args.seeds, tokenizer=tokenizer)
        v07_sim = stored["tokens"]["v07"]
        out["tokens"] = {"v08": sim, "v07": v07_sim,
                         "growth_vs_v07": {seed: {"tokens": sim[seed]["tokens"] / v07_sim[seed]["tokens"],
                                                  "padded_tokens": sim[seed]["padded_tokens"] / v07_sim[seed]["padded_tokens"]} for seed in map(str, args.seeds)},
                         "dev_identical_to_v07": sim["dev"] == v07_sim["dev"]}
        out["time_and_budget"] = time_and_budget(sim, input_bytes=args.input.stat().st_size)
    out["pass"] = all(all(v.values()) for v in same.values()) and all(split_same.values()) and not any(
        e["exact_check"]["length_mismatches"] for k, e in (out.get("tokens") or {}).get("v08", {}).items() if k != "dev")
    out["created"] = dt.datetime.now().astimezone().isoformat(timespec="seconds")
    _write_json(args.out, out)
    _log(f"exposure: identical to v0.7 per seed {same} · split {split_same} · pass {out['pass']}")
    if tokenizer is not None:
        _log(f"tokens growth {out['tokens']['growth_vs_v07']} · time/budget {json.dumps(out['time_and_budget'])[:800]}")
    return 0 if out["pass"] else 2


# --------------------------------------------------------------------------
# weights — the H-prior table (the FP arm)
# --------------------------------------------------------------------------


def cmd_weights(args: argparse.Namespace) -> int:
    """The H-prior class table and weight statistics per family over each seed's epoch-0 plan (the trainer's own computation,
    :func:`robo_jev.decisions.prior_weights.plan_weights`), the class base rates per family over the training records' questions, and the
    action-kind rule's check on every action option (text rule vs structure)."""
    from robo_jev.decisions import prior_weights as W
    from robo_jev.decisions.examples import epoch_plan, split_dev
    from robo_jev.decisions.train import load_training_records

    refuse_write(args.out)
    config = _config(args.train_config)
    records, _ = load_training_records(args.input)
    train, _ = split_dev(records, fraction=config["dev_fraction"], salt=config["dev_salt"])
    seeds: dict[str, Any] = {}
    for seed in args.seeds:
        plan = epoch_plan(train, seed=int(seed), epoch=0, q=int(config["questions_per_request"]), augment=True)
        weights, table = W.plan_weights(train, plan, cap=args.cap)
        families = table["families"]
        ess_total = sum(f["ess"] for f in families.values())
        by_type: dict[str, list[float]] = defaultdict(list)
        for (index, name, _), w in zip(plan, weights):
            by_type[train[index]["request"]["questions"][name]["type"]].append(w)
        seeds[str(seed)] = {**table, "summary": {"examples": table["examples"], "families": table["n_families"], "capped_classes": table["capped_classes"],
                                                 "weight_min": min(weights), "weight_max": max(weights), "weight_mean": statistics.fmean(weights),
                                                 "ess_share_overall": (sum(weights) ** 2 / sum(w * w for w in weights)) / len(weights),
                                                 "ess_share_sum_of_families": ess_total / table["examples"],
                                                 "examples_at_cap": sum(1 for w in weights if math.isclose(w, args.cap)),
                                                 "mean_weight_by_type": {k: statistics.fmean(v) for k, v in sorted(by_type.items())}}}
    static: dict[str, Counter] = defaultdict(Counter)
    kinds: Counter = Counter()
    for record in train:
        for name in record["request"]["questions"]:
            family = W.family_of(record, name)
            cls = W.label_class(record, name)
            static[family][cls] += 1
            if W.is_action_question(record, name):
                for key, description in record["request"]["questions"][name]["criteria"].items():
                    if (record["labels"][name]["probabilities"].get(key) or 0.0) <= 0.0 and key not in ((record.get("meta") or {}).get("options") or {}).get(name, {}) \
                            and key not in (((record.get("meta") or {}).get("plan") or {}).get("options") or {}):
                        continue  # a distractor (no structure): not an action option
                    kinds[(W.source_of(record).split(":")[0], (record.get("meta") or {}).get("language"), W.action_kind(record, name, key),
                           W.text_action_kind(key, description))] += 1
    groups = {"yes/no + score": lambda f, c: c in ("low", "middle", "high"), "level": lambda f, c: c.startswith("level:"),
              "action": lambda f, c: c.startswith("action:")}
    base_rates = {}
    for label, test in groups.items():
        for source in sorted({f.split("|")[0] for f in static}):
            tally = Counter()
            for family, counts in static.items():
                if family.split("|")[0] != source:
                    continue
                for cls, n in counts.items():
                    if test(family, cls):
                        tally[cls.split(":")[-1]] += n
            if tally:
                total = sum(tally.values())
                base_rates[f"{label} · {source}"] = {k: round(v / total, 4) for k, v in sorted(tally.items())} | {"n": total}
    agree = sum(v for k, v in kinds.items() if k[2] == k[3])
    payload = {"kind": "v08-prior-weights", "script_version": SCRIPT_VERSION, "git": _git(), "input": {"path": str(args.input), "sha256": _sha256(args.input)},
               "rule": W.__doc__.split("\n\n")[0], "cap": args.cap, "version": W.PRIOR_WEIGHTS_VERSION, "seeds": seeds,
               "class_base_rates_by_source": base_rates,
               "static_class_table": {f: dict(sorted(c.items())) for f, c in sorted(static.items())},
               "action_kind_check": {"what": "every own action option of the training records: the structured kind (engine canonical name, MuJoCo plan) vs "
                                             "the text rule alone", "agreement": agree / max(1, sum(kinds.values())), "options": sum(kinds.values()),
                                     "table": {"|".join(map(str, k)): v for k, v in sorted(kinds.items(), key=str)}},
               "created": dt.datetime.now().astimezone().isoformat(timespec="seconds")}
    _write_json(args.out, payload)
    _log(f"weights: cap {args.cap}; " + "; ".join(f"seed {s}: {e['summary']}" for s, e in seeds.items()))
    return 0


# --------------------------------------------------------------------------
# all
# --------------------------------------------------------------------------


def _run(argv: list[str]) -> int:
    _log("$ " + " ".join(argv))
    return subprocess.run([sys.executable, *argv], cwd=REPO).returncode


def cmd_all(args: argparse.Namespace) -> int:
    out = args.out_dir
    refuse_write(out)
    out.mkdir(parents=True, exist_ok=True)
    train = out / "train-v08.jsonl"
    me = str(Path(__file__).resolve().relative_to(REPO))
    if _run([me, "transform", "--input", str(args.input), "--v06", str(args.v06), "--out", str(train), "--train-config", str(args.train_config)]) != 0:
        raise SystemExit("transform failed")
    rc = {"overlap": _run([me, "overlap", "--input", str(train), "--reference", str(args.input), "--out-dir", str(out)]),
          "exposure": _run([me, "exposure", "--input", str(train), "--train-config", str(args.train_config), "--out", str(out / "train-v08.exposure.json")]),
          "weights": _run([me, "weights", "--input", str(train), "--train-config", str(args.train_config), "--out", str(out / "train-v08.prior-weights.json")])}
    load = lambda name: json.loads((out / name).read_text(encoding="utf-8")) if (out / name).is_file() else {}  # noqa: E731
    changes, overlap, extra, exposure = (load(n) for n in ("train-v08.changes.json", "train-v08.overlap.json", "train-v08.sealed-extra.json",
                                                           "train-v08.exposure.json"))
    summary = {"kind": "v08-assembly", "script_version": SCRIPT_VERSION, "git": _git(), "returncodes": rc,
               "files": {"train-v08.jsonl": {"path": str(train), "sha256": _sha256(train)}, "input": {"path": str(args.input), "sha256": _sha256(args.input)}},
               "shares": changes.get("shares"), "n_by_bin": (changes.get("many_options") or {}).get("n_by_bin"),
               "parts_identical": {p: e["identical"] for p, e in (changes.get("parts") or {}).items()},
               "overlap_hits": overlap.get("total_hits"), "sealed_checked": sorted(overlap.get("sealed") or {}), "extra_hits": extra.get("total_hits"),
               "text_subset": (extra.get("text_subset") or {}).get("subset"), "exposure_identical_to_v07": exposure.get("pass"),
               "tokens_growth": (exposure.get("tokens") or {}).get("growth_vs_v07"), "time_and_budget": exposure.get("time_and_budget"),
               "pass": all(v == 0 for v in rc.values()) and overlap.get("total_hits") == 0 and extra.get("total_hits") == 0 and bool(exposure.get("pass")),
               "created": dt.datetime.now().astimezone().isoformat(timespec="seconds")}  # fmt: skip
    _write_json(out / "train-v08.assembly.json", summary)
    _log(f"done: {train} sha256 {summary['files']['train-v08.jsonl']['sha256'][:16]}… · pass {summary['pass']}")
    return 0 if summary["pass"] else 2


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    tr = sub.add_parser("transform", help="train-v07 (+ train-v06 in lockstep) → train-v08 (+ manifest, change report, audit sample)")
    tr.add_argument("--input", type=Path, default=TRAIN_V07)
    tr.add_argument("--v06", type=Path, default=TRAIN_V06)
    tr.add_argument("--out", type=Path, required=True)
    tr.add_argument("--train-config", type=Path, default=TRAIN_CONFIG)
    tr.set_defaults(func=cmd_transform)
    ov = sub.add_parser("overlap", help="sealed checks (system1_v03 overlap, outside units, the v0.7 gates' scans, the text-subset proof)")
    ov.add_argument("--input", type=Path, required=True)
    ov.add_argument("--reference", type=Path, default=TRAIN_V07, help="the input whose texts every text of --input must be (train-v07)")
    ov.add_argument("--out-dir", type=Path, required=True)
    ov.set_defaults(func=cmd_overlap)
    ex = sub.add_parser("exposure", help="the trainer's view per seed vs v0.7, N, positions and codes, tokens, time and budget")
    ex.add_argument("--input", type=Path, required=True)
    ex.add_argument("--stored", type=Path, default=EXPOSURE_V07)
    ex.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    ex.add_argument("--train-config", type=Path, default=TRAIN_CONFIG)
    ex.add_argument("--no-tokens", action="store_true", help="skip the token simulation")
    ex.add_argument("--out", type=Path, required=True)
    ex.set_defaults(func=cmd_exposure)
    we = sub.add_parser("weights", help="the H-prior class table and weight statistics per family (the FP arm)")
    we.add_argument("--input", type=Path, required=True)
    we.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    we.add_argument("--cap", type=float, default=5.0)
    we.add_argument("--train-config", type=Path, default=TRAIN_CONFIG)
    we.add_argument("--out", type=Path, required=True)
    we.set_defaults(func=cmd_weights)
    every = sub.add_parser("all", help="transform → overlap → exposure → weights (one command)")
    every.add_argument("--input", type=Path, default=TRAIN_V07)
    every.add_argument("--v06", type=Path, default=TRAIN_V06)
    every.add_argument("--train-config", type=Path, default=TRAIN_CONFIG)
    every.add_argument("--out-dir", type=Path, default=OUT_DIR)
    every.set_defaults(func=cmd_all)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
