#!/usr/bin/env python
"""Task v0.7a — the v0.7 training input: v0.6's input with some records' **shape** transformed, labels unchanged (hypothesis H-shape).

    # the whole assembly, one command (CPU unit, ≤ 32 GB): transform → sealed checks → exposure, tokens and time
    python scripts/assemble_v07.py all --out-dir artifacts/simgen/v07
    # the steps one by one
    python scripts/assemble_v07.py transform --out artifacts/simgen/v07/train-v07.jsonl
    python scripts/assemble_v07.py overlap --input artifacts/simgen/v07/train-v07.jsonl --out-dir artifacts/simgen/v07
    python scripts/assemble_v07.py exposure --input artifacts/simgen/v07/train-v07.jsonl --out artifacts/simgen/v07/train-v07.exposure.json

**The one factor** (PRE-DECLARED v0.7 VERDICT RULE, progress.md 2026-10-15; :mod:`robo_jev.simgen.shape` holds the pure parts): records
of ``train-v06.jsonl`` (sha256 42d6bd15…) are **transformed, never added**:

* ≈ 25 % of the physics records (engine and MuJoCo) become **sectioned sheets** of 1,500–8,000 characters in six styles, the original
  state verbatim and in order, the context sampled from the record's key alone (tone independent of the labels), a LOCAL PLAN that names
  the best action with p = 0.5 and a wrong one otherwise where an action question has a known best option;
* ≈ 5 % get their **facts moved into every question's instructions** with a minimal state;
* ≈ 8 % of the physics choice questions (English; action, outcome, direction, comparison) and ≈ 8 % of the general records get
  **N ∈ 9–26 options** with 0-probability distractors from other records' options of other answer families; a record with more than 8
  options carries the opt-in ``meta.option_range`` (:data:`robo_jev.decisions.records.OPTION_RANGE_MARK`).

Records the trainer holds out as dev (its own split: ``dev_fraction`` / ``dev_salt`` of the training config) keep v0.6's form, so the dev
curve stays comparable with v0.6's; they are never trained either way.

**Everything else is v0.6's, byte for byte**: ids, order, labels (an expanded question's label gains only its distractors' 0.0 — and the
worst own cost where it stores expected costs), ``meta`` (``meta.question_pick`` included; a transformed record gains ``meta.shape`` and,
when expanded beyond 8, ``meta.option_range``), every untouched line, the d1 part. ``transform`` proves it record by record and with
part digests; ``exposure`` proves the trainer's view (examples, steps, every engine category, per seed) equals v0.6's stored report, and
measures the shapes' exposure, the token growth and the expected training time (padded tokens simulated with the trainer's own packing,
scaled from v0.6's measured pods). ``overlap`` runs the sealed checks: T1–T7, G1, G2, T8a, T8b, T9 (``system1_v03.py overlap``), B1, the
H-set, A4 (outside: text units too), the M1 / M2 diagnostic sets (``artifacts/iface``: ids, requests, states, and their text units that are
not B1 / H-set template text — read as hashes only; no sheet text of M2 is ever printed).
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
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

SCRIPT_VERSION = "v07a-assemble-1"
CODE_PATHS = ("scripts/assemble_v07.py", "src/robo_jev/simgen/shape.py", "src/robo_jev/decisions/records.py", "scripts/assemble_v05.py",
              "scripts/assemble_v04.py", "scripts/system1_v03.py")
ARTIFACTS = REPO / "artifacts"
V06 = ARTIFACTS / "simgen" / "v06"
TRAIN_V06 = V06 / "train-v06.jsonl"
TRAIN_V06_SHA = "42d6bd150ce286bb0aff66c400e7a485443a25f5c52c2cf644b91292c6901aa4"
EXPOSURE_V06 = V06 / "train-v06.exposure.json"
OUT_DIR = ARTIFACTS / "simgen" / "v07"
TRAIN_CONFIG = REPO / "configs" / "train" / "decisions-v0-lora-h100.yaml"
MUJOCO_SETS = ("v03b-mujoco-train", "v06-a1", "v06-a2", "v06-a3")
PHYSICS = ("engine", "mujoco")
B1 = ARTIFACTS / "b1" / "b1-infamily.jsonl"
HSET = ARTIFACTS / "simgen" / "v04" / "scene-holdout.jsonl"
A4 = ARTIFACTS / "v06" / "a4" / "test.jsonl"
IFACE = {"M1": ARTIFACTS / "iface" / "m1" / "m1-set.jsonl", "M2-short": ARTIFACTS / "iface" / "m2" / "M2-short.jsonl",
         "M2-w2k": ARTIFACTS / "iface" / "m2" / "M2-w2k.jsonl", "M2-w4k": ARTIFACTS / "iface" / "m2" / "M2-w4k.jsonl",
         "M2-w8k": ARTIFACTS / "iface" / "m2" / "M2-w8k.jsonl"}
#: v0.6's measured pods (artifacts/cloud/runs/v06-s{0,1}-01-fetched/runs/*/metrics.json): step seconds, tokens and padded tokens of the
#: training steps (dev passes excluded), dev seconds, whole-step-loop seconds, and the job's train-step estimate.
V06_MEASURED = {"0": {"step_seconds": 7140.0, "tokens": 69971062, "padded_tokens": 104061695, "dev_seconds": 661.0, "seconds_total": 7884.0},
                "1": {"step_seconds": 7340.0, "tokens": 70001796, "padded_tokens": 104125027, "dev_seconds": 654.0, "seconds_total": 8092.0}}
V06_TRAIN_ESTIMATE_MIN = 135
HOURLY_USD = 3.99
RUN_CAP_USD = 35.0
#: Places never written to (the brief): v0.3–v0.6 inputs, t8 / t9 / b1, the authors' folders, iface, reports.
NEVER_WRITE = tuple(ARTIFACTS / p for p in ("simgen/v03", "simgen/v04", "simgen/v05", "simgen/v06", "simgen/v0b/tests", "simgen/probes",
                                            "simgen/v02", "t8", "t9", "b1", "v06", "iface", "reports"))
#: The physics share of the many-option rule (8 % of physics choice questions) and the general share (8 % of general records).
TARGET_SHARES = {"sheet": 0.25, "facts": 0.05, "many_physics": 0.08, "many_general": 0.08}


def _log(message: str) -> None:
    print(f"[v07a {dt.datetime.now().strftime('%H:%M:%S')}] {message}", file=sys.stderr, flush=True)


@functools.lru_cache(maxsize=None)
def _script(name: str):
    spec = importlib.util.spec_from_file_location(f"v07a_{Path(name).stem}", REPO / "scripts" / name)
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
    return _script("assemble_v04.py")._part(dict(record), set(MUJOCO_SETS))


# --------------------------------------------------------------------------
# Planning (pass 1) and the record transform (pass 2) — pure given the plan
# --------------------------------------------------------------------------


class Plan:
    """What pass 1 learns from the whole input: the distractor pools, the group facts that keep a pair's two records alike (the longest
    state per sheet key; the union of own keys, option texts and words per expanded question), and the counts the shares are read from."""

    def __init__(self, config: Mapping[str, Any]) -> None:
        from robo_jev.simgen.shape import DistractorPool

        self.config = dict(config)
        self.pool = DistractorPool()
        self.general_pool: list[tuple[str, str, str]] = []
        self._general_seen: set[str] = set()
        self.reference: dict[str, int] = {}
        self.questions: dict[tuple[str, str], dict[str, set]] = {}
        self.counts: Counter = Counter()

    def dev(self, record: Mapping[str, Any]) -> bool:
        from robo_jev.decisions.examples import is_dev

        return is_dev(dict(record), fraction=self.config["dev_fraction"], salt=self.config["dev_salt"])

    def selected_questions(self, record: Mapping[str, Any], part: str, key: str) -> list[tuple[str, str]]:
        """(question name, family) of a physics record's questions the many-option rule expands (English; families act/out/dir/cmp)."""
        from robo_jev.simgen import shape

        if part not in PHYSICS or (record.get("meta") or {}).get("language") != "en":
            return []
        out = []
        for name in record["request"]["questions"]:
            family = shape.question_family(record, name, part)
            if family in shape.PHYSICS_FAMILIES and shape.unit(shape.SHAPE_VERSION, "many", key, name) < shape.PHYSICS_MANY_P:
                out.append((name, family))
        return out

    def general_selected(self, record: Mapping[str, Any], key: str) -> bool:
        from robo_jev.simgen import shape

        return shape.general_eligible(record) and shape.unit(shape.SHAPE_VERSION, "many-general", key) < shape.GENERAL_MANY_P

    def scan(self, record: Mapping[str, Any]) -> None:
        from robo_jev.decisions.records import shuffle_key
        from robo_jev.simgen import shape

        part = part_of(record)
        self.counts[f"records:{part}"] += 1
        dev = self.dev(record)
        key = shuffle_key(dict(record))
        if part in PHYSICS:
            choice = [n for n, q in record["request"]["questions"].items() if q.get("type") == "choice"]
            self.counts[f"choice_questions:{part}"] += len(choice)
            if dev:
                self.counts[f"dev:{part}"] += 1
                return
            self.pool.add_record(record, part)
            for name in choice:
                family = shape.question_family(record, name, part)
                lang = (record.get("meta") or {}).get("language")
                self.counts[f"family:{part}:{lang}:{family}"] += 1
            if shape.assign_kind(key) == "sheet":
                self.reference[key] = max(self.reference.get(key, 0), len(record["request"]["state"]))
            for name, _family in self.selected_questions(record, part, key):
                question = record["request"]["questions"][name]
                info = self.questions.setdefault((key, name), {"keys": set(), "norms": set(), "words": set()})
                info["keys"] |= set(map(str, question["criteria"]))
                info["norms"] |= {shape._norm(v) for v in question["criteria"].values() if shape._norm(v)}
                info["words"] |= shape.content_words(question.get("instructions") or "")
                for k, v in question["criteria"].items():
                    info["words"] |= shape.content_words(f"{k} {v}")
        elif part == "general":
            if dev:
                self.counts["dev:general"] += 1
                return
            origin = str((record.get("meta") or {}).get("origin_group") or record["id"])
            source = shape.general_source(record)
            (question,) = record["request"]["questions"].values()
            for text in question["criteria"].values():
                text = " ".join(str(text).split())
                norm = shape._norm(text)
                if norm and norm not in self._general_seen and shape.general_candidate(text):
                    self._general_seen.add(norm)
                    self.general_pool.append((origin, source, text))
            self.counts["general_eligible"] += int(shape.general_eligible(record))


def transform_record(record: Mapping[str, Any], plan: Plan) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    """(the transformed record — None when nothing changes —, what happened). Pure given the plan."""
    from robo_jev.decisions.records import OPTION_RANGE, OPTION_RANGE_MARK, shuffle_key
    from robo_jev.simgen import shape

    part = part_of(record)
    key = shuffle_key(dict(record))
    event: dict[str, Any] = {"part": part, "kind": None, "expanded": {}, "skipped": []}
    if part not in (*PHYSICS, "general") or plan.dev(record):
        return None, event
    new = json.loads(json.dumps(record))
    changed = False
    if part in PHYSICS:
        for name, family in plan.selected_questions(record, part, key):
            info = plan.questions[(key, name)]
            question, label = record["request"]["questions"][name], record["labels"][name]
            count = shape.draw_count(key, name) - len(question["criteria"])
            entries = plan.pool.draw(family, count=count, own_keys=info["keys"], own_norms=info["norms"], words=info["words"],
                                     scene=shape.scene_of(record, part), rnd=shape.rng(shape.SHAPE_VERSION, "draw", key, name))
            if entries is None:
                event["skipped"].append(name)
                continue
            new_q, new_l = shape.expand_choice(question, label, [(e.key, e.description) for e in entries])
            new["request"]["questions"][name], new["labels"][name] = new_q, new_l
            event["expanded"][name] = {"family": family, "from": len(question["criteria"]), "to": len(new_q["criteria"]),
                                       "sources": dict(Counter(e.family for e in entries))}
            changed = True
        kind = shape.assign_kind(key)
        if kind == "sheet":
            pick = shape.plan_pick(record)
            sheet, info = shape.build_sheet(record["request"]["state"], key=key, reference_length=plan.reference[key], pick=pick)
            new["request"]["state"] = sheet
            event.update(kind="sheet", sheet=info)
            changed = True
        elif kind == "facts":
            state, instructions, info = shape.facts_request(record, key=key)
            new["request"]["state"] = state
            for name, text in instructions.items():
                new["request"]["questions"][name]["instructions"] = text
            event.update(kind="facts", facts=info)
            changed = True
    elif part == "general" and plan.general_selected(record, key):
        (name, question), = record["request"]["questions"].items()
        count = shape.draw_count(key, name) - len(question["criteria"])
        texts = shape.general_distractors(record, plan.general_pool, count=count, rnd=shape.rng(shape.SHAPE_VERSION, "draw-general", key))
        if texts is None:
            event["skipped"].append(name)
        else:
            taken, keys, k = set(question["criteria"]), [], 1
            while len(keys) < len(texts):
                if f"o{k}" not in taken:
                    keys.append(f"o{k}")
                k += 1
            new_q, new_l = shape.expand_choice(question, record["labels"][name], list(zip(keys, texts)))
            new["request"]["questions"][name], new["labels"][name] = new_q, new_l
            event["expanded"][name] = {"family": "general", "from": len(question["criteria"]), "to": len(new_q["criteria"])}
            changed = True
    if not changed:
        return None, event
    meta = new.setdefault("meta", {})
    block: dict[str, Any] = {"version": shape.SHAPE_VERSION, "kind": event["kind"]}
    if event["kind"] == "sheet":
        block["sheet"] = event["sheet"]
    if event["kind"] == "facts":
        block["facts"] = event["facts"]
    if event["expanded"]:
        block["expanded"] = {n: {k: v for k, v in e.items() if k != "sources"} for n, e in event["expanded"].items()}
    meta["shape"] = block
    if any(q.get("type") == "choice" and len(q["criteria"]) > OPTION_RANGE[1] for q in new["request"]["questions"].values()):
        meta["option_range"] = dict(OPTION_RANGE_MARK)
    return new, event


def record_problems(old: Mapping[str, Any], new: Mapping[str, Any], event: Mapping[str, Any]) -> list[str]:
    """Every way a transformed record breaks the rule (empty = fine): everything but the shape is v0.6's (module docstring)."""
    from robo_jev.decisions.records import OPTION_RANGE, OPTION_RANGE_MARK, validate_record
    from robo_jev.simgen import shape

    problems = []
    if list(old) != list(new) or any(old[k] != new[k] for k in old if k not in ("request", "labels", "meta")):
        problems.append("a top-level field other than request / labels / meta changed")
    old_meta, new_meta = old.get("meta") or {}, new.get("meta") or {}
    if {k: v for k, v in new_meta.items() if k not in ("shape", "option_range")} != old_meta or list(new_meta)[: len(old_meta)] != list(old_meta):
        problems.append("meta changed beyond meta.shape / meta.option_range")
    if new_meta.get("question_pick") != old_meta.get("question_pick"):
        problems.append("meta.question_pick changed")
    oq, nq = old["request"]["questions"], new["request"]["questions"]
    if list(oq) != list(nq) or set(old["request"]) != set(new["request"]):
        problems.append("question names / order or request keys changed")
        return problems
    expanded = event.get("expanded") or {}
    for name in oq:
        a, b = oq[name], nq[name]
        if {k: v for k, v in a.items() if k not in ("criteria", "instructions")} != {k: v for k, v in b.items() if k not in ("criteria", "instructions")}:
            problems.append(f"{name}: a question field changed")
        if event.get("kind") != "facts" and a.get("instructions") != b.get("instructions"):
            problems.append(f"{name}: instructions changed")
        la, lb = old["labels"][name], new["labels"][name]
        if name in expanded:
            own = list(a["criteria"].items())
            if list(b["criteria"].items())[: len(own)] != own:
                problems.append(f"{name}: own options changed")
            extra = [k for k in b["criteria"] if k not in a["criteria"]]
            if {k: v for k, v in lb["probabilities"].items() if k in a["criteria"]} != la["probabilities"]:
                problems.append(f"{name}: an own probability changed")
            if any(lb["probabilities"][k] != 0.0 for k in extra):
                problems.append(f"{name}: a distractor has probability above 0")
            if {k: v for k, v in lb.items() if k not in ("probabilities", "expected_cost")} != {k: v for k, v in la.items() if k not in ("probabilities", "expected_cost")}:
                problems.append(f"{name}: a label field changed")
            if "expected_cost" in la and {k: v for k, v in lb["expected_cost"].items() if k in a["criteria"]} != la["expected_cost"]:
                problems.append(f"{name}: an own expected cost changed")
        else:
            if a.get("criteria") != b.get("criteria"):
                problems.append(f"{name}: options changed without an expansion")
            if la != lb:
                problems.append(f"{name}: label changed")
    if set(old["labels"]) != set(new["labels"]):
        problems.append("label names changed")
    if event.get("kind") == "sheet":
        info = event["sheet"]
        problems += [f"sheet: {p}" for p in shape.sheet_problems(old["request"]["state"], new["request"]["state"], cuts=info["cuts"], spans=info["spans"])]
    elif event.get("kind") == "facts":
        problems += [f"facts: {p}" for p in shape.facts_problems(old, new)]
    elif old["request"]["state"] != new["request"]["state"]:
        problems.append("state changed without a sheet / facts transform")
    wide = any(q.get("type") == "choice" and len(q["criteria"]) > OPTION_RANGE[1] for q in nq.values())
    if wide != (new_meta.get("option_range") == OPTION_RANGE_MARK) or (not wide and "option_range" in new_meta):
        problems.append("meta.option_range does not match the option counts")
    problems += [f"validation: {p}" for p in validate_record(dict(new))]
    return problems


# --------------------------------------------------------------------------
# transform — the input, its manifest and change report
# --------------------------------------------------------------------------


def _band(value: float) -> str:
    from robo_jev.bench.system1 import band

    return band(float(value))


def _length_bin(n: int) -> str:
    for edge in (2000, 3000, 4000, 6000):
        if n < edge:
            return f"<{edge}"
    return ">=6000"


def chi_square(table: Mapping[str, Mapping[str, int]]) -> dict[str, Any]:
    """Pearson's χ² test of independence of a rows × columns count table (rows with no counts dropped)."""
    from scipy.stats import chi2_contingency

    rows = [r for r in table if sum(table[r].values())]
    cols = sorted({c for r in rows for c in table[r]})
    matrix = [[table[r].get(c, 0) for c in cols] for r in rows]
    if len(rows) < 2 or len(cols) < 2:
        return {"chi2": None, "dof": None, "p_value": None}
    stat, p, dof, _ = chi2_contingency(matrix)
    total = sum(map(sum, matrix))
    cramers_v = math.sqrt(stat / (total * (min(len(rows), len(cols)) - 1))) if total else None
    return {"chi2": float(stat), "dof": int(dof), "p_value": float(p), "cramers_v": cramers_v, "n": total}


class Report:
    """Counts of what the transform did — shares, the independence tables, the many-option profile, audit samples."""

    def __init__(self) -> None:
        self.c: Counter = Counter()
        self.tables: dict[str, dict[str, Counter]] = defaultdict(lambda: defaultdict(Counter))
        self.n_hist: Counter = Counter()
        self.lengths: list[int] = []
        self.audit_physics: list[dict[str, Any]] = []
        self.audit_general: list[dict[str, Any]] = []
        self.sheet_examples: dict[str, list[dict[str, Any]]] = defaultdict(list)
        self._rng = random.Random(f"{SCRIPT_VERSION}|audit")
        self._seen_physics = 0
        self._seen_general = 0

    def add(self, old: Mapping[str, Any], new: Mapping[str, Any] | None, event: Mapping[str, Any]) -> None:
        part = event["part"]
        self.c[f"records:{part}"] += 1
        if new is None:
            return
        self.c[f"changed:{part}"] += 1
        kind = event.get("kind")
        if kind:
            self.c[f"{kind}:{part}"] += 1
        bands = [_band((old["labels"][n] or {}).get("p_yes", (old["labels"][n] or {}).get("value")))
                 for n, q in old["request"]["questions"].items() if q.get("type") in ("noul", "score")]
        if kind == "sheet":
            info = event["sheet"]
            self.lengths.append(info["length"])
            right = (info["plan_pick"] or {}).get("right")
            plan_state = info["plan"] if info["plan"] != "line" else ("line:unscored" if right is None else "line:right" if right else "line:wrong")
            key_q = (old.get("meta") or {}).get("key_question")
            key_label = (old.get("labels") or {}).get(key_q) or {}
            key_value = key_label.get("p_yes", key_label.get("value"))
            if key_value is not None:  # one count per record: the record's key question
                key_band = _band(key_value)
                self.tables["record:tone_x_key_band"][info["tone"]][key_band] += 1
                self.tables["record:style_x_key_band"][info["style"]][key_band] += 1
                self.tables["record:length_x_key_band"][_length_bin(info["length"])][key_band] += 1
                if info["plan_eligible"]:  # the plan block's coin acts only within the records that can carry a recommendation
                    self.tables["record:plan_mode_x_key_band (eligible records)"][info["plan"]][key_band] += 1
                if right is not None:
                    self.tables["record:plan_right_x_key_band (known-best line records)"][str(right)][key_band] += 1
                self.tables["record:plan_eligible_x_key_band (structure)"][str(info["plan_eligible"])][key_band] += 1
            for b in bands:
                self.tables["tone_x_band"][info["tone"]][b] += 1
                self.tables["style_x_band"][info["style"]][b] += 1
                self.tables["length_x_band"][_length_bin(info["length"])][b] += 1
                self.tables["plan_x_band"][plan_state][b] += 1
            self.c[f"tone:{info['tone']}"] += 1
            self.c[f"style:{info['style']}"] += 1
            self.c[f"chunks:{info['chunks']}"] += 1
            self.c[f"plan:{info['plan']}"] += 1
            self.c[f"topped_up"] += int(info["topped_up_lines"] > 0)
            self.c[f"plan_eligible:{info['plan_eligible']}"] += 1
            if info["plan"] == "line":
                self.c[f"plan_right:{info['plan_pick']['right']}"] += 1
            if len(self.sheet_examples[info["style"]]) < 2 and self._rng.random() < 0.02:
                self.sheet_examples[info["style"]].append({"id": new["id"], "tone": info["tone"], "length": info["length"], "sheet": new["request"]["state"]})
        if kind == "facts":
            self.c[f"facts_template:{event['facts']['template']}"] += 1
        for name, entry in (event.get("expanded") or {}).items():
            fam = entry["family"]
            self.c[f"expanded:{part}"] += 1
            self.c[f"expanded_family:{fam}"] += 1
            self.n_hist[entry["to"]] += 1
            for src, n in (entry.get("sources") or {}).items():
                self.c[f"distractor_source:{fam}<-{src}"] += n
            sample = {"id": new["id"], "question": name, "family": fam, "instructions": old["request"]["questions"][name]["instructions"],
                      "own": {k: [v, old["labels"][name]["probabilities"][k]] for k, v in old["request"]["questions"][name]["criteria"].items()},
                      "distractors": {k: v for k, v in new["request"]["questions"][name]["criteria"].items() if k not in old["request"]["questions"][name]["criteria"]}}
            if part == "general":
                sample["state"] = old["request"]["state"]
                self._seen_general += 1
                self._reservoir(self.audit_general, sample, self._seen_general, 40)
            else:
                self._seen_physics += 1
                self._reservoir(self.audit_physics, sample, self._seen_physics, 100)
        for name in event.get("skipped") or []:
            self.c[f"skipped:{part}"] += 1

    def _reservoir(self, bucket: list, item: Mapping[str, Any], seen: int, size: int) -> None:
        if len(bucket) < size:
            bucket.append(dict(item))
        else:
            j = self._rng.randrange(seen)
            if j < size:
                bucket[j] = dict(item)


def transform_file(in_path: Path, out_path: Path, *, config: Mapping[str, Any]) -> dict[str, Any]:
    """Pass 1 (plan) and pass 2 (transform, check, write) over `in_path` → `out_path`; every check that fails raises; `out_path` is
    replaced only when the whole stream passed."""
    from robo_jev.simgen import shape

    v05 = _script("assemble_v05.py")
    plan = Plan(config)
    _log(f"pass 1: {in_path}")
    for raw in _raw_lines(in_path):
        plan.scan(json.loads(raw))
    _log(f"pass 1 done: pools {plan.pool.sizes()} (excluded {plan.pool.excluded}), general pool {len(plan.general_pool)}, "
         f"sheet keys {len(plan.reference)}, expanded-question slots {len(plan.questions)}")
    report = Report()
    digests = {side: defaultdict(hashlib.sha256) for side in ("before", "after")}
    untouched = {side: hashlib.sha256() for side in ("before", "after")}
    ids = {side: hashlib.sha256() for side in ("before", "after")}
    styles: Counter = Counter()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    temp = out_path.with_name(out_path.name + ".tmp")
    failures: list[str] = []
    try:
        with temp.open("wb") as handle:
            for raw in _raw_lines(in_path):
                record = json.loads(raw)
                new, event = transform_record(record, plan)
                report.add(record, new, event)
                part = event["part"]
                ids["before"].update(str(record["id"]).encode("utf-8") + b"\n")
                if new is None:
                    line = raw
                    untouched["before"].update(raw)
                    untouched["after"].update(raw)
                else:
                    problems = record_problems(record, new, event)
                    if problems:
                        failures.append(f"{record['id']}: {problems[:3]}")
                        if len(failures) >= 20:
                            raise ValueError(f"records break the rule: {failures}")
                    style = v05.line_style(raw, record)
                    styles[style] += 1
                    line = v05.dumps_style(new, style)
                    if json.loads(line) != new:
                        raise ValueError(f"{record['id']}: the written line does not read back as the transformed record")
                ids["after"].update(str(json.loads(line)["id"]).encode("utf-8") + b"\n")
                digests["before"][part].update(raw)
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
    counts = plan.counts
    physics_records = sum(counts[f"records:{p}"] for p in PHYSICS)
    physics_choice = sum(counts[f"choice_questions:{p}"] for p in PHYSICS)
    general_records = counts["records:general"]
    c = report.c
    shares = {
        "sheet_of_physics_records": sum(c[f"sheet:{p}"] for p in PHYSICS) / physics_records,
        "facts_of_physics_records": sum(c[f"facts:{p}"] for p in PHYSICS) / physics_records,
        "expanded_of_physics_choice_questions": sum(c[f"expanded:{p}"] for p in PHYSICS) / physics_choice,
        "expanded_of_general_records": c["expanded:general"] / general_records,
    }
    lengths = sorted(report.lengths)
    q = lambda f: lengths[int(f * (len(lengths) - 1))] if lengths else None  # noqa: E731
    parts = {part: {"sha256_before": digests["before"][part].hexdigest(), "sha256_after": digests["after"][part].hexdigest(),
                    "identical": digests["before"][part].hexdigest() == digests["after"][part].hexdigest(), "records": c[f"records:{part}"],
                    "changed": c[f"changed:{part}"]} for part in sorted(digests["before"])}
    tables = {name: {row: dict(cols) for row, cols in rows.items()} for name, rows in report.tables.items()}
    return {
        "records": sum(c[k] for k in c if k.startswith("records:")), "parts": parts,
        "digests": {"id_order": {"before": ids["before"].hexdigest(), "after": ids["after"].hexdigest(), "identical": True},
                    "untouched_lines": {"before": untouched["before"].hexdigest(), "after": untouched["after"].hexdigest(), "identical": True}},
        "line_styles_of_changed_records": dict(styles),
        "plan": {"pools": plan.pool.sizes(), "pool_excluded_options": plan.pool.excluded, "general_pool": len(plan.general_pool),
                 "general_eligible_records": counts["general_eligible"], "sheet_keys": len(plan.reference), "expanded_question_slots": len(plan.questions),
                 "dev_records": {p: counts[f"dev:{p}"] for p in ("engine", "mujoco", "general")},
                 "physics_choice_questions_by_family": {k.split(":", 1)[1]: v for k, v in sorted(counts.items()) if k.startswith("family:")}},
        "counts": dict(sorted(c.items())), "shares": shares, "targets": TARGET_SHARES,
        "sheet_lengths": {"n": len(lengths), "min": lengths[0] if lengths else None, "p10": q(0.1), "p50": q(0.5), "p90": q(0.9),
                          "max": lengths[-1] if lengths else None, "mean": sum(lengths) / len(lengths) if lengths else None},
        "independence": {"what": "label bands (low / middle / high, robo_jev.bench.system1.band) of the wrapped records by the sheet's context "
                                 "draws — 'record:' tables count one record by its key question (meta.key_question; the χ² test's unit), the "
                                 "others every yes/no + score question (descriptive: questions of a record share its draws, so their χ² "
                                 "overstates the evidence); every draw comes from sha256(SHAPE_VERSION | … | shuffle key) and none reads a label",
                         "tables": tables, "chi_square": {name: chi_square(t) for name, t in tables.items()}},
        "many_options": {"n_histogram": {str(k): v for k, v in sorted(report.n_hist.items())},
                         "families": {k.split(":", 1)[1]: v for k, v in c.items() if k.startswith("expanded_family:")},
                         "distractor_sources": {k.split(":", 1)[1]: v for k, v in sorted(c.items()) if k.startswith("distractor_source:")},
                         "skipped_pool_exhausted": {k.split(":", 1)[1]: v for k, v in c.items() if k.startswith("skipped:")},
                         "rule": {"physics_p": shape.PHYSICS_MANY_P, "general_p": shape.GENERAL_MANY_P, "n_range": list(shape.N_RANGE),
                                  "allowed": shape.ALLOWED}},
        "audit": {"physics": report.audit_physics, "general": report.audit_general},
        "sheet_examples": {k: v for k, v in report.sheet_examples.items()},
    }


RULE_TEXT = ("PRE-DECLARED v0.7 VERDICT RULE (progress.md, 2026-10-15 ~05:45 KST): v0.6's training input with labels unchanged and one factor "
             "changed — request SHAPE, by transforming (not adding) records: ≈ 25 % of physics records wrapped into label-independent sectioned "
             "sheets (1.5k–8k chars), ≈ 8 % of physics choice questions expanded with 0-probability distractors to N ∈ 9–26 plus licensed general "
             "MC training items expanded likewise within the general share, ≈ 5 % of physics records with situation facts moved into question "
             "instructions")


def render_changes(report: Mapping[str, Any]) -> str:
    s = report["shares"]
    c = report["counts"]
    lines = ["# v0.7 training input — the shape transform", "",
             f"`{report['out']['path']}` sha256 `{report['out']['sha256']}` ({report['records']:,} records) from `{report['input']['path']}` "
             f"sha256 `{report['input']['sha256']}`. {RULE_TEXT}.", "", "## Parts", "",
             "| part | records | changed | sha256 before | sha256 after | identical |", "|---|---|---|---|---|---|"]
    for part, e in report["parts"].items():
        lines.append(f"| {part} | {e['records']:,} | {e['changed']:,} | `{e['sha256_before'][:16]}…` | `{e['sha256_after'][:16]}…` | {e['identical']} |")
    lines += ["", "## Shares (targets: sheet 0.25, facts 0.05, many-option 0.08 of physics choice questions, 0.08 of general records)", ""]
    lines += [f"* {k}: {v:.4f}" for k, v in s.items()]
    sl = report["sheet_lengths"]
    lines += ["", f"Sheets: {sl['n']:,}, length min {sl['min']}, p10 {sl['p10']}, median {sl['p50']}, p90 {sl['p90']}, max {sl['max']} characters.", "",
              "## Independence of the context from the labels", "", report["independence"]["what"] + ".", ""]
    for name, table in report["independence"]["tables"].items():
        chi = report["independence"]["chi_square"][name]
        lines += [f"**{name}** — χ² {chi['chi2']:.2f}, dof {chi['dof']}, p = {chi['p_value']:.3f}, Cramér's V {chi['cramers_v']:.4f}" if chi["chi2"] is not None else f"**{name}**", "",
                  "| row | low | middle | high | low share | high share |", "|---|---|---|---|---|---|"]
        for row, cols in sorted(table.items()):
            total = sum(cols.values())
            lines.append(f"| {row} | {cols.get('low', 0):,} | {cols.get('middle', 0):,} | {cols.get('high', 0):,} | {cols.get('low', 0) / total:.3f} | {cols.get('high', 0) / total:.3f} |")
        lines.append("")
    m = report["many_options"]
    lines += ["## Many options", "", f"N histogram: {m['n_histogram']}", "", f"Families expanded: {m['families']}", "",
              f"Distractor sources (question family <- source family): {m['distractor_sources']}", "", f"Skipped (pool exhausted): {m['skipped_pool_exhausted']}", ""]
    return "\n".join(lines) + "\n"


def cmd_transform(args: argparse.Namespace) -> int:
    refuse_write(args.out)
    sha = _sha256(args.input)
    if args.input.resolve() == TRAIN_V06.resolve() and sha != TRAIN_V06_SHA:
        raise SystemExit(f"{args.input}: sha256 {sha[:12]}… is not v0.6's input {TRAIN_V06_SHA[:12]}…")
    config = _config(args.train_config)
    _log(f"transform: {args.input} ({sha[:12]}…) → {args.out}")
    try:
        report = transform_file(args.input, args.out, config=config)
    except ValueError as error:
        raise SystemExit(f"refusing: {error}") from error
    out_sha = _sha256(args.out)
    head = {"kind": "v07-train-changes", "script_version": SCRIPT_VERSION, "rule": RULE_TEXT, "git": _git(),
            "input": {"path": str(args.input), "sha256": sha}, "out": {"path": str(args.out), "sha256": out_sha, "bytes": args.out.stat().st_size},
            "train_config": {"path": str(args.train_config), **{k: config[k] for k in ("dev_fraction", "dev_salt", "questions_per_request")}}}
    audit, examples = report.pop("audit"), report.pop("sheet_examples")
    full = {**head, **report, "created": dt.datetime.now().astimezone().isoformat(timespec="seconds")}
    stem = args.out.stem
    _write_json(args.out.with_name(f"{stem}.changes.json"), full)
    args.out.with_name(f"{stem}.changes.md").write_text(render_changes(full), encoding="utf-8")
    _write_json(args.out.with_name(f"{stem}.audit.json"), {"kind": "v07-distractor-audit-sample", "what": "reservoir samples of expanded "
                "questions (physics 100, general 40): instructions, own options with label probabilities, the distractors", **audit})
    _write_json(args.out.with_name(f"{stem}.sheet-examples.json"), examples)
    from robo_jev.decisions.records import OPTION_RANGE_MARK

    manifest = {"kind": "v07-train", "script_version": SCRIPT_VERSION, "git": _git(), "sha256": out_sha,
                "out": {"path": str(args.out), "sha256": out_sha, "bytes": args.out.stat().st_size, "records": report["records"]},
                "input": {"path": str(args.input), "sha256": sha, "manifest": str(args.input.with_name(args.input.stem + ".manifest.json"))},
                "rule": RULE_TEXT, "shares": report["shares"], "option_range_mark": OPTION_RANGE_MARK, "parts": report["parts"],
                "changes": str(args.out.with_name(f"{stem}.changes.json")),
                "created": dt.datetime.now().astimezone().isoformat(timespec="seconds")}  # fmt: skip
    _write_json(args.out.with_name(f"{stem}.manifest.json"), manifest)
    _log(f"transform: {report['records']} records; shares {report['shares']}; parts identical {[(p, e['identical']) for p, e in report['parts'].items()]} "
         f"→ {args.out} {out_sha[:16]}…")
    return 0


# --------------------------------------------------------------------------
# overlap — the sealed checks
# --------------------------------------------------------------------------


def iface_outside_units(paths: Mapping[str, Path], template_paths: Sequence[Path]) -> dict[str, set[str]]:
    """Per iface set, the hashes of its text units (:func:`robo_jev.data.general_sets.record_units`) that are not B1 / H-set units — the
    text written outside our generators (ARC questions and options, the M2 sheet filler, whole wrapped states). Hashes only."""
    from robo_jev.data.general_sets import record_units

    def h(unit: str) -> str:
        return hashlib.sha256(unit.encode("utf-8")).hexdigest()[:16]

    template = {h(u) for path in template_paths for raw in _raw_lines(path) for u in record_units(json.loads(raw))}
    return {name: {h(u) for raw in _raw_lines(path) for u in record_units(json.loads(raw))} - template for name, path in paths.items()}


def cmd_overlap(args: argparse.Namespace) -> int:
    from robo_jev.data.general_sets import record_units
    from robo_jev.simgen.holdout import request_hash
    from robo_jev.simgen.narration import state_sha256

    out_dir = args.out_dir
    refuse_write(out_dir)
    missing = [str(p) for p in (B1, HSET, A4, *IFACE.values()) if not p.is_file()]
    if missing:
        raise SystemExit(f"sealed files missing: {missing} — every sealed set is checked")
    stem = args.input.stem
    sets = sum((["--mujoco-set", s] for s in MUJOCO_SETS), [])
    extra = sum((["--sealed-extra", f"{name}={path}"] for name, path in IFACE.items()), [])
    rc = subprocess.run([sys.executable, "scripts/system1_v03.py", "overlap", "--input", str(args.input), "--out", str(out_dir / f"{stem}.overlap.json"),
                         "--sealed-extra", f"B1={B1}", "--sealed-extra", f"H-set={HSET}", "--sealed-outside", f"A4={A4}", *extra, *sets],
                        cwd=REPO).returncode
    _log("iface outside units (hashes only)")
    outside = iface_outside_units(IFACE, (B1, HSET))
    hits: Counter = Counter()
    by_part: dict[str, Counter] = defaultdict(Counter)

    def h(unit: str) -> str:
        return hashlib.sha256(unit.encode("utf-8")).hexdigest()[:16]

    ids_requests_states = {name: {"ids": set(), "requests": set(), "states": set()} for name in IFACE}
    for name, path in IFACE.items():
        for raw in _raw_lines(path):
            r = json.loads(raw)
            ids_requests_states[name]["ids"].add(str(r["id"]))
            ids_requests_states[name]["requests"].add(request_hash(r))
            ids_requests_states[name]["states"].add(state_sha256(r["request"]["state"]))
    for raw in _raw_lines(args.input):
        record = json.loads(raw)
        part = part_of(record)
        changed = "shape" in (record.get("meta") or {})
        units = {h(u) for u in record_units(record)}
        for name, hashes in outside.items():
            shared = len(units & hashes)
            if shared:
                hits[name] += 1
                by_part[f"{part}:{'changed' if changed else 'untouched'}"][name] += 1
        for name, index in ids_requests_states.items():
            if str(record["id"]) in index["ids"] or request_hash(record) in index["requests"] or state_sha256(record["request"]["state"]) in index["states"]:
                hits[f"{name}:id_request_state"] += 1
    report = {"kind": "v07-iface-units", "script_version": SCRIPT_VERSION, "git": _git(), "input": {"path": str(args.input), "sha256": _sha256(args.input)},
              "what": "text units of the M1 / M2 diagnostic sets that are not B1 / H-set template units (ARC text, the M2 filler, whole wrapped states), "
                      "matched by hash against every record of the input; ids, whole requests and whole states too. Hashes only — no text read out.",
              "outside_units": {name: len(v) for name, v in outside.items()}, "records_with_hits": dict(hits), "by_part": {k: dict(v) for k, v in by_part.items()},
              "total_hits": sum(hits.values()), "system1_overlap_rc": rc,
              "created": dt.datetime.now().astimezone().isoformat(timespec="seconds")}
    _write_json(out_dir / f"{stem}.iface-units.json", report)
    overlap = json.loads((out_dir / f"{stem}.overlap.json").read_text(encoding="utf-8"))
    _log(f"overlap: system1_v03 hits {overlap.get('total_hits')} (rc {rc}) · iface outside-unit hits {report['total_hits']} {dict(hits)}")
    return 0 if rc == 0 and report["total_hits"] == 0 else 2


# --------------------------------------------------------------------------
# exposure — the trainer's view, shapes, tokens and time
# --------------------------------------------------------------------------


def shape_exposure(records: Sequence[Mapping[str, Any]], *, config: Mapping[str, Any], seeds: Sequence[int]) -> dict[str, Any]:
    """Per seed, the trainer's epoch-0 examples of the transformed records: by kind (sheet / facts) and part, the many-option examples by
    part and N, and the shown letter of the correct option (argmax label) of the many-option examples under the trainer's option-order
    augmentation (:func:`robo_jev.decisions.prompt.option_order` with the plan's order seed)."""
    from robo_jev.decisions.examples import epoch_plan, split_dev
    from robo_jev.decisions.prompt import option_order
    from robo_jev.decisions.records import shuffle_key

    q = int(config["questions_per_request"])
    train, _ = split_dev(list(records), fraction=config["dev_fraction"], salt=config["dev_salt"])
    out = {}
    for seed in seeds:
        tally: Counter = Counter()
        letters: Counter = Counter()
        n_examples: Counter = Counter()
        late: Counter = Counter()
        for index, name, order_seed in epoch_plan(train, seed=int(seed), epoch=0, q=q, augment=bool(config.get("augment", True))):
            record = train[index]
            part = part_of(record)
            shape_block = (record.get("meta") or {}).get("shape") or {}
            tally[f"examples:{part}"] += 1
            if shape_block.get("kind"):
                tally[f"{shape_block['kind']}:{part}"] += 1
            expanded = (shape_block.get("expanded") or {}).get(name)
            if expanded:
                tally[f"many:{part}"] += 1
                question = record["request"]["questions"][name]
                probs = record["labels"][name]["probabilities"]
                keys = sorted(question["criteria"])
                shown = [keys[i] for i in option_order(shuffle_key(record), name, len(keys), order_seed)]
                best = max(probs, key=lambda k: (float(probs[k]), k))
                position = shown.index(best)
                letters[chr(ord("A") + position)] += 1
                n_examples[len(keys)] += 1
                late["Q-Z" if position >= 16 else "I-P" if position >= 8 else "A-H"] += 1
        total = sum(v for k, v in tally.items() if k.startswith("examples:"))
        out[str(seed)] = {"examples": total, **dict(sorted(tally.items())),
                          "shares": {k: v / total for k, v in sorted(tally.items()) if not k.startswith("examples:")},
                          "many_option_correct_letter": dict(sorted(letters.items())), "many_option_correct_bucket": dict(late),
                          "many_option_n": {str(k): v for k, v in sorted(n_examples.items())}}
    return out


def token_simulation(records: Sequence[Mapping[str, Any]], *, config: Mapping[str, Any], seeds: Sequence[int], tokenizer: Any,
                     exact_sample: int = 300) -> dict[str, Any]:
    """Per seed, the epoch-0 training steps exactly as the trainer forms them: each example's prompt length (prefix tokens per record +
    question-block tokens per question, cached; the question block's length does not depend on the option order — checked on a sample of
    exact :func:`robo_jev.decisions.examples.build_examples` prompts), each step's micro-batches by the trainer's own
    :func:`robo_jev.decisions.train.pack_micro_batches` (``micro_batch_tokens``), and the padded tokens (micro-batch size × its longest
    example). Plus the dev prompts' tokens."""
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
        """prefix tokens (the record's state; cached per record) + question tokens (a one-question request on a placeholder state —
        build_prompts tokenizes the prefix and the question block separately, so the two parts add up exactly)."""
        cached = lengths.get((index, name))
        if cached is None:
            record = train[index]
            if index not in prefixes:
                prefixes[index] = len(tokenizer.encode(_prompt_prefix(record["request"]["state"], fmt), add_special_tokens=False).ids)
            sub = {"state": "x", "questions": {name: record["request"]["questions"][name]}}
            (prompt,) = build_prompts(sub, tokenizer, key=shuffle_key(record), seed=1, fmt=fmt, mode="direct", orders="seeded", version=version)
            cached = lengths[(index, name)] = prefixes[index] + len(prompt["tokens"]) - prompt["prefix_tokens"]
        return cached

    out: dict[str, Any] = {}
    for seed in seeds:
        plan = epoch_plan(train, seed=int(seed), epoch=0, q=q, augment=True)
        tokens = padded = steps = micro_batches = 0
        longest = 0
        for start in range(0, len(plan), batch):
            chunk = plan[start:start + batch]
            ls = [length(i, n) for i, n, _ in chunk]
            groups = pack_micro_batches(ls, micro)
            tokens += sum(ls)
            padded += sum(len(g) * max(ls[i] for i in g) for g in groups)
            micro_batches += len(groups)
            steps += 1
            longest = max(longest, max(ls))
        rnd = random.Random(f"{SCRIPT_VERSION}|exact|{seed}")
        mismatches = 0
        for i, n, order_seed in rnd.sample(plan, min(exact_sample, len(plan))):
            (example,) = build_examples(train[i], [n], tokenizer, order_seed=order_seed, fmt=fmt, version=version)
            mismatches += int(len(example["tokens"]) != length(i, n))
        out[str(seed)] = {"steps": steps, "examples": len(plan), "tokens": tokens, "padded_tokens": padded, "micro_batches": micro_batches,
                          "longest_example": longest, "exact_check": {"sample": min(exact_sample, len(plan)), "length_mismatches": mismatches}}
    dev_prompts = dev_examples(dev_used, tokenizer, fmt=fmt, version=version)
    out["dev"] = {"records": len(dev_used), "prompts": len(dev_prompts), "tokens": sum(len(p["tokens"]) for p in dev_prompts)}
    return out


def time_estimate(sim_v07: Mapping[str, Any], sim_v06: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Scale v0.6's measured step seconds per padded token to v0.7's simulated padded tokens (per seed); dev passes as v0.6's, scaled by the
    dev tokens' ratio; a 10 % margin for the longer sequences' attention. Also the simulation's own reproduction of v0.6's measured padded
    tokens, when `sim_v06` is given."""
    out: dict[str, Any] = {}
    for seed, measured in V06_MEASURED.items():
        if seed not in sim_v07:
            continue
        rate = measured["step_seconds"] / measured["padded_tokens"]
        steps_seconds = rate * sim_v07[seed]["padded_tokens"]
        dev_ratio = (sim_v07["dev"]["tokens"] / sim_v06["dev"]["tokens"]) if sim_v06 else 1.0
        dev_seconds = measured["dev_seconds"] * dev_ratio
        overhead = measured["seconds_total"] - measured["step_seconds"] - measured["dev_seconds"]
        minutes = (steps_seconds + dev_seconds + overhead) / 60
        entry = {"seconds_per_padded_token_v06": rate, "step_minutes": steps_seconds / 60, "dev_minutes": dev_seconds / 60, "other_minutes": overhead / 60,
                 "train_minutes": minutes, "train_minutes_with_margin": minutes * 1.10,
                 "padded_tokens_ratio_vs_v06_measured": sim_v07[seed]["padded_tokens"] / measured["padded_tokens"],
                 "tokens_ratio_vs_v06_measured": sim_v07[seed]["tokens"] / measured["tokens"]}
        if sim_v06 and seed in sim_v06:
            entry["simulation_vs_v06_measured"] = {"tokens": sim_v06[seed]["tokens"] / measured["tokens"],
                                                   "padded_tokens": sim_v06[seed]["padded_tokens"] / measured["padded_tokens"]}
            entry["padded_tokens_ratio_vs_v06_simulated"] = sim_v07[seed]["padded_tokens"] / sim_v06[seed]["padded_tokens"]
        out[seed] = entry
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
    reports: dict[str, Any] = {}
    sims: dict[str, Any] = {}
    for name, path in (("v06", args.reference), ("v07", args.input)):
        records, validation = load_training_records(path)
        if validation["invalid"]:
            raise SystemExit(f"{path}: {validation['invalid']} records fail the trainer's validation ({validation['problems']})")
        entry = v04.exposure_report(records, config=config, seeds=args.seeds, mujoco_sets=set(MUJOCO_SETS))
        entry["input"] = {"path": str(path), "sha256": _sha256(path), "validation": validation}
        if name == "v07":
            entry["shape_exposure"] = shape_exposure(records, config=config, seeds=args.seeds)
        if tokenizer is not None:
            _log(f"tokens: {name}")
            sims[name] = token_simulation(records, config=config, seeds=args.seeds, tokenizer=tokenizer)
        reports[name] = entry
        del records
        _log(f"exposure {name}: " + "; ".join(f"seed {s}: {e['examples_per_epoch']} steps {e['steps']}" for s, e in entry["seeds"].items()))
    fields = ("examples_per_epoch", "steps", "engine_by_category", "action", "derived", "band", "comparison")
    same = {seed: {"vs_v06_same_code": {f: reports["v07"]["seeds"][seed][f] == reports["v06"]["seeds"][seed][f] for f in fields},
                   "vs_v06_stored": {f: reports["v07"]["seeds"][seed][f] == stored["seeds"][seed][f] for f in fields if f in stored["seeds"].get(seed, {})}}
            for seed in map(str, args.seeds)}
    split_same = {k: reports["v07"][k] == reports["v06"][k] for k in ("records", "train_records", "dev_records", "dev_used", "dev_parts", "train_by_part")}
    out = {"kind": "v07-exposure", "script_version": SCRIPT_VERSION, "git": _git(), "train_config": str(args.train_config),
           "v07": reports["v07"], "v06": reports["v06"], "stored_v06": str(args.stored), "identical": same, "split_identical": split_same}
    if sims:
        estimate = time_estimate(sims["v07"], sims.get("v06"))
        out["tokens"] = {"v06": sims.get("v06"), "v07": sims["v07"],
                         "growth": {seed: {"tokens": sims["v07"][seed]["tokens"] / sims["v06"][seed]["tokens"],
                                           "padded_tokens": sims["v07"][seed]["padded_tokens"] / sims["v06"][seed]["padded_tokens"]}
                                    for seed in map(str, args.seeds) if "v06" in sims}}
        out["time"] = estimate
        out["budget"] = budget_check(estimate)
    out["pass"] = all(all(v.values()) for s in same.values() for v in s.values()) and all(split_same.values())
    out["created"] = dt.datetime.now().astimezone().isoformat(timespec="seconds")
    _write_json(args.out, out)
    _log(f"exposure: identical to v0.6 per seed {same} · split {split_same} · pass {out['pass']}")
    if sims:
        _log(f"tokens growth {out['tokens']['growth']} · time {json.dumps(out['time'])[:600]} · budget {out['budget']}")
    return 0 if out["pass"] else 2


#: v0.6's pods outside training (measured minutes, the slower seed): boot + sync + env + weights, the diagnoses T2 … T6, fetch.
V06_POD_MINUTES = {"setup": 6.0, "diagnoses": 225.0, "fetch_and_margin": 12.0}


def budget_check(estimate: Mapping[str, Any]) -> dict[str, Any]:
    """The whole pod per seed: estimated train minutes (with margin) + v0.6's measured non-training minutes → hours and $ at
    :data:`HOURLY_USD`, against the brief's :data:`RUN_CAP_USD`; and the wall cap that keeps ≥ 30 min of slack."""
    out = {}
    for seed, entry in estimate.items():
        minutes = entry["train_minutes_with_margin"] + sum(V06_POD_MINUTES.values())
        hours = minutes / 60
        wall = math.ceil((hours + 0.5) * 4) / 4
        out[seed] = {"pod_hours": hours, "pod_usd": hours * HOURLY_USD, "within_run_cap": hours * HOURLY_USD <= RUN_CAP_USD,
                     "wall_cap_hours": wall, "cap_usd": wall * HOURLY_USD, "cap_within_run_cap": wall * HOURLY_USD <= RUN_CAP_USD}
    return out


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
    train = out / "train-v07.jsonl"
    me = str(Path(__file__).resolve().relative_to(REPO))
    if _run([me, "transform", "--input", str(args.input), "--out", str(train), "--train-config", str(args.train_config)]) != 0:
        raise SystemExit("transform failed")
    rc = {"overlap": _run([me, "overlap", "--input", str(train), "--out-dir", str(out)]),
          "exposure": _run([me, "exposure", "--input", str(train), "--reference", str(args.input), "--train-config", str(args.train_config),
                            "--out", str(out / "train-v07.exposure.json")])}
    load = lambda name: json.loads((out / name).read_text(encoding="utf-8")) if (out / name).is_file() else {}  # noqa: E731
    changes, overlap, iface, exposure = (load(n) for n in ("train-v07.changes.json", "train-v07.overlap.json", "train-v07.iface-units.json",
                                                           "train-v07.exposure.json"))
    summary = {"kind": "v07-assembly", "script_version": SCRIPT_VERSION, "git": _git(), "returncodes": rc,
               "files": {"train-v07.jsonl": {"path": str(train), "sha256": _sha256(train)}, "input": {"path": str(args.input), "sha256": _sha256(args.input)}},
               "shares": changes.get("shares"), "parts_identical": {p: e["identical"] for p, e in (changes.get("parts") or {}).items()},
               "overlap_hits": overlap.get("total_hits"), "sealed_checked": sorted(overlap.get("sealed") or {}), "iface_unit_hits": iface.get("total_hits"),
               "exposure_identical_to_v06": exposure.get("pass"), "tokens_growth": (exposure.get("tokens") or {}).get("growth"),
               "time": exposure.get("time"), "budget": exposure.get("budget"),
               "pass": all(v == 0 for v in rc.values()) and overlap.get("total_hits") == 0 and iface.get("total_hits") == 0 and bool(exposure.get("pass")),
               "created": dt.datetime.now().astimezone().isoformat(timespec="seconds")}  # fmt: skip
    _write_json(out / "train-v07.assembly.json", summary)
    _log(f"done: {train} sha256 {summary['files']['train-v07.jsonl']['sha256'][:16]}… · pass {summary['pass']}")
    return 0 if summary["pass"] else 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    tr = sub.add_parser("transform", help="train-v06 → train-v07 (+ manifest, change report, audit sample, sheet examples)")
    tr.add_argument("--input", type=Path, default=TRAIN_V06)
    tr.add_argument("--out", type=Path, required=True)
    tr.add_argument("--train-config", type=Path, default=TRAIN_CONFIG)
    tr.set_defaults(func=cmd_transform)
    ov = sub.add_parser("overlap", help="sealed checks (system1_v03 overlap + the iface sets' outside units)")
    ov.add_argument("--input", type=Path, required=True)
    ov.add_argument("--out-dir", type=Path, required=True)
    ov.set_defaults(func=cmd_overlap)
    ex = sub.add_parser("exposure", help="the trainer's view per seed vs v0.6, the shapes' exposure, tokens and time")
    ex.add_argument("--input", type=Path, required=True)
    ex.add_argument("--reference", type=Path, default=TRAIN_V06)
    ex.add_argument("--stored", type=Path, default=EXPOSURE_V06)
    ex.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    ex.add_argument("--train-config", type=Path, default=TRAIN_CONFIG)
    ex.add_argument("--no-tokens", action="store_true", help="skip the token simulation")
    ex.add_argument("--out", type=Path, required=True)
    ex.set_defaults(func=cmd_exposure)
    every = sub.add_parser("all", help="transform → overlap → exposure (one command)")
    every.add_argument("--input", type=Path, default=TRAIN_V06)
    every.add_argument("--train-config", type=Path, default=TRAIN_CONFIG)
    every.add_argument("--out-dir", type=Path, default=OUT_DIR)
    every.set_defaults(func=cmd_all)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
