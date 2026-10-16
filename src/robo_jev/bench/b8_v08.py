"""Task v0.8b — gate B8 (fresh bench-shaped requests, the rule's P2) and the bench part of the Jev comparison set, pure parts.

The PRE-DECLARED v0.8 VERDICT RULE (progress.md, 2026-10-16 ~12:43 KST) reads P2 on gate B8: requests built by robo-jev-web's own builders
from fresh states — plates ``move``, café ``intent`` / ``attend`` / ``request`` and delivery ``manoeuvre``, pooled; a question is the unit
and a request the cluster; option tags frozen before scoring; the rel07 check's 30 requests excluded.

* **Build** (``artifacts/scratch/v08b/b8/build_requests.mts``, Node, the read-only clone at the commit it prints): each environment's
  ``planFor`` + ``stateSheet`` + ``buildQuestions`` on facts made with the fixture constructors of the repo's policy tests and filled
  with fresh situations from a seeded generator (every candidate reseeded from sha256(seed | env | template | index)). Candidates are
  deduplicated by their state's sha256. Tubes ``work`` requests (40+ options, the "real long lists" of the format readings) ride along
  in B8 with ``meta.p2 = False``.
* **Split** (:func:`split_rows`): per environment (café also per stratum: requests that carry the ``request`` question, and the rest),
  the candidates of each template are shuffled with a seeded generator and dealt round-robin over the templates; the first ones fill B8,
  the next the Jev bench part, the rest are spares (counted, never written). B8 and Jev therefore never share a state.
* **Records** (:func:`to_record`): ``simgen-request/0`` without labels — B8 ids ``v08b8-<env>-NNN`` (``meta.set`` ``v08-b8``, sealed:
  never trained on, never sent), Jev ids ``v08jev-b-<env>-NNN`` (``meta.set`` ``v08-jev``, ``meta.part`` ``bench`` — NOT sealed: sent to
  OpenRouter). ``meta``: env, target (the scored questions), situation, planner (the plan's recommendation; None for tubes, whose work
  question has none), available, builder, generator, p2, option_range (the v0.8 opt-in, :func:`robo_jev.bench.gates_v08.option_range_mark`).
* **Tags** (:func:`tag`, :data:`RULES`): the rel07 check's 38 ordered text rules verbatim first, then the rules this build had to add
  (ids ``v08-…``, the same classes): PASSIVE = hold / keep / wait / continue unchanged / do nothing / hand to a person; ACTIVE = a place,
  a target, a movement; first match wins; an option no rule matches stops the build. Mixed texts are ``borderline`` (main verb decides).
  :func:`tags_payload` writes rel07's format; :func:`freeze` writes ``tags.json`` and ``frozen-before-scoring.json`` (sha256 of B8.jsonl
  and tags.json) BEFORE any scoring and makes the three files read-only; :func:`check_frozen` is what the Spark gate and the verdict call.
* **Independence** (:func:`overlap_report`): identical states across B8, Jev and the rel07 check's requests; per request the largest
  line-level Jaccard similarity to any other set's request of the same environment; requests whose *dynamic* lines (those not shared by
  every request of the environment) equal another set's.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import re
import stat
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from robo_jev.bench.gates_v08 import ARTIFACTS, B8_DIR, B8_PREFIX, B8_SET, JEV_DIR, P2_TARGETS, PLANNED, option_range_mark, rng
from robo_jev.decisions.records import SCHEMA, validate_record

__all__ = [
    "B8_FILE_NAME", "FROZEN_NAME", "JEV_BENCH_FILE", "JEV_PART", "JEV_PREFIX", "JEV_SET", "REL07_REQUESTS", "REL07_RULES", "RULES", "SPLIT",
    "TAGS_NAME", "V08_RULES", "check_frozen", "dynamic_lines", "freeze", "jaccard", "lines_of", "overlap_report", "record_problems",
    "sha256_file", "sha256_text", "split_rows", "tag", "tags_payload", "to_record",
]

#: The Jev comparison set (not sealed): its bench part.
JEV_SET, JEV_PART, JEV_PREFIX = "v08-jev", "bench", "v08jev-b-"
JEV_BENCH_FILE = JEV_DIR / "jev-bench.jsonl"
B8_FILE_NAME, TAGS_NAME, FROZEN_NAME, MANIFEST_NAME = "B8.jsonl", "tags.json", "frozen-before-scoring.json", "b8.manifest.json"
#: The rel07 check's 30 requests (excluded from B8; the overlap is checked against them).
REL07_REQUESTS = ARTIFACTS / "scratch" / "rel07" / "bench-check" / "requests.jsonl"
REL07_TAGS = ARTIFACTS / "scratch" / "rel07" / "bench-check" / "tags.json"
#: The rel071 tube request (46 options) — compared too, though it was never part of the rel07 check.
REL071_TUBES = ARTIFACTS / "scratch" / "rel071" / "tubes" / "tubes-work-46.jsonl"
#: Requests per environment: (B8, Jev bench part). Café is split per stratum (requests carrying the ``request`` question first).
SPLIT = {"plates": (40, 25), "cafe": (30, 20), "delivery": (30, 20), "tubes": (6, 5)}
CAFE_REQUEST_SPLIT = (9, 6)
SPLIT_SALT = "v08b-b8-split"

#: The rel07 check's rules (artifacts/scratch/rel07/bench-check/tag_options.py, RULES) — verbatim, in order.
REL07_RULES: list[tuple[str, str, str, bool]] = [
    # PASSIVE: hold / keep / wait / continue unchanged / do nothing / hand to a person
    ("wait", "PASSIVE", r"^\s*wait\b", False),
    ("let-the-move-go-on", "PASSIVE", r"^Let the move under way go on", False),
    ("hold-still-for-a-look", "PASSIVE", r"^Hold still\b", True),
    ("hold-for-the-researcher", "PASSIVE", r"^Hold the plate you are working on for the researcher", False),
    ("stay", "PASSIVE", r"^Stay where you are", False),
    ("carry-on", "PASSIVE", r"^Carry on\b", False),
    ("stop-where-you-are", "PASSIVE", r"^stop where you are", False),
    ("call-the-researcher", "PASSIVE", r"^Call the researcher", False),
    ("call-the-operator", "PASSIVE", r"^call the remote operator", False),
    ("go-on-with-the-pass", "PASSIVE", r"^go on with the pass", True),
    ("look-ahead-standing-still", "PASSIVE", r"^The way ahead — standing still", True),
    ("look-round-the-room", "PASSIVE", r"^Look round the room", True),
    ("nothing-new", "PASSIVE", r"^Nothing new", False),
    ("something-the-cafe-does-not-make", "PASSIVE", r"^Something else, which the café does not make", True),
    ("not-made-out", "PASSIVE", r"^You could not make out", True),
    # ACTIVE: a place, a target, a movement
    ("go-above-a-station", "ACTIVE", r"^Go above ", False),
    ("go-to-rest", "ACTIVE", r"^Go to rest\b", True),
    ("go-to-a-person", "ACTIVE", r"^Go to P\d+", False),
    ("go-to-the-standby-spot", "ACTIVE", r"^Go to the standby spot", True),
    ("sweep", "ACTIVE", r"^Sweep\b", False),
    ("lower", "ACTIVE", r"^Lower\b", False),
    ("close-the-gripper", "ACTIVE", r"^Close the gripper", False),
    ("open-the-gripper", "ACTIVE", r"^Open the gripper", False),
    ("raise", "ACTIVE", r"^Raise\b", False),
    ("take", "ACTIVE", r"^Take\b", False),
    ("turn-to-heard", "ACTIVE", r"^Turn on the spot", False),
    ("step-back-out", "ACTIVE", r"^Step back\b", False),
    ("follow-the-route", "ACTIVE", r"^follow the route", False),
    ("step-aside-and-stop", "ACTIVE", r"^step 0\.6 m aside", True),
    ("back-up", "ACTIVE", r"^back up\b", False),
    ("pass", "ACTIVE", r"^pass (what is ahead|on the)", False),
    ("start-crossing", "ACTIVE", r"^start crossing", False),
    ("another-way", "ACTIVE", r"^take another way", False),
    ("a-person", "ACTIVE", r"^P\d+, \d", False),
    ("a-drink-from-the-menu", "ACTIVE", r"^The [a-z ]+, from the menu\.", False),
    ("water", "ACTIVE", r"^Water: a glass of water", False),
    ("napkins", "ACTIVE", r"^Napkins\.", False),
    ("the-bill", "ACTIVE", r"^The bill\.", False),
]
#: Rules this build added (appended after rel07's; the same classes). Only the tube cell's ``work`` options needed new rules — the plates,
#: café and delivery options of the new states all match rel07's rules.
V08_RULES: list[tuple[str, str, str, bool]] = [
    # PASSIVE
    ("v08-hold-this-second", "PASSIVE", r"^Hold this second\b", False),
    ("v08-go-on-with-the-work", "PASSIVE", r"^Go on with the work under way", True),
    ("v08-set-a-tube-aside", "PASSIVE", r"^Set t\d+\b.* aside in the exception rack", True),
    # ACTIVE
    ("v08-prepare-a-tube", "ACTIVE", r"^Prepare t\d+\b", False),
    ("v08-load-a-tube", "ACTIVE", r"^Load t\d+\b", False),
    ("v08-unload-a-tube", "ACTIVE", r"^Unload t\d+\b", False),
    ("v08-weigh-a-tube", "ACTIVE", r"^Weigh t\d+\b", False),
    ("v08-open-a-tube", "ACTIVE", r"^Open t\d+\b", False),
    ("v08-draw-from-a-tube", "ACTIVE", r"^Draw from t\d+\b", False),
    ("v08-close-a-tube", "ACTIVE", r"^Close t\d+\b", False),
    ("v08-archive-a-tube", "ACTIVE", r"^Archive t\d+\b", False),
    ("v08-open-the-lid", "ACTIVE", r"^Open the centrifuge's lid", False),
    ("v08-shut-the-lid", "ACTIVE", r"^Shut the centrifuge's lid", False),
]
RULES: list[tuple[str, str, str, bool]] = REL07_RULES + V08_RULES
COMPILED = [(rid, cls, re.compile(pat, re.IGNORECASE), borderline) for rid, cls, pat, borderline in RULES]
CLASSES = {"PASSIVE": "hold / keep / wait / continue unchanged / do nothing / hand to a person", "ACTIVE": "a place, a target, a movement"}


def tag(text: str) -> tuple[str, str, bool]:
    """(rule id, class, borderline) of an option's description — the first rule that matches; raises if none does."""
    for rid, cls, pattern, borderline in COMPILED:
        if pattern.search(str(text)):
            return rid, cls, borderline
    raise ValueError(f"no rule matches the option text: {text!r}")


def sha256_text(text: str) -> str:
    return hashlib.sha256(str(text).encode("utf-8")).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


# --------------------------------------------------------------------------
# Split
# --------------------------------------------------------------------------


def _deal(rows: Sequence[Mapping[str, Any]], *, seed: int, key: str) -> list[Mapping[str, Any]]:
    """The rows in a seeded, template-balanced order: each template's rows shuffled, then dealt round-robin over the templates (templates
    in sorted order)."""
    by_template: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_template[str(row["template"])].append(row)
    queues = []
    for template in sorted(by_template):
        group = sorted(by_template[template], key=lambda r: str(r["id"]))
        rng(SPLIT_SALT, seed, key, template).shuffle(group)
        queues.append(group)
    out: list[Mapping[str, Any]] = []
    while any(queues):
        for q in queues:
            if q:
                out.append(q.pop(0))
    return out


def split_rows(rows: Sequence[Mapping[str, Any]], *, seed: int, split: Mapping[str, tuple[int, int]] = SPLIT,
               cafe_request_split: tuple[int, int] = CAFE_REQUEST_SPLIT) -> dict[str, list[Mapping[str, Any]]]:
    """``{"b8": [...], "jev": [...], "spare": [...]}`` (module docstring). Raises if an environment has too few rows."""
    out: dict[str, list[Mapping[str, Any]]] = {"b8": [], "jev": [], "spare": []}
    for env, (n_b8, n_jev) in split.items():
        env_rows = [r for r in rows if r["env"] == env]
        strata: list[tuple[str, list[Mapping[str, Any]], int, int]]
        if env == "cafe":
            with_request = [r for r in env_rows if "request" in r["target"]]
            others = [r for r in env_rows if "request" not in r["target"]]
            strata = [("request", with_request, *cafe_request_split), ("other", others, n_b8 - cafe_request_split[0], n_jev - cafe_request_split[1])]
        else:
            strata = [("all", env_rows, n_b8, n_jev)]
        for name, stratum, b, j in strata:
            dealt = _deal(stratum, seed=seed, key=f"{env}/{name}")
            if len(dealt) < b + j:
                raise ValueError(f"{env}/{name}: {len(dealt)} candidates for {b} + {j}")
            out["b8"] += dealt[:b]
            out["jev"] += dealt[b:b + j]
            out["spare"] += dealt[b + j:]
    return out


# --------------------------------------------------------------------------
# Records
# --------------------------------------------------------------------------

_RANGE_PROBLEM = re.compile(r"meta\.option_range|choice needs \d+\.\.\d+ options")


def record_problems(record: Mapping[str, Any]) -> list[str]:
    """:func:`robo_jev.decisions.records.validate_record` without labels, ignoring ONLY its option-range problems (the 255 opt-in is task
    v0.8a's — main does not know it yet), plus this module's own check: every choice question has 2..255 options, the record carries the
    v0.8 opt-in, and every target question is a choice question of the request."""
    problems = [p for p in validate_record(dict(record), require_labels=False) if not _RANGE_PROBLEM.search(p)]
    questions = (record.get("request") or {}).get("questions") or {}
    for name, question in questions.items():
        if isinstance(question, dict) and question.get("type") == "choice" and not 2 <= len(question.get("criteria") or {}) <= 255:
            problems.append(f"question {name!r}: {len(question.get('criteria') or {})} options, outside 2..255")
    meta = record.get("meta") or {}
    if meta.get("option_range") != option_range_mark():
        problems.append(f"meta.option_range {meta.get('option_range')!r} is not the v0.8 opt-in {option_range_mark()!r}")
    for name in meta.get("target") or []:
        if (questions.get(name) or {}).get("type") != "choice":
            problems.append(f"target {name!r} is not a choice question of the request")
    return problems


def to_record(row: Mapping[str, Any], *, part: str, index: int) -> dict[str, Any]:
    """One builder row → a ``simgen-request/0`` record without labels (module docstring). `part` is ``b8`` or ``jev``; `index` counts
    within the environment (1-based)."""
    env = str(row["env"])
    if part == "b8":
        rid, meta_set = f"{B8_PREFIX}{env}-{index:03d}", B8_SET
    elif part == "jev":
        rid, meta_set = f"{JEV_PREFIX}{env}-{index:03d}", JEV_SET
    else:
        raise ValueError(f"part {part!r}: b8 or jev")
    meta: dict[str, Any] = {
        "set": meta_set, "env": env, "target": list(row["target"]), "situation": row["situation"], "planner": row.get("planner"),
        "available": list(row.get("available") or []), "builder": dict(row["builder"]), "generator": {**dict(row["generator"]), "candidate": row["id"]},
        "p2": env in P2_TARGETS, "option_range": option_range_mark(),
    }
    if part == "jev":
        meta["part"] = JEV_PART
    record = {"schema": SCHEMA, "id": rid, "meta": meta, "request": {"state": row["request"]["state"], "questions": row["request"]["questions"]}}
    problems = record_problems(record)
    if problems:
        raise ValueError(f"{rid} ({row['id']}): {problems[:3]}")
    return record


# --------------------------------------------------------------------------
# Tags, freeze, check
# --------------------------------------------------------------------------


def planned_question(record: Mapping[str, Any]) -> str | None:
    """The record's question that carries the planner's recommendation, if it is a target and the plan recommends (else None)."""
    meta = record["meta"]
    name = PLANNED.get(meta["env"])
    return name if name in meta["target"] and meta.get("planner") is not None else None


def tags_payload(records: Sequence[Mapping[str, Any]], *, b8_sha256: str, created: str) -> dict[str, Any]:
    """rel07's tags format for B8: every option of every target question tagged, the planner's recommendation recorded where present."""
    tagged, counts = [], Counter()
    for record in records:
        meta = record["meta"]
        entry = {"id": record["id"], "env": meta["env"], "p2": meta["p2"], "questions": {}}
        planned = planned_question(record)
        for name in meta["target"]:
            q = record["request"]["questions"][name]
            options = []
            for key, text in q["criteria"].items():
                rid, cls, borderline = tag(text)
                options.append({"key": key, "text": text, "tag": cls, "rule": rid, "borderline": borderline})
                counts[(meta["env"], cls)] += 1
            planner = meta["planner"] if name == planned else None
            if planner is not None and planner not in q["criteria"]:
                raise ValueError(f"{record['id']}: the planner's {planner!r} is not an option of {name}")
            entry["questions"][name] = {
                "n_options": len(options), "options": options, "planner": planner,
                "planner_tag": next((o["tag"] for o in options if o["key"] == planner), None),
                "passive": sum(o["tag"] == "PASSIVE" for o in options), "active": sum(o["tag"] == "ACTIVE" for o in options),
            }
        tagged.append(entry)
    return {
        "kind": "v08-b8-tags", "rule": "progress.md: PRE-DECLARED v0.8 VERDICT RULE (P2: passive-top share and planner agreement on gate B8)",
        "classes": CLASSES, "rules": [{"id": rid, "class": cls, "pattern": pat, "borderline": b, "origin": "v0.8b" if rid.startswith("v08-") else "rel07"}
                                      for rid, cls, pat, b in RULES],
        "b8_file": B8_FILE_NAME, "b8_sha256": b8_sha256, "created": created, "requests": tagged,
        "counts": {f"{env}/{cls}": n for (env, cls), n in sorted(counts.items())},
    }


def _write_readonly(path: Path, text: str) -> None:
    if path.exists():
        path.chmod(path.stat().st_mode | stat.S_IWUSR)
    path.write_text(text, encoding="utf-8")
    path.chmod(0o444)


def freeze(directory: str | Path, *, now: str | None = None, note: str | None = None) -> dict[str, Any]:
    """Tag B8 and freeze it BEFORE any scoring: ``tags.json`` and ``frozen-before-scoring.json`` (sha256 of B8.jsonl and tags.json); B8.jsonl,
    tags.json and the freeze record become read-only (0444). Returns the freeze record."""
    directory = Path(directory)
    b8 = directory / B8_FILE_NAME
    records = read_jsonl(b8)
    created = now or dt.datetime.now().astimezone().isoformat(timespec="seconds")
    payload = tags_payload(records, b8_sha256=sha256_file(b8), created=created)
    tags_path = directory / TAGS_NAME
    _write_readonly(tags_path, json.dumps(payload, ensure_ascii=False, indent=1) + "\n")
    b8.chmod(0o444)
    record = {"tags_sha256": sha256_file(tags_path), "b8_sha256": sha256_file(b8), "frozen": created,
              "note": note or "written before any scoring: the Spark gate and the verdict refuse if B8.jsonl or tags.json changed"}
    _write_readonly(directory / FROZEN_NAME, json.dumps(record, indent=1) + "\n")
    return record


def check_frozen(directory: str | Path = B8_DIR) -> list[str]:
    """Problems with the frozen B8 (empty = unchanged since it was frozen): missing files, a changed B8.jsonl or tags.json, tags made from
    another B8.jsonl, a tags file whose requests are not B8's."""
    directory = Path(directory)
    paths = {name: directory / name for name in (B8_FILE_NAME, TAGS_NAME, FROZEN_NAME)}
    missing = [name for name, path in paths.items() if not path.is_file()]
    if missing:
        return [f"missing {', '.join(missing)} in {directory}"]
    problems = []
    try:
        frozen = json.loads(paths[FROZEN_NAME].read_text(encoding="utf-8"))
        tags = json.loads(paths[TAGS_NAME].read_text(encoding="utf-8"))
    except ValueError as error:
        return [f"unreadable freeze or tags file: {error}"]
    b8_sha, tags_sha = sha256_file(paths[B8_FILE_NAME]), sha256_file(paths[TAGS_NAME])
    if frozen.get("b8_sha256") != b8_sha:
        problems.append(f"B8.jsonl changed after it was frozen ({b8_sha[:12]} vs frozen {str(frozen.get('b8_sha256'))[:12]})")
    if frozen.get("tags_sha256") != tags_sha:
        problems.append(f"tags.json changed after it was frozen ({tags_sha[:12]} vs frozen {str(frozen.get('tags_sha256'))[:12]})")
    if tags.get("b8_sha256") != b8_sha:
        problems.append("tags.json was made from another B8.jsonl")
    ids = [str(json.loads(line)["id"]) for line in paths[B8_FILE_NAME].read_text(encoding="utf-8").splitlines() if line.strip()]
    if [str(r.get("id")) for r in tags.get("requests") or []] != ids:
        problems.append("tags.json's requests are not B8.jsonl's (ids or order differ)")
    return problems


# --------------------------------------------------------------------------
# Independence
# --------------------------------------------------------------------------


def lines_of(state: str) -> set[str]:
    """The state's non-empty lines (stripped)."""
    return {line.strip() for line in str(state).splitlines() if line.strip()}


def jaccard(a: set[str], b: set[str]) -> float:
    return len(a & b) / len(a | b) if a or b else 1.0


def dynamic_lines(states: Mapping[str, str]) -> dict[str, frozenset[str]]:
    """Per request id, its lines minus the lines every request of the given set shares (the environment's boilerplate)."""
    sets = {rid: lines_of(s) for rid, s in states.items()}
    common = set.intersection(*sets.values()) if sets else set()
    return {rid: frozenset(lines - common) for rid, lines in sets.items()}


def _quantiles(values: Sequence[float]) -> dict[str, Any]:
    if not values:
        return {"n": 0}
    xs = sorted(values)
    q = lambda p: xs[min(len(xs) - 1, int(round(p * (len(xs) - 1))))]  # noqa: E731
    return {"n": len(xs), "min": round(xs[0], 4), "p25": round(q(0.25), 4), "median": round(q(0.5), 4), "p75": round(q(0.75), 4), "max": round(xs[-1], 4)}


def overlap_report(b8: Sequence[Mapping[str, Any]], jev: Sequence[Mapping[str, Any]], rel07: Sequence[Mapping[str, Any]],
                   extra: Mapping[str, Sequence[Mapping[str, Any]]] | None = None) -> dict[str, Any]:
    """The independence proofs (module docstring). `b8` / `jev` are records; `rel07` the rel07 check's rows (``env``, ``request``);
    `extra` more named reference sets of rows (the rel071 tube request)."""
    def env_of(r: Mapping[str, Any]) -> str:
        return str((r.get("meta") or {}).get("env") or r.get("env"))

    sets: dict[str, list[Mapping[str, Any]]] = {"b8": list(b8), "jev": list(jev), "rel07": list(rel07), **{k: list(v) for k, v in (extra or {}).items()}}
    hashes = {name: {sha256_text(r["request"]["state"]) for r in rows} for name, rows in sets.items()}
    names = list(sets)
    identical = {f"{a}∩{b}": len(hashes[a] & hashes[b]) for i, a in enumerate(names) for b in names[i + 1:]}
    within = {name: len(rows) - len(hashes[name]) for name, rows in sets.items()}
    out: dict[str, Any] = {"identical_states": identical, "duplicate_states_within": within, "by_env": {}}
    envs = sorted({env_of(r) for rows in sets.values() for r in rows})
    for env in envs:
        rows = {name: [r for r in sets[name] if env_of(r) == env] for name in names}
        states = {f"{name}:{r['id']}": r["request"]["state"] for name in names for r in rows[name]}
        dyn = dynamic_lines(states)
        lines = {key: lines_of(s) for key, s in states.items()}
        entry: dict[str, Any] = {"requests": {name: len(rows[name]) for name in names}, "boilerplate_lines": len(set.intersection(*lines.values())) if lines else 0}
        for a, b in (("b8", "rel07"), ("b8", "jev"), ("jev", "rel07"), *(("b8", k) for k in (extra or {})), *(("jev", k) for k in (extra or {}))):
            if not rows[a] or not rows[b]:
                continue
            keys_a, keys_b = [f"{a}:{r['id']}" for r in rows[a]], [f"{b}:{r['id']}" for r in rows[b]]
            best = [max(jaccard(lines[x], lines[y]) for y in keys_b) for x in keys_a]
            best_dyn = [max(jaccard(set(dyn[x]), set(dyn[y])) for y in keys_b) for x in keys_a]
            same_dyn = sum(1 for x in keys_a if any(dyn[x] == dyn[y] for y in keys_b))
            entry[f"{a}_vs_{b}"] = {"max_line_jaccard": _quantiles(best), "max_dynamic_line_jaccard": _quantiles(best_dyn),
                                    "same_dynamic_lines": same_dyn, "identical_states": sum(1 for x in keys_a if any(states[x] == states[y] for y in keys_b))}
        out["by_env"][env] = entry
    return out


def describe(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Counts per environment, per target question and option count, the planner's recommendations, the templates."""
    out: dict[str, Any] = {"requests": len(records), "by_env": {}}
    for env in sorted({r["meta"]["env"] for r in records}):
        rows = [r for r in records if r["meta"]["env"] == env]
        questions = Counter((name, len(r["request"]["questions"][name]["criteria"])) for r in rows for name in r["meta"]["target"])
        out["by_env"][env] = {
            "requests": len(rows), "target_questions": dict(Counter(name for r in rows for name in r["meta"]["target"])),
            "options": {f"{name}/{n}": c for (name, n), c in sorted(questions.items())},
            "planner": dict(Counter(str(r["meta"]["planner"]) for r in rows).most_common()),
            "templates": dict(Counter(r["meta"]["generator"]["template"] for r in rows).most_common()),
            "state_chars": _quantiles([len(r["request"]["state"]) for r in rows]),
            "over_26_options": sum(1 for r in rows for name in r["meta"]["target"] if len(r["request"]["questions"][name]["criteria"]) > 26),
        }
    return out


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]], *, readonly: bool = False) -> str:
    """Write rows as JSONL (one line each, keys in insertion order) and return the file's sha256."""
    text = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if readonly:
        _write_readonly(path, text)
    else:
        if path.exists():
            path.chmod(path.stat().st_mode | stat.S_IWUSR)
        path.write_text(text, encoding="utf-8")
    return sha256_file(path)


def node_version() -> str | None:
    return os.environ.get("NODE_VERSION")
