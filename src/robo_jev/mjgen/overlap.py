"""Zero-overlap check between the v0.3b training records and the sealed outside test T8b (Task v0.3b §1.6).

T8 stays an outside test, so the training families must not share anything with T8b but physics in general:

* **families** — scene names (``meta.scene``) disjoint, and question names disjoint;
* **text hashes** — sha256 of each request (state + questions, canonical JSON), each state text and each question instruction:
  no hash in common;
* **sentences** — every state and instruction split into sentences, compared after lower-casing and collapsing whitespace, and
  again with every number replaced by ``<n>`` (so the same template with other numbers counts as shared): none in common;
* **word n-grams** — lower-cased word n-grams (numbers → ``<n>``) for n = 4 … 8: the number of distinct shared n-grams and the
  share of our n-grams that occur in T8b. The verdict requires none shared for n ≥ 5 (4-grams are reported: short runs such as
  "at <n> m s" or "<n> kg its contents" are not wording);
* **catalog names** — object, surface, robot and option names of the two generators: none in common.

Only text and names are compared; nothing from T8b is copied into the training data.
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Iterable, Sequence

__all__ = ["NARRATIVE_FIELDS", "NGRAM_SIZES", "ngrams", "overlap_report", "scene_parameters", "sentences", "train_overlap_report"]

NGRAM_SIZES = (4, 5, 6, 8)
VERDICT_MIN_N = 5
_NUM = re.compile(r"[-+]?\d+(?:[.,]\d+)?")
_WORD = re.compile(r"<n>|[a-z]+(?:['’-][a-z]+)*")
_SPLIT = re.compile(r"(?<=[.!?;:])\s+|\n+")


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _norm(text: str, *, numbers: bool) -> str:
    out = " ".join(text.lower().split())
    return _NUM.sub("<n>", out) if numbers else out


def sentences(text: str) -> list[str]:
    return [part.strip() for part in _SPLIT.split(text) if part.strip()]


def ngrams(text: str, n: int) -> set[tuple[str, ...]]:
    words = _WORD.findall(_NUM.sub(" <n> ", text.lower()))
    return {tuple(words[i : i + n]) for i in range(len(words) - n + 1)}


def _texts(records: Sequence[dict[str, Any]]) -> tuple[list[str], list[str]]:
    states = [r["request"]["state"] for r in records]
    instructions = []
    for r in records:
        for q in r["request"]["questions"].values():
            instructions.append(q["instructions"])
            criteria = q.get("criteria")
            if isinstance(criteria, dict):
                instructions.extend(str(v) for v in criteria.values())
            elif isinstance(criteria, list):
                instructions.extend(str(v) for v in criteria)
    return states, instructions


def _request_hash(record: dict[str, Any]) -> str:
    return _sha(json.dumps(record["request"], ensure_ascii=False, sort_keys=True, separators=(",", ":")))


def _catalog_names() -> tuple[set[str], set[str]]:
    """Lower-cased names of objects, surfaces, robots and option keys in the T8b catalogs and in ours."""
    from robo_jev.mjgen import families as ours
    from robo_jev.outside import moments as t8

    theirs: set[str] = set()
    for load in t8.DECK_LOADS + t8.STACK_ITEMS + t8.PUSH_BOXES:
        theirs.add(load["name"].lower())
    for support in t8.STACK_SUPPORTS:
        theirs.update({support["name"].lower(), support["short"].lower()})
    theirs.update(v["text"].lower() for v in list(t8.DECK_SURFACES.values()) + list(t8.PUSH_FLOORS.values()))
    theirs.update(r.lower() for r in t8.DECK_ROBOTS + t8.STACK_ROBOTS + t8.PUSH_ROBOTS)
    mine: set[str] = set()
    for item in ours.TILT_LOADS + ours.GRIP_ITEMS + ours.TOWER_BOXES:
        mine.add(item["name"].lower())
    mine.update(v["text"].lower() for v in list(ours.TILT_SURFACES.values()) + list(ours.GRIP_PADS.values()) + list(ours.GRIP_SURFACES.values()))
    mine.update(r.lower() for robots in ours.TILT_ROBOTS.values() for r in robots)
    mine.update(r.lower() for r in ours.GRIP_ROBOTS + ours.TOWER_ROBOTS)
    return mine, theirs


def overlap_report(records: Sequence[dict[str, Any]], t8b: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Overlap of `records` (ours) with `t8b` (module docstring) → report with a verdict."""
    ours_states, ours_instr = _texts(records)
    their_states, their_instr = _texts(t8b)
    families = {"ours": sorted({r["meta"]["scene"] for r in records}), "t8b": sorted({r["meta"]["scene"] for r in t8b})}
    families["shared"] = sorted(set(families["ours"]) & set(families["t8b"]))
    qnames = {"ours": sorted({q for r in records for q in r["request"]["questions"]}), "t8b": sorted({q for r in t8b for q in r["request"]["questions"]})}
    qnames["shared"] = sorted(set(qnames["ours"]) & set(qnames["t8b"]))
    hashes = {
        "requests": len({_request_hash(r) for r in records} & {_request_hash(r) for r in t8b}),
        "states": len({_sha(s) for s in ours_states} & {_sha(s) for s in their_states}),
        "instructions": len({_sha(s) for s in ours_instr} & {_sha(s) for s in their_instr}),
    }
    ours_sent = [s for text in ours_states + ours_instr for s in sentences(text)]
    their_sent = [s for text in their_states + their_instr for s in sentences(text)]
    shared_exact = {_norm(s, numbers=False) for s in ours_sent} & {_norm(s, numbers=False) for s in their_sent}
    shared_template = {_norm(s, numbers=True) for s in ours_sent} & {_norm(s, numbers=True) for s in their_sent}
    grams: dict[str, Any] = {}
    their_all = their_states + their_instr
    ours_all = ours_states + ours_instr
    for n in NGRAM_SIZES:
        mine: set[tuple[str, ...]] = set()
        for text in ours_all:
            mine |= ngrams(text, n)
        theirs: set[tuple[str, ...]] = set()
        for text in their_all:
            theirs |= ngrams(text, n)
        shared = mine & theirs
        grams[str(n)] = {"ours": len(mine), "t8b": len(theirs), "shared": len(shared), "share_of_ours": len(shared) / len(mine) if mine else 0.0,
                         "examples": [" ".join(g) for g in sorted(shared)[:25]]}
    mine_names, their_names = _catalog_names()
    names_shared = sorted(mine_names & their_names)
    verdict = {
        "families_disjoint": not families["shared"],
        "question_names_disjoint": not qnames["shared"],
        "no_shared_text_hashes": not any(hashes.values()),
        "no_shared_sentences": not shared_exact and not shared_template,
        "no_shared_long_ngrams": all(grams[str(n)]["shared"] == 0 for n in NGRAM_SIZES if n >= VERDICT_MIN_N),
        "catalog_names_disjoint": not names_shared,
    }
    verdict["passed"] = all(verdict.values())
    return {
        "kind": "v03b-t8b-overlap", "records": len(records), "t8b_records": len(t8b), "verdict": verdict, "families": families, "question_names": qnames,
        "shared_hashes": hashes, "sentences": {"ours": len(set(map(lambda s: _norm(s, numbers=False), ours_sent))),
                                               "t8b": len(set(map(lambda s: _norm(s, numbers=False), their_sent))),
                                               "shared_exact": sorted(shared_exact)[:25], "shared_number_normalised": sorted(shared_template)[:25]},
        "ngrams": grams, "catalog_names": {"ours": len(mine_names), "t8b": len(their_names), "shared": names_shared},
    }


def t8b_texts_from_generator(per_family: int = 6, master: int = 20261012) -> list[dict[str, Any]]:
    """T8b-style records rebuilt from the T8 generator without MuJoCo (texts only; labels are placeholders) — lets the tests run the
    overlap check without the sealed artifacts file."""
    from robo_jev.outside import moments as t8

    out = []
    for family in t8.FAMILIES:
        for index in range(per_family):
            moment = t8.candidate(family, index, master)
            outcomes = t8.OUTCOMES[family]
            fake = {"k": 1, "draws_seed": 0, "majority": outcomes[0], "unstable": 0, "off_deck": 0,
                    "variants": {name: {"freq": {o: (1.0 if o == outcomes[0] else 0.0) for o in outcomes}, "n": {}, "max_tilt_mean": 0.0, "unstable": 0}
                                 for name in moment.variants}}
            out.append(t8.moment_record(moment, fake, number=index, mujoco_version="text-only"))
    return out


def iter_text(records: Iterable[dict[str, Any]]) -> Iterable[str]:
    states, instructions = _texts(list(records))
    yield from states
    yield from instructions


# --------------------------------------------------------------------------
# A fresh set from the same generator vs the training set (task B1)
# --------------------------------------------------------------------------

#: Stated facts that only name or place things in the story (robot, domain, storyline, object names); every other stated fact enters
#: the physics or the prior behind the words.
NARRATIVE_FIELDS = frozenset({"robot", "domain", "storyline", "load_name", "item_name", "name"})


def _strip(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _strip(v) for k, v in value.items() if k not in NARRATIVE_FIELDS}
    if isinstance(value, list):
        return [_strip(v) for v in value]
    return value


def scene_parameters(record: dict[str, Any]) -> str:
    """sha256 of a record's physical scene: the family, the stated facts without the narrative ones (:data:`NARRATIVE_FIELDS`) and the
    overrides of every variant it is labelled on (planned, counterfactual, action options) — two records with the same hash pose the
    same physics question under the same priors."""
    meta = record["meta"]
    payload = {"family": meta["scene"], "visible": _strip(meta["visible"]), "variants": meta.get("variants")}
    return _sha(json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str))


def train_overlap_report(records: Sequence[dict[str, Any]], train: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Overlap of a fresh set from the v0.3b generator (`records`, e.g. B1-infamily) with the training records (`train`): shared group
    seeds, hidden-draw seeds and (family, group index) pairs, shared physical scenes (:func:`scene_parameters`) and full stated facts,
    shared record ids, state texts and requests. The verdict requires none of them shared; question instructions (templated wording with
    the scene's numbers) are counted for information only."""

    def gen(record: dict[str, Any], key: str) -> Any:
        return (record.get("generator") or {}).get(key)

    def visible(record: dict[str, Any]) -> str:
        return _sha(json.dumps(record["meta"]["visible"], ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str))

    probes = {
        "group_seeds": lambda r: gen(r, "seed"),
        "draws_seeds": lambda r: gen(r, "draws_seed"),
        "family_group_index": lambda r: f"{gen(r, 'family')}:{gen(r, 'group')}",
        "scene_parameters": scene_parameters,
        "stated_facts": visible,
        "ids": lambda r: str(r["id"]),
        "state_texts": lambda r: _sha(r["request"]["state"]),
        "requests": _request_hash,
    }
    shared: dict[str, Any] = {}
    for name, probe in probes.items():
        ours, theirs = {probe(r) for r in records}, {probe(r) for r in train}
        common = ours & theirs
        shared[name] = {"ours": len(ours), "train": len(theirs), "shared": len(common), "examples": sorted(map(str, common))[:10]}
    ours_instr = {_sha(text) for text in _texts(records)[1]}
    train_instr = {_sha(text) for text in _texts(train)[1]}
    information = {"instructions": {"ours": len(ours_instr), "train": len(train_instr), "shared": len(ours_instr & train_instr),
                                    "share_of_ours": len(ours_instr & train_instr) / len(ours_instr) if ours_instr else 0.0}}
    verdict = {f"no_shared_{name}": entry["shared"] == 0 for name, entry in shared.items()}
    verdict["passed"] = all(verdict.values())
    return {"kind": "mjgen-train-overlap", "records": len(records), "train_records": len(train), "verdict": verdict, "shared": shared,
            "information": information}
