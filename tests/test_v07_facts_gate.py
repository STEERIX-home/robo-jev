"""Task v0.7f — gate M2g-F (:mod:`robo_jev.bench.gates_v07f`; ``scripts/v07_gates_build.py build-facts`` / ``check-facts``) and its sealing.

Pinned on synthetic M2g-short records (no sealed file is read, except by the skip-if-absent checks of the built gate): the gate's own text is
free of fact and tone words and places the facts before, after and around the instruction; a facts record keeps everything of M2g-short but
the state, the instructions and three added ``meta.gate`` keys, holds every original sentence verbatim and in order in every question's
instructions and none in the state, and shows the same seeded option order; the checker catches a dropped, an altered and a moved sentence
(and a sentence left in the state, a changed instruction, label or criterion); both records of a pair get byte-identical added text; the
assignment is balanced per stratum (templates, status lines, cells) and depends on the key, its stratum and the seed only; the build writes a
sealed (0444) file with a manifest and refuses a broken input; check-facts catches a changed file, a skipped scan and a run trained on another
input; the trainer, the mix assembly and both API paths refuse M2g-F records.
"""

from __future__ import annotations

import copy
import importlib.util
import json
import stat
import sys
from collections import Counter
from pathlib import Path

import pytest
from helpers import FIXTURES, REPO

from robo_jev.bench import gates_v07 as G
from robo_jev.bench import gates_v07f as F
from robo_jev.data.sealed import NEVER_SENT_SETS, SEALED_RECORD_FILES, SEALED_RECORD_PREFIXES, refuse_sealed_records, sealed_record_set
from robo_jev.decisions.prompt import _listings
from robo_jev.decisions.records import file_sha256, read_records, shuffle_key, validate_record

REQUESTS = FIXTURES / "decisions" / "requests.jsonl"
TRAIN_CONFIG = REPO / "configs" / "train" / "decisions-v0-lora-h100.yaml"


def _script(name: str):
    spec = importlib.util.spec_from_file_location(f"v07facts_{Path(name).stem}", REPO / "scripts" / name)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


STATE_A = ("SUPERVISOR: Rover Six, Park depot. Where are you leaving the planter box?\nROBOT: On the slope below the bandstand. Ground is mossy "
           "stone flags, and my tilt sensor reads 19.5 degrees there.\nSUPERVISOR: What's in it?\nROBOT: It's half full of soil, settled at the "
           "bottom. 40 by 39 centimetres and 68 tall, 58 kilograms.")
STATE_B = STATE_A.replace("half full of soil, settled at the bottom", "packed full of soil, level with the top")
ONE_LINE = ("You are a yard robot with a dump bed in a factory. A speaker cabinet stands upright and loose at the centre of your deck: 0.69 m "
            "high, 14.0 kg. Packing: its packing is unknown. Your deck is covered with anodised aluminium plate. Your bed will tilt to 18.5° over 2.1 s.")


def _moment(rid: str, partner: str, state: str, p_yes: float, *, group: int = 7, scene: str = "bank", robust: bool = True) -> dict:
    return {
        "schema": "simgen-request/0", "id": rid, "generator": {"name": "test", "family": scene, "group": group},
        "meta": {"set": "v06-a4", "scene": scene, "pair": {"of": partner, "kind": "numeric"},
                 "robustness": {"moves": {"robust": robust}, "score_q": {"robust": False}}},
        "request": {"state": state, "questions": {
            "moves": {"type": "noul", "instructions": "Once the robot lets go, will the planter box move from where it was set?"},
            "score_q": {"type": "score", "instructions": "Rate the chance that the planter box moves if the grade were 15.5 degrees.",
                        "criteria": ["0 = it surely stays", "1 = it surely moves"]},
        }},
        "labels": {"moves": {"type": "noul", "p_yes": p_yes}, "score_q": {"type": "score", "value": 0.5}},
    }


def _short_pair(group: int = 7, scene: str = "bank", one_line: bool = False, source: str = "A4") -> list[dict]:
    """An M2g-short pair (task v0.7b's gate_record of a one-fact pair; labels differ)."""
    a_state = ONE_LINE if one_line else STATE_A
    b_state = ONE_LINE.replace("aluminium plate", "aluminium plate, still wet") if one_line else STATE_B
    base = f"v06a4-{scene}-{group:06d}"
    pair = [_moment(base, base + "p", a_state, 0.1, group=group, scene=scene), _moment(base + "p", base, b_state, 0.9, group=group, scene=scene)]
    return [G.gate_record(r, source=source, style="ticket") for r in pair]


@pytest.fixture()
def pair() -> list[dict]:
    return _short_pair()


# --------------------------------------------------------------------------
# the gate's own text
# --------------------------------------------------------------------------


def test_the_gate_text_is_free_of_fact_and_tone_words_and_places_the_facts_before_after_and_around():
    assert F.text_problems() == []
    assert len(F.STATUS_FORMS) >= 3 and len(F.TEMPLATES) >= 3
    assert {t["placement"] for t in F.TEMPLATES.values()} == {"before", "after", "around"}
    for line in F.STATUS_FORMS.values():
        assert G._word_hits(line, G.FACT_WORDS, G.FACT_STEMS) == [] and G._word_hits(line, G.TONE_WORDS, G.TONE_STEMS) == []
        assert len(line) <= 120 and "\n" not in line, "a status line is tiny and one line"
    pieces = F.fixed_pieces()
    assert {p["source"] for p in pieces} == {*(f"template:{t}" for t in F.TEMPLATES), *(f"status:{s}" for s in F.STATUS_FORMS)}
    assert {p["text"]: p["words"] for p in pieces if p["source"] == "template:tail"} == {"(Judge it from the unit's account:": 6, ")": 0}
    assert F.piece_words("desk=judgment | sender=deployed-unit") == 2 and F.piece_words("— :") == 0


def test_text_problems_catch_a_fact_word_a_tone_word_and_a_slot_glued_to_a_letter(monkeypatch):
    monkeypatch.setitem(F.STATUS_FORMS, "role", "Judgment desk: the load is fine.")
    found = F.text_problems()
    assert any("fact words" in p for p in found) and any("tone words" in p for p in found)
    monkeypatch.setitem(F.STATUS_FORMS, "role", "Judgment desk.")
    monkeypatch.setitem(F.TEMPLATES, "lead", {"placement": "before", "text": "Account:{facts}x Desk query: {instruction}"})
    assert any("touches a letter" in p for p in F.text_problems())
    monkeypatch.setitem(F.TEMPLATES, "lead", {"placement": "after", "text": "Account: {facts} Desk query: {instruction}"})
    assert any("does not match its slots" in p for p in F.text_problems())


# --------------------------------------------------------------------------
# records
# --------------------------------------------------------------------------


@pytest.mark.parametrize("template", list(F.TEMPLATES))
@pytest.mark.parametrize("status", list(F.STATUS_FORMS))
def test_a_facts_record_keeps_everything_but_the_state_and_the_instructions(pair, template, status):
    original = pair[0]
    before = copy.deepcopy(original)
    f = F.facts_record(original, template=template, status=status)
    assert F.facts_problems(f, original) == [] and validate_record(f) == []
    assert original == before, "the input is not modified"
    assert f["id"] == original["id"] and shuffle_key(f) == shuffle_key(original) and f["labels"] == original["labels"]
    assert f["request"]["state"] == F.STATUS_FORMS[status] and sealed_record_set(f) == "v07-m2g"
    assert f["meta"]["gate"] == {**original["meta"]["gate"], "condition": "facts", "template": template, "status": status}
    assert {k: v for k, v in f["meta"].items() if k != "gate"} == {k: v for k, v in original["meta"].items() if k != "gate"}
    units = G.sentences(original["request"]["state"])
    for name, q in f["request"]["questions"].items():
        old = original["request"]["questions"][name]
        assert {k: v for k, v in q.items() if k != "instructions"} == {k: v for k, v in old.items() if k != "instructions"}
        text = q["instructions"]
        assert text.count(old["instructions"]) == 1 and text == " ".join(text.split())
        positions = [text.find(u) for u in units]
        assert min(positions) >= 0 and positions == sorted(positions), "every sentence verbatim and in order"
        assert not any(u in f["request"]["state"] for u in units)
        # where the facts sit
        at = text.find(old["instructions"])
        placement = F.TEMPLATES[template]["placement"]
        if placement == "before":
            assert max(positions) < at
        elif placement == "after":
            assert min(positions) > at
        else:
            assert min(positions) < at < max(positions)
    assert " ".join(units) == " ".join(original["request"]["state"].split()), "the facts are the state with its line breaks shown as spaces"


def test_the_seeded_option_order_equals_the_short_forms(pair):
    for template in F.TEMPLATES:
        f = F.facts_record(pair[1], template=template, status="json")
        for name, q in pair[1]["request"]["questions"].items():
            settings = {"seed": 1, "orders": "seeded"}
            assert _listings(name, f["request"]["questions"][name], key=shuffle_key(f), **settings) == _listings(name, q, key=shuffle_key(pair[1]), **settings)


def test_split_cuts_after_half_the_lines_or_half_the_sentences():
    assert len(G.sentences(STATE_A)) == 7 and F.facts_split(STATE_A) == 4, "4 lines: the first two hold 4 sentences"
    assert len(G.sentences(ONE_LINE)) == 5 and F.facts_split(ONE_LINE) == 3, "one line of 5 sentences: the first 3"
    rendered = F.render("split", G.sentences(ONE_LINE), "Q?", 3)
    assert rendered.index("Desk query: Q?") > rendered.index("Packing: its packing is unknown.") and rendered.endswith("over 2.1 s.")
    with pytest.raises(ValueError):
        F.facts_split("One sentence only.")


def _mutate(record: dict, name: str, change) -> dict:
    out = copy.deepcopy(record)
    out["request"]["questions"][name]["instructions"] = change(out["request"]["questions"][name]["instructions"])
    return out


def test_the_checker_catches_a_dropped_an_altered_and_a_moved_sentence(pair):
    original = pair[0]
    units = G.sentences(original["request"]["state"])
    f = F.facts_record(original, template="lead", status="role")
    assert F.facts_problems(f, original) == []
    dropped = _mutate(f, "moves", lambda t: t.replace(units[2] + " ", ""))
    altered = _mutate(f, "moves", lambda t: t.replace("mossy stone flags", "mossy stone slabs"))
    moved = _mutate(f, "moves", lambda t: t.replace(units[0] + " ", "").replace(units[-1], units[-1] + " " + units[0]))
    for case in (dropped, altered, moved):
        found = F.facts_problems(case, original)
        assert any("not verbatim in order" in p for p in found), found
        assert any("template's rendering" in p for p in found)
    swapped = _mutate(f, "moves", lambda t: t.replace(units[1] + " " + units[2], units[2] + " " + units[1]))
    assert any("not verbatim in order" in p for p in F.facts_problems(swapped, original)), "two neighbours swapped"
    left = copy.deepcopy(f)
    left["request"]["state"] = f["request"]["state"] + " " + units[1]
    assert any("left in the state" in p for p in F.facts_problems(left, original))
    retold = _mutate(f, "score_q", lambda t: t.replace("15.5 degrees", "16 degrees"))
    assert any("instruction appears 0 times" in p for p in F.facts_problems(retold, original))
    relabelled = copy.deepcopy(f)
    relabelled["labels"]["moves"]["p_yes"] = 0.5
    assert any("labels differ" in p for p in F.facts_problems(relabelled, original))
    recriteria = copy.deepcopy(f)
    recriteria["request"]["questions"]["score_q"]["criteria"] = ["0 = no", "1 = yes"]
    assert any("other than the state" in p for p in F.facts_problems(recriteria, original))
    reid = copy.deepcopy(f)
    reid["id"] = "v07m2g-other"
    assert any("is not M2g-short's" in p for p in F.facts_problems(reid, original))
    toned = copy.deepcopy(f)
    toned["request"]["state"] = "All good: the load is fine."
    assert any("status line" in p for p in F.facts_problems(toned, original)) and any("tone words" in p for p in F.facts_problems(toned, original))


def test_both_records_of_a_pair_get_byte_identical_added_text(pair):
    for template in F.TEMPLATES:
        a, b = (F.facts_record(r, template=template, status="kv") for r in pair)
        assert a["request"]["state"] == b["request"]["state"]
        assert F.added_text(a, pair[0]) == F.added_text(b, pair[1]), template
        assert a["request"]["questions"]["moves"]["instructions"] != b["request"]["questions"]["moves"]["instructions"], "the facts differ"
    added = json.loads(F.added_text(F.facts_record(pair[0], template="split", status="kv"), pair[0]))
    assert added["questions"]["moves"].startswith("<S> <S> <S> <S> Desk query: <I> Rest of the unit's account: <S>")
    other = F.facts_record(pair[1], template="tail", status="kv")
    assert F.added_text(other, pair[1]) != F.added_text(F.facts_record(pair[0], template="lead", status="kv"), pair[0])


def test_the_assignment_is_balanced_per_stratum_and_depends_on_the_key_stratum_and_seed_only():
    strata = {"A4:bank": [f"a{i}" for i in range(33)], "A4:anchor": [f"b{i}" for i in range(34)], "B1:tower": [f"c{i}" for i in range(21)],
              "B1:grip": [f"d{i}" for i in range(22)], "H:x": [f"e{i}" for i in range(4)]}
    got = F.assign(strata, seed=1)
    assert got == F.assign({k: list(reversed(v)) for k, v in strata.items()}, seed=1), "order of the keys does not matter"
    assert got != F.assign(strata, seed=2)
    for keys in strata.values():
        for what, names in (("template", F.TEMPLATES), ("status", F.STATUS_FORMS)):
            counts = Counter(got[k][what] for k in keys)
            assert max(counts.get(n, 0) for n in names) - min(counts.get(n, 0) for n in names) <= 1, (what, counts)
        cells = Counter((got[k]["template"], got[k]["status"]) for k in keys)
        values = [cells.get((t, s), 0) for t in F.TEMPLATES for s in F.STATUS_FORMS]
        assert max(values) - min(values) <= 1, cells
    cells = F.latin_cells(["t0", "t1", "t2"], ["s0", "s1", "s2"])
    assert len(set(cells)) == 9 and all({c[0] for c in cells[i:i + 3]} == {"t0", "t1", "t2"} and len({c[1] for c in cells[i:i + 3]}) == 3 for i in (0, 3, 6))


def test_touching_shingles_hold_added_words_and_skip_text_of_the_moment_alone(pair):
    from robo_jev.data.general_sets import normalize

    f = F.facts_record(pair[0], template="lead", status="role")
    grams = F.touching_shingles(f, pair[0])
    added = set(normalize(" ".join(p["text"] for p in F.fixed_pieces() if p["source"] in ("template:lead", "status:role"))).split())
    assert grams and all(set(g.split()) & added for g in grams)
    assert all(len(g.split()) == 8 for g in grams)
    facts_only = " ".join(normalize(" ".join(G.sentences(pair[0]["request"]["state"]))).split()[:8])
    assert facts_only not in grams


# --------------------------------------------------------------------------
# build-facts / check-facts on a synthetic M2g-short
# --------------------------------------------------------------------------


def _synthetic_gates(tmp_path: Path) -> Path:
    """A gates folder with m2g/M2g-short.jsonl (a few strata of pairs) and its manifest; one other gate file to watch."""
    out = tmp_path / "gates"
    (out / "m2g").mkdir(parents=True)
    records = []
    for i in range(12):
        records += _short_pair(group=i, scene=("bank", "ledge")[i % 2], one_line=i % 3 == 0, source=("A4", "B1")[i % 2])
    short = out / "m2g" / "M2g-short.jsonl"
    short.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records), encoding="utf-8")
    (out / "m2g" / "m2g.manifest.json").write_text(json.dumps({"files": {"short": {"path": str(short), "sha256": file_sha256(short)}}}), encoding="utf-8")
    (out / "overlap.json").write_text("{}\n", encoding="utf-8")
    return out


def test_build_facts_writes_a_sealed_file_and_a_manifest_and_check_facts_reads_them(tmp_path, monkeypatch):
    script = _script("v07_gates_build.py")
    out = _synthetic_gates(tmp_path)
    watched = file_sha256(out / "overlap.json"), file_sha256(out / "m2g" / "M2g-short.jsonl")
    assert script.main(["build-facts", "--out-dir", str(out), "--skip-overlap", "--skip-tokens"]) == 0
    path = out / "m2gf" / F.FILE_NAME
    manifest = json.loads((out / "m2gf" / "manifest.json").read_text(encoding="utf-8"))
    assert stat.S_IMODE(path.stat().st_mode) == 0o444 and manifest["file"]["sha256"] == file_sha256(path) and manifest["file"]["records"] == 24
    assert (file_sha256(out / "overlap.json"), file_sha256(out / "m2g" / "M2g-short.jsonl")) == watched
    assert manifest["existing_gate_files_sha256"]["unchanged"] is True and "m2g/M2g-short.jsonl" in manifest["existing_gate_files_sha256"]["before"]
    assert manifest["proofs"]["pairs_added_text_identical"] == "12 of 12" and manifest["registry"]["name"] == "v07-m2g-facts"
    facts = read_records(path)
    assert [r["id"] for r in facts] == [r["id"] for r in read_records(out / "m2g" / "M2g-short.jsonl")]
    assert all(sealed_record_set(r) == "v07-m2g" for r in facts)
    # check-facts: a skipped overlap scan is not good enough to open the gate
    monkeypatch.setattr(script, "TRAIN_V07", "simgen/v07/absent-in-this-test.jsonl")
    problems = script.check_facts(out, [])
    assert any("overlap scan was skipped" in p for p in problems) and len(problems) == 1
    # with a zero-hit scan on record: the gate passes; a run trained on another input does not
    manifest["overlap"].update({"skipped": False, "pieces_with_exact_hits": 0, "shingles_found": 0, "train_v07": {"sha256": "a" * 64}})
    (out / "m2gf" / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    run = tmp_path / "run"
    run.mkdir()
    (run / "metrics.json").write_text(json.dumps({"manifest": {"data": {"file_sha256": "a" * 64}}}), encoding="utf-8")
    assert script.check_facts(out, [run]) == []
    assert script.main(["check-facts", "--out-dir", str(out), "--runs", str(run)]) == 0
    (run / "metrics.json").write_text(json.dumps({"manifest": {"data": {"file_sha256": "b" * 64}}}), encoding="utf-8")
    assert any("trained on" in p for p in script.check_facts(out, [run]))
    assert script.main(["check-facts", "--out-dir", str(out), "--runs", str(run)]) == 2
    # a changed facts file is caught
    path.chmod(0o644)
    rows = read_records(path)
    rows[0]["request"]["questions"]["moves"]["instructions"] += " "
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    assert any("not the sealed file" in p for p in script.check_facts(out, []))


def test_build_facts_refuses_an_input_that_is_not_the_sealed_short_or_not_whole_pairs(tmp_path):
    script = _script("v07_gates_build.py")
    out = _synthetic_gates(tmp_path)
    short = out / "m2g" / "M2g-short.jsonl"
    rows = read_records(short)
    short.write_text("".join(json.dumps(r) + "\n" for r in rows[:-1]), encoding="utf-8")
    with pytest.raises(SystemExit, match="not the sealed M2g-short"):
        script.main(["build-facts", "--out-dir", str(out), "--skip-overlap", "--skip-tokens"])
    with pytest.raises(SystemExit, match="whole one-fact pair"):
        script.build_facts(rows[:-1], seed=1)
    with pytest.raises(SystemExit, match="never written"):
        script.main(["build-facts", "--out-dir", str(REPO / "artifacts" / "iface"), "--skip-overlap", "--skip-tokens"])


def test_the_facts_scan_counts_pieces_and_shingles_and_keeps_no_text(tmp_path):
    script = _script("v07_gates_build.py")
    train = tmp_path / "train.jsonl"
    rows = [{"request": {"state": "Desk  query: hello\nworld", "questions": {"q": {"instructions": "the cat sat on the mat by the door today",
                                                                                    "criteria": {"a": "x"}}}}},
            {"request": {"state": "desk query: again", "questions": {}}}]
    train.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    found = script.facts_scan([train], ["Desk query:", "Rest of the unit's account:"], {"the cat sat on the mat by the", "never seen at all in this file x"})
    assert found["pieces"]["Desk query:"] == {"exact": 1, "folded": 2} and found["pieces"]["Rest of the unit's account:"] == {"exact": 0, "folded": 0}
    assert found["gram_hits"] == {"the cat sat on the mat by the": 1} and found["files"][str(train)]["records"] == 2
    assert set(found) == {"files", "pieces", "grams", "gram_hits", "distinct_texts"}


# --------------------------------------------------------------------------
# sealing
# --------------------------------------------------------------------------


def _facts_like(record: dict, *, mark: str = "meta") -> dict:
    out = copy.deepcopy(record)
    if mark == "meta":
        out["meta"] = {**(out.get("meta") or {}), "set": "v07-m2g", "gate": {"name": "M2g", "condition": "facts", "template": "lead", "status": "role"}}
    else:
        out["id"] = "v07m2g-" + str(out["id"])
    out["request"]["state"] = F.STATUS_FORMS["role"]
    return out


def test_the_facts_file_is_registered_and_its_records_are_sealed_and_never_sent(pair):
    assert SEALED_RECORD_FILES["v07-m2g-facts"] == "artifacts/v07/gates/m2gf/M2g-facts.jsonl"
    assert SEALED_RECORD_FILES["v07-m2g"] == "artifacts/v07/gates/m2g/M2g-all-conditions.jsonl", "M2g's own entry is unchanged"
    assert SEALED_RECORD_PREFIXES["v07-m2g"] == "v07m2g-" and "v07-m2g" in NEVER_SENT_SETS
    assert {"T9", "v06-a4", "v07-m1g"} <= set(NEVER_SENT_SETS), "earlier seals stay"
    f = F.facts_record(pair[0], template="tail", status="json")
    assert sealed_record_set(f) == "v07-m2g"
    with pytest.raises(ValueError, match="sealed outside test"):
        refuse_sealed_records([f], where="training input")


def test_the_trainer_refuses_a_facts_record():
    from robo_jev.decisions.train import DecisionTrainer

    records = read_records(REQUESTS)
    for mark in ("meta", "id"):
        with pytest.raises(ValueError, match="sealed outside test"):
            DecisionTrainer({}, records=[*records, _facts_like(records[0], mark=mark)], log=None)


def test_the_mix_assembly_refuses_a_facts_record_in_any_part(tmp_path):
    from test_v02_mix import _probe_and_sealed
    from test_v07_gates import _extra_part

    probe, sealed = _probe_and_sealed(tmp_path)
    records = read_records(REQUESTS)
    script = _script("convert_single_probe.py")
    common = ["mix", "--base", str(REQUESTS), "--probe", str(probe), "--sealed", str(sealed), "--train-config", str(TRAIN_CONFIG)]
    for i, poisoned in enumerate(([records[0], _facts_like(records[1])], [records[0], _facts_like(records[1], mark="id")])):
        part = _extra_part(tmp_path, f"facts{i}", poisoned)
        with pytest.raises(SystemExit, match="sealed outside test"):
            script.main([*common, "--extra", f"mujoco={part}", "--out", str(tmp_path / f"a{i}.jsonl")])


@pytest.mark.skipif(not (REPO / "artifacts" / "v07" / "gates" / "m2gf" / "M2g-facts.jsonl").is_file(), reason="the built gate is not here")
def test_the_mix_assembly_refuses_an_unmarked_record_that_shares_a_state_with_the_registered_facts_file(tmp_path):
    from test_v02_mix import _probe_and_sealed
    from test_v07_gates import _extra_part

    probe, sealed = _probe_and_sealed(tmp_path)
    records = read_records(REQUESTS)
    twin = copy.deepcopy(records[1])
    twin["request"]["state"] = F.STATUS_FORMS["kv"]
    part = _extra_part(tmp_path, "twin", [records[0], twin])
    script = _script("convert_single_probe.py")
    with pytest.raises(SystemExit, match="v07-m2g-facts"):
        script.main(["mix", "--base", str(REQUESTS), "--probe", str(probe), "--sealed", str(sealed), "--train-config", str(TRAIN_CONFIG),
                     "--extra", f"mujoco={part}", "--out", str(tmp_path / "b.jsonl")])


def test_both_api_paths_refuse_to_send_a_facts_record(pair):
    from robo_jev.e1 import rewrite as rw

    records = read_records(REQUESTS)
    index = rw.SealedIndex(set(), set(), {})
    rw.assert_sendable(records[0], [records[0]["request"]["state"]], index)
    f = F.facts_record(pair[0], template="split", status="kv")
    for marked in (f, _facts_like(records[0]), _facts_like(records[0], mark="id")):
        with pytest.raises(rw.SealedTextError, match="never sent"):
            rw.assert_sendable(marked, [marked["request"]["state"]], index)
    bench = _script("bench.py")
    bench.refuse_never_sent({"T2": [records[0]]})
    for marked in (f, _facts_like(records[1])):
        with pytest.raises(SystemExit, match="never sent"):
            bench.refuse_never_sent({"T2": [records[0], marked]})


# --------------------------------------------------------------------------
# the built gate (skipped where it is not built)
# --------------------------------------------------------------------------


@pytest.mark.skipif(not (REPO / "artifacts" / "v07" / "gates" / "m2gf" / "manifest.json").is_file(), reason="the built gate is not here")
def test_the_built_gate_matches_its_manifest_and_m2g_short():
    gates = REPO / "artifacts" / "v07" / "gates"
    manifest = json.loads((gates / "m2gf" / "manifest.json").read_text(encoding="utf-8"))
    path = REPO / manifest["file"]["path"]
    assert file_sha256(path) == manifest["file"]["sha256"] and stat.S_IMODE(path.stat().st_mode) == 0o444
    m2 = json.loads((gates / "m2g" / "m2g.manifest.json").read_text(encoding="utf-8"))
    assert manifest["source"]["sha256"] == m2["files"]["short"]["sha256"]
    facts, short = read_records(path), read_records(REPO / m2["files"]["short"]["path"])
    assert [r["id"] for r in facts] == [r["id"] for r in short] and len(facts) == 460
    assert all(sealed_record_set(r) == "v07-m2g" for r in facts)
    assert all(F.facts_problems(f, s) == [] for f, s in zip(facts, short))
    assert manifest["overlap"]["pieces_with_exact_hits"] == 0 and manifest["overlap"]["shingles_found"] == 0
    assert manifest["existing_gate_files_sha256"]["unchanged"] is True
