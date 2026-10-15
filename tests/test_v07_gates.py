"""Task v0.7b — the sealed v0.7 shape gates M2g / M1g (:mod:`robo_jev.bench.gates_v07`, ``scripts/v07_gates_build.py``) and their sealing.

Pinned on synthetic records (no sealed file is read): every original sentence stays verbatim, once, as the whole report section of every sheet
style, length and tone; the two records of a one-fact pair (different labels) get the same filler; the three tone versions differ only in the
tone sentences; a longer sheet extends the shorter one; no fact word in the filler and no tone word outside the tone slot (and the checks
catch each violation); styles are a function of the pair key and the seed only, balanced per stratum; the M1g filters, the answer-like
physics distractors, the nested listings at N = 8, 21, 26 and the per-item readings; the gates are registered as sealed sets so the trainer,
the mix assembly and the API paths refuse them.
"""

from __future__ import annotations

import copy
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest
from helpers import FIXTURES, REPO

from robo_jev.bench import gates_v07 as G
from robo_jev.bench import iface
from robo_jev.data.sealed import NEVER_SENT_SETS, SEALED_RECORD_FILES, SEALED_RECORD_PREFIXES, SEALED_RECORD_SETS, refuse_sealed_records, sealed_record_set
from robo_jev.decisions.records import read_records, shuffle_key, validate_record

REQUESTS = FIXTURES / "decisions" / "requests.jsonl"
TRAIN_CONFIG = REPO / "configs" / "train" / "decisions-v0-lora-h100.yaml"


def _script(name: str):
    spec = importlib.util.spec_from_file_location(f"v07gates_{Path(name).stem}", REPO / "scripts" / name)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


STATE_A = ("ROBOT: Site control, Unit Four, ready to set the crate on the bank.\nSUPERVISOR: Go ahead. What's the grade?\n"
           "ROBOT: 24 degrees, wet clay. It is 0.4 m wide and 12.5 kg. I'll leave it there.")
STATE_B = STATE_A.replace("24 degrees", "31 degrees")


def _moment(rid: str, partner: str, state: str, p_yes: float, *, robust: bool = True) -> dict:
    return {
        "schema": "simgen-request/0", "id": rid, "generator": {"name": "test", "family": "bank", "group": 7},
        "meta": {"set": "v06-a4", "scene": "bank", "pair": {"of": partner, "kind": "numeric"},
                 "robustness": {"moves": {"robust": robust}, "score_q": {"robust": False}, "dir_q": {"robust": True}},
                 "question_kind": {"moves": "key", "score_q": "score_cf", "dir_q": "direction"}},
        "request": {"state": state, "questions": {
            "moves": {"type": "noul", "instructions": "Will the crate move after release?"},
            "score_q": {"type": "score", "instructions": "Rate the chance it moves at 30 degrees.", "criteria": ["0 = it stays", "1 = it moves"]},
            "dir_q": {"type": "choice", "instructions": "Steeper: more or less likely?", "criteria": {"up": "More likely", "down": "Less likely", "same": "About as likely"}},
        }},
        "labels": {"moves": {"type": "noul", "p_yes": p_yes}, "score_q": {"type": "score", "value": 0.5},
                   "dir_q": {"type": "choice", "kind": "direction", "probabilities": {"up": 1.0, "down": 0.0, "same": 0.0}}},
    }


@pytest.fixture()
def pair() -> list[dict]:
    return [_moment("v06a4-bank-000007", "v06a4-bank-000007p", STATE_A, 0.1),
            _moment("v06a4-bank-000007p", "v06a4-bank-000007", STATE_B, 0.9)]


# --------------------------------------------------------------------------
# M2g records and sheets
# --------------------------------------------------------------------------


def test_sentences_split_lines_and_full_stops_but_not_decimals():
    assert G.sentences("A 0.4 m box. It is 12.5 kg.\nB: Go ahead. What's next?") == ["A 0.4 m box.", "It is 12.5 kg.", "B: Go ahead.", "What's next?"]


def test_a_gate_record_keeps_the_call_view_questions_and_labels_and_shares_the_pair_order_key(pair):
    gated = [G.gate_record(r, source="A4", style="ticket") for r in pair]
    a, b = gated
    assert a["id"] == "v07m2g-v06a4-bank-000007" and a["meta"]["pair"]["of"] == "v07m2g-v06a4-bank-000007p"
    assert a["meta"]["set"] == G.M2G_SET and a["meta"]["source"] == {"set": "A4", "id": "v06a4-bank-000007", "meta_set": "v06-a4"}
    assert list(a["request"]["questions"]) == ["moves", "score_q"] and a["request"]["questions"]["moves"] == pair[0]["request"]["questions"]["moves"]
    assert a["labels"] == {k: pair[0]["labels"][k] for k in ("moves", "score_q")} and a["request"]["state"] == pair[0]["request"]["state"]
    assert shuffle_key(a) == shuffle_key(b) == G.M2G_PREFIX + shuffle_key(pair[0])
    assert validate_record(a) == [] and pair[0]["meta"]["set"] == "v06-a4", "the original is not modified"


@pytest.mark.parametrize("style", G.STYLES)
def test_every_sheet_keeps_the_state_verbatim_once_shares_the_filler_in_a_pair_and_nests_across_lengths(pair, style):
    gated = [G.gate_record(r, source="A4", style=style) for r in pair]
    key = shuffle_key(gated[0])
    reference = max(len(r["request"]["state"]) for r in pair)
    sheets: dict = {}
    for record in gated:
        for length, target in G.LENGTHS.items():
            for tone in G.TONES:
                sheet = G.wrap_state(record["request"]["state"], style=style, key=key, target=target, reference_length=reference, tone=tone)
                assert G.sheet_problems(sheet, record["request"]["state"], style=style, key=key, tone=tone) == []
                assert sheet.count(record["request"]["state"]) == 1 and target - 150 <= len(sheet) <= target
                sheets[(record["id"], length, tone)] = sheet
    for length in G.LENGTHS:  # the same filler for both records of the pair (their labels differ)
        fillers = {sheets[(r["id"], length, "neutral")].replace(r["request"]["state"], "<STATE>") for r in gated}
        assert len(fillers) == 1
    for record in gated:  # the tone versions differ only in the three tone sentences
        assert len({G.tone_skeleton(sheets[(record["id"], "4k", t)], key, t) for t in G.TONES}) == 1
        for short, long in (("2k", "4k"), ("4k", "8k")):  # a longer sheet extends the shorter one
            assert all(unit in sheets[(record["id"], long, "neutral")] for unit in G.sentences(sheets[(record["id"], short, "neutral")]))


def test_the_sheet_checks_catch_each_violation(pair):
    record = G.gate_record(pair[0], source="A4", style="ticket")
    key, state = shuffle_key(record), record["request"]["state"]
    sheet = G.wrap_state(state, style="ticket", key=key, target=4000, tone="neutral")
    assert G.sheet_problems(sheet, state, style="ticket", key=key, tone="neutral") == []
    assert any("fact words" in p for p in G.sheet_problems(sheet.replace("Reviewer: unassigned.", "Reviewer: the crate may slide."), state,
                                                            style="ticket", key=key, tone="neutral"))
    assert any("tone words" in p for p in G.sheet_problems(sheet.replace("Reviewer: unassigned.", "Reviewer: all good."), state, style="ticket",
                                                            key=key, tone="neutral"))
    assert any("appears 0 times" in p for p in G.sheet_problems(sheet.replace("What's the grade?", "What is the grade?"), state, style="ticket",
                                                                 key=key, tone="neutral"))
    assert any("appears 2 times" in p for p in G.sheet_problems(sheet + "\n\n" + state, state, style="ticket", key=key, tone="neutral"))
    assert any("tone sentences" in p for p in G.sheet_problems(sheet, state, style="ticket", key=key, tone="alarming"))


def test_tone_banks_are_parallel_and_free_of_fact_words_and_the_neutral_bank_of_tone_words():
    lengths = {t: [len(s) for s in G.TONE_BANKS[t]] for t in G.TONES}
    assert all(len(v) == len(lengths["neutral"]) for v in lengths.values())
    assert max(abs(a - b) for t in ("reassuring", "alarming") for a, b in zip(lengths[t], lengths["neutral"])) <= 15
    for tone, bank in G.TONE_BANKS.items():
        assert G._word_hits(" ".join(bank), G.FACT_WORDS, G.FACT_STEMS) == [], tone
    assert G._word_hits(" ".join(G.TONE_BANKS["neutral"]), G.TONE_WORDS, G.TONE_STEMS) == []
    assert all(G._word_hits(s, G.TONE_WORDS, G.TONE_STEMS) for t in ("reassuring", "alarming") for s in G.TONE_BANKS[t])
    picks = {t: [G.TONE_BANKS[t].index(s) for s in G.tone_lines("k", t)] for t in G.TONES}
    assert picks["neutral"] == picks["reassuring"] == picks["alarming"] and len(set(picks["neutral"])) == G.TONE_SENTENCES


def test_word_matching_is_by_whole_word_with_plain_endings():
    assert G._word_hits("Planner input and the handover", G.FACT_WORDS, G.FACT_STEMS) == []
    assert G._word_hits("it slipped and the plan moved", G.FACT_WORDS, G.FACT_STEMS) == ["moved", "plan", "slipped"]


def test_styles_are_balanced_within_each_stratum_and_depend_on_the_keys_and_seed_only():
    strata = {"A4:bank": [f"k{i}" for i in range(9)], "Hset:tray": [f"h{i}" for i in range(7)]}
    styles = G.assign_styles(strata, seed=1)
    assert styles == G.assign_styles({k: list(reversed(v)) for k, v in strata.items()}, seed=1), "order of the keys does not matter"
    for keys in strata.values():
        counts = [sum(styles[k] == s for k in keys) for s in G.STYLES]
        assert max(counts) - min(counts) <= 1
    assert styles != G.assign_styles(strata, seed=2)


# --------------------------------------------------------------------------
# M1g
# --------------------------------------------------------------------------


def _row(i: int, question: str, choices: list[str], answer: int = 0, subject: str = "sociology") -> dict:
    return {"id": f"mmlu-test-{i:05d}", "subject": subject, "question": question, "choices": choices, "answer": answer}


def test_mmlu_filters_drop_numbers_negations_option_references_long_texts_and_the_excluded_subject():
    good = ["the state provides public services", "markets fail to clear quickly", "firms collude on prices", "labour unions gain power"]
    rows = [
        _row(0, "What does collective consumption refer to?", good),
        _row(1, "Which is NOT a cause?", good),
        _row(2, "How many?", ["one apple a day", "two apples a day", "three apples a day", "four apples a day"]),
        _row(3, "Which statements hold?", ["I and III only", "II only here", "all of them now", "the first one only"]),
        _row(4, "Which hold?", ["(D) Increased Stayed the same", "b b b", "c c c", "d d d"]),
        _row(5, "x " * 200, good),
        _row(6, "Which is wrong?", good, subject="moral_scenarios"),
        _row(7, "Which word fits?", ["energy", "heat", "light", "sound"]),
        _row(8, "What does collective consumption refer to?", good),
        _row(9, "Intergovernmental organizations SELDOM do which of the following?", good),
        _row(10, "Which proposition would be rejected by critical legal theorists?", good),
        _row(11, "Which holds?", ["Nether A nor B", "the first option here", "the second option here", "the third option here"]),
    ]
    items, dropped = G.mmlu_items(rows)
    assert [i["item"] for i in items] == ["mmlu:mmlu-test-00000", "mmlu:mmlu-test-00007"]
    assert items[0]["target"] is True and items[1]["target"] is False, "a bare-noun answer stays only in the drawing pool"
    assert items[0]["state"] == 'question: "What does collective consumption refer to?"' and items[0]["instructions"] == iface.ARC_INSTRUCTION
    assert dropped == {"negative_question": 3, "number_in_an_option": 1, "option_refers_to_options": 3, "stem_empty_or_over_300_chars": 1,
                       "excluded_subject": 1, "duplicate_question": 1}


def _physics_item(keys=("up", "down", "same")) -> dict:
    return {"item": "a4:r1:dir", "source": "a4", "group": "g1", "keys": sorted(keys), "correct": ("up", "More likely to move"),
            "own": [("down", "Less likely to move"), ("same", "About as likely")], "scene": "bank", "state": "s", "instructions": "q"}


def test_physics_distractors_never_state_a_direction_or_an_equality_nor_reuse_a_key():
    own = {"no_difference": [("r2", "g2", "About the same")], "tips": [("r2", "g2", "Tips over")], "alt": [("r3", "g3", "the alternative")],
           "plan": [("r2", "g2", "The plan as stated"), ("r1", "g1", "From the item's own pair")], "up": [("r2", "g2", "More likely to tip")]}
    sibling = {"more_likely": [("h1", "h", "more likely than as planned")], "skids": [("h1", "h", "it skidded 2 cm or more but is upright")],
               "secure": [("h2", "h2", "still where it was between the palms")]}
    drawn = G.draw_physics(_physics_item(), own, sibling, count=4, seed=0)
    texts = [d for _, d in drawn]
    assert sorted(k for k, _ in drawn) == ["plan", "secure", "skids", "tips"] and "From the item's own pair" not in texts
    assert not any(G.PHYSICS_ANSWERLIKE.search(t) for t in texts)
    with pytest.raises(ValueError, match="distinct distractor keys"):
        G.draw_physics(_physics_item(), own, sibling, count=5, seed=0)


def _general_item() -> dict:
    return {"item": "mmlu:x", "source": "MMLU", "subject": "s", "state": 'question: "q?"', "instructions": iface.ARC_INSTRUCTION, "stem": "q?",
            "correct": "the right answer", "own": ["wrong one here", "wrong two here", "wrong three here"]}


@pytest.mark.parametrize("make", [_general_item, _physics_item])
def test_listings_are_nested_across_n_with_the_same_answer_and_controlled_positions(make):
    item = make()
    distractors = [f"drawn option {i}" for i in range(22)] if item["source"] == "MMLU" else [(f"k{i}", f"drawn option {i}") for i in range(23)]
    records = G.expand_item(item, distractors, rank=5, index=3, seed=0)
    assert len(records) == 9 and {r["meta"]["n"] for r in records} == set(G.M1G_LEVELS)
    assert all(r["id"].startswith(G.M1G_PREFIX) and r["meta"]["set"] == G.M1G_SET and sealed_record_set(r) == "v07-m1g" for r in records)
    answers = {r["request"]["questions"]["q"]["criteria"][r["meta"]["correct"]] for r in records}
    assert answers == {item["correct"] if isinstance(item["correct"], str) else item["correct"][1]}
    sets = {n: set(r["request"]["questions"]["q"]["criteria"].values()) for n in G.M1G_LEVELS for r in records if r["meta"]["n"] == n}
    assert sets[8] <= sets[21] <= sets[26]
    for n in (21, 26):
        assert sorted(r["meta"]["bucket"] for r in records if r["meta"]["n"] == n) == ["A-H", "I-P", "Q-Z"]
    for r in records:
        criteria = r["request"]["questions"]["q"]["criteria"]
        assert list(criteria).index(r["meta"]["correct"]) == r["meta"]["position"] and r["meta"]["letter"] == chr(65 + r["meta"]["position"])
        assert iface.m1_problems(r, lift_cap=True) == [] and bool(iface.m1_problems(r, lift_cap=False)) == (r["meta"]["n"] > 8)


def test_the_per_item_table_reads_drop_and_penalty_from_n_8_21_26_and_ignores_other_levels():
    rows = []
    for n, values in ((8, (1, 1, 0)), (21, (1, 0, 0)), (26, (1, 0, 1)), (12, (0,))):
        for listing, correct in enumerate(values):
            bucket = iface.bucket_of({0: 2, 1: 10, 2: 18}[listing], n) if n >= 17 else None
            rows.append({"item": "i", "n": n, "listing": listing, "correct": float(correct), "p_correct": 0.5, "bucket": bucket})
    t = G.m1g_item_table(rows, ["i"])
    assert t["acc@8"][0] == pytest.approx(2 / 3) and t["acc@late"][0] == pytest.approx((1 / 3 + 2 / 3) / 2)
    assert t["drop"][0] == pytest.approx(2 / 3 - 0.5) and t["penalty"][0] == pytest.approx(1.0 - 0.5) and t["complete"][0] == 1.0


# --------------------------------------------------------------------------
# Sealing
# --------------------------------------------------------------------------


def _m2g_like(record: dict, *, mark: str = "meta") -> dict:
    out = copy.deepcopy(record)
    if mark == "meta":
        out["meta"] = {**(out.get("meta") or {}), "set": "v07-m2g"}
    else:
        out["id"] = "v07m2g-" + str(out["id"])
    return out


def test_both_gates_are_registered_sealed_never_sent_sets_with_their_files_and_prefixes():
    for name, prefix, path in (("v07-m2g", "v07m2g-", "artifacts/v07/gates/m2g/M2g-all-conditions.jsonl"),
                               ("v07-m1g", "v07m1g-", "artifacts/v07/gates/m1g/M1g.jsonl")):
        assert name in SEALED_RECORD_SETS and name in NEVER_SENT_SETS and SEALED_RECORD_PREFIXES[name] == prefix and SEALED_RECORD_FILES[name] == path
    assert {"T9", "v04a-scene-holdout", "v06-a4"} <= set(SEALED_RECORD_SETS) and {"T9", "v06-a4"} <= set(NEVER_SENT_SETS), "earlier seals stay"
    assert (G.M2G_SET, G.M2G_PREFIX, G.M1G_SET, G.M1G_PREFIX) == ("v07-m2g", "v07m2g-", "v07-m1g", "v07m1g-")


def test_a_gate_record_is_recognised_by_its_set_or_prefix_and_refused_by_the_trainer():
    from robo_jev.decisions.train import DecisionTrainer

    records = read_records(REQUESTS)
    assert sealed_record_set(records[0]) is None
    for mark in ("meta", "id"):
        assert sealed_record_set(_m2g_like(records[0], mark=mark)) == "v07-m2g"
    assert sealed_record_set(dict(records[0], id="v07m1g-00001-n08-A")) == "v07-m1g"
    with pytest.raises(ValueError, match="sealed outside test"):
        refuse_sealed_records([records[0], dict(records[1], id="v07m1g-00001-n08-A")], where="training input")
    with pytest.raises(ValueError, match="sealed outside test"):
        DecisionTrainer({}, records=[*records, _m2g_like(records[0], mark="id")], log=None)


def _extra_part(tmp_path: Path, name: str, records: list[dict]) -> Path:
    """Valid train-split records for an ``--extra`` part (fresh ids, as the MuJoCo set's — a gate id is kept)."""
    out = []
    for i, record in enumerate(records):
        record = json.loads(json.dumps(record))
        record["id"] = record["id"] if str(record["id"]).startswith(("v07m2g-", "v07m1g-")) else f"x{name}-{i:04d}"
        record["meta"] = {**(record.get("meta") or {}), "split": "train", "language": "en"}
        out.append(record)
    path = tmp_path / f"{name}.jsonl"
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in out), encoding="utf-8")
    return path


def test_the_mix_assembly_refuses_a_gate_record_in_any_part(tmp_path):
    from test_v02_mix import _probe_and_sealed

    probe, sealed = _probe_and_sealed(tmp_path)
    records = read_records(REQUESTS)
    script = _script("convert_single_probe.py")
    common = ["mix", "--base", str(REQUESTS), "--probe", str(probe), "--sealed", str(sealed), "--train-config", str(TRAIN_CONFIG)]
    for i, poisoned in enumerate(([records[0], _m2g_like(records[1])], [records[0], dict(records[1], id="v07m1g-00002-n26-Q")])):
        part = _extra_part(tmp_path, f"poisoned{i}", poisoned)
        with pytest.raises(SystemExit, match="sealed outside test"):
            script.main([*common, "--extra", f"mujoco={part}", "--out", str(tmp_path / f"a{i}.jsonl")])


def test_the_api_paths_refuse_to_send_a_gate_record():
    from robo_jev.e1 import rewrite as rw

    records = read_records(REQUESTS)
    index = rw.SealedIndex(set(), set(), {})
    rw.assert_sendable(records[0], [records[0]["request"]["state"]], index)
    for marked in (_m2g_like(records[0]), _m2g_like(records[0], mark="id"), dict(records[0], id="v07m1g-00003-n21-B")):
        with pytest.raises(rw.SealedTextError, match="never sent"):
            rw.assert_sendable(marked, [marked["request"]["state"]], index)
    bench = _script("bench.py")
    bench.refuse_never_sent({"T2": [records[0]]})
    with pytest.raises(SystemExit, match="never sent"):
        bench.refuse_never_sent({"T2": [records[0], _m2g_like(records[1])]})


def test_the_build_script_never_writes_into_the_sealed_or_earlier_folders():
    script = _script("v07_gates_build.py")
    script.refuse_write(REPO / "artifacts" / "v07" / "gates" / "x.json")
    for bad in ("t9/T9.jsonl", "v06/a4/x.json", "iface/x.json", "simgen/v07/train-v07.jsonl", "b1/x.json", "t8/x.json"):
        with pytest.raises(SystemExit, match="never written"):
            script.refuse_write(REPO / "artifacts" / bad)


@pytest.mark.skipif(not (REPO / "artifacts" / "v07" / "gates" / "m1g" / "m1g.manifest.json").is_file(), reason="the built gates are not here")
def test_the_built_gates_are_sealed_records_and_match_their_manifests():
    from robo_jev.decisions.records import file_sha256

    gates = REPO / "artifacts" / "v07" / "gates"
    m1 = json.loads((gates / "m1g" / "m1g.manifest.json").read_text(encoding="utf-8"))
    m2 = json.loads((gates / "m2g" / "m2g.manifest.json").read_text(encoding="utf-8"))
    assert file_sha256(REPO / m1["file"]["path"]) == m1["file"]["sha256"]
    assert file_sha256(REPO / m2["registry_file"]["path"]) == m2["registry_file"]["sha256"]
    short = read_records(REPO / m2["files"]["short"]["path"])
    assert all(sealed_record_set(r) == "v07-m2g" for r in short) and len({r["id"] for r in short}) == len(short)
    with (REPO / m1["file"]["path"]).open(encoding="utf-8") as handle:
        first = [json.loads(next(handle)) for _ in range(20)]
    assert all(sealed_record_set(r) == "v07-m1g" for r in first)
    assert np.isclose(sum(m2["styles"].values()), m2["records"]) and set(m2["styles"]) == set(G.STYLES)
