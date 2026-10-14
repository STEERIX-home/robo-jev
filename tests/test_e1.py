"""Task E1 (the pre-declared 2×2): C rendering, B checks and sealed guard, A question filter, analysis and the rule. No network, no GPU."""

import json
import re

import numpy as np
import pytest

from robo_jev.decisions.records import validate_record
from robo_jev.e1 import analysis as an
from robo_jev.e1 import render as rd
from robo_jev.e1 import rewrite as rw
from robo_jev.outside import moments as mo


def _fake_label(moment):
    outcomes = mo.OUTCOMES[moment.family]
    variants = {}
    for i, name in enumerate(moment.variants):
        freq = {o: 0.0 for o in outcomes}
        freq[outcomes[(i + moment.index) % len(outcomes)]] = 1.0
        variants[name] = {"freq": freq, "n": {o: int(4 * f) for o, f in freq.items()}, "max_tilt_mean": 0.0, "unstable": 0}
    return {"k": 4, "draws_seed": 0, "variants": variants, "majority": max(outcomes, key=lambda o: variants["planned"]["freq"][o]), "unstable": 0,
            "off_deck": 0}


@pytest.fixture(scope="module")
def t8b_like():
    out = []
    for family in mo.FAMILIES:
        for index in range(12):
            moment = mo.candidate(family, index, 777)
            out.append(mo.moment_record(moment, _fake_label(moment), number=index, mujoco_version="test"))
    return out


# --------------------------------------------------------------------------
# C and D
# --------------------------------------------------------------------------


def test_kept_questions_are_the_system1_t8b_events():
    from robo_jev.bench.system1 import T8B_EVENTS

    assert rd.KEPT_QUESTIONS == T8B_EVENTS


def test_c_keeps_every_visible_value_and_every_d_number(t8b_like):
    for record in t8b_like:
        c = rd.render_c(record)
        assert rd.check_c(c, record) == [], record["id"]
        assert validate_record(c) == []
        assert list(c["request"]["questions"]) == list(rd.KEPT_QUESTIONS[record["meta"]["scene"]])
        assert c["labels"] == {n: record["labels"][n] for n in rd.KEPT_QUESTIONS[record["meta"]["scene"]]}
        assert json.dumps(rd.render_c(record)) == json.dumps(c)  # deterministic


def test_c_is_second_person_engine_prose(t8b_like):
    for record in t8b_like:
        c = rd.render_c(record)
        state = c["request"]["state"]
        assert state.startswith(rd.SETTINGS[record["meta"]["visible"]["domain"]])
        assert "You are" in state and "\n\n" in state
        for q in c["request"]["questions"].values():
            assert not re.search(r"\b(I|me|my)\b", q["instructions"])


def test_check_c_catches_a_dropped_value(t8b_like):
    record = next(r for r in t8b_like if r["meta"]["scene"] == "deck_brake")
    c = rd.render_c(record)
    decel = rd._acc(record["meta"]["visible"]["decel"])
    c["request"]["state"] = c["request"]["state"].replace(decel, "a firm rate")
    c["request"]["questions"]["falls_over"]["instructions"] = c["request"]["questions"]["falls_over"]["instructions"].replace(decel, "as planned")
    assert any("decel" in p for p in rd.check_c(c, record))
    c2 = rd.render_c(record)
    c2["request"]["state"] = c2["request"]["state"].replace(rd.FILLS[record["meta"]["visible"]["fill"]], "")
    assert any("fill" in p for p in rd.check_c(c2, record))


def test_d_is_t8b_restricted(t8b_like):
    for record in t8b_like:
        d = rd.restrict_d(record)
        assert d["id"] == record["id"] and d["request"]["state"] == record["request"]["state"]
        assert set(d["request"]["questions"]) == set(rd.KEPT_QUESTIONS[record["meta"]["scene"]])
        assert "keeps_place" not in d["request"]["questions"] and "what_happens" not in d["request"]["questions"]


def test_quantities_reads_engine_units():
    found = rd.quantities("at 1.76 m/s, 3.89 m/s², 47 cm, 0.094 s, 90°, 3.7 kg, 0.25 m")
    assert [(d, round(v, 6)) for d, v, _, _ in found] == [("speed", 1.76), ("acc", 3.89), ("length", 0.47), ("time", 0.094), ("angle", 90.0),
                                                          ("mass", 3.7), ("length", 0.25)]


# --------------------------------------------------------------------------
# B
# --------------------------------------------------------------------------

A_RECORD = {
    "id": "e1a-000001", "schema": "simgen-request/0", "meta": {"scene": "p3_carrying_tall"},
    "request": {"state": "You are moving at 1.80 m/s. A stack of crates stands on a rubber mat on your deck. It is 87 cm tall; latency 200 ms; "
                         "a 45° turn ahead. The room is kept at 21 °C. A 10-minute break starts soon.",
                "questions": {"topples_hard_stop": {"type": "noul", "instructions": "If you brake as hard as you can (2.4 m/s²), will the stack of crates topple over?"}}},
    "labels": {"topples_hard_stop": {"type": "noul", "p_yes": 0.3}},
}
GOOD = {"state": "I am moving at 1.80 m/s. A stack of crates stands on a rubber mat on my deck. It is 87 cm tall; my latency is 200 ms; a 45° turn is "
                 "ahead of me. The room is kept at 21 °C. A 10 minute break starts soon.",
        "questions": {"topples_hard_stop": "If I brake as hard as I can (2.4 m/s²), will the stack of crates topple over?"}}
OBJECTS = ["stack of crates", "rubber mat"]


def test_number_tokens():
    tokens = rw.number_tokens("1.80 m/s and 1.8 m/s, 45°, 21 °C, 10-minute, 12 minutes, racks 14 and 15, 3.4 m/s² and 2.0 m/s^2")
    assert tokens == {("1.80", "m/s"): 1, ("1.8", "m/s"): 1, ("45", "°"): 1, ("21", "°C"): 1, ("10", "minutes"): 1, ("12", "minutes"): 1, ("14", ""): 1,
                      ("15", ""): 1, ("3.4", "m/s²"): 1, ("2.0", "m/s²"): 1}


def test_check_b_passes_a_faithful_rewrite():
    assert rw.check_b(A_RECORD, GOOD, OBJECTS) == {}


@pytest.mark.parametrize("change, kind", [
    (lambda r: r["state"].replace("1.80 m/s", "1.8 m/s"), "numbers"),
    (lambda r: r["state"].replace("200 ms", "0.2 s"), "numbers"),
    (lambda r: r["state"] + " It may tip in 2 s.", "numbers"),
    (lambda r: r["state"].replace("rubber mat", "mat"), "objects"),
    (lambda r: r["state"].replace("I am moving", "You are moving"), "voice"),
])
def test_check_b_catches_changes(change, kind):
    reply = {"state": change(GOOD), "questions": dict(GOOD["questions"])}
    assert kind in rw.check_b(A_RECORD, reply, OBJECTS)


def test_check_b_checks_questions_and_keys():
    reply = {"state": GOOD["state"], "questions": {"topples_hard_stop": "If you brake as hard as you can (2.4 m/s²), will the stack of crates topple over?"}}
    assert "voice" in rw.check_b(A_RECORD, reply, OBJECTS)
    reply = {"state": GOOD["state"], "questions": {"topples_hard_stop": "If I brake hard, will the stack of crates topple over?"}}
    assert "numbers" in rw.check_b(A_RECORD, reply, OBJECTS)
    reply = {"state": GOOD["state"], "questions": {"other": GOOD["questions"]["topples_hard_stop"]}}
    assert "keys" in rw.check_b(A_RECORD, reply, OBJECTS)


def test_parse_reply_and_b_record():
    text = "```json\n" + json.dumps(GOOD) + "\n```"
    reply = rw.parse_reply(text)
    assert reply == GOOD
    with pytest.raises(ValueError):
        rw.parse_reply('{"state": "x"}')
    b = rw.b_record(A_RECORD, reply, model="m", objects=OBJECTS, request_sha256="0" * 64, resolved="m-1")
    assert b["id"] == "e1b-000001" and b["labels"] == A_RECORD["labels"] and b["request"]["questions"]["topples_hard_stop"]["type"] == "noul"
    assert validate_record(b) == []


def test_messages_carry_only_the_request():
    messages = rw.build_messages({**A_RECORD, "meta": {"scene": "p3", "secret": "x"}})
    payload = json.loads(messages[1]["content"])
    assert set(payload) == {"state", "questions"} and "labels" not in messages[1]["content"] and "e1a-000001" not in messages[1]["content"]


def test_sealed_guard(tmp_path, t8b_like):
    sealed = tmp_path / "T8b.jsonl"
    sealed.write_text("\n".join(json.dumps(r) for r in t8b_like[:3]) + "\n", encoding="utf-8")
    engine = tmp_path / "T1.jsonl"
    engine.write_text(json.dumps(A_RECORD) + "\n", encoding="utf-8")
    index = rw.SealedIndex.build({"T8b": sealed, "T1": engine}, outside=("T8b",))
    with pytest.raises(rw.SealedTextError):
        rw.assert_sendable(t8b_like[0], [t8b_like[0]["request"]["state"]], index)
    with pytest.raises(rw.SealedTextError):  # a sealed engine state
        rw.assert_sendable({**A_RECORD, "id": "e1a-000009"}, [A_RECORD["request"]["state"]], index)
    sentence = t8b_like[1]["request"]["state"].split(". ")[1] + "."
    fresh = {**A_RECORD, "id": "e1a-000010", "request": {**A_RECORD["request"], "state": "You are new here. " + sentence}}
    with pytest.raises(rw.SealedTextError):  # one sentence of an outside set
        rw.assert_sendable(fresh, [fresh["request"]["state"]], index)
    ok = {**A_RECORD, "id": "e1a-000011", "request": {**A_RECORD["request"], "state": "You are moving at 1.10 m/s with a crate on your deck."}}
    rw.assert_sendable(ok, [ok["request"]["state"]], index)
    # every never-sent sealed set is refused by id prefix or meta.set, not only T9 (v0.6's held-out author A4)
    for sealed_record in ({**ok, "id": "v06a4-bank-000001"}, {**ok, "id": "e1a-000012", "meta": {"set": "v06-a4"}}):
        with pytest.raises(rw.SealedTextError):
            rw.assert_sendable(sealed_record, [ok["request"]["state"]], index)


# --------------------------------------------------------------------------
# A
# --------------------------------------------------------------------------


def test_a_records_keep_only_tip_fall_questions():
    from robo_jev.e1.engine import P3_SCENES, SEED_START, TIP_FALL_EVENTS, generate_a, plan_a, question_event
    from robo_jev.simgen.catalog import default_config

    cfg = default_config()
    units = plan_a(cfg, 1)
    assert [u["scene"] for u in units] == list(P3_SCENES) and units[0]["seed"] == SEED_START
    records, report = generate_a(cfg, units, git="test")
    assert len(records) == 4 and report["replaced_seeds"] == []
    for record in records:
        assert record["meta"]["language"] == "en" and record["meta"]["split"] == "train" and record["meta"]["scene"] in P3_SCENES
        assert "phrasing" not in record["meta"] and "fact_order" not in record["meta"]
        assert record["request"]["questions"] and all(question_event(record, n) in TIP_FALL_EVENTS for n in record["request"]["questions"])
        assert set(record["labels"]) == set(record["request"]["questions"])
        assert validate_record(record) == []


# --------------------------------------------------------------------------
# Analysis and rule
# --------------------------------------------------------------------------


def _items(n_moments=60, noise=0.0, seed=0, flip=False):
    rng = np.random.default_rng(seed)
    out = []
    for m in range(n_moments):
        for g in ("q1", "q2"):
            y = float(rng.uniform())
            p = (1 - y if flip else y) + float(rng.normal(0, noise))
            out.append((f"m{m}", ("fam", g), y, p))
    out.append(("m0", ("fam", "rare"), 0.5, 0.1))  # a group with n < 20 is not used
    return out


def test_pearson_and_pooling():
    assert an.pearson([1, 2, 3], [2, 4, 6]) == pytest.approx(1.0)
    assert an.pearson([1, 1, 1], [1, 2, 3]) is None
    assert an.pearson([1, 2], [1, 2]) is None
    reading = an.situation_reading(_items())
    assert reading["r"] == pytest.approx(1.0) and reading["groups_used"] == 2 and "fam|rare" not in reading["groups"]
    reading = an.situation_reading(_items(flip=True))
    assert reading["r"] == pytest.approx(-1.0)


def test_paired_bootstrap():
    same = an.paired_bootstrap(_items(), _items(), n_boot=200, seed=1)
    assert same["delta"]["value"] == pytest.approx(0.0) and same["delta"]["low"] == pytest.approx(0.0) and same["delta"]["high"] == pytest.approx(0.0)
    worse = an.paired_bootstrap(_items(), _items(noise=0.5, seed=0), n_boot=300, seed=1)
    assert worse["delta"]["value"] < -0.2 and worse["delta"]["high"] < 0
    assert worse["x"]["low"] <= worse["x"]["r"] <= worse["x"]["high"]
    with pytest.raises(ValueError):
        an.paired_bootstrap(_items(), _items(n_moments=50), n_boot=10)


@pytest.mark.parametrize("de, de_hi, dt, dt_lo, rc, verdict", [
    (-0.30, -0.15, 0.30, 0.15, 0.40, "H1"),
    (-0.30, -0.05, 0.30, 0.15, 0.40, "INCONCLUSIVE"),     # Δengine interval not below −0.10
    (-0.30, -0.15, 0.15, 0.05, 0.40, "INCONCLUSIVE"),     # Δt8b too small
    (-0.05, 0.05, 0.05, -0.05, 0.10, "H2"),
    (-0.15, -0.05, 0.05, -0.05, 0.10, "INCONCLUSIVE"),    # Δengine between −0.20 and −0.10
    (-0.25, -0.12, 0.05, -0.05, 0.10, "BOTH"),
    (-0.05, 0.05, 0.05, -0.05, 0.25, "INCONCLUSIVE"),     # r_C not < 0.20
])
def test_rule(de, de_hi, dt, dt_lo, rc, verdict):
    out = an.apply_rule({"value": de, "low": de - 0.1, "high": de_hi}, {"value": dt, "low": dt_lo, "high": dt + 0.1}, rc)
    assert out["verdict"] == verdict


def test_rule_overlap_is_flagged_not_resolved():
    out = an.apply_rule({"value": -0.30, "low": -0.4, "high": -0.15}, {"value": 0.25, "low": 0.12, "high": 0.4}, 0.15)
    assert out["holds"] == ["H1", "BOTH"] and "overlap" in out["verdict"]
