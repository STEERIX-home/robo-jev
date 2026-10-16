"""Task v0.8b — the v0.8 verdict helper (scripts/verdict_v08.py): the PRE-DECLARED v0.8 VERDICT RULE at every bound, each outcome and next
step, and synthetic predictions through the whole computation (no real v0.8 prediction exists; nothing here reads one)."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]


def _load():
    spec = importlib.util.spec_from_file_location("verdict_v08_under_test", REPO / "scripts" / "verdict_v08.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


V = _load()


# --------------------------------------------------------------------------
# The rule at its bounds (pure)
# --------------------------------------------------------------------------


def test_p1_bounds():
    assert V.p1_check({"value": -0.05, "high": -1e-9})["pass"]
    assert not V.p1_check({"value": -0.0499, "high": -1e-9})["pass"]
    assert not V.p1_check({"value": -0.2, "high": 0.0})["pass"]
    assert V.p1_check({"value": -0.2, "high": -1e-9})["pass"]
    assert not V.p1_check({"value": -0.2})["pass"]
    assert not V.p1_check(None)["pass"]


def test_p2_bounds():
    assert V.p2_check({"passive_top": {"value": -0.15}, "planner": {"value": -0.05}})["pass"]
    assert not V.p2_check({"passive_top": {"value": -0.1499}, "planner": {"value": 0.3}})["pass"]
    assert not V.p2_check({"passive_top": {"value": -0.5}, "planner": {"value": -0.0501}})["pass"]
    assert not V.p2_check({"passive_top": {"value": -0.5}})["pass"]
    assert not V.p2_check(None)["pass"]


def test_outcomes():
    assert V.outcome_of(True, True) == "SUPPORTED"
    assert V.outcome_of(True, False) == "PARTIAL"
    assert V.outcome_of(False, True) == "PARTIAL"
    assert V.outcome_of(False, False) == "REJECTED"


GOOD_GUARDS = {"CA": {"value": -0.05}, "GAP": {"value": -0.03}, "band": {"value": 0.0, "low": -0.0499}, "severe": {"a": 0.11, "b": 0.10},
               "G2": {"diff": -0.02, "low": -0.0449}}


def test_guards_at_their_bounds():
    assert V.guard_checks(GOOD_GUARDS)["pass"]
    for key, bad, check in (("CA", {"value": -0.0501}, "outside_CA_mean_ge_-0.05"), ("GAP", {"value": -0.0301}, "outside_GAP_mean_ge_-0.03"),
                            ("band", {"low": -0.05}, "T2-T4_band_low_gt_-0.05"), ("severe", {"a": 0.1101, "b": 0.10}, "T2-T4_severe_FP_le_F_plus_0.01"),
                            ("G2", {"diff": -0.0201, "low": 0.0}, "G2_point_ge_-0.02"), ("G2", {"diff": 0.0, "low": -0.045}, "G2_low_gt_-0.045")):
        result = V.guard_checks({**GOOD_GUARDS, key: bad})
        assert not result["pass"] and not result["checks"][check], key
    for key in GOOD_GUARDS:
        assert not V.guard_checks({k: v for k, v in GOOD_GUARDS.items() if k != key})["pass"], key


def _fmt(two_hi=0.4, single_hi=0.5, two_lo=0.4, single_lo=0.5, n26=-0.02):
    return {"bins": {"27-60": {"acc_two": {"value": two_lo}, "acc_single": {"value": single_lo}},
                     "61-255": {"acc_two": {"value": two_hi}, "acc_single": {"value": single_hi}}}, "n26": {"value": n26}}


def test_format_ratio_and_n26_bounds():
    assert V.format_check(_fmt())["pass"]  # 0.4 = 0.8 × 0.5 exactly, in both bins; N ≤ 26 at −0.02
    assert not V.format_check(_fmt(two_lo=0.3999))["pass"]
    assert not V.format_check(_fmt(two_hi=0.3999))["pass"]
    assert not V.format_check(_fmt(n26=-0.0201))["pass"]
    assert V.format_check(_fmt(two_hi=0.0, single_hi=0.0))["pass"]  # 0 ≥ 0.8 × 0 (no division)
    assert not V.format_check({"n26": {"value": 0.0}})["pass"]
    assert not V.format_check(None)["pass"]
    # float rounding of an exact tie must not flip an inclusive bound: 0.8 × 0.3 is 0.24000000000000002 in binary
    assert V.format_check(_fmt(two_lo=0.24, single_lo=0.3))["pass"]
    assert V.format_check(_fmt(n26=0.70 - 0.72))["pass"]  # −0.020000000000000018 is the tie −0.02


def test_of_bound():
    assert V.of_check({"value": 0.02})["pass"]
    assert not V.of_check({"value": 0.0201})["pass"]
    assert not V.of_check(None)["pass"]


def _flips(**over) -> dict:
    """F's flip rate per source of the amended condition, every source exactly at the bound unless overridden."""
    return {**{name: 0.20 for name in V.CONSISTENCY_SOURCES}, **over}


def test_consistency_condition_at_its_bound_for_each_source():
    for source in (*V.CONSISTENCY_SOURCES, V.R14):
        flips = {**_flips(), V.R14: 0.20}
        c = V.consistency_check(flips, r14_provided=True)
        assert c["status"] == "NOT TRIGGERED" and not c["triggered"] and c["unread"] == [], source  # exactly 0.20 does not trigger
        flips[source] = 0.2000001
        c = V.consistency_check(flips, r14_provided=True)
        assert c["status"] == "TRIGGERED" and c["over"] == [source], source
    assert V.CONSISTENCY_SOURCES == ("M8g 2-10", "M8g 11-26", "M8g 27-60", "M8g 61-255", "B8")


def test_consistency_condition_without_r14_and_with_unread_sources():
    c = V.consistency_check(_flips(), r14_provided=False)
    assert c["sources"] == list(V.CONSISTENCY_SOURCES) and c["status"] == "NOT TRIGGERED" and "M8g and B8 only" in c["r14"]
    assert not V.consistency_check({**_flips(), V.R14: 0.9}, r14_provided=False)["triggered"]  # R14 absent: not a source
    c = V.consistency_check({k: v for k, v in _flips().items() if k != "B8"}, r14_provided=False)
    assert c["status"] == "UNREAD" and c["unread"] == ["B8"]  # an unread source is never counted as <= 0.20
    c = V.consistency_check({**_flips(**{"M8g 61-255": None}), "B8": 0.5}, r14_provided=True)
    assert c["status"] == "TRIGGERED" and c["over"] == ["B8"] and c["unread"] == ["M8g 61-255", "R14"]
    c = V.consistency_check(_flips(**{"M8g 11-26": None}), r14_provided=False)
    assert c["status"] == "UNREAD" and c["unread"] == ["M8g 11-26"]


def test_next_steps_each_branch():
    quiet = V.consistency_check(_flips(), r14_provided=False)
    loud = V.consistency_check(_flips(B8=0.5), r14_provided=False)
    s = V.next_steps(True, True, True, quiet, 0.1)
    assert s["branches"] == ["SUPPORTED"] and "NOT TRIGGERED" in s["notes"][0] and "would not add" in s["notes"][1]
    s = V.next_steps(False, True, True, loud, 0.1)
    assert s["branches"] == ["P1 fails", V.CONSISTENCY_BRANCH]
    assert "System-0 margins" in s["next"][0] and "order-consistency loss" in s["next"][1]
    assert "would not add" in s["notes"][1]  # the original wording (bins >= 27 only) would not fire here
    s = V.next_steps(True, False, False, loud, 0.9)  # the step fires whatever FORMAT does; FORMAT's own step is added
    assert s["branches"] == ["P2 fails while P1 passes", "FORMAT fails", V.CONSISTENCY_BRANCH]
    assert "same-family" in s["next"][0] and "random-code share" in s["next"][1] and "FORMAT FAIL" in s["notes"][1]
    s = V.next_steps(True, True, True, quiet, 0.25)
    assert s["branches"] == ["SUPPORTED"] and "would add the consistency step" in s["notes"][1]
    s = V.next_steps(False, False, True, V.consistency_check({}, r14_provided=False), None)
    assert s["branches"] == ["P1 fails"] and "UNREAD" in s["notes"][0] and "unread" in s["notes"][1]


def test_pick_rate_rstd_biased_vs_unbiased():
    unbiased = [(position, 24) for position in range(24)] * 5  # picks spread like the shown options: every block's ratio is 1
    assert V.pick_rstd(unbiased)["value"] == pytest.approx(0.0)
    biased = [(0, 24)] * 120  # every pick on A: block ratios 3, 0, 0
    assert V.pick_rstd(biased)["value"] == pytest.approx(math.sqrt(2))
    assert V.pick_rstd([(0, 24)] * 4)["value"] is None  # blocks need >= 5 questions
    mixed = [(0, 6)] * 5 + [(9, 18)] * 5  # 6 options: block 0 only; 18 options: blocks 0-2 — block 2 is shown by 5 questions
    assert V.pick_rstd(mixed)["blocks"] == 3


R14_TAGS = {"requests": [
    {"id": "v08r14-lab-0000", "env": "lab", "questions": {
        "move": {"n_options": 3, "level": False, "planner": "lower", "options": [{"key": "lower", "tag": "ACTIVE"}, {"key": "wait", "tag": "PASSIVE"},
                                                                              {"key": "open", "tag": "ACTIVE"}]},
        "offset": {"n_options": 3, "level": True, "planner": None, "options": [{"key": "0", "tag": None}, {"key": "1", "tag": None}, {"key": "2", "tag": None}]}}},
    {"id": "v08r14-cafe-0000", "env": "cafe", "questions": {
        "say": {"n_options": 2, "level": False, "planner": None, "options": [{"key": "nothing", "tag": "PASSIVE"}, {"key": "hello", "tag": "ACTIVE"}]}}}]}


def _r14_answer(probs: dict, order: list) -> dict:
    return {"choice": max(probs, key=probs.get), "probabilities": probs, "codes": {k: V.G8.CODES_255[i] for i, k in enumerate(order)}}


def r14_tiny_scores() -> dict:
    """lab move: planner `lower`; orders 0 / 1 / 2 top lower (0.6), wait (0.7 — clears the web's override rule), a lower–wait tie (→ lower,
    the earlier key); lab offset (a level question): `1` every time; café say: `nothing` every time."""
    move = ["lower", "wait", "open"]
    answers = {"v08r14-lab-0000": {"move": {"spaces-0": _r14_answer({"lower": 0.6, "wait": 0.3, "open": 0.1}, move),
                                            "spaces-1": _r14_answer({"lower": 0.2, "wait": 0.7, "open": 0.1}, move),
                                            "spaces-2": _r14_answer({"lower": 0.45, "wait": 0.45, "open": 0.1}, move)},
                                   "offset": {f"spaces-{i}": _r14_answer({"0": 0.25, "1": 0.5, "2": 0.25}, ["0", "1", "2"]) for i in range(3)}},
               "v08r14-cafe-0000": {"say": {f"spaces-{i}": _r14_answer({"nothing": 0.8, "hello": 0.2}, ["nothing", "hello"]) for i in range(3)}}}
    return {"kind": "v08-bench-scores", "set": "R14", "answers": answers}


def r14_scores(questions: int, flips: int) -> dict:
    """`questions` one-question requests (env lab / cafe alternating, 12 options, picks on A) whose first `flips` change their top option in
    the order spaces-2."""
    keys = [f"opt_{j}" for j in range(12)]
    answers = {}
    for i in range(questions):
        top = {k: (0.5 if k == keys[0] else 0.5 / 11) for k in keys}
        other = {k: (0.5 if k == keys[1] else 0.5 / 11) for k in keys}
        answers[f"v08r14-x-{i:04d}"] = {"q": {"spaces-0": _r14_answer(top, keys), "spaces-1": _r14_answer(top, keys),
                                              "spaces-2": _r14_answer(other if i < flips else top, keys)}}
    return {"kind": "v08-bench-scores", "set": "R14", "answers": answers}


def r14_tags(questions: int) -> dict:
    keys = [f"opt_{j}" for j in range(12)]
    return {"requests": [{"id": f"v08r14-x-{i:04d}", "env": ("lab", "cafe")[i % 2],
                          "questions": {"q": {"n_options": 12, "level": False, "planner": None,
                                              "options": [{"key": k, "tag": "ACTIVE" if j else "PASSIVE"} for j, k in enumerate(keys)]}}}
                         for i in range(questions)]}


def _split_tags(env_questions: dict) -> tuple[dict, dict]:
    """Tags and records for the hold / continue split: per request one question whose options carry real tag rules (kinds from
    robo_jev.bench.r14_v08.passive_kind: wait → hold, carry-on → continue, lower → none)."""
    rule = {"lower": ("ACTIVE", "lower"), "wait": ("PASSIVE", "wait"), "carry": ("PASSIVE", "carry-on")}
    tags, records = {"requests": []}, {}
    for rid, (env, name) in env_questions.items():
        keys = ["lower", "wait", "carry"]
        tags["requests"].append({"id": rid, "env": env, "questions": {name: {"planner": "lower", "level": False,
                                                                           "options": [{"key": k, "tag": rule[k][0], "rule": rule[k][1]} for k in keys]}}})
        records[rid] = {"id": rid, "request": {"state": "s", "questions": {name: {"type": "choice", "criteria": {k: k for k in keys}}}}}
    return tags, records


def _tops(picks: dict, name_of: dict, readings=("balanced-s1", "single-s1", "single-s2")) -> dict:
    answers = {}
    for rid, top in picks.items():
        probs = {k: (0.8 if k == top else 0.1) for k in ("lower", "wait", "carry")}
        answers[rid] = {name_of[rid]: {r: {"choice": top, "probabilities": probs} for r in readings}}
    return {"kind": "v08-bench-scores", "set": "B8", "answers": answers}


def test_b8_passive_split_into_hold_and_continue():
    questions = {"v08b8-plates-000": ("plates", "move"), "v08b8-plates-001": ("plates", "move"), "v08b8-delivery-000": ("delivery", "manoeuvre"),
                 "v08b8-delivery-001": ("delivery", "manoeuvre")}
    tags, records = _split_tags(questions)
    name_of = {rid: name for rid, (_, name) in questions.items()}
    f = _tops({"v08b8-plates-000": "wait", "v08b8-plates-001": "carry", "v08b8-delivery-000": "carry", "v08b8-delivery-001": "carry"}, name_of)
    fp = _tops({"v08b8-plates-000": "wait", "v08b8-plates-001": "lower", "v08b8-delivery-000": "lower", "v08b8-delivery-001": "carry"}, name_of)
    out = V.b8_readings(tags, records, {"v08Fs0": f, "v08Fs1": f, "v08FPs0": fp, "v08FPs1": fp}, n_boot=200)
    for arm, (passive, hold, keep) in {"F": (1.0, 0.25, 0.75), "FP": (0.5, 0.25, 0.25)}.items():
        level = out["levels"][arm]
        assert (level["passive_top"]["value"], level["passive_top_hold"]["value"], level["passive_top_continue"]["value"]) == (passive, hold, keep), arm
        assert level["passive_top_hold"]["value"] + level["passive_top_continue"]["value"] == pytest.approx(level["passive_top"]["value"])
    assert out["P2"]["passive_top"]["value"] == pytest.approx(-0.5)  # P2 still reads the frozen PASSIVE tag
    assert out["P2_split"]["passive_top_hold"]["value"] == pytest.approx(0.0) and out["P2_split"]["passive_top_continue"]["value"] == pytest.approx(-0.5)
    assert out["P2_split"]["passive_top_continue"]["low"] is not None  # request-cluster interval
    assert out["levels_per_env"]["F"]["plates"]["passive_top_hold"] == 0.5 and out["levels_per_env"]["F"]["delivery"]["passive_top_continue"] == 1.0
    assert out["P2_per_env"]["delivery"]["passive_top_continue"]["value"] == pytest.approx(-0.5)


def test_r14_passive_split_into_hold_and_continue():
    questions = {"v08r14-humanoid-0000": ("humanoid", "maneuver"), "v08r14-cafe-0000": ("cafe", "say")}
    tags, _ = _split_tags(questions)
    name_of = {rid: name for rid, (_, name) in questions.items()}
    orders = ("spaces-0", "spaces-1", "spaces-2")

    def scores(tops: dict) -> dict:
        answers = {}
        for rid, per_order in tops.items():
            answers[rid] = {name_of[rid]: {o: {"probabilities": {k: (0.8 if k == t else 0.1) for k in ("lower", "wait", "carry")}} for o, t in zip(orders, per_order)}}
        return {"answers": answers}

    s = scores({"v08r14-humanoid-0000": ("carry", "carry", "wait"), "v08r14-cafe-0000": ("wait", "lower", "wait")})
    out = V.r14_readings(V.r14_units(tags, list(questions)), {"v08Fs0": s, "v08Fs1": s})
    pooled = out["models"]["F"]["pooled"]
    assert pooled["passive_top"] == pytest.approx((1.0 + 2 / 3) / 2)
    assert pooled["passive_top_hold"] == pytest.approx((1 / 3 + 2 / 3) / 2) and pooled["passive_top_continue"] == pytest.approx((2 / 3 + 0) / 2)
    assert pooled["passive_top_hold"] + pooled["passive_top_continue"] == pytest.approx(pooled["passive_top"])
    per = out["models"]["F"]["per_env_question"]
    assert per["humanoid maneuver"]["passive_top_continue"] == pytest.approx(2 / 3) and per["cafe say"]["passive_top_hold"] == pytest.approx(2 / 3)
    assert out["models"]["F"]["per_env"]["cafe"]["passive_top_continue"] == 0.0


def test_m8g_per_source_breakdown():
    m8 = m8g_data(items=8)  # ranks 3 and 7 are physics items
    preds = {}
    for m in V.MODELS:
        order = m8g_preds(m8, two_ok=True)
        for record, pred in zip(m8, order):  # physics items miss every two-letter-coded listing
            if record["meta"]["source"] == "PHYS" and record["meta"]["two_letter"]:
                listing = pred["questions"]["q"]["listings"][0]
                keys = listing["options"]
                wrong = next(k for k in keys if k != "o1")
                listing["code_probs"] = [0.7 if k == wrong else 0.3 / (len(keys) - 1) for k in keys]
        listing0 = [r for r in m8 if r["meta"]["listing"] == 0]  # the seed-2 file holds the listing-0 records, as the Spark chain scores it
        preds[m] = {"order": order, "seed2": m8g_preds(listing0, two_ok=True, kind="seed2", flip_bins={"61-255"})}
    out = V.m8g_readings(m8, preds, n_boot=0)
    phys, mmlu = out["per_source"]["F"]["PHYS"]["61-255"], out["per_source"]["F"]["MMLU"]["61-255"]
    assert (phys["acc_two"], phys["acc_single"], mmlu["acc_two"], mmlu["acc_single"]) == (0.0, 1.0, 1.0, 1.0)
    assert phys["margin"] == pytest.approx(-0.8) and mmlu["margin"] == pytest.approx(0.2)
    assert phys["flip_rate"] == 1.0 and phys["flip_records"] == 2 * 2 and mmlu["flip_rate"] == 1.0
    assert out["per_source"]["F"]["MMLU"]["2-10"]["flip_rate"] == 0.0
    assert phys["listings"] == 2 * 2 * 2 and phys["nll"] > mmlu["nll"]  # 2 items × N = 120, 255 × 2 listings
    assert "only general items reach 255" in out["per_source_note"]
    assert set(out["per_source"]["v08Fs0"]) == {"MMLU", "PHYS"}


# --------------------------------------------------------------------------
# A16 and S16 (robo-jev-web's 2026-10-16 requests; reported only)
# --------------------------------------------------------------------------

AXIS_KEYS = ["plus_far", "plus_mid", "plus_near", "plus_small", "plus_fine", "stay", "minus_far", "minus_mid", "minus_near", "minus_small", "minus_fine"]
#: x / y / z distances of four arm requests: a plus_fine, the −0.2 boundary, a 1–3 cm move, far and mid moves of both signs, a stay (z 0.1).
A16_DISTANCES = {"v08a16-0000": (0.9, -35.5, 82.0), "v08a16-0001": (-0.2, 5.0, 0.1), "v08a16-0002": (2.0, -10.0, -0.5), "v08a16-0003": (25.0, 0.5, 12.0)}


def a16_records() -> list[dict]:
    from robo_jev.bench import r16_v08 as R

    out = []
    for rid, distances in A16_DISTANCES.items():
        rule = {axis: {"distance": d, "label": R.arm_label(d), "label_boundary_stay": R.arm_label(d, boundary_stay=True), "band": R.arm_band(d)}
                for axis, d in zip("xyz", distances)}
        out.append({"id": rid, "meta": {"set": "v08-a16", "rule": rule},
                    "request": {"state": "servo", "questions": {a: {"type": "choice", "instructions": a, "criteria": {k: k for k in AXIS_KEYS}} for a in "xyz"}}})
    return out


def a16_answers(tops: dict, reading: str = "single-s1") -> dict:
    return {rid: {axis: {reading: {"choice": top, "probabilities": {k: (0.6 if k == top else 0.04) for k in AXIS_KEYS}}} for axis, top in per_axis.items()}
            for rid, per_axis in tops.items()}


#: Seed 0's tops: x — agree, stay on the −0.2 boundary, stay on a 2 cm move, agree; y — a sign error (plus_far for −35.5), then agree; z — agree.
TOPS_S0 = {"v08a16-0000": {"x": "plus_fine", "y": "plus_far", "z": "plus_far"}, "v08a16-0001": {"x": "stay", "y": "plus_near", "z": "stay"},
           "v08a16-0002": {"x": "stay", "y": "minus_mid", "z": "minus_fine"}, "v08a16-0003": {"x": "plus_far", "y": "plus_fine", "z": "plus_mid"}}
#: Seed 1: the same, but y agrees on the first request.
TOPS_S1 = {**TOPS_S0, "v08a16-0000": {**TOPS_S0["v08a16-0000"], "y": "minus_far"}}


def test_a16_readings_per_model_arm_and_ablation():
    records = a16_records()
    scores = {"v08Fs0": {"answers": a16_answers(TOPS_S0)}, "v08Fs1": {"answers": a16_answers(TOPS_S1)}}
    ablation = {m: {"answers": a16_answers({rid: {"y": AXIS_KEYS[6] if rid == "v08a16-0000" else TOPS_S0[rid]["y"]} for rid in TOPS_S0})} for m in scores}
    stored = a16_answers(TOPS_S0, reading="stored")
    out = V.a16_readings(records, scores, ablation, stored)
    s0 = out["models"]["v08Fs0"]["summary"]
    assert (s0["agreement"]["count"], s0["agreement"]["n"]) == (9, 12)
    assert [s0["agreement_by_axis"][a]["count"] for a in "xyz"] == [2, 3, 4]
    assert [(s0["sign_errors_by_axis"][a]["count"], s0["sign_errors_by_axis"][a]["n"]) for a in "xyz"] == [(0, 2), (1, 4), (0, 3)]
    assert (s0["stay_rate_small"]["count"], s0["stay_rate_small"]["n"]) == (2, 5)  # the −0.2 boundary and the 2 cm move
    assert out["models"]["v08Fs0"]["boundary_stay"]["agreement_by_axis"]["x"]["count"] == 3  # ±0.2 read as stay: x on the boundary agrees
    arm = out["models"]["F"]["summary"]
    assert arm["agreement"]["value"] == pytest.approx(9.5 / 12) and arm["sign_errors_by_axis"]["y"]["value"] == pytest.approx(0.5 / 4)
    assert out["reference_v07_stored"]["summary"]["agreement"]["count"] == 9  # the stored endpoint row reads the "stored" reading
    ab = out["models"]["F"]["ablation_y"]
    assert ab["with_phrase"]["agreement"]["value"] == pytest.approx(3.5 / 4) and ab["without_phrase"]["agreement"]["value"] == pytest.approx(1.0)
    assert ab["with_phrase"]["sign_errors"]["value"] == pytest.approx(0.5 / 4) and ab["without_phrase"]["sign_errors"]["value"] == 0.0
    assert "27 -> 50 of 90" in out["serve_ablation_v07"]


def s16_records() -> list[dict]:
    move, look = [f"m{j:02d}" for j in range(15)], [f"g{j}" for j in range(6)]
    return [{"id": f"v08s16-{i:04d}", "meta": {"set": "v08-s16", "groups": {"red1": "movement", "red1_look": "gaze"}},
             "request": {"state": f"pitch {i}", "questions": {"red1": {"type": "choice", "criteria": {k: k for k in move}},
                                                              "red1_look": {"type": "choice", "criteria": {k: k for k in look}}}}} for i in range(2)]


def s16_scores(records: list[dict], *, flip_movement: bool) -> dict:
    """Movement flips in request 0 (when asked); request 1's third order is a tie between the first two moves (→ the earlier, no flip); gaze
    never flips. Codes A, B, … in key order."""
    answers = {}
    for r in records:
        rid = r["id"]
        per = {}
        for name, q in r["request"]["questions"].items():
            keys = list(q["criteria"])
            codes = {k: V.G8.CODES_255[j] for j, k in enumerate(keys)}
            readings = {}
            for o in range(3):
                top = keys[1] if (flip_movement and name == "red1" and rid.endswith("0000") and o == 2) else keys[0]
                probs = {k: (0.4 if k == top else 0.6 / (len(keys) - 1)) for k in keys}
                if name == "red1" and rid.endswith("0001") and o == 2:
                    probs = {k: (0.3 if k in keys[:2] else 0.4 / (len(keys) - 2)) for k in keys}
                readings[f"spaces-{o}"] = {"choice": top, "probabilities": probs, "codes": codes}
            per[name] = readings
        answers[rid] = per
    return {"answers": answers}


def test_s16_flips_per_group_and_the_stored_spread():
    records = s16_records()
    scores = {m: s16_scores(records, flip_movement=m.startswith("v08Fs")) for m in ("v08Fs0", "v08Fs1", "v07s0")}
    stored = {r["id"]: {name: {"stored": {"choice": list(q["criteria"])[0], "probabilities": {k: (0.5 if j == 0 else 0.5 / (len(q["criteria"]) - 1))
                                                                                              for j, k in enumerate(q["criteria"])}}}
                           for name, q in r["request"]["questions"].items()} for r in records}
    out = V.s16_readings(records, scores, stored)
    f = out["models"]["F"]
    assert f["per_group"]["movement"]["flip_rate"] == 0.5 and f["per_group"]["gaze"]["flip_rate"] == 0.0 and f["pooled"]["flip_rate"] == 0.25
    assert out["models"]["v07s0"]["per_group"]["movement"]["flip_rate"] == 0.0  # the tie goes to the earlier key: no flip
    assert out["per_group_questions"] == {"movement": 2, "gaze": 2}
    assert f["per_group"]["gaze"]["top_prob"]["n"] == 2 * 3 * 2  # two seeds × two questions × three orders
    assert out["reference_v07_stored_top_prob"]["movement"]["mean"] == pytest.approx(0.5) and out["reference_v07_stored_top_prob"]["pooled"]["n"] == 4
    assert f["per_group"]["gaze"]["pick_rstd"]["questions"] == 12


def _r16_tree(tmp_path: Path) -> tuple[Path, Path, dict]:
    spark, r16 = tmp_path / "spark", tmp_path / "r16"
    spark.mkdir()
    r16.mkdir()
    a16 = a16_records()
    (r16 / "A16.jsonl").write_text("".join(json.dumps(r) + "\n" for r in a16), encoding="utf-8")
    (r16 / "A16-y-ablation.jsonl").write_text("".join(json.dumps(r) + "\n" for r in a16), encoding="utf-8")
    (r16 / "S16.jsonl").write_text("".join(json.dumps(r) + "\n" for r in s16_records()), encoding="utf-8")
    (r16 / "A16-stored-v07.json").write_text(json.dumps({"answers": a16_answers(TOPS_S0, reading="stored")}), encoding="utf-8")
    (r16 / "S16-stored-v07.json").write_text(json.dumps({"answers": {}}), encoding="utf-8")
    names = ["A16.jsonl", "A16-y-ablation.jsonl", "S16.jsonl", "A16-stored-v07.json", "S16-stored-v07.json"]
    frozen = {"files": {n: _sha(r16 / n) for n in names}, "frozen": "2026-10-16T15:12:54+09:00"}
    (r16 / "frozen-before-scoring.json").write_text(json.dumps(frozen), encoding="utf-8")
    V._sha.cache_clear()
    return spark, r16, frozen


def test_r16_loading_and_the_freeze_refusing_only_these_sets(tmp_path):
    spark, r16, frozen = _r16_tree(tmp_path)
    good = {"kind": "v08-bench-scores", "set": "A16", "tag": "v08Fs0-2b", "input_sha256": _sha(r16 / "A16.jsonl"), "readings": {"single-s1": {}},
            "frozen": frozen, "checkpoint": None, "answers": a16_answers(TOPS_S0)}
    (spark / "A16-v08Fs0-2b.json").write_text(json.dumps(good), encoding="utf-8")
    (spark / "A16-v08Fs1-2b.json").write_text(json.dumps({**good, "tag": "v08Fs1-2b", "frozen": {"files": {}}}), encoding="utf-8")
    loaded = V.load_r16(spark, {"v08Fs0": None, "v08Fs1": None}, [], directory=r16, exclude={"a16": ["v08a16-0003"], "s16": []})
    assert list(loaded["scores"]["A16"]) == ["v08Fs0"] and any("frozen" in p for p in loaded["left_out"])  # scored on other files: left out
    assert [r["id"] for r in loaded["records"]["A16"]] == ["v08a16-0000", "v08a16-0001", "v08a16-0002"]  # the overlap exclusion applies
    part = V.r16_part(loaded)
    assert part["A16"]["models"]["v08Fs0"]["summary"]["rows"] == 9 and "R14 (the 10-14 requests)" in part["A16"]["note"]
    assert V.load_r16(spark, {}, [], directory=tmp_path / "nothing") is None
    (r16 / "S16.jsonl").chmod(0o644)
    (r16 / "S16.jsonl").write_text("{}\n", encoding="utf-8")
    refused = V.load_r16(spark, {}, [], directory=r16)
    assert refused == {"refused": ["S16.jsonl changed after it was frozen"]}
    out = scenario(p1=True, p2=True, r16=refused)
    assert out["readings"]["A16"] == {"refused": refused["refused"]} and out["verdict"]["outcome"] == "SUPPORTED"  # only these sets are refused
    assert "R14" not in out["verdict"]["consistency"]["sources"] and "A16" not in out["verdict"]["consistency"]["sources"]


def test_r16_rendered_sections():
    records, s_records = a16_records(), s16_records()
    r16 = {"records": {"A16": records, "A16y": records, "S16": s_records}, "frozen": {"frozen": "t"}, "left_out": [], "excluded": {}, "check": [],
           "scores": {"A16": {"v08Fs0": {"answers": a16_answers(TOPS_S0)}, "v08Fs1": {"answers": a16_answers(TOPS_S1)}},
                      "A16y": {"v08Fs0": {"answers": a16_answers({rid: {"y": TOPS_S0[rid]["y"]} for rid in TOPS_S0})}},
                      "S16": {"v08Fs0": s16_scores(s_records, flip_movement=True), "v08Fs1": s16_scores(s_records, flip_movement=True)}},
           "stored": {"A16": a16_answers(TOPS_S0, reading="stored"), "S16": {}}}
    out = scenario(p1=True, p2=True, r16=r16)
    md = V.render({**out, "n_boot": 200, "seed": 0, "git": None, "notes": []})
    for needle in ("### A16 and S16 — robo-jev-web's 2026-10-16 requests (reported only)", "| v0.7 endpoint (stored) |", "y-phrase ablation",
                   "**S16 — soccer**", "whose third source stays R14 (the 10-14 requests)", "| F (arm, seed mean) | movement | 2 | 0.500 |"):
        assert needle in md, needle
    assert out["readings"]["A16"]["models"]["F"]["summary"]["agreement"]["value"] == pytest.approx(9.5 / 12)


def test_r14_readings_on_a_tiny_set():
    units = V.r14_units(R14_TAGS, ["v08r14-lab-0000", "v08r14-cafe-0000"])
    assert [(u["env"], u["question"], u["level"], u["planner"]) for u in units] == [("lab", "move", False, "lower"), ("lab", "offset", True, None),
                                                                                   ("cafe", "say", False, None)]
    out = V.r14_readings(units, {"v08Fs0": r14_tiny_scores(), "v08Fs1": r14_tiny_scores()})
    for model in ("v08Fs0", "F"):
        pooled = out["models"][model]["pooled"]
        assert pooled["questions"] == 3 and pooled["flip_rate"] == pytest.approx(1 / 3)  # lab move flips (lower, wait, tie → lower)
        assert pooled["passive_top"] == pytest.approx((1 / 3 + 1.0) / 2) and pooled["passive_questions"] == 2  # the level question has no tags
        assert pooled["planner_agreement"] == pytest.approx(2 / 3) and pooled["planner_questions"] == 1
        assert pooled["override_clearance"] == pytest.approx(1 / 3)  # spaces-1: wait 0.7 >= 0.55 and 0.5 above the planner's 0.2
        values = [0.6, 0.7, 0.45, 0.5, 0.5, 0.5, 0.8, 0.8, 0.8]
        assert pooled["top_prob"]["mean"] == pytest.approx(np.mean(values)) and pooled["top_prob"]["median"] == pytest.approx(0.6)
        assert pooled["top_prob"]["q25"] == pytest.approx(0.5) and pooled["top_prob"]["q75"] == pytest.approx(0.8)
    per = out["models"]["F"]["per_env_question"]
    assert per["lab move"]["flip_rate"] == 1.0 and per["lab offset"]["flip_rate"] == 0.0 and per["lab offset"]["passive_top"] is None
    assert out["models"]["F"]["per_env"]["cafe"]["passive_top"] == 1.0 and out["level_questions"] == 1 and out["planner_questions"] == 1
    assert out["models"]["F"]["pooled"]["top_prob"]["n"] == 18  # an arm pools both seeds' question × order values


def test_r14_pick_rate_rstd_and_the_condition_at_the_bound():
    tags = r14_tags(10)
    ids = [r["id"] for r in tags["requests"]]
    quiet = V.r14_readings(V.r14_units(tags, ids), {"v08Fs0": r14_scores(10, 2), "v08Fs1": r14_scores(10, 2)})
    assert quiet["models"]["F"]["pooled"]["flip_rate"] == pytest.approx(0.2)
    rstd = quiet["models"]["F"]["pooled"]["pick_rstd"]  # 60 picks of 12 options (two seeds × 10 × 3 orders), all in block 0 (56 on A, 4 on B) → ratios 1.5 and 0
    assert rstd["value"] == pytest.approx(np.std([12 / 8, 0.0])) and rstd["blocks"] == 2
    assert V.consistency_check({**_flips(), V.R14: quiet["models"]["F"]["pooled"]["flip_rate"]}, r14_provided=True)["status"] == "NOT TRIGGERED"
    loud = V.r14_readings(V.r14_units(tags, ids), {"v08Fs0": r14_scores(10, 3), "v08Fs1": r14_scores(10, 2)})
    assert loud["models"]["F"]["pooled"]["flip_rate"] == pytest.approx(0.25)
    assert V.consistency_check({**_flips(), V.R14: 0.2000001}, r14_provided=True)["over"] == ["R14"]


def test_decide_reports_guards_without_changing_the_outcome():
    readings = {"P1": {"value": -0.1, "high": -0.01}, "P2": {"passive_top": {"value": -0.3}, "planner": {"value": 0.0}},
                "guards": {**GOOD_GUARDS, "CA": {"value": -0.5}}, "FORMAT": _fmt(), "O-F": {"value": 0.0}, "flip_F_ge27": 0.1}
    v = V.decide(readings)
    assert v["outcome"] == "SUPPORTED" and not v["guards"]["pass"] and v["FORMAT"]["pass"] and v["O-F"]["pass"]
    assert v["release_line"].startswith("RELEASE: the user decides")


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def test_argmax_ties_go_to_the_earlier_option():
    assert V.argmax_key({"b": 0.4, "a": 0.4, "c": 0.2}, ["a", "b", "c"]) == "a"
    assert V.argmax_key({"b": 0.4, "a": 0.4, "c": 0.2}, ["b", "a", "c"]) == "b"
    assert V.argmax_key({"c": 0.9}, ["a", "b", "c"]) == "c"


def test_ece_and_rstd():
    assert V._ece([0.95, 0.95, 0.95, 0.95], [1, 1, 1, 0]) == pytest.approx(0.2)
    assert V._ece([], []) is None
    r = V._rstd([0] * 10 + [8] * 10, [1.0, 0.0] * 10)  # block 0 (positions 0-7) and block 1 (8-15) both half right
    assert r["blocks"] == 2 and r["value"] == pytest.approx(0.0)
    r = V._rstd(list(range(16)) * 5, [1.0] * 40 + [0.0] * 40)  # block 0: 3 of 5 right per position, block 1: 2 of 5
    assert r["value"] == pytest.approx(0.1)
    r = V._rstd([0] * 5 + [8] * 5, [1.0] * 5 + [0.0] * 5)
    assert r["value"] == pytest.approx(0.5)
    assert V._rstd([0] * 4 + [8] * 5, [1.0] * 9)["value"] is None  # block 0 has < 5 listings → one block only


def test_cluster_diff_pairs_units():
    d = V._cluster_diff({"a": 1.0, "b": 0.0, "c": 1.0}, {"a": 0.0, "b": 0.0, "d": 1.0}, {"a": "x", "b": "y", "c": "z", "d": "z"}, n_boot=200, seed=0)
    assert d["n"] == 2 and d["value"] == pytest.approx(0.5) and d["low"] <= d["value"] <= d["high"]


# --------------------------------------------------------------------------
# Synthetic data
# --------------------------------------------------------------------------


def call_set(name: str, pairs: int = 30) -> list[dict]:
    """A call-view set: per pair a clear-risky record (label 0.9, robust) and an unclear one (0.5); every third pair adds a clear-safe record
    (0.1, robust, alone); each record also has a choice question with a spread label."""
    out = []
    for i in range(pairs):
        a, b = f"{name}-{i:03d}a", f"{name}-{i:03d}b"
        for rid, partner, y, robust in ((a, b, 0.9, True), (b, a, 0.5, False)):
            out.append({"id": rid, "meta": {"set": name, "pair": {"of": partner}, "robustness": {"q": {"robust": robust}}},
                        "request": {"state": "s", "questions": {"q": {"type": "noul", "instructions": "x"},
                                                                 "act": {"type": "choice", "instructions": "y", "criteria": {"go": "g", "stop": "s"}}}},
                        "labels": {"q": {"type": "noul", "p_yes": y}, "act": {"type": "choice", "kind": "action", "probabilities": {"go": 0.7, "stop": 0.3}}}})
        if i % 3 == 0:
            out.append({"id": f"{name}-{i:03d}s", "meta": {"set": name, "robustness": {"q": {"robust": True}}},
                        "request": {"state": "s", "questions": {"q": {"type": "noul", "instructions": "x"}}}, "labels": {"q": {"type": "noul", "p_yes": 0.1}}})
    return out


def call_preds(records: list[dict], safe_share: float) -> list[dict]:
    """Calls: clear-safe → safe; clear-risky and unclear → safe (p 0.2) on the first `safe_share` of them (by index), else risky (0.8)."""
    risky_or_unclear = [r for r in records if r["labels"]["q"]["p_yes"] >= 0.5]
    safe_ids = {r["id"] for r in risky_or_unclear[: int(round(safe_share * len(risky_or_unclear)))]}
    out = []
    for r in records:
        p = 0.2 if r["labels"]["q"]["p_yes"] < 0.5 or r["id"] in safe_ids else 0.8
        questions = {"q": {"type": "noul", "listings": [{"answer": {"p_yes": p}}, {"answer": {"p_yes": p}}]}}
        if "act" in r["request"]["questions"]:
            questions["act"] = {"type": "choice", "listings": [{"options": ["go", "stop"], "answer": {"probabilities": {"go": 0.5, "stop": 0.5}}}]}
        out.append({"id": r["id"], "questions": questions})
    return out


def home_sets() -> dict[str, list[dict]]:
    sets = {}
    for h in ("T2", "T3", "T4"):
        sets[h] = [{"id": f"{h}-{i}", "meta": {"set": h}, "request": {"state": "s", "questions": {"q": {"type": "noul", "instructions": "x"}}},
                    "labels": {"q": {"type": "noul", "p_yes": (0.1, 0.5, 0.9)[i % 3]}}} for i in range(12)]
    sets["G2"] = [{"id": f"g2-{i}", "meta": {"domain": ("mnli", "boolq")[i % 2]},
                   "request": {"state": "s", "questions": {"q": {"type": "choice", "instructions": "x", "criteria": {"a": "A", "b": "B"}}}},
                   "labels": {"q": {"type": "choice", "kind": "probe", "probabilities": {"a": 1.0, "b": 0.0}}}} for i in range(10)]
    return sets


def home_preds(sets: dict[str, list[dict]]) -> dict[str, list[dict]]:
    out = {h: [{"id": r["id"], "questions": {"q": {"type": "noul", "listings": [{"answer": {"p_yes": r["labels"]["q"]["p_yes"]}}]}}} for r in sets[h]]
           for h in ("T2", "T3", "T4")}
    out["G2"] = [{"id": r["id"], "questions": {"q": {"type": "choice", "listings": [{"options": ["a", "b"], "answer": {"probabilities": {"a": 0.8, "b": 0.2}},
                                                                                   "code_mass": 0.9}]}}} for r in sets["G2"]]
    return out


ENV_Q = {"plates": [("move", "above_in1", ["above_in1", "lower", "wait", "look"])],
         "cafe": [("intent", "serve_9", ["serve_9", "take_up_tray", "wait"]), ("attend", None, ["P3", "P5", "look_round"])],
         "delivery": [("manoeuvre", "follow", ["follow", "pass_left", "stop", "call_operator"])],
         "tubes": [("work", "prep_t1", [f"prep_t{i}" for i in range(1, 30)] + ["wait"])]}
PASSIVE = {"wait", "look", "look_round", "stop", "call_operator"}


def b8_data(requests_per_env: int = 6) -> tuple[dict, dict]:
    tags, records = {"kind": "v08-b8-tags", "requests": []}, {}
    for env, questions in ENV_Q.items():
        for i in range(requests_per_env if env != "tubes" else 2):
            rid = f"v08b8-{env}-{i:03d}"
            entry = {"id": rid, "env": env, "questions": {}}
            qs = {}
            for name, planner, keys in questions:
                entry["questions"][name] = {"n_options": len(keys), "planner": planner,
                                            "options": [{"key": k, "text": k, "tag": "PASSIVE" if k in PASSIVE else "ACTIVE", "rule": "t", "borderline": False} for k in keys]}
                qs[name] = {"type": "choice", "instructions": "x", "criteria": {k: k for k in keys}}
            tags["requests"].append(entry)
            records[rid] = {"id": rid, "meta": {"set": "v08-b8", "env": env}, "request": {"state": f"s {rid}", "questions": qs}}
    return tags, records


def b8_scores(tags: dict, *, passive: bool, codes_two: bool = False, flip: bool = False) -> dict:
    """Every question answered: the top option is a PASSIVE one (`passive`) or the planner's / the first ACTIVE one."""
    answers = {}
    for entry in tags["requests"]:
        for name, q in entry["questions"].items():
            keys = [o["key"] for o in q["options"]]
            passive_keys = [o["key"] for o in q["options"] if o["tag"] == "PASSIVE"]
            active = q["planner"] or next(o["key"] for o in q["options"] if o["tag"] == "ACTIVE")
            top = passive_keys[0] if passive else active
            probs = {k: (0.6 if k == top else 0.4 / (len(keys) - 1)) for k in keys}
            codes = {k: (("AA" if codes_two else "A") if i == 0 else chr(66 + i % 20)) for i, k in enumerate(keys)}
            other = next(k for k in keys if k != top)
            probs2 = {k: (0.6 if k == other else 0.4 / (len(keys) - 1)) for k in keys} if flip else probs
            answers.setdefault(entry["id"], {})[name] = {"balanced-s1": {"choice": top, "probabilities": probs},
                                                         "single-s1": {"choice": top, "probabilities": probs, "codes": codes},
                                                         "single-s2": {"choice": other if flip else top, "probabilities": probs2, "codes": codes}}
    return {"kind": "v08-bench-scores", "set": "B8", "answers": answers}


LEVELS = (6, 18, 40, 120, 255)


def m8g_data(items: int = 10) -> list[dict]:
    out = []
    for rank in range(items):
        source = "PHYS" if rank % 4 == 3 else "MMLU"
        for n in LEVELS:
            for listing in (0, 1):
                if n <= 26:
                    half = (n + 1) // 2
                    position = rank % half if listing == 0 else half + rank % (n - half)
                else:
                    position = rank % 26 if listing == 0 else 26 + rank % (n - 26)
                others = [f"o{j}" for j in range(2, n + 1)]
                keys = others[:position] + ["o1"] + others[position:]
                out.append({"id": f"v08m8g-{rank:05d}-n{n:03d}-{listing}",
                            "meta": {"set": "v08-m8g", "item": f"item{rank}", "source": source, "n": n, "position": position,
                                     "code": "?", "two_letter": position >= 26, "listing": listing, "correct": "o1", "rank": rank,
                                     "bin": V.G8.bin_of(n)},
                            "request": {"state": f"question {rank}", "questions": {"q": {"type": "choice", "instructions": "x", "criteria": {k: k for k in keys}}}},
                            "labels": {"q": {"type": "choice", "kind": "probe", "probabilities": {k: float(k == "o1") for k in keys}}}})
    return out


def m8g_preds(records: list[dict], *, two_ok: bool, small_ok: bool = True, kind: str = "order", flip: bool = False,
              flip_bins: set | None = None) -> list[dict]:
    """Correct on single-letter listings (at N ≤ 26 only if `small_ok`), on two-letter ones only if `two_ok`; `seed2` lists the options
    reversed (and, with `flip`, tops another option); `balanced` gives the averaged answer."""
    out = []
    for r in records:
        keys = list(r["request"]["questions"]["q"]["criteria"])
        m = r["meta"]
        right = (m["two_letter"] and two_ok) or (not m["two_letter"] and (small_ok or m["n"] > 26))
        top = "o1" if right else next(k for k in keys if k != "o1")
        if kind == "seed2" and (flip or (flip_bins and m["bin"] in flip_bins)):
            top = next(k for k in keys if k != top)
        probs = {k: (0.7 if k == top else 0.3 / (len(keys) - 1)) for k in keys}
        if kind == "balanced":
            out.append({"id": r["id"], "questions": {"q": {"type": "choice", "answer": {"probabilities": probs}, "listings": []}}})
            continue
        shown = list(reversed(keys)) if kind == "seed2" else keys
        out.append({"id": r["id"], "questions": {"q": {"type": "choice", "listings": [{"options": shown, "code_probs": [probs[k] for k in shown]}]}}})
    return out


def scenario(*, p1: bool, p2: bool, format_ok: bool = True, flips: bool = False, of_ok: bool = True, n_boot: int = 200, flip_bins: set | None = None,
             b8_flip: bool = False, r14_flips: int | None = None, r14_refused: bool = False, r16: dict | None = None) -> dict:
    sets = {s: call_set(s) for s in ("A4", "T9", "T8b")}
    preds: dict[str, dict] = {}
    for model in V.MODELS:
        if model.startswith("v08FP"):
            share = 0.3 if p1 else 0.6
        elif model.startswith("v07"):
            share = 0.6 if of_ok else 0.3  # O-F = F − v0.7: +0.3 when v0.7 calls safe less often
        else:
            share = 0.6
        preds[model] = {s: call_preds(sets[s], share) for s in sets}
    pod_sets = home_sets()
    pod_preds = {m: home_preds(pod_sets) for m in V.V08_MODELS}
    tags, records = b8_data()
    scores = {m: b8_scores(tags, passive=not (p2 and m.startswith("v08FP")), flip=b8_flip and m.startswith("v08Fs")) for m in V.MODELS}
    m8 = m8g_data()
    m8_preds = {}
    for m in V.MODELS:
        m8_preds[m] = {"order": m8g_preds(m8, two_ok=format_ok or not m.startswith("v08F")),
                       "seed2": m8g_preds(m8, two_ok=format_ok or not m.startswith("v08F"), kind="seed2", flip=flips,
                                          flip_bins=flip_bins if m.startswith("v08Fs") else None),
                       "balanced": m8g_preds([r for r in m8 if r["meta"]["listing"] == 0 and r["meta"]["n"] <= 60], two_ok=True, kind="balanced")}
    r14 = None
    if r14_refused:
        r14 = {"refused": ["tags.json changed after it was frozen"]}
    elif r14_flips is not None:
        r14_tag = r14_tags(20)
        r14 = {"tags": r14_tag, "record_ids": [r["id"] for r in r14_tag["requests"]], "left_out": [], "check": None, "frozen": {"frozen": "t"},
               "scores": {m: r14_scores(20, r14_flips if m.startswith("v08Fs") else 0) for m in V.MODELS}}
    return V.compute(sets, preds, pod_sets, pod_preds, {"tags": tags, "records": records, "scores": scores},
                     {"records": m8, "preds": m8_preds}, r14=r14, r16=r16, posthoc=Path("/nonexistent/estimate.json"), n_boot=n_boot, seed=0)


# --------------------------------------------------------------------------
# Synthetic predictions through the whole computation
# --------------------------------------------------------------------------


def test_supported_through_the_pipeline():
    out = scenario(p1=True, p2=True)
    v, r = out["verdict"], out["readings"]
    assert v["outcome"] == "SUPPORTED", v
    assert r["standard"]["P1"]["value"] == pytest.approx(-0.3) and r["standard"]["P1"]["high"] < 0
    assert r["standard"]["P1"]["set_seeds"] == {"A4": 0, "T9": 1, "T8b": 2}
    assert r["B8"]["P2"]["passive_top"]["value"] == pytest.approx(-1.0) and r["B8"]["P2"]["planner"]["value"] == pytest.approx(1.0)
    assert r["B8"]["p2_units"] == 6 * 4 and r["B8"]["planner_units"] == 6 * 3  # café attend has no planner; tubes are not P2 units
    assert v["guards"]["pass"], v["guards"]
    assert v["FORMAT"]["pass"] and v["O-F"]["pass"] and v["flip_F_ge27"] == 0.0
    assert v["next_steps"]["branches"] == ["SUPPORTED"]
    assert r["standard"]["teacher_kl"]["F"]["2-10"]["value"] == pytest.approx(0.7 * math.log(1.4) + 0.3 * math.log(0.6))
    assert r["M8g"]["per_bin"]["F"]["61-255"]["accuracy"]["value"] == pytest.approx(1.0)
    assert r["M8g"]["single_vs_balanced"]["v08Fs0"]["27-60"]["balanced"] == pytest.approx(1.0)
    assert r["B8"]["long_lists"]["tubes work"]["models"]["v08Fs0"]["top_single_eq_balanced"] == pytest.approx(1.0)
    assert r["posthoc"].get("absent")
    md = V.render({**out, "n_boot": 200, "seed": 0, "git": None, "notes": []})
    assert "SUPPORTED" in md and "## P2" in md
    assert "### M8g per source" in md and "only general items reach 255" in md and "passive-top: hold" in md


def test_partial_both_ways_and_rejected():
    only_p1 = scenario(p1=True, p2=False)["verdict"]
    assert only_p1["outcome"] == "PARTIAL" and only_p1["P1"]["pass"] and not only_p1["P2"]["pass"]
    assert only_p1["next_steps"]["branches"][0] == "P2 fails while P1 passes"
    only_p2 = scenario(p1=False, p2=True)["verdict"]
    assert only_p2["outcome"] == "PARTIAL" and not only_p2["P1"]["pass"] and only_p2["P2"]["pass"]
    assert only_p2["next_steps"]["branches"][0] == "P1 fails"
    neither = scenario(p1=False, p2=False)["verdict"]
    assert neither["outcome"] == "REJECTED" and neither["next_steps"]["branches"] == ["P1 fails"]


def test_format_failure_flips_and_of_through_the_pipeline():
    bad = scenario(p1=True, p2=True, format_ok=False)
    v = bad["verdict"]
    assert v["outcome"] == "SUPPORTED" and not v["FORMAT"]["pass"] and "FORMAT fails" in v["next_steps"]["branches"]
    assert bad["readings"]["M8g"]["FORMAT"]["bins"]["61-255"]["acc_two"]["value"] == pytest.approx(0.0)
    flipped = scenario(p1=True, p2=True, flips=True)["verdict"]
    assert flipped["FORMAT"]["pass"] and flipped["flip_F_ge27"] == pytest.approx(1.0)
    assert flipped["consistency"]["status"] == "TRIGGERED" and flipped["consistency"]["over"] == [f"M8g {b}" for b in V.BIN_NAMES]
    assert flipped["next_steps"]["branches"][-1] == V.CONSISTENCY_BRANCH
    assert flipped["consistency_original_reading"]["would_add_step"]
    of_fail = scenario(p1=True, p2=True, of_ok=False)["verdict"]
    assert not of_fail["O-F"]["pass"] and of_fail["O-F"]["value"] == pytest.approx(0.3)


def test_consistency_sources_through_the_pipeline():
    base = scenario(p1=True, p2=True)["verdict"]
    assert base["consistency"]["status"] == "NOT TRIGGERED" and base["consistency"]["sources"] == list(V.CONSISTENCY_SOURCES)
    for name in V.BIN_NAMES:  # one M8g bin flipping at a time
        c = scenario(p1=True, p2=True, flip_bins={name})["verdict"]["consistency"]
        assert c["status"] == "TRIGGERED" and c["over"] == [f"M8g {name}"], name
    out = scenario(p1=True, p2=True, b8_flip=True)
    c = out["verdict"]["consistency"]
    assert c["over"] == ["B8"] and out["readings"]["B8"]["flips"]["pooled"]["F"]["value"] == pytest.approx(1.0)
    assert set(out["readings"]["B8"]["flips"]["per_group"]["F"]) == {"plates", "cafe", "delivery", "tubes"}  # every tagged question
    quiet = scenario(p1=True, p2=True, r14_flips=4)  # 4 of 20 = 0.20 exactly
    assert quiet["verdict"]["consistency"]["status"] == "NOT TRIGGERED" and "R14" in quiet["verdict"]["consistency"]["sources"]
    assert quiet["readings"]["R14"]["models"]["F"]["pooled"]["flip_rate"] == pytest.approx(0.2)
    md = V.render({**quiet, "n_boot": 200, "seed": 0, "git": None, "notes": []})
    assert "### R14 — robo-jev-web's collected bench requests" in md and "**NOT TRIGGERED**" in md and "the original wording" in md
    assert "| F (arm, seed mean) | 20 | 0.200 |" in md
    loud = scenario(p1=True, p2=True, r14_flips=5)["verdict"]
    assert loud["consistency"]["over"] == ["R14"] and loud["next_steps"]["branches"][-1] == V.CONSISTENCY_BRANCH
    absent = scenario(p1=True, p2=True)
    assert "R14" not in absent["verdict"]["consistency"]["sources"] and "M8g and B8 only" in absent["verdict"]["consistency"]["r14"]
    assert "absent: no R14-<tag>.json score files" in V.render({**absent, "n_boot": 200, "seed": 0, "git": None, "notes": []})
    refused = scenario(p1=True, p2=True, r14_refused=True)
    c = refused["verdict"]["consistency"]
    assert c["status"] == "UNREAD" and c["unread"] == ["R14"] and refused["verdict"]["outcome"] == "SUPPORTED"  # R14 refused, the rest stands
    assert "**refused**" in V.render({**refused, "n_boot": 200, "seed": 0, "git": None, "notes": []})
    fails = scenario(p1=True, p2=True, format_ok=False, r14_flips=5)["verdict"]  # fires regardless of FORMAT, with FORMAT's own step
    assert fails["next_steps"]["branches"] == ["SUPPORTED", "FORMAT fails", V.CONSISTENCY_BRANCH]
    assert not fails["consistency_original_reading"]["would_add_step"]


def test_pick_rate_rstd_through_the_pipeline():
    out = scenario(p1=True, p2=True, r14_flips=0)
    r14 = out["readings"]["R14"]["models"]["F"]["pooled"]["pick_rstd"]  # every pick on A of 12 options: blocks 0 (8) and 1 (4)
    assert r14["value"] == pytest.approx(np.std([12 / 8, 0.0])) and r14["blocks"] == 2
    b8 = out["readings"]["B8"]["flips"]["pick_rstd"]["F"]
    assert b8["questions"] > 0
    m8 = out["readings"]["M8g"]["per_bin"]["v08Fs0"]["61-255"]["pick_rstd"]
    assert m8["value"] is not None and m8["questions"] > 0


def test_mean_of_sets_draws_each_set_with_its_own_seed():
    sets = {s: call_set(s, pairs=20) for s in ("A4", "T9", "T8b")}
    preds = {m: {s: call_preds(sets[s], 0.3 + 0.1 * (i % 3) + (0.2 if m.startswith("v08FP") else 0.0)) for i, s in enumerate(sets)} for m in V.V08_MODELS}
    calls = V.calls_of(sets, preds, V.P1_SETS)
    mean = V.mean_of_sets(calls, "FP", "F", metric="DC", n_boot=300, seed=0)
    draws = [V.metric_draws(calls[("FP", s)], calls[("F", s)], metric="DC", n_boot=300, seed=i)["draws"] for i, s in enumerate(V.P1_SETS)]
    low, high = V._interval(np.mean(np.stack(draws), axis=0))
    assert (mean["low"], mean["high"]) == (low, high)
    assert mean["value"] == pytest.approx(np.mean([mean["per_set"][s]["value"] for s in V.P1_SETS]))
    missing = V.mean_of_sets({k: v for k, v in calls.items() if k[1] != "T8b"}, "FP", "F", metric="DC", n_boot=0, seed=0)
    assert missing["value"] is None and not V.p1_check(missing)["pass"]


def test_exclusions_leave_items_and_requests_out():
    m8 = m8g_data(items=4)
    preds = {m: {"order": m8g_preds(m8, two_ok=True)} for m in V.MODELS}
    out = V.m8g_readings(m8, preds, n_boot=0, exclude={"item0"})
    assert out["items"] == 3 and out["excluded_items"] == ["item0"]
    tags, records = b8_data(requests_per_env=3)
    scores = {m: b8_scores(tags, passive=True) for m in V.MODELS}
    b8 = V.b8_readings(tags, records, scores, n_boot=0, exclude={"v08b8-plates-000"})
    assert b8["requests"] == 3 * 3 - 1


def test_m8g_rows_refuse_a_reordered_record_order_file():
    m8 = m8g_data(items=1)
    preds = m8g_preds(m8, two_ok=True, kind="seed2")
    with pytest.raises(ValueError, match="option order"):
        V.m8g_rows(m8, preds, kind="order")
    assert len(V.m8g_rows(m8, preds, kind="seed2")) == len(m8)


def test_jev_and_probe_readings():
    records = [{"id": "j1", "meta": {"part": "bench", "target": ["move"]}, "request": {"state": "s", "questions": {"move": {"type": "choice", "criteria": {"a": "", "b": ""}}}}},
               {"id": "j2", "meta": {"part": "many", "target": ["q"], "correct": "b"}, "request": {"state": "s", "questions": {"q": {"type": "choice", "criteria": {"a": "", "b": ""}}}}}]
    jev = [{"id": "j1", "questions": {"move": {"type": "choice", "answer": {"probabilities": {"a": 0.9, "b": 0.1}}}}},
           {"id": "j2", "questions": {"q": {"type": "choice", "answer": {"probabilities": {"a": 0.2, "b": 0.8}}}}}]
    scores = {"v08Fs0": {"answers": {"j1": {"move": {"single-s1": {"probabilities": {"a": 0.6, "b": 0.4}}, "balanced-s1": {"probabilities": {"a": 0.3, "b": 0.7}}}},
                                     "j2": {"q": {"single-s1": {"probabilities": {"a": 0.6, "b": 0.4}}}}}}}
    out = V.jev_readings(records, jev, scores)
    assert out["jev_accuracy_many"]["value"] == 1.0
    assert out["models"]["v08Fs0"]["bench/single-s1"]["agreement"] == 1.0 and out["models"]["v08Fs0"]["bench/balanced-s1"]["agreement"] == 0.0
    assert out["models"]["v08Fs0"]["many/single-s1"] == {"agreement": 0.0, "questions": 1, "accuracy": 0.0}
    rows = [{"id": "bins-46@27", "meta": {"n": 3, "right": "bin_002", "position": 2, "kind": "position"},
             "request": {"state": "s", "questions": {"bin": {"type": "choice", "criteria": {"bin_000": "", "bin_001": "", "bin_002": ""}}}}}]
    probe_scores = {"v08Fs0": {"answers": {"bins-46@27": {"bin": {"order-none": {"probabilities": {"bin_000": 0.5, "bin_001": 0.3, "bin_002": 0.2},
                                                                                 "codes": {"bin_000": "A", "bin_001": "B", "bin_002": "AA"}}}}}}}
    p = V.probe_readings(rows, probe_scores)["models"]["v08Fs0"]["bins-46@27"]["readings"]["order-none"]
    assert p["rank_right"] == 3 and p["two_letter_mass"] == pytest.approx(0.2) and p["right_code"] == "AA"


# --------------------------------------------------------------------------
# Files: frozen tags, score files, pods, write guards, refusals
# --------------------------------------------------------------------------


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_frozen_tags_check(tmp_path):
    b8, tags, frozen = tmp_path / "B8.jsonl", tmp_path / "tags.json", tmp_path / "frozen-before-scoring.json"
    b8.write_text('{"id": "v08b8-plates-000"}\n', encoding="utf-8")
    tags.write_text('{"requests": []}\n', encoding="utf-8")
    frozen.write_text(json.dumps({"tags_sha256": _sha(tags), "b8_sha256": _sha(b8), "frozen": "2026-10-16T13:00:00+09:00"}), encoding="utf-8")
    V._sha.cache_clear()
    content, problems = V.check_b8_frozen(b8, tags, frozen)
    assert problems == [] and content["b8_sha256"] == _sha(b8)
    tags.write_text('{"requests": [1]}\n', encoding="utf-8")
    V._sha.cache_clear()
    _, problems = V.check_b8_frozen(b8, tags, frozen)
    assert len(problems) == 1 and "tags.json" in problems[0]
    _, problems = V.check_b8_frozen(b8, tags, tmp_path / "absent.json")
    assert problems and "missing" in problems[0]


def test_bench_score_file_checks(tmp_path):
    source = tmp_path / "B8.jsonl"
    source.write_text("x\n", encoding="utf-8")
    frozen = {"tags_sha256": "t", "b8_sha256": "b"}
    good = {"kind": "v08-bench-scores", "set": "B8", "tag": "v08Fs0-2b", "input_sha256": _sha(source), "readings": {r: {} for r in V.B8_READINGS},
            "frozen": frozen, "checkpoint": None, "answers": {}}
    path = tmp_path / "B8-v08Fs0-2b.json"
    path.write_text(json.dumps(good), encoding="utf-8")
    V._sha.cache_clear()
    _, problems = V.check_bench_source(path, set_name="B8", tag="v08Fs0-2b", input_path=source, checkpoint=None, readings=V.B8_READINGS, frozen=frozen)
    assert problems == []
    path.write_text(json.dumps({**good, "frozen": {"tags_sha256": "other"}, "readings": {"balanced-s1": {}}}), encoding="utf-8")
    _, problems = V.check_bench_source(path, set_name="B8", tag="v08Fs0-2b", input_path=source, checkpoint=None, readings=V.B8_READINGS, frozen=frozen)
    assert any("frozen" in p for p in problems) and any("readings missing" in p for p in problems)
    path.write_text(json.dumps({**good, "checkpoint": {"sha256": "abc"}}), encoding="utf-8")
    _, problems = V.check_bench_source(path, set_name="B8", tag="v08Fs0-2b", input_path=source, checkpoint=None, readings=(), frozen=frozen)
    assert problems == ["a checkpoint was applied to the base"]


def test_m8g_source_checks(tmp_path):
    gate = tmp_path / "M8g.jsonl"
    gate.write_text("x\n", encoding="utf-8")
    pred = tmp_path / "M8g-seed2-base-2b.predictions.jsonl"
    pred.write_text("", encoding="utf-8")
    V._sha.cache_clear()
    pred.with_name("M8g-seed2-base-2b.json").write_text(json.dumps({"config": {"order_seed": 2, "orders": "seeded", "input_sha256": _sha(gate)}}), encoding="utf-8")
    assert V.check_m8g_source(pred, "seed2", None, gate=gate) == []
    problems = V.check_m8g_source(pred, "order", None, gate=gate)
    assert problems == ["order_seed = 2, want None"]


def test_r14_frozen_check_refuses_r14_only(tmp_path):
    spark, r14 = tmp_path / "spark", tmp_path / "r14"
    spark.mkdir()
    r14.mkdir()
    assert V.load_r14(spark, {}, [], directory=r14) is None  # no R14-<tag>.json: R14 is absent
    (spark / "R14-v08Fs0-2b.json").write_text("{}", encoding="utf-8")
    sample, tags = r14 / "R14-v08-sample.jsonl", r14 / "tags.json"
    sample.write_text('{"id": "v08r14-lab-0000"}\n', encoding="utf-8")
    tags.write_text(json.dumps(R14_TAGS), encoding="utf-8")
    frozen = {"files": {"R14-v08-sample.jsonl": _sha(sample), "tags.json": _sha(tags)}, "frozen": "2026-10-16T13:55:28+09:00"}
    (r14 / "frozen-before-scoring.json").write_text(json.dumps(frozen), encoding="utf-8")
    V._sha.cache_clear()
    good = {"kind": "v08-bench-scores", "set": "R14", "tag": "v08Fs0-2b", "input_sha256": _sha(sample), "readings": {r: {} for r in V.R14_READINGS},
            "frozen": frozen, "checkpoint": None, "answers": {}}
    (spark / "R14-v08Fs0-2b.json").write_text(json.dumps(good), encoding="utf-8")
    loaded = V.load_r14(spark, {"v08Fs0": None}, [], directory=r14)
    assert "refused" not in loaded and list(loaded["scores"]) == ["v08Fs0"] and loaded["record_ids"] == ["v08r14-lab-0000"]
    (spark / "R14-v08Fs1-2b.json").write_text(json.dumps({**good, "tag": "v08Fs1-2b", "frozen": {"files": {}}}), encoding="utf-8")
    loaded = V.load_r14(spark, {"v08Fs0": None, "v08Fs1": None}, [], directory=r14)
    assert list(loaded["scores"]) == ["v08Fs0"] and any("frozen" in p for p in loaded["left_out"])  # a file scored on other tags is left out
    tags.chmod(0o644)
    tags.write_text(json.dumps({"requests": []}), encoding="utf-8")
    refused = V.load_r14(spark, {}, [], directory=r14)
    assert refused == {"refused": ["tags.json changed after it was frozen"]}


def test_pod_file_wants_exactly_one(tmp_path):
    with pytest.raises(SystemExit, match="0 files"):
        V.pod_file(tmp_path, "T2")
    (tmp_path / "after-v08x-s0-2b-T2.predictions.jsonl").write_text("", encoding="utf-8")
    (tmp_path / "after-v08x-s0-2b-T2x.predictions.jsonl").write_text("", encoding="utf-8")
    assert V.pod_file(tmp_path, "T2").name == "after-v08x-s0-2b-T2.predictions.jsonl"
    (tmp_path / "before-v08x-s0-2b-T2.predictions.jsonl").write_text("", encoding="utf-8")
    with pytest.raises(SystemExit, match="2 files"):
        V.pod_file(tmp_path, "T2")


def test_write_guards():
    for place in (V.G8.GATES / "verdict.json", V.ARTIFACTS / "reports" / "v07" / "verdict.json", V.ARTIFACTS / "v08" / "jev" / "x.json",
                  V.ARTIFACTS / "simgen" / "x.json", V.ARTIFACTS / "t9" / "x.json"):
        with pytest.raises(SystemExit, match="refusing to write"):
            V.refuse_write(place)
    V.refuse_write(V.ARTIFACTS / "reports" / "v08" / "verdict.json")
    V.refuse_write(V.PIPELINE_OUT)


def test_main_refuses_without_runs_or_files(tmp_path):
    out = tmp_path / "verdict.json"
    with pytest.raises(SystemExit, match="required: --f-s0-dir"):
        V.main(["--out", str(out)])
    dirs = []
    for i in range(4):
        d = tmp_path / f"run{i}" / "reports"
        d.mkdir(parents=True)
        dirs.append(str(d))
    with pytest.raises(SystemExit, match="missing files the rule reads"):
        V.main(["--f-s0-dir", dirs[0], "--f-s1-dir", dirs[1], "--fp-s0-dir", dirs[2], "--fp-s1-dir", dirs[3], "--spark-dir", str(tmp_path / "spark"),
                "--out", str(out)])
    with pytest.raises(SystemExit, match="refusing to write"):
        V.main(["--pipeline-check", "--out", str(V.G8.GATES / "x.json")])
    assert not out.exists()


def test_rule_text_is_the_pre_declared_one():
    text = (REPO / ".superpowers" / "sdd" / "progress.md")
    if not text.is_file():  # the briefs live in the main checkout, not in a worktree
        text = REPO.parents[2] / ".superpowers" / "sdd" / "progress.md"
    if text.is_file():
        rule = text.read_text(encoding="utf-8").split("PRE-DECLARED v0.8 VERDICT RULE")[-1]
        for needle in ("≤ −0.05, with the upper bound of that mean < 0", "Passive-top share ≤ −0.15 (point) AND planner agreement ≥ −0.05 (point)",
                       "outside CA, mean of the three sets ≥ −0.05 (point), and GAP ≥ −0.03 (point)",
                       "lower bound > −0.05, and severe errors ≤ F's + 0.01 (arm means)", "G2 ≥ −0.02 (point), with lower bound > −0.045",
                       "≥ 0.8 × its top-1 accuracy", "F's accuracy at N ≤ 26 ≥ v0.7's − 0.02 (point)", "mean of the three sets, ≤ +0.02 (point)",
                       "flip rate in bins ≥ 27 > 0.20"):
            assert needle in rule, needle
    assert (V.P1_POINT, V.P2_PASSIVE, V.P2_PLANNER, V.CA_POINT, V.GAP_POINT, V.BAND_LOW, V.SEVERE_SLACK, V.G_POINT, V.G_LOW, V.FORMAT_RATIO, V.FORMAT_N26,
            V.OF_POINT, V.FLIP_LIMIT) == (-0.05, -0.15, -0.05, -0.05, -0.03, -0.05, 0.01, -0.02, -0.045, 0.8, -0.02, 0.02, 0.20)
