"""The v0.7 rule helper (task v0.7b §1-5) — ``scripts/verdict_v07.py``.

The pure rule is pinned on hand-made readings: the amended S-a (components len2k, len4k, len8k and cue; each active or inactive by v0.6's gap,
passing or failing — every threshold at its bound; PASS, FAIL and NOT TESTABLE) and S-b with every threshold at its bound, the outcomes of S
(SUPPORTED, PARTIAL either way, REJECTED, and with S-a not testable "S-b PASS (S-a not testable)" or REJECTED), the guards O / N / R / G / F at
their bounds, the release line and the pre-declared next step. The readings are pinned on synthetic predictions: M2g sheets that push v0.6
to "safe" calls but not v0.7 read S-a PASS (the same push for both arms reads FAIL; no push for v0.6 reads NOT TESTABLE; a push on the
reassuring sheet only activates the cue component alone; v0.7 creating a gap where v0.6 had none fails that inactive component); M1g listings
where v0.7 keeps its N = 8 accuracy at N ≥ 17 read S-b PASS (and v0.6's late-letter loss for both reads FAIL); together with passing guards
they give every outcome of S. Seed means (an arm's DC is the mean of its seeds' DCs on the same questions), the amendment's secondary readings,
the pre-declared exclusions and the gate-file checks are pinned too. No model, no sealed file.
"""

from __future__ import annotations

import functools
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest
from helpers import REPO


@functools.lru_cache(maxsize=None)
def V():
    spec = importlib.util.spec_from_file_location("v07_verdict_helper", REPO / "scripts" / "verdict_v07.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _d(value=None, low=None, high=None, a=None, b=None) -> dict:
    return {"value": value, "low": low, "high": high, "a": a, "b": b}


COMPONENTS = ("len2k", "len4k", "len8k", "cue")


def _sa(dgap=(-0.10, -0.15, -0.05), gap=0.0, gap6=0.10) -> dict:
    """One S-a component reading: Δgap (point, low, high), v0.7's gap and v0.6's gap (points)."""
    return {"dgap": _d(*dgap), "gap_v07": _d(gap), "gap_v06": _d(gap6)}


INACTIVE_OK = _sa(dgap=(0.01, -0.02, 0.04), gap=0.01, gap6=0.0)
SA_PASS = {c: _sa() for c in COMPONENTS}                                      # every component active and passing
SA_FAIL = {**SA_PASS, "len8k": _sa(dgap=(-0.02, -0.06, 0.02))}                # one active component fails
SA_NOT_TESTABLE = {c: INACTIVE_OK for c in COMPONENTS}                         # no component active
SB_PASS = {"drop_v07": _d(0.02), "ddrop": _d(-0.05, -0.08, -0.02), "physics_penalty_minus_base": _d(0.03)}
SB_FAIL = {**SB_PASS, "drop_v07": _d(0.08)}


def _r_ok() -> dict:
    return {s: {"noul.brier": _d(0.002, -0.001, 0.0099), "score.brier": _d(0.0, -0.004, 0.004),
                "choice/action.regret_mean[default]": _d(0.1, -0.2, 0.49)} for s in ("T1", "T3", "T4", "T5", "T7")}


def _readings(sa=SA_PASS, sb=SB_PASS, *, o=None, arm=None, home=None, r=None, g=None, f=None) -> dict:
    ok_home = {"band_balanced": _d(0.0, -0.01, 0.01), "severe_error": _d(0.01, 0.0, 0.02, a=0.031, b=0.021)}
    return {"S-a": sa, "S-b": sb, "O": {"A4": _d(0.0), "T9": _d(0.01), "T8b": _d(-0.02), **(o or {})},
            "N": {"arm": {"A4_direction": _d(0.0, -0.04, 0.05), "A4_comparison": _d(0.01, -0.03, 0.06), "T9_direction": _d(0.0, -0.049, 0.05),
                          "T9_comparison": _d(0.0, -0.02, 0.03), **(arm or {})}, "home": {"0": ok_home, "1": ok_home, **(home or {})}},
            "R": r or {"0": _r_ok(), "1": _r_ok()}, "G": g or {"diff": 0.0, "low": -0.02, "high": 0.02, "a": 0.70, "b": 0.70},
            "F": f or _d(0.0, a=0.10, b=0.12)}


# --------------------------------------------------------------------------
# S-a, S-b and S
# --------------------------------------------------------------------------


@pytest.mark.parametrize("entry, status, passes, key, expected", [
    (_sa(gap6=0.05), "active", True, None, None),                                      # v0.6's gap exactly +0.05 is active (≥)
    (_sa(gap6=0.0499, dgap=(0.02, 0.0, 0.04)), "inactive", True, None, None),          # just below: inactive, Δgap not read
    (_sa(dgap=(-0.05, -0.1, -0.01)), "active", True, "dgap_le_-0.05", True),           # active: Δgap exactly −0.05 counts (≤)
    (_sa(dgap=(-0.0499, -0.1, -0.01)), "active", False, "dgap_le_-0.05", False),
    (_sa(dgap=(-0.1, -0.2, 0.0)), "active", False, "dgap_high_lt_0", False),          # an upper bound of exactly 0 does not (<)
    (_sa(dgap=(-0.1, -0.2, -1e-9)), "active", True, "dgap_high_lt_0", True),
    (_sa(gap=0.03), "active", True, "gap_v07_le_+0.03", True),                         # v0.7's gap exactly +0.03 counts (≤, point)
    (_sa(gap=0.0301), "active", False, "gap_v07_le_+0.03", False),
    (_sa(gap6=0.0, gap=0.03, dgap=(0.03, 0.01, 0.05)), "inactive", True, "gap_v07_le_+0.03", True),    # inactive passes on v0.7's gap alone
    (_sa(gap6=0.0, gap=0.0301, dgap=(0.03, 0.01, 0.05)), "inactive", False, "gap_v07_le_+0.03", False),  # v0.7 must not create a gap
    ({"dgap": _d(-0.1, -0.2, -0.05), "gap_v07": _d(0.0), "gap_v06": _d()}, "missing", False, None, None),   # no v0.6 gap: missing, fails
])  # fmt: skip
def test_each_s_a_component_state_and_threshold_at_its_bound(entry, status, passes, key, expected):
    component = V().sa_component(entry)
    assert component["status"] == status and component["pass"] is passes
    if key is not None:
        assert component["checks"][key] is expected
    if status == "inactive":
        assert set(component["checks"]) == {"gap_v07_le_+0.03"}, "an inactive component is read on v0.7's gap only"


@pytest.mark.parametrize("components, outcome, active", [
    (SA_PASS, "PASS", list(COMPONENTS)),
    ({**SA_NOT_TESTABLE, "cue": _sa()}, "PASS", ["cue"]),                                   # one active component suffices
    ({**SA_NOT_TESTABLE, "len4k": _sa()}, "PASS", ["len4k"]),
    (SA_FAIL, "FAIL", list(COMPONENTS)),                                                   # an active component fails
    ({**SA_NOT_TESTABLE, "cue": _sa(), "len2k": _sa(gap6=0.0, gap=0.04)}, "FAIL", ["cue"]),  # an inactive one fails (v0.7 made a gap)
    ({**SA_NOT_TESTABLE, "cue": _sa(dgap=(-0.04, -0.08, -0.01))}, "FAIL", ["cue"]),
    (SA_NOT_TESTABLE, "NOT TESTABLE", []),                                                 # none active
    ({**SA_NOT_TESTABLE, "len8k": _sa(gap6=0.0, gap=0.08)}, "NOT TESTABLE", []),           # none active, even if v0.7 creates a gap
    ({c: SA_PASS[c] for c in ("len2k", "len4k", "len8k")}, "FAIL", list(COMPONENTS[:3])),  # a missing component fails
    (None, "FAIL", []),
])  # fmt: skip
def test_the_amended_s_a_outcomes(components, outcome, active):
    result = V().sa_checks(components)
    assert result["outcome"] == outcome and result["active"] == active
    assert result["pass"] is (outcome == "PASS") and result["testable"] is (outcome != "NOT TESTABLE")
    created = [c for c, e in (components or {}).items() if e["gap_v06"]["value"] < 0.05 and e["gap_v07"]["value"] > 0.03]
    assert result["inactive_failing"] == created, "an inactive component where v0.7 made a gap is listed (reported for NOT TESTABLE)"


def test_s_a_reads_only_the_required_components():
    v = V()
    three = ("len2k", "len4k", "len8k")
    assert v.sa_checks({c: SA_PASS[c] for c in three}, required=three)["outcome"] == "PASS"
    assert v.sa_checks({c: SA_PASS[c] for c in three})["outcome"] == "FAIL" and v.sa_checks({c: SA_PASS[c] for c in three})["missing"] == ["cue"]
    assert tuple(v.SA_COMPONENTS) == COMPONENTS and v.SA_COMPONENTS["cue"] == "w4k-reassuring" and v.SA_COMPONENTS["len4k"] == "w4k"


@pytest.mark.parametrize("change, key, expected", [
    ({"drop_v07": _d(0.05)}, "drop_v07_le_0.05", True),                          # exactly 0.05 counts (≤, point)
    ({"drop_v07": _d(0.0501)}, "drop_v07_le_0.05", False),
    ({"ddrop": _d(-0.05, -0.1, 0.0)}, "ddrop_high_lt_0", False),                  # an upper bound of exactly 0 does not (<)
    ({"ddrop": _d(-0.05, -0.1, -1e-9)}, "ddrop_high_lt_0", True),
    ({"physics_penalty_minus_base": _d(0.10)}, "physics_penalty_minus_base_lt_0.10", False),   # exactly +0.10 fails (<)
    ({"physics_penalty_minus_base": _d(0.0999)}, "physics_penalty_minus_base_lt_0.10", True),
    ({"physics_penalty_minus_base": None}, "physics_penalty_minus_base_lt_0.10", False),
])  # fmt: skip
def test_every_s_b_threshold_at_its_bound(change, key, expected):
    assert V().sb_checks({**SB_PASS, **change})["checks"][key] is expected


@pytest.mark.parametrize("sa, sb, outcome", [
    (SA_PASS, SB_PASS, "SUPPORTED"), (SA_PASS, SB_FAIL, "PARTIAL"), (SA_FAIL, SB_PASS, "PARTIAL"), (SA_FAIL, SB_FAIL, "REJECTED"),
    (SA_NOT_TESTABLE, SB_PASS, "S-b PASS (S-a not testable)"), (SA_NOT_TESTABLE, SB_FAIL, "REJECTED"),
    (None, None, "REJECTED"),
])  # fmt: skip
def test_each_s_outcome(sa, sb, outcome):
    v = V()
    assert v.s_outcome(v.sa_checks(sa), v.sb_checks(sb)) == outcome


def test_the_next_step_follows_the_failing_part():
    v = V()
    assert v.next_step("PASS", True)["next"] == [v.NEXT["SUPPORTED"]] and v.next_step(True, True)["branch"] == "SUPPORTED"
    assert v.next_step("FAIL", True)["next"] == [v.NEXT["S-a"]] and "cue hypothesis" in v.NEXT["S-a"]
    assert v.next_step("PASS", False)["next"] == [v.NEXT["S-b"]] and "multi-candidate" in v.NEXT["S-b"]
    assert v.next_step("FAIL", False)["next"] == [v.NEXT["S-a"], v.NEXT["S-b"]]
    not_testable = v.next_step("NOT TESTABLE", True)
    assert not_testable["branch"] == "S-b PASS (S-a not testable)" and not_testable["next"] == [v.NEXT["S-a not testable"]]
    assert "no claim about context robustness" in v.NEXT["S-a not testable"] and v.NEXT["S-a"] not in not_testable["next"]
    assert v.next_step("NOT TESTABLE", False)["next"] == [v.NEXT["S-a not testable"], v.NEXT["S-b"]]


# --------------------------------------------------------------------------
# guards and the release line
# --------------------------------------------------------------------------


def test_the_release_line_needs_s_supported_and_every_guard():
    v = V()
    good = v.decide(_readings())
    assert good["release_candidate"] is True and good["release_line"].startswith("RELEASE CANDIDATE") and all(good["passes"].values())
    assert "option cap to 26" in good["release_line"] and good["S"]["outcome"] == "SUPPORTED"
    bad_cases = (
        (dict(o={"T9": _d(0.0201)}), "O"),
        (dict(o={"T8b": None}), "O"),                                       # a missing reading fails
        (dict(arm={"A4_direction": _d(-0.02, -0.05, 0.01)}), "N"),          # a lower bound of exactly −0.05 fails (>)
        (dict(arm={"T9_comparison": None}), "N"),
        (dict(home={"1": {"band_balanced": _d(-0.01, -0.06, 0.03), "severe_error": _d(0.0, a=0.02, b=0.02)}}), "N"),
        (dict(home={"0": {"band_balanced": _d(0.0, -0.01, 0.01), "severe_error": _d(0.012, a=0.0311, b=0.021)}}), "N"),
        (dict(r={"0": _r_ok(), "1": {**_r_ok(), "T4": {**_r_ok()["T4"], "noul.brier": _d(0.0, -0.01, 0.01)}}}), "R"),
        (dict(r={"0": _r_ok()}), "R"),                                      # both seeds are read
        (dict(g={"diff": -0.0201, "low": -0.04, "high": 0.0}), "G"),
        (dict(g={"diff": -0.01, "low": -0.045, "high": 0.02}), "G"),
        (dict(f=_d(0.001, a=0.121, b=0.12)), "F"),
    )  # fmt: skip
    for change, part in bad_cases:
        result = v.decide(_readings(**change))
        assert result["release_candidate"] is False and f"{part} fails" in result["release_line"] and result["passes"][part] is False, (part, change)
    at_bounds = v.decide(_readings(o={"A4": _d(0.02), "T9": _d(0.02), "T8b": _d(0.02)}, g={"diff": -0.02, "low": -0.0449, "high": 0.0},
                                   f=_d(0.0, a=0.12, b=0.12), arm={"A4_direction": _d(0.0, -0.0499, 0.05)},
                                   home={"0": {"band_balanced": _d(0.0, -0.0499, 0.01), "severe_error": _d(0.01, a=0.031, b=0.021)}}))
    assert at_bounds["release_candidate"] is True, "+0.02 (O), −0.02 / −0.0449 (G), equal (F), > −0.05 (N), +0.01 slack (N severe) pass"
    partial = v.decide(_readings(SA_FAIL))
    assert partial["S"]["outcome"] == "PARTIAL" and partial["release_line"] == "not a release candidate: S PARTIAL"
    assert partial["next_step"]["next"] == [v.NEXT["S-a"]]
    only_b = v.decide(_readings(SA_NOT_TESTABLE))
    assert only_b["S"]["outcome"] == "S-b PASS (S-a not testable)" and only_b["release_candidate"] is False
    assert only_b["release_line"] == "not a release candidate: S S-b PASS (S-a not testable)", "release needs S SUPPORTED (unchanged)"
    assert v.decide(_readings(SA_NOT_TESTABLE, SB_FAIL))["S"]["outcome"] == "REJECTED"
    three = ("len2k", "len4k", "len8k")
    assert v.decide(_readings({c: SA_PASS[c] for c in three}), sa_required=three)["S"]["outcome"] == "SUPPORTED"


# --------------------------------------------------------------------------
# readings from synthetic predictions
# --------------------------------------------------------------------------


def _u(*parts) -> float:
    return int.from_bytes(hashlib.sha256("|".join(map(str, parts)).encode()).digest()[:8], "big") / 2**64


def _m2g_records(pairs: int = 60) -> list[dict]:
    """Synthetic M2g moments: one-fact pairs with two yes/no questions and a score question; truth classes from robustness and label bands."""
    out = []
    for g in range(pairs):
        for member in ("", "p"):
            rid = f"v07m2g-v06a4-bank-{g:06d}{member}"
            partner = f"v07m2g-v06a4-bank-{g:06d}{'p' if not member else ''}"
            labels = {"a": {"type": "noul", "p_yes": 0.9 if g % 3 == 0 else 0.5}, "b": {"type": "noul", "p_yes": 0.1},
                      "c": {"type": "score", "value": 0.5}}
            out.append({"schema": "simgen-request/0", "id": rid, "generator": {"family": "bank", "group": g},
                        "meta": {"set": "v07-m2g", "pair": {"of": partner}, "source": {"set": "A4" if g % 2 else "Hset", "id": rid[7:]},
                                 "gate": {"style": ("ticket", "handover", "console")[g % 3]},
                                 "robustness": {"a": {"robust": g % 3 == 0}, "b": {"robust": True}, "c": {"robust": False}}},
                        "request": {"state": "s", "questions": {"a": {"type": "noul", "instructions": "x"}, "b": {"type": "noul", "instructions": "y"},
                                                                 "c": {"type": "score", "instructions": "z", "criteria": ["0", "1"]}}},
                        "labels": labels})
    return out


def _m2g_preds(records: list[dict], safe_share: float, model: str) -> list[dict]:
    """Each question is called "safe" (answer 0.1) iff its fixed draw u < `safe_share`, else "look again" (0.5) — the same u in every
    condition and for every model, so a condition's DC moves by the change of `safe_share`."""
    out = []
    for r in records:
        questions = {}
        for name, q in r["request"]["questions"].items():
            x = 0.1 if _u(r["id"], name) < safe_share else 0.5
            answer = {"p_yes": x} if q["type"] == "noul" else {"value": x}
            questions[name] = {"type": q["type"], "n_listings": 1, "answer": answer, "listings": [{"options": ["yes", "no"], "answer": answer}]}
        out.append({"id": r["id"], "model": model, "questions": questions})
    return out


def _share(spec, cond: str) -> float:
    """``(short, wrapped)`` (every sheet the same) or ``{condition: share}`` (``default`` for the rest)."""
    if isinstance(spec, dict):
        return spec.get(cond, spec["default"])
    return spec[0] if cond == "short" else spec[1]


def _m2g(shares: dict) -> dict:
    """M2g readings for per-model safe shares ``{model: spec}`` (:func:`_share`)."""
    records = _m2g_records()
    conditions = {c: records for c in V().CONDITIONS}
    preds = {m: {c: _m2g_preds(records, _share(spec, c), m) for c in conditions} for m, spec in shares.items()}
    return V().m2g_readings(conditions, preds, n_boot=300, seed=0)


SHARES_V07_FLAT = {"v07s0": (0.25, 0.25), "v07s1": (0.30, 0.30), "v06s0": (0.25, 0.45), "v06s1": (0.25, 0.40), "base": (0.40, 0.45)}
SHARES_BOTH_PUSHED = {**SHARES_V07_FLAT, "v07s0": (0.25, 0.45), "v07s1": (0.25, 0.40)}
SHARES_V06_FLAT = {**SHARES_V07_FLAT, "v06s0": (0.25, 0.25), "v06s1": (0.30, 0.30)}                       # no component active
SHARES_CUE_ONLY = {**SHARES_V06_FLAT, "v06s0": {"default": 0.25, "w4k-reassuring": 0.45}, "v06s1": {"default": 0.30, "w4k-reassuring": 0.48}}
SHARES_V07_MAKES_A_GAP = {**SHARES_CUE_ONLY, "v07s0": {"default": 0.25, "w8k": 0.40}, "v07s1": {"default": 0.30, "w8k": 0.45}}


def test_m2g_sheets_that_push_v06_but_not_v07_read_s_a_pass_and_the_same_push_reads_fail():
    v = V()
    flat = _m2g(SHARES_V07_FLAT)
    assert v.sa_checks(flat["S-a"])["pass"] is True
    for length in ("2k", "4k", "8k"):
        e = flat["S-a"][f"len{length}"]
        assert e["gap_v07"]["value"] == pytest.approx(0.0, abs=1e-12) and e["dgap"]["value"] < -0.10 and e["dgap"]["high"] < 0
        assert e["gap_v06"]["value"] == pytest.approx((flat["gaps"]["v06s0"][length]["value"] + flat["gaps"]["v06s1"][length]["value"]) / 2)
    pushed = _m2g(SHARES_BOTH_PUSHED)
    assert v.sa_checks(pushed["S-a"])["pass"] is False and abs(pushed["S-a"]["len4k"]["dgap"]["value"]) < 1e-12
    for m in ("v07", "v06"):  # seed means: the arm's DC is the mean of its seeds' DCs on the same questions
        seeds = v.ARMS[m]
        assert flat["levels"][m]["short"]["value"] == pytest.approx(sum(flat["levels"][s]["short"]["value"] for s in seeds) / 2)
    assert set(flat["by_style"]) == {"ticket", "handover", "console"} and set(flat["by_set"]) == {"A4", "Hset"}
    tone = flat["tone_panel"]
    assert tone["v06s0"]["reassuring_minus_neutral"]["value"] == pytest.approx(0.0) and "v07_minus_v06" in tone
    assert flat["vs_base"]["short"]["v07"]["pooled"]["value"] < 0, "v0.7's short-form DC is below the base's here"
    assert set(flat["S-a"]) == set(COMPONENTS) and v.sa_checks(flat["S-a"])["active"] == list(COMPONENTS)


@pytest.mark.parametrize("shares, outcome, active", [
    (SHARES_V06_FLAT, "NOT TESTABLE", []),
    (SHARES_CUE_ONLY, "PASS", ["cue"]),
    (SHARES_V07_MAKES_A_GAP, "FAIL", ["cue"]),
])  # fmt: skip
def test_m2g_component_activity_follows_v06s_own_gaps(shares, outcome, active):
    v = V()
    readings = _m2g(shares)
    result = v.sa_checks(readings["S-a"])
    assert result["outcome"] == outcome and result["active"] == active
    if shares is SHARES_V07_MAKES_A_GAP:
        assert result["components"]["len8k"]["status"] == "inactive" and result["components"]["len8k"]["pass"] is False
        assert result["components"]["cue"]["pass"] is True


def test_the_amendment_secondary_readings_are_per_arm_and_compared():
    readings = _m2g({**SHARES_V07_FLAT, "v06s0": {"default": 0.25, "w4k-alarming": 0.10, "w4k-reassuring": 0.45, "w4k": 0.30},
                     "v06s1": {"default": 0.25, "w4k-alarming": 0.10, "w4k-reassuring": 0.45, "w4k": 0.30}})
    extra = readings["amendment_secondary"]
    alarm, reassure = extra["alarming_minus_short"], extra["reassuring_minus_neutral"]
    assert {"v07", "v06", "v07s0", "base"} <= set(alarm) and alarm["v06"]["value"] < -0.05 and abs(alarm["v07"]["value"]) < 1e-12
    assert reassure["v06"]["value"] > 0.1 and abs(reassure["v07"]["value"]) < 1e-12
    assert alarm["v07_minus_v06"]["dgap"]["value"] == pytest.approx(alarm["v07"]["value"] - alarm["v06"]["value"])


def test_m2g_exclusions_leave_the_moments_out_of_every_reading():
    records = _m2g_records()
    conditions = {c: records for c in V().CONDITIONS}
    preds = {m: {c: _m2g_preds(records, 0.3, m) for c in conditions} for m in SHARES_V07_FLAT}
    out = V().m2g_readings(conditions, preds, n_boot=50, seed=0, exclude={records[0]["id"], records[1]["id"]})
    assert out["records"]["short"] == len(records) - 2 and out["S-a"]["len2k"]["dgap"]["n"] == 3 * (len(records) - 2)


def _m1g_records(items: int = 40, physics_every: int = 4) -> list[dict]:
    out = []
    for i in range(items):
        source = "a4" if i % physics_every == 0 else "MMLU"
        for n in (8, 21, 26):
            for listing in range(3):
                position = {8: (1, 4, 7), 21: (2, 10, 18), 26: (3, 11, 20)}[n][listing]
                keys = [f"o{k}" for k in range(n)]
                out.append({"schema": "simgen-request/0", "id": f"v07m1g-{i:05d}-n{n:02d}-{chr(65 + position)}",
                            "meta": {"set": "v07-m1g", "item": f"it{i}", "source": source, "n": n, "position": position, "letter": chr(65 + position),
                                     "listing": listing, "bucket": None if n < 17 else ("A-H" if position < 8 else "I-P" if position < 16 else "Q-Z"),
                                     "correct": keys[position], "rank": i},
                            "request": {"state": "s", "questions": {"q": {"type": "choice", "instructions": "x", "criteria": {k: k for k in keys}}}},
                            "labels": {"q": {"type": "choice", "kind": "probe", "probabilities": {k: float(k == keys[position]) for k in keys}}}})
    return out


def _m1g_preds(records: list[dict], wrong) -> list[dict]:
    """The model answers the correct option unless ``wrong(record meta)``; then it answers the option after it."""
    out = []
    for r in records:
        keys = list(r["request"]["questions"]["q"]["criteria"])
        target = r["meta"]["position"]
        pick = (target + 1) % len(keys) if wrong(r["meta"]) else target
        probs = [0.9 if k == pick else 0.1 / (len(keys) - 1) for k in range(len(keys))]
        out.append({"id": r["id"], "questions": {"q": {"type": "choice", "n_listings": 1, "listings": [{"options": keys, "code_probs": probs, "code_mass": 0.99}]}}})
    return out


def _late_q_z(meta) -> bool:
    return meta["n"] >= 17 and meta["bucket"] == "Q-Z"


def test_m1g_where_v07_keeps_its_accuracy_at_many_options_reads_s_b_pass_and_v06s_late_loss_for_both_reads_fail():
    v = V()
    records = _m1g_records()
    never = lambda meta: False  # noqa: E731
    good = v.m1g_readings(records, {"v07s0": _m1g_preds(records, never), "v07s1": _m1g_preds(records, never), "v06s0": _m1g_preds(records, _late_q_z),
                                    "v06s1": _m1g_preds(records, _late_q_z), "base": _m1g_preds(records, never)}, n_boot=200, seed=0)
    sb = good["S-b"]
    assert sb["drop_v07"]["value"] == 0.0 and sb["drop_v06"]["value"] == pytest.approx(1 / 3) and sb["ddrop"]["high"] < 0
    assert sb["physics_penalty_minus_base"]["value"] == 0.0 and v.sb_checks(sb)["pass"] is True
    assert good["items"] == 40 and good["items_physics"] == 10 and good["models"]["v06"]["penalty"]["value"] == pytest.approx(1.0)
    assert good["letters"]["v06"]["26"] == {"D": 1.0, "L": 1.0, "U": 0.0} and good["letters"]["v06"]["21"] == {"C": 1.0, "K": 1.0, "S": 0.0}
    assert good["letters"]["v07"]["26"] == {"D": 1.0, "L": 1.0, "U": 1.0} and good["weak_letters"]["v06"]["26"] == {}
    bad = v.m1g_readings(records, {m: _m1g_preds(records, _late_q_z) for m in ("v07s0", "v07s1", "v06s0", "v06s1")} | {"base": _m1g_preds(records, never)},
                         n_boot=200, seed=0)
    assert v.sb_checks(bad["S-b"])["pass"] is False and bad["S-b"]["physics_penalty_minus_base"]["value"] == pytest.approx(1.0)
    excluded = v.m1g_readings(records, {m: _m1g_preds(records, never) for m in v.GATE_MODELS}, n_boot=20, seed=0, exclude={"it0", "it1"})
    assert excluded["items"] == 38 and excluded["excluded"] == ["it0", "it1"]


def _standard_ok() -> dict:
    r = _readings()
    return {"O": r["O"], "N": r["N"], "R": r["R"], "G": r["G"], "F": r["F"]}


@pytest.mark.parametrize("shares, m1_v07_wrong, outcome", [
    (SHARES_V07_FLAT, False, "SUPPORTED"), (SHARES_V07_FLAT, True, "PARTIAL"), (SHARES_BOTH_PUSHED, False, "PARTIAL"),
    (SHARES_BOTH_PUSHED, True, "REJECTED"), (SHARES_V06_FLAT, False, "S-b PASS (S-a not testable)"), (SHARES_V06_FLAT, True, "REJECTED"),
    (SHARES_CUE_ONLY, False, "SUPPORTED"), (SHARES_V07_MAKES_A_GAP, False, "PARTIAL"),
])  # fmt: skip
def test_synthetic_predictions_give_each_outcome_through_the_whole_rule(shares, m1_v07_wrong, outcome):
    v = V()
    m2g = _m2g(shares)
    records = _m1g_records()
    never = lambda meta: False  # noqa: E731
    v07_rule = _late_q_z if m1_v07_wrong else never
    m1g = v.m1g_readings(records, {"v07s0": _m1g_preds(records, v07_rule), "v07s1": _m1g_preds(records, v07_rule), "v06s0": _m1g_preds(records, _late_q_z),
                                   "v06s1": _m1g_preds(records, _late_q_z), "base": _m1g_preds(records, never)}, n_boot=200, seed=0)
    verdict = v.decide(v.assemble(m2g, m1g, _standard_ok()))
    assert verdict["S"]["outcome"] == outcome
    assert verdict["release_candidate"] is (outcome == "SUPPORTED")


# --------------------------------------------------------------------------
# files
# --------------------------------------------------------------------------


def test_gate_prediction_files_are_checked_for_conditions_input_and_checkpoint(tmp_path):
    v = V()
    from robo_jev.decisions.prompt import PROMPT_VERSION, fixed_text_digest
    from robo_jev.decisions.records import file_sha256

    data = tmp_path / "M2g-w4k.jsonl"
    data.write_text('{"id": "x"}\n', encoding="utf-8")
    ckpt = tmp_path / "checkpoint.pt"
    ckpt.write_bytes(b"weights")
    preds = tmp_path / "M2g-w4k-v07s0-2b.predictions.jsonl"
    preds.write_text("", encoding="utf-8")
    config = {**v.M2G_CONDITIONS, "input_sha256": file_sha256(data), "prompt_digest": fixed_text_digest(PROMPT_VERSION),
              "checkpoint": {"sha256": file_sha256(ckpt)}}
    summary = tmp_path / "M2g-w4k-v07s0-2b.json"
    summary.write_text(json.dumps({"config": config}), encoding="utf-8")
    assert v.check_gate_source(preds, data, v.M2G_CONDITIONS, ckpt)["problems"] == []
    summary.write_text(json.dumps({"config": {**config, "orders": "counterbalanced"}}), encoding="utf-8")
    assert any("orders" in p for p in v.check_gate_source(preds, data, v.M2G_CONDITIONS, ckpt)["problems"])
    summary.write_text(json.dumps({"config": {**config, "input_sha256": "0" * 64}}), encoding="utf-8")
    assert any("scored" in p for p in v.check_gate_source(preds, data, v.M2G_CONDITIONS, ckpt)["problems"])
    summary.write_text(json.dumps({"config": config}), encoding="utf-8")
    other = tmp_path / "other-checkpoint.pt"  # (file hashes are cached per path within a run)
    other.write_bytes(b"other weights")
    assert any("checkpoint" in p for p in v.check_gate_source(preds, data, v.M2G_CONDITIONS, other)["problems"])
    assert any("checkpoint" in p for p in v.check_gate_source(preds, data, v.M2G_CONDITIONS, tmp_path / "missing.pt")["problems"])
    assert any("base" in p for p in v.check_gate_source(preds, data, v.M2G_CONDITIONS, None)["problems"])


def test_the_verdict_never_writes_into_earlier_or_sealed_folders_and_needs_both_runs():
    v = V()
    for bad in ("reports/v06/verdict.json", "iface/verdict.json", "v07/gates/verdict.json", "t9/x.json", "simgen/v07/x.json"):
        with pytest.raises(SystemExit, match="refusing to write"):
            v.refuse_write(REPO / "artifacts" / bad)
    v.refuse_write(REPO / "artifacts" / "reports" / "v07" / "verdict.json")
    with pytest.raises(SystemExit, match="--s0-dir"):
        v.main([])
    paths = v.gate_paths(REPO / "artifacts" / "reports" / "v07" / "spark")
    assert paths["M1g"]["v07s0"].name == "M1g-v07s0-2b.predictions.jsonl"
    assert paths["M2g"]["base"]["w4k-alarming"].name == "M2g-w4k-alarming-base-2b.predictions.jsonl"
