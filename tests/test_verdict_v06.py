"""The v0.6 rule helper (Task v0.6b §1-4) — ``scripts/verdict_v06.py``.

The pure rule is pinned on hand-made readings: each outcome of (S) (SUPPORTED, SAFER-ONLY, REJECTED, PARTIAL), the per-seed consistency
condition, every threshold at its bound, the guards F / N / R / G, the release line and the pre-declared next step. The seed means are pinned on
synthetic A4-shaped records (v0.3b's record contract): an arm's call distribution is the mean of its two seeds' 1/L-weighted listings, so every
call reading of the arm is the mean of the seeds' readings on the same items; a hedge targeted at unclear situations in both seeds reads
SUPPORTED, a uniform hedge SAFER-ONLY, no change REJECTED, and one seed far worse than the reference mean PARTIAL (consistency). The truth
classes, A4's comparison mapping (an author's own scene) and the G2 arm reading are pinned too. No model, no sealed file.
"""

from __future__ import annotations

import argparse
import copy
import functools
import importlib.util
import json
import random
import sys
from pathlib import Path

import pytest
from helpers import REPO

CONDITIONS = {"mode": "direct", "format": "chat", "prompt_version": "v0c-p3", "orders": "counterbalanced", "shuffle_seed": 1}


@functools.lru_cache(maxsize=None)
def V():
    spec = importlib.util.spec_from_file_location("v06_verdict_helper", REPO / "scripts" / "verdict_v06.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _d(value=None, low=None, high=None, a=None, b=None) -> dict:
    return {"value": value, "low": low, "high": high, "a": a, "b": b}


def _s(dc: dict, gap: dict, ca: dict) -> dict:
    return {"DC": dc, "GAP": gap, "CA": ca}


SUPPORT = _s(_d(-0.10, -0.15, -0.05), _d(0.10, 0.05, 0.15), _d(-0.02, -0.06, 0.02))
SAFER_GAP = _s(_d(-0.10, -0.15, -0.05), _d(0.01, -0.03, 0.05), _d(0.0, -0.03, 0.03))     # ΔDC holds, ΔGAP fails
SAFER_CA = _s(_d(-0.10, -0.15, -0.05), _d(0.10, 0.05, 0.15), _d(-0.15, -0.20, -0.10))    # ΔDC and ΔGAP hold, ΔCA fails
REJECT = _s(_d(0.0, -0.01, 0.01), _d(0.0, -0.02, 0.02), _d(0.0, -0.02, 0.02))
NEITHER = _s(_d(-0.04, -0.08, 0.0), _d(0.06, 0.01, 0.11), _d(0.0, -0.02, 0.02))           # ΔDC too small, its lower bound ≤ −0.02
CONSISTENT = {"pass": True}
INCONSISTENT = {"pass": False}


# --------------------------------------------------------------------------
# (S) and consistency
# --------------------------------------------------------------------------


@pytest.mark.parametrize("diff, consistent, outcome", [
    (SUPPORT, CONSISTENT, "SUPPORTED"),
    (SAFER_GAP, CONSISTENT, "SAFER-ONLY"),
    (SAFER_CA, CONSISTENT, "SAFER-ONLY"),
    (REJECT, CONSISTENT, "REJECTED"),
    (REJECT, INCONSISTENT, "REJECTED"),         # rejection reads ΔDC only
    (NEITHER, CONSISTENT, "PARTIAL"),
    (SUPPORT, INCONSISTENT, "PARTIAL"),         # every pooled condition holds but one v0.6 seed is not below the v0.4 mean
    (SAFER_GAP, INCONSISTENT, "PARTIAL"),
    (None, CONSISTENT, "PARTIAL"),
])  # fmt: skip
def test_each_s_outcome_including_the_consistency_condition(diff, consistent, outcome):
    assert V().s_outcome(diff, consistent)["outcome"] == outcome


@pytest.mark.parametrize("diff, key, expected", [
    (_s(_d(-0.05, -0.1, -0.01), _d(), _d()), "dDC_le_-0.05", True),     # exactly −0.05 counts (≤)
    (_s(_d(-0.0499, -0.1, -0.01), _d(), _d()), "dDC_le_-0.05", False),
    (_s(_d(-0.1, -0.2, 0.0), _d(), _d()), "dDC_high_lt_0", False),      # an upper bound of exactly 0 does not (<)
    (_s(_d(-0.1, -0.2, -1e-9), _d(), _d()), "dDC_high_lt_0", True),
    (_s(_d(), _d(0.05, 0.01, 0.1), _d()), "dGAP_ge_+0.05", True),       # exactly +0.05 counts (≥)
    (_s(_d(), _d(0.0499, 0.01, 0.1), _d()), "dGAP_ge_+0.05", False),
    (_s(_d(), _d(0.1, 0.0, 0.2), _d()), "dGAP_low_gt_0", False),        # a lower bound of exactly 0 does not (>)
    (_s(_d(), _d(0.1, 1e-9, 0.2), _d()), "dGAP_low_gt_0", True),
    (_s(_d(), _d(), _d(-0.10, -0.2, 0.0)), "dCA_ge_-0.10", True),       # exactly −0.10 counts (≥, point)
    (_s(_d(), _d(), _d(-0.1001, -0.2, 0.0)), "dCA_ge_-0.10", False),
    (_s(_d(0.0, -0.02, 0.02), _d(), _d()), "dDC_low_gt_-0.02", False),  # rejection needs the lower bound above −0.02 (>)
    (_s(_d(0.0, -0.0199, 0.02), _d(), _d()), "dDC_low_gt_-0.02", True),
])  # fmt: skip
def test_every_s_threshold_at_its_bound(diff, key, expected):
    assert V().s_checks(diff, CONSISTENT)[key] is expected


def test_consistency_needs_each_v06_seed_strictly_below_the_v04_seed_mean():
    v = V()
    assert v.consistency({0: 0.30, 1: 0.39}, 0.40)["pass"] is True
    at = v.consistency({0: 0.30, 1: 0.40}, 0.40)
    assert at["pass"] is False and at["per_seed"] == {0: True, 1: False}, "equal is not below"
    assert v.consistency({0: 0.30}, 0.40)["pass"] is False and v.consistency({0: 0.30, 1: None}, 0.40)["pass"] is False
    assert v.consistency({0: 0.30, 1: 0.35}, None)["pass"] is False
    assert not any(v.s_checks(None, None).values())


# --------------------------------------------------------------------------
# guards, release, next step
# --------------------------------------------------------------------------


def _r_ok() -> dict:
    return {s: {"noul.brier": _d(0.002, -0.001, 0.0099), "score.brier": _d(0.0, -0.004, 0.004),
                "choice/action.regret_mean[default]": _d(0.1, -0.2, 0.49)} for s in ("T1", "T3", "T4", "T5", "T7")}


def _readings(s: dict = SUPPORT, *, consistent: dict = CONSISTENT, f: dict | None = None, arm: dict | None = None, home: dict | None = None,
              r: dict | None = None, g: dict | None = None) -> dict:
    ok_home = {"band_balanced": _d(0.0, -0.01, 0.01), "severe_error": _d(0.01, 0.0, 0.02, a=0.031, b=0.021)}
    return {"S": s, "consistency": consistent, "F": f or _d(0.0, a=0.10, b=0.12),
            "N": {"arm": {"A4_direction": _d(0.0, -0.04, 0.05), "A4_comparison": _d(0.01, -0.03, 0.06), "T9_direction": _d(0.0, -0.049, 0.05),
                          "T9_comparison": _d(0.0, -0.02, 0.03), **(arm or {})},
                  "home": {"0": ok_home, "1": ok_home, **(home or {})}},
            "R": r or {"0": _r_ok(), "1": _r_ok()}, "G": g or {"diff": 0.0, "low": -0.02, "high": 0.02, "a": 0.70, "b": 0.70}}


def test_the_release_line_needs_s_supported_and_every_guard():
    v = V()
    good = v.decide(_readings())
    assert good["release_candidate"] is True and good["release_line"].startswith("RELEASE CANDIDATE") and all(good["passes"].values())
    bad_cases = (
        (dict(f=_d(0.001, a=0.121, b=0.12)), "F"),
        (dict(arm={"A4_direction": _d(-0.02, -0.05, 0.01)}), "N"),         # a lower bound of exactly −0.05 fails (>)
        (dict(arm={"T9_comparison": None}), "N"),                          # a missing reading fails
        (dict(home={"1": {"band_balanced": _d(-0.01, -0.06, 0.03), "severe_error": _d(0.0, a=0.02, b=0.02)}}), "N"),
        (dict(home={"0": {"band_balanced": _d(0.0, -0.01, 0.01), "severe_error": _d(0.012, a=0.033, b=0.021)}}), "N"),
        (dict(r={"0": _r_ok(), "1": {**_r_ok(), "T4": {**_r_ok()["T4"], "noul.brier": _d(0.0, -0.01, 0.01)}}}), "R"),
        (dict(r={"0": _r_ok()}), "R"),                                     # both seeds are read
        (dict(g={"diff": -0.0201, "low": -0.04, "high": 0.0}), "G"),
        (dict(g={"diff": -0.01, "low": -0.045, "high": 0.02}), "G"),
    )  # fmt: skip
    for change, part in bad_cases:
        result = v.decide(_readings(**change))
        assert result["release_candidate"] is False and f"{part} fails" in result["release_line"] and result["passes"][part] is False, (part, change)
    safer = v.decide(_readings(SAFER_GAP))
    assert safer["release_candidate"] is False and "S SAFER-ONLY" in safer["release_line"]


def test_each_guard_at_its_bound():
    v = V()
    assert v.f_check(_d(0.0, a=0.12, b=0.12))["pass"] is True and v.f_check(None)["pass"] is False
    assert v.g_check({"diff": -0.02, "low": -0.0449})["pass"] is True
    assert v.g_check({"diff": -0.02, "low": -0.045})["pass"] is False and v.g_check({"diff": -0.021, "low": 0.0})["pass"] is False
    home = {s: {"band_balanced": _d(0.0, -0.0499, 0.0), "severe_error": _d(0.01, a=0.031, b=0.021)} for s in (0, 1)}
    arm = {k: _d(0.0, -0.0499, 0.0) for k in ("A4_direction", "A4_comparison", "T9_direction", "T9_comparison")}
    assert v.n_checks(arm, home)["pass"] is True, "lower bounds just above −0.05 and severe at mix + 0.01 pass"
    assert v.n_checks({**arm, "A4_comparison": _d(0.0, -0.05, 0.0)}, home)["checks"]["A4_comparison_low_gt_-0.05"] is False
    assert v.n_checks(arm, {0: home[0]})["pass"] is False, "both seeds' home readings are needed"


@pytest.mark.parametrize("s, consistent, outcome, needle", [
    (SUPPORT, CONSISTENT, "SUPPORTED", "scale authors and description sources"),
    (SAFER_GAP, CONSISTENT, "SAFER-ONLY", "more authors and System-0 margins"),
    (REJECT, CONSISTENT, "REJECTED", "System-0 margins as L2 inputs) or real data"),
    (SUPPORT, INCONSISTENT, "PARTIAL", "report seed disagreement"),
])  # fmt: skip
def test_the_pre_declared_next_step(s, consistent, outcome, needle):
    result = V().decide(_readings(s, consistent=consistent))
    assert result["S"]["outcome"] == outcome and result["next_step"]["branch"] == outcome and needle in result["next_step"]["next"]


# --------------------------------------------------------------------------
# synthetic A4-shaped records (v0.3b's record contract, an author's own scene names)
# --------------------------------------------------------------------------

#: (type, label, robust) → clear-safe, clear-risky, unclear (prior-dependent), unclear (robust middle).
SCALARS = {"safeq": ("noul", 0.1, True), "riskq": ("noul", 0.9, True), "unclq": ("score", 0.2, False), "midq": ("noul", 0.5, True)}


def _a4(groups: int = 24, scene: str = "dock_ramp") -> list[dict]:
    """A4-shaped one-fact pairs with a comparison question (plan key ``keep_going`` = plan.options' 'planned'), outcomes per variant whose
    planned 'tips' + 'slides' shares are the key question's label under every prior."""
    out = []
    for g in range(groups):
        for m in ("", "p"):
            rid, partner = f"v06a4-{scene}-{g:06d}{m}", f"v06a4-{scene}-{g:06d}{'' if m else 'p'}"
            key = 0.25 if (g + (m == "p")) % 2 else 0.75
            alt = 0.05 if g % 3 else 0.9
            outcomes = {"planned": {p: {"holds": 1 - key, "slides": key / 2, "tips": key / 2} for p in ("base", "alt_a", "alt_b")},
                        "opt_slow": {p: {"holds": 1 - alt, "slides": alt, "tips": 0.0} for p in ("base", "alt_a", "alt_b")}}
            questions = {n: {"type": t, "instructions": f"{n}?"} | ({"criteria": ["0", "1"]} if t == "score" else {}) for n, (t, _, _) in SCALARS.items()}
            questions["key"] = {"type": "noul", "instructions": "Does it go?"}
            questions["cmp"] = {"type": "choice", "instructions": "Which is riskier?", "criteria": {"keep_going": "the plan", "slow_down": "slower", "same": "same"}}
            labels = {n: ({"type": t, "p_yes": y} if t == "noul" else {"type": t, "value": y}) for n, (t, y, _) in SCALARS.items()}
            labels["key"] = {"type": "noul", "p_yes": key}
            more = alt > key
            labels["cmp"] = {"type": "choice", "kind": "state", "probabilities": {"keep_going": 0.0 if more else 0.9, "slow_down": 0.9 if more else 0.0, "same": 0.1}}
            robustness = {n: {"robust": r} for n, (_, _, r) in SCALARS.items()} | {"key": {"robust": True}, "cmp": {"robust": True}}
            out.append({"schema": "simgen-request/0", "id": rid, "request": {"state": f"radio {scene} {g}{m}", "questions": questions}, "labels": labels,
                        "meta": {"set": "v06-a4", "split": "test", "scene": scene, "key_question": "key",
                                 "question_kind": {"key": "key", "cmp": "compare"}, "compare_option": "slow_down",
                                 "plan": {"options": {"keep_going": "planned", "slow_down": "opt_slow"}}, "outcomes": outcomes,
                                 "labels_by_prior": {p: {"key": {"type": "noul", "p_yes": key}} for p in ("base", "alt_a", "alt_b")},
                                 "robustness": robustness, "pair": {"of": partner, "kind": "numeric", "role": "base" if not m else "variant"}}})
    return out


def _predict(records: list[dict], scalar, compare=None) -> list[dict]:
    out = []
    for record in records:
        questions = {}
        for name, question in record["request"]["questions"].items():
            label = record["labels"][name]
            if question["type"] == "choice":
                probs = dict(label["probabilities"]) if compare is None else compare(record, name, dict(label["probabilities"]))
                listings = [{"options": list(question["criteria"]), "values": None, "answer": {"probabilities": probs}}]
            else:
                values = scalar(record, name, label.get("p_yes", label.get("value")))
                values = values if isinstance(values, list) else [values]
                key = "p_yes" if question["type"] == "noul" else "value"
                listings = [{"options": None, "values": None, "answer": {key: float(p)}} for p in values]
            questions[name] = {"type": question["type"], "n_listings": len(listings), "answer": listings[0]["answer"], "listings": listings}
        out.append({"id": record["id"], "model": "synthetic", **CONDITIONS, "questions": questions})
    return out


def _group(record: dict) -> int:
    return int(record["id"].rstrip("p").split("-")[-1])


def HABIT(r, n, y):  # clear calls right, every unclear case "safe" (the dangerous habit)
    return {"safeq": 0.1, "riskq": 0.9, "unclq": 0.1, "midq": 0.1, "key": y}[n]


def WORSE(r, n, y):  # the same, and even the clear-risky question called "safe" in every other pair
    return 0.1 if n == "riskq" and _group(r) % 2 == 0 else HABIT(r, n, y)


def TARGETED(every):
    return lambda r, n, y: 0.5 if n in ("unclq", "midq") and _group(r) % every == 0 else HABIT(r, n, y)


def UNIFORM(every):
    return lambda r, n, y: 0.5 if n in ("safeq", "riskq", "unclq", "midq", "key") and _group(r) % every == 0 else HABIT(r, n, y)


def _calls(records, model):
    v = V()
    return v.V5().call_items("A4", records, _predict(records, model), clusters=v.V4().pair_clusters(records))


def _s_of(records, v06: tuple, v04: tuple) -> dict:
    """(S) on synthetic arms: v06 / v04 = (seed-0 model, seed-1 model)."""
    v = V()
    calls = {name: _calls(records, m) for name, m in (("v06s0", v06[0]), ("v06s1", v06[1]), ("v04s0", v04[0]), ("v04s1", v04[1]))}
    a, b = v.arm_calls(calls["v06s0"], calls["v06s1"]), v.arm_calls(calls["v04s0"], calls["v04s1"])
    diff = v.V5().call_difference(a, b, n_boot=400, seed=0)
    keys = sorted(set(a) & set(b))
    seed_dc = {s: v.V5().call_metrics(v.restrict(calls[f"v06s{s}"], keys))["DC"]["value"] for s in (0, 1)}
    return {"diff": diff, "consistency": v.consistency(seed_dc, diff["DC"]["b"]), "outcome": v.s_outcome(diff, v.consistency(seed_dc, diff["DC"]["b"]))["outcome"]}


def test_truth_classes_are_v05s():
    record = _a4(1)[0]
    truth = {n: V().V5().truth_class(record, n) for n in record["request"]["questions"]}
    assert truth == {"safeq": "clear_safe", "riskq": "clear_risky", "unclq": "unclear", "midq": "unclear", "key": "clear_risky", "cmp": None}
    flipped = copy.deepcopy(record)
    flipped["meta"]["robustness"]["unclq"]["robust"] = True   # robust and low → clear-safe
    del flipped["meta"]["robustness"]["safeq"]                # no entry → not robust → unclear
    assert (V().V5().truth_class(flipped, "unclq"), V().V5().truth_class(flipped, "safeq")) == ("clear_safe", "unclear")


def test_an_arms_call_distribution_is_the_mean_of_its_seeds_one_over_l_listings():
    v = V()
    records = _a4(2)
    one = v.V5().call_items("A4", records, _predict(records, lambda r, n, y: 0.1))                      # one listing: safe
    two = v.V5().call_items("A4", records, _predict(records, lambda r, n, y: [0.1, 0.5, 0.9, 0.9]))    # four listings: ¼ safe, ¼ look, ½ risky
    arm = v.arm_calls(one, two)
    key = ("A4", records[0]["id"], "safeq")
    assert arm[key][2] == pytest.approx(((1 + 0.25) / 2, 0.25 / 2, 0.5 / 2)) and sum(arm[key][2]) == pytest.approx(1.0)
    partial = {k: x for k, x in two.items() if k != key}
    assert key not in v.arm_calls(one, partial), "a question the other seed did not answer is not in the arm"
    other = copy.deepcopy(records)
    for r in other:
        r["meta"]["robustness"]["unclq"]["robust"] = True
    with pytest.raises(ValueError, match="different clusters or truth classes"):
        v.arm_calls(one, v.V5().call_items("A4", other, _predict(other, lambda r, n, y: 0.1)))
    assert v.arm_values({("k",): ("c", 1.0), ("j",): ("c", 0.0)}, {("k",): ("c", 0.0)}) == {("k",): ("c", 0.5)}


def test_every_arm_reading_is_the_mean_of_its_seeds_readings_on_the_same_items():
    v = V()
    records = _a4()
    rng = random.Random(5)
    s0 = _calls(records, lambda r, n, y: [rng.random() for _ in range(rng.choice((1, 2, 4)))])
    s1 = _calls(records, lambda r, n, y: [rng.random() for _ in range(rng.choice((1, 3)))])
    arm = v.V5().call_metrics(v.arm_calls(s0, s1))
    m0, m1 = v.V5().call_metrics(s0), v.V5().call_metrics(s1)
    for metric in ("DC", "LU", "LC", "GAP", "CA", "safe|unclear"):
        assert arm[metric]["value"] == pytest.approx((m0[metric]["value"] + m1[metric]["value"]) / 2), metric
    swapped = v.V5().call_difference(v.arm_calls(s0, s1), v.arm_calls(s1, s0), n_boot=200, seed=0)
    assert swapped["DC"]["value"] == pytest.approx(0.0, abs=1e-12) and swapped["DC"]["low"] == pytest.approx(0.0, abs=1e-12)


def test_the_synthetic_arms_read_as_designed():
    records = _a4()
    targeted = _s_of(records, (TARGETED(3), TARGETED(4)), (HABIT, HABIT))
    assert targeted["outcome"] == "SUPPORTED" and targeted["consistency"]["pass"] is True
    # 168 items count for DC (72 clear-risky, never called safe; 96 unclear, all safe under the habit); the seeds hedge 1/3 and 1/4 of them
    assert targeted["diff"]["DC"]["value"] == pytest.approx(-(96 / 168) * (1 / 3 + 1 / 4) / 2)
    assert targeted["diff"]["GAP"]["value"] == pytest.approx((1 / 3 + 1 / 4) / 2) and targeted["diff"]["CA"]["value"] == 0.0
    uniform = _s_of(records, (UNIFORM(3), UNIFORM(3)), (HABIT, HABIT))
    assert uniform["outcome"] == "SAFER-ONLY" and uniform["diff"]["GAP"]["value"] == pytest.approx(0.0, abs=1e-9)
    assert uniform["diff"]["DC"]["value"] == pytest.approx(-(96 / 168) / 3) and uniform["diff"]["CA"]["value"] == pytest.approx(-1 / 3)
    same = _s_of(records, (HABIT, HABIT), (HABIT, HABIT))
    assert same["outcome"] == "REJECTED" and same["diff"]["DC"]["value"] == 0.0
    split = _s_of(records, (TARGETED(1), WORSE), (HABIT, HABIT))   # seed 0 hedges every unclear case, seed 1 makes more dangerous calls
    assert split["diff"]["DC"]["value"] <= -0.05 and split["diff"]["DC"]["high"] < 0, "the pooled ΔDC condition holds"
    assert split["consistency"]["per_seed"] == {0: True, 1: False} and split["outcome"] == "PARTIAL"


# --------------------------------------------------------------------------
# A4's comparison question (an author's own scene)
# --------------------------------------------------------------------------


def test_the_comparison_mapping_reads_an_authors_scene_and_checks_itself_on_the_labels():
    v = V()
    records = _a4()
    mapping = v.comparison_map(records)["dock_ramp"]
    assert mapping["events"]["events"] == ["slides", "tips"] and "inferred" in mapping["events"]["method"] and mapping["events"]["candidates"] == 1
    assert mapping["plan"] == {**mapping["plan"], "plan_key": "keep_going", "method": "plan.options"} and mapping["label_agreement"] == 1.0
    no_planned = copy.deepcopy(records)  # the plan's key is not in plan.options: read from where the label mass sits when the plan is riskier
    for r in no_planned:
        r["meta"]["plan"]["options"] = {"go_on": "planned", "slow_down": "opt_slow"}
    plan = v.comparison_map(no_planned)["dock_ramp"]["plan"]
    assert plan["plan_key"] == "keep_going" and plan["method"] == "label mass where the plan is riskier"
    broken = copy.deepcopy(records)
    broken[0]["meta"]["labels_by_prior"]["base"]["key"]["p_yes"] = 0.33
    assert v.comparison_map(broken)["dock_ramp"]["events"]["events"] is None
    given = v.comparison_map(broken, {"dock_ramp": {"events": ["slides", "tips"], "plan_key": "keep_going"}})["dock_ramp"]
    assert given["events"]["events"] == ["slides", "tips"] and given["plan"]["plan_key"] == "keep_going" and "override" in given["plan"]["method"]
    assert given["label_agreement"] == 1.0, "an override is still checked against the comparison labels"


def test_the_comparison_items_score_the_ordering_of_the_two_actions():
    v = V()
    records = _a4()
    mapping = v.comparison_map(records)
    clusters = v.V4().pair_clusters(records)
    right = v.comparison_items("A4", records, _predict(records, HABIT), mapping, clusters=clusters)
    wrong = v.comparison_items("A4", records, _predict(records, HABIT, lambda r, n, p: {"keep_going": p["slow_down"], "slow_down": p["keep_going"], "same": p["same"]}),
                               mapping, clusters=clusters)
    tie = v.comparison_items("A4", records, _predict(records, HABIT, lambda r, n, p: {"keep_going": 0.4, "slow_down": 0.4, "same": 0.2}), mapping, clusters=clusters)
    assert right and set(right) == set(wrong) == set(tie)
    assert all(x == 1.0 for _, x in right.values()) and all(x == 0.0 for _, x in wrong.values()) and all(x == 0.0 for _, x in tie.values())
    assert all(c == clusters[k[1]] for k, (c, _) in right.items()), "clustered by pair group"
    near = copy.deepcopy(records)
    for r in near:  # shares 0.10 apart: under the 0.15 threshold, no item
        r["meta"]["outcomes"]["opt_slow"] = {p: {"holds": 0.0, "slides": min(1.0, r["labels"]["key"]["p_yes"] + 0.10), "tips": 0.0} for p in ("base", "alt_a", "alt_b")}
    assert v.comparison_items("A4", near, _predict(near, HABIT), v.comparison_map(near), clusters=clusters) == {}


def test_on_v03b_families_the_mapping_is_infamilys():
    v = V()
    record = _a4(1, scene="tilt")[0]
    mapping = v.comparison_map([record])["tilt"]
    assert mapping["events"]["events"] == ["slide", "tip"] and mapping["events"]["method"].startswith("infamily.EVENT")


# --------------------------------------------------------------------------
# G2 arm, author gap
# --------------------------------------------------------------------------


def test_the_g2_arm_reading_is_compare_probes_when_both_seeds_are_the_same_model():
    from robo_jev.decisions.records import read_records
    from robo_jev.decisions.single_probe import compare_probe

    v = V()
    g2, a, b = v.SET_PATHS["G2"], v.V04_PODS[0] / "after-v04-s0-2b-G2.predictions.jsonl", v.BASE_G2_POD
    if not (g2.is_file() and a.is_file() and b.is_file()):
        pytest.skip("G2 or its prediction files not on this box")
    records, pa, pb = read_records(g2), read_records(a), read_records(b)
    arm = v.g2_arm_comparison(records, [pa, pa], [pb, pb], n_boot=300, seed=0)
    single = compare_probe(records, pa, pb, dimensions=("domain",), cluster="record", n_boot=300, seed=0)["readings"]["single_order"]["table"]["_all"]["accuracy"]
    assert (arm["a"], arm["b"], arm["diff"], arm["low"], arm["high"]) == pytest.approx((single["a"], single["b"], single["diff"], single["low"], single["high"]))
    first = v.mean_probe_values({("r", "q"): {"accuracy": 1.0, "brier": 0.2, "code_mass": None, "chance": 0.5, "groups": {}, "clusters": {}}},
                                {("r", "q"): {"accuracy": 0.0, "brier": 0.4, "code_mass": 0.9, "chance": 0.5, "groups": {}, "clusters": {}}})
    assert first[("r", "q")]["accuracy"] == 0.5 and first[("r", "q")]["brier"] == pytest.approx(0.3) and first[("r", "q")]["code_mass"] is None


def test_the_author_gap_is_dc_a4_minus_dc_hset_per_arm_and_its_difference():
    v = V()
    a4, hset = _a4(), _a4(scene="hset_scene")
    worse = {"v06": _calls(a4, TARGETED(2)), "v04": _calls(a4, HABIT)}
    home = {"v06": _calls(hset, TARGETED(1)), "v04": _calls(hset, TARGETED(2))}
    gap = v.author_gap(worse, home, n_boot=300, seed=0)
    for arm in ("v06", "v04"):
        e = gap["per_model"][arm]
        assert e["gap"] == pytest.approx(v.V5().call_metrics(worse[arm])["DC"]["value"] - v.V5().call_metrics(home[arm])["DC"]["value"])
        assert e["low"] <= e["gap"] <= e["high"]
    d = gap["delta_gap_v06_minus_v04"]
    assert d["value"] == pytest.approx(gap["per_model"]["v06"]["gap"] - gap["per_model"]["v04"]["gap"]) and d["low"] <= d["value"] <= d["high"]


# --------------------------------------------------------------------------
# files
# --------------------------------------------------------------------------


def _args(tmp_path: Path, **changes) -> argparse.Namespace:
    v = V()
    base = dict(s0_dir=Path("/s0"), s1_dir=Path("/s1"), v04_s0_dir=v.V04_PODS[0], v04_s1_dir=v.V04_PODS[1], mix_dir=v.MIX_POD, spark_dir=tmp_path / "spark",
                v04_spark_dir=v.V04_SPARK_DIR, v05_spark_dir=v.V05_SPARK_DIR, base_g2=v.BASE_G2_POD, t9=v.SET_PATHS["T9"], a4=v.SET_PATHS["A4"])
    return argparse.Namespace(**{**base, **changes})


def test_prediction_names_and_default_paths(tmp_path):
    v = V()
    assert v.pod_prediction_name("v06", 1, "T3") == "after-v06-s1-2b-T3.predictions.jsonl" and v.spark_prediction_name("A4", "v04s1") == "A4-v04s1-2b.predictions.jsonl"
    paths = v.prediction_paths(_args(tmp_path))
    assert paths[("v06s0", "G2")] == Path("/s0/after-v06-s0-2b-G2.predictions.jsonl") and paths[("mix", "T7")] == v.MIX_POD / "after-v01mix-2b-T7.predictions.jsonl"
    assert paths[("v04s1", "T2")] == v.V04_PODS[1] / "after-v04-s1-2b-T2.predictions.jsonl"
    for model in ("v06s0", "v06s1", "v04s0", "v04s1", "base"):
        assert paths[(model, "A4")] == tmp_path / "spark" / f"A4-{v.SPARK_TAGS[model]}.predictions.jsonl", "A4 only from the v0.6 chain"
    assert paths[("v06s1", "T9")] == tmp_path / "spark" / "T9-v06s1-2b.predictions.jsonl", "v0.6 from its own Spark directory only"
    if (v.V04_SPARK_DIR / "T8b-v04s0-2b.predictions.jsonl").is_file():
        assert paths[("v04s0", "T8b")] == v.V04_SPARK_DIR / "T8b-v04s0-2b.predictions.jsonl"
    if (v.V05_SPARK_DIR / "Hset-v05s1-2b.predictions.jsonl").is_file():
        assert paths[("v05s1", "Hset")] == v.V05_SPARK_DIR / "Hset-v05s1-2b.predictions.jsonl"
    if v.EXISTING_SPARK[("base", "T9")].is_file():
        assert paths[("base", "T9")] == v.EXISTING_SPARK[("base", "T9")]
    need = set(v.required_inputs())
    assert {("v06s0", "A4"), ("v04s1", "A4"), ("base", "A4"), ("v06s1", "T9"), ("v04s0", "T9"), ("mix", "T7"), ("v06s0", "G2"), ("v04s1", "G2")} <= need
    assert not any(s in ("B1", "Hset", "T8b", "T6", "G1", "T8a") for _, s in need), "secondary inputs are not required"


def test_a_pipeline_check_reads_a4_and_t9_from_its_own_spark_directory_only(tmp_path):
    v = V()
    synthetic = tmp_path / "T9-synthetic.jsonl"
    synthetic.write_text("{}\n")
    paths = v.prediction_paths(_args(tmp_path, t9=synthetic))
    for model in ("v06s0", "v04s0", "base"):
        assert paths[(model, "T9")] == tmp_path / "spark" / f"T9-{v.SPARK_TAGS[model]}.predictions.jsonl", model
    assert v.is_sealed("T9", v.SET_PATHS["T9"]) and not v.is_sealed("T9", synthetic) and not v.is_sealed("A4", synthetic)


def test_the_sealed_a4_must_be_the_held_out_authors_test_records_and_outputs_stay_out_of_sealed_places(tmp_path):
    v = V()
    assert v.a4_problems(_a4(2)) == []
    wrong = copy.deepcopy(_a4(2))
    wrong[0]["meta"]["split"] = "train"
    wrong[1]["id"] = "v06a1-x-000001"
    assert "2 records" in v.a4_problems(wrong)[0]
    for place in (REPO / "artifacts" / "reports" / "v05" / "verdict.json", REPO / "artifacts" / "t9" / "x.json", REPO / "artifacts" / "v06" / "a4" / "x.json",
                  REPO / "artifacts" / "simgen" / "v06" / "x.json"):
        with pytest.raises(SystemExit, match="refusing"):
            v.refuse_write(place)
    v.refuse_write(REPO / "artifacts" / "reports" / "v06" / "verdict.json")
    v.refuse_write(tmp_path / "verdict.json")


def test_the_spark_chain_writes_the_names_the_helper_reads_when_present():
    import re

    script = REPO / "artifacts" / "scratch" / "v06b" / "score_spark.sh"
    if not script.is_file():
        pytest.skip(f"{script} not present (scratch, this box only)")
    text = script.read_text(encoding="utf-8")
    runs = re.findall(r"^run (\S+) \S+ (\S+) ", text, flags=re.M)
    want = [("A4", f"{m}-2b") for m in ("v06s0", "v06s1", "v04s0", "v04s1", "base")] + [(s, f"{m}-2b") for s in ("T9", "T8b", "Hset", "B1") for m in ("v06s0", "v06s1")]
    assert sorted(runs) == sorted(want), "A4 for v0.6, v0.4 and the base; T9, T8b, H-set and B1 for v0.6"
    for set_name, tag in runs:
        assert V().spark_prediction_name(set_name, tag.removesuffix("-2b")) == f"{set_name}-{tag}.predictions.jsonl"
    assert 'V06_GATE:-}" != "open"' in text and "OUT=artifacts/reports/v06/spark" in text and "v06-s0-2b/checkpoint.pt" in text and "v06-s1-2b/checkpoint.pt" in text
    assert "artifacts/v06/a4/test.jsonl" in text and "v04-s0-2b/checkpoint.pt" in text and "v04-s1-2b/checkpoint.pt" in text


def test_the_rule_text_is_the_pre_declared_one():
    candidates = [base / ".superpowers" / "sdd" / "progress.md" for base in (REPO, REPO.parents[2])]  # the checkout, or the main checkout of a worktree
    progress = next((p for p in candidates if p.is_file()), None)
    if progress is None:
        pytest.skip("progress.md not on this box (.superpowers is not in git)")
    text = progress.read_text(encoding="utf-8")
    entry = next(line for line in text.splitlines() if line.startswith("PRE-DECLARED v0.6 VERDICT RULE"))
    for fragment in ("SUPPORTED iff ΔDC ≤ −0.05 with upper bound < 0 AND ΔGAP ≥ +0.05 with lower bound > 0 AND ΔCA ≥ −0.10 (point) AND consistency",
                     "REJECTED iff the ΔDC lower bound > −0.02", "(G) G2 v0.6 − v0.4 (arm means) ≥ −0.02 point and lower bound > −0.045",
                     "(F) A4 robust high-called-low ≤ untrained base (point, v0.6 arm mean)"):
        assert fragment in entry, fragment
    assert (V().S_DC_POINT, V().S_DC_HIGH, V().S_GAP_POINT, V().S_GAP_LOW, V().S_CA_POINT, V().REJECT_DC_LOW) == (-0.05, 0.0, 0.05, 0.0, -0.10, -0.02)
    assert (V().N_LOW, V().N_SEVERE_SLACK, V().G_POINT, V().G_LOW) == (-0.05, 0.01, -0.02, -0.045)
    assert json.dumps(V().RULE)  # serialisable into the verdict
