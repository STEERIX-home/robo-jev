"""Task v0.9b — the v0.9 verdict helper (scripts/verdict_v09.py): the PRE-DECLARED v0.9 VERDICT RULE at every bound, each outcome and next
step, the five-source draw-wise bootstrap against an independent recomputation, and synthetic predictions through the whole computation (no
v0.9 prediction exists; nothing here reads one)."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]


def _load(name: str, file: str):
    spec = importlib.util.spec_from_file_location(name, REPO / "scripts" / file)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


V = _load("verdict_v09_under_test", "verdict_v09.py")


# --------------------------------------------------------------------------
# The rule at its bounds (pure)
# --------------------------------------------------------------------------


def _p1(delta=0.25, low=1e-9, band=0.60):
    return {"delta": {"value": delta, "low": low}, "band_CN": {"value": band}}


def test_pnum1_bounds():
    assert V.pnum1_check(_p1())["pass"]
    assert not V.pnum1_check(_p1(delta=0.2499))["pass"]
    assert not V.pnum1_check(_p1(low=0.0))["pass"], "the lower bound must be strictly above 0"
    assert not V.pnum1_check(_p1(band=0.5999))["pass"]
    assert V.pnum1_check(_p1(delta=0.9, low=0.5, band=1.0))["pass"]
    assert V.pnum1_check(_p1(delta=0.70 - 0.45))["pass"], "float rounding of an exact tie never flips an inclusive bound"
    assert not V.pnum1_check({"delta": {"value": 0.5, "low": 0.1}})["pass"], "a missing band reading fails"
    assert not V.pnum1_check(None)["pass"]


def test_pnum2_bounds():
    assert V.pnum2_check({"value": 0.20, "low": 1e-9})["pass"]
    assert not V.pnum2_check({"value": 0.1999, "low": 0.1})["pass"]
    assert not V.pnum2_check({"value": 0.5, "low": 0.0})["pass"]
    assert not V.pnum2_check({"value": 0.5})["pass"] and not V.pnum2_check(None)["pass"]
    assert V.pnum2_check({"value": 0.65 - 0.45, "low": 0.01})["pass"]


def test_hnum_and_hcons_outcomes():
    assert V.hnum_outcome(True, True) == "SUPPORTED"
    assert V.hnum_outcome(True, False) == V.hnum_outcome(False, True) == "PARTIAL"
    assert V.hnum_outcome(False, False) == "REJECTED"
    assert V.hcons_outcome(True, True) == "SUPPORTED"
    assert V.hcons_outcome(True, False) == V.hcons_outcome(False, True) == V.hcons_outcome(False, False) == "NOT SUPPORTED"


def test_pcons_and_accuracy_hold_bounds():
    assert V.pcons_check({"value": -0.10, "high": -1e-9})["pass"]
    assert not V.pcons_check({"value": -0.0999, "high": -0.05})["pass"]
    assert not V.pcons_check({"value": -0.3, "high": 0.0})["pass"], "the upper bound must be strictly below 0"
    assert not V.pcons_check({"value": -0.3})["pass"] and not V.pcons_check(None)["pass"]
    assert V.pcons_check({"value": 0.25 - 0.35, "high": -0.01})["pass"]
    assert V.accuracy_holds({"value": -0.02}, {"value": -0.02})["pass"]
    assert not V.accuracy_holds({"value": -0.0201}, {"value": 0.0})["pass"]
    assert not V.accuracy_holds({"value": 0.0}, {"value": -0.0201})["pass"]
    assert not V.accuracy_holds({"value": 0.0}, None)["pass"]
    assert V.accuracy_holds({"value": 0.70 - 0.72}, {"value": 0.0})["pass"], "−0.020000000000000018 is the tie −0.02"


GOOD_GUARDS = {"DC": {"value": 0.02}, "CA": {"value": -0.05}, "band": {"value": 0.0, "low": -0.0499}, "severe": {"a": 0.11, "b": 0.10},
               "G2": {"diff": -0.02, "low": -0.0449}}


def test_guards_at_their_bounds():
    assert V.guard_checks(GOOD_GUARDS)["pass"]
    for key, bad, check in (("DC", {"value": 0.0201}, "outside_DC_mean_le_+0.02"), ("CA", {"value": -0.0501}, "outside_CA_mean_ge_-0.05"),
                            ("band", {"low": -0.05}, "T2-T4_band_low_gt_-0.05"), ("severe", {"a": 0.1101, "b": 0.10}, "T2-T4_severe_a_le_b_plus_0.01"),
                            ("G2", {"diff": -0.0201, "low": 0.0}, "G2_point_ge_-0.02"), ("G2", {"diff": 0.0, "low": -0.045}, "G2_low_gt_-0.045")):
        result = V.guard_checks({**GOOD_GUARDS, key: bad})
        assert not result["pass"] and not result["checks"][check], key
    for key in GOOD_GUARDS:
        assert not V.guard_checks({k: v for k, v in GOOD_GUARDS.items() if k != key})["pass"], key


def test_next_steps_each_branch():
    s = V.next_steps("REJECTED", False, False, False)
    assert s["branches"] == ["H-num REJECTED", "H-cons NOT SUPPORTED"] and "4B backbone" in s["next"][0] and "test-time order averaging" in s["next"][1]
    s = V.next_steps("PARTIAL", False, True, True)
    assert s["branches"] == ["H-num PARTIAL with N9g passing and A16s failing"] and "paraphrased arm-style rules" in s["next"][0]
    assert any("no next step is pre-declared" in n for n in s["notes"])
    s = V.next_steps("PARTIAL", True, False, True)
    assert s["branches"] == ["H-num PARTIAL with A16s passing and N9g failing"] and "broader families" in s["next"][0]
    s = V.next_steps("SUPPORTED", True, True, False, {"CN": 0.8, "v07": 0.43, "v07_stored": 0.4296})
    assert s["branches"] == ["H-num SUPPORTED", "H-cons NOT SUPPORTED"] and "release candidate" in s["next"][0]
    assert "CN 0.800 vs v0.7 0.430 (CN − v0.7 +0.370)" in s["notes"][0] and "0.430" in s["notes"][0]
    s = V.next_steps("SUPPORTED", True, True, True, None)
    assert "unread" in s["notes"][0]


def test_decide_with_missing_readings_fails_and_guards_never_change_the_outcome():
    v = V.decide({})
    assert v["H-num"] == "REJECTED" and v["H-cons"] == "NOT SUPPORTED" and not v["guards"]["CN-C"]["pass"] and not v["guards"]["C-F"]["pass"]
    assert v["release_line"] == "RELEASE: the user decides."
    readings = {"P-num1": _p1(0.5, 0.2, 0.8), "P-num2": {"value": 0.3, "low": 0.1}, "P-cons": {"value": -0.2, "high": -0.1},
                "M8g accuracy": {"value": 0.0}, "M8g accuracy N<=26": {"value": 0.0}, "guards": {"CN-C": {**GOOD_GUARDS, "DC": {"value": 0.5}}}}
    v = V.decide(readings)
    assert v["H-num"] == "SUPPORTED" and v["H-cons"] == "SUPPORTED" and not v["guards"]["CN-C"]["pass"]
    assert v["next_steps"]["branches"] == ["H-num SUPPORTED"]


def test_the_rule_bounds_are_the_pre_declared_ones():
    text = REPO / ".superpowers" / "sdd" / "progress.md"
    if not text.is_file():  # the briefs live in the main checkout, not in a worktree
        text = REPO.parents[2] / ".superpowers" / "sdd" / "progress.md"
    if text.is_file():
        rule = text.read_text(encoding="utf-8").split("PRE-DECLARED v0.9 VERDICT RULE")[-1]
        for needle in ("A16s accuracy (single order seed 1, serve's conditions, v0.8's path) ≥ +0.25 with lower bound > 0",
                       "CN's accuracy in the 1–8 cm band (both signs pooled) ≥ 0.60 (point)", "P-num2: N9g accuracy ≥ +0.20 with lower bound > 0",
                       "SUPPORTED iff P-num1 and P-num2; PARTIAL iff exactly one; REJECTED iff neither",
                       "as the mean of the four M8g bins and B8 (five sources, equal weight), ≤ −0.10 with upper bound < 0",
                       "AND M8g accuracy ≥ −0.02 (point), AND M8g accuracy at N ≤ 26 (N 6 and 18) ≥ −0.02 (point)",
                       "outside DC (A4 / T9 / T8b-B2 mean) ≤ +0.02 (point)", "outside CA (mean) ≥ −0.05 (point)",
                       "T2–T4 balanced band lower bound > −0.05, and severe errors ≤ +0.01 (arm means)", "G2 ≥ −0.02 (point) with lower bound > −0.045",
                       "a 4B backbone trial on the same data, or System 0 computing such steps", "paraphrased arm-style rules", "Next: broader families",
                       "the label prior through data balancing, not loss weights", "a larger share or λ, or test-time order averaging for bench questions"):
            assert needle in rule, needle
            assert needle in " ".join(V.RULE.values()) or needle.replace("  ", " ") in " ".join(V.RULE.values()), needle
    assert (V.PNUM1_DELTA, V.PNUM1_LOW, V.PNUM1_BAND, V.PNUM2_DELTA, V.PNUM2_LOW, V.PCONS_DELTA, V.PCONS_HIGH, V.ACC_HOLD, V.DC_POINT, V.CA_POINT,
            V.BAND_LOW, V.SEVERE_SLACK, V.G_POINT, V.G_LOW) == (0.25, 0.0, 0.60, 0.20, 0.0, -0.10, 0.0, -0.02, 0.02, -0.05, -0.05, 0.01, -0.02, -0.045)
    assert V.CONS_SOURCES == ("M8g 2-10", "M8g 11-26", "M8g 27-60", "M8g 61-255", "B8") and V.A16S_BAND_1_8 == ("1-3", "3-8")


def test_the_private_v08_instance_reads_v09_arms_and_leaves_the_real_module_alone():
    real = _load("verdict_v08_real_for_v09_test", "verdict_v08.py")
    private = V.V8()
    assert private.ARMS == V.ARMS and private.MODELS == V.MODELS and private.SPARK_TAGS["v09CNs1"] == "v09CNs1-2b"
    assert real.ARMS == {"F": ("v08Fs0", "v08Fs1"), "FP": ("v08FPs0", "v08FPs1"), "v07": ("v07s0", "v07s1")}
    assert real.MODELS == ("v08Fs0", "v08Fs1", "v08FPs0", "v08FPs1", "v07s0", "v07s1", "base") and real is not private
    assert private.__name__ == "v09_verdict_v08"


# --------------------------------------------------------------------------
# The five-source draw-wise bootstrap
# --------------------------------------------------------------------------


def _naive_five(sources, n_boot, seed):
    """An independent recomputation: per group one draw of its clusters (seed + the group's index), per source the mean over the drawn
    clusters' units (repeats counted), the five differences averaged per draw."""
    from robo_jev.bench.system1 import _draws, _interval

    groups = []
    for s in sources:
        if s["group"] not in groups:
            groups.append(s["group"])
    clusters = {g: sorted({s["cluster"][k] for s in sources if s["group"] == g for k in set(s["a"]) & set(s["b"])}) for g in groups}
    draws = {g: _draws(len(clusters[g]), n_boot, seed + i) for i, g in enumerate(groups)}
    means = []
    for row in range(n_boot):
        deltas = []
        for s in sources:
            keys = set(s["a"]) & set(s["b"])
            picked = [clusters[s["group"]][j] for j in draws[s["group"]][row]]
            ta = tb = n = 0.0
            for c in picked:
                for k in keys:
                    if s["cluster"][k] == c:
                        ta += s["a"][k]
                        tb += s["b"][k]
                        n += 1
            deltas.append((ta - tb) / n if n else float("nan"))
        means.append(float(np.mean(deltas)))
    return _interval(np.asarray(means))


def test_five_source_mean_matches_an_independent_recomputation():
    rng = np.random.default_rng(7)
    items = [f"item{i}" for i in range(9)]
    sources = []
    for b, name in enumerate(("2-10", "11-26", "27-60", "61-255")):
        units = [f"{it}-n{b}-{j}" for it in items for j in range(1 if b < 3 else 2)]
        a = {u: float(rng.random() < 0.3) for u in units}
        bb = {u: float(rng.random() < 0.5) for u in units}
        del bb[units[0]]  # one unit only one arm has: left out
        sources.append({"name": f"M8g {name}", "group": "M8g", "a": a, "b": bb, "cluster": {u: u.split("-")[0] for u in units}})
    units = [(f"env{k % 3}", f"req{k // 2}", "q") for k in range(14)]
    sources.append({"name": "B8", "group": "B8", "a": {u: float(rng.random() < 0.2) for u in units}, "b": {u: float(rng.random() < 0.4) for u in units},
                    "cluster": {u: u[1] for u in units}})
    out = V.five_source_mean(sources, n_boot=300, seed=0)
    low, high = _naive_five(sources, 300, 0)
    assert out["low"] == pytest.approx(low, abs=1e-12) and out["high"] == pytest.approx(high, abs=1e-12)
    points = [np.mean([s["a"][k] for k in set(s["a"]) & set(s["b"])]) - np.mean([s["b"][k] for k in set(s["a"]) & set(s["b"])]) for s in sources]
    assert out["value"] == pytest.approx(float(np.mean(points)))
    assert out["group_seeds"] == {"M8g": 0, "B8": 1} and out["per_source"]["M8g 2-10"]["n"] == 8
    assert out["per_source"]["M8g 61-255"]["clusters"] == 9 and out["per_source"]["B8"]["clusters"] == 7
    missing = V.five_source_mean([*sources[:4], {**sources[4], "a": {}}], n_boot=50, seed=0)
    assert missing["value"] is None and not V.pcons_check(missing)["pass"]


# --------------------------------------------------------------------------
# Synthetic data
# --------------------------------------------------------------------------

AXIS_KEYS = ["plus_far", "plus_mid", "plus_near", "plus_small", "plus_fine", "stay", "minus_far", "minus_mid", "minus_near", "minus_small", "minus_fine"]
A16S_ARM = ("The gripper still has to travel {d} cm along {a}{up}. Choose this update's step along {a}: plus for a positive distance, minus for a "
            "negative one. Take the largest step that does not overshoot: far (20) when the distance is at least 20 cm, mid (8) when at least 8, "
            "near (3) when at least 3, small (1) when at least 1, fine (0.4) when between 0.2 and 1 cm, and stay when it is within 0.2 cm.")


def a16s_records() -> list[dict]:
    from robo_jev.bench import a16s_v08 as S

    templates = []
    for i in range(2):
        questions = {"intent": {"type": "choice", "instructions": "Name the stage. Facts: remaining travel x +0.9, y -35.5, z +82 cm.",
                                "criteria": {"look": "Look.", "grasp": "Grasp."}}}
        questions.update({a: {"type": "choice", "instructions": A16S_ARM.format(d=d, a=a, up=" (up is plus)" if a == "y" else ""),
                              "criteria": {k: k for k in AXIS_KEYS}} for a, d in (("x", "+0.9"), ("y", "-35.5"), ("z", "+82"))})
        templates.append({"id": f"v08a16-{i:04d}", "meta": {"set": "v08-a16"}, "request": {"state": "servo", "questions": questions}})
    return S.build_records(templates, seed=3)


def a16s_answers(records: list[dict], pick) -> dict:
    out: dict = {}
    for r in records:
        for axis in "xyz":
            sweep = r["meta"]["sweep"][axis]
            for reading in V.A16S_READINGS:
                choice = pick(sweep["label"], sweep["distance"], reading)
                out.setdefault(r["id"], {}).setdefault(axis, {})[reading] = {"choice": choice, "probabilities": {k: (0.55 if k == choice else 0.045) for k in AXIS_KEYS}}
    return out


def right(label, d, reading):
    return label


def far_only(label, d, reading):
    """v0.7's pattern: right at >= 25 cm, otherwise stay."""
    return label if abs(d) >= 25 else "stay"


def n9g_records(count: int = 40) -> list[dict]:
    out = []
    for i in range(count):
        n = (3, 5, 7, 11, 20)[i % 5]
        keys = [f"k{j}" for j in range(n)]
        correct = keys[i % n]
        out.append({"id": f"v09n9g-{i:04d}", "meta": {"set": "v09-n9g", "family": ("threshold", "interval", "selection", "comparison", "conversion")[i % 5],
                                                     "template": f"t{i % 7}", "domain": ("pressure", "distance", "voltage", "time")[i % 4],
                                                     "domain_seen": (i % 4) in (1, 3), "format": ("plain", "signed")[i % 2], "wording": ("plain", "distractor")[i % 2],
                                                     "near": ("rel", "abs", None)[i % 3], "n_options": n, "correct": correct, "target": ["q"]},
                    "request": {"state": f"reading {i}", "questions": {"q": {"type": "choice", "instructions": "Pick.", "criteria": {k: k for k in keys}}}},
                    "labels": {"q": {"type": "choice", "kind": "probe", "probabilities": {k: float(k == correct) for k in keys}}}})
    return out


def n9g_answers(records: list[dict], right_share: float, *, flip: bool = False) -> dict:
    """Right on the first `right_share` of the records; single-s2 tops another option when `flip`."""
    out = {}
    cut = int(round(right_share * len(records)))
    for i, r in enumerate(records):
        keys = list(r["request"]["questions"]["q"]["criteria"])
        correct = r["meta"]["correct"]
        top = correct if i < cut else next(k for k in keys if k != correct)
        other = next(k for k in keys if k != top)
        out[r["id"]] = {"q": {"single-s1": {"choice": top, "probabilities": {k: (0.6 if k == top else 0.4 / (len(keys) - 1)) for k in keys}},
                              "single-s2": {"choice": other if flip else top,
                                            "probabilities": {k: (0.6 if k == (other if flip else top) else 0.4 / (len(keys) - 1)) for k in keys}}}}
    return out


def call_set(name: str, pairs: int = 20) -> list[dict]:
    out = []
    for i in range(pairs):
        a, b = f"{name}-{i:03d}a", f"{name}-{i:03d}b"
        for rid, partner, y, robust in ((a, b, 0.9, True), (b, a, 0.5, False)):
            out.append({"id": rid, "meta": {"set": name, "pair": {"of": partner}, "robustness": {"q": {"robust": robust}}},
                        "request": {"state": "s", "questions": {"q": {"type": "noul", "instructions": "x"}}}, "labels": {"q": {"type": "noul", "p_yes": y}}})
        if i % 3 == 0:
            out.append({"id": f"{name}-{i:03d}s", "meta": {"set": name, "robustness": {"q": {"robust": True}}},
                        "request": {"state": "s", "questions": {"q": {"type": "noul", "instructions": "x"}}}, "labels": {"q": {"type": "noul", "p_yes": 0.1}}})
    return out


def call_preds(records: list[dict], safe_share: float) -> list[dict]:
    risky_or_unclear = [r for r in records if r["labels"]["q"]["p_yes"] >= 0.5]
    safe_ids = {r["id"] for r in risky_or_unclear[: int(round(safe_share * len(risky_or_unclear)))]}
    return [{"id": r["id"], "questions": {"q": {"type": "noul", "listings": [{"answer": {"p_yes": 0.2 if r["labels"]["q"]["p_yes"] < 0.5 or r["id"] in safe_ids else 0.8}}]}}}
            for r in records]


def home_sets() -> dict[str, list[dict]]:
    sets = {h: [{"id": f"{h}-{i}", "meta": {"set": h}, "request": {"state": "s", "questions": {"q": {"type": "noul", "instructions": "x"}}},
                 "labels": {"q": {"type": "noul", "p_yes": (0.1, 0.5, 0.9)[i % 3]}}} for i in range(12)] for h in ("T2", "T3", "T4")}
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


ENV_Q = {"plates": [("move", "above_in1", ["above_in1", "lower", "wait", "look"])], "cafe": [("intent", "serve_9", ["serve_9", "take_up_tray", "wait"])],
         "delivery": [("manoeuvre", "follow", ["follow", "pass_left", "stop", "call_operator"])], "tubes": [("work", None, [f"prep_t{i}" for i in range(1, 30)])]}


def b8_data(requests_per_env: int = 6) -> tuple[dict, dict]:
    tags, records = {"kind": "v08-b8-tags", "requests": []}, {}
    for env, questions in ENV_Q.items():
        for i in range(requests_per_env if env != "tubes" else 2):
            rid = f"v08b8-{env}-{i:03d}"
            entry, qs = {"id": rid, "env": env, "questions": {}}, {}
            for name, planner, keys in questions:
                entry["questions"][name] = {"n_options": len(keys), "planner": planner,
                                            "options": [{"key": k, "text": k, "tag": "PASSIVE" if k in ("wait", "look", "stop") else "ACTIVE", "rule": "t"} for k in keys]}
                qs[name] = {"type": "choice", "instructions": "x", "criteria": {k: k for k in keys}}
            tags["requests"].append(entry)
            records[rid] = {"id": rid, "meta": {"set": "v08-b8", "env": env}, "request": {"state": f"s {rid}", "questions": qs}}
    return tags, records


def b8_scores(tags: dict, *, flip: bool, bundle: str = "bundle-1") -> dict:
    answers = {}
    for entry in tags["requests"]:
        for name, q in entry["questions"].items():
            keys = [o["key"] for o in q["options"]]
            top, other = keys[0], keys[1]
            probs = {k: (0.6 if k == top else 0.4 / (len(keys) - 1)) for k in keys}
            probs2 = {k: (0.6 if k == other else 0.4 / (len(keys) - 1)) for k in keys} if flip else probs
            codes = {k: chr(65 + i % 26) for i, k in enumerate(keys)}
            answers.setdefault(entry["id"], {})[name] = {"balanced-s1": {"choice": top, "probabilities": probs},
                                                         "single-s1": {"choice": top, "probabilities": probs, "codes": codes},
                                                         "single-s2": {"choice": other if flip else top, "probabilities": probs2, "codes": codes}}
    return {"kind": "v08-bench-scores", "set": "B8", "bundle_sha256": bundle, "answers": answers}


LEVELS = (6, 18, 40, 120, 255)


def m8g_data(items: int = 8) -> list[dict]:
    from robo_jev.bench import gates_v08 as G8

    out = []
    for rank in range(items):
        for n in LEVELS:
            for listing in (0, 1):
                position = (rank % ((n + 1) // 2)) if n <= 26 and listing == 0 else ((n + 1) // 2 + rank % (n - (n + 1) // 2)) if n <= 26 else \
                    (rank % 26 if listing == 0 else 26 + rank % (n - 26))
                others = [f"o{j}" for j in range(2, n + 1)]
                keys = others[:position] + ["o1"] + others[position:]
                out.append({"id": f"v08m8g-{rank:05d}-n{n:03d}-{listing}",
                            "meta": {"set": "v08-m8g", "item": f"item{rank}", "source": "PHYS" if rank % 4 == 3 else "MMLU", "n": n, "position": position,
                                     "two_letter": position >= 26, "listing": listing, "correct": "o1", "rank": rank, "bin": G8.bin_of(n)},
                            "request": {"state": f"question {rank}", "questions": {"q": {"type": "choice", "instructions": "x", "criteria": {k: k for k in keys}}}},
                            "labels": {"q": {"type": "choice", "kind": "probe", "probabilities": {k: float(k == "o1") for k in keys}}}})
    return out


def m8g_preds(records: list[dict], *, kind: str = "order", flip: bool = False, wrong_small: bool = False) -> list[dict]:
    """Right on every listing (on N ≤ 26 wrong when `wrong_small`); the seed-2 file lists the options reversed (and tops another option when
    `flip`)."""
    out = []
    for r in records:
        keys = list(r["request"]["questions"]["q"]["criteria"])
        top = "o1" if not (wrong_small and r["meta"]["n"] <= 26) else next(k for k in keys if k != "o1")
        if kind == "seed2" and flip:
            top = next(k for k in keys if k != top)
        probs = {k: (0.7 if k == top else 0.3 / (len(keys) - 1)) for k in keys}
        shown = list(reversed(keys)) if kind == "seed2" else keys
        out.append({"id": r["id"], "questions": {"q": {"type": "choice", "listings": [{"options": shown, "code_probs": [probs[k] for k in shown]}]}}})
    return out


def scenario(*, num1: bool, num2: bool, cons: bool = True, small_hold: bool = True, n_boot: int = 200, a16: dict | None = None) -> dict:
    """Synthetic inputs through :func:`compute`: CN right everywhere on A16s / N9g when the respective P-num passes (else CN = C); C flips
    nowhere when P-cons passes (else like F, which flips on every M8g listing-0 record and every B8 question); C wrong on M8g N ≤ 26 when not
    `small_hold`."""
    sets = {s: call_set(s) for s in V.OUTSIDE_SETS}
    std_preds = {m: {s: call_preds(sets[s], 0.6) for s in sets} for m in V.POD_MODELS}
    pod_sets = home_sets()
    pod_preds = {m: home_preds(pod_sets) for m in V.POD_MODELS}
    a16s = a16s_records()
    a16s_scores = {m: {"answers": a16s_answers(a16s, right if (num1 and m.startswith("v09CN")) else far_only)} for m in (*V.V09_MODELS, *V.F_MODELS, "base")}
    n9 = n9g_records()
    n9_scores = {m: {"answers": n9g_answers(n9, 1.0 if (num2 and m.startswith("v09CN")) else 0.4)} for m in (*V.V09_MODELS, *V.F_MODELS, "base")}
    m8 = m8g_data()
    listing0 = [r for r in m8 if r["meta"]["listing"] == 0]
    m8_preds = {}
    for m in (*V.V09_MODELS, *V.F_MODELS):
        flips = m.startswith("v08F") or (m.startswith("v09C") and not m.startswith("v09CN") and not cons)
        m8_preds[m] = {"order": m8g_preds(m8, wrong_small=(m.startswith("v09Cs") and not small_hold)), "seed2": m8g_preds(listing0, kind="seed2", flip=flips)}
    tags, records = b8_data()
    b8s = {m: b8_scores(tags, flip=m.startswith("v08F") or (m.startswith("v09Cs") and not cons)) for m in (*V.V09_MODELS, *V.F_MODELS)}
    return V.compute(standard_sets=sets, standard_preds=std_preds, pod_sets=pod_sets, pod_preds=pod_preds, a16s={"records": a16s, "scores": a16s_scores},
                     n9g={"records": n9, "scores": n9_scores}, m8g={"records": m8, "preds": m8_preds}, b8={"tags": tags, "records": records, "scores": b8s},
                     a16=a16, n_boot=n_boot, seed=0)


# --------------------------------------------------------------------------
# Decision readings on synthetic data
# --------------------------------------------------------------------------


def test_pnum1_reading_pairs_axis_questions_and_clusters_by_request():
    records = a16s_records()
    scores = {m: {"answers": a16s_answers(records, right if m.startswith("v09CN") else far_only)} for m in V.V09_MODELS}
    out = V.pnum1_reading(records, scores, n_boot=200)
    d = out["delta"]
    # far_only is right at >= 25 cm (36 questions) and on the stay values 0.1 and 0.15 cm (24 questions)
    assert (d["n"], d["clusters"]) == (324, 108) and d["a"] == 1.0 and d["b"] == pytest.approx(60 / 324)
    assert d["value"] == pytest.approx(1 - 60 / 324) and d["low"] > 0
    assert out["band_CN"]["value"] == 1.0 and out["band_CN"]["n"] == 132 and out["band_1_8"]["C"]["value"] == 0.0
    assert V.pnum1_check({"delta": d, "band_CN": out["band_CN"]})["pass"]
    half = {**scores, "v09CNs1": scores["v09Cs0"]}  # one CN seed like C: the arm is the seeds' mean
    out = V.pnum1_reading(records, half, n_boot=0)
    assert out["delta"]["a"] == pytest.approx((1 + 60 / 324) / 2) and out["band_CN"]["value"] == pytest.approx(0.5)
    assert V.pnum1_reading(records, {k: v for k, v in scores.items() if k != "v09Cs1"}, n_boot=0)["missing"] == ["C"]


def test_pnum2_reading_and_its_cluster():
    records = n9g_records(20)
    scores = {m: {"answers": n9g_answers(records, 0.9 if m.startswith("v09CN") else 0.5)} for m in V.V09_MODELS}
    out = V.pnum2_reading(records, scores, n_boot=200)
    assert out["value"] == pytest.approx(0.4) and out["n"] == 20 and out["clusters"] == 20 and out["missing"] == []
    assert V.pnum2_reading(records, {"v09Cs0": scores["v09Cs0"]}, n_boot=0)["missing"] == ["CN", "C"]


def test_pcons_reading_on_m8g_bins_and_b8():
    m8 = m8g_data(items=6)
    listing0 = [r for r in m8 if r["meta"]["listing"] == 0]
    preds = {m: {"order": m8g_preds(m8), "seed2": m8g_preds(listing0, kind="seed2", flip=m.startswith("v08F"))} for m in (*V.ARMS["C"], *V.ARMS["F"])}
    tags, records = b8_data(requests_per_env=3)
    units = V.V8().b8_units(tags, records)
    scores = {m: b8_scores(tags, flip=m.startswith("v08F")) for m in preds}
    out = V.pcons_reading(m8, preds, units, scores, n_boot=200)
    assert out["value"] == pytest.approx(-1.0) and out["a"] == 0.0 and out["b"] == 1.0 and out["high"] == pytest.approx(-1.0)
    assert out["per_source"]["M8g 61-255"]["n"] == 6 * 2 and out["per_source"]["M8g 2-10"]["n"] == 6
    assert out["per_source"]["B8"]["n"] == 3 * 3 + 2 and out["per_source"]["B8"]["clusters"] == 3 * 3 + 2
    assert V.pcons_check(out)["pass"]
    half = {**preds, "v09Cs1": preds["v08Fs0"]}  # one C seed flips everywhere: C's arm flips at 0.5
    out = V.pcons_reading(m8, half, units, scores, n_boot=0)
    assert out["per_source"]["M8g 27-60"]["a"] == 0.5 and out["per_source"]["B8"]["a"] == 0.0 and out["value"] == pytest.approx((4 * -0.5 - 1) / 5)
    assert V.pcons_reading(m8, {k: v for k, v in preds.items() if k != "v08Fs1"}, units, scores, n_boot=0)["missing"] == ["F"]


def test_m8g_accuracy_holds_every_listing_and_n26():
    m8 = m8g_data(items=4)
    preds = {m: {"order": m8g_preds(m8, wrong_small=m.startswith("v09Cs"))} for m in (*V.ARMS["C"], *V.ARMS["F"])}
    out = V.m8g_accuracy_readings(m8, preds, n_boot=100)
    assert out["n26"]["value"] == pytest.approx(-1.0) and out["n26"]["n"] == 4 * 2 * 2
    assert out["all"]["value"] == pytest.approx(-0.4) and out["all"]["n"] == 4 * 5 * 2
    with pytest.raises(ValueError, match="option order"):  # a seed-2 file read as the record order is refused
        V.m8g_accuracy_readings(m8, {m: {"order": m8g_preds(m8, kind="seed2")} for m in preds}, n_boot=0)


# --------------------------------------------------------------------------
# Synthetic predictions through the whole computation
# --------------------------------------------------------------------------


def test_supported_through_the_pipeline():
    out = scenario(num1=True, num2=True)
    v, r = out["verdict"], out["readings"]
    assert v["H-num"] == "SUPPORTED" and v["H-cons"] == "SUPPORTED", v
    assert r["P-num1"]["delta"]["value"] == pytest.approx(1 - 60 / 324) and r["P-num2"]["value"] == pytest.approx(0.6)
    assert r["P-cons"]["value"] == pytest.approx(-1.0) and r["M8g accuracy"]["all"]["value"] == 0.0
    assert v["guards"]["CN-C"]["pass"] and v["guards"]["C-F"]["pass"], v["guards"]
    assert v["next_steps"]["branches"] == ["H-num SUPPORTED"] and "unread" in v["next_steps"]["notes"][0]
    n9 = r["N9g"]
    assert n9["models"]["CN"]["total"]["accuracy"]["value"] == 1.0 and n9["models"]["base"]["total"]["accuracy"]["value"] == pytest.approx(0.4)
    assert set(n9["models"]["CN"]["by_family"]) == {"threshold", "interval", "selection", "comparison", "conversion"}
    assert n9["differences"]["CN − C"]["all"]["value"] == pytest.approx(0.6) and n9["minus_base"]["CN"] == pytest.approx(0.6)
    assert r["A16s"]["models"]["CN"]["summary"]["total"]["accuracy"]["value"] == 1.0 and "base" in r["A16s"]["models"]
    assert r["B8 flips"]["pooled"]["F"]["value"] == 1.0 and r["B8 flips"]["pooled"]["C"]["value"] == 0.0
    assert r["M8g"]["per_bin"]["C"]["61-255"]["flip_rate"] == 0.0 and r["M8g"]["per_bin"]["F"]["61-255"]["flip_rate"] == 1.0
    assert r["A16"] == {"absent": "no A16 files"} and r["R14"] == {"absent": "no R14 files"} and r["Jev"]["absent"]
    md = V.render({**out, "n_boot": 200, "seed": 0, "git": None, "notes": []})
    for needle in ("# v0.9 pre-declared rule — H-num SUPPORTED · H-cons SUPPORTED", "### P-num1 — gate A16s", "**P-num2 PASS**", "| **mean of five** |",
                   "## Guards", "**H-num SUPPORTED** → CN is the release candidate", "### N9g — the numeric-rule gate", "Accuracy by family:",
                   "### A16s — accuracy per band", "### M8g per bin", "## Reported, not deciding"):
        assert needle in md, needle


def test_partial_both_ways_rejected_and_hcons_failures():
    only1 = scenario(num1=True, num2=False)["verdict"]
    assert only1["H-num"] == "PARTIAL" and only1["P-num1"]["pass"] and not only1["P-num2"]["pass"]
    assert only1["next_steps"]["branches"] == ["H-num PARTIAL with A16s passing and N9g failing"]
    only2 = scenario(num1=False, num2=True)["verdict"]
    assert only2["H-num"] == "PARTIAL" and not only2["P-num1"]["pass"] and only2["P-num2"]["pass"]
    assert only2["next_steps"]["branches"] == ["H-num PARTIAL with N9g passing and A16s failing"]
    neither = scenario(num1=False, num2=False, cons=False)
    v = neither["verdict"]
    assert v["H-num"] == "REJECTED" and v["H-cons"] == "NOT SUPPORTED" and v["next_steps"]["branches"] == ["H-num REJECTED", "H-cons NOT SUPPORTED"]
    assert neither["readings"]["P-cons"]["value"] == pytest.approx(0.0)
    hold = scenario(num1=True, num2=True, small_hold=False)
    v = hold["verdict"]
    assert v["P-cons"]["pass"] and not v["accuracy_holds"]["checks"]["M8g_accuracy_N<=26_ge_-0.02"] and v["H-cons"] == "NOT SUPPORTED"
    assert hold["readings"]["M8g accuracy"]["n26"]["value"] == pytest.approx(-1.0) and hold["readings"]["M8g accuracy"]["all"]["value"] == pytest.approx(-0.4)


def a16_records() -> list[dict]:
    from robo_jev.bench import r16_v08 as R

    out = []
    for rid, distances in {"v08a16-0000": (0.9, -35.5, 82.0), "v08a16-0001": (2.0, 5.0, -12.0)}.items():
        rule = {axis: {"distance": d, "label": R.arm_label(d), "label_boundary_stay": R.arm_label(d, boundary_stay=True), "band": R.arm_band(d)}
                for axis, d in zip("xyz", distances)}
        out.append({"id": rid, "meta": {"set": "v08-a16", "rule": rule},
                    "request": {"state": "servo", "questions": {a: {"type": "choice", "instructions": a, "criteria": {k: k for k in AXIS_KEYS}} for a in "xyz"}}})
    return out


def a16_answers(records: list[dict], right_: bool, reading: str = "single-s1") -> dict:
    out = {}
    for r in records:
        for axis, rule in r["meta"]["rule"].items():
            top = rule["label"] if right_ else "stay"
            out.setdefault(r["id"], {})[axis] = {reading: {"choice": top, "probabilities": {k: (0.6 if k == top else 0.04) for k in AXIS_KEYS}}}
    return out


def test_a16_is_compared_with_v07_under_supported():
    records = a16_records()
    scores = {m: {"answers": a16_answers(records, m.startswith("v09CN"))} for m in (*V.V09_MODELS, *V.F_MODELS, "v07s0", "v07s1")}
    a16 = {"records": {"A16": records, "A16y": records}, "scores": {"A16": scores, "A16y": {}}, "stored": {"A16": a16_answers(records, False, reading="stored")}}
    out = scenario(num1=True, num2=True, a16=a16)
    note = out["verdict"]["next_steps"]["notes"][0]
    assert "CN 1.000 vs v0.7 0.000 (CN − v0.7 +1.000)" in note and "the stored v0.7 endpoint answers 0.000" in note
    assert out["readings"]["A16"]["models"]["CN"]["summary"]["agreement"]["value"] == 1.0
    md = V.render({**out, "n_boot": 200, "seed": 0, "git": None, "notes": []})
    assert "### A16 — the arm's bench requests" in md and "| v0.7 endpoint (stored) |" in md


def test_the_a16s_figure_names_the_v09_series_and_restores_the_module():
    from robo_jev.bench import a16s_v08 as S

    before = dict(S.SERIES_NAME)
    out = scenario(num1=True, num2=False)
    svg = V.a16s_figure(out["readings"]["A16s"])
    assert "v0.9 C (consistency loss)" in svg and "v0.9 CN (+ measurement literacy)" in svg and "v0.8 F (reference)" in svg and "untrained base" in svg
    assert S.SERIES_NAME == before
    assert V.a16s_figure({"models": {}}) is None


# --------------------------------------------------------------------------
# Files: frozen gates, overlaps, trained inputs, bundles, loaders, write guards, refusals
# --------------------------------------------------------------------------


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_frozen_gate_dirs_refuse_without_override(tmp_path):
    from robo_jev.bench import n9g_v09 as N9

    (tmp_path / "N9g.jsonl").write_text('{"id": "v09n9g-0000"}\n', encoding="utf-8")
    (tmp_path / "frozen-before-scoring.json").write_text(json.dumps({"files": {"N9g.jsonl": _sha(tmp_path / "N9g.jsonl")}}), encoding="utf-8")
    assert V.load_frozen_dir(N9.check_frozen, tmp_path, "N9g")["files"]["N9g.jsonl"] == _sha(tmp_path / "N9g.jsonl")
    (tmp_path / "N9g.jsonl").write_text('{"id": "v09n9g-0001"}\n', encoding="utf-8")
    with pytest.raises(SystemExit, match="N9g's frozen files changed"):
        V.load_frozen_dir(N9.check_frozen, tmp_path, "N9g")


def test_overlaps_union_and_refusal(tmp_path, monkeypatch):
    v08, v09 = tmp_path / "overlap-v08.json", tmp_path / "overlap-v09.json"
    v08.write_text(json.dumps({"train_v08": {"sha256": "aaa"}, "excluded_items": ["mmlu:1"], "excluded_b8": ["b"]}), encoding="utf-8")
    monkeypatch.setattr(V, "OVERLAP_V08", v08)
    monkeypatch.setattr(V, "OVERLAP_V09", v09)
    with pytest.raises(SystemExit, match="overlap-v09.json is missing"):
        V.read_overlaps(required=True)
    out, trained, notes = V.read_overlaps(required=False)
    assert out["items"] == {"mmlu:1"} and trained == {"train_v08": "aaa", "train_v09n": None} and any("absent" in n for n in notes)
    v09.write_text(json.dumps({"train_v09n": {"sha256": "bbb"}, "excluded_items": ["mmlu:2"], "excluded_n9g": ["v09n9g-0003"], "excluded_a16s": []}), encoding="utf-8")
    out, trained, _ = V.read_overlaps(required=True)
    assert out["items"] == {"mmlu:1", "mmlu:2"} and out["n9g"] == {"v09n9g-0003"} and out["b8"] == {"b"} and trained["train_v09n"] == "bbb"


def test_trained_inputs_must_be_the_ones_the_rechecks_read(tmp_path):
    cks = {}
    for m in V.V09_MODELS:
        run = tmp_path / m
        run.mkdir()
        sha = "bbb" if m.startswith("v09CN") else "aaa"
        (run / "metrics.json").write_text(json.dumps({"manifest": {"data": {"file_sha256": sha}}}), encoding="utf-8")
        cks[m] = run / "checkpoint.pt"
    assert V.trained_input_problems(cks, {"train_v08": "aaa", "train_v09n": "bbb"}) == []
    problems = V.trained_input_problems(cks, {"train_v08": "aaa", "train_v09n": "ccc"})
    assert len(problems) == 2 and all(p.startswith("v09CN") for p in problems)
    assert len(V.trained_input_problems({**cks, "v09Cs0": None}, {"train_v08": "aaa", "train_v09n": None})) == 1


def test_bundle_mismatch_is_a_problem():
    tags, _ = b8_data(requests_per_env=1)
    scores = {m: b8_scores(tags, flip=False) for m in (*V.ARMS["C"], *V.ARMS["F"])}
    assert V.bundle_problems(scores, (*V.ARMS["C"], *V.ARMS["F"]), set_name="B8") == []
    scores["v08Fs1"] = b8_scores(tags, flip=False, bundle="bundle-0")
    problems = V.bundle_problems(scores, (*V.ARMS["C"], *V.ARMS["F"]), set_name="B8")
    assert len(problems) == 1 and "different bundles" in problems[0] and "v08Fs1" in problems[0]


def test_conditions_must_match_and_a16s_needs_serves_adapter():
    serve = {"serve_adapter": {"used": True}, "key": "the runner's default (sha256 of the state)"}
    scores = {m: dict(serve) for m in (*V.ARMS["C"], *V.ARMS["CN"])}
    models = (*V.ARMS["C"], *V.ARMS["CN"])
    assert V.condition_problems(scores, models, set_name="A16s", serve=True) == []
    assert V.condition_problems({**scores, "v09CNs1": {**serve, "serve_adapter": {"used": False}}}, models, set_name="A16s", serve=True)
    plain = {m: {"serve_adapter": {"used": False}, "key": "k"} for m in models}
    assert V.condition_problems(plain, models, set_name="N9g") == [] and V.condition_problems(plain, models, set_name="A16s", serve=True)
    assert "different order keys" in V.condition_problems({**plain, "v09Cs0": {"serve_adapter": {"used": False}, "key": "other"}}, models, set_name="N9g")[0]


def test_load_bench_reads_each_model_from_its_chain_and_leaves_out_reported_failures(tmp_path):
    v09, v08 = tmp_path / "v09", tmp_path / "v08"
    v09.mkdir()
    v08.mkdir()
    source = tmp_path / "N9g.jsonl"
    source.write_text("x\n", encoding="utf-8")
    V.V8()._sha.cache_clear()
    frozen = {"files": {"N9g.jsonl": _sha(source)}}
    good = {"kind": "v08-bench-scores", "set": "N9g", "input_sha256": _sha(source), "readings": {"single-s1": {}, "single-s2": {}}, "frozen": frozen,
            "checkpoint": None, "answers": {}}
    for m in (*V.V09_MODELS, "v08Fs0", "base"):
        (v09 / f"N9g-{m}-2b.json").write_text(json.dumps({**good, "tag": f"{m}-2b"}), encoding="utf-8")
    (v09 / "N9g-base-2b.json").write_text(json.dumps({**good, "tag": "base-2b", "frozen": {"files": {}}}), encoding="utf-8")
    problems, notes, left = [], [], []
    out = V.load_bench("N9g", source, ("single-s1", "single-s2"), (*V.V09_MODELS, *V.F_MODELS, "base"), {}, spark_dir=v09, v08_spark=v08, frozen=frozen,
                       required=V.V09_MODELS, problems=problems, notes=notes, left_out=left)
    assert set(out) == {*V.V09_MODELS, "v08Fs0"} and problems == [] and any("v08Fs1" in n for n in notes)
    assert len(left) == 1 and left[0].startswith("base N9g") and "frozen" in left[0]
    (v09 / "N9g-v09Cs0-2b.json").write_text(json.dumps({**good, "tag": "v09Cs0-2b", "readings": {"single-s1": {}}}), encoding="utf-8")
    problems = []
    V.load_bench("N9g", source, ("single-s1", "single-s2"), V.V09_MODELS, {}, spark_dir=v09, v08_spark=v08, frozen=frozen, required=V.V09_MODELS,
                 problems=problems, notes=[], left_out=[])
    assert len(problems) == 1 and "readings missing" in problems[0]
    assert V.source_dir("v08Fs0", "A16s", v09, v08) == v08 and V.source_dir("v08Fs0", "N9g", v09, v08) == v09 and V.source_dir("v09CNs0", "B8", v09, v08) == v09


def test_write_guards():
    for place in (V.ARTIFACTS / "v08" / "gates" / "verdict.json", V.ARTIFACTS / "v09" / "n9g" / "x.json", V.ARTIFACTS / "reports" / "v08" / "verdict.json",
                  V.ARTIFACTS / "simgen" / "x.json", V.ARTIFACTS / "t9" / "x.json", V.ARTIFACTS / "cloud" / "x.json"):
        with pytest.raises(SystemExit, match="refusing to write"):
            V.refuse_write(place)
    V.refuse_write(V.ARTIFACTS / "reports" / "v09" / "verdict.json")
    V.refuse_write(V.PIPELINE_OUT)


def test_main_refuses_without_runs_or_files(tmp_path):
    out = tmp_path / "verdict.json"
    with pytest.raises(SystemExit, match="required: --c-s0-dir --c-s1-dir --cn-s0-dir --cn-s1-dir"):
        V.main(["--out", str(out)])
    dirs = []
    for i in range(4):
        d = tmp_path / f"run{i}" / "reports"
        d.mkdir(parents=True)
        dirs.append(str(d))
    with pytest.raises(SystemExit, match="missing files the rule reads"):
        V.main(["--c-s0-dir", dirs[0], "--c-s1-dir", dirs[1], "--cn-s0-dir", dirs[2], "--cn-s1-dir", dirs[3], "--spark-dir", str(tmp_path / "spark"),
                "--out", str(out)])
    with pytest.raises(SystemExit, match="refusing to write"):
        V.main(["--pipeline-check", "--out", str(V.ARTIFACTS / "v08" / "gates" / "x.json")])
    with pytest.raises(SystemExit, match="pipeline-check folder"):
        V.main(["--c-s0-dir", dirs[0], "--out", str(V.PIPELINE_OUT)])
    assert not out.exists()


@pytest.mark.skipif(not (V.V08_VERDICT.is_file() and (V.V08_SPARK / "A16s-v08Fs0-2b.json").is_file()), reason="the v0.8 verdict files are not here")
def test_the_p_num1_path_reproduces_v08s_a16s_numbers():
    """A fast part of the pipeline check on the real files: F's and FP's A16s accuracy and FP − F through the P-num1 path."""
    from robo_jev.bench import a16s_v08 as S

    ref = json.loads(V.V08_VERDICT.read_text(encoding="utf-8"))["readings"]["A16s"]["models"]
    records = V._read(S.A16S_DIR / "A16s.jsonl")
    scores = {m: json.loads((V.V08_SPARK / f"A16s-{m}-2b.json").read_text(encoding="utf-8")) for m in ("v08Fs0", "v08Fs1", "v08FPs0", "v08FPs1")}
    out = V.pnum1_reading(records, scores, arms={"F": ("v08Fs0", "v08Fs1"), "FP": ("v08FPs0", "v08FPs1")}, a="FP", b="F", n_boot=0)
    assert out["delta"]["a"] == ref["FP"]["summary"]["total"]["accuracy"]["value"] and out["delta"]["b"] == ref["F"]["summary"]["total"]["accuracy"]["value"]
