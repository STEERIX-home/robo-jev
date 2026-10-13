"""The v0.4 rule helper (Task v0.4b §1-6) — ``scripts/verdict_v04.py``.

The pure rule is pinned on hand-made readings: each outcome of (S) (SUPPORTED, REJECTED, PARTIAL), two seeds that disagree, the bounds
of every threshold, (R), (N), (F), (G), the release-candidate line and the pre-declared next step. The T9 readings are driven by synthetic
T9-shaped records and predictions: (S) reads only the seen-law families (``meta.law_group`` seen_law_new_scene + seen_law_new_mechanism),
and (F) counts high-called-low answers on robust questions only. No model, no sealed file.
"""

from __future__ import annotations

import functools
import importlib.util
import json
import sys
from pathlib import Path

import pytest
from helpers import REPO

FAMILIES = {"conveyor": "seen_law_new_scene", "overhang": "seen_law_new_scene", "suction": "seen_law_new_mechanism",
            "roll": "unseen_law", "swing": "unseen_law", "knock": "unseen_law"}
CONDITIONS = {"mode": "direct", "format": "chat", "prompt_version": "v0c-p3", "orders": "counterbalanced", "shuffle_seed": 1}


@functools.lru_cache(maxsize=None)
def V():
    spec = importlib.util.spec_from_file_location("v04_verdict_helper", REPO / "scripts" / "verdict_v04.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _d(value=None, low=None, high=None, a=None, b=None) -> dict:
    return {"value": value, "low": low, "high": high, "a": a, "b": b}


# --------------------------------------------------------------------------
# (S)
# --------------------------------------------------------------------------


def _s(v03: dict, base: dict) -> dict:
    return {"d_v03": v03, "d_base": base}


SUPPORT = _s(_d(0.12, 0.04, 0.20), _d(0.30, 0.20, 0.40))
REJECT = _s(_d(0.01, -0.03, 0.04), _d(0.05, 0.01, 0.09))
NEITHER = _s(_d(0.06, 0.01, 0.11), _d(0.20, 0.10, 0.30))


@pytest.mark.parametrize("seed0, seed1, outcome, disagree", [
    (SUPPORT, SUPPORT, "SUPPORTED", False),
    (REJECT, REJECT, "REJECTED", False),
    (NEITHER, NEITHER, "PARTIAL", False),
    (SUPPORT, REJECT, "PARTIAL", True),   # the two seeds disagree: no claim either way
    (SUPPORT, NEITHER, "PARTIAL", True),
    (REJECT, NEITHER, "PARTIAL", True),
])  # fmt: skip
def test_each_s_outcome_and_a_seed_disagreement(seed0, seed1, outcome, disagree):
    result = V().s_outcome({0: seed0, 1: seed1})
    assert result["outcome"] == outcome and result["seeds_disagree"] is disagree


@pytest.mark.parametrize("v03, base, supported, rejected", [
    (_d(0.08, 0.001, 0.15), _d(0.2, 0.001, 0.3), True, False),     # point exactly +0.08 counts (≥)
    (_d(0.0799, 0.01, 0.15), _d(0.2, 0.1, 0.3), False, False),
    (_d(0.10, 0.0, 0.2), _d(0.2, 0.1, 0.3), False, False),         # lower bound exactly 0 does not (>)
    (_d(0.10, 0.02, 0.2), _d(0.2, 0.0, 0.3), False, False),         # v0.4 − base lower bound must be > 0
    (_d(0.00, -0.05, 0.05), _d(0.0, -0.1, 0.1), False, False),      # upper bound exactly +0.05 does not reject (<)
    (_d(0.00, -0.05, 0.0499), _d(0.0, -0.1, 0.1), False, True),
    (None, _d(0.2, 0.1, 0.3), False, False),                        # a missing reading is neither
])  # fmt: skip
def test_the_s_thresholds_at_their_bounds(v03, base, supported, rejected):
    checks = V().s_seed_checks(v03, base)
    assert (checks["v03_point_ge_0.08"] and checks["v03_low_gt_0"] and checks["base_low_gt_0"]) is supported
    assert checks["v03_high_lt_0.05"] is rejected


def test_s_needs_both_seeds():
    with pytest.raises(ValueError, match="both seeds"):
        V().s_outcome({0: SUPPORT})


# --------------------------------------------------------------------------
# (R) (N) (F) (G), release, next step
# --------------------------------------------------------------------------


def _r_ok() -> dict:
    return {s: {"noul.brier": _d(0.002, -0.001, 0.0099), "score.brier": _d(0.0, -0.004, 0.004),
                "choice/action.regret_mean[default]": _d(0.1, -0.2, 0.49)} for s in ("T1", "T3", "T4", "T5", "T7")}


def test_r_reads_v02s_c_on_t1_t3_t4_t5_t7_and_fails_a_missing_reading():
    assert V().r_checks(_r_ok())["pass"] is True
    for key, high in (("noul.brier", 0.01), ("score.brier", 0.0101), ("choice/action.regret_mean[default]", 0.5)):
        bad = _r_ok()
        bad["T4"][key] = _d(0.0, -0.01, high)
        assert V().r_checks(bad)["pass"] is False, key
    missing = _r_ok()
    del missing["T7"]
    assert V().r_checks(missing)["pass"] is False
    gap = _r_ok()
    gap["T1"]["score.brier"] = None
    assert V().r_checks(gap)["pass"] is False


def _n(**changes) -> tuple[dict, dict, dict]:
    t9 = {"direction": _d(0.0, -0.04, 0.05), "comparison": _d(0.01, -0.03, 0.06)}
    t8b = {"direction": _d(0.02, -0.049, 0.08), "comparison": _d(0.0, -0.02, 0.03)}
    home = {"band_balanced": _d(0.0, -0.01, 0.01), "severe_error": _d(0.01, 0.0, 0.02, a=0.031, b=0.021)}
    for path, value in changes.items():
        part, metric = path.split("__")
        {"t9": t9, "t8b": t8b, "home": home}[part][metric] = value
    return t9, t8b, home


@pytest.mark.parametrize("changes, ok", [
    ({}, True),
    ({"t9__direction": _d(-0.02, -0.05, 0.01)}, False),             # lower bound exactly −0.05 fails (>)
    ({"t9__comparison": _d(-0.02, -0.0499, 0.01)}, True),
    ({"t8b__direction": _d(-0.1, -0.2, 0.0)}, False),
    ({"t8b__comparison": None}, False),
    ({"home__band_balanced": _d(-0.03, -0.06, 0.0)}, False),
    ({"home__severe_error": _d(0.0101, 0.0, 0.02, a=0.0311, b=0.021)}, False),  # severe > v0.1-mix + 0.01
    ({"home__severe_error": _d(0.01, 0.0, 0.02, a=0.03, b=0.02)}, True),         # exactly + 0.01 passes (≤)
])  # fmt: skip
def test_n_non_regression_bounds(changes, ok):
    assert V().n_checks(*_n(**changes))["pass"] is ok


def test_f_and_g_bounds():
    assert V().f_check(_d(0.0, a=0.12, b=0.12))["pass"] is True
    assert V().f_check(_d(0.001, a=0.121, b=0.12))["pass"] is False
    assert V().f_check(None)["pass"] is False
    good = {"diff": -0.02, "low": -0.0449, "high": 0.01}
    assert V().g_check(good)["pass"] is True
    assert V().g_check({**good, "diff": -0.0201})["pass"] is False and V().g_check({**good, "low": -0.045})["pass"] is False


def _seed_entry(s: dict, *, g: dict | None = None, hset_low: float | None = 0.02) -> dict:
    t9, t8b, home = _n()
    return {"S": s, "R": _r_ok(), "N": {"T9": t9, "T8b": t8b, "home": home}, "F": _d(0.0, a=0.1, b=0.12),
            "G": g or {"diff": 0.0, "low": -0.02, "high": 0.02, "a": 0.67, "b": 0.67},
            "secondary": {"Hset_level": {"v04_minus_v03": _d(0.1, hset_low, 0.2)}}}


def test_the_release_line_needs_s_supported_and_every_part_for_both_seeds():
    both = V().decide({"seeds": {"0": _seed_entry(SUPPORT), "1": _seed_entry(SUPPORT)}})
    assert both["release_candidate"] is True and both["release_line"].startswith("RELEASE CANDIDATE")
    g_fail = V().decide({"seeds": {"0": _seed_entry(SUPPORT), "1": _seed_entry(SUPPORT, g={"diff": -0.03, "low": -0.06, "high": 0.0})}})
    assert g_fail["release_candidate"] is False and "G fails" in g_fail["release_line"] and g_fail["parts"]["G"] is False
    partial = V().decide({"seeds": {"0": _seed_entry(SUPPORT), "1": _seed_entry(NEITHER)}})
    assert partial["release_candidate"] is False and "S PARTIAL" in partial["release_line"]


@pytest.mark.parametrize("s0, s1, lows, branch", [
    (SUPPORT, SUPPORT, (0.02, 0.02), "S supported"),
    (SUPPORT, REJECT, (0.02, 0.02), "S partial"),
    (REJECT, REJECT, (0.01, 0.03), "S rejected and H-set up (each seed)"),
    (REJECT, REJECT, (0.01, 0.0), "S rejected and H-set flat (not up for each seed)"),   # one seed's lower bound exactly 0: not up
    (REJECT, REJECT, (None, 0.05), "S rejected and H-set flat (not up for each seed)"),  # no H-set reading: not up
])  # fmt: skip
def test_the_pre_declared_next_step(s0, s1, lows, branch):
    result = V().decide({"seeds": {"0": _seed_entry(s0, hset_low=lows[0]), "1": _seed_entry(s1, hset_low=lows[1])}})
    assert result["next_step"]["branch"] == branch
    expected = {"S supported": "law diversity", "S partial": "third seed", "S rejected and H-set up (each seed)": "authoring style",
                "S rejected and H-set flat (not up for each seed)": "engine factor coverage"}[branch]
    assert expected in result["next_step"]["next"]


# --------------------------------------------------------------------------
# T9 readings on synthetic T9-shaped records
# --------------------------------------------------------------------------


def _t9(pairs: int = 12, *, key_high: bool = False) -> list[dict]:
    out = []
    for family, group in FAMILIES.items():
        for i in range(pairs):
            for m in ("a", "b"):
                rid, partner = f"t9-{family}-{i:04d}{m}", f"t9-{family}-{i:04d}{'b' if m == 'a' else 'a'}"
                shift = i + (m == "b")
                key = 0.9 if key_high else (0.1, 0.5, 0.9)[shift % 3]
                questions = {"key": {"type": "noul", "instructions": "Will it fail?"},
                             "cf": {"type": "score", "instructions": "How likely?", "criteria": ["0 never", "1 always"]},
                             "alt": {"type": "noul", "instructions": "And if gentler?"},
                             "dir": {"type": "choice", "instructions": "Which way?", "criteria": {"likelier": "l", "less_likely": "s", "no_real_change": "n"}},
                             "outcome": {"type": "choice", "instructions": "What happens?", "criteria": {"stays": "s", "slides": "d", "tips": "t"}}}
                labels = {"key": {"type": "noul", "p_yes": key}, "cf": {"type": "score", "value": (0.9, 0.1, 0.5)[shift % 3]},
                          "alt": {"type": "noul", "p_yes": (0.1, 0.9, 0.5)[shift % 3]},
                          "dir": {"type": "choice", "kind": "state", "probabilities": {"likelier": 1.0, "less_likely": 0.0, "no_real_change": 0.0}},
                          "outcome": {"type": "choice", "kind": "state", "probabilities": {"stays": 0.6, "slides": 0.3, "tips": 0.1}}}
                robustness = {"key": {"robust": i % 2 == 0}, "cf": {"robust": True}, "alt": {"robust": False}, "dir": {"robust": True},
                              "outcome": {"robust": True}}
                out.append({"schema": "simgen-request/0", "id": rid, "request": {"state": f"{family} {i} {m}", "questions": questions}, "labels": labels,
                            "meta": {"set": "T9", "family": family, "scene": family, "law_group": group, "events": ["key", "cf", "alt"],
                                     "outcome_question": "outcome", "direction_question": "dir", "robustness": robustness,
                                     "pair": {"of": partner, "kind": "numeric", "role": "base" if m == "a" else "variant"}}})
    return out


def _predict(records: list[dict], scalar) -> list[dict]:
    """Prediction lines in the diagnosis tool's shape (one listing each): yes/no and score answers from `scalar(record, name, label)`,
    choice answers = the label."""
    out = []
    for record in records:
        questions = {}
        for name, question in record["request"]["questions"].items():
            label = record["labels"][name]
            if question["type"] == "choice":
                answer, options = {"probabilities": dict(label["probabilities"])}, list(question["criteria"])
            else:
                y = label.get("p_yes", label.get("value"))
                p = float(scalar(record, name, y))
                answer, options = ({"p_yes": p} if question["type"] == "noul" else {"value": p, "bins": None}), None
            questions[name] = {"type": question["type"], "n_listings": 1, "answer": answer, "listings": [{"options": options, "values": None, "answer": answer}]}
        out.append({"id": record["id"], "model": "synthetic", **CONDITIONS, "questions": questions})
    return out


def _seen(record: dict) -> bool:
    return record["meta"]["law_group"] != "unseen_law"


ORACLE = lambda r, n, y: y  # noqa: E731
MIDDLE = lambda r, n, y: 0.5  # noqa: E731


def test_s_reads_only_the_seen_law_families():
    t9 = _t9()
    assert {r["meta"]["family"] for r in V().seen_law(t9)} == {"conveyor", "overhang", "suction"}
    assert len(V().seen_law(t9)) == len(t9) // 2 and not {r["meta"]["family"] for r in V().unseen_law(t9)} & {"conveyor", "overhang", "suction"}
    v04 = _predict(t9, lambda r, n, y: y if _seen(r) else 0.5)   # right where the law was trained in new scenes, "middle" elsewhere
    v03 = _predict(t9, lambda r, n, y: 0.5 if _seen(r) else y)   # the mirror image
    base = _predict(t9, MIDDLE)
    readings = V().t9_readings(t9, v04, v03, base, n_boot=200, seed=0)
    s = readings["S"]["d_v03"]
    assert s["a"] == pytest.approx(1.0) and s["b"] == pytest.approx(1 / 3) and s["value"] == pytest.approx(2 / 3) and s["low"] > 0
    everything = V().level(t9, "T9", v04, v03, n_boot=0, seed=0)
    assert abs(everything["value"]) < 1e-9, "over all six families the two models tie — only the filter separates them"
    unseen = readings["secondary"]["T9_unseen_law_level"]["v04_minus_v03"]
    assert unseen["value"] == pytest.approx(-2 / 3)
    assert s["clusters"] == 3 * 12, "one cluster per pair group (a and b together)"


def test_s_outcomes_from_synthetic_predictions_including_a_seed_disagreement():
    t9 = _t9()
    v03, base = _predict(t9, MIDDLE), _predict(t9, lambda r, n, y: 0.1)
    good = _predict(t9, ORACLE)
    same = _predict(t9, MIDDLE)
    read = lambda preds: V().t9_readings(t9, preds, v03, base, n_boot=200, seed=0)["S"]  # noqa: E731
    assert V().s_outcome({0: read(good), 1: read(good)})["outcome"] == "SUPPORTED"
    assert V().s_outcome({0: read(same), 1: read(same)})["outcome"] == "REJECTED"
    split = V().s_outcome({0: read(good), 1: read(same)})
    assert split["outcome"] == "PARTIAL" and split["seeds_disagree"] is True


def test_f_counts_high_called_low_on_robust_questions_only():
    t9 = _t9(key_high=True)  # every key label high; key robust in even pairs only (cf robust and low, alt prior-dependent)

    def low_on(rule):
        return lambda r, n, y: (0.1 if n == "key" and rule(int(r["id"][-5:-1])) else y)

    base = _predict(t9, low_on(lambda i: i % 4 == 0))                     # half of the robust high keys called low
    worse = _predict(t9, low_on(lambda i: i % 2 == 0))                    # every robust high key called low
    better_robust = _predict(t9, low_on(lambda i: i % 2 == 1))            # only prior-dependent high keys called low
    v03 = _predict(t9, ORACLE)
    f = lambda preds: V().f_check(V().t9_readings(t9, preds, v03, base, n_boot=0, seed=0)["F"])  # noqa: E731
    assert f(worse)["pass"] is False and f(worse)["v04"] > f(worse)["base"]
    assert f(better_robust)["pass"] is True and f(better_robust)["v04"] == 0.0, "misses on prior-dependent questions do not count"
    assert f(base)["pass"] is True, "equal to the base passes (≤)"
    robust, ambiguous = V().t9_keys(t9)
    assert ("t9-roll-0001a", "key") in ambiguous and ("t9-roll-0000a", "key") in robust and all(q != "alt" for _, q in robust)


def test_hedge_share_and_the_robust_two_band_reading_are_reported():
    t9 = _t9()
    readings = V().t9_readings(t9, _predict(t9, ORACLE), _predict(t9, MIDDLE), _predict(t9, lambda r, n, y: 0.1), n_boot=0, seed=0)
    hedge = readings["secondary"]["T9_hedge_share_prior_dependent"]
    assert hedge["v03"] == pytest.approx(1.0) and hedge["base"] == pytest.approx(0.0)
    two = readings["secondary"]["T9_robust_two_band"]["v04_minus_v03"]
    assert two["a"] == pytest.approx(1.0) and two["b"] == pytest.approx(0.0), "two-band: low and high strata only (middle answers miss both)"


# --------------------------------------------------------------------------
# files
# --------------------------------------------------------------------------


def test_prediction_names_and_default_paths():
    v = V()
    assert v.pod_prediction_name(1, "T3") == "after-v04-s1-2b-T3.predictions.jsonl" and v.spark_prediction_name("Hset", "v03") == "Hset-v03-2b.predictions.jsonl"
    import argparse

    args = argparse.Namespace(s0_dir=Path("/s0"), s1_dir=Path("/s1"), v03_dir=v.V03_POD, mix_dir=v.MIX_POD, spark_dir=Path("/spark"), base_g2=v.BASE_G2_POD)
    paths = v.prediction_paths(args)
    assert paths[("v04s0", "G2")] == Path("/s0/after-v04-s0-2b-G2.predictions.jsonl") and paths[("v03", "T9")] == Path("/spark/T9-v03-2b.predictions.jsonl")
    assert paths[("v03", "T8b")] == Path("/spark/T8b-v03-2b.predictions.jsonl"), "v0.3 re-scored on Spark for T8b"
    assert paths[("base", "T9")] == v.EXISTING_SPARK[("base", "T9")] and paths[("v03", "B1")] == v.EXISTING_SPARK[("v03", "B1")]
    need = set(v.required_inputs())
    assert {("v03", "T9"), ("base", "T9"), ("v04s1", "T8b"), ("mix", "T7"), ("v04s0", "G2"), ("v03", "G2")} <= need
    assert not any(s in ("B1", "Hset", "T6", "G1", "T8a") for _, s in need), "secondary inputs are not required"


def test_a_prediction_file_scored_on_another_set_or_conditions_is_flagged(tmp_path):
    t9 = _t9(pairs=2)
    set_path = tmp_path / "T9.jsonl"
    set_path.write_text("".join(json.dumps(r) + "\n" for r in t9), encoding="utf-8")
    preds = _predict(t9, ORACLE)
    path = tmp_path / "T9-x.predictions.jsonl"
    path.write_text("".join(json.dumps(p) + "\n" for p in preds), encoding="utf-8")
    from robo_jev.decisions.records import file_sha256

    (tmp_path / "T9-x.json").write_text(json.dumps({"config": {"input_sha256": file_sha256(set_path), "model": "m"}}), encoding="utf-8")
    assert V().check_source(path, set_path, t9, preds)["problems"] == []
    (tmp_path / "T9-x.json").write_text(json.dumps({"config": {"input_sha256": "0" * 64}}), encoding="utf-8")
    assert "another file" in V().check_source(path, set_path, t9, preds)["problems"][0]
    odd = [dict(p, shuffle_seed=2) for p in preds]
    assert any("shuffle_seed" in x for x in V().check_source(path, set_path, t9, odd)["problems"])
