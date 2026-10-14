"""Task iface — the pre-declared interface measurements M1 (many options) and M2 (bench-shaped long states): set construction
(:mod:`robo_jev.bench.iface`), the M1 scoring script's cap flag (``scripts/iface_m1_score.py``), the two reading rules, and the verdict's
M2 helpers (``scripts/iface_verdict.py``)."""

import copy
import functools
import importlib.util
import json
import sys

import numpy as np
import pytest
from helpers import REPO

from robo_jev.bench import iface
from robo_jev.decisions.records import OPTION_RANGE, validate_record


@functools.lru_cache(maxsize=None)
def _script(name: str):
    spec = importlib.util.spec_from_file_location(f"{name}_ifacetest", REPO / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


# -- M1: general items ----------------------------------------------------------------------------------------------------------------

TOPICS = ["volcano", "magnet", "tadpole", "glacier", "comet", "fungus", "pulley", "prism", "coral", "desert", "lichen", "turbine", "plankton",
          "meteor", "cactus", "piston", "orchid", "lagoon", "beetle", "quartz", "geyser", "walrus", "nebula", "acorn", "sponge", "falcon",
          "tundra", "lever", "salmon", "basalt", "pollen", "rainbow", "kidney", "anchor", "violin", "carbon"]


def _word(i: int, prefix: str) -> str:
    """A digit-free word unique to `i` (numbers in options drop an ARC item)."""
    return prefix + "".join("abcdefghij"[int(d)] for d in str(i))


def arc_row(i: int, *, stem: str | None = None, options: list[str] | None = None, key: str = "B") -> dict:
    topic = TOPICS[i % len(TOPICS)]
    return {"id": f"q{i:03d}", "subset": "ARC-Easy" if i % 2 else "ARC-Challenge", "question": stem or f"Which {topic} {_word(i, 'zor')}?",
            "choices": {"text": options or [f"{topic} {_word(i, 'kel')} {_word(i, 'bax')}", f"{topic} {_word(i, 'mor')} {_word(i, 'dun')}",
                                            f"{_word(i, 'plu')} {_word(i, 'tav')} {_word(i, 'rim')}", f"{_word(i, 'sen')} {_word(i, 'gro')} {_word(i, 'wex')}"],
                        "label": ["A", "B", "C", "D"]}, "answerKey": key}


def test_arc_items_drop_what_drawn_distractors_could_make_ambiguous():
    rows = [arc_row(i) for i in range(6)]
    rows.append(arc_row(6, options=["a", "b", "c"]))  # three options
    rows.append(arc_row(7, options=["red", "blue", "all of the above", "green"]))
    rows.append(arc_row(8, stem="Which part of the diagram shows the root?"))
    rows.append(arc_row(9, stem="All of the following are mammals EXCEPT"))
    rows.append(arc_row(10, stem="Which of these is not a planet?"))
    rows.append(arc_row(11, options=["one year", "a week", "a day", "an hour"]))
    rows.append(arc_row(12, options=["12 cm", "a week", "a day", "an hour"]))
    rows.append(arc_row(13, key="E"))
    rows.append(arc_row(1))  # duplicate question
    items, dropped = iface.arc_items(rows)
    assert [i["item"] for i in items] == [f"arc:q{i:03d}" for i in range(6)]
    assert dropped == {"not_four_options": 1, "option_refers_to_options": 1, "refers_to_a_picture": 1, "negative_question": 2,
                       "number_in_an_option": 2, "no_gold": 1, "duplicate_question": 1}
    first = items[0]
    assert first["state"] == f'question: "{rows[0]["question"]}"' and first["instructions"] == iface.ARC_INSTRUCTION
    assert first["correct"] == rows[0]["choices"]["text"][1] and len(first["own"]) == 3 and first["correct"] not in first["own"]


def test_general_distractors_come_from_other_topics_share_no_word_and_are_unique():
    rows = [arc_row(i) for i in range(80)]
    items, _ = iface.arc_items(rows)
    assert len(items) == 80
    index = iface.general_index(items)
    item = items[0]
    drawn = iface.draw_general(item, index, count=22, seed=0)
    assert len(drawn) == 22 and len({d.lower() for d in drawn}) == 22
    stem = iface.content_tokens(item["stem"])
    own = set().union(*(iface.content_tokens(t) for t in [item["correct"], *item["own"]]))
    for text in drawn:
        tokens = iface.content_tokens(text)
        assert tokens and not tokens & (stem | own) and len(text.split()) >= iface.MIN_DISTRACTOR_WORDS
    sources = {}
    for entry in index:
        for text, _ in entry["wrong"]:
            sources.setdefault(text, entry["item"])
    assert len({sources[t] for t in drawn}) == 22 and item["item"] not in {sources[t] for t in drawn}
    assert drawn == iface.draw_general(item, index, count=22, seed=0)  # deterministic
    with pytest.raises(ValueError, match="general distractors"):
        iface.draw_general(item, index[:5], count=22, seed=0)


def test_general_targets_answer_in_phrases_and_bare_nouns_are_never_drawn():
    rows = [arc_row(i) for i in range(40)] + [arc_row(40, options=["heat", "light and sound energy", "plain wrong answer", "salt"], key="A")]
    items, _ = iface.arc_items(rows)
    assert [iface.general_target(i) for i in items] == [True] * 40 + [False]
    pool = {t for entry in iface.general_index(items) for t, _ in entry["wrong"]}
    assert "salt" not in pool and "plain wrong answer" in pool


def test_content_tokens_keep_numbers_and_drop_stop_words():
    assert iface.content_tokens("The plants grow in 3 days, one by one") == {"plant", "grow", "3", "day", "one"}
    assert iface.content_tokens("which of these is the best") == set()


# -- M1: physics items ------------------------------------------------------------------------------------------------------------------


def physics_record(rid: str, *, partner: str | None = None, best: str = "less_likely", top: float = 0.95, robust: bool = True, scene: str = "tilt") -> dict:
    criteria = {"more_likely": "more likely than with the plan", "less_likely": "less likely than with the plan", "about_same": "about the same"}
    probs = {k: (top if k == best else (1 - top) / 2) for k in criteria}
    other = {"stays": f"stays {rid}", "slides": f"slides {rid}", "topples": f"topples {rid}"}
    return {"schema": "simgen-request/0", "id": rid,
            "meta": {"scene": scene, "question_kind": {"effect": "direction", "outcome": "outcome"},
                     "robustness": {"effect": {"robust": robust}, "outcome": {"robust": True}},
                     "pair": {"of": partner, "kind": "numeric", "role": "base"} if partner else None},
            "request": {"state": f"state of {rid}", "questions": {"effect": {"type": "choice", "instructions": "more or less likely?", "criteria": criteria},
                                                                  "outcome": {"type": "choice", "instructions": "what happens?", "criteria": other}}},
            "labels": {"effect": {"type": "choice", "kind": "state", "probabilities": probs},
                       "outcome": {"type": "choice", "kind": "state", "probabilities": {"stays": 1.0, "slides": 0.0, "topples": 0.0}}}}


def test_physics_items_are_robust_direction_questions_with_a_clear_best_option():
    records = [physics_record("r1", partner="r1p"), physics_record("r1p", partner="r1"), physics_record("r2", top=0.8), physics_record("r3", robust=False)]
    items, counts = iface.physics_items(records, source="b1")
    assert [i["item"] for i in items] == ["b1:r1:effect", "b1:r1p:effect"]
    assert counts == {"direction_questions": 4, "eligible": 2, "top_below_0.9": 1, "not_robust": 1}
    assert items[0]["group"] == items[1]["group"] == "r1" and items[0]["correct"] == ("less_likely", "less likely than with the plan")
    assert [k for k, _ in items[0]["own"]] == ["more_likely", "about_same"]


def test_physics_distractors_have_new_unique_keys_own_set_first():
    own_pool = {"stays": [("x1", "x1", "stays a")], "slides": [("x1", "x1", "slides a")], "more_likely": [("x1", "x1", "m")],
                "held": [("r1", "r1", "held of my own pair")]}
    sibling = {"unmoved": [("h1", "h1", "nothing moved")], "skids": [("h1", "h1", "it skidded")], "stays": [("h1", "h1", "dup key")]}
    item = {"item": "b1:r1:effect", "group": "r1", "keys": ["about_same", "less_likely", "more_likely"]}
    drawn = iface.draw_physics(item, own_pool, sibling, count=4, seed=1)
    keys = [k for k, _ in drawn]
    assert sorted(keys[:2]) == ["slides", "stays"] and sorted(keys[2:]) == ["skids", "unmoved"]  # 'held' only from the item's own pair: skipped
    assert len(set(keys)) == 4 and not set(keys) & set(item["keys"])
    with pytest.raises(ValueError, match="distinct distractor keys"):
        iface.draw_physics(item, own_pool, sibling, count=5, seed=1)


def test_select_physics_takes_one_record_per_pair_and_spreads_the_scenes():
    items = []
    for scene in ("tilt", "grip", "tower"):
        for g in range(10):
            for member in ("", "p"):
                items.append({"item": f"{scene}{g}{member}", "group": f"{scene}{g}", "scene": scene})
    chosen = iface.select_physics(items, 12, seed=3)
    assert len(chosen) == 12 and len({c["group"] for c in chosen}) == 12
    assert {s: sum(c["scene"] == s for c in chosen) for s in ("tilt", "grip", "tower")} == {"tilt": 4, "grip": 4, "tower": 4}


# -- M1: positions and expansion --------------------------------------------------------------------------------------------------------


def test_positions_cover_the_buckets_at_large_n_and_the_thirds_below():
    for n in iface.N_LEVELS:
        seen = set()
        for rank in range(60):
            pos = iface.positions(n, rank)
            assert len(pos) == iface.LISTINGS_BY_N[n] and len(set(pos)) == len(pos) and all(0 <= p < n for p in pos)
            if n >= iface.LATE_MIN_N:
                assert [iface.bucket_of(p, n) for p in pos] == ["A-H", "I-P", "Q-Z"]
            seen |= set(pos)
        assert seen == set(range(n))  # every letter is used
    assert iface.bucket_of(20, 21) == "Q-Z" and iface.bucket_of(7, 26) == "A-H" and iface.bucket_of(8, 26) == "I-P" and iface.bucket_of(5, 16) is None


def _general_item() -> tuple[dict, list[str]]:
    item = {"item": "arc:x1", "source": "ARC-Easy", "state": 'question: "Why?"', "instructions": iface.ARC_INSTRUCTION, "stem": "Why?",
            "correct": "the right one", "own": ["wrong a", "wrong b", "wrong c"]}
    return item, [f"extra {i}" for i in range(22)]


@pytest.mark.parametrize("kind", ["general", "physics"])
def test_expand_item_keeps_item_and_answer_nests_the_options_and_controls_the_position(kind):
    if kind == "general":
        item, drawn = _general_item()
    else:
        item = {"item": "b1:r1:effect", "source": "b1", "state": "s", "instructions": "more or less?", "correct": ("less_likely", "less"),
                "own": [("more_likely", "more"), ("about_same", "same")], "keys": ["about_same", "less_likely", "more_likely"]}
        drawn = [(f"key{i}", f"desc {i}") for i in range(23)]
    records = iface.expand_item(item, drawn, rank=5, index=5, seed=0)
    assert len(records) == sum(iface.LISTINGS_BY_N.values()) and len({r["id"] for r in records}) == len(records)
    previous: set | None = None
    for n in iface.N_LEVELS:
        listings = [r for r in records if r["meta"]["n"] == n]
        texts = []
        for record in listings:
            criteria = record["request"]["questions"][iface.QUESTION]["criteria"]
            keys = list(criteria)
            assert len(keys) == n and record["meta"]["position"] == keys.index(record["meta"]["correct"])
            assert record["meta"]["letter"] == chr(65 + record["meta"]["position"])
            correct_text = criteria[record["meta"]["correct"]]
            assert correct_text == (item["correct"] if kind == "general" else item["correct"][1])
            assert record["labels"][iface.QUESTION]["probabilities"][record["meta"]["correct"]] == 1.0
            others = [(k, v) for k, v in criteria.items() if k != record["meta"]["correct"]]
            texts.append(others)
            if kind == "general":
                assert sorted(keys, key=lambda k: int(k[1:])) == [f"o{i + 1}" for i in range(n)]
        assert all(t == texts[0] for t in texts)  # the same other options, same order and keys, in every listing of this N
        shown = {v for _, v in texts[0]}
        if previous is not None:
            assert previous < shown  # nested in the larger N
        previous = shown
        assert [r["meta"]["position"] for r in listings] == iface.positions(n, 5)
    own = {t if kind == "general" else t[1] for t in item["own"]}
    small = {v for k, v in records[0]["request"]["questions"][iface.QUESTION]["criteria"].items() if k != records[0]["meta"]["correct"]}
    assert own <= small  # the item's own wrong options are always shown


def test_the_cap_holds_unless_the_m1_script_lifts_it():
    item, drawn = _general_item()
    records = iface.expand_item(item, drawn, rank=0, index=0, seed=0)
    for record in records:
        n = record["meta"]["n"]
        standard = iface.m1_problems(record)
        assert standard == validate_record(record)
        assert (not standard) == (n <= OPTION_RANGE[1])
        if n > OPTION_RANGE[1]:
            assert standard == [f"question 'q': choice needs {OPTION_RANGE[0]}..{OPTION_RANGE[1]} options (has {n})"]
        assert iface.m1_problems(record, lift_cap=True) == []
    assert OPTION_RANGE == (2, 8)
    broken = copy.deepcopy(records[-1])
    broken["request"]["state"] = "x" * 16_001  # every other check stays with the cap lifted
    assert any("characters" in p for p in iface.m1_problems(broken, lift_cap=True))
    too_many = copy.deepcopy(records[-1])
    criteria = too_many["request"]["questions"]["q"]["criteria"]
    criteria["o27"] = "one more"
    too_many["labels"]["q"]["probabilities"]["o27"] = 0.0
    assert any("2..26 options even with the cap lifted" in p for p in iface.m1_problems(too_many, lift_cap=True))


# -- M1: scoring script (tiny model, CPU) -------------------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def m1_file(tmp_path_factory):
    path = tmp_path_factory.mktemp("m1") / "m1-set.jsonl"
    item, drawn = _general_item()
    second = dict(item, item="arc:x2", correct="another right one")
    records = iface.expand_item(item, drawn, rank=0, index=0, seed=0) + iface.expand_item(second, drawn, rank=1, index=1, seed=0)
    path.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
    return path, records


def test_the_m1_script_refuses_many_options_without_the_flag(m1_file, tmp_path):
    path, _ = m1_file
    with pytest.raises(SystemExit, match="--lift-option-cap"):
        _script("iface_m1_score").main(["--model", "tiny", "--input", str(path), "--out", str(tmp_path / "x.json")])


def test_the_m1_script_scores_with_the_models_own_prompt_and_readout(m1_file, tmp_path):
    pytest.importorskip("transformers")
    from robo_jev.model.backbone_qwen import torch_reference_kernels

    path, records = m1_file
    out = tmp_path / "M1-tiny.json"
    with torch_reference_kernels():
        assert _script("iface_m1_score").main(["--model", "tiny", "--lift-option-cap", "--input", str(path), "--out", str(out), "--limit-items", "1"]) == 0
    summary = json.loads(out.read_text(encoding="utf-8"))
    predictions = [json.loads(line) for line in out.with_suffix(".predictions.jsonl").read_text(encoding="utf-8").splitlines()]
    mine = [r for r in records if r["meta"]["item"] == "arc:x1"]
    assert summary["config"]["lift_option_cap"] is True and summary["config"]["prompt_version"] == "v0c-p3" and summary["config"]["format"] == "chat"
    assert summary["run"]["prompts"] == len(mine) == len(predictions)
    for record, prediction in zip(mine, predictions):
        listing = prediction["questions"]["q"]["listings"][0]
        assert listing["options"] == list(record["request"]["questions"]["q"]["criteria"])  # shown in the record's order: no reshuffle
        assert len(listing["code_probs"]) == record["meta"]["n"] and abs(sum(listing["code_probs"]) - 1) < 1e-5
        assert listing["codes"] == [chr(65 + i) for i in range(record["meta"]["n"])]
    rows = iface.m1_rows(mine, predictions)
    assert len(rows) == len(mine) and {r["n"] for r in rows} == set(iface.N_LEVELS)
    assert set(summary["glance_by_n"]) == {str(n) for n in iface.N_LEVELS}


# -- M1: readings and the rule --------------------------------------------------------------------------------------------------------------


def synthetic_rows(items: int, *, acc: dict, late_bucket: dict | None = None, seed: int = 0) -> list[dict]:
    """Rows of a model whose accuracy is acc[n] (and, at N >= 17, late_bucket[bucket] if given) — deterministic per item."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(items):
        for n in iface.N_LEVELS:
            for listing, pos in enumerate(iface.positions(n, i)):
                bucket = iface.bucket_of(pos, n)
                p = (late_bucket or {}).get(bucket, acc[n]) if bucket else acc[n]
                hit = float(rng.random() < p)
                rows.append({"item": f"i{i}", "source": "ARC-Easy", "n": n, "position": pos, "bucket": bucket, "listing": listing, "correct": hit,
                             "p_correct": 0.5 + 0.4 * hit, "code_mass": 0.9})
    return rows


def test_m1_readings_and_the_three_outcomes():
    flat = {n: 0.8 for n in iface.N_LEVELS}
    base = synthetic_rows(1000, acc=flat, seed=1)
    good = synthetic_rows(1000, acc=flat, seed=2)
    biased = synthetic_rows(1000, acc=flat, late_bucket={"A-H": 0.85, "I-P": 0.7, "Q-Z": 0.45}, seed=3)
    worse = synthetic_rows(1000, acc={**flat, 21: 0.6, 26: 0.6}, seed=4)
    readings = iface.m1_readings({"base": base, "good": good, "biased": biased, "worse": worse}, base="base", n_boot=300, seed=0)
    assert readings["items"] == 1000
    pen = readings["paired_vs_base"]["biased"]["penalty"]
    assert pen["value"] > 0.3 and pen["low"] > 0
    assert abs(readings["models"]["good"]["late_minus_8"]["value"]) < 0.05
    outcome = {m: iface.m1_model_outcome(readings["paired_vs_base"][m], readings["models"][m])["outcome"] for m in ("good", "biased", "worse")}
    assert outcome == {"good": "GENERALISES", "biased": "FINE-TUNE BIAS", "worse": "MIXED"}
    assert iface.m1_outcome(readings, ["good"])["outcome"] == "GENERALISES"
    assert iface.m1_outcome(readings, ["good", "biased"])["outcome"] == "FINE-TUNE BIAS"
    assert iface.m1_outcome(readings, ["good", "worse"])["outcome"] == "MIXED"


def test_m1_readings_use_only_items_every_model_answered_completely():
    flat = {n: 0.7 for n in iface.N_LEVELS}
    a = synthetic_rows(50, acc=flat, seed=1)
    b = [r for r in synthetic_rows(50, acc=flat, seed=2) if not (r["item"] == "i3" and r["n"] == 26 and r["listing"] == 2)]
    readings = iface.m1_readings({"base": a, "other": b}, base="base", n_boot=50, seed=0)
    assert readings["items"] == 49


# -- M2: sheets -------------------------------------------------------------------------------------------------------------------------------


def test_wrapped_sheets_hit_the_lengths_keep_the_state_verbatim_and_share_filler_within_a_pair():
    state = "robot: hospital supply cart (hospital)\nload: boxed monitor, upright, centred\npacking: its contents fill it uniformly"
    record = {"id": "b1-tilt-1", "meta": {"pair": {"of": "b1-tilt-1p"}}, "request": {"state": state, "questions": {}}, "labels": {}}
    partner = {"id": "b1-tilt-1p", "meta": {"pair": {"of": "b1-tilt-1"}}, "request": {"state": state.replace("monitor", "printer"), "questions": {}}, "labels": {}}
    sheets = {}
    for length, target in iface.LENGTHS.items():
        sheet = iface.wrap_record(record, length=length)["request"]["state"]
        assert target - 150 <= len(sheet) <= target
        assert iface.sheet_problems(sheet, state) == []
        twin = iface.wrap_record(partner, length=length)["request"]["state"]
        assert twin == sheet.replace(state, partner["request"]["state"])  # the pair differs only in the fact
        sheets[length] = sheet
    tracks = {k: [l for l in v.splitlines() if l.startswith("- track ")] for k, v in sheets.items()}
    assert tracks["2k"] == tracks["4k"][: len(tracks["2k"])] and len(tracks["8k"]) > len(tracks["4k"]) > len(tracks["2k"]) >= 1
    assert iface.wrap_record(record, length="4k") == iface.wrap_record(record, length="4k")
    # partners whose states differ in length still share the filler exactly when wrapped for the pair's longest state
    longer = dict(partner, request={"state": state + " Extra words make this partner longer.", "questions": {}})
    reference = iface.pair_reference_lengths([record, longer])
    assert reference == {"b1-tilt-1": len(longer["request"]["state"]), "b1-tilt-1p": len(longer["request"]["state"])}
    for length, target in iface.LENGTHS.items():
        a = iface.wrap_record(record, length=length, reference_length=reference["b1-tilt-1"])["request"]["state"]
        b = iface.wrap_record(longer, length=length, reference_length=reference["b1-tilt-1p"])["request"]["state"]
        assert a.replace(state, "<S>") == b.replace(longer["request"]["state"], "<S>") and len(b) <= target
    sections = iface.sheet_sections(sheets["8k"])
    assert list(sections) == ["ROLE", "ROBOT STATUS", "CAMERA TRACKS", "CURRENT SITUATION", "EARLIER OBSERVATIONS", "LOCAL PLAN"]
    assert sections["CURRENT SITUATION"] == state and "Planner recommendation: none" in sections["LOCAL PLAN"]


def test_sheet_problems_catch_a_filler_that_talks_about_the_task():
    state = "A glass stands on the tray."
    sheet = iface.wrap_state(state, key="k", target=2000)
    assert iface.sheet_problems(sheet, state) == []
    bad = sheet.replace("- Plan source:", "- The tray is wet. Plan source:")
    assert any("filler mentions" in p for p in iface.sheet_problems(bad, state))
    assert any("exactly once" in p for p in iface.sheet_problems(sheet + "\n" + state, state))


def test_the_wrapped_records_pass_the_standard_checks_unchanged():
    record = physics_record("r1", partner="r1p")
    for length in iface.LENGTHS:
        wrapped = iface.wrap_record(record, length=length)
        assert validate_record(wrapped) == []
        assert {k: v for k, v in wrapped.items() if k != "request"} == {k: v for k, v in record.items() if k != "request"}
        assert wrapped["request"]["questions"] == record["request"]["questions"]


# -- M2: the rule ----------------------------------------------------------------------------------------------------------------------------


def _d(value, low, high):
    return {"value": value, "low": low, "high": high}


def test_m2_rule_gap_no_gap_and_mixed():
    calm = {L: {"direction": _d(-0.01, -0.03, 0.01), "comparison": _d(0.02, -0.01, 0.05), "DC": _d(0.0, -0.02, 0.02)} for L in ("2k", "4k", "8k")}
    hurt = copy.deepcopy(calm)
    hurt["8k"]["direction"] = _d(-0.08, -0.12, -0.04)
    noisy = copy.deepcopy(calm)
    noisy["4k"]["DC"] = _d(0.06, -0.01, 0.12)  # big but the interval includes 0: not a trigger, not within 0.03
    dc_up = copy.deepcopy(calm)
    dc_up["2k"]["DC"] = _d(0.05, 0.01, 0.09)
    assert iface.m2_outcome({"a": calm, "b": calm})["outcome"] == "NO GAP"
    assert iface.m2_outcome({"a": calm, "b": hurt})["outcome"] == "LONG-STATE GAP"
    assert iface.m2_outcome({"a": calm, "b": hurt})["per_model"]["b"]["triggers"] == ["8k:direction"]
    assert iface.m2_outcome({"a": calm, "b": noisy})["outcome"] == "MIXED"
    assert iface.m2_outcome({"a": dc_up})["per_model"]["a"]["triggers"] == ["2k:DC"]
    better = copy.deepcopy(calm)
    better["8k"]["comparison"] = _d(0.08, 0.04, 0.12)  # an improvement is not harm, but it is not within 0.03 either
    assert iface.m2_outcome({"a": better})["outcome"] == "MIXED"


# -- verdict helpers -------------------------------------------------------------------------------------------------------------------------


def _prediction(record: dict, *, flip: bool = False) -> dict:
    questions = {}
    for name, question in record["request"]["questions"].items():
        keys = list(question["criteria"])
        listings = []
        for shift in range(len(keys)):
            order = keys[shift:] + keys[:shift]
            best = max(record["labels"][name]["probabilities"], key=record["labels"][name]["probabilities"].get)
            target = keys[0] if (flip and best != keys[0]) else (keys[1] if flip else best)
            probs = {k: (0.8 if k == target else 0.2 / (len(keys) - 1)) for k in keys}
            listings.append({"options": order, "answer": {"probabilities": probs}})
        questions[name] = {"type": "choice", "n_listings": len(listings), "listings": listings}
    return {"id": record["id"], "questions": questions}


def test_choice_accuracy_and_the_pairing_check():
    verdict = _script("iface_verdict")
    records = [physics_record("r1", partner="r1p"), physics_record("r1p", partner="r1")]
    clusters = {r["id"]: "g" for r in records}
    right = [_prediction(r) for r in records]
    wrong = [_prediction(r, flip=True) for r in records]
    assert {k: v for k, v in verdict.choice_items("B1", records, right, clusters).items()} == {
        ("B1", "r1", "effect"): ("g", 1.0), ("B1", "r1", "outcome"): ("g", 1.0), ("B1", "r1p", "effect"): ("g", 1.0), ("B1", "r1p", "outcome"): ("g", 1.0)}
    assert all(v == 0.0 for _, v in verdict.choice_items("B1", records, wrong, clusters).values())
    assert verdict.pairing_problems(right, wrong, {"r1", "r1p"}) == []
    shuffled = copy.deepcopy(wrong)
    shuffled[0]["questions"]["effect"]["listings"][0]["options"] = list(reversed(shuffled[0]["questions"]["effect"]["listings"][0]["options"]))
    assert any("option orders differ" in p for p in verdict.pairing_problems(right, shuffled, {"r1", "r1p"}))
    assert any("missing in wrapped" in p for p in verdict.pairing_problems(right, wrong[:1], {"r1", "r1p"}))


def test_the_comparison_mapping_reproduces_infamily_on_b1():
    path = REPO / "artifacts" / "b1" / "b1-infamily.jsonl"
    if not path.is_file():
        pytest.skip("B1-infamily not on this machine")
    from robo_jev.bench import infamily, system1
    from robo_jev.decisions.records import read_records

    records = read_records(path)[:240]
    verdict = _script("iface_verdict")
    oracle = system1.oracle_predictions(records)
    mapping = verdict.V6().comparison_map(records)
    ours = verdict.V6().comparison_items("B1", records, oracle, mapping, clusters={r["id"]: infamily.group_of(r) for r in records})
    theirs = infamily.comparison_items("B1", records, oracle)
    assert ours == theirs and ours


def test_reproduction_compares_the_fresh_run_with_listing_zero():
    verdict = _script("iface_verdict")
    records = [physics_record("r1", partner="r1p"), physics_record("r1p", partner="r1")]
    existing = [_prediction(r) for r in records]  # three listings per question (cyclic shifts)
    fresh = copy.deepcopy(existing)
    for p in fresh:
        for q in p["questions"].values():
            q["listings"] = q["listings"][:1]
            q["n_listings"] = 1
            q["listings"][0]["code_probs"] = [0.8, 0.1, 0.1]
    old = copy.deepcopy(existing)
    for p in old:
        for q in p["questions"].values():
            q["listings"][0]["code_probs"] = [0.8, 0.1, 0.1]
    same = verdict.reproduction(old, fresh, {"r1", "r1p"})
    assert same == {"questions": 4, "order_mismatches": 0, "max_abs_prob_diff": 0.0, "mean_abs_prob_diff": 0.0, "same_argmax_share": 1.0}
    old[0]["questions"]["effect"]["listings"][0]["code_probs"] = [0.1, 0.8, 0.1]
    moved = verdict.reproduction(old, fresh, {"r1", "r1p"})
    assert moved["max_abs_prob_diff"] == pytest.approx(0.7) and moved["same_argmax_share"] == 0.75
    old[1]["questions"]["outcome"]["listings"][0]["options"] = list(reversed(old[1]["questions"]["outcome"]["listings"][0]["options"]))
    assert verdict.reproduction(old, fresh, {"r1", "r1p"})["order_mismatches"] == 1


def test_m2_deltas_are_wrapped_minus_short():
    verdict = _script("iface_verdict")
    keys = [("B1", f"r{i}", "q") for i in range(40)]
    short = {"direction": {k: (f"g{i % 10}", 1.0) for i, k in enumerate(keys)}, "comparison": {k: (f"g{i % 10}", 1.0) for i, k in enumerate(keys)},
             "choice_accuracy": {k: (f"g{i % 10}", 1.0) for i, k in enumerate(keys)},
             "band_balanced": {("low", *k): (f"g{i % 10}", 1.0) for i, k in enumerate(keys)},
             "calls": {k: (f"g{i % 10}", "unclear", (0.0, 1.0, 0.0)) for i, k in enumerate(keys)}}
    wrapped = copy.deepcopy(short)
    for i, k in enumerate(keys[:10]):
        wrapped["direction"][k] = (f"g{i % 10}", 0.0)
        wrapped["calls"][k] = (f"g{i % 10}", "unclear", (1.0, 0.0, 0.0))
    d = verdict.m2_delta(wrapped, short, n_boot=200, seed=0)
    assert d["direction"]["value"] == pytest.approx(-0.25) and d["direction"]["high"] < 0
    assert d["DC"]["value"] == pytest.approx(0.25) and d["DC"]["low"] > 0
    assert d["comparison"]["value"] == 0.0 and d["choice_accuracy"]["value"] == 0.0 and d["band_balanced"]["value"] == 0.0
