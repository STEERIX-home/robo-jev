"""scripts/assemble_v07.py — Task v0.7a: the v0.7 training input, transformed from v0.6's, labels unchanged.

On a small synthetic input of the four parts (engine with a one-fact pair, MuJoCo, general, d1): the transform keeps the ids, the order,
every label and every untouched line byte for byte (the d1 part whole), never transforms a dev record, proves each changed record against
its original, and writes ``meta.option_range`` exactly when a question has more than 8 options; :func:`record_problems` catches tampering;
the time and budget arithmetic; the refusal to write into the protected places. The real input's files are checked when they exist.
"""

from __future__ import annotations

import copy
import functools
import importlib.util
import json
import sys
from pathlib import Path

import pytest
from helpers import REPO

from robo_jev.decisions.examples import is_dev
from robo_jev.decisions.records import OPTION_RANGE_MARK, shuffle_key, validate_record
from robo_jev.simgen import shape as S

CONFIG = {"dev_fraction": 0.02, "dev_salt": "v0d-dev", "questions_per_request": 2}
STATE = ("Log, warehouse floor. Unit: a low-deck inventory robot. Cargo: one archive box of files standing upright at the centre of the "
         "deck. Deck covering: diamond-tread steel. The unit is heading onto a ramp: its deck pitches front up to 21.5° within 2.3 s.")
ENGINE_STATE = ("A busy city sidewalk at lunchtime. You are a food-delivery robot.\n- Braking: gentle 1.1 m/s², firm 2.0 m/s².\n"
                "- A man is approaching your path from the right.\n- The man: 4.8 m from your path, closing at 1.05 m/s.")


@functools.lru_cache(maxsize=None)
def _script():
    spec = importlib.util.spec_from_file_location("assemble_v07_under_test", REPO / "scripts" / "assemble_v07.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _tok(n: int) -> str:
    """A made-up word per number (letters only): the synthetic scenes' options share no content word with each other."""
    out = ""
    n += 26
    while n:
        n, d = divmod(n, 26)
        out += chr(97 + d)
    return "zq" + out


def _mujoco(i: int, *, pair: str | None = None, role: str = "base") -> dict:
    rid = f"v06a1-mast-{i:06d}" + ("p" if role == "partner" else "")
    scene = (i // 2) % 30
    t = _tok(scene)
    questions = {
        "key": {"type": "noul", "instructions": "Skid past the mark or tip up?"},
        "cf": {"type": "score", "instructions": "Risk score for a harder stop.", "criteria": ["0 = clean", "1 = sure"]},
        "shift": {"type": "choice", "instructions": "Harder stop: risk?", "criteria": {"higher": "goes up", "lower": "goes down", "unchanged": "about the same"}},
        "versus": {"type": "choice", "instructions": "More risk: commanded or lowered lift?", "criteria": {"current_plan": "as commanded", "drop_lift_first": "lift at 0.40 m first", "no_difference": "about equal"}},
        "mode": {"type": "choice", "instructions": "Operator: choose the stop mode.", "criteria": {f"{t}_a": f"run routine {t}a", f"{t}_b": f"run routine {t}b", f"{t}_c": f"run routine {t}c"}},
        "result": {"type": "choice", "instructions": "Expected stop result?", "criteria": {f"{t}_x": f"ends {t}x", f"{t}_y": f"ends {t}y", f"{t}_z": f"ends {t}z"}},
    }
    labels = {"key": {"type": "noul", "p_yes": (i % 10) / 10}, "cf": {"type": "score", "value": ((i * 3) % 10) / 10},
              "shift": {"type": "choice", "kind": "state", "probabilities": {"higher": 0.9, "lower": 0.0, "unchanged": 0.1}},
              "versus": {"type": "choice", "kind": "state", "probabilities": {"current_plan": 0.0, "drop_lift_first": 1.0, "no_difference": 0.0}},
              "mode": {"type": "choice", "kind": "action", "probabilities": {f"{t}_a": 0.05, f"{t}_b": 0.9, f"{t}_c": 0.05},
                       "expected_cost": {f"{t}_a": 0.4, f"{t}_b": 0.35, f"{t}_c": 0.9}},
              "result": {"type": "choice", "kind": "state", "probabilities": {f"{t}_x": 0.3, f"{t}_y": 0.7, f"{t}_z": 0.0}}}
    meta = {"set": "v06-a1", "split": "train", "scene": f"mast{scene}", "language": "en", "key_question": "key",
            "question_kind": {"key": "key", "cf": "score_cf", "shift": "direction", "versus": "compare", "mode": "action", "result": "outcome"}}
    if pair:
        meta["pair"] = {"of": pair, "role": role}
    state = STATE if role == "base" else STATE.replace("diamond-tread steel", "smooth PVC")
    return {"schema": "simgen-request/0", "id": rid, "generator": {"name": "a1"}, "meta": meta,
            "request": {"state": state.replace("21.5", f"{10 + i % 15}.5"), "questions": questions}, "labels": labels}


def _engine(i: int) -> dict:
    t = _tok(100 + i % 40)
    questions = {"action": {"type": "choice", "instructions": "Pick the best thing to do.",
                            "criteria": {"continue": f"Keep going {t}.", "stop_hard": f"Emergency halt {t}.", "slow": f"Slow down gently {t}."}},
                 "clears": {"type": "noul", "instructions": "Do you clear the crossing first?"},
                 "s1_band_1": {"type": "choice", "instructions": "Do you clear the crossing first? Pick one of three levels.",
                               "criteria": {"low": "less than 1 in 3", "medium": "between 1 in 3 and 2 in 3", "high": "more than 2 in 3"}}}
    labels = {"action": {"type": "choice", "kind": "action", "probabilities": {"continue": 0.0, "stop_hard": 0.2, "slow": 0.8}},
              "clears": {"type": "noul", "p_yes": (i % 7) / 7}, "s1_band_1": {"type": "choice", "kind": "state", "probabilities": {"low": 1.0, "medium": 0.0, "high": 0.0}}}
    meta = {"scene": f"p1_crossing_{i % 40}", "language": "en", "basis": {"action": {"kind": "action"}}, "key_question": "clears", "pair": None,
            "question_pick": {"version": "qpick-1", "secondary": ["s1_band_1"]}}
    return {"schema": "simgen-request/0", "id": f"v0b-{i:06d}", "generator": {"seed": 5}, "meta": meta,
            "request": {"state": ENGINE_STATE, "questions": questions}, "labels": labels}


def _general(i: int) -> dict:
    sources = ("hellaswag", "social_i_qa", "cosmos_qa", "qasc", "commonsense_qa")
    source = sources[i % 5]
    gold = f"walks the {['dog', 'cat', 'goat', 'horse', 'pony'][i % 5]} along the beach number {'abcdefghij'[i % 10]}"
    criteria = {"o1": gold, "o2": f"paints a {['red', 'blue'][i % 2]} fence with brushes", "o3": f"reads a long novel aloud {'xyz'[i % 3]}"}
    return {"schema": "simgen-request/0", "id": f"{source}-train-{i:05d}",
            "meta": {"origin_group": f"public/{source}/train/{i:05d}", "split": "train", "source": {"set": source, "row": i}},
            "request": {"state": f'text: "A person is outside on day {i}."', "questions": {"q_general": {"type": "choice", "instructions": "Pick one.", "criteria": criteria}}},
            "labels": {"q_general": {"type": "choice", "kind": "probe", "probabilities": {"o1": 1.0, "o2": 0.0, "o3": 0.0}}}}


def _d1(i: int) -> dict:
    return {"schema": "simgen-request/0", "id": f"spatial-{i:04d}-0", "meta": {"split": "train", "origin_group": f"spatial/x/{i:04d}"},
            "request": {"state": "objects: o1 at x=1", "questions": {"q": {"type": "choice", "instructions": "Nearest?", "criteria": {"o1": "o1", "o2": "o2"}}}},
            "labels": {"q": {"type": "choice", "probabilities": {"o1": 1.0, "o2": 0.0}}}}


def _write(tmp_path: Path) -> Path:
    rows = []
    for i in range(240):
        rows.append(_engine(i))
    for i in range(0, 600, 2):
        rows += [_mujoco(i, pair=f"v06a1-mast-{i:06d}p"), _mujoco(i, pair=f"v06a1-mast-{i:06d}", role="partner")]
    rows += [_d1(i) for i in range(10)] + [_general(i) for i in range(200)]
    path = tmp_path / "train-v06.jsonl"
    with path.open("w", encoding="utf-8") as handle:
        for r in rows:
            separators = (",", ":") if r["id"].startswith("v0b-") else None
            handle.write(json.dumps(r, ensure_ascii=False, separators=separators) + "\n")
    return path


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("v07")
    src = _write(tmp)
    out = tmp / "train-v07.jsonl"
    report = _script().transform_file(src, out, config=CONFIG)
    return src, out, report


def test_the_transform_keeps_ids_order_labels_and_untouched_lines(built):
    src, out, report = built
    old_lines, new_lines = src.read_bytes().splitlines(), out.read_bytes().splitlines()
    assert len(old_lines) == len(new_lines)
    old, new = [json.loads(x) for x in old_lines], [json.loads(x) for x in new_lines]
    assert [r["id"] for r in old] == [r["id"] for r in new]
    changed = 0
    for a, b, raw_a, raw_b in zip(old, new, old_lines, new_lines):
        if "shape" not in b.get("meta", {}):
            assert raw_a == raw_b, f"{a['id']}: an untouched record keeps its line byte for byte"
            continue
        changed += 1
        assert validate_record(b) == []
        for name, label in a["labels"].items():
            got = b["labels"][name]
            if "probabilities" in label:
                assert {k: got["probabilities"][k] for k in label["probabilities"]} == label["probabilities"]
                assert all(v == 0.0 for k, v in got["probabilities"].items() if k not in label["probabilities"])
            else:
                assert got == label
        assert ("option_range" in b["meta"]) == any(len(q.get("criteria") or {}) > 8 for q in b["request"]["questions"].values() if q["type"] == "choice")
        if "option_range" in b["meta"]:
            assert b["meta"]["option_range"] == OPTION_RANGE_MARK
        assert b["meta"].get("question_pick") == a["meta"].get("question_pick")
    assert changed > 50 and report["parts"]["d1"]["identical"]


def test_dev_records_and_the_d1_part_are_never_transformed(built):
    src, out, _ = built
    for raw in out.read_bytes().splitlines():
        record = json.loads(raw)
        if is_dev(record, fraction=CONFIG["dev_fraction"], salt=CONFIG["dev_salt"]) or record["id"].startswith("spatial-"):
            assert "shape" not in record.get("meta", {}), record["id"]


def test_a_pair_gets_one_shape_and_the_same_context(built):
    _, out, _ = built
    records = {json.loads(raw)["id"]: json.loads(raw) for raw in out.read_bytes().splitlines()}
    pairs = expanded = 0
    for rid, record in records.items():
        partner = (record["meta"].get("pair") or {}).get("of")
        if not partner or rid > partner or partner not in records:
            continue
        a, b = record["meta"].get("shape") or {}, records[partner]["meta"].get("shape") or {}
        assert a.get("kind") == b.get("kind") and set(a.get("expanded") or {}) == set(b.get("expanded") or {})
        if a.get("kind") == "sheet":
            pairs += 1
            assert (a["sheet"]["style"], a["sheet"]["tone"], a["sheet"]["sections"]) == (b["sheet"]["style"], b["sheet"]["tone"], b["sheet"]["sections"])
        for name in a.get("expanded") or {}:
            assert list(record["request"]["questions"][name]["criteria"])[3:] == list(records[partner]["request"]["questions"][name]["criteria"])[3:]
            expanded += 1
    assert pairs > 0 and expanded > 0


def test_the_report_counts_the_shapes_and_the_independence_tables(built):
    _, _, report = built
    assert set(report["shares"]) == {"sheet_of_physics_records", "facts_of_physics_records", "expanded_of_physics_choice_questions",
                                     "expanded_of_general_records"}
    assert 0.1 < report["shares"]["sheet_of_physics_records"] < 0.45
    assert "record:tone_x_key_band" in report["independence"]["tables"]
    assert report["many_options"]["n_histogram"] and all(9 <= int(n) <= 26 for n in report["many_options"]["n_histogram"])
    assert report["digests"]["id_order"]["identical"]


def test_record_problems_catch_tampering(built):
    src, out, _ = built
    module = _script()
    plan = module.Plan(CONFIG)
    olds = [json.loads(raw) for raw in src.read_bytes().splitlines()]
    for record in olds:
        plan.scan(record)
    sheet = next(r for r in olds if S.assign_kind(shuffle_key(r)) == "sheet" and r["id"].startswith("v06a1")
                 and not is_dev(r, fraction=0.02, salt="v0d-dev"))
    new, event = module.transform_record(sheet, plan)
    assert module.record_problems(sheet, new, event) == []
    for mutate in (lambda r: r["labels"]["key"].update(p_yes=0.99),
                   lambda r: r["meta"].update(set="v06-a9"),
                   lambda r: r["request"]["questions"]["key"].update(instructions="Other?"),
                   lambda r: r["request"].update(state=r["request"]["state"].replace("archive box", "crate")),
                   lambda r: r["meta"].update(option_range=dict(OPTION_RANGE_MARK)) if "option_range" not in r["meta"] else r["meta"].pop("option_range"),
                   lambda r: r.update(id="other")):
        bad = copy.deepcopy(new)
        mutate(bad)
        assert module.record_problems(sheet, bad, event), "a tampered record is caught"


def test_the_time_estimate_scales_v06s_measured_seconds_per_padded_token():
    module = _script()
    measured = module.V06_MEASURED
    sim = {"0": {"tokens": measured["0"]["tokens"], "padded_tokens": measured["0"]["padded_tokens"]},
           "1": {"tokens": measured["1"]["tokens"], "padded_tokens": 2 * measured["1"]["padded_tokens"]}, "dev": {"tokens": 100}}
    estimate = module.time_estimate(sim, {**sim, "1": sim["0"]})
    assert estimate["0"]["step_minutes"] == pytest.approx(measured["0"]["step_seconds"] / 60)
    assert estimate["1"]["step_minutes"] == pytest.approx(2 * measured["1"]["step_seconds"] / 60)
    assert estimate["0"]["train_minutes"] == pytest.approx(measured["0"]["seconds_total"] / 60)
    budget = module.budget_check(estimate)
    assert budget["0"]["pod_usd"] == pytest.approx(budget["0"]["pod_hours"] * module.HOURLY_USD)
    assert budget["0"]["wall_cap_hours"] >= budget["0"]["pod_hours"] + 0.5


@pytest.mark.parametrize("place", ["artifacts/simgen/v06/x.jsonl", "artifacts/iface/m2/x.jsonl", "artifacts/reports/v07/x.json", "artifacts/t9/x",
                                   "artifacts/v06/a4/x", "artifacts/simgen/v05/x"])
def test_outputs_never_go_into_protected_places(place):
    with pytest.raises(SystemExit):
        _script().refuse_write(REPO / place)
    _script().refuse_write(REPO / "artifacts" / "simgen" / "v07" / "x.jsonl")


def test_the_real_v07_input_matches_its_manifest_and_passed_every_check():
    summary = REPO / "artifacts" / "simgen" / "v07" / "train-v07.assembly.json"
    if not summary.is_file():
        pytest.skip("artifacts/simgen/v07 not built here (scripts/assemble_v07.py all)")
    data = json.loads(summary.read_text(encoding="utf-8"))
    manifest = json.loads((summary.parent / "train-v07.manifest.json").read_text(encoding="utf-8"))
    assert data["pass"] is True and data["overlap_hits"] == 0 and data["iface_unit_hits"] == 0 and data["exposure_identical_to_v06"] is True
    assert manifest["sha256"] == data["files"]["train-v07.jsonl"]["sha256"]
    assert data["parts_identical"]["d1"] is True
    shares = data["shares"]
    assert abs(shares["sheet_of_physics_records"] - 0.25) < 0.02 and abs(shares["facts_of_physics_records"] - 0.05) < 0.01
    assert abs(shares["expanded_of_physics_choice_questions"] - 0.08) < 0.01 and abs(shares["expanded_of_general_records"] - 0.08) < 0.01
