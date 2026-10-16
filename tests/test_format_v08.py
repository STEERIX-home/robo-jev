"""Task v0.8a — the FORMAT package: `robo_jev.simgen.format_v08` and `scripts/assemble_v08.py`.

Pure parts: N log-uniform over 9..255; the random-code share and draw; the physics key index (v0.7's filters, each key once, fewer when the
families run out); the general draw (no word shared with the item); re-keying. The transform on a small synthetic input made by v0.7's own
assembly (engine with a one-fact pair, MuJoCo pairs, general, d1): ids, order, labels and untouched lines kept, dev records never transformed,
v0.7's sheets and facts kept, expanded questions re-expanded with distractors at 0, general items re-keyed with their labels, random codes on
about a quarter of all choice questions, the 255-option mark exactly when a question has more than 8 options; :func:`record_problems`
catches tampering; the time and budget arithmetic; the protected places. The real input's files are checked when they exist.
"""

from __future__ import annotations

import copy
import functools
import importlib.util
import json
import math
import random
import sys
from collections import Counter
from pathlib import Path

import pytest
from helpers import REPO

from robo_jev.decisions.codes import CODE_POOL
from robo_jev.decisions.examples import is_dev
from robo_jev.decisions.records import OPTION_RANGE_MARK_255, shuffle_key, validate_record
from robo_jev.simgen import format_v08 as F
from robo_jev.simgen import shape as S

CONFIG = {"dev_fraction": 0.02, "dev_salt": "v0d-dev", "questions_per_request": 2}


@functools.lru_cache(maxsize=None)
def _module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _v08():
    return _module("assemble_v08_under_test", REPO / "scripts" / "assemble_v08.py")


def _v07():
    return _module("assemble_v07_for_v08", REPO / "scripts" / "assemble_v07.py")


def _v07_tests():
    return _module("test_assemble_v07_helpers", REPO / "tests" / "test_assemble_v07.py")


# -- pure parts --------------------------------------------------------------------------------------------------------------------------


def test_n_is_log_uniform_over_9_to_255():
    draws = [F.draw_n(f"k{i}", "q") for i in range(20000)]
    assert min(draws) == 9 and max(draws) == 255
    share = Counter(F.bin_of(n) for n in draws)
    total = math.log(256 / 9)
    for (low, high), name in (((9, 26), "9-26"), ((27, 60), "27-60"), ((61, 255), "61-255")):
        assert share[name] / len(draws) == pytest.approx(math.log((high + 1) / low) / total, abs=0.015), name
    assert F.draw_n("k1", "q") == F.draw_n("k1", "q") and len({F.draw_n("k1", f"q{i}") for i in range(20)}) > 5, "per key and question"


def test_about_a_quarter_of_choice_questions_get_random_codes_drawn_from_the_pool():
    picked = sum(F.codes_selected(f"k{i}", "q") for i in range(20000))
    assert picked / 20000 == pytest.approx(F.CODES_SHARE, abs=0.01)
    codes = F.draw_codes("k1", "q", 40)
    assert len(codes) == len(set(codes)) == 40 and set(codes) <= set(CODE_POOL) and codes == F.draw_codes("k1", "q", 40)
    assert codes != sorted(codes, key=CODE_POOL.index), "the drawn order, not the pool order"
    seen = Counter(c for i in range(3000) for c in F.draw_codes(f"k{i}", "q", 8))
    assert set(seen) == set(CODE_POOL), "every pool code is drawn"
    with pytest.raises(ValueError):
        F.draw_codes("k", "q", 256)


def _pool() -> S.DistractorPool:
    pool = S.DistractorPool()
    for i in range(60):
        pool.add("out", f"ends_{i}", f"zqout{i} comes to rest", f"set:scene{i % 7}")
        pool.add("rule", f"r{i % 3}", f"zqrule{i} limit holds", f"set:scene{i % 5}")
        pool.add("band", ["low", "medium", "high"][i % 3], "a level", f"set:scene{i % 3}")
        pool.add("est_obj", f"s{i}", f"zqobj{i} sits level", f"set:scene{i % 4}")
    return pool


def test_the_key_index_draws_each_allowed_key_once_with_v07s_filters():
    index = F.KeyIndex(_pool())
    keys = index.keys("act")
    drawn = index.draw("act", count=500, own_keys={"ends_3", "go"}, own_norms=set(), words={"zqout5"}, scene="set:scene0", rnd=random.Random(1))
    assert len({e.key for e in drawn}) == len(drawn) and "ends_3" not in {e.key for e in drawn}
    assert all(e.scene != "set:scene0" and not (e.words & {"zqout5"}) and not e.action for e in drawn)
    assert {e.family for e in drawn} <= set(S.ALLOWED["act"])
    assert len(drawn) < len(keys), "fewer than asked when the families run out of keys"
    few = index.draw("act", count=5, own_keys=set(), own_norms=set(), words=set(), scene="x", rnd=random.Random(2))
    assert len(few) == 5 and len({e.family for e in few}) >= 3, "a seeded round-robin over the families"
    again = index.draw("act", count=5, own_keys=set(), own_norms=set(), words=set(), scene="x", rnd=random.Random(2))
    assert [e.key for e in few] == [e.key for e in again]


def test_general_distractors_share_no_word_with_the_item_but_may_with_each_other():
    record = {"id": "qasc-train-1", "meta": {"origin_group": "g1", "source": {"set": "qasc", "row": 1}},
              "request": {"state": "facts: plants need light", "questions": {"q_general": {"type": "choice", "instructions": "?",
                                                                                          "criteria": {"o1": "green leafy plants", "o2": "rocks"}}}}}
    pool = [(f"g{i}", "hellaswag", f"a dancer spins round number {chr(97 + i % 26)}{chr(97 + i // 26 % 26)}") for i in range(400)]
    pool += [(f"h{i}", "qasc", f"another qasc answer {i}") for i in range(50)] + [("g9999", "hellaswag", "plants grow tall")]
    texts = F.general_distractors(record, pool, count=200, rnd=random.Random(3))
    assert texts is not None and len(texts) == len(set(texts)) == 200
    assert all("plants" not in t and "qasc" not in t for t in texts), "no word of the state or own options; never the item's own source"
    assert F.general_distractors(record, pool[:5], count=50, rnd=random.Random(3)) is None


def test_rekeying_gives_all_options_o1_to_on_and_moves_the_labels():
    criteria, mapping = F.rekey([("o1", "right"), ("o2", "wrong")], ["d1", "d2", "d3"], random.Random(4))
    assert sorted(criteria, key=lambda k: int(k[1:])) == [f"o{i}" for i in range(1, 6)]
    assert criteria[mapping["o1"]] == "right" and criteria[mapping["o2"]] == "wrong"
    assert list(criteria.values())[:2] == ["right", "wrong"], "own options first, in their order"


# -- the transform on a synthetic input ----------------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("v08")
    v06 = _v07_tests()._write(tmp)
    v07 = tmp / "train-v07.jsonl"
    _v07().transform_file(v06, v07, config=CONFIG)
    out = tmp / "train-v08.jsonl"
    report = _v08().transform_file(v06, v07, out, config=CONFIG)
    return v06, v07, out, report


def _read(path: Path) -> list[dict]:
    return [json.loads(raw) for raw in path.read_bytes().splitlines()]


def test_ids_order_labels_and_untouched_lines_are_kept(built):
    v06, v07, out, report = built
    old_lines, new_lines = v07.read_bytes().splitlines(), out.read_bytes().splitlines()
    assert len(old_lines) == len(new_lines)
    olds, news, sources = _read(v07), _read(out), {r["id"]: r for r in _read(v06)}
    assert [r["id"] for r in olds] == [r["id"] for r in news]
    changed = 0
    for a, b, raw_a, raw_b in zip(olds, news, old_lines, new_lines):
        if "format" not in b.get("meta", {}):
            assert raw_a == raw_b, f"{a['id']}: an untouched record keeps its line byte for byte"
            continue
        changed += 1
        assert validate_record(b) == [] and b["request"]["state"] == a["request"]["state"]
        assert {k: v for k, v in b["meta"].items() if k not in ("format", "codes", "option_range")} == {k: v for k, v in a["meta"].items() if k != "option_range"}
        expanded = b["meta"]["format"]["expanded"]
        for name, label in a["labels"].items():
            if name not in expanded:
                assert b["labels"][name] == label and b["request"]["questions"][name] == a["request"]["questions"][name]
                continue
            own = sources[a["id"]]["labels"][name]["probabilities"]
            got = b["labels"][name]["probabilities"]
            assert sorted(v for v in got.values() if v) == sorted(v for v in own.values() if v), "the label's mass is the original's"
            assert sum(1 for v in got.values() if v == 0.0) == len(got) - sum(1 for v in own.values() if v), "every distractor at 0"
            assert 9 <= len(b["request"]["questions"][name]["criteria"]) <= 255
        wide = any(len(q.get("criteria") or {}) > 8 for q in b["request"]["questions"].values() if q["type"] == "choice")
        assert (b["meta"].get("option_range") == OPTION_RANGE_MARK_255) == wide and ("option_range" in b["meta"]) == wide
    assert changed > 50 and report["digests"]["id_order"]["identical"]


def test_dev_records_are_never_transformed_and_v07s_shapes_stay(built):
    _, v07, out, _ = built
    for a, b in zip(_read(v07), _read(out)):
        if is_dev(b, fraction=CONFIG["dev_fraction"], salt=CONFIG["dev_salt"]):
            assert a == b and "format" not in b.get("meta", {})
        assert (b.get("meta") or {}).get("shape") == (a.get("meta") or {}).get("shape")


def test_expanded_questions_are_v07s_questions_with_a_new_n_and_general_items_are_rekeyed(built):
    v06, v07, out, report = built
    sources = {r["id"]: r for r in _read(v06)}
    expanded = rekeyed = 0
    for a, b in zip(_read(v07), _read(out)):
        names07 = set(((a.get("meta") or {}).get("shape") or {}).get("expanded") or {})
        fmt = (b.get("meta") or {}).get("format")
        if not names07 or not fmt:
            continue
        assert set(fmt["expanded"]) == names07, "exactly v0.7's expanded questions"
        for name, e in fmt["expanded"].items():
            expanded += 1
            assert e["to"] == len(b["request"]["questions"][name]["criteria"]) and e["to"] <= e["drawn"] and e["short"] == (e["to"] < e["drawn"])
            assert e["drawn"] == F.draw_n(shuffle_key(a), name)
            if e["rekeyed"]:
                rekeyed += 1
                keys = sorted(b["request"]["questions"][name]["criteria"], key=lambda k: int(k[1:]))
                assert keys == [f"o{i}" for i in range(1, e["to"] + 1)]
                gold = max(sources[a["id"]]["labels"][name]["probabilities"], key=lambda k: sources[a["id"]]["labels"][name]["probabilities"][k])
                text = sources[a["id"]]["request"]["questions"][name]["criteria"][gold]
                new_gold = max(b["labels"][name]["probabilities"], key=lambda k: b["labels"][name]["probabilities"][k])
                assert b["request"]["questions"][name]["criteria"][new_gold] == text, "the correct option keeps its text and its probability"
    assert expanded > 10 and report["many_options"]["n_by_bin"]  # (the synthetic general items are never expanded by v0.7 — see below)


def test_an_expanded_general_item_is_re_expanded_and_re_keyed_with_its_label():
    module = _v08()
    sources = ("hellaswag", "social_i_qa", "cosmos_qa", "qasc", "commonsense_qa")
    words = ["amber", "basil", "cedar", "delta", "ember", "fjord", "grove", "harbor", "indigo", "jasper", "kelp", "lagoon", "maple", "nectar"]
    general = []
    for i in range(300):
        source = sources[i % 5]
        def tag(n: int) -> str:
            return "".join(chr(97 + n // 26 ** k % 26) for k in range(3))

        criteria = {f"o{j + 1}": f"{words[(i + j) % 14]}x{tag(i * 4 + j)} zq{tag(i * 4 + j)} zr{tag(i * 4 + j)}" for j in range(4)}
        general.append({"schema": "simgen-request/0", "id": f"{source}-train-{i:05d}",
                        "meta": {"origin_group": f"public/{source}/train/{i:05d}", "split": "train", "source": {"set": source, "row": i}},
                        "request": {"state": f"text: zqstate{chr(97 + i % 26)}{chr(97 + i // 26 % 26)}", "questions": {"q_general": {"type": "choice", "instructions": "Pick one.", "criteria": criteria}}},
                        "labels": {"q_general": {"type": "choice", "kind": "probe", "probabilities": {"o1": 0.0, "o2": 1.0, "o3": 0.0, "o4": 0.0}}}})
    plan = module.Plan(CONFIG)
    for record in general:
        plan.scan(record)
    plan.finish()
    v06 = next(r for r in general if not is_dev(r, fraction=0.02, salt="v0d-dev"))
    v07 = copy.deepcopy(v06)
    v07["request"]["questions"]["q_general"]["criteria"].update({"o9": "zqold one two three", "o10": "zqold four five six"})
    v07["labels"]["q_general"]["probabilities"].update({"o9": 0.0, "o10": 0.0})
    v07["meta"]["shape"] = {"version": "v07-shape-1", "kind": None, "expanded": {"q_general": {"family": "general", "from": 4, "to": 6}}}
    v07["meta"]["option_range"] = {"version": "options-26-1", "max": 26}
    new, event = module.transform_record(v06, v07, plan)
    assert module.record_problems(v06, v07, new, event) == []
    entry = event["expanded"]["q_general"]
    criteria = new["request"]["questions"]["q_general"]["criteria"]
    assert entry["rekeyed"] and sorted(criteria, key=lambda k: int(k[1:])) == [f"o{i}" for i in range(1, entry["to"] + 1)]
    gold = max(new["labels"]["q_general"]["probabilities"], key=lambda k: new["labels"]["q_general"]["probabilities"][k])
    assert criteria[gold] == v06["request"]["questions"]["q_general"]["criteria"]["o2"] and new["labels"]["q_general"]["probabilities"][gold] == 1.0
    assert "zqold one two three" not in criteria.values(), "v0.7's distractors are replaced"
    assert new["meta"]["option_range"] == OPTION_RANGE_MARK_255 and new["meta"]["shape"] == v07["meta"]["shape"]
    assert validate_record(new) == []


def test_random_codes_cover_about_a_quarter_of_all_choice_questions(built):
    _, _, out, report = built
    choice = coded = 0
    for record in _read(out):
        codes = (record.get("meta") or {}).get("codes") or {}
        for name, question in record["request"]["questions"].items():
            if question["type"] != "choice" or is_dev(record, fraction=0.02, salt="v0d-dev"):
                continue
            choice += 1
            if name in codes:
                coded += 1
                assert len(codes[name]) == len(question["criteria"]) and set(codes[name]) <= set(CODE_POOL)
                assert codes[name] == F.draw_codes(shuffle_key(record), name, len(question["criteria"]))
    assert 0.15 < coded / choice < 0.35 and report["random_codes"]["questions"] == coded


def test_a_pair_shares_its_expansion_and_codes(built):
    _, _, out, _ = built
    records = {r["id"]: r for r in _read(out)}
    pairs = 0
    for rid, record in records.items():
        partner = ((record.get("meta") or {}).get("pair") or {}).get("of")
        if not partner or rid > partner or partner not in records or "format" not in record.get("meta", {}):
            continue
        a, b = record["meta"]["format"], records[partner]["meta"].get("format") or {}
        assert set(a["expanded"]) == set(b.get("expanded") or {}) and a["codes"] == b.get("codes")
        for name in a["expanded"]:
            assert list(record["request"]["questions"][name]["criteria"])[3:] == list(records[partner]["request"]["questions"][name]["criteria"])[3:]
            pairs += 1
    assert pairs > 0


def test_record_problems_catch_tampering(built):
    v06, v07, out, _ = built
    module = _v08()
    plan = module.Plan(CONFIG)
    sources = _read(v06)
    for record in sources:
        plan.scan(record)
    plan.finish()
    by_id = {r["id"]: r for r in sources}
    target = next(r for r in _read(v07) if ((r.get("meta") or {}).get("shape") or {}).get("expanded") and r["id"].startswith("v06a1")
                  and not is_dev(r, fraction=0.02, salt="v0d-dev"))
    new, event = module.transform_record(by_id[target["id"]], target, plan)
    assert module.record_problems(by_id[target["id"]], target, new, event) == []
    name = next(iter(event["expanded"]))
    extra = [k for k in new["request"]["questions"][name]["criteria"] if k not in by_id[target["id"]]["request"]["questions"][name]["criteria"]][0]
    own = next(iter(by_id[target["id"]]["labels"][name]["probabilities"]))
    for mutate in (lambda r: r["labels"]["key"].update(p_yes=0.99),
                   lambda r: r["labels"][name]["probabilities"].update({extra: 0.5}),
                   lambda r: r["labels"][name]["probabilities"].update({own: 0.123}),
                   lambda r: r["meta"].update(set="v06-a9"),
                   lambda r: r["request"].update(state=r["request"]["state"] + " More."),
                   lambda r: r["request"]["questions"][name].update(instructions="Other?"),
                   lambda r: r["meta"].pop("option_range"),
                   lambda r: r["meta"].update(codes={name: ["A"]}),
                   lambda r: r.update(id="other")):
        bad = copy.deepcopy(new)
        mutate(bad)
        assert module.record_problems(by_id[target["id"]], target, bad, event), "a tampered record is caught"


def test_the_time_and_budget_scale_v07s_measured_pods():
    module = _v08()
    m = module.V07_MEASURED
    sim = {"0": {"tokens": m["0"]["tokens"], "padded_tokens": m["0"]["padded_tokens"]},
           "1": {"tokens": m["1"]["tokens"], "padded_tokens": 2 * m["1"]["padded_tokens"]}}
    out = module.time_and_budget(sim, input_bytes=837650913)
    assert out["0"]["step_minutes"] == pytest.approx(m["0"]["step_seconds"] / 60)
    assert out["0"]["train_minutes"] == pytest.approx(m["0"]["train_step_seconds"] / 60)
    assert out["0"]["upload_minutes"] == pytest.approx(m["0"]["upload_seconds"] / 60)
    assert out["0"]["pod_hours"] == pytest.approx((m["0"]["pod_seconds"] + 0.1 * m["0"]["train_step_seconds"]) / 3600)
    assert out["1"]["step_minutes"] == pytest.approx(2 * m["1"]["step_seconds"] / 60)
    assert out["0"]["pod_usd"] == pytest.approx(out["0"]["pod_hours"] * module.HOURLY_USD)


@pytest.mark.parametrize("place", ["artifacts/simgen/v07/x.jsonl", "artifacts/simgen/v06/x.jsonl", "artifacts/v07/gates/x", "artifacts/v08/x",
                                   "artifacts/reports/v08/x.json", "artifacts/t9/x", "artifacts/iface/x"])
def test_outputs_never_go_into_protected_places(place):
    with pytest.raises(SystemExit):
        _v08().refuse_write(REPO / place)
    _v08().refuse_write(REPO / "artifacts" / "simgen" / "v08" / "x.jsonl")


def test_the_real_v08_input_matches_its_manifest_and_passed_every_check():
    summary = REPO / "artifacts" / "simgen" / "v08" / "train-v08.assembly.json"
    if not summary.is_file():
        pytest.skip("artifacts/simgen/v08 not built here (scripts/assemble_v08.py all)")
    data = json.loads(summary.read_text(encoding="utf-8"))
    manifest = json.loads((summary.parent / "train-v08.manifest.json").read_text(encoding="utf-8"))
    assert data["pass"] is True and data["overlap_hits"] == 0 and data["extra_hits"] == 0 and data["exposure_identical_to_v07"] is True
    assert data["text_subset"] is True and manifest["sha256"] == data["files"]["train-v08.jsonl"]["sha256"]
    shares = data["shares"]
    assert 0.20 <= shares["random_codes_of_all_choice_questions"] <= 0.30
    assert abs(shares["expanded_of_general_records"] - 0.077) < 0.01
    assert data["n_by_bin"]["61-255"] >= 2000
