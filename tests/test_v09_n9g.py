"""Task v0.9b — N9g, the sealed numeric-rule gate of the v0.9 rule: the generator, its independent self-check (and that the self-check
catches broken records), the readings, the sealing (trainer, mix assembly, API paths refuse N9g records), the scorer's freeze, and the build
script's build / freeze / recheck-v09 / check paths on temporary files.
"""

from __future__ import annotations

import importlib.util
import json
import stat
import sys
from collections import Counter
from decimal import Decimal
from pathlib import Path

import pytest
from helpers import FIXTURES, REPO

from robo_jev.bench import n9g_v09 as N
from robo_jev.data.sealed import NEVER_SENT_SETS, SEALED_RECORD_FILES, SEALED_RECORD_PREFIXES, SEALED_RECORD_SETS, refuse_sealed_records, sealed_record_set
from robo_jev.decisions.records import read_records, validate_record

REQUESTS = FIXTURES / "decisions" / "requests.jsonl"
TRAIN_CONFIG = REPO / "configs" / "train" / "decisions-v0-lora-h100.yaml"


def _script(name: str):
    spec = importlib.util.spec_from_file_location(f"v09n9g_{Path(name).stem}", REPO / "scripts" / name)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def records() -> list[dict]:
    return N.n9g_records()


@pytest.fixture(scope="module")
def selfcheck():
    return _script("v09_n9g_selfcheck.py")


def _copy(record: dict) -> dict:
    return json.loads(json.dumps(record, ensure_ascii=False))


# --------------------------------------------------------------------------
# The design
# --------------------------------------------------------------------------


def test_the_gate_is_deterministic_and_sized_as_planned(records):
    again = N.n9g_records()
    assert [json.dumps(r, sort_keys=True) for r in again] == [json.dumps(r, sort_keys=True) for r in records]
    assert 600 <= len(records) <= 900
    counts = Counter(r["meta"]["family"] for r in records)
    assert set(counts) == set(N.FAMILIES) and min(counts.values()) >= 140
    assert len({r["id"] for r in records}) == len(records) and all(r["id"].startswith(N.N9G_PREFIX) for r in records)
    assert all(validate_record(r) == [] for r in records)
    assert all(list(r["request"]["questions"]) == [N.N9G_QUESTION] for r in records)


def test_option_counts_cover_the_rule_and_selection_spans_4_to_20(records):
    sizes = {f: sorted({len(r["request"]["questions"]["q"]["criteria"]) for r in records if r["meta"]["family"] == f}) for f in N.FAMILIES}
    assert sizes["selection"] == list(N.SELECTION_SIZES) and min(sizes["selection"]) == 4 and max(sizes["selection"]) == 20
    assert sizes["threshold"] == [5, 7, 9, 11]
    assert max(n for v in sizes.values() for n in v) <= 20 and min(n for v in sizes.values() for n in v) >= 2


def test_domains_include_five_outside_the_generators_list_in_every_family_but_comparison_has_all(records):
    assert set(N.UNSEEN_DOMAINS).isdisjoint(N.SEEN_DOMAINS)
    assert {"pressure", "voltage", "volume", "currency"} <= set(N.UNSEEN_DOMAINS)
    for family in N.FAMILIES:
        domains = {r["meta"]["domain"] for r in records if r["meta"]["family"] == family}
        assert domains & set(N.UNSEEN_DOMAINS) and domains & set(N.SEEN_DOMAINS), family
    assert {r["meta"]["domain"] for r in records if r["meta"]["family"] == "comparison"} == set(N.DOMAINS)
    assert all(r["meta"]["domain_seen"] == (r["meta"]["domain"] in N.SEEN_DOMAINS) for r in records)


def test_every_band_of_the_threshold_rules_is_used_and_about_half_the_readings_sit_near_a_bound(records):
    thr = [r for r in records if r["meta"]["family"] == "threshold"]
    bands = Counter(r["meta"]["band"][0] if r["meta"]["band"] != "hold" else "hold" for r in thr)
    assert bands["hold"] > 10 and bands["+"] > 30 and bands["-"] > 30
    for k in range(1, 6):  # every step index of every size occurs
        assert any(r["meta"]["correct"].endswith(tuple(w[k - 1] for w in N.SIZE_WORDS.values())) for r in thr)
    near = [r for r in records if r["meta"]["family"] in ("threshold", "interval", "conversion")]
    share = sum(r["meta"]["near"] in ("rel", "abs") for r in near) / len(near)
    assert 0.35 <= share <= 0.6
    assert Counter(r["meta"]["near"] for r in near)["abs"] >= 60


def test_formats_and_wordings_are_spread_and_numbers_read_back(records):
    formats = Counter(r["meta"]["format"] for r in records)
    assert set(formats) == set(N.FORMATS) and min(formats.values()) >= 20
    assert Counter(r["meta"]["wording"] for r in records) == {"plain": len(records) // 2, "distractor": len(records) - len(records) // 2}
    assert N.write(Decimal("-0.4"), "bare") == "-.4" and N.write(Decimal("-3.5"), "unicode", signed_family=True) == "−3.5"
    assert N.write(Decimal("3.5"), "unicode", signed_family=True) == "+3.5" and N.write(Decimal("3.5"), "signed") == "+3.5"
    assert N.write(Decimal("1250.5"), "thousands") == "1,250.5" and N.write(Decimal("12"), "fixed2") == "12.00" and N.write(Decimal("7"), "int") == "7"
    with pytest.raises(ValueError):
        N.write(Decimal("2.5"), "int")
    assert N.unit_text("-3.40", "$") == "-$3.40" and N.unit_text("12", "°") == "12°" and N.unit_text("4.2", "mm") == "4.2 mm"


def test_distractor_wordings_put_option_words_into_the_texts_without_numbers(records):
    for r in records:
        if r["meta"]["wording"] != "distractor" or r["meta"]["family"] != "threshold":
            continue
        words = {w for k in r["request"]["questions"]["q"]["criteria"] for w in k.split("_")}
        extra = [line for line in r["request"]["state"].split("\n")[1:3]]
        assert any(w in " ".join(extra).lower() for w in words), r["id"]
    plain = [r for r in records if r["meta"]["wording"] == "plain" and r["meta"]["family"] == "threshold"]
    assert all("toolbox" not in r["request"]["state"] for r in plain)


def test_no_reading_sits_on_a_bound_and_selection_has_one_right_candidate(records, selfcheck):
    result = selfcheck.check(records)
    assert result["ok"], list(result["problems"].items())[:3]
    assert result["relabel_matches"] == len(records)


# --------------------------------------------------------------------------
# The self-check catches broken records
# --------------------------------------------------------------------------


def _first(records, family, **meta):
    return next(r for r in records if r["meta"]["family"] == family and all(r["meta"].get(k) == v for k, v in meta.items()))


def test_the_self_check_catches_a_wrong_key(records, selfcheck):
    r = _copy(_first(records, "interval"))
    other = next(k for k in r["request"]["questions"]["q"]["criteria"] if k != r["meta"]["correct"])
    r["meta"]["correct"] = other
    r["labels"]["q"]["probabilities"] = {k: (1.0 if k == other else 0.0) for k in r["labels"]["q"]["probabilities"]}
    _, problems = selfcheck.check_record(r)
    assert any("relabelled" in p for p in problems)


def test_the_self_check_catches_a_reading_moved_across_a_threshold_in_the_text(records, selfcheck):
    r = _copy(next(x for x in records if x["meta"]["family"] == "threshold" and x["meta"]["band"] == "hold"))
    spec = r["meta"]["spec"]
    big = spec["thresholds_text"][-1]
    old = spec["value_text"]
    r["request"]["state"] = r["request"]["state"].replace(old, big + "9", 1)
    spec["value_text"] = big + "9"
    _, problems = selfcheck.check_record(r)
    assert any("relabelled" in p for p in problems)


def test_the_self_check_catches_a_reading_on_a_bound_and_a_rule_clause_with_the_wrong_number(records, selfcheck):
    r = _copy(next(x for x in records if x["meta"]["family"] == "threshold" and x["meta"]["spec"]["style"] == "table"))
    spec = r["meta"]["spec"]
    t0 = spec["thresholds_text"][0]
    r["request"]["state"] = r["request"]["state"].replace(spec["value_text"] + " ", t0 + " ", 1).replace(spec["value_text"] + "°", t0 + "°", 1)
    spec["value_text"] = t0
    _, problems = selfcheck.check_record(r)
    assert any("sits on a threshold" in p for p in problems)
    r2 = _copy(next(x for x in records if x["meta"]["family"] == "threshold" and x["meta"]["spec"]["style"] == "prose"
                    and len(x["meta"]["spec"]["thresholds_text"]) >= 3))
    t1 = r2["meta"]["spec"]["thresholds_text"][1]
    before = r2["request"]["questions"]["q"]["instructions"]
    r2["request"]["questions"]["q"]["instructions"] = before.replace(f"from {t1} ", "from 999 ", 1)
    assert r2["request"]["questions"]["q"]["instructions"] != before
    _, problems = selfcheck.check_record(r2)
    assert any("clause starts at" in p or "not the spec" in p for p in problems)


def test_the_self_check_catches_swapped_sign_actions(records, selfcheck):
    r = _copy(_first(records, "threshold"))
    q = r["request"]["questions"]["q"]
    sign_text = next(line for line in (r["request"]["state"].split("\n") + [q["instructions"]]) if "positive" in line)
    swapped = sign_text.replace("positive", "NEGTMP").replace("negative", "positive").replace("NEGTMP", "negative")
    r["request"]["state"] = r["request"]["state"].replace(sign_text, swapped)
    q["instructions"] = q["instructions"].replace(sign_text, swapped)
    _, problems = selfcheck.check_record(r)
    assert any("clause does not name" in p for p in problems)


def test_the_self_check_catches_a_second_right_candidate_a_tie_and_a_stray_number(records, selfcheck):
    r = _copy(_first(records, "selection"))
    spec = r["meta"]["spec"]
    other = next(k for k in spec["candidates"] if k != r["meta"]["correct"])
    right = spec["candidates"][r["meta"]["correct"]]
    spec["candidates"][other] = json.loads(json.dumps(right))
    r["request"]["questions"]["q"]["criteria"][other] = r["request"]["questions"]["q"]["criteria"][r["meta"]["correct"]]
    _, problems = selfcheck.check_record(r)
    assert any("candidates meet every constraint" in p for p in problems)
    c = _copy(next(x for x in records if x["meta"]["spec"].get("kind") == "extreme"))
    names = list(c["meta"]["spec"]["items"])
    c["meta"]["spec"]["items"][names[1]] = c["meta"]["spec"]["items"][names[0]]
    _, problems = selfcheck.check_record(c)
    assert any("tied values" in p for p in problems)
    s = _copy(_first(records, "conversion"))
    s["request"]["state"] += "\nThe line has 17 stations."
    _, problems = selfcheck.check_record(s)
    assert any("not the spec" in p for p in problems)


def test_the_self_check_catches_a_rule_and_a_reading_in_different_units(records, selfcheck):
    for kind in ("tolerance", "band", "threshold"):
        r = _copy(next(x for x in records if x["meta"]["spec"].get("kind") == kind))
        spec = r["meta"]["spec"]
        unit = spec["unit"]
        wrong = "kPa" if unit != "kPa" else "bar"
        r["request"]["state"] = r["request"]["state"].replace(f"{spec['value_text']} {unit}", f"{spec['value_text']} {wrong}")
        spec["unit"] = wrong
        _, problems = selfcheck.check_record(r)
        assert any("reading's unit" in p or "band end" in p for p in problems), (kind, problems)
    tires = [x for x in records if x["meta"]["template"] == "interval-tire-tolerance"]
    assert tires and all(x["meta"]["spec"]["unit"] == "bar" and " bar." in x["request"]["state"].split("\n")[-1] for x in tires)


def test_the_self_check_converts_with_its_own_table(selfcheck):
    assert selfcheck.TO_RULE[("kPa", "bar")](selfcheck.parse("553")) == selfcheck.parse("5.53")
    assert selfcheck.TO_RULE[("K", "°C")](selfcheck.parse("326.15")) == 53
    assert selfcheck.parse("−1,250.5") == selfcheck.parse("-1250.5") and selfcheck.parse(".4") * 10 == 4 and selfcheck.parse("+$3") == 3
    pairs = {(sc["read_unit"], sc["rule_unit"]) for sc in N.CONVERSION_SCENARIOS}
    assert pairs <= set(selfcheck.TO_RULE)


# --------------------------------------------------------------------------
# Readings
# --------------------------------------------------------------------------


def _answers(records, pick):
    out = {}
    for r in records:
        keys = list(r["request"]["questions"]["q"]["criteria"])
        top = pick(r, keys)
        probs = {k: (0.7 if k == top else 0.3 / (len(keys) - 1)) for k in keys}
        out[r["id"]] = {"q": {"single-s1": {"choice": top, "probabilities": probs}, "single-s2": {"choice": top, "probabilities": probs}}}
    return out


def test_rows_seed_means_and_summary(records):
    chosen = records[:40]
    right = _answers(chosen, lambda r, keys: r["meta"]["correct"])
    wrong = _answers(chosen, lambda r, keys: next(k for k in keys if k != r["meta"]["correct"]))
    a, b = N.n9g_rows(chosen, right), N.n9g_rows(chosen, wrong)
    assert len(a) == 40 and all(x["correct"] == 1.0 and x["flip"] == 0.0 for x in a) and all(x["correct"] == 0.0 for x in b)
    mean = N.seed_mean_rows(a, b)
    assert all(x["correct"] == 0.5 for x in mean)
    summary = N.n9g_summary(mean)
    assert summary["total"]["accuracy"]["value"] == 0.5 and summary["total"]["accuracy"]["n"] == 40
    assert set(summary["by_family"]) <= set(N.FAMILIES) and "seen domain" in summary["by_domain_group"] or "unseen domain" in summary["by_domain_group"]
    tie = {chosen[0]["id"]: {"q": {"single-s1": {"probabilities": {k: 0.5 for k in chosen[0]["request"]["questions"]["q"]["criteria"]}}}}}
    first = list(chosen[0]["request"]["questions"]["q"]["criteria"])[0]
    assert N.n9g_rows(chosen[:1], tie)[0]["top"] == first and N.n9g_rows(chosen[:1], tie)[0]["flip"] is None


# --------------------------------------------------------------------------
# Sealing
# --------------------------------------------------------------------------


def _marked(record: dict, *, how: str) -> dict:
    out = _copy(record)
    if how == "meta":
        out["meta"] = {**(out.get("meta") or {}), "set": N.N9G_SET}
    else:
        out["id"] = "v09n9g-0007"
    return out


def test_n9g_is_registered_as_a_sealed_never_sent_set_with_its_file_and_prefix():
    assert N.N9G_SET in SEALED_RECORD_SETS and N.N9G_SET in NEVER_SENT_SETS and SEALED_RECORD_PREFIXES[N.N9G_SET] == N.N9G_PREFIX
    assert SEALED_RECORD_FILES[N.N9G_SET] == "artifacts/v09/n9g/N9g.jsonl" == str(N.N9G_FILE.relative_to(N.REPO))
    assert {"T9", "v06-a4", "v07-m2g", "v07-m1g", "v08-b8", "v08-m8g"} <= set(NEVER_SENT_SETS), "earlier seals stay"
    assert {"v08-r14", "v08-a16", "v08-s16", "v08-a16s"} <= set(SEALED_RECORD_SETS)


def test_an_n9g_record_is_recognised_and_refused_by_the_trainer(records):
    from robo_jev.decisions.train import DecisionTrainer

    base = read_records(REQUESTS)
    assert sealed_record_set(records[0]) == N.N9G_SET and sealed_record_set(base[0]) is None
    for how in ("meta", "id"):
        assert sealed_record_set(_marked(base[0], how=how)) == N.N9G_SET
    with pytest.raises(ValueError, match="sealed outside test"):
        refuse_sealed_records([base[0], _marked(base[1], how="id")], where="training input")
    with pytest.raises(ValueError, match="sealed outside test"):
        DecisionTrainer({}, records=[*base, _marked(base[0], how="meta")], log=None)


def test_the_mix_assembly_refuses_an_n9g_record(tmp_path):
    from test_v02_mix import _probe_and_sealed

    probe, sealed = _probe_and_sealed(tmp_path)
    base = read_records(REQUESTS)
    script = _script("convert_single_probe.py")
    rows = [_copy(base[0]), _marked(base[1], how="id")]
    rows[0]["id"] = "xpart-0000"
    for row in rows:
        row["meta"] = {**(row.get("meta") or {}), "split": "train", "language": "en"}
    part = tmp_path / "poisoned.jsonl"
    part.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    with pytest.raises(SystemExit, match="sealed outside test"):
        script.main(["mix", "--base", str(REQUESTS), "--probe", str(probe), "--sealed", str(sealed), "--train-config", str(TRAIN_CONFIG),
                     "--extra", f"mujoco={part}", "--out", str(tmp_path / "a.jsonl")])


def test_the_api_paths_refuse_to_send_an_n9g_record(records):
    from robo_jev.e1 import rewrite as rw

    index = rw.SealedIndex(set(), set(), {})
    bench = _script("bench.py")
    jev = _script("v08_jev.py")
    base = read_records(REQUESTS)
    for marked in (_marked(base[0], how="meta"), _marked(base[0], how="id"), records[3]):
        with pytest.raises(rw.SealedTextError, match="never sent"):
            rw.assert_sendable(marked, [marked["request"]["state"]], index)
        with pytest.raises(SystemExit, match="never sent"):
            bench.refuse_never_sent({"T2": [base[1], marked]})
        with pytest.raises(SystemExit):
            jev.refuse_sealed([base[1], marked])


# --------------------------------------------------------------------------
# The scorer's freeze and the build script
# --------------------------------------------------------------------------


def test_the_bench_scorer_scores_n9g_only_against_its_frozen_file(tmp_path, records):
    score = _script("v08_bench_score.py")
    assert "N9g" in score.SETS
    gate = tmp_path / "N9g.jsonl"
    gate.write_text("".join(json.dumps(r) + "\n" for r in records[:5]), encoding="utf-8")
    with pytest.raises(SystemExit, match="frozen"):
        score.check_frozen(gate, None, "N9g")
    frozen = tmp_path / "frozen-before-scoring.json"
    frozen.write_text(json.dumps({"files": {"N9g.jsonl": N.sha256_file(gate)}}), encoding="utf-8")
    assert score.check_frozen(gate, frozen, "N9g")["files"]["N9g.jsonl"] == N.sha256_file(gate)
    gate.write_text(gate.read_text(encoding="utf-8") + json.dumps(records[6]) + "\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="refusing to score N9g"):
        score.check_frozen(gate, frozen, "N9g")
    assert N.check_frozen(tmp_path) == ["N9g.jsonl changed after it was frozen"]


def test_the_build_script_never_writes_into_sealed_or_earlier_folders():
    build = _script("v09_gates_build.py")
    build.refuse_write(REPO / "artifacts" / "v09" / "n9g" / "x.json")
    for bad in ("simgen/v09/train-v09n.jsonl", "simgen/v08/x", "reports/v09/x.json", "v08/gates/x.json", "t9/T9.jsonl", "v07/x", "iface/x", "b1/x"):
        with pytest.raises(SystemExit, match="never written"):
            build.refuse_write(REPO / "artifacts" / bad)


def test_build_freeze_and_recheck_on_temporary_files(tmp_path, monkeypatch):
    build = _script("v09_gates_build.py")
    out = tmp_path / "n9g"
    assert build.main(["build-n9g", "--out-dir", str(out), "--skip-overlap"]) == 0
    manifest = json.loads((out / "n9g.manifest.json").read_text(encoding="utf-8"))
    assert manifest["file"]["records"] == len(read_records(out / "N9g.jsonl")) and manifest["self_check"]["ok"]
    assert manifest["prompt_tokens"]["per_reading"] > 100_000 and (out / "audit" / "n9g-audit-sample.md").is_file()
    assert build.main(["freeze-n9g", "--out-dir", str(out)]) == 0
    assert N.check_frozen(out) == [] and not (out / "N9g.jsonl").stat().st_mode & stat.S_IWUSR
    with pytest.raises(SystemExit, match="frozen"):
        build.main(["build-n9g", "--out-dir", str(out), "--skip-overlap"])
    gate = read_records(out / "N9g.jsonl")
    train = tmp_path / "train-v09n.jsonl"
    leak = {"id": "x-1", "request": {"state": "unrelated", "questions": {"q": {"type": "choice", "instructions": gate[5]["request"]["questions"]["q"]["instructions"],
                                                                                "criteria": {"a": "x", "b": "y"}}}}}
    copy = {"id": "x-2", "request": gate[9]["request"]}
    train.write_text(json.dumps(leak) + "\n" + json.dumps(copy) + "\n", encoding="utf-8")
    overlap = tmp_path / "overlap-v09.json"
    assert build.main(["recheck-v09", "--train-v09n", str(train), "--out", str(overlap), "--n9g-dir", str(out)]) == 0
    found = json.loads(overlap.read_text(encoding="utf-8"))
    excluded = set(found["excluded_n9g"])
    assert gate[9]["id"] in excluded
    if len(gate[5]["request"]["questions"]["q"]["instructions"].split()) >= 5:
        assert gate[5]["id"] in excluded
    assert found["train_v09n"]["sha256"] == N.sha256_file(train) and found["excluded_items"] == [] and found["excluded_a16s"] == []


def test_the_chain_check_refuses_without_the_recheck_or_with_runs_on_other_inputs(tmp_path):
    build = _script("v09_gates_build.py")

    def run(name: str, sha: str) -> Path:
        d = tmp_path / name
        d.mkdir()
        (d / "metrics.json").write_text(json.dumps({"manifest": {"data": {"file_sha256": sha}}}), encoding="utf-8")
        return d

    c = [run("c0", "aaa"), run("c1", "aaa")]
    cn = [run("cn0", "bbb"), run("cn1", "ccc")]
    problems, _ = build.gate_check(c, cn, [run("f0", "aaa")], overlap_v09=tmp_path / "missing.json")
    assert any("recheck-v09" in p for p in problems)
    assert any("v0.8 gates / C runs" in p for p in problems)  # the C runs are not on train-v08
    assert any("not a v0.8 F run" in p for p in problems)
    overlap = tmp_path / "overlap-v09.json"
    overlap.write_text(json.dumps({"train_v09n": {"sha256": "bbb"}}), encoding="utf-8")
    problems, info = build.gate_check(c, cn, [], overlap_v09=overlap)
    assert any("cn1: trained on ccc" in p for p in problems) and not any("cn0: trained" in p for p in problems)
    assert info["cn_trained_on"][str(cn[0])] == "bbb"


# --------------------------------------------------------------------------
# Fixes after the ambiguity audit (round 1)
# --------------------------------------------------------------------------


def test_selection_rules_name_every_required_state_and_bad_statuses_break_one(records):
    for sc in N.SELECTION_SCENARIOS:
        good, bad = sc["status"]
        rule = sc["rule"]
        assert good in rule, sc["name"]
        assert all(b != good for b in bad)
    for r in records:
        if r["meta"]["family"] != "selection":
            continue
        good = r["meta"]["spec"]["status_ok"]
        for key, cand in r["meta"]["spec"]["candidates"].items():
            assert (cand["status"] == good) == (cand["breaks"] != "status"), (r["id"], key)


def test_tolerance_readings_use_the_tolerances_unit_and_kelvin_items_are_not_decided_by_the_offsets_decimals(records):
    for r in records:
        spec = r["meta"]["spec"]
        if spec.get("kind") == "tolerance":
            rule = next(line for line in r["request"]["state"].split("\n") if "plus or minus" in line)
            assert N.unit_text(spec["center_text"], spec["unit"]) in rule and N.unit_text(spec["tol_text"], spec["unit"]) in rule, r["id"]
            assert r["request"]["state"].split("\n")[-1].rstrip(".").endswith(N.unit_text(spec["value_text"], spec["unit"])), r["id"]
        if spec.get("kind") == "conversion" and spec.get("offset"):
            assert r["meta"]["near"] in (None, "rel")


def test_step_names_take_the_right_article_and_comparison_asides_fit_their_domain(records):
    texts = [r["request"]["state"] + " " + r["request"]["questions"]["q"]["instructions"] for r in records]
    assert not any(" a extreme" in t or " a eight" in t for t in texts)
    assert any(" an extreme step" in t for t in texts)
    assert set(N.COMPARE_ASIDES) == set(N.COMPARE_DOMAINS)
    assert not any("Pa " in r["request"]["state"] and "kPa" not in r["request"]["state"] for r in records)


def test_lures_name_wrong_options_and_the_statistic_is_recorded(records):
    for r in records:
        assert (r["meta"]["distractor_names_key"] is None) == (r["meta"]["wording"] == "plain")
    counts = N.design_counts(records)["distractor_names_key"]
    assert set(counts) == set(N.FAMILIES)
    for family in ("interval", "selection", "comparison", "conversion"):  # a wrong option's label, candidate, item or option
        assert counts[family]["distractor_items"] > 50 and counts[family]["names_the_key"] == 0, family
    assert 0 < counts["threshold"]["names_the_key"] < counts["threshold"]["distractor_items"]  # the words of several options
    seconds = [r for r in records if r["meta"]["spec"].get("kind") == "second"]
    assert seconds and min(len(r["meta"]["spec"]["items"]) for r in seconds) >= 4
    for r in records:
        if r["meta"]["spec"].get("kind") == "closest" and r["meta"]["format"] == "unicode" and r["meta"]["spec"]["target_text"].startswith(("-", "−")):
            assert r["meta"]["spec"]["target_text"].startswith("−")
    blind = 0
    for r in records:
        spec = r["meta"]["spec"]
        if r["meta"]["family"] != "conversion" or r["meta"]["wording"] != "distractor":
            continue
        sign = next(line for line in r["request"]["state"].split("\n") if line.startswith("A sign"))
        named = sign.split('"')[1].lower().replace(" ", "_")
        assert named in spec["keys"] and named != r["meta"]["correct"], r["id"]
        reading = float(spec["value_text"].replace(",", ""))
        blind += named == spec["keys"][sum(1 for b in spec["bounds_text"] if reading > float(b))]
    assert blind > 30, "the faded sign names the unit-blind answer whenever that differs from the key"


def test_ranking_questions_use_the_domains_words(records):
    texts = [r["request"]["questions"]["q"]["instructions"] for r in records if r["meta"]["spec"].get("kind") in ("second", "order")]
    assert not any("highest first" in t and "by distance" in t for t in texts)
    assert any("farthest first" in t or "nearest first" in t for t in texts) and any("from coldest to warmest" in t or "from warmest to coldest" in t for t in texts)
