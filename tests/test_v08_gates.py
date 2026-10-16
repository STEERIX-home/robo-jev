"""Task v0.8b — gate M8g (many options up to 255), the probes, the Jev set, the scorers' pure parts, the post-hoc estimate and the sealing.

Pinned on synthetic items (no sealed file is read, except the last tests, which check the built files when they are present): the 255 codes
are the runner's; the correct position walks its whole band (single-letter at N ≥ 40 for listing 0, two-letter for listing 1); every
physics item has exactly one candidate meeting every constraint (re-derived from the texts); general draws stay topic-distant; the listings
are nested across N with the same answer, the opt-in and one-hot labels; the gates are registered as sealed, never-sent sets so the trainer,
the mix assembly and the API paths refuse them.
"""

from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path

import pytest
from helpers import FIXTURES, REPO

from robo_jev.bench import gates_v08 as G
from robo_jev.bench import iface
from robo_jev.data.sealed import NEVER_SENT_SETS, SEALED_RECORD_FILES, SEALED_RECORD_PREFIXES, SEALED_RECORD_SETS, refuse_sealed_records, sealed_record_set
from robo_jev.decisions.records import OPTION_RANGE_MARK, read_records

REQUESTS = FIXTURES / "decisions" / "requests.jsonl"
TRAIN_CONFIG = REPO / "configs" / "train" / "decisions-v0-lora-h100.yaml"


def _script(name: str):
    spec = importlib.util.spec_from_file_location(f"v08gates_{Path(name).stem}", REPO / "scripts" / name)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _w(i: int, k: int) -> str:
    """A made-up word unique to (item i, slot k): letters only (a digit would make the text "numeric" and keep it out of the pool)."""
    n, out = i * 7 + k, ""
    for _ in range(4):
        out += chr(97 + n % 26)
        n //= 26
    return out + "q" + chr(97 + k)


@pytest.fixture()
def pool() -> list[dict]:
    """300 general items, every word unique to its item (so any other item is topic-distant)."""
    out = []
    for i in range(300):
        t = _w(i, 0)
        out.append({"item": f"mmlu:p{i:04d}", "source": G.GENERAL_SOURCE, "subject": "s", "state": f'question: "Which {t} {_w(i, 1)}?"',
                    "instructions": iface.ARC_INSTRUCTION, "stem": f"Which {t} {_w(i, 1)}?", "correct": f"{_w(i, 2)} {t} {_w(i, 3)}",
                    "own": [f"{_w(i, 4)} {_w(i, 5)} {t}", f"{_w(i, 5)} {_w(i, 6)} {t}", f"{_w(i, 6)} {_w(i, 4)} {t}"], "target": True})
    return out


# --------------------------------------------------------------------------
# Codes, bins, positions
# --------------------------------------------------------------------------


def test_the_codes_are_the_runners_first_255_in_order_and_the_bins_cover_2_to_255():
    from robo_jev.hf.robo_jev_decide import CODES

    assert G.CODES_255 == tuple(CODES[:255]) and len(set(G.CODES_255)) == 255
    assert G.CODES_255[:26] == tuple("ABCDEFGHIJKLMNOPQRSTUVWXYZ") and all(len(c) == 2 for c in G.CODES_255[26:])
    assert [G.bin_of(n) for n in (2, 10, 11, 26, 27, 60, 61, 255)] == ["2-10", "2-10", "11-26", "11-26", "27-60", "27-60", "61-255", "61-255"]
    with pytest.raises(ValueError):
        G.bin_of(256)
    assert G.option_range_mark() == {"version": G.OPTION_RANGE_VERSION, "max": 255}
    assert [G.bin_of(n) for n in G.M8G_LEVELS] == ["2-10", "11-26", "27-60", "61-255", "61-255"]


@pytest.mark.parametrize("n", G.M8G_LEVELS)
def test_the_correct_position_walks_its_whole_band_and_the_code_kind_is_fixed_by_the_listing(n):
    for listing in (0, 1):
        if n <= 26:
            half = (n + 1) // 2
            band = range(0, half) if listing == 0 else range(half, n)
        else:
            band = range(0, 26) if listing == 0 else range(26, n)
        seen = {G.m8g_position(n, rank, listing) for rank in range(len(band))}
        assert seen == set(band), (n, listing)
        if n > 26:
            assert all(G.two_letter(p) == (listing == 1) for p in seen)
    with pytest.raises(ValueError):
        G.m8g_position(40, 0, 2)


# --------------------------------------------------------------------------
# Physics items
# --------------------------------------------------------------------------


@pytest.mark.parametrize("family", G.PHYSICS_FAMILIES)
def test_a_physics_item_has_exactly_one_candidate_meeting_every_constraint(family):
    build = _script("v08_gates_build.py")
    for index in (0, 3, 7):
        item = G.physics_item(family, index, seed=20261016)
        assert build.physics_violations(item) == []
        keys = [item["correct"][0]] + [k for k, _ in item["pool"]]
        texts = [item["correct"][1]] + [t for _, t in item["pool"]]
        assert len(keys) == 255 and len(set(keys)) == 255 and len(set(texts)) == 255
        width = len(item["constraints"])
        assert set(item["broken"][:width]) == set(item["constraints"]), "the first draws break every constraint once"
        assert item == G.physics_item(family, index, seed=20261016), "deterministic"
    jev = G.physics_item(family, 500, seed=20261016, numbers=(1000, 10000))
    assert all(1000 <= int(k.split("_")[1]) < 10000 for k, _ in [jev["correct"], *jev["pool"]])


def test_the_construction_check_catches_a_second_feasible_candidate():
    build = _script("v08_gates_build.py")
    item = G.physics_item("bins", 1, seed=5)
    key, text = item["correct"]
    broken = dict(item, pool=[(f"{key}x", text.replace(key.split("_")[1], "999x"))] + item["pool"][1:])
    assert build.physics_violations(broken), "a distractor meeting every constraint is flagged"


# --------------------------------------------------------------------------
# General draws and listings
# --------------------------------------------------------------------------


def test_an_item_whose_options_point_at_other_options_is_found_even_without_of(pool):
    item = dict(pool[0])
    assert not G.refers_to_options(item)
    for text in ("All the above", "all of the above", "None the above", "Both of these", "All are correct", "Both A and B", "the above answers"):
        assert G.refers_to_options(dict(item, own=[*item["own"][:2], text])), text
        assert G.refers_to_options(dict(item, correct=text)), text
    for text in ("the stones above the river", "a pile of rubble", "both hands on the wheel"):
        assert not G.refers_to_options(dict(item, own=[*item["own"][:2], text])), text


def test_general_draws_stay_topic_distant_one_per_question_and_the_first_ones_follow_the_strict_rule(pool):
    item = pool[0]
    index = iface.general_index(pool[1:])
    drawn = G.draw_general_many(item, index, count=120, seed=3)
    assert len(drawn) == len(set(drawn)) == 120
    topic = iface._topic(item)
    assert all(not iface.content_tokens(t) & topic for t in drawn)
    strict = drawn[: G.STRICT_DRAWS]
    seen: set[str] = set()
    for text in strict:
        tokens = iface.content_tokens(text)
        assert not tokens & seen
        seen |= tokens
    sources = [next(e["item"] for e in index if any(t == w for w, _ in e["wrong"])) for t in drawn]
    assert len(set(sources)) == len(sources), "one draw per source question"
    with pytest.raises(ValueError, match="needed"):
        G.draw_general_many(item, index, count=2000, seed=3)


def test_the_listings_are_nested_with_the_same_answer_the_opt_in_and_one_hot_labels(pool):
    build = _script("v08_gates_build.py")
    item = pool[0]
    drawn = G.draw_general_many(item, iface.general_index(pool[1:]), count=251, seed=1)
    phys = G.physics_item("doors", 2, seed=1)
    records = G.expand_m8g(item, drawn, rank=5, index=0, seed=1) + G.expand_m8g(phys, phys["pool"], rank=6, index=1, seed=1)
    assert len(records) == 2 * len(G.M8G_LEVELS) * G.M8G_LISTINGS
    build._check_records(records)  # nested, same answer, position / code match the shown order, every record passes gate_problems
    for r in records:
        m = r["meta"]
        q = r["request"]["questions"][G.M8G_QUESTION]
        assert len(q["criteria"]) == m["n"] and list(q["criteria"]).index(m["correct"]) == m["position"]
        assert m["code"] == G.CODES_255[m["position"]] and m["two_letter"] == (m["position"] >= 26) and m["bin"] == G.bin_of(m["n"])
        assert m["option_range"] == G.option_range_mark() and m["set"] == G.M8G_SET and r["id"].startswith(G.M8G_PREFIX)
        assert r["labels"][G.M8G_QUESTION]["probabilities"][m["correct"]] == 1.0 and sum(r["labels"][G.M8G_QUESTION]["probabilities"].values()) == 1.0
        assert sealed_record_set(r) == "v08-m8g" and G.gate_problems(r) == []
    general = [r for r in records if r["meta"]["source"] == G.GENERAL_SOURCE and r["meta"]["n"] == 255]
    assert all(set(r["request"]["questions"][G.M8G_QUESTION]["criteria"]) == {f"o{i}" for i in range(1, 256)} for r in general)


def test_gate_problems_keeps_every_check_but_the_range_and_reads_the_opt_in(pool):
    item = pool[0]
    drawn = G.draw_general_many(item, iface.general_index(pool[1:]), count=251, seed=1)
    record = next(r for r in G.expand_m8g(item, drawn, rank=0, index=0, seed=1) if r["meta"]["n"] == 40)
    assert G.gate_problems(record) == []
    wrong = json.loads(json.dumps(record))
    wrong["meta"]["option_range"] = dict(OPTION_RANGE_MARK)
    assert any("opt-in" in p for p in G.gate_problems(wrong))
    bad = json.loads(json.dumps(record))
    bad["labels"][G.M8G_QUESTION]["probabilities"] = {k: 0.5 for k in bad["labels"][G.M8G_QUESTION]["probabilities"]}
    assert any("sum" in p for p in G.gate_problems(bad)), "labels of > 8-option questions are still checked"
    big = json.loads(json.dumps(record))
    big["request"]["questions"][G.M8G_QUESTION]["criteria"] = {f"k{i}": "x" for i in range(256)}
    assert any("2..255" in p for p in G.gate_problems(big))


# --------------------------------------------------------------------------
# Probes, scorers, Jev, post-hoc
# --------------------------------------------------------------------------


def test_the_probe_has_rel071s_20_storage_bin_cases():
    build = _script("v08_gates_build.py")
    rows = build.probe_rows()
    assert len(rows) == 20 and sum(r["meta"]["kind"] == "position" for r in rows) == 18
    for r in rows:
        keys = list(r["request"]["questions"]["bin"]["criteria"])
        assert len(keys) == r["meta"]["n"] and r["meta"]["right"] in keys
        if r["meta"]["kind"] == "position":
            assert keys.index(r["meta"]["right"]) == r["meta"]["position"]
    rel071 = REPO / "artifacts" / "scratch" / "rel071" / "gpu_check.py"
    if rel071.is_file():  # bins() is rel071's code, verbatim after its docstring
        source = rel071.read_text(encoding="utf-8")
        body = source[source.index("def bins("):source.index("def main(")].split('"""', 2)[2].strip()
        assert body in Path(build.__file__).read_text(encoding="utf-8")


def test_the_bench_scorer_reads_readings_targets_and_refuses_unfrozen_b8(tmp_path):
    score = _script("v08_bench_score.py")
    assert score.parse_reading("single-s2") == {"name": "single-s2", "orders": "single", "seed": 2}
    assert score.parse_reading("balanced-s1")["orders"] == "balanced" and score.parse_reading("order-none")["seed"] is None
    with pytest.raises(ValueError):
        score.parse_reading("single")
    record = read_records(REQUESTS)[0]
    choice = [n for n, q in record["request"]["questions"].items() if q["type"] == "choice"]
    assert score.targets_of(record) == choice and score.readings_of(record, ["single-s1"])[0]["seed"] == 1
    b8, tags = tmp_path / "B8.jsonl", tmp_path / "tags.json"
    b8.write_text("{}\n", encoding="utf-8")
    tags.write_text("{}\n", encoding="utf-8")
    frozen = tmp_path / "frozen-before-scoring.json"
    frozen.write_text(json.dumps({"b8_sha256": score.sha256(b8), "tags_sha256": score.sha256(tags)}), encoding="utf-8")
    assert score.check_frozen(b8, frozen, "B8")["b8_sha256"] == score.sha256(b8)
    assert score.check_frozen(b8, None, "Jev") is None
    tags.write_text('{"changed": 1}\n', encoding="utf-8")
    with pytest.raises(SystemExit, match="frozen tags"):
        score.check_frozen(b8, frozen, "B8")
    with pytest.raises(SystemExit, match="frozen"):
        score.check_frozen(b8, None, "B8")


def test_the_m8g_scorer_selects_listings_and_takes_no_lifted_range(tmp_path, pool):
    score = _script("v08_m8g_score.py")
    item = pool[0]
    drawn = G.draw_general_many(item, iface.general_index(pool[1:]), count=251, seed=1)
    records = G.expand_m8g(item, drawn, rank=0, index=0, seed=1) + G.expand_m8g(pool[2], drawn, rank=1, index=1, seed=1)
    picked = score.select(records, listing=0, max_n=60, items=1)
    assert {r["meta"]["n"] for r in picked} == {6, 18, 40} and {r["meta"]["listing"] for r in picked} == {0} and {r["meta"]["rank"] for r in picked} == {0}
    assert score.order_seed("none") is None and score.order_seed("2") == 2
    small = [json.loads(json.dumps(r)) for r in records if r["meta"]["n"] <= 26]
    for r in small:
        r["meta"]["option_range"] = dict(OPTION_RANGE_MARK)  # v0.7's opt-in: main accepts up to 26 options
    path = tmp_path / "small.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in small), encoding="utf-8")
    assert len(score.load_checked(path)) == len(small)
    for r in small:
        r["meta"]["option_range"] = {"version": "not-an-opt-in", "max": 255}
    path.write_text("".join(json.dumps(r) + "\n" for r in small), encoding="utf-8")
    with pytest.raises(SystemExit, match="fail main's checks"):
        score.load_checked(path)


def test_the_jev_runner_refuses_sealed_records_and_sends_only_the_target_questions():
    jev = _script("v08_jev.py")
    records = read_records(REQUESTS)
    jev.refuse_sealed(records)
    for marked in (dict(records[0], id="v08m8g-00001-n040-A"), dict(records[0], id="v08b8-plates-001"),
                   dict(records[0], meta={**(records[0].get("meta") or {}), "set": "v08-b8"})):
        with pytest.raises(SystemExit, match="refusing"):
            jev.refuse_sealed([records[1], marked])
    target = [n for n, q in records[0]["request"]["questions"].items() if q["type"] == "choice"][:1]
    record = dict(records[0], meta={**(records[0].get("meta") or {}), "target": target})
    job = jev.jobs_for([record], {"prompt": 1e-7, "completion": 0.0}, None)[0]
    assert list(job.payload["questions"]) == target and job.payload["provider"] == {"data_collection": "deny"}
    assert "meta" not in job.payload and "labels" not in job.payload


def test_the_posthoc_shift_moves_yes_no_and_score_answers_by_the_prior_logit(tmp_path):
    post = _script("v08_posthoc_prior.py")
    pred = {"id": "r", "questions": {"y": {"type": "noul", "answer": {"p_yes": 0.25}, "listings": [{"answer": {"p_yes": 0.2}}, {"answer": {"p_yes": 0.3}}]},
                                     "s": {"type": "score", "answer": {"value": 0.5}, "listings": [{"answer": {"value": 0.5}}]},
                                     "c": {"type": "choice", "answer": {"probabilities": {"a": 1.0}}, "listings": [{"answer": {"probabilities": {"a": 1.0}}}]}}}
    out = post.shift_predictions([pred], {"noul": math.log(3.0), "score": 0.0})[0]
    assert out["questions"]["y"]["listings"][0]["answer"]["p_yes"] == pytest.approx(0.2 * 3 / (0.2 * 3 + 0.8))
    assert out["questions"]["s"]["listings"][0]["answer"]["value"] == pytest.approx(0.5)
    assert out["questions"]["c"] == pred["questions"]["c"] and pred["questions"]["y"]["listings"][0]["answer"]["p_yes"] == 0.2, "a copy"
    rows = []
    for i, y in enumerate([0.1, 0.1, 0.1, 0.5, 0.9]):
        rows.append({"meta": {"body": 1}, "request": {"questions": {"q": {"type": "noul"}}}, "labels": {"q": {"p_yes": y}}})
    rows.append({"meta": {"scene": "x"}, "request": {"questions": {"q": {"type": "score"}}}, "labels": {"q": {"value": 0.8}}})
    rows.append({"meta": {"source": {"set": "qasc"}}, "request": {"questions": {"q": {"type": "noul"}}}, "labels": {"q": {"p_yes": 0.9}}})
    path = tmp_path / "train.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    prior = post.priors(path)
    e = prior["engine/noul"]
    assert e["questions"] == 5 and e["band_shares"]["low"] == pytest.approx(0.6) and e["mu"] == pytest.approx(0.34)
    assert e["mu_balanced"] == pytest.approx((0.1 + 0.5 + 0.9) / 3) and e["logit_shift"] == pytest.approx(post.logit(0.5) - post.logit(0.34))
    assert prior["physics/noul"]["questions"] == 5 and prior["mujoco/score"]["questions"] == 1 and "general" not in json.dumps(list(prior))


# --------------------------------------------------------------------------
# Sealing
# --------------------------------------------------------------------------


def test_both_gates_are_registered_sealed_never_sent_sets_with_their_files_and_prefixes():
    for name, prefix, path in (("v08-b8", "v08b8-", "artifacts/v08/gates/b8/B8.jsonl"), ("v08-m8g", "v08m8g-", "artifacts/v08/gates/m8g/M8g.jsonl")):
        assert name in SEALED_RECORD_SETS and name in NEVER_SENT_SETS and SEALED_RECORD_PREFIXES[name] == prefix and SEALED_RECORD_FILES[name] == path
    assert {"T9", "v04a-scene-holdout", "v06-a4", "v07-m2g", "v07-m1g"} <= set(SEALED_RECORD_SETS), "earlier seals stay"
    assert {"T9", "v06-a4", "v07-m2g", "v07-m1g"} <= set(NEVER_SENT_SETS)
    assert (G.B8_SET, G.B8_PREFIX, G.M8G_SET, G.M8G_PREFIX) == ("v08-b8", "v08b8-", "v08-m8g", "v08m8g-")
    assert str(G.B8_FILE.relative_to(G.REPO)) == SEALED_RECORD_FILES["v08-b8"] and str(G.M8G_FILE.relative_to(G.REPO)) == SEALED_RECORD_FILES["v08-m8g"]


def _marked(record: dict, *, how: str, name: str) -> dict:
    out = json.loads(json.dumps(record))
    if how == "meta":
        out["meta"] = {**(out.get("meta") or {}), "set": name}
    else:
        out["id"] = {"v08-b8": "v08b8-plates-001", "v08-m8g": "v08m8g-00001-n040-A"}[name]
    return out


@pytest.mark.parametrize("name", ["v08-b8", "v08-m8g"])
def test_a_gate_record_is_recognised_by_its_set_or_prefix_and_refused_by_the_trainer(name):
    from robo_jev.decisions.train import DecisionTrainer

    records = read_records(REQUESTS)
    assert sealed_record_set(records[0]) is None
    for how in ("meta", "id"):
        assert sealed_record_set(_marked(records[0], how=how, name=name)) == name
    with pytest.raises(ValueError, match="sealed outside test"):
        refuse_sealed_records([records[0], _marked(records[1], how="id", name=name)], where="training input")
    with pytest.raises(ValueError, match="sealed outside test"):
        DecisionTrainer({}, records=[*records, _marked(records[0], how="meta", name=name)], log=None)


def test_the_mix_assembly_refuses_a_gate_record_in_any_part(tmp_path):
    from test_v02_mix import _probe_and_sealed

    probe, sealed = _probe_and_sealed(tmp_path)
    records = read_records(REQUESTS)
    script = _script("convert_single_probe.py")
    common = ["mix", "--base", str(REQUESTS), "--probe", str(probe), "--sealed", str(sealed), "--train-config", str(TRAIN_CONFIG)]
    for i, name in enumerate(("v08-b8", "v08-m8g")):
        rows = [json.loads(json.dumps(r)) for r in (records[0], _marked(records[1], how="id", name=name))]
        for j, row in enumerate(rows):
            row["id"] = row["id"] if str(row["id"]).startswith(("v08b8-", "v08m8g-")) else f"xpart{i}-{j:04d}"
            row["meta"] = {**(row.get("meta") or {}), "split": "train", "language": "en"}
        part = tmp_path / f"poisoned{i}.jsonl"
        part.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
        with pytest.raises(SystemExit, match="sealed outside test"):
            script.main([*common, "--extra", f"mujoco={part}", "--out", str(tmp_path / f"a{i}.jsonl")])


def test_the_api_paths_refuse_to_send_a_gate_record():
    from robo_jev.e1 import rewrite as rw

    records = read_records(REQUESTS)
    index = rw.SealedIndex(set(), set(), {})
    rw.assert_sendable(records[0], [records[0]["request"]["state"]], index)
    bench = _script("bench.py")
    bench.refuse_never_sent({"T2": [records[0]]})
    for name in ("v08-b8", "v08-m8g"):
        for how in ("meta", "id"):
            marked = _marked(records[0], how=how, name=name)
            with pytest.raises(rw.SealedTextError, match="never sent"):
                rw.assert_sendable(marked, [marked["request"]["state"]], index)
            with pytest.raises(SystemExit, match="never sent"):
                bench.refuse_never_sent({"T2": [records[1], marked]})


def test_the_build_script_never_writes_into_the_sealed_or_earlier_folders():
    script = _script("v08_gates_build.py")
    script.refuse_write(REPO / "artifacts" / "v08" / "gates" / "x.json")
    for bad in ("t9/T9.jsonl", "v06/a4/x.json", "v07/gates/x.json", "iface/x.json", "simgen/v08/train-v08.jsonl", "reports/v08/x.json", "b1/x", "t8/x"):
        with pytest.raises(SystemExit, match="never written"):
            script.refuse_write(REPO / "artifacts" / bad)


# --------------------------------------------------------------------------
# The built files (when present)
# --------------------------------------------------------------------------


@pytest.mark.skipif(not (G.M8G_DIR / "m8g.manifest.json").is_file(), reason="the built gate is not here")
def test_the_built_m8g_is_sealed_matches_its_manifest_and_the_jev_set_is_independent():
    from robo_jev.decisions.records import file_sha256

    manifest = json.loads((G.M8G_DIR / "m8g.manifest.json").read_text(encoding="utf-8"))
    assert file_sha256(REPO / manifest["file"]["path"]) == manifest["file"]["sha256"]
    records = read_records(REPO / manifest["file"]["path"])
    assert len(records) == manifest["file"]["records"] and all(sealed_record_set(r) == "v08-m8g" for r in records)
    assert len({r["id"] for r in records}) == len(records)
    assert all(r["meta"]["option_range"] == manifest["option_range"] for r in records)
    jev = G.JEV_DIR / "jev-set.jsonl"
    if jev.is_file():
        rows = read_records(jev)
        assert not any(sealed_record_set(r) for r in rows)
        assert not {r["meta"]["item"] for r in rows if r["meta"].get("part") == "many"} & {r["meta"]["item"] for r in records}


@pytest.mark.skipif(not (G.M8G_DIR / "m8g.manifest.json").is_file(), reason="the built gate is not here")
def test_the_chain_check_refuses_a_placeholder_name_and_a_missing_recheck(monkeypatch, tmp_path):
    script = _script("v08_gates_build.py")
    monkeypatch.setattr(G, "OPTION_RANGE_VERSION_CONFIRMED", False)
    problems, _ = script.gate_check([], G.GATES)
    assert any("placeholder" in p for p in problems)
    if not G.OVERLAP_V08.is_file():
        assert any("overlap-v08.json is missing" in p for p in problems)


# --------------------------------------------------------------------------
# R14 — robo-jev-web's collected bench requests under robo-jev-serve's conditions
# --------------------------------------------------------------------------

SERVE_SRC = REPO / "artifacts" / "scratch" / "v08b" / "serve" / "src"


def _r14_line(env: str, state: str, questions: dict, model: str = "robojev") -> dict:
    return {"env": env, "route": "/x", "status": 200, "asked": model, "replyModel": None, "request": {"state": state, "questions": questions, "model": model}}


def test_r14_dedup_and_samples_follow_serves_appendix_b():
    from robo_jev.bench import r14_v08 as R

    q = {"a": {"type": "choice", "instructions": "x?", "criteria": {"k1": "one", "k2": "two"}}}
    lines = [_r14_line("lab", "s1", q), _r14_line("lab", "s1", q), _r14_line("lab", "s1", q, model="jev"), _r14_line("lab", "s2", q)]
    assert [l["request"]["state"] for l in R.dedup(lines)] == ["s1", "s1", "s2"], "the whole request (model included) is the key"
    assert R.serve_sample(list(range(116))) == list(range(0, 100, 5)) and len(R.serve_sample(list(range(640)))) == 20
    assert R.serve_sample(list(range(10))) == list(range(10))
    even = R.even_sample(list(range(116)), 40)
    assert len(even) == 40 and even[0] == 0 and even[-1] == (39 * 116) // 40 and len(set(even)) == 40
    assert R.even_sample(list(range(5)), 40) == list(range(5))


def test_r14_adapter_replicates_serves_trivial_split_and_score_to_level_choice():
    from robo_jev.bench import r14_v08 as R

    questions = {"one": {"type": "choice", "instructions": "only?", "criteria": {"only": "the only way"}},
                 "move": {"type": "choice", "instructions": "move?", "criteria": {"go": "go", "wait": "wait"}},
                 "offset": {"type": "score", "instructions": "offset?", "criteria": ["far left", "left", "centre", "right", "far right"]},
                 "flag": {"type": "noul", "instructions": "safe?"}, "lone": {"type": "score", "instructions": "s?", "criteria": ["only"]}}
    mq = R.model_questions(questions)
    assert list(mq["asked"]) == ["move", "offset"] and mq["trivial"] == ["one"] and mq["converted"] == ["offset"]
    assert mq["asked"]["offset"]["criteria"] == {"0": "far left", "1": "left", "2": "centre", "3": "right", "4": "far right"}
    assert set(mq["skipped"]) == {"flag", "lone"}
    if (SERVE_SRC / "robo_jev_serve" / "scores.py").is_file():  # serve's own code (read-only clone): the same split and conversion
        sys.path.insert(0, str(SERVE_SRC))
        try:
            from robo_jev_serve import scores, trivial

            rest, single = trivial.split(questions)
            asked, converted = scores.to_choice(rest)
            mine_rest, mine_single = R.split_trivial(questions)
            mine_asked, mine_converted = R.to_level_choice(mine_rest)
            assert (rest, single, asked, converted) == (mine_rest, mine_single, mine_asked, mine_converted)
        finally:
            sys.path.remove(str(SERVE_SRC))


def test_r14_planner_is_read_from_the_requests_local_plan():
    from robo_jev.bench import r14_v08 as R

    lab = "LOCAL PLAN:\nThe recommended move: wait. Wait: hold.\nThe recommended reader command: none.\nThe recommended correction: taught, the camera…"
    assert R.planner_of("lab", lab) == {"move": "wait", "reader": "none", "correction": "taught"}
    assert R.planner_of("delivery", "LOCAL PLAN: Executor manoeuvre now: stop (planner recommends: follow). Available now: follow") == {"manoeuvre": "follow"}
    assert R.planner_of("cafe", "Executor intent now: wait (planner recommends: serve_9).") == {"intent": "serve_9"}
    assert R.planner_of("humanoid", "Executor manoeuvre now: x (planner recommends: crouch_now).") == {"maneuver": "crouch_now"}
    assert R.planner_of("forklift", "nothing here") == {}


def test_r14_tag_rules_come_after_rel07s_and_b8s_and_classify_the_new_texts():
    from robo_jev.bench import b8_v08
    from robo_jev.bench import r14_v08 as R

    assert R.RULES[: len(b8_v08.RULES)] == list(b8_v08.RULES)
    cases = {"Do not move the feet this second.": "PASSIVE", "Say nothing.": "PASSIVE", "\"Coming!\" — to the person you attend to.": "ACTIVE",
             "nothing": "PASSIVE", "on the walking line": "PASSIVE", "0.4 m to the left of it": "ACTIVE", "Hold the mast height.": "PASSIVE",
             "Brake to a standstill (lift/lower the load, wait, or the task is done).": "PASSIVE", "Proceed at the plan's nominal speed.": "ACTIVE",
             "Stand still this second and keep watching.": "PASSIVE", "Cross the forklift lane now, straight across.": "ACTIVE",
             "Send the reader nothing now.": "PASSIVE", "Start a read of the plate in the reader.": "ACTIVE", "Halt the move under way where it is.": "PASSIVE",
             "Look at the goal: the carton, or the drop slot.": "ACTIVE"}
    for text, cls in cases.items():
        assert R.tag(text)[1] == cls, text
    with pytest.raises(ValueError, match="no rule"):
        R.tag("Juggle three oranges.")


def test_r14_is_sealed_for_training_but_not_never_sent():
    from robo_jev.decisions.train import DecisionTrainer

    assert "v08-r14" in SEALED_RECORD_SETS and SEALED_RECORD_PREFIXES["v08-r14"] == "v08r14-" and "v08-r14" not in NEVER_SENT_SETS
    assert SEALED_RECORD_FILES["v08-r14"] == "artifacts/v08/r14/R14-all.jsonl"
    records = read_records(REQUESTS)
    marked = dict(records[0], id="v08r14-lab-0001")
    assert sealed_record_set(marked) == "v08-r14"
    with pytest.raises(ValueError, match="sealed outside test"):
        DecisionTrainer({}, records=[*records, marked], log=None)
    _script("bench.py").refuse_never_sent({"T2": [records[0], marked]})  # sendable: the bench has already sent these requests


def test_r14_freeze_check_finds_a_changed_file(tmp_path):
    from robo_jev.bench import r14_v08 as R

    (tmp_path / "a.jsonl").write_text("{}\n", encoding="utf-8")
    (tmp_path / "frozen-before-scoring.json").write_text(json.dumps({"files": {"a.jsonl": R.sha256_file(tmp_path / "a.jsonl")}}), encoding="utf-8")
    assert R.check_frozen(tmp_path) == []
    (tmp_path / "a.jsonl").write_text("{\"x\": 1}\n", encoding="utf-8")
    assert R.check_frozen(tmp_path) == ["a.jsonl changed after it was frozen"]


def test_the_bench_scorer_reads_serves_order_seeds_adapter_and_collected_numbering():
    score = _script("v08_bench_score.py")
    assert score.parse_reading("spaces-2") == {"name": "spaces-2", "orders": "single", "seed": 1, "spaces": 2}
    record = {"id": "r", "meta": {}, "request": {"state": "s", "questions": {
        "one": {"type": "choice", "instructions": "x?", "criteria": {"k": "only"}},
        "offset": {"type": "score", "instructions": "o?", "criteria": ["a", "b", "c"]}, "flag": {"type": "noul", "instructions": "y?"}}}}
    adapted = score.serve_adapted(record)
    assert adapted["meta"]["target"] == ["offset"] and list(adapted["request"]["questions"]) == ["offset"]
    assert adapted["meta"]["serve"] == {"converted": ["offset"], "trivial": ["one"], "skipped": ["flag"]}
    lines = ['{"state": "a", "questions": {}}', "", '{"request": {"state": "b", "questions": {}}}', '{"id": "z", "request": {"state": "c", "questions": {}}}']
    assert [r["id"] for r in score.normalise_collected(lines)] == ["collected-1", "collected-3", "z"]


def test_the_r14_pipeline_check_counts_as_serves_appendix_b():
    check = _script("v08_r14_check.py")
    q = {"type": "choice", "instructions": "x?", "criteria": {"a": "A", "b": "B"}}
    sample = [{"id": f"r{i}", "meta": {"env": "lab"}, "request": {"state": "s", "questions": {"move": q, "lone": {**q, "criteria": {"a": "A"}},
                                                                                              "offset": {"type": "score", "criteria": ["x", "y"]}}}} for i in range(3)]
    runs = {"r0": ["a", "a", "a"], "r1": ["a", "b", "a"], "r2": ["b", "b", "b"]}
    scores = {"answers": {rid: {"move": {o: {"choice": c} for o, c in zip(check.ORDERS, tops)}, "offset": {o: {"choice": "0"} for o in check.ORDERS}}
                          for rid, tops in runs.items()}}
    counted = check.count(sample, scores)
    assert set(counted) == {("lab", "move")}, "only original choice questions with >= 2 options count"
    assert counted[("lab", "move")]["counted"] == 3 and counted[("lab", "move")]["flips"] == 1 and counted[("lab", "move")]["tops"]["a"] == 5


@pytest.mark.skipif(not (REPO / "artifacts" / "v08" / "r14" / "r14.manifest.json").is_file(), reason="R14 is not built here")
def test_the_built_r14_matches_serves_dedup_counts_and_is_frozen():
    from robo_jev.bench import r14_v08 as R

    manifest = json.loads((R.R14_DIR / "r14.manifest.json").read_text(encoding="utf-8"))
    assert {env: c["dedup"] for env, c in manifest["counts"].items()} == {"lab": 116, "cafe": 118, "delivery": 207, "humanoid": 119, "forklift": 640}
    assert R.check_frozen(R.R14_DIR) == []
    with R.R14_ALL.open(encoding="utf-8") as handle:
        first = [json.loads(next(handle)) for _ in range(5)]
    assert all(sealed_record_set(r) == "v08-r14" for r in first)


def test_every_passive_rule_is_split_into_hold_or_continue_and_no_active_rule_is():
    from robo_jev.bench import r14_v08 as R

    passive = {rid for rid, cls, _, _ in R.RULES if cls == "PASSIVE"}
    active = {rid for rid, cls, _, _ in R.RULES if cls == "ACTIVE"}
    assert set(R.PASSIVE_KIND) == passive and set(R.PASSIVE_KIND.values()) == {"hold", "continue"}
    assert all(R.passive_kind(rid) is None for rid in active)
    assert R.passive_kind("let-the-move-go-on") == "continue" and R.passive_kind("wait") == "hold" and R.passive_kind("call-the-operator") == "hold"
