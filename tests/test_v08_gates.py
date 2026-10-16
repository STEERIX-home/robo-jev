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


# --------------------------------------------------------------------------
# A16 / S16 — robo-jev-web's requests of 2026-10-16 (task v0.8c)
# --------------------------------------------------------------------------

ARM_TEXT = ("The gripper still has to travel {d} cm along {a}{up}. Choose this update's step along {a}: plus for a positive distance, minus for a "
            "negative one. Take the largest step that does not overshoot: far (20) when the distance is at least 20 cm, mid (8) when at least 8, "
            "near (3) when at least 3, small (1) when at least 1, fine (0.4) when between 0.2 and 1 cm, and stay when it is within 0.2 cm.")
ARM_OPTIONS = {"plus_far": "+20 cm.", "plus_mid": "+8 cm.", "plus_near": "+3 cm.", "plus_small": "+1 cm.", "plus_fine": "+0.4 cm.", "stay": "No move on this axis.",
               "minus_far": "-20 cm.", "minus_mid": "-8 cm.", "minus_near": "-3 cm.", "minus_small": "-1 cm.", "minus_fine": "-0.4 cm."}


def _arm_line(x: str, y: str, z: str) -> dict:
    questions = {a: {"type": "choice", "instructions": ARM_TEXT.format(d=d, a=a, up=" (up is plus)" if a == "y" else ""), "criteria": dict(ARM_OPTIONS)}
                 for a, d in (("x", x), ("y", y), ("z", z))}
    questions["fingers"] = {"type": "choice", "instructions": "Command the fingers.", "criteria": {"open": "Open.", "close": "Close.", "keep": "Keep."}}
    return {"env": "arm", "status": 200, "request": {"state": '{"protocol":"robojev_web_servo_v1"}', "questions": questions, "model": "robojev"}}


def test_a16_rule_parses_the_distance_and_labels_by_the_thresholds_with_the_boundary_reading():
    from robo_jev.bench import r16_v08 as R

    assert R.arm_distance(ARM_TEXT.format(d="+0.9", a="x", up="")) == ("x", 0.9)
    assert R.arm_distance(ARM_TEXT.format(d="-35.5", a="y", up=" (up is plus)")) == ("y", -35.5)
    assert R.arm_distance(ARM_TEXT.format(d="0", a="z", up="")) == ("z", 0.0)
    with pytest.raises(ValueError):
        R.arm_distance("Command the fingers.")
    cases = {0.0: "stay", 0.19: "stay", 0.2: "plus_fine", -0.2: "minus_fine", 0.9: "plus_fine", 1.0: "plus_small", -2.9: "minus_small",
             3.0: "plus_near", -7.99: "minus_near", 8.0: "plus_mid", 19.9: "plus_mid", 20.0: "plus_far", -82.0: "minus_far"}
    for d, want in cases.items():
        assert R.arm_label(d) == want, d
    assert R.arm_label(-0.2, boundary_stay=True) == "stay" and R.arm_label(0.21, boundary_stay=True) == "plus_fine"
    assert [R.arm_band(d) for d in (0.1, 0.2, 1, 3, 8, 20)] == ["<0.2", "0.2-1", "1-3", "3-8", "8-20", ">=20"]


def test_a16_records_ablation_and_readings_count_agreement_sign_errors_and_stay():
    from robo_jev.bench import r16_v08 as R

    lines = [_arm_line("+0.9", "-35.5", "+82"), _arm_line("-0.2", "-2.5", "-30"), _arm_line("0", "-5", "+10")]
    records = [R.arm_record(i, line, source_line=i + 1) for i, line in enumerate(lines)]
    assert records[0]["id"] == "v08a16-0000" and sealed_record_set(records[0]) == "v08-a16"
    assert records[1]["meta"]["rule"]["x"] == {"distance": -0.2, "label": "minus_fine", "label_boundary_stay": "stay", "band": "0.2-1", "sign": "minus"}
    assert G.gate_problems(records[0]) == []
    ablated = R.ablate_y(records[0])
    assert list(ablated["request"]["questions"]) == ["y"] and "(up is plus)" not in ablated["request"]["questions"]["y"]["instructions"]
    assert ablated["id"] == records[0]["id"] and ablated["request"]["questions"]["y"]["instructions"].startswith("The gripper still has to travel -35.5 cm along y.")
    with pytest.raises(ValueError):
        R.ablate_y(ablated | {"request": {"state": "s", "questions": {"y": {"instructions": "no phrase"}}}})
    tops = {"v08a16-0000": {"x": "stay", "y": "plus_far", "z": "plus_far"},          # x: stay on 0.9 (small); y: sign error; z: agree
            "v08a16-0001": {"x": "stay", "y": "minus_small", "z": "minus_far"},      # x: stay on -0.2 (agree only with boundary_stay)
            "v08a16-0002": {"x": "stay", "y": "minus_near", "z": "minus_mid"}}       # x: agree (0 -> stay); z: sign error
    answers = {rid: {a: {"single-s1": {"choice": c, "probabilities": {c: 0.6}}} for a, c in q.items()} for rid, q in tops.items()}
    s = R.arm_summary(R.arm_rows(records, answers, reading="single-s1"))
    assert s["agreement"]["count"] == 5 and s["agreement"]["n"] == 9
    assert s["sign_errors_by_axis"]["y"]["count"] == 1 and s["sign_errors_by_axis"]["z"]["count"] == 1 and s["sign_errors"]["n"] == 6, "rows where rule and model both move"
    assert s["stay_rate_small"]["count"] == 2 and s["stay_rate_small"]["n"] == 3, "x 0.9 and -0.2 (0.2-1) stay; y -2.5 (1-3) moves"
    alt = R.arm_summary(R.arm_rows(records, answers, reading="single-s1", boundary_stay=True))
    assert alt["agreement_by_axis"]["x"]["count"] == 2
    other = {rid: {a: {"single-s1": {"choice": "stay", "probabilities": {"stay": 0.5}}} for a in "xyz"} for rid in tops}
    arm = R.seed_mean_rows(R.arm_rows(records, answers, reading="single-s1"), R.arm_rows(records, other, reading="single-s1"))
    by = {(r["id"], r["axis"]): r for r in arm}
    assert by[("v08a16-0000", "y")]["sign_error"] == 1.0 and by[("v08a16-0000", "z")]["agree"] == 0.5 and by[("v08a16-0000", "x")]["top_is_stay"] == 1.0


def test_a16_stored_answers_must_be_the_same_calls():
    from robo_jev.bench import r16_v08 as R

    records = [R.arm_record(0, _arm_line("+0.9", "-35.5", "+82"), source_line=1)]
    servo = [{"request": {"facts": {"move": {"x": 0.9, "y": -35.5, "z": 82}}}, "answers": {"x": {"choice": "stay", "probabilities": {"stay": 1.0}}}}]
    assert R.stored_answers(records, servo)["v08a16-0000"]["x"]["stored"]["choice"] == "stay"
    servo[0]["request"]["facts"]["move"]["y"] = -30.0
    with pytest.raises(ValueError, match="not the record"):
        R.stored_answers(records, servo)


def test_s16_groups_and_sealing_of_the_2026_10_16_sets():
    from robo_jev.bench import r16_v08 as R

    line = {"env": "soccer", "request": {"state": "pitch", "questions": {"red1": {"type": "choice", "instructions": "move?", "criteria": {"a": "A", "b": "B"}},
                                                                        "red1_look": {"type": "choice", "instructions": "look?", "criteria": {"l": "L", "r": "R"}}}}}
    record = R.soccer_record(3, line, source_line=4)
    assert record["id"] == "v08s16-0003" and record["meta"]["groups"] == {"red1": "movement", "red1_look": "gaze"}
    for name, prefix, path in (("v08-a16", "v08a16-", "artifacts/v08/r16/A16.jsonl"), ("v08-s16", "v08s16-", "artifacts/v08/r16/S16.jsonl")):
        assert name in SEALED_RECORD_SETS and SEALED_RECORD_PREFIXES[name] == prefix and SEALED_RECORD_FILES[name] == path and name not in NEVER_SENT_SETS
    assert sealed_record_set(record) == "v08-s16"
    _script("bench.py").refuse_never_sent({"T2": [record]})  # sendable: the bench has already sent these requests


def test_the_a16_check_compares_tops_and_reports_serves_numbers(tmp_path):
    from robo_jev.bench import r16_v08 as R

    check = _script("v08_a16_check.py")
    records = [R.arm_record(0, _arm_line("+0.9", "-35.5", "+82"), source_line=1)]
    ours = {"v08a16-0000": {"x": {"single-s1": {"choice": "stay", "probabilities": {"stay": 0.5, "plus_fine": 0.5}}}}}
    stored = {"v08a16-0000": {"x": {"stored": {"choice": "plus_fine", "probabilities": {"stay": 0.49, "plus_fine": 0.51}}}}}
    got = check.compare_tops(records, ours, stored, reading="single-s1")
    assert got["total"] == {"n": 1, "same_top": 0, "max_abs_dp": pytest.approx(0.01)}
    assert check.SERVE_V07["agreement"]["total"] == [116, 270] and check.SERVE_V07["ablation_y"]["sign_errors"] == {"before": 27, "after": 4}


@pytest.mark.skipif(not (REPO / "artifacts" / "v08" / "r16" / "r16.manifest.json").is_file(), reason="A16 / S16 are not built here")
def test_the_built_a16_reproduces_serves_stored_v07_agreement_and_is_frozen():
    from robo_jev.bench import r16_v08 as R

    assert R.check_frozen(R.R16_DIR) == []
    records = read_records(R.R16_DIR / "A16.jsonl")
    stored = json.loads((R.R16_DIR / "A16-stored-v07.json").read_text(encoding="utf-8"))["answers"]
    s = R.arm_summary(R.arm_rows(records, stored, reading="stored"))
    assert [int(s["agreement"]["count"]), *[int(s["agreement_by_axis"][a]["count"]) for a in "xyz"]] == [116, 1, 28, 87]
    assert len(records) == 90 and len(read_records(R.R16_DIR / "S16.jsonl")) == 55


# --------------------------------------------------------------------------
# A16s — the arm's distance sweep (task v0.8d)
# --------------------------------------------------------------------------

INTENT_TEXT = ("Name the stage the robot is in. Facts: holding=false, target=object_grasp_point, at_target=false, placed=false, height above "
               "table +55.5 cm, remaining travel x +0.9, y -35.5, z +82 cm.")


def _a16s_templates(n: int = 4) -> list[dict]:
    """A16-like records (the recorded constant state; intent first, then x, y, z with " (up is plus)" on y, then fingers)."""
    from robo_jev.bench import r16_v08 as R

    out = []
    for i in range(n):
        line = _arm_line("+0.9", "-35.5", "+82")
        intent = {"type": "choice", "instructions": INTENT_TEXT, "criteria": {"look": "Look.", "approach": "Approach.", "grasp": "Grasp."}}
        line["request"]["questions"] = {"intent": intent, **line["request"]["questions"]}
        out.append(R.arm_record(i, line, source_line=i + 1))
    return out


def test_a16s_grid_formats_labels_and_bands_follow_the_questions_rule():
    from robo_jev.bench import a16s_v08 as S
    from robo_jev.bench import r16_v08 as R

    values = S.sweep_values()
    assert len(values) == 54 and len(set(values)) == 54 and sorted(values) == sorted(-v for v in values)
    assert not {abs(v) for v in values} & set(S.BOUNDARIES), "never an exact boundary"
    assert [S.format_distance(d, "web") for d in (12, -12, 0.15, -0.95, 40)] == ["+12", "-12", "+0.15", "-0.95", "+40"]
    assert [S.format_distance(d, "plain") for d in (12, -12, 0.15, -0.95, 40)] == ["12.0", "-12.0", "0.15", "-0.95", "40.0"]
    for d in values:
        for kind in S.FORMATS:
            assert float(S.format_distance(d, kind)) == d
        assert S.label(d) == R.arm_label(d) == R.arm_label(d, boundary_stay=True), "the grid has no value where the two readings differ"
    with pytest.raises(ValueError):
        S.format_distance(1.0, "words")
    cases = {0.1: "stay", -0.15: "stay", 0.3: "plus_fine", -0.95: "minus_fine", 1.05: "plus_small", -2.95: "minus_small", 3.05: "plus_near",
             7.9: "plus_near", 8.1: "plus_mid", -19.9: "minus_mid", 20.1: "plus_far", -40.0: "minus_far"}
    assert {d: S.label(d) for d in cases} == cases
    bands = {0.95: "<1", -0.1: "<1", 1.05: "1-3", 2.95: "1-3", 3.05: "3-8", 7.9: "3-8", 8.1: "8-20", -19.9: "8-20", 20.1: "20-25", 25.0: ">=25",
             -40.0: ">=25"}
    assert {d: S.read_band(d) for d in bands} == bands


def test_a16s_build_covers_each_value_axis_and_format_once_and_rewrites_only_the_distances():
    from collections import Counter

    from robo_jev.bench import a16s_v08 as S
    from robo_jev.bench import r16_v08 as R

    templates = _a16s_templates()
    records = S.build_records(templates, seed=7)
    assert len(records) == 108 and [r["id"] for r in records[:2]] == ["v08a16s-0000", "v08a16s-0001"] and records[-1]["id"] == "v08a16s-0107"
    assert Counter(r["meta"]["format"] for r in records) == {"web": 54, "plain": 54}
    seen = Counter((r["meta"]["sweep"][a]["distance"], a, r["meta"]["format"]) for r in records for a in S.AXES)
    assert len(seen) == 324 and set(seen.values()) == {1}
    for k, record in enumerate(records):
        template = templates[k % len(templates)]
        assert record["meta"]["template"] == template["id"] and record["meta"]["target"] == ["x", "y", "z"]
        assert sealed_record_set(record) == "v08-a16s" and G.gate_problems(record) == []
        assert record["request"]["state"] == template["request"]["state"]
        q, tq = record["request"]["questions"], template["request"]["questions"]
        assert list(q) == list(tq) and q["fingers"] == tq["fingers"]
        assert q["intent"]["criteria"] == tq["intent"]["criteria"]
        for axis in S.AXES:
            sweep = record["meta"]["sweep"][axis]
            assert R.arm_distance(q[axis]["instructions"]) == (axis, sweep["distance"])
            assert sweep["text"] == S.format_distance(sweep["distance"], record["meta"]["format"]) and sweep["label"] == S.label(sweep["distance"])
            assert q[axis]["criteria"] == tq[axis]["criteria"]
            expected = tq[axis]["instructions"].replace(R.UP_IS_PLUS, "").replace(f"travel {R.arm_distance(tq[axis]['instructions'])[1]:+g} cm",
                                                                                  f"travel {sweep['text']} cm")
            assert q[axis]["instructions"] == expected, "only the number (and on y the phrase) changes"
        assert "(up is plus)" not in q["y"]["instructions"]
        s = record["meta"]["sweep"]
        assert q["intent"]["instructions"] == INTENT_TEXT.replace("remaining travel x +0.9, y -35.5, z +82 cm",
                                                                  f"remaining travel x {s['x']['text']}, y {s['y']['text']}, z {s['z']['text']} cm")
    assert S.build_records(templates, seed=7) == records
    assert S.build_records(templates, seed=8) != records
    broken = json.loads(json.dumps(templates[0]))
    broken["request"]["questions"]["intent"]["instructions"] = "Name the stage."
    with pytest.raises(ValueError, match="remaining travel"):
        S.build_records([broken])
    no_phrase = json.loads(json.dumps(templates[0]))
    no_phrase["request"]["questions"]["y"]["instructions"] = no_phrase["request"]["questions"]["y"]["instructions"].replace(R.UP_IS_PLUS, "")
    with pytest.raises(ValueError, match="up is plus"):
        S.build_records([no_phrase])


def _a16s_answers(records: list[dict], pick, reading: str = "single-s1") -> dict:
    """answers[id][axis][reading] = {choice, probabilities} with choice = pick(label, distance) and P(choice) = 0.6, P(label) = 0.6 or 0.2."""
    out: dict = {}
    for r in records:
        for axis in ("x", "y", "z"):
            sweep = r["meta"]["sweep"][axis]
            choice = pick(sweep["label"], sweep["distance"])
            probs = {choice: 0.6} if choice == sweep["label"] else {choice: 0.6, sweep["label"]: 0.2}
            out.setdefault(r["id"], {}).setdefault(axis, {})[reading] = {"choice": choice, "probabilities": probs}
    return out


def test_a16s_readings_count_accuracy_stay_sign_errors_flips_and_curves():
    from robo_jev.bench import a16s_v08 as S

    records = S.build_records(_a16s_templates(), seed=7)
    always_stay = _a16s_answers(records, lambda lab, d: "stay")
    rows = S.sweep_rows(records, always_stay, second=None)
    assert len(rows) == 324 and all(r["flip"] is None and r["sign_error"] is None for r in rows)
    s = S.sweep_summary(rows)
    assert s["rows"] == 324 and s["total"]["stay"]["value"] == 1.0
    assert s["total"]["accuracy"]["value"] == pytest.approx(24 / 324), "only the 24 stay labels (|d| 0.1 and 0.15) are right"
    assert s["by_band"]["<1"]["accuracy"]["value"] == pytest.approx(24 / 72) and s["by_band"][">=25"]["accuracy"]["n"] == 36
    assert {"<1|+", "<1|web", "<1|-|plain", "8-20|+|web"} <= set(s["by_band"])
    # the far side right, the rest the opposite sign: sign errors on every moving non-far question
    def pick(lab, d):
        return lab if abs(d) >= 20 else ("stay" if lab == "stay" else ("minus" if d > 0 else "plus") + lab[lab.index("_"):])
    first = _a16s_answers(records, pick)
    second = _a16s_answers(records, lambda lab, d: lab if d > 0 else pick(lab, d), reading="single-s2")
    both = {rid: {a: {**first[rid][a], **second[rid][a]} for a in first[rid]} for rid in first}
    rows = S.sweep_rows(records, both)
    s = S.sweep_summary(rows)
    assert s["by_band"][">=25"]["accuracy"]["value"] == 1.0 and s["by_band"]["20-25"]["accuracy"]["value"] == 1.0
    assert s["by_band"]["3-8"]["sign_errors"]["value"] == 1.0 and s["by_band"]["3-8"]["accuracy"]["value"] == 0.0
    assert s["by_band"]["3-8|+"]["flips"]["value"] == 1.0 and s["by_band"]["3-8|-"]["flips"]["value"] == 0.0
    assert s["by_band"]["3-8"]["p_correct"]["value"] == pytest.approx(0.2) and s["by_band"][">=25"]["p_correct"]["value"] == pytest.approx(0.6)
    arm = S.seed_mean_rows(S.sweep_rows(records, first), S.sweep_rows(records, second, reading="single-s2", second=None))
    by = {(r["distance"], r["axis"], r["format"]): r for r in arm}
    assert by[(5.0, "x", "web")]["correct"] == 0.5 and by[(5.0, "x", "web")]["sign_error"] == 0.5 and by[(-5.0, "x", "web")]["correct"] == 0.0
    c = S.curves(rows)["points"]
    assert len(c) == 54 and {p["n"] for p in c} == {6} and [p["distance"] for p in c] == sorted(S.sweep_values())
    point = next(p for p in c if p["distance"] == 30.0)
    assert point["accuracy"] == point["accuracy_web"] == point["accuracy_plain"] == 1.0


def test_a16s_curves_svg_is_well_formed_with_the_base_dashed_and_every_point_hoverable():
    import xml.etree.ElementTree as ET

    from robo_jev.bench import a16s_v08 as S

    records = S.build_records(_a16s_templates(), seed=7)
    rows = S.sweep_rows(records, _a16s_answers(records, lambda lab, d: lab if abs(d) >= 25 else "stay"), second=None)
    curve = S.curves(rows)
    svg = S.curves_svg({"F": curve, "FP": curve, "v07": curve, "base": curve}, seeds={"F": [curve, curve]})
    root = ET.fromstring(svg)
    ns = "{http://www.w3.org/2000/svg}"
    assert "&lt; fine &lt; 1" in svg, "the axis caption's < is escaped"
    lines = root.findall(f"{ns}polyline")
    assert any(p.get("class") == "c-base" and p.get("stroke-dasharray") for p in lines) and not any(
        p.get("class") == "c-F" and p.get("stroke-dasharray") for p in lines)
    assert len(root.findall(f"{ns}circle")) == 4 * 54 * 2, "a hover target per series, point and panel"
    texts = [t.text or "" for t in root.iter(f"{ns}text")]
    assert {"untrained base", "v0.8 F (format)", "v0.7"} <= set(texts), "a legend entry per series"
    assert "@media (prefers-color-scheme: dark)" in svg
    single = ET.fromstring(S.curves_svg({"v07": curve}))
    assert len(single.findall(f"{ns}circle")) == 54 * 2


def test_a16s_is_sealed_for_training_and_its_freeze_check_finds_a_changed_file(tmp_path):
    from robo_jev.bench import a16s_v08 as S

    assert "v08-a16s" in SEALED_RECORD_SETS and SEALED_RECORD_PREFIXES["v08-a16s"] == "v08a16s-"
    assert SEALED_RECORD_FILES["v08-a16s"] == "artifacts/v08/a16s/A16s.jsonl" and "v08-a16s" not in NEVER_SENT_SETS
    with pytest.raises(ValueError, match="v08-a16s"):
        refuse_sealed_records([{"id": "v08a16s-0003", "meta": {}}], where="the trainer")
    assert S.check_frozen(tmp_path) == [f"missing frozen-before-scoring.json in {tmp_path}"]
    (tmp_path / "A16s.jsonl").write_text("{}\n", encoding="utf-8")
    (tmp_path / "frozen-before-scoring.json").write_text(json.dumps({"files": {"A16s.jsonl": S.sha256_file(tmp_path / "A16s.jsonl")}}),
                                                         encoding="utf-8")
    assert S.check_frozen(tmp_path) == []
    (tmp_path / "A16s.jsonl").write_text("{\"x\": 1}\n", encoding="utf-8")
    assert S.check_frozen(tmp_path) == ["A16s.jsonl changed after it was frozen"]


def test_serves_adapter_keeps_a_records_own_target():
    from robo_jev.bench import a16s_v08 as S

    score = _script("v08_bench_score.py")
    record = S.build_records(_a16s_templates(), seed=7)[0]
    adapted = score.serve_adapted(record)
    assert adapted["meta"]["target"] == ["x", "y", "z"] and list(adapted["request"]["questions"]) == ["x", "y", "z"]
    assert adapted["request"]["questions"]["y"] == record["request"]["questions"]["y"] and adapted["meta"]["sweep"] == record["meta"]["sweep"]
    whole = score.serve_adapted({**record, "meta": {k: v for k, v in record["meta"].items() if k != "target"}})
    assert whole["meta"]["target"] == ["intent", "x", "y", "z", "fingers"], "no own target: every question serve asks"


def test_the_a16s_check_reads_serves_pattern_and_writes_the_table_and_figure(tmp_path):
    import xml.etree.ElementTree as ET

    from robo_jev.bench import a16s_v08 as S

    check = _script("v08_a16s_check.py")
    records = S.build_records(_a16s_templates(), seed=7)
    (tmp_path / "A16s.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
    (tmp_path / "frozen-before-scoring.json").write_text(json.dumps({"files": {"A16s.jsonl": S.sha256_file(tmp_path / "A16s.jsonl")}}),
                                                         encoding="utf-8")
    answers = _a16s_answers(records, lambda lab, d: lab if abs(d) >= 25 else "stay")
    (tmp_path / "scores.json").write_text(json.dumps({"answers": answers}), encoding="utf-8")
    assert check.main(["--records", str(tmp_path / "A16s.jsonl"), "--scores", f"v07={tmp_path / 'scores.json'}", "--out", str(tmp_path / "c.json")]) == 0
    got = json.loads((tmp_path / "c.json").read_text(encoding="utf-8"))["series"]["v07"]["serve_pattern"]
    assert got["y"]["stay_-20..+15"] == 1.0 and got["y"]["accuracy_>=25"] == 1.0 and got["y"]["p_correct_-25..-40"] == pytest.approx(0.6)
    ET.parse(tmp_path / "c.svg")
    assert "| 40 | 1.00 | 1.00 |" in (tmp_path / "c.md").read_text(encoding="utf-8")
    (tmp_path / "A16s.jsonl").write_text("", encoding="utf-8")
    with pytest.raises(SystemExit, match="changed after it was frozen"):
        check.main(["--records", str(tmp_path / "A16s.jsonl"), "--scores", f"v07={tmp_path / 'scores.json'}", "--out", str(tmp_path / "c.json")])


@pytest.mark.skipif(not (REPO / "artifacts" / "v08" / "a16s" / "frozen-before-scoring.json").is_file(), reason="A16s is not built here")
def test_the_built_a16s_is_frozen_complete_and_derived_from_a16():
    from collections import Counter

    from robo_jev.bench import a16s_v08 as S
    from robo_jev.bench import r16_v08 as R

    assert S.check_frozen(S.A16S_DIR) == []
    assert S.sha256_file(S.A16S_DIR / "A16s.jsonl") == "e05ebb9384aef75c199ae6609fbc7346ebda215451f05dd27010dfa38bd0a787"
    records = read_records(S.A16S_DIR / "A16s.jsonl")
    assert records == S.build_records(read_records(R.R16_DIR / "A16.jsonl"), seed=S.SEED), "rebuilds byte-for-byte from A16"
    labels = Counter(r["meta"]["sweep"][a]["label"] for r in records for a in S.AXES)
    assert len(records) == 108 and sum(labels.values()) == 324 and labels["stay"] == 24 and labels["plus_far"] == labels["minus_far"] == 24
