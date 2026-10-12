"""General-reading training mix (Task v0.2 §1-1): public train rows → judgment-v0 → decisions records.

Hand-made rows pin the conversion of each source (state fields, one choice question, seeded option permutation with keys ``o1 … oN``,
one-hot label on the gold answer, dropped rows), the overlap units and the build (sealed overlap dropped, seeded sample, balanced
interleaving). When the built file ``artifacts/simgen/v02/general-train.jsonl`` exists, it is checked against its manifest, its
licences and exclusions, the one-hot labels, and — recomputed from the files — zero text-unit overlap with G1, G2, T1–T7 and T8 (when
T8 exists); the trainer's loader and example builder and the diagnosis tool read it as is.
"""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path

import pytest
from helpers import REPO

from robo_jev.contracts import validate_record as validate_source
from robo_jev.data import general_sets as gs
from robo_jev.decisions.records import file_sha256, read_records, validate_record

GENERAL = REPO / "artifacts" / "simgen" / "v02" / "general-train.jsonl"
MANIFEST = GENERAL.with_name("general-train.manifest.json")
PROBES = REPO / "artifacts" / "simgen" / "probes"
TESTS = REPO / "artifacts" / "simgen" / "v0b" / "tests"
T8_DIR = REPO / "artifacts" / "simgen" / "t8"
PERMISSIVE = {"MIT", "Apache-2.0", "CC-BY-4.0"}
G2_INSTRUCTIONS = {"Given the passage, is the answer to the question yes?", "What is the relation of the hypothesis to the premise?"}


def _rows() -> dict[str, list[dict]]:
    """Two rows per source in the parquet shapes (the second row of each has a different gold)."""
    return {
        "commonsense_qa": [
            {"id": "a1", "question": "Where would you keep a bottle of milk cold?", "question_concept": "milk",
             "choices": {"label": list("ABCDE"), "text": ["oven", "refrigerator", "garden", "attic", "car roof"]}, "answerKey": "B"},
            {"id": "a2", "question": "What do people use to cut paper?", "question_concept": "paper",
             "choices": {"label": list("ABCDE"), "text": ["scissors", "spoon", "pillow", "cup", "sock"]}, "answerKey": "A"},
        ],
        "hellaswag": [
            {"ind": 1, "activity_label": "Baking", "ctx": "A woman pours flour into a bowl. She", "endings": ["adds eggs and stirs.", "flies away.",
             "paints the bowl blue.", "sings to the oven."], "label": "0"},
            {"ind": 2, "activity_label": "Rowing", "ctx": "Two men push a boat into the water. They", "endings": ["eat the oars.", "climb in and start rowing.",
             "bury the boat.", "fold the river."], "label": "1"},
        ],
        "social_i_qa": [
            {"context": "Kai forgot his friend's birthday party last night.", "question": "How would Kai feel afterwards?", "answerA": "sorry",
             "answerB": "hungry", "answerC": "taller", "label": "1"},
            {"context": "Sasha gave her seat on the bus to an old man.", "question": "Why did Sasha do this?", "answerA": "to sleep",
             "answerB": "to win money", "answerC": "to be kind", "label": "3"},
        ],
        "qasc": [
            {"id": "q1", "question": "What do plants need to make food?", "choices": {"label": list("ABCDEFGH"),
             "text": ["sunlight", "darkness", "plastic", "rocks", "noise", "metal", "glass", "sand"]}, "answerKey": "A",
             "fact1": "plants make food by photosynthesis", "fact2": "photosynthesis needs sunlight"},
            {"id": "q2", "question": "What carries blood through the body?", "choices": {"label": list("ABCDEFGH"),
             "text": ["roots", "vessels", "wires", "pipes of steel", "wind", "leaves", "shells", "feathers"]}, "answerKey": "B",
             "fact1": "the heart pumps blood", "fact2": "blood moves in vessels"},
        ],
        "cosmos_qa": [
            {"id": "c1", "context": "We missed the last train, so we walked home in the rain and arrived soaked.", "question": "Why were they wet?",
             "answer0": "They walked in the rain.", "answer1": "They swam.", "answer2": "They washed the car.", "answer3": "None of the above choices .", "label": 0},
            {"id": "c2", "context": "My phone died during the call, so I borrowed my sister's charger.", "question": "What happened to the phone?",
             "answer0": "It was stolen.", "answer1": "Its battery ran out.", "answer2": "It got wet.", "answer3": "None of the above choices .", "label": 1},
        ],
    }  # fmt: skip


GOLD = {
    "commonsense_qa": ["refrigerator", "scissors"], "hellaswag": ["adds eggs and stirs.", "climb in and start rowing."],
    "social_i_qa": ["sorry", "to be kind"], "qasc": ["sunlight", "vessels"], "cosmos_qa": ["They walked in the rain.", "Its battery ran out."],
}  # fmt: skip


@pytest.mark.parametrize("name", sorted(gs.SOURCES))
def test_each_source_row_becomes_one_choice_question_with_a_one_hot_label_on_the_gold_answer(name):
    records, counts = gs.convert_source(name, _rows()[name])
    assert counts["rows"] == 2 and counts["kept"] == 2 and len(records) == 2
    for index, record in enumerate(records):
        assert validate_record(record) == [], record["id"]
        assert record["id"] == f"{name}-train-{index:05d}" and record["meta"]["split"] == "train" and record["meta"]["domain"] == f"public:{name}"
        assert record["meta"]["source"] == {"set": name, "row": index} and record["meta"]["origin_group"] == f"public/{name}/train/{index:05d}"
        (question_name,) = record["request"]["questions"]
        question = record["request"]["questions"][question_name]
        assert question["type"] == "choice" and question["instructions"] in gs.INSTRUCTIONS[name]
        keys = list(question["criteria"])
        assert keys == [f"o{i + 1}" for i in range(len(keys))]
        label = record["labels"][question_name]
        assert label["kind"] == "probe" and sorted(label["probabilities"].values()) == [0.0] * (len(keys) - 1) + [1.0]
        gold = next(k for k, p in label["probabilities"].items() if p == 1.0)
        assert question["criteria"][gold] == GOLD[name][index], (name, index)
        assert not any(line.startswith(tuple(keys)) for line in record["request"]["state"].split("\n"))  # the state carries no option keys
    source = gs.convert_row(name, 0, _rows()[name][0])
    validate_source(source)
    assert source["split"] == "train" and source["provenance"]["generator"] == gs.GENERAL_VERSION


def test_the_option_order_is_permuted_per_row_so_the_gold_key_carries_no_information():
    row = _rows()["commonsense_qa"][0]
    golds = Counter()
    orders = set()
    for index in range(400):
        record = gs.convert_row("commonsense_qa", index, row)
        criteria = record["request"]["questions"][0]["criteria"]
        orders.add(tuple(c["description"] for c in criteria))
        golds[record["labels"][0]["answer"]] += 1
    assert len(orders) > 50 and set(golds) == {f"o{i}" for i in range(1, 6)}
    assert all(40 <= n <= 120 for n in golds.values()), golds  # ≈ 80 each
    again = gs.convert_row("commonsense_qa", 7, row)
    assert again == gs.convert_row("commonsense_qa", 7, row)  # deterministic


def test_bad_rows_are_dropped_and_counted():
    rows = _rows()["commonsense_qa"]
    no_gold = {**rows[0], "answerKey": "Z"}
    duplicate = copy.deepcopy(rows[1])
    duplicate["choices"]["text"] = ["scissors", "Scissors!", "pillow", "cup", "sock"]
    empty = copy.deepcopy(rows[1])
    empty["choices"]["text"] = ["scissors", "", "pillow", "cup", "sock"]
    records, counts = gs.convert_source("commonsense_qa", [rows[0], no_gold, duplicate, empty, rows[0]])
    assert len(records) == 1 and counts["kept"] == 1
    assert counts["drop:no_gold"] == 1 and counts["drop:duplicate_options"] == 1 and counts["drop:empty_option"] == 1 and counts["drop:duplicate_row"] == 1


def test_no_instruction_or_state_field_copies_g2():
    assert not set().union(*map(set, gs.INSTRUCTIONS.values())) & G2_INSTRUCTIONS
    for name, rows in _rows().items():
        for record in gs.convert_source(name, rows)[0]:
            fields = {line.split(":", 1)[0] for line in record["request"]["state"].split("\n")}
            assert not fields & {"passage", "premise", "hypothesis"}, (name, fields)
    assert {"piqa", "boolq", "openbookqa", "winogrande"} <= {k.split(" ")[0] for k in gs.EXCLUDED}
    assert not set(gs.SOURCES) & {"boolq", "multinli", "snli", "anli", "piqa"}
    assert {spec["license"] for spec in gs.SOURCES.values()} <= PERMISSIVE and all(spec["split"] == "train" for spec in gs.SOURCES.values())


def test_text_units_cover_the_state_its_lines_and_long_options():
    state = 'facts: ["plants make food by photosynthesis", "short one"]\nquestion: "What do plants need to make food?"'
    units = gs.text_units(state, ["sunlight", "they need a lot of light"])
    assert gs.normalize(state) in units
    assert {"plants make food by photosynthesis", "what do plants need to make food", "they need a lot of light"} <= units
    assert "short one" not in units and "sunlight" not in units
    free = gs.text_units("You are a mobile base carrying a full cup of coffee.\nSpeed 1.2.")
    assert "you are a mobile base carrying a full cup of coffee" in free and "speed 1 2" not in free


def test_the_build_drops_rows_that_share_text_with_a_sealed_set_and_interleaves_the_sources(tmp_path):
    rows = _rows()
    sealed_record = {"request": {"state": 'question: "Where would you keep a bottle of milk cold?"', "questions": {}}}
    sealed_path = tmp_path / "sealed.jsonl"
    sealed_path.write_text(json.dumps(sealed_record) + "\n", encoding="utf-8")
    sealed = gs.sealed_unit_sets({"S": sealed_path}, extra={"P": gs.raw_units(["Two men push a boat into the water. They"])})
    records, report = gs.build_general(rows, per_source=1, seed=3, sealed=sealed)
    assert report["commonsense_qa"]["overlap_dropped"] == {"S": 1} and report["hellaswag"]["overlap_dropped"] == {"P": 1}
    assert len(records) == 5 and sorted(r["meta"]["source"]["set"] for r in records) == sorted(gs.SOURCES)
    assert {r["id"] for r in records} >= {"commonsense_qa-train-00001", "hellaswag-train-00000"}
    with pytest.raises(ValueError, match="eligible"):
        gs.build_general(rows, per_source=2, seed=3, sealed=sealed)
    again, _ = gs.build_general(rows, per_source=1, seed=3, sealed=sealed)
    assert [r["id"] for r in again] == [r["id"] for r in records]


def test_interleaving_keeps_every_prefix_balanced():
    per = {name: [{"id": f"{name}-{i}"} for i in range(40)] for name in ("a", "b", "c", "d", "e")}
    out = gs.interleave(per, seed=1)
    assert len(out) == 200
    for prefix in (7, 33, 101, 200):
        counts = Counter(r["id"].split("-")[0] for r in out[:prefix])
        assert max(counts.values()) - min(counts.values()) <= 1, (prefix, counts)
    assert [r["id"] for r in out[:5]] != [r["id"] for r in out[5:10]]  # rounds are not all in the same order or the same records


# --------------------------------------------------------------------------
# the built file
# --------------------------------------------------------------------------

needs_file = pytest.mark.skipif(not MANIFEST.is_file(), reason="general-train not built yet (scripts/convert_general_sets.py)")


@pytest.fixture(scope="module")
def built():
    return read_records(GENERAL), json.loads(MANIFEST.read_text(encoding="utf-8"))


@needs_file
def test_the_built_file_matches_its_manifest_licences_and_exclusions(built):
    records, manifest = built
    assert manifest["file"]["sha256"] == file_sha256(GENERAL) and manifest["file"]["records"] == len(records)
    assert manifest["converter"]["git_dirty"] is False
    assert set(manifest["sources"]) == set(gs.SOURCES) and set(manifest["licenses"].values()) <= PERMISSIVE
    for name, spec in manifest["sources"].items():
        assert spec["split"] == "train" and spec["license"] == gs.SOURCES[name]["license"] and spec["revision"] == gs.SOURCES[name]["revision"]
        assert spec["downloaded"]["sha256"] and spec["sampled"] == manifest["per_source"]
    assert manifest["overlap_check"]["overlap_after_build"] and not any(manifest["overlap_check"]["overlap_after_build"].values())
    assert set(manifest["excluded"]) == set(gs.EXCLUDED)
    by_source = Counter(r["meta"]["source"]["set"] for r in records)
    assert by_source == {name: manifest["per_source"] for name in gs.SOURCES}


@needs_file
def test_every_built_record_is_valid_train_split_and_one_hot_on_one_option(built):
    records, _ = built
    keys = Counter()
    for record in records:
        assert validate_record(record) == [] and record["meta"]["split"] == "train"
        (label,) = record["labels"].values()
        values = sorted(label["probabilities"].values())
        assert values[-1] == 1.0 and sum(values[:-1]) == 0.0
        keys[(record["meta"]["source"]["set"], len(values), next(k for k, p in label["probabilities"].items() if p == 1.0))] += 1
    for (name, n, key), count in keys.items():  # the gold key is spread evenly (the option order was permuted)
        total = sum(c for (s, m, _), c in keys.items() if s == name and m == n)
        assert abs(count / total - 1 / n) < 0.05, (name, key, count, total)


def _hashes(records) -> set[str]:
    return {hashlib.sha256(u.encode("utf-8")).hexdigest() for r in records for u in gs.record_units(r)}


@needs_file
def test_the_built_file_shares_no_text_unit_with_g1_g2_t1_t7_or_t8(built):
    records, manifest = built
    mine = _hashes(records)
    sealed = {"G1": PROBES / "G1.jsonl", "G2": PROBES / "G2.jsonl", **{f"T{i}": TESTS / f"T{i}.jsonl" for i in range(1, 8)}}
    sealed.update({f"T8:{p.name}": p for p in sorted(T8_DIR.glob("*.jsonl"))} if T8_DIR.is_dir() else {})
    checked = 0
    for name, path in sealed.items():
        if not path.is_file():
            continue
        assert not mine & gs.sealed_unit_sets({name: path})[name], name
        checked += 1
    assert checked >= 9
    assert manifest["overlap_check"]["general_units_digest"] == gs.unit_digest(set().union(*(gs.record_units(r) for r in records)))


@needs_file
def test_the_trainer_and_the_diagnosis_tool_read_the_built_file(built, tmp_path):
    from robo_jev.decisions.examples import build_examples
    from robo_jev.decisions.train import load_training_records
    from robo_jev.model.tokenizer import available_tokenizer, load_tokenizer

    loaded, check = load_training_records(GENERAL)
    assert check["invalid"] == 0 and len(loaded) == len(built[0])
    tokenizer = load_tokenizer(available_tokenizer()[0])
    picked = {}
    for record in loaded:
        picked.setdefault(record["meta"]["source"]["set"], record)
    for record in picked.values():
        (name,) = record["request"]["questions"]
        (example,) = build_examples(record, [name], tokenizer, order_seed=11, fmt="chat")
        assert example["target"] == pytest.approx([record["labels"][name]["probabilities"][k] for k in example["options"]])
        assert example["text"].count("\nOptions:\n") == 1 and "Situation:\n" in example["text"]
    pytest.importorskip("transformers")
    from robo_jev.model.backbone_qwen import torch_reference_kernels

    spec = importlib.util.spec_from_file_location("diagnose_decisions_general", REPO / "scripts" / "diagnose_decisions.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    path = tmp_path / "general5.jsonl"
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in picked.values()), encoding="utf-8")
    out = tmp_path / "run.json"
    with torch_reference_kernels():
        assert module.main(["--model", "tiny", "--input", str(path), "--out", str(out), "--n-boot", "10", "--batch", "8"]) == 0
    summary = json.loads(out.read_text(encoding="utf-8"))
    assert summary["validation"]["invalid"] == 0 and summary["config"]["records"] == len(picked)
    assert summary["summary"]["readings"]["single_order"]["counts"]["by_block"] == {"choice/other": len(picked)}
