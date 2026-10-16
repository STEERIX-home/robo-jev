"""Task v0.8b — gate B8 (fresh bench-shaped requests): tag rules, records, split, freeze / check, independence helpers."""

from __future__ import annotations

import importlib.util
import json
import stat
from pathlib import Path

import pytest

from robo_jev.bench import b8_v08 as B
from robo_jev.bench.gates_v08 import B8_PREFIX, B8_SET, option_range_mark
from robo_jev.data.sealed import sealed_record_set

REL07_TAG_SCRIPT = B.ARTIFACTS / "scratch" / "rel07" / "bench-check" / "tag_options.py"
REL07_IDS = [
    "wait", "let-the-move-go-on", "hold-still-for-a-look", "hold-for-the-researcher", "stay", "carry-on", "stop-where-you-are", "call-the-researcher",
    "call-the-operator", "go-on-with-the-pass", "look-ahead-standing-still", "look-round-the-room", "nothing-new", "something-the-cafe-does-not-make",
    "not-made-out", "go-above-a-station", "go-to-rest", "go-to-a-person", "go-to-the-standby-spot", "sweep", "lower", "close-the-gripper",
    "open-the-gripper", "raise", "take", "turn-to-heard", "step-back-out", "follow-the-route", "step-aside-and-stop", "back-up", "pass",
    "start-crossing", "another-way", "a-person", "a-drink-from-the-menu", "water", "napkins", "the-bill",
]


# --------------------------------------------------------------------------
# Tag rules
# --------------------------------------------------------------------------


def test_rel07_rules_come_first_in_order_then_only_v08_rules():
    assert len(B.REL07_RULES) == 38
    assert [r[0] for r in B.RULES[:38]] == REL07_IDS
    assert B.RULES[:38] == B.REL07_RULES
    assert all(r[0].startswith("v08-") for r in B.RULES[38:]) and B.RULES[38:] == B.V08_RULES
    assert {r[1] for r in B.RULES} == {"PASSIVE", "ACTIVE"}
    assert len({r[0] for r in B.RULES}) == len(B.RULES)


@pytest.mark.skipif(not REL07_TAG_SCRIPT.is_file(), reason="the rel07 check's tag script is not on this machine")
def test_rel07_rules_are_verbatim():
    spec = importlib.util.spec_from_file_location("rel07_tag_options_test", REL07_TAG_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert B.REL07_RULES == module.RULES


@pytest.mark.skipif(not B.REL07_TAGS.is_file(), reason="the rel07 check's frozen tags are not on this machine")
def test_rel07_frozen_tags_are_reproduced_exactly():
    tags = json.loads(B.REL07_TAGS.read_text(encoding="utf-8"))
    options = [o for r in tags["requests"] for q in r["questions"].values() for o in q["options"]]
    assert len(options) == 485
    assert all(B.tag(o["text"]) == (o["rule"], o["tag"], o["borderline"]) for o in options)


@pytest.mark.parametrize("text, rule, cls, borderline", [
    ("Wait: hold where you are for a second.", "wait", "PASSIVE", False),
    ("Hold still over this station for a second, for a fresh look.", "hold-still-for-a-look", "PASSIVE", True),
    ("Go to rest, over the bench's front middle at the clear height.", "go-to-rest", "ACTIVE", True),
    ("Go above input nest 3 at the clear height, rising to it first: about 2.1 s from rest.", "go-above-a-station", "ACTIVE", False),
    ("Stay where you are this second and keep watching.", "stay", "PASSIVE", False),
    ("Take the latte for table 6 (seat 2) to them: walk to the table, come in beside the customer and offer the tray.", "take", "ACTIVE", False),
    ("The way ahead — standing still, the head looks round the room.", "look-ahead-standing-still", "PASSIVE", True),
    ("The iced tea, from the menu.", "a-drink-from-the-menu", "ACTIVE", False),
    ("step 0.6 m aside toward the kerb or the wall and stop, to let others by", "step-aside-and-stop", "ACTIVE", True),
    ("call the remote operator (you are blocked); an operator answers in about 20–40 s", "call-the-operator", "PASSIVE", False),
    ("take another way: 180 m with 1 crossings, about 150 s", "take", "ACTIVE", False),
    ("Hold this second, taking up no work.", "v08-hold-this-second", "PASSIVE", False),
    ("Go on with the work under way: Prepare t1, …", "v08-go-on-with-the-work", "PASSIVE", True),
    ("Set t3, P-1234 at position 3 of rack 2 aside in the exception rack, for: clot.", "v08-set-a-tube-aside", "PASSIVE", True),
    ("Prepare t2, P-4321 at position 2 of rack 2: from the presented rack's position 2, turn its label …", "v08-prepare-a-tube", "ACTIVE", False),
    ("Load t9, balance tube BAL-1 of 5 mL into the centrifuge: from …", "v08-load-a-tube", "ACTIVE", False),
    ("Shut the centrifuge's lid and press it until it latches.", "v08-shut-the-lid", "ACTIVE", False),
    ("Open the centrifuge's lid by its handle.", "v08-open-the-lid", "ACTIVE", False),
])
def test_tag_examples(text, rule, cls, borderline):
    assert B.tag(text) == (rule, cls, borderline)


def test_an_option_no_rule_matches_stops_the_build():
    with pytest.raises(ValueError, match="no rule matches"):
        B.tag("Dance a little jig.")


# --------------------------------------------------------------------------
# Records and split
# --------------------------------------------------------------------------


def _row(env: str, rid: str, template: str = "t", n: int = 17, target: str = "move", planner: str | None = "o0", state: str | None = None) -> dict:
    texts = ["Wait: hold where you are for a second."] + [f"Go above input nest {i} at the clear height." for i in range(1, n)]
    criteria = {f"o{i}": texts[i] for i in range(n)}
    return {"id": rid, "env": env, "template": template, "index": 0, "target": [target], "situation": "s", "planner": planner,
            "available": list(criteria), "builder": {"repo": "STEERIX-home/robo-jev-web", "commit": "c", "module": f"src/{env}/judge/policy.ts",
                                                     "functions": ["planFor", "stateSheet", "buildQuestions"]},
            "generator": {"seed": 1, "template": template, "index": 0, "facts_sha256": "f", "node": "v24"},
            "request": {"state": state or f"state of {rid}", "questions": {target: {"type": "choice", "instructions": "What next?", "criteria": criteria},
                                                                       "safe": {"type": "noul", "instructions": "Is it safe?"}}}}


def test_to_record_b8_meta_ids_and_seal():
    record = B.to_record(_row("plates", "plates-x-0"), part="b8", index=7)
    assert record["id"] == f"{B8_PREFIX}plates-007" and record["meta"]["set"] == B8_SET
    assert record["meta"]["option_range"] == option_range_mark() and record["meta"]["p2"] is True
    assert record["meta"]["generator"]["candidate"] == "plates-x-0" and "labels" not in record
    assert B.record_problems(record) == []
    assert sealed_record_set(record) in (None, B8_SET)  # registered by the parent task in robo_jev.data.sealed


def test_to_record_jev_is_not_sealed_and_tubes_are_not_p2():
    record = B.to_record(_row("tubes", "tubes-work-0", target="work", n=46, planner=None), part="jev", index=1)
    assert record["id"] == "v08jev-b-tubes-001" and record["meta"]["set"] == B.JEV_SET and record["meta"]["part"] == "bench"
    assert record["meta"]["p2"] is False and sealed_record_set(record) is None


def test_record_problems_catch_a_missing_opt_in_a_bad_target_and_too_many_options():
    record = B.to_record(_row("plates", "p"), part="b8", index=1)
    bad = json.loads(json.dumps(record))
    bad["meta"]["option_range"] = {"version": "options-26-1", "max": 26}
    bad["meta"]["target"] = ["safe"]
    assert any("opt-in" in p for p in B.record_problems(bad)) and any("not a choice" in p for p in B.record_problems(bad))
    with pytest.raises(ValueError):
        B.to_record(_row("plates", "p", n=256), part="b8", index=1)
    with pytest.raises(ValueError):
        B.to_record(_row("plates", "p"), part="test", index=1)


def test_split_is_seeded_disjoint_and_counts_hold():
    rows = [_row(env, f"{env}-{t}-{i}", template=f"t{t}") for env in ("plates", "delivery") for t in range(3) for i in range(5)]
    split = {"plates": (6, 4), "delivery": (5, 5)}
    a = B.split_rows(rows, seed=3, split=split)
    b = B.split_rows(list(reversed(rows)), seed=3, split=split)
    assert [r["id"] for r in a["b8"]] == [r["id"] for r in b["b8"]] and [r["id"] for r in a["jev"]] == [r["id"] for r in b["jev"]]
    ids = [r["id"] for part in a.values() for r in part]
    assert sorted(ids) == sorted(r["id"] for r in rows) and len(set(ids)) == len(ids)
    assert sum(r["env"] == "plates" for r in a["b8"]) == 6 and sum(r["env"] == "delivery" for r in a["jev"]) == 5
    # template-balanced: the first three B8 plates come from three templates
    assert len({r["template"] for r in [r for r in a["b8"] if r["env"] == "plates"][:3]}) == 3
    c = B.split_rows(rows, seed=4, split=split)
    assert [r["id"] for r in c["b8"]] != [r["id"] for r in a["b8"]]
    with pytest.raises(ValueError):
        B.split_rows(rows, seed=3, split={"plates": (14, 4)})


def test_split_cafe_stratum_carrying_the_request_question():
    rows = [_row("cafe", f"cafe-r-{i}", template="beside", target="request", n=13) for i in range(6)]
    rows += [_row("cafe", f"cafe-o-{t}-{i}", template=f"o{t}", target="intent", n=10) for t in range(2) for i in range(5)]
    out = B.split_rows(rows, seed=1, split={"cafe": (6, 4)}, cafe_request_split=(3, 2))
    assert sum("request" in r["target"] for r in out["b8"]) == 3 and sum("request" in r["target"] for r in out["jev"]) == 2
    assert len(out["b8"]) == 6 and len(out["jev"]) == 4


# --------------------------------------------------------------------------
# Freeze and check
# --------------------------------------------------------------------------


def _b8_dir(tmp_path: Path) -> Path:
    records = [B.to_record(_row("plates", f"plates-{i}", planner=f"o{i % 3}"), part="b8", index=i + 1) for i in range(3)]
    records.append(B.to_record(_row("tubes", "tubes-0", target="work", n=46, planner=None), part="b8", index=1))
    B.write_jsonl(tmp_path / B.B8_FILE_NAME, records, readonly=True)
    return tmp_path


def test_freeze_writes_readonly_files_and_check_passes(tmp_path):
    d = _b8_dir(tmp_path)
    record = B.freeze(d, now="2026-10-16T13:00:00+09:00")
    assert B.check_frozen(d) == []
    for name in (B.B8_FILE_NAME, B.TAGS_NAME, B.FROZEN_NAME):
        assert not (d / name).stat().st_mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH)
    tags = json.loads((d / B.TAGS_NAME).read_text(encoding="utf-8"))
    assert record["b8_sha256"] == tags["b8_sha256"] == B.sha256_file(d / B.B8_FILE_NAME)
    plates = tags["requests"][0]["questions"]["move"]
    assert plates["planner"] == "o0" and plates["planner_tag"] == "PASSIVE" and plates["passive"] == 1 and plates["active"] == 16
    assert tags["requests"][3]["questions"]["work"]["planner"] is None
    assert tags["counts"]["plates/PASSIVE"] == 3


def test_check_frozen_detects_a_changed_byte_and_missing_files(tmp_path):
    d = _b8_dir(tmp_path)
    B.freeze(d)
    b8 = d / B.B8_FILE_NAME
    b8.chmod(0o644)
    text = b8.read_text(encoding="utf-8")
    b8.write_text(text.replace("state of plates-0", "state of plates-0 "), encoding="utf-8")
    problems = B.check_frozen(d)
    assert any("B8.jsonl changed" in p for p in problems) and any("another B8.jsonl" in p for p in problems)
    b8.write_text(text, encoding="utf-8")
    assert B.check_frozen(d) == []
    tags = d / B.TAGS_NAME
    tags.chmod(0o644)
    tags.write_text(tags.read_text(encoding="utf-8").replace('"PASSIVE"', '"ACTIVE"', 1), encoding="utf-8")
    assert any("tags.json changed" in p for p in B.check_frozen(d))
    (d / B.FROZEN_NAME).chmod(0o644)
    (d / B.FROZEN_NAME).unlink()
    assert B.check_frozen(d) and "missing" in B.check_frozen(d)[0]


def test_planner_not_an_option_is_refused(tmp_path):
    bad = B.to_record(_row("plates", "p"), part="b8", index=1)
    bad["meta"]["planner"] = "nowhere"
    with pytest.raises(ValueError, match="planner"):
        B.tags_payload([bad], b8_sha256="x", created="now")


# --------------------------------------------------------------------------
# Independence helpers
# --------------------------------------------------------------------------


def test_jaccard_and_dynamic_lines():
    assert B.jaccard({"a", "b"}, {"b", "c"}) == pytest.approx(1 / 3) and B.jaccard(set(), set()) == 1.0
    dyn = B.dynamic_lines({"x": "common\nonly x\n", "y": "common\nonly y\n\n", "z": "common\n"})
    assert dyn == {"x": frozenset({"only x"}), "y": frozenset({"only y"}), "z": frozenset()}


def test_overlap_report_counts_identical_states_and_same_dynamic_lines():
    b8 = [B.to_record(_row("plates", "a", state="head\nA\nB"), part="b8", index=1), B.to_record(_row("plates", "b", state="head\nC"), part="b8", index=2)]
    jev = [B.to_record(_row("plates", "c", state="head\nD"), part="jev", index=1)]
    rel07 = [{"id": "plates-01", "env": "plates", "request": {"state": "head\nA\nB"}}, {"id": "plates-02", "env": "plates", "request": {"state": "head\nE"}}]
    report = B.overlap_report(b8, jev, rel07)
    assert report["identical_states"]["b8∩rel07"] == 1 and report["identical_states"]["b8∩jev"] == 0
    entry = report["by_env"]["plates"]
    assert entry["b8_vs_rel07"]["same_dynamic_lines"] == 1 and entry["b8_vs_rel07"]["identical_states"] == 1
    assert entry["b8_vs_jev"]["same_dynamic_lines"] == 0 and entry["boilerplate_lines"] == 1


@pytest.mark.skipif(not (B.B8_DIR / B.FROZEN_NAME).is_file(), reason="gate B8 not built on this machine")
def test_the_built_gate_is_frozen_unchanged_and_sealed():
    assert B.check_frozen(B.B8_DIR) == []
    records = B.read_jsonl(B.B8_DIR / B.B8_FILE_NAME)
    assert all(r["id"].startswith(B8_PREFIX) and r["meta"]["set"] == B8_SET and B.record_problems(r) == [] for r in records)
    envs = {env: sum(r["meta"]["env"] == env for r in records) for env in ("plates", "cafe", "delivery", "tubes")}
    assert envs["plates"] >= 36 and envs["cafe"] >= 24 and envs["delivery"] >= 24 and envs["tubes"] >= 1
    if B.JEV_BENCH_FILE.is_file():
        jev = B.read_jsonl(B.JEV_BENCH_FILE)
        assert not {B.sha256_text(r["request"]["state"]) for r in jev} & {B.sha256_text(r["request"]["state"]) for r in records}
        assert all(sealed_record_set(r) is None for r in jev)
