"""`robo_jev.simgen.shape` — Task v0.7a: deployment-shaped requests from v0.6's records, labels unchanged.

Checked here: the assignment shares and N range; the cut rules (never inside a block, after a header, before a pronoun; sentence cuts only
in paragraph-like states); sheets hold the original verbatim and in order (two proofs) within 1,500–8,000 characters, are deterministic,
give a pair's two records the same filler, and never read a label (the LOCAL PLAN aside); the checker catches a dropped, altered or moved
sentence; the filler never names the scenes' physics; the plan coin is fair and its wrong pick is clearly wrong; facts move into every
question verbatim; distractors respect their families, keys, scenes, words and the comparison / action / outcome filters; expansion keeps
the own options and labels and gives distractors 0; general items take distractors only from compatible other sources.
"""

from __future__ import annotations

import copy
import random
import re
import statistics

import pytest

from robo_jev.simgen import shape as S

ENGINE_PLANNER = """CONTEXT: Afternoon on a geriatric ward.
TASK: you are a hospital delivery robot, returning empty specimen boxes to the lab
ROBOT STATE
  speed: 1.10 m/s in the service corridor behind the pharmacy
  braking: gentle 1.1 m/s² | firm 1.7 m/s² | max 3.0 m/s²; latency 0.13 s; footprint radius 0.31 m
PERCEIVED
  aisle width: 1.75 m; your offset: 0.01 m left of its centre
  an elderly patient using a walking frame (width 0.75 m): oncoming, 0.10 m left of the aisle centre
    3.35 m ahead, moving toward you at 0.5 m/s (last update 0.70 s ago)
NOTES
  - Your next delivery is due in 12 minutes.
AVAILABLE ACTIONS
  - keep going at your current speed
  - brake as hard as you can"""
LOG_PARAGRAPH = ("Log, warehouse floor. Unit: a low-deck inventory robot. Cargo: one archive box of files standing upright and unstrapped at the "
                 "centre of the deck; 0.32 m high on a base 0.36 m long and 0.32 m across; mass 6.2 kg. Packing note: its heavier end faces the "
                 "front of the deck. Deck covering: diamond-tread steel. The unit is heading onto a ramp, climbing: its deck pitches front up "
                 "from level to 21.5° within 2.3 s, and keeps that pitch on the slope.")
BRIEFING = ("Next you'll turn the lazy Susan on a round table in the breakfast room by half a turn to bring the jug to seat 10. You've timed "
            "it at roughly 0.9 s.\n\nOn it: a square juice jug, nearly empty, around 10 cm square and 23.5 cm tall, 17-ish cm from the centre, "
            "on a rubber ring on the bottom. The top's lacquered wood and clean and dry.")
KOREAN = "상황: 대형 물류 창고의 야간 근무 시간입니다.\n과제: 무인 팔레트 운반차 — 지금 생수 팔레트를 4번 도크로 옮기는 중\n로봇 상태\n  제동: 부드럽게 1.0 m/s²\n  속도: 1.55 m/s\n메모\n  - 추정 정지 거리는 148 cm입니다."
STATES = (ENGINE_PLANNER, LOG_PARAGRAPH, BRIEFING, KOREAN)


def _record(state: str = LOG_PARAGRAPH, *, rid: str = "v06a1-x-000001", action_probs: dict | None = None, reco: bool = False) -> dict:
    questions = {"moves": {"type": "noul", "instructions": "Will it move?"},
                 "risk": {"type": "score", "instructions": "Rate the risk.", "criteria": ["0 = none", "1 = sure"]}}
    labels = {"moves": {"type": "noul", "p_yes": 0.2}, "risk": {"type": "score", "value": 0.7}}
    meta = {"set": "v06-a1", "scene": "x", "language": "en", "key_question": "moves", "question_kind": {"moves": "key", "risk": "score_cf"}}
    if action_probs is not None:
        questions["best"] = {"type": "choice", "instructions": "Your best move?",
                             "criteria": {k: f"option {k}" for k in action_probs}}
        labels["best"] = {"type": "choice", "kind": "action", "probabilities": dict(action_probs)}
        meta["question_kind"]["best"] = "action"
    if reco:
        meta["recommendation"] = {"present": True}
    return {"schema": "simgen-request/0", "id": rid, "meta": meta, "request": {"state": state, "questions": questions}, "labels": labels}


# -- assignment -------------------------------------------------------------------------------------------------------


def test_the_kinds_are_drawn_from_the_key_at_the_rules_shares():
    kinds = [S.assign_kind(f"k{i}") for i in range(40_000)]
    assert abs(kinds.count("sheet") / len(kinds) - S.SHEET_SHARE) < 0.01
    assert abs(kinds.count("facts") / len(kinds) - S.FACTS_SHARE) < 0.006
    assert [S.assign_kind(f"k{i}") for i in range(50)] == kinds[:50], "a pure function of the key"


def test_n_is_uniform_over_9_to_26():
    counts = [S.draw_count(f"k{i}", "q") for i in range(36_000)]
    assert min(counts) == 9 and max(counts) == 26
    shares = [counts.count(n) / len(counts) for n in range(9, 27)]
    assert max(shares) - min(shares) < 0.012


# -- cuts and units ---------------------------------------------------------------------------------------------------


def test_cuts_never_split_a_block_follow_a_header_or_precede_a_pronoun():
    cuts = S.cut_points(ENGINE_PLANNER)
    lines_at = {c: ENGINE_PLANNER[c:].split("\n")[0] for c in cuts}
    assert all(not text[:1].isspace() for text in lines_at.values()), "never before an indented (child) line"
    assert "ROBOT STATE" in [t.strip() for t in lines_at.values()] and "PERCEIVED" in [t.strip() for t in lines_at.values()]
    header = "Cargo:\nA box of files.\nIt sits upright.\nThe unit climbs."
    starts = [header[c:].split("\n")[0] for c in S.cut_points(header)]
    assert "A box of files." not in starts, "not right after a line that ends with ':'"
    assert "It sits upright." not in starts, "not before a pronoun"
    assert "The unit climbs." in starts


def test_sentence_cuts_only_inside_paragraph_like_states():
    cuts = S.cut_points(LOG_PARAGRAPH)
    assert cuts and all(LOG_PARAGRAPH[c - 1] == " " for c in cuts)
    assert not any(LOG_PARAGRAPH[c:].startswith("its ") for c in cuts)
    bullets = "Facts first.\n- Braking: gentle 1.1 m/s². Firm 2.0 m/s².\n- Speed 0.4 m/s.\n- More. Still more.\n- Last line."
    assert all(bullets[c - 1] == "\n" for c in S.cut_points(bullets)), "a list state is cut only between lines"


def test_sentence_units_cover_every_sentence():
    units = S.sentence_units(LOG_PARAGRAPH)
    assert units[0] == "Log, warehouse floor." and units[-1].endswith("keeps that pitch on the slope.")
    assert " ".join(units) == " ".join(LOG_PARAGRAPH.split())


# -- sheets -----------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("state", STATES)
@pytest.mark.parametrize("key", ["a", "b", "c", "d", "e", "f", "g", "h"])
def test_a_sheet_holds_the_original_verbatim_in_order_within_the_length_range(state, key):
    sheet, info = S.build_sheet(state, key=key, reference_length=len(state), pick=None)
    assert S.sheet_problems(state, sheet, cuts=info["cuts"], spans=info["spans"]) == []
    assert S.SHEET_CHARS[0] <= len(sheet) <= S.SHEET_CHARS[1] and info["length"] == len(sheet)
    assert info["style"] in {s.name for s in S.STYLES} and info["tone"] in S.TONES
    assert S.build_sheet(state, key=key, reference_length=len(state), pick=None) == (sheet, info), "deterministic"


def test_every_style_tone_and_chunk_count_occurs():
    seen = {"style": set(), "tone": set(), "chunks": set()}
    for i in range(300):
        _, info = S.build_sheet(LOG_PARAGRAPH, key=f"k{i}", reference_length=len(LOG_PARAGRAPH), pick=None)
        seen["style"].add(info["style"]), seen["tone"].add(info["tone"]), seen["chunks"].add(info["chunks"])
    assert seen == {"style": {s.name for s in S.STYLES}, "tone": set(S.TONES), "chunks": {1, 2, 3}}
    assert len(S.STYLES) >= 5


def test_the_two_records_of_a_pair_get_the_same_filler():
    base, partner = LOG_PARAGRAPH, LOG_PARAGRAPH.replace("diamond-tread steel", "smooth PVC")
    reference = max(len(base), len(partner))
    for key in ("p1", "p2", "p3", "p4", "p5"):
        a, ia = S.build_sheet(base, key=key, reference_length=reference, pick=None)
        b, ib = S.build_sheet(partner, key=key, reference_length=reference, pick=None)
        assert _outside(a, ia) == _outside(b, ib), "everything outside the original chunks is identical"
        assert len(a) - len(b) == len(base) - len(partner)


def _outside(text: str, info: dict) -> list[str]:
    """The sheet's text between the original chunks (the filler, headers and separators), piece by piece."""
    return [text[a:b] for n, (a, b) in enumerate(_gaps(text, info)) if n % 2 == 0]


def _gaps(text: str, info: dict) -> list[tuple[int, int]]:
    out, last = [], 0
    for start, end in info["spans"]:
        out += [(last, start), (start, end)]
        last = end
    return out + [(last, len(text))]


def test_a_sheet_never_reads_a_label():
    """The sheet is built from the state, the key and the plan pick only: two records that differ only in their labels (and carry no
    recommendation) get the same sheet."""
    one = _record()
    two = copy.deepcopy(one)
    two["labels"]["moves"]["p_yes"], two["labels"]["risk"]["value"] = 0.95, 0.05
    assert S.plan_pick(one) is None and S.plan_pick(two) is None
    for key in ("x", "y", "z"):
        assert S.build_sheet(one["request"]["state"], key=key, reference_length=500, pick=S.plan_pick(one)) == \
            S.build_sheet(two["request"]["state"], key=key, reference_length=500, pick=S.plan_pick(two))


def test_the_checker_catches_a_dropped_altered_or_moved_sentence():
    sheet, info = S.build_sheet(LOG_PARAGRAPH, key="k-3", reference_length=len(LOG_PARAGRAPH), pick=None)
    assert S.sheet_problems(LOG_PARAGRAPH, sheet, cuts=info["cuts"], spans=info["spans"]) == []
    dropped = sheet.replace("Deck covering: diamond-tread steel.", "")
    assert S.sheet_problems(LOG_PARAGRAPH, dropped)
    altered = sheet.replace("21.5°", "12.5°")
    assert S.sheet_problems(LOG_PARAGRAPH, altered, cuts=info["cuts"], spans=info["spans"])
    assert S.sheet_problems(LOG_PARAGRAPH, LOG_PARAGRAPH), "the bare state is too short to be a sheet"
    first, last = S.sentence_units(LOG_PARAGRAPH)[0], S.sentence_units(LOG_PARAGRAPH)[-1]
    moved = sheet.replace(last, "") + "\n" + last + "\n" + first
    assert S.sheet_problems(LOG_PARAGRAPH, moved.replace(first, "", 1))


def test_the_filler_never_names_the_scenes_physics_or_instructs_the_robot():
    texts = S.filler_texts()
    assert len(texts) > 100
    assert [t for t in texts if S.FILLER_FORBIDDEN.search(t)] == []
    assert set(S._NOTICE) == set(S.TONES) and all(len(lines) >= 8 for lines in S._NOTICE.values())


def test_a_status_field_appears_once_per_sheet():
    for i in range(80):
        sheet, info = S.build_sheet(LOG_PARAGRAPH, key=f"u{i}", reference_length=len(LOG_PARAGRAPH), pick=None)
        style = next(s for s in S.STYLES if s.name == info["style"])
        head = style.header.format(name=style.names["status"], index=info["sections"].index("status") + 1) if "status" in info["sections"] else None
        if head is None:
            continue
        block = sheet.split(head, 1)[1].split(style.sep + style.header.split("{")[0], 1)[0]
        fields = [line.split(":", 1)[0] for line in block.strip().split("\n") if ":" in line]
        assert len(fields) == len(set(fields)), fields


def test_sheet_filler_lines_stay_clear_of_the_forbidden_words_in_built_sheets():
    for i in range(60):
        sheet, info = S.build_sheet(LOG_PARAGRAPH, key=f"f{i}", reference_length=len(LOG_PARAGRAPH), pick=None)
        outside = "".join(_outside(sheet, info))
        assert not S.FILLER_FORBIDDEN.search(outside), f"key f{i}: {S.FILLER_FORBIDDEN.search(outside)}"


# -- LOCAL PLAN -------------------------------------------------------------------------------------------------------


def test_the_plan_names_the_known_best_half_the_time_and_otherwise_a_clearly_wrong_option():
    probs = {"go": 0.05, "slow": 0.85, "stop": 0.1, "wait": 0.0}
    picks = [S.plan_pick(_record(rid=f"r{i}", action_probs=probs)) for i in range(4000)]
    right = [p for p in picks if p["right"]]
    assert abs(len(right) / len(picks) - S.PLAN_RIGHT_P) < 0.03
    assert all(p["option"] == "slow" for p in right)
    assert all(p["option"] != "slow" and probs[p["option"]] <= S.PLAN_WRONG_MAX for p in picks if not p["right"])
    assert all(p["known_best"] for p in picks)


def test_without_a_known_best_the_plan_is_a_random_unscored_option_and_structure_alone_decides_eligibility():
    soft = {"go": 0.5, "slow": 0.45, "stop": 0.05}
    picks = [S.plan_pick(_record(rid=f"s{i}", action_probs=soft)) for i in range(600)]
    assert all(p is not None and p["right"] is None and not p["known_best"] for p in picks)
    assert {p["option"] for p in picks} == set(soft)
    assert S.plan_pick(_record(action_probs=None)) is None, "no action question"
    assert S.plan_pick(_record(action_probs={"a": 0.9, "b": 0.1}, reco=True)) is None, "the state already carries a recommendation"


def test_a_sheet_shows_a_recommendation_only_by_the_keys_coin():
    record = _record(action_probs={"go": 0.0, "slow": 1.0})
    modes = []
    for i in range(400):
        _, info = S.build_sheet(LOG_PARAGRAPH, key=f"m{i}", reference_length=len(LOG_PARAGRAPH), pick=S.plan_pick(record))
        modes.append(info["plan"])
        assert info["plan_eligible"]
    assert 0.42 < modes.count("line") / len(modes) < 0.58 and modes.count("none") and modes.count("omit")
    for i in range(100):
        _, info = S.build_sheet(LOG_PARAGRAPH, key=f"m{i}", reference_length=len(LOG_PARAGRAPH), pick=None)
        assert info["plan"] in ("none", "omit") and info["plan_pick"] is None


# -- facts in instructions --------------------------------------------------------------------------------------------


@pytest.mark.parametrize("state", STATES)
def test_facts_move_into_every_questions_instructions_verbatim(state):
    record = _record(state, action_probs={"a": 0.9, "b": 0.1})
    minimal, instructions, info = S.facts_request(record, key="kk")
    new = copy.deepcopy(record)
    new["request"]["state"] = minimal
    for name, text in instructions.items():
        new["request"]["questions"][name]["instructions"] = text
    assert S.facts_problems(record, new) == []
    assert set(instructions) == set(record["request"]["questions"]) and info["kind"] == "facts"
    assert not any(u in minimal for u in S.sentence_units(state) if len(u.split()) >= 3)
    broken = copy.deepcopy(new)
    broken["request"]["questions"]["moves"]["instructions"] = "Will it move?"
    assert S.facts_problems(record, broken)


# -- many options -----------------------------------------------------------------------------------------------------


def _pool() -> S.DistractorPool:
    pool = S.DistractorPool()
    rows = {
        "act": [("brake_soft", "Brake softly until you stop."), ("detour", "Take the long route around the hall."), ("call_human", "Call a person to check it."),
                ("turn_wide", "Take the corner wide at 0.4 m/s."), ("press_down", "Press it down firmly."), ("reseat", "Lift it out and place it again.")],
        "out": [("tips_over", "tips over"), ("runaway", "keeps rolling"), ("dock_drop", "goes off the dock"), ("lean", "tips and leans on the edge"),
                ("hangs", "hangs back on your hand"), ("creeps", "creeps 1-8 mm")],
        "dir": [("higher", "goes up"), ("safer", "less likely"), ("same", "about the same")],
        "band": [("low", "less than 1 in 3"), ("medium", "between 1 in 3 and 2 in 3"), ("high", "more than 2 in 3")],
        "rule": [("close", "less than 390 cm"), ("accept", "within 10.5 mm of nominal"), ("fresh", "last seen 0.5 s ago or more recently")],
        "est_obj": [("seated", "Properly seated in the dish rack"), ("perched", "Resting high on a guide or lip"), ("lid", "Yes, it has a lid on")],
    }
    for family, entries in rows.items():
        for i, (key, desc) in enumerate(entries):
            pool.add(family, key, desc, scene=f"other:{family}:{i % 2}")
    pool.add("cmp", "as_planned", "turning as planned", "other:cmp")
    return pool


def test_the_pool_drops_same_plan_and_meta_options():
    pool = _pool()
    assert "same" not in {e.key for e in pool.entries["dir"]}
    assert "cmp" not in pool.entries, "'as planned' is never a distractor"
    assert pool.excluded == 2


@pytest.mark.parametrize("family", S.PHYSICS_FAMILIES)
def test_draws_respect_families_keys_scenes_words_and_filters(family):
    pool = _pool()
    for seed in range(30):
        own_words = S.content_words("Brake softly and stop. Which plan is riskier? stays put")
        drawn = pool.draw(family, count=4, own_keys={"stay", "go"}, own_norms={"stays put"}, words=own_words, scene="other:act:0",
                          rnd=random.Random(seed))
        if drawn is None:
            continue
        assert {e.family for e in drawn} <= set(S.ALLOWED[family])
        assert len({e.key for e in drawn}) == len(drawn) and not {e.key for e in drawn} & {"stay", "go"}
        assert all(e.scene != "other:act:0" and not (e.words & own_words) for e in drawn)
        if family in ("dir", "cmp"):
            assert not any(S.COMPARISON_WORDS.search(f"{e.key} {e.description}".replace("_", " ")) for e in drawn)
        if family == "act":
            assert not any(S.ACTION_WORDS.search(f"{e.key} {e.description}".replace("_", " ")) for e in drawn)
        if family == "out":
            assert not any(S.OUTCOME_WORDS.search(f"{e.key} {e.description}".replace("_", " ")) for e in drawn)


def test_direction_options_never_go_to_action_or_outcome_questions():
    assert "dir" not in S.ALLOWED["act"] and "dir" not in S.ALLOWED["out"]
    assert all(family not in sources for family, sources in S.ALLOWED.items()), "never the question's own family"


def test_a_draw_that_cannot_be_filled_returns_none():
    assert _pool().draw("out", count=40, own_keys=set(), own_norms=set(), words=set(), scene="x", rnd=random.Random(0)) is None


def test_expansion_keeps_own_options_and_labels_and_gives_distractors_zero():
    question = {"type": "choice", "instructions": "Best move?", "criteria": {"a": "go", "b": "stop"}}
    label = {"type": "choice", "kind": "action", "probabilities": {"a": 0.3, "b": 0.7}, "expected_cost": {"a": 2.0, "b": 0.5}}
    new_q, new_l = S.expand_choice(question, label, [("x", "tips over"), ("y", "")])
    assert list(new_q["criteria"].items())[:2] == list(question["criteria"].items()) and list(new_q["criteria"])[2:] == ["x", "y"]
    assert new_l["probabilities"] == {"a": 0.3, "b": 0.7, "x": 0.0, "y": 0.0} and new_l["kind"] == "action"
    assert new_l["expected_cost"] == {"a": 2.0, "b": 0.5, "x": 2.0, "y": 2.0}, "the worst own cost"
    assert question["criteria"] == {"a": "go", "b": "stop"}, "the input is not modified"
    with pytest.raises(ValueError):
        S.expand_choice(question, label, [("a", "dup")])


def test_question_families_follow_the_records_kinds():
    record = _record(action_probs={"a": 1.0, "b": 0.0})
    assert S.question_family(record, "best", "mujoco") == "act" and S.question_family(record, "moves", "mujoco") is None
    engine = {"request": {"questions": {"s1_band_1": {"type": "choice"}, "s1_cmp_1_2": {"type": "choice"}, "how_full": {"type": "choice"},
                                        "likely_move": {"type": "choice"}, "zone": {"type": "choice"}, "act": {"type": "choice"}}},
              "labels": {"s1_band_1": {"kind": "state"}, "s1_cmp_1_2": {"kind": "state"}, "how_full": {"kind": "state"}, "likely_move": {"kind": "state"},
                         "zone": {"kind": "rule"}, "act": {"kind": "action"}}}
    assert {n: S.question_family(engine, n, "engine") for n in engine["request"]["questions"]} == {
        "s1_band_1": "band", "s1_cmp_1_2": "cmp", "how_full": "est_obj", "likely_move": "est", "zone": "rule", "act": "act"}


# -- general items ----------------------------------------------------------------------------------------------------


def _general(source: str, rid: str, gold: str, others: list[str], state: str = 'question: "Which works?"') -> dict:
    criteria = {"o1": gold, **{f"o{i + 2}": t for i, t in enumerate(others)}}
    return {"id": rid, "meta": {"origin_group": f"public/{source}/train/{rid}", "source": {"set": source, "row": 1}},
            "request": {"state": state, "questions": {"q_general": {"type": "choice", "instructions": "Pick one.", "criteria": criteria}}},
            "labels": {"q_general": {"type": "choice", "kind": "probe", "probabilities": {"o1": 1.0, **{f"o{i + 2}": 0.0 for i in range(len(others))}}}}}


def test_general_items_need_a_long_answer_and_no_meta_option():
    assert S.general_eligible(_general("hellaswag", "h1", "takes the plate and eats", ["runs away fast", "sings a long song"]))
    assert not S.general_eligible(_general("qasc", "q1", "plants", ["fish", "bees"]))
    assert not S.general_eligible(_general("cosmos_qa", "c1", "They went to the shop .", ["None of the above choices .", "They slept all day ."]))
    assert S.general_candidate("pours the water into a pot") and not S.general_candidate("they pour water out")
    assert S.general_candidate("two cups of flour")
    assert not S.general_candidate("adds 2 cups of flour") and not S.general_candidate("plants")


def test_general_distractors_come_only_from_compatible_other_sources_and_share_no_word():
    item = _general("social_i_qa", "s1", "thankful to her boss", ["get more famous", "move up in life"],
                    state='situation: "Sydney got a promotion from the boss." question: "How will Sydney feel?"')
    pool = [("o-h", "hellaswag", "pours the batter into a round pan"), ("o-c", "cosmos_qa", "They felt proud of the work ."),
            ("o-s", "social_i_qa", "happy it happened at last"), ("o-q", "qasc", "warm air meeting cold air"),
            ("o-k", "commonsense_qa", "a big city hall building"), ("o-h2", "hellaswag", "the boss laughs at the joke")]
    for seed in range(20):
        drawn = S.general_distractors(item, pool, count=3, rnd=random.Random(seed))
        assert drawn is not None and len(set(drawn)) == 3
        assert "They felt proud of the work ." not in drawn and "happy it happened at last" not in drawn
        assert "the boss laughs at the joke" not in drawn, "shares 'boss' with the item"
    assert S.general_distractors(item, pool, count=4, rnd=random.Random(0)) is None
    assert set(S.GENERAL_SOURCES_ALLOWED) == {"hellaswag", "social_i_qa", "cosmos_qa", "qasc", "commonsense_qa"}
    assert all(source not in allowed for source, allowed in S.GENERAL_SOURCES_ALLOWED.items())


def test_content_words_stem_and_drop_stop_words():
    assert "slid" in S.content_words("it slides") and "slid" in S.content_words("sliding") and "the" not in S.content_words("the box")
    assert S.content_words("stops") == S.content_words("stopped") == {"stop"}
