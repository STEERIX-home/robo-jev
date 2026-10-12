"""Task v0.3a — varied narration: fact-preservation checks, question re-voicing checks, record assembly, the sealed-set guard and
the request-log audit, the stratified plan, and the paid runner against a fake transport (no network, no spend)."""

from __future__ import annotations

import copy
import json
import threading

import pytest

from robo_jev.bench.openrouter import DecisionsClient, Ledger, ResponseCache
from robo_jev.launch.providers.runpod import Secret
from robo_jev.simgen import narration as N
from robo_jev.simgen import narration_checks as C
from robo_jev.simgen.narration_run import evaluate, make_job, run_jobs
from robo_jev.simgen.record import dumps, validate_record

BOUNDS = (0.5, 2.5)

ORIGINAL = (
    "Second shift on an automotive sub-assembly line. You are a parts-delivery AMR, returning empty bins to the supermarket rack. "
    "You are moving along the lane behind the stamping presses at 0.65 m/s. You can brake gently (0.9 m/s²), firmly (1.5 m/s²) or "
    "at maximum (3.0 m/s²). Reaction latency is 290 ms, and your footprint radius is 38 cm.\n\n"
    "A line worker in safety glasses is coming toward you down the aisle, 432 cm ahead; the line worker is 5 cm right of the aisle "
    "centre and you are 2 cm left of its centre. The line worker has been slowing down over the last half second.\n\n"
    "You are carrying a mug of black coffee on your top deck: about three-quarters full; it has no lid. Front edge 4 cm, side 4 cm. "
    "Planner advice: brake firmly to a full stop."
)

#: A first-person retelling that keeps every fact (hand-written, as a passing reference).
GOOD = (
    "Planner advice for me right now: brake firmly to a full stop. I'm a parts-delivery AMR working the second shift of an automotive "
    "sub-assembly line, and my job is returning empty bins to the supermarket rack. My speed along the lane behind the stamping presses "
    "is 0.65 m/s; my brakes give 0.9 m/s² gently, 1.5 m/s² firmly and 3.0 m/s² at the maximum. It takes me 290 ms to react, and my "
    "footprint radius measures 38 cm.\n"
    "Coming my way down the aisle: a line worker in safety glasses, 432 cm ahead of me. The line worker sits 5 cm right of the "
    "aisle centre, whereas I sit 2 cm left of it. Over the last half second the line worker kept slowing down.\n"
    "Up on my top deck there is a mug of black coffee, about three-quarters full, with no lid. Front edge 4 cm, side 4 cm."
)

ORIGINAL_KO = (
    "대형 물류 창고의 야간 근무 시간입니다. 당신은 무인 팔레트 운반차입니다.\n"
    "속도: 1.55 m/s, 제동: 부드럽게 1.0 m/s² | 단단히 1.5 m/s² | 최대 3.3 m/s²; 지연 0.44 s\n"
    "휴대용 스캐너를 든 관리자: 170 cm 앞, 당신의 중심선에서 왼쪽으로 12 cm (마지막 갱신 0.05 s 전)\n"
    "단서: 관리자는 누군가와 이야기하고 있으며 당신 쪽을 보고 있지 않습니다."
)
GOOD_KO = (
    "나는 대형 물류 창고에서 야간 근무 중인 무인 팔레트 운반차다. 지금 1.55 m/s로 달리는 중이고, 제동은 부드럽게 1.0 m/s², 단단히 "
    "1.5 m/s², 최대 3.3 m/s²까지 걸 수 있다. 반응 지연은 0.44 s다. 170 cm 앞에 휴대용 스캐너를 든 관리자가 있다. 내 중심선에서 왼쪽으로 "
    "12 cm 떨어져 있고, 이 정보는 0.05 s 전에 갱신됐다. 관리자는 누군가와 이야기하느라 내 쪽을 보지 않는다."
)


def _state(original: str, rewrite: str, language: str = "en") -> dict[str, list[str]]:
    return C.check_state(original, rewrite, language, BOUNDS)


def _failing(result: dict[str, list[str]]) -> set[str]:
    return {name for name, problems in result.items() if problems}


# --------------------------------------------------------------------------
# state checks
# --------------------------------------------------------------------------


def test_a_faithful_retelling_passes_every_check_in_both_languages():
    assert _failing(_state(ORIGINAL, GOOD)) == set()
    assert _failing(_state(ORIGINAL_KO, GOOD_KO, "ko")) == set()


def test_the_original_itself_keeps_every_fact_but_is_rejected_as_a_copy():
    assert _failing(_state(ORIGINAL, ORIGINAL)) == {"copy"}


@pytest.mark.parametrize(
    "old,new,expected",
    [
        ("432 cm ahead", "423 cm ahead", {"numbers", "new_numbers"}),  # a changed number
        ("0.65 m/s", "0.65 m", {"numbers", "new_numbers"}),  # a changed unit
        ("290 ms", "0.29 s", {"numbers", "new_numbers"}),  # a unit conversion (not accepted)
        ("Front edge 4 cm, side 4 cm.", "Front edge 4 cm.", {"numbers"}),  # two facts with one value: both must survive
        ("a mug of black coffee", "a mug of coffee", {"names"}),  # a dropped description
        ("a line worker in safety glasses, 432", "a worker, 432", {"names"}),  # a dropped agent name
        ("4 cm, side 4 cm.", "4 cm, side 4 cm. Bay 7 is closed.", {"new_numbers"}),  # a new number
        ("5 cm right of the", "5 cm left of the", {"directions"}),  # a swapped side
        ("about three-quarters full", "about half full", {"fill"}),  # a changed fill band
        ("with no lid", "with its lid on", {"lid"}),  # a changed lid state
        ("Front edge 4 cm", "That is risky, so I should stop. Front edge 4 cm", {"judgement"}),  # an added judgement
        ("Front edge 4 cm", "Front edge 4 cm (전면)", {"language"}),  # another language
    ],
)
def test_planted_violations_are_rejected(old, new, expected):
    assert old in GOOD
    bad = GOOD.replace(old, new, 1)
    assert _failing(_state(ORIGINAL, bad)) == expected


def test_unit_spellings_are_aliases_but_a_tolerance_must_keep_its_plus_minus():
    assert _failing(_state(ORIGINAL, GOOD.replace("290 ms", "290 milliseconds").replace("38 cm", "38 centimetres"))) == set()
    original = "Measured: peak insertion force 34.0 N (±1.99 N); spec within 2.0 mm."
    assert C.extract_numbers(original) == [C.Mention("34", "N"), C.Mention("1.99", "N", True), C.Mention("2", "mm")]
    assert "numbers" in _failing(_state(original, "Peak force read 34.0 N, give or take 1.99 N; spec says within 2.0 mm."))


def test_number_words_with_a_unit_count_as_numbers_and_codes_are_not_numbers():
    assert C.extract_numbers("Your delivery is two minutes ahead of schedule at gate B12, LH 718.") == [C.Mention("2", "min"), C.Mention("718", "")]
    assert C.extract_numbers("It is 7:45; 1,400 ms; −0.3 mm; AMR-2") == [C.Mention("7:45", "time"), C.Mention("1400", "ms"),
                                                                       C.Mention("-0.3", "mm"), C.Mention("2", "")]
    assert C.extract_numbers("4번 도크, 12분 뒤, 약 4분의 3, 0.5초") == [C.Mention("4", ""), C.Mention("12", "min"), C.Mention("4", ""),
                                                                C.Mention("3", ""), C.Mention("0.5", "s")]


def test_korean_checks_catch_a_dropped_name_a_changed_side_and_an_english_rewrite():
    assert "names" in _failing(_state(ORIGINAL_KO, GOOD_KO.replace("휴대용 스캐너를 든 관리자", "관리자"), "ko"))
    assert "directions" in _failing(_state(ORIGINAL_KO, GOOD_KO.replace("왼쪽으로", "오른쪽으로"), "ko"))
    english = ("I am a driverless pallet mover on the night shift of a large distribution warehouse, at 1.55 m/s; brakes 1.0 m/s², "
               "1.5 m/s², 3.3 m/s²; latency 0.44 s. 휴대용 스캐너를 든 관리자 170 cm ahead, 12 cm to the 왼쪽으로, updated 0.05 s ago.")
    assert "language" in _failing(_state(ORIGINAL_KO, english, "ko"))


def test_time_stamped_log_lines_of_the_held_out_T2_style_are_rejected():
    log = GOOD + "\n[−0.85 s] sensors: line worker 432 cm ahead\n[now] robot: 0.65 m/s"
    assert "log_format" in _failing(_state(ORIGINAL, log))


def test_length_bounds_are_enforced():
    assert "length" in _failing(C.check_state(ORIGINAL, GOOD, "en", (0.5, 0.9)))
    assert "length" in _failing(C.check_state(ORIGINAL, GOOD + " " + "Nothing else is going on here at all." * 40, "en", (0.5, 1.5)))


def test_the_checks_accept_the_engine_texts_as_they_are_except_for_copying():
    from robo_jev.simgen.catalog import default_config
    from robo_jev.simgen.generate import generate_batch

    for record in generate_batch(40, base_seed=5, cfg=default_config()):
        state = record["request"]["state"]
        assert _failing(C.check_state(state, state, record["meta"]["language"], BOUNDS)) == {"copy"}, record["id"]


# --------------------------------------------------------------------------
# question re-voicing checks
# --------------------------------------------------------------------------

QUESTION = {
    "type": "choice",
    "instructions": "If you brake firmly to a full stop, do you touch the line worker within the next 3 seconds, or do you keep going?",
    "criteria": {"a1": "Step aside toward your left", "a2": "Keep going at your current speed (0.65 m/s)."},
}


def test_a_person_only_revoicing_passes_in_first_and_third_person():
    first = {"instructions": "If I brake firmly to a full stop, do I touch the line worker within the next 3 seconds, or do I keep going?",
             "criteria": {"a1": "Step aside toward my left", "a2": "Keep going at my current speed (0.65 m/s)."}}
    third = {"instructions": "If the robot brakes firmly to a full stop, does it touch the line worker within the next 3 seconds, or "
                             "does it keep going?",
             "criteria": {"a1": "Step aside toward its left", "a2": "Keep going at the robot's current speed (0.65 m/s)."}}
    assert C.check_question(QUESTION, first, "en") == []
    assert C.check_question(QUESTION, third, "en") == []
    score = {"type": "score", "instructions": "How likely is it that you touch the patient?", "criteria": ["0 = no chance", "1 = you touch the patient"]}
    assert C.check_question(score, {"instructions": "How likely is it that I touch the patient?",
                                    "criteria": ["0 = no chance", "1 = I touch the patient"]}, "en") == []


@pytest.mark.parametrize(
    "change",
    [
        lambda q: q["criteria"].pop("a2"),  # a dropped option
        lambda q: q.update(criteria={"b1": q["criteria"]["a1"], "a2": q["criteria"]["a2"]}),  # a renamed key
        lambda q: q.update(instructions=q["instructions"].replace("3 seconds", "2 seconds")),  # a changed time window
        lambda q: q.update(instructions=q["instructions"].replace("firmly", "gently")),  # a changed hypothetical action
        lambda q: q["criteria"].update(a2="Keep going at my current speed (0.60 m/s)."),  # a changed number in an option
        lambda q: q.update(instructions=q["instructions"].replace("line worker", "worker")),  # a dropped name
    ],
)
def test_a_revoiced_question_that_changes_more_than_the_person_is_rejected(change):
    revoiced = {"instructions": QUESTION["instructions"].replace("you", "I"), "criteria": {k: v.replace("your", "my") for k, v in QUESTION["criteria"].items()}}
    assert C.check_question(QUESTION, revoiced, "en") == []
    bad = copy.deepcopy(revoiced)
    change(bad)
    assert C.check_question(QUESTION, bad, "en")


def test_korean_revoicing_may_change_only_the_person_token():
    q = {"type": "noul", "instructions": "가령 단단히 제동해 완전히 서면 앞으로 3초 동안 당신은 관리자와 접촉합니까?"}
    assert C.check_question(q, {"instructions": q["instructions"].replace("당신은", "나는")}, "ko") == []
    assert C.check_question(q, {"instructions": q["instructions"].replace("당신은", "로봇은")}, "ko") == []
    assert C.check_question(q, {"instructions": q["instructions"].replace("당신은", "나는").replace("단단히", "부드럽게")}, "ko")
    assert C.check_question(q, {"instructions": q["instructions"].replace("당신은", "나는").replace("3초", "2초")}, "ko")


# --------------------------------------------------------------------------
# reply parsing and record assembly
# --------------------------------------------------------------------------


def _source_record():
    from robo_jev.simgen.catalog import default_config
    from robo_jev.simgen.generate import generate_batch

    for record in generate_batch(30, base_seed=11, cfg=default_config()):
        if record["meta"]["language"] == "en" and any(q["type"] == "choice" for q in record["request"]["questions"].values()):
            return record
    raise AssertionError("no English record with a choice question")


def test_parse_reply_reads_strings_json_objects_and_fences_and_refuses_nested_questions():
    prose, json_voice, first = N.VOICES["en"]["supervisor_note"], N.VOICES["en"]["sensor_json"], N.VOICES["en"]["robot_monologue"]
    assert N.parse_reply('{"state": "  hello  "}', prose) == ("hello", None, [])
    state, _, problems = N.parse_reply('```json\n{"state": {"self": {"speed": "0.65 m/s"}}}\n```', json_voice)
    assert problems == [] and json.loads(state) == {"self": {"speed": "0.65 m/s"}} and state == N.render_state({"self": {"speed": "0.65 m/s"}})
    assert N.parse_reply('{"state": {"narrative": "x", "questions": {}}}', prose)[2] == ["state is not a string"]
    assert N.parse_reply('{"state": "x"}', first)[2] == ["no re-voiced questions"]
    assert N.parse_reply("not json", prose)[2][0].startswith("reply is not JSON")


def test_assembled_records_keep_labels_truth_rationale_generator_and_meta_of_the_source():
    source = _source_record()
    frozen = json.loads(dumps(source))
    voice = N.VOICES["en"]["robot_monologue"]
    names = list(source["request"]["questions"])
    revoiced = {name: {"instructions": q["instructions"] + " ", **({"criteria": q["criteria"]} if "criteria" in q else {})}
                for name, q in source["request"]["questions"].items()}
    status = {name: ("revoiced" if i == 0 else "original (test)") for i, name in enumerate(names)}
    record = N.assemble_record(source, voice=voice, state="I am the robot.", questions=revoiced, question_status=status,
                               narration={"voice": voice.key, "source_id": source["id"]})
    assert json.loads(dumps(source)) == frozen  # the source is untouched
    assert validate_record(record) == []
    assert record["id"] == f"{source['id']}~robot_monologue"
    for key in ("schema", "generator", "labels", "truth", "rationale"):
        assert dumps(record[key]) == dumps(source[key])
    assert {k: v for k, v in record["meta"].items() if k != "narration"} == source["meta"]
    assert record["meta"]["narration"] == {"voice": "robot_monologue", "source_id": source["id"]}
    assert record["request"]["state"] == "I am the robot."
    assert record["request"]["questions"][names[0]]["instructions"].endswith(" ")
    for name in names[1:]:
        assert record["request"]["questions"][name] == source["request"]["questions"][name]
    assert list(record["request"]["questions"]) == names and list(record["labels"]) == names


# --------------------------------------------------------------------------
# sealed-set guard and request-log audit
# --------------------------------------------------------------------------


def _sealed_files(tmp_path, engine_record, outside_text):
    engine = tmp_path / "T1.jsonl"
    engine.write_text(dumps(engine_record) + "\n", encoding="utf-8")
    outside = tmp_path / "T8b.jsonl"
    outside_record = {"id": "t8b-0", "request": {"state": outside_text, "questions": {"q": {"type": "noul", "instructions": "Does it tip?"}}}}
    outside.write_text(json.dumps(outside_record) + "\n", encoding="utf-8")
    return N.SealedIndex.build({"T1": engine, "T8b": outside})


OUTSIDE_TEXT = "An invented outside moment for the test.\nThe crate on the cart is tall and narrow and it rests near the front edge."


def test_the_sealed_index_flags_a_sealed_source_and_outside_text_units_but_not_ordinary_training_text(tmp_path):
    from robo_jev.simgen.catalog import default_config
    from robo_jev.simgen.generate import generate_batch

    records = list(generate_batch(3, base_seed=21, cfg=default_config()))
    index = _sealed_files(tmp_path, records[0], OUTSIDE_TEXT)
    assert index.hits(source=records[1], texts=[records[1]["request"]["state"]]) == []
    assert "T1: source request" in index.hits(source=records[0], texts=[])
    assert any(h.startswith("T1: text") for h in index.hits(texts=["  " + records[0]["request"]["state"].replace("\n", " \n ") + " "]))
    leaked = records[1]["request"]["state"] + "\nThe crate on the cart is tall and narrow and it rests near the front edge."
    assert any(h.startswith("T8b:") for h in index.hits(texts=[leaked]))
    with pytest.raises(N.SealedTextError):
        index.guard(source=records[1], texts=[leaked])
    round_trip = N.SealedIndex.from_json(json.loads(json.dumps(index.to_json())))
    assert round_trip.hits(source=records[0], texts=[]) == index.hits(source=records[0], texts=[])


class FakeTransport:
    """Chat-completions stand-in: returns a reply built by `reply(body)` and a usage cost; counts calls; never sees the network."""

    def __init__(self, reply, cost=0.0001):
        self.reply, self.cost, self.calls, self.bodies = reply, cost, 0, []
        self.lock = threading.Lock()

    def __call__(self, method, url, headers, body, timeout):
        with self.lock:
            self.calls += 1
            self.bodies.append(body)
        data = json.loads(body)
        content = self.reply(data)
        response = {"id": "gen-test", "model": data["model"], "provider": "Fake",
                    "choices": [{"message": {"role": "assistant", "content": content}, "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 100, "completion_tokens": 50, "cost": self.cost}}
        return 200, {}, json.dumps(response).encode("utf-8")


def _setup(tmp_path, reply, *, cap=1.0, cost=0.0001, sealed_source=None):
    from robo_jev.simgen.catalog import default_config
    from robo_jev.simgen.generate import generate_batch

    records = [r for r in generate_batch(12, base_seed=31, cfg=default_config()) if r["meta"]["language"] == "en"][:3]
    index = _sealed_files(tmp_path, sealed_source or {"id": "x", "request": {"state": "nothing", "questions": {}}}, OUTSIDE_TEXT)
    rows = [{"id": r["id"], "scene": r["meta"]["scene"], "group": r["meta"]["group"], "language": "en"} for r in records]
    plan = N.build_plan(rows)
    by_id = {r["id"]: r for r in records}
    price = {"prompt": 1e-7, "completion": 4e-7}
    jobs = [make_job(e, by_id[e["source_id"]], v, "fake/model", price) for e in plan for v in e["voices"]]
    transport = FakeTransport(reply, cost=cost)
    client = DecisionsClient(Secret("sk-or-v1-TESTKEY-0123456789abcdef"), url=N.CHAT_URL, transport=transport, sleep=lambda s: None)
    ledger = Ledger(tmp_path / "ledger.json", max_total_usd=cap, max_model_usd=cap)
    cache = ResponseCache(tmp_path / "cache")
    return jobs, client, ledger, cache, index, transport, records


def _identity_reply(body):
    user = body["messages"][1]["content"]
    state = user.split("ORIGINAL TEXT:\n<<<\n", 1)[1].split("\n>>>", 1)[0]
    return json.dumps({"state": state})


def test_the_runner_caches_replies_so_a_rerun_costs_nothing_and_never_writes_the_key(tmp_path):
    jobs, client, ledger, cache, index, transport, _ = _setup(tmp_path, _identity_reply)
    log = tmp_path / "requests.log.jsonl"
    first = run_jobs("fake/model", jobs, client=client, cache=cache, ledger=ledger, price={}, index=index, calls_log=log, concurrency=4)
    assert first["stats"]["sent"] == len(jobs) == transport.calls and not first["errors"]
    again = run_jobs("fake/model", jobs, client=client, cache=cache, ledger=ledger, price={}, index=index, calls_log=log, concurrency=4)
    assert again["stats"]["cached"] == len(jobs) and transport.calls == len(jobs)
    assert ledger.spent() == pytest.approx(0.0001 * len(jobs))
    for path in [log, tmp_path / "ledger.json", *(tmp_path / "cache").rglob("*.json")]:
        assert "TESTKEY" not in path.read_text(encoding="utf-8")
    # an identity "rewrite" keeps every fact but is rejected as a copy (and first/third-person voices lack questions)
    for job in jobs:
        result = evaluate(job, first["responses"][job.job_id])
        assert not result["passed"]
        assert result["checks"].get("copy") or result["checks"].get("parse")


def test_the_runner_stops_before_the_cap(tmp_path):
    jobs, client, ledger, cache, index, transport, _ = _setup(tmp_path, _identity_reply, cap=0.0003, cost=0.0001)
    out = run_jobs("fake/model", jobs, client=client, cache=cache, ledger=ledger, price={"prompt": 1e-7, "completion": 4e-7}, index=index,
                   calls_log=tmp_path / "log.jsonl", concurrency=1)
    assert out["stats"]["budget"] and "budget" in out["skipped"].values()
    assert ledger.spent() <= 0.0003 + 1e-12 and transport.calls < len(jobs)


def test_a_sealed_source_is_never_sent_and_the_audit_flags_a_planted_sealed_hash(tmp_path):
    from robo_jev.simgen.catalog import default_config
    from robo_jev.simgen.generate import generate_batch

    records = [r for r in generate_batch(12, base_seed=31, cfg=default_config()) if r["meta"]["language"] == "en"][:3]
    jobs, client, ledger, cache, index, transport, _ = _setup(tmp_path, _identity_reply, sealed_source=records[0])
    log = tmp_path / "requests.log.jsonl"
    out = run_jobs("fake/model", jobs, client=client, cache=cache, ledger=ledger, price={}, index=index, calls_log=log, concurrency=2)
    sealed_jobs = [j for j in jobs if j.source["id"] == records[0]["id"]]
    assert sealed_jobs and out["stats"]["refused_sealed"] == len(sealed_jobs)
    sent_states = [json.loads(b)["messages"][1]["content"] for b in transport.bodies]
    assert not any(records[0]["request"]["state"] in s for s in sent_states)
    audit = N.audit_request_log(log, index, source_ids={r["id"] for r in records})
    assert audit["requests"] == len(jobs) - len(sealed_jobs) and audit["violations"] == []
    # plant violations in a copy of the log: a sealed state hash, an outside unit, an unknown source id
    entry = json.loads(log.read_text(encoding="utf-8").splitlines()[0])
    planted = [
        {**entry, "sent": {**entry["sent"], "text_sha256": [N.state_sha256(records[0]["request"]["state"])]}},
        {**entry, "sent": {**entry["sent"], "units": sorted(index.units["T8b"])[:1]}},
        {**entry, "source_id": "t8b-0"},
    ]
    bad_log = tmp_path / "bad.log.jsonl"
    bad_log.write_text("".join(json.dumps(e) + "\n" for e in planted), encoding="utf-8")
    violations = N.audit_request_log(bad_log, index, source_ids={r["id"] for r in records})["violations"]
    assert len(violations) == 3


# --------------------------------------------------------------------------
# plan
# --------------------------------------------------------------------------


def test_the_plan_is_stratified_deterministic_and_balances_two_distinct_voices_per_moment():
    rows = []
    for i in range(600):
        scene = ["a", "b", "c"][i % 3] if i < 450 else "d"
        language = "ko" if i % 7 == 0 else "en"
        rows.append({"id": f"r{i:04d}", "scene": scene, "group": scene.upper(), "language": language})
    plan = N.build_plan(rows, seed=3)
    assert plan == N.build_plan(list(reversed(rows)), seed=3)
    assert [e["rank"] for e in plan] == list(range(600)) and len({e["source_id"] for e in plan}) == 600
    for entry in plan:
        voices = entry["voices"]
        assert len(set(voices)) == 2 and all(v in N.VOICES[entry["language"]] for v in voices)
        assert set(entry["order"]) == set(voices)
    # any prefix is close to proportional per stratum
    total = {}
    for row in rows:
        total[(row["scene"], row["language"])] = total.get((row["scene"], row["language"]), 0) + 1
    prefix = plan[:120]
    for key, n in total.items():
        got = sum(1 for e in prefix if (e["scene"], e["language"]) == key)
        assert abs(got - n * 120 / 600) <= 1.5, key
    # voices are balanced within a stratum
    en_a = [v for e in plan if (e["scene"], e["language"]) == ("a", "en") for v in e["voices"]]
    counts = {v: en_a.count(v) for v in N.VOICES["en"]}
    assert max(counts.values()) - min(counts.values()) <= 2


def test_every_voice_language_pair_has_ten_voices_and_person_rules_where_it_revoices():
    for language in ("en", "ko"):
        assert len(N.VOICES[language]) == 10
        for voice in N.VOICES[language].values():
            assert voice.language == language and voice.bounds[0] < voice.bounds[1]
            if N.revoices_questions(voice):
                assert (language, voice.person) in N._PERSON_RULES


# --------------------------------------------------------------------------
# actions list, above/below, the back-check and the full pipeline
# --------------------------------------------------------------------------

PLANNER_TEXT = (
    "CONTEXT: Night shift in a large distribution warehouse.\nTASK: you are a driverless pallet mover\nROBOT STATE\n  speed: 1.55 m/s\n"
    "PLANNER RECOMMENDATION: brake as hard as you can\nAVAILABLE ACTIONS\n  - keep going at your current speed\n  - ease to a gentle stop\n"
    "  - brake as hard as you can"
)


def test_every_item_of_the_list_of_possible_actions_must_survive():
    assert C.action_items(PLANNER_TEXT) == ["keep going at your current speed", "ease to a gentle stop", "brake as hard as you can"]
    kept = "I am a driverless pallet mover on the night shift of a large distribution warehouse, at 1.55 m/s. Planner: brake as hard as I can.\nMy options: keep going at my current speed, ease to a gentle stop, or brake as hard as I can."
    assert C._action_problems(PLANNER_TEXT, kept, "en") == []
    dropped = kept.replace("keep going at my current speed, ease to a gentle stop, or ", "")
    assert C._action_problems(PLANNER_TEXT, dropped, "en") == ["action not kept: keep going at your current speed", "action not kept: ease to a gentle stop"]


def test_above_and_below_a_nominal_reading_must_not_flip():
    assert "directions" in _failing(_state("Measured: height 1.8 mm above nominal (gauge accuracy ±1.02 mm).",
                                           "Height reads 1.8 mm below nominal; gauge accuracy ±1.02 mm."))
    assert "directions" in _failing(_state("측정값: 높이가 기준보다 1.8 mm 높음(게이지 정확도 ±1.02 mm)",
                                           "높이는 기준 대비 1.8 mm 낮음, 게이지 정확도는 ±1.02 mm다.", "ko"))


def _chat(content: str, cost: float = 0.0001) -> dict:
    return {"choices": [{"message": {"content": content}, "finish_reason": "stop"}], "usage": {"prompt_tokens": 10, "completion_tokens": 5, "cost": cost}}


def test_the_back_check_reads_the_last_json_object_and_ignores_self_cleared_items():
    from robo_jev.simgen.narration_run import judge_accepts, parse_judge

    assert parse_judge(_chat('{"differences": []}'))["ok"] is True
    fenced = '```json\n{"differences": [{"severity": "minor", "what": "the setting is missing"}]}\n```'
    assert parse_judge(_chat(fenced)) == {"ok": True, "major": 0, "minor": 1, "differences": [{"severity": "minor", "what": "the setting is missing"}]}
    corrected = ('{"differences": [{"severity": "major", "what": "the side is swapped"}]}\nCorrection, on review: {"differences": '
                 '[{"severity": "major", "what": "The pace wording differs but is equivalent, so no error"}]}')
    assert parse_judge(_chat(corrected))["ok"] is True
    major = parse_judge(_chat('{"differences": [{"severity": "major", "what": "the worker is now said to be visible"}]}'))
    assert major["ok"] is False and not judge_accepts(major)
    two_minor = parse_judge(_chat('{"differences": [{"severity": "minor", "what": "a"}, {"severity": "minor", "what": "b"}]}'))
    assert two_minor["ok"] is False
    assert parse_judge(_chat("I cannot judge this."))["ok"] is None


def test_the_pipeline_judges_only_rewrites_that_pass_the_checks_and_counts_both_models_against_one_cap(tmp_path):
    from robo_jev.simgen.narration_run import judge_accepts, run_pipeline

    def reply(body):
        if body["model"] == "fake/judge":
            return json.dumps({"differences": []})
        user = body["messages"][1]["content"]
        state = user.split("ORIGINAL TEXT:\n<<<\n", 1)[1].split("\n>>>", 1)[0]
        # a "rewrite" that reorders lines (passes the copy check when the original has several lines) and keeps the questions
        lines = state.split("\n")
        rewritten = "\n".join(reversed(lines))
        out = {"state": rewritten}
        if "QUESTIONS to re-voice" in user:
            block = user.split("QUESTIONS to re-voice", 1)[1].split("<<<\n", 1)[1].split("\n>>>", 1)[0]
            out["questions"] = json.loads(block)
        return json.dumps(out)

    jobs, client, ledger, cache, index, transport, _ = _setup(tmp_path, reply, cap=1.0)
    prices = {"fake/model": {"prompt": 1e-7, "completion": 4e-7}, "fake/judge": {"prompt": 1e-7, "completion": 4e-7}}
    out = run_pipeline("fake/model", "fake/judge", jobs, client=client, cache=cache, ledger=ledger, prices=prices, index=index,
                       calls_log=tmp_path / "log.jsonl", concurrency=3)
    counts = out["stats"]["counts"]
    assert counts["done"] == len(jobs) == counts["replies"]
    judged_models = [json.loads(b)["model"] for b in transport.bodies]
    assert judged_models.count("fake/judge") == counts["checks_passed"] == counts["judged"]
    for job_id, item in out["results"].items():
        assert (item["verdict"] is not None) == item["result"]["passed"]
        if item["verdict"] is not None:
            assert judge_accepts(item["verdict"])
    assert ledger.spent() == pytest.approx(0.0001 * transport.calls)
    assert set(ledger.models) <= {"fake/model", "fake/judge"}
