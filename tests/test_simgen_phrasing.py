"""학습 문구 판 v01과 사실 순서 섞기 (Task v0.1a §1-1 ~ §1-4).

* **판의 기본값은 v0b**이고 v0b의 학습·빼 둔 문구 목록은 바이트 단위로 그대로다(고정한 해시). 봉인된 시험 세트 T1~T7과 v0b 학습 세트의
  첫 레코드들을 지금 코드로 다시 만들면 고정한 해시(그리고 파일이 있으면 그 줄)와 같다 — 시험 세트 생성 경로는 늘 v0b이고, 기본값이 아닌
  손잡이를 시험 나눔에 주면 거절한다.
* **새 문구**(영어 질문 열쇠 92개·선택지 설명 41개, 한국어 같은 표): 수(영어 ≥ 4, 가정 질문 ≥ 6, 한국어 ≥ 2), 빼 둔 문구와 같지 않고
  낱말(어절) 유사도 < 0.8, 빼 둔 문구에만 있는 3-gram 없음, 자리 표시는 그 열쇠의 v0b 학습 문구의 것만, 뜻 규칙(충돌을 세는 열쇠는 충돌을
  말하고 세지 않는 열쇠는 말하지 않는다, 닿음·가까움의 3초 창, 숨은 이름 없음, 한국어 조사 자리 표시).
* **같은 seed → 같은 순간·같은 정답**: v0b와 v01(문구 + 사실 순서 섞기)로 같은 seed 200개를 만들면 정답이 바이트 단위로 같고 요청 글만
  다르다; 사실 순서를 섞어도 정답은 같고 상태 글은 같은 사실의 재배열이다; v01 생성은 결정적이고 레코드의 칸만으로 다시 만들어진다.
* 만들어 둔 train-50k-v01이 있으면: manifest의 sha256, 사실 순서 섞기 비율 ≈ 0.5, 판 장부.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

import pytest

from robo_jev.simgen import phrasing as ph
from robo_jev.simgen.catalog import TRAIN_GROUPS, default_config
from robo_jev.simgen.generate import DEFAULT_PHRASING, assemble, build_moment, generate_batch, generate_unit, plan_units, rebuild_record
from robo_jev.simgen.holdout import load_holdout, split_for
from robo_jev.simgen.questions import Surface
from robo_jev.simgen.record import dumps, validate_record

ARTIFACTS = Path(__file__).resolve().parents[1] / "artifacts" / "simgen"
V0B_TRAIN = ARTIFACTS / "v0b" / "train-50k.jsonl"
V0B_TESTS = ARTIFACTS / "v0b" / "tests"
V01_TRAIN = ARTIFACTS / "v01" / "train-50k.jsonl"
V02_TRAIN = ARTIFACTS / "v02" / "train-50k.jsonl"

#: v0b 문구 목록(학습·빼 둔 — 영어·한국어, 질문 열쇠와 선택지 설명 전부)의 sha256 — 바이트 단위로 손대지 않는다
V0B_BANKS_SHA256 = "131d9def2a5c826cde118c55ed626f59f10e4e30d0c8dac0760202882f4dbba1"
#: 봉인된 시험 세트의 첫 레코드들(세트마다 build_set의 앞부분: T1~T4 셋, T5 쌍 둘, T6 기준 하나와 그 변형, T7 둘)과 v0b 학습 세트의 첫 단위 여섯 —
#: 생성기 git 030062b(봉인할 때의 커밋)로 다시 만든 줄들의 sha256(봉인된 파일의 같은 줄과 같다)
SEALED_GIT = "030062b"
SEALED_PREFIX = {
    "T1": (3, 3, "5639202a68b1712e42e399be9e13191628fda09334b9a97d8ce7a0a590ecb3d8"),
    "T2": (3, 3, "6c11bdb9b4849f4e1372d9af8f4bce4b24631f1e9ec8d89aef50f18f3f1076a5"),
    "T3": (3, 3, "4c794ae82fc271eaaaa2921f24e949345397e019da2f752632f949af77ed04b1"),
    "T4": (3, 3, "716d068fe6b246f77c2ee1338ec4a1a45657d5401015ffe3a166123442c08594"),
    "T5": (2, 4, "6eb60098265fa63132fefa3831c3b7068d190bcbd2ec5716117753ed21b95a0a"),
    "T6": (1, 12, "454a495e403ef3f50b027fa7ff0faf24af42c579da81d4beb88eeccaf8d9a096"),
    "T7": (2, 2, "0e5e257ece3b48076366d2f14e06acb063dc6b7946e19526286e351cc69c9b6a"),
}
V0B_TRAIN_PREFIX = (6, 6, "2c1354f4e2b0ea01b0d7ab197cde4bdbe112d99db0e9461cdfb34a052a0cbc81")


@pytest.fixture(scope="module")
def cfg():
    return default_config()


def _v0b_banks_digest() -> str:
    data = {lang: {table: {key: {"train": bank["train"], "heldout": bank["heldout"]} for key, bank in keys.items()}
                   for table, keys in ph.base_tables(lang).items()} for lang in ("en", "ko")}  # fmt: skip
    return hashlib.sha256(json.dumps(data, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()


def _sha(lines: list[str]) -> str:
    return hashlib.sha256("".join(lines).encode("utf-8")).hexdigest()


# ----------------------------------------------------------------------------------------------------------------
# 기본값과 재현
# ----------------------------------------------------------------------------------------------------------------


def test_the_default_phrasing_is_v0b():
    assert DEFAULT_PHRASING == ph.DEFAULT_PHRASING == "v0b"
    assert Surface().phrasing == "v0b"
    import inspect

    from robo_jev.simgen import generate

    for fn in (generate.assemble, generate.generate_unit, generate.generate_batch, generate.generate_record):
        params = inspect.signature(fn).parameters
        assert params["phrasing"].default == "v0b" and params["fact_shuffle"].default == 0.0, fn.__name__


def test_the_v0b_banks_are_untouched():
    assert _v0b_banks_digest() == V0B_BANKS_SHA256


@pytest.mark.parametrize("name", sorted(SEALED_PREFIX))
def test_sealed_test_sets_reproduce_byte_for_byte(cfg, name):
    """T1~T7의 앞부분을 지금 코드로(기본값 — 시험 세트 생성 경로) 다시 만들면 고정한 해시와 같고, 봉인된 파일이 있으면 그 줄과 같다."""
    from itertools import islice

    from robo_jev.simgen.testsets import build_set

    n, lines_expected, digest = SEALED_PREFIX[name]
    records, _ = build_set(cfg, name, n=n, git=SEALED_GIT, holdout=load_holdout())
    lines = [dumps(r) + "\n" for r in records]
    assert len(lines) == lines_expected and _sha(lines) == digest, name
    path = V0B_TESTS / f"{name}.jsonl"
    if path.exists():
        with path.open(encoding="utf-8") as handle:
            assert lines == list(islice(handle, len(lines)))


def test_the_v0b_training_set_reproduces_byte_for_byte(cfg):
    from itertools import islice

    holdout = load_holdout()
    n, lines_expected, digest = V0B_TRAIN_PREFIX
    units = plan_units(50_000, 0, groups=TRAIN_GROUPS, seed_start=holdout.seed_ranges["train"][0])[:n]
    lines = [dumps(r) + "\n" for u in units for r in generate_unit(cfg, u, split=split_for("train", holdout), git=SEALED_GIT)]
    assert len(lines) == lines_expected and _sha(lines) == digest
    if V0B_TRAIN.exists():
        with V0B_TRAIN.open(encoding="utf-8") as handle:
            assert lines == list(islice(handle, len(lines)))


@pytest.mark.parametrize("knob", [{"phrasing": "v01"}, {"phrasing": "v01b"}, {"fact_shuffle": 0.5}])
def test_non_default_knobs_are_refused_outside_the_training_split(cfg, knob):
    moment = build_moment(cfg, 2_000_123, "cafe", {"group": "P1"}, None, split_for("T1"))
    with pytest.raises(ValueError, match="training split only"):
        assemble(cfg, moment, 0, **knob)
    train = build_moment(cfg, 1_000_123, "cafe", {"group": "P1"}, None, split_for("train"))
    with pytest.raises(ValueError, match="training split only"):
        assemble(cfg, train, 0, t6={"fact_seed": 3}, **knob)


# ----------------------------------------------------------------------------------------------------------------
# 새 문구 표
# ----------------------------------------------------------------------------------------------------------------


def _entries():
    for lang in ("en", "ko"):
        for table, keys in ph.base_tables(lang).items():
            for key in keys:
                yield lang, table, key


@pytest.mark.parametrize("lang", ["en", "ko"])
def test_every_key_has_new_wordings_and_no_unknown_key(lang):
    base, new = ph.base_tables(lang), ph.new_tables(lang)
    assert set(new) == set(base) == set(ph.TABLES)
    for table in ph.TABLES:
        assert set(new[table]) == set(base[table]), (lang, table, set(base[table]) ^ set(new[table]))


@pytest.mark.parametrize("lang", ["en", "ko"])
def test_new_wordings_keep_away_from_held_out_ones_and_keep_the_meaning(lang):
    """수·중복·빼 둔 문구 근접(같음, 유사도 < 0.8, 빼 둔 문구에만 있는 3-gram)·자리 표시·뜻 규칙 — 열쇠마다 전부."""
    problems = []
    for lang_, table, key in _entries():
        if lang_ == lang:
            problems += ph.violations(lang, table, key, ph.new_tables(lang)[table][key])
    assert not problems, "\n".join(problems[:30])


@pytest.mark.parametrize("lang", ["en", "ko"])
def test_no_new_wording_uses_held_out_only_vocabulary_of_any_key(lang):
    """리뷰 1 M1 — 언어 전체 기준: 새 문구는 어느 열쇠의 빼 둔 문구에만 있는 내용어(불용어가 아닌 낱말·어절)와, 그런 3-gram 가운데 내용어를
    하나 이상 담은 것을 쓰지 않는다(열쇠별 금지만으로는 다른 열쇠의 빼 둔 말 — 'strike', 'What is your next step?' — 이 학습에 들어갔다)."""
    only_words, only_grams = ph.held_out_only(lang)
    assert only_words and only_grams                # 규칙이 비어 있지 않다(빼 둔 문구에만 있는 말이 실제로 있다)
    hits = []
    for _, table, key in (e for e in _entries() if e[0] == lang):
        for template in ph.new_tables(lang)[table][key]:
            words, grams = ph.language_wide_hits(lang, template)
            if words or grams:
                hits.append((key, template, words, grams))
    assert not hits, hits[:20]


@pytest.mark.parametrize("lang", ["en", "ko"])
def test_new_wordings_are_not_character_level_near_copies_of_held_out_ones(lang):
    """리뷰 1 M2 — 낱말 유사도(< 0.8)에 더해 정규화한 글의 글자 유사도도 같은 열쇠의 빼 둔 문구와 0.75 미만(낱말만 바꾼 베끼기 금지)."""
    worst = []
    for _, table, key in (e for e in _entries() if e[0] == lang):
        for template in ph.new_tables(lang)[table][key]:
            score = max(ph.char_similarity(template, h) for h in ph.base_tables(lang)[table][key]["heldout"])
            if score >= ph.MAX_CHAR_SIMILARITY:
                worst.append((key, template, round(score, 3)))
    assert not worst, worst


@pytest.mark.parametrize("lang", ["en", "ko"])
def test_new_wordings_keep_every_placeholder_their_key_always_has(lang):
    """리뷰 1 M3 — 그 열쇠의 v0b 학습 문구 모두에 있는 자리 표시는 새 문구에도 있다; 가정 질문은 행동 자리(영어 {action}, 한국어 {act_if}·{act_noun})를
    빠뜨리지 않는다(가정 행동을 읽지 않는 것이 v0의 T2 실패 꼴이다)."""
    missing = []
    for _, table, key in (e for e in _entries() if e[0] == lang):
        need = ph.required_placeholders(lang, table, key)
        for template in ph.new_tables(lang)[table][key]:
            have = ph.placeholders(template)
            if need - have:
                missing.append((key, template, sorted(need - have)))
            if table == "instructions" and key in ph.counterfactual_keys():
                assert have & ({"action"} if lang == "en" else {"act_if", "act_noun"}), (key, template)
    assert not missing, missing


def test_the_language_wide_character_and_placeholder_rules_catch_their_cases():
    """규칙 자체의 대조(리뷰 1 M1~M3): 다른 열쇠의 빼 둔 말('strike', 'your next step'), 낱말만 바꾼 빼 둔 문구, 행동 자리를 뺀 가정 질문이 걸린다."""
    def flagged(lang, key, text, what, table="instructions"):
        return any(what in v for v in ph.violations(lang, table, key, [text]))

    assert "strike" in ph.held_out_only("en")[0]
    assert flagged("en", "noul.contact", "Suppose you {action}. Do you strike {ref} within the next 3 s?", "language-wide")
    assert flagged("en", "act.p1_crossing_stopped", "Waiting at the edge of the crossing, what is your next step?", "language-wide")
    assert flagged("en", "act.p5_braking", "Which braking strength do you pick for stopping before {obstacle}?", "character similarity")
    assert flagged("en", "noul.contact", "Will you touch {ref} within the next 3 seconds?", "are missing")
    assert flagged("ko", "noul.contact", "앞으로 3초 안에 {ref:과} 닿습니까?", "action slot")
    assert flagged("ko", "noul.contact", "{act_if} 앞으로 3초 안에 {ref:과} 닿는지 판단하십시오.", "language-wide")
    assert flagged("ko", "start_gentle", "급하게 나가지 않고 약한 가속으로 서서히 출발합니다.", "are missing", table="desc.main")
    # 불용어만 겹치는 말(빼 둔 문구에만 있는 3-gram이어도 내용어가 없으면)은 막지 않는다
    assert not ph.is_content("what", "en") and not ph.is_content("{}", "en") and ph.is_content("strike", "en")


def test_counterfactual_keys_have_at_least_six_new_english_wordings():
    keys = ph.counterfactual_keys()
    assert len(keys) == 39 and {"noul.topple_stack", "score.topple_stack", "noul.contact", "noul.overrun"} <= keys
    for key in keys:
        assert len(ph.new_tables("en")["instructions"][key]) >= 6, key


def test_the_proximity_rules_catch_a_held_out_wording():
    """규칙 자체의 대조: 빼 둔 문구(또는 그 3-gram 하나)를 새 문구로 넣으면 걸린다."""
    held = ph.base_tables("en")["instructions"]["noul.topple_stack"]["heldout"][0]
    assert any("equals a held-out wording" in v for v in ph.violations("en", "instructions", "noul.topple_stack", [held]))
    near = "Suppose you {action}: does {sref} come off the top?"
    assert any("held-out-only 3-grams" in v for v in ph.violations("en", "instructions", "noul.topple_stack", [near]))
    held_ko = ph.base_tables("ko")["instructions"]["noul.topple_stack"]["heldout"][0]
    assert any("similarity" in v for v in ph.violations("ko", "instructions", "noul.topple_stack", [held_ko]))
    assert any("placeholders" in v for v in ph.violations("en", "instructions", "noul.contact", ["If you {action}, will {Ref} be touched within the next 3 s?"]))
    assert any("collision" in v for v in ph.violations("en", "instructions", "noul.tip", ["If you {action}, will a collision tip {lref} over?"]))


def test_new_wordings_fill_with_every_placeholder_their_key_allows():
    from robo_jev.simgen.korean import kfmt

    for lang, table, key in _entries():
        bank = ph.base_tables(lang)[table][key]
        names = set().union(*(ph.placeholders(t) for t in bank["train"]))
        values = {name: f"<{name}>" for name in names}
        for template in ph.new_tables(lang)[table][key]:
            text = kfmt(template, **values) if lang == "ko" else template.format(**values)
            assert "{" not in text and "}" not in text, (lang, key, template)


def test_the_wide_pool_starts_with_the_v0b_pool_in_the_same_order(cfg):
    """넓은 묶음의 앞부분 = v0b 학습 묶음(같은 순서) — 질문 열쇠와 선택지 설명(실제 관측으로 채워서, 두 언어)."""
    from robo_jev.simgen import phrases, phrases_ko
    from robo_jev.simgen.rollout import available_options

    for key in phrases.EN:
        for lang in ("en", "ko"):
            v0b = phrases.bank(key, lang, "train")
            assert phrases.bank(key, lang, "train_v01")[: len(v0b)] == v0b
    cases = [("p1_step_in", "mobile_base"), ("p1_step_in", "arm"), ("p1_head_on", "mobile_base"), ("p5_turning", "humanoid"), ("p6_blind_corner", "mobile_base"),
             ("p4_insert", "arm"), ("p3_place_stack", "humanoid"), ("p3_ramp", "mobile_base"), ("p6_sensor_conflict", "mobile_base")]  # fmt: skip
    for i, (scene, body) in enumerate(cases):
        for lang, describe in (("en", phrases.option_descriptions), ("ko", phrases_ko.option_descriptions)):
            m = build_moment(cfg, 4_400 + i, "warehouse" if body != "arm" else "lab", {"scene": scene, "body": body, "lang": lang})
            for canon in available_options(m.obs):
                v0b = describe(canon, m.obs, "train")
                wide = describe(canon, m.obs, "train_v01")
                table, key = ph.description_table(canon, m.obs.scene, m.obs.body)
                assert wide[: len(v0b)] == v0b and len(wide) == len(ph.wide(lang, table, key)), (scene, body, canon)


# ----------------------------------------------------------------------------------------------------------------
# 같은 seed → 같은 순간·같은 정답
# ----------------------------------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def paired_batches(cfg):
    """같은 seed 200개(쌍 포함): v0b 기본값과 v01(문구 + 사실 순서 섞기 0.5)."""
    a = list(generate_batch(200, base_seed=21, cfg=cfg, git="fixed"))
    b = list(generate_batch(200, base_seed=21, cfg=cfg, git="fixed", phrasing="v01", fact_shuffle=0.5))
    return a, b


def test_v0b_and_v01_give_the_same_moments_and_labels_and_differ_only_in_the_request_text(paired_batches):
    a, b = paired_batches
    assert len(a) == len(b) == 200
    differs = Counter()
    for ra, rb in zip(a, b, strict=True):
        assert dumps(ra["labels"]) == dumps(rb["labels"]), ra["id"]
        assert ph.without_text(ra) == ph.without_text(rb), ra["id"]
        assert not validate_record(rb), rb["id"]
        assert rb["meta"]["phrasing"]["version"] == "v01" and rb["generator"]["phrasing"] == "v01" and "phrasing" not in ra["meta"]
        qa, qb = ra["request"]["questions"], rb["request"]["questions"]
        differs["instructions"] += sum(qa[n]["instructions"] != qb[n]["instructions"] for n in qa)
        differs["questions"] += len(qa)
        differs["state"] += ra["request"]["state"] != rb["request"]["state"]
        if ra["request"]["state"] != rb["request"]["state"]:
            assert rb["meta"]["fact_order"]["shuffled"] and rb["meta"]["fact_order"]["changed"]
            assert Counter(ra["request"]["state"]) == Counter(rb["request"]["state"])     # 같은 글자들의 재배열
    # 새 문구가 실제로 쓰인다(넓은 묶음에서 새 문구의 몫 ≈ 절반 이상) — 지시문의 상당수가 v0b와 다르다
    assert differs["instructions"] > 0.4 * differs["questions"], differs
    assert 50 <= differs["state"] <= 120, differs


def test_v01_picks_are_recorded_and_new_wordings_are_used(paired_batches):
    _, b = paired_batches
    stats = ph.wording_stats(b)
    uses = [row for lang in stats.values() for table in lang.values() for row in table.values()]
    assert sum(r["uses"] for r in uses) > 1500
    new_share = sum(r["new_share"] * r["uses"] for r in uses) / sum(r["uses"] for r in uses)
    assert 0.45 < new_share < 0.8, new_share


def test_fact_shuffle_keeps_the_labels_and_only_reorders_the_facts(cfg):
    """같은 순간에서 fact_seed가 있든 없든 정답이 바이트 단위로 같고, 글머리 문체는 같은 줄의 재배열이다."""
    seen = 0
    for i, style in enumerate(("facts", "prose", "planner") * 4):
        m = build_moment(cfg, 7_700 + i, "hospital", {"style": style})
        plain = assemble(cfg, m, 0)
        shuffled = assemble(cfg, m, 0, t6={"fact_seed": 12_345 + i})
        assert dumps(plain["labels"]) == dumps(shuffled["labels"])
        assert plain["request"]["questions"] == shuffled["request"]["questions"]
        if style in ("facts", "planner"):
            assert Counter(plain["request"]["state"].split("\n")) == Counter(shuffled["request"]["state"].split("\n"))
        seen += plain["request"]["state"] != shuffled["request"]["state"]
    assert seen >= 6


def test_fact_shuffle_share_is_about_one_half(paired_batches):
    _, b = paired_batches
    stats = ph.fact_order_stats(b)
    assert 0.38 <= stats["shuffled_share"] <= 0.62, stats
    assert stats["changed_share"] <= stats["shuffled_share"]
    assert all(r["meta"]["fact_order"]["p"] == 0.5 for r in b)


def test_v01_generation_is_deterministic_and_rebuilds_from_its_own_fields(cfg, paired_batches):
    _, b = paired_batches
    again = list(generate_batch(60, base_seed=21, cfg=cfg, git="fixed", phrasing="v01", fact_shuffle=0.5))
    assert [dumps(r) for r in again] == [dumps(r) for r in b[:60]]
    pairs = [r for r in b if r["meta"]["pair"]]
    for record in b[:25] + pairs[:6]:
        assert dumps(rebuild_record(cfg, record)) == dumps(record), record["id"]


def test_v01_requests_carry_no_hidden_names_and_keep_the_collision_and_window_wording(paired_batches):
    import re

    from test_simgen_leak import HIDDEN_NAMES

    _, b = paired_batches
    collision = {"en": re.compile(r"collision|bump|knock", re.I), "ko": re.compile(r"충돌|부딪")}
    window = {"en": re.compile(r"3 s(?:econds)?\b|3 seconds|next 3", re.I), "ko": re.compile(r"3초")}
    checked = Counter()
    for record in b:
        text = dumps(record["request"]).lower()
        assert not [n for n in HIDDEN_NAMES if n in text], record["id"]
        meta = record["meta"]
        lang = meta["language"]
        impact = meta["scene"] in ("p1p5_step_in_carrying", "p5_braking") or (meta["scene"] == "p3p5_carrying_mixed" and meta.get("maneuver") == "brake")
        for name, q in record["request"]["questions"].items():
            basis = meta["basis"][name]
            what = basis.get("event") or basis.get("name")
            criteria = q.get("criteria")
            full = q["instructions"] + " " + (" ".join(criteria) if isinstance(criteria, list) else "")
            if basis["kind"] in ("noul", "score") and what in ("spill", "rim_reached", "headroom"):
                assert bool(collision[lang].search(full)) == impact, (record["id"], name, q["instructions"])
                checked["spill"] += 1
            if basis["kind"] in ("noul", "score") and what in ("contact", "closeness"):
                assert window[lang].search(q["instructions"]), (record["id"], name, q["instructions"])
                checked["contact"] += 1
    assert checked["spill"] > 20 and checked["contact"] > 40, checked


# ----------------------------------------------------------------------------------------------------------------
# 만들어 둔 train-50k-v01
# ----------------------------------------------------------------------------------------------------------------


@pytest.mark.skipif(not V01_TRAIN.exists(), reason="v01 training set not generated yet (scripts/simgen_v0b.py train --phrasing v01)")
def test_the_v01_training_set_matches_its_manifest_and_shuffles_about_half():
    manifest = json.loads(V01_TRAIN.with_suffix(".manifest.json").read_text(encoding="utf-8"))
    digest = hashlib.sha256()
    shuffled = n = 0
    with V01_TRAIN.open("rb") as handle:
        for line in handle:
            digest.update(line)
            record = json.loads(line)
            assert record["meta"]["phrasing"]["version"] == "v01" and record["generator"]["phrasing"] == "v01"
            shuffled += record["meta"]["fact_order"]["shuffled"]
            n += 1
    assert digest.hexdigest() == manifest["sha256"] and n == manifest["records"] == 50_000
    assert manifest["phrasing"] == "v01" and manifest["fact_shuffle"] == 0.5
    assert abs(shuffled / n - 0.5) < 0.01, shuffled / n


# ----------------------------------------------------------------------------------------------------------------
# Task v0.2 §1-2 — wording balance v01b
# ----------------------------------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def v01b_batch(cfg):
    """The same 200 seeds as `paired_batches`, phrasing v01b with the same fact shuffle."""
    return list(generate_batch(200, base_seed=21, cfg=cfg, git="fixed", phrasing="v01b", fact_shuffle=0.5))


def test_v01b_keeps_the_moments_and_labels_of_v0b_and_the_state_text_of_v01(paired_batches, v01b_batch):
    a, b = paired_batches
    differs = Counter()
    for ra, rb, rc in zip(a, b, v01b_batch, strict=True):
        assert dumps(ra["labels"]) == dumps(rc["labels"]), rc["id"]
        assert ph.without_text(ra) == ph.without_text(rc), rc["id"]
        assert rc["request"]["state"] == rb["request"]["state"] and rc["meta"]["fact_order"] == rb["meta"]["fact_order"], rc["id"]
        assert not validate_record(rc), rc["id"]
        assert rc["meta"]["phrasing"]["version"] == "v01b" and rc["generator"]["phrasing"] == "v01b"
        qa, qc = ra["request"]["questions"], rc["request"]["questions"]
        differs["questions"] += len(qa)
        differs["from_v0b"] += sum(qa[n]["instructions"] != qc[n]["instructions"] for n in qa)
    assert 0.3 * differs["questions"] < differs["from_v0b"] < 0.65 * differs["questions"], differs


def test_v01b_keeps_the_original_wording_on_half_of_the_picks_and_it_is_the_one_v0b_drew(v01b_batch):
    summary = ph.wording_summary(ph.wording_stats(v01b_batch))
    assert summary["all"]["uses"] > 1500
    for lang in ("en", "ko", "all"):
        assert abs(summary[lang]["original_share"] - ph.V01B_ORIGINAL_SHARE) < 0.08, (lang, summary[lang])
    for record in v01b_batch:
        lang = record["meta"]["language"]
        for name, i_v0b, i_wide in record["meta"]["phrasing"]["picks"]:
            table, key = ph.pick_table(name)
            n_v0b = len(ph.base_tables(lang)[table][key]["train"])
            assert 0 <= i_wide < len(ph.wide(lang, table, key))
            if i_wide < n_v0b:
                assert i_wide == i_v0b, (record["id"], name)    # an original pick is exactly the wording v0b used


def test_v01b_differs_from_v01_only_in_the_wording_and_rebuilds_from_its_own_fields(cfg, paired_batches, v01b_batch):
    _, b = paired_batches
    assert any(dumps(rb["request"]["questions"]) != dumps(rc["request"]["questions"]) for rb, rc in zip(b, v01b_batch))
    again = list(generate_batch(40, base_seed=21, cfg=cfg, git="fixed", phrasing="v01b", fact_shuffle=0.5))
    assert [dumps(r) for r in again] == [dumps(r) for r in v01b_batch[:40]]
    pairs = [r for r in v01b_batch if r["meta"]["pair"]]
    for record in v01b_batch[:20] + pairs[:4]:
        assert dumps(rebuild_record(cfg, record)) == dumps(record), record["id"]


def test_the_v01b_pick_rule_itself():
    import numpy as np

    rng = np.random.default_rng(5)
    picks = [ph.pick_v01b(rng, 2, 3, 10) for _ in range(4000)]
    original = sum(p == 2 for p in picks) / len(picks)
    assert abs(original - 0.5) < 0.03 and set(picks) <= {2, *range(3, 10)} and set(range(3, 10)) <= set(picks)
    with pytest.raises(ValueError):
        ph.pick_v01b(rng, 3, 3, 10)      # the v0b index must be inside the v0b list
    with pytest.raises(ValueError):
        ph.pick_v01b(rng, 0, 3, 3)       # a key without new wordings cannot take v01b
    assert ph.PHRASINGS == ("v0b", "v01", "v01b") and ph.WIDE_PHRASINGS == ("v01", "v01b") and ph.DEFAULT_PHRASING == "v0b"


@pytest.mark.skipif(not V02_TRAIN.exists(), reason="v02 training set not generated yet (scripts/simgen_v0b.py train --phrasing v01b)")
def test_the_v02_training_set_matches_its_manifest_keeps_half_original_wordings_and_shuffles_about_half():
    manifest = json.loads(V02_TRAIN.with_suffix(".manifest.json").read_text(encoding="utf-8"))
    stats = json.loads(V02_TRAIN.with_suffix(".stats.json").read_text(encoding="utf-8"))
    digest = hashlib.sha256()
    shuffled = n = 0
    with V02_TRAIN.open("rb") as handle:
        for line in handle:
            digest.update(line)
            record = json.loads(line)
            assert record["meta"]["phrasing"]["version"] == "v01b" and record["generator"]["phrasing"] == "v01b"
            shuffled += record["meta"]["fact_order"]["shuffled"]
            n += 1
    assert digest.hexdigest() == manifest["sha256"] and n == manifest["records"] == 50_000
    assert manifest["phrasing"] == "v01b" and manifest["fact_shuffle"] == 0.5
    assert abs(shuffled / n - 0.5) < 0.01, shuffled / n
    for lang in ("en", "ko", "all"):
        assert abs(stats["wording_summary"][lang]["original_share"] - 0.5) < 0.01, stats["wording_summary"]
    for name in ("compare-v0b", "compare-v01"):
        path = V02_TRAIN.with_name(f"train-50k.{name}.json")
        if path.exists():
            compare = json.loads(path.read_text(encoding="utf-8"))
            assert compare["violations"] == 0 and compare["counts"]["records"] == 50_000 and compare["v01"]["sha256"] == manifest["sha256"]
            assert all(compare["stats_identical"].values()), compare["stats_identical"]
