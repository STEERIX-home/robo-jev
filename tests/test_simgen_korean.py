"""한국어 (Task v0b 지시서 §3·§8): 서술기 세 문체(+ 빼 둔 log)·질문·선택지 설명의 한국어판.

* 학습 표본의 약 15 %가 한국어이고, 한국어 레코드의 글(상태·지시문·선택지 설명·점수 기준)에는 채우지 못한 자리 표시가 없고 영어 낱말이
  없다(단위 기호와 규칙 질문의 답 열쇠 이름만 예외).
* 같은 순간을 영어와 한국어로 쓰면 관측·정답이 같고, 상태 글의 수(상황 옷 문구를 지운 나머지)가 같다 — 한국어 글이 같은 사실을 전한다.
* 문구의 뜻(통제자 요청의 반대 상관 점검을 한국어로도): 충돌이 있을 수 있는 순간의 쏟아짐·뚜껑·빈 높이 질문은 충돌을 말하고, 닿음·가까움은
  '3초'를 말하며 가까움은 지금 간격도 센다.
* 조사: 받침에 따라 이/가·은/는·을/를·과/와·으로/로(ㄹ은 로), 수·단위·영문 끝소리.
* 한국어 문구 묶음은 영어와 같은 열쇠·같은 묶음 크기를 갖는다(학습 묶음 셋 이상 + 빼 둔 묶음).
"""

from __future__ import annotations

import re
from collections import Counter

import pytest
from simgen_helpers import numeric_view, sample_records, skin_strings

from robo_jev.simgen import phrases
from robo_jev.simgen.catalog import default_config
from robo_jev.simgen.generate import assemble
from robo_jev.simgen.korean import josa, kfmt, ko_skins, skin_strings as card_strings
from robo_jev.simgen.narrate_ko import FILL_KO

_LATIN = re.compile(r"[A-Za-z][A-Za-z_/²]*")
#: 한국어 글에 남아도 되는 라틴 문자: 단위 기호, 규칙 질문의 답 열쇠(따옴표 안), 센서·장치 이름 약어
_ALLOWED = {"m", "cm", "mm", "s", "ms", "m/s", "km/h", "mm/s", "m/s²", "N", "ml", "kg", "g", "km", "h", "kW", "kWh"}
_RULE_KEYS = {"far", "near", "close", "stale", "fresh", "over_limit", "within_limit", "reduced", "normal", "single_file", "two_way", "sharp", "gentle",
              "secure", "as_is", "steep", "recheck", "accept"}  # fmt: skip
_COLLISION = re.compile(r"충돌|부딪")
_NUMBER = re.compile(r"\d+(?:\.\d+)?")


@pytest.fixture(scope="module")
def korean():
    return [r for r in sample_records() if r["meta"]["language"] == "ko"]


def _texts(record: dict) -> list[str]:
    out = [record["request"]["state"]]
    for q in record["request"]["questions"].values():
        out.append(q["instructions"])
        criteria = q.get("criteria")
        if isinstance(criteria, dict):
            out += list(criteria.values())
        elif isinstance(criteria, list):
            out += criteria
    return out


def test_about_fifteen_percent_of_training_records_are_korean(korean):
    share = len(korean) / len(sample_records())
    assert 0.11 <= share <= 0.19, share


def test_korean_records_have_no_unfilled_slot_and_no_english_words(korean):
    for record in korean:
        for text in _texts(record):
            assert "{" not in text and "}" not in text, (record["id"], text)
            words = {w for w in _LATIN.findall(text)}
            quoted = set(re.findall(r"'([a-z_]+)'", text))
            # 단위·답 열쇠, 그리고 이름 구실을 하는 대문자 부호·약어(랙 C, B4 선반, °C, AMR)는 영어 낱말이 아니다
            leftover = {w for w in words if w not in _ALLOWED and w not in quoted and w not in _RULE_KEYS and not re.fullmatch(r"[A-Z]{1,4}", w)}
            assert not leftover, (record["id"], leftover, text[:200])
        assert re.search(r"[가-힣]", record["request"]["state"])


def test_korean_spill_and_contact_questions_keep_the_corrected_meaning(korean):
    """충돌이 있을 수 있는 순간의 쏟아짐·뚜껑·빈 높이 질문은 충돌을 말하고(없으면 말하지 않고), 닿음·가까움은 3초 창을 말한다."""
    seen = Counter()
    for record in korean:
        meta = record["meta"]
        impact = meta["scene"] in ("p1p5_step_in_carrying", "p5_braking") or (meta["scene"] == "p3p5_carrying_mixed" and meta.get("maneuver") == "brake")
        for name, q in record["request"]["questions"].items():
            basis = meta["basis"][name]
            what = basis.get("event") or basis.get("name")
            text = q["instructions"] + " " + " ".join(q.get("criteria") or [] if isinstance(q.get("criteria"), list) else [])
            if basis["kind"] in ("noul", "score") and what in ("spill", "rim_reached", "headroom"):
                assert bool(_COLLISION.search(text)) == impact, (record["id"], name, q["instructions"])
                seen["spill"] += 1
            if basis["kind"] in ("noul", "score") and what in ("contact", "closeness"):
                assert "3초" in q["instructions"], (record["id"], name, q["instructions"])
                if what == "closeness":
                    assert "지금" in q["instructions"] and not re.search("지나가|지나치", q["instructions"])
                seen["contact"] += 1
    assert seen["spill"] > 15 and seen["contact"] > 30


CASES = [
    (101, "cafe", {"scene": "p1p5_step_in_carrying", "body": "mobile_base"}), (102, "warehouse", {"scene": "p1_crossing", "body": "forklift_amr"}),
    (103, "hospital", {"scene": "p1_head_on"}), (104, "lab", {"scene": "p5_braking", "body": "arm"}), (105, "home", {"scene": "p5_turning"}),
    (106, "cafe", {"scene": "p3_carrying_tall"}), (107, "warehouse", {"scene": "p3_place_stack", "body": "forklift_amr"}),
    (108, "factory", {"scene": "p3_ramp"}), (109, "lab", {"scene": "p4_place_check"}), (110, "factory", {"scene": "p4_insert"}),
    (111, "lab", {"scene": "p4p6_stale_check", "body": "arm"}), (112, "hospital", {"scene": "p6_stale_track"}), (113, "sidewalk", {"scene": "p6_blind_corner"}),
    (114, "warehouse", {"scene": "p6_sensor_conflict"}), (115, "retail", {"scene": "p1p6_emerging_step_in"}), (116, "cafe", {"scene": "p3p5_carrying_mixed"}),
    (117, "transit", {"scene": "p1_step_in", "body": "humanoid", "agent_kind": "adult"}),
]  # fmt: skip


@pytest.mark.parametrize(("seed", "domain", "forced"), CASES)
@pytest.mark.parametrize("style", ["facts", "prose", "planner", "log"])
def test_the_same_moment_in_english_and_korean_has_the_same_numbers_and_labels(seed, domain, forced, style):
    cfg = default_config()
    from robo_jev.simgen.generate import build_moment
    from robo_jev.simgen.holdout import split_for

    split = split_for("T2", styles=("facts", "prose", "planner", "log"))      # log 문체도 같은 사실을 전하는지(빼 둔 문체지만 같은 길)
    en = build_moment(cfg, seed, domain, {**forced, "style": style, "lang": "en"}, None, split)
    ko = build_moment(cfg, seed, domain, {**forced, "style": style, "lang": "ko"}, None, split)
    assert numeric_view(en.obs) == numeric_view(ko.obs)
    a, b = assemble(cfg, en, 0), assemble(cfg, ko, 0)
    assert a["labels"] == b["labels"]
    assert a["meta"]["basis"] == b["meta"]["basis"]

    def numbers(record, obs):
        text = record["request"]["state"]
        # 상황 옷 문구와 채움의 말 표현('about two-thirds full' ↔ '3분의 2쯤 차 있습니다' — 한국어는 분수를 숫자로 쓴다)은 뺀다
        for phrase in skin_strings(obs) + [t for pair in FILL_KO for t in pair]:
            text = text.replace(phrase, " ")
        return Counter(_NUMBER.findall(text))

    assert numbers(a, en.obs) == numbers(b, ko.obs), (a["request"]["state"], b["request"]["state"])


def test_josa_follows_the_final_sound():
    assert kfmt("{x:이}", x="피커") == "피커가" and kfmt("{x:이}", x="손님") == "손님이"
    assert kfmt("{x:을}", x="머그잔") == "머그잔을" and kfmt("{x:을}", x="카트") == "카트를"
    assert kfmt("{x:으로}", x="팔") == "팔로" and kfmt("{x:으로}", x="책") == "책으로" and kfmt("{x:으로}", x="카트") == "카트로"
    assert kfmt("{x:과}", x="라이다") == "라이다와" and kfmt("{x:은}", x="센서") == "센서는"
    assert josa("1.0 m", "이") == "가" and josa("3 N", "이") == "이" and josa("8.0 s", "이") == "가" and josa("2.5 s", "으로") == "로"
    assert josa("10", "이") == "이" and josa("2", "이") == "가" and josa("7", "으로") == "로"


def test_korean_phrase_banks_mirror_the_english_ones():
    from robo_jev.simgen import phrases_ko_data as data

    assert set(data.KO) == set(phrases.EN)
    for key, bank in phrases.EN.items():
        for pool in ("train", "heldout"):
            assert len(data.KO[key][pool]) == len(bank[pool]), (key, pool)
        assert set(data.KO[key]["train"]).isdisjoint(data.KO[key]["heldout"])
    assert set(data.ANCHORS_KO) == set(phrases.ANCHORS)
    assert set(data.SHORT_KO) == set(phrases._SHORT)
    for ko, en in ((data.SHORT_P4_KO, phrases._SHORT_P4), (data.SHORT_P6_KO, phrases._SHORT_P6), (data.SHORT_ARM_KO, phrases._SHORT_ARM),
                   (data.DESCRIPTIONS_P4_KO, phrases._DESCRIPTIONS_P4), (data.DESCRIPTIONS_P6_KO, phrases._DESCRIPTIONS_P6)):  # fmt: skip
        assert set(ko) == set(en)


def test_the_korean_skin_table_covers_every_english_skin_phrase():
    cfg = default_config()
    assert ko_skins().missing(cfg.skins) == {}
    assert sum(len(v) for v in card_strings(cfg.skins).values()) > 1000


@pytest.mark.parametrize("style", ["facts", "prose", "planner", "log"])
def test_sensor_conflict_sentences_name_the_reporting_sensor_and_hedge_object_cues_in_both_languages(style):
    """수정 1 M3·M2: 센서 불일치의 글이 '있다'고 보고한 센서를 바르게 말한다(리뷰의 MR2 — 한국어 문장의 센서를 바꾼 변이체가 수만 비교하는
    시험을 통과했다) — 영어와 한국어 각각 관측(라이다가 봤나)과 대조; 유리·검은 물체 단서는 '있을 수 있다'로(단정하지 않음), 먼지·눈부심·김은
    그 자리의 상태라 그대로."""
    from dataclasses import replace

    from robo_jev.simgen.generate import build_moment
    from robo_jev.simgen.holdout import split_for
    from robo_jev.simgen.narrate import HEDGED_CUES, RenderPlan, render_state
    from robo_jev.simgen.rollout import available_options
    from robo_jev.simgen.spec import stream

    cfg = default_config()
    split = split_for("T2", styles=("facts", "prose", "planner", "log"))
    seen = Counter()
    for i, domain in enumerate(("warehouse", "hospital", "retail", "factory")):
        for lang in ("en", "ko"):
            m = build_moment(cfg, 9_400 + i, domain, {"scene": "p6_sensor_conflict", "style": style, "lang": lang}, None, split)
            for lidar in (True, False):
                for cue in ("glass", "black", "dust", "glare", "steam"):
                    obs = replace(m.obs, conflict=replace(m.obs.conflict, lidar=lidar, camera=not lidar, cue=cue))
                    text = render_state(obs, RenderPlan(style, available_options(obs)), stream(1, "render"))
                    if lang == "en":
                        says_lidar = re.search(r"lidar(?: reports|: object)", text, re.I) is not None
                        says_camera = re.search(r"camera(?: reports|: object)", text, re.I) is not None
                        hedged = f"may be {obs.conflict.phrase}" in text or f"possibly in that direction: {obs.conflict.phrase}" in text
                    else:
                        says_lidar = re.search(r"라이다(?:는[^,]*물체가 있다고 보고|: 약 [^;]*물체)", text) is not None
                        says_camera = re.search(r"카메라(?:는[^,]*물체가 있다고 보고|: 약 [^;]*물체)", text) is not None
                        hedged = "있을 수 있" in text
                    assert (says_lidar, says_camera) == (lidar, not lidar), (lang, style, lidar, text)
                    assert hedged == (cue in HEDGED_CUES), (lang, style, cue, text)
                    seen[lang] += 1
    assert seen["en"] == seen["ko"] == 40
