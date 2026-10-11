"""질문 문구가 정답과 같은 것을 묻는가 (Task v0b — v0c 진단의 반대 상관 점검; 통제자 요청).

v0c 미리보기(v0a.1 표본)에서 학습 없는 Qwen이 정답과 반대로 간 칸을 읽어 보니 엔진 쪽 어긋남이 있었다:

* **쏟아짐 정답은 충돌(넘어섬·사람·옆 장애물)의 쏟아짐을 세는데 문구는 그 원인을 말하지 않았다.** 그래서 감속(P5)에서 "부드럽게 서면
  쏟아지나?"의 정답(넘어서 부딪혀 쏟아짐 0.78)이 "최대로 서면"(0.59~0.81)보다 높았다 — 문구만 읽으면 반대가 맞아 보인다.
* **빈 높이 사용(headroom) score와 '뚜껑까지 차오름'은 충돌을 빼고 셌는데** 기준 문구는 '1 = 림에 닿거나 쏟아짐'이었다: 한 요청 안에서
  "그대로 가면: 쏟아짐 1.0, 빈 높이 사용 0.03".
* 닿음·가까움 score는 앞으로 3초를 세는데 문구에 시간 창이 없었고, 가까움은 멈춤 선택지에도 '지나간다(pass)'라고 물었다.

고친 것: headroom = 충돌이면 1, rim_reached = 출렁임 또는 충돌(rollout); 충돌이 있을 수 있는 순간의 쏟아짐·뚜껑·빈 높이 문구는 충돌을
말한다(questions.impact_possible); 닿음·가까움 문구는 3초를 말한다. 이 시험들이 그 뜻을 고정한다(영어; 한국어는 test_simgen_korean).
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import replace

import numpy as np
import pytest
from simgen_helpers import moment, sample_records

from robo_jev.simgen.catalog import default_config
from robo_jev.simgen.rollout import rollout
from robo_jev.simgen.spec import Batch, Reading

_COLLISION = re.compile(r"collision|bump|knock", re.I)
_WINDOW = re.compile(r"3 s(?:econds)?\b|3 seconds|next 3", re.I)


@pytest.fixture(scope="module")
def cfg():
    return default_config()


def _braking_case(cfg):
    obs = moment(14, "lab", scene="p5_braking", body="mobile_base").obs
    robot = replace(obs.robot, speed=1.0, cruise=1.0, latency=0.2, decel_gentle=1.0, decel_moderate=1.75, decel_hard=3.0, radius=0.3)
    obs = replace(obs, robot=robot, container=replace(obs.container, diameter=0.08, height=0.10, lid="unknown"))
    # 장애물 면까지 0.55 m: gentle은 넘어선다(멈춤 거리 0.70 m), moderate·hard는 선다; 채움 0.5 — 출렁임만으로는 어느 것도 넘치지 않는다
    rows = [{"obstacle": 0.85, "fill": 0.5, "k": 1.0, "tilt": 0.0, "lid": lid} for lid in (0.0, 1.0)]
    return obs, Batch({k: np.array([r[k] for r in rows], dtype=float) for k in rows[0]})


def test_a_collision_reaches_the_rim_and_uses_all_the_headroom(cfg):
    obs, batch = _braking_case(cfg)
    out = rollout(cfg, obs, batch)
    gentle, hard = out.index("stop_gentle"), out.index("stop_hard")
    assert out.events["overrun"][gentle].tolist() == [True, True]
    assert out.events["rim_reached"][gentle].tolist() == [True, True]      # 부딪히면 액면이 림(뚜껑)까지 친다
    assert out.events["spill"][gentle].tolist() == [True, False]           # 뚜껑이 있으면 쏟아지지 않는다
    assert out.quantities["headroom"][gentle].tolist() == [1.0, 1.0]
    assert out.events["overrun"][hard].tolist() == [False, False]
    assert out.events["rim_reached"][hard].tolist() == [False, False]      # 출렁임만: 3.0 m/s² × 0.04 m / 9.81 ≈ 1.2 cm < 5 cm
    assert 0.0 < out.quantities["headroom"][hard][0] < 0.5


def test_spill_implies_rim_reached_and_full_headroom_in_every_world(cfg):
    """사건의 포함 관계: 쏟아짐 ⊂ 림까지 차오름, 쏟아진 세계의 빈 높이 사용 = 1 — 그래서 정답도 P(쏟아짐) ≤ P(림) ≤ … 이고
    E[빈 높이 사용] ≥ P(쏟아짐)이다(기준 문구 '1 = 림에 닿거나 쏟아짐'과 같은 뜻)."""
    from robo_jev.simgen.posterior import sample_posterior
    from robo_jev.simgen.spec import stream

    for seed, domain, forced in ((22, "cafe", {"scene": "p5_braking", "body": "mobile_base"}), (31, "cafe", {"scene": "p1p5_step_in_carrying", "body": "mobile_base"}),
                                 (15, "home", {"scene": "p5_turning"})):  # fmt: skip
        obs = moment(seed, domain, **forced).obs
        out = rollout(cfg, obs, sample_posterior(cfg, obs, 512, stream(seed, "posterior")))
        spill, rim, head = out.events["spill"], out.events["rim_reached"], out.quantities["headroom"]
        assert np.all(~spill | rim)
        assert np.all(head[spill] == 1.0)


def _records_with(kind_filter):
    for record in sample_records():
        meta = record["meta"]
        if meta["language"] != "en":      # 한국어 문구의 뜻은 test_simgen_korean
            continue
        for name, q in record["request"]["questions"].items():
            basis = meta["basis"][name]
            if kind_filter(basis):
                yield record, name, q, basis


def test_questions_about_spills_name_the_collision_when_one_can_happen():
    """충돌이 있을 수 있는 순간의 쏟아짐·뚜껑·빈 높이 질문은 충돌을 말하고(정답이 그것을 센다), 그렇지 않은 순간은 말하지 않는다."""
    seen = {True: 0, False: 0}
    for record, name, q, basis in _records_with(lambda b: (b.get("event") or b.get("name")) in ("spill", "rim_reached", "headroom")):
        meta = record["meta"]
        impact = meta["scene"] in ("p1p5_step_in_carrying", "p5_braking") or (meta["scene"] == "p3p5_carrying_mixed" and meta.get("maneuver") == "brake")
        mentions = bool(_COLLISION.search(q["instructions"] + " ".join(q.get("criteria") or [])))
        assert mentions == impact, (record["id"], name, q["instructions"])
        seen[impact] += 1
    assert seen[True] > 100 and seen[False] > 50


def test_contact_and_closeness_questions_state_the_three_second_window():
    checked = 0
    for record, name, q, basis in _records_with(lambda b: b["kind"] in ("noul", "score") and (b.get("event") or b.get("name")) in ("contact", "closeness")):
        assert _WINDOW.search(q["instructions"]), (record["id"], name, q["instructions"])
        if basis.get("name") == "closeness":
            assert not re.search(r"\bpass\b", q["instructions"]), q["instructions"]  # 멈춤 선택지에도 쓰이므로 '지나간다'고 하지 않는다
            assert "now" in q["instructions"].lower()  # 지금 간격도 센다
        checked += 1
    assert checked > 300


def test_gentle_braking_spill_labels_are_impact_spills_and_say_so():
    """v0c가 찾은 칸 그대로: 감속(P5)에서 부드러운 멈춤의 쏟아짐 정답이 단단한·최대 멈춤보다 높은 것은 넘어선 충돌 때문이다 — 그런 질문은
    언제나 충돌을 말한다."""
    rows = []
    for record, name, q, basis in _records_with(lambda b: b.get("event") == "spill" or b.get("name") == "spill"):
        if record["meta"]["scene"] != "p5_braking":
            continue
        label = record["labels"][name]
        value = label.get("p_yes", label.get("value"))
        rows.append((basis["option"], value))
        assert _COLLISION.search(q["instructions"]), q["instructions"]
    means = {o: np.mean([v for opt, v in rows if opt == o]) for o in ("stop_gentle", "stop_moderate", "stop_hard") if any(opt == o for opt, _ in rows)}
    assert len(rows) > 30 and len(means) >= 2


# ----------------------------------------------------------------------------------------------------------------
# 실은 물건의 넘어짐·떨어짐 (수정 1 I2 — 리뷰의 v0b-000106: '넘어지거나 떨어지나'의 정답이 넘어선 충돌을 셌는데 글은 말하지 않았다)
# ----------------------------------------------------------------------------------------------------------------

_COLLISION_KO = re.compile(r"충돌|부딪")
_LOAD_WHAT = ("fall", "fall_motion", "tip", "slide", "slide_frac")
_LOAD_CASES = (("p3_carrying_tall", "brake"), ("p3p5_carrying_mixed", "brake"), ("p3_carrying_tall", "turn"), ("p3_carrying_tall", "start"),
               ("p3_ramp", None), ("p3p5_carrying_mixed", "turn"))  # fmt: skip


@pytest.fixture(scope="module")
def load_records(cfg):
    """P3 갈래마다 영어·한국어를 고르게(강제 틀 48개 — 표본에는 감속 장면이 적어서)."""
    from robo_jev.simgen.catalog import DOMAINS
    from robo_jev.simgen.generate import generate_record

    out = []
    for i in range(48):
        scene, maneuver = _LOAD_CASES[i % len(_LOAD_CASES)]
        forced = {"scene": scene, "lang": "en" if (i // len(_LOAD_CASES)) % 2 == 0 else "ko"}
        if maneuver:
            forced["maneuver"] = maneuver
        out.append(generate_record(8_800_000 + i, i, DOMAINS[i % len(DOMAINS)], cfg=cfg, forced=forced))
    return out


def _criteria_text(q: dict) -> str:
    criteria = q.get("criteria")
    if isinstance(criteria, dict):
        return " ".join(criteria.values())
    return " ".join(criteria or [])


def test_load_questions_say_whether_a_collision_counts(load_records):
    """충돌로 떨어질 수 있는 순간(P3·P3+P5 감속): 충돌까지 세는 떨어짐(fall)의 글은 충돌을 밝히고, 움직임만 세는 질문(fall_motion·tip·slide·
    미끄러진 거리)의 글은 '충돌은 빼고 움직임만'을 붙인다; 충돌이 없는 순간(회전·출발·경사로)의 글은 충돌을 말하지 않는다 — 영어와 한국어."""
    from robo_jev.simgen import phrases, phrases_ko

    seen = Counter()
    for record in load_records:
        meta = record["meta"]
        ko = meta["language"] == "ko"
        impact = meta["scene"] in ("p3_carrying_tall", "p3p5_carrying_mixed") and meta.get("maneuver") == "brake"
        motion_only = (phrases_ko.MOTION_ONLY if ko else phrases.MOTION_ONLY).strip()
        collision = _COLLISION_KO if ko else _COLLISION
        for name, q in record["request"]["questions"].items():
            basis = meta["basis"][name]
            what = basis.get("event") or basis.get("name")
            if basis["kind"] not in ("noul", "score") or what not in _LOAD_WHAT:
                continue
            text = q["instructions"] + " " + _criteria_text(q)
            if impact and what == "fall":
                assert motion_only not in text and collision.search(text), (record["id"], name, text)
                seen["counts the collision"] += 1
            elif impact:
                assert motion_only in q["instructions"], (record["id"], name, text)
                seen["motion only"] += 1
            else:
                assert motion_only not in text and not collision.search(text), (record["id"], name, text)
                seen["no collision possible"] += 1
            seen["ko" if ko else "en"] += 1
    assert seen["counts the collision"] >= 5 and seen["motion only"] >= 15 and seen["no collision possible"] >= 15, seen
    assert seen["ko"] >= 10 and seen["en"] >= 10, seen


def test_motion_only_load_labels_hold_together_within_a_request(load_records):
    """한 요청 안에서(같은 선택지): P(넘어짐) ≤ P(움직임으로 떨어짐) ≤ P(넘어짐) + P(미끄러짐) — 움직임만 세는 정답에 충돌이 들어가면 마지막
    부등식이 깨진다(리뷰의 v0b-000106: 넘어짐 0·미끄러짐 0인데 '넘어지거나 떨어지나' 1.0); 그리고 P(움직임으로 떨어짐) ≤ P(충돌까지 센 떨어짐)."""
    checked = 0
    for record in load_records:
        meta = record["meta"]
        by_option: dict[str, dict[str, float]] = {}
        for name, basis in meta["basis"].items():
            if basis["kind"] == "noul" and basis.get("event") in ("tip", "slide", "fall_motion", "fall"):
                by_option.setdefault(basis["option"], {})[basis["event"]] = record["labels"][name]["p_yes"]
        for option, p in by_option.items():
            if "fall_motion" in p:
                assert p["fall_motion"] >= p.get("tip", 0.0) - 1e-6, (record["id"], option, p)
                if "tip" in p and "slide" in p:
                    assert p["fall_motion"] <= p["tip"] + p["slide"] + 1e-6, (record["id"], option, p)
                    checked += 1
                if "fall" in p:
                    assert p["fall_motion"] <= p["fall"] + 1e-6, (record["id"], option, p)
                    checked += 1
    assert checked >= 1
