"""v0a에서 넘어온 것 (Task v0b 지시서 §6): 시험이 없던 수정 경로 셋(m-a), 옆 장애물 닿음의 쏟아짐(m-b), 같은 질문 중복 제거(m-c),
끼어듦의 글 전제를 쓰는 사후(리뷰 1 재검증의 끼어듦 잔여).

* m-a ① 마주 옴의 관측 쪽(C1): 글에서 읽은 수로 계산한 지금 간격 − 닻 세계의 참 지금 간격의 평균 ≈ 0 (관측 쪽을 되돌리면 −v·나이만큼 치우친다).
* m-a ② 닫힌 꼴 정답: 닫힌 꼴이 있는 질문은 맞출 수 있으면(`meta.calibration.matched`) 모두 `meta.exact_labels`에 있고 정답이
  `exact_labels(관측)`와 같다 — 수정 1 M5부터는 표본의 무게를 닫힌 꼴에 맞춘 같은 표본의 빈도로(닫힌 꼴 맞추기를 끄면 떨어진다).
* m-a ③ 끼어듦의 마지막 위치 상한: 닻 세계의 가장 최근 위치 보고가 진로 중심선에서 `last_fix_max` 안(상한을 지우면 떨어진다).
* m-b 비켜서다 옆의 숨은 장애물에 닿으면 열린 용기가 쏟아진다(뚜껑이 있으면 아니다).
* m-c 한 요청 안의 질문은 묻는 것의 정체(`questions.identity`)가 모두 다르다.
* 끼어듦 잔여: 사후의 지금 옆 위치가 글의 전제 띠 안에 있고, 오래된 추적(≥ 0.8 s)에서도 닿음 정답이 실제 비율과 맞는다(v0a 코드는
  +0.05, z ≈ 4.8 — artifacts/scratch/v0b/a-stepin-sbc-*.log).
"""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pytest
from simgen_helpers import moment, sample_records

from robo_jev.simgen.catalog import DOMAINS, SCENE_MARGINS, default_config
from robo_jev.simgen.generate import build_moment
from robo_jev.simgen.holdout import split_for
from robo_jev.simgen.labels import uses_exact
from robo_jev.simgen.posterior import exact_labels, sample_posterior
from robo_jev.simgen.questions import identity, write_questions
from robo_jev.simgen.rollout import available_options, rollout
from robo_jev.simgen.scenes import Infeasible, build_world, draw_context, observe
from robo_jev.simgen.spec import Batch, Reading, stream


@pytest.fixture(scope="module")
def cfg():
    return default_config()


def _moment_of(cfg, record):
    g = record["generator"]
    forced = {k: v for k, v in g["forced"].items() if k != "domain"}
    return build_moment(cfg, g["seed"], g["forced"]["domain"], forced or None, g["variant"])


# ----------------------------------------------------------------------------------------------------------------
# m-a ① 마주 옴의 관측 쪽
# ----------------------------------------------------------------------------------------------------------------


def test_head_on_text_gap_matches_the_anchor_worlds_current_gap(cfg):
    """글에서 읽은 수(간격·나이·상대 속도·로봇 속도, 또는 위치 두 개)로 계산한 지금 간격과 닻 세계의 참 지금 간격(면 사이)의 차이는 감각
    잡음뿐이다 — 평균 ≈ 0. 관측 쪽(scenes.observe)이 '그 시각의 간격'을 로봇 이동 없이 만들면 −(로봇 속도 × 나이)만큼 치우친다."""
    diffs, biases = [], []
    for i in range(240):
        m = build_moment(cfg, 70_000 + i, DOMAINS[i % len(DOMAINS)], {"scene": "p1_head_on"})
        a, r, v = m.obs.agent, m.obs.robot, m.world.values
        if any(f.censored for f in a.track) or (a.speed is not None and a.speed.censored):
            continue
        if a.speed is not None:
            literal = a.track[-1].value - (a.speed.value + r.speed) * a.age
        else:
            closing = (a.track[0].value - a.track[1].value) / (a.track_ages[0] - a.track_ages[1])
            literal = a.track[-1].value - closing * a.age
        diffs.append(literal - (v["dist"] - r.radius - v["ra"]))
        biases.append(r.speed * a.age)
    diffs = np.array(diffs)
    assert len(diffs) > 200
    se = diffs.std() / math.sqrt(len(diffs))
    assert abs(diffs.mean()) < 4 * se + 0.01, (diffs.mean(), se)
    assert np.mean(biases) > 3 * (4 * se + 0.01)  # 되돌린 관측 쪽의 치우침(평균 −v·나이 ≈ −0.4 m)은 이 시험이 분명히 잡는 크기다


# ----------------------------------------------------------------------------------------------------------------
# m-a ② 닫힌 꼴 정답
# ----------------------------------------------------------------------------------------------------------------


def test_every_question_with_a_closed_form_uses_it(cfg):
    checked, unmatched, trimmed = 0, 0, 0
    for record in sample_records()[:300]:
        obs = _moment_of(cfg, record).obs
        exact = exact_labels(cfg, obs)
        meta = record["meta"]
        assert set(meta["calibration"]["matched"]) | set(meta["calibration"]["unmatched"]) <= set(exact), record["id"]
        for name, label in record["labels"].items():
            recipe = {**meta["basis"][name], **({"options": meta["options"][name]} if name in meta["options"] else {})}
            if recipe["kind"] not in ("noul", "state") or not uses_exact(recipe, exact):
                continue
            if not uses_exact(recipe, meta["calibration"]["matched"]):
                unmatched += 1      # 끝내 맞추지 못한 닫힌 꼴(없어야 한다) — 정답은 같은 표본의 빈도
                continue
            assert name in meta["exact_labels"], (record["id"], name)
            key = recipe["state"] if recipe["kind"] == "state" else (recipe["event"] if recipe["source"] == "world" else f"{recipe['event']}|{recipe['option']}")
            # 표본이 없는 꼬리 범주를 잘라 냈으면 그 질량만큼까지 다를 수 있다(적혀 있다); 아니면 같다
            slack = meta["calibration"]["trimmed"].get(key, 0.0) + 1e-6
            if recipe["kind"] == "noul":
                p = float(exact[key])
                assert abs(label["p_yes"] - (1.0 - p if recipe.get("negate") else p)) <= slack, (record["id"], name)
            else:
                value = exact[recipe["state"]]
                dist = np.array([1.0 - value, value]) if recipe["state"] == "lid" else np.asarray(value)
                for option, accepted in recipe["options"].items():
                    assert abs(label["probabilities"][option] - float(sum(dist[int(x)] for x in accepted))) <= slack, (record["id"], name, option)
            checked += 1
            trimmed += key in meta["calibration"]["trimmed"]
    assert checked >= 150 and unmatched == 0 and trimmed <= 0.25 * checked, (checked, unmatched, trimmed)


# ----------------------------------------------------------------------------------------------------------------
# m-a ③ 끼어듦의 마지막 위치 상한
# ----------------------------------------------------------------------------------------------------------------


def test_step_in_newest_fix_stays_within_the_cap(cfg):
    """가장 최근 위치 보고 = 지금 옆 위치에서 나이만큼 되돌린 자리: |lat0| + u0·나이 ≤ last_fix_max. 상한이 실제로 걸린다(가까운 것이 있다)."""
    near_cap = 0
    for i in range(400):
        m = build_moment(cfg, 71_000 + i, DOMAINS[i % len(DOMAINS)], {"scene": ("p1_step_in", "p1p5_step_in_carrying")[i % 2]})
        v, age = m.world.values, m.world.at_observation["age"]
        cap = cfg.body(m.ctx.body)["last_fix_max"]
        newest = abs(v["lat0"]) + v["u0"] * age
        assert newest <= cap + 1e-9, (m.seed, newest, cap)
        near_cap += newest > cap - 0.1 * cap
    assert near_cap >= 10


# ----------------------------------------------------------------------------------------------------------------
# m-b 옆 장애물 닿음 → 쏟아짐
# ----------------------------------------------------------------------------------------------------------------


def test_hitting_a_hidden_side_obstacle_spills_an_open_container(cfg):
    obs = moment(41, "retail", scene="p1p5_step_in_carrying", body="humanoid", agent_kind="adult").obs
    container = replace(obs.container, fill=Reading(0.5, 0.1, 0.05), fill_mode="percent", lid="unknown")
    obs = replace(obs, container=container)
    row = {"gap": 6.0, "lat0": -0.5, "dir": 1.0, "u0": 0.0, "ra": 0.25, "intent": 0.0, "stop_decel": 2.0, "stay_s": 6.0, "cross_speed": 1.0,
           "cross_accel": 1.0, "back_decel": 2.0, "back_speed": 0.5, "roll_decel": 0.0, "side_left": 1.0, "side_right": 1.0,
           "fill": 0.30, "k": 1.0, "tilt": 0.0, "lid": 0.0}  # fmt: skip
    batch = Batch({key: np.array([row[key], row[key]], dtype=float) for key in row} | {"lid": np.array([0.0, 1.0])})
    out = rollout(cfg, obs, batch, ("sidestep_left", "continue"))
    i = out.index("sidestep_left")
    assert out.events["side_contact"][i].tolist() == [True, True]
    assert out.events["spill"][i].tolist() == [True, False]          # 열린 용기만 쏟아진다
    assert out.events["spill"][out.index("continue")].tolist() == [False, False]  # 낮은 채움, 충격 없음 → 쏟아지지 않는다


# ----------------------------------------------------------------------------------------------------------------
# m-c 중복 질문
# ----------------------------------------------------------------------------------------------------------------


def _recipe(meta: dict, name: str) -> dict:
    return {**meta["basis"][name], **({"options": meta["options"][name]} if name in meta["options"] else {})}


def test_no_request_asks_the_same_thing_twice():
    for record in sample_records():
        meta = record["meta"]
        idents = [identity(_recipe(meta, name)) for name in record["request"]["questions"]]
        assert len(idents) == len(set(idents)), (record["id"], idents)


def test_duplicates_are_removed_but_every_request_keeps_its_required_kinds(cfg):
    """후보 묶음은 이미 물은 것을 빼고 고른다 — 감속(P5)의 핵심 noul(단단한 제동의 쏟아짐)을 다시 묻는 noul·score는 나오지 않는다."""
    obs = moment(14, "lab", scene="p5_braking", body="mobile_base").obs
    for i in range(200):
        qset = write_questions(cfg, obs, available_options(obs), stream(i, "questions"))
        idents = [identity(q.recipe) for q in qset.questions]
        assert len(idents) == len(set(idents))
        kinds = {q.recipe["kind"] for q in qset.questions}
        assert {"action", "noul", "score"} <= kinds


# ----------------------------------------------------------------------------------------------------------------
# 끼어듦 잔여 — 글의 전제를 쓰는 사후
# ----------------------------------------------------------------------------------------------------------------


def test_step_in_posterior_keeps_the_agent_inside_the_premise_band(cfg):
    """1.0 s 전 −0.20 m(오른쪽)에서 왼쪽으로 0.6 m/s — 자르지 않은 외삽은 이미 중심선을 넘은 +0.40 m다. 글은 '들어오고 있다'고 하므로
    사후는 들어오는 쪽 띠 [0, 0.6 × (로봇 + 상대 반지름)] 안에만 둔다."""
    base = moment(41, "retail", scene="p1_step_in", body="mobile_base", agent_kind="adult").obs
    agent = replace(base.agent, track=(Reading(-0.20, 0.03, 0.01),), track_ages=(1.0,), speed=Reading(0.60, 0.10, 0.05), heading="left")
    obs = replace(base, agent=agent)
    batch = sample_posterior(cfg, obs, 2000, np.random.default_rng(3))
    band = cfg.catalog["intent_motion"]["step_in_band"] * (obs.robot.radius + batch["ra"])
    entering = -batch["lat0"]  # 왼쪽으로 가니 들어오는 쪽은 오른쪽(−)
    assert np.all(entering >= 0.0) and np.all(entering <= band + 1e-12)
    assert entering.mean() < 0.1  # 띠의 중심선 쪽 끝에 몰린다(외삽이 넘어가 있으니)


def test_old_step_in_tracks_are_calibrated_in_forward_sbc(cfg):
    """속성 목록 사전(넓은 여유)에서 뽑은 끼어듦 세계 → 관측 → 사후 닿음 정답(선택지 하나) 대 그 세계의 닿음, 추적 나이 ≥ 0.8 s.
    v0a 코드(전제 없음)는 정답 0.432 대 실제 0.483(z +4.8, 같은 4,000 세계); 전제 띠로 자르면 z ≈ +0.5."""
    rows = []
    for seed in range(4000):
        # 나이 > 1.0 s는 T3 칸(학습 쪽은 잘라 낸다) — 이 검사는 사후 자체를 재므로 범위를 좁히지 않는 나눔(T3, 칸 없음)으로 뽑는다
        ctx = draw_context(cfg, stream(seed, "context"), DOMAINS[seed % 10], {"scene": "p1_step_in"}, split_for("T3"))
        world = None
        for attempt in range(64):
            rng = stream(seed, "wide", attempt)
            margins = {n: float(rng.uniform(-3, 3) * cfg.catalog["margins"][n]["half_range"][ctx.body]) for n in SCENE_MARGINS[ctx.scene]}
            try:
                world = build_world(cfg, ctx, margins, stream(seed, "world", attempt))
                break
            except Infeasible:
                continue
        if world is None or ctx.agent_kind == "ball" or world.at_observation["age"] < 0.8:
            continue
        obs = observe(cfg, world, ctx, stream(seed, "observe"), stream(seed, "format"), stream(seed, "skin", 0))
        options = available_options(obs)
        out = rollout(cfg, obs, sample_posterior(cfg, obs, 256, stream(seed, "posterior")), options)
        true = rollout(cfg, obs, world.as_batch(), options)
        i = int(stream(seed, "pick").integers(len(options)))
        rows.append((float(out.events["contact"][i].mean()), bool(true.events["contact"][i][0])))
    a = np.array(rows, dtype=float)
    z = (a[:, 1].sum() - a[:, 0].sum()) / math.sqrt(np.sum(a[:, 0] * (1 - a[:, 0])))
    assert len(a) > 700
    assert abs(z) < 3.0, (len(a), a[:, 0].mean(), a[:, 1].mean(), z)
