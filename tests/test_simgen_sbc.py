"""simgen 보정 (리뷰 1 I1·I4, Task v0b §8 — 새 현상도): 정답(사후 확률)이 실제로 일어나는 비율과 맞는가.

* **순방향 SBC** — 정답의 사전(채움 = 속성 목록 범위, 의도·뚜껑·출렁임·반응 = 속성 목록 과정, 무게중심·치우침·마찰·무게 치우침과 그 단서·
  놓기 오차, 안착 상태와 참 측정값·보고, 떠남·출현·따라옴, 센서의 있음과 늦은 감지 거리 = 사전 — :func:`~robo_jev.simgen.scenes.prior_redraw`,
  관측되는 거리·폭·치수 = 감각 잡음보다 넓게 퍼진 분포 — 여유를 U(−3M, 3M)로)에서 세계를 뽑아 관측을 만들고, 그 관측의 사후 정답을 그
  세계에서 실제로 일어난 일과 대조한다. 사후가 감각 모형의 올바른 역이면 장면·사건마다 보정된다(추적 외삽을 뒤집거나 채움 사후를 바꾸면
  깨진다). 한 레코드에서 선택지 하나만 골라 레코드끼리 독립으로 센다. 무게 있는 사후(P4·P6)는 무게 평균으로 센다. 학습에 없는 P1+P3도
  넣는다(같은 사후).
* **레코드의 참 결과** — 참 세계를 정답의 사후에서 뽑으므로(I4) `truth.realized`는 정답에 대해 보정된다(옛 생성기는 회전에서
  z = +12.8이었다).
"""

from __future__ import annotations

import math
from collections import defaultdict

import numpy as np
import pytest
from simgen_helpers import sample_records

from robo_jev.simgen.catalog import DOMAINS, GROUPS, SCENES, default_config, scene_margins
from robo_jev.simgen.holdout import split_for
from robo_jev.simgen.labels import wmean
from robo_jev.simgen.posterior import exact_labels, sample_posterior
from robo_jev.simgen.rollout import available_options, rollout
from robo_jev.simgen.scenes import Infeasible, build_world, draw_context, observe, prior_redraw
from robo_jev.simgen.spec import fill_bins, stream

N_WORLDS = 5400
WIDE = 3.0
#: 모든 묶음(빼 둔 P1+P3 포함)·범위를 좁히지 않는 나눔(T3, 칸 없음) — 사후 자체를 잰다
SPLIT = split_for("T3", groups=tuple(GROUPS))
GROUPS_CYCLE = tuple(GROUPS)


def _prior_world(cfg, seed: int):
    """정답의 사전에 맞춘 세계 하나와 그 관측(넓은 균등 여유 → 닻, 숨은 값은 사전에서 다시)."""
    group = GROUPS_CYCLE[(seed // len(DOMAINS)) % len(GROUPS_CYCLE)]      # 분야 × 묶음을 층화(둘 다 seed로 돌면 늘 같은 짝이 된다)
    ctx = draw_context(cfg, stream(seed, "context"), DOMAINS[seed % len(DOMAINS)], {"group": group}, SPLIT)
    for attempt in range(64):
        rng = stream(seed, "wide", attempt)
        margins = {n: float(rng.uniform(-WIDE, WIDE) * cfg.catalog["margins"][n]["half_range"][ctx.body]) for n in scene_margins(ctx.scene, ctx.maneuver)}
        try:
            world = build_world(cfg, ctx, margins, stream(seed, "world", attempt))
        except Infeasible:
            continue
        world = prior_redraw(cfg, world, stream(seed, "prior"))
        return world, observe(cfg, world, ctx, stream(seed, "observe"), stream(seed, "format"), stream(seed, "skin", 0))
    return None, None


def _z(rows) -> tuple[float, float, float]:
    a = np.asarray(rows, dtype=float)
    var = float(np.sum(a[:, 0] * (1 - a[:, 0])))
    return a[:, 0].mean(), a[:, 1].mean(), (a[:, 1].sum() - a[:, 0].sum()) / math.sqrt(max(var, 1e-12))


@pytest.fixture(scope="module")
def forward_groups():
    cfg = default_config()
    groups: dict[tuple[str, str], list] = defaultdict(list)
    for seed in range(N_WORLDS):
        world, obs = _prior_world(cfg, seed)
        if world is None:
            continue
        options = available_options(obs)
        out = rollout(cfg, obs, sample_posterior(cfg, obs, 256, stream(seed, "posterior")), options)
        true = rollout(cfg, obs, world.as_batch(), options, rng=stream(seed, "truth-outcome"))
        exact = exact_labels(cfg, obs)
        pick = stream(seed, "pick")
        for event, values in out.events.items():
            if "__" in event:          # 상대 종류별로 나눈 닿음(비용 항목용) — 'contact'에 이미 있다
                continue
            i = int(pick.integers(len(options)))
            groups[(obs.scene, event)].append((float(wmean(values[i], out)), bool(true.events[event][i][0] > 0.5)))
        for event, values in out.world_events.items():
            groups[(obs.scene, f"world:{event}")].append((float(exact.get(event, wmean(values, out))), bool(true.world_events[event][0] > 0.5)))
        if "intent" in exact and world.values.get("intent", -1) >= 0:
            j = int(pick.integers(len(exact["intent"])))
            groups[(obs.scene, "state:intent")].append((float(exact["intent"][j]), int(world.values["intent"]) == j))
        if "fill_bin" in exact:
            j = int(pick.integers(len(exact["fill_bin"])))
            groups[(obs.scene, "state:fill_bin")].append((float(exact["fill_bin"][j]), int(fill_bins(np.array([world.values["fill"]]))[0]) == j))
        for state, values in (out.states or {}).items():
            if state in ("intent", "fill_bin"):
                continue
            dist = None
            if state in exact:
                dist = np.atleast_1d(np.asarray(exact[state], dtype=float))
                dist = np.array([1.0 - dist[0], dist[0]]) if len(dist) == 1 else dist      # 확률 하나(예: 뚜껑) = P(값 1)
            n = len(dist) if dist is not None else int(values.max()) + 1
            j = int(pick.integers(n))
            p = float(dist[j]) if dist is not None else float(wmean(values == j, out))
            groups[(obs.scene, f"state:{state}")].append((p, int(round(float(true.states[state][0]))) == j))
    return groups


def test_forward_sbc_covers_every_scene_and_the_main_events(forward_groups):
    scenes = {scene for scene, _ in forward_groups}
    assert scenes == set(SCENES)
    for needed in (("p1_head_on", "contact"), ("p1_crossing", "contact"), ("p1p5_step_in_carrying", "spill"), ("p5_braking", "overrun"),
                   ("p5_turning", "spill"), ("p1_step_in", "world:stops_short"), ("p1_head_on", "world:passable"),
                   ("p3_carrying_tall", "fall"), ("p3_carrying_tall", "tip"), ("p3_ramp", "fall"), ("p3_place_stack", "topple"),
                   ("p3p5_carrying_mixed", "spill"), ("p3p5_carrying_mixed", "fall"), ("p1p3_step_in_carrying_tall", "fall"),
                   ("p4_place_check", "world:ok_now"), ("p4_insert", "world:ok_now"), ("p4p6_stale_check", "world:ok_now"), ("p4_place_check", "ok_after"),
                   ("p6_stale_track", "world:present"), ("p6_stale_track", "contact"), ("p6_blind_corner", "world:stops_before_corner"),
                   ("p6_blind_corner", "contact"), ("p6_sensor_conflict", "world:present"), ("p1p6_emerging_step_in", "world:follower"),
                   ("p1p6_emerging_step_in", "contact")):  # fmt: skip
        assert len(forward_groups[needed]) >= 100, (needed, len(forward_groups[needed]))


def test_labels_are_calibrated_against_worlds_drawn_from_their_prior(forward_groups):
    """장면 × 사건마다 평균 정답 대 실제 비율: |z| < 4 (분산이 아주 작은 묶음은 뺀다; 묶음이 작아 절대 차이는 문턱으로 쓰지 않는다)."""
    checked = 0
    for key, rows in sorted(forward_groups.items()):
        a = np.asarray(rows, dtype=float)
        if np.sum(a[:, 0] * (1 - a[:, 0])) < 3.0:
            continue
        label, realized, z = _z(rows)
        assert abs(z) < 4.0, (key, len(rows), round(label, 3), round(realized, 3), round(z, 2))
        checked += 1
    assert checked >= 60


def test_forward_sbc_is_calibrated_within_probability_bins(forward_groups):
    rows = np.array([r for key, group in forward_groups.items() if not key[1].startswith("state:") for r in group], dtype=float)
    for lo, hi in ((0.0, 0.1), (0.1, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 0.9), (0.9, 1.0001)):
        mask = (rows[:, 0] >= lo) & (rows[:, 0] < hi)
        label, realized, z = _z(rows[mask])
        assert abs(z) < 4.5 and abs(realized - label) < 0.08, (lo, hi, int(mask.sum()), round(label, 3), round(realized, 3), round(z, 2))


def test_record_truth_is_calibrated_against_the_labels_because_it_is_a_posterior_draw():
    groups: dict[tuple[str, str], list] = defaultdict(list)
    for r in sample_records():
        for name, label in r["labels"].items():
            if label["type"] == "noul":
                basis = r["meta"]["basis"][name]
                groups[(r["meta"]["scene"], basis["event"])].append((label["p_yes"], bool(r["truth"]["realized"][name])))
    checked = 0
    for key, rows in sorted(groups.items()):
        a = np.asarray(rows, dtype=float)
        if np.sum(a[:, 0] * (1 - a[:, 0])) < 5.0:
            continue
        label, realized, z = _z(rows)
        assert abs(z) < 4.0, (key, len(rows), round(label, 3), round(realized, 3), round(z, 2))
        checked += 1
    assert checked >= 10
