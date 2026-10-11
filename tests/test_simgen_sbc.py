"""simgen 보정 (리뷰 1 I1·I4): 정답(사후 확률)이 실제로 일어나는 비율과 맞는가.

* **순방향 SBC** — 정답의 사전(채움 = 속성 목록 범위, 의도·뚜껑·출렁임·반응 = 속성 목록 과정, 관측되는 거리·폭 = 감각 잡음보다 넓게
  퍼진 분포 — 여유를 U(−3M, 3M)로)에서 세계를 뽑아 관측을 만들고, 그 관측의 사후 정답을 그 세계에서 실제로 일어난 일과 대조한다.
  사후가 감각 모형의 올바른 역이면 장면·사건마다 보정된다(추적 외삽을 뒤집거나 채움 사후를 바꾸면 깨진다). 한 레코드에서 선택지
  하나만 골라 레코드끼리 독립으로 센다.
* **레코드의 참 결과** — 참 세계를 정답의 사후에서 뽑으므로(I4) `truth.realized`는 정답에 대해 보정된다(옛 생성기는 회전에서
  z = +12.8이었다).
"""

from __future__ import annotations

import math
from collections import defaultdict

import numpy as np
import pytest
from simgen_helpers import sample_records

from robo_jev.simgen.catalog import DOMAINS, SCENE_MARGINS, default_config
from robo_jev.simgen.posterior import exact_labels, sample_posterior
from robo_jev.simgen.rollout import available_options, rollout
from robo_jev.simgen.scenes import Infeasible, build_world, draw_context, observe
from robo_jev.simgen.spec import fill_bins, stream

N_WORLDS = 1500
WIDE = 3.0


def _prior_world(cfg, seed: int):
    """정답의 사전에 맞춘 세계 하나와 그 관측(넓은 균등 여유, 채움 ~ U(용기의 채움 범위))."""
    ctx = draw_context(cfg, stream(seed, "context"), DOMAINS[seed % len(DOMAINS)])
    for attempt in range(64):
        rng = stream(seed, "wide", attempt)
        margins = {n: float(rng.uniform(-WIDE, WIDE) * cfg.catalog["margins"][n]["half_range"][ctx.body]) for n in SCENE_MARGINS[ctx.scene]}
        try:
            world = build_world(cfg, ctx, margins, stream(seed, "world", attempt))
        except Infeasible:
            continue
        if world.container is not None:
            lo, hi = cfg.container(world.container["kind"])["fill"]
            world = world.with_values(fill=float(stream(seed, "fill").uniform(lo, hi)))
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
        true = rollout(cfg, obs, world.as_batch(), options)
        exact = exact_labels(cfg, obs)
        pick = stream(seed, "pick")
        for event, values in out.events.items():
            i = int(pick.integers(len(options)))
            groups[(obs.scene, event)].append((float(values[i].mean()), bool(true.events[event][i][0])))
        for event, values in out.world_events.items():
            groups[(obs.scene, f"world:{event}")].append((float(exact.get(event, values.mean())), bool(true.world_events[event][0])))
        if "intent" in exact and world.values.get("intent", -1) >= 0:
            j = int(pick.integers(len(exact["intent"])))
            groups[(obs.scene, "state:intent")].append((float(exact["intent"][j]), int(world.values["intent"]) == j))
        if "fill_bin" in exact:
            j = int(pick.integers(len(exact["fill_bin"])))
            groups[(obs.scene, "state:fill_bin")].append((float(exact["fill_bin"][j]), int(fill_bins(np.array([world.values["fill"]]))[0]) == j))
    return groups


def test_forward_sbc_covers_every_scene_and_the_main_events(forward_groups):
    scenes = {scene for scene, _ in forward_groups}
    assert scenes == {"p1_step_in", "p1p5_step_in_carrying", "p1_crossing", "p1_head_on", "p5_braking", "p5_turning"}
    for needed in (("p1_head_on", "contact"), ("p1_crossing", "contact"), ("p1p5_step_in_carrying", "spill"), ("p5_braking", "overrun"),
                   ("p5_turning", "spill"), ("p1_step_in", "world:stops_short"), ("p1_head_on", "world:passable")):  # fmt: skip
        assert len(forward_groups[needed]) >= 100, needed


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
    assert checked >= 20


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
