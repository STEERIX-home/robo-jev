"""결정 여유 표본기 (Task v0a 지시서 §3.7·§5 '표본기', docs/12 §8).

* 목표 여유는 혼합분포 0.6·N(0, σ_m) + 0.4·U(−M, M)에서 뽑는다.
* 참 세계는 그 여유를 정확히 갖는다(풀이와 따로 잰 여유 = 목표).
* 생성한 2,000개에서 여유가 0 근처에 몰린다: |m| < 1.5·σ_m 비율이 설계값과 맞는다.
* 핵심 noul 정답의 전체 평균이 0.3~0.7.
"""

import math

import numpy as np
import pytest
from simgen_helpers import sample_records

from robo_jev.simgen.catalog import DOMAINS, SCENE_MARGINS, default_config
from robo_jev.simgen.scenes import Infeasible, build_world, draw_context, draw_margin, draw_margins, margins_of
from robo_jev.simgen.spec import stream


def design_share(cfg, name: str, body: str, k: float = 1.5) -> float:
    """설계상 P(|m| < k·σ) = 0.6·P(|Z| < k) + 0.4·min(1, k·σ / M)."""
    spec = cfg.catalog["margins"][name]
    sigma, half = spec["sigma"][body], spec["half_range"][body]
    w = cfg.catalog["margins"]["normal_weight"]
    return w * math.erf(k / math.sqrt(2.0)) + (1.0 - w) * min(1.0, k * sigma / half)


@pytest.mark.parametrize(("name", "body"), [("step_in", "mobile_base"), ("crossing", "arm"), ("spill", "forklift_amr"), ("head_on", "humanoid")])
def test_margin_draws_follow_the_documented_mixture(name, body):
    cfg = default_config()
    rng = np.random.default_rng(123)
    draws = np.array([draw_margin(cfg, name, body, rng) for _ in range(20_000)])
    spec = cfg.catalog["margins"][name]
    sigma, half = spec["sigma"][body], spec["half_range"][body]
    assert np.mean(np.abs(draws) < 1.5 * sigma) == pytest.approx(design_share(cfg, name, body), abs=0.012)
    # M 밖은 정규 성분에서만 나온다: 0.6·P(|Z| > M/σ)
    beyond = 0.6 * (1.0 - math.erf(half / sigma / math.sqrt(2.0)))
    assert np.mean(np.abs(draws) > half) == pytest.approx(beyond, abs=0.006)
    assert abs(float(np.mean(draws))) < 0.03 * half


def test_built_worlds_realise_the_target_margin_exactly():
    cfg = default_config()
    checked = 0
    for seed in range(400):
        ctx = draw_context(cfg, stream(seed, "context"), DOMAINS[seed % 10])
        margins = draw_margins(cfg, ctx, stream(seed, "margin"))
        try:
            world = build_world(cfg, ctx, margins, stream(seed, "world"))
        except Infeasible:
            continue
        measured = margins_of(cfg, world)
        assert set(measured) == set(SCENE_MARGINS[ctx.scene])
        for name, target in margins.items():
            assert measured[name] == pytest.approx(target, abs=1e-9), (seed, ctx.scene, name)
        checked += 1
    assert checked >= 300


def test_margins_concentrate_near_zero_as_designed_in_the_2000_sample():
    """쌍의 변형(여유를 일부러 옮긴 것)과 여유를 다시 뽑은 레코드는 뺀다 — 나머지는 표본기 그대로다."""
    cfg = default_config()
    hits, expected, n = 0, 0.0, 0
    for r in sample_records():
        meta = r["meta"]
        if (meta["pair"] and meta["pair"]["role"] == "variant") or meta["margin_redraws"]:
            continue
        for name, m in meta["margins"].items():
            sigma = cfg.catalog["margins"][name]["sigma"][meta["body"]]
            hits += abs(m) < 1.5 * sigma
            expected += design_share(cfg, name, meta["body"])
            n += 1
    observed, design = hits / n, expected / n
    se = math.sqrt(design * (1 - design) / n)
    assert n > 2000
    assert abs(observed - design) < 4 * se + 0.01, (observed, design, n)


def test_key_noul_mean_is_between_0_3_and_0_7():
    records = sample_records()
    values = [r["labels"][r["meta"]["key_question"]]["p_yes"] for r in records]
    assert 0.3 <= float(np.mean(values)) <= 0.7
    # 장면마다도 한쪽으로 쏠리지 않는다(0.15~0.85)
    by_scene: dict[str, list[float]] = {}
    for r, v in zip(records, values, strict=True):
        by_scene.setdefault(r["meta"]["scene"], []).append(v)
    for scene, vals in by_scene.items():
        assert 0.15 <= float(np.mean(vals)) <= 0.85, (scene, float(np.mean(vals)))
