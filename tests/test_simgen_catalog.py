"""simgen 속성 목록·상황 옷 카드·가치표 (Task v0a 지시서 §3.2·§3.3·§3.6·§3.10, docs/12 §4).

속성 목록의 값이 현실 범위(지시서 표) 안인지, 확률 표가 확률인지, 분야 카드 열 개가 생성에 필요한 문구를 다 갖췄는지 본다.
"""

import math

import pytest

from robo_jev.simgen import catalog as cat

# 지시서 §3.2 표: 몸별 속도·감속 세 단계·반응·반지름의 허용 범위.
BRIEF_BODIES = {
    "mobile_base": {"speed": (0.3, 1.5), "gentle": (0.8, 1.2), "moderate": (1.5, 2.0), "hard": (2.5, 3.5), "latency": (0.1, 0.3), "radius": (0.25, 0.40)},
    "humanoid": {"speed": (0.3, 1.2), "gentle": (0.5, 0.8), "moderate": (1.0, 1.5), "hard": (1.8, 2.5), "latency": (0.15, 0.35), "radius": (0.25, 0.35)},
    "forklift_amr": {"speed": (0.5, 2.5), "gentle": (0.5, 1.0), "moderate": (1.2, 2.0), "hard": (2.5, 3.5), "latency": (0.2, 0.5), "radius": (0.6, 1.0)},
    "arm": {"speed": (0.1, 1.0), "gentle": (1.0, 2.0), "moderate": (3.0, 5.0), "hard": (6.0, 10.0), "latency": (0.02, 0.1), "radius": (0.05, 0.15)},
}  # fmt: skip

# 지시서 §3.3: 상대 주체의 속도 범위.
BRIEF_AGENT_SPEEDS = {
    "adult": (0.8, 1.6), "child": (1.0, 3.0), "elder_walker": (0.3, 0.9), "cyclist": (2.5, 6.0), "forklift": (1.0, 3.0),
    "cart": (0.5, 1.2), "ball": (1.0, 5.0), "other_robot": (0.3, 1.5), "hand": (0.2, 1.5),
}  # fmt: skip


@pytest.fixture(scope="module")
def config():
    return cat.load_config()


def within(inner, outer) -> bool:
    return outer[0] - 1e-12 <= inner[0] <= inner[1] <= outer[1] + 1e-12


def test_config_versions_are_the_v0_files(config):
    assert config.catalog["version"] == "catalog-v0"
    assert config.skins["version"] == "skins-v0"
    assert config.values["version"] == "values-v0"


def test_body_ranges_stay_inside_the_brief_table(config):
    for body, brief in BRIEF_BODIES.items():
        spec = config.catalog["bodies"][body]
        assert within(spec["speed"], brief["speed"]), body
        for level in ("gentle", "moderate", "hard"):
            assert within(spec["decel"][level], brief[level]), (body, level)
        assert within(spec["latency"], brief["latency"]), body
        assert within(spec["radius"], brief["radius"]), body


def test_decel_levels_are_ordered_gentle_below_moderate_below_hard(config):
    for body, spec in config.catalog["bodies"].items():
        decel = spec["decel"]
        assert decel["gentle"][1] <= decel["moderate"][0] and decel["moderate"][1] <= decel["hard"][0], body


def test_agent_speed_ranges_stay_inside_the_brief_list(config):
    assert set(config.catalog["agents"]) == set(BRIEF_AGENT_SPEEDS)
    for kind, brief in BRIEF_AGENT_SPEEDS.items():
        assert within(config.catalog["agents"][kind]["speed"], brief), kind


def test_child_intent_prior_is_wider_than_adult(config):
    """아이는 의도 분산이 넓다(지시서 §3.3) — 끼어든 아이의 사전 엔트로피가 어른보다 크다."""

    def entropy(p):
        return -sum(v * math.log(v) for v in p.values() if v > 0)

    agents = config.catalog["agents"]
    assert entropy(agents["child"]["intent_prior"]["step_in"]) > entropy(agents["adult"]["intent_prior"]["step_in"])


def test_intent_priors_are_distributions_over_the_scene_intents(config):
    for kind, spec in config.catalog["agents"].items():
        for family, prior in spec.get("intent_prior", {}).items():
            assert tuple(prior) == cat.INTENTS[family], (kind, family)
            assert sum(prior.values()) == pytest.approx(1.0, abs=1e-9), (kind, family)
            assert all(0 < v < 1 for v in prior.values()), (kind, family)


def test_cue_likelihood_columns_sum_to_one_for_every_intent(config):
    for group, families in config.catalog["cues"].items():
        for family, table in families.items():
            for intent in cat.INTENTS[family]:
                column = [row[intent] for row in table.values()]
                assert sum(column) == pytest.approx(1.0, abs=1e-9), (group, family, intent)
                assert all(v > 0 for v in column), (group, family, intent)


def test_every_agent_with_intents_has_a_cue_table_for_each_family(config):
    for kind, spec in config.catalog["agents"].items():
        for family in spec.get("intent_prior", {}):
            assert family in config.catalog["cues"][spec["group"]], (kind, family)


def test_every_cue_has_at_least_one_phrase(config):
    for group, families in config.catalog["cues"].items():
        keys = {cue for table in families.values() for cue in table}
        for cue in keys:
            phrases = config.skins["cue_phrases"][group][cue]
            assert phrases and all("{Ref}" in p or "{ref}" in p for p in phrases), (group, cue)


def test_containers_are_physically_plausible(config):
    for kind, spec in config.catalog["containers"].items():
        assert 0.04 <= spec["diameter"] <= 0.6, kind
        assert 0.03 <= spec["height"] <= 0.5, kind
        lo, hi = spec["fill"]
        assert 0.0 < lo < hi <= 1.0, kind
        assert 0.0 <= spec["lid_prior"] < 1.0, kind
        assert set(spec["bodies"]) <= set(cat.BODIES), kind


def test_values_table_is_the_brief_proposal(config):
    costs = config.values["costs"]
    assert costs == {
        "contact_person": 100, "contact_vulnerable": 150, "contact_vehicle": 120, "contact_object": 15,
        "spill_hot": 25, "spill_chemical": 40, "spill_other": 8, "delay_per_s": 1.0,
    }  # fmt: skip
    assert config.values["tau"] == 2.0
    for criterion in config.values["criteria"].values():
        assert set(criterion) <= set(costs)


def test_agent_contact_categories_are_cost_items(config):
    for kind, spec in config.catalog["agents"].items():
        assert spec["contact"] in config.values["costs"], kind


def test_scene_weights_are_distributions_and_arm_has_no_head_on(config):
    for body, weights in config.catalog["scenes_by_body"].items():
        assert set(weights) <= set(cat.SCENES)
        assert sum(weights.values()) == pytest.approx(1.0, abs=1e-9), body
    assert "p1_head_on" not in config.catalog["scenes_by_body"]["arm"]


def test_margin_designs_cover_every_body_that_uses_them(config):
    margins = config.catalog["margins"]
    for body, weights in config.catalog["scenes_by_body"].items():
        for scene in weights:
            for name in cat.SCENE_MARGINS[scene]:
                assert body in margins[name]["sigma"] and body in margins[name]["half_range"], (scene, name, body)
                assert 0 < margins[name]["sigma"][body] < margins[name]["half_range"][body]


def test_there_are_exactly_the_ten_domains_of_the_brief(config):
    assert tuple(config.skins["domains"]) == cat.DOMAINS
    assert len(cat.DOMAINS) == 10


def test_every_domain_card_has_what_generation_needs(config):
    """분야 카드마다: 몸 가중치, 몸별 이름·과제·나르기 과제, 쓰는 장면의 상대 종류와 그 묘사, 몸이 들 수 있는 용기, 장소·장애물·방해 문구."""
    assert cat.config_errors(config) == []
    for domain, card in config.skins["domains"].items():
        assert sum(card["bodies"].values()) == pytest.approx(1.0, abs=1e-9), domain
        assert len(card["distractors"]) >= 5, domain
        assert len(card["settings"]) >= 3, domain


def test_config_errors_reports_a_broken_cue_table(config):
    broken = cat.SimgenConfig(
        catalog={**config.catalog, "cues": {**config.catalog["cues"], "person": {**config.catalog["cues"]["person"], "step_in": {"slowing": {"stay": 1.0, "cross": 0.5, "back": 1.0}}}}},
        skins=config.skins,
        values=config.values,
    )  # fmt: skip
    errors = cat.config_errors(broken)
    assert any("person" in e and "step_in" in e for e in errors)
