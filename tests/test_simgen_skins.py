"""상황 옷의 의미 일관성 (리뷰 1 m1): 분야 카드는 상황(settings)을 먼저 고르고, 목록 항목은 상황 태그·몸·장면 제약을 지킨다.

* 구조: 모든 (분야 × 상황 × 몸 × 장면) 조합에 쓸 수 있는 항목이 있다(설정 검사 config_errors가 같은 것을 본다).
* 생성된 관측 수천 개에서 알려진 어긋남이 나오지 않는다: 식품 공장 + 에폭시·땜납·나사 조임·프레스·코일·주물·용접, 고정 팔 + 배터리,
  견인차·터거 + 포크, 과일 따는 팔이 액체를 나름, 창고·공장·온실의 팔이 커피·차를 나름, 측정 실린더 문구 + 다른 용기 기하,
  약국·백화점·지하철 승강장·한밤 탑승구의 지게차(상황의 bodies 제약).
"""

from __future__ import annotations

import pytest

from robo_jev.simgen.catalog import DOMAINS, SCENES, config_errors, default_config
from robo_jev.simgen.generate import build_moment
from robo_jev.simgen.skins import compatible, entry_ok, entry_text, setting_allows, setting_tags

N_MOMENTS = 3000


@pytest.fixture(scope="module")
def cfg():
    return default_config()


@pytest.fixture(scope="module")
def moments(cfg):
    return [build_moment(cfg, 50_000 + i, DOMAINS[i % len(DOMAINS)]) for i in range(N_MOMENTS)]


def _skin_text(obs) -> str:
    parts = [obs.skin.setting, obs.skin.body_name, obs.skin.task, obs.skin.carry_where, obs.env.place, obs.env.obstacle or "", *obs.skin.distractors]
    if obs.container is not None:
        parts.append(obs.container.phrase)
    if obs.agent is not None:
        parts += [obs.agent.descriptor.np, obs.agent.cue_phrase or ""]
    if obs.env.sides is not None:
        parts += [s.phrase or "" for s in obs.env.sides]
    return " | ".join(parts).lower()


def test_the_cards_pass_the_structural_check(cfg):
    assert config_errors(cfg) == []


def test_every_setting_body_and_scene_has_entries_for_every_list(cfg):
    for domain in DOMAINS:
        card = cfg.domain(domain)
        for index in range(len(card["settings"])):
            tags = setting_tags(card, index)
            for body in card["bodies"]:
                if not setting_allows(card["settings"][index], body):
                    continue
                for scene in cfg.catalog["scenes_by_body"][body]:
                    where = (domain, index, body, scene)
                    names = card.get("body_names", {}).get(body) or cfg.skins["body_names"][body]
                    assert compatible(names, tags, body, scene), where
                    key = "bench" if body == "arm" else {"p1_crossing": "crossing", "p1_head_on": "corridor", "p5_turning": "turn"}.get(scene, "floor")
                    assert compatible(card["places"][key], tags, body, scene), where
                    assert len(compatible(card["distractors"], tags, body, scene)) >= 3, where


def test_entry_constraints_follow_tags_bodies_and_scenes():
    entry = {"text": "a six-axis arm in the screwdriving cell", "tags": ["auto", "electronics"], "bodies": ["arm"], "scenes": ["p1_step_in"]}
    assert entry_ok(entry, ("auto",), "arm", "p1_step_in")
    assert not entry_ok(entry, ("food",), "arm", "p1_step_in")
    assert not entry_ok(entry, ("auto",), "mobile_base", "p1_step_in")
    assert not entry_ok(entry, ("auto",), "arm", "p5_braking")
    assert entry_ok("a plain string entry", ("food",), "arm", "p5_braking") and entry_text("a plain string entry") == "a plain string entry"


FOOD_FORBIDDEN = ("epoxy", "solder", "screwdriv", "stamping", "press shop", "press area", "pneumatic press", "steel coil", "casting", "weld", "coolant", "cutting fluid", "paint booth")


def test_known_incoherent_combinations_never_appear(moments):
    seen_food = seen_arm = seen_tug = 0
    for m in moments:
        obs, text = m.obs, _skin_text(m.obs)
        if obs.domain == "factory" and "food-packaging" in obs.skin.setting:
            seen_food += 1
            assert not [w for w in FOOD_FORBIDDEN if w in text], (m.seed, text)
        if obs.body == "arm":
            seen_arm += 1
            assert "battery" not in text, (m.seed, text)
            if obs.container is not None and obs.domain in ("warehouse", "factory", "greenhouse"):
                assert not any(w in obs.container.phrase for w in ("coffee", "tea", "cocoa")), (m.seed, obs.container.phrase)
            if "fruit-picking" in obs.skin.body_name or "transplanting" in obs.skin.body_name:
                assert obs.container is None, (m.seed, obs.skin.body_name, obs.scene)
        if any(w in obs.skin.body_name for w in ("tug", "tow")):
            seen_tug += 1
            assert "fork" not in obs.skin.carry_where, (m.seed, obs.skin.body_name, obs.skin.carry_where)
        if "fork" in obs.skin.carry_where:
            assert obs.body == "forklift_amr"
        if obs.container is not None and "measuring cylinder" in obs.container.phrase:
            assert obs.container.kind == "measuring_cylinder"
        if obs.body == "forklift_amr":
            assert not any(w in obs.skin.setting for w in ("pharmacy", "department store", "metro station", "gate area")), (m.seed, obs.skin.setting)
    assert seen_food > 30 and seen_arm > 200 and seen_tug > 5


def test_every_domain_and_scene_appears_in_the_skin_sample(moments):
    assert {m.obs.domain for m in moments} == set(DOMAINS)
    assert {m.obs.scene for m in moments} == set(SCENES)
