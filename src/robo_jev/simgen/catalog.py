"""속성 목록·상황 옷 카드·가치표의 적재와 구조 검사 (docs/12 §4, Task v0a 지시서 §3.2–§3.6·§3.10).

세 설정 파일(`configs/simgen/`)을 읽어 :class:`SimgenConfig` 하나로 묶는다. 값 자체는 YAML이 정하고, 여기서는 생성기가 기대는
구조(확률 표의 합, 분야 카드가 갖춰야 할 문구, 몸·장면·여유 설계의 짝)를 검사한다 — :func:`config_errors` 가 오류 목록을
돌려주고 :func:`load_config` 는 오류가 있으면 거절한다. 현실 범위 검사(지시서 표와 대조)는 tests/test_simgen_catalog.py가 한다.

이 모듈은 :mod:`~robo_jev.simgen.skins` (항목 고르기 규칙)만 import한다(맨 아래 층).
"""

from __future__ import annotations

import functools
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from robo_jev.simgen.skins import compatible, entry_text, setting_allows

__all__ = [
    "AGENT_COLUMN",
    "AGENT_FAMILIES",
    "GROUPS",
    "HELDOUT_GROUPS",
    "SCENE_BODIES",
    "SCENE_GROUP",
    "TRAIN_GROUPS",
    "scene_kind",
    "scene_margins",
    "BODIES",
    "CATALOG_PATH",
    "CONTENTS",
    "DOMAINS",
    "INTENTS",
    "SCENES",
    "SCENE_FAMILY",
    "SCENE_MARGINS",
    "SCENE_PHENOMENA",
    "SKINS_PATH",
    "VALUES_PATH",
    "VALUES_V01_PATH",
    "SimgenConfig",
    "config_errors",
    "default_config",
    "load_config",
    "distraction_probability",
    "reaction_probability",
]

REPO = Path(__file__).resolve().parents[3]
CONFIG_DIR = REPO / "configs" / "simgen"
CATALOG_PATH = CONFIG_DIR / "catalog-v0.2.yaml"
SKINS_PATH = CONFIG_DIR / "skins-v0.2.yaml"
VALUES_PATH = CONFIG_DIR / "values-v0.yaml"
VALUES_V01_PATH = CONFIG_DIR / "values-v0.1.yaml"

#: 분야 10개 (지시서 §3.10의 순서: 창고, 공장 라인, 카페·식당, 병원 복도, 실험대, 보도·거리, 가정, 매장, 온실·농장, 공항·역).
DOMAINS = ("warehouse", "factory", "cafe", "hospital", "lab", "sidewalk", "home", "retail", "greenhouse", "transit")
BODIES = ("mobile_base", "humanoid", "forklift_amr", "arm")
SCENES = (
    "p1_step_in", "p1_crossing", "p1_head_on", "p5_braking", "p5_turning", "p1p5_step_in_carrying",
    "p3_carrying_tall", "p3_ramp", "p3_place_stack", "p4_place_check", "p4_insert", "p6_stale_track", "p6_blind_corner", "p6_sensor_conflict",
    "p3p5_carrying_mixed", "p1p6_emerging_step_in", "p4p6_stale_check", "p1p3_step_in_carrying_tall",
)  # fmt: skip
CONTENTS = ("hot", "chemical", "other")

#: 장면 → 레코드의 `meta.phenomena`.
SCENE_PHENOMENA = {
    "p1_step_in": ("P1",),
    "p1_crossing": ("P1",),
    "p1_head_on": ("P1",),
    "p5_braking": ("P5",),
    "p5_turning": ("P5",),
    "p1p5_step_in_carrying": ("P1", "P5"),
    "p3_carrying_tall": ("P3",),
    "p3_ramp": ("P3",),
    "p3_place_stack": ("P3",),
    "p4_place_check": ("P4",),
    "p4_insert": ("P4",),
    "p6_stale_track": ("P6",),
    "p6_blind_corner": ("P6",),
    "p6_sensor_conflict": ("P6",),
    "p3p5_carrying_mixed": ("P3", "P5"),
    "p1p6_emerging_step_in": ("P1", "P6"),
    "p4p6_stale_check": ("P4", "P6"),
    "p1p3_step_in_carrying_tall": ("P1", "P3"),
}
#: 현상·조합 묶음(학습은 묶음을 고르게 — Task v0b §5). P1+P3은 시험 세트 T4 전용이다(학습 생성기는 만들지 못한다 — holdout-v0b.yaml).
GROUPS = {
    "P1": ("p1_step_in", "p1_crossing", "p1_head_on"),
    "P3": ("p3_carrying_tall", "p3_ramp", "p3_place_stack"),
    "P4": ("p4_place_check", "p4_insert"),
    "P5": ("p5_braking", "p5_turning"),
    "P6": ("p6_stale_track", "p6_blind_corner", "p6_sensor_conflict"),
    "P1+P5": ("p1p5_step_in_carrying",),
    "P3+P5": ("p3p5_carrying_mixed",),
    "P1+P6": ("p1p6_emerging_step_in",),
    "P4+P6": ("p4p6_stale_check",),
    "P1+P3": ("p1p3_step_in_carrying_tall",),
}
HELDOUT_GROUPS = ("P1+P3",)
TRAIN_GROUPS = tuple(g for g in GROUPS if g not in HELDOUT_GROUPS)
SCENE_GROUP = {scene: group for group, scenes in GROUPS.items() for scene in scenes}
#: 장면 → 그 일을 하는 몸(분야 카드의 몸 가중치를 이 안으로 줄여 고른다).
SCENE_BODIES = {
    "p1_step_in": BODIES, "p1_crossing": BODIES, "p1_head_on": ("mobile_base", "humanoid", "forklift_amr"),
    "p5_braking": BODIES, "p5_turning": BODIES, "p1p5_step_in_carrying": BODIES,
    "p3_carrying_tall": ("mobile_base", "humanoid", "forklift_amr"), "p3_ramp": ("mobile_base", "humanoid", "forklift_amr"),
    "p3_place_stack": ("arm", "humanoid", "forklift_amr"), "p4_place_check": ("arm", "humanoid", "forklift_amr"), "p4_insert": ("arm", "humanoid"),
    "p6_stale_track": ("mobile_base", "humanoid", "forklift_amr"), "p6_blind_corner": ("mobile_base", "humanoid", "forklift_amr"),
    "p6_sensor_conflict": ("mobile_base", "humanoid", "forklift_amr"), "p3p5_carrying_mixed": ("mobile_base", "humanoid"),
    "p1p6_emerging_step_in": ("mobile_base", "humanoid", "forklift_amr"), "p4p6_stale_check": ("arm", "humanoid", "forklift_amr"),
    "p1p3_step_in_carrying_tall": ("mobile_base", "humanoid"),
}  # fmt: skip
#: 장면 → 상대 주체의 의도 갈래(의도 집합·단서 표의 열쇠). 의도를 묻지 않는 장면은 None.
SCENE_FAMILY = {scene: None for scene in SCENES} | {
    "p1_step_in": "step_in",
    "p1_crossing": "crossing",
    "p1_head_on": "head_on",
    "p1p5_step_in_carrying": "step_in",
    "p1p6_emerging_step_in": "step_in",
    "p1p3_step_in_carrying_tall": "step_in",
}
#: 장면 → 상대 종류를 고르는 분야 카드 칸(의도가 없어도 상대가 있는 P6 장면 포함). 팔은 언제나 "arm"(작업 영역의 손).
AGENT_COLUMN = {scene: family for scene, family in SCENE_FAMILY.items() if family is not None} | {
    "p6_stale_track": "step_in",
    "p6_blind_corner": "crossing",
}
#: 갈래별 이산 의도(상태 choice의 선택지).
INTENTS = {
    "step_in": ("stay", "cross", "back"),
    "crossing": ("cross", "yield", "hurry"),
    "head_on": ("keep", "aside_left", "aside_right", "stop"),
}
#: 장면 → 그 장면이 정의하는 결정 여유(catalog `margins`의 열쇠). 움직임 갈래가 있는 장면은 :func:`scene_margins`.
SCENE_MARGINS = {
    "p1_step_in": ("step_in",),
    "p1_crossing": ("crossing",),
    "p1_head_on": ("head_on",),
    "p5_braking": ("spill", "stop"),
    "p5_turning": ("spill",),
    "p1p5_step_in_carrying": ("step_in", "spill"),
    "p3_ramp": ("tip",),
    "p3_place_stack": ("stack",),
    "p4_place_check": ("seat",),
    "p4_insert": ("seat",),
    "p4p6_stale_check": ("seat",),
    "p6_stale_track": ("sight",),
    "p6_blind_corner": ("corner",),
    "p6_sensor_conflict": ("late",),
    "p1p6_emerging_step_in": ("step_in",),
    "p1p3_step_in_carrying_tall": ("step_in", "tip"),
}
_MANEUVER_MARGINS = {
    ("p3_carrying_tall", "brake"): ("tip", "stop"), ("p3_carrying_tall", "turn"): ("tip",), ("p3_carrying_tall", "start"): ("tip",),
    ("p3p5_carrying_mixed", "brake"): ("tip", "spill", "stop"), ("p3p5_carrying_mixed", "turn"): ("tip", "spill"),
}  # fmt: skip


_KIND = {"p3": "p3", "p1p3": "p3", "p3p5": "p3", "p4": "p4", "p4p6": "p4", "p6": "p6", "p1p6": "p6"}


def scene_kind(scene: str) -> str | None:
    """새 현상 모듈의 이름(p3 / p4 / p6) — v0a 장면(P1·P5)이면 None."""
    return _KIND.get(scene.split("_")[0])


#: 숫자 쌍이 옮기는 결정 변수(앞의 것부터) — 핵심 질문이 묻는 것을 정하는 여유(수정 1 I4: 라운드 0은 장면의 여유 가운데 하나를 무작위로
#: 옮겨, 핵심이 묻지 않는 여유를 옮긴 쌍(쏟아짐을 옮겼는데 핵심은 멈춤)·핵심 정답이 그대로인 쌍이 있었다). 오래된 추적의 핵심(아직 있나)은
#: 본 지 얼마나 됐나(나이)가 정한다; 센서 불일치의 핵심(정말 있나)은 이산 보고(단서·두 센서)만 정해 숫자 하나로 넘길 수 없다 — 단어 쌍만.
PAIR_TARGETS = {
    "p1_step_in": ("step_in",), "p1_crossing": ("crossing",), "p1_head_on": ("head_on",), "p5_braking": ("spill", "stop"), "p5_turning": ("spill",),
    "p1p5_step_in_carrying": ("step_in",), "p3_carrying_tall": ("tip",), "p3p5_carrying_mixed": ("tip",), "p3_ramp": ("tip",),
    "p3_place_stack": ("stack",), "p4_place_check": ("seat",), "p4_insert": ("seat",), "p4p6_stale_check": ("seat",), "p6_stale_track": ("age",),
    "p6_blind_corner": ("corner",), "p6_sensor_conflict": (), "p1p6_emerging_step_in": ("step_in",), "p1p3_step_in_carrying_tall": ("step_in",),
}  # fmt: skip


def pair_targets(scene: str) -> tuple[str, ...]:
    """숫자 쌍이 옮길 결정 변수(핵심 질문이 묻는 여유) — 비면 그 장면은 단어 쌍만."""
    return PAIR_TARGETS[scene]


def scene_margins(scene: str, maneuver: str | None = None) -> tuple[str, ...]:
    """장면(과 움직임 갈래)이 정의하는 결정 여유 이름 — 첫 것이 쌍(사실 하나 바꾸기)의 기본 변수다."""
    if (scene, maneuver) in _MANEUVER_MARGINS:
        return _MANEUVER_MARGINS[(scene, maneuver)]
    return SCENE_MARGINS[scene]
#: 상대 주체를 고르는 분야 카드 칸: 팔은 언제나 "arm"(작업 영역의 손), 그 밖은 장면 갈래.
AGENT_FAMILIES = ("step_in", "crossing", "head_on", "arm")
_P5_SCENES = ("p5_braking", "p5_turning", "p1p5_step_in_carrying")


@dataclass(frozen=True)
class SimgenConfig:
    """세 설정 파일의 내용(파싱한 dict 그대로)."""

    catalog: dict
    skins: dict
    values: dict

    @property
    def versions(self) -> dict[str, str]:
        return {"catalog": self.catalog["version"], "skins": self.skins["version"], "values": self.values["version"]}

    def body(self, name: str) -> dict:
        return self.catalog["bodies"][name]

    def agent(self, kind: str) -> dict:
        return self.catalog["agents"][kind]

    def container(self, kind: str) -> dict:
        return self.catalog["containers"][kind]

    def domain(self, name: str) -> dict:
        return self.skins["domains"][name]

    def agent_family(self, scene: str, body: str) -> str | None:
        """그 장면·몸에서 상대 주체를 고르는 분야 카드 칸(없으면 None)."""
        family = AGENT_COLUMN.get(scene)
        if family is None:
            return None
        return "arm" if body == "arm" else family


def reaction_probability(config: SimgenConfig, kind: str, cue: str | None) -> float:
    """가로지르는 상대가 로봇이 자기 줄을 막고 있을 때 늦게라도 설 확률: 갈래 기본값 × 단서 배수(최대 max). 공은 0."""
    spec = config.catalog["crossing_reaction"]
    base = spec["base"][config.agent(kind)["group"]]
    factor = spec["cue_factor"].get(cue, 1.0) if cue else 1.0
    return float(min(spec["max"], base * factor))


def distraction_probability(config: SimgenConfig, kind: str, cue: str | None) -> float:
    """주의가 딴 데 있을 확률: 단서가 정해 주면 그 값(static_avoidance.distracted_cue), 아니면 갈래의 사전값. 공은 0(피하지 않으므로 무의미)."""
    spec = config.catalog["static_avoidance"]
    group = config.agent(kind)["group"]
    if cue is not None and cue in spec["distracted_cue"]:
        return float(spec["distracted_cue"][cue])
    return float(spec["distracted_prior"][group])


def _read_yaml(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def load_config(catalog_path: Path = CATALOG_PATH, skins_path: Path = SKINS_PATH, values_path: Path = VALUES_PATH) -> SimgenConfig:
    """세 파일을 읽고 구조를 검사한다. 오류가 하나라도 있으면 ValueError."""
    config = SimgenConfig(catalog=_read_yaml(catalog_path), skins=_read_yaml(skins_path), values=_read_yaml(values_path))
    errors = config_errors(config)
    if errors:
        raise ValueError("simgen config errors:\n" + "\n".join(f"- {e}" for e in errors))
    return config


@functools.lru_cache(maxsize=4)
def default_config(values_path: Path = VALUES_PATH) -> SimgenConfig:
    """저장소의 설정(한 프로세스에서 한 번만 읽는다). 가치표는 values-v0가 기본이다(사용자가 고르기 전까지)."""
    return load_config(values_path=values_path)


def _is_range(value: Any) -> bool:
    return isinstance(value, list) and len(value) == 2 and all(isinstance(v, (int, float)) for v in value) and value[0] <= value[1]


def _sums_to_one(weights: dict) -> bool:
    return abs(sum(weights.values()) - 1.0) <= 1e-9 and all(v >= 0 for v in weights.values())


def _catalog_errors(catalog: dict) -> list[str]:
    errors: list[str] = []
    bodies = catalog.get("bodies", {})
    if tuple(bodies) != BODIES:
        errors.append(f"catalog bodies must be {BODIES}, got {tuple(bodies)}")
    for name, spec in bodies.items():
        for key in ("speed", "latency", "radius", "accel"):
            if not _is_range(spec.get(key)):
                errors.append(f"body {name}: {key} is not a [lo, hi] range")
        for level in ("gentle", "moderate", "hard"):
            if not _is_range(spec.get("decel", {}).get(level)):
                errors.append(f"body {name}: decel.{level} is not a range")
        if spec.get("can_sidestep") and not _is_range(spec.get("sidestep", {}).get("offset")):
            errors.append(f"body {name}: can_sidestep without sidestep.offset")
        if not sum(spec.get("sensor_suites", {}).values()) or not _sums_to_one(spec.get("sensor_suites", {})):
            errors.append(f"body {name}: sensor_suites weights must sum to 1")
        for suite in spec.get("sensor_suites", {}):
            if suite not in catalog.get("sensor_suites", {}):
                errors.append(f"body {name}: unknown sensor suite {suite}")
    for kind, spec in catalog.get("loads", {}).items():
        if not all(_is_range(spec.get(k)) for k in ("height", "base", "com_ratio")) or not set(spec.get("bodies", [])) <= set(BODIES):
            errors.append(f"load {kind}: bad ranges or bodies")
        if spec.get("fall") not in ("fall_light", "fall_fragile", "fall_heavy", "fall_chemical"):
            errors.append(f"load {kind}: unknown fall cost {spec.get('fall')}")
    for kind, spec in catalog.get("decks", {}).items():
        if not _is_range(spec.get("mu")):
            errors.append(f"deck {kind}: mu must be a range")
    for kind, spec in catalog.get("p4", {}).get("items", {}).items():
        family = spec.get("family")
        if family not in ("place", "insert") or spec.get("class") not in catalog["p4"][family]["classes"]:
            errors.append(f"p4 item {kind}: bad family/class")
    for family in ("place", "insert"):
        for name, cls in catalog.get("p4", {}).get(family, {}).get("classes", {}).items():
            states = catalog["p4"][family]["states"]
            if set(cls["prior"]) != set(states) or not _sums_to_one(cls["prior"]):
                errors.append(f"p4 {family}.{name}: prior must be a distribution over {states}")
    for kind, spec in catalog.get("agents", {}).items():
        if spec.get("group") not in ("person", "vehicle", "robot", "hand", "ball"):
            errors.append(f"agent {kind}: unknown group {spec.get('group')}")
        if not _is_range(spec.get("speed")) or not _is_range(spec.get("radius")):
            errors.append(f"agent {kind}: speed/radius must be ranges")
        for family, prior in spec.get("intent_prior", {}).items():
            if tuple(prior) != INTENTS.get(family) or not _sums_to_one(prior):
                errors.append(f"agent {kind}: intent_prior.{family} must be a distribution over {INTENTS.get(family)}")
            if family not in catalog.get("cues", {}).get(spec.get("group"), {}):
                errors.append(f"agent {kind}: no cue table for group {spec.get('group')} family {family}")
        if spec.get("group") != "ball" and not spec.get("intent_prior"):
            errors.append(f"agent {kind}: only balls may lack intents")
    for group, families in catalog.get("cues", {}).items():
        for family, table in families.items():
            intents = INTENTS.get(family, ())
            for cue, row in table.items():
                if tuple(row) != intents:
                    errors.append(f"cues {group}.{family}.{cue}: columns must be {intents}")
            for intent in intents:
                column = [row.get(intent, 0.0) for row in table.values()]
                if abs(sum(column) - 1.0) > 1e-9 or any(v <= 0 for v in column):
                    errors.append(f"cues {group}.{family}: column {intent} must be a positive distribution (sum {sum(column):.3f})")
    for kind, spec in catalog.get("containers", {}).items():
        if not _is_range(spec.get("fill")) or not set(spec.get("bodies", [])) <= set(BODIES):
            errors.append(f"container {kind}: bad fill range or bodies")
    for name, margin in catalog.get("margins", {}).items():
        if isinstance(margin, dict) and "sigma" in margin:
            for body, sigma in margin["sigma"].items():
                if not 0 < sigma < margin["half_range"].get(body, 0):
                    errors.append(f"margin {name}.{body}: need 0 < sigma < half_range")
    return errors


_CARRY_SCENES = ("p5_braking", "p5_turning", "p1p5_step_in_carrying", "p3p5_carrying_mixed")
_LOAD_SCENES = ("p3_carrying_tall", "p3_ramp", "p3p5_carrying_mixed", "p1p3_step_in_carrying_tall")
_STEP_IN_SCENES = ("p1_step_in", "p1p5_step_in_carrying", "p1p6_emerging_step_in", "p1p3_step_in_carrying_tall")
_PLACE_KEY = {"p1_crossing": "crossing", "p1_head_on": "corridor", "p5_turning": "turn"}


def _all_tags(card: dict) -> set[str]:
    return {t for e in card.get("settings", []) if isinstance(e, dict) for t in e.get("tags", [])}


def _entries(card: dict):
    """카드의 모든 제약 항목(태그 오타 검사용)."""
    places = card.get("places", {})
    lists = [card.get(k, []) for k in ("obstacles", "bench_obstacles", "occluders", "blockers", "distractors", "containers")]
    lists += list(places.values()) + list(card.get("tasks", {}).values()) + list(card.get("carry_tasks", {}).values())
    lists += list(card.get("descriptors", {}).values()) + list(card.get("body_names", {}).values())
    for entries in lists:
        yield from (e for e in entries if isinstance(e, dict))


def _skins_errors(config: SimgenConfig) -> list[str]:
    """분야 카드 구조와 의미 일관성: 모든 (상황 × 몸 × 장면) 조합에 쓸 항목이 목록마다 있는가."""
    catalog, skins = config.catalog, config.skins
    errors: list[str] = []
    domains = skins.get("domains", {})
    if tuple(domains) != DOMAINS:
        errors.append(f"skins domains must be {DOMAINS}, got {tuple(domains)}")
    for group, families in catalog.get("cues", {}).items():
        for table in families.values():
            for cue in table:
                if not skins.get("cue_phrases", {}).get(group, {}).get(cue):
                    errors.append(f"cue_phrases.{group}.{cue} missing")
    for domain, card in domains.items():
        where = f"domain {domain}"
        bodies = card.get("bodies", {})
        if not _sums_to_one(bodies) or not set(bodies) <= set(BODIES):
            errors.append(f"{where}: bodies must be a distribution over known bodies")
        floors = card.get("floor", {})
        if not _sums_to_one(floors) or not set(floors) <= set(catalog.get("floors", {})):
            errors.append(f"{where}: floor must be a distribution over catalog floors")
        if not 0.0 <= card.get("side_obstacle_prior", -1) <= 1.0:
            errors.append(f"{where}: side_obstacle_prior must be in [0, 1]")
        settings = card.get("settings", [])
        if len(settings) < 3 or not all(isinstance(e, dict) and e.get("text") and isinstance(e.get("tags"), list) for e in settings):
            errors.append(f"{where}: settings must be >= 3 entries of {{text, tags}}")
            continue
        known = _all_tags(card)
        for entry in _entries(card):
            unknown = set(entry.get("tags", [])) - known
            if unknown:
                errors.append(f"{where}: unknown tags {sorted(unknown)} in {entry}")
        for entry in card.get("containers", []):
            if entry.get("kind") not in catalog.get("containers", {}) or entry.get("content") not in CONTENTS or not entry.get("phrases"):
                errors.append(f"{where}: bad container entry {entry}")
        for body, weight in bodies.items():
            if weight > 0 and not any(setting_allows(s, body) for s in settings):
                errors.append(f"{where}: no setting allows {body}")
        for index, setting in enumerate(settings):
            tags = tuple(setting["tags"])
            if not set(setting.get("bodies") or []) <= set(BODIES):
                errors.append(f"{where} setting {index}: unknown bodies {setting.get('bodies')}")
            for body, weight in bodies.items():
                if weight <= 0 or not setting_allows(setting, body):
                    continue
                for scene in SCENES:
                    if body in SCENE_BODIES[scene]:
                        errors.extend(_combination_errors(config, domain, card, tags, body, scene, f"{where} setting {index} {body} {scene}"))
    return sorted(set(errors))


def _combination_errors(config: SimgenConfig, domain: str, card: dict, tags, body: str, scene: str, where: str) -> list[str]:
    errors: list[str] = []
    carry = scene in _CARRY_SCENES or scene in _LOAD_SCENES

    def need(entries, what: str, at_least: int = 1):
        ok = compatible(entries or [], tags, body, scene)
        if len(ok) < at_least:
            errors.append(f"{where}: {what} has {len(ok)} usable entries (need {at_least})")
        return ok

    names = need(card.get("body_names", {}).get(body) or config.skins["body_names"][body], "body_names")
    for name in names:
        own = name.get("carry_tasks" if carry else "tasks") if isinstance(name, dict) else None
        tasks = own if own else card.get("carry_tasks" if carry else "tasks", {}).get(body, [])
        usable = need(tasks, f"{'carry_tasks' if carry else 'tasks'} for {entry_text(name)!r}")
        if carry and not all("{container}" in entry_text(t) for t in usable):
            errors.append(f"{where}: carry task without {{container}}")
    places = card.get("places", {})
    need(places.get("bench") if body == "arm" else places.get(_PLACE_KEY.get(scene, "floor")), "places")
    catalog = config.catalog
    if scene in _LOAD_SCENES:
        loads = [e for e in compatible(card.get("loads", []), tags, body, scene) if body in catalog["loads"][e["kind"]]["bodies"]]
        if not loads:
            errors.append(f"{where}: no load a {body} can carry")
        if not config.skins.get("decks", {}).get(body):
            errors.append(f"{where}: no deck phrases for {body}")
    if scene == "p3_ramp":
        need(card.get("ramps"), "ramps")
        need(card.get("ramp_ends"), "ramp_ends")
        need(card.get("detours"), "detours")
    if scene == "p3_place_stack":
        items = [e for e in compatible(card.get("stack_items", []), tags, body, scene) if body in catalog["stack_items"][e["kind"]]["bodies"]]
        if not items:
            errors.append(f"{where}: no stack item a {body} can place")
        need(card.get("stack_spots"), "stack_spots")
    if scene in ("p4_place_check", "p4_insert", "p4p6_stale_check"):
        family = "insert" if scene == "p4_insert" else "place"
        items = [e for e in compatible(card.get("p4_items", []), tags, body, scene)
                 if catalog["p4"]["items"][e["kind"]]["family"] == family and body in catalog["p4"]["items"][e["kind"]]["bodies"]]  # fmt: skip
        if not items:
            errors.append(f"{where}: no {family} item for {body}")
        if scene == "p4p6_stale_check" and not config.skins.get("stale_camera", {}).get(body):
            errors.append(f"{where}: no stale-camera phrases for {body}")
    if scene == "p6_blind_corner":
        need(card.get("corners"), "corners")
    if scene == "p6_sensor_conflict":
        need(card.get("conflicts"), "conflicts")
    if scene in ("p6_stale_track", "p1p6_emerging_step_in"):
        need(card.get("occluders"), "occluders")
    family = config.agent_family(scene, body)
    if family is not None:
        weights = card.get("agents", {}).get(family)
        if not weights or not _sums_to_one(weights):
            errors.append(f"{where}: agents.{family} must be a distribution")
        else:
            for kind, weight in weights.items():
                spec = config.catalog["agents"].get(kind)
                if spec is None:
                    errors.append(f"{where}: unknown agent {kind}")
                    continue
                if (family == "arm") != (spec["group"] == "hand"):
                    errors.append(f"{where}: {kind} does not fit (arm ↔ hand only)")
                if family in INTENTS and spec["group"] != "ball" and family not in spec.get("intent_prior", {}):
                    errors.append(f"{where}: {kind} has no intent prior for {family}")
                if weight > 0:
                    usable = need(card.get("descriptors", {}).get(kind), f"descriptors.{kind}")
                    if not all(d.get("np") and d.get("ref") for d in usable):
                        errors.append(f"{where}: descriptors.{kind} missing np/ref")
    if scene in _CARRY_SCENES:
        containers = [c for c in compatible(card.get("containers", []), tags, body, scene) if body in config.catalog["containers"][c["kind"]]["bodies"]]
        if not containers:
            errors.append(f"{where}: no container a {body} can carry")
    if scene in ("p5_braking", "p3_carrying_tall", "p3p5_carrying_mixed"):
        need(card.get("bench_obstacles") if body == "arm" else card.get("obstacles"), "obstacles")
    if scene in _STEP_IN_SCENES and config.catalog["bodies"][body]["can_sidestep"]:
        need(card.get("occluders"), "occluders", 2)
        need(card.get("blockers"), "blockers", 2)
    need(card.get("distractors"), "distractors", 3)
    return errors


def _values_errors(values: dict) -> list[str]:
    errors = []
    costs = values.get("costs", {})
    if values.get("tau", 0) <= 0:
        errors.append("values: tau must be positive")
    speed = values.get("contact_speed", {})
    if set(speed.get("v_ref", {})) != set(BODIES) or not 0.0 < speed.get("floor", 0.0) <= 1.0:
        errors.append("values: contact_speed needs v_ref for every body and 0 < floor <= 1")
    if set(values.get("hard_brake_cost", {})) != set(BODIES) | {"forklift_amr_loaded"}:
        errors.append("values: hard_brake_cost needs every body and forklift_amr_loaded")
    for name, override in values.get("criteria", {}).items():
        if not set(override) <= set(costs):
            errors.append(f"values: criterion {name} overrides unknown costs")
    return errors


def config_errors(config: SimgenConfig) -> list[str]:
    """설정의 구조 오류 목록(비면 통과)."""
    return _catalog_errors(config.catalog) + _skins_errors(config) + _values_errors(config.values)
