"""속성 목록·상황 옷 카드·가치표의 적재와 구조 검사 (docs/12 §4, Task v0a 지시서 §3.2–§3.6·§3.10).

세 설정 파일(`configs/simgen/`)을 읽어 :class:`SimgenConfig` 하나로 묶는다. 값 자체는 YAML이 정하고, 여기서는 생성기가 기대는
구조(확률 표의 합, 분야 카드가 갖춰야 할 문구, 몸·장면·여유 설계의 짝)를 검사한다 — :func:`config_errors` 가 오류 목록을
돌려주고 :func:`load_config` 는 오류가 있으면 거절한다. 현실 범위 검사(지시서 표와 대조)는 tests/test_simgen_catalog.py가 한다.

이 모듈은 다른 simgen 모듈을 import하지 않는다(맨 아래 층).
"""

from __future__ import annotations

import functools
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

__all__ = [
    "AGENT_FAMILIES",
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
    "SimgenConfig",
    "config_errors",
    "default_config",
    "load_config",
]

REPO = Path(__file__).resolve().parents[3]
CONFIG_DIR = REPO / "configs" / "simgen"
CATALOG_PATH = CONFIG_DIR / "catalog-v0.yaml"
SKINS_PATH = CONFIG_DIR / "skins-v0.yaml"
VALUES_PATH = CONFIG_DIR / "values-v0.yaml"

#: 분야 10개 (지시서 §3.10의 순서: 창고, 공장 라인, 카페·식당, 병원 복도, 실험대, 보도·거리, 가정, 매장, 온실·농장, 공항·역).
DOMAINS = ("warehouse", "factory", "cafe", "hospital", "lab", "sidewalk", "home", "retail", "greenhouse", "transit")
BODIES = ("mobile_base", "humanoid", "forklift_amr", "arm")
SCENES = ("p1_step_in", "p1_crossing", "p1_head_on", "p5_braking", "p5_turning", "p1p5_step_in_carrying")
CONTENTS = ("hot", "chemical", "other")

#: 장면 → 레코드의 `meta.phenomena`.
SCENE_PHENOMENA = {
    "p1_step_in": ("P1",),
    "p1_crossing": ("P1",),
    "p1_head_on": ("P1",),
    "p5_braking": ("P5",),
    "p5_turning": ("P5",),
    "p1p5_step_in_carrying": ("P1", "P5"),
}
#: 장면 → 상대 주체의 갈래(의도 집합·단서 표의 열쇠). P5 단독 장면에는 상대가 없다.
SCENE_FAMILY = {
    "p1_step_in": "step_in",
    "p1_crossing": "crossing",
    "p1_head_on": "head_on",
    "p5_braking": None,
    "p5_turning": None,
    "p1p5_step_in_carrying": "step_in",
}
#: 갈래별 이산 의도(상태 choice의 선택지).
INTENTS = {
    "step_in": ("stay", "cross", "back"),
    "crossing": ("cross", "yield", "hurry"),
    "head_on": ("keep", "aside_left", "aside_right", "stop"),
}
#: 장면 → 그 장면이 정의하는 결정 여유(catalog `margins`의 열쇠). 첫 것이 쌍(사실 하나 바꾸기)의 기본 변수다.
SCENE_MARGINS = {
    "p1_step_in": ("step_in",),
    "p1_crossing": ("crossing",),
    "p1_head_on": ("head_on",),
    "p5_braking": ("spill", "stop"),
    "p5_turning": ("spill",),
    "p1p5_step_in_carrying": ("step_in", "spill"),
}
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
        family = SCENE_FAMILY[scene]
        if family is None:
            return None
        return "arm" if body == "arm" else family


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


@functools.lru_cache(maxsize=1)
def default_config() -> SimgenConfig:
    """저장소의 v0 설정(한 프로세스에서 한 번만 읽는다)."""
    return load_config()


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
    for body, weights in catalog.get("scenes_by_body", {}).items():
        if body not in BODIES:
            errors.append(f"scenes_by_body: unknown body {body}")
        if not _sums_to_one(weights) or not set(weights) <= set(SCENES):
            errors.append(f"scenes_by_body.{body}: weights must be a distribution over known scenes")
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


def _skins_errors(config: SimgenConfig) -> list[str]:
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
        for key in ("settings", "distractors", "obstacles", "occluders", "blockers"):
            if not card.get(key):
                errors.append(f"{where}: {key} is empty")
        places = card.get("places", {})
        for key in ("floor", "turn", "crossing"):
            if not places.get(key):
                errors.append(f"{where}: places.{key} is empty")
        for body, weight in bodies.items():
            if weight <= 0:
                continue
            if not card.get("tasks", {}).get(body):
                errors.append(f"{where}: tasks.{body} is empty")
            carry = card.get("carry_tasks", {}).get(body, [])
            if not carry or not all("{container}" in t for t in carry):
                errors.append(f"{where}: carry_tasks.{body} must be non-empty with {{container}}")
            scenes = catalog.get("scenes_by_body", {}).get(body, {})
            if body == "arm":
                if not places.get("bench") or not card.get("bench_obstacles"):
                    errors.append(f"{where}: arm needs places.bench and bench_obstacles")
            elif "p1_head_on" in scenes and not places.get("corridor"):
                errors.append(f"{where}: places.corridor is empty")
            for scene in scenes:
                family = config.agent_family(scene, body)
                if family is not None:
                    errors.extend(_family_errors(config, domain, family))
            if any(s in scenes for s in _P5_SCENES):
                fits = [c for c in card.get("containers", []) if body in catalog["containers"].get(c.get("kind"), {}).get("bodies", [])]
                if not fits:
                    errors.append(f"{where}: no container a {body} can carry")
        for entry in card.get("containers", []):
            if entry.get("kind") not in catalog.get("containers", {}) or entry.get("content") not in CONTENTS or not entry.get("phrases"):
                errors.append(f"{where}: bad container entry {entry}")
    return sorted(set(errors))


def _family_errors(config: SimgenConfig, domain: str, family: str) -> list[str]:
    card = config.skins["domains"][domain]
    weights = card.get("agents", {}).get(family)
    where = f"domain {domain} agents.{family}"
    if not weights or not _sums_to_one(weights):
        return [f"{where}: must be a distribution"]
    errors = []
    for kind, weight in weights.items():
        spec = config.catalog["agents"].get(kind)
        if spec is None:
            errors.append(f"{where}: unknown agent {kind}")
            continue
        if (family == "arm") != (spec["group"] == "hand"):
            errors.append(f"{where}: {kind} does not fit (arm ↔ hand only)")
        if family in ("crossing", "head_on", "step_in") and spec["group"] != "ball" and family not in spec.get("intent_prior", {}):
            errors.append(f"{where}: {kind} has no intent prior for {family}")
        descriptors = card.get("descriptors", {}).get(kind, [])
        if weight > 0 and (not descriptors or not all(d.get("np") and d.get("ref") for d in descriptors)):
            errors.append(f"{where}: descriptors.{kind} missing np/ref")
    return errors


def _values_errors(values: dict) -> list[str]:
    errors = []
    costs = values.get("costs", {})
    if values.get("tau", 0) <= 0:
        errors.append("values: tau must be positive")
    for name, override in values.get("criteria", {}).items():
        if not set(override) <= set(costs):
            errors.append(f"values: criterion {name} overrides unknown costs")
    return errors


def config_errors(config: SimgenConfig) -> list[str]:
    """설정의 구조 오류 목록(비면 통과)."""
    return _catalog_errors(config.catalog) + _skins_errors(config) + _values_errors(config.values)
