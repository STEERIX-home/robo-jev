"""E0/E1 장면과 일정 — "어수선한 다물체 정리" (docs/02 §1, docs/05 §2).

장면(물체·속성·목표 영역), 지시 변경 일정, 외란 일정을 **reset seed 하나**에서 만든다.
일정은 전부 **모의 시간**으로 적히고 정책 호출 횟수와 무관하다(docs/05 §2). 시각은
제어 주기의 배수로 양자화해 주기 경계가 어디에 놓이든 같은 틱에 적용되게 한다.

`ScenePlan`은 값(불변 dataclass)이고 MuJoCo를 모른다. `TidyClutter`가 그 값을
robosuite 모델로 옮긴다. 이렇게 나눠야 일정 생성을 물리 없이 검사할 수 있다.
"""

from __future__ import annotations

import copy
import hashlib
import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass, replace
from typing import Any

import numpy as np

__all__ = [
    "Disturbance",
    "Instruction",
    "SceneObject",
    "ScenePlan",
    "Zone",
    "build_plan",
    "family_id",
    "family_signature",
    "merge_profile",
    "origin_group",
]

_ATTRIBUTES = ("fragile", "forbidden")

#: 일정 시각 하나를 잡는 데 쓰는 최대 시도 횟수. 넘으면 설정이 모순이다.
_SCHEDULE_ATTEMPTS = 200
#: 물체 배치 전체를 다시 뽑는 최대 횟수 — 금지 물체를 이웃에서 떼어 놓을 자리가 없는 배치(E1 seed 243)는 그 배치를 버리고
#: 같은 난수 스트림으로 다음 배치를 뽑는다. 처음 배치가 성립한 seed는 난수를 더 쓰지 않으므로 그대로다.
_LAYOUT_ATTEMPTS = 8
#: 목표 영역 재가중(기각 표본)의 최대 제안 횟수. 채택 확률이 min(w)/max(w) 이상이라 실제로는 몇 번 안에 끝난다.
_REWEIGHT_ATTEMPTS = 64
#: 영역 id → 계열 이름의 글자 (`zoneL` → `L`).
_ZONE_LETTER_PREFIX = "zone"


@dataclass(frozen=True)
class SceneObject:
    id: str
    shape: str
    colour: str
    colour_ko: str
    rgba: tuple[float, float, float, float]
    half_size_mm: tuple[int, int, int]  # box: 반변 / cylinder: (r, r, 반높이)
    pos_mm: tuple[int, int, int]  # 테이블 중심 기준, z는 테이블 윗면 기준
    yaw_deg: float
    attributes: tuple[str, ...]

    @property
    def obb_mm(self) -> tuple[int, int, int]:
        return tuple(2 * value for value in self.half_size_mm)  # type: ignore[return-value]

    def describe(self, shape_labels: dict[str, str]) -> str:
        return f"{self.colour_ko} {shape_labels[self.shape]}"


@dataclass(frozen=True)
class Zone:
    id: str
    desc: str
    bounds_mm: tuple[int, int, int, int]


@dataclass(frozen=True)
class Instruction:
    """지시 하나. 텍스트와 함께 **구조화된 목표**를 든다 (docs/08 §3.2 `goal`).

    `target`·`zone`은 지시가 가리키는 물체·영역의 id, `protected`는 지시가 "건드리지 마라"로
    부른 물체의 id다. 계획은 지시를 만들 때 이미 알고 있으므로 텍스트를 다시 파싱하지
    않는다. 세 필드는 선택이다 — 없는 옛 snapshot도 읽힌다.
    """

    version: int
    sim_ms: int
    text: str
    target: str | None = None
    zone: str | None = None
    protected: tuple[str, ...] = ()
    #: 문구 템플릿 변형 id (`v1#<n>`·`v2#<n>`, 설정 `instruction.v1_templates`의 번호) — provenance와 템플릿 holdout의 근거.
    template: str | None = None


@dataclass(frozen=True)
class Disturbance:
    sim_ms: int
    object: str
    delta_mm: tuple[int, int]
    delta_yaw_deg: float


@dataclass(frozen=True)
class ScenePlan:
    """한 에피소드의 장면과 일정 전체. seed에서만 나오고 그 뒤로 바뀌지 않는다."""

    seed: int
    profile: str
    objects: tuple[SceneObject, ...]
    zones: tuple[Zone, ...]
    instructions: tuple[Instruction, ...]
    disturbances: tuple[Disturbance, ...]

    def model_signature(self) -> tuple:
        """MuJoCo 모델을 다시 지어야 하는지 가르는 부분만 뽑는다."""
        return tuple(
            (obj.id, obj.shape, obj.half_size_mm, obj.rgba) for obj in self.objects
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "seed": self.seed,
            "profile": self.profile,
            "objects": [asdict(obj) for obj in self.objects],
            "zones": [asdict(zone) for zone in self.zones],
            "instructions": [asdict(step) for step in self.instructions],
            "disturbances": [asdict(item) for item in self.disturbances],
        }

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> ScenePlan:
        def tuples(item: dict[str, Any], *fields: str) -> dict[str, Any]:
            out = dict(item)
            for field in fields:
                out[field] = tuple(out[field])
            return out

        return cls(
            seed=int(data["seed"]),
            profile=str(data["profile"]),
            objects=tuple(
                SceneObject(**tuples(obj, "rgba", "half_size_mm", "pos_mm", "attributes"))
                for obj in data["objects"]
            ),
            zones=tuple(Zone(**tuples(zone, "bounds_mm")) for zone in data["zones"]),
            instructions=tuple(
                Instruction(**tuples(step, "protected")) if "protected" in step else Instruction(**step)
                for step in data["instructions"]
            ),
            disturbances=tuple(
                Disturbance(**tuples(item, "delta_mm")) for item in data["disturbances"]
            ),
        )


# --------------------------------------------------------------------------
# 설정 병합
# --------------------------------------------------------------------------


def merge_profile(config: dict[str, Any], profile: str) -> dict[str, Any]:
    """`profiles.<name>`의 덮어쓰기를 기본 설정에 겹친다 (깊은 병합)."""
    merged = copy.deepcopy(config)
    overrides = (config.get("profiles") or {}).get(profile)
    if overrides is None:
        raise KeyError(f"설정에 없는 프로파일이다: {profile!r}")

    def overlay(base: dict[str, Any], patch: dict[str, Any]) -> None:
        for key, value in patch.items():
            if isinstance(value, dict) and isinstance(base.get(key), dict):
                overlay(base[key], value)
            else:
                base[key] = copy.deepcopy(value)

    overlay(merged, overrides)
    merged.pop("profiles", None)
    return merged


# --------------------------------------------------------------------------
# 장면 생성
# --------------------------------------------------------------------------


def _quantise(value: float, grid_ms: int) -> int:
    """고정 격자의 배수로 내린다.

    격자는 **제어 주기와 무관**하다. 주기로 내리면 제어 주기를 바꾼 순간 같은 seed의
    일정까지 달라져서, "외란은 모의 시간으로 정해진다"가 주기마다 다른 말이 된다.
    """
    return int(value // grid_ms) * grid_ms


def build_plan(config: dict[str, Any], seed: int, profile: str) -> ScenePlan:
    """seed 하나에서 장면·지시 일정·외란 일정을 만든다.

    호출 순서가 곧 난수 소비 순서이므로 순서를 바꾸면 같은 seed의 장면이 달라진다.
    """
    settings = merge_profile(config, profile)
    rng = np.random.default_rng(seed)
    grid_ms = int(settings["episode"]["schedule_grid_ms"])

    objects = _sample_objects(settings, rng)
    zones = _sample_zones(settings, rng)
    instructions, objects = _sample_instructions(settings, rng, objects, zones, grid_ms, seed=int(seed))
    disturbances = _sample_disturbances(settings, rng, objects, grid_ms)
    plan = ScenePlan(
        seed=int(seed),
        profile=profile,
        objects=objects,
        zones=zones,
        instructions=instructions,
        disturbances=disturbances,
    )
    # 문구 템플릿 변형은 난수가 아니라 **origin group의 해시**로 고른다 — 같은 group(프로파일·장면 계열·목표 영역)의 에피소드는
    # 같은 변형이라 템플릿 holdout이 한 group을 두 split에 걸치게 하지 않는다(docs/04 §5). 장면·일정의 난수 소비와 무관하다.
    return replace(plan, instructions=_phrase_instructions(settings, instructions, objects, zones, group=origin_group(profile, plan)))


# --------------------------------------------------------------------------
# 보조 결정 — 장면·일정의 난수 스트림을 건드리지 않는 결정적 선택
# --------------------------------------------------------------------------


def _hash_unit(*parts: object) -> float:
    """조각들의 sha256 앞 8바이트 → [0, 1). 장면·일정의 난수 스트림(`rng`)을 바꾸면 안 되는 선택(문구 변형, 목표 영역
    재가중, 대체 영역)에 쓴다 — 같은 입력이면 같은 값이고, 도입 전 seed의 장면·일정은 그대로다."""
    digest = hashlib.sha256("|".join(str(part) for part in parts).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


def _weighted_index(point: float, weights: Sequence[float]) -> int:
    """[0, 1)의 점을 비중 구간에 떨어뜨린다 (비중은 상대값)."""
    total = float(sum(weights))
    if total <= 0:
        raise ValueError(f"비중의 합이 양수여야 한다: {list(weights)}")
    edge = 0.0
    for index, weight in enumerate(weights):
        edge += float(weight) / total
        if point < edge:
            return index
    return len(weights) - 1


def _zone_weights(spec: dict[str, Any], zones: Sequence[Zone]) -> list[float] | None:
    """설정 `instruction.goal_zone_weights`를 이 장면의 영역 순서로. 없으면 None(균등). 표에 없는 영역은 설정의 모순이다."""
    table = spec.get("goal_zone_weights")
    if not table:
        return None
    missing = [zone.id for zone in zones if zone.id not in table]
    if missing:
        raise ValueError(f"instruction.goal_zone_weights에 영역이 없다: {missing} (있는 것: {sorted(table)})")
    return [float(table[zone.id]) for zone in zones]


def _reweight_zone(first: int, zones: Sequence[Zone], weights: list[float] | None, seed: int) -> Zone:
    """주 난수의 **균등** 추첨(`first`)을 목표 영역 비중으로 다시 가중한다 — 기각 표본: 제안을 확률 w/max(w)로 채택하고,
    기각하면 seed 해시로 균등 제안을 다시 낸다. 결과의 분포는 비중에 비례하고, 주 난수 스트림은 그대로다: 비중이 최대인
    영역(zoneL/zoneR)을 뽑은 seed의 계획은 비중 도입 전과 같고, zoneF를 뽑은 seed만 일부가 L/R로 옮겨진다."""
    if weights is None:
        return zones[first]
    top = max(weights)
    index = first
    for attempt in range(_REWEIGHT_ATTEMPTS):
        if _hash_unit(seed, "goal-zone", attempt) * top < weights[index]:
            return zones[index]
        index = _weighted_index(_hash_unit(seed, "goal-zone-proposal", attempt), [1.0] * len(zones))
    return zones[weights.index(top)]


def family_signature(plan: ScenePlan) -> dict[str, Any]:
    """장면 계획의 구조 서명: 물체 수, 형상 다중집합, 영역 집합.

    자세·색·속성 배정·일정은 넣지 않는다 — 같은 구조의 장면이 같은 계열이어야 표현만 다른 장면이
    train/test로 갈라지지 않는다(docs/04 §5 "holdout이 단지 물체 이름과 후보 ID를 바꾼 버전인지").
    """
    shapes: dict[str, int] = {}
    for obj in plan.objects:
        shapes[obj.shape] = shapes.get(obj.shape, 0) + 1
    return {
        "objects": len(plan.objects),
        "shapes": dict(sorted(shapes.items())),
        "zones": sorted(zone.id for zone in plan.zones),
    }


def family_id(plan: ScenePlan) -> str:
    """구조 서명 → 계열 id. 읽을 수 있고 결정적이다: `family-n8-b3c5-zLRF`
    (물체 8개, 상자 3·원통 5, 영역 L·R·F)."""
    signature = family_signature(plan)
    boxes = int(signature["shapes"].get("box", 0))
    cylinders = int(signature["shapes"].get("cylinder", 0))
    letters = "".join(
        zone[len(_ZONE_LETTER_PREFIX):] if zone.startswith(_ZONE_LETTER_PREFIX) else zone
        for zone in signature["zones"]
    )
    return f"family-n{signature['objects']}-b{boxes}c{cylinders}-z{letters}"


def origin_group(profile: str, plan: ScenePlan) -> str:
    """`robot/<profile>/<계열>/goal-<목표 영역>` — 장면 계열에 목표 생성 계열(지시의 목표 영역)을 붙인다 (docs/04 §5
    "목표 생성 계열의 계보"). split의 단위이자 문구 변형의 열쇠라, 영역별 holdout도 템플릿 holdout도 한 group을 두 split에
    걸치게 하지 않는다."""
    zone = plan.instructions[0].zone if plan.instructions and plan.instructions[0].zone else "none"
    return f"robot/{profile}/{family_id(plan)}/goal-{zone}"


def _sample_objects(settings: dict[str, Any], rng: np.random.Generator) -> tuple[SceneObject, ...]:
    """물체 배치. 금지 물체를 이웃에서 떼어 놓을 자리가 없는 배치는 버리고 같은 스트림으로 다시 뽑는다(`_LAYOUT_ATTEMPTS`번) —
    처음 배치가 성립한 seed는 난수를 더 쓰지 않는다. 그래도 안 되면 설정의 모순이라 멈춘다."""
    spec = settings["objects"]
    for attempt in range(_LAYOUT_ATTEMPTS):
        try:
            return _sample_layout(spec, rng)
        except RuntimeError as error:
            if attempt == _LAYOUT_ATTEMPTS - 1:
                raise RuntimeError(f"{error} (배치를 {_LAYOUT_ATTEMPTS}번 다시 뽑아도 같다)") from error
    raise AssertionError("unreachable")


def _sample_layout(spec: dict[str, Any], rng: np.random.Generator) -> tuple[SceneObject, ...]:
    count = int(rng.integers(spec["count_min"], spec["count_max"] + 1))
    palette = list(spec["palette"])
    colour_indices = rng.permutation(len(palette))[:count]

    x_low, x_high = spec["spawn_x_mm"]
    y_low, y_high = spec["spawn_y_mm"]
    separation = float(spec["min_separation_mm"])
    attempts = int(spec["spawn_attempts"])

    placed: list[tuple[int, int]] = []
    objects: list[SceneObject] = []
    for index in range(count):
        shape = str(spec["shapes"][int(rng.integers(len(spec["shapes"])))])
        if shape == "box":
            half = tuple(
                int(rng.integers(spec["box_half_size_mm"][0], spec["box_half_size_mm"][1] + 1))
                for _ in range(3)
            )
        else:
            radius = int(
                rng.integers(spec["cylinder_radius_mm"][0], spec["cylinder_radius_mm"][1] + 1)
            )
            half_height = int(
                rng.integers(
                    spec["cylinder_half_height_mm"][0], spec["cylinder_half_height_mm"][1] + 1
                )
            )
            half = (radius, radius, half_height)

        position = None
        for _ in range(attempts):
            candidate = (int(rng.integers(x_low, x_high + 1)), int(rng.integers(y_low, y_high + 1)))
            if all(math.dist(candidate, other) >= separation for other in placed):
                position = candidate
                break
        if position is None:
            raise RuntimeError(
                f"물체 {index}를 {attempts}번 안에 놓지 못했다 — 간격·범위 설정을 확인하라"
            )
        placed.append(position)

        colour = palette[int(colour_indices[index])]
        objects.append(
            SceneObject(
                id=f"o{index}",
                shape=shape,
                colour=str(colour["name"]),
                colour_ko=str(colour["ko"]),
                rgba=tuple(float(value) for value in colour["rgba"]),  # type: ignore[arg-type]
                half_size_mm=half,  # type: ignore[arg-type]
                pos_mm=(position[0], position[1], half[2]),
                yaw_deg=float(rng.uniform(-180.0, 180.0)),
                attributes=(),
            )
        )

    with_attributes = _assign_attributes(objects, spec, rng)
    return _enforce_forbidden_separation(with_attributes, spec, rng, separation)


def _forbidden_margins_mm(spec: dict[str, Any]) -> tuple[float, float] | None:
    """(planner.margin_mm, planner.forbidden_margin_mm) — 하네스 설정에서 읽는다. 설정이 없으면 규칙도 없다."""
    section = spec.get("forbidden_separation")
    if not section:
        return None
    import yaml

    from robo_jev.sim.controller import resolve_config_path

    planner = yaml.safe_load(
        resolve_config_path(section["harness_config"]).read_text(encoding="utf-8")
    )["planner"]
    return float(planner["margin_mm"]), float(planner["forbidden_margin_mm"])


def _required_separation_mm(
    a: SceneObject, b: SceneObject, base: float, margins: tuple[float, float] | None
) -> float:
    """두 물체의 최소 중심 간격(xy). 금지 접촉 물체가 끼면 하네스가 그 물체를 보는 장애물 반지름이다:
    외접 반지름 + planner.margin_mm + planner.forbidden_margin_mm (robot.py `_first_blocker`)."""
    required = base
    if margins is None:
        return required
    margin, forbidden_margin = margins
    for member in (a, b):
        if "forbidden" in member.attributes:
            circumradius = math.dist((0.0, 0.0, 0.0), member.half_size_mm)
            required = max(required, circumradius + margin + forbidden_margin)
    return required


def _enforce_forbidden_separation(
    objects: tuple[SceneObject, ...],
    spec: dict[str, Any],
    rng: np.random.Generator,
    base: float,
) -> tuple[SceneObject, ...]:
    """금지 접촉 물체가 낀 쌍의 간격을 보장한다 (3b 리뷰 (a)).

    속성은 자세 뒤에 정해지므로 여기서 **위반한 금지 물체만** 다시 놓는다. 위반이 없는 seed는 난수를
    더 쓰지 않아 장면·일정이 그대로다 — 계보를 지키려는 선택이다. 놓을 자리가 없으면 설정의 모순이므로
    그 자리에서 멈춘다(조용히 좁은 간격을 두지 않는다).
    """
    margins = _forbidden_margins_mm(spec)
    if margins is None:
        return objects
    placed = list(objects)
    x_low, x_high = spec["spawn_x_mm"]
    y_low, y_high = spec["spawn_y_mm"]
    attempts = int(spec["spawn_attempts"])

    def far_enough(candidate: SceneObject, index: int) -> bool:
        return all(
            math.dist(candidate.pos_mm[:2], other.pos_mm[:2])
            >= _required_separation_mm(candidate, other, base, margins)
            for position, other in enumerate(placed)
            if position != index
        )

    for index, obj in enumerate(placed):
        if "forbidden" not in obj.attributes or far_enough(obj, index):
            continue
        for _ in range(attempts):
            x, y = int(rng.integers(x_low, x_high + 1)), int(rng.integers(y_low, y_high + 1))
            moved = SceneObject(**{**asdict(obj), "pos_mm": (x, y, obj.pos_mm[2])})
            if far_enough(moved, index):
                placed[index] = moved
                break
        else:
            raise RuntimeError(
                f"금지 물체 {obj.id}를 이웃에서 {attempts}번 안에 떼어 놓지 못했다 — "
                "간격·범위 설정과 하네스의 planner 여유를 확인하라"
            )
    return tuple(placed)


def _assign_attributes(
    objects: list[SceneObject], spec: dict[str, Any], rng: np.random.Generator
) -> tuple[SceneObject, ...]:
    """취약·금지 속성을 서로 겹치지 않게 붙인다 (docs/02 §1)."""
    order = list(rng.permutation(len(objects)))
    attributes: dict[int, tuple[str, ...]] = {}
    cursor = 0
    for attribute in _ATTRIBUTES:
        low, high = spec[f"{attribute}_count"]
        count = int(rng.integers(low, high + 1))
        if cursor + count > len(order):
            # 속성은 겹치지 않는다. 물체가 모자라면 장면 설정이 모순이므로 그 자리에서 멈춘다.
            raise RuntimeError(
                f"{attribute} {count}개를 붙일 물체가 없다: 물체 {len(order)}개 중 "
                f"{cursor}개가 이미 쓰였다 — objects.count_min과 *_count를 확인하라"
            )
        for _ in range(count):
            attributes[int(order[cursor])] = (attribute,)
            cursor += 1
    return tuple(
        SceneObject(**{**asdict(obj), "attributes": attributes.get(index, ())})
        for index, obj in enumerate(objects)
    )


def _sample_zones(settings: dict[str, Any], rng: np.random.Generator) -> tuple[Zone, ...]:
    spec = settings["zones"]
    candidates = list(spec["candidates"])
    count = int(rng.integers(spec["count_min"], min(spec["count_max"], len(candidates)) + 1))
    chosen = sorted(int(index) for index in rng.permutation(len(candidates))[:count])
    return tuple(
        Zone(
            id=str(candidates[index]["id"]),
            desc=str(candidates[index]["desc"]),
            bounds_mm=tuple(int(value) for value in candidates[index]["bounds_mm"]),  # type: ignore[arg-type]
        )
        for index in chosen
    )


def _sample_instructions(
    settings: dict[str, Any],
    rng: np.random.Generator,
    objects: tuple[SceneObject, ...],
    zones: tuple[Zone, ...],
    grid_ms: int,
    *,
    seed: int,
) -> tuple[tuple[Instruction, ...], tuple[SceneObject, ...]]:
    """지시 일정과, 대상이 목표 영역 밖으로 옮겨졌을 수 있는 물체 목록을 돌려준다 (:func:`_target_outside_zone`)."""
    spec = settings["instruction"]
    labels = dict(settings["objects"]["shape_labels"])
    plain = [obj for obj in objects if not obj.attributes]
    fragile = next((obj for obj in objects if "fragile" in obj.attributes), None)
    if not plain:
        raise RuntimeError("지시를 만들 수 있는 평범한 물체가 없다 — 속성 개수를 줄여라")

    first = plain[int(rng.integers(len(plain)))]
    # 목표 영역: 주 난수의 균등 추첨을 설정 `goal_zone_weights`(zoneF는 봉인 개념이라 드물게)로 다시 가중한다 — 주 스트림은 그대로.
    weights = _zone_weights(spec, zones)
    zone = _reweight_zone(int(rng.integers(len(zones))), zones, weights, seed)
    # 지시의 대상이 이미 목표 영역 안에 놓여 있으면 에피소드가 틱 0에 끝난다(E1 seed 400100). 난수 소비는 그대로
    # 두고 **위반한 조합만** 바꾼다 — 영역은 두고 그 밖의 다른 대상(seed 해시로)을, 없으면 대상을 영역 밖의 빈자리로
    # 옮기고(E0의 평범한 물체 하나), 그것도 안 되면 대상을 담지 않는 다른 영역(목표 영역 비중대로)을.
    fixed = _target_outside_zone(
        first, zone, plain, zones, weights=weights, unit=_hash_unit(seed, "zone-fix"),
        relocate=lambda obj, area: _relocated_outside_zone(obj, area, objects, settings["objects"], seed),
    )
    if fixed is None:
        raise RuntimeError(
            f"지시의 대상 {first.id}가 모든 영역 안에 있고 영역 {zone.id} 밖의 평범한 물체도 빈자리도 없다 — 장면 설정을 확인하라"
        )
    first, zone = fixed
    if first.id in {obj.id for obj in objects} and first not in objects:
        objects = tuple(first if obj.id == first.id else obj for obj in objects)  # 옮겨진 대상
        plain = [first if obj.id == first.id else obj for obj in plain]
    # 텍스트는 변형 0으로 먼저 채우고 :func:`_phrase_instructions`가 맨 뒤에 변형을 고른다.
    text = _instruction_text(spec, "v1", 0, first, zone, labels, fragile=fragile)
    protected = (fragile.id,) if fragile else ()
    steps = [
        Instruction(version=1, sim_ms=0, text=text, target=first.id, zone=zone.id, protected=protected, template="v1#0")
    ]

    if not spec.get("enabled", False):
        return tuple(steps), objects

    others = [obj for obj in plain if obj.id != first.id]
    if not others:
        return tuple(steps), objects
    second = others[int(rng.integers(len(others)))]
    # v2의 대상도 같은 영역 밖이어야 한다(영역은 v1의 것으로 고정이므로 대상만 바꾼다). 영역 밖의 다른 대상이
    # 없으면 지시 변경은 없다 — 틱 0에 끝난 두 번째 목표를 만들지 않는다.
    fixed = _target_outside_zone(second, zone, others, (zone,), unit=_hash_unit(seed, "target-fix"))
    if fixed is None:
        return tuple(steps), objects
    second, _ = fixed
    low, high = spec["change_window_ms"]
    at_ms = _quantise(float(rng.uniform(low, high)), grid_ms)
    at_ms = min(max(at_ms, _quantise(low, grid_ms) + grid_ms), _quantise(high, grid_ms))
    steps.append(
        Instruction(
            version=2,
            sim_ms=at_ms,
            text=_instruction_text(spec, "v2", 0, first, zone, labels, second=second),
            target=second.id,
            zone=zone.id,
            # v2는 제약을 다시 말하지 않지만 v1의 보호 물체는 그대로다 — 지시는 덧붙는다 (docs/08 §3.1).
            protected=protected,
            template="v2#0",
        )
    )
    return tuple(steps), objects


def _instruction_text(
    spec: dict[str, Any],
    version: str,
    variant: int,
    first: SceneObject,
    zone: Zone,
    labels: dict[str, str],
    *,
    fragile: SceneObject | None = None,
    second: SceneObject | None = None,
) -> str:
    """지시 텍스트 — 설정 `instruction.<version>_templates[variant]`에 슬롯(`{color} {shape} {zone} {fragile}`, v2는
    `{color2} {shape2}`)을 채운다. 변형은 표현만 다르고 구조화된 목표(대상·영역·보호 물체)는 같다."""
    templates = spec[f"{version}_templates"]
    template = str(templates[variant])
    if version == "v1":
        return template.format(
            color=first.colour_ko,
            shape=labels[first.shape],
            zone=zone.desc,
            fragile=fragile.describe(labels) if fragile else "취약한 물체",
        )
    assert second is not None
    return template.format(
        color=first.colour_ko,
        shape=labels[first.shape],
        color2=second.colour_ko,
        shape2=labels[second.shape],
        zone=zone.desc,
    )


def _phrase_instructions(
    settings: dict[str, Any],
    instructions: tuple[Instruction, ...],
    objects: tuple[SceneObject, ...],
    zones: tuple[Zone, ...],
    *,
    group: str,
) -> tuple[Instruction, ...]:
    """지시마다 문구 템플릿 변형을 고른다 (v1·v2 각각 `<version>_templates`에서 하나, `template_weights`의 비중으로).
    구조화된 목표는 그대로다. 변형은 **origin group의 해시**가 정한다 — 같은 group의 에피소드는 같은 변형(v1·v2는 따로)."""
    spec = settings["instruction"]
    labels = dict(settings["objects"]["shape_labels"])
    by_id = {obj.id: obj for obj in objects}
    zone_by_id = {zone.id: zone for zone in zones}
    fragile = next((obj for obj in objects if "fragile" in obj.attributes), None)
    phrased: list[Instruction] = []
    first_target = by_id[instructions[0].target] if instructions and instructions[0].target else None
    for step in instructions:
        version = f"v{step.version}"
        templates = spec.get(f"{version}_templates")
        if not templates or step.target is None or step.zone is None:
            phrased.append(step)
            continue
        weights = spec.get("template_weights")
        if weights and len(weights) != len(templates):
            raise ValueError(f"instruction.template_weights는 {version}_templates와 길이가 같아야 한다")
        variant = _weighted_index(_hash_unit(group, "template", version), weights or [1.0] * len(templates))
        target = by_id[step.target]
        zone = zone_by_id[step.zone]
        if step.version == 1:
            text = _instruction_text(spec, "v1", variant, target, zone, labels, fragile=fragile)
        else:
            text = _instruction_text(spec, "v2", variant, first_target or target, zone, labels, second=target)
        phrased.append(Instruction(**{**asdict(step), "text": text, "template": f"{version}#{variant}"}))
    return tuple(phrased)


def _inside_zone(obj: SceneObject, zone: Zone) -> bool:
    x0, y0, x1, y1 = [float(value) for value in zone.bounds_mm]
    x, y = float(obj.pos_mm[0]), float(obj.pos_mm[1])
    return min(x0, x1) <= x <= max(x0, x1) and min(y0, y1) <= y <= max(y0, y1)


def _target_outside_zone(
    target: SceneObject,
    zone: Zone,
    targets: list[SceneObject],
    zones: tuple[Zone, ...],
    *,
    weights: list[float] | None = None,
    unit: float = 0.0,
    relocate: Any = None,
) -> tuple[SceneObject, Zone] | None:
    """지시의 (대상, 영역) 조합에서 대상이 영역 안에 놓인 것을 고친다 — 생성 제약(계약 v0.3 이월 항목).

    조합이 이미 괜찮으면 그대로다. 아니면 **영역은 두고** (1) 그 영역 밖의 다른 대상 중 하나(`unit`(seed 해시)로 균등하게),
    (2) 없으면 `relocate(대상, 영역)`로 대상을 영역 밖의 빈자리로 옮긴 것(E0의 평범한 물체 하나), (3) 그것도 안 되면 대상을
    담지 않는 다른 영역 중 하나(`unit`으로 `weights`(목표 영역 비중, `zones` 순서)에 따라)를 고른다. 영역을 맨 뒤에 바꾸는
    까닭: 목표 영역은 봉인 개념(zoneF)과 목표 생성 계열의 열쇠라 그 분포를 흔들지 않는다 — 설정 순서의 첫 영역으로 고정하면
    zoneL에, 영역을 먼저 바꾸면 두 영역뿐인 장면에서 zoneF에 목표가 쏠린다. 셋 다 없으면 `None`. 주 난수를 쓰지 않으므로
    위반이 없는 seed의 장면·일정은 그대로다.
    """
    if not _inside_zone(target, zone):
        return target, zone
    others = [candidate for candidate in sorted(targets, key=lambda obj: obj.id) if not _inside_zone(candidate, zone)]
    if others:
        return others[_weighted_index(unit, [1.0] * len(others))], zone
    moved = relocate(target, zone) if relocate is not None else None
    if moved is not None:
        return moved, zone
    outside = [index for index, other in enumerate(zones) if not _inside_zone(target, other)]
    if outside:
        chosen = outside[_weighted_index(unit, [1.0 if weights is None else weights[index] for index in outside])]
        return target, zones[chosen]
    return None


def _relocated_outside_zone(
    target: SceneObject, zone: Zone, objects: tuple[SceneObject, ...], spec: dict[str, Any], seed: int
) -> SceneObject | None:
    """대상을 목표 영역 밖의 빈자리로 옮긴 사본 — 영역 밖의 다른 대상이 없을 때(E0의 평범한 물체 하나). 자리는 seed 해시로
    뽑고(주 난수는 그대로) 배치 규칙(간격·금지 물체 여유·생성 범위)을 지킨다. `spawn_attempts` 안에 자리가 없으면 None."""
    margins = _forbidden_margins_mm(spec)
    base = float(spec["min_separation_mm"])
    x_low, x_high = spec["spawn_x_mm"]
    y_low, y_high = spec["spawn_y_mm"]
    others = [obj for obj in objects if obj.id != target.id]
    for attempt in range(int(spec["spawn_attempts"])):
        x = int(x_low) + int(_hash_unit(seed, "relocate", target.id, attempt, "x") * (int(x_high) - int(x_low) + 1))
        y = int(y_low) + int(_hash_unit(seed, "relocate", target.id, attempt, "y") * (int(y_high) - int(y_low) + 1))
        moved = replace(target, pos_mm=(x, y, target.pos_mm[2]))
        if _inside_zone(moved, zone):
            continue
        if all(
            math.dist(moved.pos_mm[:2], other.pos_mm[:2]) >= _required_separation_mm(moved, other, base, margins)
            for other in others
        ):
            return moved
    return None


def _sample_disturbances(
    settings: dict[str, Any],
    rng: np.random.Generator,
    objects: tuple[SceneObject, ...],
    grid_ms: int,
) -> tuple[Disturbance, ...]:
    spec = settings["disturbance"]
    low, high = spec["count"]
    count = int(rng.integers(low, high + 1))
    if count == 0:
        return ()

    window_low, window_high = spec["window_ms"]
    gap = int(spec["min_gap_ms"])
    times: list[int] = []
    for index in range(count):
        for _ in range(_SCHEDULE_ATTEMPTS):
            candidate = _quantise(float(rng.uniform(window_low, window_high)), grid_ms)
            candidate = max(candidate, grid_ms)
            if all(abs(candidate - other) >= gap for other in times):
                times.append(candidate)
                break
        else:
            # 조용히 적게 만들면 "외란 N개인 장면"이라고 믿은 쪽이 틀린 수를 쓴다.
            raise RuntimeError(
                f"외란 {index + 1}/{count}번째 시각을 {_SCHEDULE_ATTEMPTS}번 안에 잡지 못했다 "
                f"— disturbance.window_ms {[window_low, window_high]}와 min_gap_ms {gap}을 확인하라"
            )
    times.sort()

    return tuple(
        Disturbance(
            sim_ms=at_ms,
            object=objects[int(rng.integers(len(objects)))].id,
            delta_mm=(
                int(rng.integers(spec["delta_x_mm"][0], spec["delta_x_mm"][1] + 1)),
                int(rng.integers(spec["delta_y_mm"][0], spec["delta_y_mm"][1] + 1)),
            ),
            delta_yaw_deg=float(rng.uniform(*spec["delta_yaw_deg"])),
        )
        for at_ms in times
    )
