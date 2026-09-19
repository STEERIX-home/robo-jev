"""E0/E1 장면과 일정 — "어수선한 다물체 정리" (docs/02 §1, docs/05 §2).

장면(물체·속성·목표 영역), 지시 변경 일정, 외란 일정을 **reset seed 하나**에서 만든다.
일정은 전부 **모의 시간**으로 적히고 정책 호출 횟수와 무관하다(docs/05 §2). 시각은
제어 주기의 배수로 양자화해 주기 경계가 어디에 놓이든 같은 틱에 적용되게 한다.

`ScenePlan`은 값(불변 dataclass)이고 MuJoCo를 모른다. `TidyClutter`가 그 값을
robosuite 모델로 옮긴다. 이렇게 나눠야 일정 생성을 물리 없이 검사할 수 있다.
"""

from __future__ import annotations

import copy
import math
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

__all__ = [
    "Disturbance",
    "Instruction",
    "SceneObject",
    "ScenePlan",
    "Zone",
    "build_plan",
    "merge_profile",
]

_ATTRIBUTES = ("fragile", "forbidden")

#: 일정 시각 하나를 잡는 데 쓰는 최대 시도 횟수. 넘으면 설정이 모순이다.
_SCHEDULE_ATTEMPTS = 200


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
    instructions = _sample_instructions(settings, rng, objects, zones, grid_ms)
    disturbances = _sample_disturbances(settings, rng, objects, grid_ms)
    return ScenePlan(
        seed=int(seed),
        profile=profile,
        objects=objects,
        zones=zones,
        instructions=instructions,
        disturbances=disturbances,
    )


def _sample_objects(settings: dict[str, Any], rng: np.random.Generator) -> tuple[SceneObject, ...]:
    spec = settings["objects"]
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
) -> tuple[Instruction, ...]:
    spec = settings["instruction"]
    labels = dict(settings["objects"]["shape_labels"])
    plain = [obj for obj in objects if not obj.attributes]
    fragile = next((obj for obj in objects if "fragile" in obj.attributes), None)
    if not plain:
        raise RuntimeError("지시를 만들 수 있는 평범한 물체가 없다 — 속성 개수를 줄여라")

    first = plain[int(rng.integers(len(plain)))]
    zone = zones[int(rng.integers(len(zones)))]
    # 지시의 대상이 이미 목표 영역 안에 놓여 있으면 에피소드가 틱 0에 끝난다(E1 seed 400100). 난수 소비는 그대로
    # 두고 **위반한 조합만** 바꾼다 — 대상을 담지 않는 영역이 있으면 그것을, 없으면 그 영역 밖의 다른 대상을.
    fixed = _target_outside_zone(first, zone, plain, zones)
    if fixed is None:
        raise RuntimeError(
            f"지시의 대상 {first.id}가 모든 영역 안에 있고 영역 {zone.id} 밖의 평범한 물체도 없다 — 장면 설정을 확인하라"
        )
    first, zone = fixed
    text = spec["v1_template"].format(
        color=first.colour_ko,
        shape=labels[first.shape],
        zone=zone.desc,
        fragile=fragile.describe(labels) if fragile else "취약한 물체",
    )
    protected = (fragile.id,) if fragile else ()
    steps = [
        Instruction(version=1, sim_ms=0, text=text, target=first.id, zone=zone.id, protected=protected)
    ]

    if not spec.get("enabled", False):
        return tuple(steps)

    others = [obj for obj in plain if obj.id != first.id]
    if not others:
        return tuple(steps)
    second = others[int(rng.integers(len(others)))]
    # v2의 대상도 같은 영역 밖이어야 한다(영역은 v1의 것으로 고정이므로 대상만 바꾼다). 영역 밖의 다른 대상이
    # 없으면 지시 변경은 없다 — 틱 0에 끝난 두 번째 목표를 만들지 않는다.
    fixed = _target_outside_zone(second, zone, others, (zone,))
    if fixed is None:
        return tuple(steps)
    second, _ = fixed
    low, high = spec["change_window_ms"]
    at_ms = _quantise(float(rng.uniform(low, high)), grid_ms)
    at_ms = min(max(at_ms, _quantise(low, grid_ms) + grid_ms), _quantise(high, grid_ms))
    steps.append(
        Instruction(
            version=2,
            sim_ms=at_ms,
            text=spec["v2_template"].format(
                color=first.colour_ko,
                shape=labels[first.shape],
                color2=second.colour_ko,
                shape2=labels[second.shape],
                zone=zone.desc,
            ),
            target=second.id,
            zone=zone.id,
            # v2는 제약을 다시 말하지 않지만 v1의 보호 물체는 그대로다 — 지시는 덧붙는다 (docs/08 §3.1).
            protected=protected,
        )
    )
    return tuple(steps)


def _inside_zone(obj: SceneObject, zone: Zone) -> bool:
    x0, y0, x1, y1 = [float(value) for value in zone.bounds_mm]
    x, y = float(obj.pos_mm[0]), float(obj.pos_mm[1])
    return min(x0, x1) <= x <= max(x0, x1) and min(y0, y1) <= y <= max(y0, y1)


def _target_outside_zone(
    target: SceneObject, zone: Zone, targets: list[SceneObject], zones: tuple[Zone, ...]
) -> tuple[SceneObject, Zone] | None:
    """지시의 (대상, 영역) 조합에서 대상이 영역 안에 놓인 것을 고친다 — 생성 제약(계약 v0.3 이월 항목).

    조합이 이미 괜찮으면 그대로다. 아니면 먼저 대상을 담지 않는 다른 영역(설정 순서의 첫 것), 그것도 없으면 그
    영역 밖의 다른 대상(id 순서의 첫 것)을 고른다. 둘 다 없으면 `None`. 난수를 더 쓰지 않으므로 위반이 없는
    seed의 장면·일정은 그대로다.
    """
    if not _inside_zone(target, zone):
        return target, zone
    for other in zones:
        if not _inside_zone(target, other):
            return target, other
    for candidate in sorted(targets, key=lambda obj: obj.id):
        if not _inside_zone(candidate, zone):
            return candidate, zone
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
