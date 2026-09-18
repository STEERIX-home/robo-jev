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
from robosuite.environments.manipulation.manipulation_env import ManipulationEnv
from robosuite.models.arenas import TableArena
from robosuite.models.objects import BoxObject, CylinderObject
from robosuite.models.tasks import ManipulationTask

__all__ = ["Disturbance", "Instruction", "SceneObject", "ScenePlan", "TidyClutter", "Zone", "build_plan"]

_ATTRIBUTES = ("fragile", "forbidden")


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
    version: int
    sim_ms: int
    text: str


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
            instructions=tuple(Instruction(**step) for step in data["instructions"]),
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


def _quantise(value: float, period_ms: int) -> int:
    """제어 주기의 배수로 내린다. 주기 경계와 일정이 항상 맞물리게 한다."""
    return int(value // period_ms) * period_ms


def build_plan(config: dict[str, Any], seed: int, profile: str) -> ScenePlan:
    """seed 하나에서 장면·지시 일정·외란 일정을 만든다.

    호출 순서가 곧 난수 소비 순서이므로 순서를 바꾸면 같은 seed의 장면이 달라진다.
    """
    settings = merge_profile(config, profile)
    rng = np.random.default_rng(seed)
    period_ms = 1000 // int(settings["simulator"]["control_hz"])

    objects = _sample_objects(settings, rng)
    zones = _sample_zones(settings, rng)
    instructions = _sample_instructions(settings, rng, objects, zones, period_ms)
    disturbances = _sample_disturbances(settings, rng, objects, period_ms)
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

    return _assign_attributes(objects, spec, rng)


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
        for _ in range(count):
            if cursor >= len(order):
                break
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
    period_ms: int,
) -> tuple[Instruction, ...]:
    spec = settings["instruction"]
    labels = dict(settings["objects"]["shape_labels"])
    plain = [obj for obj in objects if not obj.attributes]
    fragile = next((obj for obj in objects if "fragile" in obj.attributes), None)
    if not plain:
        raise RuntimeError("지시를 만들 수 있는 평범한 물체가 없다 — 속성 개수를 줄여라")

    first = plain[int(rng.integers(len(plain)))]
    zone = zones[int(rng.integers(len(zones)))]
    text = spec["v1_template"].format(
        color=first.colour_ko,
        shape=labels[first.shape],
        zone=zone.desc,
        fragile=fragile.describe(labels) if fragile else "취약한 물체",
    )
    steps = [Instruction(version=1, sim_ms=0, text=text)]

    if not spec.get("enabled", False):
        return tuple(steps)

    others = [obj for obj in plain if obj.id != first.id]
    if not others:
        return tuple(steps)
    second = others[int(rng.integers(len(others)))]
    low, high = spec["change_window_ms"]
    at_ms = _quantise(float(rng.uniform(low, high)), period_ms)
    at_ms = min(max(at_ms, _quantise(low, period_ms) + period_ms), _quantise(high, period_ms))
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
        )
    )
    return tuple(steps)


def _sample_disturbances(
    settings: dict[str, Any],
    rng: np.random.Generator,
    objects: tuple[SceneObject, ...],
    period_ms: int,
) -> tuple[Disturbance, ...]:
    spec = settings["disturbance"]
    low, high = spec["count"]
    count = int(rng.integers(low, high + 1))
    if count == 0:
        return ()

    window_low, window_high = spec["window_ms"]
    gap = int(spec["min_gap_ms"])
    times: list[int] = []
    for _ in range(count):
        for _ in range(200):
            candidate = _quantise(float(rng.uniform(window_low, window_high)), period_ms)
            candidate = max(candidate, period_ms)
            if all(abs(candidate - other) >= gap for other in times):
                times.append(candidate)
                break
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


# --------------------------------------------------------------------------
# robosuite 환경
# --------------------------------------------------------------------------


class TidyClutter(ManipulationEnv):
    """`ScenePlan`을 그대로 세우는 단일 팔 정리 장면.

    robosuite의 placement sampler를 쓰지 않는다. 자세는 이미 `ScenePlan`이 seed에서
    정했고, 여기서 다시 뽑으면 난수 소비가 두 군데로 갈라져 재현이 흐려진다.
    """

    def __init__(self, plan: ScenePlan, settings: dict[str, Any], **kwargs: Any) -> None:
        self.plan = plan
        self.settings = settings
        table = settings["table"]
        self.table_full_size = tuple(float(value) for value in table["full_size_m"])
        self.table_friction = tuple(float(value) for value in table["friction"])
        self.table_offset = np.array([float(value) for value in table["offset_m"]])
        self.scene_objects: list[Any] = []
        super().__init__(**kwargs)

    def reward(self, action: Any = None) -> float:
        """성공 판정은 하네스(3b)가 목표·영역으로 한다. 환경은 보상을 만들지 않는다."""
        return 0.0

    def _load_model(self) -> None:
        super()._load_model()
        xpos = self.robots[0].robot_model.base_xpos_offset["table"](self.table_full_size[0])
        self.robots[0].robot_model.set_base_xpos(xpos)

        arena = TableArena(
            table_full_size=self.table_full_size,
            table_friction=self.table_friction,
            table_offset=self.table_offset,
        )
        arena.set_origin([0, 0, 0])

        spec = self.settings["objects"]
        self.scene_objects = [self._build_object(obj, spec) for obj in self.plan.objects]
        self.model = ManipulationTask(
            mujoco_arena=arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=self.scene_objects,
        )

    @staticmethod
    def _build_object(obj: SceneObject, spec: dict[str, Any]) -> Any:
        density = float(spec["density"])
        friction = [float(value) for value in spec["friction"]]
        rgba = list(obj.rgba)
        if obj.shape == "box":
            size = [value / 1000.0 for value in obj.half_size_mm]
            return BoxObject(
                name=obj.id, size=size, rgba=rgba, density=density, friction=friction
            )
        size = [obj.half_size_mm[0] / 1000.0, obj.half_size_mm[2] / 1000.0]
        return CylinderObject(
            name=obj.id, size=size, rgba=rgba, density=density, friction=friction
        )

    def _setup_references(self) -> None:
        super()._setup_references()
        self.object_body_ids = {
            plan_object.id: self.sim.model.body_name2id(model.root_body)
            for plan_object, model in zip(self.plan.objects, self.scene_objects)
        }
        self.object_joints = {
            plan_object.id: model.joints[0]
            for plan_object, model in zip(self.plan.objects, self.scene_objects)
        }
        self.object_geom_names = {
            plan_object.id: list(model.contact_geoms)
            for plan_object, model in zip(self.plan.objects, self.scene_objects)
        }

    def _reset_internal(self) -> None:
        super()._reset_internal()
        if self.deterministic_reset:
            return
        for plan_object, model in zip(self.plan.objects, self.scene_objects):
            position = [
                self.table_offset[0] + plan_object.pos_mm[0] / 1000.0,
                self.table_offset[1] + plan_object.pos_mm[1] / 1000.0,
                self.table_offset[2] + plan_object.pos_mm[2] / 1000.0,
            ]
            half = math.radians(plan_object.yaw_deg) / 2.0
            quat = [math.cos(half), 0.0, 0.0, math.sin(half)]  # MuJoCo는 wxyz
            self.sim.data.set_joint_qpos(model.joints[0], np.array(position + quat))
