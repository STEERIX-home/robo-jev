"""`ScenePlan`을 세우는 robosuite 환경.

`scene.py`(순수 값)와 나눠 둔다. 일정·장면 생성은 MuJoCo 없이 검사할 수 있어야 하고,
컨트롤러 계약 검사도 robosuite를 import하지 않고 돌아야 하기 때문이다.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from robosuite.environments.manipulation.manipulation_env import ManipulationEnv
from robosuite.models.arenas import TableArena
from robosuite.models.objects import BoxObject, CylinderObject
from robosuite.models.tasks import ManipulationTask

from robo_jev.sim.scene import SceneObject, ScenePlan

__all__ = ["TidyClutter"]




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
