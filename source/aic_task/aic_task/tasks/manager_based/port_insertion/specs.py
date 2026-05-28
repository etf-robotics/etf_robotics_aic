"""Task assembly choices for the port-insertion environment.

This module chooses reusable asset specs, layout specs, controller assumptions,
and the single NIC port-0 insertion goal. It intentionally does not create
IsaacLab config objects; that conversion belongs in ``builders.py``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

from aic_task.asset_specs import (
    AIC_PORT_INSERTION_LAYOUT,
    NIC_CARD_ASSET,
    ROBOT_ROLE_TCP,
    SCENE_SLOT_ROBOT,
    SCENE_SLOT_TARGET,
    UR5E_CABLE_ASSET,
    RobotAssetSpec,
    SceneLayoutSpec,
    TargetAssetSpec,
    TargetPortSpec,
)


ActionType = Literal["diff_ik"]
ControllerCommandType = Literal["pose"]


@dataclass(frozen=True)
class ControllerSpec:
    """Action/controller assumptions for one robot slot in the task layout."""

    name: str
    action_type: ActionType
    robot_slot: str
    joint_group: str
    controlled_body_role: str
    command_type: ControllerCommandType
    use_relative_mode: bool
    ik_method: str
    ik_params: dict[str, float]
    scale: tuple[float, float, float, float, float, float]


@dataclass(frozen=True)
class InsertionGoalSpec:
    """Episode-level insertion goal consumed by the path planner/oracle."""

    command_name: str
    target_slot: str
    port_name: str
    target_xz_offset: tuple[float, float]
    approach_offset_local: tuple[float, float, float]
    approach_pos_noise_local: tuple[float, float, float]
    approach_tilt_noise_deg: float
    approach_twist_noise_deg: float
    resampling_time_range: tuple[float, float]
    debug_vis: bool = False


@dataclass(frozen=True)
class PortInsertionTerminationSpec:
    """Success/failure thresholds for the port-insertion episode."""

    success_position_threshold: float
    success_orientation_threshold_rad: float
    success_required_seconds: float
    stationary_movement_threshold: float
    stationary_success_position_threshold: float
    stationary_required_seconds: float


@dataclass(frozen=True)
class PortInsertionAssemblySpec:
    """Complete task assembly selected by ``PortInsertionEnvCfg``."""

    name: str
    robot: RobotAssetSpec
    target: TargetAssetSpec
    layout: SceneLayoutSpec
    controller: ControllerSpec
    goal: InsertionGoalSpec
    termination: PortInsertionTerminationSpec

    def selected_port(self) -> TargetPortSpec:
        """Return the target port selected by the insertion goal."""

        return self.target.port(self.goal.port_name)

    def validate(self) -> None:
        """Fail early if the assembly wires incompatible spec pieces."""

        if self.controller.robot_slot != self.layout.robot_slot.name:
            raise ValueError(
                f"Controller '{self.controller.name}' controls slot "
                f"'{self.controller.robot_slot}', but layout robot slot is "
                f"'{self.layout.robot_slot.name}'."
            )
        if self.goal.target_slot != self.layout.target_slot.name:
            raise ValueError(
                f"Goal '{self.goal.command_name}' targets slot "
                f"'{self.goal.target_slot}', but layout target slot is "
                f"'{self.layout.target_slot.name}'."
            )

        self.robot.joint_group(self.controller.joint_group)
        self.robot.body_name_for_role(self.controller.controlled_body_role)
        self.selected_port()


UR5E_DIFF_IK_CONTROLLER = ControllerSpec(
    name="ur5e_diff_ik_tcp",
    action_type="diff_ik",
    robot_slot=SCENE_SLOT_ROBOT,
    joint_group="arm",
    controlled_body_role=ROBOT_ROLE_TCP,
    command_type="pose",
    use_relative_mode=True,
    ik_method="dls",
    ik_params={"lambda_val": 0.01},
    scale=(0.015, 0.015, 0.015, 0.025, 0.025, 0.025),
)

NIC_PORT_0_INSERTION_GOAL = InsertionGoalSpec(
    command_name="insertion_goal",
    target_slot=SCENE_SLOT_TARGET,
    port_name="sfp_port_0",
    target_xz_offset=(0.0, 0.001),
    approach_offset_local=(0.0, -0.07, 0.0),
    approach_pos_noise_local=(0.01, 0.0, 0.01),
    approach_tilt_noise_deg=5.0,
    approach_twist_noise_deg=10.0,
    resampling_time_range=(1.0e9, 1.0e9 + 1.0),
    debug_vis=False,
)

AIC_PORT_INSERTION_TERMINATION = PortInsertionTerminationSpec(
    success_position_threshold=0.003,
    success_orientation_threshold_rad=math.radians(4.0),
    success_required_seconds=0.5,
    stationary_movement_threshold=0.001,
    stationary_success_position_threshold=0.003,
    stationary_required_seconds=1.0,
)

AIC_PORT_INSERTION_ASSEMBLY = PortInsertionAssemblySpec(
    name="aic_port_insertion",
    robot=UR5E_CABLE_ASSET,
    target=NIC_CARD_ASSET,
    layout=AIC_PORT_INSERTION_LAYOUT,
    controller=UR5E_DIFF_IK_CONTROLLER,
    goal=NIC_PORT_0_INSERTION_GOAL,
    termination=AIC_PORT_INSERTION_TERMINATION,
)
AIC_PORT_INSERTION_ASSEMBLY.validate()


__all__ = [
    "AIC_PORT_INSERTION_ASSEMBLY",
    "AIC_PORT_INSERTION_TERMINATION",
    "ControllerSpec",
    "InsertionGoalSpec",
    "NIC_PORT_0_INSERTION_GOAL",
    "PortInsertionAssemblySpec",
    "PortInsertionTerminationSpec",
    "UR5E_DIFF_IK_CONTROLLER",
]
