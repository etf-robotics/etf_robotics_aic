# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""V1 port insertion task configuration.

This task keeps the shared AIC scene and marks success when the gripper has
stayed still for a short window.
"""

from __future__ import annotations

import aic_task.mdp as mdp
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from aic_task.envs.aic_task_env_cfg import (
    AICTaskEnvCfg,
    AICTaskSceneCfg,
    ObservationsCfg,
    RewardsCfg,
    TerminationsCfg,
)


TARGET_NAME = "nic_card"
PORT_NAME = "sfp_port_0"
PLUG_CENTER_BODY = "sfp_module_link"
PLUG_TIP_BODY = "sfp_tip_link"


@configclass
class PortInsertionSceneCfg(AICTaskSceneCfg):
    """AIC scene for insertion."""


@configclass
class PortInsertionObservationsCfg(ObservationsCfg):
    """Start from the shared visual/robot observations."""


@configclass
class PortInsertionRewardsCfg(RewardsCfg):
    """Minimal privileged rewards for later RL fine-tuning."""

    seat_distance = RewTerm(
        func=mdp.cheat_plug_tip_to_port_seat_distance_l2,
        weight=-4.0,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "target_cfg": SceneEntityCfg(TARGET_NAME),
            "plug_tip_body": PLUG_TIP_BODY,
            "port_name": PORT_NAME,
        },
    )
    axis_alignment = RewTerm(
        func=mdp.cheat_plug_axis_alignment_error,
        weight=-0.5,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "target_cfg": SceneEntityCfg(TARGET_NAME),
            "plug_center_body": PLUG_CENTER_BODY,
            "plug_tip_body": PLUG_TIP_BODY,
            "port_name": PORT_NAME,
        },
    )

@configclass
class PortInsertionTerminationsCfg(TerminationsCfg):
    """Insertion success is a stable gripper for 2.5 seconds."""

    success = DoneTerm(
        func=mdp.GripperStationarySuccess,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="gripper_tcp"),
            "movement_threshold": 0.01,
            "orientation_threshold": 0.001,
            "required_seconds": 5.0,
        },
    )


@configclass
class PortInsertionEnvCfg(AICTaskEnvCfg):
    """AIC V1 insertion env for scripted data generation and later RL."""

    scene: PortInsertionSceneCfg = PortInsertionSceneCfg(num_envs=1, env_spacing=4.0)
    observations: PortInsertionObservationsCfg = PortInsertionObservationsCfg()
    rewards: PortInsertionRewardsCfg = PortInsertionRewardsCfg()
    terminations: PortInsertionTerminationsCfg = PortInsertionTerminationsCfg()

    def __post_init__(self) -> None:
        super().__post_init__()

        for name in (
            "end_effector_position_tracking",
            "end_effector_position_tracking_fine_grained",
            "end_effector_position_tracking_exp",
            "end_effector_orientation_tracking",
            "end_effector_orientation_tracking_fine_grained",
            "reaching_bonus",
            "cheat_distance_to_nic_card",
            "cheat_ee_orientation_error_to_nic",
        ):
            setattr(self.rewards, name, None)

        self.episode_length_s = 120.0

        arm_actuator = self.scene.robot.actuators["arm"]
        arm_actuator.effort_limit_sim = 110.0
        arm_actuator.stiffness = 2600.0
        arm_actuator.damping = 130.0
