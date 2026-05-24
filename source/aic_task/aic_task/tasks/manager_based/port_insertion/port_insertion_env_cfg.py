# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""V1 port insertion task configuration.

This task keeps the shared AIC scene and uses a command-owned insertion goal
for oracle control, success, and failure termination.
"""

from __future__ import annotations

import math

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from . import mdp
from .aic_task_env_cfg import (
    AICTaskEnvCfg,
    AICTaskSceneCfg,
    CommandsCfg,
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
class PortInsertionCommandsCfg(CommandsCfg):
    """Task goal for inserting the SFP module into NIC port 0."""

    insertion_goal = mdp.InsertionGoalCommandCfg(
        target_name=TARGET_NAME,
        port_name=PORT_NAME,
        port_index=0,
        target_xz_offset=(0.0, 0.001),
        approach_offset_local=(0.0, -0.09, 0.0),
    )


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
    """Command-aware success, failure, and timeout for insertion."""

    success = DoneTerm(
        func=mdp.InsertionGoalReachedSuccess,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "command_name": "insertion_goal",
            "tip_body": PLUG_TIP_BODY,
            "position_threshold": 0.003,
            "orientation_threshold": math.radians(4.0),
            "required_seconds": 0.5,
        },
    )
    failed_stationary = DoneTerm(
        func=mdp.InsertionGoalStationaryFailure,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "command_name": "insertion_goal",
            "tip_body": PLUG_TIP_BODY,
            "movement_threshold": 0.001,
            "success_position_threshold": 0.003,
            "required_seconds": 1.0,
        },
    )


@configclass
class PortInsertionEnvCfg(AICTaskEnvCfg):
    """AIC V1 insertion env for scripted data generation and later RL."""

    # The robot USD contains a PhysX collision group under the rope/cable.
    # Keep full per-env cloning, but do not add IsaacLab inter-env collision
    # groups on top of the cable's own collision setup.
    scene: PortInsertionSceneCfg = PortInsertionSceneCfg(
        num_envs=1,
        env_spacing=4.0,
        replicate_physics=False,
        filter_collisions=False,
    )
    observations: PortInsertionObservationsCfg = PortInsertionObservationsCfg()
    commands: PortInsertionCommandsCfg = PortInsertionCommandsCfg()
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
        # Demo/debug insertion needs extra authority because the cable and plug
        # contacts can otherwise overpower the relative IK command.
        arm_actuator.effort_limit_sim = 300.0
        arm_actuator.stiffness = 6000.0
        arm_actuator.damping = 300.0
