# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Port approach task configuration.

This task reuses the AIC scene and robot, but narrows the objective to placing
the cable tip at an approach pose in front of the NIC-card port.
"""

from __future__ import annotations

import math

import aic_task.mdp as mdp
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from aic_task.envs.aic_task_env_cfg import (
    AICTaskEnvCfg,
    ObservationsCfg,
    RewardsCfg,
    TerminationsCfg,
)


TARGET_NAME = "nic_card"

# Robot point of interest: cable tip expressed from gripper_tcp.
#CABLE_TIP_OFFSET_FROM_TCP = (0.0, 0.05402, -0.0185)
CABLE_TIP_OFFSET_FROM_TCP = (0.0, -0.01346, 0.0435)
CABLE_TIP_RPY_FROM_TCP = (math.radians(17.2), 0.0, 0.0)

# Port entry expressed in the nic_card frame, with an extra -7 cm approach offset along local Y.
NIC_PORT_ENTRY_OFFSET = (-0.014, -0.08, 0.007)
NIC_PORT_APPROACH_OFFSET = (-0.014, -0.09, 0.007)
#NIC_PORT_APPROACH_OFFSET = (-0.07, -0.1, 0.01035)
# Target frame convention from the Isaac viewport axes: cable-tip +Z is the
# insertion axis. The approach point is farther along nic_card -Y than the port
# entry, so insertion from approach into the connector is nic_card +Y. The
# 180-degree pitch keeps +Z on +Y while flipping the plug keying.
NIC_PORT_APPROACH_RPY = (math.radians(-90.0), math.radians(180.0), 0.0)


@configclass
class PortApproachObservationsCfg(ObservationsCfg):
    """Policy observations for moving to the port approach pose."""

    @configclass
    class PolicyCfg(ObservationsCfg.PolicyCfg):
        """Robot state, visual features, and ground-truth cable-tip approach vector."""

        eef_pose = ObsTerm(
            func=mdp.body_pose_w,
            params={"asset_cfg": SceneEntityCfg("robot", body_names="gripper_tcp")},
            noise=Unoise(n_min=-0.001, n_max=0.001),
        )
        pose_command = None
        target_pos = ObsTerm(
            func=mdp.root_pos_w,
            params={"asset_cfg": SceneEntityCfg(TARGET_NAME)},
            noise=Unoise(n_min=-0.001, n_max=0.001),
        )
        target_quat = ObsTerm(
            func=mdp.root_quat_w,
            params={"asset_cfg": SceneEntityCfg(TARGET_NAME), "make_quat_unique": True},
            noise=Unoise(n_min=-0.001, n_max=0.001),
        )
        cable_tip_to_port_approach = ObsTerm(
            func=mdp.body_point_to_asset_point_position,
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names="gripper_tcp"),
                "target_cfg": SceneEntityCfg(TARGET_NAME),
                "asset_point_offset": CABLE_TIP_OFFSET_FROM_TCP,
                "target_point_offset": NIC_PORT_APPROACH_OFFSET,
            },
            noise=Unoise(n_min=-0.001, n_max=0.001),
        )

    policy: PolicyCfg = PolicyCfg()


@configclass
class PortApproachRewardsCfg(RewardsCfg):
    """Reward terms for reaching the NIC-card port approach pose with the cable tip."""

    approach_distance = RewTerm(
        func=mdp.cheat_body_point_to_asset_point_distance_l2,
        weight=-4.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="gripper_tcp"),
            "target_cfg": SceneEntityCfg(TARGET_NAME),
            "asset_point_offset": CABLE_TIP_OFFSET_FROM_TCP,
            "target_point_offset": NIC_PORT_APPROACH_OFFSET,
        },
    )
    approach_orientation = RewTerm(
        func=mdp.cheat_body_point_orientation_error_to_asset_point,
        weight=-0.4,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="gripper_tcp"),
            "target_cfg": SceneEntityCfg(TARGET_NAME),
            "asset_point_rpy": CABLE_TIP_RPY_FROM_TCP,
            "target_point_rpy": NIC_PORT_APPROACH_RPY,
        },
    )
    approach_reached = RewTerm(
        func=mdp.cheat_body_point_to_asset_point_reaching_bonus,
        weight=2.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="gripper_tcp"),
            "target_cfg": SceneEntityCfg(TARGET_NAME),
            "threshold": 0.02,
            "asset_point_offset": CABLE_TIP_OFFSET_FROM_TCP,
            "target_point_offset": NIC_PORT_APPROACH_OFFSET,
        },
    )


@configclass
class PortApproachTerminationsCfg(TerminationsCfg):
    """Termination terms for port approach."""

    success = DoneTerm(
        func=mdp.cheat_body_point_to_asset_point_pose_success,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="gripper_tcp"),
            "target_cfg": SceneEntityCfg(TARGET_NAME),
            "position_threshold": 0.01,
            "orientation_threshold": 0.01,
            "asset_point_offset": CABLE_TIP_OFFSET_FROM_TCP,
            "target_point_offset": NIC_PORT_APPROACH_OFFSET,
            "asset_point_rpy": CABLE_TIP_RPY_FROM_TCP,
            "target_point_rpy": NIC_PORT_APPROACH_RPY,
        },
    )


@configclass
class PortApproachEnvCfg(AICTaskEnvCfg):
    """AIC port approach env: move the cable tip to a pre-insertion pose."""

    observations: PortApproachObservationsCfg = PortApproachObservationsCfg()
    rewards: PortApproachRewardsCfg = PortApproachRewardsCfg()
    terminations: PortApproachTerminationsCfg = PortApproachTerminationsCfg()

    def __post_init__(self) -> None:
        super().__post_init__()

        self.rewards.end_effector_position_tracking = None
        self.rewards.end_effector_position_tracking_fine_grained = None
        self.rewards.end_effector_position_tracking_exp = None
        self.rewards.end_effector_orientation_tracking = None
        self.rewards.end_effector_orientation_tracking_fine_grained = None
        self.rewards.reaching_bonus = None
        self.rewards.cheat_distance_to_nic_card = None
        self.rewards.cheat_ee_orientation_error_to_nic = None

        self.episode_length_s = 80.0
        self.commands.ee_pose.ranges.pos_x = (0.24, 0.34)
        self.commands.ee_pose.ranges.pos_y = (0.12, 0.23)
        self.commands.ee_pose.ranges.pos_z = (0.05, 0.12)
        self.commands.ee_pose.ranges.yaw = (-math.pi, math.pi)
