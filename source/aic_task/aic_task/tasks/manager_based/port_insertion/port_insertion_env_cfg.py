# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""V1 port insertion task configuration.

This task keeps the shared AIC scene, enables contact reporting on the robot,
and terminates only when the plug tip reaches the USD-resolved seat frame for
``sfp_port_0`` with the plug axis aligned to the insertion axis.
"""

from __future__ import annotations

import math

import aic_task.mdp as mdp
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.sensors import ContactSensorCfg
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
    """AIC scene with robot contact sensors enabled for insertion."""

    plug_contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/aic_unified_robot/.*",
        update_period=0.0,
        history_length=5,
        debug_vis=False,
    )

    def __post_init__(self) -> None:
        super().__post_init__()
        self.robot.spawn.activate_contact_sensors = True


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
    force_penalty = RewTerm(
        func=mdp.plug_contact_force_norm,
        weight=-0.01,
        params={
            "sensor_name": "plug_contact_forces",
            "body_regex": ".*sfp.*|.*plug.*|.*tip.*",
        },
    )


@configclass
class PortInsertionTerminationsCfg(TerminationsCfg):
    """Insertion success is seated, not merely above the port."""

    success = DoneTerm(
        func=mdp.PlugInsertedSuccess,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "target_cfg": SceneEntityCfg(TARGET_NAME),
            "plug_center_body": PLUG_CENTER_BODY,
            "plug_tip_body": PLUG_TIP_BODY,
            "port_name": PORT_NAME,
            "position_threshold": 0.004,
            "axis_threshold": math.radians(8.0),
            "required_steps": 5,
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
        self.scene.robot.spawn.activate_contact_sensors = True
        super().__post_init__()
        self.scene.robot.spawn.activate_contact_sensors = True

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
        self.actions.arm_action.scale = 0.035
