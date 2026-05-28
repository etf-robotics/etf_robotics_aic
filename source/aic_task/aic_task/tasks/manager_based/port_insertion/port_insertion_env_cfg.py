# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Port-insertion env cfg, assembled from asset specs via builders."""

from __future__ import annotations

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.utils import configclass

from .builders import (
    build_action_cfg,
    build_command_cfg,
    build_empty_reward_cfg,
    build_event_cfg,
    build_observation_cfg,
    build_scene_cfg,
    build_termination_cfg,
)
from .specs import AIC_PORT_INSERTION_ASSEMBLY


ASSEMBLY = AIC_PORT_INSERTION_ASSEMBLY


@configclass
class PortInsertionEnvCfg(ManagerBasedRLEnvCfg):
    scene = build_scene_cfg(ASSEMBLY)
    actions = build_action_cfg(ASSEMBLY)
    commands = build_command_cfg(ASSEMBLY)
    observations = build_observation_cfg(ASSEMBLY)
    events = build_event_cfg(ASSEMBLY)
    rewards = build_empty_reward_cfg()
    terminations = build_termination_cfg(ASSEMBLY)

    def __post_init__(self):
        super().__post_init__()
        self.decimation = 4
        self.sim.render_interval = self.decimation
        self.episode_length_s = 120.0
        self.sim.dt = 1.0 / 120.0
