# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from aic_task.agents.rsl_rl_ppo_cfg import PPORunnerCfg as AICPPOCfg


@configclass
class PPORunnerCfg(AICPPOCfg):
    """PPO settings for the port approach stage."""

    experiment_name = "port_approach"
    max_iterations = 1000
