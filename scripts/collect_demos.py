# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Scripted demo collection for AIC-Port-Insertion-v0.

Multi-env, time-parameterized, open-loop. Each episode:

1. ``PortInsertionPlanner.plan()`` produces a 3-phase TCP-frame trajectory
   per env (APPROACH → ALIGN → INSERT) from the current obs.
2. ``PortInsertionExecutor`` walks the plan in lockstep across envs.
3. Envs auto-reset on termination inside ``env.step``; the executor
   re-plans only the reset envs while others continue.

The writer interface is stubbed — record / commit calls are marked with
``# TODO(writer)`` so they can be wired up once the writer API lands.
"""

import argparse
import os
import sys

from isaaclab.app import AppLauncher

# Make `il.*` importable when running from anywhere.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

parser = argparse.ArgumentParser(description="Scripted port-insertion demo collector.")
parser.add_argument("--num_envs", type=int, default=4, help="Number of parallel envs.")
parser.add_argument("--task", type=str, default="AIC-Port-Insertion-v0", help="Gym task ID.")
parser.add_argument("--seed", type=int, default=None, help="Seed for planner RNG.")
parser.add_argument(
    "--standoff_m", type=float, default=0.05,
    help="Distance above entrance along the −insertion axis for the approach point.",
)
parser.add_argument(
    "--approach_jitter_m", type=float, default=0.02,
    help="Max lateral jitter (perpendicular to insertion axis) for the approach point.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch

import isaaclab_tasks  # noqa: F401
import aic_task.tasks  # noqa: F401

from il.env_wrapper import PortInsertionEnv
from il.path_planners.port_insertion import PortInsertionExecutor, PortInsertionPlanner

# TODO(writer): replace stub once the writer API is pinned down.
# from il.writer import DemoWriter


def main():
    env = PortInsertionEnv.make(
        task=args_cli.task,
        num_envs=args_cli.num_envs,
        device=args_cli.device,
    )

    rng = None
    if args_cli.seed is not None:
        rng = torch.Generator(device=env.device)
        rng.manual_seed(args_cli.seed)

    planner = PortInsertionPlanner(
        env,
        standoff_m=args_cli.standoff_m,
        approach_jitter_m=args_cli.approach_jitter_m,
        rng=rng,
    )
    executor = PortInsertionExecutor(env, planner)

    # TODO(writer): construct the writer once its API is pinned down.
    # writer = DemoWriter(env, path=args_cli.out_dir, async_=True)

    print(f"[INFO]: Gym observation space: {env.gym_env.observation_space}")
    print(f"[INFO]: Gym action space: {env.gym_env.action_space}")

    obs, _ = env.reset()
    all_envs = torch.arange(env.num_envs, device=env.device)
    executor.reset_plan(all_envs, obs)

    while simulation_app.is_running():
        with torch.inference_mode():
            action, info = executor.step(obs)
            obs, _, terminated, truncated, _ = env.step(action)

            # TODO(writer): record the per-step sample. The writer is
            # expected to consume per-env data, so it should index obs
            # and action by env. `info["phase"]` is (N,) int8.
            # writer.record(obs=obs, action=action, phase=info["phase"],
            #               terminated=terminated, truncated=truncated)

            done = (terminated | truncated).nonzero(as_tuple=False).flatten()
            if done.numel() > 0:
                # TODO(writer): commit each finished episode with its
                # termination reason. `terminated[done]` distinguishes
                # task termination (success / stationary-failure) from
                # `truncated[done]` (timeout).
                # writer.commit_episodes(done, terminated=terminated[done],
                #                        truncated=truncated[done])

                # `obs` for `done` envs is already the post-reset obs
                # (ManagerBasedRLEnv auto-resets inside step), so we can
                # re-plan straight from it.
                executor.reset_plan(done, obs)

    # TODO(writer): writer.close() to flush any in-flight episodes.
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
