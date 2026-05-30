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
4. ``PortInsertionWriter`` (LeRobot dataset) buffers ``(s_t, a_t)`` per
   env and commits a buffer to disk iff the ``success`` termination
   fired for that env — failed_stationary and time_out endings are
   dropped.
"""

import argparse
import os
import signal
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

# Make `il.*` importable when running from anywhere.
_SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPT_DIR))

# Default dataset root: <repo_root>/datasets/port_insertion. Resolved from
# the script's own path so the location does not depend on the working dir
# of whoever launches it (notably `docker exec -w /workspace/isaaclab`).
_DEFAULT_OUT_DIR = str(_SCRIPT_DIR.parent / "datasets" / "port_insertion")

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
parser.add_argument(
    "--out_dir", type=str, default=_DEFAULT_OUT_DIR,
    help="Root directory for LeRobot dataset runs. Each run lands in NNN_<timestamp>/.",
)
parser.add_argument(
    "--append", action="store_true",
    help="Append to the most recent NNN_* run under --out_dir instead of creating a new one.",
)
parser.add_argument(
    "--task_label", type=str, default="AIC-Port-Insertion-v0",
    help="String written to LeRobot's per-episode `task` field.",
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
from il.writer import PortInsertionWriter


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

    print(f"[INFO]: Gym observation space: {env.gym_env.observation_space}")
    print(f"[INFO]: Gym action space: {env.gym_env.action_space}")

    obs, _ = env.reset()
    all_envs = torch.arange(env.num_envs, device=env.device)
    executor.reset_plan(all_envs, obs)

    writer = PortInsertionWriter(
        env,
        root_dir=args_cli.out_dir,
        append=args_cli.append,
        task=args_cli.task_label,
    )

    # Cooperative SIGINT: relying on KeyboardInterrupt races against
    # Isaac's own signal handler, which can hard-exit before our finally
    # runs. Flag-and-poll guarantees the loop exits at a known point so
    # the dataset can be finalized.
    stop_requested = {"flag": False}

    def _request_stop(signum, frame):  # noqa: ARG001
        stop_requested["flag"] = True
        print(
            "[collect_demos]: SIGINT received — will stop after this step.",
            file=sys.stderr,
            flush=True,
        )

    signal.signal(signal.SIGINT, _request_stop)

    try:
        while simulation_app.is_running() and not stop_requested["flag"]:
            with torch.inference_mode():
                action, info = executor.step(obs)
                # Record (s_t, a_t, phase_t) BEFORE stepping; env.step overwrites obs with s_{t+1}.
                # `info["phase"]` is the planner's per-env phase the action was emitted under.
                writer.record(obs, action, info["phase"])
                obs, _, terminated, truncated, _ = env.step(action)

                done = (terminated | truncated).nonzero(as_tuple=False).flatten()
                if done.numel() > 0:
                    # `success` fires only on success terminations — failed_stationary
                    # and time_out leave it False, so their buffers get dropped.
                    success_mask = env.unwrapped.termination_manager.get_term("success")
                    writer.commit(done, success_mask)
                    # `obs` for `done` envs is already the post-reset obs
                    # (ManagerBasedRLEnv auto-resets inside step), so we can
                    # re-plan straight from it.
                    executor.reset_plan(done, obs)
    except KeyboardInterrupt:
        # Fallback path: if SIGINT arrived before our handler was installed
        # or another path raised KeyboardInterrupt, still finalize cleanly.
        print("[collect_demos]: KeyboardInterrupt — finalizing dataset.", file=sys.stderr, flush=True)
    finally:
        writer.close()
        env.close()
        print("[collect_demos]: shutdown complete — safe to close terminal.", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
    simulation_app.close()
