# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Open a saved port-insertion demo in the LeRobot / Rerun viewer.

Thin wrapper around ``lerobot-dataset-viz``:

- Resolves the latest ``NNN_<timestamp>/`` run under ``--out_dir`` unless
  ``--run`` is given.
- Defaults to episode 0; override with ``--episode``.
- Default ``--mode save`` writes a ``.rrd`` file you open later with
  ``rerun file.rrd`` (no GUI needed inside the container). Pass
  ``--mode local`` if you have X11 forwarding into the container, or
  ``--mode distant`` to serve over gRPC.

Run inside the container:

    docker exec -w /workspace/isaaclab isaac-lab-base \\
      ./isaaclab.sh -p etf_robotics_aic/scripts/view_demos.py [--episode N]

The ``.rrd`` lands at ``<run>/viz/episode_<N>.rrd``; copy it to a machine
with the ``rerun-sdk`` package and run ``rerun episode_N.rrd``.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_DEFAULT_OUT_DIR = _SCRIPT_DIR.parent / "datasets" / "port_insertion"
_RUN_RE = re.compile(r"^(\d{3})_")


def _pick_run(out_dir: Path, run: str | None) -> Path:
    if run is not None:
        run_dir = out_dir / run
        if not run_dir.is_dir():
            sys.exit(f"[view_demos]: no such run: {run_dir}")
        return run_dir
    existing = sorted(d for d in out_dir.iterdir() if d.is_dir() and _RUN_RE.match(d.name))
    if not existing:
        sys.exit(f"[view_demos]: no NNN_* runs found under {out_dir}")
    return existing[-1]


def _chown_tree_to(path: Path, ref: Path) -> None:
    """Recursively chown `path` to the owner of `ref`.

    `ref` is the dataset root (or any ancestor the host user owns) — copying
    its uid/gid onto the freshly-created viz output makes the .rrd readable
    from the host without sudo.
    """
    try:
        st = ref.stat()
        uid, gid = st.st_uid, st.st_gid
        os.chown(path, uid, gid)
        if path.is_dir():
            for child in path.rglob("*"):
                try:
                    os.chown(child, uid, gid)
                except (PermissionError, FileNotFoundError):
                    pass
    except (PermissionError, FileNotFoundError):
        pass


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out_dir", type=Path, default=_DEFAULT_OUT_DIR,
                        help="Root that contains the NNN_<timestamp>/ runs.")
    parser.add_argument("--run", type=str, default=None,
                        help="Specific run dir name (e.g. 001_20260530-210934). Defaults to most recent.")
    parser.add_argument("--episode", type=int, default=0,
                        help="Episode index to visualize.")
    parser.add_argument("--mode", choices=("save", "local", "distant"), default="save",
                        help="save → write .rrd next to dataset (default; works headless). "
                             "local → spawn Rerun viewer (needs GUI / X11). "
                             "distant → gRPC server you connect to from a remote Rerun viewer.")
    parser.add_argument("--grpc_port", type=int, default=9876,
                        help="gRPC port for --mode distant.")
    args = parser.parse_args()

    out_dir = args.out_dir.resolve()
    if not out_dir.is_dir():
        sys.exit(f"[view_demos]: --out_dir does not exist: {out_dir}")

    run_dir = _pick_run(out_dir, args.run)
    viz_dir = run_dir / "viz"
    viz_dir.mkdir(exist_ok=True)
    _chown_tree_to(viz_dir, out_dir)
    print(f"[view_demos]: run={run_dir.name}, episode={args.episode}, mode={args.mode}")

    cmd = [
        sys.executable, "-m", "lerobot.scripts.lerobot_dataset_viz",
        "--repo-id", run_dir.name,
        "--root", str(run_dir),
        "--episode-index", str(args.episode),
        # Run the DataLoader in-process: forked workers crash in this container
        # (Isaac's CUDA-initialized torch + fork = unhappy). One thread is fine
        # for one episode of viz.
        "--num-workers", "0",
    ]
    if args.mode == "save":
        cmd += ["--save", "1", "--output-dir", str(viz_dir)]
    elif args.mode == "distant":
        cmd += ["--mode", "distant", "--grpc-port", str(args.grpc_port)]

    print(f"[view_demos]: $ {' '.join(cmd)}")
    result = subprocess.run(cmd, check=False)

    if args.mode == "save":
        rrd = viz_dir / f"{run_dir.name}_episode_{args.episode}.rrd"
        if rrd.exists():
            _chown_tree_to(rrd, out_dir)
            print(f"\n[view_demos]: wrote {rrd}")
            print("[view_demos]: view it with:")
            print(f"    rerun {rrd}")
            print("[view_demos]: install the viewer on the host first if needed:")
            print("    pip install rerun-sdk")
        else:
            sys.exit(f"[view_demos]: expected output file not found: {rrd}")

    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
