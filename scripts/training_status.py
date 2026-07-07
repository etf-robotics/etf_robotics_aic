# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Print a snapshot of a running (or finished) `lerobot-train` job.

Reads only files on disk (checkpoint directory names + mtimes + the
``train_config.json`` lerobot writes at startup), so it works from the
host without touching the container or the live process. Run it
whenever you want a status line:

    python scripts/training_status.py                # latest run
    python scripts/training_status.py --run NAME     # specific run

The estimate assumes a constant step rate after the first checkpoint.
That's usually fine after step ~5000; the first checkpoint includes
model init + dataset scan overhead, which would bias the rate.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_DEFAULT_TRAIN_DIR = _SCRIPT_DIR.parent / "outputs" / "train"


def _pick_run_dir(train_dir: Path, run: str | None) -> Path:
    if run is not None:
        d = train_dir / run
        if not d.is_dir():
            sys.exit(f"[status]: no such run: {d}")
        return d
    candidates = [d for d in train_dir.iterdir() if d.is_dir() and (d / "checkpoints").exists()]
    if not candidates:
        sys.exit(f"[status]: no runs with a checkpoints/ dir under {train_dir}")
    return max(candidates, key=lambda d: d.stat().st_mtime)


def _format_hms(seconds: float) -> str:
    seconds = max(0, int(seconds))
    h, r = divmod(seconds, 3600)
    m, s = divmod(r, 60)
    if h:
        return f"{h}h{m:02d}m"
    if m:
        return f"{m}m{s:02d}s"
    return f"{s}s"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--train_dir", type=Path, default=_DEFAULT_TRAIN_DIR,
                   help="Parent dir holding training runs (each a subdir).")
    p.add_argument("--run", type=str, default=None,
                   help="Specific run subdir. Defaults to most-recently-modified one.")
    args = p.parse_args()

    run_dir = _pick_run_dir(args.train_dir.resolve(), args.run)
    ckpt_dir = run_dir / "checkpoints"
    cfg_path = run_dir / "checkpoints" / "last" / "pretrained_model" / "train_config.json"
    if not cfg_path.exists():
        # First-checkpoint case: read from the only numbered checkpoint.
        for d in sorted(ckpt_dir.iterdir()):
            alt = d / "pretrained_model" / "train_config.json"
            if alt.exists():
                cfg_path = alt
                break

    total_steps = None
    if cfg_path.exists():
        try:
            cfg = json.loads(cfg_path.read_text())
            total_steps = int(cfg.get("steps", 0)) or None
        except (json.JSONDecodeError, OSError):
            pass

    numbered = sorted(
        (d for d in ckpt_dir.iterdir() if d.name.isdigit()),
        key=lambda d: int(d.name),
    )
    if not numbered:
        sys.exit(f"[status]: no numbered checkpoints yet under {ckpt_dir}")

    first_step = int(numbered[0].name)
    last_step = int(numbered[-1].name)
    first_mtime = numbered[0].stat().st_mtime
    last_mtime = numbered[-1].stat().st_mtime
    now = time.time()

    if len(numbered) >= 2:
        elapsed = last_mtime - first_mtime
        steps_done_between = last_step - first_step
        steps_per_s = steps_done_between / max(elapsed, 1e-6)
    else:
        elapsed = last_mtime - run_dir.stat().st_mtime
        steps_per_s = last_step / max(elapsed, 1e-6) if elapsed > 0 else 0.0

    seconds_since_last_ckpt = now - last_mtime
    finished = total_steps is not None and last_step >= total_steps

    if finished:
        extrapolated_step = total_steps
    elif steps_per_s > 0:
        extrapolated_step = min(last_step + steps_per_s * seconds_since_last_ckpt,
                                total_steps if total_steps else float("inf"))
    else:
        extrapolated_step = last_step

    print(f"run:              {run_dir.name}")
    print(f"checkpoints:      {len(numbered)}  (first={first_step}, last={last_step})")
    print(f"step rate:        {steps_per_s:.2f} steps/s")
    if finished:
        wall = last_mtime - run_dir.stat().st_mtime
        print(f"status:           FINISHED at step {total_steps}  (wall time {_format_hms(wall)})")
        print(f"last ckpt age:    {_format_hms(seconds_since_last_ckpt)} ago")
    else:
        print(f"current step:     ~{int(extrapolated_step)}  ({seconds_since_last_ckpt:.0f}s since last ckpt)")
        if total_steps:
            progress = extrapolated_step / total_steps
            remaining_steps = max(0, total_steps - extrapolated_step)
            eta_s = remaining_steps / steps_per_s if steps_per_s > 0 else float("inf")
            bar = "#" * int(progress * 40) + "-" * (40 - int(progress * 40))
            print(f"progress:         [{bar}] {progress * 100:5.1f}%  ({int(extrapolated_step)}/{total_steps})")
            print(f"ETA:              {_format_hms(eta_s)}  (finish ~{time.strftime('%H:%M', time.localtime(now + eta_s))})")
        else:
            print("progress:         (total --steps not found in train_config.json)")
    print(f"last ckpt path:   {numbered[-1]}")


if __name__ == "__main__":
    main()
