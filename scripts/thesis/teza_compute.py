# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Compute-utilization summary + figure for the thesis Experimental-setup section.

Reads wandb system stats (gpu util, VRAM, power) and run summaries from each
training run's local wandb datastore, prints a per-run table, and writes a
GPU-utilization + VRAM over-time figure for a representative run (ACT full) to
``DIPLOMSKI/images/rez_compute.pdf``.

    docker exec -w /workspace/isaaclab isaac-lab-base ./isaaclab.sh -p \\
      etf_robotics_aic/scripts/teza_compute.py
"""

import glob
import json
import statistics as st
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from wandb.sdk.internal.datastore import DataStore
from wandb.proto import wandb_internal_pb2 as pb

_ROOT = Path(__file__).resolve().parent.parent.parent
_IMG = _ROOT / "DIPLOMSKI" / "images"

# label -> (training-run dir holding wandb/, known step count from config)
RUNS = [
    ("ACT 1000", "outputs/act_1000_run1", 17000),
    ("ACT 10000", "outputs/act_10000_run1", 169000),
    ("ACT 30290", "outputs/act_full_5090_run1", 512000),
    ("DP 1000", "outputs/dp_phase_smoke1k", 5000),
    ("DP 10000", "outputs/dp_10000_run1", 66000),
    ("DP 30290", "outputs/dp_phase_full_001", 200000),
]


def _wandb_file(run_dir: str):
    hits = glob.glob(f"etf_robotics_aic/{run_dir}/wandb/latest-run/*.wandb")
    return hits[0] if hits else None


def _scan(fpath):
    """Return (summary dict, list of (util%, vram_bytes, power_w, cpu%))."""
    ds = DataStore()
    ds.open_for_scan(fpath)
    summary = {}
    stats = []
    while True:
        try:
            d = ds.scan_data()
        except Exception:
            break
        if d is None:
            break
        r = pb.Record()
        try:
            r.ParseFromString(d)
        except Exception:
            continue
        t = r.WhichOneof("record_type")
        if t == "stats":
            item = {it.key: it.value_json for it in r.stats.item}

            def g(k):
                v = item.get(k)
                return float(json.loads(v)) if v is not None else None

            stats.append((g("gpu.0.gpu"), g("gpu.0.memoryAllocatedBytes"),
                          g("gpu.0.powerWatts"), g("cpu")))
        elif t == "summary":
            for it in r.summary.update:
                summary[it.key] = it.value_json
    return summary, stats


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return st.mean(xs) if xs else float("nan")


def main() -> None:
    rows = []
    series = {}
    for label, run_dir, steps in RUNS:
        f = _wandb_file(run_dir)
        if not f:
            print(f"{label}: no wandb file")
            continue
        summary, stats = _scan(f)
        runtime = None
        for k, v in summary.items():
            try:
                if k == "_runtime":
                    runtime = float(json.loads(v))
            except Exception:
                pass
        util = _mean([s[0] for s in stats])
        vram_peak = max((s[1] for s in stats if s[1] is not None), default=float("nan")) / 1e9
        power = _mean([s[2] for s in stats])
        thr = (steps / runtime) if runtime else float("nan")
        rows.append((label, steps, runtime, thr, util, vram_peak, power))
        series[label] = (stats, runtime)

    print("run | steps | wall_h | step/s | GPU% | VRAMpeak_GB | power_W")
    for label, steps, runtime, thr, util, vram, power in rows:
        wh = runtime / 3600 if runtime else float("nan")
        print(f"{label} | {steps} | {wh:.2f} | {thr:.1f} | {util:.0f} | {vram:.1f} | {power:.0f}")

    # --- figure: rolling-mean GPU util over the full run vs training progress,
    #     ACT (augmentation-bound -> lower util) vs DP (no augmentation -> pinned). ---
    def _rolling(label, win=60):
        stats, runtime = series[label]
        u = [s[0] for s in stats if s[0] is not None]
        if not u:
            return [], []
        out = []
        for i in range(len(u)):
            lo = max(0, i - win // 2)
            hi = min(len(u), i + win // 2)
            out.append(sum(u[lo:hi]) / (hi - lo))
        x = [100 * i / (len(out) - 1) for i in range(len(out))]  # % progress
        return x, out

    plt.rcParams.update({"font.size": 11})
    fig, ax = plt.subplots(figsize=(7.0, 3.4))
    xa, ua = _rolling("ACT 30290")
    xd, ud = _rolling("DP 30290")
    ax.plot(xd, ud, color="#3b6ea5", lw=1.8, label="DP (без аугментације)")
    ax.plot(xa, ua, color="#c0392b", lw=1.8, label="ACT (са аугментацијом)")
    ax.set_xlabel("Напредак обуке [%]")
    ax.set_ylabel("Искоришћеност GPU [%]")
    ax.set_ylim(0, 105)
    ax.set_xlim(0, 100)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower center", fontsize=10)
    fig.tight_layout()
    fig.savefig(_IMG / "rez_compute.pdf")
    print(f"[compute] wrote {_IMG / 'rez_compute.pdf'}")


if __name__ == "__main__":
    main()
