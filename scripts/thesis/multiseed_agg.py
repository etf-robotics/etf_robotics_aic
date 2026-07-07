# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Aggregate the full-policy evals across seeds {0,1,2} -> mean +/- std.

Seed 0 = the canonical curated run (ACT_full / DP_full); seeds 1,2 = the
multi-seed runs under outputs/eval/ms_*. Prints a thesis-ready summary for the
success rate and the precision metrics (median d_min, median off-axis).
"""

import csv
import statistics as st
from pathlib import Path

_EVAL = Path(__file__).resolve().parent.parent.parent / "outputs" / "eval"

SEEDS = {
    "ACT": ["ACT_full", "multiseed/ms_act_s1", "multiseed/ms_act_s2"],
    "DP": ["DP_full", "multiseed/ms_dp_s1", "multiseed/ms_dp_s2"],
}


def _one_run_dir(folder: str) -> Path:
    cands = [d for d in (_EVAL / folder).rglob("metrics.csv")]
    if len(cands) != 1:
        raise SystemExit(f"expected one metrics.csv under {folder}, found {len(cands)}")
    return cands[0]


def _per_seed(folder: str):
    rows = list(csv.DictReader(_one_run_dir(folder).open()))
    n = len(rows)
    sr = 100.0 * sum(r["outcome"] == "success" for r in rows) / n
    closest = [float(r["closest_mm"]) for r in rows if r.get("closest_mm")]
    offax = [float(r["axis_offset_mm"]) for r in rows if r.get("axis_offset_mm")]
    return sr, st.median(closest), (st.median(offax) if offax else float("nan")), n


def _ms(xs):
    return st.mean(xs), (st.pstdev(xs) if len(xs) > 1 else 0.0)


def main() -> None:
    print("method | seed-folder | SR% | median d_min | median off-axis | N")
    agg = {}
    for method, folders in SEEDS.items():
        srs, dmins, offs = [], [], []
        for f in folders:
            try:
                sr, dmin, off, n = _per_seed(f)
            except SystemExit as e:
                print(f"  {method} {f}: MISSING ({e})")
                continue
            print(f"{method} | {f} | {sr:.1f} | {dmin:.1f} | {off:.1f} | {n}")
            srs.append(sr); dmins.append(dmin); offs.append(off)
        agg[method] = (srs, dmins, offs)

    print("\n== aggregate across seeds (mean +/- std) ==")
    for method, (srs, dmins, offs) in agg.items():
        if not srs:
            continue
        sr_m, sr_s = _ms(srs)
        d_m, d_s = _ms(dmins)
        o_m, o_s = _ms(offs)
        print(f"{method} (K={len(srs)}): SR = {sr_m:.1f} +/- {sr_s:.1f} % ; "
              f"median d_min = {d_m:.1f} +/- {d_s:.1f} mm ; "
              f"median off-axis = {o_m:.1f} +/- {o_s:.1f} mm")


if __name__ == "__main__":
    main()
