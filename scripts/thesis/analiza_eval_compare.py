# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Cross-run comparison stats + scaling figure for the thesis (Серије А/Б/В).

Reads the four curated eval runs (DP/ACT × {1000, 30290}), all evaluated with
the SAME tip->seat metric and N=50 episodes, and emits:

- ``DIPLOMSKI/analiza_eval/statistika.txt`` — Cyrillic aggregate report per run.
- ``DIPLOMSKI/images/rez_serija_a.pdf``      — d_min (median closest, Q1..Q3
  whiskers) vs number of demonstrations, ACT vs DP (log x). Success rate is 0
  for all four runs, so the informative scaling axis is the closest-approach
  distance, not SR.

No Isaac imports; run with the container python (has matplotlib):

    docker exec -w /workspace/isaaclab isaac-lab-base ./isaaclab.sh -p \\
      etf_robotics_aic/scripts/analiza_eval_compare.py
"""

import csv
import statistics as st
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

_ROOT = Path(__file__).resolve().parent.parent.parent
_EVAL = _ROOT / "outputs" / "eval"

# (label, N, curated folder) — each folder holds exactly one canonical run dir
# (single eval, tip->seat metric + off-axis column). We glob the run dir inside
# so the paths stay valid if a run is re-evaluated and swapped in.
_CURATED = {
    ("DP", 1000): "DP_1000ep",
    ("DP", 10000): "DP_10000",
    ("DP", 30290): "DP_full",
    ("ACT", 1000): "ACT_1000",
    ("ACT", 10000): "ACT_10000",
    ("ACT", 30290): "ACT_full",
}


def _run_dir(folder: str) -> Path:
    cands = [d for d in (_EVAL / folder).iterdir() if d.is_dir() and (d / "metrics.csv").exists()]
    if len(cands) != 1:
        raise SystemExit(f"expected exactly one run dir with metrics.csv under {folder}, found {len(cands)}")
    return cands[0]


RUNS = [(label, n, _run_dir(folder)) for (label, n), folder in _CURATED.items()]

_OUT_STATS = _ROOT / "DIPLOMSKI" / "analiza_eval" / "statistika.txt"
_OUT_FIG = _ROOT / "DIPLOMSKI" / "images" / "rez_serija_a.pdf"


def _load(run_dir: Path):
    rows = list(csv.DictReader((run_dir / "metrics.csv").open()))
    outcomes = [r["outcome"] for r in rows]
    length = [int(r["length"]) for r in rows]
    closest = [float(r["closest_mm"]) for r in rows if r.get("closest_mm")]
    offax = [float(r["axis_offset_mm"]) for r in rows if r.get("axis_offset_mm")]
    return rows, outcomes, length, closest, offax


def _q(xs, p):
    xs = sorted(xs)
    k = (len(xs) - 1) * p
    lo = int(k)
    hi = min(lo + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def _frac_le(xs, thr):
    return 100.0 * sum(x <= thr for x in xs) / len(xs)


def main() -> None:
    lines = []
    lines.append("# Аналитика евалуације тренираних политика")
    lines.append("(праг успешне инсерције: 3 mm; N=50 епизода по run-у; метрика: врх конектора → седиште)")
    lines.append("")

    fig_pts = {"ACT": [], "DP": []}  # (N, med, q1, q3, best)

    for label, n, run_dir in RUNS:
        rows, outcomes, length, closest, offax = _load(run_dir)
        n_ep = len(rows)
        succ = sum(o == "success" for o in outcomes)
        fail = sum(o == "failed_stationary" for o in outcomes)
        tout = sum(o == "time_out" for o in outcomes)

        lines.append(f"### {label} · {n} демонстрација   (N={n_ep})")
        lines.append(f"  Стопа успешности:        {succ}/{n_ep}  ({100*succ/n_ep:.1f}%)")
        lines.append(f"  failed_stationary    {fail}/{n_ep}  ({100*fail/n_ep:.1f}%)")
        lines.append(f"  time_out             {tout}/{n_ep}  ({100*tout/n_ep:.1f}%)")
        lines.append("")
        lines.append(f"  closest tip->seat [mm]:  mean {st.mean(closest):.1f}  median {st.median(closest):.1f}  "
                     f"std {st.pstdev(closest):.1f}  min {min(closest):.1f}  max {max(closest):.1f}")
        lines.append(f"     квартили Q1/Q3:       {_q(closest,0.25):.1f} / {_q(closest,0.75):.1f} mm")
        for thr in (20, 30, 40, 50):
            lines.append(f"     удео епизода <= {thr} mm: {_frac_le(closest,thr):.0f}%")
        if offax:
            lines.append(f"  off-axis @ closest [mm]: mean {st.mean(offax):.1f}  median {st.median(offax):.1f}  "
                         f"min {min(offax):.1f}  max {max(offax):.1f}")
        lines.append(f"  дужина епизоде [корака]: mean {st.mean(length):.0f}  median {st.median(length):.0f}  "
                     f"min {min(length)}  max {max(length)}")
        lines.append("")

        fig_pts[label].append((n, st.median(closest), _q(closest, 0.25), _q(closest, 0.75),
                               st.median(offax) if offax else float("nan"),
                               _q(offax, 0.25) if offax else float("nan"),
                               _q(offax, 0.75) if offax else float("nan")))

    _OUT_STATS.write_text("\n".join(lines) + "\n")
    print(f"[compare] wrote {_OUT_STATS}")

    # --- scaling figure: two panels, median (Q1..Q3 whiskers) vs N, ACT vs DP.
    # Median closest d_min is a near-flat ~47 mm stall plateau for all four runs,
    # so the informative scaling signal lives in the off-axis alignment (right
    # panel). Success rate is 0 everywhere, hence not plotted.
    plt.rcParams.update({"font.size": 10.5, "axes.grid": True, "grid.alpha": 0.3})
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.4, 3.7))
    styles = {"ACT": dict(color="#c0392b", marker="o"), "DP": dict(color="#3b6ea5", marker="s")}
    for label in ("ACT", "DP"):
        pts = sorted(fig_pts[label])
        ns = [p[0] for p in pts]
        cmed = [p[1] for p in pts]
        c_lo = [p[1] - p[2] for p in pts]
        c_hi = [p[3] - p[1] for p in pts]
        omed = [p[4] for p in pts]
        o_lo = [p[4] - p[5] for p in pts]
        o_hi = [p[6] - p[4] for p in pts]
        ax1.errorbar(ns, cmed, yerr=[c_lo, c_hi], capsize=4, lw=2, markersize=7, label=label, **styles[label])
        ax2.errorbar(ns, omed, yerr=[o_lo, o_hi], capsize=4, lw=2, markersize=7, label=label, **styles[label])
    for ax in (ax1, ax2):
        ax.set_xscale("log")
        ax.set_xticks([1000, 10000, 30290])
        ax.set_xticklabels(["1000", "10000", "30290"])
        ax.set_xlabel("Број демонстрација $N_{\\mathrm{demo}}$")
        ax.set_ylim(0, None)
        ax.legend()
    ax1.set_ylabel("Медијана $d_{\\min}$ врх–седиште [mm]")
    ax1.set_title("(а) Прилаз седишту")
    ax2.set_ylabel("Медијана бочног одступања [mm]")
    ax2.set_title("(б) Поравнање са осом")
    fig.tight_layout()
    fig.savefig(_OUT_FIG)
    print(f"[compare] wrote {_OUT_FIG}")


if __name__ == "__main__":
    main()
