# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Thesis-ready per-run analysis figures (Cyrillic) for the two full policies.

Reads the canonical eval runs ACT_full and DP_full (tip->seat metric, N=50) and
writes to ``DIPLOMSKI/images/``:

- ``rez_ecdf_dmin.pdf``     — ECDF of closest d_min, ACT vs DP overlaid.
- ``rez_len_vs_dmin.pdf``   — episode length vs closest d_min scatter, ACT vs DP.

Run with the container python (matplotlib):

    docker exec -w /workspace/isaaclab isaac-lab-base ./isaaclab.sh -p \\
      etf_robotics_aic/scripts/teza_figure.py
"""

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

_ROOT = Path(__file__).resolve().parent.parent.parent
_EVAL = _ROOT / "outputs" / "eval"
_IMG = _ROOT / "DIPLOMSKI" / "images"

# Pool the three eval seeds {0,1,2} of each full policy (150 episodes) so the
# figures reflect the multi-seed reality (incl. the rare successes reaching the
# 3 mm threshold), not the single-seed-0 draw.
CANON = {
    "ACT": ["ACT_full", "multiseed/ms_act_s1", "multiseed/ms_act_s2"],
    "DP": ["DP_full", "multiseed/ms_dp_s1", "multiseed/ms_dp_s2"],
}
STYLE = {"ACT": dict(color="#c0392b", marker="o"), "DP": dict(color="#3b6ea5", marker="s")}


def _run_metrics(folder: str) -> Path:
    cands = list((_EVAL / folder).rglob("metrics.csv"))
    if len(cands) != 1:
        raise SystemExit(f"expected one metrics.csv under {folder}, found {len(cands)}")
    return cands[0]


def _load(folders):
    closest, length = [], []
    for folder in folders:
        rows = list(csv.DictReader(_run_metrics(folder).open()))
        closest += [float(r["closest_mm"]) for r in rows if r.get("closest_mm")]
        length += [int(r["length"]) for r in rows if r.get("closest_mm")]
    return closest, length


def main() -> None:
    data = {k: _load(v) for k, v in CANON.items()}
    plt.rcParams.update({"font.size": 11, "axes.grid": True, "grid.alpha": 0.3})

    # --- ECDF overlay ---
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for label in ("ACT", "DP"):
        closest, _ = data[label]
        xs = sorted(closest)
        ys = [(i + 1) / len(xs) for i in range(len(xs))]
        ax.step(xs, ys, where="post", lw=2, color=STYLE[label]["color"], label=label)
    ax.axvspan(0, 3, color="green", alpha=0.12, label="праг успеха ($\\leq 3$ mm)")
    ax.set_xlabel("Минимална удаљеност врх–седиште $d_{\\min}$ [mm]")
    ax.set_ylabel("Удео епизода $\\leq x$")
    ax.set_xlim(0, None)
    ax.set_ylim(0, 1.02)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(_IMG / "rez_ecdf_dmin.pdf")
    plt.close(fig)
    print(f"[teza] wrote {_IMG / 'rez_ecdf_dmin.pdf'}")

    # --- length vs closest scatter ---
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for label in ("ACT", "DP"):
        closest, length = data[label]
        ax.scatter(length, closest, s=34, alpha=0.8, edgecolor="white", linewidth=0.5,
                   color=STYLE[label]["color"], marker=STYLE[label]["marker"], label=label)
    ax.set_xlabel("Дужина епизоде [корака]")
    ax.set_ylabel("Најближи прилаз $d_{\\min}$ [mm]")
    ax.set_ylim(0, None)
    ax.legend()
    fig.tight_layout()
    fig.savefig(_IMG / "rez_len_vs_dmin.pdf")
    plt.close(fig)
    print(f"[teza] wrote {_IMG / 'rez_len_vs_dmin.pdf'}")


if __name__ == "__main__":
    main()
