# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Aggregate-statistics + failure-classification report for an eval run.

Reads ``metrics.csv`` (written by ``eval_demos.py``) and emits, under
``<eval_dir>/analysis/``:

- ``stats_report.md``      — human/thesis-ready table of aggregate stats.
- ``per_episode.csv``      — the input rows plus a ``category`` column.
- ``fig_closest_hist.png`` — histogram of closest TCP→seat distance.
- ``fig_closest_ecdf.png`` — empirical CDF of the same.
- ``fig_len_vs_closest.png``— scatter of episode length vs closest distance.
- ``fig_length_hist.png``  — histogram of episode lengths.

No Isaac/sim imports — run it with the container python that has matplotlib:

    docker exec -w /workspace/isaaclab isaac-lab-base ./isaaclab.sh -p \\
      etf_robotics_aic/scripts/analyze_eval.py --eval_dir <dir>

Failure categories (closest TCP→seat distance, mm), tuned to the observed
bimodal gap; override with --far_mm / --near_mm:

- ``stalled_far``  (>= far_mm)         : parks at the approach/standoff pose,
                                          never commits to insertion.
- ``approached``   (< near_mm)         : reaches near the entrance but does
                                          not seat (alignment / contact finish).
- ``partial``      (in between)        : gets partway, neither clear stall nor
                                          clean approach.
"""

import argparse
import csv
import statistics as st
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

_SCRIPT_DIR = Path(__file__).resolve().parent
_DEFAULT_EVAL_ROOT = _SCRIPT_DIR.parent / "outputs" / "eval"


def _newest_eval_dir(root: Path) -> Path:
    dirs = sorted(d for d in root.iterdir() if d.is_dir() and (d / "metrics.csv").exists())
    if not dirs:
        raise SystemExit(f"No eval dir with metrics.csv under {root}")
    return dirs[-1]


def _fmt(x: float) -> str:
    return f"{x:.1f}"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--eval_dir", type=str, default=None, help="Eval run dir (default: newest under outputs/eval).")
    p.add_argument("--far_mm", type=float, default=90.0, help="closest >= this -> stalled_far.")
    p.add_argument("--near_mm", type=float, default=60.0, help="closest < this -> approached.")
    args = p.parse_args()

    eval_dir = Path(args.eval_dir) if args.eval_dir else _newest_eval_dir(_DEFAULT_EVAL_ROOT)
    rows = list(csv.DictReader((eval_dir / "metrics.csv").open()))
    if not rows:
        raise SystemExit(f"empty metrics.csv in {eval_dir}")

    out = eval_dir / "analysis"
    out.mkdir(exist_ok=True)

    n = len(rows)
    outcomes = [r["outcome"] for r in rows]
    length = [int(r["length"]) for r in rows]
    dist = [float(r["closest_mm"]) for r in rows if r["closest_mm"]]

    def categorize(d: float) -> str:
        if d >= args.far_mm:
            return "stalled_far"
        if d < args.near_mm:
            return "approached"
        return "partial"

    # --- per-episode csv with category ---
    cat_by_ep = {}
    with (out / "per_episode.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["episode", "outcome", "length", "best_insert", "closest_mm", "category"])
        for r in rows:
            d = float(r["closest_mm"]) if r["closest_mm"] else float("nan")
            cat = categorize(d) if r["closest_mm"] else "no_metric"
            cat_by_ep[int(r["episode"])] = cat
            w.writerow([r["episode"], r["outcome"], r["length"], r["best_insert"], r["closest_mm"], cat])

    cats = [cat_by_ep[int(r["episode"])] for r in rows]
    cat_counts = {c: cats.count(c) for c in ("stalled_far", "partial", "approached", "no_metric") if cats.count(c)}

    def stat_block(name, xs, unit):
        return (f"| {name} | {min(xs):.1f} | {st.median(xs):.1f} | {st.mean(xs):.1f} | "
                f"{st.pstdev(xs):.1f} | {max(xs):.1f} | {unit} |")

    succ = sum(o == "success" for o in outcomes)
    fail = sum(o == "failed_stationary" for o in outcomes)
    tout = sum(o == "time_out" for o in outcomes)

    # --- markdown report ---
    md = []
    md.append(f"# Eval analysis — `{eval_dir.name}`\n")
    md.append(f"Episodes: **{n}**\n")
    md.append("## Outcomes\n")
    md.append("| outcome | count | pct |")
    md.append("| --- | --- | --- |")
    for name, c in (("success", succ), ("failed_stationary", fail), ("time_out", tout)):
        md.append(f"| {name} | {c} | {100*c/n:.1f}% |")
    md.append("\n## Distributions\n")
    md.append("| metric | min | median | mean | std | max | unit |")
    md.append("| --- | --- | --- | --- | --- | --- | --- |")
    md.append(stat_block("closest TCP→seat", dist, "mm"))
    md.append(stat_block("episode length", length, "steps"))
    md.append("\n## Failure categories (by closest distance)\n")
    md.append(f"Thresholds: `stalled_far >= {args.far_mm:.0f} mm`, "
              f"`approached < {args.near_mm:.0f} mm`, `partial` in between.\n")
    md.append("| category | count | pct | meaning |")
    md.append("| --- | --- | --- | --- |")
    meanings = {
        "stalled_far": "parks at approach/standoff pose, never commits",
        "partial": "gets partway, neither clean stall nor clean approach",
        "approached": "reaches near entrance, fails to seat (align/contact)",
        "no_metric": "closest distance unavailable",
    }
    for c, cnt in cat_counts.items():
        md.append(f"| {c} | {cnt} | {100*cnt/n:.1f}% | {meanings[c]} |")
    # correlation note
    if len(dist) == len(length):
        try:
            corr = st.correlation(length, dist)
            md.append(f"\n**length vs closest correlation:** r = {corr:.2f} "
                      f"(negative ⇒ longer episodes get closer).\n")
        except Exception:
            pass
    (out / "stats_report.md").write_text("\n".join(md) + "\n")

    # --- figures ---
    plt.rcParams.update({"figure.dpi": 160, "font.size": 11, "axes.grid": True, "grid.alpha": 0.3})

    # 1. closest histogram
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(dist, bins=20, color="#3b6ea5", edgecolor="white")
    ax.axvline(args.far_mm, color="#c0392b", ls="--", lw=1, label=f"far ≥ {args.far_mm:.0f} mm")
    ax.axvline(args.near_mm, color="#27ae60", ls="--", lw=1, label=f"near < {args.near_mm:.0f} mm")
    ax.set_xlabel("closest TCP→seat distance (mm)")
    ax.set_ylabel("episodes")
    ax.set_title("Closest approach per episode")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "fig_closest_hist.png")
    plt.close(fig)

    # 2. ECDF
    fig, ax = plt.subplots(figsize=(6, 4))
    xs = sorted(dist)
    ys = [(i + 1) / len(xs) for i in range(len(xs))]
    ax.step(xs, ys, where="post", color="#3b6ea5")
    ax.set_xlabel("closest TCP→seat distance (mm)")
    ax.set_ylabel("fraction of episodes ≤ x")
    ax.set_title("Closest-approach ECDF")
    fig.tight_layout()
    fig.savefig(out / "fig_closest_ecdf.png")
    plt.close(fig)

    # 3. length vs closest scatter
    fig, ax = plt.subplots(figsize=(6, 4))
    colors = {"stalled_far": "#c0392b", "partial": "#e0a800", "approached": "#27ae60", "no_metric": "#888"}
    for c in cat_counts:
        xs_ = [length[i] for i in range(len(rows)) if cats[i] == c]
        ys_ = [float(rows[i]["closest_mm"]) for i in range(len(rows)) if cats[i] == c and rows[i]["closest_mm"]]
        ax.scatter(xs_, ys_, s=28, c=colors[c], label=c, alpha=0.8, edgecolor="white", linewidth=0.5)
    ax.set_xlabel("episode length (steps)")
    ax.set_ylabel("closest TCP→seat distance (mm)")
    ax.set_title("Episode length vs closest approach")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "fig_len_vs_closest.png")
    plt.close(fig)

    # 4. length histogram
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(length, bins=20, color="#7d5ba6", edgecolor="white")
    ax.set_xlabel("episode length (steps)")
    ax.set_ylabel("episodes")
    ax.set_title("Episode-length distribution")
    fig.tight_layout()
    fig.savefig(out / "fig_length_hist.png")
    plt.close(fig)

    print(f"[analyze] wrote report + 4 figures to {out}")
    print("\n".join(md[:40]))


if __name__ == "__main__":
    main()
