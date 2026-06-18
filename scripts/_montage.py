import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

A = Path("/workspace/isaaclab/etf_robotics_aic/outputs/eval/015_20260618-121914/analysis")
F = A / "frames"
rows = [
    ("stalled_far  (ep1, 280 steps, closest 106 mm)", ["ep001_10pct.png", "ep001_50pct.png", "ep001_95pct.png"]),
    ("approached  (ep9, 2581 steps, closest 30 mm)", ["ep009_10pct.png", "ep009_50pct.png", "ep009_95pct.png"]),
]
col_titles = ["start (~10%)", "middle (~50%)", "end (~95%)"]
fig, axes = plt.subplots(2, 3, figsize=(12, 5))
for r, (label, files) in enumerate(rows):
    for c, fn in enumerate(files):
        ax = axes[r][c]
        ax.imshow(plt.imread(F / fn))
        ax.set_xticks([]); ax.set_yticks([])
        if r == 0:
            ax.set_title(col_titles[c], fontsize=11)
        if c == 0:
            ax.set_ylabel(label, fontsize=10)
fig.suptitle("Failure modes of ACT policy (ckpt 150k) — third-person overview", fontsize=13)
fig.tight_layout()
fig.savefig(A / "fig_failure_modes.png", dpi=160)
print("saved", A / "fig_failure_modes.png")
