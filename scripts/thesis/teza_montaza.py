# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Failure-mode montage (Cyrillic) for the thesis: a representative stall episode
of each full policy (ACT, DP), three frames each (start / middle / end) from the
third-person overview video. Writes ``DIPLOMSKI/images/rez_montaza_otkaza.pdf``.

    docker exec -w /workspace/isaaclab isaac-lab-base ./isaaclab.sh -p \\
      etf_robotics_aic/scripts/teza_montaza.py
"""

import subprocess
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

_ROOT = Path(__file__).resolve().parent.parent.parent
_EVAL = _ROOT / "outputs" / "eval"
_OUT = _ROOT / "DIPLOMSKI" / "images" / "rez_montaza_otkaza.pdf"

# (column label, curated folder, episode index) — representative ~47 mm stalls.
POLICIES = [
    ("ACT (пун скуп)\nеп. 45, $d_{\\min}$ = 47,2 mm", "ACT_full", 45),
    ("DP (пун скуп)\nеп. 13, $d_{\\min}$ = 47,1 mm", "DP_full", 13),
]
TIMES = ["почетак (~10%)", "средина (~50%)", "крај (~95%)"]
FRACS = [0.10, 0.50, 0.95]

# Crop (x0, x1, y0, y1) as fractions of each 1280x720 frame: removes the white
# light-stand poles at the frame edges plus the empty sky/floor margins, so the
# robot and the port board fill the panel.
CROP = (0.095, 0.762, 0.10, 0.97)


def _run_dir(folder: str) -> Path:
    cands = [d for d in (_EVAL / folder).iterdir() if d.is_dir() and (d / "metrics.csv").exists()]
    return cands[0]


def _overview_mp4(run_dir: Path, ep: int) -> Path:
    hits = list(run_dir.glob(f"episode_{ep:03d}_env0_*_overview.mp4"))
    if not hits:
        raise SystemExit(f"no overview mp4 for ep {ep} in {run_dir}")
    return hits[0]


def _duration_s(mp4: Path) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(mp4)],
        capture_output=True, text=True, check=True,
    )
    return float(out.stdout.strip())


def _extract(mp4: Path, t: float, dst: Path) -> None:
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-ss", f"{t:.3f}", "-i", str(mp4),
         "-frames:v", "1", str(dst)],
        check=True,
    )


def main() -> None:
    plt.rcParams.update({"font.size": 11})
    # Transposed grid (policies as columns, time top->down) so the figure fills
    # a portrait A4 page with ~2x larger panels than a 2x3 row layout.
    fig, axes = plt.subplots(len(FRACS), len(POLICIES), figsize=(8.0, 8.6))
    with tempfile.TemporaryDirectory() as td:
        tdp = Path(td)
        for c, (label, folder, ep) in enumerate(POLICIES):
            mp4 = _overview_mp4(_run_dir(folder), ep)
            dur = _duration_s(mp4)
            for r, frac in enumerate(FRACS):
                frame = tdp / f"{folder}_{ep}_{r}.png"
                _extract(mp4, max(0.0, frac * dur), frame)
                img = plt.imread(frame)
                h, w = img.shape[:2]
                x0, x1 = int(CROP[0] * w), int(CROP[1] * w)
                y0, y1 = int(CROP[2] * h), int(CROP[3] * h)
                ax = axes[r][c]
                ax.imshow(img[y0:y1, x0:x1])
                ax.set_xticks([]); ax.set_yticks([])
                if r == 0:
                    ax.set_title(label, fontsize=11, pad=6)
                if c == 0:
                    ax.set_ylabel(TIMES[r], fontsize=11, rotation=90, labelpad=10, va="center")
    fig.tight_layout()
    fig.savefig(_OUT, dpi=200)
    print(f"[montaza] wrote {_OUT}")


if __name__ == "__main__":
    main()
