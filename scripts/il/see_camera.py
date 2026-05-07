"""Compatibility launcher for the raw camera viewer/recorder."""

import runpy
from pathlib import Path


if __name__ == "__main__":
    runpy.run_path(str(Path(__file__).resolve().parents[1] / "see_camera.py"), run_name="__main__")
