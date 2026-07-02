# Diagnostic: render the scene from a re-aimed overview + policy cams, with a
# forced clean light, to see what the env actually looks like (connector etc).
import sys
from pathlib import Path
from isaaclab.app import AppLauncher
import argparse
_SD = Path(__file__).resolve().parent
sys.path.insert(0, str(_SD))
parser = argparse.ArgumentParser()
parser.add_argument("--task", default="AIC-Port-Insertion-v0")
parser.add_argument("--eye", type=float, nargs=3, default=(0.5, 0.4, 0.6))
parser.add_argument("--target", type=float, nargs=3, default=(-0.3, -0.35, 0.12))
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
args.enable_cameras = True
app = AppLauncher(args).app

import numpy as np
import torch
import imageio.v3 as iio
import omni.usd
from pxr import UsdLux
import isaaclab.sim as sim_utils
from isaaclab.sensors import TiledCameraCfg
from isaaclab.utils.math import create_rotation_matrix_from_view, quat_from_matrix
import isaaclab_tasks  # noqa: F401  registers tasks
import aic_task.tasks  # noqa: F401  registers AIC-Port-Insertion-v0
from il.env_wrapper import PortInsertionEnv

OUT = _SD.parent / "outputs" / "diag"
OUT.mkdir(parents=True, exist_ok=True)
_OV = "overview_camera"


def overview_cfg(eye, target, h=720, w=1280):
    rot = create_rotation_matrix_from_view(
        torch.tensor([eye], dtype=torch.float32),
        torch.tensor([target], dtype=torch.float32),
        up_axis="Z",
    )
    quat = tuple(quat_from_matrix(rot)[0].tolist())
    return TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/overview_cam",
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=15.0, focus_distance=0.0,
            horizontal_aperture=20.955, vertical_aperture=20.955 * h / w,
            clipping_range=(0.05, 50.0),
        ),
        height=h, width=w, data_types=["rgb"],
        offset=TiledCameraCfg.OffsetCfg(pos=tuple(eye), rot=quat, convention="opengl"),
    )


def cams_strip(obs):
    p = obs["policy"]
    views = [p[k] for k in ("center_camera_rgb", "left_camera_rgb", "right_camera_rgb")]
    return torch.cat(views, dim=2)[0, ..., :3].detach().cpu().numpy().astype(np.uint8)


def find_dome(stage):
    for prim in stage.Traverse():
        if prim.IsA(UsdLux.DomeLight):
            return prim
    return None


env = PortInsertionEnv.make(
    task=args.task, num_envs=1, device=args.device,
    extra_sensors={_OV: overview_cfg(args.eye, args.target)},
)
obs, _ = env.reset(seed=0)
stage = omni.usd.get_context().get_stage()

# Force a clean, bright, neutral light so darkness can't be blamed on lighting.
dome = find_dome(stage)
if dome:
    dl = UsdLux.DomeLight(dome)
    dl.GetIntensityAttr().Set(5000.0)
    dl.GetColorAttr().Set((1.0, 1.0, 1.0))

try:
    adim = env.action_space.shape[-1]
except Exception:
    adim = env.unwrapped.action_manager.total_action_dim
zero = torch.zeros((env.num_envs, adim), device=env.device)
for _ in range(10):
    obs, *_ = env.step(zero)

iio.imwrite(OUT / "scene_overview.png", env.unwrapped.scene[_OV].data.output["rgb"][0, ..., :3].detach().cpu().numpy().astype(np.uint8))
iio.imwrite(OUT / "scene_cams.png", cams_strip(obs))

print("=== POZICIJE ===")
print("  eye:", args.eye, "target:", args.target)
print("  env_origin:", env.unwrapped.scene.env_origins[0].tolist())
print("  robot base (world):", env.unwrapped.scene["robot"].data.root_pos_w[0].tolist())
ch = obs.get("cheatcode", {})
if "seat_pos_b" in ch:
    print("  seat/port (base frame):", ch["seat_pos_b"][0].tolist())
print("  tcp (base frame):", obs["policy"]["tcp_pos_b"][0].tolist())

import os
for f in OUT.glob("scene_*.png"):
    try:
        os.chown(f, 1000, 1000)
    except Exception:
        pass
print("DONE — frames:", [p.name for p in OUT.glob("scene_*.png")])
env.close()
app.close()
