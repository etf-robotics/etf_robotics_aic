"""Fast check: strip the cable USD's baked sfp Body_005 override, confirm the full
referenced mesh composes (point count) and render overview + policy cams.

In-memory only: retarget .glb->.usd, then remove the baked geom attrs the cable
layer authors on the sfp Body_005 over (points/faceVertexCounts/faceVertexIndices/
primvars:st) so our full 8126-pt connector mesh shows through instead of the
partial 2457-pt baked one.
"""
import argparse
import sys
from pathlib import Path
from isaaclab.app import AppLauncher

_SD = Path(__file__).resolve().parent
sys.path.insert(0, str(_SD))
pa = argparse.ArgumentParser()
pa.add_argument("--out", default="/tmp/shot")
pa.add_argument("--no_strip", action="store_true", help="skip the override strip (show the broken baseline)")
AppLauncher.add_app_launcher_args(pa)
args = pa.parse_args(); args.headless = True; args.enable_cameras = True
app = AppLauncher(args).app

import numpy as np  # noqa: E402
import torch  # noqa: E402
import imageio.v3 as iio  # noqa: E402
import omni.usd  # noqa: E402
import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.sensors import TiledCameraCfg  # noqa: E402
from isaaclab.utils.math import create_rotation_matrix_from_view, quat_from_matrix  # noqa: E402
import isaaclab_tasks  # noqa: F401,E402
import aic_task.tasks  # noqa: F401,E402
from il.env_wrapper import PortInsertionEnv  # noqa: E402
from pxr import Sdf, Usd, UsdGeom  # noqa: E402

_CABLE = (_SD.parent / "source/aic_task/aic_task/assets/robots/ur5e_cable"
          / "aic_unified_robot_cable_sdf.usd")
_BAKED = ("points", "faceVertexCounts", "faceVertexIndices", "primvars:st",
          "primvars:st:indices", "normals", "extent")
_KEEP = []


def retarget_and_strip():
    layer = Sdf.Layer.FindOrOpen(str(_CABLE)); _KEEP.append(layer)
    for s in ("lc_plug_visual", "sc_plug_visual", "sfp_module_visual"):
        layer.UpdateCompositionAssetDependency(f"./visuals/{s}.glb", f"./visuals/{s}.usd")
    if args.no_strip:
        return
    body = layer.GetPrimAtPath("/World/cable/sfp_module/sfp_module_link/visual/Body_005")
    if body is not None:
        for prop in list(body.properties):
            if prop.name in _BAKED:
                body.RemoveProperty(prop)
        print("[shot] stripped baked sfp Body_005 geom override", file=sys.stderr)


def ov_cfg(h, w, eye, tgt):
    rot = create_rotation_matrix_from_view(torch.tensor([eye]), torch.tensor([tgt]), up_axis="Z")
    q = tuple(quat_from_matrix(rot)[0].tolist())
    return TiledCameraCfg(prim_path="{ENV_REGEX_NS}/overview_cam",
                          spawn=sim_utils.PinholeCameraCfg(focal_length=15.0, focus_distance=0.0,
                                                           horizontal_aperture=20.955,
                                                           vertical_aperture=20.955 * h / w,
                                                           clipping_range=(0.05, 50.0)),
                          height=h, width=w, data_types=["rgb"],
                          offset=TiledCameraCfg.OffsetCfg(pos=tuple(eye), rot=q, convention="opengl"))


retarget_and_strip()
env = PortInsertionEnv.make(task="AIC-Port-Insertion-v0", num_envs=1, device=args.device,
                            extra_sensors={"overview_camera": ov_cfg(720, 1280, (1.4, 0.2, 0.7), (0.0, 0.0, 0.25))})
obs, _ = env.reset(seed=0)
act = torch.zeros((env.num_envs, env.unwrapped.action_manager.total_action_dim), device=env.device)
for _ in range(30):
    obs, *_ = env.step(act)

stage = omni.usd.get_context().get_stage()
m = stage.GetPrimAtPath("/World/envs/env_0/Robot/cable/sfp_module/sfp_module_link/visual/Body_005")
pts = UsdGeom.Mesh(m).GetPointsAttr().Get() if m and m.IsValid() else None
print(f"[shot] composed sfp Body_005 points = {len(pts) if pts else 0} (full=8126, baked=2457)", file=sys.stderr)

ov = env.unwrapped.scene["overview_camera"].data.output["rgb"][0, ..., :3].detach().cpu().numpy()
cams = torch.cat([obs["policy"][k] for k in ("center_camera_rgb", "left_camera_rgb", "right_camera_rgb")],
                 dim=2)[0, ..., :3].detach().cpu().numpy()
iio.imwrite(f"{args.out}_overview.png", ov.astype(np.uint8))
iio.imwrite(f"{args.out}_cams.png", cams.astype(np.uint8))
print(f"[shot] saved {args.out}_*.png", file=sys.stderr)
env.close(); app.close()
