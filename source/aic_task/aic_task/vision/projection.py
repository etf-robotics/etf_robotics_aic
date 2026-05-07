"""Projection helpers for camera-supervised port keypoint labels."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch

import isaaclab.utils.math as math_utils

from .port_keypoints import PortKeypointLayout


def compute_port_keypoint_labels(
    env,
    camera_names: Sequence[str],
    layout: PortKeypointLayout,
    *,
    target_name: str = "nic_card",
    min_depth: float = 0.01,
    occlusion_depth_tolerance: float = 0.015,
    camera_frame: str = "ros",
) -> dict:
    """Compute projected port keypoint labels for all envs and requested cameras.

    The returned tensors keep the env batch dimension.  A dataset writer can then
    select one env index, while vectorized jobs can still reuse the same labels.
    """
    points_w = compute_port_keypoints_w(env, layout, target_name=target_name)
    camera_labels = {}
    for camera_name in camera_names:
        camera = env.scene.sensors[camera_name]
        rgb = camera.data.output["rgb"]
        height, width = _image_height_width(rgb)
        camera_quat_w = _camera_quat_w(camera, camera_frame)
        points_camera = _world_to_camera_points(points_w, camera.data.pos_w, camera_quat_w)
        projected = math_utils.project_points(points_camera, camera.data.intrinsic_matrices)
        projected = _ensure_projected_points_batched(projected)
        uv = projected[..., :2]
        depth = projected[..., 2]
        in_frame = _points_in_frame(uv, depth, width=width, height=height, min_depth=min_depth)
        visible = in_frame.clone()

        depth_image = _get_depth_image(camera.data.output)
        if depth_image is not None:
            visible &= _depth_visibility(depth_image, uv, depth, in_frame, tolerance=occlusion_depth_tolerance)

        camera_labels[camera_name] = {
            "uv": uv,
            "depth": depth,
            "visible": visible,
            "in_frame": in_frame,
            "points_camera": points_camera,
            "intrinsic": camera.data.intrinsic_matrices,
            "pos_w": camera.data.pos_w,
            "quat_w_ros": camera.data.quat_w_ros,
            "quat_w_projection": camera_quat_w,
        }

    return {
        "keypoint_names": layout.names,
        "points_w": points_w,
        "cameras": camera_labels,
    }


def compute_port_keypoints_w(env, layout: PortKeypointLayout, *, target_name: str = "nic_card") -> torch.Tensor:
    """Return port keypoint positions in world frame as ``(N, K, 3)``."""
    target = env.scene[target_name]
    points_nic = layout.as_tensor(device=target.data.root_pos_w.device, dtype=target.data.root_pos_w.dtype)
    num_envs = target.data.root_pos_w.shape[0]
    num_points = points_nic.shape[0]
    points_nic_batched = points_nic.unsqueeze(0).expand(num_envs, -1, -1)
    quat_batched = target.data.root_quat_w[:, None, :].expand(-1, num_points, -1)
    rotated = math_utils.quat_apply(
        quat_batched.reshape(-1, 4),
        points_nic_batched.reshape(-1, 3),
    ).reshape(num_envs, num_points, 3)
    return target.data.root_pos_w[:, None, :] + rotated


def _image_height_width(image: torch.Tensor) -> tuple[int, int]:
    if image.dim() == 4:
        return int(image.shape[1]), int(image.shape[2])
    return int(image.shape[0]), int(image.shape[1])


def _camera_quat_w(camera, camera_frame: str) -> torch.Tensor:
    if camera_frame == "ros":
        return camera.data.quat_w_ros
    if camera_frame == "opengl":
        return camera.data.quat_w_opengl
    if camera_frame == "world":
        return camera.data.quat_w_world
    raise ValueError(f"Unsupported camera_frame '{camera_frame}'. Expected ros, opengl, or world.")


def labels_for_env(labels: Mapping, env_index: int) -> dict:
    """Slice a batched label dict down to one env index for serialization."""
    sliced_cameras = {}
    for camera_name, camera_labels in labels["cameras"].items():
        sliced_cameras[camera_name] = {
            key: _slice_value(value, env_index) for key, value in camera_labels.items()
        }
    return {
        "keypoint_names": labels["keypoint_names"],
        "points_w": labels["points_w"][env_index],
        "cameras": sliced_cameras,
    }


def _world_to_camera_points(
    points_w: torch.Tensor,
    camera_pos_w: torch.Tensor,
    camera_quat_w_ros: torch.Tensor,
) -> torch.Tensor:
    rel_points = points_w - camera_pos_w[:, None, :]
    num_envs, num_points, _ = rel_points.shape
    quat_batched = camera_quat_w_ros[:, None, :].expand(-1, num_points, -1)
    return math_utils.quat_apply_inverse(
        quat_batched.reshape(-1, 4),
        rel_points.reshape(-1, 3),
    ).reshape(num_envs, num_points, 3)


def _points_in_frame(
    uv: torch.Tensor,
    depth: torch.Tensor,
    *,
    width: int,
    height: int,
    min_depth: float,
) -> torch.Tensor:
    return (
        (depth > min_depth)
        & (uv[..., 0] >= 0.0)
        & (uv[..., 0] <= float(width - 1))
        & (uv[..., 1] >= 0.0)
        & (uv[..., 1] <= float(height - 1))
    )


def _get_depth_image(output: Mapping[str, torch.Tensor]) -> torch.Tensor | None:
    for key in ("distance_to_image_plane", "depth", "distance_to_camera"):
        if key in output:
            depth_image = output[key]
            if depth_image.dim() == 4 and depth_image.shape[-1] == 1:
                depth_image = depth_image[..., 0]
            elif depth_image.dim() == 3 and depth_image.shape[-1] == 1:
                depth_image = depth_image[..., 0]
            if depth_image.dim() == 2:
                depth_image = depth_image.unsqueeze(0)
            return depth_image
    return None


def _depth_visibility(
    depth_image: torch.Tensor,
    uv: torch.Tensor,
    depth: torch.Tensor,
    in_frame: torch.Tensor,
    *,
    tolerance: float,
) -> torch.Tensor:
    if uv.dim() == 2:
        uv = uv.unsqueeze(0)
    if depth.dim() == 1:
        depth = depth.unsqueeze(0)
    if in_frame.dim() == 1:
        in_frame = in_frame.unsqueeze(0)

    height, width = depth_image.shape[1], depth_image.shape[2]
    u = torch.clamp(torch.round(uv[..., 0]).long(), 0, width - 1)
    v = torch.clamp(torch.round(uv[..., 1]).long(), 0, height - 1)
    env_ids = torch.arange(depth_image.shape[0], device=depth_image.device)[:, None].expand_as(u)
    measured_depth = depth_image[env_ids, v, u]
    valid_depth = torch.isfinite(measured_depth) & (measured_depth > 0.0)
    return in_frame & valid_depth & (torch.abs(measured_depth - depth) <= tolerance)


def _slice_value(value, env_index: int):
    if isinstance(value, torch.Tensor) and value.dim() > 0 and value.shape[0] > env_index:
        return value[env_index]
    return value


def _ensure_projected_points_batched(projected: torch.Tensor) -> torch.Tensor:
    # Isaac Lab's project_points squeezes the batch dimension for N=1.  The rest
    # of this module keeps an explicit env batch for consistent serialization.
    if projected.dim() == 2:
        return projected.unsqueeze(0)
    return projected
