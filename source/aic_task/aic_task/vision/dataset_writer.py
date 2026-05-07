"""HDF5 writer for visual port keypoint datasets."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping, Sequence

import h5py
import numpy as np
import torch


class PortKeypointDatasetWriter:
    """Streaming HDF5 writer for RGB frames, keypoint labels, proprio, and oracle actions."""

    def __init__(
        self,
        file_path: str,
        *,
        task_name: str,
        camera_names: Sequence[str],
        keypoint_names: Sequence[str],
        phase_names: Mapping[int, str],
        step_hz: int,
        env_index: int,
    ):
        self.file_path = file_path
        self.camera_names = list(camera_names)
        self.keypoint_names = list(keypoint_names)
        self.step_hz = step_hz
        self.env_index = env_index
        self.episode_count = 0
        self.sample_count = 0

        output_dir = os.path.dirname(file_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        self._file = h5py.File(file_path, "w")
        self._data_group = self._file.create_group("data")
        self._data_group.attrs["env_args"] = json.dumps(
            {
                "env_name": task_name,
                "type": "visual_port_keypoint_dataset",
                "camera_names": self.camera_names,
                "keypoint_names": self.keypoint_names,
                "phase_names": {str(key): value for key, value in phase_names.items()},
            }
        )
        self._episode_group: h5py.Group | None = None
        self._datasets: dict[str, h5py.Dataset] = {}
        self._episode_sample_count = 0

    def start_episode(self) -> None:
        """Open a new demo group."""
        if self._episode_group is not None:
            raise RuntimeError("Previous episode is still open.")
        episode_name = f"demo_{self.episode_count}"
        self._episode_group = self._data_group.create_group(episode_name)
        self._episode_group.attrs["step_hz"] = self.step_hz
        self._episode_group.attrs["env_index"] = self.env_index
        self._episode_group.attrs["camera_names"] = json.dumps(self.camera_names)
        self._episode_group.attrs["keypoint_names"] = json.dumps(self.keypoint_names)
        self._episode_group.create_group("obs")
        self._episode_group.create_group("labels")
        self._episode_group.create_group("proprio")
        self._episode_group.create_group("actions")
        self._episode_group.create_group("camera")
        self._datasets = {}
        self._episode_sample_count = 0

    def append(
        self,
        *,
        frames: Mapping[str, np.ndarray],
        labels: Mapping,
        proprio: Mapping[str, np.ndarray | torch.Tensor],
        action: np.ndarray | torch.Tensor,
        phase: int,
        oracle: Mapping[str, np.ndarray | torch.Tensor | float],
    ) -> None:
        """Append one sample to the open episode."""
        if self._episode_group is None:
            raise RuntimeError("Call start_episode() before append().")

        self._append_frames(frames)
        self._append_labels(labels)
        self._append_mapping("proprio", proprio)
        self._append_array("actions/oracle", action, dtype=np.float32)
        self._append_scalar("labels/phase", phase, dtype=np.int32)
        self._append_mapping("labels/oracle", oracle)

        self._episode_sample_count += 1
        self.sample_count += 1

    def close_episode(self, *, success: bool) -> None:
        """Close the current demo group."""
        if self._episode_group is None:
            return
        self._episode_group.attrs["num_samples"] = self._episode_sample_count
        self._episode_group.attrs["success"] = bool(success)
        self._episode_group = None
        self._datasets = {}
        self.episode_count += 1
        self._file.flush()

    def close(self) -> None:
        """Flush and close the HDF5 file."""
        if self._episode_group is not None:
            self.close_episode(success=False)
        self._data_group.attrs["num_demos"] = self.episode_count
        self._data_group.attrs["num_samples"] = self.sample_count
        self._file.flush()
        self._file.close()

    def _append_frames(self, frames: Mapping[str, np.ndarray]) -> None:
        for camera_name in self.camera_names:
            frame = _to_numpy(frames[camera_name], dtype=np.uint8)
            path = f"obs/{camera_name}/rgb"
            if path not in self._datasets:
                self._episode_group.require_group(f"obs/{camera_name}").attrs["format"] = "rgb_uint8_nhwc"
            self._append_array(path, frame, dtype=np.uint8, compression="gzip")

    def _append_labels(self, labels: Mapping) -> None:
        self._append_array("labels/port_keypoints_w", labels["points_w"], dtype=np.float32)
        for camera_name in self.camera_names:
            camera_labels = labels["cameras"][camera_name]
            self._append_array(f"labels/{camera_name}/keypoints_uv", camera_labels["uv"], dtype=np.float32)
            self._append_array(f"labels/{camera_name}/keypoints_depth", camera_labels["depth"], dtype=np.float32)
            self._append_array(f"labels/{camera_name}/keypoints_visible", camera_labels["visible"], dtype=np.bool_)
            self._append_array(f"labels/{camera_name}/keypoints_in_frame", camera_labels["in_frame"], dtype=np.bool_)
            self._append_array(f"labels/{camera_name}/points_camera", camera_labels["points_camera"], dtype=np.float32)
            self._append_array(f"camera/{camera_name}/intrinsic", camera_labels["intrinsic"], dtype=np.float32)
            self._append_array(f"camera/{camera_name}/pos_w", camera_labels["pos_w"], dtype=np.float32)
            self._append_array(f"camera/{camera_name}/quat_w_ros", camera_labels["quat_w_ros"], dtype=np.float32)
            self._append_array(
                f"camera/{camera_name}/quat_w_projection",
                camera_labels["quat_w_projection"],
                dtype=np.float32,
            )

    def _append_mapping(self, group_path: str, values: Mapping[str, np.ndarray | torch.Tensor | float]) -> None:
        for key, value in values.items():
            if np.isscalar(value):
                self._append_scalar(f"{group_path}/{key}", value)
            else:
                self._append_array(f"{group_path}/{key}", value)

    def _append_scalar(self, path: str, value, *, dtype=np.float32) -> None:
        array = np.asarray(value, dtype=dtype)
        self._append_array(path, array, dtype=dtype)

    def _append_array(self, path: str, value, *, dtype=np.float32, compression: str | None = None) -> None:
        array = _to_numpy(value, dtype=dtype)
        dataset = self._datasets.get(path)
        if dataset is None:
            parent_path, name = path.rsplit("/", 1)
            parent = self._episode_group.require_group(parent_path)
            dataset = parent.create_dataset(
                name,
                shape=(0, *array.shape),
                maxshape=(None, *array.shape),
                chunks=(1, *array.shape),
                dtype=array.dtype,
                compression=compression,
            )
            self._datasets[path] = dataset
        dataset.resize((self._episode_sample_count + 1, *dataset.shape[1:]))
        dataset[self._episode_sample_count] = array


def _to_numpy(value, *, dtype=np.float32) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=dtype)
