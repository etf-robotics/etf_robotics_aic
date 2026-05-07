"""View or export raw camera streams saved by scripts/see_camera.py."""

from __future__ import annotations

import argparse
import json
import os

import numpy as np

try:
    import h5py
except ModuleNotFoundError as exc:
    raise SystemExit(
        "This script needs h5py. Run it through Isaac Lab Python, for example:\n"
        "  /home/etfrobot/IsaacLab/isaaclab.sh -p scripts/view_camera_dataset.py "
        "--dataset_file ./datasets/camera_stream.hdf5 --info"
    ) from exc


parser = argparse.ArgumentParser(description="View/export raw camera HDF5 datasets.")
parser.add_argument("--dataset_file", type=str, default="./datasets/camera_stream.hdf5", help="HDF5 camera dataset.")
parser.add_argument("--demo", type=str, default="demo_0", help="Demo group to read.")
parser.add_argument("--camera_names", nargs="+", default=None, help="Camera names to include. Defaults to all.")
parser.add_argument("--info", action="store_true", default=False, help="Print dataset layout and exit.")
parser.add_argument("--display", action="store_true", default=False, help="Open an OpenCV playback window.")
parser.add_argument("--export_dir", type=str, default=None, help="Export mosaic frames as dependency-free PPM images.")
parser.add_argument("--video_file", type=str, default=None, help="Optional video output path, e.g. camera_stream.mp4.")
parser.add_argument("--fps", type=float, default=30.0, help="Playback/export FPS.")
parser.add_argument("--start", type=int, default=0, help="First frame index.")
parser.add_argument("--max_frames", type=int, default=0, help="Maximum number of frames. 0 = all.")
parser.add_argument("--stride", type=int, default=1, help="Read every N-th frame.")
args = parser.parse_args()


def _get_demo_group(file: h5py.File) -> h5py.Group:
    path = f"data/{args.demo}"
    if path not in file:
        available = ", ".join(file.get("data", {}).keys())
        raise KeyError(f"Demo '{args.demo}' was not found. Available demos: {available}")
    return file[path]


def _get_camera_names(demo_group: h5py.Group) -> list[str]:
    obs_group = demo_group["obs"]
    if args.camera_names is None:
        return list(obs_group.keys())
    missing = [name for name in args.camera_names if name not in obs_group]
    if missing:
        available = ", ".join(obs_group.keys())
        raise KeyError(f"Missing cameras: {missing}. Available cameras: {available}")
    return list(args.camera_names)


def _print_info(file: h5py.File) -> None:
    data_group = file["data"]
    print(f"Dataset: {args.dataset_file}")
    if "env_args" in data_group.attrs:
        print(f"env_args: {data_group.attrs['env_args']}")
    for demo_name, demo_group in data_group.items():
        attrs = {key: _decode_attr(value) for key, value in demo_group.attrs.items()}
        print(f"\n{demo_name}: attrs={attrs}")
        if "obs" not in demo_group:
            continue
        for camera_name, camera_group in demo_group["obs"].items():
            if "rgb" in camera_group:
                dataset = camera_group["rgb"]
                print(f"  obs/{camera_name}/rgb: shape={dataset.shape}, dtype={dataset.dtype}")
        if "actions" in demo_group:
            dataset = demo_group["actions"]
            print(f"  actions: shape={dataset.shape}, dtype={dataset.dtype}")


def _decode_attr(value):
    if isinstance(value, bytes):
        return value.decode("utf-8")
    try:
        return json.loads(value)
    except Exception:
        return value


def _write_ppm(file_path: str, rgb: np.ndarray) -> None:
    height, width, channels = rgb.shape
    if channels != 3:
        raise ValueError(f"Expected RGB image with 3 channels, got shape {rgb.shape}.")
    with open(file_path, "wb") as file:
        file.write(f"P6\n{width} {height}\n255\n".encode("ascii"))
        file.write(np.ascontiguousarray(rgb).tobytes())


def _make_mosaic(demo_group: h5py.Group, camera_names: list[str], frame_index: int) -> np.ndarray:
    frames = [demo_group["obs"][camera_name]["rgb"][frame_index] for camera_name in camera_names]
    return np.concatenate(frames, axis=1)


def _get_frame_indices(num_frames: int) -> range:
    stop = num_frames if args.max_frames <= 0 else min(num_frames, args.start + args.max_frames * args.stride)
    return range(args.start, stop, args.stride)


def _open_cv2():
    try:
        import cv2
    except Exception as exc:
        raise RuntimeError(f"OpenCV is not available: {exc}") from exc
    return cv2


def main() -> None:
    if args.stride <= 0:
        raise ValueError(f"--stride must be positive, got {args.stride}.")

    with h5py.File(args.dataset_file, "r") as file:
        if args.info:
            _print_info(file)
            return

        demo_group = _get_demo_group(file)
        camera_names = _get_camera_names(demo_group)
        first_dataset = demo_group["obs"][camera_names[0]]["rgb"]
        num_frames = first_dataset.shape[0]
        frame_indices = _get_frame_indices(num_frames)

        if args.export_dir is not None:
            os.makedirs(args.export_dir, exist_ok=True)

        cv2 = None
        video_writer = None
        if args.display or args.video_file is not None:
            cv2 = _open_cv2()
        if args.display:
            if not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY"):
                raise RuntimeError("No DISPLAY/WAYLAND_DISPLAY found. Use --export_dir or enable X forwarding.")
            cv2.namedWindow("AIC camera dataset", cv2.WINDOW_NORMAL)

        exported_count = 0
        for frame_index in frame_indices:
            mosaic_rgb = _make_mosaic(demo_group, camera_names, frame_index)

            if args.export_dir is not None:
                frame_path = os.path.join(args.export_dir, f"frame_{frame_index:06d}.ppm")
                latest_path = os.path.join(args.export_dir, "latest.ppm")
                _write_ppm(frame_path, mosaic_rgb)
                _write_ppm(latest_path, mosaic_rgb)

            if args.video_file is not None:
                if video_writer is None:
                    height, width = mosaic_rgb.shape[:2]
                    output_dir = os.path.dirname(args.video_file)
                    if output_dir:
                        os.makedirs(output_dir, exist_ok=True)
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    video_writer = cv2.VideoWriter(args.video_file, fourcc, args.fps, (width, height))
                video_writer.write(cv2.cvtColor(mosaic_rgb, cv2.COLOR_RGB2BGR))

            if args.display:
                cv2.imshow("AIC camera dataset", cv2.cvtColor(mosaic_rgb, cv2.COLOR_RGB2BGR))
                key = cv2.waitKey(max(1, int(1000.0 / args.fps))) & 0xFF
                if key in (ord("q"), 27):
                    break

            exported_count += 1

        if video_writer is not None:
            video_writer.release()
        if args.display:
            cv2.destroyAllWindows()

        print(f"Read {exported_count} frame(s) from {args.dataset_file}.")
        if args.export_dir is not None:
            print(f"Exported PPM frames to: {args.export_dir}")
        if args.video_file is not None:
            print(f"Exported video to: {args.video_file}")


if __name__ == "__main__":
    main()
