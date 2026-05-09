#!/usr/bin/env python3
"""Train a visual behavior-cloning policy from port-insertion HDF5 datasets."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

try:
    import h5py
except ModuleNotFoundError:
    h5py = None


DEFAULT_CAMERAS = ("left_camera", "center_camera", "right_camera")
DEFAULT_PROPRIO_KEYS = ("joint_pos", "joint_vel", "tcp_pose_w")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train visual BC for AIC port insertion datasets.")
    parser.add_argument(
        "--dataset_file",
        nargs="+",
        default=["./datasets/visual_port_insertion_keypoints.hdf5"],
        help="One or more HDF5 files from record_port_insertion_keypoint_dataset.py.",
    )
    parser.add_argument("--output_dir", type=str, default="./logs/il_port_bc", help="Directory for checkpoints/logs.")
    parser.add_argument("--camera_names", nargs="+", default=None, help="Camera streams to use. Defaults to dataset metadata.")
    parser.add_argument("--proprio_keys", nargs="+", default=list(DEFAULT_PROPRIO_KEYS), help="Proprio datasets to concatenate.")
    parser.add_argument("--target_action_key", type=str, default="actions/env_action", help="Dataset path for BC targets.")
    parser.add_argument("--fallback_action_key", type=str, default="actions/oracle", help="Fallback action path for older files.")
    parser.add_argument("--successful_only", action="store_true", default=False, help="Use only episodes marked success=True.")
    parser.add_argument("--val_fraction", type=float, default=0.12, help="Episode-level validation fraction.")
    parser.add_argument("--seed", type=int, default=7, help="Random seed.")
    parser.add_argument("--max_train_samples", type=int, default=0, help="Optional cap for quick experiments. 0 = all.")

    parser.add_argument("--image_size", type=int, default=128, help="Resize square image side for training.")
    parser.add_argument("--batch_size", type=int, default=64, help="Mini-batch size.")
    parser.add_argument("--epochs", type=int, default=80, help="Training epochs.")
    parser.add_argument("--lr", type=float, default=3.0e-4, help="AdamW learning rate.")
    parser.add_argument("--weight_decay", type=float, default=1.0e-4, help="AdamW weight decay.")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="Gradient norm clip. <=0 disables.")
    parser.add_argument("--num_workers", type=int, default=2, help="DataLoader workers. Use 0 if HDF5/FS is unhappy.")
    parser.add_argument("--amp", action="store_true", default=False, help="Use CUDA automatic mixed precision.")

    parser.add_argument("--action_loss_weight", type=float, default=1.0, help="Weight for normalized action BC loss.")
    parser.add_argument("--keypoint_loss_weight", type=float, default=0.05, help="Auxiliary visual keypoint UV loss weight.")
    parser.add_argument("--phase_loss_weight", type=float, default=0.02, help="Auxiliary teacher-phase loss weight.")
    parser.add_argument("--disable_keypoint_loss", action="store_true", default=False, help="Disable keypoint auxiliary head.")
    parser.add_argument("--disable_phase_loss", action="store_true", default=False, help="Disable phase auxiliary head.")

    parser.add_argument("--still_action_norm", type=float, default=0.02, help="Raw action norm below which a sample is treated as still.")
    parser.add_argument("--still_weight", type=float, default=0.20, help="Loss weight for still samples.")
    parser.add_argument("--success_tail_seconds", type=float, default=3.0, help="Downweight this many seconds at successful episode tails.")
    parser.add_argument("--success_tail_weight", type=float, default=0.25, help="Loss weight cap for successful tails.")
    parser.add_argument("--seat_phase", type=int, default=8, help="Teacher phase id for SEAT.")
    parser.add_argument("--seat_weight", type=float, default=0.20, help="Loss weight cap for SEAT samples.")
    parser.add_argument("--backoff_phase", type=int, default=7, help="Teacher phase id for BACKOFF.")
    parser.add_argument("--backoff_weight", type=float, default=0.35, help="Loss weight cap for BACKOFF samples.")

    parser.add_argument("--save_every_epochs", type=int, default=5, help="Save epoch_N.pt every N epochs. 0 disables.")
    return parser.parse_args()


@dataclass(frozen=True)
class EpisodeRef:
    file_path: str
    name: str
    num_samples: int
    success: bool
    step_hz: float
    sample_stride: int
    action_key: str


@dataclass(frozen=True)
class SampleRef:
    file_path: str
    demo_name: str
    index: int
    weight: float
    action_key: str


class PortInsertionHdf5Dataset(Dataset):
    """Lazy HDF5 dataset for image+proprio to action BC."""

    def __init__(
        self,
        samples: list[SampleRef],
        *,
        camera_names: list[str],
        proprio_keys: list[str],
        num_keypoints: int,
        keypoint_mask_key: str = "keypoints_visible",
    ):
        self.samples = samples
        self.camera_names = camera_names
        self.proprio_keys = proprio_keys
        self.num_keypoints = num_keypoints
        self.keypoint_mask_key = keypoint_mask_key
        self._files: dict[str, h5py.File] = {}

    def __len__(self) -> int:
        return len(self.samples)

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_files"] = {}
        return state

    def close(self) -> None:
        for file in self._files.values():
            file.close()
        self._files = {}

    def __getitem__(self, item: int) -> dict[str, torch.Tensor]:
        sample = self.samples[item]
        demo = self._demo(sample)

        image, image_hw = self._read_images(demo, sample.index)
        proprio = self._read_proprio(demo, sample.index)
        action = np.asarray(demo[sample.action_key][sample.index], dtype=np.float32)
        phase = self._read_phase(demo, sample.index)
        keypoints, keypoint_mask = self._read_keypoints(demo, sample.index, image_hw)

        return {
            "image": torch.from_numpy(image),
            "proprio": torch.from_numpy(proprio),
            "action": torch.from_numpy(action),
            "phase": torch.tensor(phase, dtype=torch.long),
            "keypoints": torch.from_numpy(keypoints),
            "keypoint_mask": torch.from_numpy(keypoint_mask),
            "sample_weight": torch.tensor(sample.weight, dtype=torch.float32),
        }

    def read_tabular(self, item: int) -> tuple[np.ndarray, np.ndarray]:
        sample = self.samples[item]
        demo = self._demo(sample)
        return self._read_proprio(demo, sample.index), np.asarray(demo[sample.action_key][sample.index], dtype=np.float32)

    def _file(self, file_path: str) -> h5py.File:
        file = self._files.get(file_path)
        if file is None:
            file = h5py.File(file_path, "r")
            self._files[file_path] = file
        return file

    def _demo(self, sample: SampleRef) -> h5py.Group:
        return self._file(sample.file_path)[f"data/{sample.demo_name}"]

    def _read_images(self, demo: h5py.Group, frame_index: int) -> tuple[np.ndarray, list[tuple[int, int]]]:
        channels = []
        image_hw = []
        for camera_name in self.camera_names:
            frame = np.asarray(demo[f"obs/{camera_name}/rgb"][frame_index])
            frame = frame[..., :3]
            image_hw.append((int(frame.shape[0]), int(frame.shape[1])))
            channels.append(np.moveaxis(frame, -1, 0))
        image = np.concatenate(channels, axis=0).astype(np.float32) / 255.0
        return image, image_hw

    def _read_proprio(self, demo: h5py.Group, frame_index: int) -> np.ndarray:
        values = []
        for key in self.proprio_keys:
            path = f"proprio/{key}"
            if path not in demo:
                raise KeyError(f"Missing '{path}' in {demo.name}. Available proprio keys: {list(demo['proprio'].keys())}")
            values.append(np.asarray(demo[path][frame_index], dtype=np.float32).reshape(-1))
        return np.concatenate(values, axis=0).astype(np.float32)

    @staticmethod
    def _read_phase(demo: h5py.Group, frame_index: int) -> int:
        if "labels/phase" not in demo:
            return -1
        return int(np.asarray(demo["labels/phase"][frame_index]).reshape(()))

    def _read_keypoints(
        self,
        demo: h5py.Group,
        frame_index: int,
        image_hw: list[tuple[int, int]],
    ) -> tuple[np.ndarray, np.ndarray]:
        keypoints = np.zeros((len(self.camera_names), self.num_keypoints, 2), dtype=np.float32)
        mask = np.zeros((len(self.camera_names), self.num_keypoints), dtype=np.float32)
        if self.num_keypoints <= 0:
            return keypoints, mask

        for cam_idx, camera_name in enumerate(self.camera_names):
            group_path = f"labels/{camera_name}"
            if group_path not in demo or "keypoints_uv" not in demo[group_path]:
                continue
            uv = np.asarray(demo[f"{group_path}/keypoints_uv"][frame_index], dtype=np.float32)
            height, width = image_hw[cam_idx]
            uv_norm = uv.copy()
            uv_norm[:, 0] /= max(width - 1, 1)
            uv_norm[:, 1] /= max(height - 1, 1)
            count = min(self.num_keypoints, uv_norm.shape[0])
            keypoints[cam_idx, :count] = uv_norm[:count]

            mask_key = self.keypoint_mask_key
            if mask_key not in demo[group_path] and "keypoints_in_frame" in demo[group_path]:
                mask_key = "keypoints_in_frame"
            if mask_key in demo[group_path]:
                mask_values = np.asarray(demo[f"{group_path}/{mask_key}"][frame_index], dtype=np.float32)
                mask[cam_idx, :count] = mask_values[:count]
        return keypoints, mask


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int):
        super().__init__()
        groups = 8 if out_channels % 8 == 0 else 1
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.GroupNorm(groups, out_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(groups, out_channels),
            nn.SiLU(inplace=True),
        )
        self.skip = nn.Identity()
        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x) + self.skip(x)


class VisualBCPolicy(nn.Module):
    def __init__(
        self,
        *,
        image_channels: int,
        proprio_dim: int,
        action_dim: int,
        num_cameras: int,
        num_keypoints: int,
        phase_count: int,
    ):
        super().__init__()
        self.num_cameras = num_cameras
        self.num_keypoints = num_keypoints
        self.phase_count = phase_count

        self.visual = nn.Sequential(
            nn.Conv2d(image_channels, 32, kernel_size=5, stride=2, padding=2, bias=False),
            nn.GroupNorm(8, 32),
            nn.SiLU(inplace=True),
            ConvBlock(32, 64, stride=2),
            ConvBlock(64, 128, stride=2),
            ConvBlock(128, 192, stride=2),
            ConvBlock(192, 256, stride=2),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, 256),
            nn.SiLU(inplace=True),
        )
        self.proprio = nn.Sequential(
            nn.Linear(proprio_dim, 128),
            nn.LayerNorm(128),
            nn.SiLU(inplace=True),
            nn.Linear(128, 128),
            nn.SiLU(inplace=True),
        )
        self.fusion = nn.Sequential(
            nn.Linear(384, 256),
            nn.LayerNorm(256),
            nn.SiLU(inplace=True),
            nn.Dropout(p=0.05),
            nn.Linear(256, 256),
            nn.SiLU(inplace=True),
        )
        self.action_head = nn.Linear(256, action_dim)
        self.phase_head = nn.Linear(256, phase_count) if phase_count > 0 else None
        self.keypoint_head = (
            nn.Linear(256, num_cameras * num_keypoints * 2) if num_keypoints > 0 else None
        )

    def forward(self, image: torch.Tensor, proprio: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self.fusion(torch.cat((self.visual(image), self.proprio(proprio)), dim=1))
        output = {"action": self.action_head(features)}
        if self.phase_head is not None:
            output["phase"] = self.phase_head(features)
        if self.keypoint_head is not None:
            keypoints = torch.sigmoid(self.keypoint_head(features))
            output["keypoints"] = keypoints.view(image.shape[0], self.num_cameras, self.num_keypoints, 2)
        return output


def main() -> None:
    args = parse_args()
    if h5py is None:
        raise SystemExit(
            "This script needs h5py. Run it through Isaac Lab Python, for example:\n"
            "  /home/etfrobot/IsaacLab/isaaclab.sh -p scripts/il/train.py "
            "--dataset_file ./datasets/visual_port_insertion_keypoints.hdf5"
        )

    seed_everything(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_files = [str(Path(path).expanduser()) for path in args.dataset_file]
    camera_names = args.camera_names or discover_camera_names(dataset_files[0])
    episodes = scan_episodes(dataset_files, args, camera_names)
    if not episodes:
        raise RuntimeError("No episodes matched the requested filters.")

    train_episodes, val_episodes = split_episodes(episodes, args.val_fraction, args.seed)
    train_samples = expand_samples(train_episodes, args)
    val_samples = expand_samples(val_episodes, args)
    if args.max_train_samples > 0 and len(train_samples) > args.max_train_samples:
        rng = random.Random(args.seed)
        rng.shuffle(train_samples)
        train_samples = train_samples[: args.max_train_samples]
    if not train_samples:
        raise RuntimeError("No training samples were found.")

    num_keypoints = discover_keypoint_count(train_samples, camera_names)
    phase_count = 0 if args.disable_phase_loss else discover_phase_count(dataset_files)
    if args.disable_keypoint_loss:
        num_keypoints = 0

    train_dataset = PortInsertionHdf5Dataset(
        train_samples,
        camera_names=camera_names,
        proprio_keys=args.proprio_keys,
        num_keypoints=num_keypoints,
    )
    val_dataset = PortInsertionHdf5Dataset(
        val_samples,
        camera_names=camera_names,
        proprio_keys=args.proprio_keys,
        num_keypoints=num_keypoints,
    )

    proprio_mean, proprio_std, action_mean, action_std = compute_stats(train_dataset)
    sample0 = train_dataset[0]
    train_dataset.close()
    val_dataset.close()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    stats = {
        "proprio_mean": torch.from_numpy(proprio_mean).to(device),
        "proprio_std": torch.from_numpy(proprio_std).to(device),
        "action_mean": torch.from_numpy(action_mean).to(device),
        "action_std": torch.from_numpy(action_std).to(device),
    }

    image_channels = int(sample0["image"].shape[0])
    proprio_dim = int(sample0["proprio"].numel())
    action_dim = int(sample0["action"].numel())
    model = VisualBCPolicy(
        image_channels=image_channels,
        proprio_dim=proprio_dim,
        action_dim=action_dim,
        num_cameras=len(camera_names),
        num_keypoints=num_keypoints,
        phase_count=phase_count,
    ).to(device)

    train_loader = make_loader(train_dataset, args, shuffle=True, device=device)
    val_loader = make_loader(val_dataset, args, shuffle=False, device=device) if val_samples else None
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs))
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp and device.type == "cuda")

    run_config = {
        "args": vars(args),
        "camera_names": camera_names,
        "proprio_keys": args.proprio_keys,
        "target_action_key": args.target_action_key,
        "image_channels": image_channels,
        "image_size": args.image_size,
        "proprio_dim": proprio_dim,
        "action_dim": action_dim,
        "num_keypoints": num_keypoints,
        "phase_count": phase_count,
        "train_episodes": len(train_episodes),
        "val_episodes": len(val_episodes),
        "train_samples": len(train_samples),
        "val_samples": len(val_samples),
    }
    (output_dir / "config.json").write_text(json.dumps(to_jsonable(run_config), indent=2), encoding="utf-8")

    best_val = math.inf
    log_path = output_dir / "metrics.csv"
    with log_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=["epoch", "lr", "train_loss", "train_action", "train_keypoint", "train_phase", "val_loss", "val_action"],
        )
        writer.writeheader()

        for epoch in range(1, args.epochs + 1):
            start_time = time.time()
            train_metrics = run_epoch(
                model,
                train_loader,
                args,
                stats,
                device,
                optimizer=optimizer,
                scaler=scaler,
            )
            val_metrics = (
                run_epoch(model, val_loader, args, stats, device, optimizer=None, scaler=None)
                if val_loader is not None
                else {"loss": math.nan, "action": math.nan, "keypoint": math.nan, "phase": math.nan}
            )
            scheduler.step()

            row = {
                "epoch": epoch,
                "lr": optimizer.param_groups[0]["lr"],
                "train_loss": train_metrics["loss"],
                "train_action": train_metrics["action"],
                "train_keypoint": train_metrics["keypoint"],
                "train_phase": train_metrics["phase"],
                "val_loss": val_metrics["loss"],
                "val_action": val_metrics["action"],
            }
            writer.writerow(row)
            csv_file.flush()

            checkpoint = build_checkpoint(model, args, run_config, stats, epoch, train_metrics, val_metrics)
            torch.save(checkpoint, output_dir / "last.pt")
            if val_metrics["action"] < best_val:
                best_val = val_metrics["action"]
                torch.save(checkpoint, output_dir / "best.pt")
            if args.save_every_epochs > 0 and epoch % args.save_every_epochs == 0:
                torch.save(checkpoint, output_dir / f"epoch_{epoch:04d}.pt")

            elapsed = time.time() - start_time
            print(
                f"epoch {epoch:03d}/{args.epochs} "
                f"train={train_metrics['loss']:.5f} action={train_metrics['action']:.5f} "
                f"val={val_metrics['loss']:.5f} val_action={val_metrics['action']:.5f} "
                f"lr={optimizer.param_groups[0]['lr']:.2e} time={elapsed:.1f}s"
            )

    train_dataset.close()
    val_dataset.close()
    print(f"Saved checkpoints and metrics to {output_dir}")


def make_loader(dataset: Dataset, args: argparse.Namespace, *, shuffle: bool, device: torch.device) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=shuffle and len(dataset) >= args.batch_size,
        persistent_workers=args.num_workers > 0,
    )


def run_epoch(
    model: nn.Module,
    loader: DataLoader | None,
    args: argparse.Namespace,
    stats: dict[str, torch.Tensor],
    device: torch.device,
    *,
    optimizer: torch.optim.Optimizer | None,
    scaler: torch.cuda.amp.GradScaler | None,
) -> dict[str, float]:
    if loader is None:
        return {"loss": math.nan, "action": math.nan, "keypoint": math.nan, "phase": math.nan}
    training = optimizer is not None
    model.train(training)
    totals = {"loss": 0.0, "action": 0.0, "keypoint": 0.0, "phase": 0.0, "weight": 0.0}

    for batch in loader:
        image = batch["image"].to(device, non_blocking=True)
        proprio = batch["proprio"].to(device, non_blocking=True)
        action = batch["action"].to(device, non_blocking=True)
        phase = batch["phase"].to(device, non_blocking=True)
        keypoints = batch["keypoints"].to(device, non_blocking=True)
        keypoint_mask = batch["keypoint_mask"].to(device, non_blocking=True)
        sample_weight = batch["sample_weight"].to(device, non_blocking=True)

        image = prepare_images(image, args.image_size, augment=training)
        proprio = (proprio - stats["proprio_mean"]) / stats["proprio_std"]
        target_action = (action - stats["action_mean"]) / stats["action_std"]

        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type=device.type, enabled=args.amp and device.type == "cuda"):
            output = model(image, proprio)
            losses = compute_losses(output, target_action, phase, keypoints, keypoint_mask, sample_weight, args)
            loss = losses["loss"]

        if training:
            assert scaler is not None
            scaler.scale(loss).backward()
            if args.grad_clip > 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()

        batch_weight = float(sample_weight.sum().detach().cpu())
        for key in ("loss", "action", "keypoint", "phase"):
            totals[key] += float(losses[key].detach().cpu()) * batch_weight
        totals["weight"] += batch_weight

    denom = max(totals["weight"], 1.0e-9)
    return {key: totals[key] / denom for key in ("loss", "action", "keypoint", "phase")}


def compute_losses(
    output: dict[str, torch.Tensor],
    target_action: torch.Tensor,
    phase: torch.Tensor,
    keypoints: torch.Tensor,
    keypoint_mask: torch.Tensor,
    sample_weight: torch.Tensor,
    args: argparse.Namespace,
) -> dict[str, torch.Tensor]:
    denom = sample_weight.sum().clamp_min(1.0e-6)
    per_sample_action = F.smooth_l1_loss(output["action"], target_action, reduction="none").mean(dim=1)
    action_loss = (per_sample_action * sample_weight).sum() / denom
    loss = args.action_loss_weight * action_loss

    keypoint_loss = target_action.new_tensor(0.0)
    if args.keypoint_loss_weight > 0 and "keypoints" in output and keypoints.numel() > 0:
        mask = keypoint_mask.unsqueeze(-1)
        if float(mask.sum().detach().cpu()) > 0.0:
            kp_error = F.smooth_l1_loss(output["keypoints"], keypoints, reduction="none") * mask
            per_sample = kp_error.sum(dim=(1, 2, 3)) / (2.0 * mask.sum(dim=(1, 2, 3)).clamp_min(1.0))
            keypoint_loss = (per_sample * sample_weight).sum() / denom
            loss = loss + args.keypoint_loss_weight * keypoint_loss

    phase_loss = target_action.new_tensor(0.0)
    if args.phase_loss_weight > 0 and "phase" in output:
        valid = phase >= 0
        if bool(valid.any()):
            phase_per_sample = F.cross_entropy(output["phase"][valid], phase[valid], reduction="none")
            phase_weight = sample_weight[valid]
            phase_loss = (phase_per_sample * phase_weight).sum() / phase_weight.sum().clamp_min(1.0e-6)
            loss = loss + args.phase_loss_weight * phase_loss

    return {"loss": loss, "action": action_loss, "keypoint": keypoint_loss, "phase": phase_loss}


def prepare_images(image: torch.Tensor, image_size: int, *, augment: bool) -> torch.Tensor:
    if image_size > 0 and (image.shape[-2] != image_size or image.shape[-1] != image_size):
        image = F.interpolate(image, size=(image_size, image_size), mode="bilinear", align_corners=False)
    if augment:
        brightness = torch.empty((image.shape[0], 1, 1, 1), device=image.device).uniform_(0.85, 1.15)
        contrast = torch.empty((image.shape[0], 1, 1, 1), device=image.device).uniform_(0.85, 1.15)
        mean = image.mean(dim=(-2, -1), keepdim=True)
        image = (image - mean) * contrast + mean
        image = image * brightness
        image = image + torch.randn_like(image) * 0.01
        image = image.clamp(0.0, 1.0)
    return (image - 0.5) / 0.5


def scan_episodes(dataset_files: list[str], args: argparse.Namespace, camera_names: list[str]) -> list[EpisodeRef]:
    episodes: list[EpisodeRef] = []
    for file_path in dataset_files:
        with h5py.File(file_path, "r") as file:
            if "data" not in file:
                raise KeyError(f"{file_path} has no /data group.")
            data = file["data"]
            for demo_name in sorted(data.keys()):
                demo = data[demo_name]
                if not isinstance(demo, h5py.Group) or "obs" not in demo:
                    continue
                success = bool(demo.attrs.get("success", False))
                if args.successful_only and not success:
                    continue
                for camera_name in camera_names:
                    if f"obs/{camera_name}/rgb" not in demo:
                        raise KeyError(f"Missing obs/{camera_name}/rgb in {file_path}:{demo_name}")
                action_key = args.target_action_key if args.target_action_key in demo else args.fallback_action_key
                if action_key not in demo:
                    raise KeyError(
                        f"Missing action target in {file_path}:{demo_name}. Tried "
                        f"'{args.target_action_key}' and '{args.fallback_action_key}'."
                    )
                num_samples = int(demo[action_key].shape[0])
                if num_samples <= 0:
                    continue
                episodes.append(
                    EpisodeRef(
                        file_path=file_path,
                        name=demo_name,
                        num_samples=num_samples,
                        success=success,
                        step_hz=float(demo.attrs.get("step_hz", 30.0)),
                        sample_stride=int(demo.attrs.get("sample_stride", 1)),
                        action_key=action_key,
                    )
                )
    return episodes


def split_episodes(episodes: list[EpisodeRef], val_fraction: float, seed: int) -> tuple[list[EpisodeRef], list[EpisodeRef]]:
    shuffled = list(episodes)
    random.Random(seed).shuffle(shuffled)
    if len(shuffled) <= 1 or val_fraction <= 0:
        return shuffled, []
    val_count = max(1, int(round(len(shuffled) * val_fraction)))
    val_count = min(val_count, len(shuffled) - 1)
    return shuffled[val_count:], shuffled[:val_count]


def expand_samples(episodes: list[EpisodeRef], args: argparse.Namespace) -> list[SampleRef]:
    samples: list[SampleRef] = []
    for episode in episodes:
        with h5py.File(episode.file_path, "r") as file:
            demo = file[f"data/{episode.name}"]
            actions = np.asarray(demo[episode.action_key], dtype=np.float32)
            action_norms = np.linalg.norm(actions.reshape(actions.shape[0], -1), axis=1)
            phases = np.asarray(demo["labels/phase"], dtype=np.int64).reshape(-1) if "labels/phase" in demo else None
            tail_count = 0
            if episode.success and args.success_tail_seconds > 0:
                tail_count = int(math.ceil(args.success_tail_seconds * episode.step_hz / max(1, episode.sample_stride)))
            tail_start = max(0, episode.num_samples - tail_count)

            for index in range(episode.num_samples):
                weight = 1.0
                if action_norms[index] < args.still_action_norm:
                    weight = min(weight, args.still_weight)
                if tail_count > 0 and index >= tail_start:
                    weight = min(weight, args.success_tail_weight)
                if phases is not None:
                    phase = int(phases[index])
                    if phase == args.seat_phase:
                        weight = min(weight, args.seat_weight)
                    elif phase == args.backoff_phase:
                        weight = min(weight, args.backoff_weight)
                samples.append(SampleRef(episode.file_path, episode.name, index, float(weight), episode.action_key))
    return samples


def compute_stats(dataset: PortInsertionHdf5Dataset) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    proprio_values = []
    action_values = []
    for index in range(len(dataset)):
        proprio, action = dataset.read_tabular(index)
        proprio_values.append(proprio)
        action_values.append(action.reshape(-1))
    proprio = np.stack(proprio_values, axis=0)
    action = np.stack(action_values, axis=0)
    return (
        proprio.mean(axis=0).astype(np.float32),
        np.maximum(proprio.std(axis=0).astype(np.float32), 1.0e-6),
        action.mean(axis=0).astype(np.float32),
        np.maximum(action.std(axis=0).astype(np.float32), 1.0e-6),
    )


def discover_camera_names(file_path: str) -> list[str]:
    with h5py.File(file_path, "r") as file:
        env_args = parse_env_args(file)
        names = env_args.get("camera_names")
        if names:
            return list(names)
        first_demo = next(iter(file["data"].values()))
        return list(first_demo["obs"].keys()) if "obs" in first_demo else list(DEFAULT_CAMERAS)


def discover_keypoint_count(samples: list[SampleRef], camera_names: list[str]) -> int:
    if not samples:
        return 0
    sample = samples[0]
    with h5py.File(sample.file_path, "r") as file:
        demo = file[f"data/{sample.demo_name}"]
        for camera_name in camera_names:
            path = f"labels/{camera_name}/keypoints_uv"
            if path in demo:
                return int(demo[path].shape[1])
    return 0


def discover_phase_count(dataset_files: list[str]) -> int:
    max_phase = -1
    for file_path in dataset_files:
        with h5py.File(file_path, "r") as file:
            env_args = parse_env_args(file)
            phase_names = env_args.get("phase_names", {})
            for key in phase_names.keys():
                try:
                    max_phase = max(max_phase, int(key))
                except ValueError:
                    pass
            for demo in file["data"].values():
                if isinstance(demo, h5py.Group) and "labels/phase" in demo:
                    phases = np.asarray(demo["labels/phase"], dtype=np.int64)
                    if phases.size:
                        max_phase = max(max_phase, int(phases.max()))
    return max_phase + 1 if max_phase >= 0 else 0


def parse_env_args(file: h5py.File) -> dict[str, Any]:
    raw = file["data"].attrs.get("env_args", "{}")
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    try:
        return json.loads(raw)
    except Exception:
        return {}


def build_checkpoint(
    model: nn.Module,
    args: argparse.Namespace,
    run_config: dict[str, Any],
    stats: dict[str, torch.Tensor],
    epoch: int,
    train_metrics: dict[str, float],
    val_metrics: dict[str, float],
) -> dict[str, Any]:
    return {
        "model_state_dict": model.state_dict(),
        "epoch": epoch,
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
        "config": to_jsonable(run_config),
        "proprio_mean": stats["proprio_mean"].detach().cpu(),
        "proprio_std": stats["proprio_std"].detach().cpu(),
        "action_mean": stats["action_mean"].detach().cpu(),
        "action_std": stats["action_std"].detach().cpu(),
        "action_semantics": {
            "target_dataset": args.target_action_key,
            "order": ["dx", "dy", "dz", "dax", "day", "daz"],
            "controller": "DifferentialInverseKinematicsActionCfg",
            "use_relative_mode": True,
        },
    }


def to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    return value


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":
    main()
