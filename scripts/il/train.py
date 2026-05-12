#!/usr/bin/env python3
"""Train robomimic BC checkpoints from AIC Isaac Lab recorder datasets.

The expected dataset is produced by ``record_port_insertion_keypoint_dataset.py``:

    data/demo_N/obs/{joint_pos,joint_vel,eef_pose,wrist_wrench,actions,center_rgb,left_rgb,right_rgb}
    data/demo_N/actions

The saved checkpoints are robomimic ``.pth`` files, so they can be loaded
directly by ``aic_model.IsaacLabPolicy`` with ``FileUtils.policy_from_checkpoint``.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader

import robomimic.utils.file_utils as FileUtils
import robomimic.utils.obs_utils as ObsUtils
import robomimic.utils.torch_utils as TorchUtils
import robomimic.utils.train_utils as TrainUtils
from robomimic.algo import algo_factory
from robomimic.config import config_factory


CONTRACT_LOW_DIM_KEYS = ("joint_pos", "joint_vel", "eef_pose", "wrist_wrench", "actions")
CONTRACT_RGB_KEYS = ("center_rgb", "left_rgb", "right_rgb")
CONTRACT_OBS_KEYS = (*CONTRACT_LOW_DIM_KEYS, *CONTRACT_RGB_KEYS)
EXPECTED_LOW_DIM_SHAPES = {
    "joint_pos": (6,),
    "joint_vel": (6,),
    "eef_pose": (7,),
    "wrist_wrench": (6,),
    "actions": (6,),
}
ACTION_ORDER = ("dx", "dy", "dz", "dax", "day", "daz")
ISAAC_ACTION_SCALE = (0.015, 0.015, 0.015, 0.025, 0.025, 0.025)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train AIC robomimic BC from Isaac Lab recorder HDF5.")
    parser.add_argument(
        "--dataset_file",
        "--dataset",
        dest="dataset_file",
        type=str,
        required=True,
        help="HDF5 file from record_port_insertion_keypoint_dataset.py.",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="AIC-Port-Insertion-v0",
        help="Task name stored in checkpoint metadata.",
    )
    parser.add_argument("--algo", type=str, default="bc", help="robomimic algorithm. This script is tuned for bc.")
    parser.add_argument("--name", type=str, default=None, help="Experiment name. Defaults to dataset stem.")
    parser.add_argument("--output_dir", type=str, default="./logs/aic/robomimic", help="Root output directory.")

    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1.0e-4)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--steps_per_epoch", type=int, default=500, help="Gradient steps per epoch.")
    parser.add_argument("--save_every_epochs", type=int, default=25, help="Save model_epoch_N.pth cadence. 0 disables.")
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--cpu", action="store_true", default=False, help="Force CPU training.")

    parser.add_argument("--obs_keys", nargs="+", default=None, help="Override all observation keys.")
    parser.add_argument("--rgb_keys", nargs="+", default=None, help="Override RGB observation keys.")
    parser.add_argument("--low_dim_keys", nargs="+", default=None, help="Override low-dimensional observation keys.")
    parser.add_argument("--allow_extra_obs", action="store_true", default=False, help="Use extra dataset obs keys too.")
    parser.add_argument("--dry_run", action="store_true", default=False, help="Validate dataset/config and exit.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_path = Path(args.dataset_file).expanduser()
    if not dataset_path.is_file():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    seed_everything(args.seed)
    low_dim_keys, rgb_keys = discover_obs_keys(dataset_path, args)
    validate_dataset(dataset_path, low_dim_keys, rgb_keys)

    config = build_config(args, str(dataset_path), low_dim_keys, rgb_keys)
    ObsUtils.initialize_obs_utils_with_config(config)

    env_meta = load_env_meta(str(dataset_path), args.task)
    shape_meta = FileUtils.get_shape_metadata_from_dataset(
        dataset_path=str(dataset_path),
        all_obs_keys=config.all_obs_keys,
        verbose=True,
    )
    assert int(shape_meta["ac_dim"]) == 6, f"Expected 6-D actions, got {shape_meta['ac_dim']}."

    print("\n============= AIC Robomimic Contract =============")
    print(f"dataset: {dataset_path}")
    print(f"low_dim: {list(low_dim_keys)}")
    print(f"rgb: {list(rgb_keys)}")
    print(f"action_dim: {shape_meta['ac_dim']} order={list(ACTION_ORDER)}")
    print(f"env_name: {env_meta.get('env_name')}")
    if args.dry_run:
        print("[INFO] Dry run complete.")
        return

    device = TorchUtils.get_torch_device(try_to_use_cuda=(not args.cpu and config.train.cuda))
    model = algo_factory(
        algo_name=config.algo_name,
        config=config,
        obs_key_shapes=shape_meta["all_shapes"],
        ac_dim=shape_meta["ac_dim"],
        device=device,
    )

    trainset, _ = TrainUtils.load_data_for_training(config, obs_keys=shape_meta["all_obs_keys"])
    train_loader = DataLoader(
        dataset=trainset,
        sampler=trainset.get_dataset_sampler(),
        batch_size=config.train.batch_size,
        shuffle=trainset.get_dataset_sampler() is None,
        num_workers=config.train.num_data_workers,
        drop_last=len(trainset) >= config.train.batch_size,
    )

    run_dir, ckpt_dir, config_path = make_run_dirs(args, dataset_path)
    config_path.write_text(json.dumps(json.loads(config.dump()), indent=4), encoding="utf-8")
    write_contract_metadata(run_dir, args, low_dim_keys, rgb_keys)

    print("\n============= Training =============")
    print(model)
    print(trainset)
    print(f"[INFO] Run dir: {run_dir}")

    best_loss = float("inf")
    obs_normalization_stats = trainset.get_obs_normalization_stats() if config.train.hdf5_normalize_obs else None

    for epoch in range(1, config.train.num_epochs + 1):
        start_time = time.time()
        step_log = TrainUtils.run_epoch(
            model=model,
            data_loader=train_loader,
            epoch=epoch,
            validate=False,
            num_steps=config.experiment.epoch_every_n_steps,
            obs_normalization_stats=obs_normalization_stats,
        )
        loss = float(step_log.get("Loss", np.nan))
        elapsed = time.time() - start_time
        print(f"epoch {epoch:04d}/{config.train.num_epochs} loss={loss:.6f} time={elapsed:.1f}s")

        last_path = ckpt_dir / "last.pth"
        save_checkpoint(model, config, env_meta, shape_meta, last_path, obs_normalization_stats)

        if loss < best_loss:
            best_loss = loss
            save_checkpoint(model, config, env_meta, shape_meta, ckpt_dir / "best.pth", obs_normalization_stats)

        if args.save_every_epochs > 0 and epoch % args.save_every_epochs == 0:
            save_checkpoint(
                model,
                config,
                env_meta,
                shape_meta,
                ckpt_dir / f"model_epoch_{epoch}.pth",
                obs_normalization_stats,
            )

    save_checkpoint(
        model,
        config,
        env_meta,
        shape_meta,
        ckpt_dir / f"model_epoch_{config.train.num_epochs}.pth",
        obs_normalization_stats,
    )
    print(f"[INFO] Saved robomimic checkpoints to {ckpt_dir}")


def discover_obs_keys(dataset_path: Path, args: argparse.Namespace) -> tuple[tuple[str, ...], tuple[str, ...]]:
    with h5py.File(dataset_path, "r") as file:
        demo = first_demo(file)
        available = set(demo["obs"].keys())

        if args.low_dim_keys is not None or args.rgb_keys is not None:
            low_dim = tuple(args.low_dim_keys or ())
            rgb = tuple(args.rgb_keys or ())
        elif args.obs_keys is not None:
            requested = tuple(args.obs_keys)
            rgb = tuple(key for key in requested if key.endswith("_rgb"))
            low_dim = tuple(key for key in requested if key not in rgb)
        else:
            low_dim = tuple(key for key in CONTRACT_LOW_DIM_KEYS if key in available)
            rgb = tuple(key for key in CONTRACT_RGB_KEYS if key in available)
            if args.allow_extra_obs:
                extra = tuple(key for key in sorted(available) if key not in set(low_dim) | set(rgb))
                extra_rgb = tuple(key for key in extra if is_rgb_dataset(demo["obs"][key]))
                extra_low_dim = tuple(key for key in extra if key not in extra_rgb)
                low_dim = (*low_dim, *extra_low_dim)
                rgb = (*rgb, *extra_rgb)

    missing = [key for key in CONTRACT_OBS_KEYS if key not in set(low_dim) | set(rgb)]
    if not args.allow_extra_obs and missing:
        raise KeyError(f"Dataset is missing IsaacLabPolicy obs keys: {missing}")
    return low_dim, rgb


def validate_dataset(dataset_path: Path, low_dim_keys: tuple[str, ...], rgb_keys: tuple[str, ...]) -> None:
    obs_keys = set(low_dim_keys) | set(rgb_keys)
    with h5py.File(dataset_path, "r") as file:
        if "data" not in file:
            raise KeyError(f"{dataset_path} has no /data group.")
        for demo_name, demo in file["data"].items():
            if not isinstance(demo, h5py.Group):
                continue
            if "actions" not in demo:
                raise KeyError(f"{demo.name} has no top-level actions dataset.")
            if int(demo["actions"].shape[-1]) != 6:
                raise ValueError(f"{demo.name}/actions must be 6-D, got shape {demo['actions'].shape}.")
            if "processed_actions" in demo:
                print(f"[INFO] {demo.name} contains processed_actions; training target remains raw actions.")
            available = set(demo["obs"].keys()) if "obs" in demo else set()
            missing = sorted(obs_keys - available)
            if missing:
                raise KeyError(f"{demo.name} missing obs keys: {missing}")
            for key in low_dim_keys:
                expected_shape = EXPECTED_LOW_DIM_SHAPES.get(key)
                if expected_shape is None:
                    continue
                shape = tuple(demo[f"obs/{key}"].shape[1:])
                if shape != expected_shape:
                    raise ValueError(
                        f"{demo.name}/obs/{key} must have per-step shape {expected_shape}, got {shape}. "
                        "Re-record with the IsaacLabPolicy observation contract."
                    )
            for key in rgb_keys:
                shape = demo[f"obs/{key}"].shape
                if len(shape) != 4 or shape[-1] != 3:
                    raise ValueError(f"{demo.name}/obs/{key} must be NHWC RGB, got shape {shape}.")
                if tuple(shape[1:3]) != (224, 224):
                    print(f"[WARN] {demo.name}/obs/{key} is {shape[1:3]}, expected (224, 224).")


def build_config(
    args: argparse.Namespace,
    dataset_path: str,
    low_dim_keys: tuple[str, ...],
    rgb_keys: tuple[str, ...],
):
    config = config_factory(args.algo)
    with config.values_unlocked():
        config.train.data = dataset_path
        config.train.output_dir = os.path.abspath(args.output_dir)
        config.train.num_data_workers = args.num_workers
        config.train.hdf5_cache_mode = "low_dim"
        config.train.hdf5_use_swmr = True
        config.train.hdf5_load_next_obs = False
        config.train.hdf5_normalize_obs = False
        config.train.hdf5_filter_key = None
        config.train.hdf5_validation_filter_key = None
        config.train.seq_length = 1
        config.train.pad_seq_length = True
        config.train.frame_stack = 1
        config.train.pad_frame_stack = True
        config.train.dataset_keys = ["actions"]
        config.train.goal_mode = None
        config.train.cuda = not args.cpu
        config.train.batch_size = args.batch_size
        config.train.num_epochs = args.epochs
        config.train.seed = args.seed

        config.experiment.name = args.name or Path(dataset_path).stem
        config.experiment.validate = False
        config.experiment.epoch_every_n_steps = args.steps_per_epoch
        config.experiment.render = False
        config.experiment.render_video = False
        config.experiment.rollout.enabled = False
        config.experiment.save.enabled = True
        config.experiment.save.every_n_epochs = args.save_every_epochs
        config.experiment.save.on_best_validation = False
        config.experiment.save.on_best_rollout_return = False
        config.experiment.save.on_best_rollout_success_rate = False

        config.algo.optim_params.policy.learning_rate.initial = args.lr
        config.algo.loss.l2_weight = 1.0
        config.algo.loss.l1_weight = 0.0
        config.algo.loss.cos_weight = 0.0
        config.algo.actor_layer_dims = [1024, 1024]
        if "gaussian" in config.algo:
            config.algo.gaussian.enabled = False
        if "gmm" in config.algo:
            config.algo.gmm.enabled = False
        if "vae" in config.algo:
            config.algo.vae.enabled = False
        if "rnn" in config.algo:
            config.algo.rnn.enabled = False
        if "transformer" in config.algo:
            config.algo.transformer.enabled = False

        config.observation.modalities.obs.low_dim = list(low_dim_keys)
        config.observation.modalities.obs.rgb = list(rgb_keys)
        config.observation.modalities.obs.depth = []
        config.observation.modalities.obs.scan = []
        config.observation.modalities.goal.low_dim = []
        config.observation.modalities.goal.rgb = []
        config.observation.modalities.goal.depth = []
        config.observation.modalities.goal.scan = []

        rgb_encoder = config.observation.encoder.rgb
        rgb_encoder.core_class = "VisualCore"
        rgb_encoder.core_kwargs.feature_dimension = 64
        rgb_encoder.core_kwargs.flatten = True
        rgb_encoder.core_kwargs.backbone_class = "ResNet18Conv"
        rgb_encoder.core_kwargs.backbone_kwargs.pretrained = False
        rgb_encoder.core_kwargs.backbone_kwargs.input_coord_conv = False
        rgb_encoder.core_kwargs.pool_class = "SpatialSoftmax"
        rgb_encoder.core_kwargs.pool_kwargs.num_kp = 32
        rgb_encoder.core_kwargs.pool_kwargs.learnable_temperature = False
        rgb_encoder.core_kwargs.pool_kwargs.temperature = 1.0
        rgb_encoder.core_kwargs.pool_kwargs.noise_std = 0.0
        rgb_encoder.core_kwargs.pool_kwargs.output_variance = False
        rgb_encoder.obs_randomizer_class = None
        rgb_encoder.obs_randomizer_kwargs = {}

    config.lock()
    return config


def load_env_meta(dataset_path: str, task_name: str) -> dict[str, Any]:
    try:
        env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path=dataset_path)
    except Exception:
        env_meta = {}
    env_meta = dict(env_meta or {})
    env_meta.setdefault("env_name", task_name)
    env_meta.setdefault("type", 2)
    return env_meta


def make_run_dirs(args: argparse.Namespace, dataset_path: Path) -> tuple[Path, Path, Path]:
    run_name = args.name or dataset_path.stem
    timestamp = time.strftime("%Y%m%d%H%M%S")
    run_dir = Path(args.output_dir).expanduser() / run_name / timestamp
    ckpt_dir = run_dir / "models"
    ckpt_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "logs").mkdir(parents=True, exist_ok=True)
    return run_dir, ckpt_dir, run_dir / "config.json"


def write_contract_metadata(
    run_dir: Path,
    args: argparse.Namespace,
    low_dim_keys: tuple[str, ...],
    rgb_keys: tuple[str, ...],
) -> None:
    metadata = OrderedDict(
        dataset_file=str(Path(args.dataset_file).expanduser()),
        task=args.task,
        low_dim_keys=list(low_dim_keys),
        rgb_keys=list(rgb_keys),
        action_order=list(ACTION_ORDER),
        isaac_action_scale=list(ISAAC_ACTION_SCALE),
        target_dataset="actions",
        obs_contract="IsaacLabPolicy",
    )
    (run_dir / "aic_contract.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def save_checkpoint(model, config, env_meta, shape_meta, path: Path, obs_normalization_stats) -> None:
    TrainUtils.save_model(
        model=model,
        config=config,
        env_meta=env_meta,
        shape_meta=shape_meta,
        ckpt_path=str(path),
        obs_normalization_stats=obs_normalization_stats,
    )


def first_demo(file: h5py.File) -> h5py.Group:
    if "data" not in file:
        raise KeyError("Dataset has no /data group.")
    for name in sorted(file["data"].keys()):
        demo = file["data"][name]
        if isinstance(demo, h5py.Group) and "obs" in demo:
            return demo
    raise RuntimeError("No demos with an obs group were found.")


def is_rgb_dataset(dataset: h5py.Dataset) -> bool:
    return len(dataset.shape) == 4 and dataset.shape[-1] == 3 and dataset.name.rsplit("/", 1)[-1].endswith("_rgb")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":
    main()
