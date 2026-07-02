# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Train a Diffusion Policy with auxiliary phase supervision on the AIC port-insertion dataset.

This is the Diffusion-Policy counterpart of ``train_demos.py`` (which does the
same thing for ACT). It wraps ``lerobot.scripts.lerobot_train``:

1. Subclasses ``DiffusionPolicy`` to add a 3-class linear head on top of the
   **flattened global conditioning vector** (the concatenation of the
   ``observation.state`` vector and the per-camera ResNet features that the
   U-Net is FiLM-conditioned on) and an auxiliary cross-entropy loss against the
   dataset's ``annotation.phase`` column (one-hot {APPROACH, ALIGN, INSERT}).

   Unlike ACT — whose phase head reads a transformer encoder via a forward hook —
   Diffusion Policy has no action transformer; its only learned "summary" of the
   observation is the global-cond vector fed to the conditional U-Net. So that is
   where we attach the head. Training the head pushes gradients back through the
   RGB encoders and the state path, which is exactly the auxiliary signal we want.

   The aux loss fires **only when ``self.training`` is True**, so:

   - During ``lerobot-train`` forward: diffusion MSE loss + phase CE.
   - During ``select_action`` / ``predict_action_chunk`` (eval, deploy):
     the head is skipped entirely; behavior matches vanilla Diffusion Policy.

2. Monkey-patches ``lerobot.policies.factory.get_policy_class("diffusion")``
   to return our subclass, so all the standard ``--policy.type=diffusion``
   plumbing in lerobot's CLI / config / checkpointing still works.

3. Forwards remaining CLI args to ``lerobot.scripts.lerobot_train.main``.

Run inside the container (same way you'd run the stock trainer, just point at
this script):

    docker exec -w /workspace/isaaclab isaac-lab-base \\
      ./isaaclab.sh -p etf_robotics_aic/scripts/train_dp_demos.py \\
      --policy.type=diffusion --policy.device=cuda \\
      --policy.n_obs_steps=2 --policy.horizon=16 --policy.n_action_steps=8 \\
      --policy.crop_shape='[216, 216]' --policy.crop_is_random=true \\
      --dataset.repo_id=aic/port_insertion \\
      --dataset.root=etf_robotics_aic/datasets/port_insertion/001_20260612-132535 \\
      --output_dir=etf_robotics_aic/outputs/dp_phase_001 \\
      --batch_size=64 --steps=200000 --num_workers=4 \\
      --save_freq=5000 --eval_freq=0 --log_freq=200 --wandb.enable=false
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy

PHASE_KEY = "annotation.phase"
PHASE_N_CLASSES = 3
PHASE_LOSS_WEIGHT = 0.1  # tune if phase CE dominates / disappears


def _flattened_global_cond_dim(policy: DiffusionPolicy) -> int:
    """Replicate lerobot's global-cond dimension calc (see ``DiffusionModel.__init__``).

    This must run *after* ``super().__init__`` so the RGB encoder(s) exist, and
    the result must be known here (not lazily) so the head's parameters are in
    ``policy.parameters()`` when lerobot builds the optimizer.
    """
    cfg = policy.config
    diff = policy.diffusion
    per_step = cfg.robot_state_feature.shape[0]
    if cfg.image_features:
        num_images = len(cfg.image_features)
        if cfg.use_separate_rgb_encoder_per_camera:
            per_step += diff.rgb_encoder[0].feature_dim * num_images
        else:
            per_step += diff.rgb_encoder.feature_dim * num_images
    if cfg.env_state_feature:
        per_step += cfg.env_state_feature.shape[0]
    # _prepare_global_conditioning concatenates the per-step features over
    # n_obs_steps and flattens to (B, per_step * n_obs_steps).
    return per_step * cfg.n_obs_steps


class DiffusionPolicyWithPhaseHead(DiffusionPolicy):
    """Diffusion Policy + auxiliary subtask-phase classifier.

    The head reads the flattened global-conditioning vector that the conditional
    U-Net is conditioned on, captured by wrapping
    ``DiffusionModel._prepare_global_conditioning`` (it is called once per
    ``compute_loss``). The head is trained but never read at inference, so
    deployment is identical to vanilla Diffusion Policy — you can even load this
    checkpoint into a stock ``DiffusionPolicy`` with ``strict=False``.
    """

    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        self.phase_head = nn.Linear(_flattened_global_cond_dim(self), PHASE_N_CLASSES)
        self._gcond_buf: torch.Tensor | None = None

        # Capture the global-cond tensor without re-running the encoders. The
        # wrapper fires on every compute_loss / generate_actions; we only consume
        # the buffer in train mode and clear it after each forward so it doesn't
        # pin the autograd graph during inference.
        original = self.diffusion._prepare_global_conditioning

        def _capturing(batch, _orig=original):
            out = _orig(batch)
            self._gcond_buf = out
            return out

        self.diffusion._prepare_global_conditioning = _capturing

    def forward(self, batch):
        loss, _ = super().forward(batch)

        gcond = self._gcond_buf
        self._gcond_buf = None
        if not (self.training and PHASE_KEY in batch and gcond is not None):
            return loss, None

        phase_logits = self.phase_head(gcond)  # (B, 3)

        target = batch[PHASE_KEY].to(phase_logits.device)
        # The writer stores phase as a per-frame float32 one-hot of shape (3,).
        # If annotation.phase ends up with an obs-history dim (B, T, 3), align to
        # the most recent observation step (the one driving the action chunk).
        if target.dim() == 3:
            target = target[:, -1, :]
        target_idx = target.argmax(dim=-1).long()  # (B,)
        phase_ce = F.cross_entropy(phase_logits, target_idx)

        loss = loss + PHASE_LOSS_WEIGHT * phase_ce
        loss_dict = {"phase_ce_loss": float(phase_ce.detach())}
        with torch.no_grad():
            phase_acc = (phase_logits.argmax(dim=-1) == target_idx).float().mean()
            loss_dict["phase_acc"] = float(phase_acc)
        return loss, loss_dict


def _install_factory_patch() -> None:
    """Make ``--policy.type=diffusion`` yield our subclass without touching draccus.

    lerobot's CLI hardcodes the set of accepted ``--policy.type`` values via a
    draccus union, so adding a new name there would mean editing site-packages.
    Swapping the class behind the existing "diffusion" name is transparent to the
    rest of the train script — config, processors, checkpoint format are all
    inherited unchanged.
    """
    import lerobot.policies.factory as factory

    original = factory.get_policy_class

    def patched(name: str):
        if name == "diffusion":
            return DiffusionPolicyWithPhaseHead
        return original(name)

    factory.get_policy_class = patched


def main() -> None:
    _install_factory_patch()
    from lerobot.scripts.lerobot_train import main as lerobot_train_main

    lerobot_train_main()


if __name__ == "__main__":
    main()
