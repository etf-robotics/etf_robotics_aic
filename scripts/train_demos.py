# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Train ACT with auxiliary phase supervision on the AIC port-insertion dataset.

This wrapper around ``lerobot.scripts.lerobot_train``:

1. Subclasses ``ACTPolicy`` to add a 3-class linear head on top of the
   transformer encoder's pooled output and an auxiliary cross-entropy
   loss against the dataset's ``annotation.phase`` column (one-hot
   {APPROACH, ALIGN, INSERT}). The aux loss fires **only when
   ``self.training`` is True**, so:

   - During ``lerobot-train`` forward: action L1 loss + KLD + phase CE.
   - During ``select_action`` / ``predict_action_chunk`` (eval, deploy):
     the head is skipped entirely; behavior matches vanilla ACT.

2. Monkey-patches ``lerobot.policies.factory.get_policy_class("act")``
   to return our subclass, so all the standard ``--policy.type=act``
   plumbing in lerobot's CLI / config / checkpointing still works.

3. Forwards remaining CLI args to ``lerobot.scripts.lerobot_train.main``.

Run inside the container (the same way you'd run the stock trainer,
just point at this script):

    docker exec -w /workspace/isaaclab isaac-lab-base \\
      ./isaaclab.sh -p etf_robotics_aic/scripts/train_demos.py \\
      --policy.type=act --policy.device=cuda \\
      --dataset.repo_id=... --dataset.root=... --dataset.episodes='[...]' \\
      --output_dir=... --batch_size=8 --steps=100000 --num_workers=0 \\
      --save_freq=5000 --eval_freq=0 --wandb.enable=false
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from lerobot.policies.act.modeling_act import ACTPolicy

PHASE_KEY = "annotation.phase"
PHASE_N_CLASSES = 3
PHASE_LOSS_WEIGHT = 0.1  # tune if phase CE dominates / disappears


class ACTPolicyWithPhaseHead(ACTPolicy):
    """ACT + auxiliary subtask-phase classifier.

    The head reads the transformer encoder's last hidden states via a
    forward hook (no surgery on the ACT graph), mean-pools over tokens,
    and predicts the 3-class phase. The head is trained but never read
    at inference, so deployment is identical to vanilla ACT — you can
    even load this checkpoint into vanilla ``ACTPolicy`` with
    ``strict=False`` and it works.
    """

    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        self.phase_head = nn.Linear(config.dim_model, PHASE_N_CLASSES)
        self._enc_buf: torch.Tensor | None = None
        # Hook fires on every ACT forward (train + eval). We only consume
        # the buffer in train mode, and clear it after each use so it
        # doesn't pin tensors during inference.
        self.model.encoder.register_forward_hook(self._cap_encoder_output)

    def _cap_encoder_output(self, _module, _inputs, output):
        self._enc_buf = output

    def forward(self, batch):
        loss, loss_dict = super().forward(batch)

        if not (self.training and PHASE_KEY in batch and self._enc_buf is not None):
            self._enc_buf = None
            return loss, loss_dict

        # encoder output: (S, B, D) — token-major from the underlying nn.TransformerEncoder.
        enc = self._enc_buf
        self._enc_buf = None
        pooled = enc.mean(dim=0)  # (B, D)
        phase_logits = self.phase_head(pooled)  # (B, 3)

        target = batch[PHASE_KEY].to(phase_logits.device)
        # The writer stores phase as a per-frame float32 one-hot of shape (3,).
        # LeRobot's loader gives us (B, 3) for non-action features. Be defensive
        # in case any future delta-timestamp setup yields (B, T, 3).
        if target.dim() == 3:
            target = target[:, 0, :]
        target_idx = target.argmax(dim=-1).long()  # (B,)
        phase_ce = F.cross_entropy(phase_logits, target_idx)

        loss = loss + PHASE_LOSS_WEIGHT * phase_ce
        loss_dict["phase_ce_loss"] = float(phase_ce.detach())
        with torch.no_grad():
            phase_acc = (phase_logits.argmax(dim=-1) == target_idx).float().mean()
            loss_dict["phase_acc"] = float(phase_acc)
        return loss, loss_dict


def _install_factory_patch() -> None:
    """Make ``--policy.type=act`` yield our subclass without touching draccus.

    lerobot's CLI hardcodes the set of accepted ``--policy.type`` values
    via a draccus union, so adding a new name there would mean editing
    site-packages. Swapping the class behind the existing "act" name is
    transparent to the rest of the train script — config, processors,
    checkpoint format are all inherited unchanged.
    """
    import lerobot.policies.factory as factory

    original = factory.get_policy_class

    def patched(name: str):
        if name == "act":
            return ACTPolicyWithPhaseHead
        return original(name)

    factory.get_policy_class = patched


def main() -> None:
    _install_factory_patch()
    from lerobot.scripts.lerobot_train import main as lerobot_train_main
    lerobot_train_main()


if __name__ == "__main__":
    main()
