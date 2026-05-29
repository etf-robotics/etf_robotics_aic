# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Recorder terms for the port-insertion task.

Adds a pre-step recorder that captures the full observation buffer (every
configured obs group, not just the policy group) so that the privileged
``cheatcode`` group lands in the HDF5 dataset alongside ``policy``. The
stock :class:`~isaaclab.envs.mdp.recorders.recorders.PreStepFlatPolicyObservationsRecorder`
only writes ``obs_buf["policy"]``, silently dropping every other group.

The shipped ``ActionStateRecorderManagerCfg`` is subclassed to swap the
policy-only recorder for the grouped one. Use
:class:`GroupedActionStateRecorderManagerCfg` from ``scripts/record_demos.py``.

HDF5 layout produced by the grouped recorder (with ``concatenate_terms=False``
on each obs group, which is what the BC pipeline uses):

```
demo_X/
  obs/
    policy/<term>     # e.g. policy/joint_pos, policy/center_camera_rgb
    cheatcode/<term>  # e.g. cheatcode/seat_pos_b
```
"""

from __future__ import annotations

from isaaclab.envs.mdp.recorders.recorders_cfg import ActionStateRecorderManagerCfg
from isaaclab.managers import RecorderTerm
from isaaclab.managers.manager_term_cfg import RecorderTermCfg
from isaaclab.utils import configclass


class PreStepGroupedObservationsRecorder(RecorderTerm):
    """Records the full observation buffer keyed by obs group.

    Returns ``("obs", env.obs_buf)`` so :class:`EpisodeData.add` recurses into
    the group dict and stores each leaf at ``obs/<group>/<term>``. Compatible
    with both ``concatenate_terms=True`` (each group is a flat tensor) and
    ``concatenate_terms=False`` (each group is a ``{term: tensor}`` dict).
    """

    def record_pre_step(self):
        return "obs", self._env.obs_buf


@configclass
class PreStepGroupedObservationsRecorderCfg(RecorderTermCfg):
    """Configuration for :class:`PreStepGroupedObservationsRecorder`."""

    class_type: type[RecorderTerm] = PreStepGroupedObservationsRecorder


@configclass
class GroupedActionStateRecorderManagerCfg(ActionStateRecorderManagerCfg):
    """Drop-in replacement for ``ActionStateRecorderManagerCfg`` that writes
    every observation group instead of only the policy group.

    Disables the inherited ``record_pre_step_flat_policy_observations`` term
    (``None`` is skipped by ``RecorderManagerBase._prepare_terms``) and adds
    :class:`PreStepGroupedObservationsRecorderCfg` in its place.
    """

    record_pre_step_flat_policy_observations = None
    record_pre_step_grouped_observations = PreStepGroupedObservationsRecorderCfg()


__all__ = [
    "PreStepGroupedObservationsRecorder",
    "PreStepGroupedObservationsRecorderCfg",
    "GroupedActionStateRecorderManagerCfg",
]
