# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Open-loop, time-parameterized planner for AIC-Port-Insertion-v0.

Three phases per episode:

* ``APPROACH`` — translate from start to a randomized approach point
  (entrance lifted by ``standoff_m`` along the insertion axis, plus
  uniform lateral jitter). Orientation is held at the start quat.
* ``ALIGN`` — translate the small lateral correction to "above entrance"
  while rotating from the start quat to the entrance quat.
* ``INSERT`` — translate along the insertion axis from above-entrance to
  the seat pose. Orientation held at the entrance quat. Terminal: exits
  on env termination, not on time.

Each segment is a quintic in normalized time ``τ ∈ [0, 1]`` with
``s(τ) = 10τ³ − 15τ⁴ + 6τ⁵`` (rest-rest in position parameter and SLERP
parameter). Per-segment duration ``T_seg`` is sized per env from the
segment's pos/rot amplitudes and a per-phase ``speed_scale ∈ [0, 1]``
that caps action utilization at the velocity peak.

Cheatcode publishes goal poses in the **EEF frame**; the planner shifts
each segment endpoint to the **TCP frame** at plan time using the
constant ``tcp_in_eef`` offset. The env wrapper then takes TCP-frame
goals and emits delta actions — it does no frame work.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import IntEnum

import torch

from isaaclab.utils.math import (
    axis_angle_from_quat,
    combine_frame_transforms,
    quat_from_angle_axis,
    quat_inv,
    quat_mul,
    subtract_frame_transforms,
)


class Phase(IntEnum):
    APPROACH = 0
    ALIGN = 1
    INSERT = 2


NUM_PHASES = len(Phase)

# Rest-rest quintic peak rates (per unit amplitude per unit duration):
#   max |s'(τ)| = 15/8 / T   →  T_min = (15/8) * L / v_peak
#   max |s''(τ)| = (10/√3) / T²
# We size duration from velocity only; the speed_scale ≤ 1 leaves
# acceleration headroom implicitly because peak accel scales as 1/T².
_QUINTIC_PEAK_VEL = 15.0 / 8.0


@dataclass
class PlanBatch:
    """Per-env, per-segment plan. All poses in TCP root frame."""

    seg_start_pos: torch.Tensor   # (N, P, 3)
    seg_end_pos: torch.Tensor     # (N, P, 3)
    seg_start_quat: torch.Tensor  # (N, P, 4) wxyz
    seg_end_quat: torch.Tensor    # (N, P, 4) wxyz
    T_seg: torch.Tensor           # (N, P) seconds, > 0
    phase: torch.Tensor           # (N, P) int8

    @classmethod
    def empty(cls, num_envs: int, num_phases: int, device) -> "PlanBatch":
        return cls(
            seg_start_pos=torch.zeros(num_envs, num_phases, 3, device=device),
            seg_end_pos=torch.zeros(num_envs, num_phases, 3, device=device),
            seg_start_quat=torch.zeros(num_envs, num_phases, 4, device=device),
            seg_end_quat=torch.zeros(num_envs, num_phases, 4, device=device),
            T_seg=torch.ones(num_envs, num_phases, device=device),
            phase=torch.zeros(num_envs, num_phases, dtype=torch.int8, device=device),
        )

    def scatter_(self, env_ids: torch.Tensor, src: "PlanBatch") -> None:
        """Overwrite rows ``env_ids`` with ``src``. ``src`` rows are aligned to env_ids."""
        self.seg_start_pos[env_ids] = src.seg_start_pos
        self.seg_end_pos[env_ids] = src.seg_end_pos
        self.seg_start_quat[env_ids] = src.seg_start_quat
        self.seg_end_quat[env_ids] = src.seg_end_quat
        self.T_seg[env_ids] = src.T_seg
        self.phase[env_ids] = src.phase


def quat_slerp_batched(q0: torch.Tensor, q1: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    """Batched SLERP via axis-angle. q0,q1: (..., 4) wxyz; s: (...,) in [0, 1].

    Returns the interpolated quaternion (..., 4). Short-path is handled
    inside ``axis_angle_from_quat`` (it flips sign of the delta quat
    when w<0). Degenerate case (q0 ≈ q1) yields a zero axis-angle, which
    ``quat_from_angle_axis`` maps to identity via the safe ``normalize``.
    """
    delta = quat_mul(quat_inv(q0), q1)
    axis_angle = axis_angle_from_quat(delta)            # (..., 3)
    scaled = axis_angle * s.unsqueeze(-1)               # (..., 3)
    angle = scaled.norm(dim=-1)                         # (...,)
    delta_partial = quat_from_angle_axis(angle, scaled) # normalize() handles zero axis
    return quat_mul(q0, delta_partial)


def _quat_angle(q0: torch.Tensor, q1: torch.Tensor) -> torch.Tensor:
    """Short-path rotation magnitude between two unit quats. (..., 4) → (...,)."""
    dot = (q0 * q1).sum(dim=-1).abs().clamp(max=1.0)
    return 2.0 * dot.acos()


def _perp_basis(axis: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Two orthonormal vectors perpendicular to each row of ``axis``.

    ``axis``: (M, 3) unit. Picks ``e_x`` as reference unless the axis is
    nearly x-aligned, then ``e_y``. Gram-Schmidt to get ``e1``, then
    ``e2 = axis × e1``.
    """
    e_x = torch.tensor([1.0, 0.0, 0.0], device=axis.device).expand_as(axis)
    e_y = torch.tensor([0.0, 1.0, 0.0], device=axis.device).expand_as(axis)
    use_x = (axis * e_x).sum(dim=-1).abs() < 0.9
    ref = torch.where(use_x.unsqueeze(-1), e_x, e_y)
    e1 = ref - (ref * axis).sum(dim=-1, keepdim=True) * axis
    e1 = e1 / e1.norm(dim=-1, keepdim=True).clamp_min(1e-9)
    e2 = torch.cross(axis, e1, dim=-1)
    return e1, e2


class PortInsertionPlanner:
    """Generates open-loop 3-phase TCP plans for batched envs."""

    NUM_PHASES = NUM_PHASES

    DEFAULT_SPEED_SCALE: dict[Phase, float] = {
        Phase.APPROACH: 0.6,
        Phase.ALIGN: 0.8,
        Phase.INSERT: 0.1,
    }

    def __init__(
        self,
        env,
        *,
        standoff_m: float = 0.05,
        approach_jitter_m: float = 0.02,
        speed_scale: dict[Phase, float] | None = None,
        min_T_seg: float | None = None,
        rng: torch.Generator | None = None,
    ):
        self.env = env
        self.device = env.device
        self.policy_dt = env.policy_dt

        scale = env.action_scale  # (6,)
        self.v_max = float(scale[0]) / self.policy_dt  # peak feasible m/s per axis
        self.w_max = float(scale[3]) / self.policy_dt  # peak feasible rad/s per axis

        self.standoff_m = standoff_m
        self.approach_jitter_m = approach_jitter_m
        self.speed_scale = dict(speed_scale or self.DEFAULT_SPEED_SCALE)
        self.min_T_seg = self.policy_dt if min_T_seg is None else min_T_seg
        self.rng = rng

        # Per-phase speed-scale vector for vectorized duration math.
        self._speed_vec = torch.tensor(
            [self.speed_scale[Phase(p)] for p in range(NUM_PHASES)],
            device=self.device,
            dtype=torch.float32,
        )
        self._phase_row = torch.tensor(
            [int(Phase.APPROACH), int(Phase.ALIGN), int(Phase.INSERT)],
            device=self.device,
            dtype=torch.int8,
        )

    @property
    def num_phases(self) -> int:
        return self.NUM_PHASES

    def plan(self, env_ids: torch.Tensor, obs: dict) -> PlanBatch:
        """Compute fresh plans for ``env_ids``, reading only those rows of ``obs``.

        Args:
            env_ids: (M,) int index of envs to plan for.
            obs: full obs dict from ``env.step``/``env.reset``. The
                planner indexes into it by ``env_ids``.

        Returns:
            ``PlanBatch`` with rows of shape ``(M, P, ...)``, ordered to
            match ``env_ids``. Caller scatters into the global plan.
        """
        M = int(env_ids.numel())
        P = self.NUM_PHASES

        policy = obs["policy"]
        cheat = obs["cheatcode"]
        eef_pos_b = policy["eef_pos_b"][env_ids]
        eef_quat_b = policy["eef_quat_b"][env_ids]
        tcp_pos_b = policy["tcp_pos_b"][env_ids]
        tcp_quat_b = policy["tcp_quat_b"][env_ids]
        entrance_pos = cheat["entrance_pos_b"][env_ids]
        entrance_quat = cheat["entrance_quat_b"][env_ids]
        seat_pos = cheat["seat_pos_b"][env_ids]

        # Insertion axis points from entrance into the port (entrance→seat).
        seat_minus_entrance = seat_pos - entrance_pos
        axis = seat_minus_entrance / seat_minus_entrance.norm(dim=-1, keepdim=True).clamp_min(1e-9)

        # "Above entrance" = entrance lifted by standoff along −axis.
        above_entrance_pos = entrance_pos - self.standoff_m * axis

        # Approach point = above_entrance + uniform-in-disk lateral jitter ⟂ axis.
        approach_pt_pos = above_entrance_pos + self._sample_perp_jitter(axis, M)

        # EEF-frame segment endpoints, stacked into (M, P, ...).
        # APPROACH:  (eef_pose)              → (approach_pt, start_quat)
        # ALIGN:     (approach_pt, start_q)  → (above_entrance, entrance_q)
        # INSERT:    (above_entrance, ent_q) → (seat,           entrance_q)
        eef_start_pos = torch.stack(
            [eef_pos_b, approach_pt_pos, above_entrance_pos], dim=1
        )
        eef_end_pos = torch.stack(
            [approach_pt_pos, above_entrance_pos, seat_pos], dim=1
        )
        eef_start_quat = torch.stack(
            [eef_quat_b, eef_quat_b, entrance_quat], dim=1
        )
        eef_end_quat = torch.stack(
            [eef_quat_b, entrance_quat, entrance_quat], dim=1
        )

        # EEF→TCP shift: constant offset per env, applied to every endpoint.
        tcp_in_eef_pos, tcp_in_eef_quat = subtract_frame_transforms(
            eef_pos_b, eef_quat_b, tcp_pos_b, tcp_quat_b
        )
        off_pos = tcp_in_eef_pos.unsqueeze(1).expand(-1, P, -1).reshape(-1, 3)
        off_quat = tcp_in_eef_quat.unsqueeze(1).expand(-1, P, -1).reshape(-1, 4)

        seg_start_pos_tcp, seg_start_quat_tcp = combine_frame_transforms(
            eef_start_pos.reshape(-1, 3), eef_start_quat.reshape(-1, 4),
            off_pos, off_quat,
        )
        seg_end_pos_tcp, seg_end_quat_tcp = combine_frame_transforms(
            eef_end_pos.reshape(-1, 3), eef_end_quat.reshape(-1, 4),
            off_pos, off_quat,
        )
        seg_start_pos_tcp = seg_start_pos_tcp.reshape(M, P, 3)
        seg_start_quat_tcp = seg_start_quat_tcp.reshape(M, P, 4)
        seg_end_pos_tcp = seg_end_pos_tcp.reshape(M, P, 3)
        seg_end_quat_tcp = seg_end_quat_tcp.reshape(M, P, 4)

        # Per-segment duration: T = (15/8) · L / (speed_scale · v_max).
        # Take max(T_pos, T_rot) so the slower-DOF governs the segment.
        L_pos = (seg_end_pos_tcp - seg_start_pos_tcp).norm(dim=-1)              # (M, P)
        L_rot = _quat_angle(
            seg_start_quat_tcp.reshape(-1, 4), seg_end_quat_tcp.reshape(-1, 4),
        ).reshape(M, P)
        speed = self._speed_vec.unsqueeze(0).expand(M, -1)                      # (M, P)
        T_pos = _QUINTIC_PEAK_VEL * L_pos / (speed * self.v_max).clamp_min(1e-9)
        T_rot = _QUINTIC_PEAK_VEL * L_rot / (speed * self.w_max).clamp_min(1e-9)
        T_seg = torch.maximum(T_pos, T_rot).clamp_min(self.min_T_seg)

        phase = self._phase_row.unsqueeze(0).expand(M, -1).contiguous()

        return PlanBatch(
            seg_start_pos=seg_start_pos_tcp,
            seg_end_pos=seg_end_pos_tcp,
            seg_start_quat=seg_start_quat_tcp,
            seg_end_quat=seg_end_quat_tcp,
            T_seg=T_seg,
            phase=phase,
        )

    def _sample_perp_jitter(self, axis: torch.Tensor, count: int) -> torch.Tensor:
        """Uniform-in-disk lateral jitter ⟂ axis, radius ≤ ``approach_jitter_m``.

        axis: (M, 3) unit, M==count. Returns (M, 3), each row ⟂ to its axis.
        """
        if self.approach_jitter_m == 0.0:
            return torch.zeros_like(axis)
        e1, e2 = _perp_basis(axis)
        rand_kwargs = {"device": axis.device}
        if self.rng is not None:
            rand_kwargs["generator"] = self.rng
        # r = R·√U for uniform area density; θ = 2π·U.
        r = self.approach_jitter_m * torch.rand(count, **rand_kwargs).sqrt()
        theta = 2.0 * math.pi * torch.rand(count, **rand_kwargs)
        return r.unsqueeze(-1) * (theta.cos().unsqueeze(-1) * e1 + theta.sin().unsqueeze(-1) * e2)


class PortInsertionExecutor:
    """Drives a batched ``PlanBatch`` per step. Owns per-env (seg_idx, t_in_seg)."""

    def __init__(self, env, planner: PortInsertionPlanner):
        self.env = env
        self.planner = planner
        N = env.num_envs
        P = planner.num_phases
        self.policy_dt = env.policy_dt
        self.plans = PlanBatch.empty(N, P, device=env.device)
        self.seg_idx = torch.zeros(N, dtype=torch.long, device=env.device)
        self.t_in_seg = torch.zeros(N, dtype=torch.float32, device=env.device)
        self._arange_N = torch.arange(N, device=env.device)
        self._num_phases = P

    def reset_plan(self, env_ids: torch.Tensor, obs: dict) -> None:
        """Generate fresh plans for ``env_ids`` and reset their segment cursor."""
        new = self.planner.plan(env_ids, obs)
        self.plans.scatter_(env_ids, new)
        self.seg_idx[env_ids] = 0
        self.t_in_seg[env_ids] = 0.0

    def step(self, obs: dict) -> tuple[torch.Tensor, dict]:
        """Emit one action across all envs. Returns ``(action, info{phase: (N,) int8})``.

        ``phase`` is the phase the action was generated under — i.e., the
        phase associated with the current ``obs``.
        """
        tcp_goal_pos, tcp_goal_quat, phase = self._evaluate_current_segment()
        action = self.env.tcp_goal_to_action(tcp_goal_pos, tcp_goal_quat, obs)
        self._advance_time()
        return action, {"phase": phase}

    def _evaluate_current_segment(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        i = self.seg_idx
        T = self.plans.T_seg[self._arange_N, i]
        tau = (self.t_in_seg / T.clamp_min(1e-9)).clamp(0.0, 1.0)
        s = tau ** 3 * (10.0 - 15.0 * tau + 6.0 * tau ** 2)  # 10τ³ − 15τ⁴ + 6τ⁵
        sp = self.plans.seg_start_pos[self._arange_N, i]
        ep = self.plans.seg_end_pos[self._arange_N, i]
        pos_target = sp + s.unsqueeze(-1) * (ep - sp)
        sq = self.plans.seg_start_quat[self._arange_N, i]
        eq = self.plans.seg_end_quat[self._arange_N, i]
        quat_target = quat_slerp_batched(sq, eq, s)
        phase_cur = self.plans.phase[self._arange_N, i]
        return pos_target, quat_target, phase_cur

    def _advance_time(self) -> None:
        i = self.seg_idx
        T = self.plans.T_seg[self._arange_N, i]
        self.t_in_seg = self.t_in_seg + self.policy_dt
        can_advance = (self.t_in_seg >= T) & (i < self._num_phases - 1)
        self.seg_idx = torch.where(can_advance, i + 1, i)
        self.t_in_seg = torch.where(can_advance, torch.zeros_like(self.t_in_seg), self.t_in_seg)
        # Terminal segment is sticky: clamp t_in_seg at its T_seg so τ saturates at 1.
        T_after = self.plans.T_seg[self._arange_N, self.seg_idx]
        self.t_in_seg = torch.minimum(self.t_in_seg, T_after)


__all__ = [
    "Phase",
    "PlanBatch",
    "PortInsertionPlanner",
    "PortInsertionExecutor",
    "quat_slerp_batched",
]
