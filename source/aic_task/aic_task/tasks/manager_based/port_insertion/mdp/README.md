---
scope: directory index for the port_insertion task's MDP terms (commands, events, terminations)
audience: AI agents working in this repo
last_verified_commit: 8d9a44e
related:
  - ../../../../../docs/05_mdp_terms.md
  - ../../../../../docs/02_gym_registration.md
---

# `port_insertion/mdp/`

Task-specific MDP terms for `AIC-Port-Insertion-v0`. The full reference
for each term — signatures, inputs, outputs, internal state, edge cases —
lives in [05_mdp_terms.md](../../../../../docs/05_mdp_terms.md). This file
is just a directory index.

This folder is **blacklisted from task auto-discovery**: the entry
`".mdp"` in `_BLACKLIST_PKGS` at
[tasks/__init__.py](../../../__init__.py) tells the walker to skip every
dotted name containing `.mdp`. Nothing in this directory becomes a Gym
ID; everything here is imported on demand by
[`builders.py`](../builders.py) (and by the local
[`__init__.py`](__init__.py) re-export). See
[02_gym_registration.md](../../../../../docs/02_gym_registration.md) for
the walker contract.

## Files and owned terms

| File | Owned MDP terms (linked to the detailed entry) |
|---|---|
| [`__init__.py`](__init__.py) | Re-exports `isaaclab.envs.mdp.*` plus `InsertionGoalCommand[Cfg]` from `commands`. See [the re-export pattern entry](../../../../../docs/05_mdp_terms.md#re-export-pattern-in-mdp__init__py). |
| [`commands.py`](commands.py) | [`InsertionGoalCommand`](../../../../../docs/05_mdp_terms.md#insertiongoalcommand) + `InsertionGoalCommandCfg`. |
| [`events.py`](events.py) | [`reset_robot_to_default_joint_pose`](../../../../../docs/05_mdp_terms.md#reset_robot_to_default_joint_pose) *(available, not wired)*, [`randomize_dome_light`](../../../../../docs/05_mdp_terms.md#randomize_dome_light), [`randomize_board_and_parts`](../../../../../docs/05_mdp_terms.md#randomize_board_and_parts). |
| [`terminations.py`](terminations.py) | [`InsertionGoalReachedSuccess`](../../../../../docs/05_mdp_terms.md#insertiongoalreachedsuccess), [`InsertionGoalStationaryFailure`](../../../../../docs/05_mdp_terms.md#insertiongoalstationaryfailure). |
