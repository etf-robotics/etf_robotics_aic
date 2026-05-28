---
scope: tasks/ auto-discovery entry point — what gets walked, what gets skipped, where the concrete tasks live
audience: AI agents working in this repo
last_verified_commit: 8d9a44e
related:
  - ../../docs/02_gym_registration.md
  - ../../docs/01_package_structure.md
---

# `tasks/`

This directory is the auto-registration entry point for every Gym ID this
package exposes. Importing `aic_task` triggers
[`aic_task/__init__.py`](../__init__.py), which runs
`from .tasks import *`, which runs the auto-discovery walk defined right
here in [`__init__.py`](__init__.py).

Anything below this folder whose dotted name passes the discovery filter
will be imported eagerly, which is how `gym.register(...)` calls in
per-task `__init__.py` files take effect. The walker is described in
detail in [02_gym_registration.md](../../docs/02_gym_registration.md);
this README is a shorter directory index.

## Discovery rule

```python
# tasks/__init__.py
_BLACKLIST_PKGS = ["utils", ".mdp"]
import_packages(__name__, _BLACKLIST_PKGS)
```

Each entry is a **substring match** against the full dotted name of every
candidate package the walker visits — not a path prefix. The two entries
exist for these reasons:

| Entry | Why it's blacklisted |
|---|---|
| `"utils"` | Reserved name for any future `aic_task.tasks.utils` helpers. They should not be imported as if they were tasks. Today no such package exists; the entry is a guard rail for the future. |
| `".mdp"` | Every task subpackage owns an `mdp/` submodule (`commands.py`, `events.py`, `terminations.py`). Those are *not* standalone tasks — they're imported by `builders.py` on demand. Eagerly importing them at registration time would be wasted work and would expose `omni.usd` imports (in `events.py` / `commands.py`) before they're needed. |

Substring matching means the blacklist applies to any future task too:
when you add `aic_task.tasks.manager_based.<new_name>/mdp/`, it is already
covered.

## Task subpackages

| Gym ID | Subpackage | Entry point |
|---|---|---|
| `AIC-Port-Insertion-v0` | [manager_based/port_insertion/](manager_based/port_insertion/) | `port_insertion_env_cfg:PortInsertionEnvCfg` |

The full behavioral spec lives in
[03_port_insertion_overview.md](../../docs/03_port_insertion_overview.md);
the directory index for the task lives in
[manager_based/port_insertion/README.md](manager_based/port_insertion/README.md).

## Adding a new task subpackage

The walker finds new tasks automatically. Drop a package at
`aic_task/tasks/manager_based/<new_task>/` with a `gym.register(...)` call
in its `__init__.py`, and `import aic_task` will register it.

The detailed recipe — env cfg shape, builders, MDP terms — will live in
`08_adding_a_new_task.md` *(planned)*. Until then, the assembly-pattern
doc ([04_assembly_pattern.md](../../docs/04_assembly_pattern.md)) and the
port-insertion implementation are the working reference.
