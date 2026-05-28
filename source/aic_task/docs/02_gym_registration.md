---
scope: how Gym IDs in this package get registered when aic_task is imported; what the entry_point kwargs mean
audience: AI agents working in this repo
last_verified_commit: bb6a606
related:
  - 01_package_structure.md
  - 03_port_insertion_overview.md
---

# 02 · Gym Registration

The single Gym ID this package exposes is `AIC-Port-Insertion-v0`. This doc
explains exactly how `gym.make("AIC-Port-Insertion-v0")` succeeds — i.e. what
chain of imports has to fire, and what the registration kwargs mean to
downstream consumers (`parse_env_cfg`, training runners).

If you only want the rule of thumb: **anything that calls `gym.make` on an
AIC task must import `aic_task` first.** The repo scripts do this at the top
of the file, e.g. [scripts/random_agent.py:61](../../../scripts/random_agent.py#L61).

## The import chain (one line at a time)

```text
import aic_task
   │
   │  aic_task/__init__.py
   ▼
from .tasks import *
   │
   │  aic_task/tasks/__init__.py
   ▼
from isaaclab_tasks.utils import import_packages
import_packages(__name__, _BLACKLIST_PKGS)    # __name__ == "aic_task.tasks"
   │
   │  recursively imports every subpackage under aic_task.tasks
   │  except names containing any string in _BLACKLIST_PKGS
   ▼
aic_task.tasks.manager_based.port_insertion.__init__
   │
   │  contains the actual gym.register(...)
   ▼
Gym ID "AIC-Port-Insertion-v0" is now resolvable
```

Two side notes worth knowing:

- `aic_task/__init__.py` *also* tries `from .extension import *`, wrapped in
  a `try/except ModuleNotFoundError` that swallows missing `omni.*`. In
  headless / pure-IsaacLab runs the extension module fails to import (no
  `omni`), the exception is caught, and the chain proceeds normally. **It is
  expected behavior; not a bug.**
- `aic_task/tasks/__init__.py` also wraps the `import_packages` call in a bare
  `try/except` so running inside the Isaac Sim UI (where `isaaclab_tasks` may
  not be on the path yet) prints `running in Isaac Sim` instead of crashing.
  Outside Isaac Sim that branch never fires.

## `import_packages` — what it actually does

Source: `isaaclab_tasks/utils/importer.py` (in the parent IsaacLab tree at
`/home/etfrobot/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/utils/importer.py`,
which sits outside this repo's workspace).

```python
def import_packages(package_name: str, blacklist_pkgs: list[str] | None = None):
    package = importlib.import_module(package_name)
    for _ in _walk_packages(package.__path__, package.__name__ + ".", blacklist_pkgs):
        pass
```

Internally `_walk_packages` is a modified `pkgutil.walk_packages` that:

1. Iterates every module / subpackage under the given package path.
2. **Skips a candidate if any string in `blacklist_pkgs` appears anywhere in
   its dotted name.** Note: this is a substring match, not a path match —
   `".mdp"` is enough to skip every module whose name contains `.mdp`.
3. For each non-blacklisted subpackage, calls `__import__(info.name)` —
   this is the line that triggers the package's `__init__.py` and therefore
   any `gym.register(...)` calls it contains.
4. Recurses into the imported subpackage.

So the registration model is: **put the `gym.register` call inside the
task's `__init__.py`, and the walker will find it automatically.** You do
not edit any central list.

## The blacklist used here

[aic_task/tasks/__init__.py](../aic_task/tasks/__init__.py):

```python
_BLACKLIST_PKGS = ["utils", ".mdp"]
import_packages(__name__, _BLACKLIST_PKGS)
```

Why each entry:

| Entry | Why it's blacklisted |
|---|---|
| `"utils"` | Reserved name for any future `aic_task.tasks.utils` helpers — they should not be imported as if they were tasks. Today the package has none, so the entry is a guard rail. |
| `".mdp"` | Every task subpackage has an `mdp/` submodule with `commands.py`, `events.py`, `terminations.py`. Those are not standalone tasks — they're imported by `builders.py` on demand. Eagerly importing them at registration time is wasted work. |

When you add `aic_task.tasks.manager_based.<new_task>/mdp/`, the existing
blacklist already covers it.

## What `gym.register` writes for `AIC-Port-Insertion-v0`

Source: [aic_task/tasks/manager_based/port_insertion/__init__.py](../aic_task/tasks/manager_based/port_insertion/__init__.py).

```python
gym.register(
    id="AIC-Port-Insertion-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.port_insertion_env_cfg:PortInsertionEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPORunnerCfg",
    },
)
```

Field-by-field:

| Field | Meaning |
|---|---|
| `id` | The string passed to `gym.make`. Must follow Gym's `Foo-Bar-vN` style. |
| `entry_point` | The env class Gym instantiates. We always use IsaacLab's `ManagerBasedRLEnv`; the task-specific behavior lives in the `*EnvCfg` passed via kwargs. |
| `disable_env_checker` | Skips Gym's input/output sanity checks. Required for IsaacLab task envs — they don't follow the single-env Gym API. |
| `kwargs["env_cfg_entry_point"]` | A `"module.path:ClassName"` string that `isaaclab_tasks.utils.parse_env_cfg` resolves to the env cfg class. Consumers call `parse_env_cfg(id, num_envs=..., device=...)` to materialize a cfg instance from this string. |
| `kwargs["rsl_rl_cfg_entry_point"]` | Same convention, points at the training-side cfg used by `scripts/rsl_rl/*.py`. Sim-time code never touches this. |

`{__name__}` expands to `aic_task.tasks.manager_based.port_insertion`, so the
resolved env-cfg string is
`aic_task.tasks.manager_based.port_insertion.port_insertion_env_cfg:PortInsertionEnvCfg`.

## How a consumer actually launches the env

The canonical sequence used by every script in `scripts/`:

```python
import gymnasium as gym
import isaaclab_tasks                # populates IsaacLab core tasks
import aic_task.tasks                # populates this repo's tasks
from isaaclab_tasks.utils import parse_env_cfg

env_cfg = parse_env_cfg(             # resolves env_cfg_entry_point, instantiates it,
    "AIC-Port-Insertion-v0",         # applies device / num_envs overrides
    device="cuda:0",
    num_envs=1,
    use_fabric=True,
)
env = gym.make("AIC-Port-Insertion-v0", cfg=env_cfg)
```

`import aic_task` is enough — the `from .tasks import *` line inside it makes
the explicit `import aic_task.tasks` redundant but harmless.

## Practical implications for adding a task

- Drop a new package under `aic_task/tasks/manager_based/<new_name>/`.
- Give its `__init__.py` a `gym.register(...)` call with an `env_cfg_entry_point`
  pointing at your `*EnvCfg` class.
- Do **not** edit `aic_task/tasks/__init__.py` — the walker already finds
  your package.
- Keep `mdp/`, `agents/`, and `builders.py` in that subpackage; the `.mdp`
  blacklist will skip the term modules during walk and the env cfg can import
  them lazily.
- If you add a `utils/` submodule under your task, name it something other
  than `"utils"` — that substring is blacklisted globally. Common choice:
  put helpers directly in `builders.py` or a `_helpers.py`.

The walk-and-register pattern means **a typo in your task's `__init__.py`
silently produces no Gym ID** — the import error is caught nowhere obvious.
The first symptom is usually `gym.error.NameNotFound` from the script. If
that happens, `python -c "import aic_task.tasks.manager_based.<new_name>"`
will surface the real error.
